"""
RAG Q&A Tool for Patient Medical Records
Uses OpenAI embeddings and FAISS vector database for similarity search.
"""

import os
import json
import numpy as np
import faiss
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain.tools import BaseTool
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from colorama import init, Fore, Style
from galileo import GalileoLogger
from mock_patient_data import MOCK_PATIENT_RECORDS, MEDICATION_DATABASE, SYMPTOM_DATABASE

# Initialize colorama for cross-platform colored output
init()

# Configure logging to suppress OpenAI HTTP requests
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Initialize Galileo logger once at module level
galileo_logger = None
galileo_project = None
galileo_log_stream = None

def initialize_rag_galileo(project_name=None, logstream_name=None):
    """Initialize Galileo logger for RAG tool with optional project name and logstream override."""
    global galileo_logger, galileo_project, galileo_log_stream
    
    # Initialize Galileo logger if configuration is available
    api_key = os.getenv("GALILEO_API_KEY")
    project = project_name or os.getenv("GALILEO_PROJECT")
    log_stream = logstream_name or os.getenv("GALILEO_LOG_STREAM")

    print(f"RAG Tool - Galileo Configuration:")
    print(f"  API Key: {'Set' if api_key else 'Not set'}")
    print(f"  Project: {project}")
    print(f"  Log Stream: {log_stream}")

    if all([api_key, project, log_stream]):
        galileo_project = project
        galileo_log_stream = log_stream
        galileo_logger = GalileoLogger(project=project, log_stream=log_stream)
        print(Fore.GREEN + "RAG Tool - Galileo logger initialized successfully." + Style.RESET_ALL)
    else:
        print("Warning: RAG Tool - Missing Galileo configuration. Logging will be disabled.")
 
# Global in-memory cache for embeddings and documents
_EMBEDDINGS_CACHE = {
    "documents": None,
    "embeddings": None,
    "index": None,
    "initialized": False
}

def _initialize_embeddings_cache():
    """Initialize the global embeddings cache. Called only once upon server startup."""
    if _EMBEDDINGS_CACHE["initialized"]:
        print("Embeddings cache already initialized, skipping...")
        return
    
    print("Initializing embeddings cache...")
    
    # Load documents
    documents = []
    
    # Load patient records
    for patient in MOCK_PATIENT_RECORDS:
        # Create a comprehensive text representation of the patient
        lab_results_text = ""
        for key, value in patient['lab_results'].items():
            lab_results_text += f"  {key}: {value}\n"
        
        patient_text = f"""
        Patient: {patient['name']}
        Age: {patient['age']} years old
        Gender: {patient['gender']}
        
        Medical History:
        {', '.join(patient['medical_history'])}
        
        Current Medications:
        {', '.join(patient['current_medications'])}
        
        Allergies:
        {', '.join(patient['allergies'])}
        
        Conditions:
        {', '.join(patient['conditions'])}
        
        Lab Results:
        {lab_results_text}
        
        Last Visit: {patient['last_visit']}
        Next Appointment: {patient['next_appointment']}
        """
        
        documents.append({
            "content": patient_text,
            "metadata": {
                "type": "patient_record",
                "patient_id": patient['patient_id'],
                "name": patient['name'],
                "age": patient['age'],
                "gender": patient['gender'],
                "conditions": patient['conditions'],
                "allergies": patient['allergies'],
                "medications": patient['current_medications']
            }
        })
    
    # Load medication information
    for med_name, med_info in MEDICATION_DATABASE.items():
        med_text = f"""
        Medication: {med_name}
        Description: {med_info['description']}
        
        Contraindications:
        {', '.join(med_info['contraindications'])}
        
        Drug Interactions:
        {', '.join(med_info['interactions'])}
        """
        
        documents.append({
            "content": med_text,
            "metadata": {
                "type": "medication",
                "name": med_name,
                "description": med_info['description'],
                "contraindications": med_info['contraindications'],
                "interactions": med_info['interactions']
            }
        })
    
    # Load symptom information
    for symptom_name, symptom_info in SYMPTOM_DATABASE.items():
        symptom_text = f"""
        Symptom: {symptom_name}
        
        Common Causes:
        {', '.join(symptom_info['common_causes'])}
        
        Recommended Actions:
        {', '.join(symptom_info['recommended_actions'])}
        """
        
        documents.append({
            "content": symptom_text,
            "metadata": {
                "type": "symptom",
                "name": symptom_name,
                "common_causes": symptom_info['common_causes'],
                "recommended_actions": symptom_info['recommended_actions']
            }
        })
    
    _EMBEDDINGS_CACHE["documents"] = documents
    
    # Create embeddings
    print("Creating embeddings for patient records, medications, and symptoms...")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    
    for doc in documents:
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=doc["content"]
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 1536)  # OpenAI embedding dimension
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    _EMBEDDINGS_CACHE["embeddings"] = embeddings_array
    print(f"Created embeddings for {len(embeddings_array)} patient records, medications, and symptoms")
    
    # Build FAISS index
    if len(embeddings_array) > 0:
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        _EMBEDDINGS_CACHE["index"] = index
        print(f"Built FAISS index with {index.ntotal} vectors")
    else:
        print("No embeddings available to build index")
    
    _EMBEDDINGS_CACHE["initialized"] = True
    print("Embeddings cache initialization complete!")

class RAGQueryInput(BaseModel):
    query: str = Field(description="The medical question or query to search for")
    patient_name: Optional[str] = Field(default=None, description="Optional patient name to filter results")

class RAGTool:
    """RAG Q&A Tool for Patient Medical Records using OpenAI embeddings and FAISS vector database."""
    
    def __init__(self):
        self.name = "rag_qa_tool"
        self.description = "Search patient medical records, medications, and symptoms to answer medical questions"
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize the cache if not already done
        if not _EMBEDDINGS_CACHE["initialized"]:
            _initialize_embeddings_cache()
        
        # Use cached data
        self.documents = _EMBEDDINGS_CACHE["documents"]
        self.embeddings = _EMBEDDINGS_CACHE["embeddings"]
        self.index = _EMBEDDINGS_CACHE["index"]
        
        print(f"RAGTool initialized using cached embeddings ({len(self.documents)} documents)")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _search(self, query: str, patient_name: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using FAISS."""
        if not self.index:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        # Normalize query embedding
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Print search results in red color
        print(Fore.RED + f"🔍 Vector Search Results:" + Style.RESET_ALL)
        print(Fore.RED + f"   Query: {query[:50]}{'...' if len(query) > 50 else ''}" + Style.RESET_ALL)
        print(Fore.RED + f"   Patient Filter: {patient_name or 'None'}" + Style.RESET_ALL)
        print(Fore.RED + f"   Documents Retrieved: {len(indices[0])}" + Style.RESET_ALL)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                print(Fore.YELLOW + f"   Document {i}: Type = {doc['metadata']['type']}" + Style.RESET_ALL)
                
                # Filter by patient name if specified
                if patient_name and doc["metadata"]["type"] == "patient_record":
                    if patient_name.lower() not in doc["metadata"]["name"].lower():
                        continue
                
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
        
        # Print final results count after filtering
        print(Fore.RED + f"   Final Results After Filtering: {len(results)}" + Style.RESET_ALL)
        print(Fore.RED + f"   Top Score: {scores[0][0]:.4f}" + Style.RESET_ALL)
        
        return results
    
    def _run(self, query: str, patient_name: Optional[str] = None, galileo_logger=None) -> Dict[str, Any]:
        """Run the RAG tool to answer medical questions."""
        start_time = time.time()
        
        try:
            # Search for relevant documents
            search_start_time = time.time()
            results = self._search(query, patient_name, top_k=5)
            search_end_time = time.time()
            
            # Log the search operation to Galileo
            if galileo_logger:
                galileo_logger.add_retriever_span(
                    input=f"Query: {query}\nPatient: {patient_name or 'None'}",
                    output=results,
                    name="RAG Document Search",
                    duration_ns=int((search_end_time - search_start_time) * 1000000),
                    metadata={
                        "source": "rag_tool.py",
                        "type": "rag_search",
                        "query": query,
                        "patient_name": patient_name or "None",
                        "results_count": str(len(results)),
                        "top_k": str(5)
                    }
                )
                print(Fore.GREEN + f"✅ RAG retriever span logged successfully to project '{galileo_project}', logstream '{galileo_log_stream}'" + Style.RESET_ALL)
            
            if not results:
                return {
                    "response": "I couldn't find relevant information to answer your question. Please try rephrasing or ask a different question.",
                    "documents": []
                }
            
            # Format the context from search results
            context_parts = []
            for i, result in enumerate(results[:3]):  # Use top 3 results
                context_parts.append(f"Document {i+1}:\n{result['content']}\n")
            
            context = "\n".join(context_parts)
            
            # Use OpenAI to generate answer
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical assistant answering questions based on patient records, medication information, and symptom data.
                
                Use the provided context to answer the user's question accurately and professionally.
                Always recommend consulting a healthcare provider for serious medical decisions.
                Be clear about what information you have and what you don't know.
                If the context doesn't contain enough information, say so."""),
                ("human", f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a helpful answer based on the context.")
            ])
            
            chain = prompt | llm
            
            response = chain.invoke({})
            
            return {
                "response": response.content,
                "documents": results
            }
            
        except Exception as e:
            return {
                "response": f"Error processing your question: {str(e)}",
                "documents": []
            }
    
    async def _arun(self, query: str, patient_name: Optional[str] = None, galileo_logger=None) -> Dict[str, Any]:
        """Async version of the run method."""
        return self._run(query, patient_name, galileo_logger)

# Create a retriever class for LangChain compatibility
class PatientRecordRetriever(BaseRetriever):
    def __init__(self, rag_tool: RAGTool):
        self.rag_tool = rag_tool
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        results = self.rag_tool._search(query, top_k=5)
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query) 