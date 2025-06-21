"""
RAG Q&A Tool for Patient Medical Records
Uses OpenAI embeddings and FAISS vector database for similarity search.
"""

import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain.tools import BaseTool
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from mock_patient_data import MOCK_PATIENT_RECORDS, MEDICATION_DATABASE, SYMPTOM_DATABASE

# Configure logging to suppress OpenAI HTTP requests
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

class RAGQueryInput(BaseModel):
    query: str = Field(description="The medical question or query to search for")
    patient_name: Optional[str] = Field(default=None, description="Optional patient name to filter results")

class RAGTool:
    """RAG Q&A Tool for Patient Medical Records using OpenAI embeddings and FAISS vector database."""
    
    def __init__(self):
        self.name = "rag_qa_tool"
        self.description = "Search patient medical records, medications, and symptoms to answer medical questions"
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = []
        self.embeddings = []
        self.index = None
        self._load_documents()
        self._create_embeddings()
        self._build_faiss_index()
    
    def _load_documents(self):
        """Load all documents from patient records, medications, and symptoms."""
        self.documents = []
        
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
            
            self.documents.append({
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
            
            self.documents.append({
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
            
            self.documents.append({
                "content": symptom_text,
                "metadata": {
                    "type": "symptom",
                    "name": symptom_name,
                    "common_causes": symptom_info['common_causes'],
                    "recommended_actions": symptom_info['recommended_actions']
                }
            })
    
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
    
    def _create_embeddings(self):
        """Create embeddings for all documents."""
        print("Creating embeddings for patient records, medications, and symptoms...")
        self.embeddings = []
        
        for doc in self.documents:
            embedding = self._get_embedding(doc["content"])
            if embedding:
                self.embeddings.append(embedding)
            else:
                # Use zero vector as fallback
                self.embeddings.append([0.0] * 1536)  # OpenAI embedding dimension
        
        self.embeddings = np.array(self.embeddings, dtype=np.float32)
        print(f"Created embeddings for {len(self.embeddings)} patient records, medications, and symptoms")
    
    def _build_faiss_index(self):
        """Build FAISS index for similarity search."""
        if len(self.embeddings) == 0:
            print("No embeddings available to build index")
            return
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
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
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Filter by patient name if specified
                if patient_name and doc["metadata"]["type"] == "patient_record":
                    if patient_name.lower() not in doc["metadata"]["name"].lower():
                        continue
                
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
        
        return results
    
    def _run(self, query: str, patient_name: Optional[str] = None) -> str:
        """Run the RAG tool to answer medical questions."""
        try:
            # Search for relevant documents
            results = self._search(query, patient_name, top_k=5)
            
            if not results:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or ask a different question."
            
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
            
            return response.content
            
        except Exception as e:
            return f"Error processing your question: {str(e)}"
    
    async def _arun(self, query: str, patient_name: Optional[str] = None) -> str:
        """Async version of the run method."""
        return self._run(query, patient_name)

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