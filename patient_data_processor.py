"""
Patient Data Processor - Similar to pdf_reader.py but for patient medical data.
Uses OpenAI embeddings and in-memory similarity search instead of Pinecone.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from mock_patient_data import MOCK_PATIENT_RECORDS, MEDICATION_DATABASE, SYMPTOM_DATABASE
from colorama import Fore, Style

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PatientDataProcessor:
    def __init__(self):
        self.patient_documents = []
        self.medication_documents = []
        self.symptom_documents = []
        self.all_embeddings = []
        self.all_documents = []
        
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def create_patient_documents(self) -> List[Dict[str, Any]]:
        documents = []
        
        for patient in MOCK_PATIENT_RECORDS:
            # Create a comprehensive text representation of the patient
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
            {json.dumps(patient['lab_results'], indent=2)}
            
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
        
        return documents
    
    def create_medication_documents(self) -> List[Dict[str, Any]]:
        documents = []
        
        for med_name, med_info in MEDICATION_DATABASE.items():
            # Create comprehensive text representation of medication
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
        
        return documents
    
    def create_symptom_documents(self) -> List[Dict[str, Any]]:
        documents = []
        
        for symptom_name, symptom_info in SYMPTOM_DATABASE.items():
            # Create comprehensive text representation of symptom
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
        
        return documents
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def similarity_search(self, query: str, document_type: str = None, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if not self.all_embeddings or not self.all_documents:
            print("No documents loaded. Please call load_all_documents() first.")
            return []
        
        # Verify synchronization
        if len(self.all_documents) != len(self.all_embeddings):
            print(f"Error: Document count ({len(self.all_documents)}) doesn't match embedding count ({len(self.all_embeddings)})")
            return []
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.all_embeddings):
            try:
                doc = self.all_documents[i]
                
                # Filter by document type if specified
                if document_type and doc["metadata"]["type"] != document_type:
                    continue
                
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append((doc, similarity))
            except IndexError as e:
                print(f"Index error at position {i}: {e}")
                print(f"Documents length: {len(self.all_documents)}, Embeddings length: {len(self.all_embeddings)}")
                continue
        
        # Sort by similarity (descending) and return top_k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def load_all_documents(self):
        print("Creating patient documents...")
        self.patient_documents = self.create_patient_documents()
        
        print("Creating medication documents...")
        self.medication_documents = self.create_medication_documents()
        
        print("Creating symptom documents...")
        self.symptom_documents = self.create_symptom_documents()
        
        # Combine all documents
        self.all_documents = self.patient_documents + self.medication_documents + self.symptom_documents
        
        print(f"Creating embeddings for {len(self.all_documents)} documents...")
        
        # Create embeddings for all documents
        self.all_embeddings = []  # Reset embeddings list
        successful_docs = []  # Track successful documents
        
        for i, doc in enumerate(self.all_documents):
            embedding = self.get_embedding(doc["content"])
            if embedding:
                self.all_embeddings.append(embedding)
                successful_docs.append(doc)
                print(f"Created embedding {len(self.all_embeddings)}/{len(self.all_documents)}")
            else:
                print(f"Failed to create embedding for document {i+1}, skipping...")
        
        # Update all_documents to only include documents with successful embeddings
        self.all_documents = successful_docs
        
        print(f"Successfully loaded {len(self.all_embeddings)} documents with embeddings")
        
        # Verify synchronization
        if len(self.all_documents) != len(self.all_embeddings):
            print(f"Warning: Document count ({len(self.all_documents)}) doesn't match embedding count ({len(self.all_embeddings)})")
        else:
            print("Document and embedding lists are synchronized")
    
    def search_patient_records(self, patient_name: str) -> List[Tuple[Dict[str, Any], float]]:
        print(Fore.YELLOW + f"doing a similarity search for patient: {patient_name}" + Style.RESET_ALL)
        docs = self.similarity_search(
            f"patient {patient_name} medical record",
            document_type="patient_record",
            top_k=3
        )
        print(Fore.YELLOW + f"Returned docs from similarity search: {docs}" + Style.RESET_ALL)
        return docs
    
    def search_medication_info(self, medication_name: str) -> List[Tuple[Dict[str, Any], float]]:
        return self.similarity_search(
            f"medication {medication_name} contraindications interactions",
            document_type="medication",
            top_k=2
        )
    
    def search_symptom_info(self, symptom: str) -> List[Tuple[Dict[str, Any], float]]:
        return self.similarity_search(
            f"symptom {symptom} causes treatment",
            document_type="symptom",
            top_k=2
        )

# Global instance for the chatbot to use
patient_processor = PatientDataProcessor()

if __name__ == "__main__":
    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY environment variable")
        exit(1)
    
    print("Loading patient data and creating embeddings...")
    patient_processor.load_all_documents()
    print("Patient data processor ready!") 