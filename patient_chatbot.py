"""
Patient Query Q&A Chatbot using Chainlit and Langchain.
This application allows patients to ask medical questions and get personalized advice
based on their medical history using OpenAI embeddings and in-memory similarity search.
"""

import os
import time
from typing import cast, List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from colorama import init, Fore, Back, Style

import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from patient_data_processor import patient_processor
from galileo import GalileoLogger
from galileo_core.schemas.shared.document import Document

# Initialize colorama for cross-platform colored output
init()

# Load environment variables
# 1) load global/shared first
load_dotenv(os.path.expanduser("~/.config/secrets/myapps.env"), override=False)
# 2) then load per-app .env (if present) to override selectively
load_dotenv(find_dotenv(usecwd=True), override=True)

# Initialize Galileo logger once at module level
galileo_logger = None
galileo_project = None
galileo_log_stream = None

# Initialize Galileo logger if configuration is available
api_key = os.getenv("GALILEO_API_KEY")
project = os.getenv("GALILEO_PROJECT")
log_stream = os.getenv("GALILEO_LOG_STREAM")

print(f"Galileo Configuration:")
print(f"  API Key: {'Set' if api_key else 'Not set'}")
print(f"  Project: {project}")
print(f"  Log Stream: {log_stream}")

if all([api_key, project, log_stream]):
    galileo_project = project
    galileo_log_stream = log_stream
    galileo_logger = GalileoLogger(project=project, log_stream=log_stream)
    print(Fore.GREEN + "Galileo logger initialized successfully." + Style.RESET_ALL)
else:
    print("Warning: Missing Galileo configuration. Logging will be disabled.")

def extract_patient_name(query: str) -> str:
    """Extract patient name from query using OpenAI."""
    if not query or len(query.strip()) < 5:  # Skip very short queries
        return None
    
    try:
        # Use OpenAI to extract any name from the sentence
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the person's name from the given sentence. 
            
            Look for a person's name mentioned in the text. This could be:
            - The speaker introducing themselves
            - A doctor or nurse asking about a patient
            - Someone being referred to
            - Any name in the sentence
            
            Return ONLY the person's name, nothing else. If no name is found, return "None".
            
            Examples:
            - "I am John Smith and I have a headache" → "John Smith"
            - "I'm Sarah, can you help me?" → "Sarah"
            - "My name is Michael Chen" → "Michael Chen"
            - "I am Atin Sanyal" → "Atin Sanyal"
            - "Can you help Emily Rodriguez?" → "Emily Rodriguez"
            - "I have a fever" → "None"
            - "The patient Sarah Johnson needs medication" → "Sarah Johnson"
            - Can I give Atin Sanyal amoxicillin for a sinus infection?
            - What’s a good painkiller for Sarah Johnson’s headache?
            - Can I prescribe ibuprofen to Michael Chen for back pain?
            - Which medication works well for Emily Rodriguez’s depression and anemia?
            """),
            ("human", f"Extract the person's name from this sentence: {query}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        
        # Clean up the result
        name = result.strip()
        if name.lower() in ["none", "no name", "not found", ""]:
            return None
        
        # Remove any extra punctuation or words
        name = name.replace('"', '').replace("'", "").strip()
        
        return name if name else None
        
    except Exception as e:
        print(f"Error extracting patient name with OpenAI: {e}")
        return None

def search_patient_records(patient_name: str) -> List[Dict[str, Any]]:
    """Search for patient records using similarity search."""
    if not patient_name:
        return []
    

    if not patient_processor.all_documents or not patient_processor.all_embeddings:
        print("Warning: Documents not loaded. Returning empty results.")
        return []
    
    results = patient_processor.search_patient_records(patient_name)
    
    return [doc for doc, score in results]

def search_medication_info(medication_name: str) -> List[Dict[str, Any]]:
    """Search for medication information using similarity search."""
    if not medication_name:
        return []
    
    if not patient_processor.all_documents or not patient_processor.all_embeddings:
        print("Warning: Documents not loaded. Returning empty results.")
        return []
    
    results = patient_processor.search_medication_info(medication_name)
    
    return [doc for doc, score in results]

def search_symptom_info(symptom: str) -> List[Dict[str, Any]]:
    """Search for symptom information using similarity search."""
    if not symptom:
        return []
    
    if not patient_processor.all_documents or not patient_processor.all_embeddings:
        print("Warning: Documents not loaded. Returning empty results.")
        return []
    
    results = patient_processor.search_symptom_info(symptom)
    
    return [doc for doc, score in results]

def check_contraindications(patient_info: str, medication_info: str, glog=None) -> str:
    """Check for contraindications between patient and medication."""
    if not patient_info or not medication_info:
        return ""
    
    start_time = time.time()
    
    # Use OpenAI to analyze contraindications
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical assistant analyzing potential contraindications between a patient's medical history and a medication. 
        
        Analyze the patient information and medication information provided. Look for:
        1. Allergies to the medication or similar drugs
        2. Medical conditions that contraindicate the medication
        3. Drug interactions with current medications
        4. Age-related contraindications
        
        If you find any contraindications, explain them clearly and recommend against taking the medication.
        If no contraindications are found, state that the medication appears safe based on the provided information.
        
        Always recommend consulting a healthcare provider for final medical decisions."""),
        ("human", f"""
        Patient Information:
        {patient_info}
        
        Medication Information:
        {medication_info}
        
        Please analyze for contraindications and provide a clear recommendation.
        """)
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({})
        
        # Log the contraindication analysis to Galileo (if available)
        if glog:
            try:
                print(f"[Galileo] Adding contraindication analysis span to trace...")
                glog.add_llm_span(
                    input=f"Patient Info: {patient_info}\nMedication Info: {medication_info}",
                    output=result,
                    name="Contraindication Analysis",
                    model="gpt-4o-mini",
                    duration_ns=int((time.time() - start_time) * 1000000),
                    metadata={
                        "source": "patient_chatbot.py",
                        "type": "contraindication_analysis"
                    }
                )
                print(f"[Galileo] Contraindication analysis span added successfully to project '{galileo_project}', log stream '{galileo_log_stream}'.")
            except Exception as e:
                print(f"Warning: Could not log to Galileo (project: {galileo_project}, log stream: {galileo_log_stream}): {e}")
        
        return result
    except Exception as e:
        error_msg = f"Error analyzing contraindications: {str(e)}"
        
        # Log the error to Galileo (if available)
        if glog:
            try:
                print(f"[Galileo] Adding contraindication analysis error span to trace...")
                glog.add_llm_span(
                    input=f"Patient Info: {patient_info}\nMedication Info: {medication_info}",
                    output=error_msg,
                    name="Contraindication Analysis Error",
                    model="gpt-4o-mini",
                    duration_ns=int((time.time() - start_time) * 1000000),
                    metadata={
                        "source": "patient_chatbot.py",
                        "type": "contraindication_analysis_error",
                        "error": str(e)
                    }
                )
                print(f"[Galileo] Contraindication analysis error span added successfully to project '{galileo_project}', log stream '{galileo_log_stream}'.")
            except Exception as log_error:
                print(f"Warning: Could not log error to Galileo (project: {galileo_project}, log stream: {galileo_log_stream}): {log_error}")
        
        return error_msg

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    await cl.Message(
        content="🔄 Loading patient data and creating embeddings... This may take a moment."
    ).send()
    
    try:
        # Load all documents and create embeddings
        patient_processor.load_all_documents()
        await cl.Message(
            content="✅ Patient data loaded successfully! I'm ready to help you with your health questions."
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"❌ Error loading patient data: {str(e)}"
        ).send()
        return

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from the user."""
    user_query = message.content
    start_time = time.time()

    try:
        if galileo_logger:
            print(f"Starting Galileo trace for query: {user_query[:50]}...")
            galileo_logger.start_trace(
                input=user_query,
                name=f"user_query: {user_query[:50]}...",
                tags=["patient"]
            )
            print(f"Galileo trace started successfully")

        patient_name = extract_patient_name(user_query)
        print(Fore.MAGENTA + f"Patient name extracted from query: {patient_name}" + Style.RESET_ALL)
        patient_records = search_patient_records(patient_name) if patient_name else []
        print(Fore.YELLOW + f"Patient records: {patient_records}" + Style.RESET_ALL)
        if galileo_logger:
            galileo_logger.add_retriever_span(
                input=patient_name or "Patient",
                output=patient_records,
                name="Patient Records Retrieval",
                duration_ns=int((time.time() - start_time) * 1000000),
                metadata={
                    "document_count": str(len(patient_records)),
                    "patient_name": patient_name or "UNK",
                }
            )
            print(Fore.GREEN + "added retriever span." + Style.RESET_ALL)

        # Search for medication information in the query
        medication_keywords = ["medication", "medicine", "drug", "pill", "prescription"]
        medication_info = []
        for keyword in medication_keywords:
            if keyword in user_query.lower():
                words = user_query.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        med_results = search_medication_info(word)
                        medication_info.extend(med_results)
                break
        
        symptom_keywords = ["symptom", "pain", "ache", "fever", "headache", "nausea", "dizziness"]
        symptom_info = []
        for keyword in symptom_keywords:
            if keyword in user_query.lower():
                symptom_results = search_symptom_info(keyword)
                symptom_info.extend(symptom_results)
                break
        
        context_parts = []
        
        if patient_records:
            patient_context = "\n\n".join([doc["content"] for doc in patient_records])
            context_parts.append(f"Patient Information:\n{patient_context}")
        
        if medication_info:
            med_context = "\n\n".join([doc["content"] for doc in medication_info])
            context_parts.append(f"Medication Information:\n{med_context}")
        
        if symptom_info:
            symptom_context = "\n\n".join([doc["content"] for doc in symptom_info])
            context_parts.append(f"Symptom Information:\n{symptom_context}")
        
        context = "\n\n".join(context_parts)
        
        # Create the LLM chain
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful medical assistant. You have access to patient records, medication information, and symptom data. 
            
            When responding:
            1. Be empathetic and professional
            2. Use the provided context to give personalized advice
            3. Always recommend consulting a healthcare provider for serious medical decisions
            4. If you don't have enough information, ask for clarification
            5. Never make definitive diagnoses - only provide general guidance
            
            If you have patient information, use it to personalize your response.
            If you have medication information, explain it clearly.
            If you have symptom information, provide relevant guidance."""),
            ("human", "{input}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        if context:
            full_input = f"Context:\n{context}\n\nUser Question: {user_query}"
        else:
            full_input = user_query
        
        llm_start_time = time.time()
        response = chain.invoke({"input": full_input})
        llm_end_time = time.time()
        
        if galileo_logger:
            print(f"Adding LLM span to Galileo trace...")
            galileo_logger.add_llm_span(
                input=full_input,
                output=response,
                name="Health Bot Response",
                model="gpt-4o",
                duration_ns=int((llm_end_time - llm_start_time) * 1000000),
                metadata={
                    "type": "health_bot_response",
                    "patient_name": patient_name or "",
                    "has_patient_records": str(bool(patient_records)),
                    "has_medication_info": str(bool(medication_info))
                }
            )
            print(Fore.GREEN + "LLM span added successfully." + Style.RESET_ALL)

        contraindication_analysis = ""
        if patient_records and medication_info:
            patient_summary = "\n".join([doc["content"] for doc in patient_records[:2]])  # Use first 2 records
            medication_summary = "\n".join([doc["content"] for doc in medication_info[:2]])  # Use first 2 records
            
            contraindication_analysis = check_contraindications(
                patient_summary, 
                medication_summary, 
                galileo_logger
            )
            
            if contraindication_analysis:
                response += f"\n\n**Important Safety Check:**\n{contraindication_analysis}"
        
        await cl.Message(content=response).send()
        
        if galileo_logger:
            print(f"Concluding Galileo trace...")
            galileo_logger.conclude(
                output=response,
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            print(f"Flushing Galileo trace to UI...")
            galileo_logger.flush()
            print(Fore.GREEN + "Galileo trace concluded and flushed successfully." + Style.RESET_ALL)
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
        
        # Log the error to Galileo (if available)
        if galileo_logger:
            print(Fore.RED + f"Concluding Galileo trace with error. project '{galileo_project}', log stream '{galileo_log_stream}'..." + Style.RESET_ALL)
            galileo_logger.conclude(
                output={"error": str(e)},
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=500
            )
            print(f"Flushing error trace to Galileo UI...")
            galileo_logger.flush()
            print(Fore.RED + "Error trace flushed successfully." + Style.RESET_ALL)

        await cl.Message(content=error_msg).send()

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        exit(1)
    
    print("Patient Query Chatbot is ready!")
    print("Run with: chainlit run patient_chatbot.py") 