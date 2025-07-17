"""
LangGraph Agent-Based Medical Assistant
Orchestrates RAG Q&A, Claim Approval, and Intent Classification in a DAG workflow.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from colorama import init, Fore, Style

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from galileo import GalileoLogger

from rag_tool import RAGTool
from claim_approval_tool import ClaimApprovalTool
from intent_classifier import IntentClassifier, IntentResult
from prior_auth_api_tool import PriorAuthAPITool
from instructions import fda_compliant_regulations_prompt

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

# Define the state schema
class AgentState(TypedDict):
    user_input: str
    intent_result: Optional[IntentResult]
    rag_response: Optional[str]
    rag_documents: Optional[List[Dict[str, Any]]]
    claim_response: Optional[str]
    prior_auth_response: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]

class MedicalAgent:
    def __init__(self):
        """Initialize the medical agent with all tools and components."""
        print("Initializing Medical Agent...")
        
        # Initialize tools
        self.rag_tool = RAGTool()
        self.claim_tool = ClaimApprovalTool()
        self.intent_classifier = IntentClassifier()
        self.prior_auth_tool = PriorAuthAPITool(induce_prior_auth_error=os.getenv("INDUCE_PRIOR_AUTH_ERROR", "False").lower() == "true")
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("Medical Agent initialized successfully!")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("route_request", lambda state: state)  # Simple pass-through node
        workflow.add_node("rag_lookup", self._rag_lookup_node)  # Always do RAG first
        workflow.add_node("claim_approval", self._claim_approval_node)
        workflow.add_node("prior_auth", self._prior_auth_node)
        workflow.add_node("combine_responses", self._combine_responses_node)
        workflow.add_node("generate_final_response", self._generate_final_response_node)
        
        workflow.add_edge("classify_intent", "route_request")
        
        workflow.add_conditional_edges(
            "route_request",
            self._route_request_node,
            {
                "rag_lookup": "rag_lookup",
                "generate_final_response": "generate_final_response"
            }
        )
        
        # Add conditional edge from rag_lookup to claim_approval or combine_responses
        workflow.add_conditional_edges(
            "rag_lookup",
            self._check_after_rag,
            {
                "claim_approval": "claim_approval",
                "combine_responses": "combine_responses"
            }
        )
        
        # Add conditional edge from claim_approval to prior_auth or combine_responses
        workflow.add_conditional_edges(
            "claim_approval",
            self._check_claim_approval_result,
            {
                "prior_auth": "prior_auth",
                "combine_responses": "combine_responses"
            }
        )
        
        workflow.add_edge("prior_auth", "combine_responses")
        workflow.add_edge("combine_responses", "generate_final_response")
        workflow.add_edge("generate_final_response", END)
        workflow.set_entry_point("classify_intent")

        return workflow.compile()
    
    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Classify the intent of the user input."""
        start_time = time.time()
        
        try:
            print(f"Classifying intent for: {state['user_input'][:50]}...")
            
            intent_result = self.intent_classifier.classify_intent(state['user_input'])
            
            print(f"Intent classified as: {intent_result.intent_type} (confidence: {intent_result.confidence:.2f})")
            
            # Log the intent classification to Galileo
            if galileo_logger:
                if intent_result.intent_type != "unknown":
                    galileo_logger.add_tool_span(
                        input=state['user_input'],
                        output=json.dumps({
                            "planner_output": intent_result.intent_type,
                            "confidence": intent_result.confidence,
                            "reason": intent_result.reasoning
                        }),
                        name="Intent Classification",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={
                            "source": "langgraph_agent.py",
                            "type": "intent_classification",
                            "intent_type": intent_result.intent_type,
                            "confidence": str(intent_result.confidence),
                            "reasoning": intent_result.reasoning,
                            "extracted_data": json.dumps(intent_result.extracted_data or {})
                        }
                    )
                else:
                    galileo_logger.add_llm_span(
                        input=state['user_input'],
                        output=json.dumps({
                            "planner_output": intent_result.intent_type,
                            "confidence": intent_result.confidence,
                            "reason": intent_result.reasoning
                        }),
                        name="Intent Classification",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        model="gpt-4o",
                        metadata={
                            "source": "langgraph_agent.py",
                            "type": "intent_classification",
                            "intent_type": intent_result.intent_type,
                            "confidence": str(intent_result.confidence),
                            "reasoning": intent_result.reasoning,
                            "extracted_data": json.dumps(intent_result.extracted_data or {})
                        }
                    )
            
            return {
                **state,
                "intent_result": intent_result,
                "metadata": {
                    **state.get("metadata", {}),
                    "intent_classification": {
                        "intent_type": intent_result.intent_type,
                        "confidence": intent_result.confidence,
                        "reasoning": intent_result.reasoning
                    }
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error in intent classification: {str(e)}"
            }
    
    def _route_request_node(self, state: AgentState) -> str:
        """Route the request based on intent classification."""
        intent_result = state.get("intent_result")
        
        if not intent_result:
            return "generate_final_response"
        
        intent_type = intent_result.intent_type
        
        # Always do RAG lookup first for any query that might involve patient information
        if intent_type in ["qa", "claim_approval", "both"]:
            return "rag_lookup"
        else:
            return "generate_final_response"
    
    def _check_after_rag(self, state: AgentState) -> str:
        """Check after RAG lookup whether to proceed with claim approval."""
        intent_result = state.get("intent_result")
        
        if not intent_result:
            return "combine_responses"
        
        intent_type = intent_result.intent_type
        
        # If it's a pure Q&A query, we're done after RAG
        if intent_type == "qa":
            return "combine_responses"
        
        # If it's claim_approval or both, proceed to claim approval
        elif intent_type in ["claim_approval", "both"]:
            return "claim_approval"
        
        return "combine_responses"
    
    def _check_claim_approval_result(self, state: AgentState) -> str:
        """Check if claim was approved and route to prior auth if needed."""
        claim_response = state.get("claim_response")
        
        if not claim_response:
            return "combine_responses"
        
        # Parse the claim response to check if it was approved
        try:
            if isinstance(claim_response, str):
                # Try to parse JSON response
                import json
                claim_data = json.loads(claim_response)
                decision = claim_data.get("decision", "").lower()
            else:
                decision = str(claim_response).lower()
            
            # Check if claim was approved
            if "approved" in decision:
                print("Claim approved - proceeding to prior authorization check...")
                return "prior_auth"
            else:
                print("Claim denied - skipping prior authorization...")
                return "combine_responses"
                
        except (json.JSONDecodeError, AttributeError):
            # If we can't parse the response, assume it's not approved
            print("Could not parse claim response - skipping prior authorization...")
            return "combine_responses"
    
    def _rag_lookup_node(self, state: AgentState) -> AgentState:
        """Process RAG lookup request."""
        try:
            print("Processing RAG lookup request...")
            
            user_input = state['user_input']
            intent_result = state.get('intent_result')
            
            # Extract patient name if available
            patient_name = None
            if intent_result and intent_result.extracted_data:
                patient_name = intent_result.extracted_data.get("patient_name")
            
            # Use RAG tool
            rag_result = self.rag_tool._run(user_input, patient_name, galileo_logger)
            
            # Extract response and documents from the result
            rag_response = rag_result["response"]
            rag_documents = rag_result["documents"]
            
            print(f"RAG response generated (length: {len(rag_response)})")
            print(f"RAG documents retrieved: {len(rag_documents)}")
            
            return {
                **state,
                "rag_response": rag_response,
                "rag_documents": rag_documents,
                "metadata": {
                    **state.get("metadata", {}),
                    "rag_processed": True,
                    "rag_documents_count": len(rag_documents)
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error in RAG lookup processing: {str(e)}"
            }
    
    def _claim_approval_node(self, state: AgentState) -> AgentState:
        """Process claim approval request using RAG information."""
        start_time = time.time()
        
        try:
            print("Processing claim approval request with RAG information...")
            
            user_input = state['user_input']
            intent_result = state.get('intent_result')
            rag_response = state.get('rag_response')
            
            # Extract claim information
            claim_info = self._extract_claim_info(user_input, intent_result)
            
            if not claim_info:
                # This is an error. 
                error_msg = "Could not extract the necessary information for claim approval. Please provide: patient name, treatment type, cost, diagnosis, age, and insurance type."

                claim_info_input = {
                    "input": user_input,
                    "claim_info": claim_info,
                    "error": error_msg
                }
                
                if galileo_logger:
                    galileo_logger.add_tool_span(
                        input=json.dumps(claim_info_input),  # Convert dict to string
                        output=error_msg,
                        name="Claim Prediction",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={
                            "type": "claim_prediction",
                            "user_input": user_input,
                            "error": "Could not extract claim information"
                        }
                    )
                
                return {
                    **state,
                    "claim_response": error_msg,
                    "metadata": {
                        **state.get("metadata", {}),
                        "claim_processed": True,
                        "claim_error": "Missing required information"
                    }
                }
            
            # Enhance claim info with RAG information if available
            if rag_response:
                print("Enhancing claim decision with patient medical history...")
                # Add RAG information to claim metadata for better decision making
                claim_info["patient_medical_history"] = rag_response
                claim_info["has_medical_history"] = True
            else:
                claim_info["has_medical_history"] = False
            
            # Use claim approval tool
            claim_start_time = time.time()
            
            # Filter out non-standard parameters for the claim tool
            claim_params = {
                'patient_name': claim_info['patient_name'],
                'treatment_type': claim_info['treatment_type'],
                'cost': claim_info['cost'],
                'diagnosis': claim_info['diagnosis'],
                'age': claim_info['age'],
                'insurance_type': claim_info['insurance_type']
            }
            
            claim_response = self.claim_tool._run(**claim_params)
            claim_end_time = time.time()
            
            print(f"Claim response generated: {claim_response[:100]}...")
            
            if galileo_logger:
                galileo_logger.add_tool_span(
                    input=f"{json.dumps(claim_info, indent=2)}",
                    output=claim_response,
                    name="Claim Prediction",
                    duration_ns=int((claim_end_time - claim_start_time) * 1000000),
                    metadata={
                        "type": "claim_prediction",
                        "claim_info": json.dumps(claim_info),
                        "user_input": user_input,
                        "used_rag_info": str(bool(rag_response))
                    }
                )
            
            return {
                **state,
                "claim_response": claim_response,
                "metadata": {
                    **state.get("metadata", {}),
                    "claim_processed": True,
                    "used_rag_info": str(bool(rag_response)),
                    "claim_info": claim_info
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error in claim processing: {str(e)}"
            }
    
    def _prior_auth_node(self, state: AgentState) -> AgentState:
        """Process prior authorization request."""
        start_time = time.time()
        
        try:
            print("Processing prior authorization request...")
            
            user_input = state['user_input']
            intent_result = state.get('intent_result')
            
            # Extract claim information for prior auth
            claim_info = self._extract_claim_info(user_input, intent_result)
            
            if not claim_info:
                error_msg = "Could not extract the necessary information for prior authorization. Please provide: patient name, treatment type, cost, diagnosis, age, and insurance type."
                
                if galileo_logger:
                    galileo_logger.add_tool_span(
                        input=user_input,
                        output=error_msg,
                        name="Prior Authorization",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={
                            "type": "prior_authorization",
                            "user_input": user_input,
                            "error": "Could not extract claim information"
                        }
                    )
                
                return {
                    **state,
                    "prior_auth_response": error_msg,
                    "metadata": {
                        **state.get("metadata", {}),
                        "prior_auth_processed": True,
                        "prior_auth_error": "Missing required information"
                    }
                }
            
            # Use prior authorization tool
            prior_auth_start_time = time.time()
            prior_auth_response = self.prior_auth_tool.check_prior_auth_requirement(
                patient_id=f"P{claim_info['age']}{hash(claim_info['patient_name']) % 1000:03d}",
                patient_name=claim_info['patient_name'],
                treatment_type=claim_info['treatment_type'],
                diagnosis=claim_info['diagnosis'],
                insurance_type=claim_info['insurance_type'],
                cost=claim_info['cost']
            )
            prior_auth_end_time = time.time()
            
            # Convert response to JSON string for consistency
            if isinstance(prior_auth_response, dict):
                prior_auth_response_str = json.dumps(prior_auth_response, indent=2)
            else:
                prior_auth_response_str = str(prior_auth_response)
            
            print(f"Prior auth response generated: {prior_auth_response_str[:100]}...")
            
            if galileo_logger:
                galileo_logger.add_tool_span(
                    input=f"{json.dumps(claim_info, indent=2)}",
                    output=prior_auth_response_str,
                    name="Prior Authorization",
                    duration_ns=int((prior_auth_end_time - prior_auth_start_time) * 1000000),
                    metadata={
                        "type": "prior_authorization",
                        "claim_info": json.dumps(claim_info),
                        "user_input": user_input
                    }
                )
            
            return {
                **state,
                "prior_auth_response": prior_auth_response_str,
                "metadata": {
                    **state.get("metadata", {}),
                    "prior_auth_processed": True,
                    "claim_info": claim_info
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error in prior authorization processing: {str(e)}"
            }
    
    def _extract_claim_info(self, user_input: str, intent_result: Optional[IntentResult]) -> Optional[Dict[str, Any]]:
        """Extract claim information from user input."""
        extracted_data = intent_result.extracted_data if intent_result else {}
        
        patient_info = self.intent_classifier.extract_patient_info(user_input)
        
        # Use LLM to extract missing information
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract medical claim information from the user input. Return a JSON object with the following fields:
                - patient_name: string
                - treatment_type: one of ['surgery', 'medication', 'therapy', 'imaging', 'lab_test', 'emergency_room', 'specialist_consultation', 'preventive_care']
                - cost: float (in dollars) - extract the dollar amount mentioned
                - diagnosis: one of ['hypertension', 'diabetes', 'heart_disease', 'cancer', 'asthma', 'depression', 'arthritis', 'infection', 'injury', 'chronic_pain']
                - age: integer - estimate age if not provided (use 45 as default for adults)
                - insurance_type: one of ['private', 'medicare', 'medicaid', 'uninsured'] - assume 'private' if not specified
                
                IMPORTANT: Do not use null values. Make reasonable assumptions:
                - If cost is mentioned as "$15,000", extract 15000.0
                - If age is not mentioned, use 45
                - If insurance type is not mentioned, use "private"
                - If diagnosis mentions "heart disease", use "heart_disease"
                - If treatment mentions "surgery", use "surgery"
                
                Only return the JSON object."""),
                ("human", f"Extract claim information from: {user_input}")
            ])
            
            chain = extraction_prompt | self.llm | StrOutputParser()
            response = chain.invoke({})
            
            # Defensive: check if response is empty or not valid JSON
            if not response or not response.strip():
                print("LLM claim extraction returned empty response!")
                return None
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            try:
                claim_info = json.loads(cleaned_response)
            except Exception as e:
                print(f"Error extracting claim info: {e}\nRaw LLM response: {response}\nCleaned response: {cleaned_response}")
                return None
            
            # Merge with extracted data
            if patient_info.get("name"):
                claim_info["patient_name"] = patient_info["name"]
            if patient_info.get("age"):
                claim_info["age"] = patient_info["age"]
            if patient_info.get("cost"):
                claim_info["cost"] = patient_info["cost"]
            
            # Handle null values and provide defaults
            if claim_info.get("age") is None:
                claim_info["age"] = 45  # Default age
            if claim_info.get("insurance_type") is None:
                claim_info["insurance_type"] = "private"  # Default insurance type
            if claim_info.get("cost") is None:
                # Try to extract cost from the query
                import re
                cost_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', user_input)
                if cost_match:
                    cost_str = cost_match.group(1).replace(',', '')
                    claim_info["cost"] = float(cost_str)
                else:
                    claim_info["cost"] = 5000.0  # Default cost
            
            # Check if we have enough information
            required_fields = ["patient_name", "treatment_type", "cost", "diagnosis", "age", "insurance_type"]
            missing_fields = [field for field in required_fields if claim_info.get(field) is None]
            
            if missing_fields:
                print(f"Missing claim fields: {missing_fields}")
                return None
            
            return claim_info
            
        except Exception as e:
            print(f"Error extracting claim info (outer): {e}")
            return None
    
    def _generate_conversational_response(self, claim_response: str = None, prior_auth_response: str = None, rag_response: str = None) -> str:
        """Generate a conversational response from the tool outputs using FDA-compliant regulations."""
        
        # Prepare results by removing patient names for anonymity
        def anonymize_patient_names(text):
            if not text:
                return text
            # Simple regex to replace patient names with "Patient"
            import re
            # Replace common name patterns with "Patient"
            anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'Patient', text)
            # Also replace single names that might be patient names
            anonymized = re.sub(r'\b[A-Z][a-z]{2,}\b(?=\s+(?:has|is|was|needs|requires|undergoing))', 'Patient', anonymized)
            return anonymized
        
        # Anonymize all responses
        anonymized_rag = anonymize_patient_names(rag_response)
        anonymized_claim = anonymize_patient_names(claim_response)
        anonymized_prior_auth = anonymize_patient_names(prior_auth_response)
        
        # Combine all results
        results_parts = []
        if anonymized_rag:
            results_parts.append(f"Medical Information: {anonymized_rag}")
        if anonymized_claim:
            results_parts.append(f"Claim Status: {anonymized_claim}")
        if anonymized_prior_auth:
            results_parts.append(f"Prior Authorization: {anonymized_prior_auth}")
        
        results = "\n\n".join(results_parts) if results_parts else "No specific information available."
        
        # Create the prompt with results
        prompt = fda_compliant_regulations_prompt.format(results=results)
        
        # Generate response using OpenAI
        try:
            response = self.llm.invoke(prompt)
            return response.content, prompt
        except Exception as e:
            # Fallback response if LLM generation fails
            return "I have the information you requested, but I'm having trouble formatting it right now. Please try asking your question again."

    def _combine_responses_node(self, state: AgentState) -> AgentState:
        """Combine responses from different tools."""
        start_time = time.time()
        
        rag_response = state.get("rag_response")
        rag_documents = state.get("rag_documents", [])  # Get RAG documents
        claim_response = state.get("claim_response")
        prior_auth_response = state.get("prior_auth_response")
        
        # Generate conversational response
        final_response, prompt = self._generate_conversational_response(
            claim_response=claim_response,
            prior_auth_response=prior_auth_response,
            rag_response=rag_response
        )
        
        # Log the final response generation to Galileo
        if galileo_logger:
            intent_result = state.get("intent_result")
            intent_type = intent_result.intent_type if intent_result else "unknown"
            
            # Prepare RAG documents metadata
            rag_docs_metadata = []
            rag_docs_content = []
            for i, doc in enumerate(rag_documents[:3]):  # Include top 3 documents
                rag_docs_metadata.append({
                    "document_index": i,
                    "type": doc.get("metadata", {}).get("type", "unknown"),
                    "score": doc.get("score", 0.0),
                    "content_preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                })
                # Add raw content for the input
                rag_docs_content.append(doc.get("content", ""))
            
            # Create the input with user query and RAG document content
            user_query = state.get("user_input", "")
            rag_content_text = "\n\n".join(rag_docs_content) if rag_docs_content else ""

            print(Fore.YELLOW + f"RAG content text: {rag_content_text}" + Style.RESET_ALL)
            
            formatted_input = f"""
                Answer the user query based on the context provided.
                User query: {user_query}
                Context:
                {rag_content_text}
            """
            
            galileo_logger.add_llm_span(
                input=formatted_input,
                output=final_response,
                name="Final Response Generation",
                model="gpt-4o",
                duration_ns=int((time.time() - start_time) * 1000000),
                metadata={
                    "type": "LLM",
                    "intent_type": intent_type
                }
            )
        
        return {
            **state,
            "final_response": final_response
        }
    
    def _generate_final_response_node(self, state: AgentState) -> AgentState:
        """Generate the final response to the user."""
        try:
            # If we already have a final response, use it
            if state.get("final_response"):
                return state
            
            # If there's an error, handle it
            if state.get("error"):
                error_response = f"I apologize, but I encountered an error: {state['error']}. Please try again."
                
                return {
                    **state,
                    "final_response": error_response
                }
            
            # If intent is unknown, provide a helpful response
            intent_result = state.get("intent_result")
            if intent_result and intent_result.intent_type == "unknown":
                unknown_response = """I'm not sure what you're asking about. I can help you with:
                
1. Medical questions about symptoms, medications, treatments, etc.
2. Insurance claim approvals and coverage questions
3. Both medical and insurance questions

Please try rephrasing your question or let me know what specific information you need."""
                
                return {
                    **state,
                    "final_response": unknown_response
                }
            
            # Default response
            default_response = "I'm here to help with your medical and insurance questions. Please let me know what you need assistance with."
            
            return {
                **state,
                "final_response": default_response
            }
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            
            return {
                **state,
                "final_response": error_response
            }
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query through the entire workflow."""
        start_time = time.time()
        
        try:
            # Start Galileo trace
            if galileo_logger:
                galileo_logger.start_trace(
                    input=user_input,
                    name=f"langgraph_query: {user_input[:50]}...",
                    tags=["langgraph", "medical_agent"]
                )
            
            # Initialize state
            initial_state = AgentState(
                user_input=user_input,
                intent_result=None,
                rag_response=None,
                rag_documents=None,
                claim_response=None,
                prior_auth_response=None,
                final_response=None,
                error=None,
                metadata={"start_time": start_time}
            )
            
            # Run the graph
            print(f"Processing query: {user_input[:50]}...")
            result = self.graph.invoke(initial_state)
            
            # Add processing time
            processing_time = time.time() - start_time
            result["metadata"]["processing_time"] = processing_time
            
            print(f"Query processed in {processing_time:.2f} seconds")
            
            # Conclude Galileo trace
            if galileo_logger:
                final_response = result.get("final_response", "No response generated")
                if not final_response:
                    final_response = "No response generated"
                
                galileo_logger.conclude(
                    output=final_response,
                    duration_ns=int((time.time() - start_time) * 1000000),
                    status_code=200
                )
                galileo_logger.flush()
                print(f"Successfully flushed 1 traces to project {galileo_project}, logstream {galileo_log_stream}")
            
            return result
            
        except Exception as e:
            # Conclude Galileo trace with error
            if galileo_logger:
                error_output = {"error": str(e)}
                if not str(e):
                    error_output = {"error": "Unknown error occurred"}
                
                galileo_logger.conclude(
                    output=error_output,
                    duration_ns=int((time.time() - start_time) * 1000000),
                    status_code=500
                )
                galileo_logger.flush()
                print(f"Successfully flushed 1 traces to project {galileo_project}, logstream {galileo_log_stream}")
            
            return {
                "user_input": user_input,
                "final_response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                "error": str(e),
                "metadata": {
                    "start_time": start_time,
                    "processing_time": time.time() - start_time
                }
            }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        return {
            "rag_tool": {
                "name": self.rag_tool.name,
                "description": self.rag_tool.description,
                "capabilities": "Search patient records, medications, and symptoms"
            },
            "claim_tool": {
                "name": self.claim_tool.name,
                "description": self.claim_tool.description,
                "capabilities": "Approve or deny medical claims based on patient and treatment data"
            },
            "prior_auth_tool": {
                "name": self.prior_auth_tool.name,
                "description": self.prior_auth_tool.description,
                "capabilities": "Check if procedures require prior authorization and submit authorization requests"
            },
            "intent_classifier": {
                "capabilities": "Classify user intent as Q&A, claim approval, prior authorization, or both"
            }
        }

# Create a simple interface for testing
def main():
    """Simple test interface for the medical agent."""
    agent = MedicalAgent()
    
    print("\nMedical Agent is ready!")
    print("Type 'quit' to exit")
    print("Type 'tools' to see available tools")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'tools':
                tool_info = agent.get_tool_info()
                print("\nAvailable Tools:")
                for tool_name, info in tool_info.items():
                    print(f"- {tool_name}: {info['description'] if 'description' in info else info['capabilities']}")
                continue
            elif not user_input:
                continue
            
            # Process the query
            result = agent.process_query(user_input)
            
            # Display the result
            print(f"\nAgent: {result['final_response']}")
            
            # Show metadata if available
            if result.get("metadata"):
                print(f"\n[Processing time: {result['metadata'].get('processing_time', 0):.2f}s]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main() 