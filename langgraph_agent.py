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

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

from rag_tool import RAGTool
from claim_approval_tool import ClaimApprovalTool
from intent_classifier import IntentClassifier, IntentResult

# Configure logging to suppress OpenAI HTTP requests
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Define the state schema
class AgentState(TypedDict):
    user_input: str
    intent_result: Optional[IntentResult]
    rag_response: Optional[str]
    claim_response: Optional[str]
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
        workflow.add_node("rag_qa", self._rag_qa_node)
        workflow.add_node("claim_approval", self._claim_approval_node)
        workflow.add_node("combine_responses", self._combine_responses_node)
        workflow.add_node("generate_final_response", self._generate_final_response_node)
        
        workflow.add_edge("classify_intent", "route_request")
        
        workflow.add_conditional_edges(
            "route_request",
            self._route_request_node,
            {
                "rag_qa": "rag_qa",
                "claim_approval": "claim_approval", 
                "generate_final_response": "generate_final_response"
            }
        )
        
        workflow.add_edge("rag_qa", "combine_responses")
        workflow.add_edge("claim_approval", "combine_responses")
        workflow.add_edge("combine_responses", "generate_final_response")
        workflow.add_edge("generate_final_response", END)
        workflow.set_entry_point("classify_intent")
        return workflow.compile() 
    
    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Classify the intent of the user input."""
        try:
            print(f"Classifying intent for: {state['user_input'][:50]}...")
            
            intent_result = self.intent_classifier.classify_intent(state['user_input'])
            
            print(f"Intent classified as: {intent_result.intent_type} (confidence: {intent_result.confidence:.2f})")
            
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
        
        if intent_type == "qa":
            return "rag_qa"
        elif intent_type == "claim_approval":
            return "claim_approval"
        elif intent_type == "both":
            # For "both" intent, we'll route to rag_qa first, then combine later
            return "rag_qa"
        else:
            return "generate_final_response"
    
    def _rag_qa_node(self, state: AgentState) -> AgentState:
        """Process RAG Q&A request."""
        try:
            print("Processing RAG Q&A request...")
            
            user_input = state['user_input']
            intent_result = state.get('intent_result')
            
            # Extract patient name if available
            patient_name = None
            if intent_result and intent_result.extracted_data:
                patient_name = intent_result.extracted_data.get("patient_name")
            
            # Use RAG tool
            rag_response = self.rag_tool._run(user_input, patient_name)
            
            print(f"RAG response generated (length: {len(rag_response)})")
            
            # If this is a "both" intent, also process claim approval
            claim_response = None
            if intent_result and intent_result.intent_type == "both":
                print("Also processing claim approval for 'both' intent...")
                claim_info = self._extract_claim_info(user_input, intent_result)
                if claim_info:
                    claim_response = self.claim_tool._run(**claim_info)
                    print(f"Claim response generated: {claim_response[:100]}...")
            
            return {
                **state,
                "rag_response": rag_response,
                "claim_response": claim_response,
                "metadata": {
                    **state.get("metadata", {}),
                    "rag_processed": True,
                    "claim_processed": bool(claim_response)
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error in RAG processing: {str(e)}"
            }
    
    def _claim_approval_node(self, state: AgentState) -> AgentState:
        """Process claim approval request."""
        try:
            print("Processing claim approval request...")
            
            user_input = state['user_input']
            intent_result = state.get('intent_result')
            
            # Extract claim information
            claim_info = self._extract_claim_info(user_input, intent_result)
            
            if not claim_info:
                return {
                    **state,
                    "claim_response": "I couldn't extract the necessary information for claim approval. Please provide: patient name, treatment type, cost, diagnosis, age, and insurance type.",
                    "metadata": {
                        **state.get("metadata", {}),
                        "claim_processed": True,
                        "claim_error": "Missing required information"
                    }
                }
            
            # Use claim approval tool
            claim_response = self.claim_tool._run(**claim_info)
            
            print(f"Claim response generated: {claim_response[:100]}...")
            
            return {
                **state,
                "claim_response": claim_response,
                "metadata": {
                    **state.get("metadata", {}),
                    "claim_processed": True,
                    "claim_info": claim_info
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error in claim processing: {str(e)}"
            }
    
    def _extract_claim_info(self, user_input: str, intent_result: Optional[IntentResult]) -> Optional[Dict[str, Any]]:
        """Extract claim information from user input."""
        # Try to extract from intent result first
        extracted_data = intent_result.extracted_data if intent_result else {}
        
        # Extract patient info
        patient_info = self.intent_classifier.extract_patient_info(user_input)
        
        # Use LLM to extract missing information
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract medical claim information from the user input. Return a JSON object with the following fields:
                - patient_name: string
                - treatment_type: one of ['surgery', 'medication', 'therapy', 'imaging', 'lab_test', 'emergency_room', 'specialist_consultation', 'preventive_care']
                - cost: float (in dollars)
                - diagnosis: one of ['hypertension', 'diabetes', 'heart_disease', 'cancer', 'asthma', 'depression', 'arthritis', 'infection', 'injury', 'chronic_pain']
                - age: integer
                - insurance_type: one of ['private', 'medicare', 'medicaid', 'uninsured']
                
                If any field cannot be determined, use null. Only return the JSON object."""),
                ("human", f"Extract claim information from: {user_input}")
            ])
            
            chain = extraction_prompt | self.llm | StrOutputParser()
            response = chain.invoke({})
            
            # Parse JSON response
            claim_info = json.loads(response)
            
            # Merge with extracted data
            if patient_info.get("name"):
                claim_info["patient_name"] = patient_info["name"]
            if patient_info.get("age"):
                claim_info["age"] = patient_info["age"]
            if patient_info.get("cost"):
                claim_info["cost"] = patient_info["cost"]
            
            # Check if we have enough information
            required_fields = ["patient_name", "treatment_type", "cost", "diagnosis", "age", "insurance_type"]
            missing_fields = [field for field in required_fields if not claim_info.get(field)]
            
            if missing_fields:
                print(f"Missing claim fields: {missing_fields}")
                return None
            
            return claim_info
            
        except Exception as e:
            print(f"Error extracting claim info: {e}")
            return None
    
    def _combine_responses_node(self, state: AgentState) -> AgentState:
        """Combine responses from different tools."""
        rag_response = state.get("rag_response")
        claim_response = state.get("claim_response")
        
        combined_responses = []
        
        if rag_response:
            combined_responses.append(f"Medical Information:\n{rag_response}")
        
        if claim_response:
            combined_responses.append(f"Claim Approval:\n{claim_response}")
        
        if not combined_responses:
            combined_responses.append("I couldn't process your request. Please try rephrasing your question.")
        
        return {
            **state,
            "final_response": "\n\n".join(combined_responses)
        }
    
    def _generate_final_response_node(self, state: AgentState) -> AgentState:
        """Generate the final response to the user."""
        try:
            # If we already have a final response, use it
            if state.get("final_response"):
                return state
            
            # If there's an error, handle it
            if state.get("error"):
                return {
                    **state,
                    "final_response": f"I apologize, but I encountered an error: {state['error']}. Please try again."
                }
            
            # If intent is unknown, provide a helpful response
            intent_result = state.get("intent_result")
            if intent_result and intent_result.intent_type == "unknown":
                return {
                    **state,
                    "final_response": """I'm not sure what you're asking about. I can help you with:
                    
1. Medical questions about symptoms, medications, treatments, etc.
2. Insurance claim approvals and coverage questions
3. Both medical and insurance questions

Please try rephrasing your question or let me know what specific information you need."""
                }
            
            # Default response
            return {
                **state,
                "final_response": "I'm here to help with your medical and insurance questions. Please let me know what you need assistance with."
            }
            
        except Exception as e:
            return {
                **state,
                "final_response": f"I apologize, but I encountered an error: {str(e)}. Please try again."
            }
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query through the entire workflow."""
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = AgentState(
                user_input=user_input,
                intent_result=None,
                rag_response=None,
                claim_response=None,
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
            
            return result
            
        except Exception as e:
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
            "intent_classifier": {
                "capabilities": "Classify user intent as Q&A, claim approval, or both"
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