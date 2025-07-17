"""
Intent Classifier
Determines whether the input is a Q&A request, claim approval request, or both.
"""

import re
import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Configure logging to suppress OpenAI HTTP requests
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class IntentResult(BaseModel):
    intent_type: str  # "qa", "claim_approval", "both", "unknown"
    confidence: float
    extracted_data: Dict[str, Any]
    reasoning: str

class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.qa_keywords = [
            "what", "how", "why", "when", "where", "which", "can you", "tell me",
            "explain", "describe", "information", "help", "question", "symptom",
            "medication", "treatment", "diagnosis", "allergy", "side effect",
            "interaction", "dosage", "frequency", "duration"
        ]
        
        self.claim_keywords = [
            "claim", "approval", "denied", "approved", "insurance", "coverage",
            "cost", "price", "bill", "payment", "reimbursement", "authorization",
            "pre-authorization", "medical necessity", "benefits", "policy",
            "deductible", "copay", "coinsurance", "out-of-pocket"
        ]
        
        self.patient_keywords = [
            "patient", "name", "age", "gender", "medical history", "condition",
            "diagnosis", "treatment", "procedure", "surgery", "therapy"
        ]
    
    def classify_intent(self, user_input: str) -> IntentResult:
        """Classify the intent of the user input."""
        user_input_lower = user_input.lower()
        
        # Check for explicit keywords
        qa_score = self._calculate_qa_score(user_input_lower)
        claim_score = self._calculate_claim_score(user_input_lower)
        
        # Use LLM for more sophisticated classification
        llm_result = self._llm_classify_intent(user_input)
        
        # Combine keyword and LLM results
        final_intent = self._combine_results(qa_score, claim_score, llm_result)
        
        return final_intent
    
    def _calculate_qa_score(self, text: str) -> float:
        """Calculate score for Q&A intent based on keywords."""
        score = 0.0
        total_keywords = len(self.qa_keywords)
        
        for keyword in self.qa_keywords:
            if keyword in text:
                score += 1.0
        
        return score / total_keywords if total_keywords > 0 else 0.0
    
    def _calculate_claim_score(self, text: str) -> float:
        """Calculate score for claim approval intent based on keywords."""
        score = 0.0
        total_keywords = len(self.claim_keywords)
        
        for keyword in self.claim_keywords:
            if keyword in text:
                score += 1.0
        
        return score / total_keywords if total_keywords > 0 else 0.0
    
    def _llm_classify_intent(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to classify intent more accurately."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a medical chatbot. Analyze the user input and determine the intent.

            Possible intents:
            1. "qa" - User is asking a medical question, seeking information about symptoms, medications, treatments, etc.
            2. "claim_approval" - User is asking about insurance claims, approvals, coverage, costs, etc.
            3. "both" - User is asking both a medical question AND about claims/insurance
            4. "unknown" - Intent is unclear

            Return a JSON object with:
            - intent_type: one of the above intents
            - confidence: float between 0 and 1
            - reasoning: brief explanation of your classification
            - extracted_data: any relevant data extracted (patient names, conditions, costs, etc.)

            Examples:
            - "What are the side effects of aspirin?" → {{"intent_type": "qa", "confidence": 0.9, "reasoning": "Medical question about medication side effects", "extracted_data": {{"medication": "aspirin"}}}}
            - "Will my insurance cover this surgery?" → {{"intent_type": "claim_approval", "confidence": 0.8, "reasoning": "Question about insurance coverage", "extracted_data": {{"treatment": "surgery"}}}}
            - "What medication should I take for my headache and will it be covered?" → {{"intent_type": "both", "confidence": 0.7, "reasoning": "Both medical question and insurance question", "extracted_data": {{"symptom": "headache"}}}}
            """),
            ("human", f"Classify the intent of this user input: {user_input}")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            
            print(f"🔍 LLM Intent Classification Debug:")
            print(f"   Input: {user_input}")
            print(f"   Raw LLM Response: {response.content}")
            
            # Parse JSON response
            import json
            result = json.loads(response.content)
            
            print(f"   Parsed Result: {result}")
            
            return result
        except Exception as e:
            print(f"❌ Error in LLM classification: {e}")
            print(f"   Raw response was: {response.content if 'response' in locals() else 'No response'}")
            return {
                "intent_type": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error in classification: {str(e)}",
                "extracted_data": {}
            }
    
    def _combine_results(self, qa_score: float, claim_score: float, llm_result: Dict[str, Any]) -> IntentResult:
        """Combine keyword scores and LLM results for final classification."""
        llm_intent = llm_result.get("intent_type", "unknown")
        llm_confidence = llm_result.get("confidence", 0.0)
        
        print(f"🔍 Intent Combination Debug:")
        print(f"   QA Score: {qa_score:.3f}")
        print(f"   Claim Score: {claim_score:.3f}")
        print(f"   LLM Intent: {llm_intent}")
        print(f"   LLM Confidence: {llm_confidence:.3f}")
        
        # Weight the results (LLM gets higher weight)
        llm_weight = 0.7
        keyword_weight = 0.3
        
        # Calculate combined confidence
        if llm_intent == "qa":
            combined_confidence = (llm_confidence * llm_weight) + (qa_score * keyword_weight)
        elif llm_intent == "claim_approval":
            combined_confidence = (llm_confidence * llm_weight) + (claim_score * keyword_weight)
        elif llm_intent == "both":
            combined_confidence = (llm_confidence * llm_weight) + (max(qa_score, claim_score) * keyword_weight)
        else:
            combined_confidence = max(qa_score, claim_score)
        
        print(f"   Combined Confidence: {combined_confidence:.3f}")
        
        # Determine final intent
        if combined_confidence < 0.3:
            final_intent = "unknown"
        elif llm_intent in ["qa", "claim_approval", "both"]:
            final_intent = llm_intent
        else:
            # Fallback to keyword-based classification
            if qa_score > claim_score and qa_score > 0.1:
                final_intent = "qa"
            elif claim_score > qa_score and claim_score > 0.1:
                final_intent = "claim_approval"
            elif qa_score > 0.1 and claim_score > 0.1:
                final_intent = "both"
            else:
                final_intent = "unknown"
        
        print(f"   Final Intent: {final_intent}")
        print(f"   Final Confidence: {combined_confidence:.3f}")
        
        return IntentResult(
            intent_type=final_intent,
            confidence=combined_confidence,
            extracted_data=llm_result.get("extracted_data", {}),
            reasoning=llm_result.get("reasoning", f"Keyword scores: QA={qa_score:.2f}, Claim={claim_score:.2f}")
        )
    
    def extract_patient_info(self, user_input: str) -> Dict[str, Any]:
        """Extract patient information from the input."""
        # Simple regex-based extraction
        patient_info = {}
        
        # Extract patient name (simple pattern)
        name_pattern = r"(?:patient|for|about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        name_match = re.search(name_pattern, user_input, re.IGNORECASE)
        if name_match:
            patient_info["name"] = name_match.group(1)
        
        # Extract age
        age_pattern = r"(\d+)\s*(?:years?\s*old|yo|y\.o\.)"
        age_match = re.search(age_pattern, user_input, re.IGNORECASE)
        if age_match:
            patient_info["age"] = int(age_match.group(1))
        
        # Extract cost
        cost_pattern = r"\$?(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:dollars?|USD)?"
        cost_match = re.search(cost_pattern, user_input, re.IGNORECASE)
        if cost_match:
            cost_str = cost_match.group(1).replace(",", "")
            patient_info["cost"] = float(cost_str)
        
        return patient_info 