"""
Test Script for LangGraph Medical Agent
Demonstrates the functionality with various types of queries.
"""

import os
import json
import time
from dotenv import load_dotenv, find_dotenv
from langgraph_agent import MedicalAgent

# Load environment variables
# 1) load global/shared first
load_dotenv(os.path.expanduser("~/.config/secrets/myapps.env"), override=False)
# 2) then load per-app .env (if present) to override selectively
load_dotenv(find_dotenv(usecwd=True), override=True)

def test_medical_agent():
    """Test the medical agent with various queries."""
    
    # Initialize the agent
    print("Initializing Medical Agent...")
    agent = MedicalAgent()
    print("Agent initialized successfully!\n")
    
    # Test queries
    test_queries = [
        # Q&A queries
        {
            "query": "What are the side effects of aspirin?",
            "expected_intent": "qa",
            "description": "Medical Q&A about medication side effects"
        },
        {
            "query": "Can you tell me about Atin Sanyal's medical history?",
            "expected_intent": "qa",
            "description": "Patient-specific medical information"
        },
        {
            "query": "What medication should I take for a headache?",
            "expected_intent": "qa",
            "description": "Medical advice query"
        },
        
        # Claim approval queries
        {
            "query": "Will insurance cover a $15,000 surgery for heart disease for a 45-year-old patient with private insurance?",
            "expected_intent": "claim_approval",
            "description": "Claim approval for expensive surgery"
        },
        {
            "query": "Is a $500 lab test approved for diabetes diagnosis for a 35-year-old with Medicare?",
            "expected_intent": "claim_approval",
            "description": "Claim approval for diagnostic test"
        },
        {
            "query": "Can I get approval for $2,000 therapy for depression for a 28-year-old patient?",
            "expected_intent": "claim_approval",
            "description": "Claim approval for therapy"
        },
        
        # Combined queries
        {
            "query": "What medication should I take for my headache and will it be covered by insurance?",
            "expected_intent": "both",
            "description": "Combined medical and insurance question"
        },
        {
            "query": "Tell me about Sarah Johnson's condition and if her treatment will be approved",
            "expected_intent": "both",
            "description": "Patient info + claim approval"
        },
        
        # Edge cases
        {
            "query": "Hello, how are you?",
            "expected_intent": "unknown",
            "description": "General greeting"
        },
        {
            "query": "What's the weather like?",
            "expected_intent": "unknown",
            "description": "Non-medical question"
        }
    ]
    
    # Run tests
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Intent: {test_case['expected_intent']}")
        print(f"{'='*60}")
        
        try:
            # Process the query
            start_time = time.time()
            result = agent.process_query(test_case['query'])
            processing_time = time.time() - start_time
            
            # Extract results
            actual_intent = result.get('metadata', {}).get('intent_classification', {}).get('intent_type', 'unknown')
            confidence = result.get('metadata', {}).get('intent_classification', {}).get('confidence', 0)
            final_response = result.get('final_response', 'No response')
            
            # Check if intent matches
            intent_correct = actual_intent == test_case['expected_intent']
            
            # Store results
            test_result = {
                'test_number': i,
                'description': test_case['description'],
                'query': test_case['query'],
                'expected_intent': test_case['expected_intent'],
                'actual_intent': actual_intent,
                'confidence': confidence,
                'intent_correct': intent_correct,
                'processing_time': processing_time,
                'response_length': len(final_response),
                'has_error': bool(result.get('error'))
            }
            
            results.append(test_result)
            
            # Print results
            print(f"Actual Intent: {actual_intent}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Intent Correct: {'✅' if intent_correct else '❌'}")
            print(f"Processing Time: {processing_time:.2f}s")
            print(f"Response Length: {len(final_response)} characters")
            
            if result.get('error'):
                print(f"Error: {result['error']}")
            
            # Show a snippet of the response
            response_snippet = final_response[:200] + "..." if len(final_response) > 200 else final_response
            print(f"Response Snippet: {response_snippet}")
            
        except Exception as e:
            print(f"Error in test {i}: {e}")
            results.append({
                'test_number': i,
                'description': test_case['description'],
                'query': test_case['query'],
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if 'error' not in r]
    failed_tests = [r for r in results if 'error' in r]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        intent_accuracy = sum(1 for r in successful_tests if r['intent_correct']) / len(successful_tests)
        avg_processing_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
        avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests)
        
        print(f"Intent Accuracy: {intent_accuracy:.2%}")
        print(f"Average Processing Time: {avg_processing_time:.2f}s")
        print(f"Average Confidence: {avg_confidence:.2f}")
    
    # Show detailed results
    print(f"\nDetailed Results:")
    for result in results:
        status = "✅" if result.get('intent_correct', False) else "❌"
        if 'error' in result:
            status = "💥"
        
        print(f"{status} Test {result['test_number']}: {result['description']}")
        if 'error' not in result:
            print(f"   Intent: {result['actual_intent']} (expected: {result['expected_intent']})")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Time: {result['processing_time']:.2f}s")

def test_individual_components():
    """Test individual components of the agent."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Test RAG Tool
    print("\n1. Testing RAG Tool...")
    from rag_tool import RAGTool
    rag_tool = RAGTool()
    
    rag_test_queries = [
        "What are the side effects of aspirin?",
        "Tell me about Atin Sanyal's medical history",
        "What medications interact with metformin?"
    ]
    
    for query in rag_test_queries:
        print(f"\n   Query: {query}")
        result = rag_tool._run(query)
        response = result["response"]
        documents = result["documents"]
        print(f"   Response: {response[:100]}...")
        print(f"   Documents retrieved: {len(documents)}")
    
    # Test Claim Approval Tool
    print("\n2. Testing Claim Approval Tool...")
    from claim_approval_tool import ClaimApprovalTool
    claim_tool = ClaimApprovalTool()
    
    claim_test_cases = [
        {
            "patient_name": "John Doe",
            "treatment_type": "surgery",
            "cost": 15000,
            "diagnosis": "heart_disease",
            "age": 45,
            "insurance_type": "private"
        },
        {
            "patient_name": "Jane Smith",
            "treatment_type": "lab_test",
            "cost": 500,
            "diagnosis": "diabetes",
            "age": 35,
            "insurance_type": "medicare"
        }
    ]
    
    for case in claim_test_cases:
        print(f"\n   Case: {case['patient_name']} - {case['treatment_type']} for {case['diagnosis']}")
        response = claim_tool._run(**case)
        print(f"   Response: {response[:100]}...")
    
    # Test Intent Classifier
    print("\n3. Testing Intent Classifier...")
    from intent_classifier import IntentClassifier
    intent_classifier = IntentClassifier()
    
    intent_test_queries = [
        "What are the side effects of aspirin?",
        "Will my insurance cover this surgery?",
        "What medication should I take and will it be covered?",
        "Hello, how are you?"
    ]
    
    for query in intent_test_queries:
        print(f"\n   Query: {query}")
        intent_result = intent_classifier.classify_intent(query)
        print(f"   Intent: {intent_result.intent_type} (confidence: {intent_result.confidence:.2f})")
        print(f"   Reasoning: {intent_result.reasoning}")

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        exit(1)
    
    print("Starting LangGraph Medical Agent Tests...")
    
    # Test individual components first
    test_individual_components()
    
    # Test the full agent
    test_medical_agent()
    
    print("\nAll tests completed!") 