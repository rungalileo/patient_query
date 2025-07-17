#!/usr/bin/env python3
"""
Test script to demonstrate the Medical Agent's capabilities
"""

from langgraph_agent import MedicalAgent
import time

def test_pure_qa_scenario():
    """Test a pure Q&A scenario with a patient in the database."""
    print("=" * 60)
    print("TEST 1: Pure Q&A Scenario")
    print("=" * 60)
    
    agent = MedicalAgent()
    
    # Test with Sarah Johnson who has detailed medical data
    query = "What are Sarah Johnson's current medications and symptoms?"
    
    print(f"Query: {query}")
    print("-" * 40)
    
    start_time = time.time()
    result = agent.process_query(query)
    end_time = time.time()
    
    print(f"Response:\n{result['final_response']}")
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    
    # Show workflow path
    intent_result = result.get('intent_result')
    if intent_result:
        print(f"\nIntent: {intent_result.intent_type} (confidence: {intent_result.confidence:.2f})")
    
    print(f"RAG Response: {'Yes' if result.get('rag_response') else 'No'}")
    print(f"Claim Response: {'Yes' if result.get('claim_response') else 'No'}")
    print(f"Prior Auth Response: {'Yes' if result.get('prior_auth_response') else 'No'}")

def test_claim_approval_scenario():
    """Test a claim approval + prior authorization scenario."""
    print("\n" + "=" * 60)
    print("TEST 2: Claim Approval + Prior Authorization Scenario")
    print("=" * 60)
    
    agent = MedicalAgent()
    
    # Test with a new patient that doesn't exist in the database
    # This will allow the claim approval tool to process the request
    query = "I need to file a claim for John Davis's heart surgery. The cost is $45,000 and he has private insurance. He has diabetes and is 52 years old."
    
    print(f"Query: {query}")
    print("-" * 40)
    
    start_time = time.time()
    result = agent.process_query(query)
    end_time = time.time()
    
    print(f"Response:\n{result['final_response']}")
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    
    # Show workflow path
    intent_result = result.get('intent_result')
    if intent_result:
        print(f"\nIntent: {intent_result.intent_type} (confidence: {intent_result.confidence:.2f})")
    
    print(f"RAG Response: {'Yes' if result.get('rag_response') else 'No'}")
    print(f"Claim Response: {'Yes' if result.get('claim_response') else 'No'}")
    print(f"Prior Auth Response: {'Yes' if result.get('prior_auth_response') else 'No'}")
    
    # Show more detailed workflow information
    if result.get('metadata'):
        metadata = result['metadata']
        print(f"\nWorkflow Details:")
        print(f"- Intent Classification: {metadata.get('intent_classification', {}).get('intent_type', 'N/A')}")
        print(f"- RAG Processed: {metadata.get('rag_processed', False)}")
        print(f"- Claim Processed: {metadata.get('claim_processed', False)}")
        print(f"- Prior Auth Processed: {metadata.get('prior_auth_processed', False)}")
        if metadata.get('claim_info'):
            print(f"- Claim Info Extracted: Yes")
        if metadata.get('used_rag_info'):
            print(f"- Used RAG Info: {metadata.get('used_rag_info', False)}")

def main():
    """Run both test scenarios."""
    print("Medical Agent Test Suite")
    print("Testing two different scenarios...")
    
    try:
        # Test 1: Pure Q&A
        test_pure_qa_scenario()
        
        # Test 2: Claim Approval + Prior Auth
        test_claim_approval_scenario()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 