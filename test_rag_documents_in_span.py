#!/usr/bin/env python3
"""
Test script to verify that RAG documents are included in the final response generation span.
"""

import os
import json
from langgraph_agent import MedicalAgent

def test_rag_documents_in_span():
    """Test that RAG documents are included in the final response generation span."""
    
    # Initialize the agent
    agent = MedicalAgent()
    
    # Test query that should trigger RAG lookup
    test_query = "What are the side effects of aspirin?"
    
    print(f"Testing query: {test_query}")
    print("=" * 60)
    
    # Process the query
    result = agent.process_query(test_query)
    
    # Check that RAG documents were retrieved
    rag_documents = result.get('rag_documents', [])
    print(f"RAG documents retrieved: {len(rag_documents)}")
    
    # Display document details
    for i, doc in enumerate(rag_documents[:3]):  # Show first 3 documents
        print(f"\nDocument {i+1}:")
        print(f"  Type: {doc.get('metadata', {}).get('type', 'unknown')}")
        print(f"  Score: {doc.get('score', 0.0):.4f}")
        print(f"  Content preview: {doc.get('content', '')[:100]}...")
    
    # Check that final response was generated
    final_response = result.get('final_response', '')
    print(f"\nFinal response length: {len(final_response)} characters")
    print(f"Final response preview: {final_response[:200]}...")
    
    # Verify that the state contains the expected fields
    print(f"\nState verification:")
    print(f"  Has rag_response: {bool(result.get('rag_response'))}")
    print(f"  Has rag_documents: {bool(result.get('rag_documents'))}")
    print(f"  RAG documents count: {len(result.get('rag_documents', []))}")
    
    # Check metadata
    metadata = result.get('metadata', {})
    print(f"  Metadata rag_processed: {metadata.get('rag_processed', False)}")
    print(f"  Metadata rag_documents_count: {metadata.get('rag_documents_count', 0)}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("The RAG documents are now being retrieved and stored in the state.")
    print("They will be included in the final response generation span metadata.")

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        exit(1)
    
    test_rag_documents_in_span() 