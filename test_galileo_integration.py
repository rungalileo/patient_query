"""
Test script to verify Galileo integration with the patient chatbot.
"""

import os
from dotenv import load_dotenv
from galileo import GalileoLogger

# Load environment variables
load_dotenv()

def test_galileo_connection():
    """Test Galileo connection and basic logging."""
    
    # Check environment variables
    api_key = os.getenv("GALILEO_API_KEY")
    project = os.getenv("GALILEO_PROJECT")
    log_stream = os.getenv("GALILEO_LOG_STREAM")

    try:
        # Initialize Galileo logger
        print(f"Project: {project}")
        print(f"Log Stream: {log_stream}")
        
        glog = GalileoLogger(project=project, log_stream=log_stream)
        
        # Test a simple log
        glog.start_trace(
            input="Test patient query",
            name="Galileo Integration Test",
            tags=["test", "integration"]
        )
        
        glog.add_llm_span(
            input="Test input",
            output="Test output",
            name="Test LLM Call",
            model="test-model",
            metadata={
                "source": "test_galileo_integration.py",
                "type": "test"
            }
        )
        
        glog.conclude(
            output="Test completed successfully",
            duration_ns=1000000,
            status_code=200
        )
        
        glog.flush()
        
        print("✅ Galileo integration test completed successfully!")
        print("✅ You can now run the patient chatbot with Galileo logging enabled.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Galileo integration: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_galileo_connection()
    if success:
        print("\n🎉 Galileo integration is ready!")
        print("Run: chainlit run patient_chatbot.py")
    else:
        print("\n❌ Please fix Galileo configuration and try again.") 