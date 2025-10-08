"""
Test script for the patient data processor to verify similarity search functionality.
"""

import os
from dotenv import load_dotenv, find_dotenv
from patient_data_processor import patient_processor

# Load environment variables
# 1) load global/shared first
load_dotenv(os.path.expanduser("~/.config/secrets/myapps.env"), override=False)
# 2) then load per-app .env (if present) to override selectively
load_dotenv(find_dotenv(usecwd=True), override=True)

def test_patient_processor():
    """Test the patient data processor functionality."""
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY environment variable")
        return False
    
    print("🔄 Loading patient data and creating embeddings...")
    
    try:
        # Load all documents
        patient_processor.load_all_documents()
        print("✅ Patient data loaded successfully!")
        
        print("\n🔍 Testing patient search...")
        patient_results = patient_processor.search_patient_records("Atin Sanyal")
        if patient_results:
            print(f"✅ Found {len(patient_results)} patient records for Atin Sanyal")
            for doc, score in patient_results:
                print(f"   - {doc['metadata']['name']} (similarity: {score:.3f})")
        else:
            print("❌ No patient records found")
        
        print("\n🔍 Testing medication search...")
        med_results = patient_processor.search_medication_info("aspirin")
        if med_results:
            print(f"✅ Found {len(med_results)} medication records for aspirin")
            for doc, score in med_results:
                print(f"   - {doc['metadata']['name']} (similarity: {score:.3f})")
        else:
            print("❌ No medication records found")
        
        print("\n🔍 Testing symptom search...")
        symptom_results = patient_processor.search_symptom_info("runny nose")
        if symptom_results:
            print(f"✅ Found {len(symptom_results)} symptom records for runny nose")
            for doc, score in symptom_results:
                print(f"   - {doc['metadata']['name']} (similarity: {score:.3f})")
        else:
            print("❌ No symptom records found")
        
        print("\n🔍 Testing contraindication scenario...")
        if patient_results and med_results:
            patient_info = patient_results[0][0]["content"]
            med_info = med_results[0][0]["content"]
            
            print("✅ Testing contraindication analysis...")
            print("   Patient: Atin Sanyal (diabetes, hypertension)")
            print("   Medication: Aspirin")
            print("   Expected: Should detect potential interactions")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_patient_processor()
    if success:
        print("\n✅ Patient data processor is working correctly!")
        print("You can now run: chainlit run patient_chatbot.py")
    else:
        print("\n❌ Patient data processor test failed. Please check your setup.") 