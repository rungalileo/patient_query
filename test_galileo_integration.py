import os
import time
import csv
import random
import argparse
from dotenv import load_dotenv
from langgraph_agent import MedicalAgent

# Load environment variables
load_dotenv()

def load_utterances_from_csv(csv_file="utterances.csv"):
    """Load utterances from CSV file"""
    utterances = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                utterances.append({
                    'text': row['text'],
                    'induce_prior_auth_error': row['induce_prior_auth_error'].lower() == 'true'
                })
        return utterances
    except FileNotFoundError:
        print(f"❌ Error: {csv_file} not found")
        return []
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")
        return []

def test_galileo_integration(production_mode=False):
    # Check if Galileo is configured
    api_key = os.getenv("GALILEO_API_KEY")
    project = os.getenv("GALILEO_PROJECT")
    log_stream = os.getenv("GALILEO_LOG_STREAM")
    
    print("Testing Galileo Integration")
    print("=" * 50)
    print(f"Mode: {'Production' if production_mode else 'Test'}")
    print(f"Galileo API Key: {'Set' if api_key else 'Not set'}")
    print(f"Galileo Project: {project}")
    print(f"Galileo Log Stream: {log_stream}")
    print()
    
    if not all([api_key, project, log_stream]):
        print("❌ Galileo not configured. Please set GALILEO_API_KEY, GALILEO_PROJECT, and GALILEO_LOG_STREAM in your .env file")
        return
    
    # Load utterances from CSV
    utterances = load_utterances_from_csv()
    if not utterances:
        print("❌ No utterances loaded from CSV. Exiting.")
        return
    
    print(f"✅ Loaded {len(utterances)} utterances from CSV")
    
    try:
        # Initialize the medical agent
        print("🔄 Initializing Medical Agent...")
        agent = MedicalAgent()
        print("✅ Medical Agent initialized successfully!")
        
        if production_mode:
            print("\n🚀 Starting production mode - running in infinite loop...")
            print("Press Ctrl+C to stop")
            print()
            
            iteration = 1
            while True:
                # Pick a random utterance
                utterance = random.choice(utterances)
                
                print(f"\n{'='*20} Iteration {iteration} {'='*20}")
                print(f"Query: {utterance['text']}")
                print(f"Induce Prior Auth Error: {utterance['induce_prior_auth_error']}")
                print("-" * 60)
            
                start_time = time.time()
                result = agent.process_query(utterance['text'])
                processing_time = time.time() - start_time
                
                print(f"Response: {result.get('final_response', 'No response')[:200]}...")
                print(f"Processing Time: {processing_time:.2f} seconds")
                
                # Check metadata
                metadata = result.get("metadata", {})
                if metadata:
                    print("Metadata:")
                    for key, value in metadata.items():
                        if key != "start_time":
                            print(f"  {key}: {value}")
                
                # Check for errors
                if result.get("error"):
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Query processed successfully")
                
                print(f"\n⏳ Waiting 4 seconds before next iteration...")
                time.sleep(4)
                iteration += 1
                
        else:
            # Test mode - pick 3 random utterances
            selected_utterances = random.sample(utterances, min(3, len(utterances)))
            
            for i, utterance in enumerate(selected_utterances, 1):
                print(f"\n{'='*20} Test Query {i} {'='*20}")
                print(f"Query: {utterance['text']}")
                print(f"Induce Prior Auth Error: {utterance['induce_prior_auth_error']}")
                print("-" * 60)
                
                start_time = time.time()
                result = agent.process_query(utterance['text'])
            processing_time = time.time() - start_time
            
            print(f"Response: {result.get('final_response', 'No response')[:200]}...")
            print(f"Processing Time: {processing_time:.2f} seconds")
            
            # Check metadata
            metadata = result.get("metadata", {})
            if metadata:
                print("Metadata:")
                for key, value in metadata.items():
                    if key != "start_time":
                        print(f"  {key}: {value}")
            
            # Check for errors
            if result.get("error"):
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Query processed successfully")
            
            print()
        
        print("🎉 All tests completed! Check your Galileo dashboard for traces.")
        
    except KeyboardInterrupt:
        if production_mode:
            print("\n\n🛑 Production mode stopped by user (Ctrl+C)")
        else:
            raise
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test Galileo Integration')
    parser.add_argument('--production', action='store_true', 
                       help='Run in production mode (infinite loop with 4-second intervals)')
    
    args = parser.parse_args()
    test_galileo_integration(production_mode=args.production)

if __name__ == "__main__":
    main()