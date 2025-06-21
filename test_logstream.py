import os
import sys
import argparse
from dotenv import load_dotenv
from galileo import GalileoLogger
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_galileo_project():
    """Get the current Galileo project name."""
    return os.getenv("GALILEO_PROJECT")

def log_utterances_to_stream(log_stream_name: str):
    """Read utterances from CSV and log them to the specified log stream."""
    # Get required environment variables
    api_key = os.getenv("GALILEO_API_KEY")
    project_name = get_galileo_project()
    
    if not all([api_key, project_name, log_stream_name]):
        raise ValueError("Missing required environment variables")
    
    print(f"\nWriting to Project: {project_name}")
    print(f"Writing to Log Stream: {log_stream_name}")
    
    # Initialize Galileo logger
    glog = GalileoLogger(project=project_name, log_stream=log_stream_name)
    
    try:
        # Read utterances from CSV
        with open('utterances.csv', 'r') as f:
            # Skip header and read all lines
            utterances = [line.strip() for line in f.readlines()[1:]]
        
        print(f"\nLoaded {len(utterances)} utterances from CSV")
        
        # Process each utterance
        for i, utterance in enumerate(utterances, 1):
            start_time = time.time()
            
            # Start a new trace for each utterance
            glog.start_trace(
                input=utterance,
                name=f"Utterance {i}",
                tags=["utterance"]
            )
            
            # Log the utterance using add_llm_span
            glog.add_llm_span(
                input=utterance,
                output=f"Processed utterance {i}",
                name="Utterance Processing",
                model="manual",
                metadata={
                    "source": "test_logstream.py",
                    "type": "utterance",
                    "utterance_number": i
                }
            )
            
            # Conclude the trace
            glog.conclude(
                output=f"Processed utterance {i}",
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            
            # Flush after each utterance to ensure it's sent
            glog.flush()
            
            print(f"Logged utterance {i}: {utterance[:50]}...")
            
    except Exception as e:
        logger.error(f"Error processing utterances: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Log utterances to a Galileo log stream.')
    parser.add_argument('--logstream', required=True, help='Name of the log stream to write to')
    args = parser.parse_args()
    
    try:
        log_utterances_to_stream(args.logstream)
        print("\nSuccessfully logged all utterances!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 