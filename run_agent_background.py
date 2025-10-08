#!/usr/bin/env python3
"""
Background Chainlit Application Runner
Loads queries from CSV and processes them through the LangGraph agent.
"""

import os
import csv
import time
import asyncio
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress OpenAI HTTP requests
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables
# 1) load global/shared first
load_dotenv(os.path.expanduser("~/.config/secrets/myapps.env"), override=False)
# 2) then load per-app .env (if present) to override selectively
load_dotenv(find_dotenv(usecwd=True), override=True)

def get_logstream_name() -> str:
    """Prompt user for logstream name."""
    print(f"{Fore.CYAN}Enter the logstream name for Galileo logging:{Style.RESET_ALL}")
    logstream_name = input().strip()
    
    if not logstream_name:
        print(f"{Fore.YELLOW}Warning: No logstream name provided. Using default 'background_test'{Style.RESET_ALL}")
        logstream_name = "background_test"
    
    return logstream_name

def load_utterances_from_csv(csv_file: str = "utterances.csv") -> List[Dict[str, Any]]:
    """Load utterances from CSV file."""
    utterances = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                utterances.append({
                    'text': row['text'].strip(),
                    'induce_prior_auth_error': row['induce_prior_auth_error'].lower() == 'true'
                })
        
        logger.info(f"Loaded {len(utterances)} utterances from {csv_file}")
        return utterances
    
    except FileNotFoundError:
        logger.error(f"CSV file '{csv_file}' not found!")
        return []
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return []

def setup_environment(logstream_name: str):
    """Set up environment variables for the background run."""
    # Set Galileo logstream
    os.environ["GALILEO_LOG_STREAM"] = logstream_name
    
    # Set other required environment variables if not already set
    if not os.getenv("GALILEO_API_KEY"):
        print(f"{Fore.YELLOW}Warning: GALILEO_API_KEY not set in environment{Style.RESET_ALL}")
    
    if not os.getenv("GALILEO_PROJECT"):
        print(f"{Fore.YELLOW}Warning: GALILEO_PROJECT not set in environment{Style.RESET_ALL}")
    
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.YELLOW}Warning: OPENAI_API_KEY not set in environment{Style.RESET_ALL}")

def process_utterance(agent, utterance: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single utterance through the LangGraph agent."""
    text = utterance['text']
    induce_error = utterance['induce_prior_auth_error']
    
    logger.info(f"Processing utterance: {text[:50]}...")
    logger.info(f"Induce prior auth error: {induce_error}")
    
    # Set the error flag for this specific utterance
    original_error_setting = os.getenv("INDUCE_PRIOR_AUTH_ERROR", "False")
    os.environ["INDUCE_PRIOR_AUTH_ERROR"] = str(induce_error).lower()
    
    try:
        # Process the query
        start_time = time.time()
        result = agent.process_query(text)
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'utterance': text,
            'induce_prior_auth_error': induce_error,
            'final_response': result.get('final_response', 'No response generated'),
            'processing_time': processing_time,
            'error': result.get('error'),
            'metadata': result.get('metadata', {})
        }
        
        # Log the result
        if result.get('error'):
            logger.error(f"Error processing utterance: {result['error']}")
        else:
            logger.info(f"Successfully processed utterance in {processing_time:.2f}s")
        
        return response
    
    except Exception as e:
        logger.error(f"Exception processing utterance: {e}")
        return {
            'utterance': text,
            'induce_prior_auth_error': induce_error,
            'final_response': f"Error: {str(e)}",
            'processing_time': 0,
            'error': str(e),
            'metadata': {}
        }
    
    finally:
        # Restore original error setting
        os.environ["INDUCE_PRIOR_AUTH_ERROR"] = original_error_setting

def save_results(results: List[Dict[str, Any]], logstream_name: str):
    """Save processing results to a file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"background_results_{logstream_name}_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"Background Chainlit Processing Results\n")
            file.write(f"Logstream: {logstream_name}\n")
            file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Total utterances: {len(results)}\n")
            file.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                file.write(f"Utterance {i}:\n")
                file.write(f"Text: {result['utterance']}\n")
                file.write(f"Induce Error: {result['induce_prior_auth_error']}\n")
                file.write(f"Processing Time: {result['processing_time']:.2f}s\n")
                
                if result.get('error'):
                    file.write(f"Error: {result['error']}\n")
                
                file.write(f"Response: {result['final_response']}\n")
                
                # Add metadata if available
                metadata = result.get('metadata', {})
                if metadata:
                    file.write("Metadata:\n")
                    for key, value in metadata.items():
                        file.write(f"  {key}: {value}\n")
                
                file.write("-" * 80 + "\n\n")
        
        logger.info(f"Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main function to run the background Chainlit application."""
    print(f"{Fore.GREEN}Background Chainlit Application Runner{Style.RESET_ALL}")
    print("=" * 50)
    
    # Get logstream name from user
    logstream_name = get_logstream_name()
    
    # Set up environment
    setup_environment(logstream_name)
    
    # Load utterances from CSV
    utterances = load_utterances_from_csv()
    
    if not utterances:
        print(f"{Fore.RED}No utterances loaded. Exiting.{Style.RESET_ALL}")
        return
    
    # Initialize the medical agent
    print(f"{Fore.CYAN}Initializing Medical Agent...{Style.RESET_ALL}")
    try:
        from langgraph_agent import MedicalAgent
        agent = MedicalAgent()
        print(f"{Fore.GREEN}Medical Agent initialized successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error initializing Medical Agent: {e}{Style.RESET_ALL}")
        return
    
    # Process each utterance
    print(f"\n{Fore.CYAN}Processing {len(utterances)} utterances...{Style.RESET_ALL}")
    print("-" * 50)
    
    results = []
    
    for i, utterance in enumerate(utterances, 1):
        print(f"\n{Fore.YELLOW}[{i}/{len(utterances)}] Processing utterance...{Style.RESET_ALL}")
        
        result = process_utterance(agent, utterance)
        results.append(result)
        
        # Print summary
        if result.get('error'):
            print(f"{Fore.RED}❌ Error: {result['error']}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✅ Success ({result['processing_time']:.2f}s){Style.RESET_ALL}")
        
        # Add a small delay between requests
        time.sleep(1)
    
    # Save results
    print(f"\n{Fore.CYAN}Saving results...{Style.RESET_ALL}")
    save_results(results, logstream_name)
    
    # Print summary
    print(f"\n{Fore.GREEN}Processing complete!{Style.RESET_ALL}")
    print(f"Total utterances processed: {len(results)}")
    
    successful = sum(1 for r in results if not r.get('error'))
    failed = len(results) - successful
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n{Fore.YELLOW}Failed utterances:{Style.RESET_ALL}")
        for result in results:
            if result.get('error'):
                print(f"  - {result['utterance'][:50]}... (Error: {result['error']})")

if __name__ == "__main__":
    main() 