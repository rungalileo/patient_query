import os
from dotenv import load_dotenv
from galileo import GalileoLogger
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment
from galileo import galileo_context
from openai import OpenAI
import time
import datetime
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_galileo_project():
    """Get the current Galileo project name."""
    return os.getenv("GALILEO_PROJECT")

def set_galileo_project(project_name: str):
    """Set the Galileo project name."""
    os.environ["GALILEO_PROJECT"] = project_name

def log_to_a_logstream(prompt: str, model: str, log_stream_name: str):
    """Make an OpenAI API call and log the interaction to a log stream."""
    api_key = os.getenv("GALILEO_API_KEY")
    project_name = get_galileo_project()
    
    if not all([api_key, project_name, log_stream_name]):
        raise ValueError("Missing required environment variables")
    
    print(f"\nWriting to Project: {project_name}")
    print(f"Writing to Log Stream: {log_stream_name}")
    
    # Initialize OpenAI client
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize Galileo logger
    logger = GalileoLogger(project=project_name, log_stream=log_stream_name)
    
    start_time = time.time()
    
    # Start a new trace
    logger.start_trace(
        input=prompt,
        name="OpenAI Chat",
        tags=["chat"]
    )
    
    # Make the OpenAI API call
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    response_content = response.choices[0].message.content
    
    logger.add_llm_span(
        input=prompt,
        output=response_content,
        name="OpenAI Chat",
        model=model,
        metadata={
            "source": "test_sdk.py",
            "type": "chat"
        }
    )
    
    logger.conclude(
        output=response_content,
        duration_ns=int((time.time() - start_time) * 1000000),
        status_code=200
    )
    
    logger.flush()
    
    print(f"\nUser: {prompt}")
    print(f"\nAssistant: {response_content}")

def process_experiment_prompt(example):
    try:
        return input_output_map.get(example, '')
        
    except Exception as e:
        logger.error(f"Error processing example: {e}")
        return f"Error: {str(e)}"

def log_an_experiment(experiment_name: str, dataset_path: str):
    required_vars = ["GALILEO_API_KEY", "GALILEO_PROJECT", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    galileo_project = get_galileo_project()
    print(f"\nCreating experiment in project: {galileo_project}")
    
    try:
        try:
            df = pd.read_csv(dataset_path)
            df.columns = df.columns.str.strip()
            
            print(f"\nColumns found in CSV: {list(df.columns)}")
            
            if "input" not in df.columns or "expected_output" not in df.columns:
                logger.error("CSV must contain both 'input' and 'expected_output' columns")
                return

            global input_output_map
            input_output_map = dict(zip(df['input'], df['expected_output']))
            
            dataset = df['input'].tolist()
            print(f"Loaded {len(dataset)} examples from dataset")
            
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return
        
        logger.info(f"Starting experiment: {experiment_name}")
        
        run_experiment(
            experiment_name,
            dataset=dataset,
            function=process_experiment_prompt,
            metrics=[
                "correctness"
            ],
            project=galileo_project
        )
        
        logger.info(f"Experiment completed: {experiment_name}")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")

def main():
    while True:
        print("\n=== Galileo SDK Test Interface ===")
        print(f"Current Project: {get_galileo_project()}")
        print("\n1. Log to a log stream")
        print("2. Log an experiment")
        print("3. Change Galileo Project")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4":
            print("Goodbye!")
            break
            
        elif choice == "1":
            log_stream_name = input("\nEnter log stream name: ").strip()
            
            print("\nAvailable models:")
            print("1. gpt-4o (OpenAI)")
            print("2. gpt-4o-mini (OpenAI)")
            
            model_choice = input("\nSelect model (1-2): ").strip()
            model_map = {
                "1": "gpt-4o",
                "2": "gpt-4o-mini"
            }
            
            if model_choice not in model_map:
                print("Invalid model choice!")
                continue
                
            model = model_map[model_choice]
            
            print("\nEnter your prompts (type 'exit' to return to main menu):")
            while True:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() == "exit":
                    break
                    
                try:
                    log_to_a_logstream(prompt, model, log_stream_name)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
        elif choice == "2":
            experiment_name = input("\nEnter experiment name: ").strip()
            dataset_path = input("Enter full path to CSV dataset: ").strip()
            
            try:
                log_an_experiment(experiment_name, dataset_path)
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == "3":
            new_project = input("\nEnter new Galileo project name: ").strip()
            if new_project:
                set_galileo_project(new_project)
                print(f"Project changed to: {new_project}")
            else:
                print("Project name cannot be empty!")
                
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
