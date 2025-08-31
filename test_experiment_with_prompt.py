import os
import sys
import argparse
from dotenv import load_dotenv
from galileo import Message, MessageRole
from galileo.prompts import create_prompt_template
from galileo.experiments import run_experiment
from galileo.datasets import create_dataset
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

def execute_experiment(csv_path: str, system_prompt: str, project: str):
    print(f"\nRunning experiment in project: {project}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        print(f"\nColumns found in CSV: {list(df.columns)}")
        
        if "input" not in df.columns or "expected_output" not in df.columns:
            print("Error: CSV must contain both 'input' and 'expected_output' columns")
            return
            
        dataset = df['input'].tolist()
        print(f"Loaded {len(dataset)} examples from dataset")
        
    except Exception as e:
        print(f"Error loading dataset from {csv_path}: {e}")
        return
    
    prompt_template = create_prompt_template(
        name=f"experiment-prompt-{timestamp}",
        project=project,
        messages=[
            Message(role=MessageRole.system, content=system_prompt),
            Message(role=MessageRole.user, content="{input}")
        ]
    )

    try:
        results = run_experiment(
            "experiment",
            dataset=dataset,
            prompt_template=prompt_template,
            prompt_settings={
                "max_tokens": 256,
                "model_alias": "GPT-4o",
                "temperature": 0.0
            },
            metrics=["correctness"],
            project=project
        )
        print("\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Error running experiment: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run a Galileo experiment with a CSV dataset.')
    parser.add_argument('csv_path', help='Path to the CSV file containing the dataset')
    parser.add_argument('--system-prompt', default="You are a helpful assistant.", help='System prompt for the experiment')
    parser.add_argument('--project', default=os.getenv("GALILEO_PROJECT"), help='Galileo project name')
    args = parser.parse_args()
    
    if not args.project:
        print("Error: Project name must be provided either via --project argument or GALILEO_PROJECT environment variable")
        sys.exit(1)
    
    execute_experiment(args.csv_path, args.system_prompt, args.project)

if __name__ == "__main__":
    main()
