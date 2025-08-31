import os
import sys
import argparse
from dotenv import load_dotenv
from galileo import Message, MessageRole
from galileo.prompts import create_prompt_template
from galileo.experiments import run_experiment
import pandas as pd
from datetime import datetime
from galileo.datasets import get_dataset
import openai

load_dotenv()

def llm_call(input):
    """LLM function that will be tested in the experiment"""
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": "You are a world class stock market analyst."},
          {"role": "user", "content": f"{input}"}
        ],
    ).choices[0].message.content

def read_csv_data(csv_path):
    """Read CSV file and convert to dataset format"""
    print(f"Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV with {len(df)} rows")
        
        # Convert DataFrame to list of dictionaries
        data = []
        for index, row in df.iterrows():
            data_point = {}
            for column in df.columns:
                data_point[column] = str(row[column])
            data.append(data_point)
        
        print(f"Converted {len(data)} data points for dataset")
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def create_dataset_with_retry(dataset_name, content):
    """Create dataset with error handling for existing datasets"""
    try:
        print(f"Attempting to create dataset: {dataset_name}")
        dataset = create_dataset(
            name=dataset_name,
            content=content
        )
        print(f"Successfully created dataset: {dataset_name}")
        return dataset
    except Exception as e:
        if "dataset exists" in str(e).lower():
            print(f"Dataset '{dataset_name}' already exists. Creating with timestamp...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dataset_name = f"{dataset_name}_{timestamp}"
            print(f"Creating new dataset with name: {new_dataset_name}")
            dataset = create_dataset(
                name=new_dataset_name,
                content=content
            )
            print(f"Successfully created dataset: {new_dataset_name}")
            return dataset
        else:
            # Re-raise the exception if it's not a "dataset exists" error
            raise e

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Galileo experiment with CSV dataset')
    parser.add_argument('--project', required=True, help='Project name for the experiment')
    parser.add_argument('--experiment-name', required=True, help='Name of the experiment')
    parser.add_argument('--dataset-name', required=True, help='Name for the dataset')
    parser.add_argument('--csv-path', required=True, help='Path to the CSV file containing the dataset')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GALILEO EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Project: {args.project}")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"CSV Path: {args.csv_path}")
    print("=" * 60)
    
    try:
        # Step 1: Read CSV and create dataset
        print("\nSTEP 1: Reading CSV and creating dataset")
        print("-" * 40)
        
        # Read data from CSV
        test_data = read_csv_data(args.csv_path)
        
        # Create dataset with retry logic
        dataset = create_dataset_with_retry(args.dataset_name, test_data)
        
        # Step 2: Run experiment
        print("\nSTEP 2: Running experiment")
        print("-" * 40)
        print(f"Starting experiment: {args.experiment_name}")
        print(f"Using dataset: {dataset.name if hasattr(dataset, 'name') else 'Unknown'}")
        
        run_experiment(
            args.experiment_name,
            dataset=dataset,
            function=llm_call,
            prompt_settings={
                "max_tokens": 256,
                "model_alias": "GPT-4o",
                "temperature": 0.0
            },
            metrics=["correctness"],
            project=args.project
        )

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()