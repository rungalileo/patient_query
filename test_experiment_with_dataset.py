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
from galileo.datasets import get_dataset
import openai

load_dotenv()

def llm_call(input):
	return openai.chat.completions.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": "You are a world class stock market analyst."},
          {"role": "user", "content": f"{input}"}
        ],
    ).choices[0].message.content


def main():
    try:
        test_data = [
             {
                  "input": "trade a 100 shares of ebay",
                  "output": "I have successfully executed the trade. You have purchased 100 shares of Ebay (EBAY.MOCK) at a price of $100.00 per share. The total cost of the trade is $10,000.00, plus fees of $10.00. The total cost of the transaction, including fees, is $10,010.00. The order ID is ORD-626500."
             },
             {
                  "input": "how's ebay stock doing? trade a 100 shares if it's 1% lower than yesterday?",
                  "output": "The current stock price for eBay (EBAY.MOCK) is $100.0. There has been no price change from the previous trading day. Therefore, since the price is not 1% lower than yesterday, I will not execute the trade of 100 shares as per your instructions."
             },
             {
                  "input": "how's the apple stock doing? if it's doing well, buy 10 shares.",
                  "output": "I'm sorry, but I don't have the required information to assess the current performance of Apple's stock. I can make use of the getStockPrice function to fetch the current stock price and other market data for Apple. Would you like me to proceed with that?"
             },
        ]

        dataset = create_dataset(
            name="shares-test",
            content=test_data
        )

        run_experiment(
            "experiment-shares",
            dataset=dataset,
            function=llm_call,
            prompt_settings={
                "max_tokens": 256,
                "model_alias": "GPT-4o",
                "temperature": 0.0
            },
            metrics=["correctness", "ground_truth_adherence"],
            project="financial-chat-demo"
        )

        print("\nExperiment completed successfully!")
    
    except Exception as e:
        print(f"Error running experiment: {e}")



if __name__ == "__main__":
    main()