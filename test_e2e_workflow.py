import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from galileo import GalileoLogger, Message, MessageRole
from galileo.datasets import create_dataset, get_dataset
from galileo.experiments import run_experiment
from galileo.prompts import create_prompt_template
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tools import get_ticker_symbol_tool, get_stock_price_tool
from colorama import Fore, Style

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)

def get_rag_response(query: str, namespace: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = pc.Index(index_name)
        
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace if namespace and namespace != "" else None,
            include_metadata=True
        )
        
        documents = [
            {
                "content": match.metadata.get("text", ""),
                "metadata": {
                    "score": match.score,
                    **match.metadata
                }
            }
            for match in query_response.matches
        ]
        
        return documents
        
    except Exception as e:
        logger.error(f"Error in RAG request: {str(e)}")
        return []

def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
        Executes a tool call and returns the result.
        This is a mock implementation of the tool call.
        In a real application, this would call the actual tool.
    """
    try:
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        if function_name == "getTickerSymbol":
            # Mock implementation - in real app, this would call the actual tool
            company = arguments["company"]
            return {"ticker": f"{company[:4].upper()}"}
            
        elif function_name == "getStockPrice":
            # Mock implementation
            ticker = arguments["ticker"]
            return {"price": 150.0, "ticker": ticker}
            
        elif function_name == "purchaseStocks":
            # Mock implementation
            return {
                "status": "success",
                "ticker": arguments["ticker"],
                "quantity": arguments["quantity"],
                "price": arguments["price"]
            }
            
        elif function_name == "sellStocks":
            # Mock implementation
            return {
                "status": "success",
                "ticker": arguments["ticker"],
                "quantity": arguments["quantity"],
                "price": arguments["price"]
            }
            
        else:
            raise ValueError(f"Unknown tool: {function_name}")
            
    except Exception as e:
        logger.error(f"Error executing tool call: {str(e)}")
        return {"error": str(e)}

def process_workflow(prompt: str, glog: GalileoLogger) -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        rag_documents = get_rag_response(prompt, "sp500-qa-demo", top_k=3)
        
        glog.add_retriever_span(
            input=prompt,
            output=[doc["content"] for doc in rag_documents],
            name="RAG Retriever",
            duration_ns=int((time.time() - start_time) * 1000000),
            metadata={
                "document_count": str(len(rag_documents)),
                "namespace": "sp500-qa-demo"
            }
        )
        
        context = "\n\n".join(doc["content"] for doc in rag_documents)
        messages = [
            {"role": "system", "content": "You are a stock market analyst and trading assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if context:
            messages.insert(1, {"role": "system", "content": f"Here is relevant context:\n\n{context}"})
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=[
                get_ticker_symbol_tool,
                get_stock_price_tool
            ]
        )
        
        response_message = response.choices[0].message
        
        glog.add_llm_span(
            input=messages,
            output={
                "role": response_message.role,
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments
                        }
                    } for call in (response_message.tool_calls or [])
                ] if response_message.tool_calls else None
            },
            model="gpt-4",
            name="Initial LLM Call",
            duration_ns=int((time.time() - start_time) * 1000000)
        )
        
        tool_results = []
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tool_result = execute_tool_call({
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                
                glog.add_tool_span(
                    input={
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    },
                    output=tool_result,
                    name=f"Tool: {tool_call.function.name}",
                    duration_ns=int((time.time() - start_time) * 1000000)
                )
                
                tool_results.append(tool_result)
        
        final_response = {
            "input": prompt,
            "rag_documents": rag_documents,
            "tool_calls": response_message.tool_calls,
            "tool_results": tool_results,
            "final_response": response_message.content
        }
        
        return final_response
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        raise

def create_golden_dataset(project: str, log_stream: str) -> None:
    """Create a dataset from high-quality traces."""
    
    test_prompts = [
        "What is the current stock price of Adobe?",
        "Tell me about Microsoft's recent earnings",
        "Should I buy Apple stock?",
        "What is the ticker symbol for Tesla?",
        "How has Amazon performed in the last quarter?"
    ]
    
    workflow_results = []
    for prompt in test_prompts:
        print("Running workflow for prompt: ", prompt)
        try:
            # Create a new logger instance for each workflow
            workflow_logger = GalileoLogger(project=project, log_stream=log_stream)
            
            # Start a new workflow for each prompt
            workflow_logger.start_trace(
                input=prompt,
                name=f"Workflow for: {prompt[:30]}...",
                tags=["workflow"]
            )
            
            start_time = time.time()
            result = process_workflow(prompt, workflow_logger)
            
            # End the trace using the logger
            workflow_logger.conclude(
                output=result,
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            
            workflow_results.append({
                "input": prompt,
                "output": result["final_response"],
                "metadata": {
                    "rag_documents": result["rag_documents"],
                    "tool_calls": result["tool_calls"],
                    "tool_results": result["tool_results"]
                }
            })
        except Exception as e:
            print(Fore.RED + f"ERROR processing prompt '{prompt}': {str(e)}" + Style.RESET_ALL)
            logger.error(f"ERROR processing prompt '{prompt}': {str(e)}")
            if 'workflow_logger' in locals():
                workflow_logger.conclude(
                    output={"error": str(e)},
                    duration_ns=int((time.time() - start_time) * 1000000),
                    status_code=500
                )
    
    dataset = create_dataset(
        name="my_golden_dataset",
        data=workflow_results,
        project=project
    )
    
    logger.info(f"Created dataset 'my_golden_dataset' with {len(workflow_results)} examples")

def run_experiments(project: str) -> None:
    """Run experiments with different models on the golden dataset."""
    dataset = get_dataset(
        name="my_golden_dataset",
        project=project
    )
    
    models = ["gpt-4o", "gpt-4o-mini"]
    
    for model in models:
        try:
            prompt_template = create_prompt_template(
                name=f"experiment-prompt-{model}",
                project=project,
                messages=[
                    Message(role=MessageRole.system, content="You are a stock market analyst and trading assistant."),
                    Message(role=MessageRole.user, content="{input}")
                ]
            )
            
            run_experiment(
                f"experiment-{model}",
                dataset=dataset,
                prompt_template=prompt_template,
                prompt_settings={
                    "max_tokens": 256,
                    "model_alias": model,
                    "temperature": 0.0
                },
                metrics=["correctness", "ground_truth_adherence"],
                project=project
            )
            
            logger.info(f"Completed experiment for {model}")
            
        except Exception as e:
            logger.error(f"Error running experiment for {model}: {str(e)}")

def main():
    project = os.getenv("GALILEO_PROJECT")
    log_stream = os.getenv("GALILEO_LOG_STREAM")

    if not all([project, log_stream]):
        logger.error("Missing required environment variables")
        sys.exit(1)
    
    try:
        logger.info("Creating golden dataset...")
        create_golden_dataset(project, log_stream)
        
        logger.info("Running experiments...")
        run_experiments(project)
        
        logger.info("Workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
