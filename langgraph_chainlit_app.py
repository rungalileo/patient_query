"""
Chainlit Integration for LangGraph Medical Agent
Provides a web interface for the medical assistant with RAG Q&A and claim approval capabilities.
"""

import os
import json
import time
import logging
from typing import Dict, Any
from dotenv import load_dotenv

import chainlit as cl
from langgraph_agent import MedicalAgent

# Configure logging to suppress OpenAI HTTP requests
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Initialize the medical agent
agent = None

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session and medical agent."""
    global agent
    
    await cl.Message(
        content="🔄 Initializing Medical Agent... This may take a moment."
    ).send()
    
    try:
        # Initialize the medical agent
        agent = MedicalAgent()
        
        await cl.Message(
            content="✅ Medical Agent initialized successfully! I'm ready to help you with:\n\n"
                   "• Medical questions about symptoms, medications, and treatments\n"
                   "• Insurance claim approvals and coverage questions\n"
                   "• Both medical and insurance questions\n\n"
                   "Just ask me anything!"
        ).send()
        
        # Add action buttons after agent is initialized
        await add_action_buttons()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Error initializing Medical Agent: {str(e)}"
        ).send()
        return

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from the user."""
    global agent
    
    if not agent:
        await cl.Message(
            content="❌ Medical Agent not initialized. Please refresh the page and try again."
        ).send()
        return
    
    user_query = message.content
    start_time = time.time()
    
    try:
        # Process the query through the LangGraph agent
        result = agent.process_query(user_query)
        
        # Extract the response
        final_response = result.get("final_response", "I couldn't process your request.")
        
        # Create a rich response with metadata
        response_parts = []
        
        # Add the main response
        response_parts.append(final_response)
        
        # Add metadata if available
        metadata = result.get("metadata", {})
        if metadata:
            # Add processing time
            processing_time = metadata.get("processing_time", 0)
            if processing_time > 0:
                response_parts.append(f"\n\n⏱️ Processed in {processing_time:.2f} seconds")
            
            # Add intent classification info
            intent_info = metadata.get("intent_classification")
            if intent_info:
                intent_type = intent_info.get("intent_type", "unknown")
                confidence = intent_info.get("confidence", 0)
                response_parts.append(f"\n🎯 Intent: {intent_type.title()} (confidence: {confidence:.2f})")
            
            # Add tool usage info
            if metadata.get("rag_processed"):
                response_parts.append("\n📚 Used medical knowledge base")
            if metadata.get("claim_processed"):
                response_parts.append("\n💼 Processed claim approval")
        
        # Send the response
        await cl.Message(content="\n".join(response_parts)).send()
        
        # If there's an error, log it
        if result.get("error"):
            print(f"Error in processing: {result['error']}")
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
        await cl.Message(content=error_msg).send()
        print(f"Error in message processing: {e}")

@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end."""
    print("Chat session ended")

# Add some helpful commands
@cl.action_callback("show_tools")
async def show_tools(action):
    """Show available tools."""
    global agent
    
    if not agent:
        await cl.Message(content="❌ Medical Agent not available. Please wait for initialization to complete.").send()
        return
    
    tool_info = agent.get_tool_info()
    
    tool_description = "🔧 **Available Tools:**\n\n"
    for tool_name, info in tool_info.items():
        if "description" in info:
            tool_description += f"• **{tool_name}**: {info['description']}\n"
        else:
            tool_description += f"• **{tool_name}**: {info['capabilities']}\n"
    
    await cl.Message(content=tool_description).send()

@cl.action_callback("show_examples")
async def show_examples(action):
    """Show example queries."""
    examples = """📝 **Example Queries:**

**Medical Questions:**
• "What are the side effects of aspirin?"
• "Can you tell me about Atin Sanyal's medical history?"
• "What medication should I take for a headache?"
• "Are there any drug interactions with metformin?"

**Claim Approval Questions:**
• "Will insurance cover a $15,000 surgery for heart disease for a 45-year-old patient with private insurance?"
• "Is a $500 lab test approved for diabetes diagnosis for a 35-year-old with Medicare?"
• "Can I get approval for $2,000 therapy for depression for a 28-year-old patient?"

**Combined Questions:**
• "What medication should I take for my headache and will it be covered by insurance?"
• "Tell me about Sarah Johnson's condition and if her treatment will be approved"

Try asking any of these or your own questions!"""
    
    await cl.Message(content=examples).send()

# Add action buttons to the chat interface
async def add_action_buttons():
    """Add action buttons to the chat interface."""
    actions = [
        cl.Action(name="show_tools", label="🔧 Show Tools", description="View available tools", payload={"action": "show_tools"}),
        cl.Action(name="show_examples", label="📝 Show Examples", description="View example queries", payload={"action": "show_examples"})
    ]
    
    await cl.Message(
        content="Use the buttons below to explore the available tools and see example queries!",
        actions=actions
    ).send()

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        exit(1)
    
    print("LangGraph Medical Agent Chainlit App is ready!")
    print("Run with: chainlit run langgraph_chainlit_app.py") 