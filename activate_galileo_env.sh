#!/bin/bash
# Activation script for the Galileo-compatible Python environment

echo "Activating Galileo Python environment..."
echo "Python version: $(python --version)"
echo "Virtual environment: galileo_env"
echo ""
echo "Environment is ready for Galileo logging!"
echo ""
echo "To run your application:"
echo "  python patient_chatbot.py          # For Chainlit web app"
echo "  python healthcare_agent.py         # For CLI healthcare agent"
echo "  python test_galileo_env.py         # To test the environment"
echo ""
echo "Make sure to set your API keys:"
echo "  export OPENAI_API_KEY='your_key_here'"
echo "  export GALILEO_API_KEY='your_key_here'"
echo "  export GALILEO_PROJECT='your_project'"
echo "  export GALILEO_LOG_STREAM='your_logstream'"
