# LangGraph Medical Agent

A sophisticated agent-based medical assistant built with LangGraph that combines RAG (Retrieval-Augmented Generation) for medical Q&A and machine learning for insurance claim approval.

## 🏗️ Architecture Overview

The application is built as a modular LangGraph DAG (Directed Acyclic Graph) with the following components:

```
User Input → Intent Classifier → Router → [RAG Tool | Claim Tool | Both] → Response Combiner → Final Response
```

### Core Components

1. **Intent Classifier**: Determines whether the input is a Q&A request, claim approval request, or both
2. **RAG Tool**: Uses OpenAI embeddings and FAISS vector database for medical knowledge retrieval
3. **Claim Approval Tool**: Logistic regression classifier for insurance claim decisions
4. **LangGraph Orchestrator**: Manages the workflow and tool coordination

## 🚀 Features

### Medical Q&A (RAG)
- **Patient Record Search**: Query patient medical history, conditions, and medications
- **Medication Information**: Get details about drugs, side effects, and interactions
- **Symptom Analysis**: Find information about symptoms and recommended actions
- **FAISS Vector Database**: Fast similarity search using OpenAI embeddings

### Insurance Claim Approval
- **ML-Powered Decisions**: Logistic regression model trained on synthetic claim data
- **Business Logic**: Rule-based approval system with ML enhancement
- **Multiple Factors**: Considers age, cost, treatment type, diagnosis, and insurance type
- **Confidence Scoring**: Provides approval probability and confidence levels

### Intent Classification
- **Hybrid Approach**: Combines keyword matching with LLM-based classification
- **Multiple Intents**: Supports Q&A, claim approval, combined, and unknown intents
- **Confidence Scoring**: Provides confidence levels for classification decisions

## 📁 File Structure

```
patient-query/
├── langgraph_agent.py              # Main LangGraph orchestrator
├── rag_tool.py                     # RAG Q&A tool with FAISS
├── claim_approval_tool.py          # ML-based claim approval tool
├── intent_classifier.py            # Intent classification system
├── langgraph_chainlit_app.py       # Chainlit web interface
├── test_langgraph_agent.py         # Comprehensive test suite
├── mock_patient_data.py            # Mock medical data
├── patient_data_processor.py       # Original patient data processor
├── patient_chatbot.py              # Original Chainlit chatbot
└── requirements.txt                # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd patient-query
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env_template.txt .env
   # Edit .env and add your OpenAI API key
   ```

4. **Install additional dependencies** (if not in requirements.txt):
   ```bash
   pip install langgraph faiss-cpu scikit-learn pandas
   ```

## 🚀 Usage

### 1. Command Line Interface

Run the agent directly from the command line:

```bash
python langgraph_agent.py
```

Example interactions:
```
You: What are the side effects of aspirin?
Agent: [Medical information about aspirin side effects]

You: Will insurance cover a $15,000 surgery for heart disease for a 45-year-old with private insurance?
Agent: [Claim approval decision with confidence score]

You: What medication should I take for my headache and will it be covered?
Agent: [Combined medical advice and insurance information]
```

### 2. Chainlit Web Interface

Launch the web interface:

```bash
chainlit run langgraph_chainlit_app.py
```

Features:
- Interactive chat interface
- Action buttons for tools and examples
- Real-time processing feedback
- Rich response formatting

### 3. Programmatic Usage

```python
from langgraph_agent import MedicalAgent

# Initialize the agent
agent = MedicalAgent()

# Process a query
result = agent.process_query("What are the side effects of aspirin?")

# Access results
print(result['final_response'])
print(result['metadata']['intent_classification'])
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_langgraph_agent.py
```

The test suite includes:
- Individual component testing
- Full workflow testing
- Intent classification accuracy
- Performance benchmarking
- Error handling validation

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
GALILEO_API_KEY=your_galileo_api_key  # Optional
GALILEO_PROJECT=your_project_name      # Optional
GALILEO_LOG_STREAM=your_log_stream     # Optional
```

### Model Configuration

The application uses:
- **GPT-4o**: Main LLM for response generation
- **GPT-4o-mini**: Intent classification and information extraction
- **text-embedding-ada-002**: OpenAI embeddings for RAG

## 📊 Data Sources

### Mock Patient Data
- 4 sample patients with comprehensive medical records
- Includes medical history, medications, allergies, and lab results
- Realistic medical scenarios and conditions

### Medication Database
- Common medications with descriptions
- Contraindications and drug interactions
- Safety information and warnings

### Symptom Database
- Common symptoms and their causes
- Recommended actions and treatments
- Medical guidance for symptom management

## 🔄 Workflow

### 1. Intent Classification
```
Input: "What are the side effects of aspirin?"
↓
Intent: "qa" (confidence: 0.85)
```

### 2. Routing
```
Intent "qa" → RAG Tool
Intent "claim_approval" → Claim Tool
Intent "both" → Both Tools
Intent "unknown" → Help Response
```

### 3. Tool Execution
- **RAG Tool**: Searches medical knowledge base using FAISS
- **Claim Tool**: Runs ML model for approval decision
- **Combination**: Merges responses from multiple tools

### 4. Response Generation
- Combines tool outputs
- Adds metadata (processing time, confidence, etc.)
- Formats final response

## 🎯 Example Queries

### Medical Q&A
- "What are the side effects of aspirin?"
- "Can you tell me about Atin Sanyal's medical history?"
- "What medication should I take for a headache?"
- "Are there any drug interactions with metformin?"

### Claim Approval
- "Will insurance cover a $15,000 surgery for heart disease for a 45-year-old with private insurance?"
- "Is a $500 lab test approved for diabetes diagnosis for a 35-year-old with Medicare?"
- "Can I get approval for $2,000 therapy for depression for a 28-year-old patient?"

### Combined Queries
- "What medication should I take for my headache and will it be covered by insurance?"
- "Tell me about Sarah Johnson's condition and if her treatment will be approved"

## 🔍 Technical Details

### RAG Implementation
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Database**: FAISS with cosine similarity
- **Document Types**: Patient records, medications, symptoms
- **Search**: Top-k retrieval with relevance scoring

### Claim Approval Model
- **Algorithm**: Logistic Regression
- **Features**: Age, cost, treatment type, diagnosis, insurance type
- **Training Data**: 1000 synthetic claims with business logic
- **Accuracy**: ~85% on test set
- **Output**: Approval/denial with confidence score

### Intent Classification
- **Method**: Hybrid (keywords + LLM)
- **LLM**: GPT-4o-mini for classification
- **Keywords**: Domain-specific medical and insurance terms
- **Confidence**: Weighted combination of keyword and LLM scores

## 🚀 Performance

### Processing Times
- **Intent Classification**: ~1-2 seconds
- **RAG Search**: ~2-3 seconds
- **Claim Approval**: ~0.5-1 second
- **Total Response**: ~3-6 seconds

### Accuracy Metrics
- **Intent Classification**: ~90% accuracy
- **Claim Approval**: ~85% accuracy
- **RAG Relevance**: High relevance scores for medical queries

## 🔧 Customization

### Adding New Tools
1. Create a new tool class inheriting from `BaseTool`
2. Implement `_run()` and `_arun()` methods
3. Add the tool to the `MedicalAgent` constructor
4. Update the routing logic in `_route_request_node()`

### Modifying the Workflow
1. Edit the graph structure in `_build_graph()`
2. Add new nodes and edges as needed
3. Implement node functions following the existing pattern
4. Update the state schema if needed

### Extending the Knowledge Base
1. Add new documents to the mock data files
2. Update the document loading in `RAGTool`
3. Rebuild the FAISS index
4. Test with relevant queries

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Check API key in `.env` file
   - Verify API quota and billing
   - Ensure proper model access

2. **FAISS Installation Issues**
   - Use `faiss-cpu` for CPU-only systems
   - Install conda version for GPU support
   - Check system architecture compatibility

3. **Memory Issues**
   - Reduce batch sizes in embedding generation
   - Use smaller FAISS index types
   - Monitor memory usage during initialization

4. **Slow Performance**
   - Check internet connection for API calls
   - Consider caching embeddings
   - Optimize FAISS index parameters

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export DEBUG=1
export LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and embeddings
- LangChain for the framework
- FAISS for vector similarity search
- Scikit-learn for machine learning components
- Chainlit for the web interface

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the test examples
- Consult the documentation 