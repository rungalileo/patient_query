# Patient Query Healthcare Agent

A medical assistant built with LangGraph and Langchain that provides personalized health advice based on patient records using OpenAI embeddings and FAISS vector search.

## Demo Scenario

The agent demonstrates how it can prevent patients from taking contraindicated medications. For example:

- **Patient**: "I am Atin Sanyal. I'm having a runny nose, should I take aspirin?"
- **Agent**: Analyzes Atin's medical history (diabetes, hypertension, allergies) and warns against aspirin due to potential interactions with his current medications (Lisinopril, Metformin)

## Prerequisites

- OpenAI API key
- Galileo API key for logging

## Installation

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
   Create a `.env` file based on `env_template.txt`:
   ```bash
   cp env_template.txt .env
   ```
   
   Fill in your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GALILEO_PROJECT=xyz
   GALILEO_LOG_STREAM=abc
   GALILEO_API_KEY=your_galileo_key
   GALILEO_CONSOLE_URL=your_galileo_url
   ```

## Setup Steps

### Run the Healthcare Agent

Run the healthcare agent:

```bash
python healthcare_agent.py --project PROJECT_NAME --logstream LOGSTREAM_NAME
```

**Optional Arguments:**
- `--project PROJECT_NAME`: Override the Galileo project in .env
- `--logstream LOGSTREAM_NAME`: Override the Galileo logstream name in .env


**Features of the Healthcare Agent:**
- **Interactive Terminal Interface**: Chat directly in your terminal
- **Advanced Workflow**: Uses LangGraph for intelligent request routing
- **Multi-Tool Integration**: Combines RAG, claim approval, and prior authorization
- **Intent Classification**: Automatically determines what type of help you need
- **Galileo Logging**: Optional integration for monitoring and analytics

**Extra .env options:**
```bash
INDUCE_PRIOR_AUTH_ERROR=True
SIMULATE_SLOW_CLAIM_PROCESSING=True
```

The healthcare agent provides medical advice with safety features, insurance claim processing, and prior authorization capabilities.

## Usage

1. **Run the agent** with `python healthcare_agent.py --project PROJECT --logstream LOGSTREAM`
2. **Wait for initialization** - embeddings and models will load
3. **Ask questions** about symptoms, medications, claims, or prior authorization
4. **Review the personalized advice** and any safety warnings

### Example Queries
- "I am Atin Sanyal. I need approval for a $15,000 heart surgery due to heart disease."
- "Sarah Johnson needs medication coverage for asthma treatment costing $500."
- "Can you check if Michael Chen's imaging procedure requires prior authorization?"
- "I'm Emily Rodriguez. I have depression symptoms and need therapy coverage."

## Mock Data

The application includes comprehensive mock data:

### Patient Records
- **Atin Sanyal**: 35-year-old male with diabetes, hypertension, and allergies
- **Sarah Johnson**: 28-year-old female with asthma, migraines, and anxiety
- **Michael Chen**: 45-year-old male with heart disease and sleep apnea
- **Emily Rodriguez**: 32-year-old female with hypothyroidism and depression

### Medications Database
- Common medications with contraindications and interactions
- Examples: Aspirin, Ibuprofen, Acetaminophen, Amoxicillin, etc.

### Symptoms Database
- Common symptoms with causes and recommended actions
- Examples: Runny nose, fever, headache, cough

## How It Works

1. **Data Loading**: On startup, creates embeddings for all patient records, medications, and symptoms
2. **Query Processing**: Extracts patient name, symptoms, and medications from user input
3. **Similarity Search**: Uses OpenAI embeddings and cosine similarity to find relevant information
4. **Contraindication Analysis**: Uses OpenAI to analyze potential drug interactions
5. **Personalized Response**: Generates tailored medical advice with safety warnings

## Architecture

The system follows a similar pattern to `pdf_reader.py`:

- **PatientDataProcessor**: Handles document creation, embedding generation, and similarity search
- **In-Memory Storage**: All embeddings and documents stored in memory for fast access
- **Cosine Similarity**: Uses numpy for efficient similarity calculations
- **OpenAI Embeddings**: Leverages text-embedding-ada-002 for semantic search

## Safety Features

- **Contraindication Detection**: Automatically identifies potential drug interactions
- **Medical Disclaimers**: Always recommends consulting healthcare providers
- **Safety Warnings**: Prominent warnings for potentially dangerous combinations
- **Evidence-Based**: Uses established medical databases and guidelines

## Future Enhancements

### Phase 2: Agentic Claims Processing
- Integration with mock insurance claims API
- Automatic claim submission and approval/denial responses
- Real-time benefit verification
- Prior authorization workflows

### Additional Features
- Multi-language support
- Voice interface
- Integration with EHR systems
- Advanced symptom analysis
- Prescription refill requests
- Persistent storage for embeddings (optional)

## Performance Notes

- **Initial Load Time**: Creating embeddings for all documents takes ~30-60 seconds on first startup
- **Memory Usage**: All embeddings stored in memory for fast similarity search
- **Scalability**: For production use, consider persistent storage solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and demonstration purposes only. Medical advice should always be obtained from qualified healthcare providers.

## Disclaimer

⚠️ **This is a demonstration application and should not be used for actual medical decisions. Always consult with qualified healthcare providers for medical advice.** 