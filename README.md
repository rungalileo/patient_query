# Patient Query Q&A Chatbot

A medical assistant chatbot built with Chainlit and Langchain that provides personalized health advice based on patient records using OpenAI embeddings and in-memory similarity search.

## Features

- **Personalized Medical Advice**: Get recommendations based on your medical history
- **Medication Safety Checks**: Automatic contraindication detection
- **Symptom Analysis**: Information about common symptoms and treatments
- **In-Memory Vector Search**: Fast retrieval using OpenAI embeddings and cosine similarity
- **Real-time Streaming**: Interactive chat experience with streaming responses

## Demo Scenario

The chatbot demonstrates how it can prevent patients from taking contraindicated medications. For example:

- **Patient**: "I am Atin Sanyal. I'm having a runny nose, should I take aspirin?"
- **Chatbot**: Analyzes Atin's medical history (diabetes, hypertension, allergies) and warns against aspirin due to potential interactions with his current medications (Lisinopril, Metformin)

## Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) Galileo API key for logging

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
   ```

## Setup Steps

### Step 1: Run the Chatbot

Start the Chainlit application:

```bash
chainlit run patient_chatbot.py
```

The chatbot will automatically load patient data and create embeddings on startup. The chatbot will be available at `http://localhost:8000`

## Usage

1. **Open the chatbot** in your browser
2. **Wait for data loading** - you'll see a message when embeddings are ready
3. **Introduce yourself** with your name and health concern
4. **Ask questions** about symptoms, medications, or general health
5. **Review the personalized advice** and any safety warnings

### Example Queries

- "I am Atin Sanyal. I'm having a runny nose, should I take aspirin?"
- "I'm Sarah Johnson and I have a headache. Can I take ibuprofen?"
- "I'm Michael Chen. I have chest pain, what should I do?"

## Project Structure

```
patient-query/
├── patient_chatbot.py          # Main Chainlit application
├── patient_data_processor.py   # Data processor with embeddings and similarity search
├── mock_patient_data.py        # Mock patient records and medical data
├── pdf_reader.py               # Reference PDF processing utility
├── requirements.txt            # Python dependencies
├── chainlit.md                 # Chainlit UI configuration
├── env_template.txt            # Environment variables template
└── README.md                   # This file
```

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
6. **Streaming Output**: Provides real-time responses with Chainlit's streaming interface

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