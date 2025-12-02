# RAG ChatBot - PDF Question Answering System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based exclusively on the content of a PDF document. Built with Streamlit, LangChain, and Groq API.

## 🎯 What Does This Agent Do?

This chatbot:
- **Reads and understands PDF documents** using advanced text processing
- **Answers questions ONLY from the PDF content** - it won't use outside knowledge
- **Retrieves relevant sections** from the document to provide accurate answers
- **Maintains conversation history** for a seamless chat experience
- **Refuses to answer** questions not covered in the PDF to prevent hallucinations

## ✨ Features

- 📄 PDF document processing and indexing
- 🔍 Semantic search using FAISS vector store
- 💬 Interactive chat interface with conversation history
- 🚫 Strict context-based responses (no external knowledge)
- ⚡ Fast responses using Groq's LLM API

## 🛠️ Technology Stack

- **Streamlit** - Web interface
- **LangChain** - RAG framework
- **FAISS** - Vector database for semantic search
- **HuggingFace Embeddings** - Text embeddings (all-MiniLM-L12-v2)
- **Groq API** - LLM inference (llama-3.1-8b-instant)
- **PyPDF** - PDF processing

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key ([Get it here](https://console.groq.com/))

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. **Install required packages**
```bash
pip install streamlit langchain langchain-groq langchain-community
pip install faiss-cpu sentence-transformers pypdf python-dotenv
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Update PDF path**

In the code, change the PDF file path to your document:
```python
pdf_file = "/path/to/your/document.pdf"
```

## 🎮 Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Access the app**
- Open your browser and go to `http://localhost:8501`

3. **Start chatting**
- Type your question in the chat input
- The bot will search the PDF and provide answers based on the document content
- If your question isn't covered in the PDF, it will inform you

## 📝 Example Questions

Assuming your PDF is about a research paper:
- ✅ "What is the main conclusion of this paper?"
- ✅ "Explain the methodology used in section 3"
- ✅ "What are the key findings?"
- ❌ "What's the weather today?" (Will be refused - not in PDF)

## 🔧 Configuration

You can customize the following parameters in the code:

- **Chunk size**: `chunk_size=1000` - Size of text chunks for processing
- **Chunk overlap**: `chunk_overlap=100` - Overlap between chunks
- **Number of retrieved chunks**: `search_kwargs={'k': 3}` - Number of relevant sections to retrieve
- **Embedding model**: `model_name="all-MiniLM-L12-v2"` - HuggingFace embedding model
- **LLM model**: `model="llama-3.1-8b-instant"` - Groq model

## 📂 Project Structure

```
rag-chatbot/
│
├── rag-pdf.py                 # Main application file
├── .env                   # Environment variables (not committed)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 📦 Requirements.txt

```
streamlit
langchain
langchain-groq
langchain-community
faiss-cpu
sentence-transformers
pypdf
python-dotenv
```



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Your Name - [(https://github.com/yourusername)](https://github.com/ArslanFarooq3715)

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Groq for the fast LLM API
- Streamlit for the web interface
- HuggingFace for the embedding models

---

**Note**: This chatbot is designed for educational and research purposes. Always verify critical information from the original source documents.
