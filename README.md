# IntelliVault — Intelligent PDF Question Answering with RAG

**An AI-powered document question-answering system that retrieves relevant information from uploaded PDFs and generates context-grounded answers.**

[Live Demo](https://intellivault-hmapnjckzfxqxoqruapnec.streamlit.app/) | [GitHub Repository](https://github.com/aryamohands25-lab/Intellivault)

## Overview

IntelliVault is a Retrieval-Augmented Generation (RAG) application designed to help users interact with PDF documents using natural-language questions. The system retrieves relevant document passages and uses them as context for answering questions.

The project includes multiple RAG approaches so their retrieval behaviour and answer generation can be explored and compared.

## Problem Statement

Finding specific information in lengthy PDF documents can be time-consuming. General-purpose language models may also generate answers that are not supported by the document.

IntelliVault addresses this problem by combining document processing, retrieval, and language-model-based answer generation to make document information easier to access.

## Objectives

- Allow users to upload PDF documents and ask questions in natural language.
- Retrieve relevant passages from the uploaded document.
- Generate answers grounded in retrieved document content.
- Handle questions for which the available document context does not provide an answer.
- Explore baseline, hybrid, and agentic RAG approaches.
- Provide a simple web interface for interacting with documents.

## Key Features

- **PDF Question Answering:** Ask questions about uploaded PDF content.
- **Retrieval-Augmented Generation:** Use retrieved passages as context for answer generation.
- **Multiple RAG Modes:** Baseline RAG, Hybrid RAG, and Agentic RAG.
- **Context-Aware Responses:** Generate answers using information retrieved from the document.
- **Unsupported-Question Handling:** The system can return a not-found response when sufficient supporting information is unavailable.
- **Streamlit Interface:** Interact with the application through a web browser.
- **Evaluation Tooling:** The project includes evaluation scripts for assessing RAG outputs.

## RAG Modes

| Mode | Description |
|---|---|
| Baseline RAG | A straightforward retrieval-and-generation pipeline. |
| Hybrid RAG | Uses the project's hybrid retrieval pipeline to identify relevant context. |
| Agentic RAG | Uses an agent-based workflow to coordinate retrieval and answer generation. |

The approaches can be compared to study retrieval behaviour, response quality, and execution time.

## System Architecture

1. **PDF Upload:** The user uploads a PDF through the Streamlit interface.
2. **Text Extraction:** Text is extracted from the PDF.
3. **Document Indexing:** Extracted content is prepared for retrieval using the project's embedding and vector-store workflow.
4. **Question Input:** The user enters a natural-language question.
5. **Context Retrieval:** The selected RAG pipeline retrieves relevant passages.
6. **Answer Generation:** The language model generates a response using the retrieved context.
7. **Response Display:** The answer and any source information supported by the application are displayed.

## Technology Stack

- **Programming Language:** Python
- **Web Interface:** Streamlit
- **PDF Processing:** PyMuPDF
- **Vector Store:** ChromaDB
- **Language Model Integration:** Groq or other providers supported by the application configuration
- **RAG Workflows:** Baseline, hybrid, and agentic pipelines
- **Database Integration:** MongoDB for supported history or metadata workflows
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## Project Structure

```text
Intellivault/
├── app.py              # Streamlit application and user interface
├── agent.py            # Agentic RAG workflow
├── baseline_rag.py     # Baseline retrieval and generation
├── hybrid.py           # Hybrid retrieval workflow
├── ingest.py           # Document ingestion utilities
├── config.py           # Application configuration
├── database.py         # Database integration
├── evaluate.py         # Evaluation and benchmarking utilities
├── requirements.txt    # Python dependencies
├── packages.txt        # Additional system packages, if required
└── README.md           # Project documentation
```

Additional files may be present in the repository. The `.env` file is for local configuration and should not be committed.

## Getting Started

### Prerequisites

- Python 3.11 or a version compatible with the project's dependencies
- Git
- A supported language-model API key
- MongoDB configuration if using database-backed features

### 1. Clone the Repository

```bash
git clone https://github.com/aryamohands25-lab/Intellivault.git
cd Intellivault
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file using the variable names expected by your application's configuration. For example, if Groq is the configured provider:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

Add MongoDB settings only if required by the features you use. Check `config.py` and `database.py` for the exact variable names.

**Security:** Never commit `.env`, API keys, database passwords, or other credentials to GitHub. For Streamlit Community Cloud, configure secrets in the app's Secrets settings.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## How to Use

1. Open IntelliVault.
2. Upload a PDF document.
3. Select a RAG mode.
4. Enter a question about the document.
5. Review the answer and any displayed source information.
6. Ask a question unrelated to the document to test unsupported-question handling.

## Evaluation

Useful evaluation dimensions for a RAG application include:

- **Retrieval Relevance:** Whether retrieved passages contain information relevant to the question.
- **Answer Correctness:** Whether the answer accurately reflects the document.
- **Groundedness:** Whether claims are supported by the retrieved context.
- **Citation Quality:** Whether source references support the answer, when citations are provided.
- **Latency:** How long the application takes to return a response.
- **Unsupported-Question Handling:** Whether the system avoids answering when adequate evidence is unavailable.

Use the project's evaluation scripts and actual test results to report measured performance. Do not report accuracy percentages or performance improvements unless they have been measured.

## Deployment

**Live Application:** https://intellivault-hmapnjckzfxqxoqruapnec.streamlit.app/

**GitHub Repository:** https://github.com/aryamohands25-lab/Intellivault

## Current Limitations

- Retrieval quality depends on PDF text extraction, chunking, embeddings, and retrieval settings.
- A language model may produce incorrect answers if the retrieved context is insufficient or misleading.
- The current document index uses in-memory storage, so uploaded documents may not persist across sessions or application restarts.
- Persistence and isolation of user-specific documents and chat history depend on the configured storage implementation.
- API availability, rate limits, and model response times may affect performance.

## Future Enhancements

- Persistent document storage and vector indexing.
- Reliable user-specific document and conversation history.
- Improved retrieval through reranking and retrieval evaluation.
- More comprehensive automated and human-reviewed answer evaluation.
- Better source citations and document-level traceability.
- Performance optimization for large PDFs and repeated questions.
- Expanded testing of privacy, security, and multi-user access.

## Conclusion

IntelliVault demonstrates how Retrieval-Augmented Generation can make PDF documents easier to query using natural language. By exploring baseline, hybrid, and agentic retrieval workflows, the project provides a foundation for studying context-grounded question answering, retrieval quality, and document interaction.

---

**Project:** IntelliVault  
**Category:** Artificial Intelligence · Natural Language Processing · Retrieval-Augmented Generation  
**Live Demo:** [Try IntelliVault](https://intellivault-hmapnjckzfxqxoqruapnec.streamlit.app/)  
**Source Code:** [GitHub](https://github.com/aryamohands25-lab/Intellivault)
