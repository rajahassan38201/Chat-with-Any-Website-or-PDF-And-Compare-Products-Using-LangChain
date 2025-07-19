RAG-Based Document & Product Analysis
Table of Contents
Introduction

Key Features

Technologies Used

Project Structure

Getting Started

Prerequisites

Cloning the Repository

Setting Up the Virtual Environment

Installing Dependencies

Setting Up Environment Variables

Running the Backend (FastAPI)

Running the Frontend (Streamlit)

How to Use

Process Documents

Chat with Document

Compare Products

Retrieval-Augmented Generation (RAG)

LangChain for Orchestration

FAISS for Vector Similarity Search

Google Gemini LLM and Embeddings

Asynchronous Programming with FastAPI

Streaming Responses for Enhanced UX

Pydantic for Data Validation

Contact

1. Introduction
   This project delivers a robust and interactive RAG (Retrieval-Augmented Generation) system, designed to enhance the capabilities of Large Language Models (LLMs) by grounding their responses in specific, user-provided data. It offers a seamless experience for interacting with various document types and performing intelligent product comparisons. The application is built with a high-performance FastAPI backend and a user-friendly Streamlit frontend, leveraging the power of LangChain and Google Gemini.

2. Key Features
   Chat with PDF: Upload PDF documents and engage in a conversational Q&A session based only on the content of the uploaded PDFs.

Chat with Website: Provide URLs of websites, and the AI will answer questions by extracting and understanding information from those web pages.

Combined Document Processing: Ability to combine content from multiple PDFs and websites into a single, comprehensive knowledge base for unified querying.

Product Comparison: Input multiple product page URLs, and the system will intelligently compare features, specifications, and prices, providing a structured summary.

Scalable Backend: Built with FastAPI for high performance and asynchronous handling of I/O operations.

Intuitive Frontend: A clean and easy-to-use Streamlit interface with clear navigation.

3. Technologies Used
   Backend:

FastAPI: Modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

uvicorn: ASGI server for running FastAPI applications.

python-dotenv: For loading environment variables from a .env file.

Frontend:

Streamlit: An open-source app framework for creating interactive web applications with Python.

requests: Python HTTP library for making requests to the FastAPI backend.

LLM & RAG Framework:

LangChain: A powerful framework for developing applications powered by language models. It facilitates chaining LLM calls, integrating with external data sources, and managing prompts.

langchain-google-genai: Integration for Google Gemini models within LangChain.

Google Gemini (gemini-pro): The Large Language Model used for generating responses.

Google Generative AI Embeddings (models/embedding-001): Used to convert text into numerical vector representations (embeddings).

pypdf: Python library for working with PDF files, used by LangChain's PyPDFLoader.

faiss-cpu: (Facebook AI Similarity Search) A library for efficient similarity search and clustering of dense vectors. It serves as the vector store for storing and retrieving document embeddings.

pydantic: Used in FastAPI for data validation and settings management.

4. Project Structure
   The project is organized into three main files:

rag_project/
├── .env
├── main.py
└── app.py

.env: Stores sensitive information like your Google Gemini API key.

main.py: Contains the FastAPI backend logic, including API endpoints for document processing, chatting, and product comparison. It handles LLM interactions, embedding generation, and FAISS vector store management.

app.py: Implements the Streamlit frontend, providing the user interface for interacting with the backend API.

5. Getting Started
   Follow these steps to set up and run the project locally.

Prerequisites
Python 3.9+

A Google Gemini API Key (obtainable from Google AI Studio or Google Cloud).

Cloning the Repository
git clone https://github.com/rajahassan38201/rag-based-document-&Any-Website-product-analysis.git #

cd rag-based-document-product-analysis

Setting Up the Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv venv

# On Windows:

.\venv\Scripts\activate

# On macOS/Linux:

source venv/bin/activate

Installing Dependencies
Once your virtual environment is activated, install the required Python packages:

pip install -r requirements.txt

# If you don't have a requirements.txt, you can use:

# pip install fastapi uvicorn python-dotenv langchain langchain-google-genai pypdf faiss-cpu requests streamlit

Setting Up Environment Variables
Create a file named .env in the root directory of your project (same level as main.py and app.py). Add your Google Gemini API key to this file:

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE

Replace YOUR_GEMINI_API_KEY_HERE with your actual API key.

Running the Backend (FastAPI)
Open your terminal or command prompt, navigate to your project directory, activate your virtual environment, and run the FastAPI application:

uvicorn main:app --reload

You should see output indicating the server is running, typically on http://127.0.0.1:8000. You can test the API documentation by visiting http://127.0.0.1:8000/docs in your web browser. Keep this terminal open.

Running the Frontend (Streamlit)
Open a new terminal or command prompt, navigate to your project directory, activate your virtual environment, and run the Streamlit application:

streamlit run app.py

This will automatically open the Streamlit application in your default web browser, usually on http://localhost:8501.

Important: Ensure your FastAPI backend is running before you start the Streamlit frontend.

6. Contact
   For any questions or collaborations, feel free to reach out:

Name: Hafiz Hassan Abdullah

Email: rajahassan38201@gmail.com

GitHub: rajahassan38201
