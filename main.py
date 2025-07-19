import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse # Import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import asyncio
import uuid

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")

# Initialize Google Gemini LLM and Embeddings
# Using 'gemini' for text generation
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
# Using 'models/embedding-001' for embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Directory to store FAISS indices temporarily
FAISS_INDEX_DIR = "faiss_indices"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

app = FastAPI(
    title="RAG Based Project Backend",
    description="Backend for Chat with PDF/Website and Product Comparison using LangChain, Gemini, and FAISS.",
    version="1.0.0"
)

# --- Pydantic Models for Request Body Validation ---

class ProcessWebsiteRequest(BaseModel):
    urls: List[HttpUrl]

class ChatRequest(BaseModel):
    question: str
    index_id: str # Unique ID for the FAISS index

class CompareProductsRequest(BaseModel):
    urls: List[HttpUrl]
    product_names: Optional[List[str]] = None # Optional list of product names to help LLM focus

# --- Utility Functions ---

async def load_and_split_pdf(file_path: str) -> List[Any]:
    """Loads a PDF file and splits its content into manageable chunks."""
    try:
        loader = PyPDFLoader(file_path)
        documents = await asyncio.to_thread(loader.load) # Run blocking I/O in a separate thread
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        print(f"Error loading and splitting PDF: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process PDF: {e}")

async def load_and_split_web(urls: List[HttpUrl]) -> List[Any]:
    """Loads content from given URLs and splits it into manageable chunks."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for url in urls:
        try:
            loader = WebBaseLoader(str(url))
            documents = await asyncio.to_thread(loader.load) # Run blocking I/O in a separate thread
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error loading and splitting web content from {url}: {e}")
            # Continue processing other URLs even if one fails
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process website {url}: {e}")
    if not all_chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content could be loaded from the provided URLs.")
    return all_chunks

async def create_faiss_index(documents: List[Any]) -> str:
    """
    Creates a FAISS vector store from documents and saves it to a temporary directory.
    Returns the unique ID of the created index.
    """
    if not documents:
        raise ValueError("No documents provided to create FAISS index.")
    try:
        # Create a unique ID for this FAISS index
        index_id = str(uuid.uuid4())
        index_path = os.path.join(FAISS_INDEX_DIR, index_id)

        # Create FAISS index - this can be a blocking operation, so run in thread
        vector_store = await asyncio.to_thread(FAISS.from_documents, documents, embeddings)
        await asyncio.to_thread(vector_store.save_local, index_path)

        return index_id
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create FAISS index: {e}")

def get_faiss_index_path(index_id: str) -> str:
    """Returns the full path to a FAISS index given its ID."""
    return os.path.join(FAISS_INDEX_DIR, index_id)

def load_faiss_index(index_id: str) -> FAISS:
    """Loads a FAISS vector store from a given index ID."""
    index_path = get_faiss_index_path(index_id)
    if not os.path.exists(index_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"FAISS index with ID '{index_id}' not found.")
    try:
        # Load FAISS index - this can be a blocking operation, so run in thread
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index {index_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load FAISS index: {e}")

# --- FastAPI Endpoints ---

@app.post("/process_pdf", summary="Process a PDF file and create a FAISS index")
async def process_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file, processes its content, and creates a FAISS vector store.
    Returns a unique ID for the created index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed.")

    temp_file_path = f"temp_{file.filename}_{uuid.uuid4().hex}" # Add UUID to prevent name conflicts
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and split PDF documents
        documents = await load_and_split_pdf(temp_file_path)
        
        # Create and save FAISS index
        index_id = await create_faiss_index(documents)
        
        return JSONResponse(content={"message": "PDF processed successfully", "index_id": index_id}, status_code=status.HTTP_200_OK)
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions directly
    except Exception as e:
        print(f"Unhandled error during PDF processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/process_website", summary="Process content from URLs and create a FAISS index")
async def process_website(request: ProcessWebsiteRequest):
    """
    Loads content from provided URLs, processes it, and creates a FAISS vector store.
    Returns a unique ID for the created index.
    """
    if not request.urls:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one URL must be provided.")

    try:
        # Load and split web documents
        documents = await load_and_split_web(request.urls)
        
        # Create and save FAISS index
        index_id = await create_faiss_index(documents)
        
        return JSONResponse(content={"message": "Website content processed successfully", "index_id": index_id}, status_code=status.HTTP_200_OK)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled error during website processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@app.post("/process_combined", summary="Process combined PDF and website content and create a FAISS index")
async def process_combined(
    files: Optional[List[UploadFile]] = File(None),
    urls: Optional[str] = Form(None) # URLs as a comma-separated string from form data
):
    """
    Combines content from uploaded PDF files and provided URLs, then creates a FAISS vector store.
    Returns a unique ID for the created index.
    """
    all_documents = []
    temp_file_paths = []
    
    parsed_urls = []
    if urls:
        try:
            # Parse comma-separated URLs
            parsed_urls = [HttpUrl(url.strip()) for url in urls.split(',') if url.strip()]
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid URL format in combined request: {e}")

    if not files and not parsed_urls:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one PDF file or URL must be provided.")

    try:
        # Process PDF files
        if files:
            for file in files:
                if not file.filename.endswith(".pdf"):
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.")
                temp_file_path = f"temp_{file.filename}_{uuid.uuid4().hex}"
                temp_file_paths.append(temp_file_path)
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                pdf_docs = await load_and_split_pdf(temp_file_path)
                all_documents.extend(pdf_docs)

        # Process URLs
        if parsed_urls:
            web_docs = await load_and_split_web(parsed_urls)
            all_documents.extend(web_docs)

        if not all_documents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid content could be processed from the provided inputs.")

        # Create and save FAISS index
        index_id = await create_faiss_index(all_documents)
        
        return JSONResponse(content={"message": "Combined content processed successfully", "index_id": index_id}, status_code=status.HTTP_200_OK)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled error during combined processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up temporary files
        for temp_path in temp_file_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.post("/chat", summary="Chat with the processed document using a FAISS index", response_class=StreamingResponse)
async def chat_with_document(request: ChatRequest):
    """
    Answers a question based on the context provided by the FAISS index, streaming the response token by token.
    """
    if not request.index_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="FAISS index ID is required for chatting.")
    if not request.question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    try:
        # Load the FAISS index
        vector_store = await asyncio.to_thread(load_faiss_index, request.index_id)
        
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever()

        # Retrieve relevant documents asynchronously
        relevant_docs: List[Document] = await asyncio.to_thread(retriever.get_relevant_documents, request.question)
        
        # Concatenate retrieved document content to form the context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Professional prompt template
        prompt_template = """
        You are a helpful AI assistant. Answer the question based on the provided context only.
        If the answer cannot be found in the context, politely state that you don't have enough information.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Format the prompt with the retrieved context and user's question
        formatted_prompt = PROMPT.format(context=context, question=request.question)

        # Define an async generator to stream LLM response
        async def generate_response_chunks():
            try:
                # Stream the LLM response
                for chunk in llm.stream(formatted_prompt):
                    yield chunk.content
            except Exception as e:
                print(f"Error during LLM streaming: {e}")
                yield f"Error: Failed to generate response. {e}"

        return StreamingResponse(generate_response_chunks(), media_type="text/plain")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled error during chat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during chat: {e}")

@app.post("/compare_products", summary="Compare products from provided URLs")
async def compare_products(request: CompareProductsRequest):
    """
    Loads content from provided URLs and uses the LLM to compare products based on features and prices.
    """
    if not request.urls or len(request.urls) < 1: # Changed to 1 as a single URL might contain multiple products
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one URL must be provided for product comparison.")

    try:
        # Load content from all provided URLs
        all_web_documents = await load_and_split_web(request.urls)
        
        if not all_web_documents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not retrieve content from the provided URLs for comparison.")

        # Combine all document contents into a single string for the prompt
        combined_content = "\n\n".join([doc.page_content for doc in all_web_documents])

        # Craft a specific prompt for product comparison
        product_names_hint = ""
        if request.product_names:
            product_names_hint = f"Focus on comparing products like: {', '.join(request.product_names)}. "

        comparison_prompt_template = f"""
        You are an expert product analyst. Your task is to compare products based on the provided web content.
        Identify key features, specifications, and especially prices for each product mentioned or implied in the text.
        {product_names_hint}
        Present your comparison in a clear, concise, and structured manner, highlighting similarities, differences, and value propositions.
        If specific prices are not found, state that.

        Web Content for Analysis:
        ---
        {combined_content}
        ---

        Please provide a detailed comparison:
        """
        
        # Invoke the LLM with the comparison prompt
        # We don't need a vector store here, as we're feeding the entire relevant content directly
        # The LLM will perform the "comparison" based on the provided text.
        response = await asyncio.to_thread(llm.invoke, comparison_prompt_template)
        
        return JSONResponse(content={"comparison_result": response.content}, status_code=status.HTTP_200_OK)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled error during product comparison: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during product comparison: {e}")

@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the RAG Based Project Backend! Visit /docs for API documentation."}

# --- Cleanup on shutdown (Optional but good practice for temporary files) ---
@app.on_event("shutdown")
async def shutdown_event():
    print("Cleaning up FAISS index directory...")
    # This will remove the entire directory and its contents
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    print("FAISS index directory cleaned.")
