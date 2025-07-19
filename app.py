import streamlit as st
import requests
import json
import os

BACKEND_URL = "http://127.0.0.1:8000" # Default FastAPI URL

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="RAG-Based Document & Product Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for FAISS index ID and chat history at the very beginning
# This ensures these variables are always present before any UI element tries to access them.
if 'faiss_index_id' not in st.session_state:
    st.session_state.faiss_index_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Initialize selected option for sidebar navigation
if 'selected_main_option' not in st.session_state:
    st.session_state.selected_main_option = "Process Documents"


# --- Helper Functions for API Calls ---

def call_backend_api(endpoint: str, method: str = "POST", files=None, data=None, json_data=None):
    """
    Generic function to call the FastAPI backend for non-streaming responses.
    Handles different request methods and data types (files, form data, JSON).
    """
    full_url = f"{BACKEND_URL}/{endpoint}"
    try:
        if method == "POST":
            if files:
                response = requests.post(full_url, files=files, data=data)
            elif json_data:
                response = requests.post(full_url, json=json_data)
            else:
                response = requests.post(full_url, data=data) # For form data without files
        elif method == "GET":
            response = requests.get(full_url)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = response.json().get('detail', str(e))
        st.error(f"Backend API Error ({response.status_code}): {error_detail}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend API at {BACKEND_URL}. Please ensure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Backend API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected error occurred during API request: {e}")
        return None

def stream_backend_api(endpoint: str, json_data: dict):
    """
    Function to call the FastAPI backend for streaming responses.
    This assumes the backend endpoint supports streaming (e.g., Server-Sent Events or chunked transfer).
    """
    full_url = f"{BACKEND_URL}/{endpoint}"
    try:
        with requests.post(full_url, json=json_data, stream=True) as response:
            response.raise_for_status()
            # Iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=8192): # Adjust chunk_size as needed
                if chunk:
                    yield chunk.decode('utf-8') # Decode bytes to string and yield
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = response.json().get('detail', str(e))
        except json.JSONDecodeError:
            error_detail = response.text # Fallback to raw text if not JSON
        st.error(f"Backend API Error ({response.status_code}): {error_detail}")
        yield f"Error: {error_detail}"
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend API at {BACKEND_URL}. Please ensure the FastAPI server is running and supports streaming.")
        yield "Error: Could not connect to backend for streaming."
    except requests.exceptions.Timeout:
        st.error("Backend API streaming request timed out.")
        yield "Error: Streaming request timed out."
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected error occurred during streaming API request: {e}")
        yield f"Error: An unexpected error occurred: {e}"


# --- Sidebar for Navigation ---
with st.sidebar:
    st.header("Navigation")
    # Use st.radio for clear navigation options
    st.session_state.selected_main_option = st.radio(
        "Choose a functionality:",
        ("Process Documents", "Chat with Document", "Compare Products"),
        index=("Process Documents", "Chat with Document", "Compare Products").index(st.session_state.selected_main_option)
    )
    st.markdown("---")
    st.subheader("Current Status")
    if st.session_state.faiss_index_id:
        st.success(f"Document Index Active: `{st.session_state.faiss_index_id[:8]}...`")
        if st.button("Clear Current Document Index", help="This will clear the current FAISS index and chat history."):
            st.session_state.faiss_index_id = None
            st.session_state.chat_history = []
            st.session_state.selected_main_option = "Process Documents" # Redirect to process page
            st.rerun() # Rerun to update the UI immediately
    else:
        st.info("No document index loaded.")
    st.markdown("---")
    st.caption("🤖 Powered by LangChain, Google Gemini, FastAPI, and Streamlit.")


# --- Main Content Area ---

st.title("📚 RAG-Based Document & Product Analysis")
st.markdown("Welcome! Select an option from the sidebar to begin interacting with the application.")
st.markdown("---")

# Conditional rendering based on sidebar selection
if st.session_state.selected_main_option == "Process Documents":
    st.header("1. Process Documents for Chat")
    st.markdown("Choose a method to upload or link your data to create a searchable knowledge base.")

    process_tab_pdf, process_tab_website, process_tab_combined = st.tabs(["📄 Process PDF", "🌐 Process Website", "🔗 Process Combined"])

    with process_tab_pdf:
        st.subheader("Upload PDF File(s)")
        st.markdown("Upload one or more PDF documents to create a searchable knowledge base.")
        uploaded_pdfs = st.file_uploader("Select PDF file(s)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader_main")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            process_pdf_btn = st.button("Process PDF(s)", key="process_pdf_btn_main", use_container_width=True)
        
        if process_pdf_btn:
            if uploaded_pdfs:
                # Backend's process_pdf endpoint currently expects a single file.
                # For multiple PDFs, the 'combined' tab is more appropriate.
                if len(uploaded_pdfs) > 1:
                    st.warning("For multiple PDFs, please use the 'Process Combined' tab for better results. Processing only the first PDF for now.")
                    first_pdf = uploaded_pdfs[0]
                    files = {"file": (first_pdf.name, first_pdf.getvalue(), "application/pdf")}
                else:
                    files = {"file": (uploaded_pdfs[0].name, uploaded_pdfs[0].getvalue(), "application/pdf")}

                with st.spinner("Processing PDF... This may take a moment."):
                    response_data = call_backend_api("process_pdf", files=files)
                    if response_data:
                        st.success(response_data.get("message", "PDF processed!"))
                        st.session_state.faiss_index_id = response_data.get("index_id")
                        st.write(f"**FAISS Index ID:** `{st.session_state.faiss_index_id}`")
                        st.session_state.chat_history = [] # Clear chat history for new document
                        st.session_state.selected_main_option = "Chat with Document" # Auto-navigate to chat
                        st.rerun() # Rerun to update UI
            else:
                st.warning("Please upload at least one PDF file first.")

    with process_tab_website:
        st.subheader("Link Website URLs")
        st.markdown("Enter URLs of websites to extract content and create a searchable knowledge base.")
        website_urls_input = st.text_area(
            "Enter website URLs (one per line or comma-separated)",
            placeholder="e.g., https://www.example.com/about-us\nhttps://blog.example.com/latest-post",
            height=100,
            key="website_urls_input_main"
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            process_website_btn = st.button("Process Websites", key="process_website_btn_main", use_container_width=True)
        
        if process_website_btn:
            urls_list = [url.strip() for url in website_urls_input.split(',') if url.strip()]
            if not urls_list:
                urls_list = [url.strip() for url in website_urls_input.split('\n') if url.strip()]

            if urls_list:
                valid_urls = []
                for url_str in urls_list:
                    try:
                        _ = requests.utils.urlparse(url_str) 
                        valid_urls.append(url_str)
                    except Exception:
                        st.warning(f"Invalid URL skipped: `{url_str}`")
                
                if valid_urls:
                    with st.spinner("Processing websites... This may take a moment."):
                        json_data = {"urls": valid_urls}
                        response_data = call_backend_api("process_website", json_data=json_data)
                        if response_data:
                            st.success(response_data.get("message", "Website content processed!"))
                            st.session_state.faiss_index_id = response_data.get("index_id")
                            st.write(f"**FAISS Index ID:** `{st.session_state.faiss_index_id}`")
                            st.session_state.chat_history = [] # Clear chat history for new document
                            st.session_state.selected_main_option = "Chat with Document" # Auto-navigate to chat
                            st.rerun() # Rerun to update UI
                else:
                    st.warning("No valid URLs provided to process.")
            else:
                st.warning("Please enter at least one website URL.")

    with process_tab_combined:
        st.subheader("Combine PDFs and Websites")
        st.markdown("Upload multiple PDF files and/or enter website URLs to create a single, comprehensive knowledge base.")
        uploaded_pdfs_combined = st.file_uploader("Select PDF file(s) for combined processing (multiple allowed)", type=["pdf"], accept_multiple_files=True, key="combined_pdf_uploader_main")
        website_urls_combined_input = st.text_area(
            "Enter website URLs for combined processing (one per line or comma-separated)",
            placeholder="e.g., https://example.com/report.html, https://docs.example.com/faq",
            height=100,
            key="combined_website_urls_input_main"
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            process_combined_btn = st.button("Process Combined Content", key="process_combined_btn_main", use_container_width=True)
        
        if process_combined_btn:
            files_to_send = []
            for pdf_file in uploaded_pdfs_combined:
                files_to_send.append(("files", (pdf_file.name, pdf_file.getvalue(), "application/pdf")))
            
            urls_list_combined = [url.strip() for url in website_urls_combined_input.split(',') if url.strip()]
            if not urls_list_combined:
                urls_list_combined = [url.strip() for url in website_urls_combined_input.split('\n') if url.strip()]

            valid_urls_combined = []
            for url_str in urls_list_combined:
                try:
                    _ = requests.utils.urlparse(url_str)
                    valid_urls_combined.append(url_str)
                except Exception:
                    st.warning(f"Invalid URL skipped in combined input: `{url_str}`")

            if files_to_send or valid_urls_combined:
                with st.spinner("Processing combined content... This may take a moment."):
                    data_to_send = {}
                    if valid_urls_combined:
                        data_to_send["urls"] = ",".join(valid_urls_combined) # Send as comma-separated string

                    response_data = call_backend_api("process_combined", files=files_to_send, data=data_to_send)
                    if response_data:
                        st.success(response_data.get("message", "Combined content processed!"))
                        st.session_state.faiss_index_id = response_data.get("index_id")
                        st.write(f"**FAISS Index ID:** `{st.session_state.faiss_index_id}`")
                        st.session_state.chat_history = [] # Clear chat history for new document
                        st.session_state.selected_main_option = "Chat with Document" # Auto-navigate to chat
                        st.rerun() # Rerun to update UI
            else:
                st.warning("Please upload at least one PDF or enter at least one URL for combined processing.")

elif st.session_state.selected_main_option == "Chat with Document":
    st.header("2. Chat with Processed Document")
    st.markdown("Ask questions about the content you've processed. The AI will answer based on the loaded knowledge base.")

    if st.session_state.faiss_index_id:
        st.info(f"Chatting with document index: `{st.session_state.faiss_index_id}`")
        
        # Display chat history
        chat_container = st.container(height=400, border=True)
        with chat_container:
            # Welcome message for new chat sessions
            if not st.session_state.chat_history:
                st.chat_message("assistant").write("Hello! I'm ready to answer questions about your document. What would you like to know?")
            
            for chat_item in st.session_state.chat_history:
                st.chat_message(chat_item["role"]).write(chat_item["content"])

        # Chat input
        user_question = st.chat_input("Type your question here...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            # Immediately display user's message
            with chat_container:
                st.chat_message("user").write(user_question)

            # Create a placeholder for the assistant's response
            with chat_container:
                assistant_response_placeholder = st.chat_message("assistant").empty()
            
            full_response_content = ""
            with st.spinner("Thinking..."):
                json_data = {
                    "question": user_question,
                    "index_id": st.session_state.faiss_index_id
                }
                
                try:
                    # Use the streaming function
                    for chunk in stream_backend_api("chat", json_data=json_data):
                        full_response_content += chunk
                        # Update the placeholder with the new content
                        assistant_response_placeholder.write(full_response_content + "▌") # Add blinking cursor
                    
                    # Remove blinking cursor after full response
                    assistant_response_placeholder.write(full_response_content)

                    if not full_response_content.strip():
                        full_response_content = "Sorry, I couldn't get an answer at this time. Please check the backend server."

                    st.session_state.chat_history.append({"role": "assistant", "content": full_response_content})

                except Exception as e:
                    error_message = f"An error occurred during streaming: {e}. Please check backend logs."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    with chat_container:
                        st.chat_message("assistant").write(error_message)

    else:
        st.warning("Please process a PDF or website first in the 'Process Documents' section to enable chat functionality.")
        if st.button("Go to Process Documents"):
            st.session_state.selected_main_option = "Process Documents"
            st.rerun()

elif st.session_state.selected_main_option == "Compare Products":
    st.header("3. Compare Products from Websites")
    st.markdown("Enter URLs of product pages (e.g., from e-commerce sites) to get an AI-powered comparison of features and prices.")

    compare_urls_input = st.text_area(
        "Enter product URLs (one per line or comma-separated)",
        placeholder="e.g., https://www.amazon.com/productA\nhttps://www.bestbuy.com/productB",
        height=100,
        key="compare_urls_input_main"
    )
    product_names_input = st.text_input(
        "Optional: Enter specific product names to guide the AI (comma-separated)",
        placeholder="e.g., 'iPhone 15 Pro', 'Samsung Galaxy S24'",
        key="product_names_input_main"
    )

    col1_comp, col2_comp = st.columns([1, 2])
    with col1_comp:
        compare_products_btn = st.button("Compare Products", key="compare_products_btn_main", use_container_width=True)

    if compare_products_btn:
        urls_for_comparison = [url.strip() for url in compare_urls_input.split(',') if url.strip()]
        if not urls_for_comparison:
            urls_for_comparison = [url.strip() for url in compare_urls_input.split('\n') if url.strip()]

        product_names_for_comparison = [name.strip() for name in product_names_input.split(',') if name.strip()]

        if urls_for_comparison:
            valid_urls_for_comparison = []
            for url_str in urls_for_comparison:
                try:
                    _ = requests.utils.urlparse(url_str)
                    valid_urls_for_comparison.append(url_str)
                except Exception:
                    st.warning(f"Invalid URL skipped for comparison: `{url_str}`")

            if valid_urls_for_comparison:
                with st.spinner("Comparing products... This may take a while depending on content complexity and number of URLs."):
                    json_data = {
                        "urls": valid_urls_for_comparison,
                        "product_names": product_names_for_comparison if product_names_for_comparison else None
                    }
                    response_data = call_backend_api("compare_products", json_data=json_data)
                    if response_data:
                        st.subheader("Comparison Result:")
                        st.markdown(response_data.get("comparison_result", "Could not generate a comparison result."))
                # Error handling is already in call_backend_api
            else:
                st.warning("No valid URLs provided for comparison.")
        else:
            st.warning("Please enter at least one URL to compare products.")

st.markdown("---")
st.caption("🤖 Powered by LangChain, Google Gemini, FastAPI, and Streamlit.")
