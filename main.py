import streamlit as st
import requests
import json
import os
import base64
import tempfile
from mistralai import Mistral
from PIL import Image
import io
from docx import Document
from dotenv import load_dotenv
import PyPDF2  # For PDF file handling

# Load environment variables from .env file (for API key)
load_dotenv()

# Hardcoded API key
API_KEY = "your_mistral_api-key"

# Initialize Mistral API client
client = Mistral(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Maitri AI OCR",
    page_icon="📄",
    layout="wide"
)

# Sidebar for feature selection
st.sidebar.title("Features")
feature = st.sidebar.radio("Choose a feature:", ["Document Classifier", "OCR Processor"])

# Function to create a Word document
def create_word_document(content):
    doc = Document()
    doc.add_paragraph(content)
    doc_bytes = io.BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    return doc_bytes.getvalue()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to upload PDF to the API
def upload_pdf(client, content, filename):
    """
    Uploads a PDF to the API and retrieves a signed URL for processing.

    Args:
        client: API client instance.
        content (bytes): The content of the PDF file.
        filename (str): The name of the PDF file.

    Returns:
        str: Signed URL for the uploaded PDF.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, "wb") as tmp:
            tmp.write(content)

        try:
            with open(temp_path, "rb") as file_obj:
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )

            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Function to process OCR using the API
def process_ocr(client, document_source):
    """
    Processes a document using the OCR API.

    Args:
        client: API client instance.
        document_source (dict): The source of the document (URL or image).

    Returns:
        OCRResponse: The response from the OCR API.
    """
    return client.ocr.process(
        model="mistral-ocr-latest",  # Use mistral-ocr-latest for OCR
        document=document_source,
        include_image_base64=True
    )

# Function to safely parse JSON or return raw text
def parse_json_or_raw(response_text):
    try:
        # Try to parse the response as JSON
        json_result = json.loads(response_text)
        return json_result, True  # Return parsed JSON and a flag indicating success
    except json.JSONDecodeError:
        return response_text, False  # Return raw text and a flag indicating failure

# Document Classifier Feature
if feature == "Document Classifier":
    st.title("Maitri AI: Document Classifier and Analyzer")
    st.markdown("Upload or paste a document to classify it as legal or normal and extract key information.")

    # Function to call the AI API
    def analyze_with_ai(text):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        system_prompt = """You are an AI document classifier and analyzer. Your task is to determine whether a given document is a **legal document** or a **normal document** based on its content.  
        ### Classification Criteria:
        - A **legal document** typically includes legal terminology (e.g., "whereas," "pursuant to," "hereinafter"), structured clauses, references to laws or contracts, and formal legal language.
        - A **normal document** is general in nature, lacking legal-specific language, and may include emails, reports, articles, or casual text.

        ### Instructions:
        1. **Classify the document** as either:
           - `Legal Document`
           - `Normal Document`
        2. **If it's a legal document**, extract the following:
           - Key clauses (e.g., Termination, Indemnity, Liability, Confidentiality)
           - Any referenced laws or statutes
           - A brief summary of the document in simple terms
        3. **If it's a normal document**, extract:
           - A brief summary
           - Key topics covered
           - Sentiment analysis (Positive, Neutral, Negative)
        4. **Provide the output in the following JSON format**:
        ```json
        {
          "document_type": "Legal Document" | "Normal Document",
          "key_clauses": ["Clause 1", "Clause 2", ...],  // Only for legal documents
          "referenced_laws": ["Law 1", "Law 2", ...],   // Only for legal documents
          "summary": "Brief summary of the document",
          "key_topics": ["Topic 1", "Topic 2", ...],    // Only for normal documents
          "sentiment": "Positive" | "Neutral" | "Negative"  // Only for normal documents
        }
        ```

        Ensure your response is **ONLY** valid JSON without any additional text or explanation."""

        payload = {
            "model": "mistral-large-latest",  # Use mistral-large-latest for Document Classifier
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.2,
            "max_tokens": 4000
        }

        try:
            response = requests.post("https://api.mistral.ai/v1/chat/completions",
                                     headers=headers,
                                     json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            response_json = response.json()
            # Extract the actual content from the response
            if "choices" in response_json and len(response_json["choices"]) > 0:
                result = response_json["choices"][0]["message"]["content"]
                return result
            else:
                return "Error: No content in API response"
        except requests.exceptions.RequestException as e:
            return f"Error calling AI API: {str(e)}"

    # Main content area
    tab1, tab2 = st.tabs(["Text Input", "File Upload"])

    # Text input tab
    with tab1:
        user_input = st.text_area("Paste your document text here:", height=300)
        analyze_button = st.button("Analyze Document", key="analyze_text")

        if analyze_button and user_input:
            with st.spinner("Analyzing document..."):
                result = analyze_with_ai(user_input)

                # Parse the result as JSON or display raw text
                parsed_result, is_json = parse_json_or_raw(result)

                # Display the result
                st.subheader("Analysis Result")
                if is_json:
                    st.json(parsed_result)  # Display as JSON if parsing succeeded
                else:
                    st.write(parsed_result)  # Display raw text if parsing failed

                # Download as Word file
                word_doc = create_word_document(result)
                st.download_button(
                    label="Download as Word",
                    data=word_doc,
                    file_name="classified_document.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    # File upload tab
    with tab2:
        uploaded_file = st.file_uploader("Upload a document file", type=["txt", "pdf", "docx"])

        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)

            # Extract text based on file type
            if uploaded_file.type == "text/plain":
                text_content = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                text_content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text_content = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file type")
                text_content = ""

            if text_content:
                st.text_area("File content preview:", text_content[:1000] + ("..." if len(text_content) > 1000 else ""),
                             height=150)
                analyze_file_button = st.button("Analyze Document", key="analyze_file")

                if analyze_file_button:
                    with st.spinner("Analyzing document..."):
                        result = analyze_with_ai(text_content)

                        # Parse the result as JSON or display raw text
                        parsed_result, is_json = parse_json_or_raw(result)

                        # Display the result
                        st.subheader("Analysis Result")
                        if is_json:
                            st.json(parsed_result)  # Display as JSON if parsing succeeded
                        else:
                            st.write(parsed_result)  # Display raw text if parsing failed

                        # Download as Word file
                        word_doc = create_word_document(result)
                        st.download_button(
                            label="Download as Word",
                            data=word_doc,
                            file_name="classified_document.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

# OCR Processor Feature
elif feature == "OCR Processor":
    st.title("Maitri AI: OCR Processor")

    # Input method selection: PDF Upload or Image Upload
    input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload"])

    document_source = None
    preview_content = None
    content_type = None

    if input_method == "PDF Upload":
        # Handle PDF file upload
        uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
        if uploaded_file:
            content = uploaded_file.read()
            preview_content = uploaded_file

            # Save the uploaded PDF temporarily for display purposes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                pdf_path = tmp.name

            # Display the uploaded PDF
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            # Prepare document source for OCR processing
            document_source = {
                "type": "document_url",
                "document_url": upload_pdf(client, content, uploaded_file.name)
            }
            content_type = "pdf"

    elif input_method == "Image Upload":
        # Handle image file upload
        uploaded_image = st.file_uploader("Choose Image file", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Prepare document source for OCR processing
            document_source = {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_str}"
            }
            content_type = "image"

    if document_source and st.button("Process Document"):
        # Process the document when the user clicks the button
        with st.spinner("Extracting content..."):
            try:
                ocr_response = process_ocr(client, document_source)

                if ocr_response and ocr_response.pages:
                    # Combine extracted text from all pages into one string
                    extracted_content = "\n\n".join(
                        [f"**Page {i + 1}**\n{page.markdown}"
                         for i, page in enumerate(ocr_response.pages)]
                    )

                    # Display extracted content in Markdown format
                    st.subheader("Extracted Content")
                    st.markdown(extracted_content)

                    # Prepare plain text version
                    plain_text_content = "\n\n".join(
                        [f"Page {i + 1}\n{page.markdown}"
                         for i, page in enumerate(ocr_response.pages)]
                    )

                    # Create a Word document
                    word_doc = create_word_document(plain_text_content)

                    # Add download buttons for text, Markdown, and Word formats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="Download as Text",
                            data=plain_text_content,
                            file_name="extracted_content.txt",
                            mime="text/plain"
                        )
                    with col2:
                        st.download_button(
                            label="Download as Markdown",
                            data=extracted_content,
                            file_name="extracted_content.md",
                            mime="text/markdown"
                        )
                    with col3:
                        st.download_button(
                            label="Download as Word",
                            data=word_doc,
                            file_name="extracted_content.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                    # Optional: Show raw response for debugging purposes
                    with st.expander("Raw API Response"):
                        st.json(ocr_response.model_dump())

                else:
                    st.warning("No content extracted.")

            except Exception as e:
                # Display an error message if processing fails
                st.error(f"Processing error: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Maitri AI OCR")
