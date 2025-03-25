import pandas as pd
import streamlit as st
import os
import httpx
import tempfile
import io
import uuid
import re
from typing import List
from streamlit_pdf_viewer import pdf_viewer
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Docling importers
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.base_models import InputFormat
from docling.backend.msword_backend import MsWordDocumentBackend
from tempfile import NamedTemporaryFile
import json
import fitz
from languages import AB_LANG
from langdetect import detect
from stop_words import get_stop_words
from dotenv import load_dotenv


def main():
    """Main function to run the Streamlit app."""
    # Load environment variables
    load_dotenv()

    # Set page config
    st.set_page_config(page_title="📚💬 Document Q&A Assistant")

    # Main content
    st.markdown(
        "<h1 style='text-align: center;'>📚💬 Document Q&A Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.subheader("Upload documents to get started.")

    # Global constants
    TEMP_FILE_DICT = dict()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
    HYPOTHESIS_MAX_TOKENS = 200
    HYPOTHESIS_TEMP = 0.0
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4-turbo-preview")
    RERANKER_MODEL_NAME = "./models/bge-reranker-v2-m3"

    # Sidebar configuration
    st.sidebar.title("RAG Settings")

    # Document parsing settings
    st.sidebar.header("📑 Parsing settings")
    USE_SELECTIVE_PARSER = st.sidebar.checkbox("Use advanced page parsing", value=True)
    if USE_SELECTIVE_PARSER:
        K_PDF_PAGES_SELECTIVE_PARSER = st.sidebar.slider(
            "Number of top relevant PDF pages to parse",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )
        K_PAGES_SELECTIVE_PARSER = K_PDF_PAGES_SELECTIVE_PARSER * 1

    st.sidebar.markdown("---")

    # Retriever settings
    st.sidebar.header("🔍 Search (Retriever) settings")
    embedding_method = st.sidebar.selectbox(
        "Semantic Method (Embedding model)",
        ["multilingual-e5-large-instruct", "bge-m3"],
        index=0,
    )

    st.sidebar.markdown("---")
    USE_RERANKER = st.sidebar.checkbox(
        "Use reranker model to filter semantic results", value=False
    )
    if USE_RERANKER:
        RERANKER_MODEL = st.sidebar.selectbox(
            "Reranker model", ["bge-reranker-v2-m3"], index=0
        )
        RERANKER_FILTER_FACTOR = st.sidebar.slider(
            "Filter factor of reranker", min_value=1, max_value=20, value=3, step=1
        )

    st.sidebar.markdown("---")
    WEIGHT_BM25 = st.sidebar.slider(
        "Keywords Search Emphasis (hybrid lexical BM25 method weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
    )
    st.sidebar.write(
        f"Semantic Search Emphasis (hybrid semantic method weight) = {1.0-WEIGHT_BM25}"
    )

    st.sidebar.markdown("---")
    USE_MULTIQUERY_RETRIEVAL = st.sidebar.checkbox(
        "Use multiquery retrieval (rephrase query with LLM)", value=False
    )
    USE_HYDE = st.sidebar.checkbox("Generate Hypothetical Answers", value=False)
    if USE_HYDE:
        WEIGHT_HYDE = st.sidebar.slider(
            "LLM Hypothetical Answer Emphasis (HyDE method weight)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        st.sidebar.write(f"Standard Query Search weight = {1.0-WEIGHT_HYDE}")
        HIDDEN_LLM_HYPOTHESIS = st.sidebar.checkbox(
            "Hidden LLM Hypothetical Answer", value=True
        )

    # Calculate derived parameters
    K_FINAL = 3 * K_PDF_PAGES_SELECTIVE_PARSER
    K_EMBEDDING = 3 * K_FINAL
    K_BM25 = 2 * K_PDF_PAGES_SELECTIVE_PARSER

    # LLM generation settings
    st.sidebar.markdown("---")
    st.sidebar.header("📝 LLM Answer Generation settings")
    ANSWER_TEMP = st.sidebar.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )
    ANSWER_MAX_TOKENS = st.sidebar.slider(
        "Max tokens Generation", min_value=0, max_value=8000, value=4000, step=100
    )

    # Document chunking settings
    st.sidebar.markdown("---")
    st.sidebar.header("✂️ Chunking settings")
    CHUNK_SIZE = st.sidebar.slider(
        "Chunk size (number of chars)", min_value=0, max_value=1600, value=900, step=50
    )
    CHUNK_OVERLAP = st.sidebar.slider(
        "Chunk overlap (number of chars)",
        min_value=0,
        max_value=CHUNK_SIZE,
        value=200,
        step=50,
    )

    st.sidebar.markdown("---")


    def get_llm_answer():
        """Initialize and return the LLM for generating answers.

        Returns:
            OpenAI: Configured OpenAI LLM instance with specified parameters.
        """
        api_params = {
            "model": LLM_MODEL_NAME,
            "max_tokens": ANSWER_MAX_TOKENS,
            "temperature": ANSWER_TEMP,
        }
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=api_params["model"],
            max_tokens=api_params["max_tokens"],
            temperature=api_params["temperature"],
        )
        return llm


    def clean_images_description(markdown_string):
        """Remove image description tags from markdown text.

        Args:
            markdown_string (str): Input markdown text containing image descriptions.

        Returns:
            str: Cleaned markdown text with image descriptions removed.
        """
        pattern = r"<image description>.*?</image description>"
        clean_text = re.sub(pattern, "", markdown_string, flags=re.DOTALL)
        return clean_text


    def get_stopwords_for_languages(languages):
        """Get combined stopwords for multiple languages.

        Args:
            languages (List[str]): List of language codes.

        Returns:
            set: Combined set of stopwords for all languages.
        """
        # Map language codes to stopwords
        lang_to_stopwords = {
            'fr': 'french',
            'en': 'english',
            'es': 'spanish',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'nl': 'dutch',
            'pl': 'polish',
            'ru': 'russian',
            'ar': 'arabic',
            'hi': 'hindi',
            'bn': 'bengali',
            'ja': 'japanese',
            'ko': 'korean',
            'zh': 'chinese',
            'da': 'danish',
            'fi': 'finnish',
            'el': 'greek',
            'he': 'hebrew',
            'hu': 'hungarian',
            'id': 'indonesian',
            'no': 'norwegian',
            'ro': 'romanian',
            'sk': 'slovak',
            'sv': 'swedish',
            'tr': 'turkish',
            'uk': 'ukrainian',
            'vi': 'vietnamese'
        }
        
        # Combine stopwords from all detected languages
        all_stopwords = set()
        for lang in languages:
            try:
                lang_name = lang_to_stopwords.get(lang, 'english')
                all_stopwords.update(get_stop_words(lang_name))
            except:
                # If there's an error getting stopwords for a language, skip it
                continue
        
        # If no stopwords were found, use English as fallback
        if not all_stopwords:
            all_stopwords = set(get_stop_words('english'))
        
        return all_stopwords


    def get_language_docs(documents):
        """Detect the languages of the documents.

        Args:
            documents (List[Document]): List of documents to analyze.

        Returns:
            List[str]: List of detected language codes.
        """
        detected_languages = set()
        
        # Process each document separately to detect multiple languages
        for doc in documents:
            try:
                lang = detect(doc.page_content)
                if lang in AB_LANG:  # Only add if language is in our supported list
                    detected_languages.add(lang)
            except:
                continue
        
        # If no languages detected, default to English
        if not detected_languages:
            detected_languages.add('en')
        
        return list(detected_languages)  # Return list of language codes


    def remove_multilingual_stopwords(text, languages):
        """Remove stopwords from text using pre-computed language-specific stopwords.

        Args:
            text (str): Input text to remove stopwords from.
            languages (List[str]): List of language codes to use for stopwords.

        Returns:
            str: Text with stopwords removed.
        """
        # Get combined stopwords for all detected languages
        stop_words = get_stopwords_for_languages(languages)
        
        # Split text into sentences to handle multiple languages
        sentences = text.split('.')
        processed_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            words = sentence.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            processed_sentences.append(" ".join(filtered_words))
        
        return " ".join(processed_sentences)


    def simple_tokenize(text):
        """Tokenize text by removing punctuation and converting to lowercase.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens with punctuation removed and converted to lowercase.
        """
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [token for token in text.split() if token]


    def preprocess_func(text):
        """Preprocess text by removing stopwords and tokenizing.

        Args:
            text (str): Input text to preprocess.

        Returns:
            List[str]: List of preprocessed tokens.
        """
        # Get languages from session state if available, otherwise detect them
        if 'detected_languages' not in st.session_state:
            if 'documents' in st.session_state:
                st.session_state.detected_languages = get_language_docs(st.session_state.documents)
            else:
                st.session_state.detected_languages = ['en']  # Default to English
        
        text = remove_multilingual_stopwords(text, st.session_state.detected_languages)
        return simple_tokenize(text)


    class SelectivePageParserRetriever(BaseRetriever):
        """Retriever that selectively parses only the most relevant document pages.

        This retriever ensures that only the most relevant pages from documents are parsed,
        with different limits for PDF and non-PDF documents.

        Attributes:
            base_retriever (BaseRetriever): The underlying retriever to use for initial document selection.
            k (int): Maximum number of non-PDF pages to parse.
            k_pdf (int): Maximum number of PDF pages to parse.
        """
        
        base_retriever: BaseRetriever
        k: int
        k_pdf: int

        def _get_relevant_documents(self, query, *, run_manager):
            """Get relevant documents with selective page parsing.

            Args:
                query (str): The query to search for.
                run_manager (CallbackManagerForRetrieverRun): Callback manager for the retriever run.

            Returns:
                List[Document]: List of relevant documents with selective page parsing applied.
            """
            docs = self.base_retriever._get_relevant_documents(query, run_manager=run_manager)
            doc_pages_selected = set()
            docs_pages_pdf_selected = set()
            result = list()

            for d in docs:
                doc_id = (d.metadata["source"], d.metadata["page"])
                if doc_id not in doc_pages_selected:
                    doc_pages_selected.add(doc_id)
                    if doc_id[0].endswith(".pdf"):
                        docs_pages_pdf_selected.add(doc_id)
                        if len(docs_pages_pdf_selected) <= self.k_pdf:
                            result.append(load_doc_with_docling(doc_id[0], doc_id[1]))
                    else:
                        if len(doc_pages_selected) <= self.k:
                            result.append(load_doc_with_docling(doc_id[0], doc_id[1]))
            return result


    class HyDERetriever(BaseRetriever):
        """Hypothetical Document Embedding retriever that generates synthetic documents.

        This retriever uses a language model to generate hypothetical documents based on
        the query and uses them to retrieve relevant documents.

        Attributes:
            documents (List[Document]): List of documents to search through.
            base_retriever (BaseRetriever): The underlying retriever to use for document retrieval.
        """
        
        documents: List[Document]
        base_retriever: BaseRetriever

        def _get_relevant_documents(self, query, *, run_manager):
            """Get relevant documents using hypothetical document generation.

            Args:
                query (str): The query to search for.
                run_manager (CallbackManagerForRetrieverRun): Callback manager for the retriever run.

            Returns:
                List[Document]: List of relevant documents.
            """
            # Generate a hypothetical document based on the query
            gen_doc = self.generate_hypothesis_document(query)
            if not HIDDEN_LLM_HYPOTHESIS:
                st.success(f"Generated document: {gen_doc}")

            # Use the generated document to retrieve relevant documents
            docs = self.base_retriever._get_relevant_documents(gen_doc, run_manager=run_manager)
            return docs

        def generate_hypothesis_document(self, query):
            """Generate a hypothetical document based on the query.

            Args:
                query (str): The query to generate a hypothesis for.

            Returns:
                str: Generated hypothetical document.
            """
            language = get_language_docs(self.documents)
            template = """Imagine you are an expert writing a detailed explanation on the topic: '{query}'
    Your response should be comprehensive and include all key points that would be found in the top search result.
    Your answer must be in '{language}' and it should contain between 100-150 words"""
            system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
            messages = chat_prompt.format_prompt(query=query, language=language).to_messages()
            response = self.get_llm().invoke(messages)
            return response

        def get_llm(self):
            """Initialize and return the LLM for hypothesis generation.

            Returns:
                ChatOpenAI: Configured ChatOpenAI LLM instance for hypothesis generation.
            """
            api_params = {
                "model": LLM_MODEL_NAME,
                "max_tokens": HYPOTHESIS_MAX_TOKENS,
                "temperature": HYPOTHESIS_TEMP,
            }
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=api_params["model"],
                max_tokens=api_params["max_tokens"],
                temperature=api_params["temperature"]
            )
            return llm


    class DoclingMultiFormatLoader(BaseLoader):
        """Custom document loader that handles multiple document formats.

        This loader supports various document formats including PDF, DOCX, HTML, XLSX, PPTX,
        ASCIIDOC, and MD files. It uses Docling for document processing and conversion.

        Attributes:
            _file_paths (Union[str, List[str]]): Path or list of paths to the documents to load.
            _converter (DocumentConverter): The document converter instance for processing files.
        """

        def __init__(self, file_paths):
            """Initialize the loader with file paths and configure the document converter.

            Args:
                file_paths (Union[str, List[str]]): Path or list of paths to the documents to load.
            """
            self._file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
            # Configure Docling pipeline options
            pipeline_options = PdfPipelineOptions(
                artifacts_path="./models/docling-models",
                do_ocr=False,
                do_table_structure=True,
                use_gpu=True,
            )
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.table_structure_options.do_cell_matching = True

            # Initialize the document converter
            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.IMAGE,
                    InputFormat.HTML,
                    InputFormat.XLSX,
                    InputFormat.PPTX,
                    InputFormat.ASCIIDOC,
                    InputFormat.MD,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.DOCX: WordFormatOption(
                        pipeline_cls=SimplePipeline, backend=MsWordDocumentBackend
                    ),
                },
            )

        def lazy_load(self):
            """Load documents lazily, yielding one document at a time.

            This method processes documents page by page, handling both multi-page
            and single-page documents appropriately.

            Yields:
                Document: Processed document with metadata.
            """
            for source in self._file_paths:
                dl_doc = self._converter.convert(source).document
                if dl_doc.num_pages() > 0:
                    # Process multi-page documents
                    for page_num in range(1, dl_doc.num_pages() + 1):
                        text = dl_doc.export_to_markdown(page_no=page_num)
                        yield Document(
                            page_content=text, metadata={"source": source, "page": page_num}
                        )
                else:
                    # Process single-page documents
                    text_splitter = MarkdownTextSplitter()
                    text = dl_doc.export_to_markdown()
                    for page_num, page_text in enumerate(text_splitter.split_text(text)):
                        yield Document(
                            page_content=page_text,
                            metadata={"source": source, "page": page_num + 1},
                        )


    def load_docs_with_pypdfium2(uploaded_files):
        """Load PDF documents using PyPDFium2"""
        docs = []
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(suffix=".pdf") as f:
                f.write(st.session_state.doc[uploaded_file.name])
                f.flush()
                doc = PyPDFium2Loader(f.name).load()
                for d in doc:
                    d.metadata["source"] = uploaded_file.name
                docs.extend(doc)

        # Add document ID to metadata
        for doc in docs:
            doc.metadata["doc_id"] = f"{doc.metadata['source']}_{doc.metadata['page']}"
        return docs


    @st.cache_resource
    def load_doc_with_docling(uploaded_file, page_number):
        """Load a specific page from a document using Docling"""
        # Determine file extension
        file_extension = os.path.splitext(uploaded_file)[1].lower()
        unique_suffix = f"_{uuid.uuid4()}{file_extension}"

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=unique_suffix, mode="wb"
        ) as temp_file:
            temp_file_path = temp_file.name

            if file_extension == ".pdf":
                # Extract specific page from PDF
                pdf_document = fitz.open(
                    stream=io.BytesIO(st.session_state.doc[uploaded_file]), filetype="pdf"
                )
                new_pdf_document = fitz.open()
                new_pdf_document.insert_pdf(
                    pdf_document, from_page=page_number - 1, to_page=page_number
                )
                new_pdf_document.save(temp_file_path)
                new_pdf_document.close()

                # Load the extracted page
                doc_loader = DoclingMultiFormatLoader([temp_file_path])
                documents = list(doc_loader.lazy_load())
                doc = documents[0] if documents else None
            else:
                # Handle other document types
                temp_file.write(st.session_state.doc[uploaded_file])
                temp_file.flush()

                doc_loader = DoclingMultiFormatLoader([temp_file_path])
                doc = None
                for doc_candidate in doc_loader.lazy_load():
                    if doc_candidate.metadata["page"] == page_number:
                        doc = doc_candidate
                        break

            # Update metadata if document was found
            if doc:
                doc.metadata["source"] = uploaded_file
                doc.metadata["page"] = page_number
                doc.metadata["doc_id"] = f"{doc.metadata['source']}_{doc.metadata['page']}"
                return doc

        return None


    def extract_documents_from_files(uploaded_files):
        """Extract and process documents from uploaded files"""
        documents = []

        for uploaded_file in uploaded_files:
            # Store document in session state
            st.session_state.doc[uploaded_file.name] = uploaded_file.read()

            # Create temporary file with unique name
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            unique_suffix = f"_{uuid.uuid4()}{file_extension}"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=unique_suffix
            ) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(st.session_state.doc[uploaded_file.name])
                temp_file.flush()

                # Process according to file type
                if file_extension == ".pdf":
                    doc_loader = PyPDFium2Loader(temp_file_path)
                    docs = doc_loader.load()
                else:
                    doc_loader = DoclingMultiFormatLoader([temp_file_path])
                    docs = doc_loader.lazy_load()

                # Add metadata and collect documents
                for d in docs:
                    d.metadata["source"] = uploaded_file.name
                    documents.append(d)

        # Add document ID to metadata for each document
        for doc in documents:
            doc.metadata["doc_id"] = f"{doc.metadata['source']}_{doc.metadata['page']}"

        return documents


    @st.cache_resource
    def initialize_language_model():
        """Initialize and cache the LLM"""
        return get_llm_answer()


    @st.cache_resource
    def get_embeddings(model_name):
        """Initialize and cache the embedding model"""
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, encode_kwargs={"batch_size": 32}
        )
        return embeddings


    @st.cache_resource
    def get_reranker_cross_encoder(model_name):
        """Initialize and cache the cross-encoder reranker model"""
        return HuggingFaceCrossEncoder(model_name=model_name)

    @st.cache_resource
    def setup_qa_system(_documents, _embeddings_model):
        """
        Set up the complete RAG pipeline with configurable components
        """
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            text_chunks = text_splitter.split_documents(_documents)
            
            # Set up BM25 retriever (keyword-based)
            bm25_retriever = BM25Retriever.from_documents(text_chunks, preprocess_func=preprocess_func)
            bm25_retriever.k = K_BM25
            
            # Set up vector store and semantic retriever
            vector_store = FAISS.from_documents(text_chunks, _embeddings_model)
            SEMANTIC_K = K_EMBEDDING if not USE_RERANKER else K_EMBEDDING * RERANKER_FILTER_FACTOR
            semantic_query_retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": SEMANTIC_K}
            )
            
            # Configure retrieval pipeline based on user settings
            if USE_RERANKER:
                # Add cross-encoder reranking
                compressor = CrossEncoderReranker(
                    model=get_reranker_cross_encoder(reranker_model_name_map[RERANKER_MODEL]), 
                    top_n=K_EMBEDDING
                )
                rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=semantic_query_retriever
                )
                # Combine BM25 and reranked semantic search
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, rerank_retriever],
                    weights=[WEIGHT_BM25, max(0.0, 1.0 - WEIGHT_BM25)],
                    id_key="doc_id",
                    search_kwargs={"k": K_FINAL}
                )
            else:
                # Combine BM25 and semantic search without reranking
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, semantic_query_retriever],
                    weights=[WEIGHT_BM25, max(0.0, 1.0 - WEIGHT_BM25)],
                    id_key="doc_id",
                    search_kwargs={"k": K_FINAL}
                )
                
            # Add multi-query retrieval if enabled
            if USE_MULTIQUERY_RETRIEVAL:
                ensemble_retriever_multiquery = MultiQueryRetriever.from_llm(
                    include_original=True,
                    retriever=ensemble_retriever,
                    llm=initialize_language_model()
                )
                
            # Add hypothetical document embeddings if enabled
            if USE_HYDE:
                hyp_retriever = HyDERetriever(
                    documents=_documents,
                    base_retriever=ensemble_retriever
                )
                
                if USE_MULTIQUERY_RETRIEVAL:
                    # Combine multi-query and HyDE
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[ensemble_retriever_multiquery, hyp_retriever],
                        weights=[max(0.0, 1.0 - WEIGHT_HYDE), WEIGHT_HYDE],
                        id_key="doc_id",
                        search_kwargs={"k": K_FINAL}
                    )
                else:
                    # Combine regular retriever and HyDE
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[ensemble_retriever, hyp_retriever],
                        weights=[max(0.0, 1.0 - WEIGHT_HYDE), WEIGHT_HYDE],
                        id_key="doc_id",
                        search_kwargs={"k": K_FINAL}
                    )
                    
            # Add selective page parsing if enabled
            if USE_SELECTIVE_PARSER:
                ensemble_retriever = SelectivePageParserRetriever(
                    base_retriever=ensemble_retriever,
                    k=K_PAGES_SELECTIVE_PARSER,
                    k_pdf=K_PDF_PAGES_SELECTIVE_PARSER
                )

            # Create the final QA chain
            return RetrievalQA.from_chain_type(
                llm=initialize_language_model(),
                chain_type="stuff",
                retriever=ensemble_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": CUSTOM_PROMPT},
            )
        except Exception as e:
            st.error(f"Error setting up QA system: {str(e)}")
            return None
    # Custom prompt for the QA chain
    CUSTOM_PROMPT_TEMPLATE = """
    Use the following pieces of context to answer the user question. If you don't know the answer, just say that you don't know.

    {context}

    Question: {question}

    Provide your response strictly in the following JSON format without extra text or markdown:

    {{
        "answer": "Your detailed answer here",
        "sources": ["Source sentence 1", "Source sentence 2"]
    }}
    """

    CUSTOM_PROMPT = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )

    # Model mappings
    embedding_model_name_map = {
        "bge-m3": "./models/bge-m3/",
        "multilingual-e5-large-instruct": "./models/multilingual-e5-large-instruct/",
    }

    reranker_model_name_map = {"bge-reranker-v2-m3": RERANKER_MODEL_NAME}


    def preprocess_result(result_str):
        result_str = result_str.strip()
        result_str = result_str.strip("```json").strip("```").strip("`")
        result_str = result_str.replace("\n", "").replace("\r", "")
        if not result_str.endswith("}"):
            result_str += "}"
        return result_str



    def clear_cache_and_session_state():
        """Clear all Streamlit cache and session state variables."""
        for key in st.session_state.keys():
            st.session_state.pop(key)
        st.session_state.clear()
        st.cache_resource.clear()


    # Reset button to clear all state and cached resources
    if st.button("Reset All"):
        clear_cache_and_session_state()
        st.rerun()  # Rerun the app to reset the UI
        st.write("App is resetting...")  # Should not be reached due to rerun

    # Main file uploader
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "html", "pptx", "ppt", "pptm", "md"],
        accept_multiple_files=True,
    )

    # Initialize session states
    if "doc" not in st.session_state:
        st.session_state.doc = {}

    if "processing_message" not in st.session_state:
        st.session_state.processing_message = ""

    if "qa_message" not in st.session_state:
        st.session_state.qa_message = ""

    # Display status messages
    if st.session_state.processing_message:
        st.success(st.session_state.processing_message)

    if st.session_state.qa_message:
        st.success(st.session_state.qa_message)

    # Document processing button
    if (
        st.button("Process Uploaded Documents")
        and uploaded_files is not None
        and len(uploaded_files) > 0
    ):
        # Get the selected embedding model
        embedding_name = embedding_model_name_map[embedding_method]

        # Process the uploaded files
        with st.spinner("Processing files..."):
            documents = extract_documents_from_files(uploaded_files)
            st.session_state.documents = documents
            st.session_state.processing_message = "Files processed successfully."

        # Set up the QA system if documents were successfully processed
        if documents:
            with st.spinner("Setting up QA system..."):
                qa_system = setup_qa_system(documents, get_embeddings(embedding_name))
                st.session_state.qa_system = qa_system
                st.session_state.qa_message = "QA system ready!"

                if qa_system is None:
                    st.error(
                        "Failed to set up QA system. Please check if LLM is running and try again."
                    )
                else:
                    st.success("QA system ready!")

    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! How can I assist you today? "}
        ]

    # Display chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input, disabled if QA system is not ready
    if "qa_system" in st.session_state:
        user_input = st.chat_input("Your message")
    else:
        user_input = st.chat_input("Your message", disabled=True)

    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Process the query if QA system is ready
        if "qa_system" in st.session_state:
            with st.spinner("Generating response..."):
                try:
                    # Ensure documents are loaded
                    if (
                        not hasattr(st.session_state, "documents")
                        or not st.session_state.documents
                    ):
                        st.error(
                            "Documents are not loaded. Please upload documents and click 'Process Uploaded Documents'."
                        )
                        st.stop()

                    # Query the QA system
                    result = st.session_state.qa_system.invoke({"query": user_input})

                    # Preprocess and parse the result
                    result["result"] = preprocess_result(result["result"])
                    parsed_result = json.loads(result["result"])
                except json.JSONDecodeError:
                    # Handle JSON parsing errors
                    st.error(f"JSON decoding error: {e}")
                    st.write("Raw LLM response:", result["result"])
                    st.error("There was an error parsing the response. Please try again.")
                    parsed_result = dict()
                    parsed_result["answer"] = result["result"]
                except Exception as e:
                    # Handle other errors
                    st.error(f"An unexpected error occurred: {str(e)}")

                # Extract source documents
                source_documents_text = "\n".join(
                    [doc.page_content for doc in result["source_documents"]]
                )
                answer = parsed_result["answer"]

                # Format source information
                sources_info = []
                for doc in result.get("source_documents", []):
                    source = doc.metadata["source"]
                    page = doc.metadata["page"]
                    sources_info.append(f"**Source:** {source}, **Page:** {page + 1}")
                formatted_sources = "\n".join(sources_info)

                # Create the combined response with answer and sources
                combined_response = f"{answer}\n\n**References:**\n{formatted_sources}"

                # Add to chat history and display
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": combined_response}
                )
                st.chat_message("assistant").write(combined_response)
                st.session_state.chat_occurred = True

            # Display source documents if chat has occurred
            if uploaded_files and st.session_state.get("chat_occurred", False):
                # Display selected pages header
                st.markdown(
                    "<h1 style='text-align: center;'>📑 Selected pages: </h1>",
                    unsafe_allow_html=True,
                )

                # Show PDF previews for cited sources
                displayed = set()
                for doc in result.get("source_documents", []):
                    source = doc.metadata["source"]
                    if source not in displayed:
                        if source.endswith(".pdf"):
                            # Get all pages from this source
                            pages_to_render = [
                                doc_sub.metadata["page"] + 1
                                for doc_sub in result["source_documents"]
                                if doc.metadata["source"] == doc_sub.metadata["source"]
                            ]

                            # Display the PDF with selected pages
                            pdf_viewer(
                                st.session_state.doc[source],
                                width=800,
                                height=800 * len(pages_to_render),
                                annotations=None,
                                pages_to_render=pages_to_render,
                            )
                        displayed.add(source)

                # Display document sources
                st.markdown(
                    "<h1 style='text-align: center;'>🔎 Documents sources: </h1>",
                    unsafe_allow_html=True,
                )
                st.markdown(source_documents_text)
        else:
            # Show error if QA system is not set up
            st.error(
                "QA system not set up. Please upload documents and click 'Process Uploaded Documents'."
            )

if __name__ == "__main__":
    main()
