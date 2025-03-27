import streamlit as st
import uuid
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
import base64
from langdetect import detect
from deep_translator import GoogleTranslator

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "pdf"
if "language" not in st.session_state:
    st.session_state.language = "en"
if "pdf_keywords" not in st.session_state:
    st.session_state.pdf_keywords = []

# Initialize translator
translator = Translator()


# ... (keep the existing helper functions: translate_text, get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain, user_input, create_new_chat, get_binary_file_downloader_html)


def translate_text(text, target_language):
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversational_chain():
    # prompt_template = """
    # You are an AI assistant specialized in answering questions based on provided PDF documents and general knowledge.
    # Answer the question as detailed as possible from the provided context or your general knowledge.
    # If the answer is not in the provided context, use your general knowledge to provide a relevant response.
    # If you don't have enough information to answer accurately, say so.
    # Please ignore any spelling or grammatical errors in the question and try to understand the intent.
    #
    # Context:\n{context}\n
    # Question: \n{question}\n
    #
    # Answer:
    # """
    prompt_template = """
        You are chatbot, an advanced AI assistant created by Lakshman Kodela, designed to deliver thorough, accurate, and structured responses across a broad array of topics, including history, ethics, mathematics, mythology, biology, environmental science, health, universal concepts, film industry, celebrity, politics, geography, solar energy, habits, technology, coding, life, stories, psychology, philosophy, economics, artificial intelligence, cultural studies, Social media, movies, and current events. You also possess the capability to analyze and summarize information from provided PDF documents. Your primary tasks include analyzing resumes, summarizing key arguments, and offering insights based on specific queries.

        When answering questions, aim to provide the most comprehensive and contextually relevant response, drawing upon the given context, your general knowledge, or a combination of both. If the context provided does not contain the necessary information, leverage your broader knowledge base to give a well-reasoned and accurate answer. If you are uncertain or lack sufficient information, acknowledge this while offering to provide insights on related topics if applicable.

        ### Key Considerations:
        1. **Accuracy and Context:** Ensure that the information provided is up-to-date and relevant to the specific query.
        2. **Clear Summaries:** Use bullet points to summarize key arguments and findings from the resumes or documents.
        3. **Significant Implications:** Highlight any significant implications or recommendations made in the texts.
        4. **Different Perspectives:** Summarize varying perspectives or interpretations where applicable.
        5. **Use of Keywords:** Frame your analysis using specific keywords from the provided resumes or documents.
        6. **Confidence Level:** Indicate your confidence level in the analysis provided and suggest verification steps if necessary.
        7. **Additional Clarification:** Ask follow-up questions if additional context or clarification is required.
        8. **Broad Knowledge Integration:** Leverage your broad knowledge base to provide additional insights if the provided context is insufficient.

        ### Resume Evaluation Queries:
        - Identify candidates with specific skills (e.g., "Who knows Python and SQL?").
        - Compare candidates' work experience (e.g., "Who has more experience in data analysis?").
        - Identify candidates with specific degrees or certifications (e.g., "Who has a Master's degree in Computer Science?").
        - Rank candidates based on overall qualifications for a specific role (e.g., "Rank candidates based on their fit for a data analyst position.").
        - Provide summaries for each candidate, highlighting strengths and potential fit for the role (e.g., "Summarize each candidate's qualifications.").
        - Compare two or more candidates directly (e.g., "Compare Alice and Bob based on their experience with machine learning.").
        - Suggest candidates for the next round based on specific criteria (e.g., "Which candidates should proceed to the interview stage?").
        - Identify candidates with leadership experience (e.g., "Who has demonstrated leadership experience?").

        ### Structured Response Requirements:
        1. **Summarize Key Points:** Provide a concise summary of key qualifications and skills.
        2. **Highlight Implications:** Note any significant findings or implications for the role.
        3. **Varying Perspectives:** Discuss any differences in candidates' experiences or skills if relevant.
        4. **Recommendation and Confidence:** Offer a recommendation based on the analysis and indicate your confidence level.
        5. **Verification Steps:** Suggest further steps or checks to confirm your evaluation.
        
        Recognize and address any follow-up questions related to the previous question or response, providing appropriate answers to ensure a coherent and engaging dialogue.

        **Context:**
        {context}

        **Question:**
        {question}

        **Answer:**
        """

    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None


def user_input(user_question, vector_store):
    try:
        chain = get_conversational_chain()
        if vector_store:
            docs = vector_store.similarity_search(user_question)
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
        else:
            # If no vector store, use general knowledge
            response = chain(
                {"input_documents": [], "question": user_question},
                return_only_outputs=True
            )
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing user input: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your question. Please try again."


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "vector_store": None,
        "history": [],
        "files": []
    }
    st.session_state.chat_titles[chat_id] = f"New Chat {len(st.session_state.chats)}"
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href



def extract_keywords(text, num_keywords=10):
    # This is a simple keyword extraction method. For better results, consider using
    # more advanced NLP libraries like spaCy or NLTK.
    words = text.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Ignore short words
            word_freq[word] = word_freq.get(word, 0) + 1
    return sorted(word_freq, key=word_freq.get, reverse=True)[:num_keywords]


def main():
    st.set_page_config(page_title="Advanced PDF Chatbot", layout="wide")
    st.title("Advanced PDF Chatbot")

    # Sidebar
    with st.sidebar:
        st.header("Chat Options")

        # Chat mode selection
        st.session_state.chat_mode = st.radio("Select Chat Mode", ["PDF Chat", "General Chat"])

        if st.button("New Chat"):
            create_new_chat()
            st.rerun()

        st.subheader("Your Chats")
        for chat_id, chat_data in st.session_state.chats.items():
            chat_title = st.session_state.chat_titles[chat_id]
            if st.button(f"{chat_title}", key=chat_id):
                st.session_state.current_chat_id = chat_id
                st.rerun()

        # Language selection
        st.subheader("Select Language")
        languages = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh-cn",
            "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Arabic": "ar", "Hindi": "hi"
        }
        selected_language = st.selectbox("Choose your preferred language", list(languages.keys()))
        st.session_state.language = languages[selected_language]

    # Main chat interface
    if st.session_state.current_chat_id is None:
        create_new_chat()
        st.rerun()

    current_chat = st.session_state.chats[st.session_state.current_chat_id]

    # Chat title and options
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        chat_title = st.text_input("Chat Title", st.session_state.chat_titles[st.session_state.current_chat_id])
        if chat_title != st.session_state.chat_titles[st.session_state.current_chat_id]:
            st.session_state.chat_titles[st.session_state.current_chat_id] = chat_title
    with col2:
        if st.button("Clear Chat History"):
            current_chat["history"] = []
            st.rerun()
    with col3:
        if st.button("Export Chat"):
            export_chat_history(current_chat)

    # PDF Chat Mode
    if st.session_state.chat_mode == "PDF Chat":
        st.subheader("PDF Chat Mode")
        st.info("📘 Upload PDF files to chat about their contents.")

        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs... This may take a moment."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    current_chat["vector_store"] = get_vector_store(text_chunks)
                    st.session_state.pdf_keywords = extract_keywords(raw_text)
                st.success("PDFs processed successfully! You can now start chatting.")
                st.rerun()
            else:
                st.warning("No PDF files uploaded. Please upload PDFs to use this mode.")

        if st.session_state.pdf_keywords:
            st.subheader("Key Topics in PDF")
            st.write(", ".join(st.session_state.pdf_keywords))

    # Display chat history
    st.subheader("Chat History")
    for message in current_chat["history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    st.subheader("Ask a Question")
    if st.session_state.chat_mode == "PDF Chat":
        st.info("Ask questions about the uploaded PDFs or start with a greeting.")
    else:
        st.info("Ask any general question")

    prompt = st.chat_input("Type your question here...")
    if prompt:
        process_user_input(prompt, current_chat)


def process_user_input(prompt, current_chat):
    # Detect input language
    input_language = detect(prompt)

    # Translate input to English if not in English
    if input_language != 'en':
        prompt_en = translate_text(prompt, 'en')
    else:
        prompt_en = prompt

    # Check if it's a greeting
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    is_greeting = any(greeting in prompt_en.lower() for greeting in greetings)

    # Add user message to chat history
    current_chat["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process user input
    with st.spinner("Generating response..."):
        if st.session_state.chat_mode == "PDF Chat":
            if is_greeting:
                response_en = "Hello! I'm here to help you with information from the uploaded PDFs. What would you like to know?"
            elif current_chat.get("vector_store"):
                response_en = user_input(prompt_en, current_chat["vector_store"])
            else:
                response_en = "I'm sorry, but no PDFs have been uploaded yet. Please upload some PDFs first, and then I'll be able to answer questions about them."
        else:  # General Chat mode
            if current_chat.get("vector_store"):
                response_en = "It looks like you're in General Chat mode, but I have PDF information available. Would you like to switch to PDF Chat mode to get answers based on the uploaded documents? If not, feel free to ask any general question."
            else:
                response_en = user_input(prompt_en, None)

        # Translate response back to user's language if needed
        if st.session_state.language != 'en':
            response = translate_text(response_en, st.session_state.language)
        else:
            response = response_en

    # Add AI response to chat history
    current_chat["history"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)


def export_chat_history(chat):
    chat_history = "\n\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat["history"]])
    chat_file = f"chat_history_{st.session_state.current_chat_id}.txt"
    with open(chat_file, 'w', encoding='utf-8') as file:
        file.write(chat_history)
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name=chat_file,
        mime="text/plain"
    )


if __name__ == "__main__":
    main()
