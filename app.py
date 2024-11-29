import streamlit as st
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

# Maintain conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def get_pdf_text(pdf_path):
    combined_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text
            page_text = page.extract_text()
            combined_text += f"\n\n### Page {page_num} Text Start ###\n{page_text}\n### Page {page_num} Text End ###\n"
            
            # Extract tables
            page_tables = page.extract_tables()
            for table_num, table in enumerate(page_tables, start=1):
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    csv_text = df.to_csv(index=False)
                    combined_text += f"\n### Page {page_num} Table {table_num} Start ###\n{csv_text}\n### Page {page_num} Table {table_num} End ###\n"
    return combined_text



def get_text_chunks(text):
    """
    Splits the input text into chunks with a specified size and overlap.
    This ensures that context is preserved across chunks.
    """
    # Create a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    # Split the text into chunks and return them
    return text_splitter.split_text(text)

def get_vector_store(chunks):
    """
    Creates a FAISS vector store from text chunks and saves it locally.
    """
    
    # Use a pre-trained HuggingFace model to generate embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a FAISS vector store from the text chunks and their embeddings
    vector_store = FAISS.from_texts(chunks, embedding_model)
    
    # Save the vector store locally for reuse
    vector_store.save_local("faiss_index")
    
    # Return the vector store
    return vector_store

def get_conversational_chain():
    """
    Sets up a conversational chain for answering questions based on a context.
    """
    
    # Define the prompt template
    prompt_template = '''
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "Answer is not available in the provided PDF."
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    '''
    # Load the Llama 2 chat model
    model_name = OllamaLLM(model="llama2:7b-chat-q2_K")
    
    # Wrap the prompt template for use in a chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create and return the Q&A chain
    return load_qa_chain(model_name, chain_type="stuff", prompt=prompt)

def user_input(question):
    """
    Handles user input to search the vector store and generate an answer.
    """
    
    # Initializing the same embedding model used for the vector store
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Define a function to encode the user query
    def embed_query(text):
        return embedding_model.encode([text])[0]
    
    # Load the saved FAISS vector store with the query embedding function
    new_db = FAISS.load_local("faiss_index", embed_query, allow_dangerous_deserialization=True)
    
    # Search the vector store for relevant chunks
    docs = new_db.similarity_search(question)
    
    # Load the conversational chain
    chain = get_conversational_chain()
    
    # Generate a response using the conversational chain
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
    # Return the generated response
    return response["output_text"]


def main():
    st.title("Chat with PDFs 💬")

    # File upload section
    with st.sidebar:
        uploaded_pdf = st.file_uploader("Upload your PDF here", type=["pdf"])
        if uploaded_pdf and st.button("Submit"):
            with st.spinner("Processing PDF..."):
                pdf_text = get_pdf_text(uploaded_pdf)
                chunks = get_text_chunks(pdf_text)
                get_vector_store(chunks)
                st.success("PDF processed and indexed!")

    # Chat interface
    st.subheader("powered by LLAMA 2 @Meta AI")
    user_question = st.text_input("Type your message...")

    if user_question:
        # Add user message to session state and display it
        st.session_state["messages"].append({"role": "user", "content": user_question})
        
        # Get response from the LLM
        with st.spinner("Thinking..."):
            bot_response = user_input(user_question)

        # Add bot response to session state
        st.session_state["messages"].append({"role": "bot", "content": bot_response})

    # Display all messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()

