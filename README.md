# Chat with PDFs using LLAMA 2 by Meta AI
# 🚀 About the Project
This project allows users to upload a PDF document and chat with it using the powerful LLAMA 2 language model by Meta AI. The application extracts the text and tables from PDFs, processes the data, and answers questions based on the content of the document. It's perfect for summarizing documents, extracting insights, or interacting with complex text-based files.

# 🛠 Features
PDF Upload:                     Upload PDF files to interact with their content.
Text and Table Extraction:      Extracts both plain text and tabular data from the PDF.
Conversational Interface:       Chat with the PDF content using a question-and-answer interface.
Embeddings and Vector Search:   Uses sentence-transformers for semantic search and FAISS for vector storage.
Powered by LLAMA 2:             Utilizes the LLAMA 2 7B Chat model for generating answers.

# 🧩 Prerequisites
Before running the application, make sure you have the following installed:

Python version between 3.8 and 3.10
pip (Python package manager)
download ollama from below 
https://ollama.com/download


# 📦 Installation and Setup

# Clone the Repository
- git clone https://github.com/ItsAnkitRajput/Chat-with-your-PDFs.git
- cd Chat-with-your-PDFs

# Create a Virtual Environment
- python -m venv myenv
- source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install Dependencies
- pip install -r requirements.txt

# Download the LLAMA 2 Model
- https://ollama.com/download
- Go to terminal and type "ollama run "name of model""
- ollama run llama2:7b-chat-q2_K
- once its installed press ctrl+q
- type "ollama serve" to start the llm serve locally
- if a instance of llm is already running then it will throw an error which you can ignore

# Run the Application
streamlit run app.py

# 📂 Project Structure
Chat-with-your-PDFs/
├── app.py                     # Main application code
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Ignored files list
├── example_data/              # Sample data folder
│   └── Marcus_Aurelius.pdf    # Example PDF file
├── screenshots/               # Screenshots of the project

# 🧪 How to Use
- Run the application with the streamlit command.
- Upload a PDF file using the drag-and-drop feature or the file browser.
- Type your questions in the chat interface to interact with the PDF's content.

# ⚙️ Technologies Used
Streamlit: For building the web application interface.

FAISS: For efficient vector similarity search.

LangChain: To manage chains for question answering.

Sentence-Transformers: For generating embeddings from text.

LLAMA 2: Language model used for generating responses.


# 📬 Contact
Author: Ankit Raput

Email: work.itsankitrajput.email@gmail.com

GitHub: [ItsAnkitRajput](https://github.com/ItsAnkitRajput)







