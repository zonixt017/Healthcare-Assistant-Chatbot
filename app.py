import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

@st.cache_resource
def load_resources():
    #create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device':"cpu"})

    # Check if vectorstore exists
    if os.path.exists("vectorstore"):
        vector_store = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        return vector_store

    #load the pdf files from the path
    loader = DirectoryLoader('data/',glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()

    #split text into chunks
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    #vectorstore
    vector_store = FAISS.from_documents(text_chunks,embeddings)
    vector_store.save_local("vectorstore")
    return vector_store

vector_store = load_resources()

@st.cache_resource
def load_llm():
    # Check for Hugging Face API token
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.1,
            max_length=128
        )
    else:
        #create llm
        # Replaced CTransformers with LlamaCpp (requires llama-cpp-python)
        # Note: GGML is deprecated. Please download a GGUF model (e.g., llama-2-7b-chat.Q4_K_M.gguf) and install llama-cpp-python
        llm = LlamaCpp(
            model_path="llama-2-7b-chat.Q4_K_M.gguf",
            temperature=0.01,
            max_tokens=128,
            n_ctx=2048
        )
    return llm

llm = load_llm()

st.title("HealthCare ChatBot 🧑🏽‍⚕️")

# Setup Chain (LCEL)
# 1. Contextualize question: Reformulate the question based on history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, vector_store.as_retriever(search_kwargs={"k": 2}), contextualize_q_prompt
)

# 2. Answer question: Use retrieved context to answer
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello My Friend! Ask me anything about 🤗"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about your Mental Health"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Convert Streamlit messages to LangChain format for history
            history_messages = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))
            
            result = rag_chain.invoke({"input": prompt, "chat_history": history_messages})
            response = result["answer"]
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            #ig