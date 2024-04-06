import streamlit as st
from file_utils import read_uploaded_file
from rag import load_corpus,rag,process_query
import time
import torch
import gc
import json 
import pickle
from llama_index.core import SimpleDirectoryReader 

@st.cache_resource
def process_data(uploaded_file):
    docs = read_uploaded_file(uploaded_file)
    corpus  = load_corpus(docs)
    retriever = rag(corpus)
    return retriever
def main():
    gc.collect()

    st.title("Chat with your PDF Document(s)")

    
    uploaded_file = None
    # Upload Documents
    
    st.sidebar.header("Upload Documents. The system will notify you when it is done processing")
    # def procee_document_callback():

    uploaded_file = st.sidebar.file_uploader("Choose a document", type=["txt", "pdf"])
    if uploaded_file != None:
        if "uploaded" not in st.session_state:
            st.session_state['uploaded'] = {}
        # if uploaded_file.name not in  st.session_state['uploaded']:
        retriever = process_data(uploaded_file)
        st.sidebar.success("Document processed successfully!")
        st.session_state['uploaded'][uploaded_file.name] = retriever
        # else:
        #     retriever = st.session_state['uploaded'][uploaded_file.name]
    # if uploaded_file != None:
    #     if uploaded_file.name not in st.session_state['uploaded']:
    #         print("found file")
    #         doc = read_uploaded_file(uploaded_file)
    #         corpus  = load_corpus(doc)
    #         retriever = rag(corpus)
    #         st.sidebar.success("Document processed successfully!")
    #         # st.session_state.uploaded = True
    #         # retrievers[uploaded_file.name] = retriever
    #         # pickle.dump(retrievers,open(retriever_file_path,'wb'))
            
    #         st.session_state.uploaded[uploaded_file.name] = retriever
    #         gc.collect()
    #     else:
    #         print("retrieved")
    #         retriever =  st.session_state.uploaded[uploaded_file.name] 
    # Column 2: Chat Screen
    # with col2:
    st.header("Chat Screen")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Hello! How can I help today"):
            # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            time.sleep(2)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        if uploaded_file==None:
            response = "Kindly upload a document"
        else:
            response = process_query(prompt,retriever)
            # response = "Success"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        del retriever
        torch.cuda.empty_cache()
        
        
        
     



if __name__ == "__main__":
    main()
