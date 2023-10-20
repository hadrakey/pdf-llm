#coding part
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import  HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import pickle
import os

from dotenv import load_dotenv
import base64

#load api key lib
os.environ["OPENAI_API_KEY"] = "...."

#Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('images.jpeg')  

#sidebar contents

with st.sidebar:
    st.title('🦜️🔗CATIE - PDF BASED LLM-LANGCHAIN CHATBOT🤗')
    st.markdown('''
    ## About APP:

    The app's primary resources:

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [openai](https://openai.com/gpt-4)

    ## About us:

    - [CATIE](https://www.catie.fr/en/home/)
    
    ''')

    add_vertical_space(4)
    st.sidebar.image("./Logo_CATIE_CMJN.png", use_column_width=True)
    # st.write('💡All about pdf based chatbot, created by CATIE🤗')

load_dotenv()

def main():
    st.header("📄Chat with your pdf file🤗")

    #upload your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    # st.write(pdf)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        #Once we have the document content, the next step is to convert that into fixed-sized
    #  chunks so that the text fits into our choice-of-models context window.
    #  We’ll use RecursiveCharacterTextSplitter with a chunk size of 1000
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 150,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            embeddings = OpenAIEmbeddings()
            
            # embeddings = HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large')

            #Store the chunks part in db (vector)
            # Once we have broken the document down into chunks, next step is to create embeddings for 
            # the text and store it in vector store. We can do it as shown below.
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        #st.write(chunks)
        
        #Accept user questions/query

        query = st.text_input("Ask questions about related your upload pdf file")
        #st.write(query)

        if query:
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            # Here we define the type of model in the argument model_name. ef model_name = "gpt-4"
            chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),  retriever=vectorstore.as_retriever(), memory=memory)
            history = []
            
            # result = chain({"question": query, 'chat_history': history}, return_only_outputs=True)


            # docs = vectorstore.similarity_search(query=query,k=3)
            # # llm = HuggingFaceHub(repo_id="CATIE-AQ/frenchT0",
            # #            model_kwargs={"temperature": 10, "max_length":200})
            # # chain = load_qa_chain(llm=llm, chain_type="refine")
            # # response = chain({"input_documents": docs, "question": query})
            # #st.write(docs)
            
            # #openai rank lnv process
            # llm = OpenAI(model_name="gpt-4",temperature=0)
            # chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response =  chain({"question": query, 'chat_history': history}, return_only_outputs=True)
                print(cb)
            st.write(response["answer"])



if __name__=="__main__":
    main()