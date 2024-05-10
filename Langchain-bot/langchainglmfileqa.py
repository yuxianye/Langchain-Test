from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3
# from langchain.chains import retrieval_qa
# from langchain.document_loaders import  Dir

# tokenizer= AutoTokenizer.from_pretrained(r"D:\huggingface\THUDM\chatglm3-6b",trust_remote_code=True)

# model=AutoModel.from_pretrained(r"D:\huggingface\THUDM\chatglm3-6b",trust_remote_code=True).cuda()

# response,history= model.chat(tokenizer,"你好",history=[])
# print(response)

# response,history= model.chat(tokenizer,"晚上睡不着怎么办",history=history)
# print(response)
def load_documents(directory="books"):
    loader=DirectoryLoader(directory)
    documents=loader.load()
    text_spliter=CharacterTextSplitter(chunk_size=256,chunk_overlap=0)
    split_docs=text_spliter.split_documents(documents=documents)
    return split_docs

def load_embedding_mode(model_name="ernie-tiny"):
    encode_kwargs={"normalize_embeddings":False}
    model_kwargs={"device":"cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def store_chroma(docs,embeddings,persist_directory="D:\langchain-bot\VectorStore"):
    db= Chroma.from_documents(docs,embeddings,persist_directory=persist_directory)
    db.persist()
    return db


embeddings=load_embedding_mode('D:\\huggingface\\shibing624\\text2vec-base-chinese')
if not os.path.exists("D:\langchain-bot\VectorStore"):
    documents=load_documents()
    db=store_chroma(documents,embeddings)
else :
    db=Chroma(persist_directory="D:\langchain-bot\VectorStore",embedding_function=embeddings)

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"
llm = ChatGLM3(
        endpoint_url=endpoint_url,
        max_tokens=8096,
        # prefix_messages=messages,
        top_p=0.9,
        streaming=False, # Set to True for streaming completions

    )

retriever=db.as_retriever()
qa=RetrievalQA.from_chain_type(
    llm=llm,
    #chain_type='stuff'
    retriever=retriever
)

# response=qa.run("三国演义一共有多少回")
# response=qa.run("《三国演义（白话文版）.txt》文件一共多少文字")

# print (response)
response=qa.run("Project 1 Assigned to 给谁了")
# print (response)
# response=qa.run("Project 2 Assigned to 给谁了")
print (response)