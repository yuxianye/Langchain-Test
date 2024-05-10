from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
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


