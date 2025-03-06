import chainlit as cl
import torch
import os
import yaml
# from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.docstore.document import Document
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain import hub


def create_document_annotations(doc_folder_path, force_update=False):
    """
    Create a document annotation file for the given document folder path.
    If the file already exists and force_update is False, the function will return the existing file.
    """
    if os.path.exists("dataset/doc_annotation.yaml") and not force_update:
        return

    documents = list()
    if os.path.isdir(doc_folder_path):
        for file in os.listdir(doc_folder_path):
            if file.endswith(".txt") or file.endswith(".pdf") or \
               file.endswith(".doc") or file.endswith(".docx"):
                file_name = file.split(".")[0]
                file_path = os.path.join(doc_folder_path, file)
                documents.append({"title": file_name, "file_path": file_path})

        with open("dataset/doc_annotation.yaml", "a", encoding="utf-8") as f:
            yaml.dump({"document_type": {"collection": documents}}, f)
    else:
        raise ValueError(f"Invalid document folder path: {doc_folder_path}")

    return documents


def process_file():
    """
    Process the documents in the document folder path and return the restructured documents.
    """
    if os.path.exists("dataset/doc_annotation.yaml"):
        with open("dataset/doc_annotation.yaml", "r", encoding="utf-8") as f:
            document_annotation = yaml.safe_load(f)
            doc_collection = document_annotation["document_type"]["collection"]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            restructured_documents = []
            for doc_item in doc_collection:
                if doc_item["file_path"].endswith(".txt"):
                    loader = TextLoader(doc_item["file_path"])
                elif doc_item["file_path"].endswith(".pdf"):
                    loader = PyPDFLoader(doc_item["file_path"])
                elif doc_item["file_path"].endswith(".docx") or \
                        doc_item["file_path"].endswith(".doc"):
                    loader = Docx2txtLoader(doc_item["file_path"])
                else:
                    raise ValueError(f"Unsupported file type: {doc_item['file_path']}")

                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk.page_content,
                        metadata={"title": doc_item['title']}
                    )
                    restructured_documents.append(doc)
    else:
        raise ValueError("Document annotation file not found")

    return restructured_documents


def get_vector_db(
        docs: list[Document],
        embedding: HuggingFaceEmbeddings,
        persist_directory: str):
    # cl.user_session.set("docs", docs)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory)

    return vector_db


def get_huggingface_llm(
        model_name: str = "lmsys/vicuna-7b-v1.5",
        max_new_token: int = 512):
    if model_name == "lmsys/vicuna-7b-v1.5":
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True
        )
        
        task = "text-generation"
    elif model_name == "google/flan-t5-small":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True)
        task = "text-generation"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="cpu"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )

    return llm


create_document_annotations(
    doc_folder_path="dataset/docs",
    force_update=False)
docs = process_file()
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)
print("********Loading LLM********")
LLM = get_huggingface_llm(model_name="google/flan-t5-small")
# LLM = get_huggingface_llm(model_name="distilgpt2")

vector_db_path = os.path.join(os.path.dirname(__file__), "vector_db")
os.makedirs(vector_db_path, exist_ok=True)
print("********Creating Vector DB********")
vector_db = get_vector_db(docs, embedding, vector_db_path)
print("********Vector DB Created********")
welcome_message = """ Welcome to the User support system! To get started:
Ask a question about system's documents
"""

@cl.on_chat_start
async def on_chat_start():
    print("on_chat_start")
    msg = cl.Message(content=welcome_message)
    await msg.send()

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 2}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    print("********On Message********")
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    print("********Response********")
    print(res.get('source_documents'))
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for _source_idx, source_doc in enumerate(source_documents):
            print("********Source Document********")
            print(source_doc.page_content)
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=source_doc.metadata.get('title'))
            )

        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()



# !chainlit run app.py --host 0.0.0.0 --port 8000 &>/logs/chainlit_log.txt &

# import urllib

# print("Password/Enpoint IP for localtunnel is:",
#   urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

# !lt --port 8000 --subdomain aivn-simple-rag
