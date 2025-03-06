import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

Loader = PyPDFLoader
FILE_PATH = "./aio2024_sample.pdf"
loader = Loader(FILE_PATH)
documents = loader.load()

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(" Number of sub-documents: ", len(docs))
print(docs[0])

# Vectorize
embedding = HuggingFaceEmbeddings()
vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vector_db.as_retriever()

# Sample
result = retriever.invoke("What is YOLO ?")
print ("Number of relevant documents: ", len(result))

# LLM vicuna
# This quantization code couldn't be run in a computer without CUDA
print("=====quantization")
bit8_config = BitsAndBytesConfig(
  load_in_8bit=True,
  llm_int8_enable_fp32_cpu_offload=True
)

device_map = {
  "transformer.word_embeddings": 0,
  "transformer.word_embeddings_layernorm": 0,
  "lm_head": "cpu",
  "transformer.h": 0,
  "transformer.ln_f": 0,
}

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  quantization_config=bit8_config,
  low_cpu_mem_usage=True,
  device_map=device_map,
  use_cache=False
)

print("=====tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Make pipeline: tokenizer and model
print("=====model_pipeline")

model_pipeline = pipeline(
  "text-generation",
  model=model ,
  tokenizer=tokenizer,
  max_new_tokens=512,
  pad_token_id=tokenizer.eos_token_id,
  device_map="auto"
)

# Test model_pipeline
print("Test model pipeline")
output = model_pipeline("Once upon a time")[0]['generated_text']
print(output)

print("=====llm HuggingFacePipeline")
llm = HuggingFacePipeline(pipeline=model_pipeline)

# Prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs) :
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
  {" context ": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

USER_QUESTION = "YOLOv10 là gì?"
output = rag_chain.invoke(USER_QUESTION)
answer = output.split('Answer :')[1].strip()
print(answer)
