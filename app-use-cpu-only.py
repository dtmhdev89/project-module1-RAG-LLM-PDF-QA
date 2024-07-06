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
device = torch.device('cpu')

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

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

print("Tokenized stage")
tokenized_quantized_model = "../AI_model_data/tokenized_quantized_model"
tokenizer = AutoTokenizer.from_pretrained(tokenized_quantized_model)

# print("Quantized model stage")
# quantized_model = "../AI_model_data/quantized_model_cpu"
# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model = AutoModelForCausalLM.from_pretrained(quantized_model, quantization_config=nf4_config, use_safetensors=True, local_files_only=True, low_cpu_mem_usage=True, device=device)

# # Load the saved state dictionary directly
# model.load_state_dict(torch.load(f"{quantized_model}/pytorch_model.bin", map_location=device))

# # Set model to evaluation mode and ensure it's on CPU
# model.eval()
# model.to(device)

quantized_model = "../AI_model_data/quantized_model_cpu"

model_pipeline = pipeline(
  "text-generation",
  model=quantized_model,
  use_safetensors=True,
  tokenizer=tokenizer,
  max_new_tokens=512,
  pad_token_id=tokenizer.eos_token_id,
  device_map="cpu",
  device=device
)

# Test model_pipeline
print("Test model pipeline")
output = model_pipeline("Once upon a time")[0]['generated_text']
print(output)

