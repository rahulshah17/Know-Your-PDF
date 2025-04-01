import os 
from huggingface_hub import login
HF_TOKEN = os.getenv("read_token")
login(HF_TOKEN)

import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch

# Load Embedding Model
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load LLaMA 3B Instruct LLM
llama_model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
model = AutoModelForCausalLM.from_pretrained(llama_model_id, device_map="auto", torch_dtype=torch.float32)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024, do_sample=True, temperature=0.7)
llm = HuggingFacePipeline(pipeline=pipe)

# Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# Build FAISS index from uploaded PDF
def build_faiss_index_from_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding_function)
    return vector_store

# Main RAG pipeline
def rag_query(pdf_file, question):
    if not pdf_file or not question.strip():
        return "Please upload a PDF and ask a question."

    # Extract text and create FAISS index
    text = extract_text_from_pdf(pdf_file)
    vector_store = build_faiss_index_from_text(text)

    # Create Retriever + QA chain
    retriever = vector_store.as_retriever(search_type="similarity", search_k=3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Run the query
    response = qa_chain.run(question)
    return response

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📄 RAG PDF QA with LLaMA 3B + LangChain")
    with gr.Row():
        pdf_file = gr.File(label="Upload your PDF", type="filepath")
        question = gr.Textbox(lines=1, placeholder="Ask a question about your PDF...")
    output = gr.Textbox(label="Answer", lines=4)

    submit_btn = gr.Button("Submit")
    submit_btn.click(rag_query, inputs=[pdf_file, question], outputs=output)

demo.launch()
