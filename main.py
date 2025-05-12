from qa_chain import create_vector_store, load_qa_chain
import os

DATA_DIR = "data"

# Always recreate the vector store to ensure latest data is indexed
print("[+] Creating vector store from PDFs...")
vectorstore = create_vector_store(DATA_DIR)
print("[+] Vector store created successfully!")

# Initialize QA chain
print("[+] Loading QA chain from Pinecone...")
qa = load_qa_chain()

while True:
    q = input("\nAsk a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    answer = qa(q)
    print("\n📘 Answer:\n", answer)
