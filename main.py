from qa_chain import create_vector_store, load_qa_chain
import os
import json

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
        
    result = qa(q)
    
    print("\n📘 Answer:")
    print(result["answer"])
    
    if result["evaluation"]:
        print("\n⚖️ Judge's Evaluation:")
        evaluation = result["evaluation"]
        if isinstance(evaluation, str):
            try:
                # Try to parse if it's a JSON string
                evaluation = json.loads(evaluation)
            except json.JSONDecodeError:
                # If not valid JSON, print as is
                print(evaluation)
                continue
                
        # Now evaluation should be a dict
        print(f"Score: {evaluation.get('score', 'N/A')}/10")
        print(f"Reasoning: {evaluation.get('reasoning', 'N/A')}")
        print(f"Suggestions: {evaluation.get('suggestions_for_improvement', 'N/A')}")
