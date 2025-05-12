from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load and verify environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

print("Environment check:")
print(f"API Key exists: {bool(api_key)}")
print(f"Index name: {index_name}")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

try:
    # List indexes to verify connection
    print("\nListing available indexes:")
    print(pc.list_indexes())
    
    # Get index instance
    print(f"\nTrying to connect to index: {index_name}")
    index = pc.Index(index_name)

    # Test data
    test_data = [{
        "id": "test1",
        "text": "This is a test document for Pinecone indexing."
    }]

    print("\nCreating test embedding...")
    embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[d['text'] for d in test_data],
        parameters={"input_type": "passage"}
    )

    vectors = []
    for d, e in zip(test_data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e.values,
            "metadata": {'text': d['text']}
        })

    print("Upserting test vector to Pinecone...")
    print(f"Vector dimension: {len(vectors[0]['values'])}")
    result = index.upsert(vectors=vectors, namespace="")
    print(f"Upsert result: {result}")

    # Query to verify
    print("\nQuerying the index...")
    query = "test document"
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query"}
    )

    results = index.query(
        namespace="",
        vector=query_embedding[0].values,
        top_k=1,
        include_values=False,
        include_metadata=True
    )

    print("\nQuery results:")
    print(results)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    raise
