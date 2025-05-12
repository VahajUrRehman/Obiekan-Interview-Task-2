import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from hf_clients import embed_text, HFChatLLM
import uuid
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
print("\n[+] Initializing Pinecone...")
pc = Pinecone(api_key=api_key)

# List available indexes
print("\n[+] Available Pinecone indexes:")
indexes = pc.list_indexes()
print(indexes)

index = pc.Index(index_name)

def create_vector_store(pdf_folder):
    try:
        # Load PDFs with explicit encoding
        print("\n[+] Loading PDFs...")
        docs = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                print(f"\n[+] Processing {filename}...")
                file_path = os.path.join(pdf_folder, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    print(f"    - Loaded {len(loaded_docs)} pages")
                    
                    if loaded_docs:
                        # Print the first 200 characters of content for verification
                        first_doc = loaded_docs[0]
                        content_preview = first_doc.page_content[:200].strip()
                        print(f"    - First page preview: {content_preview}")
                        
                        # Only extend if we have valid content
                        if content_preview:
                            docs.extend(loaded_docs)
                        else:
                            print("    ! Warning: Empty content detected")
                    
                except Exception as e:
                    print(f"    ! Error loading {filename}: {str(e)}")
                    continue

        if not docs:
            raise ValueError("No valid documents were loaded!")

        # Split documents with more aggressive settings
        print(f"\n[+] Creating chunks from {len(docs)} pages...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        split_docs = splitter.split_documents(docs)
        
        if not split_docs:
            raise ValueError("No chunks were created from the documents!")
        
        print(f"    Created {len(split_docs)} chunks")
        print(f"    Sample chunk: {split_docs[0].page_content[:200]}")

        # Instead of clearing vectors, we'll just upsert new ones
        print("\n[+] Processing chunks...")
        batch_size = 5
        successful_uploads = 0
        
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(split_docs) + batch_size - 1)//batch_size
            print(f"\n[+] Processing batch {batch_num}/{total_batches}")
            
            try:
                # Create embeddings
                print("    Creating embeddings...")
                texts = [doc.page_content for doc in batch]
                embeddings = pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=texts,
                    parameters={"input_type": "passage"}
                )

                # Create vectors
                vectors = []
                for doc, emb in zip(batch, embeddings):
                    vector_id = str(uuid.uuid4())
                    vectors.append({
                        "id": vector_id,
                        "values": emb.values,
                        "metadata": {
                            "text": doc.page_content,
                            "source": doc.metadata.get('source', ''),
                            "page": doc.metadata.get('page', 0)
                        }
                    })

                # Upsert to Pinecone
                print(f"    Upserting {len(vectors)} vectors...")
                result = index.upsert(vectors=vectors)
                successful_uploads += len(vectors)
                print(f"    ✓ Upserted {len(vectors)} vectors successfully")
                
                # Small delay between batches
                time.sleep(1)

            except Exception as e:
                print(f"    ! Error in batch {batch_num}: {str(e)}")
                continue

        if successful_uploads == 0:
            raise ValueError("No vectors were successfully uploaded to Pinecone!")

        print(f"\n[✓] Vector store creation completed!")
        print(f"    - Total chunks processed: {len(split_docs)}")
        print(f"    - Successfully uploaded: {successful_uploads}")
        
        # Create and return the vector store
        return PineconeVectorStore(
            index_name=index_name,
            embedding=HFCustomEmbeddings(),
        )

    except Exception as e:
        print(f"\n[!] Error in create_vector_store: {str(e)}")
        raise

class HFCustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        result = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=texts,
            parameters={"input_type": "passage"}
        )
        return [r.values for r in result]
    
    def embed_query(self, text):
        result = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[text],
            parameters={"input_type": "query"}
        )
        return result[0].values

def load_qa_chain():
    print("\n[+] Loading QA chain...")
    try:
        embeddings = HFCustomEmbeddings()
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        llm = HFChatLLM()
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        def run_qa(query):
            try:
                result = qa_chain({"question": query})
                return result["answer"]
            except Exception as e:
                print(f"\n[!] Error in QA: {str(e)}")
                return f"Error: {str(e)}"
        
        return run_qa
        
    except Exception as e:
        print(f"\n[!] Error in load_qa_chain: {str(e)}")
        raise
