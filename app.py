from flask import Flask, render_template, request, jsonify
from qa_chain import create_vector_store, load_qa_chain
import json

app = Flask(__name__)

# Initialize the QA system
print("[+] Creating vector store from PDFs...")
vectorstore = create_vector_store("data")
print("[+] Vector store created successfully!")

print("[+] Loading QA chain from Pinecone...")
qa = load_qa_chain()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        # Get response from QA system
        result = qa(question)
        
        # Return the result directly since evaluation is already properly formatted
        return jsonify({
            'answer': result.get('answer', 'No answer provided'),
            'evaluation': result.get('evaluation', {})
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
