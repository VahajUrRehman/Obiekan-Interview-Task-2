import os
import json
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.schema.output import ChatResult, ChatGeneration
from typing import List, Any, Optional, Dict
from pydantic import Field, ConfigDict

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

HF_TOKEN = os.getenv("HF_TOKEN")
LLM_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
JUDGE_MODEL = "deepseek-ai/DeepSeek-Prover-V2-671B"  # Using a different model for judge

def embed_text(text: str) -> list:
    # Use Pinecone's inference API for embeddings
    result = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={"input_type": "passage"}
    )
    return result[0].values

# Chat LLM wrapper for LangChain
class HFChatLLM(BaseChatModel):
    client: Any = Field(default=None)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(provider="sambanova", api_key=HF_TOKEN)
    
    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                chat_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                chat_messages.append({"role": "user", "content": message.content})
            else:
                chat_messages.append({"role": "assistant", "content": message.content})
        
        if not chat_messages:
            chat_messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer ONLY from context."},
                {"role": "user", "content": messages[0].content if messages else ""}
            ]
        
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=chat_messages
        )
        
        text = response.choices[0].message["content"]
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": LLM_MODEL}

class JudgeAgent:
    def __init__(self):
        self.client = InferenceClient(provider="novita", api_key=HF_TOKEN)
    
    def evaluate_response(self, question: str, answer: str, context: str) -> dict:
        """
        Evaluate the main agent's response for accuracy and relevance.
        """
        judge_prompt = f"""You are an expert judge evaluating AI responses. Analyze the following:

Question: {question}
Context: {context}
AI's Answer: {answer}

Evaluate the response on:
1. Accuracy (based on provided context)
2. Relevance to the question
3. Completeness of the answer

Provide your evaluation in JSON format with these fields:
- score (0-10)
- reasoning
- suggestions_for_improvement

Return ONLY valid JSON without any markdown formatting or prefixes.
"""
        try:
            response = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert judge who evaluates AI responses critically and fairly. Return ONLY valid JSON without any markdown formatting or prefixes."},
                    {"role": "user", "content": judge_prompt}
                ]
            )
            content = response.choices[0].message["content"]
            
            # Clean up the response - remove markdown formatting if present
            content = content.replace('### Evaluation', '').strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            try:
                evaluation = json.loads(content)
                return evaluation  # Return the dictionary directly
            except json.JSONDecodeError:
                return {
                    "score": 0,
                    "reasoning": "Error: Invalid evaluation format",
                    "suggestions_for_improvement": None
                }
                
        except Exception as e:
            print(f"Judge evaluation error: {str(e)}")
            return {
                "score": 0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "suggestions_for_improvement": "Unable to provide suggestions due to error"
            }
