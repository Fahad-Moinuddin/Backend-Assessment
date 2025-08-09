import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL= "gpt-3.5-turbo"
MAX_TOKENS = 1000

def ask_llm(prompt: str) -> str:
    """
    Ask the LLM a question and return the response.

    Args:
        prompt: The question to ask the LLM.

    Returns:
        str: The response from the LLM.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"