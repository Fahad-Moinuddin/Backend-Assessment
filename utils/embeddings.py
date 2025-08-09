from openai import OpenAI
client = OpenAI()

def generate_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding