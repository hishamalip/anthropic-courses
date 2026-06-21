import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Replace with your actual key
)

try:
    completion = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct:free",
        messages=[
            {"role": "user", "content": "Hello! Confirm you are running."}
        ]
    )
    
    # Check if the response is just a string error message
    if isinstance(completion, str):
        print(f"OpenRouter Error Message: {completion}")
    else:
        print(completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
