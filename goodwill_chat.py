# https://www.youtube.com/watch?v=H5SczGgJyRQ
# Thursday, November 20, 2025

import os
from openai import OpenAI

#print("Now pulling from .env")
from dotenv import load_dotenv


load_dotenv()  # This will load the variables from .env

value = os.getenv("OPENAI_API_KEY")
    #database_url = os.getenv("DATABASE_URL")

OpenAI.api_key = OpenAI(api_key = value)

#print("                                          ")
#print("OpenAI.api_key is API value but will not show - only a representation")
#print(OpenAI.api_key)
#print("                                       ")

client = OpenAI(api_key = value)
#print("                                          ")
#print("Trying to print api_key as client")
#print(client)

prompt = ""

while True:
    prompt = input("Volunteer Leslie:")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role" : "user",
                "content" : prompt
            }
        ],
        model = "gpt-3.5-turbo"
    )

    #print the response
    response_message = chat_completion.choices[0].message.content
    print("Goodwill Customer:", response_message)

