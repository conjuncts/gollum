from litellm import completion
# from litellm.router import Router

response = completion(
    model="openai/gpt-5.6-luna",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)

print(response.choices[0].message.content)
# router = Router()
