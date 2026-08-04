import asyncio

from gollum.client.singleton import acompletion

async def main():
    response = await acompletion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())