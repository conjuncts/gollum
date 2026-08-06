import asyncio

from dotenv import load_dotenv

from gollum.client.singleton import acompletion, completion

async def amain():
    response = await acompletion(
        # model="openai/gpt-5.6-luna",
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    print(response.choices[0].message.content)


def main():
    response = completion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    print(response.choices[0].message.content)



if __name__ == "__main__":
    load_dotenv()
    asyncio.run(amain())
    # main()