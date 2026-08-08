import asyncio

from dotenv import load_dotenv

from gollum.client.litellm import GollumRouter


async def amain_10():
    # response = await acompletion(
    # router = GollumRouter()
    router = GollumRouter(cache_responses=True)
    futures = []
    for country in [
        "France",
        "Germany",
        "Italy",
        "Spain",
        "Portugal",
        "Russia",
        "Turkey",
        "Greece",
        "Norway",
        "Sweden",
    ]:
        futures.append(router.acompletion(
            model="openai/gpt-5.6-luna",
            # model="anthropic/claude-haiku-4-5",
            messages=[
                {"role": "user", "content": f"What is the capital of {country}?"}
            ],
            gollum_salt=1,
        ))

    # results = await asyncio.gather(*futures)
    # for response in results:

    # await as they come in
    for future in asyncio.as_completed(futures):
        response = await future
        print(response.choices[0].message.content)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(amain_10())
