from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.testing.examples import STANDARD_EXAMPLES


async def amain():
    # response = await acompletion(
    # router = GollumRouter()
    client = GollumClient.create(
        ".gollum/standard-examples-demo",
    )
    router = GollumRouter(cache_responses=True, client=client)
    # model = "openai/gpt-5.6-luna"
    # model = "openai/gpt-5-nano"
    # model = "anthropic/claude-haiku-4-5"
    model = "openai-responses/gpt-5.6-luna"
    for example in STANDARD_EXAMPLES:
        if "unsupported" in example:
            print(f"Skipping {example['id']} (unsupported for this provider)")
            continue
        request = dict(example["request"])
        request["model"] = model
        response = await router.acompletion(
            **request,
        )

        if response.choices[0].message.tool_calls:
            print(response.choices[0].message.tool_calls)
        else:
            print(response.choices[0].message.content)

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(amain())
