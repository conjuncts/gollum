"""
High-level demo: using the batch API through the exact same interface as
litellm's Router -- `router.completion(model=..., messages=[...])` -- with
batching happening transparently in the background.

DO NOT RUN as part of CI/automated checks: this hits the real OpenAI Batch
API (costs money, takes real wall-clock time for the batch window) and is
meant to be driven live, by hand: `python scripts/batch_router_demo.py`.

How this works, end to end
---------------------------
`GollumRouter.acompletion()` is a thin wrapper: it builds a request and
does `await gollum_client.worklist.enroll(request)` -- see
gollum.client.litellm.acompletion. It has NO idea whether `worklist` is an
EagerWorklist, a ConcurrentWorklist, or (as here) a BatchWorklist. So
plugging BatchWorklist in is enough to get batching for free, with an
*identical* call shape to the quickstart scripts -- the caller can't tell
the difference from the outside.

Threading model -- fully async, ONE loop, no extra thread
-----------------------------------------------------------
`GollumClient` has a `run_coroutine_sync()` helper that spins up its own
background thread with its own event loop, purely so *synchronous* callers
(`router.completion()`) can block on async work. That machinery is
unrelated to whether BatchWorklist itself needs a background thread to
keep polling -- it doesn't, as long as everything touching it runs on one
consistent event loop.

Importantly: `GollumRouter.acompletion()` does NOT route through
`run_coroutine_sync` -- it's a bare coroutine that runs on whatever loop
awaits it. So if `worklist.start()` were booted via
`client.run_coroutine_sync(...)` (pinning BatchHandler's poll loop to
GollumClient's private background-thread loop) while `amain()` awaits
`router.acompletion(...)` on `asyncio.run(amain())`'s own main-thread loop,
you'd have BatchHandler's per-job `asyncio.Lock`s being acquired/released
from two different loops on two different OS threads -- unlike
`WorklistEntry.finish()/.fail()`, which are deliberately
`call_soon_threadsafe`-safe for exactly this kind of cross-thread case,
plain `asyncio.Lock` is not, and under contention this can hang rather
than raise (confirmed empirically -- send_batch and check_batch ended up
running on different threads when tried).

The fix, used below: since `amain()` is already async and already owns one
live loop (via `asyncio.run`), there's no reason to involve
`GollumClient`'s background thread at all here. Just `await
worklist.start()` directly on `amain()`'s own loop before enrolling
anything, and call `router.acompletion(...)` the normal async way. Every
BatchWorklist/BatchHandler operation -- enroll, reconnect, submit_entries,
the poll loop, batch_arrival -- then shares that single loop, which is
exactly the safety requirement. (`GollumClient.run_coroutine_sync()` stays
relevant for a *sync* caller like `router.completion()`, but you'd want to
call `worklist.start()` through it too in that case, so everything -- poll
loop included -- ends up on that one dedicated thread instead. Don't mix
the two styles against the same worklist.)
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from gollum.batch.batch_handler import BatchHandler
from gollum.batch.batch_trigger import CompositeTrigger, IntervalTrigger, SizeTrigger
from gollum.batch.storage.duckdb_batch_storage import DuckDBBatchStorage
from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.folder.file_manager import FileManager
from gollum.permacache.cache_method import CacheMethod
from gollum.permacache.duckdb_permacache import DuckDBPermacache
from gollum.provider.openai_batch import BatchOpenAIWorker
from gollum.worklist.batch_worklist import BatchWorklist


async def make_batch_router() -> GollumRouter:
    """
    Wires BatchWorklist up behind a GollumClient/GollumRouter, so callers
    get the plain `router.acompletion(...)` interface while batching
    happens behind the scenes. This is the whole integration -- nothing
    about GollumClient/GollumRouter needed to change to support it.

    Must be awaited from the same event loop that will later drive
    `router.acompletion(...)` calls -- see module docstring.
    """
    fm = FileManager(".gollum/batch-router-demo")

    handler = BatchHandler(
        batch_storage=DuckDBBatchStorage(fm),
        batch_worker=BatchOpenAIWorker(AsyncOpenAI()),
        permacache=DuckDBPermacache(fm),
        cache_method=CacheMethod(),
        polling_frequency=30.0,  # how often to check OpenAI for batch completion
        confirm_before_submit=True,  # ask before spending money on a real batch
    )

    # Flush every 20 queued requests, or every 5 seconds -- whichever comes
    # first -- so a handful of concurrent callers get batched together
    # instead of each triggering its own tiny batch.
    trigger = CompositeTrigger([SizeTrigger(20), IntervalTrigger(5.0)])

    worklist = BatchWorklist(handler, trigger=trigger)
    client = GollumClient(worklist)

    # Boot the poll loop on THIS loop -- the same one that will later
    # await router.acompletion(...) -- rather than GollumClient's private
    # background thread, which router.acompletion() doesn't use anyway.
    await worklist.start()

    return GollumRouter(client=client)


async def ask(router: GollumRouter, question: str) -> str:
    """
    Looks and behaves exactly like a normal litellm.Router async call.
    Resolves once this question's batch comes back -- could be minutes,
    since it's riding OpenAI's real Batch API completion window.
    """
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


async def amain():
    router = await make_batch_router()

    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Peru?",
    ]

    # Enroll all three concurrently, on this same loop. Under the hood
    # they all land in the same BatchWorklist and (per the trigger above)
    # go out as ONE batch -- transparently, from the caller's point of view.
    print("asking all three questions concurrently...")
    answers = await asyncio.gather(*(ask(router, q) for q in questions))

    for question, answer in zip(questions, answers):
        print(f"  Q: {question}\n  A: {answer}")

    # Graceful shutdown: flush anything still queued, let the poll loop
    # finish its current tick, then stop it -- all on this same loop.
    await router.client.worklist.shutdown()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(amain())
