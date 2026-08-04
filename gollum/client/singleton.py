"""
A global client

"""

from gollum.client.base import GollumClient
from gollum.client.litellm import LiteLLMWorklistEntry
from gollum.convert.input.litellm import litellm_completion_to_request

from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Type, Union

import httpx

from gollum.types.chat_completions import AnthropicThinkingParam, OpenAIWebSearchOptions
from gollum.worklist.workers.mock_worker import MockWorker
from gollum.worklist.worklist import EagerWorklist

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionAudioParam,
        ChatCompletionMessageParam,
        ChatCompletionModality,
        ChatCompletionPredictionContentParam,
    )
    from openai.types.chat.completion_create_params import (
        Moderation,
        PromptCacheOptions,
    )
    from aiohttp import ClientSession
    from pydantic import BaseModel


_singleton = None
def get_singleton_client() -> GollumClient:
    global _singleton
    if _singleton is None:
        worklist = EagerWorklist()
        worklist.enroll_worker(MockWorker(parroted_value="Hello, World!"))
        _singleton = GollumClient(worklist)
    return _singleton

def acompletion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: Iterable["ChatCompletionMessageParam"] = None,
    timeout: Optional[Union[float, str, httpx.Timeout]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    modalities: Optional[List["ChatCompletionModality"]] = None,
    prediction: Optional["ChatCompletionPredictionContentParam"] = None,
    audio: Optional["ChatCompletionAudioParam"] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    # openai v1.0+ new params
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]] = None,
    verbosity: Optional[Literal["low", "medium", "high"]] = None,
    response_format: Optional[Union[dict, Type["BaseModel"]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    web_search_options: Optional[OpenAIWebSearchOptions] = None,
    include_server_side_tool_invocations: Optional[bool] = None,
    deployment_id=None,
    extra_headers: Optional[dict] = None,
    safety_identifier: Optional[str] = None,
    service_tier: Optional[str] = None,
    # soon to be deprecated params by OpenAI
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    thinking: Optional[AnthropicThinkingParam] = None,
    # Session management
    shared_session: Optional["ClientSession"] = None,
    # Per-request JSON schema validation (overrides litellm.enable_json_schema_validation)
    enable_json_schema_validation: Optional[bool] = None,
    *,
    # added by gollum: gpt 5.6+ new params
    metadata: Optional[Dict[str, str]] = None,
    moderation: Optional["Moderation"] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_options: Optional["PromptCacheOptions"] = None,
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None,
    # store: Optional[bool] = None, # soon to be deprecated: related to openai's fine tuning api
    **kwargs,
):
    """
    1-to-1 to litellm
    """
    request = litellm_completion_to_request(
        model=model,
        messages=messages,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stream=stream,
        stream_options=stream_options,
        stop=stop,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        modalities=modalities,
        prediction=prediction,
        audio=audio,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        user=user,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        response_format=response_format,
        seed=seed,
        tools=tools,
        tool_choice=tool_choice,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        parallel_tool_calls=parallel_tool_calls,
        web_search_options=web_search_options,
        include_server_side_tool_invocations=include_server_side_tool_invocations,
        deployment_id=deployment_id,
        extra_headers=extra_headers,
        safety_identifier=safety_identifier,
        service_tier=service_tier,
        functions=functions,
        function_call=function_call,
        base_url=base_url,
        api_version=api_version,
        api_key=api_key,
        model_list=model_list,
        thinking=thinking,
        shared_session=shared_session,
        enable_json_schema_validation=enable_json_schema_validation,
        metadata=metadata,
        moderation=moderation,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_options=prompt_cache_options,
        prompt_cache_retention=prompt_cache_retention,
        **kwargs,
    )
    worklist_entry = get_singleton_client().worklist.enroll(request)
    return LiteLLMWorklistEntry(worklist_entry)
