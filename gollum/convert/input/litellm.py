from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Type, Union

import httpx

from gollum.types import GollumRequest
from gollum.types.chat_completions import AnthropicThinkingParam, ChatCompletionRequest, OpenAIWebSearchOptions

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


def litellm_completion_to_request(
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
    modalities: Optional[list["ChatCompletionModality"]] = None,
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
) -> GollumRequest:
    """
    Exact method signature, nearly identical to `completion` in litellm 1.95.0
    Compatible with `client.chat.completions.create` in openai 2.37.0
    """
    chat_completion: ChatCompletionRequest = {
        "messages": messages,
        "model": model,
    }
    optional = {
        "audio": audio,
        "frequency_penalty": frequency_penalty,
        "function_call": function_call,
        "functions": functions,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "metadata": metadata,
        "modalities": modalities,
        "moderation": moderation,
        "n": n,
        "parallel_tool_calls": parallel_tool_calls,
        "prediction": prediction,
        "presence_penalty": presence_penalty,
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_options": prompt_cache_options,
        "prompt_cache_retention": prompt_cache_retention,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "safety_identifier": safety_identifier,
        "seed": seed,
        "service_tier": service_tier,
        "stop": stop,
        "store": None,
        "stream": stream,
        "stream_options": stream_options,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_logprobs": top_logprobs,
        "top_p": top_p,
        "user": user,
        "verbosity": verbosity,
        "web_search_options": web_search_options
    }
    optional = {k: v for k, v in optional.items() if v is not None}
    chat_completion.update(optional)

    # attempt to 
    # TODO: model_list or model aliases
    if "/" in model:
        provider_type = model.split("/")[0]
    else:
        provider_type = None
    gollum_request = GollumRequest(
        request=chat_completion,
        extras={
            "timeout": timeout,
            "include_server_side_tool_invocations": include_server_side_tool_invocations,
            "deployment_id": deployment_id,
            "extra_headers": extra_headers,
            "base_url": base_url,
            "api_version": api_version,
            "api_key": api_key,
            "model_list": model_list,
            "thinking": thinking,
            "shared_session": shared_session,
            "enable_json_schema_validation": enable_json_schema_validation
        },
        metadata={},
        provider_type=provider_type,
    )
    return gollum_request