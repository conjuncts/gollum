from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Type, Union

import httpx
from aiohttp import ClientSession

from gollum.client.base import GollumClient
from gollum.client.litellm_helper import LiteLLMWorklistEntry
from gollum.convert.input.litellm import litellm_completion_to_request
from gollum.types.chat_completions import AnthropicThinkingParam, ChatCompletionResponseModel, OpenAIWebSearchOptions


if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionModality,
        ChatCompletionAudioParam,
        ChatCompletionPredictionContentParam,
    )
    from openai.types.chat.completion_create_params import Moderation, PromptCacheOptions
    from litellm import AlertingConfig, AllowedFailsPolicy, AssistantsTypedDict, DeploymentTypedDict, GuardrailTypedDict, OptionalPreCallChecks, RetryPolicy, RouterGeneralSettings, RouterModelGroupAliasItem, RoutingGroup, RoutingPlugin, SearchToolTypedDict
    from litellm.types.utils import GenericBudgetConfigType
    from pydantic import BaseModel


_singleton = None
def _get_singleton_client() -> GollumClient:
    global _singleton
    if _singleton is None:
        _singleton = GollumClient.create()
    return _singleton


async def acompletion(
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
    gollum_salt: Optional[str] = None,
    gollum_client: Optional[GollumClient],
    **kwargs,
) -> ChatCompletionResponseModel:
    """
    1-to-1 to litellm's acompletion
    """
    if gollum_client is None:
        gollum_client = _get_singleton_client()
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
        gollum_salt=gollum_salt,
        **kwargs,
    )
    worklist_entry = await gollum_client.worklist.enroll(request)
    return await LiteLLMWorklistEntry(worklist_entry)


def completion(
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
    gollum_salt: Optional[str] = None,
    # store: Optional[bool] = None, # soon to be deprecated: related to openai's fine tuning api
    gollum_client: Optional[GollumClient],
    **kwargs,
) -> ChatCompletionResponseModel:
    """
    WARNING: unlike litellm, this function invokes acompletion().
    """
    coro = acompletion(
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
        gollum_salt=gollum_salt,
        gollum_client=gollum_client,
        **kwargs,
    )

    return gollum_client.run_coroutine_sync(coro)


class GollumRouter:
    """
    1-to-1 with LiteLLM's `Router` class
    """

    def __init__(
        self,
        model_list: Optional[Union[List["DeploymentTypedDict"], List[Dict[str, Any]]]] = None,
        ## ASSISTANTS API ##
        assistants_config: Optional["AssistantsTypedDict"] = None,
        ## SEARCH API ##
        search_tools: Optional[List["SearchToolTypedDict"]] = None,
        ## GUARDRAIL API ##
        guardrail_list: Optional[List["GuardrailTypedDict"]] = None,
        ## CACHING ##
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        redis_db: Optional[int] = None,
        cache_responses: Optional[Union[str, bool]] = False,
        cache_kwargs: dict = {},  # additional kwargs to pass to RedisCache (see caching.py)
        caching_groups: Optional[List[tuple]] = None,  # if you want to cache across model groups
        client_ttl: int = 3600,  # ttl for cached clients - will re-initialize after this time in seconds
        ## SCHEDULER ##
        polling_interval: Optional[float] = None,
        default_priority: Optional[int] = None,
        ## RELIABILITY ##
        num_retries: Optional[int] = None,
        max_fallbacks: Optional[int] = None,  # max fallbacks to try before exiting the call. Defaults to 5.
        timeout: Optional[float] = None,
        stream_timeout: Optional[float] = None,
        default_litellm_params: Optional[dict] = None,  # default params for Router.chat.completion.create
        default_max_parallel_requests: Optional[int] = None,
        set_verbose: bool = False,
        debug_level: Literal["DEBUG", "INFO"] = "INFO",
        default_fallbacks: Optional[List[str]] = None,  # generic fallbacks, works across all deployments
        fallbacks: List = [],
        context_window_fallbacks: List = [],
        content_policy_fallbacks: List = [],
        model_group_alias: Optional[Dict[str, Union[str, "RouterModelGroupAliasItem"]]] = {},
        enable_pre_call_checks: bool = False,
        enable_tag_filtering: bool = False,
        tag_filtering_match_any: bool = True,
        plugins: list["RoutingPlugin"] | None = None,
        retry_after: int = 0,  # min time to wait before retrying a failed request
        retry_policy: Optional[Union["RetryPolicy", dict]] = None,  # set custom retries for different exceptions
        model_group_retry_policy: Dict[str, "RetryPolicy"] = {},  # set custom retry policies based on model group
        allowed_fails: Optional[int] = None,  # Number of times a deployment can failbefore being added to cooldown
        allowed_fails_policy: Optional["AllowedFailsPolicy"] = None,  # set custom allowed fails policy
        cooldown_time: Optional[float] = None,  # (seconds) time to cooldown a deployment after failure
        disable_cooldowns: Optional[bool] = None,
        routing_strategy: Literal[
            "simple-shuffle",
            "least-busy",
            "usage-based-routing",
            "latency-based-routing",
            "cost-based-routing",
            "usage-based-routing-v2",
            "lar1",
        ] = "simple-shuffle",
        optional_pre_call_checks: Optional["OptionalPreCallChecks"] = None,
        routing_strategy_args: dict = {},  # just for latency-based
        routing_groups: Optional[List[Union["RoutingGroup", dict]]] = None,
        provider_budget_config: Optional["GenericBudgetConfigType"] = None,
        alerting_config: Optional["AlertingConfig"] = None,
        router_general_settings: Optional["RouterGeneralSettings"] = None, # RouterGeneralSettings(),
        deployment_affinity_ttl_seconds: int = 3600,
        model_group_affinity_config: Optional[Dict[str, List[str]]] = None,
        ignore_invalid_deployments: bool = False,
        enable_health_check_routing: bool = False,
        health_check_staleness_threshold: Optional[int] = None,
        health_check_ignore_transient_errors: bool = False,
        enable_weighted_failover: bool = False,
        *,
        client: Optional[GollumClient] = None,
    ) -> None:
        if client is None:
            if isinstance(cache_responses, (Path, str)):
                storage_destination = cache_responses
            elif cache_responses:
                storage_destination = ".gollum"
            else:
                storage_destination = None
            client = GollumClient.create(cache_location=storage_destination)
        self.client = client

    @wraps(acompletion)
    def acompletion(self, *args, **kwargs):
        return acompletion(*args, gollum_client=self.client, **kwargs)

    @wraps(completion)
    def completion(self, *args, **kwargs):
        return completion(*args, gollum_client=self.client, **kwargs)

    completion.__name__ = "completion"
    completion.__doc__ = (
        "Sync chat completion (blocks the calling thread). Same signature as "
        "litellm.Router.completion; routes through this router's client."
    )


