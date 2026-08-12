import pytest

from gollum.provider.guess_provider import guess_provider_name


@pytest.mark.parametrize("name", [
    "openai/gpt-5.6-luna",
    "o1",
    "gpt-4o",
    "gpt-5.6-sol"
])
def test_guess_openai(name):
    provider, model = guess_provider_name(name)
    assert provider == "openai"
    assert model == name.removeprefix("openai/")


@pytest.mark.parametrize("name", [
    "gemini-2.5-pro",
    "google/gemini-2.5-pro",
])
def test_guess_google(name):
    provider, model = guess_provider_name(name)
    assert provider == "google"
    assert model == name.removeprefix("google/")


@pytest.mark.parametrize("name", [
    "claude-sonnet-5",
])
def test_guess_anthropic(name):
    provider, model = guess_provider_name(name)
    assert provider == "anthropic"
    assert model == name.removeprefix("anthropic/")


@pytest.mark.parametrize("name", [
    "mistral-7b-instruct-v0.1",
])
def test_guess_mistral(name):
    provider, model = guess_provider_name(name)
    assert provider == "mistral"
    assert model == name.removeprefix("mistral/")