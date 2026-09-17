from ygobench.config import default_model_config, missing_api_key


def test_dashscope_model_config_uses_qwen_default(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = default_model_config("dashscope")
    assert config.provider == "dashscope"
    assert config.backend == "openai"
    assert config.model == "qwen3.7-flash"
    assert missing_api_key("dashscope") == "DASHSCOPE_API_KEY"


def test_dashscope_model_config_accepts_explicit_qwen_model(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    config = default_model_config("dashscope", "qwen3.8-max")
    assert config.model == "qwen3.8-max"
    assert missing_api_key("dashscope") is None


def test_bailian_qwen_uses_explicit_thinking_provider(monkeypatch) -> None:
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    config = default_model_config("bailian", "qwen3.7-flash")
    assert config.provider == "bailian"
    assert config.backend == "openai"
    assert config.model == "qwen3.7-flash"
    assert config.api_key == "test-key"
