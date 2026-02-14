# LLM Backends & REPL Environments

DeepRecall supports multiple LLM providers and sandboxed execution environments via the underlying [RLM](https://github.com/alexzhang13/rlm) engine.

---

## Supported LLM Backends

Set via `backend` in `DeepRecallConfig` or the `DeepRecall()` constructor.

| Backend | `backend` value | Install | `backend_kwargs` keys |
|---|---|---|---|
| **OpenAI** | `"openai"` | Included with `rlms` | `model_name`, `api_key`, `base_url` |
| **Anthropic** | `"anthropic"` | Included with `rlms` | `model_name`, `api_key` |
| **Google Gemini** | `"gemini"` | Included with `rlms` | `model_name`, `api_key` |
| **Azure OpenAI** | `"azure_openai"` | Included with `rlms` | `model_name`, `api_key`, `base_url`, `api_version` |
| **Portkey** | `"portkey"` | Included with `rlms` | `model_name`, `api_key`, `virtual_key` |
| **OpenRouter** | `"openrouter"` | Included with `rlms` | `model_name`, `api_key` |
| **vLLM** | `"vllm"` | Self-hosted | `model_name`, `base_url` |
| **LiteLLM** | `"litellm"` | `pip install litellm` | `model_name`, `api_key` |
| **Vercel** | `"vercel"` | Via Portkey | `model_name`, `api_key` |

### Examples

**OpenAI (default)**:

```python
engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": "sk-..."},
)
```

**Anthropic Claude**:

```python
engine = DeepRecall(
    vectorstore=store,
    backend="anthropic",
    backend_kwargs={"model_name": "claude-sonnet-4-20250514", "api_key": "sk-ant-..."},
)
```

**Google Gemini**:

```python
engine = DeepRecall(
    vectorstore=store,
    backend="gemini",
    backend_kwargs={"model_name": "gemini-2.0-flash", "api_key": "AIza..."},
)
```

**Azure OpenAI**:

```python
engine = DeepRecall(
    vectorstore=store,
    backend="azure_openai",
    backend_kwargs={
        "model_name": "my-gpt-4o-deployment",
        "api_key": "...",
        "base_url": "https://my-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview",
    },
)
```

**Self-hosted vLLM**:

```python
engine = DeepRecall(
    vectorstore=store,
    backend="vllm",
    backend_kwargs={"model_name": "meta-llama/Llama-3-70b", "base_url": "http://localhost:8000"},
)
```

### Sub-LLM Backends

The LLM can call `llm_query()` inside the REPL to make sub-calls to a different (often cheaper) model. Configure with `other_backends`:

```python
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o"},           # main model
    other_backends=["openai"],
    other_backend_kwargs=[{"model_name": "gpt-4o-mini"}],  # sub-LLM
)
```

---

## REPL Environments

The LLM executes Python code in a sandboxed REPL. Set via `environment` in `DeepRecallConfig`.

| Environment | `environment` value | Description |
|---|---|---|
| **Local** | `"local"` | Subprocess on the same machine. Default, zero-config |
| **Docker** | `"docker"` | Isolated Docker container. Requires Docker |
| **Modal** | `"modal"` | [Modal](https://modal.com) cloud sandbox |
| **E2B** | `"e2b"` | [E2B](https://e2b.dev) cloud sandbox |
| **Daytona** | `"daytona"` | [Daytona](https://www.daytona.io) cloud sandbox |
| **Prime** | `"prime"` | Prime cloud environment |

### Environment kwargs

Pass environment-specific settings via `environment_kwargs`:

```python
# Docker with custom image
config = DeepRecallConfig(
    environment="docker",
    environment_kwargs={"image": "python:3.12-slim"},
)

# E2B with API key
config = DeepRecallConfig(
    environment="e2b",
    environment_kwargs={"api_key": "e2b-..."},
)
```

### When to Use Each

| Scenario | Recommended Environment |
|---|---|
| Development / local testing | `"local"` (default) |
| Production with untrusted queries | `"docker"` or `"e2b"` |
| Serverless deployment | `"modal"` or `"e2b"` |
| Maximum isolation | `"docker"` |
