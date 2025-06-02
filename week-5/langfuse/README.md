# Langfuse LLM Observability Platform

## What is Langfuse and Why AI Engineers Need It

Langfuse is an open-source LLM engineering platform that provides observability, evaluation, and prompt management for AI applications. Unlike traditional monitoring tools, Langfuse is designed for the challenges of building with large language models, where non-deterministic outputs, multi-step workflows, and the need to track costs and quality make debugging challenging.

The platform is particularly valuable because it's model and framework agnostic, meaning you can use it with OpenAI, Anthropic, local models, PydanticAI, LlamaIndex, or any other AI toolchain. This flexibility makes it an essential tool for production AI applications where you need consistent monitoring across different components and providers.

## Getting Started with Langfuse

Setting up Langfuse is straightforward and free to get started. Head to [langfuse.com](https://langfuse.com) and create an account using GitHub or email. The platform offers a free tier to get started.

Once you have an account, create a new project and you'll receive three credentials: a public key, secret key, and host URL. These should be stored in your environment variables as `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`. The setup is minimally invasive. In most cases, you can start getting value from Langfuse by adding a single decorator to your functions or importing the OpenAI integration.

Langfuse also offers [self-hosting options](https://langfuse.com/docs/deployment/self-host) if you prefer to keep your data within your own infrastructure, which is valuable for enterprise AI applications with strict data governance requirements.

## Understanding the Langfuse Data Model

[Langfuse's data](https://langfuse.com/docs/tracing-data-model) model is built around hierarchical observability, inspired by OpenTelemetry but optimized for LLM applications. The core building blocks are **Traces**, **Observations**, **Sessions**, and **Scores**.

- **Traces** represent a single request or operation in your application, typically corresponding to one API call or user interaction. Each trace captures the input and output, along with metadata like user information, session details, and custom tags. Within each trace, you can have multiple **Observations** that log individual steps of the execution.

- **Observations** come in three types: **Events** for discrete actions, **Spans** for durations of work, and **Generations** for LLM API calls. Generations are powerful because they automatically calculate token usage and costs while capturing the full prompt and completion context. Observations can be nested to represent hierarchical workflows, making it easy to trace through multi-step AI processes.

- **Sessions** allow you to group related traces together, which is essential for conversational AI applications where multiple interactions form a user experience. This is useful for chatbots, multi-turn assistants, or any application where context carries across multiple requests.

- **Scores** provide the evaluation layer, allowing you to attach quality metrics to any trace, observation, or session. These can be numeric values, boolean flags, or categorical assessments, and they can come from automated evaluations, user feedback, or manual annotation.

## The Benefits of Langfuse

Languse can help with:

- Tracing and Debugging
- Cost Optimization and Usage Tracking
- Advanced Prompt Management
- Scoring and Evaluations

## Prompt Management in Practice

Langfuse's prompt management acts as a Content Management System (CMS) for your AI prompts, enabling you to decouple prompt engineering from application deployment. Instead of hardcoding prompts in your codebase, you can store, version, and manage them centrally through Langfuse.

### Key Features

- **Model Configuration Storage**: Prompts can include a `config` object that stores model parameters like temperature, max tokens, and even the model name itself. This means you can change not just the prompt text but also the model behavior without touching your code.

- **Labels for Environment Management**: Use labels like `production`, `staging`, or `experimental` to control which prompt version your application uses. When you call `langfuse.get_prompt("prompt-name")`, it automatically fetches the version labeled as `production`. This enables instant prompt deployments and easy rollbacks.

- **Automatic Versioning**: Every time you create or update a prompt, Langfuse automatically increments the version number. You can track the history of changes, compare performance across versions, and rollback to previous versions if needed.

- **Template Variables**: Prompts support template variables using double bracket syntax `{{variable_name}}`, which you can compile at runtime with actual values using the `prompt.compile()` method.

### Workflow Example

```python
# 1. Register a prompt (usually done once)
langfuse.create_prompt(
    name="customer-email",
    prompt="Write a {{tone}} email to {{customer_name}} about {{topic}}",
    config={"model": "gpt-4o", "temperature": 0.7},
    labels=["production"]
)

# 2. Use in your application
prompt = langfuse.get_prompt("customer-email")
compiled_prompt = prompt.compile(
    tone="friendly", 
    customer_name="John", 
    topic="service update"
)
```

This approach enables prompt engineers and product teams to iterate on prompts through the Langfuse UI while developers focus on application logic. The `prompt-management.py` example demonstrates this workflow with a minimal implementation.

## Examples and Getting Started

The `langfuse-setup.py` file in this directory demonstrates both basic Langfuse integration and a business use case. The examples show how to use the `@observe()` decorator for automatic tracing, structured output with Pydantic models, and custom metadata tracking. These examples illustrate how minimal the integration can be while providing observability.

The customer support automation example demonstrates real-world usage patterns including multi-step workflows, structured data extraction, and business-relevant metadata tracking. This provides a template for integrating Langfuse into production AI applications.

## Additional Resources and Next Steps

Langfuse offers extensive documentation and integration guides at [langfuse.com/docs](https://langfuse.com/docs), including specific integration examples for popular frameworks. The platform also provides an [interactive demo](https://langfuse.com/docs/demo) where you can explore the interface and features without setting up your own account.
