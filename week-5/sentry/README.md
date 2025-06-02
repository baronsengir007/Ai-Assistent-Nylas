# Sentry Error Tracking

## What is Sentry and Why AI Engineers Need It

Sentry is a real-time error tracking and performance monitoring platform that has become essential for any serious Python application, especially in AI engineering. When you're building AI systems that process data, make API calls to language models, or handle complex business logic, things will inevitably break in production. Sentry captures these errors automatically, provides detailed context about what went wrong, and helps you fix issues before they impact your users.

For AI engineers specifically, Sentry is invaluable because AI applications often involve multiple moving parts: API calls to OpenAI or other providers, data processing pipelines, model inference, and complex business logic. When something fails, you need to know immediately what went wrong, with which data, and under what conditions. Sentry gives you this visibility without requiring you to add extensive logging throughout your codebase.

## Getting Started with Sentry

Creating a Sentry account is straightforward and free for developers. Head to [sentry.io](https://sentry.io) and sign up using your GitHub account or email. The Developer plan is completely free and is enough to get you started

Once you have an account, you'll need to create a new project for your application. Sentry will walk you through selecting your platform (choose Python), and it will generate a unique DSN (Data Source Name) that connects your application to your Sentry project. This DSN is essentially your project's API key and should be stored securely in your environment variables.

## FastAPI Integration vs Vanilla Python

Sentry offers different integration approaches depending on your application structure. For FastAPI applications, Sentry provides automatic integration that's incredibly powerful with minimal setup. When you initialize Sentry in a FastAPI app with just `sentry_sdk.init(dsn=your_dsn)`, it automatically captures all unhandled exceptions, HTTP request context, performance data, and user information without any additional code.

The FastAPI integration is particularly useful because it captures the full HTTP context of requests that caused errors. This means when your AI model API fails, you'll see not just the Python stack trace, but also the exact HTTP request that triggered the error, including headers, request body, and user information. This is crucial for debugging issues in production AI applications where the same code might work fine with some inputs but fail with others.

In contrast, vanilla Python integration requires more manual work to capture this context. While Sentry will still automatically capture unhandled exceptions in any Python application, you won't get the rich HTTP context that comes automatically with framework integrations. For command-line scripts, data processing pipelines, or standalone Python applications, the vanilla integration is perfect, but for web APIs serving AI models, the FastAPI integration provides significantly more value.

## Examples in This Directory

The `sentry-setup.py` file in this directory demonstrates both basic error capture and FastAPI integration. The examples show how Sentry automatically captures different types of errors (like `ZeroDivisionError` and `KeyError`) with full context, including HTTP request details when using FastAPI. You can run these examples to see how Sentry works in practice and explore the rich error details in your Sentry dashboard.

The setup is intentionally minimal to show how little code is required to get comprehensive error monitoring. This simplicity is one of Sentry's greatest strengths - you get powerful error tracking and performance monitoring with just a few lines of initialization code, allowing you to focus on building your AI applications rather than implementing monitoring infrastructure.
