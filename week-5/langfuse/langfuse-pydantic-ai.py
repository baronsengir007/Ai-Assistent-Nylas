import base64
import os

import logfire
import nest_asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

nest_asyncio.apply()
load_dotenv()

# Configure LangFuse via OpenTelemetry
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH = base64.b64encode(
    f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
).decode()

# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
#     "https://cloud.langfuse.com/api/public/otel"  # EU data region
# )
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    "https://us.cloud.langfuse.com/api/public/otel"  # US data region
)
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Initialize Logfire (disable sending to Logfire, only send to LangFuse)
logfire.configure(
    service_name="pydantic-ai-demo",
    send_to_logfire=False,
)

# Setup OpenTelemetry tracer for additional metadata
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
tracer = trace.get_tracer("pydantic-ai-demo")

# Create Pydantic AI agent with instrumentation enabled
agent = Agent(
    "openai:gpt-4.1",
    instructions="You are a helpful assistant. Be concise.",
    instrument=True,  # This enables LangFuse tracing
)

# Run the agent with user input
if __name__ == "__main__":
    user_prompt = "What is the capital of France?"

    # Create parent span with additional metadata
    with tracer.start_as_current_span("Pydantic-AI-Query") as span:
        span.set_attribute("langfuse.tags", ["pydantic-ai", "demo"])
        result = agent.run_sync(user_prompt)

        # Add input/output to span
        span.set_attribute("input.value", user_prompt)
        span.set_attribute("output.value", result.output)

        print(f"Response: {result.output}")
        print("Check your LangFuse dashboard to see the traced call!")
