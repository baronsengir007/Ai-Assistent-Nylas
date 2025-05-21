import os

from celery import Celery


def get_redis_url():
    """
    Get the Redis URL for Celery configuration.

    Returns:
        str: The Redis URL.
    """
    # Use environment variable or default to Docker service name
    default_redis_url = "redis://redis:6379/0"
    redis_url = os.environ.get("CELERY_BROKER_URL", default_redis_url)
    return redis_url


def get_celery_config():
    """
    Get the Celery configuration.

    Returns:
        dict: The Celery configuration.
    """
    redis_url = get_redis_url()
    # Allow result backend to be configured separately if needed
    result_backend_url = os.environ.get("CELERY_RESULT_BACKEND", redis_url)
    return {
        "broker_url": redis_url,
        "result_backend": result_backend_url,
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "broker_connection_retry_on_startup": True,
        "broker_transport_options": {
            "visibility_timeout": 3600,
        },
        "broker_connection_max_retries": 0,  # Retry forever
    }


celery_app = Celery("demo")
celery_app.config_from_object(get_celery_config())

# Automatically discover and register tasks
celery_app.autodiscover_tasks(["tasks"], force=True)
