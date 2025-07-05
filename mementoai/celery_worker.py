from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from config.settings import settings
from celery import Celery

# Initialize Celery app
celery_app = Celery(
    'memento_tasks',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True
)

# Import tasks to ensure they are registered with the Celery app
# This import needs to happen after celery_app is defined
from services import celery_tasks

if __name__ == '__main__':
    # This block is for running the worker directly
    # In production, you'd typically use `celery -A celery_worker worker -l INFO`
    print("Starting Celery worker...")
    celery_app.worker_main(['worker', '-l', 'INFO', '--pool=solo'])

