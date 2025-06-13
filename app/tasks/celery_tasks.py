# File: C:\Users\agraw\OneDrive\Desktop\CodeBase_Archaelogist\app\tasks\celery_tasks.py

from celery import Celery
import os
import time # For an example task

# --- IMPORTANT ---
# DO NOT 'import streamlit as st' IN THIS FILE.
# DO NOT CALL ANY STREAMLIT FUNCTIONS LIKE st.title(), st.write(), st.session_state etc.
# This file is loaded by the Celery worker, which is a background process,
# NOT a Streamlit web application.
# --- ----------- ---

# Configuration for Celery. Ensure your Redis server is running.
# It's good practice to use environment variables, but defaults are fine for local dev.
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# This is the Celery application instance that your command:
# `celery -A app.tasks.celery_tasks.celery_app_instance worker ...`
# is trying to load. The variable name MUST be 'celery_app_instance'.
celery_app_instance = Celery(
    'my_worker_app', # This is the name of THIS Celery application. Can be anything.
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        # If you have tasks defined in other modules that you want this worker to run,
        # list them here. For example, if tasks are in 'app.tasks.processing':
        # 'app.tasks.processing'
        # For now, we will define tasks directly in this file or assume they are
        # imported from a 'clean' module (not your main Streamlit UI file).
    ]
)

# Optional Celery configuration
celery_app_instance.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    # This can be helpful for robust startup, especially with Docker
    broker_connection_retry_on_startup=True,
)

# --- Define Your Tasks Here or Import Them ---
# These tasks will be registered with 'celery_app_instance'.

@celery_app_instance.task(name="tasks.example_add") # Explicit naming is good practice
def example_add(x, y):
    print(f"Executing example_add task: {x} + {y}")
    time.sleep(5) # Simulate some work
    result = x + y
    print(f"example_add task completed. Result: {result}")
    return result

@celery_app_instance.task(name="tasks.another_example")
def another_task(message):
    print(f"Executing another_task with message: {message}")
    time.sleep(2)
    return f"Processed: {message.upper()}"

# If you want this worker to run the 'process_and_index_repository_task'
# from your api.py, you have a few options:
# 1. Re-define it here, decorated with '@celery_app_instance.task'. (Not DRY)
# 2. Import it from api.py IF AND ONLY IF api.py is also "clean" of Streamlit UI code
#    at the module level and doesn't cause circular imports.
#    e.g., from api import process_and_index_repository_task
#    (This task in api.py is decorated with @api.celery_app.task, so it's tied to
#    THAT Celery app instance. This can get confusing. See "Alternative Approach" below.)

print(f"Celery application 'celery_app_instance' (named '{celery_app_instance.main}') is defined in app.tasks.celery_tasks.")
print("This worker is ready to pick up tasks registered with it.")