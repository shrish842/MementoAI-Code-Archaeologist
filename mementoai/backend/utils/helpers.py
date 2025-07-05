import validators
import uuid

def validate_repo_url(repo_url: str) -> bool:
    return validators.url(repo_url) and repo_url.endswith(".git")

def generate_repo_namespace(repo_url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))