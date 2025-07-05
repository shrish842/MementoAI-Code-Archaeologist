import validators

def is_valid_git_url(url: str) -> bool:
    """
    Checks if a given string is a valid Git repository URL.
    """
    return validators.url(url) and url.endswith(".git")

