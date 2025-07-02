import os

# Define the full directory and file structure
structure = {
    "mementoai": {
        ".gitignore": "",
        "README.md": "",
        "requirements.txt": "",
        "backend": {
            "main.py": "",
            "config.py": "",
            "celery_app.py": "",
            "services": {
                "git_services.py": "",
                "analysis_services.py": "",
                "embedding_services.py": "",
                "pinecone_services.py": "",
            },
            "models": {
                "schemas.py": "",
            },
            "tasks": {
                "indexing_tasks.py": "",
            },
            "utils": {
                "helpers.py": "",
            },
        },
        "frontend": {
            "app.py": "",
        }
    }
}

def create_structure(base_path, structure_dict):
    """
    Create folders and files from the nested dictionary structure.
    Files are created empty.
    """
    for name, content in structure_dict.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # It's a file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)


if __name__ == "__main__":
    create_structure(".", structure)
    print("Directory and file structure created successfully.")

