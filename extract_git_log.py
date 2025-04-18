import subprocess
import os
import json # Needed for saving the output file

# --- Configuration ---
# IMPORTANT: This path MUST point to the folder containing the .git directory
# Based on your setup, this should be correct:
REPO_PATH = "C:/Users/agraw/OneDrive/Desktop/CodeBase_Archaelogist/requests"
# Use forward slashes as they work reliably in Python on Windows.

OUTPUT_FILENAME = 'requests_commits.json' # Name of the file to save data to

def extract_commits_from_repo(repo_path):
    """
    Runs git log on the specified repository and parses the output.

    Args:
        repo_path (str): The absolute path to the cloned Git repository.

    Returns:
        list: A list of dictionaries, where each dictionary represents a commit.
              Returns an empty list if an error occurs.
    """
    print(f"Attempting to read Git history from: {repo_path}") # Debug print
    if not os.path.isdir(repo_path):
        print(f"Error: Repository path not found or is not a directory: {repo_path}")
        return []
    # Check specifically for the .git subdirectory which confirms it's likely a git repo root
    if not os.path.isdir(os.path.join(repo_path, '.git')):
         print(f"Error: '.git' directory not found in {repo_path}. Is this the correct repository root?")
         return []


    # The format string we designed earlier
    log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
    command = ["git", "log", f"--pretty=format:{log_format}"]

    try:
        print(f"Running command: {' '.join(command)}") # Debug print
        # Run the git command within the repository's directory
        result = subprocess.run(
            command,
            cwd=repo_path, # Execute command *inside* the repo directory
            capture_output=True,
            text=True,
            check=True, # Raise error if git command fails
            encoding='utf-8',
            errors='replace' # Handle potential weird characters
        )

        raw_log_output = result.stdout
        print(f"Git log command successful. Processing output...") # Debug print
        commits_data = []

        # Split the entire output into individual commit strings
        commit_entries = raw_log_output.strip().split("-----COMMIT_END-----")

        for entry in commit_entries:
            if not entry.strip(): # Skip empty entries if any
                continue

            # Split the entry by our field separator '||', but only 3 times.
            parts = entry.strip().split('||', 3)

            if len(parts) == 4:
                commit_hash, author, timestamp_str, message_part = parts

                # The message part contains subject and body separated by the first newline
                subject_body = message_part.split('\n', 1)
                subject = subject_body[0].strip()
                body = subject_body[1].strip() if len(subject_body) > 1 else ""

                # Combine subject and body for the full message
                full_message = f"{subject}\n{body}".strip()

                # Store the data in a dictionary
                commits_data.append({
                    "hash": commit_hash,
                    "author": author,
                    "timestamp": int(timestamp_str) if timestamp_str.isdigit() else 0,
                    "subject": subject,
                    "body": body,
                    "full_message": full_message # Primary text for the AI model
                })
            # else: # Optional debug for parsing issues
                # print(f"Warning: Could not parse entry fragment: {entry[:100]}...")


        print(f"Successfully parsed {len(commits_data)} commits.")
        return commits_data

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("'git' command not found. Is Git installed correctly?")
        print("Make sure Git is added to your system's PATH environment variable.")
        print("You might need to restart your terminal/computer after installing Git.")
        print("-------------")
        return []
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR ---")
        print(f"Error running git log command. Git returned an error:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Output (stderr): {e.stderr}")
        print("Is the REPO_PATH correct? Is it a valid Git repository?")
        print("-------------")
        return []
    except Exception as e:
        # Catch any other unexpected errors during parsing
        print(f"\n--- UNEXPECTED ERROR ---")
        print(f"An error occurred during commit extraction: {e}")
        print("------------------------")
        return []

# --- This part runs only when you execute the script directly ---
if __name__ == "__main__":
    # Call the function to get commit data
    extracted_commits = extract_commits_from_repo(REPO_PATH)

    # Check if data was extracted successfully
    if extracted_commits:
        print("\n--- Example of First Extracted Commit ---")
        # Pretty print the first commit's dictionary using json module
        print(json.dumps(extracted_commits[0], indent=2))
        print("-----------------------------------------")

        # --- Save the extracted data to a JSON file ---
        output_filepath = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME) # Save in same dir as script
        print(f"\nAttempting to save extracted commits to: {output_filepath}")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                # Use json.dump to write the list of dictionaries to the file
                # indent=2 makes the file human-readable
                # ensure_ascii=False helps with potential non-English characters
                json.dump(extracted_commits, f, indent=2, ensure_ascii=False)
            print(f"\nSuccessfully saved {len(extracted_commits)} commits to {OUTPUT_FILENAME}")
        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"Could not save commits to JSON file '{OUTPUT_FILENAME}': {e}")
            print("---------------")
        # --- End of saving part ---

    else:
        print("\nScript finished, but no commit data was extracted or parsed successfully.")
        print("Please review the error messages above.")