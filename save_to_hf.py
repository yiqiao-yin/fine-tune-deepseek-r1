#!/usr/bin/env python
import os
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class HFPusher:
    """
    A class to handle logging into Hugging Face Hub, creating a repository,
    and pushing a locally saved model and tokenizer to the Hub.
    """
    
    def __init__(self, local_dir: str) -> None:
        """
        Initialize the HFPusher with the local directory containing the model artifacts.
        
        Parameters:
            local_dir (str): Path to the local directory with the fine-tuned model and tokenizer.
        """
        self.local_dir: str = local_dir

    def login_to_hf(self) -> None:
        """
        Log in to the Hugging Face Hub using the user's credentials.
        """
        login()
        # Print the contents of the current directory for verification.
        print("Current directory contents:", os.listdir())

    def create_repository(self, repo_id: str) -> str:
        """
        Create a repository on the Hugging Face Hub.
        
        Parameters:
            repo_id (str): The repository identifier in the format 'username/repo_name'.
        
        Returns:
            str: The URL of the created repository.
        """
        api: HfApi = HfApi()
        repo_url: str = api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"Model repository created at: {repo_url}")
        return repo_url

    def push_model_to_hf(self, repo_id: str) -> None:
        """
        Push the model and tokenizer stored in the local directory to the Hugging Face Hub.
        
        Parameters:
            repo_id (str): The repository identifier where the model should be pushed.
        """
        # Load the model and tokenizer from local storage.
        model = AutoModelForCausalLM.from_pretrained(self.local_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.local_dir)

        # Push the model and tokenizer to the Hugging Face Hub.
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

        print(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")

    def run(self) -> None:
        """
        Run the entire process:
        1. Log in to the Hugging Face Hub.
        2. Ask the user for the repository name and username.
        3. Create the repository.
        4. Push the local model and tokenizer to the repository.
        """
        # Log in to the Hugging Face Hub.
        self.login_to_hf()

        # Ask the user for the repository name and username.
        repo_name: str = input("Enter your repository name (e.g., fine-tuned-deepseek-r1-1.5b): ").strip()
        username: str = input("Enter your Hugging Face username: ").strip()

        # Construct the repository ID.
        repo_id: str = f"{username}/{repo_name}"

        # Create the repository on Hugging Face Hub.
        self.create_repository(repo_id)

        # Push the model and tokenizer to the created repository.
        self.push_model_to_hf(repo_id)

def main() -> None:
    """
    Main function to instantiate HFPusher and run the process.
    """
    # Define the local directory where the model and tokenizer are saved.
    local_dir: str = "fine-tuned-deepseek-r1-1.5b-medical-o1"  # Adjust this path as needed.

    # Instantiate HFPusher with the local directory.
    pusher = HFPusher(local_dir=local_dir)
    
    # Run the process to push the model to the Hugging Face Hub.
    pusher.run()

if __name__ == "__main__":
    main()
