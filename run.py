#!/usr/bin/env python
import os
from trainer import FineTuner

def main() -> None:
    """
    Main function to run the fine-tuning pipeline.

    This function instantiates the FineTuner with a specified model name and dataset identifier,
    processes the data by loading a remote dataset and formatting prompts,
    sets up training configuration including LoRA and Trainer parameters,
    fits the model using the processed dataset, and finally saves the fine-tuned model 
    and tokenizer locally.
    """
    # Define the model name identifier for fine-tuning
    default_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model_name = input(f"Enter model name (default: {default_model_name}): ") or default_model_name

    # Define the dataset identifier for fine-tuning
    default_dataset_identifier = "FreedomIntelligence/medical-o1-reasoning-SFT"
    dataset_identifier = input(f"Enter dataset identifier (default: {default_dataset_identifier}): ") or default_dataset_identifier

    # Define the domain identifier for fine-tuning
    default_domain = "name-of-domain-you-want"
    domain = input(f"Enter domain (default: {default_domain}): ") or default_domain
    
    # Instantiate the FineTuner object with the provided model name, dataset identifier, and domain
    fine_tuner = FineTuner(model_name=model_name, text=dataset_identifier, domain=domain)
    
    # Process the remote dataset by formatting prompts and tokenizing the text
    fine_tuner.process_data()
    
    # Set up the LoRA configuration and training parameters
    fine_tuner.set_up_training_config()
    
    # Execute the training process
    fine_tuner.fit()
    
    # Save the fine-tuned model and tokenizer to a local directory
    fine_tuner.save_model()

if __name__ == "__main__":
    main()
