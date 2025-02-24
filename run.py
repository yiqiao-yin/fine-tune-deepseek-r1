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
    # Define the model name and dataset identifier for fine-tuning
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    dataset_identifier = "FreedomIntelligence/medical-o1-reasoning-SFT"
    domain = "some-ai-domain-v3"
    
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
