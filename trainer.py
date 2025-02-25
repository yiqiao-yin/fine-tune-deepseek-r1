#!/usr/bin/env python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import gc
from typing import Optional, Dict, Any

class FineTuner:
    def __init__(self, model_name: str, text: str, domain: str = "default-domain") -> None:
        """
        Initialize the FineTuner with a model, tokenizer, dataset identifier, and domain identifier.
        
        Parameters:
            model_name (str): The name or path of the pre-trained model.
            text (str): The dataset identifier used for loading the remote dataset 
                        (e.g., "FreedomIntelligence/medical-o1-reasoning-SFT").
            domain (str): Domain identifier for naming the saved model.
        """
        self.model_name: str = model_name
        self.model_name_tag: str = self.model_name.split("/")[1]
        self.text: str = text  # This now represents the dataset identifier
        self.domain: str = domain

        # Load pre-trained model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set device to GPU if available, else CPU
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Placeholders for later configuration
        self.tokenized_dataset: Optional[Dataset] = None
        self.training_args: Optional[TrainingArguments] = None
        self.trainer: Optional[Trainer] = None
        self.lora_config: Optional[LoraConfig] = None

    def process_data(self) -> None:
        """
        Process the remote dataset by formatting prompts and tokenizing the text.
        
        This method loads the dataset using the dataset identifier stored in self.text,
        applies a custom prompt formatting function to combine the question, complex 
        chain-of-thought, and response into a single prompt, and then tokenizes the 
        resulting texts for causal language modeling training.
        """
        # Load the dataset using self.text as the dataset identifier
        dataset = load_dataset(self.text, "en", split="train[0:500]", trust_remote_code=True)
        
        # Define the prompt style template
        train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""
        
        # Function to format the prompts based on the dataset fields
        def formatting_prompts_func(examples: Dict[str, Any]) -> Dict[str, Any]:
            inputs = examples["Question"]
            cots = examples["Complex_CoT"]
            outputs = examples["Response"]
            texts = []
            for question, cot, output in zip(inputs, cots, outputs):
                # Append the tokenizer's end-of-sequence token after the formatted prompt
                text = train_prompt_style.format(question, cot, output) + self.tokenizer.eos_token
                texts.append(text)
            return {"text": texts}
        
        # Map the formatting function to the dataset in batched mode
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        # Define a preprocessing function for tokenization
        def preprocess_function(examples: Dict[str, Any], key: str = 'text') -> Dict[str, Any]:
            inputs = self.tokenizer(
                examples[key], truncation=True, padding="max_length", max_length=512
            )
            # For causal LM training, set labels as a shifted copy of input_ids
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
        
        # Apply tokenization to the formatted dataset
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        self.tokenized_dataset = tokenized_dataset

    def set_up_training_config(self) -> None:
        """
        Set up the training configuration including LoRA settings and Trainer parameters.
        """
        # Define the LoRA configuration for the model
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        
        # Wrap the model with LoRA using the configuration
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Define training arguments for the Trainer
        self.training_args = TrainingArguments(
            per_device_train_batch_size=1,  # Adjusted for GPU memory limitations
            gradient_accumulation_steps=8,  # To simulate a larger batch size
            warmup_steps=200,
            num_train_epochs=1000,  # Control training duration via number of epochs
            learning_rate=2e-4,
            fp16=True,  # Enable mixed precision training for faster computation
            logging_steps=10,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Initialize the Trainer with the model, training arguments, and the tokenized dataset
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset,
        )
    
    def fit(self) -> None:
        """
        Run the training process.
        """
        # Move model to CPU to free up GPU memory before training configuration
        self.model = self.model.to("cpu")
        
        # Free up memory before starting training
        gc.collect()  # Run garbage collection
        torch.cuda.empty_cache()  # Clear CUDA cache
        
        # Optimize model with torch.compile for improved execution speed (requires PyTorch 2.0+)
        self.model = torch.compile(self.model)
        
        # Move model back to GPU for training if available
        self.model = self.model.to("cuda") if torch.cuda.is_available() else self.model.to("cpu")
        
        # Start training using the Trainer
        self.trainer.train()
    
    def save_model(self, save_directory: Optional[str] = None) -> None:
        """
        Save the fine-tuned model and tokenizer locally.
        
        Parameters:
            save_directory (str, optional): Directory where the model and tokenizer will be saved.
                If not provided, a default name based on the domain is used.
        """
        if save_directory is None:
            save_directory = f"{self.model_name_tag}-{self.domain}"
        
        # Save the model and tokenizer to the specified directory
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved in directory: {save_directory}")
