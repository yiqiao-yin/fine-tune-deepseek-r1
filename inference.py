#!/usr/bin/env python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_length: int) -> str:
    """
    Generate text from a given prompt using the provided model and tokenizer.
    
    This function tokenizes the prompt, generates output using the model, and decodes the result.
    
    Parameters:
        model (AutoModelForCausalLM): The fine-tuned language model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        prompt (str): The input prompt provided by the user.
        max_length (int): The maximum length of the generated sequence.
        
    Returns:
        str: The generated text output.
    """
    # Determine the available device: use CUDA if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move the model to the appropriate device
    model = model.to(device)
    
    # Tokenize the prompt and move inputs to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text with the model
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, temperature=0.7, top_k=50, top_p=0.9)
    
    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main() -> None:
    """
    Main function to load the fine-tuned model and tokenizer from local storage,
    prompt the user for a medical question and max token length, format the input using the inference prompt style,
    and generate text based on the provided prompt.
    
    The model is expected to be saved in the local directory with a name that was
    specified during training (e.g., 'fine-tuned-deepseek-r1-1.5b-some-ai-domain-v3').
    """
    # Define the local directory where the fine-tuned model and tokenizer are saved
    model_path = "fine-tuned-deepseek-r1-1.5b-some-ai-domain-v3"
    
    # Load the fine-tuned model and tokenizer from the specified directory
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prompt the user to enter a medical question
    user_question = input("Enter a medical question for text generation: ")
    
    # Prompt the user to enter the maximum token length for generation
    max_token_input = input("Enter the maximum token length for generation (e.g., 1024): ")
    try:
        max_token = int(max_token_input)
    except ValueError:
        print("Invalid input for maximum token length. Using default value 1024.")
        max_token = 1024

    # Define the inference prompt style template
    inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
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
    # For inference, fill in empty chain-of-thought and response parts
    prompt = inference_prompt_style.format(user_question, "", "")
    
    # Generate text from the provided prompt
    generated_output = generate_text(model, tokenizer, prompt, max_length=max_token)
    
    # Display the generated text
    print("\nGenerated text:")
    print(generated_output)

if __name__ == "__main__":
    main()
