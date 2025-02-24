# fine-tune-deepseek-r1
This package is built to assist in fine-tuning deepseek-like models.

## Overview
This package provides scripts to fine-tune a deepseek-like model and perform inference using the fine-tuned model. It includes:

- **run.py**: Fine-tuning pipeline that loads a dataset, processes and tokenizes the data, sets up training configuration, fine-tunes the model, and saves the resulting artifacts.
- **inference.py**: A script to load the fine-tuned model and generate text from a user prompt.

## Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (if available) or CPU
- The dependencies are listed in the `requirements.txt` file.

## Setting Up the Virtual Environment

### On macOS / Linux

1. **Create a virtual environment:**
   ```bash
   python3 -m venv env
   ```

2. **Activate the virtual environment:**
   ```bash
   source env/bin/activate
   ```

### On Windows

1. **Create a virtual environment:**
   ```cmd
   python -m venv env
   ```

2. **Activate the virtual environment:**
   ```cmd
   .\env\Scripts\activate
   ```

## Installing Dependencies
With the virtual environment activated, install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Fine-Tuning Pipeline
The `run.py` script starts the fine-tuning process:
1. It loads a remote dataset using a specified dataset identifier.
2. It processes the data by formatting prompts and tokenizing them.
3. It sets up the training configuration using LoRA settings.
4. It fine-tunes the model and saves both the model and tokenizer locally.

To run the fine-tuning pipeline:
```bash
python run.py
```

## Running Inference
After fine-tuning, you can generate text using the `inference.py` script:
1. The script loads the fine-tuned model and tokenizer from local storage.
2. It prompts you to enter a medical question and specify the maximum token length.
3. It generates and displays the output based on the provided prompt.

To run the inference script:
```bash
python inference.py
```

## Additional Notes
- Ensure that you have proper access permissions and credentials for any remote datasets (e.g., Hugging Face datasets).
- You may need to adjust parameters like model names, dataset identifiers, and training settings in the scripts according to your needs.
- If you encounter GPU-related issues, verify that your system’s drivers and CUDA installation (if applicable) are correctly set up.

Enjoy fine-tuning with deepseek-r1!