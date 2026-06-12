# Mini LLM Next-Word Predictor

🚀 **Live Demo:** [https://mini-llm-next-word-predictor.onrender.com](https://mini-llm-next-word-predictor.onrender.com)

This is a complete, runnable project that trains a small next-word prediction language model (a mini LLM) on Simple English text, and provides a simple Gradio web front end for interactive predictions.

## Requirements

Python 3.9+ is required.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**
   Download the datasets (Project Gutenberg and WikiText-2), clean the text, train the BPE tokenizer, and create the PyTorch dataset splits.
   ```bash
   python data_prep.py
   ```

2. **Training**
   Train the GPT-style decoder-only Transformer. The script saves the best model checkpoint to the `model_output` directory.
   ```bash
   python train.py
   ```

3. **Web Front End**
   Launch the Gradio web interface to test the model interactively.
   ```bash
   python app.py
   ```
   Open the local URL provided in the terminal in your browser.

## Architecture

- **Model**: Decoder-only Transformer (Hugging Face GPT2 implementation)
- **Context Length**: 128
- **Embedding Dimension**: 256
- **Layers**: 6
- **Attention Heads**: 8
- **Vocabulary Size**: 8000
- **Activation**: GELU
