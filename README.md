# ExpertChat
Studying the interaction of LLM models when discussing different topics. Using open-source models in the Hugging Face Transformers library.

## Overview
The goal of this project is to analyze the interaction between different AI models and explore ways to fine-tune them for more expert-like communication. The approach is inspired by a bidirectional GAN (Generative Adversarial Network) concept, where each model attempts to convince the other that they are human experts rather than AI. After several interactions, the models rate each other across multiple categories, and these ratings are used to fine-tune them, improving their ability to communicate convincingly. Reinforcement Learning with Human Feedback (RLHF) will be used to further refine this process.

## Approach
The core idea is to allow two models to "compete" by convincing each other of their expertise on a given topic. Each model engages in a back-and-forth dialogue, rating the other on various aspects of communication. This feedback will then be used to fine-tune the models using RLHF, pushing them towards more expert-level conversational capabilities.

## Models

To work with the models in this project, ensure you're using `git lfs` (Git Large File Storage) for efficient model downloading. Here’s how to set up and retrieve the models:

1. **Install Git LFS**  
   ```bash
   module load git-lfs/3.4.0
   git lfs install
   ```

2. **Pull Models**  
   Navigate into the desired directory and use the following commands to pull the models:

   - **Mixtral 8x7B Instruct (with Flash Attention)**  
     - URL: [Mixtral 8x7B Instruct v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main)  
     - Clone:  
       ```bash
       git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
       ```

   - **Llama3 8B Instruct**  
     - URL: [Llama3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main)  
     - Clone:  
       ```bash
       git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
       ```

## Required Libraries

All dependencies are listed in the `requirements.txt`. The key libraries include:

- Huggingface `transformers`
- `Pytorch`
- Flash Attention (Only on A100 GPUs, not included in current implementation)

To install the required packages:
```bash
pip install --no-index transformers
pip install --no-index torch
pip install --no-index bitsandbytes
pip install -U flash-attn --no-build-isolation (Only on A100 GPUs)
```

## GPU RAM Calculation

On the Beluga cluster, each node provides 64GB of GPU RAM, 16GB in each GPU. By using quantization with 4-bit parameters, the Mixtral model will require approximately 27GB of GPU memory. Llama 3-8B is listed to require 16Gb of GPU memory. Running both models requires 43GB. I found that there was just under the required amount of memory with 3 GPUs. I requested 4 GPUs for a total of 64GB GPU RAM.


## Running on the Cluster

To run the models on a cluster, use the following command:
```bash
sbatch llm-job.sh
```
Make sure all file paths in the script are correctly set for the cluster's file system.

## Note on Generation

I am using contrastive search as seen in this [blog post](https://huggingface.co/docs/transformers/en/generation_strategies). The parameters are automatically set to defaults but can be changed for tasks like topic generation.
