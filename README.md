# ExpertChat
Studying the interaction of LLM models when discussing different topics. Using open-source models in the Hugging Face Transformers library.

## Overview
The goal of this project is to analyze the interaction between different AI models and explore ways to fine-tune them for more expert-like communication. The approach is inspired by a bidirectional GAN (Generative Adversarial Network) concept, where each model attempts to convince the other that they are human experts rather than AI. After several interactions, the models rate each other across multiple categories, and these ratings are used to fine-tune them, improving their ability to communicate convincingly. 

Ratings are first determined by a numerical score from 1-10 (10 being high). The second rating metric is a WIM (What Is Missing) response from the judging model. The WIM response will be compared to the actor model's response using cosine similarity to "rate" the knowledge content. The idea is that the actor will try to maximize the knowledge content given in its response. [Online DPO](https://huggingface.co/papers/2402.04792) (Direct Preference Optimization) is used to fine-tune the models based on these rewards.

## Approach
I created a custom ExpertChat parent class that allows the seamless interaction between two LLMs. Each model has its own child class for model specific parameters. A conversation loops gives a straightforward method to create a topic, create interaction between the models, and rate each model after the conversation. Using the TRL toolkit for training. 

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

    - **all-mpnet-base-v2 (Sentence Transformer)**  
      - URL: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main)  
      - Clone:  
       ```bash
       clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
       ```

  Note: The job is written to use zip files to speed up GPU load time on the cluster. Make sure to use -r flag.

## Required Libraries

All dependencies are listed in the `requirements.txt`. The key libraries include:

- Huggingface `transformers`
- `Pytorch`
- Flash Attention (Only on A100 GPUs, not included in current implementation)
- Huggingface `trl`
- Sentence Transformers

To install the required packages:
```bash
pip install --no-index transformers
pip install --no-index torch
pip install --no-index bitsandbytes
pip install --no-index -U flash-attn --no-build-isolation (Only on A100 GPUs)
pip install --no-index -U sentence-transformers
pip install --no-index trl
```

## GPU RAM Calculation

On the Beluga cluster, each node provides 64GB of GPU RAM, 16GB in each GPU. By using quantization with 4-bit parameters, the Mixtral model will require approximately 27GB of GPU memory. Llama 3-8B is listed to require 16Gb of GPU memory. Running both models requires 43GB. I found that there was just under the required amount of memory with 3 GPUs. I requested 4 GPUs for a total of 64GB GPU RAM. RAM is also required for the sentence transformer model but 64GB should still be sufficient.


## Running on the Cluster

To run the models on a cluster, use the following command:
```bash
sbatch llm-interaction-job.sh
```

To run the Online DPO trainer on a cluster, use the following command:
```bash
sbatch odpo-trainer-job.sh
```

Make sure all file paths in the script are correctly set for the cluster's file system.

## Note on Generation

I am using contrastive search as seen in this [blog post](https://huggingface.co/docs/transformers/en/generation_strategies). The parameters are automatically set to defaults but can be changed for tasks like topic generation. Alternatively, high temperature sampling can produce better results in terms of training.

## Online DPO

To stay within the Hugging Face toolset, I will be using the TRL library found [here](https://huggingface.co/docs/trl/index).

## Sentence Transformers

Producing useful embeddings is important to actually tell the model what knowledge is missing. I am using [Sentence Transformers](https://huggingface.co/sentence-transformers) to extract the useful embeddings. These are usually used for semantic search and is more useful than the standard LLM tokenizers.

## Helpful and Similar Papers

- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267)
- [ETA-REWARDING LANGUAGE MODELS: Self-Improving Alignment with LLM-as-a-Meta-Judge](https://arxiv.org/pdf/2407.19594)
- [Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://arxiv.org/pdf/2312.01823)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685)
- [Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020)
- [WEAK-TO-STRONG GENERALIZATION: ELICITING STRONG CAPABILITIES WITH WEAK SUPERVISION](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)