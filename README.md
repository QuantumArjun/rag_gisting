# Adaptive GISTs to Create a Cache for Retrieval Augmented Generation

[Link to our Final Project Report](https://github.com/QuantumArjun/rag_gisting/blob/f6976622783f6cd0b2fef2704f7aa2240d1f312a/CS_229s_Final_Project_Report.pdf)

## Problem Statement
The issue of better utilizing the limited space of context length during Retrieval Augmented Generation (RAG) and increasing the spatial and temporal efficiency of knowledge base retrieval for a large language model is an open problem. Recent research investigates compressing prompts with gist tokens to encode textual instruction into a single representative token to reduce memory utilization and appears promising at enabling larger
contexts without a quadratic increase in computation. In this project, we show that we can further improve RAGs by implementing gist tokens to encode longer-sequence length passages (e.g., the retrieved documents) and applying a robust caching strategy to access relevant documents quicker and drastically reduce the length of document-specific information allocated in the context window. The original paper that ideated gist tokens can be found at: ["Learning to compress prompts with gist tokens"](https://arxiv.org/abs/2304.08467). The original use case was for caching LLM instructions, we extend to be used for RAG.  

## Dataset
We used the Wikipedia TriviaQA Dataset made by the University of Washington, which asks
trivia questions that are obscure enough that a language model could not possibly encode all of
the information in its weights. It contains over 650K question-answer-evidence triples, where the
evidence is sourced from relevant Wikipedia articles. We used the existing HuggingFace datasets
train test validation split for our procedures

## Setup
We tested this on GCP with pytorch 2.0, python 3.10, and an NVIDIA V100 GPU. In a conda environment, run
~~~ 
pip install -r requirements.txt
~~~

## Training
Edit `src/conf/config.yaml` to edit `entity` with wandb username.
Run
~~~
./train.sh
~~~

## Evaluations
Edit `eval_rag.py` and include the appropriate `gist_model` path from the `exp` folder. 
When calling `gist_compress`, edit the number of gist tokens to amount desired. Our optimal number was 2. 
Run:
~~~
python -m src.eval_rag
~~~
