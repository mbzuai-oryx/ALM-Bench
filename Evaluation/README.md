## Installation 
We provide installation instructions for:
- Setting up environments for evaluating LMMs on ALM-Bench
- GPT-Scoring evaluation for LLM as a judge
- Image Scrapping code.

## Installing the requirements 
Note: Ensure that the following packages are installed. Each LMM might have a slightly different requirement for torch and transformer versions; refer to their repositories for the exact versions. 
We created multiple environments for different models to run them individually. 

```shell
torch
datasets
transformers
tqdm
Pillow
python-magic
beautifulsoup4
python-dotenv
google-generative-ai
requests
progressbar2
```

Once the requirements are installed, you can run the scripts using the following command:
```shell
python3 -m llava15.py --device cuda --dataset_name MBZUAI/ALM-Bench --split test --batch_size BATCH_SIZE --save_path PATH_TO_SAVE_RESULTS
```
