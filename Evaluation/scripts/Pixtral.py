import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value, Image as HFImage
import json
import time
from argparse import ArgumentParser
from tqdm import tqdm
import os
import base64
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login
from dotenv import load_dotenv


# Load the Hugging Face dataset
def load_hf_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split, features=features)
    return dataset


# Define the features of the dataset
features = Features({
    "file_name": HFImage(),
    "ID": Value("string"),
    "Language": Value("string"),
    "Category": Value("string"),
    "Question_Type": Value("string"),
    "English_Question": Value("string"),
    "English_Answer": Value("string"),
    "Translated_Question": Value("string"),
    "Translated_Answer": Value("string"),
    "Image_Url": Value("string"),
})

def huggingface_image_to_data_url(hf_image, format: str = "JPEG") -> str:
    """
    Convert a Hugging Face image file to a data URL.
    
    Args:
        hf_image: Hugging Face image file object (e.g., a PIL Image).
        format (str): Desired image format (default is "JPEG").
    
    Returns:
        str: Data URL of the image.
    """
    # Save the image to an in-memory buffer
    from io import BytesIO
    buffer = BytesIO()
    hf_image.save(buffer, format=format)
    buffer.seek(0)  # Rewind to the beginning of the buffer

    # Encode the image data as Base64
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Determine MIME type based on format
    mime_type = f"image/{format.lower()}"

    # Construct and return the data URL
    return f"data:{mime_type};base64,{encoded_string}"

# Collate function
def collate_fn(batch):
    images = [item['file_name'] for item in batch]  # Images
    questions = [item['Translated_Question'] for item in batch]
    answers = [item['Translated_Answer'] for item in batch]
    question_types = [item['Question_Type'] for item in batch]
    language = [item['Language'] for item in batch]
    ids = [item['ID'] for item in batch]

    return {
        'images': images,
        'questions': questions,
        'answers': answers,
        'question_type': question_types,
        'ids': ids,
        'language': language
    }

# Evaluate function
def evaluate(model, dataloader, processor, device):
    results = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch") as pbar:
            for batch in dataloader:
                images = batch['images']
                questions = batch['questions']
                ground_truth = batch['answers']
                question_types = batch['question_type']
                ids = batch['ids']
                languages = batch['language']

                for img, question, answer, question_type, id_, lang in zip(images, questions, ground_truth, question_types, ids, languages):
                    try:
                        # Generate the prompt based on the question type
                        if question_type == "Multiple Choice Questions":
                            if "（" in answer:
                                choices = answer.split("（")[1].split("）")[0]
                            else:
                                choices = answer.split("(")[1].split(")")[0]
                            prompt = "For the given Multiple Choice Question, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text.\n" + question + "\n" + choices
                        elif question_type == "True False Question":
                            prompt = question + f"\nThe above question is a True/False question. Please provide the answer as one word in {lang}"
                        elif question_type == "Long Question":
                            prompt = question + f"\nAnswer the question in detail in {lang} language."
                        else:
                            prompt = question + f"\nPlease provide brief, clear responses in {lang} language."
                        
                        image = huggingface_image_to_data_url(img)
                        messages = [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image}}]
                            },
                        ]

                        outputs = llm.chat(messages, sampling_params=sampling_params)

                        predicted_answer = outputs[0].outputs[0].text
                        
                        results.append({
                            "id": id_,
                            "lang": lang,
                            "predicted_answer": predicted_answer,
                            "ground_truth": answer,
                            "question": question,
                            "prompt": prompt
                        })
                    except Exception as e:
                        print(f"Error in prediction: {e}")
                        print(f"Question: {question}")
            pbar.update(1)

    return results

# Main evaluation function
def main(dataset_name, split, processor, batch_size=32, save_path="results.json", device="cuda"):
    all_results = {}
    dataset = load_hf_dataset(dataset_name, split)

    # for testing purposes
    # dataset = dataset.select(range(10))


    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Perform evaluation
    scores = evaluate(None, dataloader, processor, device)

    with open(save_path, "w") as f:
        json.dump(scores, f, indent=4, default=str)

    print(f"Results saved to {save_path}.")

if __name__ == "__main__":
    time1 = time.time()

    default_output_path = os.path.join(os.getcwd(), "results.json")
    # add args
    parser = ArgumentParser()
    # device
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    # dataset
    parser.add_argument("--dataset_name", type=str, default="MBZUAI/ALM-Bench", help="Dataset name")
    # split
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    # batch size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    # save_path
    parser.add_argument("--save_path", type=str, default=default_output_path, help="Path to save the results")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the token from the environment variable
    token = os.getenv("GENAI_API_KEY")

    login(token=token)

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the model and tokenizer
    model_name = "mistralai/Pixtral-12B-2409"

    sampling_params = SamplingParams(max_tokens=8192)

    llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=32768, device=device)

    dataset_name = args.dataset_name
    split = args.split
    main(dataset_name, split,None, args.batch_size, args.save_path, args.device)
    print(f"Time taken: {time.time() - time1} seconds")
