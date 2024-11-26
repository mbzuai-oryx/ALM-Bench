import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value, Image as HFImage
import json
import time
from argparse import ArgumentParser
from tqdm import tqdm
import os
import base64
import openai
from io import BytesIO

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

# Load the Hugging Face dataset
def load_hf_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split, features=features)
    return dataset

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

def gpt4_with_image(prompt, hf_image):
    try:
        # Convert the Hugging Face image (PIL or array) to Base64
        buffer = BytesIO()
        hf_image.save(buffer, format="PNG")  # Ensure the image is saved in PNG format
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Call GPT-4o with the image and prompt
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": prompt}
                ]}
            ],
        )

        # Extract the score text from the response
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        print(f"Error in GPT-4 call: {e}")
        return 0

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
                        
                        
                        predicted_answer = gpt4_with_image(prompt, img)

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
def main(dataset_name, split, batch_size=32, save_path="results.json", device="cuda"):
    all_results = {}
    dataset = load_hf_dataset(dataset_name, split)

    # # for testing purposes
    # dataset = dataset.select(range(10))


    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    scores = evaluate(None, dataloader, None, device)

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

    # Specify the model ID
    model_id = "gpt-4o"
    dataset_name = args.dataset_name
    split = args.split
    main(dataset_name, split, args.batch_size, args.save_path)
    print(f"Time taken: {time.time() - time1} seconds")
