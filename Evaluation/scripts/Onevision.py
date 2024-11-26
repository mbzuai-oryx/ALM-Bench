import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value, Image as HFImage
import json
import time
from argparse import ArgumentParser
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import os

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

# Evaluate function
def evaluate(model, dataloader, processor, device):
    model.eval()
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
                        
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        
                        # Process inputs for the model
                        image_tensor = process_images([img], image_processor, model.config)
                        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

                        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
                        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                        conv = copy.deepcopy(conv_templates[conv_template])
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()

                        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                        image_sizes = [img.size]


                        cont = model.generate(
                            input_ids,
                            images=image_tensor,
                            image_sizes=image_sizes,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=2048,
                        )
                        predicted_answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
                        
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

    # for testing purposes
    # dataset = dataset.select(range(10))


    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Perform evaluation
    scores = evaluate(model, dataloader, None, device)

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

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the model and tokenizer
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-si"
    model_name = "llava_qwen"
    device_map = device
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

    model.to(device)

    dataset_name = args.dataset_name
    split = args.split
    main(dataset_name, split, args.batch_size, args.save_path, device)
    print(f"Time taken: {time.time() - time1} seconds")
