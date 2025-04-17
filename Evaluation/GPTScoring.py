import openai
import json
import pandas as pd
from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from datasets import load_dataset
from datasets import load_dataset, Features, Value, Image as HFImage


# Load environment variables from .env file
load_dotenv()

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

# Retrieve the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file!")

# Function to perform GPT-4 scoring for a single prediction
def gpt4_score(prompt, test_mode=False):
    if test_mode:
        print("\n--- GPT-4 Prompt (TEST MODE) ---")
        print(prompt)
        print("--- End Prompt ---\n")
        return "TEST_SCORE"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        score_text = response.choices[0].message.content.strip()
        return score_text
    except Exception as e:
        print(f"Error in GPT-4 call: {e}")
        return 0


# Function to evaluate predictions and add scores
def evaluate_predictions(results_path, dataset, output_path, test_mode=False):
    with open(results_path, 'r') as f:
        results = json.load(f)

    for lang in results.keys():
        

        dataset_lang = dataset.filter(lambda x: x['Language'] == lang)
        for i in range(len(results[lang])):
            question = results[lang][i]['question']
            id = results[lang][i]['id']

            # get the question row from the id
            question_row = dataset_lang.filter(lambda x: x['ID'].strip() == id.strip())
            assert len(question_row) == 1, f"{len(question_row)} entries found for ID: {id}"
            question_row = question_row[0]
            
            if len(question_row) == 0:
                print(f"Question not found: {question} in {lang}")
                continue

            # Add question type to the results
            results[lang][i]['question_type'] = question_row['Question_Type']
            results[lang][i]['Translated_Answer'] = question_row['Translated_Answer']
            results[lang][i]['Translated_Question'] = question_row['Translated_Question']

    for lang in results.keys():
        for i in range(len(results[lang])):
            entry = results[lang][i]
            question = entry["Translated_Question"]
            ground_truth = entry["Translated_Answer"]
            predicted_answer = entry["predicted_answer"]
            
            question_type = entry.get("question_type")
            if not question_type:
                print(f"Missing question type for question: {question} in {lang}")
                continue
            
            # Prepare prompt based on question type
            if question_type == "True False Question":
                prompt_eval = (
                    f"Evaluate the following answer based on Accuracy:\n\n"
                    f"Question: {question}\n"
                    f"Ground Truth: {ground_truth}\n"
                    f"Model Prediction: {predicted_answer}\n\n"
                    f"Match the meaning of the ground truth with the model prediction and if it matches give a 10. Otherwise 0.\n"
                    f"Strictly return only the numeric score, without any additional commentary."
                )
                print(prompt_eval)
            elif question_type == "Multiple Choice Questions":
                if "（" in ground_truth:
                    choices = ground_truth.split("（")[1].split("）")[0]
                    answer = ground_truth.split("（")[0]
                else:
                    answer = ground_truth.split("(")[0]
                prompt_eval = (
                    f"Evaluate the following answer based on Accuracy:\n\n"
                    f"Question: {question}\n"
                    f"Ground Truth: {answer}\n"
                    f"Model Prediction: {predicted_answer}\n\n"
                    f"Match the meaning of the ground truth with the model prediction and if it matches give a 10. Otherwise 0.\n"
                    f"Strictly return only the numeric score, without any additional commentary."
                )
            elif question_type == "Long Question":
                prompt_eval = (
                    f"Evaluate the following answer based on Consistency, Fluency, and Relevance based on the Ground Truth answer.\n"
                    f"A high score example will be where the predicted response matches closely with the ground truth and a low score example will be where the predicted response lacks knowledge or is not related to the ground truth:\n\n"
                    f"Question: {question}\n"
                    f"Ground Truth: {ground_truth}\n"
                    f"Model Prediction: {predicted_answer}\n\n"
                    f"Provide a single overall score out of 10 based on these three criteria.\n"
                    f"Strictly return only the numeric score, without any additional commentary."
                )
            else:
                prompt_eval = (
                    f"Evaluate the following answer based on accuracy and correctness based on the Ground Truth answer.\n"
                    f"A high score example will be where the predicted response matches closely with the ground truth and a low score example will be where the predicted response lacks knowledge or is not related to the ground truth:\n\n"
                    f"Question: {question}\n"
                    f"Ground Truth: {ground_truth}\n"
                    f"Model Prediction: {predicted_answer}\n\n"
                    f"Provide a single overall score out of 10 based on these three criteria.\n"
                    f"Strictly return only the numeric score, without any additional commentary."
                )

            # Call GPT-4 for scoring
            score = gpt4_score(prompt_eval, test_mode=test_mode)
            results[lang][i]["score"] = score

    # Save the updated results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

def test_evaluate_predictions(dataset):
    # Provide a small sample JSON to test
    sample_results = {
        "English": [
            {   "id" : "001_1_01_001",
                "question": "What type of outerwear is the woman in the foreground wearing?",
                "predicted_answer": "The woman in the foreground is wearing a tan coat.",
            }
        ]
    }

    # Save a temporary test file
    with open("test_results.json", "w") as f:
        json.dump(sample_results, f, indent=4)

    # Call evaluate_predictions with test_mode=True
    evaluate_predictions(
        results_path="test_results.json",
        dataset=dataset,
        output_path="test_output.json",
        test_mode=True
    )


# Main function with argument parser
if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate predictions using GPT-4.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the JSON file containing predictions.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the updated results JSON file.")
    parser.add_argument("--dataset_name", type=str, default="MBZUAI/ALM-Bench", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")

    args = parser.parse_args()
    
    dataset = load_hf_dataset(args.dataset_name, args.split)
    

    evaluate_predictions(args.results_path, dataset, args.output_path, test_mode=False)
    # Uncomment the following line to test with a small sample
    # test_evaluate_predictions(dataset)