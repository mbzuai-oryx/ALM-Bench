import openai
import json
import pandas as pd
from argparse import ArgumentParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file!")

# Function to perform GPT-4 scoring for a single prediction
def gpt4_score(prompt):
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
def evaluate_predictions(results_path, original_files_path, output_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    for lang in results.keys():
        # Load the original Excel file for the corresponding language
        try:
            df = pd.read_excel(f"{original_files_path}/{lang}.xlsx")
        except FileNotFoundError:
            print(f"File not found: {original_files_path}/{lang}.xlsx")
            continue

        for i in range(len(results[lang])):
            question = results[lang][i]['question']
            question_row = df[df['Translated_Question'] == question]
            
            if question_row.empty:
                print(f"Question not found: {question} in {lang}")
                continue

            # Add question type to the results
            results[lang][i]['question_type'] = question_row['Question_Type'].values[0]

    for lang in results.keys():
        for i in range(len(results[lang])):
            entry = results[lang][i]
            question = entry["question"]
            ground_truth = entry["ground_truth"]
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
            score = gpt4_score(prompt_eval)
            results[lang][i]["score"] = score

    # Save the updated results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

# Main function with argument parser
if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate predictions using GPT-4.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the JSON file containing predictions.")
    parser.add_argument("--original_files_path", type=str, required=True, help="Path to the directory containing original Excel files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the updated results JSON file.")
    
    args = parser.parse_args()
    
    evaluate_predictions(args.results_path, args.original_files_path, args.output_path)
