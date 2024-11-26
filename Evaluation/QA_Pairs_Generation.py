import os
import json
import base64
from openai import OpenAI
import openai
from tqdm import tqdm
from argparse import ArgumentParser

client= OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_files_from_subfolder(subfolder_path, language, save_path):
    dl = []
    invalid_image_count = 0
    # image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heif", ".heic", ".raw", ".cr2", ".nef", ".svg", ".eps"]
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            file_path = os.path.join(root, file)


            if file_path.endswith("json"):
                new_file_path = file_path.replace("Caption", "caption")
                with open(new_file_path, 'r') as f:
                    data = json.load(f)
                    data = {key.lower(): value for key, value in data.items()}

                    caption = data.get('caption', None)
                    image_url = data.get('url', None)  # Assuming the path to the image is stored in the 'path' key
                    landesc = data.get('path', None)
                    image_file_name = data.get('file_name', None)
                    search_query = data.get('query', None)
                    
                    if caption == None or image_url == None or landesc == None or image_file_name == None or search_query == None:
                        print(f"Missing data in JSON file: {file_path}")
                    sub_landesc = landesc.split("\\")[-1]


                    json_file_name = os.path.basename(file_path).split(".")[0]
                    image_file_name_without_extension = image_file_name.split(".")[0]

                    folder_path = os.path.dirname(file_path)
                    image_path = os.path.join(folder_path, image_file_name)



                    base64_image = encode_image(image_path)
                    if caption:
                    
                        PROMPT_MESSAGES = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": search_query + "\n" + caption
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                            
                                        }
                                    }
                                ]
                            },
                            {
                                "role": "system",
                                "content": f"""Given an image and its caption, generate two short questions and answers, two multiple-choice questions and answers, one true/false question, and one long question and answer. Refer to the caption for the context/hint. Take into account the cultural diversity of {sub_landesc}"""
                                + """
                                Follow the following rules while designing questions and answers:
                                1. The question must be answerable only by looking at the image.
                                2. Ensure that the questions are culturally relevant and specific to the image.
                                3. Provide answers that are concise, accurate, and directly related to the question.
                                4. You will also need to provide 1 correct option and 3 other incorrect options (distractors). 
                                   For the distractors, choose options that are relevant, not obvious wrong answers.
                                5. The question must be answerable even without the multiple-choice.
                                   Example of the invalid question: (“What song is not performed by this musician” – 
                                   not answerable if you don’t know the choices).
                                6. Make sure the questions are written fluently in English.
                                7. Be mindful of cultural sensitivities and avoid stereotyping or misrepresenting cultural aspects.
                                8. Ensure there are variations in your questions. Identity questions are fine, eg “What is this”, 
                                   or “where is this”. But additionally adding more complex/difficult questions would be great. 
                                   For example, multi-hop reasoning, counting, referencing, or questions that require local commonsense knowledge to be answered.
                                9. Just generate these in English.
                                10. For short questions and answers, don't keep it very short, include at least 2 sentences.
                                11. Make the questions distinct and unique from each other.
                                Give the answers in the following JSON format and make sure to only output a valid JSON,
                                {
                                    "short_questions": [
                                        {
                                            "question": "What is the name of this building?",
                                            "answer": "Eiffel Tower"
                                        },
                                        {
                                            "question": "What is the name of this building?",
                                            "answer": "Eiffel Tower"
                                        }
                                    ],
                                    "multiple_choice_questions": [
                                        {
                                            "question": "What is the name of this building?",
                                            "answer": "Eiffel Tower",
                                            "options": ["Eiffel Tower", "Empire State Building", "Burj Khalifa", "Petronas Towers"]
                                        },
                                        {
                                            "question": "What is the name of this building?",
                                            "answer": "Eiffel Tower",
                                            "options": ["Eiffel Tower", "Empire State Building", "Burj Khalifa", "Petronas Towers"]
                                        }
                                    ],
                                    "true_false_question":
                                    {
                                        "question": "Does this image depict/symbolize some history",
                                        "answer": "True, it does"    
                                    },
                                    "long_question": {
                                        "question": "What is the name of this building? Describe why it was originally built, the initial public reception, and how it became a global cultural icon over time.",
                                        "answer": "The Eiffel Tower, constructed between 1887 and 1889, was originally built as the entrance arch to the 1889 World’s Fair, held in Paris to commemorate the 100th anniversary of the French Revolution. Designed by engineer Gustave Eiffel, the tower was initially met with mixed reactions. Many Parisians, including prominent artists and intellectuals, criticized its modern design, labeling it as an eyesore. However, despite the criticism, the Eiffel Tower quickly gained recognition as a groundbreaking achievement in engineering and design. Standing at 300 meters tall (about 984 feet), it was the tallest man-made structure in the world at the time, made possible by the innovative use of wrought iron."
                                    }
                                }
                                """
                            }
                        ]

                        try:
                            assert image_file_name_without_extension == json_file_name, f"Image file name {file_path} does not match JSON file name {json_file_name}"
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=PROMPT_MESSAGES
                                                            )

                            t_data = {
                                "image_url": image_url,
                                "questions": response.choices[0].message.content,
                                "langdesc": landesc,
                                "caption": caption
                            }
                            dl.append(t_data)

                        except Exception as e:
                            if "invalid_image" in str(e):
                                invalid_image_count += 1
                                print(f"Invalid image: {image_url}")
                            else:
                                print(f"Error processing image {image_url}: {e}")

                with open(f"{save_path}/{language}.json", 'w') as f:
                    json.dump(dl, f, indent=2, default=str)

    print(f"Number of images with errors: {invalid_image_count}")


def main():

    # add arguments
    parser = ArgumentParser()

    parser.add_argument("--folder_path", type=str, help="Path to the folder containing the subfolders with images and JSON files")
    parser.add_argument("--save_path", type=str, help="Path to save the output JSON files")


    args = parser.parse_args()

    for language in tqdm(os.listdir(args.folder_path)):
        subfolder_path = os.path.join(args.folder_path, language)
        print(f'language {language}')
        read_files_from_subfolder(subfolder_path, language, args.save_path)

