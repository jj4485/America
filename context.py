import json
from openai import OpenAI
from gpt_key import gpt_key
import os
import time

# Set up the OpenAI client xwith the API key\
gpt_key =  os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=gpt_key)

def split_text_into_chunks(text, max_tokens=2000):
    """Splits a large text into smaller chunks to fit within token limits."""
    words = text.split()
    chunks = []
    while words:
        chunk = ' '.join(words[:max_tokens])
        chunks.append(chunk)
        words = words[max_tokens:]
    return chunks

def ask_about_speech(file_path, question, output_file):
    """Processes a speech file and sends it to OpenAI, saving the output to a file."""
    # Read the content of the text file
    with open(file_path, 'r') as file:
        speech_text = file.read()

    # Split the text into smaller chunks
    chunks = split_text_into_chunks(speech_text)

    # Process each chunk
    responses = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)} for {file_path}...")

        # Send the request to OpenAI
        completion =  client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is a speech excerpt:\n\n{chunk}\n\n{question}"}
            ]
        )

        # Extract and store the response
        response = completion.choices[0].message.content.strip()
        responses.append(response)

    # Combine all responses into one
    full_response = "\n".join(responses)

    # Write the combined response to the output file
    with open(output_file, 'w') as out_file:
        out_file.write(full_response)

    print(f"The response has been saved to {output_file}.")

def process_folder(input_folder, output_folder, question):
    """Iterates through all text files in the input folder and processes them."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all text files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):  # Process only text files
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, f"analysis_{file_name}")

            print(f"\nProcessing file: {file_name}...")
            # Process the speech and get the model's response
            ask_about_speech(input_file_path, question, output_file_path)

def main():
    # Folder containing the input text files
    input_folder = "text"  # Replace with your input folder path
    # Folder to save the output files
    output_folder = "speech_analyses_mini"  # Replace with your output folder path
    # Question to ask the model
    question = "how does this speech reference America?"

    # Process all text files in the input folder
    process_folder(input_folder, output_folder, question)

if __name__ == "__main__":
    main()
