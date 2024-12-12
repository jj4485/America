from transformers import pipeline
import requests

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")

def query_llama(api_url, text, question):
    """
    Send a query to the LLaMA model API with the text and question.
    """
    payload = {
        "context": text,
        "question": question
    }

    # Send a POST request to the LLaMA API
    response = requests.post(api_url, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get("answer", "No answer returned.")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    # File containing the speech
    input_file = "text/1853.inaugural-address-32.txt"  # Replace with your file path
    # Output file to save the response
    output_file = "analysis_output.txt"
    # LLaMA model API endpoint
    llama_api_url = "http://localhost:8000/query"  # Replace with your actual API endpoint

    # Question to ask the LLaMA model
    question = "How does this speech reference America?"

    # Read the text from the input file
    with open(input_file, 'r') as file:
        text = file.read()

    # Query the LLaMA model
    answer = query_llama(llama_api_url, text, question)

    # Save the response to the output file
    if answer:
        with open(output_file, 'w') as out_file:
            out_file.write(f"Question: {question}\n\n")
            out_file.write(f"Answer: {answer}\n")
        print(f"Analysis saved to {output_file}.")
    else:
        print("Failed to retrieve an answer from the LLaMA model.")

if __name__ == "__main__":
    main()