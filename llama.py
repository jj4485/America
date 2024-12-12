from transformers import pipeline

# Initialize the LLaMA pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")

def query_llama(pipe, text, question):
    """
    Query the LLaMA model using the pipeline and return the response.
    """
    # Combine the context and question for generation
    input_prompt = f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"
    
    # Generate the answer using the model
    response = pipe(input_prompt, max_length=500, num_return_sequences=1)
    return response[0]["generated_text"]

def main():
    # File containing the speech
    input_file = "text/1853.inaugural-address-32.txt"  # Replace with your file path
    # Output file to save the response
    output_file = "analysis_output.txt"

    # Question to ask the LLaMA model
    question = "How does this speech reference America?"

    # Read the text from the input file
    with open(input_file, 'r') as file:
        text = file.read()

    # Query the LLaMA model
    answer = query_llama(pipe, text, question)

    # Save the response to the output file
    with open(output_file, 'w') as out_file:
        out_file.write(f"Question: {question}\n\n")
        out_file.write(f"Answer: {answer}\n")
    print(f"Analysis saved to {output_file}.")

if __name__ == "__main__":
    main()
