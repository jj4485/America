from transformers import pipeline

# Initialize the LLaMA pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")

def chunk_text(text, max_tokens):
    """
    Splits the text into chunks of approximately `max_tokens` tokens.
    """
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def query_llama(pipe, text_chunks, question, max_new_tokens=200):
    """
    Query the LLaMA model using the pipeline for each chunk and return combined responses.
    """
    responses = []
    for i, chunk in enumerate(text_chunks):
        input_prompt = f"Context: {chunk}\n\nQuestion: {question}\n\nAnswer:"
        print(f"Processing chunk {i + 1}/{len(text_chunks)}...")
        response = pipe(input_prompt, max_length=max_new_tokens + len(chunk.split()), num_return_sequences=1)
        responses.append(response[0]["generated_text"])
    return ' '.join(responses)

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

    # Chunk the text to avoid exceeding model limits
    max_chunk_size = 400  # Adjust based on your model's token limit
    text_chunks = list(chunk_text(text, max_chunk_size))

    # Query the LLaMA model
    answer = query_llama(pipe, text_chunks, question)

    # Save the response to the output file
    with open(output_file, 'w') as out_file:
        out_file.write(f"Question: {question}\n\n")
        out_file.write(f"Answer: {answer}\n")
    print(f"Analysis saved to {output_file}.")

if __name__ == "__main__":
    main()
