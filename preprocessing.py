from transformers import AutoProcessor, AutoModelForImageTextToText

# Load the processor and model
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

# Define your input prompt
input_text = "Describe the implications of AI in education."

# Preprocess the text input
inputs = processor(text=input_text, return_tensors="pt")

# Pass the input to the model and generate output
outputs = model.generate(
    **inputs,
    max_length=500,  # Adjust the max length to ensure the response fits
    temperature=0.7,  # Optional: Adjust for creativity
    num_beams=5      # Optional: Use beam search for better results
)

# Decode and print the full response
response = processor.batch_decode(outputs, skip_special_tokens=True)
print("Full response:\n", response[0])

