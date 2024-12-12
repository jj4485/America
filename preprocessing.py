# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-90B-Vision-Instruct")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-90B-Vision-Instruct")