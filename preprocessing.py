
import spacy
import os

def sentence_splitter(text_path):
    nlp = spacy.load("en_core_web_sm")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create a spaCy Doc object
    doc = nlp(text)
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def write_sentences_to_file(sentences, output_path):
    # Open the output file in write mode
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # Enumerate sentences starting at 1
        for i, sentence in enumerate(sentences, start=1):
            # Write "Sentence 1: ..." etc., to the file
            out_file.write(f"Sentence {i}: {sentence}\n")

def process_folder(input_folder, output_folder):
    """
    - Iterates through all .txt files in `input_folder`.
    - Splits each file into sentences.
    - Writes each file’s sentences into a corresponding output file in `output_folder`.
    """
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over .txt files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_folder, file_name)
            
            # Build a corresponding output file path
            # e.g., input file "my_speech.txt" -> "my_speech_sentences.txt"
            base_name, _ = os.path.splitext(file_name)  # "my_speech", ".txt"
            output_file_name = f"{base_name}_sentences.txt"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Split and write
            sentences = sentence_splitter(input_file_path)
            write_sentences_to_file(sentences, output_file_path)
            
            print(f"Processed: {file_name} -> {output_file_name}")

def main():
    input_folder = r"C:\Users\JJ\Desktop\Senior Thesis\America\text"
    output_folder = r"C:\Users\JJ\Desktop\Senior Thesis\America\sentences"

    # Process all .txt files in the input folder
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()