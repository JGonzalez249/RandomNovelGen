import os
import subprocess
import platform
import gpt_2_simple as gpt2
from grammar import get_grammar
import re
import random
from nltk.tokenize import sent_tokenize # Import the sentence tokenizer

# variable for the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# variable for the directory of the pre-trained model
pretrained_model_dir = './checkpoint'

# Load the pre-trained GPT-2 model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, checkpoint_dir=pretrained_model_dir)

# Function to generate text using the GPT-2 model
def generate_text():
    from tensorflow._api.v2.math import top_k
    # create a Tracery instance
    tracery_instance = get_grammar()

    # Set the number of chunks to generate (1024 Tokens each)
    num_chunks = 8
    chunk_size = 1024  # 1024 is the maximum number of tokens the model can generate at a time
    total_chunks = num_chunks
    
    temp = 1  # 0.7 - 1.0 is recommended range for the temperature
    top_k = 40 # 40 is recommended due to the small size of the model
    top_p = 0.9  # 0.9 is recommended 

    # Set the maximum and minimum number of sentences per paragraph
    min_sentences = 5
    max_sentences = 7

    text = ""
    prefix = tracery_instance.flatten("#story#")  # initialize prefix
    chunks = []
    sentence_count = 0  # Initialize sentence count
    for i in range(total_chunks):
        # Include prefix for first chunk
        if i == 0:
            chunk_prefix = prefix
        else:
            # Use the last 3 sentences of the previous chunk as prefix
            last_chunk = chunks[-1]
            last_sentences = sent_tokenize(last_chunk)[-3:]
            chunk_prefix = ' '.join(last_sentences)

        # Generate text using the model and the prefix from above
        chunk = gpt2.generate(sess, run_name='run1', prefix=chunk_prefix, include_prefix=False, length=chunk_size,
                              temperature=temp, top_k=top_k, top_p=top_p, return_as_list=True)[0]
        # Check if the generated text is not None
        if chunk is not None:
            # Clean up the chunk using regular expressions
            chunk = re.sub(r'\s+', ' ', chunk)  # Remove extra whitespace
            chunk = re.sub(r'[^\x00-\x7F]+', '', chunk)  # Remove non-ASCII characters
            chunk = re.sub(r'(\n )+', '\n', chunk)  # Remove extra spaces after newlines
            chunk = re.sub(r'\n+', '\n', chunk)  # Remove extra newlines
            # Split the chunk into sentences and count the number of sentences
            sentences = sent_tokenize(chunk)
            sentence_count += len(sentences)

 # If the number of sentences exceeds the range, add a new paragraph
            if sentence_count + len(sentences) > random.randint(5, 7):
                chunks.append('\n\n')  # Add a new paragraph
                sentence_count = len(sentences)  # Reset the sentence count
            else:
                sentence_count += len(sentences)

            chunks.append(chunk.strip())  # Add the cleaned chunk to the list
            print(chunk)
        else:
            print("Error: generated text is None.")


        # Set the new prefix to be the last 3 sentences of the generated chunk
        last_sentences = sent_tokenize(chunk)[-3:]
        prefix = ' '.join(last_sentences)

    # Concatenate the generated chunks with an empty line between them
    text = '\n\n'.join(chunks)

    # Remove repetitions in the generated text
    for i in range(num_chunks):
        text = re.sub(r'(\. ){3,}', '. ', text)  # Remove repetitions of 3 or more ". "

    # create a directory for the output files
    output_dir = os.path.join(script_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create a file with the generated text
    filename = os.path.join(output_dir, 'genText{:02d}.txt'.format(len([f for f in os.listdir(output_dir) if f.startswith('genText')]) + 1))
    with open(filename, 'w') as file:
        file.write(text)
    # Open the file using the system's default text editor
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(('open', filename))
    elif platform.system() == 'Windows':  # Windows
        os.startfile(filename)
    else:  # Linux
        subprocess.call(('xdg-open', filename))

    return text

# Run the script to generate text and print it to the terminal
if __name__ == '__main__':
    generated_text = generate_text()
    print(generated_text)