import subprocess
subprocess.run(["pip", "install", "gpt-2-simple"])
subprocess.run(["pip", "install", "tracery"])
subprocess.run(["pip", "install", "tensorflow"])
subprocess.run(["pip", "install", "nltk"])

import gpt_2_simple as gpt2
from grammar import get_grammar
import random
import re
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# variable for the directory of the pre-trained model
pretrained_model_dir = './checkpoint'

# Load the pre-trained GPT-2 model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, checkpoint_dir=pretrained_model_dir)

def generate_last_chunk():
    # create a Tracery instance
    tracery_instance = get_grammar()

    # Set the number of chunks to generate (1024 Tokens each)
    num_chunks = 3
    chunk_size = 1024  # 1024 is the maximum number of tokens the model can generate at a time
    total_chunks = num_chunks
    #gen_length = chunk_size * total_chunks  # Total number of tokens to generate

    temp = 1  # 0.7 - 1.0 is recommended range
    top_k = 40  # 0 - 40 is recommended range
    top_p = 0.9  # 0.9 is recommended

    text = ""
    prefix = tracery_instance.flatten("#story#")  # initialize prefix
    chunks = []
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

        if chunk is not None:
            # Clean up the chunk using regular expressions
            chunk = re.sub(r'\s+', ' ', chunk)  # Remove extra whitespace
            chunk = re.sub(r'[^\x00-\x7F]+', '', chunk)  # Remove non-ASCII characters
            chunk = re.sub(r'(\n )+', '\n', chunk)  # Remove extra spaces after newlines
            chunk = re.sub(r'\n+', '\n', chunk)  # Remove extra newlines
            chunks.append(chunk.strip())  # Add the cleaned chunk to the list
            print(chunk)
        else:
            print("Error: generated text is None.")

        # Set the new prefix to be the last 3 sentences of the generated chunk
        last_sentences = sent_tokenize(chunk)[-3:]
        prefix = ' '.join(last_sentences)

        # Print progress
        #print(f"\033[1mGenerated chunk {i + 1}/{total_chunks}\033[0m")

    # Concatenate the generated chunks and split into paragraphs
    text = '\n\n'.join(chunks)
    return chunks[-1]

if __name__ == '__main__':
    last_chunk = generate_last_chunk()
    print(last_chunk)