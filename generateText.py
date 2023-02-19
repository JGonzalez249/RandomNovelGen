# IF NOT INSTALLED:
# pip install -q gpt-2-simple
# pip install tracery
# pip install tensorflow (tensorflow-gpu if you have a GPU)
# pip install nltk

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

def generate_text():
    # create a Tracery instance
    tracery_instance = get_grammar()
    
    # Set the number of chunks to generate (1024 Tokens each)
    num_chunks = 3
    chunk_size = 1024 # 1024 is the maximum number of tokens the model can generate at a time
    total_chunks = num_chunks  
    gen_length = chunk_size * total_chunks # Total number of tokens to generate

    temp = 1 # 0.7 - 1.0 is recommended range
    top_k = 15 # 0 - 40 is recommended range
    top_p = 0.9 # 0.9 is recommended

    text = ""
    prefix = tracery_instance.flatten("#story#")  # initialize prefix
    chunks = []
    for i in range(total_chunks):
        # Include prefix for first chunk
        if i == 0:
            chunk_prefix = prefix
        else:
            # Use the last 3 sentences of the previous chunk as prefix
            last_sentences = sent_tokenize(chunks[-1])[-3:]
            chunk_prefix = ' '.join(last_sentences)

        # Generate text using the model and the prefix from above
        chunk = gpt2.generate(sess, run_name='run1', prefix=chunk_prefix, include_prefix=False, length=chunk_size, temperature=temp, top_k=top_k, top_p=top_p, return_as_list=True)[0]

        if chunk is not None:
            chunks.append(chunk)
            # remove the prefix from the beginning of the chunk
            last_sentences = sent_tokenize(chunk)[-3:]
            prefix = ' '.join(last_sentences)
            print(chunk)
        else:
            print("Error: generated text is None.")
        
        # Print progress
        print(f"\033[1mGenerated chunk {i+1}/{total_chunks}\033[0m")
        
    # Concatenate the generated chunks and print to screen
    text = ''.join(chunks)
    return text

if __name__ == '__main__':
    text = generate_text()
    print(text)
