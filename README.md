# RandomNovelGen
A generative novel using GPT-2 via Google Colab and prompted with Tracery.

# Summary of work
A generated novel that uses GPT-2 and Tracery to produce a science fiction novel with the intent of training a model with data from various books from Project Gutenberg and using it with Tracery to make a prompt for the text generation. The technologies used to make this possible are GPT-2-simple for the model, Google Colab for training and finetuning and Tracery for prompting as well as ChatPGT for ideas and helping generate a Tracery script.

The model uses data from 9 novels mostly focused on science fiction and other to provide a diverse set of data for training and finetuning.

Tracery is used for providing random prompt generation and used as a prefix for GPT-2 to generate random short novels 3000 tokens each time, in chunks of 1024 tokens with the next prefix of the chunk being the last 3 sentences of the previous chunk. For help in creating the Tracery prompts, ChatGPT was used to add in formulating a story for the Tracerry prefix. Github Copilot was also used to help with the text generation code.

# How to use
To use this, you need to Python 3.6 or higher and pip installed. You also need these packages installed:

> Running ```python setup.py ```in the root folder of the project on your terminal will check and install these packages for you.
 * **gpt-2-simple**
 
 ```
 pip install gpt-2-simple
 ```

 * **Tracery**
 
 ```
 pip install tracery
 ```

 * **Tensorflow**
 
 ```
 pip install tensorflow
 ```

 * **ntlk**
 
 ```
pip install ntlk
```

Once all the packages are install, you can run the **generate.py** file to generate a novel. The generated novel will be saved in the **output** folder, it can be run multiple times to produce multiple **genTextXX.txt** files. You can run the **generate.py** file with the following command in the terminal: ```python generate.py```


