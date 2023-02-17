# RandomNovelGen
A generative novel using GPT-2 via Google Colab and prompted with Tracery.

# Summary of work
A generated novel that uses GPT-2 and Tracery via Google Colab to produce a science fiction novel with the intent of training a model with data from various books from Project Gutenberg and using it with Tracery to make a prompt for the text generation. The technologies used to make this possible are GPT-2-simple for the model, Google Colab for training and finetuning and Tracery for prompting as well as ChatPGT for ideas and helping generate Tracery JSON.

The model uses data from 9 novels mostly focused on science fiction and other to provide a diverse set of data for training and finetuning.

Tracery is used for providing random prompt generation and used as a prefix for GPT-2 to generate random short novels 3000 tokens each time. For help in creating the Tracery prompts, ChatGPT was used to add in formulating a story for the Tracerry prefix.