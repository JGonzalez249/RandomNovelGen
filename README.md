# RandomNovelGen
A generative novel using GPT-2 and prompted with Tracery.

# Summary of work
A generated short novel that uses GPT-2 and Tracery to produce a, hopefully, science fiction novel with the intent of training a model with data from various books from Project Gutenberg and using it with Tracery to make a prompt for the text generation. The chances of having a science fiction novel will more than likely be low due to the narure and formatting of gpt-2-simple and finetuning using a 124M model. 

The technologies used to make this possible are GPT-2, Google Colab for training and finetuning and Tracery for prompting as well as ChatPGT for ideas and helping generate a Tracery and Python script. The model uses data from 9 novels mostly focused on science fiction and others to provide a diverse set of data for training and finetuning.

Tracery is used for providing random prompt generation and used as a prefix for GPT-2 to generate random short novels each time, in chunks of 1024 tokens with the next prefix of the chunk being the last 3 sentences of the previous chunk. For help in creating the Tracery prompts, ChatGPT was used to formulate a story for the Tracerry prefix. Github Copilot was also used to help with the text generation code.

I may eventually package this into an executable file so that it can be used by anyone without having to install Python and the packages needed to run the script.

# How to use
You need to download/clone this project from Github, [download and extract the checkpoint folder](https://drive.google.com/file/d/1-P_wwxipOteS4ePXZyN5YqJqn3cs8tX1/view?usp=sharing) into the repo, cd into the repo.

To use this, you need to Python 3.8 or higher and pip installed. You also need these packages installed:

> Running  ```python setup.py ```in the root folder of the project on your terminal will check and install these packages for you, make sure to also download the pre-trained model.


 **gpt-2-simple**
 
 
      pip install gpt-2-simple


 **tensorflow**
 
      pip install tensorflow
 

 **Tracery**
 
 
      pip install tracery
 

 **ntlk**
 

      pip install --user -U ntlk



Once all the packages are installed, you can run the **generateText.py** file to generate a short novel once you download the pre-trained model. To download the pre-trained folder if you have not done so yet, click [here](https://drive.google.com/file/d/1-P_wwxipOteS4ePXZyN5YqJqn3cs8tX1/view?usp=sharing), extract the **checkpoint folder** into the root of the project directory. You can then delete the tar file to save space. You can then run generateText.py script to generate a novel using the downloaded model. The generated novel will be saved in the **output** directory that's created by the script and will open on your default text editor. It can be run multiple times to produce multiple **genTextXX.txt** files. You can run the **generateText.py** file with the following command in the terminal: 

    python generateText.py

**Warning:** The script may take a while to generate text, depending on your computer specs.

# Samples
Some samples of the generated text are below, they're cut and edited to fit in the README:


**Sample 1**

Calypso is tasked with building in a future where climate change and must prevent a catastrophic event from occurring before it's too late. A cold, indeed oppressive wind, just to rub people on the back. I have no doubt that this is what is behind your ears, remarked Holmes as we reached the door. Warm, indeed, London, certainly, said the man, as we entered. Lord Backwater and Mr. Holder were twins, and no idle conflux of heat will do either of them any good. My dear fellow, I see that you draw your conclusions from the facts. As they say,vehement"! Theres more than one reason for that, said Holmes. I think that I have four more that will not be long in recovering. For one thing, it is evident that the elevation of temperature decreases with increasing thickness of hair. It is quite possible, even probable, that hairs one above the other get more firmly in the centre as the temperature goes up, so that there is a definite increase in the security of the hair. It is characteristic of early human growths of late Augusts that there is something of the quaintly poetical accompaniment of feverish feverish laughter.

***
**Sample 2**

The future of a future with future measurement P equals Eternall: this proves that P equals Energetic mass; that is to say, mass equivalent to one atom of mass. That is all of them said to have information contained in them outside of the ordinary Time Chains, and in themselves neither an Information Archive, nor a Memory, nor a Reasoning Archive. The story of the bronze goblets that destroyed Zambo, the story of the bronze goblets that perfect the movements of the men of Wansbaradze and Metallurg, and the glorious Battle of Lepanto are just such are the tales of the bronze goblets which destroyed Cuvier, Challenger, and Challengerese; or the story of the bronze goblets which stimulated the development and evolution of Darwin, John Maynard James, and the perfect storm of natural selection which has endowed the world with the qualities of intellectual property, which are then of vital moment. It is for this reason that I am in this opinion somewhere or other, that this catalogue is no Lambertian. It is a literary manuscript; one may read the full words aloud upon demand, or choose the alternate openings open at the moment of writing. There are, of course, passages which show the genesis of one of the sons of Man, for instance, and there are further traces which show that the great art of man is in some way dependent on the inheritance of his brother--namely, the evolution of powers in the hands of the powerful.

***
**Sample 3**

Im the Medical Man of this City, I cried, but the key? Is Maldonado written all over it, said he. My attention was soon drawn from the inquiry which was just made, and from the desire to know whether my companion was a Fellow of University teaching, and whether, at the Institution of Unison, he had made a solemn and secret pact with the notorious Committee of Lecturers, which he signified by a word, a Section, or a Section Two, or a Section Three, or any Section of the Body, in as many Sections as the most scrupulous men may solemnly signifie. It was difficult, for he had alledged this letter, to write, neither because he was not conducting his thoughts with great ardour nor for easy berating. He was just on the point of finding out what all the fuss was about, in detailing some simple cogitations in the travellers magazine, and making himself conspicuous in the insignia of the cabs, and of the lamps, and of the so-called jack-in-the-box and of the limousine. The immense and so almost invisible minority of the people interested in these events, or who had seen or heard of them, they were few in number, and had no written acquaintance with the matter.

# Artist Statement

Creating this project was meant to challenge myself in using a language I'm not familiar with and using AI tools helped me in the creation of this project. I've always been interested in AI and ML and I wanted to see if I could use AI to help me write a short novel, even with my limited knowledge. While I was able to create a short novel, I feel that I could have done more with the project and I'm might be doing more with it in the future. Maybe even use GPT-3 to create a longer novel with more data and more prompts. 

When I first started this, I was inspired by the various works from others using AI to generate long novels. When first researching into text generation, the project that inspired me the most was "Shapeshifting" by Dave LeCompte. I found his work very well structured and interesting in his goal to make a tabletop "game" using GPT-2. I also found his work to be very well documented and I used his work as a reference for my own project. While my intent differs from his, I learned a lot from his work in how Python works and how to use GPT-2.

With that inspiration, I decided to make this and learn about how AI and ML works. It took my a while to get this right, a lot of trial and error and training the model multiple times as well as issues with Tracery prompts not being used in the gpt-2 prefix. However, the biggest challenge was porting the Google Colab notebook I made for this project into scripts that can be used on a local machine. In order to facilitate this, I had used ChatGPT and GitHub Copilot, to help me script this into Python. In the end, I was able to get this working and I'm happy with the results. I'm also happy that I was able to learn more about AI and ML and how to use them in my own projects. 
