# IF NOT INSTALLED:
#!pip install -q gpt-2-simple
#!pip install tracery
#!pip install tensorflow (tensorflow-gpu if you have a GPU)


import gpt_2_simple as gpt2
import tracery
import random
import re


from tracery.modifiers import base_english
from tensorflow._api.v2.math import top_k

# variable for the directory of the pre-trained model
pretrained_model_dir = './checkpoint'


# Load the pre-trained GPT-2 model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, checkpoint_dir=pretrained_model_dir)

# Tracery grammar for the story
grammar = tracery.Grammar({
    "story": [
    "#hero.capitalize# wakes up in #location# and discovers #discovery#",
    "In a world where #technology# is #adjective#, #hero.capitalize# must find a way to #goal# before it's too late",
    "#hero.capitalize# travels forward in time to #year# and is surprised to find they #surprise#",
    "After a mission to explore a distant #planet#, #hero.capitalize# discovers a #alien# that changes everything",
    "In a future where #disaster# has led to widespread #adjectiveSituation#, #hero.capitalize# must #mission# to #goal# and #survive#",
    "In the year #year#, #hero.capitalize# discovers a new form of #technology# that alters the course of #history# #adverb#",
    "After a freak accident, A #hero.capitalize# gains the power of #superpower# and must use it to #responsibility#",
    "The government has been experimenting with #technology# for years, but #hero.capitalize# discovers the #adjective# truth #conspiracy#",
    "#hero.capitalize# is a #occupation# who discovers a plot to #conspiracy# and must stop it before it's too late",
    "In a world where #humans# are #adjective#, #hero.capitalize# must navigate a dangerous #dilemma# to #survive#",
    "#hero.capitalize# wakes up in a #location# and discovers that they are the last person alive",
    "In a world where #technology# can manipulate reality, #hero.capitalize# must uncover the conspiracy to control #goal#",
    "#hero.capitalize# travels to a distant #planet# in search of #discovery# but ends up in the middle of a #disaster#",
    "After a mission to investigate an abandoned facility, #hero.capitalize# #surprise.s# that could destroy the world",
    "In a future where #technology# has advanced beyond imagination, #hero.capitalize# must #mission# to prevent #disaster#",
    "In the year #year#, #hero.capitalize# is tasked with building a #technology# technology that could change the world #adverb#",
    "After a lab accident, #hero.capitalize# gains the power of #superpower# but quickly realizes it comes at a terrible cost",
    "The government has been covering up the existence of #alien# for years, but #hero.capitalize# is about to uncover the #adjective# truth",
    "#hero.capitalize# is a #occupation# who discovers a hidden society of #humans# with incredible #superpower#",
    "In a world where humans are #humans# and are no longer the dominant species, #hero.capitalize# must adapt or #challenge#",
    "#hero.capitalize# wakes up in a parallel universe where everything they know is #adjective#, and they must find a way home",
    "In a world where #technology# can rewrite memories, #hero.capitalize# must use their skills as a #occupation# to uncover the #conspiracy#",
    "#hero.capitalize# travels to the distant future and finds a world that is both #adjective# and #adjective#",
    "After a mission to retrieve a powerful artifact, #hero.capitalize# is betrayed by their team and must find a way to #survive#",
],
    "hero": ["Aaliyah", "Aiden", "Akira", "Alara", "Alexa", "Alexios", "Alyssa", "Amara", "Anakin", "Andromeda", "Apollo", "Aria", "Aric", "Arya", "Atlas", "Aurora", "Avril", "Ayana", "Azura", "Ben", "Calypso", "Cameron", "Cassius", "Celeste", "Ceres", "Chandra", "Chara", "Chris", "Cleo", "Cygnus", "Dahlia", "Darian", "Dash", "Demeter", "Dione", "Dominic", "Drake", "Eris", "Esmeralda", "Ethan", "Evangeline", "Gaia", "Gideon", "Halcyon", "Harper", "Hestia", "Icarus", "Io", "Jada", "Jaxon", "Kaida", "Kalen", "Kalliope", "Kamari", "Kato", "Kenji", "Kian", "Korbin", "Kyra", "Landon", "Lavinia", "Leo", "Lysandra", "Lystra", "Mace", "Maia", "Mako", "Marcella", "Mars", "Micah", "Mira", "Nadia", "Neo", "Nova", "Orion", "Pandora", "Phoenix", "Qadira", "Rhea", "Rigel", "Saffron", "Sage", "Samara", "Selena", "Serena", "Sirius", "Sol", "Stella", "Tahlia", "Talos", "Tara", "Terra", "Thalia", "Thetis", "Titania", "Triton", "Ulysses", "Vega", "Venus", "Xander", "Xanthe", "Yara", "Zara", "Zephyr", "Zora"],
    "location": ["on a space station orbiting a black hole", "in a city on a planet with a toxic atmosphere", "in an underground research facility on a distant moon", "on a terraformed asteroid colony", "in a cyberpunk megacity", "on a generation ship traveling through deep space", "in a virtual reality simulation", "on a planet overrun by alien vegetation", "in a post-apocalyptic wasteland", "in a time-traveling space vessel", "in a floating city above the clouds", "in a post-apocalyptic wasteland", "on a space station orbiting a black hole", "in a virtual reality simulation", "on a terraformed moon of #planet#"],
    "discovery": ["a new species of alien life", "a hidden government conspiracy", "a rogue artificial intelligence", "the secret to time travel", "a cure for a deadly disease"],
    "technology": ["robotics", "holography", "teleportation", "time travel", "antigravity", "cloning", "stem cells", "nanobots", "superconductors", "lasers", "space travel", "terraforming", "warp drive", "wormholes", "dyson spheres", "mind uploading", "brain-computer interface", "simulated reality", "digital immortality", "mind control"],
    "adjectiveSituation": ["out of control", "the key to salvation", "banned by the government", "beyond human comprehension", "on the brink of collapse", "under investigation", "shrouded in mystery", "plagued by corruption", "in need of reform", "threatened by a new enemy", "ripe for revolution", "haunted by a dark past", "fueled by greed and ambition", "inspired by a noble cause", "ravaged by a deadly virus", "dominated by a powerful faction", "torn apart by civil war", "protected by a secret order", "infested with zombies and mutants", "infiltrated by spies and traitors","dependent on a scarce resource","governed by a strict code of honor","exposed to a cosmic anomaly","at war with an alien race","facing an imminent disaster"],
    "challenge": ["outwit the government", "uncover a hidden truth", "rescue a loved one", "save the planet from destruction", "solve a mystery"],
    "goal": ["save the galaxy from destruction", "discover a way to travel faster than light", "defeat an alien invasion force", "prevent a catastrophic event from occurring", "create a new era of peace and prosperity", "make groundbreaking scientific discoveries", "establish diplomatic relations with an alien species", "find a cure for a deadly space virus", "uncover the truth behind a mysterious phenomenon", "build a new society on #planet#"],    
    "year": ["2077", "2250", "2376", "2400", "2439", "2467", "2499", "2522", "2568", "2671", "2775", "2899", "2981", "3057", "3120", "3228", "3386", "3537", "3701", "3849", "3986", "4092", "4209", "4345", "4500"],
    "surprise": ["are stranded on an alien planet", "discover a long-lost artifact", "uncover a conspiracy", "meet a sentient alien race", "are transported to another dimension", "travel back in time", "found a hidden portal to another world", "unearth a mysterious object", "are confronted by a ghost from their past", "encounter a hostile AI", "discover a secret society", "are abducted by aliens", "awaken a dormant force", "solve a centuries-old mystery", "face their darkest fear"],
    "planet": ["Alderaan", "Arrakis", "Caprica", "Coruscant", "Dagobah", "Earth", "Ego", "Gallifrey", "Hoth", "Krypton", "Mars", "Naboo", "Namek", "Pandora", "Sakaar", "Tatooine", "Titan", "Vega", "Vulcan", "Yavin 4", "Cygnus", "Jupiter", "Mercury", "Neptune", "Saturn", "Uranus", "Venus", "Pern", "Kobol"],
    "alien": ["an alien artifact that holds untold secrets", "an alien symbiote that grants superhuman abilities", "a group of stranded aliens seeking refuge on Earth", "an alien species with a unique form of communication", "a powerful alien entity that threatens the galaxy", "a team of alien scientists studying #planet#'s ecosystems", "an alien invasion force disguised as humans", "an alien race with a deep connection to the cosmos", "an alien civilization that has mastered time travel", "a rogue alien warrior on a mission of revenge"],
    "disaster": ["a global pandemic", "nuclear war", "climate change", "an alien invasion", "a catastrophic asteroid impact"],
    "mission": ["find a new home for humanity", "restore #planet# to its former glory", "uncover the secrets of the universe", "stop a dangerous experiment", "make first contact with alien life", "build a colony on a distant planet", "protect #planet# from a cosmic threat", "explore a newly discovered star system", "retrieve a lost artifact from a distant world", "survive a crash landing on an alien planet", "investigate a strange signal from deep space"],    
    "survive": ["survive on a hostile planet", "survive a deadly virus outbreak", "survive a robot uprising", "survive a natural disaster", "survive a post-apocalyptic world"],
    "adverb": ["hyperspatially", "telepathically", "quantumly", "neurologically", "cybernetically", "genetically", "interdimensionally", "synthetically", "teleportationally",    "intergalactically",    "gravitationally",    "nanotechnologically",    "biomechanically", "transdimensionally", "transcendentally", "antigravitationally", "cosmically", "chronologically", "nebulously", "dimensionally", "post-apocalyptically", "cyberpunks", "transhumanly", "paranormally", "bionically"],
    "adjective": ["cybernetic", "transdimensional", "hyperspace", "intergalactic", "neural", "post-apocalyptic", "augmented", "artificial", "nano", "quantum", "cyborg", "extraterrestrial", "cyberpunk", "cyber", "genetically engineered", "mutant", "transhuman", "intelligent", "self-aware", "sentient", "virtual", "android", "bionic", "synthetic", "cosmic", "spatial", "futuristic", "interstellar", "interplanetary", "interdimensional", "time-traveling", "hybrid", "holo", "mind-bending", "immersive", "disembodied", "apocalyptic", "dystopian", "cloned", "multidimensional", "supernatural", "technological", "cyberspace", "alien", "otherworldly", "galactic", "gravitational", "robotic", "mechanical"],
    "occupation": ["bounty hunter", "space pirate", "rebel leader", "cyber detective", "time traveler", "alien diplomat", "quantum mechanic", "intergalactic trader", "cyborg engineer", "artificial intelligence specialist"],
    "superpower": ["gravity manipulation", "energy projection", "dimensional travel", "mind control", "teleportation", "phasing through matter", "time dilation", "intangibility", "electrokinesis", "matter manipulation"],
    "conspiracy": ["mind control experiments", "alien cover-ups", "secret societies", "time travel experiments", "government surveillance"],
    "humans": ["dying out", "losing their emotions", "being replaced by robots", "being forced to live in space", "losing their memories"],
    "dilemma": ["decide between personal gain and the greater good", "help someone they love or save the world", "choose between two conflicting responsibilities", "stand up for what's right or protect themselves"],
    "responsibility": ["take down a corrupt government", "stop a mad scientist from destroying the world", "protect a valuable artifact from falling into the wrong hands", "save the life of someone they hate"],
}, )
grammar.add_modifiers(base_english)

# Set the number of chucks to generate (1024 Tokens each)
num_chunks = 3
chunk_size = 1024 # 1024 is the maximum number of tokens the model can generate at a time
chunks = num_chunks  
gen_length = chunk_size * chunks # Total number of tokens to generate

temp = 1 # 0.7 - 1.0 is recommended range
top_k = 15 # 0 - 40 is recommended range
top_p = 0.9 # 0.9 is recommended

text = ""
prefix = grammar.flatten("#story#")  # initialize prefix outside of loop
for i in range(chunks):
    
    if i == 0:
        # Generate text using the model and the prefix from above
        chunk = gpt2.generate(sess, run_name='run1', prefix=prefix, include_prefix=False, length=chunk_size, temperature=temp, top_k=top_k, top_p=top_p, return_as_list=True)[0]
    else:
        # Generate text using the model and the last sentence of the previous chunk as prefix
        sentences = re.split("(?<=[.!?]) +", text)
        prefix = sentences[-1] if len(sentences) > 1 else prefix
        chunk = gpt2.generate(sess, run_name='run1', prefix=prefix, include_prefix=False, length=chunk_size, temperature=temp, top_k=top_k, top_p=top_p, return_as_list=True)[0]
    
    if chunk is not None:
        text += chunk
        # remove the prefix from the beginning of the chunk
        sentences = re.split("(?<=[.!?]) +", chunk)
        prefix = sentences[-1] if len(sentences) > 1 else chunk
        print(chunk)
    else:
        print("Error: generated text is None.")
    
    # Print progress
    print("\033[1mGenerated chunk {}/{}\033[0m".format(i+1, chunks))
    
# Print the generated text to the screen
print(text)