# IF NOT INSTALLED:
#!pip install -q gpt-2-simple
#!pip install tracery
#!pip install tensorflow

import os
import gpt_2_simple as gpt2
import tracery
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
    "#hero.capitalize# travels forward in time to #year# and is surprised to find #surprise#",
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
    "After a mission to investigate an abandoned facility, #hero.capitalize# #surprise# that could destroy the world",
    "In a future where #technology# has advanced beyond imagination, #hero.capitalize# must #mission# to prevent a catastrophic #disaster#",
    "In the year #year#, #hero.capitalize# is tasked with building a massive #technology# that could change the world #adverb#",
    "After a lab accident, #hero.capitalize# gains the power of #superpower# but quickly realizes it comes at a terrible cost",
    "The government has been covering up the existence of #alien# for years, but #hero.capitalize# is about to uncover the #adjective# truth",
    "#hero.capitalize# is a #occupation# who discovers a hidden society of #humans# with incredible #superpower#",
    "In a world where humans are #humans# and are no longer the dominant species, #hero.capitalize# must adapt or #challenge#",
    "#hero.capitalize# wakes up in a parallel universe where everything they know is #adjective#, and they must find a way home",
    "In a world where #technology# can rewrite memories, #hero.capitalize# must use their skills as a #occupation# to uncover the #conspiracy#",
    "#hero.capitalize# travels to the distant future and finds a world that is both #adjective# and #adjective#",
    "After a mission to retrieve a powerful artifact, #hero.capitalize# is betrayed by their team and must find a way to #survive#",
],
    "hero": ["scientist", "astronaut", "robot", "alien", "time traveler", "cyborg", "mutant", "cloning experiment"],
    "location": ["on a space station orbiting a black hole", "in a city on a planet with a toxic atmosphere", "in an underground research facility on a distant moon", "on a terraformed asteroid colony", "in a cyberpunk megacity", "on a generation ship traveling through deep space", "in a virtual reality simulation", "on a planet overrun by alien vegetation", "in a post-apocalyptic wasteland", "in a time-traveling space vessel", "in a floating city above the clouds", "in a post-apocalyptic wasteland", "on a space station orbiting a black hole", "in a virtual reality simulation", "on a terraformed moon of #planet#"],
    "discovery": ["a new species of alien life", "a hidden government conspiracy", "a rogue artificial intelligence", "the secret to time travel", "a cure for a deadly disease"],
    "technology": ["nanotechnology", "virtual reality", "artificial intelligence", "genetic engineering", "quantum computing"],
    "adjectiveSituation": ["out of control", "the key to salvation", "banned by the government", "beyond human comprehension", "on the brink of collapse"],
    "challenge": ["outwit the government", "uncover a hidden truth", "rescue a loved one", "save the planet from destruction", "solve a mystery"],
    "goal": ["save the galaxy from destruction", "discover a way to travel faster than light", "defeat an alien invasion force", "prevent a catastrophic event from occurring", "create a new era of peace and prosperity", "make groundbreaking scientific discoveries", "establish diplomatic relations with an alien species", "find a cure for a deadly space virus", "uncover the truth behind a mysterious phenomenon", "build a new society on #planet#"],    
    "year": ["2077", "2250", "2376", "2400", "2439", "2467", "2499", "2522", "2568", "2671", "2775", "2899", "2981", "3057", "3120", "3228", "3386", "3537", "3701", "3849", "3986", "4092", "4209", "4345", "4500"],
    "surprise": ["meets their own ancestors", "changes the course of history", "discovers a new civilization", "comes face-to-face with their worst enemy"],
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
prefix = grammar.flatten("#story# ")

# Generate text using the model and the prefix from above
gen_text = gpt2.generate(sess,
              run_name='run1',
              length=50000,
              temperature=1,
              prefix=prefix,
              top_k=20,
              top_p=0.9,
              nsamples=1,
              batch_size=1
              )

# Print the generated text to the screen
print(gen_text)