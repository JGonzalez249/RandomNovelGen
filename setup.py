import subprocess
subprocess.run(["pip", "install", "gpt-2-simple"])
subprocess.run(["pip", "install", "tracery"])
subprocess.run(["pip", "install", "--user", "-U" "nltk"])

import nltk
nltk.download('punkt')