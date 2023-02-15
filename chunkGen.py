# This script takes a text file as input and creates a new text file with chunks of 2000 characters each
# The chunks are separated by a line break
# The script assumes that the input file is in a folder named "books" and the output file is in a folder named "chunks"
# The output file name is the same as the input file name
# The input file is encoded in utf-8
# The output file is encoded in utf-8

# Import the os module to access the file path
import os

# Define the input file path as the books folder plus the file name, this needs to be changed everytime if using a different file

input_file_path = os.path.join("books", "twoCities.txt")

# Get the input file name from the input file path
input_file_name = os.path.basename(input_file_path)

# Define the output file path as the chunks folder plus the input file name
output_file_path = os.path.join("chunks", input_file_name)

# Open the input file in read mode with utf-8 encoding
input_file = open(input_file_path, "r", encoding="utf-8")

# Read the content of the input file as a string
text = input_file.read()

# Close the input file
input_file.close()

# Initialize an empty list to store the chunks
chunks = []

# Initialize a variable to keep track of the current position in the text
position = 0

# Loop through the text until the end is reached
while position < len(text):
    # Find the last whitespace character before the 2000th character from the current position
    # If there is no whitespace character, use the 2000th character as the end of the chunk
    end = text.rfind(" ", position, position + 2000)
    if end == -1:
        end = position + 2000
    
    # Extract the chunk from the text and append it to the list
    chunk = text[position:end]
    chunks.append(chunk)

    # Update the current position to the next character after the end of the chunk
    position = end + 1

# Open the output file in write mode with utf-8 encoding
output_file = open(output_file_path, "w", encoding="utf-8")

# Write the chunks to the output file, separated by a line break
for chunk in chunks:
    output_file.write(chunk + "\n")

# Close the output file
output_file.close()