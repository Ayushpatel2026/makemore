import os

# Ensure the file path is correct
file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
words = open(file_path, 'r').read().splitlines()
print(words[0:10])