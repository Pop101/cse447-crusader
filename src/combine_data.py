import os
import random

multilingual_data_folder = "/Users/kastenwelsh/Documents/cse447-project/Monolingual_data"

# step through each language file in the folder and load 200 random lines from each file
# append the data to a list
data = []

for file in os.listdir(multilingual_data_folder):
    with open(os.path.join(multilingual_data_folder, file), 'r') as f:
        lines = f.readlines()
        random_lines = random.sample(lines, min(len(lines), 200))
        data.extend(random_lines)

# also sample 1000 lines from /Users/kastenwelsh/Documents/cse447-project/sentences.txt
with open("/Users/kastenwelsh/Documents/cse447-project/sentences.txt", 'r') as f:
    lines = f.readlines()
    random_lines = random.sample(lines, 1000)
    data.extend(random_lines)

# Save the data to a new file
with open("combined_data_train.txt", "w") as f:
    f.writelines(data)
