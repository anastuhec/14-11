import os


os.chdir('/Users/ana/Desktop/ta2nise5')


#base_dir = "/scratch/stuhecana/ta2nise5"    # <-- your remote directory path

# Create a dated folder inside your remote directory
#folder_name = "results"
#full_path = os.path.join(base_dir, folder_name)

#os.makedirs(full_path, exist_ok=True)

# Now save your file inside this dated folder
#filepath = os.path.join(full_path, "output.txt")

collect = []

for i in range(5):
    collect.append(i)

with open('poskus.txt', "w") as f:
    for i in range(len(collect)):
        f.write(str(collect[i]))
        f.write('\n')