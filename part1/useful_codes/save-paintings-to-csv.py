# Define the base path
base_path = "dataset-paintings\\all paintings-photos\\"

# Define the start and end numbers
start = 1
end = 1789

# Open the output file
with open("part1\\image_urls.csv", "w") as file:
    # Generate and write the file paths
    for i in range(start, end + 1):
        file_path = base_path + str(i) + ".jpg"
        file.write(file_path + "\n")