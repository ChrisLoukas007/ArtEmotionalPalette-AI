import csv

with open('part1/image_paths_color.csv', 'r') as in_file, open('part1/image_paths_Right.csv', 'w', newline='') as out_file:
    reader = csv.reader(in_file)
    writer = csv.writer(out_file)

    for row in reader:
        # Split the row at the comma and take the first part
        new_row = row[0].split(",")[0]
        writer.writerow([new_row])