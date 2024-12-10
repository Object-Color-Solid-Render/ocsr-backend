import csv

# Function to process lambda_max values
def process_lambda_values(row):
    # Convert all lambda_max fields to integers, replacing "N/A" with 0
    return [
        int(row[f'lambda_max{i}']) if row[f'lambda_max{i}'] != "N/A" else 0
        for i in range(1, 5)
    ]

# Function to read and process the CSV
def read_csv(file_path):
    data = []  # List to store processed rows
    with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=',')  # Tab-delimited CSV
        for row in reader:
            # Extract data in the specified format
            common_name = row['Common Name']            # Common Name (string)
            scientific_name = row['Scientific Name']    # Scientific Name (string)
            phylum = row['Phylum']                      # Phylum (string)
            class_name = row['Class']                   # Class (string)
            order = row['Order']                        # Order (string)
            template = row['Recommended Template']      # Template (string)
            chromophores = row['Chromophores']          # Chromophores (string)
            peaks = process_lambda_values(row)          # List of lambda_max values (float[4])
            source = row['Source']                      # Source (string)
            note = row['Note']                          # Note (string)

            # Append a tuple with the processed data to the list
            data.append({
                "common_name": common_name,
                "scientific_name": scientific_name,
                "phylum": phylum,
                "class": class_name,
                "order": order,
                "template": template,
                "chromophores": chromophores,
                "lambda_max_values": peaks,
                "source": source,
                "note": note,
            })

    # sort data by common_name
    data.sort(key=lambda x: x["common_name"])
    return data

# Example usage
#file_path = "res/Spectral Sensitivity Database.csv"
#rows = read_csv(file_path)

# Print the processed data
#for row in rows:
#    print(row)
