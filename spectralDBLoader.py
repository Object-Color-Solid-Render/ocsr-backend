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
            peaks = process_lambda_values(row)          # List of lambda_max values (int[4])
            source = row['Source']                      # Source (string)    
            note = row['Note']                          # Note (string)  

            # Append a tuple with the processed data to the list
            data.append((common_name, scientific_name, peaks, source, note))
    return data

# Example usage
#file_path = "res/Spectral Sensitivity Database.csv"
#rows = read_csv(file_path)

# Print the processed data
#for row in rows:
#    print(row)
