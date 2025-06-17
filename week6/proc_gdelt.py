import pandas as pd

# Read the input file
input_file = "./gdelt/LOOKUP-GKGTHEMES.TXT"  # Replace with your actual filename
output_file = "./gdelt/output.csv"

# Read the file and split it into columns
data = []
with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")  # Splitting by tab
        if len(parts) == 2:
            data.append(parts)

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Category", "Value"])

# Save to CSV
df.to_csv(output_file, index=False)

print(f"CSV file saved as {output_file}")
