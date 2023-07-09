import csv
import statistics

def calculate_statistics(filename):
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present

        for row in reader:
            try:
                value = float(row[0])
                data.append(value)
            except ValueError:
                print(f"Ignoring non-numeric value: {row[0]}")

    mean = statistics.mean(data)
    stdev = statistics.stdev(data)
    top_10_mean = statistics.mean(sorted(data, reverse=True)[:10])

    return mean, stdev, top_10_mean

# Usage example
filename = 'results.csv'  # Replace with your CSV file name or path

mean, stdev, top_10_mean = calculate_statistics(filename)
print(f"Mean: {mean}")
print(f"Standard Deviation: {stdev}")
print(f"Mean of the top 10 elements: {top_10_mean}")
