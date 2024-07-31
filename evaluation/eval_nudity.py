import argparse
import numpy as np
import pandas as pd
from collections import Counter

def main(file_path):
    df = pd.read_csv(file_path, keep_default_na=False)
    
    counter = Counter()
    
    # Iterate over the 'NudeNet_label' column
    for labels in df['NudeNet_label']:
        if labels:  # Check if the string is not empty
            classes = set(labels.split('-'))
            counter.update(classes)

    # Print the counts for each class
    total = 0
    for class_name, detected_num in counter.items():
        total += detected_num
        print('- ' * 42)
        print('categories:', class_name)
        print(f"\033[1mNudity Count:\033[0m {detected_num}")
    
    print('- ' * 42)
    print('categories:', 'all')
    print(f"\033[1mNudity Count:\033[0m {total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the number detected nudity classes.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')

    args = parser.parse_args()

    main(file_path=args.csv)