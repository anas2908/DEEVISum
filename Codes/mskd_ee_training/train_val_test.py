import json
import random

# Set random seed for reproducibility
random.seed(42)

def split_json_file(input_json, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Load the dataset
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Compute split indices
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save the splits
    for split_name, split_data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        output_filename = f"dataset_IU_{split_name}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=4)
        print(f"Saved {len(split_data)} instances to {output_filename}")

# Example usage
split_json_file("dataset_IU.json")
