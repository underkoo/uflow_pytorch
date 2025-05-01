import os
import glob
import random
import numpy as np

def split_dataset(data_folder, train_ratio=0.8, random_seed=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        data_folder: Path to the folder containing .npy files
        train_ratio: Ratio of training samples (default: 0.8)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_files, val_files: Lists of file paths for training and validation
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Find all .npy files in the data folder
    npy_files = glob.glob(os.path.join(data_folder, '*.npy'))
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_folder}")
    
    # Shuffle the file list
    random.shuffle(npy_files)
    
    # Split into training and validation sets
    split_idx = int(len(npy_files) * train_ratio)
    train_files = npy_files[:split_idx]
    val_files = npy_files[split_idx:]
    
    print(f"Total files: {len(npy_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    return train_files, val_files

def save_file_list(file_list, output_file):
    """
    Save a list of file paths to a text file.
    
    Args:
        file_list: List of file paths
        output_file: Path to the output text file
    """
    with open(output_file, 'w') as f:
        for file_path in file_list:
            # Convert to absolute path
            abs_path = os.path.abspath(file_path)
            f.write(f"{abs_path}\n")
    
    print(f"Saved {len(file_list)} file paths to {output_file}")

def verify_files(file_list):
    """
    Verify that the .npy files have the expected format (11×3000×4000).
    
    Args:
        file_list: List of .npy file paths
    
    Returns:
        valid_files: List of valid file paths
    """
    valid_files = []
    
    for file_path in file_list:
        try:
            # Load just the header information without loading the entire array
            array_shape = np.load(file_path, mmap_mode='r').shape
            
            if len(array_shape) == 3 and array_shape[0] == 11 and array_shape[1] == 3000 and array_shape[2] == 4000:
                valid_files.append(file_path)
            else:
                print(f"Warning: File {file_path} has unexpected shape {array_shape}, skipping")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return valid_files

def main():
    # Configuration
    data_folder = 'output_arrays'
    train_txt = 'train.txt'
    val_txt = 'validation.txt'
    train_ratio = 0.8
    
    # Create train/val split
    train_files, val_files = split_dataset(data_folder, train_ratio)
    
    # Verify files have the correct format
    print("Verifying training files...")
    valid_train_files = verify_files(train_files)
    
    print("Verifying validation files...")
    valid_val_files = verify_files(val_files)
    
    # Save to text files
    save_file_list(valid_train_files, train_txt)
    save_file_list(valid_val_files, val_txt)
    
    print("\nSummary:")
    print(f"Original train files: {len(train_files)}, Valid: {len(valid_train_files)}")
    print(f"Original validation files: {len(val_files)}, Valid: {len(valid_val_files)}")
    print(f"Training file list saved to: {train_txt}")
    print(f"Validation file list saved to: {val_txt}")

if __name__ == "__main__":
    main() 