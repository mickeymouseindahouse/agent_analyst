import pandas as pd
from datasets import load_dataset
import os

def download_dataset():
    """
    Download the Bitext customer support dataset from Hugging Face
    and save it as a CSV file.
    """
    print("Downloading dataset from Hugging Face...")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(get_dataset_path()), exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    
    # Convert to pandas DataFrame and save as CSV
    df = pd.DataFrame(dataset['train'])
    df.to_csv(get_dataset_path(), index=False)
    
    print(f"Dataset downloaded and saved to {get_dataset_path()}")
    print(f"Dataset shape: {df.shape}")
    
    return df

def get_dataset_path():
    """
    Get the path to the dataset CSV file.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "Bitext_Sample_Customer_Support_Training_Dataset.csv")

def load_dataset_df():
    """
    Load the dataset from the CSV file.
    If the file doesn't exist, download it first.
    """
    if not os.path.exists(get_dataset_path()):
        return download_dataset()
    
    return pd.read_csv(get_dataset_path())

if __name__ == "__main__":
    # Download the dataset if running this script directly
    df = download_dataset()
    
    # Print dataset info
    print("\nDataset columns:")
    print(df.columns.tolist())
    
    print("\nSample data:")
    print(df.head(3))
    
    # Print some statistics
    print("\nIntent distribution:")
    print(df['intent'].value_counts().head(10))
    
    print("\nCategory distribution:")
    print(df['category'].value_counts().head(10))
