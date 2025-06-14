import pandas as pd
import os
from datasets import load_dataset

def load_dataset_df():
    """
    Load the dataset from a local CSV file or from Hugging Face if the CSV doesn't exist.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    csv_path = os.path.join(os.path.dirname(__file__), "customer_service_data.csv")
    
    # Check if the CSV file exists
    if os.path.exists(csv_path):
        print(f"Loading dataset from local CSV file: {csv_path}")
        # Load the CSV file
        df = pd.read_csv(csv_path)
    else:
        print("Local CSV file not found. Loading dataset from Hugging Face...")
        try:
            # Fall back to loading from Hugging Face
            df = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").to_pandas()
            
            # Save to CSV for future use
            print(f"Saving dataset to CSV file for future use: {csv_path}")
            df.to_csv(csv_path, index=False)
        except Exception as e:
            raise Exception(f"Failed to load dataset from Hugging Face: {str(e)}")
    
    # Ensure required columns exist
    required_columns = ["category", "intent", "instruction", "response"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    
    return df
