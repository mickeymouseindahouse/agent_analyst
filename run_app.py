#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import platform

def check_dependencies():
    """Check if required Python packages are installed and install if missing."""
    required_packages = [
        "streamlit",
        "pandas",
        "openai",
        "datasets",
        "matplotlib",
        "seaborn",
        "pydantic",
        "python-dotenv"
    ]
    
    print("Checking dependencies...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} has been installed")

def setup_environment():
    """Set up the environment variables and configuration."""
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Add your environment variables here\n")
            f.write("# NEBIUS_API_KEY=your_api_key_here\n")
        print("Please edit the .env file to add your API keys")
    
    # Check if NEBIUS_API_KEY is already set in environment
    api_key = os.environ.get("NEBIUS_API_KEY")
    if api_key:
        print("Using NEBIUS_API_KEY from environment variables")
    else:
        # Only try to read from file if environment variable is not set
        try:
            with open("nebius.key", "r") as key_file:
                nebius_api_key = key_file.read().strip()
            os.environ["NEBIUS_API_KEY"] = nebius_api_key
            print("Successfully loaded Nebius API key from nebius.key")
        except FileNotFoundError:
            print("Warning: NEBIUS_API_KEY not found in environment variables and nebius.key file not found.")
            print("Please set the NEBIUS_API_KEY environment variable or create a nebius.key file")

def download_dataset():
    """Download the dataset if not already present."""
    print("Checking dataset...")
    try:
        from datasets import load_dataset
        # Just load a small sample to verify it works
        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train[:10]")
        print("✓ Dataset is accessible")
    except Exception as e:
        print(f"Error accessing dataset: {e}")
        print("Please check your internet connection and Hugging Face access")

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Run the Customer Service Dataset Q&A application")
    parser.add_argument("--install-only", action="store_true", help="Only install dependencies without running the app")
    
    # Get port from environment variable or use default/command line argument
    default_port = int(os.environ.get("PORT", 8501))
    parser.add_argument("--port", type=int, default=default_port, 
                        help=f"Port to run the Streamlit app on (default: {default_port}, can be set with PORT env var)")
    
    args = parser.parse_args()
    
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Check and install dependencies
    check_dependencies()
    
    # Set up environment
    setup_environment()
    
    # Download dataset
    download_dataset()
    
    if not args.install_only:
        print(f"Starting Streamlit app on port {args.port}...")
        # Run the Streamlit app
        subprocess.run(["streamlit", "run", "app/app.py", "--server.port", str(args.port)])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)
