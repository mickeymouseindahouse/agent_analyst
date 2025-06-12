from setuptools import setup, find_packages

setup(
    name="agent_analyst",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "openai>=1.10.0",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Q&A agent for analyzing customer service datasets",
    keywords="nlp, agent, customer service, dataset",
    python_requires=">=3.8",
)
