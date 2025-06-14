from setuptools import setup, find_packages

setup(
    name="agent_analyst",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "matplotlib",
        "seaborn",
        "openai",
        "datasets",
        "pydantic",
        "python-dotenv",
    ],
)
