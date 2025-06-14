from setuptools import setup, find_packages

setup(
    name="agent_analyst",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.24.0",
        "pandas>=1.5.0",
        "openai>=1.0.0",
        "datasets>=2.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)
