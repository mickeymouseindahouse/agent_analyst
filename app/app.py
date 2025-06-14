import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.tools import get_tools
from agent.agent import ReActAgent
from data.download_dataset import load_dataset_df
import sys
import os

# Add the parent directory to the path to import agent_analyst_task
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent_analyst_task import handle_question

st.set_page_config(page_title="Customer Service Dataset Q&A", layout="wide")

st.title("Customer Service Dataset Q&A")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    tools = get_tools()
    st.session_state.agent = ReActAgent(tools=tools)

# Sidebar
st.sidebar.title("Settings")

# Toggle for planning mode
planning_mode = st.sidebar.radio(
    "Planning Mode",
    ["Pre-planning", "ReActive"],
    index=1
)

# Memory features
st.sidebar.title("Memory")
if st.sidebar.button("Summarize All Interactions"):
    with st.spinner("Generating summary of all interactions..."):
        summary = st.session_state.agent.summarize_interactions()
        st.sidebar.write(summary)

# Dataset info
st.sidebar.title("Dataset Info")
df = load_dataset_df()
st.sidebar.write(f"Total conversations: {len(df)}")
st.sidebar.write(f"Unique intents: {df['intent'].nunique()}")
st.sidebar.write(f"Unique categories: {df['category'].nunique()}")

# Example questions
st.sidebar.title("Example Questions")
example_questions = [
    "What are the most frequent categories?",
    "Show me examples of REFUND category",
    "How many refund requests did we get?",
    "Summarize how agents respond to payment_issue intent",
    "What's the distribution of intents in the ACCOUNT category?"
]
for question in example_questions:
    if st.sidebar.button(question):
        # Clear previous messages if this is a new conversation
        if len(st.session_state.messages) > 0 and not question.startswith("Tell me more"):
            st.session_state.messages = []
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get response based on planning mode
        with st.spinner("Thinking..."):
            if planning_mode == "ReActive":
                response = st.session_state.agent.run(question)
            else:
                # Use the pre-planning approach from agent_analyst_task.py
                if "history" not in st.session_state:
                    st.session_state.history = []
                response = handle_question(question, st.session_state.history, "Pre-planning", streamlit_available=True, return_full_results=False)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about the customer service dataset"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response based on planning mode
            if planning_mode == "ReActive":
                response = st.session_state.agent.run(prompt)
            else:
                # Use the pre-planning approach from agent_analyst_task.py
                if "history" not in st.session_state:
                    st.session_state.history = []
                response = handle_question(prompt, st.session_state.history, "Pre-planning", streamlit_available=True, return_full_results=False)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
