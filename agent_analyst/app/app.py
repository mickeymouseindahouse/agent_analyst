import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent_analyst.tools.tools import get_tools
from agent_analyst.agent.agent import ReActAgent
from agent_analyst.data.download_dataset import load_dataset_df

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
    ["Pre-planning + Execution", "ReActive Dynamic Planning"],
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
        
        # Get response from agent
        with st.spinner("Thinking..."):
            response = st.session_state.agent.run(
                question, 
                dynamic_planning=(planning_mode == "ReActive Dynamic Planning")
            )
        
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
            # Get response from agent
            response = st.session_state.agent.run(
                prompt, 
                dynamic_planning=(planning_mode == "ReActive Dynamic Planning")
            )
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
