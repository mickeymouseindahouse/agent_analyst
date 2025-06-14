import os
import streamlit as st
from agent.agent import ReActAgent
from tools.tools import get_tools
from agent_analyst_task import handle_question

# Test questions for both approaches
TEST_QUESTIONS = [
    "What are the most frequent categories?",
    "What are the most frequent intents?",
    "what are all the intents?",
    "what are all the categories?",
    "what are all the values in the category column?",
    "do we have category order with intent other than cancel_order?",
    "which intents exist when category is order?",
    "which categories exist when intent is Obtain invoice?",
    "Show examples of Category account",
    "Show examples of intent",
    "what are the intents of delivery",
    "Show examples of Category contact",
    "Show examples of intent View invoice",
    "Show examples of Category refund",
    "Show examples of Category order",
    "What categories exist?",
    "What intents exist?",
    "which intents exist?",
    "Show intent distributions",
    "Show category distributions",
    "can you give me an example of canceled order?",
    "can you summarize me how do custommers approach us when canceling order",
    "why are they asking for canceling order",
    "how agents respond to that",
    "what are the main tactics of response?",
    "display category distribution",
    "display intent distribution",
    "what is the intent distribution",
    "is there a category refund that its intent is not Review refund policy",
    "i want to see intents",
    "how many people asked to get a refund?",
    "what kind of data do we have in the dataset?",
    "what data do we have?",
    "what is the data",
    "what customers ask or request regarding Newsletter subscription",
    "give examples of customer questions or requests about contact",
    "what's your name",
    "are you stupid?",
    "ok whats in scope then?",
    "any suggestion for a question i can ask you",
    "are you connected to a dataset ?",
    "you are stupid",
    "do u know the name of the company",
    "ok thanks. can i teach you things and then you'll know them?",
    "do you have prices in the dataset?",
    "whats the complaint that is getting solved least times",
    "can you find requests that have replies which are inadequate",
    "how many people in total contacted us?",
    "how many customers in total sent us questions?",
    "which delivery options customers asked for",
    "what are the shipping methods?",
    "what are the categories and intents?",
    "what are the account types?",
    "to wich accounts users switched?",
    "How do I track the status of my order?",
    "do you have costs in the dataset?"
]

def run_react_tests():
    """Run tests using the ReAct agent approach"""
    st.header("ReAct Agent Tests")
    
    # Initialize ReAct agent
    tools = get_tools()
    agent = ReActAgent(tools=tools)
    
    for i, question in enumerate(TEST_QUESTIONS):
        st.markdown(f"### Test {i+1}: {question}")
        response = agent.run(question)
        st.write("Response:", response)
        st.write("---")

def run_preplanned_tests():
    """Run tests using the pre-planned approach"""
    st.header("Pre-planned Agent Tests")
    
    for i, question in enumerate(TEST_QUESTIONS):
        st.markdown(f"### Test {i+1}: {question}")
        response = handle_question(question, [], "Pre-planning")
        st.write("Response:", response)
        st.write("---")

if __name__ == "__main__":
    st.set_page_config(page_title="Agent Tests", layout="wide")
    st.title("Agent Testing")
    
    test_mode = st.sidebar.radio(
        "Test Mode",
        ["ReAct", "Pre-planned", "Both"]
    )
    
    if test_mode in ["ReAct", "Both"]:
        run_react_tests()
        
    if test_mode in ["Pre-planned", "Both"]:
        run_preplanned_tests()
