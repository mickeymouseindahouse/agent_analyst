
import pandas as pd
import os
from dotenv import load_dotenv
import json
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from datasets import load_dataset
import streamlit as st
import re

class CodeResponse(BaseModel):
    thoughts: str = Field(..., description="Step-by-step reasoning about the query")
    scope: bool = Field(..., description='boolean value. True if the user question is in-scope. False if the user question is out-of-scope') 
    pandas_code: str = Field(..., description="A complete pandas code snippet that assigns to variable 'result'")

load_dotenv()

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

# Load Bitext dataset
@st.cache_data
def load_bitext_data():
    return load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").to_pandas()

df = load_bitext_data()

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Toggle planning mode
planning_mode = st.radio("Select Planning Mode", ["Pre-planning", "ReActive"])

def remove_think_tags(text):
    """Remove all content between <think> and </think> tags, including the tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def fix_non_ascii_operators(code):
    return (
        code.replace("≤", "<=")
            .replace("≥", ">=")
            .replace("≠", "!=")
    )

# Prompt

def make_prompt(user_query, history, mode):
    messages = [{"role": "system", "content": f"""You are a helpful data analyst assistant working on a customer support dataset.
        The schema of the dataset is: {str(df.dtypes.to_dict())}.
        The unique values of the category column are: {str(list(df['category'].unique()))}.
        The unique values of the intent column are: {str(list(df['intent'].unique()))}.

        Given a user question, respond in strict JSON format with three fields:
        - 'thoughts': a string explaining your reasoning before the decision about the scope and generating the code.
        - 'scope': a boolean. True if the user question is in-scope. False if the user question is out-of-scope.
        - 'pandas_code': a single valid Python statement that assigns a DataFrame or Series to a variable named 'result'. If the scope field is False the pandas_code field will contain an empty string ''.

        Examples:
        
        1. For questions like 'what are all the categories' or 'What categories exist?' 
        or 'What are all the values in the category column?' or 'Show examples of category':
           scope: True
           pandas_code: result = pd.Series(df['category'].unique()).reset_index(drop=True)

        2. For questions like 'provide 10 examples of Category order':
           scope: True
           pandas_code: result = df[df['category']=='ORDER'].reset_index(drop=True).sample(10)

        3. For questions like 'Summarize Category invoice':
           scope: True
           pandas_code: result = df[df['category']=='INVOICE'].reset_index(drop=True).sample(15)
           
        4. For questions like 'which intents exist when category is account?':
           scope: True
           pandas_code: result = pd.Series(df.loc[df['category']=='ACCOUNT', 'intent'].unique()).reset_index(drop=True)

        5. For questions like 'do we have category order with intent other than cancel order?':
           scope: True
           pandas_code: result = pd.Series(df.loc[(df['category'] == 'ORDER') & (df['intent'] != 'cancel_order'), 'intent'].unique()).reset_index(drop=True)
        
        6. For questions like 'What are the most frequent categories?' or 'Which categories are most frequent?':
           scope: True
           pandas_code: result = df['category'].value_counts().reset_index(name='count').rename(columns={{'index': 'category'}})

        7. For questions like 'Show 5 examples of intent View invoice':
           scope: True
           pandas_code: result = df[(df['intent']=='get_invoice') | (df['intent']=='check_invoice')].reset_index(drop=True).sample(5)

        8. For questions like 'Summarize how agent respond to Intent Delivery options':
           scope: True
           pandas_code: result = df.loc[df['intent'] == 'delivery_options'), ['intent', 'response']].reset_index(drop=True).sample(15)

        9. For questions like 'what customers ask or request regarding Newsletter subscription':
           scope: True
           pandas_code: result = df.loc[df['intent'] == 'newsletter_subscription'), ['intent', 'instruction']].reset_index(drop=True).sample(15)

        10. For questions like 'give 6 examples of customer questions or requests about contact':
            scope: True
            pandas_code: result = df.loc[df['category'] == 'CONTACT'), ['category', 'instruction']].reset_index(drop=True).sample(6)

        11. For questions like 'what kind of data do we have in the dataset?' or 'what data do we have?' or "what is the data" 
        or "what is in scope" or "are you connected to a dataset" or "any suggestion for a question i can ask you":
            scope: True
            pandas_code: result = df.groupby(['category', 'intent'], as_index=False).agg({{'instruction': 'first', 'response': 'first'}}).sort_values('category').reset_index(drop=True)

        12. For questions like 'Who is Magnus Carlson?' or "What is Serj's rating?" or 'do u know the name of the company':
            scope: False
            pandas_code: ""

        13. For questions like 'do you have prices in the dataset?'
            scope: True
            pandas_code: result = df[df['instruction'].str.contains('price', case=False, na=False) | df['response'].str.contains('price', case=False, na=False)]
        
        14. For questions like 'can you find requests that have replies which are inadequate':
            scope: True
            pandas_code: result = df[['instruction', 'response']].reset_index(drop=True).sample(15)


            
        """}] 
    
    
    for past in history:
        messages.append({"role": "user", "content": past['user']})
        messages.append({"role": "assistant", "content": past['assistant']})

    if mode == "Pre-planning":
        user_query = "Please plan your steps before answering. " + str(user_query)

    messages.append({"role": "user", "content": str(user_query)})
    
    return messages

def describe_result_with_llm(result, user_query):
    """Send the result to the LLM for a natural language description."""
    if isinstance(result, (pd.DataFrame, pd.Series)):
        result_str = result.to_string()
    else:
        result_str = str(result)
    messages = [
        {"role": "system", "content": """You are a helpful data analyst assistant.
            Given the following data analysis result and the user's original query, describe the results in clear, non-technical language.
            Keep your answer concise and focused on what the user asked."""},
        {"role": "user", "content": f"User's question: {user_query}\n\nAnalysis result:\n{result_str}"}
    ]
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        temperature=0,
        messages=messages
    )
    reply = response.choices[0].message.content.strip()
    
    # Remove <think> ... </think> blocks
    return remove_think_tags(reply)

def ask_llm_to_fix_code(user_query, messages, history, mode, error_msg, code):
    """Ask the LLM to fix the code based on the error."""
    messages.append({"role": "user", "content": f"Fix this pandas code that is related to the user question: {user_query}.\n\nThe Code you generated:\n{code} \n\nThe error: {error_msg}"})
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content.strip()


# Execute structured question
def handle_question(query, history, mode, max_retries=3):
    q = query.lower()
    messages = make_prompt(q, history, mode)
    retry_count = 0
    not_executed = True

    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        temperature=0,
        messages=messages,
        extra_body={"guided_json": CodeResponse.model_json_schema()}
        )

    reply_raw = response.choices[0].message.content.strip()
    reply_cleaned = remove_think_tags(reply_raw)

    
    parsed = CodeResponse.model_validate_json(reply_cleaned)
    code = parsed.pandas_code
    thoughts = parsed.thoughts
    scope = parsed.scope
   
    
    if scope == False:
        return "Sorry, that question is out of scope for this dataset. If you're not sure what kind of data I have, feel free to ask me."

    code = fix_non_ascii_operators(code)

    if not code.strip().startswith("result ="):
        raise SyntaxError("Code must assign to variable 'result'")

    exec_env = {'df': df.copy(), 'pd': pd}
    while retry_count < max_retries and not_executed:
        try:    
            exec(code, exec_env)
            result = exec_env.get('result')

            # Limit to 20 rows if DataFrame or Series is too long to avoid reaching token limit of the model 
            if isinstance(result, (pd.DataFrame, pd.Series)) and len(result) > 20:
                result = result.sample(20, random_state=42).reset_index(drop=True)
            
            if result is not None:
                st.markdown("**LLM Thought Process:**")
                st.write(thoughts) 
                st.write("code:", code)
            else:
                return "No results generated - check code formatting"
                
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
                not_executed = False
                description = describe_result_with_llm(result, query)
                return description
            elif isinstance(result, int):
                st.write("The answer is an int type")
                return str(result)
            else:
                return str(result)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {str(e)}, code: {code if 'code' in locals() else reply_cleaned}"
            retry_count += 1
            st.write(f"Attempt {retry_count}: {error_msg}")

            fixed_code_reply = ask_llm_to_fix_code(query, messages, history, mode, error_msg, reply_cleaned)
            try:
                parsed = CodeResponse.model_validate_json(fixed_code_reply)
                code = parsed.pandas_code
            except Exception as parse_error:
                return f"There was an error that I could not fix. Please try to rephrase your question.\nParse error: {str(parse_error)}\nResponse: {fixed_code_reply}"

    return f"Could not fix the code after {max_retries} attempts. Last error: {error_msg}"

# Auto test mode (for running tests without using the Streamlit input box)
if "RUN_TESTS" in os.environ and os.environ["RUN_TESTS"] == "1":
    test_questions = [
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

    for i, question in enumerate(test_questions):
        st.markdown(f"### Test {i+1}: {question}")
        response = handle_question(question, st.session_state.history, planning_mode)
        st.write("Response:", response)
        st.write("---")
else:

    # Input from user
    st.title("Data Analyst Agent - Bitext Assistant")
    query = st.text_input("Ask a question to the data analyst agent:")
    
    if query:
            response = handle_question(query, st.session_state.history, planning_mode)
    
            # Store in history
            st.session_state.history.append({"user": query, "assistant": response})
        
            # Display full history
            st.markdown("### Conversation History")
            for turn in reversed(st.session_state.history):
                st.markdown(f"**User:** {turn['user']}")
                st.markdown(f"**Assistant:** {turn['assistant']}")
