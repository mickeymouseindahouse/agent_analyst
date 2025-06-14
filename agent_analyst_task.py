
import pandas as pd
import os
from dotenv import load_dotenv
import json
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from datasets import load_dataset
import re
# Removed global streamlit import

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
def load_bitext_data():
    return load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").to_pandas()

df = load_bitext_data()

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
def handle_question(query, history, mode, streamlit_available=True, return_full_results=False, max_retries=3):
    """
    Handle a question using the pre-planning approach.
    
    Args:
        query: User's question
        history: Conversation history
        mode: Planning mode
        streamlit_available: Whether streamlit is available for UI display
        return_full_results: Whether to return full results dict instead of just description
        max_retries: Maximum number of retries for code execution
        
    Returns:
        If return_full_results is False: Returns just the description string
        If return_full_results is True: Returns a dict with all results
    """
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
                results_data = {
                    "thoughts": thoughts,
                    "code": code,
                    "result": result,
                    "result_type": "dataframe" if isinstance(result, (pd.DataFrame, pd.Series)) else "scalar"
                }
                
                # Only use streamlit if explicitly requested
                if streamlit_available:
                    import streamlit as st
                    # Show thought process as requested
                    st.markdown("**LLM Thought Process:**")
                    st.write(thoughts) 
                    st.write("code:", code)
                    
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result)
                
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    not_executed = False
                    description = describe_result_with_llm(result, query)
                    results_data["description"] = description
                    return results_data if return_full_results else description
                elif isinstance(result, int):
                    if streamlit_available:
                        import streamlit as st
                        st.write("The answer is an int type")
                    return results_data if return_full_results else str(result)
                else:
                    return results_data if return_full_results else str(result)
            else:
                return "No results generated - check code formatting"

        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {str(e)}, code: {code if 'code' in locals() else reply_cleaned}"
            retry_count += 1
            
            if streamlit_available:
                import streamlit as st
                st.write(f"Attempt {retry_count}: {error_msg}")

            fixed_code_reply = ask_llm_to_fix_code(query, messages, history, mode, error_msg, reply_cleaned)
            try:
                parsed = CodeResponse.model_validate_json(fixed_code_reply)
                code = parsed.pandas_code
            except Exception as parse_error:
                return f"There was an error that I could not fix. Please try to rephrase your question.\nParse error: {str(parse_error)}\nResponse: {fixed_code_reply}"

    return f"Could not fix the code after {max_retries} attempts. Last error: {error_msg}"

# Tests have been moved to test_agents.py
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Data Analyst Agent - Bitext Assistant", layout="wide")
    st.title("Data Analyst Agent - Bitext Assistant")
    st.write("This file is meant to be imported, not run directly.")
    st.write("Please use the app.py file to run the application.")
    st.write("For testing, use test_agents.py")
# This code has been removed as it's now handled by app.py
