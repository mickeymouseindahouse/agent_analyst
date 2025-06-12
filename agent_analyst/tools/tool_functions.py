from typing import List, Dict, Any, Optional, Union
import pandas as pd
import openai
import os
from agent_analyst.data.download_dataset import load_dataset_df

# Load the dataset
df = load_dataset_df()

def select_semantic_intent(intent_name: List[str]) -> Dict[str, Any]:
    """
    Select conversations with specific intents.
    
    Args:
        intent_name: List of intent names to select
        
    Returns:
        Dictionary with selected intents, count, and examples
    """
    filtered_df = df[df['intent'].isin(intent_name)]
    
    return {
        "selected_intents": intent_name,
        "count": len(filtered_df),
        "examples": filtered_df.head(3)[['instruction', 'intent', 'response']].to_dict('records')
    }

def select_semantic_category(category_name: List[str]) -> Dict[str, Any]:
    """
    Select conversations with specific categories.
    
    Args:
        category_name: List of category names to select
        
    Returns:
        Dictionary with selected categories, count, and examples
    """
    filtered_df = df[df['category'].isin(category_name)]
    
    return {
        "selected_categories": category_name,
        "count": len(filtered_df),
        "examples": filtered_df.head(3)[['instruction', 'category', 'response']].to_dict('records')
    }

def sum_numbers(a: float, b: float) -> Dict[str, float]:
    """
    Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Dictionary with the result
    """
    return {"result": a + b}

def count_category(category: str) -> Dict[str, int]:
    """
    Count conversations in a category.
    
    Args:
        category: Category name to count
        
    Returns:
        Dictionary with the count
    """
    count = len(df[df['category'] == category])
    return {"count": count}

def count_intent(intent: str) -> Dict[str, int]:
    """
    Count conversations with an intent.
    
    Args:
        intent: Intent name to count
        
    Returns:
        Dictionary with the count
    """
    count = len(df[df['intent'] == intent])
    return {"count": count}

def show_examples(n: int = 3, intent: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Show n example conversations.
    
    Args:
        n: Number of examples to show
        intent: Optional intent to filter by
        category: Optional category to filter by
        
    Returns:
        Dictionary with examples
    """
    filtered_df = df.copy()
    
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    examples = filtered_df.head(n)[['instruction', 'intent', 'category', 'response']].to_dict('records')
    
    return {
        "examples": examples,
        "total_matching": len(filtered_df),
        "shown": min(n, len(filtered_df))
    }

def summarize(user_request: str, intent: Optional[str] = None, category: Optional[str] = None) -> Dict[str, str]:
    """
    Generate a summary based on the user request using an LLM.
    
    Args:
        user_request: User request to summarize
        intent: Optional intent to summarize
        category: Optional category to summarize
        
    Returns:
        Dictionary with the summary
    """
    import openai
    import os
    
    # Filter the dataset based on intent and category if provided
    filtered_df = df.copy()
    
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    # Extract relevant data for summarization
    total_count = len(filtered_df)
    
    if total_count == 0:
        return {"summary": "No data found matching the specified criteria."}
    
    # Sample conversations to send to the LLM
    # Limit to a reasonable number to avoid token limits
    sample_size = min(20, total_count)
    sample_data = filtered_df.sample(sample_size)[['instruction', 'intent', 'category', 'response']]
    
    # Format the data for the LLM
    formatted_data = ""
    for _, row in sample_data.iterrows():
        formatted_data += f"Customer: {row['instruction']}\n"
        formatted_data += f"Intent: {row['intent']}, Category: {row['category']}\n"
        formatted_data += f"Agent: {row['response']}\n\n"
    
    # Create a prompt for the LLM
    prompt = f"""Based on the following {sample_size} customer service conversations 
{f"with intent '{intent}'" if intent else ""} 
{f"in category '{category}'" if category else ""}
please provide a detailed summary addressing: "{user_request}"

The summary should include:
1. Common patterns in customer queries
2. Typical agent response strategies
3. Key phrases or approaches used by agents
4. Any notable insights about how these conversations are handled

Conversations:
{formatted_data}
"""
    
    # Call the OpenAI API for summarization
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # You can use a different model if preferred
            messages=[
                {"role": "system", "content": "You are an AI assistant that summarizes customer service conversations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000  # Adjust as needed
        )
        
        summary = response.choices[0].message.content
        
        # Add context about the data
        summary_with_context = f"Summary based on analysis of {total_count} conversations"
        if intent:
            summary_with_context += f" with intent '{intent}'"
        if category:
            summary_with_context += f" in category '{category}'"
        summary_with_context += f":\n\n{summary}"
        
        return {"summary": summary_with_context}
    
    except Exception as e:
        return {"summary": f"Error generating summary: {str(e)}. Please try again with different parameters."}

def get_intent_distribution(top_n: int = 10, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the distribution of intents in the dataset.
    
    Args:
        top_n: Number of top intents to show
        category: Optional category to filter by
        
    Returns:
        Dictionary with the intent distribution
    """
    filtered_df = df.copy()
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    intent_counts = filtered_df['intent'].value_counts().head(top_n).to_dict()
    
    return {
        "intent_distribution": intent_counts,
        "total_conversations": len(filtered_df),
        "filter_category": category
    }

def get_category_distribution(top_n: int = 10) -> Dict[str, Any]:
    """
    Get the distribution of categories in the dataset.
    
    Args:
        top_n: Number of top categories to show
        
    Returns:
        Dictionary with the category distribution
    """
    category_counts = df['category'].value_counts().head(top_n).to_dict()
    
    return {
        "category_distribution": category_counts,
        "total_conversations": len(df)
    }

# Map function names to their implementations
TOOL_FUNCTIONS = {
    "select_semantic_intent": select_semantic_intent,
    "select_semantic_category": select_semantic_category,
    "sum": sum_numbers,
    "count_category": count_category,
    "count_intent": count_intent,
    "show_examples": show_examples,
    "summarize": summarize,
    "get_intent_distribution": get_intent_distribution,
    "get_category_distribution": get_category_distribution
}
