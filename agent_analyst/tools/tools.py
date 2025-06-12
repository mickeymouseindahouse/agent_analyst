from typing import List, Dict, Any, Optional, Union
import pandas as pd
import json
import os
from agent_analyst.data.download_dataset import load_dataset_df

# Load the dataset
df = load_dataset_df()

# Extract unique intents and categories for tool definitions
INTENTS = sorted(df['intent'].unique().tolist())
CATEGORIES = sorted(df['category'].unique().tolist())

def get_tools() -> List[Dict[str, Any]]:
    """
    Get the list of tools available to the agent.
    
    Returns:
        List of tools
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "select_semantic_intent",
                "description": "Select conversations with specific intents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent_name": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": INTENTS
                            },
                            "description": "List of intent names to select"
                        }
                    },
                    "required": ["intent_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "select_semantic_category",
                "description": "Select conversations with specific categories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category_name": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": CATEGORIES
                            },
                            "description": "List of category names to select"
                        }
                    },
                    "required": ["category_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "sum",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_category",
                "description": "Count conversations in a category",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": CATEGORIES,
                            "description": "Category name to count"
                        }
                    },
                    "required": ["category"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_intent",
                "description": "Count conversations with an intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": INTENTS,
                            "description": "Intent name to count"
                        }
                    },
                    "required": ["intent"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "show_examples",
                "description": "Show n example conversations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {
                            "type": "integer",
                            "description": "Number of examples to show",
                            "default": 3
                        },
                        "intent": {
                            "type": "string",
                            "enum": INTENTS,
                            "description": "Optional: Filter by intent"
                        },
                        "category": {
                            "type": "string",
                            "enum": CATEGORIES,
                            "description": "Optional: Filter by category"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": "Generate a summary based on the user request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_request": {
                            "type": "string",
                            "description": "User request to summarize"
                        },
                        "intent": {
                            "type": "string",
                            "enum": INTENTS,
                            "description": "Optional: Intent to summarize"
                        },
                        "category": {
                            "type": "string",
                            "enum": CATEGORIES,
                            "description": "Optional: Category to summarize"
                        }
                    },
                    "required": ["user_request"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_intent_distribution",
                "description": "Get the distribution of intents in the dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top intents to show",
                            "default": 10
                        },
                        "category": {
                            "type": "string",
                            "enum": CATEGORIES,
                            "description": "Optional: Filter by category"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_category_distribution",
                "description": "Get the distribution of categories in the dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top categories to show",
                            "default": 10
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Return the final answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Final answer to the user's question"
                        }
                    },
                    "required": ["answer"]
                }
            }
        }
    ]
