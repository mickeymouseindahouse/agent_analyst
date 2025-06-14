import json
import openai
import os
from typing import List, Dict, Any, Optional, Union
from tools.tool_functions import TOOL_FUNCTIONS
from memory.memory import Memory

class ReActAgent:
    """
    ReAct agent that uses function calling to answer questions about the dataset.
    """
    
    def __init__(self, tools: List[Dict[str, Any]]):
        """
        Initialize the ReAct agent with tools.
        
        Args:
            tools: List of tools available to the agent
        """
        self.tools = tools
        self.tool_map = {tool["function"]["name"]: tool for tool in tools if "function" in tool}
        
        # Get API key from environment variable
        self.api_key = os.environ.get("NEBIUS_API_KEY")
        if not self.api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set")
        
        # Initialize OpenAI client with Nebius API endpoint
        self.client = openai.OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=self.api_key
        )
        
        # Initialize memory
        self.memory = Memory()
        
        # Track tools used in the last run
        self._last_tools_used = []
        
    def run(self, query: str) -> str:
        """
        Run the agent to answer the user's query using ReActive approach.
        
        Args:
            query: User's question
            
        Returns:
            Agent's response
        """
        # Reset tools used tracking
        self._last_tools_used = []
        
        # Get relevant memories
        relevant_memories = self.memory.get_relevant_memories(query, self.client)
        
        # Run the agent with ReActive approach
        response = self._run_reactive(query, relevant_memories)
        
        # Add tools used to the response, each on a separate line with green-blue marking
        if self._last_tools_used:
            tools_used_str = "\n\n**Tools used:**"
            for tool in self._last_tools_used:
                tools_used_str += f"\n- <span style='color:#00a0b0'>**{tool}**</span>"
            response += tools_used_str
        
        # Store the interaction in memory
        self.memory.add_interaction(
            query=query,
            response=response,
            tools_used=self._last_tools_used
        )
        
        return response
    
    def _run_reactive(self, query: str, relevant_memories: str = "") -> str:
        """
        Run the agent in ReActive mode (dynamic planning).
        
        Args:
            query: User's question
            relevant_memories: Relevant information from past interactions
            
        Returns:
            Agent's response
        """
        system_prompt = self._get_system_prompt()
        
        # Add relevant memories to the system prompt if available
        if relevant_memories:
            system_prompt += f"\n\nRelevant information from past interactions:\n{relevant_memories}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Maximum number of steps to prevent infinite loops
        max_steps = 10
        step = 0
        
        while step < max_steps:
            step += 1
            
            # Call the model
            response = self.client.chat.completions.create(
                model="Qwen/Qwen3-30B-A3B",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            # Check if the model wants to call a function
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Track tool usage
                    if function_name not in self._last_tools_used:
                        self._last_tools_used.append(function_name)
                    
                    # Execute the function
                    if function_name == "finish":
                        # Return the final answer
                        return function_args.get("answer", "No answer provided.")
                    
                    # Call the appropriate tool function
                    tool_result = self._execute_tool(function_name, function_args)
                    
                    # Special handling for show_dataframe to display pandas dataframe
                    if function_name == "show_dataframe" and "pandas_df" in tool_result:
                        import streamlit as st
                        st.markdown(f"<span style='color:#00a0b0'>**Tool called: {function_name}**</span>", unsafe_allow_html=True)
                        st.write("### Dataset Preview:")
                        st.dataframe(tool_result["pandas_df"])
                        # Remove the pandas_df from the result to avoid serialization issues
                        del tool_result["pandas_df"]
                    else:
                        import streamlit as st
                        st.markdown(f"<span style='color:#00a0b0'>**Tool called: {function_name}**</span>", unsafe_allow_html=True)
                    
                    # Add the function response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })
            else:
                # If no function call, return the response directly
                return response_message.content
        
        return "Reached maximum number of steps without a final answer."
    
    def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool function with the given arguments.
        
        Args:
            function_name: Name of the function to call
            function_args: Arguments to pass to the function
            
        Returns:
            Result of the function call
        """
        if function_name not in TOOL_FUNCTIONS:
            return {"error": f"Function {function_name} not implemented"}
        
        try:
            return TOOL_FUNCTIONS[function_name](**function_args)
        except Exception as e:
            return {"error": f"Error executing {function_name}: {str(e)}"}
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the ReAct agent.
        
        Returns:
            System prompt
        """
        return """You are an AI assistant that helps answer questions about a customer service dataset.
You have access to tools that can query and analyze the dataset.
Follow these steps:
1. Understand the user's question
2. Determine which tools you need to use to answer the question
3. Call the appropriate tools with the right parameters
4. Synthesize the results into a clear answer
5. When you have the final answer, call the finish tool

The dataset contains customer service conversations with intents and categories.
The dataset includes various intents like edit_account, switch_account, check_invoice, etc.
Categories include ACCOUNT, ORDER, REFUND, INVOICE, etc.

For out-of-scope questions not related to the dataset, politely explain that you can only answer questions about the customer service dataset.
"""
    
    def summarize_interactions(self) -> str:
        """
        Generate a summary of all interactions using the memory system.
        
        Returns:
            Summary of interactions
        """
        return self.memory.summarize_interactions(self.client)
