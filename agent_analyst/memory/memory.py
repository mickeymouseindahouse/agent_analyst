from typing import Dict, List, Any, Optional
import json
import os
import datetime

class Memory:
    def __init__(self, memory_file: str = "agent_memory.json"):
        self.memory_file = memory_file
        self.memories = self._load_memories()
        
    def _load_memories(self) -> Dict[str, Any]:
        """Load memories from file or initialize if not exists"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {
            "interactions": [],
            "summaries": {},
            "insights": {},
            "metadata": {"last_updated": None}
        }
    
    def _save_memories(self):
        """Save memories to file"""
        self.memories["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def add_interaction(self, query: str, response: str, tools_used: List[str]):
        """Add a new interaction to memory"""
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response,
            "tools_used": tools_used
        }
        self.memories["interactions"].append(interaction)
        self._save_memories()
    
    def add_summary(self, key: str, summary: str):
        """Add or update a summary in memory"""
        self.memories["summaries"][key] = {
            "content": summary,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self._save_memories()
    
    def add_insight(self, key: str, insight: str):
        """Add or update an insight in memory"""
        self.memories["insights"][key] = {
            "content": insight,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self._save_memories()
    
    def get_recent_interactions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent interactions"""
        return self.memories["interactions"][-n:]
    
    def get_summary(self, key: str) -> Optional[str]:
        """Get a summary by key"""
        if key in self.memories["summaries"]:
            return self.memories["summaries"][key]["content"]
        return None
    
    def get_insight(self, key: str) -> Optional[str]:
        """Get an insight by key"""
        if key in self.memories["insights"]:
            return self.memories["insights"][key]["content"]
        return None
    
    def summarize_interactions(self, client) -> str:
        """Generate a summary of all interactions using LLM"""
        if not self.memories["interactions"]:
            return "No interactions to summarize."
        
        # Get the last 20 interactions or all if fewer
        interactions = self.memories["interactions"][-20:]
        
        # Format interactions for the LLM
        formatted_interactions = ""
        for i, interaction in enumerate(interactions):
            formatted_interactions += f"Interaction {i+1}:\n"
            formatted_interactions += f"Query: {interaction['query']}\n"
            formatted_interactions += f"Response: {interaction['response'][:200]}...\n"
            formatted_interactions += f"Tools used: {', '.join(interaction['tools_used'])}\n\n"
        
        # Create prompt for summarization
        prompt = f"""Please summarize the following user interactions with the customer service dataset Q&A agent:

{formatted_interactions}

Provide a concise summary of:
1. Common types of questions asked
2. Patterns in tool usage
3. Key insights from these interactions
4. Suggestions for improving the agent based on these interactions
"""
        
        # Call the LLM for summarization
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that summarizes agent interactions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content
            
            # Store this summary
            self.add_summary("interaction_summary", summary)
            
            return summary
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def get_relevant_memories(self, query: str, client) -> str:
        """Retrieve memories relevant to the current query"""
        if not self.memories["interactions"]:
            return "No previous interactions available."
        
        # Format the query and recent interactions for the LLM
        recent_interactions = self.get_recent_interactions(10)
        formatted_interactions = ""
        for i, interaction in enumerate(recent_interactions):
            formatted_interactions += f"Interaction {i+1}:\n"
            formatted_interactions += f"Query: {interaction['query']}\n"
            formatted_interactions += f"Response: {interaction['response'][:100]}...\n\n"
        
        # Create prompt for memory retrieval
        prompt = f"""Given the following new user query and recent interactions with the agent, 
identify any relevant information from past interactions that could help answer this query.

New query: {query}

Recent interactions:
{formatted_interactions}

If there are relevant past interactions, summarize the key information that could help answer the current query.
If there are no relevant past interactions, respond with "No relevant past information."
"""
        
        # Call the LLM for memory retrieval
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that retrieves relevant information from past interactions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            relevant_info = response.choices[0].message.content
            
            if relevant_info == "No relevant past information.":
                return ""
                
            return relevant_info
        
        except Exception as e:
            return ""
