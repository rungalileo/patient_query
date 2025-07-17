import json
from typing import List, Dict, Type, Any

def scorer_fn(*, node_input: str, node_output: str, **kwargs) -> int:
    """
    Define a scorer function on a single node.
    This function checks the planner_output key in the JSON output.
    If planner_output is "unknown", returns 0. Otherwise returns 1.
    
    Available args: node_input, node_output, node_name, tools, credentials
    """
    try:
        # Parse the node_output as JSON
        output_data = json.loads(node_output)
        
        # Check the planner_output key
        planner_output = output_data.get("planner_output", "")
        
        # Return 0 if unknown, 1 otherwise
        if planner_output == "unknown":
            return 0
        else:
            return 1
            
    except (json.JSONDecodeError, KeyError, TypeError):
        # If there's any error parsing the JSON or accessing the key, return 0
        return 0 