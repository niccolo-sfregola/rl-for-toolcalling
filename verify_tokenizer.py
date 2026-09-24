"""
Script to check the format of the lookup tool in order to build the environment,
since I need to build the prompt that describes the tool to the model, and
parse what the model generates to figure out whether it called
the tool and with which arguments
"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct") #tokenizer object


lookup_dict = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description":"Look in the knowledge graph the value associated with (entity,relation)",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type":"string"},
                "relation": {"type":"string"}
            },
            "required": ["entity","relation"],
        },
    },
}



message = [{
    "role":"user",
    "content":"Find in the graph who governs Aldoria"

}, {
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": {
                    "entity": "Aldoria",
                    "relation": "governed_by"
                }
            }
        }
    ]
}, {
    "role": "tool",
    "content": "Brevon"
}
]


temp = tok.apply_chat_template(message,tokenize=False, tools=[lookup_dict])
print(temp, type(temp))
