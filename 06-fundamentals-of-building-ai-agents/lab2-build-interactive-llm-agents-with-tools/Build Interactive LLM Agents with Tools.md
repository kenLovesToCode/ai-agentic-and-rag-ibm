<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Build Interactive LLM Agents with Tools**


Estimated time needed: **15** minutes


In this lab, you'll explore the powerful capabilities of tool calling in large language models (LLMs) to build advanced AI agents that can dynamically interact with users. Using the LangChain framework, you’ll learn how to build an interactive agent that responds to user queries by selecting and executing the right function at the right time. This hands-on approach will help you understand how LLMs can be extended with real-world functionality, bridging natural language understanding with dynamic, tool-based actions.


## __Table of Contents__

- [Objectives](#Objectives)
- [Setup](#Setup)
    - [Installing Required Libraries](#Installing-Required-Libraries)
    - [Importing Required Libraries](#Importing-Required-Libraries)
- [Creating Custom Tools with LangChain](#Creating-Custom-Tools-with-LangChain)
    - [Anatomy of a tool](#Anatomy-of-a-tool)
    - [Key components](#Key-components)
    - [Defining an add function](#Defining-an-add-function)
    - [Add tools to the LLM](#Add-tools-to-the-LLM)
    - [Create more Tools](#Create-more-tools)
    - [Testing the functions](#Testing-the-functions)
    - [Add new tools to LLM](#Add-new-tools-to-LLM)
- [Interacting with the Model](#Interacting-with-the-Model)
    - [Craft the user query](#Craft-the-user-query)
    - [Invoke the model](#Invoke-the-model)
    - [Parse tool calls](#Parse-tool-calls)
    - [Invoke the tool](#Invoke-the-Tool)
    - [Generate a final answer from chat history](#Generate-a-final-answer-from-chat-history)
- [Building an Agent](#Building-an-Agent)
- [Conclusion](#Conclusion)
- [Exercises](#Exercises)
    - [Exercise 1: Create a New Tool](#Exercise-1:-Create-a-new-tool)
    - [Exercise 2: Tool Calling with an LLM](#Exercise-2:-Tool-calling-with-an-LLM)
    - [Exercise 3: Create a tip calculating agent](#Exercise-3:-Create-a-tip-calculating-agent)


## Objectives

After completing this lab you will be able to:

 - Initialize a chat model for tool interactions
 - Define and bind custom tools to the LLM for expanded functionality
 - Use mapping dictionaries for dynamic function calls
 - Extract tool names and functions for precise function calls
 - Build agent classes that manage the entire tool-calling process


----


## Setup


For this lab, you will be using the following libraries:

*   [`langchain`](https://python.langchain.com/docs/introduction/) is the framework you will build the agent on.
*   [`langchain-openai`](https://pypi.org/project/langchain-openai/) is a partner package of LangChain and integrates OpenAI LLMs to the framework.


### Installing Required Libraries



```python
%pip install langchain===0.3.25 | tail -n 1
%pip install langchain-openai===0.3.19 | tail -n 1
```

    Successfully installed langchain-0.3.25 langchain-core-0.3.83 langchain-text-splitters-0.3.11 langsmith-0.3.45 orjson-3.11.8 requests-toolbelt-1.0.0 uuid-utils-0.14.1
    Note: you may need to restart the kernel to use updated packages.
    Successfully installed jiter-0.13.0 langchain-openai-0.3.19 openai-1.109.1 regex-2026.4.4 tiktoken-0.12.0
    Note: you may need to restart the kernel to use updated packages.


### Importing Required Libraries
Recommendation:Import all required libraries in one place (here):_



```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
```

Let's initialize the language model that will power your tool calling capabilities. This code sets up a GPT-4o-mini model using the OpenAI provider through LangChain's interface, which you'll use to process queries and decide which tools to call.



```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
```

# API Disclaimer

This lab uses LLMs provided by Watsonx.ai and OpenAI. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to configure your own API keys. Please note that using your own API keys means that you will incur personal charges. 

### Running Locally
If you are running this lab locally, you will need to configure your own API keys. This lab uses `ChatOpenAI` and `ChatWatsonx` modules from `langchain`. Both configurations are shown below with instructions. **Replace all instances** of both modules with the completed modules below throughout the lab. **DO NOT** run the cell below if you aren't running locally, it will causes errors.



```python
# IGNORE IF YOU ARE NOT RUNNING LOCALLY
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key = "your openai api key here",
)
watsonx_llm = ChatWatsonx(
    model_id="ibm/granite-3-2-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your project id associated with the API key",
    api_key="your watsonx.ai api key here",
)
```

## Creating Custom Tools with LangChain

### Anatomy of a tool

Let's provide the basic building blocks of a tool, consider the following tool:

```python
@tool
def tool_name(input_param: input_type) -> output_type:
   """
   Clear description of what the tool does.
   
   Args:
       input_param (input_type): Description of this parameter
   
   Returns:
       output_type: Description of what is returned
   """
   # Function implementation
   result = process(input_param)
   return result
```

### Key components

You'll use the following key components

**@tool decorator**
   - Registers the function with LangChain
   - Creates tool attributes (.name, .description, .func)
   - Generates JSON schema for validation
   - Transforms regular functions into callable tools

**Function name**
   - Used by LLM to select appropriate tool
   - Used as reference in chains and tool mappings
   - Appears in tool call logs for debugging
   - Should clearly indicate the tool's purpose

**Type annotations**
   - Enable automatic input validation
   - Create schema for parameters
   - Allow proper serialization of inputs/outputs
   - Help LLM understand required input formats

**Docstring**
   - Provides context for the LLM to decide when to use the tool
   - Documents parameter requirements
   - Explains expected outputs and behavior
   - Is critical for tool selection by the LLM

6. **Implementation**
   - Executes the actual operation
   - Handles errors appropriately
   - Returns properly formatted results
   - Should be efficient and robust


### Defining an add function

Now use this tool framework to create a custom tool that enables the LLM to perform basic addition.



```python
@tool
def add(a: int, b: int) -> int:
    """
    Add a and b.
    
    Args:
        a (int): first integer to be added
        b (int): second integer to be added

    Return:
        int: sum of a and b
    """
    return a + b
```

The decorator wraps the `add()` function in LangChain's predefined tool schema. See more about defining custom LangChain tools [here](https://python.langchain.com/docs/how_to/custom_tools/).


### Add tools to the LLM

Let's connect and bind the function to the chat model.



```python
tools = [add]

llm_with_tools = llm.bind_tools(tools)
```

Use the `bind_tools(tools)` method to connect a list of tools to the LLM for use. From now on, whenever the call is invoked, the model (with tools) will recognize and use the add tool whenever it needs to compute a sum.


### Create more tools

Let's create some more basic arithmetic tools.



```python
@tool
def subtract(a: int, b:int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b."""
    return a * b
```

### Testing the functions

Let's setup a way to test your tools.



```python
tool_map = {
    "add": add, 
    "subtract": subtract,
    "multiply": multiply
}

input_ = {
    "a": 1,
    "b": 2
}

tool_map["add"].invoke(input_)
```




    3



Using LangChain's built in `.invoke(inputs)` method, you can test each tool built with dynamic inputs.Test each tool with the preceding code block.


### Add new tools to LLM

Let's add all three tools to the LLM.



```python
tools = [add, subtract, multiply]

llm_with_tools = llm.bind_tools(tools)
```

You can the same method to bind tools to the LLM, enabling more arithmetic capabilities.


## Interacting with the Model


### Craft the user query

Now that you've setup an LLM with basic tool integrations, it's time to introduce user queries.



```python
query = "What is 3 + 2?"
chat_history = [HumanMessage(content=query)]
```

First,setup the question (user query). Then,initialize a `chat_history` array that will contain the entire conversation between user and LLM. In this chat history, you insert the `query` in a `HumanMessage` wrapper that tells LangChain and the model: "This message came from the user."


### Invoke the model

Now let's run the model with the context (chat history) that contains the user query.



```python
response_1 = llm_with_tools.invoke(chat_history)
chat_history.append(response_1)

print(type(response_1))
#print(response_1)
```

    <class 'langchain_core.messages.ai.AIMessage'>


Using the `invoke(inputs)` method, you get a response from the model. You add the response to the chat history. The code block also prints out the type of the response which is the `AIMessage` class from LangChain. Uncomment the second print statement and read through the fields of the `AIMessage` response.


### Parse tool calls

Now that you have the response from the model, you can parse the response for tool calling instructions.



```python
tool_calls_1 = response_1.tool_calls

tool_1_name = tool_calls_1[0]["name"]
tool_1_args = tool_calls_1[0]["args"]
tool_call_1_id = tool_calls_1[0]["id"]

print(f'tool name:\n{tool_1_name}')
print(f'tool args:\n{tool_1_args}')
print(f'tool call ID:\n{tool_call_1_id}')
```

    tool name:
    add
    tool args:
    {'a': 3, 'b': 2}
    tool call ID:
    call_kXO8lzOjSByJkqc9JUOsboEN


- Extracting the `name` from the first call gives the name of the tool to use.
    - `add` in this case
- Extracting the `args` gives the inputs to pass into the tool.
    - `{a: 3, b: 2}` in this case
- Extracting the `id` gives the unique identifier for the tool call
    - The ID will be different each time, linking tool calls to their respective responses
    - Crucial in differentiating calls to the same tool and parallel tool calls


### Invoke the Tool

Given the tool call details from the LLM, invoke the correct tool with the correct arguments.



```python
tool_response = tool_map[tool_1_name].invoke(tool_1_args)
tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call_1_id)

print(tool_message)
```

    content='5' tool_call_id='call_kXO8lzOjSByJkqc9JUOsboEN'


Use the `tool_map`, passing in the tool name and parameters to get a response. Then wrap that response in a `ToolMessage` object from LangChain along with the tool call ID. This action allows the model and LangChain to better process tool responses and overall conversation between user and model and tool. Feel free to uncomment the print statement to see what the `tool_message` looks like.



```python
chat_history.append(tool_message)
```

Next, append the `tool_message` to the `chat_history` so the model preserves context and sees prior conversation for a better conversing experience. Now the chat history contains a `HumanMessage` (initial user query), an `AIMessage` (the response from the model), and a `ToolMessage` (the output of the tool).


### Generate a final answer from chat history

As a final step, pass the entire `chat_history` into the LLM one more time to get a final response.



```python
answer = llm_with_tools.invoke(chat_history)
print(type(answer))
print(answer.content)
```

    <class 'langchain_core.messages.ai.AIMessage'>
    3 + 2 equals 5.


Printing the `answer.content` (content field of the `AIMessage` object) gives the final result of the LLM for the user query. You have finished a complete interaction between the user and model.


## Building an Agent

You can wrap all the prior functionality in a unified Agent class.



```python
class ToolCallingAgent:
    def __init__(self, llm):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = tool_map

    def run(self, query: str) -> str:
        # Step 1: Initial user message
        chat_history = [HumanMessage(content=query)]

        # Step 2: LLM chooses tool
        response = self.llm_with_tools.invoke(chat_history)
        if not response.tool_calls:
            return response.contet # Direct response, no tool needed
        # Step 3: Handle first tool call
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # Step 4: Call tool manually
        tool_result = self.tool_map[tool_name].invoke(tool_args)

        # Step 5: Send result back to LLM
        tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
        chat_history.extend([response, tool_message])

        # Step 6: Final LLM result
        final_response = self.llm_with_tools.invoke(chat_history)
        return final_response.content
```

This agent does the exact same process as above except the interaction with the model handling is all contained within the `run()` method.



```python
my_agent = ToolCallingAgent(llm)

print(my_agent.run("one plus 2"))

print(my_agent.run("one - 2"))

print(my_agent.run("three times two"))
```

    One plus two equals three.
    The result of \(1 - 2\) is \(-1\).
    Three times two is 6.


Here are three examples of the agent in use. These agents are dynamic and can handle many different types and formats of data as input. This capability is a major benefit of AI agents as the input data doesn't need to be normalized or formatted a certain way.


## Conclusion

You've now completed this short introduction to building interactive tool calling agents. Now you can:
- Structure user interactions and setup chat models for real-time, context-aware conversations
- Extracte tool names and arguments to precisely match user intent
- Parse complex tool instructions, including handling multiple tool calls
- Build and refine an agent class to automate the entire tool-calling process
- Dempnstrate how these components work together to transform LLMs from passive responders to intelligent agents


## Exercises


### Exercise 1: Create a new tool

Use the example tool format provided in the notebook to create a new tool named `calculate_tip` that takes a `total_bill and tip_percent`, and returns the tip amount. </br>
Define and invoke the tool with sample inputs like `total_bill=120`, `tip_percent=15`. </br>
Create a `tool_map` with the `calculate_tip` tool.



```python
@tool
def calculate_tip(total_bill: int, tip_percent: int) -> int:
    """Calculate tip"""
    return total_bill * tip_percent * 0.01

inputs = {
    "total_bill": 120,
    "tip_percent": 15
}
calculate_tip.invoke(inputs)


tool_map = {
    "calculate_tip": calculate_tip
}
```

<details>
    <summary>Click here for the solution</summary>

```python
@tool
def calculate_tip(total_bill: int, tip_percent: int) -> int:
    """Calculate tip"""
    return total_bill * tip_percent * 0.01

inputs = {
    "total_bill": 120,
    "tip_percent": 15
}
calculate_tip.invoke(inputs)


tool_map = {
    "calculate_tip": calculate_tip
}
```

</details>


### Exercise 2: Tool calling with an LLM

Simulate a user query like "How much should I tip on $60 at 20%?". </br>
Bind the tool to the predefined `llm` and prompt the LLM with the query above. Then parse the LLM response for the tool calling details and invoke the tool accordingly. Finally, take the entire chat history and prompt the LLM for a final output.



```python
# TODO: Exercise 2
query = "How much should I tip on $60 at 20%?"
llm_with_tool = llm.bind_tools([calculate_tip])
chat_history = [HumanMessage(content=query)]

response = llm_with_tool.invoke(chat_history)

tool_calls = response.tool_calls
tool_name = tool_calls[0]["name"]
tool_args = tool_calls[0]["args"]
tool_call_id = tool_calls[0]["id"]

tool_response = tool_map[tool_name].invoke(tool_args)
tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call_id)

chat_history.extend([response, tool_message])

result = llm_with_tool.invoke(chat_history)
print(result.content)
```

    You should tip $12 on a $60 bill at 20%.


### Exercise 3: Create a tip calculating agent

Create an agent to automate the entire process you previously completed.



```python
# TODO: Exercise 3
# query = "How much should I tip on $60 at 20%?"

class TipAgent:
    def __init__(self, llm):
        self.llm_with_tool = llm.bind_tools([calculate_tip])
        self.tool_map = tool_map

    def run(self, query: str) -> str:
        chat_history = [HumanMessage(content=query)]
        response = llm_with_tool.invoke(chat_history)

        tool_calls = response.tool_calls
        tool_name = tool_calls[0]["name"]
        tool_args = tool_calls[0]["args"]
        tool_call_id = tool_calls[0]["id"]
        
        tool_response = tool_map[tool_name].invoke(tool_args)
        tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call_id)
        
        chat_history.extend([response, tool_message])
        
        return llm_with_tool.invoke(chat_history).content

agent = TipAgent(llm)
agent.run("How much should I tip on $60 at 20%?")
```




    'You should tip $12 on a $60 bill at 20%.'



## Authors


[Joshua Zhou](https://author.skills.network/instructors/joshua_zhou)


### Other Contributors


[Kunal Makwana](https://author.skills.network/instructors/kunal_makwana)</br>
[Karan Goswami](https://author.skills.network/instructors/karan_goswami)


## <h3 align="center"> &#169; IBM Corporation. All rights reserved. <h3/>


<!-- ## Changelog

| Date | Version | Changed by | Change Description |
|------|--------|--------|---------|
| 2024-06-06 | 0.1 |  P. Kravitz | ID review and edit. No code edits.Updated the copyright statement. Change log added. Instructional edits only for IBM style. Second person, accessibility, and other minor grammar edits.| -->

