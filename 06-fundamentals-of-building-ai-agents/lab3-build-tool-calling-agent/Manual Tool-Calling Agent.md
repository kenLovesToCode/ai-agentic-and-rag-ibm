<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Build a Tool Calling Agent**


Estimated time needed: **1** hour


In this lab, you'll explore the powerful capabilities of tool calling in large language models (LLMs) to build advanced AI agents that can interact with external systems. You'll learn how to create custom tools that enable an LLM to perform specific actions, from extracting video IDs to fetching YouTube transcripts and metadata. Through hands-on examples, you'll first implement manual tool calling to understand the underlying mechanics, then build a flexible YouTube interaction system that can search videos, extract transcripts, and generate summaries. By the end of this lab, you'll understand how to construct both fixed-sequence and recursive tool-calling chains, allowing your AI assistants to dynamically decide which tools to use and when to use them, creating truly intelligent agents that can reason about and interact with the world around them.


## __Table of Contents__

<ol>
   <li><a href="#Objectives">Objectives</a></li>
   <li>
       <a href="#Setup">Setup</a>
       <ol>
           <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
           <li><a href="#Importing-Required-Libraries">Importing required libraries</a></li>
       </ol>
   </li>
   <li>
       <a href="#Tools">Tools</a>
       <ol>
           <li><a href="#Defining-video-ID-extraction-tool">Defining video ID extraction tool</a></li>
           <li><a href="#Tool-list">Tool list</a></li>
           <li><a href="#Defining-transcript-fetching-tool">Defining transcript fetching tool</a></li>
           <li><a href="#Defining-YouTube-search-tool">Defining YouTube search tool</a></li>
           <li><a href="#Defining-metadata-extraction-tool">Defining metadata extraction tool</a></li>
           <li><a href="#Defining-thumbnail-retrieval-tool">Defining thumbnail retrieval tool</a></li>
       </ol>
   </li>
   <li>
       <a href="#Binding-tools">Binding tools</a>
       <ol>
           <li><a href="#How-the-LLM-calls-a-tool">How the LLM calls a tool</a></li>
           <li><a href="#LangChain-tool-binding-process">LangChain tool binding process</a></li>
           <li><a href="#Extracting-tool-call-information">Extracting tool call information</a></li>
       </ol>
   </li>
   <li>
       <a href="#Automating-the-tool-calling-process">Automating the tool calling process</a>
       <ol>
           <li><a href="#Building-the-summarization-chain">Building the summarization chain</a></li>
       </ol>
   </li>
   <li>
       <a href="#Recursive-chain-flow">Recursive chain flow</a>
       <ol>
           <li><a href="#Defining-the-core-processing-logic">Defining the core processing logic</a></li>
           <li><a href="#Building-the-complete-universal-chain">Building the complete universal chain</a></li>
       </ol>
   </li>
</ol>

<li><a href="#Exercise">Exercise</a></li>


## Objectives

After completing this lab you will be able to:

- Create custom tools that extend the capabilities of language models
- Build both manual and automated tool calling chains
- Implement recursive tool calling for dynamic, multi-step operations
- Develop AI agents that can interact with YouTube's content programmatically
- Apply tool calling techniques to extract, process, and summarize information from external sources
- Design flexible workflows that allow LLMs to reason about when and how to use available tools


----


## Setup


For this lab, you will be using the following libraries:

*   [`pytube`](https://pytube.io/en/latest/) for accessing YouTube videos and their metadata programmatically.
*   [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api) for fetching transcripts from YouTube videos.
*   [`langchain`](https://python.langchain.com/docs/get_started/introduction) for building tool-enabled LLM applications.
*   [`langchain-community`](https://python.langchain.com/docs/integrations/providers/) for additional LangChain integrations.
*   [`langchain-openai`](https://python.langchain.com/docs/integrations/llms/openai) for connecting to OpenAI's language models.
*   [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) for enhanced YouTube data extraction capabilities.


### Installing required libraries



```python
%%capture
%pip install pytube 
%pip install youtube-transcript-api==1.1.0
%pip install langchain-community==0.3.16
%pip install langchain==0.3.23
%pip install langchain-openai==0.3.14
%pip install yt-dlp
```

### Importing required libraries

_It is recommended that you import all required libraries in one place (here):_



```python
import re
from pytube import YouTube
from langchain_core.tools import tool
from IPython.display import display, JSON
import yt_dlp
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
import json

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)
```

Let's initialize the language model that will power your tool calling capabilities. This code sets up a GPT-4o-mini model using the OpenAI provider through LangChain's interface, which you'll use to process queries and decide which tools to call.



```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
```

# API Disclaimer
This lab uses LLMs provided by OpenAI. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to configure your own API keys. Please note that using your own API keys means that you will incur personal charges.

### Running Locally
If you are running this lab locally, you will need to configure your own API key. This lab uses the `init_chat_model` function from `langchain`. To use the model you must set the environment variable `OPENAI_API_KEY` to your OpenAI API key. **DO NOT** run the cell below if you aren't running locally, it will causes errors.



```python
# IGNORE IF YOU ARE NOT RUNNING LOCALLY
os.environ["OPENAI_API_KEY"] = "your OpenAI API key here"
```

# Tools



## Creating custom tools with LangChain

### Anatomy of a tool

Let's provide the basic building blooks a  tool, consider the following tools:

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


## Key components

1. **@tool decorator**
   - Registers the function with LangChain
   - Creates tool attributes (.name, .description, .func)
   - Generates JSON schema for validation
   - Transforms regular functions into callable tools

2. **Function name**
   - Used by LLM to select appropriate tool
   - Used as reference in chains and tool mappings
   - Appears in tool call logs for debugging
   - Should clearly indicate the tool's purpose

3. **Type annotations**
   - Enable automatic input validation
   - Create schema for parameters
   - Allow proper serialization of inputs/outputs
   - Help LLM understand required input formats

4. **Docstring**
   - Provides context for the LLM to decide when to use the tool
   - Documents parameter requirements
   - Explains expected outputs and behavior
   - Critical for tool selection by the LLM

5. **Implementation**
   - Executes the actual operation
   - Handles errors appropriately
   - Returns properly formatted results
   - Should be efficient and robust


### Defining video ID extraction tool

Now you'll define a function `extract_video_id` by denoting it as a tool that will help you to extract the video ID from a given URL. This is necessary because many YouTube API operations, including transcript extraction, require the video ID rather than the complete URL. The function uses regular expressions to handle different YouTube URL formats (standard, shortened, and embedded) and extract the 11-character video ID.



```python
@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    
    # Regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"
```

The decorator wraps your function, adding those attributes (.name, .description, .func) and registering it with LangChain's tool system. The original function becomes accessible through the .func attribute, but the overall object is an instance of LangChain's tool class, with additional methods like .run() for direct invocation.





#### Testing the video ID extraction tool


Now you'll be testing your `extract_video_id` tool to verify that it's correctly registered with LangChain. These print statements will show you:
1. The tool's name (as it will be referenced by the LLM)
2. The tool's description (which helps the LLM understand when to use this tool)
3. The actual function reference that will be called







```python
print(extract_video_id.name)
print("----------------------------")
print(extract_video_id.description)
print("----------------------------")
print(extract_video_id.func)
```

    extract_video_id
    ----------------------------
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.
    
    Returns:
        str: Extracted video ID or error message if parsing fails.
    ----------------------------
    <function extract_video_id at 0x7f0a073aff60>


#### Testing tool execution

Here, you're testing the actual execution of your `extract_video_id` tool with a real YouTube URL. You can call the tool using the `.run()` method, which is a convenient way to execute the tool directly and see its output.



```python
extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A")
```




    'hfIUstzHs9A'




```python
extract_video_id
```




    StructuredTool(name='extract_video_id', description='Extracts the 11-character YouTube video ID from a URL.\n\nArgs:\n    url (str): A YouTube URL containing a video ID.\n\nReturns:\n    str: Extracted video ID or error message if parsing fails.', args_schema=<class 'langchain_core.utils.pydantic.extract_video_id'>, func=<function extract_video_id at 0x7f0a073aff60>)



This output shows that your function has been transformed into a `StructuredTool` object by LangChain. It displays the tool's name ('extract_video_id'), its description (our docstring), a Pydantic schema for input validation, and a reference to your original function.


## Tool list 
Multiple tools will be created to enhance the LLM's capabilities. For organization, create a list called tools, which is a standard Python list that contains tool objects created with the @tool decorator. This list doesn't execute functions or determine call order - it simply collects tool objects in one place so they can be efficiently passed to the language model via llm.bind_tools(tools). This approach allows the LLM to access all available tools without requiring them to be individually registered.

Adding the ```extract_video_id``` tool to your tools list, which you can later provide to the LLM so it can use this functionality when needed.




```python
tools = []
tools.append(extract_video_id)
```

Now that you have understood the basic structure, let's define the rest of the tools you'll need.


### Defining transcript fetching tool

Now you're going to create another tool that fetches the transcript from a YouTube video. This tool uses the `YouTubeTranscriptApi` library to retrieve the captions or subtitles from a video. You'll be taking the video ID (which can be extracted using your previous tool) and an optional language parameter. The function attempts to get the transcript and joins all text segments into a continuous string, or returns an error message if the transcript can't be retrieved.



```python
from youtube_transcript_api import YouTubeTranscriptApi


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"
```

Let's test the fetch_transcript tool by directly calling it with the .run() method on a specific video ID. This will attempt to retrieve the transcript for the video with ID "hfIUstzHs9A" in the default English language.



```python
fetch_transcript.run("hfIUstzHs9A")
```




    'Over the past couple of months, large language models, or LLMs, such as chatGPT, have taken the world by storm. Whether it\'s writing poetry or helping plan your upcoming vacation, we are seeing a step change in the performance of AI and its potential to drive enterprise value. My name is Kate Soule. I\'m a senior manager of business strategy at IBM Research, and today I\'m going to give a brief overview of this new field of AI that\'s emerging and how it can be used in a business setting to drive value. Now, large language models are actually a part of a different class of models called foundation models. Now, the term "foundation models" was actually first coined by a team from Stanford when they saw that the field of AI was converging to a new paradigm. Where before AI applications were being built by training, maybe a library of different AI models, where each AI model was trained on very task-specific data to perform very specific task. They predicted that we were going to start moving to a new paradigm, where we would have a foundational capability, or a foundation model, that would drive all of these same use cases and applications. So the same exact applications that we were envisioning before with conventional AI, and the same model could drive any number of additional applications. The point is that this model could be transferred to any number of tasks. What gives this model the super power to be able to transfer to multiple different tasks and perform multiple different functions is that it\'s been trained on a huge amount, in an unsupervised manner, on unstructured data. And what that means, in the language domain, is basically I\'ll feed a bunch of sentences-- and I\'m talking terabytes of data here --to train this model. And the start of my sentence might be "no use crying over spilled" and the end of my sentence might be "milk". And I\'m trying to get my model to predict the last word of the sentence based off of the words that it saw before. And it\'s this generative capability of the model-- predicting and generating the next word --based off of previous words that it\'s seen beforehand, that is why that foundation models are actually a part of the field of AI called generative AI because we\'re generating something new in this case, the next word in a sentence. And even though these models are trained to perform, at its core, a generation past, predicting the next word in the sentence, we actually can take these models, and if you introduce a small amount of labeled data to the equation, you can tune them to perform traditional NLP tasks-- things like classification, or named-entity recognition --things that you don\'t normally associate as being a generative-based model or capability. And this process is called tuning. Where you can tune your foundation model by introducing a small amount of data, you update the parameters of your model and now perform a very specific natural language task. If you don\'t have data, or have only very few data points, you can still take these foundation models and they actually work very well in low-labeled data domains. And in a process called prompting or prompt engineering, you can apply these models for some of those same exact tasks. So an example of prompting a model to perform a classification task might be you could give a model a sentence and then ask it a question: Does this sentence have a positive sentiment or negative sentiment? The model\'s going to try and finish generating words in that sentence, and the next natural word in that sentence would be the answer to your classification problem, which would respond either positive or negative, depending on where it estimated the sentiment of the sentence would be. And these models work surprisingly well when applied to these new settings and domains. Now, this is a lot of where the advantages of foundation models come into play. So if we talk about the advantages, the chief advantage is the performance. These models have seen so much data. Again, data with a capital D-- terabytes of data --that by the time that they\'re applied to small tasks, they can drastically outperform a model that was only trained on just a few data points. The second advantage of these models are the productivity gains. So just like I said earlier, through prompting or tuning, you need far less label data to get to task-specific model than if you had to start from scratch because your model is taking advantage of all the unlabeled data that it saw in its pre-training when we created this generative task. With these advantages, there are also some disadvantages that are important to keep in mind. And the first of those is the compute cost. So that penalty for having this model see so much data is that they\'re very expensive to train, making it difficult for smaller enterprises to train a foundation model on their own. They\'re also expensive-- by the time they get to a huge size, a couple billion parameters --they\'re also very expensive to run inference. You might require multiple GPUs at a time just to host these models and run inference, making them a more costly method than traditional approaches. The second disadvantage of these models is on the trustworthiness side. So just like data is a huge advantage for these models, they\'ve seen so much unstructured data, it also comes at a cost, especially in the domain like language. A lot of these models are trained basically off of language data that\'s been scraped from the Internet. And there\'s so much data that these models have been trained on. Even if you had a whole team of human annotators, you wouldn\'t be able to go through and actually vet every single data point to make sure that it wasn\'t biased and didn\'t contain hate speech or other toxic information. And that\'s just assuming you actually know what the data is. Often we don\'t even know-- for a lot of these open source models that have been posted --what the exact datasets are that these models have been trained on leading to trustworthiness issues. So IBM recognizes the huge potential of these technologies. But my partners in IBM Research are working on multiple different innovations to try and improve also the efficiency of these models and the trustworthiness and reliability of these models to make them more relevant in a business setting. All of these examples that I\'ve talked through so far have just been on the language side. But the reality is, there are a lot of other domains that foundation models can be applied towards. Famously, we\'ve seen foundation models for vision --looking at models such as DALL-E 2, which takes text data, and that\'s then used to generate a custom image. We\'ve seen models for code with products like Copilot that can help complete code as it\'s being authored. And IBM\'s innovating across all of these domains. So whether it\'s language models that we\'re building into products like Watson Assistant and Watson Discovery, vision models that we\'re building into products like Maximo Visual Inspection, or Ansible code models that we\'re building with our partners at Red Hat under Project Wisdom. We\'re innovating across all of these domains and more. We\'re working on chemistry. So, for example, we just published and released molformer, which is a foundation model to promote molecule discovery or different targeted therapeutics. And we\'re working on models for climate change, building Earth Science Foundation models using geospatial data to improve climate research. I hope you found this video both informative and helpful. If you\'re interested in learning more, particularly how IBM is working to improve some of these disadvantages, making foundation models more trustworthy and more efficient, please take a look at the links below. Thank you.'



---
Adding the `fetch_transcript` tool to your tools list.



```python
tools.append(fetch_transcript)
```

### Defining YouTube search tool

Now let's create a search tool that allows finding videos on YouTube based on a query string. This tool uses the `Search` class from the PyTube library to perform searches on YouTube. When given a search term, it returns a list of matching videos with each video represented as a dictionary containing the title, video ID, and a shortened URL. This tool will be helpful for discovering relevant videos when you don't already have a specific URL in mind.



```python
from pytube import Search
from langchain.tools import tool
from typing import List, Dict

@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query (str): The search term to look for on YouTube
        
    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}"
            }
            for yt in s.results
        ]
    except Exception as e:
        return f"Error: {str(e)}"
```

Now, you'll test your `search_youtube` tool by calling it with the `.run()` method and the search query "Generative AI." This will return a list of YouTube videos related to generative AI.



```python
search_out=search_youtube.run("Generative AI")
display(JSON(search_out))
```


    <IPython.core.display.JSON object>


Appending the `search_youtube` tool to tools list.



```python
tools.append(search_youtube)
```

### Defining metadata extraction tool

Now you'll create a tool that extracts detailed metadata from a YouTube video using the `yt-dlp` library. This tool takes a YouTube URL and returns comprehensive information about the video, including its title, view count, duration, channel name, like count, comment count, and any chapter markers.



```python
@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including title, views, duration, channel, likes, comments, and chapters."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'views': info.get('view_count'),
            'duration': info.get('duration'),
            'channel': info.get('uploader'),
            'likes': info.get('like_count'),
            'comments': info.get('comment_count'),
            'chapters': info.get('chapters', [])
        }
```

Now, you'll test your `get_full_metadata` tool by running it on a specific YouTube video URL. This will extract comprehensive information about the video with ID "qWHaMrR5WHQ" without downloading the actual video content.

**Note: If you find any issues with the given video link below, try any Youtube video link of your choosing.**



```python
meta_data=get_full_metadata.run("https://www.youtube.com/watch?v=T-D1OfcDW1M")
display(JSON(meta_data))
```


    <IPython.core.display.JSON object>


Adding the `get_full_metadata` tool to your tools list.



```python
tools.append(get_full_metadata)
```

### Defining thumbnail retrieval tool

Now you'll create a tool to extract all available thumbnail images for a YouTube video. This tool uses `yt-dlp` to retrieve information about the various thumbnail images that YouTube generates for videos at different resolutions. For each thumbnail, collect its URL, width, height, and formatted resolution.



```python
@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]
```

Now, you'll test your `get_thumbnails` tool by running it on a specific YouTube video URL. This will extract information about all available thumbnail images for the video.



```python
thumbnails=get_thumbnails.run("https://www.youtube.com/watch?v=qWHaMrR5WHQ")

display(JSON(thumbnails))
```


    <IPython.core.display.JSON object>


Now, let's add the `get_thumbnails` tool to your tools list.



```python
tools.append(get_thumbnails)
```

##  Binding tools




Now, you'll bind your collection of tools to the language model. It enables the LLM to access and use your custom YouTube tools during conversations. By binding the tools, you're giving the model the ability to call these functions when it determines they're needed to fulfill a user request, making the LLM aware of your tools' capabilities and how to use them.



```python
llm_with_tools = llm.bind_tools(tools)
```

The ```bind_tools()``` function passes all this information to the language model. It converts each tool's attributes (name, description, parameters schema) into a standardized format that the LLM can understand and use to determine when and how to call specific tools based on user requests. Similar to the following code where the schema for each tool is stored:



```python
for tool in tools:
    schema = {
   "name": tool.name,
   "description": tool.description,
   "parameters": tool.args_schema.schema() if tool.args_schema else {},
   "return": tool.return_type if hasattr(tool, "return_type") else None}
    display(JSON(schema))
    
```


    <IPython.core.display.JSON object>



    <IPython.core.display.JSON object>



    <IPython.core.display.JSON object>



    <IPython.core.display.JSON object>



    <IPython.core.display.JSON object>


### How the LLM calls a tool

Now, define a sample user query that asks for a summary of a specific YouTube video. This query will be used to demonstrate how your LLM can understand a natural language request and use the appropriate tools you've provided to fulfill it.



```python
query = "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"
print(query)
```

    I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english


Repeating a message object to represent your user query. You'll be wrapping the query string in a HumanMessage object, which is the standard way to format user inputs in LangChain. It represents a human message as a person is expected to initiate the interaction.



```python
messages = [HumanMessage(content = query)]
print(messages)
```

    [HumanMessage(content='I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english', additional_kwargs={}, response_metadata={})]


### LangChain tool binding process

This step involves sending your message to the LLM and storing its response. Here you'll invoke the language model with your user query about summarizing a YouTube video. The response will contain both text content and potentially tool calls that the model decides to make. ``response_1`` contains the LLM's response to the user message, including any tool calls it decides to make. The response object contains the content of the LLM's reply plus structured information about which tools it wants to call and with what parameters.



```python
response_1 = llm_with_tools.invoke(messages)
response_1
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_nKrNe9fwH0RceEcPyafijjN3', 'function': {'arguments': '{"url":"https://www.youtube.com/watch?v=T-D1OfcDW1M"}', 'name': 'extract_video_id'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 380, 'total_tokens': 409, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_60831fdee3', 'id': 'chatcmpl-DSDMPKqvX944ZoCK4AWrCvkRujRQ1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--019d6b03-e4ae-7c43-b211-074024c1b7b6-0', tool_calls=[{'name': 'extract_video_id', 'args': {'url': 'https://www.youtube.com/watch?v=T-D1OfcDW1M'}, 'id': 'call_nKrNe9fwH0RceEcPyafijjN3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 380, 'output_tokens': 29, 'total_tokens': 409, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



Adding the LLM's response to your conversation history. After receiving the response from the language model (which contains the tool call to extract the video ID), append it to your messages list to maintain the conversation context. This builds up the chat history that will be used for subsequent interactions with the model.




```python
messages.append(response_1)
```

### Extracting tool call information
After receiving the LLM's response, you need to extract the structured tool call information. The line tool_calls_1 = response_1.tool_calls gets the tool call objects that contain which tool the LLM has decided to use and what parameters to pass to it. This information will be used to execute the appropriate tool with the correct inputs.

#### Creating a tool mapping dictionary

Now you'll create a dictionary that maps tool names to their corresponding function objects. This mapping will be useful later when you need to programmatically invoke specific tools based on their names. It allows you to easily look up and execute a tool function when you have only the tool name as a string, which will be important when processing tool calls from the language model.



```python
tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}
```

Extracting the tool calls from the language model's response. When the LLM determines it needs to use one of your tools, it includes structured "tool_calls" in its response. Here, you're accessing those tool calls to see which tools the model decided to use in order to fulfill the request about summarizing the YouTube video.



```python
tool_calls_1 = response_1.tool_calls
display(JSON(tool_calls_1))
```


    <IPython.core.display.JSON object>


Here you're seeing the structure of the tool call that the LLM decided to make. The tool call is formatted as a dictionary with the following key components:

1. `name`: 'extract_video_id' - This identifies which tool the LLM wants to use first (the video ID extraction tool)
2. `args`: Contains the arguments to pass to the tool - in this case, the YouTube URL from your query
3. `id`: A unique identifier for this specific tool call, which helps track the request/response pair
4. `type`: Indicates this is a tool call rather than other types of AI responses

This shows that the LLM correctly understood it needs to first extract the video ID from the URL before it can proceed with summarizing the video content.


Accessing the name of the first tool that the LLM decided to use. Here you're extracting just the name component `('extract_video_id')` from the first tool call in the list.



```python
tool_name=tool_calls_1[0]['name']
print(tool_name)
```

    extract_video_id


You need a tool ID to help the LLM know where the output came from:



```python
tool_call_id =tool_calls_1[0]['id']
print(tool_call_id)
```

    call_nKrNe9fwH0RceEcPyafijjN3


Accessing the arguments that need to be passed to the chosen tool. Here, you're extracting the arguments component from the first tool call, which contains the YouTube URL that needs to be processed.



```python
args=tool_calls_1[0]['args']
print(args)
```

    {'url': 'https://www.youtube.com/watch?v=T-D1OfcDW1M'}


Adding the LLM's response to your conversation history. After receiving the response from the language model (which contains the tool call to extract the video ID), you append it to your messages list to maintain the conversation context. This builds up the chat history that will be used for subsequent interactions with the model.


Executing the tool call that the LLM requested. Here, you're using your tool mapping dictionary to:
1. Look up the appropriate function based on the tool name ('extract_video_id')
2. Call that function with the arguments provided by the LLM
3. Capture the output (the extracted video ID)

This shows how you can programmatically execute the tools that the LLM decided to use. First, you get the tool from ```tool_mapping```.



```python
my_tool=tool_mapping[tool_calls_1[0]['name']]
```

You'll then call the tool with the arguments:



```python
video_id =my_tool.invoke(tool_calls_1[0]['args'])
video_id
```




    'T-D1OfcDW1M'



Adding the tool's output to your conversation history. You'll create a `ToolMessage` that contains:
1. The result from executing the tool (the extracted video ID)
2. The original tool call ID to link this response back to the specific request

By appending this message to your conversation history, you're informing the LLM about the results of the tool execution, which it can use in its next response.



```python
messages.append(ToolMessage(content = video_id, tool_call_id = tool_calls_1[0]['id']))
```

Send your updated conversation to the LLM and store its new response. Now that you've informed the model about the extracted video ID, invoke it again to continue the process. The model will see both the original query and the result of the video ID extraction, allowing it to determine the next step needed to summarize the YouTube video.



```python
response_2 = llm_with_tools.invoke(messages)
response_2
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AWygrUVE3uEk8KDaJ2fihHgx', 'function': {'arguments': '{"video_id":"T-D1OfcDW1M","language":"en"}', 'name': 'fetch_transcript'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 426, 'total_tokens': 453, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_60831fdee3', 'id': 'chatcmpl-DSDMgPPlf14iaSBxLHb9mynMxFeHa', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--019d6b04-307c-7dc1-b290-cd29074de687-0', tool_calls=[{'name': 'fetch_transcript', 'args': {'video_id': 'T-D1OfcDW1M', 'language': 'en'}, 'id': 'call_AWygrUVE3uEk8KDaJ2fihHgx', 'type': 'tool_call'}], usage_metadata={'input_tokens': 426, 'output_tokens': 27, 'total_tokens': 453, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



The result is a AI messege! Send your updated conversation to the LLM and store its new response. Now that you've informed the model about the extracted video ID, you'll invoke it again to continue the process. The model will see both the original query and the result of the video ID extraction, allowing it to determine the next step needed to summarize the YouTube video.



```python
messages.append(response_2)
```

Extracting the tool calls from the language model's second response. After receiving the video ID, the LLM will likely decide to use another tool to help with the summarization task. 



```python
tool_calls_2 = response_2.tool_calls
tool_calls_2
```




    [{'name': 'fetch_transcript',
      'args': {'video_id': 'T-D1OfcDW1M', 'language': 'en'},
      'id': 'call_AWygrUVE3uEk8KDaJ2fihHgx',
      'type': 'tool_call'}]



Here, you can see that the LLM has decided to use the `fetch_transcript` tool as its next step. 

The model is passing two arguments to the transcript fetching tool:
1. `video_id`: 'T-D1OfcDW1M' - The ID that was extracted from the original YouTube URL
2. `language`: 'en' - Requesting the transcript in English as specified in the user's query


---
Fetching the transcript using the video ID obtained in the previous step. Here, you're executing the second tool that the LLM requested by:
1. Looking up the appropriate function `('fetch_transcript')` from your tool mapping
2. Invoking it with the video ID and language parameters
3. Storing the resulting transcript content



```python
fetch_transcript_tool_output = tool_mapping[tool_calls_2[0]['name']].invoke(tool_calls_2[0]['args'])
fetch_transcript_tool_output
```




    'Large language models. They are everywhere. They get some things amazingly right and other things very interestingly wrong. My name\xa0is Marina Danilevsky. I am a Senior Research Scientist here at IBM Research. And I want\xa0to tell you about a framework to help large language models be more accurate and more up to\xa0date: Retrieval-Augmented Generation, or RAG. Let\'s just talk about the "Generation" part for a\xa0minute. So forget the "Retrieval-Augmented". So the\xa0generation, this refers to large language models,\xa0or LLMs, that generate text in response to a user query, referred to as a prompt. These\xa0models can have some undesirable behavior. I want to tell you an anecdote to illustrate this. So my kids, they recently asked me this question: "In our solar system, what planet has the most\xa0moons?" And my response was, “Oh, that\'s really great that you\'re asking this question. I loved\xa0space when I was your age.” Of course, that was like 30 years ago. But I know this! I read an\xa0article and the article said that it was Jupiter and 88 moons. So that\'s the answer. Now, actually,\xa0there\'s a couple of things wrong with my answer. First of all, I have no source to support what\xa0I\'m saying. So even though I confidently said “I read an article, I know the answer!”, I\'m not\xa0sourcing it. I\'m giving the answer off the top of my head. And also, I actually haven\'t kept up with\xa0this for awhile, and my answer is out of date. So we have two problems here. One is no source.\xa0And the second problem is that I am out of date.\xa0\xa0 And these, in fact, are two behaviors that are\xa0often observed as problematic when interacting with large language models. They’re LLM\xa0challenges. Now, what would have happened if I\'d taken a beat and first gone and looked\xa0up the answer on a reputable source like NASA? Well, then I would have been able to say, “Ah,\xa0okay! So the answer is Saturn with 146 moons.” And in fact, this keeps changing because scientists\xa0keep on discovering more and more moons. So I have now grounded my answer in something more\xa0\nbelievable. I have not hallucinated or made up an answer. Oh, by the way, I didn\'t leak personal\xa0information about how long ago it\'s been since I was obsessed with space. All right, so what does\xa0this have to do with large language models? Well, how would a large language model have answered\xa0this question? So let\'s say that I have a user asking this question about moons. A large language\xa0model would confidently say, OK, I have been trained and from what I know in my parameters\xa0during my training, the answer is Jupiter. The answer is wrong. But, you know, we don\'t know. The large language model is very confident in what it answered. Now, what happens when you add this\xa0retrieval augmented part here? What does that mean? That means that now, instead of just relying\xa0on what the LLM knows, we are adding a content store. This could be open like the internet. This\xa0can be closed like some collection of documents, collection of policies, whatever. The point,\xa0though, now is that the LLM first goes and talks to the content store and says,\xa0“Hey, can you retrieve for me information that is relevant to what the user\'s\xa0query was?” And now, with this retrieval-augmented answer, it\'s not Jupiter anymore. We know that\xa0it is Saturn. What does this look like? Well, first user prompts the LLM\xa0with their question. They say, this is what my question was. And originally,\xa0if we\'re just talking to a generative model, the generative model says, “Oh, okay, I know\xa0the response. Here it is. Here\'s my response.”\xa0\xa0 But now in the RAG framework, the generative\xa0model actually has an instruction that says, "No, no, no." "First, go and retrieve\xa0relevant content." "Combine that with the user\'s question and only then generate the\xa0answer." So the prompt now has three parts: the instruction to pay attention to, the retrieved\xa0content, together with the user\'s question. Now give a response. And in fact, now you can give\xa0evidence for why your response was what it was.\xa0\xa0 So now hopefully you can see, how does RAG help the two LLM challenges that I had mentioned before?\xa0\xa0 So first of all, I\'ll start with the out of\xa0date part. Now, instead of having to retrain your model, if new information comes up, like, hey,\xa0we found some more moons-- now to Jupiter again, maybe it\'ll be Saturn again in the future. All\xa0you have to do is you augment your data store with new information, update information. So now the next time that a user comes and asks the question, we\'re ready. We just go ahead and retrieve the most up to date information. The second problem, source. Well, the large language model is now being instructed to pay attention to primary source data before giving its response. And in fact, now being able to give evidence. This makes it less likely to hallucinate or to leak data because it is less likely to rely only on information that it learned during training. It also allows us to get the model to have a behavior that can be very positive, which is knowing when to say, “I don\'t know.” If\xa0the user\'s question cannot be reliably answered based on your data store, the model should say,\xa0"I don\'t know," instead of making up something that is believable and may mislead the user. This\xa0can have a negative effect as well though, because if the retriever is not sufficiently\xa0good to give the large language model the best, most high-quality grounding information, then\xa0maybe the user\'s query that is answerable doesn\'t get an answer. So this is actually why lots\xa0of folks, including many of us here at IBM, are working the problem on both sides. We are both\xa0working to improve the retriever to give the large language model the best quality data on which\xa0to ground its response, and also the generative part so that the LLM can give the richest, best\xa0response finally to the user when it generates the answer. Thank you for learning more about RAG\xa0and like and subscribe to the channel. Thank you.'



---
You're adding the transcript content to your conversation history by creating another `ToolMessage` that contains the transcript text and the ID of the tool call that requested it. This gives the LLM access to the actual video content so it can generate a summary.



```python
messages.append(ToolMessage(content = fetch_transcript_tool_output, tool_call_id = tool_calls_2[0]['id']))
```

Generating the final summary by sending your complete conversation history to the LLM. Now that the model has access to both the video ID and the full transcript, you'll invoke it one more time to generate the summary that the user requested.



```python
summary = llm_with_tools.invoke(messages)
```


```python
summary
```




    AIMessage(content="The video features Marina Danilevsky, a Senior Research Scientist at IBM Research, discussing a framework called Retrieval-Augmented Generation (RAG) that aims to enhance the accuracy and up-to-dateness of large language models (LLMs).\n\n### Key Points:\n\n1. **Introduction to LLMs:** \n   - LLMs generate text in response to user queries but can sometimes provide incorrect or outdated answers.\n\n2. **Illustration with an Anecdote:**\n   - Marina shares a personal story about how she answered her children's question regarding which planet has the most moons, initially stating Jupiter based on outdated knowledge.\n\n3. **Problems Identified:**\n   - **Lack of Source:** Her original answer lacked a credible citation.\n   - **Out-of-Date Information:** Her knowledge was outdated, highlighting a common issue with LLMs.\n\n4. **Retrieval-Augmented Approach:**\n   - Instead of solely relying on their training data, LLMs can access a content store (like the internet) to retrieve relevant, up-to-date information before generating answers.\n   - This framework allows the model to prepend the retrieval process, ensuring the accuracy and relevance of its responses.\n\n5. **Benefits of RAG:**\n   - **Up-to-Date Information:** Models can access new data without needing retraining, ensuring answers reflect the latest information.\n   - **Credible Sources:** LLMs are prompted to refer to primary source data, decreasing hallucinations and inaccuracies.\n   - **Honesty in Responses:** The LLM can confidently state when it doesn't know the answer, avoiding speculation.\n\n6. **Challenges:**\n   - If the retrieval mechanism fails to provide high-quality information, it may undermine the model’s ability to give accurate answers.\n\n7. **Ongoing Efforts:**\n   - Danilevsky mentions that many researchers at IBM are working on improving both the retrieval process and the generative capabilities of LLMs to enhance user experience.\n\nThe video concludes with a call to like and subscribe for more content related to RAG.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 415, 'prompt_tokens': 1812, 'total_tokens': 2227, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_60831fdee3', 'id': 'chatcmpl-DSDMrDSGdURZmsYwsOzIxuAIdh6Vs', 'finish_reason': 'stop', 'logprobs': None}, id='run--019d6b04-5c71-78a3-88ef-9690d2509e17-0', usage_metadata={'input_tokens': 1812, 'output_tokens': 415, 'total_tokens': 2227, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



### Automating the tool calling process

You manually saw how you input a text request to your LLM, where the LLM recognized that a tool call was required. Then, you extracted the tool content, formatted the input, made the next tool call, and repeated these steps. While this step-by-step approach helps understand the process, it would be tedious to implement for every application. Now let's automate this entire workflow.

#### Extracting tool information from LLM response
Create a function to automate tool calling. The input is the tool call object from which you extract the name, and use the tool_mapping dictionary to find the correct function to call. You'll pass the arguments from the tool call to this function and then send the output back as a ToolMessage with the tool_call_id included.
The tool_call_id is an essential part of this process as it links each tool response back to the specific tool request made by the language model. This ID ensures the LLM can match responses to its requests, which is crucial when multiple tools are called in sequence or simultaneously. Without this ID, the LLM would have no way to know which response corresponds to which request, making multi-step reasoning impossible.



```python
# Define the processing steps
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id=tool_call["id"]
        )

        
```

You are now going to chain all your functions or tools together, but before you do so, you need to format the data properly. Not only are you required to store the output of each tool, but you also need to store state information like tool IDs. To do this effectively, you must ensure the output of each tool can be properly passed to the next step in your pipeline. The RunnablePassthrough component allows you to maintain state throughout the chain while adding or transforming data at each step, making it ideal for connecting your various tools into a cohesive workflow.
The RunnableLambda, placed at the end of your chain, serves a different purpose - it extracts only the final result you want to present to the user. After all the tool calls and message processing, you have a rich state object with many fields, but the user typically only needs the final answer. The RunnableLambda transforms this complete state into just the information you want to return.



```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

```

## Building the summarization chain

Now, you'll combine your functions into a complete `summarization_chain` using the pipe operator `|`, which applies functions sequentially (similar to function composition where `f|g(x)` is equivalent to `f(g(x))`).

The workflow follows these steps:
1. Convert the input prompt to a HumanMessage
2. Pass the message to LLM with tools
3. Extract tool calls from LLM response
4. Update message history with tool results
5. Send updated messages back to LLM
6. Repeat steps 3-5 as needed
7. Finally, extract just the content from the final message using RunnableLambda

Each step maintains state using RunnablePassthrough until you reach the final message, at which point you'll apply RunnableLambda to extract only the summary text.



```python
summarization_chain = (
    # Start with initial query
    RunnablePassthrough.assign(
        messages=lambda x: [HumanMessage(content=x["query"])]
    )
    # First LLM call (extract video ID)
    | RunnablePassthrough.assign(
        ai_response=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process first tool call
    | RunnablePassthrough.assign(
        tool_messages=lambda x: [
            execute_tool(tc) for tc in x["ai_response"].tool_calls
        ]
    )
    # Update message history
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
    # Second LLM call (fetch transcript)
    | RunnablePassthrough.assign(
        ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process second tool call
    | RunnablePassthrough.assign(
        tool_messages2=lambda x: [
            execute_tool(tc) for tc in x["ai_response2"].tool_calls
        ]
    )
    # Final message update
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
    )
    # Generate final summary
    | RunnablePassthrough.assign(
        summary=lambda x: llm_with_tools.invoke(x["messages"]).content
    )
    # Return just the summary text
    | RunnableLambda(lambda x: x["summary"])
)

```

Here's how you invoke the summarization chain with a YouTube video URL; this passes your query containing a YouTube URL to the chain, which automatically extracts the video ID, fetches the transcript, and generates a summary of the content.



```python
# Usage
result = summarization_chain.invoke({
    "query": "Summarize this YouTube video: https://www.youtube.com/watch?v=1bUy-1hGZpI"
})

print("Video Summary:\n", result)
```

    Video Summary:
     The video discusses LangChain, an open-source orchestration framework for developing applications that utilize large language models (LLMs). It provides a generic interface for various LLMs, allowing users to integrate them into business applications efficiently. Key components include:
    
    1. **Abstractions**: Simplifying programming by allowing developers to create complex tasks with minimal code.
    2. **LLM Module**: Supports various models via API, enabling users to choose from closed-source or open-source options.
    3. **Prompt Templates**: Facilitate the creation of instructions for LLMs without hardcoding details.
    4. **Chains**: Core to LangChain workflows, allowing sequential execution of functions, each potentially using different models and prompts.
    5. **Indexes and Document Loaders**: These handle external data sources, making integration with different applications seamless.
    6. **Memory Management**: Helps retain conversation history for improved interaction with users.
    7. **Agents**: Use LLMs to reason and determine actions based on user input and available tools.
    
    Real-world use cases include chatbots, summarization tools, question answering systems, data augmentation for machine learning, and virtual agents leveraging robotic process automation (RPA).
    
    Overall, LangChain streamlines the development process, making it easier to create applications that leverage the power of LLMs.


---
Up to this point, you've demonstrated how to manually orchestrate the tool calling process step by step. You first invoked the LLM with the user's query, interpreted its decision to use the `extract_video_id` tool, executed that tool, fed the result back to the LLM, processed its next decision to use the `fetch_transcript` tool, executed that tool, and finally had the LLM generate a summary based on the transcript.

Now you'll see how to accomplish the same workflow more efficiently using LangChain's chain functionality, which automates this back-and-forth process of tool selection, execution, and response handling.


#### Creating the initial message setup

Here you're setting up the first step of your chain that will handle the initial user query. The `RunnablePassthrough.assign` creates a component that takes an input dictionary containing a "query" and converts it into a list containing a single `HumanMessage` object.



```python
initial_setup = RunnablePassthrough.assign(
    messages=lambda x: [HumanMessage(content=x["query"])]
)
```

#### Defining the first LLM interaction

Here, you'll create the second step of your chain, which handles the first interaction with the language model. This component takes the formatted messages from the previous step, sends them to your tool-equipped LLM, and captures the response in a field called "ai_response."



```python
first_llm_call = RunnablePassthrough.assign(
    ai_response=lambda x: llm_with_tools.invoke(x["messages"])
)
```

#### Processing the first tool call

Here, you're defining the processing step that handles the LLM's first tool call. This component:
1. Executes each tool call by passing it to your `execute_tool` function, which runs the appropriate tool and returns the result as a `ToolMessage`
2. Updates the message history by combining the original messages, the LLM's response, with the tool calls, and the tool results
3. Prepares the updated conversation state for the next interaction with the LLM



```python
first_tool_processing = RunnablePassthrough.assign(
    tool_messages=lambda x: [
        execute_tool(tc) for tc in x["ai_response"].tool_calls
    ]
).assign(
    messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
)
```

#### Defining the second LLM interaction

Here, you're creating the next step in your chain that handles the second interaction with the language model. This component takes the updated message history (which now includes the results from the first tool call) and sends it to the LLM again.



```python
second_llm_call = RunnablePassthrough.assign(
    ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
)
```

#### Processing the second tool call

Here, you're defining the processing step that handles the LLM's second tool call. Similar to the first tool processing step, this component executes the tool calls (typically fetching the transcript), creates tool messages with the results, and updates the message history by combining everything for the final summarization step.



```python
second_tool_processing = RunnablePassthrough.assign(
    tool_messages2=lambda x: [
        execute_tool(tc) for tc in x["ai_response2"].tool_calls
    ]
).assign(
    messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
)
```

#### Generating the final summary

Here, you're defining the final step that produces the summary of the YouTube video. This component:
1. Takes the complete message history (which now contains the original query, tool calls, and tool results)
2. Invokes the LLM one last time to generate a summary
3. Extracts just the content field from the LLM's response
4. Uses a RunnableLambda to return only the summary text as the final output



```python
final_summary = RunnablePassthrough.assign(
    summary=lambda x: llm_with_tools.invoke(x["messages"]).content
) | RunnableLambda(lambda x: x["summary"])
```

#### Assembling the complete chain

Now, you're combining all the individual components you've defined into a single cohesive chain. By piping each step to the next, you'll create a workflow that:
1. Formats the initial query
2. Gets the first LLM response (video ID extraction)
3. Processes the first tool call
4. Gets the second LLM response (transcript request)
5. Processes the second tool call
6. Generates the final summary



```python
chain = (
    initial_setup
    | first_llm_call
    | first_tool_processing
    | second_llm_call
    | second_tool_processing
    | final_summary
)
```

Now, you're testing your automated chain with the original video summarization query you handled manually before. By passing in the same query to your chain, you can confirm that it produces the same results but in a much more streamlined manner.



```python
query = {"query": "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"}
result = summarization_chain.invoke(query)
print("Video Summary:\n", result)
```

    Video Summary:
     The video features Marina Danilevsky, a Senior Research Scientist at IBM Research, discussing a framework called Retrieval-Augmented Generation (RAG) to enhance the accuracy and timeliness of large language models (LLMs).
    
    **Key Points from the Video:**
    
    1. **Issues with LLMs:** Large language models often produce confident but incorrect answers. For example, Danilevsky shares an anecdote about answering her children's question regarding which planet has the most moons. She incorrectly stated Jupiter, failing to provide a credible source, showcasing the challenges LLMs face.
    
    2. **Retrieval-Augmented Generation (RAG):** RAG addresses two major problems:
       - **Outdated Information:** Traditional LLMs rely solely on training data, which can become outdated. RAG allows models to retrieve the most current information from a content store (like the internet).
       - **Lack of Sources:** By incorporating a retrieval mechanism, the LLM can access credible sources before generating responses, reducing the chances of hallucinations or incorrect answers.
    
    3. **How RAG Works:** When a user prompts the model, RAG directs it to first retrieve relevant information from a content store before combining this with the user's question to create an informed response. This method allows models to provide evidence for their answers, improving reliability.
    
    4. **Improved Behaviors:** RAG enables models to acknowledge when they do not know an answer, enhancing the overall user experience. However, the effectiveness of this framework depends on the quality of the retrieved information.
    
    5. **Ongoing Improvements:** IBM is actively working to refine both the retrieval process and generative capabilities to ensure that models provide high-quality and well-grounded responses.
    
    Danilevsky invites viewers to engage with the content and subscribe to the channel for more insights on RAG and LLMs.


#### Testing the Chain with a Different Query

Here, you're testing your completed chain with a new query to demonstrate its flexibility. Instead of requesting a video summary, you're asking for information about trending videos in India. You'll create a dictionary with the query and invoke your chain, which will handle all the necessary tool calls automatically.



```python
query = {"query": "Get top 3 youtube videos in Philippines and their metadata"}
try:
    result = summarization_chain.invoke(query)
    print("Video Summary:\n", result)
except Exception as e:
    print("Non-critical network error:", e)
```

    Video Summary:
     Here are the top 3 YouTube videos in the Philippines along with their metadata:
    
    1. **Title:** [WONDERS OF THE PHILIPPINES | The Most Amazing Places in the Philippines | 4K Travel Guide](https://youtu.be/ZaDrcVh1l4E)
       - **Views:** 1,718,852
       - **Duration:** 48 minutes and 23 seconds
       - **Channel:** EpicExplorationsTV EN
       - **Likes:** 9,889
       - **Comments:** 369
       - **Chapters:** 
         - Introduction (0:00 - 0:40)
         - About Philippines (0:40 - 5:01)
         - El Nido (5:01 - 6:53)
         - Banaue Rice Terraces (6:53 - 8:33)
         - ... (and more chapters detailing various attractions).
    
    2. **Title:** [Wonders of The Philippines | The Most Amazing Places in The Philippines | Travel Video 4K](https://youtu.be/wD1w5WShII0)
       - **Views:** 3,874,552
       - **Duration:** 1 hour and 7 seconds
       - **Channel:** Top Travel
       - **Likes:** 23,103
       - **Comments:** 1,100
       - **Chapters:** 
         - Intro (0:00 - 1:52)
         - Fun facts about the Philippines (1:52 - 6:07)
         - Samar (6:07 - 10:03)
         - Leyte (10:03 - 15:08)
         - ... (and more chapters detailing different regions).
    
    3. **Title:** [MORE FUN AWAITS IN THE PHILIPPINES](https://youtu.be/eSUmkFPln_U)
       - **Views:** 327,623
       - **Duration:** 1 minute and 30 seconds
       - **Channel:** Tourism Philippines
       - **Likes:** 4,643
       - **Comments:** 142
       - **Chapters:** None available.
    
    These videos highlight the beauty and attractions of the Philippines, showcasing various tourist spots and travel information.



```python
result
```




    'Here are the top 3 YouTube videos in the Philippines along with their metadata:\n\n1. **Title:** [WONDERS OF THE PHILIPPINES | The Most Amazing Places in the Philippines | 4K Travel Guide](https://youtu.be/ZaDrcVh1l4E)\n   - **Views:** 1,718,852\n   - **Duration:** 48 minutes and 23 seconds\n   - **Channel:** EpicExplorationsTV EN\n   - **Likes:** 9,889\n   - **Comments:** 369\n   - **Chapters:** \n     - Introduction (0:00 - 0:40)\n     - About Philippines (0:40 - 5:01)\n     - El Nido (5:01 - 6:53)\n     - Banaue Rice Terraces (6:53 - 8:33)\n     - ... (and more chapters detailing various attractions).\n\n2. **Title:** [Wonders of The Philippines | The Most Amazing Places in The Philippines | Travel Video 4K](https://youtu.be/wD1w5WShII0)\n   - **Views:** 3,874,552\n   - **Duration:** 1 hour and 7 seconds\n   - **Channel:** Top Travel\n   - **Likes:** 23,103\n   - **Comments:** 1,100\n   - **Chapters:** \n     - Intro (0:00 - 1:52)\n     - Fun facts about the Philippines (1:52 - 6:07)\n     - Samar (6:07 - 10:03)\n     - Leyte (10:03 - 15:08)\n     - ... (and more chapters detailing different regions).\n\n3. **Title:** [MORE FUN AWAITS IN THE PHILIPPINES](https://youtu.be/eSUmkFPln_U)\n   - **Views:** 327,623\n   - **Duration:** 1 minute and 30 seconds\n   - **Channel:** Tourism Philippines\n   - **Likes:** 4,643\n   - **Comments:** 142\n   - **Chapters:** None available.\n\nThese videos highlight the beauty and attractions of the Philippines, showcasing various tourist spots and travel information.'



## Recursive chain flow


Now that you've created a chain that works well for your specific two-step tool calling process, you need to consider more complex scenarios. Your current chain is limited to exactly two tool calls in a fixed sequence. In real-world applications, you might need a variable number of tool calls depending on the user's query - for example searching for videos on a topic and then getting transcripts for multiple results.

To handle these more complex scenarios, you'll build a recursive chain that can dynamically decide how many tool calls are needed and continue processing until all necessary information has been gathered.



```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage
import json

def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        content = f"Error: {str(e)}"
    
    return ToolMessage(
        content=content,
        tool_call_id=tool_call["id"]
    )
```

#### Defining the core processing logic

This function handles the core processing logic of your recursive chain. It takes the current conversation history and:

1. Identifies the most recent message in the conversation
2. Extracts all tool calls from that message and executes them in parallel using your `execute_tool` helper
3. Updates the message history by adding the tool response messages
4. Gets the next response from the language model based on the updated conversation
5. Returns the complete updated message history with both tool responses and the new LLM response



```python
def process_tool_calls(messages):
    """Recursive tool call processor"""
    last_message = messages[-1]
    
    # Execute all tool calls in parallel
    tool_messages = [
        execute_tool(tc) 
        for tc in getattr(last_message, 'tool_calls', [])
    ]
    
    # Add tool responses to message history
    updated_messages = messages + tool_messages
    
    # Get next LLM response
    next_ai_response = llm_with_tools.invoke(updated_messages)
    
    return updated_messages + [next_ai_response]
```

#### Creating the recursive stopping condition

This function determines whether your recursive process should continue or terminate. It:

1. Takes the current message history and examines the last message
2. Checks if that message contains any tool calls using the `getattr` function (which safely handles cases where the attribute might not exist)
3. Returns a boolean value - `True` if there are more tool calls to process, and `False` when you reach a point where the LLM has provided a final answer without requesting additional tools



```python
def should_continue(messages):
    """Check if you need another iteration"""
    last_message = messages[-1]
    return bool(getattr(last_message, 'tool_calls', None))
```


#### Implementing the recursive function

This function implements the actual recursion that powers your dynamic tool calling process:

1. It first checks the stopping condition using the `should_continue` function to determine if more tools need to be called
2. If more tool calls are needed, it processes those calls using your `process_tool_calls` function and then recursively calls itself with the updated messages
3. If no more tool calls are needed, it returns the final message history, which contains the complete conversation, including the LLM's final response

After defining this recursive function, you'll wrap it in a `RunnableLambda` to make it compatible with LangChain's chain architecture.



```python
def _recursive_chain(messages):
    """Recursively process tool calls until completion"""
    if should_continue(messages):
        new_messages = process_tool_calls(messages)
        return _recursive_chain(new_messages)
    return messages

recursive_chain = RunnableLambda(_recursive_chain)
```

#### Building the complete universal chain

Now, you're assembling your final universal chain that can handle any type of query requiring any number of tool calls. This chain consists of three main steps:

1. The first step converts the user query into a properly formatted `HumanMessage` object
2. The second step sends this initial message to your tool-equipped LLM and adds the LLM's first response to the message history
3. The final step passes the conversation to your recursive chain, which will handle all subsequent tool calls until the LLM provides a final answer

This universal chain is much more flexible than your earlier fixed-step chain, as it can dynamically adapt to queries that require different numbers and types of tool calls.



```python
universal_chain = (
    RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
    | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
    | recursive_chain
)
```

**Note: If you find any issues with the given video link below, try any Youtube video link of your choosing.**



```python
query_us = {"query": "Show top 3 US trending videos with metadata and thumbnails"}

try:
    response = universal_chain.invoke(query_us)
    print("\nUS Trending Videos:\n", response[-1])
except Exception as e:
    print("Non-critical network error while fetching US trending videos:", e)
```

    
    US Trending Videos:
     content='Here are the top 3 trending videos in the US along with their metadata and thumbnails:\n\n### 1. **5 Most Profitable Niches on YouTube in 2025**\n- **Channel**: Josh Butcher\n- **Views**: 724,005\n- **Likes**: 18,981\n- **Comments**: 329\n- **Duration**: 52 seconds\n- **[Watch Video](https://youtu.be/Ufqc0lU-i84)**\n- **Thumbnails**:\n  - ![Thumbnail 1](https://i.ytimg.com/vi/Ufqc0lU-i84/3.jpg)\n  - ![Thumbnail 2](https://i.ytimg.com/vi/Ufqc0lU-i84/default.jpg)\n\n---\n\n### 2. **USA 3 Viral Faceless Channel 🇺🇸📈 #shorts**\n- **Channel**: Pocket Ai\n- **Views**: 728,588\n- **Likes**: Not available\n- **Comments**: 187\n- **Duration**: 49 seconds\n- **[Watch Video](https://youtu.be/1RJTuQBz-Xo)**\n- **Thumbnails**:\n  - ![Thumbnail 1](https://i.ytimg.com/vi/1RJTuQBz-Xo/3.jpg)\n  - ![Thumbnail 2](https://i.ytimg.com/vi/1RJTuQBz-Xo/default.jpg)\n\n---\n\n### 3. **USA Channel in 3 Simple Steps! 🇺🇸 | #shorts #shortsfeed**\n- **Channel**: Yt Growgenix\n- **Views**: 1,083,465\n- **Likes**: 24,942\n- **Comments**: 217\n- **Duration**: 22 seconds\n- **[Watch Video](https://youtu.be/zD_emrKhMkw)**\n- **Thumbnails**:\n  - ![Thumbnail 1](https://i.ytimg.com/vi/zD_emrKhMkw/3.jpg)\n  - ![Thumbnail 2](https://i.ytimg.com/vi/zD_emrKhMkw/default.jpg)\n\n--- \n\nLet me know if you need more information!' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 484, 'prompt_tokens': 9361, 'total_tokens': 9845, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 1152}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_60831fdee3', 'id': 'chatcmpl-DSDPQPHZRp50UC7eXA6k2A5lqpaqn', 'finish_reason': 'stop', 'logprobs': None} id='run--019d6b06-c970-7440-939a-8223aca6b743-0' usage_metadata={'input_tokens': 9361, 'output_tokens': 484, 'total_tokens': 9845, 'input_token_details': {'audio': 0, 'cache_read': 1152}, 'output_token_details': {'audio': 0, 'reasoning': 0}}


# Exercise


### Exercise 1: Try a different video with a Youtube link



```python
youtube_url = "https://www.youtube.com/watch?v=rkZzg7Vowao"
```

<details>
    <summary>Click here for hint</summary>

```python
video_id = "INSERT_VIDEO_ID_HERE"  # Replace with the actual video ID
```

</details>


### Exercise 2: Extract the video ID



```python
video_id = extract_video_id.run(youtube_url)
print(f"Extracted video ID: {video_id}")
```

    Extracted video ID: rkZzg7Vowao


### Exercise 3: Collect all necessary data about the video in one go



```python
video_metadata = get_full_metadata.run(youtube_url)
print(f"Retrieved metadata for: {video_metadata['title']}")
```

    Retrieved metadata for: The Man Who Revolutionized Computer Science With Math


### Exercise 4: Get video transcript



```python
transcript = fetch_transcript.run(video_id)
print(f"Retrieved transcript with {len(transcript)} characters")
```

    Retrieved transcript with 5966 characters


### Exercise 5: Get video thumbnails



```python
thumbnails = get_thumbnails.run(youtube_url)
print(f"Retrieved {len(thumbnails)} thumbnails")
```

    Retrieved 42 thumbnails


### Let's have a comprehensive prompt to be passed to LLM to generate a summary



```python
prompt = f"""
Please analyze this YouTube video and provide a comprehensive summary.

VIDEO TITLE: {video_metadata['title']}
CHANNEL: {video_metadata['channel']}
VIEWS: {video_metadata['views']}
DURATION: {video_metadata['duration']} seconds
LIKES: {video_metadata['likes']}

TRANSCRIPT EXCERPT:
{transcript[:3000]}... (transcript truncated for brevity)

Based on this information, please provide:
1. A concise summary of the video content (3-5 bullet points)
2. The main topics or themes discussed
3. The intended audience for this content
4. A brief analysis of why this video might be performing well (or not)
"""
```

### Exercise 6: Single LLM invocation with all the data



```python
messages = [HumanMessage(content=prompt)]
response = llm.invoke(messages)
```

### Exercise 7: Display the comprehensive analysis



```python
print("\n===== VIDEO ANALYSIS =====\n")
print(response.content)
```

    
    ===== VIDEO ANALYSIS =====
    
    ### Summary of the Video Content:
    
    1. **Leslie Lamport’s Journey**: Leslie Lamport shares his evolution from a programmer to a recognized computer scientist, highlighting how his background in mathematics shaped his approach to computer science.
    2. **Distinction Between Programming and Coding**: He emphasizes the difference between programming (designing algorithms) and coding (writing down the code), comparing it to the distinction between writing and typing.
    3. **Mathematics in Computer Science**: Lamport discusses the importance of formal proofs for algorithms, stating that an algorithm without a proof is merely a conjecture.
    4. **Teaching Programming Conceptually**: He stresses the need for a better mathematical foundation in programming education to help bridge the gap between abstract concepts and practical implementation.
    5. **Distributed Systems**: The video touches on Lamport's interest in distributed systems, explaining how they function differently compared to traditional computing models.
    
    ### Main Topics or Themes Discussed:
    
    - The evolution of computer science as a discipline.
    - The philosophy of programming versus coding.
    - The necessity of rigorous mathematics in algorithm design.
    - Challenges in mathematical education affecting programming skills.
    - Innovations in distributed computing and its practical implications.
    
    ### Intended Audience for the Content:
    
    - Aspiring and current computer scientists or programmers.
    - Mathematics enthusiasts and educators.
    - Individuals interested in the history and philosophy of technology.
    - General audiences wanting to understand complex computing concepts in a simplified manner.
    
    ### Brief Analysis of Video Performance:
    
    This video is likely performing well due to several factors:
    
    - **Engaging Subject Matter**: Leslie Lamport is a significant figure in computer science, and his insights on programming and mathematics attract viewers including students and professionals.
    - **Clear Distinctions**: The explanation of programming versus coding resonates with many who may have had misconceptions about the two, providing value to both novices and experienced programmers.
    - **Educational Value**: The focus on teaching and understanding the concepts behind programming can engage viewers who are interested in enhancing their skills or knowledge.
    - **Relevance to Modern Computing**: Topics such as distributed systems are highly relevant in today’s tech landscape, making it timely content that appeals to a broad audience. 
    
    Overall, the combination of Leslie Lamport’s credibility, the educational nature of the content, and the contemporary relevance of the topics likely contribute to the video's popularity and high view count.


## Authors


[Kunal Makwana](https://author.skills.network/instructors/kunal_makwana) is a Data Scientist at IBM and is currently pursuing his Master's in Computer Science at Dalhousie University.


### Other Contributors


[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)


Copyright © IBM Corporation. All rights reserved.

