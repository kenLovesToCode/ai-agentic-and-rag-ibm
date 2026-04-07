<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# <a id='toc1_'></a>[Build an Image Captioning System with IBM watsonx and Granite](#toc0_)


Estimated time needed: **30** minutes


In this lab, you’ll explore how to use the IBM Granite 3.2 Vision model with IBM watsonx to perform multimodal tasks like image captioning and visual question answering using Python.


**Table of contents**<a id='toc0_'></a>    
<ul>
    <li>
        <a href="#toc1_">Build an Image Captioning System with IBM watsonx and Granite 3.2 Vision</a>
        <ul>
            <li><a href="#Introduction">Introduction</a></li>
            <li><a href="#What-does-this-guided-project-do?">What does this guided project do?</a></li>
            <li><a href="#Objectives">Objectives</a></li>
            <li>
                <a href="#Background">Background</a>
                <ul>
                    <li><a href="#What-is-large-language-model-(LLM)?">What is large language model (LLM)?</a></li>
                    <li><a href="#What-is-IBM-watsonx?">What is IBM watsonx?</a></li>
                    <li><a href="#What-is-IBM-Granite-3.2-Vision?">What is IBM Granite 3.2 Vision?</a></li>
                </ul>
            </li>
            <li>
                <a href="#Setup">Setup</a>
                <ul>
                    <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
                </ul>
            </li>
            <li><a href="#watsonx-API-credentials-and-project_id">watsonx API credentials and project_id</a></li>
            <li><a href="#Image-preparation">Image preparation</a></li>
            <li>
                <a href="#Work-with-large-language-models-on-watsonx.ai">Work with large language models on watsonx.ai</a>
                <ul>
                    <li><a href="#Check-the-model-parameters">Check the model parameters</a></li>
                </ul>
            </li>
            <li><a href="#Initialize-the-model">Initialize the model</a></li>
            <li><a href="#Encode-the-image">Encode the image</a></li>
            <li><a href="#Multimodal-inference-function">Multimodal inference function</a></li>
            <li><a href="#Image-captioning">Image captioning</a></li>
            <li><a href="#Object-detection">Object detection</a></li>
            <li><a href="#Conclusion">Conclusion</a></li>
            <li>
                <a href="#Exercises">Exercises</a>
            </li>
            <li><a href="#Authors">Authors</a></li>
        </ul>
    </li>
</ul>


----


## [Introduction](#toc0_)

Visual content—like photos, screenshots, or charts—often contains important information that can be hard to interpret at a glance. Wouldn’t it be useful if an AI model could instantly describe what’s in an image, or answer questions about it?

In this guided project, we’ll explore how to use a large multimodal language model to do exactly that. You'll use IBM's Granite 3.2 Vision model, integrated with IBM watsonx, to generate text responses based on visual inputs. From scene descriptions to answering specific questions, this model can help turn images into insights.

## [What does this guided project do?](#toc0_)

This project demonstrates how to:

- Load image data from URLs.
- Encode those images so they can be processed by a language model.
- Use the IBM's Granite 3.2 Vision model through IBM watsonx to generate text responses based on each image.
## [Objectives](#toc0_)

By the end of this lab, you will be able to:

- Understand how to encode images for LLM-based visual processing.
- Use the IBM's Granite 3.2 Vision model to describe or analyze images.
- Interact with the IBM watsonx API to perform multimodal queries in Python.


## [Background](#toc0_)

### [What is large language model (LLM)?](#toc0_)

[Large language models](https://www.ibm.com/think/topics/large-language-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+an+Image+Captioning+System+with+watsonx+and+Llama-v1_1745515774) are a category of foundation models that are trained on immense amounts of data making them capable of understanding and generating natural language and other types of content to perform a wide range of tasks.

### [What is IBM watsonx?](#toc0_)

[IBM watsonx](https://www.ibm.com/watsonx?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+an+Image+Captioning+System+with+watsonx+and+Llama-v1_1745515774) is a suite of artificial intelligence (AI) tools and services that are designed to help developers build and deploy AI-driven applications. Watsonx provides a range of APIs and tools that make it easy to integrate AI capabilities into applications, including natural language processing, computer vision, and speech recognition.

**Enterprises turn to watsonx because it is:**

- **Open**: Based on open technologies that provide a variety of models to cover enterprise use cases and support compliance initiatives.
- **Targeted**: Targeted to specific enterprise domains like HR, customer service, or IT operations to unlock new value.
- **Trusted**: Designed with principles of transparency, responsibility, and governance so you can manage ethical and accuracy concerns.
- **Empowering**: You can go beyond being an AI user and become an AI value creator, owning the value that your models create.

### [What is IBM Granite 3.2 Vision?](#toc0_)

[IBM's Granite 3.2 series](https://www.ibm.com/new/announcements/ibm-granite-3-2-open-source-reasoning-and-vision?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+an+Image+Captioning+System+with+watsonx+and+Llama-v1_1745515774) introduces advanced reasoning and vision capabilities tailored for enterprise applications. The Granite Vision 3.2 2B model is a lightweight, open-source vision-language model specifically optimized for visual document understanding. Trained on a comprehensive instruction-following dataset, it excels in extracting information from tables, charts, diagrams, and infographics, making it a powerful tool for structured data analysis in business contexts. 

In addition to its vision capabilities, Granite 3.2 incorporates enhanced reasoning features. The models support conditional reasoning, allowing users to activate or deactivate reasoning processes as needed, optimizing computational resources. This flexibility enables the models to handle complex decision-making tasks efficiently, such as software engineering challenges and IT issue resolution.


## <a id='Setup'></a>[Setup](#toc0_)

For this lab, you will be using the following libraries:


*   [`ibm-watsonx-ai`](https://pypi.org/project/ibm-watsonx-ai/): `ibm-watsonx-ai` is a library that allows to work with watsonx.ai service on IBM Cloud and IBM Cloud for Data. Train, test and deploy your models as APIs for application development, share with colleagues using this python library.

* `image`: `image` from Pillow is the Python Imaging Library (PIL) fork that provides easy-to-use methods for opening, manipulating, and saving image files in various formats. It’s commonly used for preprocessing images before feeding them into machine learning models or APIs.

* `requests`: `requests` is a simple and intuitive HTTP library for Python. It sends all kinds of HTTP/1.1 requests with methods like GET and POST. In this lab, it downloads images from the web for analysis by the multimodal AI model.


### [Installing required libraries](#toc0_)

The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You must run the following cell__ to install them. Please wait until it completes.

This step could take **several minutes**; please be patient.

**NOTE**: If you encounter any issues, please restart the kernel and run the cell again.  You can do that by clicking the **Restart the kernel** icon.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/crvBKBOkg9aBzXZiwGEXbw/Restarting-the-Kernel.png" width="50%" alt="Restart kernel">



```python
%%capture
%pip install ibm-watsonx-ai==1.1.20 image==1.5.33 requests==2.32.0
```

## <a id='toc1_6_'></a>[watsonx API credentials and project_id](#toc0_)


This section provides you with the necessary credentials to access the watsonx API.

**Please note:**

In this lab environment, you don't need to specify the api_key, and the project_id is pre_set as "skills-network", but if you want to use the model locally, you need to go to [watsonx](https://www.ibm.com/watsonx?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+an+Image+Captioning+System+with+watsonx+and+Llama-v1_1745515774) to create your own keys and id.



```python
from ibm_watsonx_ai import Credentials, APIClient
import os

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    )

project_id="skills-network"
client = APIClient(credentials)
# GET TextModels ENUM
client.foundation_models.TextModels

# PRINT dict of Enums
client.foundation_models.TextModels.show()
```

    {'GRANITE_4_H_SMALL': 'ibm/granite-4-h-small', 'GRANITE_8B_CODE_INSTRUCT': 'ibm/granite-8b-code-instruct', 'GRANITE_GUARDIAN_3_8B': 'ibm/granite-guardian-3-8b', 'LLAMA_3_2_11B_VISION_INSTRUCT': 'meta-llama/llama-3-2-11b-vision-instruct', 'LLAMA_3_3_70B_INSTRUCT': 'meta-llama/llama-3-3-70b-instruct', 'LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8': 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8', 'LLAMA_GUARD_3_11B_VISION': 'meta-llama/llama-guard-3-11b-vision', 'MISTRAL_MEDIUM_2505': 'mistralai/mistral-medium-2505', 'MISTRAL_SMALL_3_1_24B_INSTRUCT_2503': 'mistralai/mistral-small-3-1-24b-instruct-2503', 'GPT_OSS_120B': 'openai/gpt-oss-120b'}


## <a href="#Image-preparation">Image preparation</a>

- Download the image
- Display the image



```python
url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4] 
```

To gain a better understanding of our data input, let's display the images.


![Image 1](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png)<figcaption>Image 1</figcaption>

![Image 2](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png)<figcaption>Image 2</figcaption>

![Image 3](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png)<figcaption>Image 3</figcaption>

![Image 4](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png)<figcaption>Image 4</figcaption>


## <a href="#Work-with-large-language-models-on-watsonx.ai"></a>[Work with large language models on watsonx.ai](#toc0_)

Specify the `model_id` of the model that you will use for the chat with image modalities.



```python
model_id = 'meta-llama/llama-3-2-11b-vision-instruct'
```

### <a id='toc1_8_1_'></a>[Check the model parameters](#toc0_)

More information about the `TextChatParameters` can be found here: [docs](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#ibm_watsonx_ai.foundation_models.schema.TextChatParameters), [source](https://ibm.github.io/watsonx-ai-python-sdk/_modules/ibm_watsonx_ai/foundation_models/schema/_api.html#TextChatParameters).

```python
@dataclass
class TextChatParameters(BaseSchema):
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    response_format: dict | TextChatResponseFormat | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    time_limit: int | None = None
    top_p: float | None = None
    n: int | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for TextChatParameters."""
        return {
            "frequency_penalty": 0.5,
            "logprobs": True,
            "top_logprobs": 3,
            "presence_penalty": 0.3,
            "response_format": TextChatResponseFormat.get_sample_params(),
            "temperature": 0.7,
            "max_tokens": 100,
            "time_limit": 600000,
            "top_p": 0.9,
            "n": 1,
        }
```



```python
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

TextChatParameters.show()

params = TextChatParameters(
    temperature=0.2,
    top_p=0.5,

)

params
```

    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | PARAMETER         | TYPE                                                                              | EXAMPLE VALUE           |
    +===================+===================================================================================+=========================+
    | frequency_penalty | float | None                                                                      | 0.5                     |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | logprobs          | bool | None                                                                       | True                    |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | top_logprobs      | int | None                                                                        | 3                       |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | presence_penalty  | float | None                                                                      | 0.3                     |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | response_format   | dict | ibm_watsonx_ai.foundation_models.schema._api.TextChatResponseFormat | None | {'type': 'json_object'} |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | temperature       | float | None                                                                      | 0.7                     |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | max_tokens        | int | None                                                                        | 100                     |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | time_limit        | int | None                                                                        | 600000                  |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | top_p             | float | None                                                                      | 0.9                     |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+
    | n                 | int | None                                                                        | 1                       |
    +-------------------+-----------------------------------------------------------------------------------+-------------------------+





    TextChatParameters(frequency_penalty=None, logprobs=None, top_logprobs=None, presence_penalty=None, response_format=None, temperature=0.2, max_tokens=None, time_limit=None, top_p=0.5, n=None)



## <a id='toc1_9_'></a>[Initialize the model](#toc0_)


Initialize the `ModelInference` class with the previously specified parameters.



```python
import os
from ibm_watsonx_ai.foundation_models import ModelInference

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)
```

## <a id='toc1_10_'></a>[Encode the image](#toc0_)

Encode the image to `base64.b64encode`. Why do you need to encode the image to `base64.b64encode`? JSON is a text-based format and does not support binary data. By encoding the image as a Base64 string, you can embed the image data directly within the JSON structure.



```python
import base64
import requests

def encode_images_to_base64(image_urls):
    """
    Downloads and encodes a list of image URLs to base64 strings.

    Parameters:
    - image_urls (list): A list of image URLs.

    Returns:
    - list: A list of base64-encoded image strings.
    """
    encoded_images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            encoded_images.append(encoded_image)
            print(type(encoded_image))
        else:
            print(f"Warning: Failed to fetch image from {url} (Status code: {response.status_code})")
            encoded_images.append(None)
    return encoded_images
```


```python
encoded_images = encode_images_to_base64(image_urls)
```

    <class 'str'>
    <class 'str'>
    <class 'str'>
    <class 'str'>


## <a id='#Multimodal-inference-function'></a>[Multimodal inference function](#toc0_)

Next, define a function to generate responses from the model.

The `generate_model_response` function is designed to interact with a multimodal AI model that accepts both text and image inputs. This function takes an image, along with a user’s query, and generates a response from the model.

#### Function purpose

The function sends an image and a query to the AI model and retrieves a description or answer. It combines a text-based prompt and an image to guide the model in generating a concise response.

#### Parameters

- **`encoded_image`** (`str`): A base64-encoded image string, which allows the model to process the image data.
- **`user_query`** (`str`): The user's question about the image, providing context for the model to interpret the image and answer appropriately.
- **`assistant_prompt`** (`str`): An optional text prompt to guide the model in responding in a specific way. By default, the prompt is set to: `"You are a helpful assistant. Answer the following user query in 1 or 2 sentences:"`.



```python
def generate_model_response(encoded_image, user_query, assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    """
    Sends an image and a query to the model and retrieves the description or answer.

    Parameters:
    - encoded_image (str): Base64-encoded image string.
    - user_query (str): The user's question about the image.
    - assistant_prompt (str): Optional prompt to guide the model's response.

    Returns:
    - str: The model's response for the given image and query.
    """
    
    # Create the messages object
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": assistant_prompt + user_query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded_image,
                    }
                }
            ]
        }
    ]
    
    # Send the request to the model
    response = model.chat(messages=messages)
    
    # Return the model's response
    return response['choices'][0]['message']['content']
```

### Steps explained

1. **Create the Messages object:**  
   The function constructs a list of messages in JSON-like format. This object includes:
   - A "user" role with a "content" array. The content array contains:
     - A text field, combining the `assistant_prompt` and the `user_query`.
     - An image URL field, which includes a base64-encoded image string. This is essential for sending image data to the model.
2. **Send the request to the model:** `response = model.chat(messages=messages)`
	The function sends the constructed messages to the model using a chat-based API. The model.chat function is invoked with the messages parameter to generate the model's response.
3. **Return the model’s response:** `return response['choices'][0]['message']['content']`
	The model’s response is returned as a string, extracted from the response object. Specifically, the function retrieves the content of the first choice in the model's response.


## <a id='#Image-captioning'></a>[Image captioning](#toc0_)

Generate an answer to your question using the `ibm/granite-vision-3-2-2b` model.

More information about the `chat` can be found here: [docs](https://ibm.github.io/watsonx-ai-python-sdk/fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.chat), [source](https://ibm.github.io/watsonx-ai-python-sdk/_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.chat).

Now, you can loop through our images to see the text descriptions produced by the model in response to the query, "Describe the photo".



```python
user_query = "Describe the photo"

for i in range(len(encoded_images)):
    image = encoded_images[i]

    response = generate_model_response(image, user_query)

    # Print the response with a formatted description
    print(f"Description for image {i + 1}: {response}/n/n")
```

    Description for image 1: The photo shows a busy street in New York City, with tall buildings lining the road and people walking across the street. The street is filled with cars, buses, and other vehicles, and there are trees and greenery along the sidewalk./n/n
    Description for image 2: The photo shows a woman running on a road in front of a large white building with a car parked nearby. The woman is wearing a yellow jacket, black pants, and black shoes, and has her hair pulled back into a ponytail./n/n
    Description for image 3: The photo shows a flooded farm with a house, barn, and silos surrounded by water. The floodwaters have risen to the level of the house's foundation, and the surrounding fields are also underwater./n/n
    Description for image 4: The photo shows a hand holding a nutrition label, with the index finger pointing to the "Total Fat" section. The label is white with black text and has a purple border around it./n/n


## <a id='#Object-detection'></a>[Object detection](#toc0_)


Now that you have showcased the model's ability to perform image captioning in the previous step, let's ask the model some questions that require object detection. Our system prompt will remain the same as in the previous section. The difference now will be in the user query. Regarding the second image depicting the woman running outdoors, you will be asking the model, "How many cars are in this image?". You can comment out the code section for image captioning if you don't want to wait for the response on that part again.



```python
image = encoded_images[1]

user_query = "How many cars are in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
```

    User Query:  How many cars are in this image?
    Model Response:  There is one car in this image. It is located on the right side of the image, near the building.


The model correctly identified the singular vehicle in the image. Now, let's inquire about the damage depicted in the image of flooding.



```python
image = encoded_images[2]

user_query = "How severe is the damage in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
```

    User Query:  How severe is the damage in this image?
    Model Response:  The damage in this image is severe, with the entire area being flooded and the water level rising to the rooftops of the buildings. The flooding has caused significant disruption and destruction, with the water likely affecting the surrounding land and infrastructure.


This response highlights the value that multimodal AI has for domains like insurance. The model was able to detect the severity of the damage caused to the flooded home. This could be a powerful tool for improving insurance claim processing time.

Next, let's ask the model how much sodium content is in the nutrition label image.



```python
image = encoded_images[3]

user_query = "How much sodium is in this product?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
```

    User Query:  How much sodium is in this product?
    Model Response:  This product contains 640mg of sodium.


Great! The model was able to discern objects within the images following user queries. We encourage you to try more queries to further demonstrate the model's performance.


## <a id='#Conclusion'></a>[Conclusion](#toc0_)


In this lab, you explored the capabilities of IBM's Granite 3.2 Vision, one of the latest and most powerful multimodal models developed by IBM. Using IBM watsonx, you were able to seamlessly integrate this model into a Python-based workflow to perform tasks like:

- Generating detailed image captions

- Answering object detection questions (e.g., number of cars in an image)

- Assessing visual damage in real-world disaster scenarios

- Extracting specific information from product labels

This lab not only introduced you to multimodal AI development but also demonstrated how cutting-edge models can turn visual content into actionable insight. Whether you're building apps for enterprise, education, or everyday use, the tools and techniques you’ve learned here are a solid foundation for what's possible with AI today.

We encourage you to extend this notebook by asking new questions, uploading your own images, or combining image and text prompts for more advanced reasoning tasks. The future of AI is multimodal—this is your starting point.


## <a id='#Exercises'></a>[Exercises](#toc0_)

Now, let's practice by exploring some other capabilities of this model. Try asking "How much cholesterol is in this product?" in the 4th image



```python
image = encoded_images[3]
user_query = "How much cholesterol is in this product?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
```

    User Query:  How much cholesterol is in this product?
    Model Response:  This product contains 20mg of cholesterol.


Try asking "What is the color of the woman's jacket?" in the 2nd image.



```python
image = encoded_images[1]
user_query = "What is the color of the woman's jacket?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
```

    User Query:  What is the color of the woman's jacket?
    Model Response:  The woman's jacket is yellow.


## Authors


[Hailey Quach](https://www.haileyq.com/)


<!--
## Change log

|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2025-04-24|1.0|Hailey Quach|Ininitial version|

-->


Copyright © IBM Corporation. All rights reserved.

