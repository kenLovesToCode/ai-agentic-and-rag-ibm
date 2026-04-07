<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# <a id='toc1_'></a>[DALL-E: Image generation Guide for Beginners](#toc0_)


Estimated time needed: **30** minutes


In this lab, you will learn how to use DALL-E series to generate images from text.


**NOTE: Due to environment limitations, currently only the **prompt** can be modified; edit and variation features are not available at this time.**



## __Table of Contents__

<ol>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#What-does-this-guided-project-do?">What does this guided project do?</a></li>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Background">Background</a>
        <ol>
            <li><a href="#What-is-large-language-model-(LLM)?">What is large language model (LLM)?</a></li>
            <li><a href="#What-is-multimodal?">What is multimodal?</a></li>
            <li><a href="#What-is-Dall-E-2?">What is Dall-E 2?</a></li>
            <li><a href="#What-is-Dall-E-3?">What is Dall-E 3?</a></li>
        </ol>
    </li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
        </ol>
    </li>
    <li>
        <a href="#Image-generation">Image generation</a>
        <ol>
            <li><a href="#Which-model-should-I-use?">Which model should I use?</a></li>
            <li><a href="#Generations">Generations</a></li>
            <li><a href="#Edits-(Dall-E-2-only)">Edits (Dall-E 2 only)</a></li>
            <li><a href="#Variations-(Dall-E-2-only)">Variations (Dall-E 2 only)</a></li>
        </ol>
    </li>
    <li>
        <a href="#Practice">Practice</a>
        <ol>
            <li><a href="#Use-Dall-E-2-to-generate-an-image-of-a-cat">Use Dall-E 2 to generate an image of a cat</a></li>
            <li><a href="#Use-Dall-E-3-to-generate-an-image-of-a-cat">Use Dall-E 3 to generate an image of a cat</a></li>
        </ol>
    </li>
    <li><a href="#Compare-the-two-images">Compare the two images</a></li>
    <li>
        <a href="#Exercises">Exercises</a>
        <ol>
            <li><a href="#Exercise-1:-Generate-another-image-using-Dall-E-2">Exercise 1: Generate another image using Dall-E 2</a></li>
            <li><a href="#Exercise-2:-Generate-another-image-using-Dall-E-3">Exercise 2: Generate another image using Dall-E 3</a></li>
        </ol>
    </li>
    <li><a href="#Authors">Authors</a></li>
    <li><a href="#Contributors">Contributors</a></li>
</ol>


<h2 id="intro"><a href="#Table-of-Contents">Introduction</a></h2>


Have you ever wanted to create stunning images from just a text description? With the power of AI image generation, this is now possible. In this project, we'll explore DALL·E series, OpenAI's revolutionary text-to-image model that can create realistic images and art from natural language descriptions.

<h2 id="What-does-this-guided-project-do?"><a href="#Table-of-Contents">What does this guided project do?</a></h2>


This project demonstrates how to use DALL·E series to generate images by:
1. Crafting effective text prompts that describe the images you want to create
2. Using the OpenAI API to generate images from these prompts
3. Exploring different parameters to control the image generation process

For example, you could input a prompt like "**a serene landscape with mountains reflected in a lake at sunset**" and DALL·E will create a beautiful image matching your description. This technology can be used for creating illustrations, concept art, design mockups, or simply exploring your creative ideas in visual form.

<h2 id="Objectives"><a href="#Table-of-Contents">Objectives</a></h2>


After completing this lab you will be able to:
 - Craft effective prompts for DALL·E image generation
 - Use the OpenAI API to generate images from text descriptions
 - Understand the parameters that control image generation
 - Save and use the generated images in your projects


<h2 id="Background"><a href="#Table-of-Contents">Background</a></h2>

<h3 id="What-is-large-language-model-(LLM)?"><a href="#Table-of-Contents">What is large language model (LLM)?</a></h3>


[Large language models](https://www.ibm.com/topics/large-language-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-DALL%C2%B7E+Image+generation+Guide+for+Beginners-v1_1741202576) are a category of foundation models trained on immense amounts of data making them capable of understanding and generating natural language and other types of content to perform a wide range of tasks.

<h3 id="What-is-multimodal?"><a href="#Table-of-Contents">What is multimodal?</a></h3>

Multimodal refers to the capability of a model to process and understand multiple types of data simultaneously. In the context of AI and machine learning, multimodal models can handle and integrate information from various modalities, such as:
 
- **Text to image**: Generating images based on textual descriptions, as seen in models like DALL·E.
- **Text to audio**: Converting written text into spoken words or sounds.
- **Image to ext**: Analyzing images to produce descriptive text or captions.
- **Audio to text**: Transcribing spoken language into written text.
- **Video analysis**: Understanding and interpreting video content by integrating visual and audio data.

This capability allows for a more comprehensive and nuanced understanding and generation of content. For example, a multimodal AI system can take a text description and generate a corresponding image or analyze an image and generate descriptive text. This integration of different types of data enables more sophisticated applications and interactions, such as creating detailed visual content from textual descriptions or providing richer context in conversational AI systems.


<h3 id="What-is-Dall-E-2?"><a href="#Table-of-Contents">What is Dall-E 2?</a></h3>


[DALL·E 2](https://openai.com/dall-e-2) is an AI system developed by OpenAI that can create realistic images and art from text descriptions. Released in 2022, it's the successor to the original DALL·E model. Key features include:

- **Text-to-image generation**: Creates images from natural language descriptions
- **Image editing**: Allows for modifications to existing images
- **Variations**: Can generate multiple variations of an image
- **Resolution control**: Creates images at different resolutions
- **Proprietary technology**: Unlike open-source models, DALL·E 2 is a commercial product from OpenAI


<h3 id="What is Dall-E-3?"><a href="#Table-of-Contents">What is Dall-E 3?</a></h3>


[DALL·E 3](https://openai.com/dall-e-3) is OpenAI's most advanced text-to-image model, released in 2023. It represents a significant improvement over DALL·E 2 with the following key features:

- **Higher quality images**: Produces more detailed, accurate, and visually stunning images
- **Better text understanding**: More accurately interprets complex prompts and follows specific instructions
- **Text rendering**: Significantly improved ability to generate readable text within images
- **Artistic styles**: Better at capturing specific artistic styles and visual aesthetics
- **Safety features**: Enhanced content filtering and safety measures
- **Integration with ChatGPT**: Can be accessed directly through ChatGPT to refine prompts interactively
 
DALL·E 3 can generate images at higher resolutions and with greater fidelity to the user's intent, making it particularly valuable for professional creative work and detailed visualizations.


<h2 id="Setup"><a href="#Table-of-Contents">Setup</a></h2>


For this lab, you will be using the following libraries:

*   [`openai`](https://pypi.org/project/openai/): `openai` is a library that allows working with the OpenAI API.


<h3 id="Installing-required-libraries"><a href="#Table-of-Contents">Installing required libraries</a></h3>


The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You must run the following cell__ to install them. Please wait until it completes.

This step could take **several minutes**, please be patient.

**NOTE**: To prevent any issues, after installing the below libraries, please restart the kernel and skip to the next cell.  You can do that by clicking the **Restart the kernel** icon.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/crvBKBOkg9aBzXZiwGEXbw/Restarting-the-Kernel.png" width="100%" alt="Restart kernel">



```python
%pip install openai==1.64.0 | tail -n 1
```

    Successfully installed jiter-0.13.0 openai-1.64.0
    Note: you may need to restart the kernel to use updated packages.


<h2 id="Image-generation"><a href="#Table-of-Contents">Image generation</a></h2>


The [Images API](https://platform.openai.com/docs/api-reference/images) has three endpoints with different abilities:

- **Generations**: Images from scratch, based on a text prompt
- **Edits**: Edited versions of images, where the model replaces some areas of a pre-existing image, based on a new text prompt
- **Variations**: Variations of an existing image

<h3 id="Which-model-should-I-use?"><a href="#Table-of-Contents">Which model should I use?</a></h3>


DALL·E 2 and DALL·E 3 have different [options](https://platform.openai.com/docs/guides/images) for generating images.

| Model | Available endpoints | Best for |
| --- | --- | --- |
| DALL·E 2 | Generations, edits, variations | More options (edits and variations), more control in prompting, more requests at once |
| DALL·E 3 | Only image generations | Higher quality, larger sizes for generated images |


<h3 id="Generations"><a href="#Table-of-Contents">Generations</a></h3>


The [image generations](https://platform.openai.com/docs/api-reference/images/create) endpoint allows you to create an original image with a text prompt. Each image can be returned either as a URL or Base64 data, using the [response_format](https://platform.openai.com/docs/api-reference/images/create#images/create-response_format) parameter. The default output is URL, and each URL expires after an hour.

**Size and quality options**

Square, standard quality images are the fastest to generate. The default size of generated images is `1024x1024` pixels, but each model has different options:

| Model | Sizes options (pixels) | Quality options | Requests you can make |
| --- | --- | --- | --- |
| DALL·E 2 | `256x256` `512x512` `1024x1024` | Only `standard` | Up to 10 images at a time, with the [n parameter](https://platform.openai.com/docs/api-reference/images/create#images/create-n) |
| DALL·E 3 | `1024x1024` `1024x1792` `1792x1024` | Defaults to `standard` Set `quality: "hd"` for enhanced detail | Only 1 at a time, but can request more by making parallel requests |


<h3 id="Edits-(Dall-E-2-only)"><a href="#Table-of-Contents">Edits (Dall-E 2 only)</a></h3>


The [image edits](https://platform.openai.com/docs/api-reference/images/create-edit) endpoint lets you edit or extend an image by uploading an image and mask indicating which areas should be replaced. This process is also known as **inpainting**.

The transparent areas of the mask indicate where the image should be edited, and the prompt should describe the full new image, **not just the erased area**.


| Image | Mask | Output |
|-------|------|--------|
| <img src="https://cdn.openai.com/API/images/guides/image_edit_original.webp" width="100%" /> | <img src="https://cdn.openai.com/API/images/guides/image_edit_mask.webp" width="100%" /> | <img src="https://cdn.openai.com/API/images/guides/image_edit_output.webp" width="100%" /> |

Prompt: a sunlit indoor lounge area with a pool containing a flamingo

The uploaded image and mask must both be square PNG images, less than 4MB in size, and have the same dimensions as each other. The non-transparent areas of the mask aren't used to generate the output, so they don’t need to match the original image like our example.



<h3 id="Variations-(Dall-E-2-only)"><a href="#Table-of-Contents">Variations (Dall-E 2 only)</a></h3>



The [image variations](https://platform.openai.com/docs/api-reference/images/create-variation) endpoint allows you to generate a variation of a given image.

| Image | Output |
|-------|--------|
| <img src="https://cdn.openai.com/API/images/guides/image_variation_original.webp" width="100%" /> | <img src="https://cdn.openai.com/API/images/guides/image_variation_output.webp" width="100%" /> |

Similar to the edits endpoint, the input image must be a square PNG image less than 4MB in size.


<h2 id="Practice"><a href="#Table-of-Contents">Practice</a></h2>


<h3 id="Use-Dall-E-2-to-generate-an-image-of-a-cat"><a href="#Table-of-Contents">Use Dall-E 2 to generate an image of a cat</a></h3>



Please use the following prompt: "a white siamese cat"



```python
from openai import OpenAI
from IPython import display

client = OpenAI()

response = client.images.generate(
    model="dall-e-2",
    prompt="a cute little panda sitting on a tree branch, digital art",
    size="1024x1024",
    # quality="standard",
    n=1,
)

url = response.data[0].url
display.Image(url=url, width=512)
```




<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-434DgKVrfIjWvK9YEpzlrvmq/labs-api-proxy-service-account/img-YDXlFMFdRJCks3LUQhzBlbzv.png?st=2026-04-07T13%3A37%3A25Z&se=2026-04-07T15%3A37%3A25Z&sp=r&sv=2026-02-06&sr=b&rscd=inline&rsct=image/png&skoid=ae240de5-197c-4e03-af8e-c66aed9a4539&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2026-04-07T08%3A28%3A31Z&ske=2026-04-08T08%3A28%3A31Z&sks=b&skv=2026-02-06&sig=1RRNR255pRBQgh7WZhzzfvQYiMLHjNQLhKwdK3rTTCE%3D" width="512"/>



<h3 id="Use-Dall-E-3-to-generate-an-image-of-a-cat"><a href="#Table-of-Contents">Use Dall-E 3 to generate an image of a cat</a></h3>



Please use the same prompt: "a white siamese cat"



```python
from openai import OpenAI
from IPython import display

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a cute little panda sitting on a tree branch, digital art",
    size="1024x1024",
    # quality="standard",
    n=1,
)

url = response.data[0].url
display.Image(url=url, width=512)
```




<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-434DgKVrfIjWvK9YEpzlrvmq/labs-api-proxy-service-account/img-4tmkIXX59pvGEdlfJvKuq4Kw.png?st=2026-04-07T13%3A36%3A49Z&se=2026-04-07T15%3A36%3A49Z&sp=r&sv=2026-02-06&sr=b&rscd=inline&rsct=image/png&skoid=ae240de5-197c-4e03-af8e-c66aed9a4539&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2026-04-06T20%3A29%3A42Z&ske=2026-04-07T20%3A29%3A42Z&sks=b&skv=2026-02-06&sig=3TPI04yh%2B6oQXgfMWQmrg9H1xxUMFT/NycEMGoc2Pmc%3D" width="512"/>



<h2 id="Compare-the-two-images"><a href="#Table-of-Contents">Compare the two images</a></h2>


Which one is better?


<h2 id="Exercises"><a href="#Table-of-Contents">Exercises</a></h2>

<h3 id="Exercise-1:-Generate-another-image-using-Dall-E-2"><a href="#Table-of-Contents">Exercise 1: Generate another image using Dall-E 2</a></h3>


Please generate another image using DALL·E 2.

Please use the following prompt: "a beautiful lake with a sunset"



```python
from openai import OpenAI
from IPython import display

client = OpenAI()

response = client.images.generate(
    model="dall-e-2",
    prompt="a beautiful lake with a sunset",
    size="1024x1024",
    n=1,
)

url = response.data[0].url
display.Image(url=url, width=512)
```




<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-434DgKVrfIjWvK9YEpzlrvmq/labs-api-proxy-service-account/img-R2n6QKE6FFToRVQ6BLncfiVm.png?st=2026-04-07T13%3A35%3A44Z&se=2026-04-07T15%3A35%3A44Z&sp=r&sv=2026-02-06&sr=b&rscd=inline&rsct=image/png&skoid=ae240de5-197c-4e03-af8e-c66aed9a4539&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2026-04-07T13%3A28%3A38Z&ske=2026-04-08T13%3A28%3A38Z&sks=b&skv=2026-02-06&sig=nk/GiDNRoqyCZjpvdMvC7j7qnaKBZb6J/v2YzeE1uVg%3D" width="512"/>



<h3 id="Exercise-2:-Generate-another-image-using-Dall-E-3"><a href="#Table-of-Contents">Exercise 2: Generate another image using Dall-E 3</a></h3>


Please generate another image using DALL·E 3.

Please use the following prompt: "a beautiful lake with a sunset"



```python
from openai import OpenAI
from IPython import display

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a beautiful lake with a sunset",
    size="1024x1024",
    # quality="standard",
    n=1,
)

url = response.data[0].url
display.Image(url=url, width=512)
```




<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-434DgKVrfIjWvK9YEpzlrvmq/labs-api-proxy-service-account/img-qMUAPRiLRgzILtorGKZTkOTU.png?st=2026-04-07T13%3A36%3A27Z&se=2026-04-07T15%3A36%3A27Z&sp=r&sv=2026-02-06&sr=b&rscd=inline&rsct=image/png&skoid=8eb2c87c-0531-4dab-acb3-b5e2adddce6c&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2026-04-07T00%3A26%3A09Z&ske=2026-04-08T00%3A26%3A09Z&sks=b&skv=2026-02-06&sig=7zqQaEuoKKCVrTaA2uP6lEFmZW40wITwBuyCDgE%2BYpc%3D" width="512"/>



<h2 id="Authors"><a href="#Table-of-Contents">Authors</a></h2>


[Ricky Shi](https://author.skills.network/instructors/ricky_shi)
[Hailey Quach](https://author.skills.network/instructors/hailey_quach)


<h2 id="Contributors"><a href="#Table-of-Contents">Contributors</a></h2>

[Karan Goswami](https://author.skills.network/instructors/karan_goswami)

```{Change Log}
```
```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2025-02-26|1.0|Ricky Shi|Create project|
```



Copyright © IBM Corporation. All rights reserved.

