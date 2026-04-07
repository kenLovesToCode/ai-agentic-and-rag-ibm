<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Build a Smarter Search with LangChain Context Retrieval**


Estimated time needed: **60** minutes


## __Table of Contents__

<ol>
    <li><a href="#Overview">Overview</a></li>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
            <li><a href="#Defining-helper-functions">Defining helper functions</a></li>
        </ol>
    </li>
    <li><a href="#Creating-a-retriever-model">Creating a retriever model</a></li>
    <ol>
        <li><a href="#Build-the-LLM">Build the LLM</a></li>
        <li><a href="#Use-the-text-splitter">Use the text splitter</a></li>
        <li><a href="#Create-the-embedding-model">Create the embedding model</a></li>
        <li>
            <a href="#Use-Retrievers">Use Retrievers</a>
            <ol>
                <li><a href="#Vector-Store-Backed-Retriever">Vector Store-Backed Retriever</a></li>
                <li><a href="#Multi-Query-Retriever">Multi-Query Retriever</a></li>
                <li><a href="#Self-Querying-Retriever">Self-Querying Retriever</a></li>
                <li><a href="#Parent-Document-Retriever">Parent Document Retriever</a></li>
            </ol>
        </li>
    </ol>

   
            
<li><a href="#Exercises">Exercises</a>
<ol>
<li><a href="#Retrieve-Top-2-Results-Using-a-Vector-Store-Backed-Retriever">Retrieve Top 2 Results Using Vector Store-Backed Retriever</a></li>
<li><a href="#Self-Querying-Retriever-for-a-Query">Self-Querying Retriever for a Query</a></li>
</ol>
</li>


## Overview


Imagine you are working on a project that involves processing a large collection of text documents, such as research papers, legal documents, or customer service logs. Your task is to develop a system that can quickly retrieve the most relevant segments of text based on a user's query. Traditional keyword-based search methods might not be sufficient, as they often fail to capture the nuanced meanings and contexts within the documents. To address this challenge, you can use different types of retrievers based on LangChain.

Using retrievers is crucial for several reasons:

- **Efficiency:** Retrievers enable fast and efficient retrieval of relevant information from large datasets, saving time and computational resources.
- **Accuracy:** By leveraging advanced retrieval techniques, these tools can provide more accurate and contextually relevant results compared to traditional search methods.
- **Versatility:** Different retrievers can be tailored to specific use cases, making them adaptable to various types of text data and query requirements.
- **Context awareness:** Some retrievers, such as the Parent Document Retriever, can consider the broader context of the document, enhancing the relevance of the retrieved segments.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/EUODrOFxvSSNL935zpwh9A/retriever.png" width="100%" alt="retriever"/>


In this lab, you will learn how to use various retrievers to efficiently extract relevant document segments from text using LangChain. 
You will learn about four types of retrievers: `Vector Store-backed Retriever`, `Multi-Query Retriever`, `Self-Querying Retriever`, and `Parent Document Retriever`. You will also learn the differences between these retrievers and understand the appropriate situations in which to use each one. By the end of this lab, you will be equipped with the skills to implement and utilize these retrievers in your projects.


## Objectives

After completing this lab, you will be able to:

- Use various types of retrievers to efficiently extract relevant document segments from text, leveraging LangChain's capabilities.
- Apply the Vector Store-backed Retriever to solve problems involving semantic similarity and relevance in large text datasets.
- Utilize the Multi-Query Retriever to address situations where multiple query variations are needed to capture comprehensive results.
- Implement the Self-Querying Retriever to automatically generate and refine queries, enhancing the accuracy of information retrieval.
- Employ the Parent Document Retriever to maintain context and relevance by considering the broader context of the parent document.


----


## Setup


For this lab, you will use the following libraries:

*   [`ibm-watson-ai`](https://ibm.github.io/watsonx-ai-python-sdk/index.html) for using LLMs from IBM's watsonx.ai.
*   [`langchain`, `langchain-ibm`, `langchain-community`](https://www.langchain.com/) for using relevant features from LangChain.
*   [`pypdf`](https://pypi.org/project/pypdf/)is an open-source pure Python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files.
*   [`chromadb`](https://www.trychroma.com/) is an open-source vector database used to store embeddings.
*   [`lark`](https://pypi.org/project/lark/) is a general-purpose parsing library for Python. It is necessary for a Self-Querying Retriever.


### Installing required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:

**Note:** The version is being pinned here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.

This might take approximately 1-2 minutes.



```python
!pip install "ibm-watsonx-ai==1.1.2" | tail -n 1
!pip install "langchain==0.2.1" | tail -n 1
!pip install "langchain-ibm==0.1.11" | tail -n 1
!pip install "langchain-community==0.2.1" | tail -n 1
!pip install "chromadb==0.4.24" | tail -n 1
!pip install "pypdf==4.3.1" | tail -n 1
!pip install "lark==1.1.9" | tail -n 1
!pip install 'posthog<6.0.0' | tail -n 1
```

    Successfully installed ibm-cos-sdk-2.13.6 ibm-cos-sdk-core-2.13.6 ibm-cos-sdk-s3transfer-2.13.6 ibm-watsonx-ai-1.1.2 jmespath-1.0.1 lomond-0.3.3 numpy-1.26.4 pandas-2.1.4 requests-2.32.2 tabulate-0.10.0 tzdata-2026.1
    Successfully installed langchain-0.2.1 langchain-core-0.2.43 langchain-text-splitters-0.2.4 langsmith-0.1.147 orjson-3.11.8 requests-toolbelt-1.0.0 tenacity-8.5.0
    Successfully installed langchain-ibm-0.1.11
    Successfully installed dataclasses-json-0.6.7 langchain-community-0.2.1 marshmallow-3.26.2 mypy-extensions-1.1.0 typing-inspect-0.9.0
    Successfully installed annotated-doc-0.0.4 asgiref-3.11.1 backoff-2.2.1 bcrypt-5.0.0 build-1.4.2 chroma-hnswlib-0.7.3 chromadb-0.4.24 click-8.3.2 durationpy-0.10 fastapi-0.135.3 filelock-3.25.2 flatbuffers-25.12.19 fsspec-2026.3.0 googleapis-common-protos-1.74.0 grpcio-1.80.0 hf-xet-1.4.3 httptools-0.7.1 huggingface-hub-1.9.0 kubernetes-35.0.0 markdown-it-py-4.0.0 mdurl-0.1.2 mmh3-5.2.1 mpmath-1.3.0 onnxruntime-1.24.4 opentelemetry-api-1.40.0 opentelemetry-exporter-otlp-proto-common-1.40.0 opentelemetry-exporter-otlp-proto-grpc-1.40.0 opentelemetry-instrumentation-0.61b0 opentelemetry-instrumentation-asgi-0.61b0 opentelemetry-instrumentation-fastapi-0.61b0 opentelemetry-proto-1.40.0 opentelemetry-sdk-1.40.0 opentelemetry-semantic-conventions-0.61b0 opentelemetry-util-http-0.61b0 posthog-7.9.12 protobuf-6.33.6 pulsar-client-3.10.0 pypika-0.51.1 pyproject_hooks-1.2.0 python-dotenv-1.2.2 requests-oauthlib-2.0.0 rich-14.3.3 shellingham-1.5.4 starlette-1.0.0 sympy-1.14.0 tokenizers-0.22.2 typer-0.24.1 typing-inspection-0.4.2 uvicorn-0.44.0 uvloop-0.22.1 watchfiles-1.1.1 websockets-16.0 wrapt-1.17.3
    Successfully installed pypdf-4.3.1
    Successfully installed lark-1.1.9
    Successfully installed posthog-5.4.0


After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/QrUNwLZfVySxQ9xvbOJgyQ/restart.png" width="80%" alt="Restart kernel">


## Defining helper functions

Use the following code to define some helper functions to reduce the repeat work in the notebook:



```python
# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
```

## Creating a retriever model


The following steps are involved  to create a retriever model using LangChain:

- Building LLMs
  
- Splitting documents into chunks
  
- Building an embedding model
  
- Retrieving related knowledge from text
  


### Build the LLM
Develop or select a pre-trained language model that can understand and generate human-like text. This model serves as the foundation for processing and interpreting language data.



```python
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
```

The following will allow us to connect to watsonx.ai and set LLM parameters



```python
def llm():
    model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'
    
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    
    project_id = "skills-network"
    
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    mixtral_llm = WatsonxLLM(model = model)
    return mixtral_llm
```

### Use the text splitter
Break down large documents into smaller, manageable pieces or chunks. This helps in processing and analyzing the text more efficiently, allowing the model to focus on specific sections rather than being overwhelmed by the entire document.



```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```


```python
def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks
```

### Create the embedding model


Create or utilize an embedding model to convert chunks of text into numerical vectors. These vectors represent the semantic meaning of the text, enabling the model to compare and retrieve relevant information based on similarity.
The following code demonstrates how to build an embedding model using the `watsonx.ai` package.

For this project, the `ibm/slate-125m-english-rtrvr-v2` embedding model is used.



```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
```


```python
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding
```

### Use Retrievers


A retriever is an interface designed to return documents based on an unstructured query. Unlike a vector store, which stores and retrieves documents, a retriever's primary function is to find and return relevant documents. While vector stores can serve as the backbone of a retriever, there are various other types of retrievers that can be used as well.


Retrievers take a string `query` as input and output a list of `Documents`.


#### Vector Store-Backed Retriever


A vector store retriever is a type of retriever that utilizes a vector store to fetch documents. It acts as a lightweight wrapper around the vector store class, enabling it to conform to the retriever interface. This retriever leverages the search methods implemented by the vector store, such as similarity search and Maximum Marginal Relevance (MMR), to query texts stored within it.


Before demonstrating this retriever, you need to load some example text. A `.txt` document has been prepared for you.



```python
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt"
```

    --2026-04-06 16:48:38--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104
    Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... connected.
    200 OKequest sent, awaiting response... 
    Length: 15660 (15K) [text/plain]
    Saving to: ‘companypolicies.txt’
    
    companypolicies.txt 100%[===================>]  15.29K  --.-KB/s    in 0s      
    
    2026-04-06 16:48:38 (34.0 MB/s) - ‘companypolicies.txt’ saved [15660/15660]
    


Use `TextLoader` to load the document.



```python
from langchain_community.document_loaders import TextLoader
```


```python
loader = TextLoader("companypolicies.txt")
txt_data = loader.load()
```

Let's take a look at this document. This is a document about different policies in a company.



```python
txt_data
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content="1.\tCode of Conduct\n\nOur Code of Conduct outlines the fundamental principles and ethical standards that guide every member of our organization. We are committed to maintaining a workplace that is built on integrity, respect, and accountability.\nIntegrity: We hold ourselves to the highest ethical standards. This means acting honestly and transparently in all our interactions, whether with colleagues, clients, or the broader community. We respect and protect sensitive information, and we avoid conflicts of interest.\nRespect: We embrace diversity and value each individual's contributions. Discrimination, harassment, or any form of disrespectful behavior is unacceptable. We create an inclusive environment where differences are celebrated and everyone is treated with dignity and courtesy.\nAccountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential violations of this code and support the investigation of such matters.\nSafety: We prioritize the safety of our employees, clients, and the communities we serve. We maintain a culture of safety, including reporting any unsafe conditions or practices.\nEnvironmental Responsibility: We are committed to minimizing our environmental footprint and promoting sustainable practices.\nOur Code of Conduct is not just a set of rules; it is the foundation of our organization's culture. We expect all employees to uphold these principles and serve as role models for others, ensuring we maintain our reputation for ethical conduct, integrity, and social responsibility.\n\n2.\tRecruitment Policy\n\nOur Recruitment Policy reflects our commitment to attracting, selecting, and onboarding the most qualified and diverse candidates to join our organization. We believe that the success of our company relies on the talents, skills, and dedication of our employees.\nEqual Opportunity: We are an equal opportunity employer and do not discriminate on the basis of race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or any other protected status. We actively promote diversity and inclusion.\nTransparency: We maintain transparency in our recruitment processes. All job vacancies are advertised internally and externally when appropriate. Job descriptions and requirements are clear and accurately represent the role.\nSelection Criteria: Our selection process is based on the qualifications, experience, and skills necessary for the position. Interviews and assessments are conducted objectively, and decisions are made without bias.\nData Privacy: We are committed to protecting the privacy of candidates' personal information and adhere to all relevant data protection laws and regulations.\nFeedback: Candidates will receive timely and constructive feedback on their application and interview performance.\nOnboarding: New employees receive comprehensive onboarding to help them integrate into the organization effectively. This includes information on our culture, policies, and expectations.\nEmployee Referrals: We encourage and appreciate employee referrals as they contribute to building a strong and engaged team.\nOur Recruitment Policy is a foundation for creating a diverse, inclusive, and talented workforce. It ensures that we attract and hire the best candidates who align with our company values and contribute to our continued success. We continuously review and update this policy to reflect evolving best practices in recruitment.\n\n3.\tInternet and Email Policy\n\nOur Internet and Email Policy is established to guide the responsible and secure use of these essential tools within our organization. We recognize their significance in daily business operations and the importance of adhering to principles that maintain security, productivity, and legal compliance.\nAcceptable Use: Company-provided internet and email services are primarily meant for job-related tasks. Limited personal use is allowed during non-work hours, provided it doesn't interfere with work responsibilities.\nSecurity: Safeguard your login credentials, avoiding the sharing of passwords. Exercise caution with email attachments and links from unknown sources. Promptly report any unusual online activity or potential security breaches.\nConfidentiality: Reserve email for the transmission of confidential information, trade secrets, and sensitive customer data only when encryption is applied. Exercise discretion when discussing company matters on public forums or social media.\nHarassment and Inappropriate Content: Internet and email usage must not involve harassment, discrimination, or the distribution of offensive or inappropriate content. Show respect and sensitivity to others in all online communications.\nCompliance: Ensure compliance with all relevant laws and regulations regarding internet and email usage, including those related to copyright and data protection.\nMonitoring: The company retains the right to monitor internet and email usage for security and compliance purposes.\nConsequences: Policy violations may lead to disciplinary measures, including potential termination.\nOur Internet and Email Policy aims to promote safe, responsible usage of digital communication tools that align with our values and legal obligations. Each employee is expected to understand and follow this policy. Regular reviews ensure its alignment with evolving technology and security standards.\n\n4.\tMobile Phone Policy\n\nThe Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance.\nAcceptable Use: Mobile devices are primarily intended for work-related tasks. Limited personal usage is allowed, provided it does not disrupt work obligations.\nSecurity: Safeguard your mobile device and access credentials. Exercise caution when downloading apps or clicking links from unfamiliar sources. Promptly report security concerns or suspicious activities related to your mobile device.\nConfidentiality: Avoid transmitting sensitive company information via unsecured messaging apps or emails. Be discreet when discussing company matters in public spaces.\nCost Management: Keep personal phone usage separate from company accounts and reimburse the company for any personal charges on company-issued phones.\nCompliance: Adhere to all pertinent laws and regulations concerning mobile phone usage, including those related to data protection and privacy.\nLost or Stolen Devices: Immediately report any lost or stolen mobile devices to the IT department or your supervisor.\nConsequences: Non-compliance with this policy may lead to disciplinary actions, including the potential loss of mobile phone privileges.\nThe Mobile Phone Policy is aimed at promoting the responsible and secure use of mobile devices in line with legal and ethical standards. Every employee is expected to comprehend and abide by these guidelines. Regular reviews of the policy ensure its ongoing alignment with evolving technology and security best practices.\n\n5.\tSmoking Policy\n\nPolicy Purpose: The Smoking Policy has been established to provide clear guidance and expectations concerning smoking on company premises. This policy is in place to ensure a safe and healthy environment for all employees, visitors, and the general public.\nDesignated Smoking Areas: Smoking is only permitted in designated smoking areas, as marked by appropriate signage. These areas have been chosen to minimize exposure to secondhand smoke and to maintain the overall cleanliness of the premises.\nSmoking Restrictions: Smoking inside company buildings, offices, meeting rooms, and other enclosed spaces is strictly prohibited. This includes electronic cigarettes and vaping devices.\nCompliance with Applicable Laws: All employees and visitors must adhere to relevant federal, state, and local smoking laws and regulations.\nDisposal of Smoking Materials: Properly dispose of cigarette butts and related materials in designated receptacles. Littering on company premises is prohibited.\nNo Smoking in Company Vehicles: Smoking is not permitted in company vehicles, whether they are owned or leased, to maintain the condition and cleanliness of these vehicles.\nEnforcement and Consequences: All employees and visitors are expected to adhere to this policy. Non-compliance may lead to appropriate disciplinary action, which could include fines, or, in the case of employees, possible termination of employment.\nReview of Policy: This policy will be reviewed periodically to ensure its alignment with evolving legal requirements and best practices for maintaining a healthy and safe workplace.\nWe appreciate your cooperation in maintaining a smoke-free and safe environment for all.\n\n6.\tDrug and Alcohol Policy\n\nPolicy Objective: The Drug and Alcohol Policy is established to establish clear expectations and guidelines for the responsible use of drugs and alcohol within the organization. This policy aims to maintain a safe, healthy, and productive workplace.\nProhibited Substances: The use, possession, distribution, or sale of illegal drugs or unauthorized controlled substances is strictly prohibited on company premises or during work-related activities. This includes the misuse of prescription drugs.\nAlcohol Consumption: The consumption of alcoholic beverages is not allowed during work hours, on company property, or while performing company-related duties. Exception may be made for company-sanctioned events.\nImpairment: Employees are expected to perform their job duties without impairment from drugs or alcohol. The use of substances that could impair job performance or pose a safety risk is prohibited.\nTesting and Searches: The organization reserves the right to conduct drug and alcohol testing as per applicable laws and regulations. Employees may be subject to testing in cases of reasonable suspicion, post-accident, or as part of routine workplace safety measures.\nReporting: Employees should report any concerns related to drug or alcohol misuse by themselves or their colleagues, as well as safety concerns arising from such misuse.\nTreatment and Assistance: Employees with substance abuse issues are encouraged to seek help. The organization is committed to providing support, resources, and information to assist those seeking treatment.\nConsequences: Violation of this policy may result in disciplinary actions, up to and including termination of employment. Legal action may also be pursued when necessary.\nPolicy Review: This policy will undergo periodic review to ensure its continued relevance and compliance with evolving legal requirements and best practices for a safe and productive work environment.\nYour adherence to this policy is appreciated as it helps to maintain a safe and drug-free workplace for all.\n\n7.\tHealth and Safety Policy\n\nOur commitment to health and safety is paramount. We prioritize the well-being of our employees, customers, and the public. We diligently comply with all relevant health and safety laws and regulations. Our objective is to maintain a workplace free from hazards, preventing accidents, injuries, and illnesses. Every individual within our organization is responsible for upholding these standards. We regularly assess and improve our safety measures, provide adequate training, and encourage open communication regarding safety concerns. Through collective dedication, we aim to ensure a safe, healthy, and secure environment for all. Your cooperation is essential in achieving this common goal.\n\n8.\tAnti-discrimination and Harassment Policy\n\nThe Anti-Discrimination and Harassment Policy is a testament to the commitment of this organization in fostering a workplace that is free from discrimination, harassment, and any form of unlawful bias. This policy applies to every individual within the organization, including employees, contractors, visitors, and clients.\nNon-Discrimination: This organization strictly prohibits discrimination based on race, color, religion, gender, national origin, age, disability, sexual orientation, or any other legally protected characteristic in all aspects of employment, including recruitment, hiring, compensation, benefits, promotions, and terminations.\nHarassment: Harassment in any form, whether based on the aforementioned characteristics or any other protected status, is unacceptable. This encompasses unwelcome advances, offensive jokes, slurs, and other verbal or physical conduct that creates a hostile or intimidating work environment.\nReporting: Individuals who experience or witness any form of discrimination or harassment are encouraged to promptly report the incident to their supervisor, manager, or the designated HR representative. The organization is committed to a timely and confidential investigation of such complaints.\nConsequences: Violation of this policy may result in disciplinary action, including termination of employment. The organization is committed to taking appropriate action against any individual found to be in violation of this policy.\nReview and Update: This policy is subject to regular review and update to remain aligned with evolving legal requirements and best practices in preventing discrimination and harassment. This organization considers it a collective responsibility to ensure a workplace free from discrimination and harassment, and it is essential that every individual within the organization plays their part in upholding these principles.\n\n9.\tDiscipline and Termination Policy\n\nThe Discipline and Termination Policy underscores the organization's commitment to maintaining a productive, ethical, and respectful work environment. This policy applies to all personnel, including employees, contractors, and temporary staff.\nPerformance and Conduct Expectations: Employees are expected to meet performance standards and adhere to conduct guidelines. The organization will provide clear expectations, feedback, and opportunities for improvement when performance or conduct issues arise.\nDisciplinary Actions: When necessary, disciplinary actions will be taken, which may include verbal warnings, written warnings, suspension, or other appropriate measures. Disciplinary actions are designed to address issues constructively and maintain performance standards.\nTermination: In situations where an employee's performance or conduct issues persist, the organization may resort to termination. Termination may also occur for reasons such as redundancy, violation of policies, or restructuring.\nTermination Procedure: The organization will follow appropriate procedures, ensuring fairness and adherence to legal requirements during the termination process. Employees may be eligible for notice periods, severance pay, or other benefits as per employment agreements and applicable laws.\nExit Process: The organization will conduct an exit process to ensure a smooth transition for departing employees, including the return of company property, final pay, and cancellation of access and benefits.\nThis policy serves as a framework for handling discipline and termination. The organization recognizes the importance of fairness and consistency in these processes, and decisions will be made after careful consideration. Every employee is expected to understand and adhere to this policy, contributing to a respectful and productive workplace. Regular reviews will ensure its alignment with evolving legal requirements and best practices.\n")]



Split `txt_data` into chunks. `chunk_size = 200`, `chunk_overlap = 20` has been set.



```python
chunks_txt = text_splitter(txt_data, 200, 20)
```

Store the embeddings into a `ChromaDB`.



```python
from langchain.vectorstores import Chroma
```


```python
vectordb = Chroma.from_documents(chunks_txt, watsonx_embedding())
```

##### Simple similarity search


Here is an example of a simple similarity search based on the vector database.

For this demonstration, the query has been set to "email policy".



```python
query = "email policy"
retriever = vectordb.as_retriever()
```


```python
docs = retriever.invoke(query)
```

By default, the number of retrieval results is four, and they are ranked by similarity level.



```python
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='Accountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Equal Opportunity: We are an equal opportunity employer and do not discriminate on the basis of race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='smoke and to maintain the overall cleanliness of the premises.'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Employee Referrals: We encourage and appreciate employee referrals as they contribute to building a strong and engaged team.')]



You can also specify `search kwargs` like `k` to limit the retrieval results.



```python
retriever = vectordb.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke(query)
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='Accountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential')]



##### MMR search


MMR in vector stores is a technique used to balance the relevance and diversity of retrieved results. It selects documents that are both highly relevant to the query and minimally similar to previously selected documents. This approach helps to avoid redundancy and ensures a more comprehensive coverage of different aspects of the query.


The following code is showing how to conduct an MMR search in a vector database. You just need to sepecify `search_type="mmr"`.



```python
retriever = vectordb.as_retriever(search_type="mmr")
docs = retriever.invoke(query)
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='Accountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Equal Opportunity: We are an equal opportunity employer and do not discriminate on the basis of race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='smoke and to maintain the overall cleanliness of the premises.'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Your adherence to this policy is appreciated as it helps to maintain a safe and drug-free workplace for all.')]



##### Similarity score threshold retrieval


You can also set a retrieval method that defines a similarity score threshold, returning only documents with a score above that threshold.



```python
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
)
docs = retriever.invoke(query)
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='Accountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Equal Opportunity: We are an equal opportunity employer and do not discriminate on the basis of race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='smoke and to maintain the overall cleanliness of the premises.'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Employee Referrals: We encourage and appreciate employee referrals as they contribute to building a strong and engaged team.')]



#### Multi-Query Retriever


Distance-based vector database retrieval represents queries in high-dimensional space and finds similar embedded documents based on "distance". However, retrieval results may vary with subtle changes in query wording or if the embeddings do not accurately capture the data's semantics.

The `MultiQueryRetriever` addresses this by using an LLM to generate multiple queries from different perspectives for a given user input query. For each query, it retrieves a set of relevant documents and then takes the unique union of these results to form a larger set of potentially relevant documents. By generating multiple perspectives on the same question, the `MultiQueryRetriever` can potentially overcome some limitations of distance-based retrieval, resulting in a richer and more diverse set of results.


The following picture shows the difference between retrievers solely based on distance and the Multi-Query Retriever.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/NCZCJ26bp3uKTa0gp8Agwg/multiquery.png" width="40%" alt="multiquery"/>


Let's consider the query sentence, `"I like cats"`.

On the upper side of the picture, you can see a retriever that relies solely on distance. This retriever calculates the distance between the query and the documents in the vector store, returning the document with the closest match.

On the lower side, you can see a multi-query retriever. It first uses an LLM to generate multiple queries from different perspectives based on the user's input query. For each generated query, it retrieves relevant documents and then returns the union of these results.


A PDF document has been prepared to demonstrate this Multi-Query Retriever.



```python
from langchain_community.document_loaders import PyPDFLoader
```


```python
loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf")
pdf_data = loader.load()
```

Let's take a look at the first page of this paper. This paper is talking about the LangChain framework.



```python
pdf_data[1]
```




    Document(metadata={'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf', 'page': 1}, page_content='LangChain helps us to unlock the ability to harness the \nLLM’s immense potential in tasks such as document analysis, \nchatbot development, code analysis, and countless other \napplications. Whether your desire is to unlock deeper natural \nlanguage understanding , enhance data, or circumvent \nlanguage barriers through translation, LangChain is ready to \nprovide the tools and programming support you need to do \nwithout it that it is not only difficult but also fresh for you . Its \ncore functionalities encompass:  \n1. Context -Aware Capabilities: LangChain facilitates the \ndevelopment of applications that are inherently \ncontext -aware. This means that these applications can \nconnect to a language model and draw from various \nsources of context, such as prompt instructions, a  few-\nshot examples, or existing content, to ground their \nresponses effectively.  \n2. Reasoning Abilities: LangChain equips applications \nwith the capacity to reason effectively. By relying on a \nlanguage model, these applications can make informed \ndecisions about how to respond based on the provided \ncontext and determine the appropriate acti ons to take.  \nLangChain offers several key value propositions:  \nModular Components: It provides abstractions that \nsimplify working with language models, along with a \ncomprehensive collection of implementations for each \nabstraction. These components are designed to be modular \nand user -friendly, making them useful whethe r you are \nutilizing the entire LangChain framework or not.  \nOff-the-Shelf Chains: LangChain offers pre -configured \nchains, which are structured assemblies of components \ntailored to accomplish specific high -level tasks. These pre -\ndefined chains streamline the initial setup process and serve as \nan ideal starting point  for your projects. The MindGuide Bot \nuses below components from LangChain . \nA. ChatModel  \nWithin LangChain, a ChatModel is a specific kind of \nlanguage model crafted to manage conversational \ninteractions. Unlike traditional language models that take one \nstring as input and generate a single string as output, \nChatModels operate with a list of mes sages as input, \ngenerating a message as output.  \nEach message in the list has two parts: the content and the \nrole. The content is the actual text or substance of the message, \nwhile the role denotes the role or source of the message (such \nas "User," "Assistant," "System," etc.).  \nThis approach with ChatModels opens the door to more \ndynamic and interactive conversations with the language \nmodel. It empowers the creation of chatbot applications, \ncustomer support systems, or any other application involving \nmulti -turn conversations. We utilized the ChatOpenAI \nChatModel to create MindGuide chatbots specifically \ndesigned to function as mental health therapists. In our \ninteraction with OpenAI, we opted for an OpenAI API key to \nengage with the ChatGpt3 turbo model and utilized a \ntemperature value of 0.5. The steps to create an OpenAI API \nkey are outlined [ 9].  B. Message  \nIn the context of LangChain, messages  [10] refer to a list of \nmessages that are used as input when interacting with a \nChatModel. Each message in the list represents a specific turn \nor exchange in a conversation.  Each message in the messages \nlist typically consists of two components:  \n• content: This represents the actual text or content of \nthe message. It can be a user query, a system \ninstruction, or any other relevant information.  \n• role: This represents the role or source of the \nmessage. It defines who is speaking or generating \nthe message. Common roles include "User", \n"Assistant", "System", or any other custom role you \ndefine . \nThe chat model interface is based around messages rather \nthan raw text. The types of messages supported in LangChain \nare Systen Message, HumanMessage, and AIMessage . \nSystemMessage  is the ChatMessage coming from the system  \nin its LangChain template  as illustrated in Figure 1. Human \nMessage  is a ChatMessage coming from a human/user.  \nAIMessage is a ChatMessage  coming from an AI/assistant as \nillustrated in Figure 2 .  \n \n                   Figure 1. A System Message illustration   You are a compassionate and experienced mental \nhealth therapist with a proven track record of \nhelping patients overcome anxiety and other mental \nhealth challenges. Your primary objective is to \nsupport the patient in addressing their concerns \nand guiding th em towards positive change. In this \ninteractive therapy session, you will engage with \nthe patient by asking open -ended questions, \nactively listening to their responses, and providing \nempathetic feedback. Your approach is \ncollaborative, and you strive to cr eate a safe and \nnon-judgmental space for the patient to share their \nthoughts and feelings.  \nAs the patient shares their struggles, you will \nprovide insightful guidance and evidence -based \nstrategies tailored to their unique needs. You may \nalso offer practical exercises or resources to help \nthem manage their symptoms and improve their \nmental wellbe ing. When necessary, you will gently \nredirect the conversation back to the patient\'s \nprimary concerns related to anxiety, mental health, \nor family issues. This ensures that each session is \nproductive and focused on addressing the most \npressing issues. Thro ughout the session, you \nremain mindful of the patient\'s emotional state and \nadjust your approach accordingly.  \nYou recognize that everyone\'s journey is \ndifferent, and that progress can be incremental.  \nBy building trust and fostering a strong \ntherapeutic relationship, you empower the patient \nto take ownership of their growth and development. \nAt the end of the session, you will summarize key \npoints from your discussion, highlighting the \npatient\'s strength s and areas for improvement.  \nTogether, you will set achievable goals for future \nsessions, reinforcing a sense of hope and \nmotivation. Your ultimate goal is to equip the \npatient with the tools and skills needed to navigate \nlife\'s challenges with confidence and resilience . ')



Split the document and store the embeddings into a vector database.



```python
# Split
chunks_pdf = text_splitter(pdf_data, 500, 20)

# VectorDB
ids = vectordb.get()["ids"]
vectordb.delete(ids) # We need to delete existing embeddings from previous documents and then store current document embeddings in.
vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
```

The `MultiQueryRetriever` function from LangChain is used.



```python
from langchain.retrievers.multi_query import MultiQueryRetriever

query = "What does the paper say about langchain?"

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm()
)
```

Set logging for the queries.



```python
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```


```python
docs = retriever.invoke(query)
docs
```

    INFO:langchain.retrievers.multi_query:Generated queries: ['1. What information does the paper provide regarding langchain?', '    2. Can you summarize the content related to langchain mentioned in the paper?', '    3. What are the key points about langchain discussed in the paper?']





    [Document(metadata={'page': 4, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content="question (Fig. 4b).  \n• MindGuide Chatbot's AI response to the \nsubsequent human message, followed by another \nmental health question from the human (Fig. 4c).  \n• MindGuide Chatbot's AI response after \nanalyzing the latest human message (Fig. 4d).  \n \n   s \n                                                         (a)      (b) \n      \n                                                         (c)      (d)"),
     Document(metadata={'page': 3, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content="and the chatbot's responses, allowing for a \ndynamic and coherent conversation flow.  \n• Chatmodel Class of LangChain : The LangChain \nframework leverages the Chatmodel  class, a \ncritical component for interfacing with the \nOpenAI model (GPT -4) for making requests to \nthe language model and processing its \nresponses, ensuring seamless communication \nbetween the chatbot and the AI model . \n• Memory Concept:  To enhance the chatbot's \nconversational capabilities and provide context -"),
     Document(metadata={'page': 1, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content='• content: This represents the actual text or content of \nthe message. It can be a user query, a system \ninstruction, or any other relevant information.  \n• role: This represents the role or source of the \nmessage. It defines who is speaking or generating \nthe message. Common roles include "User", \n"Assistant", "System", or any other custom role you \ndefine . \nThe chat model interface is based around messages rather \nthan raw text. The types of messages supported in LangChain'),
     Document(metadata={'page': 2, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content="single run, a chain will interact with its memory system twice.  \n1. A chain will READ from its memory system and \naugment the user inputs AFTER receiving the initial \nuser inputs but BEFORE performing the core logic . \n2. After running the basic logic but before providing the \nsolution, a chain will WRITE the current run's inputs \nand outputs to memory so that they may be referred \nto in subsequent runs.  \nAny memory system's two primary design decisions are:  \n1. How state is stored ?"),
     Document(metadata={'page': 0, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content='* corresponding author - jkim72@kent.edu  Revolutionizing Mental Health Care through \nLangChain: A Journey with a Large Language \nModel\nAditi Singh  \n Computer Science   \n Cleveland State University   \n a.singh22@csuohio.edu  Abul Ehtesham   \nThe Davey Tree Expert \nCompany   \nabul.ehtesham@davey.com  Saifuddin Mahmud  \nComputer Science & \nInformation Systems   \n Bradley University  \nsmahmud@bradley.edu   Jong -Hoon Kim * \n Computer Science,  \nKent State University,  \njkim72@kent.edu'),
     Document(metadata={'page': 2, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content='problematic component. By isolating the chain and \ntesting each component individually, you can \nidentify and troubleshoot any errors or unexpected \nbehavior . \n• Maintenance: Chains make it easier to update or \nreplace specific components without affecting the \nentire application. If a new version of a component \nbecomes available or if you want to switch to a \ndiffer.  \nTo build a chain, you simply combine the desired components \nin the order they should be executed. Each component in the')]



From the log results, you can see that the LLM generated three additional queries from different perspectives based on the given query.

The returned results are the union of the results from each query.


#### Self-Querying Retriever


A Self-Querying Retriever, as the name suggests, has the ability to query itself. Specifically, given a natural language query, the retriever uses a query-constructing LLM chain to generate a structured query. It then applies this structured query to its underlying vector store. This enables the retriever to not only use the user-input query for semantic similarity comparison with the contents of stored documents but also to extract and apply filters based on the metadata of those documents.


The following code demonstrates how to use a Self-Querying Retriever.



```python
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from lark import lark
```

A couple of document pieces have been prepared where the `page_content` contains descriptions of movies, and the `meta_data` includes different attributes for each movie, such as `year`, `rating`, `genre`, and `director`. These attributes are crucial in the Self-Querying Retriever, as the LLM will use the metadata information to apply filters during the retrieval process.



```python
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
```

Now you can instantiate your retriever. To do this, you'll need to provide some upfront information about the metadata fields that your documents support and a brief description of the document contents.



```python
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
```

Store the document's embeddings into a vector database.



```python
vectordb = Chroma.from_documents(docs, watsonx_embedding())
```

Use the `SelfQueryRetriever`.



```python
document_content_description = "Brief summary of a movie."

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)
```

Now you can actually try using your retriever.



```python
# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")
```




    [Document(metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006}, page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea'),
     Document(metadata={'director': 'Andrei Tarkovsky', 'genre': 'thriller', 'rating': 9.9, 'year': 1979}, page_content='Three men walk into the Zone, three men walk out of the Zone')]




```python
# This example specifies a query and a filter
retriever.invoke("Has Greta Gerwig directed any movies about women")
```




    [Document(metadata={'director': 'Greta Gerwig', 'rating': 8.3, 'year': 2019}, page_content='A bunch of normal-sized women are supremely wholesome and some men pine after them')]



When running the following cell, you might encounter some errors or blank content. This is because the LLM cannot get the answer at first. Don't worry; if you re-run it several times, you will get the answer.



```python
# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
```




    [Document(metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006}, page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea'),
     Document(metadata={'director': 'Andrei Tarkovsky', 'genre': 'thriller', 'rating': 9.9, 'year': 1979}, page_content='Three men walk into the Zone, three men walk out of the Zone')]



#### Parent Document Retriever


When splitting documents for retrieval, there are often conflicting desires:

1. You may want to have small documents so that their embeddings can most accurately reflect their meaning. If the documents are too long, the embeddings can lose meaning.
2. You want to have long enough documents so that the context of each chunk is retained.

The `ParentDocumentRetriever` strikes that balance by splitting and storing small chunks of data. During retrieval, it first fetches the small chunks but then looks up the parent IDs for those chunks and returns those larger documents.



```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore
```


```python
# Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
parent_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='\n')
```


```python
vectordb = Chroma(
    collection_name="split_parents", embedding_function=watsonx_embedding()
)
#vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
# The storage layer for the parent documents
store = InMemoryStore()
```


```python
retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```


```python
retriever.add_documents(txt_data)
```

    WARNING:langchain_text_splitters.base:Created a chunk of size 223, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 274, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 262, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 282, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 262, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 270, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 224, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 325, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 300, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 216, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 226, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 235, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 300, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 294, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 234, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 321, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 256, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 241, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 248, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 249, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 246, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 211, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 267, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 206, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 694, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 323, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 326, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 296, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 233, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 421, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 243, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 260, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 229, which is longer than the specified 200
    WARNING:langchain_text_splitters.base:Created a chunk of size 290, which is longer than the specified 200


Ignore the warnings as the documents before chunking have variable lengths.

These are the number of large chunks:



```python
len(list(store.yield_keys()))
```




    19



Let's make sure the underlying vector store still retrieves the small chunks.



```python
sub_docs = vectordb.similarity_search("smoking policy")
```


```python
print(sub_docs[0].page_content)
```

    Smoking Restrictions: Smoking inside company buildings, offices, meeting rooms, and other enclosed spaces is strictly prohibited. This includes electronic cigarettes and vaping devices.


Then, retrieve the relevant large chunk.



```python
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)
```

    5.	Smoking Policy
    Policy Purpose: The Smoking Policy has been established to provide clear guidance and expectations concerning smoking on company premises. This policy is in place to ensure a safe and healthy environment for all employees, visitors, and the general public.
    Designated Smoking Areas: Smoking is only permitted in designated smoking areas, as marked by appropriate signage. These areas have been chosen to minimize exposure to secondhand smoke and to maintain the overall cleanliness of the premises.
    Smoking Restrictions: Smoking inside company buildings, offices, meeting rooms, and other enclosed spaces is strictly prohibited. This includes electronic cigarettes and vaping devices.
    Compliance with Applicable Laws: All employees and visitors must adhere to relevant federal, state, and local smoking laws and regulations.


# Exercises


### Exercise 1
### Retrieve Top 2 Results Using a Vector Store-Backed Retriever

Retrieve the top two results for the company policy document for the query "smoking policy" using the Vector Store-Backed Retriever.



```python
vectordb = Chroma.from_documents(documents=chunks_txt, embedding=watsonx_embedding())
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
query = "smoking policy"
docs = retriever.invoke(query)
docs
```




    [Document(metadata={'page': 2, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content='problematic component. By isolating the chain and \ntesting each component individually, you can \nidentify and troubleshoot any errors or unexpected \nbehavior . \n• Maintenance: Chains make it easier to update or \nreplace specific components without affecting the \nentire application. If a new version of a component \nbecomes available or if you want to switch to a \ndiffer.  \nTo build a chain, you simply combine the desired components \nin the order they should be executed. Each component in the'),
     Document(metadata={'page': 4, 'source': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf'}, page_content="question (Fig. 4b).  \n• MindGuide Chatbot's AI response to the \nsubsequent human message, followed by another \nmental health question from the human (Fig. 4c).  \n• MindGuide Chatbot's AI response after \nanalyzing the latest human message (Fig. 4d).  \n \n   s \n                                                         (a)      (b) \n      \n                                                         (c)      (d)")]



<details>
    <summary>Click here for the solution</summary>

```python

vectordb = Chroma.from_documents(documents=chunks_txt, embedding=watsonx_embedding())
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
query = "smoking policy"
docs = retriever.invoke(query)
docs
```

</details>


### Exercise 2
### Self-Querying Retriever for a Query


Use the Self-Querying Retriever to invoke a query with a filter.



```python
# You might encouter some errors or blank content when run the following code.
# It is becasue LLM cannot get the answer at first. Don't worry, re-run it several times you will get the answer.

vectordb = Chroma.from_documents(docs, watsonx_embedding())

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)

# This example specifies a query with filter
retriever.invoke(
    "I want to watch a movie directed by Christopher Nolan"
)
```




    [Document(metadata={'director': 'Christopher Nolan', 'rating': 8.2, 'year': 2010}, page_content='Leo DiCaprio gets lost in a dream within a dream within a dream within a ...')]



<details>
    <summary>Click here for the solution</summary>

```python

# You might encouter some errors or blank content when run the following code.
# It is becasue LLM cannot get the answer at first. Don't worry, re-run it several times you will get the answer.

vectordb = Chroma.from_documents(docs, watsonx_embedding())

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)

# This example specifies a query with filter
retriever.invoke(
    "I want to watch a movie directed by Christopher Nolan"
)
```

</details>


## Authors


[Kang Wang](https://author.skills.network/instructors/kang_wang) is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.

[Fateme Akbari](https://author.skills.network/instructors/fateme_akbari) is a Ph.D. candidate in Information Systems at McMaster University and data scientist at IBM with demonstrated research experience in Machine Learning and NLP.


### Other Contributors


[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo) has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

[Ricky Shi](https://author.skills.network/instructors/ricky_shi) is a Data Scientist at IBM, specializing in deep learning, computer vision, and Large Language Models. He applies advanced machine learning and generative AI techniques to solve complex challenges across various sectors. As an enthusiastic mentor, Ricky is committed to helping colleagues and peers master technical intricacies and drive innovation.

[Wojciech "Victor" Fulmyk](https://author.skills.network/instructors/wojciech_fulmyk) is a Data Scientist at IBM


## Change Log


<details>
    <summary>Click here for the changelog</summary>

|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2024-07-29|0.1|Kang Wang|Create the lab|
|2024-09-06|0.2|Fateme Akbari|Revised the lab|
|2025-06-24|0.3|Steve Ryan|ID review/typo/format fixes|
|2025-06-25|0.4|Mercedes Schneider|QA pass with edits|
|2025-07-25|0.5|Wojciech "Victor" Fulmyk|Fixed warning from chromadb and changed model from mixtral 8x7b because that is slated for deprecation|
|2025-10-31|0.6|Joshua Zhou|Updated slate embedding model to non-deprecated version|

</details>


Copyright © IBM Corporation. All rights reserved.

