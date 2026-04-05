from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )

# Supported models: ['ibm/granite-4-h-small', 'ibm/granite-8b-code-instruct', 'ibm/granite-guardian-3-8b', 'meta-llama/llama-3-2-11b-vision-instruct', 'meta-llama/llama-3-3-70b-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8', 'meta-llama/llama-guard-3-11b-vision', 'mistralai/mistral-medium-2505', 'mistralai/mistral-small-3-1-24b-instruct-2503', 'openai/gpt-oss-120b']

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
	GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

model = ModelInference(
    model_id='ibm/granite-8b-code-instruct',
    # model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

# text = """
# Only reply with the answer. What is the capital of Canada?
# """

# # Llama special tokens
# text = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are an expert assistant who provides concise and accurate answers.<|eot_id|>

# <|start_header_id|>user<|end_header_id|>
# What is the capital of Canada?<|eot_id|>

# <|start_header_id|>assistant<|end_header_id|>
# """

# ibm:granite special tokens
text = """
<|system|> You are an expert assistant who provides concise and accurate answers. <|user|> What is the capital of France? <|assistant|> Mistral:

[INST] You are a highly specialized assistant. You only provide answers based on facts and logic. [/INST]

[INST] What is the capital of France? [/INST]

Granite:
"""


print(model.generate(text)['results'][0]['generated_text'])
