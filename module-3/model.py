from functools import lru_cache
from pydantic import BaseModel, Field
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID


class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry")
    action: str = Field(description="Recommended action for the support rep")
    response: str = Field(description="Suggested response to the user")


json_parser = JsonOutputParser(pydantic_object=AIResponse)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n\n{format_instructions}"),
    ("human", "{user_prompt}"),
]).partial(
    format_instructions=json_parser.get_format_instructions()
)


@lru_cache(maxsize=None)
def get_model(model_id: str) -> ChatWatsonx:
    return ChatWatsonx(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=PARAMETERS,
    )


def get_ai_response(model_id: str, system_prompt: str, user_prompt: str):
    chain = PROMPT | get_model(model_id) | json_parser
    return chain.invoke({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    })


def llama_response(system_prompt: str, user_prompt: str):
    return get_ai_response(LLAMA_MODEL_ID, system_prompt, user_prompt)


def granite_response(system_prompt: str, user_prompt: str):
    return get_ai_response(GRANITE_MODEL_ID, system_prompt, user_prompt)


def mistral_response(system_prompt: str, user_prompt: str):
    return get_ai_response(MISTRAL_MODEL_ID, system_prompt, user_prompt)