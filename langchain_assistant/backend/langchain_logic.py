# langchain_logic.py
import transformers
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# --- Setup model ---
transformers.utils.logging.set_verbosity_error()
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=300)
local_llm = HuggingFacePipeline(pipeline=pipe)

# --- Define prompts ---
explain_prompt = PromptTemplate.from_template("Explain {topic} in simple words.")
adv_prompt = PromptTemplate.from_template("List 3 advantages of {topic}.")
example_prompt = PromptTemplate.from_template("Give one real-world example of using {topic}.")

# --- Define function ---
def explain_topic(topic: str):#topic entered by user
    explanation = explain_prompt | local_llm  #call explain propmt
    advantages = adv_prompt | local_llm
    example = example_prompt | local_llm

    explain_text = explanation.invoke({"topic": topic})#does 3 thingsfills rext propmt template,,llm.invoke send text to mdel and get response back and result store in explain trxt
    adv_text = advantages.invoke({"topic": topic})
    example_text = example.invoke({"topic": topic})
# bundle all responses into one py dictionary and send to fastapi and route in app.py use it whenever explain_topic() get called in fast_api analyze_topic
    return {
        "topic": topic,
        "explanation": explain_text,
        "advantages": adv_text,
        "example": example_text
    }

