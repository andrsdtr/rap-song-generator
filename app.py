import os
from apikey import gpt_apikey

import streamlit as st
from langchain_openai import OpenAI

os.environ['OPENAI_API_KEY'] = gpt_apikey

# repo_id = "openai-community/gpt2"

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     huggingfacehub_api_token=huggingface_apikey2,
# )
llm = OpenAI()

# #App framework
st.title('👻 Historical Rap Lyric Creator')
prompt = st.text_input('Plug in your prompt here')


# # Showing responses
if prompt:
    answer = llm(prompt)
    st.write(answer)

