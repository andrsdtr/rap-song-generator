import os
from apikey import gpt_apikey

import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# for album cover
import requests

os.environ['OPENAI_API_KEY'] = gpt_apikey

#Prompt templates
title_template = PromptTemplate(
    input_variable = ['topic'],
    template='Come up with a title for this rap song: {topic}. Use markdown # for of the title'
)

songtext_template = PromptTemplate(
    input_variable = ['title'],
    template='Write a rap song based on this title: {title}. Limit output to 90,000 tokens. Use markdown ### for the titles of verse and chorus and leave a space after the ###. Be sure to end every line with \n'
)

# LLMs and Chains
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title', verbose=True) #verbose=True prints prompts
songtext_chain = LLMChain(llm=llm, prompt=songtext_template, output_key='songtext', verbose=True)
sequential_chain = SequentialChain(chains=[title_chain, songtext_chain], input_variables=['topic'], output_variables=['title', 'songtext'], verbose=True)

# #App framework
st.title('👻 Rap Song Creator')
prompt = st.text_input('What do you want your rap song to be about?')

def generate_cover(title):
    prompt = f'Album Cover for the Rap Song with the {title} in 90s style with the title written on the mid to bottom section.'
    width = 768
    height = 768
    seed = 42
    model = 'flux'
    image_url = f"https://pollinations.ai/p/{prompt}?width={width}&height={height}&seed={seed}&model={model}"
    return image_url

# # Show response
if prompt:
    response = sequential_chain({'topic':prompt})
    cover = generate_cover(response['title'])
    print(cover)
    print(response)
    st.markdown(response['title'])
    st.markdown(response['songtext'])
    st.image(cover)