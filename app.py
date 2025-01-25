import os
from apikey import gpt_apikey

import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# for album cover
import requests

os.environ['OPENAI_API_KEY'] = gpt_apikey

#Prompt templates
title_template = PromptTemplate(
    input_variable = ['topic'],
    template='Come up with a title for this rap song: {topic}. Use markdown # for of the title'
)

songtext_template = PromptTemplate(
    input_variable = ['title', 'wiki_research'],
    template= """
        Write a rap song based on this title: {title} while leveraging this wikipedia research: {wiki_research}. Use markdown formatting with the following rules:
        1. Each verse is preceded by `### Verse` followed by a space.
        2. Each chorus is preceded by `### Chorus` followed by a space.
        3. Insert a blank line after each line of the rap for proper markdown rendering.
        4. Ensure every section starts with the correct markdown header and spacing.

        The output should look like this:

        ### Verse 1  
        First line of the verse.  
        Second line of the verse.  

        ### Chorus  
        First line of the chorus.  
        Second line of the chorus.  

        ### Verse 2  
        First line of the next verse.  
        Second line of the next verse.  

        Follow this structure exactly. Don't name the title in the beginning. Now, write the rap!
        """)
    #'Write a rap song based on this title: {title}. Format the text using Markdown. Structure the song with verses and choruses, marking each with ### followed by a single space. Ensure every line ends with \\n Here is an example of the structure: ### Verse 1\\n Line one of the verse.\\n Line two of the verse.\\n ### Chorus\\n First line of the chorus.\\n Second line of the chorus.\\n')

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
songtext_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs and Chains
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title', memory=title_memory, verbose=True) #verbose=True prints prompts
songtext_chain = LLMChain(llm=llm, prompt=songtext_template, output_key='songtext', memory=songtext_memory, verbose=True)

wiki = WikipediaAPIWrapper()

# sequential_chain = SequentialChain(chains=[title_chain, songtext_chain], input_variables=['topic'], output_variables=['title', 'songtext'], verbose=True)

# #App framework
st.title('👻 Rap Song Creator')
prompt = st.text_input('What do you want your rap song to be about?')

def generate_cover(title):
    prompt = f'Album Cover for a Rap Song with the title: {title} In 90s style with the title written on the mid to bottom section.'
    width = 768
    height = 768
    seed = 42
    model = 'flux'
    image_url = f"https://pollinations.ai/p/{prompt}?width={width}&height={height}&seed={seed}&model={model}"
    return image_url

# # Show response
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    songtext = songtext_chain.run(title=title, wiki_research=wiki_research)
    #response = sequential_chain({'topic':prompt})
    #cover = generate_cover(response['title'])
    #print(cover)
    #print(response)
    st.markdown(title)
    st.markdown(songtext)
    #st.image(cover)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Songtext History'):
        st.info(songtext_memory.buffer)

    with st.expander('Wikpedia Research History'):
        st.info(wiki_research)