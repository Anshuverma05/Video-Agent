from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.retry_utils import call_with_retry

import os

def get_llm():
    model = os.getenv("MISTRAL_MODEL", "open-mistral-7b")
    return ChatMistralAI(model=model, mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.3)



def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

import time

def summarize(transcript : str) -> str:
    llm = get_llm()

    # For shorter transcripts (under 4000 chars), summarize directly in 1 call to save rate limit quota
    if len(transcript) <= 4000:
        direct_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert meeting summarizer. Provide a comprehensive professional meeting summary in bullet points."),
            ("human", "{text}"),
        ])
        direct_chain = (
            RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | direct_prompt | llm | StrOutputParser()
        )
        return call_with_retry(direct_chain.invoke, transcript)

    # For longer transcripts, split and map-reduce with retries and pacing
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | map_prompt | llm | StrOutputParser()
    )

    chunks = split_transcript(transcript)
    chunk_summaries = []
    for chunk in chunks:
        chunk_sum = call_with_retry(map_chain.invoke, chunk)
        chunk_summaries.append(chunk_sum)
        time.sleep(2)  # Avoid bursting the Mistral rate limit

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return call_with_retry(combined_chain.invoke, combined)

def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return call_with_retry(title_chain.invoke, transcript[:2000])