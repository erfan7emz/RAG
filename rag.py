from langchain import hub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import LLMChain
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub
# Set up RetrievelQA model
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")

#load the LLM
def load_llm():
    llm = HuggingFaceHub(
        model_kwargs={"max_length": 1000},
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token="",
    )
    return llm


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    return qa_chain


def qa_bot(): 
    llm=load_llm() 
    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=HuggingFaceEmbeddings())

    qa = retrieval_qa_chain(llm, vectorstore)
    return qa 

@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content= "Hi, welcome to chatbot. What is your question?"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True
    # res=await chain.acall(message, callbacks=[cb])
    res=await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer=res["result"]
    answer=answer.replace(".",".\n")
    sources=res["source_documents"]

    if sources:
        answer+=f"\nSources: "+str(str(sources))
    else:
        answer+=f"\nNo Sources found"

    await cl.Message(content=answer).send() 