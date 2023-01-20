import os
import requests
import tarfile
from dotenv import load_dotenv
import logging
from doc_utils import parse_tex, remove_latex

load_dotenv()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import urllib.request
from bs4 import BeautifulSoup

import openai
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

def gunzip(path: str, temp_dir: str):
    file = tarfile.open(path)
    file.extractall(os.path.join("./temp", temp_dir))
    file.close()

def cleanup(path: str):
    os.remove(path)

def get_gzip_source_files(arxiv_id: str):
    base_url = "https://arxiv.org/e-print/"
    target = base_url+arxiv_id
    joined_id = "".join(arxiv_id.split("."))
    filename = joined_id + ".tar.gz"
    urllib.request.urlretrieve(target, filename)
    gunzip(filename, joined_id)
    cleanup(filename)
    return joined_id


def build_docstore(context, context_map):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    docsearch = FAISS.from_texts(context, embeddings)
    for i, d in enumerate(docsearch.docstore._dict.values()):
        d.metadata = {'source':f'{context_map[i]}'}
    return docsearch

def build_context(parsed_tex: dict):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    context = []
    context_map = {}
    for k,v in parsed_tex.items():
        temp_ctx = text_splitter.split_text(
            v
        )
        context_map.update({
            i+len(context):(k, temp_ctx[i]) for i in range(len(temp_ctx))
        })
        context+=temp_ctx
    return context, context_map


def lambda_handler(event, context):
    result = None
    action = event.get('ids')
    query = event.get('query')

    id_tex_map = {}
    for id in action:
       clean_id = get_gzip_source_files(id)
       files = os.listdir(os.path.join("./temp", clean_id))
       tex_path = [f for f in files if ".tex" in f][0]
       tex_path_dir = os.path.join("./temp",clean_id,tex_path)
       parsed_tex = parse_tex(tex_path_dir)
       id_tex_map.update({clean_id:parsed_tex})

    context, context_map = build_context(id_tex_map[list(id_tex_map.keys())[0]])
    
    docstore = build_docstore(context, context_map)
    
    chain = VectorDBQAWithSourcesChain.from_llm(OpenAI(temperature=0), vectorstore=docstore)
    result = chain({
        "question":query
    }, return_only_outputs=True)

    resp = {'result':result}

    return resp


    