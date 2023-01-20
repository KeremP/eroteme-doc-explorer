import os
import uuid
import json
import requests
import tarfile
from typing import Optional, Union
from dotenv import load_dotenv
import logging
from doc_utils import parse_tex, remove_latex
from decorators import timer_func
import pprint

pp = pprint.PrettyPrinter(indent=4)
load_dotenv()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import redis
from redis import UsernamePasswordCredentialProvider
import numpy as np

import hashlib
import urllib.request
from bs4 import BeautifulSoup

import openai
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

# REDIS CONFIG
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB   = 0

class CustomFAISS(FAISS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def decode_cached_embeddings(cache_result):
        return json.loads(cache_result)

    @classmethod
    def from_precomputed(cls, embedding, texts, metadatas, redis: redis.Redis, doc_hash, **kwargs) -> FAISS:
        try:
            import faiss
        except:
            raise ValueError(
                "Could not import faiss python package. "
                "Please it install it with `pip install faiss` "
                "or `pip install faiss-cpu` (depending on Python version)."
            )
        precomputed_embeddings = redis.get(doc_hash)
        if precomputed_embeddings is None:
            embeddings = embedding.embed_documents(texts)
            try:
                redis.set(doc_hash, json.dumps(embeddings))
            except Exception as e:
                print(f"Unable to cache embeddings in Redis: {e}")
        else:
            embeddings = CustomFAISS.decode_cached_embeddings(precomputed_embeddings)

        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        docs = [
            Document(page_content=text, metadata=metadatas[i]) for i,text in enumerate(texts)
        ]
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(docs))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(docs)}
        )
        return cls(embedding.embed_query, index, docstore, index_to_id)


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

def init_redis(host: str, port: int, db: int, username: Optional[str] = None, password: Optional[str] = None) -> Union[redis.Redis, None]:
    if username and password:
        cred_provider = UsernamePasswordCredentialProvider(username, password)
    else:
        cred_provider = None
    
    client = redis.Redis(host, port, db, credential_provider=cred_provider, decode_responses=False)
    try:
        pong = client.ping()
    except:
        pong = None
    
    if pong:
        print("Redis initialized")
        return client
    else:
        raise Exception("Unable to connect to Redis server")

def hash_doc(arxiv_id):
    return hashlib.md5(
        bytes(arxiv_id, 'utf-8')
    ).hexdigest()

@timer_func
def build_docstore(context, metadata, doc_hash, redis):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    # docsearch = FAISS.from_texts(context, embeddings, metadatas=metadata)
    docsearch = CustomFAISS.from_precomputed(embeddings, context, metadatas=metadata, redis=redis, doc_hash=doc_hash)
    return docsearch
 
@timer_func
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
    metadata = [{'source':f'{context_map[i]}'} for i in context_map.keys()]
    return context, context_map, metadata


def lambda_handler(event, context):
    REDIS_CLIENT: redis.Redis = init_redis(REDIS_HOST, REDIS_PORT, REDIS_DB)

    result = None
    action = event.get('ids')
    query = event.get('query')

    id_tex_map = {}
    doc_hashes = []
    for id in action:
       clean_id = get_gzip_source_files(id)
       files = os.listdir(os.path.join("./temp", clean_id))
       tex_path = [f for f in files if ".tex" in f][0]
       tex_path_dir = os.path.join("./temp",clean_id,tex_path)
       parsed_tex = parse_tex(tex_path_dir)
       id_tex_map.update({clean_id:parsed_tex})
       doc_hashes.append(
        hash_doc(id)
       )

    context, context_map, metadata = build_context(id_tex_map[list(id_tex_map.keys())[0]])
    
    # pp.pprint(context_map)
    docstore = build_docstore(context, metadata, doc_hashes[0], REDIS_CLIENT)
    
    chain = VectorDBQAWithSourcesChain.from_llm(OpenAI(temperature=0), vectorstore=docstore)
    result = chain({
        "question":query
    }, return_only_outputs=True)

    resp = {'result':result}

    return resp


    