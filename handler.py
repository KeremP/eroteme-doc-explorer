import os
import sys
import uuid
import json
import tarfile
import subprocess
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

import urllib3
import redis
from redis import UsernamePasswordCredentialProvider
import numpy as np

import hashlib

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

USE_REDIS = bool(int(os.getenv("USE_REDIS")))

DEBUG = bool(int(os.getenv("DEBUG")))


class CustomFAISS(FAISS):
    """
    Extension of langchain `FAISS` class to use cached embeddings from a Redis instance

    """
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
        precomputed_embeddings = None
        if redis is not None:
            precomputed_embeddings = redis.get(doc_hash)
        if precomputed_embeddings is None:
            embeddings = embedding.embed_documents(texts)
            try:
                if redis is not None:
                    redis.set(doc_hash, json.dumps(embeddings))
            except Exception as e:
                logger.warning(f"Unable to cache embeddings in Redis: {e}")
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


def gunzip(path: str, temp_dir: str, sub_dir: str):
    """
    Unzip gzipped tarballs
    """
    file = tarfile.open(path)
    file.extractall(os.path.join(temp_dir, sub_dir))
    file.close()

def cleanup(path: str):
    os.remove(path)

def download_stream(resp, path):
    CHUNK_SIZE = 1024*8
    with open(path, 'wb') as out:
        while True:
            data = resp.read(CHUNK_SIZE)
            if not data:
                break
            out.write(data)
    resp.release_conn()

def get_gzip_source_files(arxiv_id: str, parent_path: str):
    """
    Download document source files from arxiv, save to tmp directory, unzip, and return name of directory where files are saved
    """
    base_url = "https://arxiv.org/e-print/"
    target = base_url+arxiv_id
    joined_id = "".join(arxiv_id.split("."))

    filename = parent_path + joined_id + ".tar.gz"
    http = urllib3.PoolManager()
    resp = http.request('GET', target, preload_content=False)
    download_stream(resp, filename)
    gunzip(filename, parent_path, joined_id)
    cleanup(filename)
    return joined_id

def init_redis(host: str, port: int, username: Optional[str] = None, password: Optional[str] = None, ssl: bool = True) -> Union[redis.Redis, None]:
    """
    Initialize Redis instance @host:port

    Optionally pass `username` and `password` for auth, and `ssl` if needed

    By default `decode_responses` is set to False
    """
    if username and password:
        cred_provider = UsernamePasswordCredentialProvider(username, password)
    else:
        cred_provider = None
    
    client = redis.Redis(host, port, credential_provider=cred_provider, decode_responses=False, ssl=ssl)
    try:
        pong = client.ping()
    except:
        pong = None
    
    if pong:
        logger.info("Redis initialized")
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
    if USE_REDIS:
        logger.info("Using REDIS")
        if DEBUG: ssl = False 
        else: ssl = True
        REDIS_CLIENT: redis.Redis = init_redis(REDIS_HOST, REDIS_PORT, REDIS_DB, ssl=ssl)
    else:
        REDIS_CLIENT = None

    if DEBUG: parent_path = "./temp"
    else: parent_path = "/tmp/"

    result = None
    action = event.get('ids')
    query = event.get('query')

    id_tex_map = {}
    doc_hashes = []
    for id in action:
        clean_id = get_gzip_source_files(id, parent_path)
        files = os.listdir(os.path.join(parent_path, clean_id))
        tex_path = [f for f in files if ".tex" in f][0]
        tex_path_dir = os.path.join(parent_path,clean_id,tex_path)
        parsed_tex = parse_tex(tex_path_dir)
        id_tex_map.update({clean_id:parsed_tex})
        doc_hashes.append(
            hash_doc(id)
        )

    context, context_map, metadata = build_context(id_tex_map[list(id_tex_map.keys())[0]])
    
    docstore = build_docstore(context, metadata, doc_hashes[0], REDIS_CLIENT)
    
    chain = VectorDBQAWithSourcesChain.from_llm(OpenAI(temperature=0), vectorstore=docstore)
    result = chain({
        "question":query
    }, return_only_outputs=True)

    resp = {'result':result}

    return resp


    