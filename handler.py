import os
import sys
import uuid
import json
import tarfile
import subprocess
from typing import Optional, Union, List
from dotenv import load_dotenv
import logging
from doc_utils import parse_tex, remove_latex, get_paper_title
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
import faiss
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
    def from_precomputed(cls, embedding, texts: list, metadatas: List[dict], dims: int, redis: redis.Redis, doc_hash: str, **kwargs) -> FAISS:
        embeddings = []
        docs = []
        for i in range(len(doc_hash)):
            hash = doc_hash[i]
            embed_text = texts[i]
            metadata = metadatas[i]
            if redis.exists(hash) == 0 or redis is None:
                tmp = embedding.embed_documents(embed_text)
                embeddings+=tmp
                if redis is not None:
                    try:
                        redis.set(hash, json.dumps(tmp))
                    except Exception as e:
                        logger.warning(f"Unable to cache embeddings in Redis: {e}")
            else:
                cached_embedding = redis.get(hash)
                embeddings+=CustomFAISS.decode_cached_embeddings(cached_embedding)
            docs += [Document(page_content=text, metadata=metadata[i]) for i,text in enumerate(embed_text)]

        index = faiss.IndexFlatL2(dims)
        index.add(np.array(embeddings, dtype=np.float32))
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
def build_docstore(context, metadata, doc_hash, redis, dims=1536):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    docsearch = CustomFAISS.from_precomputed(embeddings, context, metadatas=metadata, dims=dims, redis=redis, doc_hash=doc_hash)
    return docsearch
 
@timer_func
def build_context(parsed_tex: dict):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    context = []
    context_map = {}
    for k,v in parsed_tex.items():
        temp_ctx = text_splitter.split_text(
            v['content']
        )
        context_map.update({
            i+len(context):{
                "section":k, "content":temp_ctx[i], "title":v['title'], "id":v['arxiv_id']
            } for i in range(len(temp_ctx))
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
    paper_titles = get_paper_title(action)

    contexts = []
    metadatas = []
    metadata_cache = {

    }

    for id in action:
        title = paper_titles[id]
        clean_id = get_gzip_source_files(id, parent_path)
        files = os.listdir(os.path.join(parent_path, clean_id))
        tex_path = [f for f in files if ".tex" in f][0]
        tex_path_dir = os.path.join(parent_path,clean_id,tex_path)
        parsed_tex = parse_tex(tex_path_dir, title, id)
        id_tex_map.update({clean_id:parsed_tex})
        doc_hashes.append(
            hash_doc(id)
        )
        context, context_map, metadata = build_context(parsed_tex)
        contexts.append(context)
        metadata_cache.update({id:metadata})
        metadatas.append([{"source":f"{id}:{i}"} for i in range(len(metadata))])

    docstore = build_docstore(contexts, metadatas, doc_hashes, REDIS_CLIENT)
    
    chain = VectorDBQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="map_reduce", vectorstore=docstore)
    result = chain({
        "question":query
    }, return_only_outputs=True)

    answer = result['answer']
    raw_source = result['sources'].split(",")
    if len(raw_source) > 1:

        sources = [
            metadata_cache[rs.split(":")[0].strip()][int(rs.split(":")[1].strip())] for rs in raw_source
        ]
    else:
        raw_source = raw_source[0].split(":")
        sources = metadata_cache[raw_source[0].strip()][int(raw_source[1].strip())]
    resp = {'answer':answer, "sources":sources}

    return resp


    