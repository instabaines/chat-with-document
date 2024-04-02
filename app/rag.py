

from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.legacy.node_parser import SentenceWindowNodeParser
from llama_index.legacy import Document

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
import torch

from llama_index.legacy import set_global_tokenizer
from transformers import AutoTokenizer

from constant import model_url
llm =None
embed_model =None

    
node_parser = SentenceWindowNodeParser.from_defaults(
    # how many sentences on either side to capture
    window_size=2,
    # the metadata key that holds the window of surrounding sentences
    window_metadata_key="window",
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence",
    # sentence_splitter =split
)
def load_corpus(docs):
    nodes = node_parser.get_nodes_from_documents([Document(text=docs)], show_progress=False)
    return nodes

def load_models():
    global llm, embed_model

    print("intializing llm")
    try:
        llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="llama-2-7b-chat.Q2_K.gguf",
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
        )
    except:
        raise Exception("unable to fetch model, check your connection and retry")
    print("setting tokenizer")
    #set tokenizer
    set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )
    try:
        print ("loading embedding model")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except:
       raise Exception("unable to fetch embedding model, check your connection and retry")
  
    
 
def rag(corpus):
    global llm, embed_model
    if llm==None:
        print("loading models")
        load_models()
    
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    index = VectorStoreIndex.from_documents(
        corpus, 
        show_progress=True,
        service_context=service_context
    )
    retriever = index.as_query_engine(similarity_top_k=5,node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],)
    return retriever

def process_query(query,retriever):
    retrieved_nodes = retriever.query(query)
    return retrieved_nodes

