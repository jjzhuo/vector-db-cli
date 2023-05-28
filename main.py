import os
import click
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tiktoken
import colorama
from colorama import Fore

colorama.init()
embeddings = OpenAIEmbeddings()
ADA_EMBED_PRICE = 0.0004

@click.group()
def cli():
    pass

@cli.command()
def list_indices():
    indices_dir = os.environ["VDB_DIR"]
    indices = os.listdir(indices_dir)
    for index in indices:
        if not index.startswith("."):
            print(index)

@cli.command()
@click.argument('index_name')
@click.argument('input_file')
def create_index(index_name, input_file):
    loader = TextLoader(input_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings, 
                                     persist_directory=index_directory(index_name))
    db.persist()

@cli.command()
@click.argument('index_name')
def describe_index(index_name):
    db = Chroma(persist_directory=index_directory(index_name), embedding_function=embeddings)
    collection = db.get()
    print("count=%d" % len(collection["ids"]))
    sources = set([x["source"] for x in collection["metadatas"]])
    print(f"sources: {sources}")

@cli.command()
@click.argument('index_name')
def contents(index_name):
    db = Chroma(persist_directory=index_directory(index_name), embedding_function=embeddings)
    collection = db.get()
    print_collection(collection)

@cli.command()
@click.argument('index_name')
@click.argument('query')
def search_similarity(index_name, query):
    db = Chroma(persist_directory=index_directory(index_name), embedding_function=embeddings)
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.page_content)

@cli.command()
@click.argument('index_name')
@click.argument('query')
@click.option('--temperature', default=0)
@click.option('--model', default="gpt-3.5-turbo")
def chat(index_name, query, temperature, model):
    db = Chroma(persist_directory=index_directory(index_name), embedding_function=embeddings)
    retriever = db.as_retriever()
    chat = ChatOpenAI(temperature=temperature, model=model)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
    print(qa.run(query))
    
@cli.command()
@click.argument('index_name')
@click.argument('keyword')
def search_keyword(index_name, keyword):
    db = open_db(index_name)
    results = db._collection.get(where_document={"$contains": keyword})
    print_collection(results)


@cli.command()
@click.argument('index_name')
@click.argument('input_file')
@click.option('--chunk_size', default=1000)
def insert_text(index_name, input_file, chunk_size):
    loader = TextLoader(input_file)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    documents = text_splitter.split_documents(doc)
    
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    db = Chroma(persist_directory=index_directory(index_name), embedding_function=embeddings)
    db.add_texts(texts=texts, metadatas=metadatas)
    db.persist()

@cli.command()
@click.argument('index_name')
@click.argument('id')
def remove_text(index_name, id):
    db = open_db(index_name)
    db._collection.delete(
        ids=[id]
    )

@cli.command()
@click.argument('input_file')
def estimate_cost(input_file):
    print(f"${embedding_cost(open(input_file).read())}")

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def embedding_cost(string: str):
    ntokens = num_tokens_from_string(string, "cl100k_base")
    return ntokens * ADA_EMBED_PRICE / 1000

def open_db(index_name):
    return Chroma(persist_directory=index_directory(index_name), embedding_function=embeddings)

def print_collection(collection):
    for id, metadata, text in zip(collection["ids"], collection["metadatas"], collection["documents"]):
        print(Fore.GREEN + f"{id} Metadata: {metadata}")
        print(Fore.WHITE + text)
        
def index_directory(index_name):
    return os.path.join(os.environ["VDB_DIR"], index_name)

if __name__ == '__main__':
    cli()