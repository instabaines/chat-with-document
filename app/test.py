from rag import load_corpus,rag,process_query

from llama_index.legacy.readers.file import docs_reader
from pathlib import Path
loader = docs_reader.PDFReader()
docs =loader.load_data(
        file=Path("1015660.1015661.pdf")
    )

retriever = rag(docs)
response = process_query('tell me about research methods in computer communication',retriever)
print (response)