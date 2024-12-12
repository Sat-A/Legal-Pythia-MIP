# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env/pyvenv.cfg")

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

import nest_asyncio
nest_asyncio.apply()

filename = "kpmg_report"

# set up parser
parser = LlamaParse(
    result_type="text"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=[filename+'.pdf'], file_extractor=file_extractor).load_data()

with open(filename+".txt", 'w+') as f:
    for i in range(len(documents)):
        f.write(documents[i].text+"\n")

# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine
query = "What capabilities are we looking for?"
response = query_engine.query(query)
print(response)