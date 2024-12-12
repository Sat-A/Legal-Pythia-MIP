import re
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import nest_asyncio

# Load environment variables
load_dotenv(dotenv_path=".env/pyvenv.cfg")

# Apply nested asyncio to avoid loop issues
nest_asyncio.apply()

# Parse the PDF
filename = "kpmg_report"
parser = LlamaParse(result_type="text")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=[filename + '.pdf'], file_extractor=file_extractor).load_data()

# Combine text for analysis
pdf_text = " ".join([doc.text for doc in documents])

# Create an index for semantic queries
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Define helper functions for regex-based extraction
def extract_carbon_emissions(text):
    match = re.search(r"Carbon Emissions\s*\(tons\):?\s*([\d,\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_renewable_energy(text):
    match = re.search(r"Renewable Energy\s*Source\s*\(%\):?\s*([\d\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

# Add similar regex-based functions for other features
# e.g., Recycling Rate, Water Usage, etc.

# Query Llama Index for semantic extraction
def query_feature(query):
    response = query_engine.query(query)
    return response.response.strip()

# Main talking points and extraction
features = {
    "Carbon Emissions": {
        "query": "What are the carbon emissions mentioned in the report?",
        "regex": extract_carbon_emissions(pdf_text)
    },
    "Energy Source (%)": {
        "query": "What percentage of energy comes from renewable sources?",
        "regex": extract_renewable_energy(pdf_text)
    },
    "Waste Management Metrics": {
        "query": "What is the recycling rate or waste management information in the report?",
        "regex": None  # Add regex function if applicable
    },
    # Add other features here...
}

# Extract data
summary = {}
for feature, methods in features.items():
    semantic_result = query_feature(methods["query"])
    regex_result = methods.get("regex")
    summary[feature] = {
        "Semantic Extraction": semantic_result,
        "Regex Extraction": regex_result if regex_result else "Not applicable"
    }

# Display the extracted summary
for feature, details in summary.items():
    print(f"Feature: {feature}")
    print(f"  Semantic Extraction: {details['Semantic Extraction']}")
    print(f"  Regex Extraction: {details['Regex Extraction']}")
    print()

# You can pass the `summary` dictionary to OpenAI's API for further summarisation.
