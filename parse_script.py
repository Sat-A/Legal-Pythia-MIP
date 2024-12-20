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

def extract_recycling_rate(text):
    match = re.search(r"Recycling Rate\s*\(%\):?\s*([\d\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_water_usage(text):
    match = re.search(r"Water Usage\s*\(cubic meters/year\):?\s*([\d,\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_deforestation_impact(text):
    match = re.search(r"Deforestation Impact\s*\(hectares\):?\s*([\d,\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_certifications(text):
    match = re.search(r"Certifications\s*:\s*(.*?)(?:\.|\n|$)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not explicitly mentioned"

def extract_violations(text):
    match = re.search(r"Number of Violations\s*:\s*([\d]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_esg_score(text):
    match = re.search(r"ESG Investment Attractiveness Score\s*:\s*([\d\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_stock_impact(text):
    match = re.search(r"Impact on Stock Price\s*\(%\):?\s*([\d\.]+)", text, re.IGNORECASE)
    return match.group(1) if match else "Not explicitly mentioned"

def extract_sustainability_initiatives(text):
    match = re.search(r"Sustainability Initiatives\s*:\s*(.*?)(?:\.|\n|$)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not explicitly mentioned"

# Query Llama Index for semantic extraction
def query_feature(query):
    response = query_engine.query(query)
    return response.response.strip()

# Define the features and extraction methods
features = {
    "Carbon Emissions (tons)": {
        "query": "What are the carbon emissions mentioned in the report?",
        "regex": extract_carbon_emissions(pdf_text)
    },
    "Energy Source (% Renewable)": {
        "query": "What percentage of energy comes from renewable sources?",
        "regex": extract_renewable_energy(pdf_text)
    },
    "Waste Management Metrics (Recycling Rate %)": {
        "query": "What is the recycling rate or waste management information in the report?",
        "regex": extract_recycling_rate(pdf_text)
    },
    "Water Usage (cubic meters/year)": {
        "query": "What is the water usage mentioned in the report?",
        "regex": extract_water_usage(pdf_text)
    },
    "Deforestation Impact (hectares)": {
        "query": "What is the impact on deforestation (in hectares)?",
        "regex": extract_deforestation_impact(pdf_text)
    },
    "Certifications": {
        "query": "What certifications are mentioned in the report?",
        "regex": extract_certifications(pdf_text)
    },
    "Number of Violations": {
        "query": "How many violations are reported in the document?",
        "regex": extract_violations(pdf_text)
    },
    "ESG Investment Attractiveness Score": {
        "query": "What is the ESG investment attractiveness score?",
        "regex": extract_esg_score(pdf_text)
    },
    "Impact on Stock Price (%)": {
        "query": "What is the reported impact on stock price?",
        "regex": extract_stock_impact(pdf_text)
    },
    "Sustainability Initiatives": {
        "query": "What sustainability initiatives are described in the report?",
        "regex": extract_sustainability_initiatives(pdf_text)
    }
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
