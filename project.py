import os

os.environ["OPENAI_API_KEY"] = ""

import pandas as pd
import chromadb
import tiktoken
from io import StringIO

from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Initialize model
model_name = 'gpt-3.5-turbo-instruct'

llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model_name=model_name, temperature=0, max_tokens=2000)

# Generate listings
data_gen_template = """
Generate {num_reviews} real estate listings in CSV format. Each listing MUST include these features:

NAME; PRICE; NUMBER_OF_BEDROOMS; NUMBER_OF_BATHROOMS; HOUSE_SIZE; DESCRIPTION_HOUSE; DESCRIPTION_NEIGHBOURHOOD.

Be creative in your listings. The CSV format is a must, separator is ";", one line per listing, labels in first line.
"""


# Format the prompt template with the desired number of reviews
prompt = PromptTemplate.from_template(data_gen_template).format(num_reviews=30)

# Generate listings using LLMChain
listings_csv = llm(prompt=prompt)

# Output for verification
print(listings_csv)

# Convert the generated text to a DataFrame
df = pd.read_csv(StringIO(listings_csv), sep=';')

# Save the DataFrame to a CSV file
df.to_csv('listings.csv', index=False)


# Load Data for Indexing
loader = CSVLoader(file_path='./listings.csv')
listing_docs = loader.load()

# Create embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

home_match_db = Chroma.from_documents(
    documents=listing_docs,
    embedding=embeddings,
    persist_directory="data",
    collection_name="listings"

# Similarity search

# User input/query
query= "I am looking for a big house"

# Fetching listings
results = home_match_db.similarity_search(query, k=5)

for doc in results:
    print(f"{doc.page_content}, \n\nMetadata: {doc.metadata}, \nScore: {score}")
    print("---")

    # Similarity search with filter "Pool"

# Fetching results (same query)
results_with_scores = home_match_db.similarity_search_with_score("Pool", k=3)

for doc, score in results_with_scores:
    print(f"{doc.page_content}, \n\nMetadata: {doc.metadata}, \nScore: {score}")
    print("---")



# Set up RAG for a more comprehensive search

retriever = home_match_db.as_retriever()

chain = RetrievalQA.from_chain_type(llm=llm,
                                    retriever=retriever,
)

response = chain(query)

print(f"You: {response['query']} ")
print(f"Chatbot: {response['result']}")


# "Collect" user preferences, should be done with a proper frontend normally

questions = [
    "How big do you want your house to be?"
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]


# RAG with document retrieval

# Convert answers to embeddings
answers_embeddings = embeddings.embed_documents(answers)

# Search database with answer embeddings
results_embeddings = home_match_db.similarity_search_by_vector(answers_embeddings)

for doc in results_embeddings:
    print(f"{doc.page_content}, \n\nMetadata: {doc.metadata}, \nScore: {score}")
    print("---")



# RAG with chain

prompt = hub.pull("rlm/rag-prompt")

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# Concatenate the user needs/answers
query = ''.join(answers)

# Fetching results
response = rag_chain_with_source.invoke(query)

print(f"Your needs: ")
for answer in answers:
    print(f"- {answer}")

print(" \n")
print(f"Selected Listings based on your needs: ")

for doc in response['context']:
    print(f"{doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
    print("---")