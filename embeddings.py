from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter # splits the text from document into chunks
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# see output of an embedding
# emb = embeddings.embed_query('hi there')

# print(emb)

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200, # max 200 characters per chunk
    chunk_overlap=0 # ensure that we don't divide into weird chunks that are not related to each other
)
loader = TextLoader('facts.txt')

docs = loader.load_and_split(
    text_splitter=text_splitter
)

for doc in docs:
    print(doc.page_content)
    print("\n")