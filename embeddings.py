from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter # splits the text from document into chunks
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
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
 
db = Chroma.from_documents( #creating chroma instance and telling that you want to calculate embeddings for docs param
    docs,
    embedding=embeddings,
    persist_directory='emb'
) # every time re-inserts same set of embeddings

results = db.similarity_search_with_score('What is an interesint fact about English language?', k=1)

for result in results:
    print("\n")
    print(result[1]) # similarity_score
    print(result[0].page_content)