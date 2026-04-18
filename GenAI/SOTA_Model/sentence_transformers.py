import os
#os.environ["GOOGLE_API_KEY"] =os.getenv("GOOGLE_API_KEY")
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from sentence_transformers import SentenceTransformer

open_source_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

text = "machine"
embedding = open_source_embedding_model.encode(text)
print(embedding)
print(len(embedding))
