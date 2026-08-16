from sentence_transformers import SentenceTransformer
from loader import load_pdf
from splitter import split_documents

# Load the PDF
documents = load_pdf("NIPS-2017-attention-is-all-you-need-Paper.pdf")

# Split into chunks
chunks = split_documents(documents)

# Extract only the text
texts = [chunk.page_content for chunk in chunks]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(texts)

print(f"Total Chunks: {len(chunks)}")
print(f"Total Embeddings: {len(embeddings)}")

print(type(embeddings))
print(embeddings.shape)

print("\nFirst Embedding (first 10 values):")
print(embeddings[0][:10])