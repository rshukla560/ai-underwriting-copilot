from app.core.rag.document_processor import process_pdf
from app.core.rag.embedder import embed_chunks, embed_query
from app.core.rag.vector_store import upsert_chunks, query_similar_chunks

# Step 1: Process
chunks = process_pdf("tests/fixtures/sample_application.pdf")
print(f"✓ Chunks created: {len(chunks)}")

# Step 2: Embed
chunks = embed_chunks(chunks)
print(f"✓ Chunks embedded, dimensions={len(chunks[0]['embedding'])}")

# Step 3: Upsert
upsert_chunks(chunks, applicant_id="TEST_001")
print(f"✓ Upserted {len(chunks)} chunks")

# Step 4: Query
query_vector = embed_query("what is the applicant BMI?")
results = query_similar_chunks(query_vector, applicant_id="TEST_001")
print(f"✓ Retrieved {len(results)} results")

for r in results:
    print(f"  score={r['similarity_score']} | page={r['page_number']} | {r['text'][:80]}")
