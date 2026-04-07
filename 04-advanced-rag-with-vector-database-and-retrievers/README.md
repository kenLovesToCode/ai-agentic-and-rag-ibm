### LangChain retriever

- multi-query retriever (use llm to create different versions)
- self-query retriever (convert query into string metadata filter; uses query-constructing LLM chain to gnerate structured query, and then applies this structured query to its underlying vectore state)
- parent document retriever (2 splitters, parent and child)

### BM25 Retriever - Advanced Keyword-Based Search

Ideal for:

- Technical documentation where exact terms matter
- Legal documents with specific terminology
- Product catalogs with precise specifications
- Academic papers with specialized vocabulary
- Applications requiring keyword-based retrieval rather than semantic similarity

### HNSW

- Imagine you're trying to find a specific restaurant in a huge city. Instead of checking every single restaurant one by one, you use a smart navigation system such as Google Maps. You start with a zoomed-out view showing major highways, then gradually zoom in to see smaller streets, and finally arrive at your exact destination. This is essentially how the Hierarchical Navigable Small World (HNSW) algorithm works - but instead of finding restaurants, it finds similar data points in massive databases.
