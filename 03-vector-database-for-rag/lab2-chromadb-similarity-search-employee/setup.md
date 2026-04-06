```bash
# setup environment
pip install chromadb==1.0.12
pip install sentence-transformers==4.1.0
```

Potentially, could make an app for modern companies needed for complex searching for recommendation system. Or to extract information (e.g. people, books, etc.) then use the data to perform similarity search and metadata filtering for getting the best or by preferences.

### books_advanced_search

- 1: Implement Similarity Search for Book Recommendations. Create meaningful text documents for each book that combine title, description, themes, and setting information for semantic search.
- 2: Implement Metadata Filtering. Add the book data to your collection with comprehensive metadata for filtering capabilities.
- 3: Create an Advanced Search Function. Implement a function that demonstrates multiple search types:
  - Similarity search for "magical fantasy adventure"
  - Filter books by genre (Fantasy or Science Fiction)
  - Filter books by rating (4.0 or higher)
  - Combined search: Find highly-rated dystopian books with similarity search
