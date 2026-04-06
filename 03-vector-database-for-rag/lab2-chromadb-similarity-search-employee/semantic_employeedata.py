# Importing necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions

"""
TALENT SEARCH PLATFORM

Similarity Search
Semantic Understanding: For the Python query, John Doe ranks highest (0.5156) because his skills explicitly include "Python, JavaScript, React, Node.js" - a perfect match for web development. Notice how the system found relevant candidates even though the query didn't exactly match the text.
Cross-Department Results: The leadership search finds managers across different departments (Marketing, HR, Engineering), showing the system understands that "manager" roles exist in various contexts.
Ranking Logic: Distance scores increase (0.5382 → 0.5467 → 0.5497) as semantic similarity decreases, demonstrating how Chroma DB quantifies relevance.
Skills vs. Roles: Matthew Garcia appears in Python results despite his skills being "JavaScript, HTML/CSS" because his role as a "Software Engineer" is semantically related to development work.

Metadata Filtering
Exact Filtering Power: The Engineering department filter returns exactly 8 out of 15 employees, showing that over half the workforce is in Engineering - a typical distribution for tech companies.
Seniority Distribution: The 10+ years filter reveals 6 senior employees (40% of workforce), indicating a good mix of experience levels. Notice how roles reflect seniority: "Senior," "Lead," "Principal," and "Manager" titles.
Role Hierarchy Visible: In the senior employees list, you can see the career progression: Software Engineer → Senior Software Engineer → Lead/Principal Engineer → Engineering Manager → Senior Architect.
Geographic Concentration: Only 3 employees in California highlight the distributed nature of the workforce, with San Francisco having multiple employees as expected for a tech hub.
Perfect Precision: Unlike similarity search, metadata filtering gives 100% precise results - every employee returned exactly matches the specified criteria.

Combined Search: Similarity + Medatada
Filtered Pool: From 15 total employees, only 4 meet both the semantic similarity AND the exact criteria (8+ years in major tech cities), demonstrating how filters narrow down results effectively.
Semantic Ranking within Constraints: Michael Brown ranks highest (0.6726) because his Java/Spring Boot skills are semantically closest to "senior Python developer full-stack" among the filtered candidates.
Interesting Discoveries: Chris Evans (Senior Architect) ranks second despite having "System design, distributed systems" skills rather than Python - showing the system understands that senior architects often have full-stack capabilities.
Geographic + Experience Filtering: All results are from San Francisco, New York, or Seattle (major tech cities) and have 10+ years experience, proving the metadata filters worked perfectly.
Real-World Relevance: This type of search mimics actual HR/recruiting scenarios where you need "senior developers with specific skills in key locations" - combining fuzzy skill matching with hard requirements.
Distance Score Interpretation: Higher distances (0.8761 for Olivia) indicate she's still relevant to the query but with a looser semantic match, which is valuable for finding candidates with transferable skills.
"""

# Define the embedding function using SentenceTransformers
# This function will be used to generate embeddings (vector representations) for the data
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Creating an instance of ChromaClient to establish a connection with the Chroma database
client = chromadb.Client()

# Defining a name for the collection where data will be stored or accessed
# This collection is likely used to group related records, such as employee data
collection_name = "employee_collection"

# Defining a function named 'main'
# This function is used to encapsulate the main operations for creating collections,
# generating embeddings, and performing similarity search
def main():
    try:
        # Creating a collection using the ChromaClient instance
        # The 'create_collection' method creates a new collection with the specified configuration
        collection = client.create_collection(
            # Specifying the name of the collection to be created
            name=collection_name,
            # Adding metadata to describe the collection
            metadata={"description": "A collection for storing employee data"},
            # Configuring the collection with cosine distance and embedding function
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # Defining a list of employee dictionaries
        # Each dictionary represents an individual employee with comprehensive information
        employees = [
            {
                "id": "employee_1",
                "name": "John Doe",
                "experience": 5,
                "department": "Engineering",
                "role": "Software Engineer",
                "skills": "Python, JavaScript, React, Node.js, databases",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_2",
                "name": "Jane Smith",
                "experience": 8,
                "department": "Marketing",
                "role": "Marketing Manager",
                "skills": "Digital marketing, SEO, content strategy, analytics, social media",
                "location": "Los Angeles",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_3",
                "name": "Alice Johnson",
                "experience": 3,
                "department": "HR",
                "role": "HR Coordinator",
                "skills": "Recruitment, employee relations, HR policies, training programs",
                "location": "Chicago",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_4",
                "name": "Michael Brown",
                "experience": 12,
                "department": "Engineering",
                "role": "Senior Software Engineer",
                "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_5",
                "name": "Emily Wilson",
                "experience": 2,
                "department": "Marketing",
                "role": "Marketing Assistant",
                "skills": "Content creation, email marketing, market research, social media management",
                "location": "Austin",
                "employment_type": "Part-time"
            },
            {
                "id": "employee_6",
                "name": "David Lee",
                "experience": 15,
                "department": "Engineering",
                "role": "Engineering Manager",
                "skills": "Team leadership, project management, software architecture, mentoring",
                "location": "Seattle",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_7",
                "name": "Sarah Clark",
                "experience": 8,
                "department": "HR",
                "role": "HR Manager",
                "skills": "Performance management, compensation planning, policy development, conflict resolution",
                "location": "Boston",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_8",
                "name": "Chris Evans",
                "experience": 20,
                "department": "Engineering",
                "role": "Senior Architect",
                "skills": "System design, distributed systems, cloud platforms, technical strategy",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_9",
                "name": "Jessica Taylor",
                "experience": 4,
                "department": "Marketing",
                "role": "Marketing Specialist",
                "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
                "location": "Miami",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_10",
                "name": "Alex Rodriguez",
                "experience": 18,
                "department": "Engineering",
                "role": "Lead Software Engineer",
                "skills": "Full-stack development, React, Python, machine learning, data science",
                "location": "Denver",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_11",
                "name": "Hannah White",
                "experience": 6,
                "department": "HR",
                "role": "HR Business Partner",
                "skills": "Strategic HR, organizational development, change management, employee engagement",
                "location": "Portland",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_12",
                "name": "Kevin Martinez",
                "experience": 10,
                "department": "Engineering",
                "role": "DevOps Engineer",
                "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
                "location": "Phoenix",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_13",
                "name": "Rachel Brown",
                "experience": 7,
                "department": "Marketing",
                "role": "Marketing Director",
                "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
                "location": "Atlanta",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_14",
                "name": "Matthew Garcia",
                "experience": 3,
                "department": "Engineering",
                "role": "Junior Software Engineer",
                "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
                "location": "Dallas",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_15",
                "name": "Olivia Moore",
                "experience": 12,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": "Technical leadership, system architecture, performance optimization, mentoring",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
        ]

        # Create comprehensive text documents for each employee
        # These documents will be used for similarity search based on skills, roles, and experience
        employee_documents = []
        for employee in employees:
            document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
            document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
            document += f"Employment type: {employee['employment_type']}."
            employee_documents.append(document)
        
        # Adding data to the collection in the Chroma database
        # The 'add' method inserts or updates data into the specified collection
        collection.add(
            # Extracting employee IDs to be used as unique identifiers for each record
            ids=[employee["id"] for employee in employees],
            # Using the comprehensive text documents we created
            documents=employee_documents,
            # Adding comprehensive metadata for filtering and search
            metadatas=[{
                "name": employee["name"],
                "department": employee["department"],
                "role": employee["role"],
                "experience": employee["experience"],
                "location": employee["location"],
                "employment_type": employee["employment_type"]
            } for employee in employees]
        )

        # Retrieving all items from the specified collection
        # The 'get' method fetches all records stored in the collection
        all_items = collection.get()
        # Logging the retrieved items to the console for inspection or debugging
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform various types of searches within the collection
        def perform_advanced_search(collection, all_items):
            try:
                print("=== Similarity Search Examples ===")

                # Example 1: Search for Python developers
                print("\n1. Searching for Python developers:")
                query_text = "Python developer with web development experience"
                results = collection.query(
                    query_texts=[query_text],
                    n_results=3
                )
                print(f"Query: '{query_text}'")
                for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     Role: {metadata['role']}, Department: {metadata['department']}")
                    print(f"     Document: {document[:100]}...")

                # Example 2: Search for leadership roles
                print("\n2. Searching for leadership and management roles:")
                query_text = "team leader manager with experience"
                results = collection.query(
                    query_texts=[query_text],
                    n_results=3
                )
                print(f"Query: '{query_text}'")
                for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     Role: {metadata['role']}, Experience: {metadata['experience']} years")
                
                print("\n=== Metadata Filtering Examples ===")

                # Example 1: Filter by department
                print("\n3. Finding all Engineering employees:")
                results = collection.get(
                    where={"department": "Engineering"}
                )
                print(f"Found {len(results['ids'])} Engineering employees:")
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    print(f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)")

                # Example 2: Filter by experience range
                print("\n4. Finding employees with 10+ years experience:")
                results = collection.get(
                    where={"experience": {"$gte": 10}}
                )
                print(f"Found {len(results['ids'])} senior employees:")
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    print(f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)")

                # Example 3: Filter by location
                print("\n5. Finding employees in California:")
                results = collection.get(
                    where={"location": {"$in": ["San Francisco", "Los Angeles"]}}
                )
                print(f"Found {len(results['ids'])} employees in California:")
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    print(f"  - {metadata['name']}: {metadata['location']}")
                
                print("\n=== Combined Search: Similarity + Metadata Filtering ===")

                # Example: Find experienced Python developers in specific locations
                print("\n6. Finding senior Python developers in major tech cities:")
                query_text = "senior Python developer full-stack"
                results = collection.query(
                    query_texts=[query_text],
                    n_results=5,
                    where={
                        "$and": [
                            {"experience": {"$gte": 8}},
                            {"location": {"$in": ["San Francisco", "New York", "Seattle"]}}
                        ]
                    }
                )
                print(f"Query: '{query_text}' with filters (8+ years, major tech cities)")
                print(f"Found {len(results['ids'][0])} matching employees:")
                for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     {metadata['role']} in {metadata['location']} ({metadata['experience']} years)")
                    print(f"     Document snippet: {document[:80]}...")
                
                # Check if the results are empty or undefined
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    # Log a message if no similar documents are found for the query term
                    print(f'No documents found similar to "{query_text}"')
                    return
                
                # Log the header for the top 3 similar documents based on the query term
                print(f'Top 3 similar documents to "{query_text}":')
                # Loop through the top 3 results and log the document details
                for i in range(min(3, len(results['ids'][0]))):
                    # Extract the document ID and similarity score from the results
                    doc_id = results['ids'][0][i]
                    score = results['distances'][0][i]
                    # Retrieve the document text corresponding to the current ID from the results
                    text = results['documents'][0][i]
                    # Check if the text is available; if not, log 'Text not available'
                    if not text:
                        print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                    else:
                        print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
            except Exception as error:
                print(f"Error in advanced search: {error}")
        
        # Call the perform_advanced_search function with the collection and all_items as arguments
        perform_advanced_search(collection, all_items)

    except Exception as error:
        # Catching and handling any errors that occur within the 'try' block
        # Logs the error message to the console for debugging purposes
        print(f"Error: {error}")

if __name__ == "__main__":
    main()
