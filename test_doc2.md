# Advanced RAG Techniques

## Overview
Retrieval-Augmented Generation (RAG) enhances large language models by providing them with relevant context from external knowledge bases.

## Graph Enhancement

### Why Graphs?
Traditional RAG uses vector similarity alone. Graph-enhanced RAG adds relationship-based retrieval:

- **Entities**: People, organizations, technologies
- **Relationships**: Works with, depends on, relates to
- **Topics**: Shared concepts and themes

### Benefits
1. More contextual results
2. Better handling of multi-hop questions
3. Discovery of non-obvious connections

## Implementation Details

### Document Processing Pipeline
1. Load documents (PDF, Markdown, HTML)
2. Extract text and metadata
3. Generate embeddings with OpenAI
4. Identify entities with GLiNER
5. Extract keywords with KeyBERT
6. Store in graph vector store

### Query Processing
- Vector similarity for initial candidates
- Graph traversal for related documents
- MMR for diversity in results

## Technologies

- **Cassandra/Astra DB**: Distributed graph database
- **LangChain**: RAG orchestration framework
- **GLiNER**: Named entity recognition
- **KeyBERT**: Keyword extraction

## Conclusion
Graph RAG represents the next evolution in intelligent document retrieval systems.
