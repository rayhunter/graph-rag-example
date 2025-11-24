"""
Automated test script to query the Graph RAG knowledge base.
Shows all results without pausing.
"""
import asyncio
from search_executor import ChainManager, get_similarity_result, get_mmr_result

async def test_queries():
    """Run test queries against the knowledge graph."""
    
    # Initialize the chain manager
    print("🔧 Initializing Chain Manager...")
    manager = ChainManager()
    manager.setup_chains(k=5, depth=2, lambda_mult=0.5)
    print("✅ Ready!\n")
    
    # Test questions
    questions = [
        "What is Graph RAG and how does it work?",
        "What technologies are used in this system?",
        "How does vector search compare to graph traversal?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"📝 Question {i}: {question}")
        print('='*80)
        
        # Similarity Search (vector-only)
        print("\n🔍 SIMILARITY SEARCH (Vector-only):")
        print("-" * 80)
        sim_result, sim_usage = await get_similarity_result(manager, question)
        print(f"{sim_result}")
        print(f"\n📊 Tokens used: {sim_usage.get('total_tokens', 'N/A')}")
        
        # Graph Traversal Search (MMR with graph links)
        print("\n\n🕸️  GRAPH TRAVERSAL SEARCH (MMR + Graph Links):")
        print("-" * 80)
        mmr_result, mmr_usage = await get_mmr_result(manager, question)
        print(f"{mmr_result}")
        print(f"\n📊 Tokens used: {mmr_usage.get('total_tokens', 'N/A')}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         Graph RAG Knowledge Base - Automated Test             ║
╚═══════════════════════════════════════════════════════════════╝

Comparing:
  🔍 Similarity Search: Pure vector-based retrieval
  🕸️  Graph Traversal: Vector + relationship-based retrieval

""")
    
    asyncio.run(test_queries())
    
    print("""
✨ Test Complete!

KEY INSIGHT: Notice how Graph Traversal provides:
  • More comprehensive technology mentions
  • Better cross-document connections
  • Richer contextual information
  
This is the power of combining vector embeddings with graph relationships!
""")

