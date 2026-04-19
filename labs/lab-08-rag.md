# Lab 8: Retrieval Augmented Generation

## Overview

This lab covers Retrieval Augmented Generation (RAG) using DSPy. RAG combines the power of LLMs with external knowledge bases to create more accurate, up-to-date, and factual AI systems.

## Learning Objectives

- Understand RAG architecture and benefits
- Implement basic RAG with DSPy
- Design effective retrieval strategies
- Optimize retrieval and generation quality
- Build production-ready RAG systems

## Prerequisites

- Completed Lab 7: Multi-Stage Programs
- Understanding of vector databases (helpful)
- Experience with information retrieval concepts

## What is RAG?

### RAG Architecture

Retrieval Augmented Generation combines:
1. **Retrieval**: Find relevant information from knowledge base
2. **Augmentation**: Augment prompts with retrieved information
3. **Generation**: Generate responses using augmented prompts

### Why RAG Matters

- **Accuracy**: Grounds responses in factual information
- **Up-to-date**: Can access current information
- **Explainable**: Sources can be cited
- **Customizable**: Uses your own knowledge base
- **Cost-effective**: Reduces need for large context windows

## Basic RAG with DSPy

### Simple RAG Implementation

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Knowledge base (simplified)
knowledge_base = [
    "Python is a high-level programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence.",
    "Docker is a platform for developing, shipping, and running applications.",
    "Git is a distributed version control system."
]

# Simple retrieval function
def retrieve_documents(query, knowledge_base, top_k=2):
    """Simple keyword-based retrieval."""
    query_words = set(query.lower().split())
    scores = []
    
    for doc in knowledge_base:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        scores.append((overlap, doc))
    
    # Sort by score and return top k
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scores[:top_k]]

# RAG signature
class RAGQA(dspy.Signature):
    """Answer questions using retrieved context."""
    context = dspy.InputField(desc="retrieved context documents")
    question = dspy.InputField(desc="question to answer")
    answer = dspy.OutputField(desc="answer based on context")
    confidence = dspy.OutputField(desc="confidence in answer")

# RAG pipeline
def rag_pipeline(question):
    # Retrieval stage
    retrieved_docs = retrieve_documents(question, knowledge_base)
    context = "\n".join(retrieved_docs)
    
    # Generation stage
    rag_qa = dspy.Predict(RAGQA)
    result = rag_qa(context=context, question=question)
    
    return {
        "retrieved_docs": retrieved_docs,
        "answer": result.answer,
        "confidence": result.confidence
    }

# Test
question = "Who created Python?"
result = rag_pipeline(question)
print(f"Retrieved: {result['retrieved_docs']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## Exercise 1: Vector-Based RAG

Implement RAG with vector similarity:

```python
# For this example, we'll use a simple vector approach
# In production, you'd use a proper vector database like FAISS, Pinecone, etc.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorRAG:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents)
    
    def retrieve(self, query, top_k=3):
        """Retrieve documents using vector similarity."""
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            (self.documents[i], similarities[i]) 
            for i in top_indices
        ]

# Initialize vector RAG
docs = [
    "The Eiffel Tower is located in Paris, France.",
    "Python was created by Guido van Rossum in 1991.",
    "Machine learning algorithms learn patterns from data.",
    "Docker containers package applications with dependencies."
]

vector_rag = VectorRAG(docs)

# RAG with vector retrieval
def vector_rag_pipeline(question):
    # Vector retrieval
    retrieved = vector_rag.retrieve(question, top_k=2)
    context = "\n".join([doc for doc, score in retrieved])
    
    # Generation
    rag_qa = dspy.ChainOfThought(RAGQA)
    result = rag_qa(context=context, question=question)
    
    return {
        "retrieved_docs": [doc for doc, score in retrieved],
        "scores": [score for doc, score in retrieved],
        "answer": result.answer,
        "reasoning": result.reasoning
    }

# Test
result = vector_rag_pipeline("What is Docker used for?")
print(f"Retrieved: {result['retrieved_docs']}")
print(f"Scores: {result['scores']}")
print(f"Answer: {result['answer']}")
```

## Exercise 2: Multi-Source RAG

Build RAG that retrieves from multiple sources:

```python
class MultiSourceRAG:
    def __init__(self):
        self.sources = {
            "technical": [
                "Python supports multiple programming paradigms.",
                "Docker uses containerization technology.",
                "Git enables distributed version control."
            ],
            "historical": [
                "Python was first released in 1991.",
                "Docker was released in 2013.",
                "Git was created by Linus Torvalds in 2005."
            ],
            "practical": [
                "Python is widely used in data science.",
                "Docker simplifies deployment processes.",
                "Git is essential for collaborative development."
            ]
        }
    
    def retrieve_from_sources(self, query, sources=None):
        """Retrieve from specified sources."""
        if sources is None:
            sources = list(self.sources.keys())
        
        all_results = []
        for source in sources:
            docs = self.sources[source]
            retrieved = self._retrieve_from_single_source(query, docs)
            all_results.extend([(doc, source) for doc in retrieved])
        
        return all_results
    
    def _retrieve_from_single_source(self, query, docs, top_k=1):
        """Retrieve from a single source."""
        # Simple keyword matching (would use vectors in production)
        query_words = set(query.lower().split())
        scored = []
        
        for doc in docs:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]

# Multi-source RAG pipeline
class SourceAwareRAG(dspy.Signature):
    """Answer questions with source attribution."""
    context = dspy.InputField(desc="retrieved context with sources")
    question = dspy.InputField(desc="question to answer")
    answer = dspy.OutputField(desc="answer with source attribution")
    sources_used = dspy.OutputField(desc="which sources were used")

def multi_source_rag_pipeline(question):
    multi_rag = MultiSourceRAG()
    
    # Retrieve from all sources
    retrieved = multi_rag.retrieve_from_sources(question)
    context_with_sources = "\n".join([
        f"[{source}] {doc}" for doc, source in retrieved
    ])
    
    # Generate with source awareness
    source_aware_qa = dspy.ChainOfThought(SourceAwareRAG)
    result = source_aware_qa(
        context=context_with_sources,
        question=question
    )
    
    return {
        "retrieved": retrieved,
        "answer": result.answer,
        "sources_used": result.sources_used
    }

# Test
result = multi_source_rag_pipeline("When was Python created?")
print(f"Retrieved: {result['retrieved']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources_used']}")
```

## Exercise 3: RAG with Query Expansion

Implement query expansion for better retrieval:

```python
class QueryExpander(dspy.Signature):
    """Expand query with related terms."""
    original_query = dspy.InputField(desc="original search query")
    expanded_queries = dspy.OutputField(desc="list of expanded queries")
    reasoning = dspy.OutputField(desc="reasoning for expansions")

class RAGWithExpansion:
    def __init__(self, documents):
        self.documents = documents
        self.vector_rag = VectorRAG(documents)
    
    def retrieve_with_expansion(self, query, top_k=3):
        """Retrieve using query expansion."""
        # Expand query
        expander = dspy.ChainOfThought(QueryExpander)
        expansion = expander(original_query=query)
        
        # Parse expanded queries
        expanded_queries = [query]  # Include original
        if "\n" in expansion.expanded_queries:
            expanded_queries.extend(expansion.expanded_queries.split("\n"))
        
        # Retrieve for each query
        all_retrieved = []
        for expanded_q in expanded_queries[:3]:  # Limit to 3 queries
            retrieved = self.vector_rag.retrieve(expanded_q.strip(), top_k=2)
            all_retrieved.extend(retrieved)
        
        # Deduplicate and re-rank
        unique_docs = list(set([doc for doc, score in all_retrieved]))
        
        # Re-rank by combined score
        doc_scores = {}
        for doc, score in all_retrieved:
            if doc in doc_scores:
                doc_scores[doc] += score
            else:
                doc_scores[doc] = score
        
        # Return top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_k]]

# Test
docs = [
    "Machine learning uses statistical techniques to give computers the ability to learn from data.",
    "Deep learning is a subset of machine learning using neural networks.",
    "Natural language processing applies machine learning to text data.",
    "Computer vision applies machine learning to image data."
]

rag_expansion = RAGWithExpansion(docs)

def rag_with_expansion_pipeline(question):
    # Retrieve with expansion
    retrieved_docs = rag_expansion.retrieve_with_expansion(question)
    context = "\n".join(retrieved_docs)
    
    # Generate
    rag_qa = dspy.ChainOfThought(RAGQA)
    result = rag_qa(context=context, question=question)
    
    return {
        "retrieved_docs": retrieved_docs,
        "answer": result.answer,
        "reasoning": result.reasoning
    }

result = rag_with_expansion_pipeline("What are applications of ML?")
print(f"Retrieved: {result['retrieved_docs']}")
print(f"Answer: {result['answer']}")
```

## Advanced RAG Techniques

### 1. Hybrid Retrieval

```python
class HybridRAG:
    def __init__(self, documents):
        self.documents = documents
        self.vector_rag = VectorRAG(documents)
    
    def hybrid_retrieve(self, query, top_k=3, alpha=0.5):
        """Combine keyword and vector retrieval."""
        # Vector retrieval
        vector_results = self.vector_rag.retrieve(query, top_k=top_k*2)
        vector_scores = {doc: score for doc, score in vector_results}
        
        # Keyword retrieval
        keyword_results = self._keyword_retrieve(query, top_k=top_k*2)
        keyword_scores = {doc: 1.0 for doc in keyword_results}
        
        # Combine scores
        combined_scores = {}
        all_docs = set(vector_scores.keys()) | set(keyword_scores.keys())
        
        for doc in all_docs:
            vector_score = vector_scores.get(doc, 0)
            keyword_score = keyword_scores.get(doc, 0)
            combined_scores[doc] = alpha * vector_score + (1-alpha) * keyword_score
        
        # Return top k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_k]]
    
    def _keyword_retrieve(self, query, top_k):
        """Simple keyword retrieval."""
        query_words = set(query.lower().split())
        scored = []
        
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]
```

### 2. Reranking Strategy

```python
class Reranker(dspy.Signature):
    """Rerank retrieved documents based on relevance."""
    query = dspy.InputField(desc="search query")
    documents = dspy.InputField(desc="retrieved documents")
    reranked_docs = dspy.OutputField(desc="documents reranked by relevance")
    reasoning = dspy.OutputField(desc="reasoning for reranking")

def rag_with_reranking(query, retriever, top_k=5):
    """RAG with reranking stage."""
    # Initial retrieval
    initial_docs = retriever.retrieve(query, top_k=top_k*2)
    
    # Rerank
    reranker = dspy.ChainOfThought(Reranker)
    rerank_result = reranker(
        query=query,
        documents="\n".join([doc for doc, score in initial_docs])
    )
    
    # Parse reranked results (simplified)
    reranked_docs = initial_docs[:top_k]  # Would parse actual reranking
    
    return reranked_docs
```

### 3. Citation Generation

```python
class CitatedRAG(dspy.Signature):
    """Generate answers with citations."""
    context = dspy.InputField(desc="retrieved context with document IDs")
    question = dspy.InputField(desc="question")
    answer = dspy.OutputField(desc="answer with inline citations")
    citations = dspy.OutputField(desc="list of cited document IDs")

def cited_rag_pipeline(question, retriever):
    """RAG that provides citations."""
    # Retrieve with document IDs
    retrieved = retriever.retrieve_with_ids(question)
    context_with_ids = "\n".join([
        f"[DOC{i}] {doc}" for i, (doc, score) in enumerate(retrieved)
    ])
    
    # Generate with citations
    cited_qa = dspy.ChainOfThought(CitedRAG)
    result = cited_qa(context=context_with_ids, question=question)
    
    return {
        "answer": result.answer,
        "citations": result.citations,
        "retrieved_docs": [doc for doc, score in retrieved]
    }
```

## RAG Optimization

### 1. Retrieval Evaluation

```python
def evaluate_retrieval(retriever, test_queries, ground_truth_docs, k=5):
    """Evaluate retrieval quality."""
    metrics = {
        "precision_at_k": [],
        "recall_at_k": [],
        "mrr": []  # Mean Reciprocal Rank
    }
    
    for query, relevant_docs in zip(test_queries, ground_truth_docs):
        retrieved = retriever.retrieve(query, top_k=k)
        retrieved_docs = [doc for doc, score in retrieved]
        
        # Precision@K
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        precision = relevant_retrieved / k
        metrics["precision_at_k"].append(precision)
        
        # Recall@K
        recall = relevant_retrieved / len(relevant_docs)
        metrics["recall_at_k"].append(recall)
        
        # MRR
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                mrr = 1 / (i + 1)
                metrics["mrr"].append(mrr)
                break
        else:
            metrics["mrr"].append(0)
    
    # Average metrics
    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

### 2. End-to-End RAG Evaluation

```python
def evaluate_rag_pipeline(rag_pipeline, test_data, metric_func):
    """Evaluate complete RAG pipeline."""
    scores = []
    detailed_results = []
    
    for item in test_data:
        result = rag_pipeline(item["question"])
        
        # Evaluate answer quality
        score = metric_func(result["answer"], item["ground_truth"])
        scores.append(score)
        
        detailed_results.append({
            "question": item["question"],
            "retrieved_docs": result["retrieved_docs"],
            "answer": result["answer"],
            "score": score
        })
    
    return {
        "mean_score": sum(scores) / len(scores),
        "detailed_results": detailed_results
    }
```

### 3. Adaptive Retrieval

```python
class AdaptiveRAG:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def adaptive_retrieve(self, query, min_confidence=0.7):
        """Adaptively determine how much to retrieve."""
        # Start with small retrieval
        initial_results = self.retriever.retrieve(query, top_k=3)
        
        # Check if results are confident
        if self._check_confidence(initial_results) >= min_confidence:
            return initial_results
        
        # If not confident, retrieve more
        expanded_results = self.retriever.retrieve(query, top_k=10)
        return expanded_results
    
    def _check_confidence(self, results):
        """Check confidence in retrieval results."""
        if not results:
            return 0.0
        
        # Simple confidence based on score spread
        scores = [score for doc, score in results]
        if len(scores) < 2:
            return 0.5
        
        # Higher confidence if top result is much better than others
        return min(1.0, scores[0] / (scores[1] + 0.1))
```

## RAG Best Practices

1. **Quality Knowledge Base**: Ensure documents are accurate and well-structured
2. **Effective Retrieval**: Use appropriate retrieval strategies for your use case
3. **Context Management**: Handle context length limits effectively
4. **Citation**: Provide sources for transparency and verification
5. **Evaluation**: Measure both retrieval and generation quality

## Summary

In this lab, you learned:

- **RAG Fundamentals**: Understanding retrieval-augmented generation
- **Basic RAG**: Implementing simple RAG systems
- **Advanced Techniques**: Query expansion, hybrid retrieval, reranking
- **Multi-Source RAG**: Retrieving from multiple knowledge sources
- **Optimization**: Evaluating and improving RAG systems
- **Best Practices**: Guidelines for production RAG systems

## Next Steps

Proceed to [Lab 9: Building AI Agents](lab-09-ai-agents.md) to learn about creating autonomous AI systems with DSPy.

## Challenge Project

Build a comprehensive RAG system that:
1. Supports multiple retrieval strategies (vector, keyword, hybrid)
2. Implements query expansion and reranking
3. Provides citations and source attribution
4. Evaluates both retrieval and generation quality
5. Adapts retrieval based on query complexity

This will demonstrate advanced RAG implementation skills!