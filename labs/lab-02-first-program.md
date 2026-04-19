# Lab 2: Your First DSPy Program

## Overview

In this lab, you'll build your first complete DSPy program from scratch. This hands-on experience will solidify your understanding of DSPy concepts and prepare you for more advanced applications.

## Learning Objectives

- Build a complete DSPy program end-to-end
- Understand the program development lifecycle
- Test and debug DSPy programs
- Handle errors and edge cases
- Optimize basic program performance

## Prerequisites

- Completed Lab 1: DSPy Fundamentals
- Working DSPy environment
- Basic Python programming skills

## Project: Question Answering System

We'll build a question answering system that can answer questions about a given context.

### Step 1: Define the Problem

Our program should:
1. Take a question and context as input
2. Find the answer within the context
3. Return the answer with confidence

### Step 2: Design the Signature

```python
import dspy

class ContextQA(dspy.Signature):
    """Answer questions based on the provided context."""
    context = dspy.InputField(desc="relevant context or document")
    question = dspy.InputField(desc="question about the context")
    answer = dspy.OutputField(desc="answer to the question")
    confidence = dspy.OutputField(desc="confidence score from 0 to 1")
```

### Step 3: Create the Program

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define signature
class ContextQA(dspy.Signature):
    """Answer questions based on the provided context."""
    context = dspy.InputField(desc="relevant context or document")
    question = dspy.InputField(desc="question about the context")
    answer = dspy.OutputField(desc="answer to the question")
    confidence = dspy.OutputField(desc="confidence score from 0 to 1")

# Create program with Chain of Thought
qa_program = dspy.ChainOfThought(ContextQA)
```

### Step 4: Test with Sample Data

```python
# Sample context and questions
context = """
Python is a high-level, interpreted programming language known for its simplicity. 
It was created by Guido van Rossum and first released in 1991. 
Python supports multiple programming paradigms including object-oriented, 
imperative, and functional programming.
"""

# Test questions
test_questions = [
    "Who created Python?",
    "When was Python first released?",
    "What programming paradigms does Python support?"
]

# Test the program
for question in test_questions:
    result = qa_program(context=context, question=question)
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print("-" * 50)
```

### Step 5: Handle Edge Cases

```python
def safe_qa_answer(context, question, qa_program):
    """Safely answer questions with error handling."""
    try:
        if not context or not question:
            return {"answer": "Invalid input", "confidence": 0.0}
        
        result = qa_program(context=context, question=question)
        return {
            "answer": result.answer,
            "confidence": result.confidence
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "confidence": 0.0}

# Test edge cases
edge_cases = [
    ("", "What is Python?"),  # Empty context
    (context, ""),  # Empty question
    ("Short context", "What is the meaning of life?"),  # Unanswerable
]

for ctx, q in edge_cases:
    result = safe_qa_answer(ctx, q, qa_program)
    print(f"Context: {ctx[:30]}...")
    print(f"Question: {q}")
    print(f"Result: {result}")
    print("-" * 50)
```

## Exercise 1: Text Summarization Program

Build a text summarization program:

```python
import dspy

class TextSummarizer(dspy.Signature):
    """Summarize the given text."""
    text = dspy.InputField(desc="text to summarize")
    summary = dspy.OutputField(desc="concise summary")
    key_points = dspy.OutputField(desc="3-5 key bullet points")

# Create program
summarizer = dspy.ChainOfThought(TextSummarizer)

# Test
long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, 
as opposed to the natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals.
"""

result = summarizer(text=long_text)
print(f"Summary: {result.summary}")
print(f"Key Points: {result.key_points}")
```

## Exercise 2: Sentiment Analysis Program

Build a sentiment analysis program:

```python
class SentimentAnalyzer(dspy.Signature):
    """Analyze the sentiment of the given text."""
    text = dspy.InputField(desc="text to analyze")
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")
    reasoning = dspy.OutputField(desc="reasoning for sentiment classification")
    confidence = dspy.OutputField(desc="confidence from 0 to 1")

analyzer = dspy.ChainOfThought(SentimentAnalyzer)

# Test with different sentiments
test_texts = [
    "I love this product! It's amazing!",
    "This is terrible. I want my money back.",
    "The product is okay, nothing special."
]

for text in test_texts:
    result = analyzer(text=text)
    print(f"Text: {text}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence}")
    print("-" * 50)
```

## Exercise 3: Multi-Task Program

Build a program that handles multiple tasks:

```python
class TextAnalyzer(dspy.Signature):
    """Perform comprehensive text analysis."""
    text = dspy.InputField(desc="text to analyze")
    summary = dspy.OutputField(desc="brief summary")
    sentiment = dspy.OutputField(desc="sentiment analysis")
    topics = dspy.OutputField(desc="main topics or themes")
    language = dspy.OutputField(desc="detected language")

analyzer = dspy.Predict(TextAnalyzer)

sample_text = """
Machine learning is a subset of artificial intelligence that focuses on 
algorithms that can learn from data. It has applications in various fields 
including healthcare, finance, and technology.
"""

result = analyzer(text=sample_text)
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")
print(f"Topics: {result.topics}")
print(f"Language: {result.language}")
```

## Program Development Lifecycle

### 1. Requirements
- Define what the program should do
- Identify inputs and outputs
- Specify constraints and requirements

### 2. Design
- Create appropriate signatures
- Choose the right modules
- Plan the program structure

### 3. Implementation
- Write the DSPy code
- Configure LLM settings
- Set up the program

### 4. Testing
- Test with sample data
- Handle edge cases
- Validate outputs

### 5. Optimization
- Use teleprompters if needed
- Fine-tune parameters
- Improve performance

## Debugging Tips

### 1. Enable Verbose Output
```python
import dspy

# Enable detailed logging
dspy.settings.configure(lm=lm, verbose=True)
```

### 2. Test Step by Step
```python
# Test individual components
result = qa_program(context="test context", question="test question")
print(result)  # Inspect the full result object
```

### 3. Validate Inputs
```python
def validate_input(context, question):
    if not context or len(context) < 10:
        raise ValueError("Context too short")
    if not question or len(question) < 3:
        raise ValueError("Question too short")
    return True
```

### 4. Handle API Errors
```python
import time
from openai import RateLimitError

def safe_call(program, max_retries=3):
    for attempt in range(max_retries):
        try:
            return program()
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Performance Considerations

### 1. Choose the Right Model
```python
# For simple tasks, use smaller/faster models
lm = dspy.OpenAI(model="gpt-3.5-turbo")

# For complex tasks, use more capable models
lm = dspy.OpenAI(model="gpt-4")
```

### 2. Cache Results
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_qa(context, question):
    return qa_program(context=context, question=question)
```

### 3. Batch Processing
```python
# Process multiple questions efficiently
questions = ["Q1", "Q2", "Q3"]
results = [qa_program(context=context, question=q) for q in questions]
```

## Summary

In this lab, you built your first complete DSPy program:

- **Complete Program**: Built a question answering system
- **Error Handling**: Added safety and edge case handling
- **Multiple Exercises**: Created various types of programs
- **Development Lifecycle**: Understood the full development process
- **Debugging**: Learned techniques for troubleshooting
- **Performance**: Considered optimization strategies

## Next Steps

Proceed to [Lab 3: Signatures and Modules](lab-03-signatures-modules.md) to master DSPy's core building blocks.

## Challenge Project

Build a complete document analysis system that:
1. Accepts any document text
2. Provides comprehensive analysis
3. Handles multiple types of questions
4. Includes error handling
5. Optimizes for performance

This will test all the skills you've learned so far!