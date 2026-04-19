# Lab 3: Signatures and Modules

## Overview

This lab dives deep into DSPy's core building blocks: Signatures and Modules. Mastering these components is essential for building complex, reusable DSPy programs.

## Learning Objectives

- Master signature design patterns
- Understand different module types and when to use them
- Build custom modules for specific tasks
- Compose modules into complex programs
- Optimize module performance

## Prerequisites

- Completed Lab 2: Your First DSPy Program
- Comfortable with basic DSPy concepts
- Understanding of Python classes and functions

## Signatures: Deep Dive

### Signature Anatomy

A signature defines the interface between your program and the LLM:

```python
import dspy

class MySignature(dspy.Signature):
    """Clear description of what this signature does."""
    
    # Input fields
    input_field = dspy.InputField(
        desc="clear description of expected input",
        type=str  # Optional type hint
    )
    
    # Output fields
    output_field = dspy.OutputField(
        desc="clear description of expected output",
        type=str
    )
```

### Signature Design Principles

1. **Clear Descriptions**: Be specific about what each field represents
2. **Appropriate Types**: Use type hints for better validation
3. **Single Responsibility**: Each signature should do one thing well
4. **Reusability**: Design signatures that can be used in multiple contexts

### Advanced Signature Patterns

#### Pattern 1: Conditional Outputs

```python
class ConditionalOutput(dspy.Signature):
    """Generate different outputs based on conditions."""
    input_text = dspy.InputField(desc="text to process")
    category = dspy.OutputField(desc="category classification")
    explanation = dspy.OutputField(desc="explanation if category is uncertain")
    confidence = dspy.OutputField(desc="confidence score")
```

#### Pattern 2: Structured Outputs

```python
class StructuredExtraction(dspy.Signature):
    """Extract structured information from text."""
    text = dspy.InputField(desc="unstructured text")
    entities = dspy.OutputField(desc="list of extracted entities")
    relationships = dspy.OutputField(desc="relationships between entities")
    summary = dspy.OutputField(desc="brief summary of extracted info")
```

#### Pattern 3: Multi-Step Reasoning

```python
class MultiStepReasoning(dspy.Signature):
    """Perform complex reasoning with multiple steps."""
    problem = dspy.InputField(desc="complex problem")
    analysis = dspy.OutputField(desc="step 1: problem analysis")
    approach = dspy.OutputField(desc="step 2: solution approach")
    solution = dspy.OutputField(desc="step 3: final solution")
    verification = dspy.OutputField(desc="step 4: solution verification")
```

## Modules: Core Building Blocks

### Module Types Overview

DSPy provides several module types for different use cases:

#### 1. Predict (Basic)

Simple input-output transformation:

```python
class SimpleQA(dspy.Signature):
    """Answer simple questions."""
    question = dspy.InputField(desc="question to answer")
    answer = dspy.OutputField(desc="answer")

basic_qa = dspy.Predict(SimpleQA)
result = basic_qa(question="What is the capital of France?")
```

**When to use**: Simple, direct tasks without complex reasoning

#### 2. Chain of Thought

Reasoning before output:

```python
class ComplexQA(dspy.Signature):
    """Answer questions requiring reasoning."""
    question = dspy.InputField(desc="complex question")
    reasoning = dspy.OutputField(desc="step-by-step reasoning")
    answer = dspy.OutputField(desc="final answer")

complex_qa = dspy.ChainOfThought(ComplexQA)
result = complex_qa(question="If a train travels at 60 mph for 2.5 hours, how far does it travel?")
```

**When to use**: Tasks that benefit from explicit reasoning

#### 3. ReAct

Multi-step reasoning with actions:

```python
class ToolUse(dspy.Signature):
    """Use tools to answer questions."""
    question = dspy.InputField(desc="question requiring tools")
    tool_calls = dspy.OutputField(desc="tools to use and parameters")
    observations = dspy.OutputField(desc="results from tool calls")
    answer = dspy.OutputField(desc="final answer based on observations")

react_agent = dspy.ReAct(ToolUse)
```

**When to use**: Tasks requiring external tools or APIs

#### 4. Program of Thought

Complex reasoning chains:

```python
class ComplexProblem(dspy.Signature):
    """Solve complex multi-step problems."""
    problem = dspy.InputField(desc="complex problem statement")
    decomposition = dspy.OutputField(desc="break down into sub-problems")
    sub_solutions = dspy.OutputField(desc="solve each sub-problem")
    integration = dspy.OutputField(desc="integrate sub-solutions")
    final_answer = dspy.OutputField(desc="final integrated answer")

pot_solver = dspy.ProgramOfThought(ComplexProblem)
```

**When to use**: Very complex problems requiring structured thinking

## Exercise 1: Custom Module Composition

Build a custom program by composing multiple modules:

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define signatures
class TextCleaner(dspy.Signature):
    """Clean and normalize text."""
    raw_text = dspy.InputField(desc="raw, messy text")
    clean_text = dspy.OutputField(desc="cleaned, normalized text")

class InformationExtractor(dspy.Signature):
    """Extract key information from text."""
    text = dspy.InputField(desc="text to analyze")
    key_info = dspy.OutputField(desc="key information extracted")

class Summarizer(dspy.Signature):
    """Summarize the key information."""
    information = dspy.InputField(desc="information to summarize")
    summary = dspy.OutputField(desc="concise summary")

# Create modules
cleaner = dspy.Predict(TextCleaner)
extractor = dspy.ChainOfThought(InformationExtractor)
summarizer = dspy.Predict(Summarizer)

# Compose into a pipeline
def analyze_text(raw_text):
    # Step 1: Clean the text
    clean_result = cleaner(raw_text=raw_text)
    cleaned_text = clean_result.clean_text
    
    # Step 2: Extract information
    extract_result = extractor(text=cleaned_text)
    key_info = extract_result.key_info
    
    # Step 3: Summarize
    summary_result = summarizer(information=key_info)
    
    return {
        "cleaned_text": cleaned_text,
        "key_info": key_info,
        "summary": summary_result.summary
    }

# Test the pipeline
messy_text = """
THIS IS A MESSY TEXT!!! It has... weird punctuation and CAPS. 
But it contains important info about AI and machine learning.
"""

result = analyze_text(messy_text)
print(f"Cleaned: {result['cleaned_text']}")
print(f"Key Info: {result['key_info']}")
print(f"Summary: {result['summary']}")
```

## Exercise 2: Conditional Module Selection

Build a program that selects modules based on input:

```python
class SimpleTask(dspy.Signature):
    """Handle simple tasks."""
    task = dspy.InputField(desc="simple task description")
    result = dspy.OutputField(desc="task result")

class ComplexTask(dspy.Signature):
    """Handle complex tasks with reasoning."""
    task = dspy.InputField(desc="complex task description")
    analysis = dspy.OutputField(desc="task analysis")
    result = dspy.OutputField(desc="task result")

class TaskClassifier(dspy.Signature):
    """Classify task complexity."""
    task = dspy.InputField(desc="task to classify")
    complexity = dspy.OutputField(desc="simple or complex")
    reasoning = dspy.OutputField(desc="reasoning for classification")

# Create modules
simple_handler = dspy.Predict(SimpleTask)
complex_handler = dspy.ChainOfThought(ComplexTask)
classifier = dspy.Predict(TaskClassifier)

def adaptive_task_handler(task):
    # Classify task complexity
    classification = classifier(task=task)
    
    # Route to appropriate handler
    if classification.complexity.lower() == "complex":
        result = complex_handler(task=task)
        return {
            "handler": "complex",
            "analysis": result.analysis,
            "result": result.result
        }
    else:
        result = simple_handler(task=task)
        return {
            "handler": "simple",
            "result": result.result
        }

# Test with different tasks
simple_task = "Calculate 2 + 2"
complex_task = "Design a machine learning pipeline for image classification"

print(f"Simple task result: {adaptive_task_handler(simple_task)}")
print(f"Complex task result: {adaptive_task_handler(complex_task)}")
```

## Exercise 3: Custom Module Class

Create a custom DSPy module class:

```python
import dspy

class CustomAnalysisModule(dspy.Module):
    """Custom module for text analysis."""
    
    def __init__(self):
        super().__init__()
        # Initialize sub-modules
        self.sentiment = dspy.Predict(
            dspy.Signature("text -> sentiment")
        )
        self.keywords = dspy.Predict(
            dspy.Signature("text -> keywords")
        )
    
    def forward(self, text):
        # Run sentiment analysis
        sentiment_result = self.sentiment(text=text)
        
        # Extract keywords
        keywords_result = self.keywords(text=text)
        
        # Combine results
        return dspy.Prediction(
            sentiment=sentiment_result.sentiment,
            keywords=keywords_result.keywords,
            text=text
        )

# Use the custom module
analyzer = CustomAnalysisModule()
result = analyzer(text="I love DSPy! It's amazing for building AI systems.")
print(f"Sentiment: {result.sentiment}")
print(f"Keywords: {result.keywords}")
```

## Module Optimization

### 1. Parameter Tuning

```python
# Configure module parameters
lm = dspy.OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,  # Control randomness
    max_tokens=500    # Limit output length
)
dspy.settings.configure(lm=lm)
```

### 2. Caching Module Results

```python
from functools import lru_cache

class CachedModule(dspy.Module):
    def __init__(self, base_module):
        self.base_module = base_module
        self._cache = {}
    
    def forward(self, **kwargs):
        cache_key = str(sorted(kwargs.items()))
        if cache_key not in self._cache:
            self._cache[cache_key] = self.base_module(**kwargs)
        return self._cache[cache_key]
```

### 3. Batch Processing

```python
def batch_process(module, inputs, batch_size=10):
    """Process inputs in batches for efficiency."""
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        batch_results = [module(**item) for item in batch]
        results.extend(batch_results)
    return results
```

## Best Practices

1. **Start Simple**: Begin with basic modules, add complexity as needed
2. **Test Incrementally**: Test each module before composing
3. **Document Signatures**: Clear docstrings prevent confusion
4. **Handle Errors**: Add error handling in custom modules
5. **Monitor Performance**: Track token usage and latency

## Summary

In this lab, you mastered:

- **Signature Design**: Advanced patterns for different use cases
- **Module Types**: Understanding when to use each module type
- **Module Composition**: Building complex programs from simple modules
- **Custom Modules**: Creating your own module classes
- **Optimization**: Techniques for improving module performance

## Next Steps

Proceed to [Lab 4: Data and Evaluation](lab-04-data-evaluation.md) to learn how to measure and improve your DSPy programs.

## Challenge Project

Build a modular document processing system that:
1. Cleans and normalizes input text
2. Classifies document type
3. Extracts relevant information based on type
4. Summarizes findings
5. Provides confidence scores

Use custom modules and demonstrate proper composition patterns!