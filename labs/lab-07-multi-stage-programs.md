# Lab 7: Multi-Stage Programs

## Overview

This lab teaches you how to build sophisticated AI systems by composing multiple DSPy modules into multi-stage pipelines. You'll learn to design and implement complex programs that handle real-world tasks.

## Learning Objectives

- Understand multi-stage program architecture
- Design effective program pipelines
- Compose modules for complex tasks
- Handle data flow between stages
- Optimize multi-stage programs

## Prerequisites

- Completed Lab 6: Chain of Thought
- Experience with basic DSPy modules
- Understanding of pipeline concepts

## Multi-Stage Architecture

### What are Multi-Stage Programs?

Multi-stage programs break complex tasks into sequential steps, where each stage:
1. **Processes input** from the previous stage
2. **Performs specific operations**
3. **Passes output** to the next stage
4. **Handles errors** gracefully

### Benefits of Multi-Stage Design

- **Modularity**: Each stage has a clear responsibility
- **Reusability**: Stages can be reused in different pipelines
- **Debugging**: Easier to isolate and fix issues
- **Scalability**: Stages can be optimized independently
- **Flexibility**: Easy to modify or replace stages

## Basic Multi-Stage Pipeline

### Simple Two-Stage Pipeline

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Stage 1: Text Cleaning
class TextCleaner(dspy.Signature):
    """Clean and normalize text."""
    raw_text = dspy.InputField(desc="raw, messy text")
    clean_text = dspy.OutputField(desc="cleaned, normalized text")

# Stage 2: Summarization
class TextSummarizer(dspy.Signature):
    """Summarize the cleaned text."""
    clean_text = dspy.InputField(desc="cleaned text to summarize")
    summary = dspy.OutputField(desc="concise summary")

# Create stages
cleaner = dspy.Predict(TextCleaner)
summarizer = dspy.Predict(TextSummarizer)

# Compose pipeline
def clean_and_summarize(raw_text):
    # Stage 1: Clean
    clean_result = cleaner(raw_text=raw_text)
    cleaned = clean_result.clean_text
    
    # Stage 2: Summarize
    summary_result = summarizer(clean_text=cleaned)
    summary = summary_result.summary
    
    return {
        "cleaned_text": cleaned,
        "summary": summary
    }

# Test
messy_text = "THIS IS MESSY TEXT!!! With... weird punctuation."
result = clean_and_summarize(messy_text)
print(f"Cleaned: {result['cleaned_text']}")
print(f"Summary: {result['summary']}")
```

## Exercise 1: Three-Stage Document Pipeline

Build a comprehensive document processing pipeline:

```python
# Stage 1: Document Classification
class DocumentClassifier(dspy.Signature):
    """Classify the document type."""
    document = dspy.InputField(desc="document text")
    document_type = dspy.OutputField(desc="type: article, report, email, etc.")
    confidence = dspy.OutputField(desc="confidence in classification")

# Stage 2: Information Extraction
class InformationExtractor(dspy.Signature):
    """Extract key information based on document type."""
    document = dspy.InputField(desc="document text")
    document_type = dspy.InputField(desc="document type")
    key_information = dspy.OutputField(desc="extracted key information")
    entities = dspy.OutputField(desc="important entities mentioned")

# Stage 3: Summary Generation
class ContextualSummarizer(dspy.Signature):
    """Generate summary considering document type and extracted info."""
    document = dspy.InputField(desc="document text")
    document_type = dspy.InputField(desc="document type")
    key_information = dspy.InputField(desc="key information")
    summary = dspy.OutputField(desc="contextual summary")

# Create stages
classifier = dspy.Predict(DocumentClassifier)
extractor = dspy.ChainOfThought(InformationExtractor)
summarizer = dspy.ChainOfThought(ContextualSummarizer)

# Compose pipeline
def process_document(document):
    # Stage 1: Classify
    classification = classifier(document=document)
    doc_type = classification.document_type
    
    # Stage 2: Extract information
    extraction = extractor(
        document=document,
        document_type=doc_type
    )
    
    # Stage 3: Generate contextual summary
    summary = summarizer(
        document=document,
        document_type=doc_type,
        key_information=extraction.key_information
    )
    
    return {
        "document_type": doc_type,
        "key_information": extraction.key_information,
        "entities": extraction.entities,
        "summary": summary.summary
    }

# Test
sample_doc = """
Q3 Financial Report
Our company showed strong growth in Q3 with revenue increasing by 25% compared to Q2.
The main drivers were increased sales in the Asian market and successful product launches.
CEO John Smith stated that this trend is expected to continue into Q4.
"""

result = process_document(sample_doc)
print(f"Document Type: {result['document_type']}")
print(f"Key Information: {result['key_information']}")
print(f"Entities: {result['entities']}")
print(f"Summary: {result['summary']}")
```

## Exercise 2: Conditional Routing

Build a pipeline with conditional routing based on stage outputs:

```python
class TaskRouter(dspy.Signature):
    """Route tasks to appropriate handlers."""
    task_description = dspy.InputField(desc="description of the task")
    task_type = dspy.OutputField(desc="type: math, writing, analysis, coding")
    reasoning = dspy.OutputField(desc="reasoning for classification")

class MathHandler(dspy.Signature):
    """Handle mathematical tasks."""
    problem = dspy.InputField(desc="math problem")
    solution = dspy.OutputField(desc="step-by-step solution")
    answer = dspy.OutputField(desc="final answer")

class WritingHandler(dspy.Signature):
    """Handle writing tasks."""
    prompt = dspy.InputField(desc="writing prompt")
    content = dspy.OutputField(desc="written content")
    style = dspy.OutputField(desc="writing style used")

class AnalysisHandler(dspy.Signature):
    """Handle analysis tasks."""
    data = dspy.InputField(desc="data to analyze")
    analysis = dspy.OutputField(desc="analysis results")
    insights = dspy.OutputField(desc="key insights")

# Create modules
router = dspy.Predict(TaskRouter)
math_handler = dspy.ChainOfThought(MathHandler)
writing_handler = dspy.Predict(WritingHandler)
analysis_handler = dspy.ChainOfThought(AnalysisHandler)

def adaptive_task_processor(task_description):
    # Stage 1: Route task
    routing = router(task_description=task_description)
    task_type = routing.task_type.lower()
    
    # Stage 2: Route to appropriate handler
    if "math" in task_type:
        result = math_handler(problem=task_description)
        return {"handler": "math", "result": result}
    elif "writing" in task_type:
        result = writing_handler(prompt=task_description)
        return {"handler": "writing", "result": result}
    elif "analysis" in task_type:
        result = analysis_handler(data=task_description)
        return {"handler": "analysis", "result": result}
    else:
        return {"handler": "unknown", "result": "Task type not recognized"}

# Test with different tasks
tasks = [
    "Calculate the area of a circle with radius 5",
    "Write a short story about a robot learning to love",
    "Analyze the trends in this sales data: Q1: 100, Q2: 150, Q3: 200"
]

for task in tasks:
    result = adaptive_task_processor(task)
    print(f"Task: {task}")
    print(f"Handler: {result['handler']}")
    print(f"Result: {result['result']}")
    print("-" * 50)
```

## Exercise 3: Parallel Processing

Build a pipeline with parallel stages:

```python
class SentimentAnalyzer(dspy.Signature):
    """Analyze sentiment of text."""
    text = dspy.InputField(desc="text to analyze")
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")
    confidence = dspy.OutputField(desc="confidence score")

class TopicExtractor(dspy.Signature):
    """Extract main topics from text."""
    text = dspy.InputField(desc="text to analyze")
    topics = dspy.OutputField(desc="main topics or themes")

class EntityRecognizer(dspy.Signature):
    """Recognize named entities in text."""
    text = dspy.InputField(desc="text to analyze")
    entities = dspy.OutputField(desc="named entities found")

class ResultAggregator(dspy.Signature):
    """Aggregate results from parallel analyses."""
    sentiment = dspy.InputField(desc="sentiment analysis result")
    topics = dspy.InputField(desc="topic extraction result")
    entities = dspy.InputField(desc="entity recognition result")
    comprehensive_analysis = dspy.OutputField(desc="combined analysis")

# Create parallel stages
sentiment_analyzer = dspy.Predict(SentimentAnalyzer)
topic_extractor = dspy.Predict(TopicExtractor)
entity_recognizer = dspy.Predict(EntityRecognizer)
aggregator = dspy.ChainOfThought(ResultAggregator)

def parallel_text_analysis(text):
    # Run parallel analyses
    sentiment_result = sentiment_analyzer(text=text)
    topic_result = topic_extractor(text=text)
    entity_result = entity_recognizer(text=text)
    
    # Aggregate results
    aggregated = aggregator(
        sentiment=sentiment_result.sentiment,
        topics=topic_result.topics,
        entities=entity_result.entities
    )
    
    return {
        "sentiment": sentiment_result.sentiment,
        "topics": topic_result.topics,
        "entities": entity_result.entities,
        "comprehensive_analysis": aggregated.comprehensive_analysis
    }

# Test
sample_text = """
The new AI model released by OpenAI has shown impressive results in various benchmarks.
Researchers are excited about the potential applications in healthcare and education.
However, some experts raise concerns about ethical implications and regulation.
"""

result = parallel_text_analysis(sample_text)
print(f"Sentiment: {result['sentiment']}")
print(f"Topics: {result['topics']}")
print(f"Entities: {result['entities']}")
print(f"Comprehensive Analysis: {result['comprehensive_analysis']}")
```

## Advanced Pipeline Patterns

### 1. Feedback Loops

```python
class QualityChecker(dspy.Signature):
    """Check the quality of generated content."""
    content = dspy.InputField(desc="content to check")
    quality_score = dspy.OutputField(desc="quality score from 0-1")
    issues = dspy.OutputField(desc="identified issues")
    needs_improvement = dspy.OutputField(desc="yes if needs improvement")

class ContentImprover(dspy.Signature):
    """Improve content based on feedback."""
    content = dspy.InputField(desc="content to improve")
    issues = dspy.InputField(desc="issues to address")
    improved_content = dspy.OutputField(desc="improved version")

quality_checker = dspy.Predict(QualityChecker)
content_improver = dspy.Predict(ContentImprover)

def iterative_content_generator(initial_content, max_iterations=3):
    content = initial_content
    
    for iteration in range(max_iterations):
        # Check quality
        check = quality_checker(content=content)
        
        # If quality is good enough, stop
        if check.needs_improvement == "no":
            break
        
        # Improve content
        improvement = content_improver(
            content=content,
            issues=check.issues
        )
        content = improvement.improved_content
    
    return content
```

### 2. Ensemble Methods

```python
class EnsembleAnalyzer(dspy.Signature):
    """Combine results from multiple analyses."""
    analysis1 = dspy.InputField(desc="first analysis")
    analysis2 = dspy.InputField(desc="second analysis")
    analysis3 = dspy.InputField(desc="third analysis")
    consensus = dspy.OutputField(desc="consensus analysis")
    confidence = dspy.OutputField(desc="confidence in consensus")

def ensemble_analysis(text, analyzers, n=3):
    """Run multiple analyzers and combine results."""
    analyses = []
    
    # Run multiple analyses
    for i in range(n):
        analyzer = analyzers[i % len(analyzers)]
        result = analyzer(text=text)
        analyses.append(result)
    
    # Combine results
    ensemble = dspy.Predict(EnsembleAnalyzer)
    consensus = ensemble(
        analysis1=str(analyses[0]),
        analysis2=str(analyses[1]),
        analysis3=str(analyses[2])
    )
    
    return consensus
```

### 3. Hierarchical Pipelines

```python
class SubTaskBreaker(dspy.Signature):
    """Break complex task into subtasks."""
    complex_task = dspy.InputField(desc="complex task")
    subtasks = dspy.OutputField(desc="list of subtasks")
    dependencies = dspy.OutputField(desc="dependencies between subtasks")

class SubTaskExecutor(dspy.Signature):
    """Execute a specific subtask."""
    subtask = dspy.InputField(desc="subtask to execute")
    context = dspy.InputField(desc="context from previous tasks")
    result = dspy.OutputField(desc="subtask result")

class ResultIntegrator(dspy.Signature):
    """Integrate results from subtasks."""
    subtask_results = dspy.InputField(desc="results from all subtasks")
    final_result = dspy.OutputField(desc="integrated final result")

def hierarchical_executor(complex_task):
    # Break down task
    breaker = dspy.Predict(SubTaskBreaker)
    breakdown = breaker(complex_task=complex_task)
    
    # Execute subtasks (simplified)
    executor = dspy.Predict(SubTaskExecutor)
    results = []
    context = ""
    
    for subtask in breakdown.subtasks.split('\n'):
        result = executor(subtask=subtask, context=context)
        results.append(result.result)
        context += f"{subtask}: {result.result}\n"
    
    # Integrate results
    integrator = dspy.Predict(ResultIntegrator)
    final = integrator(subtask_results="\n".join(results))
    
    return final.final_result
```

## Pipeline Optimization

### 1. Stage-wise Optimization

```python
def optimize_pipeline_stages(pipeline_stages, train_data, val_data):
    """Optimize each stage independently."""
    optimized_stages = []
    
    for i, stage in enumerate(pipeline_stages):
        print(f"Optimizing stage {i}")
        
        # Optimize this stage
        optimizer = BootstrapFewShot(metric=stage_metric)
        optimized_stage = optimizer.compile(
            student=stage,
            trainset=train_data
        )
        
        # Validate
        score = evaluate_stage(optimized_stage, val_data)
        print(f"Stage {i} score: {score}")
        
        optimized_stages.append(optimized_stage)
    
    return optimized_stages
```

### 2. Caching Intermediate Results

```python
from functools import lru_cache

class CachedPipeline:
    def __init__(self, stages):
        self.stages = stages
        self.cache = {}
    
    def execute(self, input_data):
        cache_key = str(input_data)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Execute pipeline
        result = input_data
        for stage in self.stages:
            result = stage(**result)
        
        self.cache[cache_key] = result
        return result
```

### 3. Error Recovery

```python
class ResilientPipeline:
    def __init__(self, stages, max_retries=3):
        self.stages = stages
        self.max_retries = max_retries
    
    def execute_with_retry(self, input_data):
        result = input_data
        
        for i, stage in enumerate(self.stages):
            retries = 0
            while retries < self.max_retries:
                try:
                    result = stage(**result)
                    break
                except Exception as e:
                    retries += 1
                    if retries == self.max_retries:
                        print(f"Stage {i} failed after {self.max_retries} retries")
                        raise
                    print(f"Stage {i} failed, retrying... ({retries}/{self.max_retries})")
        
        return result
```

## Best Practices

1. **Clear Stage Boundaries**: Each stage should have a single, well-defined responsibility
2. **Data Validation**: Validate inputs and outputs between stages
3. **Error Handling**: Implement robust error handling and recovery
4. **Monitoring**: Track performance and errors at each stage
5. **Testing**: Test each stage independently and the pipeline as a whole

## Summary

In this lab, you learned:

- **Multi-Stage Architecture**: Understanding pipeline design principles
- **Basic Pipelines**: Building simple multi-stage programs
- **Advanced Patterns**: Conditional routing, parallel processing, feedback loops
- **Optimization**: Techniques for improving pipeline performance
- **Best Practices**: Guidelines for building robust multi-stage systems

## Next Steps

Proceed to [Lab 8: Retrieval Augmented Generation](lab-08-rag.md) to learn about building RAG systems with DSPy.

## Challenge Project

Build a comprehensive document intelligence system that:
1. Classifies documents by type
2. Extracts relevant information based on type
3. Performs sentiment analysis
4. Generates contextual summaries
5. Provides recommendations for further action

Use multi-stage pipeline architecture and demonstrate proper composition patterns!