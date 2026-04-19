# Lab 4: Data and Evaluation

## Overview

This lab focuses on the critical aspects of data management and evaluation in DSPy. Understanding how to work with data and measure program performance is essential for building effective AI systems.

## Learning Objectives

- Understand data requirements for DSPy programs
- Create effective training and test datasets
- Implement evaluation metrics
- Measure program performance accurately
- Use data to improve program quality

## Prerequisites

- Completed Lab 3: Signatures and Modules
- Basic understanding of data structures
- Familiarity with evaluation concepts

## Data in DSPy

### Why Data Matters

DSPy programs improve with data through:
1. **Few-shot examples**: Better prompts
2. **Training data**: Optimization
3. **Evaluation data**: Performance measurement
4. **Test data**: Validation

### Data Types

#### 1. Training Data

Used for optimizing DSPy programs:

```python
train_data = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe.",
        "answer": "Paris"
    },
    {
        "question": "What is 2 + 2?",
        "context": "Basic arithmetic.",
        "answer": "4"
    },
    # ... more examples
]
```

#### 2. Validation Data

Used for tuning and selection:

```python
val_data = [
    {
        "question": "What is the largest planet?",
        "context": "Solar system information.",
        "answer": "Jupiter"
    },
    # ... more examples
]
```

#### 3. Test Data

Used for final evaluation:

```python
test_data = [
    {
        "question": "What is the speed of light?",
        "context": "Physics constants.",
        "answer": "299,792,458 meters per second"
    },
    # ... more examples
]
```

## Creating Quality Datasets

### Data Collection Strategies

#### Strategy 1: Manual Curation

```python
def manually_curate_dataset():
    """Create a dataset by manually selecting examples."""
    dataset = []
    
    # Select diverse, representative examples
    examples = [
        # Easy examples
        {"input": "2+2", "output": "4"},
        # Medium examples  
        {"input": "What is AI?", "output": "Artificial Intelligence"},
        # Hard examples
        {"input": "Explain quantum computing", "output": "..."}
    ]
    
    return examples
```

#### Strategy 2: Synthetic Generation

```python
import dspy

def generate_synthetic_data(base_examples, n=100):
    """Generate synthetic examples from base cases."""
    generator = dspy.Predict(
        dspy.Signature("example -> similar_example")
    )
    
    synthetic_data = []
    for example in base_examples:
        for _ in range(n // len(base_examples)):
            new_example = generator(example=str(example))
            synthetic_data.append(new_example.similar_example)
    
    return synthetic_data
```

#### Strategy 3: Data Augmentation

```python
def augment_data(original_data):
    """Augment data with variations."""
    augmenter = dspy.Predict(
        dspy.Signature("text -> augmented_text")
    )
    
    augmented_data = []
    for item in original_data:
        # Create variations
        variations = [
            augmenter(text=item),
            augmenter(text=item.upper()),
            augmenter(text=item.lower())
        ]
        augmented_data.extend(variations)
    
    return original_data + augmented_data
```

### Data Quality Checks

```python
def validate_dataset(dataset, required_fields):
    """Validate dataset quality."""
    issues = []
    
    for i, example in enumerate(dataset):
        # Check required fields
        for field in required_fields:
            if field not in example:
                issues.append(f"Example {i}: Missing field '{field}'")
        
        # Check for empty values
        for field, value in example.items():
            if not value or str(value).strip() == "":
                issues.append(f"Example {i}: Empty field '{field}'")
        
        # Check for duplicates
        if dataset.count(example) > 1:
            issues.append(f"Example {i}: Duplicate example")
    
    return issues
```

## Evaluation Metrics

### Common Evaluation Metrics

#### 1. Exact Match

```python
def exact_match(predicted, ground_truth):
    """Check if prediction exactly matches ground truth."""
    return predicted.strip().lower() == ground_truth.strip().lower()

# Usage
predicted = "Paris"
ground_truth = "Paris"
score = exact_match(predicted, ground_truth)  # True
```

#### 2. Semantic Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_similarity(predicted, ground_truth):
    """Calculate semantic similarity between texts."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([predicted, ground_truth])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return similarity

# Usage
predicted = "The capital of France is Paris"
ground_truth = "Paris is France's capital city"
score = semantic_similarity(predicted, ground_truth)  # High similarity
```

#### 3. F1 Score for Extraction

```python
def calculate_f1(predicted_entities, ground_truth_entities):
    """Calculate F1 score for entity extraction."""
    predicted_set = set(predicted_entities)
    ground_truth_set = set(ground_truth_entities)
    
    true_positives = len(predicted_set & ground_truth_set)
    false_positives = len(predicted_set - ground_truth_set)
    false_negatives = len(ground_truth_set - predicted_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1
```

## Exercise 1: Build Evaluation Pipeline

Create a complete evaluation pipeline:

```python
import dspy
from typing import List, Dict

class EvaluationPipeline:
    """Comprehensive evaluation pipeline for DSPy programs."""
    
    def __init__(self, program, test_data):
        self.program = program
        self.test_data = test_data
        self.results = []
    
    def evaluate(self, metric_func):
        """Evaluate program against test data."""
        scores = []
        
        for example in self.test_data:
            # Get prediction
            prediction = self.program(**example)
            
            # Calculate score
            score = metric_func(prediction, example)
            scores.append(score)
            
            # Store results
            self.results.append({
                "input": example,
                "prediction": prediction,
                "score": score
            })
        
        return {
            "mean_score": sum(scores) / len(scores),
            "individual_scores": scores,
            "detailed_results": self.results
        }
    
    def analyze_failures(self, threshold=0.5):
        """Analyze failed examples."""
        failures = [
            result for result in self.results 
            if result["score"] < threshold
        ]
        
        return {
            "failure_count": len(failures),
            "failure_rate": len(failures) / len(self.results),
            "failures": failures
        }

# Usage
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

class SimpleQA(dspy.Signature):
    """Answer questions."""
    question = dspy.InputField(desc="question")
    answer = dspy.OutputField(desc="answer")

qa_program = dspy.Predict(SimpleQA)

test_data = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
]

def simple_metric(prediction, ground_truth):
    return 1.0 if prediction.answer == ground_truth["answer"] else 0.0

pipeline = EvaluationPipeline(qa_program, test_data)
results = pipeline.evaluate(simple_metric)
print(f"Mean Score: {results['mean_score']}")
```

## Exercise 2: Data Splitting and Cross-Validation

Implement proper data splitting:

```python
import random
from typing import List, Tuple

def split_data(data: List[Dict], 
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15) -> Tuple[List, List, List]:
    """Split data into train, validation, and test sets."""
    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Calculate split points
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test

def cross_validate(program, data, k=5, metric_func=None):
    """Perform k-fold cross-validation."""
    fold_size = len(data) // k
    scores = []
    
    for i in range(k):
        # Split into train and test for this fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test_data = data[test_start:test_end]
        train_data = data[:test_start] + data[test_end:]
        
        # Train on train data (if program supports it)
        # For now, just evaluate on test data
        fold_score = evaluate_on_data(program, test_data, metric_func)
        scores.append(fold_score)
    
    return {
        "mean_score": sum(scores) / len(scores),
        "std_score": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5,
        "fold_scores": scores
    }
```

## Exercise 3: Advanced Metrics

Implement sophisticated evaluation metrics:

```python
import re

def extract_answer(text):
    """Extract answer from text."""
    # Remove common prefixes
    text = re.sub(r'^(answer|response|result):\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def normalize_text(text):
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def relaxed_exact_match(predicted, ground_truth):
    """Relaxed exact match with normalization."""
    norm_pred = normalize_text(predicted)
    norm_truth = normalize_text(ground_truth)
    return norm_pred == norm_truth

def contains_answer(predicted, ground_truth):
    """Check if predicted text contains the answer."""
    norm_pred = normalize_text(predicted)
    norm_truth = normalize_text(ground_truth)
    return norm_truth in norm_pred

def multi_metric_evaluation(prediction, ground_truth):
    """Evaluate using multiple metrics."""
    metrics = {
        "exact_match": exact_match(prediction, ground_truth),
        "relaxed_match": relaxed_exact_match(prediction, ground_truth),
        "contains_answer": contains_answer(prediction, ground_truth),
        "semantic_similarity": semantic_similarity(prediction, ground_truth)
    }
    
    # Calculate composite score
    weights = {
        "exact_match": 0.4,
        "relaxed_match": 0.3,
        "contains_answer": 0.2,
        "semantic_similarity": 0.1
    }
    
    composite_score = sum(
        metrics[metric] * weights[metric] 
        for metric in metrics
    )
    
    return {
        "individual_metrics": metrics,
        "composite_score": composite_score
    }
```

## Data-Centric Improvement

### Using Evaluation to Improve Programs

```python
def iterative_improvement(program, train_data, val_data, iterations=5):
    """Iteratively improve program based on evaluation."""
    best_score = 0
    best_program = program
    
    for i in range(iterations):
        # Evaluate current program
        current_score = evaluate_on_data(program, val_data)
        
        print(f"Iteration {i}: Score = {current_score}")
        
        # If improved, save this version
        if current_score > best_score:
            best_score = current_score
            best_program = program
        
        # Use failures to improve (this would involve optimization)
        # For now, just continue
    
    return best_program, best_score
```

## Best Practices

1. **Data Quality**: Ensure high-quality, diverse training data
2. **Proper Splitting**: Use proper train/val/test splits
3. **Multiple Metrics**: Evaluate using multiple metrics
4. **Error Analysis**: Analyze failures to understand weaknesses
5. **Iterative Improvement**: Use evaluation to guide improvements

## Summary

In this lab, you learned:

- **Data Management**: Creating and managing quality datasets
- **Data Collection**: Strategies for gathering training data
- **Evaluation Metrics**: Implementing various evaluation approaches
- **Evaluation Pipelines**: Building comprehensive evaluation systems
- **Data-Centric Improvement**: Using data to improve programs

## Next Steps

Proceed to [Lab 5: Optimization Strategies](lab-05-optimization.md) to learn how to make your DSPy programs self-improving.

## Challenge Project

Create a comprehensive evaluation framework that:
1. Supports multiple evaluation metrics
2. Provides detailed failure analysis
3. Generates evaluation reports
4. Suggests improvements based on failures
5. Works with different DSPy program types

This will solidify your understanding of data and evaluation in DSPy!