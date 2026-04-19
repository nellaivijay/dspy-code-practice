# Lab 5: Optimization Strategies

## Overview

This lab explores DSPy's powerful optimization capabilities. You'll learn how to make your programs self-improving through various optimization strategies and teleprompters.

## Learning Objectives

- Understand DSPy optimization concepts
- Master different teleprompter types
- Implement custom optimization strategies
- Use few-shot learning effectively
- Optimize programs for specific tasks

## Prerequisites

- Completed Lab 4: Data and Evaluation
- Understanding of evaluation metrics
- Quality training data available

## Optimization in DSPy

### What is Optimization?

DSPy optimization automatically improves your programs by:
1. **Finding better prompts**: Discovering effective prompt patterns
2. **Selecting examples**: Choosing the best few-shot examples
3. **Tuning parameters**: Optimizing hyperparameters
4. **Architecture search**: Finding the best module combinations

### The Optimization Process

```python
# Basic optimization workflow
import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Define your base program
class MySignature(dspy.Signature):
    """Your task description."""
    input_field = dspy.InputField()
    output_field = dspy.OutputField()

base_program = dspy.Predict(MySignature)

# 2. Define evaluation metric
def evaluation_metric(prediction, ground_truth):
    return 1.0 if prediction.output_field == ground_truth else 0.0

# 3. Create optimizer
optimizer = BootstrapFewShot(metric=evaluation_metric)

# 4. Optimize with training data
optimized_program = optimizer.compile(
    student=base_program,
    trainset=training_data
)

# 5. Use the optimized program
result = optimized_program(input_field="test input")
```

## Teleprompters: Optimization Tools

### 1. BootstrapFewShot

The most commonly used optimizer:

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=your_metric,           # Evaluation metric
    max_bootstrapped_demos=5,     # Number of few-shot examples
    max_labeled_demos=3,          # Number of labeled examples to include
    teacher_settings=None          # Optional teacher model settings
)

optimized = optimizer.compile(
    student=your_program,
    trainset=your_training_data
)
```

**When to use**: General-purpose optimization for most tasks

**How it works**:
1. Uses the LLM to generate candidate examples
2. Evaluates each candidate with your metric
3. Selects the best examples for few-shot learning
4. Compiles an optimized version of your program

### 2. BootstrapFewShotWithRandomSearch

Adds random search for hyperparameter tuning:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=your_metric,
    num_trials=10,                # Number of random trials
    max_bootstrapped_demos=5
)

optimized = optimizer.compile(
    student=your_program,
    trainset=your_training_data
)
```

**When to use**: When you want to explore different hyperparameters

### 3. KNNFewShot

Uses k-nearest neighbors for example selection:

```python
from dspy.teleprompt import KNNFewShot

optimizer = KNNFewShot(
    metric=your_metric,
    k=4,                         # Number of neighbors
    trainset=your_training_data
)

optimized = optimizer.compile(
    student=your_program
)
```

**When to use**: When you have a large dataset and want efficient example selection

### 4. COPRO (Cooperative Prompt Optimization)

Advanced optimization for complex tasks:

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=your_metric,
    verbose=True,
    max_rounds=4                  # Number of optimization rounds
)

optimized = optimizer.compile(
    student=your_program,
    trainset=your_training_data
)
```

**When to use**: Complex tasks requiring sophisticated optimization

## Exercise 1: Basic Optimization

Optimize a simple question answering program:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define signature
class QA(dspy.Signature):
    """Answer questions accurately."""
    question = dspy.InputField(desc="question to answer")
    answer = dspy.OutputField(desc="answer to the question")

# Create base program
base_qa = dspy.Predict(QA)

# Create training data
train_data = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
    {"question": "Largest planet?", "answer": "Jupiter"},
    {"question": "Speed of light?", "answer": "299,792,458 m/s"}
]

# Define metric
def qa_metric(prediction, example):
    return 1.0 if prediction.answer == example["answer"] else 0.0

# Create optimizer
optimizer = BootstrapFewShot(metric=qa_metric, max_bootstrapped_demos=3)

# Optimize
optimized_qa = optimizer.compile(student=base_qa, trainset=train_data)

# Test both versions
test_question = "What is 3+3?"
print(f"Base: {base_qa(question=test_question).answer}")
print(f"Optimized: {optimized_qa(question=test_question).answer}")
```

## Exercise 2: Multi-Metric Optimization

Optimize using multiple evaluation metrics:

```python
def combined_metric(prediction, ground_truth):
    """Combine multiple evaluation metrics."""
    # Exact match
    exact_score = 1.0 if prediction.answer == ground_truth["answer"] else 0.0
    
    # Semantic similarity
    semantic_score = semantic_similarity(prediction.answer, ground_truth["answer"])
    
    # Length appropriateness (not too short or too long)
    pred_len = len(prediction.answer)
    target_len = len(ground_truth["answer"])
    length_score = 1.0 - abs(pred_len - target_len) / max(pred_len, target_len)
    
    # Combined score
    return 0.5 * exact_score + 0.3 * semantic_score + 0.2 * length_score

# Use combined metric in optimization
optimizer = BootstrapFewShot(metric=combined_metric, max_bootstrapped_demos=4)
optimized_qa = optimizer.compile(student=base_qa, trainset=train_data)
```

## Exercise 3: Custom Optimization Strategy

Create a custom optimization strategy:

```python
class CustomOptimizer:
    """Custom optimization strategy."""
    
    def __init__(self, metric, max_iterations=10):
        self.metric = metric
        self.max_iterations = max_iterations
    
    def compile(self, student, trainset):
        """Custom compilation logic."""
        best_program = student
        best_score = 0
        
        for iteration in range(self.max_iterations):
            # Evaluate current program
            current_score = self.evaluate_program(student, trainset)
            
            print(f"Iteration {iteration}: Score = {current_score}")
            
            # If improved, save this version
            if current_score > best_score:
                best_score = current_score
                best_program = student
            
            # Generate new candidate (simplified)
            # In practice, this would involve more sophisticated logic
            student = self.generate_candidate(student, trainset)
        
        return best_program
    
    def evaluate_program(self, program, data):
        """Evaluate program on data."""
        scores = [self.metric(program(**item), item) for item in data]
        return sum(scores) / len(scores)
    
    def generate_candidate(self, program, trainset):
        """Generate a candidate program."""
        # This is a simplified version
        # Real implementation would modify prompts, examples, etc.
        return program

# Use custom optimizer
custom_optimizer = CustomOptimizer(metric=qa_metric, max_iterations=5)
optimized_qa = custom_optimizer.compile(student=base_qa, trainset=train_data)
```

## Advanced Optimization Techniques

### 1. Few-Shot Example Selection

```python
def select_best_examples(program, trainset, metric, n=5):
    """Select the best few-shot examples from training data."""
    # Evaluate each example
    example_scores = []
    for example in trainset:
        score = metric(program(**example), example)
        example_scores.append((score, example))
    
    # Sort by score and select top n
    example_scores.sort(key=lambda x: x[0], reverse=True)
    best_examples = [example for score, example in example_scores[:n]]
    
    return best_examples
```

### 2. Prompt Engineering Automation

```python
def optimize_prompt(signature, trainset, metric):
    """Automatically optimize prompt descriptions."""
    current_best = signature.__doc__
    best_score = 0
    
    # Generate prompt variations
    variations = generate_prompt_variations(current_best)
    
    for variation in variations:
        # Update signature with new prompt
        signature.__doc__ = variation
        
        # Create and evaluate program
        program = dspy.Predict(signature)
        score = evaluate_program(program, trainset, metric)
        
        # Keep best
        if score > best_score:
            best_score = score
            current_best = variation
    
    return current_best, best_score
```

### 3. Hyperparameter Tuning

```python
def tune_hyperparameters(program, trainset, valset, metric):
    """Tune hyperparameters like temperature, max_tokens, etc."""
    best_params = {}
    best_score = 0
    
    # Define search space
    temperature_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_tokens_range = [100, 200, 500, 1000]
    
    # Grid search
    for temp in temperature_range:
        for tokens in max_tokens_range:
            # Configure model with parameters
            lm = dspy.OpenAI(
                model="gpt-3.5-turbo",
                temperature=temp,
                max_tokens=tokens
            )
            dspy.settings.configure(lm=lm)
            
            # Evaluate
            score = evaluate_program(program, valset, metric)
            
            # Keep best
            if score > best_score:
                best_score = score
                best_params = {"temperature": temp, "max_tokens": tokens}
    
    return best_params, best_score
```

## Optimization Best Practices

### 1. Start Simple

```python
# Begin with basic optimization
basic_optimizer = BootstrapFewShot(metric=metric)
optimized = basic_optimizer.compile(student=program, trainset=data)
```

### 2. Use Quality Data

```python
# Ensure training data is high-quality and representative
quality_data = filter_and_validate_data(raw_data)
optimizer = BootstrapFewShot(metric=metric)
optimized = optimizer.compile(student=program, trainset=quality_data)
```

### 3. Validate on Separate Set

```python
# Train on training data, validate on validation data
optimized = optimizer.compile(student=program, trainset=train_data)
validation_score = evaluate_program(optimized, val_data, metric)
```

### 4. Monitor Overfitting

```python
# Compare train and validation scores
train_score = evaluate_program(optimized, train_data, metric)
val_score = evaluate_program(optimized, val_data, metric)

if train_score - val_score > 0.2:  # Large gap indicates overfitting
    print("Warning: Possible overfitting")
```

### 5. Iterate and Experiment

```python
# Try different optimizers and strategies
optimizers = [
    BootstrapFewShot(metric=metric),
    BootstrapFewShotWithRandomSearch(metric=metric, num_trials=5),
    KNNFewShot(metric=metric, k=4)
]

for optimizer in optimizers:
    optimized = optimizer.compile(student=program, trainset=data)
    score = evaluate_program(optimized, val_data, metric)
    print(f"{optimizer.__class__.__name__}: {score}")
```

## Common Optimization Issues

### Issue 1: Overfitting

**Symptoms**: Great training performance, poor validation performance

**Solution**:
- Use more diverse training data
- Reduce number of few-shot examples
- Add regularization

### Issue 2: Slow Optimization

**Symptoms**: Optimization takes too long

**Solution**:
- Reduce training data size
- Use faster LLM models for optimization
- Limit number of optimization iterations

### Issue 3: No Improvement

**Symptoms**: Optimized program performs similarly to base

**Solution**:
- Improve evaluation metric
- Check training data quality
- Try different optimizer
- Increase optimization iterations

## Summary

In this lab, you learned:

- **Optimization Concepts**: Understanding DSPy's self-improving nature
- **Teleprompters**: Different optimization tools and when to use them
- **Basic Optimization**: Implementing simple optimization workflows
- **Advanced Techniques**: Custom optimization strategies
- **Best Practices**: Guidelines for effective optimization

## Next Steps

Proceed to [Lab 6: Chain of Thought](lab-06-chain-of-thought.md) to learn about implementing complex reasoning patterns in DSPy.

## Challenge Project

Create an optimization framework that:
1. Supports multiple optimizer types
2. Automatically selects the best optimizer
3. Provides optimization reports
4. Tracks optimization history
5. Suggests improvements based on results

This will demonstrate your mastery of DSPy optimization!