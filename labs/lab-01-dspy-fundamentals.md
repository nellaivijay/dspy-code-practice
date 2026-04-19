# Lab 1: DSPy Fundamentals

## Overview

This lab introduces the core concepts of DSPy (Declarative Self-improving Language Programs). Understanding these fundamentals is essential for building effective AI systems with DSPy.

## Learning Objectives

- Understand the DSPy declarative programming paradigm
- Learn the difference between imperative and declarative LLM programming
- Master core DSPy concepts: Signatures, Modules, and Teleprompters
- Understand the self-improving nature of DSPy programs
- Build your first simple DSPy program

## Prerequisites

- Completed Lab 0: Environment Setup
- Working LLM provider configuration
- Basic Python knowledge

## The DSPy Paradigm

### Traditional LLM Programming vs DSPy

**Traditional (Imperative) Approach:**
```python
# You manually craft prompts
prompt = """
You are a helpful assistant. 
Answer the following question: {question}
"""
response = openai.ChatCompletion.create(
    messages=[{"role": "user", "content": prompt.format(question="What is AI?")}]
)
```

**DSPy (Declarative) Approach:**
```python
import dspy

# You declare what you want
class QuestionAnswering(dspy.Signature):
    """Answer questions with short, factual responses."""
    question = dspy.InputField(desc="the question to answer")
    answer = dspy.OutputField(desc="the answer to the question")

# DSPy handles the implementation
qa = dspy.ChainOfThought(QuestionAnswering)
result = qa(question="What is AI?")
```

### Key Benefits of DSPy

1. **Declarative**: Focus on WHAT, not HOW
2. **Self-Improving**: Programs optimize themselves
3. **Modular**: Reusable components
4. **Portable**: Works across different LLMs
5. **Testable**: Easy to evaluate and improve

## Core Concepts

### 1. Signatures

Signatures define the input/output interface of your DSPy program:

```python
import dspy

class TextSummary(dspy.Signature):
    """Summarize the given text concisely."""
    text = dspy.InputField(desc="text to summarize")
    summary = dspy.OutputField(desc="concise summary")
```

**Key Elements:**
- **Docstring**: Describes what the signature does
- **InputField**: Defines expected inputs
- **OutputField**: Defines expected outputs

### 2. Modules

Modules are the building blocks of DSPy programs:

```python
# Basic module usage
summarizer = dspy.ChainOfThought(TextSummary)
result = summarizer(text="Long text here...")
```

**Common Module Types:**
- `dspy.Predict`: Basic prediction
- `dspy.ChainOfThought`: Reasoning before output
- `dspy.ReAct`: Multi-step reasoning
- `dspy.ProgramOfThought`: Complex reasoning chains

### 3. Teleprompters (Optimizers)

Teleprompters optimize your DSPy programs:

```python
from dspy.teleprompt import BootstrapFewShot

# Create optimizer
optimizer = BootstrapFewShot(metric=answer_match)

# Optimize your program
optimized_qa = optimizer.compile(student=qa, trainset=training_data)
```

## Exercise 1: Your First Signature

Create a signature for sentiment analysis:

```python
import dspy

class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of the given text."""
    text = dspy.InputField(desc="text to analyze")
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")
    confidence = dspy.OutputField(desc="confidence score from 0 to 1")
```

## Exercise 2: Build a Simple Program

Create a complete DSPy program:

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define signature
class QuestionAnswering(dspy.Signature):
    """Answer questions accurately and concisely."""
    question = dspy.InputField(desc="the question to answer")
    answer = dspy.OutputField(desc="the answer")

# Create program
qa = dspy.Predict(QuestionAnswering)

# Test it
result = qa(question="What is the capital of France?")
print(f"Answer: {result.answer}")
```

## Exercise 3: Chain of Thought

Use ChainOfThought for complex tasks:

```python
class ComplexQA(dspy.Signature):
    """Answer complex questions with step-by-step reasoning."""
    question = dspy.InputField(desc="complex question requiring reasoning")
    reasoning = dspy.OutputField(desc="step-by-step reasoning")
    answer = dspy.OutputField(desc="final answer")

# Create ChainOfThought program
complex_qa = dspy.ChainOfThought(ComplexQA)

# Test with reasoning
result = complex_qa(question="If a train travels at 60 mph for 2.5 hours, how far does it travel?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
```

## The Self-Improving Nature

DSPy programs improve themselves through optimization:

1. **Define Task**: Specify what you want (signature)
2. **Provide Examples**: Give training data
3. **Optimize**: Let DSPy find the best implementation
4. **Deploy**: Use the optimized program

```python
# Example of self-improvement
from dspy.teleprompt import BootstrapFewShot

# Training data
trainset = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 5*3?", "answer": "15"},
    # ... more examples
]

# Create optimizer
optimizer = BootstrapFewShot(metric=lambda x, y: x.answer == y.answer)

# Optimize the program
optimized_qa = optimizer.compile(student=qa, trainset=trainset)

# Use optimized version
result = optimized_qa(question="What is 7*8?")
```

## Comparing Approaches

### Manual Prompt Engineering
```python
# You must manually craft and tune prompts
prompt = f"""
You are a math expert. Solve this step by step:
{question}

Show your work and give the final answer.
"""
```

### DSPy Declarative Approach
```python
# Declare what you want, DSPy handles the rest
class MathSolver(dspy.Signature):
    """Solve math problems step by step."""
    problem = dspy.InputField(desc="math problem to solve")
    steps = dspy.OutputField(desc="solution steps")
    answer = dspy.OutputField(desc="final answer")
```

## Best Practices

1. **Start Simple**: Begin with basic signatures
2. **Iterate**: Gradually add complexity
3. **Test**: Always test with examples
4. **Optimize**: Use teleprompters for improvement
5. **Document**: Add clear docstrings

## Common Patterns

### Pattern 1: Simple Prediction
```python
class SimpleTask(dspy.Signature):
    """Simple input-output task."""
    input_text = dspy.InputField()
    output_text = dspy.OutputField()

program = dspy.Predict(SimpleTask)
```

### Pattern 2: Chain of Thought
```python
class ReasoningTask(dspy.Signature):
    """Task requiring reasoning."""
    question = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(ReasoningTask)
```

### Pattern 3: Multi-Field Output
```python
class MultiOutput(dspy.Signature):
    """Task with multiple outputs."""
    text = dspy.InputField()
    summary = dspy.OutputField()
    keywords = dspy.OutputField()
    sentiment = dspy.OutputField()

program = dspy.Predict(MultiOutput)
```

## Summary

In this lab, you learned:

- **DSPy Paradigm**: Declarative vs imperative LLM programming
- **Signatures**: Define input/output interfaces
- **Modules**: Building blocks for programs
- **Teleprompters**: Optimizers for self-improvement
- **Self-Improving Nature**: Programs that optimize themselves

## Next Steps

Proceed to [Lab 2: Your First DSPy Program](lab-02-first-program.md) to build your first complete DSPy application.

## Additional Resources

- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)
- [Signature Design Guide](https://dspy-docs.vercel.app/)