# Lab 6: Chain of Thought

## Overview

This lab explores Chain of Thought (CoT) reasoning in DSPy. Chain of Thought is a powerful technique that enables AI systems to solve complex problems through step-by-step reasoning.

## Learning Objectives

- Understand Chain of Thought reasoning principles
- Implement CoT in DSPy programs
- Design effective reasoning chains
- Handle complex multi-step problems
- Optimize CoT performance

## Prerequisites

- Completed Lab 5: Optimization Strategies
- Understanding of basic DSPy modules
- Experience with complex problem-solving

## What is Chain of Thought?

### The Concept

Chain of Thought encourages AI models to:
1. **Break down problems** into smaller steps
2. **Show reasoning** before providing answers
3. **Verify solutions** through logical steps
4. **Handle complexity** through systematic approach

### Why Chain of Thought Matters

- **Improved Accuracy**: Step-by-step reasoning reduces errors
- **Better Explainability**: You can see the reasoning process
- **Complex Problem Solving**: Handles multi-step problems
- **Error Detection**: Easier to spot reasoning mistakes

## Basic Chain of Thought in DSPy

### Simple CoT Implementation

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define CoT signature
class MathProblem(dspy.Signature):
    """Solve math problems with step-by-step reasoning."""
    problem = dspy.InputField(desc="math problem to solve")
    reasoning = dspy.OutputField(desc="step-by-step reasoning")
    answer = dspy.OutputField(desc="final numerical answer")

# Create CoT program
math_solver = dspy.ChainOfThought(MathProblem)

# Test
result = math_solver(problem="If a train travels at 60 mph for 2.5 hours, how far does it travel?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
```

### CoT vs Direct Prediction

```python
# Direct prediction (no reasoning)
class DirectMath(dspy.Signature):
    """Solve math problems directly."""
    problem = dspy.InputField(desc="math problem")
    answer = dspy.OutputField(desc="answer")

direct_solver = dspy.Predict(DirectMath)

# Compare approaches
complex_problem = "A store sells apples for $2 each and oranges for $3 each. If John buys 5 apples and 3 oranges, how much does he spend?"

print("Direct Approach:")
direct_result = direct_solver(problem=complex_problem)
print(f"Answer: {direct_result.answer}")

print("\nChain of Thought Approach:")
cot_result = math_solver(problem=complex_problem)
print(f"Reasoning: {cot_result.reasoning}")
print(f"Answer: {cot_result.answer}")
```

## Exercise 1: Multi-Step Reasoning

Build a CoT program for complex logical reasoning:

```python
class LogicalReasoning(dspy.Signature):
    """Solve logical reasoning problems step by step."""
    problem = dspy.InputField(desc="logical problem to solve")
    step1_analysis = dspy.OutputField(desc="first step: understand the problem")
    step2_breakdown = dspy.OutputField(desc="second step: break into components")
    step3_reasoning = dspy.OutputField(desc="third step: logical reasoning")
    step4_verification = dspy.OutputField(desc="fourth step: verify solution")
    final_answer = dspy.OutputField(desc="final answer")

logical_solver = dspy.ChainOfThought(LogicalReasoning)

# Test with logical puzzle
logic_problem = """
If all roses are flowers, and some flowers fade quickly, 
can we conclude that some roses fade quickly? Explain your reasoning.
"""

result = logical_solver(problem=logic_problem)
print(f"Step 1: {result.step1_analysis}")
print(f"Step 2: {result.step2_breakdown}")
print(f"Step 3: {result.step3_reasoning}")
print(f"Step 4: {result.step4_verification}")
print(f"Final Answer: {result.final_answer}")
```

## Exercise 2: CoT for Document Analysis

Use CoT for complex document understanding:

```python
class DocumentAnalyzer(dspy.Signature):
    """Analyze documents with detailed reasoning."""
    document = dspy.InputField(desc="document text to analyze")
    understanding = dspy.OutputField(desc="step 1: understand the main topic")
    key_points = dspy.OutputField(desc="step 2: extract key points")
    relationships = dspy.OutputField(desc="step 3: identify relationships")
    conclusions = dspy.OutputField(desc="step 4: draw conclusions")
    summary = dspy.OutputField(desc="final summary")

analyzer = dspy.ChainOfThought(DocumentAnalyzer)

document = """
Climate change refers to long-term shifts in global temperatures and weather patterns. 
While natural factors have influenced Earth's climate throughout history, 
scientific evidence shows that human activities have been the main driver of climate change since the 1800s. 
The burning of fossil fuels like coal, oil, and gas generates greenhouse gas emissions that act like a blanket 
wrapped around the Earth, trapping the sun's heat and raising temperatures.
"""

result = analyzer(document=document)
print(f"Understanding: {result.understanding}")
print(f"Key Points: {result.key_points}")
print(f"Relationships: {result.relationships}")
print(f"Conclusions: {result.conclusions}")
print(f"Summary: {result.summary}")
```

## Exercise 3: Custom CoT Patterns

Create custom CoT patterns for specific domains:

```python
class MedicalDiagnosis(dspy.Signature):
    """Medical diagnosis with clinical reasoning."""
    symptoms = dspy.InputField(desc="patient symptoms")
    history = dspy.InputField(desc="patient medical history")
    symptom_analysis = dspy.OutputField(desc="step 1: analyze symptoms")
    differential_diagnosis = dspy.OutputField(desc="step 2: consider possibilities")
    risk_factors = dspy.OutputField(desc="step 3: evaluate risk factors")
    likely_diagnosis = dspy.OutputField(desc="step 4: determine most likely cause")
    recommendations = dspy.OutputField(desc="step 5: provide recommendations")
    final_diagnosis = dspy.OutputField(desc="final diagnosis")

medical_cot = dspy.ChainOfThought(MedicalDiagnosis)

patient_case = {
    "symptoms": "Severe headache, sensitivity to light, nausea",
    "history": "History of migraines, recent stress at work"
}

result = medical_cot(**patient_case)
print(f"Symptom Analysis: {result.symptom_analysis}")
print(f"Differential Diagnosis: {result.differential_diagnosis}")
print(f"Risk Factors: {result.risk_factors}")
print(f"Likely Diagnosis: {result.likely_diagnosis}")
print(f"Recommendations: {result.recommendations}")
print(f"Final Diagnosis: {result.final_diagnosis}")
```

## Advanced CoT Techniques

### 1. Self-Consistency

Run CoT multiple times and select the most common answer:

```python
def self_consistent_solve(problem, solver, n=5):
    """Run CoT multiple times and select most common answer."""
    results = []
    
    for _ in range(n):
        result = solver(problem=problem)
        results.append(result.answer)
    
    # Find most common answer
    from collections import Counter
    most_common = Counter(results).most_common(1)[0][0]
    
    return most_common, results

# Use self-consistency
answer, all_answers = self_consistent_solve(complex_problem, math_solver)
print(f"Final Answer: {answer}")
print(f"All Answers: {all_answers}")
```

### 2. Tree of Thoughts

Explore multiple reasoning paths:

```python
class TreeOfThoughts(dspy.Signature):
    """Explore multiple reasoning paths."""
    problem = dspy.InputField(desc="complex problem")
    path1_reasoning = dspy.OutputField(desc="first reasoning path")
    path1_answer = dspy.OutputField(desc="answer from path 1")
    path2_reasoning = dspy.OutputField(desc="second reasoning path")
    path2_answer = dspy.OutputField(desc="answer from path 2")
    path3_reasoning = dspy.OutputField(desc="third reasoning path")
    path3_answer = dspy.OutputField(desc="answer from path 3")
    final_reasoning = dspy.OutputField(desc="compare and select best path")
    final_answer = dspy.OutputField(desc="final answer")

tot_solver = dspy.ChainOfThought(TreeOfThoughts)
```

### 3. Analogical Reasoning

Use analogies in reasoning:

```python
class AnalogicalReasoning(dspy.Signature):
    """Use analogies to solve problems."""
    problem = dspy.InputField(desc="problem to solve")
    similar_problem = dspy.OutputField(desc="step 1: find similar problem")
    analogy_solution = dspy.OutputField(desc="step 2: how similar problem was solved")
    application = dspy.OutputField(desc="step 3: apply analogy to current problem")
    verification = dspy.OutputField(desc="step 4: verify analogy works")
    final_answer = dspy.OutputField(desc="final answer")

analogical_solver = dspy.ChainOfThought(AnalogicalReasoning)
```

## Optimizing Chain of Thought

### 1. Reasoning Quality Metrics

```python
def evaluate_reasoning_quality(result, ground_truth):
    """Evaluate the quality of reasoning."""
    metrics = {}
    
    # Check if final answer is correct
    metrics["answer_correct"] = (result.final_answer == ground_truth["answer"])
    
    # Check if reasoning is coherent
    metrics["reasoning_coherent"] = len(result.reasoning) > 50  # Simple check
    
    # Check if reasoning steps are logical
    metrics["logical_steps"] = "step" in result.reasoning.lower() or "first" in result.reasoning.lower()
    
    # Overall quality score
    metrics["quality_score"] = sum([
        metrics["answer_correct"],
        metrics["reasoning_coherent"],
        metrics["logical_steps"]
    ]) / 3
    
    return metrics
```

### 2. Reasoning Length Optimization

```python
def optimize_reasoning_length(program, val_data, target_length=200):
    """Find optimal reasoning length."""
    best_length = target_length
    best_score = 0
    
    for length in [100, 200, 300, 400, 500]:
        # Configure to target this length
        # (This would require custom DSPy configuration)
        score = evaluate_with_length(program, val_data, length)
        
        if score > best_score:
            best_score = score
            best_length = length
    
    return best_length, best_score
```

### 3. Few-Shot CoT Examples

```python
# Create few-shot examples for CoT
cot_examples = [
    {
        "problem": "What is 15% of 200?",
        "reasoning": "To find 15% of 200, I multiply 200 by 0.15. 200 * 0.15 = 30.",
        "answer": "30"
    },
    {
        "problem": "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
        "reasoning": "First, calculate the discount: 20% of $25 = $5. Then subtract the discount from the original price: $25 - $5 = $20.",
        "answer": "$20"
    }
]

# Use these in optimization
optimizer = BootstrapFewShot(metric=your_metric)
optimized_cot = optimizer.compile(
    student=math_solver,
    trainset=cot_examples
)
```

## CoT Best Practices

### 1. Design Clear Steps

```python
class WellStructuredCoT(dspy.Signature):
    """Clear, well-structured reasoning."""
    problem = dspy.InputField(desc="problem")
    understand = dspy.OutputField(desc="Step 1: Understand the problem clearly")
    plan = dspy.OutputField(desc="Step 2: Plan the approach")
    execute = dspy.OutputField(desc="Step 3: Execute the plan step by step")
    verify = dspy.OutputField(desc="Step 4: Verify the answer")
    answer = dspy.OutputField(desc="Final answer")
```

### 2. Handle Uncertainty

```python
class CoTWithUncertainty(dspy.Signature):
    """CoT that acknowledges uncertainty."""
    problem = dspy.InputField(desc="problem")
    reasoning = dspy.OutputField(desc="reasoning process")
    confidence = dspy.OutputField(desc="confidence in answer (0-1)")
    answer = dspy.OutputField(desc="answer")
    caveats = dspy.OutputField(desc="any caveats or limitations")
```

### 3. Enable Verification

```python
class VerifiableCoT(dspy.Signature):
    """CoT with built-in verification."""
    problem = dspy.InputField(desc="problem")
    solution_path = dspy.OutputField(desc="step-by-step solution")
    verification = dspy.OutputField(desc="verify each step is correct")
    final_answer = dspy.OutputField(desc="final answer")
    confidence = dspy.OutputField(desc="confidence in verification")
```

## Common CoT Issues

### Issue 1: Circular Reasoning

**Symptoms**: Reasoning goes in circles without progress

**Solution**: Structure reasoning with clear forward progression

### Issue 2: Overly Verbose

**Symptoms**: Excessive reasoning that doesn't add value

**Solution**: Set clear length limits and focus on essential steps

### Issue 3: Hallucinated Steps

**Symptoms**: Reasoning includes incorrect or made-up information

**Solution**: Add verification steps and use reliable knowledge sources

## Summary

In this lab, you learned:

- **CoT Fundamentals**: Understanding chain of thought reasoning
- **Basic CoT**: Implementing simple CoT in DSPy
- **Complex Reasoning**: Multi-step reasoning patterns
- **Advanced Techniques**: Self-consistency, tree of thoughts, analogical reasoning
- **Optimization**: Improving CoT quality and performance
- **Best Practices**: Designing effective CoT programs

## Next Steps

Proceed to [Lab 7: Multi-Stage Programs](lab-07-multi-stage-programs.md) to learn about building sophisticated AI pipelines.

## Challenge Project

Build a comprehensive CoT system that:
1. Supports multiple reasoning patterns
2. Automatically selects the best pattern for each problem
3. Provides reasoning quality metrics
4. Handles uncertainty gracefully
5. Optimizes reasoning length based on task complexity

This will demonstrate advanced CoT implementation skills!