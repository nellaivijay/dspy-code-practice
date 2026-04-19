# Lab 0: Environment Setup

## Overview

This lab will guide you through setting up your DSPy development environment. A properly configured environment is essential for building and testing DSPy programs effectively.

## Learning Objectives

- Install DSPy and required dependencies
- Configure LLM provider credentials
- Set up Python development environment
- Validate installation with test programs
- Understand different LLM provider options

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Basic command line familiarity
- API key for at least one LLM provider (OpenAI, Anthropic, etc.)

## Step 1: Check Python Version

Verify you have Python 3.8+ installed:

```bash
python --version
# Should show Python 3.8 or higher
```

If Python is not installed or version is too old, install Python 3.8+ from [python.org](https://python.org).

## Step 2: Create Virtual Environment (Recommended)

Create a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv dspy_env

# Activate virtual environment
# On Linux/Mac:
source dspy_env/bin/activate
# On Windows:
dspy_env\Scripts\activate
```

## Step 3: Install Dependencies

Install DSPy and required packages:

```bash
pip install -r requirements.txt
```

This will install:
- `dspy-ai` - DSPy library
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `pandas`, `numpy` - Data processing
- `jupyter` - Jupyter notebooks
- `pytest` - Testing framework

## Step 4: Configure LLM Provider

DSPy needs access to an LLM provider. Set up your API credentials:

### Option 1: OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Option 2: Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Option 3: Using .env file (Recommended)

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Then load it in Python:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Step 5: Verify Installation

Test your DSPy installation:

```python
import dspy

# Configure DSPy with your LLM provider
import os

# For OpenAI
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Test basic functionality
print("DSPy version:", dspy.__version__)
print("Installation successful!")
```

## Step 6: Test LLM Connection

Verify your LLM provider connection works:

```python
import dspy

# Configure with your provider
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Simple test
response = lm("What is DSPy?")
print("LLM Response:", response)
```

## Step 7: Jupyter Notebook Setup

Set up Jupyter for interactive learning:

```bash
# Install Jupyter kernel
python -m ipykernel install --user --name=dspy_env --display-name "DSPy"

# Start Jupyter
jupyter notebook
```

## Understanding LLM Providers

DSPy supports multiple LLM providers. Choose based on your needs:

### OpenAI
- **Models**: GPT-3.5, GPT-4, GPT-4-turbo
- **Pros**: High quality, widely used, good documentation
- **Cons**: Cost, rate limits
- **Best for**: General purpose, production applications

### Anthropic
- **Models**: Claude 3, Claude 3.5
- **Pros**: Strong performance, good for complex tasks
- **Cons**: Newer ecosystem, different API
- **Best for**: Complex reasoning, analysis

### Local Models
- **Models**: LLaMA, Mistral, etc.
- **Pros**: Free, private, customizable
- **Cons**: Requires GPU, setup complexity
- **Best for**: Privacy, cost optimization, experimentation

## Common Issues and Solutions

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'dspy'`

**Solution**:
```bash
pip install dspy-ai
```

### API Key Errors

**Problem**: Authentication errors with LLM provider

**Solution**:
- Verify API key is correct
- Check environment variables are set
- Ensure you have credits/usage available

### Version Conflicts

**Problem**: Dependency version conflicts

**Solution**:
```bash
pip install --upgrade pip
pip install --upgrade dspy-ai
```

## Environment Variables Reference

Create a `.env.example` file for reference:

```bash
# LLM Provider Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Model Configuration
DSPY_MODEL=gpt-3.5-turbo
DSPY_TEMPERATURE=0.7
DSPY_MAX_TOKENS=1000
```

## Next Steps

After completing this lab:

1. ✅ Verify DSPy is installed correctly
2. ✅ Test LLM provider connection
3. ✅ Set up Jupyter notebooks
4. ✅ Understand provider options
5. ➡️ Proceed to [Lab 1: DSPy Fundamentals](lab-01-dspy-fundamentals.md)

## Additional Resources

- [DSPy Installation Guide](https://github.com/stanfordnlp/dspy#installation)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)