# Lab 10: Real-World Applications

## Overview

This lab explores practical, real-world applications of DSPy. You'll learn how to apply DSPy concepts to solve actual business and technical problems.

## Learning Objectives

- Understand real-world DSPy use cases
- Implement practical applications
- Handle production considerations
- Optimize for specific domains
- Scale DSPy applications

## Prerequisites

- Completed Lab 9: Building AI Agents
- Understanding of business requirements
- Experience with system design

## Real-World Use Cases

### 1. Customer Support Automation

Build an intelligent customer support system:

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

class SupportTicketClassifier(dspy.Signature):
    """Classify customer support tickets."""
    ticket_text = dspy.InputField(desc="customer support ticket")
    category = dspy.OutputField(desc="category: technical, billing, general")
    urgency = dspy.OutputField(desc="urgency: low, medium, high")
    sentiment = dspy.OutputField(desc="customer sentiment")

class TicketResponseGenerator(dspy.Signature):
    """Generate appropriate responses for tickets."""
    ticket_text = dspy.InputField(desc="original ticket")
    category = dspy.InputField(desc="ticket category")
    knowledge_base = dspy.InputField(desc="relevant knowledge base articles")
    response = dspy.OutputField(desc="helpful response")
    suggested_actions = dspy.OutputField(desc="actions for support team")

class CustomerSupportSystem:
    def __init__(self):
        self.classifier = dspy.ChainOfThought(SupportTicketClassifier)
        self.response_generator = dspy.ChainOfThought(TicketResponseGenerator)
        
        # Knowledge base (simplified)
        self.knowledge_base = {
            "technical": [
                "For login issues, try resetting your password",
                "Clear browser cache for display problems",
                "Check internet connection for connectivity issues"
            ],
            "billing": [
                "Billing inquiries are processed within 24-48 hours",
                "Refunds take 5-7 business days to process",
                "Update payment info in account settings"
            ],
            "general": [
                "Our support team is available 24/7",
                "Feature requests can be submitted in our feedback portal",
                "Tutorial videos are available in our help center"
            ]
        }
    
    def process_ticket(self, ticket_text):
        # Classify ticket
        classification = self.classifier(ticket_text=ticket_text)
        
        # Get relevant knowledge
        relevant_kb = "\n".join(
            self.knowledge_base.get(classification.category.lower(), [])
        )
        
        # Generate response
        response = self.response_generator(
            ticket_text=ticket_text,
            category=classification.category,
            knowledge_base=relevant_kb
        )
        
        return {
            "category": classification.category,
            "urgency": classification.urgency,
            "sentiment": classification.sentiment,
            "response": response.response,
            "suggested_actions": response.suggested_actions
        }

# Test
support_system = CustomerSupportSystem()

ticket1 = "I can't log into my account. I've tried resetting my password but it's not working."
result1 = support_system.process_ticket(ticket1)
print(f"Category: {result1['category']}")
print(f"Urgency: {result1['urgency']}")
print(f"Response: {result1['response']}")

ticket2 = "I was charged twice for my subscription this month. Please refund the extra charge."
result2 = support_system.process_ticket(ticket2)
print(f"\nCategory: {result2['category']}")
print(f"Urgency: {result2['urgency']}")
print(f"Response: {result2['response']}")
```

### 2. Document Analysis System

Build a system for analyzing business documents:

```python
class DocumentAnalyzer(dspy.Signature):
    """Analyze business documents."""
    document_text = dspy.InputField(desc="document to analyze")
    doc_type = dspy.OutputField(desc="type: contract, report, invoice, email")
    key_points = dspy.OutputField(desc="main points in the document")
    risks = dspy.OutputField(desc="potential risks or issues")
    action_items = dspy.OutputField(desc="required actions")

class DocumentSummarizer(dspy.Signature):
    """Summarize document with specific focus."""
    document = dspy.InputField(desc="document text")
    summary_type = dspy.InputField(desc="executive, technical, financial")
    summary = dspy.OutputField(desc="focused summary")

class BusinessDocumentSystem:
    def __init__(self):
        self.analyzer = dspy.ChainOfThought(DocumentAnalyzer)
        self.summarizer = dspy.Predict(DocumentSummarizer)
    
    def analyze_document(self, document_text, summary_type="executive"):
        # Analyze document
        analysis = self.analyzer(document_text=document_text)
        
        # Generate focused summary
        summary = self.summarizer(
            document=document_text,
            summary_type=summary_type
        )
        
        return {
            "document_type": analysis.doc_type,
            "key_points": analysis.key_points,
            "risks": analysis.risks,
            "action_items": analysis.action_items,
            "summary": summary.summary
        }

# Test
doc_system = BusinessDocumentSystem()

contract = """
SERVICE AGREEMENT
This agreement between Company A and Company B outlines the terms of service delivery.
Company A will provide software development services for a period of 12 months.
Payment terms: Net 30 days from invoice.
Liability limited to contract value.
Termination requires 30-day notice.
"""

result = doc_system.analyze_document(contract, "executive")
print(f"Document Type: {result['document_type']}")
print(f"Key Points: {result['key_points']}")
print(f"Risks: {result['risks']}")
print(f"Summary: {result['summary']}")
```

### 3. Content Recommendation System

Build a personalized content recommendation system:

```python
class UserProfileAnalyzer(dspy.Signature):
    """Analyze user preferences and behavior."""
    user_history = dspy.InputField(desc="user's content consumption history")
    preferences = dspy.OutputField(desc="user preferences and interests")
    engagement_pattern = dspy.OutputField(desc("engagement patterns"))

class ContentRecommender(dspy.Signature):
    """Recommend content based on user profile."""
    user_profile = dspy.InputField(desc="user preferences and interests")
    available_content = dspy.InputField(desc="available content items")
    recommendations = dspy.OutputField(desc="top 5 recommended items")
    reasoning = dspy.OutputField(desc="why these items were recommended")

class RecommendationSystem:
    def __init__(self):
        self.profile_analyzer = dspy.ChainOfThought(UserProfileAnalyzer)
        self.recommender = dspy.ChainOfThought(ContentRecommender)
        
        # Content catalog
        self.content_catalog = [
            {"id": 1, "title": "Introduction to Machine Learning", "category": "tech", "difficulty": "beginner"},
            {"id": 2, "title": "Advanced Neural Networks", "category": "tech", "difficulty": "advanced"},
            {"id": 3, "title": "Business Strategy 101", "category": "business", "difficulty": "beginner"},
            {"id": 4, "title": "Financial Markets Analysis", "category": "finance", "difficulty": "intermediate"},
            {"id": 5, "title": "Python for Data Science", "category": "tech", "difficulty": "intermediate"},
        ]
    
    def get_recommendations(self, user_history):
        # Analyze user profile
        profile = self.profile_analyzer(user_history=user_history)
        
        # Format available content
        content_text = "\n".join([
            f"{item['id']}: {item['title']} ({item['category']}, {item['difficulty']})"
            for item in self.content_catalog
        ])
        
        # Get recommendations
        recommendations = self.recommender(
            user_profile=profile.preferences,
            available_content=content_text
        )
        
        return {
            "user_preferences": profile.preferences,
            "engagement_pattern": profile.engagement_pattern,
            "recommendations": recommendations.recommendations,
            "reasoning": recommendations.reasoning
        }

# Test
rec_system = RecommendationSystem()

user_history = """
User has completed: "Introduction to Python", "Data Analysis Basics"
User frequently engages with: technology content, beginner-level material
User spends: 2-3 hours per week on learning
"""

result = rec_system.get_recommendations(user_history)
print(f"User Preferences: {result['user_preferences']}")
print(f"Recommendations: {result['recommendations']}")
print(f"Reasoning: {result['reasoning']}")
```

## Exercise 1: Automated Code Review

Build a system for automated code review:

```python
class CodeAnalyzer(dspy.Signature):
    """Analyze code for quality and issues."""
    code = dspy.InputField(desc="code to analyze")
    language = dspy.InputField(desc="programming language")
    issues = dspy.OutputField(desc="potential issues found")
    suggestions = dspy.OutputField(desc="improvement suggestions")
    security_concerns = dspy.OutputField(desc="potential security issues")

class CodeReviewGenerator(dspy.Signature):
    """Generate comprehensive code review."""
    code = dspy.InputField(desc="code to review")
    analysis = dspy.InputField(desc="code analysis results")
    review_comments = dspy.OutputField(desc="detailed review comments")
    overall_score = dspy.OutputField(desc="code quality score (1-10)")

class CodeReviewSystem:
    def __init__(self):
        self.analyzer = dspy.ChainOfThought(CodeAnalyzer)
        self.review_generator = dspy.ChainOfThought(CodeReviewGenerator)
    
    def review_code(self, code, language="Python"):
        # Analyze code
        analysis = self.analyzer(code=code, language=language)
        
        # Generate review
        review = self.review_generator(
            code=code,
            analysis=f"Issues: {analysis.issues}\nSuggestions: {analysis.suggestions}\nSecurity: {analysis.security_concerns}"
        )
        
        return {
            "issues": analysis.issues,
            "suggestions": analysis.suggestions,
            "security_concerns": analysis.security_concerns,
            "review_comments": review.review_comments,
            "overall_score": review.overall_score
        }

# Test
code_review_system = CodeReviewSystem()

sample_code = """
def process_data(data):
    result = []
    for item in data:
        if item['value'] > 100:
            result.append(item)
    return result
"""

review = code_review_system.review_code(sample_code, "Python")
print(f"Issues: {review['issues']}")
print(f"Suggestions: {review['suggestions']}")
print(f"Security Concerns: {review['security_concerns']}")
print(f"Overall Score: {review['overall_score']}")
```

## Exercise 2: Market Research Assistant

Build a market research analysis system:

```python
class MarketTrendAnalyzer(dspy.Signature):
    """Analyze market trends from data."""
    market_data = dspy.InputField(desc="market data and statistics")
    trends = dspy.OutputField(desc="identified market trends")
    opportunities = dspy.OutputField(desc="market opportunities")
    threats = dspy.OutputField(desc="potential threats")

class CompetitorAnalyzer(dspy.Signature):
    """Analyze competitive landscape."""
    competitor_info = dspy.InputField(desc="information about competitors")
    strengths = dspy.OutputField(desc="competitor strengths")
    weaknesses = dspy.OutputField(desc="competitor weaknesses")
    market_position = dspy.OutputField(desc("competitor market position"))

class MarketResearchSystem:
    def __init__(self):
        self.trend_analyzer = dspy.ChainOfThought(MarketTrendAnalyzer)
        self.competitor_analyzer = dspy.ChainOfThought(CompetitorAnalyzer)
    
    def analyze_market(self, market_data, competitor_info):
        # Analyze trends
        trend_analysis = self.trend_analyzer(market_data=market_data)
        
        # Analyze competitors
        competitor_analysis = self.competitor_analyzer(competitor_info=competitor_info)
        
        return {
            "trends": trend_analysis.trends,
            "opportunities": trend_analysis.opportunities,
            "threats": trend_analysis.threats,
            "competitor_analysis": competitor_analysis
        }

# Test
market_system = MarketResearchSystem()

market_data = """
Market size: $10B growing at 15% annually
Key segments: Enterprise (60%), SMB (30%), Consumer (10%)
Technology trends: AI adoption, cloud migration, automation
"""

competitor_info = """
Competitor A: Market leader, 40% share, strong enterprise presence
Competitor B: Growing fast, 20% share, innovative technology
Competitor C: Niche player, 10% share, specialized solutions
"""

analysis = market_system.analyze_market(market_data, competitor_info)
print(f"Trends: {analysis['trends']}")
print(f"Opportunities: {analysis['opportunities']}")
print(f"Threats: {analysis['threats']}")
```

## Production Considerations

### 1. Performance Optimization

```python
class CachedDSPySystem:
    def __init__(self, base_system):
        self.base_system = base_system
        self.cache = {}
    
    def process(self, input_data):
        cache_key = str(input_data)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.base_system.process(input_data)
        self.cache[cache_key] = result
        
        return result

class BatchProcessor:
    def __init__(self, system, batch_size=10):
        self.system = system
        self.batch_size = batch_size
    
    def process_batch(self, inputs):
        results = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i+self.batch_size]
            batch_results = [self.system.process(item) for item in batch]
            results.extend(batch_results)
        
        return results
```

### 2. Error Handling

```python
class ResilientDSPySystem:
    def __init__(self, base_system, max_retries=3):
        self.base_system = base_system
        self.max_retries = max_retries
    
    def process_with_retry(self, input_data):
        for attempt in range(self.max_retries):
            try:
                return self.base_system.process(input_data)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"error": str(e), "input": input_data}
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def process_with_fallback(self, input_data, fallback_response):
        try:
            return self.base_system.process(input_data)
        except Exception as e:
            return {
                "fallback": True,
                "response": fallback_response,
                "error": str(e)
            }
```

### 3. Monitoring and Logging

```python
class MonitoredDSPySystem:
    def __init__(self, base_system):
        self.base_system = base_system
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0
        }
    
    def process(self, input_data):
        import time
        
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        try:
            result = self.base_system.process(input_data)
            self.metrics["successful_requests"] += 1
            latency = time.time() - start_time
            
            # Update average latency
            self.metrics["avg_latency"] = (
                self.metrics["avg_latency"] * (self.metrics["successful_requests"] - 1) + latency
            ) / self.metrics["successful_requests"]
            
            return result
        except Exception as e:
            self.metrics["failed_requests"] += 1
            raise
    
    def get_metrics(self):
        return self.metrics.copy()
```

## Scaling Strategies

### 1. Horizontal Scaling

```python
class LoadBalancedDSPySystem:
    def __init__(self, systems):
        self.systems = systems
        self.current_index = 0
    
    def process(self, input_data):
        # Simple round-robin load balancing
        system = self.systems[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.systems)
        
        return system.process(input_data)
```

### 2. Queue-Based Processing

```python
import queue
import threading

class QueuedDSPySystem:
    def __init__(self, system, num_workers=4):
        self.system = system
        self.queue = queue.Queue()
        self.workers = []
        
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        while True:
            input_data, callback = self.queue.get()
            try:
                result = self.system.process(input_data)
                callback(result, None)
            except Exception as e:
                callback(None, e)
            finally:
                self.queue.task_done()
    
    def process_async(self, input_data, callback):
        self.queue.put((input_data, callback))
```

## Best Practices

1. **Start Simple**: Begin with basic implementations, add complexity gradually
2. **Monitor Everything**: Track performance, errors, and user satisfaction
3. **Handle Errors Gracefully**: Implement robust error handling and fallbacks
4. **Optimize Iteratively**: Profile and optimize based on real usage data
5. **Security First**: Validate inputs, sanitize outputs, implement rate limiting

## Summary

In this lab, you learned:

- **Real-World Applications**: Practical DSPy use cases
- **Customer Support**: Building intelligent support systems
- **Document Analysis**: Processing business documents
- **Content Recommendation**: Personalized content systems
- **Production Considerations**: Performance, error handling, monitoring
- **Scaling Strategies**: Handling increased load and demand

## Next Steps

Proceed to [Lab 11: Production Deployment](lab-11-production-deployment.md) to learn about deploying DSPy applications to production.

## Challenge Project

Build a comprehensive real-world application that:
1. Solves a practical business problem
2. Includes proper error handling and monitoring
3. Implements caching and performance optimization
4. Provides clear metrics and logging
5. Can scale to handle production loads

Choose a domain that interests you and apply all the DSPy concepts you've learned!