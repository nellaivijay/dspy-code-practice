# Lab 9: Building AI Agents

## Overview

This lab explores building autonomous AI agents using DSPy. AI agents can perceive, reason, act, and learn to accomplish complex goals autonomously.

## Learning Objectives

- Understand AI agent architecture
- Implement basic agent behaviors
- Build agents with tools and capabilities
- Design multi-agent systems
- Handle agent memory and state

## Prerequisites

- Completed Lab 8: Retrieval Augmented Generation
- Understanding of agent concepts
- Experience with complex DSPy programs

## What are AI Agents?

### Agent Architecture

AI agents consist of:
1. **Perception**: Understanding the environment and inputs
2. **Reasoning**: Deciding on actions based on goals
3. **Action**: Executing actions in the environment
4. **Memory**: Storing and retrieving information
5. **Learning**: Improving from experience

### Why Agents Matter

- **Autonomy**: Can operate without constant human intervention
- **Adaptability**: Can adjust to changing conditions
- **Complexity**: Can handle multi-step, complex tasks
- **Scalability**: Can coordinate multiple agents
- **Specialization**: Can be designed for specific domains

## Basic Agent Implementation

### Simple Task Agent

```python
import dspy

# Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

class TaskAgent(dspy.Signature):
    """An agent that can complete tasks."""
    task_description = dspy.InputField(desc="description of the task")
    available_tools = dspy.InputField(desc="list of available tools")
    plan = dspy.OutputField(desc="step-by-step plan to complete the task")
    tool_selection = dspy.OutputField(desc="which tools to use and when")
    execution = dspy.OutputField(desc="how to execute the plan")

# Create agent
task_agent = dspy.ChainOfThought(TaskAgent)

# Test
task = "Research the current state of AI in healthcare and write a summary"
tools = "web_search, document_reader, text_summarizer"

result = task_agent(task_description=task, available_tools=tools)
print(f"Plan: {result.plan}")
print(f"Tool Selection: {result.tool_selection}")
print(f"Execution: {result.execution}")
```

## Exercise 1: Agent with Tool Use

Build an agent that can use external tools:

```python
class ToolUsingAgent(dspy.Signature):
    """Agent that can use tools to accomplish tasks."""
    task = dspy.InputField(desc="task to accomplish")
    tool_descriptions = dspy.InputField(desc="descriptions of available tools")
    reasoning = dspy.OutputField(desc="reasoning about which tools to use")
    tool_calls = dspy.OutputField(desc="specific tool calls to make")
    expected_results = dspy.OutputField(desc="what results to expect from tools")

# Simulated tools
class ToolSet:
    def __init__(self):
        self.tools = {
            "calculator": self.calculate,
            "search": self.search,
            "text_analyzer": self.analyze_text
        }
    
    def calculate(self, expression):
        """Simple calculator."""
        try:
            return str(eval(expression))
        except:
            return "Error in calculation"
    
    def search(self, query):
        """Simulated search."""
        return f"Search results for: {query}"
    
    def analyze_text(self, text):
        """Simple text analysis."""
        return {
            "length": len(text),
            "words": len(text.split()),
            "uppercase": sum(1 for c in text if c.isupper())
        }

# Agent with tool execution
class AgentWithTools:
    def __init__(self):
        self.agent = dspy.ChainOfThought(ToolUsingAgent)
        self.tools = ToolSet()
    
    def execute_task(self, task):
        # Get tool descriptions
        tool_descriptions = """
        calculator: perform mathematical calculations
        search: search for information
        text_analyzer: analyze text properties
        """
        
        # Plan tool usage
        plan = self.agent(
            task=task,
            tool_descriptions=tool_descriptions
        )
        
        # Execute tool calls (simplified parsing)
        tool_calls = self._parse_tool_calls(plan.tool_calls)
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            params = tool_call.get("params", {})
            
            if tool_name in self.tools.tools:
                result = self.tools.tools[tool_name](**params)
                results.append(result)
        
        return {
            "plan": plan.reasoning,
            "tool_calls": tool_calls,
            "results": results
        }
    
    def _parse_tool_calls(self, tool_calls_text):
        """Parse tool calls from agent output."""
        # Simplified parsing - in production, use more robust parsing
        return [{"tool": "calculator", "params": {"expression": "2+2"}}]

# Test
agent = AgentWithTools()
result = agent.execute_task("Calculate 15 * 23 and tell me the result")
print(f"Plan: {result['plan']}")
print(f"Results: {result['results']}")
```

## Exercise 2: Conversational Agent

Build an agent that can maintain conversation context:

```python
class ConversationalAgent(dspy.Signature):
    """Agent that can maintain conversation context."""
    conversation_history = dspy.InputField(desc="previous conversation turns")
    current_input = dspy.InputField(desc="current user input")
    reasoning = dspy.OutputField(desc="reasoning about response")
    response = dspy.OutputField(desc="response to user")
    context_update = dspy.OutputField(desc="what to remember from this turn")

class ConversationManager:
    def __init__(self):
        self.agent = dspy.ChainOfThought(ConversationalAgent)
        self.conversation_history = []
        self.memory = {}
    
    def converse(self, user_input):
        # Format conversation history
        history_text = "\n".join([
            f"User: {turn['user']}\nAgent: {turn['agent']}"
            for turn in self.conversation_history[-5:]  # Last 5 turns
        ])
        
        # Get agent response
        result = self.agent(
            conversation_history=history_text,
            current_input=user_input
        )
        
        # Update memory
        if result.context_update:
            self._update_memory(result.context_update)
        
        # Add to history
        self.conversation_history.append({
            "user": user_input,
            "agent": result.response
        })
        
        return result.response
    
    def _update_memory(self, context_update):
        """Update agent memory."""
        # Simple memory update
        if "name" in context_update.lower():
            # Extract name (simplified)
            pass
    
    def get_summary(self):
        """Get conversation summary."""
        return {
            "turns": len(self.conversation_history),
            "memory": self.memory,
            "recent_history": self.conversation_history[-3:]
        }

# Test
conv_agent = ConversationManager()

responses = [
    conv_agent.converse("Hi, I'm John"),
    conv_agent.converse("I'm interested in learning about AI"),
    conv_agent.converse("Can you recommend some resources?"),
    conv_agent.converse("What did I say my name was?")
]

for i, response in enumerate(responses, 1):
    print(f"Turn {i}: {response}")

print(f"\nSummary: {conv_agent.get_summary()}")
```

## Exercise 3: Goal-Oriented Agent

Build an agent that works toward specific goals:

```python
class GoalOrientedAgent(dspy.Signature):
    """Agent that works toward specific goals."""
    current_state = dspy.InputField(desc="current state of the world")
    goal = dspy.InputField(desc="goal to achieve")
    available_actions = dspy.InputField(desc="actions the agent can take")
    reasoning = dspy.OutputField(desc="reasoning about next action")
    next_action = dspy.OutputField(desc="next action to take")
    expected_state = dspy.OutputField(desc="expected state after action")

class GoalAgent:
    def __init__(self):
        self.planner = dspy.ChainOfThought(GoalOrientedAgent)
        self.state = {}
        self.goal = None
        self.actions_taken = []
    
    def set_goal(self, goal):
        """Set the agent's goal."""
        self.goal = goal
        self.actions_taken = []
    
    def get_state(self):
        """Get current state."""
        return self.state
    
    def step(self):
        """Take one step toward the goal."""
        if not self.goal:
            return "No goal set"
        
        # Check if goal is achieved
        if self._check_goal_achieved():
            return "Goal achieved!"
        
        # Plan next action
        available_actions = self._get_available_actions()
        
        plan = self.planner(
            current_state=str(self.state),
            goal=self.goal,
            available_actions=str(available_actions)
        )
        
        # Execute action
        result = self._execute_action(plan.next_action)
        self.actions_taken.append(plan.next_action)
        
        return {
            "action": plan.next_action,
            "result": result,
            "reasoning": plan.reasoning
        }
    
    def _check_goal_achieved(self):
        """Check if the current goal is achieved."""
        # Simplified check
        if "file_created" in self.goal.lower():
            return self.state.get("file_created", False)
        return False
    
    def _get_available_actions(self):
        """Get available actions based on current state."""
        return [
            "create_file",
            "read_file",
            "write_to_file",
            "search_web",
            "analyze_data"
        ]
    
    def _execute_action(self, action):
        """Execute an action (simulated)."""
        # Simulated action execution
        if "create_file" in action.lower():
            self.state["file_created"] = True
            return "File created successfully"
        elif "read_file" in action.lower():
            return "File content read"
        return f"Executed: {action}"

# Test
goal_agent = GoalAgent()
goal_agent.set_goal("Create a file named 'hello.txt' with content 'Hello World'")

# Take steps toward goal
for i in range(5):
    result = goal_agent.step()
    print(f"Step {i+1}: {result}")
    
    if result == "Goal achieved!":
        break

print(f"\nFinal State: {goal_agent.get_state()}")
print(f"Actions Taken: {goal_agent.actions_taken}")
```

## Advanced Agent Patterns

### 1. Multi-Agent Collaboration

```python
class SpecialistAgent(dspy.Signature):
    """Specialist agent for specific domain."""
    task = dspy.InputField(desc="task requiring specialist knowledge")
    domain = dspy.InputField(desc="specialist domain")
    analysis = dspy.OutputField(desc="domain-specific analysis")
    recommendation = dspy.OutputField(desc="recommendation from specialist perspective")

class CoordinatorAgent(dspy.Signature):
    """Coordinates multiple specialist agents."""
    task = dspy.InputField(desc="complex task")
    specialist_analyses = dspy.InputField(desc="analyses from specialists")
    coordination = dspy.OutputField(desc="how to combine specialist inputs")
    final_decision = dspy.OutputField(desc="coordinated final decision")

class MultiAgentSystem:
    def __init__(self):
        self.specialists = {
            "technical": dspy.ChainOfThought(SpecialistAgent),
            "business": dspy.ChainOfThought(SpecialistAgent),
            "design": dspy.ChainOfThought(SpecialistAgent)
        }
        self.coordinator = dspy.ChainOfThought(CoordinatorAgent)
    
    def solve_task(self, task):
        # Get specialist analyses
        analyses = {}
        for domain, specialist in self.specialists.items():
            analysis = specialist(task=task, domain=domain)
            analyses[domain] = {
                "analysis": analysis.analysis,
                "recommendation": analysis.recommendation
            }
        
        # Coordinate
        coordination = self.coordinator(
            task=task,
            specialist_analyses=str(analyses)
        )
        
        return {
            "specialist_analyses": analyses,
            "coordination": coordination.coordination,
            "final_decision": coordination.final_decision
        }

# Test
multi_agent = MultiAgentSystem()
result = multi_agent.solve_task("Design a new mobile app for fitness tracking")
print(f"Specialist Analyses: {result['specialist_analyses']}")
print(f"Final Decision: {result['final_decision']}")
```

### 2. Hierarchical Agent System

```python
class SupervisorAgent(dspy.Signature):
    """Supervises and delegates to sub-agents."""
    goal = dspy.InputField(desc="high-level goal")
    sub_goals = dspy.OutputField(desc="break down into sub-goals")
    delegation = dspy.OutputField(desc="which sub-agent handles each sub-goal")

class SubAgent(dspy.Signature):
    """Sub-agent that handles specific sub-goals."""
    sub_goal = dspy.InputField(desc="specific sub-goal")
    context = dspy.InputField(desc="context from supervisor")
    execution_plan = dspy.OutputField(desc="how to achieve sub-goal")
    result = dspy.OutputField(desc="result of sub-goal execution")

class HierarchicalAgentSystem:
    def __init__(self):
        self.supervisor = dspy.ChainOfThought(SupervisorAgent)
        self.sub_agents = [dspy.ChainOfThought(SubAgent) for _ in range(3)]
    
    def execute_goal(self, goal):
        # Supervise and break down
        supervision = self.supervisor(goal=goal)
        
        # Execute sub-goals
        sub_results = []
        for i, sub_agent in enumerate(self.sub_agents):
            sub_goal = supervision.sub_goals.split('\n')[i] if i < len(supervision.sub_goals.split('\n')) else "No sub-goal"
            result = sub_agent(sub_goal=sub_goal, context=supervision.delegation)
            sub_results.append(result.result)
        
        return {
            "sub_goals": supervision.sub_goals,
            "delegation": supervision.delegation,
            "sub_results": sub_results
        }
```

### 3. Reflective Agent

```python
class ReflectiveAgent(dspy.Signature):
    """Agent that can reflect on its actions."""
    action_taken = dspy.InputField(desc="action that was taken")
    result = dspy.InputField(desc="result of the action")
    goal = dspy.InputField(desc="goal being pursued")
    reflection = dspy.OutputField(desc="reflection on action effectiveness")
    improvement = dspy.OutputField(desc="how to improve next time")

class AgentWithReflection:
    def __init__(self):
        self.actor = dspy.ChainOfThought(GoalOrientedAgent)
        self.reflector = dspy.ChainOfThought(ReflectiveAgent)
        self.action_history = []
    
    def act_and_reflect(self, state, goal):
        # Take action
        action_plan = self.actor(
            current_state=str(state),
            goal=goal,
            available_actions="action1, action2, action3"
        )
        
        # Execute (simulated)
        result = f"Executed: {action_plan.next_action}"
        
        # Reflect
        reflection = self.reflector(
            action_taken=action_plan.next_action,
            result=result,
            goal=goal
        )
        
        self.action_history.append({
            "action": action_plan.next_action,
            "result": result,
            "reflection": reflection.reflection,
            "improvement": reflection.improvement
        })
        
        return {
            "action": action_plan.next_action,
            "result": result,
            "reflection": reflection.reflection
        }
```

## Agent Memory Management

### Long-term Memory

```python
class AgentMemory:
    def __init__(self):
        self.episodic_memory = []  # Specific experiences
        self.semantic_memory = {}  # General knowledge
        self.procedural_memory = []  # Skills and procedures
    
    def store_experience(self, experience):
        """Store an experience in episodic memory."""
        self.episodic_memory.append({
            "timestamp": self._get_timestamp(),
            "experience": experience
        })
    
    def retrieve_relevant(self, query, k=3):
        """Retrieve relevant experiences."""
        # Simple retrieval (would use semantic search in production)
        relevant = []
        for memory in self.episodic_memory:
            if query.lower() in str(memory["experience"]).lower():
                relevant.append(memory)
                if len(relevant) >= k:
                    break
        return relevant
    
    def learn_skill(self, skill):
        """Learn a new skill."""
        self.procedural_memory.append(skill)
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
```

## Agent Best Practices

1. **Clear Goals**: Define specific, measurable goals for agents
2. **Tool Boundaries**: Clearly define what tools agents can use
3. **Safety Constraints**: Implement safety checks and limits
4. **Memory Management**: Design effective memory strategies
5. **Monitoring**: Track agent behavior and performance

## Summary

In this lab, you learned:

- **Agent Architecture**: Understanding AI agent components
- **Basic Agents**: Implementing simple task agents
- **Tool-Using Agents**: Building agents with external tools
- **Conversational Agents**: Maintaining conversation context
- **Goal-Oriented Agents**: Working toward specific objectives
- **Advanced Patterns**: Multi-agent systems, hierarchical agents, reflection

## Next Steps

Proceed to [Lab 10: Real-World Applications](lab-10-real-world-applications.md) to learn about practical DSPy applications.

## Challenge Project

Build a comprehensive agent system that:
1. Can maintain conversation context
2. Uses multiple tools effectively
3. Works toward complex goals
4. Reflects on and learns from actions
5. Collaborates with other specialist agents

This will demonstrate advanced agent implementation skills!