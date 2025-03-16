# Week 2: Design Patterns for AI Engineering

## Overview

This week, we'll explore three essential design patterns that are particularly useful when building applications with LLMs:

1. **Chain of Responsibility Pattern**: Passing requests along a chain of handlers
2. **Strategy Pattern**: Defining a family of algorithms and making them interchangeable
3. **Adapter Pattern**: Converting the interface of a class into another interface clients expect

Understanding these patterns will help you build more maintainable, flexible, and robust AI applications.

## Getting Started: Understanding Python Classes

Before diving into the design patterns, we recommend reviewing the `1-python-classes.md` document first. This guide provides a concise overview of how classes work in Python, which is essential for understanding the design patterns we'll be implementing.

**Note for beginners**: Don't worry if you don't fully understand all the concepts in the classes guide yet! The goal is to get familiar with the terminology and basic structure of classes. You'll gain a deeper understanding as you work through the exercises. The most important sections to focus on initially are:

- What Are Classes?
- Basic Class Structure
- Creating and Using Objects
- Inheritance (basic concept)
- Abstract Base Classes (ABC)

Understanding these concepts will help you work with design patterns effectively, but remember that learning is an iterative process. You'll become more comfortable with these concepts as you practice.

## Exercise 1: Chain of Responsibility Pattern

### Concept

The Chain of Responsibility pattern passes requests along a chain of handlers. Each handler decides either to process the request or to pass it to the next handler in the chain.

### AI Engineering Context

In LLM applications, we often need to process data through multiple stages (preprocessing, prompt construction, post-processing, etc.). The Chain of Responsibility pattern allows us to:

- Create modular processing steps
- Easily add, remove, or reorder steps
- Handle errors at appropriate levels

### Exercise

In `2-chain-of-responsibility-pattern.py`, we've implemented a simple text processing pipeline for an AI application. The pipeline consists of several handlers:

1. **TextNormalizationHandler**: Cleans and normalizes text
2. **PromptConstructionHandler**: Builds a prompt for the LLM
3. **LLMProcessingHandler**: Sends the prompt to an LLM

Your task:

- Review the implementation and understand how the chain is constructed and processed
- Try to add another step to the end of the chain

## Exercise 2: Strategy Pattern

### Concept

The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. It lets the algorithm vary independently from clients that use it.

### AI Engineering Context

When working with LLMs, we often need different strategies for:

- Different text processing approaches
- Different prompt techniques
- Different output parsing approaches

### Exercise

In `3-strategy-pattern.py`, we've implemented a system that can use different text processing strategies:

1. **SummarizationStrategy**: Extracts the first and last sentences
2. **KeywordExtractionStrategy**: Identifies and counts important keywords
3. **SentimentAnalysisStrategy**: Analyzes the sentiment of the text

Your task:

- Review the implementation and understand how strategies can be swapped at runtime

## Exercise 3: Adapter Pattern

### Concept

The Adapter pattern converts the interface of a class into another interface clients expect. It allows classes to work together that couldn't otherwise because of incompatible interfaces.

### AI Engineering Context

When building LLM applications, we often need to integrate with external APIs and services that have different interfaces.

### Exercise

In `4-adapter-pattern.py`, we've implemented a system that adapts the Open-Meteo weather API to a consistent interface that our application can use.

Your task:

- Review the implementation and understand how the adapter translates between interfaces

## Exercise 4: Structured Output with Instructor

### Concept

Structured output transforms free-form LLM responses into well-defined data structures that your application can reliably use for decision-making and routing.

### AI Engineering Context

When building AI applications, we often need to:

- Extract specific information from LLM outputs
- Make routing decisions based on content classification
- Ensure responses contain all required fields in expected formats
- Connect LLM outputs directly to other systems

The Instructor library provides a clean interface for extracting structured data from LLMs using Pydantic models.

### Exercise

In `5-structured-output.py`, we've implemented a customer inquiry classification system that:

1. Takes customer messages as input
2. Uses Instructor to extract structured data with the OpenAI API
3. Classifies the inquiry type, priority, and other attributes
4. Routes the inquiry to the appropriate handler based on the classification

Your tasks:

1. **Try with your own custom message**: Enter a customer inquiry and see how the system classifies and routes it
2. **Modify the model**: Add a 'suggested_response' field to the CustomerInquiry model and update the route_inquiry function to print this suggested response
3. **Create a new handler type**: Add a 'technical_support' handler that triggers when the inquiry is a question about a product with confidence > 0.8
4. **Error handling challenge**: Add error handling to catch and handle validation errors that might occur during structured output extraction

This exercise demonstrates how well-designed data models can control application flow and make intelligent routing decisions with minimal code.

## Resources

- [Refactoring Guru: Design Patterns](https://refactoring.guru/design-patterns)
- [Instructor Documentation](https://python.useinstructor.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)