# Python Classes: Essential Concepts

## What Are Classes?

Classes are blueprints for creating objects in Python. They allow you to bundle data (attributes) and functionality (methods) together.

## Basic Class Structure

```python
class MyClass:
    # Class variable (shared by all instances)
    class_variable = "I'm shared across all instances"
    
    # Constructor method
    def __init__(self, name):
        # Instance variable (unique to each instance)
        self.name = name
        self._private_attr = 0  # Convention for "private" attributes
    
    # Regular instance method
    def say_hello(self):
        return f"Hello, my name is {self.name}"
```

## Creating and Using Objects

```python
# Create an instance
obj = MyClass("Example")

# Access attributes
print(obj.name)  # "Example"
print(obj.class_variable)  # "I'm shared across all instances"

# Call methods
print(obj.say_hello())  # "Hello, my name is Example"
```

## Special Methods

Python classes can implement special methods (surrounded by double underscores) that enable specific behaviors:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representation
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    # Addition
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # Length/magnitude
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
```

## Inheritance

Inheritance allows a class to inherit attributes and methods from another class:

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass  # Base method to be overridden

class Dog(Animal):  # Dog inherits from Animal
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):  # Cat inherits from Animal
    def speak(self):
        return f"{self.name} says Meow!"
```

## Class Methods and Static Methods

```python
class MathUtils:
    # Class variable
    pi = 3.14159
    
    # Regular instance method (needs an instance)
    def calculate_area(self, radius):
        return self.pi * radius * radius
    
    # Class method (works with the class itself)
    @classmethod
    def from_diameter(cls, diameter):
        return cls(diameter / 2)
    
    # Static method (doesn't need class or instance)
    @staticmethod
    def is_positive(number):
        return number > 0
```

## Properties

Properties provide controlled access to attributes:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self._age = age  # "Private" attribute
    
    # Getter
    @property
    def age(self):
        return self._age
    
    # Setter
    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value
```

## Abstract Base Classes (ABC)

Abstract Base Classes define interfaces that derived classes must implement:

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
```

## Key Points About ABCs

1. **Purpose**: Define interfaces that subclasses must implement
2. **Import**: `from abc import ABC, abstractmethod`
3. **Usage**: Inherit from ABC and use @abstractmethod decorator
4. **Enforcement**: Cannot instantiate an ABC directly or a subclass that doesn't implement all abstract methods
5. **Benefits**: Ensures consistent interfaces across related classes

## When to Use Classes

- When you need to bundle data and behavior together
- When you want to create multiple similar objects
- When implementing design patterns
- When modeling real-world entities
- When creating reusable components

## Best Practices

1. Follow naming conventions (CamelCase for classes, snake_case for methods/attributes)
2. Keep classes focused on a single responsibility
3. Use docstrings to document your classes and methods
4. Use properties instead of direct attribute access when validation is needed
5. Use abstract base classes to define interfaces
6. Prefer composition over inheritance when appropriate
