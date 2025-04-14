# Week 1 — Foundations of AI Engineering

## Setting Up Your Local Development Environment

Before diving into AI engineering, we need to establish a solid development environment. This foundation will serve you throughout the program and your future AI projects.

### Python Installation

Ensure you have Python 3.10 or newer installed on your system. This provides the necessary features and compatibility with modern AI libraries.

**Installation Resources:**
- [Python Downloads Page](https://www.python.org/downloads/)
- [Official Installation Guide](https://docs.python.org/3/using/index.html)

### IDE Configuration

For this program, we strongly recommend using Cursor IDE—a VS Code clone specifically optimized for AI development workflows.

**Why Cursor?**

AI-assisted coding has rapidly evolved from a novelty to an essential professional skill. Cursor integrates this capability directly into your development environment, providing intelligent code completion, refactoring suggestions, and contextual help.

I cannot overstate the productivity impact of AI-assisted coding tools. They represent a fundamental shift in how we write code, not just a marginal improvement. In my experience, coding with Cursor enables at least 10x faster development compared to traditional methods. This isn't hyperbole—the combination of intelligent autocompletion, contextual understanding, and automatic documentation generation transforms the development process.

Engineers who adapt to these tools will maintain a substantial competitive advantage. Those who don't risk becoming significantly less productive compared to peers who leverage these capabilities. The productivity gap is widening rapidly, making this adaptation non-negotiable for professional developers.

**Setup Resources:**
- [Video: How I Set Up VS Code for AI Projects](https://youtu.be/mpk4Q5feWaw)
- [Video: How to Work With Cursor](https://youtu.be/CqkZ-ybl3lg)
- [Cursor Download Page](https://cursor.sh/)

### UV Package Manager

We'll be using UV, a modern Python package manager that significantly improves upon pip with faster installation times, better dependency resolution, and improved reproducibility.

UV installs packages up to 10-100x faster than pip, handles complex dependency graphs more reliably, and creates consistent environments across different systems—critical for production AI applications.

**Installation Resources:**
- [Comprehensive Guide: Getting Started with UV](https://daveebbelaar.com/blog/2024/03/20/getting-started-with-uv-the-ultra-fast-python-package-manager/)

### Ruff Linter and Formatter

We will also use Ruff, an extremely fast Python linter and formatter written in Rust. It combines the functionality of multiple Python tools (like flake8, black, isort, and more) into a single, high-performance package.

Key benefits of Ruff include:

- 10-100x faster than traditional Python linters
- Automatic code fixing capabilities
- Comprehensive rule set covering style, bugs, and complexity
- Configurable to match your project's coding standards

Ruff helps maintain code quality and consistency across your AI projects, which is essential when working in teams or on complex systems.

**Installation Resources:**
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [VS Code/Cursor Extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

## Week 1 Exercises

### Exercise 1: Environment Setup

1. Install Python 3.10+ on your system and verify the installation:

    ```bash
    python --version
    ```

2. Download and install Cursor IDE from [cursor.com](https://cursor.com)

3. Configure Cursor with the following extensions:
   - [Python Extension Package](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack)
   - [Jupyter](https://marketplace.cursorapi.com/items?itemName=ms-toolsai.jupyter)
   - [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

4. You can configure Ruff to format Python code on-save by enabling the `editor.formatOnSave` action in `settings.json`, and setting Ruff as your default formatter:

    ```
    {
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff"
        }
    }
    ```


5. Install UV using the instructions from the [Getting Started with UV guide](https://daveebbelaar.com/blog/2024/03/20/getting-started-with-uv-the-ultra-fast-python-package-manager/)
   
    ```bash

    # macOS (using Homebrew)
    brew install uv


    # On macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

5. Verify UV installation:

    ```bash
    uv --version
    ```

### Exercise 2: Project Configuration

1. Create a new GitHub repository named "genai-accelerator-projects"

2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/genai-accelerator-projects.git
   cd genai-accelerator-projects
   ```

3. Create a Python virtual environment using UV:
   ```bash
   uv venv
   ```

4. Activate the virtual environment by selecting it within your IDE or by using the terminal via:
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

5. Create a `requirements.txt` file with the following packages:
   ```
   numpy
   pandas
   matplotlib
   openai
   tiktoken
   python-dotenv
   ```

6. Install the requirements using UV:
   ```bash
   uv pip install -r requirements.txt
   ```

7. Create a simple Python script named `environment_test.py` that imports all the installed packages and prints a success message:

   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import openai
   import tiktoken
   import os
   from dotenv import load_dotenv
   
   def check_environment():
       print("Environment successfully configured!")
       print(f"Python version: {os.sys.version}")
       print("\nInstalled packages:")
       print(f"- NumPy: {np.__version__}")
       print(f"- Pandas: {pd.__version__}")
       print(f"- Matplotlib: {plt.matplotlib.__version__}")
       print(f"- OpenAI: {openai.__version__}")
       print(f"- Tiktoken: {tiktoken.__version__}")
   
   if __name__ == "__main__":
       check_environment()
   ```

8. Run the script to verify your environment:
   ```bash
   python environment_test.py
   ```

9. Commit and push your changes to GitHub:
   ```bash
   git add .
   git commit -m "Initial setup and environment configuration"
   git push origin main
   ```

By completing these exercises, you'll have established a solid development environment that we'll build upon throughout the accelerator program. This foundation will enable you to focus on learning AI engineering concepts rather than troubleshooting environment issues.