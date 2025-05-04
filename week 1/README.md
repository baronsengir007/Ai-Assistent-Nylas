# Week 1 — Foundations of AI Engineering

## Setting Up Your Local Development Environment

Before diving into AI engineering, we need to establish a solid development environment. This foundation will serve you throughout the program and your future AI projects.

### Python Installation

Ensure you have Python 3.10 or newer installed on your system. This provides the necessary features and compatibility with modern AI libraries.

**Installation Resources:**
- [Python Downloads Page](https://www.python.org/downloads/)
- [Official Installation Guide](https://docs.python.org/3/using/index.html)

### Git Installation

Make sure you have Git installed to manage your code repositories and collaborate with others effectively.

**Installation Resources:**
- [Git Downloads Page](https://git-scm.com/downloads)
- [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

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
- [Getting Started with UV](https://daveebbelaar.com/blog/2024/03/20/getting-started-with-uv-the-ultra-fast-python-package-manager/)

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

### Jupyter Notebooks

Jupyter Notebooks are interactive documents that let you combine code, text, and visual output (like charts or tables) in one place. They originally gained popularity in data science because they're perfect for exploring data, running small code snippets step by step, and documenting your findings along the way. Instead of writing one long script, you can test and explain things cell by cell, making your workflow more flexible and transparent. Throughout the lab exercises, you'll see that some parts are in regular Python files, while others are in Jupyter Notebooks, depending on what fits best for the task.

```bash
python_file.py
jupyter_notebook_file.ipynb
```

## Week 1 Exercises

### Exercise 1: Environment Setup

1. Install Python 3.10+ on your system and verify the installation:

    ```bash
    python --version
    ```

2. Clone the GenAI Accelerator Labs repository:

    ```bash
    git clone https://github.com/datalumina/genai-accelerator-labs.git
    cd genai-accelerator-labs
    ```

3. Create a virtual environment using UV:

    ```bash
    uv venv
    ```

4. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

5. Sync dependencies with UV:

    ```bash
    uv sync
    ```

6. You can configure Ruff to format Python code on-save by enabling the `editor.formatOnSave` action in `settings.json`, and setting Ruff as your default formatter:

    ```
    {
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff"
        }
    }
    ```

### Exercise 2: Run Python File

1. With your environment activated, open the repository in Cursor IDE

2. Navigate to `week 1/python-venv-check.py`

3. Execute the file

### Exercise 3: Run the Jupyter Notebook Example

1. With your environment activated, open the repository in Cursor IDE

2. Navigate to `week 1/notebook-introduction.ipynb`

3. Select the right kernel (your newly created venv)

4. Run each cell in the notebook by pressing `Shift+Enter`

5. Experiment with:
   - Creating new cells (press `Esc` then `a` for above or `b` for below)
   - Converting cells between code and markdown (press `Esc` then `m` for markdown or `y` for code)
   - Writing your own code and markdown

6. This exercise will help you become comfortable with Jupyter notebooks, which we'll use throughout the course for interactive exercises and demonstrations.
