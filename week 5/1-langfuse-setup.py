import json
from typing import Any, Dict

from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai

"""
Make sure you have set the environment variables in the .env file:
LANGFUSE_SECRET_KEY=your-langfuse-api-secret-here
LANGFUSE_PUBLIC_KEY=your-langfuse-api-key-here
LANGFUSE_HOST=your-langfuse-host-here
"""

load_dotenv()

# --------------------------------------------------------------
# Using the observe decorator to log the LLM calls
# --------------------------------------------------------------


@observe()
def story():
    return (
        openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a great storyteller."},
                {
                    "role": "user",
                    "content": "Once upon a time in a galaxy far, far away...",
                },
            ],
        )
        .choices[0]
        .message.content
    )


# --------------------------------------------------------------
# Run the main function
# --------------------------------------------------------------


@observe()
def main():
    return story()


main()

# --------------------------------------------------------------
# Check the Langfuse dashboard to see the logged LLM calls
# --------------------------------------------------------------

"""
You can see the logged LLM calls in the Langfuse dashboard:
https://us.cloud.langfuse.com/ or https://eu.cloud.langfuse.com/
"""


# --------------------------------------------------------------
# Using the langfuse_context to update the observations
# --------------------------------------------------------------


@observe()
def generate_story_prompt() -> str:
    """Generate the initial story prompt.

    Returns:
        str: The generated story prompt.
    """
    prompt = (
        "Once upon a time in a galaxy far, far away, "
        "there was a brave explorer who discovered a mysterious planet."
    )

    # Update observation with metadata
    langfuse_context.update_current_observation(
        metadata={
            "step": "prompt_generation",
            "prompt_type": "story",
            "prompt_length": len(prompt),
        },
        name="generate_story_prompt",
        input={"request": "Generate story prompt"},
        output={"prompt": prompt},
    )

    return prompt


@observe()
def generate_story(prompt: str) -> str:
    """Generate a story based on the provided prompt.

    Args:
        prompt (str): The story prompt to use.

    Returns:
        str: The generated story.
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a great storyteller."},
            {"role": "user", "content": prompt},
        ],
    )

    story = response.choices[0].message.content

    # Update observation with metadata
    langfuse_context.update_current_observation(
        metadata={
            "step": "story_generation",
            "model": "gpt-4o",
            "story_length": len(story),
        },
        name="generate_story",
        input={"prompt": prompt},
        output={"story": story},
    )

    return story


@observe()
def analyze_story(story: str) -> Dict[str, Any]:
    """Analyze the generated story for key metrics.

    Args:
        story (str): The story to analyze.

    Returns:
        Dict[str, Any]: Analysis results.
    """
    analysis = {
        "word_count": len(story.split()),
        "sentence_count": len(story.split(".")),
        "has_hero": "hero" in story.lower(),
        "has_conflict": any(
            word in story.lower() for word in ["conflict", "battle", "fight"]
        ),
    }

    # Update observation with metadata
    langfuse_context.update_current_observation(
        metadata={
            "step": "story_analysis",
            "analysis_type": "basic_metrics",
        },
        name="analyze_story",
        input={"story": story},
        output={"analysis": analysis},
    )

    return analysis


@observe()
def main():
    """Main function to orchestrate the story generation and analysis process."""
    # Step 1: Generate the story prompt
    prompt = generate_story_prompt()

    # Step 2: Generate the story
    story = generate_story(prompt)

    # Step 3: Analyze the story
    analysis = analyze_story(story)

    # Update the main observation with overall metadata
    langfuse_context.update_current_observation(
        metadata={
            "total_steps": 3,
            "final_story_length": len(story),
            "analysis_results": analysis,
        },
        name="story_generation_and_analysis",
        input={"request": "Generate and analyze story"},
        output={"story": story, "analysis": analysis},
        tags=["demo"],
    )

    return story, analysis


if __name__ == "__main__":
    story, analysis = main()
    print("\nGenerated Story:")
    print(story)
    print("\nAnalysis Results:")
    print(json.dumps(analysis, indent=2))


"""
You can see the logged LLM calls and metadata in the Langfuse dashboard:
https://us.cloud.langfuse.com/ or https://eu.cloud.langfuse.com/

This example demonstrates how to:
1. Track detailed metadata at each step of your process
2. Use langfuse_context to update observations with additional information
3. Structure your code to capture meaningful metrics and context
4. Track the flow of data through multiple steps
"""
