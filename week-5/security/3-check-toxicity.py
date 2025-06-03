"""
Toxicity Detection Example

Demonstrates how to detect toxic or harmful content in prompts
using LLM Guard's Toxicity scanner.
"""

from llm_guard import scan_prompt
from llm_guard.input_scanners import Toxicity
from llm_guard.input_scanners.toxicity import DEFAULT_MODEL, MatchType

# Initialize the toxicity detector
toxicity_detector = Toxicity(
    threshold=0.5, model=DEFAULT_MODEL, match_type=MatchType.SENTENCE
)


def check_for_toxicity(prompt: str) -> tuple[str, bool, float]:
    """Check prompt for toxic content."""
    sanitized_prompt, is_valid_dict, risk_score_dict = scan_prompt(
        [toxicity_detector], prompt
    )

    # Extract the actual values from the dictionaries
    is_valid = is_valid_dict["Toxicity"]
    risk_score = risk_score_dict["Toxicity"]

    return sanitized_prompt, is_valid, risk_score


if __name__ == "__main__":
    # Test prompt without toxicity
    safe_prompt = (
        "What are some effective strategies for team collaboration and productivity?"
    )
    print("Testing safe prompt:")
    print(f"Prompt: {safe_prompt}")

    _, is_valid, risk_score = check_for_toxicity(safe_prompt)
    print(f"Valid: {is_valid}, Risk Score: {risk_score:.2f}")

    if is_valid:
        print("✅ Prompt is safe - no toxic content detected")
    else:
        print("⚠️  Toxic content detected! Request blocked.")

    print("\n" + "=" * 50 + "\n")

    # Test prompt with toxic content
    toxic_prompt = """
    You're such an idiot! I hate dealing with incompetent people like you. 
    This is absolutely terrible and you should be ashamed of yourself.
    Everyone thinks you're worthless and pathetic.
    """

    print("Testing toxic prompt:")
    print(f"Prompt: {toxic_prompt}")

    _, is_valid, risk_score = check_for_toxicity(toxic_prompt)
    print(f"Valid: {is_valid}, Risk Score: {risk_score:.2f}")

    if is_valid:
        print("✅ Prompt is safe - no toxic content detected")
    else:
        print("⚠️  Toxic content detected! Request blocked.")
        print("\nNote: This helps prevent harmful content from reaching the LLM.")
