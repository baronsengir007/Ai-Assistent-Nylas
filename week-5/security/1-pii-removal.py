"""
PII Removal Example

Demonstrates how to anonymize personal identifiable information
using LLM Guard's Anonymize scanner.
"""

from llm_guard import scan_prompt
from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import DEBERTA_AI4PRIVACY_v2_CONF
from llm_guard.vault import Vault

# Initialize the vault and anonymizer
vault = Vault()
anonymizer = Anonymize(vault, recognizer_conf=DEBERTA_AI4PRIVACY_v2_CONF, language="en")


def remove_pii(prompt: str) -> tuple[str, bool]:
    """Remove PII from prompt and return sanitized text."""
    sanitized_prompt, is_valid, risk_score = scan_prompt([anonymizer], prompt)
    return sanitized_prompt, is_valid


if __name__ == "__main__":
    # Example with PII
    unsafe_prompt = """
    Hi, my name is John Smith and my email is john.smith@company.com. 
    My phone number is 555-123-4567 and I live at 123 Main Street, 
    New York, NY 10001. Can you help me with my account?
    """

    print("Original prompt:")
    print(unsafe_prompt)

    # Remove PII
    safe_prompt, is_valid = remove_pii(unsafe_prompt)

    print("\nSanitized prompt:")
    print(safe_prompt)
    print(f"\nValid: {is_valid}")
