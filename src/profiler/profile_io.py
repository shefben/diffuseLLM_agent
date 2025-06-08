import json
from pathlib import Path
from typing import Dict, Any, Optional # Added Optional

# Default path for the style fingerprint, can be overridden.
# The pyproject.toml placeholder used 'config/style_fingerprint.json'
DEFAULT_FINGERPRINT_PATH = Path("config") / "style_fingerprint.json"

def save_style_profile(
    profile_data: Dict[str, Any],
    output_path: Path = DEFAULT_FINGERPRINT_PATH
) -> None:
    """
    Serializes the final style profile dictionary to a JSON file.

    Args:
        profile_data: A dictionary containing the unified style profile.
        output_path: The path where the JSON file will be saved.
                     Defaults to 'config/style_fingerprint.json'.

    Raises:
        IOError: If there's an error writing the file.
        TypeError: If profile_data is not serializable to JSON.
    """
    try:
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4) # Use indent for readability
        print(f"Style profile saved to {output_path.resolve()}")
    except TypeError as e:
        # This might happen if the profile_data contains non-serializable types (e.g., sets directly)
        # The .to_dict() methods in dataclasses should handle this, but good to be aware.
        print(f"Error: Profile data is not JSON serializable: {e}")
        raise
    except IOError as e:
        print(f"Error writing style profile to {output_path.resolve()}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving style profile: {e}")
        raise

def load_style_profile(
    input_path: Path = DEFAULT_FINGERPRINT_PATH
) -> Optional[Dict[str, Any]]:
    """
    Loads a style profile from a JSON file.

    Args:
        input_path: The path to the JSON file.
                    Defaults to 'config/style_fingerprint.json'.

    Returns:
        A dictionary containing the style profile, or None if the file
        is not found or cannot be parsed.
    """
    if not input_path.exists():
        print(f"Warning: Style profile file not found at {input_path.resolve()}")
        return None

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        return profile_data
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {input_path.resolve()}: {e}")
        return None
    except IOError as e:
        print(f"Error reading style profile from {input_path.resolve()}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading style profile: {e}")
        return None


if __name__ == '__main__':
    # Example Usage:
    # First, get a mock unified profile from the diffusion_interfacer (or create one manually)

    # This direct import assumes llm_interfacer and diffusion_interfacer are in the same directory
    # or installed in a way Python can find them.
    # For robust testing, one might mock these or ensure PYTHONPATH is set.
    try:
        # To make this runnable when 'src' is the current directory or in PYTHONPATH:
        if __package__ is None or __package__ == '': # If running as a script
            from llm_interfacer import get_style_fingerprint_from_llm
            from diffusion_interfacer import unify_fingerprints_with_diffusion
        else: # If imported as part of a package
            from .llm_interfacer import get_style_fingerprint_from_llm
            from .diffusion_interfacer import unify_fingerprints_with_diffusion

        # Create some mock sample fingerprints
        mock_samples_for_io = [
            get_style_fingerprint_from_llm("def foo(): pass"),
            get_style_fingerprint_from_llm("class Bar: \n  x=1"),
            get_style_fingerprint_from_llm("def baz(a: int) -> str: \n  return str(a)")
        ]
        test_profile = unify_fingerprints_with_diffusion(mock_samples_for_io)
    except ImportError as e:
        print(f"Note: llm_interfacer or diffusion_interfacer not found for full example (Error: {e}). Using dummy profile.")
        # Fallback to a simpler dummy profile if imports fail (e.g. running file directly)
        test_profile = {
            "indent_width": 4,
            "preferred_quotes": "single",
            "docstring_style": "google",
            "max_line_length": 88,
            "identifier_snake_case_pct": 0.85,
            "directory_overrides": {"src/legacy/": {"max_line_length": 100}},
            "confidence_score": 0.90
        }

    # Define a custom path for this example
    example_output_path = Path("temp_style_fingerprint.json")

    print(f"\n--- Saving Profile to {example_output_path.name} ---")
    try:
        save_style_profile(test_profile, output_path=example_output_path)

        print(f"\n--- Loading Profile from {example_output_path.name} ---")
        loaded_profile = load_style_profile(input_path=example_output_path)

        if loaded_profile:
            print("Profile loaded successfully. Comparing (subset):")
            for key in ["indent_width", "preferred_quotes", "confidence_score"]:
                if key in loaded_profile and key in test_profile:
                    print(f"  {key}: Loaded='{loaded_profile[key]}', Original='{test_profile[key]}', Match: {loaded_profile[key] == test_profile[key]}")
                elif key in loaded_profile:
                     print(f"  {key}: Loaded='{loaded_profile[key]}'")
                elif key in test_profile:
                     print(f"  {key}: Original='{test_profile[key]}'")
            if loaded_profile == test_profile:
                print("Full loaded profile matches original profile.")
            else:
                print("Full loaded profile DOES NOT match original profile (or some keys were missing).")
        else:
            print("Failed to load profile.")

    except Exception as e:
        print(f"An error occurred in the example: {e}")
    finally:
        # Clean up the temporary file
        if example_output_path.exists():
            example_output_path.unlink()
            print(f"Cleaned up {example_output_path.name}")
