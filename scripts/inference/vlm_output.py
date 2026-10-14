"""Parse one VLM response without importing model or GPU dependencies."""

import json
import re


def format_output_to_string(output_value) -> str:
    """Keep the existing display format for diagnosis and suggestion fields."""
    if isinstance(output_value, str):
        return output_value
    if isinstance(output_value, list):
        formatted_items = []
        for item in output_value:
            if isinstance(item, dict):
                dimension = item.get("Dimension", "")
                detail = item.get("Detail", "")
                if detail and detail.lower() != "none":
                    formatted_items.append(f"【{dimension}】{detail}")
            else:
                formatted_items.append(str(item))
        return "\n".join(formatted_items)
    return str(output_value)


def parse_vlm_output(output_text: str) -> dict:
    """Return result fields, preserving invalid responses for downstream rejection.

    Only the JSON object and renderer prompt are validated here. Diagnosis and
    suggestion fields retain their existing display formatting; this is not a
    validator for the full photography response schema.
    """
    result = {
        "Issues_Analysis": "",
        "Professional_Suggestions": "",
        "Image_Editing_Prompt": "",
        "model_output_raw": output_text,
    }
    if output_text.startswith("ERROR"):
        result["error"] = output_text
        return result

    try:
        json_text = output_text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL | re.IGNORECASE)
        if fence:
            json_text = fence.group(1)
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("VLM output must be a JSON object.")
        prompt = parsed.get("Image_Editing_Prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Image_Editing_Prompt must be a non-empty string.")

        # Build all fields before updating so a formatting failure cannot leave
        # a usable renderer prompt attached to a partially parsed response.
        fields = {
            "Issues_Analysis": format_output_to_string(parsed.get("Issues_Analysis", "")),
            "Professional_Suggestions": format_output_to_string(parsed.get("Professional_Suggestions", "")),
            "Image_Editing_Prompt": prompt,
        }
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        # JSON types that the legacy formatter cannot display must fail this
        # sample rather than interrupting the rest of the inference batch.
        result["parse_error"] = str(exc)
    else:
        result.update(fields)
    return result
