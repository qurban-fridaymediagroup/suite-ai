import pandas as pd
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime
import re
import calendar
# from V3.period_utils import get_period_range, normalize_period_string, validate_period_order
from formula_file.period_utils import get_period_range, normalize_period_string, validate_period_order
# from V3.smart_intent_correction import smart_intent_correction_restricted
from formula_file.smart_intent_correction import smart_intent_correction_restricted
import os
from formula_file.constants import unified_columns


# Load NetSuite data
# Get the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
csv_path = os.path.join(current_dir, "Netsuite info all final data.csv")

netsuite_df = pd.read_csv(csv_path, encoding="ISO-8859-1")

# Extract canonical values
canonical_values = {}
for alias, col in unified_columns.items():
    if col not in canonical_values:
        canonical_values[col] = set()
    if col in netsuite_df.columns:
        canonical_values[col].update(netsuite_df[col].dropna().astype(str).str.lower().unique())
for col in canonical_values:
    canonical_values[col] = sorted(list(canonical_values[col]))


def best_partial_match(input_val, possible_vals):
    input_val = input_val.strip().lower()
    for val in possible_vals:
        if input_val in val:
            return val
    best_score = 0
    best_match = None
    for val in possible_vals:
        score = SequenceMatcher(None, input_val, val.strip().lower()).ratio()
        if score > best_score and score > 0.7:
            best_score = score
            best_match = val
    return best_match if best_match else None


def validate_intent_fields_v2(intent_dict):
    validated = {}
    notes = {}
    warnings = []

    # Check for placeholders in the input
    placeholder_patterns = [
        r'\[.*?\]',  # Matches [placeholder]
        r'\{.*?\}',  # Matches {placeholder}
        r'^\{\".*?\"\}$',  # Matches {"placeholder"}
    ]

    # Step 1: Handle periods first
    from_val = intent_dict.get("From Period", "")
    to_val = intent_dict.get("To Period", "")

    # Clean values from extra quotes and braces for checking
    from_val_clean = re.sub(r'[{}\"]', '', str(from_val)).strip()
    to_val_clean = re.sub(r'[{}\"]', '', str(to_val)).strip()

    # Check if periods are placeholders
    from_is_placeholder = any(re.search(pattern, str(from_val)) for pattern in placeholder_patterns)
    to_is_placeholder = any(re.search(pattern, str(to_val)) for pattern in placeholder_patterns)

    # ✅ Updated period handling logic
    try:
        from_p, to_p = get_period_range(from_val_clean, to_val_clean or from_val_clean)
        from_val_final, to_val_final = validate_period_order(from_p, to_p)

        if "Check: From > To" in to_val_final:
            validated["From Period"] = from_p
            validated["To Period"] = to_p + " invalid input"
            notes["From Period"] = validated["From Period"]
            notes["To Period"] = validated["To Period"]
            warnings.append("To Period is earlier than From Period.")
        else:
            validated["From Period"] = from_val_final
            validated["To Period"] = to_val_final
            notes["From Period"] = from_val_final
            notes["To Period"] = to_val_final
    except Exception as e:
        validated["From Period"] = from_val_clean
        validated["To Period"] = to_val_clean or from_val_clean
        notes["From Period"] = "Could not normalize period"
        notes["To Period"] = "Could not normalize period"
        warnings.append(f"Period normalization error: {str(e)}")
    else:
        # Normal period processing
        try:
            from_p, to_p = get_period_range(from_val_clean, to_val_clean)
            from_val_final, to_val_final = validate_period_order(from_p, to_p)

            if "Check: From > To" in to_val_final:
                validated["From Period"] = from_p
                validated["To Period"] = to_p + " invalid input"
                notes["From Period"] = validated["From Period"]
                notes["To Period"] = validated["To Period"]
                warnings.append("To Period is earlier than From Period.")
            else:
                validated["From Period"] = from_val_final
                validated["To Period"] = to_val_final
                notes["From Period"] = from_val_final
                notes["To Period"] = to_val_final
        except Exception as e:
            # If period processing fails, use the original values
            validated["From Period"] = from_val_clean
            validated["To Period"] = to_val_clean or from_val_clean  # Use from_val for to_val if to_val is empty
            notes["From Period"] = "Could not normalize period"
            notes["To Period"] = "Could not normalize period"
            warnings.append(f"Period normalization error: {str(e)}")

    # Step 2: Handle other fields
    for key, value in intent_dict.items():
        if key in ["From Period", "To Period"]:
            continue

        # Check if the value is a placeholder
        is_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)

        # Extract the actual value from the placeholder if it exists
        clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()

        # If it's a generic placeholder like [Customer] or {High_Low}, preserve it
        if is_placeholder and (clean_val.lower() in [key.lower().replace(" ", "_"),
                                                     key.lower(),
                                                     key.lower() + "_number",
                                                     key.lower() + "_name"]):
            validated[key] = value
            notes[key] = "Generic placeholder preserved"
            continue

        # If it's a placeholder with a specific value, extract and validate that value
        value_to_validate = clean_val
        canonical_col = unified_columns.get(key)
        possible_values = canonical_values.get(canonical_col, [])

        # === Special Handling for high/low ===
        if key.lower() == "high/low":
            if value_to_validate in ["high", "low"]:
                validated[key] = value_to_validate
                notes[key] = "Exact match"
            else:
                # Check if it's a placeholder like {High_Low}
                if is_placeholder and "high_low" in clean_val.lower():
                    validated[key] = value
                    notes[key] = "Placeholder preserved"
                else:
                    validated[key] = "high"  # Default to high if not specified
                    notes[key] = "Default value used"
            continue

        # === Special Handling for Limit of record ===
        if key.lower() == "limit of record":
            if value_to_validate.isdigit():
                validated[key] = value_to_validate
                notes[key] = "Valid integer"
            else:
                # Check if it's a placeholder like {Limit_of_Record}
                if is_placeholder and "limit" in clean_val.lower():
                    validated[key] = value
                    notes[key] = "Placeholder preserved"
                else:
                    validated[key] = "10"  # Default to 10 if not specified
                    notes[key] = "Default value used"
            continue

        # === Generic Matching Logic ===
        if value_to_validate in ["", "-", "!", "not found"]:
            validated[key] = ""
            notes[key] = "Empty or already invalid"
            continue

        # Check for exact match in the canonical values
        if value_to_validate in possible_values:
            matched = next((v for v in netsuite_df[canonical_col].dropna().astype(str)
                            if v.strip().lower() == value_to_validate), value_to_validate)
            validated[key] = matched
            notes[key] = "Exact match"
        else:
            # Try partial matching
            partial = best_partial_match(value_to_validate, possible_values)
            if partial:
                matched = next((v for v in netsuite_df[canonical_col].dropna().astype(str)
                                if v.strip().lower() == partial.strip().lower()), partial)
                validated[key] = matched
                notes[key] = "Partial match" if matched.strip().lower() != value_to_validate else "Exact match"
            else:
                # Try to find the value in other columns
                found_in_other_column = False
                for other_col, other_values in canonical_values.items():
                    if other_col != canonical_col:
                        # Try exact match in other column
                        if value_to_validate in other_values:
                            validated[key] = value_to_validate
                            notes[key] = f"Found in {other_col} column"
                            found_in_other_column = True
                            break

                        # Try partial match in other column
                        partial_in_other = best_partial_match(value_to_validate, other_values)
                        if partial_in_other:
                            validated[key] = partial_in_other
                            notes[key] = f"Partial match in {other_col} column"
                            found_in_other_column = True
                            break

                if not found_in_other_column:
                    # If it's a placeholder with a value we couldn't match, preserve the original
                    if is_placeholder:
                        validated[key] = value
                        notes[key] = "Placeholder with unmatched value preserved"
                    else:
                        validated[key] = value_to_validate
                        notes[key] = "Not matched in any known column"
                        warnings.append(f"'{value_to_validate}' in {key} not recognized in any column.")

    # Step 3: Smart reassignment — lock Limit of record and high/low
    smart_input = validated.copy()
    locked_keys = ["Limit of record", "high/low"]
    for lk in locked_keys:
        if lk in smart_input:
            smart_input[lk] = "LOCKED_VALUE"

    smart = smart_intent_correction_restricted(smart_input)
    smart_validated = smart["validated_intent"]

    # Restore locked fields after smart correction
    for lk in locked_keys:
        if lk in validated:
            smart_validated[lk] = validated.get(lk, "")
            smart["match_notes"][lk] = "Locked and preserved"

    # Final merge
    notes.update(smart["match_notes"])
    final_validated = {key: smart_validated.get(key, "") for key in intent_dict.keys()}

    # Clean up unmatched values
    for k, v in final_validated.items():
        # Skip placeholders
        is_placeholder = any(re.search(pattern, str(intent_dict.get(k, ""))) for pattern in placeholder_patterns)
        if is_placeholder:
            continue

        original_val = intent_dict.get(k, "").strip().lower()
        if (
                k not in ["From Period", "To Period", "high/low", "Limit of record"]
                and notes.get(k) == "Not matched in any known column"
                and v.strip().lower() == original_val
        ):
            final_validated[k] = f"{v} is not found"

    return {
        "validated_intent": final_validated,
        "match_notes": notes
    }
