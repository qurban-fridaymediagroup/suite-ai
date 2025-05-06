import pandas as pd
from difflib import SequenceMatcher
from formula_file.constants import unified_columns
import re
import os

# Load NetSuite canonical data
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file relative to the script location
csv_path = os.path.join(script_dir, "Netsuite info all final data.csv")
netsuite_df = pd.read_csv(csv_path, encoding="ISO-8859-1")

# Canonical dictionary for each field
canonical_dict = {
    col: netsuite_df[col].dropna().astype(str).str.lower().unique().tolist()
    for col in netsuite_df.columns
}

# Define structure: single or multi value fields
field_structure_map = {
    "Subsidiary": "single",
    "Customer Name": "multi",
    "Customer Number": "multi",
    "Vendor Name": "multi",
    "Vendor Number": "multi",
    "From Period": "single",
    "To Period": "single",
    "Account Name": "multi",
    "Account Number": "multi",
    "Class": "multi",
    "Classification": "multi",
    "Brand": "multi",
    "Cost Center": "multi",
    "Department": "multi",
    "Location": "multi",
    "Budget category": "single",
    "high/low": "single",
    "Limit of record": "single"
}

def find_best_column_for_value(value, current_column, canonical_dict, field_structure_map, intent):
    value = value.strip().lower()
    best_match_column = None
    best_score = 0.0
    matched_type = None

    for col, values in canonical_dict.items():
        for ref_val in values:
            score = SequenceMatcher(None, value, ref_val).ratio()
            if score > best_score:
                best_score = score
                best_match_column = col
                matched_type = 'exact' if score > 0.95 else 'partial'

    if best_score >= 0.8 and best_match_column != current_column:
        return best_match_column, matched_type
    return None, None

def smart_intent_correction_restricted(intent):
    corrected_intent = intent.copy()
    match_notes = {}
    displaced_values = []

    for key, value in intent.items():
        if not value or str(value).lower() in ["!", "not found", ""]:
            match_notes[key] = "Empty or already invalid"
            continue

        clean_val = re.sub(r'[{}"]', '', value.lower().strip())
        canonical_key = unified_columns.get(key, key)
        valid_values = canonical_dict.get(canonical_key, [])


        if clean_val in valid_values:
            match_notes[key] = "Exact match"
            continue
        elif any(clean_val in val for val in valid_values):
            best_partial = next((val for val in valid_values if clean_val in val), value)
            corrected_intent[key] = best_partial.title()
            match_notes[key] = "Partial match"
            continue


        best_col, match_type = find_best_column_for_value(value, key, canonical_dict, field_structure_map, intent)

        if best_col and best_col in intent:
            structure = field_structure_map.get(best_col, "single")
            existing_val = corrected_intent.get(best_col, "")

            if structure == "single":
                current_val = corrected_intent.get(key, "")
                if (
                    existing_val.strip() not in ["", "!", "not found"]
                    and current_val.strip() not in ["", "!", "not found"]
                    and existing_val.strip().lower() != value.strip().lower()
                ):
                    temp_value = existing_val
                    corrected_intent[best_col] = value
                    corrected_intent[key] = ""
                    match_notes[key] = f"Moved to {best_col} ({match_type} match)"
                    match_notes[best_col] = f"Inserted from {key}"
                    displaced_values.append((key, temp_value))
                else:
                    corrected_intent[best_col] = value
                    corrected_intent[key] = ""
                    match_notes[key] = f"Moved to {best_col} ({match_type} match)"
                    match_notes[best_col] = f"Inserted from {key}"
            elif structure == "multi":
                current_val = corrected_intent.get(key, "")
                if existing_val:
                    corrected_intent[best_col] = existing_val + "," + value
                else:
                    corrected_intent[best_col] = value
                if current_val.strip().lower() not in ["", "!", "not found"]:
                    displaced_values.append((key, current_val))
                corrected_intent[key] = ""
                match_notes[key] = f"Moved to {best_col} ({match_type} match)"
                match_notes[best_col] = f"Inserted from {key}"
        else:
            match_notes[key] = "Not matched in any known column"

    for original_field, temp_val in displaced_values:
        best_col_2, match_type_2 = find_best_column_for_value(temp_val, "", canonical_dict, field_structure_map, intent)
        if best_col_2 and best_col_2 in corrected_intent:
            if corrected_intent[best_col_2].strip() in ["", "!", "not found"]:
                corrected_intent[best_col_2] = temp_val
                match_notes[best_col_2] = f"Inserted from {original_field}"

    validated_intent = {key: corrected_intent.get(key, "") for key in intent.keys()}
    return {
        "validated_intent": validated_intent,
        "match_notes": match_notes
    }
