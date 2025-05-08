import pandas as pd
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime
import re
import calendar
import os

# Import custom modules
from formula_file.period_utils import get_period_range, normalize_period_string, validate_period_order
from formula_file.smart_intent_correction import smart_intent_correction_restricted
from formula_file.constants import unified_columns

# Load NetSuite data
current_dir = os.path.dirname(os.path.abspath(__file__))
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

# Field placeholder formatting rules
field_format_map = {
    "Subsidiary": "Subsidiary",
    "Budget category": '"Budget category"',
    "Account Number": '{"Account Number"}',
    "Account Name": '{"Account Name"}',
    "From Period": '"From Period"',
    "To Period": '"To Period"',
    "Classification": '{"Classification"}',
    "Department": '{"Department"}',
    "Location": '{"Location"}',
    "Customer Number": '{"Customer Number"}',
    "Customer Name": '{"Customer Name"}',
    "Vendor Name": '{"Vendor Name"}',
    "Vendor Number": '{"Vendor Number"}',
    "Class": '{"Class"}',
    "high/low": '"high/low"',
    "Limit of record": '"Limit of record"',
    "TABLE_NAME": '"TABLE_NAME"'
}

placeholder_keys = list(field_format_map.keys())

# Define formula mappings with correct parameter sequences
formula_mapping = {
    "SUITEGEN": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITECUS": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEGENREP": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEREC": ["TABLE_NAME"],
    "SUITEBUD": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVEN": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"]
}

# Extended formula mappings with variations - ensuring correct parameter order
extended_formula_mapping = {
    # SUITECUS variations - Customer/Account combinations
    "SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NUMBER": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME": ["Subsidiary", "Customer Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NAME_ACCOUNT_NUMBER": ["Subsidiary", "Customer Name", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    
    # SUITEGEN variations - Account types
    "SUITEGEN_ACCOUNT_NUMBER": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEGEN_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    
    # SUITEVEN variations - Vendor/Account combinations
    "SUITEVEN_VENDOR_NAME_ACCOUNT_NAME": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NUMBER_ACCOUNT_NUMBER": ["Subsidiary", "Vendor Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NAME_ACCOUNT_NUMBER": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Vendor Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    
    # SUITEBUD variations - Budget with Account types
    "SUITEBUD_ACCOUNT_NUMBER": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUD_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    
    # SUITEGENREP variations - Account types
    "SUITEGENREP_ACCOUNT_NUMBER": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEGENREP_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    
    # SUITEBUDREP variations - Budget with Account types
    "SUITEBUDREP_ACCOUNT_NUMBER": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    
    # SUITEVAR variations - Budget with Account types
    "SUITEVAR_ACCOUNT_NUMBER": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"]
}

def get_formula_template(formula_type, intent_dict):
    """
    Determine the correct formula template based on formula type and intent fields
    """
    if formula_type == "SUITEREC":
        return formula_mapping["SUITEREC"]
    
    # Check for specific field combinations
    if formula_type == "SUITECUS":
        has_customer_number = "Customer Number" in intent_dict and intent_dict["Customer Number"]
        has_customer_name = "Customer Name" in intent_dict and intent_dict["Customer Name"]
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_customer_number and has_account_number:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NUMBER"]
        elif has_customer_name and has_account_name:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME"]
        elif has_customer_name and has_account_number:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NAME_ACCOUNT_NUMBER"]
        elif has_customer_number and has_account_name:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME"]
    
    elif formula_type == "SUITEGEN":
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_account_name:
            return extended_formula_mapping["SUITEGEN_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEGEN_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEVEN":
        has_vendor_number = "Vendor Number" in intent_dict and intent_dict["Vendor Number"]
        has_vendor_name = "Vendor Name" in intent_dict and intent_dict["Vendor Name"]
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_vendor_name and has_account_name:
            return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NAME"]
        elif has_vendor_number and has_account_number:
            return extended_formula_mapping["SUITEVEN_VENDOR_NUMBER_ACCOUNT_NUMBER"]
        elif has_vendor_name and has_account_number:
            return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NUMBER"]
        elif has_vendor_number and has_account_name:
            return extended_formula_mapping["SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME"]
    
    elif formula_type == "SUITEBUD":
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_account_name:
            return extended_formula_mapping["SUITEBUD_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEBUD_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEGENREP":
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_account_name:
            return extended_formula_mapping["SUITEGENREP_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEGENREP_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEBUDREP":
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_account_name:
            return extended_formula_mapping["SUITEBUDREP_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEBUDREP_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEVAR":
        has_account_number = "Account Number" in intent_dict and intent_dict["Account Number"]
        has_account_name = "Account Name" in intent_dict and intent_dict["Account Name"]
        
        if has_account_name:
            return extended_formula_mapping["SUITEVAR_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEVAR_ACCOUNT_NUMBER"]
    
    # Default to the base formula mapping if no specific match
    return formula_mapping.get(formula_type, [])

def format_formula_with_intent(formula_type, intent_dict):
    """
    Format a formula string based on formula type and intent dictionary
    """
    # Get the appropriate template based on formula type and intent fields
    template = get_formula_template(formula_type, intent_dict)
    
    # For SUITEREC, handle the special case
    if formula_type == "SUITEREC":
        table_name = intent_dict.get("TABLE_NAME", "")
        return f'SUITEREC("{table_name}")'
    
    # For other formulas, build the parameter string based on the template
    params = []
    for field in template:
        value = intent_dict.get(field, "")
        
        # Format the value based on field type
        if field in ["Subsidiary", "Budget category", "From Period", "To Period", "high/low", "Limit of record"]:
            # These fields should be quoted but not in braces
            params.append(f'"{value}"')
        elif field in ["Customer Number", "Customer Name", "Account Number", "Account Name", 
                      "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
            # These fields should be in braces
            params.append(f'{{"{value}"}}')
        else:
            # Default formatting
            params.append(f'"{value}"')
    
    # Join parameters and return the formatted formula
    return f'{formula_type}({", ".join(params)})'

def validate_gpt_formula_output(gpt_formula: str) -> dict:
    """
    Validate and extract parameters from GPT formula output
    Returns a dictionary of parameters that can be used for validation
    """
    # First, check if it's a SUITEREC formula with a special format
    if "SUITEREC" in gpt_formula.upper():
        # Extract the table names if they're in a set format
        table_set_match = re.search(r'SUITEREC\(\{(.*?)\}\)', gpt_formula, re.IGNORECASE)
        if table_set_match:
            table_names = table_set_match.group(1)
            # Convert to proper format
            return {"TABLE_NAME": table_names}
        return {}
    
    # For other formulas, extract the formula type and parameters
    formula_match = re.search(r'([A-Z]+)\((.*?)\)', gpt_formula, re.IGNORECASE)
    if not formula_match:
        return {}
    
    formula_type = formula_match.group(1).upper()
    params_str = formula_match.group(2)
    
    # Parse parameters
    params = []
    current_param = ""
    in_braces = False
    in_quotes = False
    in_brackets = False
    
    for char in params_str:
        if char == '{' and not in_quotes and not in_brackets:
            in_braces = True
            current_param += char
        elif char == '}' and not in_quotes and not in_brackets:
            in_braces = False
            current_param += char
        elif char == '[' and not in_quotes and not in_braces:
            in_brackets = True
            current_param += char
        elif char == ']' and not in_quotes and not in_braces:
            in_brackets = False
            current_param += char
        elif char == '"' and not in_braces and not in_brackets:
            in_quotes = not in_quotes
            current_param += char
        elif char == ',' and not in_braces and not in_quotes and not in_brackets:
            # Clean parameter value before adding
            clean_param = current_param.strip()
            if clean_param.startswith('[') and clean_param.endswith(']'):
                clean_param = clean_param[1:-1]
            elif clean_param.startswith('{') and clean_param.endswith('}'):
                clean_param = clean_param[1:-1]
            params.append(clean_param)
            current_param = ""
        else:
            current_param += char
    
    if current_param:
        clean_param = current_param.strip()
        if clean_param.startswith('[') and clean_param.endswith(']'):
            clean_param = clean_param[1:-1]
        elif clean_param.startswith('{') and clean_param.endswith('}'):
            clean_param = clean_param[1:-1]
        params.append(clean_param)
    
    # Create a dictionary from the parameters
    intent_dict = {}
    
    # Get the template for this formula type
    template = formula_mapping.get(formula_type, [])
    
    # Map parameters to fields based on their position in the template
    for i, field in enumerate(template):
        if i < len(params):
            # Clean and format the parameter value
            value = params[i].strip('"\'')
            if field in ["Subsidiary", "Budget category", "From Period", "To Period", "high/low", "Limit of record"]:
                intent_dict[field] = value
            else:
                intent_dict[field] = f'"{value}"'
    
    return intent_dict

def process_gpt_formula(gpt_formula: str):
    """
    Process a GPT formula and return validated intent
    """
    # Extract formula type
    formula_match = re.search(r'([A-Z]+)\(', gpt_formula, re.IGNORECASE)
    if not formula_match:
        return {"error": "Invalid formula format"}
    
    formula_type = formula_match.group(1).upper()
    
    # Extract parameters as a dictionary
    intent_dict = validate_gpt_formula_output(gpt_formula)
    
    # Create a new dictionary to store the processed values
    processed_intent = {}
    
    # Process each parameter based on its field type
    for field, value in intent_dict.items():
        # Check if it's a placeholder (in brackets or braces)
        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            processed_intent[field] = value
        else:
            # For non-placeholder values, extract the actual value
            clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip()
            processed_intent[field] = clean_val
    
    # Validate the processed intent fields
    validation_result = validate_intent_fields_v2(processed_intent)
    
    # Add formula type to the result
    validation_result["formula_type"] = formula_type
    
    return validation_result

def format_final_formula(validation_result):
    """
    Format the final formula based on validation result
    """
    formula_type = validation_result.get("formula_type")
    validated_intent = validation_result.get("validated_intent", {})
    original_intent = validation_result.get("original_intent", {})

    # Get the template for this formula type
    template = formula_mapping.get(formula_type, [])

    # Build parameters list
    params = []
    for field in template:
        # Use original intent if present and is a string
        if field in original_intent and isinstance(original_intent[field], str):
            value = original_intent[field]
        else:
            value = validated_intent.get(field, "")

        # Special handling for Subsidiary
        if field == "Subsidiary":
            params.append('"Friday Media Group (Consolidated)"')
            continue

        # Step 1: Convert square brackets to curly
        value = value.replace('[', '{').replace(']', '}')
        # Step 2: Remove all curly brackets
        value = value.replace('{', '').replace('}', '')
        
        # Handle empty values - leave them as empty strings to create commas
        if not value.strip():
            params.append("")  # Empty string creates a comma with nothing between
        else:
            # Add quotes around non-empty values
            params.append(f'"{value}"')

    # Join parameters and return the formatted formula
    return f'{formula_type}({", ".join(params)})'

def format_placeholder(value):
    """Ensures consistent placeholder formatting."""
    return value if isinstance(value, str) else str(value)

# Utility functions
def best_partial_match(input_val, possible_vals, field_name=None):
    """Find the best partial match for a value in a list of possible values.
    Uses field-specific thresholds for better matching."""
    input_val = input_val.strip().lower()
    
    # First check for direct substring match
    for val in possible_vals:
        if input_val in val:
            return val
    
    # Set field-specific thresholds
    field_thresholds = {
        "Subsidiary": 0.6,
        "Classification": 0.65,
        "Class": 0.65,
        "Department": 0.7,
        "Location": 0.7,
        "Budget category": 0.7,
        "Currency": 0.8,
        "Account Number": 0.8,
        "Account Name": 0.65,
        "Customer Number": 0.8,
        "Customer Name": 0.65,
        "Vendor Number": 0.8,
        "Vendor Name": 0.65
    }
    
    # Then check for similarity match with appropriate threshold
    threshold = field_thresholds.get(field_name, 0.7)  # Default to 0.7 if field not specified
    best_score = 0
    best_match = None
    
    for val in possible_vals:
        score = SequenceMatcher(None, input_val, val.strip().lower()).ratio()
        if score > best_score and score > threshold:
            best_score = score
            best_match = val
    
    return best_match if best_match else None

# Main validation function
def validate_intent_fields_v2(intent_dict):
    validated = {}
    notes = {}
    warnings = []

    placeholder_patterns = [
        r'\[.*?\]', r'\{.*?\}', r'^\{\".*?\"\}$'
    ]

    # Handle periods
    from_val = intent_dict.get("From Period", "")
    to_val = intent_dict.get("To Period", "")
    from_val_clean = re.sub(r'[{}\"]', '', str(from_val)).strip()
    to_val_clean = re.sub(r'[{}\"]', '', str(to_val)).strip()
    from_is_placeholder = any(re.search(pattern, str(from_val)) for pattern in placeholder_patterns)
    to_is_placeholder = any(re.search(pattern, str(to_val)) for pattern in placeholder_patterns)

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

    # Handle other fields
    for key, value in intent_dict.items():
        if key in ["From Period", "To Period"]:
            continue

        is_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)
        clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()

        if is_placeholder and clean_val in [
            key.lower(), key.lower().replace(" ", "_"), key.lower() + "_name", key.lower() + "_number"
        ]:
            validated[key] = value
            notes[key] = "Generic placeholder preserved"
            continue

        if is_placeholder:
            validated[key] = format_placeholder(value)
            notes[key] = "Placeholder preserved"
            continue

        canonical_col = unified_columns.get(key)
        possible_values = canonical_values.get(canonical_col, [])

        if key.lower() == "high/low":
            if clean_val in ["high", "low"]:
                validated[key] = clean_val
                notes[key] = "Exact match"
            elif "high_low" in clean_val:
                validated[key] = value
                notes[key] = "Placeholder preserved"
            else:
                validated[key] = "high"
                notes[key] = "Default value used"
            continue

        if key.lower() == "limit of record":
            if clean_val.isdigit():
                validated[key] = clean_val
                notes[key] = "Valid integer"
            elif "limit" in clean_val:
                validated[key] = clean_val
                notes[key] = "Placeholder preserved"
            else:
                validated[key] = "10"
                notes[key] = "Default value used"
            continue

        if clean_val in ["", "-", "!", "not found"]:
            validated[key] = ""
            notes[key] = "Empty or already invalid"
            continue

        if clean_val in possible_values:
            matched = next((v for v in netsuite_df[canonical_col].dropna().astype(str)
                            if v.strip().lower() == clean_val), clean_val)
            validated[key] = matched
            notes[key] = "Exact match"
        else:
            # Pass the field name to best_partial_match for field-specific thresholds
            partial = best_partial_match(clean_val, possible_values, key)
            if partial:
                matched = next((v for v in netsuite_df[canonical_col].dropna().astype(str)
                                if v.strip().lower() == partial.strip().lower()), partial)
                validated[key] = matched
                notes[key] = "Partial match" if matched.strip().lower() != clean_val else "Exact match"
            else:
                found = False
                for other_col, other_vals in canonical_values.items():
                    if other_col != canonical_col:
                        if clean_val in other_vals:
                            validated[key] = clean_val
                            notes[key] = f"Found in {other_col} column"
                            found = True
                            break
                        partial_other = best_partial_match(clean_val, other_vals)
                        if partial_other:
                            validated[key] = partial_other
                            notes[key] = f"Partial match in {other_col} column"
                            found = True
                            break
                if not found:
                    validated[key] = value if is_placeholder else clean_val
                    note_msg = "Placeholder with unmatched value preserved" if is_placeholder else "Not matched in any known column"
                    notes[key] = note_msg
                    if not is_placeholder:
                        warnings.append(f"'{clean_val}' in {key} not recognized in any column.")

    # Smart correction
    smart_input = validated.copy()
    for lk in ["Limit of record", "high/low"]:
        if lk in smart_input:
            smart_input[lk] = "LOCKED_VALUE"

    smart = smart_intent_correction_restricted(smart_input)
    smart_validated = smart["validated_intent"]

    for lk in ["Limit of record", "high/low"]:
        if lk in validated:
            smart_validated[lk] = validated.get(lk, "")

    # Apply fuzzy matching correction
    final_validated = correct_validated_intent_with_fuzzy(smart_validated, canonical_values)

    return {
        "validated_intent": final_validated,
        "match_notes": notes,
        "warnings": warnings
    }


def correct_validated_intent_with_fuzzy(validated_intent: dict, canonical_values: dict) -> dict:
    """
    Apply fuzzy matching to correct values in validated intent against canonical values.
    Uses a 60% similarity threshold for matching.
    """
    corrected_intent = {}

    for field, value in validated_intent.items():
        # Skip From Period and To Period
        if field in ["From Period", "To Period"]:
            corrected_intent[field] = value
            continue

        # Skip empty values
        if not value:
            corrected_intent[field] = value
            continue

        # Clean raw value - strip brackets, quotes, etc.
        raw = re.sub(r'^[{\["]*|[}\]"]*$', '', str(value)).lower()
        
        # Skip placeholder values
        if raw in [field.lower(), field.lower().replace(" ", "_"), 
                  field.lower() + "_name", field.lower() + "_number"]:
            corrected_intent[field] = value
            continue
            
        # Skip if field not in canonical values
        canonical_col = unified_columns.get(field)
        if not canonical_col or canonical_col not in canonical_values:
            corrected_intent[field] = value
            continue
            
        possible_values = canonical_values[canonical_col]
        
        # Try to find a close match with 60% similarity
        matches = get_close_matches(raw, possible_values, n=1, cutoff=0.6)
        
        if matches:
            corrected_value = matches[0]
            
            # Get the original case from the NetSuite data
            original_case = next((v for v in netsuite_df[canonical_col].dropna().astype(str)
                                if v.strip().lower() == corrected_value), corrected_value)
            
            # Format based on field type
            if field in ["Customer Number", "Customer Name", "Account Number", "Account Name",
                         "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
                formatted = f'{{"{original_case}"}}'
            elif field in ["Subsidiary", "Budget category", "high/low", "Limit of record", "TABLE_NAME"]:
                formatted = f"'{original_case}'"  # Using single quotes as requested
            else:
                formatted = f"'{original_case}'"
                
            corrected_intent[field] = formatted
        else:
            # Keep original if no match
            corrected_intent[field] = value

    return corrected_intent
    