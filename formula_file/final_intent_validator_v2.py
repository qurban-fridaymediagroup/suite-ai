import pandas as pd
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime
import re
import calendar
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import custom modules (assuming they don't rely on removed dependencies)
from formula_file.period_utils import get_period_range, normalize_period_string, validate_period_order
from formula_file.smart_intent_correction import smart_intent_correction_restricted
from formula_file.constants import unified_columns

# Load NetSuite data
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Netsuite info all final data.csv")
    netsuite_df = pd.read_csv(csv_path, encoding="ISO-8859-1")
except Exception as e:
    print(f"Error loading NetSuite data: {e}")
    netsuite_df = pd.DataFrame()

# Initialize canonical values for all columns in the CSV
canonical_values = {col.lower(): set() for col in netsuite_df.columns}
# Add specific fields expected by the application
expected_fields = [
    'Subsidiary', 'Classification', 'Department', 'Location', 'Budget category',
    'Currency', 'Account Number', 'Account Name', 'Customer Number', 'Customer Name',
    'Vendor Number', 'Vendor Name', 'Class'
]
for field in expected_fields:
    if field.lower() not in canonical_values:
        canonical_values[field.lower()] = set()

# Updated column variations with new aliases
column_variations = {
    'Subsidiary': ['Subsidiary', 'Sub', 'Subsidiaries', 'Subsidiary_Name'],
    'Classification': ['Classification', 'Brand', 'Cost Center', 'Cost_Center', 'Class'],
    'Department': ['Department', 'Dept', 'Departments'],
    'Location': ['Location', 'Loc', 'Locations', 'location', 'loc'],
    'Budget category': ['Budget category', 'Budget_Category', 'Category', 'budget_category', 'bud', 'budget catgory'],
    'Currency': ['Currency', 'Currencies'],
    'Account Number': ['Account Number', 'Account_No', 'Acct_Number', 'Account_Number', 'a/c', 'account_name'],
    'Account Name': ['Account Name', 'Account_Name', 'Acct_Name', 'a/c', 'account_name'],
    'Customer Number': ['Customer Number', 'Customer_No', 'Cust_Number', 'Customer_Number'],
    'Customer Name': ['Customer Name', 'Customer_Name', 'Cust_Name'],
    'Vendor Number': ['Vendor Number', 'Vendor_No', 'Vend_Number', 'Vendor', 'Vendors'],
    'Vendor Name': ['Vendor Name', 'Vendor_Name', 'Vend_Name', 'Vendor', 'Vendors'],
    'Class': ['Class', 'Classes', 'Clas']
}

# Match column names (without rapidfuzz)
column_mapping = {}
csv_columns = list(netsuite_df.columns)
csv_columns_lower = [col.lower() for col in csv_columns]
for expected_col, variations in column_variations.items():
    matched = False
    for variation in variations:
        if variation.lower() in csv_columns_lower:
            idx = csv_columns_lower.index(variation.lower())
            column_mapping[expected_col.lower()] = csv_columns[idx]
            matched = True
            break
    if not matched:
        column_mapping[expected_col.lower()] = expected_col  # Fallback to expected name

# Extract canonical values with updated column mappings
for alias, col in unified_columns.items():
    mapped_col = column_mapping.get(col.lower(), col)
    if mapped_col in netsuite_df.columns:
        canonical_values[col.lower()].update(netsuite_df[mapped_col].dropna().astype(str).str.lower().unique())

# Handle aliases for Classification/Brand/Cost Center/Class
classification_aliases = ['Classification', 'Brand', 'Cost Center', 'Class']
for alias in classification_aliases:
    mapped_col = column_mapping.get(alias.lower(), alias)
    if mapped_col in netsuite_df.columns:
        canonical_values['classification'].update(netsuite_df[mapped_col].dropna().astype(str).str.lower().unique())

# Handle aliases for Subsidiary/Sub
subsidiary_aliases = ['Subsidiary', 'Sub']
for alias in subsidiary_aliases:
    mapped_col = column_mapping.get(alias.lower(), alias)
    if mapped_col in netsuite_df.columns:
        canonical_values['subsidiary'].update(netsuite_df[mapped_col].dropna().astype(str).str.lower().unique())

# Handle Vendor column for Vendor Name and Vendor Number
vendor_col = column_mapping.get('vendor name', 'Vendor')
if vendor_col in netsuite_df.columns:
    vendor_values = netsuite_df[vendor_col].dropna().astype(str).str.lower().unique()
    canonical_values['vendor name'].update(vendor_values)
    canonical_values['vendor number'].update(vendor_values)

# Convert sets to sorted lists
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
    "SUITEGEN": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITECUS": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEGENREP": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEREC": ["TABLE_NAME"],
    "SUITEBUD": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVEN": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"]
}

# Extended formula mappings with variations
extended_formula_mapping = {
    "SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME": ["Subsidiary", "Customer Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEGEN_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVEN_VENDOR_NAME_ACCOUNT_NAME": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Vendor Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEBUD_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEGENREP_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"]
}

def get_formula_template(formula_type, intent_dict):
    """
    Determine the correct formula template based on formula type and intent fields
    """
    try:
        if formula_type == "SUITEREC":
            return formula_mapping["SUITEREC"]

        def has_valid_match(field, value):
            if not value or not isinstance(value, str):
                return False
            clean_val = re.sub(r'[{}"\'\[\]]', '', value).strip().lower()
            if clean_val in ['none', '', 'null', 'placeholder']:
                return False
            possible_values = canonical_values.get('account name', [])
            if not possible_values:
                return False
            match = best_partial_match(clean_val, possible_values, 'Account Name')
            return match is not None

        if formula_type == "SUITECUS":
            customer_number = intent_dict.get("Customer Number", "")
            customer_name = intent_dict.get("Customer Name", "")
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))

            has_customer_number = has_valid_match("Customer Number", customer_number)
            has_customer_name = has_valid_match("Customer Name", customer_name)
            has_account_name = has_valid_match("Account Name", account_name)

            if has_customer_number and has_account_name:
                return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME"]
            elif has_customer_name and has_account_name:
                return extended_formula_mapping["SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME"]
            return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME"]

        elif formula_type == "SUITEGEN":
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))
            has_account_name = has_valid_match("Account Name", account_name)
            return extended_formula_mapping["SUITEGEN_ACCOUNT_NAME"]

        elif formula_type == "SUITEVEN":
            vendor_number = intent_dict.get("Vendor Number", "")
            vendor_name = intent_dict.get("Vendor Name", "")
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))

            has_vendor_number = has_valid_match("Vendor Number", vendor_name)
            has_vendor_name = has_valid_match("Vendor Name", vendor_name)
            has_account_name = has_valid_match("Account Name", account_name)

            if has_vendor_name and has_account_name:
                return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NAME"]
            elif has_vendor_number and has_account_name:
                return extended_formula_mapping["SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME"]
            return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NAME"]

        elif formula_type == "SUITEBUD":
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))
            has_account_name = has_valid_match("Account Name", account_name)
            return extended_formula_mapping["SUITEBUD_ACCOUNT_NAME"]

        elif formula_type == "SUITEGENREP":
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))
            has_account_name = has_valid_match("Account Name", account_name)
            return extended_formula_mapping["SUITEGENREP_ACCOUNT_NAME"]

        elif formula_type == "SUITEBUDREP":
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))
            has_account_name = has_valid_match("Account Name", account_name)
            return extended_formula_mapping["SUITEBUDREP_ACCOUNT_NAME"]

        elif formula_type == "SUITEVAR":
            account_name = intent_dict.get("Account Name", intent_dict.get("Account Number", ""))
            has_account_name = has_valid_match("Account Name", account_name)
            return extended_formula_mapping["SUITEVAR_ACCOUNT_NAME"]

        return formula_mapping.get(formula_type, [])
    except Exception as e:
        print(f"Error in get_formula_template: {e}")
        return []

def format_formula_with_intent(formula_type, intent_dict):
    """
    Format a formula string based on formula type and intent dictionary
    """
    try:
        template = get_formula_template(formula_type, intent_dict)

        if formula_type == "SUITEREC":
            table_name = intent_dict.get("TABLE_NAME", "")
            return f'SUITEREC("{table_name}")'

        # Get current month and year
        # Format current month in 3-letter abbreviation (e.g. "Apr 2025")
        current_month_year = datetime.now().strftime("%b %Y") 
        
        params = []
        for field in template:
            value = intent_dict.get(field, "").strip()

            # Handle Account Name/Number missing case
            if field in ["Account Name", "Account Number"]:
                if not value or value.lower() in ['', 'none', 'null', 'placeholder']:
                    params.append('"*"')
                    continue

            # Default to current month and year if From Period or To Period is missing
            if field in ["From Period", "To Period"] and not value:
                value = current_month_year

            if field in ["Subsidiary", "Budget category", "From Period", "To Period", "high/low", "Limit of record"]:
                params.append(f'"{value}"')
            elif field in ["Customer Number", "Customer Name", "Account Name",
                          "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
                params.append(f'"{value}"')
            else:
                params.append(f'"{value}"')

        return f'{formula_type}({", ".join(params)})'
    except Exception as e:
        print(f"Error in format_formula_with_intent: {e}")
        return ""

def validate_gpt_formula_output(gpt_formula: str) -> dict:
    """
    Validate and extract parameters from GPT formula output
    """
    if "SUITEREC" in gpt_formula.upper():
        table_set_match = re.search(r'SUITEREC\(\{(.*?)\}\)', gpt_formula, re.IGNORECASE)
        if table_set_match:
            table_names = table_set_match.group(1)
            return {"TABLE_NAME": table_names}
        return {}

    formula_match = re.search(r'([A-Z]+)\((.*?)\)', gpt_formula, re.IGNORECASE)
    if not formula_match:
        return {}

    formula_type = formula_match.group(1).upper()
    params_str = formula_match.group(2)

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

    intent_dict = {}
    template = formula_mapping.get(formula_type, [])

    for i, field in enumerate(template):
        if i < len(params):
            value = params[i].strip('"\'')
            if field in ["Subsidiary", "Budget category", "From Period", "To Period", "high/low", "Limit of record"]:
                intent_dict[field] = value
            else:
                intent_dict[field] = f'"{value}"'

    return intent_dict

def process_gpt_formula(gpt_formula: str, original_query: str = ""):
    """
    Process a GPT formula and return validated intent
    """
    formula_match = re.search(r'([A-Z]+)\(', gpt_formula, re.IGNORECASE)
    if not formula_match:
        return {"error": "Invalid formula format"}

    formula_type = formula_match.group(1).upper()
    intent_dict = validate_gpt_formula_output(gpt_formula)
    processed_intent = {}

    for field, value in intent_dict.items():
        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            processed_intent[field] = value
        else:
            clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip()
            processed_intent[field] = clean_val

    validation_result = validate_intent_fields_v2(processed_intent, original_query)
    validation_result["formula_type"] = formula_type

    return validation_result

def format_final_formula(validation_result):
    """
    Format the final formula based on validation result
    """
    formula_type = validation_result.get("formula_type")
    validated_intent = validation_result.get("validated_intent", {})
    original_intent = validation_result.get("original_intent", {})
    template = formula_mapping.get(formula_type, [])
    params = []

    for field in template:
        if field in original_intent and isinstance(original_intent[field], str):
            value = original_intent[field]
        else:
            value = validated_intent.get(field, "")

        if field == "Subsidiary":
            params.append('"Friday Media Group (Consolidated)"')
            continue

        value = value.replace('[', '{').replace(']', '}')
        value = value.replace('{', '').replace('}', '')

        if not value.strip():
            params.append("")
        else:
            params.append(f'"{value}"')

    return f'{formula_type}({", ".join(params)})'

def format_placeholder(value):
    """Ensures consistent placeholder formatting."""
    return value if isinstance(value, str) else str(value)

def best_partial_match(input_val, possible_vals, field_name=None):
    """Find the best partial match for a value using difflib.get_close_matches."""
    if not input_val or not possible_vals:
        return None

    input_val = input_val.strip().lower()
    # Check for exact match first
    for val in possible_vals:
        if input_val == val.lower():
            return val

    # Use difflib.get_close_matches for fuzzy matching
    matches = get_close_matches(input_val, possible_vals, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    # Handle travel aliases for Account Name
    if field_name in ["Account Name", "Account Number"]:
        travel_aliases = ["travel", "trav", "expense", "exp"]
        if input_val in travel_aliases:
            travel_matches = [val for val in possible_vals if "travel" in val.lower()]
            if travel_matches:
                return travel_matches[0]

    return None

def validate_intent_fields_v2(intent_dict, original_query=""):
    try:
        validated = {}
        notes = {}
        warnings = []
        placeholder_patterns = [r'\[.*?\]', r'\{.*?\}', r'^\{\".*?\"\}$']

        # Define period mapping
        current_date = datetime(2025, 5, 12)  # Fixed date for consistency
        period_mapping = {
            "current month": current_date.strftime("%B %Y"),  # May 2025
            "last month": (current_date - relativedelta(months=1)).strftime("%b %Y")  # April 2025
        }

        # Define strict placeholder values to preserve
        placeholder_values = [
            'subsidiary', 'classification', 'class', 'department', 'location',
            'currency', 'account number', 'account name',
            'customer number', 'customer name', 'vendor number', 'vendor name',
            'from period', 'to period', 'high/low', 'limit of record', 'table_name'
        ]

        # Handle other fields
        for key, value in intent_dict.items():
            is_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)
            clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()

            # Handle From Period and To Period
            if key in ["From Period", "To Period"]:
                if clean_val in period_mapping:
                    validated[key] = period_mapping[clean_val]
                    notes[key] = f"Mapped '{clean_val}' to '{validated[key]}'"
                else:
                    try:
                        from_val = intent_dict.get("From Period", "")
                        to_val = intent_dict.get("To Period", "")
                        from_val_clean = re.sub(r'[{}\"]', '', str(from_val)).strip()
                        to_val_clean = re.sub(r'[{}\"]', '', str(to_val)).strip()
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
                        validated[key] = clean_val or "Current month"
                        notes[key] = f"Could not normalize period: {str(e)}"
                        warnings.append(f"Period normalization error: {str(e)}")
                continue

            # Check if Account Name or Account Number is specified
            account_name_val = intent_dict.get("Account Name", "")
            account_number_val = intent_dict.get("Account Number", "")
            account_name_clean = re.sub(r'[{}"\'\[\]]', '', str(account_name_val)).strip().lower()
            account_number_clean = re.sub(r'[{}"\'\[\]]', '', str(account_number_val)).strip().lower()
            account_is_placeholder = (
                any(re.search(pattern, str(account_name_val)) for pattern in placeholder_patterns) or
                account_name_clean in ['account_name', 'account name', ''] or
                any(re.search(pattern, str(account_number_val)) for pattern in placeholder_patterns) or
                account_number_clean in ['account_number', 'account number', '']
            )

            if key == "Account Name" and account_is_placeholder and not account_number_clean:
                validated[key] = '"*"'
                notes[key] = "No account specified, using wildcard"
                continue

            if key == "Budget category" and (
                is_placeholder or clean_val in ['budget category', 'budget_category', 'category', 'bud', 'budget catgory']
            ):
                validated[key] = '"Standard Budget"'
                notes[key] = "Default to Standard Budget"
                continue

            if is_placeholder and (
                clean_val in placeholder_values or
                clean_val in [key.lower(), key.lower().replace(" ", "_"), key.lower() + "_name", key.lower() + "_number"]
            ):
                validated[key] = value
                notes[key] = "Generic placeholder preserved"
                continue

            if is_placeholder and key not in ["Account Name"]:
                validated[key] = format_placeholder(value)
                notes[key] = "Placeholder preserved"
                continue

            canonical_col = unified_columns.get(key, key).lower()
            possible_values = canonical_values.get(canonical_col, [])

            if key == "Account Number":
                canonical_col = 'account name'  # Always match Account Number against Account Name column
                possible_values = canonical_values.get(canonical_col, [])

            if key.lower() == "high/low":
                if original_query.lower().find("lowest") != -1:
                    validated[key] = "low"
                    notes[key] = "Set to low based on query"
                elif clean_val in ["high", "low"]:
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
                validated[key] = '"Standard Budget"' if key == "Budget category" else ""
                notes[key] = "Default to Standard Budget" if key == "Budget category" else "Empty or already invalid"
                continue

            # Try exact match first
            if clean_val in possible_values:
                mapped_col = column_mapping.get(canonical_col, canonical_col)
                if mapped_col in netsuite_df.columns:
                    matched = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                    if v.strip().lower() == clean_val), clean_val)
                    validated[key] = f'"{matched}"'
                    notes[key] = "Exact match"
                else:
                    validated[key] = f'"{clean_val}"'
                    notes[key] = "Exact match (no column mapping)"
            else:
                # Try fuzzy matching
                partial = best_partial_match(clean_val, possible_values, key)
                if partial:
                    mapped_col = column_mapping.get(canonical_col, canonical_col)
                    if mapped_col in netsuite_df.columns:
                        matched = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                        if v.strip().lower() == partial), partial)
                        validated[key] = f'"{matched}"'
                        notes[key] = "Partial match"
                    else:
                        validated[key] = f'"{partial}"'
                        notes[key] = "Partial match (no column mapping)"
                else:
                    if key == "Budget category":
                        validated[key] = '"Standard Budget"'
                        notes[key] = "Default to Standard Budget"
                    else:
                        validated[key] = value if is_placeholder else clean_val
                        notes[key] = "Placeholder with unmatched value preserved" if is_placeholder else "Not matched"
                        if not is_placeholder:
                            warnings.append(f"'{clean_val}' in {key} not recognized.")

        smart_input = validated.copy()
        for lk in ["Limit of record", "high/low"]:
            if lk in smart_input:
                smart_input[lk] = "LOCKED_VALUE"

        smart = smart_intent_correction_restricted(smart_input)
        smart_validated = smart["validated_intent"]

        for lk in ["Limit of record", "high/low"]:
            if lk in validated:
                smart_validated[lk] = validated.get(lk, "")

        final_validated = correct_validated_intent_with_fuzzy(smart_validated, canonical_values)

        return {
            "validated_intent": final_validated,
            "match_notes": notes,
            "warnings": warnings,
            "original_intent": intent_dict
        }
    except Exception as e:
        print(f"Error in validate_intent_fields_v2: {e}")
        return {"validated_intent": {}, "original_intent": intent_dict, "match_notes": {}, "warnings": []}

def correct_validated_intent_with_fuzzy(validated_intent: dict, canonical_values: dict) -> dict:
    """
    Apply fuzzy matching to correct values in validated intent against canonical values.
    """
    corrected_intent = {}
    field_thresholds = {
        "Subsidiary": 70,
        "Classification": 75,
        "Class": 75,
        "Department": 75,
        "Location": 80,
        "Budget category": 75,
        "Currency": 80,
        "Account Number": 80,
        "Account Name": 80,
        "Customer Number": 75,
        "Customer Name": 80,
        "Vendor Number": 80,
        "Vendor Name": 80
    }

    placeholder_values = [
        'subsidiary', 'classification', 'class', 'department', 'location',
        'currency', 'account number', 'account name',
        'customer number', 'customer name', 'vendor number', 'vendor name',
        'from period', 'to period', 'high/low', 'limit of record', 'table_name'
    ]

    key_fields = [
        "Location", "Account Name", "Budget category",
        "Customer Name", "Customer Number"
    ]

    for field, value in validated_intent.items():
        if field in ["From Period", "To Period"]:
            corrected_intent[field] = value
            continue

        if not value:
            if field == "Budget category":
                corrected_intent[field] = '"Standard Budget"'
                continue
            corrected_intent[field] = value
            continue

        raw = re.sub(r'^[{\["]*|[}\]"]*$', '', str(value)).lower()

        if field == "Budget category" and (
            raw in ['budget category', 'budget_category', 'category', 'bud', 'budget catgory'] or
            raw in [field.lower(), field.lower().replace(" ", "_")]
        ):
            corrected_intent[field] = '"Standard Budget"'
            continue

        if raw in placeholder_values or raw in [
            field.lower(), field.lower().replace(" ", "_"),
            field.lower() + "_name", field.lower() + "_number"
        ]:
            corrected_intent[field] = value
            continue

        if field == "Account Name" and value == '"*"':
            corrected_intent[field] = value
            continue

        canonical_col = unified_columns.get(field, field).lower()
        possible_values = canonical_values.get(canonical_col, [])
        if field == "Account Number":
            canonical_col = 'account name'  # Always match Account Number against Account Name column
            possible_values = canonical_values.get(canonical_col, [])

        if not possible_values:
            if field == "Budget category":
                corrected_intent[field] = '"Standard Budget"'
            else:
                corrected_intent[field] = value
            continue

        if raw in possible_values:
            mapped_col = column_mapping.get(canonical_col, canonical_col)
            if mapped_col in netsuite_df.columns:
                original_case = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                     if v.strip().lower() == raw), raw)
            else:
                original_case = raw
        else:
            match = best_partial_match(raw, possible_values, field)
            if match:
                mapped_col = column_mapping.get(canonical_col, canonical_col)
                if mapped_col in netsuite_df.columns:
                    original_case = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                         if v.strip().lower() == match), match)
                else:
                    original_case = match
            else:
                if field == "Budget category":
                    corrected_intent[field] = '"Standard Budget"'
                    continue
                if field in key_fields:
                    retry_match = best_partial_match(raw, possible_values, field)
                    if retry_match:
                        mapped_col = column_mapping.get(canonical_col, canonical_col)
                        if mapped_col in netsuite_df.columns:
                            original_case = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                                 if v.strip().lower() == retry_match), retry_match)
                        else:
                            original_case = retry_match
                    else:
                        corrected_intent[field] = value
                        continue
                else:
                    corrected_intent[field] = value
                    continue

        if field in ["Customer Number", "Customer Name", "Account Name",
                     "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
            formatted = f'{{"{original_case}"}}'
        elif field in ["Subsidiary", "Budget category", "high/low", "Limit of record", "TABLE_NAME"]:
            formatted = f'"{original_case}"'
        else:
            formatted = f'"{original_case}"'

        corrected_intent[field] = formatted

    return corrected_intent