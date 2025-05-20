import os
import re
from datetime import datetime
from difflib import get_close_matches

import pandas as pd
from dateutil.relativedelta import relativedelta

from formula_file.constants import unified_columns
# Import custom modules (assuming they don't rely on removed dependencies)
from formula_file.period_utils import get_period_range, validate_period_order
# from formula_file.smart_intent_correction import smart_intent_correction_restricted

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
expected_fields = ['Subsidiary', 'Classification', 'Department', 'Location', 'Budget category', 'Currency',
                   'Account Number', 'Account Name', 'Customer Number', 'Customer Name', 'Vendor Number', 'Vendor Name', 'Class']
for field in expected_fields:
    if field.lower() not in canonical_values:
        canonical_values[field.lower()] = set()

# Updated column variations with new aliases
column_variations = {'Subsidiary': ['Subsidiary', 'Sub', 'Subsidiaries', 'Subsidiary_Name'],
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
                     'Class': ['Class', 'Classes', 'Clas']}

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


def search_values(pattern, values):
    """Search the array using regex and return matching items"""
    regex = re.compile(pattern, re.IGNORECASE)
    return [item for item in values if regex.search(item)]

# Field placeholder formxatting rules
field_format_map = {"Subsidiary": "Subsidiary", "Budget category": '"Budget category"',
                    "Account Number": '{"Account Number"}', "Account Name": '{"Account Name"}', "From Period": '"From Period"',
                    "To Period": '"To Period"', "Classification": '{"Classification"}', "Department": '{"Department"}',
                    "Location": '{"Location"}', "Customer Number": '{"Customer Number"}', "Customer Name": '{"Customer Name"}',
                    "Vendor Name": '{"Vendor Name"}', "Vendor Number": '{"Vendor Number"}', "Class": '{"Class"}',
                    "high/low": '"high/low"', "Limit of record": '"Limit of record"', "TABLE_NAME": '"TABLE_NAME"'}

placeholder_keys = list(field_format_map.keys())

# Define formula mappings with correct parameter sequences
formula_mapping = {
    "SUITEGEN": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITECUS": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Name", "Class", "high/low",
                 "Limit of record"],
    "SUITEGENREP": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department",
                    "Location"], "SUITEREC": ["TABLE_NAME"],
    "SUITEBUD": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification",
                 "Department", "Location"],
    "SUITEBUDREP": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification",
                    "Department", "Location"],
    "SUITEVAR": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification",
                 "Department", "Location"],
    "SUITEVEN": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low",
                 "Limit of record"]}

# Extended formula mappings with variations
extended_formula_mapping = {
    "SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Customer Number", "From Period", "To Period",
                                              "Account Name", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME": ["Subsidiary", "Customer Name", "From Period", "To Period", "Account Name",
                                            "Class", "high/low", "Limit of record"],
    "SUITEGEN_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department",
                              "Location"],
    "SUITEVEN_VENDOR_NAME_ACCOUNT_NAME": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name",
                                          "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Vendor Number", "From Period", "To Period", "Account Name",
                                            "Class", "high/low", "Limit of record"],
    "SUITEBUD_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period",
                              "Classification", "Department", "Location"],
    "SUITEGENREP_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification",
                                 "Department", "Location"],
    "SUITEBUDREP_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period",
                                 "Classification", "Department", "Location"],
    "SUITEVAR_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period",
                              "Classification", "Department", "Location"]}



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
    """
    Find the best partial match for a value using wildcard-based matching (like SQL's '%word%').
    If multiple matches are found, use fuzzy matching to select the best one.
    For Account Name and Account Number, search in both columns.
    Returns the matched value in its original case from the NetSuite DataFrame.
    """
    if not input_val or not possible_vals:
        print("No match found: Empty input or no possible values")
        return None

    input_val = input_val.strip().lower()

    # For Account Name or Account Number, merge the search space
    if field_name in ["Account Name", "Account Number"]:
        # Get values from both account name and account number columns
        account_name_vals = canonical_values.get('account name', [])  # Updated to correct key
        account_number_vals = canonical_values.get('account number', [])  # Updated to correct key

        # Merge the values (use a set to avoid duplicates)
        merged_vals = list(set(account_name_vals + account_number_vals))

        # If we have merged values, use them instead of the provided possible_vals
        if merged_vals:
            possible_vals = merged_vals

    # Determine the DataFrame column to look up original case
    canonical_col = unified_columns.get(field_name, field_name).lower()
    mapped_col = column_mapping.get(canonical_col, canonical_col)

    # Check for exact match first
    for val in possible_vals:
        if input_val == val.lower():
            # Look up the original case in netsuite_df
            if mapped_col in netsuite_df.columns:
                matches = [v for v in netsuite_df[mapped_col].dropna().astype(str) if v.lower() == val]
                original_case = matches[0] if matches else val
            else:
                original_case = val
            print(f"Exact match found: {original_case}")
            return original_case

    # Use wildcard-based matching (like SQL's '%word%')
    # Find all values that contain the input value
    wildcard_matches = [val for val in possible_vals if input_val in val.lower()]
    print("\n wildcard_matches", wildcard_matches)
    if wildcard_matches:
        # If multiple matches, use fuzzy matching to find the best one
        if len(wildcard_matches) > 1:
            matches = get_close_matches(input_val, wildcard_matches, n=1, cutoff=0.6)
            print("\n", matches)
            if matches:
                # Look up the original case in netsuite_df
                if mapped_col in netsuite_df.columns:
                    matches_df = [v for v in netsuite_df[mapped_col].dropna().astype(str) if v.lower() == matches[0].lower()]
                    original_case = matches_df[0] if matches_df else matches[0]
                else:
                    original_case = matches[0]
                print(f"Partial match found: {original_case}")
                return original_case
        # If only one match, return it
        if mapped_col in netsuite_df.columns:
            matches_df = [v for v in netsuite_df[mapped_col].dropna().astype(str) if v.lower() == wildcard_matches[0].lower()]
            original_case = matches_df[0] if matches_df else wildcard_matches[0]
        else:
            original_case = wildcard_matches[0]
        print(f"Partial match found: {original_case}")
        return original_case

    # If no wildcard matches, try fuzzy matching as fallback
    matches = get_close_matches(input_val, possible_vals, n=1, cutoff=0.6)
    if matches:
        # Look up the original case in netsuite_df
        if mapped_col in netsuite_df.columns:
            matches_df = [v for v in netsuite_df[mapped_col].dropna().astype(str) if v.lower() == matches[0].lower()]
            original_case = matches_df[0] if matches_df else matches[0]
        else:
            original_case = matches[0]
        print(f"Fuzzy match found: {original_case}")
        return original_case

    print("No match found")
    return None


def validate_intent_fields_v2(intent_dict, original_query=""):
    try:
        validated = {}
        notes = {}
        warnings = []
        placeholder_patterns = [r'\[.*?\]', r'\{.*?\}', r'^\{\".*?\"\}$']

        # Define period mapping
        current_date = datetime(2025, 5, 12)  # Fixed date for consistency
        period_mapping = {"current month": current_date.strftime("%b %Y"),  # May 2025
                          "last month": (current_date - relativedelta(months=1)).strftime("%b %Y")  # April 2025
                          }

        # Define strict placeholder values to preserve
        placeholder_values = ['subsidiary', 'classification', 'class', 'department', 'location', 'currency',
                              'account number', 'account name', 'account', 'customer'
                                                                           'customer number', 'customer name', 'vendor number',
                              'vendor name', 'vendor', 'from period', 'to period', 'high/low', 'limit of record', 'table_name']

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
            account_is_placeholder = (any(re.search(pattern, str(account_name_val)) for pattern in
                                          placeholder_patterns) or account_name_clean in ['account_name',
                                                                                          'account name', ''] or any(
                re.search(pattern, str(account_number_val)) for pattern in
                placeholder_patterns) or account_number_clean in ['account_number', 'account number', ''])

            if key == "Account Name" and account_is_placeholder and not account_number_clean:
                validated[key] = '"*"'
                notes[key] = "No account specified, using wildcard"
                continue

            if key == "Account Name" or key == "Account Number":
                account_series = netsuite_df.get("Account", pd.Series(dtype='object'))
                acctnumber_series = netsuite_df.get("AcctNumber", pd.Series(dtype='object'))

                account_values = []
                if account_series is not None:
                    account_values.extend(account_series.to_list())
                if acctnumber_series is not None:
                    account_values.extend(acctnumber_series.to_list())
                
                account_values = [x for x in account_values if pd.notna(x)]
                # Clean and normalize the incoming value
                clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()
                if "*" in clean_val:
                    validated[key] = f'"*"' if clean_val.endswith("_name") else f'"{clean_val}"'
                    notes[key] = f"Wildcard match found: {clean_val}"
                    continue

                # Search for the value in the column values
                account_value = search_values(clean_val, account_values)
                account_value_length = len(account_value)

                # Check if the cleaned value exists in the dataset
                if account_value_length > 0:
                    validated[key] = account_value[0]
                    notes[key] = f"Exact match found in account values: {account_value[0]}"
                else:
                    validated[key] = '"*"'
                    notes[key] = "No match found, using wildcard"

                continue


            if key in unified_columns:
                # Get the target column from the unified mapping
                target_column = unified_columns[key]

                # Get the column series from netsuite_df, default to empty Series if None
                column_series = netsuite_df.get(target_column, pd.Series(dtype='object'))
                column_values = [] if column_series is None or len(column_series) == 0 else column_series.tolist()
                column_values = [x for x in column_values if pd.notna(x)]

                # Clean and normalize the incoming value
                clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()

                # Search for the value in the column values
                find_value = search_values(clean_val, column_values)
                find_value_length = len(find_value)

                # Check if the cleaned value exists in the dataset
                if find_value_length > 0:
                    actual_value = find_value[0]
                    validated[key] = f'"{actual_value}"'
                    notes[key] = f"Exact match found in {target_column} values: {actual_value}"
                else:
                    if key == "Subsidiary":
                        validated[key] = '"Friday Media Group (Consolidated)"'
                    elif key == "Budget category":
                        validated[key] = '"Standard Budget"'
                    else:
                        validated[key] = '""'
                    notes[key] = f"No match found in {target_column}, using wildcard"

                continue


            if is_placeholder and (
                    clean_val in placeholder_values or clean_val in [key.lower(), key.lower().replace(" ", "_"),
                                                                     key.lower() + "_name", key.lower() + "_number"]):
                validated[key] = value
                notes[key] = "Generic placeholder preserved"
                continue

            if is_placeholder and key not in ["Account Name"]:
                validated[key] = format_placeholder(value)
                notes[key] = "Placeholder preserved"
                continue


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

            

        return {"validated_intent": validated, "match_notes": notes, "warnings": warnings,
                "original_intent": intent_dict}
    except Exception as e:
        print(f"Error in validate_intent_fields_v2: {e}")
        return {"validated_intent": {}, "original_intent": intent_dict, "match_notes": {}, "warnings": []}
