import pandas as pd
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime
import re
import calendar
import os
from rapidfuzz import process, fuzz

# Import custom modules
from formula_file.period_utils import get_period_range, normalize_period_string, validate_period_order
from formula_file.smart_intent_correction import smart_intent_correction_restricted
from formula_file.constants import unified_columns

# Load NetSuite data
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "Netsuite info all final data.csv")
netsuite_df = pd.read_csv(csv_path, encoding="ISO-8859-1")

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
    'Budget category': ['Budget category', 'Budget_Category', 'Category', 'budget_category','bud','budget catgory'],
    'Currency': ['Currency', 'Currencies'],
    'Account Number': ['Account Number', 'Account_No', 'Acct_Number', 'Account_Number'],
    'Account Name': ['Account Name', 'Account_Name', 'Acct_Name'],
    'Customer Number': ['Customer Number', 'Customer_No', 'Cust_Number', 'Customer_Number'],
    'Customer Name': ['Customer Name', 'Customer_Name', 'Cust_Name'],
    'Vendor Number': ['Vendor Number', 'Vendor_No', 'Vend_Number', 'Vendor', 'Vendors'],
    'Vendor Name': ['Vendor Name', 'Vendor_Name', 'Vend_Name', 'Vendor', 'Vendors'],
    'Class': ['Class', 'Classes', 'Clas']
}

# Fuzzy match column names
column_mapping = {}
csv_columns = list(netsuite_df.columns)
for expected_col, variations in column_variations.items():
    best_match = None
    best_score = 0
    for variation in variations:
        match, score, _ = process.extractOne(variation.lower(), csv_columns, scorer=fuzz.WRatio)
        if score > best_score and score >= 80:
            best_match = match
            best_score = score
    if best_match:
        column_mapping[expected_col.lower()] = best_match
    else:
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
    "SUITEGEN": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITECUS": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEGENREP": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEREC": ["TABLE_NAME"],
    "SUITEBUD": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVEN": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"]
}

# Extended formula mappings with variations
extended_formula_mapping = {
    "SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NUMBER": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME": ["Subsidiary", "Customer Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NAME_ACCOUNT_NUMBER": ["Subsidiary", "Customer Name", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEGEN_ACCOUNT_NUMBER": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEGEN_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVEN_VENDOR_NAME_ACCOUNT_NAME": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NUMBER_ACCOUNT_NUMBER": ["Subsidiary", "Vendor Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NAME_ACCOUNT_NUMBER": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME": ["Subsidiary", "Vendor Number", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"],
    "SUITEBUD_ACCOUNT_NUMBER": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUD_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEGENREP_ACCOUNT_NUMBER": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEGENREP_ACCOUNT_NAME": ["Subsidiary", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP_ACCOUNT_NUMBER": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR_ACCOUNT_NUMBER": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR_ACCOUNT_NAME": ["Subsidiary", "Budget category", "Account Name", "From Period", "To Period", "Classification", "Department", "Location"]
}

def get_formula_template(formula_type, intent_dict):
    """
    Determine the correct formula template based on formula type and intent fields
    """
    if formula_type == "SUITEREC":
        return formula_mapping["SUITEREC"]
    
    def has_valid_match(field, value):
        if not value or not isinstance(value, str):
            return False
        clean_val = re.sub(r'[{}"\'\[\]]', '', value).strip().lower()
        if clean_val in ['none', '', 'null', 'placeholder']:
            return False
        possible_values = canonical_values.get(field.lower(), [])
        if not possible_values:
            return False
        match = best_partial_match(clean_val, possible_values, field)
        return match is not None

    if formula_type == "SUITECUS":
        customer_number = intent_dict.get("Customer Number", "")
        customer_name = intent_dict.get("Customer Name", "")
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_customer_number = has_valid_match("Customer Number", customer_number)
        has_customer_name = has_valid_match("Customer Name", customer_name)
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_customer_number and has_account_number:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NUMBER"]
        elif has_customer_name and has_account_name:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NAME_ACCOUNT_NAME"]
        elif has_customer_name and has_account_number:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NAME_ACCOUNT_NUMBER"]
        elif has_customer_number and has_account_name:
            return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NAME"]
        return extended_formula_mapping["SUITECUS_CUSTOMER_NUMBER_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEGEN":
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_account_name:
            return extended_formula_mapping["SUITEGEN_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEGEN_ACCOUNT_NUMBER"]
        return extended_formula_mapping["SUITEGEN_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEVEN":
        vendor_number = intent_dict.get("Vendor Number", "")
        vendor_name = intent_dict.get("Vendor Name", "")
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_vendor_number = has_valid_match("Vendor Number", vendor_number)
        has_vendor_name = has_valid_match("Vendor Name", vendor_name)
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_vendor_name and has_account_name:
            return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NAME"]
        elif has_vendor_number and has_account_number:
            return extended_formula_mapping["SUITEVEN_VENDOR_NUMBER_ACCOUNT_NUMBER"]
        elif has_vendor_name and has_account_number:
            return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NUMBER"]
        elif has_vendor_number and has_account_name:
            return extended_formula_mapping["SUITEVEN_VENDOR_NUMBER_ACCOUNT_NAME"]
        return extended_formula_mapping["SUITEVEN_VENDOR_NAME_ACCOUNT_NAME"]
    
    elif formula_type == "SUITEBUD":
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_account_name:
            return extended_formula_mapping["SUITEBUD_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEBUD_ACCOUNT_NUMBER"]
        return extended_formula_mapping["SUITEBUD_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEGENREP":
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_account_name:
            return extended_formula_mapping["SUITEGENREP_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEGENREP_ACCOUNT_NUMBER"]
        return extended_formula_mapping["SUITEGENREP_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEBUDREP":
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_account_name:
            return extended_formula_mapping["SUITEBUDREP_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEBUDREP_ACCOUNT_NUMBER"]
        return extended_formula_mapping["SUITEBUDREP_ACCOUNT_NUMBER"]
    
    elif formula_type == "SUITEVAR":
        account_number = intent_dict.get("Account Number", "")
        account_name = intent_dict.get("Account Name", "")
        
        has_account_number = has_valid_match("Account Number", account_number)
        has_account_name = has_valid_match("Account Name", account_name)
        
        if has_account_name:
            return extended_formula_mapping["SUITEVAR_ACCOUNT_NAME"]
        elif has_account_number:
            return extended_formula_mapping["SUITEVAR_ACCOUNT_NUMBER"]
        return extended_formula_mapping["SUITEVAR_ACCOUNT_NUMBER"]
    
    return formula_mapping.get(formula_type, [])

def format_formula_with_intent(formula_type, intent_dict):
    """
    Format a formula string based on formula type and intent dictionary
    """
    template = get_formula_template(formula_type, intent_dict)
    
    if formula_type == "SUITEREC":
        table_name = intent_dict.get("TABLE_NAME", "")
        return f'SUITEREC("{table_name}")'
    
    params = []
    for field in template:
        value = intent_dict.get(field, "")
        
        if field in ["Subsidiary", "Budget category", "From Period", "To Period", "high/low", "Limit of record"]:
            params.append(f'"{value}"')
        elif field in ["Customer Number", "Customer Name", "Account Number", "Account Name", 
                       "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
            params.append(f'{{"{value}"}}')
        else:
            params.append(f'"{value}"')
    
    return f'{formula_type}({", ".join(params)})'

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

def process_gpt_formula(gpt_formula: str):
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
    
    validation_result = validate_intent_fields_v2(processed_intent)
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

# Utility functions
def best_partial_match(input_val, possible_vals, field_name=None):
    """Find the best partial match for a value using field-specific thresholds."""
    if not input_val or not possible_vals:
        return None
    
    input_val = input_val.strip().lower()
    
    # Field-specific thresholds
    field_thresholds = {
        "Subsidiary": 60,
        "Classification": 60,
        "Class": 60,
        "Department": 60,
        "Location": 70,
        "Budget category": 60,
        "Currency": 80,
        "Account Number": 75,
        "Account Name": 60,
        "Customer Number": 75,
        "Customer Name": 60,
        "Vendor Number": 80,
        "Vendor Name": 65
    }
    
    threshold = field_thresholds.get(field_name, 60)
    
    # First try exact match
    for val in possible_vals:
        if input_val == val.lower():
            return val
    
    # Enhanced substring match with word overlap and partial ratio for key fields
    if field_name in ["Location", "Account Name", "Account Number", "Budget category", "Customer Number", "Customer Name"]:
        for val in possible_vals:
            val_lower = val.lower()
            input_words = input_val.split()
            val_words = val_lower.split()
            # Check substring, first three chars, and word overlap
            if (input_val in val_lower or val_lower in input_val or 
                input_val[:3] in val_lower or val_lower[:3] in input_val or
                any(word in val_words for word in input_words if len(word) > 2)):
                return val
    
    # General substring match
    for val in possible_vals:
        val_lower = val.lower()
        if input_val in val_lower or val_lower in input_val:
            return val
    
    # Use partial_ratio for key fields to catch typos
    if field_name in ["Location", "Account Name", "Budget category", "Customer Name", "Customer Number"]:
        match, score, _ = process.extractOne(input_val, possible_vals, scorer=fuzz.partial_ratio)
        if score >= threshold - 5:  # Slightly lower threshold for partial_ratio
            return match
    
    # Fuzzy matching with rapidfuzz WRatio
    match, score, _ = process.extractOne(input_val, possible_vals, scorer=fuzz.WRatio)
    if score >= threshold:
        return match
    
    return None

# Main validation function
def validate_intent_fields_v2(intent_dict):
    validated = {}
    notes = {}
    warnings = []
    placeholder_patterns = [r'\[.*?\]', r'\{.*?\}', r'^\{\".*?\"\}$']
    
    # Define strict placeholder values to preserve
    placeholder_values = [
        'subsidiary', 'classification', 'class', 'department', 'location',
        'budget category', 'currency', 'account number', 'account name',
        'customer number', 'customer name', 'vendor number', 'vendor name',
        'from period', 'to period', 'high/low', 'limit of record', 'table_name'
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
    
    # Ensure matching for all formula fields
    key_fields = [
        "Location", "Account Name", "Account Number", "Budget category",
        "Customer Name", "Customer Number"
    ]
    
    # Handle other fields
    for key, value in intent_dict.items():
        if key in ["From Period", "To Period"]:
            continue
        
        is_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)
        clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()
        
        # Strictly preserve placeholders that match field names or known placeholder values
        if is_placeholder and (
            clean_val in placeholder_values or
            clean_val in [key.lower(), key.lower().replace(" ", "_"), key.lower() + "_name", key.lower() + "_number"]
        ):
            validated[key] = value
            notes[key] = "Generic placeholder preserved"
            continue
        
        if is_placeholder:
            validated[key] = format_placeholder(value)
            notes[key] = "Placeholder preserved"
            continue
        
        canonical_col = unified_columns.get(key, key).lower()
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
            validated[key] = '"Standard Budget"' if key == "Budget category" else ""
            notes[key] = "Default to Standard Budget" if key == "Budget category" else "Empty or already invalid"  # FIXED: Corrected typo and indentation
            continue
        
        if possible_values and clean_val in possible_values:
            mapped_col = column_mapping.get(canonical_col, canonical_col)
            if mapped_col in netsuite_df.columns:
                matched = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                if v.strip().lower() == clean_val), clean_val)
                validated[key] = matched
                notes[key] = "Exact match"
            else:
                validated[key] = clean_val
                notes[key] = "Exact match (no column mapping)"
        else:
            partial = best_partial_match(clean_val, possible_values, key)
            if partial:
                mapped_col = column_mapping.get(canonical_col, canonical_col)
                if mapped_col in netsuite_df.columns:
                    matched = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                    if v.strip().lower() == partial.strip().lower()), partial)
                    validated[key] = matched
                    notes[key] = "Partial match" if matched.strip().lower() != clean_val else "Exact match"
                else:
                    validated[key] = partial
                    notes[key] = "Partial match (no column mapping)"
            else:
                found = False
                for other_col, other_vals in canonical_values.items():
                    if other_col != canonical_col:
                        if clean_val in other_vals:
                            validated[key] = clean_val
                            notes[key] = f"Found in {other_col} column"
                            found = True
                            break
                        partial_other = best_partial_match(clean_val, other_vals, key)
                        if partial_other:
                            mapped_col = column_mapping.get(other_col, other_col)
                            if mapped_col in netsuite_df.columns:
                                matched = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                                if v.strip().lower() == partial_other.strip().lower()), partial_other)
                                validated[key] = matched
                                notes[key] = f"Partial match in {other_col} column"
                                found = True
                                break
                if not found:
                    if key in key_fields:
                        retry_match = best_partial_match(clean_val, possible_values, key)
                        if retry_match:
                            mapped_col = column_mapping.get(canonical_col, canonical_col)
                            if mapped_col in netsuite_df.columns:
                                matched = next((v for v in netsuite_df[mapped_col].dropna().astype(str)
                                                if v.strip().lower() == retry_match.strip().lower()), retry_match)
                                validated[key] = matched
                                notes[key] = "Retry partial match"
                            else:
                                validated[key] = retry_match
                                notes[key] = "Retry partial match (no column mapping)"
                        else:
                            if key == "Budget category":
                                validated[key] = '"Standard Budget"'
                                notes[key] = "Default to Standard Budget"
                            else:
                                validated[key] = value if is_placeholder else clean_val
                                notes[key] = "Placeholder with unmatched value preserved" if is_placeholder else "Not matched in any known column"
                                if not is_placeholder:
                                    warnings.append(f"'{clean_val}' in {key} not recognized in any column.")
                    else:
                        if key == "Budget category":
                            validated[key] = '"Standard Budget"'
                            notes[key] = "Default to Standard Budget"
                        else:
                            validated[key] = value if is_placeholder else clean_val
                            notes[key] = "Placeholder with unmatched value preserved" if is_placeholder else "Not matched in any known column"
                            if not is_placeholder:
                                warnings.append(f"'{clean_val}' in {key} not recognized in any column.")
    
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

def correct_validated_intent_with_fuzzy(validated_intent: dict, canonical_values: dict) -> dict:
    """
    Apply fuzzy matching to correct values in validated intent against canonical values.
    """
    corrected_intent = {}
    field_thresholds = {
        "Subsidiary": 60,
        "Classification": 60,
        "Class": 60,
        "Department": 60,
        "Location": 55,
        "Budget category": 55,
        "Currency": 80,
        "Account type": 60,
        "Account Number": 75,
        "Account_Name": 60,
        "Account Name": 60,
        "Account_Number": 75,
        "Customer Number": 75,
        "Customer Name": 60,
        "Vendor Number": 80,
        "Vendor Name": 65
    }
    
    placeholder_values = [
        'subsidiary', 'classification', 'class', 'department', 'location',
        'budget category', 'currency', 'account number', 'account name',
        'customer number', 'customer name', 'vendor number', 'vendor name',
        'from period', 'to period', 'high/low', 'limit of record', 'table_name'
    ]
    
    key_fields = [
        "Location", "Account Name", "Account Number", "Budget category",
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
        
        # Skip fuzzy matching for strict placeholders
        if raw in placeholder_values or raw in [
            field.lower(), field.lower().replace(" ", "_"),
            field.lower() + "_name", field.lower() + "_number"
        ]:
            corrected_intent[field] = value
            continue
            
        canonical_col = unified_columns.get(field, field).lower()
        possible_values = canonical_values.get(canonical_col, [])
        if not possible_values:
            if field == "Budget category":
                corrected_intent[field] = '"Standard Budget"'
            else:
                corrected_intent[field] = value
            continue
            
        threshold = field_thresholds.get(field, 60)
        
        # Try exact match first
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
        
        # Format based on field type
        if field in ["Customer Number", "Customer Name", "Account Number", "Account Name",
                     "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
            formatted = f'{{"{original_case}"}}'
        elif field in ["Subsidiary", "Budget category", "high/low", "Limit of record", "TABLE_NAME"]:
            formatted = f'"{original_case}"'
        else:
            formatted = f'"{original_case}"'
            
        corrected_intent[field] = formatted
    
    return corrected_intent