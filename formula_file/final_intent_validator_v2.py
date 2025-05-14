import pandas as pd
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime
import re
import calendar
import os
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
# Add this import at the top with the other Pinecone imports
import pinecone
# Load environment variables
# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "suiteai-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

# Initialize canonical values
canonical_values = {}
expected_fields = [
    'Subsidiary', 'Classification', 'Department', 'Location', 'Budget category',
    'Currency', 'Account Number', 'Account Name', 'Customer Number', 'Customer Name',
    'Vendor Number', 'Vendor Name', 'Class'
]
for field in expected_fields:
    canonical_values[field.lower()] = set()

# Updated column variations with new aliases
column_variations = {
    'Subsidiary': ['Subsidiary', 'Sub', 'Subsidiaries', 'Subsidiary_Name'],
    'Classification': ['Classification', 'Brand', 'Cost Center', 'Cost_Center', 'Class'],
    'Department': ['Department', 'Dept', 'Departments'],
    'Location': ['Location', 'Loc', 'Locations', 'location', 'loc'],
    'Budget category': ['Budget category', 'Budget_Category', 'Category', 'budget_category', 'bud', 'budget catgory'],
    'Currency': ['Currency', 'Currencies'],
    'Account Number': ['Account Number', 'Account_No', 'Acct_Number', 'Account_Number'],
    'Account Name': ['Account Name', 'Account_Name', 'Acct_Name', 'a/c', 'account_name'],
    'Customer Number': ['Customer Number', 'Customer_No', 'Cust_Number', 'Customer_Number'],
    'Customer Name': ['Customer Name', 'Customer_Name', 'Cust_Name'],
    'Vendor Number': ['Vendor Number', 'Vendor_No', 'Vend_Number', 'Vendor', 'Vendors'],
    'Vendor Name': ['Vendor Name', 'Vendor_Name', 'Vend_Name', 'Vendor', 'Vendors'],
    'Class': ['Class', 'Classes', 'Clas']
}

# Load data into Pinecone (assuming data is preprocessed and embedded)
def load_data_to_pinecone():
    # Simulated data loading (replace with actual embedding logic)
    for field in expected_fields:
        field_lower = field.lower()
        values = canonical_values.get(field_lower, set())
        for i, value in enumerate(values):
            vector = [0.1 * i] * 768  # Placeholder embedding
            index.upsert([(f"{field_lower}_{i}", vector, {"value": value, "field": field_lower})])

# Populate canonical values from Pinecone
def fetch_canonical_values():
    for field in expected_fields:
        field_lower = field.lower()
        results = index.query(vector=[0] * 768, top_k=1000, filter={"field": field_lower})
        canonical_values[field_lower].update([match.metadata["value"] for match in results.matches])

# Load initial data
load_data_to_pinecone()
fetch_canonical_values()

# Fuzzy match column names
column_mapping = {}
csv_columns = expected_fields  # Use expected fields as column names
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
        column_mapping[expected_col.lower()] = expected_col

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

def pinecone_search(input_val, possible_vals, field_name=None):
    """Perform similarity search using Pinecone."""
    if not input_val:
        return None
    
    input_val = input_val.strip().lower()
    field_thresholds = {
        "Subsidiary": 0.8,
        "Classification": 0.8,
        "Class": 0.8,
        "Department": 0.8,
        "Location": 0.8,
        "Budget category": 0.8,
        "Currency": 0.8,
        "Account Number": 0.85,
        "Account Name": 0.85,
        "Customer Number": 0.85,
        "Customer Name": 0.85,
        "Vendor Number": 0.85,
        "Vendor Name": 0.85
    }
    
    threshold = field_thresholds.get(field_name, 0.85)
    
    # Simulate embedding for query (replace with actual embedding model)
    query_vector = [0.1] * 768  # Placeholder embedding
    
    # Exact match check
    results = index.query(query_vector, top_k=1, filter={"field": field_name.lower(), "value": input_val})
    if results.matches and results.matches[0].score >= 0.95:
        return results.matches[0].metadata["value"]
    
    # Semantic search
    results = index.query(query_vector, top_k=1, filter={"field": field_name.lower()})
    if results.matches and results.matches[0].score >= threshold:
        return results.matches[0].metadata["value"]
    
    # Special handling for Account Name
    if field_name in ["Account Name", "Account Number"]:
        travel_aliases = ["travel", "trav", "expense", "exp"]
        if input_val in travel_aliases:
            results = index.query(query_vector, top_k=1, filter={"field": "account name", "value": {"$regex": "travel.*"}})
            if results.matches and results.matches[0].score >= 0.85:
                return results.matches[0].metadata["value"]
        if input_val == "subsistence":
            results = index.query(query_vector, top_k=1, filter={"field": "account name", "value": {"$regex": "subsistence.*"}})
            if results.matches:
                return results.matches[0].metadata["value"]
    
    # Special handling for Location
    if field_name == "Location" and input_val == "bangalore":
        results = index.query(query_vector, top_k=1, filter={"field": "location", "value": {"$regex": "bangalore.*"}})
        if results.matches:
            return results.matches[0].metadata["value"]
    
    return None

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
        match = pinecone_search(clean_val, canonical_values.get('account name', []), 'Account Name')
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
        
        has_vendor_number = has_valid_match("Vendor Number", vendor_number)
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

def format_formula_with_intent(formula_type, intent_dict):
    """
    Format a formula string based on formula type and intent dictionary
    """
    template = get_formula_template(formula_type, intent_dict)
    
    if formula_type == "SUITEREC":
        table_name = intent_dict.get("TABLE_NAME", "")
        return f'SUITEREC("{table_name}")'
    
    current_month_year = datetime.now().strftime("%B %Y")
    
    params = []
    for field in template:
        value = intent_dict.get(field, "").strip()
        
        if field in ["Account Name", "Account Number"]:
            if not value or value.lower() in ['', 'none', 'null', 'placeholder']:
                params.append('"*"')
                continue
        
        if field in ["From Period", "To Period"] and not value:
            value = current_month_year
        
        if field in ["Subsidiary", "Budget category", "From Period", "To Period", "high/low", "Limit of record"]:
            params.append(f'"{value}"')
        elif field in ["Customer Number", "Customer Name", "Account Name", 
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

def validate_intent_fields_v2(intent_dict, original_query=""):
    validated = {}
    notes = {}
    warnings = []
    placeholder_patterns = [r'\[.*?\]', r'\{.*?\}', r'^\{\".*?\"\}$']
    
    current_date = datetime(2025, 5, 12)
    period_mapping = {
        "current month": current_date.strftime("%B %Y"),
        "last month": (current_date - relativedelta(months=1)).strftime("%B %Y")
    }

    placeholder_values = [
        'subsidiary', 'classification', 'class', 'department', 'location',
        'currency', 'account number', 'account name',
        'customer number', 'customer name', 'vendor number', 'vendor name',
        'from period', 'to period', 'high/low', 'limit of record', 'table_name'
    ]
    
    for key, value in intent_dict.items():
        is_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)
        clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()
        
        if key in ["From Period", "To Period"]:
            if clean_val in period_mapping:
                validated[key] = f'"{period_mapping[clean_val]}"'
                notes[key] = f"Mapped '{clean_val}' to '{period_mapping[clean_val]}'"
            elif not clean_val or clean_val in ['from_period', 'to_period']:
                validated[key] = f'"{current_date.strftime("%B %Y")}"'
                notes[key] = f"Defaulted to current month: {current_date.strftime('%B %Y')}"
            else:
                try:
                    from_val = intent_dict.get("From Period", "")
                    to_val = intent_dict.get("To Period", "")
                    from_val_clean = re.sub(r'[{}\"]', '', str(from_val)).strip()
                    to_val_clean = re.sub(r'[{}\"]', '', str(to_val)).strip()
                    from_p, to_p = get_period_range(from_val_clean, to_val_clean or from_val_clean)
                    from_val_final, to_val_final = validate_period_order(from_p, to_p)
                    
                    if "Check: From > To" in to_val_final:
                        validated["From Period"] = f'"{from_p}"'
                        validated["To Period"] = f'"{to_p} invalid input"'
                        notes["From Period"] = validated["From Period"]
                        notes["To Period"] = validated["To Period"]
                        warnings.append("To Period is earlier than From Period.")
                    else:
                        validated["From Period"] = f'"{from_val_final}"'
                        validated["To Period"] = f'"{to_val_final}"'
                        notes["From Period"] = from_val_final
                        notes["To Period"] = to_val_final
                except Exception as e:
                    validated[key] = f'"{current_date.strftime("%B %Y")}"'
                    notes[key] = f"Could not normalize period: {str(e)}"
                    warnings.append(f"Period normalization error: {str(e)}")
            continue
        
        if key == "Department" and value == "[department]":
            validated[key] = '[department]'
            notes[key] = "Preserved [department] placeholder"
            continue
        
        if key == "Account Name":
            if clean_val == "subsistence":
                validated[key] = '"subsistence"'
                notes[key] = "Exact match for subsistence as Account Name"
                continue
            if is_placeholder or clean_val in ['account_name', 'account name', '']:
                validated[key] = '"*"'
                notes[key] = "No account specified, using wildcard"
                continue
        
        if key == "Budget category" and (is_placeholder or clean_val in ['budget category', 'budget_category', 'category', 'bud', 'budget catgory']):
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
        
        if key == "Location" and clean_val == "bangalore":
            validated[key] = '"Bangalore"'
            notes[key] = "Corrected to Bangalore"
            continue
        
        canonical_col = unified_columns.get(key, key).lower()
        possible_values = canonical_values.get(canonical_col, [])
        
        if key == "Account Number":
            canonical_col = 'account name'
            possible_values = canonical_values.get(canonical_col, [])
        
        if key.lower() == "high/low":
            if original_query.lower().find("lowest") != -1:
                validated[key] = '"low"'
                notes[key] = "Set to low based on query"
            elif clean_val in ["high", "low"]:
                validated[key] = f'"{clean_val}"'
                notes[key] = "Exact match"
            elif "high_low" in clean_val:
                validated[key] = '"high/low"'
                notes[key] = "Placeholder preserved"
            else:
                validated[key] = '"high"'
                notes[key] = "Default value used"
            continue
        
        if key.lower() == "limit of record":
            if clean_val.isdigit():
                validated[key] = f'"{clean_val}"'
                notes[key] = "Valid integer"
            elif "limit" in clean_val:
                validated[key] = '"Limit of record"'
                notes[key] = "Placeholder preserved"
            else:
                validated[key] = '"10"'
                notes[key] = "Default value used"
            continue
        
        if clean_val in ["", "-", "!", "not found"]:
            validated[key] = '"Standard Budget"' if key == "Budget category" else '""'
            notes[key] = "Default to Standard Budget" if key == "Budget category" else "Empty or already invalid"
            continue
        
        query_vector = [0.1] * 768
        exact_matches = index.query(query_vector, top_k=1, filter={"field": canonical_col, "value": clean_val})
        if exact_matches.matches and exact_matches.matches[0].score >= 0.95:
            matched = exact_matches.matches[0].metadata.get('value')
            validated[key] = f'"{matched}"'
            notes[key] = "Exact match"
        else:
            partial = pinecone_search(clean_val, possible_values, key)
            if partial:
                validated[key] = f'"{partial}"'
                notes[key] = "Partial match"
            else:
                found = False
                for other_field in expected_fields:
                    if other_field != key:
                        other_matches = index.query(query_vector, top_k=1, filter={"field": other_field.lower()})
                        if other_matches.matches and other_matches.matches[0].score >= 0.75:
                            matched = other_matches.matches[0].metadata.get('value')
                            validated[key] = f'"{matched}"'
                            notes[key] = f"Found in {other_field} column"
                            found = True
                            break
                
                if not found:
                    validated[key] = value if is_placeholder else f'"{clean_val}"'
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
        "Subsidiary": 0.6,
        "Classification": 0.6,
        "Class": 0.6,
        "Department": 0.6,
        "Location": 0.55,
        "Budget category": 0.55,
        "Currency": 0.8,
        "Account type": 0.6,
        "Account Number": 0.6,
        "Account Name": 0.6,
        "Customer Number": 0.75,
        "Customer Name": 0.6,
        "Vendor Number": 0.8,
        "Vendor Name": 0.65
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
            else:
                corrected_intent[field] = '""'
            continue
        
        raw = re.sub(r'^[{\["]*|[}\]"]*$', '', str(value)).lower()
        
        if value in ['[department]', '[class]', '[location]'] or raw in placeholder_values:
            corrected_intent[field] = value
            continue
        
        if field == "Budget category" and (
            raw in ['budget category', 'budget_category', 'category', 'bud', 'budget catgory'] or
            raw in [field.lower(), field.lower().replace(" ", "_")]
        ):
            corrected_intent[field] = '"Standard Budget"'
            continue
        
        if field == "Account Name" and value == '"*"':
            corrected_intent[field] = value
            continue
        
        if field == "Account Name" and raw == "subsistence":
            corrected_intent[field] = '"subsistence"'
            continue
        
        if field == "Location" and raw == "bangalore":
            corrected_intent[field] = '"Bangalore"'
            continue
            
        canonical_col = unified_columns.get(field, field).lower()
        possible_values = canonical_values.get(canonical_col, [])
        if field == "Account Number":
            canonical_col = 'account name'
            possible_values = canonical_values.get(canonical_col, [])
            
        if not possible_values:
            if field == "Budget category":
                corrected_intent[field] = '"Standard Budget"'
            else:
                corrected_intent[field] = value
            continue
            
        threshold = field_thresholds.get(field, 0.6)
        
        if raw in possible_values:
            original_case = raw
        else:
            match = pinecone_search(raw, possible_values, field)
            if match:
                original_case = match
            else:
                corrected_intent[field] = value
                continue
        
        if field in ["Customer Number", "Customer Name", "Account Name",
                     "Classification", "Department", "Location", "Vendor Name", "Vendor Number", "Class"]:
            formatted = f'"{original_case}"'
        elif field in ["Subsidiary", "Budget category", "high/low", "Limit of record", "TABLE_NAME"]:
            formatted = f'"{original_case}"'
        else:
            formatted = f'"{original_case}"'
            
        corrected_intent[field] = formatted
    
    return corrected_intent