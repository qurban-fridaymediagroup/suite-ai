import streamlit as st
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from formula_file.dictionary import normalisation_dict, formula_mapping
from formula_file.final_intent_validator_v2 import validate_intent_fields_v2
from rapidfuzz import process, fuzz
from datetime import datetime

# Load environment variables
load_dotenv()

# Get API key and model from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

def normalize_prompt(text, threshold=85):
    """
    Normalize user prompts by expanding abbreviations and standardizing business terms.
    This function:
    1. Expands common abbreviations (acc -> account, loc -> location)
    2. Normalizes business terms to their standard form using the normalization dictionary
    3. Uses fuzzy matching for misspelled terms
    4. Preserves common words, numbers, and other non-business terms
    """
    text = text.lower().strip()

    # Common abbreviations mapping - expand short forms to their full forms
    abbrev_mapping = {
        'acc': 'account',
        'acct': 'account',
        'accnt': 'account',
        'a / c': 'account',
        'a/c': 'account',
        'locati': 'location',
        'loc': 'location',
        'subs': 'subsidiary',
        'sub': 'subsidiary',
        'dept': 'department',
        'dep': 'department',
        'cat': 'budget category',
        'cls': 'classification',
        'class': 'classification',
        'cust': 'customer number',
        'vend': 'vendor name',
        'vendor': 'vendor name',
        'banglor': 'location Bangalore',  # Added to correct misspelling
        'friday-ad': 'subsidiary Friday Media Group (Consolidated)'  # Added for subsidiary
    }

    # Split text into words, preserving spaces and special characters
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    result = []
    dictionary_keys = list(normalisation_dict.keys())

    # Process words one by one
    i = 0
    while i < len(words):
        # Check for multi-word keys (like "budget category", "account number")
        found_multi_word = False
        for j in range(min(3, len(words) - i), 1, -1):  # Try phrases up to 3 words long
            phrase = ' '.join(words[i:i+j])
            # Skip normalization for specific phrases
            if phrase in ["budget variance", "variance analysis"]:
                result.append(phrase)
                i += j
                found_multi_word = True
                break
            if phrase in normalisation_dict:
                # Special case for "budget report" or similar phrases
                if "budget" in phrase and "category" not in phrase:
                    result.append("Budget")
                    # Add the remaining words separately
                    for k in range(1, j):
                        if words[i+k] in normalisation_dict:
                            result.append(normalisation_dict[words[i+k]])
                        else:
                            result.append(words[i+k])
                    i += j
                    found_multi_word = True
                    break
                result.append(normalisation_dict[phrase])
                i += j
                found_multi_word = True
                break

        if found_multi_word:
            continue

        # If no multi-word match, try single word
        word = words[i]
        # Check for abbreviations first
        if word in abbrev_mapping:
            result.append(abbrev_mapping[word])
        elif word in normalisation_dict:
            # Special case for "budget" - don't convert to "Budget_Category"
            if word.lower() == "budget":
                result.append("Budget")
            # Special case for "subsistence" - treat as account name
            elif word.lower() == "subsistence":
                result.append("Account_Name subsistence")
            else:
                result.append(normalisation_dict[word])
        else:
            # Only apply fuzzy matching to potential business terms
            # Skip common words, numbers, months, etc.
            common_words = {
                'get', 'show', 'me', 'need', 'for', 'in', 'and', 'the', 'of', 'to', 'from',
                'report', 'data', 'info', 'details', 'forecast', 'standard', 'office', 'supplies',
                'top'  # Added 'top' to preserve it
            }
            months = {
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september',
                'october', 'november', 'december'
            }

            if (word in common_words or word in months or word.isdigit() or
                len(word) <= 2 or any(c.isdigit() for c in word)):
                result.append(word)
            else:
                # Try fuzzy matching for potential business terms
                match, score, _ = process.extractOne(word, dictionary_keys, scorer=fuzz.WRatio)
                if score >= threshold:
                    # Special case for fuzzy matching "budget"
                    if match.lower() == "budget" or "budget" in match.lower():
                        result.append("Budget")
                    # Special case for fuzzy matching "subsistence"
                    elif match.lower() == "subsistence" or word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    # Special case for fuzzy matching "subsidiary"
                    elif match.lower() == "subsidiary" and word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    else:
                        result.append(normalisation_dict[match])
                else:
                    # Special case for "subsistence" when no match is found
                    if word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    else:
                        result.append(word)
        i += 1

    # Join words with single spaces and remove extra spaces
    normalized_text = ' '.join(result)
    # Remove spaces around underscores and hyphens
    normalized_text = re.sub(r'\s*([-_])\s*', r'\1', normalized_text)
    # Remove multiple spaces
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
    
    return normalized_text

def parse_formula_to_intent(formula_str: str):
    """
    Parse a formula string into its type and intent dictionary.
    Handles complex parameters like arrays in SUITEGENPIV and ensures correct argument counts.
    """
    match = re.match(r'(\w+)\((.*)\)', formula_str, re.DOTALL)
    if not match:
        return {"error": "Invalid formula format."}

    formula_type = match.group(1).upper()
    raw_params = match.group(2)

    params = []
    current_param = ""
    in_braces = 0
    in_brackets = 0
    in_quotes = False

    for char in raw_params:
        if char == '{' and not in_quotes:
            in_braces += 1
            current_param += char
        elif char == '}' and not in_quotes:
            in_braces -= 1
            current_param += char
        elif char == '[' and not in_quotes:
            in_brackets += 1
            current_param += char
        elif char == ']' and not in_quotes:
            in_brackets -= 1
            current_param += char
        elif char == '"' and not (in_braces or in_brackets):
            in_quotes = not in_quotes
            current_param += char
        elif char == ',' and not (in_braces or in_brackets or in_quotes):
            params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char

    if current_param.strip():
        params.append(current_param.strip())

    # Clean parameters
    cleaned_params = []
    for param in params:
        param = param.strip()
        if param.startswith('[') and param.endswith(']'):
            cleaned_params.append(param)
        else:
            cleaned_param = re.sub(r'^[\'"]|[\'"]$', '', param).strip()
            cleaned_params.append(cleaned_param)

    if formula_type not in formula_mapping:
        return {"error": f"Unknown formula type: {formula_type}", "formula": formula_str}

    expected_params = formula_mapping[formula_type]
    if len(cleaned_params) != len(expected_params):
        # Handle SUITEBUDREP missing Budget category
        if formula_type == "SUITEBUDREP" and len(cleaned_params) == len(expected_params) - 1:
            cleaned_params.insert(1, '"Standard Budget"')
        else:
            return {"error": "Mismatch in formula parameters.", "formula": formula_str}

    return {
        "formula_type": formula_type,
        "intent": dict(zip(expected_params, cleaned_params))
    }

def format_all_formula_mappings(mapping: dict) -> str:
    return "\n".join(
        [f"{name}({', '.join(params)})" for name, params in mapping.items()]
    )

def convert_date_range_to_period(date_range: str) -> str:
    """
    Convert a date range (e.g., 'Jan 2025 - Mar 2025') to a single period identifier (e.g., 'Q1 2025').
    Based on the current date (May 12, 2025).
    """
    date_range = date_range.strip().lower()
    # Map known periods
    period_mappings = {
        "current quarter": "Q2 2025",  # Apr 2025 - Jun 2025
        "last quarter": "Q1 2025",     # Jan 2025 - Mar 2025
        "ytd": "YTD 2025",             # Jan 2025 - May 2025
        "current month": "May 2025",
        "last month": "Apr 2025",
        "current year": "2025",
        "last year": "2024",
        "year to date": "YTD 2025"
    }
    if date_range in period_mappings:
        return period_mappings[date_range]

    # Handle date ranges like 'Jan 2025 - Mar 2025'
    range_match = re.match(r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4})', date_range)
    if range_match:
        start_date, end_date = range_match.groups()
        try:
            start = datetime.strptime(start_date, "%B %Y")
            end = datetime.strptime(end_date, "%B %Y")
            # Check if the range corresponds to a quarter
            if start.year == end.year and (end.month - start.month + 1) == 3:
                quarters = {
                    1: "Q1",  # Jan - Mar
                    4: "Q2",  # Apr - Jun
                    7: "Q3",  # Jul - Sep
                    10: "Q4"  # Oct - Dec
                }
                for start_month, qtr in quarters.items():
                    if start.month == start_month:
                        return f"{qtr} {start.year}"
            # Fallback to start month/year if not a quarter
            return start_date
        except ValueError:
            return date_range
    # Fallback to original if no match
    return date_range

def generate_formula_from_intent(formula_type: str, intent: dict, formula_mapping: dict, query: str = "", parsed_intent: dict = None) -> str:
    """
    Generate NetSuite formula using the validated intent, maintaining the order and formula type.
    Outputs '*' for placeholders or missing values in Account Name, Account Number, Vendor Name,
    Vendor Number, Customer Name, Customer Number; otherwise, outputs empty string for empty fields.
    For SUITEGENPIV, sets Grouping and Filtering to "" if no grouping fields are mentioned in the query.
    Converts date ranges to single period identifiers for From/To Period.
    Uses parsed_intent's Account Name if validated Account Name is '*', preserving 'subsistence'.
    """
    if formula_type not in formula_mapping:
        raise ValueError(f"Unknown formula type: {formula_type}")

    ordered_fields = formula_mapping[formula_type]
    formula_str = formula_type + "("

    # Handle SUITEREC specifically
    if formula_type == "SUITEREC":
        table_name = intent.get("TABLE_NAME", "").strip()
        clean_table_name = re.sub(r'[{}"\'\[\]]', '', table_name).strip()
        formula_str += f'"{clean_table_name}"' if clean_table_name else '""'
        formula_str += ")"
        return formula_str

    for i, field in enumerate(ordered_fields):
        value = intent.get(field, "").strip()

        # Define all fields to check for placeholders
        all_fields = [
            "Subsidiary", "Account Number", "Account Name", "Classification", "account_name",
            "Department", "Location", "Customer Number", "Customer Name",
            "Vendor Name", "Vendor Number", "Class", "high/low", "Limit of Records",
            "Budget category", "From Period", "To Period", "TABLE_NAME",
            "Grouping and Filtering"
        ]

        # Placeholder formats
        placeholder_formats = {
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
            "Limit of Records": '"Limit of record"',
            "TABLE_NAME": '"TABLE_NAME"',
            "Grouping and Filtering": '"Grouping and Filtering"'
        }

        # Fields that should output '*' when placeholder or missing
        asterisk_fields = [
            "Account Name", "Account Number", "Vendor Name", "Vendor Number", "account_name",
            "Customer Name", "Customer Number"
        ]

        # Normalize field variations for placeholder detection
        field_variations = [
            field.lower(),
            field.replace(' ', '_').lower(),
            field.replace('/', '_').lower(),
            field.lower().replace('limit of records', 'limit of record'),
            field.lower().replace('high/low', 'high_low'),
            field.lower().replace('grouping and filtering', 'grouping_and_filtering')
        ]

        # Additional values to treat as placeholders
        invalid_values = [
            'vendor_name', 'high_low', 'limit of record', 'limit_of_record',
            'account_number', 'account_name', 'customer_number',
            'customer_name', 'vendor_number', 'budget_category', 'classification',
            'department', 'location', 'subsidiary', 'class', 'limit of records',
            'from period', 'to period', 'from_period', 'to_period', 'vendor name',
            'customer number', 'customer name', 'vendor number', 'account number',
            'account name', 'budget category', 'budget', 'table_name', 'grouping and filtering',
            'grouping_and_filtering', '', 'none', 'null', 'placeholder'
        ]

        # Check if the value is a placeholder, empty, or invalid
        clean_value = re.sub(r'[{}"\'\[\]]', '', value).strip().lower()
        is_placeholder = (
            value == placeholder_formats.get(field, '') or
            bool(re.match(r'^{' + re.escape(field) + r'}$|^{' + re.escape(field.lower()) + r'}$', value)) or
            bool(re.match(r'^\[' + re.escape(field) + r'\]$|^\[' + re.escape(field.lower()) + r'\]$', value)) or
            bool(re.match(r'^{"' + re.escape(field) + r'"}$|^{"' + re.escape(field.lower()) + r'"}$', value)) or
            not value or
            clean_value in field_variations or
            clean_value in [v.lower() for v in all_fields] or
            clean_value in invalid_values or
            clean_value == field.lower() or
            value in [f'"{v}"' for v in invalid_values] or
            value in [f'"{v}"' for v in field_variations] or
            clean_value in [v.replace(' ', '_').lower() for v in all_fields]
        )

        # Handle Grouping and Filtering for SUITEGENPIV
        if field == "Grouping and Filtering":
            query_lower = query.lower()
            grouping_fields = []
            if 'class' in query_lower or 'classification' in query_lower or 'cls' in query_lower:
                grouping_fields.append('class')
            if 'department' in query_lower or 'dept' in query_lower or 'dep' in query_lower:
                grouping_fields.append('department')
            if 'location' in query_lower or 'loc' in query_lower or 'locati' in query_lower:
                grouping_fields.append('location')
            if 'vendor' in query_lower or 'vend' in query_lower:
                grouping_fields.append('vendor')
            if 'customer' in query_lower or 'cust' in query_lower:
                grouping_fields.append('customer')
            if grouping_fields and not is_placeholder:
                formula_str += f'[group_by:{",".join(grouping_fields)}]'
            else:
                formula_str += '""'
        # Handle Subsidiary
        elif field == "Subsidiary":
            if is_placeholder:
                formula_str += '"Friday Media Group (Consolidated)"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"Friday Media Group (Consolidated)"'
        # Handle From Period and To Period
        elif field in ["From Period", "To Period"]:
            if is_placeholder or (isinstance(value, str) and "current month" in value.lower()):
                current_month_year = datetime.now().strftime("%B %Y")
                formula_str += f'"{current_month_year}"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    # Convert date range to single period identifier
                    period_value = convert_date_range_to_period(value)
                    formula_str += f'"{period_value}"'
                else:
                    current_month_year = datetime.now().strftime("%B %Y")
                    formula_str += f'"{current_month_year}"'
        # Handle Account Name with fallback to parsed intent, preserving subsistence
        elif field == "Account Name" and parsed_intent and (value == '"*"' or clean_value in ['account_name', '']):
            # Fallback to parsed intent's Account Name if validated intent is '*' or placeholder
            parsed_value = parsed_intent.get("Account Name", "").strip()
            clean_parsed_value = re.sub(r'[{}"\'\[\]]', '', parsed_value).strip().lower()
            if clean_parsed_value and clean_parsed_value not in invalid_values:
                # Special case for subsistence
                if clean_parsed_value == "subsistence":
                    formula_str += '"subsistence"'
                else:
                    formula_str += f'"{clean_parsed_value}"'
            else:
                formula_str += '"*"'
        # Handle Department placeholder
        elif field == "Department" and value == "[department]":
            formula_str += '""'
        # Handle Location to ensure proper capitalization
        elif field == "Location":
            value = re.sub(r'[{}"\'\[\]]', '', value).strip()
            if value.lower() == 'bangalore':
                formula_str += '"Bangalore"'
            elif value:
                formula_str += f'"{value}"'
            else:
                formula_str += '""'
        # Handle other fields
        elif is_placeholder:
            if field in asterisk_fields:
                formula_str += '"*"'
            else:
                formula_str += '""'
        else:
            value = re.sub(r'[{}"\'\[\]]', '', value).strip()
            if value.startswith('[') and value.endswith(']'):
                value = value[1:-1]
                values = [v.strip().strip('"').strip("'") for v in value.split(',') if v.strip()]
                if values:
                    formula_str += f'[{", ".join(f'"{v}"' for v in values)}]'
                else:
                    if field in asterisk_fields:
                        formula_str += '"*"'
                    else:
                        formula_str += '""'
            else:
                # Handle Budget category
                if field == "Budget category":
                    if is_placeholder:
                        formula_str += '"Standard Budget"'
                    else:
                        value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                        if value:
                            formula_str += f'"{value}"'
                        else:
                            formula_str += '"Standard Budget"'
                elif value:
                    formula_str += f'"{value}"'
                elif field in asterisk_fields:
                    formula_str += '"*"'
                else:
                    formula_str += '""'

        # Add comma if not the last parameter
        if i < len(ordered_fields) - 1:
            formula_str += ", "

    formula_str += ")"
    return formula_str

# Initialize session state
if 'fine_tuned_model' not in st.session_state:
    st.session_state.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:hellofriday::BU8GWu9n"
if 'gpt_key' not in st.session_state:
    st.session_state.gpt_key = os.getenv("OPENAI_API_KEY", "")
if 'has_valid_api_key' not in st.session_state:
    st.session_state.has_valid_api_key = bool(st.session_state.gpt_key)
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = """
You are SuiteAI.
Instructions:
- Return one or more valid SuiteReport formulas depending on the user's intent:
  - For most prompts, return a single formula.
- For comparison or trend-analysis prompts, return **two SUITEGENPIV formulas** (one per time period), using identical structure except the From/To period fields.
    - This includes:
      - **Explicit comparison prompts** such as: "this year vs last year", "Q1 vs Q2", "compare", "versus", "vs"
      - **Implicit comparison prompts** such as: "Why did revenue grow this year?", "What changed in Q1?", "Why did expenses increase?", "Where did costs rise recently?"
- Output each formula on a new line. Do not explain or summarise the formula.
------------------------------------------
Supported formulas (strictly with required argument structure only - Never change or guess them):
- SUITEGEN: Strictly return exactly 7 arguments in exactly below order:
- SUITEGEN({Subsidiary}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
- SUITEGENREP: Strictly return exactly 7 arguments in exactly below order:
- SUITEGENREP({Subsidiary}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
- SUITECUS: Strictly return exactly 8 arguments in exactly below order:
- SUITECUS({Subsidiary}, [Customer], {From_Period}, {To_Period}, [Account], [Class], {High_Low}, {Limit_of_Record})
- SUITEVEN: Strictly return exactly 8 arguments in exactly below order:
- SUITEVEN({Subsidiary}, [Vendor], {From_Period}, {To_Period}, [Account], [Class], {High_Low}, {Limit_of_Record})
- SUITEBUD: Strictly return exactly 8 arguments in exactly below order:
- SUITEBUD({Subsidiary}, {Budget_Category}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
- SUITEBUDREP: Strictly return exactly 8 arguments in exactly below order:
- SUITEBUDREP({Subsidiary}, {Budget_Category}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
- SUITEVAR: Strictly return exactly 8 arguments in exactly below order:
- SUITEVAR({Subsidiary}, {Budget_Category}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
- SUITEREC: Strictly return exactly 1 argument in exactly below order:
- SUITEREC({EntityType})
- SUITEGENPIV: Strictly return exactly 5 arguments in exactly below order:
- SUITEGENPIV({Subsidiary}, [Account], {From_Period}, {To_Period}, [Grouping and Filtering])
❗ Always match formulas exactly as shown above. Never guess or create new argument patterns. Only use the exact structure and argument count listed.
Never ever invent a new formula, or change order, or reduce or increase fields. You must strictly follow the exact formula.
------------------------------------------
Strict Rules for (SUITEGEN, SUITEGENREP, SUITECUS, SUITEVEN, SUITEBUD, SUITEBUDREP, SUITEVAR, SUITEREC):
- Follow the required argument sequence and argument count for each formula.
- Use {} for dynamic single values, [] for dynamic multiple selections, and must put either curly or square bracket with "" just like {""} or [""] for fixed literal values.
- If a field is not provided, always leave a placeholder with its original name like {subsidiary}, [class], [department], [location] etc. Never mix or swap subsidiary, budget_category, account_name, vendor, customer, class, department, and location based on guesswork.
- If a prompt contains **multiple account names or customers, vendors, classes, departments, locations**, include them inside a single square-bracket array, only applicable to [account_name], [customer], [vendor], [class], [department], [location].
- Do not invent formulas. Only return those listed above in exactly same order and exactly same number of arguments.
- If returning a formula, return only the formula — no explanation, no commentary.
- Always return SUITEVAR (not SUITEBUD) when prompt implies actual vs budget comparison (e.g., "compare", "variance", "difference", "actual vs budget", "budget vs spend").
- {High_Low} and {Limit_of_Record} are only valid for SUITECUS and SUITEVEN, not for any other formula.
- If the prompt cannot map to a valid formula, never invent or fabricate a formula. Instead: Provide NetSuite guidance where appropriate — e.g., explain how the action can be completed using NetSuite’s core functionality.
- For SUITEBUD and SUITEBUDREP, if Budget_Category is not specified, use {"Standard Budget"} as the default.
- For Account Name, preserve exact values like ["subsistence"], ["revenue"] without mapping to placeholders unless explicitly unspecified.
------------------------------------------
Strict rules for only SUITEGENPIV:
- SUITEGENPIV strictly only include exactly five arguments:
  ({Subsidiary}, [Account], {From_Period}, {To_Period}, [Grouping and Filtering])
- The fifth and last argument must be a **single array** containing:
  - One "group_by:" instruction (e.g. "group_by:class") if grouping fields are specified
  - Otherwise, use an empty string: ""
- Only include fields in group_by that are explicitly mentioned in the query (e.g., class, department, location, vendor, customer).
- If no grouping fields are mentioned, output: ""
- For comparison or trend-analysis prompts, return **two SUITEGENPIV formulas** (one per time period), using identical structure except the From/To period fields.
  This includes:(only for SUITEGENPIV)
  - **Explicit comparison prompts**: "this year vs last year", "Q1 vs Q2", "compare", "versus", "vs"
  - **Implicit comparison prompts**, even if phrased passively, such as:
    - "Why did revenue grow this year?"
    - "What changed in Q1?"
    - "Why did expenses increase by customer?"
    - "How did overhead increase?"
- Follow the required argument sequence and argument count for SUITEGENPIV.
- Use {} for dynamic single values, [] for dynamic multiple selections, and "" for fixed literal values.
- If a field is not provided, always leave a placeholder name. Never makeup even if it does not make sense.
- For From Period and To Period, use single period identifiers (e.g., "Q1 2025", "Jan 2025") instead of date ranges (e.g., "Jan 2025 - Mar 2025").
SUITEGENPIV USES:
- Typical use cases include:
  - Identifying why revenue or spend changed over time
  - Comparing grouped totals across periods for trend analysis
  - Diagnosing unexpected changes in account-level performance
------------------------------------------
Supported formulas purpose:
- SUITEGEN: Fetch general ledger/account totals, spend, and balances.
- SUITEGENREP: Fetch general ledger/account transaction lists or summary reports.
- SUITECUS: Fetch customer transactions or invoices.
- SUITEVEN: Fetch vendor transactions or invoices.
- SUITEBUD: Fetch budgeted account totals, spend, and balances.
- SUITEBUDREP: Fetch budgeted account lists or budget detailed reports or budget list of transactions.
- SUITEVAR: Perform actual vs budget variance analysis.
- SUITEREC: Fetch master lists of records (e.g., customers, vendors, subsidiaries, accounts, classes, departments, employees, currencies, and budget categories).
- SUITEGENPIV: Fetch general ledger aggregated totals grouped by dimensions. Strictly only used for comparative, diagnostic, and exploratory analysis across accounts, departments, classes, vendors, customers, or locations. Must not replace it with either SUITEGEN OR SUITEGENREP.
------------------------------------------
⏳ TIME PERIOD INSTRUCTIONS:
- Always preserve the user's original time expressions exactly as stated.
- Never assume actual dates or modify time phrases like "current month" or "last year".
✅ Acceptable values (keep as-is):
  - {"current month"}
  - {"last month"}
  - {"current quarter"}
  - {"last quarter"}
  - {"current year"}
  - {"last year"}
  - {"year to date"}
  - {"ytd"}
🗓️ Only convert to date format (e.g. "Jan 2024") **if and only if** the user states the period explicitly in that format.
✅ Accept only this structure for dates:
  - Month followed by 4-digit year (e.g. "Feb 2024", "Sep 2023")
  - Do not reformat these. Use exactly as entered.
🚫 Do NOT guess or inject specific dates like:
  - "2024-05-01" or "March 1, 2024"
  - Do NOT use date ranges like "Jan 2025 - Mar 2025" unless explicitly provided by the user.
- For SUITEGENPIV, convert validated date ranges (e.g., "Jan 2025 - Mar 2025") to single period identifiers (e.g., "Q1 2025").
If unsure, keep the exact text (e.g. "current year") as the formula placeholder.
🔁 **If no time period is mentioned in the prompt**, keep default placeholders:
  - {From_Period} and {To_Period}
    """

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Netsuite Formula Generator Chat")

# API Key Check
if not st.session_state.has_valid_api_key:
    st.warning("**OpenAI API Key Required**: Please enter your OpenAI API key to use this application.", icon="⚠️")
    new_key = st.text_input("OpenAI API Key", value="", type="password",
                           help="Enter your OpenAI API key. This is required to use the application.")
    if new_key:
        st.session_state.gpt_key = new_key
        st.session_state.has_valid_api_key = True
        st.success("API key added successfully! You can now use the application.")
        st.rerun()

# Settings section
with st.expander("Settings"):
    new_model = st.text_input("Fine-tuned Model ID", value=st.session_state.fine_tuned_model)
    if new_model != st.session_state.fine_tuned_model:
        st.session_state.fine_tuned_model = new_model
        st.success("Model ID updated!")
    new_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=300)
    if new_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_prompt
        st.success("System prompt updated!")
    new_key = st.text_input("OpenAI API Key", value=st.session_state.gpt_key, type="password")
    if new_key != st.session_state.gpt_key:
        st.session_state.gpt_key = new_key
        st.session_state.has_valid_api_key = bool(new_key)
        st.success("API key updated!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if "normalized_query" in message:
                st.markdown("**Normalized Query:**")
                st.code(message["normalized_query"], language="text")
            if "formula" in message:
                st.markdown("**GPT Response:**")
                st.code(message["formula"], language="text")
            if "validated" in message:
                if isinstance(message["validated"], list):
                    for val in message["validated"]:
                        clean_val = {k: v for k, v in val.items() if k != "warnings"}
                        st.code(str(clean_val), language="text")
                else:
                    clean_val = {k: v for k, v in message["validated"].items() if k != "warnings"}
                    st.code(str(clean_val), language="text")

            if "regenerated_query" in message:
                st.markdown("**Regenerated Query:**")
                st.code(message["regenerated_query"], language="text")
                    
            if "regenerated_formula" in message:
                st.markdown("**Final Formula:**")
                if isinstance(message["regenerated_formula"], list):
                    for idx, formula in enumerate(message["regenerated_formula"], 1):
                        st.markdown(f"**Final Formula {idx}:**")
                        st.code(formula, language="text")
                else:
                    st.code(message["regenerated_formula"], language="text")
    st.divider()

# Chat input
query = st.chat_input("Ask about Netsuite formulas...", disabled=not st.session_state.has_valid_api_key)

if query and st.session_state.has_valid_api_key:
    normalized_query = normalize_prompt(query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    try:
        client = OpenAI(api_key=st.session_state.gpt_key)
        response = client.chat.completions.create(
            model=st.session_state.fine_tuned_model,
            messages=[
                {"role": "system", "content": st.session_state.system_prompt},
                {"role": "user", "content": normalized_query}
            ],
            temperature=0.7,
        )

        content = response.choices[0].message.content
        formula_text = content.strip() if content else ""
        formulas = [f.strip() for f in formula_text.split('\n') if f.strip()]
        parsed_results = []
        validated_results = []
        regenerated_formulas = []

        for formula in formulas:
            parsed = parse_formula_to_intent(formula)
            if "error" in parsed:
                parsed_results.append({"formula": formula, "error": parsed["error"]})
                validated_results.append({"error": parsed["error"]})
                regenerated_formulas.append(formula)
            else:
                validated = validate_intent_fields_v2(parsed["intent"], normalized_query)
                regenerated_formula = generate_formula_from_intent(
                    parsed["formula_type"], validated["validated_intent"], formula_mapping, normalized_query, parsed["intent"]
                )
                parsed_results.append(parsed)
                validated_results.append(validated)
                regenerated_formulas.append(regenerated_formula)

        if formulas:
            assistant_message = {
                "role": "assistant",
                "content": "Here's the formula(s) you requested:",
                "formula": formula_text,
                "normalized_query": normalized_query,
                "validated": validated_results,
                "regenerated_formula": regenerated_formulas
            }
            st.session_state.messages.append(assistant_message)
            with st.chat_message("assistant"):
                st.markdown("**Normalized Query:**")
                st.code(normalized_query, language="text")
                st.markdown("**GPT Response:**")
                st.code(formula_text, language="text")
                if any("error" in result for result in parsed_results):
                    st.markdown("**Note:** Some formulas could not be validated due to parsing errors.")
            for val, regen in zip(validated_results, regenerated_formulas):
                clean_val = {k: v for k, v in val.items() if k != "warnings"}
                st.markdown("**Validation Results:**")
                st.code(str(clean_val), language="text")
                # st.markdown("**Final Formula:**")
                st.code(regen, language="text")
                st.divider()

        else:
            st.error("No response generated")
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        # Display partial results if available
        assistant_message = {
            "role": "assistant",
            "content": f"Error occurred: {str(e)}. Partial results (if any):",
            "normalized_query": normalized_query,
            "formula": formula_text,
            "validated": validated_results,
            "regenerated_formula": regenerated_formulas
        }
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown("**Normalized Query:**")
            st.code(normalized_query, language="text")
            st.markdown("**GPT Response:**")
            st.code(formula_text, language="text")
            if validated_results:
                # For a single formula (no loop)
                st.subheader("Validation Results:")
                st.write(validated_intent)
                
                # And for the final formula
                st.subheader("Final Response:")
                st.code(final_formula, language="text")
            else:
                st.markdown("**Note:** No validation results available due to error.")

# Copy chat history button
if st.session_state.messages:
    if st.button("Copy Chat History"):
        chat_history = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}\n"
            f"Normalized Query: {msg.get('normalized_query', '')}\n"
            f"GPT Response: {msg.get('formula', '')}\n"
            f"Validation Results:"
            # f"Final Formula:"
            for msg in st.session_state.messages
        )
        st.toast("Chat history copied to clipboard!")
        st.code(chat_history, language="text")

# Clear chat button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
