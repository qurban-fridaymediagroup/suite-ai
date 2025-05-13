import streamlit as st
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from formula_file.dictionary import normalisation_dict, formula_mapping
from formula_file.final_intent_validator_v2 import validate_intent_fields_v2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

# Get API key and model from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

# Load embeddings (same as final_intent_validator_v2.py)
embeddings_path = r"C:\Users\abc\Downloads\suite-ai\formula_file\all_column_embeddings.pkl"
with open(embeddings_path, "rb") as f:
    column_embeddings = pickle.load(f)

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_prompt(text, threshold=0.75):
    """
    Normalize user prompts by expanding abbreviations and standardizing business terms.
    This function:
    1. Expands common abbreviations (acc -> account, loc -> location)
    2. Normalizes business terms to their standard form using semantic similarity
    3. Preserves common words, numbers, and other non-business terms
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
        'vendor': 'vendor name'
    }

    # Load normalization dictionary keys
    dictionary_keys = list(normalisation_dict.keys())
    # Check if embeddings for normalisation_dict exist, else generate
    dictionary_embeddings = column_embeddings.get('NormalisationDict', None)
    if dictionary_embeddings is None:
        dictionary_embeddings = sentence_model.encode(dictionary_keys, convert_to_tensor=False)

    # Split text into words, preserving spaces and special characters
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    result = []

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
            # Special case for "subsistence" - treat as account name, not subsidiary
            elif word.lower() == "subsistence":
                result.append("Account_Name subsistence")
            else:
                result.append(normalisation_dict[word])
        else:
            # Only apply semantic matching to potential business terms
            # Skip common words, numbers, months, etc.
            common_words = {
                'get', 'show', 'me', 'need', 'for', 'in', 'and', 'the', 'of', 'to', 'from',
                'report', 'data', 'info', 'details', 'forecast', 'standard', 'office', 'supplies',
                'top'
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
                # Semantic similarity matching
                query_embedding = sentence_model.encode([word], convert_to_tensor=False)
                similarities = cosine_similarity(query_embedding, dictionary_embeddings)[0]
                top_idx = np.argmax(similarities)
                top_similarity = similarities[top_idx]

                if top_similarity >= threshold:
                    match = dictionary_keys[top_idx]
                    # Special case for "budget"
                    if match.lower() == "budget" or "budget" in match.lower():
                        result.append("Budget")
                    # Special case for "subsistence"
                    elif match.lower() == "subsistence" or word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    # Special case for "subsidiary" vs "subsistence"
                    elif match.lower() == "subsidiary" and word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    else:
                        result.append(normalisation_dict[match])
                else:
                    # Special case for "subsistence" when no match
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
    match = re.match(r'(\w+)\((.*)\)', formula_str)
    if not match:
        return {"error": "Invalid formula format."}

    formula_type = match.group(1).upper()
    raw_params = match.group(2)
    params = [p.strip().strip('"').strip("'") for p in re.split(r',(?![^{}]*\})', raw_params)]

    if formula_type not in formula_mapping or len(params) != len(formula_mapping[formula_type]):
        return {"error": "Mismatch in formula parameters.", "formula": formula_str}
    
    # Create the intent dictionary
    intent_dict = dict(zip(formula_mapping[formula_type], params))
    
    # Check for [account_name] or [account_number] in the formula and mark it for asterisk
    has_account_name_placeholder = '[account_name]' in formula_str.lower() or '[account_number]' in formula_str.lower()
    
    return {
        "formula_type": formula_type,
        "intent": intent_dict,
        "has_account_name_placeholder": has_account_name_placeholder
    }

def format_all_formula_mappings(mapping: dict) -> str:
    return "\n".join(
        [f"{name}({', '.join(params)})" for name, params in mapping.items()]
    )

def generate_formula_from_intent(formula_type: str, intent: dict, formula_mapping: dict, has_account_name_placeholder: bool = False) -> str:
    """
    Generate NetSuite formula using the validated intent, maintaining the order and formula type.
    Outputs '*' for placeholders or missing values in Account Name, Account Number, Vendor Name, account_name,
    Vendor Number, Customer Name, Customer Number; outputs '*' for [account_name] in validation and final response.
    Hard-codes 'Accounts Receivable' for Account Name in the final formula.
    Uses 'Friday Media Group (Consolidated)' for Subsidiary if placeholder or missing.
    Uses 'Current month' for From Period and To Period if not mentioned.
    Use "*" for account_name in validation and final response if it present in GPT response.
    For SUITEREC, uses TABLE_NAME directly from validated intent with proper quoting.
    """
    if formula_type not in formula_mapping:
        raise ValueError(f"Unknown formula type: {formula_type}")

    ordered_fields = formula_mapping[formula_type]

    # Build the formula string
    formula_str = formula_type + "("

    # Handle SUITEREC specifically
    if formula_type == "SUITEREC":
        table_name = intent.get("TABLE_NAME", "").strip()
        # Remove any curly braces or quotes that might come from validated intent
        clean_table_name = re.sub(r'[{}"\'\[\]]', '', table_name).strip()
        if clean_table_name:
            formula_str += f'"{clean_table_name}"'
        else:
            formula_str += '""'  # Default to empty string if no valid table name
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

        # Placeholder formats from field_format_map
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
            "TABLE_NAME": '"TABLE_NAME"'
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

        # Handle Subsidiary
        if field == "Subsidiary":
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
                # Get current month and year
                current_month_year = datetime.now().strftime("%B %Y")
                formula_str += f'"{current_month_year}"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    # Get current month and year as default
                    current_month_year = datetime.now().strftime("%B %Y")
                    formula_str += f'"{current_month_year}"'
        # Handle Account Name with hard-coded value
        elif field == "Account Name":
            if has_account_name_placeholder or is_placeholder or clean_value == 'account_name':
                formula_str += '"*"'  # Output * for [account_name], [account_number], or account_name
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"Accounts Receivable"'  # Hard-code Accounts Receivable
        # Handle other fields
        elif is_placeholder:
            # Force asterisk for Account Name if [account_name] was in the original formula
            if field == "Account Name" and has_account_name_placeholder:
                formula_str += '"*"'  # Output * for validation
            elif field in asterisk_fields:
                formula_str += '"*"'
            else:
                # For non-asterisk placeholder fields, output empty string to ensure comma
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
    
# Initialize session state for model ID, system prompt and GPT key
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
- Return one or more valid SuiteReport formulas depending on the user's Generations of Programming Languages
intent:
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
- SUITEREC: Always return exactly 5 arguments in exactly below order:
- SUITEGENPIV({Subsidiary}, [Account], {From_Period}, {To_Period}, [Grouping and Filtering])
❗ Always match formulas exactly as shown above. Never guess or create new argument patterns. Only use the exact structure and argument count listed.
Never ever invent a new formula, or change order, or reduce or increase fields. You must strictly follow the exact formula.
------------------------------------------
Strict Rules for (SUITEGEN, SUITEGENREP, SUITECUS, SUITEVEN, SUITEBUD, SUITEBUDREP, SUITEVAR, SUITEREC):
- Follow the required argument sequence and argument count for each formula.
- Use {} for dynamic single values, [] for dynamic multiple selections, and must put either curly or square bracket with "" just like {""} or [""] for fixed literal values.
- If a field is not provided, always leave a placeholder with it's original name like {subsidiary}, [class], [department], [location] etc. Never mix or swap subsidiary, budget_category, account_name, vendor, customer, class, department, and location based on guesswork.
- If a prompt contains **multiple account names or customers, vendors, classes, departments, locations**, include them inside a single square-bracket array, only applicable to [account_name], [customer], [vendor], [class], [department], [location].
- Do not invent formulas. Only return those listed above in exactly same order and exactly same number of arguments.
- If returning a formula, return only the formula — no explanation, no commentary.
- Always return SUITEVAR (not SUITEBUD) when prompt implies actual vs budget comparison (e.g., "compare", "variance", "difference", "actual vs budget", "budget vs spend").
- {High_Low} and {Limit_of_Record} are only valid for SUITECUS and SUITEVEN, not for any other formula.
- If the prompt cannot map to a valid formula, never invent or fabricate a formula. Instead: Provide NetSuite guidance where appropriate — e.g., explain how the action can be completed using NetSuite’s core functionality.
- For [account_name] or [account_number], output [*] in formula.
------------------------------------------
Strict rules for only SUITEGENPIV:
- SUITEGENPIV strictly only include exactly five arguments:
  ({Subsidiary}, [Account], {From_Period}, {To_Period}, [Grouping/Filtering Array])
- The fifth and last argument must be a **single array** containing:
  - One "group_by:" instruction (e.g. "group_by:department_sales")
  - Optionally, one or more "filter_by:" instructions (e.g. filter_by:location="lahore")
- ✅ If the user does not specify grouping dimensions explicitly,
  **default to this structure**:
  [group_by:class,department,location,vendor,customer]
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
- SUITEGENPIV: Fetch general ledger aggregated totals grouped by dimensions.  Strictly only used for comparative,diagnostic, and exploratory analysis across accounts, departments, classes, vendors, customers, or locations. Must not replace it with either SUITEGEN OR SUITEGENREP.
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
  - Even if user says “this year” or “current month”, DO NOT convert them to date ranges.
If unsure, keep the exact text (e.g. "current year") as the formula placeholder.
🔁 **If no time period is mentioned in the prompt**, keep default placeholders:
  - {From_Period} and {To_Period}
    """

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Netsuite Formula Generator Chat")

# API Key Check - Display prominent message if API key is missing
if not st.session_state.has_valid_api_key:
    st.warning("**OpenAI API Key Required**: Please enter your OpenAI API key to use this application.", icon="⚠️")

    # API Key input outside of expander for visibility
    new_key = st.text_input("OpenAI API Key", value="", type="password",
                           help="Enter your OpenAI API key. This is required to use the application.")
    if new_key:
        st.session_state.gpt_key = new_key
        st.session_state.has_valid_api_key = True
        st.success("API key added successfully! You can now use the application.")
        st.rerun()

# Settings section with expander
with st.expander("Settings"):
    # Model ID input
    new_model = st.text_input("Fine-tuned Model ID", value=st.session_state.fine_tuned_model)
    if new_model != st.session_state.fine_tuned_model:
        st.session_state.fine_tuned_model = new_model
        st.success("Model ID updated!")

    # System prompt input
    new_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=300)
    if new_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_prompt
        st.success("System prompt updated!")

    # GPT key input (also in settings)
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
                st.markdown("**Validation Results:**")
                st.code(message["validated"], language="text")
            if "regenerated_formula" in message:
                st.markdown("**Final Formula:**")
                st.code(message["regenerated_formula"], language="text")
    st.divider()

# Chat input - disabled if no valid API key
query = st.chat_input("Ask about Netsuite formulas...", disabled=not st.session_state.has_valid_api_key)

if query and st.session_state.has_valid_api_key:
    # Normalize the query before processing
    normalized_query = normalize_prompt(query)

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Get AI response
    try:
        # Initialize OpenAI client with the current API key
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
        # Check if content exists before calling strip()
        formula = content.strip() if content else ""
        # Replace [account_name] or account_name with [*] in GPT response for SUITECUS
        if formula.startswith('SUITECUS'):
            formula = re.sub(r'\[account_name\]|account_name', '[*]', formula, flags=re.IGNORECASE)
        parsed = parse_formula_to_intent(formula)

        # Check if there was an error in parsing
        if "error" in parsed:
            # If there's a mismatch in formula parameters, just display the GPT response
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here's the GPT response:",
                "formula": formula,
                "normalized_query": normalized_query
            })
            with st.chat_message("assistant"):
                # Display normalized query
                st.markdown("**Normalized Query:**")
                st.code(normalized_query, language="text")

                # Display GPT Response
                st.markdown("**GPT Response:**")
                st.code(formula, language="text")
        else:
            validated = validate_intent_fields_v2(parsed["intent"])
            # Replace [account_name], account_name, or [*] with "*" in validated intent for Account Name
            if 'Account Name' in validated['validated_intent'] and (
                validated['validated_intent']['Account Name'].lower() in ['[account_name]', 'account_name', '[*]']
            ):
                validated['validated_intent']['Account Name'] = '"*"'
            regenerated_formula = generate_formula_from_intent(
                parsed["formula_type"], 
                validated["validated_intent"], 
                formula_mapping,
                has_account_name_placeholder=parsed["has_account_name_placeholder"]
            )

            if content:
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Here's the formula you requested:",
                    "formula": formula,
                    "normalized_query": normalized_query,
                    "intent_map": parsed,
                    "validated": validated,
                    "regenerated_formula": regenerated_formula
                })
                with st.chat_message("assistant"):
                    # Display normalized query
                    st.markdown("**Normalized Query:**")
                    st.code(normalized_query, language="text")

                    # Display GPT Response
                    st.markdown("**GPT Response:**")
                    st.code(formula, language="text")

                    # Display Validation Results
                    st.markdown("**Validation Results:**")
                    st.code(str(validated), language="text")

                    # Display Regenerated Formula
                    st.markdown("**Final Formula:**")
                    st.code(regenerated_formula, language="text")
            else:
                st.error("No response generated")
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Add copy chat history button
if st.session_state.messages:
    if st.button("Copy Chat History"):
        chat_history = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}\nNormalized Query: {msg.get('normalized_query', '')}\nGPT Response: {msg.get('formula', '')}\nValidation Results: {msg.get('validated', '')}\nFinal Formula: {msg.get('regenerated_formula', '')}"
            for msg in st.session_state.messages
        )
        st.toast("Chat history copied to clipboard!")
        st.code(chat_history, language="text")

# Add clear chat button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()