import streamlit as st
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from formula_file.dictionary import normalisation_dict, formula_mapping
from formula_file.final_intent_validator_v2 import validate_intent_fields_v2
from rapidfuzz import process, fuzz
import json

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
                    result.append(normalisation_dict[match])
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

    formula_type = match.group(1).upper()  # ✅ Convert formula name to UPPERCASE here
    raw_params = match.group(2)
    params = [p.strip().strip('"').strip("'") for p in re.split(r',(?![^{}]*\})', raw_params)]

    if formula_type not in formula_mapping or len(params) != len(formula_mapping[formula_type]):
        return {"error": "Mismatch in formula parameters.", "formula": formula_str}

    return {
        "formula_type": formula_type,
        "intent": dict(zip(formula_mapping[formula_type], params))
    }

def format_all_formula_mappings(mapping: dict) -> str:
    return "\n".join(
        [f"{name}({', '.join(params)})" for name, params in mapping.items()]
    )

def generate_formula_from_intent(formula_type: str, intent: dict, formula_mapping: dict) -> str:
    """
    Generate NetSuite formula using the validated intent, maintaining the order and formula type.
    Outputs '*' for placeholders or missing values in Account Name, Account Number, Vendor Name,
    Vendor Number, Customer Name, Customer Number; otherwise, outputs a comma.
    Uses 'Friday Media Group (Consolidated)' for Subsidiary if placeholder or missing.
    Uses 'Current month' for From Period and To Period if not mentioned.
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
            "Subsidiary", "Account Number", "Account Name", "Classification",
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
            "Account Name", "Account Number", "Vendor Name", "Vendor Number",
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
            'account name', 'budget category', 'table_name', 'grouping and filtering',
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
            if is_placeholder:
                formula_str += '"Current month"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"Current month"'
        # Handle high/low specifically
        elif field == "high/low":
            if clean_value in ['high', 'low']:
                formula_str += f'"{clean_value}"'
            elif is_placeholder:
                formula_str += '"high/low"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"high/low"'
        # Handle Limit of record
        elif field == "Limit of record":
            if clean_value.isdigit():
                formula_str += f'"{clean_value}"'
            elif is_placeholder:
                formula_str += '"10"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value.isdigit():
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"10"'
        # Handle other fields
        elif is_placeholder:
            if field in asterisk_fields:
                formula_str += '"*"'
            else:
                pass  # Outputs just a comma
        else:
            value = re.sub(r'[{}"\'\[\]]', '', value).strip()
            if value.startswith('[') and value.endswith(']'):
                value = value[1:-1]
                values = [v.strip().strip('"').strip("'") for v in value.split(',') if v.strip()]
                if values:
                    formula_str += f'[{", ".join(f"\"{v}\"" for v in values)}]'
                else:
                    if field in asterisk_fields:
                        formula_str += '"*"'
                    else:
                        pass
            else:
                # Handle Budget_Category
                if field == "Budget_Category":
                    if is_placeholder:
                        formula_str += '"Standard Budget"'
                    else:
                        value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                        if value:
                            formula_str += f'"{value}"'
                        else:
                            formula_str += '"Standard Budget"'
                if value:
                    formula_str += f'"{value}"'
                elif field in asterisk_fields:
                    formula_str += '"*"'

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
    st.session_state.system_prompt = """You are a NetSuite formula classification AI that maps user questions to the correct formula type and identifies key labeled inputs from the query. Respond ONLY in the following JSON format: 

{ 
  "formula": "<formula_name>", 
  "inputs": { 
    "<input_key_1>": "<input_value_1>", 
    "<input_key_2>": "<input_value_2>", 
    ... 
  } 
} 

Here are the valid formula types and their descriptions: 
- suitecus: Customer Number/Name in Customer Transactional Data 
- suitegen: GL Account Number/Name in Actual GL Transaction Balance 
- suiteven: Vendor Number/Name in Vendor Transactional Data 
- suitebud: GL Account Number/Name in Budget Balance 
- suitegenrep: Actual transaction report 
- suitebudrep: Actual budget transaction report 
- suitevar: Actual vs Budget comparisons 
- suiterec: Record lookup (Customer, Vendor, Subsidiary, Class, AccountType, Location, Department, Employee, BudgetCategory, AccountNumber, accountsearchdisplayname etc) 

Only extract clearly stated input values. Examples of input labels include: "subsidiary", "department", "class", "account", "customer", "vendor", "start_date", "end_date", "record_type". 

Always respond with lowercase formula names and input keys. Do not include any explanations or text before or after the JSON."""

# Formula patterns for substitution
FORMULA_PATTERNS = { 
    "suitecus": 'SUITECUS("Subsidiary",{"Customer"},"From Period","To Period",{"Account"},{"Class"}, "high/low", "Limit of record")', 
    "suitegen": 'SUITEGEN("Subsidiary",{"Account"},"From Period","To Period",{"Class"},{"Department"},{"Location"})', 
    "suiterec": 'SUITEREC("TABLE_NAME")', 
    "suiteven": 'SUITEVEN("Subsidiary",{"Vendor"},"From Period","To Period",{"Account"},{"Class"}, "high/low", "Limit of record")', 
    "suitebud": 'SUITEBUD("Subsidiary","Budget category",{"Account"},"From Period","To Period",{"Class"},{"Department"},{"Location"})', 
    "suitegenrep": 'SUITEGENREP("Subsidiary",{"Account"},"From Period","To Period",{"Class"},{"Department"},{"Location"})', 
    "suitebudrep": 'SUITEBUDREP("Subsidiary","Budget category",{"Account"},"From Period","To Period",{"Class"},{"Department"},{"Location"})', 
    "suitevar": 'SUITEVAR("Subsidiary","Budget category",{"Account"},"From Period","To Period",{"Class"},{"Department"},{"Location"})' 
}

# Formula substitution function
def build_formula_string(formula_key, inputs):
    pattern = FORMULA_PATTERNS.get(formula_key)
    if not pattern:
        return "Unknown formula type"

    result = pattern

    replace_map = {
        "Subsidiary": inputs.get("subsidiary", "*"),
        "Customer": inputs.get("customer", "*"),
        "Vendor": inputs.get("vendor", "*"),
        "Account": inputs.get("account", "*"),
        "Class": inputs.get("class", "*"),
        "Department": inputs.get("department", "*"),
        "Location": inputs.get("location", "*"),
        "Budget category": inputs.get("budget category", "*"),
        "From Period": inputs.get("from period", inputs.get("start_date", inputs.get("period", "Start"))),
        "To Period": inputs.get("to period", inputs.get("end_date", inputs.get("period", "End"))),
        "TABLE_NAME": inputs.get("record_type", "list"),
    }

    for key, value in replace_map.items():
        result = result.replace(f'"{key}"', f'"{value}"')

    return result

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
        data = json.loads(content)
        raw_gpt_response = content  # Store the raw JSON response

        formula = build_formula_string(data["formula"], data["inputs"])
        # formula = content.strip() if content else ""
        parsed = parse_formula_to_intent(formula)

        # Check if there was an error in parsing
        if "error" in parsed:
            # If there's a mismatch in formula parameters, just display the GPT response
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here's the GPT response:",
                "formula": formula,
                "gpt_response": raw_gpt_response,  # Store the raw GPT response
                "normalized_query": normalized_query
            })
            with st.chat_message("assistant"):
                # Display normalized query
                st.markdown("**Normalized Query:**")
                st.code(normalized_query, language="text")

                # Display GPT Response (JSON)
                st.markdown("**GPT Response (JSON):**")
                st.code(raw_gpt_response, language="json")
                
                # Display Formula
                st.markdown("**Formula:**")
                st.code(formula, language="text")
        else:
            validated = validate_intent_fields_v2(parsed["intent"])
            regenerated_formula = generate_formula_from_intent(parsed["formula_type"], validated["validated_intent"], formula_mapping)

            if content:
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Here's the formula you requested:",
                    "formula": formula,
                    "gpt_response": raw_gpt_response,  # Store the raw GPT response
                    "normalized_query": normalized_query,
                    "intent_map": parsed,
                    "validated": validated,
                    "regenerated_formula": regenerated_formula
                })
                with st.chat_message("assistant"):
                    # Display normalized query
                    st.markdown("**Normalized Query:**")
                    st.code(normalized_query, language="text")

                    # Display GPT Response (JSON)
                    st.markdown("**GPT Response (JSON):**")
                    st.code(raw_gpt_response, language="json")
                    
                    # Display Formula
                    st.markdown("**Formula:**")
                    st.code(formula, language="text")

                    # Display Validation Results
                    st.markdown("**Validation Results:**")
                    st.code(validated, language="text")

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
            f"{msg['role'].upper()}: {msg['content']}\nNormalized Query: {msg.get('normalized_query', '')}\nGPT Response (JSON): {msg.get('gpt_response', '')}\nFormula: {msg.get('formula', '')}\nValidation Results: {msg.get('validated', '')}\nFinal Formula: {msg.get('regenerated_formula', '')}"
            for msg in st.session_state.messages
        )
        st.toast("Chat history copied to clipboard!")
        st.code(chat_history, language="text")

# Add clear chat button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()