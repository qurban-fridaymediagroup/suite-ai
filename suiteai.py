import streamlit as st
from openai import OpenAI
import os
import re  
from dotenv import load_dotenv
from formula_file.dictionary import normalisation_dict, formula_mapping
from formula_file.final_intent_validator_v2 import validate_intent_fields_v2
from rapidfuzz import process, fuzz

# Load environment variables
load_dotenv()

# OpenAI client will be initialized after API key validation

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

    # First, expand abbreviations
    words = text.split()
    for i in range(len(words)):
        if words[i] in abbrev_mapping:
            words[i] = abbrev_mapping[words[i]]

    # Reconstruct the text with expanded abbreviations
    expanded_text = ' '.join(words)

    # Now process the text for normalization
    words = expanded_text.split()
    dictionary_keys = list(normalisation_dict.keys())
    result = []

    # Process words one by one
    i = 0
    while i < len(words):
        # Check for multi-word keys (like "budget category", "account number")
        found_multi_word = False
        for j in range(min(3, len(words) - i), 1, -1):  # Try phrases up to 3 words long
            phrase = ' '.join(words[i:i+j])
            if phrase in normalisation_dict:
                result.append(normalisation_dict[phrase])
                i += j
                found_multi_word = True
                break

        if found_multi_word:
            continue

        # If no multi-word match, try single word
        word = words[i]
        if word in normalisation_dict:
            result.append(normalisation_dict[word])
        else:
            # Only apply fuzzy matching to potential business terms
            # Skip common words, numbers, months, etc.
            common_words = {'get', 'show', 'me', 'need', 'for', 'in', 'and', 'the', 'of', 'to', 'from',
                           'report', 'data', 'info', 'details', 'forecast', 'standard', 'office', 'supplies'}
            months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                     'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september',
                     'october', 'november', 'december'}

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

    return ' '.join(result)


def parse_formula_to_intent(formula_str: str):
    match = re.match(r'(\w+)\((.*)\)', formula_str)
    if not match:
        raise ValueError("Invalid formula format.")

    formula_type = match.group(1).upper()  # ✅ Convert formula name to UPPERCASE here
    raw_params = match.group(2)
    params = [p.strip().strip('"').strip("'") for p in re.split(r',(?![^{}]*\})', raw_params)]

    if formula_type not in formula_mapping or len(params) != len(formula_mapping[formula_type]):
        raise ValueError("Mismatch in formula parameters.")

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
    Handles correct formatting for each parameter type (quoted, braced, or both).
    If value is empty or a placeholder, replace with * for specific fields or keep as comma for others.
    """
    if formula_type not in formula_mapping:
        raise ValueError(f"Unknown formula type: {formula_type}")

    ordered_fields = formula_mapping[formula_type]
    ordered_values = []

    # Define placeholder patterns to detect
    placeholder_patterns = [
        r'\[.*?\]', r'\{.*?\}', r'^\{\'.*?\'\}$'
    ]

    # Fields that should use * when empty or containing generic placeholders
    asterisk_fields = ["Account Name", "Account Number", "Account", "Vendor Name", "Vendor Number", "Vendor"]

    for field in ordered_fields:
        value = intent.get(field, "").strip()
        
        # Handle specific default fields first
        if field == "Subsidiary":
            raw_val = value.strip()
            if raw_val in ("{Subsidiary}", "[Subsidiary]", "") or not raw_val:
                ordered_values.append('"Friday Media Group (Consolidated)"')
                continue
        
        # Check if field is an Account or Vendor field that should use * when empty or generic
        if field in asterisk_fields:
            # Check if it's empty or a generic placeholder
            is_generic_placeholder = re.search(r'\[(Account|Vendor|Account Number|Account Name|Vendor Number|Vendor Name)\]', value, re.IGNORECASE)
            if not value or is_generic_placeholder:
                ordered_values.append('"*"')
                continue
        
        # Handle Budget category default
        if field == "Budget category":
            raw_val = value.strip()
            if raw_val in ("{Budget category}", "[Budget category]", "") or not raw_val:
                ordered_values.append('"Standard Budget"')
                continue
        
        # Check if it's an empty placeholder or empty string for other fields
        is_empty_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)
        
        if is_empty_placeholder or not value:
            ordered_values.append("")
            continue
        
        # Convert any square brackets to curly brackets but preserve the content
        if isinstance(value, str) and ('[' in value or ']' in value):
            content_match = re.search(r'\[(.*?)\]', value)
            if content_match:
                content = content_match.group(1).strip()
                if content:
                    value = content  # Just use the content without any brackets
                else:
                    value = ""

        # Remove any extra brackets and quotes to clean the value
        clean_val = re.sub(r'[{}\"\'\[\]]', '', str(value)).strip()
        
        # Format all fields with quotes only, no curly brackets
        ordered_values.append(f'"{clean_val}"')

    # Join with commas, preserving empty values
    formula_params = ", ".join(ordered_values)
    return f'{formula_type}({formula_params})'




# Initialize session state for model ID, system prompt and GPT key
if 'fine_tuned_model' not in st.session_state:
    st.session_state.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:hellofriday::BU8GWu9n"
if 'gpt_key' not in st.session_state:
    st.session_state.gpt_key = os.getenv("OPENAI_API_KEY", "")
if 'has_valid_api_key' not in st.session_state:
    st.session_state.has_valid_api_key = bool(st.session_state.gpt_key)
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = """...

# Initialization and Streamlit components continue...


        You are SuiteAI.

        Instructions:
        - Return exactly one valid SuiteReport formula if the user's request matches a formula.
        - Supported formulas (strictly with required argument structure only):
        - SUITEGEN({Subsidiary}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
        - SUITEGENREP({Subsidiary}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
        - SUITECUS({Subsidiary}, [Customer], {From_Period}, {To_Period}, [Account], [Class], {High_Low}, {Limit_of_Record})
        - SUITEVEN({Subsidiary}, [Vendor], {From_Period}, {To_Period}, [Account], [Class], {High_Low}, {Limit_of_Record})
        - SUITEBUD({Subsidiary}, {Budget_Category}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
        - SUITEBUDREP({Subsidiary}, {Budget_Category}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
        - SUITEVAR({Subsidiary}, {Budget_Category}, [Account], {From_Period}, {To_Period}, [Class], [Department], [Location])
        - SUITEREC({EntityType})

        Formula Purposes:
        - SUITEGEN → Fetch general ledger/account aggregated totals, spend, and balances.
        - SUITEGENREP → Fetch detailed transaction lists or summary reports for general ledger/accounts.
        - SUITECUS → Fetch customer transactions or customer invoices.
        - SUITEVEN → Fetch vendor transactions or vendor invoices.
        - SUITEBUD → Fetch budgeted account totals, spend, and balances (month-wise).
        - SUITEBUDREP → Fetch detailed or summary reports for budgeted accounts.
        - SUITEVAR → Perform actual vs budget variance analysis.
        - SUITEREC → SUITEREC → Fetch master lists of entity records such as customers, vendors, subsidiaries, accounts, classes, departments, employees, currencies, and budget categories.

        - Strictly follow the required argument sequence and argument count for each formula. Do not add or remove arguments.
        - Use {} for dynamic single values, [] for dynamic multiple selections, and "" for fixed literal values.
        - If a field is optional and not provided, leave a placeholder.
        - If the prompt cannot map to a valid formula, politely guide the user without inventing a formula.
        - If returning a formula, output only the formula — do not include any explanation, summary, or extra text.
    """

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Netsuite Formula Generator Chat")

# API Key Check - Display prominent message if API key is missing
if not st.session_state.has_valid_api_key:
    st.warning("⚠️ **OpenAI API Key Required**: Please enter your OpenAI API key to use this application.", icon="⚠️")

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
        parsed = parse_formula_to_intent(formula)
        validated = validate_intent_fields_v2(parsed["intent"])
        regenerated_formula = generate_formula_from_intent(parsed["formula_type"], validated["validated_intent"], formula_mapping)



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
