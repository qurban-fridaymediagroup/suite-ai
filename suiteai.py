import streamlit as st
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from formula_file.dictionary import normalisation_dict, formula_mapping
from formula_file.final_intent_validator_v2 import validate_intent_fields_v2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
# Load environment variables
load_dotenv()
# Get API key and model from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "suiteai-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Fetch normalization dictionary from Pinecone
def fetch_normalization_dict():
    dictionary_keys = []
    dictionary_embeddings = []
    # Change this line
    results = index.query(vector=[0] * 1536, top_k=1000, filter={"field": {"$in": list(normalisation_dict.keys())}})  # Changed from 768 to 1536
    for match in results.matches:
        value = match.metadata.get("value")
        if value:
            dictionary_keys.append(value)
            # Generate embedding for the value (since Pinecone stores embeddings, we re-encode for consistency)
            embedding = sentence_model.encode([value], convert_to_tensor=False)[0]
            dictionary_embeddings.append(embedding)
    return dictionary_keys, np.array(dictionary_embeddings)

# Load normalization dictionary and embeddings
try:
    dictionary_keys, dictionary_embeddings = fetch_normalization_dict()
except Exception as e:
    st.error(f"Failed to fetch normalization dictionary from Pinecone: {str(e)}")
    dictionary_keys = list(normalisation_dict.keys())
    dictionary_embeddings = sentence_model.encode(dictionary_keys, convert_to_tensor=False)

def normalize_prompt(text, threshold=0.75):
    """
    Normalize user prompts by expanding abbreviations and standardizing business terms.
    This function:
    1. Expands common abbreviations (acc -> account, loc -> location)
    2. Normalizes business terms to their standard form using semantic similarity
    3. Preserves common words, numbers, and other non-business terms
    """
    text = text.lower().strip()

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
        'banglor': 'location Bangalore',
        'friday-ad': 'subsidiary Friday Media Group (Consolidated)'
    }

    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    result = []

    i = 0
    while i < len(words):
        found_multi_word = False
        for j in range(min(3, len(words) - i), 1, -1):
            phrase = ' '.join(words[i:i+j])
            if phrase in ["budget variance", "variance analysis"]:
                result.append(phrase)
                i += j
                found_multi_word = True
                break
            if phrase in normalisation_dict:
                if "budget" in phrase and "category" not in phrase:
                    result.append("Budget")
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

        word = words[i]
        if word in abbrev_mapping:
            result.append(abbrev_mapping[word])
        elif word in normalisation_dict:
            if word.lower() == "budget":
                result.append("Budget")
            elif word.lower() == "subsistence":
                result.append("Account_Name subsistence")
            else:
                result.append(normalisation_dict[word])
        else:
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
                query_embedding = sentence_model.encode([word], convert_to_tensor=False)
                similarities = cosine_similarity(query_embedding, dictionary_embeddings)[0]
                top_idx = np.argmax(similarities)
                top_similarity = similarities[top_idx]

                if top_similarity >= threshold:
                    match = dictionary_keys[top_idx]
                    if match.lower() == "budget" or "budget" in match.lower():
                        result.append("Budget")
                    elif match.lower() == "subsistence" or word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    elif match.lower() == "subsidiary" and word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    else:
                        result.append(normalisation_dict.get(match, match))
                else:
                    if word.lower() == "subsistence":
                        result.append("Account_Name subsistence")
                    else:
                        result.append(word)
        i += 1

    normalized_text = ' '.join(result)
    normalized_text = re.sub(r'\s*([-_])\s*', r'\1', normalized_text)
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
    
    return normalized_text

def parse_formula_to_intent(formula_str: str):
    match = re.match(r'(\w+)\((.*)\)', formula_str)
    if not match:
        return {"error": "Invalid formula format."}

    formula_type = match.group(1).upper()
    raw_params = match.group(2)
    params = [p.strip() for p in raw_params.split(',') if p.strip()]

    if formula_type not in formula_mapping or len(params) != len(formula_mapping[formula_type]):
        return {"error": "Mismatch in formula parameters.", "formula": formula_str}
    
    intent_dict = dict(zip(formula_mapping[formula_type], params))
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
    formula_str = formula_type + "("

    if formula_type == "SUITEREC":
        table_name = intent.get("TABLE_NAME", "").strip()
        clean_table_name = re.sub(r'[{}"\'\[\]]', '', table_name).strip()
        if clean_table_name:
            formula_str += f'"{clean_table_name}"'
        else:
            formula_str += '""'
        formula_str += ")"
        return formula_str

    for i, field in enumerate(ordered_fields):
        value = intent.get(field, "").strip()

        all_fields = [
            "Subsidiary", "Account Number", "Account Name", "Classification", "account_name",
            "Department", "Location", "Customer Number", "Customer Name",
            "Vendor Name", "Vendor Number", "Class", "high/low", "Limit of Records",
            "Budget category", "From Period", "To Period", "TABLE_NAME",
            "Grouping and Filtering"
        ]

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

        asterisk_fields = [
            "Account Name", "Account Number", "Vendor Name", "Vendor Number", "account_name",
            "Customer Name", "Customer Number"
        ]

        field_variations = [
            field.lower(),
            field.replace(' ', '_').lower(),
            field.replace('/', '_').lower(),
            field.lower().replace('limit of records', 'limit of record'),
            field.lower().replace('high/low', 'high_low'),
            field.lower().replace('grouping and filtering', 'grouping_and_filtering')
        ]

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

        if field == "Subsidiary":
            if is_placeholder:
                formula_str += '"Friday Media Group (Consolidated)"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"Friday Media Group (Consolidated)"'
        elif field in ["From Period", "To Period"]:
            if is_placeholder or (isinstance(value, str) and "current month" in value.lower()):
                current_month_year = datetime.now().strftime("%B %Y")
                formula_str += f'"{current_month_year}"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    current_month_year = datetime.now().strftime("%B %Y")
                    formula_str += f'"{current_month_year}"'
        elif field == "Account Name":
            if has_account_name_placeholder or is_placeholder or clean_value == 'account_name':
                formula_str += '"*"'
            else:
                value = re.sub(r'[{}"\'\[\]]', '', value).strip()
                if value:
                    formula_str += f'"{value}"'
                else:
                    formula_str += '"Accounts Receivable"'
        elif is_placeholder:
            if field == "Account Name" and has_account_name_placeholder:
                formula_str += '"*"'
            elif field in asterisk_fields:
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

        if i < len(ordered_fields) - 1:
            formula_str += ", "

    formula_str += ")"
    return formula_str

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

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Netsuite Formula Generator Chat")

if not st.session_state.has_valid_api_key:
    st.warning("**OpenAI API Key Required**: Please enter your OpenAI API key to use this application.", icon="⚠️")
    new_key = st.text_input("OpenAI API Key", value="", type="password",
                           help="Enter your OpenAI API key. This is required to use the application.")
    if new_key:
        st.session_state.gpt_key = new_key
        st.session_state.has_valid_api_key = True
        st.success("API key added successfully! You can now use the application.")
        st.rerun()

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
                st.code(str(message["validated"]), language="text")
            if "regenerated_formula" in message:
                st.markdown("**Final Formula:**")
                st.code(message["regenerated_formula"], language="text")
    st.divider()

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
        formula = content.strip() if content else ""
        if formula.startswith('SUITECUS'):
            formula = re.sub(r'\[account_name\]|account_name', '[*]', formula, flags=re.IGNORECASE)
        parsed = parse_formula_to_intent(formula)

        if "error" in parsed:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here's the GPT response:",
                "formula": formula,
                "normalized_query": normalized_query
            })
            with st.chat_message("assistant"):
                st.markdown("**Normalized Query:**")
                st.code(normalized_query, language="text")
                st.markdown("**GPT Response:**")
                st.code(formula, language="text")
        else:
            validated = validate_intent_fields_v2(parsed["intent"])
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
                    st.markdown("**Normalized Query:**")
                    st.code(normalized_query, language="text")
                    st.markdown("**GPT Response:**")
                    st.code(formula, language="text")
                    st.markdown("**Validation Results:**")
                    st.code(str(validated), language="text")
                    st.markdown("**Final Formula:**")
                    st.code(regenerated_formula, language="text")
            else:
                st.error("No response generated")
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error occurred: {str(e)}. Partial results (if any):",
            "normalized_query": normalized_query,
            "formula": formula if 'formula' in locals() else "",
            "validated": validated if 'validated' in locals() else {},
            "regenerated_formula": regenerated_formula if 'regenerated_formula' in locals() else ""
        })
        with st.chat_message("assistant"):
            st.markdown("**Normalized Query:**")
            st.code(normalized_query, language="text")
            if 'formula' in locals():
                st.markdown("**GPT Response:**")
                st.code(formula, language="text")
            if 'validated' in locals() and validated:
                st.markdown("**Validation Results:**")
                st.code(str(validated), language="text")
            if 'regenerated_formula' in locals():
                st.markdown("**Final Formula:**")
                st.code(regenerated_formula, language="text")
            else:
                st.markdown("**Note:** No further results available due to error.")

if st.session_state.messages:
    if st.button("Copy Chat History"):
        chat_history = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}\nNormalized Query: {msg.get('normalized_query', '')}\nGPT Response: {msg.get('formula', '')}\nValidation Results: {msg.get('validated', '')}\nFinal Formula: {msg.get('regenerated_formula', '')}"
            for msg in st.session_state.messages
        )
        st.toast("Chat history copied to clipboard!")
        st.code(chat_history, language="text")

if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()