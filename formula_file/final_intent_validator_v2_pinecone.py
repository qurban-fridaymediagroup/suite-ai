import pandas as pd
from datetime import datetime
import re
import calendar
import os
from sentence_transformers import SentenceTransformer
import pinecone
from formula_file.period_utils import get_period_range, normalize_period_string, validate_period_order
from formula_file.smart_intent_correction import smart_intent_correction_restricted
from formula_file.constants import unified_columns
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")
index_name = "suiteai-index"

# Create or connect to Pinecone index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")
index = pinecone.Index(index_name)

# Initialize SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Simulate canonical values (replace with your actual data source)
canonical_values = {
    'account name': ['account1', 'account2', 'account3'],
    'classification': ['class1', 'class2', 'class3'],
    'subsidiary': ['sub1', 'sub2', 'sub3'],
    'vendor name': ['vendor1', 'vendor2', 'vendor3'],
    'vendor number': ['vendor1', 'vendor2', 'vendor3']
}
# Convert to sorted lists
for col in canonical_values:
    canonical_values[col] = sorted(list(set(canonical_values[col])))

# Handle aliases for Classification/Brand/Cost Center/Class
classification_aliases = ['Classification', 'Brand', 'Cost Center', 'Class']
for alias in classification_aliases:
    mapped_col = column_mapping.get(alias.lower(), alias)
    if mapped_col.lower() in canonical_values:
        canonical_values['classification'].extend(canonical_values.get(mapped_col.lower(), []))
canonical_values['classification'] = sorted(list(set(canonical_values['classification'])))

# Handle aliases for Subsidiary/Sub
subsidiary_aliases = ['Subsidiary', 'Sub']
for alias in subsidiary_aliases:
    mapped_col = column_mapping.get(alias.lower(), alias)
    if mapped_col.lower() in canonical_values:
        canonical_values['subsidiary'].extend(canonical_values.get(mapped_col.lower(), []))
canonical_values['subsidiary'] = sorted(list(set(canonical_values['subsidiary'])))

# Index canonical values in Pinecone
def index_canonical_values():
    for col, values in canonical_values.items():
        vectors = model.encode(values)
        upsert_data = [(f"{col}_{i}", vec.tolist(), {"value": val, "column": col}) 
                      for i, (val, vec) in enumerate(zip(values, vectors))]
        index.upsert(vectors=upsert_data)

# Run indexing (should be done once or when data updates)
index_canonical_values()

def semantic_match(input_val, column, top_k=1):
    if not input_val or input_val.strip() in ["", "-", "!", "not found"]:
        return None
    input_vec = model.encode([input_val])[0]
    query_result = index.query(vector=input_vec.tolist(), 
                             top_k=top_k, 
                             include_metadata=True, 
                             filter={"column": column.lower()})
    if query_result["matches"] and query_result["matches"][0]["score"] > 0.7:
        return query_result["matches"][0]["metadata"]["value"]
    return None

def validate_intent_fields_v2(intent_dict, original_query=""):
    validated = {}
    notes = {}
    warnings = []

    # Check for placeholders in the input
    placeholder_patterns = [
        r'\[.*?\]',  # Matches [placeholder]
        r'\{.*?\}',  # Matches {placeholder}
        r'^\{\".*?\"\}$',  # Matches {"placeholder"}
    ]

    # Step 1: Handle periods first
    from_val = intent_dict.get("From Period", "")
    to_val = intent_dict.get("To Period", "")

    # Clean values from extra quotes and braces for checking
    from_val_clean = re.sub(r'[{}\"]', '', str(from_val)).strip()
    to_val_clean = re.sub(r'[{}\"]', '', str(to_val)).strip()

    # Check if periods are placeholders
    from_is_placeholder = any(re.search(pattern, str(from_val)) for pattern in placeholder_patterns)
    to_is_placeholder = any(re.search(pattern, str(to_val)) for pattern in placeholder_patterns)

    # Updated period handling logic
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
    else:
        # Normal period processing
        try:
            from_p, to_p = get_period_range(from_val_clean, to_val_clean)
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

    # Step 2: Handle other fields
    for key, value in intent_dict.items():
        if key in ["From Period", "To Period"]:
            continue

        # Check if the value is a placeholder
        is_placeholder = any(re.search(pattern, str(value)) for pattern in placeholder_patterns)

        # Extract the actual value from the placeholder if it exists
        clean_val = re.sub(r'[{}"\'\[\]]', '', str(value)).strip().lower()

        # If it's a generic placeholder like [Customer] or {High_Low}, preserve it
        if is_placeholder and (clean_val.lower() in [key.lower().replace(" ", "_"),
                                                     key.lower(),
                                                     key.lower() + "_number",
                                                     key.lower() + "_name"]):
            validated[key] = value
            notes[key] = "Generic placeholder preserved"
            continue

        # If it's a placeholder with a specific value, extract and validate that value
        value_to_validate = clean_val
        canonical_col = unified_columns.get(key)
        possible_values = canonical_values.get(canonical_col, [])
        
        if key == "Account Number":
            canonical_col = 'account name'  # Always match Account Number against Account Name column
            possible_values = canonical_values.get(canonical_col, [])
        
        if key.lower() == "high/low":
            if value_to_validate in ["high", "low"]:
                validated[key] = value_to_validate
                notes[key] = "Exact match"
            else:
                if is_placeholder and "high_low" in clean_val.lower():
                    validated[key] = value
                    notes[key] = "Placeholder preserved"
                else:
                    validated[key] = "high"  # Default to high
                    notes[key] = "Default value used"
            continue
        
        if key.lower() == "limit of record":
            if value_to_validate.isdigit():
                validated[key] = value_to_validate
                notes[key] = "Valid integer"
            else:
                if is_placeholder and "limit" in clean_val.lower():
                    validated[key] = value
                    notes[key] = "Placeholder preserved"
                else:
                    validated[key] = "10"  # Default to 10
                    notes[key] = "Default value used"
            continue

        # Semantic Matching with Pinecone
        if value_to_validate in ["", "-", "!", "not found"]:
            validated[key] = ""
            notes[key] = "Empty or already invalid"
            continue

        # Use Pinecone semantic search
        matched = semantic_match(value_to_validate, canonical_col)
        if matched:
            validated[key] = matched
            notes[key] = "Semantic match via Pinecone"
        else:
            # Try other columns if no match found
            found_in_other_column = False
            for other_col in canonical_values.keys():
                if other_col != canonical_col:
                    matched = semantic_match(value_to_validate, other_col)
                    if matched:
                        validated[key] = matched
                        notes[key] = f"Semantic match in {other_col} column"
                        found_in_other_column = True
                        break

            if not found_in_other_column:
                if is_placeholder:
                    validated[key] = value
                    notes[key] = "Placeholder with unmatched value preserved"
                else:
                    validated[key] = value_to_validate
                    notes[key] = "Not matched in any known column"
                    warnings.append(f"'{value_to_validate}' in {key} not recognized in any column.")

    # Step 3: Smart reassignment — lock Limit of record and high/low
    smart_input = validated.copy()
    for lk in ["Limit of record", "high/low"]:
        if lk in smart_input:
            smart_input[lk] = "LOCKED_VALUE"
    
    smart = smart_intent_correction_restricted(smart_input)
    smart_validated = smart["validated_intent"]
    
    for lk in ["Limit of record", "high/low"]:
        if lk in validated:
            smart_validated[lk] = validated.get(lk, "")
    
    final_validated = smart_validated
    
    return {
        "validated_intent": final_validated,
        "match_notes": notes
    }