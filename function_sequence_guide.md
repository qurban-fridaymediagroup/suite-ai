# Function Call Sequence Guide

## Overview
This guide explains the sequence for calling functions in the SuiteAI application, starting with `validate_intent_fields_v2`.

## Function Call Sequence

1. **`validate_intent_fields_v2(intent_dict, original_query="")`**
   - This is the first function in the sequence
   - Input: A dictionary of intent fields and an optional original query string
   - Output: A validation result dictionary containing:
     - `validated_intent`: The validated and corrected intent fields
     - `match_notes`: Notes about the validation process
     - `warnings`: Any warnings generated during validation
     - `original_intent`: The original intent dictionary

2. **`generate_formula_from_intent(formula_type, intent, formula_mapping, has_account_name_placeholder=False)`**
   - This function is called after `validate_intent_fields_v2`
   - Input:
     - `formula_type`: The type of formula to generate (e.g., "SUITEGEN", "SUITECUS")
     - `intent`: The validated intent dictionary from `validate_intent_fields_v2`
     - `formula_mapping`: A mapping of formula types to their required parameters
     - `has_account_name_placeholder`: A boolean indicating if the account name is a placeholder
   - Output: A formatted formula string

## Example with Provided Parameters

Given the parameters:
```
{
  'Subsidiary': '{subsidiary}',
  'Account Number': '["ppa"]',
  'From Period': '{"last month"}',
  'To Period': '{"last month"}',
  'Class': '[class]',
  'Department': '["agency"]',
  'Location': '[location]'
}
```

The sequence would be:

1. Call `validate_intent_fields_v2` with these parameters:
   ```python
   validated = validate_intent_fields_v2({
     'Subsidiary': '{subsidiary}',
     'Account Number': '["ppa"]',
     'From Period': '{"last month"}',
     'To Period': '{"last month"}',
     'Class': '[class]',
     'Department': '["agency"]',
     'Location': '[location]'
   })
   ```

2. The function will validate and correct each field:
   - It will handle placeholders like `{subsidiary}` and `[class]`
   - It will normalize period values like `{"last month"}`
   - It will match values against canonical values in the database

3. Call `generate_formula_from_intent` with the validated intent:
   ```python
   formula_type = "SUITEGEN"  # Determined based on the fields present
   regenerated_formula = generate_formula_from_intent(
     formula_type,
     validated["validated_intent"],
     formula_mapping
   )
   ```

4. This will generate a formula string like:
   ```
   SUITEGEN("Friday Media Group (Consolidated)", "ppa", "Apr 2025", "Apr 2025", "", "agency", "")
   ```

## Notes
- The formula type (e.g., "SUITEGEN", "SUITECUS") is determined based on the fields present in the intent dictionary
- The `validate_intent_fields_v2` function handles normalization, validation, and correction of intent fields
- The `generate_formula_from_intent` function formats the validated intent into a proper formula string
- Some fields have default values if they are missing or invalid (e.g., "Friday Media Group (Consolidated)" for Subsidiary)