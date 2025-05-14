from fastapi import FastAPI, HTTPException
from suiteai import normalize_prompt, parse_formula_to_intent, generate_formula_from_intent
from pydantic import BaseModel
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from formula_file.dictionary import normalisation_dict, formula_mapping


# Load environment variables
load_dotenv()

# Get API key and model from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "ft:gpt-4o-mini-2024-07-18:hellofriday::BU8GWu9n")  

app = FastAPI()

class FormulaRequest(BaseModel):
    text: str


system_prompt = """
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

@app.post("/process-formula")
def process_formula(request: FormulaRequest):
    try:
        # Normalize text
        normalized = normalize_prompt(request.text)
        print("\n Normalized text: ", normalized)

        # Initialize OpenAI client with the current API key
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": normalized}
            ],
            temperature=0.7,
        )

        content = response.choices[0].message.content
        # Check if content exists before calling strip()
        formula = content.strip() if content else ""
        # Replace [account_name] or account_name with [*] in GPT response for SUITECUS

        formula = re.sub(r'\[account\]|account', '[*]', formula, flags=re.IGNORECASE)
        print("\n Formula: ", formula)
        # Parse formula to intent
        intent = parse_formula_to_intent(formula)
        if "error" in intent:
            raise HTTPException(status_code=400, detail=intent["error"])
        print("\n Intent: ", intent)
        # Generate formula from intent
        formula = generate_formula_from_intent(
            intent["formula_type"],
            intent["intent"],
            formula_mapping,
            intent.get("has_account_name_placeholder", False)
        )
        
        return {
            "normalized_text": normalized,
            "intent": intent,
            "formula": formula
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))