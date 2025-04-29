import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state for model ID and system prompt
if 'fine_tuned_model' not in st.session_state:
    st.session_state.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:hellofriday::BRICgOMR"
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = """
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "formula" in message:
            st.code(message["formula"], language="text")

# Chat input
query = st.chat_input("Ask about Netsuite formulas...")

if query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Get AI response
    try:
        response = client.chat.completions.create(
            model=st.session_state.fine_tuned_model,
            messages=[
                {"role": "system", "content": st.session_state.system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        if content:
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here's the formula you requested:",
                "formula": content.strip()
            })
            with st.chat_message("assistant"):
                st.write("Here's the formula you requested:")
                st.code(content.strip(), language="text")
        else:
            st.error("No response generated")
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Add copy chat history button
if st.session_state.messages:
    if st.button("Copy Chat History"):
        chat_history = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}\n{msg.get('formula', '')}" 
            for msg in st.session_state.messages
        )
        st.toast("Chat history copied to clipboard!")
        st.code(chat_history, language="text")

# Add clear chat button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
