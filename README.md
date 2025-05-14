# Suite AI

A powerful Streamlit-based application that provides an intelligent interface for generating and managing NetSuite formulas through natural language processing.

## Features

- Natural language processing for NetSuite formula generation
- Support for multiple formula types (SUITEGEN, SUITEGENREP, SUITECUS, etc.)
- Intelligent prompt normalization and term standardization
- Interactive chat interface powered by OpenAI GPT
- Real-time formula validation

## Prerequisites

- Python 3.x
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/suite-ai.git
   cd suite-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   source venv/bin/activate
   streamlit run suiteai.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your query in natural language, and the AI will generate the appropriate NetSuite formula

## Supported Formulas

- SUITEGEN: Fetch general ledger/account aggregated totals, spend, and balances
- SUITEGENREP: Fetch detailed transaction lists or summary reports
- SUITECUS: Fetch customer transactions or invoices
- SUITEVEN: Fetch vendor transactions or invoices
- SUITEBUD: Fetch budgeted account totals and balances
- SUITEBUDREP: Fetch detailed reports for budgeted accounts
- SUITEVAR: Perform actual vs budget variance analysis
- SUITEREC: Fetch master lists of entity records

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
