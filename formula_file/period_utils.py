import calendar
import re
from datetime import datetime
from difflib import get_close_matches

# Known aliases for fuzzy matching
KNOWN_PERIOD_ALIASES = {
    "this q": ["this q", "this quarter", "thisquater", "thisq"],
    "last q": ["last q", "last quarter", "lastquater", "lastq"],
    "this month": ["this month", "current month", "currnt month", "curr month", "thismonth"],
    "start of the year": ["start of the year", "start year", "startofyear"],
    "year to date": ["year to date", "ytd", "yeartodate", "yer to date", "yr to date","this year", "this yr", "this y", "this YEAR","This Year"],
    "q1": ["q1", "quarter 1", "quarter1", "1st quarter", "first quarter", "q 1"],
    "q2": ["q2", "quarter 2", "quarter2", "2nd quarter", "second quarter", "q 2"],
    "q3": ["q3", "quarter 3", "quarter3", "3rd quarter", "third quarter", "q 3"],
    "q4": ["q4", "quarter 4", "quarter4", "4th quarter", "fourth quarter", "q 4"]
}

def normalize_period_term(term: str):
    term = term.strip().lower()
    all_aliases = {alias: base for base, group in KNOWN_PERIOD_ALIASES.items() for alias in group}
    best_match = get_close_matches(term, list(all_aliases.keys()), n=1, cutoff=0.7)
    return all_aliases.get(best_match[0]) if best_match else term

def normalize_period_string(period_str):
    if not period_str:
        return ""
        # Handle "30 04 2024" → "Apr 2024"
    match = re.search(r'(\d{1,2})[^\dA-Za-z]?(\d{2})[^\dA-Za-z]?(\d{4})', period_str)
    if match:
        day, month, year = match.groups()
        try:
            month_int = int(month)
            if 1 <= month_int <= 12:
                month_abbr = calendar.month_abbr[month_int]
                return f"{month_abbr} {year}"
        except:
            pass
    # Remove ordinal suffixes like '20th', '30st'
    period_str = re.sub(r'\b(\d{1,2})(st|nd|rd|th)\b', r'\1', period_str)
    # Remove non-alphanumeric characters except space
    period_str = re.sub(r'[^a-zA-Z0-9 ]', ' ', period_str).strip()
    parts = period_str.split()

    # Find month and year components
    month_candidates = [p for p in parts if p[:3].capitalize() in calendar.month_abbr]
    year_candidates = [p for p in parts if re.match(r'\d{4}', p)]

    if month_candidates and year_candidates:
        month = month_candidates[0][:3].capitalize()
        year = year_candidates[0]
        return f"{month} {year}"

    # Handle cases like 'Mar2025'
    match = re.match(r'([a-zA-Z]+)(\d{4})', period_str)
    if match:
        month, year = match.groups()
        month_abbr = month[:3].capitalize()
        if month_abbr in calendar.month_abbr:
            return f"{month_abbr} {year}"

    return period_str.strip().title()

def get_period_range(term_from: str, term_to: str | None = None, today=None):
    term_from_norm = normalize_period_term(term_from)
    term_to_norm = normalize_period_term(term_to or term_from)
    today = today or datetime.today()

    month = today.month
    year = today.year

    if month == 1:
        fiscal_year_start = year - 1
    else:
        fiscal_year_start = year

    def resolve_special(term: str):
        term = term.lower().strip()
        if term == "this q":
            return "Feb " + str(fiscal_year_start), "Apr " + str(fiscal_year_start)
        elif term == "last q":
            if month in [2, 3, 4]:
                return "Nov " + str(fiscal_year_start - 1), "Jan " + str(fiscal_year_start)
            elif month in [5, 6, 7]:
                return "Feb " + str(fiscal_year_start), "Apr " + str(fiscal_year_start)
            elif month in [8, 9, 10]:
                return "May " + str(fiscal_year_start), "Jul " + str(fiscal_year_start)
            else:
                return "Aug " + str(fiscal_year_start), "Oct " + str(fiscal_year_start)
        elif term == "this month":
            month_abbr = calendar.month_abbr[month]
            return f"{month_abbr} {year}", f"{month_abbr} {year}"
        elif term == "start of the year":
            return f"Feb {fiscal_year_start}", f"Feb {fiscal_year_start}"
        elif term == "year to date":
            month_abbr = calendar.month_abbr[month]
            return f"Feb {fiscal_year_start}", f"{month_abbr} {year}"
        elif term == "q1":
            return f"Feb {fiscal_year_start}", f"Apr {fiscal_year_start}"
        elif term == "q2":
            return f"May {fiscal_year_start}", f"Jul {fiscal_year_start}"
        elif term == "q3":
            return f"Aug {fiscal_year_start}", f"Oct {fiscal_year_start}"
        elif term == "q4":
            return f"Nov {fiscal_year_start}", f"Jan {fiscal_year_start + 1}"
        else:
            norm = normalize_period_string(term)
            return norm, norm

    # Resolve special cases or use default normalization
    from_period, _ = resolve_special(str(term_from_norm)) if term_from_norm is not None else ("", "")
    _, to_period = resolve_special(str(term_from_norm)) if term_from_norm is not None else ("", "")

    # Final normalization for safety
    from_period = normalize_period_string(from_period)
    to_period = normalize_period_string(to_period)

    return validate_period_order(from_period, to_period)

def month_year_to_int(period_str):
    try:
        month_str, year = period_str.split()
        month_num = list(calendar.month_abbr).index(month_str.capitalize())
        return int(year) * 100 + month_num
    except:
        return 0

def validate_period_order(from_period, to_period):
    from_val = month_year_to_int(from_period)
    to_val = month_year_to_int(to_period)
    if from_val > to_val:
        return from_period, to_period + " (Check: From > To)"
    return from_period, to_period
