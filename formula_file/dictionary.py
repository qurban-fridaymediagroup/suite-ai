normalisation_dict = {
    # Subsidiary
    'subsidiary': 'Subsidiary', 'subsidary': 'Subsidiary', 'subsidiry': 'Subsidiary',
    'subsidiery': 'Subsidiary', 'subsdiary': 'Subsidiary', 'subsiduary': 'Subsidiary',
    'subisidary': 'Subsidiary',  'subsidiaries': 'Subsidiary',
    'subsidies': 'Subsidiary', 'subs': 'Subsidiary', 'sub.': 'Subsidiary',
    'subsid.': 'Subsidiary', 'subsidiary company': 'Subsidiary', 'affiliate': 'Subsidiary',
    'child company': 'Subsidiary', 'satellite branch': 'Subsidiary', 'sub comp': 'Subsidiary',
    'sub co': 'Subsidiary', 'affiliate branch': 'Subsidiary', 'subcompany': 'Subsidiary',
    'your_subsidiary': 'Subsidiary', 'your subsidiary': 'Subsidiary',

    # Account Number
    'account number': 'Account', 'acount number': 'Account',
    'accout number': 'Account', 'accont number': 'Account',
    'acoount no': 'Account', 'account numbers': 'Account',
    'acc no': 'Account', 'a/c no': 'Account', 'acc#': 'Account', 'a#': 'Account',
    'acct no': 'Account', 'ac no': 'Account', 'account id': 'Account',
    'gl no': 'Account', 'ledger id': 'Account', 'ledger number': 'Account',
    'gl acc no': 'Account', 'acc id': 'Account_Number',

    # Account Name
    'account name': 'Account', 'acount name': 'Account',
    'accnt name': 'Account', 'accont name': 'Account','a / c': 'Account',
    'acc name': 'Account', 'a/c name': 'Account', 'acct name': 'Account',
    'ledger account': 'Account', 'gl account name': 'Account',
    'ledger acc name': 'Account', 'gl name': 'Account',

    # From Period
    'from period': 'From_Period', 'frm period': 'From_Period', 'from peroid': 'From_Period',
    'frm peroid': 'From_Period', 'from peiod': 'From_Period', 'from per': 'From_Period',
    'frm per': 'From_Period', 'strt per': 'From_Period', 'starting period': 'From_Period',
    'beginning period': 'From_Period', 'init period': 'From_Period',
    'start date': 'From_Period', 'start per': 'From_Period',

    # To Period
    'to period': 'To_Period', 'to peroid': 'To_Period', 'to perod': 'To_Period',
    'top eriod': 'To_Period', 'to per': 'To_Period', 'till per': 'To_Period',
    'end per': 'To_Period', 'ending period': 'To_Period', 'closing period': 'To_Period',
    'end date': 'To_Period', 'till date': 'To_Period',

    # Class
    'classification': 'Class', 'classfication': 'Class',
    'clssfication': 'Class', 'classifcation': 'Class',
    'cls': 'Class', 'clss': 'Class', 'cat cls': 'Class',
    'category': 'Class', 'type': 'Class',
    'brand': 'Class', 'brnd': 'Class', 'class': 'Class',
    'grp class': 'Class', 'cat class': 'Class',

    # Department
    'department': 'Department', 'deprtment': 'Department', 'departmnt': 'Department',
    'deparment': 'Department', 'depertment': 'Department',
    'dep': 'Department', 'dpmt': 'Department', 'unit': 'Department',
    'branch office': 'Department', 'section': 'Department', 'dept id': 'Department',
    'department id': 'Department', 'dept': 'Department', 'division': 'Department',

    # Location
    'location': 'Location', 'locaiton': 'Location', 'locaton': 'Location',
    'locatoin': 'Location', 'loction': 'Location', 'loaction': 'Location',
    'loacation': 'Location', 'loc': 'Location', 'loc.': 'Location', 'locn': 'Location',
    'site': 'Location', 'branch': 'Location', 'office': 'Location',
    'warehouse': 'Location', 'facility': 'Location', 'wh loc': 'Location',
    'site office': 'Location', 'loc id': 'Location', 'loc code': 'Location',

    # Customer Number
    'customer number': 'Customer', 'custmer number': 'Customer',
    'custmor no': 'Customer', 'customeer id': 'Customer',
    'cust no': 'Customer', 'cust id': 'Customer', 'cstm no': 'Customer',
    'cust num': 'Customer', 'client number': 'Customer', 'buyer number': 'Customer',
    'cust#': 'Customer', 'cstm id': 'Customer', 'customer#': 'Customer',

    # Vendor Name
    'vendor name': 'Vendor', 'vendor': 'Vendor', 'vedor name': 'Vendor',
    'vendr name': 'Vendor', 'vendorr': 'Vendor', 'supp': 'Vendor',
    'supp name': 'Vendor', 'sup': 'Vendor', 'sup name': 'Vendor',
    'vend name': 'Vendor', 'vndr name': 'Vendor', 'supplier name': 'Vendor',
    'provider name': 'Vendor', 'merchant name': 'Vendor',
    'vendor acc': 'Vendor', 'vend acc': 'Vendor', 'supplier acc': 'Vendor',

    # Budget Category
    'budget category': 'Budget_Category', 'bdgt catgory': 'Budget_Category',
    'budgt category': 'Budget_Category', 'bud category': 'Budget_Category',
    'bud cat': 'Budget_Category', 'bgt cat': 'Budget_Category', 'budg cat': 'Budget_Category',
    'budget type': 'Budget_Category', 'budget group': 'Budget_Category',
    'bud grp': 'Budget_Category', 'budtype': 'Budget_Category', 'bgt grp': 'Budget_Category',

}

formula_mapping = {
    "SUITEGEN": ["Subsidiary", "Account Number", "From Period", "To Period", "Class", "Department", "Location"],
    "SUITECUS": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEGENREP": ["Subsidiary", "Account Number", "From Period", "To Period", "Class", "Department", "Location"],
    "SUITEREC": ["TABLE_NAME"],
    "SUITEBUD": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Class", "Department", "Location"],
    "SUITEBUDREP": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Class", "Department", "Location"],
    "SUITEVAR": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Class", "Department", "Location"],
    "SUITEVEN": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"]
}