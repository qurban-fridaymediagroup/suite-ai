normalisation_dict = {
    # Subsidiary
    'subsidiary': 'Subsidiary', 'subsidary': 'Subsidiary', 'subsidiry': 'Subsidiary',
    'subsidiery': 'Subsidiary', 'subsdiary': 'Subsidiary', 'subsiduary': 'Subsidiary',
    'subisidary': 'Subsidiary',  'subsidiaries': 'Subsidiary',
    'subsidies': 'Subsidiary', 'subs': 'Subsidiary', 'sub.': 'Subsidiary',
    'subsid.': 'Subsidiary', 'subsidiary company': 'Subsidiary', 'affiliate': 'Subsidiary',
    'child company': 'Subsidiary', 'satellite branch': 'Subsidiary', 'sub comp': 'Subsidiary',
    'sub co': 'Subsidiary', 'affiliate branch': 'Subsidiary', 'subcompany': 'Subsidiary',

    # Account Number
    'account number': 'Account_Number', 'acount number': 'Account_Number',
    'accout number': 'Account_Number', 'accont number': 'Account_Number',
    'acoount no': 'Account_Number', 'account numbers': 'Account_Number',
    'acc no': 'Account_Number', 'a/c no': 'Account_Number', 'acc#': 'Account_Number',
    'acct no': 'Account_Number', 'ac no': 'Account_Number', 'account id': 'Account_Number',
    'gl no': 'Account_Number', 'ledger id': 'Account_Number', 'ledger number': 'Account_Number',
    'gl acc no': 'Account_Number', 'acc id': 'Account_Number',

    # Account Name
    'account name': 'Account_Name', 'acount name': 'Account_Name',
    'accnt name': 'Account_Name', 'accont name': 'Account_Name',
    'acc name': 'Account_Name', 'a/c name': 'Account_Name', 'acct name': 'Account_Name',
    'ledger account': 'Account_Name', 'gl account name': 'Account_Name',
    'ledger acc name': 'Account_Name', 'gl name': 'Account_Name',

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

    # Classification
    'classification': 'Classification', 'classfication': 'Classification',
    'clssfication': 'Classification', 'classifcation': 'Classification',
    'cls': 'Classification', 'clss': 'Classification', 'cat cls': 'Classification',
    'category': 'Classification', 'type': 'Classification', 'division': 'Classification',
    'brand': 'Classification', 'brnd': 'Classification', 'dept class': 'Classification',
    'grp class': 'Classification', 'cat class': 'Classification',

    # Department
    'department': 'Department', 'deprtment': 'Department', 'departmnt': 'Department',
    'deparment': 'Department', 'depertment': 'Department', 'dept': 'Department',
    'dep': 'Department', 'dpmt': 'Department', 'unit': 'Department',
    'branch office': 'Department', 'section': 'Department', 'dept id': 'Department',
    'department id': 'Department', 'dept code': 'Department',

    # Location
    'location': 'Location', 'locaiton': 'Location', 'locaton': 'Location',
    'locatoin': 'Location', 'loction': 'Location', 'loaction': 'Location',
    'loacation': 'Location', 'loc': 'Location', 'loc.': 'Location', 'locn': 'Location',
    'site': 'Location', 'branch': 'Location', 'office': 'Location',
    'warehouse': 'Location', 'facility': 'Location', 'wh loc': 'Location',
    'site office': 'Location', 'loc id': 'Location', 'loc code': 'Location',

    # Customer Number
    'customer number': 'Customer_Number', 'custmer number': 'Customer_Number',
    'custmor no': 'Customer_Number', 'customeer id': 'Customer_Number',
    'cust no': 'Customer_Number', 'cust id': 'Customer_Number', 'cstm no': 'Customer_Number',
    'cust num': 'Customer_Number', 'client number': 'Customer_Number', 'buyer number': 'Customer_Number',
    'cust#': 'Customer_Number', 'cstm id': 'Customer_Number', 'customer#': 'Customer_Number',

    # Vendor Name
    'vendor name': 'Vendor_Name', 'vendor': 'Vendor_Name', 'vedor name': 'Vendor_Name',
    'vendr name': 'Vendor_Name', 'vendorr': 'Vendor_Name', 'supp': 'Vendor_Name',
    'supp name': 'Vendor_Name', 'sup': 'Vendor_Name', 'sup name': 'Vendor_Name',
    'vend name': 'Vendor_Name', 'vndr name': 'Vendor_Name', 'supplier name': 'Vendor_Name',
    'provider name': 'Vendor_Name', 'merchant name': 'Vendor_Name',
    'vendor acc': 'Vendor_Name', 'vend acc': 'Vendor_Name', 'supplier acc': 'Vendor_Name',

    # Budget Category
    'budget category': 'Budget_Category', 'bdgt catgory': 'Budget_Category',
    'budgt category': 'Budget_Category', 'bud category': 'Budget_Category',
    'bud cat': 'Budget_Category', 'bgt cat': 'Budget_Category', 'budg cat': 'Budget_Category',
    'budget type': 'Budget_Category', 'budget group': 'Budget_Category',
    'bud grp': 'Budget_Category', 'budtype': 'Budget_Category', 'bgt grp': 'Budget_Category',

}

formula_mapping = {
    "SUITEGEN": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITECUS": ["Subsidiary", "Customer Number", "From Period", "To Period", "Account Number", "Class", "high/low", "Limit of record"],
    "SUITEGENREP": ["Subsidiary", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEREC": ["TABLE_NAME"],
    "SUITEBUD": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEBUDREP": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVAR": ["Subsidiary", "Budget category", "Account Number", "From Period", "To Period", "Classification", "Department", "Location"],
    "SUITEVEN": ["Subsidiary", "Vendor Name", "From Period", "To Period", "Account Name", "Class", "high/low", "Limit of record"]
}