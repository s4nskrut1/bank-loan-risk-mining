"""
Generates an authentic historical banking dataset for loan credit analysis:
'historical_loan_records.csv' with realistic distributions and relationships matching the reference UI.
Also creates a copy on the user's Desktop for convenient upload!
"""

import os
import numpy as np
import pandas as pd

def generate_curated_loan_dataset(n=1200, seed=42):
    np.random.seed(seed)
    
    applicant_ids = [f"APP-{1001 + i}" for i in range(n)]
    
    # 1. Demographics & Employment
    ages = np.random.randint(22, 65, size=n)
    emp_statuses = np.random.choice(
        ['Salaried', 'Self-Employed', 'Business', 'Unemployed'],
        size=n,
        p=[0.58, 0.23, 0.15, 0.04]
    )
    
    emp_years = []
    for age, status in zip(ages, emp_statuses):
        if status == 'Unemployed':
            emp_years.append(0.0)
        else:
            max_tenure = max(1, age - 21)
            emp_years.append(round(float(np.clip(np.random.exponential(scale=5.0), 0.5, max_tenure)), 1))
    emp_years = np.array(emp_years)
    
    # 2. Financials (Annual Income & Account Liquidity in INR)
    base_incomes = np.random.lognormal(mean=13.6, sigma=0.52, size=n)
    annual_incomes = np.clip(np.round(base_incomes, -3), 250000, 3500000)
    annual_incomes = np.where(emp_statuses == 'Unemployed', np.random.choice([0, 120000, 180000], size=n), annual_incomes)
    
    account_balances = []
    for inc in annual_incomes:
        bal = np.random.uniform(0.08, 0.55) * max(inc, 100000)
        account_balances.append(int(np.clip(round(bal, -2), 5000, 1500000)))
    account_balances = np.array(account_balances)
    
    # 3. Credit Bureau History & Existing Debts
    credit_histories = np.random.choice(['Good', 'Fair', 'Poor'], size=n, p=[0.55, 0.30, 0.15])
    existing_loans = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.28, 0.35, 0.20, 0.10, 0.05, 0.02])
    
    # 4. Loan Requested
    loan_multipliers = np.random.uniform(0.25, 2.4, size=n)
    loan_amounts = np.clip(np.round(annual_incomes * loan_multipliers, -3), 50000, 5000000).astype(int)
    
    # 5. Loan-to-Income Ratio
    loan_to_income = np.round(loan_amounts / np.maximum(annual_incomes, 100000), 2)
    
    # 6. Repayment Track Record
    repayment_histories = []
    for ch in credit_histories:
        if ch == 'Good':
            rh = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.88, 0.10, 0.02])
        elif ch == 'Fair':
            rh = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.48, 0.44, 0.08])
        else: # Poor
            rh = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.12, 0.42, 0.46])
        repayment_histories.append(rh)
    repayment_histories = np.array(repayment_histories)
    
    # 7. Loan Purpose
    loan_purposes = np.random.choice(['Home Loan', 'Personal Loan', 'Auto Loan', 'Education Loan', 'Business Loan'], size=n, p=[0.35, 0.30, 0.15, 0.10, 0.10])
    
    # 8. Realistic Risk & Historical Default Logic (Calibrated for Basel Stratification)
    risk_score = np.zeros(n)
    for i in range(n):
        s = 0.0
        if repayment_histories[i] == 'Frequent':
            s += 50
        elif repayment_histories[i] == '1-2 Missed':
            s += 24
        else:
            s += 2
            
        if credit_histories[i] == 'Poor':
            s += 26
        elif credit_histories[i] == 'Fair':
            s += 12
            
        if loan_to_income[i] > 1.8:
            s += 20
        elif loan_to_income[i] > 1.0:
            s += 10
            
        if existing_loans[i] >= 3:
            s += 14
        elif existing_loans[i] == 2:
            s += 6
            
        if emp_statuses[i] == 'Unemployed':
            s += 25
        elif emp_years[i] < 1.0:
            s += 8
        elif emp_years[i] >= 5.0:
            s -= 8
            
        s += np.random.normal(0, 3)
        risk_score[i] = max(0, s)
        
    risk_levels = []
    historical_defaults = []
    
    for s, rh in zip(risk_score, repayment_histories):
        if s < 32 and rh != 'Frequent':
            risk_levels.append('Low')
            historical_defaults.append(0)
        elif s < 60:
            risk_levels.append('Medium')
            historical_defaults.append(1 if np.random.random() < 0.18 else 0)
        else:
            risk_levels.append('High')
            historical_defaults.append(1 if np.random.random() < 0.84 else 0)
            
    df = pd.DataFrame({
        'Applicant_ID': applicant_ids,
        'Age': ages,
        'Annual_Income': annual_incomes.astype(int),
        'Employment_Status': emp_statuses,
        'Employment_Years': emp_years,
        'Loan_Amount': loan_amounts,
        'Credit_History': credit_histories,
        'Existing_Loans': existing_loans,
        'Repayment_History': repayment_histories,
        'Account_Balance': account_balances,
        'Loan_Purpose': loan_purposes,
        'Loan_to_Income_Ratio': loan_to_income,
        'Historical_Default': historical_defaults,
        'Risk_Level': risk_levels
    })
    
    return df

if __name__ == '__main__':
    df = generate_curated_loan_dataset(1200)
    
    # Save in project root
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'historical_loan_records.csv')
    df.to_csv(project_path, index=False)
    print(f"Dataset generated at {project_path}")
    
    # Save a copy on the user's Downloads / Desktop for quick drag-and-drop
    downloads_path = os.path.expanduser(r'~\Downloads\historical_loan_records.csv')
    try:
        df.to_csv(downloads_path, index=False)
        print(f"Convenience copy also saved to Downloads: {downloads_path}")
    except Exception as e:
        print(f"Could not save to Downloads: {e}")
        
    # Save sample template
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_loan_template.csv')
    df.head(10).to_csv(template_path, index=False)
    print("Template CSV updated.")
