import os
import shutil
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_dataset(scenario_name, n_rows, base_income, loan_multiplier, default_prob_base):
    records = []
    
    purposes = ['Home Loan', 'Personal Loan', 'Auto Loan', 'Education Loan', 'Business Loan']
    
    for i in range(1, n_rows + 1):
        app_id = f"{scenario_name[:3].upper()}-{1000 + i}"
        
        if scenario_name == 'fintech':
            age = int(np.random.triangular(21, 28, 45))
        elif scenario_name == 'microfinance':
            age = int(np.random.triangular(22, 38, 62))
        elif scenario_name == 'tier1_retail':
            age = int(np.random.triangular(26, 42, 65))
        else: # stressed
            age = int(np.random.triangular(24, 40, 60))
            
        # Income distribution
        if scenario_name == 'tier1_retail':
            income = int(np.random.lognormal(mean=14.0, sigma=0.5)) # Mean ~1.2M - 3M
            income = max(500000, min(5000000, round(income, -3)))
            emp_statuses = ['Salaried', 'Business', 'Self-Employed']
            emp_probs = [0.70, 0.20, 0.10]
        elif scenario_name == 'microfinance':
            income = int(np.random.lognormal(mean=12.5, sigma=0.4)) # ~250k - 600k
            income = max(120000, min(900000, round(income, -3)))
            emp_statuses = ['Self-Employed', 'Business', 'Salaried']
            emp_probs = [0.55, 0.30, 0.15]
        elif scenario_name == 'fintech':
            income = int(np.random.lognormal(mean=13.2, sigma=0.5)) # ~500k - 1.2M
            income = max(240000, min(2200000, round(income, -3)))
            emp_statuses = ['Salaried', 'Self-Employed', 'Unemployed']
            emp_probs = [0.75, 0.20, 0.05]
        else: # stressed
            income = int(np.random.lognormal(mean=12.8, sigma=0.6))
            income = max(150000, min(1500000, round(income, -3)))
            emp_statuses = ['Salaried', 'Self-Employed', 'Unemployed', 'Business']
            emp_probs = [0.45, 0.30, 0.15, 0.10]
            
        emp_status = np.random.choice(emp_statuses, p=emp_probs)
        
        if emp_status == 'Unemployed':
            emp_years = 0.0
        else:
            emp_years = round(float(np.random.exponential(scale=max(1.0, (age - 20) * 0.25))), 1)
            emp_years = min(emp_years, float(age - 18))
            
        # Loan ticket size
        lir = round(float(np.random.gamma(shape=loan_multiplier, scale=0.45)), 2)
        lir = max(0.15, min(4.5, lir))
        loan_amount = int(round(income * lir, -3))
        loan_amount = max(30000, min(8000000, loan_amount))
        
        # Credit history & Repayment
        if scenario_name == 'tier1_retail':
            cred_choice = np.random.choice(['Good', 'Fair', 'Poor'], p=[0.72, 0.22, 0.06])
            repay_choice = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.80, 0.16, 0.04])
            exist_loans = int(np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.20, 0.05]))
            bal_mult = np.random.uniform(0.15, 0.60)
        elif scenario_name == 'microfinance':
            cred_choice = np.random.choice(['Good', 'Fair', 'Poor'], p=[0.45, 0.38, 0.17])
            repay_choice = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.60, 0.26, 0.14])
            exist_loans = int(np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.10, 0.05]))
            bal_mult = np.random.uniform(0.08, 0.25)
        elif scenario_name == 'fintech':
            cred_choice = np.random.choice(['Good', 'Fair', 'Poor'], p=[0.50, 0.35, 0.15])
            repay_choice = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.65, 0.23, 0.12])
            exist_loans = int(np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.22, 0.08]))
            bal_mult = np.random.uniform(0.05, 0.30)
        else: # stressed
            cred_choice = np.random.choice(['Good', 'Fair', 'Poor'], p=[0.20, 0.40, 0.40])
            repay_choice = np.random.choice(['On-Time', '1-2 Missed', 'Frequent'], p=[0.35, 0.35, 0.30])
            exist_loans = int(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.30, 0.30, 0.15, 0.10]))
            bal_mult = np.random.uniform(0.02, 0.18)
            
        balance = int(round(income * bal_mult, -2))
        
        if scenario_name == 'tier1_retail':
            purpose = np.random.choice(purposes, p=[0.45, 0.15, 0.20, 0.10, 0.10])
        elif scenario_name == 'microfinance':
            purpose = np.random.choice(purposes, p=[0.10, 0.25, 0.15, 0.10, 0.40])
        elif scenario_name == 'fintech':
            purpose = np.random.choice(purposes, p=[0.05, 0.45, 0.15, 0.30, 0.05])
        else:
            purpose = np.random.choice(purposes, p=[0.15, 0.40, 0.15, 0.10, 0.20])
            
        # Ground Truth Risk Calculation (Strict Bayesian Logic)
        risk_score = 0.0
        
        # Repayment history weight
        if repay_choice == 'Frequent':
            risk_score += 4.5
        elif repay_choice == '1-2 Missed':
            risk_score += 2.0
            
        # Credit history weight
        if cred_choice == 'Poor':
            risk_score += 3.8
        elif cred_choice == 'Fair':
            risk_score += 1.5
            
        # LIR weight
        if lir > 2.0:
            risk_score += 3.0
        elif lir > 1.2:
            risk_score += 1.6
            
        # Existing loans
        if exist_loans >= 3:
            risk_score += 2.0
        elif exist_loans >= 2:
            risk_score += 1.0
            
        # Employment
        if emp_status == 'Unemployed' or emp_years < 1.0:
            risk_score += 1.8
            
        # Balance buffer (mitigating factor)
        if balance > 0.4 * income:
            risk_score -= 1.2
            
        # Final Risk Level
        if risk_score >= 5.5:
            risk_level = 'High'
            hist_default = 1 if np.random.rand() < 0.85 else 0
        elif risk_score >= 2.5:
            risk_level = 'Medium'
            hist_default = 1 if np.random.rand() < 0.25 else 0
        else:
            risk_level = 'Low'
            hist_default = 1 if np.random.rand() < 0.03 else 0
            
        records.append({
            'Applicant_ID': app_id,
            'Age': age,
            'Annual_Income': income,
            'Employment_Status': emp_status,
            'Employment_Years': emp_years,
            'Loan_Amount': loan_amount,
            'Credit_History': cred_choice,
            'Existing_Loans': exist_loans,
            'Repayment_History': repay_choice,
            'Account_Balance': balance,
            'Loan_Purpose': purpose,
            'Loan_to_Income_Ratio': lir,
            'Historical_Default': hist_default,
            'Risk_Level': risk_level
        })
        
    return pd.DataFrame(records)

datasets = {
    'retail_banking_tier1.csv': ('tier1_retail', 800, 1500000, 1.8, 0.10),
    'microfinance_rural_lending.csv': ('microfinance', 1000, 450000, 1.5, 0.25),
    'fintech_digital_loans.csv': ('fintech', 1500, 750000, 2.2, 0.20),
    'stressed_credit_portfolio.csv': ('stressed', 600, 600000, 2.8, 0.45)
}

downloads_dir = r"C:\Users\SANSKRUTI\Downloads"
sample_dir = r"C:\Users\SANSKRUTI\.gemini\antigravity\scratch\bank_loan_risk_mining\sample_datasets"
os.makedirs(sample_dir, exist_ok=True)

for fname, (scen, n, inc, mult, def_p) in datasets.items():
    df = generate_dataset(scen, n, inc, mult, def_p)
    
    # Save in project sample_datasets
    proj_path = os.path.join(sample_dir, fname)
    df.to_csv(proj_path, index=False)
    
    # Copy to user's Downloads folder
    dl_path = os.path.join(downloads_dir, fname)
    df.to_csv(dl_path, index=False)
    
    risk_dist = df['Risk_Level'].value_counts().to_dict()
    def_rate = round(float(df['Historical_Default'].mean() * 100), 1)
    print(f"Generated {fname}: {len(df)} records | Default Rate: {def_rate}% | Risk Dist: {risk_dist}")

print("\nAll 4 datasets generated and copied to Downloads folder successfully!")
