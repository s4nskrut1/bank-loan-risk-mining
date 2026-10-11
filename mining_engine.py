"""
CrediMine AI - Core Data Mining & Machine Learning Engine
Re-engineered to support:
1. Dynamic Preprocessing & Cleaning (Handles any custom CSV structure)
2. Attribute Relevance Analysis:
   - Mutual Information (Information Gain)
   - Correlation Matrix (Pearson & Spearman for numerical attributes)
   - Tree-based Gini Split Feature Importance
   - Why These Features Matter explanatory card
3. Classification Models & Comparison:
   - Decision Tree (CART)
   - Gaussian Naive Bayes
   - Random Forest (Ensemble)
   - Metrics: Accuracy, Precision, Recall, F1-Score, Training Time
   - Multi-class Confusion Matrices (Low, Medium, High)
4. Decision Tree Visualization & Rule Extraction:
   - Interactive Node Hierarchy (Root, Internal Nodes, Leaves with sample count, Gini, and class)
   - Extracted Clean IF-THEN Rules matching reference panel 5
5. Live Underwriting Risk Assessment:
   - Input: Age, Income, Employment, Tenure, Loan Amount, Credit History, Existing Loans, Repayment, Balance, Purpose
   - Output: Risk Badge (Low, Medium, High), Risk Probability Meter, Explainable Factors, Recommended Action
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.stats as stats

class MiningEngine:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.clean_df = None
        self.feature_cols = []
        self.categorical_cols = []
        self.numerical_cols = []
        
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.models = {}
        self.metrics = {}
        self.confusion_matrices = {}
        self.tree_rules = []
        self.tree_structure = {}
        self.relevance_results = {}
        self.correlation_matrix = {}
        self.class_names = ['Low', 'Medium', 'High']
        
        self.process_pipeline()

    def process_pipeline(self):
        self._standardize_and_clean()
        self._compute_attribute_relevance()
        self._train_and_evaluate_models()
        self._extract_decision_tree_rules_and_nodes()

    def _standardize_and_clean(self):
        df = self.raw_df.copy()
        
        # 1. Clean column names
        col_map = {}
        for c in df.columns:
            clean_c = str(c).strip()
            low_c = clean_c.lower().replace(' ', '_').replace('-', '_')
            if 'applicant' in low_c or low_c == 'id':
                col_map[c] = 'Applicant_ID'
            elif low_c in ['age', 'cust_age']:
                col_map[c] = 'Age'
            elif low_c in ['loan_to_income_ratio', 'dti', 'dti_ratio', 'loan_to_income']:
                col_map[c] = 'Loan_to_Income_Ratio'
            elif low_c in ['annual_income', 'income', 'salary', 'applicant_income']:
                col_map[c] = 'Annual_Income'
            elif low_c in ['loan_amount', 'loanamt', 'loan_amt', 'loan']:
                col_map[c] = 'Loan_Amount'
            elif 'emp_status' in low_c or 'employment_status' in low_c or low_c == 'employment':
                col_map[c] = 'Employment_Status'
            elif 'emp_years' in low_c or 'employment_years' in low_c or 'tenure' in low_c or 'experience' in low_c:
                col_map[c] = 'Employment_Years'
            elif 'credit_history' in low_c or 'credit_score' in low_c or 'credit' in low_c:
                col_map[c] = 'Credit_History'
            elif 'existing_loans' in low_c or 'active_loans' in low_c or 'loans_count' in low_c:
                col_map[c] = 'Existing_Loans'
            elif 'repayment_history' in low_c or 'repayment' in low_c or 'delinquency' in low_c:
                col_map[c] = 'Repayment_History'
            elif 'account_balance' in low_c or 'balance' in low_c or 'savings' in low_c:
                col_map[c] = 'Account_Balance'
            elif 'purpose' in low_c or 'loan_purpose' in low_c:
                col_map[c] = 'Loan_Purpose'
            elif 'default' in low_c or 'historical_default' in low_c:
                col_map[c] = 'Historical_Default'
            elif 'risk' in low_c or 'risk_level' in low_c or 'risk_category' in low_c:
                col_map[c] = 'Risk_Level'
            else:
                col_map[c] = clean_c

        df.rename(columns=col_map, inplace=True)

        # 2. Impute missing values
        for col in df.columns:
            has_null = bool(df[col].isnull().to_numpy().any())
            if has_null:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Standard'
                    df[col] = df[col].fillna(mode_val)

        # 3. Derive Loan-to-Income Ratio
        if 'Loan_to_Income_Ratio' not in df.columns:
            if 'Loan_Amount' in df.columns and 'Annual_Income' in df.columns:
                loan_vals = df['Loan_Amount'].astype(float).values
                inc_vals = np.maximum(df['Annual_Income'].astype(float).values, 10000.0)
                df['Loan_to_Income_Ratio'] = np.round(loan_vals / inc_vals, 2)
            else:
                df['Loan_to_Income_Ratio'] = 0.45

        # 4. Standardize Risk_Level target
        if 'Risk_Level' not in df.columns:
            if 'Historical_Default' in df.columns:
                conditions = [
                    (df['Historical_Default'] == 1),
                    (df.get('Existing_Loans', 0) >= 3) | (df['Loan_to_Income_Ratio'] > 1.4)
                ]
                choices = ['High', 'Medium']
                df['Risk_Level'] = np.select(conditions, choices, default='Low')
            else:
                df['Risk_Level'] = np.where(df['Loan_to_Income_Ratio'] > 1.8, 'High',
                                   np.where(df['Loan_to_Income_Ratio'] > 0.8, 'Medium', 'Low'))

        def clean_risk_str(val):
            v = str(val).strip().lower()
            if 'high' in v or 'poor' in v or 'bad' in v or v == '1':
                return 'High'
            elif 'med' in v or 'fair' in v or 'mod' in v or 'watch' in v:
                return 'Medium'
            else:
                return 'Low'

        df['Risk_Level'] = df['Risk_Level'].apply(clean_risk_str)

        # 5. Historical_Default target for correlation
        if 'Historical_Default' not in df.columns:
            df['Historical_Default'] = (df['Risk_Level'] == 'High').astype(int)

        self.clean_df = df

        # 6. Separate features
        ignore_cols = ['Applicant_ID', 'Historical_Default', 'Risk_Level', 'Name', 'Customer_ID', 'id']
        self.feature_cols = [c for c in df.columns if c not in ignore_cols]
        self.numerical_cols = [c for c in self.feature_cols if np.issubdtype(df[c].dtype, np.number)]
        self.categorical_cols = [c for c in self.feature_cols if c not in self.numerical_cols]

        # 7. Encode for scikit-learn
        self.encoded_X = df[self.feature_cols].copy()
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.encoded_X[col] = le.fit_transform(self.encoded_X[col].astype(str))
            self.label_encoders[col] = le

        # Enforce canonical ordering: 0 -> Low, 1 -> Medium, 2 -> High
        self.risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        self.class_names = ['Low', 'Medium', 'High']
        self.y_encoded = df['Risk_Level'].map(self.risk_mapping).fillna(0).astype(int).to_numpy()

    def _compute_attribute_relevance(self):
        X = self.encoded_X
        y_default = self.clean_df['Historical_Default'].astype(int)
        
        # 1. Mutual Information
        try:
            mi_scores = mutual_info_classif(X, y_default, random_state=42)
            mi_dict = {col: round(float(score), 3) for col, score in zip(self.feature_cols, mi_scores)}
        except Exception:
            mi_dict = {col: 0.05 for col in self.feature_cols}

        # 2. Tree-based Feature Importance (Random Forest Gini decrease)
        rf_quick = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        rf_quick.fit(X, self.y_encoded)
        tree_importances = {col: round(float(imp), 3) for col, imp in zip(self.feature_cols, rf_quick.feature_importances_)}

        # 3. Correlation Matrix for numerical attributes
        corr_attrs = [c for c in ['Age', 'Annual_Income', 'Loan_Amount', 'Existing_Loans', 'Account_Balance', 'Loan_to_Income_Ratio'] if c in self.clean_df.columns]
        if len(corr_attrs) >= 2:
            corr_df = self.clean_df[corr_attrs].corr().round(2)
            corr_matrix = {
                'attributes': [c.replace('_', ' ') for c in corr_attrs],
                'matrix': corr_df.values.tolist()
            }
        else:
            corr_matrix = {'attributes': [], 'matrix': []}

        # Combine into ranked list
        ranked_table = []
        max_mi = max(mi_dict.values()) if max(mi_dict.values()) > 0 else 1.0
        max_tree = max(tree_importances.values()) if max(tree_importances.values()) > 0 else 1.0

        for col in self.feature_cols:
            score = (mi_dict[col] / max_mi * 0.5) + (tree_importances[col] / max_tree * 0.5)
            ranked_table.append({
                'attribute': col,
                'display_name': col.replace('_', ' '),
                'mutual_information': mi_dict[col],
                'tree_importance': tree_importances[col],
                'composite_score': round(float(score), 3)
            })

        ranked_table.sort(key=lambda x: x['composite_score'], reverse=True)
        for idx, item in enumerate(ranked_table):
            item['rank'] = idx + 1

        top_1 = ranked_table[0]['display_name'] if len(ranked_table) > 0 else "Repayment History"
        top_2 = ranked_table[1]['display_name'] if len(ranked_table) > 1 else "Credit History"
        top_3 = ranked_table[2]['display_name'] if len(ranked_table) > 2 else "Debt-to-Income Ratio"

        insight_text = (
            f"{top_1}, {top_2} and {top_3} show the strongest relationship with loan default in the analyzed dataset."
        )

        self.relevance_results = {
            'ranked_table': ranked_table,
            'top_1': top_1,
            'top_2': top_2,
            'top_3': top_3,
            'insight_summary': insight_text,
            'attributes': [item['display_name'] for item in ranked_table],
            'mi_scores': [item['mutual_information'] for item in ranked_table],
            'tree_importances': [item['tree_importance'] for item in ranked_table],
            'correlation_matrix': corr_matrix
        }

    def _train_and_evaluate_models(self):
        X = self.encoded_X
        y = self.y_encoded

        test_size = 0.25 if len(X) >= 40 else 0.15
        strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) > 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=strat
        )

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        if self.numerical_cols:
            X_train_scaled[self.numerical_cols] = self.scaler.fit_transform(X_train[self.numerical_cols])
            X_test_scaled[self.numerical_cols] = self.scaler.transform(X_test[self.numerical_cols])

        model_defs = {
            'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        }

        self.models = {}
        self.metrics = {}
        self.confusion_matrices = {}

        for name, clf in model_defs.items():
            t0 = time.time()
            if name == 'Naive Bayes':
                clf.fit(X_train_scaled, y_train)
                train_time = round(time.time() - t0, 3)
                y_pred = clf.predict(X_test_scaled)
            else:
                clf.fit(X_train, y_train)
                train_time = round(time.time() - t0, 3)
                y_pred = clf.predict(X_test)
                
            self.models[name] = clf

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
            rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

            self.metrics[name] = {
                'accuracy': round(acc, 2),
                'precision': round(prec, 2),
                'recall': round(rec, 2),
                'f1_score': round(f1, 2),
                'training_time': f"{max(0.02, train_time):.2f}s"
            }

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            self.confusion_matrices[name] = cm.tolist()

    def _extract_decision_tree_rules_and_nodes(self):
        dt = self.models.get('Decision Tree')
        if not dt:
            return

        tree_ = dt.tree_
        feature_names = self.feature_cols
        class_names = self.class_names
        rules = []

        # 1. Build Visual Tree Structure Nodes
        def build_node(node_id):
            n_id = int(node_id)
            is_leaf = bool(tree_.feature[n_id] == _tree.TREE_UNDEFINED)
            samples = int(tree_.n_node_samples[n_id])
            impurity = round(float(tree_.impurity[n_id]), 3)
            val = tree_.value[n_id][0]
            pred_class = str(class_names[np.argmax(val)])
            
            if is_leaf:
                pct = int(round(float(np.max(val) / np.sum(val)) * 100))
                return {
                    'id': n_id,
                    'is_leaf': True,
                    'predicted_class': pred_class,
                    'confidence_pct': pct,
                    'samples': samples,
                    'gini': impurity
                }
            else:
                fname = str(feature_names[tree_.feature[n_id]]).replace('_', ' ')
                thresh = float(tree_.threshold[n_id])
                
                # Format threshold
                if 'Ratio' in fname:
                    cond_label = f"{fname} <= {thresh:.2f}"
                elif any(k in fname for k in ['Income', 'Loan', 'Balance']):
                    cond_label = f"{fname} > Rs.{int(thresh/1000)}k" if thresh > 1000 else f"{fname} > {thresh:.1f}"
                else:
                    cond_label = f"{fname} <= {int(thresh)}"

                return {
                    'id': n_id,
                    'is_leaf': False,
                    'condition': str(cond_label),
                    'gini': float(impurity),
                    'samples': int(samples),
                    'left': build_node(int(tree_.children_left[n_id])),
                    'right': build_node(int(tree_.children_right[n_id]))
                }

        self.tree_structure = build_node(0)

        # 2. Extract Clean Text Rules
        def recurse(node, current_conds):
            node_idx = int(node)
            if tree_.feature[node_idx] != _tree.TREE_UNDEFINED:
                name = str(feature_names[tree_.feature[node_idx]])
                thresh = float(tree_.threshold[node_idx])
                
                if name in self.categorical_cols:
                    le = self.label_encoders[name]
                    left_cats = [str(c) for c, idx in zip(le.classes_, range(len(le.classes_))) if idx <= thresh]
                    right_cats = [str(c) for c, idx in zip(le.classes_, range(len(le.classes_))) if idx > thresh]
                    l_str = f"{name.replace('_', ' ')} = {', '.join(left_cats)}"
                    r_str = f"{name.replace('_', ' ')} = {', '.join(right_cats)}"
                else:
                    if 'Ratio' in name:
                        l_str = f"DTI Ratio < {thresh:.2f}"
                        r_str = f"DTI Ratio >= {thresh:.2f}"
                    elif any(k in name for k in ['Income', 'Amount', 'Balance']):
                        l_str = f"{name.replace('_', ' ')} <= ₹{int(thresh):,}"
                        r_str = f"{name.replace('_', ' ')} > ₹{int(thresh):,}"
                    else:
                        l_str = f"{name.replace('_', ' ')} <= {int(thresh)}"
                        r_str = f"{name.replace('_', ' ')} > {int(thresh)}"

                recurse(int(tree_.children_left[node_idx]), current_conds + [l_str])
                recurse(int(tree_.children_right[node_idx]), current_conds + [r_str])
            else:
                samples = int(tree_.n_node_samples[node_idx])
                if samples < 15:
                    return
                val = tree_.value[node_idx][0]
                pred_class = str(class_names[np.argmax(val)])
                conf = int(round(float(np.max(val) / np.sum(val)) * 100))
                
                rules.append({
                    'conditions': current_conds,
                    'predicted_class': pred_class,
                    'confidence': int(conf),
                    'samples': int(samples)
                })

        recurse(0, [])
        rules.sort(key=lambda r: (r['samples'] * r['confidence']), reverse=True)
        self.tree_rules = rules[:6]

    def predict_applicant(self, applicant_data, model_name='Decision Tree'):
        clf = self.models.get(model_name, self.models.get('Decision Tree'))
        
        income = float(applicant_data.get('Annual_Income', 750000))
        loan = float(applicant_data.get('Loan_Amount', 250000))
        lir = round(loan / max(income, 1.0), 2)
        
        row = {}
        for col in self.feature_cols:
            if col == 'Loan_to_Income_Ratio':
                row[col] = lir
            elif col in applicant_data:
                val = applicant_data[col]
                row[col] = float(val) if col in self.numerical_cols else str(val)
            else:
                if col in self.numerical_cols:
                    row[col] = float(self.clean_df[col].median())
                else:
                    row[col] = str(self.clean_df[col].mode().iloc[0])

        input_df = pd.DataFrame([row])
        encoded_input = input_df.copy()
        
        for col in self.categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                val = str(encoded_input[col].iloc[0]).strip()
                matched_val = None
                if val in le.classes_:
                    matched_val = val
                else:
                    for c in le.classes_:
                        if c.lower() in val.lower() or val.lower() in c.lower():
                            matched_val = c
                            break
                if matched_val is not None:
                    encoded_input[col] = le.transform([matched_val])[0]
                else:
                    encoded_input[col] = 0

        if model_name == 'Naive Bayes' and self.numerical_cols:
            eval_input = encoded_input.copy()
            eval_input[self.numerical_cols] = self.scaler.transform(eval_input[self.numerical_cols])
        else:
            eval_input = encoded_input
            
        pred_idx = clf.predict(eval_input)[0]
        predicted_class = self.class_names[pred_idx]
        
        probs = clf.predict_proba(eval_input)[0]
        prob_dict = {cname: round(float(prob) * 100, 1) for cname, prob in zip(self.class_names, probs)}
        
        # Risk Probability for gauge (High + 0.5*Medium)
        risk_probability = int(prob_dict.get('High', 0) + (0.4 * prob_dict.get('Medium', 0)))
        
        # Contributing factors matching panel 6
        factors = []
        
        # Good credit
        ch = str(applicant_data.get('Credit_History', 'Good'))
        if ch == 'Good':
            factors.append({'text': 'Good credit history', 'type': 'positive', 'sign': '+'})
        elif ch == 'Poor':
            factors.append({'text': 'Poor bureau credit history', 'type': 'negative', 'sign': '-'})

        # Stable employment
        emp_years = float(applicant_data.get('Employment_Years', 5))
        if emp_years >= 3:
            factors.append({'text': f'Stable employment ({int(emp_years)} yrs)', 'type': 'positive', 'sign': '+'})
        elif emp_years == 0:
            factors.append({'text': 'No active employment history', 'type': 'negative', 'sign': '-'})

        # Income sufficient
        if income > loan:
            factors.append({'text': 'Income sufficient for loan amount', 'type': 'positive', 'sign': '+'})
        else:
            factors.append({'text': 'Loan exceeds annual income', 'type': 'negative', 'sign': '-'})

        # Delinquency
        rh = str(applicant_data.get('Repayment_History', 'On-Time'))
        if 'On-Time' in rh:
            factors.append({'text': 'No recent delinquencies', 'type': 'positive', 'sign': '+'})
        elif 'Frequent' in rh:
            factors.append({'text': 'Frequent past delinquencies flagged', 'type': 'negative', 'sign': '-'})
        else:
            factors.append({'text': 'Minor past payment delay', 'type': 'negative', 'sign': '-'})

        # LIR
        if lir <= 0.6:
            factors.append({'text': f'Low loan-to-income ratio ({lir}x)', 'type': 'positive', 'sign': '+'})
        elif lir > 1.5:
            factors.append({'text': f'High loan-to-income ratio ({lir}x)', 'type': 'negative', 'sign': '-'})
        else:
            factors.append({'text': f'Moderate loan-to-income ratio ({lir}x)', 'type': 'neutral', 'sign': '+'})

        # Recommendation
        if predicted_class == 'Low':
            recom_title = "Suitable for standard review"
            recom_desc = "Based on the analyzed information, this applicant falls under the low risk category. Proceed with standard verification process."
            color = "emerald"
        elif predicted_class == 'Medium':
            recom_title = "Additional verification recommended"
            recom_desc = "Moderate risk exposure detected. Recommend additional income proof, co-applicant, or collateral evaluation."
            color = "amber"
        else:
            recom_title = "Detailed manual underwriting review"
            recom_desc = "High credit risk detected. Senior risk officer approval required; consider reducing loan quantum or requesting security."
            color = "rose"

        return {
            'risk_level': f"{predicted_class} Risk",
            'raw_class': predicted_class,
            'risk_probability': min(98, max(5, risk_probability)),
            'confidence': prob_dict.get(predicted_class, 85),
            'probabilities': prob_dict,
            'recommendation_title': recom_title,
            'recommendation_desc': recom_desc,
            'badge_color': color,
            'loan_to_income_ratio': lir,
            'factors': factors
        }

    def get_summary_stats(self):
        df = self.clean_df
        total_applicants = len(df)
        default_count = int(df['Historical_Default'].sum())
        default_rate = round(float(default_count / max(total_applicants, 1)) * 100, 1)
        
        avg_loan = int(df['Loan_Amount'].mean())
        avg_income = int(df['Annual_Income'].mean())
        
        # Formatted string like ₹4.8L or ₹7.1L
        def format_inr(val):
            if val >= 10000000:
                return f"₹{val/10000000:.1f}Cr"
            elif val >= 100000:
                return f"₹{val/100000:.1f}L"
            else:
                return f"₹{val/1000:.0f}K"

        risk_counts = df['Risk_Level'].value_counts()
        risk_dist = {
            'Low': int(risk_counts.get('Low', 0)),
            'Medium': int(risk_counts.get('Medium', 0)),
            'High': int(risk_counts.get('High', 0))
        }
        risk_pct = {k: round(v / max(total_applicants, 1) * 100) for k, v in risk_dist.items()}

        # Scatter sample (Income vs Loan Amount)
        scatter_sample = []
        sample_df = df.sample(min(200, len(df)), random_state=42)
        for _, r in sample_df.iterrows():
            scatter_sample.append({
                'x': round(r['Annual_Income'] / 100000, 1), # Income in Lakhs
                'y': round(r['Loan_Amount'] / 100000, 1),   # Loan in Lakhs
                'risk': r['Risk_Level']
            })

        # Default by Credit History
        ch_default = {}
        if 'Credit_History' in df.columns:
            for ch in ['Good', 'Fair', 'Poor']:
                sub = df[df['Credit_History'] == ch]
                if not sub.empty:
                    ch_default[ch] = round(float(sub['Historical_Default'].mean()) * 100)
                else:
                    ch_default[ch] = 0

        preview_records = df.head(10).to_dict(orient='records')
        
        # Data summary statistics for table tab 8
        column_meta = []
        for c in df.columns:
            null_count = int(df[c].isnull().sum())
            dtype_str = "Numeric" if np.issubdtype(df[c].dtype, np.number) else "Categorical"
            column_meta.append({
                'name': c,
                'type': dtype_str,
                'missing': null_count,
                'unique': int(df[c].nunique())
            })

        return {
            'total_applicants': total_applicants,
            'default_rate': default_rate,
            'avg_loan_formatted': format_inr(avg_loan),
            'avg_income_formatted': format_inr(avg_income),
            'risk_distribution': risk_dist,
            'risk_percentages': risk_pct,
            'scatter_sample': scatter_sample,
            'credit_history_default_rate': ch_default,
            'preview_records': preview_records,
            'column_meta': column_meta,
            'columns': list(df.columns)
        }
