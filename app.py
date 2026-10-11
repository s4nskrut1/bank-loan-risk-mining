import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from mining_engine import MiningEngine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# State: Initially loaded with None until user clicks or uploads
engine = None
current_dataset_name = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global engine, current_dataset_name
    if engine is None:
        return jsonify({'status': 'no_data'})
    return jsonify({
        'status': 'loaded',
        'dataset_name': current_dataset_name,
        'records_count': len(engine.clean_df)
    })

@app.route('/api/summary', methods=['GET'])
def get_summary():
    global engine
    if engine is None:
        return jsonify({'status': 'no_data'}), 400
    stats = engine.get_summary_stats()
    return jsonify({'status': 'success', 'data': stats})

@app.route('/api/attribute-relevance', methods=['GET'])
def get_attribute_relevance():
    global engine
    if engine is None:
        return jsonify({'status': 'no_data'}), 400
    return jsonify({'status': 'success', 'data': engine.relevance_results})

@app.route('/api/models', methods=['GET'])
def get_models():
    global engine
    if engine is None:
        return jsonify({'status': 'no_data'}), 400
    return jsonify({
        'status': 'success',
        'metrics': engine.metrics,
        'confusion_matrices': engine.confusion_matrices,
        'class_names': engine.class_names,
        'tree_rules': engine.tree_rules,
        'tree_structure': engine.tree_structure
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    global engine
    if engine is None:
        return jsonify({'status': 'no_data'}), 400
    try:
        data = request.get_json(force=True)
        model_name = data.get('model_name', 'Decision Tree')
        applicant_data = data.get('applicant', {})
        result = engine.predict_applicant(applicant_data, model_name)
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global engine, current_dataset_name
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file was provided in upload.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Please choose a CSV file.'}), 400
        
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Invalid file format. Please upload a .csv file.'}), 400

    try:
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)
        
        uploaded_df = pd.read_csv(save_path)
        if len(uploaded_df) < 5:
            return jsonify({
                'status': 'error',
                'message': 'Dataset is too small (needs at least 10 rows for mining analysis).'
            }), 400

        engine = MiningEngine(uploaded_df)
        current_dataset_name = file.filename
        
        return jsonify({
            'status': 'success',
            'message': f"Dataset '{file.filename}' successfully uploaded and mined ({len(uploaded_df)} records)!",
            'records_count': len(uploaded_df),
            'dataset_name': current_dataset_name
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Mining pipeline error: {str(e)}"}), 500

@app.route('/api/load-sample-benchmark', methods=['POST'])
def load_sample_benchmark():
    global engine, current_dataset_name
    benchmark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'historical_loan_records.csv')
    if os.path.exists(benchmark_path):
        df = pd.read_csv(benchmark_path)
        engine = MiningEngine(df)
        current_dataset_name = "historical_loan_records.csv"
        return jsonify({
            'status': 'success',
            'message': f"Benchmark dataset loaded ({len(df)} records)!",
            'records_count': len(df),
            'dataset_name': current_dataset_name
        })
    return jsonify({'status': 'error', 'message': 'Sample dataset not found.'}), 404

@app.route('/api/load-preset/<preset_name>', methods=['POST'])
def load_preset(preset_name):
    global engine, current_dataset_name
    preset_map = {
        'benchmark': 'historical_loan_records.csv',
        'tier1': 'sample_datasets/retail_banking_tier1.csv',
        'microfinance': 'sample_datasets/microfinance_rural_lending.csv',
        'fintech': 'sample_datasets/fintech_digital_loans.csv',
        'stressed': 'sample_datasets/stressed_credit_portfolio.csv'
    }
    
    rel_path = preset_map.get(preset_name)
    if not rel_path:
        return jsonify({'status': 'error', 'message': f"Unknown preset '{preset_name}'"}), 404
        
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    if not os.path.exists(full_path):
        return jsonify({'status': 'error', 'message': f"File not found: {rel_path}"}), 404
        
    df = pd.read_csv(full_path)
    engine = MiningEngine(df)
    current_dataset_name = os.path.basename(rel_path)
    
    return jsonify({
        'status': 'success',
        'message': f"Loaded preset '{current_dataset_name}' ({len(df)} records)!",
        'records_count': len(df),
        'dataset_name': current_dataset_name
    })

@app.route('/api/download-template', methods=['GET'])
def download_template():
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_loan_template.csv')
    return send_file(template_path, as_attachment=True, download_name='historical_loan_template.csv')

@app.route('/api/download-dataset', methods=['GET'])
@app.route('/api/download-dataset/<filename>', methods=['GET'])
def download_dataset(filename=None):
    if filename:
        # Check sample_datasets
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_datasets', filename)
        if os.path.exists(sample_path):
            return send_file(sample_path, as_attachment=True, download_name=filename)
            
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'historical_loan_records.csv')
    return send_file(dataset_path, as_attachment=True, download_name='historical_loan_records.csv')

if __name__ == '__main__':
    print("CrediMine Server ready on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
