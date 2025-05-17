from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import joblib
import numpy as np
from flask import Response


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # change this to a strong secret in production

# Load model
model = joblib.load('model/best_model.pkl')

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    Total_Trans_Amt REAL,
                    Total_Trans_Ct REAL,
                    Total_Ct_Chng_Q4_Q1 REAL,
                    Total_Revolving_Bal REAL,
                    Avg_Utilization_Ratio REAL,
                    Total_Relationship_Count REAL,
                    Total_Amt_Chng_Q4_Q1 REAL,
                    Customer_Age INTEGER,
                    Credit_Limit REAL,
                    Avg_Open_To_Buy REAL,
                    prediction TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    username = request.form['username']
    email = request.form['email']
    password = generate_password_hash(request.form['password'])

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (first_name, last_name, username, email, password) VALUES (?, ?, ?, ?, ?)',
                  (first_name, last_name, username, email, password))
        conn.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth'))
    except sqlite3.IntegrityError:
        flash('Username or email already taken.', 'danger')
        return redirect(url_for('auth'))
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[5], password):
        session['user_id'] = user[0]
        session['username'] = user[3]  # username index
        return redirect(url_for('dashboard'))  # Redirect to dashboard after login
    else:
        flash('Invalid username or password.', 'danger')
        return redirect(url_for('auth'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('auth'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    prediction = None
    if request.method == 'POST':
        try:
            # Extract top 10 important features from form
            user_inputs = [
                float(request.form['Total_Trans_Amt']),
                float(request.form['Total_Trans_Ct']),
                float(request.form['Total_Ct_Chng_Q4_Q1']),
                float(request.form['Total_Revolving_Bal']),
                float(request.form['Avg_Utilization_Ratio']),
                float(request.form['Total_Relationship_Count']),
                float(request.form['Total_Amt_Chng_Q4_Q1']),
                float(request.form['Customer_Age']),
                float(request.form['Credit_Limit']),
                float(request.form['Avg_Open_To_Buy'])
            ]

            # Get Income_Category from form
            income_category = int(request.form['Income_Category'])

            # Default values for the remaining features
            default_values = [
                1,  # Gender (encoded)
                0,  # Dependent_count
                2,  # Education_Level (encoded)
                1,  # Marital_Status (encoded)
                income_category,  # Income_Category from form
                1,  # Card_Category (encoded)
                36.0,  # Months_on_book
                2.0,   # Months_Inactive_12_mon
                3.0    # Contacts_Count_12_mon
            ]

            full_features = [
                user_inputs[7],       # Customer_Age
                default_values[0],    # Gender
                default_values[1],    # Dependent_count
                default_values[2],    # Education_Level
                default_values[3],    # Marital_Status
                default_values[4],    # Income_Category
                default_values[5],    # Card_Category
                default_values[6],    # Months_on_book
                user_inputs[5],       # Total_Relationship_Count
                default_values[7],    # Months_Inactive_12_mon
                default_values[8],    # Contacts_Count_12_mon
                user_inputs[8],       # Credit_Limit
                user_inputs[3],       # Total_Revolving_Bal
                user_inputs[9],       # Avg_Open_To_Buy
                user_inputs[6],       # Total_Amt_Chng_Q4_Q1
                user_inputs[0],       # Total_Trans_Amt
                user_inputs[1],       # Total_Trans_Ct
                user_inputs[2],       # Total_Ct_Chng_Q4_Q1
                user_inputs[4]        # Avg_Utilization_Ratio
            ]

            result = model.predict([np.array(full_features)])
            prediction = 'Will Churn' if result[0] == 1 else 'Will Not Churn'

            # Save prediction to database including Income_Category
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''INSERT INTO predictions (
                            user_id, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,
                            Total_Revolving_Bal, Avg_Utilization_Ratio, Total_Relationship_Count,
                            Total_Amt_Chng_Q4_Q1, Customer_Age, Credit_Limit, Avg_Open_To_Buy,
                            Income_Category, prediction
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], *user_inputs, income_category, prediction))
            conn.commit()
            conn.close()

            flash(f'Prediction result: {prediction}', 'info')
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))

    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    user_id = session['user_id']
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    prediction_type = request.args.get('prediction_type')

    filters = "user_id = ?"
    params = [user_id]

    if start_date:
        filters += " AND DATE(created_at) >= ?"
        params.append(start_date)
    if end_date:
        filters += " AND DATE(created_at) <= ?"
        params.append(end_date)
    if prediction_type:
        filters += " AND prediction = ?"
        params.append(prediction_type)

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Summary
    c.execute(f'''
        SELECT prediction, COUNT(*) 
        FROM predictions 
        WHERE {filters}
        GROUP BY prediction
    ''', params)
    results = dict(c.fetchall())

    will_churn = results.get('Will Churn', 0)
    will_not_churn = results.get('Will Not Churn', 0)

    # Total analyzed clients
    c.execute(f'''
        SELECT COUNT(*) FROM predictions WHERE {filters}
    ''', params)
    total_analyzed = c.fetchone()[0]

    # Recent Predictions
    c.execute(f'''
        SELECT id, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,
               Total_Revolving_Bal, Avg_Utilization_Ratio, prediction, created_at
        FROM predictions
        WHERE {filters}
        ORDER BY created_at DESC
        LIMIT 10
    ''', params)
    recent_predictions = c.fetchall()

    # Time Series
    c.execute(f'''
        SELECT DATE(created_at), prediction, COUNT(*)
        FROM predictions
        WHERE {filters}
        GROUP BY DATE(created_at), prediction
    ''', params)
    churn_over_time = c.fetchall()

    # Average number of products (assumed to be Total_Relationship_Count)
    c.execute(f'''
        SELECT AVG(Total_Relationship_Count)
        FROM predictions
        WHERE {filters}
    ''', params)
    row = c.fetchone()
    avg_products = row[0] if row and row[0] is not None else 0

    # Distribution
    c.execute(f'''
        SELECT Credit_Limit, Customer_Age
        FROM predictions
        WHERE {filters}
    ''', params)
    dist_data = c.fetchall()

    # Segmentation by Age Group (only non-null ages)
    c.execute(f'''
        SELECT
          CASE
            WHEN Customer_Age IS NULL THEN 'Unknown'
            WHEN Customer_Age < 30 THEN '<30'
            WHEN Customer_Age BETWEEN 30 AND 45 THEN '30-45'
            WHEN Customer_Age BETWEEN 46 AND 60 THEN '46-60'
            ELSE '60+'
          END AS age_group,
          prediction,
          COUNT(*)
        FROM predictions
        WHERE {filters} AND Customer_Age IS NOT NULL
        GROUP BY age_group, prediction
    ''', params)
    age_group_data = c.fetchall()

    c.execute(f'''
        SELECT AVG(Total_Trans_Amt)
        FROM predictions
        WHERE {filters}
    ''', params)
    row = c.fetchone()
    avg_transaction = row[0] if row and row[0] is not None else 0

    # Calculate average utilization ratio (MUST be before conn.close)
    c.execute(f'''
        SELECT AVG(Avg_Utilization_Ratio)
        FROM predictions
        WHERE {filters}
    ''', params)
    row = c.fetchone()
    avg_utilization = row[0] if row and row[0] is not None else 0

    # Segmentation by Income Category (only non-null)
    c.execute(f'''
        SELECT Income_Category, prediction, COUNT(*)
        FROM predictions
        WHERE {filters} AND Income_Category IS NOT NULL
        GROUP BY Income_Category, prediction
    ''', params)
    income_cat_data = c.fetchall()

    conn.close()

    # Format Age Group Data
    age_groups = sorted(list(set(row[0] for row in age_group_data)))
    age_churn_counts = {ag: 0 for ag in age_groups}
    age_no_churn_counts = {ag: 0 for ag in age_groups}

    for ag, pred, count in age_group_data:
        if pred == 'Will Churn':
            age_churn_counts[ag] = count
        else:
            age_no_churn_counts[ag] = count

    age_group_summary = {
        'labels': age_groups,
        'will_churn': [age_churn_counts[ag] for ag in age_groups],
        'will_not_churn': [age_no_churn_counts[ag] for ag in age_groups]
    }

    # Format Income Category Data
    income_cats = sorted(list(set(row[0] for row in income_cat_data)))
    income_churn_counts = {ic: 0 for ic in income_cats}
    income_no_churn_counts = {ic: 0 for ic in income_cats}

    for ic, pred, count in income_cat_data:
        if pred == 'Will Churn':
            income_churn_counts[ic] = count
        else:
            income_no_churn_counts[ic] = count

    income_cat_summary = {
        'labels': income_cats,
        'will_churn': [income_churn_counts[ic] for ic in income_cats],
        'will_not_churn': [income_no_churn_counts[ic] for ic in income_cats]
    }

    return render_template(
        'dashboard.html',
        will_churn=will_churn,
        will_not_churn=will_not_churn,
        total_analyzed=total_analyzed,
        recent_predictions=recent_predictions,
        churn_over_time=churn_over_time,
        dist_data=dist_data,
        age_group_summary=age_group_summary,
        income_cat_summary=income_cat_summary,
        avg_transaction=avg_transaction,
        avg_utilization=avg_utilization,
        avg_products=avg_products
    )



@app.route('/delete/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, session['user_id']))
    conn.commit()
    conn.close()
    flash('Prediction deleted.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/export')
def export_predictions():
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT created_at, prediction, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,
               Total_Revolving_Bal, Avg_Utilization_Ratio
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    rows = c.fetchall()
    conn.close()

    # Create CSV content
    def generate():
        yield 'Date,Prediction,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Total_Revolving_Bal,Avg_Utilization_Ratio\n'
        for row in rows:
            yield ','.join(map(str, row)) + '\n'

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=predictions.csv"})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    batch_summary = None
    batch_results = None
    batch_error = None

    file = request.files.get('file')
    if not file:
        batch_error = 'Please upload a CSV file.'
    else:
        try:
            import pandas as pd
            df = pd.read_csv(file)

            required_cols = [
                'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                'Total_Revolving_Bal', 'Avg_Utilization_Ratio', 'Total_Relationship_Count',
                'Total_Amt_Chng_Q4_Q1', 'Customer_Age', 'Credit_Limit', 'Avg_Open_To_Buy'
            ]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                batch_error = f"Missing required columns: {', '.join(missing)}"
            else:
                df = df[required_cols].fillna(0)

                default_values = [1, 0, 2, 1, 3, 1, 36.0, 2.0, 3.0]

                def build_features(row):
                    return [
                        row['Customer_Age'],
                        default_values[0],
                        default_values[1],
                        default_values[2],
                        default_values[3],
                        default_values[4],
                        default_values[5],
                        default_values[6],
                        row['Total_Relationship_Count'],
                        default_values[7],
                        default_values[8],
                        row['Credit_Limit'],
                        row['Total_Revolving_Bal'],
                        row['Avg_Open_To_Buy'],
                        row['Total_Amt_Chng_Q4_Q1'],
                        row['Total_Trans_Amt'],
                        row['Total_Trans_Ct'],
                        row['Total_Ct_Chng_Q4_Q1'],
                        row['Avg_Utilization_Ratio']
                    ]

                features = df.apply(build_features, axis=1).tolist()
                preds = model.predict(features)
                df['Prediction'] = ['Will Churn' if p == 1 else 'Will Not Churn' for p in preds]

                batch_summary = df['Prediction'].value_counts().to_dict()
                batch_results = df.head(20).to_dict(orient='records')

        except Exception as e:
            batch_error = f'Error processing file: {str(e)}'

    # Render dashboard with batch prediction info, plus existing dashboard data
    return dashboard_with_batch(batch_summary, batch_results, batch_error)


def dashboard_with_batch(batch_summary=None, batch_results=None, batch_error=None):
    # Your existing dashboard data fetching logic here:
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    user_id = session['user_id']
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Churn summary
    c.execute('''
        SELECT prediction, COUNT(*) 
        FROM predictions 
        WHERE user_id = ? 
        GROUP BY prediction
    ''', (user_id,))
    results = dict(c.fetchall())
    will_churn = results.get('Will Churn', 0)
    will_not_churn = results.get('Will Not Churn', 0)

    # Recent predictions
    c.execute('''
        SELECT id, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,
               Total_Revolving_Bal, Avg_Utilization_Ratio, prediction, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 10
    ''', (user_id,))
    recent_predictions = c.fetchall()

    conn.close()

    return render_template('dashboard.html',
                           will_churn=will_churn,
                           will_not_churn=will_not_churn,
                           recent_predictions=recent_predictions,
                           batch_summary=batch_summary,
                           batch_results=batch_results,
                           batch_error=batch_error)




if __name__ == '__main__':
    app.run(debug=True)
