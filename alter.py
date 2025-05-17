import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Add the column if it doesn't already exist
try:
    c.execute('ALTER TABLE predictions ADD COLUMN Income_Category INTEGER;')
    print("Column added successfully.")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")  # This will show if the column already exists

conn.commit()
conn.close()
