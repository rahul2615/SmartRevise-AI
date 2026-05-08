import sqlite3

def add_columns():
    conn = sqlite3.connect('instance/database_v2.db')
    cursor = conn.cursor()
    
    try:
        print("Adding last_active_date...")
        cursor.execute("ALTER TABLE user ADD COLUMN last_active_date DATE")
    except sqlite3.OperationalError as e:
        print(f"Skipping last_active_date: {e}")

    try:
        print("Adding current_streak...")
        cursor.execute("ALTER TABLE user ADD COLUMN current_streak INTEGER DEFAULT 0")
    except sqlite3.OperationalError as e:
        print(f"Skipping current_streak: {e}")

    try:
        print("Adding total_study_minutes...")
        cursor.execute("ALTER TABLE user ADD COLUMN total_study_minutes INTEGER DEFAULT 0")
    except sqlite3.OperationalError as e:
        print(f"Skipping total_study_minutes: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == '__main__':
    add_columns()
