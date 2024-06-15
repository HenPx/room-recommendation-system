from database.connection import create_connection

def login_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
    data = cursor.fetchone()
    conn.close()
    return data