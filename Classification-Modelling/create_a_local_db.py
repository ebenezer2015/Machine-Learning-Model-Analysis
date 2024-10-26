import sqlite3
import time

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=10)  # Setting a timeout
    except sqlite3.Error as e:
        print(e)
    return conn

def execute_query(conn, query, data=None):
    try:
        cur = conn.cursor()
        if data:
            cur.execute(query, data)
        else:
            cur.execute(query)
        conn.commit()
        return cur
    except sqlite3.OperationalError as e:
        if 'database is locked' in str(e):
            print("Database is locked, retrying...")
            time.sleep(1)
            return execute_query(conn, query, data)
        else:
            raise

conn = create_connection('customer.db')

create_table_query = '''CREATE TABLE IF NOT EXISTS users
                        (
                        user_id INTEGER PRIMARY KEY, 
                        first_name TEXT, 
                        last_name TEXT, 
                        email TEXT, 
                        address TEXT, 
                        city TEXT, 
                        state TEXT, 
                        age INTEGER
                        )'''

execute_query(conn, create_table_query)

insert_data_query = "INSERT INTO users (first_name, last_name, email, address, city, state, age) VALUES (?, ?, ?, ?, ?, ?, ?)"
execute_query(conn, insert_data_query, ('Adeboye', 'bade','bade@hotmail.com','11 Afolaju str','Lagos Island','Lagos', 30))
execute_query(conn, insert_data_query, ('Peter', 'Doll','p.doll@gmail.com','103 Churchill Crescent','NewPort','UK', 64))
execute_query(conn, insert_data_query, ('Chris', 'Haba','chris.haba@yahoomail.com','111 Afolaju str','Lagos Island','Lagos', 33))
execute_query(conn, insert_data_query, ('Petri', 'Asjru','p_asjru@gmail.com','103 Hakaniemi Crescent','Helsinki','Finland', 44))
execute_query(conn, insert_data_query, ('boye', 'badru','b.badru1984e@hotmail.com','101 Fola Agoro Layout','Lagos Island','Lagos', 42))
execute_query(conn, insert_data_query, ('James', 'Justine','jay_jay@gmail.com','103 Churchill Crescent','NewPort','UK', 64))
execute_query(conn, insert_data_query, ('Juha', 'Teemu','teemu@ymail.com','12 Cole str','London','London', 27))
execute_query(conn, insert_data_query, ('James', 'Buki','buki@hotmail.com','17A Vieraskuja', 'Espoo', 'Helsinki', 32))

select_query = "SELECT * FROM users"
rows = execute_query(conn, select_query).fetchall()

for row in rows:
    print(row)

conn.close()