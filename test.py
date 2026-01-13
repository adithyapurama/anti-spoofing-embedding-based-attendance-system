import sqlite3, numpy as np
conn = sqlite3.connect("models/users.db")
row = conn.execute("SELECT embedding, dim FROM users WHERE user_id='Adi'").fetchone()
emb = np.frombuffer(row[0], dtype=np.float32)
print("SQLite emb len:", len(emb), "first5:", emb[:5])
