import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def test_db_connection():
    database_url = os.getenv("DATABASE_URL")
    
    connection = None
    cursor = None

    try:
        if database_url:
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            connection = psycopg2.connect(database_url)
        else:
            USER = os.getenv("user")
            PASSWORD = os.getenv("password")
            HOST = os.getenv("host")
            PORT = os.getenv("port")
            DBNAME = os.getenv("dbname")

            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            )
            
        print("Connection successful!")
        
        # Create a cursor to execute SQL queries
        cursor = connection.cursor()
        
        # Example query
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        print("Current Time:", result)

    except Exception as e:
        print(f"Failed to connect: {e}")
        
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            print("Connection closed.")

if __name__ == "__main__":
    test_db_connection()
