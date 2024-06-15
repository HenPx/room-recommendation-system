import pymysql

def create_connection():
    return pymysql.connect(
        host='localhost',
        user='root',       # Sesuaikan dengan user MySQL Anda
        password='',       # Sesuaikan dengan password MySQL Anda
        db='recommendation_system'
    )