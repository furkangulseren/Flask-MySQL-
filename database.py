"""
import mysql.connector
database = mysql.connector.connect(host="localhost",
                                    user="root",
                                    password = "1234",
                                    database = "database")
cursor = database.cursor()
cursor.execute("DATABASE COMMANDS")
"""

def Articles():
    articles = [
        {
            'id': 1,
            'title': 'Article One',
            'body': 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do.',
            'author': 'Brad Traversy',
            'created_date': '04-25-2017'
        },
        {
            'id': 2,
            'title': 'Article Two',
            'body': 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do.',
            'author': 'Brad Traversy',
            'created_date': '04-25-2017'
        },
        {
            'id': 3,
            'title': 'Article Three',
            'body': 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do.',
            'author': 'Brad Traversy',
            'created_date': '04-25-2017'
        }
    ]
    return articles
