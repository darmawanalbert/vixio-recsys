# Flask-related Import
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from flaskext.mysql import MySQL

# ML-related Import
import pandas as pd
import random

# Flask Configuration
app = Flask(__name__)

# MySQL Configuration
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'vixio'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_PORT'] = 8889
mysql.init_app(app)

# REST API Configuration
api = Api(app)

# Class Definition
class InsertStories(Resource):
    def get(self):
        random.seed(3)
        connection = mysql.connect()
        cursor = connection.cursor()
        dataset = pd.read_csv("indonesianfolklore.csv", delimiter='.')
        for index, row in dataset.iterrows():
            year = random.randint(2000, 2018)
            month = random.randint(1,12)
            day = random.randint(1,28)
            created_string = str(year) + '-' + str(month) + '-' + str(day) + ' 02:00:29'
            cursor.execute("INSERT INTO stories (id, user_id, title, description, played, created_at) VALUES (" +  str(index+1) + ", 1, \'" + str(row['title']) + "\' , \' Sebuah Cerita Rakyat Indonesia \', " +  str(random.randint(0,100)) + ", \'" + created_string + "\')")
        connection.commit()
        return "Insert stories finished!"

class InsertReviews(Resource):
    def get(self, total_user):
        connection = mysql.connect()
        cursor = connection.cursor()
        cursor.execute("TRUNCATE TABLE story_reviews")
        cursor.execute("TRUNCATE TABLE story_playeds")
        connection.commit()
        # Perform random walk
        random.seed(3)
        for user in range(1, total_user):
            for item in range(1,250):
                random_played = random.randint(0,1)
                if random_played == 1:
                    random_star = random.randint(1,5)
                    cursor.execute("INSERT INTO story_playeds (story_id, user_id) VALUES (" + str(item) + ", " + str(user) + ")")
                    cursor.execute("INSERT INTO story_reviews (story_id, user_id, star) VALUES (" + str(item) + ", " + str(user) + ", " + str(random_star) +")")
        connection.commit()
        return "Insert reviews finished!"


# API Route
api.add_resource(InsertStories, '/story')
api.add_resource(InsertReviews, '/reviews/<int:total_user>')

# Running the Flask App
if __name__ == "__main__":
    app.run()
