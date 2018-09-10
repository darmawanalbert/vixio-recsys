# Flask-related Import
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from flaskext.mysql import MySQL

# ML-related Import
import pandas as pd
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering, KNNBasic, KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Flask Configuration
application = Flask(__name__)

# MySQL Configuration
mysql = MySQL()
# application.config['MYSQL_DATABASE_USER'] = 'root'
# application.config['MYSQL_DATABASE_PASSWORD'] = 'root'
# application.config['MYSQL_DATABASE_DB'] = 'vixio'
# application.config['MYSQL_DATABASE_HOST'] = 'localhost'
# application.config['MYSQL_DATABASE_PORT'] = 8889
application.config['MYSQL_DATABASE_USER'] = 'zeonzero1996'
application.config['MYSQL_DATABASE_PASSWORD'] = 'DragonKnight25'
application.config['MYSQL_DATABASE_DB'] = 'vixio'
application.config['MYSQL_DATABASE_HOST'] = 'vixioinstance.cmkr2i0amwng.ap-southeast-1.rds.amazonaws.com'
application.config['MYSQL_DATABASE_PORT'] = 3306
mysql.init_app(application)

# REST API Configuration
api = Api(application)

# PRODUCTION Class Definition
class MostPopular(Resource):
	def get(self):
		cursor = mysql.connect().cursor()
		cursor.execute("SELECT id FROM stories ORDER BY played DESC LIMIT 10")
		data = cursor.fetchall()
		most_popular_list = []
		for item in data:
			most_popular_list.append(item[0])
		return jsonify(recommendations = most_popular_list)

class NewReleases(Resource):
	def get(self):
		cursor = mysql.connect().cursor()
		cursor.execute("SELECT id FROM stories ORDER BY created_at DESC LIMIT 10")
		data = cursor.fetchall()
		new_releases_list = []
		for item in data:
			new_releases_list.append(item[0])
		return jsonify(recommendations = new_releases_list)

class Personalized(Resource):
	def get(self, user_id):
		# SQL query
		conn = mysql.connect()
		cursor = conn.cursor()
		df = pd.read_sql_query("SELECT * FROM story_reviews", conn)

		# Data and Model
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(df[['user_id', 'story_id', 'star']], reader)
		model = SlopeOne()
		
		# Training
		training_set = data.build_full_trainset()
		model.fit(training_set)

		# Prediction
		anti_training_set = training_set.build_anti_testset()
		prediction_set = [x for x in anti_training_set if x[0]==user_id]
		predictions = model.test(prediction_set)
		
		# Return Top N Recommendations
		n = 10
		predictions.sort(key=lambda x:x.est, reverse=True)
		top_n_predictions = predictions[:n]

		story_recommendations = []
		
		for predictionItem in top_n_predictions:
			story_recommendations.append(predictionItem.iid)

		return jsonify(recommendations = story_recommendations)

class SimilarStories(Resource):
	def get(self, item_id):
		# SQL query
		conn = mysql.connect()
		cursor = conn.cursor()
		# STEP 1 : KNN WITH MSD
		df = pd.read_sql_query("SELECT * FROM story_reviews", conn)

		# Data and Model
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(df[['user_id', 'story_id', 'star']], reader)
		sim_options = {'name': 'pearson_baseline', 'user_based': False}

		model = KNNBaseline(sim_options=sim_options)
		
		# Training
		training_set = data.build_full_trainset()
		model.fit(training_set)

		item_inner_id = model.trainset.to_inner_iid(item_id)
		item_neighbors_inner = model.get_neighbors(item_inner_id, k=10)
		item_neighbors = [model.trainset.to_raw_iid(inner_id) for inner_id in item_neighbors_inner]

		# STEP 2 : CASCADE IT WITH TF-IDF
		df_stories = pd.read_sql_query("SELECT * FROM stories", conn)

		# Model
		tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
		tf_idf_matrix = tf.fit_transform(df_stories['title'])
		cosine_similarities = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

		# Retrieve similar items
		cosine_similarities_row = cosine_similarities[item_id-1]
		recommendations_list = []
		n = 10
		for i in range(n):
			recommendations_list.append((item_neighbors[i], cosine_similarities_row[item_neighbors[i]-1]))
				
		recommendations_list.sort(key=lambda x:x[1], reverse=True)
		formatted_recommendations_list = [item[0] for item in recommendations_list]

		# Return K Nearest Neighbors
		return jsonify(recommendations = formatted_recommendations_list)

# DEVELOPMENT Class Definition
class User(Resource):
	def get(self, algorithm, user_id):
		# SQL query
		conn = mysql.connect()
		cursor = conn.cursor()
		df = pd.read_sql_query("SELECT * FROM story_reviews", conn)

		# Data and Model
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(df[['user_id', 'story_id', 'star']], reader)

		if algorithm=='svd':
			print('Using SVD')
			model = SVD()
		elif algorithm=='svdpp':
			print('Using SVD++')
			model = SVDpp()
		elif (algorithm=='nmf'):
			print('Using NMF')
			model = NMF()
		elif (algorithm=='slopeone'):
			print('Using Slope One')
			model = SlopeOne()
		elif (algorithm=='coclustering'):
			print('Using Co-Clustering')
			model = CoClustering()
		else:
			print('Using SVD')
			model = SVD()
		
		# Training
		training_set = data.build_full_trainset()
		model.fit(training_set)

		# Prediction
		anti_training_set = training_set.build_anti_testset()
		prediction_set = [x for x in anti_training_set if x[0]==user_id]
		predictions = model.test(prediction_set)

		# TESTING : Run 5-fold Cross Validation using Root Mean Square Error and Mean Average Error
		# cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
		
		# Return Top N Recommendations
		n = 10
		predictions.sort(key=lambda x:x.est, reverse=True)
		top_n_predictions = predictions[:n]

		story_recommendations = []
		
		for predictionItem in top_n_predictions:
			story_recommendations.append(predictionItem.iid)

		return jsonify(recommendations = story_recommendations)

class Story(Resource):
	def get(self, algorithm, item_id):
		# SQL query
		conn = mysql.connect()
		cursor = conn.cursor()
		df = pd.read_sql_query("SELECT * FROM story_reviews", conn)

		# Data and Model
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(df[['user_id', 'story_id', 'star']], reader)

		if algorithm == 'pearson':
			sim_options = {'name': 'pearson', 'user_based': False}
		elif algorithm == 'cosine':
			sim_options = {'name': 'cosine', 'user_based': False}
		elif algorithm == 'pearsonbaseline':
			sim_options = {'name': 'pearson_baseline', 'user_based': False}
		elif algorithm == 'msd':
			sim_options = {'name': 'msd', 'user_based': False}
		else:
			sim_options = {'name': 'pearson_baseline', 'user_based': False}

		model = KNNBaseline(sim_options=sim_options)
		
		# Training
		training_set = data.build_full_trainset()
		model.fit(training_set)

		# TESTING
		# cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

		item_inner_id = model.trainset.to_inner_iid(item_id)
		item_neighbors_inner = model.get_neighbors(item_inner_id, k=10)
		item_neighbors = [model.trainset.to_raw_iid(inner_id) for inner_id in item_neighbors_inner]

		# Return K Nearest Neighbors
		return jsonify(recommendations = item_neighbors)

class Content(Resource):
	def get(self, item_id):
		# SQL query
		conn = mysql.connect()
		cursor = conn.cursor()

		# Retrieve all stories
		df = pd.read_sql_query("SELECT * FROM stories", conn)

		# Model
		tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
		tf_idf_matrix = tf.fit_transform(df['title'])
		cosine_similarities = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

		# Retrieve similar items
		cosine_similarities_row = cosine_similarities[item_id-1]
		recommendations_list = []
		n = 10
		for i in range(len(cosine_similarities_row)):
			recommendations_list.append((i+1, cosine_similarities_row[i]))
				
		recommendations_list.sort(key=lambda x:x[1], reverse=True)
		recommendations_list_n = recommendations_list[1:n+1]
		formatted_recommendations_list_n = [item[0] for item in recommendations_list_n]

		# Return K Nearest Neighbors
		return jsonify(recommendations = formatted_recommendations_list_n)

# PRODUCTION API Route
api.add_resource(MostPopular, '/mostpopular')
api.add_resource(NewReleases, '/newreleases')
api.add_resource(Personalized, '/personalized/<int:user_id>')
api.add_resource(SimilarStories, '/similarstories/<int:item_id>')


# DEVELOPMENT API Route
# Personalized Story Recommendations
api.add_resource(User, '/user/<algorithm>/<int:user_id>')
# Similar Stories
api.add_resource(Story, '/story/<algorithm>/<int:item_id>')
api.add_resource(Content, '/content/<int:item_id>')

# Running the Flask App
if __name__ == "__main__":
	application.run()
#    application.run(host='0.0.0.0', port=8080, debug=True)
