import requests

def test_similarstories():
	url = 'http://localhost:5000/similarstories/1'
	response = requests.get(url)
	assert response.status_code == 200
	recommendationList = response.json()['recommendations']
	assert type(recommendationList) == type([])
	assert len(recommendationList) == 10
