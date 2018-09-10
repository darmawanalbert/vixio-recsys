# Recommender API

`recommender-api` is a RESTful API which provides interactive fiction recommendations to the users, having access to the same database as Vixio `back-end` system.

## Table of Contents

* [Installation](#installation)
* [Directory Structures](#directory-structures)
* [Usage](#usage)
* [Authors](#authors)

## Installation

Since this is a Flask application that use Python 3, there are some steps that you need to do before using this application:
1. Download and install [Python 3](https://www.python.org/)
2. Install some essential Python 3 packages using pip:
```
pip install Cython numpy scipy pandas scikit-learn scikit-surprise
```
3. Install some Flask related packages using pip:
```
pip install Flask flask-mysql flask-restful pytest requests
```

## Directory Structures

```
recommender-api
├── application.py            	# Flask Application
├── unitTest/                   # Unit test modules 
├── requirements.txt            # All Python dependencies 

```

## Usage

### Run Development Server
To run development server, simply run:
```
python3 application.py
```
Important note: DO NOT use this method in Production Server!!!

### Run Tests
To run tests, simply run:
```
pytest
```

## Authors

- [Albert Darmawan](https://github.com/darmawanalbert)
