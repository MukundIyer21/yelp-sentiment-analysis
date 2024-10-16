# Yelp Review Star Prediction

This project focuses on predicting Yelp review star ratings based on review text. The dataset contains thousands of Yelp reviews, and the goal is to build a model that can accurately predict the star rating (1 to 5) associated with each review. The project integrates machine learning algorithms for text classification with MongoDB for efficient data storage and retrieval.

## Project Overview

The Yelp Review Star Prediction project leverages natural language processing (NLP) techniques to process review text and train machine learning models to predict star ratings. The review text is preprocessed through text cleaning, followed by vectorization using Term Frequency-Inverse Document Frequency (TF-IDF). The models tested in this project include Logistic Regression, Support Vector Machine (SVM), and Random Forest, among others. The dataset is stored in MongoDB, and the project is structured to facilitate both training and prediction phases.

## Features

- **Text Preprocessing**: The review text is cleaned by removing special characters, converting to lowercase, and eliminating stop words.
- **TF-IDF Vectorization**: The cleaned text is converted into numerical vectors using the TF-IDF technique, enabling machine learning models to interpret the text data.
- **Machine Learning Models**: Multiple models are used to predict review star ratings, including Logistic Regression, Support Vector Machines (SVM), and Random Forest.
- **MongoDB Integration**: MongoDB is used for storing the Yelp review dataset, providing an efficient way to manage and query large datasets.
- **Word Cloud Visualization**: The project can generate word clouds that visually display the most frequent words in positive, neutral, and negative reviews.
- **Cross-Validation**: Various models are evaluated using cross-validation to select the most suitable model for prediction.

## Technologies Used

- **Python**: The main programming language used for model development and data processing.
- **Pandas**: For data manipulation and analysis.
- **scikit-learn**: To perform machine learning tasks like TF-IDF vectorization and model training.
- **MongoDB**: A NoSQL database used to store and manage the large dataset of Yelp reviews.
- **PyMongo**: Python MongoDB client used to interface with the MongoDB database.
- **TfidfVectorizer**: A feature extraction method from scikit-learn used to convert text into vectors.

## Dataset

The dataset consists of Yelp reviews, stored in MongoDB, and includes the following columns:
- **review_id**: Unique identifier for the review.
- **text**: The content of the review.
- **stars**: The star rating (1 to 5) given by the user.
- **cleaned_text**: Preprocessed version of the review text used for model training.
- **Other Metadata**: Additional columns such as user_id, business_id, and date may be available for further analysis.

## Project Structure

- **Data Loading**: The dataset is loaded from a MongoDB instance, where it has been stored after preprocessing.
- **Text Preprocessing**: A `clean_text` function is used to clean the review text by removing non-alphabetic characters and stopwords.
- **Model Training**: The project includes a variety of machine learning models, trained using TF-IDF vectorized review text and the corresponding star ratings.
- **Prediction**: A prediction function is included to allow new reviews to be classified and assigned a predicted star rating.

