# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:55:10 2024

@author: HP
"""

import streamlit as st
import pandas as pd
import pickle
from surprise import Dataset, Reader

# Function to load data
def load_data():
    try:
        ratings_df = pd.read_csv('ratings.csv', encoding='latin-1')
        books_df = pd.read_csv('books.csv', encoding='latin-1', low_memory=False)
        users_df = pd.read_csv('users.csv', encoding='latin-1')
    except UnicodeDecodeError as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None, None
    return ratings_df, books_df, users_df

# Function to load the trained model
def load_model():
    try:
        with open('svd_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    return model

# Function to recommend books
def recommend_books(model, user_id, n=10):
    all_books = books_df['ISBN'].unique()
    rated_books = ratings_df[ratings_df['User-ID'] == user_id]['ISBN'].values
    books_to_predict = [book for book in all_books if book not in rated_books]

    predictions = []
    for book in books_to_predict:
        pred = model.predict(user_id, book)
        predictions.append((book, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    recommended_books = []
    for isbn, rating in top_n:
        title = books_df.loc[books_df['ISBN'] == isbn, 'Book-Title'].values[0]
        recommended_books.append((title, rating))

    return recommended_books

# Load data
ratings_df, books_df, users_df = load_data()
if ratings_df is None or books_df is None or users_df is None:
    st.stop()

# Load model
model = load_model()
if model is None:
    st.stop()

# Streamlit app
st.title("Book Recommendation System")

user_id = st.number_input("Enter User ID", min_value=0, step=1)
if user_id:
    recommendations = recommend_books(model, user_id)
    st.write(f"Top 10 recommended books for user {user_id}:")
    for i, (title, rating) in enumerate(recommendations, start=1):
        st.write(f"{i}. {title} (Predicted Rating: {rating:.2f})")
