# Movie Recommendation System

This is a Django-based project for recommending movies using various algorithms including Content-Based Filtering, K-Nearest Neighbors (KNN), and Hybrid Approaches.

## Features

- **Content-Based Filtering:** Recommends movies based on the attributes of the movie itself.
- **K-Nearest Neighbors (KNN):** Recommends movies based on the preferences of users with similar taste profiles.
- **Hybrid Approaches:** Combines both content-based and collaborative filtering techniques for more accurate and diverse suggestions.

## Requirements

- Python  3.12
- Django 5.0.6
- Other dependencies listed in `requirements.txt`

## Setup Instructions

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Apply migrations:**
    ```sh
    pytho manage.py migrate
    ```

5. **Run the development server:**
    ```sh
    python manage.py runserver
    ```

6. **Access the application:**
    Open your browser and go to `http://127.0.0.1:8000/`.
