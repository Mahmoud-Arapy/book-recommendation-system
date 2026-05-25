# Book Recommendation System

A hybrid book recommendation system that provides personalized book suggestions using both **Content-Based Filtering** and **Collaborative Filtering** techniques.
The project is built with **Python**, **FastAPI**, **MySQL**, and **Machine Learning** concepts such as **TF-IDF** and **Cosine Similarity**.

---

## Features

* Hybrid Recommendation System
* Content-Based Recommendations
* Collaborative Filtering Recommendations
* Cold Start Handling
* REST API using FastAPI
* MySQL Database Integration
* User Interaction Tracking
* Dynamic Recommendation Refreshing
* CORS Support for Frontend Integration

---

## Technologies Used

* Python
* FastAPI
* MySQL
* SQLAlchemy
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity
* Uvicorn

---

## Recommendation Strategies

### 1. Content-Based Filtering

Recommends books based on:

* Book title
* Keywords
* Description
* Author

Using:

* TF-IDF Vectorization
* Cosine Similarity

---

### 2. Collaborative Filtering

Recommends books based on:

* Similar users
* User ratings
* Interaction history

---

### 3. Hybrid Recommendation

The system combines:

* Collaborative Filtering
* Content-Based Filtering
* Cold Start Strategy

To generate more accurate recommendations.

---

## API Endpoints

### Get Recommendations

```http
GET /api/recommendations/{user_id}
```

### Add User Interaction

```http
POST /api/interactions
```

---

## Project Structure

```bash
recommendation-system/
│
├── main.py
├── .env
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/book-recommendation-system.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=your_database
```

### Run the Server

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

---

## Example Response

```json
{
  "success": true,
  "user_id": 1,
  "data": {
    "strategy": "Collaborative",
    "books": [
      {
        "book_id": 5,
        "title": "Machine Learning Basics",
        "category": "AI"
      }
    ]
  }
}
```

---

## Future Improvements

* Deploy the API online
* Add Authentication & Authorization
* Improve Recommendation Accuracy
* Add Deep Learning Recommendation Models
* Build Full Frontend Dashboard
* Docker Support

---

## Author

Mahmoud Araby Rifai

Computer Science Student passionate about:

* Machine Learning
* Data Analysis
* Backend Development
* Recommendation Systems
