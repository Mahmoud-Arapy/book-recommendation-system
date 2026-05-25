USE recommendation_db;

CREATE TABLE books (
    book_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    keywords TEXT
);

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    preferred_categories VARCHAR(255)
);

CREATE TABLE interactions (
    interaction_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    book_id INT,
    action VARCHAR(50),
    rating INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);