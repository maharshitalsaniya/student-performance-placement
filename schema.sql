-- MySQL 8+ schema. The app currently uses SQLite by default.
CREATE DATABASE IF NOT EXISTS student_performance;
USE student_performance;
CREATE TABLE IF NOT EXISTS students (
 id BIGINT PRIMARY KEY AUTO_INCREMENT,
 student_name VARCHAR(150) NOT NULL,
 previous_percentage DECIMAL(5,2) NOT NULL,
 attendance DECIMAL(5,2) NOT NULL,
 study_hours DECIMAL(5,2) NOT NULL,
 projects INT NOT NULL,
 internships INT NOT NULL,
 certifications INT NOT NULL,
 coding_skill DECIMAL(4,2) NOT NULL,
 technical_skill DECIMAL(4,2) NOT NULL,
 communication_skill DECIMAL(4,2) NOT NULL,
 aptitude_score DECIMAL(5,2) NOT NULL,
 created_at DATETIME NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
 id BIGINT PRIMARY KEY AUTO_INCREMENT,
 student_id BIGINT NOT NULL,
 performance_label VARCHAR(40) NOT NULL,
 placement_probability DECIMAL(5,2) NOT NULL,
 placement_status VARCHAR(40) NOT NULL,
 strong_areas TEXT NOT NULL,
 weak_areas TEXT NOT NULL,
 recommendations TEXT NOT NULL,
 created_at DATETIME NOT NULL,
 FOREIGN KEY (student_id) REFERENCES students(id)
);
