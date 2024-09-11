import numpy as np
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QMessageBox
from PyQt5.QtCore import Qt
import sys
from surprise import SVD, Dataset, Reader
import pandas as pd
import random
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Define BJJ techniques with categories
techniques = {
    "Upper Body": ["Gi: Armbar", "Gi: Triangle Choke", "No-Gi: Kimura", "No-Gi: Rear Naked Choke"],
    "Lower Body": ["Gi: Kneebar", "Gi: Footlock", "No-Gi: Heelhook", "No-Gi: Ankle Lock", "No-Gi: Toe hold", "Gi: Toe hold"],
    "Submissions": ["Gi: Ezekiel Choke", "Gi: Bow and Arrow Choke", "No-Gi: Guillotine", "No-Gi: D'Arce Choke", "Triangle"],
    "Takedowns": ["Gi: Double Leg", "Gi: Single Leg", "No-Gi: Ankle Pick", "No-Gi: Blast Double"],
    "Guard": ["Gi: Closed Guard", "Gi: Spider Guard", "No-Gi: Butterfly Guard", "No-Gi: X-Guard"]
}

# Simple user database (user_id, skill, level, preferred_techniques)
users = [
    (1, "guard", "beginner", ["Gi: Closed Guard", "No-Gi: Butterfly Guard"]),
    (2, "submissions", "intermediate", ["Gi: Armbar", "No-Gi: Heelhook"]),
    (3, "takedowns", "advanced", ["Gi: Double Leg", "No-Gi: Ankle Pick"])
]

class Database:
    def __init__(self, db_name='bjj_recommender.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    skill TEXT,
                    level TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS techniques (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    user_id INTEGER,
                    technique_id INTEGER,
                    rating INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (technique_id) REFERENCES techniques (id),
                    PRIMARY KEY (user_id, technique_id)
                )
            ''')

    def add_user(self, username, password, skill, level):
        hashed_password = generate_password_hash(password)
        with self.conn:
            self.conn.execute('INSERT INTO users (username, password, skill, level) VALUES (?, ?, ?, ?)',
                              (username, hashed_password, skill, level))

    def verify_user(self, username, password):
        cursor = self.conn.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user[1], password):
            return user[0]
        return None

    def get_user_info(self, user_id):
        cursor = self.conn.execute('SELECT skill, level FROM users WHERE id = ?', (user_id,))
        return cursor.fetchone()

    def add_technique(self, name, category):
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO techniques (name, category) VALUES (?, ?)', (name, category))

    def get_technique_id(self, name):
        cursor = self.conn.execute('SELECT id FROM techniques WHERE name = ?', (name,))
        result = cursor.fetchone()
        return result[0] if result else None

    def add_rating(self, user_id, technique_name, rating):
        technique_id = self.get_technique_id(technique_name)
        if technique_id is None:
            return False
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO ratings (user_id, technique_id, rating) VALUES (?, ?, ?)',
                              (user_id, technique_id, rating))
        return True

    def get_user_ratings(self, user_id):
        cursor = self.conn.execute('''
            SELECT t.name, r.rating
            FROM ratings r
            JOIN techniques t ON r.technique_id = t.id
            WHERE r.user_id = ?
        ''', (user_id,))
        return cursor.fetchall()

    def get_all_ratings(self):
        cursor = self.conn.execute('''
            SELECT r.user_id, t.name, r.rating
            FROM ratings r
            JOIN techniques t ON r.technique_id = t.id
        ''')
        return cursor.fetchall()

    def close(self):
        self.conn.close()



class MatrixFactorizationRecommender:
    def __init__(self, db):
        self.db = db
        self.update_model()

    def update_model(self):
        ratings = self.db.get_all_ratings()
        if not ratings:
            # If there are no ratings yet, we can't build a model
            self.model = None
            return

        df = pd.DataFrame(ratings, columns=['user', 'item', 'rating'])
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
        self.model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.model.fit(self.data.build_full_trainset())

    def recommend_techniques(self, user_id, k=5):
        if self.model is None:
            # If we don't have a model, return random techniques
            all_techniques = [technique for category in techniques.values() for technique in category]
            return random.sample(all_techniques, k)

        user_ratings = dict(self.db.get_user_ratings(user_id))
        all_techniques = set(technique for category in techniques.values() for technique in category)
        unrated_techniques = all_techniques - set(user_ratings.keys())
        
        predictions = []
        for technique in unrated_techniques:
            predicted_rating = self.model.predict(user_id, technique).est
            predictions.append((technique, predicted_rating))
        
        return [technique for technique, _ in sorted(predictions, key=lambda x: x[1], reverse=True)[:k]]

class BJJRecommenderGUI(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.user_id = None
        self.recommender = MatrixFactorizationRecommender(db)
        self.initUI()


    def initUI(self):
        main_layout = QVBoxLayout()

        # Login/Register section
        login_layout = QHBoxLayout()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        login_button = QPushButton('Login')
        register_button = QPushButton('Register')
        login_button.clicked.connect(self.login)
        register_button.clicked.connect(self.register)
        login_layout.addWidget(QLabel('Username:'))
        login_layout.addWidget(self.username_input)
        login_layout.addWidget(QLabel('Password:'))
        login_layout.addWidget(self.password_input)
        login_layout.addWidget(login_button)
        login_layout.addWidget(register_button)
        main_layout.addLayout(login_layout)

        # Skill input
        skill_layout = QHBoxLayout()
        skill_layout.addWidget(QLabel('Primary BJJ Skill:'))
        self.skill_input = QLineEdit()
        skill_layout.addWidget(self.skill_input)
        main_layout.addLayout(skill_layout)

        # Level input
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel('BJJ Level:'))
        self.level_input = QComboBox()
        self.level_input.addItems(['Beginner', 'Intermediate', 'Advanced'])
        level_layout.addWidget(self.level_input)
        main_layout.addLayout(level_layout)

        # Techniques input
        techniques_layout = QHBoxLayout()
        techniques_layout.addWidget(QLabel('Preferred Techniques (comma-separated):'))
        self.techniques_input = QLineEdit()
        techniques_layout.addWidget(self.techniques_input)
        main_layout.addLayout(techniques_layout)

        # Weaknesses/Goals input
        weaknesses_layout = QHBoxLayout()
        weaknesses_layout.addWidget(QLabel('Weaknesses/Goals (comma-separated):'))
        self.weaknesses_input = QLineEdit()
        weaknesses_layout.addWidget(self.weaknesses_input)
        main_layout.addLayout(weaknesses_layout)

        # Rate technique section
        rate_layout = QHBoxLayout()
        self.technique_to_rate = QLineEdit()
        self.rating_input = QComboBox()
        self.rating_input.addItems(['1', '2', '3', '4', '5'])
        rate_button = QPushButton('Rate Technique')
        rate_button.clicked.connect(self.rate_technique)
        rate_layout.addWidget(QLabel('Technique to rate:'))
        rate_layout.addWidget(self.technique_to_rate)
        rate_layout.addWidget(QLabel('Rating:'))
        rate_layout.addWidget(self.rating_input)
        rate_layout.addWidget(rate_button)
        main_layout.addLayout(rate_layout)

        # Submit button
        self.submit_button = QPushButton('Get Recommendations')
        self.submit_button.clicked.connect(self.get_recommendations)
        main_layout.addWidget(self.submit_button)

        # Results display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        main_layout.addWidget(self.results_display)

        self.setLayout(main_layout)
        self.setWindowTitle('BJJ Technique Recommender')
        self.show()


    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        user_id = self.db.verify_user(username, password)
        if user_id:
            self.user_id = user_id
            QMessageBox.information(self, 'Success', 'Logged in successfully!')
        else:
            QMessageBox.warning(self, 'Error', 'Invalid username or password')

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        skill = self.skill_input.text()
        level = self.level_input.currentText().lower()
        try:
            self.db.add_user(username, password, skill, level)
            QMessageBox.information(self, 'Success', 'Registered successfully! You can now log in.')
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, 'Error', 'Username already exists')

    def rate_technique(self):
        if not self.user_id:
            QMessageBox.warning(self, 'Error', 'Please log in first')
            return
        technique = self.technique_to_rate.text()
        rating = int(self.rating_input.currentText())
        if self.db.add_rating(self.user_id, technique, rating):
            QMessageBox.information(self, 'Success', 'Rating added successfully!')
            self.recommender.update_model()
        else:
            QMessageBox.warning(self, 'Error', 'Invalid technique name')

    def get_recommendations(self):
        if not self.user_id:
            QMessageBox.warning(self, 'Error', 'Please log in first')
            return
        
        skill, level = self.db.get_user_info(self.user_id)
        weaknesses = [w.strip() for w in self.weaknesses_input.text().split(',')]

        recommended_techniques = self.recommender.recommend_techniques(self.user_id)
        periodized_plan = create_periodized_plan(skill, level, weaknesses)
        weekly_plan = create_weekly_plan(skill, level, recommended_techniques, weaknesses)


        results = f"Recommended techniques:\n"
        for technique in recommended_techniques:
            results += f"- {technique}\n"
        
        results += "\nPeriodized Training Plan:\n"
        for week in periodized_plan:
            results += f"{week}\n"
        
        results += "\nWeekly Training Plan:\n"
        for session in weekly_plan:
            results += f"{session}\n"

        self.results_display.setText(results)

def create_periodized_plan(skill, level, weaknesses):
    weeks = 4 if level == "beginner" else 6 if level == "intermediate" else 8
    plan = []
    for week in range(1, weeks + 1):
        plan.append(f"Week {week}:")
        plan.append("  Monday: High Intensity - Upper Body Focus")
        plan.append("  Tuesday: Technical - Lower Body Focus")
        plan.append("  Wednesday: Active Recovery")
        plan.append("  Thursday: High Volume - Submissions Focus")
        plan.append("  Friday: Technical - Guard and Takedowns")
        plan.append("  Saturday: Competition Simulation")
        plan.append("  Sunday: Rest")
        
        # Add focus on weaknesses/goals
        if weaknesses:
            weakness_focus = random.choice(weaknesses)
            plan.append(f"  Weakness Focus: {weakness_focus}")
    return plan

def create_weekly_plan(skill, level, recommended_techniques, weaknesses):
    days = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
    focus_areas = ["Upper Body", "Lower Body", "Submissions", "Guard", "Takedowns"]
    weekly_plan = []
    
    for day, focus in zip(days, focus_areas):
        intensity = "High Intensity" if day in ["Monday", "Thursday"] else "Technical"
        focus_techniques = [t for t in recommended_techniques if any(t.startswith(f"{gi_nogi}: ") for gi_nogi in ["Gi", "No-Gi"]) and t.split(": ")[1] in techniques[focus]]
        if not focus_techniques:
            focus_techniques = [f"{gi_nogi}: {t}" for gi_nogi in ["Gi", "No-Gi"] for t in random.sample(techniques[focus], 2)]
        else:
            focus_techniques = focus_techniques[:2]
        
        weekly_plan.append(f"{day} ({intensity} - {focus} Focus):")
        weekly_plan.append(f"  1. {focus_techniques[0]}")
        weekly_plan.append(f"  2. {focus_techniques[1]}")
        weekly_plan.append(f"  3. Conditioning: {random.choice(['HIIT', 'Strength Training', 'Cardio'])}")
        
        # Add detailed explanations for each focus area
        if focus == "Upper Body":
            weekly_plan.append("  Upper Body Focus Explanation:")
            weekly_plan.append("   - Emphasizes techniques that primarily use arms, shoulders, and chest")
            weekly_plan.append("   - Includes submissions like armbars, kimuras, and chokes")
            weekly_plan.append("   - Drill grip fighting and upper body control positions")
        elif focus == "Lower Body":
            weekly_plan.append("  Lower Body Focus Explanation:")
            weekly_plan.append("   - Concentrates on techniques involving legs and hips")
            weekly_plan.append("   - Includes leg locks, sweeps, and guard retention drills")
            weekly_plan.append("   - Practice hip mobility and leg dexterity exercises")
        elif focus == "Submissions":
            weekly_plan.append("  Submissions Focus Explanation:")
            weekly_plan.append("   - Focuses on finishing techniques from various positions")
            weekly_plan.append("   - Drill submission setups and transitions between submissions")
            weekly_plan.append("   - Practice both gi and no-gi specific submissions")
        elif focus == "Guard":
            weekly_plan.append("  Guard Focus Explanation:")
            weekly_plan.append("   - Emphasizes guard retention, recovery, and attacks")
            weekly_plan.append("   - Practice different guard types (closed, open, half)")
            weekly_plan.append("   - Drill sweeps and submissions from guard positions")
        elif focus == "Takedowns":
            weekly_plan.append("  Takedowns Focus Explanation:")
            weekly_plan.append("   - Concentrates on techniques to bring the fight to the ground")
            weekly_plan.append("   - Practice both gi and no-gi takedowns")
            weekly_plan.append("   - Drill takedown defense and sprawls")
        
        # Add focus on weaknesses/goals
        if weaknesses and random.random() < 0.5:  # 50% chance to add a weakness focus
            weakness_focus = random.choice(weaknesses)
            weekly_plan.append(f"  4. Weakness Focus: {weakness_focus}")
            weekly_plan.append(f"   - Dedicate extra time to drilling and situational sparring")
            weekly_plan.append(f"   - Focus on specific techniques or positions related to this weakness")
    
    weekly_plan.append("Wednesday: Active Recovery (Light drilling, Yoga, or Mobility work)")
    weekly_plan.append("Sunday: Rest")
    
    return weekly_plan