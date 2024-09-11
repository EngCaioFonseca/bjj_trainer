import numpy as np
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox
from PyQt5.QtCore import Qt
import sys
from surprise import SVD, Dataset, Reader
import pandas as pd
import random

# Define BJJ techniques with categories
techniques = {
    "Upper Body": ["Gi: Armbar", "Gi: Triangle Choke", "No-Gi: Kimura", "No-Gi: Rear Naked Choke"],
    "Lower Body": ["Gi: Kneebar", "Gi: Footlock", "No-Gi: Heelhook", "No-Gi: Ankle Lock"],
    "Submissions": ["Gi: Ezekiel Choke", "Gi: Bow and Arrow Choke", "No-Gi: Guillotine", "No-Gi: D'Arce Choke"],
    "Takedowns": ["Gi: Double Leg", "Gi: Single Leg", "No-Gi: Ankle Pick", "No-Gi: Blast Double"],
    "Guard": ["Gi: Closed Guard", "Gi: Spider Guard", "No-Gi: Butterfly Guard", "No-Gi: X-Guard"]
}

# Simple user database (user_id, skill, level, preferred_techniques)
users = [
    (1, "guard", "beginner", ["Gi: Closed Guard", "No-Gi: Butterfly Guard"]),
    (2, "submissions", "intermediate", ["Gi: Armbar", "No-Gi: Heelhook"]),
    (3, "takedowns", "advanced", ["Gi: Double Leg", "No-Gi: Ankle Pick"])
]


class MatrixFactorizationRecommender:
    def __init__(self, users):
        self.users = users
        self.user_ids = [user[0] for user in users]
        self.techniques = set()
        for user in users:
            self.techniques.update(user[3])
        self.technique_ids = {technique: i for i, technique in enumerate(self.techniques)}
        
        # Prepare data for Surprise
        ratings = []
        for user in users:
            for technique in self.techniques:
                rating = 1 if technique in user[3] else 0
                ratings.append((user[0], self.technique_ids[technique], rating))
        
        # Convert ratings to a pandas DataFrame
        df = pd.DataFrame(ratings, columns=['user', 'item', 'rating'])
        
        reader = Reader(rating_scale=(0, 1))
        self.data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
        
        # Train the SVD model
        self.model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.model.fit(self.data.build_full_trainset())

    def recommend_techniques(self, user_techniques, k=5):
        user_id = max(self.user_ids) + 1
        user_ratings = []
        for technique in self.techniques:
            rating = 1 if technique in user_techniques else 0
            user_ratings.append((user_id, self.technique_ids[technique], rating))
        
        # Add user ratings to the dataset and retrain
        df_new = pd.DataFrame(user_ratings, columns=['user', 'item', 'rating'])
        new_data = Dataset.load_from_df(df_new[['user', 'item', 'rating']], Reader(rating_scale=(0, 1)))
        new_trainset = new_data.build_full_trainset()
        self.model.fit(new_trainset)
        
        # Get recommendations
        recommendations = []
        for technique in self.techniques:
            if technique not in user_techniques:
                predicted_rating = self.model.predict(user_id, self.technique_ids[technique]).est
                recommendations.append((technique, predicted_rating))
        
        return [technique for technique, _ in sorted(recommendations, key=lambda x: x[1], reverse=True)[:k]]

class BJJRecommenderGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.recommender = MatrixFactorizationRecommender(users)

    def initUI(self):
        layout = QVBoxLayout()

        # Skill input
        skill_layout = QHBoxLayout()
        skill_layout.addWidget(QLabel('Primary BJJ Skill:'))
        self.skill_input = QLineEdit()
        skill_layout.addWidget(self.skill_input)
        layout.addLayout(skill_layout)

        # Level input
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel('BJJ Level:'))
        self.level_input = QComboBox()
        self.level_input.addItems(['Beginner', 'Intermediate', 'Advanced'])
        level_layout.addWidget(self.level_input)
        layout.addLayout(level_layout)

        # Techniques input
        techniques_layout = QHBoxLayout()
        techniques_layout.addWidget(QLabel('Preferred Techniques (comma-separated):'))
        self.techniques_input = QLineEdit()
        techniques_layout.addWidget(self.techniques_input)
        layout.addLayout(techniques_layout)

        # Weaknesses/Goals input
        weaknesses_layout = QHBoxLayout()
        weaknesses_layout.addWidget(QLabel('Weaknesses/Goals (comma-separated):'))
        self.weaknesses_input = QLineEdit()
        weaknesses_layout.addWidget(self.weaknesses_input)
        layout.addLayout(weaknesses_layout)

        # Submit button
        self.submit_button = QPushButton('Get Recommendations')
        self.submit_button.clicked.connect(self.get_recommendations)
        layout.addWidget(self.submit_button)

        # Results display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display)

        self.setLayout(layout)
        self.setWindowTitle('BJJ Technique Recommender')
        self.show()

    def get_recommendations(self):
        skill = self.skill_input.text()
        level = self.level_input.currentText().lower()
        techniques = [t.strip() for t in self.techniques_input.text().split(',')]
        weaknesses = [w.strip() for w in self.weaknesses_input.text().split(',')]

        recommended_techniques = self.recommender.recommend_techniques(techniques)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BJJRecommenderGUI()
    sys.exit(app.exec_())