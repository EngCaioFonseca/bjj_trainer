# BJJ Technique Recommender System

## Overview

The BJJ Technique Recommender System is an advanced Python application designed to provide personalized Brazilian Jiu-Jitsu (BJJ) training recommendations. It utilizes collaborative filtering and matrix factorization to suggest techniques based on user preferences, skill level, and feedback. The system also generates periodized training plans and weekly schedules, taking into account the user's weaknesses or specific training goals.

## Features

- User registration and login system
- Personalized technique recommendations using matrix factorization
- Dynamic user database for storing profiles and technique ratings
- Periodized training plan generation
- Detailed weekly training schedule creation
- Incorporation of user-specified weaknesses or training goals
- Technique rating system for continuous improvement of recommendations
- Graphical User Interface (GUI) for easy interaction
- Comprehensive explanations of recommended techniques and training focus areas

## Requirements

- Python 3.7+
- NumPy
- scikit-learn
- PyQt5
- pandas
- scikit-surprise
- SQLite3
- Werkzeug

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bjj-recommender.git
   cd bjj-recommender
   ```

2. Install the required packages:
   ```
   pip install numpy scikit-learn PyQt5 pandas scikit-surprise werkzeug
   ```

## Usage

Run the main script:

python Recommender_4_bjj.py


This will launch the GUI application. Follow these steps:

1. Register a new account or log in with existing credentials
2. Enter your primary BJJ skill (e.g., "guard", "submissions", "takedowns")
3. Select your BJJ level (Beginner, Intermediate, Advanced)
4. Input your preferred techniques, separated by commas
5. List your weaknesses or specific training goals, separated by commas
6. Click "Get Recommendations"
7. After trying recommended techniques, you can rate them for improved future recommendations

The system will display:
- Recommended techniques with explanations
- A periodized training plan
- A detailed weekly training schedule

## How It Works

1. **User Management**: The system uses SQLite to store user profiles, including their skills, level, and technique ratings.

2. **Matrix Factorization Recommender**: Utilizes collaborative filtering to suggest techniques based on user preferences and similar users' ratings.

3. **Periodized Training Plan**: Generates a multi-week plan tailored to the user's skill level and incorporating focus on specified weaknesses.

4. **Weekly Training Schedule**: Creates a detailed weekly plan with specific focus areas for each day, including recommended techniques and explanations.

5. **Weakness Integration**: Incorporates user-specified weaknesses or goals into both the periodized plan and weekly schedule.

6. **Feedback Loop**: Users can rate techniques, which are stored in the database and used to improve future recommendations.

## File Structure

- `Recommender_4_bjj.py`: Main script to run the application
- `recommender_bjj_func.py`: Contains core functionality, database operations, and GUI implementation
- `bjj_recommender.db`: SQLite database file storing user data and ratings

## Customization

You can customize the system by modifying:
- The `techniques` dictionary in `recommender_bjj_func.py` to add or change BJJ techniques
- The `create_periodized_plan` and `create_weekly_plan` functions to adjust training structures
- The `MatrixFactorizationRecommender` class to tweak the recommendation algorithm

## Contributing

Contributions to improve the BJJ Technique Recommender System are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Thanks to the scikit-surprise team for their matrix factorization implementation
- PyQt5 for providing the GUI framework
- The BJJ community for inspiration and technique knowledge
