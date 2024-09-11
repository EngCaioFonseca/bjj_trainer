# BJJ Technique Recommender System

## Overview

The BJJ Technique Recommender System is a Python-based application that provides personalized Brazilian Jiu-Jitsu (BJJ) training recommendations. It uses collaborative filtering and matrix factorization to suggest techniques based on user preferences and skill level. The system also generates periodized training plans and weekly schedules, taking into account the user's weaknesses or specific training goals.

## Features

- Personalized technique recommendations using matrix factorization
- Periodized training plan generation
- Detailed weekly training schedule creation
- Incorporation of user-specified weaknesses or training goals
- Graphical User Interface (GUI) for easy interaction
- Comprehensive explanations of recommended techniques and training focus areas

## Requirements

- Python 3.7+
- NumPy
- scikit-learn
- PyQt5
- pandas
- scikit-surprise

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bjj-recommender.git
   cd bjj-recommender
   ```

2. Install the required packages:
   ```
   pip install numpy scikit-learn PyQt5 pandas scikit-surprise
   ```

## Usage

Run the main script:

python Recommender_4_bjj.py


This will launch the GUI application. Follow these steps:

1. Enter your primary BJJ skill (e.g., "guard", "submissions", "takedowns")
2. Select your BJJ level (Beginner, Intermediate, Advanced)
3. Input your preferred techniques, separated by commas
4. List your weaknesses or specific training goals, separated by commas
5. Click "Get Recommendations"

The system will then display:
- Recommended techniques with explanations
- A periodized training plan
- A detailed weekly training schedule

## How It Works

1. **Matrix Factorization Recommender**: Uses collaborative filtering to suggest techniques based on user preferences and similar users' data.

2. **Periodized Training Plan**: Generates a multi-week plan tailored to the user's skill level and incorporating focus on specified weaknesses.

3. **Weekly Training Schedule**: Creates a detailed weekly plan with specific focus areas for each day, including recommended techniques and explanations.

4. **Weakness Integration**: Incorporates user-specified weaknesses or goals into both the periodized plan and weekly schedule.

## Customization

You can customize the system by modifying:
- The `techniques` dictionary to add or change BJJ techniques
- The `users` list to expand the user database for better recommendations
- The `create_periodized_plan` and `create_weekly_plan` functions to adjust training structures


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

