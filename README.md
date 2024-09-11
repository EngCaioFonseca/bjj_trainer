# BJJ Technique Recommender System

## Overview

This project implements a recommender system for Brazilian Jiu-Jitsu (BJJ) techniques. It uses a combination of user input, collaborative filtering, and a knowledge base of BJJ techniques to provide personalized recommendations and create training plans for practitioners.

## Features

- Personalized technique recommendations based on user preferences and skill level
- Periodized training plan generation
- Weekly training schedule creation
- Categorization of techniques by body area and gi/no-gi variants
- Simple user database for collaborative filtering

## Requirements

- Python 3.7+
- NumPy
- scikit-learn

Install the required packages using:

pip install numpy scikit-learn


## Project Structure

- `Recommender_4_bjj.py`: Main script containing the implementation
- `README.md`: This file, containing project documentation

## Key Components

1. `techniques`: Dictionary of BJJ techniques categorized by body area and gi/no-gi variants
2. `users`: Simple database of user profiles for collaborative filtering
3. `get_user_input()`: Function to collect user preferences and skill level
4. `create_user_vector()`: Creates a binary vector representation of user techniques
5. `recommend_techniques()`: Generates technique recommendations using collaborative filtering
6. `create_periodized_plan()`: Generates a periodized training plan based on user level
7. `create_weekly_plan()`: Creates a detailed weekly training schedule

## Usage

Run the main script:

python Recommender_4_bjj.py


The script will prompt the user for input, generate recommendations, and display a training plan.

## How It Works

1. The user inputs their primary BJJ skill, level, and preferred techniques.
2. The system creates a user vector based on the input.
3. Collaborative filtering is used to find similar users and recommend techniques.
4. A periodized training plan is generated based on the user's skill level.
5. A detailed weekly training schedule is created, incorporating recommended techniques.

## Customization

You can customize various aspects of the system:
- Add or modify techniques in the `techniques` dictionary
- Adjust the user database in the `users` list
- Modify the training plan structure in `create_periodized_plan()` and `create_weekly_plan()`


## Contributing

Contributions to improve the recommender system are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is open-source and available under the MIT License.

