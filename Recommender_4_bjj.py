import numpy as np
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox
from PyQt5.QtCore import Qt
import sys
from surprise import SVD, Dataset, Reader
import pandas as pd
import random
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from recommender_bjj_func import *



if __name__ == '__main__':
    db = Database()
    app = QApplication(sys.argv)
    ex = BJJRecommenderGUI(db)
    sys.exit(app.exec_())
