import os
import time
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
from prettytable import PrettyTable
from collections import Counter
import psutil
from google.colab import drive

# Suppress warnings
warnings.filterwarnings("ignore")

# Mount Google Drive (only needs to be done once)
drive.mount('/content/drive', force_remount=True)

# Define a common results directory for all algorithms
RESULTS_DIR = "/content/drive/MyDrive/SEM8/DataMining/PROJECT/User_Management_Analysis/engagement_analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper function to save metrics to CSV
def save_metrics(algorithm_name, execution_time, memory_usage):
    metrics_df = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'Time': [execution_time],
        'Memory': [memory_usage]
    })
    metrics_file = f"{RESULTS_DIR}/{algorithm_name}_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
