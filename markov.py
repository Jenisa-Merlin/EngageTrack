#### MARKOV CHAINS
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

warnings.filterwarnings("ignore")

# Mount Google Drive
drive.mount('/content/drive')

# Directory for storing results
RESULTS_DIR = "/content/drive/MyDrive/SEM8/DataMining/PROJECT/User_Management_Analysis/engagement_analysis_results/MarkovChain"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to preprocess data and save pickle
def preprocess_data(file):
    df = pd.read_csv(file)
    df.fillna({'Likes': 0, 'Shares': 0, 'Comments': 0, 'Engagement Rate': 0, 'Audience Age': df['Audience Age'].median()}, inplace=True)
    
    sequences = []
    sequence_details = [] # To store post type and gender for each sequence
    
    for idx, row in df.iterrows():
        sequence = []
        # Time of day first in sequence
        sequence.append('Morning Post' if pd.to_datetime(row['Post Timestamp']).hour < 12 else 'Evening Post')
        
        # Engagement metrics
        if row['Likes'] > df['Likes'].median(): sequence.append('High Likes')
        if row['Shares'] > df['Shares'].median(): sequence.append('More Shares')
        if row['Comments'] > df['Comments'].median(): sequence.append('More Comments')
        if row['Engagement Rate'] > df['Engagement Rate'].median(): sequence.append('Higher Engagement Rate')
        if row['Audience Age'] > df['Audience Age'].median(): sequence.append('Senior Adults')
        
        # Store the sequence and its metadata separately
        sequences.append(sequence)
        sequence_details.append({
            'sequence_id': idx,
            'post_type': row['Post Type'],
            'gender': row['Audience Gender'],
            'sequence': sequence
        })
    
    # Save both sequences and details as pickle
    dataset_name = file.split('/')[-1].replace('.csv', '')
    seq_pickle_path = f"{RESULTS_DIR}/{dataset_name}_sequences.pkl"
    details_pickle_path = f"{RESULTS_DIR}/{dataset_name}_sequence_details.pkl"
    
    with open(seq_pickle_path, 'wb') as f:
        pickle.dump(sequences, f)
    with open(details_pickle_path, 'wb') as f:
        pickle.dump(sequence_details, f)
    
    return sequences, sequence_details

# Function to build Markov Chain transition matrix
def build_markov_chain(sequences):
    # Identify all unique states
    all_states = set()
    for seq in sequences:
        for state in seq:
            all_states.add(state)
    
    all_states = sorted(list(all_states))
    n_states = len(all_states)
    
    # Create state to index mapping
    state_to_idx = {state: idx for idx, state in enumerate(all_states)}
    
    # Initialize transition count matrix
    transition_counts = np.zeros((n_states, n_states))
    
    # Count transitions
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            from_state = sequence[i]
            to_state = sequence[i + 1]
            from_idx = state_to_idx[from_state]
            to_idx = state_to_idx[to_state]
            transition_counts[from_idx, to_idx] += 1
    
    # Calculate transition probabilities
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        row_sum = np.sum(transition_counts[i, :])
        if row_sum > 0:
            transition_matrix[i, :] = transition_counts[i, :] / row_sum
    
    # Calculate initial state distribution
    initial_distribution = np.zeros(n_states)
    for sequence in sequences:
        if sequence:  # Check if sequence is not empty
            first_state = sequence[0]
            initial_distribution[state_to_idx[first_state]] += 1
    
    if np.sum(initial_distribution) > 0:
        initial_distribution = initial_distribution / np.sum(initial_distribution)
    
    return {
        'states': all_states,
        'state_to_idx': state_to_idx,
        'transition_matrix': transition_matrix,
        'initial_distribution': initial_distribution
    }

# Function to generate sequences using the Markov Chain
def generate_sequence(markov_chain, length=5, start_state=None):
    states = markov_chain['states']
    state_to_idx = markov_chain['state_to_idx']
    transition_matrix = markov_chain['transition_matrix']
    initial_distribution = markov_chain['initial_distribution']
    
    # Choose initial state
    if start_state is None:
        current_idx = np.random.choice(len(states), p=initial_distribution)
    else:
        current_idx = state_to_idx[start_state]
    
    sequence = [states[current_idx]]
    
    # Generate the rest of the sequence
    for _ in range(length - 1):
        # Get transition probabilities for current state
        transition_probs = transition_matrix[current_idx, :]
        
        # If all probabilities are zero, break
        if np.sum(transition_probs) == 0:
            break
        
        # Choose next state based on transition probabilities
        next_idx = np.random.choice(len(states), p=transition_probs)
        sequence.append(states[next_idx])
        current_idx = next_idx
    
    return sequence

# Function to analyze Markov Chain with demographic information
def analyze_markov_chain(markov_chain, sequence_details):
    states = markov_chain['states']
    transition_matrix = markov_chain['transition_matrix']
    state_to_idx = markov_chain['state_to_idx']
    
    # Analyze state frequencies by post type and gender
    state_demographics = {}
    for state in states:
        state_demographics[state] = {
            'post_types': Counter(),
            'genders': Counter()
        }
    
    for detail in sequence_details:
        sequence = detail['sequence']
        post_type = detail['post_type']
        gender = detail['gender']
        
        for state in sequence:
            state_demographics[state]['post_types'][post_type] += 1
            state_demographics[state]['genders'][gender] += 1
    
    # Find most common post type and gender for each state
    state_analysis = {}
    for state, demographics in state_demographics.items():
        post_types = demographics['post_types']
        genders = demographics['genders']
        
        most_common_post_type = post_types.most_common(1)[0][0] if post_types else "N/A"
        most_common_gender = genders.most_common(1)[0][0] if genders else "N/A"
        
        state_analysis[state] = {
            'most_common_post_type': most_common_post_type,
            'post_type_distribution': dict(post_types),
            'most_common_gender': most_common_gender,
            'gender_distribution': dict(genders)
        }
    
    # Calculate expected number of visits to each state
    n_states = len(states)
    fundamental_matrix = np.linalg.inv(np.eye(n_states) - transition_matrix)
    expected_visits = np.sum(fundamental_matrix, axis=0)
    
    # Add expected visits to state analysis
    for i, state in enumerate(states):
        state_analysis[state]['expected_visits'] = expected_visits[i]
    
    return state_analysis

# Function to visualize Markov Chain transition matrix
def visualize_transition_matrix(markov_chain, title):
    states = markov_chain['states']
    transition_matrix = markov_chain['transition_matrix']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=states, yticklabels=states)
    plt.title(f'Markov Chain Transition Matrix - {title}', fontsize=16)
    plt.xlabel('To State')
    plt.ylabel('From State')
    
    img_path = f"{RESULTS_DIR}/{title}_transition_matrix.png"
    plt.savefig(img_path)
    plt.close()
    
    return img_path

# Function to visualize Markov Chain as a directed graph
def visualize_markov_chain_graph(markov_chain, state_analysis, title):
    states = markov_chain['states']
    transition_matrix = markov_chain['transition_matrix']
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for state in states:
        G.add_node(state, 
                   post_type=state_analysis[state]['most_common_post_type'],
                   gender=state_analysis[state]['most_common_gender'],
                   visits=state_analysis[state]['expected_visits'])
    
    # Add edges with weights
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if transition_matrix[i, j] > 0:
                G.add_edge(from_state, to_state, weight=transition_matrix[i, j])
    
    # Create positions for nodes
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Set node colors based on expected visits
    node_sizes = [G.nodes[node]['visits'] * 500 for node in G.nodes()]
    node_colors = []
    for node in G.nodes():
        if 'High Likes' in node or 'More Shares' in node or 'More Comments' in node or 'Higher Engagement Rate' in node:
            node_colors.append('lightgreen')
        elif 'Morning Post' in node:
            node_colors.append('skyblue')
        elif 'Evening Post' in node:
            node_colors.append('orange')
        else:
            node_colors.append('lightgray')
    
    # Set edge widths based on transition probabilities
    edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    plt.figure(figsize=(14, 12))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', 
                          arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f'Markov Chain State Transition Graph - {title}', fontsize=16)
    plt.axis('off')
    
    img_path = f"{RESULTS_DIR}/{title}_markov_chain_graph.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    
    return img_path

# Function to visualize demographic distributions
def visualize_demographics(state_analysis, title):
    states = list(state_analysis.keys())
    
    # Prepare data for post type distribution
    plt.figure(figsize=(14, 10))
    
    # Post type distribution
    plt.subplot(2, 1, 1)
    post_types = set()
    for state in states:
        post_types.update(state_analysis[state]['post_type_distribution'].keys())
    
    post_types = sorted(list(post_types))
    post_type_data = []
    
    for state in states:
        state_data = []
        for post_type in post_types:
            state_data.append(state_analysis[state]['post_type_distribution'].get(post_type, 0))
        post_type_data.append(state_data)
    
    post_type_data = np.array(post_type_data)
    
    # Normalize data
    row_sums = post_type_data.sum(axis=1, keepdims=True)
    normalized_data = np.zeros_like(post_type_data, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            normalized_data[i] = post_type_data[i] / row_sums[i]
    
    # Create stacked bar chart
    bottom = np.zeros(len(states))
    for i, post_type in enumerate(post_types):
        plt.bar(states, normalized_data[:, i], bottom=bottom, label=post_type)
        bottom += normalized_data[:, i]
    
    plt.xlabel('State')
    plt.ylabel('Proportion')
    plt.title('Post Type Distribution by State')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    # Gender distribution
    plt.subplot(2, 1, 2)
    genders = set()
    for state in states:
        genders.update(state_analysis[state]['gender_distribution'].keys())
    
    genders = sorted(list(genders))
    gender_data = []
    
    for state in states:
        state_data = []
        for gender in genders:
            state_data.append(state_analysis[state]['gender_distribution'].get(gender, 0))
        gender_data.append(state_data)
    
    gender_data = np.array(gender_data)
    
    # Normalize data
    row_sums = gender_data.sum(axis=1, keepdims=True)
    normalized_data = np.zeros_like(gender_data, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            normalized_data[i] = gender_data[i] / row_sums[i]
    
    # Create stacked bar chart
    bottom = np.zeros(len(states))
    for i, gender in enumerate(genders):
        plt.bar(states, normalized_data[:, i], bottom=bottom, label=gender)
        bottom += normalized_data[:, i]
    
    plt.xlabel('State')
    plt.ylabel('Proportion')
    plt.title('Gender Distribution by State')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    img_path = f"{RESULTS_DIR}/{title}_demographic_distribution.png"
    plt.savefig(img_path)
    plt.close()
    
    return img_path

# Function to generate a comprehensive report
def generate_report(file, markov_chain, state_analysis, generated_sequences):
    dataset_name = file.split('/')[-1].replace(".csv", "")
    
    report = "# Markov Chain Analysis Report\n\n"
    
    # Overview
    report += "## Overview\n"
    report += "This report analyzes user engagement data using Markov Chains to model state transitions with a focus on post type and gender demographics.\n\n"
    
    # Dataset information
    report += f"## Dataset: {dataset_name}\n\n"
    
    # Markov Chain properties
    report += "## Markov Chain Properties\n\n"
    report += f"- Number of states: {len(markov_chain['states'])}\n"
    report += f"- States: {', '.join(markov_chain['states'])}\n\n"
    
    # Initial state distribution
    report += "### Initial State Distribution\n\n"
    for i, state in enumerate(markov_chain['states']):
        prob = markov_chain['initial_distribution'][i]
        report += f"- {state}: {prob:.4f}\n"
    report += "\n"
    
    # State analysis
    report += "## State Analysis\n\n"
    for state, analysis in state_analysis.items():
        report += f"### State: {state}\n\n"
        report += f"- Expected visits: {analysis['expected_visits']:.4f}\n"
        report += f"- Most common post type: {analysis['most_common_post_type']}\n"
        report += f"- Most common gender: {analysis['most_common_gender']}\n\n"
        
        report += "#### Post Type Distribution\n\n"
        for post_type, count in analysis['post_type_distribution'].items():
            total = sum(analysis['post_type_distribution'].values())
            percentage = (count / total) * 100 if total > 0 else 0
            report += f"- {post_type}: {count} ({percentage:.1f}%)\n"
        report += "\n"
        
        report += "#### Gender Distribution\n\n"
        for gender, count in analysis['gender_distribution'].items():
            total = sum(analysis['gender_distribution'].values())
            percentage = (count / total) * 100 if total > 0 else 0
            report += f"- {gender}: {count} ({percentage:.1f}%)\n"
        report += "\n"
    
    # Generated sequences
    report += "## Generated Sequences\n\n"
    for i, sequence in enumerate(generated_sequences):
        report += f"### Sequence {i+1}\n\n"
        report += f"- States: {' → '.join(sequence)}\n\n"
    
    # Insights and recommendations
    report += "## Insights and Recommendations\n\n"
    
    # Find high engagement states
    high_engagement_states = []
    for state in markov_chain['states']:
        if 'High Likes' in state or 'More Shares' in state or 'More Comments' in state or 'Higher Engagement Rate' in state:
            high_engagement_states.append(state)
    
    report += "### High Engagement States\n\n"
    for state in high_engagement_states:
        report += f"- {state}\n"
    report += "\n"
    
    # Find common transitions to high engagement
    report += "### Common Transitions to High Engagement\n\n"
    for i, from_state in enumerate(markov_chain['states']):
        for j, to_state in enumerate(markov_chain['states']):
            if markov_chain['transition_matrix'][i, j] > 0.5 and to_state in high_engagement_states:
                report += f"- {from_state} → {to_state} (Probability: {markov_chain['transition_matrix'][i, j]:.2f})\n"
    report += "\n"
    
    # Recommendations based on Markov Chain analysis
    report += "### Recommendations\n\n"
    
    # Identify most influential initial states
    initial_probs = markov_chain['initial_distribution']
    top_initial_states = [(markov_chain['states'][i], prob) for i, prob in enumerate(initial_probs)]
    top_initial_states.sort(key=lambda x: x[1], reverse=True)
    
    if top_initial_states:
        report += "#### Initial State Recommendations\n\n"
        for state, prob in top_initial_states[:3]:
            report += f"- Focus on {state} as it's a common starting point (Probability: {prob:.2f})\n"
        report += "\n"
    
    # Identify states with high expected visits
    top_visited_states = [(state, analysis['expected_visits']) for state, analysis in state_analysis.items()]
    top_visited_states.sort(key=lambda x: x[1], reverse=True)
    
    if top_visited_states:
        report += "#### High Frequency State Recommendations\n\n"
        for state, visits in top_visited_states[:3]:
            report += f"- Optimize for {state} as it has high expected visits ({visits:.2f})\n"
        report += "\n"
    
    # Demographic-specific recommendations
    report += "#### Demographic-Specific Recommendations\n\n"
    
    # Find states with strong gender bias
    gender_biased_states = []
    for state, analysis in state_analysis.items():
        gender_dist = analysis['gender_distribution']
        if gender_dist:
            total = sum(gender_dist.values())
            if total > 0:
                max_gender, max_count = max(gender_dist.items(), key=lambda x: x[1])
                if max_count / total > 0.7:  # More than 70% of one gender
                    gender_biased_states.append((state, max_gender, max_count / total))
    
    gender_biased_states.sort(key=lambda x: x[2], reverse=True)
    
    for state, gender, proportion in gender_biased_states[:3]:
        report += f"- {state} strongly appeals to {gender} users ({proportion:.1%})\n"
    
    # Save the report
    report_path = f"{RESULTS_DIR}/{dataset_name}_markov_chain_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Generated comprehensive analysis report: {report_path}")
    return report_path

# Function to apply Markov Chain analysis and save results
def apply_markov_chain(sequences, sequence_details, file):
    # Build Markov Chain
    markov_chain = build_markov_chain(sequences)
    
    # Analyze Markov Chain with demographic information
    state_analysis = analyze_markov_chain(markov_chain, sequence_details)
    
    # Generate sample sequences
    generated_sequences = []
    for _ in range(5):  # Generate 5 sample sequences
        generated_sequences.append(generate_sequence(markov_chain, length=5))
    
    # Save results
    dataset_name = file.split('/')[-1].replace('.csv', '')
    markov_chain_path = f"{RESULTS_DIR}/{dataset_name}_markov_chain.pkl"
    state_analysis_path = f"{RESULTS_DIR}/{dataset_name}_state_analysis.pkl"
    generated_sequences_path = f"{RESULTS_DIR}/{dataset_name}_generated_sequences.pkl"
    
    with open(markov_chain_path, 'wb') as f:
        pickle.dump(markov_chain, f)
    with open(state_analysis_path, 'wb') as f:
        pickle.dump(state_analysis, f)
    with open(generated_sequences_path, 'wb') as f:
        pickle.dump(generated_sequences, f)
    
    return markov_chain, state_analysis, generated_sequences

# Function to process file and save as pickle
def process_file(file):
    start_time = time.time()
    sequences, sequence_details = preprocess_data(file)
    markov_chain, state_analysis, generated_sequences = apply_markov_chain(sequences, sequence_details, file)
    end_time = time.time()
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
    
    # Generate visualizations
    dataset_name = file.split('/')[-1].replace(".csv", "")
    visualize_transition_matrix(markov_chain, dataset_name)
    visualize_markov_chain_graph(markov_chain, state_analysis, dataset_name)
    visualize_demographics(state_analysis, dataset_name)
    
    # Generate report
    generate_report(file, markov_chain, state_analysis, generated_sequences)
    
    return markov_chain, state_analysis, generated_sequences, end_time - start_time, memory_usage

# Function to visualize performance metrics
def visualize_performance(times, memory_usages, files):
    plt.figure(figsize=(16, 8))
    
    # Dataset names for x-axis
    dataset_names = [f.split('/')[-1].replace('.csv', '') for f in files]
    
    # Create execution time graph with improved scale
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x=dataset_names, y=times, palette="coolwarm")
    
    # Set y-axis limits with padding for better visualization
    max_time = max(times) * 1.2 # Add 20% padding
    plt.ylim(0, max_time)
    
    # Add exact values on top of bars
    for i, v in enumerate(times):
        ax1.text(i, v + (max_time * 0.03), f"{v:.2f}s", ha='center')
    
    plt.ylabel("Time (seconds)")
    plt.xlabel("Dataset")
    plt.title("Markov Chain Execution Time per Dataset")
    
    # Create memory usage graph with improved scale
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=dataset_names, y=memory_usages, palette="viridis")
    
    # Set y-axis limits with padding
    max_memory = max(memory_usages) * 1.2 # Add 20% padding
    plt.ylim(0, max_memory)
    
    # Add exact values on top of bars
    for i, v in enumerate(memory_usages):
        ax2.text(i, v + (max_memory * 0.03), f"{v:.1f}MB", ha='center')
    
    plt.ylabel("Memory (MB)")
    plt.xlabel("Dataset")
    plt.title("Markov Chain Memory Usage per Dataset")
    
    plt.tight_layout()
    
    img_path = f"{RESULTS_DIR}/markov_chain_performance_metrics.png"
    plt.savefig(img_path)
    plt.close()
    
    return img_path

# Main function to run the entire analysis
def main():
    files = [
        "/content/drive/MyDrive/SEM8/DataMining/PROJECT/User_Management_Analysis/engagement_analysis_results/high_engagement_cluster.csv"
    ]
    
    print("Starting Markov Chain Analysis with Post Type and Gender Demographics")
    
    all_markov_chains = []
    all_state_analyses = []
    all_generated_sequences = []
    times = []
    memory_usages = []
    
    for file in files:
        print(f"Processing {file.split('/')[-1]}...")
        markov_chain, state_analysis, generated_sequences, execution_time, memory_usage = process_file(file)
        all_markov_chains.append(markov_chain)
        all_state_analyses.append(state_analysis)
        all_generated_sequences.append(generated_sequences)
        times.append(execution_time)
        memory_usages.append(memory_usage)
        print(f"Done! Analysis completed in {execution_time:.2f} seconds")
    
    # Save metrics to CSV for parallel processing analysis
    metrics_df = pd.DataFrame({
        'Algorithm': ['Markov'],
        'Time': [sum(times)],
        'Memory': [sum(memory_usages)]
    })
    
    metrics_df.to_csv(f"{RESULTS_DIR}/markov_metrics.csv", index=False)
    
    # Visualize performance metrics
    visualize_performance(times, memory_usages, files)
    
    print(f"Analysis complete! Results saved to {RESULTS_DIR}")
    print(f"Performance metrics visualization saved")

if __name__ == "__main__":
    main()
