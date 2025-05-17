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
RESULTS_DIR = "/content/drive/MyDrive/SEM8/DataMining/PROJECT/User_Management_Analysis/engagement_analysis_results/PrefixSpan"
os.makedirs(RESULTS_DIR, exist_ok=True)

# PrefixSpan algorithm implementation
class PrefixSpan:
    def __init__(self, sequences, min_support=0.1):
        self.sequences = sequences
        self.min_support = min_support if min_support < 1 else min_support / len(sequences)
        self.min_count = int(self.min_support * len(sequences))
        self.patterns = []
        
    def _find_frequent_items(self, projected_db):
        """Find all frequent items in the projected database."""
        items_count = {}
        for sequence, start_pos in projected_db:
            # Get unique items in the sequence starting from start_pos
            visited = set()
            for i in range(start_pos, len(sequence)):
                item = sequence[i]
                if item not in visited:
                    items_count[item] = items_count.get(item, 0) + 1
                    visited.add(item)
        
        # Filter items by min_count
        frequent_items = {item: count for item, count in items_count.items() 
                         if count >= self.min_count}
        return frequent_items
    
    def _project_db(self, projected_db, item):
        """Create a projected database for a given item."""
        new_projected_db = []
        for sequence, start_pos in projected_db:
            # Find all occurrences of item in the sequence after start_pos
            for i in range(start_pos, len(sequence)):
                if sequence[i] == item:
                    new_projected_db.append((sequence, i + 1))
                    break
        return new_projected_db
    
    def _mine_patterns(self, prefix, projected_db):
        """Recursively mine patterns with the given prefix."""
        frequent_items = self._find_frequent_items(projected_db)
        
        for item, count in frequent_items.items():
            # Form a new pattern by appending item to prefix
            new_pattern = prefix + [item]
            support = count / len(self.sequences)
            
            # Add the pattern to results
            self.patterns.append((new_pattern, support))
            
            # Create a new projected database
            new_projected_db = self._project_db(projected_db, item)
            
            # Recursively mine with the new prefix
            if new_projected_db:
                self._mine_patterns(new_pattern, new_projected_db)
    
    def mine(self):
        """Start the mining process."""
        # Initialize with the complete database
        initial_projected_db = [(sequence, 0) for sequence in self.sequences]
        self._mine_patterns([], initial_projected_db)
        return self.patterns

# Function to preprocess data and save pickle
def preprocess_data(file):
    df = pd.read_csv(file)
    df.fillna({'Likes': 0, 'Shares': 0, 'Comments': 0, 'Engagement Rate': 0, 'Audience Age': df['Audience Age'].median()}, inplace=True)
    
    # For PrefixSpan, we need to maintain the sequence order
    sequences = []
    sequence_details = [] # To store post type and gender for each sequence
    
    for idx, row in df.iterrows():
        sequence = []
        # Time of day first in sequence
        sequence.append('Morning Post' if pd.to_datetime(row['Post Timestamp']).hour < 12 else 'Evening Post')
        
        # Engagement metrics - maintain order for sequential patterns
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

# Function to apply PrefixSpan algorithm and save pickle
def apply_prefixspan(sequences, sequence_details, file):
    # Apply PrefixSpan algorithm
    prefixspan = PrefixSpan(sequences, min_support=0.1)
    frequent_patterns = prefixspan.mine()
    
    # Convert results to a more usable format
    patterns = []
    for pattern, support in frequent_patterns:
        if len(pattern) >= 2:  # Only consider patterns with at least 2 items
            # Split the pattern into antecedents and consequents
            antecedents = pattern[:-1]
            consequents = [pattern[-1]]
            
            # Calculate confidence and lift
            # Confidence = support(pattern) / support(antecedents)
            antecedent_support = 0
            for seq in sequences:
                # Check if antecedents appear in sequence in order
                for i in range(len(seq) - len(antecedents) + 1):
                    if seq[i:i+len(antecedents)] == antecedents:
                        antecedent_support += 1
                        break
            
            confidence = support / (antecedent_support / len(sequences)) if antecedent_support > 0 else 0
            
            # Lift = confidence / support(consequents)
            consequent_support = 0
            for seq in sequences:
                if consequents[0] in seq:
                    consequent_support += 1
            
            lift = confidence / (consequent_support / len(sequences)) if consequent_support > 0 else 0
            
            patterns.append({
                'antecedents': antecedents,
                'consequents': consequents,
                'support': support,
                'confidence': confidence,
                'lift': lift
            })
    
    # Convert to DataFrame
    patterns_df = pd.DataFrame(patterns)
    
    # Analyze patterns by post type and gender
    pattern_analysis = []
    for idx, pattern in patterns_df.iterrows():
        # Extract pattern elements
        antecedents = pattern['antecedents']
        consequents = pattern['consequents']
        
        # Find sequences that contain the pattern in order
        matching_sequences = []
        for detail in sequence_details:
            seq = detail['sequence']
            
            # For PrefixSpan, check if pattern appears in the sequence in order
            for i in range(len(seq) - len(antecedents) - len(consequents) + 1):
                if seq[i:i+len(antecedents)] == antecedents and \
                   seq[i+len(antecedents):i+len(antecedents)+len(consequents)] == consequents:
                    matching_sequences.append(detail)
                    break
        
        # Count post types and genders for matching sequences
        post_types = Counter([seq['post_type'] for seq in matching_sequences])
        genders = Counter([seq['gender'] for seq in matching_sequences])
        
        # Add to pattern analysis
        pattern_analysis.append({
            'pattern_id': idx,
            'antecedents': antecedents,
            'consequents': consequents,
            'support': pattern['support'],
            'confidence': pattern['confidence'],
            'lift': pattern['lift'],
            'most_common_post_type': post_types.most_common(1)[0][0] if post_types else "N/A",
            'post_type_distribution': dict(post_types),
            'most_common_gender': genders.most_common(1)[0][0] if genders else "N/A",
            'gender_distribution': dict(genders),
            'sequence_count': len(matching_sequences)
        })
    
    # Save patterns and analysis as pickle
    dataset_name = file.split('/')[-1].replace('.csv', '')
    pattern_pickle_path = f"{RESULTS_DIR}/{dataset_name}_prefixspan_patterns.pkl"
    analysis_pickle_path = f"{RESULTS_DIR}/{dataset_name}_prefixspan_pattern_analysis.pkl"
    
    with open(pattern_pickle_path, 'wb') as f:
        pickle.dump(patterns_df.to_dict(orient='records'), f)
    with open(analysis_pickle_path, 'wb') as f:
        pickle.dump(pattern_analysis, f)
    
    # Create an extended DataFrame with pattern analysis for visualizations
    extended_patterns = pd.DataFrame(pattern_analysis)
    
    return patterns_df, extended_patterns

# Function to save pattern table as an image with post type and gender information
def save_table_as_image(patterns, extended_patterns, title):
    table = PrettyTable()
    table.field_names = ["Antecedents", "Consequents", "Support", "Confidence", "Lift", "Most Common Post Type", "Most Common Gender"]
    
    # Calculate max width for table
    table._max_width = {"Antecedents": 30, "Consequents": 30}
    
    for i, row in patterns.iterrows():
        ext_row = extended_patterns[extended_patterns['pattern_id'] == i].iloc[0] if i < len(extended_patterns) else None
        post_type = ext_row['most_common_post_type'] if ext_row is not None else "N/A"
        gender = ext_row['most_common_gender'] if ext_row is not None else "N/A"
        
        table.add_row([
            ', '.join(map(str, row['antecedents'])),
            ', '.join(map(str, row['consequents'])),
            round(row['support'], 4),
            round(row['confidence'], 4),
            round(row['lift'], 4),
            post_type,
            gender
        ])
    
    # Convert table to image
    table_str = str(table)
    lines = table_str.split('\n')
    
    # Create larger image for the detailed table
    img_height = len(lines) * 20 + 50
    img = Image.new('RGB', (1200, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    
    y_position = 10
    for line in lines:
        draw.text((10, y_position), line, fill=(0, 0, 0), font=font)
        y_position += 20
    
    img_path = f"{RESULTS_DIR}/{title}_prefixspan_patterns_table.png"
    img.save(img_path)

# Function to visualize pattern sequence with post type and gender distribution
def visualize_pattern_sequence(extended_patterns, title):
    # Select top patterns for visualization
    top_patterns = extended_patterns.sort_values(by='lift', ascending=False).head(10)
    
    # Create pattern sequence network graph
    plt.figure(figsize=(14, 10))
    G = nx.DiGraph()
    
    # Add nodes and edges from patterns
    for _, pattern in top_patterns.iterrows():
        for item in pattern['antecedents']:
            G.add_node(item, type='antecedent')
        for item in pattern['consequents']:
            G.add_node(item, type='consequent')
        
        # Add edges from each antecedent to each consequent
        # For PrefixSpan, we connect items in sequence order
        for i in range(len(pattern['antecedents']) - 1):
            current = pattern['antecedents'][i]
            next_item = pattern['antecedents'][i+1]
            if G.has_edge(current, next_item):
                G[current][next_item]['weight'] += pattern['lift']
            else:
                G.add_edge(current, next_item, weight=pattern['lift'])
        
        # Connect last antecedent to consequents
        if pattern['antecedents'] and pattern['consequents']:
            last_ant = pattern['antecedents'][-1]
            for cons in pattern['consequents']:
                if G.has_edge(last_ant, cons):
                    G[last_ant][cons]['weight'] += pattern['lift']
                else:
                    G.add_edge(last_ant, cons, weight=pattern['lift'])
    
    # Define node colors based on node type
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('type') == 'antecedent':
            node_colors.append('skyblue')
        else:
            node_colors.append('lightgreen')
    
    # Define edge weights
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    # Create positions for nodes - use hierarchical layout for sequential patterns
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray', arrows=True, arrowsize=20)
    
    # Calculate and display edge labels (distances)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = f"{data['weight']:.2f}" # Format to 2 decimal places
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f'PrefixSpan Pattern Sequence Analysis - {title}', fontsize=16)
    plt.axis('off')
    
    img_path = f"{RESULTS_DIR}/{title}_prefixspan_pattern_sequence_network.png"
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    
    # Create post type distribution chart
    plt.figure(figsize=(12, 8))
    
    # Prepare data for the chart
    pattern_ids = [f"P{pid}" for pid in top_patterns['pattern_id']]
    post_types = top_patterns['post_type_distribution'].apply(lambda x: dict(sorted(x.items())))
    
    # Stack plot for post type distribution
    plt.subplot(2, 1, 1)
    bottoms = None
    for post_type in set().union(*[d.keys() for d in post_types]):
        values = [d.get(post_type, 0) for d in post_types]
        if bottoms is None:
            bottoms = values
            plt.bar(pattern_ids, values, label=post_type)
        else:
            plt.bar(pattern_ids, values, bottom=bottoms, label=post_type)
            bottoms = [sum(x) for x in zip(bottoms, values)]
    
    plt.xlabel('Pattern ID')
    plt.ylabel('Count')
    plt.title('Post Type Distribution per PrefixSpan Pattern')
    plt.legend(loc='upper right')
    
    # Gender distribution chart
    plt.subplot(2, 1, 2)
    genders = top_patterns['gender_distribution'].apply(lambda x: dict(sorted(x.items())))
    bottoms = None
    for gender in set().union(*[d.keys() for d in genders]):
        values = [d.get(gender, 0) for d in genders]
        if bottoms is None:
            bottoms = values
            plt.bar(pattern_ids, values, label=gender)
        else:
            plt.bar(pattern_ids, values, bottom=bottoms, label=gender)
            bottoms = [sum(x) for x in zip(bottoms, values)]
    
    plt.xlabel('Pattern ID')
    plt.ylabel('Count')
    plt.title('Gender Distribution per PrefixSpan Pattern')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    img_path = f"{RESULTS_DIR}/{title}_prefixspan_pattern_demographics.png"
    plt.savefig(img_path)
    plt.close()

# Function to visualize original patterns with improved visualization
def visualize_patterns(patterns, extended_patterns, title):
    # Top patterns by lift
    plt.figure(figsize=(12, 8))
    patterns_sorted = patterns.sort_values(by='lift', ascending=False).head(10)
    
    # Create a more detailed bar chart
    ax = sns.barplot(
        x=[round(lift, 2) for lift in patterns_sorted['lift']],
        y=[f"{' → '.join([', '.join(map(str, a)), ', '.join(map(str, c))])}" for a, c in zip(patterns_sorted['antecedents'], patterns_sorted['consequents'])],
        palette="YlOrRd"
    )
    
    # Add confidence values as text
    for i, (_, row) in enumerate(patterns_sorted.iterrows()):
        ax.text(row['lift'] + 0.1, i, f"Conf: {round(row['confidence'], 2)}", va='center')
    
    plt.xlabel('Lift')
    plt.ylabel('Sequential Rule')
    plt.title(f'Top PrefixSpan Pattern Rules by Lift - {title}')
    plt.tight_layout()
    
    img_path = f"{RESULTS_DIR}/{title}_prefixspan_pattern_rules.png"
    plt.savefig(img_path)
    plt.close()

# Function to visualize performance metrics with improved scale limits and type/gender information
def visualize_metrics(times, memory_usages, files, type_gender_info):
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
    plt.title("PrefixSpan Execution Time per Dataset")
    
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
    plt.title("PrefixSpan Memory Usage per Dataset")
    
    # Add type and gender information as text
    plt.figtext(0.5, 0.01,
                f"Analysis includes Post Type and Gender Demographics: {type_gender_info}",
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make room for text
    
    img_path = f"{RESULTS_DIR}/prefixspan_performance_metrics_with_type_gender.png"
    plt.savefig(img_path)
    plt.close()

# Function to generate a comprehensive report
def generate_report(files, all_patterns, all_extended_patterns):
    report = "# Sequential Pattern Analysis Report (PrefixSpan)\n\n"
    
    # Overview
    report += "## Overview\n"
    report += "This report analyzes sequential patterns in user engagement data across different datasets with focus on post type and gender demographics using the PrefixSpan algorithm.\n\n"
    
    # For each dataset
    for i, file in enumerate(files):
        dataset_name = file.split('/')[-1].replace(".csv", "")
        patterns = all_patterns[i]
        ext_patterns = all_extended_patterns[i]
        
        report += f"## Dataset: {dataset_name}\n\n"
        
        # Top 5 patterns by lift
        top_patterns = patterns.sort_values(by='lift', ascending=False).head(5)
        report += "### Top Patterns by Lift\n\n"
        
        for j, (_, row) in enumerate(top_patterns.iterrows()):
            # Find corresponding extended pattern
            ext_row = ext_patterns[ext_patterns['pattern_id'] == row.name].iloc[0] if row.name < len(ext_patterns) else None
            
            if ext_row is not None:
                report += f"Pattern {j+1}: "
                report += f"{', '.join(map(str, row['antecedents']))} → {', '.join(map(str, row['consequents']))}\n"
                report += f"- Support: {round(row['support'], 4)}\n"
                report += f"- Confidence: {round(row['confidence'], 4)}\n"
                report += f"- Lift: {round(row['lift'], 4)}\n"
                report += f"- Most Common Post Type: {ext_row['most_common_post_type']}\n"
                report += f"- Most Common Gender: {ext_row['most_common_gender']}\n"
                report += f"- Sequence Count: {ext_row['sequence_count']}\n\n"
        
        # Pattern distribution
        report += "### Pattern Distribution\n\n"
        report += f"- Total patterns found: {len(patterns)}\n"
        report += f"- Average lift: {patterns['lift'].mean():.4f}\n"
        report += f"- Average confidence: {patterns['confidence'].mean():.4f}\n\n"
        
        # Type and gender distribution
        if not ext_patterns.empty:
            # Top post types across all patterns
            all_post_types = []
            for _, row in ext_patterns.iterrows():
                all_post_types.extend([(post_type, count) for post_type, count in row['post_type_distribution'].items()])
            post_type_counter = Counter()
            for post_type, count in all_post_types:
                post_type_counter[post_type] += count
            
            report += "### Post Type Distribution\n\n"
            for post_type, count in post_type_counter.most_common():
                report += f"- {post_type}: {count} occurrences ({count/sum(post_type_counter.values())*100:.1f}%)\n"
            
            # Top genders across all patterns
            all_genders = []
            for _, row in ext_patterns.iterrows():
                all_genders.extend([(gender, count) for gender, count in row['gender_distribution'].items()])
            gender_counter = Counter()
            for gender, count in all_genders:
                gender_counter[gender] += count
            
            report += "\n### Gender Distribution\n\n"
            for gender, count in gender_counter.most_common():
                report += f"- {gender}: {count} occurrences ({count/sum(gender_counter.values())*100:.1f}%)\n"
            
            report += "\n"
        
        # Common items in patterns
        all_items = []
        for items in patterns['antecedents'].tolist() + patterns['consequents'].tolist():
            all_items.extend(items)
        common_items = Counter(all_items).most_common(5)
        
        report += "### Most Common Elements in Patterns\n\n"
        for item, count in common_items:
            report += f"- {item}: {count} occurrences\n"
        
        report += "\n---\n\n"
    
    # Overall insights
    report += "## Overall Insights\n\n"
    report += "### Cross-Dataset Patterns\n"
    
    # Find common patterns across datasets
    common_pattern_elements = set()
    for patterns in all_patterns:
        for _, row in patterns.iterrows():
            for item in row['antecedents'] + row['consequents']:
                common_pattern_elements.add(item)
    
    report += "Elements appearing across multiple datasets:\n"
    for element in common_pattern_elements:
        datasets_with_element = []
        for i, patterns in enumerate(all_patterns):
            all_elements = []
            for _, row in patterns.iterrows():
                all_elements.extend(row['antecedents'] + row['consequents'])
            if element in all_elements:
                datasets_with_element.append(files[i].split('/')[-1].replace('.csv', ''))
        
        if len(datasets_with_element) > 1:
            report += f"- {element}: Found in {', '.join(datasets_with_element)}\n"
    
    # Recommendations based on findings
    report += "\n## Recommendations\n\n"
    report += "Based on the pattern analysis, consider the following recommendations:\n\n"
    
    # Analyze patterns to make data-driven recommendations
    high_engagement_patterns = []
    for i, patterns in enumerate(all_patterns):
        dataset_name = files[i].split('/')[-1].replace('.csv', '')
        for _, row in patterns.iterrows():
            if any('High Likes' in str(item) or 'More Shares' in str(item) or 'More Comments' in str(item)
                   or 'Higher Engagement Rate' in str(item) for item in row['consequents']):
                high_engagement_patterns.append((dataset_name, row['antecedents'], row['consequents'], row['lift']))
    
    # Sort by lift to get the most significant patterns
    high_engagement_patterns.sort(key=lambda x: x[3], reverse=True)
    
    # Generate recommendations based on top patterns
    if high_engagement_patterns:
        for i, (dataset, ant, cons, lift) in enumerate(high_engagement_patterns[:3]):
            report += f"{i+1}. When {', '.join(map(str, ant))} occurs, it often leads to {', '.join(map(str, cons))} (Dataset: {dataset}, Lift: {lift:.2f}).\n"
            
            # Add specific recommendation based on pattern
            if any('Morning Post' in str(item) for item in ant):
                report += "   - Consider focusing content distribution in morning hours.\n"
            if any('Evening Post' in str(item) for item in ant):
                report += "   - Consider focusing content distribution in evening hours.\n"
            if any('Senior Adults' in str(item) for item in ant or cons):
                report += "   - Content appears to resonate well with older demographics.\n"
    else:
        report += "- No clear high-engagement patterns were identified across datasets.\n"
    
    # Save the report
    report_path = f"{RESULTS_DIR}/prefixspan_sequential_pattern_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Generated comprehensive analysis report: {report_path}")
    return report_path

# Function to process file and save as pickle
def process_file(file):
    start_time = time.time()
    sequences, sequence_details = preprocess_data(file)
    patterns, extended_patterns = apply_prefixspan(sequences, sequence_details, file)
    end_time = time.time()
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
    
    # Generate enhanced visuals
    dataset_name = file.split('/')[-1].replace(".csv", "")
    save_table_as_image(patterns.head(10), extended_patterns, dataset_name)
    visualize_patterns(patterns, extended_patterns, dataset_name)
    visualize_pattern_sequence(extended_patterns, dataset_name)
    
    return patterns, extended_patterns, end_time - start_time, memory_usage

# Main function to run the entire analysis
def main():
    files = [
        "/content/drive/MyDrive/SEM8/DataMining/PROJECT/User_Management_Analysis/engagement_analysis_results/high_engagement_cluster.csv"
    ]
    
    print("Starting PrefixSpan Sequential Pattern Mining Analysis with Post Type and Gender Demographics")
    
    all_patterns = []
    all_extended_patterns = []
    times = []
    memory_usages = []
    
    for file in files:
        print(f"Processing {file.split('/')[-1]}...")
        patterns, extended_patterns, execution_time, memory_usage = process_file(file)
        all_patterns.append(patterns)
        all_extended_patterns.append(extended_patterns)
        times.append(execution_time)
        memory_usages.append(memory_usage)
        print(f"Done! Found {len(patterns)} patterns in {execution_time:.2f} seconds")
    
     # Save metrics to CSV for parallel processing analysis
    metrics_df = pd.DataFrame({
        'Algorithm': ['PrefixSpan'],
        'Time': [sum(times)],
        'Memory': [sum(memory_usages)]
    })
    
    metrics_df.to_csv(f"{RESULTS_DIR}/prefixspan_metrics.csv", index=False)

    # Visualize performance metrics
    type_gender_info = "Included in analysis"
    visualize_metrics(times, memory_usages, files, type_gender_info)
    
    # Generate comprehensive report
    report_path = generate_report(files, all_patterns, all_extended_patterns)
    
    print(f"Analysis complete! Results saved to {RESULTS_DIR}")
    print(f"Performance metrics visualization saved")
    print(f"Comprehensive report saved to {report_path}")

if __name__ == "__main__":
    main()

