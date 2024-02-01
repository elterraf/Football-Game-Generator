from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
import time
import pandas as pd
import os

app = Flask(__name__)

# Load your trained model
model = load_model('generator_model.h5')

# Utility Functions
def denormalize(scaled_values, min_value, max_value):
    return [value * (max_value - min_value) + min_value for value in scaled_values]

def calculate_mean_norms(df):
    label_mean_norms = {}
    for label in df['label'].unique():
        label_subset = df[df['label'] == label]
        mean_norm = np.mean([np.mean(row) for row in label_subset['norm']])
        label_mean_norms[label] = mean_norm
    return label_mean_norms

def filter_zeros_from_norms(norms):
    return [value for value in norms if value != 0.0]

def norm_to_label(norm, df):
    label_mean_norms_global = calculate_mean_norms(df)
    norm_mean = np.mean(norm)
    closest_label = min(label_mean_norms_global.keys(), key=lambda label: abs(label_mean_norms_global[label] - norm_mean))
    return closest_label


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_game', methods=['POST'])
def generate_game():


    with open('match_1.json', 'r') as file:
        data = json.load(file)

    # Create a DataFrame for easier analysis
    df = pd.DataFrame(data)
    max_value = max([max(seq) for seq in df['norm'].tolist() if len(seq) > 0])  # Find global max
    min_value = min([min(seq) for seq in df['norm'].tolist() if len(seq) > 0])  # Find global min

    game_length_seconds = int(request.form['game_length_seconds'])
    num_actions = game_length_seconds * 50
    num_games = int(request.form['num_games'])
    saved_file_paths = []

    for _ in range(num_games):

        noise = np.random.normal(0, 1, (num_actions, 100))
        generated_norms = model.predict(noise)
        generated_norms = [filter_zeros_from_norms(norm) for norm in generated_norms]
        generated_norms = [denormalize(norm, min_value, max_value) for norm in generated_norms]
        generated_labels = [norm_to_label(norm, df) for norm in generated_norms]

        generated_df = pd.DataFrame({'label': generated_labels, 'norm': list(generated_norms)})

        # Convert the DataFrame to a list of dictionaries
        generated_data = generated_df.to_dict(orient='records')
    
        # Save as a JSON file
        file_path = os.path.join("saved_games", f"generated_game_{int(time.time())}_{int(game_length_seconds)}.json")
        with open(file_path, 'w') as outfile:
            json.dump(generated_data, outfile)
        
        saved_file_paths.append(file_path)
        time.sleep(1)
    # Return the path to the saved file or any other related message
    return jsonify({"message": "Game generated successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
