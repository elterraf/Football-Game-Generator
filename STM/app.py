from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import time
import pandas as pd

app = Flask(__name__, template_folder="templates")

def load_data(file_path):
    return pd.read_json(file_path)

def compute_transition_matrix(data):
    return pd.crosstab(data['label'], data['next_action'], normalize='index')

def modify_transition_matrix(transition_matrix, style_modifications, style):
    # Apply the style of play modifications to the transition matrix
    adjusted_transition_matrix = transition_matrix.copy()

    for action in transition_matrix.index:
        for next_action in transition_matrix.columns:
            adjusted_transition_matrix.loc[action, next_action] += style_modifications.get(style, {}).get(next_action, 1.0)
    # Normalize the adjusted transition matrix to make row sums equal to 1
    adjusted_transition_matrix = adjusted_transition_matrix.div(adjusted_transition_matrix.sum(axis=1), axis=0)
    return adjusted_transition_matrix

def calculate_action_distributions(data):
    action_norm_distributions = {}
    action_length_distributions = {}
    
    for action in data['label'].unique():
        norms_lists = data[data['label'] == action]['norm']

        mean_norms = [np.mean(norms) for norms in norms_lists]
        mean_norm = np.mean(mean_norms)

        std_norm = np.std(mean_norms)
        action_norm_distributions[action] = (mean_norm, std_norm)

        norm_lengths = [len(norms) for norms in norms_lists]
        mean_length = np.mean(norm_lengths)
        std_length = np.std(norm_lengths)

        action_length_distributions[action] = (mean_length, std_length)
    
    return action_norm_distributions, action_length_distributions

def simulate_game(desired_duration_seconds, acceleration_frequency, adjusted_transition_matrix, action_norm_distributions, action_length_distributions):
    desired_num_data_points = int(desired_duration_seconds * acceleration_frequency)
    current_action = 'walk'
    simulated_game = []
    current_duration_seconds = 0
    
    while current_duration_seconds < desired_duration_seconds:
        next_action = np.random.choice(adjusted_transition_matrix.columns, p=adjusted_transition_matrix.loc[current_action].values)
        mean_norm, std_norm = action_norm_distributions[current_action]
        mean_length, std_length = action_length_distributions[current_action]
        
        norm_length = int(np.clip(np.random.normal(mean_length, std_length), 1, 227))
        simulated_norm = np.random.normal(mean_norm, std_norm, norm_length).tolist()
        
        simulated_game.append({'label': current_action, 'norm': simulated_norm})
        current_action = next_action
        
        current_duration_seconds = len(simulated_game) / acceleration_frequency
    
    return simulated_game

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_game', methods=['POST'])
def generate_game_route():
    try:
        desired_duration_seconds = int(request.form['duration'])
        selected_style = request.form['style']
        number_of_games = int(request.form.get('number-of-games', 1))  # Default to 1 if not provided
        style_modifications = {
            "attacking": {"shot": 0.2, "pass": 0.1, "dribble": 0.1, "run": 0, "walk": 0, "tackle": 0, "rest": 0.0, "cross": 0.0},
            "defensive": {"tackle": 0.2, "rest": 0.1, "run": 0, "walk": 0., "shot": 0.0, "pass": 0.0, "dribble": 0.0, "cross": 0.0},
            "normal": {"run": 0.0, "walk": 0.0, "tackle": 0.0, "rest": 0.0, "cross": 0.0, "dribble": 0.0, "pass": 0.0, "shot": 0.0},
        }

        def generate_game(desired_duration_seconds, style_modifications):
            # Your game generation code here
            # Use the desired_duration_seconds and style_modifications as inputs
            # Return the generated game as a dictionary
            # Load and preprocess data
            data = load_data('match_1.json')
            data['next_action'] = data['label'].shift(-1)
            data.dropna(inplace=True)
            
            # Compute the transition matrix
            transition_matrix = compute_transition_matrix(data)
            # Modify the transition matrix based on the selected style
            adjusted_transition_matrix = modify_transition_matrix(transition_matrix, style_modifications, selected_style)
            # Calculate action distributions
            action_norm_distributions, action_length_distributions = calculate_action_distributions(data)
            
            # Simulate a game
            acceleration_frequency = 50
            simulated_game = simulate_game(desired_duration_seconds, acceleration_frequency, adjusted_transition_matrix, action_norm_distributions, action_length_distributions)
 
                # Return the generated game as a dictionary
            return {
                "message": "Game generated successfully",
                "duration": desired_duration_seconds,
                "style": selected_style,
                "game_data": simulated_game  # Include the game data in the dictionary
            }
        
                # List to store filenames of all generated games
        filenames = []

        # Loop through the number of games specified by the user
        for _ in range(number_of_games):
            # Generate a unique filename for each game based on style, duration, and timestamp
            timestamp = int(time.time())
            game_filename = f'games_generated/game_{selected_style}_{desired_duration_seconds}_{timestamp}_with_norms.json'
            filenames.append(game_filename)  # Append the filename to the list

            # Generate the game
            game_data = generate_game(desired_duration_seconds, style_modifications)

            # Save the generated game data to the unique JSON filename
            with open(game_filename, 'w') as f:
                json.dump(game_data["game_data"], f)

            # Sleep for a second to ensure a different timestamp for the next game if generated within the same second
            time.sleep(1)

        return jsonify({"success": True, "message": "Game generated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
