import json
import pandas as pd

def create_training_data(race_df):
    training_data = []
    
    # Example 1: Race prediction
    for _, race in race_df.iterrows():
        instruction = f"Predict the performance of {race['horse_name']} in a {race['distance']} race at {race['track_name']} with {race['track_conditions']} conditions."
        
        response = f"Based on historical data, {race['horse_name']} with jockey {race['jockey']} finished in position {race['finish_position']} with odds of {race['odds']}. The track conditions ({race['track_conditions']}) and distance ({race['distance']}) were key factors."
        
        training_data.append({
            "instruction": instruction,
            "input": "",
            "output": response
        })
    
    # Example 2: Factor analysis
    for _, race in race_df.iterrows():
        instruction = "What factors contributed to this race outcome?"
        input_data = f"Race: {race['track_name']}, Distance: {race['distance']}, Conditions: {race['track_conditions']}, Winner: {race['horse_name']}"
        
        response = f"The key factors were: 1) Track conditions ({race['track_conditions']}) favored this horse's running style, 2) The distance ({race['distance']}) suited the horse's stamina, 3) The jockey ({race['jockey']}) had experience with this track."
        
        training_data.append({
            "instruction": instruction,
            "input": input_data,
            "output": response
        })
    
    return training_data

# Save as JSONL
def save_training_data(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')