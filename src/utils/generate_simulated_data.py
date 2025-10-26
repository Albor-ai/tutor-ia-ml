import pandas as pd
import numpy as np
import os

def generate_simulated_lp_data(num_samples=5000):
    """
    Generates more realistic simulated data for Learning Profile (LP) inference.
    Includes softened correlations, profile overlap, unbalanced distribution, and noise.
    """
    np.random.seed(42) # for reproducibility

    # 1. Unbalanced Profile Distribution (+/- 10% margin from 0.25)
    # Order: ['Visual', 'Auditivo', 'Leitura/Escrita', 'Cinestésico']
    learning_profiles = ['Visual', 'Auditivo', 'Leitura/Escrita', 'Cinestésico']
    profile_distribution = [0.275, 0.225, 0.275, 0.225]

    # Define features and their typical ranges/values
    data = {}

    # Time-based features (in hours)
    data['time_spent_on_video'] = np.random.normal(loc=5, scale=3, size=num_samples).clip(0, 40)
    data['time_spent_on_audio'] = np.random.normal(loc=3, scale=2, size=num_samples).clip(0, 40)
    data['time_spent_on_reading'] = np.random.normal(loc=4, scale=3, size=num_samples).clip(0, 40)
    data['time_spent_on_writing'] = np.random.normal(loc=3, scale=2, size=num_samples).clip(0, 40)
    data['time_spent_on_quizz'] = np.random.normal(loc=2, scale=1.5, size=num_samples).clip(0, 40)
    data['time_spent_on_flashcards'] = np.random.normal(loc=2, scale=1.5, size=num_samples).clip(0, 40)
    data['time_spent_on_projects'] = np.random.normal(loc=2, scale=2, size=num_samples).clip(0, 40)

    # Completion-based features
    data['completed_exercices'] = np.random.randint(0, 20, size=num_samples)
    data['completed_quizzes'] = np.random.randint(0, 15, size=num_samples)
    data['completed_flashcards'] = np.random.randint(0, 20, size=num_samples)

    # Accuracy-based features (0-100)
    data['text_quizzes_accuracy'] = np.random.normal(loc=75, scale=15, size=num_samples).clip(20, 100)
    data['visual_quizzes_accuracy'] = np.random.normal(loc=75, scale=15, size=num_samples).clip(20, 100)

    # Most preferred resource type (categorical)
    data['most_preferred_resource_type'] = np.random.choice(['video', 'audio', 'text', 'practical'], size=num_samples, p=[0.25, 0.25, 0.25, 0.25])



    df = pd.DataFrame(data)

    # 3. & 4. Soften Correlations and Introduce Overlap
    lp_data = []
    for i in range(num_samples):
        profile = np.random.choice(learning_profiles, p=profile_distribution)
        
        resource_options = ['video', 'audio', 'text', 'practical']
        
        if profile == 'Visual':
            # Primary trait (softened)
            df.loc[i, 'time_spent_on_video'] *= np.random.uniform(1.1, 1.6)
            df.loc[i, 'visual_quizzes_accuracy'] *= np.random.uniform(1.05, 1.1)
            # Secondary trait (overlap)
            df.loc[i, 'time_spent_on_reading'] *= np.random.uniform(1.0, 1.2) # Visuals may read diagrams
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.35, 0.2, 0.25, 0.2])

        elif profile == 'Auditivo':
            # Primary trait (softened)
            df.loc[i, 'time_spent_on_audio'] *= np.random.uniform(1.2, 1.7)
            # Secondary trait (overlap)
            df.loc[i, 'time_spent_on_video'] *= np.random.uniform(1.0, 1.3) # Audio learners might watch lectures
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.25, 0.35, 0.2, 0.2])

        elif profile == 'Leitura/Escrita':
            # Primary trait (softened)
            df.loc[i, 'time_spent_on_reading'] *= np.random.uniform(1.2, 1.6)
            df.loc[i, 'time_spent_on_writing'] *= np.random.uniform(1.1, 1.5)
            df.loc[i, 'text_quizzes_accuracy'] *= np.random.uniform(1.05, 1.1)
            # Secondary trait (overlap)
            df.loc[i, 'time_spent_on_flashcards'] *= np.random.uniform(1.0, 1.3)
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.2, 0.2, 0.35, 0.25])

        elif profile == 'Cinestésico':
            # Primary trait (softened)
            df.loc[i, 'time_spent_on_projects'] *= np.random.uniform(1.5, 2.5)
            df.loc[i, 'completed_exercices'] *= np.random.uniform(1.2, 1.8)
            # Secondary trait (overlap)
            df.loc[i, 'time_spent_on_quizz'] *= np.random.uniform(1.1, 1.4)
            df.loc[i, 'visual_quizzes_accuracy'] *= np.random.uniform(1.0, 1.05) # Spatial/visual aspect of doing
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.2, 0.2, 0.25, 0.35])
        
        lp_data.append(profile)

    df['learning_profiles'] = lp_data

    # Ensure numerical columns are within reasonable bounds
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].clip(lower=0)
        if 'accuracy' in col:
            df[col] = df[col].clip(upper=100)

    return df

if __name__ == "__main__":
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    simulated_df = generate_simulated_lp_data(num_samples=2000)
    output_path = os.path.join(output_dir, 'simulated_lp_data.csv')
    simulated_df.to_csv(output_path, index=False)
    print(f"Simulated data saved to {output_path}")
    print(simulated_df.head())