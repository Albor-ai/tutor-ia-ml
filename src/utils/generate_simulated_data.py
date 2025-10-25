import pandas as pd
import numpy as np
import os

def generate_simulated_lp_data(num_samples=5000):
    """
    Generates simulated data for Learning Profile (LP) inference.
    """
    np.random.seed(42) # for reproducibility

    # Define learning profiles
    learning_profiles = ['Visual', 'Auditivo', 'Leitura/Escrita', 'Cinestésico']

    # Define features and their typical ranges/values
    data = {}

    # Time-based features (in minutes)
    data['time_spent_on_video'] = np.random.normal(loc=60, scale=30, size=num_samples).clip(0, 200)
    data['time_spent_on_audio'] = np.random.normal(loc=40, scale=20, size=num_samples).clip(0, 150)
    data['time_spent_reading'] = np.random.normal(loc=70, scale=35, size=num_samples).clip(0, 250)
    data['time_spent_writing'] = np.random.normal(loc=30, scale=15, size=num_samples).clip(0, 100)
    data['time_spent_on_quizz'] = np.random.normal(loc=20, scale=10, size=num_samples).clip(0, 60)
    data['time_spent_on_flashcards'] = np.random.normal(loc=15, scale=8, size=num_samples).clip(0, 45)

    # Completion-based features
    data['completed_exercices'] = np.random.randint(0, 50, size=num_samples)
    data['completed_quizzes'] = np.random.randint(0, 30, size=num_samples)
    data['completed_flashcards'] = np.random.randint(0, 100, size=num_samples)

    # Accuracy-based features (0-100)
    data['text_quizzes_accuracy'] = np.random.normal(loc=75, scale=10, size=num_samples).clip(50, 100)
    data['visual_quizzes_accuracy'] = np.random.normal(loc=80, scale=10, size=num_samples).clip(50, 100)

    # Most preferred resource type (categorical)
    data['most_preferred_resource_type'] = np.random.choice(['video', 'audio', 'text', 'practical'], size=num_samples, p=[0.25, 0.2, 0.3, 0.25])

    df = pd.DataFrame(data)

    # Introduce correlation with learning profiles
    # This is a simplified way to create some patterns
    lp_data = []
    for i in range(num_samples):
        profile = np.random.choice(learning_profiles, p=[0.25, 0.25, 0.25, 0.25]) # Even distribution initially
        
        # Probabilistic assignment of preferred resource with reduced correlation
        resource_options = ['video', 'audio', 'text', 'practical']
        if profile == 'Visual':
            df.loc[i, 'time_spent_on_video'] *= np.random.uniform(1.2, 1.8)
            df.loc[i, 'visual_quizzes_accuracy'] *= np.random.uniform(1.05, 1.15)
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.4, 0.2, 0.2, 0.2])
        elif profile == 'Auditivo':
            df.loc[i, 'time_spent_on_audio'] *= np.random.uniform(1.2, 1.8)
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.2, 0.4, 0.2, 0.2])
        elif profile == 'Leitura/Escrita':
            df.loc[i, 'time_spent_reading'] *= np.random.uniform(1.2, 1.8)
            df.loc[i, 'time_spent_writing'] *= np.random.uniform(1.2, 1.8)
            df.loc[i, 'text_quizzes_accuracy'] *= np.random.uniform(1.05, 1.15)
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.2, 0.2, 0.4, 0.2])
        elif profile == 'Cinestésico':
            df.loc[i, 'completed_exercices'] *= np.random.uniform(1.5, 2.5)
            df.loc[i, 'time_spent_on_quizz'] *= np.random.uniform(1.2, 1.8)
            df.loc[i, 'most_preferred_resource_type'] = np.random.choice(resource_options, p=[0.2, 0.2, 0.2, 0.4])
        
        lp_data.append(profile)

    df['perfil_aprendizagem'] = lp_data

    # Ensure numerical columns are within reasonable bounds after correlation adjustments
    for col in ['time_spent_on_video', 'time_spent_on_audio', 'time_spent_reading', 'time_spent_writing',
                'time_spent_on_quizz', 'time_spent_on_flashcards', 'completed_exercices', 'completed_quizzes',
                'completed_flashcards', 'text_quizzes_accuracy', 'visual_quizzes_accuracy']:
        df[col] = df[col].clip(lower=0) # Ensure no negative values
        if 'accuracy' in col:
            df[col] = df[col].clip(upper=100) # Ensure accuracy is max 100

    return df

if __name__ == "__main__":
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    simulated_df = generate_simulated_lp_data(num_samples=2000)
    output_path = os.path.join(output_dir, 'simulated_lp_data.csv')
    simulated_df.to_csv(output_path, index=False)
    print(f"Simulated data saved to {output_path}")
    print(simulated_df.head())
