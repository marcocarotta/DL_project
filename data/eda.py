import string
from collections import Counter
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load the Shakespeare dataset from a text file."""
    try:
        with open(filepath, "r") as file:
            data = file.read()
        print(f"Loaded dataset with {len(data)} characters.")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def dataset_statistics(data):
    """Print basic statistics about the dataset."""
    total_characters = len(data)
    unique_characters = set(data)
    
    print("\n### Dataset Statistics ###")
    print(f"Total Characters: {total_characters}")
    print(f"Unique Characters: {len(unique_characters)}")
    print(f"Character Set: {unique_characters}")
    
    return unique_characters

def character_frequency(data):
    """Calculate and plot the character frequency distribution."""
    char_counts = Counter(data)
    sorted_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print top 10 characters
    print("\n### Top 11 Characters ###")
    for char, freq in sorted_counts[:11]:
        if char == "\n":
            char = "\\n"
        if char == " ":
            char = "space"
        print(f"'{char}': {freq}")
    
    # Plot the distribution
    print("\nPlotting character frequency distribution...")
    characters, frequencies = zip(*sorted_counts)
    characters = list(characters)
    for i, char in enumerate(characters):
        if char == "\n":
            characters[i] = "\\n"
        if char == " ":
            characters[i] = "\" \""
    plt.figure(figsize=(12, 6))
    plt.bar(characters, frequencies)
    plt.title("Character Frequency Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.show()

def sequence_analysis(data):
    """Analyze sequence lengths in the dataset."""
    lines = data.split("\n")
    line_lengths = [len(line) for line in lines]
    
    avg_length = sum(line_lengths) / len(line_lengths)
    max_length = max(line_lengths)
    
    print("\n### Sequence Analysis ###")
    print(f"Number of Lines: {len(lines)}")
    print(f"Average Line Length (in char): {avg_length:.2f}")
    print(f"Max Line Length (in char): {max_length}")
    
    return line_lengths

def check_for_noise(unique_characters):
    """Identify unusual characters in the dataset."""
    valid_chars = string.ascii_letters + string.digits + string.punctuation + " \n"
    unusual_chars = [c for c in unique_characters if c not in valid_chars]
    
    print("\n### Noise Check ###")
    if unusual_chars:
        print(f"Unusual Characters Found: {unusual_chars}")
    else:
        print("No unusual characters found.")
    
    return unusual_chars

def main():
    # Filepath to the Shakespeare dataset
    filepath = "data/dataset.txt" 

    print("### Starting Dataset Analysis ###")
    data = load_data(filepath)
    
    if data:
        # Perform dataset analysis
        unique_characters = dataset_statistics(data)
        character_frequency(data)
        sequence_analysis(data)
        check_for_noise(unique_characters)

    print("\n### Analysis Complete ###")

if __name__ == "__main__":
    main()
