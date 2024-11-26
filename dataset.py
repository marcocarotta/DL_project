import torch
from torch.utils.data import Dataset
from typing import Tuple
import string
import random
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO)  # Use INFO, WARNING, or DEBUG as needed

class CharDataset(Dataset):
    """
    Character-level dataset.

    Handles text preprocessing, tokenization, and basic dataset analysis 
    with a focus on flexibility and simple manipulation of character-based data.

    Example:
        ```python
        # Load and preprocess dataset text
        with open('dataset.txt', 'r') as file:
            text = file.read()
        
        # Create dataset
        dataset = CharDataset(text, block_size=100)
        
        # Analyze original dataset
        dataset.dataset_analysis()
        
        # Preprocess and re-analyze
        dataset.preprocess(lowercase=True, punctuation=True)
        dataset.dataset_analysis()
        ```

    Attributes:
        block_size (int): Length of input sequences for model training
        noise_prob (float): Probability of each char in the input sequence to be noised in multiple way (see _inject_noise method).
        original_text (str): Unmodified original input text
        text (str): Current processed text
        chars (List[str]): Sorted list of unique characters in current text
        stoi (Dict[str, int]): String-to-index mapping for characters
        itos (Dict[int, str]): Index-to-string mapping for characters
        data (List[int]): Tokenized text as list of character indices
    """
    
    def __init__(self, text: str, block_size: int, noise_prob: float = 0.0):
        """
        Initializes the dataset with original text.

        Args:
            text (str): Input text data.
            block_size (int): Length of input sequences.
            noise_prob (float): Probability of each char in the input sequence to be noised in multiple ways (see `_inject_noise` method).
        
        Raises:
            ValueError: If `block_size` is not greater than 0.
            ValueError: If `noise_prob` is not between 0 and 1.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if not (0 <= noise_prob <= 1):
            raise ValueError(f"noise_prob must be between 0 and 1, got {noise_prob}")

        self.block_size = block_size
        self.noise_prob = noise_prob
        self.original_text = text
        self.original_chars = set(text)

        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)  # Default level; users can change as needed.

        # Initially set up with original text
        self.update_dataset(text)

    def preprocess(self, lowercase: bool = True, punctuation: bool = False):
        """
        Preprocess the text and update the dataset.

        Args:
            lowercase (bool): Whether to convert text to lowercase.
            punctuation (bool): Whether to remove punctuation.
        
        Returns:
            self: Allows method chaining
        """
        # Preprocess the original text
        processed_text = self._preprocess_text(
            self.original_text, 
            lowercase=lowercase, 
            punctuation=punctuation
        )
        
        # Update dataset with processed text
        self.update_dataset(processed_text)
        
        return self

    def update_dataset(self, text: str):
        """
        Update the dataset with a new text.

        This method updates all derived attributes of the dataset, including 
        the character set, string-to-index mapping, and tokenized data.

        Args:
            text (str): New text to update the dataset with.

        Returns:
            self: Allows method chaining
        
        Note:
            This method can be called directly to replace the current dataset 
            with entirely new text. For preprocessing before updates, see 
            `preprocess`.
        """
        self.text = text
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = [self.stoi[ch] for ch in text]
        self.vocabulary_size = len(self.chars)
        return self

    @staticmethod
    def _preprocess_text(text: str, lowercase: bool = True, punctuation: bool = False) -> str:
        """
        Preprocesses the text data by converting to lowercase and removing punctuation according to arguments.
        The punctuation is removed according to the string.punctuation list.

        Args:
            text (str): Input text data.
            lowercase (bool): Whether to convert text to lowercase.
            punctuation (bool): Whether to remove punctuation.

        Returns:
            str: Preprocessed text data.
        """
        if lowercase:
            text = text.lower()
        if punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def dataset_analysis(self):
        """
        Comprehensive analysis of current dataset state.

        Provides insights into:
        - Character composition
        - Sequence characteristics
        - Transformations from original text

        Outputs include:
        - Total and unique character counts
        - Character frequency distribution
        - Sequence length statistics
        - Changes in character set during preprocessing
        """
        self.logger.info("### Dataset Analysis ###")

        total_characters = len(self.text)
        unique_characters = set(self.text)

        self.logger.info("### Comparison with Original Dataset ###")
        self.logger.info(f"Original Total Characters: {len(self.original_text)}")
        self.logger.info(f"Current Total Characters: {total_characters}")
        self.logger.info(f"Original Unique Characters: {len(self.original_chars)}")
        self.logger.info(f"Current Unique Characters: {len(unique_characters)}")

        self.logger.info("### Detailed Current Dataset Analysis ###")
        self._character_frequency_analysis()
        self._sequence_analysis()
        self._character_set_changes()

    def _character_frequency_analysis(self):
        """
        Calculate and print character frequency distribution.
        """
        char_counts = Counter(self.text)
        sorted_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

        self.logger.info("### Characters and Frequencies ###")
        for char, freq in sorted_counts:
            printable_char = self._get_printable_char(char)
            self.logger.info(f"'{printable_char}': {freq}")

    def _sequence_analysis(self):
        """
        Analyze sequence lengths in the dataset.
        """
        lines = self.text.split("\n")
        line_lengths = [len(line) for line in lines]
        
        avg_length = sum(line_lengths) / len(lines) if lines else 0
        max_length = max(line_lengths) if lines else 0
        
        self.logger.info("### Sequence Analysis ###")
        self.logger.info(f"Number of Lines: {len(lines)}")
        self.logger.info(f"Average Line Length (in char): {avg_length:.2f}")
        self.logger.info(f"Max Line Length (in char): {max_length}")

    def _character_set_changes(self):
        """Compare original and current character sets."""
        
        removed_chars = self.original_chars - set(self.text)
        new_chars = set(self.text) - self.original_chars

        self.logger.info("### Character Set Changes ###")
        if removed_chars:
            self.logger.info(f"Characters Removed: {removed_chars}")
        if new_chars:
            self.logger.info(f"New Characters Added: {new_chars}")
        if not removed_chars and not new_chars:
            self.logger.info("No changes in character set.")

    def remove_less_frequent_chars(self, threshold: int = 1):
        """
        Remove characters with frequency below a given threshold and substitute them with a UNK token.

        Args:
            threshold (int): Minimum frequency threshold for character removal.
        
        Returns:
            self: Allows method chaining
        """
        char_counts = Counter(self.text)
        filtered_chars = {ch for ch, freq in char_counts.items() if freq >= threshold}
        unk_token = "@"

        # Replace characters below threshold with UNK token
        self.text = "".join([ch if ch in filtered_chars else unk_token for ch in self.text])
        self.update_dataset(self.text)

        self.logger.info(f"Removed characters with frequency < {threshold} and replaced them with '{unk_token}' token.")
        return self

    def _get_printable_char(self, char: str) -> str:
        """
        Convert character to a printable representation.
        """
        if char == "\n":
            return "\\n"
        elif char == " ":
            return "\u2423"  # Unicode for "open box" character (visual representation of space)
        return char
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the input-output pair at a given index.
        """
        if idx < 0 or idx >= len(self.data) - self.block_size:
            raise IndexError(f"Index {idx} is out of bounds")
        
        chunk = self.data[idx:idx + self.block_size + 1]

        noisy_chunk = self._inject_noise(chunk)

        x = noisy_chunk[:-1]  # Input sequence
        y = chunk[1:]   # Target sequence

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def get_stoi(self) -> dict:
        """
        Returns the string-to-index mapping.
        """
        return self.stoi
    
    def get_itos(self) -> dict:
        """
        Returns the index-to-string mapping.
        """
        return self.itos
    
    def _inject_noise(self, chunk: list) -> list:
        """
        Injects different types of noise into the input chunk.

        Noise is applied only to characters up to the second-to-last position (last character is excluded, as it is the target). 
        The following noise types are applied based on random selection:
        - Character flipping with its neighbor.
        - Change of case (if applicable).
        - Random substitution with a character from `self.chars`.

        The likelihood of injecting noise into a character is controlled by `self.noise_prob`.

        Args:
            chunk (list): Input sequence of character indices.

        Returns:
            list: The modified noisy sequence.
        """
        noisy_chunk = chunk[:]
        for i in range(len(noisy_chunk) - 1):  # Skip the last character as it is the target (see range man if you have doubt)
            if random.random() < self.noise_prob:
                noise_type = random.randint(0, 2)
                
                if noise_type == 0:  # Character flipping with the following one, if i is not the last but one character
                    if i < len(noisy_chunk) - 2:  # Ensure there is a next character
                        noisy_chunk[i], noisy_chunk[i + 1] = noisy_chunk[i + 1], noisy_chunk[i]
                    else: # If i is the last character, replace it with a random character
                        noisy_chunk[i] = random.choice(self.chars)

                elif noise_type == 1:  # Change of case
                    if noisy_chunk[i].isalpha():  # Ensure the character is alphabetic
                        noisy_chunk[i] = noisy_chunk[i].lower() if noisy_chunk[i].isupper() else noisy_chunk[i].upper()
                    else: # If the character is not alphabetic, replace it with a random character
                        noisy_chunk[i] = random.choice(self.chars)
                else:  # Random substitution with a character from `self.chars`
                    noisy_chunk[i] = random.choice(self.chars)  

        return noisy_chunk

if __name__ == "__main__":
    # Example usage
    # Suppose that the dataset.txt file is already downloaded in the folder ./data

    # Load and preprocess dataset text
    with open('data/dataset.txt', 'r') as file:
        text = file.read()

    # Create dataset
    dataset = CharDataset(text, block_size=100)

    # Analyze original dataset
    dataset.dataset_analysis()

    # Preprocess and re-analyze
    dataset.preprocess(lowercase=True, punctuation=True).dataset_analysis()


