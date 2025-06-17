import numpy as np
from collections import defaultdict
import Levenshtein
import re
import os
from typing import List, Set, Tuple, Dict
import json
import pickle

def load_nepali_words() -> Set[str]:
    """Load Nepali words from the corpus file."""
    try:
        with open("data/nepali_corpus.txt", encoding='utf-8') as f:
            words = set(line.strip() for line in f if line.strip())
        return words
    except FileNotFoundError:
        print("Error: nepali_corpus.txt not found in data directory")
        return set()

def save_word_frequencies(words: Set[str], filename: str = 'word_frequencies.pkl'):
    """Save word frequencies to a file."""
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    with open(filename, 'wb') as f:
        pickle.dump(dict(word_freq), f)
    return word_freq

def load_word_frequencies(filename: str = 'word_frequencies.pkl') -> Dict[str, int]:
    """Load word frequencies from a file."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def get_word_completions(partial_word: str, word_list: Set[str], word_freq: Dict[str, int], max_suggestions: int = 5) -> List[str]:
    """Get word completions for a partial Nepali word."""
    # First try exact prefix matches
    completions = [(word, word_freq.get(word, 0)) for word in word_list if word.startswith(partial_word)]
    
    # Sort by frequency
    completions.sort(key=lambda x: x[1], reverse=True)
    completions = [word for word, _ in completions]
    
    # If we don't have enough suggestions, add similar words
    if len(completions) < max_suggestions:
        # Find words that start with similar prefixes
        similar_prefixes = []
        for word in word_list:
            if word != partial_word and len(word) >= len(partial_word):
                prefix = word[:len(partial_word)]
                distance = Levenshtein.distance(partial_word, prefix)
                if distance <= 1:  # Allow for one character difference
                    similar_prefixes.append((word, distance, word_freq.get(word, 0)))
        
        # Sort by distance and frequency
        similar_prefixes.sort(key=lambda x: (x[1], -x[2]))
        completions.extend([word for word, _, _ in similar_prefixes])
    
    return completions[:max_suggestions]

def get_spelling_suggestions(word: str, word_list: Set[str], word_freq: Dict[str, int], max_distance: int = 2, max_suggestions: int = 5) -> List[str]:
    """Get spelling suggestions for a potentially misspelled word."""
    suggestions = []
    
    # First try exact matches
    if word in word_list:
        return [word]
    
    # Then try Levenshtein distance
    for dict_word in word_list:
        distance = Levenshtein.distance(word, dict_word)
        if distance <= max_distance:
            suggestions.append((dict_word, distance, word_freq.get(dict_word, 0)))
    
    # Sort by distance and frequency
    suggestions.sort(key=lambda x: (x[1], -x[2]))
    return [word for word, _, _ in suggestions[:max_suggestions]]

def get_suggestions(input_text: str, words: Set[str], word_freq: Dict[str, int]) -> List[str]:
    """Get both completions and spelling suggestions for input text with improved ranking."""
    try:
        # Get traditional suggestions
        completions = get_word_completions(input_text, words, word_freq)
        spelling_suggestions = get_spelling_suggestions(input_text, words, word_freq)
        
        # Combine and rank all suggestions
        all_suggestions = []
        for suggestion in set(completions + spelling_suggestions):
            # Calculate relevance score
            prefix_match = suggestion.startswith(input_text)
            length_diff = abs(len(suggestion) - len(input_text))
            edit_distance = Levenshtein.distance(input_text, suggestion[:len(input_text)])
            frequency = word_freq.get(suggestion, 0)
            
            # Weighted scoring
            score = (prefix_match * 3) - (length_diff * 0.5) - (edit_distance * 0.5) + (frequency * 0.1)
            all_suggestions.append((suggestion, score))
        
        # Sort by score and return top suggestions
        all_suggestions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in all_suggestions[:5]]
    except Exception as e:
        print(f"Error getting suggestions: {str(e)}")
        return []

def interactive_demo():
    """Run an interactive demo of the suggestion system."""
    print("Loading Nepali words...")
    words = load_nepali_words()
    print(f"Loaded {len(words)} unique words")
    
    # Load or create word frequencies
    word_freq = load_word_frequencies()
    if not word_freq:
        print("Training word frequencies...")
        word_freq = save_word_frequencies(words)
    
    while True:
        user_input = input("\nEnter a Nepali word (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        suggestions = get_suggestions(user_input, words, word_freq)
        print(f"\nSuggestions for '{user_input}':")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

def test_examples():
    """Run some test examples."""
    print("Loading Nepali words...")
    words = load_nepali_words()
    print(f"Loaded {len(words)} unique words")
    
    # Load or create word frequencies
    word_freq = load_word_frequencies()
    if not word_freq:
        print("Training word frequencies...")
        word_freq = save_word_frequencies(words)
    
    test_inputs = ["ति", "नमस", "काठ", "नेपा", "भाष"]
    for input_text in test_inputs:
        suggestions = get_suggestions(input_text, words, word_freq)
        print(f"\nInput: {input_text}")
        print("Suggestions:", suggestions)

def retrain_model():
    """Retrain the word frequency model."""
    print("Loading Nepali words...")
    words = load_nepali_words()
    print(f"Loaded {len(words)} unique words")
    
    print("Training new word frequencies...")
    word_freq = save_word_frequencies(words)
    print("Training completed!")
    
    # Test the new model
    print("\nTesting with example words:")
    test_inputs = ["ति", "नमस", "काठ", "नेपा", "भाष"]
    for input_text in test_inputs:
        suggestions = get_suggestions(input_text, words, word_freq)
        print(f"\nInput: {input_text}")
        print("Suggestions:", suggestions)

if __name__ == "__main__":
    print("Nepali Word Completion and Spelling Correction System")
    print("1. Run test examples")
    print("2. Start interactive demo")
    print("3. Retrain model")
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == "1":
        test_examples()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        retrain_model()
    else:
        print("Invalid choice!")