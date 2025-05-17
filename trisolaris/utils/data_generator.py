import os
import random
from pathlib import Path
from typing import Dict

# Sample text snippets for each category
SAMPLE_TEXTS = {
    'bible': [
        "In the beginning God created the heaven and the earth.",
        "And God said, Let there be light: and there was light.",
        "And God saw the light, that it was good: and God divided the light from the darkness."
    ],
    'shakespeare': [
        "To be, or not to be, that is the question.",
        "All the world's a stage, and all the men and women merely players.",
        "What's in a name? That which we call a rose by any other name would smell as sweet."
    ],
    'war_and_peace': [
        "Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes.",
        "The war was not a game of cards, but a serious business.",
        "The strongest of all warriors are these two — Time and Patience."
    ],
    'e40': [
        "Tell me when ta GO",
        "It's all good.",
        "GO STUPID"
    ],
    'gurbani': [
        "Ik Onkar, Sat Naam, Karta Purakh.",
        "Waheguru Ji Ka Khalsa, Waheguru Ji Ki Fateh.",
        "Guru Ram Das Ji, the fourth Guru, spread the message of love and equality."
    ]
}

def generate_text_files(output_dir: Path) -> Dict[str, str]:
    """
    Generate a set of text files with scrambled filenames.
    Each file contains content from a different source.
    Returns a mapping of filenames to their true categories.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate files
    ground_truth = {}
    for category, texts in SAMPLE_TEXTS.items():
        # Create 3 files per category
        for i in range(3):
            # Generate a random filename
            filename = f"file_{random.randint(1000, 9999)}.txt"
            
            # Write content to file
            with open(output_dir / filename, 'w') as f:
                # Write 5 random lines from the category
                content = random.sample(texts, min(5, len(texts)))
                f.write('\n'.join(content))
            
            # Store the ground truth
            ground_truth[filename] = category
    
    return ground_truth

if __name__ == "__main__":
    mapping = generate_text_files(Path("trisolaris/data/text_sorting"))
    print("Ground truth mapping:", mapping)
