import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_to_gloss_dir.rules import *

def main():
    text = input("Entrez le texte à convertir en glosses: ")
    print("Texte: ", text)

    gloss = text_to_gloss(text, "fr")

    print("Glosses:", gloss)

if __name__ == "__main__":
    main()