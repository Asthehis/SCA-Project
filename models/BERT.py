import json
import re
import os
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification

tokenizer = CamembertTokenizer.from_pretrained("./camembert_custom_model")
model = CamembertForSequenceClassification.from_pretrained("./camembert_custom_model")

# Fonctions utilitaires
def load_keywords(file_path):
    """ 
    Cette fonction permet d'ouvrir les fichiers .json contenant les mots-clés SCA et non SCA, et d'en récupérer les données.
    Retourne une liste composé de dictionnaire, chaque dic représente un mot-clé. Celui-ci est associé à ses synonymes, ses triggers et sa sévérité.

    - file_path : le chemin d'accès au fichier
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("keywords", []) # On récupère toutes les infos sous la balise "keywords"
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return [] # Affichage d'un message d'erreur et renvoie d'une liste vide si le fichier n'est pas trouvé


def load_transcript(file_path):
    """
    Cette fonction permet d'ouvrir le fichier .txt contenant la transcription de l'audio à analyser.
    Retourne une liste de dictionnaire, chaque dic est une phrase prononcée, avec le timecode et le texte. 

    - file_path : le chemin d'accès au fichier
    """
    transcript = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file: # On parcourt chaque ligne du fichier
                if " : " in line: # Le texte est sous ce format : "0.03s - 5.48s: Bonjour monsieur"
                    timecode, text = line.strip().split(" : ", 1) # On peut donc récupérer le timecode à gauche et le texte à droite du symbole ':'
                    transcript.append({"timecode": timecode, "text": text}) # A modif ici car on utilise plus le speaker
    except FileNotFoundError:
        print(f"Erreur : fichier {file_path} introuvable.") # Affichage d'un message d'erreur si le fichier n'est pas trouvé
    return transcript


def get_matched_keywords(text, keyword_entry):
    """
    Cette fonction permet d'obtenir les mots-clés (et synonymes) présents dans une phrase. 
    Retourne les mots-clés détectés dans le texte.

    - text : le texte dans lequel on cherche les mots.
    - keyword_entry : un dictionnaire contenant :
        - "word" : le mot-clé principal
        - "synonyms" : une liste de synonymes (optionnelle)

    Exemple :
    keyword_entry = {"word": "homme", "synonyms": ["Mari", "frère", "fils", "époux", "compagnon", "père" ]}
    text = "Mon mari a mal à la poitrine."
    => retourne ["mari"]
    """
    keywords = [keyword_entry.get("word", "")] + keyword_entry.get("synonyms", []) # On fait une liste avec tous les mots clés et leurs synonymes
    return [ #Liste en compréhension : on garde uniquement les mots détectés dans le texte
        word for word in keywords 
        if word and re.search( # Expression pour rechercher le mot dans 'text' (ici la phrase)
            rf"\b{re.escape(word)}\b", # \b = délimiteur de mot entier (ex : dou ne matchera pas douleur)
            text, 
            re.IGNORECASE # ignore maj et min
        )
    ]



def is_positive_response(context, keyword):
    """
    Cette fonction permet de déterminer si une réponse est positive ou non.
    Elle renvoie un booléen. True si la réponse est affirmative, false sinon.

    -context : le contexte contenant le mot clé, la phrase où est détecté le mot et les 2 phrases avant et après.
    -keyword : le mot clé présent dans la phrase et en cours d'analyse
    """
    sentence = f"{context}\nMot-clé : {keyword}"
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax().item()

    if predicted_class == 1:
        return True  # Affirmatif
    elif predicted_class == 0:
        return False  # Négatif
    else:
        return False  # Sécurité


def analyze_transcript(transcript, keywords):
    """
    Retourne le dic des mots clés valides.
    -transcript : la transcription de l'audio à analyser
    -keywords : les mots-clés
    """
    validated_words = {} # dic vide qui va être remplis par les mots-clés validés
    
    for i, line in enumerate(transcript): # transcript : liste de dic, enumerate : 
        # print("line:", line)
        # print(line["text"])
        for entry in keywords: # On parcourt les mots-clés
            base_word = entry.get("word", "")
            matched_words = get_matched_keywords(line["text"], entry) # On regarde si des mots clés match avec les mots dans une ligne

            for matched in matched_words: # On vérifie maintenant que les mots 'matché' sont valides ou non
                print(f"Mot trouvé : '{matched}' (lié à '{base_word}') à la ligne {i + 1}")
                # Contexte local (2 lignes avant et 2 après)
                start = max(i - 2, 0)
                end = min(i + 3, len(transcript))
                context_lines = [f"{l['timecode']} : {l['text']}" for l in transcript[start:end]]
                context = "\n".join(context_lines)

                if is_positive_response(context, matched): # Si le mots clés est valide alors on l'ajoute au dic
                    validated_words[base_word] = {
                        "severity": entry.get("severity"),
                        "catégorie_diagnostic": entry.get("catégorie_diagnostic"),
                        "terrain_à_risque": entry.get("terrain_à_risque")
                    }


    return validated_words


def save_validated_words(validated_words, output_file):
    """
    Cette fonction permet de sauvegarder les mots-clés validés.
    Créé un fichier .txt avec les mots et leur sévérité.

    -validated_words : liste des mots-clés validés
    -output_file : le chemin où sera sauvegarder le fichier
    """
    mode = "a" if os.path.exists(output_file) else "w"
    with open(output_file, mode, encoding="utf-8") as file:
        for word, info in validated_words.items():
            file.write(f"{word} (Sévérité: {info.get('severity')}, Diagnostic: {info.get('catégorie_diagnostic')}, Risque: {info.get('terrain_à_risque')})\n")



# Script principal
if __name__ == "__main__":

    keywords = load_keywords("data/sca_non_sca_words.json")

    transcript_dir = "data/transcript"
    for file_name in os.listdir(transcript_dir):
        if file_name.lower().endswith(".txt"):
            transcript_path = os.path.join(transcript_dir, file_name)
            transcript = load_transcript(transcript_path)

            print("Analyse des mots-clés...")
            validated = analyze_transcript(transcript, keywords)

            base_name = os.path.splitext(os.path.basename(file_name))[0]
            output = os.path.join("data/keywords", f"{base_name}_keywords.txt")

            for word, info in validated.items():
                save_validated_words({word: info}, output)

            print("\nMots-clés validés enregistrés dans les fichiers de sortie.")
