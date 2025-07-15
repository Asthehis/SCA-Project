import whisperx
import gc
import os
import torch

def transcription(audio_file):
    """
    Cette fonction permet la transcription d'un fichier audio en fichier .txt.
    Elle permet d'afficher la transcription et de la sauvegarder en .txt.

    -audio_file: chemin vers l'audio à trancrire
    """
    # on détermine les constantes importantes
    device = "cuda"
    batch_size = 4
    compute_type = "float16"
    language = "fr"

    # chargement du modèle whisperX
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language)

    # transcription de l'audio
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    segments = result["segments"]
    # print(segments)

    # on supprime le modèle
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # alignement des segments
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)

    segments = result["segments"]
    # print(segments)

    # on supprime le modèle
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    transcription_text = " ".join([seg["text"].strip() for seg in segments])
    avg_logprobs = [seg["avg_logprob"] for seg in segments if "avg_logprob" in seg]
    global_conf = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else -999
    rejeter = False

    if len(transcription_text.strip()) < 10:
        rejeter = True
    if global_conf < -1.2:
        rejeter = True
    avg_count = sum(1 for avg in avg_logprobs if avg > 0.6)
    if avg_count > len(avg_logprobs) / 2:
        rejeter = True

    print(f"→ Confiance : {global_conf:.2f} | Longueur : {len(transcription_text.strip())} | Rejeté : {rejeter}")

    if not rejeter:
        save_to_txt(audio_file, segments)
    else:
        with open("data/transcript/rejected_transcriptions.txt", "a") as f:
            f.write(f"REJECTED: {audio_file}\n")

    return transcription_text, rejeter

def save_to_txt(audio_file, segments):
    # format de temps, permet d'afficher les timecodes
    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes:02d}:{sec:02d}"

    # nom de fichier de sortie, on enregistre dans un fichier différent de l'audio
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_txt = os.path.join("data/transcript", f"{base_name}.txt")

    # sauvegarde de la transcription en fichier txt
    with open(output_txt, "w", encoding="utf-8") as f:
        for segment in segments:
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start} - {end}] : {text}\n")

    print(f"Transcription sauvegardée : {output_txt}")

# script principal
if __name__ == "__main__":
    audio_dir = "data/audio/hospital"
    for file_name in os.listdir(audio_dir):
        if file_name.lower().endswith("_cleaned.wav"):
            audio_path = os.path.join(audio_dir, file_name)
            print(f"\nTranscription de : {audio_path}")
            transcription(audio_path) 