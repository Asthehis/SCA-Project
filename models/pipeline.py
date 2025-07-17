import os
import whisper
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from audio_processor import AudioProcessor
from transcriber import Transcriber
from file_cleaner import FileCleaner

from threading import current_thread

NUM_THREADS_AUDIO = 4
NUM_TRHEADS_TRANSCRIPTION = 1
AUDIO_DIR = "data/audio/hospital"
TRANSCRIPT_DIR = "data/transcript"
CSV_DIR = "data"

def clean():
    # Nettoyage des audio & transcriptions
    for directory in [AUDIO_DIR, TRANSCRIPT_DIR, CSV_DIR]:
        cleaner = FileCleaner(directory)
        cleaner.remove_files()

def process_audio_pipeline(audio_path):
    cleaned_audio = []
    try:
        thread_name = current_thread().name
        print(f"\n[{thread_name}] : Traitement de la qualité audio de {audio_path}")

        processor = AudioProcessor(audio_path)
        if not processor.process():
            print(f"[{thread_name}] : Audio rejeté (qualité) pour {audio_path}")
            return
        else:
            cleaned_audio.append(processor.cleaned_path)

        print(f"[{thread_name}] : Qualité audio traité avec succès ({audio_path}).")
        return cleaned_audio

    except Exception as e:
        print(f"[{thread_name}] : Erreur lors du traitement de ({audio_path}) : {e}")

def transcription(audio_path, model):
    try:
        thread_name = current_thread().name
        transcriber = Transcriber(audio_path, model=model)
        transcriber.transcribe()
    except Exception as e:
        print(f"[{thread_name}] : Erreur lors du traitement de ({audio_path}) : {e}")


def main():
    clean()
    whisper_model = whisper.load_model("large-v2")

    audio_files = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.endswith(".wav") and not f.endswith("_cleaned.wav")
    ]

    if not audio_files:
        print("Aucun fichier audio à traiter.")
        return

    print(f"Lancement du traitement avec {NUM_THREADS_AUDIO} threads...\n")
    cleaned_audios = []

    with ThreadPoolExecutor(max_workers=NUM_THREADS_AUDIO) as executor:
        futures = {executor.submit(process_audio_pipeline, audio): audio for audio in audio_files}

        for future in as_completed(futures):
            result = future.result()
            if result:  # non vide
                cleaned_audios.extend(result)

    if not cleaned_audios:
        print("Aucun audio n'a passé le filtre qualité. Fin du programme.")
        return

    print(f"\nLancement de la transcription avec {NUM_TRHEADS_TRANSCRIPTION} threads...\n")
    with ThreadPoolExecutor(max_workers=NUM_TRHEADS_TRANSCRIPTION) as executor:
        transcribe_fn = partial(transcription, model=whisper_model)
        transcriptions = [executor.submit(transcribe_fn, audio) for audio in cleaned_audios]

        for future in as_completed(transcriptions):
            future.result()

    print("\nTraitement terminé pour tous les fichiers.")

if __name__ == "__main__":
    main()