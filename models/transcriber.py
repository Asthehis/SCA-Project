import torch
import csv
import gc
import os
import threading

PLOT_LOCK = threading.Lock()
MIN_AVG_LOGPROB = -0.3

class Transcriber:
    def __init__(self, audio_path, model=None, language="fr", verbose=True):
        self.audio_path = audio_path
        self.model = model
        self.language = language
        self.device = "cuda"
        self.transcription = ""
        self.avg_logprob = 0
        self.should_reject = False
        self.verbose = verbose

    def run_transcription(self):

        try:
            if self.model is None:
                raise ValueError("Le modèle Whisper n'a pas été fourni.")
        
            if self.verbose:
                print(f"Transcription de {self.audio_path} en cours...")
        # transcription de l'audio
            result = self.model.transcribe(self.audio_path)
            self.avg_logprob = sum(s['avg_logprob'] for s in result['segments']) / len(result['segments'])
            # print(f"Score de confiance moyen : {avg_confidence:.3f}")


            self.segments = result.get("segments", [])

            self.transcription = " ".join([seg["text"].strip() for seg in self.segments])
            if self.verbose:
                print(f"Transcription de {self.audio_path} terminée.")

        except Exception as e:
            print(f"Erreur lors de la transcription : {e}")
            raise

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
    def save_transcript(self, path):
        if not self.segments:
            print("Aucune transcription à sauvegarder.")
            return
        
        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, f"{base_name}.txt")

        with PLOT_LOCK:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    for seg in self.segments:
                        start = self.format_time(seg["start"])
                        end = self.format_time(seg["end"])
                        f.write(f"[{start} - {end}] : {seg['text'].strip()}\n")

                print(f"Transcription sauvegardée : {output_path}")

            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")
                raise

    def save_csv(self):
        file_exists = os.path.isfile("data/transcripts_log.csv")
        with open("data/transcripts_log.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["file", "avg_logprob", "rejected"])

            writer.writerow([
                os.path.basename(self.audio_path),
                round(self.avg_logprob, 3) if self.avg_logprob else "",
                self.should_reject,
            ])

    @staticmethod
    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes:02d}:{sec:02d}"
    
    def transcribe(self):
        self.run_transcription()
        if self.avg_logprob < MIN_AVG_LOGPROB:
            self.should_reject = True
            print(f"La qualité de la transcription est trop mauvaise. Audio non retenu.")
            self.save_transcript(path="data/transcript/REJECTED")
        else:
            self.save_transcript(path="data/transcript")
        self.save_csv()