import os
import subprocess

def convert_m4a_to_wav(directory):
    """
    Convertit tous les fichiers .m4a d'un dossier en .wav sans perte de qualité.
    
    Args:
        directory (str): Chemin du dossier contenant les fichiers .m4a.
    """
    if not os.path.isdir(directory):
        print("Le dossier spécifié n'existe pas.")
        return

    for filename in os.listdir(directory):
        if filename.lower().endswith(".m4a"):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename.rsplit(".", 1)[0] + ".wav")

            # Commande FFmpeg pour convertir sans perte de qualité
            command = [
                "ffmpeg", "-i", input_path,
                "-acodec", "pcm_s16le",  # Format WAV standard
                "-ar", "44100",  # Taux d'échantillonnage standard (peut être ajusté)
                "-ac", "1",  # Stéréo
                output_path
            ]

            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ Converti : {filename} → {output_path}")
            except subprocess.CalledProcessError:
                print(f"❌ Erreur lors de la conversion de {filename}")

if __name__ == "__main__":
    dossier = input("Entrez le chemin du dossier contenant les fichiers .m4a : ")
    convert_m4a_to_wav(dossier)
