#!/usr/bin/env python3
"""
Audio Training Data Generator
Generates sine and sawtooth waves at multiple frequencies for ML training.
"""

import numpy as np
import wave
import struct
import os
import random

# ─── Configuration ──────────────────────────────────────────────────────────

SAMPLE_RATE = 44100       # Hz
DURATION = 2.0            # seconds per file
AMPLITUDE = 0.5           # max amplitude (keep < 1.0 to avoid clipping)
SEED = 42                 # for reproducible noise

# Frequency ranges for each class
SINE_FREQUENCIES = [220, 330, 440, 550, 660, 880, 1100, 1760]
SAW_FREQUENCIES = [220, 330, 440, 550, 660, 880, 1100, 1760]

# Number of variations per frequency (different phases, slight detune, noise levels)
VARIATIONS_PER_FREQ = 5

# Output directories
OUTPUT_DIR = "./training_audio"
CLASS_DIRS = {
    "sine": os.path.join(OUTPUT_DIR, "sine"),
    "sawtooth": os.path.join(OUTPUT_DIR, "sawtooth"),
}

# ─── Waveform Generators ────────────────────────────────────────────────────

def generate_sine_wave(freq, duration, sample_rate, amplitude=0.5, phase=0.0):
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return wave

def generate_sawtooth_wave(freq, duration, sample_rate, amplitude=0.5, phase=0.0):
    """Generate a band-limited sawtooth wave using additive synthesis."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t)

    # Additive synthesis: sum of harmonics with amplitude 1/n
    # Limit harmonics to Nyquist frequency to avoid aliasing
    max_harmonic = int((sample_rate / 2) / freq)

    for n in range(1, max_harmonic + 1):
        wave += (amplitude / n) * np.sin(2 * np.pi * n * freq * t + phase)

    # Normalize to prevent clipping
    # Theoretical sum of 1/n diverges, but we truncate
    # Normalize based on actual max amplitude
    max_val = np.max(np.abs(wave))
    if max_val > 0:
        wave = wave * (amplitude / max_val)

    return wave

def add_noise(wave, noise_level=0.01):
    """Add small amount of Gaussian noise for robustness."""
    noise = np.random.normal(0, noise_level, len(wave))
    return wave + noise

def apply_fade(wave, fade_duration=0.01, sample_rate=44100):
    """Apply fade in/out to avoid clicks at boundaries."""
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples * 2 >= len(wave):
        return wave

    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    wave = wave.copy()
    wave[:fade_samples] *= fade_in
    wave[-fade_samples:] *= fade_out
    return wave

# ─── WAV File Writer ────────────────────────────────────────────────────────

def save_wav(filename, wave_data, sample_rate=44100):
    """Save numpy array as 16-bit WAV file."""
    # Ensure no clipping
    wave_data = np.clip(wave_data, -1.0, 1.0)

    # Convert to 16-bit integers
    wave_int = (wave_data * 32767).astype(np.int16)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_int.tobytes())

    print(f"  Saved: {filename}")

# ─── Main Generator ─────────────────────────────────────────────────────────

def generate_dataset():
    """Generate the complete training dataset."""
    random.seed(SEED)
    np.random.seed(SEED)

    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dir_path in CLASS_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

    print("=" * 60)
    print("AUDIO TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Variations per frequency: {VARIATIONS_PER_FREQ}")
    print()

    total_files = 0

    # ─── Generate Sine Waves ──────────────────────────────────────────────
    print("[Generating Sine Waves]")
    for freq in SINE_FREQUENCIES:
        for i in range(VARIATIONS_PER_FREQ):
            # Random phase for variety
            phase = random.uniform(0, 2 * np.pi)
            # Slight detune (+/- 2%)
            detuned_freq = freq * random.uniform(0.98, 1.02)
            # Random noise level
            noise_level = random.uniform(0.0, 0.005)

            wave_data = generate_sine_wave(
                detuned_freq, DURATION, SAMPLE_RATE, 
                amplitude=AMPLITUDE, phase=phase
            )
            wave_data = add_noise(wave_data, noise_level)
            wave_data = apply_fade(wave_data, sample_rate=SAMPLE_RATE)

            filename = os.path.join(
                CLASS_DIRS["sine"], 
                f"sine_{freq}Hz_v{i+1}.wav"
            )
            save_wav(filename, wave_data, SAMPLE_RATE)
            total_files += 1

    print()

    # ─── Generate Sawtooth Waves ──────────────────────────────────────────
    print("[Generating Sawtooth Waves]")
    for freq in SAW_FREQUENCIES:
        for i in range(VARIATIONS_PER_FREQ):
            phase = random.uniform(0, 2 * np.pi)
            detuned_freq = freq * random.uniform(0.98, 1.02)
            noise_level = random.uniform(0.0, 0.005)

            wave_data = generate_sawtooth_wave(
                detuned_freq, DURATION, SAMPLE_RATE,
                amplitude=AMPLITUDE, phase=phase
            )
            wave_data = add_noise(wave_data, noise_level)
            wave_data = apply_fade(wave_data, sample_rate=SAMPLE_RATE)

            filename = os.path.join(
                CLASS_DIRS["sawtooth"],
                f"sawtooth_{freq}Hz_v{i+1}.wav"
            )
            save_wav(filename, wave_data, SAMPLE_RATE)
            total_files += 1

    print()
    print("=" * 60)
    print(f"Done! Generated {total_files} files in '{OUTPUT_DIR}/'")
    print()
    print("Directory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    sine/          -> {len(SINE_FREQUENCIES) * VARIATIONS_PER_FREQ} files")
    print(f"    sawtooth/      -> {len(SAW_FREQUENCIES) * VARIATIONS_PER_FREQ} files")
    print()
    print("Test files you can generate separately:")
    print("  - 880Hz sine (unseen frequency for sine class)")
    print("  - 880Hz sawtooth (your original test case)")
    print("  - 1000Hz sine (completely unseen frequency)")


def generate_test_file(waveform, freq, filename, duration=2.0):
    """Generate a single test file."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    if waveform == "sine":
        wave_data = generate_sine_wave(freq, duration, SAMPLE_RATE, AMPLITUDE)
    elif waveform == "sawtooth":
        wave_data = generate_sawtooth_wave(freq, duration, SAMPLE_RATE, AMPLITUDE)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")

    wave_data = apply_fade(wave_data, sample_rate=SAMPLE_RATE)
    save_wav(filename, wave_data, SAMPLE_RATE)


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset()

    # Optionally generate some test files
    print("\n[Generating Test Files]")
    generate_test_file("sine", 880, "./test_audio/sine_880Hz.wav")
    generate_test_file("sawtooth", 880, "./test_audio/sawtooth_880Hz.wav")
    generate_test_file("sine", 1000, "./test_audio/sine_1000Hz.wav")
    generate_test_file("sawtooth", 1000, "./test_audio/sawtooth_1000Hz.wav")
    print("\nAll done!")
