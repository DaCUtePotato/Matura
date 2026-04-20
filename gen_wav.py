import wave, struct, math

filename = "audio.wav"
sample_rate = 44100
num_channels = 1
sample_width = 2  # 16-bit
duration_seconds = 1
num_samples = sample_rate * duration_seconds

with wave.open(filename, 'w') as wf:
    wf.setnchannels(num_channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    for i in range(num_samples):
        # Simple 440 Hz sine wave
        value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        wf.writeframes(struct.pack('<h', value))

print(f"Generated {filename}: {num_samples} samples @ {sample_rate}Hz, 16-bit mono")
