use crate::fourier::{fft::*, split_audio::*};
use crate::nn::model::*;

mod fourier;
mod nn;

fn main() {
    // Define frames
    let frame_size: usize = 2048;
    let hop_size: usize = frame_size / 4; // For 75% overlap
    let memory_lane_size = 128;
    let num_classes = 3;
    let learning_rate: f64 = 1e-4;
    let actual: Vec<f64> = vec![];
    // Read WAV file and apply framing
    let mut audio = hound::WavReader::open("audio.wav").unwrap();
    let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
    // Apply Hann windowing algorithm on the windows
    let hanned = hann(&frames);
    // Apply Fourier transform on the "Hanned" windows
    let mut fouriered: Vec<Vec<f64>> = vec![];
    for frame in hanned {
        fouriered.push(normVecComplex(&ft(frame)));
    }
    // Train the model
    let mut model = Model::new(&frame_size, &memory_lane_size, &num_classes);
    let (hidden_states, output) = model.train_forward(&fouriered);
    model.gitgud(&learning_rate, &output, &actual, &hidden_states);
}
