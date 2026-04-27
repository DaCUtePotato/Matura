use crate::fourier::{fft::*, split_audio::*};
use crate::nn::model::*;

use num::Complex;

mod fourier;
mod nn;

fn main() {
    // Define frames
    let frame_size: usize = 2048;
    let hop_size: usize = frame_size / 4; // For 75k% overlap
    // Read WAV file and apply framing
    let mut audio = hound::WavReader::open("audio.wav").unwrap();
    let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
    // Apply Hann windowing algorithm on the windows
    let hanned = hann(&frames);
    // Apply Fourier transform on the "Hanned" windows
    let mut fouriered: Vec<Vec<Complex<f32>>> = vec![];
    for frame in hanned {
        fouriered.push(ft(frame));
    }
    let normed = normVecVecComplex(&fouriered);
    let memory_lane: NN;
    let other_lane: NN;
    // run the model
    for frame in normed {}
}
