use crate::fourier::{fft::*, split_audio::*};
use num::Complex;

mod fourier;
mod nn;

fn main() {
    let frame_size: usize = 2048;
    let hop_size: usize = frame_size / 4; // For 75k% overlap
    let mut audio = hound::WavReader::open("audio.wav").unwrap();
    let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
    let hanned = hann(&frames);
    let mut fouriered: Vec<Vec<Complex<f32>>> = vec![];
    for frame in hanned {
        fouriered.push(ft(frame));
    }
    let normed = normVecVecComplex(&fouriered);
    // Map, iter and collect shinanigans to actually read the thing
    println!("{:?}", normed);
}
