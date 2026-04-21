use crate::Fourier::{fft::*, split_audio::*};
use num::Complex;

#[allow(non_snake_case)]
mod Fourier;

fn main() {
    let frame_size: usize = 20;
    let hop_size: usize = 10;
    let mut audio = hound::WavReader::open("audio.wav").unwrap();
    let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
    let mut fouriered: Vec<Vec<Complex<f32>>> = vec![];
    for frame in frames {
        fouriered.push(ft(frame));
    }
    println!("{:?}", fouriered);
}
