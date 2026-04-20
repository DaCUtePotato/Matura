use crate::Fourier::split_audio::*;

#[allow(non_snake_case)]
mod Fourier;

fn main() {
    let frame_size: usize = 20;
    let hop_size: usize = 10;
    let mut audio = hound::WavReader::open("audio.wav").unwrap();
    let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
    println!("{:?}", frames);
}
