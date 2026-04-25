use hound::WavReader;
use std::fs::File;
use std::io::BufReader;

// This function splits a given wave file of format Vec<f32> into segments
// that are frame_size long. The overlap is calculated through frame_size-hop_size,
// this is good to keep in mind :)
pub fn split_audio(audio: Vec<f32>, frame_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    let mut i: usize = 0;
    let mut split: Vec<Vec<f32>> = vec![];
    while i + frame_size <= audio.len() {
        let frame: Vec<f32> = audio[i..i + frame_size].to_vec();
        split.push(frame);
        i += hop_size;
    }
    split
}

// This function converts an input variable wave file into a Vec<f32> with very
// fancy matches :)
pub fn unhound(audio: &mut WavReader<BufReader<File>>) -> Vec<f32> {
    match audio.spec().sample_format {
        hound::SampleFormat::Float => audio.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => audio
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    }
}

// Holy complex, this tapers the ends of the frames by basically scaling the samples
// by a squared sine wave (sin^2(πk/N)=0.5 * (1 - cos(2πk/N))) so the frame doesn't
// get cut off but instead gets scaled by a more extreme sine curve :3
// This prevents spectral leakage
pub fn hann(audio: &[Vec<f32>]) -> Vec<Vec<f32>> {
    audio
        .iter()
        .map(|s| {
            s.iter()
                .enumerate()
                .map(|(k, sample)| {
                    sample
                        * (std::f32::consts::PI * k as f32 / (s.len() as f32 - 1.))
                            .sin()
                            .powi(2)
                })
                .collect()
        })
        .collect()
}

// Testing :3
//
// fn main() {
//     let mut audio = hound::WavReader::open("audio.wav").unwrap();
//     let frame_size: usize = 10;
//     let hop_size: usize = 10;
//     let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
//     println!("{:?}", frames)
// }
