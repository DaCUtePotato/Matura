use crate::fourier::{fft::*, split_audio::*};
use crate::nn::model::*;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;

mod fourier;
mod nn;

fn main() {
    let mut fourier: bool = false;
    let mut train: bool = false;
    let mut run: bool = false;
    // Define frames
    let frame_size: usize = 2048;
    let hop_size: usize = frame_size / 4; // For 75% overlap
    let memory_lane_size = 128;
    let num_classes = 3;
    let learning_rate: f64 = 1e-3;
    let actual: Vec<f64> = vec![1., 0., 0.];

    let args: Vec<String> = env::args().collect();
    for arg in args {
        match arg.as_str() {
            "fourier" => fourier = true,
            "train" => train = true,
            "run" => run = true,
            _ => {}
        }
    }
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
    if train {
        // Train the model
        let mut model = Model::new(&frame_size, &memory_lane_size, &num_classes);
        for _ in 0..100 {
            let (hidden_states, output) = model.train_forward(&fouriered);
            model.gitgud(&learning_rate, &output, &actual, &hidden_states);
        }
        let json = serde_json::to_string(&model)
            .expect("Idk uhm error ehehe making the model a json failed aifink :) Have funnn :D");

        // Save weights
        let mut file = File::create("Timmy.json").unwrap();
        file.write_all(json.as_bytes())
            .expect("Oh whoopsies, writing no worky work");
        println!("Model saved successfully...")
    }
    if run {
        // Read weights
        let content = fs::read_to_string("Timmy.json").expect("Reading Timmy didn't work");
        let model: Model =
            serde_json::from_str(&content).expect("Ran into a problem trying to deserialize Timmy");

        let output: Vec<f64> = model.forward(&fouriered);
        println!("Result: {:?}", output)
    }
}
