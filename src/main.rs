use crate::fourier::{fft::*, split_audio::*};
use crate::nn::model::*;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

mod fourier;
mod nn;

fn main() {
    let mut fourier: bool = false;
    let mut train: bool = false;
    let mut run: bool = false;

    let args: Vec<String> = env::args().collect();
    for arg in args {
        match arg.as_str() {
            "fourier" => fourier = true,
            "train" => train = true,
            "run" => run = true,
            _ => {}
        }
    }
    if fourier {
        // Definitions
        let frame_size: usize = 2048;
        let hop_size: usize = frame_size / 4; // For 75% overlap
        let label: Vec<f64> = vec![0., 1., 0.];
        let path = "./2";

        // Read WAV file and apply framing
        let audios = read_wavs(path);

        let mut data_collection: Vec<Data> = vec![];
        for mut audio in audios {
            let class = label.clone();

            let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
            // Apply Hann windowing algorithm on the windows
            let hanned = hann(&frames);
            // Apply Fourier transform on the "Hanned" windows
            let mut fourier: Vec<Vec<f64>> = vec![];
            for frame in hanned {
                fourier.push(normVecComplex(&ft(frame)));
            }
            data_collection.push(Data { fourier, class });
        }
        // Save labelled data
        let json = serde_json::to_string_pretty(&data_collection)
            .expect("Oh magaaaaud failed to convert ur goofy struct to string D:");

        // Save weights
        let mut file = File::create("class2.json").unwrap();
        file.write_all(json.as_bytes())
            .expect("Oh whoopsies, writing no worky work");
        println!("Data saved successfully...")
    }
    if train {
        // Define model parameters
        let memory_lane_size = 128;
        let num_classes = 3;
        let learning_rate: f64 = 1e-3;

        // This should be equivalent to the fourier frame size
        let input_size: usize = 2048;

        // Read data
        let content1 = fs::read_to_string("class1.json").expect("Reading data didn't work");
        let data_collection1: Vec<Data> =
            serde_json::from_str(&content1).expect("Ran into a problem trying to deserialize data");

        let content2 = fs::read_to_string("class2.json").expect("Reading data didn't work");
        let data_collection2: Vec<Data> =
            serde_json::from_str(&content2).expect("Ran into a problem trying to deserialize data");
        let mut data_collection = data_collection1;
        data_collection.extend(data_collection2);
        let mut model = Model::new(&input_size, &memory_lane_size, &num_classes);

        // Train the model
        // This part has to be made better, I think it should just pick a data from data_collection
        for _ in 0..100 {
            for data in &data_collection {
                let (data, class) = (&data.fourier, &data.class);
                let (hidden_states, output) = model.train_forward(data);
                model.gitgud(&learning_rate, &output, class, &hidden_states);
            }
        }
        // Save weights
        let json = serde_json::to_string(&model)
            .expect("Idk uhm error ehehe making the model a json failed aifink :) Have funnn :D");

        let mut file = File::create("Timmy.json").unwrap();
        file.write_all(json.as_bytes())
            .expect("Oh whoopsies, writing no worky work");
        println!("Model saved successfully...")
    }
    if run {
        // Define frames
        let frame_size: usize = 2048;
        let hop_size: usize = frame_size / 4; // For 75% overlap

        // Read WAV file and apply framing
        let mut audio = hound::WavReader::open("./1/audio.wav").unwrap();
        let frames = split_audio(unhound(&mut audio), frame_size, hop_size);
        // Apply Hann windowing algorithm on the windows
        let hanned = hann(&frames);
        // Apply Fourier transform on the "Hanned" windows
        let mut fouriered: Vec<Vec<f64>> = vec![];
        for frame in hanned {
            fouriered.push(normVecComplex(&ft(frame)));
        }

        // Read weights
        let content = fs::read_to_string("Timmy.json").expect("Reading Timmy didn't work");
        let model: Model =
            serde_json::from_str(&content).expect("Ran into a problem trying to deserialize Timmy");

        let output: Vec<f64> = model.forward(&fouriered);
        println!("Result: {:?}", output)
    }
}
