use crate::fourier::{fft::*, split_audio::*};
use crate::nn::model::*;
use rand::rng;
use rand::seq::SliceRandom;
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
        let label: Vec<f64> = vec![1., 0.];

        // Will take all .wav files in this directory
        let path = "./training_audio/sine";

        // File name of the raw fourier data
        let data_file_name = "class1.json";

        // Read WAV files
        let audios = read_wavs(path);

        let mut data_collection: Vec<Data> = vec![];
        for mut audio in audios {
            // Ownership things
            let class = label.clone();

            // Split audio
            let frames = split_audio(unhound(&mut audio), frame_size, hop_size);

            // Apply Hann windowing algorithm on the windows
            let hanned = hann(&frames);

            // Apply Fourier transform on the "Hanned" windows
            let mut fourier: Vec<Vec<f64>> = vec![];
            for frame in hanned {
                fourier.push(normVecComplex(&ft(frame)));
            }

            // Add the transformed audio to the collection
            data_collection.push(Data { fourier, class });
        }
        // Save labelled data
        let json = serde_json::to_string_pretty(&data_collection)
            .expect("Oh magaaaaud failed to convert ur goofy struct to string D:");

        // Save weights
        let mut file = File::create(data_file_name).unwrap();
        file.write_all(json.as_bytes())
            .expect("Oh whoopsies, writing no worky work");
        println!("Data saved successfully...")
    }
    if train {
        // Define model parameters
        let memory_lane_size = 512;
        let num_classes = 2;
        // This must be equivalent to the fourier frame size
        let input_size: usize = 2048;

        // Define training parameters
        let learning_rate: f64 = 1e-3;
        let epochs = 15;

        // Read data
        let mut data_collection: Vec<Data> = vec![];
        let data = ["class1.json", "class2.json"];
        for content in data {
            let stuff = fs::read_to_string(content).expect("Failed to read data");
            let data_collection_temp: Vec<Data> =
                serde_json::from_str(&stuff).expect("Failed to deserialize content");
            data_collection.extend(data_collection_temp);
        }
        let mut model = Model::new(&input_size, &memory_lane_size, &num_classes);

        let mut progress: usize = 0;
        // Train the model
        let mut rng = rng();
        for epoch in 0..epochs {
            data_collection.shuffle(&mut rng); // shuffle each epoch
            for data in &data_collection {
                println!("-----------------------------------------");
                println!("On epoch {} out of {}", epoch + 1, epochs);
                let (fourier, class) = (&data.fourier, &data.class);
                let (hidden_states, output) = model.train_forward(fourier);
                model.gitgud(&learning_rate, &output, class, &hidden_states);
                println!(
                    "Progress: {}%",
                    100. * (progress as f64) / (data_collection.len() as f64 * epochs as f64)
                );
                progress += 1;
            }
            println!("Epoch {}/{} done!", epoch + 1, epochs);
            println!("-----------------------------------------");
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
        let audio_dir = "audio.wav";

        // Read WAV file and apply framing
        let mut audio = hound::WavReader::open(audio_dir).unwrap();
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
