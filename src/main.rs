// Imports...
use crate::fourier::{fft::*, split_audio::*};
use crate::nn::helpers::*;
use crate::nn::model::*;
use rand::seq::SliceRandom;
use rand::{RngExt, rng};
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;

// For multi-threading
use rayon::prelude::*;

mod fourier;
mod nn;

// Constants
// Constant used for fouriering and training
const FRAME_SIZE: usize = 2048;
const HOP_SIZE: usize = FRAME_SIZE / 4;

// Define model parameters
const MEMORY_LANE_SIZE: usize = 1024;
const NUM_CLASSES: usize = 2;
const LOSS_FILE_PATH: &str = "losses";
const VAL_LOSS_FILE_PATH: &str = "validation_losses";
const BATCH_SIZE: usize = 4;

// Define training parameters
const LEARNING_RATE: f64 = 1e-3; // Initial learning rate
const DEC_CONST: f64 = 0.1; // For short trainings, fairly fast decay; half-life of ~6 epochs
const MODEL_FILE_PATH: &str = "Timmy.json";
const TRAINING_SPLIT: f64 = 0.8;
const DATA: [&str; NUM_CLASSES] = ["class1.json", "class2.json"]; // Data for all classes
const PATIENCE: usize = 5; // How many epochs do you let validation loss be less than a threshold?
// Use only if using ratio to stop training
// const VAL_THRESHOLD: f64 = 0.9; // Validation loss ratio threshold

// Running parameters
const AUDIO_DIR: &str = "test2.wav";

// Temporary
const LR_FILE_PATH: &str = "lr";

// Plotting
const GRAPH_FILE_PATH: &str = "graph.svg";

fn main() {
    // For handling arguments
    let mut fourier: bool = false;
    let mut training: bool = false;
    let mut run: bool = false;
    let mut hyper: bool = false;
    let mut train_continue: bool = false;
    let mut plot = false;
    let mut accuracy = false;

    let args: Vec<String> = env::args().collect();
    for arg in args {
        match arg.as_str() {
            "fourier" => fourier = true,
            "train" => training = true,
            "run" => run = true,
            "hyperparameters" => hyper = true,
            "train-continue" => train_continue = true,
            "plot" => plot = true,
            "accuracy" => accuracy = true,
            _ => {}
        }
    }
    if fourier {
        // Definitions
        let hop_size: usize = FRAME_SIZE / 4; // For 75% overlap
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
            let frames = split_audio(unhound(&mut audio), FRAME_SIZE, hop_size);

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
            .expect("Error turning data collection to string");
        let mut file = File::create(data_file_name).unwrap();
        file.write_all(json.as_bytes())
            .expect(&format!("Failed to write to {}", data_file_name));
        println!("Data saved successfully...")
    }
    if training {
        // Read data from files
        let all_data = read_data(&DATA);

        // Split data into training and validation sets
        let (training_set, validation_set) = split_data(all_data, TRAINING_SPLIT);

        // Create model
        let mut model = Model::new(&FRAME_SIZE, &MEMORY_LANE_SIZE, &NUM_CLASSES);

        // Initializing dynamic constants for loop
        let mut running = true; // Always leave this! Even if you're testing!
        let mut epoch = 1;
        let mut current_lr = LEARNING_RATE;
        let mut loss_window = [f64::INFINITY; PATIENCE];
        let mut best_val_loss: f64 = f64::INFINITY;
        while running {
            // For testing:
            // while epoch <= 20 {
            // For graphing + information: Write lr to file
            write_to_file(LR_FILE_PATH, &current_lr.to_string());

            // Train the model for one epoch, i.e. once through all of the training set
            train(
                BATCH_SIZE,
                current_lr,
                &training_set,
                &mut model,
                Some(&LOSS_FILE_PATH.to_string()),
            );

            // Calculate validation loss after every epoch
            let validation_loss = calculate_validation_loss(
                &model,
                &validation_set,
                Some(&VAL_LOSS_FILE_PATH.to_string()),
            );

            // Save checkpoint if current val loss is the best
            if validation_loss < best_val_loss {
                save_model(&model, MODEL_FILE_PATH);
                best_val_loss = validation_loss;
            }

            // Terminate if current val loss is worse than any before it
            if loss_window.iter().filter(|&x| x < &validation_loss).count() == PATIENCE {
                println!("Validation loss {}", validation_loss);
                println!("Validation loss worsened, terminating training");
                running = false;
                break;
            }

            // Shift window and add newest val loss
            loss_window.rotate_left(1);
            loss_window[PATIENCE - 1] = validation_loss;

            println!("Validation loss: {}", validation_loss);

            // Modify learning rate
            current_lr = calc_new_lr_exp(LEARNING_RATE, DEC_CONST, epoch);
            epoch += 1;

            // To kill or not to kill...
            // If using ratios:
            // running = !kill_no_kill_ratios(&loss_window, VAL_THRESHOLD);
        }
        println!(
            "Training stopped. Reason: {}",
            if running {
                "Idk you did something"
            } else {
                "Validation loss difference did not meet expectations..."
            }
        );

        // Save model
        save_model(&model, MODEL_FILE_PATH);

        // Print accuracy
        println!("Accuracy: {}", calc_accuracy(&model, &validation_set));
    }
    if train_continue {
        // Read data from files
        let all_data = read_data(&DATA);

        // Split data into training and validation sets
        let (training_set, validation_set) = split_data(all_data, TRAINING_SPLIT);

        // Read weights, define model
        let content = fs::read_to_string("Timmy.json").expect("Reading Timmy didn't work");
        let mut model: Model =
            serde_json::from_str(&content).expect("Ran into a problem trying to deserialize Timmy");

        let mut running = true;
        let mut epoch = 1;
        let mut current_lr = LEARNING_RATE;
        let mut loss_window = [f64::INFINITY; PATIENCE];
        let mut best_val_loss: f64 = f64::INFINITY;

        while running {
            // while epoch <= 20 {
            // For graphing + information: Write lr to file
            write_to_file(LR_FILE_PATH, &current_lr.to_string());
            // Train the model for one epoch, i.e. once through all of the training set
            train(
                BATCH_SIZE,
                current_lr,
                &training_set,
                &mut model,
                Some(&LOSS_FILE_PATH.to_string()),
            );

            // Calculate validation loss after every epoch
            let validation_loss = calculate_validation_loss(
                &model,
                &validation_set,
                Some(&VAL_LOSS_FILE_PATH.to_string()),
            );

            // Save checkpoint if current val loss is the best
            if validation_loss < best_val_loss {
                save_model(&model, MODEL_FILE_PATH);
                best_val_loss = validation_loss;
            }

            // Terminate if current val loss is worse than any before it
            if loss_window.iter().filter(|&x| x < &validation_loss).count() == PATIENCE {
                println!("Validation loss {}", validation_loss);
                println!("Validation loss worsened, terminating training");
                running = false;
                break;
            }

            // Shift window and add newest val loss
            loss_window.rotate_left(1);
            loss_window[PATIENCE - 1] = validation_loss;

            println!("Validation loss: {}", validation_loss);

            // Modify learning rate
            current_lr = calc_new_lr_exp(LEARNING_RATE, DEC_CONST, epoch);
            epoch += 1;

            // To kill or not to kill
            // If using ratios
            // running = !kill_no_kill(&loss_window, VAL_THRESHOLD);
        }
        println!(
            "Training stopped. Reason: {}",
            if running {
                "Idk you did something"
            } else {
                "Validation loss difference did not meet expectations..."
            }
        );
        // Save model
        save_model(&model, MODEL_FILE_PATH);

        // Print accuracy
        println!("Accuracy: {}", calc_accuracy(&model, &validation_set));
    }
    if hyper {
        // Define model parameters
        let num_classes = 2;

        // This must be equivalent to the fourier frame size
        let input_size: usize = 2048;

        // Define training parameters
        let epochs = 15;

        // Define ranges
        let init_learning_rate_exponent_range: Vec<f64> = vec![2., 3.];
        let final_learning_rate_exponent_range: Vec<f64> = vec![4., 6.];
        let memory_lane_sizes: Vec<usize> = vec![64, 128, 256, 512, 1024];
        let batch_size_range: Vec<usize> = vec![1, 4, 8];

        // Hyper parameters/initialization
        let trials = memory_lane_sizes.len() * 20; //Number of full training runs
        let mut best_lr: f64 = 1e-3;
        let mut best_mem_lane = 512;
        let mut best_val_loss = f64::INFINITY; //MUAHMUAH INFINITE LOSS
        let mut best_final_lr = 0.;
        let mut best_batch_size = 1;

        // Define splits
        let training_split = 0.8;
        let validation_split = 1. - training_split;

        // Read data from files
        let mut data_collection: Vec<Data> = vec![];
        let data = ["class1.json", "class2.json"];
        for content in data {
            let stuff = fs::read_to_string(content).expect("Failed to read data");
            let data_collection_temp: Vec<Data> =
                serde_json::from_str(&stuff).expect("Failed to deserialize content");
            data_collection.extend(data_collection_temp);
        }

        // Split data into training and validation sets
        let mut rng = rng();
        data_collection.shuffle(&mut rng);
        let split_index = (data_collection.len() as f32 * validation_split).round() as usize;
        let training_set = data_collection.split_off(split_index);
        let validation_set = data_collection;

        for i in 0..trials {
            // Randomize hyperparameters
            let init_lr_exponent: f64 = -rng.random_range(
                init_learning_rate_exponent_range[0]..init_learning_rate_exponent_range[1],
            ) as f64;
            let init_learning_rate: f64 = f64::powf(10., init_lr_exponent);

            let final_lr_exponent: f64 = -rng.random_range(
                final_learning_rate_exponent_range[0]..final_learning_rate_exponent_range[1],
            ) as f64;
            let final_learning_rate: f64 = f64::powf(10., final_lr_exponent);

            let memory_lane_size = memory_lane_sizes[rng.random_range(0..memory_lane_sizes.len())];

            let batch_size = batch_size_range[rng.random_range(0..batch_size_range.len())];

            // Create model
            let mut model = Model::new(&input_size, &memory_lane_size, &num_classes);

            if batch_size == 1 {
                // Train the model on training set
                let mut progress: usize = 0;
                for epoch in 0..epochs {
                    let current_lr = init_learning_rate
                        - (init_learning_rate - final_learning_rate) * (epoch + 1) as f64
                            / epochs as f64;
                    for data in &training_set {
                        println!("-----------------------------------------");
                        println!("On epoch {} out of {}", epoch + 1, epochs);
                        let (fourier, class) = (&data.fourier, &data.class);
                        let (hidden_states, output) = model.train_forward(fourier);
                        model.gitgud(&current_lr, &output, class, &hidden_states);
                        println!(
                            "Progress: {}%",
                            100. * (progress as f64) / (training_set.len() as f64 * epochs as f64)
                        );
                        progress += 1;
                    }
                    println!("Epoch {}/{} done!", epoch + 1, epochs);
                    println!("-----------------------------------------");
                }
            } else {
                let chunks = training_set.chunks(batch_size).collect::<Vec<&[Data]>>();

                let mut all_states = Vec::with_capacity(batch_size);
                let mut all_output = Vec::with_capacity(batch_size);
                let mut all_actual = Vec::with_capacity(batch_size);

                let mut progress = 0;
                for epoch in 0..epochs {
                    for batch in &chunks {
                        println!("-----------------------------------------");
                        println!("On epoch {} out of {}", epoch + 1, epochs);

                        let current_lr = init_learning_rate
                            - (init_learning_rate - final_learning_rate) * (epoch + 1) as f64
                                / epochs as f64;
                        let current_batch_size = batch.len();

                        // Clear vectors
                        all_states.clear();
                        all_output.clear();
                        all_actual.clear();

                        let model_ref = &model;

                        let results: Vec<_> = batch
                            .par_iter()
                            .map(|data| {
                                let (hidden_states, output) =
                                    model_ref.train_forward(&data.fourier);
                                (hidden_states, output, data.class.clone())
                            })
                            .collect();

                        for (states, output, actual) in results {
                            all_states.push(states);
                            all_actual.push(actual);
                            all_output.push(output);
                        }

                        println!(
                            "Progress: {}%",
                            100. * (progress as f64) / (training_set.len() as f64 * epochs as f64)
                        );
                        progress += current_batch_size;

                        model.gitgud_batched(&current_lr, &all_output, &all_actual, &all_states);
                    }
                    println!("Epoch {}/{} done!", epoch + 1, epochs);
                    println!("-----------------------------------------");
                }
            }

            let mut validation_losses: Vec<f64> = vec![];

            for data in &validation_set {
                let fourier = &data.fourier;
                let output: Vec<f64> = model.forward(fourier);
                validation_losses.push(loss(&output, &data.class));
            }

            let validation_loss =
                validation_losses.iter().sum::<f64>() / (validation_losses.len() as f64);

            if validation_loss < best_val_loss {
                best_val_loss = validation_loss;
                best_lr = init_learning_rate;
                best_final_lr = final_learning_rate;
                best_mem_lane = memory_lane_size;
                best_batch_size = batch_size;
            };
            // Print some information
            println!("-----------------------------------------");
            println!("Progress: {}%", 100. * (i as f64) / (trials as f64));
            println!("Validation Loss: {}", validation_loss);
            println!(
                "Best initial lr: {}, best final lr: {}, best memory lane size: {}, best batch size: {}",
                best_lr, best_final_lr, best_mem_lane, best_batch_size
            );
            println!("Best val loss: {}", best_val_loss);
            println!("-----------------------------------------");
        }
        println!("Best validation loss: {}", best_val_loss);
    }
    if run {
        // Read WAV file and apply framing
        let mut audio = hound::WavReader::open(AUDIO_DIR).unwrap();
        let frames = split_audio(unhound(&mut audio), FRAME_SIZE, HOP_SIZE);
        // Apply Hann windowing algorithm on the windows
        let hanned = hann(&frames);
        // Apply Fourier transform on the "Hanned" windows
        let mut fouriered: Vec<Vec<f64>> = vec![];
        for frame in hanned {
            fouriered.push(normVecComplex(&ft(frame)));
        }

        // Read weights
        let model = read_model(MODEL_FILE_PATH);
        let output: Vec<f64> = model.forward(&fouriered);
        println!("Result: {:?}", output)
    }
    if plot {
        let _ = plot_losses(
            LOSS_FILE_PATH,
            VAL_LOSS_FILE_PATH,
            LR_FILE_PATH,
            GRAPH_FILE_PATH,
        );
    }
    if accuracy {
        let model = read_model(MODEL_FILE_PATH);
        let data = read_data(&DATA);
        println!("Accuracy: {:?}", calc_accuracy(&model, &data))
    }
}
