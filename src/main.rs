// Imports...
use crate::fourier::{fft::*, split_audio::*};
use crate::nn::bo_helpers::*;
use crate::nn::helpers::*;
use crate::nn::model::*;
use bincode::config;
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
const BATCH_SIZE: usize = 1;

// Define training parameters
const LEARNING_RATE: f32 = 1e-5; // Initial learning rate
const DEC_CONST: f32 = 0.08; // For short trainings, fairly fast decay; half-life of ~6 epochs
const MODEL_FILE_PATH: &str = "Timmy.bin";
const TRAINING_SPLIT: f32 = 0.8;
const DATA: [&str; NUM_CLASSES] = ["class1.bin", "class2.bin"]; // Data for all classes
const PATIENCE: usize = 5; // How many epochs do you let validation loss be less than a threshold?
// Use only if using ratio to stop training
// const VAL_THRESHOLD: f32 = 0.9; // Validation loss ratio threshold

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
    let mut bo = false;

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
            "bo" => bo = true,
            _ => {}
        }
    }
    if fourier {
        // Definitions
        let hop_size: usize = FRAME_SIZE / 4; // For 75% overlap
        let label: Vec<f32> = vec![1., 0.];

        // Will take all .wav files in this directory
        let path = "./sine-sawtooth/sine";

        // File name of the raw fourier data
        let data_file_name = "class1.bin";

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
            let mut fourier: Matrix = Matrix::new(&0, &0);
            for frame in hanned {
                let mut appendix = normVecComplex(&ft(frame));
                minmax(&mut appendix);
                fourier.append(&appendix);
            }

            // Add the transformed audio to the collection
            data_collection.push(Data { fourier, class });
        }
        // Save labelled data
        let bytes: Vec<u8> = bincode::encode_to_vec(&data_collection, config::standard())
            .expect("Failed to serialize data collection");
        fs::write(data_file_name, bytes).expect("Failed to write");
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
        let mut loss_window = [f32::INFINITY; PATIENCE];
        let mut best_val_loss: f32 = f32::INFINITY;
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

        // Print accuracy
        println!("Accuracy: {}", calc_accuracy(&model, &validation_set));
    }
    if train_continue {
        // Read data from files
        let all_data = read_data(&DATA);

        // Split data into training and validation sets
        let (training_set, validation_set) = split_data(all_data, TRAINING_SPLIT);

        // Read weights, define model
        let mut model = read_model(MODEL_FILE_PATH);

        let mut running = true;
        let mut epoch = 1;
        let mut current_lr = LEARNING_RATE;
        let mut loss_window = [f32::INFINITY; PATIENCE];
        let mut best_val_loss: f32 = f32::INFINITY;

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
        // Print accuracy
        println!("Accuracy: {}", calc_accuracy(&model, &validation_set));
    }
    if hyper {
        // Define ranges
        let init_learning_rate_exponent_range: Vec<f32> = vec![2., 3.];
        let final_learning_rate_exponent_range: Vec<f32> = vec![4., 6.];
        let memory_lane_sizes: Vec<usize> = vec![64, 128, 256, 512, 1024];
        let batch_size_range: Vec<usize> = vec![1, 4, 8];

        // Hyper parameters/initialization
        let trials = memory_lane_sizes.len() * 20; //Number of full training runs
        let mut best_lr: f32 = 1e-3;
        let mut best_mem_lane = 512;
        let mut best_val_loss = f32::INFINITY; //MUAHMUAH INFINITE LOSS
        let mut best_final_lr = 0.;
        let mut best_batch_size = 1;

        // Read data from files
        let mut data_collection: Vec<Data> = read_data(&DATA);

        // Split data into training and validation sets
        let (training_set, validation_set) = split_data(data_collection, TRAINING_SPLIT);

        let mut rng = rand::rng();
        for i in 0..trials {
            // Randomize hyperparameters
            let init_lr_exponent: f32 = -rng.random_range(
                init_learning_rate_exponent_range[0]..init_learning_rate_exponent_range[1],
            ) as f32;
            let init_learning_rate: f32 = f32::powf(10., init_lr_exponent);

            let final_lr_exponent: f32 = -rng.random_range(
                final_learning_rate_exponent_range[0]..final_learning_rate_exponent_range[1],
            ) as f32;
            let final_learning_rate: f32 = f32::powf(10., final_lr_exponent);

            let memory_lane_size = memory_lane_sizes[rng.random_range(0..memory_lane_sizes.len())];

            let batch_size = batch_size_range[rng.random_range(0..batch_size_range.len())];

            // Create model
            let mut model = Model::new(&FRAME_SIZE, &memory_lane_size, &NUM_CLASSES);

            // Initializing dynamic constants for loop
            let mut running = true; // Always leave this! Even if you're testing!
            let mut epoch = 1;
            let mut current_lr = LEARNING_RATE;
            let mut loss_window = [f32::INFINITY; PATIENCE];
            let mut best_accuracy: f32 = f32::INFINITY;

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

            let accuracy = calc_accuracy(&model, &validation_set);

            if accuracy < best_accuracy {
                best_accuracy = accuracy;
                best_lr = init_learning_rate;
                best_final_lr = final_learning_rate;
                best_mem_lane = memory_lane_size;
                best_batch_size = batch_size;
            };
        }
    }
    if run {
        // Read WAV file and apply framing
        let mut audio = hound::WavReader::open(AUDIO_DIR).unwrap();
        let frames = split_audio(unhound(&mut audio), FRAME_SIZE, HOP_SIZE);
        // Apply Hann windowing algorithm on the windows
        let hanned = hann(&frames);
        // Apply Fourier transform on the "Hanned" windows
        let mut fouriered: Matrix = Matrix::new(&0, &0);
        for frame in hanned {
            fouriered.append(&normVecComplex(&ft(frame)));
        }

        // Read weights
        let model = read_model(MODEL_FILE_PATH);
        let output: Vec<f32> = model.forward(&fouriered);
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
    // COMPLETELY LLM-GENERATED!!!!
    if bo {
        // Define hyperparameter search bounds
        let lr_init_range = (-3.0, -2.0);
        let lr_final_range = (-6.0, -4.0);
        let memory_lane_options: Vec<usize> = vec![64, 128, 256, 512, 1024];
        let batch_size_options: Vec<usize> = vec![1, 4, 8];

        let total_trials = 30;
        let initial_random_trials = 5;

        let mut evaluated_x: Vec<Vec<f32>> = Vec::new();
        let mut evaluated_y: Vec<f32> = Vec::new();

        let mut best_val_loss = f32::INFINITY;
        let mut best_params = (1e-3, 1e-5, 512, 4);

        let data_collection: Vec<Data> = read_data(&DATA);
        let (training_set, validation_set) = split_data(data_collection, TRAINING_SPLIT);

        for trial in 0..total_trials {
            let current_params = if trial < initial_random_trials {
                sample_random_hyperparameters(
                    lr_init_range,
                    lr_final_range,
                    &memory_lane_options,
                    &batch_size_options,
                )
            } else {
                suggest_next_hyperparameters(
                    &evaluated_x,
                    &evaluated_y,
                    lr_init_range,
                    lr_final_range,
                    &memory_lane_options,
                    &batch_size_options,
                )
            };

            let (init_lr, final_lr, memory_lane_size, batch_size) = current_params;

            println!(
                "\n--- Trial {}/{} --- [Init LR: {:.1e}, Final LR: {:.1e}, Mem Lane: {}, Batch: {}]",
                trial + 1,
                total_trials,
                init_lr,
                final_lr,
                memory_lane_size,
                batch_size
            );

            let mut model = Model::new(&FRAME_SIZE, &memory_lane_size, &NUM_CLASSES);
            let val_loss = train_and_evaluate_model(
                &mut model,
                &training_set,
                &validation_set,
                init_lr,
                batch_size,
            );

            let x_vector = encode_hyperparameters(init_lr, final_lr, memory_lane_size, batch_size);
            evaluated_x.push(x_vector);
            evaluated_y.push(val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_params = current_params;
                save_model(&model, MODEL_FILE_PATH);
                println!("New best validation loss achieved: {:.4}", val_loss);
            }
        }

        println!("\n[BO Complete] Best Validation Loss: {:.4}", best_val_loss);
        println!(
            "Best Hyperparameters: Init LR: {:.1e}, Final LR: {:.1e}, Mem Lane: {}, Batch Size: {}",
            best_params.0, best_params.1, best_params.2, best_params.3
        );
    }
}
