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

// For multi-threading
use rayon::prelude::*;

mod fourier;
mod nn;

// Constants
// Constant used for fouriering and training
const FRAME_SIZE: usize = 2048;
const HOP_SIZE: usize = FRAME_SIZE / 4;
const LABELS: [[f32; NUM_CLASSES]; NUM_CLASSES] = [[1., 0.], [0., 1.]];

const WAV_PATHS: [&str; NUM_CLASSES] = ["./voices/males/", "./voices/females/"];
const TRAINING_SET_PATH: &str = "train.bin";
const VALIDATION_SET_PATH: &str = "validation.bin";
const TEST_SET_PATH: &str = "test.bin";
const ALL_PATH: &str = "class-1-m.bin";

// Define model parameters
const MEMORY_LANE_SIZE: usize = 1024;
const NUM_CLASSES: usize = 2;
const BATCH_SIZE: usize = 4;

// Define training parameters
const LEARNING_RATE: f32 = 1e-2; // Initial learning rate
const DEC_CONST: f32 = 0.08; // For short trainings, fairly fast decay; half-life of ~6 epochs
const MODEL_FILE_PATH: &str = "Timmy.bin";
const DATA: [&str; NUM_CLASSES] = ["class1.bin", "class2.bin"]; // Data for all classes
const PATIENCE: usize = 5; // How many epochs do you let validation loss be less than a threshold?
const START_EXP: f32 = 3e-1; // val loss threshold where exponentially decaying lr should start
// Use only if using ratio to stop training
// const VAL_THRESHOLD: f32 = 0.9; // Validation loss ratio threshold

// Splits
const SPLIT_DATA: bool = true;
const TRAINING_SPLIT: f32 = 0.7;
const VALIDATION_SPLIT: f32 = 0.2;
const TEST_SPLIT: f32 = 0.1;

// Running parameters
const AUDIO_DIR: &str = "test2.wav";

// Plotting
const GRAPH_FILE_PATH: &str = "graph.svg";
const LOSS_FILE_PATH: &str = "losses";
const VAL_LOSS_FILE_PATH: &str = "validation_losses";
const LR_FILE_PATH: &str = "lr";

fn main() {
    // Check that splits add up to <= 100%
    const { assert!(VALIDATION_SPLIT + TRAINING_SPLIT + TEST_SPLIT <= 1.) };

    // For handling arguments
    let mut fourier: bool = false;
    let mut training: bool = false;
    let mut run: bool = false;
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
            "train-continue" => (training, train_continue) = (true, true),
            "plot" => plot = true,
            "accuracy" => accuracy = true,
            "bo" => bo = true,
            _ => {}
        }
    }
    if fourier {
        let mut all_data: Vec<Data> = vec![];
        for x in 0..NUM_CLASSES {
            // Read WAV files
            let mut audios = read_wavs(WAV_PATHS[x]);

            // Parallel processing
            let mut data_collection: Vec<Data> = audios
                .par_iter_mut()
                .map(|audio| {
                    // Split audio
                    let frames = split_audio(unhound(audio), FRAME_SIZE, HOP_SIZE);

                    // Apply Hann windowing algorithm on the windows
                    let hanned = hann(&frames);

                    // Apply Fourier transform on the "Hanned" windows
                    let mut fourier: Matrix = Matrix::new(&0, &0);
                    for frame in hanned {
                        let mut appendix = norm_vec_complex_log(&ft(frame));
                        minmax(&mut appendix);
                        fourier.append(&appendix);
                    }

                    // Add the transformed audio to the collection
                    Data {
                        fourier,
                        class: LABELS[x].to_vec(),
                    }
                })
                .collect::<Vec<Data>>();
            all_data.append(&mut data_collection);
        }
        if SPLIT_DATA {
            let (training_set, val_set, test_set) = split_data(all_data);

            // Encode
            let bytes_training: Vec<u8> = bincode::encode_to_vec(&training_set, config::standard())
                .expect("Failed to serialize training set");

            let bytes_val: Vec<u8> = bincode::encode_to_vec(&val_set, config::standard())
                .expect("Failed to serialize validation set");

            let bytes_test: Vec<u8> = bincode::encode_to_vec(&test_set, config::standard())
                .expect("Failed to serialize test set");

            // Write
            fs::write(TRAINING_SET_PATH, bytes_training).expect("Failed to write training set");
            println!("Training set written successfully...");

            fs::write(VALIDATION_SET_PATH, bytes_val).expect("Failed to write validation set");
            println!("Validation set written successfully...");

            fs::write(TEST_SET_PATH, bytes_test).expect("Failed to write test set");
            println!("Test set written successfully...");
        } else {
            // Save all of the data
            let bytes: Vec<u8> = bincode::encode_to_vec(&all_data, config::standard())
                .expect("Failed to serialize data collection");
            fs::write(ALL_PATH, bytes).expect("Failed to write");
            println!("Data saved successfully...")
        }
    }
    if training {
        let (training_set, validation_set, test_set) = if SPLIT_DATA {
            (
                read_data(TRAINING_SET_PATH),
                read_data(VALIDATION_SET_PATH),
                read_data(TEST_SET_PATH),
            )
        } else {
            // Read data from files
            let data = read_all_data(&DATA);
            // Split data into training and validation sets
            split_data(data)
        };

        // Create model
        let mut model = if train_continue {
            // Read and load existing model
            read_model(MODEL_FILE_PATH)
        } else {
            // Create brand-new model
            Model::new(&FRAME_SIZE, &MEMORY_LANE_SIZE, &NUM_CLASSES)
        };

        // Initializing dynamic constants for loop
        let mut running = true; // Always leave this! Even if you're testing!
        let mut epoch = 1;
        let mut current_lr = LEARNING_RATE;
        let mut loss_window = [f32::INFINITY; PATIENCE];
        let mut best_val_loss: f32 = f32::INFINITY;
        while running {
            // For testing:
            // for _ in 0..20 {
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

            println!(
                "Validation loss: {}, ∆: {}",
                validation_loss,
                validation_loss - loss_window[PATIENCE - 2]
            );

            // Modify learning rate
            if validation_loss < START_EXP {
                current_lr = calc_new_lr_exp(LEARNING_RATE, DEC_CONST, epoch);
                epoch += 1;
            }

            // To kill or not to kill...
            // If using ratios:
            // running = !kill_no_kill_ratios(&loss_window, VAL_THRESHOLD);
        }
        println!(
            "Training stopped. Reason: {}",
            if running {
                "Inexplicable... Perhaps some bits got changed cosmically..."
            } else {
                "Validation loss difference did not meet expectations..."
            }
        );

        // Print accuracy
        println!("Accuracy: {}", calc_accuracy(&model, &test_set));
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
            fouriered.append(&norm_vec_complex_log(&ft(frame)));
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
        let data = read_data(TEST_SET_PATH);
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

        let (training_set, validation_set, _) = if SPLIT_DATA {
            (
                read_data(TRAINING_SET_PATH),
                read_data(VALIDATION_SET_PATH),
                read_data(TEST_SET_PATH),
            )
        } else {
            // Read data from files
            let data = read_all_data(&DATA);
            // Split data into training and validation sets
            split_data(data)
        };

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
