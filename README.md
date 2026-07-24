# Matura

A little STFT and LSTM in Rust :3

Usage:
Make sure you have Rust (and Cargo) installed. Then, cd into this directory. There are three ways to run it:

1. "fourier": This goes to the path specified in the code (in main.rs) and combines all .wav files there into one json in a format that's readable to the model (with label and already fourier-transformed, all things can be adjusted in main.rs).
2. "train": This reads all files specified in main.rs in the train section (in my case "class1.json" and "class2.json"...) and trains the model on those. You can adjust parameters in main.rs. The weights and model parameters get saved in the end.
3. "run": This searches for a "Timmy.json" wherein lie all the data for the model. It reads a specified wav file and outputs a vector with the prediction.

The syntax is:
cargo run --release -- fourier/train/run
