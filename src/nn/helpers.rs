use plotters::prelude::*;
use std::f64;
use std::fs::OpenOptions;
use std::io::Write;

// This is a file full of helper stuff because main.rs is getting too cluttered at this point
use crate::*;

pub fn calculate_validation_loss(
    model: &Model,
    validation_set: &[Data],
    file_path: Option<&String>,
) -> f64 {
    let mut validation_losses: Vec<f64> = vec![];

    for data in validation_set {
        let fourier = &data.fourier;
        let output: Vec<f64> = model.forward(fourier);
        validation_losses.push(loss(&output, &data.class));
    }
    let validation_loss: f64 =
        validation_losses.iter().sum::<f64>() / (validation_losses.len() as f64);

    if let Some(x) = file_path {
        let mut file = OpenOptions::new()
            .create(true) // creates if there is none
            .append(true) // Appends new writes to the end of file
            .open(x)
            .expect("Failed to open file");
        writeln!(file, "{:?}", validation_loss).expect("Ran into issue writing val loss");
    }
    validation_loss
}

pub fn train(
    batch_size: usize,
    learning_rate: f64,
    training_set: &[Data],
    model: &mut Model,
    loss_file_path: Option<&String>,
) {
    if batch_size == 1 {
        // Train the model on training set
        for data in training_set {
            let (fourier, class) = (&data.fourier, &data.class);
            let (hidden_states, output) = model.train_forward(fourier);
            let loss = model.gitgud(&learning_rate, &output, class, &hidden_states);

            if let Some(x) = loss_file_path {
                let mut file = OpenOptions::new()
                    .create(true) // creates if there is none
                    .append(true) // Appends new writes to the end of file
                    .open(x)
                    .expect("Failed to open file");
                writeln!(file, "{:?}", loss).expect("Ran into issue writing loss");
            } else {
                println!("Loss: {}", loss)
            }
        }
    } else {
        let chunks = training_set.chunks(batch_size).collect::<Vec<&[Data]>>();

        let mut all_states = Vec::with_capacity(batch_size);
        let mut all_output = Vec::with_capacity(batch_size);
        let mut all_actual = Vec::with_capacity(batch_size);
        for batch in &chunks {
            // Clear vectors
            all_states.clear();
            all_output.clear();
            all_actual.clear();

            let model_ref = &model;

            let results: Vec<_> = batch
                .par_iter()
                .map(|data| {
                    let (hidden_states, output) = model_ref.train_forward(&data.fourier);
                    (hidden_states, output, data.class.clone())
                })
                .collect();

            for (states, output, actual) in results {
                all_states.push(states);
                all_actual.push(actual);
                all_output.push(output.to_vec());
            }

            let loss = model.gitgud_batched(&learning_rate, &all_output, &all_actual, &all_states);

            if let Some(x) = loss_file_path {
                let mut file = OpenOptions::new()
                    .create(true) // creates if there is none
                    .append(true) // Appends new writes to the end of file
                    .open(x)
                    .expect("Failed to open file");
                writeln!(file, "{:?}", loss).expect("Ran into issue writing avg loss");
            } else {
                println!("Average loss: {}", loss)
            }
        }
    }
}

pub fn calc_new_lr_exp(initial_learning_rate: f64, dec_const: f64, epoch: i64) -> f64 {
    initial_learning_rate * (-(epoch as f64) * dec_const).exp()
}

fn read_floats(path: &str) -> Vec<f64> {
    fs::read_to_string(path)
        .expect("Failed to open file")
        .lines()
        .map(|line| line.trim().parse::<f64>().expect("Failed to parse number"))
        .collect()
}

// LLM-generated
pub fn plot_losses(
    losses_file_path: &str,
    val_losses_file_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Output file
    let root = BitMapBackend::new(output_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Read files
    let train_losses = read_floats(losses_file_path);
    let val_losses = read_floats(val_losses_file_path);

    let num_epochs = val_losses.len();
    let num_steps = train_losses.len();

    // Find min and max values for clean scaling
    let all_values: Vec<f64> = train_losses
        .iter()
        .chain(val_losses.iter())
        .cloned()
        .collect();
    let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training and Validation Loss", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..num_epochs as f64, min_val..max_val)?;

    chart
        .configure_mesh()
        .x_labels(val_losses.len())
        .x_desc("Epochs")
        .y_desc("Loss")
        .draw()?;

    // Plot training loss
    chart.draw_series(LineSeries::new(
        train_losses.iter().enumerate().map(|(i, &y)| {
            let epoch_x = (i as f64) * ((num_epochs as f64) / (num_steps as f64));
            (epoch_x, y)
        }),
        &BLUE.mix(0.6), // Slightly transparent so it looks softer in the background
    ))?;
    // Plot val loss
    chart.draw_series(LineSeries::new(
        val_losses.iter().enumerate().map(|(j, &y)| (j as f64, y)),
        &RED,
    ))?;

    root.present()?;
    println!("Graph saved successfully as {}!", output_path);
    Ok(())
}

pub fn save_model(model: &Model, path: &str) {
    // Convert model into json string
    let json =
        serde_json::to_string(&model).expect("Error making the model a json string. Fun times.");

    let mut file = File::create(path).unwrap();
    file.write_all(json.as_bytes())
        .expect(&format!("Failed to write model to file {}", path));
    println!("Model saved successfully...")
}

// For testing and graphing things
pub fn write_to_file(path: &str, content: &str) {
    let mut file = OpenOptions::new()
        .create(true) // creates if there is none
        .append(true) // Appends new writes to the end of file
        .open(path)
        .expect("Failed to open file");
    writeln!(file, "{}", content).expect("Ran into issue writing loss");
}
