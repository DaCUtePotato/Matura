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
    let validation_loss = validation_set
        .par_iter()
        .map(|data| loss(&model.forward(&data.fourier), &data.class))
        .sum::<f64>()
        / (validation_set.len() as f64);

    if let Some(x) = file_path {
        let mut file = OpenOptions::new()
            .create(true) // creates if there is none
            .append(true) // Appends new writes to the end of file
            .open(x)
            .expect("Failed to open file");
        writeln!(file, "{}", validation_loss).expect("Ran into issue writing val loss");
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
    lr_file_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Use SVGBackend for infinitely scalable vector graphics (no pixelation!)
    let root = SVGBackend::new(output_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Read files
    let train_losses = read_floats(losses_file_path);
    let val_losses = read_floats(val_losses_file_path);
    let lr_values = read_floats(lr_file_path);

    // Ensure num_epochs accurately reflects the full duration
    let num_epochs = val_losses.len().max(lr_values.len());
    let num_steps = train_losses.len();

    // Find min and max values for Losses (Primary Y-axis)
    let loss_values: Vec<f64> = train_losses
        .iter()
        .chain(val_losses.iter())
        .cloned()
        .collect();
    let min_loss = loss_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_loss = loss_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Find min and max values for Learning Rate (Secondary Y-axis)
    let min_lr = lr_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_lr = lr_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Set up chart with a right Y-label area for the dual axis
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Training Loss, Validation Loss & Learning Rate",
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .right_y_label_area_size(50) // Space for the right-hand Y-axis
        .build_cartesian_2d(0f64..num_epochs as f64, min_loss..max_loss)?
        .set_secondary_coord(0f64..num_epochs as f64, min_lr..max_lr);

    // Configure primary mesh (Losses on the left)
    chart
        .configure_mesh()
        .x_labels(num_epochs.min(20))
        .x_desc("Epochs")
        .y_desc("Loss")
        .label_style(("sans-serif", 12))
        .draw()?;

    // Configure secondary mesh (Learning Rate on the right)
    chart
        .configure_secondary_axes()
        .y_desc("Learning Rate")
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .label_style(("sans-serif", 12))
        .draw()?;

    // Plot training loss (Primary axis)
    chart
        .draw_series(LineSeries::new(
            train_losses.iter().enumerate().map(|(i, &y)| {
                let epoch_x = (i as f64) * ((num_epochs as f64) / (num_steps as f64));
                (epoch_x, y)
            }),
            &BLUE.mix(0.6),
        ))?
        .label("Train Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.6)));

    // Plot validation loss (Primary axis)
    chart
        .draw_series(LineSeries::new(
            val_losses.iter().enumerate().map(|(j, &y)| (j as f64, y)),
            &RED,
        ))?
        .label("Val Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Plot learning rate (Secondary axis on the right)
    chart
        .draw_secondary_series(LineSeries::new(
            lr_values.iter().enumerate().map(|(j, &y)| (j as f64, y)),
            &GREEN,
        ))?
        .label("Learning Rate")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // Draw the legend box
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;

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

pub fn split_data(mut data_collection: Vec<Data>, training_split: f64) -> (Vec<Data>, Vec<Data>) {
    let mut rng = rng();
    data_collection.shuffle(&mut rng);
    let split_index = (data_collection.len() as f64 * (1. - training_split)).round() as usize;
    let training_set = data_collection.split_off(split_index);
    (training_set, data_collection)
}

pub fn read_data(paths: &[&str; NUM_CLASSES]) -> Vec<Data> {
    let mut data_collection: Vec<Data> = vec![];
    for content in paths {
        let stuff = fs::read_to_string(content).expect("Failed to read data");
        let data_collection_temp: Vec<Data> =
            serde_json::from_str(&stuff).expect("Failed to deserialize content");
        data_collection.extend(data_collection_temp);
    }
    data_collection
}

pub fn kill_no_kill_ratios(validation_losses: &[f64; PATIENCE], threshold: f64) -> bool {
    validation_losses
        .windows(2)
        .map(|w| w[0] / w[1])
        .sum::<f64>()
        / ((PATIENCE - 1) as f64)
        < threshold
}

pub fn calc_accuracy(model: &Model, validation_set: &[Data]) -> f64 {
    let sum = validation_set
        .par_iter()
        // Run the model on every piece of data in the dataset and filter
        // only the ones where it was correct
        .filter(|data| {
            let output = model.forward(&data.fourier);
            // find max index
            let max_index = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1)) // Finds maximum value
                .map(|(i, _)| i) //
                .unwrap();

            // Find index where target is 1.
            let target_index = data.class.iter().position(|&x| x == 1.).unwrap();

            // Compare the two indices
            max_index == target_index
        })
        // Count how many times the model was right
        .count() as f64;

    // Average
    sum / (validation_set.len() as f64)
}

pub fn read_model(model_path: &str) -> Model {
    let content = fs::read_to_string(model_path).expect("Reading model didn't work");
    let model: Model =
        serde_json::from_str(&content).expect("Ran into a problem trying to deserialize model");
    model
}
