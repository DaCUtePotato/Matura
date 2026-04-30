// This file contains all the prerequisites and the model. It alone doesn't do anything
// and the model has to be created in main.rs and run/trained there too. This is just
// for all the framework and functions and stuff
use rand::RngExt;
use rand::distr::Uniform;
use std::f64;

pub fn sigmoid(aaaaa: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    for i in aaaaa {
        result.push(1. / (1. + std::f64::consts::E.powf(-i)));
    }
    result
}

pub fn tanh(bbbbb: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    let e: f64 = std::f64::consts::E;
    for i in bbbbb {
        // major computation save by only computing e^x and e^-x once :oo!!!!!!
        let ei: f64 = e.powf(*i);
        let mei: f64 = e.powf(-i);
        result.push((ei - mei) / (ei + mei));
    }
    result
}

pub fn linear(input: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    for (i, value) in weights.iter().enumerate() {
        let mut sum: f64 = 0.;
        for (j, valuevalue) in value.iter().enumerate() {
            sum += valuevalue * input[j];
        }
        sum += biases[i];
        result.push(sum);
    }
    result
}

pub fn multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
}

pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
}

pub fn concatify(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = a.to_vec();
    result.extend_from_slice(b);
    result
}

pub fn softmax(vector: &[f64]) -> Vec<f64> {
    // Huuuuge computational save by only doing e^valueofvector once
    let eed: Vec<f64> = vector
        .iter()
        .map(|s| std::f64::consts::E.powf(*s))
        .collect();
    let sum: f64 = eed.iter().sum();
    eed.iter().map(|s| s / sum).collect()
}

pub fn xavier_value(num_inputs: &i64, num_outputs: &i64) -> f64 {
    let mut rng = rand::rng();
    let range: f64 = f64::sqrt(6. / (*num_inputs as f64 + *num_outputs as f64));
    rng.sample(Uniform::new(-range, range).unwrap())
}

pub fn loss(output: &[f64], actual: &[f64]) -> f64 {
    let mut sum: f64 = 0.;
    for (i, _) in output.iter().enumerate().filter(|(_, s)| **s > 0.5) {
        sum += -actual[i] * output[i].ln();
    }
    sum
}

// extremely simple, barebones single-layered NN because we only have single-layered ones in lstm :3
pub struct NN {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl NN {
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        linear(input, &self.weights, &self.biases)
    }
    pub fn new(num_inputs: &i64, num_outputs: &i64) -> Self {
        let mut x = NN {
            weights: vec![],
            biases: vec![],
        };
        for i in 0..(*num_outputs as usize) {
            x.weights.push(vec![]);
            for _ in 0..*num_inputs {
                x.weights[i].push(xavier_value(num_inputs, num_outputs));
            }
            x.biases.push(0.)
        }
        x
    }
}

pub struct LSTM {
    // Forget gate
    s1: NN,
    // Input gate (new memory!!)
    s2: NN,
    // Cell/Candidate gate
    t: NN,
    // Output gate
    s3: NN,
}

impl LSTM {
    pub fn forward_pass(
        &self,
        memory_lane: &[f64],
        main_lane: &[f64],
        input: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let concatified = concatify(main_lane, input);
        let memory_lane = add(
            &multiply(&sigmoid(&self.s1.forward(&concatified)), memory_lane),
            &multiply(
                &sigmoid(&self.s2.forward(&concatified)),
                &tanh(&self.t.forward(&concatified)),
            ),
        );
        let output = multiply(
            &tanh(&memory_lane),
            &sigmoid(&self.s3.forward(&concatified)),
        );
        (output, memory_lane)
    }
    pub fn new(num_inputs: &i64, num_memory_lane: &i64) -> Self {
        LSTM {
            s1: NN::new(&(num_inputs + num_memory_lane), num_memory_lane),
            s2: NN::new(&(num_inputs + num_memory_lane), num_memory_lane),
            t: NN::new(&(num_inputs + num_memory_lane), num_memory_lane),
            s3: NN::new(&(num_inputs + num_memory_lane), num_memory_lane),
        }
    }
    pub fn training_forward_pass(
        &self,
        memory_lane: &[f64],
        main_lane: &[f64],
        input: &[f64],
    ) -> LSTMHiddenState {
        let concatified = concatify(main_lane, input);
        let s1 = sigmoid(&self.s1.forward(&concatified));
        let s2 = sigmoid(&self.s2.forward(&concatified));
        let s3 = sigmoid(&self.s3.forward(&concatified));
        let t = tanh(&self.t.forward(&concatified));
        let memory_lane = add(&multiply(&s1, memory_lane), &multiply(&s2, &t));
        let output = multiply(&tanh(&memory_lane), &s3);
        LSTMHiddenState {
            input: input.to_vec(),
            memory_lane,
            main_lane: output,
            s1,
            s2,
            t,
            s3,
        }
    }

    // AAAAAAAAH BPTT IS HELL WAAAAAAH HELPPPP
    pub fn gitgud(&self, a_t: &[f64], saved_hidden_states: &[LSTMHiddenState]) {
        for i in saved_hidden_states {}
    }
}

pub struct ClassificationHead {
    n: NN,
}

impl ClassificationHead {
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        softmax(&self.n.forward(input))
    }
    pub fn new(num_memory_lane: &i64, num_classes: &i64) -> Self {
        ClassificationHead {
            n: NN::new(num_memory_lane, num_classes),
        }
    }
    // Backprop for the ClassificationHead
    pub fn gitgud(
        &mut self,
        learning_rate: &f64,
        actual: &[f64],
        output: &[f64],
        input: &[f64],
    ) -> Vec<f64> {
        let mut new_loss: Vec<f64> = vec![];
        for i in 0..input.len() {
            let mut sum = 0.;
            for j in 0..self.n.weights.len() {
                sum += (output[j] - actual[j]) * self.n.weights[j][i];
            }
            new_loss.push(sum);
        }
        for i in 0..self.n.weights.len() {
            for j in 0..self.n.weights[i].len() {
                // Review and understand exactly what the hell is going on with this goofy aah
                // gradient
                self.n.weights[i][j] -= learning_rate * (output[i] - actual[i]) * input[j];
            }
            self.n.biases[i] -= learning_rate * (output[i] - actual[i]);
        }
        new_loss
    }
}

// Yet another struct because of the stupid hidden states that have to be saved
// in the forward pass during training aaaaaaaaaaaaaaaaaaaaaa helpppp
struct LSTMHiddenState {
    input: Vec<f64>,
    memory_lane: Vec<f64>,
    main_lane: Vec<f64>,
    s1: Vec<f64>, // The output of the Sigmoid 1
    s2: Vec<f64>, // Same here but for Sigmoid 2
    t: Vec<f64>,  // lo and behold a tanh!
    s3: Vec<f64>, // You'll never guess...
}

pub struct Model {
    num_memory_lane: usize,
    classification_head: ClassificationHead,
    lstm: LSTM,
}

impl Model {
    pub fn new(frame_size: &i64, num_memory_lane: &i64, num_classes: &i64) -> Self {
        Model {
            num_memory_lane: *num_memory_lane as usize,
            classification_head: ClassificationHead::new(num_memory_lane, num_classes),
            lstm: LSTM::new(frame_size, num_memory_lane),
        }
    }
    // This is a normal run
    pub fn forward(&self, frames: &[Vec<f64>]) -> Vec<f64> {
        let mut memory_lane: Vec<f64> = vec![0.; self.num_memory_lane];
        let mut main_lane: Vec<f64> = vec![0.; self.num_memory_lane];
        for frame in frames {
            (main_lane, memory_lane) =
                LSTM::forward_pass(&self.lstm, &memory_lane, &main_lane, frame);
        }
        ClassificationHead::forward(&self.classification_head, &main_lane)
    }
    // This is where u pull all the stuff u need to backprop
    pub fn train_forward(&self, frames: &[Vec<f64>]) -> (Vec<LSTMHiddenState>, Vec<f64>) {
        let memory_lane: Vec<f64> = vec![0.; self.num_memory_lane];
        let main_lane: Vec<f64> = vec![0.; self.num_memory_lane];
        let mut hidden_states: Vec<LSTMHiddenState> = vec![];
        for frame in frames {
            hidden_states.push(LSTM::training_forward_pass(
                &self.lstm,
                &memory_lane,
                &main_lane,
                frame,
            ));
        }
        let output = ClassificationHead::forward(&self.classification_head, &main_lane);
        (hidden_states, output)
    }
    // This is where u call the backprops
    pub fn gitgud(
        &mut self,
        learning_rate: &f64,
        output: &[f64],
        actual: &[f64],
        saved_hidden_states: &[LSTMHiddenState],
    ) {
        let classification_head_gradient = ClassificationHead::gitgud(
            &mut self.classification_head,
            learning_rate,
            actual,
            output,
            &saved_hidden_states[saved_hidden_states.len()].main_lane,
        );
        LSTM::gitgud(&classification_head_gradient, &saved_hidden_states);
    }
}
