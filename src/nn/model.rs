// This file contains all the prerequisites and the model. It alone doesn't do anything
// and the model has to be created in main.rs and run/trained there too. This is just
// for all the framework and functions and stuff
use rand::RngExt;
use rand::distr::Uniform;
use rand::rngs::ThreadRng;
use std::f64;

pub fn sigmoid(aaaaa: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    for i in aaaaa {
        result.push(1. / (1. + (-i).exp()));
    }
    result
}

pub fn tanh(bbbbb: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    for i in bbbbb {
        // major computation save by only computing e^x and e^-x once :oo!!!!!!
        let ei: f64 = (*i).exp();
        let mei: f64 = (-i).exp();
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
    let max = vector.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let eed: Vec<f64> = vector.iter().map(|s| (*s - max).exp()).collect();
    let sum: f64 = eed.iter().sum();
    eed.iter().map(|s| s / sum).collect()
}

pub fn xavier_value(rng: &mut ThreadRng, num_inputs: &i64, num_outputs: &i64) -> f64 {
    let range: f64 = f64::sqrt(6. / (*num_inputs as f64 + *num_outputs as f64));
    rng.sample(Uniform::new(-range, range).unwrap())
}

pub fn loss(output: &[f64], actual: &[f64]) -> f64 {
    let mut sum: f64 = 0.;
    for (i, _) in output.iter().enumerate() {
        sum += -actual[i] * (output[i] + 69e-9).ln();
    }
    sum
}

pub fn transmatmult(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![0.0; matrix[0].len()];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            result[j] += value * vector[i];
        }
    }
    result
}

pub fn update_weights(weights: &mut [Vec<f64>], gradient: &[Vec<f64>], learning_rate: &f64) {
    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            weights[i][j] -= clip(&gradient[i][j]) * learning_rate
        }
    }
}

pub fn outer_product(vector1: &[f64], vector2: &[f64]) -> Vec<Vec<f64>> {
    vector1
        .iter()
        .map(|&s| vector2.iter().map(|&g| g * s).collect())
        .collect()
}

pub fn madd(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matrix1
        .iter()
        .zip(matrix2.iter())
        .map(|(val1, val2)| add(&val1, &val2))
        .collect()
}

// Not yet used but may come in handy if the model starts producing NaN (since
// gradients can explode)
pub fn clip(gradient: &f64) -> f64 {
    gradient.clamp(-1., 1.)
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
        let mut rng = rand::rng();
        for i in 0..(*num_outputs as usize) {
            x.weights.push(vec![]);
            for _ in 0..*num_inputs {
                x.weights[i].push(xavier_value(&mut rng, num_inputs, num_outputs));
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
        let t = self.t.forward(&concatified);
        let memory_lane = add(&multiply(&s1, memory_lane), &multiply(&s2, &tanh(&t)));
        let output = multiply(&tanh(&memory_lane), &s3);
        LSTMHiddenState {
            concatified,
            memory_lane: memory_lane.clone(),
            main_lane: output.clone(),
            s1,
            s2,
            t,
            s3,
        }
    }

    // AAAAAAAAH BPTT IS HELL WAAAAAAH HELPPPP
    pub fn gitgud(
        &mut self,
        a_t: &[f64],
        saved_hidden_states: &[LSTMHiddenState],
        learning_rate: &f64,
    ) {
        let concatified_size = saved_hidden_states[0].concatified.len();
        let gate_size = saved_hidden_states[0].main_lane.len();

        let mut sum_s1: Vec<Vec<f64>> = vec![vec![0.; concatified_size]; gate_size];
        let mut sum_s2: Vec<Vec<f64>> = sum_s1.clone();
        let mut sum_s3: Vec<Vec<f64>> = sum_s1.clone();
        let mut sum_t: Vec<Vec<f64>> = sum_s1.clone();

        let mut bias_grad_s1 = vec![0.; gate_size];
        let mut bias_grad_s2 = bias_grad_s1.clone();
        let mut bias_grad_s3 = bias_grad_s1.clone();
        let mut bias_grad_t = bias_grad_s1.clone();

        let mut dc_next = vec![0.; gate_size];
        let mut a_t = a_t.to_vec();

        for (i, state) in saved_hidden_states.iter().rev().enumerate() {
            let c_prev = if i + 1 < saved_hidden_states.len() {
                saved_hidden_states[saved_hidden_states.len() - i - 2]
                    .memory_lane
                    .clone()
            } else {
                vec![0.; state.memory_lane.len()]
            };
            let c_now = multiply(
                &a_t,
                &multiply(
                    &state.s3,
                    &tanh(&state.memory_lane)
                        .iter()
                        .map(|s| 1. - s * s)
                        .collect::<Vec<f64>>(),
                ),
            );
            let c = add(&dc_next, &c_now);

            let dc_bob: Vec<f64> = multiply(&c, &state.s2);
            let c_bob: Vec<f64> = tanh(&state.t);
            let t: Vec<f64> = multiply(
                &dc_bob,
                &(tanh(&state.t)
                    .iter()
                    .map(|s| 1. - s * s)
                    .collect::<Vec<f64>>()),
            );
            let s1 = multiply(
                &c,
                &multiply(
                    &c_prev,
                    &multiply(
                        &state.s1,
                        &state.s1.iter().map(|s| 1. - s).collect::<Vec<f64>>(),
                    ),
                ),
            );
            let s2 = multiply(
                &c,
                &multiply(
                    &c_bob,
                    &multiply(
                        &state.s2,
                        &state.s2.iter().map(|s| 1. - s).collect::<Vec<f64>>(),
                    ),
                ),
            );
            let s3 = multiply(
                &multiply(
                    &a_t,
                    &multiply(
                        &state.s3,
                        &state.s3.iter().map(|s| 1. - s).collect::<Vec<f64>>(),
                    ),
                ),
                &tanh(&state.memory_lane),
            );
            sum_s1 = madd(&outer_product(&s1, &state.concatified), &sum_s1);
            sum_s2 = madd(&outer_product(&s2, &state.concatified), &sum_s2);
            sum_s3 = madd(&outer_product(&s3, &state.concatified), &sum_s3);
            sum_t = madd(&outer_product(&t, &state.concatified), &sum_t);

            bias_grad_s1 = add(&bias_grad_s1, &s1);
            bias_grad_s2 = add(&bias_grad_s2, &s2);
            bias_grad_s3 = add(&bias_grad_s3, &s3);
            bias_grad_t = add(&bias_grad_t, &t);

            dc_next = multiply(&c, &state.s1)
                .iter()
                .map(|s| s.clamp(-0.5, 0.5))
                .collect();

            a_t = add(
                &add(
                    &transmatmult(&self.s1.weights, &s1),
                    &transmatmult(&self.s2.weights, &s2),
                ),
                &add(
                    &transmatmult(&self.s3.weights, &s3),
                    &transmatmult(&self.t.weights, &t),
                ),
            )[..state.main_lane.len()]
                .to_vec()
                .iter()
                .map(|s| s.clamp(-0.5, 0.5))
                .collect();
        }
        update_weights(&mut self.s1.weights, &sum_s1, learning_rate);
        update_weights(&mut self.s2.weights, &sum_s2, learning_rate);
        update_weights(&mut self.s3.weights, &sum_s3, learning_rate);
        update_weights(&mut self.t.weights, &sum_t, learning_rate);
        self.s1.biases = add(
            &self.s1.biases,
            &bias_grad_s1
                .iter()
                .map(|s| -clip(s) * learning_rate)
                .collect::<Vec<f64>>(),
        );
        self.s2.biases = add(
            &self.s2.biases,
            &bias_grad_s2
                .iter()
                .map(|s| -clip(s) * learning_rate)
                .collect::<Vec<f64>>(),
        );
        self.s3.biases = add(
            &self.s3.biases,
            &bias_grad_s3
                .iter()
                .map(|s| -clip(s) * learning_rate)
                .collect::<Vec<f64>>(),
        );
        self.t.biases = add(
            &self.t.biases,
            &bias_grad_t
                .iter()
                .map(|s| -clip(s) * learning_rate)
                .collect::<Vec<f64>>(),
        );
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
pub struct LSTMHiddenState {
    concatified: Vec<f64>,
    memory_lane: Vec<f64>,
    main_lane: Vec<f64>,
    s1: Vec<f64>, // The output of the Sigmoid 1
    s2: Vec<f64>, // Same here but for Sigmoid 2
    t: Vec<f64>, // DIFFERENT!! Here, the input is saved, not the output (the vector before tanh) :o
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
        let mut memory_lane: Vec<f64> = vec![0.; self.num_memory_lane];
        let mut main_lane: Vec<f64> = vec![0.; self.num_memory_lane];
        let mut hidden_states: Vec<LSTMHiddenState> = vec![];
        for frame in frames {
            let state = LSTM::training_forward_pass(&self.lstm, &memory_lane, &main_lane, frame);
            memory_lane = state.memory_lane.clone();
            main_lane = state.main_lane.clone();
            hidden_states.push(state);
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
        let first_loss = ClassificationHead::gitgud(
            &mut self.classification_head,
            learning_rate,
            actual,
            output,
            &saved_hidden_states[saved_hidden_states.len() - 1].main_lane,
        );
        LSTM::gitgud(
            &mut self.lstm,
            &first_loss,
            saved_hidden_states,
            learning_rate,
        );
    }
}
