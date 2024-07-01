use crate::layer::Layer;
use std::vec;
pub struct NeuralNetwork {
    pub loss: String,
    pub optimizer: String,
    pub metrics: String,
    layers: Vec<Layer>,
}

#[allow(dead_code)]
impl NeuralNetwork {
    pub fn new(
        units: i32,
        input_dim: i32,
        loss: String,
        optimizer: String,
        metrics: String,
        activation: String,
    ) -> NeuralNetwork {
        let input = Layer::new(units, input_dim, activation);
        NeuralNetwork {
            loss,
            optimizer,
            metrics,
            layers: vec![input],
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut output = inputs.clone();
        for i in 0..self.layers.len() {
            output = self.layers[i].forward(output);
        }
        output
    }

    pub fn add(&mut self, units: i32, activation: String) {
        let input_dim = self.layers[self.layers.len() - 1].rows;
        let layer = Layer::new(units, input_dim, activation);
        self.layers.push(layer);
    }

    fn softmax(&self, x: Vec<f64>) -> Vec<f64> {
        let mut sum = 0.0;
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            sum += x[i].exp();
        }
        for i in 0..x.len() {
            result[i] = x[i].exp() / sum;
        }
        result
    }

    pub fn show_details(&self) {
        println!("---------------Neural Network Details------------------");
        println!("Loss: {:?}", self.loss);
        println!("Optimizer: {:?}", self.optimizer);
        println!("Metrics: {:?}\n", self.metrics);
        for i in 0..self.layers.len() {
            println!("Layer: {}", i);
            self.layers[i].show_details();
            println!();
        }
    }
}
