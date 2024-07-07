use crate::gradient_descend::{adjust_weights, calculate_deltas};
use crate::layer::Layer;
use crate::numrs::math::normalize_ouput;
pub struct NeuralNetwork {
    pub loss: String,
    pub optimizer: String,
    pub layers: Vec<Layer>,
}

#[allow(dead_code)]
impl NeuralNetwork {
    pub fn new(
        units: i32,
        input_dim: i32,
        loss: String,
        optimizer: String,
        activation: String,
    ) -> NeuralNetwork {
        let input = Layer::new(units, input_dim, activation);
        NeuralNetwork {
            loss,
            optimizer,
            layers: vec![input],
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut output = inputs.clone();
        for i in 0..self.layers.len() {
            output = self.layers[i].forward(output);
        }
        normalize_ouput(output)
    }

    //ahora la etiqueta del ejemplo es un vector one-hot-encoding
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<i32>>, learning_rate: f64, epochs: i32) {
        for i in 0..epochs {
            println!("------------------------> Epoch: {:?}", i);
            self.show_layers();
            for i in 0..x.len() {
                match self.optimizer.as_str() {
                    "gd" => {
                        let output = self.forward(x[i].clone());
                        calculate_deltas(self, output, y[i].clone());
                        adjust_weights(self, learning_rate);
                    }
                    _ => {
                        panic!("Optimizer not implemented");
                    }
                }
            }
        }
    }

    pub fn add(&mut self, units: i32, activation: String) {
        let input_dim = self.layers[self.layers.len() - 1].rows;
        let layer = Layer::new(units, input_dim, activation);
        self.layers.push(layer);
    }

    fn show_details(&self) {
        println!("---------------Neural Network Details------------------");
        println!("Loss: {:?}", self.loss);
        println!("Optimizer: {:?}", self.optimizer);
        for i in 0..self.layers.len() {
            println!("Layer: {}", i);
            self.layers[i].show_details();
            println!();
        }
    }

    fn show_layers(&self) {
        for i in 0..self.layers.len() {
            println!("Layer: {}", i);
            self.layers[i].show_details();
            println!();
        }
    }
}
