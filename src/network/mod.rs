use crate::layer::Layer;
use std::vec;
pub struct NeuronalNetwork {
    pub loss: String,
    pub optimizer: String,
    pub metrics: String,
    layers: Vec<Layer>,
}

#[allow(dead_code)]
impl NeuronalNetwork {
    pub fn new(
        units: i32,
        input_dim: i32,
        loss: String,
        optimizer: String,
        metrics: String,
        activation: String,
    ) -> NeuronalNetwork {
        let input = Layer::new(units, input_dim, activation);
        NeuronalNetwork {
            loss,
            optimizer,
            metrics,
            layers: vec![input],
        }
    }

    pub fn show_details(&self) {
        println!("---------------Neuronal Network Details------------------");
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
