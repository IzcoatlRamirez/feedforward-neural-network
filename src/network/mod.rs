use crate::gradient_descend::calculate_deltas;
use crate::layer::Layer;

/*considerar eliminar metrics para que un modulo externo se encarge del calculo */
pub struct NeuralNetwork {
    pub loss: String,
    pub optimizer: String,
    pub metrics: String,
    pub layers: Vec<Layer>,
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

    fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut output = inputs.clone();
        for i in 0..self.layers.len() {
            output = self.layers[i].forward(output);
        }
        output
    }

    //ahora la etiqueta del ejemplo es un vector one-hot-encoding
    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<i32>>, epochs: i32) {
        for _ in 0..epochs {
            for i in 0..x.len() {
                match self.optimizer.as_str() {
                    "gd" => {
                        let output = self.forward(x[i].clone());
                        calculate_deltas(self, output, y[i].clone());
                        /*adjust weight gradient descend */
                    }
                    _ => {
                        panic!("Optimizer not implemented");
                    }
                }
            }
        }
    }

    fn add(&mut self, units: i32, activation: String) {
        let input_dim = self.layers[self.layers.len() - 1].rows;
        let layer = Layer::new(units, input_dim, activation);
        self.layers.push(layer);
    }

    fn show_details(&self) {
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
