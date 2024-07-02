use crate::layer::Layer;
use crate::numrs::math::{hadamard, lineal_transform, tranpose};
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
                        self.calculate_deltas(output.clone(), y[i].clone());
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

    /*funcion para obtener el vector ac/aa de la capa de salida
    mse es la funcion de perdida
     */
    fn ac_aa_4_outputlayer(&self, activation: Vec<f64>, prediction: Vec<i32>) -> Vec<f64> {
        let mut result = vec![0.0; activation.len()];
        for i in 0..activation.len() {
            result[i] = activation[i] - prediction[i] as f64;
        }
        return result;
    }

    /*funcion para calcular todos los deltas */
    fn calculate_deltas(&mut self, activation: Vec<f64>, prediction: Vec<i32>) {
        /*primero calculamos el delta para la output layer */
        let oputput_layer = self.layers.len() - 1;
        let ac_aa = self.ac_aa_4_outputlayer(activation.clone(), prediction.clone());
        let aa_az = self.layers[self.layers.len() - 1].aa_az.clone();
        let delta = hadamard(ac_aa, aa_az);
        self.layers[oputput_layer].deltas = delta.clone();

        /*ahora calculamos los deltas para las capas ocultas */
        for i in self.layers.len() - 2..0 {
            let delta = hadamard(
                lineal_transform(
                    tranpose(self.layers[i + 1].clone().weights),
                    self.layers[i + 1].clone().deltas,
                ),
                self.layers[i].aa_az.clone(),
            );
            self.layers[i].deltas = delta.clone();
        }
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
