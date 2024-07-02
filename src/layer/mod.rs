use crate::numrs::math::{add_vecs, lineal_transform};
use crate::numrs::randgen::randfloatmatrix;
use std::vec;

#[derive(Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,  //matrix of weights
    pub biases: Vec<f64>,        //vec of bias 
    pub activation: String,      //activation function
    pub input: Vec<f64>,         //The input of the layer is necessary for backpropagation
    pub deltas: Vec<f64>,         //The delta of the layer is necessary for backpropagation
    pub aa_az: Vec<f64>,          //derivada de la funcion de activacion con respecto a la entrada z
    pub ac_aa: Vec<f64>,          //derivada de la funcion de costo con respecto a la activacion a
    pub rows: i32,
    pub cols: i32,
}

#[allow(dead_code)]
impl Layer {
    /*
    cantidad de neuronas -> units
    cantidad de entradas -> input_dim (cantidad de neuronas de la capa anterior o 
    cantidad de features del dataset, ademas coincide con la cantidad de columnas de la matriz de pesos)
    */
    pub fn new(units: i32, input_dim: i32, activation: String) -> Layer {
        Layer {
            /*decidir si generar una matriz aleatoria cambiando la semilla */
            weights: randfloatmatrix(-1.0, 1.0, units, input_dim, 0),
            biases: vec![1.0; units as usize],
            activation,
            rows: units,
            cols: input_dim,
            input: Vec::new(), /*este valor se puede obtener durante el forward */
            deltas: Vec::new(), /*este valor se puede obtener durante el backward */
            aa_az: Vec::new(), /*este valor se puede obtener durante el forward */
            ac_aa: Vec::new(), /*este valor se puede obtener durante el backward */
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.input = inputs.clone();
        let mut output = lineal_transform(self.weights.clone(), inputs.clone());
        output = add_vecs(output, self.biases.clone());
        match self.activation.as_str() {
            "relu" => {
                /*en este punto aun no se aplica la funcion de activacion */
                self.aa_az = self.relu_derivative(output.clone());
                output = self.relu(output);
            }
            "sigmoid" => {
                self.aa_az = self.sigmoid_derivative(output.clone());
                output = self.sigmoid(output);
            }
            "softmax" => {
                self.aa_az = self.softmax_derivative(output.clone());
                output = self.softmax(output);
            }
            _ => {
                panic!("Activation function not implemented");
            }
        }
        return output;
    }

    fn relu(&self, mut x: Vec<f64>) -> Vec<f64> {
        for i in 0..x.len() {
            if x[i] < 0.0 {
                x[i] = 0.0;
            }
            x[i] = x[i];
        }
        return x;
    }

    fn relu_derivative(&self, x: Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            if x[i] > 0.0 {
                result[i] = 1.0;
            }
        }
        return result;
    }

    fn softmax_derivative(&self, x: Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            result[i] = x[i] * (1.0 - x[i]);
        }
        return result;
    }

    fn sigmoid_derivative(&self, x: Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            result[i] = x[i] * (1.0 - x[i]);
        }
        return result;
    }

    fn sigmoid(&self, mut x: Vec<f64>) -> Vec<f64> {
        for i in 0..x.len() {
            x[i] = 1.0 / (1.0 + (-x[i]).exp());
        }
        return x;
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
        println!("Biases: {:?}", self.biases);
        println!("Activation: {:?}", self.activation);
        println!("Weights: ");
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                print!("{} ", self.weights[i][j]);
            }
            println!();
        }
    }
}
