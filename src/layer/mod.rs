use crate::activation_fn::derivate::{relu_derivative, sigmoid_derivative, softmax_derivative};
use crate::activation_fn::{relu, sigmoid, softmax};
use crate::numrs::math::{add_vecs, lineal_transform};
use crate::numrs::randgen::randfloatmatrix;
#[derive(Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>, //matrix of weights
    pub biases: Vec<f64>,       //vec of bias
    pub activation: String,     //activation function
    pub input: Vec<f64>,        //The input of the layer is necessary for backpropagation
    pub deltas: Vec<f64>,       //The delta of the layer is necessary for backpropagation
    pub aa_az: Vec<f64>,        //derivada de la funcion de activacion con respecto a la entrada z
    pub ac_aa: Vec<f64>,        //derivada de la funcion de costo con respecto a la activacion a
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
            input: Vec::new(),  /*este valor se puede obtener durante el forward */
            deltas: Vec::new(), /*este valor se puede obtener durante el backward */
            aa_az: Vec::new(),  /*este valor se puede obtener durante el forward */
            ac_aa: Vec::new(),  /*este valor se puede obtener durante el backward */
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.input = inputs.clone();
        let mut output = lineal_transform(self.weights.clone(), inputs.clone());
        output = add_vecs(output, self.biases.clone());
        match self.activation.as_str() {
            "relu" => {
                /*en este punto aun no se aplica la funcion de activacion */
                self.aa_az = relu_derivative(output.clone());
                output = relu(output);
            }
            "sigmoid" => {
                self.aa_az = sigmoid_derivative(output.clone());
                output = sigmoid(output);
            }
            "softmax" => {
                self.aa_az = softmax_derivative(output.clone());
                output = softmax(output);
            }
            _ => {
                panic!("Activation function not implemented");
            }
        }
        return output;
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
