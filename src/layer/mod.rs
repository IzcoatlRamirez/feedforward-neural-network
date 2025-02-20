use crate::activation_fn::derivate::{relu_derivative, sigmoid_derivative, softmax_derivative};
use crate::activation_fn::{relu, sigmoid, softmax};
use crate::numrs::math::{add_vecs, lineal_transform, round_vec};
use crate::numrs::randgen::{xavier_initialization,he_initialization, xavier_initialization_vec,he_initialization_vec};
#[derive(Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>, 
    pub biases: Vec<f64>,       
    pub activation: String,     
    pub input: Vec<f64>,        
    pub deltas: Vec<f64>,       
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
            weights: match activation.as_str() {
                "sigmoid" => xavier_initialization(units as usize, input_dim as usize),
                "relu" => he_initialization(units as usize, input_dim as usize),
                "softmax" => xavier_initialization(units as usize, input_dim as usize),
                _ => panic!("Activation function not implemented"),
                
            },
            biases: match activation.as_str() {
                "sigmoid" => xavier_initialization_vec(units as usize),
                "relu" => he_initialization_vec(units as usize),
                "softmax" => xavier_initialization_vec(units as usize),
                _ => panic!("Activation function not implemented"),
            },
            activation,
            rows: units,
            cols: input_dim,
            input: Vec::new(),  
            deltas: Vec::new(), 
            aa_az: Vec::new(),  
            ac_aa: Vec::new(),  
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.input = inputs.clone();
        let mut output = lineal_transform(self.weights.clone(), inputs.clone());
        output = add_vecs(output, self.biases.clone());
        match self.activation.as_str() {
            "relu" => {
                self.aa_az = relu_derivative(output.clone());
                output = round_vec(relu(output));
            }
            "sigmoid" => {
                self.aa_az = sigmoid_derivative(output.clone());
                output = round_vec(sigmoid(output));
            }
            "softmax" => {
                self.aa_az = softmax_derivative(output.clone());
                output = round_vec(softmax(output));
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
