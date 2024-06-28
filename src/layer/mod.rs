use crate::numrs::{add_vecs, lineal_transform, randfloatmatrix};
use std::vec;
pub struct Layer {
    pub weights: Vec<Vec<f64>>, /*matrix of weights*/
    pub biases: Vec<f64>,       /*vec of bias */
    pub activation: String,     /*activation function*/
    pub output: Vec<f64>,       /*The output of the layer is necessary for backpropagation*/
}
/*en el constructor instanciamos la capa de entrada(debemos saber cuantas neuronas y cuantas entradas tendra el modelo) ,
despues las capas ocultas se agregaran con el metodo add
*/
#[allow(dead_code)]
impl Layer {
    /*este constructor es usado para crear todas las capas de la red
    para la primer capa necesitamos conocer la cantidad de neuronas (units) y la cantidad de input_dim (pesos)*/
    /*para las capas ocultas y el metodo add input_dim sera la cantidad de salidas producida por la capa anterior (una por cada neurona de la capa anterior)
    en cambio units sera la cantidad de neuronas de la capa actual (puede ser cualquiera)*/
    pub fn new(units: i32, input_dim: i32, activation: String) -> Layer {
        Layer {
            /*decidir si generar una matriz aleatoria cambiando la semilla */
            weights: randfloatmatrix(-1.0, 1.0, units, input_dim, 0),
            biases: vec![1.0; input_dim as usize],
            activation,
            output: Vec::new(),
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut output = lineal_transform(self.weights.clone(), inputs.clone());
        output = add_vecs(output, self.biases.clone());
        match self.activation.as_str() {
            "relu" => {
                output = self.relu(output);
            }
            "sigmoid" => {
                output = self.sigmoid(output);
            }
            _ => {
                panic!("Activation function not implemented");
            }
        }
        self.output = output.clone();
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

    fn sigmoid(&self, mut x: Vec<f64>) -> Vec<f64> {
        for i in 0..x.len() {
            x[i] = 1.0 / (1.0 + (-x[i]).exp());
        }
        return x;
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
