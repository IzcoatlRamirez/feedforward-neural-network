use crate::layer::Layer;
use crate::loss_fn::derivate::{cross_entropy_derivative, mse_derivative};
use crate::network::NeuralNetwork;
use crate::numrs::math::{hadamard, lineal_transform, outer, tranpose};

fn gd_ac_aa_4_outputlayer(
    nn: &mut NeuralNetwork,
    activation: Vec<f64>,
    prediction: Vec<i32>,
) -> Vec<f64> {
    match nn.loss.as_str() {
        "mse" => mse_derivative(activation, prediction),
        "cross_entropy" => cross_entropy_derivative(activation, prediction),
        _ => panic!("Loss function not implemented"),
    }
}

pub fn calculate_deltas(nn: &mut NeuralNetwork, activation: Vec<f64>, prediction: Vec<i32>) {
    let output_layer = nn.layers.len() - 1;
    let ac_aa = gd_ac_aa_4_outputlayer(nn, activation.clone(), prediction.clone());
    let aa_az = nn.layers[output_layer].aa_az.clone();
    let delta = hadamard(ac_aa, aa_az);
    nn.layers[output_layer].deltas = delta.clone();

    for i in (0..nn.layers.len() - 1).rev() {
        let delta = hadamard(
            lineal_transform(
                tranpose(nn.layers[i + 1].clone().weights),
                nn.layers[i + 1].clone().deltas,
            ),
            nn.layers[i].aa_az.clone(),
        );
        nn.layers[i].deltas = delta.clone();
    }

}

fn subtrac_weight(l: &mut Layer, gradient: Vec<Vec<f64>>, learning_rate: f64) {
    for i in 0..l.rows {
        for j in 0..l.cols {
            l.weights[i as usize][j as usize] -= learning_rate * gradient[i as usize][j as usize];
        }
    }
}
pub fn adjust_weights(nn: &mut NeuralNetwork, learning_rate: f64) {
    for i in 0..nn.layers.len() {
        let deltas = nn.layers[i].deltas.clone();
        let input = nn.layers[i].input.clone();
        let gradient = outer(deltas, input);
        // println!("Gradient: {:?}", gradient.clone());
        // println!("Weights: {:?}", nn.layers[i].weights.clone());
        // println!("Deltas: {:?}", nn.layers[i].deltas.clone());


        subtrac_weight(&mut nn.layers[i], gradient, learning_rate);
        rounded_weights(&mut nn.layers[i]);
    }
}

fn rounded_weights(l: &mut Layer){
    for i in 0..l.rows {
        for j in 0..l.cols {
            l.weights[i as usize][j as usize] = l.weights[i as usize][j as usize];
        }
    }
}
