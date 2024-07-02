use crate::loss_fn::derivate::{cross_entropy_derivative, mse_derivative};
use crate::network::NeuralNetwork;
use crate::numrs::math::{hadamard, lineal_transform, tranpose};

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
    let oputput_layer = nn.layers.len() - 1;
    let ac_aa = gd_ac_aa_4_outputlayer(nn, activation.clone(), prediction.clone());
    let aa_az = nn.layers[nn.layers.len() - 1].aa_az.clone();
    let delta = hadamard(ac_aa, aa_az);
    nn.layers[oputput_layer].deltas = delta.clone();

    for i in nn.layers.len() - 2..0 {
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
