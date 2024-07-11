use crate::layer::Layer;
use crate::loss_fn::derivate::{cross_entropy_derivative, mse_derivative};
use crate::network::NeuralNetwork;
use crate::numrs::math::{clamped_matrix, hadamard, lineal_transform, outer, tranpose,clamped};

fn gd_ac_aa_4_outputlayer(
    nn: &mut NeuralNetwork,
    activation: Vec<f64>,
    prediction: Vec<i32>,
) ->Vec<f64> {
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


fn update_weights(l: &mut Layer, gradient: Vec<Vec<f64>>, learning_rate: f64) {
    for i in 0..l.rows {
        for j in 0..l.cols {
            l.weights[i as usize][j as usize] -= learning_rate * gradient[i as usize][j as usize];
        }
    }

    l.weights = clamped_matrix(l.weights.clone(), -1.0, 1.0);
}

fn update_bias(l : &mut Layer,deltas : Vec<f64>, learning_rate : f64){

    let d = clamped(deltas.clone(), -5.0, 5.0);

    for i in 0..l.rows {
        l.biases[i as usize] -= learning_rate * d[i as usize];
    }
    l.biases = clamped(l.biases.clone(), -1.0, 1.0);

}

pub fn adjust_weights(nn: &mut NeuralNetwork, learning_rate: f64) {
    for i in 0..nn.layers.len() {
        let deltas = nn.layers[i].deltas.clone();
        let input = nn.layers[i].input.clone();
        let gradient = clamped_matrix(outer(deltas.clone(), input), -5.0, 5.0);
        update_weights(&mut nn.layers[i], gradient, learning_rate);
        update_bias(&mut nn.layers[i], deltas, learning_rate);
        //rounded_weights(&mut nn.layers[i]);
    }
}

// fn rounded_weights(l: &mut Layer){
//     for i in 0..l.rows {
//         for j in 0..l.cols {
//             l.weights[i as usize][j as usize] = l.weights[i as usize][j as usize];
//         }
//     }
// }
