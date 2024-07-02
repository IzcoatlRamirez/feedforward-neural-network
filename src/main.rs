mod dataframe;
mod layer;
mod network;
mod numrs;
use numrs::ohe::{one_hot_encoding, one_hot_encoding_target};

fn main() {
    let n_examples = 100;

    let x = one_hot_encoding(n_examples, 10);
    let y = one_hot_encoding_target(n_examples, 3);

    let (x_train, x_test, y_train, y_test) = dataframe::df::simple_split_one_hot(x, y, 0.7);

    let mut model = network::NeuralNetwork::new(
        10,
        10,
        "categorical_crossentropy".to_string(),
        "gd".to_string(),
        "accuracy".to_string(),
        "relu".to_string(),
    );

    
}
