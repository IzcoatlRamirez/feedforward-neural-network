mod dataframe;
mod network;
mod numrs;
mod layer;
use network::NeuralNetwork;
fn main() {
    let mut nn = NeuralNetwork::new(3, 3, "mse".to_string(), "sgd".to_string(), "accuracy".to_string(), "relu".to_string());
    nn.add(3, "relu".to_string());
    nn.add(10, "softmax".to_string());
    nn.show_details();

    let inputs = vec![1.0, 2.0, 3.0];
    let output = nn.forward(inputs);
    println!("{:?}", output);

}
