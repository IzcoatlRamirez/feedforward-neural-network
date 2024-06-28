mod dataframe;
mod network;
mod numrs;
mod layer;
use network::NeuronalNetwork;
//use layer::Layer;
fn main() {
    let nn = NeuronalNetwork::new(3, 2, "mse".to_string(), "sgd".to_string(), "accuracy".to_string(), "relu".to_string());
    nn.show_details();

    /*
    let inputs = vec![1.0, 2.0, 3.0];
    let mut layer1 = Layer::new(3, 3, "relu".to_string());
    layer1.show_details();
    let output = layer1.forward(inputs);
    println!("{:?}", output);
     */
}
