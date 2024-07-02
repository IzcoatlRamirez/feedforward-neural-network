mod dataframe;
mod layer;
mod network;
mod numrs;
mod gradient_descend;
mod loss_fn;
mod activation_fn;
use numrs::ohe::{one_hot_encoding, one_hot_encoding_target};

fn main() {
    let n_examples = 10;

    let x = one_hot_encoding(n_examples, 10);
    let y = one_hot_encoding_target(n_examples, 3);

    let (x_train, x_test, y_train, y_test) = dataframe::df::simple_split_one_hot(x, y, 0.7);

    println!("x_train: {:?}", x_train);
    println!("y_train: {:?}", y_train);
    println!("x_test: {:?}", x_test);
    println!("y_test: {:?}", y_test);

    /*construir un modulo que permita cargar datasets de csv obteniendo un vector para cada componente (x,y)
    donde x es una fila del dataset y y es la etiqueta correspondiente la cual sera convertida a one-hot-encoding
    */    
}
