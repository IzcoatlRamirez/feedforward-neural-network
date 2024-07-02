mod dataframe;
mod network;
mod numrs;
mod layer;

/*construir funciones para obtener (ademas se almacenaran en su capa correspondiente durante el forward pass?):
-> derivada de la funcion de activacion con respecto a la entrada z 
-> derivada de la funcion de costo con respecto a la activacion a
-> vector de deltas para la capa de salida
-> vector de deltas para las capas ocultas

Esto implica verificar que campos nuevos se agregaran a la estructura de la capa:
    inputs
    outputs
    deltas
    vector de derivadas de la funcion de activacion con respecto a la entrada z (ac/az)
    vector de derivadas de la funcion de costo con respecto a la activacion a   (ac/aa)
*/
fn main() {

}
