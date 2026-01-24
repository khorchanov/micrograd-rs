#![allow(warnings)]

use crate::neuron::Neuron;
use crate::value::Value;

mod neuron;
mod value;
mod visualize;

fn main() {
    let neuron = Neuron::<2>::new();
    let x = [Value::new(1.0, "x1"), Value::new(2.0, "x2")];
    let output = neuron.call(&x);
    println!("Neuron output: {}", output);
}
