#![allow(warnings)]

use crate::neuron::{Layer, Neuron};
use crate::value::Value;

mod neuron;
mod value;
mod visualize;

fn main() {
    let layer = Layer::<3, 2>::new();
    let x = [Value::new(1.0, "x1"), Value::new(2.0, "x2"), Value::new(-1.0, "x3")];
    let output = layer.call(&x);
    for (i, out) in output.iter().enumerate() {
        println!("Output {}: data = {:.4}, grad = {:.4}", i, out.data.borrow(), out.grad.borrow());
    }
}
