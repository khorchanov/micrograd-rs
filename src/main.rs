#![allow(warnings)]

use std::iter::zip;

use crate::neuron::{Layer, Neuron};
use crate::value::Value;

mod neuron;
mod value;
mod visualize;

fn main() {
    let mut params: Vec<Value> = vec![];
    let mlp_fn = mlp!(3, 4, 4, 1);//TODO: macro should create a class to preserve the layers across training

    let xs = [
        [2.0.into(), 3.0.into(), (1.0).into()],
        [1.0.into(), 0.0.into(), (-9.0).into()],
        [4.0.into(), 4.0.into(), (-3.0).into()],
        [(-3.0).into(), 2.0.into(), (5.0).into()],
    ];

    let ys: [Value; 3] = [6.0.into(), (-8.0).into(), (4.0).into()];

    let mut ypred: Vec<Value> = Vec::new();
    for x in xs {
        let result = mlp_fn(&x, &mut params);
        ypred.push(result[0].clone());
    }

    // for y in ypred {
    //     println!("Result: {}", y.data.borrow());
    // }

    let mut loss: Value = 0.0.into();
    for (ypr, yout) in zip(ys, ypred) {
        loss = loss + (ypr - yout).powf(2.0);
    }

    loss.full_backward();

    println!("Loss: {}", loss.data.borrow());
    println!("Number of parameters: {}", params.len());
}
