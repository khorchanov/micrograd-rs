#![allow(warnings)]

use std::iter::{Sum, zip};

use crate::neuron::Layer;
use crate::value::Value;

mod neuron;
mod value;
mod visualize;

fn main() {
    let network = mlp!(3, 4, 4, 1);

    let xs = [
        [2.0.into(), 3.0.into(), (-1.0).into()],
        [3.0.into(), (-1.0).into(), 0.5.into()],
        [0.5.into(), 1.0.into(), 1.0.into()],
        [1.0.into(), 1.0.into(), (-1.0).into()],
    ];

    let ys: [Value; 4] = [1.0.into(), (-1.0).into(), (-1.0).into(), 1.0.into()];

    let loss_fn = |ypred: Vec<Value>| -> Value {
        Sum::sum(zip(ys.clone(), ypred).map(|(ypr, yout)| (ypr - yout).powf(2.0)))
    };

    for _ in 0..1000 {
        // Predict
        let mut ypred = Vec::new();
        for x in xs.iter() {
            let result = network.predict(x);
            ypred.push(result[0].clone());
        }

        // Compute the loss
        let mut loss: Value = loss_fn(ypred.clone());
        println!("Loss: {}", loss.data.borrow());
        let params = network.parameters();

        // Zero gradients before backward pass
        for p in params.iter() {
            *p.grad.borrow_mut() = 0.0;
        }

        //Train
        loss.full_backward();
        for p in params.iter() {
            *p.data.borrow_mut() -= 0.05 * *p.grad.borrow();
        }
    }
    let result = network.predict(&[1.0.into(), 1.0.into(), (-1.0).into()]);
    println!("result= {}", result[0].data.borrow());
}
