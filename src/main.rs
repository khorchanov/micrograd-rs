#![allow(warnings)]

use crate::neuron::{Layer, Neuron};
use crate::value::Value;

mod neuron;
mod value;
mod visualize;

fn main() {
    // Original layer example
    let layer = Layer::<3, 2>::new();
    let x = [
        Value::new(1.0, "x1"),
        Value::new(2.0, "x2"),
        Value::new(-1.0, "x3"),
    ];
    let output = layer.call(&x);
    for (i, out) in output.iter().enumerate() {
        println!(
            "Output {}: data = {:.4}, grad = {:.4}",
            i,
            out.data.borrow(),
            out.grad.borrow()
        );
    }

    println!("\n--- Using MLP macro ---");

    // Create an MLP with dimensions: 3 -> 4 -> 4 -> 1
    // This creates a function that takes [Value; 3] and returns [Value; 1]
    let mlp_fn = mlp!(3, 4, 4, 1);

    let input = [
        Value::new(2.0, "x1"),
        Value::new(3.0, "x2"),
        Value::new(-1.0, "x3"),
    ];

    let result = mlp_fn(&input);
    println!("MLP (3->4->4->1) result: {:.4}", result[0].data.borrow());

    // Another example: 2 -> 3 -> 1
    let mlp_fn2 = mlp!(2, 3, 1);
    let input2 = [Value::new(1.5, "a"), Value::new(-2.5, "b")];
    let result2 = mlp_fn2(&input2);
    println!("MLP (2->3->1) result: {:.4}", result2[0].data.borrow());

    // Test with many layers: 3 -> 8 -> 6 -> 4 -> 3 -> 2 -> 1
    let deep_mlp = mlp!(3, 8, 6, 4, 3, 2, 1);
    let input3 = [
        Value::new(1.0, "x1"),
        Value::new(0.5, "x2"),
        Value::new(-0.5, "x3"),
    ];
    let result3 = deep_mlp(&input3);
    println!(
        "Deep MLP (3->8->6->4->3->2->1) result: {:.4}",
        result3[0].data.borrow()
    );

    // Even deeper: 2 -> 5 -> 5 -> 5 -> 5 -> 5 -> 3 -> 1
    let very_deep = mlp!(2, 5, 5, 5, 5, 5, 3, 1);
    let input4 = [Value::new(0.8, "a"), Value::new(-0.3, "b")];
    let result4 = very_deep(&input4);
    println!(
        "Very Deep MLP (2->5->5->5->5->5->3->1) result: {:.4}",
        result4[0].data.borrow()
    );

    // Test extreme depth: 10 layers
    let extreme = mlp!(3, 10, 8, 7, 6, 5, 4, 3, 2, 1);
    let input5 = [
        Value::new(1.0, "x"),
        Value::new(0.0, "y"),
        Value::new(-1.0, "z"),
    ];
    let result5 = extreme(&input5);
    println!(
        "Extreme MLP (10 layers) result: {:.4}",
        result5[0].data.borrow()
    );
}
