use rand::Rng;

use crate::value::Value;

/// Macro to create a Multi-Layer Perceptron (MLP) with static dimensions.
/// 
/// # Usage
/// 
/// The macro takes a comma-separated list of layer dimensions and returns a function
/// that performs the forward pass through the network.
/// 
/// # Examples
/// 
/// ```
/// // Create a 3->4->1 network (3 inputs, 1 hidden layer with 4 neurons, 1 output)
/// let network = mlp!(3, 4, 1);
/// 
/// let input = [Value::new(1.0, "x1"), Value::new(2.0, "x2"), Value::new(3.0, "x3")];
/// let output = network(&input);
/// ```
/// 
/// ```
/// // Create a 2->5->3->1 network (2 inputs, 2 hidden layers, 1 output)
/// let network = mlp!(2, 5, 3, 1);
/// 
/// // Or with many layers: 10->8->6->4->2->1
/// let deep_network = mlp!(10, 8, 6, 4, 2, 1);
/// ```
/// 
/// The macro supports networks with any number of layers!
#[macro_export]
macro_rules! mlp {
    // Entry point: collect all dimensions and start processing
    ($($dims:literal),+ $(,)?) => {{
        mlp!(@build_fn [$($dims),+])
    }};
    
    // Build the function with input and output dimensions
    (@build_fn [$input:literal, $($rest:literal),+]) => {{
        mlp!(@get_output [$input, $($rest),+] -> $($rest),+)
    }};
    
    // Extract the output dimension (last in the list)
    (@get_output [$($all:literal),+] -> $last:literal) => {{
        move |input: &[Value; mlp!(@first $($all),+)]| -> [Value; $last] {
            mlp!(@chain_layers input, [$($all),+])
        }
    }};
    (@get_output [$($all:literal),+] -> $first:literal, $($rest:literal),+) => {{
        mlp!(@get_output [$($all),+] -> $($rest),+)
    }};
    
    // Get the first element from a list
    (@first $first:literal $(, $rest:literal)*) => { $first };
    
    // Chain layers: base case with just two dimensions (one layer)
    (@chain_layers $input:expr, [$in:literal, $out:literal]) => {{
        let layer = Layer::<$in, $out>::new();
        layer.call($input)
    }};
    
    // Chain layers: recursive case with more than two dimensions
    (@chain_layers $input:expr, [$in:literal, $next:literal $(, $rest:literal)+]) => {{
        let layer = Layer::<$in, $next>::new();
        let output = layer.call($input);
        mlp!(@chain_layers &output, [$next $(, $rest)+])
    }};
}


pub struct Neuron<const N: usize> {
    pub w: [Value; N],
    pub b: Value,
}

impl<const N: usize> Neuron<N> {
    pub fn new() -> Neuron<N> {
        let w = std::array::from_fn(|i| {
            Value::new(rand::rng().random_range(-1.0..1.0), &format!("w{}", i))
        });
        let b = Value::new(rand::rng().random_range(-1.0..1.0), "b");
        Neuron { w, b }
    }

    pub fn call(&self, x: &[Value; N]) -> Value {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = act + wi.clone() * xi.clone();
        }
        act.tanh()
    }
}

pub struct Layer<const N: usize, const M: usize> {
    pub neurons: [Neuron<N>; M],
}

impl<const N: usize, const M: usize> Layer<N, M> {
    pub fn new() -> Layer<N, M> {
        let neurons = std::array::from_fn(|_| Neuron::<N>::new());
        Layer { neurons }
    }

    pub fn call(&self, x: &[Value; N]) -> [Value; M] {
        std::array::from_fn(|i| self.neurons[i].call(x))
    }
}