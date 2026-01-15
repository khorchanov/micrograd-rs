use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

mod visualize;

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, grad={}, label={})",
            self.data.borrow(),
            self.grad.borrow(),
            self.label.as_deref().unwrap_or("None")
        )
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let result = Value {
            data: Rc::new(RefCell::new(*self.data.borrow() + *other.data.borrow())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Add(
                Rc::new(RefCell::new(self.clone())),
                Rc::new(RefCell::new(other.clone())),
            )),
            label: Some(
                self.label.clone().unwrap_or_default()
                    + "+"
                    + &other.label.clone().unwrap_or_default(),
            ),
        };
        result
    }
}

impl Add<f32> for Value {
    type Output = Value;
    fn add(self, other: f32) -> Value {
        let other_value = Value::new(other, other.to_string().as_str());
        self + other_value
    }
}

impl Add<Value> for f32 {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let self_value = Value::new(self, self.to_string().as_str());
        self_value + other
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let result = Value {
            data: Rc::new(RefCell::new(*self.data.borrow() * *other.data.borrow())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Mul(
                Rc::new(RefCell::new(self.clone())),
                Rc::new(RefCell::new(other.clone())),
            )),
            label: Some(
                self.label.clone().unwrap_or_default() + &other.label.clone().unwrap_or_default(),
            ),
        };
        result
    }
}

impl Mul<f32> for Value {
    type Output = Value;
    fn mul(self, other: f32) -> Value {
        let other_value = Value::new(other, other.to_string().as_str());
        self * other_value
    }
}

impl Mul<Value> for f32 {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let self_value = Value::new(self, self.to_string().as_str());
        self_value * other
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Tanh(Rc<RefCell<Value>>),
    Add(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Mul(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
}

#[derive(Debug, Clone)]
pub struct Value {
    pub data: Rc<RefCell<f32>>,
    pub grad: Rc<RefCell<f32>>,
    pub op: Option<Operation>,
    pub label: Option<String>,
}

impl Value {
    fn new(data: f32, label: &str) -> Self {
        Value {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            op: None,
            label: Some(label.to_string()),
        }
    }

    fn tanh(&self, label: String) -> Self {
        let x = *self.data.borrow();
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Value {
            data: Rc::new(RefCell::new(t)),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Tanh(Rc::new(RefCell::new(self.clone())))),
            label: Some(label),
        }
    }

    fn backward(&self) {
        match self.op {
            Some(Operation::Tanh(ref a)) => {
                let x = *a.borrow().data.borrow();
                let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
                *a.borrow_mut().grad.borrow_mut() += (1.0 - t.powi(2)) * *self.grad.borrow();
            }
            Some(Operation::Add(ref a, ref b)) => {
                *a.borrow_mut().grad.borrow_mut() += *self.grad.borrow();
                *b.borrow_mut().grad.borrow_mut() += *self.grad.borrow();
            }
            Some(Operation::Mul(ref a, ref b)) => {
                *a.borrow_mut().grad.borrow_mut() +=
                    *b.borrow().data.borrow() * *self.grad.borrow();
                *b.borrow_mut().grad.borrow_mut() +=
                    *a.borrow().data.borrow() * *self.grad.borrow();
            }
            None => {}
        }
    }

    fn full_backward(&mut self) {
        let topo = self.reversed_topo();
        *self.grad.borrow_mut() = 1.0;
        for node in topo {
            node.backward();
        }
    }

    fn reversed_topo(&self) -> Vec<Value> {
        let mut nodes = Vec::new();

        fn build_topo(v: Value, nodes: &mut Vec<Value>) {
            if let Some(ref op) = v.op {
                match op {
                    Operation::Tanh(a) => {
                        build_topo(a.borrow().clone(), nodes);
                    }
                    Operation::Add(a, b) => {
                        build_topo(a.borrow().clone(), nodes);
                        build_topo(b.borrow().clone(), nodes);
                    }
                    Operation::Mul(a, b) => {
                        build_topo(a.borrow().clone(), nodes);
                        build_topo(b.borrow().clone(), nodes);
                    }
                }
            }
            nodes.push(v);
        }

        build_topo(self.clone(), &mut nodes);
        nodes.reverse();
        nodes
    }
}


fn main() {
    let x1 = Value::new(2.0, "x1");
    let x2 = Value::new(0.0, "x2");
    let w1 = Value::new(-3.0, "w1");
    let w2 = Value::new(1.0, "w2");
    let b = Value::new(6.8813735870195432, "b");
    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let mut o = n.tanh("o".to_string());
    o.full_backward();

    println!("n: {}", n);
    
    // Test topo
    let topo_order = o.reversed_topo();
    println!("\nTopological order ({} nodes):", topo_order.len());
    for (i, node) in topo_order.iter().enumerate() {
        println!("  {}. {}", i + 1, node);
    }
    // Draw the computational graph
    visualize::draw_dot(&o, "computation_graph");
}

fn _test() {
    let x1 = Value::new(2.0, "x1");
    let x2 = Value::new(4.0, "x2");

    println!("=== Addition Tests ===");
    println!("x1 + 3.0 = {}", x1.clone() + 3.0);
    println!("3.0 + x1 = {}", 3.0 + x1.clone());
    println!("x1 + x2 = {}", x1.clone() + x2.clone());

    println!("\n=== Multiplication Tests ===");
    println!("x1 * 5.0 = {}", x1.clone() * 5.0);
    println!("5.0 * x1 = {}", 5.0 * x1.clone());
    println!("x1 * x2 = {}", x1.clone() * x2.clone());

    println!("\n=== Combined Operations ===");
    println!("(x1 + 1.0) * 2.0 = {}", (x1.clone() + 1.0) * 2.0);
    println!("x1 * x2 + 3.0 = {}", x1.clone() * x2.clone() + 3.0);
    println!(
        "2.0 * x1 + 3.0 * x2 = {}",
        2.0 * x1.clone() + 3.0 * x2.clone()
    );
}