use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
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

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * -1f32
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + -other
    }
}

impl Sub<f32> for Value {
    type Output = Value;
    fn sub(self, other: f32) -> Value {
        let other_value = Value::new(other, other.to_string().as_str());
        self - other_value
    }
}

impl Sub<Value> for f32 {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        let self_value = Value::new(self, self.to_string().as_str());
        self_value - other
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

impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.powf(-1f32)
    }
}

impl Div<f32> for Value {
    type Output = Value;
    fn div(self, other: f32) -> Value {
        self * other.powf(-1f32)
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
    Pow(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Exp(Rc<RefCell<Value>>),
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

    fn tanh(&self) -> Self {
        let x = *self.data.borrow();
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Value {
            data: Rc::new(RefCell::new(t)),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Tanh(Rc::new(RefCell::new(self.clone())))),
            label: Some(format!("tan({})", &self.label.clone().unwrap_or_default())),
        }
    }

    fn exp(&self) -> Self {
        let x = *self.data.borrow();
        Value {
            data: Rc::new(RefCell::new(x.exp())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Exp(Rc::new(RefCell::new(self.clone())))),
            label: Some(format!("e^({})", &self.label.clone().unwrap_or_default())),
        }
    }

    fn pow(&self, other: Value) -> Self {
        let x = *self.data.borrow();
        Value {
            data: Rc::new(RefCell::new(x.powf(*other.data.borrow()))),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Pow(
                Rc::new(RefCell::new(self.clone())),
                Rc::new(RefCell::new(other.clone())),
            )),
            label: Some(format!(
                "({})^({})",
                &self.label.clone().unwrap_or_default(),
                *other.data.borrow()
            )),
        }
    }

    fn powf(&self, other: f32) -> Self {
        let other_value = Value::new(other, other.to_string().as_str());
        self.pow(other_value)
    }

    fn backward(&self) {
        match self.op {
            Some(Operation::Tanh(ref a)) => {
                let x = *self.data.borrow();
                let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
                // cal also use 1 - tanh^2(self.data)
                *a.borrow_mut().grad.borrow_mut() += (1.0 - t.powi(2)) * *self.grad.borrow();
            }
            Some(Operation::Exp(ref a)) => {
                *a.borrow_mut().grad.borrow_mut() += *self.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Pow(ref a, ref k)) => {
                let x = *a.borrow().data.borrow();
                let k = *k.borrow().data.borrow();
                *a.borrow_mut().grad.borrow_mut() += k * x.powf(k - 1f32) * *self.grad.borrow();
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
                    Operation::Add(a, b) | Operation::Mul(a, b) | Operation::Pow(a, b) => {
                        build_topo(a.borrow().clone(), nodes);
                        build_topo(b.borrow().clone(), nodes);
                    }
                    Operation::Tanh(a) | Operation::Exp(a) => build_topo(a.borrow().clone(), nodes),
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
    let e = (2f32 * n.clone()).exp();
    let mut o = (e.clone() - 1f32) / (e + 1f32);
    o.full_backward();

    println!("n: {}", n);

    // Test topo
    let topo_order = o.reversed_topo();
    println!("\nTopological order ({} nodes):", topo_order.len());
    for (i, node) in topo_order.iter().enumerate() {
        println!("  {}. {}", i + 1, node);
    }
    // Draw the computational graph
    _test();
}

fn _test() {
    let x1 = Value::new(0.6931471805599453, "x1");
    let mut e = x1.tanh();
    e.full_backward();
    print!("e: {}\n", e);
    visualize::draw_dot(&e, "computation_graph");
}
