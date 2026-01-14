use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Debug, Clone)]
enum Operation {
    Tanh(Rc<RefCell<Value>>),
    Add(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
}

#[derive(Debug, Clone)]
struct Value {
    data: f32,
    grad: f32,
    op: Option<Operation>,
}

impl Value {
    fn new(data: f32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            op: None,
        }))
    }

    fn tanh(self_rc: &Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let x = self_rc.borrow().data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Rc::new(RefCell::new(Value {
            data: t,
            grad: 0.0,
            op: Some(Operation::Tanh(Rc::clone(self_rc))),
        }))
    }

    fn backward(self_rc: &Rc<RefCell<Self>>) {
        match self_rc.borrow().op {
            Some(Operation::Tanh(ref a)) => {
                let x = self_rc.borrow().data;
                let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
                a.borrow_mut().grad += (1.0 - t.powi(2)) * self_rc.borrow().grad;
            }
            Some(Operation::Add(ref a, ref b)) => {
                a.borrow_mut().grad += self_rc.borrow().grad;
                b.borrow_mut().grad += self_rc.borrow().grad;
            }
            None => {}
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

impl Add for Value {
    type Output = Rc<RefCell<Value>>;

    fn add(self, other: Value) -> Self::Output {
        Rc::new(RefCell::new(Value {
            data: self.data + other.data,
            grad: 0.0,
            op: Some(Operation::Add(
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(other)),
            )),
        }))
    }
}

fn main() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(1.0);
    let x1x2 = x1.as_ptr() + x1.as_ptr();
    let o = Value::tanh(&x1x2);

    o.borrow_mut().grad = 1.0;
    Value::backward(&o);

    println!("x1: {:#?}", x1);
    println!("Output: {:#?}", o);
}

fn _test() {
    // let x1 = Value::new(2.0);
    // let x2 = Value::new(0.0);
    // let w1 = Value::new(-3.0);
    // let w2 = Value::new(1.0);
    // let b = Value::new(6.8813735870195432);
    // let x1w1 = x1 * w1;
    // let x2w2 = x2 * w2;
    // let x1w1x2w2 = x1w1 + x2w2;
    // let n = x1w1x2w2 + b;
    // let mut o = n.tanh();
    // o.grad = 1.0;
    // o.backward();

    // println!("n: {}", n);
    // println!("Output: {:#?}", o);
}
