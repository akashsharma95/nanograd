use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub struct Value {
    inner: Rc<RefCell<Node>>,
}

struct Node {
    data: f64,
    grad: f64,
    prev: Vec<Value>,
    op: Op,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Leaf,
    Add,
    Mul,
    Relu,
}