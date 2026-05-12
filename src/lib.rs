use std::{cell::RefCell, rc::Rc};
use std::ops::{Add, Mul};

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

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Node {
                data,
                grad: 0.0,
                prev: Vec::new(),
                op: Op::Leaf,
            })),
        }
    }

    fn from_op(data: f64, prev: Vec<Value>, op: Op) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Node {
                data,
                grad: 0.0,
                prev,
                op,
            })),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.inner.borrow_mut().grad = grad;
    }

    pub fn add_grad(&self, grad: f64) {
        self.inner.borrow_mut().grad += grad;
    }

    pub fn relu(&self) -> Self {
        let data = if self.data() > 0.0 { self.data() } else { 0.0 };
        Self::from_op(data, vec![self.clone()], Op::Relu)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let data = self.data() + rhs.data();

        Value::from_op(data, vec![self, rhs], Op::Add)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let data = self.data() * rhs.data();
        Value::from_op(data, vec![self, rhs], Op::Mul)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let a = Value::new(1.0);
        let b = a.relu();
        assert_eq!(b.data(), 1.0);
    }

    #[test]
    fn test_add() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = a + b;
        assert_eq!(c.data(), 3.0);
    }
    
    #[test]
    fn test_mul() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = a * b;
        assert_eq!(c.data(), 2.0);
    }
}