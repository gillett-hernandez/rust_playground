use packed_simd::f32x4;
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Type {
    uint,
    f32,
    f32x4,
    bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Value {
    uint(u128),
    int(i128),
    f32(f32),
    f32x4(f32x4),
    bool(bool),
}

impl SubAssign for Value {
    fn sub_assign(&mut self, rhs: Value) {
        // match (self, rhs) {
        //     (Value::uint(a), Value::uint(b)) => {
        //         *a -= b;
        //     }
        // }
    }
}

// all ops check the types of their incoming args
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum OP {
    ADD,              // r0 + r1
    SUB,              // r0 - r1
    MUL,              // r0 * r1
    DIV,              // r0 / r1
    POWI,             // r0 ^ r1
    POWF,             // r0 ^ r1
    MOD,              // r0 % r1
    SQRT,             // alias for CONST(f32(0.5), 1) -> POWF
    INV,              // alias for MOV(0, 1) -> CONST(f32(1.0), 0) -> DIV
    MOV(u8, u8), // move the contents of register 1 to register 2, overwriting whatever was in register 2 and clearing register 1
    COPY(u8, u8), // similar to the above, except not erasing register 1.
    SWAP(u8, u8), // swap the contents of these two registers
    PUSH(u8),    // push register onto stack
    POP(u8),     // pop stack into register
    CONST(Value, u8), // load const Value into register
    CAST(u8, Type), // cast register to type
    JUMP(usize),
    JUMPIF(u8, usize), // conditionally jump based on register ri to address ai
}

pub struct Machine {
    pub instructions: Vec<OP>,
    pub stack: Vec<Value>, // consider replacing with smallvec to limit stack depth.
    pub registers: [Value; 256],
}

pub enum VMError {
    TypeError(String),
}

/*

Basic Virtual Machine Requirements:
* Must allow for arbitrary bound inputs of varying types
* Must allow for arbitrary bound "textures" or other machines
* Must return arbitrary types, as described in the virtual type enum

*/

impl Machine {
    pub fn new(instructions: Vec<OP>, stack: Vec<Value>) -> Self {
        Machine {
            instructions,
            stack,
            registers: [Value::uint(0); 256],
        }
    }
    pub fn execute(&mut self) -> Result<(), VMError> {
        let mut instruction = self.instructions[0];
        loop {
            match instruction {
                OP::JUMP(destination) => instruction = self.instructions[destination],
                OP::JUMPIF(register, destination) => {
                    if let Value::bool(r) = self.registers[register as usize] {
                        if r {
                            instruction = self.instructions[destination];
                        }
                    } else {
                        return Err(VMError::TypeError(
                            "JUMPIF instruction's condition was not a bool".to_owned(),
                        ));
                    }
                }
                OP::ADD => {
                    // self.registers[0] += self.registers[1];
                }
                OP::SUB => {
                    // self.registers[0] -= self.registers[1];
                }

                _ => {}
            }
        }
    }
}

fn main() {}
