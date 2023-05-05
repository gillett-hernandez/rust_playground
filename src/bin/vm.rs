use packed_simd::f32x4;
use std::{
    error::Error,
    fmt::Display,
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Type {
    uint,
    int,
    f32,
    f32x4,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Value {
    uint(u128),
    int(i128),
    f32(f32),
    f32x4(f32x4),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VMError {
    CastError { source_type: Type, dest_type: Type },
    TypeError,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for VMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VMError::CastError {
                source_type,
                dest_type,
            } => f.write_fmt(format_args!(
                "tried to convert {} into {}, which is not supported",
                source_type, dest_type
            )),
            VMError::TypeError => todo!(),
        }
    }
}

impl Error for VMError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn Error> {
        self.source()
    }
}

impl Into<Value> for f32 {
    fn into(self) -> Value {
        Value::f32(self)
    }
}
impl Into<Value> for f32x4 {
    fn into(self) -> Value {
        Value::f32x4(self)
    }
}

impl Into<Value> for u128 {
    fn into(self) -> Value {
        Value::uint(self)
    }
}
impl Into<Value> for i128 {
    fn into(self) -> Value {
        Value::int(self)
    }
}

impl Value {
    pub fn ty(&self) -> Type {
        match self {
            Value::uint(_) => Type::uint,
            Value::int(_) => Type::int,
            Value::f32(_) => Type::f32,
            Value::f32x4(_) => Type::f32x4,
        }
    }
    pub fn cast(self, t: Type) -> Result<Self, VMError> {
        match (self, t) {
            (Value::uint(v), Type::uint) => Ok(self),
            (Value::uint(v), Type::int) => Ok(Value::int(v as i128)),
            (Value::uint(v), Type::f32) => Ok(Value::f32(v as f32)),
            (Value::uint(_), Type::f32x4) => Err(VMError::CastError {
                source_type: self.ty(),
                dest_type: t,
            }),
            (Value::int(v), Type::uint) => Ok(Value::uint(v as u128)),
            (Value::int(v), Type::int) => Ok(self),
            (Value::int(v), Type::f32) => Ok(Value::f32(v as f32)),
            (Value::int(_), Type::f32x4) => Err(VMError::CastError {
                source_type: self.ty(),
                dest_type: t,
            }),
            (Value::f32(v), Type::uint) => Ok((v.abs().floor() as u128).into()),
            (Value::f32(v), Type::int) => Ok((v.floor() as i128).into()),
            (Value::f32(v), Type::f32) => Ok(self),
            (Value::f32(v), Type::f32x4) => Ok(Value::f32x4(f32x4::splat(v))),
            (Value::f32x4(v), Type::uint) => Value::f32(v.extract(0)).cast(Type::uint),
            (Value::f32x4(v), Type::int) => Value::f32(v.extract(0)).cast(Type::int),
            (Value::f32x4(v), Type::f32) => Ok(Value::f32(v.extract(0))),
            (Value::f32x4(_), Type::f32x4) => Ok(self),
        }
    }
}

impl AddAssign for Value {
    fn add_assign(&mut self, rhs: Self) {}
}

impl SubAssign for Value {
    fn sub_assign(&mut self, rhs: Self) {
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
    ADD,              // r0 = r0 + r1
    SUB,              // r0 = r0 - r1
    MUL,              // r0 = r0 * r1
    DIV,              // r0 = r0 / r1
    POWI,             // r0 = r0 ^ r1
    POWF,             // r0 = r0 ^ r1
    MOD,              // r0 = r0 % r1
    SQRT,             // alias for CONST(f32(0.5), 1) -> POWF
    INV,              // alias for MOV(0, 1) -> CONST(f32(1.0), 0) -> DIV
    FMA,              // fused multiply add.
    MOV(u8, u8), // move the contents of register a to register b, overwriting whatever was in register b
    SWAP(u8, u8), // swap the contents of these two registers. uses the stack.
    PUSH(u8),    // push register a onto stack
    POP(u8),     // pop stack into register a
    CONST(Value, u8), // load const Value into register a
    CAST(u8, Type), // cast register a to type T
                 // JUMP(usize),
                 // JUMPIF(u8, usize), // conditionally jump based on register ri to address ai
}

pub struct Machine {
    pub registers: [Value; 8], //each value is 128 bits. 8 values is thus 1kb
    pub stack: Vec<Value>,     // consider replacing with smallvec to limit stack depth.
    pub instructions: Vec<OP>,
}

impl Machine {
    pub fn new(instructions: Vec<OP>, stack: Vec<Value>) -> Self {
        Machine {
            instructions,
            stack,
            registers: [Value::uint(0); 8],
        }
    }
    pub fn execute(&mut self) -> Result<(), VMError> {
        for instruction in &self.instructions {
            match &instruction {
                OP::ADD => {
                    self.registers[0] += self.registers[1];
                }
                OP::SUB => {
                    // self.registers[0] -= self.registers[1];
                }
                OP::MUL => todo!(),
                OP::DIV => todo!(),
                OP::POWI => todo!(),
                OP::POWF => todo!(),
                OP::MOD => todo!(),
                OP::SQRT => todo!(),
                OP::INV => todo!(),
                OP::FMA => todo!(),
                OP::MOV(_, _) => todo!(),

                OP::SWAP(_, _) => todo!(),
                OP::PUSH(_) => todo!(),
                OP::POP(_) => todo!(),
                OP::CONST(_, _) => todo!(),
                OP::CAST(_, _) => todo!(),
            }
        }
        Ok(())
    }
}

fn main() {
    let instructions = vec![];

    let mut machine = Machine::new(instructions, vec![]);
    let res = machine.execute();
}
