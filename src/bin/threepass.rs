// language syntax:
//
//
// function   ::= '[' arg-list ']' '{' expression '}'
//
// arg-list   ::= /* nothing */
//              | arg-list variable
//
// expression ::= term
//              | expression '+' term
//              | expression '-' term
//
// term       ::= factor
//              | term '*' factor
//              | term '/' factor
//
// factor     ::= number
//              | variable
//              | '(' expression ')'
// variable   ::= [a-zA-Z]+
// number     ::= [0-9]+
//
// numbers and variables are marked by the tokenize function
//
// asm opcodes:
// IM = immediate n
// AR = argument n
// SW = swap r0 and r1
// PU = push r0 to stack
// PO = pop value from stack to r0
// AD = add
// SU = subtract
// MU = multiply
// DI = divide
//
// example
// [ a b ] a*a + b*b
//
// {
//     'op':'+',
//     'a': {
//         'op':'*',
//         'a': {
//             'op':'arg',
//             'n': 'a'
//         },
//         'b': {
//             'op':'arg',
//             'n': 'a'
//         }
//     },
//     'b': {
//         'op':'*',
//         'a': {
//             'op':'arg',
//             'n': 'b'
//         },
//         'b': {
//             'op':'arg',
//             'n': 'b'
//         }
//     }
// }
//
// no optimizations, instructions = 23
// ['AR0', 'PU', 'AR0', 'PU', 'PO', 'SW', 'PO', 'MU', 'PU', 'AR1', 'PU', 'AR1', 'PU', 'PO' 'SW', 'PO', 'MU', 'PU', 'PO', 'SW', 'PO', 'AD', 'PU']
//
// 2 optimizations types, instructions = 17
// ['AR0', 'PU', 'AR0', 'SW', 'PO', 'MU', 'PU', 'AR1', 'PU', 'AR1', 'SW', 'PO', 'MU', 'SW', 'PO', 'AD', 'PU']
//
// 3 optimizations types, instructions = 15
// ['AR0', 'SW', 'AR0', 'SW', 'MU', 'PU', 'AR1', 'SW', 'AR1', 'SW', 'MU', 'SW', 'PO', 'AD', 'PU']
//
// hand written, instructions = 13
// ['AR0', 'SW', 'AR0', 'MU', 'PU', 'AR1', 'SW', 'AR1', 'MU', 'SW', 'PO', 'AD', 'PU']
//

use std::fmt::Display;

pub enum OP {
    Add,      // add r0 and r1 and push the result
    Subtract, // subtract r0 and r1 and push the result
    Multiply, // multiply r0 by r1 and push the result
    Divide,   // divide r0 by r1 and push the result
    Push,     // push the first register onto the stack
    Pop,      // pop the stack onto the first register
    Swap,     // swap the two values in the registers
    Load(u8), // push an argument from the stack to the Stack.
}

pub enum OpType {
    Binary,
    Unary,
    Immediate,
    Argument,
}

impl Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::Binary => f.write_str("OP::Binary"),
            OpType::Unary => f.write_str("OP::Unary"),
            OpType::Immediate => f.write_str("OP::Immediate"),
            OpType::Argument => f.write_str("OP::Argument"),
        }
    }
}

// trait ASTNode: Display {
//     fn token(&self) -> &str;
//     fn idx(&self) -> usize;
//     fn op(&self) -> OpType;
// }

trait Transform {
    fn transform(&self) -> Vec<OP>;
}

pub enum Token {
    LeftBracket,
    RightBracket,
    Identifier,
    Plus,
    Minus,
    Multiply,
    Divide,
    Exponent,
}

pub enum FactorNode {
    // factor     ::= number
    //              | variable
    //              | '(' expression ')'
    Literal,
    Variable(Token),
    Parenthesized(Box<ExpressionNode>),
}

pub enum TermNode {
    // term       ::= factor
    //              | term '*' factor
    //              | term '/' factor
    Factor(FactorNode),
    Multiply(Box<TermNode>, FactorNode),
    Divide(Box<TermNode>, FactorNode),
}

pub enum ExpressionNode {
    // expression ::= term
    //              | expression '+' term
    //              | expression '-' term
    TermNode,
    Add(Box<ExpressionNode>, TermNode),
    Subtract(Box<ExpressionNode>, TermNode),
}

pub struct FunctionNode {
    // function   ::= '[' arg-list ']' '{' expression '}'
    args: Vec<Token>,
    expression: ExpressionNode,
}

pub struct AST {
    root: FunctionNode,
}

impl Transform for FunctionNode {
    fn transform(&self) -> Vec<OP> {
        todo!()
    }
}

impl Transform for AST {
    fn transform(&self) -> Vec<OP> {
        self.root.transform()
    }
}

mod mini_compiler {
    use super::*;
    fn parse_factor(tokens: &mut [Token]) -> FactorNode {
        todo!()
    }
    fn parse_term(tokens: &mut [Token]) -> TermNode {
        todo!()
    }
    fn parse_arglist(tokens: &mut [Token]) -> Vec<Token> {
        todo!()
    }
    fn parse_expression(tokens: &mut [Token]) -> ExpressionNode {
        todo!()
    }
    fn parse_function(tokens: &mut [Token]) -> FunctionNode {
        FunctionNode {
            args: parse_arglist(tokens),
            expression: parse_expression(tokens),
        }
    }
    fn tokenize(input: String) -> Vec<Token> {
        vec![]
    }
    fn unoptimized_ast(mut tokens: &mut [Token]) -> AST {
        AST {
            root: parse_function(&mut tokens),
        }
    }
    fn reduced_ast(mut ast: AST) -> AST {
        ast
    }
    fn optimized_ast(mut ast: AST) -> AST {
        ast
    }

    pub fn compile(input: String) -> Vec<OP> {
        let mut tokens = tokenize(input);
        let ast = unoptimized_ast(&mut tokens);
        let ast = reduced_ast(ast);
        let ast = optimized_ast(ast);

        ast.transform()
    }
}

pub struct Machine<T> {
    registers: [T; 2],
    stack: Vec<T>,
}

impl<T> Machine<T>
where
    T: num::Zero,
{
    pub fn new() -> Self {
        Machine {
            registers: [T::zero(), T::zero()],
            stack: vec![],
        }
    }
    pub fn execute(&mut self, instruction: OP) {
        match instruction {
            OP::Add => todo!(),
            OP::Subtract => todo!(),
            OP::Multiply => todo!(),
            OP::Divide => todo!(),
            OP::Push => todo!(),
            OP::Pop => todo!(),
            OP::Swap => todo!(),
            OP::Load(_) => todo!(),
        }
    }
}

fn main() {
    let input = "";
    let compiled = mini_compiler::compile(input.to_string());
    let mut machine: Machine<f32> = Machine::new();

    for instruction in compiled {
        machine.execute(instruction);
    }
}
