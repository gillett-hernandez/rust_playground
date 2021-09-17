use linefeed::{Interface, ReadResult};
use std::error::Error;
trait Generator {
    // fn voltage(&self) -> f32;
}

trait Consumer {
    // fn resistance(&self) -> f32;
}

struct CoalGenerator {}

impl Generator for CoalGenerator {
    // fn voltage(&self) -> f32 {

    // }
}
struct BasicConsumer {}

impl Consumer for BasicConsumer {}

enum Connection {
    Generator(usize),
    Consumer(usize),
    // Connector(usize),
}

struct ConnectorNode {
    // pub id: usize,
    pub connections: Vec<Connection>,
}

// struct

fn main() -> Result<(), Box<dyn Error>> {
    let mut generators: Vec<Box<dyn Generator>> = Vec::new();
    let mut consumers: Vec<Box<dyn Consumer>> = Vec::new();
    let mut connections: Vec<ConnectorNode> = Vec::new();

    generators.push(Box::new(CoalGenerator {}));
    consumers.push(Box::new(BasicConsumer {}));
    connections.push(ConnectorNode {
        connections: vec![Connection::Generator(0), Connection::Consumer(0)],
    });

    // simulation loop

    // let mut equations = Vec::new();

    let reader = Interface::new("electric-network-sim")?;

    reader.set_prompt("ens> ")?;

    while let ReadResult::Input(input) = reader.read_line()? {
        println!("got input {:?}", input);
    }

    println!("Goodbye.");
    Ok(())
}
