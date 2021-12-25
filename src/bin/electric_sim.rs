use linefeed::{Interface, ReadResult};
use std::error::Error;

trait Component {
}


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
    let mut connections: Vec<ConnectorNode> = Vec::new();

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
