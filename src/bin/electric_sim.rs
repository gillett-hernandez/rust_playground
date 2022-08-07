#![feature(drain_filter)]

use linefeed::{Interface, ReadResult};
use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;

#[derive(Debug)]
struct Connection([usize; 2]);

trait Node {
    fn id(&self) -> usize; //{
                           //                        // self.id
                           //                        // }
}

#[derive(Debug)]
struct Network<T> {
    pub nodes: HashMap<usize, T>,
    pub connections: Vec<Connection>,
}

impl<T> Network<T>
where
    T: Node,
{
    pub fn new() -> Self {
        Network {
            nodes: HashMap::new(),
            connections: vec![],
        }
    }

    pub fn add_node(&mut self, node: T) {
        if !self.nodes.contains_key(&node.id()) {
            self.nodes.insert(node.id(), node);
        }
    }

    pub fn update_node(&mut self, node: T) -> Option<T> {
        self.nodes.insert(node.id(), node)
    }

    pub fn add_connection(&mut self, connection: Connection) -> Result<(), ()> {
        if self.nodes.contains_key(&connection.0[0]) && self.nodes.contains_key(&connection.0[1]) {
            self.connections.push(connection);
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn connect_new_node_to(&mut self, id: usize, new_node: T) -> Result<(), usize> {
        let new_id = new_node.id();
        self.add_node(new_node);
        if let Ok(()) = self
            .add_connection(Connection([id, new_id]))
            .map_err(|e| id)
        {
            Ok(())
        } else {
            // remove node since connection failed
            self.remove_node_unchecked(new_id);
            Err(id)
        }
    }

    pub fn remove_node(&mut self, id: usize) -> Result<T, ()> {
        // iterate through connections and remove all that have a reference to this node
        if let Some(node) = self.nodes.remove(&id) {
            // successfully removed
            let _ = self
                .connections
                .drain_filter(|e| e.0[0] == id || e.0[1] == id)
                .collect::<Vec<_>>();
            Ok(node)
        } else {
            // not present
            Err(())
        }
    }

    fn remove_node_unchecked(&mut self, id: usize) {
        let _ = self.nodes.remove(&id);
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct Generator {
    id: usize,
}

impl Node for Generator {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct Consumer {
    id: usize,
}

impl Node for Consumer {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct Connector {
    id: usize,
}

impl Node for Connector {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct Buffer {
    id: usize,
}

impl Node for Buffer {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
enum NodeType {
    Connector(Connector),
    Generator(Generator),
    Consumer(Consumer),
    Buffer(Buffer),
}

impl Node for NodeType {
    fn id(&self) -> usize {
        match self {
            NodeType::Generator(inner) => inner.id,
            NodeType::Consumer(inner) => inner.id,
            NodeType::Connector(inner) => inner.id,
            NodeType::Buffer(inner) => inner.id,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // simulation setup

    let mut network: Network<NodeType> = Network::new();

    network.add_node(NodeType::Connector(Connector { id: 0 }));
    network.add_node(NodeType::Generator(Generator { id: 1 }));
    network.add_node(NodeType::Consumer(Consumer { id: 2 }));

    network.add_connection(Connection([0, 1]));
    network.add_connection(Connection([0, 2]));

    // command line interface
    let reader = Interface::new("electric-network-sim")?;

    reader.set_prompt("ens> ")?;

    while let ReadResult::Input(input) = reader.read_line()? {
        if input.starts_with("exit") {
            break;
        }
        if input.starts_with("print") {
            println!("{:?}", network);
        }
        println!("got input {:?}", input);
    }

    println!("Goodbye.");
    Ok(())
}
