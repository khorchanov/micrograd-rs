use std::{
    collections::{HashMap, HashSet},
    fmt::format,
    fs::File,
    io::Write,
};

use graphviz_rust::{
    dot_generator::*,
    dot_structures::*,
    printer::{DotPrinter, PrinterContext},
};

use crate::{Operation, Value};

pub fn draw_dot(root: &Value, filename: &str) {
    let mut graph = graph!(strict di id!("computation_graph"));
    let mut visited = HashSet::new();
    let mut node_counter = 0;
    let mut node_ids = HashMap::new();

    fn get_node_id(
        v: &Value,
        node_ids: &mut HashMap<usize, String>,
        node_counter: &mut usize,
    ) -> String {
        let ptr = v.data.as_ptr() as usize;
        if let Some(id) = node_ids.get(&ptr) {
            id.clone()
        } else {
            let id = format!("n{}", node_counter);
            *node_counter += 1;
            node_ids.insert(ptr, id.clone());
            id
        }
    }

    fn trace(
        v: &Value,
        graph: &mut Graph,
        visited: &mut HashSet<usize>,
        node_ids: &mut HashMap<usize, String>,
        node_counter: &mut usize,
    ) {
        let ptr = v.data.as_ptr() as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let node_id = get_node_id(v, node_ids, node_counter);
        let label = if let Some(ref name) = v.label {
            format!(
                "\"{}|data={:.4}|grad={:.4}\"",
                name,
                v.data.borrow(),
                v.grad.borrow()
            )
        } else {
            format!(
                "\"data={:.4}|grad={:.4}\"",
                v.data.borrow(),
                v.grad.borrow()
            )
        };

        graph.add_stmt(Stmt::Node(
            node!(node_id.clone(); attr!("label", label), attr!("shape", "record")),
        ));

        if let Some(ref op) = v.op {
            let op_id = format!("{}_op", node_id);
            let (op_label, children) = match op {
                Operation::Add(a, b) => ("\"+\"", vec![a, b]),
                Operation::Mul(a, b) => (r#""x""#, vec![a, b]),
                Operation::Tanh(a) => ("\"tanh\"", vec![a]),
                Operation::Exp(a) => ("\"exp\"", vec![a]),
                Operation::Pow(a, k) => (r#""pow""#, vec![a, k]),
            };

            graph.add_stmt(Stmt::Node(
                node!(op_id.clone(); attr!("label", op_label), attr!("shape", "circle")),
            ));
            graph.add_stmt(Stmt::Edge(
                edge!(node_id!(op_id.clone()) => node_id!(node_id.clone())),
            ));

            for child in children {
                let child_borrowed = child.borrow();
                let child_id = get_node_id(&child_borrowed, node_ids, node_counter);
                graph.add_stmt(Stmt::Edge(
                    edge!(node_id!(child_id.clone()) => node_id!(op_id.clone())),
                ));
                trace(&child_borrowed, graph, visited, node_ids, node_counter);
            }
        }
    }

    trace(
        root,
        &mut graph,
        &mut visited,
        &mut node_ids,
        &mut node_counter,
    );

    // Generate DOT string
    let dot_string = graph.print(&mut PrinterContext::default());

    // Save DOT file
    let dot_filename = format!("{}.dot", filename);
    let mut file = File::create(&dot_filename).expect("Unable to create file");
    file.write_all(dot_string.as_bytes())
        .expect("Unable to write data");

    println!("✓ Graph saved to {}", dot_filename);
    println!("  View it at: https://dreampuf.github.io/GraphvizOnline/");
    println!(
        "  Or install Graphviz and run: dot -Tpng {} -o {}.png",
        dot_filename, filename
    );
}
