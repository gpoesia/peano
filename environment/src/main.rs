extern crate pest;
#[macro_use]
extern crate pest_derive;

use std::io::{stdin, stdout, Write, Read};
use std::fs;
use peano::universe::*;
use peano::universe::term::*;
use peano::universe::universe::*;

fn main() {
    let mut universe = Universe::new();
    let mut stdout = stdout();
    let stdin = stdin();

    println!("Peano");

    loop {
        stdout.write_all("> ".as_bytes()).and_then(|_| { stdout.flush() }).unwrap();

        let mut line = String::new();

        match stdin.read_line(&mut line) {
            Err(err) => {
                println!("{:?}", err);
                break;
            },
            Ok(n_bytes) => {
                if n_bytes == 0 {
                    break;
                }

                if line.starts_with("!") {
                    let line = line[1..].trim();
                    let (command, args) = match line.split_once(" ") {
                        Some((c, a)) => (c, a),
                        None => (line, ""),
                    };

                    if command == "context" {
                        println!("{}", universe.dump_context());
                    } else if command == "egraph" {

                    } else if command == "load" {
                        let path = args;
                        match fs::read_to_string(args) {
                            Err(err) => {
                                println!("Error reading {}: {}", path, err);
                            },
                            Ok(content) => {
                                match content.parse::<Context>() {
                                    Err(err) => {
                                        println!("Error while parsing {}: {}", path, err);
                                    },
                                    Ok(ctx) => {
                                        universe.incorporate(&ctx);
                                        println!("{} definitions loaded.", ctx.definitions.len());
                                    }
                                }
                            }
                        }
                    } else if command == "apply" {
                        let action = args;
                        universe.apply(&action.to_string());
                        println!("{}", universe.dump_context());
                    } else {
                        println!("Unknown command {}", command);
                    }
                }
            }
        }
    }

    println!("Bye!");
}
