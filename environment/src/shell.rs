use std::fs;
use std::io;
use std::process::{Command, ExitStatus};

use rustyline::error::ReadlineError;
use rustyline::Editor;
use tempfile::NamedTempFile;

use peano::universe::{Universe, Context};

// Program used to open images (used to visualize the underlying egraph).
#[cfg(target_os = "linux")]
const OPEN_IMAGE: &'static str = "display";

#[cfg(target_os = "macos")]
const OPEN_IMAGE: &'static str = "open";

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
const OPEN_IMAGE: &'static str = "";


pub struct Shell {
    universe: Universe,
}

impl Shell {
    pub fn new() -> Shell {
        Shell { universe: Universe::new() }
    }

    pub fn load_path(&mut self, path: &str) -> Result<usize, String> {
        match fs::read_to_string(path) {
            Err(err) => {
                Err(err.to_string())
            },
            Ok(content) => {
                match content.parse::<Context>() {
                    Err(err) => {
                        Err(err.to_string())
                    },
                    Ok(ctx) => {
                        self.universe.incorporate(&ctx);
                        Ok(ctx.definitions.len())
                    }
                }
            }
        }
    }

    pub fn visualize_egraph(&mut self) -> Result<ExitStatus, io::Error> {
        let file = NamedTempFile::new()?;
        let path = file.into_temp_path();
        self.universe.to_png(&path)?;

        Command::new(OPEN_IMAGE)
            .args([&path])
            .status()
    }

    pub fn repl(&mut self) {
        let mut rl = Editor::<()>::new();

        loop {
            match rl.readline("> ") {
                Ok(line) => {
                    if line.starts_with("!") {
                        let line = line[1..].trim();
                        let (command, args) = match line.split_once(" ") {
                            Some((c, a)) => (c, a),
                            None => (line, ""),
                        };

                        if command == "context" {
                            println!("{}", self.universe.dump_context());
                        } else if command == "egraph" {
                            if let Err(err) = self.visualize_egraph() {
                                println!("Error visualizing e-graph: {}", err);
                            }
                        } else if command == "load" {
                            match self.load_path(args) {
                                Ok(n) => println!("{} definitions loaded.", n),
                                Err(err) => println!("Error loading {}: {}", args, err)
                            }
                        } else if command == "apply" {
                            self.universe.apply(&args.to_string());
                            println!("{}", self.universe.dump_context());
                        } else if command == "actions" {
                            let actions: Vec<&String> = self.universe.actions().collect();
                            println!("{}", actions.iter().map(|x| x.as_str()).collect::<Vec<&str>>().join(" "));
                        } else {
                            println!("Unknown command {}", command);
                        }
                    }

                    rl.add_history_entry(line);
                },
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                    break;
                },
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }
    }
}
