use std::fs;
use std::io;
use std::process::{Command, ExitStatus};
use std::rc::Rc;
use colored::Colorize;

use rustyline::error::ReadlineError;
use rustyline::Editor;
use tempfile::NamedTempFile;
use num_rational::Rational64;

use peano::universe::{Universe, Derivation, Context, Term};

// Program used to open images (used to visualize the underlying egraph).
#[cfg(target_os = "linux")]
const OPEN_IMAGE: &'static str = "display";

#[cfg(target_os = "macos")]
const OPEN_IMAGE: &'static str = "open";

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
const OPEN_IMAGE: &'static str = "";


pub struct Shell {
    universe: Box<dyn Universe>,
}

impl Shell {
    pub fn new() -> Shell {
        // Can also replace Derivation by EGraphUniverse.
        Shell { universe: Box::new(Derivation::new()) }
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
                        Ok(ctx.insertion_order.len())
                    }
                }
            }
        }
    }

    pub fn verify_all(&self) -> Result<(), String> {
        let mut successes = 0;
        let mut failures = 0;
        for name in self.universe.verification_scripts() {
            if self.verify(name).is_ok() {
                successes += 1;
            } else {
                failures += 1;
            }
        }

        let total = successes + failures;
        if total == 0 {
            println!("{}", "No verifications found.".yellow());
        } else {
            println!("\nVerified {} derivation(s){}{}.",
                     total,
                     if successes > 0 { format!(", {}", format!("{} succeeded", successes).green()) }
                     else { format!("") },
                     if failures > 0 { format!(", {}", format!("{} failed", failures).red()) }
                     else { format!("") },
            );
        }

        if failures > 0 {
            return Err(String::from("At least one verification failed."));
        }
        Ok(())
    }

    pub fn verify(&self, script_name: &String) -> Result<(), String> {
        print!("Verifying {}... ", script_name);
        match self.universe.execute_verification_script(script_name) {
            Ok(_) => {
                println!("{}", "ok".green());
                Ok(())
            },
            Err(e) => {
                println!("{}: {}", "error".red(), e);
                Err(e.to_string())
            }
        }
    }

    pub fn visualize_egraph(&mut self) -> Result<ExitStatus, io::Error> {
        let file = NamedTempFile::new()?;
        let path = file.into_temp_path();
        self.universe.to_png(path.to_str().unwrap())?;

        Command::new(OPEN_IMAGE)
            .args([&path])
            .status()
    }

    pub fn is_inhabited(&self, type_str: &str) -> Result<Option<String>, String> {
        match type_str.parse::<Term>() {
            Err(e) => Err(e.to_string()),
            Ok(t) => Ok(self.universe.inhabited(&Rc::new(t))),
        }
    }

    pub fn value_of(&self, t: &str) -> Result<Option<Rational64>, String> {
        match t.parse::<Term>() {
            Err(e) => Err(e.to_string()),
            Ok(t) => Ok(self.universe.value_of(&Rc::new(t))),
        }
    }

    pub fn apply(&mut self, action: &str, rl: &mut Editor<()>) -> () {
        let defs = self.universe.application_results(&action.to_string());

        if defs.len() == 0 {
            println!("No results.");
        } else {
            for (i, def) in defs.iter().enumerate() {
                println!("{} - {} : {}",
                         i,
                         if let Some(v) = &def.value { v.to_string() } else { String::from("_") },
                         def.dtype);
            }

            match rl.readline("  Result to add: ") {
                Ok(line) => {
                    match line.parse::<usize>() {
                        Ok(idx) if idx < defs.len() => {
                            self.universe.define(String::from("r"), defs[idx].clone(), true);
                        },
                        _ => { println!("Nothing added.") }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn apply_explode(&mut self, args: &str) -> () {
        for a in args.split(" ") {
            self.universe.apply(&a.to_string());
        }
    }

    pub fn check(&mut self, args: &str) -> Result<Option<String>, String> {
        if let Some((term, actions)) = args.split_once(" by ") {
            match term.parse::<Term>() {
                Err(e) => { return Err(e.to_string()); },
                Ok(t) => {
                    for a in actions.split(" ") {
                        self.universe.apply(&a.to_string());
                    }

                    return Ok(self.universe.inhabited(&Rc::new(t)))
                },
            }
        }
        Err(String::from("Syntax: !check <type> by <action,action,...>"))
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
                        } else if command == "apply_explode" {
                            self.apply_explode(args);
                            println!("{}", self.universe.dump_context());
                        } else if command == "apply" {
                            self.apply(args, &mut rl);
                            println!("{}", self.universe.dump_context());
                        } else if command == "inhabited" {
                            match self.is_inhabited(&args) {
                                Err(err) => println!("Error: {}", err),
                                Ok(None) => println!("no"),
                                Ok(Some(witness)) => println!("yes ({})", witness),
                            }
                        } else if command == "value" {
                            match self.value_of(&args) {
                                Err(err) => println!("Error: {}", err),
                                Ok(None) => println!("unknown"),
                                Ok(Some(value)) => println!("{} = {}", args, value),
                            }
                        } else if command == "check" {
                            match self.check(&args) {
                                Err(err) => println!("Error: {}", err),
                                Ok(None) => println!("no"),
                                Ok(Some(witness)) => println!("yes ({})", witness),
                            }
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
