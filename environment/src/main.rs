use std::env;

extern crate pest;
extern crate pest_derive;

pub mod shell;

fn main() -> Result<(), String>{
    let args: Vec<String> = env::args().collect();
    let mut shell = shell::Shell::new();

    match args.len() {
        1 => {
            println!("Peano interactive shell");
            shell.repl();
            println!("Bye!");
        },
        2 | 3 => {
            println!("Loading {}", args[1]);
            if let Err(e) = shell.load_path(args[1].as_str()) {
                println!("Error: {}", e);
                return Err(format!("Error loading {}", args[1]));
            }
            if args.len() == 2 {
                return shell.verify_all();
            }
        },
        _ => {
            println!("Usage:");
            println!("{:30} -- runs interactive shell", args[0]);
            println!("{:30} -- loads and fully verifies the given file",
                     format!("{} <file>", args[0]));
            println!("{:30} -- loads and verifies one derivation from the given file",
                     format!("{} <file> <id>", args[0]));
            return Err(String::from("Wrong number of arguments"));
        }
    }

    Ok(())
}
