extern crate pest;
extern crate pest_derive;

pub mod shell;

fn main() {
    println!("Peano");

    let mut shell = shell::Shell::new();
    shell.repl();

    println!("Bye!");
}
