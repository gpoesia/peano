use std::option::Option;
use std::rc::Rc;
use std::collections::hash_set::{HashSet, Iter};
use std::collections::hash_map::HashMap;
use std::io;
use std::path::Path;
use std::iter::once;

use egg::*;
use num_rational::Rational64;

use super::universe::Universe;
use super::term::{Context, Term, Definition, is_parameter_name, VerificationNames};
use super::equivalence::AbstractSExp;
use super::verifier::{VerificationScript, VerificationError};

const IS_NODE: &str = &"$is";
const PARAM_NODE: &str = &"$param";
const ARROW_NODE: &str = &"$arrow";
const APPLY_NODE: &str = &"$app";
const LAMBDA_NODE: &str = &"$lambda";
const REAL_TYPE_CONST: &str = &"real";

#[derive(Clone)]
pub struct Derivation {
    pub context_: Context,
    next_id: usize,
    inhabited_types: HashSet<Rc<Term>>,
    existing_values: HashSet<Rc<Term>>,
}

// Returns whether the string contains a valid `real` constant.
// For now, we only handle integers in built-in operations.
// In the future, we'll have to decide what do built-in reals really mean.
fn is_real_const(s: &str) -> bool {
    s.parse::<Rational64>().is_ok()
}

impl Derivation {
    pub fn new() -> Derivation {
        Derivation {
            context_: Context::new_with_builtins(&["eval", "rewrite", "eq_symm", "eq_refl"]),
            next_id: 0,
            inhabited_types: HashSet::new(),
            existing_values: HashSet::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.context_.insertion_order.len()
    }

    fn next_term_id(&mut self) -> usize {
        self.next_id += 1;
        self.next_id
    }

    fn apply_builtin_eval(&self, new_terms: &mut Vec<Definition>) {
        for eq_name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(eq_name).unwrap();
            if let Some(val) = &def.value {
                if let Term::Application { function, arguments } = val.as_ref() {
                    if let Term::Atom { name } = &function.as_ref() {
                        if arguments.len() == 2 && (name == "+" || name == "-" ||
                                                    name == "/" || name == "*") {
                            let lhs = arguments[0].to_string().parse();
                            let rhs = arguments[1].to_string().parse();

                            if let (Ok(n1), Ok(n2)) = (lhs, rhs) {
                                if let Some(result) = apply_builtin_binary_op(n1, n2, name) {
                                    let eq_type = Rc::new(Term::Application {
                                        function: Rc::new(Term::Atom { name: String::from("=") }),
                                        arguments: vec![
                                            val.clone(),
                                            Rc::new(Term::Atom { name: result.to_string() }),
                                        ]
                                    });

                                    new_terms.push(Definition {
                                        dtype: eq_type,
                                        value: Some(Rc::new(Term::Application {
                                            function: Rc::new(Term::Atom { name: String::from("eval") }),
                                            arguments: vec![Rc::new(Term::Atom { name: eq_name.clone() })],
                                        }))
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_builtin_eq_refl(&self, new_terms: &mut Vec<Definition>) {
        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(name).unwrap();

            if def.is_prop(&self.context_) {
                continue;
            }

            let val = match &def.value {
                Some(v) => v.clone(),
                None => Rc::new(Term::Atom { name: name.clone() })
            };

            let eq_type = Rc::new(Term::Application {
                function: Rc::new(Term::Atom { name: String::from("=") }),
                arguments: vec![val.clone(), val]
            });

            new_terms.push(Definition {
                dtype: eq_type,
                value: Some(Rc::new(Term::Application {
                    function: Rc::new(Term::Atom { name: String::from("eq_refl") }),
                    arguments: vec![Rc::new(Term::Atom { name: name.clone() })],
                }))
            });
        }
    }

    fn apply_builtin_eq_symm(&self, new_terms: &mut Vec<Definition>) {
        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(name).unwrap();

            if let Some((t1, t2)) = def.dtype.extract_equality() {
                let eq_type = Rc::new(Term::Application {
                    function: Rc::new(Term::Atom { name: String::from("=") }),
                    arguments: vec![t2, t1]
                });
                new_terms.push(Definition {
                    dtype: eq_type,
                    value: Some(Rc::new(Term::Application {
                        function: Rc::new(Term::Atom { name: String::from("eq_symm") }),
                        arguments: vec![Rc::new(Term::Atom { name: name.clone() })],
                    }))
                });
            }
        }
    }

    fn apply_builtin_rewrite(&self, new_terms: &mut Vec<Definition>) {
        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(name).unwrap();

            if let Some((t1, t2)) = def.dtype.extract_equality() {
                for prop_name in self.context_.insertion_order.iter() {
                    let prop_def = self.context_.lookup(prop_name).unwrap();

                    if prop_def.is_prop(&self.context_) {
                        for (i, rw) in rewrite_all(&prop_def.dtype, &t1, &t2).into_iter().enumerate() {
                            new_terms.push(
                                Definition {
                                    dtype: rw,
                                    value: Some(Rc::new(Term::Application {
                                        function: Rc::new(Term::Atom { name: String::from("rewrite") }),
                                        arguments: vec![
                                            Rc::new(Term::Atom { name: name.clone() }),
                                            Rc::new(Term::Atom { name: prop_name.clone() }),
                                            Rc::new(Term::Atom { name: i.to_string() }),
                                        ],
                                    }))
                                }
                            );
                        }
                    }
                }
            }
        }
    }

    fn nondeterministically_apply_arrow(&self,
                                        arrow_object: &Rc<Term>,
                                        input_types: &Vec<Rc<Term>>,
                                        inputs: &mut Vec<Rc<Term>>,
                                        results: &mut Vec<Rc<Term>>) {
        let next = inputs.len();

        // If we have filled up all necessary arguments.
        if next == input_types.len() {
            results.push(Rc::new(
                Term::Application {
                    function: arrow_object.clone(),
                    arguments: inputs.clone(),
                }).eval(&self.context_));
            return;
        }

        // Otherwise, we pick the next argument.
        let param_type: Rc<Term> = match input_types[next].as_ref() {
            Term::Declaration { name: _, dtype } => dtype.clone(),
            _ => input_types[next].clone(),
        };

        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(&name).unwrap();

            // If types exactly match.
            if def.dtype == param_type {
                let val = if let Some(val) = &def.value {
                    val.clone()
                } else {
                    Term::Atom { name: name.clone() }.rc()
                };
                inputs.push(val);

                match input_types[next].as_ref() {
                    Term::Declaration { name: input_type_name, dtype: _ } => {
                        // If this is a declaration, then other input types might depend on the
                        // value of this argument. Substitute in them and proceed.
                        let mut remaining_types : Vec<Rc<Term>> = input_types.clone();
                        for i in inputs.len()..input_types.len() {
                            remaining_types[i] = remaining_types[i].replace(&input_type_name,
                                                                            &inputs[inputs.len() - 1]);
                        }
                        self.nondeterministically_apply_arrow(
                            &arrow_object,
                            &remaining_types,
                            inputs,
                            results,
                        );
                    },
                    _ => {
                        // Otherwise, either this is not a dependent arrow or it is but no type
                        // depends on this argument specifically. In any case, just continue.
                        self.nondeterministically_apply_arrow(
                            &arrow_object,
                            &input_types,
                            inputs,
                            results,
                        );
                    }
                }

                inputs.pop();
            }
        }
    }

    // Returns whether there already exists an object in the context with the same value
    // (or same type, in case the type is a prop, because of proof irrelevance).
    fn is_redundant(&self, dtype: &Rc<Term>, value: &Option<Rc<Term>>) -> bool {
        if dtype.is_prop(&self.context_) {
            // Proof irrelevance - check for anything with this same type.
            self.inhabited_types.contains(dtype)
        } else {
            match value {
                Some(val) => self.existing_values.contains(val),
                None => false,
            }
        }
    }

    fn define_subterms(&mut self, t: &Rc<Term>, is_root: bool) {
        let is_atom = match t.as_ref() {
            Term::Atom { name } => {
                let real_type = Rc::new(Term::Atom { name: String::from(REAL_TYPE_CONST) });
                if is_real_const(name) && !self.is_redundant(&real_type, &Some(t.clone())) {
                    self.context_.define(name.clone(),
                                         Definition { dtype: real_type,
                                                      value: None });
                    self.existing_values.insert(t.clone());
                }
                true
            },
            Term::Application { function, arguments } => {
                self.define_subterms(&function, false);

                for t in arguments.iter() {
                    self.define_subterms(&t, false);
                }

                false
            },
            _ => false,
        };

        if !is_root && !is_atom && !self.existing_values.contains(t) {
            let dname = format!("!sub{}", self.next_term_id());
            self.context_.define(dname,
                                 Definition { dtype: t.get_type(&self.context_),
                                              value: Some(t.clone()) });
        }
    }
}

impl Universe for Derivation {
    fn define(&mut self, name: String, def: Definition, rename: bool) {
        let name = if rename {
            format!("{}{}", name, self.next_term_id())
        } else {
            name
        };

        if cfg!(debug_assertions) {
            println!("define {} : {}{}", name, &def.dtype, match &def.value {
                Some(v) => format!(" = {}", v),
                None => format!("")
            });
        }

        self.define_subterms(&def.dtype, true);

        if let Some(value) = &def.value {
            self.define_subterms(&value, true);
            self.existing_values.insert(value.clone());
        }

        self.inhabited_types.insert(def.dtype.clone());
        self.context_.define(name, def);
    }

    fn rebuild(&mut self) {}

    fn incorporate_definitions(&mut self, defs: &Vec<Definition>, name_prefix: &str) {
        for d in defs.iter() {
            self.define(name_prefix.to_string(), d.clone(), true);
        }

        self.rebuild();
    }

    // Applies an action and checks whether it constructs the target object.
    // If it does, adds that object to the universe, returning Ok.
    // Otherwise, returns an Err with all the objects that were constructed by the action.
    fn construct_by(&mut self, action: &String, target: &Term) -> Result<(), Vec<Definition>> {
        let results = self.application_results(action);

        for def in results.iter() {
            if let Some(value) = &def.value {
                if target == value.as_ref() {
                    self.define(format!("r_{}_", action), def.clone(), true);
                    self.rebuild();
                    return Ok(())
                }
            }
        }

        Err(results)
    }

    // Applies an action and checks whether it constructs an object of the target type (typically a prop).
    // If it does, adds that object to the universe, returning Ok.
    // Otherwise, returns an Err with all the objects that were constructed by the action.
    fn show_by(&mut self, action: &String, target_type: &Term) -> Result<(), Vec<Definition>> {
        let results = self.application_results(action);

        for def in results.iter() {
            if target_type == def.dtype.as_ref() {
                self.define(format!("r_{}_", action), def.clone(), true);
                self.rebuild();
                return Ok(())
            }
        }

        Err(results)
    }

    // Applies an action with all possible distinct arguments.
    // Returns a vector with all produced results.
    fn application_results(&self, action: &String) -> Vec<Definition> {
        let mut new_terms = Vec::new();

        match self.context_.lookup(action) {
            None => {
                match action.as_str() {
                    "eval" => { self.apply_builtin_eval(&mut new_terms); }
                    "eq_refl" => { self.apply_builtin_eq_refl(&mut new_terms); }
                    "eq_symm" => { self.apply_builtin_eq_symm(&mut new_terms); }
                    "rewrite" => { self.apply_builtin_rewrite(&mut new_terms); }
                    _ => {}
                }
            },
            Some(action_def) => {
                match action_def.dtype.as_ref() {
                    Term::Arrow { input_types, output_type: _ } => {
                        let mut results = Vec::new();
                        self.nondeterministically_apply_arrow(
                            &Rc::new(Term::Atom { name: action.clone() }),
                            input_types,
                            &mut Vec::new(),
                            &mut results
                        );

                        for r in results.into_iter() {
                            new_terms.push(Definition { dtype: r.get_type(&self.context_),
                                                        value: Some(r) });
                        }
                    },
                    _ => {}
                }
            },
        }

        new_terms
    }

    // Checks if a type is inhabited in the current context, i.e., if we have any object
    // of that type. For proof objects, this amounts to knowing whether we have a proof
    // of the corresponding proposition. If so, returns the name of an object of the given type.
    fn inhabited(&self, term_type: &Rc<Term>) -> Option<String> {
        for name in self.context_.insertion_order.iter() {
            if let Some(def) = self.context_.lookup(&name) {
                if &def.dtype == term_type {
                    return Some(name.clone())
                }
            }
        }
        None
    }

    fn value_of(&self, _t: &Rc<Term>) -> Option<Rational64> {
        panic!()
    }

    fn context_summary(&self) -> Vec<(Vec<String>, String)> {
        panic!()
    }

    fn context(&self) -> &Context {
        &self.context_
    }

    fn context_mut(&mut self) -> &mut Context {
        &mut self.context_
    }

    fn to_png(&self, _path: &str) -> Result<(), io::Error> {
        Err(io::Error::from(io::ErrorKind::Other))
    }

    fn box_clone(&self) -> Box<dyn Universe> {
        Box::new(self.clone())
    }
}

fn apply_builtin_binary_op(n1: Rational64, n2: Rational64, op: &str) -> Option<Rational64> {
    match op {
        "+" => { Some(n1 + n2) },
        "-" => { Some(n1 - n2) },
        "*" => { Some(n1 * n2) },
        "/" if *n2.numer() != 0 => { Some(n1 / n2) },
        "/" => None,
        _ => panic!("Unknown builtin binary operator {}", op)
    }
}

// NOTE: This is a potentially conservative rewrite since we don't want to introduce occurrences
// of bound variables, so here we don't recurse into arrows or lambdas. This is not limiting for
// now, but something to think about for the future.
fn rewrite_all(term: &Rc<Term>, source: &Rc<Term>, target: &Rc<Term>) -> Vec<Rc<Term>> {
    let mut results = Vec::new();

    if term == source {
        results.push(target.clone());
    }

    match term.as_ref() {
        Term::Application { function, arguments } => {
            for alt in rewrite_all(function, source, target).into_iter() {
                results.push(Rc::new(Term::Application {
                    function: alt,
                    arguments: arguments.clone(),
                }));
            }

            for i in 0..arguments.len() {
                for alt in rewrite_all(&arguments[i], source, target).iter() {
                    results.push(Rc::new(Term::Application {
                        function: function.clone(),
                        arguments: arguments[..i].iter()
                                                 .chain(once(alt))
                                                 .chain(arguments[i+1..].iter())
                                                 .map(|t| t.clone())
                                                 .collect()
                    }));
                }
            }
        },
        _ => {}
    }

    results
}


#[cfg(test)]
pub mod tests {
    use crate::universe::{Universe, Derivation, Context, Term, Definition};
    use std::rc::Rc;
    use num_rational::Rational64;

    #[test]
    fn test_create_derivation() {
        let mut u = Derivation::new();

        let a = u.actions().collect::<Vec<&String>>();

        // Should have 'eval' and equality axioms.
        assert_eq!(a.len(), 4);

        u.define("nat".to_string(), Definition::new_opaque(u.get_type_constant().clone()), false);
        u.define("z".to_string(), Definition::new_opaque(Rc::new("nat".parse().unwrap())), false);
        u.define("s".to_string(), Definition::new_opaque(Rc::new("[nat -> nat]".parse().unwrap())), false);

        let a = u.actions().collect::<Vec<&String>>();
        assert_eq!(a.len(), 5);

        u.define("ss".to_string(), Definition::new_concrete(Rc::new("[nat -> nat]".parse().unwrap()),
                                                            Rc::new("lambda (n : nat) (s (s n))"
                                                                    .parse().unwrap())),
                 false);
        let a = u.actions().collect::<Vec<&String>>();
        assert_eq!(a.len(), 6);
    }

    #[test]
    fn test_simple_transitivity_proof() {
        let nat_theory: Context = "
        nat : type.
        z : nat.
        s : [nat -> nat].

        leq : [nat -> nat -> prop].

        leq_n_n : [(n : nat) -> (leq n n)].
        leq_s : [(n : nat) -> (m : nat) -> (leq n m) -> (leq n (s m))].
        "
        .parse()
        .unwrap();
        let mut u = Derivation::new();
        u.incorporate(&nat_theory);

        assert!(u.construct_by(&"s".to_string(), &"(s z)".parse().unwrap()).is_ok());
        assert!(u.construct_by(&"s".to_string(), &"(s (s z))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"leq_n_n".to_string(), &"(leq z z)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"leq_s".to_string(), &"(leq z (s z))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"leq_s".to_string(), &"(leq z (s (s z)))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"leq_s".to_string(), &"(leq z (s (s (s z))))".parse().unwrap()).is_ok());
    }

    #[test]
    fn test_eval() {
        let real_theory: Context = "
        real : type.
        + : [real -> real -> real].
        * : [real -> real -> real].
        a : real = (+ 2 2).
        b : real = (* 2/3 9).
        "
        .parse()
        .unwrap();
        let mut u = Derivation::new();
        u.incorporate(&real_theory);

        assert!(u.show_by(&"eval".to_string(), &"(= (+ 2 2) 4)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"eval".to_string(), &"(= (* 2/3 9) 6)".parse().unwrap()).is_ok());
    }

    #[test]
    fn test_equation_solution() {
        let real_theory: Context = "
        real : type.

        = : [(t : type) -> t -> t -> type].
        + : [real -> real -> real].
        - : [real -> real -> real].
        * : [real -> real -> real].

        +_comm : [(a : real) -> (b : real) -> (= (+ a b) (+ b a))].
        +-_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (- (+ a b) c) (+ a (- b c)))].
        +0_id : [(a : real) -> (= (+ a 0) a)].

        x : real.
        y : real.

        x_eq : (= (+ x 3) 10).
        y_eq : (= y (* x x)).
        "
        .parse()
        .unwrap();

        let mut u = Derivation::new();
        u.incorporate(&real_theory);

        assert!(u.construct_by(&"-".to_string(), &"(- (+ x 3) 3)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"+-_assoc".to_string(), &"(= (- (+ x 3) 3) (+ x (- 3 3)))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"eval".to_string(), &"(= (- 3 3) 0)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= (- (+ x 3) 3) (+ x 0))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"+0_id".to_string(), &"(= (+ x 0) x)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= (- (+ x 3) 3) x)".parse().unwrap()).is_ok());
        assert!(u.construct_by(&"-".to_string(), &"(- 10 3)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"eq_refl".to_string(), &"(= (- (+ x 3) 3) (- (+ x 3) 3))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= x (- (+ x 3) 3))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= x (- 10 3))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"eval".to_string(), &"(= (- 10 3) 7)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= x 7)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= y (* 7 x))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= y (* 7 7))".parse().unwrap()).is_ok());
        assert!(u.show_by(&"eval".to_string(), &"(= (* 7 7) 49)".parse().unwrap()).is_ok());
        assert!(u.show_by(&"rewrite".to_string(), &"(= y 49)".parse().unwrap()).is_ok());
    }
}
