use std::option::Option;
use std::rc::Rc;
use std::collections::hash_set::Iter;
use std::io;
use std::path::Path;

use egg::*;
use super::term::{Context, Term, Definition};

const OPAQUE_NODE : &str = &"$opaque";
const IS_NODE: &str = &"$is";
const DECL_NODE: &str = &"$decl";
const ARROW_NODE: &str = &"$arrow";
const APPLY_NODE: &str = &"$app";
const LAMBDA_NODE: &str = &"$lambda";

pub struct Universe {
    egraph: EGraph<SymbolLang, ()>,
    context: Context,
    id_counter: u32,
}

impl Universe {
    pub fn new() -> Universe {
        let mut u = Universe {
            egraph: Default::default(),
            id_counter: 0,
            context: Context::new()
        };

        for s in [OPAQUE_NODE, IS_NODE, DECL_NODE] {
            u.egraph.add(SymbolLang::leaf(s));
        }

        u
    }

    pub fn size(&self) -> usize {
        self.egraph.number_of_classes()
    }

    pub fn get_type_constant(&self) -> &Rc<Term> {
        self.context.get_type_constant()
    }

    pub fn define(&mut self, name: String, def: Definition, rename: bool) -> Id {
        let name = if rename {
            self.id_counter += 1;
            format!("{}{}", name, self.id_counter)
        } else {
            name
        };

        self.context.define(name.clone(), def.clone());

        let decl = Rc::new(Term::Declaration { name: name, dtype: def.dtype });

        let decl_id = self.add_term(&decl);

        if let Some(value) = def.value {
            let value_id = self.add_term(&value);
            self.egraph.union(decl_id, value_id);
        }

        self.egraph.rebuild();

        decl_id
    }

    pub fn incorporate(&mut self, context: &Context) {
        // First, add all definitions to self.context.
        for (name, defs) in context.definitions.iter() {
            self.context.define(name.clone(), defs[defs.len() - 1].clone());
        }
        // Then, self.define on each new definition. This duplicates some work, but avoids
        // the problem that we don't know the right order to add definitions in.
        for (name, defs) in context.definitions.iter() {
            self.define(name.clone(), defs[defs.len() - 1].clone(), false);
        }
    }

    fn add_term(&mut self, t: &Rc<Term>) -> Id {
        match t.as_ref() {
            Term::Declaration{ name, dtype } => {
                let type_id = self.add_term(dtype);
                let name_id = self.egraph.add(SymbolLang::leaf(&name));
                self.egraph.add(SymbolLang::new(IS_NODE, vec![name_id, type_id]));

                name_id
            },
            Term::Atom { name } => {
                self.egraph.add(SymbolLang::leaf(name))
            },
            Term::Arrow { input_types, output_type } => {
                let mut v = Vec::new();
                for t in input_types.iter() {
                    v.push(self.add_term(&t));
                }
                v.push(self.add_term(&output_type));
                self.egraph.add(SymbolLang::new(ARROW_NODE, v))
            },
            Term::Lambda { parameters, body } => {
                let mut v = Vec::new();
                for t in parameters.iter() {
                    v.push(self.add_term(&t));
                }
                v.push(self.add_term(&body));
                self.egraph.add(SymbolLang::new(LAMBDA_NODE, v))
            },
            Term::Application { function, arguments } => {
                let type_expr = &t.get_type(&self.context);
                let type_id = self.add_term(type_expr);

                let mut v = Vec::new();
                v.push(self.add_term(&function));

                for t in arguments.iter() {
                    v.push(self.add_term(&t));
                }

                let app_id = self.egraph.add(SymbolLang::new(APPLY_NODE, v));
                self.egraph.add(SymbolLang::new(IS_NODE, vec![app_id, type_id]));

                app_id
            }
        }
    }

    pub fn actions(&self) -> Iter<'_, String> {
        self.context.actions()
    }

    pub fn apply(&mut self, action: &String) -> Vec<Rc<Term>> {
        let mut new_terms = Vec::new();

        match self.context.lookup(action) {
            None => new_terms,
            Some(action_def) => {
                match action_def.dtype.as_ref() {
                    Term::Arrow { input_types, output_type: _ } => {
                        self.nondeterministically_apply_arrow(
                            &Rc::new(Term::Atom { name: action.clone() }),
                            input_types,
                            &mut Vec::new(),
                            &mut new_terms
                            );

                        for t in new_terms.iter() {
                            self.define("obj".to_string(),
                                        Definition { dtype: t.get_type(&self.context),
                                                     value: Some(t.clone()) }, true);
                        }

                        new_terms
                    },
                    _ => {
                        new_terms
                    }
                }
            },
        }
    }

    pub fn to_png(&self, path: impl AsRef<Path>) -> Result<(), io::Error> {
        self.egraph.dot().to_png(path)
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
                }).eval(&self.context));
            return;
        }

        // Otherwise, we pick the next argument.
        let param_type: Rc<Term> = match input_types[next].as_ref() {
            Term::Declaration { name: _, dtype } => dtype.clone(),
            _ => input_types[next].clone(),
        };

        // Find all objects that match the desired type.
        // FIXME this can be optimized in a number of ways (e.g., lookup table if type is atomic).
        for (name, defs) in self.context.definitions.iter() {
            let last_def = &defs[defs.len() - 1];
            // FIXME we likely need to be careful here about what notion of equality to apply here.
            // For now we just conservatively check if their term structure is identical.
            if param_type == last_def.dtype {
                // Can use this as an argument.
                let val = Rc::new(Term::Atom { name: name.clone() }).eval(&self.context);
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

    pub fn dump_context(&self) -> String {
        self.context.to_string()
    }
}

pub mod tests {
    use crate::universe::Universe;
    use crate::universe::{Context, Term, Definition};
    use std::rc::Rc;

    #[test]
    fn test_create_universe() {
        let mut u = Universe::new();

        let a = u.actions().collect::<Vec<&String>>();
        assert_eq!(a.len(), 0);

        u.define("nat".to_string(), Definition::new_opaque(u.get_type_constant().clone()), false);
        u.define("z".to_string(), Definition::new_opaque(Rc::new("nat".parse().unwrap())), false);
        u.define("s".to_string(), Definition::new_opaque(Rc::new("[nat -> nat]".parse().unwrap())), false);

        let a = u.actions().collect::<Vec<&String>>();
        assert_eq!(a.len(), 1);

        u.define("ss".to_string(), Definition::new_concrete(Rc::new("[nat -> nat]".parse().unwrap()),
                                                            Rc::new("lambda (n : nat) (s (s n))"
                                                                    .parse().unwrap())),
                 false);
        let a = u.actions().collect::<Vec<&String>>();
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_simple_actions() {
        let mut u = Universe::new();

        let a = u.actions().collect::<Vec<&String>>();
        assert_eq!(a.len(), 0);

        u.define("nat".to_string(), Definition::new_opaque(u.get_type_constant().clone()), false);
        u.define("z".to_string(), Definition::new_opaque(Rc::new("nat".parse().unwrap())), false);
        u.define("s".to_string(), Definition::new_opaque(Rc::new("[nat -> nat]".parse().unwrap())), false);

        u.dot("s0.png");

        u.apply(&"s".to_string());

        u.dot("s1.png");

        u.apply(&"s".to_string());

        u.dot("s2.png");

        u.apply(&"s".to_string());

        u.dot("s3.png");

        u.apply(&"s".to_string());

        u.dot("s4.png");

        u.apply(&"s".to_string());

        u.dot("s5.png");

        assert_eq!(true, false);
    }
}
