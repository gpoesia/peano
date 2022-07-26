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
use super::term::{Context, Term, Definition, is_parameter_name, VerificationNames, Unifier};
use super::equivalence::AbstractSExp;
use super::verifier::{VerificationScript, VerificationError};

const REAL_TYPE_CONST: &str = &"real";

#[derive(Clone)]
pub struct Derivation {
    pub context_: Context,
    next_id: usize,
    inhabited_types: HashMap<Rc<Term>, Vec<String>>,
    existing_values: HashMap<Rc<Term>, String>,
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
            inhabited_types: HashMap::new(),
            existing_values: HashMap::new(),
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
            self.apply_builtin_eval_with(&eq_name, def, new_terms);
        }
    }

    fn apply_builtin_eq_refl(&self, new_terms: &mut Vec<Definition>) {
        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(name).unwrap();
            self.apply_builtin_eq_refl_with(name, def, new_terms);
        }
    }

    fn apply_builtin_eq_symm(&self, new_terms: &mut Vec<Definition>) {
        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(name).unwrap();
            self.apply_builtin_eq_symm_with(name, def, new_terms);
        }
    }

    fn apply_builtin_rewrite(&self, new_terms: &mut Vec<Definition>) {
        for name in self.context_.insertion_order.iter() {
            let def = self.context_.lookup(name).unwrap();

            if let Some((t1, t2)) = def.dtype.extract_equality() {
                if t1 != t2 {
                    // NOTE: This might be avoided if we maintain the invariant that
                    // defined terms are already in their canonical form.
                    let t1 = t1.eval(&self.context_);
                    let t2 = t2.eval(&self.context_);

                    for prop_name in self.context_.insertion_order.iter() {
                        let prop_def = self.context_.lookup(prop_name).unwrap();

                        if prop_def.is_prop(&self.context_) {
                            let prop_dtype = prop_def.dtype.eval(&self.context_);
                            self.rewrite_in(prop_name, &prop_dtype, &t1, &t2, name, new_terms);
                        }
                    }
                }
            }
        }
    }

    fn nondeterministically_apply_arrow(&self,
                                        arrow_object: &Rc<Term>,
                                        input_types: &Vec<Rc<Term>>,
                                        output_type: &Rc<Term>,
                                        inputs: &mut Vec<Rc<Term>>,
                                        predetermined: &Vec<Option<&String>>,
                                        results: &mut Vec<(Rc<Term>, Rc<Term>)>) {
        let next = inputs.len();

        // If we have filled up all necessary arguments.
        if next == input_types.len() {
            results.push((Rc::new(
                Term::Application {
                    function: arrow_object.clone(),
                    // Substitute the evaluated terms back for names if possible.
                    // This should always be possible when using Peano as an environment,
                    // since all sub-terms will always be named.
                    arguments: inputs.iter().map(|v| match self.existing_values.get(v) {
                        Some(name) => Term::Atom { name: name.clone() }.rc(),
                        None => v.clone()
                    }).collect()
                }),
                output_type.clone()));
            return;
        }

        // Otherwise, we pick the next argument.
        let (value_pattern, param_type) = match input_types[next].as_ref() {
            Term::PatternDeclaration { pattern, dtype } => (Some(pattern.clone()), dtype.clone()),
            _ => (None, input_types[next].clone()),
        };

        let param = predetermined.get(next).unwrap_or(&None).map(|name| vec![name.clone()]);

        for name in param.as_ref().unwrap_or(&self.context_.insertion_order) {
            let def = self.context_.lookup(&name).unwrap();
            let val = Term::Atom { name: name.clone() }.rc().eval(&self.context_);

            let mut unifier = Unifier::new();

            // If this parameter has a value_pattern, def must have a value and unify with it.
            if let Some(p) = &value_pattern {
                if !p.unify_params(&val, &mut unifier) {
                    continue;
                }
            }

            if param_type.unify_params(&def.dtype, &mut unifier) {
                // `val` can be used as this argument. Make the necessary substitutions to
                // other parameter types (if any, i.e., if this is a dependent arrow) and
                // keep going.
                inputs.push(val);

                let mut remaining_types : Vec<Rc<Term>> = input_types.clone();
                let mut output_type = output_type.clone();

                for (name, value) in unifier.iter() {
                    for i in inputs.len()..input_types.len() {
                        remaining_types[i] = remaining_types[i].replace(name, &value);
                    }
                    output_type = output_type.replace(name, &value);
                }

                self.nondeterministically_apply_arrow(
                            &arrow_object, &remaining_types, &output_type, inputs, predetermined, results);

                inputs.pop();
            }
        }
    }

    pub fn filter_equalities(&self, eq: Vec<Definition>) -> Vec<Definition> {
        return eq;
        /* eq.into_iter().filter(|def| {
            match def.dtype.extract_equality() {
                Some((t1, t2)) => self.existing_values.contains_key(&t1),
                None => true,
            }
        }).collect() */
    }

    pub fn apply_with(&self, action: &String, param_name: &String) -> Vec<Definition> {
        let mut new_terms = Vec::new();

        match (self.context_.lookup(action), self.context_.lookup(param_name)) {
            (_, None) => { },
            (None, Some(def)) => {
                match action.as_str() {
                    "eval" => { self.apply_builtin_eval_with(param_name, def, &mut new_terms); }
                    "eq_refl" => { self.apply_builtin_eq_refl_with(param_name, def, &mut new_terms); }
                    "eq_symm" => { self.apply_builtin_eq_symm_with(param_name, def, &mut new_terms); }
                    "rewrite" => { self.apply_builtin_rewrite_with(param_name, def, &mut new_terms); }
                    _ => {}
                }
            },
            (Some(action_def), Some(def)) => {
                match action_def.dtype.as_ref() {
                    Term::Arrow { input_types, output_type } => {
                        // Try putting the given value in each of the parameter slots.
                        for (i, input_type) in input_types.iter().enumerate() {
                            let val = Term::Atom { name: param_name.clone() }.rc().eval(&self.context_);

                            let mut unifier = Unifier::new();

                            let (value_pattern, param_type) = match input_type.as_ref() {
                                Term::PatternDeclaration { pattern, dtype } => (Some(pattern.clone()), dtype.clone()),
                                _ => (None, input_type.clone())
                            };

                            // If this parameter has a value_pattern, def must have a value and unify with it.
                            if let Some(p) = &value_pattern {
                                if !p.unify_params(&val, &mut unifier) {
                                    continue;
                                }
                            }

                            if !param_type.unify_params(&def.dtype, &mut unifier) {
                                continue;
                            }

                            let mut fixed_params = Vec::new();
                            for (j, _t) in input_types.iter().enumerate() {
                                if i == j {
                                    fixed_params.push(Some(param_name));
                                } else {
                                    fixed_params.push(None);
                                }
                            }

                            let mut results = Vec::new();
                            self.nondeterministically_apply_arrow(
                                &Rc::new(Term::Atom { name: action.clone() }),
                                input_types,
                                output_type,
                                &mut Vec::new(),
                                &fixed_params,
                                &mut results
                            );

                            for (r, t) in results.into_iter() {
                                new_terms.push(Definition { dtype: t,
                                                            value: Some(r) });
                            }
                        }

                        new_terms = self.filter_equalities(new_terms);
                    },
                    _ => {}
                }
            },
        }

        new_terms
    }

    fn apply_builtin_eval_with(&self, obj_name: &String, def: &Definition, new_terms: &mut Vec<Definition>) {
        if let Some(val) = &def.value {
            /* Numeric evaluation */
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
                                        arguments: vec![Rc::new(Term::Atom { name: obj_name.clone() })],
                                    }))
                                });
                            }
                        }
                    }
                }
            }

            /* Lambda evaluation */
            let evaluated_term = val.eval(&self.context_);
            if evaluated_term != *val {
                new_terms.push(Definition {
                    dtype: Rc::new(Term::Application {
                        function: Rc::new(Term::Atom { name: String::from("=") }),
                        arguments: vec![val.clone(), evaluated_term.clone()],
                    }),
                    value: Some(Rc::new(Term::Application {
                        function: Rc::new(Term::Atom { name: String::from("eval") }),
                        arguments: vec![Rc::new(Term::Atom { name: obj_name.clone() })],
                    }))
                });
            }
        }
    }

    fn apply_builtin_eq_refl_with(&self, name: &String, def: &Definition, new_terms: &mut Vec<Definition>) {
        if def.is_prop(&self.context_) || def.is_type(&self.context_) || def.is_arrow(&self.context_) {
            return;
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

    fn apply_builtin_eq_symm_with(&self, name: &String, def: &Definition, new_terms: &mut Vec<Definition>) {
        if let Some((t1, t2)) = def.dtype.extract_equality() {
            if t1 == t2 {
                return;
            }

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

    fn apply_builtin_rewrite_with(&self, name: &String, def: &Definition, new_terms: &mut Vec<Definition>) {
        // If def is an equality, try using it to rewrite in other props.
        if let Some((t1, t2)) = def.dtype.extract_equality() {
            if t1 != t2 {
                for prop_name in self.context_.insertion_order.iter() {
                    let prop_def = self.context_.lookup(prop_name).unwrap();
                    if prop_def.is_prop(&self.context_) {
                        self.rewrite_in(&prop_name, &prop_def.dtype, &t1, &t2, name, new_terms);
                    }
                }
            }
        }

        // If def is a prop, try using other equalities to rewrite it.
        if def.is_prop(&self.context_) {
            for eq_name in self.context_.insertion_order.iter() {
                let eq_def = self.context_.lookup(eq_name).unwrap();
                if let Some((t1, t2)) = eq_def.dtype.extract_equality() {
                    if t1 != t2 {
                        self.rewrite_in(&name, &def.dtype, &t1, &t2, eq_name, new_terms);
                    }
                }
            }
        }
    }

    fn rewrite_in(&self,
                  prop_name: &String,
                  prop_type: &Rc<Term>,
                  t1: &Rc<Term>,
                  t2: &Rc<Term>,
                  eq_name: &String,
                  new_terms: &mut Vec<Definition>) {
        for (i, rw) in rewrite_all(&prop_type, &t1, &t2).into_iter().enumerate() {
            new_terms.push(
                Definition {
                    dtype: rw,
                    value: Some(Rc::new(Term::Application {
                        function: Rc::new(Term::Atom { name: String::from("rewrite") }),
                        arguments: vec![
                            Rc::new(Term::Atom { name: eq_name.clone() }),
                            Rc::new(Term::Atom { name: prop_name.clone() }),
                            Rc::new(Term::Atom { name: i.to_string() }),
                        ],
                    }))
                }
            );
        }
    }

    // Returns whether there already exists an object in the context with the same value
    // (or same type, in case the type is a prop, because of proof irrelevance).
    fn is_redundant(&self, dtype: &Rc<Term>, value: &Option<Rc<Term>>) -> bool {
        if dtype.is_prop(&self.context_) {
            // Proof irrelevance - check for anything with this same type.
            self.inhabited_types.contains_key(dtype)
        } else {
            match value {
                Some(val) => self.existing_values.contains_key(val),
                None => false,
            }
        }
    }

    fn define_subterms(&mut self, t: &Rc<Term>, is_root: bool, subterm_names: &mut Vec<String>) {
        let is_real_atom = match t.as_ref() {
            Term::Atom { name } => {
                let real_type = Rc::new(Term::Atom { name: String::from(REAL_TYPE_CONST) });
                if is_real_const(name) && !self.is_redundant(&real_type, &Some(t.clone())) {
                    self.existing_values.entry(t.clone()).or_insert_with(|| name.clone());
                    self.inhabited_types.entry(real_type.clone()).or_insert_with(Vec::new).push(name.clone());
                    self.context_.define(name.clone(),
                                         Definition { dtype: real_type,
                                                      value: None });
                    // subterm_names.push(name.clone());
                    true
                } else {
                    false
                }
            },
            Term::Application { function, arguments } => {
                self.define_subterms(&function, false, subterm_names);

                for t in arguments.iter() {
                    self.define_subterms(&t, false, subterm_names);
                }

                false
            },
            _ => false,
        };

        if !is_root && !is_real_atom && !self.existing_values.contains_key(t) &&
            !self.is_builtin_application(t) {
            let dname = format!("!sub{}", self.next_term_id());
            let dtype = t.get_type(&self.context_);

            self.existing_values.insert(t.clone(), dname.clone());
            self.inhabited_types.entry(dtype.clone()).or_insert_with(Vec::new).push(dname.clone());

            self.context_.define(dname.clone(),
                                 Definition { dtype,
                                              value: Some(t.clone()) });
            subterm_names.push(dname);
        }
    }

    fn is_builtin_application(&self, t: &Rc<Term>) -> bool {
        match t.as_ref() {
            Term::Application { function, arguments: _ } => {
                match function.as_ref() {
                    Term::Atom { name } => {
                        name == "rewrite" || name == "eq_symm" || name == "eq_refl" || name == "eval"
                    },
                    _ => false
                }
            },
            _ => false
        }
    }
}

impl Universe for Derivation {
    fn define(&mut self, name: String, def: Definition, rename: bool) -> Vec<String> {
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

        let mut subterm_names = vec![];

        self.define_subterms(&def.dtype, false, &mut subterm_names);

        if !def.is_prop(&self.context_) {
            if let Some(value) = &def.value {
                self.define_subterms(&value, true, &mut subterm_names);
                self.existing_values.entry(value.clone()).or_insert_with(|| name.clone());
            }
        }

        self.existing_values.entry(Term::Atom { name: name.clone() }.rc()).or_insert_with(|| name.clone());
        self.inhabited_types.entry(def.dtype.clone()).or_insert_with(Vec::new).push(name.clone());

        self.context_.define(name, def);

        subterm_names
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
        let target_type = Rc::new(target_type.clone());
        let target_type_e = target_type.eval(&self.context_);

        for def in results.iter() {
            if target_type_e == def.dtype.eval(&self.context_) {
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
                    Term::Arrow { input_types, output_type } => {
                        let mut results = Vec::new();
                        self.nondeterministically_apply_arrow(
                            &Rc::new(Term::Atom { name: action.clone() }),
                            input_types,
                            output_type,
                            &mut Vec::new(),
                            &vec![],
                            &mut results
                        );

                        for (r, t) in results.into_iter() {
                            new_terms.push(Definition { dtype: t,
                                                        value: Some(r) });
                        }

                        new_terms = self.filter_equalities(new_terms)
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

        leq_n_n : [('n : nat) -> (leq 'n 'n)].
        leq_s : [('n : nat) -> ('m : nat) -> (leq 'n 'm) -> (leq 'n (s 'm))].
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

        +_comm : [('a : real) -> ('b : real) -> (= (+ 'a 'b) (+ 'b 'a))].
        +-_assoc : [('a : real) -> ('b : real) -> ('c : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].
        +0_id : [('a : real) -> (= (+ 'a 0) 'a)].

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

    #[test]
    fn test_equation_solution_with_pattern_declarations() {
        let real_theory: Context = "
        real : type.

        = : [(t : type) -> t -> t -> type].
        + : [real -> real -> real].
        - : [real -> real -> real].
        * : [real -> real -> real].

        +_comm : [((+ 'a 'b) : real) -> (= (+ 'a 'b) (+ 'b 'a))].
        +-_assoc : [((- (+ 'a 'b) 'c) : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].
        +0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].

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

    #[test]
    fn test_arrow_with_pattern_declaration() {
        let real_theory: Context = "
        real : type.

        = : [('t : type) -> 't -> 't -> prop].
        + : [real -> real -> real].
        * : [real -> real -> real].

        /* This arrow can only be applied to an existing (+ 'a 0) term */
        +0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].

        x : real.
        y : real.

        x_eq : (= (+ (* 2 x) 0) 2).
        y_eq : (= y (* x x)).
        "
        .parse()
        .unwrap();

        let mut u = Derivation::new();
        u.incorporate(&real_theory);

        // This should work.
        assert!(u.show_by(&"+0_id".to_string(), &"(= (+ (* 2 x) 0) (* 2 x))".parse().unwrap()).is_ok());
        // This should not - there's no (+ x 0) anywhere!
        assert!(u.show_by(&"+0_id".to_string(), &"(= (+ x 0) x)".parse().unwrap()).is_err());
        // Nor any (+ (* 2 y) 0) anywhere.
        assert!(u.show_by(&"+0_id".to_string(), &"(= (+ (* 2 y) 0) (* 2 y))".parse().unwrap()).is_err());

    }

    #[test]
    fn test_apply_with() {
        let real_theory: Context = "
        real : type.

        = : [('t : type) -> 't -> 't -> prop].
        + : [real -> real -> real].

        +0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].

        let yes : real = (+ (+ 1 2) 0). /* Arrow applies. */
        let no : real = (+ 2 3). /* Arrow does not apply. */
        "
        .parse()
        .unwrap();

        let mut u = Derivation::new();
        u.incorporate(&real_theory);

        // This should work.
        assert_eq!(u.apply_with(&"+0_id".to_string(), &"yes".to_string()).len(), 1);
        assert_eq!(u.apply_with(&"+0_id".to_string(), &"no".to_string()).len(), 0);
    }
}
