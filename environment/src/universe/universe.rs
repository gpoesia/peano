use std::option::Option;
use std::rc::Rc;
use std::collections::hash_set::{HashSet, Iter};
use std::collections::hash_map::HashMap;
use std::io;
use std::path::Path;

use egg::*;
use super::term::{Context, Term, Definition, is_parameter_name};

const IS_NODE: &str = &"$is";
const PARAM_NODE: &str = &"$param";
const ARROW_NODE: &str = &"$arrow";
const APPLY_NODE: &str = &"$app";
const LAMBDA_NODE: &str = &"$lambda";

pub struct Universe {
    egraph: EGraph<SymbolLang, ()>,
    context: Context,
    eclass_name: HashMap<Id, String>,
    next_id: usize,
}

impl Universe {
    pub fn new() -> Universe {
        Universe { egraph: Default::default(), context: Context::new(),
                   eclass_name: HashMap::new(), next_id: 0 }
    }

    pub fn size(&self) -> usize {
        self.egraph.number_of_classes()
    }

    pub fn get_type_constant(&self) -> &Rc<Term> {
        self.context.get_type_constant()
    }

    pub fn get_prop_constant(&self) -> &Rc<Term> {
        self.context.get_prop_constant()
    }

    pub fn define(&mut self, name: String, def: Definition, rename: bool) -> Option<Id> {
        let name = if rename {
            format!("{}{}", name, self.next_term_id())
        } else {
            name
        };

        let decl = Rc::new(Term::Declaration { name: name.clone(), dtype: def.dtype });
        let decl_id = self.define_term(&decl, false, rename);

        if let Some(decl_id) = decl_id {
            if let Some(value) = def.value {
                if let Some(value_id) = self.define_term(&value, false, rename) {
                    if self.egraph[decl_id].nodes[0].children.len() != 2 {
                        panic!("Expected the declaration {} to be represented by a $is node with two children!", decl);
                    }
                    self.egraph.union(self.egraph[decl_id].nodes[0].children[0], value_id);
                    self.context.set(&name, value.clone());
                    self.eclass_name.insert(value_id, name.clone());
                }
            }
        }

        decl_id
    }

    // Rebuilds the e-graph and removes duplicate definitions from the context (named objects with values
    // represented in the same e-class).
    pub fn rebuild(&mut self) {
        self.egraph.rebuild();

        let mut duplicates = HashSet::new();

        self.eclass_name.clear();

        for name in self.context.insertion_order.iter() {
            if let Some(def) = self.context.lookup(&name) {
                if let Some(val) = &def.value {
                    let eclass_id = self.is_represented(val).unwrap();
                    let current_value = self.eclass_name.get(&eclass_id);

                    match current_value {
                        None => { self.eclass_name.insert(eclass_id, name.clone()); },
                        Some(v) => {
                            if v != name {
                                duplicates.insert(name.clone());
                            }
                        }
                    }
                }
            }
        }

        let mut proofs = HashSet::new();

        for name in self.context.insertion_order.iter() {
            if let Some(def) = self.context.lookup(&name) {
                // If the dtype is of type proposition, check for irrelevance
                if def.dtype.get_type(&self.context) == *self.get_prop_constant() {
                    if let Some(id) = self.is_represented(&def.dtype) {
                        if proofs.contains(&id) {
                            duplicates.insert(name.clone());
                        }
                        proofs.insert(id);
                    }
                }
            }
        }

        for name in duplicates {
            self.context.destroy(&name);
        }

        self.context.rebuild();
    }

    pub fn incorporate(&mut self, context: &Context) {
        for name in context.insertion_order.iter() {
            if let Some(def) = context.lookup(name) {
                self.define(name.clone(), def.clone(), false);
            }
        }

        self.rebuild();
    }

    fn is_represented(&self, t: &Rc<Term>) -> Option<Id> {
        self.egraph.lookup_expr(&t.to_recexpr())
    }

    // Recursively defines a term and all of its sub-terms in both the Context and the e-graph.
    // Returns the e-class ID associated with t in the e-graph after it is defined,
    // or None if the term was rejected (the only case for now is if it's an equality object
    // such that both sides are new and constrain_equality is true).
    fn define_term(&mut self, t: &Rc<Term>, is_param: bool, constrain_equality: bool) -> Option<Id> {
        // FIXME: This call causes repeated computation at every recursion level.
        // If this becomes a performance issue in the future,
        // we can either cache free variables (likely the best, can be done when constructing the term)
        // or at least traverse bottom-up to minimize repeated work.
        let has_free_variables = t.free_variables().len() > 0;

        match t.as_ref() {
            Term::Declaration{ name, dtype } => {
                if !is_param {
                    // Equality objects are dematerialized and result in merging two e-classes.
                    if let Some((t1, t2)) = dtype.extract_equality() {
                        // When equality objects are constrained, one of the sides has to already exist
                        // in order for the equality to be added to the e-graph.
                        // This keeps equality-producing arrows from exploding the universe by
                        // adding two completely new objects every time they're applied (typically enabling
                        // even more equalities and objects to be produced in a follow-up application).
                        // FIXME(gpoesia) we should add equality objects to the context somehow.
                        if !constrain_equality {
                            let t1_id = self.define_term(&t1, false, false)?;
                            let t2_id = self.define_term(&t2, false, false)?;
                            self.egraph.union(t1_id, t2_id);
                        } else if let Some(t1_id) = self.is_represented(&t1) {
                            let t2_id = self.define_term(&t2, false, false)?;
                            self.egraph.union(t1_id, t2_id);
                        } else if let Some(t2_id) = self.is_represented(&t2) {
                            let t1_id = self.define_term(&t1, false, false)?;
                            self.egraph.union(t1_id, t2_id);
                        }

                        return None;
                    }
                }

                let type_id = self.define_term(dtype, false, constrain_equality)?;
                let name_id = self.egraph.add(SymbolLang::leaf(&name));

                if !has_free_variables && !is_parameter_name(name.as_str()) {
                    self.context.define(name.clone(),
                                        Definition { dtype: dtype.clone(), value: None });
                }

                Some(self.egraph.add(SymbolLang::new(if is_param { PARAM_NODE } else { IS_NODE },
                                                     vec![name_id, type_id])))
            },
            Term::Atom { name } => {
                // FIXME: If this atom contains a real constant, we should add a type annotation and
                // add it to the context.
                Some(self.egraph.add(SymbolLang::leaf(name)))
            },
            Term::Arrow { input_types, output_type } => {
                let mut v = Vec::new();
                for t in input_types.iter() {
                    v.push(self.define_term(&t, true, constrain_equality)?);
                }
                v.push(self.define_term(&output_type, true, constrain_equality)?);
                Some(self.egraph.add(SymbolLang::new(ARROW_NODE, v)))
            },
            Term::Lambda { parameters, body } => {
                // FIXME: If this lambda does not have free variables, we could also add it
                // to the context and add a type annotation (it would then become a new action).
                let mut v = Vec::new();
                for t in parameters.iter() {
                    v.push(self.define_term(&t, true, constrain_equality)?);
                }
                v.push(self.define_term(&body, false, constrain_equality)?);
                Some(self.egraph.add(SymbolLang::new(LAMBDA_NODE, v)))
            },
            Term::Application { function, arguments } => {
                let type_expr = &t.get_type(&self.context);
                let type_id = self.define_term(type_expr, false, constrain_equality)?;

                let mut v = Vec::new();
                v.push(self.define_term(&function, false, constrain_equality)?);

                for t in arguments.iter() {
                    v.push(self.define_term(&t, false, constrain_equality)?);
                }

                let app_id = self.egraph.add(SymbolLang::new(APPLY_NODE, v));

                self.egraph.add(SymbolLang::new(IS_NODE, vec![app_id, type_id]));

                if !has_free_variables {
                    let name = format!("!sub{}", self.next_term_id());
                    self.context.define(name.clone(),
                                        Definition { dtype: type_expr.clone(),
                                                     value: Some(t.clone()) });
                    let name_id = self.egraph.add(SymbolLang::leaf(name));
                    self.egraph.union(app_id, name_id);
                }

                Some(app_id)
            }
        }
    }

    fn next_term_id(&mut self) -> usize {
        self.next_id += 1;
        self.next_id
    }

    pub fn actions(&self) -> Iter<'_, String> {
        self.context.actions()
    }

    pub fn apply(&mut self, action: &String) -> Vec<Rc<Term>> {
        let mut new_terms = Vec::new();

        match self.context.lookup(action) {
            None => {
                match action.as_str() {
                    "eval" => { self.apply_builtin_eval(&mut new_terms); }
                    _ => {}
                }
            },
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
                            self.define(format!("!res_{}_", action),
                                        Definition { dtype: t.get_type(&self.context),
                                                     value: Some(t.clone()) },
                                        true);
                        }
                    },
                    _ => {}
                }
            },
        }
        self.rebuild();

        new_terms
    }

    fn apply_builtin_eval(&mut self, new_terms: &mut Vec<Rc<Term>>) {
        // FIXME Cache these.
        let lhs = "?lhs".parse::<egg::Var>().unwrap();
        let rhs = "?rhs".parse::<egg::Var>().unwrap();

        for op in ["+", "-", "*", "/"] {
            let query: egg::Pattern<SymbolLang> = format!("($app {} ?lhs ?rhs)", op)
                .as_str().parse().unwrap();

            let matches = query.search(&self.egraph);

            for m in matches.iter() {
                for s in &m.substs {
                    let mut lhs_val : Option<i64> = None;
                    let mut rhs_val : Option<i64> = None;

                    for n in self.egraph[s[lhs]].nodes.iter() {
                        if n.is_leaf() {
                            if let Ok(n) = n.op.as_str().parse::<i64>() {
                                lhs_val = Some(n);
                                break;
                            }
                        }
                    }

                    for n in self.egraph[s[rhs]].nodes.iter() {
                        if n.is_leaf() {
                            if let Ok(n) = n.op.as_str().parse::<i64>() {
                                rhs_val = Some(n);
                                break;
                            }
                        }
                    }

                    match (lhs_val, rhs_val) {
                        (Some(n1), Some(n2)) => {
                            if let Some(result) = apply_builtin_binary_op(n1, n2, op) {
                                let eq_type = Rc::new(Term::Application {
                                    function: Rc::new(Term::Atom { name: String::from("=") }),
                                    arguments: vec![
                                        Rc::new(Term::Atom { name: self.eclass_name[&m.eclass].clone() }),
                                        Rc::new(Term::Atom { name: result.to_string() }),
                                    ]
                                });

                                let obj_name = format!("eval{}", self.next_term_id());
                                new_terms.push(Rc::new(Term::Declaration { name: obj_name,
                                                                           dtype: eq_type.clone() }));
                            }
                        },
                        _ => {}
                    }
                }
            }
        }

        for nt in new_terms {
            if let Term::Declaration { name, dtype } = nt.as_ref() {
                let def = Definition { dtype: dtype.clone(), value: None };
                self.define(name.clone(), def, false);
            }
        }
    }

    // Checks if a type is inhabited in the current context, i.e., if we have any object
    // of that type. For proof objects, this amounts to knowing whether we have a proof
    // of the corresponding proposition. If so, returns the name of an object of the given type.
    pub fn inhabited(&self, term_type: &Rc<Term>) -> Option<String> {
        // Equality objects are implicit in the e-graph.
        if let Some((t1, t2)) = term_type.extract_equality() {
            return match (self.is_represented(&t1), self.is_represented(&t2)) {
                (id, id2) if id == id2 => Some(String::from("eq")),
                _ => None,
            }
        }

        let var = "?var".parse::<egg::Var>().unwrap();
        let query: egg::Pattern<SymbolLang> = format!("({} ?var {})", IS_NODE, term_type.to_pattern())
                                              .as_str().parse().unwrap();

        let matches = query.search(&self.egraph);

        if matches.len() == 0 {
            return None;
        }

        for m in matches.iter() {
            for n in self.egraph[m.substs[0][var]].nodes.iter() {
                if n.is_leaf() {
                    return Some(n.to_string());
                }
            }
        }

        None
    }

    pub fn value_of(&self, t: &Rc<Term>) -> Option<i64> {
        let query: egg::Pattern<SymbolLang> = t.to_pattern().as_str().parse().unwrap();
        let matches = query.search(&self.egraph);

        if matches.len() == 0 {
            return None;
        }

        for m in matches.iter() {
            for n in self.egraph[m.eclass].nodes.iter() {
                if n.is_leaf() {
                    if let Ok(v) = n.op.as_str().parse::<i64>() {
                        return Some(v);
                    }
                }
            }
        }

        None
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
        // FIXME this can be very heavily sped-up. One easy way to do this is to query the e-graph
        // with ($is ?name param_type), and then lookup the name associated with the e-class that
        // matches ?name using self.eclass_name. For any matches, we already know that they have
        // the right type.
        for name in self.context.insertion_order.iter() {
            if is_parameter_name(&name) {
                continue;
            }

            if let Some(def) = self.context.lookup(&name) {
                if self.are_equivalent(&param_type, &def.dtype) {
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
    }

    fn are_equivalent(&self, t1: &Rc<Term>, t2: &Rc<Term>) -> bool {
        let q1: egg::Pattern<SymbolLang> = t1.to_pattern().as_str().parse().unwrap();
        let m1 = q1.search(&self.egraph);

        if m1.len() == 0 {
            return t1 == t2;
        }

        let q2: egg::Pattern<SymbolLang> = t2.to_pattern().as_str().parse().unwrap();

        let m2 = q2.search(&self.egraph);

        if m2.len() == 0 {
            return t1 == t2;
        }

        return self.egraph.find(m1[0].eclass) == self.egraph.find(m2[0].eclass);
    }

    pub fn dump_context(&self) -> String {
        self.context.to_string()
    }

    pub fn canonicalize_equal_terms(&mut self) {
        return;
    }
}

fn apply_builtin_binary_op(n1: i64, n2: i64, op: &str) -> Option<i64> {
    match op {
        "+" => { Some(n1 + n2) },
        "-" => { Some(n1 - n2) },
        "*" => { Some(n1 * n2) },
        "/" if n2 != 0 && n1 % n2 == 0 => { Some(n1 / n2) },
        "/" => None,
        _ => panic!("Unknown builtin binary operator {}", op)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::universe::{Universe, Context, Term, Definition};
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

        // u.to_png("s0.png").unwrap();

        u.apply(&"s".to_string());

        // u.to_png("s1.png").unwrap();

        u.apply(&"s".to_string());

        // u.to_png("s2.png").unwrap();

        u.apply(&"s".to_string());

        // u.to_png("s3.png").unwrap();

        u.apply(&"s".to_string());

        // u.to_png("s4.png").unwrap();

        u.apply(&"s".to_string());

        // u.to_png("s5.png").unwrap();
        // FIXME test things here.
    }

    #[test]
    fn test_composite_type_inhabited() {
        let nat_theory : Context = "
nat : type.
z : nat.
s : [nat -> nat].

leq : [nat -> nat -> prop].
leq_n_sn : [(n : nat) -> (leq n (s n))].
leq_trans : [(n : nat) -> (m : nat) -> (o : nat) -> (leq n m) -> (leq m o) -> (leq n o)].
".parse().unwrap();

        let mut u = Universe::new();
        u.incorporate(&nat_theory);

        u.apply(&"s".to_string());
        u.apply(&"s".to_string());
        u.apply(&"s".to_string());
        u.apply(&"leq_n_sn".to_string());
        u.apply(&"leq_trans".to_string());
        u.apply(&"leq_trans".to_string());

        assert!(u.inhabited(&Rc::new("nat".parse().unwrap())).is_some());
        assert!(u.inhabited(&Rc::new("(leq z (s z))".parse().unwrap())).is_some());
        assert!(u.inhabited(&Rc::new("(leq (s z) (s (s (s z))))".parse().unwrap())).is_some());
    }
}
