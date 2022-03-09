use std::option::Option;
use std::rc::Rc;
use std::collections::hash_set::Iter;
use std::collections::hash_map::HashMap;
use std::io;
use std::path::Path;

use egg::*;
use super::term::{Context, Term, Definition};

const IS_NODE: &str = &"$is";
const PARAM_NODE: &str = &"$param";
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
        Universe { egraph: Default::default(), context: Context::new(), id_counter: 0 }
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

        let decl_id = self.add_term(&decl, false);

        if let Some(value) = def.value {
            let value_id = self.add_term(&value, false);
            self.egraph.union(self.egraph[decl_id].nodes[0].children[0], value_id);
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

    fn add_term(&mut self, t: &Rc<Term>, is_param: bool) -> Id {
        match t.as_ref() {
            Term::Declaration{ name, dtype } => {
                if !is_param {
                    // Equality objects are dematerialized and translate to a merge in the e-graph.
                    if let Some((t1, t2)) = dtype.extract_equality() {
                        let t1_id = self.add_term(&t1, false);
                        let t2_id = self.add_term(&t2, false);
                        self.egraph.union(t1_id, t2_id);
                        return t1_id
                    }
                }

                let type_id = self.add_term(dtype, false);
                let name_id = self.egraph.add(SymbolLang::leaf(&name));
                self.egraph.add(SymbolLang::new(if is_param { PARAM_NODE } else { IS_NODE },
                                                vec![name_id, type_id]))
            },
            Term::Atom { name } => {
                self.egraph.add(SymbolLang::leaf(name))
            },
            Term::Arrow { input_types, output_type } => {
                let mut v = Vec::new();
                for t in input_types.iter() {
                    v.push(self.add_term(&t, true));
                }
                v.push(self.add_term(&output_type, true));
                self.egraph.add(SymbolLang::new(ARROW_NODE, v))
            },
            Term::Lambda { parameters, body } => {
                let mut v = Vec::new();
                for t in parameters.iter() {
                    v.push(self.add_term(&t, true));
                }
                v.push(self.add_term(&body, false));
                self.egraph.add(SymbolLang::new(LAMBDA_NODE, v))
            },
            Term::Application { function, arguments } => {
                let type_expr = &t.get_type(&self.context);
                let type_id = self.add_term(type_expr, false);

                let mut v = Vec::new();
                v.push(self.add_term(&function, false));

                for t in arguments.iter() {
                    v.push(self.add_term(&t, false));
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

    pub fn inhabited(&self, term_type: &Rc<Term>) -> Option<String> {
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
            // FIXME this is *very* expensive: we should find a way to get around using the e-graph by
            // canonizaling terms in some way. At the very least we can cache the e-classes of the types
            // of terms in the context and just use half of the queries in `are_equivalent`.
            if self.are_equivalent(&param_type, &last_def.dtype) {
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
        self.context = extract_context_from_egraph(&self.egraph);
    }
}

// Returns the list of e-classes in topological order (leaves first);
fn topologically_sort_eclasses(egraph: &EGraph<SymbolLang, ()>) -> Vec<Id> {
    let mut dependents = HashMap::<Id, Vec<Id>>::new();
    let mut n_dependencies = HashMap::<Id, usize>::new();

    for eclass in egraph.classes() {
        for node in eclass.nodes.iter() {
            for dep in node.children.iter() {
                if *dep != eclass.id {
                    *(n_dependencies.entry(eclass.id).or_default()) += 1;
                    dependents.entry(*dep).or_default().push(eclass.id);
                }
            }
        }
    }

    let mut free = Vec::new();

    for eclass in egraph.classes() {
        if n_dependencies.get(&eclass.id).is_none() {
            free.push(eclass.id);
        }
    }

    let mut eclasses_in_topological_order = Vec::<Id>::new();

    while free.len() > 0 {
        let next = free.pop().unwrap();
        eclasses_in_topological_order.push(next);

        if let Some(deps) = dependents.get(&next) {
            for r_dep in deps.iter() {
                n_dependencies.entry(*r_dep).and_modify(|cnt| {
                    *cnt -= 1;
                    if *cnt == 0 {
                        free.push(*r_dep);
                    }
                });
            }
        }
    }

    eclasses_in_topological_order
}

#[derive(Default)]
struct EClassTerms {
    name: String,
    values: Vec<Rc<Term>>,
    expressions: Vec<Rc<Term>>,
    declaration: Option<Rc<Term>>,
}

impl EClassTerms {
    pub fn is_intermediary_node(&self) -> bool {
        self.name.starts_with("!sub")
    }

    pub fn set_intermediary_name(&mut self, id: usize) {
        self.name = format!("!sub{}", id);
    }

    pub fn canonical_representer(&self) -> Rc<Term> {
        // If any constant is associated with this node, return one of them.
        if self.values.len() > 0 {
            return self.values[0].clone();
        }

        match &self.declaration {
            Some(d) => d.clone(),
            None => {
                if self.is_intermediary_node() {
                    self.expressions[0].clone()
                } else {
                    Rc::new(Term::Atom { name: self.name.clone() })
                }
            }
        }
    }

    pub fn canonical_value(&self) -> Option<Rc<Term>> {
        // If any constant is associated with this node, return one of them.
        if self.values.len() > 0 {
            return Some(self.values[0].clone());
        }

        return self.expressions.get(0).map(|e| e.clone());
    }

    pub fn alternative_values(&self) -> Vec<Rc<Term>> {
        if self.values.len() > 0 {
            return self.values[1..].iter().map(|v| v.clone())
                                          .chain(self.expressions.iter().map(|v| v.clone()))
                                          .collect();
        }
        self.expressions[1..].iter().map(|v| v.clone()).collect()
    }
}

fn extract_context_from_egraph(egraph: &EGraph<SymbolLang, ()>) -> Context {
    // 1- Topologically sort e-classes.
    let eclasses = topologically_sort_eclasses(egraph);

    // 2- For each e-class, gather (a) names, (b) constants, (c) expressions
    let mut next_id = 0;

    let mut eterms : HashMap<Id, EClassTerms> = HashMap::new();
    let mut c = Context::new();

    for id in eclasses.iter() {
        let eclass = &egraph[*id];
        let mut terms : EClassTerms = Default::default();
        terms.set_intermediary_name(next_id);
        next_id += 1;

        for node in &eclass.nodes {
            let op = node.op.as_str();

            // Decide where to put this child: name, value or expression?
            if op == LAMBDA_NODE {
                // lambdas are irreducible values.
                let n_params = node.children.len() - 1;
                terms.values.push(Rc::new(Term::Lambda {
                    parameters: node.children[0..n_params].iter()
                                                          .map(|id| eterms[id].canonical_representer())
                                                          .collect(),
                    body: eterms[&node.children[n_params]].canonical_representer(),
                }))
            } else if op == APPLY_NODE {
                terms.expressions.push(Rc::new(Term::Application {
                    function: eterms[&node.children[0]].canonical_representer(),
                    arguments: node.children[1..].iter().map(|id| eterms[id].canonical_representer()).collect(),
                }))
            } else if op == ARROW_NODE {
                terms.expressions.push(Rc::new(Term::Arrow {
                    input_types: node.children[0..node.len() - 1].iter().map(|id| eterms[id].canonical_representer()).collect(),
                    output_type: eterms[&node.children[node.len() - 1]].canonical_representer(),
                }))
            } else if op == IS_NODE || op == PARAM_NODE {
                let decl_node = &eterms[&node.children[0]];
                let dtype_node = &eterms[&node.children[1]];

                terms.declaration = Some(Rc::new(Term::Declaration {
                    name: decl_node.name.clone(),
                    dtype: dtype_node.canonical_representer(),
                }));

                // 3- For each named declaration e-class, add it to the context.
                if op == IS_NODE {
                    let canonical_value = decl_node.canonical_value();
                    c.define(decl_node.name.clone(),
                             Definition { dtype: dtype_node.canonical_representer(),
                                          value: canonical_value.clone() });

                    // If the node's e-class has alternative equal values, materialize the
                    // equality in the context with new equality objects.
                    if let Some(cv) = canonical_value {
                        for v in decl_node.alternative_values() {
                            c.define(format!("eq{}", next_id),
                                     Definition { dtype: Term::make_equality_object(cv.clone(),
                                                                                    v.clone()),
                                                  value: None });
                        }
                    }
                }
            } else {
                match op.parse::<i64>() {
                    Ok(_) => {
                        // It's a number, add to values.
                        terms.values.push(Rc::new(Term::Atom { name: op.to_string() }))
                    }
                    Err(_) => {
                        // It's a name, set as name (will make it a non-intermediary node).
                        terms.name = op.to_string();
                    }
                }
            }
        }

        eterms.insert(*id, terms);
    }

    c
}

#[cfg(test)]
pub mod tests {
    use crate::universe::{Universe, Context, Term, Definition};
    use crate::universe::universe::extract_context_from_egraph;
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
    fn test_context_from_egraph() {
        let nat_theory : Context = "
nat : type.
z : nat.
s : [nat -> nat].
plus : [nat -> nat -> nat].

one : nat = (s z).
two : nat = (s one).
three : nat = 3.

twice : [nat -> nat] = lambda (n : nat) (plus n n).
five : nat = (plus (plus two two) one).
ten : nat = (twice five).

_1 : (= (plus two two) 4).
_2 : (= (plus 4 one) 5).
_3 : (= (twice 5) 10).

leq : [nat -> nat -> prop].
leq_n_n : [(n : nat) -> (leq n n)].
leq_n_sn : [(n : nat) -> (leq n (s n))].
leq_trans : [(n : nat) -> (m : nat) -> (o : nat) -> (leq n m) -> (leq m o) -> (leq n o)].
".parse().unwrap();

        let mut u = Universe::new();
        u.incorporate(&nat_theory);

        // u.to_png("u.png").unwrap();

        let egraph = u.egraph;
        let context = extract_context_from_egraph(&egraph);

        // FIXME add automated test instead of just looking at the output below.
        // println!("Extracted context: \n{}", context.to_string());
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

        // assert!(u.inhabited(&Rc::new("nat".parse().unwrap())).is_some());
        // assert!(u.inhabited(&Rc::new("(leq z (s z))".parse().unwrap())).is_some());
        // assert!(u.inhabited(&Rc::new("(leq (s z) (s (s (s z))))".parse().unwrap())).is_some());
    }
}
