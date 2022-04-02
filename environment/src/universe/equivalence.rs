// Equivalence comparison modulo the equalities represented in the e-graph.

use std::option::Option;
use std::rc::Rc;
use std::collections::hash_set::{HashSet, Iter};
use std::collections::hash_map::HashMap;
use std::io;
use std::path::Path;

use egg::*;
use super::term::{Context, Term, Definition, is_parameter_name};

// Represents a s-expr where some elements might be entire e-graph equivalence classes.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum AbstractSExp {
    ENode { id: Id },
    Application { op: String, children: Vec<AbstractSExp> }
}

impl AbstractSExp {
    pub fn new_application(op: String, children: Vec<AbstractSExp>) -> AbstractSExp {
        AbstractSExp::Application { op, children }
    }

    pub fn new_atom(op: String) -> AbstractSExp {
        AbstractSExp::Application { op, children: vec![] }
    }

    pub fn from_enode(id: Id) -> AbstractSExp {
        AbstractSExp::ENode { id }
    }

    /// Recursively abstract out this s-exp as much as possible by replacing
    /// applications with ENodes when they're represented in the given e-graph.
    pub fn abstract_with_egraph(&self, egraph: &EGraph<SymbolLang, ()>) -> AbstractSExp {
        match self {
            AbstractSExp::ENode { id: _ } => self.clone(),
            AbstractSExp::Application { op, children } => {
                let mut all_children_have_ids = true;
                let mut children_enodes = Vec::new();
                let mut children_sexps = Vec::new();

                for c in children.iter() {
                    let c_sexp = c.abstract_with_egraph(egraph);

                    if all_children_have_ids {
                        if let AbstractSExp::ENode { id: child_id } = &c_sexp {
                            children_enodes.push(*child_id);
                        } else {
                            all_children_have_ids = false;
                        }
                    }
                    children_sexps.push(c_sexp);
                }

                if all_children_have_ids {
                    if let Some(id) = egraph.lookup(SymbolLang::new(op.clone(), children_enodes)) {
                        return AbstractSExp::ENode { id };
                    }
                }

                AbstractSExp::Application { op: op.clone(), children: children_sexps }
            }
        }
    }
}
