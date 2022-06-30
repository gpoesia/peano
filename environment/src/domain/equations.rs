use std::rc::Rc;
use crate::universe::{Term, Universe, EGraphUniverse, Definition};

use commoncore::domain::equations::{Equations as CCEquations, Term as CCTerm};

use super::Domain;

pub struct Equations {
    #[allow(dead_code)]
    cc_equations: CCEquations,
    base_universe: EGraphUniverse,
    real_dtype: Rc<Term>,
    variable_term: Rc<Term>,
}

impl Equations {
    pub fn _new_with_templates(templates: &str) -> Equations {
        let u = EGraphUniverse::new();
        // u.incorporate(&include_str!("../../theories/equations.p").parse().unwrap());

        Equations {
            cc_equations: CCEquations::new(templates),
            base_universe: u,
            real_dtype: Rc::new(Term::Atom { name: "real".to_string() }),
            variable_term: Rc::new(Term::Atom { name: "x".to_string() }),
        }
    }

    pub fn new_ct() -> Equations {
        Self::_new_with_templates(include_str!("./templates/equations-ct.txt"))
    }

    pub fn new_easy() -> Equations {
        Self::_new_with_templates(include_str!("./templates/equations-easy.txt"))
    }
}

impl Domain for Equations {
    fn name(&self) -> String {
        String::from("equations")
    }

    fn generate(&self, seed: u64) -> (EGraphUniverse, String) {
        let eq = self.cc_equations.generate_eq_term(seed);
        let eq_term = commoncore_term_to_peano_term(eq.t.as_ref());
        let problem_str = format!("{}", eq_term);
        let u = self.base_universe.clone();
        (u, problem_str)
    }

    fn size(&self) -> u64 {
        u64::MAX
    }

    fn reward(&self, u: &EGraphUniverse) -> bool {
        return u.value_of(&self.variable_term).is_some();
    }
}

fn commoncore_term_to_peano_term(t: &CCTerm) -> Rc<Term> {
    match t {
        CCTerm::Equality(lhs, rhs) => {
            Rc::new(Term::Application { function: Rc::new(Term::Atom { name: String::from("=") }),
                                        arguments: vec![
                                            commoncore_term_to_peano_term(lhs.t.as_ref()),
                                            commoncore_term_to_peano_term(rhs.t.as_ref()),
                                        ]})
        },
        CCTerm::BinaryOperation(op, lhs, rhs) => {
            Rc::new(Term::Application { function: Rc::new(Term::Atom { name: op.to_string() }),
                                        arguments: vec![
                                            commoncore_term_to_peano_term(lhs.t.as_ref()),
                                            commoncore_term_to_peano_term(rhs.t.as_ref()),
                                        ]})
        },
        CCTerm::UnaryMinus(expr) => {
            Rc::new(Term::Application { function: Rc::new(Term::Atom { name: String::from("un-") }),
                                        arguments: vec![commoncore_term_to_peano_term(expr.t.as_ref())] })
        },
        CCTerm::Variable(v) => Rc::new(Term::Atom { name: v.clone() }),
        CCTerm::Number(n) => {
            if n.is_integer() {
                Rc::new(Term::Atom { name: n.to_string() })
            } else {
             Rc::new(Term::Application { function: Rc::new(Term::Atom { name: String::from("/") }),
                                         arguments: vec![
                                             Rc::new(Term::Atom { name: n.numer().to_string() }),
                                             Rc::new(Term::Atom { name: n.denom().to_string() }),
                                        ]})
            }
        },
        CCTerm::AnyNumber => Rc::new(Term::Atom { name: String::from("?") })
    }
}
