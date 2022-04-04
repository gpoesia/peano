use std::rc::Rc;
use std::fmt;
use crate::universe::{Universe, Term};
use crate::universe::term::{TermParser, parse_term, parse_definition, Rule, Definition};
use std::collections::HashMap;

use pest::Parser;
use pest::iterators::Pair;
use pest::error::{Error as PestError};

#[derive(Clone)]
pub struct VerificationScript {
    pub name: String,
    instructions: Vec<VerificationInstruction>
}

#[derive(Debug, Clone)]
pub enum VerificationError {
    ScriptNotFound(String),
    StepFailed(&'static str, Rc<Term>, String, Vec<Rc<Term>>)
}

impl VerificationScript {
    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn execute(&self, mut u: Universe) -> Result<(), VerificationError> {
        for instr in self.instructions.iter() {
            match instr {
                VerificationInstruction::Definition(name, def) => {
                    u.define(name.clone(), def.clone(), false);
                    u.rebuild();
                },
                VerificationInstruction::ConstructBy(target, action) => {
                    if let Err(defs) = u.construct_by(action, target.as_ref()) {
                        return Err(VerificationError::StepFailed(
                            &"construct",
                            target.clone(),
                            action.clone(),
                            defs.into_iter().filter_map(|d| d.value).collect()
                        ));
                    }
                },
                VerificationInstruction::ShowBy(target, action) => {
                    if let Err(defs) = u.show_by(action, target.as_ref()) {
                        return Err(VerificationError::StepFailed(
                            &"show",
                            target.clone(),
                            action.clone(),
                            defs.into_iter().map(|d| d.dtype).collect()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum VerificationInstruction {
    Definition(String, Definition),
    ConstructBy(Rc<Term>, String),
    ShowBy(Rc<Term>, String),
}

pub(super) fn parse_verification_script(pair: Pair<Rule>) -> VerificationScript {
    let rule = pair.as_rule();
    let mut sub : Vec<Pair<Rule>> = pair.into_inner().collect();

    assert_eq!(rule, Rule::verify);
    assert_eq!(sub.len(), 2);

    let name = sub[0].as_str();
    let instructions = sub.remove(1).into_inner().map(parse_verification_instruction).collect();
    VerificationScript { name: String::from(name), instructions }
}

fn parse_verification_instruction(pair: Pair<Rule>) -> VerificationInstruction {
    let rule = pair.as_rule();
    let mut sub : Vec<Pair<Rule>> = pair.into_inner().collect();

    match rule {
        Rule::show_by => {
            let action = String::from(sub[1].as_str());
            let dtype = parse_term(sub.remove(0), &mut HashMap::new());
            VerificationInstruction::ShowBy(dtype, action)
        },
        Rule::construct_by => {
            let action = String::from(sub[1].as_str());
            let object = parse_term(sub.remove(0), &mut HashMap::new());
            VerificationInstruction::ConstructBy(object, action)
        },
        Rule::definition => {
            let (name, def) = parse_definition(sub);
            VerificationInstruction::Definition(name, def)
        }
        _ => unreachable!()
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationError::ScriptNotFound(name) => write!(f,
                                                              "Verification script '{}' not found.",
                                                              name),
            VerificationError::StepFailed(step_type, target_term, action, action_results) => {
                write!(f, "'{}' step failed: action {} did not {} {}\nAction {}ed {} results:",
                       step_type, action, step_type, target_term, step_type, action_results.len()
                )?;

                for (i, t) in action_results.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }

                Ok(())
            }
        }
    }
}
