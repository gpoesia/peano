use std::rc::Rc;
use std::vec::Vec;
use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::collections::hash_set::Iter;
use std::fmt;
use core::str::FromStr;

use pest::Parser;
use pest::iterators::Pair;
use pest::error::{Error as PestError};

const TYPE: &str = "type";

pub struct Context {
    type_const: Rc<Term>, // The global constant `type` used to define other types.
    pub definitions: HashMap<String, Vec<Definition>>, // Map of names to definitions.
    arrows: HashSet<String>, // Set of global definitions that are arrows.
}

#[derive(Clone, Debug, PartialEq)]
pub struct Definition {
    pub dtype: Rc<Term>,
    pub value: Option<Rc<Term>>,
}

impl Definition {
    pub fn new_concrete(dtype: Rc<Term>, value: Rc<Term>) -> Definition {
        Definition { dtype, value: Some(value) }
    }

    pub fn new_opaque(dtype: Rc<Term>) -> Definition {
        Definition { dtype: dtype, value: None }
    }
}

#[derive(Parser)]
#[grammar = "universe/term.pest"]
struct TermParser;

impl Context {
    pub fn new() -> Context {
        let type_const = Rc::new(Term::Atom { name: TYPE.to_string() });
        let type_const_def = Definition { dtype: type_const.clone(), value: None };
        let mut c = Context { definitions: HashMap::new(),
                              type_const: type_const.clone(),
                              arrows: HashSet::new() };
        c.definitions.insert(TYPE.to_string(), vec![type_const_def]);
        c
    }

    pub fn lookup(&self, name: &String) -> Option<&Definition> {
        if let Some(v) = self.definitions.get(name) {
            return v.get(v.len() - 1)
        }
        None
    }

    pub fn define_push(&mut self, name: String, def: Definition) {
        if let Term::Arrow { input_types: _, output_type: _ } = def.dtype.as_ref() {
            self.arrows.insert(name.clone());
        }

        if let Some(v) = self.definitions.get_mut(&name) {
            v.push(def);
        } else {
            self.definitions.insert(name, vec!(def));
        }
    }

    pub fn define(&mut self, name: String, def: Definition) {
        if let Term::Arrow { input_types: _, output_type: _ } = def.dtype.as_ref() {
            self.arrows.insert(name.clone());
        }

        if let Some(v) = self.definitions.get_mut(&name) {
            let idx = v.len() - 1;
            v[idx] = def;
        } else {
            self.definitions.insert(name, vec!(def));
        }
    }


    pub fn actions(&self) -> Iter<'_, String> {
        self.arrows.iter()
    }

    pub fn destroy(&mut self, name: &String) {
        if let Some(v) = self.definitions.get_mut(name) {
            if v.len() > 0 {
                v.pop();
            }
        }
    }

    pub fn get_type_constant(&self) -> &Rc<Term> {
        &self.type_const
    }

    pub fn size(&self) -> usize {
        self.definitions.len()
    }
}

impl FromStr for Context {
    type Err = PestError<Rule>;

    fn from_str(s : &str) -> Result<Context, PestError<Rule>> {
        let root = TermParser::parse(Rule::context, s)?.next().unwrap();

        let mut c = Context::new();
        let mut v: Vec<_> = root.into_inner().collect();

        for def_rule in v.drain(0..) {
            let rule = def_rule.as_rule();

            if rule != Rule::definition {
                continue;
            }

            let mut sub : Vec<Pair<Rule>> = def_rule.into_inner().collect();
            let mut children : Vec<Rc<Term>> = sub.drain(0..).map(|p| parse_term(p)).collect();
            let value = if children.len() == 2 { children.pop() } else { None };

            if let Term::Declaration { name, dtype } = children[0].as_ref() {
                let def = Definition { dtype: dtype.clone(), value: value };
                c.define(name.clone(), def);
            }
        }

        Ok(c)
    }
}

impl ToString for Context {
    fn to_string(&self) -> String {
        let mut s = String::new();

        s.push_str(format!("// {} definitions.\n", self.definitions.len()).as_str());

        for (name, defs) in self.definitions.iter() {
            let last_def = &defs[defs.len() - 1];

            s += &format!("{} : {}", name, last_def.dtype);

            if let Some(val) = &last_def.value {
                s += &format!(" = {}", val.to_string());
            }

            s.push_str(".\n")
        }
        s
    }
}


#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Term {
    Declaration { name: String, dtype: Rc<Term> },
    Atom { name: String },
    Arrow { input_types: Vec<Rc<Term>>, output_type: Rc<Term> },
    Lambda { parameters: Vec<Rc<Term>>, body: Rc<Term> },
    Application { function: Rc<Term>, arguments: Vec<Rc<Term>> },
}

fn parse_term(pair: Pair<Rule>) -> Rc<Term> {
    let rule = pair.as_rule();
    let s = pair.as_str();
    let mut sub : Vec<Pair<Rule>> = pair.into_inner().collect();

    match rule {
        Rule::lambda => {
            let mut children : Vec<Rc<Term>> = sub.drain(0..).map(|p| parse_term(p)).collect();
            let body = children.pop().unwrap();

            return Rc::new(Term::Lambda {
                parameters: children,
                body: body,
            });
        },
        Rule::declaration => {
            let mut children : Vec<Rc<Term>> = sub.drain(0..).map(parse_term).collect();
            assert_eq!(children.len(), 2, "Declaration should have two children: atom and type.");
            let dtype = children.pop().unwrap();
            let atom = children.pop().unwrap();
            if let Term::Atom { name } = Rc::try_unwrap(atom).unwrap() {
                Rc::new(Term::Declaration { name: name, dtype: dtype })
            } else {
                panic!("First child of a Declaration node should be an atom.")
            }
        },
        Rule::atom => {
            Rc::new(Term::Atom { name: s.to_string() })
        },
        Rule::arrow => {
            let input_types : Vec<Rc<Term>> = sub.drain(0..(sub.len() - 1)).map(parse_term).collect();
            let output_type = parse_term(sub.pop().unwrap());
            Rc::new(Term::Arrow { input_types, output_type })
        },
        Rule::application => {
            let arguments : Vec<Rc<Term>> = sub.drain(1..).map(parse_term).collect();
            let function = parse_term(sub.pop().unwrap());
            Rc::new(Term::Application { function, arguments })
        },
        _ => unreachable!(),
    }
}

impl Term {
    pub fn rc(&self) -> Rc<Term> {
        Rc::new(self.clone())
    }

    pub fn get_type(self: &Rc<Term>, ctx: &Context) -> Rc<Term> {
        match self.as_ref() {
            Term::Declaration { name: _, dtype } => dtype.clone(),
            Term::Atom { name } => {
                if let Some(def) = ctx.lookup(&name) {
                    return def.dtype.clone();
                }
                panic!("{} undeclared", name)
            },
            Term::Arrow { input_types: _, output_type: _ } => ctx.get_type_constant().clone(),
            Term::Lambda { parameters, body } =>
                Rc::new(Term::Arrow {
                    input_types: parameters.iter().map(|p| p.get_type(ctx)).collect(),
                    output_type: body.get_type(ctx)
                }),
            Term::Application { function, arguments } => {
                if let Term::Arrow { input_types, output_type } = function.get_type(ctx).as_ref() {
                    let mut input_types = input_types.clone();
                    let mut output_type = output_type.clone();

                    for (i, arg) in arguments.iter().enumerate() {
                        let (types_before, types_after) = input_types.split_at_mut(i+1);
                        // dtype is ignored since we here assume that arguments have the expected types.
                        if let Term::Declaration { name, dtype: _ } = types_before[i].as_ref() {
                            let v_ctx = arg.eval(ctx);
                            for j in 0..types_after.len() {
                                types_after[j] = types_after[j].replace(name, &v_ctx).eval(ctx)
                            }
                            output_type = output_type.replace(name, &v_ctx).eval(ctx)
                        }
                    }

                    if arguments.len() == input_types.len() {
                        return output_type;
                    } else {
                        return Rc::new(Term::Arrow { input_types, output_type });
                    }
                }
                panic!()
            }
        }
    }

    pub fn eval(self: &Rc<Term>, ctx: &Context) -> Rc<Term> {
        match self.as_ref() {
            Term::Declaration { name, dtype } => {
                Rc::new(Term::Declaration { name: name.clone(),
                                            dtype: dtype.eval(ctx) })
            },
            Term::Atom { name } => {
                if let Some(def) = ctx.lookup(&name) {
                    match &def.value {
                        Some(value) => value.clone(),
                        None => self.clone(),
                    }
                } else {
                    self.clone()
                }
            },
            Term::Arrow { input_types: _, output_type: _ } => self.clone(),
            Term::Lambda { parameters: _, body: _ } => self.clone(),
            Term::Application { function, arguments } => {
                let f = function.eval(ctx);

                if let Term::Lambda { parameters, body } = f.as_ref() {
                    let mut p = parameters.clone();
                    let mut b = body.clone();

                    for (i, a) in arguments.iter().enumerate() {
                        if let Term::Declaration { name, dtype } = parameters[i].as_ref() {
                            let arg = a.eval(ctx);
                            b = b.replace(name, &arg);

                            for j in (i+1)..parameters.len() {
                                p[j] = p[j].replace(name, &arg);
                            }
                        } else {
                            panic!("Lambda parameter should be a Term::Declaration")
                        }
                    }

                    let remaining = &p[arguments.len()..];
                    if remaining.len() > 0 {
                        return Rc::new(Term::Lambda { parameters: remaining.to_vec(), body: b });
                    }
                    return b;
                } else if let Term::Application { function: function, arguments: first_args } = f.as_ref() {
                    let mut v = first_args.clone();
                    v.append(&mut arguments.clone());
                    return Rc::new(Term::Application { function: function.clone(), arguments: v });
                }

                Rc::new(Term::Application {
                    function: function.clone(),
                    arguments: arguments.clone().iter().map(|v| v.eval(ctx)).collect()
                })
            }
        }
    }

    pub fn replace(self: &Rc<Term>, r_name: &String, r_value: &Rc<Term>) -> Rc<Term> {
        match self.as_ref() {
            Term::Declaration { name, dtype } => {
                if name == r_name {
                    return self.clone()
                }
                Rc::new(Term::Declaration {
                    name: name.clone(),
                    dtype: dtype.replace(r_name, r_value),
                })
            },
            Term::Atom { name } => {
                if name == r_name {
                    r_value.clone()
                } else {
                    self.clone()
                }
            },
            Term::Arrow { input_types, output_type } => {
                Rc::new(Term::Arrow {
                    input_types: input_types.clone().iter().map(|v| v.replace(r_name, r_value)).collect(),
                    output_type: output_type.replace(r_name, r_value),
                })
            },
            Term::Lambda { parameters, body } => {
                // If any lambda parameters shadow the substitution, then return self unchanged.
                for p in parameters.iter() {
                    if let Term::Declaration { name, dtype: _, } = p.as_ref() {
                        if name == r_name {
                            return self.clone()
                        }
                    }
                }
                // Otherwise, substitute in each declaration (r_name might occur in parameter types)
                // and in the body.
                let mut parameters = parameters.clone();
                for i in 0..parameters.len() {
                    parameters[i] = parameters[i].replace(r_name, r_value);
                }
                Rc::new(Term::Lambda { parameters, body: body.replace(r_name, r_value) })
            },
            Term::Application { function, arguments } => {
                Rc::new(Term::Application {
                    function: function.replace(r_name, r_value),
                    arguments: arguments.clone().iter().map(|a| a.replace(r_name, r_value)).collect()
                })
            },
        }
    }
}

impl FromStr for Term {
    type Err = PestError<Rule>;

    fn from_str(s : &str) -> Result<Term, PestError<Rule>> {
        let root = TermParser::parse(Rule::term, s)?.next().unwrap();
        Ok(Rc::try_unwrap(parse_term(root)).unwrap())
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Atom { name } => write!(f, "{}", name),
            Term::Declaration { name, dtype } => write!(f, "{} : {}", name, dtype),
            Term::Arrow { input_types, output_type } => {
                write!(f, "[")?;
                for t in input_types.iter() {
                    match t.as_ref() {
                        Term::Declaration { name, dtype } => write!(f, "({} : {}) -> ", name, dtype),
                        _ => write!(f, "{} -> ", t),
                    }?;
                }
                write!(f, "{}]", output_type)
            },
            Term::Application { function, arguments } => {
                write!(f, "({}", function)?;
                for a in arguments.iter() {
                    write!(f, " {}", a)?;
                }
                write!(f, ")")
            },
            Term::Lambda { parameters, body } => {
                write!(f, "lambda (")?;
                for (i, p) in parameters.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") {}", body)
            }
        }
    }
}

pub mod tests {
    #![allow(unused_imports)]
    use std::rc::Rc;
    use crate::universe::term::{Context, Term, Definition};

    #[test]
    fn build_context() {
        let mut c = Context::new();

        let atom = Rc::new(Term::Atom { name: "asdf".to_string() });
        let def = Definition::new_opaque(atom);

        c.define("x".to_string(), def.clone());
        assert_eq!(c.lookup(&"x".to_string()), Some(&def));
        assert_eq!(c.lookup(&"y".to_string()), None);
    }

    #[test]
    fn parse_and_format_terms() {
        let tests = [
            &"x",
            &"(s n)",
            &"lambda (n : nat, f : [nat -> nat -> nat]) (f n (f (s n) (s n)))",
            &"lambda (n : nat, f : [nat -> [nat -> prop] -> prop]) (f n lambda (k : nat) (is_odd k))",
        ];
        for &s in tests {
            let result: Result<Term, _> = s.parse();
            assert!(result.is_ok());
            let s2 = format!("{}", result.unwrap());
            assert_eq!(s, s2.as_str());
        }
    }

    #[test]
    fn parse_simple_context() {
        let result: Result<Context, _> = (concat!("nat : type. ",
                                                  "\n\n /* Some comment */ ",
                                                  "   leq : [nat -> nat -> /* hello */ prop]. ",
                                                  "leq_z : [nat -> prop] = lambda (n : nat) (leq z n).")
                                          .parse());
        assert!(result.is_ok());
        let context = result.unwrap();

        // 3 declared above, plus `type`.
        assert_eq!(context.size(), 3 + 1);
        assert!(context.lookup(&"type".to_string()).is_some());

        assert!(context.lookup(&"nat".to_string()).is_some());
        assert!(context.lookup(&"nat".to_string()).unwrap().value.is_none());

        assert!(context.lookup(&"leq".to_string()).is_some());
        assert!(context.lookup(&"leq".to_string()).unwrap().value.is_none());

        assert!(context.lookup(&"leq".to_string()).is_some());
        assert!(context.lookup(&"leq_z".to_string()).unwrap().value.is_some());

        assert!(context.lookup(&"other".to_string()).is_none());
    }
}
