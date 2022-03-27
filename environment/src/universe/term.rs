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
use smallset::SmallSet;

use egg::{RecExpr, SymbolLang};

const TYPE: &str = "type";
const PROP: &str = "prop";

// Prefix added to the names of all lambda and arrow parameters.
// This is used to simplify the test of whether a term has free variables.
const PARAMETER_PREFIX: &str = "$";
type VarSet = SmallSet<[String; 5]>;

pub struct Context {
    type_const: Rc<Term>, // The global constant `type` used to define other types.
    prop_const: Rc<Term>, // The global constant `prop,` which is a type used to define propositions.
    definitions: HashMap<String, Vec<Definition>>, // Map of names to definitions.
    pub insertion_order: Vec<String>, // Order in which definitions were added.
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
        let prop_const = Rc::new(Term::Atom {name: PROP.to_string() });
        let type_const_def = Definition { dtype: type_const.clone(), value: None };
        let mut c = Context { definitions: HashMap::new(),
                              insertion_order: Vec::new(),
                              type_const: type_const.clone(),
                              prop_const: prop_const.clone(),
                              arrows: HashSet::new() };
        c.definitions.insert(TYPE.to_string(), vec![type_const_def.clone()]);
        c.definitions.insert(PROP.to_string(), vec![type_const_def]);
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
            self.insertion_order.push(name.clone());
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
            self.insertion_order.push(name.clone());
            self.definitions.insert(name, vec!(def));
        }
    }

    // Sets a new value to a previously defined name.
    // FIXME This should not change its type, although that is not checked here.
    pub fn set(&mut self, name: &String, value: Rc<Term>) {
        if let Some(v) = self.definitions.get_mut(name) {
            let idx = v.len() - 1;
            v[idx].value = Some(value);
        }
    }

    pub fn actions(&self) -> Iter<'_, String> {
        self.arrows.iter()
    }

    pub fn destroy(&mut self, name: &String) {
        /* if let Some(v) = self.definitions.get_mut(name) {
            if v.len() > 1 {
                v.pop();
                return;
            }
        } */
        self.definitions.remove(name);
    }

    // Removes objects from insertion_order that have been destroied, to speed-up iteration.
    pub fn rebuild(&mut self) {
        self.insertion_order = self.insertion_order
                                   .drain(0..)
                                   .filter(|p| self.definitions.contains_key(p))
                                   .collect();
    }

    pub fn get_type_constant(&self) -> &Rc<Term> {
        &self.type_const
    }

    pub fn get_prop_constant(&self) -> &Rc<Term> {
        &self.prop_const
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
            let mut children : Vec<Rc<Term>> = sub.drain(0..)
                                                  .map(|p| parse_term(p, &mut HashMap::new()))
                                                  .collect();
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

        for name in self.insertion_order.iter() {
            if let Some(def) = self.lookup(&name) {
                s += &format!("{} : {}", name, def.dtype);

                if let Some(val) = &def.value {
                    s += &format!(" = {}", val.to_string());
                }

                s.push_str(".\n")
            }
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

fn rename_param_declaration(t: &mut Rc<Term>) -> Option<String> {
    let (name, dtype) = match t.as_ref() {
        Term::Declaration { name, dtype } => (name.clone(), dtype.clone()),
        _ => { return None },
    };

    *t = Rc::new(Term::Declaration { name: format!("{}{}", PARAMETER_PREFIX, name),
                                     dtype: dtype });

    Some(name)
}

fn parse_term(pair: Pair<Rule>, decls: &mut HashMap<String, usize>) -> Rc<Term> {
    let rule = pair.as_rule();
    let s = pair.as_str();
    let mut sub : Vec<Pair<Rule>> = pair.into_inner().collect();

    match rule {
        Rule::lambda => {
            let mut params : Vec<Rc<Term>> = Vec::new(); // sub.drain(0..sub.len() - 1).map(|p| parse_term(p, decls)).collect();
            let mut param_names : Vec<String> = Vec::new();

            for s in sub.drain(0..sub.len() - 1) {
                let mut p = parse_term(s, decls);
                if let Some(name) = rename_param_declaration(&mut p) {
                    param_names.push(name.clone());
                    *decls.entry(name).or_insert(0) += 1;
                }
                params.push(p);
            }

            let body = parse_term(sub.pop().unwrap(), decls);

            // Append special prefix to all lambda parameters.
            let t = Rc::new(Term::Lambda {
                parameters: params,
                body: body,
            });

            for p in param_names {
                *decls.get_mut(&p).unwrap() -= 1;
            }

            t
        },
        Rule::declaration => {
            let mut children : Vec<Rc<Term>> = sub.drain(0..).map(|p| parse_term(p, decls)).collect();
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
            Rc::new(Term::Atom { name: format!("{}{}",
                                               if *decls.get(s).unwrap_or(&0) > 0 { PARAMETER_PREFIX }
                                               else { "" },
                                               s) })
        },
        Rule::arrow => {
            let mut input_types : Vec<Rc<Term>> = Vec::new();
            let mut param_names : Vec<String> = Vec::new();

            for s in sub.drain(0..sub.len() - 1) {
                let mut p = parse_term(s, decls);
                if let Some(name) = rename_param_declaration(&mut p) {
                    param_names.push(name.clone());
                    *decls.entry(name).or_insert(0) += 1;
                }
                input_types.push(p);
            }

            let output_type = parse_term(sub.pop().unwrap(), decls);

            for p in param_names {
                *decls.get_mut(&p).unwrap() -= 1;
            }

            Rc::new(Term::Arrow { input_types, output_type })
        },
        Rule::application => {
            let arguments : Vec<Rc<Term>> = sub.drain(1..).map(|p| parse_term(p, decls)).collect();
            let function = parse_term(sub.pop().unwrap(), decls);
            Rc::new(Term::Application { function, arguments })
        },
        _ => unreachable!(),
    }
}

impl Term {
    pub fn rc(&self) -> Rc<Term> {
        Rc::new(self.clone())
    }

    pub fn free_variables(self: &Rc<Term>) -> VarSet {
        match self.as_ref() {
            Term::Declaration { name: _, dtype } => dtype.free_variables(),
            Term::Atom { name } => {
                if name.starts_with(PARAMETER_PREFIX) {
                    SmallSet::from_iter([name.clone()])
                } else {
                    SmallSet::new()
                }
            },
            Term::Application { function, arguments } => {
                let mut s = function.free_variables();
                for a in arguments {
                    for p in a.free_variables().iter() {
                        s.insert(p.clone());
                    }
                }
                s
            },
            Term::Arrow { input_types, output_type } => {
                let mut s = output_type.free_variables();
                for t in input_types {
                    if let Term::Declaration { name, dtype: _ } = t.as_ref() {
                        s.remove(&name);
                    }
                }
                s
            },
            Term::Lambda { parameters, body } => {
                let mut s = body.free_variables();
                for t in parameters {
                    if let Term::Declaration { name, dtype: _ } = t.as_ref() {
                        s.remove(&name);
                    }
                }
                s
            }
        }
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
                        if let Term::Declaration { name, dtype: _ } = parameters[i].as_ref() {
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
                } else if let Term::Application { function, arguments: first_args } = f.as_ref() {
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

    pub fn extract_equality(self: &Rc<Term>) -> Option<(Rc<Term>, Rc<Term>)> {
        if let Term::Application { function, arguments } = self.as_ref() {
            if let Term::Atom { name } = function.as_ref() {
                if name == "=" {
                    assert_eq!(arguments.len(), 2,
                               "Equality used as a non-binary relation.");
                    return Some((arguments[0].clone(), arguments[1].clone()));
                }
            }
        }
        None
    }

    pub fn make_equality_object(t1: Rc<Term>, t2: Rc<Term>) -> Rc<Term> {
        Rc::new(
            Term::Application {
                function: Rc::new(Term::Atom { name: "=".to_string() }),
                arguments: vec![t1.clone(), t2.clone()],
            })
    }
}

impl FromStr for Term {
    type Err = PestError<Rule>;

    fn from_str(s : &str) -> Result<Term, PestError<Rule>> {
        let root = TermParser::parse(Rule::term, s)?.next().unwrap();
        Ok(Rc::try_unwrap(parse_term(root, &mut HashMap::new())).unwrap())
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Atom { name } => write!(f, "{}",
                                          if name.starts_with(PARAMETER_PREFIX) { &name[0..] }
                                          else { &name[0..] }),
            Term::Declaration { name, dtype } => write!(f, "{} : {}", &name[0..], dtype),
            Term::Arrow { input_types, output_type } => {
                write!(f, "[")?;
                for t in input_types.iter() {
                    match t.as_ref() {
                        Term::Declaration { name, dtype } => write!(f, "({} : {}) -> ", &name[0..], dtype),
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

impl Term {
    pub fn to_pattern(&self) -> String {
        let mut s = String::new();
        self.write_pattern_string(&mut s);
        s
    }

    pub fn to_recexpr(&self) -> RecExpr<SymbolLang> {
        // This is a naïve implementation, but works for now.
        RecExpr::from_str(self.to_pattern().as_str()).unwrap()
    }

    fn write_pattern_string(&self, s: &mut String) {
        match self {
            Term::Atom { name } => { s.push_str(name.as_str()); },
            Term::Declaration { name, dtype } => {
                s.push_str("($is ");
                s.push_str(name.as_str());
                s.push_str(" ");
                dtype.write_pattern_string(s);
                s.push_str(")");
            },
            Term::Arrow { input_types, output_type } => {
                s.push_str("($arrow");
                for t in input_types.iter() {
                    match t.as_ref() {
                        Term::Declaration { name, dtype } => {
                            s.push_str(" ($param ");
                            s.push_str(name.as_str());
                            s.push_str(" ");
                            dtype.write_pattern_string(s);
                            s.push_str(")");
                        },
                        _ => {
                            s.push_str(" ");
                            t.write_pattern_string(s);
                        },
                    };
                }
                s.push_str(" ");
                output_type.write_pattern_string(s);
                s.push_str(")");
            },
            Term::Application { function, arguments } => {
                s.push_str("($app ");
                function.write_pattern_string(s);
                for a in arguments.iter() {
                    s.push_str(" ");
                    a.write_pattern_string(s);
                }
                s.push_str(")");
            },
            Term::Lambda { parameters, body } => {
                s.push_str("($lambda");
                for p in parameters.iter() {
                    s.push_str(" ");
                    p.write_pattern_string(s);
                }
                s.push_str(" ");
                body.write_pattern_string(s);
                s.push_str(")");
            },
        }
    }
}

pub fn is_parameter_name(name: &str) -> bool {
    return name.starts_with(PARAMETER_PREFIX);
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
