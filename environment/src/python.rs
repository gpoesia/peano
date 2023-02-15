use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::Python;

use crate::universe::{Context, Universe, Derivation, Definition, Term};
use crate::domain::{Domain, Blank, Equations};

#[pyclass(unsendable)]
struct PyDerivation {
    pub universe: Derivation,
}

#[pyclass(unsendable)]
struct PyDefinition {
    pub def: Definition,
    pub action: String,
}

#[pyclass(unsendable)]
struct PyDomain {
    pub domain: Arc<dyn Domain>
}

#[pymethods]
impl PyDefinition {
    pub fn __str__(&self) -> String {
        format!(
            "{} : {}",
            match &self.def.value {
                None => String::from("_"),
                Some(v) => v.to_string()
            },
            self.def.dtype.to_string()
        )
    }

    pub fn generating_action(&self) -> &str {
        &self.action
    }

    pub fn generating_arguments(&self) -> Option<Vec<String>> {
        match &self.def.value {
            Some(v) => match v.as_ref() {
                Term::Application { function: _, arguments } => {
                    Some(arguments.iter().map(|v| v.to_string()).collect())
                },
                _ => None,
            },
            None => None,
        }
    }

    pub fn dependencies(&self) -> Vec<String> {
        match &self.def.value {
            Some(v) => v.free_atoms().iter().map(|v| v.clone()).collect(),
            None => vec![],
        }
    }

    pub fn clean_dtype(&self, u: &PyDerivation) -> String {
        self.def.dtype.in_context(&u.universe.context()).to_string()
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pymethods]
impl PyDerivation {
    #[new]
    pub fn new() -> PyDerivation {
        PyDerivation {
            universe: Derivation::new()
        }
    }

    pub fn actions(&self) -> Vec<String> {
        self.universe.actions().map(|a| a.clone()).collect()
    }

    pub fn apply(&self, action: String, scope: Option<Vec<String>>,
                 predetermined: Option<Vec<Option<String>>>) -> Vec<PyDefinition> {
        self.universe.application_results(&action, &scope.map(|v| v.into_iter().collect()),
                                          &(predetermined.unwrap_or(vec![])))
            .into_iter().map(|d| PyDefinition { def: d, action: action.clone() }).collect()
    }

    pub fn next_id(&mut self) -> usize {
        self.universe.next_term_id()
    }

    pub fn peek_next_id(&self) -> usize {
        self.universe.peek_next_term_id()
    }

    pub fn fast_forward_next_id(&mut self, i: usize) {
        self.universe.fast_forward_next_term_id(i)
    }

    pub fn apply_with(&self, action: String, param_name: String) -> Vec<PyDefinition> {
        self.universe.apply_with(&action, &param_name)
            .into_iter().map(|d| PyDefinition { def: d, action: action.clone() }).collect()
    }

    pub fn define(&mut self, name: String, d: &PyDefinition) -> Vec<String> {
        self.universe.define(name, d.def.clone(), false)
    }

    pub fn lookup(&self, name: String) -> Option<PyDefinition> {
        match self.universe.lookup(&name) {
            None => None,
            Some(def) => Some(PyDefinition { def: def.clone(), action: String::new() })
        }
    }

    pub fn is_prop(&self, d: &PyDefinition) -> bool {
        d.def.is_prop(&self.universe.context_)
    }

    pub fn incorporate(&mut self, context: &str) -> PyResult<bool> {
        match context.parse() {
            Ok(context) => {
                self.universe.incorporate(&context);
                Ok(true)
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to parse context: {}", e)))
        }
    }

    pub fn clone(&self) -> PyDerivation {
        PyDerivation {
            universe: self.universe.clone(),
        }
    }

    pub fn state(&self, ignore: Option<HashSet<String>>) -> Vec<(String, String, Option<String>, bool, Vec<String>)> {
        let mut s = Vec::new();

        for name in self.universe.context_.insertion_order.iter() {
            if ignore.as_ref().map_or(false, |s| s.contains(name)) {
                continue;
            }

            let def = self.universe.context_.lookup(name).unwrap();

            let dependencies = match &def.value {
                Some(v) => v.free_atoms().iter().map(|v| v.clone()).collect(),
                None => vec![],
            };

            s.push((name.clone(),
                    def.dtype.to_string(),
                    def.value.as_ref().map(|v| v.to_string()),
                    def.is_prop(&self.universe.context_),
                    dependencies));
        }

        s
    }

    pub fn value_of(&self, def: &PyDefinition) -> String {
        value_of(&def.def, &self.universe.context_)
    }
}

fn value_of(def: &Definition, ctx: &Context) -> String {
    if def.is_prop(ctx) {
        def.dtype.in_context(ctx).to_string()
    } else {
        match &def.value {
            Some(v) => v.in_context(ctx).to_string(),
            None => String::new()
        }
    }
}

#[pymethods]
impl PyDomain {
    pub fn name(&self) -> String {
        self.domain.name()
    }

    pub fn size(&self) -> u64 {
        self.domain.size()
    }
}

thread_local!{
    pub static DOMAINS: HashMap<&'static str, Arc<dyn Domain>> = {
        let mut map : HashMap<&'static str, Arc<dyn Domain>> = HashMap::new();
        map.insert("blank", Arc::new(Blank::new()));
        map.insert("equations-ct", Arc::new(Equations::new_ct()));
        map.insert("equations-easy", Arc::new(Equations::new_easy()));
        map
    };
}

#[pyfunction]
fn domains() -> Vec<String> {
    DOMAINS.with(|m| {
        m.keys().map(|v| String::from(*v)).collect()
    })
}

#[pyfunction]
fn get_domain(name: String) -> PyResult<PyDomain> {
    DOMAINS.with(|m| {
        match m.get(name.as_str()) {
            Some(v) => Ok(PyDomain { domain: v.clone() }),
            None => Err(PyValueError::new_err(format!("No domain named {}.", name)))
        }
    })
}

#[pymodule]
fn peano(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDefinition>()?;
    m.add_class::<PyDerivation>()?;
    m.add_function(wrap_pyfunction!(domains, m)?)?;
    m.add_function(wrap_pyfunction!(get_domain, m)?)?;
    Ok(())
}
