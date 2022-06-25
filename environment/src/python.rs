use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::Python;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::universe::{Context, Universe, EGraphUniverse, Derivation, Definition};
use crate::domain::{new_rng, Domain, Blank};

#[pyclass(unsendable)]
struct PyUniverse {
    pub universe: EGraphUniverse,
    pub domain: Arc<dyn Domain>,
    pub start_state: String,
}

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

    pub fn clean_str(&self, u: &PyUniverse) -> String {
        format!(
            "{} : {}",
            match &self.def.value {
                None => String::from("_"),
                Some(v) => v.in_context(&u.universe.context()).to_string()
            },
            self.def.dtype.in_context(&u.universe.context()).to_string()
        )
    }

    pub fn generating_action(&self) -> &str {
        &self.action
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
impl PyUniverse {
    pub fn starting_state(&self) -> &String {
        &self.start_state
    }

    pub fn actions(&self) -> Vec<String> {
        self.universe.actions().map(|a| a.clone()).collect()
    }

    pub fn apply(&self, action: String) -> Vec<PyDefinition> {
        self.universe.application_results(&action)
            .into_iter().map(|d| PyDefinition { def: d, action: action.clone() }).collect()
    }

    pub fn define(&mut self, name: String, d: &PyDefinition) {
        self.universe.define(name, d.def.clone(), true);
        self.universe.rebuild();
    }

    pub fn reward(&self) -> bool {
        self.domain.reward(&self.universe)
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

    pub fn are_equivalent(&mut self, lhs: &str, rhs: &str) -> PyResult<bool> {
        match (lhs.parse(), rhs.parse()) {
            (Ok(t1), Ok(t2)) => Ok(self.universe.are_equivalent(&t1, &t2)),
            (Err(e), _) => Err(PyValueError::new_err(format!("Failed to parse {}: {}", lhs, e))),
            (_, Err(e)) => Err(PyValueError::new_err(format!("Failed to parse {}: {}", rhs, e))),
        }
    }

    pub fn clone(&self) -> PyUniverse {
        PyUniverse {
            universe: self.universe.clone(),
            start_state: self.start_state.clone(),
            domain: self.domain.clone(),
        }
    }

    pub fn state(&self, ignore: Option<HashSet<String>>) -> Vec<(Vec<String>, String)> {
        let mut summary = self.universe.context_summary();
        match ignore {
            Some(v) => {
                for (objs, _) in summary.iter_mut() {
                    objs.retain(|s| !v.contains(s));
                }
                summary.retain(|s| s.0.len() > 0);
                summary
            },
            None => summary,
        }
    }

    pub fn random_rollout(&mut self, actions: Vec<String>, n_actions: u32, seed: u64) -> bool {
        let mut rng: Pcg64 = new_rng(seed);

        for i in 0..n_actions {
            let actions: Vec<&String> = if actions.len() > 0 {
                actions.iter().collect()
            } else{
                self.universe.actions().collect()
            };

            let j = rng.gen_range(0..actions.len());

            let results = self.universe.application_results(&actions[j]);

            if results.len() == 0 {
                continue;
            }

            let j = rng.gen_range(0..results.len());
            self.universe.define(format!("r{}", i), results[j].clone(), true);
            self.universe.rebuild();
        }

        self.reward()
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

    pub fn apply(&self, action: String) -> Vec<PyDefinition> {
        self.universe.application_results(&action)
            .into_iter().map(|d| PyDefinition { def: d, action: action.clone() }).collect()
    }

    pub fn apply_all_with(&self, actions: Vec<String>, param_name: String) -> Vec<PyDefinition> {
        let mut result = Vec::new();
        for a in actions {
            result.extend(
                self.universe.apply_with(&a, &param_name)
                             .into_iter().map(|d| PyDefinition { def: d, action: a.clone() }));
        }
        result
    }

    pub fn apply_with(&self, action: String, param_name: String) -> Vec<PyDefinition> {
        self.universe.apply_with(&action, &param_name)
            .into_iter().map(|d| PyDefinition { def: d, action: action.clone() }).collect()
    }

    pub fn define(&mut self, name: String, d: &PyDefinition) -> Vec<String> {
        self.universe.define(name, d.def.clone(), false)
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

    pub fn state(&self, ignore: Option<HashSet<String>>) -> Vec<(String, String, bool, Vec<String>)> {
        let mut s = Vec::new();

        for name in self.universe.context_.insertion_order.iter() {
            if ignore.as_ref().map_or(false, |s| s.contains(name)) {
                continue;
            }

            let def = self.universe.context_.lookup(name).unwrap();

            let value = value_of(&def, &self.universe.context_);

            let dependencies = match &def.value {
                Some(v) => v.free_atoms().iter().map(|v| v.clone()).collect(),
                None => vec![],
            };

            s.push((name.clone(), value, def.is_prop(&self.universe.context_), dependencies));
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

    pub fn generate(&mut self, seed: u64) -> PyUniverse {
        let (u, s) = self.domain.generate(seed);
        PyUniverse { universe: u, start_state: s, domain: self.domain.clone() }
    }
}

thread_local!{
    pub static DOMAINS: HashMap<&'static str, Arc<dyn Domain>> = {
        let mut map : HashMap<&'static str, Arc<dyn Domain>> = HashMap::new();
        map.insert("blank", Arc::new(Blank::new()));
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
    m.add_class::<PyUniverse>()?;
    m.add_class::<PyDefinition>()?;
    m.add_class::<PyDerivation>()?;
    m.add_function(wrap_pyfunction!(domains, m)?)?;
    m.add_function(wrap_pyfunction!(get_domain, m)?)?;
    Ok(())
}
