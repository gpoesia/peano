use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::Python;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::universe::{Universe, Definition};
use crate::domain::{new_rng, Domain, Equations, Blank};

#[pyclass(unsendable)]
struct PyUniverse {
    pub universe: Universe,
    pub domain: Arc<dyn Domain>,
    pub start_state: String,
}

#[pyclass(unsendable)]
struct PyDefinition {
    pub def: Definition,
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
            .into_iter().map(|d| PyDefinition { def: d }).collect()
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
        map.insert("equations", Arc::new(Equations::new_ct()));
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
    m.add_class::<PyUniverse>()?;
    m.add_class::<PyDefinition>()?;
    m.add_function(wrap_pyfunction!(domains, m)?)?;
    m.add_function(wrap_pyfunction!(get_domain, m)?)?;
    Ok(())
}
