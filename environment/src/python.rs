use std::rc::Rc;
use std::sync::Arc;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::universe::{Term, Universe, Definition};
use crate::domain::{Domain, Equations};

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
        map.insert("equations", Arc::new(Equations::new()));
        // map.insert("fractions", Arc::new(Fractions::new(4, 4)));
        // map.insert("ternary-addition", Arc::new(TernaryAddition::new(15)));
        // map.insert("ternary-addition-small", Arc::new(TernaryAddition::new(8)));
        // map.insert("multiplication", Arc::new(Multiplication {}));
        // map.insert("sorting", Arc::new(Sorting::new(12)));

        // map.insert("key-to-door", Arc::new(KeyToDoor::new(5, 0.1)));
        // map.insert("rubiks-cube-20", Arc::new(RubiksCube::new(20)));
        // map.insert("rubiks-cube-50", Arc::new(RubiksCube::new(50)));
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
