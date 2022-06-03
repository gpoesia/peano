// A domain that just gives access to the Peano engine without pre-defining
// rewards or a space of problems.
// This is used as the base domain to implement domains in Python instead.

use crate::universe::{EGraphUniverse};

use super::Domain;

pub struct Blank {}

impl Blank {
    pub fn new() -> Blank {
        Blank {}
    }
}

impl Domain for Blank {
    fn name(&self) -> String {
        String::from("blank")
    }

    fn generate(&self, _seed: u64) -> (EGraphUniverse, String) {
        (EGraphUniverse::new(), String::new())
    }

    fn size(&self) -> u64 {
        1
    }

    fn reward(&self, _u: &EGraphUniverse) -> bool {
        false
    }
}
