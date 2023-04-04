mod domain;
mod blank;

pub use domain::*;
pub use blank::*;

use rand_pcg::Pcg64;

pub fn new_rng(seed: u64) -> Pcg64 {
    Pcg64::new((0xcafef00dd15ea5e5 + seed).into(),
               0xa02bdbf7bb3c0a7ac28fa16a64abf96)
}
