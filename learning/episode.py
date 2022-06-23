from dataclasses import dataclass

@dataclass
class ProofSearchEpisode:
    success: bool
    iterations: int
    steps_added: int
    steps_created: int
    problem: str
    solution: list[str]
    visited_negatives: list[str]
    discovered_negatives: list[str]
