/*
  Simple examples of propositional logic in Peano.

  This uses arrows to represent implication, and modus ponens becomes application of
  the arrow as an action.

  This construction makes proofs by cases unwieldy since the current action space
  does not provide a clean way to derive new arrows unless those are given as outputs
  of existing arrows. A cleaner Peano representation might be to have Implies as a
  declared proposition type, then have an explicit 'modus_ponens' rule.
*/

A : prop.
B : prop.
C : prop.

A_implies_B : [A -> B].
B_implies_C : [B -> C].

verify a_implies_c {
    assume A.
    show B by A_implies_B.
    show C by B_implies_C.
}

and : [prop -> prop -> prop].
or : [prop -> prop -> prop].

false : prop.
not : [prop -> prop].
contradiction : [(P : prop) -> P -> (not P) -> false].

verify contradiction_example {
    assume A.
    assume (not A).
    show false by contradiction.
}

and_cons : [(A : prop) -> (B : prop) -> A -> B -> (and A B)].
and_l : [(A : prop) -> (B : prop) -> (and A B) -> A].
and_r : [(A : prop) -> (B : prop) -> (and A B) -> B].

implies_cons : [(A : prop) -> A -> (B : prop) -> [B -> A]].

or_l : [(A : prop) -> (B : prop) -> A -> (or A B)].
or_r : [(A : prop) -> (B : prop) -> B -> (or A B)].

or_cases : [(A : prop) -> (B : prop) -> (C : prop) -> [A -> C] -> [B -> C] -> (or A B) -> C].

verify and_is_commutative {
    assume (and A B).
    show A by and_l.
    show B by and_r.
    show (and B A) by and_cons.
}

verify and_is_associative {
    assume (and A (and B C)).
    show A by and_l.
    show (and B C) by and_r.
    show B by and_l.
    show C by and_r.
    show (and A B) by and_cons.
    show (and (and A B) C) by and_cons.
}

verify proof_by_cases {
    /* FIXME: Does not work if we replace C with a compund proposition (e.g., (or C C)). */
    assume [A -> C].
    assume [B -> C].
    assume (or A B).
    show C by or_cases.
}

verify and_distributes_over_or {
    assume (and A (or B C)).
    show A by and_l.
    show (or B C) by and_r.

    let b_and : [B -> (and A B)] = lambda (b : B) (and_cons A B A b).
    let c_and : [C -> (and A C)] = lambda (c : C) (and_cons A C A c).

    let b_or : [B -> (or (and A B) (and A C))]. /* = lambda (b : B) (or_l (and A B)
                                                                      (and B C)
                                                                      (b_and b)). */
    let c_or : [C -> (or (and A B) (and A C))]. /* = lambda (c : C) (or_r (and A B)
                                                                      (and B C)
                                                                      (c_and c)). */
    show (or (and A B) (and A C)) by or_cases.
    /* Need to manually construct too many objects. */
}

verify a_implies_a {
    assume A.
    show [A -> A] by implies_cons.
}
