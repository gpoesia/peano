/*
 * Implementation of Kleene's propositional logic formulation.
 */

implies : [prop -> prop -> prop].

/* Axiom 1a.. */
implies_intro : [(A : prop) -> (B : prop) -> (implies A (implies B A))].

/* Axiom 1b.. */
implies_trans : [(A : prop) -> (B : prop) -> (C : prop) ->
                 (implies (implies A B)
                          (implies (implies A (implies B C))
                                   (implies A C)))].

/* Axiom 2.*/
modus_ponens : [(A : prop) -> (B : prop) -> A -> (implies A B) -> B].

and : [prop -> prop -> prop].
or : [prop -> prop -> prop].

/* Axiom 3 */
and_intro : [(A : prop) -> (B : prop) -> (implies A (implies B (and A B)))].

/* Axioms 4a and 4b. */
and_l : [(A : prop) -> (B : prop) -> (and A B) -> A].
and_r : [(A : prop) -> (B : prop) -> (and A B) -> B].

/* Axioms 5a and 5b. */
or_l : [(A : prop) -> (B : prop) -> A -> (or A B)].
or_r : [(A : prop) -> (B : prop) -> B -> (or A B)].

not : [prop -> prop].

/* Axiom 6. */
or_cases : [(A : prop) -> (B : prop) -> (C : prop) ->
            (implies (implies A C)
                     (implies (implies B C)
                              (implies (or A B) C)))].

/* Axiom 7.. */
contra : [(A : prop) -> (B : prop) -> (implies (implies A B) (implies (implies A (not B)) (not A)))].

/* Axiom 8.. */
notnot_elim : [(A : prop) -> (implies (not (not A)) A)].

/* Example 2, chapter 4. */
verify a_implies_a {
    let P : prop.

    construct (implies P P) by implies. /* Constructs, but does not prove! */
    construct (implies (implies P P) P) by implies.
    show (implies P (implies P P)) by implies_intro.

    show (implies (implies P (implies P P))
                  (implies (implies P (implies (implies P P) P))
                           (implies P P))) by implies_trans.

    show (implies (implies P (implies (implies P P) P))
                           (implies P P)) by modus_ponens.

    show (implies P (implies (implies P P) P)) by implies_intro.
    show (implies P P) by modus_ponens.
}

/* Example 1, chapter 5. */
verify example1_deduction {
    let P : prop.
    let Q : prop.
    let R : prop.

    assume P.
    assume Q.
    assume (implies P (implies Q R)).

    show (implies Q R) by modus_ponens.
    show R by modus_ponens.
}

/* Example 2 */
verify intro_and_implication {
    let P : prop.
    let Q : prop.
    let R : prop.
    assume P.
    assume Q.
    assume (implies (and P Q) R).

    show (implies P (implies Q (and P Q))) by and_intro.
    show (implies Q (and P Q)) by modus_ponens.
    show (and P Q) by modus_ponens.
    show R by modus_ponens.
}
