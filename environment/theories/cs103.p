real : type.

not : [prop -> prop].
contradiction : prop.

p_and_not_p : [('p : prop) -> 'p -> (not 'p) -> contradiction].

= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].
/ : [real -> real -> real].

/* Operate on both sides of an equation */
add_eq : [(= 'a 'b) -> ('c : real) -> (= (+ 'a 'c) (+ 'b 'c))].
sub_eq : [(= 'a 'b) -> ('c : real) -> (= (- 'a 'c) (- 'b 'c))].
mul_eq : [(= 'a 'b) -> ('c : real) -> (= (* 'a 'c) (* 'b 'c))].
div_eq : [(= 'a 'b) -> ('c : real) -> (= (/ 'a 'c) (/ 'b 'c))].

/* Commutativity */
+_comm : [((+ 'a 'b) : real) -> (= (+ 'a 'b) (+ 'b 'a))].

/* Associativity */
+_assoc_l : [((+ (+ 'a 'b) 'c) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].
+_assoc_r : [((+ 'a (+ 'b 'c)) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].

+-_assoc_r : [((- (+ 'a 'b) 'c) : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].
+-_assoc_l : [((+ 'a (- 'b 'c)) : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].

*_assoc_l : [((* (* 'a 'b) 'c) : real) -> (= (* (* 'a 'b) 'c) (* 'a (* 'b 'c)))].
*_assoc_r : [((* 'a (* 'a 'b)) : real) -> (= (* 'a (* 'b 'c)) (* (* 'a 'b) 'c))].

*/_assoc : [((/ (* 'a 'b) 'c) : real) -> (= (/ (* 'a 'b) 'c) (* 'a (/ 'b 'c)))].

/* Distributivity */
+_assoc : [((+ (+ 'a 'b) 'c) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].

+*_dist_r : [((* (+ 'a 'b) 'c) : real) -> (= (* (+ 'a 'b) 'c) (+ (* 'a 'c) (* 'b 'c)))].
+*_dist_l : [((+ (* 'a 'c) (* 'b 'c)) : real) -> (= (* (+ 'a 'b) 'c) (+ (* 'a 'c) (* 'b 'c)))].

/* Cancellation axioms */
+0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].
-0_id : [((- 'a 0) : real) -> (= (- 'a 0) 'a)].
*1_id : [((* 'a 1) : real) -> (= (* 'a 1) 'a)].
/1_id : [((/ 'a 1) : real) -> (= (/ 'a 1) 'a)].
div_self_id : [((/ 'a 'a) : real) -> (= (/ 'a 'a) 1)].
-self_null: [((- 'a 'a) : real) -> (= (- 'a 'a) 0)].

*0_null : [((* 'a 0) : real) -> (= (* 'a 0) 0)].
0_div_null : [((/ 0 'a) : real) -> (= (/ 0 'a) 0)].

/* Being odd or being even are properties of numbers (reals here, but read integer) */
odd : [real -> prop].
even : [real -> prop].

/* Here's their defining property: any number of the form 2*k is even, 2*k + 1 is odd. */
even_def : [((* 2 'k) : real) -> (even (* 2 'k))].
odd_def : [(+ (* 2 'k) 1) -> (odd (+ (* 2 'k) 1))].

/* And vice-versa: if a number is even, then there exists an "eveness witness" */
even_witness : [(even 'n) -> real].
even_witness_def : [('h : (even 'n)) -> (= (* 2 (even_witness 'h)) 'n)].

/* Or an "oddness witness" */
odd_witness : [(odd 'n) -> real].
odd_witness_def : [('h : (odd 'n)) -> (= 'n (+ (* 2 (odd_witness 'h)) 1))].

let square : [real -> real] = lambda (n : real) (* n n).

verify n_even_implies_n_squared_even {
    let n : real.
    let n_even : (even n).

    construct /* k = */ (even_witness n_even) by even_witness.
    show (= (* 2 (even_witness n_even)) n) by even_witness_def.
    show (= n (* 2 (even_witness n_even))) by eq_symm.

    construct (* n n) by *.
    show (= (* n n) (* n n)) by eq_refl.
    show (= (* n n) (* (* 2 (even_witness n_even)) n)) by rewrite.

    show (= (* (* 2 (even_witness n_even)) n) (* 2 (* (even_witness n_even) n))) by *_assoc_l.
    show (= (* n n) (* 2 (* (even_witness n_even) n))) by rewrite.
    show (= (* 2 (* (even_witness n_even) n)) (* n n)) by eq_symm.

    show (even (* 2 (* (even_witness n_even) n))) by even_def.
    show (even (* n n)) by rewrite.

    construct (square n) by square.
    show (= (square n) (* n n)) by eval.
    show (= (* n n) (square n)) by eq_symm.

    show (even (square n)) by rewrite.
} /* Therefore: */ n_even_implies_n_squared_even : [(even 'n) -> (even (square 'n))].
                   n_odd_implies_n_squared_odd : [(odd 'n) -> (odd (square 'n))].

/* Some more axioms */
/* If a number is even, it is not odd. */
even_not_odd : [(even 'n) -> (not (odd 'n))].
odd_not_even : [(odd 'n) -> (not (even 'n))].

verify two_is_even {
    let two : real = 2.

    let one : real = 1.
    construct (* 2 1) by *.
    show (= (* 2 1) 2) by eval.

    show (even (* 2 1)) by even_def.
    show (even 2) by rewrite.
} /* Therefore: */ two_is_even : (even 2).

verify sum_of_even_is_even {
    let a : real.
    let b : real.

    let a_even : (even a).
    let b_even : (even b).

    /* Goal: show that a + b is even. */
}


verify sum_of_odds_is_even {
    let a : real.
    let b : real.

    let a_odd : (odd a).
    let b_odd : (odd b).

    /* Goal: show that a + b is even. */

    /* If a and b a are odd, we can "call" odd_witness to obtain their k s.t. 2k + 1 = a (resp. b) */
    construct (odd_witness a_odd) by odd_witness.
    construct (odd_witness b_odd) by odd_witness.

    /* And get the equality 2*k + 1 = a (resp. b) by the definition of the witness. */
    show (= a (+ (* 2 (odd_witness a_odd)) 1)) by odd_witness_def.
    show (= b (+ (* 2 (odd_witness b_odd)) 1)) by odd_witness_def.

    /* Now rewrite sum to be the sum of these two things. */
    construct (+ a b) by +.
    show (= (+ a b) (+ a b)) by eq_refl.

    show (= (+ a b) (+ (+ (* 2 (odd_witness a_odd)) 1) b)) by rewrite.

    show (= (+ a b) (+ (+ (* 2 (odd_witness a_odd)) 1)
                       (+ (* 2 (odd_witness b_odd)) 1))) by rewrite.
} /* Therefore: */ sum_of_odds_is_even : [(odd 'a) -> (odd 'b) -> (even (+ 'a 'b))].


verify sum_of_squares_one_is_even {
    let a : real.
    let b : real.
    let c : real.

    assume (odd a).
    assume (odd b).
    assume (odd c).

    assume (= (+ (square a) (square b)) (square c)).

    /* If a, b and c are odd, then all of their squares are odd. */
    show (odd (square a)) by n_odd_implies_n_squared_odd.
    show (odd (square b)) by n_odd_implies_n_squared_odd.
    show (odd (square c)) by n_odd_implies_n_squared_odd.

    /* The sum of two odds is even. */
    show (odd (+ (square a) (square b))) by sum_of_odds_is_even.

    /* Then square of c is even. */
    show (even (* c c)) by rewrite.

    /* But if it is even, then it is not odd. */
    show (not (odd (square c))) by even_not_odd.

    /* Since we had (odd (square c)), this is a contradiction.
    show contradiction by p_and_not_p.
    */
}
