real : type.

= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].
/ : [real -> real -> real].

/* Commutativity */
+_comm : [(a : real) -> (b : real) -> (= (+ a b) (+ b a))].
*_comm : [(a : real) -> (b : real) -> (= (* a b) (* b a))].

/* Associativity */
+_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (+ a (+ b c)) (+ (+ a b) c))].
+-_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (- (+ a b) c) (+ a (- b c)))]. /* (a + b) - c = a + (b - c) */
*/_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (/ (* a b) c) (* a (/ b c)))]. /* (a * b) / c = a * (b / c) */

/* Distributivity */
+*_dist : [(a : real) -> (b : real) -> (c : real) -> (= (* (+ a b) c) (+ (* a c) (* b c)))].

/* Cancellation axioms */
+0_id : [(a : real) -> (= (+ a 0) a)].
-0_id : [(a : real) -> (= (- a 0) a)].
*1_id : [(a : real) -> (= (* a 1) a)].
/1_id : [(a : real) -> (= (/ a 1) a)].
div_self_id : [(a : real) -> (!= a 0) -> (= (/ a a) 1)].

*0_null : [(a : real) -> (= (* a 0) 0)].
0_div_null : [(a : real) -> (!= a 0) -> (= (/ 0 a) 0)].

let 3 : real.
let 10 : real.
let !sub1 : real = (+ x 3).
let !sub2 : real = (- 3 3).
assume (= (+ x 3) 10).

verify simple_xy_eqs {
    let x : real.
    let y : real.
    assume (= (+ x 3) 10).
    assume (= y (* x x)).

    construct (- (+ x 3) 3) by -.
    show (= (- (+ x 3) 3) 7) by eval.
    show (= (+ x (- 3 3)) 7) by +-_assoc.
    show (= (- 3 3) 0) by eval.
    show (= x 7) by +0_id.
    show (= y 49) by eval.
}

verify eq_with_div {
    assume (= (* 2 x) 10).

    construct (/ (* 2 x) 2) by /.
    show (= (* x 2) (* 2 x)) by *_comm.
    show (= (/ (* x 2) 2) (* x (/ 2 2))) by */_assoc.
    show (= (/ 2 2) 1) by eval.
    show (= (/ 10 2) 5) by eval.
    show (= x 5) by *1_id.
}

verify eq_with_nonzero_assumption {
    let x : real.
    assume (!= x (- 5 5)).
    assume (= (/ 10 x) 5).

    show (= (- 5 5) 0) by eval. /* Implicitly gives (!= x 0) */

    construct (* (/ 10 x) x) by *.
    show (= (* (/ 10 x) x) (* x (/ 10 x))) by *_comm.
    show (= (* x (/ 10 x)) (/ (* x 10) x)) by */_assoc.
    show (= (* x 10) (* 10 x)) by *_comm.
    show (= (/ (* 10 x) x) (* 10 (/ x x))) by */_assoc.

    /* This fails if we remove the assumption that x is not zero. */
    show (= (/ x x) 1) by div_self_id.
    show (= (* 5 x) 10) by *1_id.

    /* Solution now follows just like eq_with_div */
    construct (/ (* 5 x) 5) by /.
    show (= (* x 5) (* 5 x)) by *_comm.
    show (= (/ (* x 5) 5) (* x (/ 5 5))) by */_assoc.
    show (= (/ 5 5) 1) by eval.
    show (= (/ 10 5) 2) by eval.
    show (= x 2) by *1_id.
}
