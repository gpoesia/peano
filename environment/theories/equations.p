real : type.

= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].
/ : [real -> real -> real].

/* Operate on both sides */
add_eq: [(a : real) -> (b : real) -> (= a b) -> (c : real) -> (= (+ a c) (+ b c))].
sub_eq: [(a : real) -> (b : real) -> (= a b) -> (c : real) -> (= (- a c) (- b c))].
mul_eq: [(a : real) -> (b : real) -> (= a b) -> (c : real) -> (= (* a c) (* b c))].
div_eq: [(a : real) -> (b : real) -> (= a b) -> (c : real) -> (= (/ a c) (/ b c))].

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

verify simple_xy_eqs {
    let x : real.
    let y : real.
    assume (= (+ x 3) 10).
    assume (= y (* x x)).

    construct (- (+ x 3) 3) by -.
    show (= (- (+ x 3) 3) (+ x (- 3 3))) by +-_assoc.
    show (= (- 3 3) 0) by eval.
    show (= (- (+ x 3) 3) (+ x 0)) by rewrite.
    show (= (+ x 0) x) by +0_id.
    show (= (- (+ x 3) 3) x) by rewrite.

    show (= (- (+ x 3) 3) (- (+ x 3) 3)) by eq_refl.
    show (= x (- (+ x 3) 3)) by rewrite.
    show (= x (- 10 3)) by rewrite.
    show (= (- 10 3) 7) by eval.
    show (= x 7) by rewrite.
    show (= y (* 7 x)) by rewrite.
    show (= y (* 7 7)) by rewrite.
    show (= (* 7 7) 49) by eval.
    show (= y 49) by rewrite.
}
