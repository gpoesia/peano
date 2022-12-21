/* FRACTIONS */

/* Reals (necessary for fractions, copied from equations.p) */

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


/* Fractions */
/* Based on Axioms of a Fraction Domain (Table 5) in Contrastive Reinforcement Learning of Symbolic Reasoning Domains */

/* Note: "factorize" and "eval" are written as the actions "factorize" and "eval_i" in Peano */

/* cancel */
cancel : [(a : real) -> (b: real) -> (c: real) -> (= (/ (* a b) (* a c)) (/ b c))].

/* scale */
scale_2 : [(a : real) -> (b : real) -> (= (/ a b) (/ (* a 2) (* b 2)))].
scale_3 : [(a : real) -> (b : real) -> (= (/ a b) (/ (* a 3) (* b 3)))].
scale_5 : [(a : real) -> (b : real) -> (= (/ a b) (/ (* a 5) (* b 5)))].
scale_7 : [(a : real) -> (b : real) -> (= (/ a b) (/ (* a 7) (* b 7)))].

/* simpl1 */
simpl_1 : [(a : real) -> (= (/ a 1) a)].

/* Note: mfrac is not needed because in Peano it is equivalent to simpl_1 */

/* mul */
mul : [(a : real) -> (b : real) -> (c : real) -> (d : real) -> (= (* (/ a b) (/ c d)) (/ (* a b) (* c d)))].

/* combine */
combine : [(a : real) -> (b : real) -> (c : real) -> (= (+ (/ a c) (/ b c)) (/ (+ a b) c))].
