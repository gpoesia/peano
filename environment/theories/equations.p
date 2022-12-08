real : type.

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
*_comm : [((* 'a 'b) : real) -> (= (* 'a 'b) (* 'b 'a))].

/* Associativity */
+_assoc_l : [((+ (+ 'a 'b) 'c) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].
+_assoc_r : [((+ 'a (+ 'b 'c)) : real) -> (= (+ 'a (+ 'b 'c)) (+ (+ 'a 'b) 'c))].

-+_assoc : [((- (+ 'a 'b) 'c) : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].
+-_assoc : [((+ (- 'a 'b) 'c) : real) -> (= (+ (- 'a 'b) 'c) (+ 'a (- 'c 'b)))].

*/_assoc_r : [((/ (* 'a 'b) 'c) : real) -> (= (/ (* 'a 'b) 'c) (* 'a (/ 'b 'c)))].
*/_assoc_l : [((* 'a (/ 'b 'c)) : real) -> (= (* 'a (/ 'b 'c)) (/ (* 'a 'b) 'c))].

/* Distributivity */
+*_dist_l : [((+ (* 'a 'c) (* 'b 'c)) : real) -> (= (+ (* 'a 'c) (* 'b 'c)) (* (+ 'a 'b) 'c))].
-*_dist_l : [((- (* 'a 'c) (* 'b 'c)) : real) -> (= (- (* 'a 'c) (* 'b 'c)) (* (- 'a 'b) 'c))].

/* Cancellation axioms */
+0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].
-0_id : [((- 'a 0) : real) -> (= (- 'a 0) 'a)].
*1_id : [((* 'a 1) : real) -> (= (* 'a 1) 'a)].
/1_id : [((/ 'a 1) : real) -> (= (/ 'a 1) 'a)].
div_self_id : [((/ 'a 'a) : real) -> (= (/ 'a 'a) 1)].
-self_null: [((- 'a 'a) : real) -> (= (- 'a 'a) 0)].

*0_null : [((* 'a 0) : real) -> (= (* 'a 0) 0)].
0_div_null : [((/ 0 'a) : real) -> (= (/ 0 'a) 0)].

verify simple_xy_eqs {
    let x : real.
    let y : real.
    assume (= (+ x 3) 10).
    assume (= y (* x x)).

    show (= (- (+ x 3) 3) (- 10 3)) by sub_eq.
    show (= (- (+ x 3) 3) (+ x (- 3 3))) by -+_assoc.
    show (= (+ x (- 3 3)) (- 10 3)) by rewrite.
    show (= (- 3 3) 0) by eval.
    show (= (- 10 3) 7) by eval.
    show (= (+ x 0) (- 10 3)) by rewrite.
    show (= (+ x 0) x) by +0_id.
    show (= x (- 10 3)) by rewrite.
    show (= x 7) by rewrite.

    show (= y (* 7 x)) by rewrite.
    show (= y (* 7 7)) by rewrite.
    show (= (* 7 7) 49) by eval.
    show (= y 49) by rewrite.
}

verify comb_like_example_1 {
    let x : real. let answer : real.
    assume (= answer (- (+ 3 x) 1)).
    show (= (+ 3 x) (+ x 3)) by +_comm.
    show (= answer (- (+ x 3) 1)) by rewrite.
    show (= (- (+ x 3) 1) (+ x (- 3 1))) by -+_assoc.
    show (= answer (+ x (- 3 1))) by rewrite.
    show (= (- 3 1) 2) by eval.
    show (= answer (+ x 2)) by rewrite.
}

verify comb_like_example_2 {
    let x : real. let answer : real.
    assume (= answer (- (+ x 9) 9)).
    show (= (- (+ x 9) 9) (+ x (- 9 9))) by -+_assoc.
    show (= answer (+ x (- 9 9))) by rewrite.
    show (= (- 9 9) 0) by eval.
    show (= answer (+ x 0)) by rewrite.
    show (= (+ x 0) x) by +0_id.
    show (= answer x) by rewrite.
}
