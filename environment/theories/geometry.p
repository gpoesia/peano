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

double_x : [((+ 'x 'x) : real) -> (= (+ 'x 'x) (* 2 'x))].

*0_null : [((* 'a 0) : real) -> (= (* 'a 0) 0)].
0_div_null : [((/ 0 'a) : real) -> (= (/ 0 'a) 0)].

/* Points, triangles and angles are types. */
point : type.
triangle : type.
angle : type.

/* We can construct a triangle from 3 points, similar for an angle. */
tri_pts : [point -> point -> point -> triangle].
angle_pts : [point -> point -> point -> angle].

/* Angles have a measure (in degrees). */
measure : [angle -> real].

/* The sum of the internal angles of a triangle is 180. */
tri_angle_sum : [((tri_pts 'P 'Q 'R) : triangle) ->
                 (= (+ (+ (measure (angle_pts 'P 'Q 'R))
                          (measure (angle_pts 'Q 'R 'P)))
                          (measure (angle_pts 'R 'P 'Q))) 180)].

verify isosceles_triangle_angle {
    let P : point. let Q : point. let R : point.

    let T : triangle = (tri_pts P Q R).
    let apqr : angle = (angle_pts P Q R).
    let aqrp : angle = (angle_pts Q R P).
    let arpq : angle = (angle_pts R P Q).

    let alpha : real = (measure apqr).
    let beta : real = (measure aqrp).
    let gamma : real = (measure arpq).

    assume (= alpha beta). assume (= gamma 100).

    show (= (+ (+ alpha beta) gamma) 180) by tri_angle_sum.
    show (= (+ (+ beta beta) gamma) 180) by rewrite.
    show (= (+ beta beta) (* 2 beta)) by double_x.
    show (= (+ (* 2 beta) gamma) 180) by rewrite.
    show (= (+ (* 2 beta) 100) 180) by rewrite.
    show (= (- (+ (* 2 beta) 100) 100) (- 180 100)) by sub_eq.

    show (= (- (+ (* 2 beta) 100) 100) (+ (* 2 beta) (- 100 100))) by +-_assoc_r.
    show (= (- 100 100) 0) by eval.
    show (= (- (+ (* 2 beta) 100) 100) (+ (* 2 beta) 0)) by rewrite.

    show (= (+ (* 2 beta) 0) (* 2 beta)) by +0_id.
    show (= (- (+ (* 2 beta) 100) 100) (* 2 beta)) by rewrite.
    show (= (* 2 beta) (- 180 100)) by rewrite.
    show (= (- 180 100) 80) by eval.
    show (= (* 2 beta) 80) by rewrite.

    show (= (/ (* 2 beta) 2) (/ 80 2)) by div_eq.
    show (= (* 2 beta) (* beta 2)) by *_comm.
    show (= (/ (* beta 2) 2) (/ 80 2)) by rewrite.
    show (= (/ (* beta 2) 2) (* beta (/ 2 2))) by */_assoc.
    show (= (/ 2 2) 1) by eval.
    show (= (/ (* beta 2) 2) (* beta 1)) by rewrite.
    show (= (* beta 1) beta) by *1_id.
    show (= (/ (* beta 2) 2) beta) by rewrite.
    show (= beta (/ 80 2)) by rewrite.
    show (= (/ 80 2) 40) by eval.

    show (= (measure aqrp) 40) by rewrite.
    show (= beta alpha) by eq_symm.
    show (= (measure apqr) 40) by rewrite.
}
