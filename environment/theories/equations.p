real : type.

= : [(t : type) -> t -> t -> prop].
+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].

+_comm : [(a : real) -> (b : real) -> (= (+ a b) (+ b a))].
+-_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (- (+ a b) c) (+ a (- b c)))].
+0_id : [(a : real) -> (= (+ a 0) a)].

x : real.
y : real.

x_eq : (= (+ x 3) 10).
y_eq : (= y (* x x)).
