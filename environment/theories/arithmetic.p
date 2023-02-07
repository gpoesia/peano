bool : type.
true : bool. false : bool.

nat : type.
real : type.

= : [('a : 't) -> 't -> prop].

/* Natural numbers */
z : nat.
s : [nat -> nat].

/* Natural numbers in binary. */
b0 : [nat -> nat].  /* Appends a zero to the right. So, effectively, returns 2*n */
b1 : [nat -> nat].  /* Appends a one to the right. So, effectively, returns 2*n + 1 */

/* Converting from unary to binary */
s_z : (= (s z) (b1 z)).
s_b0 : [((s (b0 'n)) : nat) -> (= (s (b0 'n)) (b1 'n))].
s_b1 : [((s (b1 'n)) : nat) -> (= (s (b1 'n)) (b0 (s 'n)))].

/* Converting from binary to unary */
s_z : (= (s z) (b1 z)).
s_b0 : [((s (b0 'n)) : nat) -> (= (s (b0 'n)) (b1 'n))].
s_b1 : [((s (b1 'n)) : nat) -> (= (s (b1 'n)) (b0 (s 'n)))].

/* Decrementing natural numbers */
/* Note: this is a partial function */
pred : [nat -> nat].

/* Predecessor is left-inverse of successor (in the unary representation) */
pred_s : [((pred (s 'n)) : nat) -> (= (pred (s 'n)) 'n)].

/* Predecessor of numbers in binary. */
pred_b1 : [((pred (b1 'n)) : nat) -> (= (pred (b1 'n)) (b0 'n))].
pred_b0 : [((pred (b0 'n)) : nat) -> (= (pred (b0 'n)) (b1 (pred 'n)))].

/* Less than or equal to. */
n<= : [nat -> nat -> bool].
n<=_zl : [((n<= z 'n) : bool) -> (= (n<= z 'n) true)].  /* z <= n is true for any n */
n<=_zr : [((n<= (s 'n) z) : bool) -> (= (n<= (s 'n) z) false)].  /* (s n) <= z is false for any n */
n<=_ss : [((n<= (s 'n) (s 'm)) : bool) -> (= (n<= (s 'n) (s 'm)) (n<= 'n 'm))]. /* Remaining case: lhs and rhs are both successors. */

/* Addition */
n+ : [nat -> nat -> nat].

/* Cases where one of the numbers is zero */
+zl : [((n+ z 'n) : nat) -> (= (n+ z 'n) 'n)].
+zr : [((n+ 'n z) : nat) -> (= (n+ 'n z) 'n)].

/* Unary addition. */
+s : [((n+ (s 'n) 'm) : nat) -> (= (n+ (s 'n) 'm) (s (n+ 'n 'm)))].

/* Cases depending on the last digit of each of the numbers. */
+00 : [((n+ (b0 'a) (b0 'b)) : nat) -> (= (n+ (b0 'a) (b0 'b)) (b0 (n+ 'a 'b)))].
+10 : [((n+ (b1 'a) (b0 'b)) : nat) -> (= (n+ (b1 'a) (b0 'b)) (b1 (n+ 'a 'b)))].
+01 : [((n+ (b0 'a) (b1 'b)) : nat) -> (= (n+ (b0 'a) (b1 'b)) (b1 (n+ 'a 'b)))].
+11 : [((n+ (b1 'a) (b1 'b)) : nat) -> (= (n+ (b1 'a) (b1 'b)) (b0 (s (n+ 'a 'b))))].

/* Multiplication */
n* : [nat -> nat -> nat].

/* Multiplication by zero. */
*0 : [((n* z 'n) : nat) -> (= (n* z 'n) z)].

/* Cases depending on the last digit of the first number. */
*b0 : [((n* (b0 'n) 'm) : nat) -> (= (n* (b0 'n) 'm) (b0 (n* 'n 'm)))].
*b1 : [((n* (b1 'n) 'm) : nat) -> (= (n* (b1 'n) 'm) (n+ (b0 (n* 'n 'm)) 'm))].

/* Signs: positive or negative. */
sign : type.
neg : sign.
pos : sign.

/* Integers: a signed natural number */
int : type.
snat : [sign -> nat -> int].

verify two_plus_one {
    let sum : nat = (n+ (b0 (b1 z))
                        (b1 z)).

    show (= sum (b1 (n+ (b1 z) z))) by +01.
    show (= (n+ (b1 z) z) (b1 z)) by +zr.
    show (= sum (b1 (b1 z))) by rewrite.
}

verify two_plus_three {
    let sum : nat = (n+ (b0 (b1 z))
                        (b1 (b1 z))).
}
