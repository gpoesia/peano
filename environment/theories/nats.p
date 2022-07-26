nat : type.

z : nat.
s : [nat -> nat].

= : [('t : type) -> 't -> 't -> prop].

leq : [nat -> nat -> prop].
leq_n_n : [('n : nat) -> (leq 'n 'n)].
leq_n_sn : [('n : nat) -> (leq 'n (s 'n))].

leq_trans : [(leq 'a 'b) -> (leq 'b 'c) -> (leq 'a 'c)].

add : [nat -> nat -> nat].

add_z : [((add z 'n) : nat) -> (= (add z 'n) 'n)].
add_s : [((add (s 'm) 'n) : nat) -> (= (add (s 'm) 'n) (s (add 'm 'n)))].

verify two_plus_two {
    let one : nat = (s z).
    let two : nat = (s one).
    let three : nat = (s two).
    let four : nat = (s three).

    let two_plus_two : nat = (add two two).

    show (= (add two two) (s (add one two))) by add_s.
    show (= (add one two) (s (add z two))) by add_s.
    show (= (add z two) two) by add_z.
    show (= (add one two) three) by rewrite.

    show (= two_plus_two four) by rewrite.
}
