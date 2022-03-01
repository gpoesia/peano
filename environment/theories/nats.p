nat : type.

z : nat.
s : [nat -> nat].

leq : [nat -> nat -> type].
leq_n_n : [(n : nat) -> (leq n n)].
leq_n_sn : [(n : nat) -> (leq n (s n))].

leq_trans : [(a : nat) -> (b : nat) -> (leq a b) -> (c : nat) -> (leq b c) -> (leq a c)].
