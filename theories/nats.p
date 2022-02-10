nat : type.
z : nat.
s : [nat -> nat].

nat_ind : [(p : [nat -> prop]) -> (p z) -> [(n : nat) -> (p n) -> (p (s n))] -> (m : nat) -> (p m)].

one : nat = (s z).
two : nat = (s one).

leq : [nat -> nat -> prop].
leq_n_n : [(n : nat) -> (leq n n)].
leq_n_sn : [(n : nat) -> (leq n (s n))].
leq_trans : [(n : nat) -> (m : nat) -> (o : nat) -> (leq n m) -> (leq m o) -> (leq n o)].

z_leq_n : [nat -> prop] = lambda (n : nat) (leq z n).

thm_z_leq_all_n : [(n : nat) -> (leq z n)] =
    (nat_ind
        z_leq_n
        (leq_n_n z)
        lambda (n : nat, IndH : (leq_z_n n))
               (leq_trans z n (s n) IndH (leq_n_sn n))
    ).
