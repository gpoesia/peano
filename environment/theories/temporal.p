event: type.

before: [event -> event -> prop].
after: [event -> event -> prop].

before_trans: [(before 'a 'b) -> (before 'b 'c) -> (before 'a 'c)].
after_trans: [(after 'a 'b) -> (after 'b 'c) -> (after 'a 'c)].

not : [prop -> prop].

not_after : [(before 'a 'b) -> (not (after 'a 'b))].
not_before : [(after 'a 'b) -> (not (before 'a 'b))].

after_inv : [(after 'a 'b) -> (before 'b 'a)].
before_inv : [(before 'a 'b) -> (after 'b 'a)].

verify trans_example {
    let buy_coffee_bean : event. let make_coffee : event. let be_awake : event. let crash : event.
    assume (before buy_coffee_bean make_coffee).
    assume (before make_coffee be_awake).
    assume (before be_awake crash).
    
    show (before buy_coffee_bean be_awake) by before_trans.
    show (before buy_coffee_bean crash) by before_trans.
    show (after crash buy_coffee_bean) by before_inv.        
}
