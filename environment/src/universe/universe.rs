use std::option::Option;

use egg::*;

const OPAQUE_NODE : &str = &"$opaque";
const IS_NODE: &str = &"$is";
const DECL_NODE: &str = &"$decl";
const ARROW_NODE: &str = &"$arrow";
const ACTION_NODE: &str = &"$action";

pub struct Universe {
    egraph: EGraph<SymbolLang, ()>
}

impl Universe {
    pub fn new() -> Universe {
        let mut u = Universe {
            egraph: Default::default(),
        };

        for s in [OPAQUE_NODE, IS_NODE, DECL_NODE] {
            u.egraph.add(SymbolLang::leaf(s));
        }

        u
    }

    pub fn size(&self) -> usize {
        self.egraph.number_of_classes()
    }

    pub fn add_declaration(&mut self, name: &String, type_: &str, value: Option<&str>) {
        // TODO(gpoesia): Need to recursively add type annotations for each sub-term.
        // TODO(gpoesia): ensure the name is unique.

        let name_id = self.egraph.add(SymbolLang::leaf(name));
        let type_expr: RecExpr<SymbolLang> = type_.parse().unwrap();

        let type_id = self.egraph.add_expr(&type_expr);

        self.egraph.add(SymbolLang::new(IS_NODE, vec![name_id, type_id]));

        println!("Root type node: {}", type_expr[0.into()].op.as_str());

        if type_expr[0.into()].op.as_str() == ARROW_NODE {
            self.egraph.add(SymbolLang::new(ACTION_NODE, vec![name_id]));
        }

        if let Some(def_value) = value {
            let def_value_id = self.egraph.add_expr(&def_value.parse().unwrap());
            self.egraph.union(name_id, def_value_id);
        }

        self.egraph.rebuild();
    }

    pub fn find_actions(&self) -> Vec<String> {
        let mut actions = Vec::new();
        let pattern : Pattern<SymbolLang> = "($action ?a)".parse().unwrap();

        let matches = pattern.search(&self.egraph);

        for m in matches {
            let action_id: &Id = &m.substs[0]["?a".parse().unwrap()];
            let action_eclass = &self.egraph[*action_id];
            actions.push(action_eclass.nodes[0].to_string());
        }

        actions
    }
}

pub mod tests {
    use crate::universe::Universe;

    #[test]
    fn test_create_universe() {
        let mut u = Universe::new();
        assert_eq!(u.size(), 3);

        u.add_declaration(&"nat".to_string(), &"type".to_string(), None);
        u.add_declaration(&"z".to_string(), &"nat".to_string(), None);

        assert_eq!(u.find_actions().len(), 0);

        u.add_declaration(&"s".to_string(), &"$arrow(nat,nat)".to_string(), None);
        println!("Actions: {:?}", u.find_actions());

        assert_eq!(u.find_actions().len(), 1);

    }
}
