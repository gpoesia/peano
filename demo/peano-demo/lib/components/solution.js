import React from 'react';
import { find, union, without }  from 'lodash';

const SolutionStep = ({ step, key }) => (
  <p>{ step.value }</p>
);

const ANNOTATION = {
  "add_eq": "Add on both sides; see Same Operation on Both Sides",
  "sub_eq": "Subtract from both sides; see Same Operation on Both Sides",
  "mul_eq": "Multiply on both sides; see Same Operation on Both Sides",
  "div_eq": "Divide both sides; see Same Operation on Both Sides",

  "-+_assoc": "Associativity; see Properties of addition",
  "+-_assoc": "Associativity; see Properties of addition",

  "*/_assoc_l": "Associativity; see Properties of multiplication",
  "*/_assoc_r": "Associativity; see Properties of multiplication",

  "eval": "Calculate",
  "rewrite": "Substitute; see Substitution and Evaluating Expressions",

  "+0_id": "Identity of addition; see Properties of addition",
  "-0_id": "Identity of subtraction; see Properties of subtraction",

  "*1_id": "Identity of multiplication; see Properties of multiplication",
  "/1_id": "Identity of division; see Properties of division",

  "tactic000": "Evaluate and substitute; see Substitution and Evaluating Expressions",
  "tactic002": "Simplify using identity of multiplication; see Properties of multiplication",
  "tactic004": "Simplify; see Combining Like Terms",
  "tactic005": "Rearrange and simplify; see Substitution and Evaluating Expressions",
  "tactic006": "Simplify using identity of addition; see Properties of addition",
  "tactic007": "Simplify; see Combining Like Terms",
  "tactic010": "Simplify; see Combining Like Terms",
  "tactic011": "Simplify; see Combining Like Terms",
  "tactic013": "Rearrange and simplify; see Combining Like Terms",
  "tactic015": "Rearrange and simplify; see Combining Like Terms",
  "tactic018": "Rearrange and simplify; see Combining Like Terms",
  "tactic021": "Rearrange and simplify; see Combining Like Terms",

  "tactic016": "Solve for division; One-Step Multiplication and Division Equations",
  "tactic019": "Solve multiplicative equation; See One-Step Multiplication and Division Equations",
  "tactic022": "Solve additive equation; See One-Step Addition and Subtraction Equations",
  "tactic025": "Solve subtractive equation; See One-Step Addition and Subtraction Equations",
};

const Trace = ({ trace, index, expandedTraces, setExpandedTraces }) => {
  console.log('Index', index, 'expanded:', expandedTraces);

  let content;

  const isExpanded = expandedTraces.indexOf(index) !== -1;
  const isExpandable = trace.type !== 'axiom';

  if (trace.type == "axiom" || !isExpanded) {
    console.log('Index', index, 'not expanded');
    content = (
      <div>
        { trace.arrow && <span className="step-annotation">[{ANNOTATION[trace.arrow] || trace.arrow}]</span> }
        <SolutionStep step={trace} />
      </div>
    );
  } else {
    console.log('Index', index, 'expanded');

    content = (
      <div>
        { trace.steps.map((s, i) => (
          <Trace trace={s} key={i} index={index + "-" + i}
            expandedTraces={expandedTraces}
            setExpandedTraces={setExpandedTraces}
          />
        )) }
      </div>
    );
  }

  const toggle = () => {
    if (!isExpanded) {
      setExpandedTraces(union(expandedTraces, [index]));
    } else {
      console.log('Expanding', index);
      setExpandedTraces(without(expandedTraces, index));
    }
  }

  return (
    <div className="trace">
      <TraceBorder expanded={isExpanded} expandable={isExpandable} onClick={toggle}/>
      { content }
    </div>
  );
};

const TraceBorder = ({ onClick, expandable, expanded }) => (
  <div className="traceColumn">
    { expandable &&
      <div className="traceButton" onClick={onClick}>
        { expanded ? "➖" : "⋮" }
      </div>
    }
  </div>
);

const Solution = ({ problem, trace, expandedTraces, setExpandedTraces }) => {
  return (
    <div>
      <p>Problem: { problem }</p>
      <p>
        Solution:
        <Trace
          trace={trace}
          index='#'
          expandedTraces={expandedTraces}
          setExpandedTraces={setExpandedTraces}
        />
      </p>
    </div>
  );
};


export default Solution;
