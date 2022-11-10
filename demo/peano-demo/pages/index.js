import dynamic from 'next/dynamic'
import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'

import { apiRequest } from '../lib/api';

import { useState } from 'react';

const Solution = dynamic(() => import('../lib/components/solution'), {
  ssr: false,
})

const RESPONSE = {"problem": "(((x / 2) + -1) = 3)", "solution": "(x = 8)", "trace": {"type": "trace", "arrow": null, "value": "(x = 8)", "steps": [{"type": "trace", "arrow": "tactic022", "value": "((x / 2) = 4)", "steps": [{"type": "axiom", "arrow": "sub_eq", "value": "((((x / 2) + -1) - -1) = (3 - -1))"}, {"type": "trace", "arrow": "tactic018", "value": "((x / 2) = (3 - -1))", "steps": [{"type": "trace", "arrow": "tactic010", "value": "(((x / 2) + 0) = (3 - -1))", "steps": [{"type": "axiom", "arrow": "-+_assoc", "value": "((((x / 2) + -1) - -1) = ((x / 2) + (-1 - -1)))"}, {"type": "trace", "arrow": "tactic005", "value": "(((x / 2) + 0) = (3 - -1))", "steps": [{"type": "trace", "arrow": "tactic000", "value": "((((x / 2) + -1) - -1) = ((x / 2) + 0))", "steps": [{"type": "axiom", "arrow": "eval", "value": "((-1 - -1) = 0)"}, {"type": "axiom", "arrow": "rewrite", "value": "((((x / 2) + -1) - -1) = ((x / 2) + 0))"}]}, {"type": "axiom", "arrow": "rewrite", "value": "(((x / 2) + 0) = (3 - -1))"}]}]}, {"type": "trace", "arrow": "tactic006", "value": "((x / 2) = (3 - -1))", "steps": [{"type": "axiom", "arrow": "+0_id", "value": "(((x / 2) + 0) = (x / 2))"}, {"type": "axiom", "arrow": "rewrite", "value": "((x / 2) = (3 - -1))"}]}]}, {"type": "trace", "arrow": "tactic000", "value": "((x / 2) = 4)", "steps": [{"type": "axiom", "arrow": "eval", "value": "((3 - -1) = 4)"}, {"type": "axiom", "arrow": "rewrite", "value": "((x / 2) = 4)"}]}]}, {"type": "trace", "arrow": "tactic016", "value": "(x = 8)", "steps": [{"type": "axiom", "arrow": "mul_eq", "value": "(((x / 2) * 2) = (4 * 2))"}, {"type": "trace", "arrow": "tactic013", "value": "(x = (4 * 2))", "steps": [{"type": "axiom", "arrow": "*/_assoc_l", "value": "(((x / 2) * 2) = (x * (2 / 2)))"}, {"type": "trace", "arrow": "tactic007", "value": "(x = (4 * 2))", "steps": [{"type": "trace", "arrow": "tactic005", "value": "((x * 1) = (4 * 2))", "steps": [{"type": "trace", "arrow": "tactic000", "value": "(((x / 2) * 2) = (x * 1))", "steps": [{"type": "axiom", "arrow": "eval", "value": "((2 / 2) = 1)"}, {"type": "axiom", "arrow": "rewrite", "value": "(((x / 2) * 2) = (x * 1))"}]}, {"type": "axiom", "arrow": "rewrite", "value": "((x * 1) = (4 * 2))"}]}, {"type": "trace", "arrow": "tactic002", "value": "(x = (4 * 2))", "steps": [{"type": "axiom", "arrow": "*1_id", "value": "((x * 1) = x)"}, {"type": "axiom", "arrow": "rewrite", "value": "(x = (4 * 2))"}]}]}]}, {"type": "trace", "arrow": "tactic000", "value": "(x = 8)", "steps": [{"type": "axiom", "arrow": "eval", "value": "((4 * 2) = 8)"}, {"type": "axiom", "arrow": "rewrite", "value": "(x = 8)"}]}]}]}};

export default function Home() {
  console.log('Solution:', Solution);

  const [expandedTraces, setExpandedTraces] = useState([]);
  const [solution, setSolution] = useState("");

  const response = RESPONSE;

  console.log('Expanded traces:', expandedTraces);

  const [isLoading, setIsLoading] = useState(false);
  const [solutionMarkers, setSolutionMarkers] = useState([]);
  const [workedExample, setWorkedExample] = useState(null);

  const [message, setMessage] = useState('');

  const check = async () => {
    const lines = solution.split('\n');
    if (lines.length === 1) {
      return setMessage('You have no solution steps yet.')
    }

    const problem = lines[0];
    const lastStep = lines[lines.length - 1];

    const response = await apiRequest('check', { solution: solution });

    const markers = [""];

    response.checks.forEach(c => {
      markers.push(c ? "✔" : "❌");
    });

    setSolutionMarkers(markers);
  };

  const showExample = async () => {
    const lines = solution.split('\n');
    const lastStep = lines[lines.length - 1];

    const lastStepSolution = await apiRequest('solve', { equation: lastStep });

    if (lastStepSolution.error) {
      return setMessage(lastStepSolution.error);
    }

    setExpandedTraces([]);
    setWorkedExample(lastStepSolution);
  }

  return (
    <div className="root">
      <div className="solution-column">
        <div className="interactive-solution">
          <div className="solution-markers">
            {
              solutionMarkers.map((m, i) => (
                <span key={i} className="solution-marker">
                  { m }
                </span>
              ))
            }
          </div>
          <textarea
            className="solution-box"
            value={solution}
            onChange={e => { setSolutionMarkers([]); setSolution(e.target.value); }} />
        </div>
        <div className="solution-actions">
          <button className="solution-action" onClick={check}>Check</button>
          <button className="solution-action" onClick={showExample}>I'm stuck!</button>
          <p>
            { message }
          </p>
        </div>
      </div>
      <div className="solution-example">
        { workedExample &&
          <Solution
            problem={workedExample.problem}
            trace={workedExample.trace}
            expandedTraces={expandedTraces}
            setExpandedTraces={setExpandedTraces}
          />
        }
      </div>
    </div>
  )
}
