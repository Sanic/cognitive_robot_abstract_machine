/** Test linked selection and alternative previews through the actual controller. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {SourceTextModule, SyntheticModule, createContext} from 'node:vm';
import {ExportManifest} from '../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/model.js';
import {ComparisonData} from '../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/comparison_model.js';
import {Element} from './interface_checks.mjs';

// %% Two inspector windows and saved correspondences

export async function runChecks() {
  const identifiers = ['prediction-frame', 'ground-truth-frame', 'open-comparison', 'comparison-url',
    'review-filter', 'review-search', 'pair-opacity', 'pair-isolate', 'fit-pair', 'reset-pair',
    'comparison-status', 'review-count', 'review-objects', 'match-details', 'provenance'];
  const elements = new Map(identifiers.map(identifier => [identifier, new Element(identifier)]));
  elements.get('review-filter').value = 'all';
  const messages = {prediction: [], ground_truth: []};
  elements.get('prediction-frame').contentWindow = {postMessage: message => messages.prediction.push(message)};
  elements.get('ground-truth-frame').contentWindow = {postMessage: message => messages.ground_truth.push(message)};
  const handlers = new Map();
  const origin = 'http://localhost:8002';
  const context = createContext({
    document: {querySelectorAll: () => [...elements.values()], createElement: () => new Element()},
    URL, URLSearchParams,
    window: {location: {search: '', origin}, addEventListener: (event, listener) => handlers.set(event, listener)},
  });
  const source = await readFile(new URL('../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/comparison.js', import.meta.url), 'utf8');
  const module = new SourceTextModule(source + '\nexport {ComparisonInspector};', {context});
  await module.link(name => name === './model.js'
    ? new SyntheticModule(['ExportManifest'], function() { this.setExport('ExportManifest', ExportManifest); }, {context})
    : new SyntheticModule(['ComparisonData'], function() { this.setExport('ComparisonData', ComparisonData); }, {context}));
  await module.evaluate();
  const inspector = new module.namespace.ComparisonInspector();
  const manifest = new ExportManifest(JSON.parse(await readFile(new URL('./manifest.json', import.meta.url))));
  const candidate = {ground_truth_identifier: 1, cost: 0.1, distance: 0.02, size_difference: 0.2};
  const table = {
    comparison: {bodies: [
      {predicted_identifier: 1, ground_truth_identifier: 1, class_disagreement: false, parent_disagreement: false, margin: 0.2, candidates: [candidate]},
      {predicted_identifier: 2, ground_truth_identifier: 1, class_disagreement: true, parent_disagreement: true, margin: 0.2, candidates: [candidate, {...candidate, ground_truth_identifier: 3}]},
      {predicted_identifier: 3, ground_truth_identifier: null, class_disagreement: null, parent_disagreement: null, margin: null, candidates: []},
    ], unmatched_ground_truth: [2, 3]},
  };
  inspector.data = new ComparisonData(table, manifest, manifest);
  inspector.ready.add('prediction');
  inspector.ready.add('ground_truth');
  inspector.select('prediction', 2);
  assert.equal(messages.ground_truth.at(-1).identifiers.join(','), '1');
  inspector.select('ground_truth', 1);
  assert.equal(messages.prediction.at(-1).identifiers.join(','), '1,2');
  inspector.select('prediction', 3);
  assert.equal(messages.ground_truth.at(-1).identifiers.length, 0);
  assert.equal(messages.ground_truth.at(-1).active, true);
  assert.match(elements.get('match-details').textContent, /No accepted GT match/);
  inspector.select('prediction', 2, true, 3);
  assert.equal(messages.ground_truth.at(-1).identifiers.join(','), '3');
  assert.equal(inspector.data.forPrediction(2).ground_truth_identifier, 1);
  assert.match(elements.get('match-details').textContent, /saved match above has not changed/);
  elements.get('review-filter').value = 'class';
  inspector.list();
  assert.equal(elements.get('review-objects').children.length, 1);
  handlers.get('message')({origin: 'http://another-host', source: elements.get('ground-truth-frame').contentWindow, data: {kind: 'body-selected', identifier: 1}});
  assert.equal(inspector.selection.identifier, 2);
  return 'Linked prediction/GT clicks, many-to-one selection, unmatched isolation, candidate preview and class filter passed';
}

// %% Command-line execution

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) console.log(await runChecks());
