/** Exercise saved matching, disagreement filters and reverse many-to-one selection. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {ExportManifest} from '../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/model.js';

// %% Comparison navigation

export async function runChecks() {
  const {ComparisonData} = await import('../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/comparison_model.js');
  const payload = JSON.parse(await readFile(new URL('./manifest.json', import.meta.url)));
  const manifest = new ExportManifest(payload);
  const table = {
    comparison: {
      bodies: [
        {predicted_identifier: 1, ground_truth_identifier: 1, class_disagreement: false, parent_disagreement: false, margin: 0, candidates: []},
        {predicted_identifier: 2, ground_truth_identifier: 1, class_disagreement: true, parent_disagreement: true, margin: 0.1, candidates: []},
        {predicted_identifier: 3, ground_truth_identifier: null, class_disagreement: null, parent_disagreement: null, margin: null, candidates: []},
      ],
      unmatched_ground_truth: [2, 3],
    },
    prediction_objects: [{identifier: 1}, {identifier: 2}, {identifier: 3}],
    ground_truth_objects: [{identifier: 1}, {identifier: 2}, {identifier: 3}],
  };
  const model = new ComparisonData(table, manifest, manifest);
  assert.deepEqual(model.forGroundTruth(1).map(record => record.predicted_identifier), [1, 2]);
  assert.equal(model.forPrediction(3).ground_truth_identifier, null);
  assert.deepEqual(model.rows('unmatched-prediction', '').map(row => row.identifier), [3]);
  assert.deepEqual(model.rows('unmatched-ground-truth', '').map(row => row.identifier), [2, 3]);
  assert.deepEqual(model.rows('class', '').map(row => row.identifier), [2]);
  assert.deepEqual(model.rows('parent', '').map(row => row.identifier), [2]);
  assert.deepEqual(model.rows('many-to-one', '').map(row => row.identifier), [1, 2]);
  assert.deepEqual(model.rows('all', 'handle').map(row => row.identifier), [3]);
  assert.deepEqual(ComparisonData.decode({__collection_type__: 'builtins.list', __items__: [1, 2]}), [1, 2]);
  assert.throws(() => new ComparisonData({...table, comparison: {...table.comparison, bodies: [{predicted_identifier: 99}]}}, manifest, manifest), /Unknown prediction/);
  return 'Saved match, reverse selection, unmatched, class, parent and fragmentation filters passed';
}

// %% Command-line execution

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) console.log(await runChecks());
