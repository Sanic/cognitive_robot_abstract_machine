/** Exercise object identity, hierarchy, and selection without WebGL. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

// %% Model contract

export async function runChecks() {
  const {ExportManifest} = await import('../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/model.js');
  const document = JSON.parse(await readFile(new URL('./manifest.json', import.meta.url)));
  const model = new ExportManifest(document);
  const cabinet = model.bodiesById.get(1);
  const drawer = model.bodiesById.get(2);
  const handle = model.bodiesById.get(3);
  assert.deepEqual(model.classesFor(handle), ['Handle']);
  assert.deepEqual(model.findBodies('Cabinet', ''), [cabinet]);
  assert.deepEqual(model.findBodies('', 'HANDLE'), [handle]);
  assert.deepEqual(model.findBodies('', 'Drawer'), [drawer]);
  assert.equal(model.parentOf(drawer), cabinet);
  assert.equal(model.parentOf(cabinet), null);
  assert.deepEqual(model.childrenOf(cabinet), [drawer]);
  assert.deepEqual(model.assemblyOf(cabinet), [cabinet, drawer, handle]);
  assert.deepEqual(model.ancestorsOf(handle), [drawer, cabinet]);
  assert.equal(model.bodiesById.get(4).name, 'empty');
  const unknown = structuredClone(document);
  unknown.bodies[1].direct_semantic_annotation_ids = [999];
  assert.throws(() => new ExportManifest(unknown), /Unknown annotation/);
  const cyclic = structuredClone(document);
  cyclic.bodies[1].parent_id = 3;
  assert.throws(() => new ExportManifest(cyclic), /cycle/i);
  const missingParent = structuredClone(document);
  missingParent.bodies[2].parent_id = 99;
  assert.throws(() => new ExportManifest(missingParent), /Unknown parent/);
  assert.throws(() => new ExportManifest({...document, schema_version: 1}), /version 2/);
  return '14 object and hierarchy checks passed';
}

// %% Command-line execution

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) console.log(await runChecks());
