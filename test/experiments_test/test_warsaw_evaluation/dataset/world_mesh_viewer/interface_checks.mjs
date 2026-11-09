/** Check selection and visibility with a DOM fixture and a renderer substitute. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {SourceTextModule, SyntheticModule, createContext} from 'node:vm';
import {ExportManifest} from '../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/model.js';

// %% DOM and renderer fixtures

export class Element {
  /** Preserve child text and listeners so relationship controls can be exercised. */
  constructor(id = '') {
    this.id = id;
    this.children = [];
    this.listeners = new Map();
    this.value = '';
    this.checked = false;
    this.content = '';
    this.classList = {add() {}, remove() {}};
  }
  set textContent(text) { this.content = String(text); }
  get textContent() { return this.content + this.children.map(child => child.textContent).join(' '); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this.content = ''; }
  addEventListener(event, listener) { this.listeners.set(event, listener); }
  setAttribute(name, value) { this[name] = value; }
}

class GeometryDisplay {
  /** Record per-object visibility and framing without allocating a WebGL context. */
  constructor() { this.meshesByBody = new Map(); }
  async read(bytes, manifest) {
    return {meshesByBody: new Map(manifest.bodies.filter(body => body.face_count).map(body => [body.id, []]))};
  }
  display(loaded) { this.meshesByBody = loaded.meshesByBody; }
  paint(states) { this.states = states; }
  fit(bodies) { this.framed = bodies; }
  dispose() {}
}

// %% Object navigation and rendering state

export async function runChecks() {
  const identifiers = ['viewport', 'folder', 'open-run', 'run-url', 'search', 'classes',
    'isolate', 'opacity', 'fit', 'reset', 'status', 'details', 'scene-name', 'selected-label', 'object-count', 'objects'];
  const elements = new Map(identifiers.map(identifier => [identifier, new Element(identifier)]));
  const markup = await readFile(new URL('../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/index.html', import.meta.url), 'utf8');
  elements.get('opacity').value = markup.match(/id="opacity"[^>]*value="([^"]+)"/)[1];
  const document = {querySelectorAll: () => [...elements.values()], createElement: () => new Element()};
  const context = createContext({document, URLSearchParams, window: {location: {search: ''}}});
  const modelModule = new SyntheticModule(['ExportManifest'], function() { this.setExport('ExportManifest', ExportManifest); }, {context});
  const sceneModule = new SyntheticModule(['WorldScene'], function() { this.setExport('WorldScene', GeometryDisplay); }, {context});
  context.Option = class extends Element {
    constructor(label, value) { super(); this.textContent = label; this.value = value; }
  };
  const source = await readFile(new URL('../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/viewer.js', import.meta.url), 'utf8');
  const module = new SourceTextModule(source + '\nexport {WorldInspector};', {context});
  await module.link(name => name === './model.js' ? modelModule : sceneModule);
  await module.evaluate();
  const inspector = new module.namespace.WorldInspector();
  const payload = JSON.parse(await readFile(new URL('./manifest.json', import.meta.url)));
  await inspector.open(payload, new ArrayBuffer(0), 'fixture', 0);
  assert.equal(elements.get('objects').children.length, payload.bodies.length - 1);
  inspector.select(2, true);
  assert.equal(inspector.scene.states.get(4).opacity, 1, 'Selection should preserve the surrounding world by default.');
  assert.equal(inspector.scene.states.get(2).color, '#ff45df');
  assert.equal(inspector.scene.states.get(3).color, '#00e5ff');
  assert.equal(inspector.scene.states.get(1).color, '#ffb74b');
  assert.deepEqual(inspector.scene.framed.map(body => body.id), [2, 3]);
  assert.match(elements.get('details').textContent, /cabinet/);
  assert.match(elements.get('details').textContent, /handle/);
  elements.get('isolate').checked = true;
  inspector.update();
  assert.equal(inspector.scene.states.get(1).visible, false);
  assert.equal(inspector.scene.states.get(3).visible, true);
  const parentButton = elements.get('details').children.find(element => element.className === 'relation');
  parentButton.listeners.get('click')();
  assert.equal(inspector.selected.id, 1);
  assert.deepEqual(inspector.scene.framed.map(body => body.id), [1, 2, 3]);
  const actionPanel = elements.get('details').children.find(element => element.className === 'actions');
  actionPanel.children.find(button => button.textContent === 'Hide object').listeners.get('click')();
  assert.equal(inspector.scene.states.get(1).visible, false);
  assert.equal(inspector.scene.states.get(2).visible, true);
  elements.get('reset').listeners.get('click')();
  assert.equal(inspector.scene.states.get(1).visible, true);
  assert.equal(elements.get('isolate').checked, false);
  elements.get('search').value = 'handle';
  inspector.update();
  assert.equal(elements.get('objects').children.length, 1);
  inspector.select(4);
  assert.match(elements.get('details').textContent, /no geometry/);
  return 'Selection, relationship navigation, isolation, hiding, reset, search and empty-object checks passed';
}

// %% Command-line execution

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) console.log(await runChecks());
