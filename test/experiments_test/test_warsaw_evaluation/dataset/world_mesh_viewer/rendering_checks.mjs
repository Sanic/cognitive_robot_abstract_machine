/** Dense reconstructed surfaces must not become a blanket of dark triangle edges. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {SourceTextModule, SyntheticModule, createContext} from 'node:vm';

// %% A dense surface supplied through the scene loader

export async function runChecks() {
  const base = new URL('../../../../../experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/', import.meta.url);
  const context = createContext({console, AbortController});
  const modules = new Map();
  let three;
  class TriangulatedSurface {
    /** Supply real Three.js geometry without browser file or graphics APIs. */
    async parseAsync() {
      const mesh = new three.Mesh(new three.IcosahedronGeometry(1, 3),
        new three.MeshBasicMaterial({color: '#111111', vertexColors: true}));
      mesh.name = 'scan';
      const scene = new three.Scene();
      scene.add(mesh);
      return {scene, parser: {associations: new Map(), json: {nodes: []}}};
    }
  }
  async function sourceModule(url) {
    if (!modules.has(url.href)) modules.set(url.href, readFile(url, 'utf8').then(source => new SourceTextModule(source, {identifier: url.href, context})));
    return await modules.get(url.href);
  }
  const module = await sourceModule(new URL('scene.js', base));
  const loader = new SyntheticModule(['GLTFLoader'], function() { this.setExport('GLTFLoader', TriangulatedSurface); }, {context});
  await module.link((specifier, referrer) => {
    if (specifier === './vendor/GLTFLoader.js') return loader;
    return sourceModule(specifier === 'three' ? new URL('vendor/three.module.js', base) : new URL(specifier, referrer.identifier));
  });
  await module.evaluate();
  three = (await sourceModule(new URL('vendor/three.module.js', base))).namespace;
  const display = Object.create(module.namespace.WorldScene.prototype);
  const loaded = await display.read(new ArrayBuffer(0), {bodies: [{id: 1, node_name: 'scan'}]});
  const mesh = loaded.meshesByBody.get(1)[0];
  const frame = new three.Box3Helper(new three.Box3());
  const outline = mesh.userData.outline.geometry;
  assert.equal(outline.index?.count ?? outline.attributes.position.count, frame.geometry.index.count,
    'Highlight outlines must have a fixed box edge count, independent of mesh triangulation.');
  const selected = {color: '#33ffcc', visible: true, emphasized: true, opacity: 1};
  display.meshesByBody = loaded.meshesByBody;
  display.paint(new Map([[1, selected]]));
  assert.equal(mesh.material.color.getHexString(), new three.Color(selected.color).getHexString());
  assert.equal(mesh.material.vertexColors, false);
  assert.equal(mesh.userData.outline.visible, true);
  display.paint(new Map([[1, {...selected, emphasized: false}]]));
  assert.equal(mesh.userData.outline.visible, false);
  display.dispose(loaded.model);
  frame.geometry.dispose();
  frame.material.dispose();
  return 'Dense surface highlight colour and fixed-size outline checks passed';
}

// %% Command-line execution

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) console.log(await runChecks());
