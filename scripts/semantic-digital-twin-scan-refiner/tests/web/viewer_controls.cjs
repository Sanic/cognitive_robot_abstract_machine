"use strict";

// %% Test inputs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const [assets, figurePath, scenario] = process.argv.slice(2);
const figure = JSON.parse(fs.readFileSync(figurePath, "utf8"));
const collision = structuredClone(figure.data[1]);
collision.legendgroup = "collision";
collision.visible = false;
figure.data.push(collision);

// %% DOM and renderer fixtures
function element(value = "") {
  return { value, listeners: {}, options: [], add(option) { this.options.push(option); },
    addEventListener(event, callback) { this.listeners[event] = callback; },
    querySelector() { return this.heatmapOption ||= {}; } };
}
const elements = Object.fromEntries([
  "candidate", "geometry", "scan", "status", "point-size", "mesh-opacity", "point-hover", "rotate-view",
  "display-origin", "region-status", "region-lower", "region-upper", "load-region", "load-overview", "point-budget"
].map(identifier => [identifier, element()]));
elements.geometry.value = "visual";
elements.scan.value = "rgb";
const initial = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const shifted = [[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]];
elements["viewer-state"] = {textContent: JSON.stringify({
  registration: {candidates: [{scan_T_root: initial}, {scan_T_root: shifted}]},
  selected_candidate: 0, accepted_candidate: null, has_heatmap: false, display_origin: [0,0,0], scan_distances: null
})};
elements.scene = {data: figure.data, layout: figure.layout, _fullLayout: {scene: {camera: {eye:{x:1,y:1,z:1}}}},
  dataset: {plotReady: "true"}, on() {}, addEventListener() {}};
let active = 0;
let maximumActive = 0;
async function render(update, indices, layout = {}) {
  active++;
  maximumActive = Math.max(maximumActive, active);
  await new Promise(resolve => setTimeout(resolve, 5));
  indices.forEach((index, position) => {
    for (const [key, values] of Object.entries(update)) {
      const value = Array.isArray(values) ? values[position] : values;
      if (key.startsWith("marker.")) elements.scene.data[index].marker[key.slice(7)] = value;
      else elements.scene.data[index][key] = value;
    }
  });
  Object.assign(elements.scene.layout, layout);
  active--;
}
const context = vm.createContext({
  document: {getElementById(identifier) { return elements[identifier]; }, querySelectorAll() { return []; }},
  Option: class {constructor(text, value) {this.text=text;this.value=value;}},
  Plotly: {
    restyle(scene, update, indices) {return render(update, indices);},
    update(scene, update, layout, indices) {return render(update, indices, layout);},
    relayout(scene, layout) {return render({}, [], layout);}
  },
  location: {protocol:"file:"}, AbortController, URLSearchParams, DataView, atob, setTimeout, console
});

// %% Actual controller behaviour
(async () => {
  const helper = path.join(assets, "plot_buffers.js");
  if (fs.existsSync(helper)) vm.runInContext(fs.readFileSync(helper, "utf8"), context);
  await vm.runInContext(fs.readFileSync(path.join(assets, "viewer.js"), "utf8"), context);
  await new Promise(resolve => setTimeout(resolve, 30));
  if (scenario === "coordinates") {
    elements.candidate.value = "1";
    elements.candidate.listeners.change();
    await new Promise(resolve => setTimeout(resolve, 100));
    for (const trace of elements.scene.data.filter(trace => trace.type === "mesh3d")) {
      assert.equal(trace.x.length, 8, "candidate switch discarded mesh vertices");
      assert.deepEqual([Math.min(...trace.x),Math.max(...trace.x)], [0.5,1.5]);
      assert.deepEqual([Math.min(...trace.y),Math.max(...trace.y)], [1.5,2.5]);
      assert.deepEqual([Math.min(...trace.z),Math.max(...trace.z)], [2.5,3.5]);
    }
  } else if (scenario === "live-camera") {
    const orbitCamera = {
      eye: {x: -2.2, y: 0.6, z: 1.1},
      up: {x: -0.17, y: -0.96, z: 0.21},
      center: {x: 0, y: 0, z: 0},
      projection: {type: "perspective"}
    };
    elements.scene._fullLayout.scene._scene = {getCamera: () => orbitCamera};
    await vm.runInContext('new PlotUpdates(document.getElementById("scene")).updateGeometry({x:[[1]]},[1])', context);
    assert.deepEqual(elements.scene.layout["scene.camera"], orbitCamera, "the redraw reset a live orbit to the older layout camera");
  } else {
    for (const choice of ["collision", "both", "none", "visual"]) {
      elements.geometry.value = choice;
      elements.geometry.listeners.change();
    }
    await new Promise(resolve => setTimeout(resolve, 150));
    assert.equal(maximumActive, 1, "geometry switches overlapped renderer operations");
    assert.deepEqual(elements.scene.data.filter(trace => trace.type === "mesh3d").map(trace => trace.visible), [true,false]);
    assert.equal(elements.scene.data[0].visible, true, "geometry choice hid scan points");
  }
})();
