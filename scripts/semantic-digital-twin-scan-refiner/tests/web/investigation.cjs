"use strict";

// %% Clipboard handoff
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

async function exercise() {
  const elements = new Map();
  const getElementById = id => {
    if (!elements.has(id)) elements.set(id, { value: "", focus() {}, select() {} });
    return elements.get(id);
  };
  getElementById("candidate").value = "0";
  getElementById("geometry").value = "both";
  getElementById("comparison-role").value = "visual";
  const navigator = {};
  const runtime = vm.createContext({
    document: { getElementById }, location: { origin: "http://127.0.0.1:8815" }, navigator,
  });
  const Handoff = vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + "\nInvestigationHandoff", runtime);
  const view = { selected_body: "apartment/handle", selection_lower: [1, 2, 3] };
  const bounds = { lower: [1, 2, 3], upper: [4, 5, 6] };
  const frame = [[0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]];
  const placement = [[1, 0, 0, 3], [0, 1, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]];
  const tools = { viewRevision: "displayed", session: { revision: "newer" }, inspection: { view } };
  const controller = {
    tools, frame: { matrix: frame }, frameMode: "world", frameYaw: 90,
    readBounds: () => bounds, placement: () => placement,
    state: {
      accepted_candidate: 0,
      project_record: { model_source: "apartment.urdf", world_sha256: "world", scan_source: "Apartment-Lab.e57", scan_sha256: "scan" },
    },
  };
  const handoff = new Handoff({ controller, notice() {} });
  const button = getElementById("copy-investigation"), manual = getElementById("investigation-context");
  let clipboard;
  navigator.clipboard = { writeText: async text => { clipboard = text; } };
  await button.onclick();
  const context = JSON.parse(clipboard);
  assert.equal(context.project.url, "http://127.0.0.1:8815");
  assert.equal(context.selected_body, "apartment/handle");
  assert.deepEqual(context.region_query, {
    expected_revision: "displayed", candidate_index: 0, frame_T_scan: frame, ...bounds,
  });
  assert.deepEqual(context.registration, { accepted_candidate: 0, scan_T_root: placement });
  assert.equal(manual.hidden, true);

  view.selected_body = "";
  navigator.clipboard.writeText = async () => { throw Error("Clipboard denied"); };
  await button.onclick();
  assert.equal(manual.hidden, false);
  assert.equal(JSON.parse(manual.value).selected_body, null);
  assert.deepEqual(JSON.parse(manual.value).region_query, context.region_query);

  view.selection_lower = [];
  handoff.update();
  assert.equal(button.disabled, true);
  view.selected_body = "apartment/handle";
  handoff.update();
  assert.equal(button.disabled, false);
  await button.onclick();
  assert.equal(JSON.parse(manual.value).region_query.body_name, view.selected_body);
}

exercise().catch(error => { console.error(error); process.exitCode = 1; });
