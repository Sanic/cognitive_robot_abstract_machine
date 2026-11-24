"use strict";

// %% Assembly bounds and handoff
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map();
const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {value: ""});
      return elements.get(id);
    },
  },
  location: {origin: "http://127.0.0.1:8815"},
});
for (const asset of ["viewer.js", "assemblies.js", "investigation.js"]) {
  const code = fs.readFileSync(path.join(process.argv[2],asset),"utf8");
  vm.runInContext(code.replace("initializeViewer();",""),context);
}
const AssemblyPanel = vm.runInContext("AssemblyPanel",context);
const InvestigationHandoff = vm.runInContext("InvestigationHandoff",context);
const group = {identifier:"saved:cell",label:"Workcell",body_names:["test/left","test/right"],root_name:""};
const view = {selected_body:"test/neighbor",selection_lower:[]};
const frame = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const tool = {
  inspection:{view,saved_assemblies:[group]},
  viewRevision:"displayed-revision",
  async regionSelected(lower,upper) {
    view.selection_lower=lower;
    view.selection_upper=upper;
  },
  async change(label, action) { await action(); return true; },
};
const controller = {
  tools:tool,frame:{matrix:frame},frameOrigin:[10,20,30],
  state:{project_record:{}},plotUpdates:{pending:Promise.resolve()},
  scene:{data:[
    {x:[0,1],y:[0,1],z:[0,1]},
    {x:[3,4],y:[2,3],z:[1,2]},
    {x:[80],y:[80],z:[80]},
  ]},
  readBounds:()=>({lower:view.selection_lower,upper:view.selection_upper}),
  async focusSelection(lower,upper) { this.focus={lower,upper}; },
};
const workspace = {
  controller,tools:tool,
  objects:new Map([["test/left",{indices:[0]}],["test/right",{indices:[1]}]]),
};
(async()=>{
  const assemblies = new AssemblyPanel(workspace);
  workspace.assemblies=assemblies;
  await assemblies.select(group);
  assert.equal(view.selected_body,"");
  assert.equal(view.selected_assembly,group.identifier);
  assert.equal(view.focus_enabled,true);
  const handoff = new InvestigationHandoff(workspace);
  const copied=JSON.parse(JSON.stringify(handoff.context()));
  assert.deepEqual(copied.selected_assembly,{
    identifier:group.identifier,label:group.label,origin:"saved",body_names:group.body_names,
  });
  assert.deepEqual(copied.region_query,{
    expected_revision:"displayed-revision",candidate_index:null,frame_T_scan:frame,
    lower:[9.85,19.85,29.85],upper:[14.15,23.15,32.15],
  });
  assert.deepEqual(JSON.parse(JSON.stringify(controller.focus)),{
    lower:copied.region_query.lower,upper:copied.region_query.upper,
  });
})().catch(error=>{console.error(error);process.exitCode=1;});
