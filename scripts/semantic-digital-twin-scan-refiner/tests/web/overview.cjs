// %% Frozen overview restoration
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=Object.fromEntries(['scan','clip-scan','section-axis','region-status'].map(name=>[name,{}]));
const context=vm.createContext({document:{getElementById:name=>elements[name]},Float32Array});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('initializeViewer();',''),context);
vm.runInContext(fs.readFileSync(process.argv[3],'utf8'),context);
(async()=>{
 const view={region_lower:[1,2,3],region_upper:[2,3,4],comparison_job:'old',clip_enabled:true,selected_body:'box',selection_lower:[1,2,3],section_axis:'xy',focus_enabled:true};
 let renders=0,aborted=false,label;
 const preview={coordinates:[[0,1],[0,2],[0,3]],colors:['a','b'],identities:['0:1','0:2']};
 const controller={indexInformation:{index:{lower:[0,0,0],upper:[10,10,10]}},previewCloud:preview,currentCloud:{},regionalHeatmap:{},pendingRegion:{abort(){aborted=true;}},frame:{bounds:()=>({lower:[0,0,0],upper:[10,10,10]})},
  tools:{inspection:{view},restoredPreview:false,async render(){renders++;},async save(value){label=value;}},async loadRegion(){throw Error('Overview must not read the dense index');},writeBounds(){},regionBusy(){},async resetFocus(){},refreshScan(){},workspace:{synchronize(){}}};
 await vm.runInContext('InspectionController',context).prototype.loadOverview.call(controller);
 assert.equal(controller.currentCloud,preview);assert.equal(controller.tools.restoredPreview,true);assert.equal(renders,1);assert.equal(aborted,true);
 assert.deepEqual(Array.from(view.region_lower),[]);assert.deepEqual(Array.from(view.region_upper),[]);assert.equal(view.comparison_job,'');assert.equal(view.selected_body,'');assert.equal(view.clip_enabled,false);assert.equal(view.section_axis,'3d');assert.equal(label,'World overview');
})().catch(error=>{console.error(error);process.exitCode=1;});
