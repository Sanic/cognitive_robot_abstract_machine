// %% Browser heatmap palette
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements={scan:{value:'heatmap',querySelector:()=>({disabled:false})},candidate:{value:'0'},status:{},'comparison-label':{}};
const context=vm.createContext({document:{getElementById:id=>elements[id]}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('initializeViewer();',''),context);
vm.runInContext(`{
 const scale=[[0,'rgb(0,104,55)'],[1,'rgb(165,0,38)']];let update;
 const controller={scene:{data:[{marker:{colorscale:scale}}]},state:{has_heatmap:true,selected_candidate:0,scan_distances:[0,.1]},scanIsOverview:true,originalRGB:['white','white'],placement:()=>[],scanIndices:()=>[0],plotUpdates:{restyle:value=>{update=value;}}};
 InspectionController.prototype.refreshScan.call(controller);
 globalThis.result={actual:update['marker.colorscale'],expected:scale};
}`,context);
assert.equal(JSON.stringify(context.result.actual),JSON.stringify(context.result.expected),'Interactive distances must reuse the explicit green-to-red scale generated for the preview');
