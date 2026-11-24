// %% Reversible geometry buffers
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({Float32Array,document:{getElementById:()=>({textContent:''})}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('initializeViewer();',''),context);
vm.runInContext(`(async()=>{
 let values;
 const controller={state:{display_origin:[0,0,0],overrides:[{body_name:'moved',translation:[0,0,.02]}]},scene:{data:[{name:'moved'},{name:'fixed'}]},rootCoordinates:[[[0],[0],[0]],[[1],[0],[0]]],meshIndices:[0,1],placement:()=>[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],plotUpdates:{async restyle(data){values=data;},async relayout(){}},frame:null};
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);
 globalThis.moved=values.z.map(axis=>Array.from(axis));
 controller.state.overrides=[];
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);
 globalThis.restored=values.z.map(axis=>Array.from(axis));
})()`,context).then(()=>{
 assert.ok(Math.abs(context.moved[0][0]-.02)<1e-8,'The selected body must move in the browser');
 assert.equal(context.moved[1][0],0,'Other bodies must remain fixed');
 assert.equal(context.restored[0][0],0,'Undo must rebuild coordinates from the immutable baseline');
}).catch(error=>{console.error(error);process.exitCode=1;});
