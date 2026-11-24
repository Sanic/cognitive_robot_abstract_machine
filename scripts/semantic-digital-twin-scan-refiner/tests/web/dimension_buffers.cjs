// %% Dimensional root-frame composition
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({Float32Array,document:{getElementById:()=>({textContent:''})}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('initializeViewer();',''),context);
vm.runInContext(`(async()=>{
 let values;
 const controller={state:{display_origin:[0,0,0],overrides:[{body_name:'changed',translation:[.1,0,0]}],dimension_overrides:[{body_name:'changed',scale:[1.2,1,1],offset:[-.02,0,0]}]},scene:{data:[{name:'changed'},{name:'fixed'}]},rootCoordinates:[[[1],[0],[0]],[[1],[0],[0]]],meshIndices:[0,1],placement:()=>[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],plotUpdates:{async restyle(data){values=data;},async relayout(){}},frame:null};
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);globalThis.current=values.x.map(axis=>Array.from(axis));
 controller.previewDimensions={body_name:'changed',scale:[1.1,1,1],offset:[-.028,0,0]};
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);globalThis.preview=values.x.map(axis=>Array.from(axis));
 controller.previewDimensions=null;controller.state.overrides=[];controller.state.dimension_overrides=[];
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);globalThis.restored=values.x.map(axis=>Array.from(axis));
})()`,context).then(()=>{
 assert.ok(Math.abs(context.current[0][0]-1.28)<1e-6,'Resize must precede the separate root translation');
 assert.equal(context.current[1][0],1,'Other bodies stay fixed');
 assert.ok(Math.abs(context.preview[0][0]-1.38)<1e-6,'Preview deformation applies to current effective geometry');
 assert.equal(context.restored[0][0],1,'Undo rebuilds the immutable baseline coordinates');
}).catch(error=>{console.error(error);process.exitCode=1;});
