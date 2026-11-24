// %% Full root-space stretch matrices
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({Float32Array,document:{getElementById:()=>({textContent:''})}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('initializeViewer();',''),context);
vm.runInContext(`(async()=>{
 let values;
 const matrix=[[1.1,.1,0],[.1,1.1,0],[0,0,1]];
 const controller={state:{display_origin:[0,0,0],overrides:[{body_name:'changed',translation:[.1,.2,0]}],dimension_overrides:[{body_name:'changed',scale:[1,1,1],offset:[.3,-.1,0],matrix}]},scene:{data:[{name:'changed'},{name:'fixed'}]},rootCoordinates:[[[1],[2],[3]],[[1],[2],[3]]],meshIndices:[0,1],placement:()=>[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],plotUpdates:{async restyle(data){values=data;},async relayout(){}},frame:null};
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);globalThis.current=[values.x[0][0],values.y[0][0],values.z[0][0]];globalThis.fixed=[values.x[1][0],values.y[1][0],values.z[1][0]];
 controller.previewDimensions={body_name:'changed',scale:[1,1,1],offset:[0,0,0],matrix};
 await InspectionController.prototype.orientFrame.call(controller,'scan',0);globalThis.preview=[values.x[0][0],values.y[0][0],values.z[0][0]];
})()`,context).then(()=>{
 [1.7,2.4,3].forEach((v,i)=>assert.ok(Math.abs(context.current[i]-v)<1e-6));
 assert.equal(JSON.stringify(context.fixed),'[1,2,3]');
 [2.11,2.81,3].forEach((v,i)=>assert.ok(Math.abs(context.preview[i]-v)<1e-6));
}).catch(error=>{console.error(error);process.exitCode=1;});
