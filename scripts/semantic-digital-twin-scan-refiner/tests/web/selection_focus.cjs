'use strict';

// %% A rotated object must fit even with stale crop bounds and a panned camera
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context=vm.createContext({console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('initializeViewer();',''),context);
const Controller=vm.runInContext('InspectionController',context);
const corners=Array.from({length:8},(_,mask)=>[(mask&1)?2:-2,(mask&2)?.2:-.2,(mask&4)?.5:-.5]);
const yaw=Math.PI/4,rotated=corners.map(([x,y,z])=>[Math.cos(yaw)*x-Math.sin(yaw)*y,Math.sin(yaw)*x+Math.cos(yaw)*y,z]);
const origin=[19,-11,135],crop={lower:[18.9,-11.1,134.9],upper:[19.1,-10.9,135.1]};
const originalCrop=structuredClone(crop);
let layout;
let tools;
const camera={eye:{x:0,y:0,z:.2},center:{x:.6,y:-.4,z:.3},up:{x:0,y:1,z:0},projection:{type:'orthographic'}};
const scene={data:[{type:'mesh3d',name:'world/selected',x:rotated.map(p=>p[0]),y:rotated.map(p=>p[1]),z:rotated.map(p=>p[2])},{type:'mesh3d',name:'world/neighbor',x:[100],y:[100],z:[100]}],_fullLayout:{scene:{camera}},getBoundingClientRect:()=>({width:800,height:500})};
const controller={scene,frameOrigin:origin,meshIndices:[0,1],tools:tools={applying:false,inspection:{view:{selected_body:'world/selected'}}},plotUpdates:{pending:Promise.resolve(),async relayout(value){assert.equal(tools.applying,true,'Programmatic camera fitting must not create a separate camera history action');layout=value;}},readBounds:()=>crop};
(async()=>{
 await Controller.prototype.focusSelection.call(controller,crop.lower,crop.upper);
 for(const point of rotated)for(const [axis,i] of ['x','y','z'].map((axis,i)=>[axis,i])){
  const range=layout[`scene.${axis}axis.range`];
  assert.ok(point[i]>range[0]&&point[i]<range[1],`Rotated ${axis} extent was clipped`);
 }
 assert.equal(layout['scene.aspectmode'],'cube','Equal numeric focus ranges must render with equal physical axis lengths');
 const spans=['x','y','z'].map(axis=>{const r=layout[`scene.${axis}axis.range`];return r[1]-r[0];});
 assert.equal(spans[0],spans[1]);assert.equal(spans[1],spans[2]);assert.ok(spans[0]<10,'An unrelated neighbor must not enlarge focus');
 assert.deepEqual({...layout['scene.camera'].center},{x:0,y:0,z:0});
 assert.ok(Math.hypot(...Object.values(layout['scene.camera'].eye))>Math.hypot(...Object.values(camera.eye)),'Focus must fit rather than reuse the previous close zoom');
 assert.deepEqual({...layout['scene.camera'].projection},camera.projection);
 assert.deepEqual({...layout['scene.camera'].up},camera.up);
 assert.equal(tools.applying,false);
 assert.deepEqual(crop,originalCrop,'Focus must not rewrite the scan crop');
 controller.tools.inspection.view.selected_body='';
 await Controller.prototype.focusSelection.call(controller,[18,-13,132],[20,-9,138]);
 for(const [axis,i] of ['x','y','z'].map((axis,i)=>[axis,i])){
  const range=layout[`scene.${axis}axis.range`];assert.equal((range[0]+range[1])/2,0);
 }
 await Controller.prototype.resetFocus.call({plotUpdates:{async relayout(value){layout=value;}}});
 assert.equal(layout['scene.aspectmode'],'data','Show all must restore the full-world metric aspect');
})().catch(error=>{console.error(error);process.exitCode=1;});
