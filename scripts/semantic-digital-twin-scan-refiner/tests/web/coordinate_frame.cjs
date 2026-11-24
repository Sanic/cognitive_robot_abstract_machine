"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const values=vm.runInContext(`(()=>{
 const placement=[[0,-1,0,4],[1,0,0,-2],[0,0,1,-136],[0,0,0,1]];
 const world=CoordinateFrame.fromPlacement(placement,'world',0);
 const point=world.point([4,-1,-136]);
 const recovered=world.toScan(point);
 const custom=CoordinateFrame.fromPlacement(placement,'world',90).point([4,-1,-136]);
 return {point,recovered,custom};
})()`,context);
assert.deepEqual(Array.from(values.point),[1,0,0]);
assert.deepEqual(Array.from(values.recovered),[4,-1,-136]);
assert.ok(Math.abs(values.custom[0])<1e-12);
assert.equal(values.custom[1],-1);
