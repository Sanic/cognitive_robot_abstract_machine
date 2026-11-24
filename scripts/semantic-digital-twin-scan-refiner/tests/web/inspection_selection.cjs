"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const result=vm.runInContext(`InspectionTools.select(
 {coordinates:[[0,1,2],[0,1,2],[0,1,2]]},
 {clip_enabled:true,clip_lower:[10,20,30],clip_upper:[12,22,32],section_axis:'xy',section_position:31,section_thickness:0.2},
 [10,20,30])`,context);
assert.deepEqual(Array.from(result),[1]);
const full=vm.runInContext(`InspectionTools.select({coordinates:[[0,1],[0,1],[0,1]]},{clip_enabled:false,section_axis:'3d'},[10,20,30])`,context);
assert.deepEqual(Array.from(full),[0,1]);
