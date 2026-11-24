"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({console,Plotly:{async addTraces(scene,traces){await new Promise(r=>setTimeout(r,5));scene.data.push(...traces);return scene;}}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(async()=>{
 const scene={data:[]},updates=new PlotUpdates(scene);
 const indices=await Promise.all([updates.addTraces([{name:'scan'}]),updates.addTraces([{name:'measurement'}])]);
 if(JSON.stringify(indices)!=='[[0],[1]]')throw Error('Concurrent layers did not receive their actual trace indices');
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
