"use strict";
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({console});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
vm.runInContext(`(async () => {
 const scene={data:[{type:'scatter3d',marker:{size:2}}]};
 const updates={async addTraces(traces){const indices=traces.map((_,i)=>scene.data.length+i);scene.data.push(...traces);return indices;},async restyle(properties,indices){indices.forEach((index,p)=>{for(const [key,value]of Object.entries(properties))scene.data[index][key]=Array.isArray(value)?value[p]:value;});}};
 const cloud=new ScanCloud(scene,updates,50000);
 for(const count of [50000,150000,10000,150000]){
  await cloud.replace([0,1,2].map(()=>new Float32Array(count)),Array(count).fill('#00ff00'),Array(count).fill('record'));
  if(cloud.indices.reduce((sum,index)=>sum+scene.data[index].x.length,0)!==count)throw Error('Point budget lost records');
  if(cloud.indices.some(index=>scene.data[index].x.length>50000))throw Error('Oversized render batch');
 }
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
