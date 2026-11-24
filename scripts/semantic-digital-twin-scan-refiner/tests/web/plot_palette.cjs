// %% Nested Plotly values
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let received;
const context=vm.createContext({Plotly:{restyle:async(scene,properties)=>{received=properties;}}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(async()=>{let palette=[[0,'green'],[1,'red']];await new PlotUpdates({}).restyle({'marker.colorscale':palette},[0,1]);})()`,context).then(()=>{
 assert.equal(JSON.stringify(received['marker.colorscale']),'[[[0,"green"],[1,"red"]]]','Plotly needs an outer per-trace array around a colour scale');
}).catch(error=>{console.error(error);process.exitCode=1;});
