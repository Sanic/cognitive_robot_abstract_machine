"use strict";
const fs=require('node:fs'),vm=require('node:vm');
let width=400,calls=0;
const scene={getBoundingClientRect:()=>({width,height:200})};
const context=vm.createContext({console,scene,setTimeout,Plotly:{Plots:{async resize(){if(width===0)throw Error('Hidden plot was resized');calls++;}}}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
context.hide=()=>{width=0;};context.show=()=>{width=400;};context.count=()=>calls;
vm.runInContext(`(async()=>{
 const updates=new PlotUpdates(scene);updates.pending=new Promise(resolve=>setTimeout(resolve,15));
 const queued=updates.resize();hide();await queued;
 if(count()!==0)throw Error('Resize was not skipped for a hidden stage');
 show();await updates.resize();if(count()!==1)throw Error('Visible stage did not resize');
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
