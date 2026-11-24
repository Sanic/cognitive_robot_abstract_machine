"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({console,setTimeout});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(async()=>{
 const tool=new InspectionTools({});let value=0;const labels=[];
 tool.historyButtons=()=>{};tool.save=async label=>{labels.push(label);};
 const first=tool.change('First',async()=>{await new Promise(r=>setTimeout(r,15));value=1;});
 const second=tool.change('Overlap',async()=>{value=2;});
 await Promise.all([first,second]);
 if(value!==1||JSON.stringify(labels)!=='["First"]'||tool.applying)throw Error('Overlapping view actions were interleaved');
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
