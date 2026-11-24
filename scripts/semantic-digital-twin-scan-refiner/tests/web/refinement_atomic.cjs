// %% Atomic model application
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let release;
const context=vm.createContext({fetch:()=>new Promise(resolve=>{release=resolve;}),document:{getElementById:()=>({textContent:''})}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(async()=>{
 const tools={pending:Promise.resolve(),historyButtons(){},changingHistory:false};
 const panel=new RefinementPanel({controller:{tools},tools,notice(){}});
 panel.proposal={outcome:'proposed'};panel.job={identifier:'test'};panel.clear=async()=>{};panel.controls=()=>{};
 const operation=panel.apply();
 await Promise.resolve();await Promise.resolve();await Promise.resolve();
 globalThis.locked=tools.changingHistory;
 globalThis.operation=operation;globalThis.tools=tools;
})()`,context).then(async()=>{
 assert.equal(context.locked,true,'Undo must be disabled for the entire model mutation');
 release({ok:false});await context.operation;
 assert.equal(context.tools.changingHistory,false,'A failed application must release the history guard');
}).catch(error=>{console.error(error);process.exitCode=1;});
