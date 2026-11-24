'use strict';

// %% Visible activity outlives nested scene operations
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const fields=Object.fromEntries(['title','detail','retry'].map(key=>[key,{hidden:false,textContent:''}]));
const lock={inert:false},alreadyLocked={inert:true};
const classes=new Set();
const document={querySelectorAll:()=>[lock,alreadyLocked],body:{classList:{toggle(name,on){if(on)classes.add(name);else classes.delete(name);}}}};
const element={hidden:false,dataset:{},ownerDocument:document,parentElement:{setAttribute(name,value){this[name]=value;}},querySelector:selector=>fields[selector.match(/data-activity-(\w+)/)[1]]};
const context=vm.createContext({requestAnimationFrame:callback=>setTimeout(callback,0),setTimeout,clearTimeout,console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const Activity=vm.runInContext('ViewerActivity',context),activity=new Activity(element);
(async()=>{
 const undo=activity.begin('Undoing last action','Restoring project state');
 assert.equal(element.hidden,false);assert.equal(lock.inert,true);assert.equal(alreadyLocked.inert,true);
 const region=activity.begin('Loading scan region','Reading original scan points');
 assert.equal(fields.title.textContent,'Undoing last action');assert.equal(fields.detail.textContent,'Reading original scan points');
 activity.end(region);assert.equal(element.hidden,false);assert.equal(fields.detail.textContent,'Restoring project state');
 activity.end(undo);assert.equal(element.hidden,true);assert.equal(lock.inert,false);assert.equal(alreadyLocked.inert,true);
 let painted=false;context.requestAnimationFrame=callback=>{painted=true;setTimeout(callback,0);};
 await activity.run('Updating geometry','Rendering the selected surfaces',async()=>{assert.equal(painted,true);assert.equal(element.hidden,false);});
 assert.equal(element.hidden,true);
 await assert.rejects(activity.run('Restoring view','Rendering',async()=>{throw Error('Render failed');}));
 assert.equal(element.hidden,true);assert.equal(lock.inert,false);
 activity.fail('Cannot initialize the scene');assert.equal(element.hidden,false);assert.equal(element.dataset.state,'error');assert.equal(fields.retry.hidden,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
