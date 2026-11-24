'use strict';

// %% Plotly can emit a click before the drag gesture is resolved
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const handlers={},plotly={};
const target={addEventListener(name,callback){handlers[name]=callback;}};
const scene={ownerDocument:{defaultView:target},addEventListener(name,callback){handlers[name]=callback;},on(name,callback){plotly[name]=callback;}};
const context=vm.createContext({setTimeout,clearTimeout,console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const Updates=vm.runInContext('PlotUpdates',context),updates=new Updates(scene);
let selections=0,measurements=0;
updates.registerClick(()=>selections++);
updates.registerClick(()=>measurements++);
const point={points:[{curveNumber:1,x:0,y:0,z:0}]};
const mouse=(x,y)=>({button:0,clientX:x,clientY:y});
const wait=()=>new Promise(resolve=>setTimeout(resolve,10));
(async()=>{
 handlers.mousedown(mouse(100,100));plotly.plotly_click(point);
 assert.equal(selections,0,'A press must not redraw a scene while Plotly is starting an orbit');
 handlers.mousemove(mouse(180,120));handlers.mouseup(mouse(180,120));
 await wait();assert.equal(selections,0);assert.equal(measurements,0);
 plotly.plotly_click(point);await wait();assert.equal(selections,0,'Ignore a late click belonging to the drag');
 handlers.mousedown(mouse(100,100));plotly.plotly_click(point);handlers.mouseup(mouse(101,101));
 await wait();assert.equal(selections,1);assert.equal(measurements,1);
 handlers.mousedown(mouse(100,100));handlers.mouseup(mouse(100,100));plotly.plotly_click(point);
 await wait();assert.equal(selections,2,'A renderer emitting its click on release must also work');
 handlers.mousedown(mouse(100,100));plotly.plotly_click(point);handlers.blur();
 await wait();assert.equal(selections,2,'Leaving the browser cancels an unfinished gesture');
})().catch(error=>{console.error(error);process.exitCode=1;});
