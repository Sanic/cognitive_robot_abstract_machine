'use strict';

// %% Heavy work must wait until its loading feedback has painted
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({console,setTimeout});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const Tools=vm.runInContext('InspectionTools',context),tool=new Tools({});
const events=[];
tool.activity={begin(label){events.push(['begin',label]);return label;},async paint(){events.push(['paint']);},end(label){events.push(['end',label]);}};
tool.historyButtons=()=>{};tool.save=async label=>events.push(['save',label]);
(async()=>{
 await tool.change('Orient axes',async()=>{assert.equal(events.at(-1)[0],'paint','Transforms must not block before feedback is painted');events.push(['work']);});
 assert.deepEqual(events.map(item=>item[0]),['begin','paint','work','end','save']);
 events.length=0;
 await assert.rejects(tool.change('Failing redraw',async()=>{throw Error('Renderer rejected');}));
 assert.equal(tool.applying,false);assert.equal(events.at(-1)[0],'end');
})().catch(error=>{console.error(error);process.exitCode=1;});
