// %% Review applicability follows Undo and Redo
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=Object.fromEntries(['review-source','publish-source','source-status','source-download'].map(key=>[key,{hidden:false}]));
const context=vm.createContext({document:{getElementById:key=>elements[key]}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const Panel=vm.runInContext('SourceApplicationPanel',context);
const workspace={tools:{},controller:{state:{project_record:{model_source:'world.urdf'},overrides:[],dimension_overrides:[]}},refiner:{}};
const panel=new Panel(workspace);panel.review={identifier:'review'};panel.reviewKey=panel.revisionKey();panel.published=true;
workspace.controller.state.overrides=[{body_name:'body',translation:[1,0,0]}];panel.controls();
assert.equal(elements['source-download'].hidden,true);assert.equal(elements['publish-source'].disabled,true);
workspace.controller.state.overrides=[];panel.controls();
assert.equal(elements['source-download'].hidden,false,'Redo to the reviewed geometry should restore the validated download');
assert.equal(elements['publish-source'].disabled,true,'An already published review remains immutable');
