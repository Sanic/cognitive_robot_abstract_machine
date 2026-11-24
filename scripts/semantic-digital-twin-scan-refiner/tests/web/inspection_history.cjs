"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
function element(){return {disabled:false,value:'',addEventListener(){},replaceChildren(){},append(){}};}
const session={history:{current:{record:{accepted_candidate:null},registration:{candidates:[]},inspection:{view:{},measurements:[],bookmarks:[]},label:'Saved view'},past:[{}],future:[]}};
const context=vm.createContext({location:{protocol:'http:'},console,fetch:async()=>({ok:true,json:async()=>session}),document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element},Option:class{}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(async()=>{
 const tools=new InspectionTools({state:{registration:{candidates:[]}},scene:{on(){}},plotUpdates:{registerClick(){}},decodeRecord:value=>value});
 tools.apply=async()=>{tools.applying=true;tools.regionLoading(false);tools.applying=false;};
 await tools.initialize();
 if(document.getElementById('undo').disabled)throw Error('Restoring a saved view left Undo disabled');
 tools.applying=true;tools.historyButtons();
 if(!document.getElementById('undo').disabled)throw Error('Undo enabled while the previous restore was rendering');
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
