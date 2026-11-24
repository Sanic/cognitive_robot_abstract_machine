"use strict";
const fs=require('node:fs'),vm=require('node:vm');
const elements={'coordinate-frame':{value:'scan'},'frame-yaw':{value:'30'},'clip-scan':{checked:false}};
const context=vm.createContext({console,document:{getElementById:id=>elements[id]}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(async()=>{
 const view={coordinate_frame:'world',frame_yaw:0,selected_body:'',focus_enabled:false};let applied;
 const tools={inspection:{view},async change(label,operation){document.getElementById('coordinate-frame').value='world';await operation();},restoreBounds(){},async render(){}};
 const controller={tools,async orientFrame(mode,yaw){applied=[mode,yaw];},async resetFocus(){}};
 const workspace=new RefinerWorkspace(controller);workspace.synchronize=()=>{};
 await workspace.orient();
 if(view.coordinate_frame!=='scan'||JSON.stringify(applied)!=='["scan",30]')throw Error('Refreshing controls overwrote the requested frame choice');
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
