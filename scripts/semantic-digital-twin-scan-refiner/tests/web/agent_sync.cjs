// %% Shared-agent revision notifications
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let revision='before',refreshes=0,notices=0;
const context=vm.createContext({fetch:async()=>({ok:true,json:async()=>({revision})}),InspectionController:{decode:value=>value}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
(async()=>{
 const Tools=vm.runInContext('WorkspaceJobs',context);
 const tools={pending:Promise.resolve(),async refreshProject(){refreshes++;}};
 const jobs=new Tools({controller:{tools},synchronize(){},notice(){notices++;}});
 await jobs.synchronizeRevision();assert.equal(refreshes,0);
 revision='after';tools.applying=true;await jobs.synchronizeRevision();assert.equal(refreshes,0);
 tools.applying=false;await jobs.synchronizeRevision();assert.equal(refreshes,1);assert.equal(notices,1);
 await jobs.synchronizeRevision();assert.equal(refreshes,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
