"use strict";
const fs=require('node:fs'),vm=require('node:vm');
const summary={hidden:false,rows:[],replaceChildren(){this.rows=[];},append(row){this.rows.push(row);}};
const context=vm.createContext({console,document:{getElementById:()=>summary,createElement:()=>({textContent:''})}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`(()=>{
 const transform=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 const job={identifier:'comparison',scan_T_root:transform,total:100,median_distance:0.01,percentile90_distance:0.02,within_tolerance:0.9,request:{geometry_role:'visual',tolerance:0.02}};
 const controller={tools:{inspection:{view:{comparison_job:job.identifier}}},currentCloud:{comparison:{job}},placement:()=>transform};
 const jobs=new WorkspaceJobs({controller});jobs.records=[{identifier:job.identifier,status:'stale'}];jobs.updateSummary();
 if(!document.getElementById('comparison-summary').hidden)throw Error('Stale metrics remained visible');
 jobs.records[0].status='succeeded';jobs.updateSummary();
 if(document.getElementById('comparison-summary').hidden)throw Error('Valid restored metrics remained hidden');
})()`,context);
