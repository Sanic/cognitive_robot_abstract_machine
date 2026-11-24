"use strict";

// %% Local analysis job workspace
class WorkspaceJobs {
  constructor(workspace){this.workspace=workspace;this.controller=workspace.controller;this.records=[];this.starting=false;}

  async initialize() {
    this.reportPanel=new ComparisonPanel();
    document.getElementById('start-registration').onclick=()=>this.start('registration');
    document.getElementById('start-comparison').onclick=()=>this.start('heatmap');
    document.getElementById('upright-confirmed').onchange=()=>this.controls();
    if(location.protocol==='file:'){document.getElementById('start-comparison').disabled=true;return;}
    await this.refresh();
    this.timer=setInterval(()=>this.refresh(),1500);
    window.addEventListener('beforeunload',()=>clearInterval(this.timer),{once:true});
  }

  controls() {
    const active=!!(this.controller.tools?.applying||this.controller.tools?.loading)||this.starting||this.records.some(record=>['queued','running'].includes(record.status));
    document.getElementById('start-registration').disabled=active||!document.getElementById('upright-confirmed').checked;
    document.getElementById('start-comparison').disabled=active||!this.controller.state.registration.candidates.length||!this.controller.tools?.inspection;
    document.getElementById('job-count').textContent=String(this.records.filter(record=>['queued','running'].includes(record.status)).length||this.records.length);
  }

  async start(kind) {
    const tool=this.controller.tools;if(!tool?.inspection)return;
    if(kind==='registration'&&!document.getElementById('upright-confirmed').checked)return;
    const selected=document.getElementById('candidate').value;
    const bounds=this.controller.readBounds();if(kind==='heatmap'&&!bounds)return;
    const body=document.getElementById('comparison-scope').value==='body'?tool.inspection.view.selected_body:'';
    if(kind==='heatmap'&&document.getElementById('comparison-scope').value==='body'&&!body){this.workspace.notice('Select a model body before comparing its surfaces.');return;}
    const request={__json_type__:'scan_refiner.jobs.JobRequest',kind,candidate_index:selected===''?null:Number(selected),
      geometry_role:document.getElementById(kind==='registration'?'registration-role':'comparison-role').value,body_name:kind==='heatmap'?body:'',
      lower:kind==='heatmap'?bounds.lower:[],upper:kind==='heatmap'?bounds.upper:[],frame_T_scan:this.controller.frame.matrix,
      maximum_count:Number(document.getElementById('comparison-budget').value),model_sample_count:Number(document.getElementById('comparison-model-budget').value),tolerance:Number(document.getElementById('comparison-tolerance').value)/1000,
      upright:document.getElementById('upright-confirmed').checked,
      search:{__json_type__:'scan_refiner.registration.RegistrationSearch',cell_width:Number(document.getElementById('registration-cell').value),yaw_step_degrees:Number(document.getElementById('registration-yaw').value)}};
    if(!Number.isFinite(request.tolerance)||request.tolerance<=0){this.workspace.notice('Enter a positive comparison tolerance.');return;}
    this.starting=true;this.controls();
    try {
      await tool.pending;
      const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(InspectionTools.encode(request))});
      if(!response.ok)throw Error(response.statusText||`Cannot start job (${response.status}). Check the selected geometry and settings.`);
      document.getElementById('jobs-drawer').hidden=false;
      await this.refresh();
    }catch(error){this.workspace.notice(error.message);}
    finally{this.starting=false;this.controls();}
  }

  async refresh() {
    if(this.refreshing)return;this.refreshing=true;
    try {
      const response=await fetch('/api/jobs');if(!response.ok)return;
      this.records=InspectionController.decode(await response.json());this.draw();this.controls();this.updateSummary();this.workspace.refiner?.controls();this.workspace.changeSets?.controls();
      await this.synchronizeRevision();
    }catch(error){document.getElementById('job-count').textContent='—';}
    finally{this.refreshing=false;}
  }

  async synchronizeRevision() {
    const response=await fetch('/api/agent/revision');if(!response.ok)return;
    const current=InspectionController.decode(await response.json()).revision;
    const tools=this.controller.tools;
    const shown=tools?.viewRevision||this.observedRevision;
    if(!shown||current===shown){this.observedRevision=current;return;}
    if(!tools||tools.applying||tools.loading||tools.changingHistory||this.workspace.refiner?.busy||this.workspace.sourceApplication?.busy||this.workspace.changeSets?.busy)return;
    await tools.pending;
    await tools.refreshProject();
    this.observedRevision=current;
    this.workspace.synchronize();
    this.workspace.notice('The shared project changed. The view now shows the current geometry and registration.');
  }

  async action(record,operation) {
    try {
      if(operation==='review-refinement'){await this.workspace.refiner.review(record);return;}
      if(operation==='show'){
        await this.controller.showComparison(record);
        document.getElementById('comparison-role').value=record.request.geometry_role;
        this.summaryKey=null;this.updateSummary();this.workspace.navigate('inspect');this.workspace.inspector('results');
        document.getElementById('jobs-drawer').hidden=true;this.workspace.synchronize();return;
      }
      await this.controller.tools.pending;
      const response=await fetch(`/api/jobs/${record.identifier}/${operation}`,{method:'POST'});
      if(!response.ok)throw Error(`Job action failed (${response.status}); the result may be stale.`);
      if(operation==='apply'){
        await this.controller.tools.refreshProject();this.workspace.navigate('align');document.getElementById('jobs-drawer').hidden=true;this.workspace.synchronize();
      }
      await this.refresh();
    }catch(error){this.workspace.notice(error.message);}
  }

  updateSummary() {
    const source=this.controller.tools?.restoredPreview?this.controller.previewCloud:this.controller.currentCloud;
    const job=source?.comparison?.job,view=this.controller.tools?.inspection?.view;
    const known=this.records.find(record=>record.identifier===job?.identifier);
    const valid=!(this.controller.previewDelta||this.controller.previewDimensions)&&JSON.stringify(this.controller.state?.dimension_overrides||[])===JSON.stringify(source?.comparison?.dimension_overrides||[])&&JSON.stringify(this.controller.state?.overrides||[])===JSON.stringify(source?.comparison?.overrides||[])&&job&&view?.comparison_job===job.identifier&&JSON.stringify(this.controller.placement())===JSON.stringify(job.scan_T_root)&&known?.status!=='stale';
    const summary=document.getElementById('comparison-summary');
    if(!valid){summary.hidden=true;this.summaryKey=null;this.reportPanel?.clear();return;}
    if(this.summaryKey===job.identifier){summary.hidden=false;return;}
    summary.replaceChildren();
    for(const text of [`${job.request.geometry_role==='visual'?'Visual':'Collision'} surfaces · ${job.total.toLocaleString()} evaluated points`,
      `Median ${(job.median_distance*1000).toFixed(1)} mm · P90 ${(job.percentile90_distance*1000).toFixed(1)} mm`,
      `${(job.within_tolerance*100).toFixed(1)}% of points within ${(job.request.tolerance*1000).toFixed(0)} mm`,
      'Unsigned scan-to-surface distances; this is a sampled comparison.']){
      const row=document.createElement('div');row.textContent=text;summary.append(row);
    }
    this.reportPanel?.show(source.comparison.report,job.identifier);
    this.summaryKey=job.identifier;summary.hidden=false;
  }

  draw() {
    const list=document.getElementById('job-list');list.replaceChildren();
    if(!this.records.length){const empty=document.createElement('p');empty.className='hint';empty.textContent='Registration and regional comparison jobs will appear here. Inspecting the scene remains available while a worker runs.';list.append(empty);return;}
    for(const record of this.records){
      const card=document.createElement('article');card.className='job-card';card.dataset.job=record.identifier;
      const heading=document.createElement('h3');heading.textContent=record.request.kind==='registration'?'Automatic placement search':record.request.kind==='refinement'?'Constrained body translation':record.request.kind==='dimensions'?'Constrained box dimensions':record.request.kind==='change-set-heatmap'?'Paired proposal heatmaps':'Regional surface comparison';card.append(heading);
      const state=document.createElement('span');state.className='job-state';state.textContent=record.status;card.append(state);
      const message=document.createElement('p');message.textContent=record.message;card.append(message);
      if(['running','queued'].includes(record.status)){
        const progress=document.createElement('progress');if(record.total){progress.max=record.total;progress.value=record.completed;}card.append(progress);
        if(record.total){const count=document.createElement('p');count.textContent=`${record.completed.toLocaleString()} / ${record.total.toLocaleString()} records`;card.append(count);}
      }
      if(record.status==='succeeded'&&record.request.kind==='heatmap'){
        const metrics=document.createElement('p');metrics.textContent=`${record.total.toLocaleString()} points · median ${(record.median_distance*1000).toFixed(1)} mm · P90 ${(record.percentile90_distance*1000).toFixed(1)} mm`;card.append(metrics);
      }
      let operation=null,label='';
      if(['running','queued'].includes(record.status)){operation='cancel';label='Cancel job';}
      else if(record.status==='succeeded'&&record.request.kind!=='change-set-heatmap'){operation=record.request.kind==='registration'?'apply':['refinement','dimensions'].includes(record.request.kind)?'review-refinement':'show';label=operation==='apply'?'Review candidates':operation==='review-refinement'?(record.request.kind==='dimensions'?'Review dimensions':'Review translation'):'Show heatmap';}
      if(operation){const button=document.createElement('button');button.textContent=label;button.className=operation==='cancel'?'':'secondary';button.onclick=()=>this.action(record,operation);card.append(button);}
      list.append(card);
    }
  }
}
