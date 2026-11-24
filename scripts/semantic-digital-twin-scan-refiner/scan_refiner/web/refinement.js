"use strict";

// %% Selected-body proposal review
class RefinementPanel {
  constructor(workspace){this.workspace=workspace;this.controller=workspace.controller;this.tools=workspace.tools;this.active=false;this.job=null;this.proposal=null;this.ghostIndices=[];this.axisIndices=[];this.busy=false;}

  initialize() {
    document.getElementById('start-refinement').onclick=()=>this.start();
    document.getElementById('refine-preview').onchange=()=>this.preview();
    document.getElementById('clear-refinement-preview').onclick=()=>this.clear();
    document.getElementById('apply-refinement').onclick=()=>this.apply();
    for(const id of ['refine-x','refine-y','refine-z'])document.getElementById(id).onchange=()=>this.controls();
    document.getElementById('refine-mode').onchange=()=>this.discardReview();
    document.getElementById('refine-role').onchange=()=>this.discardReview();
    document.getElementById('dimension-frame').onchange=()=>this.discardReview();
    for(const axis of ['x','y','z'])document.getElementById(`dimension-${axis}`).onchange=()=>this.controls();
    this.controls();
  }

  controls() {
    document.querySelectorAll('.stage-nav [data-stage]').forEach(button=>{button.disabled=this.busy;});
    for(const id of ['geometry','scan','point-size','mesh-opacity','point-hover','candidate'])document.getElementById(id).disabled=this.active||this.busy||(id==='candidate'&&!this.controller.state.registration.candidates.length);
    const dimensions=document.getElementById('refine-mode').value==='dimensions';
    document.getElementById('translation-settings').hidden=dimensions;document.getElementById('dimension-settings').hidden=!dimensions;
    document.getElementById('refine-mode').disabled=this.busy;document.getElementById('refine-role').disabled=this.busy;document.getElementById('dimension-frame').disabled=this.busy;
    document.getElementById('start-refinement').textContent=dimensions?'Propose box dimensions':'Propose translation';
    const body=this.tools?.inspection?.view.selected_body;
    if(dimensions&&body){const key=JSON.stringify([body,document.getElementById('refine-role').value,document.getElementById('dimension-frame').value,this.controller.state.overrides||[],this.controller.state.dimension_overrides||[]]);if(key!==this.dimensionKey)this.loadDimensionInfo(key,body);}
    for(const axis of ['x','y','z'])document.getElementById(`dimension-anchor-${axis}`).disabled=!document.getElementById(`dimension-${axis}`).checked;

    document.getElementById('refine-body').textContent=body||'Select a body in the world contents';
    const pending=this.busy||this.tools?.applying||this.tools?.loading||this.tools?.changingHistory||this.workspace.changeSets?.active||this.workspace.jobs?.records.some(job=>['running','queued'].includes(job.status));
    document.getElementById('start-refinement').disabled=!!pending||!body||!this.controller.state.registration.candidates.length||!['x','y','z'].some(axis=>document.getElementById(`${dimensions?'dimension':'refine'}-${axis}`).checked)||(dimensions&&(this.dimensionLoading||!this.dimensionInfo||!!this.dimensionInfo.problem));
    const known=this.workspace.jobs?.records.find(job=>job.identifier===this.job?.identifier);
    const valid=this.job&&(!known||known.status==='succeeded');
    document.getElementById('apply-refinement').disabled=!!pending||!valid||this.proposal?.outcome!=='proposed';
    document.getElementById('refine-preview').disabled=!!pending||!valid||!this.proposal?.matched_points||!this.hasMotion();
    document.getElementById('clear-refinement-preview').disabled=!this.active||!!pending;
    const sizeOverride=this.controller.state.dimension_overrides?.find(item=>item.body_name===body);
    const override=this.controller.state.overrides?.find(item=>item.body_name===body);
    document.getElementById('body-override').textContent=override?`Current override XYZ: ${override.translation.map(value=>(value*1000).toFixed(2)).join(', ')} mm`:sizeOverride?'A dimensional snapshot override is applied to this body.':'Current body uses the frozen baseline pose.';
    if(known?.status==='applied')document.getElementById('refine-status').textContent=[...(this.controller.state.overrides||[]),...(this.controller.state.dimension_overrides||[])].some(item=>item.source_job===this.job.identifier)?'This snapshot override is applied in the current revision. Undo restores the previous model.':'This proposal was applied in an earlier revision. Undo/Redo controls the current model state.';
    if(known?.status==='stale')document.getElementById('refine-status').textContent='This proposal belongs to earlier geometry or registration. Generate a new proposal.';
  }

  hasMotion() {
    if(!this.proposal)return false;
    return this.job?.request.kind==='dimensions'?this.proposal.scale.some(value=>Math.abs(value-1)>1e-12)||this.proposal.offset.some(value=>value!==0):this.proposal.translation.some(value=>value!==0);
  }

  async loadDimensionInfo(key,body) {
    this.dimensionKey=key;this.dimensionInfo=null;this.dimensionLoading=true;
    document.getElementById('dimension-info').textContent='Checking box geometry in both roles…';
    try {
      const query=new URLSearchParams({body,role:document.getElementById('refine-role').value,frame:document.getElementById('dimension-frame').value});
      const response=await fetch(`/api/body-dimensions?${query}`);if(!response.ok)throw Error('Cannot check box geometry.');
      const info=InspectionController.decode(await response.json());if(this.dimensionKey!==key)return;
      this.dimensionInfo=info;
      document.getElementById('dimension-info').textContent=info.problem||`Current ${document.getElementById('dimension-frame').value==='box'?'box':'root'} X/Y/Z extents: ${info.bounds.upper.map((value,axis)=>((value-info.bounds.lower[axis])*1000).toFixed(2)).join(', ')} mm`;
    }catch(error){if(this.dimensionKey===key){this.dimensionInfo={problem:error.message};document.getElementById('dimension-info').textContent=error.message;}}
    finally{if(this.dimensionKey===key){this.dimensionLoading=false;this.controls();}}
  }

  async discardReview() {
    if(this.busy)return;
    await this.clear();this.job=null;this.proposal=null;
    document.getElementById('translation-report').hidden=true;
    document.getElementById('refine-status').textContent='Generate a proposal or choose an existing one in Jobs.';this.controls();
  }

  async start() {
    if(this.busy)return;
    const bounds=this.controller.readBounds(),body=this.tools.inspection.view.selected_body;if(!bounds||!body)return;
    await this.clear();
    const dimensions=document.getElementById('refine-mode').value==='dimensions';
    const request={__json_type__:'scan_refiner.jobs.JobRequest',kind:dimensions?'dimensions':'refinement',candidate_index:Number(document.getElementById('candidate').value),body_name:body,
      geometry_role:document.getElementById('refine-role').value,lower:bounds.lower,upper:bounds.upper,frame_T_scan:this.controller.frame.matrix,
      maximum_count:Number(document.getElementById('refine-budget').value),tolerance:Number(document.getElementById('refine-tolerance').value)/1000,
      constraints:{__json_type__:'scan_refiner.refinement.TranslationConstraints',axes:['x','y','z'].map(axis=>document.getElementById(`refine-${axis}`).checked),maximum_translation:Number(document.getElementById('refine-limit').value)/1000,association_distance:Number(document.getElementById('refine-association').value)/1000}};
    if(dimensions)request.dimension_constraints={__json_type__:'scan_refiner.dimensions.DimensionConstraints',frame_mode:document.getElementById('dimension-frame').value,axes:['x','y','z'].map(axis=>document.getElementById(`dimension-${axis}`).checked),anchors:['x','y','z'].map(axis=>document.getElementById(`dimension-anchor-${axis}`).value),maximum_change:Number(document.getElementById('dimension-limit').value)/1000,association_distance:Number(document.getElementById('refine-association').value)/1000};
    this.busy=true;this.controls();
    try {
      await this.tools.pending;
      const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(InspectionTools.encode(request))});
      if(!response.ok)throw Error('Cannot start this fit. Check the crop, axes and positive limits.');
      document.getElementById('jobs-drawer').hidden=false;await this.workspace.jobs.refresh();
    }catch(error){this.workspace.notice(error.message);}
    finally{this.busy=false;this.controls();}
  }

  async review(job) {
    await this.workspace.changeSets?.close();
    const advanced=document.getElementById('single-body-refinement');if(advanced)advanced.open=true;
    if(this.busy)return;this.busy=true;this.controls();
    try {
    await this.clear();
    const response=await fetch(`/api/jobs/${job.identifier}/proposal`);
    if(!response.ok){this.workspace.notice('This proposal is unfinished or stale.');return;}
    this.job=job;this.proposal=InspectionController.decode(await response.json());
    const dimensions=job.request.kind==='dimensions';document.getElementById('refine-mode').value=dimensions?'dimensions':'translation';
    document.getElementById('refine-role').value=job.request.geometry_role;
    if(dimensions)document.getElementById('dimension-frame').value=this.proposal.constraints.frame_mode||'root';
    document.getElementById('refine-tolerance').value=String(job.request.tolerance*1000);
    document.getElementById('refine-association').value=String(this.proposal.constraints.association_distance*1000);
    document.getElementById(dimensions?'dimension-limit':'refine-limit').value=String((dimensions?this.proposal.constraints.maximum_change:this.proposal.constraints.maximum_translation)*1000);
    const budget=document.getElementById('refine-budget');if(!Array.from(budget.options).some(option=>option.value===String(job.request.maximum_count)))budget.add(new Option(`${job.request.maximum_count.toLocaleString()} (saved job)`,String(job.request.maximum_count)));budget.value=String(job.request.maximum_count);
    ['x','y','z'].forEach((axis,index)=>{document.getElementById(`${dimensions?'dimension':'refine'}-${axis}`).checked=this.proposal.constraints.axes[index];if(dimensions)document.getElementById(`dimension-anchor-${axis}`).value=this.proposal.constraints.anchors[index];});
    this.workspace.navigate('refine');document.getElementById('jobs-drawer').hidden=true;
    const outcome={proposed:'Improved geometry proposal ready for review.',not_converged:'The optimizer did not finish. This intermediate solution cannot be applied.',too_few_points:'Too few nearby records. Tighten the crop or check the association distance.',underconstrained:dimensions?'Required face evidence is missing. Centered resizing needs both faces; anchored resizing needs its moving face.':'The scan does not constrain all unlocked axes. Lock unsupported directions.',at_bound:'The fit reached a movement limit. It cannot be applied; inspect the evidence and limits.',no_improvement:'No sufficient residual improvement. The current pose is retained.'};
    document.getElementById('refine-status').textContent=outcome[this.proposal.outcome]||this.proposal.outcome;
    const report=document.getElementById('translation-report');report.replaceChildren();report.hidden=false;
    const geometry=dimensions?[
      `Before XYZ: ${this.proposal.before_box.upper.map((value,axis)=>((value-this.proposal.before_box.lower[axis])*1000).toFixed(2)).join(', ')} mm`,
      `Proposal XYZ: ${this.proposal.after_box.upper.map((value,axis)=>((value-this.proposal.after_box.lower[axis])*1000).toFixed(2)).join(', ')} mm`,
      ...this.proposal.face_support.map(face=>`${face.axis} minimum / maximum returns: ${face.minimum_count} / ${face.maximum_count}`)
    ]:[`Normal rank ${this.proposal.normal_rank} / ${this.proposal.constraints.axes.filter(Boolean).length}`,`ΔXYZ: ${this.proposal.translation.map(value=>(value*1000).toFixed(2)).join(', ')} mm`];
    for(const text of [`${dimensions?'Box dimensions':'Translation'} · ${this.proposal.body_name}`,`${this.proposal.matched_points.toLocaleString()} / ${this.proposal.available_points.toLocaleString()} associated records`,...geometry,
      ...['before','after'].map(key=>{const metric=this.proposal[key];return metric.sample_count?`${key==='before'?'Before':'Proposal'}: median ${(metric.median_distance*1000).toFixed(2)} mm · P90 ${(metric.percentile90_distance*1000).toFixed(2)} mm`:`${key}: no associated records`;})]){
      const row=document.createElement('div');row.textContent=text;report.append(row);
    }
    document.getElementById('refine-preview').value='proposal';this.controls();
    }finally{this.busy=false;this.controls();}
    if(this.hasMotion()&&this.proposal.matched_points)await this.preview();
  }

  async preview() {
    if(!this.job||!this.proposal||this.busy)return;
    const applying=this.tools.applying;this.tools.applying=true;this.tools.historyButtons();
    this.busy=true;this.controls();
    const activity=this.controller.activity?.begin('Preparing refinement preview','Loading the proposal evidence and drawing current/proposed geometry…');
    try {
      await this.controller.activity?.paint();
      if(!this.active){
        const response=await fetch(`/api/jobs/${this.job.identifier}/points`);if(!response.ok)throw Error('The proposal evidence is stale or unavailable.');
        const source=this.controller.decodeCloud(await response.arrayBuffer());
        this.previous={cloud:this.controller.currentCloud,restoredPreview:this.tools.restoredPreview,view:structuredClone(this.tools.inspection.view),meshColors:this.controller.meshIndices.map(index=>this.controller.scene.data[index].color)};
        const originals=this.controller.meshIndices.filter(index=>this.controller.scene.data[index].name===this.proposal.body_name&&this.controller.scene.data[index].legendgroup===this.job.request.geometry_role);
        const ghosts=originals.map(index=>{const trace=this.controller.scene.data[index];return {...trace,x:Array.from(trace.x),y:Array.from(trace.y),z:Array.from(trace.z),name:'Current pose',legendgroup:'refinement-original',color:'#3174ad',opacity:.22,hoverinfo:'skip',showlegend:false,visible:true};});
        if(!this.ghostIndices.length)this.ghostIndices=await this.controller.plotUpdates.addTraces(ghosts);
        else await this.controller.plotUpdates.restyle(Object.fromEntries(['x','y','z','i','j','k'].map(key=>[key,ghosts.map(trace=>trace[key])])),this.ghostIndices);
        this.controller.currentCloud=source;this.tools.restoredPreview=false;this.active=true;
      }
      const proposed=document.getElementById('refine-preview').value==='proposal';
      const dimensions=this.job.request.kind==='dimensions';
      this.controller.previewDelta=dimensions?null:{body_name:this.proposal.body_name,translation:proposed?this.proposal.translation:[0,0,0]};
      this.controller.previewDimensions=dimensions?{body_name:this.proposal.body_name,scale:proposed?this.proposal.scale:[1,1,1],offset:proposed?this.proposal.offset:[0,0,0],matrix:proposed?(this.proposal.matrix||[]):[]}:null;
      const view=this.tools.inspection.view;view.section_axis='3d';view.clip_enabled=false;view.scan_mode='rgb';view.geometry_mode=this.job.request.geometry_role;
      document.getElementById('section-axis').value='3d';document.getElementById('scan').value='rgb';document.getElementById('geometry').value=view.geometry_mode;
      await this.controller.orientFrame(this.controller.frameMode,this.controller.frameYaw);await this.controller.refreshGeometry();
      const fittedIndices=this.controller.meshIndices.filter(index=>this.controller.scene.data[index].name===this.proposal.body_name&&this.controller.scene.data[index].legendgroup===this.job.request.geometry_role);
      await this.controller.plotUpdates.restyle({color:'#f3a33c'},fittedIndices);
      await this.controller.plotUpdates.restyle({visible:proposed,opacity:.22,color:'#3174ad'},this.ghostIndices);await this.tools.render();
      if(dimensions)await this.showBoxAxes();
      document.getElementById('preview-banner').hidden=false;document.getElementById('preview-banner').textContent=proposed?'Temporary proposal · blue = current pose, orange = proposed pose · not applied':'Temporary review · current pose · not applied';
    }catch(error){this.workspace.notice(error.message);await this.clear();}
    finally{this.tools.applying=applying;this.busy=false;this.tools.historyButtons();this.controls();this.controller.activity?.end(activity);}
  }

  async showBoxAxes() {
    const frame=this.proposal.frame?.root_T_box||CoordinateFrame.identity(),bounds=this.proposal.before_box;
    const center=bounds.lower.map((value,axis)=>(value+bounds.upper[axis])/2);
    const length=Math.max(...bounds.upper.map((value,axis)=>value-bounds.lower[axis]))*.65;
    const display=point=>this.controller.frame.point(CoordinateFrame.transform(this.controller.placement(),CoordinateFrame.transform(frame,point))).map((value,axis)=>value-this.controller.frameOrigin[axis]);
    const origin=display(center),colors=['#e34b4b','#29a36a','#4979e3'];
    const traces=['X','Y','Z'].map((name,axis)=>{
      const end=display(center.map((value,column)=>value+(axis===column?length:0)));
      return {type:'scatter3d',mode:'lines+text',x:[origin[0],end[0]],y:[origin[1],end[1]],z:[origin[2],end[2]],text:['',`${this.proposal.constraints.frame_mode==='box'?'Box':'Root'} ${name}`],textposition:'top center',line:{color:colors[axis],width:6},textfont:{color:colors[axis],size:14},hoverinfo:'skip',showlegend:false,visible:true};
    });
    if(!this.axisIndices.length)this.axisIndices=await this.controller.plotUpdates.addTraces(traces);
    else await this.controller.plotUpdates.restyle(Object.fromEntries(['x','y','z','text','visible'].map(key=>[key,traces.map(trace=>trace[key])])),this.axisIndices);
  }

  async clear() {
    if(!this.active)return;
    this.active=false;this.controller.previewDelta=null;this.controller.previewDimensions=null;
    const applying=this.tools.applying;this.tools.applying=true;this.tools.historyButtons();
    try {
      this.controller.currentCloud=this.previous.cloud;this.tools.restoredPreview=this.previous.restoredPreview;this.tools.inspection.view=this.previous.view;
      const view=this.previous.view;document.getElementById('geometry').value=view.geometry_mode;document.getElementById('scan').value=view.scan_mode;document.getElementById('section-axis').value=view.section_axis;
      await this.controller.plotUpdates.restyle({visible:false},this.ghostIndices);
      if(this.axisIndices.length)await this.controller.plotUpdates.restyle({visible:false},this.axisIndices);
      await this.controller.plotUpdates.restyle({color:this.previous.meshColors},this.controller.meshIndices);
      await this.controller.orientFrame(this.controller.frameMode,this.controller.frameYaw);await this.controller.refreshGeometry();await this.tools.render();
      document.getElementById('preview-banner').hidden=true;
    }finally{this.tools.applying=applying;this.tools.historyButtons();this.controls();}
  }

  async apply() {
    if(this.busy||this.proposal?.outcome!=='proposed')return;
    await this.clear();this.busy=true;this.tools.changingHistory=true;this.tools.historyButtons();this.controls();
    const activity=this.controller.activity?.begin('Applying refinement','Saving the reviewed geometry and restoring the view…');
    try {
      await this.controller.activity?.paint();
      await this.tools.pending;
      const response=await fetch(`/api/jobs/${this.job.identifier}/apply`,{method:'POST'});
      if(!response.ok)throw Error('This proposal cannot be applied. It may be rejected or stale.');
      await this.tools.refreshProject();await this.workspace.jobs.refresh();
      document.getElementById('refine-status').textContent='Selected-body override applied. Undo restores the previous model revision.';
      this.workspace.notice('Snapshot override applied to this body in both geometry roles. Source files stay unchanged.');
    }catch(error){this.workspace.notice(error.message);}
    finally{this.tools.changingHistory=false;this.busy=false;this.tools.historyButtons();this.controls();this.controller.activity?.end(activity);}
  }
}

// %% Reviewed source copies
class SourceApplicationPanel {
  constructor(workspace){this.workspace=workspace;this.tools=workspace.tools;this.controller=workspace.controller;this.review=null;this.reviewKey=null;this.busy=false;this.published=false;}

  initialize(){document.getElementById('review-source').onclick=()=>this.run('review');document.getElementById('publish-source').onclick=()=>this.run('publish');this.controls();}

  revisionKey(){return JSON.stringify([this.controller.state.project_record?.model_source,this.controller.state.overrides||[],this.controller.state.dimension_overrides||[]]);}

  controls(){
    const source=this.controller.state.project_record?.model_source||'';
    const pending=this.busy||this.tools?.applying||this.tools?.loading||this.tools?.changingHistory||this.workspace.refiner?.active||this.workspace.changeSets?.active;
    document.getElementById('review-source').disabled=!!pending||!(source.endsWith('.urdf')||source.includes(':'));
    document.getElementById('publish-source').disabled=!!pending||!this.review||this.published||this.reviewKey!==this.revisionKey();
    if(this.review&&this.reviewKey===this.revisionKey()&&this.published)document.getElementById('source-download').hidden=false;
    if(this.review&&this.reviewKey!==this.revisionKey()){document.getElementById('source-status').textContent='The geometry changed after review. Review the current source changes again.';document.getElementById('source-download').hidden=true;}
  }

  async run(operation){
    if(this.busy)return;
    const previous={applying:this.tools.applying,changingHistory:this.tools.changingHistory};
    this.busy=true;this.tools.applying=true;this.tools.changingHistory=true;this.tools.historyButtons();this.workspace.synchronize();this.controls();
    const status=document.getElementById('source-status');status.textContent=operation==='review'?'Reloading the original source and checking it against the frozen baseline…':'Writing a separate source copy, reloading it, validating and exporting both geometry roles…';
    const activity=this.controller.activity?.begin(operation==='review'?'Reviewing source changes':'Exporting validated source','Reloading and validating the world in a background process…');
    try{
      await this.controller.activity?.paint();
      await this.tools.pending;
      const response=await fetch(`/api/source/${operation}`,{method:'POST',body:operation==='publish'?this.review.identifier:''});
      if(!response.ok){const text=await response.text(),page=new DOMParser().parseFromString(text,'text/html');throw Error(page.body.textContent.trim());}
      const value=InspectionController.decode(await response.json());
      if(operation==='review'){
        this.review=value;this.reviewKey=this.revisionKey();this.published=false;
        const report=document.getElementById('source-review');report.replaceChildren();
        for(const text of [`Source: ${value.revision.model_source}`,`${value.changed_bodies.length} bodies with accepted changes`,...value.changed_bodies,
          ...value.revision.overrides.map(item=>`${item.body_name}: ΔXYZ ${item.translation.map(number=>(number*1000).toFixed(2)).join(', ')} mm`),
          ...value.revision.dimension_overrides.map(item=>`${item.body_name}: accepted dimension stretch`),
          'Only applied overrides are included. Unapplied proposals are excluded.']){const row=document.createElement('div');row.textContent=text;report.append(row);}
        status.textContent='Baseline source geometry verified. Inspect this change list, then write the validated copy.';
        document.getElementById('source-download').hidden=true;
      }else{
        status.textContent=`Validated copy ready: ${value.body_count} bodies, ${value.connection_count} joints and ${value.annotation_count} semantic annotations. Both geometry roles match the accepted revision.`;
        this.published=true;const link=document.getElementById('source-download');link.href=`source-reviews/${this.review.identifier}/world.tar.gz`;link.hidden=false;
      }
    }catch(error){status.textContent=error.message;}
    finally{this.tools.applying=previous.applying;this.tools.changingHistory=previous.changingHistory;this.busy=false;this.tools.historyButtons();this.workspace.synchronize();this.controls();this.controller.activity?.end(activity);}
  }
}
