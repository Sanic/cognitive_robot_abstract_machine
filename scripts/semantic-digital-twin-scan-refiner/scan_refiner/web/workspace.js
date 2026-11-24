"use strict";

// %% Staged local workspace
class RefinerWorkspace {
  constructor(controller){this.controller=controller;this.tools=controller.tools;this.stage='inspect';this.objects=new Map();this.toastTimer=null;}

  async initialize() {
    for(const index of this.controller.meshIndices){const trace=this.controller.scene.data[index];if(!this.objects.has(trace.name))this.objects.set(trace.name,{name:trace.name,indices:[],roles:[]});const item=this.objects.get(trace.name);item.indices.push(index);item.roles.push(trace.legendgroup);}
    document.getElementById('object-count').textContent=String(this.objects.size);
    this.assemblies=new AssemblyPanel(this);
    this.worldTree=new WorldTree(this,this.controller.state.kinematic_nodes || []);
    document.getElementById('object-search').oninput=()=>this.drawObjects();this.drawObjects();
    document.querySelectorAll('[data-stage]').forEach(button=>button.onclick=()=>this.navigate(button.dataset.stage));
    document.querySelectorAll('[data-inspector]').forEach(button=>button.onclick=()=>this.inspector(button.dataset.inspector));
    for(const kind of ['jobs','history']){
      document.getElementById(`open-${kind}`).onclick=()=>{const drawer=document.getElementById(`${kind}-drawer`);drawer.hidden=!drawer.hidden;};
      document.getElementById(`close-${kind}`).onclick=()=>{document.getElementById(`${kind}-drawer`).hidden=true;};
    }
    document.getElementById('world-overview').onclick=()=>this.controller.loadOverview();
    document.getElementById('focus-selection').onclick=async()=>{if(!this.tools?.inspection)return;const bounds=this.controller.readBounds();if(!bounds)return;await this.tools.change('Focus selection',async()=>{this.tools.inspection.view.focus_enabled=true;await this.controller.focusSelection(bounds.lower,bounds.upper);});};
    document.getElementById('reset-focus').onclick=async()=>{if(this.tools?.inspection){await this.tools.change('Show whole scene',async()=>{this.tools.inspection.view.focus_enabled=false;await this.controller.resetFocus();});}};
    document.getElementById('apply-frame').onclick=()=>this.orient();
    document.getElementById('draw-box').onclick=async()=>{if(!this.tools?.inspection)return;this.navigate('inspect');await this.tools.beginCrop();this.notice('Drag a rectangle in the companion plan. Then choose Load region.');};
    document.getElementById('use-object-region').onclick=async()=>{const group=this.assemblies.current;if(group){await this.assemblies.select(group);return;}const name=this.tools?.inspection?.view.selected_body;if(name&&this.objects.has(name))await this.controller.selectBody(this.objects.get(name).indices[0]);};
    for(const id of ['region-lower','region-upper'])document.getElementById(id).addEventListener('change',async()=>{const bounds=this.controller.readBounds();if(bounds&&this.tools?.inspection)await this.tools.regionSelected(bounds.lower,bounds.upper,'Edit crop bounds');});
    document.getElementById('hide-section').onclick=async()=>{if(!this.tools?.inspection)return;this.tools.inspection.view.section_axis='3d';this.tools.cropping=false;document.getElementById('section-axis').value='3d';await this.tools.render();await this.tools.save('Close companion section');};
    document.getElementById('accept-placement').onclick=()=>this.acceptPlacement();
    this.investigation=new InvestigationHandoff(this);
    this.changeSets=new ChangeSetPanel(this);
    this.prepareSummary();this.observeSize();this.synchronize();
    this.refiner=new RefinementPanel(this);this.refiner.initialize();
    this.sourceApplication=new SourceApplicationPanel(this);this.sourceApplication.initialize();
    this.jobs=new WorkspaceJobs(this);await this.jobs.initialize();
    if(location.protocol==='file:'){
      document.querySelectorAll('.task-inspector button,.task-inspector input,.task-inspector select').forEach(control=>{if(control.id!=='candidate')control.disabled=true;});
      document.getElementById('coordinate-frame').value='scan';
      this.notice('Use the local serve command for oriented crops, persistent inspection and analysis jobs.');
    }
    this.controller.scene.dataset.workspaceReady='true';
  }

  pretty(name){return name.split('/').at(-1).replace(/_/g,' ').replace(/\b\w/g,letter=>letter.toUpperCase());}

  drawObjects() {
    const query=document.getElementById('object-search').value.trim().toLowerCase();
    const list=document.getElementById('body-list');
    list.replaceChildren();
    this.assemblies?.draw(list,query);
    if(!this.worldTree?.draw(list,query)){
      const heading=document.createElement('div');
      heading.className='assembly-heading';
      heading.textContent='All bodies';
      list.append(heading);
      for(const item of this.objects.values()){
        if(!item.name.toLowerCase().includes(query))continue;
        list.append(this.bodyButton(item));
      }
    }
    this.assemblies?.updateSelection();
  }

  bodyButton(item,{focus=false}={}) {
    const button=document.createElement('button');button.className='body-item';button.dataset.body=item.name;button.title=item.name;
    const symbol=document.createElement('span');symbol.className='body-symbol';symbol.textContent='▧';
    const name=document.createElement('span');name.className='body-title';name.textContent=this.pretty(item.name);
    const roles=document.createElement('span');roles.className='body-role';roles.textContent=item.roles.includes('visual')?'V':'C';roles.title=item.roles.join(' + ');
    button.append(symbol,name,roles);
    button.onclick=async()=>{
      const index=item.indices.find(index=>this.controller.scene.data[index].legendgroup==='visual')??item.indices[0];
      if(focus&&this.tools?.inspection){
        await this.changeSets?.close();
        await this.refiner?.clear();
        await this.tools.change(`Focus ${this.pretty(item.name)} body`,async()=>{
          await this.controller.selectBody(index);
          const view=this.tools.inspection.view;
          view.focus_enabled=true;
          await this.controller.focusSelection(view.selection_lower,view.selection_upper);
        });
      }else await this.controller.selectBody(index);
      this.synchronize();
    };
    const view=this.tools?.inspection?.view;
    button.classList.toggle('active',view?.selected_body===item.name);
    button.disabled=!!(this.tools?.loading||this.tools?.applying||this.tools?.changingHistory);
    return button;
  }

  navigate(stage) {
    if(stage!=='refine'&&this.changeSets?.active){this.changeSets.close().then(()=>this.navigate(stage));return;}
    if(stage!=='refine'&&this.refiner?.active){this.refiner.clear().then(()=>this.navigate(stage));return;}
    this.stage=stage;
    document.querySelectorAll('.stage-nav [data-stage]').forEach(button=>{const active=button.dataset.stage===stage;button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');});
    document.getElementById('prepare-stage').hidden=stage!=='prepare';document.getElementById('scene-stage').hidden=stage==='prepare';
    for(const name of ['prepare','align','inspect','refine'])document.getElementById(`${name}-inspector`).hidden=name!==stage;
    document.getElementById('workspace-step').textContent=stage==='align'?'REGISTRATION REVIEW':stage==='refine'?'CONSTRAINED REFINEMENT':'INSPECTION WORKSPACE';
    if(stage==='refine')this.changeSets?.refresh();
    this.resize();
  }

  inspector(name) {
    document.querySelector('.task-inspector').scrollTop=0;
    document.querySelectorAll('[data-inspector]').forEach(button=>{const active=button.dataset.inspector===name;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));});
    document.querySelectorAll('[data-inspector-panel]').forEach(panel=>{panel.hidden=panel.dataset.inspectorPanel!==name;});
  }

  async orient() {
    if(!this.tools?.inspection)return;
    const mode=document.getElementById('coordinate-frame').value;
    const yaw=Number(document.getElementById('frame-yaw').value);if(!Number.isFinite(yaw)){this.notice('Enter a finite yaw angle.');return;}
    const view=this.tools.inspection.view;
    await this.tools.change('Orient view and crop axes',async()=>{
      view.coordinate_frame=mode;view.frame_yaw=yaw;view.clip_enabled=false;view.selection_lower=[];view.selection_upper=[];document.getElementById('clip-scan').checked=false;
      await this.controller.orientFrame(view.coordinate_frame,yaw);this.tools.restoreBounds();
      if(this.assemblies?.current)await this.assemblies.selectRegion(this.assemblies.current);
      else if(view.selected_body&&this.objects.has(view.selected_body))await this.controller.selectBody(this.objects.get(view.selected_body).indices[0]);
      await this.tools.render();
      if(view.focus_enabled){const bounds=this.controller.readBounds();if(bounds)await this.controller.focusSelection(bounds.lower,bounds.upper);}
      else await this.controller.resetFocus();
    });
    this.synchronize();
  }

  async acceptPlacement() {
    const selected=document.getElementById('candidate').value;if(selected===''||!this.tools?.inspection)return;
    try {
      await this.tools.pending;
      const action={__json_type__:'scan_refiner.web_service.AcceptanceAction',candidate_index:Number(selected)};
      const response=await fetch('/api/accept',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action)});
      if(!response.ok)throw Error(`Cannot accept this placement (${response.status}).`);
      await this.tools.refreshProject(false);this.controller.refreshScan();this.synchronize();this.notice('Placement accepted. Undo can restore the preceding decision.');
    }catch(error){this.notice(error.message);}
  }

  synchronize() {
    this.refiner?.controls();this.sourceApplication?.controls();this.changeSets?.controls();
    this.assemblies?.update();
    this.investigation?.update();
    const view=this.tools?.inspection?.view;
    const selected=document.getElementById('candidate').value;
    const accepted=selected!==''&&Number(selected)===this.controller.state.accepted_candidate;
    const badge=document.getElementById('workspace-placement');badge.textContent=selected===''?'Unregistered':accepted?'Placement accepted':'Provisional placement';badge.classList.toggle('accepted',accepted);
    document.getElementById('accept-placement').disabled=selected===''||accepted||!!this.tools?.loading;
    if(view){
      const group=this.assemblies?.current;
      document.getElementById('selection-name').textContent=group?.label||(view.selected_body?this.pretty(view.selected_body):view.selection_lower.length?'Selected scan region':'World overview');
      document.getElementById('world-overview').classList.toggle('active',!group&&!view.selected_body&&!view.selection_lower.length);
      document.querySelectorAll('[data-body]').forEach(button=>button.classList.toggle('active',button.dataset.body===view.selected_body));
      if(document.activeElement!==document.getElementById('coordinate-frame'))document.getElementById('coordinate-frame').value=view.coordinate_frame;
      if(document.activeElement!==document.getElementById('frame-yaw'))document.getElementById('frame-yaw').value=view.frame_yaw;
      document.getElementById('use-object-region').disabled=!group&&!view.selected_body;
      const busy=!!(this.tools.loading||this.tools.applying||this.tools.changingHistory);
      for(const id of ['apply-frame','draw-box','apply-clip','apply-section','restore-view','save-view','measure','focus-selection','reset-focus'])document.getElementById(id).disabled=busy;
      for(const id of ['load-region','load-overview'])document.getElementById(id).disabled=busy||!this.controller.indexInformation;
      document.querySelectorAll('[data-body]').forEach(button=>{button.disabled=busy;});
    }
    const bounds=this.controller.readBounds();if(bounds){const dimensions=bounds.lower.map((v,a)=>bounds.upper[a]-v);document.getElementById('region-dimensions').textContent=`${dimensions.map(value=>value.toFixed(3)).join(' × ')} m`;}
    this.jobs?.controls();this.jobs?.updateSummary();this.resize();
  }

  prepareSummary() {
    const record=this.controller.state.project_record,index=this.controller.indexInformation?.index;
    const values=[['WORLD BASELINE',`${this.objects.size} bodies`,`${this.controller.meshIndices.length} visual/collision surfaces`],['SCAN EVIDENCE',index?`${(index.point_count/1000000).toFixed(1)} million`:`${this.controller.previewCloud.coordinates[0].length.toLocaleString()} displayed points`,index?'Original records in the completed index':'Preview available; no dense index attached'],['SOURCE',record?.model_source?.endsWith('.urdf')?'URDF world':'SemDT world',record?.model_source||'Frozen world geometry'],['PROJECT DECISIONS','Undo & redo','Registration, crops, sections, measurements and named views']];
    const cards=document.getElementById('project-summary');cards.replaceChildren();
    for(const [label,stat,detail]of values){const card=document.createElement('article');card.className='project-card';const title=document.createElement('span');title.className='eyebrow';title.textContent=label;const number=document.createElement('span');number.className='stat';number.textContent=stat;const text=document.createElement('p');text.textContent=detail;card.append(title,number,text);cards.append(card);}
    document.getElementById('project-provenance').textContent=record?`World source: ${record.model_source}\nWorld SHA256: ${record.world_sha256}\nScan source: ${record.scan_source}\nScan SHA256: ${record.scan_sha256}\nSource-to-metre scale: ${record.scan_scale}\nIndex: ${index?index.source_identity:'Not attached'}`:'Frozen inputs available in the project directory.';
  }

  observeSize() {
    if(typeof ResizeObserver==='undefined')return;
    this.observer=new ResizeObserver(()=>this.resize());
    this.observer.observe(document.getElementById('scene-container'));this.observer.observe(document.getElementById('section-shell'));
    window.addEventListener('resize',()=>this.resize());
  }

  resize() {
    clearTimeout(this.resizeTimer);
    this.resizeTimer=setTimeout(async()=>{
      const rectangle=this.controller.scene.getBoundingClientRect();
      if(rectangle.width>0&&rectangle.height>0){await this.controller.plotUpdates.resize();}
      const section=document.getElementById('section-panel');
      if(!section.hidden&&section.data&&section.getBoundingClientRect().height>0)await Plotly.Plots.resize(section);
    },100);
  }

  notice(message) {
    const toast=document.getElementById('workspace-notice');toast.textContent=message;toast.hidden=false;clearTimeout(this.toastTimer);this.toastTimer=setTimeout(()=>{toast.hidden=true;},6500);
  }
}
