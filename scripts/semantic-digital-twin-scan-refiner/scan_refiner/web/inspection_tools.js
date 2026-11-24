"use strict";

// %% Reversible inspection and scoped evidence
class InspectionTools {
  constructor(context) {
    Object.assign(this,context);this.applying=false;this.pending=Promise.resolve();
    this.measurementIndex=null;this.cropIndex=null;this.startPoint=null;this.measuring=false;this.cropping=false;this.sectionRequest=0;
  }

  static encode(value) {
    if(value===null||typeof value!=='object')return value;
    if(Array.isArray(value))return value.map(InspectionTools.encode);
    if(value.__json_type__)return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,Array.isArray(item)?{__collection_type__:'builtins.list',__items__:item.map(InspectionTools.encode)}:InspectionTools.encode(item)]));
    return {keys:Object.keys(value),values:Object.values(value).map(InspectionTools.encode)};
  }

  async initialize() {
    if(location.protocol==='file:')return;
    const response=await fetch('/api/state');if(!response.ok)return;
    this.session=this.decodeRecord(await response.json());this.inspection=this.session.history.current.inspection;
    this.useRegistration(this.session.history.current);this.message=document.getElementById('history-status');
    document.getElementById('undo').onclick=()=>this.traverse('undo');document.getElementById('redo').onclick=()=>this.traverse('redo');
    for(const identifier of ['geometry','scan','point-size','mesh-opacity','point-hover'])document.getElementById(identifier).addEventListener('change',()=>this.save(`Change ${identifier}`));
    this.scene.on('plotly_relayout',event=>{if(!this.applying&&Object.keys(event).some(key=>key==='scene.camera'))this.save('Move camera');});
    document.getElementById('apply-clip').onclick=async()=>{
      const bounds=this.inputBounds();if(!bounds)return;
      const view=this.inspection.view;view.clip_lower=bounds.lower;view.clip_upper=bounds.upper;view.clip_enabled=document.getElementById('clip-scan').checked;
      await this.render();await this.save('Clip displayed scan');
    };
    document.getElementById('centre-section').onclick=()=>{
      const normal={xy:2,xz:1,yz:0}[document.getElementById('section-axis').value],bounds=this.inputBounds();
      if(normal!==undefined&&bounds)document.getElementById('section-position').value=(bounds.lower[normal]+bounds.upper[normal])/2;
    };
    document.getElementById('apply-section').onclick=async()=>{
      const view=this.inspection.view;view.section_axis=document.getElementById('section-axis').value;view.section_position=Number(document.getElementById('section-position').value);view.section_thickness=Number(document.getElementById('section-thickness').value);
      if(!Number.isFinite(view.section_position)||!(view.section_thickness>0))return;
      this.cropping=false;await this.render();await this.save('Change section');
    };
    document.getElementById('measure').onclick=()=>{
      this.measuring=!this.measuring;this.startPoint=null;this.cropping=false;
      document.getElementById('measure').textContent=this.measuring?'Cancel measurement':'Measure two scan points';
      this.plotUpdates.restyle({hoverinfo:this.measuring?'all':'skip'},this.scanIndices());
      this.message.textContent=this.measuring?'Click two scan points. Endpoints retain their original scan coordinates.':'';
    };
    this.plotUpdates.registerClick(event=>{
      const point=event.points[0];
      if(this.measuring&&this.scanIndices().includes(point.curveNumber)){
        const origin=this.getFrameOrigin?this.getFrameOrigin():this.state.display_origin;
        const position=[point.x,point.y,point.z].map((v,a)=>v+origin[a]);
        this.pick(this.getFrame?this.getFrame().toScan(position):position);
      }
    });
    document.getElementById('clear-measurements').onclick=async()=>{this.inspection.measurements=[];await this.render();this.annotations();await this.save('Clear measurements');};
    document.getElementById('save-view').onclick=async()=>{
      this.capture();const bookmark=structuredClone(this.session.bookmark_template);
      bookmark.label=document.getElementById('view-label').value.trim()||`View ${this.inspection.bookmarks.length+1}`;bookmark.note=document.getElementById('view-note').value;bookmark.view=structuredClone(this.inspection.view);
      this.inspection.bookmarks.push(bookmark);await this.save('Save annotated view');this.annotations();
    };
    document.getElementById('restore-view').onclick=async()=>{const bookmark=this.inspection.bookmarks[Number(document.getElementById('saved-views').value)];if(!bookmark)return;this.inspection.view=structuredClone(bookmark.view);await this.apply();await this.save('Restore annotated view');};
    document.getElementById('remove-view').onclick=async()=>{const index=Number(document.getElementById('saved-views').value);if(!this.inspection.bookmarks[index])return;this.inspection.bookmarks.splice(index,1);await this.save('Remove annotated view');this.annotations();};
    this.historyButtons();
    if(this.session.history.past.length||this.session.history.future.length)await this.apply();
    else {
      if(this.configureFrame){this.inspection.view.coordinate_frame=this.state.registration.candidates.length?'world':'scan';await this.configureFrame(this.inspection.view);await this.render();this.restoreBounds();}
      this.capture();
    }
    this.historyButtons();this.annotations();
  }

  capture() {
    const view=this.inspection.view;
    view.geometry_mode=document.getElementById('geometry').value;view.scan_mode=document.getElementById('scan').value;
    const selected=document.getElementById('candidate').value;view.selected_candidate=selected===''?null:Number(selected);
    view.point_size=Number(document.getElementById('point-size').value);view.mesh_opacity=Number(document.getElementById('mesh-opacity').value);
    const camera=this.scene._fullLayout.scene.camera;
    for(const vector of ['eye','up','center'])view.camera[vector]=['x','y','z'].map(axis=>camera[vector][axis]);
    view.camera.projection=camera.projection.type;
  }

  async change(label, operation) {
    if(this.applying||this.loading||this.changingHistory)return false;
    this.applying=true;this.historyButtons();this.onViewChanged?.();
    const activity=this.activity?.begin(label,'Updating the view…');
    try {await this.activity?.paint();await operation();}
    finally {this.applying=false;this.historyButtons();this.onViewChanged?.();this.activity?.end(activity);}
    await this.save(label);return true;
  }

  save(label) {
    if(this.applying||this.changingHistory||!this.session||this.isPreview?.())return Promise.resolve();this.capture();
    const action={__json_type__:'scan_refiner.web_service.InspectionAction',label,inspection:structuredClone(this.inspection)};
    this.pending=this.pending.then(async()=>{
      const response=await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(InspectionTools.encode(action))});
      if(!response.ok)throw Error(`Cannot save inspection (${response.status})`);
      this.session=this.decodeRecord(await response.json());this.historyButtons();this.onViewChanged?.();
    }).catch(error=>{this.message.textContent=error.message;});
    return this.pending;
  }

  historyButtons() {
    const history=this.session.history;
    document.getElementById('undo').disabled=this.applying||this.changingHistory||this.loading||!history.past.length;
    document.getElementById('redo').disabled=this.applying||this.changingHistory||this.loading||!history.future.length;
    this.message.textContent=`${history.current.label} · ${history.past.length} undoable actions`;
    const list=document.getElementById('action-history');list.replaceChildren();
    for(const item of [...history.past,history.current].slice(-30)){const entry=document.createElement('li');entry.textContent=item.label;list.append(entry);}
  }

  useRegistration(current) {
    this.viewRevision=this.session.revision;
    if(JSON.stringify(current.registration)!==JSON.stringify(this.state.registration)){
      this.state.registration.candidates.splice(0,this.state.registration.candidates.length,...current.registration.candidates);
      const selector=document.getElementById('candidate');selector.replaceChildren();
      this.state.registration.candidates.forEach((candidate,index)=>selector.add(new Option(`${index}: ${(candidate.support_fraction*100).toFixed(1)}% sample support`,index)));
      if(!this.state.registration.candidates.length)selector.add(new Option('Original coordinates',''));
      selector.disabled=!this.state.registration.candidates.length;this.state.has_heatmap=false;
    }
    this.state.overrides=current.overrides||[];
    this.state.dimension_overrides=current.dimension_overrides||[];
    this.state.registration.settings=current.registration.settings;this.state.accepted_candidate=current.record.accepted_candidate;this.state.project_record=current.record;
  }

  async refreshProject(applyView=true) {
    const activity=this.activity?.begin('Refreshing project','Loading the current model and saved view…');
    try {
      await this.activity?.paint();
    await this.onBeforeRestore?.();await this.pending;const response=await fetch('/api/state');if(!response.ok)throw Error('Cannot restore project decisions.');
    this.session=this.decodeRecord(await response.json());this.useRegistration(this.session.history.current);this.inspection=this.session.history.current.inspection;
    if(applyView)await this.apply();this.historyButtons();this.onViewChanged?.();

    }finally{this.activity?.end(activity);}
  }

  async traverse(direction) {
    if(this.applying||this.loading||this.changingHistory)return;
    this.changingHistory=true;this.historyButtons();
    const activity=this.activity?.begin(direction==='undo'?'Undoing last action':'Redoing action','Restoring the saved project state…');
    await this.activity?.paint();
    try {
      await this.onBeforeRestore?.();await this.pending;const response=await fetch(`/api/${direction}`,{method:'POST'});
      if(!response.ok){this.message.textContent=`${direction} failed`;return;}
      this.session=this.decodeRecord(await response.json());this.useRegistration(this.session.history.current);this.inspection=this.session.history.current.inspection;
      await this.apply();this.annotations();this.onViewChanged?.();
    }finally{this.changingHistory=false;this.historyButtons();this.onViewChanged?.();this.activity?.end(activity);}
  }

  async apply() {
    this.applying=true;
    const activity=this.activity?.begin('Restoring saved view','Restoring geometry and display settings…');
    await this.activity?.paint();
    try {
      const view=this.inspection.view;
      for(const [id,value]of [['geometry',view.geometry_mode],['scan',view.scan_mode],['point-size',view.point_size],['mesh-opacity',view.mesh_opacity],['point-budget',view.point_budget],['section-axis',view.section_axis],['section-position',view.section_position],['section-thickness',view.section_thickness]])document.getElementById(id).value=value;
      document.getElementById('clip-scan').checked=view.clip_enabled;
      const candidate=document.getElementById('candidate');candidate.value=view.selected_candidate===null?(this.state.registration.candidates.length?(this.state.selected_candidate??0):''):view.selected_candidate;
      candidate.dispatchEvent(new Event('change'));
      if(this.configureFrame)await this.configureFrame(view);
      await this.refreshGeometry();
      await this.plotUpdates.restyle({opacity:view.mesh_opacity},this.scene.data.map((trace,index)=>trace.type==='mesh3d'?index:-1).filter(index=>index>=0));
      await this.plotUpdates.restyle({'marker.size':view.point_size},this.scanIndices());
      this.activity?.update(activity,'Restoring scan points and comparison results…');
      if(view.region_lower.length===3){
        this.restoredPreview=false;
        const restored=view.comparison_job&&this.restoreComparison?await this.restoreComparison(view.comparison_job):false;
        if(!restored)await this.loadRegion(view.region_lower,view.region_upper,view.region_frame_T_scan?.length?view.region_frame_T_scan:CoordinateFrame.identity());
        document.getElementById('scan').value=view.scan_mode==='heatmap'&&!restored?'rgb':view.scan_mode;
      }
      else {this.restoredPreview=true;this.setOverview(true);this.setRGB(this.previewCloud.colors);}
      this.activity?.update(activity,'Drawing the restored view…');
      await this.activity?.paint();
      const camera={projection:{type:view.camera.projection}};
      for(const vector of ['eye','up','center'])camera[vector]=Object.fromEntries(['x','y','z'].map((axis,i)=>[axis,view.camera[vector][i]]));
      await this.plotUpdates.relayout({'scene.camera':camera,'scene.dragmode':'orbit'});
      this.restoreBounds();await this.render();
      if(view.focus_enabled&&this.focusSelection){const bounds=this.inputBounds();if(bounds)await this.focusSelection(bounds.lower,bounds.upper);}
      else if(this.resetFocus)await this.resetFocus();
      this.annotations();
    }finally{this.applying=false;this.historyButtons();this.onViewChanged?.();this.activity?.end(activity);}
  }

  regionLoading(loading){this.loading=loading;this.historyButtons();this.onViewChanged?.();}

  async regionLoaded(lower,upper,count,matrix=null) {
    this.restoredPreview=false;
    if(!this.applying){
      const view=this.inspection.view;view.region_lower=lower;view.region_upper=upper;view.point_budget=count;
      view.region_frame_T_scan=matrix||this.getFrame?.().matrix||[];view.comparison_job="";
    }
    await this.render();
    if(!this.applying)await this.save('Load oriented scan region');
  }

  inputBounds() {
    const lower=document.getElementById('region-lower').value.split(',').map(Number),upper=document.getElementById('region-upper').value.split(',').map(Number);
    return lower.length===3&&upper.length===3&&[...lower,...upper].every(Number.isFinite)&&lower.every((v,a)=>v<=upper[a])?{lower,upper}:null;
  }

  sourceBounds() {
    const view=this.inspection.view,frame=this.getFrame?.();
    if(view.region_lower.length===3&&frame){const queryFrame=new CoordinateFrame(view.region_frame_T_scan.length?view.region_frame_T_scan:CoordinateFrame.identity());return frame.bounds(CoordinateFrame.corners(view.region_lower,view.region_upper).map(point=>queryFrame.toScan(point)));}
    const source=this.restoredPreview?this.previewCloud:this.getCloud();
    const lower=[Infinity,Infinity,Infinity],upper=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<source.coordinates[0].length;i++){
      const raw=source.coordinates.map((axis,a)=>axis[i]+this.state.display_origin[a]),point=frame?frame.point(raw):raw;
      point.forEach((value,a)=>{lower[a]=Math.min(lower[a],value);upper[a]=Math.max(upper[a],value);});
    }
    return {lower,upper};
  }

  restoreBounds() {
    const view=this.inspection.view;
    const bounds=view.selection_lower?.length===3?{lower:view.selection_lower,upper:view.selection_upper}:this.sourceBounds();
    document.getElementById('region-lower').value=bounds.lower.map(v=>v.toFixed(6)).join(',');document.getElementById('region-upper').value=bounds.upper.map(v=>v.toFixed(6)).join(',');
  }

  async regionSelected(lower,upper,label='Select crop') {
    this.inspection.view.selection_lower=lower;this.inspection.view.selection_upper=upper;
    document.getElementById('region-lower').value=lower.map(v=>v.toFixed(6)).join(',');document.getElementById('region-upper').value=upper.map(v=>v.toFixed(6)).join(',');
    await this.drawSelection();await this.save(label);this.onViewChanged?.();
  }

  async beginCrop() {
    return this.change('Open crop plan',async()=>{
      const view=this.inspection.view,bounds=this.sourceBounds();this.cropping=true;
      view.section_axis='xy';view.section_position=(bounds.lower[2]+bounds.upper[2])/2;view.section_thickness=Math.max(.01,bounds.upper[2]-bounds.lower[2]);
      document.getElementById('section-axis').value='xy';document.getElementById('section-position').value=view.section_position;document.getElementById('section-thickness').value=view.section_thickness;
      await this.render();
      document.getElementById('region-status').textContent='Drag a rectangle in the plan view, then load the selected region.';
    });
  }

  static select(source,view,origin,frame=null) {
    const normal={xy:2,xz:1,yz:0}[view.section_axis],selected=[];
    for(let i=0;i<source.coordinates[0].length;i++){
      const raw=source.coordinates.map((axis,a)=>axis[i]+origin[a]);const point=frame?frame.point(raw):raw;
      if(view.clip_enabled&&view.clip_lower.length===3&&point.some((v,a)=>v<view.clip_lower[a]||v>view.clip_upper[a]))continue;
      if(normal!==undefined&&Math.abs(point[normal]-view.section_position)>view.section_thickness/2)continue;
      selected.push(i);
    }
    return selected;
  }

  async render() {
    const activity=this.activity?.begin('Updating scan display','Preparing scan points, sections and measurements…');
    try {
      await this.activity?.paint();
    const view=this.inspection.view,source=this.restoredPreview?this.previewCloud:this.getCloud(),frame=this.getFrame?.();
    const origin=this.getFrameOrigin?this.getFrameOrigin():this.state.display_origin;
    const selected=InspectionTools.select(source,view,this.state.display_origin,frame);
    const scanPositions=selected.map(i=>source.coordinates.map((axis,a)=>axis[i]+this.state.display_origin[a]));
    const positions=frame?scanPositions.map(point=>frame.point(point)):scanPositions;
    const axes=[0,1,2].map(axis=>Float32Array.from(positions,point=>point[axis]-origin[axis]));
    const colors=selected.map(i=>source.colors[i]);await this.cloud.replace(axes,colors,selected.map(i=>source.identities[i]));
    this.setRGB(colors);this.setOverview(source===this.previewCloud&&selected.length===source.coordinates[0].length);
    const distances=source.distances?selected.map(i=>source.distances[i]):null;
    this.setRegionalHeatmap?.(distances,source.comparison||null);
    const proposal=source.proposalComparison;
    this.setProposalHeatmap?.(proposal?{job:proposal.job,original:selected.map(i=>proposal.original[i]),proposed:selected.map(i=>proposal.proposed[i])}:null);
    await this.refreshScan();
    this.updateCounts?.(selected.length,source.coordinates[0].length);
    await this.drawSection(axes,colors,scanPositions,distances);await this.drawMeasurements();await this.drawSelection();this.onViewChanged?.();

    }finally{this.activity?.end(activity);}
  }

  async drawSection(axes,colors,positions,distances=null) {
    const view=this.inspection.view,panel=document.getElementById('section-panel');panel.hidden=view.section_axis==='3d';
    document.getElementById('section-shell')?.toggleAttribute('hidden',panel.hidden);document.getElementById('viewport-stack')?.classList.toggle('has-section',!panel.hidden);
    if(panel.hidden)return;
    const request=++this.sectionRequest,selected=document.getElementById('candidate').value;
    const query=new URLSearchParams({plane:view.section_axis,position:view.section_position,candidate:selected===''?'none':selected,role:view.geometry_mode});
    if(this.getFrame)query.set('frame_T_scan',this.getFrame().matrix.flat().join(','));
    const response=await fetch(`/api/section?${query}`);if(!response.ok){this.message.textContent='Section geometry unavailable';return;}
    const outlines=this.decodeRecord(await response.json());if(request!==this.sectionRequest)return;
    const pair={xy:[0,1],xz:[0,2],yz:[1,2]}[view.section_axis],origin=this.getFrameOrigin?this.getFrameOrigin():this.state.display_origin;
    const heatmap=distances&&document.getElementById('scan').value==='heatmap';
    const traces=[{type:'scattergl',mode:'markers',name:'Scan slab',x:Array.from(axes[pair[0]],v=>v+origin[pair[0]]),y:Array.from(axes[pair[1]],v=>v+origin[pair[1]]),customdata:positions,marker:{size:view.point_size,color:heatmap?distances:colors,...(heatmap?{colorscale:this.scene.data[0].marker.colorscale,cmin:0,cmax:this.state.distance_limit||.1}:{})},hoverinfo:'x+y'}];
    for(const outline of outlines){const x=[],y=[];for(const segment of outline.segments){x.push(segment[0][pair[0]],segment[1][pair[0]],null);y.push(segment[0][pair[1]],segment[1][pair[1]],null);}traces.push({type:'scattergl',mode:'lines',name:`${outline.body_name} (${outline.role})`,x,y,line:{color:outline.role==='visual'?'#d88917':'#00aabb',width:2}});}
    for(const measurement of this.inspection.measurements){const endpoints=[measurement.start,measurement.end].map(point=>this.getFrame?this.getFrame().point(point):point);traces.push({type:'scatter',mode:'lines+markers',x:endpoints.map(point=>point[pair[0]]),y:endpoints.map(point=>point[pair[1]]),line:{color:'#bb00aa'},name:measurement.label});}
    const bounds=this.cropping?this.sourceBounds():(this.inputBounds()||this.sourceBounds());
    const label=view.coordinate_frame==='world'?'World':'Scan';
    await Plotly.react(panel,traces,{title:{text:`${view.section_axis.toUpperCase()} · ${view.section_position.toFixed(3)} m · slab ${view.section_thickness.toFixed(3)} m`,font:{size:11,color:'#718894'}},
      xaxis:{title:{text:`${label} ${view.section_axis[0].toUpperCase()} (m)`},range:[bounds.lower[pair[0]],bounds.upper[pair[0]]],gridcolor:'#e3ebee',zerolinecolor:'#d6e1e5'},
      yaxis:{title:{text:`${label} ${view.section_axis[1].toUpperCase()} (m)`},range:[bounds.lower[pair[1]],bounds.upper[pair[1]]],scaleanchor:'x',scaleratio:1,gridcolor:'#e3ebee',zerolinecolor:'#d6e1e5'},
      margin:{t:36,l:55,b:43,r:18},font:{family:'Arial, sans-serif',size:10,color:'#6f8792'},uirevision:`section-${view.coordinate_frame}-${view.frame_yaw}-${view.section_axis}-${bounds.lower}-${bounds.upper}`,showlegend:false,dragmode:this.cropping?'select':'pan'},{responsive:true,displaylogo:false});
    panel.removeAllListeners('plotly_click');panel.on('plotly_click',event=>{if(this.measuring&&event.points[0].curveNumber===0)this.pick(event.points[0].customdata);});
    panel.removeAllListeners('plotly_selected');panel.on('plotly_selected',async event=>{
      if(!this.cropping||!event?.range)return;
      const selection=this.inputBounds()||this.sourceBounds();
      pair.forEach((axis,index)=>{const range=index===0?event.range.x:event.range.y;selection.lower[axis]=Math.min(...range);selection.upper[axis]=Math.max(...range);});
      this.cropping=false;this.inspection.view.selected_body='';this.inspection.view.selected_assembly='';
      await this.regionSelected(selection.lower,selection.upper,'Draw oriented crop');
      await Plotly.restyle(panel,{selectedpoints:[null]},[0]);await Plotly.relayout(panel,{dragmode:'pan'});
      document.getElementById('region-status').textContent='Oriented crop ready. Load region to retrieve its original scan records.';
    });
  }

  async drawSelection() {
    const view=this.inspection.view;
    if(!view.selection_lower?.length&&this.cropIndex===null)return;
    if(this.cropIndex===null)[this.cropIndex]=await this.plotUpdates.addTraces([{type:'scatter3d',mode:'lines',x:[],y:[],z:[],name:'Selected crop',line:{color:'#058c76',width:4},showlegend:false,hoverinfo:'skip'}]);
    const origin=this.getFrameOrigin?this.getFrameOrigin():this.state.display_origin;
    const corners=view.selection_lower?.length?CoordinateFrame.corners(view.selection_lower,view.selection_upper):[];
    const edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
    const axes=[0,1,2].map(axis=>corners.length?edges.flatMap(([a,b])=>[corners[a][axis]-origin[axis],corners[b][axis]-origin[axis],null]):[]);
    await this.plotUpdates.restyle({x:[axes[0]],y:[axes[1]],z:[axes[2]]},[this.cropIndex]);
    const panel=document.getElementById('section-panel');
    if(panel.data&&!panel.hidden){const pair={xy:[0,1],xz:[0,2],yz:[1,2]}[view.section_axis];const shapes=corners.length?[{type:'rect',x0:view.selection_lower[pair[0]],x1:view.selection_upper[pair[0]],y0:view.selection_lower[pair[1]],y1:view.selection_upper[pair[1]],line:{color:'#058c76',width:2,dash:'dot'},fillcolor:'rgba(0,0,0,0)'}]:[];await Plotly.relayout(panel,{shapes});}
  }

  async pick(point) {
    if(!this.startPoint){this.startPoint=point;this.message.textContent='Select the second scan point.';return;}
    const measurement=structuredClone(this.session.measurement_template);measurement.start=this.startPoint;measurement.end=point;measurement.label=document.getElementById('measurement-label').value||`Distance ${this.inspection.measurements.length+1}`;
    this.inspection.measurements.push(measurement);this.measuring=false;this.startPoint=null;document.getElementById('measure').textContent='Measure two scan points';
    await this.render();this.annotations();await this.save('Measure distance');
  }

  async drawMeasurements() {
    await this.plotUpdates.pending;
    if(this.measurementIndex===null)[this.measurementIndex]=await this.plotUpdates.addTraces([{type:'scatter3d',mode:'lines+markers',x:[],y:[],z:[],name:'Measurements',line:{color:'#bb00aa',width:5},marker:{size:4,color:'#bb00aa'},showlegend:false}]);
    const origin=this.getFrameOrigin?this.getFrameOrigin():this.state.display_origin;
    const endpoints=this.inspection.measurements.map(measurement=>[measurement.start,measurement.end].map(point=>this.getFrame?this.getFrame().point(point):point));
    const axes=[0,1,2].map(axis=>endpoints.flatMap(pair=>[pair[0][axis]-origin[axis],pair[1][axis]-origin[axis],null]));
    await this.plotUpdates.restyle({x:[axes[0]],y:[axes[1]],z:[axes[2]]},[this.measurementIndex]);
  }

  annotations() {
    const list=document.getElementById('measurements');list.replaceChildren();
    for(const measurement of this.inspection.measurements){const item=document.createElement('li');item.textContent=`${measurement.label}: ${(1000*Math.hypot(...measurement.end.map((value,axis)=>value-measurement.start[axis]))).toFixed(1)} mm`;list.append(item);}
    const views=document.getElementById('saved-views');views.replaceChildren();this.inspection.bookmarks.forEach((bookmark,index)=>views.add(new Option(`${bookmark.label}${bookmark.note?' — '+bookmark.note:''}`,index)));
  }
}
