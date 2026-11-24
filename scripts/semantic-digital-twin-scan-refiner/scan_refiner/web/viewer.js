"use strict";

// %% Scene, immutable source coordinates and frame conversion
class InspectionController {
  constructor(activity = null) {
    this.state=InspectionController.decode(JSON.parse(document.getElementById('viewer-state').textContent));
    this.scene=document.getElementById('scene');
    this.activity=activity;
    this.frame=null;this.frameMode='scan';this.frameYaw=0;this.tools=null;this.workspace=null;
    this.indexInformation=null;this.pendingRegion=null;this.regionalHeatmap=null;this.previewDelta=null;this.previewDimensions=null;
    this.embeddedDimensions=(this.state.dimension_overrides||[]).map(item=>({...item,scale:[...item.scale],offset:[...item.offset],matrix:item.matrix||[]}));
    this.state.dimension_overrides=this.embeddedDimensions;
    this.embeddedOverrides=(this.state.overrides||[]).map(item=>({...item,translation:[...item.translation]}));
    this.frameOrigin=this.state.display_origin;
  }

  static decode(value) {
    if(value===null || typeof value!=='object')return value;
    if(Array.isArray(value))return value.map(InspectionController.decode);
    if(value.__items__)return value.__items__.map(InspectionController.decode);
    if(Array.isArray(value.keys)&&Array.isArray(value.values))return Object.fromEntries(value.keys.map((key,i)=>[key,InspectionController.decode(value.values[i])]));
    return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,InspectionController.decode(item)]));
  }

  static identity() {return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];}

  static sizeMatrix(dimensions,inverse=false) {
    if(!dimensions.matrix?.length)return [0,1,2].map(row=>[0,1,2].map(column=>row===column?(inverse?1/dimensions.scale[row]:dimensions.scale[row]):0));
    if(!inverse)return dimensions.matrix;
    const a=dimensions.matrix;
    const matrix=new DOMMatrix([a[0][0],a[1][0],a[2][0],0,a[0][1],a[1][1],a[2][1],0,a[0][2],a[1][2],a[2][2],0,0,0,0,1]).inverse();
    return [[matrix.m11,matrix.m21,matrix.m31],[matrix.m12,matrix.m22,matrix.m32],[matrix.m13,matrix.m23,matrix.m33]];
  }

  static baselineCoordinates(trace,placement,origin,translation,dimensions) {
    const inverse=InspectionController.sizeMatrix(dimensions,true);
    const axes=[new Float64Array(trace.x.length),new Float64Array(trace.x.length),new Float64Array(trace.x.length)];
    for(let point=0;point<trace.x.length;point++){
      const effective=[0,1,2].map(axis=>[trace.x[point],trace.y[point],trace.z[point]].reduce((sum,value,coordinate)=>sum+placement[coordinate][axis]*(value+origin[coordinate]-placement[coordinate][3]),0)-translation[axis]-dimensions.offset[axis]);
      inverse.forEach((row,axis)=>{axes[axis][point]=row.reduce((sum,value,column)=>sum+value*effective[column],0);});
    }
    return axes;
  }

  async initialize() {
    if(this.scene.dataset.plotReady!=='true')await new Promise(resolve=>this.scene.addEventListener('scan-refiner-ready',resolve,{once:true}));
    PlotBuffers.normalize(this.scene.data);
    this.plotUpdates=new PlotUpdates(this.scene,this.activity);
    this.cloud=typeof ScanCloud==='undefined'?null:new ScanCloud(this.scene,this.plotUpdates);
    this.currentCloud={coordinates:[this.scene.data[0].x,this.scene.data[0].y,this.scene.data[0].z],colors:Array.from(this.scene.data[0].marker.color),identities:Array.from(this.scene.data[0].customdata||[])};
    this.previewCloud=this.currentCloud;this.originalRGB=this.previewCloud.colors;this.scanIsOverview=true;
    if(this.cloud&&this.previewCloud.coordinates[0].length>this.cloud.batchCapacity)await this.cloud.replace(this.previewCloud.coordinates,this.previewCloud.colors,this.previewCloud.identities);
    this.meshIndices=this.scene.data.map((trace,index)=>trace.type==='mesh3d'?index:-1).filter(index=>index>=0);
    const initial=this.state.selected_candidate===null?InspectionController.identity():this.state.registration.candidates[this.state.selected_candidate].scan_T_root;
    this.rootCoordinates=this.meshIndices.map(index=>{
      const trace=this.scene.data[index];
      const offset=this.embeddedOverrides.find(item=>item.body_name===trace.name)?.translation||[0,0,0];
      const dimensions=this.embeddedDimensions.find(item=>item.body_name===trace.name)||{scale:[1,1,1],offset:[0,0,0]};
      return InspectionController.baselineCoordinates(trace,initial,this.state.display_origin,offset,dimensions);
    });
    document.querySelectorAll('[data-scene-control]').forEach(control=>{control.disabled=false;});
    this.bindSceneControls();this.refreshScan();
    await this.discoverScanIndex();
    if(typeof InspectionTools!=='undefined'){
      this.tools=new InspectionTools({scene:this.scene,plotUpdates:this.plotUpdates,activity:this.activity,cloud:this.cloud,state:this.state,decodeRecord:InspectionController.decode,
        loadRegion:(...args)=>this.loadRegion(...args),getCloud:()=>this.currentCloud,previewCloud:this.previewCloud,
        refreshScan:()=>this.refreshScan(),refreshGeometry:()=>this.refreshGeometry(),scanIndices:()=>this.scanIndices(),
        setRGB:colors=>{this.originalRGB=colors;},setOverview:value=>{this.scanIsOverview=value;},
        getFrame:()=>this.frame,getFrameOrigin:()=>this.frameOrigin,configureFrame:view=>this.orientFrame(view.coordinate_frame,view.frame_yaw),
        setRegionalHeatmap:(distances,metadata)=>{this.regionalHeatmap=metadata?{distances,...metadata}:null;},
        setProposalHeatmap:heatmap=>{this.proposalHeatmap=heatmap;},
        focusSelection:(lower,upper)=>this.focusSelection(lower,upper),resetFocus:()=>this.resetFocus(),restoreComparison:identifier=>this.restoreComparison(identifier),
        onBeforeRestore:async()=>{await this.workspace?.changeSets?.close();await this.workspace?.refiner?.clear();},isPreview:()=>!!(this.previewDelta||this.previewDimensions||this.workspace?.changeSets?.active),onViewChanged:()=>this.workspace?.synchronize(),updateCounts:(shown,total)=>this.updateCounts(shown,total)});
      await this.tools.initialize();
    }
    if(typeof RefinerWorkspace!=='undefined'){
      this.workspace=new RefinerWorkspace(this);await this.workspace.initialize();
    }
    this.scene.dataset.controllerReady='true';
  }

  scanIndices(){return this.cloud?this.cloud.indices:[0];}

  placement(){const value=document.getElementById('candidate').value;return value===''?InspectionController.identity():this.state.registration.candidates[Number(value)].scan_T_root;}

  bindSceneControls() {
    const select=document.getElementById('candidate');
    if(!this.state.registration.candidates.length){select.add(new Option('Original coordinates',''));select.disabled=true;}
    else {this.state.registration.candidates.forEach((candidate,index)=>select.add(new Option(`${index}: ${(100*candidate.support_fraction).toFixed(1)}% sample support · median ${(1000*candidate.median_distance).toFixed(1)} mm`,index)));select.value=this.state.selected_candidate;}
    select.addEventListener('change',async()=>{
      if(this.workspace?.refiner?.active)await this.workspace.refiner.clear();
      if(this.tools?.inspection&&!this.tools.applying){const view=this.tools.inspection.view;view.clip_enabled=false;view.selected_assembly='';view.selection_lower=[];view.selection_upper=[];}
      await this.orientFrame(this.frameMode,this.frameYaw);
      if(this.tools?.inspection&&!this.tools.applying){await this.tools.render();await this.tools.save('Change placement');}
      this.refreshScan();this.workspace?.synchronize();
    });
    document.getElementById('geometry').addEventListener('change',()=>this.refreshGeometry());
    document.getElementById('scan').addEventListener('change',()=>this.refreshScan());
    document.getElementById('point-size').addEventListener('input',event=>this.plotUpdates.restyle({'marker.size':Number(event.target.value)},this.scanIndices()));
    document.getElementById('mesh-opacity').addEventListener('change',event=>this.plotUpdates.restyle({opacity:Number(event.target.value)},this.meshIndices));
    document.getElementById('point-hover').addEventListener('change',event=>this.plotUpdates.restyle({hoverinfo:event.target.checked?'all':'skip'},this.scanIndices()));
    document.getElementById('rotate-view').addEventListener('click',()=>this.plotUpdates.relayout({'scene.dragmode':'orbit'}));
    document.querySelectorAll('[data-camera]').forEach(button=>button.addEventListener('click',()=>{
      const view=button.dataset.camera;
      const camera=view==='top'?{eye:{x:0,y:0,z:2.5},up:{x:0,y:1,z:0},projection:{type:'orthographic'}}:view==='front'?{eye:{x:0,y:-2.5,z:0},up:{x:0,y:0,z:1},projection:{type:'orthographic'}}:{eye:{x:2.2,y:2.2,z:1.8},up:{x:0,y:0,z:1},projection:{type:'perspective'}};
      this.plotUpdates.relayout({'scene.camera':camera,'scene.dragmode':'orbit'});
    }));
    this.plotUpdates.registerClick(event=>{if(event.points?.length&&this.scene.data[event.points[0].curveNumber].type==='mesh3d'&&!this.tools?.measuring&&!this.tools?.applying&&!this.tools?.loading)this.selectBody(event.points[0].curveNumber);});
    document.getElementById('load-region').addEventListener('click',()=>{
      const bounds=this.readBounds();if(!bounds)return;
      this.loadRegion(bounds.lower,bounds.upper);
    });
    document.getElementById('load-overview').addEventListener('click',()=>this.loadOverview());
  }

  async orientFrame(mode='scan',yaw=0,preserveCamera=false) {
    const activity=this.activity?.begin('Orienting axes','Transforming model geometry into the selected coordinate frame…');
    try {
      await this.activity?.paint();
    this.frameMode=mode;this.frameYaw=yaw;
    if(typeof CoordinateFrame!=='undefined')this.frame=CoordinateFrame.fromPlacement(this.placement(),mode,yaw);
    this.frameOrigin=this.frame?this.frame.point(this.state.display_origin):this.state.display_origin;
    const transform=this.placement();
    const coordinates=this.rootCoordinates.map((root,index)=>{
      const name=this.scene.data[this.meshIndices[index]].name;
      const offset=this.state.overrides?.find(item=>item.body_name===name)?.translation||[0,0,0];
      const proposed=this.changeSetOffsets?.get(name)||(this.previewDelta?.body_name===name?this.previewDelta.translation:[0,0,0]);
      const dimensions=this.state.dimension_overrides?.find(item=>item.body_name===name)||{scale:[1,1,1],offset:[0,0,0]};
      const resize=this.previewDimensions?.body_name===name?this.previewDimensions:{scale:[1,1,1],offset:[0,0,0]};
      const linear=InspectionController.sizeMatrix(dimensions),preview=InspectionController.sizeMatrix(resize);
      const axes=[new Float32Array(root[0].length),new Float32Array(root[0].length),new Float32Array(root[0].length)];
      for(let point=0;point<root[0].length;point++){
        const effective=linear.map((row,axis)=>row.reduce((sum,value,column)=>sum+value*root[column][point],0)+dimensions.offset[axis]+offset[axis]+proposed[axis]);
        const proposedPoint=preview.map((row,axis)=>row.reduce((sum,value,column)=>sum+value*effective[column],0)+resize.offset[axis]);
        const scan=[0,1,2].map(axis=>transform[axis][3]+proposedPoint.reduce((sum,value,column)=>sum+transform[axis][column]*value,0));
        const framed=this.frame?this.frame.point(scan):scan;
        framed.forEach((value,axis)=>{axes[axis][point]=value-this.frameOrigin[axis];});
      }
      return axes;
    });
    const vertices={x:coordinates.map(a=>a[0]),y:coordinates.map(a=>a[1]),z:coordinates.map(a=>a[2])};
    if(preserveCamera)await this.plotUpdates.updateGeometry(vertices,this.meshIndices);
    else await this.plotUpdates.restyle(vertices,this.meshIndices);
    if(this.frame){
      await this.plotUpdates.relayout({'scene.xaxis.title.text':`${mode==='world'?'World':'Scan'} ΔX (m)`,'scene.yaxis.title.text':`${mode==='world'?'World':'Scan'} ΔY (m)`,'scene.zaxis.title.text':`${mode==='world'?'World':'Scan'} ΔZ (m)`});
      const label=document.getElementById('frame-label');if(label)label.textContent=`${mode==='world'?'World':'Scan'} axes${yaw?` · yaw ${yaw}°`:''} · metres`;
    }
    document.getElementById('display-origin').textContent=`${mode==='world'?'World':'Scan'} view origin: ${this.frameOrigin.map(v=>v.toFixed(3)).join(', ')} m`;

    }finally{this.activity?.end(activity);}
  }

  refreshScan() {
    const selector=document.getElementById('scan');
    const value=document.getElementById('candidate').value,selected=value===''?null:Number(value);
    const regional=this.regionalHeatmap&&JSON.stringify(this.placement())===JSON.stringify(this.regionalHeatmap.transform)&&JSON.stringify(this.state.overrides||[])===JSON.stringify(this.regionalHeatmap.overrides||[])&&JSON.stringify(this.state.dimension_overrides||[])===JSON.stringify(this.regionalHeatmap.dimension_overrides||[]);
    const initial=this.scanIsOverview&&this.state.has_heatmap&&selected===this.state.selected_candidate&&JSON.stringify(this.state.overrides||[])===JSON.stringify(this.embeddedOverrides||[])&&JSON.stringify(this.state.dimension_overrides||[])===JSON.stringify(this.embeddedDimensions||[]);
    const review=this.workspace?.changeSets;
    const pair=this.proposalHeatmap;
    const paired=review?.active&&pair&&pair.job.request.change_set_id===review.review?.proposal.identifier&&JSON.stringify(this.placement())===JSON.stringify(pair.job.scan_T_root);
    const valid=!!paired||(!(this.previewDelta||this.previewDimensions||review?.active)&&!!(regional||initial));
    selector.querySelector('[value="heatmap"]').disabled=!valid;
    if(!valid&&selector.value==='heatmap')selector.value='rgb';
    const heatmap=selector.value==='heatmap';
    const distances=paired?(review.proposed?pair.proposed:pair.original):regional?this.regionalHeatmap.distances:this.state.scan_distances;
    const colors=heatmap?distances:this.originalRGB;
    const limit=paired?pair.job.request.tolerance*2:this.state.distance_limit||.1;
    const capacity=this.cloud?this.cloud.batchCapacity:50000;
    const drawing=this.plotUpdates.restyle({visible:selector.value!=='none','marker.color':this.scanIndices().map((_,i)=>colors.slice(i*capacity,(i+1)*capacity)),
      'marker.colorscale':this.scene.data[0].marker.colorscale,'marker.cmin':0,'marker.cmax':limit,
      'marker.showscale':this.scanIndices().map((_,i)=>heatmap&&i===0)},this.scanIndices());
    const comparisonLabel=document.getElementById('comparison-label');if(comparisonLabel)comparisonLabel.textContent=heatmap?(paired?`${review.proposed?'Proposed':'Original'} · distance to visual surfaces`:`Distance to ${regional?this.regionalHeatmap.role:(this.state.heatmap_role||'registration')} surfaces`):'';
    const caption=document.getElementById('heatmap-limit');if(caption)caption.textContent=`0–${Number((limit*1000).toFixed(2))} mm`;
    document.getElementById('status').textContent=selected===null?'No registration candidates. Use Find placements to generate alternatives.':`${selected===this.state.accepted_candidate?'Accepted':'Provisional'} candidate ${selected}. ${valid?'Heatmap available for this placement.':'Compute a heatmap for the current region and placement.'}`;
    return drawing;
  }

  refreshGeometry() {
    const role=document.getElementById('geometry').value;
    return this.plotUpdates.geometry(this.meshIndices.map(index=>role==='both'||role===this.scene.data[index].legendgroup),this.meshIndices);
  }

  async selectBody(index,margin=.15) {
    if(this.workspace?.refiner?.active)await this.workspace.refiner.clear();
    const trace=this.scene.data[index];
    if(trace.type!=='mesh3d')return;
    const {lower,upper}=InspectionController.meshBounds(this.scene,[index],this.frameOrigin,margin);
    this.writeBounds(lower,upper);
    document.getElementById('region-status').textContent=`${trace.name} selected with ${Math.round(margin*100)} cm of scan context. Load the region for more detail.`;
    if(this.tools?.inspection){this.tools.inspection.view.selected_assembly='';this.tools.inspection.view.selected_body=trace.name;await this.tools.regionSelected(lower,upper,'Select object region');}
    this.workspace?.synchronize();
  }

  static meshBounds(scene, indices, origin, margin=0) {
    const lower=[Infinity,Infinity,Infinity],upper=[-Infinity,-Infinity,-Infinity];
    for(const index of indices){
      const trace=scene.data[index];
      for(const [axis,values] of [trace.x,trace.y,trace.z].entries()){
        for(const value of values){
          lower[axis]=Math.min(lower[axis],value);
          upper[axis]=Math.max(upper[axis],value);
        }
      }
    }
    return lower.every(Number.isFinite)&&upper.every(Number.isFinite)?{
      lower:lower.map((value,axis)=>value+origin[axis]-margin),
      upper:upper.map((value,axis)=>value+origin[axis]+margin)
    }:null;
  }

  readBounds() {
    const lower=document.getElementById('region-lower').value.split(',').map(Number),upper=document.getElementById('region-upper').value.split(',').map(Number);
    if(lower.length!==3||upper.length!==3||![...lower,...upper].every(Number.isFinite)||lower.some((value,axis)=>value>upper[axis])){
      document.getElementById('region-status').textContent='Enter three finite ordered coordinates for each inspection bound.';return null;
    }
    return {lower,upper};
  }

  writeBounds(lower,upper){document.getElementById('region-lower').value=lower.map(v=>v.toFixed(6)).join(',');document.getElementById('region-upper').value=upper.map(v=>v.toFixed(6)).join(',');}

  decodeCloud(payload) {
    const view=new DataView(payload),layout=this.indexInformation||this.state.point_layout;
    if(payload.byteLength%layout.record_size)throw Error('Incomplete evidence response.');
    const count=payload.byteLength/layout.record_size;
    const coordinates=[new Float32Array(count),new Float32Array(count),new Float32Array(count)],colors=new Array(count),identities=new Array(count);
    for(let point=0;point<count;point++){
      const offset=point*layout.record_size;
      for(let axis=0;axis<3;axis++)coordinates[axis][point]=view.getFloat64(offset+layout.record_offsets.points+axis*8,true)-this.state.display_origin[axis];
      colors[point]='#'+[0,1,2].map(channel=>view.getUint8(offset+layout.record_offsets.colors+channel).toString(16).padStart(2,'0')).join('');
      identities[point]=`${view.getUint32(offset+layout.record_offsets.station_indices,true)}:${view.getBigUint64(offset+layout.record_offsets.source_indices,true)}`;
    }
    return {coordinates,colors,identities};
  }

  async loadRegion(lower,upper,frozenFrame=null) {
    if(!this.indexInformation)return;
    if(this.pendingRegion)this.pendingRegion.abort();
    this.pendingRegion=new AbortController();const request=this.pendingRegion;
    const status=document.getElementById('region-status');status.textContent='Loading indexed oriented region…';
    this.regionBusy(true);
    const regionActivity=this.activity?.begin('Loading scan region','Reading original scan points…');
    const count=Number(document.getElementById('point-budget').value);
    const matrix=frozenFrame||this.frame?.matrix||InspectionController.identity();
    const query=new URLSearchParams({lower:lower.join(','),upper:upper.join(','),count,frame_T_scan:matrix.flat().join(',')});
    try {
      await this.activity?.paint();
      const response=await fetch(`/api/region?${query}`,{signal:request.signal});
      if(!response.ok)throw Error(`Region request failed (${response.status}). Check its bounds and frame.`);
      const payload=await response.arrayBuffer();if(this.pendingRegion!==request)return;
      this.currentCloud=this.decodeCloud(payload);this.regionalHeatmap=null;this.scanIsOverview=false;
      document.getElementById('scan').value='rgb';
      if(this.tools?.inspection)await this.tools.regionLoaded(lower,upper,count,matrix);
      else await this.cloud.replace(this.currentCloud.coordinates,this.currentCloud.colors,this.currentCloud.identities);
      if(this.pendingRegion!==request)return;
      this.originalRGB=this.tools?.inspection?this.originalRGB:this.currentCloud.colors;
      this.refreshScan();
      const retained=this.currentCloud.coordinates[0].length;
      status.textContent=`${retained.toLocaleString()} indexed original records loaded (budget ${count.toLocaleString()}).`;
      this.updateCounts(retained,retained);this.workspace?.synchronize();
    } catch(error){if(error.name!=='AbortError')status.textContent=error.message;}
    finally{this.activity?.end(regionActivity);if(this.pendingRegion===request)this.regionBusy(false);}
  }

  regionBusy(busy) {
    document.getElementById('load-region').disabled=document.getElementById('load-overview').disabled=busy;
    if(this.tools?.session)this.tools.regionLoading(busy);
  }

  async loadOverview() {
    const activity=this.activity?.begin('Returning to overview','Restoring the scan preview and full scene…');
    try {
      await this.activity?.paint();
    await this.workspace?.refiner?.clear();
    if(this.pendingRegion)this.pendingRegion.abort();this.pendingRegion=null;
    this.regionBusy(false);this.currentCloud=this.previewCloud;this.regionalHeatmap=null;this.scanIsOverview=true;
    document.getElementById('scan').value='rgb';
    if(this.tools?.inspection){
      const view=this.tools.inspection.view;
      view.region_lower=[];view.region_upper=[];view.region_frame_T_scan=[];view.comparison_job='';
      view.clip_enabled=false;view.section_axis='3d';view.selected_body='';view.selected_assembly='';view.selection_lower=[];view.selection_upper=[];view.focus_enabled=false;
      document.getElementById('clip-scan').checked=false;document.getElementById('section-axis').value='3d';
      this.tools.restoredPreview=true;await this.tools.render();await this.resetFocus();await this.tools.save('World overview');
    }else {await this.cloud.replace(this.previewCloud.coordinates,this.previewCloud.colors,this.previewCloud.identities);await this.resetFocus();this.refreshScan();}
    if(this.indexInformation){const bounds=this.frame.bounds(CoordinateFrame.corners(this.indexInformation.index.lower,this.indexInformation.index.upper));this.writeBounds(bounds.lower,bounds.upper);}
    document.getElementById('region-status').textContent=`${this.previewCloud.coordinates[0].length.toLocaleString()} cached overview records. Load a local region for dense detail.`;
    this.workspace?.synchronize();

    }finally{this.activity?.end(activity);}
  }

  async focusSelection(lower,upper,paddingFraction=.1,cameraDistance=2.5) {
    if(!lower||!upper){const bounds=this.readBounds();if(!bounds)return;({lower,upper}=bounds);}
    await this.plotUpdates.pending;
    const minimum=lower.map((value,axis)=>value-this.frameOrigin[axis]);
    const maximum=upper.map((value,axis)=>value-this.frameOrigin[axis]);
    const body=this.tools?.inspection?.view.selected_body;
    const members=new Set(this.workspace?.assemblies?.current?.body_names||(body?[body]:[]));
    const indices=this.meshIndices.filter(index=>members.has(this.scene.data[index].name));
    const geometry=InspectionController.meshBounds(this.scene,indices,[0,0,0]);
    if(geometry)for(const axis of [0,1,2]){
      minimum[axis]=Math.min(minimum[axis],geometry.lower[axis]);
      maximum[axis]=Math.max(maximum[axis],geometry.upper[axis]);
    }
    const extent=Math.max(...maximum.map((value,axis)=>value-minimum[axis]));
    if(!Number.isFinite(extent)||extent<=0)return;
    const halfExtent=extent*(1+paddingFraction)/2,layout={'scene.aspectmode':'cube'};
    ['x','y','z'].forEach((axis,i)=>{
      const center=(minimum[i]+maximum[i])/2;
      layout[`scene.${axis}axis.range`]=[center-halfExtent,center+halfExtent];layout[`scene.${axis}axis.autorange`]=false;
    });
    const camera=this.scene._fullLayout.scene.camera;
    let direction=['x','y','z'].map(axis=>camera.eye[axis]-camera.center[axis]);
    let length=Math.hypot(...direction);
    if(!length){direction=[1,1,1];length=Math.sqrt(3);}
    const viewport=this.scene.getBoundingClientRect();
    const distance=cameraDistance*Math.max(1,viewport.height/Math.max(1,viewport.width));
    layout['scene.camera']={up:{...camera.up},projection:{...camera.projection},center:{x:0,y:0,z:0},
      eye:Object.fromEntries(['x','y','z'].map((axis,i)=>[axis,direction[i]*distance/length]))};
    const applying=this.tools?.applying;if(this.tools)this.tools.applying=true;
    return this.plotUpdates.relayout(layout).finally(()=>{if(this.tools)this.tools.applying=applying;});
  }

  resetFocus(){return this.plotUpdates.relayout({'scene.aspectmode':'data','scene.xaxis.autorange':true,'scene.yaxis.autorange':true,'scene.zaxis.autorange':true});}

  updateCounts(shown,total){const element=document.getElementById('display-count');if(element)element.textContent=shown===total?`${total.toLocaleString()} scan points`:`${shown.toLocaleString()} shown / ${total.toLocaleString()} loaded`;}

  async comparisonData(job) {
    const responses=await Promise.all([fetch(`/api/jobs/${job.identifier}/points`),fetch(`/api/jobs/${job.identifier}/distances`)]);
    if(responses.some(response=>!response.ok))throw Error('This comparison is unfinished or stale.');
    const payloads=await Promise.all(responses.map(response=>response.arrayBuffer()));
    const source=this.decodeCloud(payloads[0]);source.distances=Array.from(new Float32Array(payloads[1]));
    if(source.coordinates[0].length!==source.distances.length)throw Error('Comparison identity/colour payload mismatch.');
    const reportResponse=await fetch(`/api/jobs/${job.identifier}/report`);
    if(!reportResponse.ok&&reportResponse.status!==404)throw Error('This comparison report is unavailable or stale.');
    const report=reportResponse.ok?InspectionController.decode(await reportResponse.json()):null;
    source.comparison={transform:job.scan_T_root,role:job.request.geometry_role,job,report,overrides:structuredClone(this.state.overrides||[]),dimension_overrides:structuredClone(this.state.dimension_overrides||[])};
    return source;
  }

  async restoreComparison(identifier) {
    const response=await fetch(`/api/jobs/${identifier}`);if(!response.ok)return false;
    const job=InspectionController.decode(await response.json());
    if(job.status!=='succeeded'||JSON.stringify(job.scan_T_root)!==JSON.stringify(this.placement()))return false;
    this.currentCloud=await this.comparisonData(job);return true;
  }

  async showComparison(job) {
    if(this.tools.applying||this.tools.loading)throw Error('Wait for the current view change to finish.');
    await this.tools.change('Show regional heatmap',async()=>{
      const source=await this.comparisonData(job);
      document.getElementById('candidate').value=String(job.request.candidate_index);
      await this.orientFrame(this.frameMode,this.frameYaw);
      this.currentCloud=source;this.scanIsOverview=false;
      const view=this.tools.inspection.view;view.clip_enabled=false;view.section_axis='3d';view.focus_enabled=true;
      view.region_lower=job.request.lower;view.region_upper=job.request.upper;view.region_frame_T_scan=job.request.frame_T_scan;view.point_budget=job.request.maximum_count;view.comparison_job=job.identifier;
      document.getElementById('clip-scan').checked=false;document.getElementById('section-axis').value='3d';document.getElementById('scan').value='heatmap';
      document.getElementById('point-budget').value=String(job.request.maximum_count);
      await this.tools.regionLoaded(job.request.lower,job.request.upper,job.request.maximum_count,job.request.frame_T_scan);
      if(job.request.lower.length){const frame=new CoordinateFrame(job.request.frame_T_scan);const bounds=this.frame.bounds(CoordinateFrame.corners(job.request.lower,job.request.upper).map(p=>frame.toScan(p)));await this.tools.regionSelected(bounds.lower,bounds.upper,'Inspect comparison result');await this.focusSelection(bounds.lower,bounds.upper);}
      this.refreshScan();
    });
  }

  async discoverScanIndex() {
    const status=document.getElementById('region-status');
    if(location.protocol==='file:'){status.textContent='Serve this project locally for indexed detail and persistent actions.';return;}
    try {
      const response=await fetch('/api/scan-info');
      if(!response.ok){status.textContent='Preview only: attach a completed scan index to load local detail.';return;}
      this.indexInformation=InspectionController.decode(await response.json());
      if(this.indexInformation.index.schema_version!==1)throw Error('Unsupported scan index format.');
      document.getElementById('load-region').disabled=document.getElementById('load-overview').disabled=false;
      this.writeBounds(this.indexInformation.index.lower,this.indexInformation.index.upper);
      status.textContent=`${this.indexInformation.index.point_count.toLocaleString()} indexed records. Pick an object or draw a crop.`;
    }catch(error){status.textContent=`Regional detail unavailable: ${error.message}`;}
  }
}

// %% Entry point
async function initializeViewer(){
  const activity=new ViewerActivity(document.getElementById('viewer-busy'));
  await activity.run('Preparing 3D view','Building the model and scan preview…',async()=>{
    await globalThis.prepareViewerFigure?.();
    await new InspectionController(activity).initialize();
  }).catch(error=>{activity.fail(error.message);console.error(error);});
}
initializeViewer();
