"use strict";

// %% Plotly numeric buffers
class PlotBuffers {
  static read(value) {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return value;
    if (ArrayBuffer.isView(value?._inputArray)) return value._inputArray;
    const constructors = {f4:Float32Array, f8:Float64Array, i1:Int8Array, u1:Uint8Array,
      i2:Int16Array, u2:Uint16Array, i4:Int32Array, u4:Uint32Array};
    const constructor = constructors[value?.dtype];
    if (!constructor || typeof value.bdata !== "string") throw new Error("Unsupported Plotly numeric buffer.");
    const bytes = Uint8Array.from(atob(value.bdata), character => character.charCodeAt(0));
    return new constructor(bytes.buffer);
  }

  static normalize(traces) {
    for (const trace of traces) {
      for (const coordinate of ["x", "y", "z", "i", "j", "k"]) {
        if (trace[coordinate] !== undefined) trace[coordinate] = PlotBuffers.read(trace[coordinate]);
      }
    }
  }
}

// %% Ordered scene updates
class PlotUpdates {
  constructor(scene, activity = null) {
    this.scene = scene;
    this.activity = activity;
    this.pending = Promise.resolve();
  }

  registerClick(callback) {
    this.clicks ||= new SceneClicks(this.scene);
    this.clicks.callbacks.push(callback);
  }

  restyle(properties, indices) {
    const update = {...properties};
    if (Array.isArray(update["marker.colorscale"])) update["marker.colorscale"] = [update["marker.colorscale"]];
    return this.enqueue(() => Plotly.restyle(this.scene, update, indices),'Updating 3D display','Drawing the selected model and scan layers…');
  }

  addTraces(traces) {
    return this.enqueue(async () => {
      const indices = traces.map((_, position) => this.scene.data.length + position);
      await Plotly.addTraces(this.scene, traces);
      return indices;
    },'Preparing 3D layers','Adding the selected geometry to the scene…');
  }

  relayout(properties) {
    this.pending = this.pending.then(() => Plotly.relayout(this.scene, properties));
    return this.pending;
  }

  resize() {
    this.pending = this.pending.then(() => {
      const bounds=this.scene.getBoundingClientRect();
      if(bounds.width>0 && bounds.height>0)return Plotly.Plots.resize(this.scene);
    });
    return this.pending;
  }

  geometry(visible, indices) {
    return this.updateGeometry({visible}, indices);
  }

  updateGeometry(properties, indices) {
    return this.enqueue(() => {
      const current = this.scene._fullLayout.scene;
      const camera = JSON.parse(JSON.stringify(current._scene?.getCamera?.() || current.camera));
      return Plotly.update(this.scene, properties, {"scene.camera":camera, "scene.dragmode":current.dragmode || "orbit"}, indices);
    },'Switching geometry','Drawing the selected visual and collision layers…');
  }

  enqueue(operation,title,detail) {
    this.pending=this.pending.then(()=>this.activity?this.activity.run(title,detail,operation):operation());
    return this.pending;
  }
}

// %% Defer picking until the renderer has finished the mouse gesture
class SceneClicks {
  constructor(scene, dragThreshold = 5) {
    this.callbacks=[];this.pressed=false;this.dragged=false;this.event=null;this.dragThreshold=dragThreshold;
    const target=scene.ownerDocument?.defaultView||scene;
    scene.addEventListener('mousedown',event=>{
      clearTimeout(this.timer);this.event=null;this.pressed=event.button===0;this.dragged=!this.pressed;
      this.startX=event.clientX;this.startY=event.clientY;
    },true);
    target.addEventListener('mousemove',event=>{
      if(this.pressed&&Math.hypot(event.clientX-this.startX,event.clientY-this.startY)>this.dragThreshold){this.dragged=true;this.event=null;}
    },true);
    target.addEventListener('mouseup',event=>{
      if(event.button!==0)return;
      if(this.pressed&&Math.hypot(event.clientX-this.startX,event.clientY-this.startY)>this.dragThreshold){this.dragged=true;this.event=null;}
      this.pressed=false;this.deliverClick();
    },true);
    target.addEventListener('blur',()=>{this.pressed=false;this.dragged=true;this.event=null;clearTimeout(this.timer);});
    scene.on('plotly_click',event=>{
      if(this.dragged)return;
      this.event=event;if(!this.pressed)this.deliverClick();
    });
  }

  deliverClick() {
    clearTimeout(this.timer);
    if(this.pressed||this.dragged||!this.event)return;
    this.timer=setTimeout(()=>{
      const event=this.event;this.event=null;
      if(event&&!this.pressed&&!this.dragged)for(const callback of this.callbacks)callback(event);
    },0);
  }
}

// %% Visible foreground operations and nested interaction locks
class ViewerActivity {
  constructor(element) {
    this.element=element;this.operations=new Map();this.sequence=0;this.locks=new Map();
  }

  begin(title,detail) {
    const token=++this.sequence;this.operations.set(token,{title,detail});
    if(this.element&&this.operations.size===1){
      for(const control of this.element.ownerDocument.querySelectorAll('[data-viewer-lock]')){this.locks.set(control,control.inert);control.inert=true;}
    }
    this.draw();return token;
  }

  update(token,detail) {
    if(this.operations.has(token))this.operations.get(token).detail=detail;
    this.draw();
  }

  end(token) {
    this.operations.delete(token);this.draw();
  }

  draw() {
    if(!this.element)return;
    const operations=[...this.operations.values()],active=operations.length>0;
    this.element.hidden=!active;this.element.dataset.state='loading';
    this.element.parentElement.setAttribute('aria-busy',String(active));
    this.element.ownerDocument.body.classList.toggle('viewer-busy',active);
    if(active){
      this.element.querySelector('[data-activity-title]').textContent=operations[0].title;
      this.element.querySelector('[data-activity-detail]').textContent=operations.at(-1).detail;
      this.element.querySelector('[data-activity-retry]').hidden=true;
    }else{
      for(const [control,inert] of this.locks)control.inert=inert;
      this.locks.clear();
    }
  }

  async paint() {
    if(!this.element)return;
    await new Promise(resolve=>requestAnimationFrame(()=>setTimeout(resolve,0)));
  }

  run(title,detail,operation) {
    const token=this.begin(title,detail);
    return this.paint().then(operation).finally(()=>this.end(token));
  }

  fail(message) {
    if(!this.element)return;
    this.begin('Could not prepare the 3D view',message);
    this.element.dataset.state='error';this.element.querySelector('[data-activity-retry]').hidden=false;
  }
}
