"use strict";

// %% Bounded WebGL point batches
class ScanCloud {
  constructor(scene, updates, batchCapacity = 50000) {
    this.scene = scene;
    this.updates = updates;
    this.batchCapacity = batchCapacity;
    this.indices = [0];
  }

  async replace(coordinates, colors, identities) {
    const batches = Math.max(1, Math.ceil(coordinates[0].length / this.batchCapacity));
    while (this.indices.length < batches) {
      const [index] = await this.updates.addTraces([{type:'scatter3d',mode:'markers',x:[],y:[],z:[],
        marker:{...this.scene.data[0].marker},showlegend:false,hoverinfo:'skip',hovertemplate:this.scene.data[0].hovertemplate,name:'Scan detail'}]);
      this.indices.push(index);
    }
    const parts = this.indices.map((_, batch) => {
      const start = batch * this.batchCapacity, end = start + this.batchCapacity;
      return {axes: coordinates.map(values => values.slice(start,end)),
        colors:colors.slice(start,end),ids:identities.slice(start,end)};
    });
    await this.updates.restyle({x:parts.map(p=>p.axes[0]),y:parts.map(p=>p.axes[1]),z:parts.map(p=>p.axes[2]),
      customdata:parts.map(p=>p.ids),'marker.color':parts.map(p=>p.colors),
      'marker.showscale':false,visible:parts.map(p=>p.axes[0].length>0)},this.indices);
  }
}
