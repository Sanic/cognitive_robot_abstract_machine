"use strict";

// %% Rigid inspection coordinates
class CoordinateFrame {
  constructor(matrix = CoordinateFrame.identity()) {
    this.matrix = matrix;
    this.inverse = CoordinateFrame.invert(matrix);
  }

  static identity() {
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  }

  static invert(matrix) {
    return [0,1,2].map(axis => [matrix[0][axis],matrix[1][axis],matrix[2][axis],
      -[0,1,2].reduce((sum,column)=>sum+matrix[column][axis]*matrix[column][3],0)]).concat([[0,0,0,1]]);
  }

  static fromPlacement(scanTRoot, mode='scan', yawDegrees=0) {
    const base=mode==='world'?CoordinateFrame.invert(scanTRoot):CoordinateFrame.identity();
    const radians=-yawDegrees*Math.PI/180,c=Math.cos(radians),s=Math.sin(radians);
    const yaw=[[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]];
    return new CoordinateFrame(yaw.map(row=>[0,1,2,3].map(column=>row.reduce((sum,value,index)=>sum+value*base[index][column],0))));
  }

  static transform(matrix, point) {
    return [0,1,2].map(axis=>matrix[axis][3]+point.reduce((sum,value,column)=>sum+matrix[axis][column]*value,0));
  }

  point(scanPoint) {
    return CoordinateFrame.transform(this.matrix,scanPoint);
  }

  toScan(framePoint) {
    return CoordinateFrame.transform(this.inverse,framePoint);
  }

  bounds(scanPoints) {
    const lower=[Infinity,Infinity,Infinity],upper=[-Infinity,-Infinity,-Infinity];
    for(const point of scanPoints){
      const values=this.point(point);
      values.forEach((value,axis)=>{lower[axis]=Math.min(lower[axis],value);upper[axis]=Math.max(upper[axis],value);});
    }
    return {lower,upper};
  }

  static corners(lower,upper) {
    return Array.from({length:8},(_,mask)=>[0,1,2].map(axis=>(mask&(1<<axis))?upper[axis]:lower[axis]));
  }
}
