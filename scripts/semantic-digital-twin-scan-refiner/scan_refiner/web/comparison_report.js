"use strict";

// %% Directional evidence inspector
class ComparisonPanel {
  constructor(){this.report=null;}

  clear() {
    this.report=null;
    document.getElementById('comparison-report').hidden=true;
    document.getElementById('report-empty').hidden=false;
  }

  show(report,identifier) {
    if(!report){this.clear();return;}
    this.report=report;
    document.getElementById('comparison-report').hidden=false;
    document.getElementById('report-empty').hidden=true;
    const overview=document.getElementById('report-overview');overview.replaceChildren();
    const value=report.supported_area_fraction===null?'Not sampled':`${(100*report.supported_area_fraction).toFixed(1)}%`;
    const statistic=document.createElement('strong');statistic.textContent=value;
    const caption=document.createElement('span');caption.textContent='Estimated model area with nearby evidence';
    const detail=document.createElement('small');detail.textContent=`${report.model_sample_count.toLocaleString()} in-crop model samples · ${(1000*report.tolerance).toFixed(0)} mm tolerance`;
    overview.append(statistic,caption,detail);
    const download=document.getElementById('report-download');download.href=`/api/jobs/${identifier}/report`;download.download=`comparison-${identifier}.json`;
    for(const id of ['report-search','report-sort','report-sampled-only'])document.getElementById(id).oninput=()=>this.drawBodies();
    this.drawBodies();
  }

  drawBodies() {
    if(!this.report)return;
    const query=document.getElementById('report-search').value.toLowerCase();
    const sampledOnly=document.getElementById('report-sampled-only').checked;
    const order=document.getElementById('report-sort').value;
    const bodies=this.report.bodies.filter(body=>body.body_name.toLowerCase().includes(query)&&(!sampledOnly||body.scan.sample_count||body.model.sample_count));
    bodies.sort((a,b)=>order==='name'?a.body_name.localeCompare(b.body_name):order==='residual'?(b.scan.percentile90_distance??-1)-(a.scan.percentile90_distance??-1):(a.model.within_tolerance??Infinity)-(b.model.within_tolerance??Infinity));
    const list=document.getElementById('report-body-list');list.replaceChildren();
    document.getElementById('report-count').textContent=`${bodies.length} / ${this.report.bodies.length} bodies`;
    for(const body of bodies){
      const card=document.createElement('article');card.className='body-result';card.dataset.body=body.body_name;
      const name=document.createElement('h4');name.textContent=body.body_name;card.append(name);
      const grid=document.createElement('div');grid.className='body-result-metrics';
      const fit=document.createElement('div');const fitLabel=document.createElement('span');fitLabel.className='metric-label';fitLabel.textContent='SCAN → MODEL';fit.append(fitLabel);
      const fitValue=document.createElement('strong');fitValue.textContent=body.scan.sample_count?`P90 ${(body.scan.percentile90_distance*1000).toFixed(1)} mm`:'No assigned scan records';fit.append(fitValue);
      const fitDetail=document.createElement('small');fitDetail.textContent=body.scan.sample_count?`Median ${(body.scan.median_distance*1000).toFixed(1)} mm · ${body.scan.sample_count.toLocaleString()} records`:'Residual unknown';fit.append(fitDetail);
      const support=document.createElement('div');const supportLabel=document.createElement('span');supportLabel.className='metric-label';supportLabel.textContent='MODEL → SCAN';support.append(supportLabel);
      const supportValue=document.createElement('strong');supportValue.textContent=body.model.within_tolerance===null?'Not sampled in crop':body.model.within_tolerance===0?'No nearby evidence':`${(100*body.model.within_tolerance).toFixed(1)}% supported`;support.append(supportValue);
      const supportDetail=document.createElement('small');supportDetail.textContent=body.model.sample_count?`${body.model.sample_count.toLocaleString()} surface samples`:'Increase model samples or enlarge the crop';support.append(supportDetail);
      if(body.model.within_tolerance!==null){const bar=document.createElement('progress');bar.max=1;bar.value=body.model.within_tolerance;bar.setAttribute('aria-label',`Nearby-evidence fraction for ${body.body_name}`);support.append(bar);}
      grid.append(fit,support);card.append(grid);list.append(card);
    }
    if(!bodies.length){const empty=document.createElement('p');empty.className='hint';empty.textContent='No bodies match these filters. Try showing unsampled bodies.';list.append(empty);}
  }
}
