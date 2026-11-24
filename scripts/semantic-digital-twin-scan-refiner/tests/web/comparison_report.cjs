// %% Missing evidence presentation
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this.textContent='';this.hidden=false;this.checked=true;this.value='';this.dataset={};}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;}
 setAttribute(key,value){this[key]=value;}
 get text(){return this.textContent+this.children.map(child=>child.text||'').join(' ');}
}
const elements=Object.fromEntries(['comparison-report','report-empty','report-body-list','report-search','report-sort','report-sampled-only','report-overview','report-download','report-count'].map(id=>[id,new Element()]));
elements['report-sort'].value='support';
const context=vm.createContext({document:{getElementById:id=>elements[id],createElement:()=>new Element()}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext(`{
 const empty={sample_count:0,median_distance:null,percentile90_distance:null,within_tolerance:null};
 const panel=new ComparisonPanel();
 panel.show({tolerance:.02,model_sample_count:4,supported_area_fraction:0,bodies:[
  {body_name:'test/unobserved',scan:empty,model:{sample_count:4,median_distance:2,percentile90_distance:2,within_tolerance:0}},
  {body_name:'test/outside',scan:empty,model:empty}
 ]},'abc');
 globalThis.panel=panel;
}`,context);
assert.equal(elements['report-body-list'].children.length,1);
assert.match(elements['report-body-list'].text,/No assigned scan records/);
assert.match(elements['report-body-list'].text,/No nearby evidence/);
assert.doesNotMatch(elements['report-body-list'].text,/Median 0/);
elements['report-sampled-only'].checked=false;context.panel.drawBodies();
assert.equal(elements['report-body-list'].children.length,2);
assert.match(elements['report-body-list'].text,/Not sampled in crop/);
assert.equal(elements['report-download'].href,'/api/jobs/abc/report');
context.panel.clear();assert.equal(elements['comparison-report'].hidden,true);
