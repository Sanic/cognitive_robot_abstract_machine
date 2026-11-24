// %% Inspector navigation
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const pane={scrollTop:600};
const context=vm.createContext({document:{querySelectorAll:()=>[],getElementById:()=>pane,querySelector:()=>pane}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext("RefinerWorkspace.prototype.inspector.call({},'results')",context);
assert.equal(pane.scrollTop,0,'A newly opened results tab must start at its heading rather than the preceding form scroll position');
