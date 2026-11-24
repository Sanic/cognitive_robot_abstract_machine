"use strict";
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context=vm.createContext({console});
vm.runInContext(fs.readFileSync(path.join(process.argv[2],'inspection_tools.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(process.argv[2],'viewer.js'),'utf8').split('// %% Entry point')[0],context);
context.payload=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
process.stdout.write(vm.runInContext('JSON.stringify(InspectionTools.encode(InspectionController.decode(payload)))',context));
