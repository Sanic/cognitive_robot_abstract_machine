"use strict";

// %% Branch identity and membership
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const WorldTree = vm.runInContext("WorldTree", context);
const tree = new WorldTree({
  objects: new Map([
    ["test/box", {indices: [1, 2]}],
    ["test/handle", {indices: [3]}],
    ["test/door", {indices: [4]}],
  ]),
  pretty: name => name.split("/").at(-1),
}, [
  {name: "test/root", parent_name: null},
  {name: "test/support", parent_name: "test/root"},
  {name: "test/box", parent_name: "test/support"},
  {name: "test/handle", parent_name: "test/box"},
  {name: "test/door", parent_name: "test/root"},
  {name: "test/empty", parent_name: "test/root"},
]);

assert.deepEqual(JSON.parse(JSON.stringify(tree.scope("kinematic:test/support"))), {
  identifier: "kinematic:test/support",
  label: "support",
  root_name: "test/support",
  origin: "kinematic",
  body_names: ["test/box", "test/handle"],
});
assert.equal(tree.scope("kinematic:test/empty"), null);
assert.equal(tree.scope("saved:test/support"), null);
