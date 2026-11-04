## The labels
cabinet, drawer, kitchen_island

## What the ontology says
cabinet_8 was read as Cabinet
drawer_5 was read as Drawer
kitchen_island_1 was read as KitchenIsland

Cabinet(Furniture, HasCaseAsRootBody, HasDoors, HasDrawers)
  part drawers -> Drawer (many)
  part doors -> Door (many)
Drawer(Furniture, HasCaseAsRootBody, HasHandle, HasMechanicalJoint)
  part mechanical_joint -> MechanicalJoint
  part handle -> Handle
KitchenIsland(Table, HasDrawers, HasSupportingSurface)
  part drawers -> Drawer (many)

Cabinet.drawers may hold a Drawer (part)
KitchenIsland.drawers may hold a Drawer (part)

## The picture
cabinet_8 (labelled "cabinet") is tomato
drawer_5 (labelled "drawer") is royalblue
kitchen_island_1 (labelled "kitchen_island") is greenyellow
the faces all of them claim are mediumorchid

## What was measured
cabinet_8: 1589 faces, 1.43 m2, middle 0.87 m up, 1 piece(s)
drawer_5: 1741 faces, 1.53 m2, middle 0.87 m up, 1 piece(s)
kitchen_island_1: 27501 faces, 30.695 m2, middle 1.69 m up, 1 piece(s)
of cabinet_8 the contested 1503 faces are 95%
of drawer_5 the contested 1503 faces are 86%
of kitchen_island_1 the contested 1503 faces are 5%

## How often this happens
Objects with these labels are labelled over the same faces 4 time(s) in this room, 4318 faces in all. The pictures show the largest of them.

Picture 1 -- the objects alone, with nothing in front of them.

Picture 2 -- where they are in the room, painted the same way.

Picture 3 -- the same objects, in the colors they were scanned in.