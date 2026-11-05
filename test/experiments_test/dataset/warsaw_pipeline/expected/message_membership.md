## The part
door_10, labelled "door"

## What the ontology says
door_10 was read as Door
cabinet_20 was read as Cabinet
cabinet_4 was read as Cabinet

Door(HasHandle, HasMechanicalJoint)
  part mechanical_joint -> MechanicalJoint
  part handle -> Handle
  part entry_way -> EntryWay
Cabinet(Furniture, HasCaseAsRootBody, HasDoors, HasDrawers)
  part drawers -> Drawer (many)
  part doors -> Door (many)

Cabinet.doors may hold a Door (part)

## What was measured
door_10: 3800 faces, 1.982 m2, middle 1.72 m up, 1 piece(s)
cabinet_20: 1775 faces, 2.0 m2, middle 1.57 m up, 1 piece(s)
cabinet_4: 3839 faces, 1.997 m2, middle 1.73 m up, 1 piece(s)

## The candidates
cabinet_20: shares 9 faces with it, touches it along 44 edges, 0.0 m between their surfaces, and would hold it in its doors
cabinet_4: shares 3789 faces with it, touches it along 5626 edges, 0.0 m between their surfaces, and would hold it in its doors

## The picture
door_10 (labelled "door") is painted in the colour tomato
cabinet_20 (labelled "cabinet") is painted in the colour royalblue
cabinet_4 (labelled "cabinet") is painted in the colour greenyellow
the faces all of them claim are painted in the colour mediumorchid

Picture 1 -- the objects alone, with nothing in front of them.

Picture 2 -- where they are in the room, painted the same way.

Picture 3 -- the same objects, in the colors they were scanned in.