## The proposal
Give CounterTop the mixin HasDrawers, which introduces: drawers -> Drawer.

## The class as it stands
CounterTop(Furniture, HasSupportingSurface, HasSink, HasCooktops)
  part cooktops -> Cooktop (many)
  part sink -> Sink

## The part
Drawer(Furniture, HasCaseAsRootBody, IsGraspedAtHandle, HasMovableJoint, HasKnobs)
  part knobs -> Knob (many)
  part handles -> Handle (many)

## What those parts hold in turn
Cooktop(HasRootBody, HasControls)
  part controls -> Control (many)
Sink(HasRootBody, HasDrainPipes, HasFaucets)
  part faucets -> Faucet (many)
  part drain_pipes -> DrainPipe (many)

## What was measured
In one scanned room, objects labelled countertop were read as CounterTop, and objects labelled drawer as Drawer. They share faces over 7 measured pairs, 1234 shared faces in all.