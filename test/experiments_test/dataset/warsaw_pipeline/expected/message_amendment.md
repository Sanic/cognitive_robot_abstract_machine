## The proposal
Give CounterTop the mixin HasDrawers, which introduces: drawers -> Drawer.

## The class as it stands
CounterTop(Furniture, HasSupportingSurface, HasSink, HasCooktops)
  part cooktops -> Cooktop (many), mounted with add()
  part sink -> Sink, mounted with add()
  contains objects -> HasRootBody (many), mounted with add_object()
  supports supporting_surface -> Region, mounted with add_supporting_surface()

## The part
Drawer(Furniture, HasCaseAsRootBody, IsGraspedAtHandle, HasMovableJoint, HasKnobs)
  part knobs -> Knob (many), mounted with add()
  part handles -> Handle (many), mounted with add()
  contains objects -> HasRootBody (many), mounted with add_object()
  supports supporting_surface -> Region, mounted with add_supporting_surface()

## What those parts hold in turn
Cooktop(HasRootBody, HasControls)
  part controls -> Control (many), mounted with add()
Sink(HasRootBody, HasDrainPipes, HasFaucets)
  part faucets -> Faucet (many), mounted with add()
  part drain_pipes -> DrainPipe (many), mounted with add()

## What was measured
In one scanned room, objects labelled countertop were read as CounterTop, and objects labelled drawer as Drawer. They share faces over 7 measured pairs, 1234 shared faces in all.