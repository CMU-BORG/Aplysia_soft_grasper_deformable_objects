model = createpde;
importGeometry(model,'SoftJaw_Sealed_NoCavity.STL');
generateMesh(model,"GeometricOrder","linear",'Hmin',2);
pdeplot3D(model)
stlwrite(model.Mesh,'SoftJaw_Sealed_NoCavity_MATLAB.STL');
