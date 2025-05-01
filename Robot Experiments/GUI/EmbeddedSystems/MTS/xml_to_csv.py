import pandas as pd
from lxml import etree
from io import StringIO
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def readXML(fname,grasper_type,experiment):

    with open(fname, 'r') as file:
        content = file.read()

    parser = etree.XMLParser(load_dtd=True, resolve_entities=True)
    root = etree.parse(os.path.normpath(fname), parser=parser)

    # Parse through each variable data
    varnames = root.xpath("//VariableData//Name") #get variable names
    result_dict = {"variable":[],"value":[], "unit":[], "grasper_type":[],"experiment":[],"idx":[]}

    for (i,x) in enumerate(varnames):

        r = root.xpath("//VariableData[%i]//Values//Value"%(i+1))
        u = root.xpath("//VariableData[%i]//Unit"%(i+1))


        result_dict["value"].extend([float(vv.text) for vv in r])
        result_dict["variable"].extend([x.text for vv in r])
        result_dict["unit"].extend([u[0].text for vv in r])
        result_dict["grasper_type"].extend([grasper_type for vv in r])
        result_dict["experiment"].extend([experiment for vv in r])
        result_dict["idx"].extend([vv for vv in range(0,len(r))])


    DF = pd.DataFrame.from_dict(result_dict)
    return DF

fnames = ["Soft Grasper\\TwoSoftJaws_BlueGreenPlaydoh_Test1_11_26.xml",
          "Soft Grasper\\TwoSoftJaws_BlueGreenPlaydoh_Test2_11_26.xml",
          "Soft Grasper\\TwoSoftJaws_BlueGreenPlaydoh_Test3_11_26.xml",
          "Rigid Grasper\\RigidGrasper_BlueGreenPlaydoh_Test1_11_27.xml",
          "Rigid Grasper\\RigidGrasper_BlueGreenPlaydoh_Test2_11_27.xml",
          "Rigid Grasper\\RigidGrasper_BlueGreenPlaydoh_Test3_11_27.xml",
          ]

grasper_type = ["Soft",
               "Soft",
               "Soft",
               "Rigid",
               "Rigid",
               "Rigid"]

experiment = ["1",
              "2",
              "3",
              "1",
              "2",
              "3"]

DF = pd.DataFrame()

for (i,x) in enumerate(fnames):
    DF2 = readXML(fnames[i], grasper_type[i], experiment[i])
    DF= pd.concat([DF,DF2])

rigid_axial = (DF["grasper_type"]=="Rigid") &  (DF["variable"]=="AxialDisplacementArray")
DF.loc[rigid_axial,"value"] = DF.loc[rigid_axial,"value"]*1000 #convert from m to mm

soft_p = (DF["grasper_type"]=="Soft") &  ((DF["variable"]=="Pressure1_valueArray") | (DF["variable"]=="Pressure2vArray"))
DF.loc[soft_p,"value"] = DF.loc[soft_p,"value"]*145.038 #convert from m to mm

print(DF)

DF2 = pd.pivot_table( DF, index = ["grasper_type","experiment","idx"], columns = ['variable'],values = "value").reset_index() #unpivot table.

DF2.to_excel("Pivoted_Soft_Rigid_MTS.xlsx",index=False)


fig = px.scatter(DF2, x="AxialDisplacementArray", y="AxialForceArray", color="grasper_type",symbol="experiment")
fig.show()
#fig_tot = go.Figure()
#fig_tot.add_trace(go.Scatter(x=DF2["AxialDisplacementArray"], y=DF2["AxialForceArray"], color=DF2["grasper_type"]))



