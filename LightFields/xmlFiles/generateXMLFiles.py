import xml.etree.ElementTree as etree
import xml.dom.minidom
import subprocess 
import os
import imageio
import h5py
import numpy as np 

def createXMLstring(filename,scaleVal,cameraPosX,cameraPosY):
	scene = etree.Element("scene",version="0.5.0")
	sensor = etree.SubElement(scene, "sensor", type="perspective")
	sensor_transform = etree.SubElement(sensor,"transform",name="toWorld")
	etree.SubElement(sensor_transform,"lookat",origin=str(5)+","+cameraPosX+","+cameraPosY,target="0,0,0",up="0,1,0")
	sensor_sampler = etree.SubElement(sensor,"sampler",type="ldsampler")
	etree.SubElement(sensor_sampler,"integer",name="sampleCount",value="128")
	sensor_film = etree.SubElement(sensor,"film",type="ldrfilm")
	etree.SubElement(sensor_film,"boolean",name="banner",value="false")
	etree.SubElement(sensor_film,"integer",name="width",value="400")
	etree.SubElement(sensor_film,"integer",name="height",value="400")

	shapeObj = etree.SubElement(scene,"shape",type="obj")
	shapeObj_string = etree.SubElement(shapeObj,"string",name="filename",value=filename+".obj")
	shapeObj_transform = etree.SubElement(shapeObj,"transform",name="toWorld")
	etree.SubElement(shapeObj_transform,"scale",value=scaleVal)
	etree.SubElement(shapeObj_transform,"rotate",angle="60",y="1")

	rough_string = etree.tostring(scene, "utf-8")
	reparsed = xml.dom.minidom.parseString(rough_string)
	reparsed_pretty = reparsed.toprettyxml(indent=" " * 4)
	return reparsed_pretty

def create_h5(data, label, path, file_name):

	with h5py.File(os.path.join(path, file_name), 'w') as file:
		file.create_dataset("data", data = data)
		file.create_dataset("label", data = label)


filenames = ["airboat","al","alfa147","cessna","cube","diamond","dodecahedron","gourd","humanoid_quad","humanoid_tri","icosahedron","lamp","magnolia","minicooper","octahedron","power_lines","roi","sandal","shuttle","skyscraper","slot_machine","teapot","tetrahedron","violin_case"]
scaleVal  = [0.5,0.5,0.01,0.08,0.5,0.01,0.5,0.5,0.1,0.1,0.5,0.2,0.025,0.01,0.5,0.07,0.02,0.2,0.1,0.03,0.1,0.01,0.5,0.5]
index = 0
cameraPosOrigin = [5,1,-3]
deltaCam = 0.1
hr_image = []
lr_image = []
destination_path     = "/home/sudarshan/git/OptimizationDeepLearningImageProcessing/LightFields/h5Files/"
dataset_name         = "generatedLightFields"

for filename in filenames:
	HRindex  = 0
	with imageio.get_writer(filename+"/"+filename+".gif", mode='I') as writer:
		for indx in range(-2,3):
			for indy in range(-2,3):
				cwd = os.getcwd()
				directory = cwd+"/"+filename+"/"
				if not os.path.exists(directory):
					os.makedirs(directory)
				cameraPos = [5, cameraPosOrigin[1]+indx*deltaCam,cameraPosOrigin[2]+indy*deltaCam]		
				XMLstring = createXMLstring(filename,str(scaleVal[index]),str(cameraPos[1]),str(cameraPos[2]))
				with open(directory+filename+str(indx)+str(indy)+".xml", "w") as cube_xml:
					cube_xml.write(XMLstring)
				cmd = ["mitsuba", filename+"/"+filename+str(indx)+str(indy)+".xml"]    
				cmd_out = subprocess.check_output(cmd)
				image = imageio.imread(filename+"/"+filename+str(indx)+str(indy)+".png")
				hr_image.append(np.asarray(image))
				HRindex = HRindex+1
				if indx == 0 and indy == 0: 
					lr_image.append(np.asarray(image))
				writer.append_data(image)
	print(["Completed index: "+str(index)])				
	index = index+1

create_h5(data = lr_image, label = hr_image, path = destination_path, file_name = dataset_name+"training.h5")
print("data of length ", len(lr_image), "and label of length ", len(hr_image))