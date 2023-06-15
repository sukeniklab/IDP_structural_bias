masterdir = getDirectory("Choose a Directory");
print("Starting script");
print(masterdir);
masterlist=getFileList(masterdir);
method="setValues";
//method="Triangle";
//method="MinError";
for (l=0; l<masterlist.length; l++) {
	subdirname=File.getName(masterlist[l]);
	files = getFileList(masterdir+masterlist[l]);
	File.makeDirectory(masterdir+masterlist[l]+method+"_Results"); 
	saveto=masterdir+masterlist[l]+method+"_Results/";
	for (fIdx=0; fIdx<files.length; fIdx++) {
		tmp=substring(masterlist[l],0,lengthOf(masterlist[l])-1);
		filename = masterdir+tmp+"\\"+files[fIdx];
		print(tmp);
		print(filename);
		if (endsWith(filename, "_B.czi")) {
			print("splitting well name");
				well = split(files[fIdx], "_");
				open(masterdir+masterlist[l]+well[0]+"_B.czi");
				open(masterdir+masterlist[l]+well[0]+"_A.czi");
				run("Concatenate...", "  title="+subdirname+"_"+well[0]+" open image1="+masterlist[l]+well[0]+"_B.czi image2="+masterlist[l]+well[0]+"_A.czi");
				openimages=getList("image.titles");
				print("length of array for open images is "+lengthOf(openimages));
				for (o=0; o < lengthOf(openimages); o++){
					selectWindow(openimages[o]);
					imageName=getTitle();
					selectWindow(openimages[o]);
					run("Duplicate...", "title=duplicate duplicate channels=1 frames=1");
					//the different thresholds used to compare. 
					setThreshold(1500, 40000);//setValues
					//setAutoThreshold("Triangle dark no-reset"); //Triangle
					//setAutoThreshold("MaxEntropy dark no-reset"); //MinError
					run("Convert to Mask");
					run("Open");
					run("Watershed");
					//size limits in pixels but output area is in squared microns Image resoultion is 1.3 um/pixel
					run("Analyze Particles...", "size=50-Infinity pixel circularity=0.1-1.00 show=Outlines display exclude add"); 
					if (roiManager("count") >= 1) {
						selectWindow("Drawing of duplicate");
						close();
						selectWindow("duplicate");
						close();
						selectWindow(openimages[o]);
						print("aligning channels");
						run("Hyperstack to Stack");
						run("StackReg", "transformation=[Rigid Body]");
						run("Stack to Hyperstack...", "order=xyczt(default) channels=4 slices=1 frames=2 display=Color"); //to align B&A
						print("measuring selected rois");
						roiManager("Select All");
						roiManager("multi-measure measure_all one append"); //appending results instead of saving as individual files per well
						roiManager("Select All");
						roiManager("Save", saveto+well[0]+"_"+method+"_RoiSet.zip");
						roiManager("delete");
						close(imageName);
						print("closing "+imageName);
						print("end 1");
					}
					else{
						selectImage("duplicate");
						close();
						selectImage("Drawing of duplicate");
						close();
						selectImage(openimages[o]);
						close();
						close(imageName);
						print("end 2");
						}}}}
		saveAs("Results",saveto+subdirname+"_"+method+"_Results.csv");
		print("Done running script! :D");
		run("Clear Results");
		close("Results");
		print("moving to next subdir");
	};
print("Done with the entire analysis!")