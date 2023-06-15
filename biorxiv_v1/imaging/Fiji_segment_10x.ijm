// user dialogue to select directory where files are
dir = getDirectory("Choose a Directory ");
list = getFileList(dir);
for (i=0; i<list.length; i++) {
	// selects only those files that end with "_B.czi" - feel free to change those.
	if (endsWith(list[i], "_B.czi")) {
		B_name = list[i];
		tmp = split(list[i], "_");
		well = tmp[0];
		A_name = well+"_A.czi";
		open(dir+A_name);
		open(dir+B_name);
		run("Concatenate...", "  title="+well+" image1="+B_name+" image2="+A_name );
		run("Hyperstack to Stack");
		// align all channels
		run("StackReg", "transformation=[Rigid Body]");
		// duplicate first channel to create segmentation mask
		run("Duplicate...", "  channels=1");
		// segmentation of cells by thresholding and some binary operations
		// setAutoThreshold("Triangle dark");
		// setOption("BlackBackground", true);
		// run("Convert to Mask", "method=Triangle background=Dark calculate black");
		setThreshold(2000, 65535);
		setOption("BlackBackground", true);
		run("Convert to Mask");
		run("Open");
		run("Erode");
		run("Watershed");
		// select cells in segmented mask based on size/shape criteria
		run("Analyze Particles...", "size=100.00-10000.00 circularity=0.30-1.00 exclude add");
		selectWindow(well+"-1");
		run("Close");
		// back to original image to perform measurements
		selectWindow(well);
		roiManager("Select all")
		roiManager("Multi Measure");
		saveAs("Results", dir+well+".TH.csv");
		roiManager("Select all")
		roiManager("Delete")
		selectWindow(well);
		run("Close");
		selectWindow("Results"); 
	    run("Close" );
	}
}