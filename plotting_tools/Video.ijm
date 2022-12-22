setBatchMode(true);


parent_folder = getArgument();
run("Image Sequence...", "open=" parent_folder + "0.png sort");
run("3D Viewer");
call("ij3d.ImageJ3DViewer.setCoordinateSystem", "false");
call("ij3d.ImageJ3DViewer.add", "tiff_stack", "None", "tiff_stack", "0", "true", "true", "true", "2", "0");
call("ij3d.ImageJ3DViewer.record360");
selectWindow("Movie");
saveAs("Gif", "full/halfMovie" + "images_part0" + part + "_bins_" + n_bins + "/video.gif");
call("ij3d.ImageJ3DViewer.close");
close("*");

run("Quit");
