folder = getArgument();
//name  = split(folder_name, "SPLIT")[1];
print(folder)
File.openSequence(folder,  'filter =.png');
run("3D Viewer");
call("ij3d.ImageJ3DViewer.setCoordinateSystem", "false");
call("ij3d.ImageJ3DViewer.add", "sample", "None", "sample", "0", "true", "true", "true", "2", "0");
call("ij3d.ImageJ3DViewer.record360");
selectWindow("Movie");
saveAs("Gif", folder + "Movie.gif");


run("Quit");



