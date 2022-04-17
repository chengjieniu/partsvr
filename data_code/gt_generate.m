datapath = '/home/ncj/Desktop/GMP2020/Image2pc_semantic/data/gt/downsampled/';
synth_set = '02958343'
folderlist = dir([datapath synth_set]);
name = {}
points ={}
for kk=3:length(folderlist)
   name{kk-2} = folderlist(kk).name;
   a = load([datapath synth_set filesep folderlist(kk).name]);
   points{kk-2} = a.points;
   kk
end