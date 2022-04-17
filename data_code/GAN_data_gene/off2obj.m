%change from off and label.txt to obj with diffrent group 
clear
%datapath = 'H:\0retry_cv\labeled_meshes\Chair';
%datapath='H:\0retry_cv\labeled_meshes\off';
datapath = 'H:\0retry_cv\shapenet_chair_pointcloud\03001627\off_seg';
folderlist = dir(datapath);
%savepath='H:\0retry_cv\labeled_meshes\Chair_obj_seg';
savepath='H:\0retry_cv\shapenet_chair_pointcloud\03001627\obj_seg';

for kk=185:2:length(folderlist)
    clear face;
    clear vertices;
    kk
    offfilename=[datapath filesep folderlist(kk,1).name];    
    segofffilename=[datapath filesep folderlist(kk+1,1).name];   
    fid=fopen(offfilename,'r');
    line=fgets(fid);
    fidseg=fopen(segofffilename,'r');
    lineseg=fgets(fidseg);
    objfilename=[savepath filesep folderlist(kk,1).name(1:end-4) '.obj'];
    fidw=fopen(objfilename,'w');
    fprintf(fidw,'%s','mtllib groupseg.mtl');
    fprintf(fidw,'\r\n');
    while ~feof(fid)
        line=fgets(fid);
        pids = sscanf(line, '%d');
        for i=1:pids(1)
            line=fgets(fid);
            fprintf(fidw,'%s ','v');
            vertices(i,:)= sscanf(line,'%f');
            fprintf(fidw,'%s',line);            
        end
        
        for j=1:pids(2)
            line=fgets(fid);
            faline=sscanf(line,'%d');
            face(j,:)=faline(2:end)+1;
        end
        
        if strcmp(lineseg(1:12),'Chair_labelA')
            lineseg=fgets(fidseg);
            pidseg=sscanf(lineseg,'%d');
            fprintf(fidw,'%s','g Chair_labelA');
            fprintf(fidw,'\r\n');
            fprintf(fidw,'%s','usemtl red');
            fprintf(fidw,'\r\n');
            for j=1:length(pidseg)
                fprintf(fidw,'%s ','f');
                fprintf(fidw,'%d ',face(pidseg(j),:));
                fprintf(fidw,'\r\n');
            end
            lineseg=fgets(fidseg);
           	if lineseg==-1
                continue;
            end
       
            lineseg=fgets(fidseg);
        end


        if strcmp(lineseg(1:12),'Chair_labelB')
            lineseg=fgets(fidseg);
            pidseg=sscanf(lineseg,'%d');
            fprintf(fidw,'%s','g Chair_labelB');
            fprintf(fidw,'\r\n');
            fprintf(fidw,'%s','usemtl yellow');
            fprintf(fidw,'\r\n');
            for j=1:length(pidseg)
                fprintf(fidw,'%s ','f');
                fprintf(fidw,'%d ',face(pidseg(j),:));
                fprintf(fidw,'\r\n'); 
            end
            lineseg=fgets(fidseg);
            if lineseg==-1
                continue;
            end
      
            lineseg=fgets(fidseg);
        end

        
        
        if strcmp(lineseg(1:12),'Chair_labelC')
            lineseg=fgets(fidseg);
            pidseg=sscanf(lineseg,'%d');                
            fprintf(fidw,'%s','g Chair_labelC');
            fprintf(fidw,'\r\n');
            fprintf(fidw,'%s','usemtl blue'); 
            fprintf(fidw,'\r\n'); 
            for j=1:length(pidseg)

                fprintf(fidw,'%s ','f');
                fprintf(fidw,'%d ',face(pidseg(j),:));
                fprintf(fidw,'\r\n'); 
            end
            lineseg=fgets(fidseg);
            if lineseg==-1
              continue;
            end
            lineseg=fgets(fidseg);            
        end

        if strcmp(lineseg(1:12),'Chair_labelD')
            lineseg=fgets(fidseg);
            pidseg=sscanf(lineseg,'%d');
            fprintf(fidw,'%s','g Chair_labelD');
            fprintf(fidw,'\r\n');
            fprintf(fidw,'%s','usemtl green');
            fprintf(fidw,'\r\n');
            for j=1:length(pidseg)
                
                fprintf(fidw,'%s ','f');
                fprintf(fidw,'%d ',face(pidseg(j),:));
                fprintf(fidw,'\r\n'); 
            end
            lineseg=fgets(fidseg);
        end      
       
       
    end
    fclose(fid);
    fclose(fidseg);
    fclose(fidw); 
    
end