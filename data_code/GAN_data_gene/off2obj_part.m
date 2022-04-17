%change from off and label.txt to obj with diffrent group 
clear
%datapath = 'H:\0retry_cv\labeled_meshes\Chair';
%datapath='H:\0retry_cv\labeled_meshes\off';
datapath = 'H:\0retry_cv\shapenet_chair_pointcloud\03001627\off_seg';
folderlist = dir(datapath);
%savepath='H:\0retry_cv\labeled_meshes\Chair_obj_seg';
savepath='H:\0retry_cv\shapenet_chair_pointcloud\03001627\obj_seg_part';

for kk=3:2:length(folderlist)
    clear face;
    clear vertices;
    kk
    offfilename=[datapath filesep folderlist(kk,1).name];    
    segofffilename=[datapath filesep folderlist(kk+1,1).name];   
    fid=fopen(offfilename,'r');
    line=fgets(fid);
    fidseg=fopen(segofffilename,'r');
    lineseg=fgets(fidseg);
    objfilename_part1=[savepath filesep folderlist(kk,1).name(1:end-4) '_1.obj'];
    objfilename_part2=[savepath filesep folderlist(kk,1).name(1:end-4) '_2.obj'];
    objfilename_part3=[savepath filesep folderlist(kk,1).name(1:end-4) '_3.obj'];
    objfilename_part4=[savepath filesep folderlist(kk,1).name(1:end-4) '_4.obj'];
    fidw1=fopen(objfilename_part1,'w');
    fidw2=fopen(objfilename_part2,'w');
    fidw3=fopen(objfilename_part3,'w');
    fidw4=fopen(objfilename_part4,'w');
    fprintf(fidw1,'%s','# part1');
    fprintf(fidw1,'\r\n');
    fprintf(fidw2,'%s','# part2');
    fprintf(fidw2,'\r\n');
    fprintf(fidw3,'%s','# part3');
    fprintf(fidw3,'\r\n');
    fprintf(fidw4,'%s','# part4');
    fprintf(fidw4,'\r\n');
    while ~feof(fid)
        line=fgets(fid);
        pids = sscanf(line, '%d');
        for i=1:pids(1)
            line=fgets(fid);
            fprintf(fidw1,'%s ','v');
            fprintf(fidw2,'%s ','v');
            fprintf(fidw3,'%s ','v');
            fprintf(fidw4,'%s ','v');
            vertices(i,:)= sscanf(line,'%f');
            fprintf(fidw1,'%s',line);   
            fprintf(fidw2,'%s',line);  
            fprintf(fidw3,'%s',line);
            fprintf(fidw4,'%s',line);
        end
        
        for j=1:pids(2)
            line=fgets(fid);
            faline=sscanf(line,'%d');
            face(j,:)=faline(2:end)+1;
        end
        
        if strcmp(lineseg(1:12),'Chair_labelA')
            lineseg=fgets(fidseg);
            pidseg=sscanf(lineseg,'%d');
%             fprintf(fidw1,'%s','# g Chair_labelA');
%             fprintf(fidw1,'\r\n');
%             fprintf(fidw1,'%s','# usemtl red');
%             fprintf(fidw1,'\r\n');
            for j=1:length(pidseg)
                fprintf(fidw1,'%s ','f');
                fprintf(fidw1,'%d ',face(pidseg(j),:));
                fprintf(fidw1,'\r\n');
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
%             fprintf(fidw2,'%s','g Chair_labelB');
%             fprintf(fidw2,'\r\n');
%             fprintf(fidw2,'%s','usemtl yellow');
%             fprintf(fidw2,'\r\n');
            for j=1:length(pidseg)
                fprintf(fidw2,'%s ','f');
                fprintf(fidw2,'%d ',face(pidseg(j),:));
                fprintf(fidw2,'\r\n'); 
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
%             fprintf(fidw3,'%s','g Chair_labelC');
%             fprintf(fidw3,'\r\n');
%             fprintf(fidw3,'%s','usemtl blue'); 
%             fprintf(fidw3,'\r\n'); 
            for j=1:length(pidseg)

                fprintf(fidw3,'%s ','f');
                fprintf(fidw3,'%d ',face(pidseg(j),:));
                fprintf(fidw3,'\r\n'); 
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
%             fprintf(fidw4,'%s','g Chair_labelD');
%             fprintf(fidw4,'\r\n');
%             fprintf(fidw4,'%s','usemtl green');
%             fprintf(fidw4,'\r\n');
            for j=1:length(pidseg)
                
                fprintf(fidw4,'%s ','f');
                fprintf(fidw4,'%d ',face(pidseg(j),:));
                fprintf(fidw4,'\r\n'); 
            end
            lineseg=fgets(fidseg);
        end      
       
       
    end
    fclose(fid);
    fclose(fidseg);
    fclose(fidw1); 
    fclose(fidw2); 
    fclose(fidw3); 
    fclose(fidw4); 
    
end