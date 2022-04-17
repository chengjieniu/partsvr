%writeMesh()
%��������Ϣ�ͷָ��label��Ϣ�洢ΪOFF�ļ��Ͷ�Ӧ�ķָ���txt�ļ�
%load('ptsmeshLabel.mat');
% rootpath='H:\0retry_cv\shapenet_chair_pointcloud\02691156\off_seg';
rootpath='/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/02691156/off_seg';
for ii=1:length(ptsmeshLabel)
    ii
    name = ptsmeshLabel{ii}.name;
    off_filename = [rootpath filesep name '.off'];
    seg_filename = [rootpath filesep name '.txt'];
    fid=fopen(off_filename,'w+');
    fprintf(fid,'OFF');
    fprintf(fid,'\n');
    fprintf(fid,'%d %d %d',length(ptsmeshLabel{ii}.FV.vertices),length(ptsmeshLabel{ii}.FV.faces),0);
    fprintf(fid,'\n');
    for i=1:length(ptsmeshLabel{ii}.FV.vertices)
        fprintf(fid,'%f %f %f',ptsmeshLabel{ii}.FV.vertices(i,1),ptsmeshLabel{ii}.FV.vertices(i,2),ptsmeshLabel{ii}.FV.vertices(i,3));
        fprintf(fid,'\n');
    end

    for i=1:length(ptsmeshLabel{ii}.FV.faces)
        fprintf(fid,'%d %d %d %d',3,ptsmeshLabel{ii}.FV.faces(i,1)-1,ptsmeshLabel{ii}.FV.faces(i,2)-1,ptsmeshLabel{ii}.FV.faces(i,3)-1);
        faces(i,:)=ptsmeshLabel{ii}.FV.faces(i,:);
        fprintf(fid,'\n');
    end
    fclose(fid);

    fidt=fopen( seg_filename, 'w+');
    %back
    label_1=find(ptsmeshLabel{ii}.facelabel==1);
    if ~isempty(label_1)
        fprintf(fidt,'Chair_labelA');
        fprintf(fidt,'\n');

        for i=1:length(label_1)
            fprintf(fidt,'%d ',label_1(i));
        end
        fprintf(fidt,'\n');
        fprintf(fidt,'\n');
    end
   
    %seat
    label_2=find(ptsmeshLabel{ii}.facelabel==2);
    if ~isempty(label_2)

        fprintf(fidt,'Chair_labelB');
        fprintf(fidt,'\n');

        for i=1:length(label_2)
            fprintf(fidt,'%d ',label_2(i));
        end
        
        fprintf(fidt,'\n');
        fprintf(fidt,'\n');
    end
   
    %legs
    label_3=find(ptsmeshLabel{ii}.facelabel==3);
    if ~isempty(label_3)
        fprintf(fidt,'Chair_labelC');
        fprintf(fidt,'\n');

        for i=1:length(label_3)
            fprintf(fidt,'%d ',label_3(i));
        end
        
        fprintf(fidt,'\n');
        fprintf(fidt,'\n');
    end

   

    label_4=find(ptsmeshLabel{ii}.facelabel==4);
    if ~isempty(label_4)
        fprintf(fidt,'Chair_labelD');
        fprintf(fidt,'\n');

        for i=1:length(label_4)
            fprintf(fidt,'%d ',label_4(i));
        end
        fprintf(fidt,'\n');
        fprintf(fidt,'\n');

    end

    
    fclose(fidt);

end
