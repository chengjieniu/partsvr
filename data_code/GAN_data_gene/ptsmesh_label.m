%��ȡshapeNet�б�ǵĵ��ƣ����洢��.mat��
%03001627���� 04379243 ����  02691156 �ɻ�  03790512 Ħ�г�
% filelist=dir('H:\0retry_cv\shapenet_chair_pointcloud\03001627\03001627_3746');
% rootpath='H:\0retry_cv\shapenet_chair_pointcloud\03001627\03001627_3746';
filelist=dir('/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/02691156/02691156');
rootpath='/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/02691156/02691156';
%load('p2volume');
fileNum = length(filelist);

pointindex=1;
for ii = 1:fileNum   
    if strcmp(filelist(ii).name,p2volume{pointindex}.name(1:end-4))     
        ii
         %  ��ȡobj�ļ��ĵ�����Ϣ
        [vertices,faces]=obj__read([rootpath filesep filelist(ii).name filesep 'model.obj']);            
                
        FV.vertices=vertices';
        FV.faces=faces';
        ptsmeshLabel{pointindex}.FV=FV;
        ptsmeshLabel{pointindex}.name=filelist(ii).name;
        
         % ��obj�ļ���һ��:��ȷ��������Բ�ģ�shapenetģ���Լ�����ã����ڽ��г�������š�
        MULT1 = (max(vertices(1,:)) - min(vertices(1,:)));
        MULT2 = (max(vertices(2,:)) - min(vertices(2,:)));
        MULT3 = (max(vertices(3,:)) - min(vertices(3,:)));
        MULT = max(max(MULT1,MULT2), MULT3);
        vertices(1,:) =  vertices(1,:) / MULT;
        vertices(2,:) =  vertices(2,:) / MULT;
        vertices(3,:) =  vertices(3,:) / MULT;
        cenvertices = vertices';
        
        for n=1:length(FV.faces)
             centralpointofface(n,:)=(cenvertices(FV.faces(n,1),:)+cenvertices(FV.faces(n,2),:)+cenvertices(FV.faces(n,3),:))/3;
         %    centralpointofface(n,:)=FV.vertices(FV.faces(n,1),:);
        end
        ptsmeshLabel{pointindex}.cenPointofface=centralpointofface;
        
        pvertices=p2volume{pointindex}.ptc';
        MULT1 = (max(pvertices(1,:)) - min(pvertices(1,:)));
        MULT2 = (max(pvertices(2,:)) - min(pvertices(2,:)));
        MULT3 = (max(pvertices(3,:)) - min(pvertices(3,:)));
        MULT = max(max(MULT1,MULT2), MULT3);
        pvertices(1,:) =  pvertices(1,:) / MULT;
        pvertices(2,:) =  pvertices(2,:) / MULT;
        pvertices(3,:) =  pvertices(3,:) / MULT;
        


        %k = dsearchn(X,T,XI) Ϊ XI �е�ÿ���㷵�� X ����������� k��X �Ǳ�ʾ�� n ά�ռ��о��� m ����� m��n ����
        %XI �Ǳ�ʾ�� n ά�ռ��о��� p ����� p��n ����T �� numt��n+1 ���󣬼� delaunayn ��ɵ���� X ������ʷ֡���� k �ǳ���Ϊ p ����������
        %[k, d] = dsearchn(pvertices', ptsmeshLabel{pointindex}.cenPointofface);
        [k, d] = knnsearch(pvertices', ptsmeshLabel{pointindex}.cenPointofface);
        for j=1:length(k)        
             ptsmeshLabel{pointindex}.facelabel(j) = p2volume{pointindex}.label(k(j));
        end       
        pointindex = pointindex + 1;   
        
        clear pvertices;
        clear FV;
        clear vertices;
        clear cenvertices;
        clear faces; 
        clear centralpointofface;
    end
end    