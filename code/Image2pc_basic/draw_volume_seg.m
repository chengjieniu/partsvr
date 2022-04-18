function draw_volume_seg(vol)
%figure(4);
[X,Y,Z]=ind2sub(size(vol),find(vol==1));
plot3(X,Y,Z,['r' '.']); hold on;
[X,Y,Z]=ind2sub(size(vol),find(vol==2));
plot3(X,Y,Z,['b' '.']); hold on;
[X,Y,Z]=ind2sub(size(vol),find(vol==3));
plot3(X,Y,Z,['y' '.']); hold on;
[X,Y,Z]=ind2sub(size(vol),find(vol==4));
plot3(X,Y,Z,['g' '.']); hold on;
[X,Y,Z]=ind2sub(size(vol),find(vol==5));
plot3(X,Y,Z,['m' '.']); hold on;
[X,Y,Z]=ind2sub(size(vol),find(vol==6));
plot3(X,Y,Z,['c' '.']); hold on;
[X,Y,Z]=ind2sub(size(vol),find(vol==7));
plot3(X,Y,Z,['k' '.']); hold on;
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');

end