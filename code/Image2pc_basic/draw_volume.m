function draw_volume(vol)
%figure(4);
[X,Y,Z]=ind2sub(size(vol),find(vol(:)>0.5));
plot3(X,Y,Z,['r' '.']); %hold on;
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');

end