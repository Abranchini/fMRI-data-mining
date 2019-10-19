cent=load('nodesCenters.mat');
centers=cent.centers;
k=20;
[idx, clusters] = kmeans(centers,k);
figure;
cmap = hsv(k);
for i=1:k
    scatter3(centers(idx==i,1),centers(idx==i,2),centers(idx==i,3),'MarkerEdgeColor', cmap(i,:),'MarkerFaceColor',cmap(i,:));
    scatter3(clusters(i,1),clusters(i,2),clusters(i,3),'*', 'MarkerEdgeColor', cmap(i,:));
    hold on;
end
