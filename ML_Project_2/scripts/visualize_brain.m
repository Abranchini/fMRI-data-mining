load coefficients_90_lag_1.mat 
load ../data/CodeBook_90.mat
load ../data/IC_AAL.mat

% Determine where your m-file's folder is.
folder = fileparts(which(mfilename)); 
% Add that folder plus all subfolders to the path.
addpath(genpath(folder));

%% Plot coefficients matrix
AA =  squeeze(A(1,:,:));
cause_nodes={};
figure(1)
imagesc(log2(abs(AA)))
colormap(jet)
colorbar
%% Plot iCAPs matrix
figure(2)
F=zeros(20,20);
thresh = .1;
for j=1:20
    S=IC_AAL{j};
    INDS_ALL = [];
    for i =1:length(S)
        v=squeeze(A(1,:,S(i)));
        INDS = find(abs(v)>thresh); 
        INDS_ALL = [INDS_ALL INDS];
        clear INDS
    end
        for i=1:20
         F(i,j) = length(intersect(INDS_ALL,IC_AAL{i}));
        end
        cause_nodes{j} = unique(INDS_ALL);    
        clear INDS_ALL   
end
modularity = 12;
FF=F(1:13,1:13);%FF(i,j) gives for a region j the number of nodes from region i that are causal 
figure(2)
names = {'AUD ';'ATT ';'pVIS';'sVIS';'PRE ';'VISP';'MOT ';'DMN ';'EXEC';'pDMN';'ASAL';'SUB ';'ACC '};
imagesc(FF.*(FF>=modularity))
xticks([1:13])
yticks([1:13])
xticklabels(names)
yticklabels(names)
colormap(jet)
colorbar 

%% Circular plot: causality between regions
x = FF;
thresh = 10;
thres_mat = double(x > thresh).*x;
% Create custom node labels
myLabel = {'AUD ';'ATT ';'pVIS';'sVIS';'PRE ';'VISP';'MOT ';'DMN ';'EXEC';'pDMN';'ASAL';'SUB ';'ACC '};
figure;
myColorMap = lines(length(x));
circularGraph(thres_mat,'Colormap',myColorMap,'Label',myLabel);

%% Circular plot: causality between regions
x = FF;
thresh = 15;
thres_mat = double(x > thresh).*x;
% Create custom node labels
myLabel = {'AUD ';'ATT ';'pVIS';'sVIS';'PRE ';'VISP';'MOT ';'DMN ';'EXEC';'pDMN';'ASAL';'SUB ';'ACC '};
figure;
myColorMap = lines(length(x));
circularGraph(thres_mat,'Colormap',myColorMap,'Label',myLabel);

%% PLOT BRAIN
% k is the region of intersest between 1 and 13
% the code plots the nodes of this region in small and the nodes that are
% causing them in big 

k=3; % pVIS
N=90; % 90 regions
S=IC_AAL{k};
vec =zeros(N,1);
vec(cause_nodes{k})=2;
vec(S)=1;
W = zeros(N,N);
% Plot without connections
PlotBrainGraph(W,abs(vec),vec,CodeBook.full,0,max(vec),...
max(abs(vec)),2,1,'jet','jet',1,...
0.6,[]);

adj=AA';
W(S,:)=~~adj(S,:)*0.25;
W(cause_nodes{k},S)=~~adj(cause_nodes{k},S)*0.5;
W(S,S)=0;
% Plot with connections
W(intersect(S,cause_nodes{k}),intersect(S,cause_nodes{k}))=0;
PlotBrainGraph(W,abs(vec),vec,CodeBook.full,0,max(vec),...
max(abs(vec)),2,1,'jet','jet',1,...
0.6,[]);

