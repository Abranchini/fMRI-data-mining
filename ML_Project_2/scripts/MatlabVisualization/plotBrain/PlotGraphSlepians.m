%Giulia - plots for Slepians
%select signal to plot = CC2
addpath('/Applications/freesurfer/matlab/');
%% Atlas + Selection
N=360; %number of regions
load 'labels_Glasser_FPN'
load 'labels_Glasser_DMN'
CC2=0.1*ones(N,1);
CC2(newlistDMN)=1;
CC2(newlistFPN)=-1;
%% 
CM=zeros(N,N);

%%CodeBook2 Craddock is the one without cerebellum
CodeBook=load('Glasser360_2mm_codebook');
CodeBook=CodeBook.codeBook;

%%CodeBook950 is the one without reordering
%CodeBook=load('/Users/gpreti/Documents/Data/HCP/Slepians_DTI_fMRI/CodeBookCraddock2mm.mat');
%CodeBook=CodeBook.CodeBook838;
 %% UW
CC2=V(:,1);
figure;plot(CC2)
%% Slepians
CC2=SLEP(:,4);
figure;plot(CC2)
%% liberal/aligned
CC2=mean_a;
CC2(find(CC2<percentile(mean_a,90)))=0;
%% adjust values for saturation (to eliminate outliers peaks)
thr=1;
CC2new=CC2;
CC2new(find(CC2>thr))=0;
CC2new(find(CC2>thr))=max(CC2new);
CC2new(find(CC2<-thr))=0;
CC2new(find(CC2<-thr))=min(CC2new);
CC2=CC2new;
figure;plot(CC2)
%% %
CC=abs(CC2);
T_conn=0;
Factor_SphereSize=max(CC);
Factor_Col=1;%/15;%max(CC);
Exp_Sphere=2;
View=1;

Colormap_edges='jet';
Colormap_nodes='jet';
Gamma=0.5;
LinearWeight=1;
CA=[-1 1];

PlotBrainGraph(CM,CC,CC2,CodeBook,T_conn,Factor_SphereSize,...
    Factor_Col,Exp_Sphere,View,Colormap_nodes,Colormap_edges,Gamma,...
    LinearWeight,CA)