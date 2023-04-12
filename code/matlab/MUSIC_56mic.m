
%%
%The code is for the 3-NSM with a prototype  array of 56 microphones
%%
clc
clear all
close all
%%
%parameter define
mat_path = 'G:\CGT\Matrix_completion\matlab\code\ADMM\ADMM_56_3\CSM\simulation\56mic_3_-09d\25';
dirOutput = dir(fullfile(mat_path,'*.mat'));
file_num=size(dirOutput,1);

fig=0;
load('56_spiral_array.mat');
array = array';

c=340;                            % The speed of the sound in the medium

FS=16384;
dt=1/FS;

Loc_M_x=array(1,:);
Loc_M_y=array(2,:);
for m0=-0.9:-0.9:-2
    loc(1,:)=array(1,:)+m0;
    loc(2,:)=array(2,:); 
    Loc_M_x=[Loc_M_x,loc(1,:)];
    Loc_M_y=[Loc_M_y,loc(2,:)];
end

Loc(3,:)=0;  % microphone location
Loc_M_z=Loc(3,:);  % microphone location
%----------------------
mic_x=Loc_M_x;
mic_y=Loc_M_y;
mic_z=Loc_M_z;


coordinate_MUSIC_x = [];
coordinate_MUSIC_y = [];
loc_S_all = [];
N_mic=56*3;
M =N_mic;
zm=0.6;
% design the mesh


[x,y]=meshgrid(0:-0.03:-2,0:0.03:2);                     % The coordinates of the scanning points in the filed
[ny,nx]=size(x);

%% load data
localization_error=0;
for file_index= 79:79
    file_index
    inR_evAD_guitong_A=zeros(ny,nx);
     for frequency_analysis =5600:100:5600
        frequency_analysis
         data_name =[num2str(frequency_analysis),'_',num2str(25),'_',num2str(file_index),'.mat']

%         data_name = dirOutput(file_index).name;
        load(([mat_path filesep data_name]));
        str_filename = strsplit(data_name,'_');
        f0 =str2num(char(str_filename(1)))
        kk=2*pi*f0/c; 
        
        %%
        [V,D]=eig((date_nonchron_ADMM)); 
        [V,S] = schur(squeeze(date_nonchron_ADMM(:,:)));

        EVA = diag(D)';                                              %  find diagonal elements
        [EVA,I] = sort(EVA);                         
        V = fliplr(V(:,I));                                          %  arrange eigenvectors from large to small
        Num_source=1;                                                              % The number of sources is predefined as 1.
        %--------------------------MUSIC
        beta_MUSIC=zeros(size(x));
        Vn=V(:,Num_source+1:M);   
        inR_ev=Vn*Vn';
        for k=1:nx
            for n=1:ny        
                w=zeros(1,M);
                for j=1:M
                    r(j)=sqrt((x(1,k)-mic_x(j)).^2+(y(n,1)-mic_y(j)).^2+(zm-mic_z)^2);
                    w(j)=exp((1i)*kk*r(j))/M;
                end
                beta_MUSIC(n,k)=w*inR_ev*w';
            end
        end

        inR_evAD_guitong_A = inR_evAD_guitong_A+beta_MUSIC;

        surf(x,y,abs(beta_MUSIC),'EdgeColor','none');  
        xlabel('X(m)')
        ylabel('Y(m)')
        title(num2str(f0),'FontSize',10);
        view([0,0,1]);
        colormap(jet)
        colorbar
     
     end

    
end


