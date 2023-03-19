%%

%%
clc
clear all
close all
fig=0;
     
load('56_spiral_array.mat') % positions of all elements in array
array=array';
array(2,:) = array(2,:);
Times=-0.9; % shift interval
c=340; % sound speed
%---------------------------------------------------------------------
mi=0;
M=56; % array element number
N=M;
zs=0.6; % the interval between array plane and source plane
dx=0.03;
dy=0.03;
xi=0;
yi=0;
xa=-2;
ya=2;
%%
% path of the wav file.
filepath = 'G:\CGT\Matrix_completion\matlab\data\premix_data\test';

dirOutput = dir(fullfile(filepath,'*.wav'));
m=size(dirOutput,1);

Fs = 16000;

for sound_source_index=79:79
    sound_source_index
    
    Loc_S = 1.3*rand(2,1)+0.1;
    Loc_S(1,:) = -Loc_S(1,:);


    date_nonchron_corr =[];
    date_synchron_corr =[];
    xs1=Loc_S(1,1);
    ys1=Loc_S(2,1);        % source location

    Loc_S_x = Loc_S(1,:);
    Loc_S_y = Loc_S(2,:);
    

    ratio1=1;ratio2=1;ratio3=1;
   
    dirOutput = dir(fullfile(filepath,'*.wav'));
    iiii=0;

%     sound_source_111=wgn(Fs*10,1,1);

    [sound_source_111,Fs]=audioread([filepath filesep dirOutput(sound_source_index).name]);
%     [sound_source_222,Fs]=audioread([filepath filesep dirOutput(sound_source_index*3+1).name]);
%     [sound_source_333,Fs]=audioread([filepath filesep dirOutput(sound_source_index*3+2).name]);
    FS = Fs;
    sound_source_111 =  sound_source_111/max(sound_source_111);
%     sound_source_222 =  sound_source_222/max(sound_source_222);
%     sound_source_333 =  sound_source_333/max(sound_source_333);
    for ii = 1:length(sound_source_111)
        if(sound_source_111(ii))>0.03
            if(iiii==0)
                amin = ii;
            end
          iiii=1;
        end
    end
    iiii=0;
    for ii = length(sound_source_111):-1:1
        if(sound_source_111(ii))>0.03
            if(iiii==0)
                amax = ii;
            end
          iiii=1;
        end
    end
    sig1 = ratio1 * sound_source_111(amin:amax,1);
    

    min_length=length(sig1(:,1));
    %-----------------------------------------------------------------------------
    Prv=[];
    TA=zeros(56*3,56*3);
    Sig1=sig1';
%     Sig2=sig2';
%     Sig3=sig3';

     
        %% coordinates

        Loc_M_x=array(1,:);
        Loc_M_y=array(2,:);
        for m0=-0.9:-0.9:-2
            loc(1,:)=array(1,:)+m0;
            loc(2,:)=array(2,:); 
            Loc_M_x=[Loc_M_x,loc(1,:)];
            Loc_M_y=[Loc_M_y,loc(2,:)];
        end
        Loc_M_z=zeros(size(Loc_M_x));
        
         len1=length(Sig1);
         dt =1/FS;
         [prv_168]=signal_received(168,xs1,ys1,zs,Loc_M_x,Loc_M_y,Loc_M_z,Sig1,c,dt,len1);
        

    for snr = 25:5:25
        for frequency_point=1200:1100:6000
            frequency_point
    %%
            prv_syn=prv_168;
            if snr ~=25
                [prv_syn,kpkp]= mix_noise(prv_syn,snr);
            end
            prv_syn=prv_syn';
            f0 = frequency_point;

            spectrum_DFT =fft(prv_syn,FS,2);
            date_synchron_corr_DFT=spectrum_DFT(:,frequency_point+1)*spectrum_DFT(:,frequency_point+1)';

            for m0=mi:Times:-1.8

                %-----------------------------------
                Loc(1,:)=array(1,:)+m0;
                Loc(2,:)=array(2,:); 
                Loc(3,:)=0;
                Loc_M_x=Loc(1,:);
                Loc_M_y=Loc(2,:);
                Loc_M_z=Loc(3,:); % microphone location
                %----------S1-------------------------------------------------------
                rand_length = round((0.3*rand(1)+0.7)*min_length);
                if rand_length<Fs
                    rand_length=Fs;
                end
                Sig = Sig1(1,1:rand_length);
                len1=length(Sig);
                [prv1]=signal_received56(M,xs1,ys1,zs,Loc_M_x,Loc_M_y,Loc_M_z,Sig,c,dt,len1);

                prv=prv1;

                if snr ~=25
                    [prv,kpkp]= mix_noise(prv,snr);
                end
                prv=prv';

                f0 = frequency_point;
                spectrum_DFT =fft(prv,FS,2);
                Rall=spectrum_DFT(:,frequency_point+1)*spectrum_DFT(:,frequency_point+1)';

                TA(((round(m0/Times))*M+1):((round(m0/Times)+1)*M),((round(m0/Times))*M+1):((round(m0/Times)+1)*M),:)=Rall;
            end
            date_nonchron_corr =TA;
            %%

            % save mat files
            save_path=['..\..\data\test_data\single\56mic_3_-09d_100Hz_100Hz_6000Hz\',num2str(snr)];
            mkdir(save_path)
            save(fullfile(save_path,[num2str(frequency_point),'_',num2str(snr),'_',num2str(sound_source_index), '.mat']), 'date_nonchron_corr','date_synchron_corr','Loc_S');

            fprintf('saving is ok\n')
        end

   end

end
