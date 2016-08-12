

clusters = [6 7 8 9];
stnd_scaler = 0; %zscore ?
norm_max = 1; % normalize max
nclust = length(clusters);
nonov = Flat.p.nonoverlap;
buffersize = Flat.p.nfft; %no. of samples that make one spectrogram column 128 = 4 msec

%Frequency bins to keep
freqstart = 9;
freqend = 128; 

%% want to get whole motifs 
%To do this one needs to get a syllable and then the following 'n'
%syllables and attach them together 

%initial guess
nsylls  = length(clusters);
nmotifs = 20000;

DATA_motif = cell(nmotifs,1);
label_motif = cell(nmotifs,1);


%go through cluster one (which is the syllable 1 cluster and extract 
nsyl_1 = length(find(Flat.X.clust_ID==clusters(1)));
all_el1 = find(Flat.X.clust_ID==clusters(1));
g = 1; nskipped = 0;

fprintf('\n Looping over %d syllables in first cluster ... \n',nsyl_1);

for ii = 1:nsyl_1
    
    el = all_el1(ii);
    %find out if this syllable has next n-1 syllables following it
    cluster_elplus = nan(nsylls-1,1); 
    flagg = 1; %means valid syllable for motif extraction
    for jj = 1:nsylls-1
        cluster_elplus(jj) = Flat.X.clust_ID(el+jj);
        if cluster_elplus(jj)~= clusters(1)+jj
            flagg = 0; %means invalid syllable, so skip over this guy
            nskipped = nskipped + 1;
            break 
        end
    end
    
    if flagg
        
        %calculate onset and offset of each syllable in this motif
        
        
        FIL = Flat.X.DATindex(el);
        on1 = 1+floor((Flat.X.indices_all(el)-buffersize)/nonov);
        onend = 1+floor((Flat.X.indices_all(el+nsylls-1)-buffersize)/nonov);
        offend = onend + floor(Flat.X.data_off(el+nsylls-1)/nonov);
       
         tmpdata = double(Flat.DAT.data{FIL}(freqstart:freqend,on1:offend));
         %normalization by max
         if norm_max
            tmpdata = tmpdata/max(max(tmpdata));
         end
         
         
        %FOR labels!!!
        tmplabls = zeros(1,size(tmpdata,2));
      
        l = 1;
        %loop over the next syllables after the first one and determine
        %their onsets and offsets
        for kk = 1:nsylls
            
            %syllable onset
            on = 1+floor((Flat.X.indices_all(el+kk-1)-buffersize)/nonov);
        
            %syllable offset
            off = on + floor(Flat.X.data_off(el+kk-1)/nonov);
            %set labels
            tmplabls(l:l + off-on) = kk;
                     
            if kk ~= nsylls
                onnext = 1+floor((Flat.X.indices_all(el+kk+1-1)-buffersize)/nonov);
                gap = onnext - off;
                l = l+ off-on + gap;
            end
        end
        %tmplabls(isnan(tmplabls)) = 0; %this means gap
        
        
        %standardize? over time and frequency?
        if stnd_scaler
        tmpdata = (tmpdata - repmat(mean(tmpdata,2),1,size(tmpdata,2)))./ ...
            repmat(std(tmpdata,0,2),1,size(tmpdata,2)); %over time
        tmpdata = (tmpdata - repmat(mean(tmpdata,1),size(tmpdata,1),1))./ ...
            repmat(std(tmpdata,0,1),size(tmpdata,1),1); %over frequency
        end
        
        DATA_motif{g,1} = tmpdata;
        label_motif{g,1} = tmplabls;
        g = g + 1;
        if 0
        figure(1111);clf;set(gcf,'Position',[300 300 700 700]);        
        subplot(311);imagesc(tmpdata(end:-1:1,:));
        subplot(312);plot(tmplabls,'-ok');ylim([-1 nsylls+1]);
        subplot(313);plot(sum(tmpdata),'LineWidth',2); ylabel 'Energy';
        pause
        end
    end
end
fprintf('\n Total %d syllables skipped \n',nskipped);
%DATA_motif = DATA_motif(1:g-1,1);
%label_motif = label_motif(1:g-1,1);
