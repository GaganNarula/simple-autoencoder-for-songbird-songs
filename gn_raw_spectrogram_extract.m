

clusters = [2 5 6 7 8];

nclust = length(clusters);
nonov = Flat.p.nonoverlap;
buffersize = Flat.p.nfft; %no. of samples that make one spectrogram column 128 = 4 msec
DATA = cell(nclust,1);

%Frequency bins to keep
freqstart = 9;
freqend = 128; 
for ii = 1:nclust 
    
    nsylls = length(find(Flat.X.clust_ID==clusters(ii)));
    allelements = find(Flat.X.clust_ID==clusters(ii)); %all elements belonging to this cluster
    
    DATA{ii,1} = cell(nsylls,1);
    
    %for all syllables in current cluster, find onsets and offsets
    for jj = 1:nsylls
        
        %this syllable is contained in file FIL
        FIL = Flat.X.DATindex(allelements(jj));
        
        %syllable onset
        on = floor((Flat.X.indices_all(allelements(jj))-buffersize)/nonov)+1;
        
        %syllable offset
        off = on + floor(Flat.X.data_off(allelements(jj))/nonov);
        
        data = Flat.DAT.data{FIL}(freqstart:freqend,on:off);
        
        DATA{ii,1}{jj,1} = double(data);
    end
end
