%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% gn_clusterby_resize_pca_kmeans
% author: Gagan Narula 05/02/2016
% data_in : vector of cluster numbers containing data from flatclust
% K : number of output clusters
% resize_len : length (nmb steps) to resize to 
% freqs : vector of frequencies to consider from spectrogram max 1:128
% method: method of resizing data
% varexpl : variance to be explained with PCA number b/w 0 and 1
% methodclus : 'fuzzyc' or 'kmeans'
% plotConMat : creates output confusion matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [OUT,IDX,Centroidss,sumd,Dists,elnmbs] = gn_clusterby_resize_pca_kmeans(Flat,data_in,K,resize_len,freqs,method,varexpl,methodclus,plotConMat)

if nargin<2
    error 'Please provide the flatclust cluster number(s) of your data as a vector!';
end

if nargin<3
    error 'How many clusters (K) do you want dude? '
end

if nargin<4
    disp('Output time length from resize operation not provided! Default = 20');
    resize_len = 20;
end

if nargin<5
    disp('Special subset of frequency range not provided! Default = all');
    freqs = 1:128;
end

if nargin<6
    disp('Default nearest neighbour interpolation!')
    method = 'nearest';
end

if nargin<7
    disp('Default: 80 percent explained variance for PCA!');
    varexpl = 0.8;
end

if nargin<8
    methodclus = 'kmeans';
    disp('Default: kmeans')
end


tic
%% data prep

%data matrix
X = nan(50000,length(freqs)*resize_len);
origlabls = nan(50000,1);
elnmbs = nan(50000,1);

%some useful params
nonov = Flat.p.nonoverlap;
buffsize = Flat.p.nfft;
Nclusts = length(data_in);
%% extract normalize and resize and save data into X
l=1; nskipped = 0;
%loop through each cluster provided 
tic
for ii = 1:Nclusts
   
    %get all elements belonging to this cluster
    allels = find(Flat.X.clust_ID==data_in(ii));
    
    %loop through them and extract data
    
    for jj = allels
        
        on = 1+floor((Flat.X.indices_all(jj)-buffsize)/nonov);
        
        off = on + floor(Flat.X.data_off(jj)/nonov);
        datfil = Flat.DAT.data{Flat.X.DATindex(jj)};
        if off <= size(datfil,2)
        data_el = double(datfil(freqs,on:off));
        else
            nskipped = nskipped + 1;
            continue
        end
       
        %resize it with imresize        
        data_el = imresize(data_el,[length(freqs) resize_len],method);
        
        %find any extreme values and clip them
        stdd_f = mean(std(data_el,[],2));
        stdd_t = mean(std(data_el));
        avgg = (stdd_f + stdd_t)/2;
        extremm = data_el > 2*avgg;
        data_el(extremm) = sign(data_el(extremm))*avgg;
        
        %normalize
        if 1
            data_el = data_el/max(max(data_el));
        end
        
        if (sum(sum(isfinite(data_el))))~=(size(data_el,1)*size(data_el,2))
            nskipped = nskipped + 1;
            continue
        end
        
%        figure(11011);clf;imagesc(data_el);
%        pause
     
        X(l,:) = data_el(:);
%         if sum(isinf(X(l,:)))>0 || sum(isnan(X(l,:)))
%             keyboard
%         end
        origlabls(l) = ii;
        elnmbs(l) = jj;
        l = l+1;
        
    end       
    
    fprintf('\n ... Cluster %d of %d Data extraction, resizing and normalizing done! ... \n',ii,Nclusts);
    nskipped 
end
toc

%% randomize order
X = X(1:l-1,:);
origlabls = origlabls(1:l-1);
elnmbs = elnmbs(1:l-1);
order = randperm(l-1);

X = X(order,:);
origlabls = origlabls(order);
elnmbs = elnmbs(order);
%% Now do PCA on this motherfucka

%first center the data
M = size(X,1); %no. of samples

%D = size(X,2);
Xm = X - repmat(mean(X),M,1);
fprintf('\n ... Computing covariance matrix of %d samples ... \n',M);
tic
C = (1/(M-1))*(Xm'*Xm);
toc

fprintf('\n ... Computing eigen values and eigen vectos ... \n');
%find eigen vectors
[V,lambda] = eig(C);

%find eigen vectors that explain 80% of the variance
%sort the eigen values
ll = diag(lambda);
[ll,I] = sort(ll,'descend'); V = V(:,I); %sort eigen vector columns in that order

ll = ll/sum(ll);
cumL = cumsum(ll);
validvecs = length(find(cumL < varexpl));
fprintf('\n ... %d PCs to explain %f variance ... \n',validvecs,varexpl);
PCs = V(:,1:validvecs);


%% Project data onto PCs and do Kmeans
fprintf('\n Projecting data onto PC basis ... \n');

Xpc = PCs'*X';

%% do K-means or Fuzzy c means
fprintf('\n Performing clustering \n');

switch methodclus
    case 'kmeans'
        [IDX,Centroidss,sumd,Dists] = kmeans(Xpc',K);
    case 'fuzzyc'
        [Centroidss,U] = fcm(Xpc',K,[nan;200;nan;nan]);
        [~,IDX] = max(U); IDX = IDX';
end

%% plot confusion matrix
if plotConMat && (K==Nclusts)
    OUT = nan(Nclusts,Nclusts);
%find the output indices for each input cluster
for ii = 1:Nclusts
    for jj = 1:Nclusts
        g = IDX(origlabls==ii); %subset of all input 
        OUT(ii,jj) = sum(g == jj);
    end
    
end

fprintf('\n ... DONE ...\n');
fprintf('\n ... total %d skipped ... \n',nskipped);
toc
end
