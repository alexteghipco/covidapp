%% Prep data
addpath(genpath(pwd)) % have consensus clustering package in your current directory....
data = h5read('abstract_embeddings.h5', '/data');

%% Reduce sample size by initial clustering
[nidx, ~] = kmeans(data', 1000, ...
    'MaxIter', 1000, ...
    'Replicates', 20, ...
    'Distance', 'cosine', ...
    'Display', 'final');
% The above did not finish, saved out as nidx.mat from within function on
% 10th replicate.
save('nidx.mat', 'nidx');

% Get centroids manually
load('nidx.mat')
data = data';
k = max(nidx); % Number of clusters
[n, p] = size(data); % n is the number of points, p is the dimensionality of each point
centroids = zeros(k, p); % Initialize centroids matrix
for i = 1:k
    cluster_points = data(nidx == i, :); % Select all points in the i-th cluster
    centroids(i, :) = mean(cluster_points, 1); % Compute mean for each cluster
end
save('nidx_centroids.mat','centroids')

%% Consensus cluster the centroids
[clusterCon,~,~,~] = consensusClustering_v4(centroids,1000,0.5,[2:2:15 20:5:50 60:10:200],'kmeans','cosine',[],[],[],'false','false');
save('clusterCon.mat','clusterCon')

of = 'D:\Science\covid\clusterEval';
mkdir(of)
[pac,~,~] = consensusSelection_v3(clusterCon,'auc','false','dist','true','cdf','true','pac','true','outDir',of,'outAppend',['embeddings']);
save('pac.mat','pac')
for i = 1:size(clusterCon,3)
    test = clusterCon(:,:,i);
    bc(i,1) = bimodalCoeff(test(:));
    [dip(i,1), dipP(i,1), ~,~] = HartigansDipSignifTest(sort(test),5000);
end
k = [2:2:15 20:5:50 60:10:200];
save('bc_dip.mat','bc','dip','k','dipP')

% get final solution 
find(dipP < 0.05) % acceptable range
kChosen=8; % chosen by manual inspection (see of folder and also the figure below)
figure; plot(bc);

addpath(genpath('C:\Users\alext\Downloads\affinityProp'))
[finalClusters, ~, ~, ~, ~] = apclusterK(squeeze(clusterCon(:,:,kChosen)), k(kChosen), 0);
save('finalClusters.mat','finalClusters')

%% Assign Original Data Points to Consensus Clusters
d = pdist2(data, centroids, 'cosine');
[~, nearest_centroid_idx] = min(d, [], 2);
save('centroidAssign1.mat','nearest_centroid_idx')

finalClustersAll = finalClusters(nearest_centroid_idx);

% Save results
save('final_consensus_cluster_labels.mat', 'finalClustersAll');

%data=data';
sampling_ratio = 0.1;
num_folds = round(1 / sampling_ratio);
c = cvpartition(finalClustersAll, 'KFold', num_folds);
id = find(c.test(1));  % You can use any fold; here, we're using the first fold
tmpdata = data(id, :);

% this takes too long...let's pca
% Y = tsne(tmpdata, 'Distance', 'cosine', 'Perplexity', 50, 'Options', statset('MaxIter', 2000));
% save('tsne.mat','Y')
% figure; gscatter(Y(:,1),Y(:,2),finalClustersAll(id))
% title('Cosine')

[~,score,~,~,~,~] = pca(tmpdata,'NumComponents',50);
Y2 = tsne(score, 'Distance', 'euclidean', 'Perplexity', 100); %, 'Options', statset('MaxIter', 2000));
save('tsne_v2.mat','Y2','id','finalClustersAll')
figure; gscatter(Y2(:,1),Y2(:,2),finalClustersAll(id))