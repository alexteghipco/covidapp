%% For evaluation
baseDir = '/Volumes/HICKOK-LAB/inSubs/kR';
mkdir(baseDir)
of = [baseDir '/eval3'];
mkdir(of)
clear itemTracker clusterTracker
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(1).name,'_');
    fStop(i,1) = str2double(f(i).name(tmp(end)+1:end-4));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

% now load each consensus matrix and go through them one-by-one
for i = 1:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterConsensus'); % load in data
   if i == 1 % get endpoints
       fStartS = 71;
   else
       fStartS = fStopS(i-1)+1;
   end
   
   if sum(sum(isinf(tmp.clusterConsensus{fStartS}(:,:,2)))) > 0
       tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker');
       for j = fStartS:fStopS(i) % loop over subjects
           for k = 1:size(tmp.clusterTracker{j},3) % look over k
               tmp.clusterConsensus{j}(:,:,k) = tmp.clusterTracker{j}(:,:,k)./tmp.itemTracker{j}(:,:,1);
           end
       end
   end
       
   for j = fStartS:fStopS(i) % loop over subjects in this subset
       disp(['Working on subject -- ' 
           num2str(j)])
       % for each consensus matrix, run pac, cdf, deltaK curves
       [kRPac(:,j),~,~] = consensusSelection_v3(tmp.clusterConsensus{j},'auc','false','dist','true','cdf','true','pac','true','outDir',of,'outAppend',['sub_' num2str(j)]);
       close all
       for k = 1:size(tmp.clusterConsensus{j},3) % now loop over each k in consensus matrix of this subject
           test = tmp.clusterConsensus{j}(:,:,k);
           kR_bc(k,j) = bimodalCoeff(test(:));
           [kR_dip(k,j), kR_dip_p(k,j), ~,~] = HartigansDipSignifTest(sort(test),3000);
           %[kR_dip(k,j),~,~, ~, ~, ~, ~, ~] = HartigansDipTest(sort(test));
       end
   end
   clear tmp
end


%% load everything in 
load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/conStats_kR_godzilla.mat')
test = load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/PAC_BC_DIP_curves_all_v2.mat','KR_bc','KR_dip','KR_dip_p','kRPac');
kRPac(:,1:30) = flipud(test.kRPac(:,1:30));
kR_dip(:,1:30) = (test.KR_dip(:,1:30));
kR_dip_p(:,1:30) = (test.KR_dip_p(:,1:30));
kR_bc(:,1:30) = (test.KR_bc(:,1:30));

clear ans test tmp1
save('kR_allSubs.mat')

% figure
% plot(kR_dip(:,31));
% hold on
% plot(test.KR_dip(:,31));
% 
% figure
% plot(kR_dip_p(:,31));
% hold on
% plot(test.KR_dip_p(:,31));
% 
% figure
% plot(kR_bc(:,31));
% hold on
% plot(test.KR_bc(:,31));


clear all
subs1 = load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/conStats_SCEuc_godzilla.mat');
SCEucPac = subs1.SCEucPac;
SCEuc_bc = subs1.SCEuc_bc;
SCEuc_dip = subs1.SCEuc_dip;
SCEuc_dip_p = subs1.SCEuc_dip_p;

subs2 = load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/SCEuc_conMetrics_mac.mat');
SCEucPac(:,69:77) = subs2.SCEucPac(:,69:77);
SCEuc_bc(:,69:77) = subs2.SCEuc_bc(:,69:77);
SCEuc_dip(:,69:77) = subs2.SCEuc_dip(:,69:77);
SCEuc_dip_p(:,69:77) = subs2.SCEuc_dip_p(:,69:77);

SCEucPac(:,80:105) = subs2.SCEucPac(:,80:105);
SCEuc_bc(:,80:105) = subs2.SCEuc_bc(:,80:105);
SCEuc_dip(:,80:105) = subs2.SCEuc_dip(:,80:105);
SCEuc_dip_p(:,80:105) = subs2.SCEuc_dip_p(:,80:105);

subs3 = load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/SCEuc_conMetrics_macbook.mat');

SCEucPac(:,41:70) = subs3.SCEucPac(:,41:70);
SCEuc_bc(:,41:70) = subs3.SCEuc_bc(:,41:70);
SCEuc_dip(:,41:70) = subs3.SCEuc_dip(:,41:70);
SCEuc_dip_p(:,41:70) = subs3.SCEuc_dip_p(:,41:70);

SCEucPac(:,132:137) = subs3.SCEucPac(:,132:137);
SCEuc_bc(:,132:137) = subs3.SCEuc_bc(:,132:137);
SCEuc_dip(:,132:137) = subs3.SCEuc_dip(:,132:137);
SCEuc_dip_p(:,132:137) = subs3.SCEuc_dip_p(:,132:137);

subs4 = load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/PAC_BC_DIP_curves_all_v2.mat','SCEuc_bc','SCEuc_dip','SCEuc_dip_p','SCEucPac');

SCEucPac(:,1:40) = flipud(subs4.SCEucPac(:,1:40));
SCEuc_bc(:,1:40) = subs4.SCEuc_bc(:,1:40);
SCEuc_dip(:,1:40) = subs4.SCEuc_dip(:,1:40);
SCEuc_dip_p(:,1:40) = subs4.SCEuc_dip_p(:,1:40);

clear subs1 subs2 subs3 subs4
save('SCEuc_allSubs.mat')

% missing 78 - 79
load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/SCEuc_allSubs.mat')
tmp = load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/conStats_SCEuc_straggler.mat');
SCEuc_bc(:,78:79) = tmp.SCEuc_bc(:,78:79);
SCEuc_dip(:,78:79) = tmp.SCEuc_dip(:,78:79);
SCEuc_dip_p(:,78:79) = tmp.SCEuc_dip_p(:,78:79);
SCEucPac(:,78:79) = tmp.SCEucPac(:,78:79);

clear tmp
save('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/SCEuc_allSubs.mat')


%% Step 1) Fuse kR and SCEuc matrices at each k 
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2.mat','clusterTracker','itemTracker');
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_sceuc_knn_200_10.mat','clusterTracker','itemTracker');

for i = 1:10
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/10.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear clusterConsensus clusterTracker itemTracker tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_sceuc_knn_200_20.mat','clusterTracker','itemTracker');
for i = 11:20
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/20.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear clusterConsensus clusterTracker itemTracker tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_sceuc_knn_200_30.mat','clusterTracker','itemTracker');
for i = 21:30
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/30.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear clusterConsensus clusterTracker itemTracker tmp2
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_40.mat','clusterTracker','itemTracker');
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_sceuc_knn_200_40.mat','clusterTracker','itemTracker');
for i = 31:40
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/40.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear clusterConsensus clusterTracker itemTracker tmp2 tmp1
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_50.mat','clusterTracker','itemTracker');
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_41.mat','clusterTracker','itemTracker');
for i = 41
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_44.mat','clusterTracker','itemTracker');
for i = 42:44
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_47.mat','clusterTracker','itemTracker');
for i = 45:47
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_50.mat','clusterTracker','itemTracker');
for i = 48:50
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/50.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear clusterConsensus clusterTracker itemTracker tmp2 tmp1
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_60.mat','clusterTracker','itemTracker');
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_53.mat','clusterTracker','itemTracker');
for i = 51:53
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_56.mat','clusterTracker','itemTracker');
for i = 54:56
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_59.mat','clusterTracker','itemTracker');
for i = 57:59
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_62.mat','clusterTracker','itemTracker');
for i = 60
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/60.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 clusterConsensus clusterTracker itemTracker
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_70.mat','clusterTracker','itemTracker');
for i = 61:62
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_65.mat','clusterTracker','itemTracker');
for i = 63:65
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_68.mat','clusterTracker','itemTracker');
for i = 66:68
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_71.mat','clusterTracker','itemTracker');
for i = 69:70
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/70.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear clusterConsensus clusterTracker itemTracker tmp1
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_80.mat','clusterTracker','itemTracker');
for i = 71
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_74.mat','clusterTracker','itemTracker');
for i = 72:74
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_77.mat','clusterTracker','itemTracker');
for i = 75:77
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_79.mat','clusterTracker','itemTracker');
for i = 78:79
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_80.mat','clusterTracker','itemTracker');
for i = 80
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/80.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 tmp2 clusterConsensus clusterTracker itemTracker
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_90.mat','clusterTracker','itemTracker');
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_83.mat','clusterTracker','itemTracker');
for i = 81:83
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_86.mat','clusterTracker','itemTracker');
for i = 84:86
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_89.mat','clusterTracker','itemTracker');
for i = 87:89
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_92.mat','clusterTracker','itemTracker');
for i = 90
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/90.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 clusterConsensus clusterTracker itemTracker
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_100.mat','clusterTracker','itemTracker');
%tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_83.mat','clusterTracker','itemTracker');
for i = 91:92
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_95.mat','clusterTracker','itemTracker');
for i = 93:95
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_98.mat','clusterTracker','itemTracker');
for i = 96:98
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_101.mat','clusterTracker','itemTracker');
for i = 99:100
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/100.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 clusterConsensus clusterTracker itemTracker
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_110.mat','clusterTracker','itemTracker');
for i = 101
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_104.mat','clusterTracker','itemTracker');
for i = 102:104
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_105.mat','clusterTracker','itemTracker');
for i = 105
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_107.mat','clusterTracker','itemTracker');
for i = 106:107
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_110.mat','clusterTracker','itemTracker');
for i = 108:110
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/110.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 tmp2 itemTracker clusterTracker clusterConsensus
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_120.mat','clusterTracker','itemTracker');
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_113.mat','clusterTracker','itemTracker');
for i = 111:113
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_116.mat','clusterTracker','itemTracker');
for i = 114:116
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_119.mat','clusterTracker','itemTracker');
for i = 117:119
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_122.mat','clusterTracker','itemTracker');
for i = 120
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/120.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 clusterConsensus clusterTracker itemTracker
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_130.mat','clusterTracker','itemTracker');
for i = 121:122
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_130.mat','clusterTracker','itemTracker');
for i = 121:122
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_125.mat','clusterTracker','itemTracker');
for i = 123:125
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_128.mat','clusterTracker','itemTracker');
for i = 126:128
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_131.mat','clusterTracker','itemTracker');
for i = 129:130
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/130.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

clear tmp1 clusterConsensus clusterTracker itemTracker
tmp1 = load('/Volumes/HICKOK-LAB/inSubs/kR/WithinSubPerms_v2_137.mat','clusterTracker','itemTracker');
for i = 131
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_134.mat','clusterTracker','itemTracker');
for i = 132:134
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end

clear tmp2
tmp2 = load('/Volumes/HICKOK-LAB/inSubs/SCEuc/WithinSubPerms_v2_SC_euclidean_137.mat','clusterTracker','itemTracker');
for i = 135:137
    itemTracker{i} = tmp2.itemTracker{i}(:,:,1) + tmp1.itemTracker{i}(:,:,1);
    for j = 1:29
        clusterTracker{i}(:,:,j) = tmp2.clusterTracker{i}(:,:,j) + tmp1.clusterTracker{i}(:,:,j);
        clusterConsensus{i}(:,:,j) = clusterTracker{i}(:,:,j) ./ itemTracker{i};
    end
end
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/137.mat','clusterTracker','clusterConsensus','itemTracker','-v7.3')

%% now generate pac etc for fused matrices
%% For evaluation
baseDir = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR';
mkdir(baseDir)
of = [baseDir '/eval3'];
mkdir(of)
clear itemTracker clusterTracker
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(1).name,'.mat');
    fStop(i,1) = str2double(f(i).name(1:tmp(end)-1));
    fName{i} = f(i).name;
end
fStop = [10 100 110 120 130 137 20 30 40 50 60 70 80 90];

[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

% now load each consensus matrix and go through them one-by-one
for i = 2:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterConsensus'); % load in data
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
   
   for j = fStartS:fStopS(i) % loop over subjects in this subset
       disp(['Working on subject -- ' num2str(j)])
       % for each consensus matrix, run pac, cdf, deltaK curves
       [fusedPac(:,j),~,~] = consensusSelection_v3(tmp.clusterConsensus{j},'auc','false','dist','true','cdf','true','pac','true','outDir',of,'outAppend',['sub_' num2str(j)]);
       close all
       for k = 1:size(tmp.clusterConsensus{j},3) % now loop over each k in consensus matrix of this subject
           test = tmp.clusterConsensus{j}(:,:,k);
           fused_bc(k,j) = bimodalCoeff(test(:));
           [fused_dip(k,j), fused_dip_p(k,j), ~,~] = HartigansDipSignifTest(sort(test),3000);
           %[kR_dip(k,j),~,~, ~, ~, ~, ~, ~] = HartigansDipTest(sort(test));
       end
   end
   clear tmp
end

%% Step 3) Generate BC x pac and dip x pac for fused metrics
% Kmeds R
oF = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/eval3';
for i = 109:size(fused_bc,2)
    % Make BC fig
    try
        f = figure('Renderer', 'painters', 'Position', [10 10 1200 900]);
        p = plot(kRange,fused_bc(:,i),'LineStyle','--','Marker','o','Color',[0.2 0.2 0.2],'MarkerFaceColor',[1 1 1],'LineWidth',3,'MarkerSize',10);
        set(gcf,'color','w')
        set(gca,'FontSize',16)
        set(gca,'linewidth',3)
        box off
        set(gca,'TickLength',[0 0])
        
        title(['Binomial coefficient -- subject ' num2str(i)],'fontsize',22)
        ylabel('BC','fontsize',22)
        xlabel('k','fontsize',22)

        saveas(gcf,[oF '/BCsub_' num2str(i) '.fig'])
        saveas(gcf,[oF '/BCsub_' num2str(i) '.pdf'])
        close all
    catch
        
    end
    
    % Make Dip fig
    try
        f = figure('Renderer', 'painters', 'Position', [10 10 1200 900]);
        p = plot(kRange,fused_dip(:,i),'LineStyle','--','Marker','o','Color',[0.2 0.2 0.2],'MarkerFaceColor',[1 1 1],'LineWidth',3,'MarkerSize',10);
        set(gcf,'color','w')
        set(gca,'FontSize',16)
        set(gca,'linewidth',3)
        box off
        set(gca,'TickLength',[0 0])
        
        title(['Dip statistic -- subject ' num2str(i)],'fontsize',22)
        ylabel('DIP','fontsize',22)
        xlabel('k','fontsize',22)
 
        saveas(gcf,[oF '/Dipsub_' num2str(i) '.fig'])
        saveas(gcf,[oF '/Dipsub_' num2str(i) '.pdf'])
        
        close all
    catch
        close all
    end
    close all
    
    % Make PAC x Dip fig
    try
        f1 = [oF '/Dipsub_' num2str(i) '.fig'];
        f2 = [oF '/PACsub_' num2str(i) '.fig'];
        
        f1Fig = openfig(f1);
        f2Fig = openfig(f2);
        
        f1X(:,i) = f1Fig.CurrentAxes.Children(1).XData;
        f1Y(:,i) = f1Fig.CurrentAxes.Children(1).YData;
        
        f2X(:,i) = f2Fig.CurrentAxes.Children(1).XData;
        f2Y(:,i) = f2Fig.CurrentAxes.Children(1).YData;
        
        figure(f2Fig)
        hold on
        yyaxis right
        p = plot(f1X(:,i)-1,f1Y(:,i),'LineStyle','--','Marker','s','MarkerFaceColor',[0 0 0],'Color',[0.7 0.7 0.7],'LineWidth',3,'MarkerSize',10);
        
        f2Fig.CurrentAxes.YAxis(1).Label.String = 'PAC';
        f2Fig.CurrentAxes.YAxis(2).Label.String = 'Dip statistic';
        f2Fig.CurrentAxes.YAxis(2).Parent.YColor = [0 0 0];
        
        title(['PAC vs Dip statistic -- subject ' num2str(i)],'fontsize',22)
        ylabel('PAC','fontsize',22)
        xlabel('k','fontsize',22)
        
        saveas(gcf,[oF '/PAC+DIPsub_' num2str(i) '.fig'])
        saveas(gcf,[oF '/PAC+DIPsub_' num2str(i) '.pdf'])
        close all
    catch
        close all
    end

    % Make PAC x BC fig
    try
        f1 = [oF '/BCsub_' num2str(i) '.fig'];
        f2 = [oF '/PACsub_' num2str(i) '.fig'];
        
        f1Fig = openfig(f1);
        f2Fig = openfig(f2);
        
        f1X(:,i) = f1Fig.CurrentAxes.Children(1).XData;
        f1Y(:,i) = f1Fig.CurrentAxes.Children(1).YData;
        
        f2X(:,i) = f2Fig.CurrentAxes.Children(1).XData;
        f2Y(:,i) = f2Fig.CurrentAxes.Children(1).YData;
        
        figure(f2Fig)
        hold on
        yyaxis right
        p = plot(f1X(:,i)-1,f1Y(:,i),'LineStyle','--','Marker','s','MarkerFaceColor',[0 0 0],'Color',[0.7 0.7 0.7],'LineWidth',3,'MarkerSize',10);
        
        f2Fig.CurrentAxes.YAxis(1).Label.String = 'PAC';
        f2Fig.CurrentAxes.YAxis(2).Label.String = 'BC';
        f2Fig.CurrentAxes.YAxis(2).Parent.YColor = [0 0 0];
        
        title(['PAC vs BC -- subject ' num2str(i)],'fontsize',22)
        ylabel('PAC','fontsize',22)
        xlabel('k','fontsize',22)
        
        saveas(gcf,[oF '/PAC+BCsub_' num2str(i) '.fig'])
        saveas(gcf,[oF '/PAC+BCsub_' num2str(i) '.pdf'])
        close all
    catch
        close all
    end
end

%% Step 4) Compare fused metrics to unfused...
load('fusedMetrics.mat')
load('SCEuc_allSubs.mat')
load('kR_allSubs.mat')
kRange = 2:30;

% pac
fusedPacM = mean(fusedPac');
fusedPacStd = std(fusedPac');
fusedPacSEM = fusedPacStd./sqrt(137);

SCEucPacM = mean(SCEucPac');
SCEucPacStd = std(SCEucPac');
SCEucPacSEM = SCEucPacStd./sqrt(137);

kRPacM = mean(kRPac');
kRPacStd = std(kRPac');
kRPacSEM = kRPacStd./sqrt(137);

cMat = distinguishable_colors(3);
figure('Color','w','pos',[10 10 1100 900])
shadedErrorBar(kRange,fusedPacM,fusedPacSEM,'lineProps',cMat(1,:))
hold on
shadedErrorBar(kRange,SCEucPacM,SCEucPacSEM,'lineProps',cMat(2,:))
shadedErrorBar(kRange,kRPacM,kRPacSEM,'lineProps',cMat(3,:))

title(['Mean PAC across solutions'])
xlabel('k');
ylabel('PAC');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
hold off

% bc
fusedBCM = mean(fused_bc');
fusedBCStd = std(fused_bc');
fusedBCSEM = fusedBCStd./sqrt(137);

SCEucBCM = mean(SCEuc_bc');
SCEucBCStd = std(SCEuc_bc');
SCEucBCSEM = SCEucBCStd./sqrt(137);

kRBCM = mean(kR_bc');
kRBCStd = std(kR_bc');
kRBCSEM = kRBCStd./sqrt(137);

cMat = distinguishable_colors(3);
figure('Color','w','pos',[10 10 1100 900])
shadedErrorBar(kRange,fusedBCM,fusedBCSEM,'lineProps',cMat(1,:))
hold on
shadedErrorBar(kRange,SCEucBCM,SCEucBCSEM,'lineProps',cMat(2,:))
shadedErrorBar(kRange,kRBCM,kRBCSEM,'lineProps',cMat(3,:))

title(['Mean PAC across solutions'])
xlabel('k');
ylabel('PAC');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
hold off

%% compare some individual CDF curves between fused consensus, SC Euclidean, and k-means R
openfig('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/eval3/PAC+DIPsub_1.fig')
openfig('/Volumes/HICKOK-LAB/inSubs/SCEuc/eval4/firstSubset/Sub_1_PAC+BC.fig')

%% new plan: 
% run optimization on SCEuc and kR
% choose winning  model for each subject
% run optimization seperately on fused matrices
% check which does best....

type = 'globMax';
numStdDevs = 1;%1.5;
pThresh = 0.001;
kRange = 2:30;
dipCase = 'noise';
nstd = 1.5;
conThresh = 2;
sigmaTol = 31; % had been set to 20
sigmaAdj = 2;
[optFused,~,dipFused,~,~,~,~,~,~,~,~,fusedSigma] = optimalConsensus(fusedPac,fused_bc,fused_dip,fused_dip_p,dipCase,type,numStdDevs,pThresh,sigmaTol,kRange,sigmaAdj,conThresh);
nstd = 1.5;
[optFusedAdj_pac,optFusedAdj] = adjustOptCon(optFused,fusedPac,nstd,kRange);
optFusedAdj_pac2 = optFusedAdj_pac;
id = find(optFusedAdj_pac2 == 0);
optFusedAdj_pac2(id) = NaN;
    
figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
vs2 = violinplot([optFusedAdj'; optFusedAdj2' nan([1,112])]', {'whole';'subset'},'Bandwidth',0.02);
cTmp = distinguishable_colors((2));
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 1;
    vs2(i).BoxPlot.LineWidth = 1.5;
    vs2(i).WhiskerPlot.LineWidth = 0.5;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 1.5;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 4;
end
ylim([0 0.6])
title(['PAC values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('PAC statistic');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

optFusedAdj2 = optFusedAdj;
id = find(optFusedAdj2 == 0);
optFusedAdj2(id) = 1;

figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
vs2 = violinplot(optFusedAdj2, {'Fused'},'Bandwidth',0.5);
cTmp = distinguishable_colors(1);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 1;
    vs2(i).BoxPlot.LineWidth = 1.5;
    vs2(i).WhiskerPlot.LineWidth = 0.5;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 1.5;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 4;
end
title(['PAC values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('PAC statistic');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

% also get subset violins
optFusedAdj_pac2 = optFusedAdj_pac([1:22 24 26 27]);
id = find(optFusedAdj_pac2 == 0);
optFusedAdj_pac2(id) = NaN;
    
figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
%vs2 = violinplot(optFusedAdj_pac2, {'Fused'},'Bandwidth',0.02);
vs2 = violinplot([optFusedAdj_pac'; optFusedAdj_pac2' nan([1,112])]', {'whole';'subset'},'Bandwidth',0.02);
cTmp = distinguishable_colors(2);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 1;
    vs2(i).BoxPlot.LineWidth = 1.5;
    vs2(i).WhiskerPlot.LineWidth = 0.5;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 1.5;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 4;
end
ylim([0 0.6])
title(['PAC values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('PAC statistic');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

optFusedAdj2 = optFusedAdj([1:22 24 26 27]);
id = find(optFusedAdj2 == 0);
optFusedAdj2(id) = 1;

figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
vs2 = violinplot(optFusedAdj2, {'Fused'},'Bandwidth',0.5);
cTmp = distinguishable_colors(1);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 1;
    vs2(i).BoxPlot.LineWidth = 1.5;
    vs2(i).WhiskerPlot.LineWidth = 0.5;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 1.5;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 4;
end
title(['PAC values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('PAC statistic');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

%% fuse subject-level con with same k
baseDir = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR';
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(i).name,'.mat');
    fStop(i,1) = str2double(f(i).name(1:tmp-1));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

unSol = unique(optFusedAdj);
id = find(unSol == 0);
unSol(id) = [];
for i = 1:length(unSol)
    subId = find(optFusedAdj == unSol(i));
    clusterTracker{i} = zeros([2159,2159]);
    %clusterConsensus{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
end
load('/Users/ateghipc/Projects/PT/matlabVars/Final/group/PCA/InfoTheory/WithinSubs_noPerms_concatenated_conMats_v1.mat', 'conID')

% now load each consensus matrix and go through them one-by-one
for i = 1:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker'); % load in data
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
   
   if sum(sum(isinf(tmp.clusterTracker{fStartS}(:,:,2)))) > 0
       tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker');
       for j = fStartS:fStopS(i) % loop over subjects
           for k = 1:size(tmp.clusterTracker{j},3) % look over k
               tmp.clusterConsensus{j}(:,:,k) = tmp.clusterTracker{j}(:,:,k)./tmp.itemTracker{j}(:,:,1);
           end
       end
   end
       
   for j = fStartS:fStopS(i) % loop over subjects in this subset
       disp(['Working on subject -- ' num2str(j)])
       if optFusedAdj(j) ~= 0
           id = find(unSol == optFusedAdj(j));
           clusterTracker{id}(conID{j},conID{j}) = clusterTracker{id}(conID{j},conID{j}) + tmp.clusterTracker{j}(:,:,optFusedAdj(j)-1);
           itemTracker{id}(conID{j},conID{j}) = itemTracker{id}(conID{j},conID{j}) + tmp.itemTracker{j};
       end
   end
   clear tmp
end

% save
for i = 1:length(clusterTracker)
   clusterConsensus(:,:,i) = clusterTracker{i} ./itemTracker{i}; 
end
save('fusedMatrices_byK.mat','clusterConsensus','clusterTracker','itemTracker')

% now cluster
load('/Users/ateghipc/Projects/PT/matlabVars/Final/group/PCA/InfoTheory/AllSubs.mat','LH_PPS_I','template')
LH_PPS_IU = unique(LH_PPS_I);
of = ['/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/clusters'];
mkdir(of)
neighbor_num = [10; 16; 25; 50; 100; 200; 300; 400; 500];
for j = 1:length(neighbor_num)
    for i = 1:size(clusterConsensus,3)
        LH_PPS_I2 = LH_PPS_IU;
        autoData = clusterConsensus(:,:,i);
%         id = find(autoData == 0);
%         autoData(id,:) = [];
%         autoData(:,id) = [];
%         LH_PPS_I2(id) = [];
         
        id = find(isnan(autoData));
        %[r,c] = ind2sub([size(autoData,1),size(autoData,2)],id);
        autoData(id) = 0;
        
        atuoData = 1 - autoData;
        
        id = find(std(autoData) == 0);
        autoData(id,:) = [];
        autoData(:,id) = [];
        LH_PPS_I2(id) = [];
       
%         autoData(:,id) = [];
%         LH_PPS_I2(id) = [];

        clear A_LS ZERO_DIAG clusts_RLS rlsBestGroupIndex qualityRLS clusters clustTmp
        [D_LS,A_LS,LS] = scale_dist(double(autoData),floor(neighbor_num(j)/2)); %% Locally scaled affinity matrix
        clear D_LS LS;
        remove = isnan(A_LS);
        idx = find(remove);
        A_LS(idx) = 0;
        % zero out diagonal
        ZERO_DIAG = ~eye(size(autoData,1));
        A_LS = A_LS.*ZERO_DIAG;
        % cluster all clustering choices
        [clusts_RLS, rlsBestGroupIndex, qualityRLS] = cluster_rotate(A_LS,unSol(i),0,1);
        for m = 1:length(clusts_RLS)
            clustTmp = clusts_RLS{m};
            clusters(:,m) = zeros([size(autoData,1), 1]);
            for l = 1:length(clustTmp)
                clusters(clustTmp{l},m) = l;
            end
        end
        template.img = zeros(91,109,91);
        template.img(LH_PPS_I2) = clusters;
        save_untouch_nii(template,[of '/SelfTuningSpectral_k_' num2str(unSol(i)) '_neighbors_' num2str(neighbor_num(j)) '.nii.gz'])       
    end
end

LH_PPS_I2 = LH_PPS_IU;
test = clusterConsensus(:,:,6);
id = find(isnan(test));
test(id) = 0;
id = find(std(test) == 0);
test(id,:) = [];
test(:,id) = [];
LH_PPS_I2(id) = [];

[idx,C] = kmeans(test,6,'Distance','correlation','Replicates',500);
template.img = zeros(91,109,91);
template.img(LH_PPS_I2) = idx;
save_untouch_nii(template,[of '/kMeans_k_6_Correl.nii.gz'])

%% testing
for i = 1:8
    c = i;
    test = clusterConsensus(:,:,c);
    
    if i == 3
        for j = 1:size(test,1)
            test(j,j) = 1;
        end
    end
    
    id = find(isnan(test));
    test(id) = 0;
    id = find(std(test) == 0);
    test(id,:) = [];
    test(:,id) = [];
    LH_PPS_I2 = LH_PPS_IU;
    LH_PPS_I2(id) = [];
    test = 1 - test;
    test2 = squareform(test);
    Z = linkage(test2,'average');
    if i ~= 3
        Cl = cluster(Z,'maxclust',c+1);
    else
        Cl = cluster(Z,'maxclust',c+2);
    end
    template.img = zeros(91,109,91);
    template.img(LH_PPS_I2) = Cl;
    save_untouch_nii(template,[of '/HC_k_' num2str(c+1) '_average.nii.gz'])
end

%% Now do pac eval for just KR and just SCEuc
%% First, we get optima for fused, KR and SCEUc and generate violin plots
%% kR
% Load in metrics
load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/kR_allSubs.mat')

% Now get optima
type = 'globMax';
numStdDevs = 1;%1.5;
pThresh = 0.001;
kRange = 2:30;
dipCase = 'noise';
nstd = 1.5;
conThresh = 2;
sigmaTol = 20; % originally set to 20
sigmaAdj = 2;
[opt_kR,~,dip_kR,~,~,~,~,~,~,~,~,sigma_kR] = optimalConsensus(kRPac,kR_bc,kR_dip,kR_dip_p,dipCase,type,numStdDevs,pThresh,sigmaTol,kRange,sigmaAdj,conThresh);
nstd = 1.5;
[opt_kR_adj_pac,opt_kR_adj] = adjustOptCon(opt_kR,kRPac,nstd,kRange);

load('/Volumes/HICKOK-LAB/inSubs/AllSubMetrics/SCEuc_allSubs.mat')

% Now get optima
[opt_SCEuc,~,dip_SCEuc,~,~,~,~,~,~,~,~,sigma_SCEuc] = optimalConsensus(SCEucPac,SCEuc_bc,SCEuc_dip,SCEuc_dip_p,dipCase,type,numStdDevs,pThresh,sigmaTol,kRange,sigmaAdj,conThresh);
nstd = 1.5;
[opt_SCEuc_adj_pac,opt_SCEuc_adj] = adjustOptCon(opt_SCEuc,SCEucPac,nstd,kRange);


% first, make some correlations and scatter plots between 
save('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/fusedViolin2.mat','opt_SCEuc_adj_pac','opt_SCEuc_adj','opt_SCEuc','opt_kR_adj_pac','opt_kR_adj','opt_kR','optFusedAdj','optFusedAdj_pac','optFused')

% correlate scEuc x kR
[scEuc_kR_post_r,scEuc_kR_post_p] = corr(opt_SCEuc_adj,opt_kR_adj);
[scEuc_kR_pre_r,scEuc_kR_pre_p] = corr(opt_SCEuc,opt_kR);

% now violin 
figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
vs2 = violinplot([optFusedAdj'; optFusedAdj2' nan([1,112]); opt_SCEuc_adj'; opt_kR_adj']', {'Fused (all subs) ';'fused (subset)';'SCEuc';'kR'},'Bandwidth',0.5); %0.02
cTmp = distinguishable_colors(length(vs2));
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 1;
    vs2(i).BoxPlot.LineWidth = 1.5;
    vs2(i).WhiskerPlot.LineWidth = 0.5;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 1.5;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 4;
end
ylim([0 12])
title(['k values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('k');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
vs2 = violinplot([optFusedAdj_pac'; optFusedAdj_pac2' nan([1,112]); opt_SCEuc_adj_pac'; opt_kR_adj_pac']', {'Fused (all subs) ';'fused (subset)';'SCEuc';'kR'},'Bandwidth',0.02); %0.02
cTmp = distinguishable_colors(length(vs2));
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 1;
    vs2(i).BoxPlot.LineWidth = 1.5;
    vs2(i).WhiskerPlot.LineWidth = 0.5;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 1.5;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 4;
end
ylim([0 0.6])
title(['PAC values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('PAC');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

%% Evaluate kR
baseDir = '/Volumes/HICKOK-LAB/inSubs/kR';
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(1).name,'_');
    fStop(i,1) = str2double(f(i).name(tmp(end)+1:end-4));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

unSol = unique(opt_kR_adj);
id = find(unSol == 1);
unSol(id) = [];
for i = 1:length(unSol)
    subId = find(opt_kR_adj == unSol(i));
    clusterTracker{i} = zeros([2159,2159]);
    %clusterConsensus{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
end
load('/Users/ateghipc/Projects/PT/matlabVars/Final/group/PCA/InfoTheory/WithinSubs_noPerms_concatenated_conMats_v1.mat', 'conID')

% now load each consensus matrix and go through them one-by-one
for i = 2:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker'); % load in data
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
%    
%    if sum(sum(isinf(tmp.clusterTracker{fStartS}(:,:,2)))) > 0
%        tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker');
%        for j = fStartS:fStopS(i) % loop over subjects
%            for k = 1:size(tmp.clusterTracker{j},3) % look over k
%                tmp.clusterConsensus{j}(:,:,k) = tmp.clusterTracker{j}(:,:,k)./tmp.itemTracker{j}(:,:,1);
%            end
%        end
%    end
       
   for j = fStartS:fStopS(i) % loop over subjects in this subset
       disp(['Working on subject -- ' num2str(j)])
       if opt_kR_adj(j) ~= 1
           id = find(unSol == opt_kR_adj(j));
           clusterTracker{id}(conID{j},conID{j}) = clusterTracker{id}(conID{j},conID{j}) + tmp.clusterTracker{j}(:,:,opt_kR_adj(j)-1);
           try
               itemTracker{id}(conID{j},conID{j}) = itemTracker{id}(conID{j},conID{j}) + tmp.itemTracker{j};
           catch
               itemTracker{id}(conID{j},conID{j}) = itemTracker{id}(conID{j},conID{j}) + tmp.itemTracker{j}(:,:,1);
           end
       end
   end
   clear tmp
end

% save
for i = 1:length(clusterTracker)
   clusterConsensus(:,:,i) = clusterTracker{i} ./itemTracker{i}; 
end
save('fusedMatrices_byK_kR.mat','clusterConsensus','clusterTracker','itemTracker')

clear clusterConsensus clusterTracker itemTracker tmp

%% Evaluate SCEuc
baseDir = '/Volumes/HICKOK-LAB/inSubs/SCEuc';
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(1).name,'_');
    fStop(i,1) = str2double(f(i).name(tmp(end)+1:end-4));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

unSol = unique(opt_SCEuc_adj);
id = find(unSol == 1);
unSol(id) = [];
for i = 1:length(unSol)
    subId = find(opt_SCEuc_adj == unSol(i));
    clusterTracker{i} = zeros([2159,2159]);
    %clusterConsensus{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
end

% now load each consensus matrix and go through them one-by-one
for i = 1:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker'); % load in data
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
%    
%    if sum(sum(isinf(tmp.clusterTracker{fStartS}(:,:,2)))) > 0
%        tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker');
%        for j = fStartS:fStopS(i) % loop over subjects
%            for k = 1:size(tmp.clusterTracker{j},3) % look over k
%                tmp.clusterConsensus{j}(:,:,k) = tmp.clusterTracker{j}(:,:,k)./tmp.itemTracker{j}(:,:,1);
%            end
%        end
%    end
       
   for j = fStartS:fStopS(i) % loop over subjects in this subset
       disp(['Working on subject -- ' num2str(j)])
       if opt_SCEuc_adj(j) ~= 1
           id = find(unSol == opt_SCEuc_adj(j));
           clusterTracker{id}(conID{j},conID{j}) = clusterTracker{id}(conID{j},conID{j}) + tmp.clusterTracker{j}(:,:,opt_SCEuc_adj(j)-1);
           try
               itemTracker{id}(conID{j},conID{j}) = itemTracker{id}(conID{j},conID{j}) + tmp.itemTracker{j};
           catch
               itemTracker{id}(conID{j},conID{j}) = itemTracker{id}(conID{j},conID{j}) + tmp.itemTracker{j}(:,:,1);
           end
       end
   end
   clear tmp
end

% save
for i = 1:length(clusterTracker)
   clusterConsensus(:,:,i) = clusterTracker{i} ./itemTracker{i}; 
end
save('fusedMatrices_byKSCEuc.mat','clusterConsensus','clusterTracker','itemTracker')

clear clusterConsensus clusterTracker itemTracker tmp

%% Now cluster
 

of = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/clusters_kR';
mkdir(of)
for i = 1:6
    c = i;
    test = clusterConsensus(:,:,c);
    id = find(isnan(test));
    test(id) = 0;
    id = find(std(test) == 0);
    test(id,:) = [];
    test(:,id) = [];
    LH_PPS_I2 = LH_PPS_IU;
    LH_PPS_I2(id) = [];
    test = 1 - test;
    test2 = squareform(test);
    Z = linkage(test2,'average');

    Cl = cluster(Z,'maxclust',c+1);
    template.img = zeros(91,109,91);
    template.img(LH_PPS_I2) = Cl;
    save_untouch_nii(template,[of '/HC_k_' num2str(c+1) '_average.nii.gz'])
end

%% Get raw fused matrices, etc
load('/Users/ateghipc/Projects/PT/matlabVars/Final/group/PCA/InfoTheory/WithinSubs_noPerms_concatenated_conMats_v1.mat', 'conID')

baseDir = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR';
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(i).name,'.mat');
    fStop(i,1) = str2double(f(i).name(1:tmp-1));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

for i = 1:29
    clusterTracker{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
end
% now load each consensus matrix and go through them one-by-one
for i = 1:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker'); % load in data
   
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
   
   a = fStartS:fStopS(i);
   for j = fStartS:fStopS(i)%1:size(tmp.clusterTracker,2)
       for k = 1:29
           clusterTracker{k}(conID{(j)},conID{(j)}) = clusterTracker{k}(conID{(j)},conID{(j)}) + tmp.clusterTracker{j}(:,:,k);
           itemTracker{k}(conID{(j)},conID{(j)}) = itemTracker{k}(conID{(j)},conID{(j)}) + tmp.itemTracker{j}(:,:);
       end
   end
   clear tmp
end

% save
for i = 1:length(clusterTracker)
   clusterConsensus(:,:,i) = clusterTracker{i} ./itemTracker{i}; 
end
save('fusedMatrices_byK_raw.mat','clusterConsensus','clusterTracker','itemTracker')

baseDir = '/Volumes/HICKOK-LAB/inSubs/kR';
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(i).name,'_');
    tmp2 = strfind(f(i).name,'.mat');
    fStop(i,1) = str2double(f(i).name(tmp(2)+1:tmp2-1));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

for i = 1:29
    clusterTracker{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
end
% now load each consensus matrix and go through them one-by-one
for i = 2:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker'); % load in data
   
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
   
   a = fStartS:fStopS(i);
   for j = fStartS:fStopS(i)%1:size(tmp.clusterTracker,2)
       for k = 1:29
           clusterTracker{k}(conID{(j)},conID{(j)}) = clusterTracker{k}(conID{(j)},conID{(j)}) + tmp.clusterTracker{j}(:,:,k);
           itemTracker{k}(conID{(j)},conID{(j)}) = itemTracker{k}(conID{(j)},conID{(j)}) + tmp.itemTracker{j}(:,:,1);
       end
   end
   clear tmp
end

% save
for i = 1:length(clusterTracker)
   clusterConsensus(:,:,i) = clusterTracker{i} ./itemTracker{i}; 
end
save('fusedMatrices_byK_kR_raw.mat','clusterConsensus','clusterTracker','itemTracker')

baseDir = '/Volumes/HICKOK-LAB/inSubs/SCEuc';
f = dir([baseDir '/*.mat']); % get all .mat files
for i = 1:length(f)
    tmp = strfind(f(i).name,'_');
    tmp2 = strfind(f(i).name,'.mat');
    fStop(i,1) = str2double(f(i).name(tmp(4)+1:tmp2-1));
    fName{i} = f(i).name;
end
[fStopS,sortI] = sort(fStop);
fNameS = fName(sortI);
clear tmp f fStop fName

for i = 1:29
    clusterTracker{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
end
% now load each consensus matrix and go through them one-by-one
for i = 1:length(fNameS) % loop over subsets of subjects
   disp(['Working on subject subset -- ' num2str(i)])
   tmp = load([baseDir '/' fNameS{i}],'clusterTracker','itemTracker'); % load in data
   
   if i == 1 % get endpoints
       fStartS = 1;
   else
       fStartS = fStopS(i-1)+1;
   end
   
   a = fStartS:fStopS(i);
   for j = fStartS:fStopS(i)%1:size(tmp.clusterTracker,2)
       for k = 1:29
           clusterTracker{k}(conID{(j)},conID{(j)}) = clusterTracker{k}(conID{(j)},conID{(j)}) + tmp.clusterTracker{j}(:,:,k);
           itemTracker{k}(conID{(j)},conID{(j)}) = itemTracker{k}(conID{(j)},conID{(j)}) + tmp.itemTracker{j}(:,:,1);
       end
   end
   clear tmp
end

% save
for i = 1:length(clusterTracker)
   clusterConsensus(:,:,i) = clusterTracker{i} ./itemTracker{i}; 
end
save('fusedMatrices_byK_SCEuc_raw.mat','clusterConsensus','clusterTracker','itemTracker')

%% now cluster
load('/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/fusedMatrices_byK_raw.mat')
of = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/clusters_raw';
mkdir(of)
for i = 1:29
    c = i;
    test = clusterConsensus(:,:,c);
    id = find(isnan(test));
    test(id) = 0;
    id = find(std(test) == 0);
    test(id,:) = [];
    test(:,id) = [];
    LH_PPS_I2 = LH_PPS_IU;
    LH_PPS_I2(id) = [];
    test = 1 - test;
    test2 = squareform(test);
    Z = linkage(test2,'average');

    Cl = cluster(Z,'maxclust',c+1);
    template.img = zeros(91,109,91);
    template.img(LH_PPS_I2) = Cl;
    save_untouch_nii(template,[of '/HC_k_' num2str(c+1) '_average.nii.gz'])
end

clear clusterConsensus
load('fusedMatrices_byK_SCEuc_raw.mat', 'clusterConsensus')
of = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/clusters_SCEuc_raw';
mkdir(of)
for i = 1:29
    c = i;
    test = clusterConsensus(:,:,c);
    id = find(isnan(test));
    test(id) = 0;
    id = find(std(test) == 0);
    test(id,:) = [];
    test(:,id) = [];
    LH_PPS_I2 = LH_PPS_IU;
    LH_PPS_I2(id) = [];
    test = 1 - test;
    test2 = squareform(test);
    Z = linkage(test2,'average');

    Cl = cluster(Z,'maxclust',c+1);
    template.img = zeros(91,109,91);
    template.img(LH_PPS_I2) = Cl;
    save_untouch_nii(template,[of '/HC_k_' num2str(c+1) '_average.nii.gz'])
end


clear clusterConsensus
load('fusedMatrices_byK_kR_raw.mat', 'clusterConsensus')
of = '/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/clusters_kR_raw';
mkdir(of)
for i = 1:29
    c = i;
    test = clusterConsensus(:,:,c);
    id = find(isnan(test));
    test(id) = 0;
    id = find(std(test) == 0);
    test(id,:) = [];
    test(:,id) = [];
    LH_PPS_I2 = LH_PPS_IU;
    LH_PPS_I2(id) = [];
    test = 1 - test;
    test2 = squareform(test);
    Z = linkage(test2,'average');

    Cl = cluster(Z,'maxclust',c+1);
    template.img = zeros(91,109,91);
    template.img(LH_PPS_I2) = Cl;
    save_untouch_nii(template,[of '/HC_k_' num2str(c+1) '_average.nii.gz'])
end















%% 
unSol = unique(optFusedAdj);
id = find(unSol == 0);
unSol(id) = [];
for i = 1:length(unSol)
    sol = unSol(i);
    subId = find(optFusedAdj == sol);
    clusterTracker{i} = zeros([2159,2159]);
    clusterConsensus{i} = zeros([2159,2159]);
    itemTracker{i} = zeros([2159,2159]);
    
    for j = [10:10:130 137]
       tmp = load(['/Volumes/HICKOK-LAB/inSubs/Fused_SCEuc_kR/' num2str(j) '.mat']);
       
        
    end
    
    
    
    for j = 1:length(subId)
        
        
        
    end
end







%% 

type = 'globMax';
numStdDevs = 1.5;
pThresh = 0.001;
kRange = 2:30;
dipCase = 'noise';
nstd = 1.5;
conThresh = 2;
sigmaTol = 20;
sigmaAdj = 2;
[optkR,~,dipkR,~,~,~,~,~,~,~,~,kRSigma] = optimalConsensus(kRPac,kR_bc,kR_dip,kR_dip_p,dipCase,type,numStdDevs,pThresh,sigmaTol,kRange,sigmaAdj,conThresh);
nstd = 1.5;
[optkRAdj_pac,optkRAdj] = adjustOptCon(optkR,kRPac,nstd,kRange);
    
type = 'globMax';
numStdDevs = 1.5;
pThresh = 0.001;
kRange = 2:30;
dipCase = 'noise';
nstd = 1.5;
conThresh = 2;
sigmaTol = 20;
sigmaAdj = 2;
[optSCEuc,~,dipSCEuc,~,~,~,~,~,~,~,~,SCEucSigma] = optimalConsensus(SCEucPac,SCEuc_bc,SCEuc_dip,SCEuc_dip_p,dipCase,type,numStdDevs,pThresh,sigmaTol,kRange,sigmaAdj,conThresh);
nstd = 1.5;
[optSCEucAdj_pac,optSCEucAdj] = adjustOptCon(optSCEuc,SCEucPac,nstd,kRange);

optPac = [optFusedAdj_pac optkRAdj_pac optSCEucAdj_pac];
optK = [optFusedAdj optkRAdj optSCEucAdj];
optLabels = {'Fused'; 'K-R'; 'SC-Euc'};
optPac2 = optPac;
id = find(optPac2 == 0);
optPac2(id) = NaN;

%% Make violin plots of pac for winning solutions
figure
%vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.03);
vs2 = violinplot(optPac2, optLabels,'Bandwidth',0.02);
cTmp = distinguishable_colors(length(optLabels));
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 5;
    vs2(i).WhiskerPlot.LineWidth = 2;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineWidth = 3;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 0.6];
    vs2(i).MedianPlot.SizeData = 120;
end
title(['PAC values of winning solutions across subjects'])
xlabel('Clustering approach');
ylabel('PAC statistic');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

figure
vs1 = violinplot(optK, optLabels,'Bandwidth',0.5);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs1(i).ViolinColor = cTmp(i,:);
    vs1(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs1(i).ViolinPlot.LineWidth = 5;
    vs1(i).BoxPlot.LineWidth = 5;
    vs1(i).WhiskerPlot.LineWidth = 2;
    vs1(i).ShowMean = 1;
    vs1(i).MeanPlot.LineWidth = 3;
    vs1(i).MeanPlot.LineStyle = '-';
    vs1(i).MeanPlot.Color = [vs1(i).MeanPlot.Color 0.6];
    vs1(i).MedianPlot.SizeData = 120;
end
title(['Number of clusters in winning solutions (across subjects)'])
xlabel('Clustering approach');
ylabel('Number of clusters');
set(gca,'FontSize',25)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
xtickangle(90)

%% load in stuff, etc
























































%% Step 4) Select k
























%% rename figures




%% 
