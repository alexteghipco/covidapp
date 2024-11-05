%% Prelim
for i = 1:length(subsOnly)
    try
        id = find(ismember(wabtmp.record_id_c84280,subsOnly{i}));
        dp = wabtmp.daysPOS(id);
        mn(i,1) = min(dp);
    catch
        mn(i,1) = NaN;
    end
end

%% We will first generate grad cam maps...
addpath(genpath('/Users/alex/Documents/CSTAR/spm12'))
addpath(genpath('/Users/alex/Downloads/NiiStat-master'))

load('PADInfo.mat')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_gradcam_maps','all_gradcam_maps_oppo','all_true_labels','all_predictions','all_probs','all_fold')
load('PyTorchInput_withFolds.mat', 'subsOnly')
subsOnly([36 38 74 75 161]) = [];

% convert preds
for i = 1:size(all_predictions,2)
    id1 = find(all_predictions(:,i) == 0);
    id2 = find(all_predictions(:,i) == 1);
    all_preds(id1,i) = "Non-severe";
    all_preds(id2,i) = "Severe";
end
clear all_predictions

all_true_labels=all_true_labels(:,1);
id1 = find(all_true_labels == 0);
id2 = find(all_true_labels == 1);
wabClass(id1,1) = "Non-severe";
wabClass(id2,1) = "Severe";

template = load_nifti('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz');
vid = find(template.vol == 0);
template = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/nii/halved_halved_overlap_crop/M2002_subsamp_subsamp_FIN.nii');
tarhdr = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz'); %load header
tarhdr2 = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/nii/halved_halved/M2002_subsamp_subsamp_FIN.nii');

bd = '/Volumes/Quattro/CNN_FeatureMaps/gCAMpp';
mkdir(bd)
for r = 1:20
    of = [bd '/' num2str(r)];
    mkdir(of)
    parfor j = 1:size(all_gradcam_maps,1)
        disp(num2str(j))
        inhdr = template; %load header
        inimg = squeeze(all_gradcam_maps(j,:,:,:,r)); %load volume
        
        % pad
        inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
        xs = setdiff([1:size(inimgo,1)],torX);
        ys = setdiff([1:size(inimgo,2)],torY);
        zs = setdiff([1:size(inimgo,3)],torZ);
        inimgo(xs,ys,zs) = inimg;%flip(tmp1,1);

        [outhdr,outimg] = nii_reslice_target(tarhdr2,inimgo,tarhdr);
        outhdr.fname = [of '/' subsOnly{j} '_gradCAM.nii'];
        spm_write_vol(outhdr,outimg);
        gzip(outhdr.fname)
        delete(outhdr.fname)
        
        inimg = squeeze(all_gradcam_maps_oppo(j,:,:,:,r)); %load volume
        inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
        xs = setdiff([1:size(inimgo,1)],torX);
        ys = setdiff([1:size(inimgo,2)],torY);
        zs = setdiff([1:size(inimgo,3)],torZ);
        inimgo(xs,ys,zs) = inimg;%flip(tmp1,1);
        
        [outhdr,outimg] = nii_reslice_target(tarhdr2,inimgo,tarhdr);
        outhdr.fname = [of '/' subsOnly{j} '_gradCAMoppo.nii'];
        spm_write_vol(outhdr,outimg);
        gzip(outhdr.fname)
        delete(outhdr.fname)
    end
end

% now do brain images
off = '/Volumes/Quattro/CNN_FeatureMaps/gCAMppUp';
mkdir(off)
for r = 1:20 % initially got through rep 5
    for i = 1:length(subsOnly)
        tmpl{1} = '/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz';
        tmpl{2} = ['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_FIN.nii'];
        [~,cont,~,~,t] = brainMontager(tmpl,[bd '/' num2str(r) '/' subsOnly{i} '_gradCAM.nii.gz'],[25:5:85 95:5:155],5,[],'gray',[],[1 1 1],true,1000,[],'jet',[],0.3,'ud',true,false,'sagittal',[0.5000    1.0000    1.5000    2.0000    2.5000    3.0000],1,1,[2 13]);
        t.TileSpacing = 'none';
        for iii = 1:size(cont,1)
            cont{iii,2}.LineWidth = 4;
            cont{iii,2}.LineColor = [0.8588    0.1137    0.4627];%[0.4941    0.1255    0.7412];
        end
        set(gcf, 'Position',  [0 500 3265 417])
        ax = gca;
        ax.Title.String = [subsOnly{i} ' rep ' num2str(r) ' gradCAM++ y: ' wabClass{i} '; yh: ' all_preds{i,r} ' (' num2str(all_probs(i,r)) ')'];
        ax.Title.Position = [350 1500 0];
        if strcmpi(wabClass{i},all_preds{i,r})
            saveas(gcf,[off '/' subsOnly{i} '_rep_' num2str(r) '_gradCAM_correct_' all_preds{i,r} '.png'])
        else
            saveas(gcf,[off '/' subsOnly{i} '_rep_' num2str(r) '_gradCAM_incorrect_' all_preds{i,r} '.png'])
        end
        close all
        
       [~,cont,~,~,t] = brainMontager(tmpl,[bd '/' num2str(r) '/' subsOnly{i} '_gradCAMoppo.nii.gz'],[25:5:85 95:5:155],5,[],'gray',[],[1 1 1],true,1000,[],'jet',[],0.3,'ud',true,false,'sagittal',[0.5000    1.0000    1.5000    2.0000    2.5000    3.0000],1,1,[2 13]);
        t.TileSpacing = 'none';
        for iii = 1:size(cont,1)
            cont{iii,2}.LineWidth = 4;
            cont{iii,2}.LineColor = [0.8588    0.1137    0.4627];%[0.4941    0.1255    0.7412];
        end
        set(gcf, 'Position',  [0 500 3265 417])
        ax = gca;
        
        if strcmpi(all_preds{i,r},'Severe')
            ax.Title.String = [subsOnly{i} ' rep ' num2str(r) ' gradCAM++ y: ' wabClass{i} '; yh(oppo): Nonsevere'];
            ax.Title.Position = [350 1500 0];
            saveas(gcf,[off '/' subsOnly{i} '_rep_' num2str(r) '_gradCAM_oppo_Nonsevere.png'])
        else
            ax.Title.String = [subsOnly{i} ' rep ' num2str(r) ' gradCAM++ y: ' wabClass{i} '; yh(oppo): Severe'];
            ax.Title.Position = [350 1500 0];
            saveas(gcf,[off '/' subsOnly{i} '_rep_' num2str(r) '_gradCAM_oppo_Severe.png'])
        end
        close all
    end
end

%% average all lesions for each class...
% but average only the lesions the network got right. 
r = 2;
tr = find(all_true_labels(:,r) == all_predictions(:,r));

lesoS = zeros(182,218,182);
gcamoS = zeros(17,22,19);
lesoNS = zeros(182,218,182);
gcamoNS = zeros(17,22,19);
for i = 1:length(tr)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{tr(i)} '_subsamp_subsamp_FIN.nii']);
    id = find(tmp.vol == 1);
    
    if all_true_labels(tr(i),r) == 1
        lesoS(id) = lesoS(id)+1;
        gcamoS = gcamoS + squeeze(all_gradcam_maps(tr(i),:,:,:,2));
    elseif all_true_labels(tr(i),r) == 0
        lesoNS(id) = lesoNS(id)+1;
        gcamoNS = gcamoNS + squeeze(all_gradcam_maps(tr(i),:,:,:,2));
    end    
end

lesoS = lesoS./sum(all_true_labels(tr(:),r));
gcamoS = gcamoS./sum(all_true_labels(tr(:),r));

lesoNS = lesoNS./(size(all_true_labels,1)-sum(all_true_labels(tr(:),r)));
gcamoNS = gcamoNS./(size(all_true_labels,1)-sum(all_true_labels(tr(:),r)));

template = load_nifti('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz');
tarhdr = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz'); %load header
tarhdr2 = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/nii/halved_halved/M2002_subsamp_subsamp_FIN.nii');
inhdr = template; %load header

% pad
inimg = gcamoS; %load volume
inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
xs = setdiff([1:size(inimgo,1)],torX);
ys = setdiff([1:size(inimgo,2)],torY);
zs = setdiff([1:size(inimgo,3)],torZ);
inimgo(xs,ys,zs) = inimg;
[outhdr,outimg] = nii_reslice_target(tarhdr2,inimgo,tarhdr);
outhdr.fname = [pwd '/gradCAM_Severe_average.nii'];
spm_write_vol(outhdr,outimg);
gzip(outhdr.fname)
delete(outhdr.fname)

inimg = gcamoNS; %load volume
inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
xs = setdiff([1:size(inimgo,1)],torX);
ys = setdiff([1:size(inimgo,2)],torY);
zs = setdiff([1:size(inimgo,3)],torZ);
inimgo(xs,ys,zs) = inimg;
[outhdr,outimg] = nii_reslice_target(tarhdr2,inimgo,tarhdr);
outhdr.fname = [pwd '/gradCAM_Nonsevere_average.nii'];
spm_write_vol(outhdr,outimg);
gzip(outhdr.fname)
delete(outhdr.fname)

tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{tr(i)} '_subsamp_subsamp_FIN.nii']);
tmp.vol = lesoS;
save_nifti(tmp,[pwd '/lesion_Severe_average.nii.gz'])

tmp.vol = lesoNS;
save_nifti(tmp,[pwd '/lesion_Nonsevere_average.nii.gz'])


% then average the grad cam maps

%% Quick t-sne
feat = all_features_test(:,:,2);
feat = reshape(all_features_test,[231,64*20]);

opts.MaxIter = 200;
[Y,~] = tsne(feat,'LearnRate',110,'NumDimensions',2,'Exaggeration',2,'Perplexity',20,'Options',opts);

mp = median(all_predictions');

idx1 = find(mp == 0);
idx2 = find(mp == 1);
idx3 = find(mp == 0.5);

cid = find(mp' == all_true_labels(:,1)==1);
wid = find(mp' == all_true_labels(:,1)==0);

figure
s1 = scatter(Y(intersect(idx1,cid),1),Y(intersect(idx1,cid),2));
s1.MarkerFaceColor = [106/255 44/255 112/255];
s1.MarkerEdgeColor = 'none';
s1.SizeData = 300;
hold on

s11 = scatter(Y(intersect(idx1,wid),1),Y(intersect(idx1,wid),2));
s11.MarkerFaceColor = [106/255 44/255 112/255];
s11.MarkerEdgeColor = [0 0 0];
s11.LineWidth = 5;
s11.SizeData = 300;

s2 = scatter(Y(intersect(idx2,cid),1),Y(intersect(idx2,cid),2));
s2.MarkerFaceColor = [240/255 138/255 93/255];
s2.MarkerEdgeColor = 'none';
s2.SizeData = 300;

s22 = scatter(Y(intersect(idx2,wid),1),Y(intersect(idx2,wid),2));
s22.MarkerFaceColor = [240/255 138/255 93/255];
s22.MarkerEdgeColor = 'none';
s22.SizeData = 300;
s22.MarkerEdgeColor = [0 0 0];
s22.LineWidth = 5;

s3 = scatter(Y(intersect(idx3,wid),1),Y(intersect(idx3,wid),2));
s3.MarkerFaceColor = [184/255 59/255 94/255];
s3.MarkerEdgeColor = 'none';
s3.SizeData = 300;
s3.MarkerEdgeColor = [0 0 0];
s3.LineWidth = 5;
set(gca,'FontSize',22)
ax = gca; ax.LineWidth = 5;
set(gca,'FontSize',25)
set(gcf,'color','w')
grid on

%% Repeat getting stratification...
% remove mystery problem subs
tor = [36 38 74 75 161];
data(:,:,:,tor) = [];
wabClass(tor) = [];
wabClass2(tor) = [];
wabClass2i(tor) = [];

[parts] = quickScatter(Y(:,2),Y(:,1),'useWholeMap',false,'gradient',false,'group',wabClass2i','box',false,'markerAlpha',1,'perfectY',false,'markSz',100,'trendPlot',false,'newFig',true,'varMarkSz',dc);


% new categories only for stratification...
datar = reshape(data,[size(data,1)*size(data,2)*size(data,3),size(data,4)]);
for i = 1:size(datar,2)
    lsz(i) = length(find(datar(:,i) == 4));
end
e = prctile(lsz,[0 33.3 66.6 100]);
[~,~,dc] = histcounts(lsz,e);

[parts] = quickScatter(Y(:,2),Y(:,1),'useWholeMap',false,'gradient',false,'group',wabClass2i','box',false,'markerAlpha',1,'perfectY',false,'markSz',100,'trendPlot',false,'newFig',true,'varMarkSz',dc);
set(gca,'FontSize',22)
ax = gca; ax.LineWidth = 5;
ax.XLim = [-8 6];
hold on

[parts1] = quickScatter(Y(cid,2),Y(cid,1),'defClrMp','viridis','gradient',false,'group',wabClass2i(cid)','box',false,'markerAlpha',1,'perfectY',false,'markSz',100,'trendPlot',false,'newFig',true,'varMarkSz',dc(cid));
hold on
[parts2] = quickScatter(Y(wid,2),Y(wid,1),'defClrMp','viridis','gradient',false,'group',wabClass2i(wid)','box',false,'markerAlpha',1,'perfectY',false,'markSz',100,'trendPlot',false,'newFig',false,'varMarkSz',dc(wid));
parts2.scat.LineWidth = 3;
parts2.scat.MarkerEdgeColor = [0 0 0];

ax = gca;
ax.YTick = [-10:2:8];
ax.YLim = [-10 8];
ax.XLim = [-8 6]
ax.XTick = [-8:2:6];



%% Experiment w/SVR in lieu of CNN (we will take best result)
addpath(genpath('~/Desktop/dnn'))
addpath(genpath('/home/rorden/PycharmProjects/pythonProject'))
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat','all_probs','all_true_labels','all_predictions','all_fold')
load('PyTorchInput_withFolds.mat', 'data', 'wabClass','wabClass2','wabClass2i')

%% Repeat preprocessing for CNN...
% remove mystery problem subs
tor = [36 38 74 75 161];
data(:,:,:,tor) = [];
wabClass(tor) = [];
wabClass2(tor) = [];
wabClass2i(tor) = [];

% new categories only for stratification...
datar = reshape(data,[size(data,1)*size(data,2)*size(data,3),size(data,4)]);
for i = 1:size(datar,2)
    lsz(i) = length(find(datar(:,i) == 4));
end
e = prctile(lsz,[0 33.3 66.6 100]);
[~,~,dc] = histcounts(lsz,e);
yC = wabClass2i * 10 + dc;
[un,~,~] = unique(yC);
counts = histcounts(yC,[un, max(un) + 1]);
id = find(yC == 13);
yC(id) = 12;
id = find(yC == 31);
yC(id) = 32;
id = find(yC == 41);
yC(id) = 43;
id = find(yC == 42);
yC(id) = 43;

scatter2(Y(:,1),Y(:,2),Y(:,3),ones([length(Y),1])*100,[1 0.4 0.3],'filled','MarkerFaceAlpha',0.6);

scatter3(Y(:,1),Y(:,2),Y(:,3),ones([length(Y),1])*100,[1 0.4 0.3],'filled','MarkerFaceAlpha',0.6);
text(Y(:,1)+1,Y(:,2)+1,Y(:,3),labels,'Color','black','FontSize',6);
set(gcf,'color','w');




opts.MaxIter = 1000;
opts.TolFun = 1e-100;
perpTest = [1 5 6 7 8 9 10 11 12 13 14 15 15 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 200 300 400 500 600 700 800 900];
learnTest = [10 50 100 150 200 250 300 350 400 450 500 600 700 800 900 1000];
exTest = [2 3 4 5 6 7 8 9 10 15 20 25];

for i = 1:length(toTest)
    disp([num2str(i) ' of ' num2str(length(toTest))])
    for j = 1:length(learnTest)
        for k = 1:length(exTest)
            disp(num2str(k))
            [Y{i,j,k},loss{i,j,k}] = tsne(keepMetas','Distance','cosine','Perplexity',perpTest(i),'LearnRate',learnTest(j),'Exaggeration',exTest(k),'Options',opts);
            %loss{i,j,k} = loss_tmp;
            %Y{i,j,k} = Y_tmp;
            %     gscat = gscatter(Y(:,1),Y(:,2),Labels3');
            %     title(num2str(toTest(i)),'FontSize',32)
            %     b = gca; legend(b,'off');
            %     pause(1)
        end
    end
end



[sIndex,optimalSPerplexity, optimalKLPerplexity] = evalTsne(perplexity,KLVals,nSamples,'true');




%% Alternative approach
% group lesions by spatial similarity 






% project grad cam maps into that space?



%% stacked models and SVR are in finStack.m



%% FIGURE 2
%% Get class accuracies...
load('optiLDA.mat')
m = mean(f1fs3,2);
[mv,mi] = max(m);
optif1 = f1fs3(mi,:);


addpath(genpath('/Users/alex/Downloads/Violinplot-Matlab-master'))
for j = 1:20
    [fincon{j},finacc(j),finaccraw(j,:),finaccm(j),finprec(j),finrec(j),finf1(j)] = getAccLoss(wabClass,all_preds(:,j),[]);
end
ctmp = cmocean('thermal',13); % f1 score 
cTmp = ctmp([2 4 6 8 10 12],:);
%cTmp2 = cTmp + 0.2;
%idx = find(cTmp2>1);
%cTmp2(idx) = 1;
cTmp2 = cTmp;
tmp = [finaccraw'; finaccm; finf1; finprec; finrec];

figure; vs2 = violinplot(tmp',{'NS accuracy','S accuracy','mean accuracy','F1','precision','recall'},'Bandwidth',0.008,'HalfViolin','right','QuartileStyle','shadow','Width',1.5);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = [{cTmp(i,:)} {cTmp2(i,:)}];
    vs2(i).ViolinPlot.EdgeColor = [0 0 0];
    vs2(i).ViolinPlot2.EdgeColor = [0 0 0];
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp(i,:);
    vs2(i).ScatterPlot.MarkerFaceAlpha = 1;
    vs2(i).ScatterPlot2.MarkerFaceColor = cTmp(i,:);
    vs2(i).ScatterPlot2.MarkerFaceAlpha = 1;
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).ViolinPlot2.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 4;
    vs2(i).BoxPlot.EdgeColor = [0 0 0];
    vs2(i).WhiskerPlot.Color = [0 0 0];
    vs2(i).MeanPlot.LineWidth = 4;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 1];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 200;
    vs2(i).ScatterPlot2.SizeData = 200;
    vs2(i).MedianColor = [1 1 1];
    vs2(i).MedianPlot.MarkerEdgeColor = [0 0 0];
    vs2(i).MedianPlot.LineWidth = 2;
    vs2(i).MedianPlot.SizeData = 100;
    vs2(i).ViolinPlot.FaceAlpha = 0.5;
    vs2(i).WhiskerPlot.LineWidth = 4;
    vs2(i).ViolinPlot2.FaceAlpha = 0.5;
end
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('F1')
view([90 -90])

% now look at overlap between F1 scores and permutations...
i=1;j=1; figure
h2{i,j} = histfit(perms.f1,[],'kernel');
h2{i,j}(1).FaceAlpha = 0;
h2{i,j}(1).EdgeAlpha = 0;
h2{i,j}(2).LineWidth = 10;
h2{i,j}(2).Color = ctmp(1,:);
hold on
p1 = patch(h2{i,j}(2).XData,h2{i,j}(2).YData,'red','FaceColor',ctmp(1,:),'FaceAlpha',0.85);

i = 1; j = 2;
h2{i,j} = histfit(finf1,[],'kernel');
h2{i,j}(1).FaceAlpha = 0;
h2{i,j}(1).EdgeAlpha = 0;
h2{i,j}(2).LineWidth = 10;
h2{i,j}(2).Color = ctmp(7,:);
p2 = patch(h2{i,j}(2).XData,h2{i,j}(2).YData,'red','FaceColor',ctmp(7,:),'FaceAlpha',0.85);
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('Counts')
view([90 -90])


% now do fold-wise
for j = 1:20
    disp(num2str(j))
    for i = 1:6
        disp(num2str(i))
        teid = find(all_fold(:,j) == i);
        trid = setdiff([1:size(all_fold,1)],teid)';
        [conFold{j,i},accFold(j,i),accRawFold(j,i,:),accmFold(j,i),precFold(j,i),recFold(j,i),f1Fold(j,i),~, aucFold(j,i)] = getAccLoss(wabClass(teid),all_preds(teid,j),[],[]);
        [conFoldtr{j,i},accFoldtr(j,i),accRawFoldtr(j,i,:),accmFoldtr(j,i),precFoldtr(j,i),recFoldtr(j,i),f1Foldtr(j,i),~, aucFoldtr(j,i)] = getAccLoss(wabClass(trid),all_preds(trid,j),[]);
    end
end

f1Fold1te = reshape(f1Fold,[1,20*6]);
f1Fold1tr = reshape(f1Foldtr,[1,20*6]);

%% now do violin plot for SVM, SVM+PCA, SVM + ICA
s = load('stacked2.mat');
load('SVR_outputs3_mdl2_fin_withStats.mat', 'yhat2')
for j = 1:20
    [finco2n{j},finacc2(j),finaccraw2(j,:),finaccm2(j),finprec2(j),finrec2(j),finf12(j), ~, finauc2(j)] = getAccLoss(wabClass,yhat2(:,j),[],[]);
end
for j = 1:20
    [~,finacc2(j),finaccraw2(j,:),finaccm2(j),finprec2(j),finrec2(j),finf12(j), ~, finauc2(j)] = getAccLoss(wabClass',yhat2(:,j),[],[]);
end
pca = load('svmPCA.mat','yhatPlus2d');
ica = load('svmPCA_ICA.mat','yhatPlus2d');
for j = 1:14
    [finco2np{j},finacc2p(j),finaccraw2p(j,:),finaccm2p(j),finprec2p(j),finrec2p(j),finf12p(j), ~, finauc2(j)] = getAccLoss(wabClass,pca.yhatPlus2d(:,j),[],[]);
end
finf12p(15:20) = NaN;
finacc2p(15:20) = NaN;
finaccraw2p(15:20,:) = NaN;
finaccm2p(15:20) = NaN;
finprec2p(15:20) = NaN;
finrec2p(15:20) = NaN;
finf12p(15:20) = NaN;
for j = 1:20
    [finco2ni{j},finacc2i(j),finaccraw2i(j,:),finaccm2i(j),finprec2i(j),finrec2i(j),finf12i(j), ~, finauc2(j)] = getAccLoss(wabClass,ica.yhatPlus2d(:,j),[],[]);
end

ctmp = cmocean('thermal',13); % f1 score 
cTmp2 = ctmp([2 2 2 2 4 4 4 4 6 6 6 6 8 8 8 8 10 10 10 10 12 12 12 12],:);

%tmp = [finaccraw(:,1)'; finaccraw(:,2)'; finaccm; finf1; finprec; finrec];
tmp = [finaccraw(:,1)'; finaccraw2(:,1)'; finaccraw2p(:,1)'; finaccraw2i(:,1)'; finaccraw(:,2)'; finaccraw2(:,2)'; finaccraw2p(:,2)'; finaccraw2i(:,2)'; finaccm; finaccm2; finaccm2p; finaccm2i; finf1; finf12; finf12p; finf12i; finprec; finprec2; finprec2p; finprec2i; finrec; finrec2; finrec2p; finrec2i];
nm = {'CNN (NS acc)','SVM (NS acc)','PCA + SVM (NS acc)','ICA + SVM (NS acc)','CNN (S acc)','SVM (S acc)','PCA + SVM (S acc)','ICA + SVM (S acc)','CNN (macc)', 'SVM (macc)','PCA + SVM (macc)','ICA + SVM (macc)', 'CNN (F1)', 'SVM (F1)','PCA + SVM (F1)','ICA + SVM (F1)', 'CNN (prec)', 'SVM (prec)','PCA + SVM (prec)','ICA + SVM (prec)', 'CNN (rec)', 'SVM (rec)','PCA + SVM (rec)','ICA + SVM (rec)'};

figure; vs2 = violinplot(tmp',nm,'Bandwidth',0.008,'HalfViolin','right','QuartileStyle','shadow','Width',0.8);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = [{cTmp2(i,:)} {cTmp2(i,:)}];
    vs2(i).ViolinPlot.EdgeColor = [0 0 0];
    vs2(i).ViolinPlot2.EdgeColor = [0 0 0];
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot.MarkerFaceAlpha = 1;
    vs2(i).ScatterPlot2.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot2.MarkerFaceAlpha = 1;
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).ViolinPlot2.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 4;
    vs2(i).BoxPlot.EdgeColor = [0 0 0];
    vs2(i).WhiskerPlot.Color = [0 0 0];
    vs2(i).MeanPlot.LineWidth = 4;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 1];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 200;
    vs2(i).ScatterPlot2.SizeData = 200;
    vs2(i).MedianColor = [1 1 1];
    vs2(i).MedianPlot.MarkerEdgeColor = [0 0 0];
    vs2(i).MedianPlot.LineWidth = 2;
    vs2(i).MedianPlot.SizeData = 100;
    vs2(i).ViolinPlot.FaceAlpha = 0.5;
    vs2(i).WhiskerPlot.LineWidth = 4;
    vs2(i).ViolinPlot2.FaceAlpha = 0.5;
end
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('F1')
view([90 -90])


%% blending figure...
pte = load('FifthTry_CV_Rep_ COMBINED_gradCAM_training.mat','all_probs');
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat','all_probs');
load('stacked2.mat', 'yhat2p_tr')
load('stacked2.mat', 'yhat2p')
wtsa = [0.2:0.01:1];
for w = 1:length(wtsa)
    wt = wtsa(w);
    for j = 1:20
        disp(num2str(j))
        for i = 1:6
            disp(num2str(i))
            teid = find(all_fold(:,j) == i);
            trid = setdiff([1:size(all_fold,1)],teid)';
            
            %trX = data(trid,:);
            trY = wabClass(trid);
            %teX = data(teid,:);
            
            svmPTr = 1./(1 + exp(-squeeze(yhat2p_tr(trid,j,i,:))));
            svmPTr = svmPTr(:,2);
            svmPTe = 1./(1 + exp(-squeeze(yhat2p(teid,j,:))));
            svmPTe = svmPTe(:,2);
            
            cnnPTr = pte.all_probs(trid,j,i);
            cnnPTe = all_probs(teid,j);
            
            cp = (1-wt) * svmPTe + wt * cnnPTe;
            id1 = find(cp > 0.5);
            id2 = find(cp < 0.5);
            yhW(teid(id1),j) = {'Severe'};
            yhW(teid(id2),j) = {'Non-severe'};
            
        end
        [~,finacc2blend(j,w),finaccraw2blend(j,:,w),finaccm2blend(j,w),finprec2blend(j,w),finrecblend(j,w),finf1blend(j,w), ~, finauc2blend(j,w)] = getAccLoss(wabClass,yhW(:,j),[],[]);
    end
end

figure; p = plot(wtsa,mean(finf1blend,1));

sem = std(finf1blend,1)./sqrt(20);
m = mean(finf1blend,1);

figure; parts = shadedErrorBar(wtsa',mean(finf1blend,1)','lineProps','-brightCyan')
figure; parts.shaded = shadedErrorBar(wtsa,mean(finf1blend,1),[m-sem; m+sem],'lineprops',[0.6353    0.0784    0.1843]);



%% stacking fig
load('stacked2.mat','yhDisc2');
load('svmPlusFin_Trun.mat', 'f1fs422','yhatPlus22')

for j = 1:20
    [finco2stack2{j},finacc2stack2(j),finaccraw2stack2(j,:),finaccm2stack2(j),finprec2stack2(j),finrec2stack2(j),finf12stack2(j), ~, finaucstack2(j)] = getAccLoss(wabClass,yhDisc2(:,j),[],[]);
end
for j = 1:20
    [finco2stack3{j},finacc2stack3(j),finaccraw2stack3(j,:),finaccm2stack3(j),finprec2stack3(j),finrec2stack3(j),finf12stack3(j), ~, finaucstack3(j)] = getAccLoss(wabClass,yhatPlus22(:,j),[],[]);
end
finacc2stack3(20) = NaN;
finaccraw2stack3(20,:) = NaN;
finaccm2stack3(20) = NaN;
finrec2stack3(20) = NaN;
finprec2stack3(20) = NaN;
finf12stack3(20) = NaN;


ctmp = cmocean('thermal',13); % f1 score 
cTmp2 = ctmp([2 2 2 4 4 4 6 6 6 8 8 8 10 10 10 12 12 12],:);

%tmp = [finaccraw(:,1)'; finaccraw(:,2)'; finaccm; finf1; finprec; finrec];
tmp = [finaccraw(:,1)'; finaccraw2stack2(:,1)'; finaccraw2stack3(:,1)'; finaccraw(:,2)'; finaccraw2stack2(:,2)'; finaccraw2stack3(:,2)'; finaccm; finaccm2stack2; finaccm2stack3; finf1; finf12stack2; finf12stack3; finprec; finprec2stack2; finprec2stack3; finrec; finrec2stack2; finrec2stack3];
nm = {'CNN','stacked','CNN features + SVM'};
nm = repmat(nm,[1,6]);

figure; vs2 = violinplot(tmp',nm,'Bandwidth',0.008,'HalfViolin','right','QuartileStyle','shadow','Width',0.8);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = [{cTmp2(i,:)} {cTmp2(i,:)}];
    vs2(i).ViolinPlot.EdgeColor = [0 0 0];
    vs2(i).ViolinPlot2.EdgeColor = [0 0 0];
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot.MarkerFaceAlpha = 1;
    vs2(i).ScatterPlot2.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot2.MarkerFaceAlpha = 1;
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).ViolinPlot2.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 4;
    vs2(i).BoxPlot.EdgeColor = [0 0 0];
    vs2(i).WhiskerPlot.Color = [0 0 0];
    vs2(i).MeanPlot.LineWidth = 4;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 1];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 200;
    vs2(i).ScatterPlot2.SizeData = 200;
    vs2(i).MedianColor = [1 1 1];
    vs2(i).MedianPlot.MarkerEdgeColor = [0 0 0];
    vs2(i).MedianPlot.LineWidth = 2;
    vs2(i).MedianPlot.SizeData = 100;
    vs2(i).ViolinPlot.FaceAlpha = 0.5;
    vs2(i).WhiskerPlot.LineWidth = 4;
    vs2(i).ViolinPlot2.FaceAlpha = 0.5;
end
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('F1')
view([90 -90])

%% compare deep SHAP and grad CAM ++ models
%% stacking fig
load('svmPlusDeepShapr_shaponly.mat','f1fs42dr') % shap only
load('svmPlusDeepShap2.mat', 'f1fs42d') % 
f1fs42d(20) = NaN;

ctmp = cmocean('thermal',13); % f1 score 
cTmp2 = ctmp([4 10],:);

%tmp = [finaccraw(:,1)'; finaccraw(:,2)'; finaccm; finf1; finprec; finrec];
tmp = [f1fs42dr; f1fs42d];
nm = {'Deep SHAP';'GRAD-CAM++'};
figure; vs2 = violinplot(tmp',nm,'Bandwidth',0.008,'HalfViolin','right','QuartileStyle','shadow','Width',0.8);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = [{cTmp2(i,:)} {cTmp2(i,:)}];
    vs2(i).ViolinPlot.EdgeColor = [0 0 0];
    vs2(i).ViolinPlot2.EdgeColor = [0 0 0];
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot.MarkerFaceAlpha = 1;
    vs2(i).ScatterPlot2.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot2.MarkerFaceAlpha = 1;
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).ViolinPlot2.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 4;
    vs2(i).BoxPlot.EdgeColor = [0 0 0];
    vs2(i).WhiskerPlot.Color = [0 0 0];
    vs2(i).MeanPlot.LineWidth = 4;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 1];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 200;
    vs2(i).ScatterPlot2.SizeData = 200;
    vs2(i).MedianColor = [1 1 1];
    vs2(i).MedianPlot.MarkerEdgeColor = [0 0 0];
    vs2(i).MedianPlot.LineWidth = 2;
    vs2(i).MedianPlot.SizeData = 100;
    vs2(i).ViolinPlot.FaceAlpha = 0.5;
    vs2(i).WhiskerPlot.LineWidth = 4;
    vs2(i).ViolinPlot2.FaceAlpha = 0.5;
end
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('F1')
view([90 -90])


%% grad cam bilat, vs lh vs rh...
load('svmPlusFin2lh.mat', 'f1fs42lh')
load('svmPlusFin2rh.mat', 'f1fs42rh')
f1fs42rh(20) = NaN;
ctmp = cmocean('thermal',13); % f1 score 
cTmp2 = ctmp([10 11 8],:);

%tmp = [finaccraw(:,1)'; finaccraw(:,2)'; finaccm; finf1; finprec; finrec];
tmp = [f1fs42d; f1fs42lh; f1fs42rh];
nm = {'Bilateral feature maps','Left hemisphere','Right hemisphere'};
figure; vs2 = violinplot(tmp',nm,'Bandwidth',0.008,'HalfViolin','right','QuartileStyle','shadow','Width',0.8);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = [{cTmp2(i,:)} {cTmp2(i,:)}];
    vs2(i).ViolinPlot.EdgeColor = [0 0 0];
    vs2(i).ViolinPlot2.EdgeColor = [0 0 0];
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot.MarkerFaceAlpha = 1;
    vs2(i).ScatterPlot2.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot2.MarkerFaceAlpha = 1;
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).ViolinPlot2.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 4;
    vs2(i).BoxPlot.EdgeColor = [0 0 0];
    vs2(i).WhiskerPlot.Color = [0 0 0];
    vs2(i).MeanPlot.LineWidth = 4;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 1];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 200;
    vs2(i).ScatterPlot2.SizeData = 200;
    vs2(i).MedianColor = [1 1 1];
    vs2(i).MedianPlot.MarkerEdgeColor = [0 0 0];
    vs2(i).MedianPlot.LineWidth = 2;
    vs2(i).MedianPlot.SizeData = 100;
    vs2(i).ViolinPlot.FaceAlpha = 0.5;
    vs2(i).WhiskerPlot.LineWidth = 4;
    vs2(i).ViolinPlot2.FaceAlpha = 0.5;
end
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('F1')
view([90 -90])
save('temporaryWorkspace.mat')

%% GET shap maps and gradcam maps
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_gradcam_maps')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_gradcam_maps_oppo')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'folds_f1')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_predictions')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_true_labels')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_probs')
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_fold')

%% ROIs for shap
clearvars -except allshap subsOnly
load('Shap_rep2_loaded.mat', 'allshap')
load('SVR_outputs3_mdl2_fin_withStats.mat', 'yhat2')
save('tmp.mat','-v7.3')
for r = 2
    disp(num2str(r))
    for i = 1:length(subsOnly)
        disp(num2str(i))
        lesTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_FIN.nii']);
        periTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_peri.nii.gz']);
        otherTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_FIN_OTHER.nii.gz']);
        
        lesid = find(lesTmp.vol ~= 0);
        periid = find(periTmp.vol ~= 0);
        oid = find(otherTmp.vol ~= 0);
        
        lesTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesionHOMOLOGUE/r' subsOnly{i} '_subsamp_subsamp_FIN_HOMOLOGUE.nii']);
        periTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesionHOMOLOGUE/r' subsOnly{i} '_subsamp_subsamp_peri.nii']);
        otherTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesionHOMOLOGUE/r' subsOnly{i} '_subsamp_subsamp_FIN_OTHER.nii.gz']);
        
        lesidr = find(lesTmp.vol ~= 0);
        periidr = find(periTmp.vol ~= 0);
        oidr = find(otherTmp.vol ~= 0);
        
        in = allshap(:,:,:,i);
        id = find(in < 0);
        in(id) = 0;
  
        tmp = in(:) / norm(in(:), 1);
        
        omat(r,i,1) = mean(tmp(lesid));
        omat(r,i,2) = mean(tmp(periid));
        omat(r,i,3) = mean(tmp(oid));
        omat(r,i,4) = mean(tmp(lesidr));
        omat(r,i,5) = mean(tmp(periidr));
        omat(r,i,6) = mean(tmp(oidr));
        
        omats(r,i,1) = std(tmp(lesid))./sqrt(length(oidr));
        omats(r,i,2) = std(tmp(periid))./sqrt(length(oidr));
        omats(r,i,3) = std(tmp(oid))./sqrt(length(oidr));
        omats(r,i,4) = std(tmp(lesidr))./sqrt(length(oidr));
        omats(r,i,5) = std(tmp(periidr))./sqrt(length(oidr));
        omats(r,i,6) = std(tmp(oidr))./sqrt(length(oidr)); 
        
%         omatsm(r,i,1) = sum(tmp(lesid));
%         omatsm(r,i,2) = sum(tmp(periid));
%         omatsm(r,i,3) = sum(tmp(oid));
%         omatsm(r,i,4) = sum(tmp(lesidr));
%         omatsm(r,i,5) = sum(tmp(periidr));
%         omatsm(r,i,6) = sum(tmp(oidr));
    end
end

% find severe correct predictions
id1 = find(wabClass == 'Severe');
id2 = find(wabClass == 'Non-severe');

id1c = find(ismember(yhat2(id1,2),'Severe'));
id2c = find(ismember(yhat2(id2,2),'Non-severe'));

clear tmp tmps tmpsem
for r = 1:20
    disp(num2str(r))
    tmp(:,1,r) = mean(squeeze(omat(r,id1(id1c),:)),'omitnan');
    tmp(:,2,r) = mean(squeeze(omat(r,id2(id2c),:)),'omitnan');
    tmps(:,1,r) = std(squeeze(omat(r,id1(id1c),:)),'omitnan');
    tmps(:,2,r) = std(squeeze(omat(r,id2(id2c),:)),'omitnan');
    tmpsem(:,1,r) = tmps(:,1,r)/sqrt(length(omat(r,id1(id1c),:)));
    tmpsem(:,2,r) = tmps(:,2,r)/sqrt(length(omat(r,id2(id2c),:)));
end

m = mean(tmp,3);
sem = std(tmp,[],3)./20;

omat2 = squeeze(omat(2,:,:));
% Initialize the arrays to hold the data and labels
values = [];
combinedGroupCondition = [];

% Loop through each condition
for cond = 6:-1:1
    % Extract the values for the current condition for both groups
    valuesCondId2c = omat2(id1(id1c), cond);
    valuesCondId1c = omat2(id2(id2c), cond);
    
    % Append the values to the main array
    values = [values; valuesCondId1c; valuesCondId2c];
    
    % Create combined labels for group and condition, and append them
    combinedGroupCondition = [combinedGroupCondition; ...
                              strcat('Group 1, Cond ', num2str(cond), repmat({' '}, size(valuesCondId1c))); ...
                              strcat('Group 2, Cond ', num2str(cond), repmat({' '}, size(valuesCondId2c)))];
end

% Plotting the boxplot with combined group and condition labels
figure;
b = boxplot(values, combinedGroupCondition, 'factorgap', 5, 'labelverbosity', 'minor', ...
        'colors', repmat('rb', 1, 6), 'factorseparator', 1, 'Notch', 'on');
ylabel('Value');
xlabel('Group and Condition');
title('Box Plot by Group and Condition');
set(gca, 'XTickLabelRotation', 45); % Rotate labels for better visibility
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
h=gca;
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('Normalized SHAP (mean)')

r=2;
fa = squeeze(omat(r,id1(id1c),:));
fa2 = squeeze(omat(r,id2(id2c),:));

for i = 1:6
    nid1 = find(~isnan(fa(:,i)));
    nid2 = find(~isnan(fa2(:,i)));
    [h,p(i,1),ci(i,:),stats{i}] = ttest2(fa(nid1,i),fa2(nid2,i));
    t(i,1) = stats{i}.tstat;   
    d(i,1) = cohensD(fa(nid1,i),fa2(nid2,i));
    df(i,1) = stats{1}.df;
end


% COHENS D FOR TTEST: mean(tmp(7,:) - tmp(8,:)) / std(tmp(7,:) - tmp(8,:))

%% might as well do the ROI analysis first...
load('PyTorchInput_withFolds.mat', 'subsOnly')
subsOnly([36 38 74 75 161]) = [];
for r = 1:20
    disp(num2str(r))
    for i = 1:length(subsOnly)
        disp(num2str(i))
        lesTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_FIN.nii']);
        periTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_peri.nii.gz']);
        otherTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{i} '_subsamp_subsamp_FIN_OTHER.nii.gz']);
        
        lesid = find(lesTmp.vol ~= 0);
        periid = find(periTmp.vol ~= 0);
        oid = find(otherTmp.vol ~= 0);
        
        lesTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesionHOMOLOGUE/r' subsOnly{i} '_subsamp_subsamp_FIN_HOMOLOGUE.nii']);
        periTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesionHOMOLOGUE/r' subsOnly{i} '_subsamp_subsamp_peri.nii']);
        otherTmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesionHOMOLOGUE/r' subsOnly{i} '_subsamp_subsamp_FIN_OTHER.nii.gz']);
        
        lesidr = find(lesTmp.vol ~= 0);
        periidr = find(periTmp.vol ~= 0);
        oidr = find(otherTmp.vol ~= 0);
        
        in = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/' num2str(r) '/' subsOnly{i} '_gradCAM.nii.gz']);
        tmp = in.vol(:) / norm(in.vol(:), 1);
        
%         omat(r,i,1) = mean(tmp(lesid));
%         omat(r,i,2) = mean(tmp(periid));
%         omat(r,i,3) = mean(tmp(oid));
%         omat(r,i,4) = mean(tmp(lesidr));
%         omat(r,i,5) = mean(tmp(periidr));
%         omat(r,i,6) = mean(tmp(oidr));
%         
%         omats(r,i,1) = std(tmp(lesid))./sqrt(length(oidr));
%         omats(r,i,2) = std(tmp(periid))./sqrt(length(oidr));
%         omats(r,i,3) = std(tmp(oid))./sqrt(length(oidr));
%         omats(r,i,4) = std(tmp(lesidr))./sqrt(length(oidr));
%         omats(r,i,5) = std(tmp(periidr))./sqrt(length(oidr));
%         omats(r,i,6) = std(tmp(oidr))./sqrt(length(oidr)); 
        
        omatsm(r,i,1) = sum(tmp(lesid));
        omatsm(r,i,2) = sum(tmp(periid));
        omatsm(r,i,3) = sum(tmp(oid));
        omatsm(r,i,4) = sum(tmp(lesidr));
        omatsm(r,i,5) = sum(tmp(periidr));
        omatsm(r,i,6) = sum(tmp(oidr));
    end
end
save('gradCam_ROI2.mat')

m = mean(tmp,3);
sem = std(tmp,[],3)./20;

% find severe correct predictions
id1 = find(wabClass == 'Severe');
id2 = find(wabClass == 'Non-severe');

% find non-severe correct predictions
id1c = find(all_preds(id1,2) == 'Severe');
id2c = find(all_preds(id2,2) == 'Non-severe');

clear tmp tmps tmpsem
for r = 1:20
    disp(num2str(r))
    tmp(:,1,r) = mean(squeeze(omat(r,id1(id1c),:)),'omitnan');
    tmp(:,2,r) = mean(squeeze(omat(r,id2(id2c),:)),'omitnan');
    tmps(:,1,r) = std(squeeze(omat(r,id1(id1c),:)),'omitnan');
    tmps(:,2,r) = std(squeeze(omat(r,id2(id2c),:)),'omitnan');
    tmpsem(:,1,r) = tmps(:,1,r)/sqrt(length(omat(r,id1(id1c),:)));
    tmpsem(:,2,r) = tmps(:,2,r)/sqrt(length(omat(r,id2(id2c),:)));
end

% interjection
omat2 = squeeze(omat(2,:,:));
% Initialize the arrays to hold the data and labels
values = [];
combinedGroupCondition = [];

% Loop through each condition
for cond = 6:-1:1
    % Extract the values for the current condition for both groups
    valuesCondId2c = omat2(id1(id1c), cond);
    valuesCondId1c = omat2(id2(id2c), cond);
    
    % Append the values to the main array
    values = [values; valuesCondId1c; valuesCondId2c];
    
    % Create combined labels for group and condition, and append them
    combinedGroupCondition = [combinedGroupCondition; ...
                              strcat('Group 1, Cond ', num2str(cond), repmat({' '}, size(valuesCondId1c))); ...
                              strcat('Group 2, Cond ', num2str(cond), repmat({' '}, size(valuesCondId2c)))];
end

% Plotting the boxplot with combined group and condition labels
figure;
b = boxplot(values, combinedGroupCondition, 'factorgap', 5, 'labelverbosity', 'minor', ...
        'colors', repmat('rb', 1, 6), 'factorseparator', 1, 'Notch', 'on');
ylabel('Value');
xlabel('Group and Condition');
title('Box Plot by Group and Condition');
set(gca, 'XTickLabelRotation', 45); % Rotate labels for better visibility
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
h=gca;
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('Normalized GRAD-CAM++ (mean)')

r=2;
fa = squeeze(omat(r,id1(id1c),:));
fa2 = squeeze(omat(r,id2(id2c),:));

for i = 1:6
    nid1 = find(~isnan(fa(:,i)));
    nid2 = find(~isnan(fa2(:,i)));
    [h,p(i,1),ci(i,:),stats{i}] = ttest2(fa(nid1,i),fa2(nid2,i));
    t(i,1) = stats{i}.tstat;   
    d(i,1) = cohensD(fa(nid1,i),fa2(nid2,i));
    df(i,1) = stats{1}.df;
end

% 






fa_combined = [fa; fa2];

% Create a subject ID variable
subject_ID = repmat((1:(size(fa, 1) + size(fa2, 1)))', size(regions, 2), 1);

% Create a group variable
group = [repmat({'Severe'}, size(fa, 1), 1); repmat({'Nonsevere'}, size(fa2, 1), 1)];
group = repmat(group, size(regions, 2), 1);

% Create a region variable
regions = {'Region1', 'Region2', 'Region3', 'Region4', 'Region5', 'Region6'};
region_var = repmat(regions, size(fa_combined, 1), 1);
region_var = region_var';

% Create a table
tbl = table(subject_ID, group, region_var(:), fa_combined(:), ...
            'VariableNames', {'SubjectID', 'Group', 'Region', 'Saliency'});
lme = fitlme(tbl, 'Saliency ~ Group*Region + (1|SubjectID)');
ranova_results = anova(lme);









m = tmp(:,:,2)


m = mean(tmp,3);
sem = std(tmp,[],3)./20;

for i = 1:6
    for j = 1:20
        [h,p(i,j),ci,stats] = ttest2(omat(j,id1(id1c),i),omat(j,id2(id2c),i));
        tv(i,j) = stats.tstat;
    end
end

for i = 1:6
    tmp1 = omat(:,id1(id1c),i);
    tmp1 = tmp1(:);
    tmp2 = omat(:,id2(id2c),i);
    tmp2 = tmp2(:);
    [h,p2(i,j),ci,stats] = ttest2(tmp1,tmp2);
    tv(i,j) = stats.tstat;
end

for i = 1:6
    tmp1 = omat(:,id1(id1c),i);
    tmp1 = tmp1(:);
    tmp2 = omat(:,id2(id2c),i);
    tmp2 = tmp2(:);
    [h,p2(i,j),ci,stats] = ttest2(tmp1,tmp2);
    tv(i,j) = stats.tstat;
end

for j = 1:20
    for i = 1:3
        tmp1 = omat(j,id1(id1c),i);
        tmp1 = tmp1(:);
        tmp2 = omat(j,id1(id1c),i+3);
        tmp2 = tmp2(:);
        [h,p1(i,j),ci,stats] = ttest2(tmp1,tmp2);
        tv1(i,j) = stats.tstat;
    end
end

for j = 1:20
    for i = 1:3
        tmp1 = omat(j,id2(id2c),i);
        tmp1 = tmp1(:);
        tmp2 = omat(j,id2(id2c),i+3);
        tmp2 = tmp2(:);
        [h,p2(i,j),ci,stats] = ttest(tmp1,tmp2);
        tv2(i,j) = stats.tstat;
    end
end

%% let's just do shap now and get on with it....
load('Shaptmp_cont_Rep2Through3.mat', 'mexp2')


%% CLUSTERING STARTS HERE...we need grad-cam maps in low-res...
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_gradcam_maps')
load('PyTorchInput_withFolds.mat', 'data')

% need to split severe and nonsevere into two datasets
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_predictions')
id = find(data ~= 0);
data(id) = 1;
m = mean(data,4);

vid = find(m > 0.99);
for i = 1:231
    tmp = all_gradcam_maps(i,:,:,:,2);
    cam(:,i) = tmp(vid);
end
sid = find(all_predictions(:,2) == 1);
nsid = find(all_predictions(:,2) == 0);
cams = cam(:,sid);
camns = cam(:,nsid);

%[clusterConsensus,itemTracker,clusterTracker,exemplarTracker] = consensusClustering_v4(cams',10,0.7,[3:15],'affinitypropagation','eta',[],[],[],'false','false');
load('clusteringOutput3.mat')

of = '/Users/alex/Documents/DNN/BNT/newestDNNCONFUSION/CNNClustering';
mkdir(of)
[pac,~,~] = consensusSelection_v3(clusterConsensus5,'auc','false','dist','true','cdf','true','pac','true','outDir',of,'outAppend',['AP severe']);
for i = 1:size(clusterConsensus5,3)
    test = clusterConsensus5(:,:,i);
    bc(i,1) = bimodalCoeff(test(:));
    [dip(i,1), dipP(i,1), ~,~] = HartigansDipSignifTest(sort(test),5000);
end

% check whether 5 and 7 produce the same results
[cnnClust,~,~,~,~] = apclusterK(squeeze(clusterConsensus5(:,:,5)),7,0);

% inspect cluster consistency. First, AP should consistently assign
% clusters....
[sanCon,sanItem,sanClustTrack,sanExemplar] = consensusClustering_v4(squeeze(clusterConsensus5(:,:,5)),500,1,[7],'affinitypropagation','raw',[],[],[],'false','false');
tmp = load('clustering_paused.mat');
sanCon = tmp.clusterConsensus;
sanExemplar = tmp.exemplarTracker;
sanClusters = tmp.alignedC(:,1);

[b,i] = sortrows([sanClusters sanExemplar],[1,2]);
sanExemplari(i) = sanExemplar;

ctmp = cmocean('thermal',10); % f1 score
ctmp(1:3,:)= [];
%ctmp(end,:)= [];
figure; g = gscatter([1:length(sanExemplari)],b(:,2),b(:,1),ctmp,[],60);
set(gcf,'color','w'); ax = gca; ax.LineWidth = 2; grid on; set(gca,'TickLength',[0.02 0.02]); set(gca,'FontSize',22)
ax.LineWidth = 4.5;
ax.XAxis.Visible = 'off';

% for fixing other kinds of figs post-hoc
ax = gca; ax.LineWidth = 5; set(gca,'TickLength',[0.02 0.02]); set(gca,'FontSize',22)
set(gca,'TickLength',[0.006 0.006])
ax.XGrid ='on';
ax.XTick = 0:2:28;
ax.XTickLabel = num2cell([0:2:28]+2);

% get into the nonsevere predictions now...
tmp = load('paused_nonsevere_eta_kmeans.mat');
clusterConsensus6 = tmp.clusterConsensus;

[pac,~,~] = consensusSelection_v3(clusterConsensus6,'auc','false','dist','true','cdf','true','pac','true','outDir',of,'outAppend',['AP nonsevere']);
for i = 1:size(clusterConsensus5,3)
    test = clusterConsensus6(:,:,i);
    bc(i,1) = bimodalCoeff(test(:));
    [dip(i,1), dipP(i,1), ~,~] = HartigansDipSignifTest(sort(test),5000);
end
ax = gca; ax.LineWidth = 5; set(gca,'TickLength',[0.02 0.02]); set(gca,'FontSize',22)
set(gca,'TickLength',[0.006 0.006])
ax.XGrid ='on';
ax.XTick = 0:2:28;
ax.XTickLabel = num2cell([0:2:28]+2);

[cnnClust,~,~,~,~] = apclusterK(squeeze(clusterConsensus6(:,:,4)),6,0,4000,400);

% inspect cluster consistency. First, AP should consistently assign
% clusters....
[sanCon,sanItem,sanClustTrack,sanExemplar, sanClusters] = consensusClustering_v4(squeeze(clusterConsensus6(:,:,4)),500,1,[6],'affinitypropagation','raw',[],[],[],'false','false');

[b,i] = sortrows([sanClusters(:, 1) sanExemplar],[1,2]);
sanExemplari(i) = sanExemplar;

ctmp = cmocean('haline',9); % f1 score
ctmp(1:3,:)= [];
%ctmp(end,:)= [];
figure; g = gscatter([1:length(sanExemplari)],b(:,2),b(:,1),ctmp,[],60);
set(gcf,'color','w'); ax = gca; ax.LineWidth = 2; grid on; set(gca,'TickLength',[0.02 0.02]); set(gca,'FontSize',22)
ax.LineWidth = 4.5;
ax.XAxis.Visible = 'off';

%% now regenerate the first set of clusters--whoops
% load('paused.mat')
nonsevereK = sanClusters(:, 1);
tmp = load('clustering_paused.mat');
severeK = tmp.alignedC(:,1);
clear tmp
load('tSNE_fig.mat', 'lsz')
load('temporaryWorkspace.mat', 'subsOnly')

groupMeans = grpstats(lsz(sid), severeK);
[pValue, tbl, stats] = anova1(lsz(sid), severeK);

groupMeans = grpstats(lsz(nsid), nonsevereK);
[pValue, tbl, stats] = anova1(lsz(nsid), nonsevereK);

for i = 1:7
    id2 = find(severeK == i);
    id = find(all_true_labels(id2,2) == all_predictions(id2,2));
    acc(i,1) = length(id)/length(id2);
end
for i = 1:6
    id2 = find(nonsevereK == i);
    id = find(all_true_labels(id2,2) == all_predictions(id2,2));
    acc(i,2) = length(id)/length(id2);
end

accuracies = []; % Array to hold accuracies for each subject
groups = []; % Array to hold group ids corresponding to each accuracy
for i = 1:7
    id2 = find(severeK == i); % Subjects in group i
    accuracy = all_true_labels(id2,2) == all_predictions(id2,2); % Accuracy for each subject in group i
    accuracies = [accuracies; accuracy]; % Append accuracies
    groups = [groups; i * ones(length(id2), 1)]; % Append group ids
end

[pValue, tbl, stats] = anova1(accuracies, groups); % Perform one-way ANOVA

accuracies = []; % Array to hold accuracies for each subject
groups = []; % Array to hold group ids corresponding to each accuracy
for i = 1:7
    id2 = find(nonsevereK == i); % Subjects in group i
    accuracy = all_true_labels(id2,2) == all_predictions(id2,2); % Accuracy for each subject in group i
    accuracies = [accuracies; accuracy]; % Append accuracies
    groups = [groups; i * ones(length(id2), 1)]; % Append group ids
end

[pValue, tbl, stats] = anova1(accuracies, groups); % Perform one-way ANOVA

%% find the exemplars
un = unique(nonsevereK);
for i = 1:length(un)
    id = find(nonsevereK == un(i));
    [mv,mi] = max(sanExemplar(id));
    ex(i,1) = id(mi);
end

un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    [mv,mi] = max( tmp.exemplarTracker(id));
    ex2(i,1) = id(mi);
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))

un = unique(nonsevereK);
for i = 1:length(un)
    id = find(nonsevereK == un(i));
    clusterSubsNonsevere{i} = subsOnly(nsid(id));
end

un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    clusterSubsSevere{i} = subsOnly(sid(id));
end

subsOnly(sid(ex2))
subsOnly(nsid(ex))

%% get similarity matrix 
% severeK nonsevereK
cama = [cams camns];
grp = [severeK; nonsevereK+max(severeK)]
[r,p] = corr(cama);

[~, idx] = sort(grp); % This returns a sorted version of grp and the indices used for sorting
r2 = r(idx, idx); % This reorders the rows and columns of r based on the sorted indices
p2 = p(idx, idx); % This reorders the rows and columns of r based on the sorted indices
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
h=gca; 
h.YAxis.TickLength = [0 0];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
h.YRuler.Axle.Visible = 'off';
h.LineWidth = 4;
axis off
ax = gca;
ax.CLim = [-1 1];

ctmp = cmocean('thermal',10);
colormap(ctmp)

% get within-subject mean r vs outside-subject mean r
for i = 1:7
    id = find(severeK == i);
    id2 = find(severeK ~= i);
    [r,~] = corr(cams(:,id));
    [r2,~] = corr(cams(:,id2));
    r = triu(r,1);
    r2 = triu(r2,1);
    id = find(r == 0);
    r(id) = NaN;
    id2 = find(r == 0);
    r2(id) = NaN;
    
    wc(i,1) = mean(r(:),'omitnan');
    oc(i,1) =  mean(r2(:),'omitnan');
    
    wcsem(i,1) = std(r(:),'omitnan')/sqrt(length(r));
    ocsem(i,1) =  mean(r2(:),'omitnan')/sqrt(length(r2));    
end

for i = 1:6
    id = find(nonsevereK == i);
    id2 = find(nonsevereK ~= i);
    [r,~] = corr(camns(:,id));
    [r2,~] = corr(camns(:,id2));
    r = triu(r,1);
    r2 = triu(r2,1);
    id = find(r == 0);
    r(id) = NaN;
    id2 = find(r == 0);
    r2(id) = NaN;
    
    wc(i,2) = mean(r(:),'omitnan');
    oc(i,2) =  mean(r2(:),'omitnan');
    
    wcsem(i,2) = std(r(:),'omitnan')/sqrt(length(r));
    ocsem(i,2) =  mean(r2(:),'omitnan')/sqrt(length(r2));    
end


%% extract demographics...
load('temporaryWorkspace.mat', 'subsOnly')
d = readtable('/Users/alex/Downloads/CSTARMasterArchivedD_DATA_2023-05-16_1657.csv');
vars = d.Properties.VariableNames;
[c,ia,ib] = intersect(d.record_id_c84280,subsOnly);
d_age = d(ia,:).stroke_age;
d_sex = d(ia,:).sex_v2_v2;

id1 = find(isnan(d_age));
d_age(id1) = d.test_age_polar(ia(id1));
d_sex(id1) = d.gender_polar(ia(id1));

id1 = find(isnan(d_age));

d = readtable('/Users/alex/Downloads/AphasiaLabMasterData_DATA_2023-05-16_1652.csv');
vars = d.Properties.VariableNames';
[c,ia,ib] = intersect(d.study_id,subsOnly);
d_age2 = d(ia,:).age;
d_sex2 = d(ia,:).sex;

id1 = find(isnan(d_age));
d_age(id1) = d_age2(id1);
id1 = find(isnan(d_sex));
d_sex(id1) = d_sex2(id1);

d = readtable('/Users/alex/Documents/CSTAR/MASTER_EXCEL/AphasiaLabMasterData_DATA_LABELS_2022-03-17_1448.csv');
vars = d.Properties.VariableNames';
[c,ia,ib] = intersect(d.StudyID_WhichMustBe_M2XXX_ieM2001_,subsOnly);
d_age2 = d(ia,:).Age_years_;
d_sex2 = d(ia,:).Gender;

id1 = find(isnan(d_age));
d_age(id1) = d_age2(id1);
id1 = find(isnan(d_sex));
d_sex(id1) = d_sex2(id1);

d_age = d(ia,:).stroke_age;
d_sex = d(ia,:).sex_v2_v2;

tmp = load_nifti('/Volumes/Quattro/CNN_FeatureMaps/gCAMppUp/1/M2002_gradCAM.nii.gz');

%% get deep shap...
tmp = load('DeepShapMat_rep2.mat');
id = find(tmp.dps < 0);
tmp.dps(id) = 0;

r = 2;
tr = find(all_true_labels(:,r) == all_predictions(:,r));
gcamoS = zeros(182,218,182);
gcamoNS = zeros(182,218,182);
for i = 1:length(tr)    
    disp(num2str(i))
    if all_true_labels(tr(i),r) == 1
        gcamoS = gcamoS + squeeze(tmp.dps(tr(i),:,:,:));
    elseif all_true_labels(tr(i),r) == 0
        gcamoNS = gcamoNS + squeeze(tmp.dps(tr(i),:,:,:));
    end    
end

tmp.tmp1.vol = gcamoS;
save_nifti(tmp.tmp1,'DeepSHAP_severe.nii.gz')
tmp.tmp1.vol = gcamoNS;
save_nifti(tmp.tmp1,'DeepSHAP_nonsevere.nii.gz')


gcamoS = gcamoS./sum(all_true_labels(tr(:),r));
gcamoNS = gcamoNS./(size(all_true_labels,1)-sum(all_true_labels(tr(:),r)));

template = load_nifti('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz');
tarhdr = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz'); %load header
tarhdr2 = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/nii/halved_halved/M2002_subsamp_subsamp_FIN.nii');
inhdr = template; %load header

% pad
inimg = gcamoS; %load volume
inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
xs = setdiff([1:size(inimgo,1)],torX);
ys = setdiff([1:size(inimgo,2)],torY);
zs = setdiff([1:size(inimgo,3)],torZ);
inimgo(xs,ys,zs) = inimg;
[outhdr,outimg] = nii_reslice_target(tarhdr2,inimgo,tarhdr);
outhdr.fname = [pwd '/gradCAM_Severe_average.nii'];
spm_write_vol(outhdr,outimg);
gzip(outhdr.fname)
delete(outhdr.fname)

inimg = gcamoNS; %load volume
inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
xs = setdiff([1:size(inimgo,1)],torX);
ys = setdiff([1:size(inimgo,2)],torY);
zs = setdiff([1:size(inimgo,3)],torZ);
inimgo(xs,ys,zs) = inimg;
[outhdr,outimg] = nii_reslice_target(tarhdr2,inimgo,tarhdr);
outhdr.fname = [pwd '/gradCAM_Nonsevere_average.nii'];
spm_write_vol(outhdr,outimg);
gzip(outhdr.fname)
delete(outhdr.fname)

tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{tr(i)} '_subsamp_subsamp_FIN.nii']);
tmp.vol = lesoS;
save_nifti(tmp,[pwd '/lesion_Severe_average.nii.gz'])

tmp.vol = lesoNS;
save_nifti(tmp,[pwd '/lesion_Nonsevere_average.nii.gz'])


%% correlate deep shap x shap, shap x grad, deep shap x grad
tmp = load('DeepShapMat_lowres_fixed_test.mat');
load('FifthTry_CV_Rep_COMBINED_gradCAM.mat', 'all_gradcam_maps')
grad = squeeze(all_gradcam_maps(:,:,:,:,2));

load('Shaptmp_cont_Rep2Through3.mat', 'mexp2')
shapSVM = sum(mexp2{2},3,'omitnan');
shapCNN = tmp.dpsl;
shapCNN = squeeze(shapCNN(:,:,2));

% reshape deep shap
for i = 1:231
    inimg = zeros(17,22,19);
    s = setdiff([1:length(inimg(:))],tmp.zid);
    inimg(s) = tmp.dpsl(:,i,2);
    shapCNN(i,:) = inimg;
    
%     inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
%     xs = setdiff([1:size(inimgo,1)],torX);
%     ys = setdiff([1:size(inimgo,2)],torY);
%     zs = setdiff([1:size(inimgo,3)],torZ);
%     
%     inimgo(xs,ys,zs) = inimg;%flip(tmp1,1);
%     
end

for i = 1:231
    inimg = zeros(17,22,19);
    s = setdiff([1:length(inimg(:))],tmp.zid);
    tmp2 = squeeze(grad(i,:,:,:));
    gradCNN(:,i) = tmp2(s);
   
%     inimgo = zeros(size(inimg,1)+length(torX),size(inimg,2)+length(torY),size(inimg,3)+length(torZ));
%     xs = setdiff([1:size(inimgo,1)],torX);
%     ys = setdiff([1:size(inimgo,2)],torY);
%     zs = setdiff([1:size(inimgo,3)],torZ);
%     
%     inimgo(xs,ys,zs) = inimg;%flip(tmp1,1);
%     
end

id = find(shapCNN < 0);
shapCNN(id) = 0;
id = find(shapSVM < 0);
shapSVM(id) = 0;
for i = 1:231
%     [r(i,1),~] = corr(gradCNN(:,i),shapCNN(:,i));
%     [r(i,2),~] = corr(gradCNN(:,i),shapSVM(:,i));
%     [r(i,3),~] = corr(shapCNN(:,i),shapSVM(:,i));
    
    [r(i,1),~] = etaSquared2_fast(gradCNN(:,i)',shapCNN(:,i)');
    [r(i,2),~] = etaSquared2_fast(gradCNN(:,i)',shapSVM(:,i)');
    [r(i,3),~] = etaSquared2_fast(shapCNN(:,i)',shapSVM(:,i)');
    
end


id1 = find(all_true_labels(:,2) == all_predictions(:,2));
load('SVR_outputs3_mdl2_fin_withStats.mat', 'yhat2')
tmp2 = yhat2(:,2);
id3 = find(ismember(tmp2,'Severe'));
id4 = find(ismember(tmp2,'Non-severe'));
yh2(id3) = 1;
yh2(id4) = 0;
id2 = find(all_true_labels(:,2) == yh2');
[c,~,~] = intersect(id1,id2);

r2 = r(c,:);
r3=atanh(r2);
m = mean(r3);
sd = std(r3);

[h,p,ci,stats] = ttest(r3(:,1),r3(:,3));
mean(r3(:,1)) - mean(r3(:,3)) / std(r3(:,1) - r3(:,3))
[h2,p2,ci2,stats2] = ttest(r3(:,2),r3(:,3));
mean(r3(:,2)) - mean(r3(:,3)) / std(r3(:,2) - r3(:,3))

%% tuning linear vs nonlinear functions...
load('SVR_outputs2.mat', 'f1')
load('SVR_outputs2.mat', 'f12')
load('SVR_outputs2.mat', 'f13')

% get pca + svm and standard svm...
ctmp = cmocean('thermal',6); % f1 score 
cTmp2 = ctmp(2:end-1,:);
ftmp = [f1 f1];

tmp = [finf12 ;finf12i; ftmp; f13];
%tmp = [finaccraw(:,1)'; finaccraw2(:,1)'; finaccraw2p(:,1)'; finaccraw2i(:,1)'; finaccraw(:,2)'; finaccraw2(:,2)'; finaccraw2p(:,2)'; finaccraw2i(:,2)'; finaccm; finaccm2; finaccm2p; finaccm2i; finf1; finf12; finf12p; finf12i; finprec; finprec2; finprec2p; finprec2i; finrec; finrec2; finrec2p; finrec2i];
%nm = {'CNN (NS acc)','SVM (NS acc)','PCA + SVM (NS acc)','ICA + SVM (NS acc)','CNN (S acc)','SVM (S acc)','PCA + SVM (S acc)','ICA + SVM (S acc)','CNN (macc)', 'SVM (macc)','PCA + SVM (macc)','ICA + SVM (macc)', 'CNN (F1)', 'SVM (F1)','PCA + SVM (F1)','ICA + SVM (F1)', 'CNN (prec)', 'SVM (prec)','PCA + SVM (prec)','ICA + SVM (prec)', 'CNN (rec)', 'SVM (rec)','PCA + SVM (rec)','ICA + SVM (rec)'};
nm = {'linear svm'; 'linear svm (w/ICA)';'tuned kernel svm';'tuned kernel svm (w/ICA)'};

tmp2 = [finf12; finf12i];
tmp2(3,:) = [0.589743589743590	0.604026845637584	0.596026490066225	0.596026490066225	0.606451612903226	0.581081081081081	0.634146341463415	0.617449664429530	0.610389610389610	0.586666666666667 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN];
tmp2(4,:) = f13;
tmp2(5,:) = [0.611764705882353	0.616279069767442	0.662790697674419	0.685082872928177	0.648351648351648	0.670807453416149	0.607142857142857	0.641711229946524	0.647727272727273	0.625000000000000	0.647398843930636	0.674157303370787	0.640000000000000	0.666666666666667	0.630303030303030	0.654761904761905	0.640000000000000	0.682926829268293	0.655737704918033	0.616352201257862];

nm2 = {'linear svm'; 'linear svm (w/ICA)';'tuned kernel svm';'tuned kernel svm (w/ICA)';'nonlinear svm'};

figure; vs2 = violinplot(tmp2',nm2,'Bandwidth',0.008,'HalfViolin','right','QuartileStyle','shadow','Width',0.8);
for i = 1:length(vs2) %vs2(1).ViolinColor
    vs2(i).ViolinColor = [{cTmp2(i,:)} {cTmp2(i,:)}];
    vs2(i).ViolinPlot.EdgeColor = [0 0 0];
    vs2(i).ViolinPlot2.EdgeColor = [0 0 0];
    vs2(i).ScatterPlot.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot.MarkerFaceAlpha = 1;
    vs2(i).ScatterPlot2.MarkerFaceColor = cTmp2(i,:);
    vs2(i).ScatterPlot2.MarkerFaceAlpha = 1;
    vs2(i).ViolinPlot.LineWidth = 5;
    vs2(i).ViolinPlot2.LineWidth = 5;
    vs2(i).BoxPlot.LineWidth = 4;
    vs2(i).BoxPlot.EdgeColor = [0 0 0];
    vs2(i).WhiskerPlot.Color = [0 0 0];
    vs2(i).MeanPlot.LineWidth = 4;
    vs2(i).ShowMean = 1;
    vs2(i).MeanPlot.LineStyle = '-';
    vs2(i).MeanPlot.Color = [vs2(i).MeanPlot.Color 1];
    vs2(i).MedianPlot.SizeData = 10;
    vs2(i).ScatterPlot.SizeData = 200;
    vs2(i).ScatterPlot2.SizeData = 200;
    vs2(i).MedianColor = [1 1 1];
    vs2(i).MedianPlot.MarkerEdgeColor = [0 0 0];
    vs2(i).MedianPlot.LineWidth = 2;
    vs2(i).MedianPlot.SizeData = 100;
    vs2(i).ViolinPlot.FaceAlpha = 0.5;
    vs2(i).WhiskerPlot.LineWidth = 4;
    vs2(i).ViolinPlot2.FaceAlpha = 0.5;
end
set(gca,'FontSize',15)
set(gca,'LineWidth',1.5)
box off
set(gcf,'color','w');
% ylim([-0.35 0.45])
% yticks(-0.35:0.05:0.45)
xtickangle(0)
h=gca; 
h.YAxis.TickLength = [0.01 0.01];
h.XAxis.TickLength = [0 0];
h.XRuler.Axle.Visible = 'off';
%xticklabels({'F1 score','Non-severe accuracy','Severe accuracy','Mean class accuracy'});
%set(gcf, 'Position',  [800, 1000, 1200, 500])
h.LineWidth = 4;
h.YGrid = 'on';
h.GridAlpha = 0.2;
h.GridLineStyle = ':';
ylabel('F1')
view([90 -90])

[h2,p2,ci2,stats2] = ttest(r3(:,2),r3(:,3))

tmp1 = [0.589743589743590	0.604026845637584	0.596026490066225	0.596026490066225	0.606451612903226	0.581081081081081	0.634146341463415	0.617449664429530	0.610389610389610	0.586666666666667];



%% Decode cluster maps...

% import neurosynth....
tmp = load_nifti(['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz']);
vid = find(tmp.vol ~= 0);
d = dir('/Users/alex/Desktop/HACKATHON/ns/*.nii.gz');
for i = 1:length(d)
    idx = strfind(d(i).name,'__');
    tmp = load_nifti(['/Users/alex/Desktop/HACKATHON/ns/' d(i).name]);
    ns(:,i) = tmp.vol(vid);
    nsnm{i} = d(i).name(idx+2:end-7);
end

% import cluster maps...
%% find the exemplars
un = unique(nonsevereK);
for i = 1:length(un)
    id = find(nonsevereK == un(i));
    [mv,mi] = max(sanExemplar(id));
    ex(i,1) = id(mi);
end

un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    [mv,mi] = max(exemplarTracker(id));
    ex2(i,1) = id(mi);
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))

%tarhdr2 = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/nii/halved_halved/M2002_subsamp_subsamp_FIN.nii');
%tarhdr = spm_vol('/Users/alex/Documents/DNN/BNT/dnn/out/FAST_1mm/MNI152_T1_1mm_brain_seg.nii.gz');
template = spm_vol(['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz']);
for i = 1:length(ex2)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{sid(ex2(i))} '_subsamp_subsamp_FIN.nii']);    
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exS(:,i) = outimg(vid);
end

for i = 1:length(ex)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{nsid(ex(i))} '_subsamp_subsamp_FIN.nii']);    
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exNS(:,i) = outimg(vid);
end

[rExS,pExS] = corr(exS,ns,'rows','pairwise');
[rExNS,pExNS] = corr(exNS,ns,'rows','pairwise');
id = find(pExS == 0);
pExS(id) = 1.1e-16;
id = find(pExNS == 0);
pExNS(id) = 1.1e-16;

[pExS2, ~]=bonf_holm(pExS,0.05);
[pExNS2, ~]=bonf_holm(pExNS,0.05);

id1 = find(rExS > 0.2); % & pExS > 0.05);
max(pExS2(id1))
id1 = find(rExNS > 0.2); % & pExS > 0.05);
max(pExNS2(id1))

% sort these....
for i = 1:7
    [sI,sV] = sort(rExS(i,:),'descend');
    sevSort(:,i) = nsnm(sV);
    sevSortR(:,i) = sI;
    sevSortP(:,i) = pExS2(i,sV);
end
for i = 1:6
    [sI,sV] = sort(rExNS(i,:),'descend');
    nonsevSort(:,i) = nsnm(sV);
    nonsevSortR(:,i) = sI;
    nonsevSortP(:,i) = pExNS2(i,sV);
end




%% just combine everything...
sevnm = sevSort(:);
sevr = sevSortR(:);
sevp = sevSortP(:);
sevclass(1:200,1) = 1;
sevclass(201:400,1) = 2;
sevclass(401:600,1) = 3;
sevclass(601:800,1) = 4;
sevclass(801:1000,1) = 5;
sevclass(1001:1200,1) = 6;
sevclass(1201:1400,1) = 7;

nsevnm = nonsevSort(:);
nsevr = nonsevSortR(:);
nsevp = nonsevSortP(:);
nsevclass(1:200) = 1;
nsevclass(201:400) = 2;
nsevclass(401:600) = 3;
nsevclass(601:800) = 4;
nsevclass(801:1000) = 5;
nsevclass(1001:1200) = 6;

% okay, we will extract all features above 0.2...then, word cloud
for i = 1:7
    id = find(sevSortR(:,i) > 0.2);
    tmp = sevSort(id,i);
    tmpR = sevSortR(id,i);
    tmpP = sevSortP(id,i);
    for j = 2:length(tmp)
        id = strfind(tmp{j},'_z_desc-specificity');
        tmp{j} = tmp{j}(1:id-1);
    end
    tmp2 = cell2table(tmp);
    tmp2.freq = tmpR;
   % tmp2.freq2 = tmpP;
    %tmp2([1 2 3 4 7 11 13],:) = [];
    %tmp2([3 10],:) = [];
    %tmp2([1 2 3 5 9 10 20 32 31],:) = [];
    %tmp2([1],:) = [];
    %tmp2([],:) = [];
    %tmp2([1 13 21 25 39 44 46],:) = [];
    tmp2([4 11],:) = [];
    figure; wordcloud(tmp2,'tmp','freq')
end
for i = 1:6
    id = find(nonsevSortR(:,i) > 0.2);
    tmp = nonsevSort(id,i);
    tmpR = nonsevSortR(id,i);
    for j = 2:length(tmp)
        id = strfind(tmp{j},'_z_desc-specificity');
        tmp{j} = tmp{j}(1:id-1);
    end
    tmp2 = cell2table(tmp);
    tmp2.freq = tmpR;
    %tmp2([8 9 11 26 30],:) = [];
    %tmp2([46 16 10 5],:) = [];
    %tmp2([],:) = [];
    %tmp2([1 2 18 20],:) = [];
    %tmp2([6 7 15 26 32],:) = [];
    tmp2([3 5],:) = [];
    figure; wordcloud(tmp2,'tmp','freq')
end

% get all of the top ones...
for i = 1:7
    id = find(sevSortR(:,i) > 0.2);
    tmp = sevSort(id,i);
    tmpR = sevSortR(id,i);
    tmpP = sevSortP(id,i);
    for j = 2:length(tmp)
        id = strfind(tmp{j},'_z_desc-specificity');
        tmp{j} = tmp{j}(1:id-1);
    end
    tmp2 = cell2table(tmp);
    sortRCell{i} = tmp2;
end
for i = 1:6
    id = find(nonsevSortR(:,i) > 0.2);
    tmp = nonsevSort(id,i);
    tmpR = nonsevSortR(id,i);
    for j = 2:length(tmp)
        id = strfind(tmp{j},'_z_desc-specificity');
        tmp{j} = tmp{j}(1:id-1);
    end
    tmp2 = cell2table(tmp);
    sortRCellNS{i} = tmp2;
end


%% for the bonfer.
clear all
orig = readtable('/Users/alex/Downloads/SupplementaryData11.xlsx');
new = readtable('~/Desktop/toFixTable.xlsx');
new.Topic = cellfun(@(x) strrep(x, '''', ''), new.Topic, 'UniformOutput', false);
new.Topic = cellfun(@(x) strrep(x, '_z_desc-specificity', ''), new.Topic, 'UniformOutput', false);
orig.Topic = cellfun(@(x) strrep(x, '''', ''), orig.Topic, 'UniformOutput', false);
orig.Topic = cellfun(@(x) strrep(x, '}', ''), orig.Topic, 'UniformOutput', false);
new.Label = cellfun(@(x) strrep(x, 'Non-severe', 'Nonsevere'), new.Label, 'UniformOutput', false);
orig.Topic = cellfun(@(x) strrep(x, '_z_desc-specificity', ''), orig.Topic, 'UniformOutput', false);

for i = 1:size(orig,1)
    subset = new(ismember(new.Label,orig.Label{i}) & new.Cluster == orig.Cluster(i) & ismember(new.Topic,orig.Topic{i}) ,:);
    if subset.CorrelationCoefficient == orig.CorrelationCoefficient(i)
        tf(i,1) = 1;
        toAdd(i,1) = subset.P_value;
        delta(i,1) = 0;
    else
        tf(i,1) = 0;
        %toAdd(i,1) = NaN;
        toAdd(i,1) = subset.P_value;
        if orig.CorrelationCoefficient(i) > 0.2 | subset.CorrelationCoefficient > 0.2
            delta(i,1) = orig.CorrelationCoefficient(i) - subset.CorrelationCoefficient;
        else
            delta(i,1) = 0;
        end
    end
end




sevnm = sevSort(:);
sevr = sevSortR(:);
sevp = sevSortP(:);
sevclass(1:200,1) = 1;
sevclass(201:400,1) = 2;
sevclass(401:600,1) = 3;
sevclass(601:800,1) = 4;
sevclass(801:1000,1) = 5;
sevclass(1001:1200,1) = 6;
sevclass(1201:1400,1) = 7;

nsevnm = nonsevSort(:);
nsevr = nonsevSortR(:);
nsevp = nonsevSortP(:);
nsevclass(1:200) = 1;
nsevclass(201:400) = 2;
nsevclass(401:600) = 3;
nsevclass(601:800) = 4;
nsevclass(801:1000) = 5;
nsevclass(1001:1200) = 6;


%% take 2--
for i = 1:7
    [sI,sV] = sort(rExS(i,:),'descend');
    sevSort(:,i) = nsnm(sV);
    sevSortR(:,i) = sI;
    sevSortP(:,i) = pExS(i,sV);
end


%% WTF is going on....this is the command history from before...
load('paused.mat')
nonsevereK = sanClusters(:, 1);
severeK = tmp.alignedC(:,1);
clear tmp
load('tSNE_fig.mat', 'lsz')
load('temporaryWorkspace.mat', 'subsOnly')
for i = 1:7
    id2 = find(severeK == i);
    id = find(all_true_labels(id2,2) == all_predictions(id2,2));
    acc(i,1) = length(id)/length(id2);
end
for i = 1:6
    id2 = find(nonsevereK == i);
    id = find(all_true_labels(id2,2) == all_predictions(id2,2));
    acc(i,2) = length(id)/length(id2);
end
load('temporaryWorkspace.mat', 'all_true_labels')
for i = 1:7
    id2 = find(severeK == i);
    id = find(all_true_labels(id2,2) == all_predictions(id2,2));
    acc(i,1) = length(id)/length(id2);
end
for i = 1:6
    id2 = find(nonsevereK == i);
    id = find(all_true_labels(id2,2) == all_predictions(id2,2));
    acc(i,2) = length(id)/length(id2);
end
severeK
all_true_labels
un = unique(nonsevereK);
for i = 1:length(un)
    id = find(nonsevereK == un(i));
    [mv,mi] = max(sanExemplar(id));
    ex(i,1) = id(mi);
end
un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    [mv,mi] = max(exemplarTracker(id));
    ex2(i,1) = id(mi);
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))
tmp = load('clustering_paused.mat');
tmp
severeK
subsOnly
un = unique(nonsevereK);
for i = 1:length(un)
    id = find(nonsevereK == un(i));
    [mv,mi] = max(sanExemplar(id));
    ex(i,1) = id(mi);
end
un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    [mv,mi] = max( tmp.exemplarTracker(id));
    ex2(i,1) = id(mi);
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))
un = unique(nonsevereK);
for i = 1:length(un)
    id = find(nonsevereK == un(i));
    clusterSubsNonsevere{i} = subsOnly(nsid(id));
end
un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    clusterSubsSevere{i} = subsOnly(sid(id));
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))
shapCNN
tmp = load_nifti(['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz']);
vid = find(tmp.vol ~= 0);
d = dir('/Users/alex/Desktop/HACKATHON/ns/*.nii.gz');
for i = 1:length(d)
    idx = strfind(d(i).name,'__');
    tmp = load_nifti(['/Users/alex/Desktop/HACKATHON/ns/' d(i).name]);
    ns(:,i) = tmp.vol(vid);
    nsnm{i} = d(i).name(idx+2:end-7);
end
un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    [mv,mi] = max( tmp.exemplarTracker(id));
    ex2(i,1) = id(mi);
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))
tmp
tmp = load('clustering_paused.mat');
un = unique(severeK);
for i = 1:length(un)
    id = find(severeK == un(i));
    [mv,mi] = max( tmp.exemplarTracker(id));
    ex2(i,1) = id(mi);
end
subsOnly(sid(ex2))
subsOnly(nsid(ex))
template = spm_vol(['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz']);
for i = 1:length(ex2)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{sid(ex2(i))} '_subsamp_subsamp_FIN.nii']);
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exS(:,i) = outimg(vid);
end
for i = 1:length(ex)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{nsid(ex(i))} '_subsamp_subsamp_FIN.nii']);
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exNS(:,i) = outimg(vid);
end
[rExS,pExS] = corr(exS,ns,'rows','pairwise');
[rExNS,pExNS] = corr(exNS,ns,'rows','pairwise');
addpath(genpath('/Users/alex/Documents/CSTAR/scripts/includesASL/neuro/spm12'))
template = spm_vol(['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz']);
for i = 1:length(ex2)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{sid(ex2(i))} '_subsamp_subsamp_FIN.nii']);
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exS(:,i) = outimg(vid);
end
for i = 1:length(ex)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{nsid(ex(i))} '_subsamp_subsamp_FIN.nii']);
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exNS(:,i) = outimg(vid);
end
[rExS,pExS] = corr(exS,ns,'rows','pairwise');
[rExNS,pExNS] = corr(exNS,ns,'rows','pairwise');
addpath(genpath('/Users/alex/Documents/CSTAR/scripts/NiiStat-master'))
template = spm_vol(['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz']);
for i = 1:length(ex2)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{sid(ex2(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{sid(ex2(i))} '_subsamp_subsamp_FIN.nii']);
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exS(:,i) = outimg(vid);
end
for i = 1:length(ex)
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    tmphdr = spm_vol(['/Volumes/Quattro/CNN_FeatureMaps/gCAMpp/2/' subsOnly{nsid(ex(i))} '_gradCAM.nii.gz']);
    [outhdr,outimg] = nii_reslice_target(tmphdr,tmp.vol,template);
    tmp = load_nifti(['/Volumes/Quattro/CNN_FeatureMaps/halved_halvedrLesion/r' subsOnly{nsid(ex(i))} '_subsamp_subsamp_FIN.nii']);
    [outhdr2,outimg2] = nii_reslice_target(tmphdr,tmp.vol,template);
    idx = find(outimg2 ~= 0);
    outimg(idx) = NaN;
    exNS(:,i) = outimg(vid);
end
[rExS,pExS] = corr(exS,ns,'rows','pairwise');
[rExNS,pExNS] = corr(exNS,ns,'rows','pairwise');
for i = 1:7
    [sI,sV] = sort(rExS(i,:),'descend');
    sevSort(:,i) = nsnm(sV);
    sevSortR(:,i) = sI;
end
for i = 1:6
    [sI,sV] = sort(rExNS(i,:),'descend');
    nonsevSort(:,i) = nsnm(sV);
    nonsevSortR(:,i) = sI;
end
i = 1
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
%tmp2([1 2 3 4 7 11 13],:) = [];
%tmp2([3 10],:) = [];
%tmp2([1 2 3 5 9 10 20 32 31],:) = [];
%tmp2([1],:) = [];
%tmp2([],:) = [];
%tmp2([1 13 21 25 39 44 46],:) = [];
tmp2([4 11],:) = [];
figure; wordcloud(tmp2,'tmp','freq')
i = 2
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
%tmp2([1 2 3 4 7 11 13],:) = [];
%tmp2([3 10],:) = [];
%tmp2([1 2 3 5 9 10 20 32 31],:) = [];
%tmp2([1],:) = [];
%tmp2([],:) = [];
%tmp2([1 13 21 25 39 44 46],:) = [];
tmp2([4 11],:) = [];
figure; wordcloud(tmp2,'tmp','freq')

for i = 1:7
    id = find(sevSortR(:,i) > 0.2);
    tmp = sevSort(id,i);
    tmpR = sevSortR(id,i);
    for j = 2:length(tmp)
        id = strfind(tmp{j},'_z_desc-specificity');
        tmp{j} = tmp{j}(1:id-1);
    end
    tmp2 = cell2table(tmp);
    tmp2.freq = tmpR;
    %tmp2([1 2 3 4 7 11 13],:) = [];
    %tmp2([3 10],:) = [];
    %tmp2([1 2 3 5 9 10 20 32 31],:) = [];
    %tmp2([1],:) = [];
    %tmp2([],:) = [];
    %tmp2([1 13 21 25 39 44 46],:) = [];
    tmp2([4 11],:) = [];
    figure; wordcloud(tmp2,'tmp','freq')
end
i
id = find(sevSortR(:,i) > 0.2)
tmp = sevSort(id,i)
tmpR = sevSortR(id,i)
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp)
tmp2.freq = tmpR
tmp2([3 10],:) = [];
tmp2
i = 1
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2([1 2 3 4 7 11 13],:) = [];
tmp2
sI
sV
for i = 1:7
    [sI,sV] = sort(rExS(i,:),'descend');
    sevSort(:,i) = nsnm(sV);
    sevSortR(:,i) = sI;
    sevSortP(:,i) = pExS(sV);
end
for i = 1:6
    [sI,sV] = sort(rExNS(i,:),'descend');
    nonsevSort(:,i) = nsnm(sV);
    nonsevSortR(:,i) = sI;
    nonsevSortP(:,i) = pExNS(sV);
end
i = 1
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.freq2 = tmpP;
tmp2([1 2 3 4 7 11 13],:) = [];
tmp2
pExS
tmp2.freq2
i = 1
[sI,sV] = sort(rExS(i,:),'descend')
sI
sevSort(:,i) = nsnm(sV)
sevSortR(:,i) = sI;
sevSortP(:,i) = pExS(sV)
sevSortP(:,i)
tmp2
isequal(sI,rExS(sV))
rExS(sV)
i = 1
[sI,sV] = sort(rExS(i,:),'descend');
sI
sV
size(pExS)


%% this is the right part of the above---
id = find(pExS == 0);
pExS(id) = 1.1e-16;
id = find(pExNS == 0);
pExNS(id) = 1.1e-16;
[pExS2, ~]=bonf_holm(pExS,0.05);
[pExNS2, ~]=bonf_holm(pExNS,0.05);

for i = 1:7
    [sI,sV] = sort(rExS(i,:),'descend');
    sevSort(:,i) = nsnm(sV);
    sevSortR(:,i) = sI;
    sevSortP(:,i) = pExS2(i,sV);
end
for i = 1:6
    [sI,sV] = sort(rExNS(i,:),'descend');
    nonsevSort(:,i) = nsnm(sV);
    nonsevSortR(:,i) = sI;
    nonsevSortP(:,i) = pExNS2(i,sV);
end

i = 1
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2.freq2 = tmpP;
tmp2([1 2 3 4 7 11 13],:) = [];
tmp2
i = 3
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([1 2 3 5 9 10 20 32 31],:) = [];
tmp2
clear tmp2
i = 4
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([1],:) = [];
tmp2
i = 5
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([],:) = [];
tmp2
i = 6
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([1 13 21 25 39 44 46],:) = []
i = 7
id = find(sevSortR(:,i) > 0.2);
tmp = sevSort(id,i);
tmpR = sevSortR(id,i);
tmpP = sevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
% tmp2.freq2 = tmpP;
tmp2([4 11],:) = []
i = 1
id = find(nonsevSortR(:,i) > 0.2);
tmp = nonsevSort(id,i);
tmpR = nonsevSortR(id,i);
tmpP = nonsevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([8 9 11 26 30],:) = [];
tmp2
i = 2
id = find(nonsevSortR(:,i) > 0.2);
tmp = nonsevSort(id,i);
tmpR = nonsevSortR(id,i);
tmpP = nonsevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([46 16 10 5],:) = []
i
i = 3
id = find(nonsevSortR(:,i) > 0.2);
tmp = nonsevSort(id,i);
tmpR = nonsevSortR(id,i);
tmpP = nonsevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([],:) = []
i = 4
id = find(nonsevSortR(:,i) > 0.2);
tmp = nonsevSort(id,i);
tmpR = nonsevSortR(id,i);
tmpP = nonsevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([1 2 18 20],:) = []
i = 5
id = find(nonsevSortR(:,i) > 0.2);
tmp = nonsevSort(id,i);
tmpR = nonsevSortR(id,i);
tmpP = nonsevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
tmp2([6 7 15 26 32],:) = []
i
i = 6
id = find(nonsevSortR(:,i) > 0.2);
tmp = nonsevSort(id,i);
tmpR = nonsevSortR(id,i);
tmpP = nonsevSortP(id,i);
for j = 2:length(tmp)
    id = strfind(tmp{j},'_z_desc-specificity');
    tmp{j} = tmp{j}(1:id-1);
end
tmp2 = cell2table(tmp);
tmp2.freq = tmpR;
tmp2.p = tmpP;
%tmp2([8 9 11 26 30],:) = [];
%tmp2([46 16 10 5],:) = [];
%tmp2([],:) = [];
%tmp2([1 2 18 20],:) = [];
%tmp2([6 7 15 26 32],:) = [];
tmp2([3 5],:) = []
load('SHAP_ttests.mat')
id1 = find(wabClass == 'Severe');
id2 = find(wabClass == 'Non-severe');
id1c = find(ismember(yhat2(id1,2),'Severe'));
id2c = find(ismember(yhat2(id2,2),'Non-severe'));
load('gradCam_ROI2.mat', 'wabClass')
id1 = find(wabClass == 'Severe');
id2 = find(wabClass == 'Non-severe');
id1c = find(ismember(yhat2(id1,2),'Severe'));
id2c = find(ismember(yhat2(id2,2),'Non-severe'));
load('SVR_outputs3_mdl2_fin_withStats.mat', 'yhat2')
id1 = find(wabClass == 'Severe');
id2 = find(wabClass == 'Non-severe');
id1c = find(ismember(yhat2(id1,2),'Severe'));
id2c = find(ismember(yhat2(id2,2),'Non-severe'));
id1(id1c)
length(id1(id1c))
length(id2(id2c))
lsz
find(lsz == 0)
length(find(lsz == 0))
length(id2(id2c))
load('gradCam_ROI2.mat')
id1 = find(wabClass == 'Severe');
id2 = find(wabClass == 'Non-severe');
% find non-severe correct predictions
id1c = find(all_preds(id1,2) == 'Severe');
id2c = find(all_preds(id2,2) == 'Non-severe');
length(id2(id2c))
length(id1(id1c))
length(id2(id2c))/length(id2)
length(id1(id1c))/length(id1)
finf1
load('temporaryWorkspace.mat', 'finf1')
mean(finf1)
finf1(2)
save('FORTHEWORDCLOUDS.mat') 