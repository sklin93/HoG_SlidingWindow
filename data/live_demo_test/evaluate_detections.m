function [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path, draw)
% 'bboxes' is Nx4, N is the number of non-overlapping detections, and each
% row is [x_min, y_min, x_max, y_max]
% 'confidences' is the Nx1 (final cascade node) confidence of each
% detection.
% 'image_ids' is the Nx1 image names for each detection.

%This code is modified from the 2010 Pascal VOC toolkit.
%http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/index.html#devkit

if(~exist('draw', 'var'))
    draw = 1;
end

%this lists the ground truth bounding boxes for the test set.

fid = fopen(label_path);
gt_info = textscan(fid, '%s %d %d %d %d');
fclose(fid);
gt_ids = gt_info{1,1};
gt_bboxes = [gt_info{1,2}, gt_info{1,3}, gt_info{1,4}, gt_info{1,5}];
gt_bboxes = double(gt_bboxes);

gt_isclaimed = zeros(length(gt_ids),1);
npos = size(gt_ids,1); %total number of true positives.

% sort detections by decreasing confidence
[sc,si]=sort(-confidences);
image_ids=image_ids(si);
bboxes   =bboxes(si,:);

% assign detections to ground truth objects
nd=length(confidences);
tp=zeros(nd,1);
fp=zeros(nd,1);
duplicate_detections = zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('pr: compute: %d/%d\n',d,nd);
        drawnow;
        tic;
    end
    cur_gt_ids = strcmp(image_ids{d}, gt_ids); %will this be slow?

    bb = bboxes(d,:);
    ovmax=-inf;

    for j = find(cur_gt_ids')
        bbgt=gt_bboxes(j,:);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0       
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax %higher overlap than the previous best?
                ovmax=ov;
                jmax=j;
            end
        end
    end
    
    % assign detection as true positive/don't care/false positive
    if ovmax >= 0.3
        if ~gt_isclaimed(jmax)
            tp(d)=1;            % true positive
            gt_isclaimed(jmax)=true;
        else
            fp(d)=1;            % false positive (multiple detection)
            duplicate_detections(d) = 1;
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute cumulative precision/recall
cum_fp=cumsum(fp);
cum_tp=cumsum(tp);
rec=cum_tp/npos;
prec=cum_tp./(cum_fp+cum_tp);

ap=VOCap(rec,prec);
% auc=VOCap(1-prec,rec);     % for area under auc curve

% Besides AP, we also compare your results on the ROC curve 
% by checking locations FP = 0, 10, 100
fp_num = [0, 10, 100];
np = length(fp_num);
recall_at = zeros(np, 1);       % corresponding recall given different FP
for j = 1:np
    idx = find(cum_fp == fp_num(j));
    if ~isempty(idx)
        recall_at(j) = rec(idx(end));
    end
end

fprintf(' - - - - - - - - - - - \n');
fprintf('Average Precision:\t\t %.3f\n', ap);
for j = 1:np
    fprintf('Detection Rate (FP = %d):\t %.3f\n', fp_num(j), recall_at(j));
end

if draw
    % plot precision/recall
    figure(12)
    plot(rec,prec,'-');
    axis([0 1 0 1])
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('Average Precision = %.3f',ap));
    set(12, 'Color', [.988, .988, .988])
    
    pause(0.1) %let's ui rendering catch up
    average_precision_image = frame2im(getframe(12));
    % getframe() is unreliable. Depending on the rendering settings, it will
    % grab foreground windows instead of the figure in question. It could also
    % return a partial image.
    imwrite(average_precision_image, 'visualizations/average_precision.png')
    
    figure(13)
    plot(cum_fp,rec,'-')
%     plot(1-prec, rec, '-')
    axis([0 500 0 1])
%     axis([0 1 0 1])
    grid;
    xlabel 'False positives'
    ylabel 'Number of correct detections (recall)'
    title(sprintf('Recall(FP=0): %.3f, Recall(FP=10): %.3f, Recall(FP=100): %.3f', ...
        recall_at(1), recall_at(2), recall_at(3)));
    pause(0.1)
%     roc_image = frame2im(getframe(13));
%     imwrite(roc_image, 'visualizations/roc.png')
end

%% Save results
out_file_pr = fopen('pr.txt', 'w+');
out_file_roc = fopen('roc.txt', 'w+');
for i = 1:size(rec, 1)
    fprintf(out_file_pr, sprintf('%.4f %.4f\n', rec(i), prec(i)));
    fprintf(out_file_roc, sprintf('%d %.4f\n', cum_fp(i), rec(i)));
end
fclose(out_file_pr);
fclose(out_file_roc);

%% Re-sort return variables so that they are in the order of the input bboxes
reverse_map(si) = 1:nd;
tp = tp(reverse_map);
fp = fp(reverse_map);
duplicate_detections = duplicate_detections(reverse_map);



