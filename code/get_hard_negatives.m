function [hard_negative_features, hard_negative_num] = get_hard_negatives( non_face_scn_path, w, b, feature_params )

D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
% image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
% num_images = length(image_files);

hard_negative_features = zeros(0,D);

fprintf('run detector on non face scenes...\n');
[bboxes, confidences, image_ids] = run_detector(non_face_scn_path, w, b, feature_params);

hard_negative_num = size(image_ids, 1);
fprintf('Extracting hard negatives...\n');
count = 1;
for i = 1: hard_negative_num
    file_name = strcat(non_face_scn_path,'/',image_ids{i});
    img = imread(file_name);
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    [m, n] = size(img);
    if (bboxes(i,2) < m) && (bboxes(i,4) < m) &&(bboxes(i,1) < n) && (bboxes(i,3) < n)
        bboxes(i,1) = int16(bboxes(i,1));bboxes(i,2) = int16(bboxes(i,2));
        bboxes(i,3) = int16(bboxes(i,3));bboxes(i,4) = int16(bboxes(i,4));
        crop_img = img(bboxes(i,2):bboxes(i,4),bboxes(i,1):bboxes(i,3));
        scale  = feature_params.template_size/(bboxes(i,3) - bboxes(i,1)+1);
        crop_img = imresize(crop_img,scale);
        hog = vl_hog(crop_img,feature_params.hog_cell_size);
        hard_negative_features(count,:) = hog(:);
        count = count + 1;
    end
end
hard_negative_num = count-1;

end

