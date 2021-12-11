% matlab used for this dataset since data was obtained as a mat file

imsize = size(images);
segmented_ims = zeros(480,640,3,1449);

for i = 1:1449
    a = labels(:,:,i);
    segmented_ims(:,:,:,i) = ind2rgb(a, jet(30));
 
end

concat_ims = zeros(256,512,3,600);


for j = 1:600
    
    imA = double(images(:,:,:,j))./255; 
    imB = segmented_ims(:,:,:,j);
    
    imA2 = imresize(imA,[256 256]); 
    imB2 = imresize(imB,[256 256]); 

    
    imAB = cat(2,imA2,imB2);
    
    concat_ims(:,:,:,j) = imAB;

end
figure
imshow(concat_ims(:,:,:,244))


for k = 1:600

    imwrite(concat_ims(:,:,:,k), strcat(int2str(k),'.jpg'));

end