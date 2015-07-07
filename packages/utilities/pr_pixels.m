function metrics = pr_pixels(truth, test)

p = [];
r = [];
for i = 0:0.01:1
   %i
   testThresh = test >= i;
   
   TP = single(sum(testThresh(:) == 1 & truth(:) == 1));
   FP = single(sum(testThresh(:) == 1 & truth(:) == 0));
   FN = single(sum(testThresh(:) == 0 & truth(:) == 1));
   % Not used
   %TN = sum(test(:) == 0 & truth(:) == 0);
   
   p(end+1) = TP/(TP+FP);
   r(end+1) = TP/(TP+FN);
end

figure(99), plot(r,p,'o'), grid on
metrics.precision = p;
metrics.recall = r;

