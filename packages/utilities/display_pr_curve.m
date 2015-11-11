function display_pr_curve(metrics)
recall = round(metrics.recall*1000)/1000;
precision = round(metrics.precision*1000)/1000;
r3 = [];
p3 = [];
iGood = [];
for i = 1:length(recall)
    idx = find(recall >= recall(i) & precision > precision(i));
    if isempty(idx)
        r3(end+1) = recall(i);
        p3(end+1) = precision(i);
        iGood(end+1) = i;
    end
end
[~,idx] = sort(r3);
hold on
figure(100), plot(r3(idx), p3(idx), 'ko-'), grid on
set(gca,'XLim', [0,1], 'YLim', [0,1])
xlabel('recall')
ylabel('precision')
set(gcf, 'color', [1 1 1])

