figure(1);
alpha_i = load('./result/dat_tmp/alpha_i');
bar(alpha_i);
xlabel('Item ID');
title('Popularity \alpha_i');

figure(2);
alpha_u = load('./result/dat_tmp/alpha_u');
bar(alpha_u);
xlabel('User ID');
title('Influence \alpha_u');

figure(3);
theta = load('./result/dat_tmp/theta_uu');
imagesc(theta);
xlabel('User ID');
ylabel('User ID');
title('Relation \Theta');
