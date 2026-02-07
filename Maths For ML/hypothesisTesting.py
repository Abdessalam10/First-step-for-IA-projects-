import scipy.stats as stats,ttest_ind



data = [10,20,30,40,50,60,70,80,90,100]
mean= sum(data)/len(data)
std_dev= (sum([(x - mean) ** 2 for x in data]) / len(data)) ** 0.5
Sample_mean = mean
z_score = 1.96
ci= (Sample_mean - z_score * (std_dev / (len(data) ** 0.5)), Sample_mean + z_score * (std_dev / (len(data) ** 0.5)))
print(f"95% Confidence Interval: {ci}")

group= [2.1,2.5,2.8,3.0,3.2]
group2= [1.8,2.0,2.3,2.6,2.9]

# Perform t-test
t_statistic, p_value = ttest_ind(group, group2)
print(f"T-statistic: {t_statistic}, P-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")