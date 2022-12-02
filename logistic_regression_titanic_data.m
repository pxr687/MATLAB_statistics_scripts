% =========================================================================
% PURPOSE

% This is a multivariate logistic regression analysis using the famous
% Titanic dataset. The dataset contains demographic information about the 
% passengers of the Titanic, as well as whether or not they survived the 
% disaster. The data were obtained from here:
% https://github.com/matthew-brett/titanic-r

% =========================================================================
% UNIVARIATE PLOTS

data = readtable("data/titanic_clean.csv")

% separate out categorical from numerical data
categorical_predictors = ["gender",  "class",  "survived"];
numeric_predictors = ["age", "fare"];

% loop over the categorical predictors in the table and plot a bar plot of 
% each
for i = 1:length(categorical_predictors)
    figure
    to_plot = categorical(data.(categorical_predictors{i}));
    hist(to_plot)
    title(categorical_predictors{i})
    xlabel("")
    ylabel("Frequency")
end

% loop over the numeric predictors in the table and plot a histogram of
% each
for i = 1:length(numeric_predictors)
    figure
    hist(data.(numeric_predictors{i}))
    title(numeric_predictors{i})
    xlabel("")
    ylabel("Frequency")
end

% =========================================================================
% MULTIVARIATE PLOTS

% convert survived to categorical
cat_survived = categorical(data.survived);

% convert survived words to indicator variable
ind_survived = renamecats(cat_survived,{'yes', 'no'},{'1','0'});
ind_survived = str2double(string(ind_survived));

% add the indicator variable to the dataframe
data.survived_indicator = ind_survived

% crate a 3D scatter plot
figure
scatter3(data.age, data.fare, data.survived_indicator)
xlabel('Age')
ylabel('Fare')
zlabel('Survived (1 == YES)')

% =========================================================================
% LOGISTIC REGRESSION ANALYSIS

% fitting the first model
logistic_model_1 = fitglm(data,"survived_indicator ~ age * fare", ...
                          'Distribution','binomial')

% removing the non-significant interaction term
logistic_model_2 = fitglm(data,"survived_indicator ~ age + fare", ...
                          'Distribution','binomial')

% =========================================================================
% DIAGNOSTIC PLOTS

% fitted vs deviance residual plot
deviance_residuals = logistic_model_2.Residuals.Deviance;
figure
scatter(logistic_model_2.predict, logistic_model_2.Residuals.Deviance)
xlabel('P(Survival)')
ylabel('Deviance Residuals')

% =========================================================================
% 3D MODEL PLOT

% plotting the logistic regression model as a plane 

% get the model coefficients
intercept = logistic_model_2.Coefficients.Estimate('(Intercept)');
age_slope = logistic_model_2.Coefficients.Estimate('age');
fare_slope = logistic_model_2.Coefficients.Estimate('fare');

% get the x-y values for the regression plane
[age_for_plot, fare_for_plot] = meshgrid(data.age, data.fare);

% get the z values for the regression plane
y_hat = intercept + age_slope * age_for_plot + fare_slope * fare_for_plot;

% create the model plot (with the outcome on the log odds scale)
figure
surf(age_for_plot, fare_for_plot, y_hat)
xlabel('Age')
ylabel('Fare')
zlabel('Log Odds (Survival)')