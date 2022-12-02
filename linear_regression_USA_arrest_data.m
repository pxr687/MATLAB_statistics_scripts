% =========================================================================
% PURPOSE

% A script to carry out linear regression on a social science dataset. The
% data is an R dataset and contains the following information:

% "This data set contains statistics, in arrests per 100,000 residents for 
% assault, murder, and rape in each of the 50 US states in 1973. Also given
% is the percent of the population living in urban areas." 
% (from https://rdrr.io/r/datasets/USArrests.html)

% The data itself was downloaded from: 
% https://forge.scilab.org/index.php/p/rdataset/source/tree/master/csv/datasets/USArrests.csv

% =========================================================================
% UNIVARIATE PLOTS

data = readtable("data/USArrests.csv")

% loop over the predictors in the table and plot a histogram of each
% begin at index 2 as index 1 is not a predictor column
for i = 2:length(data.Properties.VariableNames)
    figure
    hist(data.(data.Properties.VariableNames{i}))
    title(data.Properties.VariableNames{i})
    xlabel("Arrests per 100,000 residents")
    ylabel("Frequency")
end

% =========================================================================
% MULTIVARIATE PLOTS

figure
scatter3(data.Murder, data.Rape, data.Assault)
xlabel('Murder Arrests (per 100,000)')
ylabel('Rape Arrests (per 100,000)')
zlabel('Assault Arrests (per 100,000)')

% =========================================================================
% LINEAR REGRESSION ANALYSIS

% for a linear regression model
mod = fitlm(data, 'Assault ~ Rape + Murder')

% =========================================================================
% DIAGNOSTIC PLOTS

% Checking for the normality of residuals
raw_residuals = mod.Residuals{:, 1}
figure
hist(raw_residuals)
xlabel('Residual')
ylabel('Frequency')

% Checking for heteroscedasticity with a fitted vs residual plot
figure
scatter(mod.predict, raw_residuals)
xlabel('Fitted Values')
ylabel('Residuals')

% =========================================================================
% 3D MODEL PLOT

% plotting the linear regression model as a plane

% get the model coefficients
intercept = mod.Coefficients.Estimate(1)
murder_slope = mod.Coefficients.Estimate(2)
rape_slope = mod.Coefficients.Estimate(3)

% get the x-y values for the regression plane
[Murder_for_plot, Rape_for_plot] = meshgrid(data.Murder, data.Rape)

% get the z values for the regression plane
predicted_values_for_surface = intercept + murder_slope * Murder_for_plot + rape_slope * Rape_for_plot

% create the model plot
figure
surf(Murder_for_plot, Rape_for_plot, predicted_values_for_surface)
xlabel('Murder Arrests (per 100,000)')
ylabel('Rape Arrests (per 100,000)')
zlabel('Assault Arrests (per 100,000)')