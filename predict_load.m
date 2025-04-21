function prediction = predict_load(city, model_type, weather_input)
    % Predict 24-hour electricity load using pre-trained SVM or ANN models.
    % Inputs:
    %   - city: 'toc', 'san', or 'dav'
    %   - model_type: 'svm' or 'ann'
    %   - weather_input: [T2M, QV2M, TQL, W2M] (1x4 vector)
    % Output:
    %   - prediction: 24x1 vector of predicted load values

    prediction = [];  % Initialize output

    try
        num_lags = 24;
        future_steps = 24;

        % Load appropriate model
        model_folder = 'C:\Users\yash2\OneDrive\Desktop\AI Load Forecast\city_models';
        svm_path = fullfile(model_folder, [city '_svm_model.mat']);
        ann_path = fullfile(model_folder, [city '_ann_model.mat']);

        if strcmpi(model_type, 'svm')
            load(svm_path, 'svm_model', 'mu_X', 'sigma_X', 'mu_y', 'sigma_y');
            model = svm_model;
        elseif strcmpi(model_type, 'ann')
            load(ann_path, 'ann_net', 'mu_X', 'sigma_X', 'mu_y', 'sigma_y');
            model = ann_net;
        else
            error('Invalid model type. Choose "svm" or "ann".');
        end

        % Read national demand data
        data = readtable('C:\Users\yash2\OneDrive\Desktop\codes\EEE&UID\continuous dataset.csv');
        y = data.nat_demand;

        % Validate lag availability
        if length(y) < num_lags
            error('Insufficient demand history to generate lagged features.');
        end

        % Initialize lag features from the last 24 hours
        lagged = y(end - num_lags + 1 : end)';  % 1x24 row vector

        % Ensure weather_input is a row vector
        weather_input = reshape(weather_input, 1, []);  % 1x4

        % Prepare for recursive forecasting
        forecast = zeros(future_steps, 1);

        for i = 1:future_steps
            % Combine static weather + dynamic lagged demand
            input = [weather_input, lagged];

            % Normalize input
            input_scaled = (input - mu_X) ./ sigma_X;

            % Predict
            if strcmpi(model_type, 'svm')
                pred_scaled = predict(model, input_scaled);
            else
                pred_scaled = model(input_scaled')';  % ANN expects column input
            end

            % Denormalize prediction
            pred = pred_scaled * sigma_y + mu_y;

            % Save prediction
            forecast(i) = pred;

            % Update lagged vector (shift left, append new pred)
            lagged = [lagged(2:end), pred];
        end

        prediction = forecast;

    catch ME
        disp("Error during prediction:");
        disp(ME.message);
        prediction = nan(24, 1);  % Return NaNs on error for consistency
    end
end
