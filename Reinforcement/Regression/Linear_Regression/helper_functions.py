import numpy as np

def SS(data:np.ndarray, eq_params:tuple|list) -> float:
    """
    Calculate the sum of sqared error for a given dataset and line.

    Args:
        data: Row major numpy array with target variable in last column
        eq_params: coefficients for slope-intercept form of line, ex: if eq is y = 1 + 2x -> eq_params is (1, 2)
                   Importantly, order of the coefficients in eq_params must be the same as the variables in data
    
    Returns:
        sum_squared_diffs: Float value, sum of squared error from data to prediction line
    """
    # calculate the predicted values of the target variable using the equation parameters and all non-target dimensions
    y_intercept, *dim_coefficients = eq_params
    coefficients_applied_to_data = data[:, :-1] * dim_coefficients
    predicted_target_values = np.sum(coefficients_applied_to_data, axis=1) + y_intercept

    # calculate, square, and sum the distances from each point to its position predicted by the line
    diffs = predicted_target_values - data[:, -1]
    squared_diffs = np.square(diffs)
    sum_squared_diffs = np.sum(squared_diffs)

    return sum_squared_diffs


# def determine_line_of_best_fit_for_vis():
#     pass


def determine_LBF(data: np.array) -> tuple:
    """
    Iteratively calculate the line of best fit (LBF) for given data

    Args: 
        data: Row major representation of data in np.array w/ target variable in last column
        precision: # of decimal places to calculate the paramaters for LBF to 

    Returns: 
        line_params: tuple of parameters for LBF, ex: if line is y = 1 + 2x -> line_params is (1, 2)
    """
    pass



if __name__ == "__main__":
    points = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])
    line_params = [0, 2]
    result = SS(points, line_params)
    print(result)
    points_t = np.transpose(points)

    print(np.matmul(points_t,points))
