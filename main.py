import matplotlib.pyplot as plt
import numpy as np

# ---------------- global variables ------------------

subjects_hash = {}
subjects_devices_hash = {}
x_axis = []
malfunction = []

# ------------ end of global variables ---------------


# ---------------- Data loading and shallow description ---------------

def read_from_file(path):
    data = []
    f = open(path, 'r')
    headers = f.readline()
    headers = headers.strip('\n').split(',')
    # remove \n from string and split the header to a list by ',' as a separator

    while True:
        line = (f.readline())
        if not line:
            break
        else:
            data.append(line[:-1].split(','))  # remove the last element in line: '\n'
    return headers, np.array(data, dtype='str')


def shallow_analysis(arr):
    path = "./data content description.txt"
    f = open(path, 'w')
    f.write("-------------- Shallow Analysis --------------\n\n")
    devices = len(arr)
    f.write("Number of devices: " + str(devices) + '\n\n')
    subjects = arr[:, 0]
    data = arr[:, range(2, 9, 2)]
    unique_values = np.unique(subjects)
    bad_samples = data == '[]'
    bad_devices = data[:, 2] == '[]'
    f.write("The subjects: \n" + str(unique_values.tolist()) + '\n\n')
    f.write("Number of malfunctioning devices: " + str(bad_devices.sum()) + '\n')
    f.write("and the devices are: ")
    for device in range(0, len(bad_devices)):
        if bad_devices[device]:
            f.write(str(int(device) + 1) + ' ')
    f.write("\n\nNumber of bad samples: " + str(bad_samples.sum()) + '\n')


def Data_loading_and_Explain():
    path = "./factory_test_data.csv"
    headers_load, np_data_load = read_from_file(path)
    shallow_analysis(np_data_load)
    forming_hash(np_data_load, subjects_hash, subjects_devices_hash)
    return headers_load, np_data_load

# ------------ end of Data loading and shallow description ---------------


# ------------ Data visualization ---------------
def plot_XY():
    """ Plot all subjects. Each subject's graph will contain all of its devices samples
        could be on the same window, but too dense to understand and see properly.
        The function returns the monotonic dict. each key is the subject and every subject contain boolean array if the
        tests samples for each device were monotonic or not. """
    mono = {}
    for subject in subjects_hash:
        subject_samples = subjects_hash[subject]
        devices = subjects_devices_hash[subject]
        fig, ax = plt.subplots()
        devices_idx = 0
        mono[subject] = []
        for devices_tests in subject_samples:
            # if (devices_tests == '[]').any():
            #     ax.plot(x_axis, np.zeros(4), label="device " + str(devices[devices_idx]))
            #     devices_idx += 1
            #     continue
            mono[subject].append(is_monotonic(devices_tests))
            ax.plot(x_axis, devices_tests.astype(float), label="device " + str(devices[devices_idx]), marker='o', markersize=8)
            devices_idx += 1
            ax.legend(loc='upper left')
        mono[subject] = np.array(mono[subject])
        ax.set_title("subject " + str(subject) + " X-Y Graph")
        ax.grid(True)
        plt.show()
    return mono


def plot_regression():
    """
        plot regression for every subject in the dataset

        the function will return a dict mapped by the subject number (as str) and every key holds a tuple with the
        minimum (1-R_squared) value and the device number.

     """

    min_err_per_sub = {}
    # path = "./regression data.txt"
    # f = open(path, 'w')
    # save the quadratic errors from devices for each subject
    for subject in subjects_hash:
        subject_arr = subjects_hash[subject]
        devices = subjects_devices_hash[subject]
        fig, ax = plt.subplots()
        x_avg = np.mean(x_axis)
        min_err = (999, 0)  # (value, device num)
        # f.write("For test subject: " + str(subject) + '\n')
        # print("\nFor test subject: " + str(subject))
        denominator = sum((xi - x_avg) ** 2 for xi in x_axis)
        for device_idx, samples in enumerate(subject_arr):
            samples_avg = np.mean(samples)
            numerator = sum((xi - x_avg) * (yi - samples_avg) for xi, yi in zip(x_axis, samples))
            beta1 = numerator / denominator
            beta0 = samples_avg - beta1 * x_avg
            quadratic_err = ssres(beta0, beta1, samples) / sst(samples_avg, samples)
            if quadratic_err < min_err[0]:
                min_err = (quadratic_err, devices[device_idx])
            # f.write("the quadratic error of device " + str(devices[device_idx]) +
            #         " from the regression line: " + str(quadratic_err) + '\n')
            # print("the quadratic error of device " + str(devices[device_idx]) +
            #         " from the regression line: " + str(quadratic_err))
            # Plotting the data points
            ax.scatter(x_axis, samples, label='Device ' + str(devices[device_idx]))
            # Plotting the regression line
            ax.plot(x_axis, [beta0 + beta1 * xi for xi in x_axis])
            ax.legend(loc='upper left')
            ax.grid(True)
        # f.write('\n')
        ax.set_title("subject " + str(subject) + " Regression Graph")
        plt.show()
        min_err_per_sub[subject] = min_err
        # f.write("Therefore, the device with the least squared error is: " + str(min_err[1]) +
        #         '\nand the error is: ' + str(min_err[0]) + '\n')
        # print("Therefore, the device with the least squared error is: " + str(min_err[1]) +
        #         '\nand the error is: ' + str(min_err[0]))
        # if len(subject_arr) < 3:
        #     f.write("NOTICE THAT THERE IS AN INSUFFICIENT NUMBER OF TESTS! therefore this test isn't satisfying\n")
        #     print("NOTICE THAT THERE IS AN INSUFFICIENT NUMBER OF TESTS! therefore this test isn't satisfying\n")
        # f.write('\n')
    return min_err_per_sub


def Data_Visualization():
    path = "./data explained.txt"
    f = open(path, 'w')
    f.write("---------------- Data Explanation ----------------\n\n")
    mono_arr = plot_XY()
    min_quad = plot_regression()
    subjects_keys = min_quad.keys()
    for subject in subjects_keys:
        f.write("Subject: " + subject + '\n')
        f.write("--------------\n")
        if not((mono_arr[subject] == True).all()):
            non_mono_idx = np.where(mono_arr[subject] == False)[0]
            string_mono = "As we can see in the X-Y plot of subject " + subject + " not all of it's devices samples are" \
                          " monotonic. The not monotonic devices are: "
            for non_mono_device in non_mono_idx:
                string_mono += str(subjects_devices_hash[subject][non_mono_device]) + "  "
        else:
            string_mono = "As we can see in the X-Y plot of subject " + subject + " all of it's devices samples are" \
                          " monotonic."
        f.write(string_mono + '\n\n')

        cov_diagonal_dominance = cov_analayzer(subject)
        if cov_diagonal_dominance:
            string_cov = "From the covariance matrix of subject " + subject + " we can learn that the diagonal is " \
                        "dominant and therefore the devices' behavior is distinctly different from each other and they" \
                                                                             " NOT share common pattern or trends"
        else:
            string_cov = "From the covariance matrix of subject " + subject + " we can learn that the diagonal is not " \
                          "dominant and therefore the devices' behavior is not distinctly different from each other" \
                          " and they share common pattern or trends"
        f.write(string_cov + '\n\n')

        if len(subjects_devices_hash[subject]) < 3:
            string_reg = "In addition, because of low number of devices scanning subject number " + subject + " we can not asure by the regression " \
                     "graph and by the calculation of the R-squared value which device is a better match for our standards"
        else:
            string_reg = "In addition, as we can see in subject's " + subject + " regression graph and after calculating the R-squared value" \
                     + ' (' + str(1-min_quad[subject][0]) + ")\nwe can determine by this test that device " \
                     + str(min_quad[subject][1]) + " is the closest to the regression line and most likely to match our standards"

        f.write(string_reg + '\n\n\n')


# ------------ end of Data visualization ---------------


# -------------- Data Processing ---------------
def pearson_correlation_coefficient(subject):
    """ Will find the pearson correlation coefficient between different devices on the same subject

                                    -- NOTICE --
        Bad samples recognized by '[]'. every vector with '[]' will be ignored!


        :param subject: subject ID in subject hash table
        :return Tuple of The pearson correlation coefficient dict and the device correlation order

                The pearson correlation coefficient dict is mapped by the device number,
                    each key holds the result from the correlation

                The device correlation order dict is mapped by the device number.
                Each key holds a list of tuples of the device that were correlated,
                in the same order as in the pearson correlation coefficient dict
                    """
    # The assumption is that both arrays are with the same size
    pearson_correlation = {}
    devices_correlated = {}
    devices_samples = subjects_hash[subject]
    devices_arr = subjects_devices_hash[subject]
    for deviceX_idx in range(0, len(devices_samples)):
        deviceX = devices_arr[deviceX_idx]
        deviceX_data = devices_samples[deviceX_idx]
        # if (deviceX_data == '[]').any():
        #     continue
        # else:
        #     pearson_correlation[deviceX] = []
        #     devices_correlated[deviceX] = []
        pearson_correlation[deviceX] = []
        devices_correlated[deviceX] = []
        for deviceY_idx in range(0, len(devices_samples)):
            deviceY = devices_arr[deviceY_idx]
            if deviceX == deviceY:
                continue
            deviceY_data = devices_samples[deviceY_idx]
            # if (deviceY_data == '[]').any():
            #     continue
            n = len(deviceX_data)
            deviceX_data = devices_samples[deviceX_idx].astype(float)
            deviceY_data = devices_samples[deviceY_idx].astype(float)
            sumX = deviceX_data.sum()
            sumY = deviceY_data.sum()
            sum_mult = np.sum(deviceX_data * deviceY_data)
            Xsquared_sum = np.sum(np.square(deviceX_data))
            Ysquared_sum = np.sum(np.square(deviceY_data))
            numerator = n * sum_mult - sumX*sumY
            denominator = ((n * Xsquared_sum - sumX ** 2) * (n * Ysquared_sum - sumY ** 2)) ** 0.5
            if denominator == 0:
                return 0  # If the denominator is 0, the correlation is undefined, return 0 as a default.

            pearson_correlation[deviceX].append(numerator / denominator)
            devices_correlated[deviceX].append((deviceX, deviceY))
        pearson_correlation[deviceX] = np.array(pearson_correlation[deviceX])
    return pearson_correlation, devices_correlated


def cov_analayzer(subject):
    covariance_mat = covariance(subject)
    if covariance_mat.size > 1:
        num_devices = covariance_mat.shape[0]
        vars = np.diagonal(covariance_mat)
        covs = covariance_mat - np.diag(vars)
        is_diagonal_dominant = np.all(vars > np.sum(np.abs(covs), axis=1))
        return is_diagonal_dominant
    return True

def sst(mean, samples):
    """
    calculates the total sum of squares (SST) for a dataset.

    :param mean: mean of samples
    :param samples:
    :return:
    """
    squared_diff = [(yi - mean) ** 2 for yi in samples]
    return np.sum(squared_diff)


def ssres(beta0, beta1, samples):
    """
    calculate the sum of squared residuals (also known as the sum of squared errors) for a linear regression model.

    :param beta0: y-axis offset
    :param beta1: slope
    :param samples: samples
    :return:
    """
    predicted_values = [beta0 + beta1 * xi for xi in x_axis]
    squared_diff = [(yi - pred) ** 2 for yi, pred in zip(samples, predicted_values)]
    return np.sum(squared_diff)


def covariance(subject):
    """
    Calculate the covariance matrix of a subject.


        the matrix looks like:  var(dev1)           cov(dev1,dev2)  ...     cov(dev1,devN)
                                cov(dev2,dev1)      var(dev2)       ...     cov(dev2,devN)
                                    .                   .                       .
                                    .                   .                       .
                                    .                   .                       .
                                cov(devN,dev1)      cov(devN,devN)  ...     var(devN)


    :param subject:
    :return: covariance matrix
    """
    subject_arr = subjects_hash[subject].astype(float)
    return np.cov(subject_arr)


def variance(subject):
    """
    Calculate the variance vector of a subject for each device.


        the vector looks like:   [var(dev1)  var(dev2) ... var(devN)]



    :param subject:
    :return: covariance matrix
    """
    subject_arr = subjects_hash[subject].astype(float)
    return np.var(subject_arr, axis=1)  # axis=1 calculate the variance for each row in the array


def cross_correlation(subject, subject_arr, devices_arr):
    """ Will find the cross-correlation between different devices on the same subject

                                           -- NOTICE --
        Bad samples recognized by '[]'. every vector with '[]' will be ignored!


        :param devices_arr: devices of key = subject
        :param subject_arr: samples of key = subject
        :param subject: key for the subjects and devices hash tables
        :return Tuple of The cross-correlation dict, The auto correlation dict and The device correlation order.

                The cross-correlation dict is mapped by the device number,
                each key contains numpy arrays of the result for each cross.

                the auto-correlation dict is mapped by the device number. Each key contain a numpy array

                The device correlation order is a dict mapped by the device number.
                Each key contain a list that holds tuples with the devices that were crossed
                """

    cross_correlation = {}
    devices_correlated = {}
    auto_correlation = {}
    # subject_arr = subjects_hash[subject]
    # devices_arr = subjects_devices_hash[subject]
    for deviceX_idx in range(0, (len(subject_arr))):
        deviceX = devices_arr[deviceX_idx]
        deviceX_data = subject_arr[deviceX_idx]
        if (deviceX_data == '[]').any():
            # auto_correlation[deviceX] = np.zeros(4)
            continue
        else:
            cross_correlation[deviceX] = []
            devices_correlated[deviceX] = []
        for deviceY_idx in range(0, len(subject_arr)):
            deviceY = devices_arr[deviceY_idx]
            deviceY_data = subject_arr[deviceY_idx]
            if (deviceY_data == '[]').any():
                continue
            if deviceX != deviceY:
                correlated = np.correlate(deviceX_data.astype(float), deviceY_data.astype(float), mode='same')
                cross_correlation[deviceX].append(correlated.astype(float))
                devices_correlated[deviceX].append((deviceX, deviceY))

            else:  # auto-correlation
                correlated = np.correlate(deviceX_data.astype(float), deviceY_data.astype(float), mode='same')
                auto_correlation[deviceX] = (correlated.astype(float))
    return cross_correlation, auto_correlation, devices_correlated


def is_monotonic(values):
    return np.allclose(values, sorted(values)) or np.allclose(values, sorted(values, reverse=True))


def correl_analyze_new_devices(cross, auto):
    scoreboard = 0
    max_auto_idx = np.argmax(auto[-1])
    if max_auto_idx == 2:
        scoreboard += 1  # compared to other devices, they both have the max val of auto-corr in the third index
    if len(cross) > 1:
        # print("only relevant: \n", relevant_cross)
        relevant_cross = cross[:, -1, :]
        max_vals = np.max(relevant_cross, axis=1)
        third_col_values = relevant_cross[:, 2]
        if np.array_equal(max_vals, third_col_values):
            scoreboard += 1  # compared to other devices, they both have the max val of cross-corr in the third index

        for row in relevant_cross:
            if is_monotonic(row[:-1]):
                scoreboard += 1  # compared to other devices, they both have monotonic behavior in the first 3 crosses
                if row[-1] < row[-2]:
                    scoreboard += 1 # compared to other devices, the last cross always has lower value then the third one
    return scoreboard

# ------------ end of Data Processing ---------------


# ------------ Data Exploration ---------------


def new_devices():
    # THE ASSUMPTIONS ARE THAT THERE ARE NO MALFUNCTIONING DEVICES
    # AND THE X-AXIS HAS THE SAME POINTS AS THE ORIGINAL DATA

    path = "./new_devices.csv"
    headers_load, np_data_load = read_from_file(path)
    subjects = np_data_load[:, 1]
    new_devices_score = {}
    new_sub_hash = {}
    new_sub_device_hash = {}
    # forming a new data structure and making a LOCAL merged data structure
    forming_hash(np_data_load[:, 1:], new_sub_hash, new_sub_device_hash)
    united_subject_hash = merge_dictionaries(subjects_hash, new_sub_hash)
    next_device = max(subjects_devices_hash.values())[0]
    for subject in subjects:
        new_sub_device_hash[subject] = new_sub_device_hash[subject][0] + next_device
    united_devices_hash = merge_dictionaries(subjects_devices_hash, new_sub_device_hash, is_devices=True)
    # end of new data structure
    for subject in subjects:
        if subject in subjects_hash:  # initialize scoreboard for ranking new devices
            new_devices_score[subject] = 1
        else:
            new_devices_score[subject] = 0

        if is_monotonic(new_sub_hash[subject]):
            # if the test is monotonic, it is more likely that its part of the sample group, as most of it is
            new_devices_score[subject] += 1


        cross, auto, device_corr = cross_correlation(subject, united_subject_hash[subject], united_devices_hash[subject])
        cross_vals = np.array([val for val in cross.values()])
        auto_vals = [val for val in auto.values()]
        dev_vals = [val for val in device_corr.values()]
        new_devices_score[subject] += correl_analyze_new_devices(cross_vals, auto_vals)
    sorted_scores = sorted(new_devices_score.items(), key=lambda x: x[1])
    # Extract sorted keys and values from the sorted items
    sorted_keys = [score[0] for score in sorted_scores]
    sorted_values = [score[1] for score in sorted_scores]
    # CONCLUSION
    path = "./new_devices_ranked.txt"
    f = open(path, 'w')
    f.write("------------ New Devices Conclusions ------------\n\n")
    for idx, key in enumerate(sorted_keys):
        f.write("Device number " + str(new_sub_device_hash[key]) + " is ranked " + str(idx + 1) + " based on probability"
                                                          " to be part of the sample group from the new devices \n\n")

# --------------------------- DATA STRUCTURE ------------------------------


def merge_dictionaries(dict1, dict2, is_devices=False):  # importance for order! big dict first
    dict3 = {}
    keys1 = dict1.keys()
    keys1 = [key for key in keys1]
    keys2 = dict2.keys()
    keys2 = [key for key in keys2]
    keys = list(set(keys1 + keys2))
    for key in keys:
        if key in keys1:
            if not is_devices:
                dict3[key] = dict1[key].tolist()
            else:
                dict3[key] = dict1[key]
            if key in keys2:
                if not is_devices:
                    dict3[key].append(dict2[key][0].tolist())
                else:  # meaning we are appending devices numbers!
                    dict3[key].append(dict2[key])
        elif key in keys2:
            if not is_devices:
                dict3[key] = dict2[key].tolist()
            else:  # meaning we are appending devices numbers!
                dict3[key] = [dict2[key]]
        if not is_devices:
            dict3[key] = np.array(dict3[key], dtype=float)

    return dict3


def forming_hash(arr, subjects_h, subjects_devices_h):
    """ keys will be the subject number (as a string). every key will have a list containing lists.
        every inner list will have the device number in index 0 and it's tests results """
    global x_axis
    unique_values = np.unique(arr[:, 0])
    for value in unique_values:
        subjects_h[value] = []
        subjects_devices_h[value] = []
        find_subject_tests(value, arr, subjects_h, subjects_devices_h)
    x_axis = [int(np.unique(arr[:, 1])[0]), int(np.unique(arr[:, 3])[0]), int(np.unique(arr[:, 5])[0]),
              int(np.unique(arr[:, 7])[0])]


def find_subject_tests(sub, arr, subject_h, devices_h):
    """ Find for a specific subject all of it tests by the devices
        and map them to the subject's hash table
        NOTICE: a device with '[]'s as data refer to a malfunctioning device hence,
                I'll save the device number and ignore this test

        :param      sub: Subject number (as a string)
        :param      arr: NumPy array contain all of tests """

    for device, tests in enumerate(arr):
        if tests[0] != sub:
            continue
        else:
            data = tests[2::2]
            if (data == '[]').any():
                malfunction.append(device+1)
                continue
            subject_h[sub].append(data)
            devices_h[sub].append(device + 1)
    subject_h[sub] = np.array(subject_h[sub]).astype(float)


# --------------------------- END OF DATA STRUCTURE ------------------------------

def main():
    menu = """----- Home Assignment - pulsenmore -----
1) Data Loading - mandatory!
2) Data Visualization and Explanation
3) Data Exploration
4) Exit
----------------------------------------
"""
    data_load = False
    while True:
        print(menu)
        state = input("enter a number: ")
        try:
            state = int(state)
            if not 0 < state < 5:
                raise ValueError

            if state == 1:
                data_load = True
                Data_loading_and_Explain()

            elif state == 2:
                if data_load:
                    Data_Visualization()
                else:
                    print("Load data first!")
                    continue

            elif state == 3:
                if data_load:
                    new_devices()
                else:
                    print("Load data first!")
                    continue

            elif state == 4:
                plt.close('all')
                exit(0)

        except ValueError:
            print("Bad input, try again")


if __name__ == '__main__':
    main()
