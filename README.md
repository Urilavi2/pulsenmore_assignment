# Pulsenmore Assignment

The assignment is mentioned and previewed in Assignment.pdf.
Assumptions:
1) A '[]' indicates a bad test. The assumption is that if we get a bad test in the device row, the device is malfunctioning and being ignored after mentioning it in shallow analysis.
2) Every row represents a device.
3) column Y1 is called "subjects" as different devices test the same subject.

Files:
-------
1) data content description.txt - Contains a shallow analysis of the database with no mathematical operations, only from scanning the data.
2) data explain.txt - This file holds the deep analysis of the given database. The analyzed data are separated for each subject to describe the correlation between the subject's tests in various ways.
3) new_devices_ranked.txt - Ranks the new devices from the new_devices.csv file based on their probability to be part of the sample group.
4) factory_test_data.csv - Database.
5) new_devices.csv - New devices database.
6) main.py - Python script, explained in details below.


main.py
-------
This script is controlled by a menu. (inputs should be integers 1-4).

At first, the user must enter '1' to read from the database and construct the data structure.
By reading the database, we form the file 'data content description.txt' and constructing a data structure of 2 dictionaries, one for the tests and one for the devices.
The keys of each dictionary are the subjects. In the test dictionary, every device test (4 samples) is held by a numpy array, while in the device dictionary, the devices are held by a list.

By pressing '2', the user will get a deeper analysis of the database (the output file is 'data explained.txt' and plots). 
The deep analysis will plot the X-Y graph for each subject (on the graph, we will see every assosiated devices), will plot the regression graph for each subject, and will show scatter points of tests from different devices on the same graph. In addition, the deep analysis will calculate the R-squared value (squared error percentage from the regression line)
and will check if the tests from all devices are monotonic.

By pressing '3', the user is asking to check what is the probability of the new devices in the new_devices.scv file to fit the sample group in factory_test_data.csv.
The probability is determined by several tests, including:
1. If the subject has already been tested before.
2. If the new devices' tests are monotonic - we expect them to be monotonic.
3. The cross-correlation between the new device's tests and the existing ones - we expect the third test value to be the highest from all the tests.
    Note: As many tests we have for the same subject, the more information we have to determine the probability of the new device to be part of the sample group.
4. The auto-correlation should have the maximum value in the third test value.
The output of this operation is 'new_devices_ranked.txt'. There you will find the conclusions.


As for the bonus section:
Based on my knowledge of the sample group, I can use the same idea from operation 3 in the menu.
To specify which devices will fail my tests, first I'll need to rank the tests.
Meaning, there will be some mandatory tests that a new device must pass, and other tests to estimate how close is the new device is to the sample group.
As I see it, the mandatory test is the monotonic one, as in the sample group, most of the devices' tests are monotonic, therefore the new one must be as well.
Tests that will estimate how close the new device is are the covariance matrix, which will tell us how different the behavior of the new device is compared to the existing ones (only for the same subject),
and the cross-correlation test that will tell us if the new device is acting a bit more or less like other devices (the 'a bit' should be decided at first. For example, I can say the threshold value
will be the mean of all devices' tests at the corresponding X-Y pair). 


















