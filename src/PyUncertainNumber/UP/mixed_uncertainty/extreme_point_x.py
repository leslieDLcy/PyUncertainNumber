import numpy as np

def extreme_pointX(ranges, signX):
    """
    
    args:
        ranges: A NumPy array of shape (d, 2) representing the ranges 
                 (each row is a variable, each column is a bound).
        signX: A NumPy array of shape (1, d) representing the signs.
    
    notes: 
        Calculates the extreme points of a set of ranges based on signs.
    returns:
        A NumPy array of shape (2, d) representing the extreme points.
    """
    d = len(ranges)  # Get the number of dimensions
    pts = np.tile(signX, (2, 1))  # Repeat signX twice vertically
    
    Xsign = np.zeros((2, d))
    for j in range(d):  # Iterate over dimensions
        if pts[0, j] > 0:  # Check sign for the first row (minimum)
            Xsign[0, j] = ranges[j][0]  # Take the first element (lower bound)
            Xsign[1, j] = ranges[j][-1]  # Take the last element (upper bound)
        else:
            Xsign[0, j] = ranges[j][-1]  # Take the last element (upper bound)
            Xsign[1, j] = ranges[j][0]  # Take the first element (lower bound)
    
    return Xsign

# #example
# # Define the ranges (same as before)
# ranges = np.array([
#     [1, 3, 5],  # Lower bounds
#     [2, 4, 6]   # Upper bounds
# ])

# # Define the signs (corrected)
# signX = np.array([[-1, 1, -1]]) # One row, three columns

# # Calculate the extreme points
# Xsign = extremePointX(ranges, signX)

# # Print the result
# print(Xsign)

# def extreme_pointX(ranges, signX):
#   """
#   args:
#     ranges: A NumPy array of shape (2, d) representing the ranges.
#     signX: A NumPy array of shape (1, d) representing the signs.
  
#   notes:
#     Calculates the extreme points of a set of ranges based on signs.

#   returns:
#     A NumPy array of shape (2, d) representing the extreme points.
#   """
#   d = ranges.shape[0]  # Get the number of dimensions
#   pts = np.tile(signX, (2, 1)) # Repeat signX twice vertically
#   print(pts[0,1])
#   pts[0, pts[0, :] < 0] = 2  # Adjust indices based on signs
#   pts[1, pts[1, :] > 0] = 2
#   pts[1, pts[1, :] < 0] = 1

#   ptsI = np.zeros((2, d), dtype=int)  # Initialize ptsI
#   for i in range(2):
#     for j in range(d):
#       ptsI[i, j] = np.ravel_multi_index([pts[i, j] - 1, j], (2, d))  # Convert to linear indices

#   Xsign = ranges.flatten()[ptsI]  # Extract values from ranges using linear indices
#   return Xsign

# # #example
# # # Define the ranges (same as before)
# # ranges = np.array([
# #     [1, 3, 5],  # Lower bounds
# #     [2, 4, 6]   # Upper bounds
# # ])

# # # Define the signs (corrected)
# # signX = np.array([[-1, 1, -1]]) # One row, three columns

# # # Calculate the extreme points
# # Xsign = extremePointX(ranges, signX)

# # # Print the result
# # print(Xsign)