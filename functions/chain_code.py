import numpy as np

def directions_to_chain_code(directions):
    """"
    Maps every direction (ex: E, W, SN) to its corresponding number in chain code using

    Args:
        directions (ndarray): a numpy array containing directions (char)

    Returns:
        chain (list): A list the same size as directions carrying the corrsponding chain codes
    """
    directns_map = np.array(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    chain = []
    for direction in directions:
        if not direction:
            continue
        # mapping using directns_map
        ind = np.where(directns_map == direction)[0][0]
        chain.append(ind)
    return chain

def get_direction(curr_pt, prev_pt):
    """"
    Compares the coordinates of 2 consecutiv contour points to decide the orientation
    (ex N, SE, NW) of the line segment joining them

    Args:
        curr_pt (ndarray): a numpy array containing the x and y coords of the second pt
        prev_pt (ndarray): a numpy array containing the x and y coords of the first pt

    Returns:
        orient (string): carrying the orientation in one of 8 directions (N, S, E, W, NE, NW, SE, SW)
    """
    orient = ""

    # calculate changes in x and y
    delta_x = round(curr_pt[0]) - round(prev_pt[0])
    delta_y = curr_pt[1] - prev_pt[1]

    # they should not be the same point
    if not delta_y and not delta_x:
        return ""

    # north or south
    if delta_y > 0:
        orient += "N"
    elif delta_y < 0:
        orient += "S"

    # east or west
    if delta_x > 0:
        orient += "E"
    elif delta_x < 0:
        orient += "W"

    return orient

def get_perimeter(pts):
    """
    Calculate the perimeter of a polygon given its vertices using Euclidean distance.

    Args:
    pts (ndarray): A 2D ndarray of shape (no of points, 2) representing the x 
                    and y coordinates of the vertices of the polygon.

    Returns:
    perimeter (float): The perimeter of the polygon.
    """

    points = pts.copy()
    # x0, x1, x2, x3, ...
    x_list_1 = points[:, 0]

    x_list_2 = points[:, 0][1:].copy()
    # x1, x2, ..., x0
    x_list_2 = np.append(x_list_2, points[0, 0])

    # y0, y1, y2, y3, ...
    y_list_1 = points[:, 1]

    y_list_2 = points[:, 1][1:].copy()
    # y1, y2, ..., y0
    y_list_2 = np.append(y_list_2, points[0, 1])

    perimeter = np.sum(
        np.sqrt(
            np.power(x_list_1 - x_list_2, 2) + np.power(y_list_1 - y_list_2, 2)
            )
        )
    return perimeter

# Using this formula: https://www.youtube.com/watch?v=jiNXcnGTFbI&t=74s&ab_channel=CivilEr
def get_area(pts):
    """
    Calculate the area of a polygon given its vertices.
    Multiplying each x coordinate by the y coordinate of the next point, then multiplying
    each y coordinate by the x coordinate of the next point, summing these 2 products
    and subtracting the sums and dividing by 2.

    Args:
    pts (ndarray): A 2D ndarray of shape (no of points, 2) representing the x 
                    and y coordinates of the vertices of the polygon.

    Returns:
    area (float): The area of the polygon.
    """
    points = pts.copy()

    prod1 = 0
    prod2 = 0

    # add the last point to be circular
    np.append(points, points[0])

    for i in range(len(points)-1):
        prod1 += points[i, 0]* points[i+1, 1]
        prod2 += points[i, 1]*points[i+1, 0]
    area = (prod1 - prod2)/2
    
    return area

def list_to_string(chain_list):
    """
    Converts an array of numbers to a string of the same sequence, to be used for chain code.

    Args:
    chain_list (ndarray): Chain code as an integer list

    Returns:
    chain_string (str): The final string.
    """
    chain_string = ""
    for element in chain_list:
        chain_string += str(element)
    return chain_string


