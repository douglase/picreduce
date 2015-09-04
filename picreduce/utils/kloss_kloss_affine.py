
import numpy
def solve(point_list):
    """
    This function solves the linear equation system involved in the n
    dimensional linear extrapolation of a vector field to an arbitrary point.

    f(x) = x * A + b

    with:
        A - The "slope" of the affine function in an n x n matrix.
        b - The "offset" value for the n dimensional zero vector.

    The function takes a list of n+1 point-value tuples (x, f(x)) and returns
    the matrix A and the vector b. In case anything goes wrong, the function
    returns the tuple (None, None).

    These can then be used to compute directly any value in the linear
    vector field.
    """
    # Some helpers.
    dimensions = len(point_list[0][0])
    unknowns = dimensions ** 2 + dimensions
    number_points = len(point_list[0])
    print(point_list,len(point_list[0]))
    print('unknowns:'+str(unknowns))
    # Bail out if we do not have enough data.
    if number_points < unknowns:
        print 'For a %d dimensional problem I need at least %d data points.' \
              % (dimensions, unknowns)
        print 'Only %d data points were given.' % number_points
        #return None, None
    
    # Ensure we are working with a NumPy array.
    point_list = numpy.asarray(point_list)
    
    # For the solver we are stating the problem as
    # C * x = d
    # with the problem_matrix C and the problem_vector d

    # We're going to feed our linear problem into these arrays.
    # This one is the matrix C.
    problem_matrix = numpy.zeros([unknowns, unknowns])
    # This one is the vector d.
    problem_vector = numpy.zeros([unknowns])

    # Populate data matrix C and vector d.
    x_values, y_values = point_list[0], point_list[1]
    for i in range(dimensions):
        #for each row, get x_i vector and y_i vector
        x_i, y_i = x_values[:, i], y_values[:, i]
        print("xi:"+str([x_i,i]))
        for j in range(dimensions):
            #for each column
            #get the y_vector
            y_j = y_values[:, j]
            #calculate the problem vector row
            row = dimensions * i + j
            #find the sum of the vector multiplication of x_i and y_j and set equal to problem vector, 
            #this is only time y_j is called.
            problem_vector[row] = (x_i * y_j).sum()
            #print("result1: "+str(problem_vector[row])) 
            #find the sum of x_i
            problem_matrix[row, dimensions ** 2 + j] = x_i.sum()
            print("xi.sum()"+str(x_i.sum()))
            #sum of x_i again
            problem_matrix[dimensions ** 2 + j, dimensions * i + j] = x_i.sum()
            for k in range(dimensions):
                
                x_k = x_values[:, k]
                print("(x_i * x_k).sum()"+str((x_i * x_k).sum()))
                problem_matrix[row, dimensions * k + j] = (x_i * x_k).sum()
        row = dimensions ** 2 + i
        print("shape yi:"+str(numpy.shape(y_i)))
        problem_vector[row] = y_i.sum()
        problem_matrix[row, dimensions ** 2 + i] = number_points
    print("problem matrix:")
    print(problem_matrix)
    print("Problem Vector:")
    print(problem_vector)
    matrix_A, vector_b = None, None
    try:
        result_vector = numpy.linalg.solve(problem_matrix, problem_vector)

        # Check whether we really did get the right answer.
        # This is advised by the NumPy doc string.
        if numpy.linalg.norm(numpy.dot(problem_matrix, result_vector)
                             - problem_vector) < 1e-6:
            # We're good, so hack up the result into the matrix and vector.
            matrix_A = result_vector[:dimensions ** 2]
            matrix_A.shape = (dimensions, dimensions)
            vector_b = result_vector[dimensions ** 2:]
        else:
            print "For whatever reason our linear equations didn't solve."
            print numpy.linalg.norm(numpy.dot(result_vector, problem_matrix)
                                    - problem_vector)
    except numpy.linalg.linalg.LinAlgError:
        print "Things didn't work out as expected, eh."

    return matrix_A, vector_b



def print_solve(estimated_A,estimated_b,dimensions):
    if estimated_A is not None:

        # And test it on a nice point somewhere
        # dest_point = numpy.random.uniform(low=0.0, high=10.0,size=dimensions)

        # expected_value = numpy.dot(dest_point, mystery_matrix) + mystery_vector
        
        estimated_value = numpy.dot(dest_point, estimated_A) + estimated_b
        
        # distance = numpy.linalg.norm(expected_value - estimated_value)
        
        
        
        print 'Estimated:\n\t%s' % estimated_value.round(3)


    else:

        print 'Sorry, problem could not be solved.'
        
        
        
        print(y_values)
        
        print(x_values)
