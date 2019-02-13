#ifndef GPUFIT_PATLAK_TRANS_CUH_INCLUDED
#define GPUFIT_PATLAK_TRANS_CUH_INCLUDED

/* Description of the calculate_patlak_trans function
* ===================================================
*
* This function calculates the values of one-dimensional transformed Patlak model functions
* and their partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* The (X) coordinate of the first data value is assumed to be (0.0).
* The (X) coordinates of the data are simply the corresponding array 
* index values of the data array, starting from zero.
* Parameters: An input vector of model parameters.
*             p[0]: Ktrans
*             p[1]: vp
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* user_info: An input vector containing user information. 
*
* user_info_size: The size of user_info in bytes. 
*
* Calling the calculate_patlak_trans function
* ===========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_patlak_trans(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // parameters
    REAL const * p = parameters;
    REAL const * Cp = (REAL*)user_info;
	REAL const s1 = sin(p[0]);
    REAL const c1 = cos(p[0]);
	REAL const s2 = sin(p[1]);
	REAL const c2 = cos(p[1]);
	
    // value
	// Viene definito lo scalare a come l'integrale di Cp nel tempo, con t che va da 0 a point_index:
	// a = Cp[0] + Cp[1] + ... + Cp[point_index]
    REAL a;
	a = 0;
	int i;
	for (i = 0; i < point_index + 1; i++) {
		a = a + Cp[i];
	}
	
	// Viene definito lo scalare value[point_index] come a moltiplicato per (sin(lambda1) + 1) e 
	// al quale viene sommato 0.5 * (sin(lambda2) + 1) * Cp[point_index]
    value[point_index] = 0.5 * (s2 + 1) * Cp[point_index] + (s1 + 1) * a;   
	   
    // derivative
	// Il puntatore current_derivative punta alla posizione di memoria derivative + point_index
	// point_index varia ad ogni chiamata della funzione da 0 a n_points - 1
    REAL * current_derivative = derivative + point_index;
	
	// Derivata parziale rispetto a lambda1 calcolata nel punto point_index
    current_derivative[0 * n_points] = a * c1;
	
	// Derivata parziale rispetto a lambda2 calcolata nel punto point_index
    current_derivative[1 * n_points] = 0.5 * Cp[point_index] * c2;
}

#endif
