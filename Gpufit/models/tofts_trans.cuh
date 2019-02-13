#ifndef GPUFIT_TOFTS_TRANS_CUH_INCLUDED
#define GPUFIT_TOFTS_TRANS_CUH_INCLUDED
#include <vector>

/* Description of the calculate_tofts_trans function
* ==================================================
*
* This function calculates the values of one-dimensional transformed tofts model functions
* and their partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* The (X) coordinate of the first data value is assumed to be (0.0). 
* The (X) coordinates of the data are simply the corresponding array
* index values of the data array, starting from zero.
*
* Parameters: An input vector of model parameters.
*             p[0]: Ktrans
*             p[1]: ve
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
* Calling the calculate_tofts_trans function
* ==========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_tofts_trans(
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

    REAL x[65];
	REAL argx[65];
	REAL ex[65];
	REAL exx[65];
	
	REAL conv;
	conv = 0;
	REAL conv2;
	conv2 = 0;
    int i;
	i = 0; 
	int j;
	int k;
	
    while (i < 65){ 						// n_points = 65
        x[i] = (REAL)(i + 1); 				// Definisco x come un vettore di 65 elementi x = [0,1,...,64], x[0] = 1; x[1] = 2; ... ; x[64] = 65
		argx[i] = (49 * s2 + 51) * x[i]; 	// Definisco il vettore argx come un vettore di n_points elementi (argomento dell'esponenziale)
		ex[i] = exp(-argx[i]);				// Definisco il vettore ex come un vettore di n_points elementi (esponenziale)
		exx[i] = ex[i] * x[i];				// Definisco il vettore exx come un vettore di n_points elementi (esponenziale * t)
        i = i + 1;
		}
	
	// Definisco lo scalare conv come la convoluzione tra Cp e ex valutato nel punto point_index:
	// conv[point_index] = Cp[0] * ex[point_index] + Cp[1] * ex[point_index - 1] + ... + Cp[point_index] * ex[0]
	for(j = 0; j <= point_index; j++){
		for(k = point_index; k >= 0; k--){
			conv = conv + Cp[j] * ex[k];
			conv2 = conv2 + Cp[j] * exx[k];
		}
	}
	
	// Definisco lo scalare value come la convoluzione di sopra moltiplicata per il sin(lambda1) + 1
	value[point_index] = (s1 + 1) * conv;
	
	// derivative
	// current_derivative punta alla posizione di memoria derivative + point_index
	// ad ogni chiamata della funzione point_index è diverso e varia tra 0 e n_points - 1
    REAL * current_derivative = derivative + point_index;
	
	current_derivative[0 * n_points] = c1 * conv;
		
    current_derivative[1 * n_points] = - 49 * (s1 + 1) * c2 * conv2;
}
#endif
	
