#ifndef GPUFIT_TISSUE_UPTAKE_TRANS_CUH_INCLUDED
#define GPUFIT_TISSUE_UPTAKE_TRANS_CUH_INCLUDED

/* Description of the calculate_tissue_uptake_trans function
* ==========================================================
*
* This function calculates the values of one-dimensional transformed Tissue Uptake model functions
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
*             p[1]: Tp
*			  p[2]: Fp
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
* Calling the calculate_tissue_uptake_trans function
* ==================================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_tissue_uptake_trans(
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
    REAL const * p = parameters;			// p = [p[0], p[1], p[2]]
    REAL const * C_p = (REAL*)user_info;	// C_p è un vettore di n_points elementi (65)

	REAL const s1 = sin(p[0]);
    REAL const c1 = cos(p[0]);
	REAL const s3 = sin(p[2]);
	REAL const c3 = cos(p[2]);

	
	REAL x[65];
    REAL argx[65];
	REAL ex[65];
	REAL exx[65];
	
    int i;
	i = 0; 
    while (i < 65){	
		x[i] = i + 1; 							// Definisco x come un vettore di 65 elementi x = [0,1,...,64], x[0] = 1; x[1] = 2; ... ; x[64] = 65					
        argx[i] = 1 / abs(p[2]) * x[i];			// Definisco il vettore argx come un vettore di n_points elementi (argomento dell'esponenziale)
		ex[i] = exp(-argx[i]);					// Definisco il vettore ex come un vettore di n_points elementi (esponenziale)
        exx[i] = x[i] * exp(-argx[i]);			// Definisco il vettore exx di n_points elementi come il vettore ex moltiplicato per il vettore x (elemento per elemento) e per 7.5.
		i = i + 1;
        }
	
	// Definisco lo scalare conv(conv2-3-4) come la convoluzione tra Cp e ex valutato nel punto point_index:
	// conv[point_index] = Cp[0] * ex[point_index] + Cp[1] * ex[point_index - 1] + ... + Cp[point_index] * ex[0]
    REAL conv;
    conv = 0;
	REAL conv2;
	conv2 = 0;
	REAL conv3;
	conv3 = 0;
	REAL conv4;
	conv4 = 0;
	int j;
	int k;
    for (j = 0; j <= point_index; ++j){
		for (k = point_index; k >= 0; --k){
			conv = conv + (C_p[j] * (((s1 + 1) + (7.5 * (s3 + 1) - s1 - 1)) * ex[k]));
			conv2 = conv2 + (C_p[j] * (1 - ex[k]));
			conv3 = conv3 + (C_p[j] * exx[k]);
			conv4 = conv4 + (C_p[j] * ex[k]);
        	}
		}
	
	// Viene definito lo scalare value[point_index] come a 
    value[point_index] = conv;   
	   
    // derivative
	// Il puntatore current_derivative punta alla posizione di memoria derivative + point_index
	// point_index varia ad ogni chiamata della funzione da 0 a n_points - 1
    REAL * current_derivative = derivative + point_index;
		
	// Derivata parziale rispetto a lambda1 calcolata nel punto point_index
    current_derivative[0 * n_points] = c1 * conv2;
	
	// Derivata parziale rispetto a lambda2 calcolata nel punto point_index
    current_derivative[1 * n_points] = (7.5 * (s3 + 1) - s1 - 1) * p[1] / abs(p[1]) / abs(p[1]) / abs(p[1]) * conv3;

	// Derivata parziale rispetto a lambda3 calcolata nel punto point_index
    current_derivative[2 * n_points] = 7.5 * c3 * conv4;
}

#endif
