#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/vector.h>
#include <NTL/vec_ZZ_pE.h>
#include <NTL/matrix.h>
#include <NTL/mat_ZZ_pE.h>
#include <NTL/ZZ_pXFactoring.h>
#include <bits/stdc++.h>

using namespace std;
using namespace NTL;

// Function that converts ZZpE element to an integer
int conv_ZZ_pE_to_int(ZZ_pE x, int mod_base){
    ZZ_pX polynomial = conv<ZZ_pX>(x);
    int result=0;
    if(deg(polynomial)==-1)
        return result;
    else{
        for(int i=0; i<=deg(polynomial); i++){            
            result += conv<int>(coeff(polynomial, i))*pow(mod_base, i);
        }
        return result;
    }
}

// Function that converts ZZpE element to an integer
ZZ_pE conv_int_to_ZZ_pE(int x, int mod_base, int mod_exp, ZZ_pX P){
    if (x > 0){
        ZZ_p::init(ZZ(mod_base));
        ZZ_pE::init(P);
        ZZ_pX result;
        result.SetLength(log2(x)/log2(mod_base)+1);
        int i=0;
        while(x >= mod_base){
            result[i] = conv<ZZ_p>(x);
            x = x/mod_base;
            i +=1;
        }
        result[i] = conv<ZZ_p>(x);
        return conv<ZZ_pE>(conv<ZZ_pX>(result));
    }
    return to_ZZ_pE(0);
}

// Function iniztializing all elements of the error estimation vector to 0
void errorEstimationBuilder(Vec<ZZ_pE>& error_estimation, int columns_number){
    for(int r=0; r<columns_number; r++){
        error_estimation[r] = 0;
    }
}

// Function iniztializing all cells of the Unsatisfied Parity Checks matrix to 0
// As described in the paper, for convenience also satisfied pc are present in the row with index 0
void upcBuilder(Mat<int>& upc, int field_dimension, int columns_number){
    for(int r=0; r<field_dimension; r++){
        for(int s=0; s<columns_number; s++){
            upc[r][s]=0;
        }
    }
}

// Function counting upc for specific column j
void upcAdder(Mat<ZZ_pE>& H, Vec<ZZ_pE> syndrome, Mat<int>& upc, int j, int mod_base){
    int rows_number = H.NumRows();
    ZZ_pE element(0);
    int elem_int; 
    
    for(int k=0; k<rows_number; k++){
        if(H[k][j]!=0){
            element = syndrome[k]*inv(H[k][j]);
            elem_int = conv_ZZ_pE_to_int(element, mod_base);
            upc[elem_int][j] = upc[elem_int][j] + 1;
        }
    }   
}

// Function that returns the value of the maximum upc for column j
int max_on_column(Mat<int>& upc, int field_dimension, int j){
    int max=0;
    for(int i=1; i<field_dimension; i++){
        if(upc[i][j]>max)
            max = upc[i][j];
    }
    return max;
}

// Function that returns the index of the element of GF(q) where the maximum upc for column j sits
// If multiples max tied, uses picks randomly one element with maximum counter
int field_element_of_maxUPC(Mat<int>& upc, int max_upc_j, int field_dimension, int j){
    Vec<int> max_indices;

    // Cycle to find all indices with same macimum value
    for (int i = 1; i < field_dimension; i++) {
        if (upc[i][j] == max_upc_j) {
            max_indices.append(i);
        }
    }

    // If single index return that element
    if (max_indices.length() == 1)
        return max_indices[0];
    // If multiple indexes return a random element
    else {
        // Seed the random number generator with current time
        srand(static_cast<unsigned>(time(0)));
        int random_index = rand() % max_indices.length();
        return max_indices[random_index];
    }  
    
}

// Function that creates the statistics of the upc
void upcStats(Mat<int>& upc){
    ofstream file;
    file.open("YOUR_FILE_NAME.csv");
    for(int j=0; j<upc.NumCols(); j++){
        for(int i=0; i<upc.NumRows(); i++){
            file << upc[i][j] << ",";
        }
        file << "\n";
    }
    file.close();
}

// PSF algorithm
void PSF(Mat<ZZ_pE>& H, Vec<ZZ_pE>& syndrome, Vec<ZZ_pE>& error_estimation, int mod_base, int mod_exp, int T_E, int T_0, int T_D, ZZ_pX P, int iter){
    int field_dimension = pow(mod_base, mod_exp);
    int rows_number = H.NumRows();
    int columns_number = H.NumCols();
    int max_upc_j, l;
    Mat<int> upc;
    upc.SetDims(field_dimension, columns_number);    
    upcBuilder(upc, field_dimension, columns_number);
    for(int j=0; j<columns_number; j++){
        l = 0;
        upcAdder(H, syndrome, upc, j, mod_base);

        max_upc_j = max_on_column(upc, field_dimension, j);
        if(max_upc_j>=T_E && upc[0][j]<T_0 && max_upc_j-upc[0][j]>=T_D){
            l = field_element_of_maxUPC(upc, max_upc_j, field_dimension, j);
        }
        error_estimation[j] +=  conv_int_to_ZZ_pE(l, mod_base, mod_exp, P);
    }
    if(iter==0)
    upcStats(upc);
}

// PSF algorithm with thresholds and syndrome updating
int PSF_complete(Mat<ZZ_pE>& H, Vec<ZZ_pE>& real_error, Vec<ZZ_pE>& error_estimation, int column_weight, int mod_base, int mod_exp, ZZ_pX P){
    int field_dimension = pow(mod_base, mod_exp);
    //thresholds to be decided studying upc stats
    int T_E = 9;
    int T_0 = 10;;
    int T_D = ceil(2*sqrt(double(column_weight)*double(field_dimension-1)/double(field_dimension*field_dimension))); // piccola poco più grande di 2

    int i = 0;
    int MAX_ITER = 5;
    Vec<ZZ_pE> syndrome;


    syndrome.SetLength(H.NumRows());
    syndrome = H * real_error;   

    while((real_error != error_estimation) && i < MAX_ITER){
        PSF(H, syndrome, error_estimation, mod_base, mod_exp, T_E, T_0, T_D, P, i);
        syndrome = syndrome - (H * error_estimation);
        i++;
    }
    if(real_error != error_estimation)
        i++;
    return i;
}