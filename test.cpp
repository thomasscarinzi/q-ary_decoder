#include <omp.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/mat_ZZ_pE.h>
#include <NTL/vec_ZZ_pE.h>
#include "decoder.h"
#include <bits/stdc++.h>

using namespace std;
using namespace NTL;

void test1(){
    int mod_base = 3;
    int mod_exp = 6;
    ZZ_p::init(ZZ(mod_base)); // define GF(mod_base)

    ZZ_pX P;
    BuildIrred(P, mod_exp); // generate an irreducible polynomial P
                      // of degree mod_exp over GF(mod_base)

    ZZ_pE::init(P); // define GF(mod_base^mod_exp)
    ZZ_pE a;
    a = random_ZZ_pE();
    int pippo = conv_ZZ_pE_to_int(a, mod_base);
    cout << a << endl;
    cout << pippo << endl;
    cout << conv_int_to_ZZ_pE(pippo, mod_base, mod_exp, P) << endl;
    cout << ZZ_p::modulus() << endl;
    }

void generateRandomRow(Vec<ZZ_pE>& row, int dim, int weight, int mod_base, int mod_exp, ZZ_pX P) {
    int field_dimension = pow(mod_base, mod_exp);
    random_device r;
    vector<default_random_engine> generators;
    for (int i = 0, N = omp_get_max_threads(); i < N; ++i) {
        generators.emplace_back(default_random_engine(r()));
    }

    // Generate a sequence of indices from 0 to dim
    vector<int> allIndices(dim);
    iota(allIndices.begin(), allIndices.end(), 0);
    
    // Shuffle the indices
    uniform_int_distribution<int> distribution(1, field_dimension-1);

    shuffle(allIndices.begin(), allIndices.end(), generators[omp_get_thread_num()]);

    // Take the first row_weight indices
    vector<int> selectedIndices(allIndices.begin(), allIndices.begin() + weight);
    
    for (int i = 0; i < dim; i++) {
        row[i] = to_ZZ_pE(0); // Set every element to 0
    }

    // Setting non-zero elements of the row randomly
    for (int i = 0; i < weight; i++) {
        default_random_engine& engine = generators[omp_get_thread_num()];
        uniform_int_distribution<int> uniform_dist(1, field_dimension-1);
        row[selectedIndices[i]] = conv_int_to_ZZ_pE(uniform_dist(engine), mod_base, mod_exp, P);      
    }
}

// Function to generate a QC-LDPC block starting from the first row
void generateBlock(Mat<ZZ_pE>& block, const Vec<ZZ_pE>& firstRow, int numRows, int numCols) {

    for (int i = 0; i < numCols; i++) {
        block[0][i] = firstRow[i];
    }

    // Extend the block
    for (int i = 1; i < numRows; i++) {
        for (int j = numCols-1; j >= 0; j--) {
            block[i][j] = firstRow[(j - i) % numCols >= 0 ? (j - i) % numCols : (j - i) % numCols + numCols];
        }
    }
}

void generateHMatrix(Mat<ZZ_pE>& H, Mat<ZZ_pE>& block1, Mat<ZZ_pE>& block2, int blockDim){
    for(int i = 0; i < blockDim; i++){
        for (int j = 0; j < blockDim; j++){
        H[i][j] = block1[i][j];
        }
        for(int j = blockDim; j < 2*blockDim; j++){
        H[i][j] = block2[i][j % blockDim];
        }
    }
}

void errorWeightSimulation(int blockDim, int block_weight, int error_weight, int mod_base, int mod_exp, ZZ_pX P, int trials){
    Vec<ZZ_pE> firstRow1, firstRow2;
    Mat<ZZ_pE> block1, block2, H;

    block1.SetDims(blockDim, blockDim);
    block2.SetDims(blockDim, blockDim);
    H.SetDims(blockDim, 2*blockDim);

    firstRow1.SetLength(blockDim); // Set the length based on the desired number of columns
    firstRow2.SetLength(blockDim);

    int n_of_sim = 11;

    ofstream file;
    string filename = "PROVA_PAZZA";
    ostringstream oss;
    oss << filename << "_" << blockDim << "_" << block_weight << "_" << error_weight << "_" << pow(mod_base, mod_exp) << ".csv";
    string formattedFilename = oss.str();
    file.open(formattedFilename);

    generateRandomRow(firstRow1, blockDim, block_weight, mod_base, mod_exp, P); // first row of QC_LDPC block
    generateBlock(block1, firstRow1, blockDim, blockDim); // Now H is a QC-LDPC matrix generated from the first row
    
    generateRandomRow(firstRow2, blockDim, block_weight, mod_base, mod_exp, P); 
    generateBlock(block2, firstRow2, blockDim, blockDim);

    generateHMatrix(H, block1, block2, blockDim);

    bool terminated;
    int DFR_c, real_iter;

    for(int i=0; i<n_of_sim; i++){
        terminated = false;
        DFR_c = 0;
        real_iter = 0;

        #pragma omp parallel 
        {
            while (!terminated){
                #pragma omp atomic
                real_iter++;
                #pragma omp critical
                if (DFR_c >= (trials+9)/10 || real_iter >= trials) {
                    terminated = true;
                }
                int psf_iter;
                
                Vec<ZZ_pE> error;
                Vec<ZZ_pE> estimation_error;
                error.SetLength(2*blockDim); // Error vector dimensions are rowsNum x 1
                estimation_error.SetLength(2*blockDim); 

                #pragma omp critical
                generateRandomRow(error, 2*blockDim, error_weight, mod_base, mod_exp, P);

                // Parallelize the PSF_complete call if possible
                psf_iter = PSF_complete(H, error, estimation_error, block_weight, mod_base, mod_exp, P);

                #pragma omp critical
                {
                    cout << "N. ITERATIONS: " << psf_iter << " AT STEP " << real_iter << " IN CYCLE " << i << endl;
                    file << psf_iter << ","; 
                }
                

                // Update DFR_c based on the result and real_iter;
                #pragma omp atomic
                DFR_c += !(error == estimation_error);
                

                // Check termination condition 
                #pragma omp critical
                if (DFR_c >= (trials+9)/10 || real_iter >= trials) {
                    terminated = true;
                }
            }
        }
        cout << "DFR_c: " << DFR_c << endl;
        cout << "Simulations number: " << real_iter << endl;
        file << "\n";
        error_weight--; 
    }
   file.close();
}

void blockDimensionSimulation(int blockDim, int block_weight, int error_weight, int mod_base, int mod_exp, ZZ_pX P, int trials){
    int n_of_sim = 1;

    ofstream file;
    string filename = "PROVA_PAZZA";
    ostringstream oss;
    oss << filename << "_" << blockDim << "_" << block_weight << "_" << error_weight << "_" << pow(mod_base, mod_exp) << ".csv";
    string formattedFilename = oss.str();
    file.open(formattedFilename);

    Vec<ZZ_pE> firstRow1, firstRow2;
    Mat<ZZ_pE> block1, block2, H;

    bool terminated;
    int DFR_c, real_iter;

    for(int i=0; i<n_of_sim; i++){
        DFR_c = 0;
        real_iter = 0;

        terminated = false;

        // Initialize the matrix with the first row
        block1.SetDims(blockDim, blockDim);
        block2.SetDims(blockDim, blockDim);
        H.SetDims(blockDim, 2*blockDim);

        firstRow1.SetLength(blockDim); // Set the length based on the desired number of columns
        firstRow2.SetLength(blockDim);

        generateRandomRow(firstRow1, blockDim, block_weight, mod_base, mod_exp, P); // first row of QC_LDPC block
        generateBlock(block1, firstRow1, blockDim, blockDim); // Now H is a QC-LDPC matrix generated from the first row
        
        generateRandomRow(firstRow2, blockDim, block_weight, mod_base, mod_exp, P); 
        generateBlock(block2, firstRow2, blockDim, blockDim);
        
        generateHMatrix(H, block1, block2, blockDim);

        #pragma omp parallel 
        {
            while (!terminated){
                #pragma omp atomic
                real_iter++;
                #pragma omp critical
                if (DFR_c >= (trials+9)/10 || real_iter >= trials) {
                    terminated = true;
                }
                int psf_iter;
                Vec<ZZ_pE> error;
                error.SetLength(2*blockDim); // Error vector dimensions are rowsNum x 1
                Vec<ZZ_pE> estimation_error;
                estimation_error.SetLength(2*blockDim); 

                generateRandomRow(error, 2*blockDim, error_weight, mod_base, mod_exp, P);

                // Parallelize the PSF_complete call if possible
                psf_iter = PSF_complete(H, error, estimation_error, block_weight, mod_base, mod_exp, P);

                #pragma omp critical
                {
                    cout << "N. ITERATIONS: " << psf_iter << " AT STEP " << real_iter << " IN CYCLE " << i << endl;
                    file << psf_iter << ","; 
                }
                

                // Update DFR_c based on the result and real_iter;
                #pragma omp atomic
                DFR_c += !(error == estimation_error);
                

                // Check termination condition 
                #pragma omp critical
                if (DFR_c >= (trials+9)/10 || real_iter >= trials) {
                    terminated = true;
                }
            }   
        }
        cout << "DFR_c: " << DFR_c << endl;
        cout << "Simulations number: " << real_iter << endl;
        file << "\n";
        blockDim+=20;
    }
   file.close();
}

int main() {
    int mod_base = 2; // it must be a prime
    int mod_exp = 3;
    ZZ_p::init(ZZ(mod_base)); // define GF(mod_base)

    ZZ_pX P;
    BuildIrred(P, mod_exp); // generate an irreducible polynomial P of degree mod_exp over GF(mod_base)

    ZZ_pE::init(P); // define GF(mod_base^mod_exp)

    int blockDim = 1583; // Set the desired dimension of the cyclic block
    int block_weight = 37; // Set block weight (equal to column weight if H is b x 2b)
    int error_weight = 60;
    int trials = 10000;
    
    errorWeightSimulation(blockDim, block_weight, error_weight, mod_base, mod_exp, P, trials);
    blockDimensionSimulation(blockDim, block_weight, error_weight, mod_base, mod_exp, P, trials)

}
