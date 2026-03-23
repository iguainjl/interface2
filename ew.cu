#include<thrust/device_vector.h>
#include<thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include<thrust/reduce.h>
#include<fstream>
#include<cstdlib>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <cufft.h>
#include "cutil.h"
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>
#include <thrust/complex.h>


// harmonic elasticity constant
//#ifndef C2
//#define C2 1.0   
//#endif

// anharmonic elasticity constant
/*
#ifndef C4
#define C4 0.0   
#endif
*/

// harmonic elasticity constant
/*
#ifndef KPZ
#define KPZ 1.0   
#endif
*/

#ifndef Dt
#define Dt 0.1   
#endif


// noise temperature
#ifndef TEMP
#define TEMP 0.1
#endif

/*#ifndef seed    
#define seed 1234
#endif
*/

// noise correlation time
#ifndef TAU    
#define TAU 0.1
#endif

// tilted boundary conditions
/*
#ifndef TILT
#define TILT 0.0
#endif
*/

// monitor some quantities every MONITOR steps
#ifndef MONITOR
#define MONITOR 10000
#endif

// prints whole configurations every MONITORCONF steps
#ifndef MONITORCONF
#define MONITORCONF 100000
#endif

// define to work in double or simple precision
#ifdef DOUBLE
typedef double real;
typedef cufftDoubleComplex complex;
#else
typedef float real;
typedef cufftComplex complex;
#endif


// kernel to initialize wave numbers in Fourier space (translate to thrust later...)
__global__ void init_wave_numbers(complex* L_k, int N, real K, real L) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        int k = (i <= N/2) ? i : i - N;
        real kx = 2 * M_PI * k / L;
        #ifdef DOUBLE
        L_k[i] = make_double2(-K * kx * kx, 0.0);
        #else
        L_k[i] = make_float2(-K * kx * kx, 0.0f);
        #endif
    }
}


// file to log parameters of the run
std::ofstream logout("logfile.dat");


// main class:
class cuerda{

    public:
    cuerda(unsigned long _L, real _dt, unsigned long _seed):L(_L),dt(_dt),fourierCount(0),seed(_seed)
    {
        // interface position
        u.resize(L);
        
        // interface forces
        force_u.resize(L);
  
        // interface forces
        noise.resize(L);
        thrust::fill(noise.begin(),noise.end(),real(0.0));
        warmup_noise(); // warmup noise
 
        // flat initial condition
        thrust::fill(u.begin(),u.end(),real(0.0));
        
        // plans for the interface structure factor
        #ifdef DOUBLE
        CUFFT_SAFE_CALL(cufftPlan1d(&plan_r2c,L,CUFFT_D2Z,1));
        CUFFT_SAFE_CALL(cufftPlan1d(&plan_c2r,L,CUFFT_Z2D,1));
        #else
        CUFFT_SAFE_CALL(cufftPlan1d(&plan_r2c,L,CUFFT_R2C,1));
        CUFFT_SAFE_CALL(cufftPlan1d(&plan_c2r,L,CUFFT_C2R,1));
        #endif

	    int Lcomp=L/2+1;
	    Fou_u.resize(Lcomp); // interface position in fourier space

        acum_Sofq_u.resize(L); // average structure factor
        inst_Sofq_u.resize(L); // instantaneous structure factor

        // initialization of structure factors   
        thrust::fill(acum_Sofq_u.begin(),acum_Sofq_u.end(),real(0.0));

        #ifdef DEBUG
        std::cout << "L=" << L << ", dt=" << dt << std::endl;
        #endif
        
        #ifdef SPECTRALCN
        z.resize(Lcomp); // Fourier components of the interface position 
        z_hat.resize(Lcomp); // Fourier components of the interface position (complex)
        nonlinear.resize(Lcomp); // nonlinear term in Fourier space
        L_k.resize(Lcomp); // wave numbers in Fourier space
        
        //init_wave_numbers<<<(Lcomp+255)/256,256>>>(thrust::raw_pointer_cast(&L_k[0]), Lcomp, C2, L);

        // // initialize wave numbers in Fourier space
        // thrust::for_each(thrust::make_counting_iterator(0),
        //                   thrust::make_counting_iterator(Lcomp),
        //                   [=] __device__ (unsigned long i) {
        //                        int k = (i <= Lcomp-1) ? i : i - Lcomp;
        //                        real kx = 2 * M_PI * k / Lcomp;
        //                        L_k[i] = make_float2(-C2 * kx * kx, 0.0f);
        //                   });
        #endif

    }

    void flat_initial_condition(){
        // flat initial condition
        thrust::fill(u.begin(),u.end(),real(0.0));
    }

    ///////////// needed to reach steady state noise	
    //just start noise (one step)

    void warmup_noise(){
        real dt_ = dt;
        unsigned long seed_ = seed;
        unsigned long L_ = L;
        
    //unsigned long twarm = (unsigned long )(100.*TAU/dt_); // number of warmup steps
    unsigned long twarm = (unsigned long )(1); // one step
        for(unsigned long n=0;n<twarm;n++)
        {
            thrust::for_each(
                thrust::make_zip_iterator(
                    thrust::make_tuple(noise.begin(),thrust::make_counting_iterator((unsigned long)0))        
                ),
                thrust::make_zip_iterator(
                    thrust::make_tuple(noise.end(),thrust::make_counting_iterator((unsigned long)L_))        
                ),
                [=] __device__ (thrust::tuple<real &,unsigned long> t)
                {
                    unsigned long i=thrust::get<1>(t);
                    curandStatePhilox4_32_10_t state;
                    curand_init(seed_, i, n, &state);
                    //real ran = sqrtf(2*TEMP*dt_)*curand_normal(&state);
                    //thrust::get<0>(t) += -thrust::get<0>(t)*dt_/TAU + ran/TAU;
                    real ran = sqrtf(TEMP/TAU)*curand_normal(&state);
                    thrust::get<0>(t) = ran;
                } 
            );  
        }
    };


    // reset structure factor acumulator
    void reset_acum_Sofq(){
        thrust::fill(acum_Sofq_u.begin(),acum_Sofq_u.end(),real(0.0));
    }

    // returns the center of mass position
    real center_of_mass()
    {
        //DANGER: large sum over large numbers
        real cmu = thrust::reduce(u.begin(),u.end(),real(0.0))/L;
        return cmu;
    }

    // returns the center of mass velocity
    real center_of_mass_velocity()
    {
        //SAFE: velocities are bounded
        real vcmu = thrust::reduce(force_u.begin(),force_u.end(),real(0.0))/L;
        return vcmu;
    }

    // computes the instantaneous and acumulated structure factor
    void fourier_transform(){

        real *raw_u = thrust::raw_pointer_cast(&u[0]); 
        complex *raw_fou_u = thrust::raw_pointer_cast(&Fou_u[0]); 

        // raw_u --> transform --> raw_fou_u
        #ifdef DOUBLE
        CUFFT_SAFE_CALL(cufftExecD2Z(plan_r2c, raw_u, raw_fou_u));
        #else
	    CUFFT_SAFE_CALL(cufftExecR2C(plan_r2c, raw_u, raw_fou_u));
        #endif

        // compute the structure factor from fourier components
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(Fou_u.begin(),acum_Sofq_u.begin(),inst_Sofq_u.begin())
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(Fou_u.end(),acum_Sofq_u.end(),inst_Sofq_u.end())
            ),
            [=] __device__ (thrust::tuple<complex,real &,real &> t)
            {
                complex fu=thrust::get<0>(t);
                real sofq = fu.x*fu.x + fu.y*fu.y;
                thrust::get<1>(t) += sofq; // average structure factor 
                thrust::get<2>(t) = sofq; //instantaneous structure factor
            }
        );
        fourierCount++; // increment the number of fourier transforms
    }

    // computes the center of mass, the variance (roughness)
    // and the leading and receding points of the interface 
    thrust::tuple<real, real, real, real> roughness()
    {
        // CHECK for large numbers!
        
        // center of mass displacement
        real cmu = thrust::reduce(u.begin(),u.end(),real(0.f),thrust::plus<real>())/real(L);
	    
	// extreme displacements
	real u0=u[0]; 
        real maxu = thrust::reduce(u.begin(),u.end(),u0,thrust::maximum<real>());
        real minu = thrust::reduce(u.begin(),u.end(),u0,thrust::minimum<real>());

        // variance or roughness
        real cmu2 = 
        thrust::transform_reduce(
            u.begin(),u.end(),
            [=] __device__ __host__ (real x){
                return (x-cmu)*(x-cmu);
            },
            real(0.f),
            thrust::plus<real>()
        )/real(L);

        return thrust::make_tuple(cmu,cmu2,maxu,minu);
    }

    // just compute and prints center of mass in out stream
    void print_center_of_mass(std::ofstream &out)
    {
        real cm=center_of_mass();    
        out << cm << std::endl;
    }

    // rescale all position in order to avoid large displacements
    void rescale()
    {
        real cmu=center_of_mass();

        thrust::transform(u.begin(),u.end(),u.begin(),
        [=] __device__ (real u){
            return u-cmu;
        }
        );
    };

    // print roughness results
    void print_roughness(std::ofstream &out, real t)
    {
        real vcm=center_of_mass_velocity();

        thrust::tuple<real,real,real,real> cm = roughness();

        //get cmu,cmu2,maxu,minu
        real cmu = thrust::get<0>(cm);
        real cmu2 = thrust::get<1>(cm);
        real maxu = thrust::get<2>(cm);
        real minu = thrust::get<3>(cm);

        out << t << " " << vcm << " " << cmu << " " << " " << cmu2 << " " << maxu << " " << minu << std::endl;
    }

    // Computes the forces and advance one time step using Euler method
    void update(unsigned long n)
    {
        real *raw_u = thrust::raw_pointer_cast(&u[0]); 
        real *raw_noise = thrust::raw_pointer_cast(&noise[0]); 

        // variables to be captured by lambda (not elegant...)
        real dt_=dt;
        unsigned long L_ = L;
        unsigned long seed_ = seed;
        // Forces
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(force_u.begin(),thrust::make_counting_iterator((unsigned long)0))        
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(force_u.end(),thrust::make_counting_iterator((unsigned long)L))        
            ),
            [=] __device__ (thrust::tuple<real &,unsigned long> t)
            {
                unsigned long i=thrust::get<1>(t);
                unsigned long ileft = (i-1+L_)%L_;
                unsigned long iright = (i+1)%L_;

                real uleft = raw_u[ileft];
                real uright = raw_u[iright];
                
                // optional to impose tilted boundary conditions
                #ifdef TILT
                if(i==0) {
                    uleft -= L_*TILT;
                }  
                if(i==L_-1){
                    uright += L_*TILT;
                }  
                #endif
                
                // correlated noise update 
                curandStatePhilox4_32_10_t state;
                curand_init(seed_, i, n, &state);
                //real ran = sqrt(2*TEMP*dt_)*curand_normal(&state);
                //raw_noise[i] += -raw_noise[i]*dt_/TAU + ran/TAU;
                real ran = sqrt(TEMP*(1-1/exp(2*dt_/TAU))/TAU)*curand_normal(&state);
                raw_noise[i] = raw_noise[i]/exp(dt_/TAU) + ran;                        
                real lap_u = (uright + uleft - 2.0*raw_u[i]);

                // modify element force
                thrust::get<0>(t) = raw_noise[i];
                
                	#ifdef C2
        		thrust::get<0>(t) += C2*( powf(uright - raw_u[i],1.0) - powf(raw_u[i]-uleft,1.0) );	                
                	#endif
        		
        		#ifdef C4
        		thrust::get<0>(t) += C4*( powf(uright - raw_u[i],3.0) - powf(raw_u[i]-uleft,3.0) );	
        		#endif
        
			#ifdef C6
                        thrust::get<0>(t) += C6*( powf(uright - raw_u[i],5.0) - powf(raw_u[i]-uleft,5.0) );
                        #endif


        		#ifdef C12
        		thrust::get<0>(t) += C12*( powf(uright - raw_u[i],11.0) - powf(raw_u[i]-uleft,11.0) );	
        		#endif
        
        		#ifdef KPZ
                        thrust::get<0>(t) += KPZ*powf(0.5*(uright-uleft),2.0f);
        		#endif
            } 
        );

        #ifdef DEBUG
        std::cout << "updating" << std::endl;
        #endif

        // Euler step: u = u + dt*force_u
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(u.begin(), force_u.begin())        
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(u.end(),force_u.end())        
            ),
            [=] __device__ (thrust::tuple<real &,real> t)
            {
                thrust::get<0>(t) = thrust::get<0>(t) + dt_*thrust::get<1>(t);
            } 
        );
    };

    // update using the spectral method
    #ifdef SPECTRALCN
    void update_spectral(unsigned long n){
    
        real *raw_u = thrust::raw_pointer_cast(&u[0]); 
        complex *raw_fou_u = thrust::raw_pointer_cast(&Fou_u[0]); 

        // raw_u --> transform --> raw_fou_u
        #ifdef DOUBLE
        CUFFT_SAFE_CALL(cufftExecD2Z(plan_r2c, raw_u, raw_fou_u));
        #else
	    CUFFT_SAFE_CALL(cufftExecR2C(plan_r2c, raw_u, raw_fou_u));
        #endif
    
    }
    #endif


    // print the whole configuration to a file
    void print_config(std::ofstream &out){
        real cm = center_of_mass();

        for(int i=0;i<L;i++){
            out << u[i] << " " << cm << "\n";
        }
        out << "\n" << std::endl;
    };

    // prints the whole averaged structure factor to a file
    void print_sofq(std::ofstream &out){
        for(int i=0;i<L;i++){
            out << acum_Sofq_u[i]/fourierCount << "\n";
        }
        out << "\n" << std::endl;
    };

    // prints the instantaneous structure factor to a file
    void print_inst_sofq(std::ofstream &out){
        for(int i=0;i<L;i++){
            out << inst_Sofq_u[i] << "\n";
        }
        out << "\n" << std::endl;
    };

    // variables and arrays of the class
    private:
        real dt;
        unsigned long L;
        unsigned long seed;
        
        real f0;
        thrust::device_vector<real> u;
        thrust::device_vector<real> force_u;

        thrust::device_vector<real> noise;


        // variables for the structure factor
        int fourierCount;
        cufftHandle plan_r2c;
        cufftHandle plan_c2r;
        thrust::device_vector<complex> Fou_u;
        thrust::device_vector<real> acum_Sofq_u;
	    thrust::device_vector<real> inst_Sofq_u;
	    
	    // variables for the spectral method
        #ifdef SPECTRALCN
        thrust::device_vector<complex> z;
        thrust::device_vector<complex> z_hat;
        thrust::device_vector<complex> nonlinear;
        thrust::device_vector<complex> L_k;
        thrust::device_vector<complex> zaux;	    
        #endif
};

int main(int argc, char **argv){
    // Get the current CUDA device
    int device;
    cudaGetDevice(&device);

    // Get the properties of the current CUDA device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::ofstream confout("conf.dat");
    confout << "#u[i]" << " " << "cmu" << "\n";

    std::ofstream sofqout("sofq.dat");
    sofqout << "#av_Sofq_u[i]" << "\n";

    std::ofstream instsofqout("inst_sofq.dat");
    instsofqout << "#inst_Sofq_u[i]" << "\n";

    std::ofstream cmout("cm.dat");
    cmout << "#t" << " " << "velu" << " " << "cmu" << " " << "cmu2" << " " << "maxu" << " " << "minu" << std::endl;

    std::ofstream cmlogout("cmlog.dat");
    cmlogout << "#t" << " " << "velu" << " " << "cmu" << " " << "cmu2" << " " << "maxu" << " " << "minu" << std::endl;

    std::ofstream lastconfout("lastconf.dat");
    lastconfout << "#u[i]" << " " << "cmu" << "\n";

    if(argc!=4){
        std::cout << "Usage: " << argv[0] << " L Nrun seed" << std::endl;
        std::cout << "L: interface length" << std::endl;
        std::cout << "Nrun: number of running steps" << std::endl;
        std::cout << "seed: random seed" << std::endl;
        return 1;
    }

    unsigned int L=atoi(argv[1]); //interface lenght
    unsigned long Nrun = atoi(argv[2]); // running steps
    unsigned long seed = atoi(argv[3]); // global seed
    
    // time step
    real dt=Dt;

    // equilibration
    unsigned long Neq = int(Nrun*0.75); // number of equilibration steps

    // instance
    cuerda C(L,dt,seed);

    #ifdef DOUBLE
    logout << "double precision\n";
    #else
    logout << "simple precision\n";
    #endif
    #ifdef TILT
    logout << "TILT= " << TILT << "\n";
    #endif
    #ifdef TEMP
    logout << "TEMP= " << TEMP << "\n";
    #endif    
    #ifdef seed
    logout << "seed= " << seed << "\n";
    #endif 
    #ifdef MONITOR
    logout << "MONITOR= " << MONITOR << "\n";
    #endif
    #ifdef MONITORCONF
    logout << "MONITORCONF= " << MONITORCONF << "\n";
    #endif
    #ifdef C2
    logout << "C2= " << C2 << "\n";
    #endif
    #ifdef C4
    logout << "C4= " << C4 << "\n";
    #endif
    #ifdef C12
    logout << "C12= " << C12 << "\n";
    #endif
    #ifdef KPZ
    logout << "KPZ= " << KPZ << "\n";
    #endif
    #ifdef TAU
    logout << "TAU= " << TAU << "\n";
    #endif
    logout 
	<< "dt= " << dt << "\n"
	<< "L= " << L << std::endl;
    logout.flush();

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    unsigned long jlog=1;
    unsigned long jlogx=1;

    for(int i=0;i<=Nrun;i++){
        C.update(i);

        if(i%MONITORCONF==0){
            C.print_config(confout);
            C.fourier_transform();
            C.print_inst_sofq(instsofqout);
        }
                
        // print configs and structure factors at 1,10,100,etc...        
        if(i%jlog==0){
            C.print_config(confout);
            C.fourier_transform();
            C.print_inst_sofq(instsofqout);
            jlog*=10;
        }
        
        if(i%jlogx==0){
	        C.print_roughness(cmlogout,dt*i);
            jlogx*=2;
        }
        
        //if(i%Neq==0) C.reset_acum_Sofq();
                        
        if(i%MONITOR==0){
            C.print_roughness(cmout,dt*i);
        }
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    C.print_config(confout);
    C.print_sofq(sofqout);

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    // Output the duration
       
    logout << "Time taken: " << duration.count() << " seconds\n L=" << L << " Nrun=" << Nrun << std::endl;
    logout << "device= " << deviceProp.name << std::endl;

    logout << "Performance[s,L,Nrun]: " << duration.count() << " " << L << " " << Nrun << std::endl;

    return 0;
}

/*
nvcc --expt-extended-lambda -lcufft main.cu -DCu=0.0 -DCphi=0.0 -DEpsilon=0.001 -std=c++14 -arch=sm_61 -o a0.out
*/
