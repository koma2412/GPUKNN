// C++ program to find groups of unknown
// Points using K nearest neighbour algorithm.
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cuda.h>
#define noOfClasses 2
struct Point
{
    int val;     // Co-ordinate of point
    float distance; // Distance from test point
};

int max(int freq[]){
    int m=freq[0];
    int index=0;
    for(int i=1;i<noOfClasses;i++){
        if(m<freq[i]){
            m=freq[i];
            index=i;
        }
    }
    return index+1;
}
// Used to sort an array of points by increasing
// order of distance
bool comparison(Point a, Point b)
{
    return (a.distance < b.distance);
}

int cpuClassify(int arr[], long n, int k,int attrib){
	int sum;
    long p=0;
    Point *result = (Point *)malloc(n*sizeof(Point));
    for (long i = 0; i < attrib*n; i=i+attrib)
        {
            sum=0;
            for(int j=0;j<attrib-1;j++){
                sum = sum + (arr[i+j]-arr[attrib*n+j])*(arr[i+j]-arr[attrib*n+j]);
            }
            result[p].distance=sqrt(sum);
            result[p].val=arr[i+attrib-1];

            p++;
        }
    // Sort the Points by distance from p
    std::sort(result, result+n, comparison);
    //for(long i=0;i<n;i++)
      //  printf("\n%f %d",result[i].distance,result[i].val);
    int freq[noOfClasses];
    for(int i=0;i<noOfClasses;i++)
        freq[i]=0;
    for (int i = 0; i < k; i++)
    {
        freq[result[i].val-1]++;
    }
    return max(freq);
}
__global__ void DistanceKernel(int *ga, int size, Point *gResult,int attrib)
{
	int i;
	int sum;
	i=(blockIdx.x*blockDim.x)+threadIdx.x;	
	if(i<size)
	{
		int z=i*attrib;
		sum=0;
		for(int j=0;j<attrib-1;j++){
                sum = sum + (ga[z+j]-ga[attrib*n+j])*(ga[z+j]-ga[attrib*n+j]);
            }	
		gResult[i].distance=sqrt(sum);
		gResult[i].val=ga[z+attrib-1];
	}
}

int gpuClassify(){
	Point *result = (Point *)malloc(n*sizeof(Point));
	
	int blockSize=128, blocks;
    cudaError_t err;
    Point *ga;
    
    err=cudaMalloc((void **)&ga,n*sizeof(Point));
    
             if (cudaSuccess!=err)
            {
	                     printf("\n Memory allocation failed on GPU for ga");
	                     printf("\n error is- %s", cudaGetErrorString(err));
	                     exit(EXIT_FAILURE);
			}
	    
	         if (cudaSuccess!=cudaMemcpy(ga,arr,n*sizeof(Point),cudaMemcpyHostToDevice))
	             {
		                     printf("\n Error in copying ha to ga");
		                     exit(EXIT_FAILURE);
		     }
	
	err=cudaMalloc((void **)&gResult,n*sizeof(Point));
    
            if (cudaSuccess!=err)
            {
	                     printf("\n Memory allocation failed on GPU for result");
	                     printf("\n error is- %s", cudaGetErrorString(err));
	                     exit(EXIT_FAILURE);
			}

	blocks=(int)(n/blockSize);
	if ((n%blockSize)>0)
		blocks++;
	printf("\n The number of blocks needed=%d", blocks);
	
	DistanceKernel<<<blocks,blockSize>>>(ga,n,gResult,attrib);     

	if (cudaSuccess!=cudaMemcpy(arr,ga,n*sizeof(int),cudaMemcpyDeviceToHost))
		{
					printf("\n Error in copying ga to hb");
							exit(EXIT_FAILURE);
		} 
		
	if (cudaSuccess!=cudaMemcpy(result,gResult,n*sizeof(int),cudaMemcpyDeviceToHost))
		{
					printf("\n Error in copying ga to hb");
							exit(EXIT_FAILURE);
		} 
  	
    // Sort the Points by distance from p
    std::sort(result, result+n, comparison);

    // Now consider the first k elements and only
    // two groups
    int freq[noOfClasses];
    for(int i=0;i<noOfClasses;i++)
        freq[i]=0;
    for (int i = 0; i < k; i++)
    {
        freq[result[i].val-1]++;
    }
    return max(freq);

}
void classifyAPoint(int arr[], int n, int k, int attrib)
{
    float timespentCPU, timespentGPU;
    clock_t start1, stop1;
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); //Creates an event object 
    cudaEventCreate(&stop);
		       
    //cpu time calculation
	start1=clock();
    int result =  cpuClassify(arr,n,k,attrib);      
    stop1=clock();
    timespentCPU = ((float)(stop1 - start1))/CLOCKS_PER_SEC;
    printf("\n result of cpuClassify is %d",result);
	printf("\n timespent on CPU=%f",timespentCPU);
    getchar();
    
    //cuda time calculation
	cudaEventRecord(start, 0); //Timestamp, zero –default stream
	
	result=gpuClassify(arr,n,k,attrib);
	
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0); //Timestamp
	
	cudaEventSynchronize(stop); 
	
	cudaEventElapsedTime(&timespentGPU, start, stop); 
	
	printf("\n result of cpuClassify is %d",timespentGPU);
	printf("\n timespent on GPU=%f",timespentGPU);

	
}

// Driver code
int main()
{
    std::ifstream inFile;
    inFile.open("/home/student1/test.txt");
    if(!inFile)
      perror ( "Stream Failed to open because: " );
    int attrib=4;
    long n = 245057; // Number of data points
    int *arr = (int*)malloc((attrib*n+attrib)*sizeof(int));
    long i=0;

    std::string line;
    while (std::getline(inFile, line))
    {
        std::istringstream iss(line);
        //if (!(iss >> arr[i] >> arr[i] >> arr[i] >> arr[i] >> arr[i])) { break; }
        for(int j=0;j<attrib;j++){
                if (!(iss >> arr[i++])) { break; }
        }

        //printf("\n%f  %ld",arr[i].x,i);

    }

    /*Testing Point*/
    Point p;
    arr[i] = 7;
    arr[i+1] = 7;
    arr[i+2] = 0;

    // Parameter to decide groupr of the testing point
    int k = 1;
    classifyAPoint(arr, n, k,attrib);
    return 0;
}
