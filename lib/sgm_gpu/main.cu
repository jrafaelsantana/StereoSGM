#define BOOST_PYTHON_STATIC_LIB
#define BOOST_LIB_NAME "boost_numpy"

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace py = boost::python;
namespace np = boost::python::numpy;

cv::Mat disparity_im;
cv::Mat disparity_im_lr;
cv::Mat disparity_im_result;

#define COLOR_DIFF(x, i, j) (abs(x[i] - x[j]))

#define INDEX(dim0, dim1, dim2, dim3) \
	assert((dim1) >= 0 && (dim1) < size1 && (dim2) >= 0 && (dim2) < size2 && (dim3) >= 0 && (dim3) < size3), \
	((((dim0) * size1 + (dim1)) * size2 + (dim2)) * size3 + dim3)


void leftRightCheck(Mat &inputImage, Mat &inputImage2, Mat &imageResult, int dispMax){
  int row, col;   
  //float disparity = 0;
  float temp;  
  
  for (row = 0; row < inputImage.rows; row++){
    for (col = 0; col < inputImage.cols; col++){

        if(col - inputImage.at<uchar>(row, col) >= 0) temp = (inputImage.at<uchar>(row, col) - (inputImage2.at<uchar>(row, col - inputImage.at<uchar>(row, col))));
        else temp = dispMax - 1;        

        if (temp > dispMax){       
           // imageResult.at<float>(row, col) = 0;
           //imageResult.at<uchar>(row, col) = 0; //codigo de invalido
           //imageResult.at<uchar>(row, col) = 255;  
           imageResult.at<uchar>(row, col) = 255;  
           // imageResult.at<ushort>(row, col) = inputImage.at<ushort>(row, col);
        }
        else{ 
           imageResult.at<uchar>(row, col) = inputImage.at<uchar>(row, col);  
        }
      }   
   }
}

template <int sgm_direction>
__global__ void sgm2(float *x0, float *x1, float *input, float *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - step;
		dx = 0;
		dy = -1;
	}

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		float val = input[INDEX(0, y, x, d)];
		output[INDEX(0, y, x, d)] += val;
		tmp[d * size2 + blockIdx.x] = val;
		return;
	}	

	__shared__ float output_s[400], output_min[400];

	output_s[d] = output_min[d] = tmp[d * size2 + blockIdx.x];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	float val = input[INDEX(0, y, x, d)] + cost - output_min[0];
	output[INDEX(0, y, x, d)] += val;
	tmp[d * size2 + blockIdx.x] = val;
}

int sgm2(cv::Mat left, cv::Mat right, cv::Mat cost, cv::Mat &output, cv::Mat &tmp)
{

	float *x0 = (float *)left.data;
	float *x1 = (float *)right.data; 
	float *input = (float *)cost.data;
	float *output_pt = (float *)output.data;
	//float *tmp_pt = (float *)tmp.data;

	float *x0_cu; 
	float *x1_cu; 
	float *input_cu; 
	float *output_cu; 
	float *tmp_pt_cu; 

	cudaMalloc(&x0_cu, (cost.size[0]*cost.size[1])*sizeof(float));
	cudaMalloc(&x1_cu, (cost.size[0]*cost.size[1])*sizeof(float));
	cudaMalloc(&input_cu, (cost.size[0]*cost.size[1]*cost.size[2])*sizeof(float));
  	cudaMalloc(&output_cu, (cost.size[0]*cost.size[1]*cost.size[2])*sizeof(float));
  	cudaMalloc(&tmp_pt_cu, (cost.size[0]*cost.size[1])*sizeof(float));	

  	cudaMemcpy(x0_cu, x0, (cost.size[0]*cost.size[1])*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(x1_cu, x1, (cost.size[0]*cost.size[1])*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(input_cu, input, (cost.size[0]*cost.size[1]*cost.size[2])*sizeof(float), cudaMemcpyHostToDevice); 	


	int size1 = left.rows * cost.size[2];
	int size2 = left.cols * cost.size[2];
	int disp_max = cost.size[2];
	float pi1 = 1, pi2 = 2, tau_so = 0.08, alpha1 = 1.5, sgm_q1 = 2, sgm_q2 = 1	;
	int direction = 1;
	

	for (int step = 0; step < left.cols; step++) {
		sgm2<0><<< (size1 - 1) / disp_max + 1, disp_max>>>(
			x0_cu,
			x1_cu,			
			input_cu,
			output_cu,
			tmp_pt_cu,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			left.rows,
			left.cols,
			disp_max,
			step);
	}
	

	for (int step = 0; step < left.cols; step++) {
		sgm2<1><<< (size1 - 1) / disp_max + 1, disp_max>>>(
			x0_cu,
			x1_cu,			
			input_cu,
			output_cu,
			tmp_pt_cu,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			left.rows,
			left.cols,
			disp_max,
			step);
	}

	for (int step = 0; step < left.rows; step++) {
		sgm2<2><<< (size2 - 1) / disp_max + 1, disp_max>>>(
			x0_cu,
			x1_cu,			
			input_cu,
			output_cu,
			tmp_pt_cu,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			left.rows,
			left.cols,
			disp_max,
			step);
	}

	for (int step = 0; step < left.rows; step++) {
		sgm2<3><<< (size2 - 1) / disp_max + 1, disp_max>>>(
			x0_cu,
			x1_cu,			
			input_cu,
			output_cu,
			tmp_pt_cu,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			left.rows,
			left.cols,
			disp_max,
			step);
	}

	cudaMemcpy(output_pt, output_cu, (cost.size[0]*cost.size[1]*cost.size[2])*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(x0_cu);
  	cudaFree(x1_cu);
  	cudaFree(input_cu);
  	cudaFree(output_cu);
  	cudaFree(tmp_pt_cu);

	return 0;
}

np::ndarray dispCalc(np::ndarray& img1_nd, np::ndarray& img2_nd, np::ndarray& costs_nd)
{
    //int a = 10;
    //Mat imgL = cv::Mat(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0], img1_nd.get_shape()[2]), CV_32FC1, img1_nd.get_data()); 

    Mat imgL = cv::Mat(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0]), CV_32FC1, img1_nd.get_data()); 
    Mat imgR = cv::Mat(cv::Size(img2_nd.get_shape()[1], img2_nd.get_shape()[0]), CV_32FC1, img2_nd.get_data()); 

	/*cout << img1_nd.get_shape()[0] << endl;
	cout << img1_nd.get_shape()[1] << endl;
	cout << img1_nd.get_shape()[2] << endl;*/

	int size[3] = {costs_nd.get_shape()[0] , costs_nd.get_shape()[1] , costs_nd.get_shape()[2]  };

    Mat costs = cv::Mat(3, size, CV_32FC1, costs_nd.get_data()); 

    //int size[3] = {costs_nd.get_shape()[0] , costs_nd.get_shape()[1] , costs_nd.get_shape()[2]  };

    Mat output = cv::Mat::zeros(3, size, CV_32FC1); 
    Mat tmp = cv::Mat::zeros(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0]), CV_32FC1); 




    Mat mapa = cv::Mat::zeros(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0]), CV_8UC1); 

    //imshow("disp1",mapa);
    //waitKey(0);

    //Mat imgR = cv::Mat(cv::Size(img2_nd.get_shape()[1], img2_nd.get_shape()[0]), CV_8UC1, img2_nd.get_data()); 
    //Mat imgL = imread("imgL.png",0);
    //Mat imgR = imread("imgR.png",0);

    //cout << "OLA" << endl;
    //getchar();

    //float elapsed_time_ms;
    /*if(init_disp == 0){
        init_disparity_method(20, 60);
        init_disp = 1;
    }*/
    //double t = (double)getTickCount();


    /*for(int row = 0; row < 20; row++){
    	for(int col = 0; col < 20; col++){
    		
    		for(int disp = 0; disp < 20; disp++){
    			cout << row << " " << col << " " << output.at<float>(row,col,disp) << endl;
    			
    		}

    	}
    }*/

    // << "antes" << endl;
    //getchar();

    sgm2(imgL, imgR, costs, output, tmp);


    /*for(int row = 0; row < 25; row++){
    	for(int col = 0; col < 25; col++){
    		
    		for(int disp = 0; disp < 40; disp++){
    			cout << row << " " << col << " " << disp <<  " " << output.at<float>(row,col,disp) << endl;
    			//getchar();
    			
    		}

    	}
    }*/

    //cout << "depois" << endl;
    //getchar();


    for(int row = 0; row < costs_nd.get_shape()[0]; row++){
    	for(int col = 0; col < costs_nd.get_shape()[1]; col++){
    		int disp_f = 0;
    		float value_f = 100000;
    		for(int disp = 0; disp < costs_nd.get_shape()[2]; disp++){
    			if(output.at<float>(row,col,disp) < value_f){
    				value_f = output.at<float>(row,col,disp);
    				disp_f = disp;
    			}
    		}
    		mapa.at<uchar>(row,col) = disp_f;
    	}
    }

    //cout << "FOI" << endl;


    //getchar();
    //disparity_im_lr = compute_disparity_method_RL(imgR, imgL, &elapsed_time_ms).clone();
    //disparity_im_result = disparity_im.clone();

    //leftRightCheck(disparity_im, disparity_im_lr, disparity_im_result, 0);

    

    //t = (double)getTickCount() - t;
    //printf("tsgm time = %gms\n", t*1000./cv::getTickFrequency());
    /*disparity_im.convertTo(disparity_im, CV_32FC1, 1.0);
    normalize(disparity_im,disparity_im,0,255,NORM_MINMAX);
    disparity_im.convertTo(disparity_im, CV_8UC1, 1.0);*/
    //disparity_im = left_disparity_map_norm.astype(np.uint8)
    
    //imshow("mapa",mapa);
    //waitKey(1);

    py::tuple shape = py::make_tuple(mapa.rows, mapa.cols, mapa.channels());
    py::tuple stride = py::make_tuple(mapa.channels() * mapa.cols * sizeof(uchar), mapa.channels() * sizeof(uchar), sizeof(uchar));
    np::dtype dt = np::dtype::get_builtin<uchar>();
    np::ndarray ndImg = np::from_data(mapa.data, dt, shape, stride, py::object());

    //finish_disparity_method();
    
    //printCudaVersion(teste);
    return ndImg;
}

BOOST_PYTHON_MODULE(sgm_gpu)
{ 
  Py_Initialize();
  np::initialize(); 
  //py::def("greet", greet, "Prepends a greeting to the passed name");
  //py::def("fat", fact, "funcao de fat");
  py::def("disp_calc", dispCalc, "disp_calc"); 
  //py::def("init_disp", initDisp, "init_disp"); 
  //py::def("road_profile", road_profile, "road profile");
  //py::def("process_mat", &process_mat);
  //py::def("ConvertNDArrayToMat", ConvertNDArrayToMat, "ConvertNDArrayToMat");
}
