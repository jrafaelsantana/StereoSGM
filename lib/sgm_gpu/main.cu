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

#define TB 128

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

__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, float tau1)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dir = id;
		int x = dir % dim3;
		dir /= dim3;
		int y = dir % dim2;
		dir /= dim2;

		int dx = 0;
		int dy = 0;
		if (dir == 0) {
			dx = -1;
		} else if (dir == 1) {
			dx = 1;
		} else if (dir == 2) {
			dy = -1;
		} else if (dir == 3) {
			dy = 1;
		} else {
			assert(0);
		}

		int xx, yy, ind1, ind2, dist;
		ind1 = y * dim3 + x;
		for (xx = x + dx, yy = y + dy;;xx += dx, yy += dy) {
			if (xx < 0 || xx >= dim3 || yy < 0 || yy >= dim2) break;

			dist = max(abs(xx - x), abs(yy - y));
			if (dist == 1) continue;

			ind2 = yy * dim3 + xx;

			/* rule 1 */
			if (COLOR_DIFF(x0, ind1, ind2) >= tau1) break;

			/* rule 2 */
			if (dist >= L1) break;
		}
		out[id] = dir <= 1 ? xx : yy;
	}
}

__global__ void cbca(float *x0c, float *x1c, float *vol, float *out, int size, int dim2, int dim3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = id;
		int x = d % dim3;
		d /= dim3;
		int y = d % dim2;
		d /= dim2;

		if (x + d * direction < 0 || x + d * direction >= dim3) {
			out[id] = vol[id];
		} else {
			float sum = 0;
			int cnt = 0;

			int yy_s = max(x0c[(2 * dim2 + y) * dim3 + x], x1c[(2 * dim2 + y) * dim3 + x + d * direction]);
			int yy_t = min(x0c[(3 * dim2 + y) * dim3 + x], x1c[(3 * dim2 + y) * dim3 + x + d * direction]);
			for (int yy = yy_s + 1; yy < yy_t; yy++) {
				int xx_s = max(x0c[(0 * dim2 + yy) * dim3 + x], x1c[(0 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				int xx_t = min(x0c[(1 * dim2 + yy) * dim3 + x], x1c[(1 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				for (int xx = xx_s + 1; xx < xx_t; xx++) {
					float val = vol[(d * dim2 + yy) * dim3 + xx];
					assert(!isnan(val));
					sum += val;
					cnt++;
				}
			}

			assert(cnt > 0);
			out[id] = sum / cnt;
			assert(!isnan(out[id]));
		}
	}
}

np::ndarray cb_ca(np::ndarray& img1_nd, np::ndarray& img2_nd, np::ndarray& vol_in_nd, int L1, float tau1, int direction)
{
    Mat left = cv::Mat(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0]), CV_32FC1, img1_nd.get_data()); 
    Mat right = cv::Mat(cv::Size(img2_nd.get_shape()[1], img2_nd.get_shape()[0]), CV_32FC1, img2_nd.get_data()); 


    int size[3] = {vol_in_nd.get_shape()[0] , vol_in_nd.get_shape()[1] , vol_in_nd.get_shape()[2]  };

    Mat vol_in = cv::Mat(3, size, CV_32FC1, vol_in_nd.get_data()); 

    float *vol_in_cu; 
	float *vol_out_cu;
	float *tmp;

	static Mat vol_out = cv::Mat::zeros(3, size, CV_32FC1); 

    

    //tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))


    float *x0 = (float *)left.data;
	float *x1 = (float *)right.data; 

	float *x0_cu; 
	float *x1_cu; 
	float *x0c;
	float *x1c;




	int vol_numberElement = vol_in.size[0]*vol_in.size[1]*vol_in.size[2];
	int x0C_numberElement = left.size[0]*left.size[1]*4;

	cudaMalloc(&x0_cu, (left.size[0]*left.size[1])*sizeof(float));
	cudaMalloc(&x1_cu, (right.size[0]*right.size[1])*sizeof(float));
	cudaMalloc(&x0c, (left.size[0]*left.size[1]*4)*sizeof(float));
  	cudaMalloc(&x1c, (right.size[0]*right.size[1]*4)*sizeof(float));
  	cudaMalloc(&vol_in_cu, (vol_in.size[0]*vol_in.size[1]*vol_in.size[2])*sizeof(float));       
    cudaMalloc(&vol_out_cu, (vol_in.size[0]*vol_in.size[1]*vol_in.size[2])*sizeof(float));	



  	cudaMemcpy(vol_in_cu, (float *)vol_in.data, (vol_in.size[0]*vol_in.size[1]*vol_in.size[2])*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(x0_cu, x0, (left.size[0]*left.size[1])*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(x1_cu, x1, (right.size[0]*right.size[1])*sizeof(float), cudaMemcpyHostToDevice);

  	/*for(int i = 2; i < 5; i++){
  		for(int j = 10; j < 20; j++){
  			for(int k = 2; k < 10; k++){
  				printf("%f \n", vol_in.at<float>(i,j,k));
  			}
  		}
  	}*/

  	//printf("depois\n");

  	/*cudaMemset(x0c, 0, (left.size[0]*left.size[1]*4)*sizeof(float));
  	cudaMemset(x1c, 0, (left.size[0]*left.size[1]*4)*sizeof(float));
  	cudaMemset(vol_out_cu, 0, (left.size[0]*left.size[1]*left.size[2])*sizeof(float));*/

  	cross<<< (x0C_numberElement - 1) / TB + 1, TB>>>(
			x0_cu,
			x0c,			
			x0C_numberElement,
			left.rows,
			left.cols, L1, tau1);

  	cross<<< (x0C_numberElement - 1) / TB + 1, TB>>>(
			x1_cu,
			x1c,			
			x0C_numberElement,
			left.rows,
			left.cols, L1, tau1);


  	//for(int i = 0; i < 2; i++){
		cbca<<<(vol_numberElement - 1) / TB + 1, TB>>>(
			x0c,
			x1c,
			vol_in_cu,
			vol_out_cu,
			vol_numberElement,
			left.rows,
			left.cols,
			direction);

			//tmp = vol_in_cu;
			//vol_in_cu = vol_out_cu;
			//vol_out_cu = tmp;

			//cudaMemset(vol_out_cu, 0, (left.size[0]*left.size[1]*left.size[2])*sizeof(float));

	//}

	cudaMemcpy((float *)vol_out.data, vol_out_cu, (vol_in.size[0]*vol_in.size[1]*vol_in.size[2])*sizeof(float), cudaMemcpyDeviceToHost);

	/*cout << vol_out.size[0] << endl;
	cout << vol_out.size[1] << endl;
	cout << vol_out.size[2] << endl;*/

	cudaFree(x0_cu);
  	cudaFree(x1_cu);
  	cudaFree(vol_in_cu);
  	cudaFree(vol_out_cu);
  	cudaFree(x0c);
  	cudaFree(x1c);
  	//cudaFree(tmp_pt_cu);

  	/*for(int i = 2; i < 5; i++){
  		for(int j = 10; j < 20; j++){
  			for(int k = 2; k < 10; k++){
  				printf("%f \n", vol_out.at<float>(i,j,k));
  				//vol_out.at<float>(i,j,k) = 100;
  			}
  		}
  	}*/



	//py::tuple shape = py::make_tuple(vol_out.rows, vol_out.cols, vol_out.channels());
	py::tuple shape = py::make_tuple(vol_out.size[0], vol_out.size[1], vol_out.size[2]);
    py::tuple stride = py::make_tuple(vol_out.size[2] * vol_out.size[1] * sizeof(float), vol_out.size[2] * sizeof(float), sizeof(float));
    np::dtype dt = np::dtype::get_builtin<float>();
    np::ndarray ndImg = np::from_data((float *)vol_out.data, dt, shape, stride, py::object());

    

    return ndImg;
}





int sgm2(cv::Mat left, cv::Mat right, cv::Mat cost, cv::Mat &output, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction)
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
	//float pi1 = 1, pi2 = 2, tau_so = 0.08, alpha1 = 1.5, sgm_q1 = 2, sgm_q2 = 1	;
	//int direction = 1;
	

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



np::ndarray dispCalc(np::ndarray& img1_nd, np::ndarray& img2_nd, np::ndarray& costs_nd, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction)
{
    Mat imgL = cv::Mat(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0]), CV_32FC1, img1_nd.get_data()); 
    Mat imgR = cv::Mat(cv::Size(img2_nd.get_shape()[1], img2_nd.get_shape()[0]), CV_32FC1, img2_nd.get_data()); 	

	int size[3] = {costs_nd.get_shape()[0] , costs_nd.get_shape()[1] , costs_nd.get_shape()[2]  };

    Mat costs = cv::Mat(3, size, CV_32FC1, costs_nd.get_data()); 

    Mat output = cv::Mat::zeros(3, size, CV_32FC1); 
    
    Mat mapa = cv::Mat::zeros(cv::Size(img1_nd.get_shape()[1], img1_nd.get_shape()[0]), CV_8UC1);     



    //cross
    
    //cbca

    sgm2(imgL, imgR, costs, output, pi1, pi2, tau_so,  alpha1, sgm_q1, sgm_q2, direction);


    for(int row = 0; row < costs_nd.get_shape()[0]; row++){
    	for(int col = 0; col < costs_nd.get_shape()[1]; col++){
    		int disp_f = 0;
    		float value_f = 100000;
    		for(int disp = 0; disp < costs_nd.get_shape()[2]; disp++){
    			float tmp = output.at<float>(row,col,disp)/4.0;
    			if(tmp < value_f){
    				value_f = tmp;
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
  py::def("cbca", cb_ca, "cbca"); 
  //py::def("init_disp", initDisp, "init_disp"); 
  //py::def("road_profile", road_profile, "road profile");
  //py::def("process_mat", &process_mat);
  //py::def("ConvertNDArrayToMat", ConvertNDArrayToMat, "ConvertNDArrayToMat");
}
