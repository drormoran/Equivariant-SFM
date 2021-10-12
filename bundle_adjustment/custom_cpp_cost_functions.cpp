#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>


namespace py = pybind11;



// Parses a numpy array and extracts the pointer to the first element.
// Requires that the numpy array be either an array or a row/column vector
double* _ParseNumpyData(py::array_t<double>& np_buf) {
    py::buffer_info info = np_buf.request();
    // This is essentially just all error checking. As it will always be the info
    // ptr
    if (info.ndim > 2) {
        std::string error_msg("Number of dimensions must be <=2. This function"
            "only allows either an array or row/column vector(2D matrix) "
            + std::to_string(info.ndim));
        throw std::runtime_error(
            error_msg);
    }
    if (info.ndim == 2) {
        // Row or Column Vector. Represents 1 parameter
        if (info.shape[0] == 1 || info.shape[1] == 1) {
        }
        else {
            std::string error_msg
            ("Matrix is not a row or column vector and instead has size "
                + std::to_string(info.shape[0]) + "x"
                + std::to_string(info.shape[1]));
            throw std::runtime_error(
                error_msg);
        }
    }
    return (double*)info.ptr;
}

struct ExampleFunctor {
  template<typename T>
  bool operator()(const T *const x, T *residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<ExampleFunctor,
                                           1,
                                           1>(new ExampleFunctor);
  }
};

struct projReprojectionError {
    projReprojectionError(double observed_x, double observed_y, double* Porig, double* Xorig, double blockNum)
        : _observed_x(observed_x), _observed_y(observed_y), _Porig(Porig), _Xorig(Xorig), _blockNum(blockNum) {}

    template <typename T>
    bool operator()(const T* const P, const T* const X,
        T* residuals) const {

        T Pnow[12];
        T Xnow[4];
        T projection[3];
        for (int i = 0; i < 12; i++) {
            Pnow[i] = P[i] + _Porig[i];
        }
        for (int i = 0; i < 3; i++) {
            Xnow[i] = X[i] + _Xorig[i];
        }
        Xnow[3] = T(1.0);
        projection[0] = Pnow[0] * Xnow[0] + Pnow[3] * Xnow[1] + Pnow[6] * Xnow[2] + Pnow[9] * Xnow[3];
        projection[1] = Pnow[1] * Xnow[0] + Pnow[4] * Xnow[1] + Pnow[7] * Xnow[2] + Pnow[10] * Xnow[3];
        projection[2] = Pnow[2] * Xnow[0] + Pnow[5] * Xnow[1] + Pnow[8] * Xnow[2] + Pnow[11] * Xnow[3];

        projection[0] = projection[0] / projection[2];
        projection[1] = projection[1] / projection[2];

        residuals[0] = (projection[0] - _observed_x) / _blockNum;
        residuals[1] = (projection[1] - _observed_y) / _blockNum;


        return true;
    }

    static ceres::CostFunction* CreateMyRep(const double observed_x,
        const double observed_y, double* Porig, double* Xorig, double  blocksNum) {
        return (new ceres::AutoDiffCostFunction<projReprojectionError, 2, 12, 3>(
            new projReprojectionError(observed_x, observed_y, Porig, Xorig, blocksNum)));
    }

    double _observed_x;
    double _observed_y;

    double _blockNum;
    double* _Porig;
    double* _Xorig;


};


struct eucReprojectionError {
	eucReprojectionError(double observed_x, double observed_y, double * Porig, double * Xorig,double blockNum)
	: _observed_x(observed_x), _observed_y(observed_y), _Porig(Porig), _Xorig(Xorig), _blockNum(blockNum){}

	template <typename T>
	bool operator()(const T* const P, const T* const X,
		T* residuals) const {
		// camera[0,1,2] are the angle-axis rotation.
		
		T Rnow[3];
        T tnow[3];
        
        
		T Xnow[4];
        T Xrot[3];
        
		T projection[3];
		for (int i = 0; i < 3; i++){
			tnow[i] = P[i+3] + _Porig[i+3]; // P[3:6] camera location
            Rnow[i] = P[i] + _Porig[i]; // P[0:3] camera rotation
		}
		for (int i = 0; i < 3; i++){
			Xnow[i] = X[i] + _Xorig[i];
		}
        ceres::AngleAxisRotatePoint(Rnow, Xnow, Xrot);
        Xrot[0]+=tnow[0];
        Xrot[1]+=tnow[1];
        Xrot[2]+=tnow[2];
        
        projection[0]=(Xrot[0]*_Porig[6]+Xrot[1]*_Porig[7]+Xrot[2]*_Porig[8])/Xrot[2]; // P[6:5] K
        projection[1]=(Xrot[1]*_Porig[9]+Xrot[2]*_Porig[10])/Xrot[2];
        
	residuals[0] = (projection[0] - _observed_x) ;
	residuals[1] = (projection[1] - _observed_y) ;
	//residuals[0] = ceres::sqrt((projection[0] - _observed_x)*(projection[0] - _observed_x) + (projection[1] - _observed_y)*(projection[1] - _observed_y));
	//residuals[1] = T(0.0);

	return true;
	}

	static ceres::CostFunction* CreateMyRepEuc(const double observed_x,
		const double observed_y, double * Porig, double * Xorig, double  blocksNum) {
		return (new ceres::AutoDiffCostFunction<eucReprojectionError, 2, 6,3>(
			new eucReprojectionError(observed_x, observed_y, Porig, Xorig, blocksNum)));
	}
	double _observed_x;
	double _observed_y;
	double _blockNum;
	double * _Porig;
	double * _Xorig;
};



//****** Complete function here
/* The gateway function */
void pythonFunctionOursBA(double* Xs, double* xs, double* Ps, double* camPointmap, double* Xsu, double* Psu, int n_cam, int n_pts, int n_observe)
{

    ceres::Problem problem;
    for (int i = 0; i < n_observe; i++) {

        int camIndex = int(camPointmap[i]);
        int point3DIndex = int(camPointmap[i + n_observe]);

        ceres::CostFunction* cost_function =
            projReprojectionError::CreateMyRep(xs[2*i], xs[2*i + 1], Ps + 12 * (camIndex), Xs + 3 * (point3DIndex), 1);

        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
        problem.AddResidualBlock(cost_function,
            loss_function,
            Psu + 12 * (camIndex), Xsu + 3 * (point3DIndex));
    }

    ceres::Solver::Options options;
    options.function_tolerance = 0.0001;
    options.max_num_iterations = 100;
    options.num_threads = 24;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport().c_str() << "\n";
}
void eucPythonFunctionOursBA(double* Xs, double* xs, double* Ps, double* camPointmap , double* Xsu, double* Psu, int nrows, int ncols, int n_observe)
{

    ceres::Problem problem;
	for (int i = 0; i < n_observe; i++){
        
		int camIndex = int(camPointmap[i]);
		int point3DIndex = int(camPointmap[i+n_observe]);

		ceres::CostFunction* cost_function =
			eucReprojectionError::CreateMyRepEuc(xs[2*i], xs[2*i + 1], Ps+(12*camIndex), Xs+(3*point3DIndex),1);
		ceres::LossFunction * loss_function = new ceres::HuberLoss(0.1);

		problem.AddResidualBlock(cost_function,
			loss_function,
			Psu+(12*camIndex), Xsu+(3*point3DIndex));
    }

    ceres::Solver::Options options;
	options.function_tolerance = 0.0001;
	//options.max_num_iterations = 100;
	options.max_num_iterations = 100;
	//options.num_threads = 8;
	options.num_threads = 24;
	
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
 	std::cout << summary.FullReport() << "\n";
}


//***** complete function ends here

void add_custom_cost_functions(py::module &m) {

  // // Use pybind11 code to wrap your own cost function which is defined in C++s


  // // Here is an example
  // m.def("CreateCustomExampleCostFunction", &ExampleFunctor::Create);

	m.def("projReprojectionError", [](
			double observed_x, double observed_y, py::array_t<double>& _Porig, py::array_t<double>& _Xorig
                ) {
                    double* Porig = _ParseNumpyData(_Porig);
                    double* Xorig = _ParseNumpyData(_Xorig);					
                    py::gil_scoped_release release;

                    return projReprojectionError::CreateMyRep(observed_x, observed_y, Porig, Xorig, 1);
                } , py::return_value_policy::reference);
	m.def("eucReprojectionError", [](
			double observed_x, double observed_y, py::array_t<double>& _Porig, py::array_t<double>& _Xorig
                ) {
                    double* Porig = _ParseNumpyData(_Porig);
                    double* Xorig = _ParseNumpyData(_Xorig);					
                    py::gil_scoped_release release;

                    return eucReprojectionError::CreateMyRepEuc(observed_x, observed_y, Porig, Xorig, 1);
                } , py::return_value_policy::reference);
				
				
	            m.def("pythonFunctionOursBA", [](
                py::array_t<double>& _Xs,
                py::array_t<double>& _xs,
                py::array_t<double>& _Ps,
                py::array_t<double>& _camPointmap,
                py::array_t<double>& _Xsu,
                py::array_t<double>& _Psu,
                int n_cam, int n_pts, int n_observe
                ) {
                    double* Xs = _ParseNumpyData(_Xs);
                    double* xs = _ParseNumpyData(_xs);
                    double* Ps = _ParseNumpyData(_Ps);
                    double* camPointmap = _ParseNumpyData(_camPointmap);
                    double* Xsu = _ParseNumpyData(_Xsu);
                    double* Psu = _ParseNumpyData(_Psu);
                    py::gil_scoped_release release;

                    pythonFunctionOursBA(Xs, xs, Ps, camPointmap, Xsu, Psu, n_cam, n_pts, n_observe);

                });

            m.def("eucPythonFunctionOursBA", [](
                py::array_t<double>& _Xs,
                py::array_t<double>& _xs,
                py::array_t<double>& _Ps,
                py::array_t<double>& _camPointmap,
                py::array_t<double>& _Xsu,
                py::array_t<double>& _Psu,
                int n_cam, int n_pts, int n_observe
                ) {
                    double* Xs = _ParseNumpyData(_Xs);
                    double* xs = _ParseNumpyData(_xs);
                    double* Ps = _ParseNumpyData(_Ps);
                    double* camPointmap = _ParseNumpyData(_camPointmap);
                    double* Xsu = _ParseNumpyData(_Xsu);
                    double* Psu = _ParseNumpyData(_Psu);
                    py::gil_scoped_release release;

                    eucPythonFunctionOursBA(Xs, xs, Ps, camPointmap, Xsu, Psu, n_cam, n_pts, n_observe);

                });
}
