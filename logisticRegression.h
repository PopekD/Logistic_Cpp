#pragma once
#include <deque>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct History {
  std::vector<float> s;
  std::vector<float> y;
  float rho;
};

struct Predictions {
  int pred;
  float prob;
};

class LogisticRegression {

public:
  LogisticRegression(size_t max_iter = 100, std::string algorithm = "lbfgs",
                     float l2 = 1.0f, float LearningRate = 0.01f,
                     size_t hist_size = 10, float tolerance = 1e-4)
      : max_iter(max_iter), algorithm(std::move(algorithm)), l2(l2),
        hist_size(hist_size), LearningRate(LearningRate),
        tolerance(tolerance) {};

  // template <typename T>
  void fit(const std::vector<std::vector<float>> &x,
           const std::vector<bool> &y);

  // template <typename T>
  std::vector<Predictions> predict(const std::vector<std::vector<float>> &x);

  std::vector<float> get_weigths() { return weights; };

private:
  const size_t max_iter;
  std::vector<float> weights;

  const std::string algorithm;

  const float l2;
  const size_t hist_size;

  const float LearningRate;
  std::deque<History> lbfgs_hist;

  const float tolerance;

  // template <typename T>
  void LBFGS(const std::vector<bool> &y,
             const std::vector<std::vector<float>> &x);

  // template <typename T>
  void GradientDescent(const std::vector<bool> &y,
                       const std::vector<std::vector<float>> &x);

  float LineSearch(const std::vector<float> &w, const std::vector<float> &q,
                   const std::vector<float> &g, const float &loss_old,
                   const std::vector<bool> &y,
                   const std::vector<std::vector<float>> &x,
                   std::vector<float> &w_new, std::vector<float> &g_new,
                   float &loss_new);

  void ComputeLossAndGradient(const std::vector<bool> &y,
                              const std::vector<std::vector<float>> &x,
                              std::vector<float> *out_gradient, float *out_loss,
                              std::vector<float> &weights);
};

PYBIND11_MODULE(LR_model, m) {
  py::class_<Predictions>(m, "Predictions")
      .def_readwrite("prediction", &Predictions::pred)
      .def_readwrite("probability", &Predictions::prob);

  py::class_<History>(m, "History")
      .def_readwrite("s", &History::s)
      .def_readwrite("y", &History::y)
      .def_readwrite("rho", &History::rho);

  py::class_<LogisticRegression>(m, "LogisticRegression")
      .def(py::init<size_t, std::string, float, float, size_t, float>())
      .def("fit", &LogisticRegression::fit)
      .def("predict", &LogisticRegression::predict)
      .def("get_weigths", &LogisticRegression::get_weigths);
}
