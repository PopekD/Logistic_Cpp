#include "logisticRegression.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

// template <typename T>
void LogisticRegression::fit(const std::vector<std::vector<float>> &x,
                             const std::vector<bool> &y) {
  if (saveLoss)
    lossVect.clear();

  if (algorithm == "lbfgs") {
    LBFGS(y, x);
    return;
  } else if (algorithm == "gradientdescent") {
    GradientDescent(y, x);
    return;
  }

  std::cout << "Err: Prob typo (lbfgs, gradientdescent)" << std::endl;
}

// template <typename T>
std::vector<Predictions>
LogisticRegression::predict(const std::vector<std::vector<float>> &x) {

  size_t n_obs = x.size();

  std::vector<Predictions> preds(n_obs);

  for (auto i{0uz}; i < n_obs; ++i) {
    auto z = std::inner_product(weights.begin(), weights.end() - 1,
                                x[i].begin(), 0.0f);

    z += weights.back();

    const float y_tilde = 1 / (1 + std::exp(-z));
    int prediction = (y_tilde > 0.5f) ? 1 : 0;
    preds[i] = {prediction, y_tilde};
  }

  return preds;
}

// template <typename T>
void LogisticRegression::LBFGS(const std::vector<bool> &y,
                               const std::vector<std::vector<float>> &x) {

  size_t n_features = x[0].size();
  size_t n_params = n_features + 1;

  weights.assign(n_params, 0.0f); // wights at k = weigths k - 1
  lbfgs_hist.clear();

  std::vector<float> gradient(n_params, 0.0f);
  std::vector<float> alpha(hist_size, 0.0f);

  std::vector<float> q(n_params, 0.0f);

  std::vector<float> w_new(n_params, 0.0f);
  std::vector<float> g_new(n_params, 0.0f);
  std::vector<float> w_diff(n_params, 0.0f);
  std::vector<float> g_diff(n_params, 0.0f);

  float current_loss = 0.0f;
  float prev_loss = 1e10f;

  for (auto i{0uz}; i < max_iter; ++i) {

    if (i == 0)
      ComputeLossAndGradient(y, x, &gradient, &current_loss, weights);

    if (saveLoss)
      lossVect.push_back(current_loss);

    if (std::abs(prev_loss - current_loss) < tolerance) {
      // std::cout << "Converged at iteration" << i << std::endl;
      return;
    }
    prev_loss = current_loss;
    // LBFGS
    q = gradient;
    if (!lbfgs_hist.empty()) {
      for (int k = lbfgs_hist.size() - 1; k >= 0; --k) {

        const History &history = lbfgs_hist[k];

        float s_dot_q = std::inner_product(history.s.begin(), history.s.end(),
                                           q.begin(), 0.0f);
        alpha[k] = history.rho * s_dot_q;

        for (auto i{0uz}; i < n_params; ++i) {
          q[i] = q[i] - (alpha[k] * history.y[i]);
        }
      }

      const History &last = lbfgs_hist.back();

      float s_dot_y = std::inner_product(last.s.begin(), last.s.end(),
                                         last.y.begin(), 0.0f);
      float y_dot_y = std::inner_product(last.y.begin(), last.y.end(),
                                         last.y.begin(), 0.0f);

      float gamma = s_dot_y / (y_dot_y + 1e-10f);

      for (auto &grad : q) {
        grad *= gamma;
      }

      for (auto k{0uz}; k < lbfgs_hist.size(); ++k) {

        const History &history = lbfgs_hist[k];

        float y_dot_q = std::inner_product(history.y.begin(), history.y.end(),
                                           q.begin(), 0.0f);
        float beta = history.rho * y_dot_q;

        for (auto i{0uz}; i < n_params; ++i) {
          q[i] = q[i] + (history.s[i] * (alpha[k] - beta));
        }
      }
    }

    std::transform(q.begin(), q.end(), q.begin(),
                   [](auto &val) { return val * -1.f; });

    float loss_new;
    float a = LineSearch(weights, q, gradient, current_loss, y, x, w_new, g_new,
                         loss_new);

    for (auto i{0uz}; i < n_params; ++i) {
      w_diff[i] = a * q[i];
      g_diff[i] = g_new[i] - gradient[i];
    }

    float k_y_dot_s =
        std::inner_product(w_diff.begin(), w_diff.end(), g_diff.begin(), 0.0f);

    if (k_y_dot_s > 1e-10f) {
      float k_rho = 1.f / k_y_dot_s;
      lbfgs_hist.push_back({w_diff, g_diff, k_rho});

      if (lbfgs_hist.size() > hist_size) {
        lbfgs_hist.pop_front();
      }
    }

    weights = std::move(w_new);
    gradient = std::move(g_new);
    current_loss = std::move(loss_new);
  }

  std::cout << "The Algorithm did not converge" << std::endl;
}

float LogisticRegression::LineSearch(
    const std::vector<float> &w, const std::vector<float> &q,
    const std::vector<float> &g, const float &loss_old,
    const std::vector<bool> &y, const std::vector<std::vector<float>> &x,
    std::vector<float> &w_new, std::vector<float> &g_new, float &loss_new) {

  size_t n_features = w.size();

  float left = 0.f, right = 1.f;
  float a = 1e-5, c1 = 1e-4, c2 = 0.9, e = 1e-6;
  loss_new = 0.f;

  w_new.assign(n_features, 0.0f);
  g_new.assign(n_features, 0.0f);
  float g_dot_z = std::inner_product(g.begin(), g.end(), q.begin(), 0.f);

  while (left < right) {
    float m = (left + right) / 2;

    for (auto i{0uz}; i < w.size(); ++i) {
      w_new[i] = w[i] + (m * q[i]);
    }

    ComputeLossAndGradient(y, x, &g_new, &loss_new, w_new);
    if (loss_new - loss_old < c1 * m * g_dot_z) {

      float new_g_dot_z =
          std::inner_product(g_new.begin(), g_new.end(), q.begin(), 0.f);

      if (std::abs(new_g_dot_z) < c2 * std::abs(g_dot_z)) {
        return m;
      }
      a = m;
      left = m + e;
    } else {
      right = m - e;
    }
  }

  return a;
}

void LogisticRegression::ComputeLossAndGradient(
    const std::vector<bool> &y, const std::vector<std::vector<float>> &x,
    std::vector<float> *out_gradient, float *out_loss,
    std::vector<float> &weights) {

  size_t n_samples = y.size();
  size_t n_features = x[0].size();

  if (out_loss)
    *out_loss = 0.0f;
  if (out_gradient)
    std::fill(out_gradient->begin(), out_gradient->end(), 0.0f);

  float l2_w_sum =
      std::accumulate(weights.begin(), weights.end() - 1, 0.0f,
                      [](float sum, float w) { return sum + w * w; });

  for (auto j{0uz}; j < n_samples; ++j) {

    auto z = std::inner_product(weights.begin(), weights.end() - 1,
                                x[j].begin(), 0.0f);

    z += weights[n_features];
    float y_tilde = 1 / (1 + std::exp(-z));

    if (out_gradient) {
      float error = y_tilde - (y[j] ? 1.0f : 0.0f);
      for (auto k{0uz}; k < n_features; k++) {
        (*out_gradient)[k] += error * x[j][k];
      }

      (*out_gradient)[n_features] += error;
    }

    if (out_loss) {
      if (y[j])
        *out_loss -= std::log(y_tilde + 1e-10f);
      else
        *out_loss -= std::log(1.0f - y_tilde + 1e-10f);
    }
  }

  if (out_gradient) {
    for (auto k{0uz}; k < n_features; ++k) {
      (*out_gradient)[k] += (l2 / y.size()) * weights[k];
    }
  }

  if (out_loss) {
    *out_loss /= y.size();
    *out_loss += (l2 / (2 * y.size())) * l2_w_sum;
  }
}

// template <typename T>
void LogisticRegression::GradientDescent(
    const std::vector<bool> &y, const std::vector<std::vector<float>> &x) {

  size_t n_features = x[0].size();
  size_t n_params = n_features + 1;

  float prev_loss = 1e10f;
  float current_loss = 0.f;

  weights.assign(n_params, 0.0f);
  std::vector<float> gradient(n_params);

  for (auto i{0uz}; i < max_iter; ++i) {

    ComputeLossAndGradient(y, x, &gradient, &current_loss, weights);
    if (saveLoss)
      lossVect.push_back(current_loss);

    if (std::abs(prev_loss - current_loss) < tolerance) {
      // std::cout << "Converged at iteration" << i << std::endl;
      return;
    }

    prev_loss = current_loss;

    for (auto w{0uz}; w < n_params; ++w) {
      weights[w] -= LearningRate * gradient[w];
    }
  }

  std::cout << "The algorithm did not converge" << std::endl;
}
