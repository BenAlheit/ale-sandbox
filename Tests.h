//
// Created by alhei on 2022/08/22.
//

#ifndef ALE_TESTS_H
#define ALE_TESTS_H

#include "TimeIntegration.h"
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace std;

class TestRungeKutta {
public:
    explicit TestRungeKutta(const double & lam=1, const double & y0=1) : lam(lam), y0(y0){};

    void do_test(const unsigned int n_refines = 8);

private:

    static Vector<double> dy(const double & t, const Vector<double> & y, const double & lam){
        Vector<double> out(y.size());
        out[0] = lam*y[0];
        return out;
    }

    vector<double> approximate_solution(const vector<double> &t){
        unsigned int n_vals = t.size();
        vector<double> out(n_vals);
        out[0] = y0;
        Vector<double> y_n(1), y_n1(1);
        y_n[0] = y0;
        auto dy_in = bind(dy, placeholders::_1, placeholders::_2, this->lam);
        for (unsigned int i = 1; i < n_vals; ++i) {
            integrator->set_t_start(t.at(i-1));
            integrator->set_t_end(t.at(i));
            integrator->set_n_steps(1);

            y_n1 = integrator->integrate(y_n, dy_in);
            out.at(i) = y_n1[0];
            y_n = y_n1;
        }
        return out;
    }

    vector<double> error(const vector<double> &approx_solution,
                         const vector<double> &t) {
        unsigned int n_vals = approx_solution.size();
        vector<double> out(n_vals);
        vector<double> an_solution = analytical_solution_series(t);
        for (unsigned int i = 0; i < n_vals; ++i) {
            out.at(i) = an_solution.at(i) - approx_solution.at(i);
        }
        return out;
    }

    vector<double> analytical_solution_series(const vector<double> &t) {
        vector<double> out;
        for (const auto &val: t) out.push_back(analytical_solution(val));
        return out;
    }

    double analytical_solution(const double &t) const {
        return y0 * exp(lam * t);
    }

    void write_to_file(string name, const vector<double>& out){
        ofstream sol_file(name);
        ostream_iterator<double> sol_iterator(sol_file, "\n");
        copy(out.begin(), out.end(), sol_iterator);
    }

    double lam, y0;
    double t_end = 2;
    TimeIntegration *integrator = new RungeKutta();

};

void TestRungeKutta::do_test(const unsigned int n_refines) {
    vector<vector<double>> errors, times, solutions;
    double n_steps_base = 2;
    double dt;
    unsigned int steps;
    vector<unsigned int> range(n_refines);
    vector<double> approx;
    iota(range.begin(), range.end(), 0);

    for(const auto & i_ref: range){
        steps = pow(n_steps_base, i_ref);
        dt = t_end/steps;
        vector<double> ref_times(steps);
        ref_times.at(0) = 0;
        for (unsigned int i = 1; i < steps; ++i) {
            ref_times.at(i) = ref_times.at(i-1) + dt;
        }
        times.push_back(ref_times);
        approx = approximate_solution(ref_times);
        errors.push_back(error(approx, ref_times));
        solutions.push_back(analytical_solution_series(ref_times));

        write_to_file("times-"+to_string(i_ref)+".dat", ref_times);
        write_to_file("errors-"+to_string(i_ref)+".dat", errors.at(i_ref));
        write_to_file("solution-"+to_string(i_ref)+".dat", solutions.at(i_ref));
    }

}

#endif //ALE_TESTS_H
