//
// Created by alhei on 2022/08/14.
//

#ifndef ALE_HISTORYDEPENDENTMATERIALS_H
#define ALE_HISTORYDEPENDENTMATERIALS_H

#include "GeneralizedElastoPlasticMaterial.h"
#include "MaterialStates.h"
#include "Materials.h"
#include "cmath"
#include "TimeIntegration.h"

using namespace std;  // for _1, _2, _3...
//using namespace std::placeholders;  // for _1, _2, _3...

enum IntegrationMethod {
    GenericIntegrator, MidPoint, ExplicitPredictorImplicitCorrector
};

template<unsigned int dim>
class HardeningLaw {
//public:
//    virtual Vector<double>
//    h(const double &dlambda, const Tensor<2, dim> &tau, const Vector<double> &internal_variables) const = 0;
};

template<unsigned int dim>
class IsotropicHardeningLaw : public HardeningLaw<dim> {
public:
    virtual double sig_y(double ep) = 0;

    virtual double dsig_y(double ep) = 0;

    Vector<double>
    h(const double &dlambda, const Tensor<2, dim> &tau, const Vector<double> &internal_variables) {
        Vector<double> out(internal_variables.size());
        if (dlambda >= 0)
            out = 1.;
        else
            out = -1.;
        return out;
    }

};

template<unsigned int dim>
class PerfectPlasticity : public IsotropicHardeningLaw<dim> {
public:
    PerfectPlasticity() {};

    double sig_y(double ep) { return Y; };

    double dsig_y(double ep) { return 0; };

private:
    double Y = 60;
//    double Y = 1;
};

template<unsigned int dim>
class LinearHardening : public IsotropicHardeningLaw<dim> {
public:
    double sig_y(double ep) { return Y + H * ep; };

    double dsig_y(double ep) { return H; };

private:
    double Y = 60;
    double H = 50;
};


template<unsigned int dim>
class Voce : public IsotropicHardeningLaw<dim> {
public:
    double sig_y(double ep) { return sig_inf - (sig_inf - sig_y_init) * exp(-del * ep); };

    double dsig_y(double ep) { return del * (sig_inf - sig_y_init) * exp(-del * ep); };

private:
    double sig_y_init = 40;
    double sig_inf = 80;
    double del = 100;
//    double del = 50;
};

template<unsigned int dim>
class VoceWLinear : public IsotropicHardeningLaw<dim> {
public:
    double sig_y(double ep) { return sig_inf - (sig_inf - sig_y_init) * exp(-del * ep) + H * ep; };

    double dsig_y(double ep) { return del * (sig_inf - sig_y_init) * exp(-del * ep) + H; };

private:
    double sig_y_init = 40;
    double sig_inf = 80;
    double del = 100;
//    double H = 40;
    double H = 0;
//    double del = 50;
};

template<unsigned int dim>
class YieldSurface {
public:
    virtual double f(const Tensor<2, dim> &tau,
                     const Vector<double> &xi) = 0;

    virtual Tensor<2, dim> df_dtau(const Tensor<2, dim> &tau,
                                   const Vector<double> &xi) = 0;

    virtual Vector<double> df_dxi(const Tensor<2, dim> &tau,
                                  const Vector<double> &xi) = 0;

//    IsotropicHardeningLaw<dim> *hardening_law = new PerfectPlasticity<dim>();
//    IsotropicHardeningLaw<dim> *hardening_law = new LinearHardening<dim>();
//    IsotropicHardeningLaw<dim> *hardening_law = new Voce<dim>();
    IsotropicHardeningLaw<dim> *hardening_law = new VoceWLinear<dim>();
};

template<unsigned int dim>
class VonMises : public YieldSurface<dim> {
public:
    VonMises() = default;

    double f(const Tensor<2, dim> &tau,
             const Vector<double> &xi) override {
        return (tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.).norm()
               - sqrt(2. / 3.) * this->hardening_law->sig_y(xi[0]);
    };

    Tensor<2, dim> df_dtau(const Tensor<2, dim> &tau,
                           const Vector<double> &xi) override {
        Tensor<2, dim> t_bar = tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.;
        return t_bar / t_bar.norm();
    };

    Vector<double> df_dxi(const Tensor<2, dim> &tau,
                          const Vector<double> &xi) override {
        Vector<double> out(1);
        out[0] = -sqrt(2. / 3.) * this->hardening_law->dsig_y(xi[0]);
        return out;
    };
};

template<unsigned int dim>
class FlowRule {
public:
    virtual Tensor<2, dim> T(const Tensor<2, dim> &tau,
                             const Vector<double> &xi) = 0;


};

template<unsigned int dim>
class MaximumDissipation : public FlowRule<dim> {
public:
    MaximumDissipation() = default;

    virtual Tensor<2, dim> T(const Tensor<2, dim> &tau,
                             const Vector<double> &xi) {
        Tensor<2, dim> t_bar = tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.;
        return t_bar / t_bar.norm();
    };
};

template<unsigned int dim>
class PlasticityTheory {
public:
    PlasticityTheory() = default;

    YieldSurface<dim> *yield_surface = new VonMises<dim>();
    FlowRule<dim> *flow_rule = new MaximumDissipation<dim>();
};


template<unsigned int dim>
class RateIndependentPlasticity {
public:
    RateIndependentPlasticity() { iota(range.begin(), range.end(), 0); };

    RateIndependentPlasticity(ElasticMaterial<dim> *e_law) : RateIndependentPlasticity() {
        elastic_law = e_law;
    };

    RateIndependentPlasticity(ElasticMaterial<dim> *e_law,
                              PlasticityTheory<dim> *p_theory) :
            RateIndependentPlasticity(e_law) {
        plasticity_theory = p_theory;
    };

    void increment(const double &dt,
                   Tensor<4, dim> &c,
                   Tensor<2, dim> &tau);


    Tensor<4, dim> approximate_tangent(const double &dt);

    void increment_old(const double &dt,
                       Tensor<4, dim> &c,
                       Tensor<2, dim> &tau);

    void increment_old_old(const double &dt,
                           Tensor<4, dim> &c,
                           Tensor<2, dim> &tau);


    void set_state(PlasticityState<dim> *state_ptr) { this->state = state_ptr; };

private:
    PlasticityTheory<dim> *plasticity_theory = new PlasticityTheory<dim>();
    PlasticityState<dim> *state;
    ElasticMaterial<dim> *elastic_law = new NeoHookIsoVol<dim>();
    array<unsigned int, dim> range;
    double alpha = 0.5;
//    double alpha = 1.;
    double small_number = 1e-1;
    const Tensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;
    Tensor<2, dim> L, D, dFdt;
    double dt_plastic;
    unsigned int int_steps = 10;

//    TODO Consider explicit predictor implicit corrector

    TimeIntegration *integrator = new RungeKutta();
//    IntegrationMethod int_method = ExplicitPredictorImplicitCorrector;
    IntegrationMethod int_method = GenericIntegrator;
//    IntegrationMethod int_method = MidPoint;

    double find_yield_time_step_fraction() const;

    void use_generic_integrator(unsigned int n_steps = 1);

    void implicit_correction();

    static void unpack_y(const Vector<double> &y, Tensor<2, dim> &Fp, Vector<double> &int_vars,
                         const array<unsigned int, dim> &range);

    static void pack_dy(Vector<double> &dy, const Tensor<2, dim> &dFp, const Vector<double> &dint_vars,
                        const array<unsigned int, dim> &range);

    void mid_point_method();

    Tensor<2, dim> F(const double &t) { return state->F_n + t * dFdt; };

    static Tensor<2, dim> F_func(const double &dt, const Tensor<2, dim> &dFdt, const Tensor<2, dim> &F_n);


    static Vector<double> dy(const double &t,
                             const Vector<double> &y,
                             ElasticMaterial<dim> *elastic_law,
                             PlasticityTheory<dim> *plasticity_theory,
                             const Tensor<2, dim> &dFdt,
                             const Tensor<2, dim> &L,
                             const Tensor<2, dim> &D,
                             PlasticityState<dim> *state,
                             const array<unsigned int, dim> &range);

    static Tensor<2, dim> dtau_p(const Tensor<4, dim> &ce,
                                 const Tensor<2, dim> &tau,
                                 const Tensor<2, dim> &N,
                                 const Tensor<2, dim> &sym_N);

    static Tensor<2, dim> dtau_e(const Tensor<4, dim> &ce,
                                 const Tensor<2, dim> &tau,
                                 const Tensor<2, dim> &L,
                                 const Tensor<2, dim> &D);


    Tensor<4, dim> tangent(const Tensor<4, dim> &ce,
                           const Tensor<2, dim> &tau,
                           const Tensor<2, dim> &N,
                           const Tensor<2, dim> &sym_N) const;

    Tensor<4, dim> alt_tangent(const Tensor<4, dim> &ce,
                               const Tensor<2, dim> &tau,
                               const Tensor<2, dim> &N,
                               const Tensor<2, dim> &D,
                               const Tensor<2, dim> &sym_N);

    Tensor<2, dim> dtau_e_inner_df_dtau(const Tensor<4, dim> &ce,
                                        const Tensor<2, dim> &tau,
                                        const Tensor<2, dim> &df_dtau) const;

    double dlambda(const Tensor<4, dim> &ce,
                   const Tensor<2, dim> &tau,
                   const Vector<double> &xi,
                   const Tensor<2, dim> &N,
                   const Tensor<2, dim> &sym_N,
                   const Tensor<2, dim> &L,
                   const Tensor<2, dim> &D,
                   const double &dlambda);

    static double dlambda(const Tensor<4, dim> &ce,
                          const Tensor<2, dim> &tau,
                          const Vector<double> &xi,
                          const Tensor<2, dim> &N,
                          const Tensor<2, dim> &sym_N,
                          const Tensor<2, dim> &L,
                          const Tensor<2, dim> &D,
                          const double &dlambda,
                          PlasticityTheory<dim> *plasticity_theory);

    double r_lam(const double &dlambda,
                 const double &dt,
                 const Tensor<2, dim> &T,
                 const Tensor<2, dim> &L,
                 const Tensor<2, dim> &D);

    double solve_lam(const double &dlambda_init,
                     const double &dt,
                     const Tensor<2, dim> &T,
                     const Tensor<2, dim> &L,
                     const Tensor<2, dim> &D);

    static Tensor<2, dim> sym(const Tensor<2, dim> &in);

    Tensor<2, dim> F_pn1(const double &dt,
                         const double &dlambda,
                         const Tensor<2, dim> &T) const;

    void mid_state_variables(Vector<double> &out, const Vector<double> &int_variables_n1);

    Vector<double> increment_internal_variables(const double &dt,
                                                const Tensor<2, dim> &tau_n05,
                                                const double &dlambda_n1);


    template<class T>
    T mid_step(const T &in_n, const T &in_n1) const;

};


template<unsigned int dim>
void RateIndependentPlasticity<dim>::increment(const double &dt,
                                               Tensor<4, dim> &c,
                                               Tensor<2, dim> &tau) {
    Tensor<2, dim> F_en1 = state->F_n1 * invert(state->Fp_n1);
    Tensor<2, dim> tau_n1 = elastic_law->tau(F_en1);
    double f = plasticity_theory->yield_surface->f(tau_n1, state->int_variables_n1);


    if (f < 0) {
        tau = tau_n1;
        c = elastic_law->jaumann_tangent(F_en1);
        state->dlambda_n1 = 0;
    } else {
        Tensor<2, dim> F_en, tau_n, T_n, T_n1, N_n, N_n1, sym_N_n, sym_N_n1, L_n05, D_n05, F_n05;
        Tensor<4, dim> c_en05;
        bool cross_yield = fabs(state->dlambda_n) < 1e-12;
        double dlambda_n = state->dlambda_n;
        Tensor<2, dim> F_n = state->F_n;
        dt_plastic = dt;
        double frac;
        if (cross_yield) {
            frac = find_yield_time_step_fraction();
            dt_plastic = dt * (1 - frac);
            state->F_n = state->F_n + (state->F_n1 - state->F_n) * frac;
        }

        dFdt = (state->F_n1 - state->F_n) / dt_plastic;
        F_n05 = mid_step(state->F_n, state->F_n1);
        L_n05 = dFdt * invert(F_n05);
        D_n05 = sym(L_n05);

        L = L_n05;
        D = D_n05;

        if (cross_yield) {
            elastic_law->stress_and_tangent(state->F_n, tau_n, c_en05);
            state->T_n = plasticity_theory->flow_rule->T(tau_n, state->int_variables_n);
            N_n = state->F_n * state->T_n * invert(state->F_n);
            sym_N_n = sym(N_n);
            double dlam_guess = scalar_product(L_n05, state->T_n1) / pow(state->T_n1.norm(), 2);
            state->dlambda_n = dlambda(c_en05, tau_n, state->int_variables_n, N_n, sym_N_n, L_n05, D_n05,
                                       dlam_guess);

            state->T_n1 = state->T_n;
            state->dlambda_n1 = state->dlambda_n;
        }

        switch (int_method) {
            case ExplicitPredictorImplicitCorrector:
                use_generic_integrator(int_steps);
                implicit_correction();
                break;
            case GenericIntegrator:
                use_generic_integrator(int_steps);
                break;
            case MidPoint:
                mid_point_method();
                break;
        }
        F_en1 = state->F_n1 * invert(state->Fp_n1);
        elastic_law->stress_and_tangent(F_en1, tau, c);

        state->T_n1 = plasticity_theory->flow_rule->T(tau, state->int_variables_n1);
        N_n1 = F_en1 * state->T_n1 * invert(F_en1);
        sym_N_n1 = sym(N_n1);
        state->dlambda_n1 = dlambda(c, tau, state->int_variables_n1, N_n1, sym_N_n1, L, D, state->dlambda_n);

//        c = tangent(c, tau, N_n1, sym_N_n1);
        c = alt_tangent(c, tau, N_n1, D, sym_N_n1);

        this->state->F_n = F_n;
        this->state->dlambda_n = dlambda_n;
    }
    f = plasticity_theory->yield_surface->f(tau, state->int_variables_n1);
    state->f = f;
    state->tau_n1 = tau;
    state->tangent = c;
}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::use_generic_integrator(unsigned int n_steps) {
    integrator->set_t_start(0);
    integrator->set_t_end(dt_plastic);
    integrator->set_n_steps(n_steps);

    Vector<double> y_n(dim * dim + state->int_variables_n.size());
    pack_dy(y_n, state->Fp_n, state->int_variables_n, range);

    auto dy_in = bind(dy, placeholders::_1, placeholders::_2,
                      this->elastic_law,
                      this->plasticity_theory,
                      this->dFdt, this->L, this->D, this->state, this->range);
//    integrator->set_n_steps(200);
    Vector<double> y_n1(y_n.size());
    y_n1 = integrator->integrate(y_n, dy_in);
    unpack_y(y_n1, state->Fp_n1, state->int_variables_n1, range);
}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::implicit_correction() {
    Tensor<2, dim> tau, T_n1, predictor_Fp_n1, F_en1, T_n, T_n05;
    Tensor<4, dim> c;
    double dlam;
    unsigned int n_int_vars = state->int_variables_n1.size();
    Vector<double> predictor_int_vars(n_int_vars),
            corrected_int_vars(n_int_vars), int_vars_hat(n_int_vars);

    predictor_int_vars = state->int_variables_n1;
    predictor_Fp_n1 = state->Fp_n1;

    F_en1 = state->F_n1 * invert(predictor_Fp_n1);
    elastic_law->stress_and_tangent(F_en1, tau, c);
    T_n1 = plasticity_theory->flow_rule->T(tau, predictor_int_vars);
    T_n05 = mid_step(state->T_n, T_n1);
    dlam = (state->int_variables_n1[0] - state->int_variables_n[0]) / dt_plastic
           / plasticity_theory->yield_surface->hardening_law->h(dlam, tau, predictor_int_vars)[0];
    dlam = solve_lam(dlam, dt_plastic, T_n05, this->L, this->D);
    state->int_variables_n1[0] = state->int_variables_n[0]
                                 + dlam * dt_plastic *
                                   plasticity_theory->yield_surface->hardening_law->h(dlam, tau, predictor_int_vars)[0];
    state->Fp_n1 = F_pn1(dt_plastic, dlam, T_n05);
}


template<unsigned int dim>
void RateIndependentPlasticity<dim>::mid_point_method() {

//    Tensor<2, dim> F_en05, F_en, F_en1, tau_n05, tau_n1, T_n05, N_n05, sym_N_n05, L_n05, D_n05, F_n05, dFdt;
    Tensor<2, dim> F_en05, F_en, F_en1, tau_n05, tau_n1, T_n05, T_n1, F_n05;
    Tensor<2, dim> dtau_e, dtau_p;
    Tensor<4, dim> c_en05;
    Vector<double> xi_n05(state->int_variables_n1.size()), int_variables_n1(state->int_variables_n1.size());
    double dlambda_n05;
    double f;

    double dlambda_n = state->dlambda_n;
    double dlambda_n1 = state->dlambda_n1;
    Tensor<2, dim> F_n = state->F_n;
    bool cheat = true;

    state->Fp_n1 = F_pn1(dt_plastic,
                         mid_step(state->dlambda_n, dlambda_n1),
                         mid_step(state->T_n, state->T_n1));

    F_en = state->F_n * invert(state->Fp_n);
    F_en1 = state->F_n1 * invert(state->Fp_n1);
    F_en05 = mid_step(F_en, F_en1);
    elastic_law->stress_and_tangent(F_en05, tau_n05, c_en05);
    int_variables_n1 = increment_internal_variables(dt_plastic, tau_n05, state->dlambda_n1);
    mid_state_variables(xi_n05, int_variables_n1);
    T_n05 = plasticity_theory->flow_rule->T(tau_n05, xi_n05);
    tau_n1 = elastic_law->tau(F_en1);
    T_n1 = plasticity_theory->flow_rule->T(tau_n05, state->int_variables_n1);

    //        while(fabs(f/tau_n1.norm())>small_number){
    while (cheat) {
        dlambda_n1 = solve_lam(dlambda_n1, dt_plastic, T_n1, L, D);

        dlambda_n05 = mid_step(state->dlambda_n, dlambda_n1);
        state->Fp_n1 = F_pn1(dt_plastic, dlambda_n05, T_n05);
        F_en1 = state->F_n1 * invert(state->Fp_n1);

        F_en05 = mid_step(F_en, F_en1);
        elastic_law->stress_and_tangent(F_en05, tau_n05, c_en05);
//        increment_internal_variables(dt_plastic, tau_n05);
        int_variables_n1 = increment_internal_variables(dt_plastic, tau_n05, dlambda_n1);
        mid_state_variables(xi_n05, int_variables_n1);
        T_n05 = plasticity_theory->flow_rule->T(tau_n05, xi_n05);

        tau_n1 = elastic_law->tau(F_en1);
        T_n1 = plasticity_theory->flow_rule->T(tau_n1, int_variables_n1);

        f = plasticity_theory->yield_surface->f(tau_n1, int_variables_n1);
        cheat = false;
    }

    state->Fp_n1 = F_pn1(dt_plastic, dlambda_n05, T_n05);
    state->int_variables_n1 = increment_internal_variables(dt_plastic, tau_n05, dlambda_n1);

}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::increment_old(const double &dt,
                                                   Tensor<4, dim> &c,
                                                   Tensor<2, dim> &tau) {

    // TODO think about updates to state variables due to global Newton-Raphson iterations
    bool first_yield = true;
    Tensor<2, dim> F_en1 = state->F_n1 * invert(state->Fp_n1);

    Tensor<2, dim> tau_n1 = elastic_law->tau(F_en1);
    double f = plasticity_theory->yield_surface->f(tau_n1, state->int_variables_n1);
    if (f < 0) {
        tau = tau_n1;
        c = elastic_law->jaumann_tangent(F_en1);
//        TODO think about updating of state variables in plastic to elastic scenario
        return;
    } else {
        Tensor<2, dim> F_n_holder = state->F_n;
        double dlambda_n_holder = state->dlambda_n;
        double dt_plastic = dt;
        double dlam_diff, dlambda_n1;
        dlam_diff = small_number + 1;
        Tensor<2, dim> F_en05, F_en, tau_n05, T_n05, N_n05, sym_N_n05, L_n05, D_n05, F_n05, dFdt;
        Tensor<2, dim> dtau_e, dtau_p;
        Tensor<4, dim> c_en05;
        Vector<double> xi_n05(state->int_variables_n1.size());
        double dlambda_n05;


        if (first_yield) {
            double frac = find_yield_time_step_fraction();
            dt_plastic = dt * (1 - frac);
            state->F_n = state->F_n + (state->F_n1 - state->F_n) * frac;
            state->Fp_n1 = state->Fp_n + (state->F_n1 - state->F_n);
        } else {
            state->Fp_n1 = F_pn1(dt, state->dlambda_n1, state->T_n1);
        }

        dFdt = (state->F_n1 - state->F_n) / dt_plastic;
        F_n05 = mid_step(state->F_n, state->F_n1);
        L_n05 = dFdt * invert(F_n05);
        D_n05 = sym(L_n05);
        F_en = state->F_n * invert(state->Fp_n);

        if (first_yield) {
            elastic_law->stress_and_tangent(F_en, tau_n05, c_en05);
            T_n05 = plasticity_theory->flow_rule->T(tau_n05, xi_n05);
            N_n05 = F_en * T_n05 * invert(F_en);
            sym_N_n05 = sym(N_n05);
            state->dlambda_n = dlambda(c_en05, tau_n05, state->int_variables_n,
                                       N_n05, sym_N_n05,
                                       L_n05, D_n05,
                                       mid_step(state->dlambda_n, state->dlambda_n1));
        }

//        if(fabs(state->dlambda_n1)<small_number)
//        if(first_yield)
//            state->Fp_n1 = state->Fp_n + (state->F_n1 - state->F_n);
//        else
//            state->Fp_n1 = F_pn1(dt, state->dlambda_n1, state->T_n1);

        F_en1 = state->F_n1 * invert(state->Fp_n1);
        F_en05 = mid_step(F_en, F_en1);
        elastic_law->stress_and_tangent(F_en05, tau_n05, c_en05);
        increment_internal_variables(dt_plastic, tau_n05);
        mid_state_variables(xi_n05);

        while (fabs(dlam_diff) > small_number) {
            T_n05 = plasticity_theory->flow_rule->T(tau_n05, xi_n05);
            N_n05 = F_en05 * T_n05 * invert(F_en05);
            sym_N_n05 = sym(N_n05);
            dlambda_n05 = dlambda(c_en05, tau_n05, xi_n05,
                                  N_n05, sym_N_n05,
                                  L_n05, D_n05,
                                  mid_step(state->dlambda_n, state->dlambda_n1));
            dlambda_n1 = state->dlambda_n1;
            state->dlambda_n1 = (dlambda_n05 - (1 - alpha) * state->dlambda_n) / alpha;
            dlam_diff = dlambda_n1 - state->dlambda_n1;
            state->Fp_n1 = F_pn1(dt_plastic, dlambda_n05, T_n05);
            F_en1 = state->F_n1 * invert(state->Fp_n1);
            F_en05 = mid_step(F_en, F_en1);
            elastic_law->stress_and_tangent(F_en05, tau_n05, c_en05);
            increment_internal_variables(dt, tau_n05);
            mid_state_variables(xi_n05);

            tau_n1 = elastic_law->tau(F_en1);
            f = plasticity_theory->yield_surface->f(tau_n1, state->int_variables_n1);
        }

        this->state->F_n = F_n_holder;
        this->state->dlambda_n = dlambda_n_holder;
        elastic_law->stress_and_tangent(F_en1, tau, c);
        state->T_n1 = plasticity_theory->flow_rule->T(tau, state->int_variables_n1);
        N_n05 = F_en1 * state->T_n1 * invert(F_en1); // NB Note: this is actually the n1 *NOT* n05 value
        sym_N_n05 = sym(N_n05);
        state->dlambda_n1 = dlambda(c, tau, state->int_variables_n1,
                                    N_n05, sym_N_n05,
                                    L_n05, D_n05,
                                    state->dlambda_n1);
//        cout << "ce:" << endl;
//        cout << c << endl;
        c = tangent(c, tau, N_n05, sym_N_n05);
    }
}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::increment_old_old(const double &dt,
                                                       Tensor<4, dim> &c,
                                                       Tensor<2, dim> &tau) {

    // TODO think about updates to state variables due to global Newton-Raphson iterations
    bool first_yield = true;
    Tensor<2, dim> F_en1 = state->F_n1 * invert(state->Fp_n1);

    Tensor<2, dim> tau_n1 = elastic_law->tau(F_en1);
    double f = plasticity_theory->yield_surface->f(tau_n1, state->int_variables_n1);
    if (f < 0) {
        tau = tau_n1;
        c = elastic_law->jaumann_tangent(F_en1);
//        TODO think about updating of state variables in plastic to elastic scenario
        return;
    } else {
        Tensor<2, dim> F_n_holder = state->F_n;
        double dlambda_n_holder = state->dlambda_n;
        double dt_plastic = dt;

        Tensor<2, dim> F_en05, F_en, tau_n05, T_n05, N_n05, sym_N_n05, L_n05, D_n05, F_n05, dFdt;
        Tensor<2, dim> dtau_e, dtau_p;
        Tensor<4, dim> c_en05;
        Vector<double> xi_n05(state->int_variables_n1.size());
        double dlambda_n05;


        if (first_yield) {
            double frac = find_yield_time_step_fraction();
            dt_plastic = dt * (1 - frac);
            state->F_n = state->F_n + (state->F_n1 - state->F_n) * frac;
            state->Fp_n1 = state->Fp_n + (state->F_n1 - state->F_n);
        } else {
            state->Fp_n1 = F_pn1(dt, state->dlambda_n1, state->T_n1);
        }

        dFdt = (state->F_n1 - state->F_n) / dt_plastic;
        F_n05 = mid_step(state->F_n, state->F_n1);
        L_n05 = dFdt * invert(F_n05);
        D_n05 = sym(L_n05);
        F_en = state->F_n * invert(state->Fp_n);

        if (first_yield) {
            elastic_law->stress_and_tangent(F_en, tau_n05, c_en05);
            T_n05 = plasticity_theory->flow_rule->T(tau_n05, xi_n05);
            N_n05 = F_en * T_n05 * invert(F_en);
            sym_N_n05 = sym(N_n05);
            state->dlambda_n = dlambda(c_en05, tau_n05, state->int_variables_n,
                                       N_n05, sym_N_n05,
                                       L_n05, D_n05,
                                       mid_step(state->dlambda_n, state->dlambda_n1));
            state->dlambda_n1 = state->dlambda_n;
            state->T_n = T_n05;
            state->T_n1 = state->T_n;
        }

        F_en1 = state->F_n1 * invert(state->Fp_n1);
        F_en05 = mid_step(F_en, F_en1);
        elastic_law->stress_and_tangent(F_en05, tau_n05, c_en05);
        increment_internal_variables(dt_plastic, tau_n05);
        mid_state_variables(xi_n05);
        state->dlambda_n1 = solve_lam(state->dlambda_n1, dt_plastic, state->T_n1, L_n05, D_n05);
        state->Fp_n1 = F_pn1(dt, state->dlambda_n1, state->T_n1);
        F_en1 = state->F_n1 * invert(state->Fp_n1);
        elastic_law->stress_and_tangent(F_en1, tau, c);
        state->T_n1 = plasticity_theory->flow_rule->T(tau, xi_n05);
        this->state->F_n = F_n_holder;
        this->state->dlambda_n = dlambda_n_holder;
//        elastic_law->stress_and_tangent(F_en1, tau, c);
        N_n05 = F_en1 * state->T_n1 * invert(F_en1); // NB Note: this is actually the n1 *NOT* n05 value
        sym_N_n05 = sym(N_n05);
        c = tangent(c, tau, N_n05, sym_N_n05);
    }
}


template<unsigned int dim>
template<class T>
T RateIndependentPlasticity<dim>::mid_step(const T &in_n, const T &in_n1) const {
    return alpha * in_n1 + (1 - alpha) * in_n;
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::F_pn1(const double &dt,
                                                     const double &dlambda,
                                                     const Tensor<2, dim> &T) const {
    return invert(this->I / dt - alpha * dlambda * T) * (this->I / dt + (1 - alpha) * dlambda * T) * state->Fp_n;
}

template<unsigned int dim>
Vector<double> RateIndependentPlasticity<dim>::increment_internal_variables(const double &dt,
                                                                            const Tensor<2, dim> &tau_n05,
                                                                            const double &dlambda_n1) {
    double dlambda_n05 = mid_step(state->dlambda_n, dlambda_n1);
    Vector<double> xi_n05(state->int_variables_n1.size());
    Vector<double> out(state->int_variables_n1.size());
    xi_n05.add(alpha, state->int_variables_n1);
    xi_n05.add(1 - alpha, state->int_variables_n);
    out = state->int_variables_n;
    out.add(dt * dlambda_n05, plasticity_theory->yield_surface->hardening_law->h(dlambda_n05, tau_n05, xi_n05));
    return out;
}


template<unsigned int dim>
void RateIndependentPlasticity<dim>::mid_state_variables(Vector<double> &out, const Vector<double> &int_variables_n1) {
//    out = alpha * state->int_variables_n;
//    Vector<double> out(state->int_variables_n1.size());
    out = 0;
    out.add(alpha, state->int_variables_n1);
    out.add(1 - alpha, state->int_variables_n);
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::sym(const Tensor<2, dim> &in) {
    return (in + transpose(in)) / 2.;
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::dtau_e(const Tensor<4, dim> &ce,
                                                      const Tensor<2, dim> &tau,
                                                      const Tensor<2, dim> &L,
                                                      const Tensor<2, dim> &D) {
    return double_contract<2, 0, 3, 1>(ce, D) + L * tau + tau * transpose(L);
}

template<unsigned int dim>
Tensor<2, dim>
RateIndependentPlasticity<dim>::dtau_p(const Tensor<4, dim> &ce,
                                       const Tensor<2, dim> &tau,
                                       const Tensor<2, dim> &N,
                                       const Tensor<2, dim> &sym_N) {
    return double_contract<2, 0, 3, 1>(ce, sym_N) + N * tau + tau * transpose(N);
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::dlambda(const Tensor<4, dim> &ce,
                                               const Tensor<2, dim> &tau,
                                               const Vector<double> &xi,
                                               const Tensor<2, dim> &N,
                                               const Tensor<2, dim> &sym_N,
                                               const Tensor<2, dim> &L,
                                               const Tensor<2, dim> &D,
                                               const double &dlambda) {
    Tensor<2, dim> df_dtau = plasticity_theory->yield_surface->df_dtau(tau, xi);
    Vector<double> df_dxi = plasticity_theory->yield_surface->df_dxi(tau, xi);
    Vector<double> h = plasticity_theory->yield_surface->hardening_law->h(dlambda, tau, xi);
    return scalar_product(df_dtau, dtau_e(ce, tau, L, D)) /
           (scalar_product(df_dtau, dtau_p(ce, tau, N, sym_N)) - df_dxi * h);
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::dlambda(const Tensor<4, dim> &ce,
                                               const Tensor<2, dim> &tau,
                                               const Vector<double> &xi,
                                               const Tensor<2, dim> &N,
                                               const Tensor<2, dim> &sym_N,
                                               const Tensor<2, dim> &L,
                                               const Tensor<2, dim> &D,
                                               const double &dlambda,
                                               PlasticityTheory<dim> *plasticity_theory) {
    Tensor<2, dim> df_dtau = plasticity_theory->yield_surface->df_dtau(tau, xi);
    Vector<double> df_dxi = plasticity_theory->yield_surface->df_dxi(tau, xi);
    Vector<double> h = plasticity_theory->yield_surface->hardening_law->h(dlambda, tau, xi);
    return scalar_product(df_dtau, dtau_e(ce, tau, L, D)) /
           (scalar_product(df_dtau, dtau_p(ce, tau, N, sym_N)) - df_dxi * h);
}

template<unsigned int dim>
Tensor<4, dim>
RateIndependentPlasticity<dim>::tangent(const Tensor<4, dim> &ce,
                                        const Tensor<2, dim> &tau,
                                        const Tensor<2, dim> &N,
                                        const Tensor<2, dim> &sym_N) const {
    Tensor<2, dim> df_dtau = plasticity_theory->yield_surface->df_dtau(tau, state->int_variables_n1);
    Tensor<2, dim> dtau_p_val = dtau_p(ce, tau, N, sym_N);
    Vector<double> df_dxi = plasticity_theory->yield_surface->df_dxi(tau, state->int_variables_n1);
    Vector<double> h = plasticity_theory->yield_surface->hardening_law->h(state->dlambda_n1, tau,
                                                                          state->int_variables_n1);
    return ce - outer_product(dtau_p_val, dtau_e_inner_df_dtau(ce, tau, df_dtau)) /
                (scalar_product(df_dtau, dtau_p_val) - df_dxi * h);
}


template<unsigned int dim>
Tensor<4, dim>
RateIndependentPlasticity<dim>::alt_tangent(const Tensor<4, dim> &ce,
                                            const Tensor<2, dim> &tau,
                                            const Tensor<2, dim> &N,
                                            const Tensor<2, dim> &D,
                                            const Tensor<2, dim> &sym_N) {
    Tensor<2, dim> df_dtau = plasticity_theory->yield_surface->df_dtau(tau, state->int_variables_n1);
    Tensor<2, dim> dtau_p_val = dtau_p(ce, tau, N, sym_N);
    Vector<double> df_dxi = plasticity_theory->yield_surface->df_dxi(tau, state->int_variables_n1);
    Vector<double> h = plasticity_theory->yield_surface->hardening_law->h(state->dlambda_n1, tau,
                                                                          state->int_variables_n1);
    Tensor<4, dim> c, c_alt;
    c = ce - outer_product(dtau_p_val, dtau_e_inner_df_dtau(ce, tau, df_dtau)) /
             (scalar_product(df_dtau, dtau_p_val) - df_dxi * h);

    FullMatrix<double> mat(dim*dim, dim*dim), mat_inv_m(dim*dim, dim*dim);
    Tensor<2, dim*dim> mat_inv;

    Tensor<1, dim * dim> D_flat = GeneralizedElastoPlasticMaterial<dim>::flatten_tensor(D);
    Tensor<1, dim * dim> N_flat = GeneralizedElastoPlasticMaterial<dim>::flatten_tensor(N);
    for(unsigned int i = 0; i< dim*dim; i++)
        for(unsigned int j = 0; j < dim*dim; j++)
            mat(i, j) = N_flat[i] * D_flat[j];
//    mat.template copy_from(outer_product(N_flat, D_flat));
    mat_inv_m.template invert(mat);
//    mat_inv_m.template copy_to(mat_inv);
    for(unsigned int i = 0; i< dim*dim; i++)
        for(unsigned int j = 0; j < dim*dim; j++)
            mat_inv[i][j] = mat_inv_m(i, j);

    Tensor<2, dim> M = state->int_variables_n1[0] * GeneralizedElastoPlasticMaterial<dim>::raise_tensor(mat_inv * N_flat);
    c_alt = ce - outer_product(dtau_p_val, M);
    typedef pair<unsigned int, unsigned int> vp;
    array<vp, 6> v {vp(0, 0),
                    vp(1, 1),
                    vp(2, 2),
                    vp(0, 1),
                    vp(0, 2),
                    vp(1, 2)};

    cout << "c:" << endl;
    for(const auto& vi: v){
        for(const auto& vj: v) {
            cout << c[vi.first][vi.second][vj.first][vj.second] << "\t\t";
        }
        cout << endl;
    }


    cout << "c alt:" << endl;
    for(const auto& vi: v){
        for(const auto& vj: v) {
            cout << c_alt[vi.first][vi.second][vj.first][vj.second] << "\t\t";
        }
        cout << endl;
    }

    return c_alt;
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::dtau_e_inner_df_dtau(const Tensor<4, dim> &ce,
                                                                    const Tensor<2, dim> &tau,
                                                                    const Tensor<2, dim> &df_dtau) const {
    return double_contract<0, 0, 1, 1>(df_dtau, ce) + df_dtau * tau + tau * df_dtau;
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::find_yield_time_step_fraction() const {
    Tensor<2, dim> dF = state->F_n1 - state->F_n;
    double fraction = 0.5;
    double in_base = 0.5;

    Tensor<2, dim> Ffrac = state->F_n + dF * fraction;
    Tensor<2, dim> tau = elastic_law->tau(Ffrac);
    double f = plasticity_theory->yield_surface->f(tau, state->int_variables_n);
    double counter = 2;
    double fraction_prev = fraction + 1;
    double eps = 1e-4;
//    while (fabs(f) > small_number) {
    while (fabs(fraction_prev - fraction) > eps) {
        fraction_prev = fraction;
        if (f > 0)
            fraction -= pow(in_base, counter);
        else
            fraction += pow(in_base, counter);

        Ffrac = (state->F_n + dF * fraction) * invert(state->Fp_n);
        tau = elastic_law->tau(Ffrac);
        f = plasticity_theory->yield_surface->f(tau, state->int_variables_n);
        counter++;
    }

    return fraction;
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::r_lam(const double &dlambda,
                                             const double &dt,
                                             const Tensor<2, dim> &T,
                                             const Tensor<2, dim> &L,
                                             const Tensor<2, dim> &D) {
    double r;
    Tensor<2, dim> tau_n1, df_dtau, tau_n05, N_n1, sym_N_n1;
    Tensor<4, dim> ce_n1, ce_n05;

    Tensor<2, dim> Fp_n1 = F_pn1(dt, dlambda, T);
    Tensor<2, dim> Fe_n1 = state->F_n1 * invert(Fp_n1);
    Tensor<2, dim> Fe_n = state->F_n * invert(state->Fp_n);
    Tensor<2, dim> Fe_n05 = mid_step(Fe_n, Fe_n1);
    elastic_law->stress_and_tangent(Fe_n1, tau_n1, ce_n1);
    elastic_law->stress_and_tangent(Fe_n05, tau_n05, ce_n05);
    Vector<double> int_variables_n1 = increment_internal_variables(dt, tau_n05, dlambda);

    N_n1 = Fe_n1 * T * invert(Fe_n1);
    sym_N_n1 = sym(N_n1);

    df_dtau = plasticity_theory->yield_surface->df_dtau(tau_n1, int_variables_n1);
    Vector<double> df_dxi = plasticity_theory->yield_surface->df_dxi(tau_n1, int_variables_n1);
    Vector<double> h = plasticity_theory->yield_surface->hardening_law->h(dlambda, tau_n1, int_variables_n1);
    return scalar_product(df_dtau, dtau_e(ce_n1, tau_n1, L, D)) +
           dlambda * (df_dxi * h - scalar_product(df_dtau, dtau_p(ce_n1, tau_n1, N_n1, sym_N_n1)));
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::solve_lam(const double &dlambda_init,
                                                 const double &dt,
                                                 const Tensor<2, dim> &T,
                                                 const Tensor<2, dim> &L,
                                                 const Tensor<2, dim> &D) {
    double eps = fabs(dlambda_init) * 1.e-7;
    if (eps == 0) {
        eps = 1.e-7;
    }
    double dr_dlam;
    double dlambda = dlambda_init;
    double r = r_lam(dlambda, dt, T, L, D);
    double dlambda_hat = dlambda + eps;
    double r_prev, dlam_prev;
    r_prev = r + 1;
    dlam_prev = dlambda + 1;
    while (fabs(r) > small_number && (fabs(r_prev - r) + fabs(dlam_prev - dlambda)) > small_number) {
        r_prev = r;
        dlam_prev = dlambda;
        dlambda_hat = dlambda + eps;
        dr_dlam = (r_lam(dlambda_hat, dt, T, L, D) - r) / eps;
        dlambda -= 0.8 * r / dr_dlam;
        eps = fabs(dlambda) * 1.e-7;
        r = r_lam(dlambda, dt, T, L, D);
    }
    return dlambda;
}

template<unsigned int dim>
Tensor<4, dim> RateIndependentPlasticity<dim>::approximate_tangent(const double &dt) {
    Tensor<2, dim> F_n1_store = state->F_n1;
    Tensor<2, dim> tau, tau_hat, d_mat, c_kl;
    Tensor<4, dim> c, tangent;
    double eps = 1e-5;
//    PlasticityState<dim> * dummy_state = state->copy_vals(state);
    PlasticityState<dim> *original_state = state;
//    state = dummy_state;

    increment(dt, c, tau);
    for (const auto &k: range) {
        for (const auto &l: range) {
            state = original_state->copy_vals(original_state);
            state->F_n1 = F_n1_store;
            d_mat = 0;
            d_mat[k][l] += 1;
            d_mat[l][k] += 1;
//            d_mat[k][l] = 1;
//            d_mat[l][k] = 1;
            state->F_n1 += eps * d_mat * F_n1_store / 2.;
//            state->F_n1 += eps * d_mat * F_n1_store ;
            increment(dt, c, tau_hat);


            c_kl = (tau_hat - tau) / eps - tau[k][l] * d_mat;
//            c_kl = (tau_hat - tau) / eps;
            for (const auto &i: range) {
                for (const auto &j: range) {
                    tangent[i][j][k][l] = c_kl[i][j];
                }
            }
            delete state;
        }
    }
    state = original_state;
//    for (const auto &k: range) {
//        for (const auto &l: range) {
//            for (const auto &i: range) {
//                for (const auto &j: range) {
//                    tangent[i][j][k][l] = tangent[i][j][l][k];
//                }
//            }
//        }
//    }

    return tangent;
}

template<unsigned int dim>
Vector<double> RateIndependentPlasticity<dim>::dy(const double &t,
                                                  const Vector<double> &y,
                                                  ElasticMaterial<dim> *elastic_law,
                                                  PlasticityTheory<dim> *plasticity_theory,
                                                  const Tensor<2, dim> &dFdt,
                                                  const Tensor<2, dim> &L,
                                                  const Tensor<2, dim> &D,
                                                  PlasticityState<dim> *state,
                                                  const array<unsigned int, dim> &range) {
    unsigned int n_dofs = y.size();
    unsigned int n_Fp_entries = dim * dim;
    unsigned int n_int_vars = n_dofs - n_Fp_entries;

    Vector<double> int_vars(n_int_vars);
    Vector<double> d_int_vars(n_int_vars);
    Tensor<2, dim> Fp, dFp;
    unpack_y(y, Fp, int_vars, range);

    Tensor<2, dim> F, Fe, tau, N, sym_N, T;
    Tensor<4, dim> ce;

//    F = F(t);
    F = F_func(t, dFdt, state->F_n);
    Fe = F * invert(Fp);
    elastic_law->stress_and_tangent(Fe, tau, ce);
    T = plasticity_theory->flow_rule->T(tau, int_vars);
    N = Fe * T * invert(Fe);
    sym_N = sym(N);
    double dlambda_val = scalar_product(L, state->T_n) / pow(state->T_n.norm(), 2);

    dlambda_val = dlambda(ce, tau, int_vars, N, sym_N, L, D, dlambda_val, plasticity_theory);

    dFp = dlambda_val * T * Fp;
    d_int_vars = plasticity_theory->yield_surface->hardening_law->h(dlambda_val, tau, int_vars);
    d_int_vars *= dlambda_val;

    Vector<double> dy_val(n_dofs);
    pack_dy(dy_val, dFp, d_int_vars, range);

    return dy_val;
}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::unpack_y(const Vector<double> &y,
                                              Tensor<2, dim> &Fp,
                                              Vector<double> &int_vars,
                                              const array<unsigned int, dim> &range) {
    unsigned int n_dofs = y.size();
    unsigned int n_Fp_entries = dim * dim;
    unsigned int n_int_vars = n_dofs - n_Fp_entries;

    int_vars.reinit(n_int_vars);

    unsigned int count = 0;
    for (const auto &i: range) {
        for (const auto &j: range) {
            Fp[i][j] = y[count];
            count++;
        }
    }
    for (unsigned int i = 0; i < n_int_vars; ++i) {
        int_vars[i] = y[i + n_Fp_entries];
    }

}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::pack_dy(Vector<double> &dy,
                                             const Tensor<2, dim> &dFp,
                                             const Vector<double> &dint_vars,
                                             const array<unsigned int, dim> &range) {
    unsigned int n_dofs = dy.size();
    unsigned int n_Fp_entries = dim * dim;
    unsigned int n_int_vars = n_dofs - n_Fp_entries;

    unsigned int count = 0;
    for (const auto &i: range) {
        for (const auto &j: range) {
            dy[count] = dFp[i][j];
            count++;
        }
    }
    for (unsigned int i = 0; i < n_int_vars; ++i) {
        dy[i + n_Fp_entries] = dint_vars[i];
    }
}

template<unsigned int dim>
Tensor<2, dim>
RateIndependentPlasticity<dim>::F_func(const double &dt, const Tensor<2, dim> &dFdt, const Tensor<2, dim> &F_n) {
    return F_n + dt * dFdt;
}


//template<unsigned int dim>
//Tensor<2, dim> RateIndependentPlasticity<dim>::r_Fpn1(const double &dlambda,
//                                                      const double &dt,
//                                                      const Tensor<2, dim> &F_pn1,
//                                                      const Tensor<2, dim> &L,
//                                                      const Tensor<2, dim> &D) {
//
//}

#endif //ALE_HISTORYDEPENDENTMATERIALS_H
