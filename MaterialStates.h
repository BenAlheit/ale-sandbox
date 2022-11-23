//
// Created by alhei on 2022/08/11.
//

#ifndef ALE_MATERIALSTATES_H
#define ALE_MATERIALSTATES_H

template<unsigned int dim>
class SolidStateBase {
public:
    SolidStateBase() {};

    virtual void update() = 0;
    virtual SolidStateBase<dim> *copy() const = 0;

    virtual Tensor<2, dim> get_stress() { return tau; }

    void set_stress(const Tensor<2, dim> &new_tau) { tau = new_tau; }

    Tensor<2, dim> get_F() { return F; }

    void set_F(const Tensor<2, dim> &new_F) { F = new_F; }

    virtual Tensor<2, dim> get_E() const { return (transpose(F) * F - Physics::Elasticity::StandardTensors<dim>::I) / 2.; }

protected:
    Tensor<2, dim> tau;
    Tensor<2, dim> F;
};

template<unsigned int dim>
class ElasticSolidState : public SolidStateBase<dim> {
public:
    ElasticSolidState() {};

    ElasticSolidState<dim> *copy() const override { return new ElasticSolidState<dim>(); };
    void update() override {};

};

template<unsigned int dim>
class ALESolidState : public SolidStateBase<dim> {
public:
    ALESolidState() {};

    ALESolidState<dim> *copy() const override { return new ALESolidState<dim>(); };
    void update() override {};

    Tensor<2, dim> get_Fs() { return Fs; }

    void set_Fs(const Tensor<2, dim> &new_Fs) { Fs = new_Fs; }

    Tensor<2, dim> get_Fm() { return Fm; }

    void set_Fm(const Tensor<2, dim> &new_Fm) { Fm = new_Fm; }

protected:
    Tensor<2, dim> Fm;
    Tensor<2, dim> Fs;
};

template<unsigned int dim>
class PlasticityState : public ALESolidState<dim> {
public:
    PlasticityState() :
            F_n1(Physics::Elasticity::StandardTensors<dim>::I),
            F_n(Physics::Elasticity::StandardTensors<dim>::I),
            Fp_n1(Physics::Elasticity::StandardTensors<dim>::I),
            Fp_n(Physics::Elasticity::StandardTensors<dim>::I),
            dlambda_n(0),
            dlambda_n1(0),
            int_variables_n(1),
            int_variables_n1(1),
            f(0){};

    explicit PlasticityState(const PlasticityState<dim> *to_cpy) :
            tau_n1(to_cpy->tau_n1),
            tau_n(to_cpy->tau_n),
            F_n1(to_cpy->F_n1),
            F_n(to_cpy->F_n),
            Fp_n1(to_cpy->Fp_n1),
            Fp_n(to_cpy->Fp_n),
            dlambda_n(to_cpy->dlambda_n),
            dlambda_n1(to_cpy->dlambda_n1),
            int_variables_n(to_cpy->int_variables_n.size()),
            int_variables_n1(to_cpy->int_variables_n.size()),
            f(f){
        int_variables_n = to_cpy->int_variables_n;
        int_variables_n1 = to_cpy->int_variables_n1;
    };

    PlasticityState<dim> *copy() const override { return new PlasticityState<dim>(); };

    PlasticityState<dim> *copy_vals(const PlasticityState<dim> *to_cpy) const {
        return new PlasticityState<dim>(to_cpy);
    };
    void update() override {
        tau_n = tau_n1;
        F_n = F_n1;
        Fp_n = Fp_n1;
        dlambda_n = dlambda_n1;
        T_n = T_n1;
        int_variables_n = int_variables_n1;
    };

    Tensor<2, dim> get_stress() override { return tau_n1; }
    Tensor<2, dim> get_E() const override { return (transpose(F_n1) * F_n1 - Physics::Elasticity::StandardTensors<dim>::I) / 2.; }
    double get_ep() const { return int_variables_n1[0]; }
    double get_iso_stress_norm() const { return (tau_n1 - Physics::Elasticity::StandardTensors<dim>::I * trace(tau_n1) / 3.).norm(); }
    double get_stress_norm() const { return tau_n1.norm(); }
    double get_f() const { return f; }

    Tensor<4, dim> tangent;

    Tensor<2, dim> tau_n1;
    Tensor<2, dim> tau_n;

    Tensor<2, dim> F_n1;
    Tensor<2, dim> F_n;

    Tensor<2, dim> Fp_n1;
    Tensor<2, dim> Fp_n;

    double dlambda_n;
    double dlambda_n1;

    Tensor<2, dim> T_n1;
    Tensor<2, dim> T_n;

    Vector<double> int_variables_n;
    Vector<double> int_variables_n1;

    double f;

};

#endif //ALE_MATERIALSTATES_H
