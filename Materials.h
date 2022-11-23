//
// Created by alhei on 2022/07/31.
//

#ifndef ALE_MATERIALS_H
#define ALE_MATERIALS_H

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

using namespace dealii;
using namespace std;

template<unsigned int dim>
class Material {

public:

    virtual Tensor<2, dim> tau(const Tensor<2, dim> &F) const = 0;

    virtual Tensor<4, dim> jaumann_tangent(const Tensor<2, dim> &F) const = 0;

    virtual void stress_and_tangent(const Tensor<2, dim> &F,
                                    Tensor<2, dim> &tau,
                                    Tensor<4, dim> &tangent) const = 0;

    virtual void stress_and_tangent(const Tensor<2, dim> &F,
                                    Tensor<2, dim> &tau,
                                    SymmetricTensor<4, dim> &tangent) const = 0;

protected:
    const SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;
    const SymmetricTensor<4, dim> IotimesI = Physics::Elasticity::StandardTensors<dim>::IxI;
};

template<unsigned int dim>
class ElasticMaterial : public Material<dim> {

};

template<unsigned int dim>
class StVenantKirchhoff : public ElasticMaterial<dim> {
public:
    explicit StVenantKirchhoff<dim>(double lambda_in = 52, double mu_in = 26);

    Tensor<2, dim> tau(const Tensor<2, dim> &F) const;

    Tensor<4, dim> jaumann_tangent(const Tensor<2, dim> &F) const;

    void stress_and_tangent(const Tensor<2, dim> &F,
                            Tensor<2, dim> &tau,
                            Tensor<4, dim> &tangent) const;

    void stress_and_tangent(const Tensor<2, dim> &F,
                            Tensor<2, dim> &tau,
                            SymmetricTensor<4, dim> &tangent) const;

private:
    const double lambda;
    const double mu;
    array<unsigned int, dim> range;
};

template<unsigned int dim>
StVenantKirchhoff<dim>::StVenantKirchhoff(double lambda_in, double mu_in): lambda(lambda_in), mu(mu_in) {
    iota(range.begin(), range.end(), 0);
}

template<unsigned int dim>
Tensor<2, dim> StVenantKirchhoff<dim>::tau(const Tensor<2, dim> &F) const {
    Tensor<2, dim> B = F * transpose(F);

    return (lambda * (trace(B) - (double) dim) / 2. - mu) * B
           + mu * B * B;
}

template<unsigned int dim>
Tensor<4, dim> StVenantKirchhoff<dim>::jaumann_tangent(const Tensor<2, dim> &F) const {
    Tensor<4, dim> out;
    Tensor<2, dim> B = F * transpose(F);

    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    out[i][j][k][l] = lambda * B[i][j] * B[k][l] + 2 * mu * B[i][k] * B[j][l];

    return out;
}

template<unsigned int dim>
void StVenantKirchhoff<dim>::stress_and_tangent(const Tensor<2, dim> &F,
                                                Tensor<2, dim> &tau,
                                                Tensor<4, dim> &tangent) const {
    Tensor<2, dim> B = F * transpose(F);
    tau = B * (lambda * (trace(B) - (double) dim) / 2. - mu) + mu * B * B;
    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    tangent[i][j][k][l] = lambda * B[i][j] * B[k][l] + 2 * mu * B[i][k] * B[j][l];
}

template<unsigned int dim>
void StVenantKirchhoff<dim>::stress_and_tangent(const Tensor<2, dim> &F,
                                                Tensor<2, dim> &tau,
                                                SymmetricTensor<4, dim> &tangent) const {
    Tensor<2, dim> B = F * transpose(F);
    tau = B * (lambda * (trace(B) - (double) dim) / 2. - mu) + mu * B * B;
    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    tangent[i][j][k][l] = lambda * B[i][j] * B[k][l] + 2 * mu * B[i][k] * B[j][l];
}

template<unsigned int dim>
class NeoHookIsoVol : public ElasticMaterial<dim> {
public:
    explicit NeoHookIsoVol<dim>(double kap_in = 44.12e3, double mu_in = 16.92e3);

    Tensor<2, dim> tau(const Tensor<2, dim> &F) const;

    Tensor<4, dim> jaumann_tangent(const Tensor<2, dim> &F) const;

    void stress_and_tangent(const Tensor<2, dim> &F,
                            Tensor<2, dim> &tau,
                            Tensor<4, dim> &tangent) const;

    void stress_and_tangent(const Tensor<2, dim> &F,
                            Tensor<2, dim> &tau,
                            SymmetricTensor<4, dim> &tangent) const;

private:
//    const double lambda;
    const double kap;
    const double mu;
//    const SymmetricTensor<2, dim> IodotI = Physics::Elasticity::StandardTensors<dim>::;
    array<unsigned int, dim> range;
};


template<unsigned int dim>
void NeoHookIsoVol<dim>::stress_and_tangent(const Tensor<2, dim> &F, Tensor<2, dim> &tau,
                                            SymmetricTensor<4, dim> &tangent) const {
    double J = determinant(F);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * F;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);
//    Tensor<2, dim> I = this->I;
    SymmetricTensor<4, dim> IodotI;
    IodotI = 0;
    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    IodotI[i][j][k][l] += this->I[i][k] * this->I[j][l];

    tau = mu * (Bbar - I1bar * this->I / 3.) + kap * (pow(J, 2) - 1) * this->I / 2.;
    tangent = 2 * mu * (I1bar * IodotI / 3.
                        - (outer_product(this->I, Bbar) + outer_product(Bbar, this->I)) / 3.
                        + I1bar * this->IotimesI / 9.);
    tangent += kap * (J * J * (this->IotimesI - IodotI) + IodotI);
}

template<unsigned int dim>
void
NeoHookIsoVol<dim>::stress_and_tangent(const Tensor<2, dim> &F, Tensor<2, dim> &tau, Tensor<4, dim> &tangent) const {
    double J = determinant(F);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * F;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);
//    Tensor<2, dim> I = this->I;
    SymmetricTensor<4, dim> IodotI;
    IodotI = 0;
    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    IodotI[i][j][k][l] += this->I[i][k] * this->I[j][l];

    tau = mu * (Bbar - I1bar * this->I / 3.) + kap * (pow(J, 2) - 1) * this->I / 2.;
    tangent = 2 * mu * (I1bar * IodotI / 3.
                        - (outer_product(this->I, Bbar) + outer_product(Bbar, this->I)) / 3.
                        + I1bar * this->IotimesI / 9.);
//    tangent += kap * (J * J * (this->IotimesI - IodotI) + IodotI);
    tangent = tangent + kap * (J * J * (this->IotimesI - IodotI) + IodotI);

}

template<unsigned int dim>
Tensor<4, dim> NeoHookIsoVol<dim>::jaumann_tangent(const Tensor<2, dim> &F) const {
    double J = determinant(F);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * F;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);
    SymmetricTensor<4, dim> IodotI;
    IodotI = 0;
    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    IodotI[i][j][k][l] += this->I[i][k] * this->I[j][l];


    SymmetricTensor<4, dim> tangent = 2 * mu * (I1bar * IodotI / 3.
                                       - (outer_product(this->I, Bbar) + outer_product(Bbar, this->I)) / 3.
                                       + I1bar * this->IotimesI / 9.);
    tangent += kap * (J * J * (this->IotimesI - IodotI) + IodotI);
    return tangent;
}

template<unsigned int dim>
Tensor<2, dim> NeoHookIsoVol<dim>::tau(const Tensor<2, dim> &F) const {
    double J = determinant(F);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * F;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);

    Tensor<2, dim> tau_val = mu * (Bbar - I1bar * this->I / 3.) + kap * (pow(J, 2) - 1) * this->I / 2.;
    return tau_val;
}

template<unsigned int dim>
NeoHookIsoVol<dim>::NeoHookIsoVol(double kap_in, double mu_in): kap(kap_in), mu(mu_in) {
    iota(range.begin(), range.end(), 0);
}

#endif //ALE_MATERIALS_H
