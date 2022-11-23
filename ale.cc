#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>
#include <fstream>

#include "Materials.h"
#include "QuasistaticLagranianElasticity.h"
#include "ALEElasticity.h"
#include "ALEElastoPlasticity.h"
#include "ALEElastoPlasticityParallel.h"

#include "GeneralizedElastoPlasticMaterial.h"

#include "Tests.h"

using namespace std;
using namespace dealii;

int main() {
//int main(int argc, char **argv) {
//    TestRungeKutta rk_test = TestRungeKutta();
//    rk_test.do_test(12);
//    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const unsigned int dim = 3;
    const unsigned int fe_order = 1;
//    const double lambda = 10;
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
//    const double kappa = 0;
//    const double mu = 1;
    NeoHookIsoVol<dim>* nh = new NeoHookIsoVol<dim>(kappa, mu);
//    StVenantKirchhoff<dim>* nh = new StVenantKirchhoff<dim>(kappa, mu);
//    Tensor<2, dim> F;
//    Tensor<2, dim> tau;
//    SymmetricTensor<4, dim> tangent;
//    F = Physics::Elasticity::StandardTensors<dim>::I;
//    nh->stress_and_tangent(F, tau, tangent);



//    QuasistaticLagrangianElasticity<dim> prob(nh, fe_order);
    RateIndependentPlasticity<dim>* mat = new RateIndependentPlasticity<dim>(nh);
    UniaxialTensionConfig<dim>* config = new  UniaxialTensionConfig<dim>();
//    PSTensionConfig<dim>* config = new  PSTensionConfig<dim>();
//    StickPSCConfig<dim>* config = new  StickPSCConfig<dim>();
//    SlidePSCConfig<dim>* config = new  SlidePSCConfig<dim>();
//    SlideContactPSCConfig<dim>* config = new  SlideContactPSCConfig<dim>();
    config->n_refinements = 0;
//    config->n_refinements = 2;
    config->n_increments = 200;
    config->n_outputs = 200;
//    config->n_increments = 200;
    config->magnitude = 0.1;
////    config->n_refinements = 0;
//    config->p2[0] = 0.3;
//    config->p2[2] = 0.3;
    config->repetitions[2]=1;
//    config->repetitions[0]=3;
//    config->repetitions[2]=3;
    ALEElastoPlasticity<dim> prob(config, mat, fe_order);
    prob.run();
//
//    auto* ps = new PlasticityState<dim>();
////    RateIndependentPlasticity<dim> mat(nh);
//    mat->set_state(ps);
//    double dt = 0.1;
////    Tensor<2, dim> F;
//    Tensor<2, dim> tau;
//    Tensor<4, dim> tangent;
//    cout << "before:" << endl;
//    cout << "tau:" << endl;
//    cout << tau << endl;
//    cout << "c:" << endl;
//    cout << tangent << endl;
//    double E_xx, E_yy, E_zz;
//    E_xx = -2.9412e-4;
//    E_yy = 8.914e-4;
//    E_zz = -2.9412e-4;
//
//    typedef pair<unsigned int, unsigned int> vp;
//    array<vp, 6> v {vp(0, 0),
//                    vp(1, 1),
//                    vp(2, 2),
//                    vp(0, 1),
//                    vp(0, 2),
//                    vp(1, 2)};
////    array<pair<unsigned int, unsigned int>, 6> vj {};
////    ps->F_n[0][0] = 1.001;
////    ps->F_n *= pow(determinant(ps->F_n), -1./3.);
////    ps->F_n1[0][0] = 1.1;
////    ps->F_n1[0][1] = 1;
//    ps->F_n1[0][0] = 1.005;
////    ps->F_n1[0][2] = 0.005;
////    ps->F_n1[1][2] = 0.005;
//    ps->F_n1 *= pow(determinant(ps->F_n1), -1./3.);
//    mat->increment(dt, tangent, tau);
//    cout << "after:" << endl;
//    cout << "tau:" << endl;
//    cout << tau << endl;
//    cout << "c:" << endl;
//    for(const auto& vi: v){
//        for(const auto& vj: v) {
//            cout << tangent[vi.first][vi.second][vj.first][vj.second] << "\t\t";
//        }
//        cout << endl;
//    }
//    cout << "c elastic:" << endl;
//    tangent = nh->jaumann_tangent(ps->F_n1 * invert(ps->Fp_n1));
//    for(const auto& vi: v){
//        for(const auto& vj: v) {
//            cout << tangent[vi.first][vi.second][vj.first][vj.second] << "\t\t";
//        }
//        cout << endl;
//    }
//
////    cout << tangent << endl;
//    tangent = mat->approximate_tangent(dt);
//    cout << "c approx:" << endl;
//    for(const auto& vi: v){
//        for(const auto& vj: v) {
//            cout << tangent[vi.first][vi.second][vj.first][vj.second] << "\t\t";
//        }
//        cout << endl;
//    }

//    cout << mat->approximate_tangent(dt) << endl;

    return 0;
}