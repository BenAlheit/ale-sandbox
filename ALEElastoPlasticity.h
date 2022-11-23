//
// Created by alhei on 2022/08/02.
//

#ifndef ALE_ALEELASTOPLASTICITY_H
#define ALE_ALEELASTOPLASTICITY_H

#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

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


#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>

#include "Materials.h"
#include "Time.h"
#include "PostProcessors.h"
#include "StateFieldManager.h"
#include "MaterialStates.h"
#include "GlobalLeastSquaresProjector.h"
#include "QPField.h"
#include "HistoryDependentMaterials.h"

using namespace std;
using namespace dealii;

template<unsigned int dim>
SymmetricTensor<4, dim> symmetrize_rank4(const Tensor<4, dim> &in) {
    SymmetricTensor<4, dim> out;
    out = 0;
    array<unsigned int, dim> range;
    iota(range.begin(), range.end(), 0);
    for (const auto &i: range)
        for (const auto &j: range)
            for (const auto &k: range)
                for (const auto &l: range)
                    out[i][j][k][l] = (in[i][j][k][l] + in[j][i][k][l] +
                                       in[j][i][l][k] + in[i][j][l][k]) / 4.;

    return out;
}

template<unsigned int dim>
Tensor<1, dim> no_motion(const Point<dim> &p, const Time &time) {
    Tensor<1, dim> out;
    out = 0;
    return out;
}

template<unsigned int dim>
Tensor<1, dim> ep_translate_mesh(const Point<dim> &p, const Time &time) {
    Tensor<1, dim> out;
    out = 0;
    double scale_factor = 0.2;
    out[0] = scale_factor * (1.5 - p[1]) * time.current() / time.end();
    return out;
}

template<unsigned int dim>
Tensor<1, dim> ep_warp_cylinder(const Point<dim> &p, const Time &time) {
    Tensor<1, dim> out;
    out = 0;
    double scale_factor = 0.2;
    out[0] = scale_factor * (10 * 0.5 - p[0]) * time.current() / time.end();
    out[1] = scale_factor * (7.5 - p[1]) * time.current() / time.end();
    return out;
}


enum SimulationType {
    UniaxialTenstion,
    PlaneStrainTension,
    UniaxialCompression,
    PSUniaxialCompression,
    PSC_slide,
    PSC_stick,
    PSC_slide_contact
};

enum BoundaryIDChangeCondition {
    all, any
};

template<unsigned int dim>
class ALEElastoPlasticitySimulationConfig {
public:
    unsigned int n_refinements = 1;

    Tensor<1, dim> (*mesh_motion)(const Point<dim> &, const Time &) = &no_motion<dim>;

    Point<dim> p1 = Point<dim>({0, 0, 0});
    Point<dim> p2 = Point<dim>({10, 7.5, 15});
    vector<unsigned int> repetitions = vector<unsigned int>({4, 3, 6});
    double magnitude = 5;
    SimulationType type = PSC_stick;
    string name = "ale-elastoplastic-psc";
    unsigned int n_increments = 50;
    unsigned int n_outputs = 50;
    bool refine_close = false;
    unsigned int n_refine_close = 0;
    BoundaryIDChangeCondition condition;
};

template<unsigned int dim>
class UniaxialTensionConfig : public ALEElastoPlasticitySimulationConfig<dim> {
public:
    UniaxialTensionConfig() {
        this->p2 = Point<dim>({0.1, 1, 0.3});
        this->repetitions = vector<unsigned int>({1, 10, 3});
        this->type = UniaxialTenstion;
        this->name = "ale-elastoplastic-ut";
//        this->magnitude = 0.3;
        this->magnitude = 0.1;
    }
};

template<unsigned int dim>
class PSTensionConfig : public UniaxialTensionConfig<dim> {
public:
    PSTensionConfig() {
        this->type = PlaneStrainTension;
        this->name = "ale-elastoplastic-psut";
    }
};

template<unsigned int dim>
class UniaxialCompressionConfig : public UniaxialTensionConfig<dim> {
public:
    UniaxialCompressionConfig() {
        this->type = UniaxialCompression;
        this->name = "ale-elastoplastic-uc";
    }
};

template<unsigned int dim>
class UniaxialPSCompressionConfig : public UniaxialTensionConfig<dim> {
public:
    UniaxialPSCompressionConfig() {
        this->type = PSUniaxialCompression;
        this->name = "ale-elastoplastic-psuc";
    }
};

template<unsigned int dim>
class StickPSCConfig : public ALEElastoPlasticitySimulationConfig<dim> {
public:
    StickPSCConfig() {
        this->type = PSC_stick;
        this->name = "ale-elastoplastic-psc-stick";
    }
};

template<unsigned int dim>
class SlidePSCConfig : public ALEElastoPlasticitySimulationConfig<dim> {
public:
    SlidePSCConfig() {
        this->type = PSC_slide;
        this->name = "ale-elastoplastic-psc-slide";
    }
};

template<unsigned int dim>
class SlideContactPSCConfig : public ALEElastoPlasticitySimulationConfig<dim> {
public:
    SlideContactPSCConfig() {
        this->type = PSC_slide_contact;
        this->name = "ale-elastoplastic-psc-slide-contact";
        this->refine_close = true;
        this->n_refine_close = 4;
        this->condition = any;
    }
};


template<unsigned int dim>
class ALEElastoPlasticity {
public:
    explicit ALEElastoPlasticity(ALEElastoPlasticitySimulationConfig<dim> *config_in, unsigned int el_order = 1);

    explicit ALEElastoPlasticity(ALEElastoPlasticitySimulationConfig<dim> *config_in,
                                 RateIndependentPlasticity<dim> *material, unsigned int el_order = 1);

    explicit ALEElastoPlasticity(ALEElastoPlasticitySimulationConfig<dim> *config_in, const Time &time,
                                 RateIndependentPlasticity<dim> *material, unsigned int el_order = 1);

    void run();


private:
    double small_value = 1e-12;
    array<unsigned int, dim> range;
    ALEElastoPlasticitySimulationConfig<dim> *config;

//    unsigned int n_refinements = 1;
//    Tensor<1, dim> (*mesh_motion)(const Point<dim> &, const Time &);
//    Point<dim> p1 = Point<dim>({0, 0, 0});
//    Point<dim> p2 = Point<dim>({10, 7.5, 15});
//    vector<unsigned int> repetitions = vector<unsigned int>({4, 3, 6});

    void make_mesh();

    void refine_init();

    void apply_init_constraints();

    void uniaxial_constraints();

    void stick_psc_constraints();

    void slide_psc_constraints();

    void slide_contact_psc_constraints();

    void sym_constraints();

    void update_du_constraints();

    void setup_system();

    void make_projector();

    void do_time_increment();

    void initialize_step();

    void nr();

    void assemble_system(bool init = false);

    void solve();

    void move_mesh();

    void init_state_field();

    void advect_states();

    void update_states();

    void output_results() const;


    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    RateIndependentPlasticity<dim> *material;
    typedef PlasticityState<dim> state_type;
    StateFieldManager<dim, state_type> state_field;
    TimerOutput timer = TimerOutput(cout, TimerOutput::summary, TimerOutput::wall_times);

    FESystem<dim> fe;
    QGauss<dim> quadrature_formula;
    FEValues<dim> fe_values;

    GlobalLeastSquaresProjector<dim> projector;

    AffineConstraints<double> du_constraints;
    AffineConstraints<double> dummy_constraints;
    AffineConstraints<double> mesh_motion_constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix, system_matrix_prev;

    Vector<double> u;
    Vector<double> du;
    Vector<double> u_m;
    Vector<double> u_s_n;
    Vector<double> u_s;
    Vector<double> system_rhs, system_rhs_prev;

    Time time;
};

template<unsigned int dim>
ALEElastoPlasticity<dim>::ALEElastoPlasticity(ALEElastoPlasticitySimulationConfig<dim> *config_in,
                                              unsigned int el_order)
        :  ALEElastoPlasticity(config_in, new RateIndependentPlasticity<dim>(), el_order) {
//        : dof_handler(triangulation), fe(FE_Q<dim>(el_order), dim) {
}

template<unsigned int dim>
ALEElastoPlasticity<dim>::ALEElastoPlasticity(ALEElastoPlasticitySimulationConfig<dim> *config_in,
                                              RateIndependentPlasticity<dim> *material,
                                              unsigned int el_order)
        : config(config_in),
          dof_handler(triangulation),
          fe(FE_Q<dim>(el_order), dim),
          quadrature_formula(fe.degree + 1),
          fe_values(fe,
                    quadrature_formula,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values),
          projector(fe) {
    iota(range.begin(), range.end(), 0);
    this->material = material;
    time = Time((double) 1, (unsigned int) this->config->n_increments, this->config->n_outputs);

}

template<unsigned int dim>
ALEElastoPlasticity<dim>::ALEElastoPlasticity(ALEElastoPlasticitySimulationConfig<dim> *config_in,
                                              const Time &time,
                                              RateIndependentPlasticity<dim> *material,
                                              unsigned int el_order)
        : ALEElastoPlasticity(config_in, material, el_order) {
    this->time = time;
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::make_mesh() {
    GridGenerator::subdivided_hyper_rectangle(triangulation, config->repetitions, config->p1, config->p2);

    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            if (abs(face->center()[1] - config->p2[1]) < small_value && face->center()[0] < config->p2[0] * 0.5) {
                face->set_boundary_id(1);
            } else {
                for (const auto &i: range) {
                    if (abs(face->center()[i]) < small_value) {
                        face->set_boundary_id(i + 2);
                        break;
                    } else if (abs(face->center()[i] - config->p2[i]) < small_value) {
                        face->set_boundary_id(i + 5);
                        break;
                    }
                }
            }
        }
    }

    refine_init();


    ofstream ofs("psc-" + to_string(config->n_refinements) + ".dtri");

    {
        boost::archive::text_oarchive oa(ofs);
        triangulation.save(oa, 0);
    }
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::refine_init() {
    if (config->refine_close) {
        Point<dim> mesh_point;
        for (int i = 0; i < config->n_refine_close; ++i) {
            for (auto &cell: triangulation.active_cell_iterators()) {
                if (cell->at_boundary()) {
                    for (unsigned int i_vert = 0; i_vert < cell->n_vertices(); i_vert++) {
                        mesh_point = cell->vertex(i_vert);
                        if ((config->p2[0] * 0.3 <= mesh_point[0] && mesh_point[0] <= config->p2[0] * 0.5)
//                        if ((config->p2[0] * 0.25 <= mesh_point[0] && mesh_point[0] <= config->p2[0] * 0.5)
                            && fabs(mesh_point[1] - config->p2[1]) < small_value) {
                            cell->set_refine_flag();
                            break;
                        }
                    }
                }
            }
            triangulation.execute_coarsening_and_refinement();
        }
    }
    if (config->n_refinements > 0)
        triangulation.refine_global(config->n_refinements);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::init_state_field() {
    state_type dummy_state = state_type();
    state_field = StateFieldManager<dim, state_type>(dof_handler, fe_values, &dummy_state);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    u.reinit(dof_handler.n_dofs());
    du.reinit(dof_handler.n_dofs());
    u_s.reinit(dof_handler.n_dofs());
    u_s_n.reinit(dof_handler.n_dofs());
    u_m.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    apply_init_constraints();

}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::make_projector() {
    projector.initialize(triangulation, dof_handler, fe_values, fe);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::assemble_system(bool init) {
    timer.enter_subsection("Single threaded assembly");

    system_matrix = 0.0;
    system_rhs = 0.0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>>
            local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    Tensor<1, dim> r_i;
    Tensor<2, dim> k_ij;

    vector<Tensor<2, dim>> def_grads(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv(n_q_points);

    vector<vector<Tensor<1, dim>>> u_s_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads_s(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv_s(n_q_points);

    vector<vector<Tensor<1, dim>>> u_m_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads_m(n_q_points);
    vector<double> det_def_grads_m(n_q_points);

    Tensor<1, dim> grad_phi_i;
    Tensor<1, dim> grad_phi_j;
    Tensor<2, dim> F;
    Tensor<2, dim> tau;
    Tensor<4, dim> jaumann_tangent;
    SymmetricTensor<2, dim> sym_tau;
    SymmetricTensor<4, dim> sym_jaumann_tangent;

    Tensor<3, dim> shape_grad_i_C;
    Tensor<1, dim> shape_grad_i_tau;

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;
    state_type *qp_state;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
//        fe_values.get_function_gradients(solution,
//                                         old_solution_gradients);
        fe_values.get_function_gradients(u_s,
                                         u_s_gradients);
        fe_values.get_function_gradients(u_m,
                                         u_m_gradients);

        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            for (const auto &i: range)
                for (const auto &j: range) {
                    def_grads_s.at(q_point)[i][j] = u_s_gradients.at(q_point).at(i)[j] + ((i == j) ? 1 : 0);
                    def_grads_m.at(q_point)[i][j] = u_m_gradients.at(q_point).at(i)[j] + ((i == j) ? 1 : 0);
                }
            def_grads.at(q_point) = def_grads_s.at(q_point) * invert(def_grads_m.at(q_point));
            def_grads_inv.at(q_point) = invert(def_grads.at(q_point));
            def_grads_inv_s.at(q_point) = invert(def_grads_s.at(q_point));
            det_def_grads_m.at(q_point) = determinant(def_grads_m.at(q_point));
        }

        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            qp_state = state_field.get_state(cell->level(), cell->index(), q_point);

            if (not init) {
                qp_state->F_n1 = def_grads.at(q_point);
                material->set_state(qp_state);
                material->increment(time.get_delta_t(),
                                    jaumann_tangent,
                                    tau);
                qp_state->tau_n1 = tau;
            } else {
                tau = qp_state->tau_n1;
                jaumann_tangent = qp_state->tangent;
            }

            sym_jaumann_tangent = symmetrize_rank4<dim>(jaumann_tangent);
            sym_tau = symmetrize(tau);

            for (const auto &i_node: local_nodes) {
                grad_phi_i =
                        fe_values.shape_grad(local_node_to_dof.at(i_node).at(0), q_point) * def_grads_inv_s.at(q_point);

                shape_grad_i_tau = sym_tau * grad_phi_i;
                shape_grad_i_C = grad_phi_i * sym_jaumann_tangent;

                r_i = -shape_grad_i_tau * fe_values.JxW(q_point) * det_def_grads_m.at(q_point);
                for (const auto &i: range)
                    cell_rhs[local_node_to_dof.at(i_node).at(i)]
                            += r_i[i];

                for (const auto &j_node: local_nodes) {
                    grad_phi_j = fe_values.shape_grad(local_node_to_dof.at(j_node).at(0), q_point) *
                                 def_grads_inv_s.at(q_point);
                    k_ij = 0;
                    k_ij = (shape_grad_i_C * grad_phi_j + I * (shape_grad_i_tau * grad_phi_j))
                           * fe_values.JxW(q_point) * det_def_grads_m.at(q_point);

                    for (const auto &i: range)
                        for (const auto &j: range)
                            cell_matrix[local_node_to_dof.at(i_node).at(i)][local_node_to_dof.at(j_node).at(j)]
                                    += k_ij[i][j];
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        du_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

    timer.leave_subsection();
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::advect_states() {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>>
            local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<Tensor<2, dim>> def_grads(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv(n_q_points);

    vector<vector<Tensor<1, dim>>> u_s_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads_s(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv_s(n_q_points);

    vector<vector<Tensor<1, dim>>> u_m_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads_m(n_q_points);
    vector<double> det_def_grads_m(n_q_points);

    Tensor<2, dim> tau;
    SymmetricTensor<4, dim> sym_jaumann_tangent;


    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);

        fe_values.get_function_gradients(u_s,
                                         u_s_gradients);
        fe_values.get_function_gradients(u_m,
                                         u_m_gradients);

        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            for (const auto &i: range)
                for (const auto &j: range) {
                    def_grads_s.at(q_point)[i][j] = u_s_gradients.at(q_point).at(i)[j] + ((i == j) ? 1 : 0);
                    def_grads_m.at(q_point)[i][j] = u_m_gradients.at(q_point).at(i)[j] + ((i == j) ? 1 : 0);
                }
            def_grads.at(q_point) = def_grads_s.at(q_point) * invert(def_grads_m.at(q_point));
            def_grads_inv.at(q_point) = invert(def_grads.at(q_point));
            def_grads_inv_s.at(q_point) = invert(def_grads_s.at(q_point));
            det_def_grads_m.at(q_point) = determinant(def_grads_m.at(q_point));
        }
        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            material->stress_and_tangent(def_grads.at(q_point),
                                         tau,
                                         sym_jaumann_tangent);

            state_field.get_state(cell->level(), cell->index(), q_point)->set_F(def_grads.at(q_point));
            state_field.get_state(cell->level(), cell->index(), q_point)->set_Fm(def_grads_m.at(q_point));
            state_field.get_state(cell->level(), cell->index(), q_point)->set_Fs(def_grads_s.at(q_point));
            state_field.get_state(cell->level(), cell->index(), q_point)->set_stress(tau);
        }
    }
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::apply_init_constraints() {
    timer.enter_subsection("Apply init constraints");
    du_constraints.clear();
    switch (config->type) {
        case UniaxialTenstion :
        case PlaneStrainTension :
        case UniaxialCompression :
        case PSUniaxialCompression :
            uniaxial_constraints();
            break;
        case PSC_slide :
            slide_psc_constraints();
            break;
        case PSC_stick :
            stick_psc_constraints();
            break;
        case PSC_slide_contact :
            slide_contact_psc_constraints();
            break;
    }

    sym_constraints();
    DoFTools::make_hanging_node_constraints(dof_handler, du_constraints);
    du_constraints.close();
    du_constraints.distribute(du);

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    du_constraints,
            /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);


    mesh_motion_constraints.clear();
    ComponentMask load_region_mesh_motion_mask(dim, false);
    load_region_mesh_motion_mask.set(1, true);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ZeroFunction<dim>(dim),
                                             mesh_motion_constraints,
                                             load_region_mesh_motion_mask);
    for (const auto &i: range) {
        ComponentMask mask(dim, false);
        mask.set(i, true);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 i + 2,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 mesh_motion_constraints,
                                                 mask);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 i + 5,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 mesh_motion_constraints,
                                                 mask);
    }
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>>
            local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const vector<Point<dim> > &unit_points = fe.get_generalized_support_points();
    vector<Point<dim>> el_points;
    unsigned int x_dof;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->at_boundary()) {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            el_points.clear();
            for (auto const &val: unit_points)
                el_points.push_back(fe_values.get_mapping().transform_unit_to_real_cell(cell, val));

            for (const auto &i_node: local_nodes) {
                if (abs(el_points.at(i_node)[0] - 0.5 * config->p2[0]) < small_value &&
                    abs(el_points.at(i_node)[1] - config->p2[1]) < small_value) {
                    x_dof = local_dof_indices[local_node_to_dof.at(i_node).at(0)];
                    x_dof = local_dof_indices[local_node_to_dof.at(i_node).at(0)];
                    mesh_motion_constraints.add_line(x_dof);
                }
            }
        }
    }
    mesh_motion_constraints.close();
    timer.leave_subsection();
}


template<unsigned int dim>
void ALEElastoPlasticity<dim>::uniaxial_constraints() {
    du_constraints.clear();

    ComponentMask push_mask;
    if (config->type == PlaneStrainTension || config->type == PSUniaxialCompression) {
        push_mask = ComponentMask(dim, true);
    } else {
        push_mask = ComponentMask(dim, false);
        push_mask.set(1, true);
    }

    double mult;
    if (config->type == UniaxialTenstion || config->type == PlaneStrainTension) {
        mult = 1;
    } else {
        mult = -1;
    }

    vector<double> load_vals({0,
                              mult * config->magnitude * time.delta_stage_pct(),
                              0});

    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(load_vals),
                                             du_constraints,
                                             push_mask);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             6,
                                             Functions::ConstantFunction<dim>(load_vals),
                                             du_constraints,
                                             push_mask);

}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::stick_psc_constraints() {
    ComponentMask push_mask(dim, true);
    vector<double> load_vals({0,
                              -config->magnitude * time.delta_stage_pct(),
                              0});
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(load_vals),
                                             du_constraints,
                                             push_mask);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::slide_psc_constraints() {
    ComponentMask push_mask(dim, false);
    push_mask.set(1, true);
    vector<double> load_vals({0,
                              -config->magnitude * time.delta_stage_pct(),
                              0});

    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(load_vals),
                                             du_constraints,
                                             push_mask);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::slide_contact_psc_constraints() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    Vector<double> u_local(dofs_per_cell);

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>>
            local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const vector<Point<dim> > &unit_points = fe.get_generalized_support_points();
    vector<Point<dim>> el_points;
    unsigned int x_dof;
    double spatial_x, mesh_x;
    types::global_dof_index vert_x_dof;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->at_boundary()) {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            el_points.clear();
            for (auto const &val: unit_points)
                el_points.push_back(fe_values.get_mapping().transform_unit_to_real_cell(cell, val));

            cell->get_dof_values(u_s, u_local);
            for (auto const &face: cell->face_iterators()) {
                if (face->boundary_id() == 1) {
                    if (config->condition == any) {
                        for (unsigned int i_vert = 0; i_vert < face->n_vertices(); i_vert++) {
                            Point<dim> mesh_point = face->vertex(i_vert);
                            vert_x_dof = face->vertex_dof_index(i_vert, 0);
                            spatial_x = mesh_point[0] + u_s[vert_x_dof];
                            if (spatial_x > config->p2[0] * 0.5 ) {
                                cout << "Setting face with vertices: ";
                                for (unsigned int j_vert = 0; j_vert < face->n_vertices(); j_vert++)
                                    cout << face->vertex_index(j_vert) << " ";
                                cout << "to boundary id 6." << endl;
                                face->set_boundary_id(6);
                                break;
                            }
                        }
                    } else if (config->condition == all) {
                        bool all_over = true;
                        for (unsigned int i_vert = 0; i_vert < face->n_vertices(); i_vert++) {
                            Point<dim> mesh_point = face->vertex(i_vert);
                            vert_x_dof = face->vertex_dof_index(i_vert, 0);
                            spatial_x = mesh_point[0] + u_s[vert_x_dof];
                            all_over = spatial_x > config->p2[0] * 0.5;
                            if (not all_over) {
                                break;
                            }
                        }
                        if(all_over){
                            cout << "Setting face with vertices: ";
                            for (unsigned int j_vert = 0; j_vert < face->n_vertices(); j_vert++)
                                cout << face->vertex_index(j_vert) << " ";
                            cout << "to boundary id 6." << endl;
                            face->set_boundary_id(6);
                        }
                    } else {
                        throw invalid_argument("Need to either choose 'all' or 'any' for boundary change condition.");
                    }
                }
            }
        }
    }


    ComponentMask push_mask(dim, false);
    push_mask.set(1, true);
//    push_mask.set(2, true);
    vector<double> load_vals({0,
                              -config->magnitude * time.delta_stage_pct(),
                              0});

    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(load_vals),
                                             du_constraints,
                                             push_mask);
}


template<unsigned int dim>
void ALEElastoPlasticity<dim>::sym_constraints() {
    for (const auto &i: range) {
        ComponentMask mask(dim, false);
        mask.set(i, true);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 i + 2,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 du_constraints,
                                                 mask);
    }
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::update_du_constraints() {
    dummy_constraints.clear();
    dummy_constraints.template copy_from(du_constraints);
    du_constraints.clear();
    du_constraints.template copy_from(dummy_constraints);
    for (const auto &line: du_constraints.get_lines()) {
        du_constraints.set_inhomogeneity(line.index, 0);
    }
    du_constraints.close();
}


template<unsigned int dim>
void ALEElastoPlasticity<dim>::initialize_step() {
    apply_init_constraints();
    move_mesh();
//    SolverControl solver_control(1000, 1e-12);
    SolverControl solver_control(5000, 1e-8*system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
//    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    SparseDirectUMFPACK direct_solver;

    u = u_s;
    u.add(-1.0, u_m);
    assemble_system(true);
    preconditioner.initialize(system_matrix, 1.2);
//    preconditioner.initialize(system_matrix);
    timer.enter_subsection("Linear solve");
    cg.solve(system_matrix, du, system_rhs, preconditioner);
//    direct_solver.solve(system_matrix, system_rhs);
//    du = system_rhs;
    timer.leave_subsection();
    du_constraints.distribute(du);
    u_s.add(0.99, du);
    u = u_s;
    u.add(-1.0, u_m);
    update_du_constraints();
    update_states();
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::nr() {
//    SolverControl solver_control(1000, 1e-12);
//    SolverControl solver_control(5000, 1e-12);
    SolverControl solver_control(5000, 1e-8*system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
//    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    u = u_s;
    u.add(-1.0, u_m);

    SparseDirectUMFPACK direct_solver;

    for (unsigned int it = 0; it < 8; it++) {
        assemble_system();
        cout << "Iteration: " << it << "\t residual norm: " << system_rhs.l2_norm() << endl;
        preconditioner.initialize(system_matrix, 1.2);
//        preconditioner.initialize(system_matrix);
        timer.enter_subsection("Linear solve");
//        direct_solver.solve(system_matrix, system_rhs);
//        du = system_rhs;
        cg.solve(system_matrix, du, system_rhs, preconditioner);
        timer.leave_subsection();
        du_constraints.distribute(du);
        u_s.add(0.99, du);
        u = u_s;
        u.add(-1.0, u_m);
    }
    u_s_n = u_s;
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::do_time_increment() {
    bool output = time.increment();
    cout << "******************" << endl;
    cout << "Time step: " << time.get_timestep() << "\t Time: " << time.current() << endl;
//    apply_constraints();
//    apply_init_constraints();
//    move_mesh();
    initialize_step();
    nr();
//    update_states();
    if (output) output_results();
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::solve() {
//    time.increment();
    output_results();
    assemble_system();
    while (time.current() < time.end()) {
        do_time_increment();
    }
}

//template<unsigned int dim>
//double ALEElasticity<dim>::get_strain(unsigned int level,
//                                      unsigned int cell_id,
//                                      unsigned int qp,
//                                      unsigned int i,
//                                      unsigned int j) {
//    return state_field->get_state(level, cell_id, qp)->get_E()[i][j];
//}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    vector<DataComponentInterpretation::DataComponentInterpretation>
            tensor_component_interpretation(dim * dim, DataComponentInterpretation::component_is_part_of_tensor);

    data_out.add_data_vector(u,
                             "material_to_spatial_displacement",
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.add_data_vector(u_s,
                             "mesh_to_spatial_displacement",
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.add_data_vector(u_m,
                             "mesh_to_material_displacement",
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    DoFHandler<dim> tensor_dof_handler(triangulation);
    FESystem<dim, dim> tensor_fe(FE_Q<dim, dim>(fe.degree), dim * dim);
    tensor_dof_handler.distribute_dofs(tensor_fe);
    Vector<double> boundary_ids(triangulation.n_active_cells());
    boundary_ids = 0;
    unsigned int count = 0;
    for(const auto & cell: dof_handler.active_cell_iterators()){
        if(cell->at_boundary()){
            for(const auto & face: cell->face_iterators())
                if(face->at_boundary()){
                    boundary_ids[count] = face->boundary_id();
                    break;
                }
        }
        count++;
    }

    data_out.add_data_vector(boundary_ids,
                             "boundary_ids");

    QPField<dim, Tensor<2, dim>> strain_qp_field = QPField<dim, Tensor<2, dim>>();
    for (const auto &item: state_field.material_states) {
        strain_qp_field.add_field_value(item.first[0], item.first[1], item.first[2], item.second->get_E());
    }
    Vector<double> nodal_strain = projector.project_tensor_qp_field(strain_qp_field);
    vector<string> strain_name;
    for (const auto i: range)
        for (const auto j: range)
            strain_name.push_back("E" + to_string(i + 1) + to_string(j + 1));
    data_out.add_data_vector(tensor_dof_handler,
                             nodal_strain,
                             strain_name,
                             tensor_component_interpretation);


    QPField<dim, Tensor<2, dim>> stress_qp_field = QPField<dim, Tensor<2, dim>>();
    for (const auto &item: state_field.material_states) {
        stress_qp_field.add_field_value(item.first[0], item.first[1], item.first[2], item.second->get_stress());
    }
    Vector<double> nodal_stress = projector.project_tensor_qp_field(stress_qp_field);
    vector<string> stress_name;
    for (const auto i: range)
        for (const auto j: range)
            stress_name.push_back("S" + to_string(i + 1) + to_string(j + 1));
    data_out.add_data_vector(tensor_dof_handler,
                             nodal_stress,
                             stress_name,
                             tensor_component_interpretation);

    DoFHandler<dim> scalar_dof_handler(triangulation);
    FESystem<dim, dim> scalar_fe(FE_Q<dim, dim>(fe.degree), 1);
    scalar_dof_handler.distribute_dofs(scalar_fe);

    QPField<dim, double> ep_qp_field = QPField<dim, double>();
    for (const auto &item: state_field.material_states) {
        ep_qp_field.add_field_value(item.first[0], item.first[1], item.first[2], item.second->get_ep());
    }
    Vector<double> nodal_ep = projector.project_scalar_qp_field(ep_qp_field);
    data_out.add_data_vector(scalar_dof_handler,
                             nodal_ep,
                             "ep");


    QPField<dim, double> s_norm_qp_field = QPField<dim, double>();
    for (const auto &item: state_field.material_states) {
        s_norm_qp_field.add_field_value(item.first[0], item.first[1], item.first[2], item.second->get_stress_norm());
    }
    Vector<double> nodal_s_norm = projector.project_scalar_qp_field(s_norm_qp_field);
    data_out.add_data_vector(scalar_dof_handler,
                             nodal_s_norm,
                             "stress_norm");

    QPField<dim, double> iso_s_norm_qp_field = QPField<dim, double>();
    for (const auto &item: state_field.material_states) {
        iso_s_norm_qp_field.add_field_value(item.first[0], item.first[1], item.first[2],
                                            item.second->get_iso_stress_norm());
    }
    Vector<double> nodal_iso_s_norm = projector.project_scalar_qp_field(iso_s_norm_qp_field);
    data_out.add_data_vector(scalar_dof_handler,
                             nodal_iso_s_norm,
                             "iso_stress_norm");

    QPField<dim, double> f_qp_field = QPField<dim, double>();
    for (const auto &item: state_field.material_states) {
        f_qp_field.add_field_value(item.first[0], item.first[1], item.first[2],
                                   item.second->get_f());
    }
    Vector<double> nodal_f = projector.project_scalar_qp_field(f_qp_field);
    data_out.add_data_vector(scalar_dof_handler,
                             nodal_f,
                             "f");


    data_out.build_patches(fe_values.get_mapping());
//    string name = "solution-" + to_string(time.get_timestep()) + ".vtu";
    string name = config->name + "-" + to_string(time.get_timestep()) + ".vtu";
    ofstream output(name);
    data_out.write_vtu(output);
    static vector<pair<double, string>> times_and_names;
    times_and_names.emplace_back(pair<double, string>(time.current(), name));
    string pvd_name = config->name + ".pvd";
    ofstream pvd_output(pvd_name);
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::run() {
    make_mesh();
    setup_system();
    init_state_field();
    make_projector();
//    assemble_system();
    solve();
    output_results();
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::move_mesh() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>>
            local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const vector<Point<dim> > &unit_points = fe.get_generalized_support_points();
    vector<Point<dim>> el_points;
    Point<dim> u_node;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        el_points.clear();
        for (auto const &val: unit_points)
            el_points.push_back(fe_values.get_mapping().transform_unit_to_real_cell(cell, val));

        for (const auto &i_node: local_nodes) {
            u_node = config->mesh_motion(el_points.at(i_node), time);
            for (const auto &i_comp: range)
                u_m[local_dof_indices[local_node_to_dof.at(i_node).at(i_comp)]] = u_node[i_comp];
        }
    }
    mesh_motion_constraints.distribute(u_m);
}

template<unsigned int dim>
void ALEElastoPlasticity<dim>::update_states() {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>>
            local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            state_field.get_state(cell->level(), cell->index(), q_point)->update();
        }
    }
}


#endif //ALE_ALEELASTOPLASTICITY_H
