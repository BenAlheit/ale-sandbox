//
// Created by alhei on 2022/08/02.
//

#ifndef ALE_ALEELASTICITY_H
#define ALE_ALEELASTICITY_H

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
Tensor<1, dim> translate_mesh(const Point<dim> &p, const Time &time) {
    Tensor<1, dim> out;
    out = 0;
    double scale_factor = 0.2;
    out[0] = scale_factor * (1.5 - p[1]) * time.current() / time.end();
    return out;
}

template<unsigned int dim>
Tensor<1, dim> warp_cylinder(const Point<dim> &p, const Time &time) {
    Tensor<1, dim> out;
    out = 0;
    double scale_factor = 0.2;
    out[0] = scale_factor * (10 * 0.5 - p[0]) * time.current() / time.end();
    out[1] = scale_factor * (7.5 - p[1]) * time.current() / time.end();
    return out;
}


template<unsigned int dim>
class ALEElasticity {
public:
    explicit ALEElasticity(unsigned int el_order = 1);

    explicit ALEElasticity(Material<dim> *material, unsigned int el_order = 1);

    explicit ALEElasticity(const Time &time, Material<dim> *material, unsigned int el_order = 1);

    void run();

//    static double get_strain(unsigned int level,
//                             unsigned int cell_id,
//                             unsigned int qp,
//                             unsigned int i,
//                             unsigned int j);

private:
    double small_value = 1e-12;
    array<unsigned int, dim> range;

    void make_mesh();

    void apply_constraints();

    void setup_system();

    void make_projector();

    void do_time_increment();

    void nr();

    void assemble_system();

    void solve();

    void move_mesh();

    void init_state_field();

    void update_states();

    void output_results() const;

//    Tensor<1, dim> (*mesh_motion)(const Point<dim> &, const Time &) = & translate_mesh<dim>;
    Tensor<1, dim> (*mesh_motion)(const Point<dim> &, const Time &) = &warp_cylinder<dim>;

    Point<dim> p1 = Point<dim>({0, 0, 0});
    Point<dim> p2 = Point<dim>({10, 7.5, 15});
    vector<unsigned int> repetitions = vector<unsigned int>({4, 3, 6});

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    Material<dim> *material;
    StateFieldManager<dim, ALESolidState<dim>> state_field;
    TimerOutput timer = TimerOutput(cout, TimerOutput::summary, TimerOutput::wall_times);

    FESystem<dim> fe;
    QGauss<dim> quadrature_formula;
    FEValues<dim> fe_values;

    GlobalLeastSquaresProjector<dim> projector;

    AffineConstraints<double> sol_constraints;
    AffineConstraints<double> inc_constraints;
    AffineConstraints<double> mesh_motion_constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> u;
    Vector<double> u_m;
    Vector<double> u_s;
    Vector<double> u_s_prev;
    Vector<double> du_s_dt;
    Vector<double> solution;
    Vector<double> increment;
    Vector<double> system_rhs;

    Time time;
};

template<unsigned int dim>
ALEElasticity<dim>::ALEElasticity(unsigned int el_order)
        :  ALEElasticity(new NeoHookIsoVol<dim>(), el_order) {
//        : dof_handler(triangulation), fe(FE_Q<dim>(el_order), dim) {
}

template<unsigned int dim>
ALEElasticity<dim>::ALEElasticity(Material<dim> *material, unsigned int el_order)
        : dof_handler(triangulation),
          fe(FE_Q<dim>(el_order), dim),
          quadrature_formula(fe.degree + 1),
          fe_values(fe,
                    quadrature_formula,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values),
          projector(fe)
          {
    iota(range.begin(), range.end(), 0);
    this->material = material;
    time = Time((double) 1, (unsigned int) 50);
//    fe_values = FEValues<dim>(fe,
//                              quadrature_formula,
//                              update_values | update_gradients |
//                              update_quadrature_points | update_JxW_values);


}

template<unsigned int dim>
ALEElasticity<dim>::ALEElasticity(const Time &time,
                                  Material<dim> *material,
                                  unsigned int el_order)
        : ALEElasticity(material, el_order) {
    this->time = time;
}

template<unsigned int dim>
void ALEElasticity<dim>::make_mesh() {
//    GridGenerator::hyper_cube(triangulation, 0, 1);
    unsigned int n_refinements = 1;
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);
    if(n_refinements>0)
    triangulation.refine_global(n_refinements);
//    triangulation.refine_global(2);
//    triangulation.refine_global(3);
    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            if (abs(face->center()[1] - p2[1]) < small_value && face->center()[0] < p2[0] * 0.5) {
                face->set_boundary_id(1);
            } else {
                for (const auto &i: range) {
                    if (abs(face->center()[i]) < small_value) {
                        face->set_boundary_id(i + 2);
                        break;
                    } else if (abs(face->center()[i] - p2[i]) < small_value) {
                        face->set_boundary_id(i + 5);
                        break;
                    }
                }
            }
        }
    }

////  For checking ids
//    for(auto & face : triangulation.active_face_iterators()){
//        if(face->at_boundary()){
//            cout << "Face id: " << face->index()
//                 << "\tCenter: " << face->center()
//                 << "\tBoundary id: " << face->boundary_id()
//                 << endl;
//        }
//    }
//    ofstream out_n("grid-boundaries.vtk");
//    GridOut grid_out_n;
//    grid_out_n.write_vtk(triangulation, out_n);
//    double a = 0;
    ofstream ofs("psc-"+ to_string(n_refinements)+".dtri");

    {
        boost::archive::text_oarchive oa(ofs);
        triangulation.save(oa, 0);
    }
}

template<unsigned int dim>
void ALEElasticity<dim>::init_state_field() {
    ALESolidState<dim> dummy_state = ALESolidState<dim>();
    state_field = StateFieldManager<dim, ALESolidState<dim>>(dof_handler, fe_values, &dummy_state);
}

template<unsigned int dim>
void ALEElasticity<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    u.reinit(dof_handler.n_dofs());
    u_s.reinit(dof_handler.n_dofs());
    u_m.reinit(dof_handler.n_dofs());
    u_s_prev.reinit(dof_handler.n_dofs());
    du_s_dt.reinit(dof_handler.n_dofs());
//    solution.reinit(dof_handler.n_dofs());
    increment.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    apply_constraints();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    inc_constraints,
            /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}

template<unsigned int dim>
void ALEElasticity<dim>::make_projector(){
    projector.initialize(triangulation, dof_handler, fe_values, fe);
}

template<unsigned int dim>
void ALEElasticity<dim>::assemble_system() {
    timer.enter_subsection("Single threaded assembly");

    system_matrix = 0.0;
    system_rhs = 0.0;
//    QGauss<dim> quadrature_formula(fe.degree + 1);
//
//    FEValues<dim> fe_values(fe,
//                            quadrature_formula,
//                            update_values | update_gradients |
//                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>> local_node_to_dof(n_shape_fns);
    for (auto const &i_node: local_nodes)
        for (auto const &i_comp: range)
            local_node_to_dof.at(i_node).at(i_comp) = fe_values.get_fe().component_to_system_index(i_comp, i_node);

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    Tensor<1, dim> r_i;
    Tensor<2, dim> k_ij;

//    vector<vector<Tensor<1, dim>>> old_solution_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv(n_q_points);

    vector<vector<Tensor<1, dim>>> u_s_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads_s(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv_s(n_q_points);

    vector<vector<Tensor<1, dim>>> u_m_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads_m(n_q_points);
    vector<double> det_def_grads_m(n_q_points);
//    vector<Tensor<2, dim>> def_grads_inv_m(n_q_points);

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
            material->stress_and_tangent(def_grads.at(q_point),
                                         tau,
                                         sym_jaumann_tangent);
            sym_tau = symmetrize(tau);
            for (const auto &i_node: local_nodes) {
                grad_phi_i =
                        fe_values.shape_grad(local_node_to_dof.at(i_node).at(0), q_point) * def_grads_inv_s.at(q_point);

                shape_grad_i_tau = sym_tau * grad_phi_i;
                shape_grad_i_C = grad_phi_i * sym_jaumann_tangent;

                r_i = shape_grad_i_tau * fe_values.JxW(q_point) * det_def_grads_m.at(q_point);
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
        inc_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
    timer.leave_subsection();
}

template<unsigned int dim>
void ALEElasticity<dim>::update_states() {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>> local_node_to_dof(n_shape_fns);
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
void ALEElasticity<dim>::apply_constraints() {
    timer.enter_subsection("Apply constraints");
    sol_constraints.clear();
//    ComponentMask push_mask(dim, false);
//    push_mask.set(1, true);
    ComponentMask push_mask(dim, true);
    vector<double> load_vals({0,
//                              -5 * time.stage_pct(),
                              -4 * time.stage_pct(),
                              0});
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
//                                             Functions::ConstantFunction<dim>(-0.5 * time.stage_pct(), dim),
                                             Functions::ConstantFunction<dim>(load_vals),
                                             sol_constraints,
                                             push_mask);

//    VectorTools::interpolate_boundary_values(dof_handler,
//                                             6,
////                                             Functions::ConstantFunction<dim>(-0.5 * time.stage_pct(), dim),
//                                             Functions::ConstantFunction<dim>(load_vals),
//                                             sol_constraints,
//                                             push_mask);
    for (const auto &i: range) {
        ComponentMask mask(dim, false);
        mask.set(i, true);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 i + 2,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 sol_constraints,
                                                 mask);
    }
    sol_constraints.close();
    sol_constraints.distribute(u_s);

    inc_constraints.clear();
    inc_constraints.template copy_from(sol_constraints);
    for (const auto &line: inc_constraints.get_lines()) {
        inc_constraints.set_inhomogeneity(line.index, 0);
    }
    inc_constraints.close();

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

    vector<array<unsigned int, dim>> local_node_to_dof(n_shape_fns);
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
                if (abs(el_points.at(i_node)[0] - 0.5 * p2[0]) < small_value &&
                    abs(el_points.at(i_node)[1] - p2[1]) < small_value) {
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
void ALEElasticity<dim>::nr() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    u_s.add(time.get_delta_t(), du_s_dt);
    sol_constraints.distribute(u_s);
    u = u_s;
    u.add(-1.0, u_m);

    for (unsigned int it = 0; it < 4; it++) {
        assemble_system();
        cout << "Iteration: " << it << "\t residual norm: " << system_rhs.l2_norm() << endl;
        preconditioner.initialize(system_matrix, 1.2);
        timer.enter_subsection("Linear solve");
        cg.solve(system_matrix, increment, system_rhs, preconditioner);
        timer.leave_subsection();
        inc_constraints.distribute(increment);
//        solution -= increment;
        u_s.add(-0.99, increment);
        sol_constraints.distribute(u_s);
        u = u_s;
        u.add(-1.0, u_m);
    }
    du_s_dt = u_s;
    du_s_dt.add(-1.0, u_s_prev);
    du_s_dt /= time.get_delta_t();
    u_s_prev = u_s;
}

template<unsigned int dim>
void ALEElasticity<dim>::do_time_increment() {
    bool output = time.increment();
    cout << "******************" << endl;
    cout << "Time step: " << time.get_timestep() << "\t Time: " << time.current() << endl;
    apply_constraints();
    move_mesh();
    nr();
    update_states();
    if (output) output_results();
}

template<unsigned int dim>
void ALEElasticity<dim>::solve() {
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
void ALEElasticity<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    vector<DataComponentInterpretation::DataComponentInterpretation>
            tensor_component_interpretation(dim*dim, DataComponentInterpretation::component_is_part_of_tensor);

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
    FESystem<dim, dim> tensor_fe(FE_Q<dim, dim>(fe.degree), dim*dim);
    tensor_dof_handler.distribute_dofs(tensor_fe);

    QPField<dim, Tensor<2, dim>> strain_qp_field = QPField<dim, Tensor<2, dim>>();
    for(const auto& item: state_field.material_states){
        strain_qp_field.add_field_value(item.first[0], item.first[1], item.first[2], item.second->get_E());
    }
    Vector<double> nodal_strain = projector.project_tensor_qp_field(strain_qp_field);
    vector<string> strain_name;
    for(const auto i: range)
        for(const auto j: range)
            strain_name.push_back("E"+ to_string(i+1)+to_string(j+1));
    data_out.add_data_vector(tensor_dof_handler,
                             nodal_strain,
                             strain_name,
                             tensor_component_interpretation);



    QPField<dim, Tensor<2, dim>> stress_qp_field = QPField<dim, Tensor<2, dim>>();
    for(const auto& item: state_field.material_states){
        stress_qp_field.add_field_value(item.first[0], item.first[1], item.first[2], item.second->get_stress());
    }
    Vector<double> nodal_stress = projector.project_tensor_qp_field(stress_qp_field);
    vector<string> stress_name;
    for(const auto i: range)
        for(const auto j: range)
            stress_name.push_back("S"+ to_string(i+1)+to_string(j+1));
    data_out.add_data_vector(tensor_dof_handler,
                             nodal_stress,
                             stress_name,
                             tensor_component_interpretation);




    data_out.build_patches(fe_values.get_mapping());
//    string name = "solution-" + to_string(time.get_timestep()) + ".vtu";
    string name = "ale-elastic-psc-" + to_string(time.get_timestep()) + ".vtu";
    ofstream output(name);
//    data_out.write_vtk(output);
    data_out.write_vtu(output);
    static vector<pair<double, string>> times_and_names;
    times_and_names.emplace_back(pair<double, string>(time.current(), name));
    ofstream pvd_output("ale-elastic-psc.pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

template<unsigned int dim>
void ALEElasticity<dim>::run() {
    make_mesh();
    setup_system();
    init_state_field();
    make_projector();
//    assemble_system();
    solve();
    output_results();
}

template<unsigned int dim>
void ALEElasticity<dim>::move_mesh() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_shape_fns = fe_values.get_fe().base_element(0).n_dofs_per_cell();

    vector<unsigned int> local_nodes(n_shape_fns);
    iota(local_nodes.begin(), local_nodes.end(), 0);

    vector<array<unsigned int, dim>> local_node_to_dof(n_shape_fns);
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
            u_node = mesh_motion(el_points.at(i_node), time);
            for (const auto &i_comp: range)
                u_m[local_dof_indices[local_node_to_dof.at(i_node).at(i_comp)]] = u_node[i_comp];
        }
    }
    mesh_motion_constraints.distribute(u_m);
}


#endif //ALE_ALEELASTICITY_H
