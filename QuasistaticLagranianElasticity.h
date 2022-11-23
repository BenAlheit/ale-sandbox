//
// Created by alhei on 2022/08/02.
//

#ifndef ALE_QUASISTATICLAGRANIANELASTICITY_H
#define ALE_QUASISTATICLAGRANIANELASTICITY_H

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
#include "Time.h"

using namespace std;
using namespace dealii;

template<unsigned int dim>
class QuasistaticLagrangianElasticity {
public:
    explicit QuasistaticLagrangianElasticity(unsigned int el_order = 1);

    explicit QuasistaticLagrangianElasticity(Material<dim> *material, unsigned int el_order = 1);

    explicit QuasistaticLagrangianElasticity(const Time & time, Material<dim> *material, unsigned int el_order = 1);

    void run();

private:
    double small_value = 1e-12;
    array<unsigned int, dim> range;

    void make_mesh();

    void apply_constraints();

    void setup_system();

    void do_time_increment();

    void nr();

    void assemble_system();

    void solve();

    void output_results() const;

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    Material<dim> *material;

    FESystem<dim> fe;

    AffineConstraints<double> sol_constraints;
    AffineConstraints<double> inc_constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> increment;
    Vector<double> system_rhs;

    Time time;
};

template<unsigned int dim>
QuasistaticLagrangianElasticity<dim>::QuasistaticLagrangianElasticity(unsigned int el_order)
        :  QuasistaticLagrangianElasticity(new StVenantKirchhoff<dim>(), el_order) {
//        : dof_handler(triangulation), fe(FE_Q<dim>(el_order), dim) {
}

template<unsigned int dim>
QuasistaticLagrangianElasticity<dim>::QuasistaticLagrangianElasticity(Material<dim> *material, unsigned int el_order)
        : dof_handler(triangulation), fe(FE_Q<dim>(el_order), dim) {
    iota(range.begin(), range.end(), 0);
    this->material = material;
    time = Time((double) 1, (unsigned int) 50);
}

template<unsigned int dim>
QuasistaticLagrangianElasticity<dim>::QuasistaticLagrangianElasticity(const Time &time,
                                                                      Material<dim> *material,
                                                                      unsigned int el_order)
        : QuasistaticLagrangianElasticity(material, el_order) {
    this->time = time;
}

template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::make_mesh() {
    GridGenerator::hyper_cube(triangulation, 0, 1);
//    triangulation.refine_global(1);
    triangulation.refine_global(3);
    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            if (abs(face->center()[1] - 1) < small_value && face->center()[0] < 0.5) {
                face->set_boundary_id(1);
            } else {
                for (const auto &i: range) {
                    if (abs(face->center()[i]) < small_value) {
                        face->set_boundary_id(i + 2);
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
}

template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
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
void QuasistaticLagrangianElasticity<dim>::assemble_system() {
    system_matrix = 0.0;
    system_rhs = 0.0;
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

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

    vector<vector<Tensor<1, dim>>> old_solution_gradients(n_q_points, vector<Tensor<1, dim>>(dim));
    vector<Tensor<2, dim>> def_grads(n_q_points);
    vector<Tensor<2, dim>> def_grads_inv(n_q_points);

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
        fe_values.get_function_gradients(solution,
                                         old_solution_gradients);

        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            for (const auto &i: range)
                for (const auto &j: range)
                    def_grads.at(q_point)[i][j] = old_solution_gradients.at(q_point).at(i)[j] + ((i == j) ? 1 : 0);

            def_grads_inv.at(q_point) = invert(def_grads.at(q_point));
        }

        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()) {
            material->stress_and_tangent(def_grads.at(q_point),
                                         tau,
                                         sym_jaumann_tangent);
            sym_tau = symmetrize(tau);
            for (const auto &i_node: local_nodes) {
                grad_phi_i =
                        fe_values.shape_grad(local_node_to_dof.at(i_node).at(0), q_point) * def_grads_inv.at(q_point);
//                shape_grad_i_tau = tau * grad_phi_i;
//                shape_grad_i_C = grad_phi_i * jaumann_tangent;
                shape_grad_i_tau = sym_tau * grad_phi_i;
                shape_grad_i_C = grad_phi_i * sym_jaumann_tangent;

                r_i = shape_grad_i_tau * fe_values.JxW(q_point);
                for (const auto &i: range)
                    cell_rhs[local_node_to_dof.at(i_node).at(i)]
                            += r_i[i];


                for (const auto &j_node: local_nodes) {
                    grad_phi_j = fe_values.shape_grad(local_node_to_dof.at(j_node).at(0), q_point) *
                                 def_grads_inv.at(q_point);
                    k_ij = 0;
                    k_ij = (shape_grad_i_C * grad_phi_j + I * (shape_grad_i_tau * grad_phi_j)) * fe_values.JxW(q_point);
//                    for (const auto &I: range)
//                        for (const auto &J: range)
//                            for (const auto &K: range)
//                                for (const auto &L: range)
//                                    k_ij[I][K] += grad_phi_i[J] * (jaumann_tangent[I][J][K][L] +
//                                                                   tau[L][J] * ((I == K) ? 1 : 0)) * grad_phi_j[L]
//                                                                   * fe_values.JxW(q_point);



                    for (const auto &i: range)
                        for (const auto &j: range)
                            cell_matrix[local_node_to_dof.at(i_node).at(i)][local_node_to_dof.at(j_node).at(j)]
                                    += k_ij[i][j];
//                                    += k_ij[i][j] * fe_values.JxW(q_point);

                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        inc_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::apply_constraints() {
    sol_constraints.clear();
    ComponentMask push_mask(dim, false);
    push_mask.set(1, true);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(-0.5*time.stage_pct(), dim),
                                             sol_constraints,
                                             push_mask);
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
    sol_constraints.distribute(solution);

    inc_constraints.clear();
    inc_constraints.template copy_from(sol_constraints);
    for (const auto &line: inc_constraints.get_lines()) {
        inc_constraints.set_inhomogeneity(line.index, 0);
    }
    inc_constraints.close();

}


template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::nr(){
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;

    for (unsigned int it = 0; it < 5; it++) {
        assemble_system();
        cout << "Iteration: " << it << "\t residual norm: " << system_rhs.l2_norm() << endl;
        preconditioner.initialize(system_matrix, 1.2);
        cg.solve(system_matrix, increment, system_rhs, preconditioner);
//        inc_constraints.distribute(increment);
//        solution -= increment;
        solution.add(-0.99, increment);
        sol_constraints.distribute(solution);
    }
}

template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::do_time_increment(){
    bool output = time.increment();
    cout << "******************" << endl;
    cout << "Time step: " << time.get_timestep() << "\t Time: " << time.current() << endl;
    apply_constraints();
    nr();
    if(output) output_results();
}

template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::solve() {
    while(time.current() < time.end()){
        do_time_increment();
    }
}


template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_names;
    switch (dim) {
        case 1:
            solution_names.emplace_back("displacement");
            break;
        case 2:
            solution_names.emplace_back("x_displacement");
            solution_names.emplace_back("y_displacement");
            break;
        case 3:
            solution_names.emplace_back("x_displacement");
            solution_names.emplace_back("y_displacement");
            solution_names.emplace_back("z_displacement");
            break;
        default: Assert(false, ExcNotImplemented());
    }
    data_out.add_data_vector(solution,
                             "displacement",
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

//    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();

    ofstream output("solution-"+to_string(time.get_timestep())+".vtk");
    data_out.write_vtk(output);

}

template<unsigned int dim>
void QuasistaticLagrangianElasticity<dim>::run() {
    make_mesh();
    setup_system();
//    assemble_system();
    solve();
    output_results();
}


#endif //ALE_QUASISTATICLAGRANIANELASTICITY_H
