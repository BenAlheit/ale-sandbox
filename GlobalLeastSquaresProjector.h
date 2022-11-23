//
// Created by alhei on 2022/08/11.
//

#ifndef ALE_GLOBALLEASTSQUARESPROJECTOR_H
#define ALE_GLOBALLEASTSQUARESPROJECTOR_H

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/dofs/dof_handler.h>

#include "QPField.h"

using namespace dealii;

template<unsigned int dim>
class GlobalLeastSquaresProjector {
public:
//    GlobalLeastSquaresProjector() : fe(FE_Q<dim>(1)), quadrature_formula(fe.degree + 1),
//                                    fe_values(fe,
//                                              quadrature_formula,
//                                              update_values | update_quadrature_points | update_JxW_values) {};

    GlobalLeastSquaresProjector(
//            DoFHandler<dim> &vec_dof_handler,
//                                FEValues<dim> &vec_fe_values,
            FESystem<dim, dim> &vec_fe) : fe(vec_fe.base_element(0)), quadrature_formula(fe.degree + 1),
                                          fe_values(fe,
                                                    quadrature_formula,
                                                    update_values | update_quadrature_points | update_JxW_values) {};

    void initialize(const Triangulation<dim> &triangulation,
                    DoFHandler<dim> &vec_dof_handler,
                    FEValues<dim> &vec_fe_values,
                    FESystem<dim, dim> &vec_fe);

    void project_scalar_qp_field(Vector<double> &projected_values,
                                 function<double(unsigned int, unsigned int, unsigned int)> qp_field) const;
//                                 double (*qp_field)(unsigned int level, unsigned int cell_id, unsigned int qp));

    void project_vector_qp_field(Vector<double> &projected_values,
                                 Tensor<1, dim> (*qp_field)(unsigned int level, unsigned int cell_id, unsigned int qp));

    void project_tensor_qp_field(Vector<double> &projected_values,
                                 double (*qp_field)(unsigned int level,
                                                    unsigned int cell_id,
                                                    unsigned int qp,
                                                    unsigned int i,
                                                    unsigned int j));


    void project_tensor_qp_field(Vector<double> &projected_values,
                                 const QPField<dim, Tensor<2, dim>> &qp_field);

    Vector<double> project_tensor_qp_field(const QPField<dim, Tensor<2, dim>> &qp_field) const;

    Vector<double> project_scalar_qp_field(const QPField<dim, double> &qp_field) const;

    void project_tensor_qp_field(vector<Vector<double>> &projected_values,
                                 const QPField<dim, Tensor<2, dim>> &qp_field) const;

private:
    DoFHandler<dim> dof_handler_project;
    FESystem<dim> fe;
    QGauss<dim> quadrature_formula;
    FEValues<dim> fe_values;
    SparseMatrix<double> projection_matrix;
    SparseDirectUMFPACK projection_solver;
    array<unsigned int, dim> range;
};

template<unsigned int dim>
void GlobalLeastSquaresProjector<dim>::initialize(const Triangulation<dim> &triangulation,
                                                  DoFHandler<dim> &vec_dof_handler,
                                                  FEValues<dim> &vec_fe_values,
                                                  FESystem<dim, dim> &vec_fe) {
//    fe(FESystem<dim>(vec_fe.base_element(0));
    dof_handler_project.initialize(triangulation, fe);
//    quadrature_formula = QGauss<dim>(vec_fe.degree + 1);
//    fe_values = FEValues<dim>(fe,
//                              quadrature_formula,
//                              update_values | update_quadrature_points | update_JxW_values);
    dof_handler_project.distribute_dofs(fe);

    iota(range.begin(), range.end(), 0);
    DynamicSparsityPattern dsp(dof_handler_project.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_project, dsp);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    projection_matrix.reinit(sparsity_pattern);
    vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell: dof_handler_project.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (const auto &q_index: fe_values.quadrature_point_indices())
            for (const auto &i: fe_values.dof_indices())
                for (const auto &j: fe_values.dof_indices())
                    projection_matrix.add(local_dof_indices[i],
                                          local_dof_indices[j],
                                          fe_values.shape_value(i, q_index) *
                                          fe_values.shape_value(j, q_index) *
                                          fe_values.JxW(q_index)
                    );
    }

    projection_solver.template initialize(projection_matrix);
}

template<unsigned int dim>
void GlobalLeastSquaresProjector<dim>::project_scalar_qp_field(Vector<double> &projected_values,
//                                                               double (*qp_field)(unsigned int level, unsigned int cell_id, unsigned int qp))
                                                               function<double(unsigned int, unsigned int,
                                                                               unsigned int)> qp_field) const {
    projected_values.reinit(dof_handler_project.n_dofs());
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);
    vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell: dof_handler_project.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (const auto &q_index: fe_values.quadrature_point_indices())
            for (const auto &i: fe_values.dof_indices())
                projected_values[local_dof_indices[i]] += qp_field(cell->level(), cell->index(), q_index) *
                                                          fe_values.shape_value(i, q_index) *
                                                          fe_values.JxW(q_index);
    }
    projection_solver.solve(projected_values);
}

template<unsigned int dim>
void GlobalLeastSquaresProjector<dim>::project_vector_qp_field(Vector<double> &projected_values,
                                                               Tensor<1, dim> (*qp_field)(unsigned int level,
                                                                                          unsigned int cell_id,
                                                                                          unsigned int qp)) {
    projected_values.reinit(dim * dof_handler_project.n_dofs());
    Vector<double> working_vector;
    working_vector.reinit(dof_handler_project.n_dofs());
    vector<unsigned int> dof_range(dof_handler_project.n_dofs());
    iota(dof_range.begin(), dof_range.end(), 0);

    for (const auto &i: range) {
        project_scalar_qp_field(working_vector,
                                [qp_field, &i](unsigned int level,
                                               unsigned int cell_id,
                                               unsigned int qp) { return qp_field(level, cell_id, qp)[i]; });
        for (const auto &j: dof_range) {
            projected_values[j * dim + i] = working_vector[j];
        }
    }
}

template<unsigned int dim>
void GlobalLeastSquaresProjector<dim>::project_tensor_qp_field(Vector<double> &projected_values,
                                                               double (*qp_field)(unsigned int level,
                                                                                  unsigned int cell_id,
                                                                                  unsigned int qp,
                                                                                  unsigned int i,
                                                                                  unsigned int j)) {

    projected_values.reinit(dim * dim * dof_handler_project.n_dofs());
    Vector<double> working_vector;
    working_vector.reinit(dof_handler_project.n_dofs());
    vector<unsigned int> dof_range(dof_handler_project.n_dofs());
    iota(dof_range.begin(), dof_range.end(), 0);

    for (const auto &i: range) {
        for (const auto &j: range) {
            project_scalar_qp_field(working_vector,
                                    [qp_field, &i, &j](unsigned int level,
                                                       unsigned int cell_id,
                                                       unsigned int qp) { return qp_field(level, cell_id, qp, i, j); });
            for (const auto &k: dof_range) {
                projected_values[k * dim * dim + i * dim + j] = working_vector[k];
            }
        }
    }
}

template<unsigned int dim>
void GlobalLeastSquaresProjector<dim>::project_tensor_qp_field(Vector<double> &projected_values,
                                                               const QPField<dim, Tensor<2, dim>> &qp_field) {
    projected_values.reinit(dim * dim * dof_handler_project.n_dofs());
    Vector<double> working_vector;
    working_vector.reinit(dof_handler_project.n_dofs());
    vector<unsigned int> dof_range(dof_handler_project.n_dofs());
    iota(dof_range.begin(), dof_range.end(), 0);

    for (const auto &i: range) {
        for (const auto &j: range) {
            project_scalar_qp_field(working_vector,
                                    [&qp_field, &i, &j](unsigned int level,
                                                        unsigned int cell_id,
                                                        unsigned int qp) {
                                        return qp_field->get_field_value(level, cell_id, qp)[i][j];
                                    });
            for (const auto &k: dof_range) {
                projected_values[k * dim * dim + i * dim + j] = working_vector[k];
            }
        }
    }
}


template<unsigned int dim>
Vector<double>
GlobalLeastSquaresProjector<dim>::project_tensor_qp_field(const QPField<dim, Tensor<2, dim>> &qp_field) const {
    Vector<double> projected_values;
    projected_values.reinit(dim * dim * dof_handler_project.n_dofs());
    Vector<double> working_vector;
    working_vector.reinit(dof_handler_project.n_dofs());
    vector<unsigned int> dof_range(dof_handler_project.n_dofs());
    iota(dof_range.begin(), dof_range.end(), 0);

    for (const auto &i: range) {
        for (const auto &j: range) {
            project_scalar_qp_field(working_vector,
                                    [&qp_field, &i, &j](unsigned int level,
                                                        unsigned int cell_id,
                                                        unsigned int qp) {
                                        double val = qp_field.get_field_value(level, cell_id, qp)[i][j];
                                        return val;
                                    });
            for (const auto &k: dof_range) {
                projected_values[k * dim * dim + i * dim + j] = working_vector[k];
            }
        }
    }
    return projected_values;
}

template<unsigned int dim>
Vector<double>
GlobalLeastSquaresProjector<dim>::project_scalar_qp_field(const QPField<dim, double> &qp_field) const {
    Vector<double> projected_values;
    projected_values.reinit(dof_handler_project.n_dofs());
//    Vector<double> working_vector;
//    working_vector.reinit(dof_handler_project.n_dofs());
//    vector<unsigned int> dof_range(dof_handler_project.n_dofs());
//    iota(dof_range.begin(), dof_range.end(), 0);


    project_scalar_qp_field(projected_values,
                            [&qp_field](unsigned int level,
                                        unsigned int cell_id,
                                        unsigned int qp) {
                                double val = qp_field.get_field_value(level, cell_id, qp);
                                return val;
                            });

    return projected_values;
}

template<unsigned int dim>
void GlobalLeastSquaresProjector<dim>::project_tensor_qp_field(vector<Vector<double>> &projected_values,
                                                               const QPField<dim, Tensor<2, dim>> &qp_field) const {
//    Vector<double> projected_values;
//    projected_values.reinit(dim * dim * dof_handler_project.n_dofs());
    Vector<double> working_vector;
    working_vector.reinit(dof_handler_project.n_dofs());
    vector<unsigned int> dof_range(dof_handler_project.n_dofs());
    iota(dof_range.begin(), dof_range.end(), 0);
    unsigned int k;

    for (const auto &i: range) {
        for (const auto &j: range) {
            k = i * dim + j;
            project_scalar_qp_field(working_vector,
                                    [&qp_field, &i, &j](unsigned int level,
                                                        unsigned int cell_id,
                                                        unsigned int qp) {
                                        double val = qp_field.get_field_value(level, cell_id, qp)[i][j];
                                        return val;
                                    });
            projected_values.at(k).reinit(dof_handler_project.n_dofs());
            projected_values.at(k) = working_vector;
        }
    }
}

#endif //ALE_GLOBALLEASTSQUARESPROJECTOR_H
