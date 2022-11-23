//
// Created by alhei on 2022/08/11.
//

#ifndef ALE_POSTPROCESSORS_H
#define ALE_POSTPROCESSORS_H

#include <deal.II/numerics/data_out.h>

using namespace dealii;
using namespace std;

template<int dim>
class StrainPostprocessor : public DataPostprocessorTensor<dim> {
public:
    StrainPostprocessor()
            :
            DataPostprocessorTensor<dim>("strain",
                                         update_gradients) {}

    void
    evaluate_vector_field
            (const DataPostprocessorInputs::Vector<dim> &input_data,
             std::vector<Vector<double> > &computed_quantities) const override {
        AssertDimension (input_data.solution_gradients.size(),
                         computed_quantities.size());

        auto cell = input_data.template get_cell<dim>();
        for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {
            AssertDimension (computed_quantities[p].size(),
                             (Tensor<2, dim>::n_independent_components));
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                    computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
                            = (input_data.solution_gradients[p][d][e]
                               +
                               input_data.solution_gradients[p][e][d]
                               +
                               input_data.solution_gradients[p][d][e] * input_data.solution_gradients[p][e][d]) / 2;
        }
    }
};

template<int dim>
class DeformationGradientPostprocessor : public DataPostprocessorTensor<dim> {
public:
    DeformationGradientPostprocessor()
            :
            DataPostprocessorTensor<dim>("F",
                                         update_gradients) {}


    void
    evaluate_vector_field
            (const DataPostprocessorInputs::Vector<dim> &input_data,
             std::vector<Vector<double> > &computed_quantities) const override {
        AssertDimension (input_data.solution_gradients.size(),
                         computed_quantities.size());
        for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {
            AssertDimension (computed_quantities[p].size(),
                             (Tensor<2, dim>::n_independent_components));
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e) {
                    computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
                            = input_data.solution_gradients[p][d][e];
                    if (d == e)
                        computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(
                                TableIndices<2>(d, e))] += 1.;
                }
        }
    }
};

#endif //ALE_POSTPROCESSORS_H
