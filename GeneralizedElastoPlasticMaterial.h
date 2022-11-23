//
// Created by alhei on 2022/08/24.
//

#ifndef ALE_GENERALIZEDELASTOPLASTICMATERIAL_H
#define ALE_GENERALIZEDELASTOPLASTICMATERIAL_H

template<unsigned int dim>
class GeneralizedElastoPlasticMaterial {
public:
    GeneralizedElastoPlasticMaterial(unsigned int n_internal_variables,
                                     unsigned int n_systems) :
            n_internal_variables(n_internal_variables),
            n_systems(n_systems) {
        iota(dim_range.begin(), dim_range.end(), 0);
        iota(system_range.begin(), system_range.end(), 0);
        iota(int_vars_range.begin(), int_vars_range.end(), 0);
    }


    static Tensor<1, dim * dim> flatten_tensor(const Tensor<2, dim> &in);

    static Tensor<2, dim> raise_tensor(const Tensor<1, dim * dim> &in);

protected:

    Tensor<4, dim> calculate_tangent(const Tensor<4, dim> &elastic_tangent,
                                     const Tensor<2, dim> &tau,
                                     const Tensor<2, dim> &Fe,
                                     const Tensor<2, dim> &D,
                                     const Vector<double> &slip,
                                     const Vector<double> &int_variables,
                                     const vector<bool> &active);

    virtual vector<Tensor<2, dim>> get_Ts(const Tensor<2, dim> &tau, const Vector<double> &int_variables) = 0;

    virtual Vector<double> get_slip_rates(const Tensor<2, dim> &tau, const Vector<double> &int_variables) = 0;

private:
    vector<Tensor<2, dim>>
    get_Ns(const Tensor<2, dim> &tau, const Tensor<2, dim> &Fe, const Vector<double> &int_variables,
           const vector<bool> &active);

    vector<Tensor<2, dim>>
    get_Ms(const vector<Tensor<2, dim>> &Ns, const Tensor<2, dim> &D, const Vector<double> &slip,
           const vector<bool> &active);



    static Tensor<2, dim> sym(const Tensor<2, dim> &in);

    unsigned int n_internal_variables;
    unsigned int n_systems;
    array<unsigned int, dim> dim_range;
    vector<unsigned int> system_range = vector<unsigned int>(n_systems);
    vector<unsigned int> int_vars_range = vector<unsigned int>(n_internal_variables);
};

template<unsigned int dim>
vector<Tensor<2, dim>>
GeneralizedElastoPlasticMaterial<dim>::get_Ns(const Tensor<2, dim> &tau,
                                              const Tensor<2, dim> &Fe,
                                              const Vector<double> &int_variables,
                                              const vector<bool> &active) {
    vector<Tensor<2, dim>> Ts = get_Ts(tau, int_variables);
    Tensor<2, dim> Fe_inv = invert(Fe);
    vector<Tensor<2, dim>> out(n_systems);
    for (const auto &i_sys: system_range)
        if (active.at(i_sys)) {
            out.at(i_sys) = Fe * Ts.at(i_sys) * Fe_inv;
        }

    return out;
}

template<unsigned int dim>
vector<Tensor<2, dim>> GeneralizedElastoPlasticMaterial<dim>::get_Ms(const vector<Tensor<2, dim>> &Ns,
                                                                     const Tensor<2, dim> &D,
                                                                     const Vector<double> &slip,
                                                                     const vector<bool> &active) {
    vector<Tensor<2, dim>> out(n_systems);
    Tensor<1, dim * dim> D_flat = flatten_tensor(D);
    Tensor<1, dim * dim> N_flat;
    FullMatrix<double> mat(dim * dim, dim * dim);
    for (const auto &i_sys: system_range) {
        if (active.at(i_sys)) {
            N_flat = flatten_tensor(Ns.at(i_sys));
            mat = invert(outer_product(N_flat, D_flat));
            out.at(i_sys) = slip[i_sys] * raise_tensor(mat * N_flat);
        }
    }
    return out;
}

template<unsigned int dim>
Tensor<1, dim * dim> GeneralizedElastoPlasticMaterial<dim>::flatten_tensor(const Tensor<2, dim> &in) {
    Tensor<1, dim * dim> out;
    unsigned int count = 0;
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
            out[count] = in[i][j];
            count++;
        }
    }
    return out;
}

template<unsigned int dim>
Tensor<2, dim> GeneralizedElastoPlasticMaterial<dim>::raise_tensor(const Tensor<1, dim * dim> &in) {
    Tensor<2, dim> out;
    unsigned int count = 0;
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
            out[i][j] = in[count];
            count++;
        }
    }
    return out;
}

template<unsigned int dim>
Tensor<4, dim> GeneralizedElastoPlasticMaterial<dim>::calculate_tangent(const Tensor<4, dim> &elastic_tangent,
                                                                        const Tensor<2, dim> &tau,
                                                                        const Tensor<2, dim> &Fe,
                                                                        const Tensor<2, dim> &D,
                                                                        const Vector<double> &slip,
                                                                        const Vector<double> &int_variables,
                                                                        const vector<bool> &active) {
    Tensor<4, dim> out;
    out = elastic_tangent;
    vector<Tensor<2, dim>> Ns = get_Ns(tau, Fe, int_variables, active);
    vector<Tensor<2, dim>> Ms = get_Ms(Ns, D, slip, active);
    Tensor<2, dim> p_dir;
    for (const auto &i_sys: system_range) {
        if (active.at(i_sys)) {
            p_dir = (double_contract<2, 0, 3, 1>(elastic_tangent, sym(Ns.at(i_sys)))
                     + Ns.at(i_sys) * tau
                     + tau * transpose(Ns.at(i_sys)));

            out -= outer_product(p_dir, Ms.at(i_sys));
        }
    }

    return out;
}

template<unsigned int dim>
Tensor<2, dim> GeneralizedElastoPlasticMaterial<dim>::sym(const Tensor<2, dim> &in) {
    return (in + transpose(in)) * 0.5;
}


#endif //ALE_GENERALIZEDELASTOPLASTICMATERIAL_H
