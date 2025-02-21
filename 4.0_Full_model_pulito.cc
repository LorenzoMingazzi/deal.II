/* ---------------------------------------------------------------------
 *
 *   CODE FOR PHASE FIELD SIMULATION
 *   
 *   TRASFERIRE LA SOLUZIONE DOPO IL RAFFITTIMENTO   Vedere step 15 
 *   SNES PER IL DANNO                               Vedere step 15     
 *   SALVARE IL POST PROCESSING (deformazioni, tensioni)
 https://groups.google.com/g/dealii/c/uME5kwaJIEU/m/M-hMBIUHCAAJ 
https://groups.google.com/g/dealii/c/ymDfJIumYw4/m/hITK6VhXCgAJ

Nel futuro costruire una classe ConstitutiveLaw come 
https://www.dealii.org/developer/doxygen/deal.II/code_gallery_goal_oriented_elastoplasticity.html

// NOTE
Entrambi i campi condividono la stessa formula di quadratura

 * ---------------------------------------------------------------------
 *
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>    
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/transformations.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include <chrono>

const bool plane_strain = true;
const unsigned int model_type = 2; // 1 for full elastic energy degradation // 2 for masonry like formulation // 3 Deviatoric // 4 full-deviatoric
const unsigned int damage_type = 2; // 1 for quadratic // 2 for linear

namespace phasefield
{
    using namespace dealii;

   // Structure to store information about interface faces
    template<int dim>
    struct InterfaceFaceInfo
    {
        Point<dim> center;          // Center of the face
        unsigned int cell_index;    // Index of the cell to which the face belongs
        unsigned int face_index;    // Index of the face within the cell
        unsigned int material_id;   // Material ID of the cell

        // New members for coincident face information
        unsigned int coincident_face_index; // Index of the coincident face
        unsigned int coincident_cell_index; // Index of the coincident cell
        unsigned int coincident_material_id; // Material ID of the coincident cell

    // Non-const vectors to allow modification
        std::vector<unsigned int> face_dof_map_i; // DOF mapping for face i
        std::vector<unsigned int> face_dof_map_j; // DOF mapping for face j

        InterfaceFaceInfo()
            : cell_index(0), face_index(0), material_id(0),
            coincident_face_index(0), coincident_cell_index(0), coincident_material_id(0)
        {}
    };

    template <int dim>
    class InitialCondition : public Function<dim>
    {
        public:
        InitialCondition(double H, double m_o, double m_c)
            : Function<dim>(), H(H), m_o(m_o), m_c(m_c) {}

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (p[1] < H)  // Condizione per la parte inferiore del dominio
                return m_c;
            else  // Condizione per la parte superiore del dominio
                return m_o;
        }
        private:
        double H, m_o, m_c;
    };

    template <int dim>
    SymmetricTensor<4, dim> copy_tensor_to_symmetric_tensor(const  Tensor<4, dim> &to_be_copied)
    {
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            tmp[i][j][k][l] = to_be_copied[i][j][k][l];
    return tmp;
    }

    template <int dim>
    SymmetricTensor<2, dim> copy_tensor_to_symmetric_tensor(const  Tensor<2, dim> &to_be_copied)
    {
    SymmetricTensor<2, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
            tmp[i][j] = to_be_copied[i][j];
    return tmp;
    }
  
    template <int dim>
    struct PointHistory
    {
        SymmetricTensor<2, dim> old_stress;
        SymmetricTensor<2,dim> old_strain;
        double old_phase_field; 
    };

    template <int dim>
    SymmetricTensor<4, dim> get_stress_strain_tensor(const double lambda,
                                                     const double mu)
    {
        SymmetricTensor<4, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                            ((i == l) && (j == k) ? mu : 0.0) +
                            ((i == j) && (k == l) ? lambda : 0.0));
        return tmp;
    }

    template <int dim>
    SymmetricTensor<4, dim> get_stress_strain_tensor_dev(const double mu)
    {
        SymmetricTensor<4, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                        for (unsigned int l = 0; l < dim; ++l)
                            tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                              ((i == l) && (j == k) ? mu : 0.0) +
                              ((i == j) && (k == l) ? -2./3.*mu : 0.0));
        return tmp;
    }

    template <int dim>
    SymmetricTensor<4, dim> get_stress_strain_tensor_sph(const double lambda,
                                                         const double mu)
    {
        double bulk = lambda + 2.*mu/3.;
        SymmetricTensor<4, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        tmp[i][j][k][l] = ((i == j) && (k == l) ? bulk : 0.0);
        return tmp;
    }

    template <int dim>
    SymmetricTensor<4,dim> Identity ()
    {
        SymmetricTensor<4,dim> tmp;
        for (unsigned int i=0;i<dim;++i)
            for (unsigned int j=0;j<dim;++j)
                for (unsigned int k=0;k<dim;++k)
                    for (unsigned int l=0;l<dim;++l)
                        {
                        double a=0,b=0;
                        if (i==k & j==l)
                            a=0.5;
                        if (i==l & j==k)
                            b=0.5;
                        tmp[i][j][k][l]=a+b;
                        }
        return tmp;
    }

    template <int dim>
    inline
    Point<dim> get_tractions (const SymmetricTensor<2,dim> &stress, const Point<dim> &normal)
    {
        Assert (stress.size() == dim, ExcInternalError());

        Point<dim> traction;

        for (unsigned int i=0; i<dim; ++i)
            for (unsigned int j=0; j<dim; ++j)
	            traction[i] += stress[i][j] * normal[j];

        return traction;
    } 
 
    template <int dim>
    inline SymmetricTensor<2, dim> get_strain(const FEValues<dim>& fe_values,
        const unsigned int   shape_func,
        const unsigned int   q_point)
    {
        SymmetricTensor<2, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = i + 1; j < dim; ++j)
                tmp[i][j] =
                (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                    fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                2;
        return tmp;
    }

    template <int dim>
    inline SymmetricTensor<2, dim>
        get_strain(const std::vector<Tensor<1, dim>>& grad)
    {
        Assert(grad.size() == dim, ExcInternalError());
        SymmetricTensor<2, dim> strain;
        for (unsigned int i = 0; i < dim; ++i)
            strain[i][i] = grad[i][i];
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = i + 1; j < dim; ++j)
                strain[i][j] = (grad[i][j] + grad[j][i]) / 2;
        return strain;
    }

    Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>>& grad_u)
    {
        const double curl = (grad_u[1][0] - grad_u[0][1]);
        const double angle = std::atan(curl);
        return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
    }

    Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>>& grad_u)
    {
        const Point<3> curl(grad_u[2][1] - grad_u[1][2],
            grad_u[0][2] - grad_u[2][0],
            grad_u[1][0] - grad_u[0][1]);
        const double tan_angle = std::sqrt(curl * curl);
        const double angle = std::atan(tan_angle);
        if (std::abs(angle) < 1e-9)
        {
            static const double rotation[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
            static const Tensor<2, 3> rot(rotation);
            return rot;
        }
        const Point<3> axis = curl / tan_angle;
        return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
            -angle);
    }
 
    // ------------------------------------------------------------------------------------------
    // DAMAGE FUNCTIONS
    // ------------------------------------------------------------------------------------------
    inline double g_alpha (double &alpha)
    {
        return (1-alpha)*(1-alpha);
    }

    inline double g_prime_alpha (double &alpha)
    {
        return -2.+2.*alpha;
    }

    inline double g_second_alpha ()
    {
        return 2.;
    }

    inline double w_alpha (double &alpha)
    {   
        double tmp;
        if (damage_type == 1)
            tmp =  alpha*alpha;
        else if (damage_type == 2)
            tmp = alpha;    
        return tmp;
    }

    inline double w_prime_alpha (double &alpha)
    {   
        double tmp;
        if (damage_type == 1)
            tmp = 2.*alpha;
        else if (damage_type == 2)
            tmp = 1.;    
        return tmp;
    }

    inline double w_second_alpha ()
    {   
        double tmp;
        if (damage_type == 1)
            tmp =  2.;
        else if (damage_type == 2)
            tmp = 0.;    
        return tmp;
    }

    // ------------------------------------------------------------------------------------------
    // Class definition
    // ------------------------------------------------------------------------------------------
    template <int dim>
    class TopLevel
    {
    public:
        TopLevel();
        ~TopLevel();
        void run();

    private:
        void create_coarse_grid();
        void add_interface_constraints(const Triangulation<dim> &triangulation, std::vector<InterfaceFaceInfo<dim>> &interface_faces);
        void refine_initial_grid();
        void set_data ();
        void set_initial_values_alpha (double &a);

        // Interface functions
        void create_face_dof_map_moist(const typename DoFHandler<dim>::active_cell_iterator &cell, unsigned int face_index, std::vector<unsigned int> &face_dof_map);
        void create_face_dof_map(const typename DoFHandler<dim>::active_cell_iterator &cell, unsigned int face_index, std::vector<unsigned int> &face_dof_map);
        void update_dof_constraints(const std::vector<InterfaceFaceInfo<dim>> &interface_faces);

        // Moist diffusion
        void setup_system(std::vector<InterfaceFaceInfo<dim>> &interface_faces);
        void assemble_system(const bool flag_iter);
        void solve_linear_problem_moist(const bool flag_elastic_iter);

        double iso_canvas(double m);
        double iso_canvas_inv(double m);

        // Elasticity problem
        void evaluate_eps_c (SymmetricTensor<2,dim> &eps_pos, SymmetricTensor<2,dim> &eps_neg, const SymmetricTensor<2,dim> &eps, const double nu);
        void evaluate_stress (SymmetricTensor<2,dim> &sigma_pos, SymmetricTensor<2,dim> &sigma_neg, const SymmetricTensor<2,dim> &eps, const SymmetricTensor<2,dim> &sigma, const double EE, const double nu);
        void evaluate_stiffness_matrix (SymmetricTensor<4,dim> &C_plus, SymmetricTensor<4,dim> &C_minus, const SymmetricTensor<2,dim> &eps, const SymmetricTensor<4,dim> &C, const double EE, const double nu);   
        void stiffness_matrix(SymmetricTensor<4,dim> &C_plus, SymmetricTensor<4,dim> &C_minus, const SymmetricTensor<2,dim> &eps, const SymmetricTensor<4,dim> &C, const double EE, const double nu);         

        void setup_system_elas(std::vector<InterfaceFaceInfo<dim>> &interface_faces);
        void assemble_system_elas(const bool flag_iter);
        void solve_linear_problem(const bool flag_elastic_iter);

        // Damage problem
        void setup_system_alpha();
        void assemble_system_alpha(PETScWrappers::MPI::Vector &present_solution_alpha, PETScWrappers::MPI::Vector &system_rhs_alpha);
        void assemble_rhs_alpha(PETScWrappers::MPI::Vector &present_solution_alpha, PETScWrappers::MPI::Vector &system_rhs_alpha);
        
        static PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
        static PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void*);

        // Output
        void output_results(const unsigned int cycle) const;
        void output_results_elas(const unsigned int cycle) const;
        void output_results_alpha(const unsigned int cycle) const;     
        void output_stress(const unsigned int cycle);   
        void energy (double &bulk_energy, double &surface_energy);

        // Solve
        void solve_timestep();
        void do_initial_timestep();
        void do_timestep();

        // Moisture Matrices - Vectors
        // Elasticity Matrices - Tensor - Vectors
        PETScWrappers::MPI::SparseMatrix system_matrix;
        PETScWrappers::MPI::Vector system_rhs;

        Vector<double> solution_m;
        Vector<double> solution_m_old;
        Vector<double> newton_update_m;

        // Elasticity Matrices - Tensor - Vectors
        PETScWrappers::MPI::SparseMatrix system_matrix_elas;
        PETScWrappers::MPI::Vector system_rhs_elas;

        static const SymmetricTensor<4,dim> stress_strain_tensor;
        static const SymmetricTensor<4,dim> null_s_fourth_order_tensor;
        static const SymmetricTensor<4,dim> stress_strain_tensor_dev;
        static const SymmetricTensor<4,dim> stress_strain_tensor_sph;
        SymmetricTensor<4, dim> C_Pos, C_Neg;

        Vector<double> solution_u;
        Vector<double> newton_update_u;

        // Damage Matricies - Tensor - Vectors
        PETScWrappers::MPI::SparseMatrix system_matrix_alpha;
        PETScWrappers::MPI::Vector system_rhs_alpha;

        Vector<double> solution_alpha;
        Vector<double> solution_alpha_previous_step;

        PETScWrappers::MPI::Vector present_solution_alpha;
        PETScWrappers::MPI::Vector alpha_lb;
        PETScWrappers::MPI::Vector alpha_ub;

        // Geometry
        double L; 
        double H;
        double H_int; 

        // Diffusion parameters
        double u01; // Canvas
        double u01_bc;
        double d_coeff1;

        double u02;  // Paint
        double u02_bc;
        double d_coeff2;

        double int_coeff;

        // Elasticity / Damage parameters
        double Poisson;
        double Young;

        double k_res; 
        double alpha0;
        double ell;
        double Gc;
        double c_w;
        double error_tol;

        // Timestep parameters
        double  real_time;
        double  real_time_step;
        double  real_time_final;
        unsigned int real_timestep_no = 0;

        // Setup triangulation and FESystems
        parallel::shared::Triangulation<dim> triangulation;
        std::vector<InterfaceFaceInfo<dim>> interface_faces_moist;
        std::vector<InterfaceFaceInfo<dim>> interface_faces;

        FESystem<dim> fe_moist;
        FESystem<dim> fe;
        FESystem<dim> fe_vec;
        DoFHandler<dim> dof_handler_moist;
        DoFHandler<dim> dof_handler;
        DoFHandler<dim> dof_handler_vec;

        AffineConstraints<double> hanging_node_constraints_moist;
        AffineConstraints<double> hanging_node_constraints;
        AffineConstraints<double> hanging_node_constraints_vec;  

        const QGauss<dim> quadrature_formula;

        // Indexset
        IndexSet locally_owned_dofs_moist;
        IndexSet locally_relevant_dofs_moist;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        IndexSet locally_owned_dofs_vec;
        IndexSet locally_relevant_dofs_vec;

        // MPI communicators
        MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;

        // MPI Output on shell 
        ConditionalOStream pcout;
    };

    template <int dim>
    void TopLevel<dim>::set_data ()
    {
        // Geometry
        L = 0.05;
        H_int = 0.5;
        H = 0.6;

        // Moist
        u01 = 36.98*1e-6;  // Canvas
        u01_bc = 58.48*1e-6;
        d_coeff1 = 8e-11*1e6;

        u02 = 14.20*1e-6;  // Paint
        u02_bc = 32.067*1e-6;
        d_coeff2 = 1e-13*1e6;
        
        int_coeff = (d_coeff1+d_coeff2)*0.5;
        
        // Elasticity - Damage
        Poisson = 0.2;
        Young = 70000;

        k_res=1.e-6; 
        alpha0=0.; 
        ell=6;
        Gc=0.5;
        
        error_tol=1.e-5;
        
        if (damage_type == 1)
            c_w = 2.;
        else if (damage_type == 2)    
            c_w = 8./3.;

        // Timestepping
        real_time = 0.;
        real_time_final = 3600;
        real_time_step = 1; 
    }

    // ------------------------------------------------------------------------------------------
    // SETUP INITIAL / BC FUNCTIONS
    // ------------------------------------------------------------------------------------------
    // Alpha setup value
    template <int dim>
    void TopLevel<dim>::set_initial_values_alpha (double &alpha0)
    {  
        solution_alpha_previous_step = alpha0;     
        solution_alpha = alpha0;

        alpha_lb = alpha0;
        alpha_ub = 1-alpha0;      
    }

    // Elasticity - Body force class
    template <int dim>
    class BodyForce : public Function<dim>
    {
    public:
        BodyForce();
        virtual void vector_value(const Point<dim>& p,
            Vector<double>& values) const override;
        virtual void
            vector_value_list(const std::vector<Point<dim>>& points,
                std::vector<Vector<double>>& value_list) const override;
    };

    template <int dim>
    BodyForce<dim>::BodyForce()
        : Function<dim>(dim)
    {}

    template <int dim>
    inline void BodyForce<dim>::vector_value(const Point<dim>& /*p*/,
        Vector<double>& values) const
    {
        Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
        const double g = 9.81;
        const double rho = 7700;
        values = 0;
        values(dim - 1) = -rho * g * 0;
    }

    template <int dim>
    void BodyForce<dim>::vector_value_list(
        const std::vector<Point<dim>>& points,
        std::vector<Vector<double>>& value_list) const
    {
        const unsigned int n_points = points.size();
        Assert(value_list.size() == n_points,
            ExcDimensionMismatch(value_list.size(), n_points));
        for (unsigned int p = 0; p < n_points; ++p)
            BodyForce<dim>::vector_value(points[p], value_list[p]);
    }

    template <int dim>
    class IncrementalBoundaryValues : public Function<dim>
    {
    public:
        IncrementalBoundaryValues(const double present_time,
            const double present_timestep);
        virtual void vector_value(const Point<dim>& p,
            Vector<double>& values) const override;
        virtual void
            vector_value_list(const std::vector<Point<dim>>& points,
                std::vector<Vector<double>>& value_list) const override;
    private:
        const double velocity;
        const double present_time;
        const double present_timestep;
    };

    template <int dim>
    IncrementalBoundaryValues<dim>::IncrementalBoundaryValues(
        const double present_time,
        const double present_timestep)
        : Function<dim>(dim)
        , velocity(0.01)
        , present_time(present_time)
        , present_timestep(present_timestep)
    {}

    template <int dim>
    void IncrementalBoundaryValues<dim>::vector_value(const Point<dim>& /*p*/,
            Vector<double>& values) const
    {
        Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
        values = 0;
        values(1) = + present_time * velocity;
    }

    template <int dim>
    void IncrementalBoundaryValues<dim>::vector_value_list(
        const std::vector<Point<dim>>& points,
        std::vector<Vector<double>>& value_list) const
    {
        const unsigned int n_points = points.size();
        Assert(value_list.size() == n_points,
            ExcDimensionMismatch(value_list.size(), n_points));
        for (unsigned int p = 0; p < n_points; ++p)
            IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
    }

    // ------------------------------------------------------------------------------------------
    // ELASTICITY FUNCTIONS
    // ------------------------------------------------------------------------------------------
    // Tensors evaluations
    template <int dim>
    const SymmetricTensor<4,dim>
    TopLevel<dim>::stress_strain_tensor = get_stress_strain_tensor<dim> (/*lambda = */ (70000*0.2)/((1+0.2)*(1-2*0.2)),
                                                                         /*mu     = */ 70000/(2.*(1+0.2)));

    template <int dim>
    const SymmetricTensor<4,dim>
    TopLevel<dim>::stress_strain_tensor_dev = get_stress_strain_tensor_dev<dim> (/*mu  = */ 70000/(2.*(1+0.2)));

    template <int dim>
    const SymmetricTensor<4,dim>
    TopLevel<dim>::stress_strain_tensor_sph = get_stress_strain_tensor_sph<dim> (/*lambda = */ (70000*0.2)/((1+0.2)*(1-2*0.2)),
                                                                                 /*mu     = */ 70000/(2.*(1+0.2)));

    template <int dim>
    const SymmetricTensor<4,dim>
    TopLevel<dim>::null_s_fourth_order_tensor = get_stress_strain_tensor<dim> (0.,0.);

    template <int dim>
    void TopLevel<dim>::evaluate_eps_c(SymmetricTensor<2,dim> &eps_pos, SymmetricTensor<2,dim> &eps_neg, const SymmetricTensor<2,dim> &eps, const double nu)
    {
        double aa;
        double aa_cap;

        if (plane_strain)
            aa = nu / (1-2*nu);
        else
            aa = nu / (1-nu);
        
        aa_cap = aa/(1+aa);

        double eig_1, eig_2;
        Tensor<1,dim> n1; 
        Tensor<1,dim> n2; 

        const std::array<std::pair<double, Tensor<1, dim, double>>, dim>  eigen_eps = eigenvectors(eps);
        eig_1 = eigen_eps[0].first;
        eig_2 = eigen_eps[1].first;
        n1 = eigen_eps[0].second;
        n2 = eigen_eps[1].second;

        if (eig_2 >= 0)  // Prima condizione
        {
            eps_pos = eps;
            eps_neg = 0.;
        }
        else if ( (1+aa)*eig_1+aa*eig_2 >= 0)    //Seconda condizione
        {
            Tensor<2,dim> M1, M2;
            M1 = outer_product(n1,n1);
            M2 = outer_product(n2,n2);
            eps_pos = symmetrize((eig_1 + aa_cap*eig_2)*M1);    
            eps_neg = eps-eps_pos;
        }
        else      // Terza condizione
        {
            eps_pos = 0.;
            eps_neg = eps;
        }
    }    

    template <int dim>
    void TopLevel<dim>::evaluate_stress (SymmetricTensor<2,dim> &sigma_pos, SymmetricTensor<2,dim> &sigma_neg, const SymmetricTensor<2,dim> &eps, const SymmetricTensor<2,dim> &sigma, const double EE, const double nu)
    {
        double aa;
        double aa_cap;

        double lmbda = (EE*nu)/((1+nu)*(1-2*nu));
        double mu = EE/(2*(1+nu));

        if (plane_strain)
            aa = nu / (1-2*nu);
        else
            aa = nu / (1-nu);
        
        aa_cap = aa/(1+aa);

        double eig_1, eig_2;
        Tensor<1,dim> n1; 
        Tensor<1,dim> n2; 

        const std::array<std::pair<double, Tensor<1, dim, double>>, dim> eigen_eps = eigenvectors(eps);
        eig_1 = eigen_eps[0].first;
        eig_2 = eigen_eps[1].first;
        n1 = eigen_eps[0].second;
        n2 = eigen_eps[1].second;

        if (eig_2 >= 0)  // Prima condizione
        {
            sigma_pos = sigma;
            sigma_neg = 0.;
        }
        else if ( (1+aa)*eig_1+aa*eig_2 >= 0)    //Seconda condizione
        {
            Tensor<2,dim> M1, M2;
            M1 = outer_product(n1,n1);
            M2 = outer_product(n2,n2);
            sigma_pos = symmetrize((lmbda +2*mu)*(eig_1 + aa_cap*eig_2)*(M1 + aa_cap*M2));
            sigma_neg = symmetrize(((2.0*mu*(1.0 + 2.0*aa*(1.0 + aa)) + lmbda)/((1.0 + aa)*(1.0 + aa)))*eig_2*M2);   
        }
        else      // Terza condizione
        {
            sigma_pos = 0.;
            sigma_neg = sigma;
        }
    } 

    template <int dim>
    void TopLevel<dim>::evaluate_stiffness_matrix(SymmetricTensor<4,dim> &C_plus, SymmetricTensor<4,dim> &C_minus, const SymmetricTensor<2,dim> &eps, const SymmetricTensor<4,dim> &C, const double EE, const double nu) 
    {
    // Switch parametro alpha per plane strain/stress
        double aa;
        double aa_cap;

        double lmbda = (EE*nu)/((1+nu)*(1-2*nu));
        double mu = EE/(2*(1+nu));

        if (plane_strain)
            aa = nu / (1-2*nu);
        else
            aa = nu / (1-nu);

        aa_cap = aa/(1+aa);

        double eig_1, eig_2;
        Tensor<1,dim> n1; 
        Tensor<1,dim> n2; 

        const std::array<std::pair<double, Tensor<1, dim, double>>, dim>  eigen_eps = eigenvectors(eps);
        eig_1 = eigen_eps[0].first;
        eig_2 = eigen_eps[1].first;
        n1 = eigen_eps[0].second;
        n2 = eigen_eps[1].second;    

        if (eig_2 >= 0)    // Prima condizione
        {
            C_plus = C;
            C_minus = null_s_fourth_order_tensor;
        }  
        else if ( ((1+aa)*eig_1+aa*eig_2) >= 0)    //Seconda condizione
        {
            Tensor<4, dim>  C_plus_temp, C_minus_temp;
            Tensor<2,dim> M1, M2, M12, M21;
            M1 = outer_product(n1,n1);
            M2 = outer_product(n2,n2);
            M12 = outer_product(n1, n2);
            M21 = outer_product(n2, n1);

            Tensor<4,dim> hh, S1212, S1221, S2112, S2121, TT1, TT2;
            S1212 = outer_product(M12, M12);
            S1221 = outer_product(M12, M21);
            S2112 = outer_product(M21, M12);
            S2121 = outer_product(M21, M21);

            TT1 = (1/(2*(eig_1-eig_2)))*(S1212 + S1221 + S2112 + S2121);
            TT2 = -TT1;
        
            double coeff_pos;
            coeff_pos = lmbda + 2*mu;
            Tensor<4, dim> H = outer_product(M1 + aa_cap*M2, M1 + aa_cap*M2);
            C_plus_temp =  coeff_pos * (H + (eig_1+aa_cap*eig_2)*(TT1 + aa_cap*TT2));

            double coeff_neg;
            coeff_neg = ((2.0*mu*(1.0 + 2.0*aa*(1.0 + aa)) + lmbda)/((1.0 + aa)*(1.0 + aa)));
            C_minus_temp = coeff_neg*(outer_product(M2,M2) + eig_2*TT2);

            C_plus = copy_tensor_to_symmetric_tensor(C_plus_temp);
            C_minus = copy_tensor_to_symmetric_tensor(C_minus_temp);
        }
        else // Terza condizione
        {
            C_plus = null_s_fourth_order_tensor;
            C_minus = C;
        }
    }  

    template <int dim>
    void TopLevel<dim>::stiffness_matrix(SymmetricTensor<4,dim> &C_plus, SymmetricTensor<4,dim> &C_minus, const SymmetricTensor<2,dim> &eps_u, const SymmetricTensor<4,dim> &C, const double EE, const double nu) 
    {
        if (model_type == 1)
        {
            C_plus = C;
            C_minus = null_s_fourth_order_tensor;
        }
        else if (model_type == 2)
        {
            evaluate_stiffness_matrix(C_plus, C_minus, eps_u, C, EE, nu);
        }
        else if (model_type == 3)
        {    
            C_plus = stress_strain_tensor_dev;
            C_minus = stress_strain_tensor_sph;
        }
        else if (model_type == 4)
        {    
            if (trace(eps_u)>0)
            {
                C_plus = C;
                C_minus = null_s_fourth_order_tensor;            
            }
            else 
            {
                C_plus = stress_strain_tensor_dev;
                C_minus = stress_strain_tensor_sph;
            } 
        }
    }

    // ------------------------------------------------------------------------------------------
    // CONSTRUCTOR
    // ------------------------------------------------------------------------------------------
    template <int dim>
    TopLevel<dim>::TopLevel()
        : triangulation(MPI_COMM_WORLD)
        , fe_moist(FE_Q<dim>(1), 1)
        , fe(FE_Q<dim>(1), 1)
        , fe_vec(FE_Q<dim>(2), dim)
        , dof_handler_moist(triangulation)
        , dof_handler(triangulation)
        , dof_handler_vec(triangulation)
        , quadrature_formula(fe.degree + 1)
        , mpi_communicator(MPI_COMM_WORLD)
        , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
        , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
        , pcout(std::cout, this_mpi_process == 0)
    {}

    // Destructor
    template <int dim>
    TopLevel<dim>::~TopLevel()
    {
        dof_handler_vec.clear();
    }

    // ------------------------------------------------------------------------------------------
    // SNES functions - Function/Jacobian creation
    // ----------------------------------------------------------------------------------
    // Function used to create the FormFunction Input for the SNESSetFunction
    template <int dim>
    PetscErrorCode TopLevel<dim>::FormFunction(SNES snes, Vec x, Vec f, void* ctx)
    {
        const auto p_ctx = static_cast<TopLevel<dim>*>(ctx);
        //p_ctx->assemble_rhs_alpha(p_ctx->present_solution_alpha, p_ctx->system_rhs_alpha);
        p_ctx->assemble_system_alpha(p_ctx->present_solution_alpha, p_ctx->system_rhs_alpha);
        return 0;
    }

    // Function used to create the FormFunction Input for the SNESSetJacobian
    template <int dim>
    PetscErrorCode TopLevel<dim>::FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void* ctx)
    {
        const auto p_ctx = static_cast<TopLevel<dim>*>(ctx);
        //p_ctx->assemble_system_alpha(p_ctx->present_solution_alpha);
        p_ctx->system_matrix_alpha;

        return 0;
    }

    // ------------------------------------------------------------------------------------------
    // CREATE MESH
    // ------------------------------------------------------------------------------------------
    template <int dim>
    void TopLevel<dim>::create_coarse_grid()
    {
        Triangulation<dim> tria_1, tria_2;

        std::vector<unsigned int> sudd(dim);
        unsigned int sudd_x = 20, sudd_y = 150;
        sudd[0]=sudd_x;
        sudd[1]=sudd_y;

        std::vector<unsigned int> sudd2(dim);
        unsigned int sudd2_x = 20, sudd2_y = 150;
        sudd2[0]=sudd2_x;
        sudd2[1]=sudd2_y;

        GridGenerator::subdivided_hyper_rectangle(tria_1, sudd,
                                                 Point<dim>(0, 0),
        	                                     Point<dim>(L, H_int),
                                                           false);  

        GridGenerator::subdivided_hyper_rectangle(tria_2, sudd2,
                                                 Point<dim>(0, H_int),
        	                                     Point<dim>(L, H),
                                                           false);
                                                                  
        const double tolerance = 0;
        GridGenerator::merge_triangulations(tria_1, tria_2, triangulation, tolerance);
        
        for (auto cell : triangulation.active_cell_iterators())
        {
            if (cell->center()[1] < H_int)
                cell->set_material_id(1);
            else
                cell->set_material_id(2);
        }


        for (const auto& cell : triangulation.active_cell_iterators())
            for (const auto& face : cell->face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();
                    if (face_center[1] == 0)
                        face->set_boundary_id(1);
                    else if (face_center[1] == H)
                        face->set_boundary_id(3);
                }
        triangulation.refine_global(0);
    }

    template <int dim>
    void TopLevel<dim>::add_interface_constraints(const Triangulation<dim> &triangulation, std::vector<InterfaceFaceInfo<dim>> &interface_faces)
    {
        std::vector<InterfaceFaceInfo<dim>> local_interface_faces;

        // Identify interface faces and store their information
        for (auto cell : triangulation.active_cell_iterators())
        {
            for (unsigned int face_index = 0; face_index < GeometryInfo<dim>::faces_per_cell; ++face_index)
            {
                Point<dim> face_center = cell->face(face_index)->center();

                if (std::abs(face_center[1] -H_int) < 1e-12 )
                {
                    InterfaceFaceInfo<dim> info;
                    info.center = face_center;
                    info.cell_index = cell->index();
                    info.face_index = face_index;
                    info.material_id = cell->material_id();
                
                    info.coincident_face_index = 0;
                    info.coincident_cell_index = 0;
                    info.coincident_material_id = 0;

                    local_interface_faces.push_back(info);
                }
            }
        }

        // Match faces that belong to different subdomains but share the same position
        for (unsigned int i = 0; i < local_interface_faces.size(); ++i)
        {
            for (unsigned int j = i + 1; j < local_interface_faces.size(); ++j)
            {
                if ((local_interface_faces[i].center.distance(local_interface_faces[j].center) < 1e-12) &&
                    (local_interface_faces[i].material_id != local_interface_faces[j].material_id))
                {
                    // Update coincident face information
                    local_interface_faces[i].coincident_face_index = local_interface_faces[j].face_index;
                    local_interface_faces[i].coincident_cell_index = local_interface_faces[j].cell_index;
                    local_interface_faces[i].coincident_material_id = local_interface_faces[j].material_id;

                    local_interface_faces[j].coincident_face_index = local_interface_faces[i].face_index;
                    local_interface_faces[j].coincident_cell_index = local_interface_faces[i].cell_index;
                    local_interface_faces[j].coincident_material_id = local_interface_faces[i].material_id;
                }
            }
        }

        // Update the global vector with local interface faces
        interface_faces.insert(interface_faces.end(), local_interface_faces.begin(), local_interface_faces.end());
    }

    // ------------------------------------------------------------------------------------------
    // Iso Canvas
    // ------------------------------------------------------------------------------------------
    template <int dim>
    double TopLevel<dim>::iso_canvas(double rh)
    {
        if (rh>=95) rh=95;
        if (rh<=2.5) rh=2.5;
        double tmp;
        tmp = 9.86622319e-8 *std::pow(rh,5) + 
             -2.091092e-5   *std::pow(rh,4) +
              1.63714456e-3 *std::pow(rh,3) +
             -5.59527931e-2 *std::pow(rh,2) +
              1.30432613    *std::pow(rh,1) +
              7.19461645;
        return 1e-6*tmp;
    }

    template <int dim>
    double TopLevel<dim>::iso_canvas_inv(double m)
    {
        m *= 1e6;
        if (m >= 90) m=90;
        else if (m<=10.13) m=10.13;
        double tmp;
        tmp = 7.46093535e-6 *std::pow(m,4) +
             -1.58403906e-3 *std::pow(m,3) +
              9.97199328e-2 *std::pow(m,2) +
             -5.14983688e-1    *std::pow(m,1) +
             -1.11183970;
        return tmp;
    }

    // ------------------------------------------------------------------------------------------
    // Setup / Assemble / Solve
    // ------------------------------------------------------------------------------------------
    template <int dim>
    void TopLevel<dim>::setup_system(std::vector<InterfaceFaceInfo<dim>> &interface_faces)
    {
        // Distribute degrees of freedom
        dof_handler_moist.distribute_dofs(fe_moist);
        locally_owned_dofs_moist = dof_handler_moist.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler_moist, locally_relevant_dofs_moist);

        // Create and apply hanging node constraints
        hanging_node_constraints_moist.clear();
        DoFTools::make_hanging_node_constraints(dof_handler_moist, hanging_node_constraints_moist);
        hanging_node_constraints_moist.close();

        // Initialize the sparsity pattern
        DynamicSparsityPattern sparsity_pattern_moist(locally_relevant_dofs_moist);
        DoFTools::make_sparsity_pattern(dof_handler_moist,
                                        sparsity_pattern_moist,
                                        hanging_node_constraints_moist,
                                        /*keep constrained dofs*/ false);

        // Add interface constraints to the sparsity pattern
        for (auto &iface : interface_faces)
        {
            // Determine which cell and face indices are involved
            unsigned int cell_i_index = iface.cell_index;
            unsigned int face_i_index = iface.face_index;
            unsigned int coincident_cell_index = iface.coincident_cell_index;
            unsigned int coincident_face_index = iface.coincident_face_index;

            // Get the degrees of freedom on the faces of both cells
            std::vector<types::global_dof_index> dof_indices_face_i(fe_moist.dofs_per_face);
            std::vector<types::global_dof_index> dof_indices_face_j(fe_moist.dofs_per_face);

            // Access the cell and get the degrees of freedom for the face on cell_i
            auto cell_i = dof_handler_moist.begin_active();
            std::advance(cell_i, cell_i_index);
            cell_i->face(face_i_index)->get_dof_indices(dof_indices_face_i);

            // Access the coincident cell and get the degrees of freedom for the face on coincident_cell
            auto cell_j = dof_handler_moist.begin_active();
            std::advance(cell_j, coincident_cell_index);
            cell_j->face(coincident_face_index)->get_dof_indices(dof_indices_face_j);

            // Resize and create a map from face DOFs to cell DOFs for cell_i
            iface.face_dof_map_i.resize(fe_moist.dofs_per_face);
            create_face_dof_map_moist(cell_i, face_i_index, iface.face_dof_map_i);

            // Resize and create a map from face DOFs to cell DOFs for cell_j
            iface.face_dof_map_j.resize(fe_moist.dofs_per_face);
            create_face_dof_map_moist(cell_j, coincident_face_index, iface.face_dof_map_j);

            // Add interactions between the degrees of freedom to the sparsity pattern
            for (unsigned int m = 0; m < dof_indices_face_i.size(); ++m)
            {
                for (unsigned int n = 0; n < dof_indices_face_j.size(); ++n)
                {
                    // Add entries to the sparsity pattern
                    sparsity_pattern_moist.add(dof_indices_face_i[m], dof_indices_face_j[n]);
                    sparsity_pattern_moist.add(dof_indices_face_j[n], dof_indices_face_i[m]);
                }
            }
        }

        // Distribute the updated sparsity pattern
        SparsityTools::distribute_sparsity_pattern(sparsity_pattern_moist,
                                                   locally_owned_dofs_moist,
                                                   mpi_communicator,
                                                   locally_relevant_dofs_moist);

        // Create the final sparsity pattern and output it
        SparsityPattern final_sparsity_pattern;
        final_sparsity_pattern.copy_from(sparsity_pattern_moist);

        std::ofstream out("updated_sparsity_pattern_moist.svg");
        final_sparsity_pattern.print_svg(out);

        // Reinitialize the system matrix with the final sparsity pattern
        system_matrix.reinit(locally_owned_dofs_moist,
                            locally_owned_dofs_moist,
                            sparsity_pattern_moist,
                            mpi_communicator);

        // Reinitialize the right-hand side and solution vectors
        system_rhs.reinit(locally_owned_dofs_moist, mpi_communicator);
        solution_m.reinit(dof_handler_moist.n_dofs());
        solution_m_old.reinit(dof_handler_moist.n_dofs());
        newton_update_m.reinit(dof_handler_moist.n_dofs());

        // Setta i valori iniziali per porzioni diverse del dominio
        InitialCondition<dim> initial_condition(H_int, u02, u01);
        VectorTools::interpolate(dof_handler_moist, initial_condition, solution_m);
        VectorTools::interpolate(dof_handler_moist, initial_condition, solution_m_old);
   }

    // Function to create a map from face DOFs to cell DOFs
    template <int dim>
    void TopLevel<dim>::create_face_dof_map_moist(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            unsigned int face_index,
                                            std::vector<unsigned int> &face_dof_map)
    {
        // Get the DOF indices for the entire cell
        std::vector<types::global_dof_index> dof_indices_cell(fe_moist.dofs_per_cell);
        cell->get_dof_indices(dof_indices_cell);

        // Get the DOF indices for the face
        std::vector<types::global_dof_index> dof_indices_face(fe_moist.dofs_per_face);
        cell->face(face_index)->get_dof_indices(dof_indices_face);

        // Create the map: for each face DOF, find the corresponding cell DOF
        for (unsigned int i = 0; i < dof_indices_face.size(); ++i)
        {
            for (unsigned int j = 0; j < dof_indices_cell.size(); ++j)
            {
                if (dof_indices_face[i] == dof_indices_cell[j])
                {
                    face_dof_map[i] = j;  // Map face DOF to corresponding cell DOF
                    break;
                }
            }
        }
    }

    // Assemble system
    template <int dim>
    void TopLevel<dim>::assemble_system(const bool flag_iter)
    {
        system_rhs = 0;
        system_matrix = 0;
        
        // Parte per assemblaggio standard
        FEValues<dim> fe_values(fe_moist,
                                quadrature_formula,
                                update_values | update_gradients | 
                                update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe_moist.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> local_solution_m_old(n_q_points);  

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_moist.begin_active(),
						                               endc = dof_handler_moist.end();

        // Parte per assemblaggio interfaccia
        QGauss<dim-1> face_quadrature_formula(fe_moist.degree + 1);
        const unsigned int n_q_points_face = face_quadrature_formula.size();
        const unsigned int dofs_per_face = fe_moist.dofs_per_face;

        FEFaceValues<dim> fe_face_values_i(fe_moist,
                                           face_quadrature_formula,
                                           update_values  | update_quadrature_points |
                                           update_normal_vectors | update_JxW_values);

        FEFaceValues<dim> fe_face_values_j(fe_moist,
                                           face_quadrature_formula,
                                           update_values  | update_quadrature_points |
                                           update_normal_vectors | update_JxW_values);

        std::vector<double> phi_i(dofs_per_face);
        std::vector<double> phi_j(dofs_per_face);

        FullMatrix<double> local_interface_matrix(dofs_per_face * 2, dofs_per_face * 2);
        Vector<double>     local_interface_rhs(dofs_per_face * 2);

        std::vector<double> conc_along_interface_i(n_q_points_face);
        std::vector<double> conc_along_interface_j(n_q_points_face);
        std::vector<Tensor<1, dim>> interface_normal(n_q_points_face);

        for (unsigned int index = 0; cell!=endc; ++cell, ++index)
        {
            if (cell->is_locally_owned())
            {
                // Parte standard di assemblaggio
                cell_matrix = 0;
                cell_rhs = 0;

                fe_values.reinit(cell);               
                fe_values.get_function_values(solution_m_old, local_solution_m_old);
                
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    double u_prev = local_solution_m_old[q_point];
                    double diff_c;
                    if(cell->material_id()==1)
                    {
                        diff_c = d_coeff1;
                    } else if (cell->material_id()==2)
                    {
                        diff_c = d_coeff2;
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)         
                        {
                            cell_matrix(i,j) += (fe_values.shape_value(i, q_point) *
                                                 fe_values.shape_value(j, q_point)
                                                 +
                                                 real_time_step * diff_c *
                                                 fe_values.shape_grad(i, q_point)*
                                                 fe_values.shape_grad(j,q_point)
                                                ) * fe_values.JxW(q_point);
                        }
                        
                        // Al momento settato come problema lineare
                        cell_rhs(i) += u_prev * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
                    }
                }

                // CAPIRE SE QUESTO PU0' RIMANERE QUI O DEVE ESSERE POSTICIPATO
                cell->get_dof_indices(local_dof_indices);
                hanging_node_constraints_moist.distribute_local_to_global(cell_matrix,
                                                                    cell_rhs,
                                                                    local_dof_indices,
                                                                    system_matrix,
                                                                    system_rhs);

                // Check if the current cell is in the interface_faces list with the desired material_id
                for (const auto &iface : interface_faces)
                {
                    if (iface.cell_index == index && iface.material_id == 1)
                    {
                        // Access degrees of freedom for faces
                        std::vector<types::global_dof_index> dof_indices_face_i(dofs_per_face);
                        std::vector<types::global_dof_index> dof_indices_face_j(dofs_per_face);

                        // Get the degrees of freedom on the face
                        cell->face(iface.face_index)->get_dof_indices(dof_indices_face_i);

                        auto coincident_cell = dof_handler_moist.begin_active();
                        std::advance(coincident_cell, iface.coincident_cell_index);
                        coincident_cell->face(iface.coincident_face_index)->get_dof_indices(dof_indices_face_j);

                        // Reinitialize FEValues for both faces
                        fe_face_values_i.reinit(cell, iface.face_index);
                        fe_face_values_j.reinit(coincident_cell, iface.coincident_face_index);

                        local_interface_matrix = 0;
                        local_interface_rhs = 0;      

                        for (unsigned int q_point = 0; q_point < n_q_points_face; ++q_point)
                        {
                            const double JxW = fe_face_values_i.JxW(q_point);
        
                            fe_face_values_i.get_function_values(solution_m_old, conc_along_interface_i);
                            fe_face_values_j.get_function_values(solution_m_old, conc_along_interface_j);

                            double conc_discontinuity_along_interface;
                            conc_discontinuity_along_interface = conc_along_interface_i[q_point] - conc_along_interface_j[q_point];
                
                            double kk;
                            kk = iso_canvas(iso_canvas_inv(conc_along_interface_j[q_point]))/conc_along_interface_j[q_point];
                            double Mo_eq = (d_coeff2*conc_along_interface_j[q_point] + d_coeff1*conc_along_interface_i[q_point])/(d_coeff2+d_coeff1*kk);
                            double Mc_eq = (kk*d_coeff2*conc_along_interface_j[q_point] + kk*d_coeff1*conc_along_interface_i[q_point])/(d_coeff2+d_coeff1*kk);
                            
                            double C_jump = d_coeff1*(Mc_eq-conc_along_interface_i[q_point]);
                            double O_jump = d_coeff2*(conc_along_interface_j[q_point]-Mo_eq);

                            double jump;
                            jump = int_coeff*conc_discontinuity_along_interface;  // j = ic * j(z)
                            
                            // Loop to extract the tensor-valued shape functions at the quadrature point
                            for (unsigned int k = 0; k < dofs_per_face; ++k)
                            {                                                 
                                // Directly assign the tensor-valued shape functions
                                phi_i[k] = fe_face_values_i.shape_value(iface.face_dof_map_i[k], q_point);
                                phi_j[k] = fe_face_values_j.shape_value(iface.face_dof_map_j[k], q_point); 
                            }                        

                            for (unsigned int i = 0; i < dofs_per_face; ++i)
                            {
                                for (unsigned int j = 0; j < dofs_per_face; ++j)
                                {
                                    const double contrib_ii = (phi_i[i] * d_coeff1 * phi_i[j]) * JxW;
                                    const double contrib_ij = (phi_i[i] * d_coeff1 * phi_j[j]) * JxW;
                                    const double contrib_ji = (phi_j[i] * d_coeff2 * phi_i[j]) * JxW;
                                    const double contrib_jj = (phi_j[i] * d_coeff2 * phi_j[j]) * JxW;

                                    local_interface_matrix(i, j) += contrib_ii;
                                    local_interface_matrix(i, dofs_per_face + j) -= contrib_ij;
                                    local_interface_matrix(dofs_per_face + i, j) -= contrib_ji;
                                    local_interface_matrix(dofs_per_face + i, dofs_per_face + j) += contrib_jj;
                                }

                                // Cambio il segno a seconda di dove sono nell'interfaccia
                                if (real_time_step >= 2)
                                {
                                    local_interface_rhs(i) += phi_i[i] *C_jump*JxW;
                                    local_interface_rhs(i) -= phi_j[i] *O_jump*JxW;    
                                }
                                else
                                {
                                    local_interface_rhs(i) += phi_i[i] *jump*JxW;
                                    local_interface_rhs(i) -= phi_j[i] *jump*JxW;    
                                }
                            }
                        }

                        std::vector<types::global_dof_index> local_dof_indices_i(dofs_per_face);
                        std::vector<types::global_dof_index> local_dof_indices_j(dofs_per_face);

                        cell->face(iface.face_index)->get_dof_indices(local_dof_indices_i);
                        coincident_cell->face(iface.coincident_face_index)->get_dof_indices(local_dof_indices_j);

                        // Add contributions to the global matrix
                        for (unsigned int i = 0; i < dofs_per_face; ++i)
                        {
                            system_rhs(local_dof_indices_i[i]) += local_interface_rhs(i);
                            system_rhs(local_dof_indices_j[i]) += local_interface_rhs(dofs_per_face + i);
                            
                            for (unsigned int j = 0; j < dofs_per_face; ++j)
                            {
                            // Diagonal blocks (u_i * K * u_i and u_j * K * u_j)
                                system_matrix.add(local_dof_indices_i[i], local_dof_indices_i[j], local_interface_matrix(i, j));
                                system_matrix.add(local_dof_indices_j[i], local_dof_indices_j[j], local_interface_matrix(dofs_per_face + i, dofs_per_face + j));

                            // Off-diagonal blocks (u_i * K * u_j and u_j * K * u_i)
                                system_matrix.add(local_dof_indices_i[i], local_dof_indices_j[j], local_interface_matrix(i, dofs_per_face + j));
                                system_matrix.add(local_dof_indices_j[i], local_dof_indices_i[j], local_interface_matrix(dofs_per_face + i, j));
                            }
                        }
                    }
                }                
            }
        }  // LOOP OVER CELL
        
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);

        std::map<types::global_dof_index, double> boundary_values;

        if (flag_iter == true)
        {
            VectorTools::interpolate_boundary_values(dof_handler_moist,
                                                     1,
                                                     Functions::ConstantFunction<dim>(u01_bc),
                                                     boundary_values);

            VectorTools::interpolate_boundary_values(dof_handler_moist,
                                                     3,
                                                     Functions::ConstantFunction<dim>(u02_bc),
                                                     boundary_values);
        }
        else
        {
            VectorTools::interpolate_boundary_values(dof_handler_moist,
                                                     1,
                                                     Functions::ConstantFunction<dim>(0),
                                                     boundary_values);
        } 

        PETScWrappers::MPI::Vector tmp(locally_owned_dofs_moist, mpi_communicator);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, tmp, system_rhs, false);
        if (flag_iter == true)
            solution_m = tmp;  
    }

    // Solve diffusion problem
    template <int dim>
    void TopLevel<dim>::solve_linear_problem_moist(const bool flag_elastic_iter)
    {
        PETScWrappers::MPI::Vector distributed_solution_m(locally_owned_dofs_moist, mpi_communicator);
        SolverControl cn;
        PETScWrappers::SparseDirectMUMPS mumps(cn, mpi_communicator);

        if (flag_elastic_iter == true)
        {
            distributed_solution_m = solution_m;
            mumps.solve(system_matrix, distributed_solution_m, system_rhs);
            solution_m = distributed_solution_m;
            hanging_node_constraints_moist.distribute(solution_m);
        }
        else
        {
            distributed_solution_m = newton_update_m;
            mumps.solve(system_matrix, distributed_solution_m, system_rhs);
            newton_update_m = distributed_solution_m;
            hanging_node_constraints_moist.distribute(newton_update_m);
        }
    }

    // ------------------------------------------------------------------------------------------
    // ELASTICITY - Setup / Assemble / Solve
    // ------------------------------------------------------------------------------------------
    // Elasticity - Initialize vectors and matrix
    template <int dim>
    void TopLevel<dim>::setup_system_elas(std::vector<InterfaceFaceInfo<dim>> &interface_faces)
    {
        // Distribute degrees of freedom
        dof_handler_vec.distribute_dofs(fe_vec);
        locally_owned_dofs_vec = dof_handler_vec.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler_vec, locally_relevant_dofs_vec);

        // Create and apply hanging node constraints
        hanging_node_constraints_vec.clear();
        DoFTools::make_hanging_node_constraints(dof_handler_vec, hanging_node_constraints_vec);
        hanging_node_constraints_vec.close();

        // Initialize the sparsity pattern
        DynamicSparsityPattern sparsity_pattern_vec(locally_relevant_dofs_vec);
        DoFTools::make_sparsity_pattern(dof_handler_vec,
                                        sparsity_pattern_vec,
                                        hanging_node_constraints_vec,
                                        /*keep constrained dofs*/ false);

        // Add interface constraints to the sparsity pattern
        for (auto &iface : interface_faces)
        {
            // Determine which cell and face indices are involved
            unsigned int cell_i_index = iface.cell_index;
            unsigned int face_i_index = iface.face_index;
            unsigned int coincident_cell_index = iface.coincident_cell_index;
            unsigned int coincident_face_index = iface.coincident_face_index;

            // Get the degrees of freedom on the faces of both cells
            std::vector<types::global_dof_index> dof_indices_face_i(fe_vec.dofs_per_face);
            std::vector<types::global_dof_index> dof_indices_face_j(fe_vec.dofs_per_face);

            // Access the cell and get the degrees of freedom for the face on cell_i
            auto cell_i = dof_handler_vec.begin_active();
            std::advance(cell_i, cell_i_index);
            cell_i->face(face_i_index)->get_dof_indices(dof_indices_face_i);

            // Access the coincident cell and get the degrees of freedom for the face on coincident_cell
            auto cell_j = dof_handler_vec.begin_active();
            std::advance(cell_j, coincident_cell_index);
            cell_j->face(coincident_face_index)->get_dof_indices(dof_indices_face_j);

            // Resize and create a map from face DOFs to cell DOFs for cell_i
            iface.face_dof_map_i.resize(fe_vec.dofs_per_face);
            create_face_dof_map(cell_i, face_i_index, iface.face_dof_map_i);

            // Resize and create a map from face DOFs to cell DOFs for cell_j
            iface.face_dof_map_j.resize(fe_vec.dofs_per_face);
            create_face_dof_map(cell_j, coincident_face_index, iface.face_dof_map_j);        

            // Add interactions between the degrees of freedom to the sparsity pattern
            for (unsigned int m = 0; m < dof_indices_face_i.size(); ++m)
            {
                for (unsigned int n = 0; n < dof_indices_face_j.size(); ++n)
                {
                    // Add entries to the sparsity pattern
                    sparsity_pattern_vec.add(dof_indices_face_i[m], dof_indices_face_j[n]);
                    sparsity_pattern_vec.add(dof_indices_face_j[n], dof_indices_face_i[m]);
                }
            }
        }

        // Distribute the updated sparsity pattern
        SparsityTools::distribute_sparsity_pattern(sparsity_pattern_vec,
                                                locally_owned_dofs_vec,
                                                mpi_communicator,
                                                locally_relevant_dofs_vec);

        // Create the final sparsity pattern and output it
        SparsityPattern final_sparsity_pattern;
        final_sparsity_pattern.copy_from(sparsity_pattern_vec);

        std::ofstream out("updated_sparsity_pattern.svg");
        final_sparsity_pattern.print_svg(out);

        // Reinitialize the system matrix with the final sparsity pattern
        system_matrix_elas.reinit(locally_owned_dofs_vec,
                                locally_owned_dofs_vec,
                                sparsity_pattern_vec,
                                mpi_communicator);

        // Reinitialize the right-hand side and solution vectors
        system_rhs_elas.reinit(locally_owned_dofs_vec, mpi_communicator);
        solution_u.reinit(dof_handler_vec.n_dofs());
        newton_update_u.reinit(dof_handler_vec.n_dofs());

    /*     system_matrix_elas.reinit(locally_owned_dofs_vec,
                                locally_owned_dofs_vec,
                                updated_sparsity_pattern,
                                mpi_communicator); */
   }

    // Function to create a map from face DOFs to cell DOFs
    template <int dim>
    void TopLevel<dim>::create_face_dof_map(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            unsigned int face_index,
                                            std::vector<unsigned int> &face_dof_map)
    {
        // Get the DOF indices for the entire cell
        std::vector<types::global_dof_index> dof_indices_cell(fe_vec.dofs_per_cell);
        cell->get_dof_indices(dof_indices_cell);

        // Get the DOF indices for the face
        std::vector<types::global_dof_index> dof_indices_face(fe_vec.dofs_per_face);
        cell->face(face_index)->get_dof_indices(dof_indices_face);

        // Create the map: for each face DOF, find the corresponding cell DOF
        for (unsigned int i = 0; i < dof_indices_face.size(); ++i)
        {
            for (unsigned int j = 0; j < dof_indices_cell.size(); ++j)
            {
                if (dof_indices_face[i] == dof_indices_cell[j])
                {
                    face_dof_map[i] = j;  // Map face DOF to corresponding cell DOF
                    break;
                }
            }
        }
    }

    // Assemble system
    template <int dim>
    void TopLevel<dim>::assemble_system_elas(const bool flag_iter)
    {
        system_rhs_elas = 0;
        system_matrix_elas = 0;
        
        FEValues<dim> fe_values(fe_vec,
                                quadrature_formula,
                                update_values | update_gradients | 
                                update_quadrature_points | update_JxW_values);

        FEValues<dim> fe_values_alpha(fe,
                                      quadrature_formula,
                                      update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values);  
        
        const unsigned int dofs_per_cell = fe_vec.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
       
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<std::vector<Tensor<1, dim>>> previous_gradient(n_q_points,std::vector<Tensor<1,dim>>(dim));
        std::vector<double> local_solution_alpha(n_q_points);  

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_vec.begin_active(),
						                               endc = dof_handler_vec.end();

        typename DoFHandler<dim>::active_cell_iterator cell_alpha = dof_handler.begin_active();
        
        BodyForce<dim>              body_force; // Sono gia presenti nella formulazione ma sono settate pari a 0
        std::vector<Vector<double>> body_force_values(n_q_points, Vector<double>(dim));

//      Part for interface contribution

        QGauss<dim-1> face_quadrature_formula(fe_vec.degree + 1);
        const unsigned int n_q_points_face = face_quadrature_formula.size();
        const unsigned int dofs_per_face = fe_vec.dofs_per_face;

        FEFaceValues<dim> fe_face_values_i(fe_vec, face_quadrature_formula,
                                           update_values  | update_quadrature_points | update_normal_vectors | update_JxW_values);
        FEFaceValues<dim> fe_face_values_j(fe_vec, face_quadrature_formula,
                                           update_values  | update_quadrature_points | update_normal_vectors | update_JxW_values);

        SymmetricTensor<2, dim> K_int = 1e6 * unit_symmetric_tensor<dim>();
        std::vector<double>          phi_i (dofs_per_face);
        std::vector<double>          phi_j (dofs_per_face);

        FullMatrix<double> local_interface_matrix(dofs_per_face * 2, dofs_per_face * 2);
        Vector<double>     local_interface_rhs(dofs_per_face * 2);

        std::vector<Vector<double> >      displacement_along_interface_i(n_q_points_face, Vector<double>(dim));
        std::vector<Vector<double> >      displacement_along_interface_j(n_q_points_face, Vector<double>(dim));
        std::vector<Tensor<1, dim> >      interface_normal(n_q_points_face);
        std::vector<Point<dim> >          position(n_q_points_face);

        for (unsigned int index = 0; cell!=endc; ++cell, ++cell_alpha, ++index)
        {
            if (cell->is_locally_owned())
            {
                cell_matrix = 0;
                cell_rhs = 0;

                fe_values.reinit(cell);
                fe_values_alpha.reinit(cell_alpha);        

                fe_values.get_function_gradients(solution_u, previous_gradient);
                fe_values_alpha.get_function_values(solution_alpha, local_solution_alpha);
                
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    
                    const double g_alpha_gauss = g_alpha(local_solution_alpha[q_point]);
                    const SymmetricTensor<2,dim> eps_u = get_strain (previous_gradient[q_point]);
                    SymmetricTensor<4, dim>  C_Pos, C_Neg;

                    stiffness_matrix(C_Pos, C_Neg, eps_u, stress_strain_tensor, Young, Poisson);
                    
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const SymmetricTensor<2, dim>   eps_phi_i = get_strain(fe_values, i, q_point);
                        const SymmetricTensor<2, dim>   sigma_i =  (g_alpha_gauss + k_res) * C_Pos * eps_phi_i + C_Neg * eps_phi_i;    

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)         
                        {
                            const SymmetricTensor<2, dim>  eps_phi_j = get_strain(fe_values, j, q_point);
                            cell_matrix(i, j) +=    sigma_i*eps_phi_j * fe_values.JxW(q_point);
                        }

                        if (flag_iter == false) // RHS viene popolato solo dalla seconda iter, la prima iter tiro e genero imbalance
                        {
                            const SymmetricTensor<2, dim>  previous_sigma_u =  (g_alpha_gauss + k_res) *  C_Pos * eps_u +  C_Neg * eps_u; 
                            cell_rhs(i) -=previous_sigma_u* eps_phi_i * fe_values.JxW(q_point);
                        }
                    }
                }


//              CAPIRE SE QUESTO PU0' RIMANERE QUI O DEVE ESSERE POSTICIPATO
                cell->get_dof_indices(local_dof_indices);
                hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                                    cell_rhs,
                                                                    local_dof_indices,
                                                                    system_matrix_elas,
                                                                    system_rhs_elas);


            // Check if the current cell is in the interface_faces list with the desired material_id
            for (const auto &iface : interface_faces)
            {
                if (iface.cell_index == index && iface.material_id == 1)
                {                  
                    // Access degrees of freedom for faces


                    std::vector<types::global_dof_index> dof_indices_face_i(dofs_per_face);
                    std::vector<types::global_dof_index> dof_indices_face_j(dofs_per_face);

                    // Get the degrees of freedom on the face
                    cell->face(iface.face_index)->get_dof_indices(dof_indices_face_i);

                    auto coincident_cell = dof_handler_vec.begin_active();
                    std::advance(coincident_cell, iface.coincident_cell_index);
                    coincident_cell->face(iface.coincident_face_index)->get_dof_indices(dof_indices_face_j);

                    // Reinitialize FEValues for both faces
                    fe_face_values_i.reinit(cell, iface.face_index);
                    fe_face_values_j.reinit(coincident_cell, iface.coincident_face_index);

                    local_interface_matrix = 0;
                    local_interface_rhs = 0;      

                    for (unsigned int q_point = 0; q_point < n_q_points_face; ++q_point)
                    {
                        const double JxW = fe_face_values_i.JxW(q_point);
    
                        fe_face_values_i.get_function_values(solution_u,displacement_along_interface_i);
                        fe_face_values_j.get_function_values(solution_u,displacement_along_interface_j);
                        Tensor<1, dim> displacement_discontinuity_along_interface;
                        for (unsigned int i = 0; i < dim; ++i)
                            displacement_discontinuity_along_interface[i] = displacement_along_interface_i[q_point][i]
                                                                            -displacement_along_interface_j[q_point][i];
            
                        interface_normal= fe_face_values_i.get_normal_vectors();
                        position=fe_face_values_i.get_quadrature_points();

                        Tensor<2, dim> Rot;   // THIS WORK ONLY IN 2D...SEE STEP 18 For extension to 3D
                        // calculation of rotation
                        const double normaly = interface_normal[q_point][1];
                        const int sign_normalx = ((interface_normal[q_point][0] > 0) ? 1 : ((interface_normal[q_point][0] < 0) ? -1 : 0));
                        Rot[0][0] = normaly;
                        Rot[0][1] = -sign_normalx * std::sqrt(1 - normaly * normaly);
                        Rot[1][0] = sign_normalx * std::sqrt(1 - normaly * normaly);
                        Rot[1][1] = normaly;

                        Tensor<2, dim> Rot_transpose = transpose(Rot);

                        SymmetricTensor<2, dim> RT_K_int_R;
                        RT_K_int_R =symmetrize(Rot_transpose * static_cast<Tensor<2, dim>>(K_int) * Rot);
                        Tensor<1,dim> traction;
                        traction = - RT_K_int_R * displacement_discontinuity_along_interface;  // t = R^t * t(z) check if the correct expression is traction = - R^t * K_int * displacement_discontinuity_along_interface
 
                        // Loop to extract the tensor-valued shape functions at the quadrature point
                        for (unsigned int k = 0; k < dofs_per_face; ++k)
                        {                                                 
                            // Directly assign the tensor-valued shape functions
                            phi_i[k] = fe_face_values_i.shape_value(iface.face_dof_map_i[k], q_point);
                            phi_j[k] = fe_face_values_j.shape_value(iface.face_dof_map_j[k], q_point);                    
                        }                        

                        for (unsigned int i = 0; i < dofs_per_face; ++i)
                        {
                            const unsigned int comp_i = fe_vec.system_to_component_index(i).first;                        
                            
                            for (unsigned int j = 0; j < dofs_per_face; ++j)
                            {
                                const unsigned int comp_j = fe_vec.system_to_component_index(j).first;

                                const double contrib_ii = (phi_i[i] * RT_K_int_R[comp_i][comp_j] * phi_i[j]) * JxW;
                                const double contrib_ij = (phi_i[i] * RT_K_int_R[comp_i][comp_j] * phi_j[j]) * JxW;
                                const double contrib_ji = (phi_j[i] * RT_K_int_R[comp_i][comp_j] * phi_i[j]) * JxW;
                                const double contrib_jj = (phi_j[i] * RT_K_int_R[comp_i][comp_j] * phi_j[j]) * JxW;

                                local_interface_matrix(i, j) += contrib_ii;
                                local_interface_matrix(i, dofs_per_face + j) -= contrib_ij;
                                local_interface_matrix(dofs_per_face + i, j) -= contrib_ji;
                                local_interface_matrix(dofs_per_face + i, dofs_per_face + j) += contrib_jj; 
                            }
                        
                        if (flag_iter == false) // RHS viene popolato solo dalla seconda iter, la prima iter tiro e genero imbalance
                        {
                            local_interface_rhs(i) +=phi_i[i]*traction[comp_i]*JxW;
                            local_interface_rhs(dofs_per_face + i) -=phi_j[i]*traction[comp_i]*JxW;          
                        }

                        }
                    }

                    // Get the degrees of freedom indices for both cells (face_i and face_j)
                    std::vector<types::global_dof_index> local_dof_indices_i(dofs_per_face);
                    std::vector<types::global_dof_index> local_dof_indices_j(dofs_per_face);

                    cell->face(iface.face_index)->get_dof_indices(local_dof_indices_i);
                    coincident_cell->face(iface.coincident_face_index)->get_dof_indices(local_dof_indices_j);

                    // Add contributions to the global matrix
                    for (unsigned int i = 0; i < dofs_per_face; ++i)
                    {
                        system_rhs_elas(local_dof_indices_i[i]) +=local_interface_rhs(i);
                        system_rhs_elas(local_dof_indices_j[i]) +=local_interface_rhs(dofs_per_face + i);
                        
                        for (unsigned int j = 0; j < dofs_per_face; ++j)
                        {
                        // Diagonal blocks (u_i * K * u_i and u_j * K * u_j)
                            system_matrix_elas.add(local_dof_indices_i[i], local_dof_indices_i[j], local_interface_matrix(i, j));
                            system_matrix_elas.add(local_dof_indices_j[i], local_dof_indices_j[j], local_interface_matrix(dofs_per_face + i, dofs_per_face + j));

                        // Off-diagonal blocks (u_i * K * u_j and u_j * K * u_i)
                            system_matrix_elas.add(local_dof_indices_i[i], local_dof_indices_j[j], local_interface_matrix(i, dofs_per_face + j));
                            system_matrix_elas.add(local_dof_indices_j[i], local_dof_indices_i[j], local_interface_matrix(dofs_per_face + i, j));
                        }
                    }
                }
            }
            }
        }  // LOOP OVER CELL
        system_matrix_elas.compress(VectorOperation::add);
        system_rhs_elas.compress(VectorOperation::add);

        std::vector<bool> BC_x = {true, false};
        std::vector<bool> BC_y = {true, true};

        std::map<types::global_dof_index, double> boundary_values;

        Point<dim> point_one, point_two, point_three;
        point_one(0) = 10000.;
        point_one(1) = 0.;
        point_two(0) = 0.;
        point_two(1) = H;
        point_three(0) = 1000.;
        point_three(1) = 1000.;

        cell = dof_handler_vec.begin_active(),
        endc = dof_handler_vec.end();

        bool evaluation_point_found = false;
        for (; (cell!=endc) && !evaluation_point_found; ++cell)
        // for (;cell!=endc;++cell)
        {
            if (cell->is_locally_owned())
            {
                for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex)
                {
                    if (cell->vertex(vertex).distance (point_one) < cell->diameter() * 1e-12)
                    {
                        boundary_values[cell->vertex_dof_index(vertex,0)]=0.;
                        // boundary_values[cell->vertex_dof_index(vertex,1)]=-1.*what_time();
                        evaluation_point_found = true;
                        // break;
                    };
                    if (cell->vertex(vertex).distance (point_two) <	cell->diameter() * 1e-12)
                    {
                        boundary_values[cell->vertex_dof_index(vertex,0)]=0.;
                        evaluation_point_found = true;
                        // break;
                    };
                    if (cell->vertex(vertex).distance (point_three) <	cell->diameter() * 1e-12)
                    {
                	    // boundary_values[cell->vertex_dof_index(vertex,0)]=0;
                        boundary_values[cell->vertex_dof_index(vertex,1)]=0.;
                        evaluation_point_found = true;
                	    // break;
                    };
                };
            }
        }

        VectorTools::interpolate_boundary_values(dof_handler_vec,
                                                 1,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 boundary_values,
                                                 BC_y);

        if (flag_iter == true)
        {
            VectorTools::interpolate_boundary_values(dof_handler_vec,
                                                 3,
                                                 IncrementalBoundaryValues<dim>(0.001*real_time, 0.001*real_time_step),
                                                 boundary_values,
                                                 BC_y);
        }
        else
        {
            VectorTools::interpolate_boundary_values(dof_handler_vec,
                                                 3,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 boundary_values,
                                                 BC_y);
        } 

        PETScWrappers::MPI::Vector tmp(locally_owned_dofs_vec, mpi_communicator);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix_elas, tmp, system_rhs_elas, false);
        if (flag_iter == true)
            solution_u = tmp;   
    }

    // Solve elasticity problem
    template <int dim>
    void TopLevel<dim>::solve_linear_problem(const bool flag_elastic_iter)
    {
        PETScWrappers::MPI::Vector distributed_solution_u(locally_owned_dofs_vec, mpi_communicator);
        SolverControl cn;
        PETScWrappers::SparseDirectMUMPS mumps(cn, mpi_communicator);
        // mumps.set_symmetric_mode(true);

        if (flag_elastic_iter == true)
        {
            distributed_solution_u = solution_u;
            mumps.solve(system_matrix_elas, distributed_solution_u, system_rhs_elas);
            solution_u = distributed_solution_u;
            hanging_node_constraints.distribute(solution_u);
        }
        else
        {
            distributed_solution_u = newton_update_u;
            mumps.solve(system_matrix_elas, distributed_solution_u, system_rhs_elas);
            newton_update_u = distributed_solution_u;
            hanging_node_constraints.distribute(newton_update_u);
        }
    }

    // ------------------------------------------------------------------------------------------
    // DAMAGE - Setup / Assemble / Solve
    // ------------------------------------------------------------------------------------------
    // Setup system damage
    template <int dim>
    void TopLevel<dim>::setup_system_alpha()
    {
        dof_handler.distribute_dofs(fe);
        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);
        hanging_node_constraints.close();
        
        DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        sparsity_pattern,
                                        hanging_node_constraints,
                                        false);
        SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                                   locally_owned_dofs,
                                                   mpi_communicator,
                                                   locally_relevant_dofs);

        system_matrix_alpha.reinit(locally_owned_dofs,
                                   locally_owned_dofs,
                                   sparsity_pattern,
                                   mpi_communicator);
        system_rhs_alpha.reinit(locally_owned_dofs, mpi_communicator);
        present_solution_alpha.reinit(locally_owned_dofs, mpi_communicator);

        solution_alpha.reinit(dof_handler.n_dofs());
        solution_alpha_previous_step.reinit(dof_handler.n_dofs());

        alpha_lb.reinit(locally_owned_dofs, mpi_communicator);
        alpha_ub.reinit(locally_owned_dofs, mpi_communicator);
    }

    // Assemble system damage
    template <int dim>
    void TopLevel<dim>::assemble_system_alpha(PETScWrappers::MPI::Vector &present_solution_alpha, PETScWrappers::MPI::Vector &system_rhs_alpha)
    {
        // Copio il vettore su un vettore dummy se no mi da errore in mpi -> uso solo per ottenere valori func e grad nei punti di gauss
        Vector<double> dummy_alpha(present_solution_alpha);
        system_matrix_alpha = 0;
        system_rhs_alpha = 0;
        
        FEValues<dim> fe_values_alpha(fe,
                                      quadrature_formula,
                                      update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values);
        
        FEValues<dim> fe_values(fe_vec,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> local_solution_alpha(n_q_points);
        std::vector<double> previous_step_local_solution_alpha(n_q_points);
        std::vector<Tensor<1, dim>> previous_grad_local_solution_alpha (n_q_points);
        std::vector< std::vector<Tensor<1, dim>>> previous_gradient (n_q_points, std::vector<Tensor<1,dim>>(dim));

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_vec.begin_active();

        typename DoFHandler<dim>::active_cell_iterator cell_alpha = dof_handler.begin_active(),
					    	                           endc_alpha = dof_handler.end();

        for (unsigned int index = 0; cell_alpha!=endc_alpha; ++cell, ++cell_alpha, ++index)
        {
            if (cell_alpha->is_locally_owned())
            {
                cell_matrix = 0;
                cell_rhs = 0;

                fe_values_alpha.reinit(cell_alpha);
                fe_values.reinit (cell);

                fe_values.get_function_gradients(solution_u,previous_gradient);            
            
                fe_values_alpha.get_function_values(dummy_alpha,local_solution_alpha);
                fe_values_alpha.get_function_values(solution_alpha_previous_step,previous_step_local_solution_alpha);
                fe_values_alpha.get_function_gradients (dummy_alpha,previous_grad_local_solution_alpha);

                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                    const SymmetricTensor<2,dim> eps_u = get_strain (previous_gradient[q_point]);
                    double alpha_diff=local_solution_alpha[q_point]-previous_step_local_solution_alpha[q_point];

                    SymmetricTensor<4, dim>  C_Pos, C_Neg;
                    stiffness_matrix(C_Pos, C_Neg, eps_u, stress_strain_tensor, Young, Poisson);                           
              
                    double elastic_source=0.;
                    elastic_source = 0.5* eps_u*C_Pos*eps_u;    

                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            cell_matrix(i,j)+= (fe_values_alpha.shape_grad(i,q_point) *
                                                fe_values_alpha.shape_grad(j,q_point) *
                                                            (2.*Gc*ell/c_w)           
                                                +
                                                fe_values_alpha.shape_value(i,q_point)  *
                                                fe_values_alpha.shape_value(j,q_point)  *
                                                (Gc/(ell*c_w)*w_second_alpha()+
                                                g_second_alpha()*elastic_source))   
                                                * fe_values_alpha.JxW(q_point);
                        }
                        
                        cell_rhs(i) +=  (g_prime_alpha(local_solution_alpha[q_point])*elastic_source+
                                        w_prime_alpha(local_solution_alpha[q_point])*Gc/(ell*c_w))  *   
                                        fe_values_alpha.shape_value(i,q_point) *
                                        fe_values_alpha.JxW(q_point);

                        cell_rhs(i) +=   (previous_grad_local_solution_alpha[q_point]*2.*Gc*ell/c_w) *
                                        fe_values_alpha.shape_grad(i,q_point) *
                                        fe_values_alpha.JxW(q_point);
                    }                                                   
                }
                cell_alpha->get_dof_indices(local_dof_indices);
                hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                                    cell_rhs,
                                                                    local_dof_indices,
                                                                    system_matrix_alpha,
                                                                    system_rhs_alpha);
            }
        }
        system_matrix_alpha.compress(VectorOperation::add);
        system_rhs_alpha.compress(VectorOperation::add);
        std::map<types::global_dof_index, double> boundary_values;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                1,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                3,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);

        // Qui riuso il vettore petsc MPI cosi da applicarci le BC
        MatrixTools::apply_boundary_values(boundary_values, system_matrix_alpha, present_solution_alpha, system_rhs_alpha, false);
    }

    // Assemble rhs damage
    template <int dim>
    void TopLevel<dim>::assemble_rhs_alpha(PETScWrappers::MPI::Vector &present_solution_alpha, PETScWrappers::MPI::Vector &system_rhs_alpha)
    {
        // Copio il vettore su un vettore dummy se no mi da errore in mpi -> uso solo per ottenere valori func e grad nei punti di gauss
        Vector<double> dummy_alpha(present_solution_alpha);
        system_rhs_alpha = 0;
        
        FEValues<dim> fe_values_alpha(fe,
                                      quadrature_formula,
                                      update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values);
        
        FEValues<dim> fe_values(fe_vec,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        
        Vector<double>     cell_rhs(dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> local_solution_alpha(n_q_points);
        std::vector<double> previous_step_local_solution_alpha(n_q_points);
        std::vector<Tensor<1, dim>> previous_grad_local_solution_alpha (n_q_points);
        std::vector< std::vector<Tensor<1, dim>>> previous_gradient (n_q_points, std::vector<Tensor<1,dim>>(dim));

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_vec.begin_active();

        typename DoFHandler<dim>::active_cell_iterator cell_alpha = dof_handler.begin_active(),
					    	                           endc_alpha = dof_handler.end();

        for (unsigned int index = 0; cell_alpha!=endc_alpha; ++cell, ++cell_alpha, ++index)
        {
            if (cell_alpha->is_locally_owned())
            {
                cell_rhs = 0;
            
                fe_values_alpha.reinit(cell_alpha);
                fe_values.reinit (cell);

                fe_values.get_function_gradients(solution_u,previous_gradient);            
            
                fe_values_alpha.get_function_values(dummy_alpha, local_solution_alpha);
                fe_values_alpha.get_function_values(solution_alpha_previous_step, previous_step_local_solution_alpha);
                fe_values_alpha.get_function_gradients (dummy_alpha, previous_grad_local_solution_alpha);

                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                    const SymmetricTensor<2,dim> eps_u = get_strain (previous_gradient[q_point]);
                    double alpha_diff=local_solution_alpha[q_point]-previous_step_local_solution_alpha[q_point];

                    SymmetricTensor<4, dim>  C_Pos, C_Neg;
                    stiffness_matrix(C_Pos, C_Neg, eps_u, stress_strain_tensor, Young, Poisson);                           
              
                    double elastic_source=0.;
                    elastic_source = 0.5* eps_u*C_Pos*eps_u;    

                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        cell_rhs(i) +=  (g_prime_alpha(local_solution_alpha[q_point])*elastic_source+
                                        w_prime_alpha(local_solution_alpha[q_point])*Gc/(ell*c_w))  *   
                                        fe_values_alpha.shape_value(i,q_point) *
                                        fe_values_alpha.JxW(q_point);

                        cell_rhs(i) +=   (previous_grad_local_solution_alpha[q_point]*2.*Gc*ell/c_w) *
                                        fe_values_alpha.shape_grad(i,q_point) *
                                        fe_values_alpha.JxW(q_point);
                    }                                                     
                }
                cell_alpha->get_dof_indices(local_dof_indices);
                hanging_node_constraints.distribute_local_to_global(cell_rhs,
                                                                    local_dof_indices,
                                                                    system_rhs_alpha);
            }
        }
        system_rhs_alpha.compress(VectorOperation::add);
        std::map<types::global_dof_index, double> boundary_values;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                1,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                3,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);
        
        // Qui riuso il vettore petsc MPI cosi da applicarci le BC
        MatrixTools::apply_boundary_values(boundary_values, system_matrix_alpha, present_solution_alpha, system_rhs_alpha, false);
    }

    // ------------------------------------------------------------------------------------------
    // OUTPUT RESULTS
    // ------------------------------------------------------------------------------------------
    // Moist
    template <int dim>
    void TopLevel<dim>::output_results(const unsigned int cycle) const
    {
        const Vector<double> localized_solution(solution_m);
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler_moist);

        std::vector<std::string> solution_names;
        solution_names.push_back ("m");
        data_out.add_data_vector(localized_solution,
                                 solution_names);
        data_out.build_patches();

        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record("./", "solution_m", cycle, mpi_communicator, 4);
        if (this_mpi_process == 0)
        {
            static std::vector<std::pair<double, std::string>> times_and_names;
            times_and_names.push_back(std::pair<double, std::string>(cycle, pvtu_filename));
            std::ofstream pvd_output("./solution_m.pvd");
            DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    // Elasticity
    template <int dim>
    void TopLevel<dim>::output_results_elas(const unsigned int cycle) const
    {
        const Vector<double> localized_solution(solution_u);
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler_vec);

        std::vector<std::string> solution_names(dim, "u");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> 
                    interpretation(dim,
                                   DataComponentInterpretation::component_is_part_of_vector);
   
        data_out.add_data_vector(localized_solution,
                                   solution_names,
                                   DataOut<dim>::type_dof_data,
                                   interpretation);
        data_out.build_patches();

        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record("./", "solution_u", cycle, mpi_communicator, 4);
        if (this_mpi_process == 0)
        {
            static std::vector<std::pair<double, std::string>> times_and_names;
            times_and_names.push_back(std::pair<double, std::string>(cycle, pvtu_filename));
            std::ofstream pvd_output("./solution_u.pvd");
            DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }
 
    // Damage
    template <int dim>
    void TopLevel<dim>::output_results_alpha(const unsigned int cycle) const
    {
        const Vector<double> localized_solution(solution_alpha);
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        std::vector<std::string> solution_names;
        solution_names.push_back ("alpha");
        data_out.add_data_vector(localized_solution,
                                solution_names);
        data_out.build_patches();

        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record("./", "solution_alpha", cycle, mpi_communicator, 4);
        if (this_mpi_process == 0)
        {
            static std::vector<std::pair<double, std::string>> times_and_names;
            times_and_names.push_back(std::pair<double, std::string>(cycle, pvtu_filename));
            std::ofstream pvd_output("./solution_alpha.pvd");
            DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    // ------------------------------------------------------------------------------------------
    // RUN
    // ------------------------------------------------------------------------------------------
    template <int dim>
    void TopLevel<dim>::run()
    {
        set_data ();
        do_initial_timestep();

        std::ofstream output_text("output.txt", std::ios::out);
        output_text << " il dato riportato e' il seguente:  bulk - surface - total energy" << std::endl;

        while (real_time < real_time_final)
        {
            do_timestep();

            // Output
            output_results(real_timestep_no);
            output_results_elas(real_timestep_no);
            output_results_alpha(real_timestep_no);            
            
            double bulk_energy=0.;
            double surface_energy=0.;      
            energy (bulk_energy, surface_energy);
        }
    }

    // -------------------------------------------------------------------------------------------
    // Timestep solve - Elasticity / Damage
    // -------------------------------------------------------------------------------------------
    template <int dim>
    void TopLevel<dim>::do_initial_timestep()
    {
        real_time += real_time_step;
        ++real_timestep_no;
        pcout << "Timestep " << real_timestep_no << " at time " << real_time << std::endl;

        create_coarse_grid();
        add_interface_constraints(triangulation, interface_faces_moist);
        add_interface_constraints(triangulation, interface_faces);

            // Access the interface faces here
        std::cout << "Interface faces count: " << interface_faces.size() << std::endl;
        for (const auto &info : interface_faces)
        {
            std::cout << "Face Center: ";
            for (unsigned int d = 0; d < dim; ++d)
                std::cout << info.center[d] << " ";
            std::cout << std::endl;
            std::cout << "Cell Index: " << info.cell_index << std::endl;
            std::cout << "Face Index: " << info.face_index << std::endl;
            std::cout << "Material ID: " << info.material_id << std::endl;
            std::cout << "Coincident Face Index: " << info.coincident_face_index << std::endl;
            std::cout << "Coincident Cell Index: " << info.coincident_cell_index << std::endl;
            std::cout << "Coincident Material ID: " << info.coincident_material_id << std::endl;
        }

        pcout << "    Number of active cells:       "
              << triangulation.n_active_cells() << " (by partition:";
              for (unsigned int p = 0; p < n_mpi_processes; ++p)
                    pcout << (p == 0 ? ' ' : '+') << (GridTools::count_cells_with_subdomain_association(triangulation, p));
        pcout << ")" << std::endl;

        setup_system(interface_faces_moist);
        setup_system_elas(interface_faces);

        setup_system_alpha();
        set_initial_values_alpha(alpha0);

        pcout << "    Number of degrees of freedom for the elastic part: " << dof_handler_vec.n_dofs() << " (by partition:";
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
            pcout << (p == 0 ? ' ' : '+') << (DoFTools::count_dofs_with_subdomain_association(dof_handler_vec, p));
        pcout << ")" << std::endl;

        pcout << "    Number of degrees of freedom for the damage part: " << dof_handler.n_dofs() << " (by partition:";
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
            pcout << (p == 0 ? ' ' : '+') << (DoFTools::count_dofs_with_subdomain_association(dof_handler, p));
        pcout << ")" << std::endl;

        solve_timestep();
        
        real_time -= real_time_step;
        --real_timestep_no;
    }

    // Timestep function
    template <int dim>
    void TopLevel<dim>::do_timestep()
    {
        real_time += real_time_step;
        ++real_timestep_no;

        pcout << " --- --- --- --- --- " << std::endl;
        pcout << "Timestep " << real_timestep_no << " at time " << real_time << std::endl;

        solve_timestep();
    }

    template <int dim>
    void TopLevel<dim>::solve_timestep()
    {
        bool solve_step_moist;

        // Diffusion
        pcout << "      Solving Diffusion problem..." << std::endl;
        solve_step_moist = true;
        assemble_system(solve_step_moist);
        solve_linear_problem_moist(solve_step_moist);

        solution_m_old = solution_m;

        // Control variables declaration
        double error_alternate = 1.0;
        double  error_elastic;
        const double error_toll =1.e-4;
        const double error_toll_elastic=1.e-6;
        unsigned int iter_counter_am = 0;
        unsigned int iter_elastic;
        const unsigned int max_iteration = 2000;
        const unsigned int max_iteration_elastic = 500;
        bool solve_step;

        while (error_alternate > error_toll && iter_counter_am <max_iteration)
        {
            // Elasticity
            pcout << "      Solving Elastic problem..." << std::endl;
            
            iter_elastic = 0;            
            solve_step = true;

            assemble_system_elas(solve_step);
            solve_linear_problem(solve_step);

            do
            {
                ++iter_elastic;
                solve_step = false;

                assemble_system_elas(solve_step);
                solve_linear_problem(solve_step);
                
                solution_u.add(1.0,newton_update_u);
                error_elastic = newton_update_u.l2_norm();

            } while (error_elastic > error_toll_elastic && iter_elastic <max_iteration_elastic);

            pcout << "          Iterations: " << iter_elastic << std::endl;
            pcout << "          --- Error_on_Newton_update_u: "   << error_elastic << std::endl; 
            pcout << "          --- rhs_error_elastic: "   <<  system_rhs_elas.l2_norm()  <<  std::endl; 

            // Damage
            pcout << "      Solving Damage problem..." << std::endl;
            solution_alpha = present_solution_alpha;  // Copy MPI vector over the normal vector for output e convergence check
            Vector<double> temp_alpha;
            temp_alpha = solution_alpha;

            // SNES for the damage problem
            SNES snes;
            PetscErrorCode ierr;
            ierr = SNESCreate(mpi_communicator, &snes);
            ierr = SNESSetType(snes, SNESVINEWTONRSLS);
            ierr = SNESSetFunction(snes, system_rhs_alpha, FormFunction, this);
            ierr = SNESSetJacobian(snes, system_matrix_alpha, system_matrix_alpha, FormJacobian, this);
            ierr = SNESVISetVariableBounds(snes, alpha_lb, alpha_ub);
            ierr = SNESSetFromOptions(snes);
            
            ierr = SNESSolve(snes, nullptr, present_solution_alpha);

            //PetscInt its;
            //SNESGetIterationNumber(snes, &its);
            //PetscPrintf(MPI_COMM_WORLD, "Number of SNES iterations = %D\n", its);

            ierr = SNESDestroy(&snes);
            // End SNES    

            solution_alpha = present_solution_alpha; // Copy MPI vector over the normal vector for output e convergence check
            temp_alpha -=solution_alpha;
            error_alternate =  temp_alpha.linfty_norm();
            iter_counter_am++;
            
            pcout << " Number of iteration: " << iter_counter_am << std::endl;
            pcout << " Error_on_alpha:  " << error_alternate << std::endl;
            pcout << " Alpha_max:  " << solution_alpha.linfty_norm() << std::endl;
        }                        
        solution_alpha_previous_step = present_solution_alpha;
        alpha_lb = present_solution_alpha;
    }

    // ------------------------------------------------------------------------------------------
    // ENERGY EVALUATION
    // ------------------------------------------------------------------------------------------
    template <int dim>
    void TopLevel<dim>::energy (double &bulk_energy, double &surface_energy)
    {
        FEValues<dim> fe_values(fe_vec,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

        FEValues<dim> fe_values_alpha(fe,
                                      quadrature_formula,
                                      update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values);
    
        const unsigned int n_q_points = quadrature_formula.size();
       
        std::vector<std::vector<Tensor< 1, dim >>> previous_gradient(n_q_points,std::vector<Tensor<1,dim> >(dim));
        std::vector<double> local_solution_alpha(n_q_points);
        std::vector<Tensor< 1, dim > > local_solution_grad_alpha (n_q_points);

        double el_en, ph_el;          

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_vec.begin_active(),
						                               endc = dof_handler_vec.end();

        typename DoFHandler<dim>::active_cell_iterator cell_alpha = dof_handler.begin_active();
        
        for (; cell!=endc; ++cell, ++cell_alpha)
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                fe_values_alpha.reinit (cell_alpha);        
                fe_values_alpha.get_function_values(solution_alpha,local_solution_alpha);
                fe_values_alpha.get_function_gradients (solution_alpha,local_solution_grad_alpha);          
                fe_values.get_function_gradients(solution_u,previous_gradient);                         
                
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    SymmetricTensor<4, dim>  C_Pos, C_Neg;            
                    const SymmetricTensor<2,dim> eps_u = get_strain (previous_gradient[q_point]);                                
                    double grad_alpha_square=local_solution_grad_alpha[q_point]*local_solution_grad_alpha[q_point];
                    stiffness_matrix(C_Pos, C_Neg, eps_u, stress_strain_tensor, Young, Poisson);
                   
                    double elastic_energy_density =  0.5*(g_alpha(local_solution_alpha[q_point])+k_res)*eps_u*C_Pos*eps_u + 0.5*eps_u*C_Neg*eps_u; 
                    el_en +=elastic_energy_density*fe_values.JxW(q_point); 
                    ph_el += (Gc/c_w)*(ell*grad_alpha_square+w_alpha(local_solution_alpha[q_point])/ell)*fe_values.JxW(q_point);                        

                }              
            }
        bulk_energy = Utilities::MPI::sum(el_en,mpi_communicator);
        surface_energy = Utilities::MPI::sum(ph_el,mpi_communicator);    
    }
} // namespace phasefield

// ------------------------------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    try
    {
        using namespace dealii;
        using namespace phasefield;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        TopLevel<2> elastic_problem;
        elastic_problem.run();
    }
    catch (std::exception& exc)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    return 0;
}