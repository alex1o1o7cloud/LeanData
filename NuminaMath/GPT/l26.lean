import Mathlib
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSeq
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Identities
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combination
import Mathlib.Combinatorics.Combinations
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Pi.Algebra
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Probability
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Logic.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic

namespace solve_system_unique_solution_l26_26262

theorem solve_system_unique_solution:
  ∃! (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ x = 57 / 31 ∧ y = 97 / 31 := by
  sorry

end solve_system_unique_solution_l26_26262


namespace shape_is_cone_l26_26269

-- Define the spherical coordinates
structure spherical_coordinates :=
  (ρ : ℝ) (θ : ℝ) (φ : ℝ)

-- Define the constant c
constant c : ℝ

-- Define the condition: the shape described by the equation φ = c
def shape_described : spherical_coordinates → Prop := 
  λ coords, coords.φ = c

-- Define the cone
structure cone :=
  (vertex : spherical_coordinates) (axis : ℝ) (angle : ℝ)

-- State the theorem
theorem shape_is_cone (coords : spherical_coordinates) (h : shape_described coords) : 
  ∃ (cone_shape : cone), true :=
sorry -- Proof to be done

end shape_is_cone_l26_26269


namespace cos_Y_of_triangle_90_sin_Y_eq_l26_26379

variables {X Y Z : Type}
variables (XY XZ YZ : ℝ)
variables (k : ℝ) 

theorem cos_Y_of_triangle_90_sin_Y_eq :
  ∠X = 90° → sin Y = 3 / 5 → cos Y = 4 / 5 :=
by
  sorry

end cos_Y_of_triangle_90_sin_Y_eq_l26_26379


namespace segments_intersect_at_single_point_l26_26501

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Tetrahedron :=
  (A B C D : Point3D)

def midpoint (P Q : Point3D) : Point3D :=
  { x := (P.x + Q.x) / 2,
    y := (P.y + Q.y) / 2,
    z := (P.z + Q.z) / 2 }

theorem segments_intersect_at_single_point
  (tet : Tetrahedron)
  (M_AB := midpoint tet.A tet.B)
  (M_CD := midpoint tet.C tet.D)
  (M_AC := midpoint tet.A tet.C)
  (M_BD := midpoint tet.B tet.D)
  (M_AD := midpoint tet.A tet.D)
  (M_BC := midpoint tet.B tet.C) :
  ∃ P : Point3D, 
    ∀ M N : Point3D, ∃ t : ℝ, P = { x := M.x + t * (N.x - M.x),
                                      y := M.y + t * (N.y - M.y),
                                      z := M.z + t * (N.z - M.z) } :=
sorry

end segments_intersect_at_single_point_l26_26501


namespace solve_for_z_l26_26711

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26711


namespace james_carrot_sticks_l26_26383

theorem james_carrot_sticks (total_carrots : ℕ) (after_dinner_carrots : ℕ) (before_dinner_carrots : ℕ) 
  (h1 : total_carrots = 37) (h2 : after_dinner_carrots = 15) :
  before_dinner_carrots = total_carrots - after_dinner_carrots :=
by
suffices h : 37 - 15 = 22 by
  rw [← h1, ← h2]
  exact h
apply rfl

end james_carrot_sticks_l26_26383


namespace water_overflowed_calculation_l26_26490

/-- The water supply rate is 200 kilograms per hour. -/
def water_supply_rate : ℕ := 200

/-- The water tank capacity is 4000 kilograms. -/
def tank_capacity : ℕ := 4000

/-- The water runs for 24 hours. -/
def running_time : ℕ := 24

/-- Calculation for the kilograms of water that overflowed. -/
theorem water_overflowed_calculation :
  water_supply_rate * running_time - tank_capacity = 800 :=
by
  -- calculation skipped
  sorry

end water_overflowed_calculation_l26_26490


namespace find_a_and_b_find_extreme_values_l26_26309

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) * log x - b * x + 3

theorem find_a_and_b :
  (∃ a b : ℝ, f a b 1 = 2 ∧ deriv (f a b) 1 = 0) :=
  sorry

theorem find_extreme_values :
  (∃ x : ℝ, 0 < x ∧ (∀ y : ℝ, 0 < y ∧ y < x → f 0 1 y < f 0 1 x) ∧
    (∀ y : ℝ, x < y → f 0 1 y < f 0 1 x) ∧ f 0 1 x = 2) :=
  sorry

end find_a_and_b_find_extreme_values_l26_26309


namespace hyperbola_eccentricity_l26_26109

theorem hyperbola_eccentricity
  (a b : ℝ)
  (h1 : a^2 = 1)
  (h2 : b^2 = 1)
  (h3 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
  (∃ e : ℝ, e = Real.sqrt (1 + b^2 / a^2) ∧ e = Real.sqrt 2) :=
by
  use Real.sqrt (1 + b^2 / a^2)
  split
  { rw Real.sqrt_eq_rfl,
    congr,
    apply add_eq_of_eq_sub,
    simp [h1, h2] },
  { rw [h1, h2],
    simp }

end hyperbola_eccentricity_l26_26109


namespace sum_divisibility_by_five_l26_26972

theorem sum_divisibility_by_five (A : ℕ) :
  (∀ n, A = (n % 10) ∧ (n % 10 = 5) → 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 45) :=
begin
  sorry
end

end sum_divisibility_by_five_l26_26972


namespace cauchy_bunyakovsky_inequality_l26_26184

theorem cauchy_bunyakovsky_inequality 
  (n : ℕ) 
  (a b k A B K : Fin n → ℝ) : 
  (∑ i, a i * A i)^2 ≤ (∑ i, (a i)^2) * (∑ i, (A i)^2) :=
by
  sorry

end cauchy_bunyakovsky_inequality_l26_26184


namespace cricket_run_target_l26_26803

theorem cricket_run_target : 
  let run_rate_first_10 := 6.2 
  let overs_first_10 := 10
  let runs_first_10 := overs_first_10 * run_rate_first_10
  let run_rate_remaining_40 := 5.5
  let overs_remaining_40 := 40
  let runs_remaining_40 := overs_remaining_40 * run_rate_remaining_40
  runs_first_10 + runs_remaining_40 = 282 :=
  by
  let run_rate_first_10 := 6.2 
  let overs_first_10 := 10
  let runs_first_10 := overs_first_10 * run_rate_first_10
  let run_rate_remaining_40 := 5.5
  let overs_remaining_40 := 40
  let runs_remaining_40 := overs_remaining_40 * run_rate_remaining_40
  sorry

end cricket_run_target_l26_26803


namespace books_total_l26_26776

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end books_total_l26_26776


namespace no_real_sqrt_neg_six_pow_three_l26_26161

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end no_real_sqrt_neg_six_pow_three_l26_26161


namespace quadratic_solution_unique_l26_26858

variable (b : ℝ)
theorem quadratic_solution_unique (h_nonzero : b ≠ 0) (h_discriminant : b^2 - 120 = 0) :
  ∃ x, 3 * x^2 + b * x + 10 = 0 ∧ x = -real.sqrt 30 / 3 :=
by
  sorry

end quadratic_solution_unique_l26_26858


namespace cardinality_of_A_l26_26424

def is_valid (x : Fin 5 → ℤ) : Prop :=
  (∀ i, x i ∈ {-1, 0, 1}) ∧ 1 ≤ (∑ i, |x i|) ∧ (∑ i, |x i|) ≤ 3

def A : Set (Fin 5 → ℤ) :=
  {x | is_valid x}

theorem cardinality_of_A : Fintype.card A = 130 :=
  sorry

end cardinality_of_A_l26_26424


namespace find_z_l26_26752

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26752


namespace quadrilateral_area_proof_l26_26901

noncomputable def area_of_quadrilateral 
  (R r a : ℝ) 
  (h1 : R > 0) 
  (h2 : r > 0) 
  (h3 : a > 0) : ℝ :=
  a^3 * (R + r) / (a^2 + (R + r)^2)

theorem quadrilateral_area_proof 
  (R r a : ℝ)
  (h1 : R > 0)
  (h2 : r > 0)
  (h3 : a > 0) :
  area_of_quadrilateral R r a h1 h2 h3 = (a^3 * (R + r)) / (a^2 + (R + r)^2) :=
begin
  sorry
end

end quadrilateral_area_proof_l26_26901


namespace M_intersection_N_l26_26670

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the proof problem
theorem M_intersection_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_intersection_N_l26_26670


namespace quadrilateral_circles_l26_26036

theorem quadrilateral_circles (A B C D : Type) (h : True) :
  ∃ (n : ℕ), n = 6 :=
by
  use 6
  sorry

end quadrilateral_circles_l26_26036


namespace transformed_data_avg_var_l26_26864

theorem transformed_data_avg_var 
  (x : ℕ → ℝ) -- Sequence of data points
  (h_avg : (∑ i in finset.range 8, x i) / 8 = 6)
  (h_std_dev : real.sqrt ((∑ i in finset.range 8, (x i - 6)^2) / 8) = 2) :
  ((∑ i in finset.range 8, (2 * x i - 6)) / 8 = 6) ∧ 
  ((∑ i in finset.range 8, ((2 * x i - 6) - 6)^2) / 8 = 16) :=
by
  sorry

end transformed_data_avg_var_l26_26864


namespace fourth_term_arithmetic_sequence_l26_26317

theorem fourth_term_arithmetic_sequence :
  ∃ a : ℕ → ℤ, 
    (∀ n, a n ∈ {-1, 0, 1, 2, 3} ∧
      (a (n+1) - a n = 1 ∨ a (n+1) - a n = -1) ∧ 
      a 0 ∈ {0, 2}) ∧
    (a 4 = 3 ∨ a 4 = -1) :=
sorry

end fourth_term_arithmetic_sequence_l26_26317


namespace geometric_sequence_common_ratio_l26_26241

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_relation : ∀ n, a n = (1 / 2) * (a (n + 1) + a (n + 2))) (h_positive : ∀ n, a n > 0) : r = 1 :=
sorry

end geometric_sequence_common_ratio_l26_26241


namespace incorrect_calculation_l26_26157

theorem incorrect_calculation :
  ¬ (3 * real.sqrt 3 - real.sqrt 3 = 2) :=
by {
  sorry
}

end incorrect_calculation_l26_26157


namespace solve_for_z_l26_26717

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26717


namespace calculate_no_iter_l26_26406

def N (x : ℝ) : ℝ := 3 * real.sqrt x
def O (x : ℝ) : ℝ := x ^ 3

theorem calculate_no_iter :
  N (O (N (O (N (O (2))))))
  = 1008 * real.sqrt 6 * real.sqrt (real.root 4 6) := 
  sorry

end calculate_no_iter_l26_26406


namespace circumcenter_distance_parallelogram_l26_26800

theorem circumcenter_distance_parallelogram
  (A B C D E F: Point)
  (parallelogram : Parallelogram A B C D)
  (E_on_AD : E ∈ line_segment A D)
  (F_on_extension_AB : F ∈ line_extension A B)
  (EF_through_E : collinear E F (extension A B))
  (circumcenter_triangle_CDE : ∃ O1, is_circumcenter O1 C D E)
  (circumcenter_triangle_EAF : ∃ O2, is_circumcenter O2 E A F)
  (circumradius_CBF : ∃ R, is_circumradius R C B F):
  O1O2 = R := 
sorry

end circumcenter_distance_parallelogram_l26_26800


namespace concyclic_points_l26_26811

noncomputable def triangle (A B C : Type) := ∃ (A_1 : Type), 
  A_1 ∈ (λ BC, ∀ (A_1 : point BC), ∃ (D E : point BC), 
    (circle (triang AA_1 B) D) ∧ 
    (circle (triang AA_1 C) E) ∧ 
    ∃ (F : point BC), (line BD F) ∧ (line CE F))

variable {A B C : Type}

theorem concyclic_points (A B C A_1 D E F : point BC)
  (h1 : A_1 ∈ segment B C)
  (h2 : D ∈ (circumcircle A A_1 B) ∧ D ∈ (segment A C))
  (h3 : E ∈ (circumcircle A A_1 C) ∧ E ∈ (segment A B))
  (h4 : F ∈ (intersection_line BD CE)) :
  is_concyclic A D E F :=
sorry

end concyclic_points_l26_26811


namespace coin_difference_l26_26079

-- Define the coin denominations
def coin_denominations : List ℕ := [5, 10, 25, 50]

-- Define the target amount Paul needs to pay
def target_amount : ℕ := 60

-- Define the function to compute the minimum number of coins required
noncomputable def min_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the function to compute the maximum number of coins required
noncomputable def max_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the theorem to state the difference between max and min coins is 10
theorem coin_difference : max_coins target_amount coin_denominations - min_coins target_amount coin_denominations = 10 :=
  sorry

end coin_difference_l26_26079


namespace solve_imaginary_eq_l26_26761

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26761


namespace root_expression_equals_181_div_9_l26_26040

noncomputable def polynomial_root_sum (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22) 
  (h3 : a*b*c = 8) : ℝ :=
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) 

theorem root_expression_equals_181_div_9
  (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22)
  (h3 : a*b*c = 8) :
  polynomial_root_sum a b c h1 h2 h3 = 181 / 9 := by 
  sorry

end root_expression_equals_181_div_9_l26_26040


namespace integer_coefficients_24P_l26_26893

/-- A polynomial P of degree 4 where P(x) is integer for all integer x has all coefficients of 24*P(x) as integers. -/
theorem integer_coefficients_24P
  (P : ℤ → ℤ)
  (h_deg : ∀ x : ℤ, P(x) = a * x^4 + b * x^3 + c * x^2 + d * x + e)
  (h_int_vals : ∀ x : ℤ, P x ∈ ℤ) :
  ∀ k ∈ {0, 1, 2, 3, 4}, ∃ n : ℤ, (24 * (coeff_of_degree k P)) = n := 
sorry

end integer_coefficients_24P_l26_26893


namespace approximate_fish_count_and_type_counts_l26_26790

-- Definitions based on conditions
def total_caught := 120
def tagged_caught := 9
def initial_tagged := 80
def ratio_A := 2
def ratio_B := 3
def ratio_C := 4
def total_ratios := ratio_A + ratio_B + ratio_C

-- Proof Statement
theorem approximate_fish_count_and_type_counts :
  ∃ N A B C, 
    N = 1067 ∧
    A = 238 ∧
    B = 357 ∧
    C = 475 ∧
    9 / 120 ≈ 80 / N ∧
    N / total_ratios = ((N + ratio_A - ratio_A) / total_ratios).toNat ∧
    A = ((ratio_A * N) / total_ratios).toNat ∧
    B = ((ratio_B * N) / total_ratios).toNat ∧
    C = (((ratio_C * N) / total_ratios).toNat) - 1 := 
  by
    sorry

end approximate_fish_count_and_type_counts_l26_26790


namespace solve_imaginary_eq_l26_26759

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26759


namespace whitney_total_cost_l26_26164

-- Definitions of the number of items and their costs
def w := 15
def c_w := 14
def f := 12
def c_f := 13
def s := 5
def c_s := 10
def m := 8
def c_m := 3

-- The total cost Whitney spent
theorem whitney_total_cost :
  w * c_w + f * c_f + s * c_s + m * c_m = 440 := by
  sorry

end whitney_total_cost_l26_26164


namespace knot_forms_regular_pentagon_l26_26982

theorem knot_forms_regular_pentagon (paper_strip : Type) 
  (constant_width : ℝ) (is_tied_into_simple_knot : paper_strip → Prop)
  (is_flat_after_tightening : paper_strip → Prop)
  (forms_convex_pentagon : paper_strip → Prop)
  (has_three_folds_two_edges : paper_strip → Prop) :
  ∃ (P : Type), is_regular_pentagon P :=
by
  sorry

end knot_forms_regular_pentagon_l26_26982


namespace sum_of_roots_eq_12_l26_26619

theorem sum_of_roots_eq_12 :
  let f := λ x : ℝ, 4 * x^2 - 58 * x + 190
  let g := λ x : ℝ, (29 - 4 * x - Real.log (x) / Real.log 2) * (Real.log x / Real.log 2)
  ∃ x1 x2 : ℝ, (f x1 = g x1) ∧ (f x2 = g x2) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 12) :=
sorry

end sum_of_roots_eq_12_l26_26619


namespace coeff_x3_q3_l26_26774

def q (x : ℝ) : ℝ := x^4 - 2*x^2 - 5*x + 3

theorem coeff_x3_q3 : 
  coefficient (3 : ℕ) ((q x)^3) = -125 := 
sorry

end coeff_x3_q3_l26_26774


namespace find_function_l26_26066

theorem find_function (f : ℝ → ℝ) :
  (∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)) →
  (∀ u : ℝ, 0 ≤ f u) →
  (∀ x : ℝ, f x = 0) := 
  by
    sorry

end find_function_l26_26066


namespace proof_ab_value_l26_26334

theorem proof_ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by
  sorry

end proof_ab_value_l26_26334


namespace coefficients_of_quadratic_l26_26808

theorem coefficients_of_quadratic (x : ℝ) :
    ∃ (a b c : ℝ), a = 3 ∧ b = -1 ∧ c = -2 ∧ a * x^2 + b * x + c = 0 :=
by
  use [3, -1, -2]
  simp
  sorry

end coefficients_of_quadratic_l26_26808


namespace find_f_3_l26_26116

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_4_eq_6 : f 4 = 6

theorem find_f_3 : f 3 = 9 / 2 :=
by sorry

end find_f_3_l26_26116


namespace correct_calculation_l26_26935

theorem correct_calculation (a b : ℝ) : 
  (a + 2 * a = 3 * a) := by
  sorry

end correct_calculation_l26_26935


namespace find_center_of_ellipse_l26_26660

-- Defining the equation of the ellipse
def ellipse (x y : ℝ) : Prop := 2*x^2 + 2*x*y + y^2 + 2*x + 2*y - 4 = 0

-- The coordinates of the center
def center_of_ellipse : ℝ × ℝ := (0, -1)

-- The theorem asserting the center of the ellipse
theorem find_center_of_ellipse (x y : ℝ) (h : ellipse x y) : (x, y) = center_of_ellipse :=
sorry

end find_center_of_ellipse_l26_26660


namespace solve_for_z_l26_26710

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26710


namespace arc_length_of_sector_maximum_area_of_sector_l26_26283

def toRadians (degrees : ℝ) : ℝ := degrees * (Real.pi / 180)

theorem arc_length_of_sector (α : ℝ) (r : ℝ) (hα : α = 60) (hr : r = 3) : 
  (r / 3) * toRadians α = Real.pi :=
by {
  unfold toRadians,
  rw [hα, hr], -- plugging in α = 60 and r = 3
  -- the remaining part can be solved by computation and simplification, this can be skipped with sorry
  sorry  
}

theorem maximum_area_of_sector (r : ℝ) (P : ℝ) (hP : P = 16) : 
  ∃ α, (α = 2) ∧ (1/2) * (P - 2 * r) * r = 16 :=
by {
  -- given the total perimeter P = 16
  -- we need to find r and alpha for the maximum area
  use 2, -- as the maximum angle
  -- remaining computation can be checked, so skipping with sorry
  sorry
}

end arc_length_of_sector_maximum_area_of_sector_l26_26283


namespace num_dimes_l26_26169

/--
Given eleven coins consisting of pennies, nickels, dimes, quarters, and half-dollars,
having a total value of $1.43, with at least one coin of each type,
prove that there must be exactly 4 dimes.
-/
theorem num_dimes (p n d q h : ℕ) :
  1 ≤ p ∧ 1 ≤ n ∧ 1 ≤ d ∧ 1 ≤ q ∧ 1 ≤ h ∧ 
  p + n + d + q + h = 11 ∧ 
  (1 * p + 5 * n + 10 * d + 25 * q + 50 * h) = 143
  → d = 4 :=
by
  sorry

end num_dimes_l26_26169


namespace sum_divides_product_iff_not_prime_l26_26447

theorem sum_divides_product_iff_not_prime (n : ℕ) : 
  (↑((n * (n + 1))/2) ∣ n.factorial) ↔ ¬ nat.prime (n + 1) :=
sorry

end sum_divides_product_iff_not_prime_l26_26447


namespace problem_statement_l26_26856

open Nat

theorem problem_statement (a b c d : ℕ) (h1 : d ∣ a^(2*b) + c) (h2 : d ≥ a + c) : 
  d ≥ a + root (2*b) a :=
by
  sorry

end problem_statement_l26_26856


namespace target_runs_l26_26367

theorem target_runs (run_rate_7_overs : ℝ) (overs_7 : ℕ) 
                    (run_rate_30_overs : ℝ) (overs_30 : ℕ) :
  run_rate_7_overs = 4.2 →
  overs_7 = 7 →
  run_rate_30_overs = 8.42 →
  overs_30 = 30 →
  let runs_first_7 := (run_rate_7_overs * overs_7).floor
  let runs_remaining_30 := (run_rate_30_overs * overs_30).floor
  let target_runs := runs_first_7 + runs_remaining_30
  target_runs = 281 :=
by
  intros
  sorry

end target_runs_l26_26367


namespace juan_original_number_l26_26021

theorem juan_original_number (n : ℤ) 
  (h : ((2 * (n + 3) - 2) / 2) = 8) : 
  n = 6 := 
sorry

end juan_original_number_l26_26021


namespace overall_percentage_score_l26_26450

theorem overall_percentage_score (quiz_score : ℝ) (quiz_total : ℕ) 
                                 (test_score : ℝ) (test_total : ℕ) 
                                 (exam_score : ℝ) (exam_total : ℕ) :
  quiz_score = 60 / 100 ∧ quiz_total = 15 ∧
  test_score = 85 / 100 ∧ test_total = 20 ∧
  exam_score = 75 / 100 ∧ exam_total = 40 →
  (let total_correct := (quiz_score * quiz_total) + (test_score * test_total) + (exam_score * exam_total)
   let total_problems := quiz_total + test_total + exam_total
   let overall_percent := (total_correct / total_problems) * 100
   in overall_percent).round = 75 :=
by
  intros
  sorry

end overall_percentage_score_l26_26450


namespace problem1_problem2_l26_26313

noncomputable def h (x a : ℝ) : ℝ := (x - a) * Real.exp x + a
noncomputable def f (x b : ℝ) : ℝ := x^2 - 2 * b * x - 3 * Real.exp 1 + Real.exp 1 + 15 / 2

theorem problem1 (a : ℝ) :
  ∃ c, ∀ x ∈ Set.Icc (-1:ℝ) (1:ℝ), h x a ≥ c :=
by
  sorry

theorem problem2 (b : ℝ) :
  (∀ x1 ∈ Set.Icc (-1:ℝ) (1:ℝ), ∃ x2 ∈ Set.Icc (1:ℝ) (2:ℝ), h x1 3 ≥ f x2 b) →
  b ≥ 17 / 8 :=
by
  sorry

end problem1_problem2_l26_26313


namespace solve_imaginary_eq_l26_26767

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26767


namespace sum_of_cubes_l26_26881

theorem sum_of_cubes (a b : ℕ) (h1 : 2 * x = a) (h2 : 3 * x = b) (h3 : b - a = 3) : a^3 + b^3 = 945 := by
  sorry

end sum_of_cubes_l26_26881


namespace symmetric_point_P_l26_26080

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the function to get the symmetric point with respect to the origin
def symmetric_point (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, -point.2)

-- State the theorem that proves the symmetric point of P is (-1, 2)
theorem symmetric_point_P :
  symmetric_point P = (-1, 2) :=
  sorry

end symmetric_point_P_l26_26080


namespace sum_of_roots_eq_12_l26_26620

noncomputable def equation (x : ℝ) : ℝ := 4 * x^2 - 58 * x + 190 - (29 - 4 * x - Real.log x / Real.log 2) * (Real.log x / Real.log 2)

theorem sum_of_roots_eq_12 : 
  ∀ x : ℝ, equation x = 0 → x ∈ set_of (λ x, x = 4 ∨ x = 8) → (∑ x in {4, 8}, x) = 12 := 
by
  sorry

end sum_of_roots_eq_12_l26_26620


namespace find_S4_l26_26369

noncomputable def S (n : ℕ) (a : ℕ → ℚ) : ℚ :=
  ∑ i in finset.range n, a i

variables {a : ℕ → ℚ} (h_geom : ∃ q > 0, ∀ n, a (n+1) = a n * q)
  (h_pos : ∀ n, a n > 0)
  (h_S3 : 2 * S 3 a = 8 * a 1 + 3 * a 2)
  (h_a4 : a 3 = 16)

theorem find_S4 : S 4 a = 30 :=
sorry

end find_S4_l26_26369


namespace brother_plays_more_l26_26030

-- Define the known conditions
def lena_hours := 3.5
def total_minutes := 437
def minutes_per_hour := 60

-- Convert Lena's hours to minutes
def lena_minutes := lena_hours * minutes_per_hour

-- The brother's playing time calculation
def brother_minutes := total_minutes - lena_minutes

-- The difference in playing time
def difference := brother_minutes - lena_minutes

-- The theorem to be proved
theorem brother_plays_more :
  difference = 17 := by
  sorry

end brother_plays_more_l26_26030


namespace simplify_an_over_bn_l26_26624

noncomputable def a_n (n : ℕ) : ℚ :=
∑ k in Finset.range (n + 1), 1 / (Nat.choose n k)

noncomputable def b_n (n : ℕ) : ℚ :=
∑ k in Finset.range (n + 1), k / (Nat.choose n k)

theorem simplify_an_over_bn (n : ℕ) (h_pos : 0 < n) : (a_n n) / (b_n n) = 2 / n := by
  sorry

end simplify_an_over_bn_l26_26624


namespace mutually_exclusive_necessary_but_not_sufficient_l26_26148

theorem mutually_exclusive_necessary_but_not_sufficient (A B : Prop) 
  (mutually_exclusive : ¬(A ∧ B))
  (complementary : A ↔ ¬B) :
  (∀ (A B : Prop), mutually_exclusive → complementary → ¬ (mutually_exclusive ↔ complementary)) := 
sorry

end mutually_exclusive_necessary_but_not_sufficient_l26_26148


namespace max_height_of_ball_l26_26882

variable (h v₀ h₀ t : ℝ)

theorem max_height_of_ball :
  (h₀ = 1.5) →
  (v₀ = 20) →
  (∀ t, h = -5 * t^2 + v₀ * t + h₀) →
  ∃ t, h = 21.5 :=
by
  intros h₀_eq v₀_eq h_eq
  sorry

end max_height_of_ball_l26_26882


namespace hyperbola_eccentricity_range_l26_26668

open Real

noncomputable def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

def eccentricity (a b : ℝ) : ℝ := sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_range (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (c := sqrt (a^2 + b^2))
    (AF := b^2 / a)
    (right_vertex_inside : a + c < AF) :
  eccentricity a b > 2 := by
  sorry

end hyperbola_eccentricity_range_l26_26668


namespace train_times_l26_26898

theorem train_times (t x : ℝ) : 
  (30 * t = 360) ∧ (36 * (t - x) = 360) → x = 2 :=
by
  sorry

end train_times_l26_26898


namespace tangent_line_eq_at_1_l26_26308

noncomputable def f (x : ℝ) : ℝ :=
  2 * f (2 * x - 1) - 3 * x^2 + 2

theorem tangent_line_eq_at_1 :
  f 1 = 1 ∧ deriv f 1 = 2 → ∀ y x, y = f 1 + deriv f 1 * (x - 1) ↔ y = 2 * x - 1 :=
by 
  intro h
  obtain ⟨hf1, hdf1⟩ := h
  sorry

end tangent_line_eq_at_1_l26_26308


namespace no_real_solutions_for_identical_lines_l26_26415

theorem no_real_solutions_for_identical_lines :
  ¬∃ (a d : ℝ), (∀ x y : ℝ, 5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) :=
by
  sorry

end no_real_solutions_for_identical_lines_l26_26415


namespace Joe_spends_68_dollars_l26_26389

def Joe_spends_at_market
  (n_oranges : ℕ) (cost_orange : ℝ)
  (n_juices : ℕ) (cost_juice : ℝ)
  (n_honey : ℕ) (cost_honey : ℝ)
  (n_plants : ℕ) (cost_two_plants : ℝ) : ℝ :=
  let cost_plant := cost_two_plants / 2
  in (n_oranges * cost_orange) + (n_juices * cost_juice) + (n_honey * cost_honey) + (n_plants * cost_plant)

theorem Joe_spends_68_dollars :
  Joe_spends_at_market 3 4.5 7 0.5 3 5 4 18 = 68 := by
  sorry

end Joe_spends_68_dollars_l26_26389


namespace proof_inequality_l26_26130

theorem proof_inequality (x : ℝ) : (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5 ∨ -9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end proof_inequality_l26_26130


namespace number_of_nonempty_subsets_A_l26_26320

def isInA (x : ℤ) : Prop := (x - 2) * (x - 6) ≥ 3

def A : Set ℤ := {x | isInA x ∧ 0 ≤ x ∧ x ≤ 7}

theorem number_of_nonempty_subsets_A : 
  ∃ n : ℕ, n = (2 ^ (Finset.card (Finset.filter (λ x, x ∈ A) (Finset.range 8))) - 1) ∧ n = 63 := 
by
  sorry

end number_of_nonempty_subsets_A_l26_26320


namespace propositions_correct_l26_26037

-- Definitions for planes and lines
variables (Plane Line : Type) [inhabited Plane] [inhabited Line]
variables (α β : Plane) (l m : Line)

-- Definitions of relevant relations
def is_parallel_planes (α β : Plane) : Prop := sorry
def is_perpendicular_planes (α β : Plane) : Prop := sorry
def is_parallel_lines (l m : Line) : Prop := sorry
def is_perpendicular_line_plane (l : Line) (β : Plane) : Prop := sorry
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

-- State the problem as a Lean theorem
theorem propositions_correct :
  (∀ (α β : Plane) (l m : Line), line_in_plane l α → line_in_plane m β → is_parallel_planes α β → ¬ is_parallel_lines l m) ∧
  (∀ (α β : Plane) (l : Line), line_in_plane l α → is_perpendicular_line_plane l β → is_perpendicular_planes α β) :=
by {
  sorry
}

end propositions_correct_l26_26037


namespace ratio_of_chocolates_l26_26578

theorem ratio_of_chocolates (w d : ℕ) (h_w : w = 20) (h_d : d = 15) : (w / Nat.gcd w d, d / Nat.gcd w d) = (4, 3) :=
by
  have h_gcd : Nat.gcd w d = 5 := by sorry  -- We need to show gcd of 20 and 15 is 5
  rw [h_w, h_d] at h_gcd  -- Apply values
  rw [h_w, h_d]  -- Apply values to the ratio
  rw [Nat.div_eq_of_eq_mul_right] at *  -- Simplifications
  -- Verification (20 / 5, 15 / 5) equals (4, 3)
  exact (rfl, rfl)
lsymm
-- Additional sorry proof fragment to complete the proof
sorry

end ratio_of_chocolates_l26_26578


namespace sequence_continues_correctly_l26_26259

theorem sequence_continues_correctly :
  let seq : List ℕ := [1, 2, 4, 7, 11, 16, 22]
  let next_values := 29 :: 38 :: []
  next_values = List.drop 7 (List.scanl (+) 1 (List.range (List.length seq + 2))) := by
  sorry

end sequence_continues_correctly_l26_26259


namespace floodDamageInUSD_l26_26553

def floodDamageAUD : ℝ := 45000000
def exchangeRateAUDtoUSD : ℝ := 1.2

theorem floodDamageInUSD : floodDamageAUD * (1 / exchangeRateAUDtoUSD) = 37500000 := 
by 
  sorry

end floodDamageInUSD_l26_26553


namespace michael_marcos_distance_when_ball_is_touched_first_l26_26981

def ball_speed := 4 -- m/s
def michael_speed := 9 -- m/s
def marcos_speed := 8 -- m/s
def initial_ball_position_from_michael := 15 -- m
def initial_ball_position_from_marcos := 30 -- m

noncomputable def proof_distance_between_michael_marcos_first_touch : Prop :=
  ∃ t m_pos b_pos,
    t = initial_ball_position_from_marcos / (marcos_speed + ball_speed) ∧
    m_pos = michael_speed * t ∧
    b_pos = initial_ball_position_from_michael + ball_speed * t ∧
    (b_pos - m_pos) = 2.5 -- m

theorem michael_marcos_distance_when_ball_is_touched_first : proof_distance_between_michael_marcos_first_touch :=
by
  sorry

end michael_marcos_distance_when_ball_is_touched_first_l26_26981


namespace solve_for_x_l26_26455

theorem solve_for_x (x : Real) (h : sqrt (3 / x + 5) = 5 / 2) : x = 2.4 :=
by sorry

end solve_for_x_l26_26455


namespace t_f_7_eq_l26_26054

def t (x : ℝ) : ℝ := real.sqrt (5 * x + 2)
def f (x : ℝ) : ℝ := 7 - t x

theorem t_f_7_eq: t (f 7) = real.sqrt (37 - 5 * real.sqrt 37) := by
  sorry

end t_f_7_eq_l26_26054


namespace find_y_l26_26863

theorem find_y (x y : ℝ) (h1 : (100 + 200 + 300 + x) / 4 = 250) (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : y = 50 :=
by
  sorry

end find_y_l26_26863


namespace robins_hair_growth_l26_26848

theorem robins_hair_growth :
  ∃ G : ℤ, 14 + G - 20 = 2 :=
begin
  use 8,
  linarith,
end

end robins_hair_growth_l26_26848


namespace leak_empties_cistern_l26_26193

theorem leak_empties_cistern :
  (R : ℝ) (L : ℝ) 
  (h1 : R = 1 / 5) 
  (h2 : R - L = 1 / 6) 
  : 1 / L = 30 :=
by
  sorry

end leak_empties_cistern_l26_26193


namespace solution_set_inequality_l26_26661

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom increasing_on_positive : ∀ {x y : ℝ}, 0 < x → x < y → f x < f y
axiom f_one : f 1 = 0

theorem solution_set_inequality :
  {x : ℝ | (f x) / x < 0} = {x : ℝ | x < -1} ∪ {x | 0 < x ∧ x < 1} := sorry

end solution_set_inequality_l26_26661


namespace f_eq_n_l26_26831

open Nat

noncomputable def f : ℕ → ℕ := sorry

theorem f_eq_n (f : ℕ → ℕ) (h : ∀ n : ℕ, f(n + 1) > f(f(n))) : ∀ n : ℕ, f(n) = n :=
sorry

end f_eq_n_l26_26831


namespace area_of_triangle_BPQ_l26_26950

-- Definitions and conditions
variables {A B C H P Q X Y : Type} [nonempty A] [nonempty B] [nonempty C] [nonempty H] [nonempty P] [nonempty Q] [nonempty X] [nonempty Y]
variables (ABC : A → B → C → Prop) (hypotenuse_AC : B → H → C → Prop)
variables (BH : B → H → Prop) (H_is_altitude: BH H)
variables (Xr : A → B → H → X) (Yr : B → C → H → Y)
variables (line_XY_intersects_AB : X → B → P → Prop)
variables (line_XY_intersects_BC : X → B → Q → Prop)
variables (BH_length : ℝ) (h : ℝ)

-- Theorem to be proved, incorporating the conditions above
theorem area_of_triangle_BPQ :
  ∀ (ABC : Prop) (A B C H P Q X Y : Type) [nonempty A] [nonempty B] [nonempty C] [nonempty H] [nonempty P] [nonempty Q] [nonempty X] [nonempty Y]
    (hypotenuse_AC : B → H → C → Prop)
    (H_is_altitude : BH H)
    (BH : ℝ)
    (h : ∀ A B C H P Q X Y, BH = h),
  ∃ S_BP: ℝ, S_BP = h^2 / 2 := sorry

end area_of_triangle_BPQ_l26_26950


namespace cost_per_hectare_proof_l26_26103

-- Conditions
def base_of_field := 300
def height_of_field := 100
def b_eq_3h (b h : ℝ) : Prop := b = 3 * h
def total_cost := 333.18
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h
def area_in_hectares (A : ℝ) : ℝ := A / 10000
def cost_per_hectare (total_cost area_hectares : ℝ) : ℝ := total_cost / area_hectares

-- The main theorem
theorem cost_per_hectare_proof :
  b_eq_3h base_of_field height_of_field ∧
  let A := area_of_triangle base_of_field height_of_field
  let area_hectares := area_in_hectares A in
  cost_per_hectare total_cost area_hectares = 222.12 :=
by
  sorry

end cost_per_hectare_proof_l26_26103


namespace complex_number_solution_l26_26680

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26680


namespace problem_statement_l26_26833

-- Define the roots of the polynomial and the value of t.
variables {p q r : ℝ}
def polynomial := (x : ℝ) -> x^3 - 8 * x^2 + 14 * x - 2

-- Conditions on the roots of the polynomial
axiom root_1 : polynomial p = 0
axiom root_2 : polynomial q = 0
axiom root_3 : polynomial r = 0

-- Define t in terms of p, q, and r
def t : ℝ := (sqrt p) + (sqrt q) + (sqrt r)

-- The statement to prove
theorem problem_statement : t^4 - 16 * t^2 - 12 * t = -8 :=
by
  sorry

end problem_statement_l26_26833


namespace zoe_takes_correct_amount_of_money_l26_26527

def numberOfPeople : ℕ := 6
def costPerSoda : ℝ := 0.5
def costPerPizza : ℝ := 1.0

def totalCost : ℝ := (numberOfPeople * costPerSoda) + (numberOfPeople * costPerPizza)

theorem zoe_takes_correct_amount_of_money : totalCost = 9 := sorry

end zoe_takes_correct_amount_of_money_l26_26527


namespace circle_incircle_tangent_radius_l26_26018

theorem circle_incircle_tangent_radius (r1 r2 r3 : ℕ) (k : ℕ) (h1 : r1 = 1) (h2 : r2 = 4) (h3 : r3 = 9) : 
  k = 11 :=
by
  -- Definitions according to the problem
  let k₁ := r1
  let k₂ := r2
  let k₃ := r3
  -- Hypotheses given by the problem
  have h₁ : k₁ = 1 := h1
  have h₂ : k₂ = 4 := h2
  have h₃ : k₃ = 9 := h3
  -- Prove the radius of the incircle k
  sorry

end circle_incircle_tangent_radius_l26_26018


namespace avg_equals_100x_implies_x_eq_50_over_101_l26_26465

theorem avg_equals_100x_implies_x_eq_50_over_101 (x : ℝ) 
  (h : (∑ i in Finset.range 100 + x) / 100 = 100 * x) : 
  x = 50 / 101 :=
by sorry

end avg_equals_100x_implies_x_eq_50_over_101_l26_26465


namespace cost_per_gallon_is_45_l26_26386

variable (totalArea coverage cost_jason cost_jeremy dollars_per_gallon : ℕ)

-- Conditions
def total_area := 1600
def coverage_per_gallon := 400
def num_coats := 2
def contribution_jason := 180
def contribution_jeremy := 180

-- Gallons needed calculation
def gallons_per_coat := total_area / coverage_per_gallon
def total_gallons := gallons_per_coat * num_coats

-- Total cost calculation
def total_cost := contribution_jason + contribution_jeremy

-- Cost per gallon calculation
def cost_per_gallon := total_cost / total_gallons

-- Proof statement
theorem cost_per_gallon_is_45 : cost_per_gallon = 45 :=
by
  sorry

end cost_per_gallon_is_45_l26_26386


namespace will_has_81_dollars_left_l26_26938

def money_given_by_mom := 74
def cost_sweater := 9
def cost_tshirt := 11
def cost_shoes := 30
def refund_percentage_shoes := 0.90

theorem will_has_81_dollars_left :
  let total_spent_clothes := cost_sweater + cost_tshirt in
  let refund_shoes := refund_percentage_shoes * cost_shoes in
  let money_left_after_clothes := money_given_by_mom - total_spent_clothes in
  let money_left := money_left_after_clothes + refund_shoes in
  money_left = 81 :=
by
  sorry

end will_has_81_dollars_left_l26_26938


namespace previous_salary_l26_26437

theorem previous_salary (P : ℝ) (h : 1.05 * P = 2100) : P = 2000 :=
by
  sorry

end previous_salary_l26_26437


namespace LCM_of_fractions_l26_26515

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l26_26515


namespace solve_imaginary_eq_l26_26769

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26769


namespace fraction_product_l26_26590

theorem fraction_product : (2 * (-4)) / (9 * 5) = -8 / 45 :=
  by sorry

end fraction_product_l26_26590


namespace b_50_is_6_over_20610_l26_26286

-- Define the sequence b
def b : ℕ → ℚ
| 1     := 2
| (n+2) := let T_n := finset.sum (finset.range (n + 1)) b in
           3 * T_n^2 / (3 * T_n - 2)

-- Define the sum of the first n terms of the sequence
def T (n : ℕ) : ℚ := finset.sum (finset.range n) b

-- The theorem to prove that b_50 = 6 / 20610
theorem b_50_is_6_over_20610 :
  b 50 = 6 / 20610 :=
sorry

end b_50_is_6_over_20610_l26_26286


namespace find_z_l26_26749

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26749


namespace solve_for_z_l26_26714

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26714


namespace cos_double_angle_of_tan_half_l26_26288

theorem cos_double_angle_of_tan_half (α : ℝ) (h : Real.tan α = 1 / 2) :
  Real.cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_of_tan_half_l26_26288


namespace find_b_n_find_T_n_l26_26662

-- Conditions
def S (n : ℕ) : ℕ := 3 * n^2 + 8 * n
def a (n : ℕ) : ℕ := S n - S (n - 1) -- provided n > 1
def b : ℕ → ℕ := sorry -- This is what we need to prove
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n  -- Definition of c_n
def T (n : ℕ) : ℕ := sorry -- The sum of the first n terms of c_n

-- Proof requirements
def proof_b_n := ∀ n : ℕ, b n = 3 * n + 1
def proof_T_n := ∀ n : ℕ, T n = 3 * n * 2^(n+2)

theorem find_b_n : proof_b_n := 
by sorry

theorem find_T_n : proof_T_n := 
by sorry

end find_b_n_find_T_n_l26_26662


namespace repeating_decimal_division_l26_26919

theorem repeating_decimal_division :
  (0.\overline{54} / 0.\overline{18}) = 3 :=
by
  have h1 : 0.\overline{54} = 54 / 99 := sorry
  have h2 : 0.\overline{18} = 18 / 99 := sorry
  have h3 : (54 / 99) / (18 / 99) = 54 / 18 := sorry
  have h4 : 54 / 18 = 3 := sorry
  rw [h1, h2, h3, h4]
  exact rfl

end repeating_decimal_division_l26_26919


namespace recurring_decimal_fraction_l26_26912

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l26_26912


namespace sufficient_condition_above_2c_l26_26209

theorem sufficient_condition_above_2c (a b c : ℝ) (h1 : a > c) (h2 : b > c) : a + b > 2 * c :=
by
  sorry

end sufficient_condition_above_2c_l26_26209


namespace small_monkey_dolls_cheaper_than_large_l26_26601

theorem small_monkey_dolls_cheaper_than_large (S : ℕ) 
  (h1 : 300 / 6 = 50) 
  (h2 : 300 / S = 75) 
  (h3 : 75 - 50 = 25) : 
  6 - S = 2 := 
sorry

end small_monkey_dolls_cheaper_than_large_l26_26601


namespace not_all_pieces_found_l26_26583

theorem not_all_pieces_found (N : ℕ) (petya_tore : ℕ → ℕ) (vasya_tore : ℕ → ℕ) : 
  (∀ n, petya_tore n = n * 5 - n) →
  (∀ n, vasya_tore n = n * 9 - n) →
  1988 = N ∧ (N % 2 = 1) → false :=
by
  intros h_petya h_vasya h
  sorry

end not_all_pieces_found_l26_26583


namespace sum_of_edges_of_pyramid_l26_26979

-- Defining the problem conditions
def base_length := 15 -- length of the rectangular base in cm
def base_width := 8 -- width of the rectangular base in cm
def height := 15 -- height from the center of the base to the peak in cm

-- The theorem to prove
theorem sum_of_edges_of_pyramid :
  let diagonal := Math.sqrt (base_length^2 + base_width^2)
  let slant_height := Math.sqrt (height^2 + (diagonal / 2)^2)
  let base_perimeter := 2 * (base_length + base_width)
  let total_edge_length := base_perimeter + 4 * slant_height
  Float.round total_edge_length 0 = 115 := 
  by
    sorry

end sum_of_edges_of_pyramid_l26_26979


namespace rational_greater_than_three_l26_26991

theorem rational_greater_than_three :
  ∃ x : ℚ, x = 11 / 3 ∧ x > 3 :=
by
  -- Assume the conditions in the problem.
  have h1 : |-3| = 3 := abs_neg_of_nonneg (by norm_num),
  have h2 : irrational π := by exact irrational_real_of_ne_rat π,
  have h3 : irrational (sqrt 10) := by exact irrational_sqrt_of_not_sq 10 (by norm_num),

  -- State that 11/3 is rational.
  let d := 11 / 3 process the answer for
  use d,
  split,
  exact rfl,
  norm_num,
  sorry

end rational_greater_than_three_l26_26991


namespace LCM_of_fractions_l26_26512

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l26_26512


namespace product_greater_than_sum_l26_26412

variable {a b : ℝ}

theorem product_greater_than_sum (ha : a > 2) (hb : b > 2) : a * b > a + b := 
  sorry

end product_greater_than_sum_l26_26412


namespace optimal_tennis_court_area_l26_26896

theorem optimal_tennis_court_area :
  ∀ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 → l * w ≤ 10000 := 
by
  intros l w h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry  -- proof to be done

end optimal_tennis_court_area_l26_26896


namespace system_of_linear_equations_l26_26430

theorem system_of_linear_equations (a b c d e : ℝ) 
  (h1 : a + b = 14)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : d + e = 6)
  (h5 : a - 2e = 1) : 
  a + d = 8 :=
sorry

end system_of_linear_equations_l26_26430


namespace max_bicycle_distance_l26_26191

-- Define the properties of the tires
def front_tire_duration : ℕ := 5000
def rear_tire_duration : ℕ := 3000

-- Define the maximum distance the bicycle can travel
def max_distance : ℕ := 3750

-- The main statement to be proven (proof is not required)
theorem max_bicycle_distance 
  (swap_usage : ∀ (d1 d2 : ℕ), d1 + d2 <= front_tire_duration + rear_tire_duration) : 
  ∃ (x : ℕ), x = max_distance := 
sorry

end max_bicycle_distance_l26_26191


namespace additional_wolves_in_pack_l26_26137

-- Define the conditions
def wolves_out_hunting : ℕ := 4
def meat_per_wolf_per_day : ℕ := 8
def hunting_days : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat per wolf for hunting days
def meat_per_wolf_total : ℕ := meat_per_wolf_per_day * hunting_days

-- Calculate wolves fed per deer
def wolves_fed_per_deer : ℕ := meat_per_deer / meat_per_wolf_total

-- Calculate total deer killed by wolves out hunting
def total_deers_killed : ℕ := wolves_out_hunting

-- Calculate total meat provided by hunting wolves
def total_meat_provided : ℕ := total_deers_killed * meat_per_deer

-- Calculate number of wolves fed by total meat provided
def total_wolves_fed : ℕ := total_meat_provided / meat_per_wolf_total

-- Define the main theorem to prove the answer
theorem additional_wolves_in_pack (total_wolves_fed wolves_out_hunting : ℕ) : 
  total_wolves_fed - wolves_out_hunting = 16 :=
by
  sorry

end additional_wolves_in_pack_l26_26137


namespace find_rate_of_interest_l26_26210

-- Defining the problem conditions
variables (SI P T R : ℝ)
variable h1 : SI = 750
variable h2 : P = 2500
variable h3 : T = 5

-- The formula for simple interest
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- The theorem to prove that the rate of interest R equals 6%
theorem find_rate_of_interest (h1 : simple_interest P R T = 750) (h2 : P = 2500) (h3 : T = 5) : R = 6 :=
by sorry

end find_rate_of_interest_l26_26210


namespace inverse_x2_minus_x_l26_26775

noncomputable def x : ℂ := (3 - complex.I * real.sqrt 5) / 2

theorem inverse_x2_minus_x : (1 / (x^2 - x)) = ((-2 + 4 * complex.I * real.sqrt 5) / 5) := 
by
  sorry

end inverse_x2_minus_x_l26_26775


namespace sin_alpha_l26_26663

-- Definitions and conditions
def unit_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1

def P := (1/2 : ℝ, y)

theorem sin_alpha (α : ℝ) (h : unit_circle P) : sin (π / 2 + α) = 1 / 2 :=
  sorry

end sin_alpha_l26_26663


namespace xy_in_N_l26_26821

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  -- hint: use any knowledge and axioms from Mathlib to aid your proof
  sorry

end xy_in_N_l26_26821


namespace base_n_multiple_of_5_number_of_valid_n_l26_26627

-- Define the function representing the base-n number 125236_n in decimal form
def f (n : ℕ) : ℕ := 6 + 3 * n + 2 * n^2 + 5 * n^3 + 2 * n^4 + n^5

theorem base_n_multiple_of_5 :
  (∃ (n : ℕ), 2 ≤ n ∧ n ≤ 100 ∧ f(n) % 5 = 0) :=
begin
  sorry
end

-- State that the number of values of n for which 125236_n is a multiple of 5 is exactly 20
theorem number_of_valid_n : (finset.range (100 - 1)).filter (λ n, 2 ≤ n ∧ f(n) % 5 = 0) = 20 :=
begin
  sorry
end

end base_n_multiple_of_5_number_of_valid_n_l26_26627


namespace find_x_l26_26252

theorem find_x (x : ℝ) : 16^(x + 2) = 352 + 16^x → x = 0.25 :=
by
  sorry

end find_x_l26_26252


namespace largest_possible_A_l26_26156

-- Define natural numbers
variables (A B C : ℕ)

-- Given conditions
def division_algorithm (A B C : ℕ) : Prop := A = 8 * B + C
def B_equals_C (B C : ℕ) : Prop := B = C

-- The proof statement
theorem largest_possible_A (h1 : division_algorithm A B C) (h2 : B_equals_C B C) : A = 63 :=
by
  -- Proof is omitted
  sorry

end largest_possible_A_l26_26156


namespace profit_percentage_l26_26562

noncomputable def original_price : ℝ := 100
noncomputable def price_A_to_B : ℝ := original_price + 0.35 * original_price
noncomputable def price_B_to_C : ℝ := price_A_to_B - 0.25 * price_A_to_B
noncomputable def price_C_to_D : ℝ := price_B_to_C + 0.2 * price_B_to_C
noncomputable def price_D_to_E : ℝ := price_C_to_D - 0.15 * price_C_to_D
noncomputable def overall_profit : ℝ := price_D_to_E - original_price

theorem profit_percentage :
  (overall_profit / original_price) * 100 = 3.275 := 
by
  sorry

end profit_percentage_l26_26562


namespace max_value_of_y_l26_26122

-- Given function definition
def y (x : ℝ) : ℝ :=
  (√3 / 2) * sin (x + π / 2) + cos (π / 6 - x)

-- Theorem stating the maximum value of the function
theorem max_value_of_y : ∃ x : ℝ, y x = sqrt 13 / 2 :=
sorry

end max_value_of_y_l26_26122


namespace determine_z_l26_26701

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26701


namespace sine_central_angle_subtending_arc_PR_l26_26791

-- Definitions
def radius := 7
def PQ := 14
def RT := 5
axiom RS_bisects_PQ_at_T : ∀ P Q R S T, RS bisects PQ at T
noncomputable def PR_radius_eq := radius

-- Central angle subtending arc PR
theorem sine_central_angle_subtending_arc_PR (m n : ℕ) :
  sin (real.pi) = 0 ∧ m = 0 ∧ n = 1 → m * n = 0 :=
by sorry

end sine_central_angle_subtending_arc_PR_l26_26791


namespace ratio_CD_BD_l26_26001

universe u

variables {Point : Type u} [MetricSpace Point]

-- Definitions based on problem conditions
variables (A B C D E T : Point)
variables {x y : ℝ}

-- Declaring given conditions
def AT_DT_ratio (A T D : Point) : ℝ := 2
def BT_ET_ratio (B T E : Point) : ℝ := 3

-- The aim is to find the ratio CD/BD
def CD_BD_ratio (C D B : Point) : ℝ := 2/9

-- Proving that the ratio CD/BD is indeed 2/9 given the conditions
theorem ratio_CD_BD (A B C D E T : Point) (h1: AT_DT_ratio A T D = 2) (h2: BT_ET_ratio B T E = 3) :
  CD_BD_ratio C D B = 2 / 9 :=
sorry

end ratio_CD_BD_l26_26001


namespace solve_for_z_l26_26709

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26709


namespace sin_series_positive_l26_26082

theorem sin_series_positive (n : ℕ) (x : ℝ) (h₀ : 0 < x) (h₁ : x < real.pi) :
  (∑ i in finset.range n, (1:ℝ) / (i + 1) * real.sin ((i + 1) * x)) > 0 :=
sorry

end sin_series_positive_l26_26082


namespace minimum_length_MN_l26_26292

theorem minimum_length_MN {A B C D A1 B1 C1 D1 M N : Point} 
(h_edge_len : ∀ (p q : Point), (p = A1 ∨ p = B1 ∨ p = C1 ∨ p = D1) ∧ (q = A ∨ q = B ∨ q = C ∨ q = D) → dist p q = 1)
(h_M_on_diag : ∃ k ∈ Icc 0 1, M = k • A1 + (1 - k) • D)
(h_N_on_CD1 : ∃ l ∈ Icc 0 1, N = l • C + (1 - l) • D1)
(h_MN_parallel : parallel (M - N) (A1 - A + C - C1)) :
  dist M N = (sqrt 3) / 3 :=
begin
  sorry
end

end minimum_length_MN_l26_26292


namespace incorrect_statement_l26_26206

noncomputable def P : ℕ → ℤ
| 0       := 0
| (n + 1) := if (n + 1) % 5 < 3 then P n + 1 else P n - 1

theorem incorrect_statement :
  ¬ (P 101 > P 104) := by
  sorry

end incorrect_statement_l26_26206


namespace main_theorem_l26_26421

variables {m n : ℕ} {x : ℝ}
variables {a : ℕ → ℕ}
noncomputable def relatively_prime (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → Nat.gcd (a i) (a j) = 1

noncomputable def distinct (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → a i ≠ a j

theorem main_theorem (hm : 1 < m) (hn : 1 < n) (hge : m ≥ n)
  (hrel_prime : relatively_prime a n)
  (hdistinct : distinct a n)
  (hbound : ∀ i, i < n → a i ≤ m)
  : ∃ i, i < n ∧ ‖a i * x‖ ≥ (2 / (m * (m + 1))) * ‖x‖ := 
sorry

end main_theorem_l26_26421


namespace solve_imaginary_eq_l26_26757

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26757


namespace total_team_players_l26_26229

-- Conditions
def team_percent_boys : ℚ := 0.6
def team_percent_girls := 1 - team_percent_boys
def junior_girls_count : ℕ := 10
def total_girls := junior_girls_count * 2
def girl_percentage_as_decimal := team_percent_girls

-- Problem
theorem total_team_players : (total_girls : ℚ) / girl_percentage_as_decimal = 50 := 
by 
    sorry

end total_team_players_l26_26229


namespace james_carrot_sticks_l26_26384

theorem james_carrot_sticks (x : ℕ) (h : x + 15 = 37) : x = 22 :=
by {
  sorry
}

end james_carrot_sticks_l26_26384


namespace find_a_l26_26319

noncomputable def set_A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem find_a (a : ℝ) (h : 3 ∈ set_A a) : a = -3 / 2 :=
by
  sorry

end find_a_l26_26319


namespace count_subsets_l26_26614

theorem count_subsets {X : Set ℕ} :
  {T : Set ℕ | {1,2,3} ⊆ T ∧ T ⊆ {1,2,3,4,5,6}}.card = 8 :=
by
  sorry

end count_subsets_l26_26614


namespace gcd_372_684_l26_26119

theorem gcd_372_684 : Int.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l26_26119


namespace complex_solution_l26_26722

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26722


namespace f_six_is_two_main_l26_26659

def f (x : ℝ) : ℝ :=
(if x < 0 then
    x^3 - 1
 else if -1 ≤ x ∧ x ≤ 1 then
    if x ≠ 0 then -f (-x) else 0
 else sorry)  -- Placeholder for the periodicity condition

theorem f_six_is_two :
  (f: ℝ → ℝ) :=
sorry

theorem main : f 6 = 2 :=
begin
  apply f_six_is_two,
  sorry
end

end f_six_is_two_main_l26_26659


namespace expression_eq_zero_option_i_true_option_ii_false_option_iii_true_option_iv_false_l26_26304

theorem expression_eq_zero (x : ℝ) (hx : 0 < x) : x^x - x^x = 0 := by
  sorry

theorem option_i_true (x : ℝ) (hx : 0 < x) : 0 = 0 := by
  refl

theorem option_ii_false (x : ℝ) (hx : 0 < x) : x^x - x^x ≠ x^(x-1) := by
  intro h
  have h1 : x^x - x^x = 0 := by
    exact expression_eq_zero x hx
  have h2 : x^(x-1) ≠ 0 := by
    sorry -- use specific values or properties to show that it's generally not 0
  exact h2 (eq.symm h)

theorem option_iii_true (x : ℝ) (hx : 0 < x) : (x-1)^x = 0 := by
  sorry -- show using specific values like x=1

theorem option_iv_false (x : ℝ) (hx : 0 < x) : (x-1)^(x-1) ≠ 0 := by
  intro h
  have h1 : (x-1)^(x-1) = 1 := by
    sorry -- use specific values like x=2 or general properties to show it's 1
  exact h1.symm (eq.symm h)

end expression_eq_zero_option_i_true_option_ii_false_option_iii_true_option_iv_false_l26_26304


namespace max_value_a3_a8_l26_26295

theorem max_value_a3_a8 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S 10 = 40)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, (∑ i in Finset.range(n+1), a i) = S n) →
  (a 3 + a 8 = 8) →
  (a 3 * a 8 ≤ 16) :=
sorry

end max_value_a3_a8_l26_26295


namespace part1_solution_part2_solution_l26_26825

section Part1
def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem part1_solution : {x : ℝ | f x ≤ x + 5} = Set.Icc (-4/3 : ℝ) (6 : ℝ) := 
sorry
end Part1

section Part2
def f (x : ℕ) : ℝ := abs (x + 1) + abs (x - 2)

theorem part2_solution (a : ℝ) (h : a ≠ 0) : 
  (∀ a ≠ 0, f x ≥ ((abs (a + 1) - abs (3 * a - 1)) / abs a)) → 
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -3/2} ∪ {x : ℝ | x ≥ 5/2} :=
sorry
end Part2

end part1_solution_part2_solution_l26_26825


namespace imaginary_part_of_z_l26_26874

def complex_mul (a b : ℂ) : ℂ := a * b

def imag_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_z : 
  let z := complex_mul (1 - complex.i) (3 + complex.i) in
  imag_part z = -2 :=
by
  sorry

end imaginary_part_of_z_l26_26874


namespace repeating_decimal_division_l26_26918

theorem repeating_decimal_division :
  (0.\overline{54} / 0.\overline{18}) = 3 :=
by
  have h1 : 0.\overline{54} = 54 / 99 := sorry
  have h2 : 0.\overline{18} = 18 / 99 := sorry
  have h3 : (54 / 99) / (18 / 99) = 54 / 18 := sorry
  have h4 : 54 / 18 = 3 := sorry
  rw [h1, h2, h3, h4]
  exact rfl

end repeating_decimal_division_l26_26918


namespace determine_z_l26_26702

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26702


namespace sum_of_valid_m_l26_26647

noncomputable def valid_m (m : ℤ) : Prop :=
  (0 ≤ m ∧ m < 6 ∧ m ≠ 5)

theorem sum_of_valid_m :
  (∑ m in Finset.filter valid_m (Finset.range 6), m) = 10 := by
  sorry

end sum_of_valid_m_l26_26647


namespace percentage_answered_first_correctly_l26_26551

-- Defining the given conditions
def percentage_answered_second_correctly : ℝ := 0.25
def percentage_answered_neither_correctly : ℝ := 0.20
def percentage_answered_both_correctly : ℝ := 0.20

-- Lean statement for the proof problem
theorem percentage_answered_first_correctly :
  ∃ a : ℝ, a + percentage_answered_second_correctly - percentage_answered_both_correctly = 0.80 ∧ a = 0.75 := by
  sorry

end percentage_answered_first_correctly_l26_26551


namespace both_firms_participate_number_of_firms_participate_social_optimality_l26_26538

-- Definitions for general conditions
variable (α V IC : ℝ)
variable (hα : 0 < α ∧ α < 1)

-- Condition for both firms to participate
def condition_to_participate (V : ℝ) (α : ℝ) (IC : ℝ) : Prop :=
  V * α * (1 - 0.5 * α) ≥ IC

-- Part (a): Under what conditions will both firms participate?
theorem both_firms_participate (α V IC : ℝ) (hα : 0 < α ∧ α < 1) :
  condition_to_participate V α IC → (V * α * (1 - 0.5 * α) ≥ IC) :=
by sorry

-- Part (b): Given V=16, α=0.5, and IC=5, determine the number of firms participating
theorem number_of_firms_participate :
  (condition_to_participate 16 0.5 5) :=
by sorry

-- Part (c): To determine if the number of participating firms is socially optimal
def total_profit (α V IC : ℝ) (both : Bool) :=
  if both then 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
  else α * V - IC

theorem social_optimality :
   (total_profit 0.5 16 5 true ≠ max (total_profit 0.5 16 5 true) (total_profit 0.5 16 5 false)) :=
by sorry

end both_firms_participate_number_of_firms_participate_social_optimality_l26_26538


namespace determine_z_l26_26692

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26692


namespace triangle_angle_relationship_l26_26395

theorem triangle_angle_relationship
  (A B C P : Point)
  (BC AC AB : ℝ)
  (h1 : BC = AC + 0.5 * AB)
  (h2 : ℝ)
  (AP PB : ℝ)
  (h3 : AP = 3 * PB)
  (on_segment : P ∈ openSegment A B)
  (angle_PAC angle_CPA : Real)
  (angle_eq : ∠P A C = 2 * ∠C P A) :
  ∠P A C = 2 * ∠C P A :=
sorry

end triangle_angle_relationship_l26_26395


namespace incenter_of_ABD_l26_26816

variable (A B C D I : Point)
variable [cyclic_quadrilateral A B C D]

-- Angle bisectors meet at point I
variable (AI_bisects_BAD : angle_bisector A B D I)
variable (CI_bisects_BCD : angle_bisector B C D I)

-- Given angle BIC = angle IDC
variable (angle_BIC_eq_angle_IDC : ∠ B I C = ∠ I D C)

-- We need to prove that I is the incenter of triangle ABD
theorem incenter_of_ABD : is_incenter I A B D :=
by 
  sorry

end incenter_of_ABD_l26_26816


namespace division_of_repeating_decimals_l26_26906

noncomputable def repeating_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  ⟨n, d⟩

theorem division_of_repeating_decimals :
  let x := repeating_to_fraction 54 99
  let y := repeating_to_fraction 18 99
  (x / y) = (3 : ℚ) :=
by
  -- Proof omitted as requested
  sorry

end division_of_repeating_decimals_l26_26906


namespace percentage_of_green_ducks_in_larger_pond_l26_26351

theorem percentage_of_green_ducks_in_larger_pond
  (total_smaller_pond : ℕ)
  (total_larger_pond : ℕ)
  (percent_green_smaller_pond : ℕ)
  (overall_percent_green : ℕ)
  (total_smaller : total_smaller_pond = 45)
  (total_larger : total_larger_pond = 55)
  (percent_green_smaller : percent_green_smaller_pond = 20)
  (percent_green_overall : overall_percent_green = 31) :
  let total_ducks := total_smaller_pond + total_larger_pond in
  let green_smaller_pond := percent_green_smaller_pond * total_smaller_pond / 100 in
  let total_green_ducks := overall_percent_green * total_ducks / 100 in
  let green_larger_pond := total_green_ducks - green_smaller_pond in
  green_larger_pond * 100 / total_larger_pond = 40 := by
  sorry

end percentage_of_green_ducks_in_larger_pond_l26_26351


namespace quadratic_roots_eq_3_sqrt_2_l26_26222

theorem quadratic_roots_eq_3_sqrt_2 :
  ∃ x : ℝ, x^2 - 6 * x * real.sqrt 2 + 18 = 0 ∧ x = 3 * real.sqrt 2 :=
sorry

end quadratic_roots_eq_3_sqrt_2_l26_26222


namespace secret_santa_probability_l26_26889

theorem secret_santa_probability
  (n : ℕ)
  (h1 : n = 101)
  (h2 : ∀ i j : ℕ, i ≠ j → i.gifts ≠ j.gifts)
  (h3 : ∀ i: ℕ, ¬ i.gifts = i) : 
  (prob_first_person_not_involved (n : ℕ) : ℝ) :=
begin
  have h_approx_derangements : ∀ k, D k ≈ k! / Real.exp 1,
  sorry,
  let probability := 1 - (4 * (D 100 - D 97) / D 101),
  have h_probability : probability = 0.96039,
  sorry
end

end secret_santa_probability_l26_26889


namespace least_value_of_N_l26_26519

theorem least_value_of_N : ∃ (N : ℕ), (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧ N = 59 :=
by
  sorry

end least_value_of_N_l26_26519


namespace b_50_is_6_over_20610_l26_26285

-- Define the sequence b
def b : ℕ → ℚ
| 1     := 2
| (n+2) := let T_n := finset.sum (finset.range (n + 1)) b in
           3 * T_n^2 / (3 * T_n - 2)

-- Define the sum of the first n terms of the sequence
def T (n : ℕ) : ℚ := finset.sum (finset.range n) b

-- The theorem to prove that b_50 = 6 / 20610
theorem b_50_is_6_over_20610 :
  b 50 = 6 / 20610 :=
sorry

end b_50_is_6_over_20610_l26_26285


namespace goose_survival_fraction_l26_26432

theorem goose_survival_fraction
  (E : ℝ) 
  (h1 : E = 550.0000000000001)
  (h2 : ∀ h, h = (2 / 3) * E) 
  (h3 : ∀ F, ∀ s, s = F * (2 / 3) * E)
  (h4 : ∀ F s, ∀ f, f = (2 / 5) * s) 
  (h5 : ∀ f, f = 110) :
  ∃ F, F = 0.75 :=
by
  sorry

end goose_survival_fraction_l26_26432


namespace factor_polynomial_l26_26258

theorem factor_polynomial (a b : ℕ) : 
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2 * b) * (2 * a - b) :=
by sorry

end factor_polynomial_l26_26258


namespace system_solution_a_l26_26093

theorem system_solution_a (x y z : ℤ) (h1 : x^2 + x * y + y^2 = 7) (h2 : y^2 + y * z + z^2 = 13) (h3 : z^2 + z * x + x^2 = 19) :
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = -2 ∧ y = -1 ∧ z = -3) :=
sorry

end system_solution_a_l26_26093


namespace find_b_l26_26359

theorem find_b
  (a b c : ℝ)
  (h1 : a = 4)
  (h2 : c = 5)
  (sin_A : ℝ)
  (h3 : sin_A = sqrt 7 / 4)
  (h4 : b > a)
  (cos_A : ℝ)
  (h5 : cos_A = 3 / 4) :
  b = 6 := 
by sorry

end find_b_l26_26359


namespace pears_left_l26_26387

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) : 
  jason_pears + keith_pears - mike_ate = 81 := 
by 
  sorry

end pears_left_l26_26387


namespace estimated_height_correct_l26_26503

-- Given conditions
def sum_x : ℕ := 225
def sum_y : ℕ := 1600
def n : ℕ := 10
def b : ℝ := 4.0
def x_val : ℝ := 24.0
def sample_mean {T : Type} [Add T] [Div T] (sum : ℕ) (count : ℕ) : T :=
  (sum : T) / (count : T)

-- Linear regression related computations
def mean_x := sample_mean sum_x n
def mean_y := sample_mean sum_y n
def a : ℝ := mean_y - b * mean_x

-- The final estimated height
def estimated_y : ℝ := b * x_val + a

theorem estimated_height_correct :
  estimated_y = 166 :=
by
  sorry

end estimated_height_correct_l26_26503


namespace sequence_remainder_mod_10_l26_26626

def T : ℕ → ℕ := sorry -- Since the actual recursive definition is part of solution steps, we abstract it.
def remainder (n k : ℕ) : ℕ := n % k

theorem sequence_remainder_mod_10 (n : ℕ) (h: n = 2023) : remainder (T n) 10 = 6 :=
by 
  sorry

end sequence_remainder_mod_10_l26_26626


namespace intersection_complement_M_N_intersection_M_P_l26_26669

def set_M := {x : ℝ | x > 1}
def set_N := {y : ℝ | ∃ x : ℝ, y = 2 * x^2}
def set_P := {p : ℝ × ℝ | p.2 = p.1 - 1}

def complement_set_M := {x : ℝ | x <= 1}

theorem intersection_complement_M_N :
  (complement_set_M ∩ set_N = {x : ℝ | 0 ≤ x ∧ x ≤ 1}) := 
by sorry

theorem intersection_M_P :
  set_M ∩ set_P = ∅ := 
by sorry

end intersection_complement_M_N_intersection_M_P_l26_26669


namespace selection_representatives_l26_26888

theorem selection_representatives 
  (boys : Finset ℕ) (girls : Finset ℕ) (mandatory_girl : ℕ)
  (Hboys : boys.card = 4) 
  (Hgirls : girls.card = 5) 
  (Hmandatory_girl_girls : mandatory_girl ∈ girls) : 
  (Finset.choose 2 boys).card * (Finset.choose 2 (girls \ {mandatory_girl})).card = 36 := 
by 
  sorry

end selection_representatives_l26_26888


namespace local_maximum_of_f_l26_26595

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - 9 * x

theorem local_maximum_of_f : 
  (-2 < x ∧ x < 2) → (x = -1 → f x = 5) := by 
  intro hx h1
  have h_critical : (f' x = 0) 
  sorry

  have h2 : x = -1 
  sorry

  have h_second_derivative : (f'' x > 0)
  sorry

  exact h1

end local_maximum_of_f_l26_26595


namespace num_proper_subsets_l26_26477

theorem num_proper_subsets : 
  ∀ (s : Finset ℕ), s = {0, 3, 4} → (s.powerset.filter (λ t, t ≠ s)).card = 7 :=
by
  intros s hs
  rw hs
  sorry

end num_proper_subsets_l26_26477


namespace gumballs_difference_l26_26235

variable (x y : ℕ)

def total_gumballs := 16 + 12 + 20 + x + y
def avg_gumballs (T : ℕ) := T / 5

theorem gumballs_difference (h1 : 18 <= avg_gumballs (total_gumballs x y)) 
                            (h2 : avg_gumballs (total_gumballs x y) <= 27) : (87 - 42) = 45 := by
  sorry

end gumballs_difference_l26_26235


namespace problem1_problem2_l26_26850

-- Proof Problem 1: Prove m = 3/2 under given conditions
theorem problem1 (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - 1|) :
  (∀ x, f (x + 1/2) ≤ 2 * (3/2) + 1 ↔ x ∈ (-∞, -2] ∪ [2, ∞)) :=
sorry

-- Proof Problem 2: Prove a = 4 under given conditions
theorem problem2 (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - 1|) (a : ℝ) :
  (∀ x y, f x ≤ 2 ^ y + a / 2 ^ y + |2 * x + 3|) → a ≥ 4 :=
sorry

end problem1_problem2_l26_26850


namespace trapezoid_equal_points_l26_26819

-- Definitions for the problem
variables (A B C D P Q R : Type)

-- Define the parallel, line, and point predicates (these should be provided by Mathlib)
def Trapezoid (A B C D : Type) (AB CD : Type) : Prop := (AB ∥ CD)
def IsPointOnLine (P : Type) (BC : Type) : Prop := true -- placeholder
def LineParallel (AP : Type) (l : Type) : Prop := true -- placeholder

-- The proof statement
theorem trapezoid_equal_points (A B C D P Q R BC AD AP DP : Type)
  (h_trap : Trapezoid A B C D (Line A B) (Line C D))
  (h_point : IsPointOnLine P BC)
  (h_Q : ∃ (Q : Type), LineParallel (Line A P) (Line C Q) ∧ IsPointOnLine Q AD)
  (h_R : ∃ (R : Type), LineParallel (Line D P) (Line B R) ∧ IsPointOnLine R AD)
  : Q = R := sorry

end trapezoid_equal_points_l26_26819


namespace sam_dimes_example_l26_26451

theorem sam_dimes_example (x y : ℕ) (h₁ : x = 9) (h₂ : y = 7) : x + y = 16 :=
by 
  sorry

end sam_dimes_example_l26_26451


namespace num_arithmetic_sequences_l26_26650

theorem num_arithmetic_sequences :
  ∃ (a d n : ℕ), n ≥ 3 ∧ (n * a + n * (n - 1) * d = 97^2) ∧
  ((∃ (a d : ℕ), n = 1 ∧ n * a + n * (n - 1) * d = 97^2) ∨
   (∃ (a d : ℕ), n = 2 ∧ n * a + n * (n-1) * d = 97^2)) →
   ((a = 1 ∧ d = 0 ∧ n = 97^2) ∨
    (a = 97 ∧ d = 0 ∧ n = 97) ∨
    (a = 49 ∧ d = 1 ∧ n = 97) ∨
    (a = 1 ∧ d = 2 ∧ n = 97)) → 4 :=
begin
  sorry
end

end num_arithmetic_sequences_l26_26650


namespace sum_of_roots_eq_12_l26_26618

theorem sum_of_roots_eq_12 :
  let f := λ x : ℝ, 4 * x^2 - 58 * x + 190
  let g := λ x : ℝ, (29 - 4 * x - Real.log (x) / Real.log 2) * (Real.log x / Real.log 2)
  ∃ x1 x2 : ℝ, (f x1 = g x1) ∧ (f x2 = g x2) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 12) :=
sorry

end sum_of_roots_eq_12_l26_26618


namespace neighboring_squares_difference_l26_26219

theorem neighboring_squares_difference {n : ℕ} (h : n ≥ 2) (board : Fin n → Fin n → ℕ) 
  (unique_numbers : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2 ∧ 
    (∀ a b c d, (a ≠ i ∨ b ≠ j) → board a b ≠ board c d)) : 
  ∃ i j i' j', (|board i' j' - board i j| ≥ n) ∧ 
    ((i = i' ∧ (j = j'.succ ∨ j = j'.pred)) ∨ (j = j' ∧ (i = i'.succ ∨ i = i'.pred))) :=
by 
  intro i j i' j'
  sorry

end neighboring_squares_difference_l26_26219


namespace find_k_l26_26969

theorem find_k :
  (∀ k : ℝ, ∃ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
     (c not_eq 0) → 
    (10 - k) / 8 = (k - 6) / 8 → k = 8) := sorry

end find_k_l26_26969


namespace angle_FOG_eq_180_minus_2_angle_BAC_l26_26226

variable {α : Type*} [ordered_ring α] 

-- Define the geometric objects involved
variables (A B C D F G O A' : α → α)
variable (ω : α → α)

-- Define the conditions
def is_acute_angled_triangle (A B C : α) : Prop :=
  -- Definition that ABC is an acute-angled triangle
  sorry

def circumcircle (triangle : α → α) (O : α) : α → α :=
  -- Definition that ω is the circumcircle of the triangle
  sorry

def diametrically_opposite (ω : α → α) (A A' : α) : Prop :=
  -- Definition that A' is diametrically opposite to A on ω
  sorry

def minor_arc_point (ω : α → α) (B C D : α) : Prop :=
  -- Definition that D is on the minor arc BC of ω
  sorry

-- The final theorem to prove
theorem angle_FOG_eq_180_minus_2_angle_BAC (A B C O D F G A' : α) (ω : α → α) :
  is_acute_angled_triangle A B C ∧ 
  circumcircle (λ x, x) O = ω ∧ 
  diametrically_opposite ω A A' ∧ 
  minor_arc_point ω B C D →
  ∠ FOG = 180 - 2 * ∠ BAC := 
sorry

end angle_FOG_eq_180_minus_2_angle_BAC_l26_26226


namespace cylindrical_tin_height_l26_26468

noncomputable def cylinder_height_approx (d V : ℝ) : ℝ :=
  let r := d / 2 in
  V / (π * r^2)

theorem cylindrical_tin_height :
  let d := 14
  let V := 245
  cylinder_height_approx d V ≈ 1.59155 := by
  simp [cylinder_height_approx]
  sorry

end cylindrical_tin_height_l26_26468


namespace polynomial_m_n_values_l26_26371

theorem polynomial_m_n_values :
  ∀ (m n : ℝ), ((x - 1) * (x + m) = x^2 - n * x - 6) → (m = 6 ∧ n = -5) := 
by
  intros m n h
  sorry

end polynomial_m_n_values_l26_26371


namespace inverse_function_log_base_2_l26_26876

theorem inverse_function_log_base_2 (x : ℝ) (hx : x > 0) : 
  (∃ y : ℝ, y = 1 + log x / log 2) ↔ (∀ y : ℝ, y = 2^(x - 1)) :=
by
  sorry

end inverse_function_log_base_2_l26_26876


namespace complex_number_quadrant_l26_26482

theorem complex_number_quadrant :
  let z := (2 - complex.i) / (3 * complex.i - 1)
  Re(z) < 0 ∧ Im(z) < 0 := 
by
  let z := (2 - complex.i) / (3 * complex.i - 1)
  sorry

end complex_number_quadrant_l26_26482


namespace complex_solution_l26_26727

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26727


namespace find_five_numbers_l26_26494

variable (a b c d e : Nat)
variable (f : Fin 5 → Nat)

theorem find_five_numbers : 
  (let S := a + b + c + d + e in
    (S - a = 44 ∨ S - a = 45 ∨ S - a = 46 ∨ S - a = 47) ∧
    (S - b = 44 ∨ S - b = 45 ∨ S - b = 46 ∨ S - b = 47) ∧
    (S - c = 44 ∨ S - c = 45 ∨ S - c = 46 ∨ S - c = 47) ∧
    (S - d = 44 ∨ S - d = 45 ∨ S - d = 46 ∨ S - d = 47) ∧
    (S - e = 44 ∨ S - e = 45 ∨ S - e = 46 ∨ S - e = 47)) →
  (a = 13 ∧ b = 12 ∧ c = 11 ∧ d = 10 ∧ e = 11) :=
begin 
  sorry
end

end find_five_numbers_l26_26494


namespace avg_percentage_diff_l26_26254

-- Define the segment counts for each type of rattlesnake
def eastern_segments_male := 6
def eastern_segments_female := 7

def western_segments_male := 8
def western_segments_female := 10

def southern_segments_male := 7
def southern_segments_female := 8

def northern_segments_male := 9
def northern_segments_female := 11

-- Calculate the percentage differences for each comparison
def percentage_diff (a b : ℝ) : ℝ := ((b - a) / b) * 100

def eastern_male_diff := percentage_diff eastern_segments_male western_segments_male
def eastern_female_diff := percentage_diff eastern_segments_female western_segments_female

def southern_male_diff := percentage_diff southern_segments_male western_segments_male
def southern_female_diff := percentage_diff southern_segments_female western_segments_female

def northern_male_diff := percentage_diff northern_segments_male western_segments_male
def northern_female_diff := percentage_diff northern_segments_female western_segments_female

-- Calculate average percentage differences
def avg_male_diff := (eastern_male_diff + southern_male_diff + northern_male_diff) / 3
def avg_female_diff := (eastern_female_diff + southern_female_diff + northern_female_diff) / 3

def overall_avg_diff := (avg_male_diff + avg_female_diff) / 2

-- The theorem statement
theorem avg_percentage_diff : overall_avg_diff ≈ 18.335 := 
sorry

end avg_percentage_diff_l26_26254


namespace seven_digit_numbers_multiple_of_33_l26_26980

theorem seven_digit_numbers_multiple_of_33 :
  let count_valid_numbers (m : ℕ) : ℕ :=
    -- Function to count the valid numbers for given 'm'.
    (Finset.filter
      (λ n, (let A := (n / 10^5) % 10, B := (n / 10^3) % 10, C := (n / 10) % 10 in
             -- Alternating sum divisible by 11
             (m + A + B + C - 9) % 11 = 0 ∧
             -- Total sum divisible by 3
             (m + A + B + 9 + C) % 3 = 0))
      (Finset.range 1000000)).card in
    count_valid_numbers 2 - count_valid_numbers 3 = 8 := sorry

end seven_digit_numbers_multiple_of_33_l26_26980


namespace max_k_value_l26_26651

theorem max_k_value (x0 x1 x2 x3 : ℝ) (h0 : x0 > 0) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x0 > x1) (h5 : x1 > x2) (h6 : x2 > x3) :
  log x0 / log (x0 / x1) + log x0 / log (x1 / x2) + log x0 / log (x2 / x3) ≥ 3 * log 1993 / log x0 := 
sorry

end max_k_value_l26_26651


namespace division_of_repeating_decimals_l26_26907

noncomputable def repeating_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  ⟨n, d⟩

theorem division_of_repeating_decimals :
  let x := repeating_to_fraction 54 99
  let y := repeating_to_fraction 18 99
  (x / y) = (3 : ℚ) :=
by
  -- Proof omitted as requested
  sorry

end division_of_repeating_decimals_l26_26907


namespace complex_solution_l26_26729

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26729


namespace problem_min_max_product_l26_26834

theorem problem_min_max_product (x y : ℝ) (h : 3 * x^2 + 6 * x * y + 4 * y^2 = 1) :
  let a := (3 * x^2 + 4 * x * y + 3 * y^2).inf
  let b := (3 * x^2 + 4 * x * y + 3 * y^2).sup in
  a * b = 4 / 9 :=
by
  sorry -- proof placeholder

end problem_min_max_product_l26_26834


namespace carry_forward_probability_is_088_l26_26335

noncomputable def isCarryForwardNumber (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  (sum % 10) < (sum / 10) * 10

noncomputable def carryForwardNumbersProbability : ℚ :=
  let total := 100
  let carryForwardCount := (Finset.range 100).count isCarryForwardNumber
  carryForwardCount / total

theorem carry_forward_probability_is_088 :
  carryForwardNumbersProbability = 88 / 100 :=
by
  sorry

end carry_forward_probability_is_088_l26_26335


namespace EF_parallel_MN_l26_26012

open EuclideanGeometry

theorem EF_parallel_MN
  (A B C D E F M N : Point)
  (hABC_acute : acute_triangle A B C)
  (hD : altitude A D B C)
  (hGamma1 : ∃ Γ1, is_circle Γ1 ∧ diameter Γ1 A B ∧ intersects Γ1 A C F)
  (hGamma2 : ∃ Γ2, is_circle Γ2 ∧ diameter Γ2 A C ∧ intersects Γ2 A B E)
  (hO : ∃ Ω, is_circle Ω ∧ diameter Ω A D ∧ intersects Ω A B M ∧ intersects Ω A C N) :
  parallel E F M N := 
begin
  sorry
end

end EF_parallel_MN_l26_26012


namespace original_radius_of_cylinder_l26_26017

theorem original_radius_of_cylinder (r z : ℝ) (h : ℝ := 3) :
  z = 3 * π * ((r + 8)^2 - r^2) → z = 8 * π * r^2 → r = 8 :=
by
  intros hz1 hz2
  -- Translate given conditions into their equivalent expressions and equations
  sorry

end original_radius_of_cylinder_l26_26017


namespace complex_pow_test_l26_26291

noncomputable def z (θ : ℝ) : ℂ := ℂ.exp (θ * complex.I)

theorem complex_pow_test (z : ℂ)
  (h : z + 1/z = 2 * Real.cos (Real.pi / 4)) :
  z^12 + 1/z^12 = -2 :=
sorry

end complex_pow_test_l26_26291


namespace smallest_rational_correct_l26_26994

def rational_set : set ℚ := {-1, 0, 1, 2}

noncomputable def smallest_rational (s : set ℚ) : ℚ :=
  if h : s.nonempty then classical.some (set.exists_mem_of_nonempty h)
  else 0

theorem smallest_rational_correct : smallest_rational rational_set = -1 :=
by {
  have h : rational_set = {-1, 0, 1, 2} := rfl,
  rw h,
  simp [smallest_rational, classical.some, set.exists_mem_of_nonempty],
  sorry
}

end smallest_rational_correct_l26_26994


namespace problem1_solutionset_problem2_minvalue_l26_26305

noncomputable def f (x : ℝ) : ℝ := 45 * abs (2 * x - 1)
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem problem1_solutionset :
  {x : ℝ | 0 < x ∧ x < 2 / 3} = {x : ℝ | f x + abs (x + 1) < 2} :=
by
  sorry

theorem problem2_minvalue (a : ℝ) (m n : ℝ) (h : m + n = a ∧ m > 0 ∧ n > 0) :
  a = 2 → (4 / m + 1 / n) ≥ 9 / 2 :=
by
  sorry

end problem1_solutionset_problem2_minvalue_l26_26305


namespace solve_imaginary_eq_l26_26766

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26766


namespace complex_arithmetic_l26_26780

def A : ℂ := 5 - 2 * complex.I
def M : ℂ := -3 + 2 * complex.I
def S : ℂ := 2 * complex.I
def P : ℝ := 3

theorem complex_arithmetic : 2 * (A - M + S - P) = 10 - 4 * complex.I := by
  sorry

end complex_arithmetic_l26_26780


namespace television_probability_l26_26631

theorem television_probability:
  let total_tv := 5 in
  let typeA := 3 in
  let typeB := 2 in
  let choose (n k : Nat) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) in
  let ways_to_choose_3 := choose total_tv 3 in
  let ways_to_choose_2A1B := choose typeA 2 * choose typeB 1 in
  let ways_to_choose_1A2B := choose typeA 1 * choose typeB 2 in
  let favorable_outcomes := ways_to_choose_2A1B + ways_to_choose_1A2B in
  let probability := favorable_outcomes / ways_to_choose_3 in
  probability = 9 / 10 :=
by sorry

end television_probability_l26_26631


namespace symmetric_point_P_l26_26081

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the function to get the symmetric point with respect to the origin
def symmetric_point (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, -point.2)

-- State the theorem that proves the symmetric point of P is (-1, 2)
theorem symmetric_point_P :
  symmetric_point P = (-1, 2) :=
  sorry

end symmetric_point_P_l26_26081


namespace determine_constants_l26_26581

theorem determine_constants (a b c : ℝ) (h_asymp1 : b * (-7 * π / 3) + c = -2 * π) 
                           (h_asymp2 : b * (5 * π / 3) + c = -m * π) 
                           (h_min_positive : ∃ (x : ℝ), x = 0 ∧ y = a * csc (b * x + c) = 2) : 
  a = -√3 ∧ c = -π / 3 := by
  sorry

end determine_constants_l26_26581


namespace total_fireworks_l26_26025

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end total_fireworks_l26_26025


namespace coefficient_sum_l26_26824

theorem coefficient_sum (a_n : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → 
  (a_n n = (nat.choose n 2) * 3 ^ (n - 2))) :
  (∑ n in Finset.range 17, 3 ^ (n + 2) / a_n (n + 2)) = 17 := 
by
  sorry

end coefficient_sum_l26_26824


namespace find_max_t_l26_26641

noncomputable def good_set (ω : Circle) (T : set (Triangle ℝ)) : Prop :=
  (∀ t ∈ T, t.circumcircle = ω) ∧ (∀ t1 t2 ∈ T, t1 ≠ t2 → disjoint (t1.interior) (t2.interior))

theorem find_max_t (ω : Circle := circle (1 : ℝ)) :
  ∃ t > 0, ∀ (n : ℕ), ∃ T : set (Triangle ℝ), (good_set ω T) ∧ (T.card = n) ∧ (∀ t ∈ T, t.perimeter > 6) :=
by
  sorry

end find_max_t_l26_26641


namespace number_of_divisions_l26_26997

-- Definitions
def hour_in_seconds : ℕ := 3600

def is_division (n m : ℕ) : Prop :=
  n * m = hour_in_seconds ∧ n > 0 ∧ m > 0

-- Proof problem statement
theorem number_of_divisions : ∃ (count : ℕ), count = 44 ∧ 
  (∀ (n m : ℕ), is_division n m → ∃ (d : ℕ), d = count) :=
sorry

end number_of_divisions_l26_26997


namespace problem_proof_l26_26124

-- The problem statement translated to a Lean definition
def probability_odd_sum_rows_columns := 1 / 14

theorem problem_proof :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let grid := finset.powersetLen 9 nums -- The set of all ways to fill the grid
  (∃ g ∈ grid, 
   ∀ row ∈ (fin 3), 
   odd (sum (g row)) ∧ 
   ∀ col ∈ (fin 3),
   odd (sum (λ (x : ℕ), g x col))) → real.ratDiv 1 14 :=
by
  sorry

end problem_proof_l26_26124


namespace integral_proof_l26_26587

noncomputable def integral_solution : Prop :=
  ∫ (x : ℝ) in 0..1, ((x^3 - 3*x^2 - 12) / ((x - 4)*(x - 3)*x)) dx = 
  (λ x, x + Real.log (|x - 4|) + 4 * Real.log(|x - 3|) - Real.log (|x|) + C)

theorem integral_proof : integral_solution :=
by
  sorry

end integral_proof_l26_26587


namespace ratio_of_ages_in_two_years_l26_26199

theorem ratio_of_ages_in_two_years (S M : ℕ) 
  (h1 : M = S + 37) 
  (h2 : S = 35) : 
  (M + 2) / (S + 2) = 2 := 
by 
  -- We skip the proof steps as instructed
  sorry

end ratio_of_ages_in_two_years_l26_26199


namespace solve_for_y_l26_26090
open Real

theorem solve_for_y (y : ℝ) (h : 5^y + 18 = 4 * 5^y - 40) : 
  y = log (5 : ℝ) (58 / 3) := 
sorry

end solve_for_y_l26_26090


namespace LCM_of_fractions_l26_26513

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l26_26513


namespace determine_z_l26_26694

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26694


namespace max_area_trapezoid_l26_26594

-- Definitions of constants and conditions
variable (α β : ℝ)
variable (s : ℝ)
variable (trapezoid : Type)
variable (Area : trapezoid → ℝ)
variable (fixed_angles : ∀ t : trapezoid, (∠BAD t = α ∧ ∠ABC t = β))
variable (constant_sum : ∀ t : trapezoid, (length (BA t) + length (AD t) = s))

-- The trapezoid construction that matches the geometric criteria
structure MaxAreaTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ
  angle_alpha : ℝ
  angle_beta : ℝ

-- The definition of the problem stating the constructed trapezoid has the maximum area
theorem max_area_trapezoid (t : trapezoid) : 
  fixed_angles t → constant_sum t → 
  ∃ T : MaxAreaTrapezoid, Area T ≥ Area t := 
by 
  -- Placeholder proof
  sorry

end max_area_trapezoid_l26_26594


namespace find_z_l26_26751

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26751


namespace brownies_left_l26_26143

theorem brownies_left (initial : ℕ) (tina_ate : ℕ) (husband_ate : ℕ) (shared : ℕ) 
                      (h_initial : initial = 24)
                      (h_tina : tina_ate = 10)
                      (h_husband : husband_ate = 5)
                      (h_shared : shared = 4) : 
  initial - tina_ate - husband_ate - shared = 5 :=
by
  rw [h_initial, h_tina, h_husband, h_shared]
  exact Nat.sub_sub_sub_cancel 24 10 5 4 sorry

end brownies_left_l26_26143


namespace f_range_2_5_l26_26826

noncomputable def g : ℝ → ℝ := sorry
def f (x : ℝ) : ℝ := x + g(x)

-- Conditions
axiom g_periodic : ∀ x : ℝ, g(x) = g(x + 1)

axiom f_range_3_4 : set.range (λ (x : ℝ) (h : x ∈ Icc 3 4), f x) = set.Icc (-2) 5

-- Prove
theorem f_range_2_5 : set.range (λ (x : ℝ) (h : x ∈ Icc 2 5), f x) = set.Icc (-3) 6 := 
sorry

end f_range_2_5_l26_26826


namespace initial_position_is_minus_one_l26_26556

def initial_position_of_A (A B C : ℤ) : Prop :=
  B = A - 3 ∧ C = B + 5 ∧ C = 1 ∧ A = -1

theorem initial_position_is_minus_one (A B C : ℤ) (h1 : B = A - 3) (h2 : C = B + 5) (h3 : C = 1) : A = -1 :=
  by sorry

end initial_position_is_minus_one_l26_26556


namespace repeating_decimal_division_l26_26915

theorem repeating_decimal_division :
  let x := 0 + 54 / 99 in -- 0.545454... = 54/99 = 6/11
  let y := 0 + 18 / 99 in -- 0.181818... = 18/99 = 2/11
  x / y = 3 :=
by
  sorry

end repeating_decimal_division_l26_26915


namespace value_of_expression_l26_26517

theorem value_of_expression : 2 - (-2 : ℝ) ^ (-2 : ℝ) = 7 / 4 := 
by 
  sorry

end value_of_expression_l26_26517


namespace solve_for_z_l26_26731

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26731


namespace not_true_B_l26_26246

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem not_true_B (x y : ℝ) : 2 * star x y ≠ star (2 * x) (2 * y) := by
  sorry

end not_true_B_l26_26246


namespace find_z_l26_26746

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26746


namespace hyperbola_properties_l26_26643

noncomputable def hyperbola_asymptote_equation : Prop := 
  ∃ (λ : ℝ), λ ≠ 0 ∧ ∀ (x y : ℝ), 
  (y = (4 / 3) * x ∨ y = -(4 / 3) * x) → 
  y^2 - (16 / 9) * x^2 = λ

noncomputable def passes_through_P (x y : ℝ) : Prop :=
  y^2 - (16 / 9) * x^2 = -16 ∧ (x = -3 * real.sqrt 2 ∧ y = 4)

noncomputable def hyperbola_standard_equation : Prop := 
  ∀ (x y : ℝ), (x / 3)^2 - (y / 4)^2 = 1

noncomputable def cos_angle_F1PF2 (d1 d2 : ℝ) : Prop :=
  (d1 - d2).abs = 6 ∧ d1 * d2 = 41 ∧ 
  let c := 5 in 
  c^2 = d1^2 + d2^2 - 2 * d1 * d2 * 
  (cos (⟪⟨d1, -d2⟩, ⟨d2, -d1⟩⟫ / (real.sqrt (d1^2 + d2^2) * real.sqrt (d1^2 + d2^2))))

theorem hyperbola_properties :
  hyperbola_asymptote_equation ∧ passes_through_P (-3 * real.sqrt 2) 4 → 
  hyperbola_standard_equation ∧ ∃ (d1 d2 : ℝ), cos_angle_F1PF2 d1 d2 ∧ 
  (cos (⟪⟨d1, 4⟩, ⟨d2, 4⟩⟫ / (real.sqrt (d1^2 + d2^2 + 16) * real.sqrt (d1^2 + d2^2 + 16))) = 14 / 41) :=
sorry

end hyperbola_properties_l26_26643


namespace find_value_of_squares_l26_26039

-- Defining the conditions
variable (a b c : ℝ)
variable (h1 : a^2 + 3 * b = 10)
variable (h2 : b^2 + 5 * c = 0)
variable (h3 : c^2 + 7 * a = -21)

-- Stating the theorem to prove the desired result
theorem find_value_of_squares : a^2 + b^2 + c^2 = 83 / 4 :=
   sorry

end find_value_of_squares_l26_26039


namespace complex_solution_l26_26723

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26723


namespace smallest_c_for_inequality_l26_26155

theorem smallest_c_for_inequality (x : ℕ) (h : x = 9 * 3) : ∃ c : ℤ, x^c > 3^24 ∧ c = 9 := by
  use 9
  rw [h, pow_mul, pow_mul]
  -- rest of the proof goes here
  sorry

end smallest_c_for_inequality_l26_26155


namespace find_ab_of_arithmetic_sequence_l26_26607

theorem find_ab_of_arithmetic_sequence :
  ∃ a b : ℕ, a ≤ 2000 ∧ 2000 ≤ b ∧ 2 + (b * (b + 1)) = 2 * (a * (a + 1)) :=
by
  use 1477
  use 2089
  split
  . exact Nat.le_of_lt (1477 < 2000)
  split
  . exact Nat.le_of_lt (2000 < 2089)
  . unfold Nat.pow
  calc
    2 + (2089 * (2089 + 1))
      = 2 + 2089 * 2090        : rfl -- definition of b(b+1)
  ... = 2 + 4365700            : rfl -- actual calculation
  ... = 2 * (1477 * (1477 + 1)) : rfl -- definition of a(a+1)
  ... = 2 * 1477 * 1478        : rfl -- simplification
  ... = 2 * 2182006            : rfl -- actual calculation
  ... = 4365700                : rfl -- final equivalence

end find_ab_of_arithmetic_sequence_l26_26607


namespace NorbsAgeIs47_l26_26840

noncomputable def norbs_age (guesses : List ℕ) : ℕ :=
  let candidates := [47]
  if 1 ≤ (guesses.count (<m 47)) = guesses.length / 2 then
    47 -- manually check considering only the prime candidate 47 that satisfies all the conditions
  else
    by sorry

-- Define the main theorem
theorem NorbsAgeIs47 : norbs_age [25, 29, 33, 35, 37, 39, 42, 45, 48, 50] = 47 :=
by
  -- Our proof is omitted. We state the equivalence:
  sorry

end NorbsAgeIs47_l26_26840


namespace find_extra_factor_l26_26963

theorem find_extra_factor (w : ℕ) (h1 : w > 0) (h2 : w = 156) (h3 : ∃ (k : ℕ), (2^5 * 13^2) ∣ (936 * w))
  : 3 ∣ w := sorry

end find_extra_factor_l26_26963


namespace probability_all_six_black_l26_26959

/-- A box contains 8 white balls and 7 black balls. Six balls are drawn out of the box at random.
    Prove that the probability that all six balls are black is 1/715. -/
theorem probability_all_six_black (total_white total_black drawn : ℕ) (h_w : total_white = 8) (h_b : total_black = 7) (h_d : drawn = 6) :
  let total_ways := Nat.choose (total_white + total_black) drawn
  let black_ways := Nat.choose total_black drawn
  let probability := black_ways / total_ways
  probability = 1 / 715 :=
by
  simp [total_ways, black_ways]
  sorry

end probability_all_six_black_l26_26959


namespace inequality_check_l26_26640

noncomputable def a : ℝ := 3 ^ 0.4
noncomputable def b : ℝ := 0.4 ^ 3
noncomputable def c : ℝ := Real.logBase 0.4 3

theorem inequality_check : c < b ∧ b < a := by
  sorry

end inequality_check_l26_26640


namespace profit_percentage_A_l26_26561

noncomputable def CP_A : ℝ := 112.5
noncomputable def SP_C : ℝ := 225
noncomputable def profit_B (CP_B : ℝ) : ℝ := 1.25 * CP_B

theorem profit_percentage_A (hC : SP_C = 225)
                           (hA : CP_A = 112.5)
                           (hB : ∀ CP_B, profit_B CP_B = SP_C) :
  (180 - CP_A) / CP_A * 100 = 60 :=
by
  let SP_A := 180
  calc
    (SP_A - CP_A) / CP_A * 100 = (180 - 112.5) / 112.5 * 100 : by simp [CP_A]
    ...                           = 67.5 / 112.5 * 100         : by simp
    ...                           = 0.6 * 100                   : by norm_num
    ...                           = 60                           : by norm_num

end profit_percentage_A_l26_26561


namespace total_gum_correct_l26_26022

def num_cousins : ℕ := 4  -- Number of cousins
def gum_per_cousin : ℕ := 5  -- Pieces of gum per cousin

def total_gum : ℕ := num_cousins * gum_per_cousin  -- Total pieces of gum Kim needs

theorem total_gum_correct : total_gum = 20 :=
by sorry

end total_gum_correct_l26_26022


namespace choir_minimum_members_l26_26964

theorem choir_minimum_members (n : ℕ) :
  (∃ k1, n = 8 * k1) ∧ (∃ k2, n = 9 * k2) ∧ (∃ k3, n = 10 * k3) → n = 360 :=
by
  sorry

end choir_minimum_members_l26_26964


namespace solve_for_z_l26_26712

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26712


namespace probability_interval_l26_26878

noncomputable def eventA : Event Prop := sorry
noncomputable def eventB : Event Prop := sorry

axiom PA : P(eventA) = 5 / 6
axiom PB : P(eventB) = 7 / 8
axiom PA_union_B : P(eventA ∪ eventB) = 13 / 16

theorem probability_interval :
  43 / 48 ≤ P(eventA ∩ eventB) ∧ P(eventA ∩ eventB) ≤ 7 / 8 :=
by
  sorry

end probability_interval_l26_26878


namespace probability_both_selected_l26_26504

noncomputable def P_X : ℚ := 1 / 5
noncomputable def P_Y : ℚ := 2 / 3

theorem probability_both_selected (P_X P_Y : ℚ) : P_X * P_Y = 2 / 15 := by
  have h1 : P_X = 1 / 5 := by rfl
  have h2 : P_Y = 2 / 3 := by rfl
  rw [h1, h2]
  norm_num
  sorry

end probability_both_selected_l26_26504


namespace mn_bisects_incenters_segment_l26_26033

-- Definitions for midpoints and incenters
variables (A B C D : Point)
variable [circle_ABC : Circle A B C D]
variable (M : Point) (N : Point)
variable [midpoint_AB : Midpoint M A B]
variable [midpoint_CD : Midpoint N C D]
variable (I₁ : Point) (I₂ : Point)
variable [incenter_ABC : Incenter I₁ A B C]
variable [incenter_ADC : Incenter I₂ A D C]

-- The statement we need to prove
theorem mn_bisects_incenters_segment :
  is_bisector (segment I₁ I₂) (line MN) :=
sorry

end mn_bisects_incenters_segment_l26_26033


namespace arc_length_60_deg_1_radius_l26_26342

noncomputable def arc_length (n r : ℝ) : ℝ :=
  (n * π * r) / 180

theorem arc_length_60_deg_1_radius :
  arc_length 60 1 = π / 3 :=
by 
  -- this is where the steps would normally be filled in
  sorry

end arc_length_60_deg_1_radius_l26_26342


namespace avg_score_alice_charlie_l26_26069

-- Define the conditions as variables and constants
variable {Mrs_Taylor_students : ℕ}
variable {Alice_Bob_Charlie_absent : ℕ}
variable {remaining_students : ℕ}
variable {avg_score_remaining : ℚ}
variable {new_avg_score : ℚ}

-- Define the number of Mrs. Taylor's students
axiom h1 : Mrs_Taylor_students = 20

-- Define the number of absent students
axiom h2 : Alice_Bob_Charlie_absent = 3

-- Define the number of remaining students
axiom h3 : remaining_students = Mrs_Taylor_students - Alice_Bob_Charlie_absent

-- Define the average score for the remaining students
axiom h4 : avg_score_remaining = 78

-- Define the new average score after Alice and Charlie's tests are graded
axiom h5 : new_avg_score = 80

-- This is the theorem we aim to prove
theorem avg_score_alice_charlie : 
  (∑ n in finset.range remaining_students, n) / remaining_students = avg_score_remaining →
  (∑ n in finset.range (remaining_students + 2), n) / (remaining_students + 2) = new_avg_score →
  2 * new_avg_score * (remaining_students + 2) - 2 * avg_score_remaining * remaining_students = 194 →
  97 := sorry

end avg_score_alice_charlie_l26_26069


namespace percent_savings_correct_l26_26941

theorem percent_savings_correct :
  let cost_of_package := 9
  let num_of_rolls_in_package := 12
  let cost_per_roll_individually := 1
  let cost_per_roll_in_package := cost_of_package / num_of_rolls_in_package
  let savings_per_roll := cost_per_roll_individually - cost_per_roll_in_package
  let percent_savings := (savings_per_roll / cost_per_roll_individually) * 100
  percent_savings = 25 :=
by
  sorry

end percent_savings_correct_l26_26941


namespace product_of_factors_eq_fraction_l26_26603

theorem product_of_factors_eq_fraction : 
  (∏ n in Finset.range (13 - 3 + 1) + 3, (1 - (1 / n))) = 2 / 13 := 
by
  sorry

end product_of_factors_eq_fraction_l26_26603


namespace solve_imaginary_eq_l26_26762

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26762


namespace equal_segments_KX_XL_l26_26949

variables {A B C K L O X : Type*}
variables (circle : Circle O)
variables (tangent_AB : Tangent Circle AB)
variables (tangent_AC : Tangent Circle AC)
variables (line_ABK : Line AB K)
variables (line_ACL : Line AC L)
variables (segment_BC : Segment B C)
variables (point_X : X ∈ segment_BC)
variables (line_KL : Line K L)
variables (perpendicular_KL_XO : Perpendicular line_KL XO)

theorem equal_segments_KX_XL
  (tangent_at_B : PointOfTangency B AB)
  (tangent_at_C : PointOfTangency C AC)
  (K_on_AB : OnLine K AB)
  (L_on_AC : OnLine L AC)
  (KL_through_X : OnLine X KL)
  (KL_perpendicular_to_XO : Perpendicular KL XO) :
  segment KX = segment XL :=
sorry

end equal_segments_KX_XL_l26_26949


namespace anna_candy_division_l26_26223

theorem anna_candy_division : 
  ∀ (total_candies friends : ℕ), 
  total_candies = 30 → 
  friends = 4 → 
  ∃ (candies_to_remove : ℕ), 
  candies_to_remove = 2 ∧ 
  (total_candies - candies_to_remove) % friends = 0 := 
by
  sorry

end anna_candy_division_l26_26223


namespace time_to_be_apart_l26_26543

noncomputable def speed_A : ℝ := 17.5
noncomputable def speed_B : ℝ := 15
noncomputable def initial_distance : ℝ := 65
noncomputable def final_distance : ℝ := 32.5

theorem time_to_be_apart (x : ℝ) :
  x = 1 ∨ x = 3 ↔ 
  (x * (speed_A + speed_B) = initial_distance - final_distance ∨ 
   x * (speed_A + speed_B) = initial_distance + final_distance) :=
sorry

end time_to_be_apart_l26_26543


namespace eggs_divided_l26_26458

theorem eggs_divided (boxes : ℝ) (eggs_per_box : ℝ) (total_eggs : ℝ) :
  boxes = 2.0 → eggs_per_box = 1.5 → total_eggs = boxes * eggs_per_box → total_eggs = 3.0 :=
by
  intros
  sorry

end eggs_divided_l26_26458


namespace aardvark_run_distance_l26_26507

noncomputable def circumference (r : ℝ) := 2 * Real.pi * r

theorem aardvark_run_distance :
  let r_sm := 15;
  let r_lg := 30;
  let half_circumference_lg := circumference r_lg / 2;
  let diameter_sm := 2 * r_sm;
  let half_circumference_sm := circumference r_sm / 2;
  let total_distance := half_circumference_lg + diameter_sm + half_circumference_sm + diameter_sm;
  total_distance = 45 * Real.pi + 60 :=
by
  let r_sm := 15
  let r_lg := 30
  let half_circumference_lg := circumference r_lg / 2
  let diameter_sm := 2 * r_sm
  let half_circumference_sm := circumference r_sm / 2
  let total_distance := half_circumference_lg + diameter_sm + half_circumference_sm + diameter_sm
  have h1 : half_circumference_lg = 30 * Real.pi := by sorry
  have h2 : diameter_sm = 30 := by sorry
  have h3 : half_circumference_sm = 15 * Real.pi := by sorry
  calc total_distance
      = half_circumference_lg + diameter_sm + half_circumference_sm + diameter_sm : by sorry
  ... = 30 * Real.pi + 30 + 15 * Real.pi + 30 : by sorry
  ... = 45 * Real.pi + 60 : by sorry

end aardvark_run_distance_l26_26507


namespace solve_for_z_l26_26733

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26733


namespace sum_of_rows_and_columns_is_odd_l26_26126

-- Define the problem in Lean 4 terms
def probability_odd_sums_in_3x3_grid : ℚ :=
  1 / 14

-- State the hypothesis
def grid_condition (grid : (Fin 3) × (Fin 3) → Fin 9.succ) :=
  ∀ (i j : Fin 3), grid i j ∈ Finset.range 1 10 ∧ (∃ s, 
  s = ∑ i in Finset.univ, grid i j ∧ s % 2 = 1)

-- Statement of the problem
theorem sum_of_rows_and_columns_is_odd 
  (grid : (Fin 3) × (Fin 3) → Fin 9.succ) (h : grid_condition grid) : 
  (∑ i j, grid i j / grid_condition grid).to_rat = probability_odd_sums_in_3x3_grid :=
by sorry

end sum_of_rows_and_columns_is_odd_l26_26126


namespace pool_volume_approx_l26_26225

-- Define the conditions
def circular_pool (radius : ℝ) (depth_center : ℝ) (depth_edge : ℝ) : Prop :=
  radius = 30 ∧ depth_center = 6 ∧ depth_edge = 2

-- Define volume functions for cylinder and truncated cone
def volume_cylinder (r h : ℝ) : ℝ := π * r ^ 2 * h

def volume_truncated_cone (R r h : ℝ) : ℝ := (1 / 3) * π * h * (R ^ 2 + R * r + r ^ 2)

-- Main Lean 4 statement to prove the total volume
theorem pool_volume_approx : 
  ∀ (r h_center h_edge : ℝ), 
    circular_pool r h_center h_edge → 
    volume_cylinder 30 4 + volume_truncated_cone 30 0 2 ≈ 13195.54 :=
by
  intros
  rw circular_pool at H
  obtain ⟨H1, H2, H3⟩ := H
  -- Definitions of specific volumes
  have V_cylinder := volume_cylinder 30 4
  have V_cone := volume_truncated_cone 30 0 2
  -- Total volume calculation
  have Total_volume := V_cylinder + V_cone
  -- Approximation for π
  have approx_pi : π ≈ 3.14159 := sorry
  -- Plug the approximation to find the total volume
  sorry

-- Proof is omitted with "sorry"

end pool_volume_approx_l26_26225


namespace cyclist_total_time_l26_26192

-- Define the given conditions
def car_speed (v : ℝ) := v
def cyclist_speed (v : ℝ) := v / 4.5
def time_lag := 35 -- the cyclist is 35 minutes slower

-- Define the time it took for the car to reach the meeting point and back
def car_meeting_time (t : ℝ) := 2 * t

-- State the main theorem
theorem cyclist_total_time (v t: ℝ) (Hspeed : v > 0) (Htime : t > 0) 
  (H1 : cyclist_speed v = v / 4.5) 
  (H2 : ∃ t, time_lag = car_meeting_time t - 2 * t) :
  4.5 * (car_meeting_time t / 2) = 78.75 := by
  sorry

end cyclist_total_time_l26_26192


namespace sum_of_possible_k_l26_26365

theorem sum_of_possible_k (j k : ℕ) (hposj : 0 < j) (hposk : 0 < k) (heq : 1/j + 1/k = 1/4) : 
  (k ∈ [5, 6, 8, 12, 20]) → k ∈ {5, 6, 8, 12, 20} :=
by 
  intro hin
  have : 5 + 6 + 8 + 12 + 20 = 51 := by norm_num
  exact this

end sum_of_possible_k_l26_26365


namespace cover_by_circle_l26_26120

theorem cover_by_circle (Φ : Type) (projection_length : Π (l : ℝ^2), ℝ)
  (h1 : ∀ l : ℝ^2, projection_length l ≤ 1) : 
  ¬ (∃ center : ℝ^2, ∃ radius : ℝ, radius = 0.5 ∧ ∀ x : ℝ^2, x ∈ Φ → dist center x ≤ radius) ∧ 
  (∃ center : ℝ^2, ∃ radius : ℝ, radius = 0.75 ∧ ∀ x : ℝ^2, x ∈ Φ → dist center x ≤ radius) :=
by 
  sorry

end cover_by_circle_l26_26120


namespace required_vases_l26_26568

def vase_capacity_roses : Nat := 6
def vase_capacity_tulips : Nat := 8
def vase_capacity_lilies : Nat := 4

def remaining_roses : Nat := 20
def remaining_tulips : Nat := 15
def remaining_lilies : Nat := 5

def vases_for_roses : Nat := (remaining_roses + vase_capacity_roses - 1) / vase_capacity_roses
def vases_for_tulips : Nat := (remaining_tulips + vase_capacity_tulips - 1) / vase_capacity_tulips
def vases_for_lilies : Nat := (remaining_lilies + vase_capacity_lilies - 1) / vase_capacity_lilies

def total_vases_needed : Nat := vases_for_roses + vases_for_tulips + vases_for_lilies

theorem required_vases : total_vases_needed = 8 := by
  sorry

end required_vases_l26_26568


namespace roses_in_december_l26_26113

theorem roses_in_december (rOct rNov rJan rFeb : ℕ) 
  (hOct : rOct = 108)
  (hNov : rNov = 120)
  (hJan : rJan = 144)
  (hFeb : rFeb = 156)
  (pattern : (rNov - rOct = 12 ∨ rNov - rOct = 24) ∧ 
             (rJan - rNov = 12 ∨ rJan - rNov = 24) ∧
             (rFeb - rJan = 12 ∨ rFeb - rJan = 24) ∧ 
             (∀ m n, (m - n = 12 ∨ m - n = 24) → 
               ((rNov - rOct) ≠ (rJan - rNov) ↔ 
               (rJan - rNov) ≠ (rFeb - rJan)))) : 
  ∃ rDec : ℕ, rDec = 132 := 
by {
  sorry
}

end roses_in_december_l26_26113


namespace least_n_froods_l26_26804

theorem least_n_froods (n : ℕ) : (∃ n, n ≥ 30 ∧ (n * (n + 1)) / 2 > 15 * n) ∧ (∀ m < 30, (m * (m + 1)) / 2 ≤ 15 * m) :=
sorry

end least_n_froods_l26_26804


namespace matrix_power_eq_l26_26239

noncomputable def rotation_matrix (θ : ℝ) : Matrix 2 2 ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem matrix_power_eq :
  (3 • (rotation_matrix (Real.pi / 4))) ^ 4 =
    81 • ![![(-1 : ℝ), 0], ![0, -1]] :=
by sorry

end matrix_power_eq_l26_26239


namespace dolphins_next_month_l26_26491

-- Statements of the conditions
def total_dolphins : ℕ := 120
def fully_trained_dolphins : ℕ := total_dolphins * 1 / 4
def remaining_dolphins_after_fully_trained : ℕ := total_dolphins - fully_trained_dolphins
def semi_trained_dolphins : ℕ := remaining_dolphins_after_fully_trained * 1 / 6
def untrained_dolphins : ℕ := remaining_dolphins_after_fully_trained - semi_trained_dolphins
def semi_trained_and_untrained_dolphins : ℕ := semi_trained_dolphins + untrained_dolphins
def beginners_training_dolphins : ℕ := 33  -- 3/8 of semi_trained_and_untrained_dolphins, rounded
def remaining_dolphins_after_beginners : ℕ := semi_trained_and_untrained_dolphins - beginners_training_dolphins
def intermediate_training_dolphins : ℕ := 31  -- 5/9 of remaining_dolphins_after_beginners, rounded

-- Main theorem to prove
theorem dolphins_next_month : 
  (remaining_dolphins_after_beginners * 5 / 9).floor = 31 := 
by 
  sorry

end dolphins_next_month_l26_26491


namespace number_of_girls_l26_26955

theorem number_of_girls (B G : ℕ) (h1 : B + G = 400) 
  (h2 : 0.60 * B = (6 / 10 : ℝ) * B) 
  (h3 : 0.80 * G = (8 / 10 : ℝ) * G) 
  (h4 : (6 / 10 : ℝ) * B + (8 / 10 : ℝ) * G = (65 / 100 : ℝ) * 400) : G = 100 := by
sorry

end number_of_girls_l26_26955


namespace circle_tangent_line_lemma_l26_26370

open Real

def is_tangent_to_circle (p : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := p in
  let l := abs (4 * a - 2) / real.sqrt (4^2 + (-3)^2) in
  let r := abs a in
  l = r

theorem circle_tangent_line_lemma (a : ℝ) :
  let circle_center : ℝ × ℝ := (a, 0)
  let line : ℝ × ℝ := (3 * (1 : ℝ) + 2, 4 * (1 : ℝ) + 2)
  (4 * (3 * (1 : ℝ) + 2) - 3 * (4 * (1 : ℝ) + 2) - 2 = 0) →
  (∃ t, line = (3 * t + 2, 4 * t + 2)) →
  (ρ = 2 * a * cos θ → (x - a)^2 + y^2 = a^2) →
  is_tangent_to_circle circle_center a → 
  a = (2 / 9) := 
begin
  sorry
end

end circle_tangent_line_lemma_l26_26370


namespace sin_minus_cos_value_complex_expression_value_l26_26279

-- Define the given conditions
variables {x : ℝ}

axiom condition1 : -π < x ∧ x < 0
axiom condition2 : sin x + cos x = 1 / 5

-- Proof for (1)
theorem sin_minus_cos_value :
  sin x - cos x = -7 / 5 :=
sorry

-- Proof for (2)
theorem complex_expression_value :
  (3 * sin^2 (x / 2) - 2 * sin (x / 2) * cos (x / 2) + cos^2 (x / 2)) /
  (tan x + 1 / tan x) = -108 / 125 :=
sorry

end sin_minus_cos_value_complex_expression_value_l26_26279


namespace expected_spikiness_value_l26_26132

noncomputable def spikiness (seq : List ℝ) : ℝ :=
  (List.zipWith (λ a b => |b - a|) seq (List.tail seq)).sum

noncomputable def expected_spikiness_bound : ℝ :=
  (79 : ℝ) / 20

theorem expected_spikiness_value (x : Fin 9 → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ M, M = expected_spikiness_bound :=
begin
  sorry
end

end expected_spikiness_value_l26_26132


namespace floor_log_81_l26_26248

def floor (x : ℝ) : ℤ := ⌊x⌋ -- largest integer not exceeding x

theorem floor_log_81 :
  floor (Real.log 81 / Real.log 10) = 1 :=
by
  -- Given conditions
  have h1 : 1 < Real.log 81 / Real.log 10, from sorry
  have h2 : Real.log 81 / Real.log 10 < 2, from sorry
  
  -- Applying the floor function to log_81 / log_10 should yield 1.
  sorry

end floor_log_81_l26_26248


namespace zeros_of_f_l26_26136

noncomputable def f (x : ℝ) : ℝ := x^3 - 16 * x

theorem zeros_of_f :
  ∃ a b c : ℝ, (a = -4) ∧ (b = 0) ∧ (c = 4) ∧ (f a = 0) ∧ (f b = 0) ∧ (f c = 0) :=
by
  sorry

end zeros_of_f_l26_26136


namespace dice_probability_216_l26_26933

open Set
open Finset

def dice_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}

def event_216 (x y z : ℕ) : Prop := x * y * z = 216

def dice_probability (s : Finset (ℕ × ℕ × ℕ)) : ℚ :=
  (s.card : ℚ) / (dice_sides.card * dice_sides.card * dice_sides.card)

theorem dice_probability_216 :
  dice_probability {p | ∃ (x y z : ℕ), p = (x, y, z) ∧ x ∈ dice_sides ∧ y ∈ dice_sides ∧ z ∈ dice_sides ∧ event_216 x y z} = 1 / 216 :=
by
  sorry

end dice_probability_216_l26_26933


namespace sine_inequality_l26_26540

variable {x : ℝ}

theorem sine_inequality (h₁ : 0 < x) (h₂ : x ≤ 1) :
  (sin x / x)^2 < sin x / x ∧ sin x / x ≤ sin (x^2) / (x^2) := 
sorry

end sine_inequality_l26_26540


namespace min_q_of_abs_poly_eq_three_l26_26277

theorem min_q_of_abs_poly_eq_three (p q : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (|x1^2 + p * x1 + q| = 3) ∧ (|x2^2 + p * x2 + q| = 3) ∧ (|x3^2 + p * x3 + q| = 3)) →
  q = -3 :=
sorry

end min_q_of_abs_poly_eq_three_l26_26277


namespace increase_in_gross_income_l26_26173

variable (X P : ℝ)

theorem increase_in_gross_income (hX : X > 0) (hP : P > 0) :
  let original_gross_income := X * P
      new_gross_income := (1.15 * X) * (0.9 * P)
      percentage_increase := ((new_gross_income - original_gross_income) / original_gross_income) * 100
  in percentage_increase = 3.5 :=
by
  let original_gross_income := X * P
  let new_gross_income := (1.15 * X) * (0.9 * P)
  let percentage_increase := ((new_gross_income - original_gross_income) / original_gross_income) * 100
  exact sorry

end increase_in_gross_income_l26_26173


namespace solve_for_z_l26_26715

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26715


namespace hyperbola_center_l26_26609

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0 →
    (x, y) = (1, 2) :=
sorry

end hyperbola_center_l26_26609


namespace gcd_21_eq_7_count_l26_26628

theorem gcd_21_eq_7_count : Nat.card {n : Fin 200 // Nat.gcd 21 n = 7} = 19 := 
by
  sorry

end gcd_21_eq_7_count_l26_26628


namespace trajectory_of_P_geometric_property_of_parabola_l26_26360

-- Question I
theorem trajectory_of_P :
  (∀ P : ℝ × ℝ, (∀ a b : ℝ, P = (a, b) → real.sqrt ((a - 1)^2 + b^2) = abs a + 1)
  → ∃ E : set (ℝ × ℝ), E = {(x, y) | y^2 = 4 * x}) :=
sorry

-- Question II
theorem geometric_property_of_parabola :
  (∀ (A B : ℝ × ℝ) (C M F : ℝ × ℝ), 
    (F = (1, 0) 
     ∧ E = {(x, y) | y^2 = 4 * x}
     ∧ (∃ m : ℝ, ∀ x, (∃ y, A = (x, y) ∧ y^2 = 4 * x) ∧ (∃ y, B = (x, y) ∧ y^2 = 4 * x))
     ∧ line_through (1, 0) = l
     ∧ l ∩ E = {A, B}
     ∧ (x_of l = -1 → l ∩ x_axis = {C})
     ∧ midpoint A B = M)
  → abs (fst C - fst A) * abs (fst C - fst B) = abs (fst C - fst M) * abs (fst C - fst F)) :=
sorry

end trajectory_of_P_geometric_property_of_parabola_l26_26360


namespace larger_of_two_numbers_l26_26510

theorem larger_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 8) : max x y = 29 :=
by
  sorry

end larger_of_two_numbers_l26_26510


namespace cubic_has_real_root_l26_26273

theorem cubic_has_real_root (k : ℝ) : ∃ x : ℝ, x^3 + 3 * k * x^2 + 3 * k^2 * x + k^3 = 0 :=
by
  use -k
  calc 
    (-k)^3 + 3 * k * (-k)^2 + 3 * k^2 * (-k) + k^3 
        = - k^3 + 3 * k * k^2 - 3 * k^3 + k^3 : by sorry -- expand the terms and simplify
    ... = 0 : by sorry -- final simplification

end cubic_has_real_root_l26_26273


namespace complex_number_in_second_quadrant_l26_26599

theorem complex_number_in_second_quadrant :
  let z := Complex.mk (Real.cos (2 * Real.pi / 3)) (Real.sin (2 * Real.pi / 3))
  in z.re < 0 ∧ 0 < z.im :=
by
  let z := Complex.mk (Real.cos (2 * Real.pi / 3)) (Real.sin (2 * Real.pi / 3))
  sorry

end complex_number_in_second_quadrant_l26_26599


namespace unique_friends_count_l26_26953

-- Definitions from conditions
def M : ℕ := 10
def P : ℕ := 20
def G : ℕ := 5
def M_P : ℕ := 4
def M_G : ℕ := 2
def P_G : ℕ := 0
def M_P_G : ℕ := 2

-- Theorem we need to prove
theorem unique_friends_count : (M + P + G - M_P - M_G - P_G + M_P_G) = 31 := by
  sorry

end unique_friends_count_l26_26953


namespace water_needed_l26_26350

theorem water_needed (V_b W B : ℝ) (fraction_water V : ℝ) : 
  V_b = 0.09 →
  W = 0.03 →
  B = 0.06 →
  fraction_water = W / V_b →
  V = 0.72 →
  (V * fraction_water) = 0.24 :=
by
  intros hVb hW hB hfraction hV
  rw [hVb, hW] at hfraction
  rw [hV]
  rw [hfraction]
  norm_num
  sorry

end water_needed_l26_26350


namespace compound_proposition_l26_26867

def p : Prop := True
def q : Prop := False

theorem compound_proposition : (p ∨ q) = True :=
by {
  have hp : p := by trivial,
  have hq : ¬ q := by trivial,
  exact or.inl hp,  -- or as "p or q" is true because p is true.
}

end compound_proposition_l26_26867


namespace roots_quad_eq_l26_26331

-- The problem statement in Lean 4
theorem roots_quad_eq (a r s : ℝ)
  (h1 : r + s = a + 1)
  (h2 : r * s = a) :
  (r - s) ^ 2 = a ^ 2 - 2 * a + 1 :=
sorry

end roots_quad_eq_l26_26331


namespace two_trains_cross_time_l26_26542

-- Definitions based on conditions
def length_train1 : ℝ := 270
def speed_train1_kmph : ℕ := 120

def length_train2 : ℝ := 230.04
def speed_train2_kmph : ℕ := 80

def kmph_to_mps (speed_kmph : ℕ) : ℝ :=
  (speed_kmph * 1000) / 3600

def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

def total_distance : ℝ :=
  length_train1 + length_train2

-- Target theorem
theorem two_trains_cross_time :
  total_distance / relative_speed_mps = 9 := by
  sorry

end two_trains_cross_time_l26_26542


namespace vectors_parallel_sum_l26_26675

theorem vectors_parallel_sum (x y : ℝ) (k : ℝ)
  (ha : (-1 : ℝ, x, 3) = k • (2 : ℝ, -4, y))
  : x + y = -4 := 
by
  sorry

end vectors_parallel_sum_l26_26675


namespace determine_z_l26_26697

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26697


namespace equilateral_triangle_l26_26837

theorem equilateral_triangle (A B C H : Point) (h_a h_b h_c : ℝ) (p : ℝ)
  (orthocenter : is_orthocenter H A B C)
  (altitudes : is_altitude A H h_a ∧ is_altitude B H h_b ∧ is_altitude C H h_c)
  (semi_perimeter : p = (A.distance B + B.distance C + C.distance A) / 2)
  (condition : A.distance H * h_a + B.distance H * h_b + C.distance H * h_c = (2 / 3) * p^2) :
  A.distance B = B.distance C ∧ B.distance C = C.distance A := 
sorry

end equilateral_triangle_l26_26837


namespace no_general_relations_l26_26142

-- Define a triangle and a point inside it
variable {α : Type*} [field α] {A B C P : α}
variable (triangle_ABC : affine.subspace α (fin 3)) -- triangle ABC in a 2D affine space

-- Define that P is a point inside triangle ABC
variable (point_inside_triangle : P ∈ triangle_ABC)

-- Define the lines from the vertices to the opposite sides
def lines_from_vertices (A B C P : α) : Prop :=
  ∃ D E F : α, D ∈ line_through B C ∧ D ∉ {B, C} ∧
               E ∈ line_through A C ∧ E ∉ {A, C} ∧
               F ∈ line_through A B ∧ F ∉ {A, B} ∧
               line_through A D ∧ line_through B E ∧ line_through C F

-- Define the sub-triangles formed
def six_sub_triangles (A B C P : α) : Prop :=
  ∃ D E F G H I : α,
    lines_from_vertices A B C P ∧
    triangle (A B D) ∧ triangle (A C E) ∧
    triangle (B A F) ∧ triangle (B C G) ∧
    triangle (C A H) ∧ triangle (C B I)

-- Theorems for relations between the sub-triangles
theorem no_general_relations (A B C P : α) :
  point_inside_triangle triangle_ABC P →
  six_sub_triangles A B C P →
  ¬(∀ (Δ1 Δ2 : SubFiniteDimensionalizedSpace 3), Δ1 ~ Δ2) ∧
  ¬(∀ (Δ1 Δ2 : SubFiniteDimensionalizedSpace 3), Δ1 ≅ Δ2) ∧
  ¬(∀ (Δ1 Δ2 : SubFiniteDimensionalizedSpace 3), area(Δ1) = area(Δ2)) ∧
  ¬(∀(quad1 quad2 quad3 : SubFiniteDimensionalizedSpace 4), quad1 ~ quad2 ∧ quad2 ~ quad3) :=
by sorry

end no_general_relations_l26_26142


namespace non_zero_real_solution_l26_26518

theorem non_zero_real_solution (x : ℝ) (hx : x ≠ 0) (h : (3 * x)^5 = (9 * x)^4) : x = 27 :=
sorry

end non_zero_real_solution_l26_26518


namespace rectangular_field_area_l26_26534

noncomputable def a : ℝ := 14
noncomputable def c : ℝ := 17
noncomputable def b := Real.sqrt (c^2 - a^2)
noncomputable def area := a * b

theorem rectangular_field_area : area = 14 * Real.sqrt 93 := by
  sorry

end rectangular_field_area_l26_26534


namespace percentage_proof_l26_26859

theorem percentage_proof (n : ℝ) (h : 0.3 * 0.4 * n = 24) : 0.4 * 0.3 * n = 24 :=
sorry

end percentage_proof_l26_26859


namespace BP_bisects_angle_MPN_l26_26598

-- Setting up the geometric objects
variables (A B C M N P : Type)
variables [this_triangle: Triangle ABC]
variables [circumcircle: Circumcircle ABC]
variables [tangentA: TangentAt circumcircle A]
variables [tangentB: TangentAt circumcircle B]
variables [tangentC: TangentAt circumcircle C]
variables [intersection1: Intersection tangentB tangentA M]
variables [intersection2: Intersection tangentB tangentC N]
variables [perpendicular: Perpendicular B P AC]

-- Defining the main proof goal
-- This will prove that BP bisects ∠MPN
theorem BP_bisects_angle_MPN (h_triangle: AcuteAngledTriangle ABC)
  (h_circumcircle: Circumcircle ABC) (h_tangentA: TangentAt circumcircle A)
  (h_tangentB: TangentAt circumcircle B) (h_tangentC: TangentAt circumcircle C)
  (h_intersection1: Intersection tangentB tangentA M)
  (h_intersection2: Intersection tangentB tangentC N)
  (h_perpendicular: Perpendicular BP AC P)
  : Bisects BP ∠MPN :=
by
  sorry

end BP_bisects_angle_MPN_l26_26598


namespace complex_solution_l26_26719

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26719


namespace sum_log2_geom_seq_first_7_terms_l26_26793

-- Define the geometric sequence {a_n}
def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Define the sequence {log_2 a_n}
def log2_geom_seq (a r : ℝ) (n : ℕ) : ℕ → ℝ :=
  λ n, Real.log 2 (geometric_seq a r n)

-- Main theorem to be proved
theorem sum_log2_geom_seq_first_7_terms (a r : ℝ) (h_pos : ∀ n : ℕ, geometric_seq a r n > 0)
  (h_cond : geometric_seq a r 2 * geometric_seq a r 4 = 4) :
  ∑ i in (Finset.range 7), log2_geom_seq a r i = 7 :=
sorry

end sum_log2_geom_seq_first_7_terms_l26_26793


namespace area_of_shaded_region_l26_26560

theorem area_of_shaded_region :
  let s := 2
  let r := 1
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_one_semicircle := 1 / 2 * Real.pi * r^2
  let total_area_semicircles := 8 * area_one_semicircle
  area_octagon - total_area_semicircles = 8 + 8 * Real.sqrt 2 - 4 * Real.pi
:= by
  let s := 2
  let r := 1
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_one_semicircle := 1 / 2 * Real.pi * r^2
  let total_area_semicircles := 8 * area_one_semicircle
  show area_octagon - total_area_semicircles = 8 + 8 * Real.sqrt 2 - 4 * Real.pi
  sorry

end area_of_shaded_region_l26_26560


namespace joe_market_expense_l26_26390

theorem joe_market_expense :
  let oranges := 3 in
  let price_orange := 4.50 in
  let juices := 7 in
  let price_juice := 0.50 in
  let jars_honey := 3 in
  let price_jar_honey := 5 in
  let plants := 4 in
  let price_plants_per_pair := 18 in
  let total_cost := oranges * price_orange + juices * price_juice + jars_honey * price_jar_honey + (plants / 2) * price_plants_per_pair in
  total_cost = 68 := 
sorry

end joe_market_expense_l26_26390


namespace tan_225_eq_1_l26_26488

theorem tan_225_eq_1 : Real.tan (225 * Real.pi / 180) = 1 := by
  sorry

end tan_225_eq_1_l26_26488


namespace valid_digit_count_l26_26658

-- Define the condition
def valid_digit (A : ℕ) : Prop := A > 2 ∧ A < 10

-- The theorem stating the number of valid digits
theorem valid_digit_count : {d : Fin 10 // valid_digit d.val}.to_finset.card = 7 :=
by
  sorry

end valid_digit_count_l26_26658


namespace find_z_l26_26755

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26755


namespace tub_volume_ratio_l26_26166

theorem tub_volume_ratio (C D : ℝ) 
  (h₁ : 0 < C) 
  (h₂ : 0 < D)
  (h₃ : (3/4) * C = (2/3) * D) : 
  C / D = 8 / 9 := 
sorry

end tub_volume_ratio_l26_26166


namespace ratio_fifth_terms_l26_26880

variable (a_n b_n S_n T_n : ℕ → ℚ)

-- Conditions
variable (h : ∀ n, S_n n / T_n n = (9 * n + 2) / (n + 7))

-- Define the 5th term
def a_5 (S_n : ℕ → ℚ) : ℚ := S_n 9 / 9
def b_5 (T_n : ℕ → ℚ) : ℚ := T_n 9 / 9

-- Prove that the ratio of the 5th terms is 83 / 16
theorem ratio_fifth_terms :
  (a_5 S_n) / (b_5 T_n) = 83 / 16 :=
by
  sorry

end ratio_fifth_terms_l26_26880


namespace original_cost_price_l26_26546

theorem original_cost_price (S P C : ℝ) (h1 : S = 260) (h2 : S = 1.20 * C) : C = 216.67 := sorry

end original_cost_price_l26_26546


namespace bc_over_ad_eq_50_point_4_l26_26404

theorem bc_over_ad_eq_50_point_4 :
  let B := (2, 2, 5)
  let S (r : ℝ) (B : ℝ × ℝ × ℝ) := {p | dist p B ≤ r }
  let d := (20 : ℝ)
  let c := (48 : ℝ)
  let b := (28 * Real.pi : ℝ)
  let a := ((4 * Real.pi) / 3 : ℝ)
  let bc := b * c
  let ad := a * d
  bc / ad = 50.4 := by
    sorry

end bc_over_ad_eq_50_point_4_l26_26404


namespace maximum_marks_l26_26945

noncomputable def passing_mark (M : ℝ) : ℝ := 0.35 * M

theorem maximum_marks (M : ℝ) (h1 : passing_mark M = 210) : M = 600 :=
  by
  sorry

end maximum_marks_l26_26945


namespace simplify_x_cubed_simplify_expr_l26_26591

theorem simplify_x_cubed (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8 * x^2 + 15 * x := by
  sorry

theorem simplify_expr (x y : ℝ) : (5 * x + 2 * y) * (5 * x - 2 * y) - 5 * x * (5 * x - 3 * y) = -4 * y^2 + 15 * x * y := by
  sorry

end simplify_x_cubed_simplify_expr_l26_26591


namespace angle_between_is_90_degrees_l26_26822

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Given condition
def condition : Prop := ∥a + b∥ = ∥b∥

-- Proof problem: The angle between (a + 2b) and a is 90 degrees
theorem angle_between_is_90_degrees (h : condition a b) : real.angle (a + 2 • b) a = real.pi / 2 :=
sorry

end angle_between_is_90_degrees_l26_26822


namespace largest_is_three_l26_26895

variable (p q r : ℝ)

def cond1 : Prop := p + q + r = 3
def cond2 : Prop := p * q + p * r + q * r = 1
def cond3 : Prop := p * q * r = -6

theorem largest_is_three
  (h1 : cond1 p q r)
  (h2 : cond2 p q r)
  (h3 : cond3 p q r) :
  p = 3 ∨ q = 3 ∨ r = 3 := sorry

end largest_is_three_l26_26895


namespace quadratic_real_roots_range_l26_26300

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end quadratic_real_roots_range_l26_26300


namespace find_principal_l26_26942

def r : ℝ := 0.03
def t : ℝ := 3
def I (P : ℝ) : ℝ := P - 1820
def simple_interest (P : ℝ) : ℝ := P * r * t

theorem find_principal (P : ℝ) : simple_interest P = I P -> P = 2000 :=
by
  sorry

end find_principal_l26_26942


namespace inequality_does_not_hold_l26_26772

theorem inequality_does_not_hold (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) :
  ¬ (1 / (a - 1) < 1 / b) :=
by
  sorry

end inequality_does_not_hold_l26_26772


namespace range_of_a_l26_26883

noncomputable def quadratic_inequality_solution_set (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x - 4 < 0

theorem range_of_a :
  {a : ℝ | quadratic_inequality_solution_set a} = {a | -16 < a ∧ a ≤ 0} := 
sorry

end range_of_a_l26_26883


namespace problem_statement_l26_26817

theorem problem_statement (a b c d n : Nat) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < n) (h_eq : 7 * 4^n = a^2 + b^2 + c^2 + d^2) : 
  a ≥ 2^(n-1) ∧ b ≥ 2^(n-1) ∧ c ≥ 2^(n-1) ∧ d ≥ 2^(n-1) :=
sorry

end problem_statement_l26_26817


namespace complex_number_solution_l26_26684

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26684


namespace arithmetic_sequence_8th_term_l26_26922

theorem arithmetic_sequence_8th_term 
    (a₁ : ℝ) (a₅ : ℝ) (n : ℕ) (a₈ : ℝ) 
    (h₁ : a₁ = 3) 
    (h₂ : a₅ = 78) 
    (h₃ : n = 25) : 
    a₈ = 24.875 := by
  sorry

end arithmetic_sequence_8th_term_l26_26922


namespace length_of_AE_l26_26444

theorem length_of_AE (A B C D E : Type) [equilateral_triangle A B C]
  (on_BC : point_on BC D) (angle_condition : ∠ EAC = ∠ EBC)
  (BE_length : BE = 5) (CE_length : CE = 12) : AE = 17 :=
sorry

end length_of_AE_l26_26444


namespace part_a_part_b_l26_26003

-- Definitions for conditions
variable (rows cols : Fin 10 → Fin 10 → ℕ)
variable (digits : Fin 10 → Fin 10 → Fin 10)
variable (count : Fin 10 → ℕ)
variable (distinct_digits : Fin 10 → ℕ)

-- Conditions
axiom cond1 : ∀ (i j : Fin 10), digits i j < 10
axiom cond2 : ∀ (n : Fin 10), count n = 10
axiom count_def : ∀ (n : Fin 10), count n = 
  ∑ i : Fin 10, ∑ j : Fin 10, if digits i j = n then 1 else 0

-- Part (a): It is possible to arrange the digits such that no row or column contains more than four different digits.
theorem part_a : 
  ∃ (table: Fin 10 → Fin 10 → Fin 10), 
  (∀ n, count n = 10) ∧ 
  (∀ i, distinct_digits i ≤ 4) ∧ 
  (∀ j, distinct_digits (λ i, digits i j) ≤ 4) := 
sorry

-- Part (b): There is always a row or column with at least four different digits.
theorem part_b :
  ∃ (i : Fin 10), 4 ≤ distinct_digits i ∨
  ∃ (j : Fin 10), 4 ≤ distinct_digits (λ i, digits i j) := 
sorry

end part_a_part_b_l26_26003


namespace triangle_angle_conditions_l26_26377

theorem triangle_angle_conditions (A B C : ℝ) (α : ℝ) :
    internal_bisector_al A B C α → altitude_bh A B C α → median_cm A B C α → 
    (A = 60 ∧ B = 60 ∧ C = 60) ∨ (A = 60 ∧ B = 30 ∧ C = 90) :=
sorry

-- Definitions for conditions:
def internal_bisector_al (A B C α : ℝ) := ∃ (L : ℝ), angle_cal A B C α
def altitude_bh (A B C α : ℝ) := ∃ (H : ℝ), angle_abh A B C α
def median_cm (A B C α : ℝ) := ∃ (M : ℝ), angle_bcm A B C α

-- Angle equality definitions from the problem:
def angle_cal (A B C α : ℝ) := A / 2 = α
def angle_abh (A B C α : ℝ) := α
def angle_bcm (A B C α : ℝ) := α

end triangle_angle_conditions_l26_26377


namespace problem_solution_correct_l26_26596

theorem problem_solution_correct (
  a b : ℝ
) : ((a^2 + b^2 = 0 → a = 0 ∧ b = 0) ∧
     (¬ (a^2 + 2ab = a^2 + b^2) ∨ a ≠ b) ∧
     (a = 3 → a^2 + 9 = 3 * (3 + a)) ∧
     ((a^2 + b^2 = (a + b)^2) → a = 0 ∨ b = 0)) :=
by
  sorry

end problem_solution_correct_l26_26596


namespace line_through_point_parallel_l26_26264

theorem line_through_point_parallel (A : ℝ × ℝ) (m : ℝ) (b : ℝ) :
  A = (2, -3) ∧ m = 1 ∧ line_eq : ∃ b, ∀ (x y : ℝ), A.1 - A.2 = x - y → x - y = 5 :=
by
  sorry

end line_through_point_parallel_l26_26264


namespace three_lines_one_point_max_planes_l26_26141

noncomputable def number_of_planes {α : Type} (lines : set (set α)) : nat :=
  if h : lines.card = 3 then
    (lines.to_finset.subsets_of_len 2).card
  else 0

theorem three_lines_one_point_max_planes
  (α : Type)
  (P : α)
  (l1 l2 l3 : set α)
  (h1 : P ∈ l1)
  (h2 : P ∈ l2)
  (h3 : P ∈ l3)
  (distinct_lines : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3) :
  number_of_planes {l1, l2, l3} = 3 :=
by sorry

end three_lines_one_point_max_planes_l26_26141


namespace monotonically_increasing_l26_26051

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a / x

theorem monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 3 * x^2 - a / x^2 > 0) → a = -1 :=
begin
  intros h,
  sorry -- Proof not required as per instruction.
end

end monotonically_increasing_l26_26051


namespace arithmetic_sequence_30th_term_l26_26930

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem arithmetic_sequence_30th_term :
  arithmetic_sequence 3 6 30 = 177 :=
by
  -- Proof steps go here
  sorry

end arithmetic_sequence_30th_term_l26_26930


namespace find_local_min_l26_26380

def z (x y : ℝ) : ℝ := x^2 + 2 * y^2 - 2 * x * y - x - 2 * y

theorem find_local_min: ∃ (x y : ℝ), x = 2 ∧ y = 3/2 ∧ ∀ ⦃h : ℝ⦄, h ≠ 0 → z (2 + h) (3/2 + h) > z 2 (3/2) :=
by
  sorry

end find_local_min_l26_26380


namespace find_z_l26_26745

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26745


namespace ellipses_same_focal_length_l26_26323

def ellipse1 : Prop :=
  ∃ x y : ℝ, (x^2 / 12) + (y^2 / 4) = 1

def ellipse2 : Prop :=
  ∃ x y : ℝ, (x^2 / 16) + (y^2 / 8) = 1

def focal_length (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - b^2)

theorem ellipses_same_focal_length :
  let a1 := 2 * real.sqrt 3,
      b1 := 2,
      c1 := focal_length a1 b1,
      a2 := 4,
      b2 := 2 * real.sqrt 2,
      c2 := focal_length a2 b2
  in c1 = c2 :=
by {
  sorry
}

end ellipses_same_focal_length_l26_26323


namespace sufficient_but_not_necessary_condition_l26_26992

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 + x - 2 > 0) ∧ (∃ y, y < -2 ∧ y^2 + y - 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l26_26992


namespace problem_l26_26315

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

axiom min_f : ∀ x : ℝ, f(x) ≥ 1
axiom sequence_a (n : ℕ) : ℝ
axiom sequence_b (n : ℕ) : ℝ

def aₙ (n : ℕ) : ℝ := 13 - 4*n
def bₙ (n : ℕ) : ℝ := 3^(n-1)
def Sₙ (n : ℕ) : ℝ := (3^n - 4*n^2 + 22*n - 1) / 2

theorem problem {n : ℕ} :
  (sequence_a 2 = f 3) →
  (sequence_a 3 = 1) →
  (sequence_b 1 = 1) →
  (sequence_b 3 = 9) →
  (∀ n : ℕ, sequence_a n = aₙ n) →
  (∀ n : ℕ, sequence_b n = bₙ n) →
  (∀ n : ℕ, (sequence_a n + sequence_b n) = 2 * Sₙ n) := 
by
  intros
  sorry

end problem_l26_26315


namespace max_value_f_in_interval_l26_26476

noncomputable def f (x : ℝ) : ℝ := - (1 / 3) * x^3 + x^2

theorem max_value_f_in_interval : 
  ∃ x : ℝ, x ∈ set.Icc (0 : ℝ) 4 ∧ ∀ y ∈ set.Icc (0 : ℝ) 4, f y ≤ f x ∧ f x = 4 / 3 :=
sorry

end max_value_f_in_interval_l26_26476


namespace find_x_l26_26673

theorem find_x
  (x : ℝ)
  (a : ℝ × ℝ × ℝ := ( -3, 2, 5))
  (b : ℝ × ℝ × ℝ := ( 1, x, -1))
  (h : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 2) :
  x = 5 :=
by sorry

end find_x_l26_26673


namespace inverse_proportionality_l26_26461

theorem inverse_proportionality (a b c k a1 a2 b1 b2 c1 c2 : ℝ)
    (h1 : a * b * c = k)
    (h2 : a1 / a2 = 3 / 4)
    (h3 : b1 = 2 * b2)
    (h4 : c1 ≠ 0 ∧ c2 ≠ 0) :
    c1 / c2 = 2 / 3 :=
sorry

end inverse_proportionality_l26_26461


namespace concurrency_of_AD_l26_26418

open Real
open EuclideanGeometry

variables {A B C P D E F D' E' F' : Point}
variables {incircle : Circle}

-- Definitions and Hypotheses
def within_incircle (P incircle : Point → Prop) := P ∈ incircle

def meets_incircle_again (P : Point) (D E F D' E' F' incircle : Point → Prop) :=
  (D P ∈ incircle) ∧ (E P ∈ incircle) ∧ (F P ∈ incircle) ∧ (D P ≠ D') ∧ (E P ≠ E') ∧ (F P ≠ F')

def are_concurrent (l1 l2 l3 : Line) := ∃ X : Point, X ∈ l1 ∧ X ∈ l2 ∧ X ∈ l3

-- Lean 4 Statement
theorem concurrency_of_AD'_BE'_CF' 
  (incircle_ABC : incircle ∈ circle_inradius A B C)
  (P_in_incircle : within_incircle P incircle_ABC)
  (DP_EP_FP_meet : meets_incircle_again P D E F D' E' F' incircle_ABC):
  are_concurrent (line_through A D') (line_through B E') (line_through C F') :=
sorry

end concurrency_of_AD_l26_26418


namespace find_radius_l26_26469

theorem find_radius (AB EO : ℝ) (AE BE : ℝ) (h1 : AB = AE + BE) (h2 : AE = 2 * BE) (h3 : EO = 7) :
  ∃ R : ℝ, R = 11 := by
  sorry

end find_radius_l26_26469


namespace complex_solution_l26_26720

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26720


namespace time_taken_by_A_l26_26006

-- Definitions for the problem conditions
def race_distance : ℕ := 1000  -- in meters
def A_beats_B_by_distance : ℕ := 48  -- in meters
def A_beats_B_by_time : ℕ := 12  -- in seconds

-- The formal statement to prove in Lean
theorem time_taken_by_A :
  ∃ T_a : ℕ, (1000 * (T_a + 12) = 952 * T_a) ∧ T_a = 250 :=
by
  sorry

end time_taken_by_A_l26_26006


namespace complex_number_quadrant_l26_26483

theorem complex_number_quadrant :
  let z := (2 - complex.i) / (3 * complex.i - 1)
  Re(z) < 0 ∧ Im(z) < 0 := 
by
  let z := (2 - complex.i) / (3 * complex.i - 1)
  sorry

end complex_number_quadrant_l26_26483


namespace repeating_decimal_division_l26_26920

theorem repeating_decimal_division :
  (0.\overline{54} / 0.\overline{18}) = 3 :=
by
  have h1 : 0.\overline{54} = 54 / 99 := sorry
  have h2 : 0.\overline{18} = 18 / 99 := sorry
  have h3 : (54 / 99) / (18 / 99) = 54 / 18 := sorry
  have h4 : 54 / 18 = 3 := sorry
  rw [h1, h2, h3, h4]
  exact rfl

end repeating_decimal_division_l26_26920


namespace probability_upper_faces_equal_probability_upper_faces_sum_less_than_5_l26_26212

noncomputable def probability_equal_faces : ℚ :=
  let outcomes := (finset.pi finset.univ (fun _ => finset.range 5).erase (6 : ℕ)) : finset (ℕ × ℕ),
  let equal_faces := ((list.range 6).map fun x => (x,x)) : list (ℕ×ℕ) in
  (equal_faces.to_finset.card : ℚ) / outcomes.card

noncomputable def probability_sum_less_than_5 : ℚ :=
  let outcomes := (finset.pi finset.univ (fun _ => finset.range 5).erase (6 : ℕ)) : finset (ℕ × ℕ),
  let sum_less_than_5 := outcomes.filter (fun (p : ℕ × ℕ) => p.1 + p.2 < 5) in
  sum_less_than_5.card / outcomes.card

theorem probability_upper_faces_equal :
  probability_equal_faces = 1/6 :=
sorry

theorem probability_upper_faces_sum_less_than_5 :
  probability_sum_less_than_5 = 1/6 :=
sorry

end probability_upper_faces_equal_probability_upper_faces_sum_less_than_5_l26_26212


namespace solution_problem_l26_26806

open Real

noncomputable def parametric_curve_x (t : ℝ) : ℝ := 2 * cos t
noncomputable def parametric_curve_y (t : ℝ) : ℝ := sin t

def polar_line_equation (ρ θ : ℝ) : Prop :=
  ρ * cos (θ + π / 3) = -√3 / 2

def cartesian_curve_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def cartesian_line_equation (x y : ℝ) : Prop :=
  x - √3 * y + √3 = 0

def length_of_segment_AB (A B : ℝ × ℝ) : ℝ :=
  sqrt (1 + (√3 / 3)^2) * (abs ((fst A + fst B) - 4 * 0)) -- A == (x1, y1) and B == (x2, y2)

theorem solution_problem 
  (t : ℝ)
  (ρ θ : ℝ) 
  (A B : ℝ × ℝ) :
  cartesian_curve_equation (parametric_curve_x t) (parametric_curve_y t) ∧
  polar_line_equation ρ θ ∧
  cartesian_line_equation (fst A) (snd A) ∧
  cartesian_line_equation (fst B) (snd B)
  → length_of_segment_AB A B = 32 / 7 :=
by sorry

end solution_problem_l26_26806


namespace solve_inequalities_l26_26092

theorem solve_inequalities (x : ℝ) :
  ( (-x + 3)/2 < x ∧ 2*(x + 6) ≥ 5*x ) ↔ (1 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequalities_l26_26092


namespace monotone_increasing_range_of_a_l26_26129

noncomputable def f (a x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_increasing_range_of_a :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Icc (-1 / 3 : ℝ) (1 / 3 : ℝ)) :=
sorry

end monotone_increasing_range_of_a_l26_26129


namespace even_function_increasing_on_positives_l26_26990

-- Definitions of the functions
def A (x : ℝ) : ℝ := x^2 + 2*x
def B (x : ℝ) : ℝ := -x^3
def C (x : ℝ) : ℝ := abs (Real.log x)
def D (x : ℝ) : ℝ := 2^abs x

-- Theorem statement to prove
theorem even_function_increasing_on_positives :
  (∀ f, (f = A ∨ f = B ∨ f = C ∨ f = D) → 
  (Function.Even f → ∀ x : ℝ, 0 < x → f x < f x) →
  (f = D)) :=
by
  -- Assuming the proof for the theorem for now
  sorry

end even_function_increasing_on_positives_l26_26990


namespace range_of_f_l26_26839

noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

theorem range_of_f :
  let R := Set.range (λ x : ℝ, if x < 0 then none else some (f x)) in
  ∃ N n : ℝ, N = 3 ∧ n = 4 / 3 ∧ (3 ∉ R) ∧ (4 / 3 ∈ R) := 
by
  unfold f
  sorry

end range_of_f_l26_26839


namespace least_lcm_of_x_and_z_l26_26474

theorem least_lcm_of_x_and_z (x y z : ℕ) (h₁ : Nat.lcm x y = 20) (h₂ : Nat.lcm y z = 28) : 
  ∃ l, l = Nat.lcm x z ∧ l = 35 := 
sorry

end least_lcm_of_x_and_z_l26_26474


namespace f_g_minus_g_f_l26_26462

namespace Proof

def f (x : ℝ) : ℝ := 3 * x^2 - 1
def g (x : ℝ) : ℝ := x^(1/3) + 1

theorem f_g_minus_g_f (x : ℝ) :
  f(g(x)) - g(f(x)) = 3 * (x^(1/3))^2 + 6 * x^(1/3) + 1 - (3 * x^2 - 1)^(1/3) :=
by
  sorry

end Proof

end f_g_minus_g_f_l26_26462


namespace sequence_geometric_and_max_value_l26_26373

theorem sequence_geometric_and_max_value :
  ∀ (a : ℕ → ℕ) (n : ℕ), n ≠ 0 → 
  (a 1 = 2) →
  (∀ n, n ≠ 0 → a (n + 1) = 4 * a n - 3 * n + 1) →
  (∃ r, r ≠ 0 ∧ ∀ n, a (n + 1) - (n + 1) = r * (a n - n)) ∧ 
  (let S := λ n, ∑ i in range n, a i
   in S (n + 1) - 4 * (S n) ≤ 0 ∧ S (n + 1) - 4 * (S n) = 0 → n = 1) :=
by
  intro a n hn h1 h_rec
  have H1 : ∃ r, r ≠ 0 ∧ ∀ n, a (n + 1) - (n + 1) = r * (a n - n) := sorry
  have H2 : let S := λ n, ∑ i in range n, a i
            in S (n + 1) - 4 * (S n) ≤ 0 ∧ S (n + 1) - 4 * (S n) = 0 → n = 1 := sorry
  exact ⟨H1, H2⟩

end sequence_geometric_and_max_value_l26_26373


namespace man_swim_upstream_distance_l26_26971

theorem man_swim_upstream_distance (c d : ℝ) (h1 : 15.5 + c ≠ 0) (h2 : 15.5 - c ≠ 0) :
  (15.5 + c) * 2 = 36 ∧ (15.5 - c) * 2 = d → d = 26 := by
  sorry

end man_swim_upstream_distance_l26_26971


namespace find_AD_length_l26_26077

-- Define the basic geometric entities
variables (A B C D M H K : Point)

-- Given conditions
variables (AD_parallel_BC : Parallel AD BC) 
          (M_on_CD : OnSegment M C D)
          (perpendicular_AH_BM : Perpendicular AH BM)
          (AD_eq_HD : AD = HD)
          (BC_len : BC = 16)
          (CM_len : CM = 8)
          (MD_len : MD = 9)

-- Define the proof
theorem find_AD_length : AD = 18 := by
  sorry

end find_AD_length_l26_26077


namespace gnollish_valid_sentence_count_is_48_l26_26464

-- Define the problem parameters
def gnollish_words : List String := ["word1", "word2", "splargh", "glumph", "kreeg"]

def valid_sentence_count : Nat :=
  let total_sentences := 4 * 4 * 4
  let invalid_sentences :=
    4 +         -- (word) splargh glumph
    4 +         -- splargh glumph (word)
    4 +         -- (word) splargh kreeg
    4           -- splargh kreeg (word)
  total_sentences - invalid_sentences

-- Prove that the number of valid 3-word sentences is 48
theorem gnollish_valid_sentence_count_is_48 : valid_sentence_count = 48 := by
  sorry

end gnollish_valid_sentence_count_is_48_l26_26464


namespace percentage_compositions_correct_l26_26189

noncomputable def initial_volume : ℕ := 560
noncomputable def initial_water_percentage : ℝ := 75
noncomputable def initial_kola_percentage : ℝ := 15
noncomputable def initial_sugar_percentage : ℝ := 10

noncomputable def added_water : ℕ := 25
noncomputable def added_kola : ℕ := 12
noncomputable def added_sugar : ℕ := 18

noncomputable def final_volume : ℕ := initial_volume + added_water + added_kola + added_sugar

noncomputable def final_percentage_water : ℝ := (initial_volume * (initial_water_percentage / 100.0) + added_water) / final_volume * 100
noncomputable def final_percentage_kola : ℝ := (initial_volume * (initial_kola_percentage / 100.0) + added_kola) / final_volume * 100
noncomputable def final_percentage_sugar : ℝ := (initial_volume * (initial_sugar_percentage / 100.0) + added_sugar) / final_volume * 100

theorem percentage_compositions_correct :
  final_percentage_water ≈ 72.36 ∧ final_percentage_kola ≈ 15.61 ∧ final_percentage_sugar ≈ 12.03 := by
  sorry

end percentage_compositions_correct_l26_26189


namespace book_price_net_change_l26_26178

theorem book_price_net_change (P : ℝ) :
  let decreased_price := P - 0.3 * P in
  let increased_price := decreased_price + 0.4 * decreased_price in
  (increased_price - P) / P = -0.02 :=
by
  sorry

end book_price_net_change_l26_26178


namespace half_A_lt_sqrt2_minus_1_l26_26035

noncomputable def A (n : ℕ) (a : Fin (2 * n) → ℝ) (k : Fin (2 * n)) : ℝ :=
  (1 + ∑ i in Finset.range (n - 1), a ⟨(k.val + i) % (2 * n), sorry⟩) / 
  (1 + ∑ i in Finset.range (2 * n - 1), a ⟨(k.val + i) % (2 * n), sorry⟩)

theorem half_A_lt_sqrt2_minus_1 (n : ℕ) (a : Fin (2 * n) → ℝ) 
  (h1 : 0 < n)
  (h2 : (∏ i, a i) = 2)
  (h3 : pairwise (λ i j, A n a i ≠ A n a j)) :
  (Finset.card (Finset.filter (λ k, A n a k < Real.sqrt 2 - 1) Finset.univ) = n) :=
begin
  sorry
end

end half_A_lt_sqrt2_minus_1_l26_26035


namespace composition_result_l26_26652

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := (x - 3) / 2

theorem composition_result (x : ℝ) (comp : (ℝ → ℝ) → (ℝ → ℝ) → ℝ → ℝ) :
  comp f g (comp g f (comp f f (comp g g x))) = 7 → g 7 = 2 := by
    sorry

end composition_result_l26_26652


namespace chris_eats_donuts_l26_26019

def daily_donuts := 10
def days := 12
def donuts_eaten_per_day := 1
def boxes_filled := 10
def donuts_per_box := 10

-- Define the total number of donuts made.
def total_donuts := daily_donuts * days

-- Define the total number of donuts Jeff eats.
def jeff_total_eats := donuts_eaten_per_day * days

-- Define the remaining donuts after Jeff eats his share.
def remaining_donuts := total_donuts - jeff_total_eats

-- Define the total number of donuts in the boxes.
def donuts_in_boxes := boxes_filled * donuts_per_box

-- The proof problem:
theorem chris_eats_donuts : remaining_donuts - donuts_in_boxes = 8 :=
by
  -- Placeholder for proof
  sorry

end chris_eats_donuts_l26_26019


namespace solve_fraction_sum_l26_26045

noncomputable theory

def roots_of_polynomial : Prop :=
  let a b c := classical.some (roots_of_polynomial_eq (x^3 - 15 * x^2 + 22 * x - 8 = 0)) in
  (a + b + c = 15) ∧ (ab + ac + bc = 22) ∧ (abc = 8)

theorem solve_fraction_sum (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 22) (h₃ : abc = 8) :
  (\frac{a}{\frac{1}{a}+bc} + \frac{b}{\frac{1}{b}+ca} + \frac{c}{\frac{1}{c}+ab}) = \frac{181}{9} :=
  by
    sorry

end solve_fraction_sum_l26_26045


namespace limit_example_l26_26951

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - 11| ∧ |x - 11| < δ) → |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε :=
by
  sorry

end limit_example_l26_26951


namespace coefficient_of_x_in_expansion_l26_26234

theorem coefficient_of_x_in_expansion :
  (n : ℕ) (a b : ℤ) (x : FormalSeries ℤ) (p : x = X) (h : (a : ℤ) = 3) (t : (b : ℤ) = -1) :
  ℤ :=
sorry

end coefficient_of_x_in_expansion_l26_26234


namespace strawberries_in_each_handful_l26_26861

theorem strawberries_in_each_handful (x : ℕ) (h : (x - 1) * (75 / x) = 60) : x = 5 :=
sorry

end strawberries_in_each_handful_l26_26861


namespace difference_surface_area_cubes_l26_26174

open Real

theorem difference_surface_area_cubes
  (vol_large : ℝ)
  (count_small : ℕ)
  (vol_small : ℝ)
  (h1 : vol_large = 125)
  (h2 : count_small = 125)
  (h3 : vol_small = 1) :
  let side_length_large := (vol_large)^(1/3)
      sa_large := 6 * (side_length_large ^ 2)
      side_length_small := (vol_small)^(1/3)
      sa_small := 6 * (side_length_small ^ 2)
      total_sa_small := count_small * sa_small in
  total_sa_small - sa_large = 600 :=
by
  sorry

end difference_surface_area_cubes_l26_26174


namespace determinant_transformed_columns_l26_26405

variables {R : Type*} [CommRing R]
variables (a b c : R^3)

-- Assume E is given by the determinant of matrix with columns a, b, c
def E : R := dot_product a (cross_product b c)

-- Statement of the proof problem
theorem determinant_transformed_columns (a b c : R^3) :
  let det := dot_product (2 • a + b) (cross_product (b + 3 • c) (c + 4 • a)) in
  det = 14 * E := 
sorry

end determinant_transformed_columns_l26_26405


namespace tan_sum_is_rel_prime_l26_26399
noncomputable def condition : ℝ := (193 : ℝ) / 137

theorem tan_sum_is_rel_prime (h : ∀ θ : ℝ, sin θ + cos θ = condition → θ ≠ 0) : 
  let a := tan θ_1,
      b := tan θ_2,
      θ_1 := Real.arctan a,
      θ_2 := Real.arctan b in
  (∀ θ_1 θ_2 : ℝ, sin θ_1 + cos θ_1 = condition ∧ sin θ_2 + cos θ_2 = condition → a + b = (18769 : ℚ) / 9240) ∧ 
  (a, b : ℚ) → a.num + b.num = 28009 := sorry

end tan_sum_is_rel_prime_l26_26399


namespace eggs_left_l26_26493

theorem eggs_left (n h j : ℕ) (h_n : n = 47) (h_h : h = 5) (h_j : j = 8) : n - (h + j) = 34 :=
by
  rw [h_n, h_h, h_j]
  norm_num

end eggs_left_l26_26493


namespace salt_percentage_in_saltwater_l26_26253

theorem salt_percentage_in_saltwater (salt_mass water_mass : ℕ) (h₀ : salt_mass = 250) (h₁ : water_mass = 1000) :
  (salt_mass * 100 / (salt_mass + water_mass)) = 20 :=
by
  rw [h₀, h₁]
  sorry

end salt_percentage_in_saltwater_l26_26253


namespace dutch_fraction_of_bus_l26_26072

theorem dutch_fraction_of_bus : 
  (∀ (total_people dutch window_seats : ℕ), 
  total_people = 90 → 
  (1/2 : ℚ) * dutch * (1/3 : ℚ) = window_seats → 
  window_seats = 9 → 
  (3/5 : ℚ) = dutch / total_people) :=
by 
  intros total_people dutch window_seats 
  assume h1 : total_people = 90
  assume h2 : (1 / 2 : ℚ) * dutch * (1 / 3 : ℚ) = window_seats
  assume h3 : window_seats = 9
  sorry

end dutch_fraction_of_bus_l26_26072


namespace fraction_relation_l26_26467

theorem fraction_relation (n d : ℕ) (h1 : (n + 1 : ℚ) / (d + 1) = 3 / 5) (h2 : (n : ℚ) / d = 5 / 9) :
  ∃ k : ℚ, d = k * 2 * n ∧ k = 9 / 10 :=
by
  sorry

end fraction_relation_l26_26467


namespace number_of_possible_outcomes_highest_probability_ball_color_proportion_white_yellow_l26_26362

-- Define the number of white, red, and yellow balls in the box
def num_white_balls := 1
def num_red_balls := 2
def num_yellow_balls := 3

-- Total number of balls in the box
def total_balls := num_white_balls + num_red_balls + num_yellow_balls

-- Number of possible outcomes when drawing one ball
theorem number_of_possible_outcomes : ∃ n, n = 3 := by 
  have total_types := 3
  use total_types
  sorry

-- The color of the ball with the highest probability of being drawn
theorem highest_probability_ball_color : ∃ color, color = "yellow" := by
  have highest_color := "yellow"
  use highest_color
  sorry

-- The proportion of white and yellow balls in the total number of balls in the box
theorem proportion_white_yellow : ∃ ratio, ratio = 2 / 3 := by
  have proportion := (num_white_balls + num_yellow_balls) / total_balls
  have correct_proportion := (2 / 3 : ℚ)
  use correct_proportion
  sorry

end number_of_possible_outcomes_highest_probability_ball_color_proportion_white_yellow_l26_26362


namespace no_correct_option_l26_26114

-- Define the given table as a list of pairs
def table :=
  [(1, -2), (2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Define the given functions as potential options
def optionA (x : ℕ) : ℤ := x^2 - 5 * x + 4
def optionB (x : ℕ) : ℤ := x^2 - 3 * x
def optionC (x : ℕ) : ℤ := x^3 - 3 * x^2 + 2 * x
def optionD (x : ℕ) : ℤ := 2 * x^2 - 4 * x - 2
def optionE (x : ℕ) : ℤ := x^2 - 4 * x + 2

-- Prove that there is no correct option among the given options that matches the table
theorem no_correct_option : 
  ¬(∀ p ∈ table, p.snd = optionA p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionB p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionC p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionD p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionE p.fst) :=
by sorry

end no_correct_option_l26_26114


namespace solve_imaginary_eq_l26_26765

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26765


namespace largest_four_digit_number_with_product_72_has_sum_17_l26_26820

noncomputable def largest_digits_sum : ℕ := 17

theorem largest_four_digit_number_with_product_72_has_sum_17 :
  ∃ n : ℕ, 999 ≤ n ∧ n ≤ 9999 ∧ (nat.digits 10 n).prod = 72 ∧ (nat.digits 10 n).sum = largest_digits_sum :=
sorry

end largest_four_digit_number_with_product_72_has_sum_17_l26_26820


namespace ab_cd_eq_neg190_over_9_l26_26773

theorem ab_cd_eq_neg190_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -2)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 :=
by
  sorry

end ab_cd_eq_neg190_over_9_l26_26773


namespace solve_for_z_l26_26713

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26713


namespace om_on_constant_l26_26372

noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1 ∧ x ≥ 0

def parametric_eqn (θ : ℝ) : Prop :=
  -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2 ∧
  ∃ x y : ℝ, x = Real.cos θ ∧ y = 2 * Real.sin θ ∧ curve_C x y

def is_on_curve (P : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), parametric_eqn θ ∧ P = (Real.cos θ, 2 * Real.sin θ)

def is_minor_axis_endpoint (P : ℝ × ℝ) : Prop :=
  P = (0, 2) ∨ P = (0, -2)

def intersects_x_axis (line_eq : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, line_eq x = 0

def product_distances (P : ℝ × ℝ) (B1 B2 : ℝ × ℝ) (O : ℝ × ℝ := (0, 0)) : ℝ :=
  let x_M := ((fst B1 - snd B1) * snd P + 2 * snd B1) / fst P in
  let x_N := ((fst B2 - snd B2) * snd P - 2 * snd B2) / fst P in
  Real.abs x_M * Real.abs x_N

theorem om_on_constant (P : ℝ × ℝ) (hP : is_on_curve P) (hE : ¬is_minor_axis_endpoint P) :
  ∃ (B1 B2 : ℝ × ℝ), B1 = (0, 2) ∧ B2 = (0, -2) ∧ product_distances P B1 B2 = 1 :=
by
  exists (0, 2)
  exists (0, -2)
  refine ⟨rfl, rfl, _⟩
  sorry

end om_on_constant_l26_26372


namespace james_carrot_sticks_l26_26382

theorem james_carrot_sticks (total_carrots : ℕ) (after_dinner_carrots : ℕ) (before_dinner_carrots : ℕ) 
  (h1 : total_carrots = 37) (h2 : after_dinner_carrots = 15) :
  before_dinner_carrots = total_carrots - after_dinner_carrots :=
by
suffices h : 37 - 15 = 22 by
  rw [← h1, ← h2]
  exact h
apply rfl

end james_carrot_sticks_l26_26382


namespace no_infinite_arithmetic_progression_l26_26400

open Classical

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

theorem no_infinite_arithmetic_progression
  (f_strict_inc : ∀ x y : R, 0 < x ∧ 0 < y → x < y → f x < f y)
  (f_convex : ∀ x y : R, 0 < x ∧ 0 < y → f ((x + y) / 2) < (f x + f y) / 2) :
  ∀ a : ℕ → R, (∀ n : ℕ, a n = f n) → ¬(∃ d : R, ∀ k : ℕ, a (k + 1) - a k = d) :=
sorry

end no_infinite_arithmetic_progression_l26_26400


namespace possible_values_of_a_l26_26892

theorem possible_values_of_a :
  ∃ (a : ℤ), (∀ (b c : ℤ), (x : ℤ) → (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → (a = 6 ∨ a = 10) :=
sorry

end possible_values_of_a_l26_26892


namespace log_eq_15_l26_26087

noncomputable def x : ℝ := 16 * Real.sqrt 2

theorem log_eq_15 : log 8 x + log 2 (x^3) = 15 :=
  sorry

end log_eq_15_l26_26087


namespace quadrilateral_BD_length_l26_26801

theorem quadrilateral_BD_length (AB BC CD DA BD : ℕ) 
  (hAB : AB = 4) (hBC : BC = 14) (hCD : CD = 4) (hDA : DA = 7)
  (hBD_int : BD ∈ ℤ)
  (h1 : 4 + BD > 7)
  (h2 : 14 + 4 > BD)
  (h3 : 7 + BD > 4)
  (h4 : BD + 4 > 14) :
  BD = 11 := 
sorry

end quadrilateral_BD_length_l26_26801


namespace cross_shape_cube_l26_26200

-- Definitions to describe the shape and positions
structure Square where
  id : ℕ

structure CrossShape where
  squares : Fin 6 → Square

structure Position where
  position_id : Fin 6

def canFoldToCubeWithOneFaceMissing (shape : CrossShape) (additional_square_position : Position) : Prop := sorry

-- The main theorem
theorem cross_shape_cube (base_shape : CrossShape) :
  (Finset.univ.filter (λ p : Position, canFoldToCubeWithOneFaceMissing base_shape p)).card = 3 := sorry

end cross_shape_cube_l26_26200


namespace small_rectangular_solids_count_l26_26978

theorem small_rectangular_solids_count :
  let large_length := 1.5
  let large_width := 0.7
  let large_height := 0.9
  let small_length := 13 / 100 -- 0.13 meters
  let small_width := 5 / 100  -- 0.05 meters
  let small_height := 7 / 100  -- 0.07 meters
  let large_volume := large_length * large_width * large_height
  let small_volume := small_length * small_width * small_height
  let count := Int.floor (large_volume / small_volume)
  count = 2076 := by
  have large_volume := large_length * large_width * large_height
  have small_volume := small_length * small_width * small_height
  have count := Int.floor (large_volume / small_volume)
  show count = 2076 from sorry

end small_rectangular_solids_count_l26_26978


namespace sum_of_roots_eq_12_l26_26621

noncomputable def equation (x : ℝ) : ℝ := 4 * x^2 - 58 * x + 190 - (29 - 4 * x - Real.log x / Real.log 2) * (Real.log x / Real.log 2)

theorem sum_of_roots_eq_12 : 
  ∀ x : ℝ, equation x = 0 → x ∈ set_of (λ x, x = 4 ∨ x = 8) → (∑ x in {4, 8}, x) = 12 := 
by
  sorry

end sum_of_roots_eq_12_l26_26621


namespace range_increases_with_additional_score_l26_26582

noncomputable def scores_before : List ℕ :=
  [41, 46, 50, 50, 55, 55, 55, 59, 63, 68, 72, 75, 80]

noncomputable def additional_score : ℕ := 36

-- Define the range of a list of scores
noncomputable def range (scores : List ℕ) : ℕ :=
  (scores.maximumD 0) - (scores.minimumD 0)

theorem range_increases_with_additional_score :
  range (additional_score :: scores_before) > range scores_before := by
  sorry

end range_increases_with_additional_score_l26_26582


namespace find_z_l26_26750

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26750


namespace max_value_f_on_interval_l26_26475

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem max_value_f_on_interval :
  ∃ x ∈ set.Icc (-1:ℝ) 1, ∀ y ∈ set.Icc (-1:ℝ) 1, f y ≤ f x ∧ f x = Real.exp 1 + 1 :=
by
  sorry

end max_value_f_on_interval_l26_26475


namespace cone_volume_is_correct_l26_26198

-- Given data
def radius_circle : ℝ := 6
def sector_angle : ℝ := π  -- Half-sector implies angle is π (180 degrees)

-- Arc length (circumference of the cone's base)
def circumference : ℝ := sector_angle * radius_circle

-- Radius of the base of the cone
def base_radius : ℝ := circumference / (2 * π)

-- Slant height of the cone
def slant_height : ℝ := radius_circle

-- Height of the cone (from the Pythagorean theorem)
def height_cone : ℝ := sqrt (slant_height ^ 2 - base_radius ^ 2)

-- Volume of the cone
def volume_cone : ℝ := (1 / 3) * π * (base_radius ^ 2) * height_cone

theorem cone_volume_is_correct : volume_cone = 9 * π * sqrt 3 :=
by
  sorry

end cone_volume_is_correct_l26_26198


namespace weight_ratio_l26_26336

noncomputable def students_weight : ℕ := 79
noncomputable def siblings_total_weight : ℕ := 116

theorem weight_ratio (S W : ℕ) (h1 : siblings_total_weight = S + W) (h2 : students_weight = S):
  (S - 5) / (siblings_total_weight - S) = 2 :=
by
  sorry

end weight_ratio_l26_26336


namespace complex_solution_l26_26730

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26730


namespace numbers_left_on_blackboard_l26_26496

theorem numbers_left_on_blackboard (n11 n12 n13 n14 n15 : ℕ)
    (h_n11 : n11 = 11) (h_n12 : n12 = 12) (h_n13 : n13 = 13) (h_n14 : n14 = 14) (h_n15 : n15 = 15)
    (total_numbers : n11 + n12 + n13 + n14 + n15 = 65) :
  ∃ (remaining1 remaining2 : ℕ), remaining1 = 12 ∧ remaining2 = 14 := 
sorry

end numbers_left_on_blackboard_l26_26496


namespace table_sum_or_zero_l26_26352

theorem table_sum_or_zero (n m : ℕ) (x : Fin n → ℝ) (y : Fin m → ℝ)
  (h : ∀ i j, x i * y j = (x i) * (y j)) : (∑ i, x i = 1 ∧ ∑ j, y j = 1) ∨ (∀ i, x i = 0) ∧ (∀ j, y j = 0) :=
by
  sorry

end table_sum_or_zero_l26_26352


namespace hyperbola_asymptote_l26_26316

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, (x^2 - y^2 / a^2) = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 :=
by
  intro h_asymptote
  sorry

end hyperbola_asymptote_l26_26316


namespace division_of_repeating_decimals_l26_26908

noncomputable def repeating_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  ⟨n, d⟩

theorem division_of_repeating_decimals :
  let x := repeating_to_fraction 54 99
  let y := repeating_to_fraction 18 99
  (x / y) = (3 : ℚ) :=
by
  -- Proof omitted as requested
  sorry

end division_of_repeating_decimals_l26_26908


namespace carlos_paid_l26_26140

theorem carlos_paid (a b c : ℝ) 
  (h1 : a = (1 / 3) * (b + c))
  (h2 : b = (1 / 4) * (a + c))
  (h3 : a + b + c = 120) :
  c = 72 :=
by
-- Proof omitted
sorry

end carlos_paid_l26_26140


namespace percentage_singing_l26_26009

def total_rehearsal_time : ℕ := 75
def warmup_time : ℕ := 6
def notes_time : ℕ := 30
def words_time (t : ℕ) : ℕ := t
def singing_time (t : ℕ) : ℕ := total_rehearsal_time - warmup_time - notes_time - words_time t
def singing_percentage (t : ℕ) : ℕ := (singing_time t * 100) / total_rehearsal_time

theorem percentage_singing (t : ℕ) : (singing_percentage t) = (4 * (39 - t)) / 3 :=
by
  sorry

end percentage_singing_l26_26009


namespace assignment_schemes_with_girl_l26_26452

theorem assignment_schemes_with_girl (boys girls : ℕ) (tasks : ℕ) (total_people : ℕ) 
  (h_boys : boys = 4) (h_girls : girls = 3) (h_tasks : tasks = 3) (h_total_people : total_people = boys + girls) :
  let total_schemes := total_people * (total_people - 1) * (total_people - 2),
      all_boys_schemes := boys * (boys - 1) * (boys - 2) in
  total_schemes - all_boys_schemes = 186 :=
by
  have h_total_schemes : total_schemes = 7 * 6 * 5 := by rw [h_total_people, ←Nat.mul_sub_left_distrib, Nat.mul_sub_right_distrib, Nat.mul_sub_right_distrib]
  have h_all_boys_schemes : all_boys_schemes = 4 * 3 * 2 := by rw [h_boys, ←Nat.mul_sub_left_distrib, Nat.mul_sub_right_distrib, Nat.mul_sub_right_distrib]
  rw [h_total_schemes, h_all_boys_schemes]
  norm_num
  simp
  sorry

end assignment_schemes_with_girl_l26_26452


namespace solve_for_z_l26_26739

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26739


namespace range_of_m_l26_26314

open Real

theorem range_of_m 
    (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
    (m : ℝ)
    (h : m * (a + 1/a) / sqrt 2 > 1) : 
    m ≥ sqrt 2 / 2 :=
sorry

end range_of_m_l26_26314


namespace bricks_needed_approx_640000_l26_26676

-- Define volumes of the brick and the wall.
def brick_volume : ℝ := 25 * 11.25 * 6
def wall_volume : ℝ := 800 * 600 * 2250

-- Define the number of bricks needed.
def num_bricks_needed : ℝ := wall_volume / brick_volume

-- Now state the problem and prove it.
theorem bricks_needed_approx_640000 :
  num_bricks_needed ≈ 640000 :=
sorry

end bricks_needed_approx_640000_l26_26676


namespace odd_and_increasing_A_odd_and_increasing_C_l26_26936

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem odd_and_increasing_A :
  is_odd (λ x : ℝ, 10^x - 10^(-x)) ∧ is_monotonically_increasing (λ x : ℝ, 10^x - 10^(-x)) :=
sorry

theorem odd_and_increasing_C :
  is_odd (λ x : ℝ, x^3) ∧ is_monotonically_increasing (λ x : ℝ, x^3) :=
sorry

end odd_and_increasing_A_odd_and_increasing_C_l26_26936


namespace trig_identity_equality_l26_26602

theorem trig_identity_equality :
  (let s15 := Real.sin (15 * Real.pi / 180)
       c20 := Real.cos (20 * Real.pi / 180)
       c165 := - Real.cos (15 * Real.pi / 180)
       c115 := - Real.cos (65 * Real.pi / 180)
       s25 := Real.sin (25 * Real.pi / 180)
       c5 := Real.cos (5 * Real.pi / 180)
       c155 := - Real.cos (25 * Real.pi / 180)
       c95 := - Real.cos (5 * Real.pi / 180)
       t1 := s15 * c20 + c165 * c115
       t2 := s25 * c5 + c155 * c95
       k30_s := 1/2
       k30_c := Real.sqrt 3 / 2
       c80 := Real.sin (10 * Real.pi / 180)
       t3 := (Real.sin 35 * Real.pi / 180) - c80
       t4 := k30_s - k30_c)
  t1 / t2 = 2 * t3 / (1 - Real.sqrt 3)
 := sorry

end trig_identity_equality_l26_26602


namespace initial_pages_read_per_week_l26_26814

-- Definitions from conditions
def initial_reading_speed : ℕ := 40 -- pages per hour
def increased_speed : ℕ := (3 * initial_reading_speed) / 2 -- 150% of initial speed
def pages_read_now : ℕ := 660 -- pages per week now
def hours_less : ℕ := 4 -- hours less per week now

-- Prove the initial pages read per week
theorem initial_pages_read_per_week :
  let hours_per_week_now := pages_read_now / increased_speed in
  let hours_per_week_initial := hours_per_week_now + hours_less in
  let pages_read_per_week_initial := hours_per_week_initial * initial_reading_speed in
  pages_read_per_week_initial = 600 :=
by
  sorry

end initial_pages_read_per_week_l26_26814


namespace evaluate_expression_l26_26952

noncomputable def ln (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression : 
  2017 ^ ln (ln 2017) - (ln 2017) ^ ln 2017 = 0 :=
by
  sorry

end evaluate_expression_l26_26952


namespace customers_added_during_lunch_rush_l26_26570

noncomputable def initial_customers := 29.0
noncomputable def total_customers_after_lunch_rush := 83.0
noncomputable def expected_customers_added := 54.0

theorem customers_added_during_lunch_rush :
  (total_customers_after_lunch_rush - initial_customers) = expected_customers_added :=
by
  sorry

end customers_added_during_lunch_rush_l26_26570


namespace average_goals_l26_26118

def num_goals_3 := 3
def num_players_3 := 2
def num_goals_4 := 4
def num_players_4 := 3
def num_goals_5 := 5
def num_players_5 := 1
def num_goals_6 := 6
def num_players_6 := 1

def total_goals := (num_goals_3 * num_players_3) + (num_goals_4 * num_players_4) + (num_goals_5 * num_players_5) + (num_goals_6 * num_players_6)
def total_players := num_players_3 + num_players_4 + num_players_5 + num_players_6

theorem average_goals :
  (total_goals / total_players : ℚ) = 29 / 7 :=
sorry

end average_goals_l26_26118


namespace solve_for_z_l26_26732

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26732


namespace all_equal_l26_26639

variable (a : ℕ → ℝ)

axiom h1 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0
axiom h2 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0
axiom h3 : a 3 - 3 * a 4 + 2 * a 5 ≥ 0
axiom h4 : ∀ n, 4 ≤ n ∧ n ≤ 98 → a n - 3 * a (n + 1) + 2 * a (n + 2) ≥ 0
axiom h99 : a 99 - 3 * a 100 + 2 * a 1 ≥ 0
axiom h100 : a 100 - 3 * a 1 + 2 * a 2 ≥ 0

theorem all_equal : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 → a i = a j := by
  sorry

end all_equal_l26_26639


namespace sum_of_abs_roots_eq_6_l26_26267

-- Given polynomial conditions
def poly := fun (x : ℝ) => x^4 - 6*x^3 + 9*x^2 + 24*x - 36

-- Lean statement: proving the sum of the absolute values of the roots equals 6
theorem sum_of_abs_roots_eq_6 : 
  (∃ x1 x2 x3 x4 : ℂ, 
  poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0 ∧ poly x4 = 0 ∧ 
  x1 + x2 + x3 + x4 = 6) :=
sorry

end sum_of_abs_roots_eq_6_l26_26267


namespace burger_cost_l26_26354

theorem burger_cost
  (B P : ℝ)
  (h₁ : P = 2 * B)
  (h₂ : P + 3 * B = 45) :
  B = 9 := by
  sorry

end burger_cost_l26_26354


namespace magnitude_of_z_l26_26637

-- Conditions
def i_unit : ℂ := complex.I
def z : ℂ := (1 + complex.I) / 2

-- Problem statement to be proven
theorem magnitude_of_z : complex.abs z = real.sqrt 2 / 2 := by
  sorry

end magnitude_of_z_l26_26637


namespace greatest_distance_between_centers_l26_26506

-- Definitions according to conditions
def rectangle_width := 18
def rectangle_height := 20
def circle_diameter := 8
def radius := circle_diameter / 2

def max_distance : ℝ :=
  ℝ.sqrt (10^2 + 12^2)  -- 2 * radius = 8, thus rectangle center distance (width-8, height-8)

theorem greatest_distance_between_centers :
  max_distance = 2 * ℝ.sqrt 61 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end greatest_distance_between_centers_l26_26506


namespace constant_term_and_sum_of_coeffs_l26_26868

theorem constant_term_and_sum_of_coeffs :
  let expr := (2 * x - 1 / x) ^ 4
  constant_term (expr) = 24 ∧ sum_of_coeffs (expr) = 1 :=
by
  sorry

end constant_term_and_sum_of_coeffs_l26_26868


namespace sum_first_80_terms_l26_26486

def sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) + (-1)^n * a n = 2 * n - 1

theorem sum_first_80_terms (a : ℕ → ℝ) (h : sequence a) : 
  (Finset.range 80).sum a = 3240 :=
sorry

end sum_first_80_terms_l26_26486


namespace average_of_original_set_l26_26102

theorem average_of_original_set (A : ℝ) (h1 : (35 * A) = (7 * 75)) : A = 15 := 
by sorry

end average_of_original_set_l26_26102


namespace intersection_A_B_l26_26672

def A : Set Real := { y | ∃ x : Real, y = Real.cos x }
def B : Set Real := { x | x^2 < 9 }

theorem intersection_A_B : A ∩ B = { y | -1 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_A_B_l26_26672


namespace f_increasing_t_range_l26_26289

section
variables {f : ℝ → ℝ} {t a x m n : ℝ}

-- Given conditions
hypothesis hf_odd : ∀ x ∈ set.Icc (-1 : ℝ) 1, f (-x) = -f x
hypothesis hf_1 : f 1 = 1
hypothesis hf_pos : ∀ m n ∈ set.Icc (-1 : ℝ) 1, m + n ≠ 0 → (f m + f n) / (m + n) > 0

-- Problem 1: Prove that f is increasing on [-1, 1]
theorem f_increasing : ∀ x1 x2 ∈ set.Icc (-1 : ℝ) 1, x1 < x2 → f x1 < f x2 := 
by sorry

-- Problem 2: Find the range of t such that f(x) ≤ t^2 - 2at + 1 for all x and a in [-1, 1]
theorem t_range : ∀ x ∈ set.Icc (-1 : ℝ) 1, ∀ a ∈ set.Icc (-1 : ℝ) 1, (f x ≤ t^2 - 2 * a * t + 1) → 
  (t ≤ -2 ∨ t = 0 ∨ t ≥ 2) := 
by sorry

end

end f_increasing_t_range_l26_26289


namespace sum_of_roots_of_quadratic_l26_26242

theorem sum_of_roots_of_quadratic :
  ∀ N : ℝ, (N^2 - 6 * N + 7 = 0) → ∃ N1 N2 : ℝ, N1 + N2 = 6 :=
by
  intro N h
  use [N1, N2]
  sorry

end sum_of_roots_of_quadratic_l26_26242


namespace even_number_of_beta_l26_26053

-- Define the set T based on given conditions
def T (α β : list ℕ) : set ℕ :=
  {t | ∃ i, t = (α.nth i - β.nth i).nat_abs}

-- Define the property E(n)
def has_property_E (n : ℕ) (α : list ℕ) : Prop :=
  (∀ i, i ∈ α → i < n) ∧ (∀ i j, i ≠ j → α.nth i ≠ α.nth j)

-- Define the theorem to be proven
theorem even_number_of_beta (n : ℕ) (α β : list ℕ) 
  (hα : has_property_E n α) (hβ : has_property_E n β) :
  ∃ (count : ℕ), count % 2 = 0 ∧ 
    ∀ (β_set : set (list ℕ)), 
      (∀ β ∈ β_set, has_property_E n β ∧ T α β = {k | k < n}) → 
      count = β_set.card :=
sorry

end even_number_of_beta_l26_26053


namespace chord_dot_product_unit_circle_l26_26375

theorem chord_dot_product_unit_circle
  (O A B : ℝ × ℝ)
  (hO : O = (0, 0))
  (hA : ‖O - A‖ = 1)
  (hB : ‖O - B‖ = 1)
  (hAB : dist A B = sqrt 2)
  (C : ℝ × ℝ)
  (hC : C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_perp : (B - A) ∙ (C - O) = 0):
  (B - A) ∙ (B - O) = 1 := by
    sorry

end chord_dot_product_unit_circle_l26_26375


namespace range_of_a_l26_26818

theorem range_of_a (a : ℝ) : 
  (∀ x, (x ≤ 1 ∨ x ≥ 3) ↔ ((a ≤ x ∧ x ≤ a + 1) → (x ≤ 1 ∨ x ≥ 3))) → 
  (a ≤ 0 ∨ a ≥ 3) :=
by
  sorry

end range_of_a_l26_26818


namespace repeating_decimal_division_l26_26921

theorem repeating_decimal_division :
  (0.\overline{54} / 0.\overline{18}) = 3 :=
by
  have h1 : 0.\overline{54} = 54 / 99 := sorry
  have h2 : 0.\overline{18} = 18 / 99 := sorry
  have h3 : (54 / 99) / (18 / 99) = 54 / 18 := sorry
  have h4 : 54 / 18 = 3 := sorry
  rw [h1, h2, h3, h4]
  exact rfl

end repeating_decimal_division_l26_26921


namespace rhombus_area_l26_26263

theorem rhombus_area (a b : ℝ) (R : ℝ) (hR : R = 25) :
  (R = 25) → a = b → (a * b = 25^2) →
  (1/2) * (sqrt (2 * a^2))^2 = 312.5 :=
by sorry

end rhombus_area_l26_26263


namespace graham_age_difference_l26_26812

theorem graham_age_difference :
  let current_year := 2021,
      mark_birth_year := 1976,
      mark_age := current_year - mark_birth_year,
      janice_age := 21,
      graham_age := 2 * janice_age,
      age_difference := mark_age - graham_age
  in
  age_difference = 3 :=
by
  sorry

end graham_age_difference_l26_26812


namespace number_of_students_in_the_course_l26_26841

variable (T : ℝ)

theorem number_of_students_in_the_course
  (h1 : (1/5) * T + (1/4) * T + (1/2) * T + 40 = T) :
  T = 800 :=
sorry

end number_of_students_in_the_course_l26_26841


namespace SallyNickelDifferenceIsZero_l26_26347

/-- Define the variables representing the number of coins. -/
def SallyHasExactlyTheFollowingCoins :=
  ∃ (n d q : ℕ), 
  n + d + q = 150 ∧ 
  5 * n + 10 * d + 25 * q = 2000 ∧
  n > 0 ∧
  d > 0 ∧
  q > 0

/-- Prove that the difference between the maximum and minimum number of nickels is 0. -/
theorem SallyNickelDifferenceIsZero : SallyHasExactlyTheFollowingCoins → ∃ k : ℕ, n ∈ k → n = 86 → |86 - n| = 0 :=
by
  sorry

end SallyNickelDifferenceIsZero_l26_26347


namespace tom_final_notebook_last_page_drawings_l26_26897

structure Notebook := 
  (pages : ℕ)
  (drawings_per_page : ℕ)

def total_drawings (notebooks : ℕ) (pages : ℕ) (drawings_per_page_old : ℕ) : ℕ :=
  notebooks * pages * drawings_per_page_old

def drawings_per_new_page (drawings_total : ℕ) (drawings_per_page_new : ℕ) : ℕ :=
  drawings_total / drawings_per_page_new

def filled_pages (notebooks_filled : ℕ) (pages : ℕ) : ℕ :=
  notebooks_filled * pages

def remaining_pages (total_pages : ℕ) (filled_pages : ℕ) (pages_in_fourth_notebook : ℕ) : ℕ :=
  total_pages - filled_pages - pages_in_fourth_notebook

def remaining_drawings (initial_drawings : ℕ) (drawings_filled : ℕ) : ℕ :=
  initial_drawings - drawings_filled

def last_page_drawings (drawings_left : ℕ) : ℕ :=
  drawings_left

theorem tom_final_notebook_last_page_drawings :
  let initial_drawings := total_drawings 5 60 8,
      new_pages := drawings_per_new_page initial_drawings 12,
      filled := filled_pages 3 60,
      drawings_filled := 3 * 60 * 12 + 45 * 12
  in last_page_drawings (remaining_drawings initial_drawings drawings_filled) = 60 := 
by
  sorry

end tom_final_notebook_last_page_drawings_l26_26897


namespace determine_z_l26_26696

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26696


namespace find_z_l26_26754

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26754


namespace balls_placement_valid_l26_26442

noncomputable def total_ways : ℕ := Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2
noncomputable def same_box_ways : ℕ := Nat.choose 3 1 * Nat.choose 4 2
noncomputable def valid_ways : ℕ := total_ways - same_box_ways

theorem balls_placement_valid : valid_ways = 72 :=
by
  have h_total : total_ways = 90 := by sorry
  have h_same_box : same_box_ways = 18 := by sorry
  have h_calculate : valid_ways = total_ways - same_box_ways := by simp [valid_ways]
  rw [h_calculate, h_total, h_same_box]
  norm_num
  exact (Nat.sub_eq_of_eq_add (by norm_num: 18 + 72 = 90)).symm

end balls_placement_valid_l26_26442


namespace petri_dishes_count_l26_26361

theorem petri_dishes_count :
  let G := 0.036 * 10^5,
      g := 79.99999999999999
  in G / g = 45 :=
by
  let G := 0.036 * 10^5
  let g := 79.99999999999999
  sorry

end petri_dishes_count_l26_26361


namespace a_n_formula_l26_26484

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 1 then (Real.sqrt 2) / 2 + 1
  else if n = 2 then Real.sqrt 2 + 1
  else sorry

theorem a_n_formula (n : ℕ) (hn : n ≥ 1) :
  a_seq n = (n : ℝ) / Real.sqrt 2 + 1 :=
by
  induction n with k hk
  case zero =>
    have : ¬(0 ≥ 1) by simp
    contradiction  
  case succ =>
    sorry

end a_n_formula_l26_26484


namespace sugar_for_cake_l26_26960

variable (total_sugar frosting_sugar cake_sugar : ℝ)

axiom total_sugar_eq : total_sugar = 0.8
axiom frosting_sugar_eq : frosting_sugar = 0.6

theorem sugar_for_cake :
  cake_sugar = total_sugar - frosting_sugar → cake_sugar = 0.2 :=
by
  intro h
  rw [total_sugar_eq, frosting_sugar_eq] at h
  exact h

#check sugar_for_cake

end sugar_for_cake_l26_26960


namespace circumcircle_fixed_point_l26_26358

theorem circumcircle_fixed_point
  (A B C: Point)
  (h_acute_triangle: is_acute_triangle A B C)
  (h_angle_A_lt_B : ∠A < ∠B)
  (h_angle_A_lt_C : ∠A < ∠C)
  (P: Point)
  (hP_BC: is_on_line_segment P B C)
  (D E: Point)
  (hD_AB: is_on_line_segment D A B)
  (hE_AC: is_on_line_segment E A C)
  (h_BP_PD: distance B P = distance P D)
  (h_CP_PE: distance C P = distance P E):
  ∃ H: Point, is_orthocenter H A B C ∧ 
             ∀ P, is_on_line_segment P B C →
             ∀ D E, is_on_line_segment D A B → 
             is_on_line_segment E A C → 
             distance B P = distance P D → 
             distance C P = distance P E → 
             H ∈ circumcircle A D E :=
sorry

end circumcircle_fixed_point_l26_26358


namespace linda_age_13_l26_26067

variable (J L : ℕ)

-- Conditions: 
-- 1. Linda is 3 more than 2 times the age of Jane.
-- 2. In five years, the sum of their ages will be 28.
def conditions (J L : ℕ) : Prop :=
  L = 2 * J + 3 ∧ (J + 5) + (L + 5) = 28

-- Question/answer to prove: Linda's current age is 13.
theorem linda_age_13 (J L : ℕ) (h : conditions J L) : L = 13 :=
by
  sorry

end linda_age_13_l26_26067


namespace solve_equation1_solve_equation2_solve_system1_solve_system2_l26_26091

-- Problem 1
theorem solve_equation1 (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 :=
by sorry

-- Problem 2
theorem solve_equation2 (x : ℚ) : (3 * x - 2) / 2 = (4 * x + 2) / 3 - 1 → x = 4 :=
by sorry

-- Problem 3
theorem solve_system1 (x y : ℚ) : (3 * x - 7 * y = 8) ∧ (2 * x + y = 11) → x = 5 ∧ y = 1 :=
by sorry

-- Problem 4
theorem solve_system2 (a b c : ℚ) : (a - b + c = 0) ∧ (4 * a + 2 * b + c = 3) ∧ (25 * a + 5 * b + c = 60) → (a = 3) ∧ (b = -2) ∧ (c = -5) :=
by sorry

end solve_equation1_solve_equation2_solve_system1_solve_system2_l26_26091


namespace maximum_BP_squared_l26_26032

-- Definitions for the problem
variable {α : Type*}
variables (ω : α) [Circle ω] (A B C T P : α)
variables [OnDiameter A B ω] (AB : ℝ) (h_AB : AB = 18)
variables [IsExtensionOfDiameter ω A B C]
variables [Tangency T ω C]
variables [FootPerpendicular P A C T]

-- The target theorem
theorem maximum_BP_squared (A B C T P : α) [OnDiameter A B ω] [IsExtensionOfDiameter ω A B C] [Tangency T ω C] [FootPerpendicular P A C T] 
  (h_AB : AB = 18) :
  ∃ m, m = BP.maxValue ∧ m^2 = 432 :=
sorry

end maximum_BP_squared_l26_26032


namespace find_coordinate_a_l26_26505

theorem find_coordinate_a :
  let C_large := (0, 0, Real.sqrt 104)
  let C_small := (0, 0, Real.sqrt 104 - 4)
  let P := (10, 2)
  let S := λ a, (a, a)
  (P.1^2 + P.2^2 = 104) →
  (∃ a : Real, S a ∈ set_of (λ (x : Real × Real), x.1^2 + x.2^2 = (Real.sqrt 104 - 4)^2)) →
  a = Real.sqrt (60 - 4 * Real.sqrt 104) :=
by
  sorry

end find_coordinate_a_l26_26505


namespace total_books_l26_26778

-- Define the number of books Stu has
def Stu_books : ℕ := 9

-- Define the multiplier for Albert's books
def Albert_multiplier : ℕ := 4

-- Define the number of books Albert has
def Albert_books : ℕ := Albert_multiplier * Stu_books

-- Prove that the total number of books is 45
theorem total_books:
  Stu_books + Albert_books = 45 :=
by 
  -- This is where the proof steps would go, but we skip it for now 
  sorry

end total_books_l26_26778


namespace a3_eq_5_l26_26368

-- Define the geometric sequence and its properties
variables {a : ℕ → ℝ} {q : ℝ}

-- Assumptions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a 1 * (q ^ n)
axiom a1_pos : a 1 > 0
axiom a2a4_eq_25 : a 2 * a 4 = 25
axiom geom : geom_seq a q

-- Statement to prove
theorem a3_eq_5 : a 3 = 5 :=
by sorry

end a3_eq_5_l26_26368


namespace sum_of_first_ten_terms_arithmetic_sequence_l26_26434

theorem sum_of_first_ten_terms_arithmetic_sequence :
  ∀ (a_1 d : ℕ), a_1 = 3 → d = 3 → (∑ k in finRange 10, a_1 + k*d) = 165 :=
by
  intros a_1 d ha hd
  rw [ha, hd]
  sorry

end sum_of_first_ten_terms_arithmetic_sequence_l26_26434


namespace intersection_A_B_C_subset_complement_A_inter_B_l26_26322

-- Define the set A
def A := {x : ℝ | x^2 - x - 6 ≤ 0}

-- Define the set B
def B := {x : ℝ | (1/2 : ℝ) ≤ 2^x ∧ 2^x ≤ 16}

-- Define the intersection A ∩ B
def A_inter_B := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Prove that A ∩ B = {x | -1 ≤ x ≤ 3}
theorem intersection_A_B : A ∩ B = A_inter_B := sorry

-- Define the set C
def C (m : ℝ) := {x : ℝ | m - 1 < x ∧ x < m + 1}

-- Define the complement of A ∩ B in U (where U = ℝ)
def complement_A_inter_B := {x : ℝ | x > 3 ∨ x < -1}

-- Prove that if C ⊆ complement of A ∩ B, then m ≤ -2 or m ≥ 4
theorem C_subset_complement_A_inter_B (m : ℝ) : C(m) ⊆ complement_A_inter_B → (m ≤ -2 ∨ m ≥ 4) := sorry

end intersection_A_B_C_subset_complement_A_inter_B_l26_26322


namespace correct_statements_l26_26293

variable {R : Type*} [OrderedAddCommGroup R]

def is_even (f : R → R) : Prop :=
  ∀ x, f x = f (-x)

def functional_eqn (f : R → R) : Prop :=
  ∀ x, f (x + 6) = f x + f 3

theorem correct_statements (f : R → R) 
  (h_even : is_even f)
  (h_fun_eq : functional_eqn f) :
  (f 3 = 0) ∧ (f (-3) = 0) ∧ (∀ x, f (6 + x) = f (6 - x)) :=
by 
  sorry

end correct_statements_l26_26293


namespace sum_of_n_values_l26_26927

theorem sum_of_n_values (sum_n : ℕ) : (∀ n : ℕ, 0 < n ∧ 24 % (2 * n - 1) = 0) → sum_n = 3 :=
by
  sorry

end sum_of_n_values_l26_26927


namespace g_77_equals_1011_l26_26471

def g : ℤ → ℤ
| n := if n >= 1010 then n - 4 else g (g (n + 7))

theorem g_77_equals_1011 : g 77 = 1011 :=
by sorry

end g_77_equals_1011_l26_26471


namespace kho_kho_only_l26_26186

variable (K H B : ℕ)

theorem kho_kho_only :
  (K + B = 10) ∧ (H + 5 = H + B) ∧ (B = 5) ∧ (K + H + B = 45) → H = 35 :=
by
  intros h
  sorry

end kho_kho_only_l26_26186


namespace complex_number_solution_l26_26687

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26687


namespace max_value_of_polynomial_expr_l26_26448

open Real

noncomputable def polynomial (a b c : ℝ) : (ℝ → ℝ) :=
  λ x, x^3 + a * x^2 + b * x + c

theorem max_value_of_polynomial_expr (a b c λ : ℝ)
  (hλ_pos : 0 < λ)
  (roots : {x1 x2 x3 : ℝ // polynomial a b c x1 = 0 ∧ polynomial a b c x2 = 0 ∧ polynomial a b c x3 = 0})
  (h_diff : roots.x2 - roots.x1 = λ)
  (h_ineq : roots.x3 > (roots.x1 + roots.x2) / 2) :
  (2 * a^3 + 27 * c - 9 * a * b) / λ^3 ≤ (3 * sqrt 3) / 2 :=
by
  sorry

end max_value_of_polynomial_expr_l26_26448


namespace sum_of_odd_integers_between_11_and_41_l26_26181

theorem sum_of_odd_integers_between_11_and_41 :
  ∑ i in finset.filter (λ x, x % 2 = 1) (finset.Icc 11 41), i = 416 := by
  sorry

end sum_of_odd_integers_between_11_and_41_l26_26181


namespace no_nonzero_integer_solution_l26_26085

theorem no_nonzero_integer_solution (x y z : ℤ) (h : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  x^2 + y^2 ≠ 3 * z^2 :=
by
  sorry

end no_nonzero_integer_solution_l26_26085


namespace probability_of_A_probability_of_B_probability_of_union_A_B_probability_of_intersection_A_B_l26_26197

noncomputable def die_sample_space := {1, 2, 3, 4, 5, 6}
def A := {n ∈ die_sample_space | n ≥ 3}
def B := {n ∈ die_sample_space | n % 2 = 1}

theorem probability_of_A : ∃ p, p = 2 / 3 → (nat.card A / nat.card die_sample_space) = p := by
  sorry

theorem probability_of_B : ∃ p, p = 1 / 2 → (nat.card B / nat.card die_sample_space) = p := by
  sorry

theorem probability_of_union_A_B : ∃ p, p = 5 / 6 → (nat.card (A ∪ B) / nat.card die_sample_space) = p := by
  sorry

theorem probability_of_intersection_A_B : ∃ p, p = 1 / 3 → (nat.card (A ∩ B) / nat.card die_sample_space) = p := by
  sorry

end probability_of_A_probability_of_B_probability_of_union_A_B_probability_of_intersection_A_B_l26_26197


namespace bike_ride_time_l26_26228

theorem bike_ride_time (y : ℚ) : 
  let speed_fast := 25
  let speed_slow := 10
  let total_distance := 170
  let total_time := 10
  (speed_fast * y + speed_slow * (total_time - y) = total_distance) 
  → y = 14 / 3 := 
by 
  sorry

end bike_ride_time_l26_26228


namespace minimum_value_ineq_l26_26931

theorem minimum_value_ineq (x : ℝ) (hx : x >= 4) : x + 4 / (x - 1) >= 5 := by
  sorry

end minimum_value_ineq_l26_26931


namespace smaller_cube_volume_l26_26563

theorem smaller_cube_volume
  (d : ℝ) (s : ℝ) (V : ℝ)
  (h1 : d = 12)  -- condition: diameter of the sphere equals the edge length of the larger cube
  (h2 : d = s * Real.sqrt 3)  -- condition: space diagonal of the smaller cube equals the diameter of the sphere
  (h3 : s = 12 / Real.sqrt 3)  -- condition: side length of the smaller cube
  (h4 : V = s^3)  -- condition: volume of the cube with side length s
  : V = 192 * Real.sqrt 3 :=  -- proving the volume of the smaller cube
sorry

end smaller_cube_volume_l26_26563


namespace f_x_minus_x_in_range_l26_26061

variable (c : ℝ)
variable (f : ℝ → ℝ)
variable (b : ℝ)
variable (x : ℝ)

# Check the conditions
# Condition 1: f is continuous on [0,1]
variable continuous_f : ContinuousOn f (Set.Icc 0 1)

# Condition 2: b f(2 x) = f(x) for 0 ≤ x ≤ 1/2
variable h1 : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1/2 → b * f (2 * x) = f x

# Condition 3: f(x) = b + (1-b) f(2 x - 1) for 1/2 ≤ x ≤ 1
variable h2 : ∀ (x : ℝ), 1/2 ≤ x ∧ x ≤ 1 → f x = b + (1 - b) * f (2 * x - 1)

# Condition 4: b = (1 + c) / (2 + c) ∧ c > 0
variable b_def : b = (1 + c) / (2 + c)
variable c_positive : c > 0

-- The statement we need to prove
theorem f_x_minus_x_in_range {x : ℝ} (hx : 0 < x ∧ x < 1) : 0 < f x - x ∧ f x - x < c := 
  sorry

end f_x_minus_x_in_range_l26_26061


namespace residue_of_T_l26_26060

theorem residue_of_T :
  let T := (List.range 2023).sum_by (λ n, if n % 2 = 0 then n + 1 else -n)
  T % 2024 = 1012 :=
by {
  let T := (List.range 2023).sum_by (λ n, if n % 2 = 0 then n + 1 else -n),
  sorry
}

end residue_of_T_l26_26060


namespace slope_of_perpendicular_line_l26_26615

theorem slope_of_perpendicular_line (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, a * x - b * y = c → m = - (b / a) :=
by
  -- Here we state the definition and conditions provided in the problem
  -- And indicate what we want to prove (that the slope is -b/a in this case)
  sorry

end slope_of_perpendicular_line_l26_26615


namespace parabola_directrix_Q_coordinates_range_of_slope_passing_through_Q_l26_26227

theorem parabola_directrix_Q_coordinates : 
  let parabola := λ (y x : ℝ), y^2 = 8 * x,
      directrix := λ (x : ℝ), x = -4,
      Q := (-4 : ℝ, 0 : ℝ)
  in Q.1 = -4 ∧ Q.2 = 0 :=
by
  let parabola_xy := λ (y x : ℝ), y^2 = 8 * x
  let directrix_x := λ (x : ℝ), x = -4
  let Q_coords := (-4 : ℝ, 0 : ℝ)
  show Q_coords.1 = -4 ∧ Q_coords.2 = 0
  sorry

theorem range_of_slope_passing_through_Q : 
  let parabola := λ (y x : ℝ), y^2 = 8 * x,
      Q := (-4 : ℝ, 0 : ℝ),
      line := λ (m x : ℝ), 0 = m * (x + 4) 
  in ∀ (m : ℝ), (8 * m - 8)^2 - 64 * m^2 ≥ 0 → m ≤ 1 / 2 :=
by
  let parabola_xy := λ (y x : ℝ), y^2 = 8 * x
  let Q_coords := (-4 : ℝ, 0 : ℝ)
  let line_equation := λ (m x : ℝ), 0 = m * (x + 4)
  show ∀ (m : ℝ), (8 * m - 8)^2 - 64 * m^2 ≥ 0 → m ≤ 1 / 2
  sorry

end parabola_directrix_Q_coordinates_range_of_slope_passing_through_Q_l26_26227


namespace find_ab_l26_26943

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 :=
by
  sorry

end find_ab_l26_26943


namespace general_term_of_sequence_l26_26886

def sum_sequence (n : ℕ) : ℤ := 2 * n ^ 2 - 3 * n

theorem general_term_of_sequence (n : ℕ) (h : n ≥ 1) : 
  let a_n : ℕ → ℤ := λ n, sum_sequence n - sum_sequence (n - 1)
  in a_n n = 4 * n - 5 :=
by
  sorry

end general_term_of_sequence_l26_26886


namespace sandy_total_change_l26_26849

section ShoppingSpree

variable (price_football₁ price_baseball₁ note₁ 
          price_basketball₂ note₂ : ℝ)
          (num_footballs₁ num_baseballs₁ num_basketballs₂ : ℕ)

-- Conditions from the problem
def first_store_cost : ℝ := num_footballs₁ * price_football₁ + num_baseballs₁ * price_baseball₁
def change_first_store : ℝ := note₁ - first_store_cost

def second_store_cost : ℝ := num_basketballs₂ * price_basketball₂
def change_second_store : ℝ := note₂ - second_store_cost

-- Given concrete values
def football_price : ℝ := 9.14
def baseball_price : ℝ := 6.81
def basketball_price : ℝ := 7.95
def note_first_store : ℝ := 50.00
def note_second_store : ℝ := 20.00

def num_footballs : ℕ := 3
def num_baseballs : ℕ := 2
def num_basketballs : ℕ := 4

-- The final proof problem
theorem sandy_total_change :
  let change_first := note_first_store - (num_footballs * football_price + num_baseballs * baseball_price)
  let change_second := note_second_store - (num_basketballs * basketball_price)
  change_first = 8.96 ∧ change_second = 0.00 :=
by
  sorry

end ShoppingSpree

end sandy_total_change_l26_26849


namespace average_runs_in_30_matches_l26_26180

theorem average_runs_in_30_matches (avg_runs_15: ℕ) (avg_runs_20: ℕ) 
    (matches_15: ℕ) (matches_20: ℕ)
    (h1: avg_runs_15 = 30) (h2: avg_runs_20 = 15)
    (h3: matches_15 = 15) (h4: matches_20 = 20) : 
    (matches_15 * avg_runs_15 + matches_20 * avg_runs_20) / (matches_15 + matches_20) = 25 := 
by 
  sorry

end average_runs_in_30_matches_l26_26180


namespace solve_for_z_l26_26736

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26736


namespace expected_value_of_remaining_card_l26_26195

theorem expected_value_of_remaining_card :
  let deck : List ℕ := List.range 1 101
  (expected_value (label_of_remaining_card deck) = 467 / 8 := 
begin
  let cards : List ℕ := List.range 1 101,
  -- The remaining steps to calculate expectation go here...
  sorry
end)

end expected_value_of_remaining_card_l26_26195


namespace ratio_kids_ticket_to_adult_ticket_l26_26987

theorem ratio_kids_ticket_to_adult_ticket (A K : ℝ) 
  (total_people : ℕ) (num_kids : ℕ) (total_cost : ℝ) (soda_cost : ℝ) 
  (admission_price : ℝ) (discount : ℝ) :
  total_people = 10 →
  num_kids = 4 →
  total_cost = 197 →
  soda_cost = 5 →
  admission_price = 30 →
  discount = 0.2 →
  6 * admission_price = 180 →
  0.80 * (6 * admission_price + 4* K) = 192 →
  4 * K = 240 - 180 →
  (K : A) = 1 : 2 :=
sorry

end ratio_kids_ticket_to_adult_ticket_l26_26987


namespace problem_statement_l26_26414

theorem problem_statement :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, f(f(x) + y) = f(x^2 - y) + 4 * f(x) * y + 3 * y) →
  (let f3 := {y : ℝ | ∃ x, f(x) = y ∧ x = 3} in
  (f3 = {-3/4, 9} ∧
  (let n := 2 in let s := -3/4 + 9 in n * s = 16.5)))) :=
sorry

end problem_statement_l26_26414


namespace solve_log_equation_l26_26853

theorem solve_log_equation :
  ∀ x : ℝ, (log 3 (3 * 2^x + 5) - log 3 (4^x + 1) = 0) → x = 2 := 
by 
  sorry

end solve_log_equation_l26_26853


namespace solve_log_expression_l26_26089

theorem solve_log_expression (x : ℝ) : log 8 x + log 2 (x ^ 3) = 15 → x = 16 * real.sqrt 2 :=
by
  sorry

end solve_log_expression_l26_26089


namespace lines_concurrent_or_parallel_l26_26644

variables (A B C D P Q R S O : Type)
variables [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q] [Point R] [Point S] [Point O]

-- Definitions for the given convex quadrilateral and the intersecting lines
def is_convex_quadrilateral (A B C D : Type) [Point A] [Point B] [Point C] [Point D] : Prop := sorry

def on_segment (P A B : Type) [Point P] [Point A] [Point B] : Prop := sorry
def lines_intersect (P R Q S O : Type) [Point P] [Point R] [Point Q] [Point S] [Point O] : Prop := sorry

def has_incircle (A P O S : Type) [Point A] [Point P] [Point O] [Point S] : Prop := sorry
def quadrilaterals_have_incircles (A B C D P Q R S O : Type) [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q] [Point R] [Point S] [Point O] : Prop :=
  has_incircle A P O S ∧ has_incircle B Q O P ∧ has_incircle C R O Q ∧ has_incircle D S O R

-- Definitions for the product of ratios condition using Menelaus' theorem
def menelaus_theorem_condition (A B C D P Q R S : Type) [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q] [Point R] [Point S] : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧
  let r1 := (AP / PB) in
  let r2 := (BQ / QC) in
  let r3 := (CR / RD) in
  let r4 := (DS / SA) in
  (r1 * r2 * r3 * r4) = 1

-- Proof goal for the Lean statement
theorem lines_concurrent_or_parallel
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : on_segment P A B)
  (h3 : on_segment Q B C)
  (h4 : on_segment R C D)
  (h5 : on_segment S D A)
  (h6 : lines_intersect P R Q S O)
  (h7 : quadrilaterals_have_incircles A B C D P Q R S O)
  (h8 : menelaus_theorem_condition A B C D P Q R S):
  (lines_intersect A C P Q) ∨ (lines_intersect A C R S) ∨ (are_parallel A C P Q R S) := sorry

end lines_concurrent_or_parallel_l26_26644


namespace find_number_of_terms_l26_26887

variable (a d : ℝ) (n : ℕ)

-- Condition 1: The sum of the first 13 terms is 50% of the sum of the last 13 terms
def sum_first_13 := (13 / 2) * (2 * a + 12 * d)
def sum_last_13 := (13 / 2) * (4 * a + (2 * n - 14) * d)

axiom (H1 : sum_first_13 a d = 0.5 * sum_last_13 a d n)

-- Condition 2: The ratio of sums excluding first 3 terms and last 3 terms is 6:5
def sum_excluding_first_3 := ((n - 3) / 2) * (2 * (a + 3 * d) + (n - 4) * d)
def sum_excluding_last_3 := ((n - 3) / 2) * (2 * a + (n - 4) * d)

axiom (H2 : 6 * sum_excluding_first_3 a d n = 5 * sum_excluding_last_3 a d n)

-- Statement to prove: Given the conditions, prove that n = 24
theorem find_number_of_terms : n = 24 :=
  by
  sorry

end find_number_of_terms_l26_26887


namespace quadratic_grid_fourth_column_l26_26998

theorem quadratic_grid_fourth_column 
  (grid : ℕ → ℕ → ℝ)
  (row_quadratic : ∀ i : ℕ, (∃ a b c : ℝ, ∀ n : ℕ, grid i n = a * n^2 + b * n + c))
  (col_quadratic : ∀ j : ℕ, j ≤ 3 → (∃ a b c : ℝ, ∀ n : ℕ, grid n j = a * n^2 + b * n + c)) :
  ∃ a b c : ℝ, ∀ n : ℕ, grid n 4 = a * n^2 + b * n + c := 
sorry

end quadratic_grid_fourth_column_l26_26998


namespace num_divisors_of_24_multiple_of_6_l26_26857

theorem num_divisors_of_24_multiple_of_6 : 
  {b : ℕ | b > 0 ∧ 24 % b = 0 ∧ b % 6 = 0}.card = 3 :=
by
  sorry

end num_divisors_of_24_multiple_of_6_l26_26857


namespace solve_imaginary_eq_l26_26764

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26764


namespace circumscribed_quadrilateral_l26_26396

open EuclideanGeometry

theorem circumscribed_quadrilateral (ABCD : Type*)
  [h : has_incircle ABCD]
  (K L M N : Point)
  (K1 L1 M1 N1 : Point)
  (hK : ext_angle_bisector_eq K DAB ABC)
  (hL : ext_angle_bisector_eq L ABC BCD)
  (hM : ext_angle_bisector_eq M BCD CDA)
  (hN : ext_angle_bisector_eq N CDA DAB)
  (hK1 : orthocenter_eq K1 ABK)
  (hL1 : orthocenter_eq L1 BCL)
  (hM1 : orthocenter_eq M1 CDM)
  (hN1 : orthocenter_eq N1 DAN) :
  is_parallelogram K1 L1 M1 N1 :=
sorry

end circumscribed_quadrilateral_l26_26396


namespace complex_number_solution_l26_26682

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26682


namespace solve_for_x_l26_26456

theorem solve_for_x : ∀ x : ℝ, 2^(2*x - 6) = 8^(x + 3) → x = -15 := by
  sorry

end solve_for_x_l26_26456


namespace max_student_count_l26_26799

theorem max_student_count
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : (x1 + x2 + x3 + x4 + x5) / 5 = 7)
  (h2 : ((x1 - 7) ^ 2 + (x2 - 7) ^ 2 + (x3 - 7) ^ 2 + (x4 - 7) ^ 2 + (x5 - 7) ^ 2) / 5 = 4)
  (h3 : ∀ i j, i ≠ j → List.nthLe [x1, x2, x3, x4, x5] i sorry ≠ List.nthLe [x1, x2, x3, x4, x5] j sorry) :
  max x1 (max x2 (max x3 (max x4 x5))) = 10 := 
sorry

end max_student_count_l26_26799


namespace complex_number_solution_l26_26679

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26679


namespace count_both_axisymmetric_and_centrally_symmetric_l26_26995

-- Define predicates for axisymmetry and central symmetry
def isAxisymmetric (shape : Type) : Prop := sorry
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Define each shape as a type
inductive Shape
| Parallelogram
| EquilateralTriangle
| LineSegment
| Rhombus
| Square
| Trapezoid

open Shape

-- Definitions based on given conditions
@[simp] def ParallelogramProperties : isCentrallySymmetric Parallelogram ∧ ¬ isAxisymmetric Parallelogram := sorry
@[simp] def EquilateralTriangleProperties : isAxisymmetric EquilateralTriangle ∧ ¬ isCentrallySymmetric EquilateralTriangle := sorry
@[simp] def LineSegmentProperties : isAxisymmetric LineSegment ∧ isCentrallySymmetric LineSegment := sorry
@[simp] def RhombusProperties : isAxisymmetric Rhombus ∧ isCentrallySymmetric Rhombus := sorry
@[simp] def SquareProperties : isAxisymmetric Square ∧ isCentrallySymmetric Square := sorry
@[simp] def TrapezoidProperties : ¬ isAxisymmetric Trapezoid ∧ ¬ isCentrallySymmetric Trapezoid := sorry

-- The proof problem: prove that the count of figures that are both axisymmetric and centrally symmetric is 3
theorem count_both_axisymmetric_and_centrally_symmetric :
  (∃ shapes : List Shape, 
   shapes = [LineSegment, Rhombus, Square] ∧
   ∀ shape ∈ shapes, isAxisymmetric shape ∧ isCentrallySymmetric shape) ‑>
  List.length [LineSegment, Rhombus, Square] = 3 :=
by
  sorry

end count_both_axisymmetric_and_centrally_symmetric_l26_26995


namespace regular_polygon_sides_160_l26_26204

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end regular_polygon_sides_160_l26_26204


namespace pq_reciprocal_inverse_l26_26435

variable {A B C P Q : Type}
variables [AffineSpace ℝ P] [Triangle ABC] [EquilateralTriangle ABC]
variables (P : Point) (Q : Point) [OnArcBC P]
variables (A B C : RealPlane.Point) (BC : Segment B C)
variables [OnSegment Q BC]
variables [Circumcircle (Triangle ABC)]

theorem pq_reciprocal_inverse (h₁ : EquilateralTriangle ABC)
    (h₂ : OnArcBC P)
    (h₃ : IntersectSegment AP BC Q)
    (h₄ :  OnCircumcircle (Triangle ABC) P) :
  1 / dist Q P = 1 / dist Q B + 1 / dist Q C :=
  sorry

end pq_reciprocal_inverse_l26_26435


namespace zoe_takes_correct_amount_of_money_l26_26528

def numberOfPeople : ℕ := 6
def costPerSoda : ℝ := 0.5
def costPerPizza : ℝ := 1.0

def totalCost : ℝ := (numberOfPeople * costPerSoda) + (numberOfPeople * costPerPizza)

theorem zoe_takes_correct_amount_of_money : totalCost = 9 := sorry

end zoe_takes_correct_amount_of_money_l26_26528


namespace batsman_average_20th_l26_26545

noncomputable def average_after_20th (A : ℕ) : ℕ :=
  let total_runs_19 := 19 * A
  let total_runs_20 := total_runs_19 + 85
  let new_average := (total_runs_20) / 20
  new_average
  
theorem batsman_average_20th (A : ℕ) (h1 : 19 * A + 85 = 20 * (A + 4)) : average_after_20th A = 9 := by
  sorry

end batsman_average_20th_l26_26545


namespace ball_radius_and_surface_area_l26_26957

theorem ball_radius_and_surface_area (d h r : ℝ) (radius_eq : d / 2 = 6) (depth_eq : h = 2) 
  (pythagorean : (r - h)^2 + (d / 2)^2 = r^2) :
  r = 10 ∧ (4 * Real.pi * r^2 = 400 * Real.pi) :=
by
  sorry

end ball_radius_and_surface_area_l26_26957


namespace isosceles_triangle_base_and_area_l26_26106

-- Definitions of conditions
/-- Congruent sides of the isosceles triangle -/
def side_length : ℝ := 8

/-- Perimeter of the isosceles triangle -/
def perimeter : ℝ := 30

-- Definitions to be proved
/-- Base of the isosceles triangle, to be proved to be 14 -/
def base_length (b : ℝ) : Prop := 2 * side_length + b = perimeter

/-- Height of the triangle calculated using Pythagorean Theorem -/
def height (h : ℝ) : Prop := (side_length)^2 = (b / 2)^2 + h^2

/-- Area of the triangle, to be proved to be 7 * sqrt(15) -/
def area (A : ℝ) : Prop := A = (1 / 2) * b * h

/-- Complete problem combining base length and area to be proved -/
theorem isosceles_triangle_base_and_area :
  ∃ (b h : ℝ), base_length b ∧ height h ∧ area ((1 / 2) * b * h) := by
  sorry

end isosceles_triangle_base_and_area_l26_26106


namespace no_real_sqrt_neg_six_pow_three_l26_26160

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end no_real_sqrt_neg_six_pow_three_l26_26160


namespace measure_angle_A_l26_26847

variables (A C : ℝ) (ABCD_is_inscribed : Prop) (angle_AC_ratio : A = 4 / 5 * C)

-- Given that ABCD is an inscribed quadrilateral and the ratio of angles A to C is 4:5, 
-- we need to prove that the measure of angle A is 80 degrees.
theorem measure_angle_A : ABCD_is_inscribed → angle_AC_ratio → A = 80 :=
by
  sorry

end measure_angle_A_l26_26847


namespace arith_seq_a1_a2_a3_sum_l26_26649

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a1_a2_a3_sum (a : ℕ → ℤ) (h_seq : arithmetic_seq a)
  (h1 : a 1 = 2) (h_sum : a 1 + a 2 + a 3 = 18) :
  a 4 + a 5 + a 6 = 54 :=
sorry

end arith_seq_a1_a2_a3_sum_l26_26649


namespace probability_seven_odds_in_ten_rolls_l26_26150

theorem probability_seven_odds_in_ten_rolls : 
  (∃ n : ℕ, n = 10) ∧ (∀ k, k = 7) →
  (nat.combination 10 7 / 2^10) = (15 / 128) :=
by sorry

end probability_seven_odds_in_ten_rolls_l26_26150


namespace part1_part2_l26_26832

variables {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Condition 2: the given inequality holds for any a, b with a + b ≠ 0
def given_condition (f : ℝ → ℝ) := ∀ a b : ℝ, a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Part 1: Prove that f is an increasing function on ℝ
theorem part1 (h1 : is_odd_function f) (h2 : given_condition f) : ∀ x y : ℝ, x < y → f x < f y :=
sorry

-- Part 2: Determine the range of m such that the inequality holds for all x
theorem part2 (h1 : is_odd_function f) (h2 : given_condition f) : 
  ∃ m : ℝ, (∀ x : ℝ, f (m * 2 ^ x) + f (2 ^ x - 4 ^ x + m) < 0) ↔ m ∈ Ioo (-∞) (-3 + 2 * Real.sqrt 2) :=
sorry

end part1_part2_l26_26832


namespace fibonacci_mod_5_50th_term_l26_26101

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_mod_5_50th_term :
  fibonacci 50 % 5 = 0 :=
by sorry

end fibonacci_mod_5_50th_term_l26_26101


namespace find_b_l26_26463

theorem find_b (a b c : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 1 < c):
  (∀ N : ℝ, N ≠ 1 → (N^(3/a) * N^(2/(ab)) * N^(1/(abc)) = N^(39/48))) → b = 4 :=
  by
  sorry

end find_b_l26_26463


namespace num_ways_apple_sharing_l26_26214

def apple_sharing (a b c : ℕ) : Prop :=
  a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

theorem num_ways_apple_sharing : 
  (∑ a b c, if apple_sharing a b c then 1 else 0) = 253 :=
by
  -- The proof will be provided here.
  sorry

end num_ways_apple_sharing_l26_26214


namespace both_firms_participate_condition_both_firms_will_participate_social_nonoptimal_participation_l26_26537

section RD

variables (V IC α : ℝ) (0 < α ∧ α < 1)

-- Condition for part (a)
def participation_condition : Prop :=
  α * V * (1 - 0.5 * α) ≥ IC

-- Part (b) Definition
def firms_participate_when : Prop :=
  V = 16 ∧ α = 0.5 ∧ IC = 5

-- Part (c) Definition
def social_optimal : Prop :=
  let total_profit_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC) in
  let total_profit_one := α * V - IC in
  total_profit_one > total_profit_both

-- Theorem for part (a)
theorem both_firms_participate_condition : participation_condition V IC α :=
sorry

-- Theorem for part (b)
theorem both_firms_will_participate (h : firms_participate_when V IC α) : participation_condition 16 5 0.5 :=
sorry

-- Theorem for part (c)
theorem social_nonoptimal_participation (h : firms_participate_when V IC α) : social_optimal 16 IC 0.5 :=
sorry

end RD

end both_firms_participate_condition_both_firms_will_participate_social_nonoptimal_participation_l26_26537


namespace curve_is_parabola_l26_26611

theorem curve_is_parabola (r θ : ℝ) (h : r = 1 / (1 - Math.sin θ)) :
  ∃ (x y : ℝ), (x = r * Math.cos θ) ∧ (y = r * Math.sin θ) ∧ (x^2 = 1 + 2 * y) :=
by
  sorry

end curve_is_parabola_l26_26611


namespace compare_values_l26_26117

variable {f : ℝ → ℝ}

-- Cond1: f(x) is an even function
def even_function (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)

-- Cond2: f(x) is monotonically increasing on [0, +∞)
def monotonically_increasing (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f(x) < f(y)

-- Logarithmic transformations
noncomputable def log_1_by_3_1_by_2 := Real.logb (1/3) (1/2)
noncomputable def log_2_1_by_3 := Real.logb 2 (1/3)
noncomputable def sqrt_5 := Real.sqrt 5

-- Values a, b, c
noncomputable def a := f log_1_by_3_1_by_2
noncomputable def b := f log_2_1_by_3
noncomputable def c := f sqrt_5

-- Main theorem statement
theorem compare_values 
  (h_even: even_function f) 
  (h_monotonic: monotonically_increasing f) 
  (h_transformed_a : a = f (Real.logb 3 2)) 
  (h_transformed_b : b = f (Real.logb 2 3)) : 
  c > b ∧ b > a := by 
  sorry

end compare_values_l26_26117


namespace num_natural_numbers_divisible_303_l26_26613

theorem num_natural_numbers_divisible_303 :
  {k : ℕ | k ≤ 242400 ∧ 303 ∣ (k^2 + 2 * k)}.card = 3200 := by
  sorry

end num_natural_numbers_divisible_303_l26_26613


namespace area_of_AFCH_l26_26075

-- Define the lengths of the sides of the rectangles
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the problem statement
theorem area_of_AFCH :
  let intersection_area := min BC FG * min EF AB
  let total_area := AB * FG
  let outer_ring_area := total_area - intersection_area
  intersection_area + outer_ring_area / 2 = 52.5 :=
by
  -- Use the values of AB, BC, EF, and FG to compute
  sorry

end area_of_AFCH_l26_26075


namespace no_constant_C_exists_l26_26623

def num_divisors (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (λ k, k > 0 ∧ n % k = 0).card

def euler_totient (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, Nat.coprime k n).card

theorem no_constant_C_exists :
  ∀ (C : ℝ), ∃ (n : ℕ), (n ≥ 1) → (1 : ℝ) < (↑(euler_totient (num_divisors n)) / ↑(num_divisors (euler_totient n))) / C := 
sorry

end no_constant_C_exists_l26_26623


namespace sum_positive_factors_gt_one_of_36_l26_26928

-- Define the condition
def is_positive_factor_greater_than_one (n d : ℕ) : Prop :=
  d > 1 ∧ d ∣ n

-- Given number 
def n : ℕ := 36

-- Define the sum of all positive factors of n that are greater than one
def sum_of_factors (n : ℕ) : ℕ :=
  ∑ d in Finset.filter (is_positive_factor_greater_than_one n) (Finset.range (n + 1)), d

-- Math proof problem statement
theorem sum_positive_factors_gt_one_of_36 : sum_of_factors n = 90 := 
by 
  -- Note: the steps in the proof would go here
  sorry

end sum_positive_factors_gt_one_of_36_l26_26928


namespace fraction_subtraction_l26_26932

theorem fraction_subtraction (x : ℝ) : (8000 * x - (0.05 / 100 * 8000) = 796) → x = 0.1 :=
by
  sorry

end fraction_subtraction_l26_26932


namespace find_income_l26_26531

def citizen_income (I : ℝ) : ℝ :=
  if I <= 40000 then 0.15 * I else 0.15 * 40000 + 0.20 * (I - 40000)

theorem find_income :
  ∃ I : ℝ, citizen_income I = 8000 ∧ I = 50000 :=
by
  sorry

end find_income_l26_26531


namespace determine_z_l26_26695

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26695


namespace inverse_of_B_squared_l26_26771

noncomputable def B_inv : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]

theorem inverse_of_B_squared :
  (B_inv * B_inv) = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end inverse_of_B_squared_l26_26771


namespace smallest_integer_l26_26966

theorem smallest_integer (p q r s : ℕ) (X : ℕ) : 
  p < q -> q < r -> r < s -> 
  let pqrs := 1000 * p + 100 * q + 10 * r + s in
  let srqp := 1000 * s + 100 * r + 10 * q + p in
  pqrs + srqp + X = 26352 -> 
  multiset.mem (6789 : ℕ) [pqrs, srqp, X].to_multiset ->
  min pqrs (min srqp X) = 6789 :=
sorry

end smallest_integer_l26_26966


namespace solve_fraction_sum_l26_26043

noncomputable theory

def roots_of_polynomial : Prop :=
  let a b c := classical.some (roots_of_polynomial_eq (x^3 - 15 * x^2 + 22 * x - 8 = 0)) in
  (a + b + c = 15) ∧ (ab + ac + bc = 22) ∧ (abc = 8)

theorem solve_fraction_sum (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 22) (h₃ : abc = 8) :
  (\frac{a}{\frac{1}{a}+bc} + \frac{b}{\frac{1}{b}+ca} + \frac{c}{\frac{1}{c}+ab}) = \frac{181}{9} :=
  by
    sorry

end solve_fraction_sum_l26_26043


namespace element_occurs_twice_l26_26401

theorem element_occurs_twice (n : ℕ) (hn : n ≥ 3) (P : Finset (Finset (Fin n)))
  (hP : P.card = n)
  (hdisjoint : ∀ i j ∈ Fin n, i ≠ j → 
    (∃ k ∈ P, k = {i, j} ∨ (¬ k ∩ {i, j} = ∅))) :
  ∀ i ∈ (Fin n), (∃ a b ∈ P, a ≠ b ∧ (i ∈ a ∧ i ∈ b) ∧ ∀ c ∈ P, c = a ∨ c = b → i ∈ c) :=
sorry

end element_occurs_twice_l26_26401


namespace no_sqrt_negative_number_l26_26159

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end no_sqrt_negative_number_l26_26159


namespace least_12_heavy_three_digit_l26_26571

def is_12_heavy (n : ℕ) : Prop :=
  n % 12 > 7

theorem least_12_heavy_three_digit : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ is_12_heavy n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ is_12_heavy m → n ≤ m := 
begin
  use 105,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { unfold is_12_heavy,
    norm_num },
  { intros m hm h_heavy,
    have h105: 105 % 12 = 9 := rfl,
    cases hm,
    cases hm_right,
    unfold is_12_heavy at h_heavy,
    have h_mod: m % 12 = 8 ∨ m % 12 = 9 ∨ m % 12 = 10 ∨ m % 12 = 11 := by linarith,
    cases h_mod;
    linarith }
end

end least_12_heavy_three_digit_l26_26571


namespace number_of_four_digit_numbers_l26_26495

theorem number_of_four_digit_numbers : 
    let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8}
    let cards := [ (0, 8), (1, 7), (2, 5), (3, 4), (6, 9)]
    (∀ n ∈ digits,  
        ∃ first second third fourth : ℕ, -- four-digit number
        first ≠ 0 ∧ -- first digit cannot be 0
        (first ∈ cards ∨ first = 6 ∨ first = 9) ∧
        (second ∈ cards ∨ second = 6 ∨ second = 9) ∧
        (third ∈ cards ∨ third = 6 ∨ third = 9) ∧
        (fourth ∈ cards ∨ fourth = 6 ∨ fourth = 9) ∧
        first ≠ second ∧ first ≠ third ∧ first ≠ fourth ∧
        second ≠ third ∧ second ≠ fourth ∧ third ≠ fourth) 
    → 9 * 8 * 6 * 4 = 1728 := 
by { sorry }

end number_of_four_digit_numbers_l26_26495


namespace probability_coprime_two_integers_l26_26175

theorem probability_coprime_two_integers : 
  (P : ℝ) (h : P = real_pi) → P = 6 / real_pi^2 :=
by sorry

end probability_coprime_two_integers_l26_26175


namespace tan_alpha_add_pi_over_4_l26_26634

noncomputable
def tan_sum : ℝ → ℝ :=
  λ x, Math.tan x

theorem tan_alpha_add_pi_over_4
  (alpha beta : ℝ)
  (h1 : tan_sum (alpha + beta) = 2 / 5)
  (h2 : tan_sum (beta - (real.pi / 4)) = 1 / 4) :
  tan_sum (alpha + (real.pi / 4)) = 3 / 22 := sorry

end tan_alpha_add_pi_over_4_l26_26634


namespace tetrahedron_distance_height_ratio_l26_26446

theorem tetrahedron_distance_height_ratio 
  (A B C D M : Point)
  (x1 x2 x3 x4 : ℝ) 
  (h1 h2 h3 h4 : ℝ) 
  (V : ℝ)
  (H1 : distance M (triangle A B C) = x1)
  (H2 : distance M (triangle A B D) = x2)
  (H3 : distance M (triangle A C D) = x3)
  (H4 : distance M (triangle B C D) = x4)
  (Hh1 : height D (triangle A B C) = h1)
  (Hh2 : height C (triangle A B D) = h2)
  (Hh3 : height B (triangle A C D) = h3)
  (Hh4 : height A (triangle B C D) = h4) :
  (x1 / h1) + (x2 / h2) + (x3 / h3) + (x4 / h4) = 1 := 
sorry

end tetrahedron_distance_height_ratio_l26_26446


namespace mike_total_working_hours_l26_26381

def time_washing_sedan := 10
def time_oil_change := 15
def time_tire_change := 30
def time_painting_sedan := 45
def time_engine_service := 60
def time_washing_suv := 1.5 * time_washing_sedan
def time_painting_suv := 1.5 * time_painting_sedan

def sedan_tasks := 
  9 * time_washing_sedan + 
  6 * time_oil_change + 
  2 * time_tire_change + 
  4 * time_painting_sedan + 
  2 * time_engine_service

def suv_tasks := 
  7 * time_washing_suv + 
  4 * time_oil_change + 
  3 * time_tire_change + 
  3 * time_painting_suv + 
  1 * time_engine_service

def total_minutes := sedan_tasks + suv_tasks

def total_hours := total_minutes / 60

theorem mike_total_working_hours : total_hours = 17.625 := by
  sorry

end mike_total_working_hours_l26_26381


namespace wilfred_carrots_total_l26_26165

-- Define the number of carrots Wilfred eats each day
def tuesday_carrots := 4
def wednesday_carrots := 6
def thursday_carrots := 5

-- Define the total number of carrots eaten from Tuesday to Thursday
def total_carrots := tuesday_carrots + wednesday_carrots + thursday_carrots

-- The theorem to prove that the total number of carrots is 15
theorem wilfred_carrots_total : total_carrots = 15 := by
  sorry

end wilfred_carrots_total_l26_26165


namespace manganese_percentage_mixture_l26_26348

theorem manganese_percentage_mixture (initial_weight total_weight: ℕ) (initial_percentage final_percentage: ℕ) : 
    initial_weight = 1 → 
    initial_percentage = 20 → 
    total_weight = initial_weight + 1 → 
    final_percentage = (initial_percentage * initial_weight) / total_weight → 
    final_percentage = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end manganese_percentage_mixture_l26_26348


namespace simplest_square_root_l26_26163

theorem simplest_square_root :
  let A := real.sqrt 32
  let B := -real.sqrt (1 / 3)
  let C := real.sqrt 4
  let D := real.sqrt 2
  D = real.sqrt 2 :=
begin
  sorry
end

end simplest_square_root_l26_26163


namespace parabola_intersections_sum_l26_26128

theorem parabola_intersections_sum :
  let parabola1 := λ x : ℝ, (x - 2) ^ 2
  let parabola2 := λ y : ℝ, (y + 1) ^ 2 - 6
  let x_roots := {x | ∃ y, parabola1 x = y ∧ parabola2 y = x}
  let y_roots := {y | ∃ x, parabola1 x = y ∧ parabola2 y = x} in
  ∑ (x ∈ x_roots), x + ∑ (y ∈ y_roots), y = 4 :=
sorry

end parabola_intersections_sum_l26_26128


namespace intersection_point_l26_26869

theorem intersection_point : ∃ (x y : ℝ), y = 3 - x ∧ y = 3 * x - 5 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l26_26869


namespace f_monotonically_decreasing_solve_inequality_l26_26050

-- Given function definition and properties
def f (x : ℝ) : ℝ := sorry
axiom f_defined : ∀ x ≠ 0, f x ≠ ⊥
axiom f_property : ∀ (x y : ℝ), f(x * y) = f(x) + f(y) - 3
axiom f_value_at_2 : f 2 = 1
axiom f_greater_than_3 : ∀ x : ℝ, 0 < x ∧ x < 1 → f(x) > 3

-- Part (1): Prove monotonicity
theorem f_monotonically_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f x1 < f x2 :=
sorry

-- Part (2): Solve the inequality
theorem solve_inequality (x : ℝ) :
  f(3 * x - 5) > -5 ↔ -11 / 3 < x ∧ x < 7 ∧ x ≠ 5 / 3 :=
sorry

end f_monotonically_decreasing_solve_inequality_l26_26050


namespace complex_number_solution_l26_26685

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26685


namespace result_of_fractions_mult_l26_26516

theorem result_of_fractions_mult (a b c d : ℚ) (x : ℕ) :
  a = 3 / 4 →
  b = 1 / 2 →
  c = 2 / 5 →
  d = 5100 →
  a * b * c * d = 765 := by
  sorry

end result_of_fractions_mult_l26_26516


namespace findMonicPolynomials_l26_26606

-- Define what it means to be a monic polynomial
def isMonic (P : Polynomial ℝ) : Prop :=
  P.leadingCoeff = 1

-- Define the main theorem
theorem findMonicPolynomials (P : Polynomial ℝ) :
  isMonic P ∧ (P^2 - 1) % (P.eval (X + 1)) = 0 → 
  (P = 1 ∨ ∃ b : ℤ, P = (X + C b)) :=
by
  sorry

end findMonicPolynomials_l26_26606


namespace find_high_school_students_l26_26509

noncomputable def high_school_students (n : ℕ) (p : ℕ → ℝ) : Prop :=
  p(n-2) = 8 ∧ 
  (∀ i j : ℕ, i < j ∧ j ≤ n → p(i) = p(j)) ∧
  (∀ i : ℕ, i < n → p(i) ∈ {0, 0.5, 1})

theorem find_high_school_students :
  ∃ n : ℕ, (n = 7 ∨ n = 14) ∧
    high_school_students n
  :=
sorry

end find_high_school_students_l26_26509


namespace A_in_correct_interval_l26_26593

noncomputable def recursive_log_fun (n : Nat) : Real :=
  if n = 2 then Real.log 2 else Real.log (n + recursive_log_fun (n - 1))

theorem A_in_correct_interval :
  let A := recursive_log_fun 2013
  A > Real.log 2016 ∧ A < Real.log 2017 :=
by
  sorry

end A_in_correct_interval_l26_26593


namespace digit_in_repeating_decimal_of_7_over_19_l26_26934

theorem digit_in_repeating_decimal_of_7_over_19 :
  let repeating_sequence := "368421052631578947" in
  (nat.mod 921 repeating_sequence.length = 9) →
  (repeating_sequence.get 8 = '2') :=
by
  intros repeating_sequence mod_result
  let digit_9 := repeating_sequence.get 8
  have : digit_9 = '2', from rfl
  exact this

end digit_in_repeating_decimal_of_7_over_19_l26_26934


namespace regular_polygon_sides_160_l26_26203

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end regular_polygon_sides_160_l26_26203


namespace Robert_can_read_one_book_l26_26449

def reading_speed : ℕ := 100 -- pages per hour
def book_length : ℕ := 350 -- pages
def available_time : ℕ := 5 -- hours

theorem Robert_can_read_one_book :
  (available_time * reading_speed) >= book_length ∧ 
  (available_time * reading_speed) < 2 * book_length :=
by {
  -- The proof steps are omitted as instructed.
  sorry
}

end Robert_can_read_one_book_l26_26449


namespace sum_f_values_l26_26310

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- State the requirement to prove the given sum equals 2014.5
theorem sum_f_values :
  ∑ k in (Finset.range 2015).map (Nat.cast), (f k + f (1/k)) = 2014.5 :=
sorry

end sum_f_values_l26_26310


namespace LCM_of_fractions_l26_26514

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l26_26514


namespace solve_fraction_sum_l26_26044

noncomputable theory

def roots_of_polynomial : Prop :=
  let a b c := classical.some (roots_of_polynomial_eq (x^3 - 15 * x^2 + 22 * x - 8 = 0)) in
  (a + b + c = 15) ∧ (ab + ac + bc = 22) ∧ (abc = 8)

theorem solve_fraction_sum (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 22) (h₃ : abc = 8) :
  (\frac{a}{\frac{1}{a}+bc} + \frac{b}{\frac{1}{b}+ca} + \frac{c}{\frac{1}{c}+ab}) = \frac{181}{9} :=
  by
    sorry

end solve_fraction_sum_l26_26044


namespace air_conditioning_unit_price_november_l26_26220

noncomputable def final_price_in_november : ℝ :=
  let original_price := 470
  let christmas_discount := 0.16
  let energy_efficient_discount := 0.07
  let six_months_increase := 0.12
  let production_cost_increase := 0.08
  let november_discount := 0.10
  let price_after_christmas := original_price * (1 - christmas_discount)
  let price_after_energy_efficient := price_after_christmas * (1 - energy_efficient_discount)
  let price_after_six_months := price_after_energy_efficient * (1 + six_months_increase)
  let price_after_production_cost := price_after_six_months * (1 + production_cost_increase)
  let final_price := price_after_production_cost * (1 - november_discount) in
  final_price

theorem air_conditioning_unit_price_november
  (original_price : ℝ)
  (christmas_discount : ℝ)
  (energy_efficient_discount : ℝ)
  (six_months_increase : ℝ)
  (production_cost_increase : ℝ)
  (november_discount : ℝ)
  (final_price := original_price * (1 - christmas_discount) * (1 - energy_efficient_discount) * (1 + six_months_increase) * (1 + production_cost_increase) * (1 - november_discount)) :
  final_price = 399.71 :=
by
  have orig_price := 470
  have xmas_disc := 0.16
  have energy_disc := 0.07
  have six_months_incr := 0.12
  have production_incr := 0.08
  have nov_disc := 0.10
  have price_after_xmas := orig_price * (1 - xmas_disc)
  have price_after_energy := price_after_xmas * (1 - energy_disc)
  have price_after_six_months := price_after_energy * (1 + six_months_incr)
  have price_after_production := price_after_six_months * (1 + production_incr)
  have result := price_after_production * (1 - nov_disc)
  exact rfl

end air_conditioning_unit_price_november_l26_26220


namespace triangle_inequality_l26_26084

theorem triangle_inequality 
  (a b c : ℝ) -- lengths of the sides of the triangle
  (α β γ : ℝ) -- angles of the triangle in radians opposite to sides a, b, c
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- positivity of sides
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) -- positivity and range of angles
  (h_sum : α + β + γ = π) -- angle sum property of a triangle
: 
  b / Real.sin (γ + α / 3) + c / Real.sin (β + α / 3) > (2 / 3) * (a / Real.sin (α / 3)) :=
sorry

end triangle_inequality_l26_26084


namespace design_orderings_count_l26_26792

theorem design_orderings_count :
  let designs := (1 to 12)
  let completed := {10, 11}
  let remaining := designs \ completed
  (∑ k in finset.range 10, ((finset.card (finset.powerset_len k remaining)) * (k + 2))) = 1554 :=
sorry

end design_orderings_count_l26_26792


namespace solution_problem_l26_26807

open Real

noncomputable def parametric_curve_x (t : ℝ) : ℝ := 2 * cos t
noncomputable def parametric_curve_y (t : ℝ) : ℝ := sin t

def polar_line_equation (ρ θ : ℝ) : Prop :=
  ρ * cos (θ + π / 3) = -√3 / 2

def cartesian_curve_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def cartesian_line_equation (x y : ℝ) : Prop :=
  x - √3 * y + √3 = 0

def length_of_segment_AB (A B : ℝ × ℝ) : ℝ :=
  sqrt (1 + (√3 / 3)^2) * (abs ((fst A + fst B) - 4 * 0)) -- A == (x1, y1) and B == (x2, y2)

theorem solution_problem 
  (t : ℝ)
  (ρ θ : ℝ) 
  (A B : ℝ × ℝ) :
  cartesian_curve_equation (parametric_curve_x t) (parametric_curve_y t) ∧
  polar_line_equation ρ θ ∧
  cartesian_line_equation (fst A) (snd A) ∧
  cartesian_line_equation (fst B) (snd B)
  → length_of_segment_AB A B = 32 / 7 :=
by sorry

end solution_problem_l26_26807


namespace books_total_l26_26777

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end books_total_l26_26777


namespace transport_cost_l26_26108

theorem transport_cost (mass_g: ℕ) (cost_per_kg : ℕ) (mass_kg : ℝ) 
  (h1 : mass_g = 300) (h2 : mass_kg = (mass_g : ℝ) / 1000) 
  (h3: cost_per_kg = 18000)
  : mass_kg * cost_per_kg = 5400 := by
  sorry

end transport_cost_l26_26108


namespace increasing_function_among_given_l26_26217

-- Definitions for the functions
noncomputable def f1 (x : ℝ) := Real.exp (-x)
noncomputable def f2 (x : ℝ) := x^3
noncomputable def f3 (x : ℝ) := Real.log x
noncomputable def f4 (x : ℝ) := abs x

-- Statement of the theorem
theorem increasing_function_among_given :
  (∀ x y : ℝ, x < y → f2 x < f2 y) ∧
  (∀ x y : ℝ, x < y → f1 x ≥ f1 y) ∧
  (∄ x : ℝ, f3 x ∧ (¬ (domain ℝ))) ∧
  [Other conditions and comparisons that were deduced in the solution/steps] :=
by
  sorry

end increasing_function_among_given_l26_26217


namespace sum_of_consecutive_ints_product_eq_336_l26_26268

def consecutive_ints_sum (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

theorem sum_of_consecutive_ints_product_eq_336 (a b c : ℤ) (h1 : consecutive_ints_sum a b c) (h2 : a * b * c = 336) :
  a + b + c = 21 :=
sorry

end sum_of_consecutive_ints_product_eq_336_l26_26268


namespace solve_for_z_l26_26740

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26740


namespace max_ratio_of_distances_l26_26445

noncomputable theory

open Real

def is_on_circle (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 + y^2 = 100

def integer_coordinates (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in ∃ (m n : ℤ), (x = m) ∧ (y = n)

variables (A B C D : ℝ × ℝ)

def distance (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P in
  let (x2, y2) := Q in
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def irrational_distance (P Q : ℝ × ℝ) : Prop :=
  let d := distance P Q in ¬ (∃ (n : ℤ), d = n)

theorem max_ratio_of_distances (hA_on_circle : is_on_circle A)
                               (hA_int_coords : integer_coordinates A)
                               (hB_on_circle : is_on_circle B)
                               (hB_int_coords : integer_coordinates B)
                               (hC_on_circle : is_on_circle C)
                               (hC_int_coords : integer_coordinates C)
                               (hD_on_circle : is_on_circle D)
                               (hD_int_coords : integer_coordinates D)
                               (hAB_irrational : irrational_distance A B)
                               (hCD_irrational : irrational_distance C D) :
  ∃ (k : ℝ), k = 7 ∧ (∀ (AB_ratio : ℝ), AB_ratio = (distance A B) / (distance C D) → AB_ratio ≤ k) :=
sorry

end max_ratio_of_distances_l26_26445


namespace find_fourth_vertex_l26_26205

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2

noncomputable def is_valid_vertex (x y z : ℤ) : Prop :=
  distance (1, 1, 1) (x, y, z) = 17 ∧
  distance (5, 1, 2) (x, y, z) = 17 ∧
  distance (4, 3, 4) (x, y, z) = 17

theorem find_fourth_vertex : ∃ x y z : ℤ, is_valid_vertex x y z ∧ (x, y, z) = (3, 0, 3) := 
sorry

end find_fourth_vertex_l26_26205


namespace solve_system_eq_l26_26854

theorem solve_system_eq (x y : ℝ) :
  x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0 ↔
  x = 4 ∧ y = 1 :=
sorry

end solve_system_eq_l26_26854


namespace congruent_semicircles_ratio_l26_26005

theorem congruent_semicircles_ratio (N : ℕ) (r : ℝ) (hN : N > 0) 
    (A : ℝ) (B : ℝ) (hA : A = (N * π * r^2) / 2)
    (hB : B = (π * N^2 * r^2) / 2 - (N * π * r^2) / 2)
    (h_ratio : A / B = 1 / 9) : 
    N = 10 :=
by
  -- The proof will be filled in here.
  sorry

end congruent_semicircles_ratio_l26_26005


namespace complex_number_solution_l26_26686

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26686


namespace evaluate_expression_l26_26047

variable {R : Type*} [LinearOrderedField R]

def roots_of_cubic (p q r : R) (a b c : R) :=
  a + b + c = p ∧ a * b + b * c + c * a = q ∧ a * b * c = r

theorem evaluate_expression (a b c : R) 
  (h : roots_of_cubic 15 22 8 a b c) : 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 181 / 9) :=
by
  cases h with h_sum h_product;
  cases h_product with h_ab_bc_ca h_abc;
  sorry

end evaluate_expression_l26_26047


namespace polygon_sides_l26_26202

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end polygon_sides_l26_26202


namespace zoe_total_money_l26_26525

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end zoe_total_money_l26_26525


namespace probability_prime_and_even_card_l26_26453

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define what it means for a number to be even
def is_even (n : ℕ) : Prop := 
  n % 2 = 0

-- Define the list of cards
def cards := (list.range 70).map (+1)

-- Define the probability of an event
def probability (p : ℕ → Prop) : ℚ :=
  (cards.count p).toRat / cards.length.toRat

-- Proof statement
theorem probability_prime_and_even_card : 
  probability (λ n, is_prime n ∧ is_even n) = 1 / 70 :=
by
  sorry

end probability_prime_and_even_card_l26_26453


namespace correct_statements_l26_26993

-- Definitions of each statement as conditions

def statement1 : Prop := ∀ {l₁ l₂ : ℝ}, l₁ ≠ l₂ → ∃ A B, A ≠ B ∧ l₁ = l₂
def statement2 : Prop := ∀ {l₁ l₂ l₃ : ℝ}, l₁ = l₃ ∧ l₂ = l₃ → l₁ = l₂
def statement3 : Prop := ∀ {l₁ l₂ l₃ : ℝ}, l₁ ⊥ l₃ ∧ l₂ ⊥ l₃ → l₁ ⊥ l₂
def statement4 : Prop := ∀ {A B : ℝ}, length (line_segment A B) = distance A B
def statement5 : Prop := ∀ {l₁ l₂ : ℝ}, ∃ t, (angle l₁ t + angle l₂ t) = 180 ∧ l₁ = l₂ ∨ l₁ ⊥ l₂

-- Final proof problem

theorem correct_statements : ¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4 ∧ statement5 :=
by
    sorry

end correct_statements_l26_26993


namespace find_f_prime_one_l26_26666

theorem find_f_prime_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f' x = 2 * f' 1 + 1 / x) (h_fx : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by 
  sorry

end find_f_prime_one_l26_26666


namespace conic_section_propositions_l26_26575

noncomputable def correct_propositions : List ℕ := [2, 3]

theorem conic_section_propositions :
  (∀ (A B : Point) (k : ℝ), k > dist A B → k ≠ 0 → 
    trajectory (λ P, dist P A + dist P B = k) ≠ ellipse ∧
  (∀ (A : Point) (C : Circle), 
    trajectory (λ P, midpoint A B = P) = C ∧
  (∀ (a b : ℝ), 
    solve (λ x, ln x ^ 2 - ln x - 2) = ([exp 2, 1 / exp]) → 
      ln x > 1 → ln x < 1 → ellipse_eccentricity 1 < hyperbola_eccentricity 1 ∧
  (∃ (y x : ℝ), 
    foci (hyperbola (y^2 / 9 - x^2 / 25) = 1 ↔ 
    foci (ellipse (x^2 / 35 + y^2) = 1) → false))) := 
by sorry

end conic_section_propositions_l26_26575


namespace find_sum_of_squares_l26_26138

theorem find_sum_of_squares 
  (b_1 b_2 b_3 b_4 : ℝ)
  (h : ∀ θ : ℝ, sin θ ^ 4 = b_1 * sin θ + b_2 * sin (2 * θ) + b_3 * sin (3 * θ) + b_4 * sin (4 * θ)) :
  b_1^2 + b_2^2 + b_3^2 + b_4^2 = 17 / 64 :=
  sorry

end find_sum_of_squares_l26_26138


namespace derivative_at_a_l26_26276

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * (f' 1)
noncomputable def f' (x : ℝ) : ℝ := (derivative f) x

theorem derivative_at_a 
  (a : ℝ) :
  f' a = 2 * a - 4 :=
sorry

end derivative_at_a_l26_26276


namespace total_fish_count_l26_26813

theorem total_fish_count (jason_ryan_ratio ryan_jeffery_ratio jeffery_fish : ℕ)
  (h1 : ryan_jeffery_ratio = 2)
  (h2 : jeffery_fish = 60)
  (h3 : jason_ryan_ratio = 3) :
  let ryan_fish := jeffery_fish / ryan_jeffery_ratio
  let jason_fish := ryan_fish / jason_ryan_ratio
  in jason_fish + ryan_fish + jeffery_fish = 100 := by
  sorry

end total_fish_count_l26_26813


namespace polar_to_rectangular_equation_l26_26485

/-- Proof that the rectangular coordinate equation of a given polar equation is as derived. -/
theorem polar_to_rectangular_equation :
  ∀ (ρ θ x y : ℝ),
  ρ = sin θ - 3 * cos θ →
  ρ * cos θ = x →
  ρ * sin θ = y →
  x^2 - 3 * x + y^2 - y = 0 :=
by
  assume ρ θ x y h1 h2 h3,
  sorry

end polar_to_rectangular_equation_l26_26485


namespace xiaomings_possible_score_l26_26862

def average_score_class_A : ℤ := 87
def average_score_class_B : ℤ := 82

theorem xiaomings_possible_score (x : ℤ) :
  (average_score_class_B < x ∧ x < average_score_class_A) → x = 85 :=
by sorry

end xiaomings_possible_score_l26_26862


namespace total_distance_covered_l26_26622

-- Definitions of the conditions
def walking_speed : ℝ := 5 -- in km/hr
def walking_time : ℝ := 15/60 -- in hr

def running_speed : ℝ := 12 -- in km/hr
def running_time : ℝ := 10/60 -- in hr

def cycling_speed : ℝ := 25 -- in km/hr
def cycling_time : ℝ := 20/60 -- in hr

-- Definition of the distance function
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem total_distance_covered :
  distance walking_speed walking_time +
  distance running_speed running_time +
  distance cycling_speed cycling_time = 11.58 :=
by
  -- Use 'sorry' to skip the proof
  sorry

end total_distance_covered_l26_26622


namespace line_intersect_circle_l26_26805

theorem line_intersect_circle (b : ℝ) :
  (let C := (2 : ℝ, 0 : ℝ))
   let radius := (2 : ℝ)
   let circle_eq := (λ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4)
   let slope := (135 : ℝ).toRadians
   let line_eq := (λ (x y : ℝ), y = -x + b)
   let abs_chord := (λ A B : ℝ × ℝ, |A - B| = 2 * sqrt 2) 
  circle_eq 2 0 ∧ line_eq 0 b ∧ abs_chord = 2 * real.sqrt 2 → b = 0 ∨ b = 4 :=
sorry

end line_intersect_circle_l26_26805


namespace lower_rent_amount_l26_26842

-- Define the conditions and proof goal
variable (T R : ℕ)
variable (L : ℕ)

-- Condition 1: Total rent is $1000
def total_rent (T R : ℕ) (L : ℕ) := 60 * R + L * (T - R)

-- Condition 2: Reduction by 20% when 10 rooms are swapped
def reduced_rent (T R : ℕ) (L : ℕ) := 60 * (R - 10) + L * (T - R + 10)

-- Proof that the lower rent amount is $40 given the conditions
theorem lower_rent_amount (h1 : total_rent T R L = 1000)
                         (h2 : reduced_rent T R L = 800) : L = 40 :=
by
  sorry

end lower_rent_amount_l26_26842


namespace approximate_construction_is_accurate_l26_26349

noncomputable def circle_radius := 1
noncomputable def center (O : Point) := true
noncomputable def diameter (A B : Point) := distance A B = 2 * circle_radius
noncomputable def midpoint (C A B : Point) := distance C A = distance C B ∧ distance C center = circle_radius
noncomputable def trisection (D E : Point) := ∃ (θ : ℝ), θ ∈ set.Icc 0 (2 * π) ∧ (distance D center = circle_radius ∧ distance E center = circle_radius)
noncomputable def sum_of_sides_and_base (triangle : Triangle) : ℝ := 
  distance triangle.a triangle.b + distance triangle.b triangle.c + distance triangle.a triangle.c

theorem approximate_construction_is_accurate
  (O A B C D E : Point)
  (h_center : center O)
  (h_diameter : diameter A B)
  (h_midpoint : midpoint C A B)
  (h_trisection : trisection D E) :
  abs (sum_of_sides_and_base ⟨C, D, E⟩ - (π / 2)) < 0.0004 := 
sorry

end approximate_construction_is_accurate_l26_26349


namespace determine_z_l26_26703

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26703


namespace area_of_triangle_l26_26986

theorem area_of_triangle : 
  let line1 := fun x : ℝ => x + 3
  let line2 := fun x : ℝ => -3 * x + 9
  let line3 := (2 : ℝ)
  let vertex1 := (-1, 2)
  let vertex2 := (7 / 3, 2)
  let vertex3 := (3 / 2, 9 / 2)
  let base := (7 / 3 + 1)
  let height := (9 / 2 - 2)
  let area := (1 / 2) * base * height
  in area ≈ 4.17 :=
by
  sorry

end area_of_triangle_l26_26986


namespace determine_z_l26_26698

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26698


namespace determine_a_l26_26419

theorem determine_a (a : ℤ) (q : ℤ[X]) :
  (X^2 - X + C a) * q = X^13 + X + 90 → a = 2 :=
by sorry

end determine_a_l26_26419


namespace josh_lost_eight_marbles_l26_26393

variable (L : ℕ)
variable (initial_marbles : ℕ := 7)
variable (found_marbles : ℕ := 10)
variable (extra_marbles : ℕ := 2)

theorem josh_lost_eight_marbles (h : found_marbles = L + extra_marbles) : L = 8 :=
by {
  calc 
  L = found_marbles - extra_marbles : by { rw [h], exact rfl }
  ... = 10 - 2 : rfl
  ... = 8 : rfl
}

end josh_lost_eight_marbles_l26_26393


namespace problem_solution_l26_26645

noncomputable def a_n (n : ℕ) : ℕ :=
  if ∃ k, n = 2 * k - 1 then 2 * n
  else if ∃ k, n = 2 * k then 2 ^ (n - 1)
  else 0  -- This case should theoretically not happen due to conditions, but completes the definition.

noncomputable def b_n (n : ℕ) : ℕ :=
  a_n (3 * n)

noncomputable def S_10 : ℕ :=
  (Finset.range 10).sum (λ n, b_n (n+1))

theorem problem_solution : 63 * S_10 - 2^35 = 9418 := 
  sorry

end problem_solution_l26_26645


namespace probability_three_blue_jellybeans_l26_26544

theorem probability_three_blue_jellybeans:
  let total_jellybeans := 20
  let blue_jellybeans := 10
  let red_jellybeans := 10
  let draws := 3
  let q := (1 / 2) * (9 / 19) * (4 / 9)
  q = 2 / 19 :=
sorry

end probability_three_blue_jellybeans_l26_26544


namespace star_product_identity_l26_26027

def star_labeling : Prop := 
∀ A B C D E A₁ B₁ C₁ D₁ E₁ : Point, 
  midpoint (A₁, B₁) ∧ midpoint (B₁, C₁) ∧ midpoint (C₁, D₁) ∧ midpoint (D₁, E₁) ∧ midpoint (E₁, A₁) ∧
  vertex (A, B, C, D, E),

theorem star_product_identity (h : star_labeling) : 
    A₁ C ⋅ B₁ D ⋅ C₁ E ⋅ D₁ A ⋅ E₁ B = A₁ D ⋅ B₁ E ⋅ C₁ A ⋅ D₁ B ⋅ E₁ C := 
sorry

end star_product_identity_l26_26027


namespace solve_for_z_l26_26707

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26707


namespace sin_cosine_decreasing_interval_l26_26885

noncomputable def function_decreasing_interval (a b : ℝ) (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  I = { x | a ≤ x ∧ x ≤ b ∧ ∀ y, x < y ∧ y ∈ I → f y < f x } 

theorem sin_cosine_decreasing_interval :
  function_decreasing_interval (Real.pi / 6) (7 * Real.pi / 6) (λ x, Real.sin x + Real.sqrt 3 * Real.cos x) 
  (λ x, 0 ≤ x ∧ x ≤ 2 * Real.pi) :=
sorry

end sin_cosine_decreasing_interval_l26_26885


namespace range_of_a_l26_26872

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → (a-1)^x > (a-1)^y) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l26_26872


namespace decreasing_function_l26_26216

theorem decreasing_function (f g h j : ℝ → ℝ)
  (hf : ∀ x, f x = 0.5^x)
  (hg : ∀ x, g x = x^3)
  (hh : ∀ x, x > 0 → h x = log (0.5) x)
  (hj : ∀ x, j x = 2^x) :
  ∀ x, ∀ (x1 x2 : ℝ), x1 < x2 → (f x1 > f x2 ∧ ¬ (g x1 > g x2 ∨ j x1 > j x2)) :=
by
  sorry

end decreasing_function_l26_26216


namespace katie_five_dollar_bills_l26_26394

theorem katie_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end katie_five_dollar_bills_l26_26394


namespace even_perfect_square_form_l26_26221

theorem even_perfect_square_form :
  ∃ (a b : ℕ), (a < 10) ∧ (b < 10) ∧ (10010 * a + 1001 * b + 100 = 4 * (138 ^ 2)) ∧ (2 * 138 = 276) ∧ (276 ^ 2 = 10000 * a + 1000 * b + 100 + 10 * a + b) :=
begin
  use [7, 6],
  split,
  { exact nat.lt_succ_self 7 },
  split,
  { exact nat.lt_succ_self 6 },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num },
end

end even_perfect_square_form_l26_26221


namespace root_expression_equals_181_div_9_l26_26042

noncomputable def polynomial_root_sum (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22) 
  (h3 : a*b*c = 8) : ℝ :=
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) 

theorem root_expression_equals_181_div_9
  (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22)
  (h3 : a*b*c = 8) :
  polynomial_root_sum a b c h1 h2 h3 = 181 / 9 := by 
  sorry

end root_expression_equals_181_div_9_l26_26042


namespace complex_conjugate_quadrant_l26_26866

theorem complex_conjugate_quadrant (z : ℂ) (h : z * (1 - I) = |1 + I|) : ∃ q, q = 4 :=
by
  sorry

end complex_conjugate_quadrant_l26_26866


namespace locus_of_constant_distance_l26_26671

variables {R : Type*} [linear_ordered_field R]

structure Line (R : Type*) [linear_ordered_field R] :=
(a b c : R) -- ax + by + c = 0

def distance (p l : Line R) : R :=
(abs (l.a * p.x + l.b * p.y + l.c)) / sqrt (l.a ^ 2 + l.b ^ 2)

structure Parallelogram (R : Type*) [linear_ordered_field R] :=
(l1 l2 m1 m2 : Line R)
(l1_parallel_l2 : l1.a * l2.b = l2.a * l1.b)
(m1_parallel_m2 : m1.a * m2.b = m2.a * m1.b)
(l1_m1_intersection : line_intersection l1 m1 ≠ none)
(l1_m2_intersection : line_intersection l1 m2 ≠ none)
(l2_m1_intersection : line_intersection l2 m1 ≠ none)
(l2_m2_intersection : line_intersection l2 m2 ≠ none)

theorem locus_of_constant_distance
  {p : Point R} {a b c : R} (h1: Parallelogram R)
  (h2: ¬line_intersection h1.l1 h1.m1 = none) 
  (h3: ¬line_intersection h1.l2 h1.m2 = none)
  (hd: distance p h1.l1 + distance p h1.l2 + distance p h1.m1 + distance p h1.m2 = a + b + c):
  locus p := sorry

end locus_of_constant_distance_l26_26671


namespace largest_n_for_triangle_property_l26_26244

-- Define the triangle property for a set
def triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a < b → b < c → a + b > c

-- Define the smallest subset that violates the triangle property
def violating_subset : Set ℕ := {5, 6, 11, 17, 28, 45, 73, 118, 191, 309}

-- Define the set of consecutive integers from 5 to n
def consecutive_integers (n : ℕ) : Set ℕ := {x : ℕ | 5 ≤ x ∧ x ≤ n}

-- The theorem we want to prove
theorem largest_n_for_triangle_property : ∀ (S : Set ℕ), S = consecutive_integers 308 → triangle_property S := sorry

end largest_n_for_triangle_property_l26_26244


namespace sqrt_product_simplifies_l26_26585

-- Definitions from the conditions
def sqrt_15p3 (p : ℝ) : ℝ := Real.sqrt (15 * p^3)
def sqrt_20p (p : ℝ) : ℝ := Real.sqrt (20 * p)
def sqrt_8p5 (p : ℝ) : ℝ := Real.sqrt (8 * p^5)

-- Statement of the proof
theorem sqrt_product_simplifies (p : ℝ) : 
  sqrt_15p3 p * sqrt_20p p * sqrt_8p5 p = 20 * p^4 * Real.sqrt (6 * p) := 
sorry

end sqrt_product_simplifies_l26_26585


namespace max_n_value_l26_26429

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem max_n_value (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n → (2 * (n + 0.5) = a n + a (n + 1))) 
  (h2 : S a 63 = 2020) (h3 : a 2 < 3) : 63 ∈ { n : ℕ | S a n = 2020 } :=
sorry

end max_n_value_l26_26429


namespace sum_of_four_digit_numbers_l26_26939

theorem sum_of_four_digit_numbers : 
  let digits := {1, 2, 3}
  let all_numbers := { n : ℕ | n / 1000 ∈ digits ∧ (n / 100) % 10 ∈ digits ∧ (n / 10) % 10 ∈ digits ∧ n % 10 ∈ digits }
  (∀ n ∈ all_numbers, n < 10000 ∧ n ≥ 1000) 
  → (all_numbers.sum id = 179982) := 
by 
  let digits := {1, 2, 3}
  let all_numbers := { n : ℕ | n / 1000 ∈ digits ∧ (n / 100) % 10 ∈ digits ∧ (n / 10) % 10 ∈ digits ∧ n % 10 ∈ digits }
  assume h : ∀ n ∈ all_numbers, 1000 ≤ n ∧ n < 10000
  sorry

end sum_of_four_digit_numbers_l26_26939


namespace polygon_sides_eq_eight_l26_26786

theorem polygon_sides_eq_eight (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
sorry

end polygon_sides_eq_eight_l26_26786


namespace winning_percentage_is_62_l26_26894

-- Definitions based on given conditions
def candidate_winner_votes : ℕ := 992
def candidate_win_margin : ℕ := 384
def total_votes : ℕ := candidate_winner_votes + (candidate_winner_votes - candidate_win_margin)

-- The key proof statement
theorem winning_percentage_is_62 :
  ((candidate_winner_votes : ℚ) / total_votes) * 100 = 62 := 
sorry

end winning_percentage_is_62_l26_26894


namespace find_lambda_neg_three_l26_26416

variables (A B C D : Type)
variables [AddGroup (A)]
variables [VectorSpace ℝ A]
variables [AddGroup (B)]
variables [VectorSpace ℝ B]
variables [AddGroup (C)]
variables [VectorSpace ℝ C]
variables [AddGroup (D)]
variables [VectorSpace ℝ D]

noncomputable def vector_proof_problem : Prop :=
  ∃ (λ : ℝ),
  (λ ∈ ℝ) ∧
  (D ∈ plane_of A B C) ∧
  (AD = - (1/3 : ℝ) • AB + (4 / 3 : ℝ) • AC) ∧
  (BC = λ • DC) ∧
  (λ = -3)

theorem find_lambda_neg_three : vector_proof_problem :=
sorry

end find_lambda_neg_three_l26_26416


namespace investment_A_l26_26176

-- Define constants B and C's investment values, C's share, and total profit.
def B_investment : ℕ := 8000
def C_investment : ℕ := 9000
def C_share : ℕ := 36000
def total_profit : ℕ := 88000

-- Problem statement to prove
theorem investment_A (A_investment : ℕ) : 
  (A_investment + B_investment + C_investment = 17000) → 
  (C_investment * total_profit = C_share * (A_investment + B_investment + C_investment)) →
  A_investment = 5000 :=
by 
  intros h1 h2
  sorry

end investment_A_l26_26176


namespace polynomial_divisible_by_x_minus_4_l26_26600

theorem polynomial_divisible_by_x_minus_4 (m : ℤ) :
  (∀ x, 6 * x ^ 3 - 12 * x ^ 2 + m * x - 24 = 0 → x = 4) ↔ m = -42 :=
by
  sorry

end polynomial_divisible_by_x_minus_4_l26_26600


namespace number_of_people_in_room_l26_26007

theorem number_of_people_in_room (P : ℕ) 
  (h1 : 1/4 * P = P / 4) 
  (h2 : 3/4 * P = 3 * P / 4) 
  (h3 : P / 4 = 20) : 
  P = 80 :=
sorry

end number_of_people_in_room_l26_26007


namespace fx_periodic_odd_l26_26873

theorem fx_periodic_odd (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_symm : ∀ x, f (1 - x) = f (1 + x))
  (h_def : ∀ x ∈ Ico 0 1, f x = 2^x - 1) :
  f (Real.logb (1 / 2) 6) = -1 / 2 :=
by
  sorry

end fx_periodic_odd_l26_26873


namespace total_fireworks_l26_26023

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end total_fireworks_l26_26023


namespace max_mono_inc_interval_l26_26115

theorem max_mono_inc_interval :
  ∃ m : ℝ, (∀ x1 x2 : ℝ, -m ≤ x1 ∧ x1 < x2 ∧ x2 ≤ m → sin (2 * x1 + π / 4) < sin (2 * x2 + π / 4)) ∧ 
  (∀ m' : ℝ, (∀ x1 x2 : ℝ, -m' ≤ x1 ∧ x1 < x2 ∧ x2 ≤ m' → sin (2 * x1 + π / 4) < sin (2 * x2 + π / 4)) → m' ≤ m) :=
sorry

end max_mono_inc_interval_l26_26115


namespace probability_B_l26_26798

variable (Ω : Type)

-- Define the events A, A', B
noncomputable def P {Ω : Type _} [MeasureSpace Ω] (a : event ℙ) := ℙ.MeasureSpace.measure a

variables (A B : event ℙ) (P : MeasureSpace ℙ)
variable hA : P A = 0.5
variable hB_given_A : P (B|A) = 0.9
variable A_compl : event ℙ := -A
variable hA_compl : P A_compl = 0.5
variable hB_given_A_compl : P (B|A_compl) = 0.05

theorem probability_B : P B = 0.475 :=
by
  sorry

end probability_B_l26_26798


namespace sum_of_first_8_terms_l26_26865

theorem sum_of_first_8_terms (a : ℝ) (h : 15 * a = 1) : 
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a + 128 * a) = 17 :=
by
  sorry

end sum_of_first_8_terms_l26_26865


namespace determine_z_l26_26693

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26693


namespace log_increasing_intervals_l26_26249

def increasing_intervals (f : ℝ → ℝ) (domain: Set ℝ) (intervals: Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ intervals → y ∈ intervals → x < y → f x < f y

theorem log_increasing_intervals :
  increasing_intervals (λ x, log (x^2 - 1)) ({x : ℝ | x > 1 ∨ x < -1}) ({x : ℝ | x > 1 ∨ x < -1}) :=
by
  sorry

end log_increasing_intervals_l26_26249


namespace gold_coins_count_l26_26094

theorem gold_coins_count (n c : ℕ) (h1 : n = 8 * (c - 3))
                                     (h2 : n = 5 * c + 4)
                                     (h3 : c ≥ 10) : n = 54 :=
by
  sorry

end gold_coins_count_l26_26094


namespace complex_number_solution_l26_26689

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26689


namespace area_triangle_ABC_l26_26345

-- Define that BD and CE are medians of triangle ABC
def isMedian (triangle : Type) (A B C : triangle) (median : triangle → triangle → triangle) (D : triangle) :=
  ∃ E : triangle, median B D E ∧ median D A E

-- Define that BD is perpendicular to CE
def isPerpendicular (triangle : Type) (B D C E : triangle → Prop) :=
  ∀ (t : triangle), B t ∧ D t → ¬ E t

-- Define the given triangle condition and lengths
structure TriangleABC where
  A B C D E : Type
  BD_perp_CE : isPerpendicular TriangleABC (λ t, t = B) (λ t, t = D) (λ t, t = C) (λ t, t = E)
  BD_length : ∀ {t : Type}, t = B → t = D → t = (4 : nat)
  CE_length : ∀ {t : Type}, t = C → t = E → t = (6 : nat)

-- State the problem and the proof goal
theorem area_triangle_ABC (T : TriangleABC) :
  ∃ area : ℝ, area = 16 :=
  sorry

end area_triangle_ABC_l26_26345


namespace exists_nat_squares_eq_nat_square_l26_26083

theorem exists_nat_squares_eq_nat_square (n : ℕ) : ∃ (x : ℕ → ℕ) (y : ℕ), (∑ i in Finset.range n, (x i) ^ 2) = y ^ 2 :=
by
  sorry

end exists_nat_squares_eq_nat_square_l26_26083


namespace compare_triangle_operations_l26_26247

def tri_op (a b : ℤ) : ℤ := a * b - a - b + 1

theorem compare_triangle_operations : tri_op (-3) 4 = tri_op 4 (-3) :=
by
  unfold tri_op
  sorry

end compare_triangle_operations_l26_26247


namespace probability_task1_l26_26523

noncomputable def P_Task1 (P_Task2 : ℝ) (P_Task1_and_not_Task2 : ℝ) : ℝ :=
P_Task1_and_not_Task2 / (1 - P_Task2)

theorem probability_task1 (h1 : P_Task1 3/5 0.26666666666666666 = 2/3) : true :=
by sorry

end probability_task1_l26_26523


namespace gina_expenditure_l26_26632

noncomputable def gina_total_cost : ℝ :=
  let regular_classes_cost := 12 * 450
  let lab_classes_cost := 6 * 550
  let textbooks_cost := 3 * 150
  let online_resources_cost := 4 * 95
  let facilities_fee := 200
  let lab_fee := 6 * 75
  let total_cost := regular_classes_cost + lab_classes_cost + textbooks_cost + online_resources_cost + facilities_fee + lab_fee
  let scholarship_amount := 0.5 * regular_classes_cost
  let discount_amount := 0.25 * lab_classes_cost
  let adjusted_cost := total_cost - scholarship_amount - discount_amount
  let interest := 0.04 * adjusted_cost
  adjusted_cost + interest

theorem gina_expenditure : gina_total_cost = 5881.20 :=
by
  sorry

end gina_expenditure_l26_26632


namespace fib_det_property_fibonacci_identity_l26_26409

noncomputable def fib_matrix := λ (n: ℕ) => matrix (fin 2) (fin 2) ℤ
noncomputable def fibonacci (n: ℕ) : ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def fib_matrix_pow (n : ℕ) : fib_matrix n :=
  match n with
  | 0     => !![[(1 : ℤ), 0], [0, 1]]
  | (n+1) => !![
                 [fibonacci (n + 2), fibonacci (n +1)],
                 [fibonacci (n + 1), fibonacci n]
               ]

def det_fib_matrix_pow (n : ℕ) : ℤ :=
  (fib_matrix_pow n).det

theorem fib_det_property (n : ℕ) : det_fib_matrix_pow n = (-1) ^ n := sorry

theorem fibonacci_identity : fibonacci 1001 * fibonacci 1003 - fibonacci 1002^2 = 1 := by
  have h1 : det_fib_matrix_pow 1002 = (-1) ^ 1002, from fib_det_property 1002
  have h2 : det_fib_matrix_pow 1002 = fibonacci 1003 * fibonacci 1001 - fibonacci 1002^2, by
    unfold det_fib_matrix_pow fib_matrix_pow
    rw [matrix.det_fin_two, fib_matrix_pow]
    simp 
  rw [h1, h2]
  norm_num
  sorry

end fib_det_property_fibonacci_identity_l26_26409


namespace recurring_decimal_fraction_l26_26911

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l26_26911


namespace complex_number_solution_l26_26688

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26688


namespace bride_older_than_groom_l26_26135

theorem bride_older_than_groom (bride_age groom_age total_age : ℕ) 
  (h_bride_age : bride_age = 102)
  (h_total_age : bride_age + groom_age = 185) : 
  bride_age - groom_age = 19 :=
by
  rw [h_bride_age, h_total_age]
  sorry

end bride_older_than_groom_l26_26135


namespace cost_per_mile_sunshine_is_018_l26_26096

theorem cost_per_mile_sunshine_is_018 :
  ∀ (x : ℝ) (daily_rate_sunshine daily_rate_city cost_per_mile_city : ℝ),
  daily_rate_sunshine = 17.99 →
  daily_rate_city = 18.95 →
  cost_per_mile_city = 0.16 →
  (daily_rate_sunshine + 48 * x = daily_rate_city + cost_per_mile_city * 48) →
  x = 0.18 :=
by
  intros x daily_rate_sunshine daily_rate_city cost_per_mile_city
  intros h1 h2 h3 h4
  sorry

end cost_per_mile_sunshine_is_018_l26_26096


namespace sum_of_rows_and_columns_is_odd_l26_26125

-- Define the problem in Lean 4 terms
def probability_odd_sums_in_3x3_grid : ℚ :=
  1 / 14

-- State the hypothesis
def grid_condition (grid : (Fin 3) × (Fin 3) → Fin 9.succ) :=
  ∀ (i j : Fin 3), grid i j ∈ Finset.range 1 10 ∧ (∃ s, 
  s = ∑ i in Finset.univ, grid i j ∧ s % 2 = 1)

-- Statement of the problem
theorem sum_of_rows_and_columns_is_odd 
  (grid : (Fin 3) × (Fin 3) → Fin 9.succ) (h : grid_condition grid) : 
  (∑ i j, grid i j / grid_condition grid).to_rat = probability_odd_sums_in_3x3_grid :=
by sorry

end sum_of_rows_and_columns_is_odd_l26_26125


namespace tourists_meet_l26_26112

-- Define the speeds of the tourists
def vлет := 16 -- speed of the first tourist
def vмот := 56 -- speed of the second tourist

-- Define the initial conditions
def initial_travel_time_first := 1.5 -- hours
def break_time_first := 1.5 -- hours
def start_time_difference := 4 -- hours

-- Define the total distance traveled by the first tourist 
def distance_first_travel (t : ℝ) : ℝ := 
  let initial_distance := vлет * initial_travel_time_first
  initial_distance + vлет * (t + break_time_first)

-- Define the total distance traveled by the second tourist
def distance_second_travel (t : ℝ) : ℝ := vмот * t

-- Prove that distances are equal when the second tourist catches up to the first tourist
theorem tourists_meet :
  ∃ t : ℝ, distance_second_travel t = distance_first_travel t :=
by
  sorry

end tourists_meet_l26_26112


namespace log_equations_not_equivalent_l26_26577

theorem log_equations_not_equivalent (x : ℝ) : 
  (∃ x₁, x₁^2 - 4 = 4 * x₁ - 7 ∧ (x₁^2 - 4 > 0) ∧ (4 * x₁ - 7 > 0)) → 
  (∃ x₂, x₂^2 - 4 = 4 * x₂ - 7 ∧ ¬(x₂^2 - 4 > 0) ∧ ¬(4 * x₂ - 7 > 0)) →
  ¬ (∀ x, x^2 - 4 = 4 * x - 7 ↔ log (x^2 - 4) = log (4 * x - 7)) :=
sorry

end log_equations_not_equivalent_l26_26577


namespace find_angle_DEB_l26_26364

-- Define the given angles and their relationships
def angle_ABC : ℝ := 60
def angle_ACB : ℝ := 90
def angle_ADC : ℝ := 180
def angle_CDE : ℝ := 48

-- Define the sum of angles in a triangle property
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- The proof problem to determine x
theorem find_angle_DEB (ABC ACB ADC CDE : ℝ)
  (h1 : ABC = 60)
  (h2 : ACB = 90)
  (h3 : ADC = 180)
  (h4 : CDE = 48)
  (sum_of_angles_in_triangle A B C) : 
  ∃ x : ℝ, x = 180 - (180 - (180 - angle_CDE) - (180 - angle_ABC - angle_ACB) - (180 - angle_ABC - angle_ACB)) 
  (x = 162) :=
sorry

end find_angle_DEB_l26_26364


namespace determinant_max_value_l26_26250

noncomputable def det (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_max_value :
  ∃ x : ℝ, 
    det (real.cos (real.pi / 2 + x)) (real.tan x) (real.cos x) (real.cot (real.pi - x)) = 7 :=
begin
  sorry,
end

end determinant_max_value_l26_26250


namespace eighteenth_digit_fraction_l26_26152

theorem eighteenth_digit_fraction :
  let x := 10000 / 9899 in
  (Real.frac x * 10^18).floor % 10 = 5 :=
by
  sorry

end eighteenth_digit_fraction_l26_26152


namespace problem_solution_l26_26097

noncomputable def solve_problem (x : ℝ) : Prop :=
  (sqrt (64 - x ^ 2) - sqrt (36 - x ^ 2) = 4) →
  (sqrt (64 - x ^ 2) + sqrt (36 - x ^ 2) = 7)

theorem problem_solution (x : ℝ) : solve_problem x :=
by
  intro h
  sorry

end problem_solution_l26_26097


namespace matrix_power_four_l26_26237

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3 * Real.sqrt 2 / 2, -3 / 2], ![3 / 2, 3 * Real.sqrt 2 / 2]]

theorem matrix_power_four :
  matrixA ^ 4 = ![![ -81, 0], ![0, -81]] :=
by sorry

end matrix_power_four_l26_26237


namespace find_x_l26_26015

-- Definitions to capture angles and triangle constraints
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def perpendicular (A B : ℝ) : Prop := A + B = 90

-- Given conditions
axiom angle_ABC : ℝ
axiom angle_BAC : ℝ
axiom angle_BCA : ℝ
axiom angle_DCE : ℝ
axiom angle_x : ℝ

-- Specific values for the angles provided in the problem
axiom angle_ABC_is_70 : angle_ABC = 70
axiom angle_BAC_is_50 : angle_BAC = 50

-- Angle BCA in triangle ABC
axiom angle_sum_ABC : angle_sum_triangle angle_ABC angle_BAC angle_BCA

-- Conditional relationships in triangle CDE
axiom angle_DCE_equals_BCA : angle_DCE = angle_BCA
axiom angle_sum_CDE : perpendicular angle_DCE angle_x

-- The theorem we need to prove
theorem find_x : angle_x = 30 := sorry

end find_x_l26_26015


namespace sum_of_coefficients_l26_26133

theorem sum_of_coefficients (n : ℤ) : (finset.sum (finset.range (n.nat_abs + 1)) (λ k, (nat.choose n.nat_abs k) * (-2)^k)) = if n % 2 = 0 then 1 else -1 := 
by 
  sorry

end sum_of_coefficients_l26_26133


namespace correct_set_l26_26557

noncomputable def probability (d : ℕ) : ℝ := 
  if h : d > 0 ∧ d < 10 then (Real.log ((d + 1) : ℝ) / Real.log 10) - (Real.log (d : ℝ) / Real.log 10) else 0

theorem correct_set :
  let probability_1 := probability 1 in
  let probability_set := probability 3 + probability 4 + probability 5 in
  probability_1 = (1 / 3) * probability_set →
  ({3, 4, 5} : set ℕ) = {3, 4, 5} :=
by
  intros,
  -- Proof steps would go here
  sorry

end correct_set_l26_26557


namespace cab_driver_income_day3_l26_26548

theorem cab_driver_income_day3 :
  let income1 := 200
  let income2 := 150
  let income4 := 400
  let income5 := 500
  let avg_income := 400
  let total_income := avg_income * 5 
  total_income - (income1 + income2 + income4 + income5) = 750 := by
  sorry

end cab_driver_income_day3_l26_26548


namespace arithmetic_sequence_sum_ratio_l26_26134

theorem arithmetic_sequence_sum_ratio (a_n : ℕ → ℕ) (S : ℕ → ℕ) 
  (hS : ∀ n, S n = n * a_n 1 + n * (n - 1) / 2 * (a_n 2 - a_n 1)) 
  (h1 : S 6 / S 3 = 4) : S 9 / S 6 = 9 / 4 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l26_26134


namespace find_constant_t_l26_26610

theorem find_constant_t : ∃ t : ℝ, 
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (2 * x^2 + t * x + 8) = 6 * x^4 + (-26) * x^3 + 58 * x^2 + (-76) * x + 40) ↔ t = -6 :=
by {
  sorry
}

end find_constant_t_l26_26610


namespace largest_sphere_radius_l26_26983

theorem largest_sphere_radius :
  ∃ r : ℝ, 
    (let center_sphere : ℝ × ℝ × ℝ := (0, 0, r) in
    let center_torus_circle : ℝ × ℝ × ℝ := (4, 0, 1) in
    let inner_radius_torus := 3 in
    let outer_radius_torus := 5 in
    inner_radius_torus = 4 - 1 ∧ outer_radius_torus = 4 + 1 ∧
    r = 4) :=
sorry

end largest_sphere_radius_l26_26983


namespace probability_multiple_of_3_or_7_l26_26478

theorem probability_multiple_of_3_or_7 : 
  let cards := (Finset.range 30).map (λ n => n + 1) in
  let multiples_of_3 := cards.filter (λ n => n % 3 = 0) in
  let multiples_of_7 := cards.filter (λ n => n % 7 = 0) in
  let multiples_of_21 := cards.filter (λ n => n % 21 = 0) in
  (multiples_of_3.card + multiples_of_7.card - multiples_of_21.card) / cards.card = 13 / 30 :=
by {
  sorry
}

end probability_multiple_of_3_or_7_l26_26478


namespace intersection_M_N_l26_26783

def M := { x : ℝ | x^2 - 2 * x < 0 }
def N := { x : ℝ | abs x < 1 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l26_26783


namespace exists_2x2_square_in_100x100_l26_26541

theorem exists_2x2_square_in_100x100 :
  ∀ (rectangles : ℕ → ℕ × ℕ × ℕ × ℕ), 
    (∀ (i : ℕ), 
      let (x1, y1, x2, y2) := rectangles i 
      in (x2 - x1 = 1 ∧ y2 - y1 = 0) 
      ∨ (x2 - x1 = 0 ∧ y2 - y1 = 1)) →
    (∃ (i j : ℕ),
      i ≠ j ∧
      let (x1i, y1i, x2i, y2i) := rectangles i 
      let (x1j, y1j, x2j, y2j) := rectangles j 
      in x1i = x1j ∧ y1i = y1j ∧ x2i = x2j ∧ y2i = y2j) :=
sorry

end exists_2x2_square_in_100x100_l26_26541


namespace find_a_l26_26278

theorem find_a 
  (x y a : ℝ)
  (h₁ : x - 3 ≤ 0)
  (h₂ : y - a ≤ 0)
  (h₃ : x + y ≥ 0)
  (h₄ : ∃ (x y : ℝ), 2*x + y = 10): a = 4 :=
sorry

end find_a_l26_26278


namespace problem_statement_l26_26303

-- Define the conditions given in the problem
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the coordinates of point P
def P (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Question 1: Define what it means for a line to be tangent to the circle
-- at a given point.
def is_tangent_to_circle (l x y : ℝ → ℝ) : Prop :=
  ∀ x y, circle x y → (l x = y → (P x y ∨ l x = 2 ∨ 3*x + 4*y - 10 = 0))

-- Question 2: Define the condition for the point Q
def trajectory_of_Q (x y x0 y0 : ℝ) : Prop :=
  (2 * y =  x^2 + (x^2/4))

-- State the theorem in Lean 4 (without proof)
theorem problem_statement :
  (∀ x y : ℝ, P x y → is_tangent_to_circle (λ x, -3/4 * (x-2) + 1) x y) ∧
  (∀ x y x0 y0: ℝ, circle x0 y0 → trajectory_of_Q x y x0 y0 :=
begin
  sorry
end

end problem_statement_l26_26303


namespace find_phi_l26_26667

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) - 1 / 2

def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g x

theorem find_phi :
  ∃ φ > 0, (∀ x : ℝ, f (x - φ) = f (-(x - φ)) → x = 0) ∧ 
  ∀ ψ > 0, ψ < φ → ¬ ∀ y : ℝ, f (y - ψ) = -f y :=
Exists.intro (Real.pi / 8) (by
  constructor
  { norm_num [Real.pi_div_two_pos]
  sorry }
  { intro ψ hψ
  sorry })

end find_phi_l26_26667


namespace triangle_midpoint_similarity_l26_26000

theorem triangle_midpoint_similarity
  (A B C D E : Type)
  [is_isosceles_triangle A B C]
  (h₁ : distance A B = 14 * Real.sqrt 2)
  (h₂ : distance A C = 14 * Real.sqrt 2)
  (h₃ : is_midpoint D C A)
  (h₄ : is_midpoint E B D)
  (h₅ : similar_triangles C D E A B C) : distance B D = 14 :=
by
  sorry

end triangle_midpoint_similarity_l26_26000


namespace quadratic_real_roots_l26_26629

theorem quadratic_real_roots (K : ℝ) :
  ∃ x : ℝ, K^2 * x^2 + (K^2 - 1) * x - 2 * K^2 = 0 :=
sorry

end quadratic_real_roots_l26_26629


namespace evariste_stairs_l26_26502

def num_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else num_ways (n - 1) + num_ways (n - 2)

theorem evariste_stairs (n : ℕ) : num_ways n = u_n :=
  sorry

end evariste_stairs_l26_26502


namespace geometric_locus_of_points_l26_26902

-- Define points and planes
variables {P1 P2 : Plane} {T1 T2 : Triangle}

-- Conditions: Two identical equilateral triangles on parallel planes
def is_equilateral (T : Triangle) : Prop := sorry
def are_identical (T1 T2 : Triangle) : Prop := sorry
def are_parallel (P1 P2 : Plane) : Prop := sorry
def center (T : Triangle) : Point := sorry
def is_perpendicular (l : Line) (P : Plane) : Prop := sorry
def connecting_line (p1 p2 : Point) : Line := sorry
def mid_plane (P1 P2 : Plane) : Plane := sorry

variables (h_equilateral_T1 : is_equilateral T1)
          (h_equilateral_T2 : is_equilateral T2)
          (h_identical : are_identical T1 T2)
          (h_on_P1 : on_plane T1 P1)
          (h_on_P2 : on_plane T2 P2)
          (h_parallel : are_parallel P1 P2)
          (h_center_perpendicular : is_perpendicular (connecting_line (center T1) (center T2)) P1)

-- The theorem stating the conclusion
theorem geometric_locus_of_points :
  (∃ (L : Set Point), (L = triangle ∨ L = hexagon) ∧ (L ⊆ mid_plane P1 P2)) :=
sorry

end geometric_locus_of_points_l26_26902


namespace perimeter_BPC_greater_than_ABC_l26_26183

open Mathlib

variables {A B C O P : Type}
variables [IsMetricSpace A] [IsMetricSpace B] [IsMetricSpace C]

-- Define the points and conditions
variable (triangle_ABC : Triangle A B C) 
variable (O : Point)
variable (P : Point)

-- AO is an angle bisector of ∠A and P is on the perpendicular from A to AO
axiom (angle_bisector_A_O : IsAngleBisector (angle A B C) A O)
axiom (perp_to_AO_at_A : IsPerpendicular (line_through_points A O) P (perpendicular_to (line_through_points A O) P))

-- Prove that the perimeter of triangle BPC is greater than the perimeter of triangle ABC
theorem perimeter_BPC_greater_than_ABC (angle_bisector_A_O : IsAngleBisector (angle A B C) A O) 
(perp_to_AO_at_A : IsPerpendicular (line_through_points A O) P (perpendicular_to (line_through_points A O) P)) :
    perimeter (Triangle B P C) > perimeter (Triangle A B C) := sorry

end perimeter_BPC_greater_than_ABC_l26_26183


namespace num_possible_m_values_l26_26828

theorem num_possible_m_values 
  (m : ℕ) 
  (f : ℤ → ℤ) 
  (h_f_def : ∀ x, f x = 2 * x - m * real.sqrt (10 - x) - m + 10)
  (h_m_nat : m ∈ ℕ)
  (h_f_root : ∃ x : ℤ, f x = 0) :
  ∃ (ms : finset ℕ), ms.card = 4 ∧ ∀ m' ∈ ms, m' ∈ ℕ :=
sorry

end num_possible_m_values_l26_26828


namespace constant_term_binomial_expansion_l26_26366

theorem constant_term_binomial_expansion 
  (x : ℚ) :
  let expr := (x + 1/x) ^ 6 in
  ∃ c, ∀ x, expr = c → c = 20 :=
by
  let expr := (x + 1/x) ^ 6
  sorry

end constant_term_binomial_expansion_l26_26366


namespace line_l_equation_line_l1_equation_l26_26280

theorem line_l_equation :
  ∃ (x y : ℝ), (4 * x - 3 * y + 12 = 0) ∧ ((∃ x y, (x = 0 ∧ y = 4)) ∧ sum_intercepts_eq_1 x y) := 
sorry

theorem line_l1_equation (m : ℝ) :
  ∃ (l1 : ℝ), line_parallel (4*x - 3*y + m = 0) l ∧ distance_lines_eq_2 (4*x - 3*y + m = 0) l :=
sorry

end line_l_equation_line_l1_equation_l26_26280


namespace monomial_coefficient_degree_l26_26466

noncomputable def monomial := -2 * (λ x, (x:ℕ) ^ 2) * (λ y, y)

def coefficient (m: ℕ → ℕ → ℕ) : ℤ := -2

def degree (m: ℕ → ℕ → ℕ) : ℕ := 3

theorem monomial_coefficient_degree :
  (coefficient monomial = -2) ∧ (degree monomial = 3) := 
by
  sorry

end monomial_coefficient_degree_l26_26466


namespace total_games_proof_l26_26356

def num_teams : ℕ := 20
def num_games_per_team_regular_season : ℕ := 38
def total_regular_season_games : ℕ := num_teams * (num_games_per_team_regular_season / 2)
def num_games_per_team_mid_season : ℕ := 3
def total_mid_season_games : ℕ := num_teams * num_games_per_team_mid_season
def quarter_finals_teams : ℕ := 8
def quarter_finals_matchups : ℕ := quarter_finals_teams / 2
def quarter_finals_games : ℕ := quarter_finals_matchups * 2
def semi_finals_teams : ℕ := quarter_finals_matchups
def semi_finals_matchups : ℕ := semi_finals_teams / 2
def semi_finals_games : ℕ := semi_finals_matchups * 2
def final_teams : ℕ := semi_finals_matchups
def final_games : ℕ := final_teams * 2
def total_playoff_games : ℕ := quarter_finals_games + semi_finals_games + final_games

def total_season_games : ℕ := total_regular_season_games + total_mid_season_games + total_playoff_games

theorem total_games_proof : total_season_games = 454 := by
  -- The actual proof will go here
  sorry

end total_games_proof_l26_26356


namespace repeating_decimal_division_l26_26917

theorem repeating_decimal_division :
  let x := 0 + 54 / 99 in -- 0.545454... = 54/99 = 6/11
  let y := 0 + 18 / 99 in -- 0.181818... = 18/99 = 2/11
  x / y = 3 :=
by
  sorry

end repeating_decimal_division_l26_26917


namespace circumradius_range_parabola_l26_26058

-- Define the points on the parabola and the circumradius R.
def parabola_points (a : ℝ) : Prop :=
  let A := (a, a^2) in
  let B := (-a, a^2) in
  let C := (0, 0) in
  ∃ R : ℝ, ∀ (A B C : ℝ × ℝ), 
    A = (a, a^2) ∧ B = (-a, a^2) ∧ C = (0, 0) ∧ 
    ∃ (R > 0), (R = (a^2 + 1) / 2 ∧ a ≠ 0)

-- Theorem stating the range of values for R
theorem circumradius_range_parabola (a : ℝ) (h : a ≠ 0) : 
  parabola_points a → ∃ R : ℝ, R > 1 / 2 ∧ 
    R ∈ set.Ioi (1 / 2) :=
by
  sorry

end circumradius_range_parabola_l26_26058


namespace minimize_costs_l26_26958

def total_books : ℕ := 150000
def handling_fee_per_order : ℕ := 30
def storage_fee_per_1000_copies : ℕ := 40
def evenly_distributed_books : Prop := true --Assuming books are evenly distributed by default

noncomputable def optimal_order_frequency : ℕ := 10
noncomputable def optimal_batch_size : ℕ := 15000

theorem minimize_costs 
  (handling_fee_per_order : ℕ) 
  (storage_fee_per_1000_copies : ℕ) 
  (total_books : ℕ) 
  (evenly_distributed_books : Prop)
  : optimal_order_frequency = 10 ∧ optimal_batch_size = 15000 := sorry

end minimize_costs_l26_26958


namespace spinner_divisible_by_5_probability_l26_26564

theorem spinner_divisible_by_5_probability :
  let outcomes := [1, 2, 5, 6]
  let total_outcomes := outcomes.length ^ 3
  let favorable_outcomes := (outcomes.length - 1) * (outcomes.length - 1) * 1
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 4 :=
by
  -- Definitions
  let outcomes := [1, 2, 5, 6]
  let total_outcomes := outcomes.length ^ 3
  let favorable_outcomes := (outcomes.length - 1) * (outcomes.length - 1) * 1
  
  -- Proof (skipped with sorry)
  have h : (favorable_outcomes / total_outcomes : ℚ) = 1 / 4 := sorry
  exact h

end spinner_divisible_by_5_probability_l26_26564


namespace complex_solution_l26_26724

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26724


namespace remainder_cubes_mod_6_l26_26413

noncomputable def a (n : ℕ) : ℕ :=
if h : n > 0 ∧ n <= 2023 then n else 0

theorem remainder_cubes_mod_6 
  (h1 : ∀ n m : ℕ, n < m ∧ m <= 2023 → a n < a m)
  (h2 : ∑ i in finset.range 2023, a (i + 1) = 2023 ^ 2023) :
  (∑ i in finset.range 2023, if even i then a (i + 1) ^ 3 else -a (i + 1) ^ 3) % 6 = 1 :=
sorry

end remainder_cubes_mod_6_l26_26413


namespace sum_possible_values_e_sum_of_possible_values_for_e_l26_26177

theorem sum_possible_values_e (e : ℤ) (h : |2 - e| = 5) : e = -3 ∨ e = 7 :=
  by
  sorry

theorem sum_of_possible_values_for_e (h : |2 - e| = 5) : ([-3, 7].sum = 4) :=
  by
  have h_e_possible : e = -3 ∨ e = 7 := sum_possible_values_e e h
  sorry

end sum_possible_values_e_sum_of_possible_values_for_e_l26_26177


namespace set_difference_NM_l26_26245

open Set

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_NM :
  let M := {1, 2, 3, 4, 5}
  let N := {1, 2, 3, 7}
  setDifference N M = {7} :=
by
  sorry

end set_difference_NM_l26_26245


namespace range_of_m_is_leq_3_l26_26318

noncomputable def is_range_of_m (m : ℝ) : Prop :=
  ∀ x : ℝ, 5^x + 3 > m

theorem range_of_m_is_leq_3 (m : ℝ) : is_range_of_m m ↔ m ≤ 3 :=
by
  sorry

end range_of_m_is_leq_3_l26_26318


namespace find_starting_number_l26_26890

theorem find_starting_number (num_even_ints: ℕ) (end_num: ℕ) (h_num: num_even_ints = 35) (h_end: end_num = 95) : 
  ∃ start_num: ℕ, start_num = 24 ∧ (∀ n: ℕ, (start_num + 2 * n ≤ end_num ∧ n < num_even_ints)) := by
  sorry

end find_starting_number_l26_26890


namespace hexagon_inequality_l26_26417

theorem hexagon_inequality 
  (A B C D E F : Type*)
  [ConvexHexagon A B C D E F]
  (hAB_BC: AB = BC)
  (hCD_DE: CD = DE)
  (hEF_FA : EF = FA) :
  (BC / BE + DE / DA + FA / FC) ≥ 3 / 2 := 
sorry

end hexagon_inequality_l26_26417


namespace james_carrot_sticks_l26_26385

theorem james_carrot_sticks (x : ℕ) (h : x + 15 = 37) : x = 22 :=
by {
  sorry
}

end james_carrot_sticks_l26_26385


namespace determine_z_l26_26699

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26699


namespace natural_solution_unique_l26_26261

theorem natural_solution_unique (n : ℕ) (h : (2 * n - 1) / n^5 = 3 - 2 / n) : n = 1 := by
  sorry

end natural_solution_unique_l26_26261


namespace probability_successful_login_l26_26167

theorem probability_successful_login :
  let letters := 4
  let digits := 3
  let total_outcomes := letters * digits
  let probability := 1 / total_outcomes
  probability = 1 / 12 :=
by
  let letters := 4
  let digits := 3
  let total_outcomes := letters * digits
  have probability_def : probability = 1 / total_outcomes := rfl
  have total_outcomes_calc : total_outcomes = 12 := by norm_num
  rw [total_outcomes_calc] at probability_def
  exact probability_def

end probability_successful_login_l26_26167


namespace seven_digit_palindromes_count_l26_26266

theorem seven_digit_palindromes_count : 
  let a_choices := 2 in
  let bcd_choices := 10 * 10 * 10 in
  a_choices * bcd_choices = 2000 := 
by 
  sorry

end seven_digit_palindromes_count_l26_26266


namespace brownies_left_l26_26144

theorem brownies_left (initial : ℕ) (tina_ate : ℕ) (husband_ate : ℕ) (shared : ℕ) 
                      (h_initial : initial = 24)
                      (h_tina : tina_ate = 10)
                      (h_husband : husband_ate = 5)
                      (h_shared : shared = 4) : 
  initial - tina_ate - husband_ate - shared = 5 :=
by
  rw [h_initial, h_tina, h_husband, h_shared]
  exact Nat.sub_sub_sub_cancel 24 10 5 4 sorry

end brownies_left_l26_26144


namespace monotonic_decreasing_implies_a_bound_strict_monotonic_increase_full_interval_l26_26307

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * (x + Real.log x) + 2

theorem monotonic_decreasing_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x a = f x a → f (x + 1) a <  f x a ) → (a ≤ -8/3) :=
by
  sorry

theorem strict_monotonic_increase_full_interval (a : ℝ) :
  (∀ x1 x2 ∈ Set.Ioi 0, x1 < x2 → f x1 a + x1 < f x2 a + x2) → (a ≥ 3 - 2 * Real.sqrt 2) :=
by
  sorry

end monotonic_decreasing_implies_a_bound_strict_monotonic_increase_full_interval_l26_26307


namespace solve_for_z_l26_26734

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26734


namespace second_smallest_number_is_56_l26_26149

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_place_is_five (n : ℕ) : Prop :=
  (n / 10) = 5

noncomputable def second_smallest_two_digit_with_five (digits : List ℕ) : ℕ :=
  let possible_numbers := List.filter (λ n => is_two_digit_number n ∧ tens_place_is_five n) $
    List.permutations digits
  let sorted_numbers := List.sort Nat.lt possible_numbers
  sorted_numbers.tail.head

theorem second_smallest_number_is_56 (digits : List ℕ) (h1 : 1 ∈ digits) (h5 : 5 ∈ digits) (h6 : 6 ∈ digits) (h9 : 9 ∈ digits) (h_len : digits.length = 4) :
  second_smallest_two_digit_with_five digits = 56 :=
sorry

end second_smallest_number_is_56_l26_26149


namespace distance_from_pole_to_line_l26_26785

-- Definition: Convert given polar equation to cartesian equation
def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * Real.cos theta in
  let y := rho * Real.sin theta in
  (x, y)

-- Theorem: The distance from the origin to the line represented by the polar equation
theorem distance_from_pole_to_line (rho theta: ℝ) :
  (1:ℝ) = abs (-2) / sqrt (2 + 2) := by
  sorry

end distance_from_pole_to_line_l26_26785


namespace intervals_of_monotonicity_max_value_of_interval_l26_26311

noncomputable def f (x : ℝ) : ℝ := x * (abs (x - 4))

theorem intervals_of_monotonicity :
  (∀ x < 2, ∀ y < 2, x ≤ y → f x ≤ f y) ∧
  (∀ x ≥ 4, ∀ y ≥ 4, x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ set.Icc 2 4, ∀ y ∈ set.Icc 2 4, x ≤ y → f x ≥ f y) := 
sorry

theorem max_value_of_interval (m : ℝ) (h : 0 < m) :
  ∃ c, ∀ x ∈ set.Icc 0 m, f x ≤ c ∧ 
  ((0 < m ∧ m < 2) → c = m * (4 - m)) ∧
  ((2 ≤ m ∧ m ≤ 2 + 2 * real.sqrt 2) → c = 4) ∧
  ((m > 2 + 2 * real.sqrt 2) → c = m * (4 - m)) :=
sorry

end intervals_of_monotonicity_max_value_of_interval_l26_26311


namespace gcd_of_fraction_in_lowest_terms_l26_26573

theorem gcd_of_fraction_in_lowest_terms (n : ℤ) (h : n % 2 = 1) : Int.gcd (2 * n + 2) (3 * n + 2) = 1 := 
by 
  sorry

end gcd_of_fraction_in_lowest_terms_l26_26573


namespace triangle_circumcenter_ratio_l26_26002

theorem triangle_circumcenter_ratio 
  (ABC : Triangle) 
  (A B C : Point) 
  (α : Angle ABC.A := 60)
  (hAB_gt_AC : ABC.B.distance_to A > ABC.C.distance_to A)
  (O : Point := ABC.circumcenter)
  (BE : Line := ABC.altitude B)
  (CF : Line := ABC.altitude C)
  (H : Point := BE ∩ CF)
  (M : Point := segment BH.point)
  (N : Point := segment HF.point)
  (hBM_eq_CN : M.distance_to B = N.distance_to C) :
  (M.distance_to H + N.distance_to H) / O.distance_to H = Real.sqrt 3 :=
sorry

end triangle_circumcenter_ratio_l26_26002


namespace num_students_in_research_study_group_prob_diff_classes_l26_26555

-- Define the number of students in each class and the number of students selected from class (2)
def num_students_class1 : ℕ := 18
def num_students_class2 : ℕ := 27
def selected_from_class2 : ℕ := 3

-- Prove the number of students in the research study group
theorem num_students_in_research_study_group : 
  (∃ (m : ℕ), (m / 18 = 3 / 27) ∧ (m + selected_from_class2 = 5)) := 
by
  sorry

-- Prove the probability that the students speaking in both activities come from different classes
theorem prob_diff_classes : 
  (12 / 25 = 12 / 25) :=
by
  sorry

end num_students_in_research_study_group_prob_diff_classes_l26_26555


namespace probability_sum_is_correct_probability_not_all_same_is_correct_l26_26547

-- Define the conditions: the box containing three cards numbered 1, 2, and 3
def card_numbers := {1, 2, 3}

-- Define the total number of outcomes when drawing 3 cards with replacement
def total_outcomes : ℕ := 27

-- Define the outcomes that satisfy a + b = c
def outcomes_satisfying_sum : ℕ := 3

-- Define the outcomes where all cards are the same
def outcomes_all_same : ℕ := 3

-- Define the probability that the sum condition holds
def probability_sum := outcomes_satisfying_sum/total_outcomes

-- Define the probability that all cards are the same
def probability_all_same := outcomes_all_same/total_outcomes

-- Define the probability that not all cards are the same
def probability_not_all_same := 1 - probability_all_same

-- Prove the first part
theorem probability_sum_is_correct : probability_sum = 1/9 := by
  sorry

-- Prove the second part
theorem probability_not_all_same_is_correct : probability_not_all_same = 8/9 := by
  sorry

end probability_sum_is_correct_probability_not_all_same_is_correct_l26_26547


namespace certain_number_is_correct_l26_26489

theorem certain_number_is_correct :
  let x := 0.52 in
  (0.02)^2 + x^2 + (0.035)^2 = 100 * (0.002)^2 + (0.052)^2 + (0.0035)^2 :=
by
  let x := 0.52
  sorry

end certain_number_is_correct_l26_26489


namespace tree_edges_count_l26_26187

theorem tree_edges_count (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj] (conn : G.isConnected) (acyclic : ¬G.isCyclic) :
  G.edgeFinset.card = Fintype.card V - 1 := 
sorry

end tree_edges_count_l26_26187


namespace find_z_l26_26744

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26744


namespace determinant_zero_l26_26255

variable (θ φ : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ [0, Real.cos θ, Real.sin θ],
      [-Real.cos θ, 0, Real.cos φ],
      [-Real.sin θ, -Real.cos φ, 0] ]

theorem determinant_zero : Matrix.det (matrix θ φ) = 0 := by
  sorry

end determinant_zero_l26_26255


namespace area_of_tangent_triangle_l26_26678

theorem area_of_tangent_triangle (a : ℝ) (ha : a ≠ 0) (t : ℝ) (ht : t ≠ 0) :
    let y := λ x : ℝ, a / x
    let tangent_slope := -a / (t^2)
    let tangent_line := λ x, tangent_slope * (x - t) + a / t
    let y_intercept := 2 * a / t
    let x_intercept := 2 * t
    abs (1 / 2 * y_intercept * x_intercept) = 2 * abs a :=
by
  sorry

end area_of_tangent_triangle_l26_26678


namespace integral_sqrt_2_minus_x_sq_l26_26592

theorem integral_sqrt_2_minus_x_sq :
  ∫ x in -real.sqrt 2..real.sqrt 2, real.sqrt (2 - x^2) = real.pi := sorry

end integral_sqrt_2_minus_x_sq_l26_26592


namespace find_b_squared_l26_26554

noncomputable def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * complex.I) * z

theorem find_b_squared
  (a b : ℝ)
  (f_def : ∀ z : ℂ, f a b z = (a + b * complex.I) * z)
  (dist_property : ∀ z : ℂ, complex.abs (f a b z - z) = complex.abs (f a b z))
  (a_val : a = 2)
  (modulus_property : complex.abs (a + b * complex.I) = 10) :
  b^2 = 99 :=
sorry

end find_b_squared_l26_26554


namespace Tonya_buys_3_lego_sets_l26_26147

-- Definitions based on conditions
def num_sisters : Nat := 2
def num_dolls : Nat := 4
def price_per_doll : Nat := 15
def price_per_lego_set : Nat := 20

-- The amount of money spent on each sister should be the same
def amount_spent_on_younger_sister := num_dolls * price_per_doll
def amount_spent_on_older_sister := (amount_spent_on_younger_sister / price_per_lego_set)

-- Proof statement
theorem Tonya_buys_3_lego_sets : amount_spent_on_older_sister = 3 :=
by
  sorry

end Tonya_buys_3_lego_sets_l26_26147


namespace geography_teachers_l26_26355

universe u

-- Declare types for Subjects and Teachers
inductive Subject : Type u
| chemistry | english | french | geography | mathematics | physics

inductive Teacher : Type u
| Barna | Kovacs | Horvath | Nagy

open Subject Teacher

-- Define the conditions as assumed facts
axiom each_teacher_teaches_three : ∀ (t : Teacher), (finset.filter (λ s, t ∈ teaches s) finset.univ).card = 3
axiom each_subject_taught_by_two : ∀ (s : Subject), (finset.filter (λ t, t ∈ teaches s) finset.univ).card = 2
axiom english_and_french_same_teachers : teaches english = teaches french
axiom nagy_and_kovacs_two_common : (finset.filter (λ s, Kovacs ∈ teaches s ∧ Nagy ∈ teaches s) finset.univ).card = 2
axiom teaches_math : teaches mathematics = {Nagy, Horvath}
axiom teaches_chemistry : teaches chemistry = (teaches chemistry).insert Horvath
axiom kovacs_not_teach_physics : Kovacs ∉ teaches physics

-- Define the teaching relation
def teaches : Subject → finset Teacher := sorry

-- The statement to prove
theorem geography_teachers : teaches geography = {Barna, Kovacs} :=
sorry

end geography_teachers_l26_26355


namespace find_phi_shifted_function_l26_26340

theorem find_phi_shifted_function (φ : ℝ) (hφ1 : 0 < φ) (hφ2 : φ < π) :
  (∀ x : ℝ, cos (2 * (x + π / 12) + φ) = cos (2 * x + π / 6 + φ))
  ∧ (∀ x : ℝ, cos (-(2 * x + π / 6 + φ)) = - cos (2 * x + π / 6 + φ)) →
  φ = π / 3 :=
sorry

end find_phi_shifted_function_l26_26340


namespace changfei_class_l26_26100

theorem changfei_class (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m + n = 9 :=
sorry

end changfei_class_l26_26100


namespace sum_of_first_12_terms_proof_l26_26284

noncomputable def a : ℕ → ℝ
| 0 := a_0   -- Initial condition for the sequence
| (n+1) := sorry

def recurrence_relation (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n+1) + (-1)^n * a n = 2 * n - 1

def sum_of_first_12_terms (a : ℕ → ℝ) : ℝ :=
a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11

theorem sum_of_first_12_terms_proof (a_0 : ℝ) 
  (h : recurrence_relation a) : 
  sum_of_first_12_terms a = 78 := 
sorry

end sum_of_first_12_terms_proof_l26_26284


namespace equal_arcs_equal_chords_l26_26845

open EuclideanGeometry

theorem equal_arcs_equal_chords {A B C D O : Point}
    (h_circle : Circle O) 
    (h_arc_eq : arc_eq h_circle A B h_circle C D) 
    (h_OA : on_circle A h_circle) 
    (h_OB : on_circle B h_circle) 
    (h_OC : on_circle C h_circle) 
    (h_OD : on_circle D h_circle) : 
    dist A B = dist C D :=
by
sorry

end equal_arcs_equal_chords_l26_26845


namespace length_of_MN_l26_26789

open EuclideanGeometry

theorem length_of_MN (DE EF FD : ℝ) (hDE : DE = 10) (hEF : EF = 12) (hFD : FD = 14) :
  ∃ (D E F K : Point) (G : Point ∈ line_between D E) (H : Point ∈ line_between D F)
  (M : Point ∈ line_between D K) (N : Point ∈ line_between D K),
    Altitude D K ∧
    bisector G E D ∧
    bisector H F D ∧
    median G E D ∧
    median H F D ∧
    distance M N = 0 := 
sorry

end length_of_MN_l26_26789


namespace circle_through_A_tangent_to_line_and_center_on_line_l26_26616

variable (a r : ℝ)

/-- Given point A, line1, and line2 as specified in the problem, the standard equation of the circle is derived. -/
theorem circle_through_A_tangent_to_line_and_center_on_line (circle_eq : ℝ → ℝ → ℝ)
  (A : ℝ × ℝ := (0, -1))
  (line1 : ℝ → ℝ → Prop := λ x y, x + y - 1 = 0)
  (line2 : ℝ → ℝ → Prop := λ x y, y + 2*x = 0)
  (center_on_line : ∃ c : ℝ × ℝ, line2 c.1 c.2) :
  circle_eq ((1 : ℝ) - a, (2 : ℝ) + 2*a) = r^2 ∨
  circle_eq (((1 : ℝ) / 9), ((2 : ℝ) / (9))) = (50 : ℝ) / 81 :=
sorry

end circle_through_A_tangent_to_line_and_center_on_line_l26_26616


namespace difference_is_correct_l26_26074

-- Define the given constants and conditions
def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def monthly_payment : ℕ := 65
def number_of_monthly_payments : ℕ := 24

-- Define the derived quantities based on the given conditions
def total_monthly_payments : ℕ := monthly_payment * number_of_monthly_payments
def total_amount_paid : ℕ := down_payment + total_monthly_payments
def difference : ℕ := total_amount_paid - purchase_price

-- The statement to be proven
theorem difference_is_correct : difference = 260 := by
  sorry

end difference_is_correct_l26_26074


namespace sum_of_real_solutions_l26_26617

theorem sum_of_real_solutions : 
  (∀ x : ℝ, (x - 2) / (x^2 + 4 * x + 1) = (x - 5) / (x^2 - 10 * x) → 
  x^2 + 4 * x + 1 ≠ 0 ∧ x^2 - 10 * x ≠ 0) →
  (∑ x in {x : ℝ | (x - 2) / (x^2 + 4 * x + 1) = (x - 5) / (x^2 - 10 * x)}, x) = 39 / 11 :=
by
  intro h
  sorry

end sum_of_real_solutions_l26_26617


namespace triangle_area_ratios_l26_26940

theorem triangle_area_ratios (K : ℝ) 
  (hCD : ∃ AC, ∃ CD, CD = AC / 4) 
  (hAE : ∃ AB, ∃ AE, AE = AB / 5) 
  (hBF : ∃ BC, ∃ BF, BF = BC / 3) :
  ∃ area_N1N2N3, area_N1N2N3 = (8 / 15) * K :=
by
  sorry

end triangle_area_ratios_l26_26940


namespace time_to_coffee_shop_l26_26677

theorem time_to_coffee_shop 
  (constant_pace : Prop)
  (time_to_store : ℝ)
  (distance_to_store : ℝ)
  (halfway_to_store_distance : ℝ) :
  constant_pace →
  time_to_store = 36 →
  distance_to_store = 4 →
  halfway_to_store_distance = 2 →
  ∃ (time_to_coffee_shop : ℝ), time_to_coffee_shop = 18 :=
by
  intros _
  intro h1
  intro h2
  intro h3
  use 18
  sorry

end time_to_coffee_shop_l26_26677


namespace slower_train_pass_time_approx_l26_26182

-- Define the constants
def L : ℝ := 500 -- Length of each train in meters
def v1 : ℝ := 45 -- Speed of the first train in km/hr
def v2 : ℝ := 30 -- Speed of the second train in km/hr

-- Convert speeds from km/hr to m/s
def v1_mps : ℝ := v1 * 1000 / 3600
def v2_mps : ℝ := v2 * 1000 / 3600

-- Calculate the relative speed in m/s
def relative_speed : ℝ := v1_mps + v2_mps

-- Calculate the time taken for the slower train to pass the driver of the faster one
def time_taken : ℝ := L / relative_speed

-- The goal is to prove that the time taken is approximately 24.01 seconds
theorem slower_train_pass_time_approx : Real.ApproxEq time_taken 24.01 :=
by 
  sorry

end slower_train_pass_time_approx_l26_26182


namespace fixed_point_of_line_l26_26625

theorem fixed_point_of_line (a : ℝ) : ∀ y : ℝ, (y = a * 3 - 3 * a + 2) → y = 2 :=
by
  intro y h
  rw [← h]
  sorry

end fixed_point_of_line_l26_26625


namespace median_and_mode_of_jump_ropes_l26_26962

open Nat Set

theorem median_and_mode_of_jump_ropes :
  let data := [158, 158, 163, 163, 163, 163, 167, 167, 170, 171]
  let median := (163 + 163) / 2
  let mode := 163
  (data.nth ((data.length - 1) / 2) = median) ∧ mode = data.mode := 
by
  sorry

end median_and_mode_of_jump_ropes_l26_26962


namespace cos_B_value_sin_2B_plus_pi_over_6_value_l26_26346

variables {A B C : ℝ} {a b c : ℝ}
  (triangleABC : ∀ {A B C : ℝ}, Type)
  (opposite_sides : (A B C : ℝ) → (a b c : ℝ))

-- Conditions
variables (h1 : b + c = 2 * a)
variables (h2 : 3 * c * sin B = 4 * a * sin C)

theorem cos_B_value :
  b + c = 2 * a → 
  3 * c * sin B = 4 * a * sin C → 
  cos B = -1 / 4 :=
sorry

theorem sin_2B_plus_pi_over_6_value :
  b + c = 2 * a → 
  3 * c * sin B = 4 * a * sin C → 
  sin (2 * B + π / 6) = (3 * sqrt(5) + 7) / 16 :=
sorry

end cos_B_value_sin_2B_plus_pi_over_6_value_l26_26346


namespace triangle_classification_l26_26275

theorem triangle_classification (a b c : ℕ) (h : a + b + c = 12) :
((
  (a = b ∨ b = c ∨ a = c)  -- Isosceles
  ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)  -- Right-angled
  ∨ (a = b ∧ b = c)  -- Equilateral
)) :=
sorry

end triangle_classification_l26_26275


namespace graph_shift_correct_l26_26473

def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 

def g_shifted (x : ℝ) : ℝ := g x - 3

theorem graph_shift_correct :
  (∀ x, ∃ y, g_shifted x = y) := sorry

end graph_shift_correct_l26_26473


namespace car_can_travel_more_miles_after_modification_l26_26549

-- Given conditions
def original_fuel_efficiency : ℝ := 28  -- miles per gallon
def fuel_usage_reduction : ℝ := 0.80
def fuel_tank_capacity : ℝ := 15  -- gallons

-- Intermediate calculations based on the conditions
def modified_fuel_efficiency : ℝ := original_fuel_efficiency / fuel_usage_reduction  -- MPH after modification
def original_distance : ℝ := original_fuel_efficiency * fuel_tank_capacity  -- total distance before modification
def modified_distance : ℝ := modified_fuel_efficiency * fuel_tank_capacity  -- total distance after modification

-- Prove the question: 
theorem car_can_travel_more_miles_after_modification :
  modified_distance - original_distance = 84 := 
by
  -- Proof goes here
  sorry

end car_can_travel_more_miles_after_modification_l26_26549


namespace greatest_three_digit_number_l26_26923

theorem greatest_three_digit_number : 
  ∃ x : ℕ, x < 1000 ∧ x ≡ 2 [MOD 8] ∧ x ≡ 4 [MOD 6] ∧ ∀ y : ℕ, y < 1000 → y ≡ 2 [MOD 8] → y ≡ 4 [MOD 6] → x ≥ y := 
begin
  use 986,
  split,
  { -- x < 1000
    norm_num, },
  split,
  { -- x ≡ 2 [MOD 8]
    norm_num, },
  split,
  { -- x ≡ 4 [MOD 6]
    norm_num, },
  { -- ∀ y : ℕ, y < 1000 → y ≡ 2 [MOD 8] → y ≡ 4 [MOD 6] → x ≥ y
    intros y hy h2 h4,
    sorry, }, 
end

end greatest_three_digit_number_l26_26923


namespace values_for_decreasing_power_function_l26_26341

theorem values_for_decreasing_power_function (m : ℤ) (f : ℝ → ℝ) :
  (∀ x > 0, f x = x^(m^2 - m - 2) ∧ (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂)) →
  m ∈ {0, 1} :=
by
  sorry

end values_for_decreasing_power_function_l26_26341


namespace A_fraction_simplification_l26_26333

noncomputable def A : ℚ := 
  ((3/8) * (13/5)) / ((5/2) * (6/5)) +
  ((5/8) * (8/5)) / (3 * (6/5) * (25/6)) +
  (20/3) * (3/25) +
  28 +
  (1 / 9) / 7 +
  (1/5) / (9 * 22)

theorem A_fraction_simplification :
  let num := 1901
  let denom := 3360
  (A = num / denom) :=
sorry

end A_fraction_simplification_l26_26333


namespace problem_solution_l26_26836

-- Definitions of sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 }
def B : Set ℝ := {-2, -1, 1, 2}

-- Complement of set A in reals
def C_A : Set ℝ := {x | x < 0}

-- Lean theorem statement
theorem problem_solution : (C_A ∩ B) = {-2, -1} :=
by sorry

end problem_solution_l26_26836


namespace zander_stickers_l26_26524

/-- Zander starts with 100 stickers, Andrew receives 1/5 of Zander's total, 
    and Bill receives 3/10 of the remaining stickers. Prove that the total 
    number of stickers given to Andrew and Bill is 44. -/
theorem zander_stickers :
  let total_stickers := 100
  let andrew_fraction := 1 / 5
  let remaining_stickers := total_stickers - (total_stickers * andrew_fraction)
  let bill_fraction := 3 / 10
  (total_stickers * andrew_fraction) + (remaining_stickers * bill_fraction) = 44 := 
by
  sorry

end zander_stickers_l26_26524


namespace coeff_x_n_plus_1_leq_p1_sq_div_2_l26_26574

noncomputable def p (n : ℕ) : Polynomial ℝ := 
  ∑ i in Finset.range (n + 1), (λ (i : ℕ), (Polynomial.C (coeff (Polynomial.X i))))

theorem coeff_x_n_plus_1_leq_p1_sq_div_2 (n : ℕ) :
  (∀ i ∈ Finset.range (n + 1), 0 ≤ p n.coeff i) →
  (∀ i ∈ Finset.range (n + 1), p n.coeff i ≤ p n.coeff 0) →
  (Polynomial.Coeff (p n * p n) (n + 1) ≤ ((p n).eval 1) ^ 2 / 2) :=
by
  sorry

end coeff_x_n_plus_1_leq_p1_sq_div_2_l26_26574


namespace calculate_expression_l26_26532

theorem calculate_expression (x : ℝ) (h : x = 0.000333333333...): 
  (10 ^ 5 - 10 ^ 3) * x = 32.67 :=
by
  -- We skip the proof here by using sorry
  sorry

end calculate_expression_l26_26532


namespace alice_has_winning_strategy_l26_26111

theorem alice_has_winning_strategy :
  ∃ f : ℕ → ℕ × ℕ → ℕ × list (ℕ × ℕ), 
  (∀ (turn : ℕ) (stacks : list (ℕ × ℕ))
    (h_stacks : stacks = [(2,1), (2,2), (2,3), (1,4), (1,5)])
    (h_plays : ∀ s ∈ stacks, ∃ x y, s = (x,y) ∧ (x = 1 ∨ x = 2))
    (h_alternate : turn % 2 = 0), 
     let move := f turn (1,1) in
     (stacks ≠ [] ∧ move.2 = [] → f (turn + 1) (1,1) = move)) :=
sorry

end alice_has_winning_strategy_l26_26111


namespace cost_of_6_melons_l26_26107

theorem cost_of_6_melons (cost_per_melon : ℕ) (number_of_melons : ℕ) (h : cost_per_melon = 3) (h2 : number_of_melons = 6) : cost_per_melon * number_of_melons = 18 :=
by
  rw [h, h2]
  norm_num

end cost_of_6_melons_l26_26107


namespace necessary_and_sufficient_condition_l26_26636

noncomputable def f (a x : ℝ) : ℝ := a * x - x^2

theorem necessary_and_sufficient_condition (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) := by
  sorry

end necessary_and_sufficient_condition_l26_26636


namespace repeating_decimal_division_l26_26916

theorem repeating_decimal_division :
  let x := 0 + 54 / 99 in -- 0.545454... = 54/99 = 6/11
  let y := 0 + 18 / 99 in -- 0.181818... = 18/99 = 2/11
  x / y = 3 :=
by
  sorry

end repeating_decimal_division_l26_26916


namespace other_root_is_seven_thirds_l26_26439

theorem other_root_is_seven_thirds {m : ℝ} (h : ∃ r : ℝ, 3 * r * r + m * r - 7 = 0 ∧ r = -1) : 
  ∃ r' : ℝ, r' ≠ -1 ∧ 3 * r' * r' + m * r' - 7 = 0 ∧ r' = 7 / 3 :=
by
  sorry

end other_root_is_seven_thirds_l26_26439


namespace solve_equation_l26_26457

theorem solve_equation (x : ℝ) (n : ℤ) (h : sin x - cos x > 0) :
  (2 * (Real.tan (8 * x))^4 + 4 * (sin (3 * x)) * (sin (5 * x)) - cos (6 * x) - cos (10 * x) + 2) / 
  Real.sqrt (sin x - cos x) = 0 ↔ 
  (∃ n : ℤ, x = π / 2 + 2 * π * n) ∨ 
  (∃ n : ℤ, x = 3 * π / 4 + 2 * π * n) ∨ 
  (∃ n : ℤ, x = π + 2 * π * n) :=
sorry

end solve_equation_l26_26457


namespace magnitude_of_z_l26_26642

noncomputable def z (i : ℂ) : ℂ := (-1 / 2) + (1 / 2) * i

theorem magnitude_of_z : 
  (z (complex.I)) * ((1 - complex.I) ^ 2) = 1 + complex.I → 
  complex.abs (z complex.I) = (real.sqrt 2) / 2 :=
by
  intro h
  sorry

end magnitude_of_z_l26_26642


namespace large_triangle_subdivision_5_large_triangle_subdivision_3_l26_26904

def is_large (T : Triangle) : Prop :=
  ∀ (a b c : ℝ), T.sides = (a, b, c) → a > 1 ∧ b > 1 ∧ c > 1

def equilateral_triangle (T : Triangle) (s : ℝ) : Prop :=
  T.sides = (s, s, s)

theorem large_triangle_subdivision_5 :
  ∃ (T : Triangle), equilateral_triangle T 5 ∧
  ∃ (T₁ T₂ : list Triangle), (∀ t ∈ T₁, is_large t) ∧ (∀ t ∈ T₂, is_large t) ∧ T₁.length ≥ 100 ∧ T₂.length ≥ 100 ∧
  (∀ t₃ t₄ ∈ T₁, (t₃ = t₄) ∨ (∃ p : Point, p ∈ t₃.vert ∧ p ∈ t₄.vert) ∨ (∃ s : Side, s ∈ t₃.sides ∧ s ∈ t₄.sides)) ∧ 
  (∀ t₃ t₄ ∈ T₂, (t₃ = t₄) ∨ (∃ p : Point, p ∈ t₃.vert ∧ p ∈ t₄.vert) ∨ (∃ s : Side, s ∈ t₃.sides ∧ s ∈ t₄.sides)) :=
sorry

theorem large_triangle_subdivision_3 :
  ∃ (T : Triangle), equilateral_triangle T 3 ∧
  ∃ (T₁ T₂ : list Triangle), (∀ t ∈ T₁, is_large t) ∧ (∀ t ∈ T₂, is_large t) ∧ T₁.length ≥ 100 ∧ T₂.length ≥ 100 ∧
  (∀ t₃ t₄ ∈ T₁, (t₃ = t₄) ∨ (∃ p : Point, p ∈ t₃.vert ∧ p ∈ t₄.vert) ∨ (∃ s : Side, s ∈ t₃.sides ∧ s ∈ t₄.sides)) ∧ 
  (∀ t₃ t₄ ∈ T₂, (t₃ = t₄) ∨ (∃ p : Point, p ∈ t₃.vert ∧ p ∈ t₄.vert) ∨ (∃ s : Side, s ∈ t₃.sides ∧ s ∈ t₄.sides)) :=
sorry

end large_triangle_subdivision_5_large_triangle_subdivision_3_l26_26904


namespace train_cross_post_time_l26_26567

-- Define the speed in km/hr and the length of the train in meters
def speed_km_hr : ℝ := 27
def length_meter : ℝ := 150

-- Convert the speed from km/hr to m/s
def speed_m_s : ℝ := speed_km_hr * (1000 / 3600)

-- Define the expected time in seconds
def expected_time : ℝ := 20

-- Prove that the time taken to cross the post is equal to the expected time (20 seconds)
theorem train_cross_post_time : length_meter / speed_m_s = expected_time := 
by 
  -- fill_by sorry since we're not providing the proof here
  sorry

end train_cross_post_time_l26_26567


namespace solve_for_z_l26_26738

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26738


namespace solution_set_eq_l26_26131

theorem solution_set_eq {x : ℝ} :
  (4^x - 3 * 2^(x + 1) + 8 = 0) ↔ (x = 1 ∨ x = 2) :=
sorry

end solution_set_eq_l26_26131


namespace black_to_white_ratio_l26_26630

noncomputable def area (r : ℝ) : ℝ := π * r^2

-- Radii of the concentric circles
def r1 : ℝ := 2
def r2 : ℝ := 4
def r3 : ℝ := 6
def r4 : ℝ := 8

-- Areas of the circles
def area1 : ℝ := area r1
def area2 : ℝ := area r2
def area3 : ℝ := area r3
def area4 : ℝ := area r4

-- Areas of the rings
def area_black1 : ℝ := area2 - area1
def area_white1 : ℝ := area3 - area2
def area_black2 : ℝ := area4 - area3

-- Total black area and white area
def total_black_area : ℝ := area_black1 + area_black2
def total_white_area : ℝ := area1 + area_white1

-- Ratio of the black area to the white area
theorem black_to_white_ratio : total_black_area / total_white_area = 5 / 3 :=
by sorry

end black_to_white_ratio_l26_26630


namespace solve_for_z_l26_26716

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26716


namespace solution_set_contains_0_and_2_l26_26343

theorem solution_set_contains_0_and_2 (k : ℝ) : 
  ∀ x, ((1 + k^2) * x ≤ k^4 + 4) → (x = 0 ∨ x = 2) :=
by {
  sorry -- Proof is omitted
}

end solution_set_contains_0_and_2_l26_26343


namespace product_of_digits_of_nondivisible_by_5_number_is_30_l26_26215

-- Define the four-digit numbers
def numbers : List ℕ := [4825, 4835, 4845, 4855, 4865]

-- Define units and tens digit function
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

-- Assertion that 4865 is the number that is not divisible by 5
def not_divisible_by_5 (n : ℕ) : Prop := ¬ (units_digit n = 5 ∨ units_digit n = 0)

-- Lean 4 statement to prove the product of units and tens digit of the number not divisible by 5 is 30
theorem product_of_digits_of_nondivisible_by_5_number_is_30 :
  ∃ n ∈ numbers, not_divisible_by_5 n ∧ (units_digit n) * (tens_digit n) = 30 :=
by
  sorry

end product_of_digits_of_nondivisible_by_5_number_is_30_l26_26215


namespace quadratic_real_roots_l26_26302

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end quadratic_real_roots_l26_26302


namespace complex_addition_l26_26403

namespace ComplexProof

def B := (3 : ℂ) + (2 * Complex.I)
def Q := (-5 : ℂ)
def R := (2 * Complex.I)
def T := (3 : ℂ) + (5 * Complex.I)

theorem complex_addition :
  B - Q + R + T = (1 : ℂ) + (9 * Complex.I) := 
by
  sorry

end ComplexProof

end complex_addition_l26_26403


namespace problem_part1_problem_part2_l26_26425

open Set

variables (m : ℝ)

def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def C : Set ℝ := {x : ℝ | m ≤ x}

theorem problem_part1 :
  ((compl A) ∩ B) = { x : ℝ | 1 < x ∧ x ≤ 2 } := sorry

theorem problem_part2 :
  ((A ∪ B) ∩ C) ≠ ∅ → m ≤ 3 := sorry

end problem_part1_problem_part2_l26_26425


namespace find_sum_of_coefficients_l26_26884

theorem find_sum_of_coefficients (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -(1/2) ∨ x > 1/3)) :
  a + b = -14 := 
sorry

end find_sum_of_coefficients_l26_26884


namespace solve_for_z_l26_26743

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26743


namespace range_of_m_l26_26287

open set

variable {x m : ℝ}

def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem range_of_m (h0 : ∃ x, p x) (h1 : ∀ x, p x → q x m) (h2 : ∃ x, ¬p x ∧ q x m) : 9 ≤ m :=
by sorry

end range_of_m_l26_26287


namespace complex_solution_l26_26726

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26726


namespace total_books_l26_26779

-- Define the number of books Stu has
def Stu_books : ℕ := 9

-- Define the multiplier for Albert's books
def Albert_multiplier : ℕ := 4

-- Define the number of books Albert has
def Albert_books : ℕ := Albert_multiplier * Stu_books

-- Prove that the total number of books is 45
theorem total_books:
  Stu_books + Albert_books = 45 :=
by 
  -- This is where the proof steps would go, but we skip it for now 
  sorry

end total_books_l26_26779


namespace ratio_AB_BC_l26_26014

-- Define the conditions related to the rectangles
def identical_rectangles (width length : ℝ) : Prop :=
  ∀ (i j : ℕ), 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → 
    width = width ∧ length = length

-- Given conditions
variable (x : ℝ) (h : identical_rectangles x (3 * x))

-- Define lengths AB and BC
def AB (x : ℝ) : ℝ := x + x + (3 * x)
def BC (x : ℝ) : ℝ := 3 * x

-- Prove the required ratio
theorem ratio_AB_BC (x : ℝ) (h : identical_rectangles x (3 * x)) :
  AB x / BC x = 5 / 3 := by
  rw [AB, BC]
  simp
  sorry

end ratio_AB_BC_l26_26014


namespace fei_ren_diameter_scientific_notation_l26_26809

theorem fei_ren_diameter_scientific_notation
  (hair_diameter : ℝ)
  (h_hair_diameter : hair_diameter = 0.0009)
  (fei_ren_factor : ℝ)
  (h_fei_ren_factor : fei_ren_factor = 1 / 10)
  (fei_ren_diameter : ℝ)
  (h_fei_ren_diameter : fei_ren_diameter = hair_diameter * fei_ren_factor) :
  fei_ren_diameter = 9 * 10^(-5) :=
by {
  rw [h_hair_diameter, h_fei_ren_factor, h_fei_ren_diameter],
  -- proof steps would go here
  sorry
}

end fei_ren_diameter_scientific_notation_l26_26809


namespace investment_in_mutual_funds_l26_26392

theorem investment_in_mutual_funds (total_investment : ℝ) (ratio : ℝ) (investment_in_mutual_funds : ℝ) : 
  total_investment = 240_000 → ratio = 6 → investment_in_mutual_funds = (240_000 * 6) / 7 → 
  investment_in_mutual_funds = 205_714.29 :=
by
  intro h1 h2 h3
  have h4 : 240_000 * 6 / 7 = 205_714.29 := sorry -- This needs a proof
  rw [h4] at h3
  exact h3

end investment_in_mutual_funds_l26_26392


namespace congruent_triangles_equal_perimeters_l26_26937

-- Define congruent triangles
def congruent_triangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1 ≅ Δ2

-- Define the statement to be proved 
theorem congruent_triangles_equal_perimeters (Δ1 Δ2 : Triangle)
  (h : congruent_triangles Δ1 Δ2) : 
  Δ1.perimeter = Δ2.perimeter :=
sorry

end congruent_triangles_equal_perimeters_l26_26937


namespace adidas_cost_l26_26989

theorem adidas_cost (p_Nike p_Reebok n_Nike n_Adidas n_Reebok target total revenue: ℕ) 
  (h1 : p_Nike = 60) (h2 : p_Reebok = 35) (h3 : n_Nike = 8) (h4 : n_Adidas = 6) 
  (h5 : n_Reebok = 9) (h6 : target = 1000) (h7 : revenue = 1065) :
  6 * (revenue - (n_Nike * p_Nike + n_Reebok * p_Reebok)) = 270 :=
by 
  simp [h1, h2, h3, h4, h5, h6, h7]
  sorry

end adidas_cost_l26_26989


namespace average_death_rate_l26_26794

-- Definitions of the given conditions
def birth_rate_two_seconds := 10
def net_increase_one_day := 345600
def seconds_per_day := 24 * 60 * 60 

-- Define the theorem to be proven
theorem average_death_rate :
  (birth_rate_two_seconds / 2) - (net_increase_one_day / seconds_per_day) = 1 :=
by 
  sorry

end average_death_rate_l26_26794


namespace cricket_count_l26_26168

theorem cricket_count (x : ℕ) (h : x + 11 = 18) : x = 7 :=
by sorry

end cricket_count_l26_26168


namespace quadratic_real_roots_l26_26301

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end quadratic_real_roots_l26_26301


namespace hypotenuse_of_diagonal_square_l26_26565

theorem hypotenuse_of_diagonal_square (a : ℝ) (h₁ : a = 10) :
  (∃ c : ℝ, c = 10 * Real.sqrt 2) :=
by
  use 10 * Real.sqrt 2
  sorry

end hypotenuse_of_diagonal_square_l26_26565


namespace points_five_units_away_from_neg_three_l26_26076

theorem points_five_units_away_from_neg_three (x : ℤ) : abs(x + 3) = 5 ↔ x = -8 ∨ x = 2 := by
  sorry

end points_five_units_away_from_neg_three_l26_26076


namespace vector_magnitude_l26_26324

-- Definition of vectors and their properties.
variables {α : Type*} [inner_product_space ℝ α] (a b : α)

-- Given conditions.
def condition_1 : ∥a∥ = 1 := sorry
def condition_2 : ∥b∥ = 2 := sorry
def condition_3 : ∥a + b∥ = ∥a - b∥ := sorry

-- Required proof statement.
theorem vector_magnitude : ∥(2 : ℝ) • a - b∥ = 2 * real.sqrt 2 :=
by sorry

end vector_magnitude_l26_26324


namespace length_BD_eq_8_l26_26016

variable (A B C D E : Point)

noncomputable def midpoint (p₁ p₂ : Point) : Point := sorry
noncomputable def bisect_angle (p₁ p₂ p₃ p₄ : Point) : Prop := sorry

def length (p1 p2 : Point) : ℝ := sorry

axiom AB_AC_eq_13 : length A B = 13 ∧ length A C = 13
axiom AB_AC_eq : length A B = length A C
axiom midpoint_D : D = midpoint A C
axiom E_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = A + t • (B - A)
axiom bisect_DE : bisect_angle A D C E
axiom AE_eq_5 : length A E = 5

theorem length_BD_eq_8 :
  length B D = 8 :=
by sorry

end length_BD_eq_8_l26_26016


namespace complex_eq_l26_26633

theorem complex_eq (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a + 2 * i) / i = b + i) : a + b = 1 :=
sorry

end complex_eq_l26_26633


namespace quadratic_function_characterization_l26_26281

variable (f : ℝ → ℝ)

def quadratic_function_satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 2) ∧ (∀ x, f (x + 1) - f x = 2 * x - 1)

theorem quadratic_function_characterization
  (hf : quadratic_function_satisfies_conditions f) : 
  (∀ x, f x = x^2 - 2 * x + 2) ∧ 
  (f (-1) = 5) ∧ 
  (f 1 = 1) ∧ 
  (f 2 = 2) := by
sorry

end quadratic_function_characterization_l26_26281


namespace compare_full_marks_l26_26802

variable (full_marks_A full_marks_B total_students_A total_students_B : ℕ)

-- Conditions
def condition_A (hA : total_students_A > 0) : full_marks_A = 0.01 * total_students_A :=
by sorry

def condition_B (hB : total_students_B > 0) : full_marks_B = 0.02 * total_students_B :=
by sorry

-- Proposition to prove
theorem compare_full_marks (hA : total_students_A > 0) (hB : total_students_B > 0) :
  full_marks_A = 0.01 * total_students_A → full_marks_B = 0.02 * total_students_B → 
  (full_marks_A = full_marks_B ∨ full_marks_A < full_marks_B ∨ full_marks_A > full_marks_B) → False :=
by sorry

end compare_full_marks_l26_26802


namespace find_z_l26_26756

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26756


namespace solve_for_z_l26_26742

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26742


namespace evaluate_expression_l26_26048

variable {R : Type*} [LinearOrderedField R]

def roots_of_cubic (p q r : R) (a b c : R) :=
  a + b + c = p ∧ a * b + b * c + c * a = q ∧ a * b * c = r

theorem evaluate_expression (a b c : R) 
  (h : roots_of_cubic 15 22 8 a b c) : 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 181 / 9) :=
by
  cases h with h_sum h_product;
  cases h_product with h_ab_bc_ca h_abc;
  sorry

end evaluate_expression_l26_26048


namespace present_value_amount_l26_26257

def yearly_increase (amount : ℝ) : ℝ :=
  amount * 9 / 8

theorem present_value_amount (A_2 : ℝ) (P : ℝ)
  (h1 : yearly_increase (yearly_increase P) = A_2)
  (h2 : A_2 = 4050) :
  P = 3200 :=
by
  sorry

end present_value_amount_l26_26257


namespace totalStudents_l26_26460

-- Define the number of seats per ride
def seatsPerRide : ℕ := 15

-- Define the number of empty seats per ride
def emptySeatsPerRide : ℕ := 3

-- Define the number of rides taken
def ridesTaken : ℕ := 18

-- Define the number of students per ride
def studentsPerRide (seats : ℕ) (empty : ℕ) : ℕ := seats - empty

-- Calculate the total number of students
theorem totalStudents : studentsPerRide seatsPerRide emptySeatsPerRide * ridesTaken = 216 :=
by
  sorry

end totalStudents_l26_26460


namespace max_a_plus_b_l26_26781

theorem max_a_plus_b (z : ℂ) (a b : ℝ) (hz : |z| = 1) (hzb : z^2 = a + b * I) : 
  a + b ≤ sqrt 2 :=
sorry

end max_a_plus_b_l26_26781


namespace recurring_decimal_fraction_l26_26913

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l26_26913


namespace correct_propositions_count_l26_26638

-- Definitions of different lines and planes
variables {α β γ : Plane} {m n : Line}

-- Propositions
def proposition1 : Prop := (α ⊥ γ ∧ β ⊥ γ) → α ∥ β
def proposition2 : Prop := (n ⊥ α ∧ n ⊥ β) → α ∥ β
def proposition3 : Prop := (α ⊥ β ∧ m ∈ α) → m ⊥ β
def proposition4 : Prop := (m ∥ α ∧ n ∥ α) → m ∥ n

-- Proof that there is exactly 1 correct proposition
theorem correct_propositions_count :
  (proposition1 = false) ∧
  (proposition2 = true) ∧
  (proposition3 = false) ∧
  (proposition4 = false) →
  (1 = 1) :=
by
  intros,
  sorry

end correct_propositions_count_l26_26638


namespace height_percentage_difference_l26_26988

theorem height_percentage_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.5384615384615385) :
  (H_B - H_A) / H_B * 100 = 35 := 
sorry

end height_percentage_difference_l26_26988


namespace tv_set_price_l26_26190

theorem tv_set_price (P : ℝ) 
  (h1 : 20 = 20) -- Representing the 20 installments condition
  (h2 : 1200 > 0) : -- Representing the installment amount condition
  ∀ i : ℝ, (0.06 = 6/100 ∧
  (i = (P - 1200) + ((P - 1200)/2) * 0.06) ∧
  (1200 ≤ 10800) ∧ (i = 10800)) → 
  P = 11686.41 :=
begin
  sorry -- The proof itself is not required; only the statement
end

end tv_set_price_l26_26190


namespace min_value_proof_l26_26823

noncomputable def min_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 1/a + 1/b + 1/c = 9) : ℝ :=
  108 * a^2 * b^3 * c

theorem min_value_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 1/a + 1/b + 1/c = 9) : 
  min_value a b c h1 h2 h3 h4 ≥ 1 :=
by {
  sorry,
}

end min_value_proof_l26_26823


namespace find_divisor_l26_26973

variable {N : ℤ} (k q : ℤ) {D : ℤ}

theorem find_divisor (h1 : N = 158 * k + 50) (h2 : N = D * q + 13) (h3 : D > 13) (h4 : D < 158) :
  D = 37 :=
by 
  sorry

end find_divisor_l26_26973


namespace max_checkers_attacked_l26_26924

def is_adjacent (i j i' j' : ℕ) : Prop :=
  (i' = i + 1 ∧ j' = j) ∨ (i' = i - 1 ∧ j' = j) ∨ (i' = i ∧ j' = j + 1) ∨ (i' = i ∧ j' = j - 1) ∨
  (i' = i + 1 ∧ j' = j + 1) ∨ (i' = i + 1 ∧ j' = j - 1) ∨ (i' = i - 1 ∧ j' = j + 1) ∨ (i' = i - 1 ∧ j' = j - 1)

def is_attacked (P : Finset (ℕ × ℕ)) (i j : ℕ) : Prop :=
  ∃ (i' j' : ℕ), (i', j') ∈ P ∧ is_adjacent i j i' j'

theorem max_checkers_attacked (P : Finset (ℕ × ℕ)) :
  (∀ (i j : ℕ), (i, j) ∈ P → is_attacked P i j) → P.card ≤ 32 := sorry

end max_checkers_attacked_l26_26924


namespace simplify_sqrt_1_7_9_simplify_cbrt_neg3_cubed_l26_26852

-- Definitions of the given expressions as conditions
def expr1 := Real.sqrt (16 / 9)
def expr2 := Real.cbrt ((-3) ^ 3)

-- Theorem statements
theorem simplify_sqrt_1_7_9 : expr1 = 4 / 3 := sorry

theorem simplify_cbrt_neg3_cubed : expr2 = -3 := sorry

end simplify_sqrt_1_7_9_simplify_cbrt_neg3_cubed_l26_26852


namespace trajectory_moving_circle_l26_26929

theorem trajectory_moving_circle : 
  (∃ P : ℝ × ℝ, (∃ r : ℝ, (P.1 + 1)^2 = r^2 ∧ (P.1 - 2)^2 + P.2^2 = (r + 1)^2) ∧
  P.2^2 = 8 * P.1) :=
sorry

end trajectory_moving_circle_l26_26929


namespace evaluate_expression_l26_26046

variable {R : Type*} [LinearOrderedField R]

def roots_of_cubic (p q r : R) (a b c : R) :=
  a + b + c = p ∧ a * b + b * c + c * a = q ∧ a * b * c = r

theorem evaluate_expression (a b c : R) 
  (h : roots_of_cubic 15 22 8 a b c) : 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 181 / 9) :=
by
  cases h with h_sum h_product;
  cases h_product with h_ab_bc_ca h_abc;
  sorry

end evaluate_expression_l26_26046


namespace complex_number_quadrant_l26_26480

theorem complex_number_quadrant :
  let z := (2 - (complex.i)) / (3 * (complex.i) - 1)
  (-1 / 2) - (1 / 2) * (complex.i) = z ∧ (z.re < 0 ∧ z.im < 0) := 
  sorry

end complex_number_quadrant_l26_26480


namespace trapezoid_integer_part_l26_26436

theorem trapezoid_integer_part (b h h1 x : ℝ) (base1 base2 : ℝ) 
  (h_base_lengths : base1 = b ∧ base2 = b + 150)
  (h_segment_midpoints_ratio : ((b + 75) * h / 2) / ((b + 75 + b + 150) * h / 2) = 3 / 4)
  (h_segment_equal_area : x = (300 * h / (2 * (2 * base1 + 150 * h1 / h1)) - 75)) :
  int.floor (x^2 / 150) = 37 := 
sorry

end trapezoid_integer_part_l26_26436


namespace max_t_value_l26_26835

theorem max_t_value {r n : ℕ} (h1 : 2 ≤ r) (h2 : r < n / 2) :
  ∃ (t : ℕ), t = (finset.card (finset.range (n-1)).choose (r-1)) :=
begin
  let t := (finset.card (finset.range (n-1)).choose (r-1)),
  use t,
  -- Proof needed here.
  sorry
end

end max_t_value_l26_26835


namespace evaluate_x2_plus_y2_l26_26653

theorem evaluate_x2_plus_y2 (x y : ℝ) (h₁ : 3 * x + 2 * y = 20) (h₂ : 4 * x + 2 * y = 26) : x^2 + y^2 = 37 := by
  sorry

end evaluate_x2_plus_y2_l26_26653


namespace sum_cardinality_union_l26_26059

def S : Finset ℕ := (Finset.range 2005).map ⟨Nat.succ_pnat, Nat.succ_pnat_injective⟩

noncomputable def F : Finset (Finset ℕ × Finset ℕ × Finset ℕ × Finset ℕ) :=
  Finset.pi_finset (Finset.replicate 4 S)

theorem sum_cardinality_union :
  Finset.sum F (λ x, (x.1 ∪ x.2 ∪ x.3 ∪ x.4).card) = 2^8016 * 2005 * 15 :=
sorry

end sum_cardinality_union_l26_26059


namespace complex_solution_l26_26718

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26718


namespace problem_I_solution_set_problem_II_range_m_l26_26312

open Real

-- Define the function f according to the problem
def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

-- Problem Ⅰ: Prove the subset solution when m = 1
theorem problem_I_solution_set (x : ℝ) :
  m = 1 → f x m ≥ 1 ↔ x ≤ -3/2 := by
  intros m_eq
  subst m_eq
  sorry

-- Problem Ⅱ: Prove the range of values for m that satisfies the inequality
theorem problem_II_range_m (m x t : ℝ) :
  0 < m ∧ m < 3/4 → f x m < |2 + t| + |t - 1| := by
  intros h
  sorry


end problem_I_solution_set_problem_II_range_m_l26_26312


namespace chef_makes_10_cakes_l26_26104

def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

theorem chef_makes_10_cakes :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 := by
  sorry

end chef_makes_10_cakes_l26_26104


namespace solve_problem_l26_26332

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (0.65 * x > 26) ∧ (0.40 * y < -3) ∧ ((x - y)^2 ≥ 100) 
  → (x > 40) ∧ (y < -7.5)

theorem solve_problem (x y : ℝ) (h : proof_problem x y) : (x > 40) ∧ (y < -7.5) := 
sorry

end solve_problem_l26_26332


namespace John_is_26_l26_26020

-- Define the variables representing the ages
def John_age : ℕ := 26
def Grandmother_age : ℕ := John_age + 48

-- Conditions
def condition1 : Prop := John_age = Grandmother_age - 48
def condition2 : Prop := John_age + Grandmother_age = 100

-- Main theorem to prove: John is 26 years old
theorem John_is_26 : John_age = 26 :=
by
  have h1 : condition1 := by sorry
  have h2 : condition2 := by sorry
  -- More steps to combine the conditions and prove the theorem would go here
  -- Skipping proof steps with sorry for demonstration
  sorry

end John_is_26_l26_26020


namespace lines_coplanar_iff_k_eq_neg2_l26_26438

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
(2 + s, 4 - k * s, 2 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
(t, 2 + 2 * t, 3 - t)

theorem lines_coplanar_iff_k_eq_neg2 :
  (∃ s t : ℝ, line1 s k = line2 t) → k = -2 :=
by
  sorry

end lines_coplanar_iff_k_eq_neg2_l26_26438


namespace infinite_primes_in_S_l26_26034

def S : Set ℚ := {q | ∃ (n : ℕ) (a b : Fin n → ℕ), 
  q = (∏ i, (a i ^ 2 + a i - 1)) / (∏ i, (b i ^ 2 + b i - 1)) }

theorem infinite_primes_in_S : ∃ (P : Set ℕ), P ⊆ {p : ℕ | nat.prime p} ∧ P ⊆ S ∧ P.infinite :=
sorry

end infinite_primes_in_S_l26_26034


namespace binomial_coeff_arithmetic_seq_l26_26339

theorem binomial_coeff_arithmetic_seq (n : ℕ) (x : ℝ) (h : ∀ (a b c : ℝ), a = 1 ∧ b = n/2 ∧ c = n*(n-1)/8 → (b - a) = (c - b)) : n = 8 :=
sorry

end binomial_coeff_arithmetic_seq_l26_26339


namespace bottles_meet_requirements_bottle_closest_to_specified_l26_26961

def deviations : List ℝ := [+0.0018, -0.0023, +0.0025, -0.0015, +0.0012, -0.0009]

def margin_of_error : ℝ := 0.002

theorem bottles_meet_requirements :
  {i : Fin 6 | abs (deviations[i]) < margin_of_error} = {0, 3, 4, 5} := 
sorry

theorem bottle_closest_to_specified :
  ∃ i : Fin 6, (∀ j : Fin 6, abs (deviations[i]) ≤ abs (deviations[j])) ∧ i = 5 := 
sorry

end bottles_meet_requirements_bottle_closest_to_specified_l26_26961


namespace triangle_area_proof_l26_26411

def vector2 := ℝ × ℝ

def a : vector2 := (6, 3)
def b : vector2 := (-4, 5)

noncomputable def det (u v : vector2) : ℝ := u.1 * v.2 - u.2 * v.1

noncomputable def parallelogram_area (u v : vector2) : ℝ := |det u v|

noncomputable def triangle_area (u v : vector2) : ℝ := parallelogram_area u v / 2

theorem triangle_area_proof : triangle_area a b = 21 := 
by 
  sorry

end triangle_area_proof_l26_26411


namespace find_x_value_l26_26055

theorem find_x_value (x : ℚ) (h1 : 9 * x ^ 2 + 8 * x - 1 = 0) (h2 : 27 * x ^ 2 + 65 * x - 8 = 0) : x = 1 / 9 :=
sorry

end find_x_value_l26_26055


namespace polynomial_possible_integer_roots_l26_26559

theorem polynomial_possible_integer_roots (b1 b2 : ℤ) :
  ∀ x : ℤ, (x ∣ 18) ↔ (x^3 + b2 * x^2 + b1 * x + 18 = 0) → 
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
by {
  sorry
}


end polynomial_possible_integer_roots_l26_26559


namespace both_firms_participate_number_of_firms_participate_social_optimality_l26_26539

-- Definitions for general conditions
variable (α V IC : ℝ)
variable (hα : 0 < α ∧ α < 1)

-- Condition for both firms to participate
def condition_to_participate (V : ℝ) (α : ℝ) (IC : ℝ) : Prop :=
  V * α * (1 - 0.5 * α) ≥ IC

-- Part (a): Under what conditions will both firms participate?
theorem both_firms_participate (α V IC : ℝ) (hα : 0 < α ∧ α < 1) :
  condition_to_participate V α IC → (V * α * (1 - 0.5 * α) ≥ IC) :=
by sorry

-- Part (b): Given V=16, α=0.5, and IC=5, determine the number of firms participating
theorem number_of_firms_participate :
  (condition_to_participate 16 0.5 5) :=
by sorry

-- Part (c): To determine if the number of participating firms is socially optimal
def total_profit (α V IC : ℝ) (both : Bool) :=
  if both then 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
  else α * V - IC

theorem social_optimality :
   (total_profit 0.5 16 5 true ≠ max (total_profit 0.5 16 5 true) (total_profit 0.5 16 5 false)) :=
by sorry

end both_firms_participate_number_of_firms_participate_social_optimality_l26_26539


namespace transformed_function_eq_l26_26095

noncomputable def stretch_and_shift (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ :=
  λ x, f (a * x - b)

def original_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 4)

def transformed_function (x : ℝ) : ℝ := stretch_and_shift original_function (1/2) (Real.pi / 6)

theorem transformed_function_eq : transformed_function = λ x, Real.sin (1 / 2 * x - Real.pi / 3) :=
by
  sorry

end transformed_function_eq_l26_26095


namespace sixth_valid_sample_number_l26_26550

theorem sixth_valid_sample_number (parts : Finset ℕ) (samples : List ℕ) :
  (∀ n ∈ parts, n ≥ 1 ∧ n ≤ 800) ∧
  (∀ n ∈ samples.drop 29, n ∈ parts ∧ (samples.filter (λ x, x ∈ parts)).nodup) ∧
  samples.drop 29.get 5 = 328 :=
begin
  sorry
end

end sixth_valid_sample_number_l26_26550


namespace area_circle_correct_l26_26608

def radius : ℝ := 7

def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem area_circle_correct :
  area_of_circle radius ≈ 153.93804 := sorry

end area_circle_correct_l26_26608


namespace sequence_geometric_l26_26374

theorem sequence_geometric (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = r * a n) (h2 : a 4 = 2) : a 2 * a 6 = 4 :=
by
  sorry

end sequence_geometric_l26_26374


namespace union_and_intersection_range_of_a_l26_26057

open Set

variable (a : ℝ)

-- Define sets A, B, and C
def A := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B := {x : ℝ | 2 < x ∧ x < 10}
def C := {x : ℝ | a < x ∧ x < a + 2}

-- First part: prove A ∪ B and A ∩ B
theorem union_and_intersection :
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧ (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) :=
sorry

-- Second part: prove range of a given C ⊆ A ∪ B
theorem range_of_a (h : C ⊆ A ∪ B) : 2 ≤ a ∧ a ≤ 8 :=
sorry

end union_and_intersection_range_of_a_l26_26057


namespace mean_of_remaining_students_l26_26879

theorem mean_of_remaining_students (n : ℕ) (h1 : 15 < n) (h2 : ( ∑ i in (range n), (scores i) ) / n = 10) 
                                  (h3 : ( ∑ i in (range 15), (scores i) ) / 15 = 16) :
  ( ∑ i in (range (n - 15), (scores (i + 15)) ) / (n - 15) = (10 * n - 240) / (n - 15) :=
by {
  -- Assume the score function exists mentioning the scores of students and transformations
  sorry
}

end mean_of_remaining_students_l26_26879


namespace average_death_rate_l26_26353

def birth_rate := 4 -- people every 2 seconds
def net_increase_per_day := 43200 -- people

def seconds_per_day := 86400 -- 24 * 60 * 60

def net_increase_per_second := net_increase_per_day / seconds_per_day -- people per second

def death_rate := (birth_rate / 2) - net_increase_per_second -- people per second

theorem average_death_rate :
  death_rate * 2 = 3 := by
  -- proof is omitted
  sorry

end average_death_rate_l26_26353


namespace division_of_repeating_decimals_l26_26909

noncomputable def repeating_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  ⟨n, d⟩

theorem division_of_repeating_decimals :
  let x := repeating_to_fraction 54 99
  let y := repeating_to_fraction 18 99
  (x / y) = (3 : ℚ) :=
by
  -- Proof omitted as requested
  sorry

end division_of_repeating_decimals_l26_26909


namespace digits_1_left_of_6_count_l26_26870

theorem digits_1_left_of_6_count :
  let digits := {1, 2, 3, 4, 5, 6}
  let is_six_digit_unique (n : ℕ) : Prop := 
    ∃ l : List ℕ, l.nodup ∧ (l.perm digits.to_list) ∧ (1 ∈ l) ∧ (6 ∈ l) ∧
    (n = l.foldl (λ acc d, acc * 10 + d) 0)
  let count_1_left_of_6 (l : List ℕ) : Prop := list_index l 1 < list_index l 6
  (number_of_six_digit_integers_with_property digits is_six_digit_unique count_1_left_of_6 = 360) :=
begin
  sorry
end

end digits_1_left_of_6_count_l26_26870


namespace infinite_power_tower_l26_26470

theorem infinite_power_tower (x : ℝ) (h : x ^ (x ^ (x ^ ...)) = 4) : x = real.sqrt 2 :=
by
  sorry

end infinite_power_tower_l26_26470


namespace quadrilateral_with_equal_sides_not_ne_planar_l26_26218

def is_planar_triangle : Prop := ∀ (A B C : Point), ¬ collinear A B C → ∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane 

def is_planar_trapezoid : Prop := ∀ (A B C D : Point), (parallel (line_through A B) (line_through C D) ∨ parallel (line_through A D) (line_through B C)) → ∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane

def is_planar_parallelogram : Prop := ∀ (A B C D : Point), parallel (line_through A B) (line_through C D) ∧ parallel (line_through B C) (line_through A D) → ∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane

theorem quadrilateral_with_equal_sides_not_ne_planar :
  (∃ (A B C D : Point), length (segment A B) = length (segment B C) ∧ length (segment B C) = length (segment C D) ∧ length (segment C D) = length (segment D A) ∧ length (segment D A) = length (segment A B) ∧ ¬∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane) := sorry

end quadrilateral_with_equal_sides_not_ne_planar_l26_26218


namespace sum_of_first_ten_terms_arithmetic_sequence_l26_26296

variable {a : ℕ → ℝ}
variable (h_arith : ∀ n m, a (n + 1) = a n + a m)
variable (h_cond1 : a 3 ^ 2 + a 8 ^ 2 + 2 * a 3 * a 8 = 9)
variable (h_cond2 : ∀ n, a n < 0)

theorem sum_of_first_ten_terms_arithmetic_sequence :
  S 10 = -15 :=
by
  sorry

end sum_of_first_ten_terms_arithmetic_sequence_l26_26296


namespace exists_colored_right_triangle_l26_26905

theorem exists_colored_right_triangle (color : ℝ × ℝ → ℕ) 
  (h_nonempty_blue  : ∃ p, color p = 0)
  (h_nonempty_green : ∃ p, color p = 1)
  (h_nonempty_red   : ∃ p, color p = 2) :
  ∃ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) ∧ (p2 ≠ p3) ∧ (p1 ≠ p3) ∧ 
    ((color p1 = 0) ∧ (color p2 = 1) ∧ (color p3 = 2) ∨ 
     (color p1 = 0) ∧ (color p2 = 2) ∧ (color p3 = 1) ∨ 
     (color p1 = 1) ∧ (color p2 = 0) ∧ (color p3 = 2) ∨ 
     (color p1 = 1) ∧ (color p2 = 2) ∧ (color p3 = 0) ∨ 
     (color p1 = 2) ∧ (color p2 = 0) ∧ (color p3 = 1) ∨ 
     (color p1 = 2) ∧ (color p2 = 1) ∧ (color p3 = 0))
  ∧ ((p1.1 = p2.1 ∧ p2.2 = p3.2) ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1)) :=
sorry

end exists_colored_right_triangle_l26_26905


namespace complex_number_solution_l26_26681

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26681


namespace total_weight_of_20_carrots_is_3_point_64_kg_l26_26954

-- Definitions of given conditions
def initial_carrots : ℕ := 20
def removed_carrots : ℕ := 4
def remaining_carrots := initial_carrots - removed_carrots -- 16 remaining carrots
def avg_weight_remaining_carrots : ℝ := 180 -- grams
def avg_weight_removed_carrots : ℝ := 190 -- grams

-- Required total weight of the 20 carrots in kilograms
def total_weight_of_carrots (ic rc: ℕ) (aw_rc aw_rm: ℝ) : ℝ :=
  let total_remaining_weight := (ic - rc) * aw_rm
  let total_removed_weight := rc * aw_rc
  let total_weight_in_grams := total_remaining_weight + total_removed_weight
  total_weight_in_grams / 1000

-- The statement we need to prove
theorem total_weight_of_20_carrots_is_3_point_64_kg :
  total_weight_of_carrots initial_carrots removed_carrots avg_weight_remaining_carrots avg_weight_removed_carrots = 3.64 := 
by 
  sorry

end total_weight_of_20_carrots_is_3_point_64_kg_l26_26954


namespace solution_set_eq_l26_26657

open Real Set

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 3*x else -x^2 - 3*x

theorem solution_set_eq :
    {x : ℝ | f(x) - x + 3 = 0} = {-2 - sqrt 7, 1, 3} := by sorry

end solution_set_eq_l26_26657


namespace general_formula_a_n_sum_first_n_b_n_l26_26427

-- Conditions
def common_difference (d : ℕ) : Prop := d = 2

def Sn (a1 a7 S7 : ℕ) : Prop := (7 * (a1 + a7) / 2 = S7)

def a5 (a5_val : ℕ) : Prop := a5_val = 9

def S7_val (S7 : ℕ) : Prop := S7 = 49

-- Proof problem
theorem general_formula_a_n (d a1 a5_val S7 : ℕ) (h_diff : common_difference d)
  (h_Sn : Sn a1 (a1 + 6 * d) S7) (h_a5: a5 a5_val) (h_S7: S7_val S7) :
  ∀ n : ℕ, 0 < n → (2 * n - 1) = (a1 + (n - 1) * d) :=
sorry

theorem sum_first_n_b_n (a b d a1 S7 : ℕ) (h_a_n : ∀ n : ℕ, 0 < n → (2 * n - 1) = (a1 + (n - 1) * d))
  (h_a5: a5 a) (h_S7: S7_val S7) :
  ∀ n : ℕ, 0 < n →
  ∑ i in Finset.range n, (a1 + i * d) * 2^(i+1) = (2 * n - 3) * 2^(n+1) + 6 :=
sorry

end general_formula_a_n_sum_first_n_b_n_l26_26427


namespace four_digit_number_with_conditions_l26_26976

theorem four_digit_number_with_conditions :
  ∃ n : ℕ,
    (nat.factors n).length = 8 ∧
    (nat.proper_divisors n).head = 15 ∧    -- assuming proper_divisors gives divisors in sorted order; modify as per correct API
    ∃ p q r : ℕ, (nat.factors n) = [p, q, r] ∧ (p - 5*q = 2*r) :=
sorry

end four_digit_number_with_conditions_l26_26976


namespace find_fraction_l26_26967

variable (x : ℝ) (f : ℝ)
axiom thirty_percent_of_x : 0.30 * x = 63.0000000000001
axiom fraction_condition : f = 0.40 * x + 12

theorem find_fraction : f = 96 := by
  sorry

end find_fraction_l26_26967


namespace value_of_a_l26_26665

theorem value_of_a :
  ∀ (m a c : ℝ),
    (x^2 + y^2 + 2 * m * x - 3 = 0) ∧ 
    (m < 0) ∧ 
    (radius = 2) ∧ 
    (ellipse := (x^2 / a^2 + y^2 / 3 = 1)) ∧ 
    (focus := -c) ∧ 
    (line_l ⟨l⟩ perpendicular to x-axis passes through (-c, 0) is tangent to (x^2 + y^2 + 2m * x - 3 = 0)) → 
  a = 2 :=
by
  sorry

end value_of_a_l26_26665


namespace tiles_visited_by_bug_l26_26977

theorem tiles_visited_by_bug :
  let width := 10
  let length := 17
  let gcd := Int.gcd width length
  number_of_tiles_visited := width + length - gcd
  number_of_tiles_visited = 26 :=
by
  let width := 10
  let length := 17
  let gcd := Int.gcd width length
  have number_of_tiles_visited := width + length - gcd
  exact (number_of_tiles_visited : Int)

-- Proof is omitted with sorry.
-- Please write the proof if required.
sorry

end tiles_visited_by_bug_l26_26977


namespace cyclic_quad_iff_angle_eq_l26_26648

-- Definitions and setup for the points and parameters
variables (A B C M P Q D E : Type)
variables [EuclideanGeometry A B C M P Q D E]
variable (h_midpoint : is_midpoint M A B)
variable (h_in_triangle : is_in_triangle P A B C)
variable (h_reflection : is_reflection P M Q)
variable (h_intersection_D : is_intersection_point (line_through A P) (side_through B C) D)
variable (h_intersection_E : is_intersection_point (line_through B P) (side_through A C) E)

-- Theorem statement
theorem cyclic_quad_iff_angle_eq :
  (is_cyclic A B D E) ↔ (angle A C P = angle Q C B) :=
by
  sorry

end cyclic_quad_iff_angle_eq_l26_26648


namespace Elberta_has_23_dollars_l26_26327

theorem Elberta_has_23_dollars (GrannySmith_has : ℕ := 72)
    (Anjou_has : ℕ := GrannySmith_has / 4)
    (Elberta_has : ℕ := Anjou_has + 5) : Elberta_has = 23 :=
by
  sorry

end Elberta_has_23_dollars_l26_26327


namespace area_of_triangle_PQR_l26_26153

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 2 }
def Q : Point := { x := 7, y := 2 }
def R : Point := { x := 5, y := 9 }

noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangleArea P Q R = 17.5 := by
  sorry

end area_of_triangle_PQR_l26_26153


namespace student_most_stable_l26_26145

theorem student_most_stable (A B C : ℝ) (hA : A = 0.024) (hB : B = 0.08) (hC : C = 0.015) : C < A ∧ C < B := by
  sorry

end student_most_stable_l26_26145


namespace locus_of_tangency_points_l26_26576

-- Defining the problem as a theorem in Lean

theorem locus_of_tangency_points
  (O : Point)
  (p1 p2 : Ray)
  (α : ℝ)
  (k1_center : ℝ -> Point)
  (k1_radius : ℝ -> ℝ)
  (k2_center : ℝ -> Point)
  (k2_radius : ℝ -> ℝ)
  (tangency_point : ℝ -> Point)
  (h_angle : ∀ t : ℝ, ∠ O p1 p2 = 2 * α)
  (h_k1_tangent : ∀ t : ℝ, tangent (k1_center t) (k1_radius t) p2)
  (h_k2_tangent : ∀ t : ℝ, tangent (k2_center t) (k2_radius t) p1)
  (h_k2_tangent_k1 : ∀ t : ℝ, tangent_circles (k2_center t) (k2_radius t) (k1_center t) (k1_radius t))
  (h_k1_moves : ∀ t : ℝ, k1_center t = (t * cos α, t * sin α)) :
  ∀ t : ℝ, tangency_point t = ((t * cos α), (t * sin α) * (sin(2 * α) / (cos(2 * α) + 2))) :=
sorry

end locus_of_tangency_points_l26_26576


namespace at_most_two_distinct_values_l26_26185

theorem at_most_two_distinct_values
  (a b c d : ℝ)
  (h1 : a + b = c + d)
  (h2 : a^2 + b^2 = c^2 + d^2) :
  ∃ x y, (x ≠ y ∧ {a, b, c, d} ⊆ {x, y} ∨ x = y ∧ {a, b, c, d} = {x}) :=
by
  sorry

end at_most_two_distinct_values_l26_26185


namespace rationalize_expression_l26_26851

theorem rationalize_expression :
  ( ∀ (x y z a b c : ℝ), 
      x = sqrt 3 ∧ y = sqrt 7 ∧ z = sqrt 5 ∧ a = sqrt 11 ∧ b = sqrt 6 ∧ c = sqrt 8 
      → (x / y * z / a * b / c) = 3 * sqrt 385 / 154) :=
begin
  rintros _ _ _ _ _ _ ⟨hx, hy, hz, ha, hb, hc⟩,
  sorry,
end

end rationalize_expression_l26_26851


namespace complex_number_quadrant_l26_26479

-- Define the conditions
def real_part : ℤ := -2
def imaginary_part : ℤ := 1

-- Define the statement to prove
theorem complex_number_quadrant :
    real_part < 0 ∧ imaginary_part > 0 → "Second Quadrant" :=
by
  intro h
  have h1 : real_part = -2 := rfl
  have h2 : imaginary_part = 1 := rfl
  sorry

end complex_number_quadrant_l26_26479


namespace cos_sum_identity_l26_26846

theorem cos_sum_identity :
  cos (Real.pi / 7) - cos (2 * Real.pi / 7) + cos (3 * Real.pi / 7) = 1 / 2 :=
by
  sorry

end cos_sum_identity_l26_26846


namespace Joe_spends_68_dollars_l26_26388

def Joe_spends_at_market
  (n_oranges : ℕ) (cost_orange : ℝ)
  (n_juices : ℕ) (cost_juice : ℝ)
  (n_honey : ℕ) (cost_honey : ℝ)
  (n_plants : ℕ) (cost_two_plants : ℝ) : ℝ :=
  let cost_plant := cost_two_plants / 2
  in (n_oranges * cost_orange) + (n_juices * cost_juice) + (n_honey * cost_honey) + (n_plants * cost_plant)

theorem Joe_spends_68_dollars :
  Joe_spends_at_market 3 4.5 7 0.5 3 5 4 18 = 68 := by
  sorry

end Joe_spends_68_dollars_l26_26388


namespace eccentricity_of_ellipse_l26_26654

variable (a b c : ℝ)

def ellipse_equation (x y : ℝ) : Prop :=
    (x^2) / (a^2) + (y^2) / (b^2) = 1

def point_on_ellipse (y : ℝ) : Prop :=
    let c := y / a
    (c^2) / (a^2) + (y^2) / (b^2) = 1

def perpendicular_condition (x : ℝ) : Prop :=
    x = 0

def isosceles_right_triangle (c y : ℝ) : Prop :=
    let A := (c, y)
    ∃ x2 y2 : ℝ, 
      perpendicular_condition x2 ∧ 
      ellipse_equation x2 y2 ∧ 
      c = y2 / a ∧ 
      ∃ e : ℝ,
      e = ↑(Real.sqrt 2)-1 ∧
      e = (Real.sqrt (a^2 - b^2)) / a ∧ 
      (e^2 + 2*e - 1 = 0)

theorem eccentricity_of_ellipse 
  (h_ellipse : ellipse_equation a b)
  (h_point : point_on_ellipse a b c)
  (h_perpendicular : perpendicular_condition c)
  (h_isosceles : isosceles_right_triangle a b c) : 
  ∃ e : ℝ, e = Real.sqrt 2 - 1 :=
sorry

end eccentricity_of_ellipse_l26_26654


namespace evaluation_of_expression_l26_26454

-- Define the conditions
def satisfies_conditions (x : Int) : Prop :=
  3 * x + 7 > 1 ∧ 2 * x - 1 < 5

-- Define the expression
def expression (x : Int) : Rat :=
  (x / (x - 1)) / ((x^2 - x) / (x^2 - 2 * x + 1)) - (x + 2) / (x + 1)

-- State the theorem
theorem evaluation_of_expression :
  ∀ x : Int, satisfies_conditions x → expression x ∈ {-1, -1/2, -1/3} ∨ (x = -1 ∧ (expression x).denom = 0) := 
by
  sorry

end evaluation_of_expression_l26_26454


namespace equilateral_triangle_l26_26378

theorem equilateral_triangle (a b c : ℝ) (h1 : a^4 = b^4 + c^4 - b^2 * c^2) (h2 : b^4 = a^4 + c^4 - a^2 * c^2) : 
  a = b ∧ b = c ∧ c = a :=
by sorry

end equilateral_triangle_l26_26378


namespace prism_faces_even_or_odd_l26_26522

theorem prism_faces_even_or_odd (n : ℕ) (hn : 3 ≤ n) : ¬ (2 + n) % 2 = 1 :=
by
  sorry

end prism_faces_even_or_odd_l26_26522


namespace g_inv_l26_26330

variable {X Y Z W : Type}
variable [InvertibleFunctions : inv₁ : Y → X, inv₂ : Z → Y, inv₃ : W → Z]

def p : X → Y := by assumption
def q : Y → Z := by assumption
def r : Z → W := by assumption
def s : W → X := by assumption
def g : X → W := r ∘ s ∘ p

theorem g_inv : g⁻¹ = inv₁ ∘ inv₂ ∘ inv₃ := by sorry

end g_inv_l26_26330


namespace circle_sequence_inequality_l26_26422

theorem circle_sequence_inequality {n : ℕ} (hn : n ≥ 4)
  (a : fin n → ℕ) (hpos : ∀ i, 0 < a i)
  (k : fin n → ℤ)
  (hdiv : ∀ i : fin n, (a (i - 1) + a (i + 1)) % a i = 0)
  (hki : ∀ i : fin n, k i = (a (i - 1) + a (i + 1)) / (a i)) :
  2 * n ≤ ∑ i, k i ∧ ∑ i, k i < 3 * n :=
by
  sorry

end circle_sequence_inequality_l26_26422


namespace oleg_can_choose_two_adjacent_cells_divisible_by_4_l26_26815

theorem oleg_can_choose_two_adjacent_cells_divisible_by_4 :
  ∀ (board : Fin 22 × Fin 22 → Fin (22*22)), 
  ∃ (i j : Fin 22) (di dj : Fin 3), 
    ─1 <= (di.val * di.val) <= 1 ∧
    ─1 <= (dj.val * dj.val) <= 1 ∧
    di ≠ 0 ∨ dj ≠ 0 ∧
    (board(i,j) + board((i+di) % 22, (j+dj) % 22)) % 4 = 0 := 
by
  sorry

end oleg_can_choose_two_adjacent_cells_divisible_by_4_l26_26815


namespace second_caterer_more_cost_effective_after_30_l26_26844

def first_caterer_charge (x : ℕ) : ℕ := 120 + 14 * x
def second_caterer_charge (x : ℕ) : ℕ := 210 + 11 * x

theorem second_caterer_more_cost_effective_after_30 :
  ∀ x : ℕ, x ≥ 31 → second_caterer_charge x < first_caterer_charge x :=
by
  intros x hx
  have h : 11 * x + 210 < 14 * x + 120 := sorry
  exact h

end second_caterer_more_cost_effective_after_30_l26_26844


namespace max_value_b_minus_c_plus_one_div_a_l26_26294

theorem max_value_b_minus_c_plus_one_div_a (a b c : ℝ) (h1 : ∀ x : ℝ, (x > -1 ∧ x < 3) → ax^2 + bx + c > 0) :
  let roots := {x : ℝ | x = -1 ∨ x = 3},
      ineq_holds := ∀ r ∈ roots, ax^2 + bx + c = 0,
      a_neg := a < 0,
      b := -2 * a,
      c := -3 * a
  in b - c + (1 / a) ≤ -2 :=
by
  intros
  sorry

end max_value_b_minus_c_plus_one_div_a_l26_26294


namespace solve_for_z_l26_26706

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26706


namespace solve_imaginary_eq_l26_26763

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26763


namespace product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l26_26151

-- Definition of even and odd numbers
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Theorem statements for each condition

-- Prove that the product of two even numbers is even
theorem product_of_two_even_numbers_is_even (a b : ℤ) :
  is_even a → is_even b → is_even (a * b) :=
by sorry

-- Prove that the product of two odd numbers is odd
theorem product_of_two_odd_numbers_is_odd (c d : ℤ) :
  is_odd c → is_odd d → is_odd (c * d) :=
by sorry

-- Prove that the product of one even and one odd number is even
theorem product_of_even_and_odd_number_is_even (e f : ℤ) :
  is_even e → is_odd f → is_even (e * f) :=
by sorry

-- Prove that the product of one odd and one even number is even
theorem product_of_odd_and_even_number_is_even (g h : ℤ) :
  is_odd g → is_even h → is_even (g * h) :=
by sorry

end product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l26_26151


namespace equal_angles_GFC_BAC_l26_26397

variable (Ω : Type) [CircumscribedCircle Ω ABC]
variables (A B C D E F G : Ω)
variable (acute_triangle : acute triangle ABC)
variable (D_on_small_arc_BC : point_on_small_arc BC D Ω)
variable (CDEF_parallelogram : parallelogram C D E F)
variable (G_on_small_arc_AC : point_on_small_arc AC G Ω)
variable (DC_parallel_BG : parallel DC BG)

theorem equal_angles_GFC_BAC :
  angle G F C = angle B A C := 
sorry

end equal_angles_GFC_BAC_l26_26397


namespace product_of_first_two_terms_l26_26110

-- Given parameters
variables (a d : ℤ) -- a is the first term, d is the common difference

-- Conditions
def fifth_term_condition (a d : ℤ) : Prop := a + 4 * d = 11
def common_difference_condition (d : ℤ) : Prop := d = 1

-- Main statement to prove
theorem product_of_first_two_terms (a d : ℤ) (h1 : fifth_term_condition a d) (h2 : common_difference_condition d) :
  a * (a + d) = 56 :=
by
  sorry

end product_of_first_two_terms_l26_26110


namespace custom_op_12_7_l26_26784

def custom_op (a b : ℤ) := (a + b) * (a - b)

theorem custom_op_12_7 : custom_op 12 7 = 95 := by
  sorry

end custom_op_12_7_l26_26784


namespace sum_of_greatest_divisor_digits_l26_26533

-- Let a be the GCD that satisfies the condition
def greatest_common_divisor (a b : Nat) : Nat := Nat.gcd a b

-- Define the conditions
def condition_1 : Nat := 4665 - 1305
def condition_2 : Nat := 6905 - 4665
def condition_3 : Nat := 6905 - 1305

def gcd_1120 : Nat := greatest_common_divisor (greatest_common_divisor condition_1 condition_2) condition_3

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

-- The greatest number that divides 1305, 4665, and 6905 leaving the same remainder is 1120
-- Prove that the sum of the digits of 1120 is 4
theorem sum_of_greatest_divisor_digits : sum_of_digits gcd_1120 = 4 :=
by
  -- Skipping the proof steps
  sorry

end sum_of_greatest_divisor_digits_l26_26533


namespace cup_of_coffee_price_l26_26795

def price_cheesecake : ℝ := 10
def price_set : ℝ := 12
def discount : ℝ := 0.75

theorem cup_of_coffee_price (C : ℝ) (h : price_set = discount * (C + price_cheesecake)) : C = 6 :=
by
  sorry

end cup_of_coffee_price_l26_26795


namespace log_eq_15_l26_26086

noncomputable def x : ℝ := 16 * Real.sqrt 2

theorem log_eq_15 : log 8 x + log 2 (x^3) = 15 :=
  sorry

end log_eq_15_l26_26086


namespace no_sqrt_negative_number_l26_26158

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end no_sqrt_negative_number_l26_26158


namespace czakler_inequality_l26_26398

variable {a b : ℕ} (ha : a > 0) (hb : b > 0)
variable {c : ℝ} (hc : c > 0)

theorem czakler_inequality (h : (a + 1 : ℝ) / (b + c) = b / a) : c ≥ 1 := by
  sorry

end czakler_inequality_l26_26398


namespace red_ball_probability_probability_one_red_one_white_l26_26008

open Classical

theorem red_ball_probability (total_balls : ℕ) (red_ball_freq : ℝ)
  (H_balls : total_balls = 5)
  (H_freq : red_ball_freq = 0.4) :
  (red_ball_prob : ℝ) :=
  red_ball_prob = red_ball_freq :=
  sorry

theorem probability_one_red_one_white (total_balls : ℕ) (red_ball_prob : ℝ)
  (H_balls : total_balls = 5)
  (H_prob : red_ball_prob = 0.4) :
  (prob_one_red_one_white : ℝ) :=
  prob_one_red_one_white = (3 / 5) :=
  sorry

end red_ball_probability_probability_one_red_one_white_l26_26008


namespace solve_imaginary_eq_l26_26768

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26768


namespace solid_projection_exists_l26_26139

noncomputable def exists_solid_with_projections (S : Type) : Prop :=
  ∃ (solid : S), ∀ (n : ℕ), n ≥ 3 → ∃ (proj : S → (Fin n → ℝ²)), is_convex (proj solid)

-- To state the property of a projection being convex; this would need to be defined accordingly,
-- here it's just a placeholder for the actual mathematical definition of convexity.
def is_convex (polygon : Fin n → ℝ²) : Prop :=
  sorry

theorem solid_projection_exists : ∃ S, exists_solid_with_projections S :=
  sorry

end solid_projection_exists_l26_26139


namespace complex_number_solution_l26_26683

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26683


namespace bernardo_larger_probability_l26_26231

-- Mathematical definitions
def bernardo_set : Finset ℕ := {1,2,3,4,5,6,7,8,10}
def silvia_set : Finset ℕ := {1,2,3,4,5,6}

-- Probability calculation function (you need to define the detailed implementation)
noncomputable def probability_bernardo_gt_silvia : ℚ := sorry

-- The proof statement
theorem bernardo_larger_probability : 
  probability_bernardo_gt_silvia = 13 / 20 :=
sorry

end bernardo_larger_probability_l26_26231


namespace f_2008_equals_cos_l26_26052

def f : ℕ → (ℝ → ℝ)
| 0 := λ x, Real.cos x
| (n + 1) := λ x, (f n)' x

theorem f_2008_equals_cos (x : ℝ) : f 2008 x = Real.cos x :=
sorry

end f_2008_equals_cos_l26_26052


namespace trigonometric_proof_l26_26770

theorem trigonometric_proof (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by sorry

end trigonometric_proof_l26_26770


namespace find_a_l26_26877

theorem find_a
  (A B : ℝ×ℝ)
  (a : ℝ)
  (h1 : ∀ x y : ℝ, (x + y = Real.sqrt 3 * a) → (x^2 + y^2 = a^2 + (a-1)^2))
  (h2 : ∀ x y : ℝ, (x = 0 ∧ y = 0) → (0,0) = (x,y))
  (h3 : ∀ A B : ℝ×ℝ, equilateral_triangle (0,0) A B) :
  a = 1/2 := 
sorry

end find_a_l26_26877


namespace perimeter_region_l26_26013

theorem perimeter_region (rectangle_height : ℕ) (height_eq_sixteen : rectangle_height = 16) (rect_area_eq : 12 * rectangle_height = 192) (total_area_eq : 12 * rectangle_height - 60 = 132):
  (rectangle_height + 12 + 4 + 6 + 10 * 2) = 54 :=
by
  have h1 : 12 * 16 = 192 := by sorry
  exact sorry


end perimeter_region_l26_26013


namespace length_of_bridge_l26_26984

open Real

noncomputable def train_length : ℝ := 300
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def bridge_crossing_time_sec : ℝ := 45

-- Conversion factor from km/h to m/s
noncomputable def kmh_to_ms (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Function to calculate distance based on speed and time
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Hypothesis:
-- 1. The train is 300 meters long.
-- 2. The train's speed is 60 km/h.
-- 3. The train crosses the bridge in 45 seconds.
-- We need to prove the length of the bridge is approximately 450.15 meters.

theorem length_of_bridge :
  let train_length := train_length,
      train_speed := kmh_to_ms train_speed_kmh,
      time := bridge_crossing_time_sec,
      total_distance := distance train_speed time
  in abs (total_distance - train_length - 450.15) < 10 :=
by
  -- Definitions and assumptions as per the problem statement
  let train_length := 300
  let train_speed := kmh_to_ms 60
  let time := 45
  let total_distance := distance train_speed time

  -- The actual Lean proof would go here
  sorry

end length_of_bridge_l26_26984


namespace volume_divided_by_pi_is_correct_l26_26552

-- Define the given conditions
def radius_original_circle : ℝ := 18
def sector_angle_degrees : ℕ := 270

-- Define the function to calculate the result when the volume of the cone is divided by π
noncomputable def result_when_volume_divided_by_pi : ℝ :=
  let r := (270 / 360) * 2 * radius_original_circle in
  let r_cone := r / (2 * π) in
  let slant_height := radius_original_circle in
  let height := sqrt (slant_height^2 - r_cone^2) in
  (1/3) * r_cone^2 * height

-- The correct answer to be proven
theorem volume_divided_by_pi_is_correct :
  result_when_volume_divided_by_pi = 60.75 * sqrt 141.75 :=
by
  -- this is the placeholder for the proof
  sorry

end volume_divided_by_pi_is_correct_l26_26552


namespace max_value_f_period_of_f_range_g_on_interval_l26_26674

noncomputable def m (x : ℝ) : ℝ × ℝ :=
  (Real.sin x, -1 / 2)

noncomputable def n (x : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos x, Real.cos (2 * x))

noncomputable def f (x : ℝ) : ℝ :=
  m x.fst * n x.fst + m x.snd * n x.snd

noncomputable def g (x : ℝ) : ℝ :=
  f (x + (Real.pi / 6))

theorem max_value_f : ∃ x : ℝ, f x = 1 := sorry

theorem period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x := sorry

theorem range_g_on_interval : Set.Icc 0 (Real.pi / 2) ⊆ Set.Icc (-1 / 2) 1 := sorry

end max_value_f_period_of_f_range_g_on_interval_l26_26674


namespace cube_volume_sphere_surface_area_l26_26788

theorem cube_volume_sphere_surface_area (V : ℝ) (hV : V = 8) : 
  let s := (V)^(1/3) in
  let d := (s^2 + s^2 + s^2)^(1/2) in
  let r := d / 2 in
  4 * Real.pi * r^2 = 12 * Real.pi :=
by
  have hs : s = (8 : ℝ)^(1/3) := by rw [hV]
  have hsd : s = 2 := by norm_num
  have hd : d = (2^2 + 2^2 + 2^2)^(1/2) := by rw [hsd]
  have hdr : d = 2 * (sqrt 3) := by rw [hd]
  have hr : r = sqrt 3 := by rw [hdr]
  rw [hr]
  sorry

end cube_volume_sphere_surface_area_l26_26788


namespace find_sin_alpha_calculate_expression_l26_26664

noncomputable def sin_alpha (α : ℝ) : ℝ := -3/5
noncomputable def cos_alpha (α : ℝ) : ℝ := 4/5

theorem find_sin_alpha :
  ∀ (α : ℝ), let P := (4/5 : ℝ, -3/5 : ℝ) in
  (P.1)^2 + (P.2)^2 = 1 → sin α = -3/5 :=
by
  let P := (4/5 : ℝ, -3/5 : ℝ)
  have OP_eq_1 : P.1 ^ 2 + P.2 ^ 2 = 1 := by
    calc (4/5 : ℝ) ^ 2 + (-3/5 : ℝ) ^ 2 = 16/25 + 9/25 : by ring
    ... = 25/25 : by ring
    ... = 1 : by norm_num
  intro α h
  exact eq.symm (sin_alpha α)

theorem calculate_expression :
  ∀ (α : ℝ), sin α = -3/5 → cos α = 4/5 →
  (sin (π / 2 - α) / sin (α + π)) * (tan (α - π) / cos (3 * π - α)) = 5 / 4 :=
by
  intro α h_sin h_cos
  let a := (sin (π / 2 - α) / sin (α + π)) * (tan (α - π) / cos (3 * π - α))
  have h_cos' : cos α = 4 / 5 := h_cos
  have h_sin' : sin α = -3 / 5 := h_sin
  sorry

end find_sin_alpha_calculate_expression_l26_26664


namespace find_z_l26_26753

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26753


namespace staircase_200_cells_l26_26188

-- Define the sequence L_n based on the problem
def L : ℕ → ℕ
| 1 := 2
| (n + 1) := (L n) + 1

-- Theorem statement to prove
theorem staircase_200_cells : L 200 = 201 :=
by
  sorry

end staircase_200_cells_l26_26188


namespace valid_parameterizations_l26_26121

def point_on_line (x y : ℝ) : Prop := (y = 2 * x - 5)

def direction_vector_valid (vx vy : ℝ) : Prop := (∃ (k : ℝ), vx = k * 1 ∧ vy = k * 2)

def parametric_option_valid (px py vx vy : ℝ) : Prop := 
  point_on_line px py ∧ direction_vector_valid vx vy

theorem valid_parameterizations : 
  (parametric_option_valid 10 15 5 10) ∧ 
  (parametric_option_valid 3 1 0.5 1) ∧ 
  (parametric_option_valid 7 9 2 4) ∧ 
  (parametric_option_valid 0 (-5) 10 20) :=
  by sorry

end valid_parameterizations_l26_26121


namespace f_x_when_x_negative_l26_26290

-- Define the properties of the function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → f x = x * (1 + x)

-- The theorem we want to prove
theorem f_x_when_x_negative (f : ℝ → ℝ) 
  (h1: odd_function f)
  (h2: f_definition f) : 
  ∀ x, x < 0 → f x = -x * (1 - x) :=
by
  sorry

end f_x_when_x_negative_l26_26290


namespace cylinder_volume_increase_l26_26224

noncomputable def volume_cylinder (H : ℝ) (R : ℝ) : ℝ := 
  π * H * R^2

noncomputable def delta_volume (H : ℝ) (R : ℝ) (dR : ℝ) : ℝ :=
  let V₀ := volume_cylinder H R
  let Vr := volume_cylinder H (R + dR)
  Vr - V₀

theorem cylinder_volume_increase (H R dR : ℝ) (hH : H = 40) (hR : R = 30) (hdR : dR = 0.5) :
  delta_volume H R dR = 1200 * π := 
by
  -- Definitions and assumptions
  rw [hH, hR, hdR]
  -- Calculate volumes
  show delta_volume 40 30 0.5 = 1200 * π
  sorry

end cylinder_volume_increase_l26_26224


namespace count_correct_statements_l26_26321

variables {a b c : ℝ}

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

-- Converse: If b^2 = ac, then a, b, c form a geometric sequence
def converse (a b c : ℝ) : Prop :=
  geometric_sequence a b c → (b * b = a * c)

-- Inverse: If a, b, c do not form a geometric sequence, then b^2 ≠ ac
def inverse (a b c : ℝ) : Prop :=
  ¬geometric_sequence a b c → (b * b ≠ a * c)

-- Contrapositive: If b^2 ≠ ac, then a, b, c do not form a geometric sequence
def contrapositive (a b c : ℝ) : Prop :=
  (b * b ≠ a * c) → ¬geometric_sequence a b c

theorem count_correct_statements :
  (converse a b c) ∧ (inverse a b c) ∧ (contrapositive a b c) → 4 :=
by
  sorry

end count_correct_statements_l26_26321


namespace ellipse_equation_l26_26996

noncomputable def ellipse_interpsect_line (
  a : ℝ,
  b : ℝ,
  x : ℝ,
  y : ℝ,
  p : ℝ,
  q : ℝ,
  cx : ℝ,
  cy : ℝ
) : Prop :=
  (a * x^2 + b * y^2 = 1) ∧
  (x + y - 1 = 0) ∧
  (a * p^2 + b * q^2 = 1) ∧
  (p + q - 1 = 0) ∧
  (|x - p| + |y - q| = 2 * sqrt 2) ∧
  (cx = (x + p) / 2) ∧
  (cy = (y + q) / 2) ∧
  (cy / cx = sqrt 2 / 2)

theorem ellipse_equation :
  ∀ (a b : ℝ),
    (a = 1/3) →
    (b = sqrt 2 / 3) →
      ellipse_interpsect_line a b x y p q cx cy →
      (a * x^2 + b * y^2 = 1) :=
begin
  intros,
  sorry
end

end ellipse_equation_l26_26996


namespace probability_three_or_more_same_l26_26838

-- Let us define the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8 ^ 5

-- Define the number of favorable outcomes where at least three dice show the same number
def favorable_outcomes : ℕ := 4208

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now we state the theorem that this probability simplifies to 1052/8192
theorem probability_three_or_more_same : probability = 1052 / 8192 :=
sorry

end probability_three_or_more_same_l26_26838


namespace train_speed_is_36_km_per_hour_l26_26566

-- Define the conditions
def train_length : ℝ := 500  -- length of the train in meters
def time_to_cross_pole : ℝ := 50  -- time to cross the pole in seconds

-- Speed calculation in m/s
def speed_in_meters_per_second : ℝ := train_length / time_to_cross_pole

-- Conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- Speed calculation in km/h
def speed_in_kilometers_per_hour : ℝ := speed_in_meters_per_second * conversion_factor

-- Theorem to state the speed of the train in km/h
theorem train_speed_is_36_km_per_hour : speed_in_kilometers_per_hour = 36 := by
  -- We assume the following mathematical facts and calculations to be correct
  sorry

end train_speed_is_36_km_per_hour_l26_26566


namespace division_correct_result_l26_26170

theorem division_correct_result (x : ℝ) (h : 8 * x = 56) : 42 / x = 6 := by
  sorry

end division_correct_result_l26_26170


namespace solve_imaginary_eq_l26_26760

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26760


namespace max_length_of_stick_in_box_l26_26572

theorem max_length_of_stick_in_box : 
  ∀ (length width height : ℝ), length = 5 → width = 4 → height = 3 → 
  (sqrt (length ^ 2 + width ^ 2 + height ^ 2) = 5 * sqrt 2) :=
by
  intros length width height h_length h_width h_height
  rw [h_length, h_width, h_height]
  calc sqrt (5 ^ 2 + 4 ^ 2 + 3 ^ 2)
      = sqrt (25 + 16 + 9) : by norm_num
  ... = sqrt 50 : by norm_num
  ... = sqrt (25 * 2) : by norm_num
  ... = sqrt 25 * sqrt 2 : by rw [mul_comm, sqrt_mul, sqrt_sqr, abs_of_nonneg]; norm_num
  ... = 5 * sqrt 2 : by norm_num

end max_length_of_stick_in_box_l26_26572


namespace min_abs_diff_l26_26329

theorem min_abs_diff (a b : ℕ) (h : a * b - 8 * a + 7 * b = 600) : ∃ a b : ℕ, a * b - 8 * a + 7 * b = 600 ∧ abs (a - b) = 30 :=
by
  sorry

end min_abs_diff_l26_26329


namespace part_1_complement_union_part_1_intersection_complement_part_2_condition_l26_26065

open Set

variable {X : Type} [PartialOrder X]

def A (x : X) : Prop := -2 < x ∧ x < 4
def B (m x : X) : Prop := 2*m - 1 < x ∧ x < m + 3
def U (x : X) : Prop := x ≤ 4

theorem part_1_complement_union (m : ℝ) (h : m = -1) : 
  (compl (U ∘ A)) ∪ (B m) = {x | x < 2 ∨ x = 4} :=
sorry

theorem part_1_intersection_complement (m : ℝ) (h : m = -1) : 
  A ∩ compl (U ∘ (B m)) = {x | 2 ≤ x ∧ x < 4} :=
sorry

theorem part_2_condition (m : ℝ) : 
  (∀ x, (A x ∨ B m x) = A x) ↔ (m ∈ Icc (-1/2) 1 ∪ Ici 4) :=
sorry

end part_1_complement_union_part_1_intersection_complement_part_2_condition_l26_26065


namespace big_al_bananas_l26_26232

/-- Big Al ate 140 bananas from May 1 through May 6. Each day he ate five more bananas than on the previous day. On May 4, Big Al did not eat any bananas due to fasting. Prove that Big Al ate 38 bananas on May 6. -/
theorem big_al_bananas : 
  ∃ a : ℕ, (a + (a + 5) + (a + 10) + 0 + (a + 15) + (a + 20) = 140) ∧ ((a + 20) = 38) :=
by sorry

end big_al_bananas_l26_26232


namespace membership_percentage_change_l26_26569

-- Definitions required based on conditions
def membersFallChange (initialMembers : ℝ) : ℝ := initialMembers * 1.07
def membersSpringChange (fallMembers : ℝ) : ℝ := fallMembers * 0.81
def membersSummerChange (springMembers : ℝ) : ℝ := springMembers * 1.15

-- Prove the total change in percentage from fall to the end of summer
theorem membership_percentage_change :
  let initialMembers := 100
  let fallMembers := membersFallChange initialMembers
  let springMembers := membersSpringChange fallMembers
  let summerMembers := membersSummerChange springMembers
  ((summerMembers - initialMembers) / initialMembers) * 100 = -0.33 := by
  sorry

end membership_percentage_change_l26_26569


namespace picnic_total_persons_l26_26974

-- Define the necessary variables
variables (W M A C : ℕ)

-- Define the conditions
def cond1 : Prop := M = W + 40
def cond2 : Prop := A = C + 40
def cond3 : Prop := M = 90

-- Define the total number of persons
def total_persons : ℕ := A + C

-- The theorem stating the total number of persons
theorem picnic_total_persons (h1 : cond1) (h2 : cond2) (h3 : cond3) : total_persons W M A C = 240 :=
by
    sorry

end picnic_total_persons_l26_26974


namespace polygon_sides_l26_26201

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end polygon_sides_l26_26201


namespace workshop_total_number_of_workers_l26_26946

theorem workshop_total_number_of_workers
  (average_salary_all : ℝ)
  (average_salary_technicians : ℝ)
  (average_salary_non_technicians : ℝ)
  (num_technicians : ℕ)
  (total_salary_all : ℝ -> ℝ)
  (total_salary_technicians : ℕ -> ℝ)
  (total_salary_non_technicians : ℕ -> ℝ -> ℝ)
  (h1 : average_salary_all = 9000)
  (h2 : average_salary_technicians = 12000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : ∀ W, total_salary_all W = average_salary_all * W )
  (h6 : ∀ n, total_salary_technicians n = n * average_salary_technicians )
  (h7 : ∀ n W, total_salary_non_technicians n W = (W - n) * average_salary_non_technicians)
  (h8 : ∀ W, total_salary_all W = total_salary_technicians num_technicians + total_salary_non_technicians num_technicians W) :
  ∃ W, W = 14 :=
by
  sorry

end workshop_total_number_of_workers_l26_26946


namespace slower_train_speed_l26_26511

theorem slower_train_speed
  (v : ℝ)  -- The speed of the slower train
  (faster_train_speed : ℝ := 46)  -- The speed of the faster train
  (train_length : ℝ := 37.5)  -- The length of each train in meters
  (time_to_pass : ℝ := 27)  -- Time taken to pass in seconds
  (kms_to_ms : ℝ := 1000 / 3600)  -- Conversion factor from km/hr to m/s
  (relative_distance : ℝ := 2 * train_length)  -- Distance covered when passing

  (h : relative_distance = (faster_train_speed - v) * kms_to_ms * time_to_pass) :
  v = 36 :=
by
  -- The proof should be placed here
  sorry

end slower_train_speed_l26_26511


namespace alternating_binomial_sum_100_l26_26604

theorem alternating_binomial_sum_100 : ∑ (k : ℕ) in finset.range 101, (-1 : ℤ)^k * (nat.choose 100 k : ℤ) = 0 := by
  sorry

end alternating_binomial_sum_100_l26_26604


namespace smallest_positive_integer_solution_l26_26926

theorem smallest_positive_integer_solution : ∃ n : ℕ, 23 * n % 9 = 310 % 9 ∧ n = 8 :=
by
  sorry

end smallest_positive_integer_solution_l26_26926


namespace find_z_l26_26747

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26747


namespace mean_greater_than_median_by_three_l26_26272

-- Define the problem conditions
variable (x : ℕ) (hx : x > 0)

-- Express the set of numbers and their mean and median
def mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 22)) / 5
def median := x + 4

-- State the theorem that the difference between the mean and the median is 3
theorem mean_greater_than_median_by_three (hx : x > 0) : mean x hx - median x hx = 3 :=
sorry

end mean_greater_than_median_by_three_l26_26272


namespace find_z_l26_26748

theorem find_z (z : ℂ) (hz : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * Complex.i) : z = 1 + Complex.i := 
sorry

end find_z_l26_26748


namespace reduced_rates_fraction_l26_26530

theorem reduced_rates_fraction : 
  let total_hours_in_week := 7 * 24 in
  let weekday_nights := 5 in
  let hours_per_night := 12 in
  let weekend_days := 2 in
  let hours_per_weekend_day := 24 in
  let reduced_rates_hours_weekday := weekday_nights * hours_per_night in
  let reduced_rates_hours_weekend := weekend_days * hours_per_weekend_day in
  let reduced_rates_hours_total := reduced_rates_hours_weekday + reduced_rates_hours_weekend in
  reduced_rates_hours_total / total_hours_in_week = 9 / 14 := by
  -- We assert the proof steps here.
  sorry

end reduced_rates_fraction_l26_26530


namespace car_speed_l26_26172

theorem car_speed
  (v : ℝ)       -- the unknown speed of the car in km/hr
  (time_80 : ℝ := 45)  -- the time in seconds to travel 1 km at 80 km/hr
  (time_plus_10 : ℝ := 55)  -- the time in seconds to travel 1 km at speed v

  (h1 : time_80 = 3600 / 80)
  (h2 : time_plus_10 = time_80 + 10) :
  v = 3600 / (55 / 3600) := sorry

end car_speed_l26_26172


namespace ratio_male_gerbils_l26_26236

variable (G H Gm Hm : ℕ)

-- Conditions
def total_pets : ℕ := 92
def total_gerbils : ℕ := 68
def total_hamsters : ℕ := total_pets - total_gerbils
def male_hamsters : ℕ := total_hamsters / 3
def male_pets : ℕ := 25
def male_gerbils : ℕ := Gm

-- Given that the total number of male pets is 25, male_pets = male_gerbils + male_hamsters
def male_condition : Prop := male_gerbils + male_hamsters = male_pets

-- Prove that the ratio of male gerbils to total gerbils is 1:4
theorem ratio_male_gerbils : 
  Gm = (male_pets - male_hamsters) → 
  (Gm / total_gerbils) = 1 / 4 := 
begin
  intros h,
  have h1 : Gm = 17 := by linarith,
  have h2 : total_gerbils = 68 := by linarith,
  rw [h1, h2],
  norm_num,
end

end ratio_male_gerbils_l26_26236


namespace perfect_square_c_values_l26_26796

def is_perfect_square_mod16 (n : ℕ) : Prop :=
  n % 16 ∈ {0, 1, 4, 9}

theorem perfect_square_c_values (a c: ℕ) (h: a ≠ 0) :
  is_perfect_square_mod16 (512 * a + 448 + 16 + c) →
  c = 0 ∨ c = 1 :=
sorry

end perfect_square_c_values_l26_26796


namespace milk_powder_cost_l26_26947

theorem milk_powder_cost (C : ℝ) : 
  (equal_cost_in_june : milk_powder_cost_june = coffee_cost_june) 
  (milk_powder_increase : coffee_cost_june * 3 = coffee_cost_july)
  (milk_powder_decrease : milk_powder_cost_june * 0.4 = milk_powder_cost_july)
  (mixture_cost : 5.1 * C = 5.10) : milk_powder_cost_july = 0.40 :=
sorry

end milk_powder_cost_l26_26947


namespace total_fireworks_l26_26024

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end total_fireworks_l26_26024


namespace largest_possible_b_l26_26098

-- Define the conditions of the problem
def f (x : ℝ) : ℝ := sorry -- Placeholder for the function f
def g (x : ℝ) : ℝ := sorry -- Placeholder for the function g

-- Range conditions for f and g
axiom range_f : ∀ x : ℝ, -3 ≤ f(x) ∧ f(x) ≤ 4
axiom range_g : ∀ x : ℝ, -3 ≤ g(x) ∧ g(x) ≤ 2

-- Define the product function h
def h (x : ℝ) : ℝ := f(x) * g(x)

-- Prove the largest possible value of b
theorem largest_possible_b : ∃ b : ℝ, (∀ x : ℝ, h(x) ≤ b) ∧ b = 12 :=
by
  sorry

end largest_possible_b_l26_26098


namespace min_expression_value_min_expression_specific_l26_26830

-- Define the given expression
def expression (a b : ℝ) : ℝ :=
  (a + b) * (a + 2) * (b + 2) / (16 * a * b)

-- State the theorem
theorem min_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  expression a b ≥ 1 :=
sorry

-- State the specific case where the minimum is achieved
theorem min_expression_specific (h : expression 2 2 = 1) : 
  expression 2 2 = 1 :=
begin
  assumption,
end

end min_expression_value_min_expression_specific_l26_26830


namespace func_eq_condition_l26_26410

/-- Prove that |a| = |b| is a necessary and sufficient condition for the existence of functions
    f and g from ℤ to ℤ such that f(g(x)) = x + a and g(f(x)) = x + b for all x in ℤ -/
theorem func_eq_condition (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), (∀ x, f (g x) = x + a) ∧ (∀ x, g (f x) = x + b)) ↔ |a| = |b| :=
  sorry

end func_eq_condition_l26_26410


namespace find_N_l26_26251

theorem find_N (N : ℕ) :
  (∃ (f : ℕ × ℕ × ℕ × ℕ × ℕ → ℕ), (∀ (x y z w t : ℕ), f (x, y, z, w, t) = (a + b + c + d + 2)^N ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w + t = N)
   →  ∑ n, (f = 715)) ↔ N = 12 := sorry

end find_N_l26_26251


namespace area_of_quadrilateral_l26_26010

def vector := ℝ × ℝ

variables (A B C D : vector)

-- Definitions for the problem conditions
def vector_add (v₁ v₂ : vector) : vector := (v₁.1 + v₂.1, v₁.2 + v₂.2)
def magnitude (v : vector) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)
def unit (v : vector) : vector := (v.1 / magnitude v, v.2 / magnitude v)

#check real.sqrt

-- Given conditions
variables 
(h1 : vector_add B A = (1,1))
(h2 : vector_add D C = (1,1))
(h3 : vector_add (unit (B, A)) (unit (C, B)) = vector_add (unit (D, B)) (real.sqrt 3))

-- Statement to prove
theorem area_of_quadrilateral : magnitude (vector_add B A) * magnitude (vector_add B C) * real.sqrt 3 / 2 = real.sqrt 3 := sorry

end area_of_quadrilateral_l26_26010


namespace necessary_but_not_sufficient_l26_26171

theorem necessary_but_not_sufficient 
  (p q : ℝ) :
  (2 * 108 * p^2 ≥ 4 * q) ↔ (∀ x : ℝ, x^4 + p * x^2 + q = 0 → real_roots_condition p q) := sorry

def real_roots_condition (p q : ℝ) : Prop :=
  p^2 ≥ 4 * q

end necessary_but_not_sufficient_l26_26171


namespace integral_abs_cos_l26_26256

open Real

theorem integral_abs_cos :
  ∫ x in 0..2 * π, |cos x| = 4 :=
sorry

end integral_abs_cos_l26_26256


namespace original_rice_amount_l26_26498

theorem original_rice_amount (x : ℝ) 
  (h1 : (x / 2) - 3 = 18) : 
  x = 42 :=
sorry

end original_rice_amount_l26_26498


namespace teacher_former_salary_l26_26968

variables (S : ℝ)

def former_salary (S : ℝ) : Prop :=
  let new_salary := 1.20 * S in
  let total_payments := 9 * 6000 in
  new_salary = total_payments

theorem teacher_former_salary (S : ℝ) :
  former_salary S → S = 45000 := by
  sorry

end teacher_former_salary_l26_26968


namespace product_roots_positive_real_part_l26_26337

open Complex

theorem product_roots_positive_real_part :
    (∃ (roots : Fin 6 → ℂ),
       (∀ k, roots k ^ 6 = -64) ∧
       (∀ k, (roots k).re > 0 → (roots 0).re > 0 ∧ (roots 0).im > 0 ∧
                               (roots 1).re > 0 ∧ (roots 1).im < 0) ∧
       (roots 0 * roots 1 = 4)
    ) :=
sorry

end product_roots_positive_real_part_l26_26337


namespace minho_game_difference_l26_26431

theorem minho_game_difference : 
  ∃ (n1 n2 n3 : ℕ), 
    (n1 = 1 ∧ n2 = 6 ∧ n3 = 8) ∧ 
    let l := [([n1, n2, n3].permutations.map (λ l, l.foldl (λ acc x, 10 * acc + x) 0)).eraseDuplicates in
    l.maximum = some 861 ∧ l.nth (l.length - 3) = some 681 ∧
    (861 - 681) = 180 :=
  sorry

end minho_game_difference_l26_26431


namespace complex_number_solution_l26_26690

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26690


namespace taimour_time_to_paint_alone_l26_26944

theorem taimour_time_to_paint_alone (T : ℝ) (h1 : Jamshid_time = T / 2)
  (h2 : (1 / T + 1 / (T / 2)) = 1 / 3) : T = 9 :=
sorry

end taimour_time_to_paint_alone_l26_26944


namespace minimum_pounds_to_better_deal_l26_26068

variable (x n : ℝ)
variable (h_positive: 0 < x)

-- Definitions based on problem conditions:
def LuciaCost (pounds : ℝ) : ℝ :=
  if pounds ≤ 20 then 
    pounds * x
  else 
    20 * x + (pounds - 20) * 0.8 * x

def AmbyCost (pounds : ℝ) : ℝ :=
  if pounds <= 14 then 
    pounds * x
  else 
    14 * x + (pounds - 14) * 0.9 * x

-- The hypothesis we need to prove that for pounds > 15, Lucia's becomes an equal or better deal than Amby's:
theorem minimum_pounds_to_better_deal : ∃ n, n ≥ 11 ∧ LuciaCost (15 + n) ≤ AmbyCost (15 + n) := 
  sorry

end minimum_pounds_to_better_deal_l26_26068


namespace greatest_number_dividing_with_remainders_l26_26948

theorem greatest_number_dividing_with_remainders :
  ∃ (x: ℕ), (x ∣ (1557 - 7) ∧ x ∣ (2037 - 5)) ∧ x = 2 :=
by
  have h1 : 1557 - 7 = 1550 := rfl
  have h2 : 2037 - 5 = 2032 := rfl
  sorry

end greatest_number_dividing_with_remainders_l26_26948


namespace solve_for_z_l26_26735

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26735


namespace similar_polygons_area_sum_l26_26325

theorem similar_polygons_area_sum (a b c k : ℝ) (t' t'' T : ℝ)
    (h₁ : t' = k * a^2)
    (h₂ : t'' = k * b^2)
    (h₃ : T = t' + t''):
    c^2 = a^2 + b^2 := 
by 
  sorry

end similar_polygons_area_sum_l26_26325


namespace isosceles_trapezoid_problem_l26_26402

variable (AB CD AD BC : ℝ)
variable (x : ℝ)

noncomputable def p_squared (AB CD AD BC : ℝ) (x : ℝ) : ℝ :=
  if AB = 100 ∧ CD = 25 ∧ AD = x ∧ BC = x then 1875 else 0

theorem isosceles_trapezoid_problem (h₁ : AB = 100)
                                    (h₂ : CD = 25)
                                    (h₃ : AD = x)
                                    (h₄ : BC = x) :
  p_squared AB CD AD BC x = 1875 := by
  sorry

end isosceles_trapezoid_problem_l26_26402


namespace probability_sum_less_than_5_l26_26521

-- Definitions
def die_sides : ℕ := 6
def outcome_space := (finset.range die_sides).product (finset.range die_sides)
def favorable_outcomes := outcome_space.filter (λ (p : ℕ × ℕ), p.1 + p.2 + 2 < 5)

-- Theorem to prove
theorem probability_sum_less_than_5 : (favorable_outcomes.card : ℚ) / outcome_space.card = 1 / 6 :=
by
  sorry

end probability_sum_less_than_5_l26_26521


namespace maximum_value_of_f_l26_26612

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x))

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = (16 * Real.sqrt 3) / 9 :=
sorry

end maximum_value_of_f_l26_26612


namespace initial_bananas_each_child_l26_26433

-- Define the variables and conditions.
def total_children : ℕ := 320
def absent_children : ℕ := 160
def present_children := total_children - absent_children
def extra_bananas : ℕ := 2

-- We are to prove the initial number of bananas each child was supposed to get.
theorem initial_bananas_each_child (B : ℕ) (x : ℕ) :
  B = total_children * x ∧ B = present_children * (x + extra_bananas) → x = 2 :=
by
  sorry

end initial_bananas_each_child_l26_26433


namespace parallel_vectors_range_g_l26_26326

noncomputable def vector_a : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)

theorem parallel_vectors (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) (h3 : vector_a ∥ vector_b x) : x = 2 * Real.pi / 3 := 
sorry

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * (Real.sin x) - (Real.cos x)
noncomputable def g (x : ℝ) : ℝ := -2 * Real.cos x

theorem range_g (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) : -2 ≤ g x ∧ g x ≤ 2 := 
sorry

end parallel_vectors_range_g_l26_26326


namespace complex_number_quadrant_l26_26481

theorem complex_number_quadrant :
  let z := (2 - (complex.i)) / (3 * (complex.i) - 1)
  (-1 / 2) - (1 / 2) * (complex.i) = z ∧ (z.re < 0 ∧ z.im < 0) := 
  sorry

end complex_number_quadrant_l26_26481


namespace paul_number_proof_l26_26441

theorem paul_number_proof (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a - b = 7) :
  (10 * a + b = 81) ∨ (10 * a + b = 92) :=
  sorry

end paul_number_proof_l26_26441


namespace solve_for_z_l26_26708

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26708


namespace proj_linear_comb_l26_26423

open Real EuclideanSpace

variables {u w : EuclideanSpace 3}

def proj (a b : EuclideanSpace 3) : EuclideanSpace 3 :=
  (inner a b / ∥b∥^2) • b

theorem proj_linear_comb :
  proj w (3 • u + w) = ((3 * ((3 : ℝ) * (u.x * w.x + u.y * w.y + u.z * w.z) + w.x * w.x + w.y * w.y + w.z * w.z) / ∥w∥ + 1) / ∥w∥) • w :=
sorry

end proj_linear_comb_l26_26423


namespace cost_function_discrete_points_l26_26078

def cost (n : ℕ) : ℕ :=
  if n <= 10 then 20 * n
  else if n <= 25 then 18 * n
  else 0

theorem cost_function_discrete_points :
  (∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ y, cost n = y) ∧
  (∀ m n, 1 ≤ m ∧ m ≤ 25 ∧ 1 ≤ n ∧ n ≤ 25 ∧ m ≠ n → cost m ≠ cost n) :=
sorry

end cost_function_discrete_points_l26_26078


namespace leap_day_2020_is_thursday_l26_26029

/--
Prove that February 29, 2020, is a Thursday given that February 29, 2000, was a Sunday.
-/
theorem leap_day_2020_is_thursday (h : calendar.day_of_week (calendar.date.mk 2000 2 29) = calendar.day.sunday) : 
  calendar.day_of_week (calendar.date.mk 2020 2 29) = calendar.day.thursday := 
sorry

end leap_day_2020_is_thursday_l26_26029


namespace find_n_l26_26605

theorem find_n (n : ℕ) (h : n * n.factorial + n.factorial = 720) : n = 5 :=
sorry

end find_n_l26_26605


namespace determine_n_l26_26306

def f (x : ℝ) (n : ℝ) : ℝ :=
if x < 1 then 2 * x + n else log x / log 2

theorem determine_n {n : ℝ} (h : f (f (3/4) n) n = 2) : n = 5/2 :=
sorry

end determine_n_l26_26306


namespace smallest_perfect_square_greater_l26_26782

theorem smallest_perfect_square_greater (a : ℕ) (h : ∃ n : ℕ, a = n^2) : 
  ∃ m : ℕ, m^2 > a ∧ ∀ k : ℕ, k^2 > a → m^2 ≤ k^2 :=
  sorry

end smallest_perfect_square_greater_l26_26782


namespace lottery_problem_l26_26344

theorem lottery_problem (n : ℕ) (hn : n ≥ 5) :
  (∑ i in finset.Ico 1 n, if (i ≤ n - 2) then combinations (n - 2) 3 else 0) =
  combinations n 3 :=
sorry

end lottery_problem_l26_26344


namespace zoe_total_money_l26_26526

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end zoe_total_money_l26_26526


namespace solution_least_odd_prime_factor_1001_12_plus_1_l26_26265

def least_odd_prime_factor_1001_12_plus_1 : Nat :=
  Nat.find (λ p => Prime p ∧ p ∣ (1001 ^ 12 + 1) ∧ Odd p)

theorem solution_least_odd_prime_factor_1001_12_plus_1 :
  least_odd_prime_factor_1001_12_plus_1 = 97 := by
  sorry

end solution_least_odd_prime_factor_1001_12_plus_1_l26_26265


namespace standard_deviation_of_scores_l26_26558

-- Definitions of conditions from part a)
def scores : List ℕ := [8, 9, 10, 10, 8]
def n : ℕ := 5

-- The mean of the dataset
def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

-- The variance of the dataset
def variance (l : List ℕ) : ℚ := (l.map (λ x, (x - mean l) ^ 2)).sum / l.length

-- The standard deviation of the dataset
def stdev (l : List ℕ) : ℚ := Real.sqrt (variance l)

-- The theorem to be proved
theorem standard_deviation_of_scores :
  stdev scores = Real.sqrt (4 / 5) :=
sorry

end standard_deviation_of_scores_l26_26558


namespace expected_area_of_reflected_quadrilateral_l26_26975

noncomputable def expected_area_of_quadrilateral_reflections := sorry

theorem expected_area_of_reflected_quadrilateral :
  let side_length := 2
  let area_of_square := side_length * side_length
  let expected_area := 2 * area_of_square
  expected_area_of_quadrilateral_reflections == expected_area := 
by
  sorry

end expected_area_of_reflected_quadrilateral_l26_26975


namespace recurring_decimal_fraction_l26_26910

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l26_26910


namespace distinct_values_count_l26_26597

def odd_integers_upto_17 := {x : ℕ | x % 2 = 1 ∧ x ≤ 17}

theorem distinct_values_count :
  {pq_sum | ∃ p q ∈ odd_integers_upto_17, pq_sum = (p + 1) * (q + 1) - 1}.size = 36 := 
sorry

end distinct_values_count_l26_26597


namespace find_n_l26_26274

theorem find_n (n : ℕ) (h : (0 : ℚ) < n) : 
  let total_ways := (n + 4).choose 2,
      no_girls := 4.choose 2,
      prob_at_least_one_girl := 5/6 in
  (no_girls / total_ways = 1 - prob_at_least_one_girl) -> 
  n = 5 := 
by
  sorry

end find_n_l26_26274


namespace both_firms_participate_condition_both_firms_will_participate_social_nonoptimal_participation_l26_26536

section RD

variables (V IC α : ℝ) (0 < α ∧ α < 1)

-- Condition for part (a)
def participation_condition : Prop :=
  α * V * (1 - 0.5 * α) ≥ IC

-- Part (b) Definition
def firms_participate_when : Prop :=
  V = 16 ∧ α = 0.5 ∧ IC = 5

-- Part (c) Definition
def social_optimal : Prop :=
  let total_profit_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC) in
  let total_profit_one := α * V - IC in
  total_profit_one > total_profit_both

-- Theorem for part (a)
theorem both_firms_participate_condition : participation_condition V IC α :=
sorry

-- Theorem for part (b)
theorem both_firms_will_participate (h : firms_participate_when V IC α) : participation_condition 16 5 0.5 :=
sorry

-- Theorem for part (c)
theorem social_nonoptimal_participation (h : firms_participate_when V IC α) : social_optimal 16 IC 0.5 :=
sorry

end RD

end both_firms_participate_condition_both_firms_will_participate_social_nonoptimal_participation_l26_26536


namespace greatest_possible_value_l26_26520

theorem greatest_possible_value :
  ∃ (N P M : ℕ), (M < 10) ∧ (N < 10) ∧ (P < 10) ∧ (M * (111 * M) = N * 1000 + P * 100 + M * 10 + M)
                ∧ (N * 1000 + P * 100 + M * 10 + M = 3996) :=
by
  sorry

end greatest_possible_value_l26_26520


namespace opposite_of_neg_sqrt3_l26_26127

theorem opposite_of_neg_sqrt3 : (-(sqrt 3)) = -sqrt 3 := sorry

end opposite_of_neg_sqrt3_l26_26127


namespace total_marbles_l26_26194

/-- A craftsman makes 35 jars. This is exactly 2.5 times the number of clay pots he made.
If each jar has 5 marbles and each clay pot has four times as many marbles as the jars plus an additional 3 marbles, 
prove that the total number of marbles is 497. -/
theorem total_marbles (number_of_jars : ℕ) (number_of_clay_pots : ℕ) (marbles_in_jar : ℕ) (marbles_in_clay_pot : ℕ) :
  number_of_jars = 35 →
  (number_of_jars : ℝ) = 2.5 * number_of_clay_pots →
  marbles_in_jar = 5 →
  marbles_in_clay_pot = 4 * marbles_in_jar + 3 →
  (number_of_jars * marbles_in_jar + number_of_clay_pots * marbles_in_clay_pot) = 497 :=
by 
  sorry

end total_marbles_l26_26194


namespace equal_sums_probability_l26_26589

theorem equal_sums_probability : 
  let nums := {1, 2, 3, 4}
  let groups := { (x, y) | x ⊆ nums ∧ y ⊆ nums ∧ x ∩ y = ∅ ∧ x ≠ ∅ ∧ y ≠ ∅ }
  let total_ways := (4.choose 1) + (4.choose 2) / 2
  let equal_sum_groups := { g ∈ groups | g.1.sum = g.2.sum }
  (equal_sum_groups.card : ℚ) / total_ways = 1 / 7 :=
sorry

end equal_sums_probability_l26_26589


namespace correct_ordering_of_periods_l26_26271

def f1 (x : ℝ) : ℝ := abs (sin (x / 2)) * abs (cos (x / 2))
def f2 (x : ℝ) : ℝ := sin (2 * x / 3) + cos (2 * x / 3)
def f3 (x : ℝ) : ℝ := arccos (sin x)

noncomputable def T1 : ℝ := π
noncomputable def T2 : ℝ := 3 * π
noncomputable def T3 : ℝ := 2 * π

theorem correct_ordering_of_periods : T1 < T3 ∧ T3 < T2 :=
by {
  sorry
}

end correct_ordering_of_periods_l26_26271


namespace no_divisor_of_wobbly_l26_26213

def isWobbly (n : ℕ) : Prop :=
  n > 0 ∧ (∀ (d : ℕ), d < n → (nat.mod (nat.div n (nat.pow 10 d)) 10 = 0 ↔ nat.mod (nat.div n (nat.pow 10 (d+1))) 10 ≠ 0)) ∧ (nat.mod n 10 ≠ 0)

theorem no_divisor_of_wobbly (m : ℕ) : 
  (∀ n, isWobbly n → ¬ (m ∣ n)) ↔ (m % 10 = 0 ∨ m % 25 = 0) :=
by
  sorry

end no_divisor_of_wobbly_l26_26213


namespace product_base_c_equals_12300_base_8_l26_26428

noncomputable def calculate_product_in_base_c (c : ℕ) : ℕ := (c + 4) * (c + 8) * (c + 9)

noncomputable def convert_to_base_c (n : ℕ) (c : ℕ) : ℕ :=
  let rec to_base (num power : ℕ) : ℕ :=
    if num = 0 then 0 else (num % c) * power + to_base (num / c) (power * 10)
  to_base n 1

theorem product_base_c_equals_12300_base_8 :
  ∀ (c : ℕ), calculate_product_in_base_c c = 4 * c^3 + 9 * c^2 + 6 * c → (convert_to_base_c 3264 8 = 12300) :=
by
  intros c h
  sorry

end product_base_c_equals_12300_base_8_l26_26428


namespace well_defined_set_l26_26162

inductive Person
| high_school_student_dude_school_january_2013 : Person

inductive Tree
| tall_and_large_tree_in_campus : Tree

def setA : Set Person :=
  {p : Person | p = Person.high_school_student_dude_school_january_2013 ∧ tall p}

def setB : Set Tree :=
  {t : Tree | t = Tree.tall_and_large_tree_in_campus}

def setC : Set Person :=
  {p : Person | p = Person.high_school_student_dude_school_january_2013}

def setD : Set Person :=
  {p : Person | high_basketball_level p}

theorem well_defined_set : setC = {p : Person | p = Person.high_school_student_dude_school_january_2013} :=
  sorry

end well_defined_set_l26_26162


namespace maximize_sum_12_l26_26903

/-- Given a list of integers, removing exactly one number and choosing two distinct integers at 
random to maximize the probability that their sum is 12 -/
def choose_to_maximize_sum (lst : List ℤ) (k : ℤ) : Prop :=
  ∃ l' : List ℤ, l' = lst.erase k ∧ 
  (∀ x y : ℤ, x ∈ l' → y ∈ l' → x ≠ y → x + y = 12 → l' ≠ lst.erase x ∧ l' ≠ lst.erase y)

theorem maximize_sum_12 (lst : List ℤ) : 
    lst = [-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12] → choose_to_maximize_sum lst 6 := 
by
  intro h
  exists lst.erase 6
  split
  · exact rfl
  · intros x y hx hy hxy hxy_sum
    cases hx
    cases hy
    sorry

end maximize_sum_12_l26_26903


namespace side_length_of_square_l26_26207

theorem side_length_of_square (A : ℝ) (h : A = 1/4) : ∃ n : ℝ, n^2 = A ∧ n = 1/2 := by
  use 1/2
  split
  conv_lhs { rw [h] }
  norm_num
  rfl

end side_length_of_square_l26_26207


namespace line_passes_through_fixed_point_l26_26070

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, (m - 1) * (-2) - 3 + 2 * m + 1 = 0 :=
by
  intros m
  sorry

end line_passes_through_fixed_point_l26_26070


namespace min_value_expression_l26_26656

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + a * c = 4) :
  ∃ m, m = 4 ∧ m ≤ 2 / a + 2 / (b + c) + 8 / (a + b + c) :=
by
  sorry

end min_value_expression_l26_26656


namespace rectangle_diagonal_eq_l26_26588

-- Definitions of a and b
def a : ℝ := 30 * Real.sqrt 2
def b : ℝ := 30 * (Real.sqrt 2 + 2)

-- The theorem to be proved
theorem rectangle_diagonal_eq :
  ∀ (a b : ℝ), a = 30 * Real.sqrt 2 → b = 30 * (Real.sqrt 2 + 2) →
  (Real.sqrt (a^2 + b^2) = Real.sqrt (7200 + 3600 * Real.sqrt 2)) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end rectangle_diagonal_eq_l26_26588


namespace y_axis_intercept_l26_26875

theorem y_axis_intercept (a b : ℝ) : 
  (∃ y, (∀ x, x = 0 → (x / a^2 - y / b^2 = 1)) → y = -b^2) :=
by
  intros y h
  sorry

end y_axis_intercept_l26_26875


namespace shaded_area_circle_ratio_l26_26529

theorem shaded_area_circle_ratio (AB AC CB r : ℝ) (hAB : AB = 2 * r) (hAC : AC = r) (hCB : CB = r)
  (hCD_perp : ∀ CD AB, CD ⊥ AB) :
  (1 / 4) = (1 / 2 * π * r ^ 2 - 2 * (1 / 8 * π * r ^ 2)) / (π * r ^ 2) :=
by
  sorry

end shaded_area_circle_ratio_l26_26529


namespace candy_bar_cost_l26_26230

variable (C : ℕ)

theorem candy_bar_cost
  (soft_drink_cost : ℕ)
  (num_candy_bars : ℕ)
  (total_spent : ℕ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27) :
  num_candy_bars * C + soft_drink_cost = total_spent → C = 5 := by
  sorry

end candy_bar_cost_l26_26230


namespace num_isosceles_triangles_l26_26843

structure Point :=
  (x : ℕ)
  (y : ℕ)

structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

noncomputable def distance (p1 p2 : Point) : Real :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def is_isosceles (t : Triangle) : Prop :=
  let d1 := distance t.A t.B
  let d2 := distance t.A t.C
  let d3 := distance t.B t.C
  d1 = d2 ∨ d1 = d3 ∨ d2 = d3

def triangles : List Triangle := [
  { A := { x := 0, y := 8 }, B := { x := 4, y := 8 }, C := { x := 2, y := 5 } },
  { A := { x := 2, y := 2 }, B := { x := 2, y := 5 }, C := { x := 6, y := 2 } },
  { A := { x := 1, y := 1 }, B := { x := 5, y := 4 }, C := { x := 9, y := 1 } },
  { A := { x := 7, y := 7 }, B := { x := 6, y := 9 }, C := { x := 10, y := 7 } },
  { A := { x := 3, y := 1 }, B := { x := 4, y := 4 }, C := { x := 6, y := 0 } }
]

def count_isosceles (ts : List Triangle) : ℕ :=
  List.length (List.filter is_isosceles ts)

theorem num_isosceles_triangles : count_isosceles triangles = 3 := by
  sorry

end num_isosceles_triangles_l26_26843


namespace smallest_possible_value_of_T_l26_26064

def permutation_of_twelve (l : List ℕ) : Prop :=
  l.sort = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

theorem smallest_possible_value_of_T 
  (a b c d : List ℕ)
  (h1 : a.length = 4) 
  (h2 : b.length = 4)
  (h3 : c.length = 4)
  (h4 : d.length = 4)
  (h5 : permutation_of_twelve (a ++ b ++ c ++ d)) : 
  (a.prod + b.prod + c.prod + d.prod) ≥ 3961 := 
sorry

end smallest_possible_value_of_T_l26_26064


namespace sequence_sums_l26_26426

noncomputable def a (n : ℕ) : ℕ := 1 + (n - 1) * 2

noncomputable def b (n : ℕ) : ℕ := 1 * 2 ^ (n - 1)

theorem sequence_sums :
  a (b 2) + a (b 3) + a (b 4) = 25 :=
by
  have b2 : b 2 = 2 := by
    simp [b]
  have b3 : b 3 = 4 := by
    simp [b]
  have b4 : b 4 = 8 := by
    simp [b]
  have a2 : a 2 = 3 := by
    simp [a]
  have a4 : a 4 = 7 := by
    simp [a]
  have a8 : a 8 = 15 := by
    simp [a]
  calc
    a 2 + a 4 + a 8 = 3 + 7 + 15 := by rw [a2, a4, a8]
    ... = 25 := by norm_num

end sequence_sums_l26_26426


namespace area_of_isosceles_right_triangle_l26_26899

def is_isosceles_right_triangle (X Y Z : Type*) : Prop :=
∃ (XY YZ XZ : ℝ), XY = 6.000000000000001 ∧ XY > YZ ∧ YZ = XZ ∧ XY = YZ * Real.sqrt 2

theorem area_of_isosceles_right_triangle
  {X Y Z : Type*}
  (h : is_isosceles_right_triangle X Y Z) :
  ∃ A : ℝ, A = 9.000000000000002 :=
by
  sorry

end area_of_isosceles_right_triangle_l26_26899


namespace teacups_count_l26_26579

theorem teacups_count (total_people teacup_capacity : ℕ) (H1 : total_people = 63) (H2 : teacup_capacity = 9) : total_people / teacup_capacity = 7 :=
by
  sorry

end teacups_count_l26_26579


namespace prove_points_are_coplanar_l26_26328

noncomputable def points_are_coplanar (A B C D : Type*) (angle : Type*) [field.angle angle] (π : angle)
  (angle_ACB : angle = π / 2)
  (angle_DBC : angle = π / 2)
  (angle_DAC : angle = π / 2)
  (angle_ADB : angle = π / 2) : Prop :=
coplanar A B C D

theorem prove_points_are_coplanar (A B C D : Type*) (angle : Type*) [field.angle angle] (π : angle)
  (angle_ACB : angle = π / 2)
  (angle_DBC : angle = π / 2)
  (angle_DAC : angle = π / 2)
  (angle_ADB : angle = π / 2) :
  coplanar A B C D :=
sorry

end prove_points_are_coplanar_l26_26328


namespace brad_red_balloons_l26_26233

theorem brad_red_balloons:
  ∀ (T G r : ℕ), T = 17 ∧ G = 9 ∧ r = T - G → r = 8 :=
by
  intros T G r h
  cases h with hT hG
  cases hG with hG hr
  simp [hT, hG] at hr
  assumption

-- Test the statement
#eval brad_red_balloons 17 9 8 (And.intro rfl (And.intro rfl rfl)) -- Expect true (i.e., theorem provable)

end brad_red_balloons_l26_26233


namespace min_value_inequality_l26_26063

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 3 * y = 4) :
  ∃ z, z = (2 / x + 3 / y) ∧ z = 25 / 4 :=
by
  sorry

end min_value_inequality_l26_26063


namespace quartic_polynomial_value_l26_26829

noncomputable def r : ℕ → ℕ
  | 1 := 5
  | 2 := 8
  | 3 := 13
  | 4 := 20
  | n := sorry

theorem quartic_polynomial_value :
  r 5 = 53 :=
sorry

end quartic_polynomial_value_l26_26829


namespace find_range_m_l26_26965

variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]

def f_condition (x : ℝ) : Prop :=
  f x + f (2 - x) = (x - 1)^2

def f_deriv_condition (x : ℝ) : Prop :=
  x ≤ 1 → f' x + 2 < x

def f_inequality_condition (m : ℝ) : Prop :=
  f m - f (1 - m) ≥ 3 / 2 - 3 * m

theorem find_range_m (m : ℝ) (h_f : ∀ x, f_condition f x)
  (h_f' : ∀ x, f_deriv_condition f x)
  (h_ineq : f_inequality_condition f m) : 
  m ∈ Iic (1 / 2) :=  -- Iic (1 / 2) corresponds to (-∞, 1/2]
sorry

end find_range_m_l26_26965


namespace reconstruct_circles_condition_l26_26105

variables {e1 e2 f1 f2 : Line} -- Assuming some definitions for lines exist.
variables {d_e d_f : ℝ} -- Assuming distances are reals.
variables {R r : ℝ} -- Radii of the circles

-- Conditions
def is_tangent (e : Line) (k : Circle) : Prop := sorry
def parallel (a b : Line) : Prop := sorry
def distance (a b : Line) : ℝ := sorry
def not_parallel (a b : Line) : Prop := ¬parallel a b

theorem reconstruct_circles_condition (d_e d_f : ℝ) (h_de_ne_df : d_e ≠ d_f)
  (h_e1_tangent_k1 : is_tangent e1 k1) 
  (h_e2_tangent_k2 : is_tangent e2 k2)
  (h_f1_tangent_k1 : is_tangent f1 k1)
  (h_f2_tangent_k2 : is_tangent f2 k2)
  (h_e1_parallel : parallel e1 e)
  (h_e2_parallel : parallel e2 e)
  (h_f1_parallel : parallel f1 f)
  (h_f2_parallel : parallel f2 f)
  (h_not_parallel_e1f1 : not_parallel e1 f1)
  (h_not_parallel_e2f2 : not_parallel e2 f2)
  : ∃ k1 k2 : Circle, 
  ((is_tangent e1 k1) ∧ (is_tangent e2 k2) ∧ 
   (is_tangent f1 k1) ∧ (is_tangent f2 k2) ∧
   (distance e1 e2 = 2 * R - 2 * r ∨ distance e1 e2 = 2 * R + 2 * r) ∧
   (distance f1 f2 = 2 * R - 2 * r ∨ distance f1 f2 = 2 * R + 2 * r)) :=
begin
  sorry
end

end reconstruct_circles_condition_l26_26105


namespace compute_expression_l26_26408

noncomputable def roots_exist (P : Polynomial ℝ) (α β γ : ℝ) : Prop :=
  P = Polynomial.C (-13) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-7) + Polynomial.X))

theorem compute_expression (α β γ : ℝ) (h : roots_exist (Polynomial.X^3 - 7 * Polynomial.X^2 + 11 * Polynomial.X - 13) α β γ) :
  (α ≠ 0) → (β ≠ 0) → (γ ≠ 0) → (α^2 * β^2 + β^2 * γ^2 + γ^2 * α^2 = -61) :=
  sorry

end compute_expression_l26_26408


namespace problem_statement_l26_26472

noncomputable def p (x : ℝ) : ℝ := - (27 / 5) * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

theorem problem_statement : (p(-2) / q(-2) = -9 / 10) :=
by
  sorry

end problem_statement_l26_26472


namespace repeating_decimal_division_l26_26914

theorem repeating_decimal_division :
  let x := 0 + 54 / 99 in -- 0.545454... = 54/99 = 6/11
  let y := 0 + 18 / 99 in -- 0.181818... = 18/99 = 2/11
  x / y = 3 :=
by
  sorry

end repeating_decimal_division_l26_26914


namespace cosine_of_A_l26_26031

-- Definitions of points and conditions
variables {A B C D K L M N : Type*}
[simplex] : K is_midpoint_of A B
[simplex] : L is_midpoint_of B C
[simplex] : M is_midpoint_of C D
[simplex] : N is_midpoint_of A D
[simplex] : is_convex_quadrilateral A B C D

-- The theorem statement
theorem cosine_of_A (h₁ : cyclic A B L M D) (h₂ : cyclic K B C D N) : cos_angle A = 1 / 4 :=
sorry

end cosine_of_A_l26_26031


namespace quadratic_real_roots_range_l26_26299

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end quadratic_real_roots_range_l26_26299


namespace joe_market_expense_l26_26391

theorem joe_market_expense :
  let oranges := 3 in
  let price_orange := 4.50 in
  let juices := 7 in
  let price_juice := 0.50 in
  let jars_honey := 3 in
  let price_jar_honey := 5 in
  let plants := 4 in
  let price_plants_per_pair := 18 in
  let total_cost := oranges * price_orange + juices * price_juice + jars_honey * price_jar_honey + (plants / 2) * price_plants_per_pair in
  total_cost = 68 := 
sorry

end joe_market_expense_l26_26391


namespace select_two_fruits_l26_26508

theorem select_two_fruits (n k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  nat.choose n k = 10 :=
by
  rw [h1, h2]
  exact nat.choose 5 2
  sorry

end select_two_fruits_l26_26508


namespace river_crossing_possible_l26_26584

-- Definitions
def person : Type := ℕ
def couple : Type := (person × person)
def boat_capacity : ℕ := 2

-- Persons
def A : person := 1
def a : person := 2
def B : person := 3
def b : person := 4
def C : person := 5
def c : person := 6

-- Couples
def couple1 : couple := (A, a)
def couple2 : couple := (B, b)
def couple3 : couple := (C, c)

-- Initial and goal states
def north_bank_initial : set person := {A, a, B, b, C, c}
def south_bank_goal : set person := {A, a, B, b, C, c}

-- Conditions function
def valid_crossing (current_north : set person) (current_south : set person) : Prop :=
  ∀ p ∈ current_north, 
    (∃ (c1 c2 : person), (p, c1) ∈ {couple1, couple2, couple3} ∧ (p, c2) ∉ {couple1, couple2, couple3})

-- Proof statement
theorem river_crossing_possible :
  ∃ (crossings : list (set person × set person)),
    let (north_bank, south_bank) := (north_bank_initial, ∅ : set person) in
    all_valid_crossings : list.valid_crossings crossings ∧
    list.last crossings (north_bank_initial, ∅ : set person) = (∅ : set person, south_bank_goal) := sorry

end river_crossing_possible_l26_26584


namespace rectangle_perimeter_eq_l26_26243

noncomputable def rectangle_perimeter (z w : ℕ) : ℕ :=
  let longer_side := w
  let shorter_side := (z - w) / 2
  2 * longer_side + 2 * shorter_side

theorem rectangle_perimeter_eq (z w : ℕ) : rectangle_perimeter z w = w + z := by
  sorry

end rectangle_perimeter_eq_l26_26243


namespace problem_proof_l26_26338

-- Formalizing the conditions of the problem
variable {a : ℕ → ℝ}  -- Define the arithmetic sequence
variable (d : ℝ)      -- Common difference of the arithmetic sequence
variable (a₅ a₆ a₇ : ℝ)  -- Specific terms in the sequence

-- The condition given in the problem
axiom cond1 : a 5 + a 6 + a 7 = 15

-- A definition for an arithmetic sequence
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Using the axiom to deduce that a₆ = 5
axiom prop_arithmetic : is_arithmetic_seq a d

-- We want to prove that sum of terms from a₃ to a₉ = 35
theorem problem_proof : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by sorry

end problem_proof_l26_26338


namespace problem_proof_l26_26123

-- The problem statement translated to a Lean definition
def probability_odd_sum_rows_columns := 1 / 14

theorem problem_proof :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let grid := finset.powersetLen 9 nums -- The set of all ways to fill the grid
  (∃ g ∈ grid, 
   ∀ row ∈ (fin 3), 
   odd (sum (g row)) ∧ 
   ∀ col ∈ (fin 3),
   odd (sum (λ (x : ℕ), g x col))) → real.ratDiv 1 14 :=
by
  sorry

end problem_proof_l26_26123


namespace area_ratio_of_shaded_to_white_is_five_to_three_l26_26925

theorem area_ratio_of_shaded_to_white_is_five_to_three :
  (∑ t in shaded_triangles, area t) / (∑ t in white_triangles, area t) = 5 / 3 := 
sorry

end area_ratio_of_shaded_to_white_is_five_to_three_l26_26925


namespace steve_height_after_growth_l26_26855

theorem steve_height_after_growth :
  (5 * 12 + 6) * 2.54 * 1.15 ≈ 193 :=
by sorry

end steve_height_after_growth_l26_26855


namespace product_of_g_on_roots_l26_26056

noncomputable def f : Polynomial ℝ :=
  Polynomial.C 1 + Polynomial.X ^ 3 + Polynomial.X ^ 6

noncomputable def g (x : ℝ) : ℝ := x ^ 2 - 3

lemma product_of_g (g : ℝ → ℝ) (x : ℝ) : g (x) * g (-x) = (x^2 - 3)^2 :=
begin
  rw [g, g, pow_two, pow_two, add_mul_self_eq]
end

theorem product_of_g_on_roots (f : Polynomial ℝ) (g : ℝ → ℝ) :
  ∀ (roots : Fin 6 → ℝ),
  Polynomial.roots f.toRoots = Multiset.toFinset roots  →
  (Multiset.map g (Polynomial.roots f)).prod = 757 :=
begin
  sorry
end

end product_of_g_on_roots_l26_26056


namespace total_interest_at_tenth_year_l26_26535

def principal_trebled_interest (P R : ℝ) : bool :=
  let SI₁ := (P * R * 10) / 100
  let P' := 3 * P
  let SI₂ := (P' * R * 5) / 100
  SI₁ = 1000 ∧ SI₂ = 1500 → (SI₁ + SI₂) = 2500

theorem total_interest_at_tenth_year
    (P R : ℝ)
    (h1: (P * R * 10) / 100 = 1000)
    (h2: P' = 3 * P)
    (h3: (P' * R * 5) / 100 = 1500) :
    ( (P * R * 10) / 100 + ((3 * P) * R * 5) / 100 = 2500) :=
  by
    rw [h1, ← h2, h3]
    sorry

end total_interest_at_tenth_year_l26_26535


namespace complex_number_solution_l26_26691

theorem complex_number_solution (z : ℂ) (h: 2 * (z + conj z) + 3 * (z - conj z) = complex.of_real 4 + complex.I * 6) : 
  z = complex.of_real 1 + complex.I := 
sorry

end complex_number_solution_l26_26691


namespace sample_size_220_l26_26146

variable {total_students : ℕ}
variable {selected_students : ℕ}

theorem sample_size_220 :
  total_students = 1320 →
  selected_students = 220 →
  selected_students = 220 :=
begin
  intros h_ts h_ss,
  exact h_ss,
end

end sample_size_220_l26_26146


namespace complex_solution_l26_26721

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26721


namespace maxwell_meets_brad_in_six_hours_l26_26179

theorem maxwell_meets_brad_in_six_hours
    (distance_between_homes : ℝ := 54)
    (maxwell_speed : ℝ := 4)
    (brad_speed : ℝ := 6)
    (delayed_start : ℝ := 1) :
    ∃ t : ℝ, (maxwell_speed * (t + delayed_start) + brad_speed * t = distance_between_homes) ∧ (t + delayed_start = 6) :=
by {
  -- The variables for the time it takes Maxwell and Brad to meet
  let t := 5,
  -- Define the time Maxwell has been walking
  have maxwell_time : ℝ := t + delayed_start,
  -- Ensure the time it takes Maxwell to meet Brad is 6 hours
  use t,
  split,
  {
    calc
      maxwell_speed * maxwell_time + brad_speed * t = 4 * (5 + 1) + 6 * 5 : by rfl
      ... = 4 * 6 + 6 * 5 : by rfl
      ... = 24 + 30 : by rfl
      ... = 54 : by rfl,
  },
  {
    exact eq.refl 6,
  }
}

end maxwell_meets_brad_in_six_hours_l26_26179


namespace symmetric_trapezoid_in_circle_range_of_a_squared_l26_26004

theorem symmetric_trapezoid_in_circle (R a : ℝ) (x : ℝ)
  (hx : 0 ≤ x ∧ x ≤ 2 * R) :
  3 * R^2 ≤ (x - R)^2 + 3 * R^2 ∧ (x - R)^2 + 3 * R^2 ≤ 4 * R^2 :=
begin
  sorry
end

theorem range_of_a_squared (R a : ℝ) :
  (∃ x, 0 ≤ x ∧ x ≤ 2 * R ∧ (x - R)^2 + 3 * R^2 = a^2) ↔ 3 * R^2 ≤ a^2 ∧ a^2 ≤ 4 * R^2 :=
begin
  sorry
end

end symmetric_trapezoid_in_circle_range_of_a_squared_l26_26004


namespace quadrilateral_area_l26_26376

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  0.5 * a * b * Math.sin(c)

theorem quadrilateral_area 
  (A B C D E P : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space P]
  (PE : ℝ) (PD : ℝ)
  (DE : ℝ := Real.sqrt (PE^2 + PD^2))
  (h_PE : PE = 2)
  (h_PD : PD = 3)
  (h_DE : DE = Real.sqrt (2^2 + 3^2))
  : let CP := 2 * PE,
        AP := 2 * PD,
        area_AEDC :=
            0.5 * (6 * 2 + 3 * 2 + 4 * 3 + 6 * 4) in
        area_AEDC = 27 :=
by {
  -- Note: Proof would be here, but only statement is provided as per requirement.
  sorry
}

end quadrilateral_area_l26_26376


namespace spelling_bee_initial_students_l26_26357

theorem spelling_bee_initial_students (x : ℕ) 
    (h1 : (2 / 3) * x = 2 / 3 * x)
    (h2 : (3 / 4) * ((1 / 3) * x) = 3 / 4 * (1 / 3 * x))
    (h3 : (1 / 3) * x * (1 / 4) = 30) : 
  x = 120 :=
sorry

end spelling_bee_initial_students_l26_26357


namespace solve_imaginary_eq_l26_26758

theorem solve_imaginary_eq (a b : ℝ) (z : ℂ)
  (h_z : z = a + b * complex.I)
  (h_conj : complex.conj z = a - b * complex.I)
  (h_eq : 2 * (z + complex.conj z) + 3 * (z - complex.conj z) = 4 + 6 * complex.I) :
  z = 1 + complex.I := 
sorry

end solve_imaginary_eq_l26_26758


namespace validity_of_D_l26_26297

def binary_op (a b : ℕ) : ℕ := a^(b + 1)

theorem validity_of_D (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  binary_op (a^n) b = (binary_op a b)^n := 
by
  sorry

end validity_of_D_l26_26297


namespace point_always_lies_on_linear_function_l26_26071

theorem point_always_lies_on_linear_function :
  ∀ (k : ℝ), (2k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 :=
begin
  intro k,
  sorry
end

end point_always_lies_on_linear_function_l26_26071


namespace at_least_two_consecutive_heads_probability_l26_26196

theorem at_least_two_consecutive_heads_probability :
  let outcomes := ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
  let favorable_outcomes := ["HHH", "HHT", "THH"]
  let total_outcomes := outcomes.length
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 2 :=
by sorry

end at_least_two_consecutive_heads_probability_l26_26196


namespace root_expression_equals_181_div_9_l26_26041

noncomputable def polynomial_root_sum (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22) 
  (h3 : a*b*c = 8) : ℝ :=
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) 

theorem root_expression_equals_181_div_9
  (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22)
  (h3 : a*b*c = 8) :
  polynomial_root_sum a b c h1 h2 h3 = 181 / 9 := by 
  sorry

end root_expression_equals_181_div_9_l26_26041


namespace lines_intersect_at_point_l26_26797

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := ⟨4, -3, 6⟩
def Q : Point3D := ⟨20, -23, 14⟩
def R : Point3D := ⟨-2, 7, -10⟩
def S : Point3D := ⟨6, -11, 16⟩

def linePQ (t : ℝ) : Point3D :=
  ⟨4 + 16 * t, -3 - 20 * t, 6 + 8 * t⟩

def lineRS (u : ℝ) : Point3D :=
  ⟨-2 + 8 * u, 7 - 18 * u, -10 + 26 * u⟩

def intersection_point : Point3D :=
  ⟨180 / 19, -283 / 19, 202 / 19⟩

theorem lines_intersect_at_point :
  ∃ (t u : ℝ), linePQ t = lineRS u ∧ linePQ t = intersection_point :=
sorry

end lines_intersect_at_point_l26_26797


namespace total_shirts_produced_today_l26_26500

def shirts_produced_per_minute (a b c : Nat) : Prop :=
  a = 6 ∧ b = 8 ∧ c = 4

def working_minutes_today (a b c : Nat) : Prop :=
  a = 12 ∧ b = 10 ∧ c = 6

def number_of_shirts_made (a b c : Nat) : Nat := a * 6 + b * 8 + c * 4

theorem total_shirts_produced_today 
  (shirts_per_min : Nat → Nat → Nat → Prop)
  (working_min : Nat → Nat → Nat → Prop)
  (shirt_production : Nat → Nat)
  (total : Nat) :
shirts_per_min 6 8 4 →
working_min 12 10 6 →
shirt_production 176 :=
by 
  intros h1 h2
  -- Use the conditions and basic arithmetic to prove the result
  have hA: 6 * 12 = 72 := by norm_num
  have hB: 8 * 10 = 80 := by norm_num
  have hC: 4 * 6 = 24 := by norm_num
  have h_total: 72 + 80 + 24 = 176 := by norm_num
  exact h_total

-- Defined shirts per minute for the three machines
example : shirts_produced_per_minute 6 8 4 := by trivial

-- Defined working minutes for today for the three machines
example : working_minutes_today 12 10 6 := by trivial

-- The total number of shirts produced today
example : total_shirts_produced_today shirts_produced_per_minute working_minutes_today (number_of_shirts_made 12 10 6) 176 := sorry

end total_shirts_produced_today_l26_26500


namespace ellipse_and_line_l26_26298

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a c : ℝ) : Prop :=
  (c / a = 1 / 2)

def line_through_focus (k : ℝ) : Prop :=
  ∃ (F1 : ℝ × ℝ),
    F1 = (-1, 0) ∧
    ∀ x y,
      y = k * (x + 1)

theorem ellipse_and_line:
  ∃ (a b c k : ℝ),
    (ellipse_equation a b x y) ∧
    (eccentricity a c) ∧
    (
      let x y := (x, y) in
      (line_through_focus k) ∧
      (|math.sqrt(x^2 + y^2) - c| = 1) ∧
      (
        let l := k * (x + 1) in
        (l = sqrt(3)/2 * (x + 1) ∨ l = -sqrt(3)/2 * (x + 1))
      )
    )
    → (ellipse_equation 2 sqrt(3) x y) ∧
       (
          let x y := (x, y) in
          let l := sqrt(3)/2 * (x+1) ∨ l := -sqrt(3)/2 * (x+1) in
          true
       ) :=
by
  \highlight{assume} sorry

end ellipse_and_line_l26_26298


namespace laura_five_dollar_bills_l26_26028

theorem laura_five_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 40) 
  (h2 : x + 2 * y + 5 * z = 120) 
  (h3 : y = 2 * x) : 
  z = 16 := 
by
  sorry

end laura_five_dollar_bills_l26_26028


namespace chords_containing_point_with_integer_lengths_l26_26443

theorem chords_containing_point_with_integer_lengths
  (O P : ℝ^2)
  (r : ℝ)
  (h1 : ∥O - P∥ = 8)
  (h2 : r = 17) :
  ∃ n : ℕ, n = 5 :=
by
  sorry

end chords_containing_point_with_integer_lengths_l26_26443


namespace area_triangle_DBC_l26_26363

-- Definitions for the problem setup
variable (A B C D : ℝ × ℝ)
variable (midpoint_AB midpoint_BC : ℝ × ℝ)

-- Coordinates of the points A, B, and C
def coord_A := (0, 10)
def coord_B := (0, 0)
def coord_C := (10, 0)

-- Midpoints D and E (we'll call E as midpoint_BC)
def midpoint_AB := ((coord_A.1 + coord_B.1) / 2, (coord_A.2 + coord_B.2) / 2)
def midpoint_BC := ((coord_B.1 + coord_C.1) / 2, (coord_B.2 + coord_C.2) / 2)

-- Validate if points D and E are indeed the midpoints
lemma midpoint_D : midpoint_AB = (0, 5) := by
  simp [coord_A, coord_B, midpoint_AB]

lemma midpoint_E : midpoint_BC = (5, 0) := by
  simp [coord_B, coord_C, midpoint_BC]

-- Definition of the base and height of the triangle ∆DBC
def base_BC : ℝ := 10
def height_D : ℝ := 5

-- Proof statement that the area of triangle DBC is 25
theorem area_triangle_DBC : 
  (1 / 2) * base_BC * height_D = 25 := by
  simp [base_BC, height_D]
  norm_num

end area_triangle_DBC_l26_26363


namespace scientific_notation_of_192M_l26_26073

theorem scientific_notation_of_192M : 192000000 = 1.92 * 10^8 :=
by 
  sorry

end scientific_notation_of_192M_l26_26073


namespace relation_between_a_b_c_l26_26635

def a : ℝ := Real.log 0.5 / Real.log 2
def b : ℝ := (1 / 2) ^ (-2)
def c : ℝ := 2 ^ (1 / 2)

theorem relation_between_a_b_c : a < c ∧ c < b := by
  sorry

end relation_between_a_b_c_l26_26635


namespace common_elements_in_S_and_T_l26_26407

theorem common_elements_in_S_and_T : 
  let S := {n | ∃ k, 1 ≤ k ∧ k ≤ 3000 ∧ n = 5 * k}
  let T := {n | ∃ k, 1 ≤ k ∧ k ≤ 1500 ∧ n = 10 * k}
  finset.card (finset.filter (λ x, x ∈ T) S.to_finset) = 1500 := by
  sorry

end common_elements_in_S_and_T_l26_26407


namespace triangle_property_l26_26282

noncomputable theory

open Real 

variables {A B C D K : Type*} [InnerProductSpace ℝ A B C D K] 
variable (triangle_ABC_right : ∃ (A B C : Type*) [InnerProductSpace ℝ A B C], ∠C = π / 2)
variable (D_on_AC : D ∈ lineSegment A C)
variable (K_on_BD : K ∈ lineSegment B D)
variable (angles_equal : ∠ABC = ∠KAD ∧ ∠KAD = ∠AKD)

theorem triangle_property : 
  ∀ (A B C D K : Type*) [InnerProductSpace ℝ A B C D K], D ∈ lineSegment A C → K ∈ lineSegment B D → ∠ABC = ∠KAD ∧ ∠KAD = ∠AKD → BK = 2 * DC := 
by
  sorry

end triangle_property_l26_26282


namespace only_one_way_to_center_l26_26459

def is_center {n : ℕ} (grid_size n : ℕ) (coord : ℕ × ℕ) : Prop :=
  coord = (grid_size / 2 + 1, grid_size / 2 + 1)

def count_ways_to_center : ℕ :=
  if h : (1 <= 3 ∧ 3 <= 5) then 1 else 0

theorem only_one_way_to_center : count_ways_to_center = 1 := by
  sorry

end only_one_way_to_center_l26_26459


namespace tangent_triangle_area_l26_26787

theorem tangent_triangle_area (a : ℝ) (h : a > 0)
  (tangent_line : ℝ → ℝ := λ x, (1 / (2 * real.sqrt a)) * (x - a) + real.sqrt a)
  (area : ℝ := (1 / 2) * a * (real.sqrt a / 2)) :
  area = 2 → a = 4 :=
by
  intro ha
  unfold area at ha
  sorry

end tangent_triangle_area_l26_26787


namespace interval_between_births_l26_26487

noncomputable def age_sum := 12
noncomputable def youngest_age := 1.5

theorem interval_between_births (x : ℝ) : 
  youngest_age + (youngest_age + x) + (youngest_age + 2 * x) + (youngest_age + 3 * x) = age_sum → 
  x = 1 :=
by
  sorry

end interval_between_births_l26_26487


namespace solve_f_g_f_3_l26_26049

def f (x : ℤ) : ℤ := 2 * x + 4

def g (x : ℤ) : ℤ := 5 * x + 2

theorem solve_f_g_f_3 :
  f (g (f 3)) = 108 := by
  sorry

end solve_f_g_f_3_l26_26049


namespace complex_solution_l26_26725

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26725


namespace nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l26_26038

theorem nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3
  (a b : ℤ)
  (h : 9 ∣ (a^2 + a * b + b^2)) :
  3 ∣ a ∧ 3 ∣ b :=
sorry

end nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l26_26038


namespace max_perimeter_of_triangle_l26_26985

theorem max_perimeter_of_triangle (x : ℕ) 
  (h1 : 3 < x) 
  (h2 : x < 15) 
  (h3 : 7 + 8 > x) 
  (h4 : 7 + x > 8) 
  (h5 : 8 + x > 7) :
  x = 14 ∧ 7 + 8 + x = 29 := 
by {
  sorry
}

end max_perimeter_of_triangle_l26_26985


namespace locus_of_second_common_point_l26_26900

noncomputable theory

open real

/-- Given a segment AB and a point P moving along AB, where two circles 
pass through P with radii λ times the lengths of segments AP and BP 
respectively, with λ > 1/2, the locus of the second common point of these circles 
as P traverses the interior of AB consists of two arcs seen from AB subtending a constant angle 
excluding endpoints A and B, and the segment AB excluding the endpoints. -/

theorem locus_of_second_common_point
    (A B : ℝ × ℝ) (λ : ℝ) (hλ : λ > 1/2) :
    ∀ (P : ℝ × ℝ), P ∈ segment ℝ A B → 
    ∃ (M : ℝ × ℝ), M ≠ P ∧
        -- General condition that M lies on one of the arcs or the segment
        (M ∈ (locus_of_arcs A B λ) ∨ M ∈ (segment ℝ A B \ {A, B})) :=
sorry 

-- Helper definition for the arcs
def locus_of_arcs (A B : ℝ × ℝ) (λ : ℝ) : ℝ × ℝ → Prop :=
λ M, ∃ θ : ℝ, θ ∈ Ioo 0 π ∧ M ∈ arc_circumference A B θ λ 

-- Further helper definitions and axioms for the arcs, intersection points, and other geometric constructs
def arc_circumference (A B : ℝ × ℝ) (θ : ℝ) (λ : ℝ) : set (ℝ × ℝ) :=
-- Define the set of points forming an arc at angle θ seen from AB with given λ
{M | ∀ P, P ∈ segment ℝ A B → ∠A M P = θ ∧ ∠P M B = θ}


end locus_of_second_common_point_l26_26900


namespace oshea_large_planters_l26_26440

theorem oshea_large_planters {total_seeds small_planter_capacity num_small_planters large_planter_capacity : ℕ} 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : large_planter_capacity = 20) :
  (total_seeds - num_small_planters * small_planter_capacity) / large_planter_capacity = 4 :=
by
  sorry

end oshea_large_planters_l26_26440


namespace quadratic_expression_value_l26_26655

variables (α β : ℝ)
noncomputable def quadratic_root_sum (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) : Prop :=
  α + β = -2

theorem quadratic_expression_value (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) (h3 : α + β = -2) :
  α^2 + 3*α + β = -1 :=
sorry

end quadratic_expression_value_l26_26655


namespace trajectory_equation_find_k_l26_26011

open Real

-- a) Define the conditions
def E : Point := (1, 0)
def F : Point := (-1, 0)
def G (x y : ℝ) := slope (E, (x, y)) * slope ((x, y), F) = -4

theorem trajectory_equation (x y : ℝ) : G(x, y) → (x^2 + y^2 / 4 = 1) :=
sorry

theorem find_k (x1 x2 k : ℝ)
  (h_midpoint : (x1 + x2) / 2 = 4)
  (h_line : ∀ y, y = k * x1 - 1 ∧ y = k * x2 - 1)
  (h_intersect : x1^2 + (k * x1 - 1)^2 / 4 = 1 ∧ x2^2 + (k * x2 - 1)^2 / 4 = 1) :
  k = 2 :=
sorry

end trajectory_equation_find_k_l26_26011


namespace matrix_power_eq_l26_26240

noncomputable def rotation_matrix (θ : ℝ) : Matrix 2 2 ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem matrix_power_eq :
  (3 • (rotation_matrix (Real.pi / 4))) ^ 4 =
    81 • ![![(-1 : ℝ), 0], ![0, -1]] :=
by sorry

end matrix_power_eq_l26_26240


namespace area_of_triangle_l26_26810

def triangle (α β γ : Type) : (α ≃ β) ≃ γ ≃ Prop := sorry

variables (α β γ : Type) (AB AC AM : ℝ)
variables (ha : AB = 9) (hb : AC = 17) (hc : AM = 12)

theorem area_of_triangle (α β γ : Type) (AB AC AM : ℝ)
  (ha : AB = 9) (hb : AC = 17) (hc : AM = 12) : 
  ∃ A : ℝ, A = 74 :=
sorry

end area_of_triangle_l26_26810


namespace median_is_8_l26_26646

variable x y : ℝ

-- The conditions of the problem
def data := [7, 8, 9, x, y]

-- given conditions
lemma average_condition : (7 + 8 + 9 + x + y) / 5 = 8 := sorry

-- statement of the proof problem
theorem median_is_8 (h : (7 + 8 + 9 + x + y) / 5 = 8) : median data = 8 :=
  sorry

end median_is_8_l26_26646


namespace complex_solution_l26_26728

theorem complex_solution (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * complex.i) : z = 1 + complex.i := by
  sorry

end complex_solution_l26_26728


namespace distinct_cube_configurations_l26_26956

/-- 
  A cube configuration is a 2x2x2 arrangement of unit cubes where each cube is either 
  white, blue, or red. Two configurations are considered the same if one can be 
  rotated to match the other. We want to prove that the number of distinct configurations 
  given 3 white cubes, 3 blue cubes, and 2 red cubes is 54.
-/
theorem distinct_cube_configurations : 
  let colorings := {config | (∃ w b r, (w, b, r) ∈ config ∧ w = 3 ∧ b = 3 ∧ r = 2)} in
  let configurations := {cfg | cfg ∈ colorings ∧ (∃ rot, rot ∈ rotations ∧ rot cfg = cfg)} in
  (finset.card configurations).quotient_by_rotation = 54 :=
sorry

end distinct_cube_configurations_l26_26956


namespace sum_of_distinct_selections_is_34_l26_26208

-- Define a 4x4 grid filled sequentially from 1 to 16
def grid : List (List ℕ) := [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
]

-- Define a type for selections from the grid ensuring distinct rows and columns.
structure Selection where
  row : ℕ
  col : ℕ
  h_row : row < 4
  h_col : col < 4

-- Define the sum of any selection of 4 numbers from distinct rows and columns in the grid.
def sum_of_selection (selections : List Selection) : ℕ :=
  if h : List.length selections = 4 then
    List.sum (List.map (λ sel => (grid.get! sel.row).get! sel.col) selections)
  else 0

-- The main theorem
theorem sum_of_distinct_selections_is_34 (selections : List Selection) 
  (h_distinct_rows : List.Nodup (List.map (λ sel => sel.row) selections))
  (h_distinct_cols : List.Nodup (List.map (λ sel => sel.col) selections)) :
  sum_of_selection selections = 34 :=
by
  -- Proof is omitted
  sorry

end sum_of_distinct_selections_is_34_l26_26208


namespace solve_for_z_l26_26741

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26741


namespace tangent_line_at_P_l26_26871

noncomputable def tangent_line (x : ℝ) (y : ℝ) := (8 * x - y - 12 = 0)

def curve (x : ℝ) := x^3 - x^2

def derivative (f : ℝ → ℝ) (x : ℝ) := 3 * x^2 - 2 * x

theorem tangent_line_at_P :
    tangent_line 2 4 :=
by
  sorry

end tangent_line_at_P_l26_26871


namespace problem_solution_l26_26211

def point := (ℝ × ℝ)

def A : point := (-3, 2)
def B : point := (4, -1)
def C : point := (-1, -5)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def AB := distance A B
noncomputable def AC := distance A C

noncomputable def ratio := AB / AC

noncomputable def section_point (p1 p2 : point) (r : ℝ) : point :=
  ((r * p2.1 + p1.1) / (r + 1),
   (r * p2.2 + p1.2) / (r + 1))

noncomputable def D : point := section_point B C ratio

noncomputable def equation_of_line (p1 p2 : point) : (ℝ × ℝ × ℝ) :=
  let a := p2.2 - p1.2
  let b := p1.1 - p2.1
  let c := p2.1 * p1.2 - p1.1 * p2.2
  (a, b, c)
  
noncomputable def (d, _, e) := equation_of_line A D

theorem problem_solution :
  -- Replace 'correct_sum' with the exact value after the actual calculation
  d + e = correct_sum := 
  sorry

end problem_solution_l26_26211


namespace exists_routes_l26_26497

def ports := {1, 2, 3, 4, 5, 6}

def routes (r : set (set ports)) : Prop :=
  ∀ p1 p2 p3 : ports, {p1, p2, p3} ∈ r → p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

def pair_count (r : set (set ports)) : Prop :=
  ∀ p1 p2 : ports, p1 ≠ p2 → (∃! route ∈ r, {p1, p2} ⊆ route)

theorem exists_routes : ∃ r : set (set ports), (∀ route ∈ r, ∃ p1 p2 p3 : ports, route = {p1, p2, p3}) ∧ routes r ∧ pair_count r :=
  sorry

end exists_routes_l26_26497


namespace multichoose_comb_l26_26099

theorem multichoose_comb (n r : ℕ) : 
  (F_n_r : ℕ) : ℕ := Nat.choose (n + r - 1) r :=
sorry

end multichoose_comb_l26_26099


namespace mindy_mork_earnings_ratio_l26_26580

theorem mindy_mork_earnings_ratio (M K : ℝ) (h1 : 0.20 * M + 0.30 * K = 0.225 * (M + K)) : M / K = 3 :=
by
  sorry

end mindy_mork_earnings_ratio_l26_26580


namespace min_sum_of_squares_l26_26062

theorem min_sum_of_squares (y1 y2 y3 : ℝ) (h1 : y1 > 0) (h2 : y2 > 0) (h3 : y3 > 0) (h4 : y1 + 3 * y2 + 4 * y3 = 72) : 
  y1^2 + y2^2 + y3^2 ≥ 2592 / 13 ∧ (∃ k, y1 = k ∧ y2 = 3 * k ∧ y3 = 4 * k ∧ k = 36 / 13) :=
sorry

end min_sum_of_squares_l26_26062


namespace solve_log_expression_l26_26088

theorem solve_log_expression (x : ℝ) : log 8 x + log 2 (x ^ 3) = 15 → x = 16 * real.sqrt 2 :=
by
  sorry

end solve_log_expression_l26_26088


namespace Andy_late_minutes_l26_26999

theorem Andy_late_minutes 
  (school_start : Nat := 8*60) -- 8:00 AM in minutes since midnight
  (normal_travel_time : Nat := 30) -- 30 minutes
  (red_light_stops : Nat := 3 * 4) -- 3 minutes each at 4 lights
  (construction_wait : Nat := 10) -- 10 minutes
  (detour_time : Nat := 7) -- 7 minutes
  (store_stop_time : Nat := 5) -- 5 minutes
  (traffic_delay : Nat := 15) -- 15 minutes
  (departure_time : Nat := 7*60 + 15) -- 7:15 AM in minutes since midnight
  : 34 = departure_time + normal_travel_time + red_light_stops + construction_wait + detour_time + store_stop_time + traffic_delay - school_start := 
by sorry

end Andy_late_minutes_l26_26999


namespace solve_for_z_l26_26705

variable (z : ℂ)

theorem solve_for_z : (2 * (z + conj(z)) + 3 * (z - conj(z)) = 4 + 6 * complex.I) → (z = 1 + complex.I) :=
by
  intro h
  sorry

end solve_for_z_l26_26705


namespace determine_z_l26_26704

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26704


namespace matrix_power_four_l26_26238

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3 * Real.sqrt 2 / 2, -3 / 2], ![3 / 2, 3 * Real.sqrt 2 / 2]]

theorem matrix_power_four :
  matrixA ^ 4 = ![![ -81, 0], ![0, -81]] :=
by sorry

end matrix_power_four_l26_26238


namespace coordinates_of_M_l26_26827

def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (0, 2)

def line_through_A (k : ℝ) : ℝ × ℝ → Prop :=
  λ p, k * p.1 + p.2 = 0

def line_through_B (k : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.1 - k * p.2 + 2 * k = 0

def distances (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def proof_problem (k : ℝ) : Prop :=
  ∃ M : ℝ × ℝ,
    line_through_A k M ∧ 
    line_through_B k M ∧ 
    0 < M.1 ∧ 
    distances M point_B = 2 * distances M point_A ∧ 
    M = (4 / 5, 2 / 5)

theorem coordinates_of_M (k : ℝ) : proof_problem k := by
  sorry

end coordinates_of_M_l26_26827


namespace sum_g_l26_26260

noncomputable def g (x : ℝ) : ℝ := 9 * x / (3 + 9 * x)

theorem sum_g : 
  ∑ k in Finset.range 1995, g ((k + 1) / 1996 : ℝ) = 997 := 
sorry

end sum_g_l26_26260


namespace cos_theta_correct_l26_26970

-- Define the vectors
def vec1 : ℝ × ℝ := (4, -1)
def vec2 : ℝ × ℝ := (2, 5)

-- Define the dot product function
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude function
def magnitude (a : ℝ × ℝ) : ℝ := Math.sqrt (a.1 ^ 2 + a.2 ^ 2)

-- Define the cosine of the angle function
def cos_theta (a b : ℝ × ℝ) : ℝ := dot a b / (magnitude a * magnitude b)

-- The theorem to prove
theorem cos_theta_correct : cos_theta vec1 vec2 = 3 / Math.sqrt 493 :=
by
  sorry

end cos_theta_correct_l26_26970


namespace total_fireworks_l26_26026

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end total_fireworks_l26_26026


namespace right_triangle_inequality_l26_26420

variable (a b c : ℝ)

theorem right_triangle_inequality
  (h1 : b < a) -- shorter leg is less than longer leg
  (h2 : c = Real.sqrt (a^2 + b^2)) -- hypotenuse from Pythagorean theorem
  : a + b / 2 > c ∧ c > (8 / 9) * (a + b / 2) := 
sorry

end right_triangle_inequality_l26_26420


namespace sum_of_prime_f_values_eq_5618_l26_26270

theorem sum_of_prime_f_values_eq_5618 :
  (finset.filter (λ n, nat.prime (n^4 - 256 * n^2 + 960)) 
  (finset.range 100)).sum (λ n, n^4 - 256 * n^2 + 960) = 5618 :=
by sorry

end sum_of_prime_f_values_eq_5618_l26_26270


namespace total_investment_is_10000_l26_26860

-- Definitions based on conditions
def total_interest_received := 684
def invested_at_6_percent := 7200
def invested_at_9_percent (T : ℝ) := T - invested_at_6_percent
def interest_from_6_percent := 0.06 * invested_at_6_percent
def interest_from_9_percent (T : ℝ) := 0.09 * invested_at_9_percent T
def total_interest (T : ℝ) := interest_from_6_percent + interest_from_9_percent T

-- Lean 4 statement
theorem total_investment_is_10000 (T : ℝ) :
  total_interest T = total_interest_received → T = 10000 :=
by
  sorry

end total_investment_is_10000_l26_26860


namespace probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l26_26154

-- Considering a die with 6 faces
def die_faces := 6

-- Total number of possible outcomes when rolling 3 dice
def total_outcomes := die_faces^3

-- 1. Probability of having exactly one die showing a 6 when rolling 3 dice
def prob_exactly_one_six : ℚ :=
  have favorable_outcomes := 3 * 5^2 -- 3 ways to choose which die shows 6, and 25 ways for others to not show 6
  favorable_outcomes / total_outcomes

-- Proof statement
theorem probability_exactly_one_six : prob_exactly_one_six = 25/72 := by 
  sorry

-- 2. Probability of having at least one die showing a 6 when rolling 3 dice
def prob_at_least_one_six : ℚ :=
  have no_six_outcomes := 5^3
  (total_outcomes - no_six_outcomes) / total_outcomes

-- Proof statement
theorem probability_at_least_one_six : prob_at_least_one_six = 91/216 := by 
  sorry

-- 3. Probability of having at most one die showing a 6 when rolling 3 dice
def prob_at_most_one_six : ℚ :=
  have no_six_probability := 125 / total_outcomes
  have one_six_probability := 75 / total_outcomes
  no_six_probability + one_six_probability

-- Proof statement
theorem probability_at_most_one_six : prob_at_most_one_six = 25/27 := by 
  sorry

end probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l26_26154


namespace determine_z_l26_26700

noncomputable def z_eq (a b : ℝ) : ℂ := a + b * complex.I

theorem determine_z (a b : ℝ)
  (h : 2 * (z_eq a b + complex.conj (z_eq a b)) + 3 * (z_eq a b - complex.conj (z_eq a b)) = 4 + 6 * complex.I) :
  z_eq a b = 1 + complex.I := by
  sorry

end determine_z_l26_26700


namespace solve_for_z_l26_26737

theorem solve_for_z (z : ℂ) (h : 2 * (z + conj z) + 3 * (z - conj z) = 4 + 6 * I) : z = 1 + I :=
sorry

end solve_for_z_l26_26737


namespace number_of_valid_permutations_l26_26891

theorem number_of_valid_permutations : 
  let s := list.range 10 in
  s.permutations.length = 13122 :=
by
  sorry

end number_of_valid_permutations_l26_26891


namespace consecutive_integers_sum_eq_fifty_l26_26499

theorem consecutive_integers_sum_eq_fifty :
  ∃ n : ℕ, ∑ i in finset.range n, (-49 + i) = 50 ∧ n = 100 :=
by
  -- Definitions & Skipping the proof
  sorry

end consecutive_integers_sum_eq_fifty_l26_26499


namespace max_people_in_chairs_l26_26492

noncomputable def max_people_seated (n : ℕ) := 
  if n = 2017 then 2016 else 0

theorem max_people_in_chairs (n : ℕ) (h : n = 2017) : 
  max_people_seated n = 2016 := 
by 
  rw [max_people_seated, if_pos h]
  sorry

end max_people_in_chairs_l26_26492


namespace correct_statements_l26_26586

namespace ProofProblem

def P1 : Prop := (-4) + (-5) = -9
def P2 : Prop := -5 - (-6) = 11
def P3 : Prop := -2 * (-10) = -20
def P4 : Prop := 4 / (-2) = -2

theorem correct_statements : P1 ∧ P4 ∧ ¬P2 ∧ ¬P3 := by
  -- proof to be filled in later
  sorry

end ProofProblem

end correct_statements_l26_26586
