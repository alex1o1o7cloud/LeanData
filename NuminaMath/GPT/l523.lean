import Analysis.Convex.Basic
import Mathlib
import Mathlib.Algebra.Cubic
import Mathlib.Algebra.Exponent
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.InjSurj
import Mathlib.Algebra.Order.Ring
import Mathlib.Analysis.Calculus.Terminally
import Mathlib.Analysis.SpecialFunctions.Circle
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Algebra.EuclideanDomain
import Mathlib.Init.Data.Nat.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.MeasureTheory.Probability.MassFunction
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.ArithmeticFunction.Totient
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Instances.Real
import Real

namespace minimum_shift_for_symmetry_l523_523095

theorem minimum_shift_for_symmetry (m : ℝ) (h : m > 0) :
    (∀ x : ℝ, (λ x, (λ x, (3 / 2) * Real.cos (2 * x) + (Real.sqrt 3 / 2) * Real.sin (2 * x)) (x + m)) = (λ x, (λ x, (3 / 2) * Real.cos (2 * (-x)) + (Real.sqrt 3 / 2) * Real.sin (2 * (-x))))) ↔ m = π / 12 :=
sorry

end minimum_shift_for_symmetry_l523_523095


namespace perimeter_of_l_shaped_region_l523_523385

theorem perimeter_of_l_shaped_region : 
  ∀ (angle_looks_right : ∀ (x y z : ℝ), x + y = z),
    (ten_segments_length: ∀ n: ℕ, n > 0 ∧ n ≤ 10 → 1) 
    (region_area : ℝ),
    region_area = 72 → ∃ (perimeter : ℝ), perimeter = 39.4 := 
by
  intros angle_looks_right ten_segments_length region_area req_area_eq
  sorry

end perimeter_of_l_shaped_region_l523_523385


namespace angle_B_eq_3pi_over_10_l523_523030

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523030


namespace correct_statements_count_is_3_l523_523113

def statement1 : Prop := ¬ ∀ (X Y : Type), correlation X Y
def statement2 : Prop := ¬ correlation circle_circumference circle_radius
def statement3 : Prop := uncertain_relationship product_demand product_price
def statement4 : Prop := meaningless_regression_line scatter_plot
def statement5 : Prop := deterministic_problem_using_regression_line transformation_uncertainty

def numberOfCorrectStatements : Nat := ([statement1, statement2, statement3, statement4, statement5].count id)

theorem correct_statements_count_is_3 : numberOfCorrectStatements = 3 := by
  sorry

end correct_statements_count_is_3_l523_523113


namespace probability_excluded_probability_selected_l523_523662

-- Define the population size and the sample size
def population_size : ℕ := 1005
def sample_size : ℕ := 50
def excluded_count : ℕ := 5

-- Use these values within the theorems
theorem probability_excluded : (excluded_count : ℚ) / (population_size : ℚ) = 5 / 1005 :=
by sorry

theorem probability_selected : (sample_size : ℚ) / (population_size : ℚ) = 50 / 1005 :=
by sorry

end probability_excluded_probability_selected_l523_523662


namespace irreducible_fraction_l523_523441

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l523_523441


namespace probability_at_least_one_two_l523_523139

theorem probability_at_least_one_two :
  let n_faces := 8
  let total_outcomes := n_faces * n_faces
  let no_two_outcomes := (n_faces - 1) * (n_faces - 1)
  let at_least_one_two := total_outcomes - no_two_outcomes
  let probability := at_least_one_two / total_outcomes
  probability = 15 / 64 :=
by
  let n_faces := 8
  let total_outcomes := n_faces * n_faces
  let no_two_outcomes := (n_faces - 1) * (n_faces - 1)
  let at_least_one_two := total_outcomes - no_two_outcomes
  let probability := at_least_one_two / total_outcomes
  show probability = 15 / 64, from sorry

end probability_at_least_one_two_l523_523139


namespace waiting_time_probability_l523_523903

theorem waiting_time_probability :
  ∀ (bus_schedule : ℕ → ℕ) (arrival_time : ℕ), (∀ t, bus_schedule t = t * 60) → 
  (∃ (interval : finset ℕ), interval = finset.range 60 ∧ ∀ t ∈ interval, 
  t - arrival_time % 60 ≤ 10 → 
  (finset.card (finset.filter (λ x, x - arrival_time % 60 ≤ 10) interval).to_set) / 
  (finset.card interval.to_set) = 1 / 6) :=
by
  sorry

end waiting_time_probability_l523_523903


namespace number_of_x_intercepts_of_sin_1_over_x_l523_523274

noncomputable def x_intercepts_sin_1_over_x_in_interval (a b : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℕ :=
  let π := Real.pi
  let k₁ := Nat.floor (b⁻¹ * π) in
  let k₂ := Nat.floor (a⁻¹ * π) in
  k₁ - k₂

theorem number_of_x_intercepts_of_sin_1_over_x :
  x_intercepts_sin_1_over_x_in_interval 0.00001 0.0001 (λ x, Real.sin (x⁻¹)) = 28647 :=
by
  sorry

end number_of_x_intercepts_of_sin_1_over_x_l523_523274


namespace daria_amount_owed_l523_523242

variable (savings : ℝ)
variable (couch_price : ℝ)
variable (table_price : ℝ)
variable (lamp_price : ℝ)
variable (total_cost : ℝ)
variable (amount_owed : ℝ)

theorem daria_amount_owed (h_savings : savings = 500)
                          (h_couch : couch_price = 750)
                          (h_table : table_price = 100)
                          (h_lamp : lamp_price = 50)
                          (h_total_cost : total_cost = couch_price + table_price + lamp_price)
                          (h_amount_owed : amount_owed = total_cost - savings) :
                          amount_owed = 400 :=
by
  sorry

end daria_amount_owed_l523_523242


namespace fixed_deposit_maturity_amount_l523_523230

-- Defining the conditions
def initial_deposit : ℝ := 100000
def annual_interest_rate : ℝ := 0.05
def years : ℕ := 10

-- Proposition and proof obligation
theorem fixed_deposit_maturity_amount :
  let final_amount := initial_deposit * (1 + annual_interest_rate) ^ years
  in final_amount = 100000 * 1.05 ^ 10 := by
  sorry

end fixed_deposit_maturity_amount_l523_523230


namespace arithmetic_sequence_S9_l523_523306

theorem arithmetic_sequence_S9 (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n : ℕ, a n = a 1 + ↑n * d)
  (h2 : a 3 + a 4 + a 8 = 9) : 
  (∑ i in finset.range 9, a i) = 27 :=
by
  sorry

end arithmetic_sequence_S9_l523_523306


namespace range_of_x_l523_523926

theorem range_of_x (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (x + 2) / (x - 3) :=
by {
  sorry
}

end range_of_x_l523_523926


namespace fraction_outside_smaller_circle_l523_523924

theorem fraction_outside_smaller_circle (r : ℝ) (r_pos : 0 < r) :
  let larger_circle_area := π * r^2,
      smaller_circle_area := π * (r / 2)^2
  in (larger_circle_area - smaller_circle_area) / larger_circle_area = 3 / 4 :=
by
  let larger_circle_area := π * r^2
  let smaller_circle_area := π * (r / 2)^2
  have h1 : larger_circle_area = π * r^2 := rfl
  have h2 : smaller_circle_area = π * (r / 2)^2 := rfl
  have h3 : (larger_circle_area - smaller_circle_area) / larger_circle_area = (π * r^2 - π * (r / 2)^2) / (π * r^2) := by
    rw [h1, h2]
  have h4 : (π * r^2 - π * (r / 2)^2) / (π * r^2) = (π * r^2 - π * (r^2 / 4)) / (π * r^2) := by
    congr
    rw [sq_div, ←div_eq_mul_one_div]
    congr
  have h5 : (π * r^2 - π * (r^2 / 4)) / (π * r^2) = (π * r^2 * (1 - 1 / 4)) / (π * r^2) := by
    rw [sub_mul, one_mul]
  have h6 : (π * r^2 * (1 - 1 / 4)) / (π * r^2) = (π * r^2 * (3 / 4)) / (π * r^2) := by
    congr
    norm_num
  have h7 : (π * r^2 * (3 / 4)) / (π * r^2) = 3 / 4 := by
    rw [mul_div_cancel]
    if_neg; exact ne_of_gt pi_pos
  exact h7

end fraction_outside_smaller_circle_l523_523924


namespace complex_conjugate_x_l523_523837

theorem complex_conjugate_x (x : ℝ) (h : x^2 + x - 2 + (x^2 - 3 * x + 2 : ℂ) * Complex.I = 4 + 20 * Complex.I) : x = -3 := sorry

end complex_conjugate_x_l523_523837


namespace chessboard_division_impossible_l523_523077

theorem chessboard_division_impossible :
  ∀ (chessboard : ℕ) (lines : ℕ),
  chessboard = 64 → lines = 13 → 
  (∃ f : ℕ → (ℝ × ℝ) → Prop, ∀ i : ℕ, i < lines → 
    ∀ (p : ℝ × ℝ), f i p → (∃ q : ℝ × ℝ, p ≠ q ∧ f i q)) → 
  false := 
by
  intros chessboard lines h1 h2 f f_spec
  sorry

end chessboard_division_impossible_l523_523077


namespace anna_gets_more_candy_l523_523223

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end anna_gets_more_candy_l523_523223


namespace distance_between_foci_of_hyperbola_l523_523265

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 32 - y^2 / 4 = 1) → 2 * Real.sqrt ((32 : ℝ) + 4) = 12 :=
by
  intros x y h,
  sorry

end distance_between_foci_of_hyperbola_l523_523265


namespace contrapositive_equiv_l523_523111

variable (x : Type)

theorem contrapositive_equiv (Q R : x → Prop) :
  (∀ x, Q x → R x) ↔ (∀ x, ¬ (R x) → ¬ (Q x)) :=
by
  sorry

end contrapositive_equiv_l523_523111


namespace polynomial_remainder_x1012_l523_523279

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end polynomial_remainder_x1012_l523_523279


namespace final_solution_percentage_is_correct_l523_523539

noncomputable def final_solution_percentage_liquid_X : ℚ :=
  let solution_Y_liquid_X := 0.25 * 10
  let solution_Y_water := 0.75 * 10
  let evaporated_Y := (solution_Y_liquid_X, solution_Y_water - 3)
  let added_solution_Y := (0.25 * 2, 0.75 * 2)
  let mixed_solution := (evaporated_Y.1 + added_solution_Y.1, evaporated_Y.2 + added_solution_Y.2)
  let evaporated_Z := (0.8 * mixed_solution.1, 0.8 * mixed_solution.2)
  let added_solution_W := (0.4 * 5, 0.6 * 5)
  let final_mixture := (evaporated_Z.1 + added_solution_W.1, evaporated_Z.2 + added_solution_W.2)
  (final_mixture.1 / (final_mixture.1 + final_mixture.2)) * 100

theorem final_solution_percentage_is_correct :
  final_solution_percentage_liquid_X ≈ 36.07 := 
begin
  sorry
end

end final_solution_percentage_is_correct_l523_523539


namespace units_digit_of_j_squared_plus_3_power_j_l523_523994

def j : ℕ := 2023^3 + 3^2023 + 2023

theorem units_digit_of_j_squared_plus_3_power_j (j : ℕ) (h : j = 2023^3 + 3^2023 + 2023) : 
  ((j^2 + 3^j) % 10) = 6 := 
  sorry

end units_digit_of_j_squared_plus_3_power_j_l523_523994


namespace log_3_eq_4_of_infinite_log_pattern_l523_523761

theorem log_3_eq_4_of_infinite_log_pattern (x : ℝ) (h : 0 < x) :
  x = 4 :=
begin
  sorry
end

end log_3_eq_4_of_infinite_log_pattern_l523_523761


namespace mcq_options_l523_523716

theorem mcq_options :
  ∃ n : ℕ, (1/n : ℝ) * (1/2) * (1/2) = (1/12) ∧ n = 3 :=
by
  sorry

end mcq_options_l523_523716


namespace parallelogram_area_l523_523260

theorem parallelogram_area (base height : ℕ) (h_base : base = 3) (h_height : height = 4) : base * height = 12 := 
by {
  rw [h_base, h_height],
  norm_num,
  sorry
}

end parallelogram_area_l523_523260


namespace cost_of_five_trip_ticket_l523_523229

-- Variables for the costs of the tickets
variables (x y z : ℕ)

-- Conditions from the problem
def condition1 : Prop := 5 * x > y
def condition2 : Prop := 4 * y > z
def condition3 : Prop := z + 3 * y = 33
def condition4 : Prop := 20 + 3 * 5 = 35

-- The theorem to prove
theorem cost_of_five_trip_ticket (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z y) (h4 : condition4) : y = 5 := 
by
  sorry

end cost_of_five_trip_ticket_l523_523229


namespace arithmetic_square_root_of_sqrt_16_l523_523581

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523581


namespace find_cos_angle_and_area_l523_523915

noncomputable theory
open Real

variables (A B C D N : Type*)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace N]
variables [HasInnerProduct A] [HasInnerProduct B] [HasInnerProduct C] [HasInnerProduct D] [HasInnerProduct N]

-- AB = 3 * BC, AN = sqrt(2), BN = 4sqrt(2), DN = 2
variables (AB BC AN BN DN : ℝ)
variables (α : ℝ)

axiom Rectangle_ABCD : (AB = 3 * BC) ∧ (AN = sqrt 2) ∧ (BN = 4 * sqrt 2) ∧ (DN = 2)

theorem find_cos_angle_and_area (Rectangle_ABCD : (AB = 3 * BC) ∧ (AN = sqrt 2) ∧ (BN = 4 * sqrt 2) ∧ (DN = 2)) :
  ∃ α : ℝ, cos α = 7 / sqrt 65 ∧ AB * BC = 78 / 5 :=
sorry

end find_cos_angle_and_area_l523_523915


namespace sequence_bound_l523_523975

noncomputable def a : ℕ → ℝ := sorry
def N : ℕ := sorry

axiom a_n_ge_N (n : ℕ) (h : n ≥ N) : a n = 1
axiom recurrence (n : ℕ) (h : n ≥ 2) : a n ≤ a (n - 1) + 2^(-n) * a (2 * n)

theorem sequence_bound (k : ℕ) : a k > 1 - 2^(-k) := sorry

end sequence_bound_l523_523975


namespace intersection_of_A_and_B_l523_523330

def set_A : Set ℝ := {x : ℝ | x^2 - 5 * x + 6 > 0}
def set_B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | x < 1} :=
sorry

end intersection_of_A_and_B_l523_523330


namespace simplify_sqrt7_pow6_l523_523498

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523498


namespace angle_between_vectors_perpendicular_vectors_l523_523819

-- 1. Given |\vec{OA} + \vec{OC}| = sqrt(7), 
--    prove the angle between \vec{OB} and \vec{OC} is π / 6
theorem angle_between_vectors (α : ℝ) (h1 : 0 < α ∧ α < π) (h : (2 + Real.cos α)^2 + (Real.sin α)^2 = 7) :
  Real.arccos ((Real.cos α) / (√((Real.cos α)^2 + (Real.sin α)^2))) = π / 6 :=
sorry

-- 2. Given \vec{AC} ⊥ \vec{BC}, 
--    prove cos α = (1 + sqrt(7)) / 4
theorem perpendicular_vectors (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h : (Real.cos α - 2) * Real.cos α + Real.sin α * (Real.sin α - 2) = 0) :
  Real.cos α = (1 + Real.sqrt 7) / 4 :=
sorry

end angle_between_vectors_perpendicular_vectors_l523_523819


namespace sum_fractions_bounds_l523_523302

theorem sum_fractions_bounds {a b c : ℝ} (h : a * b * c = 1) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧ 
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 :=
  sorry

end sum_fractions_bounds_l523_523302


namespace find_angle_B_l523_523000

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l523_523000


namespace poly_remainder_l523_523276

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end poly_remainder_l523_523276


namespace simplify_sqrt_pow_six_l523_523477

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523477


namespace calculator_result_l523_523173

theorem calculator_result (x : ℝ) (n : ℕ) (hx : x ≠ 0) : 
    let y := iterate (λ z, (z^2)⁻¹) (2 * (n - 1)) (x^4)
    in y = x ^ ((-2)^(n+1)) := 
by 
  sorry

end calculator_result_l523_523173


namespace impossible_coins_l523_523461

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l523_523461


namespace more_oranges_than_apples_l523_523963

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l523_523963


namespace ivan_expected_shots_l523_523942

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l523_523942


namespace no_sin_equal_B_l523_523061

def sin_matrix (A : Matrix (Fin 2) (Fin 2) ℝ) := 
  let A3 := A ⬝ A ⬝ A 
  let A5 := A3 ⬝ A ⬝ A
  A - A3 / !3!.toReal! + A5 / !5!.toReal! -- truncated series for simplicity, in practice, this would need more terms

def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![[1, 1996], [0, 1]]

theorem no_sin_equal_B : ¬ ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), sin_matrix A = B :=
by
  sorry

end no_sin_equal_B_l523_523061


namespace find_n_l523_523979

def regular_tetrahedron_edges := 2
def starting_vertex := "A"
def crawl_choices := 3
def total_distance := 12 
def P : ℕ → ℚ
| 0      := 1
| (n + 1) := 1/3 * (1 - P n)

theorem find_n : ∃ n, P 12 = n / 6561 ∧ n = 132861 := by
  existsi 132861
  simp [P]
  sorry -- computations to show P(12) = 44287/177147 which equals 132861/6561

end find_n_l523_523979


namespace white_truck_percentage_is_17_l523_523745

-- Define the conditions
def total_trucks : ℕ := 50
def total_cars : ℕ := 40
def total_vehicles : ℕ := total_trucks + total_cars

def red_trucks : ℕ := total_trucks / 2
def black_trucks : ℕ := (total_trucks * 20) / 100
def white_trucks : ℕ := total_trucks - red_trucks - black_trucks

def percentage_white_trucks : ℕ := (white_trucks * 100) / total_vehicles

theorem white_truck_percentage_is_17 :
  percentage_white_trucks = 17 :=
  by sorry

end white_truck_percentage_is_17_l523_523745


namespace isosceles_triangle_side_length_l523_523894

theorem isosceles_triangle_side_length (n : ℕ) : 
  (∃ a b : ℕ, a ≠ 4 ∧ b ≠ 4 ∧ (a = b ∨ a = 4 ∨ b = 4) ∧ 
  a^2 - 6*a + n = 0 ∧ b^2 - 6*b + n = 0) → 
  (n = 8 ∨ n = 9) := 
by
  sorry

end isosceles_triangle_side_length_l523_523894


namespace circle_tangent_conditions_l523_523865

-- Define the centers and radii of the given circles and the new circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

variables (O1 O2 O : ℝ × ℝ) (r1 r2 r : ℝ)

-- The problem's conditions:
-- O1 and O2 are the centers of the given circles with radii r1 and r2
-- O is the center of the new circle with radius r
-- The new circle is tangent to the first circle externally and to the second circle internally

def is_tangent_externally (C1 C2 : Circle) := 
  dist C1.center C2.center = C1.radius + C2.radius

def is_tangent_internally (C1 C2 : Circle) := 
  (dist C1.center C2.center).to_real = (C1.radius - C2.radius).to_real

-- The theorem statement that shows the conclusions derived from the conditions
theorem circle_tangent_conditions 
  (c1 c2 c : Circle)
  (h1 : is_tangent_externally c1 c)
  (h2 : is_tangent_internally c2 c)
  (hO1 : c1.center = O1)
  (hO2 : c2.center = O2)
  (hOr1 : c1.radius = r1)
  (hOr2 : c2.radius = r2)
  (hO : c.center = O)
  (hOr : c.radius = r) :
  dist O O1 = (r1 + r2) / 2 ∧ r = (|r1 - r2|) / 2 := 
by sorry

end circle_tangent_conditions_l523_523865


namespace problem_I_problem_II_l523_523863

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Define the complement of A in the universal set U which is ℝ
def complement_U_A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the union of complement_U_A and B
def union_complement_U_A_B : Set ℝ := complement_U_A ∪ B

-- Proof Problem I: Prove that the set A is as specified
theorem problem_I : A = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := sorry

-- Proof Problem II: Prove that the union of the complement of A and B is as specified
theorem problem_II : union_complement_U_A_B = {x : ℝ | x > -1} := sorry

end problem_I_problem_II_l523_523863


namespace max_distance_P_to_circle_center_l523_523119

theorem max_distance_P_to_circle_center 
  (a b : ℝ)
  (line_intersects_circle : ∀ x y : ℝ, (√2 * a * x + b * y = 1 → x^2 + y^2 = 1))
  (right_triangle : ∀ x y : ℝ, (√2 * a * x + b * y = 1 → 
                                ∃ z : ℝ, z ≠ 0 ∧ (z * x, z * y) = (0,1)))
  : ∃ P : ℝ × ℝ, P = (a, b) ∧ dist P (0,1) = √2 + 1 :=
begin
  sorry
end

end max_distance_P_to_circle_center_l523_523119


namespace vertex_x_coordinate_l523_523620

theorem vertex_x_coordinate (a b c : ℝ) :
  (∀ x, x = 0 ∨ x = 4 ∨ x = 7 →
    (0 ≤ x ∧ x ≤ 7 →
      (x = 0 → c = 1) ∧
      (x = 4 → 16 * a + 4 * b + c = 1) ∧
      (x = 7 → 49 * a + 7 * b + c = 5))) →
  (2 * x = 2 * 2 - b / a) ∧ (0 ≤ x ∧ x ≤ 7) :=
sorry

end vertex_x_coordinate_l523_523620


namespace pupils_like_both_l523_523652

theorem pupils_like_both (total_pupils : ℕ) (likes_pizza : ℕ) (likes_burgers : ℕ)
  (total := 200) (P := 125) (B := 115) :
  (P + B - total_pupils) = 40 :=
by
  sorry

end pupils_like_both_l523_523652


namespace expected_flips_N_primeform_l523_523099

theorem expected_flips_N_primeform :
  let N := 2014 in
  let m := 6639 in
  let n := 512 in
  let expected_flips := m / n in
  100 * m + n = 664412 :=
by
  sorry

end expected_flips_N_primeform_l523_523099


namespace min_value_of_P_one_l523_523051

theorem min_value_of_P_one (P : Polynomial ℤ) 
  (h_deg : P.degree = 2015)
  (h_coeff : ∀ n, P.coeff n > 0)
  (ω : ℂ) 
  (h_root_unity : ω^73 = 1)
  (h_sum_zero : ∑ k in range 1 73, P(ω^(34^k)) = 0) : 
  P.eval 1 ≥ 2 := 
sorry

end min_value_of_P_one_l523_523051


namespace transformed_triangle_area_l523_523552

variable {α : Type*}
variables (x1 x2 x3 : α)
variable (g : α → ℝ)

-- Assuming g is defined on {x1, x2, x3}
noncomputable def g_domain : set α := {x1, x2, x3}

-- Initial triangle area condition
def initial_triangle_area (x1 x2 x3 : α) (g : α → ℝ) : Prop :=
  -- This encodes the initial condition that the area of the triangle formed 
  -- by (x1, g(x1)), (x2, g(x2)), and (x3, g(x3)) is 50.
  area_of_triangle (x1, g x1) (x2, g x2) (x3, g x3) = 50

-- Transformed points
def transformed_points := 
  { (x1 / 3, 3 * g x1), (x2 / 3, 3 * g x2), (x3 / 3, 3 * g x3) }

-- Problem statement: Prove that the area of the transformed triangle is 50
theorem transformed_triangle_area
  (h : initial_triangle_area x1 x2 x3 g) :
  area_of_triangle (x1 / 3, 3 * g x1) (x2 / 3, 3 * g x2) (x3 / 3, 3 * g x3) = 50 :=
sorry

end transformed_triangle_area_l523_523552


namespace log_sum_geometric_sequence_l523_523375

variable (a : ℕ → ℝ)

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def fifth_sixth_product (a : ℕ → ℝ) : Prop := 
  a 4 * a 5 = 81

-- Problem statement to prove
theorem log_sum_geometric_sequence 
  (h_geom : is_geometric_sequence a)
  (h_pos : positive_terms a)
  (h_prod : fifth_sixth_product a) :
  (Finset.sum (Finset.range 10) (λ n, Real.logBase 3 (a n))) = 20 :=
by
  sorry

end log_sum_geometric_sequence_l523_523375


namespace closest_point_on_line_l523_523275

open Real

theorem closest_point_on_line (x y : ℝ) (h_line : y = 2 * x - 3) (h_point : (x, y) = (2, 4)) :
    ∃ x_closest y_closest, (y_closest = 2 * x_closest - 3) ∧ sqrt ((x_closest - 2)^2 + (y_closest - 4)^2) = sqrt (((16:ℝ) / 5 - 2)^2 + ((7:ℝ) / 5 - 4)^2) := 
by
  use (16 / 5)
  use (7 / 5)
  split
  sorry
  sorry

end closest_point_on_line_l523_523275


namespace log_3_eq_4_of_infinite_log_pattern_l523_523762

theorem log_3_eq_4_of_infinite_log_pattern (x : ℝ) (h : 0 < x) :
  x = 4 :=
begin
  sorry
end

end log_3_eq_4_of_infinite_log_pattern_l523_523762


namespace arithmetic_sqrt_sqrt_16_l523_523589

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523589


namespace simplify_sqrt_7_pow_6_l523_523496

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523496


namespace polygon_convex_after_operations_l523_523712

theorem polygon_convex_after_operations 
  (P : Polygon)
  (non_convex : ¬(Convex P))
  (non_self_intersecting : ¬(SelfIntersecting P)) :
  ∃ n : ℕ, AfterOperations P n Convex :=
sorry

end polygon_convex_after_operations_l523_523712


namespace angle_B_in_triangle_l523_523036

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523036


namespace extra_yellow_balls_dispatched_l523_523198

/-- Given the conditions:
1. The retailer ordered white and yellow tennis balls such that W = Y.
2. W + Y = 224.
3. The ratio of dispatched white to yellow balls is 8 / 13.
4. The total number of dispatched tennis balls is 224.
Prove: The number of extra yellow balls dispatched is 26.
-/
theorem extra_yellow_balls_dispatched :
  ∃ W Y W' Y' : ℕ,
  (W = Y)
  ∧ (W + Y = 224)
  ∧ (8 * Y' = 13 * W')
  ∧ (W' + Y' = 224)
  ∧ (Y' - Y = 26) :=
begin
  sorry
end

end extra_yellow_balls_dispatched_l523_523198


namespace simplify_sqrt_power_l523_523487

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523487


namespace num_four_digit_with_product_18_l523_523878

-- Definitions based on conditions
def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999
def digit_product (n : ℕ) : ℕ := 
  n.digits.foldl (λ acc d => acc * d) 1

-- Main theorem
theorem num_four_digit_with_product_18 : 
  (count (λ n : ℕ, is_four_digit n ∧ digit_product n = 18)) = 48 :=
sorry

end num_four_digit_with_product_18_l523_523878


namespace mean_of_combined_sets_l523_523120

theorem mean_of_combined_sets (set1 : Fin 7 → ℝ) (set2 : Fin 9 → ℝ) 
  (mean1 : (∑ i, set1 i) / 7 = 16) (mean2 : (∑ i, set2 i) / 9 = 20) : 
  (∑ i, set1 i + ∑ i, set2 i) / 16 = 18.25 :=
sorry

end mean_of_combined_sets_l523_523120


namespace mutter_paid_correct_amount_l523_523049

def total_lagaan_collected : ℝ := 344000
def mutter_land_percentage : ℝ := 0.0023255813953488372
def mutter_lagaan_paid : ℝ := 800

theorem mutter_paid_correct_amount : 
  mutter_lagaan_paid = total_lagaan_collected * mutter_land_percentage := by
  sorry

end mutter_paid_correct_amount_l523_523049


namespace triangle_is_isosceles_l523_523038

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h1 : 2 * sin A * cos B = sin C)
  (h2 : A + B + C = real.pi) : A = B := 
sorry

end triangle_is_isosceles_l523_523038


namespace volume_of_solid_bounded_by_planes_l523_523660

theorem volume_of_solid_bounded_by_planes (a : ℝ) : 
  ∃ v, v = (a ^ 3) / 6 :=
by 
  sorry

end volume_of_solid_bounded_by_planes_l523_523660


namespace find_area_GCHI_l523_523165

variables
  (A B C D E F G H I : Type)
  [parallelogram A B C D]
  [point E : (AD : Type)]
  [point F : (AB : Type)]
  (S_AFIE S_triangle_BGF S_triangle_DEH : Nat)
  (area_AFIE : S_AFIE = 49)
  (area_BGF : S_triangle_BGF = 13)
  (area_DEH : S_triangle_DEH = 35)

theorem find_area_GCHI :
  S_GCHI = 97 :=
by
  sorry

end find_area_GCHI_l523_523165


namespace opposite_face_number_l523_523616

theorem opposite_face_number (sum_faces : ℕ → ℕ → ℕ) (face_number : ℕ → ℕ) :
  (face_number 1 = 6) ∧ (face_number 2 = 7) ∧ (face_number 3 = 8) ∧ 
  (face_number 4 = 9) ∧ (face_number 5 = 10) ∧ (face_number 6 = 11) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 33 + 18) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 35 + 16) →
  (face_number 2 ≠ 9 ∨ face_number 2 ≠ 11) → 
  face_number 2 = 9 ∨ face_number 2 = 11 :=
by
  intros hface_numbers hsum1 hsum2 hnot_possible
  sorry

end opposite_face_number_l523_523616


namespace find_m_and_line_l_max_distance_to_line_l523_523393

noncomputable def pointA_polar : ℝ × ℝ :=
(Real.sqrt(2), 0)

def line_l_polar_eq (rho theta m : ℝ) : Prop :=
rho * Real.sin (theta - (Real.pi / 4)) = m

def pointA_rectangular : ℝ × ℝ :=
(Real.sqrt(2), 0)

noncomputable def line_l_rect_eq (x y m : ℝ) : Prop :=
x - y + Real.sqrt(2) * m = 0

def ellipseC (x y : ℝ) : Prop :=
(y^2 / 2) + x^2 = 1

def distance_to_line (x y a b c : ℝ) : ℝ :=
Real.abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

theorem find_m_and_line_l :
  ∃ m:ℝ, m = 2 ∧ line_l_rect_eq 1 1 2 :=
sorry

theorem max_distance_to_line (m : ℝ) :
  max_dist := 4 + Real.sqrt(6) / 2  :=
sorry

end find_m_and_line_l_max_distance_to_line_l523_523393


namespace janet_semester_average_difference_l523_523398

theorem janet_semester_average_difference :
  let semester1_grades := [90, 80, 70, 100]
  let semester2_average : ℝ := 82
  let semester1_sum := List.sum semester1_grades
  let semester1_count := semester1_grades.length
  let semester1_average := semester1_sum / (semester1_count : ℝ)
  semester1_average - semester2_average = 3 :=
by  
  let semester1_grades := [90, 80, 70, 100]
  let semester2_average : ℝ := 82
  let semester1_sum := List.sum semester1_grades
  let semester1_count := semester1_grades.length
  let semester1_average := semester1_sum / (semester1_count : ℝ)
  show semester1_average - semester2_average = 3
  from sorry

end janet_semester_average_difference_l523_523398


namespace angle_B_in_triangle_l523_523031

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523031


namespace num_factors_of_x2_y_z3_l523_523415

   noncomputable def num_factors (n : ℕ) : ℕ :=
   n.factors.to_finset.card

   theorem num_factors_of_x2_y_z3
   (p1 p2 p3 : ℕ) (hp1 : p1.prime) (hp2 : p2.prime) (hp3 : p3.prime) (h_different : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :
   num_factors ((p1^2)^2 * p2 * (p3^2)^3) = 70 :=
   by
     let x := p1^2
     let y := p2
     let z := p3^2
     let xyz := x^2 * y * z^3
     have h_xyz_eq : xyz = p1^4 * p2 * p3^6 := by
       rw [pow_mul, pow_two, mul_pow]
     have h_num_divisors : num_factors (p1^4 * p2 * p3^6) = (4 + 1) * (1 + 1) * (6 + 1) := by
       rw num_factors_mul
       · exact (degree_eq_one_of_prime hp1).num_divisors
       · exact (degree_eq_one_of_prime hp2).num_divisors
       · exact (degree_eq_one_of_prime hp3).num_divisors
     rw h_xyz_eq at h_num_divisors
     exact h_num_divisors
   
end num_factors_of_x2_y_z3_l523_523415


namespace hyperbola_eccentricity_is_3_l523_523316

noncomputable def parabola_and_hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : ℝ :=
  let M : ℝ × ℝ := (2/3, 2 * real.sqrt 6 / 3)
  let C1_focus : ℝ × ℝ := (1, 0) in
  let C2_foci : ℝ × ℝ := (-1, 0), (1, 0) in
  (C2_foci.2.1 - C2_foci.1.1) / (a / 3)

theorem hyperbola_eccentricity_is_3 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) :
  parabola_and_hyperbola_eccentricity a b h₀ h₁ = 3 :=
sorry

end hyperbola_eccentricity_is_3_l523_523316


namespace thirds_side_length_valid_l523_523899

theorem thirds_side_length_valid (x : ℝ) (h1 : x > 5) (h2 : x < 13) : x = 12 :=
sorry

end thirds_side_length_valid_l523_523899


namespace train_cross_platform_time_l523_523724

noncomputable def kmph_to_mps (s : ℚ) : ℚ :=
  (s * 1000) / 3600

theorem train_cross_platform_time :
  let train_length := 110
  let speed_kmph := 52
  let platform_length := 323.36799999999994
  let speed_mps := kmph_to_mps 52
  let total_distance := train_length + platform_length
  let time := total_distance / speed_mps
  time = 30 := 
by
  sorry

end train_cross_platform_time_l523_523724


namespace line_y_axis_intersect_l523_523186

theorem line_y_axis_intersect (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3 ∧ y1 = 27) (h2 : x2 = -7 ∧ y2 = -1) :
  ∃ y : ℝ, (∀ x : ℝ, y = (y2 - y1) / (x2 - x1) * (x - x1) + y1) ∧ y = 18.6 :=
by
  sorry

end line_y_axis_intersect_l523_523186


namespace rounded_diff_greater_l523_523382

variable (x y ε : ℝ)
variable (h1 : x > y)
variable (h2 : y > 0)
variable (h3 : ε > 0)

theorem rounded_diff_greater : (x + ε) - (y - ε) > x - y :=
  by
  sorry

end rounded_diff_greater_l523_523382


namespace log_order_l523_523693

theorem log_order {x y z : ℝ} (h1 : 0 < x ∧ x < 1) (h2 : 1 < y) (h3 : ∀ x, log x < 0) : log x < x ∧ x < y :=
by
  sorry

end log_order_l523_523693


namespace range_of_a_l523_523290

noncomputable def f (x a : ℝ) : ℝ := x + a * real.sin x

noncomputable def f_prime (x a : ℝ) : ℝ := 1 + a * real.cos x

theorem range_of_a
    (a : ℝ)
    (h : ∀ x : ℝ, f_prime x a ≥ 0) :
    -1 ≤ a ∧ a ≤ 1 :=
by
  -- Proof needs to be provided here
  sorry

end range_of_a_l523_523290


namespace find_regular_time_limit_l523_523710

noncomputable def regular_time_limit (R : ℕ) : Prop :=
  let regular_pay := 3 * R
  let overtime_pay := 6 * 12
  let total_pay := regular_pay + overtime_pay
  total_pay = 192 ∧ R = 40

theorem find_regular_time_limit : ∃ R : ℕ, regular_time_limit R :=
by
  use 40
  unfold regular_time_limit
  simp [Nat.mul_succ, add_assoc, mul_add]
  sorry

end find_regular_time_limit_l523_523710


namespace function_range_l523_523619

-- Proving the range of the given function f(x) on the interval [-1, 2]
theorem function_range (a b : ℝ) (h1 : f(x) = a*x^2 + b*x - 2)
  (h2 : ∀ x ∈ Icc (-2) 2, f(-x) = f(x))
  (h3 : 1 + a = -2) :
  set.range (λ x : ℝ, a * x^2 + b * x - 2) ∩ Icc (-1 : ℝ) 2 = Icc (-14 : ℝ) (-2) := 
sorry

end function_range_l523_523619


namespace ratio_of_radii_of_cylinders_l523_523742

theorem ratio_of_radii_of_cylinders
  (r_V r_B h_V h_B : ℝ)
  (h1 : h_V = 1/2 * h_B)
  (h2 : π * r_B^2 * h_B / 2  = 4)
  (h3 : π * r_V^2 * h_V = 16) :
  r_V / r_B = 2 := 
by 
  sorry

end ratio_of_radii_of_cylinders_l523_523742


namespace area_of_triangle_l523_523263

def point3D := (ℝ × ℝ × ℝ)

def dist (a b : point3D) : ℝ := 
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

theorem area_of_triangle : 
  let A : point3D := (1, 8, 11)
  let B : point3D := (0, 7, 7)
  let C : point3D := (-3, 10, 7)
  dist A B^2 + dist B C^2 = dist A C^2 → 
  1/2 * dist A B * dist B C = 9 := 
by 
  sorry

end area_of_triangle_l523_523263


namespace mul_inv_mod_301_l523_523747

theorem mul_inv_mod_301 :
  ∃ (a : ℤ), a ≡ 29 [MOD 301] ∧ (203 * a) % 301 = 1 :=
sorry

end mul_inv_mod_301_l523_523747


namespace coin_toss_sequence_count_l523_523378

theorem coin_toss_sequence_count :
  (fact 14) / ((fact 2) * (fact 3) * (fact 4) * (fact 5)) = 2522520 :=
by
  sorry

end coin_toss_sequence_count_l523_523378


namespace sanjay_homework_l523_523730

theorem sanjay_homework
  (h_monday : 3 / 5)
  (h_tuesday : 1 / 3) :
  ∀ x y : ℚ, x = 3 / 5 ∧ y = 1 / 3 →
  x * y = 4 / 15 :=
by
  -- Define the fractions completed and remaining
  let remaining_after_monday := 2 / 5
  let completed_on_tuesday := 1 / 3 * remaining_after_monday
  
  -- Calculate the remaining homework after Tuesday
  let remaining_after_tuesday := remaining_after_monday - completed_on_tuesday
  
  -- Prove the final fraction to be completed is as stated
  have : remaining_after_tuesday = 4 / 15 := sorry
  exact this

end sanjay_homework_l523_523730


namespace count_div_by_4_last_two_digits_l523_523075

theorem count_div_by_4_last_two_digits : 
  ∃ n, n = 25 ∧ ∀ k, (0 ≤ k ∧ k < 100) → (k % 4 = 0 ↔ k % 100 / 100 ∈ (set.range (λ n, 4 * n) ∩ set.Icc 0 99)) :=
by sorry

end count_div_by_4_last_two_digits_l523_523075


namespace sum_of_distances_and_area_l523_523248

def point := (ℝ × ℝ)

def P : point := (5,1)
def A : point := (0,0)
def B : point := (12,0)
def C : point := (4,4)

def distance (p1 p2 : point) : ℝ := 
  Real.sqrt((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def area (A B C : point) : ℝ := 
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem sum_of_distances_and_area :
  distance P A + distance P B + distance P C = Real.sqrt 26 + 5 * Real.sqrt 2 + Real.sqrt 10 ∧
  area A B C = 24 ∧
  let p := 1; let q := 26; let r := 5; let s := 2 in p + r = 6 := 
by
  sorry

end sum_of_distances_and_area_l523_523248


namespace James_sold_40_percent_of_toys_l523_523954

theorem James_sold_40_percent_of_toys (number_of_toys total_cost_per_toy total_selling_per_toy profit : ℤ) (number_of_toys = 200) (total_cost_per_toy = 20) (total_selling_per_toy = 30) (profit = 800) : 
  ((total_selling_per_toy - total_cost_per_toy) * number_of_toys / 200) = 40 := 
sorry

end James_sold_40_percent_of_toys_l523_523954


namespace num_four_digit_with_product_18_l523_523880

-- Definitions based on conditions
def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999
def digit_product (n : ℕ) : ℕ := 
  n.digits.foldl (λ acc d => acc * d) 1

-- Main theorem
theorem num_four_digit_with_product_18 : 
  (count (λ n : ℕ, is_four_digit n ∧ digit_product n = 18)) = 48 :=
sorry

end num_four_digit_with_product_18_l523_523880


namespace graph_represents_y_abs_f_add_3_l523_523992

/-- Define the piecewise function f(x) --/
def f (x : ℝ) : ℝ :=
if x ≥ -3 ∧ x ≤ 0 then -2 - x else 
if x ≥ 0 ∧ x ≤ 2 then sqrt (4 - (x - 2)^2) - 2 else 
if x ≥ 2 ∧ x ≤ 3 then 2 * (x - 2) else 0

/-- Prove that for the positive constant a = 3, the graph y = |f(x) + 3| is represented by option A --/
theorem graph_represents_y_abs_f_add_3 (a : ℝ) (h1 : a = 3) :
  let y := λ x, abs (f x + 3) in 
  ∃ A : (ℝ → ℝ), (∀ x, y x = A x) := 
sorry

end graph_represents_y_abs_f_add_3_l523_523992


namespace angle_B_in_triangle_l523_523037

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523037


namespace johns_cycling_speed_needed_l523_523403

theorem johns_cycling_speed_needed 
  (swim_speed : Float := 3)
  (swim_distance : Float := 0.5)
  (run_speed : Float := 8)
  (run_distance : Float := 4)
  (total_time : Float := 3)
  (bike_distance : Float := 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 60 / 7 := 
  by
  sorry

end johns_cycling_speed_needed_l523_523403


namespace arithmetic_square_root_of_sqrt_16_l523_523583

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523583


namespace find_a_range_l523_523820

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (-1 / 3 < x ∧ x < 3) ↔ (- (a + 1) / 3 < x ∧ x < a + 1)

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, (4x < 4ax^2 + 1) ∨ (4ax^2 - 4x + 1 > 0)

theorem find_a_range (a : ℝ) :
  (proposition_p a ∨ proposition_q a) → a ∈ set.Ioi (1 : ℝ) :=
by
  sorry

end find_a_range_l523_523820


namespace simplify_sqrt_pow_six_l523_523470

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523470


namespace number_of_distinct_four_digit_integers_with_product_18_l523_523875

theorem number_of_distinct_four_digit_integers_with_product_18 : 
  ∃ l : List (List ℕ), (∀ d ∈ l, d.length = 4 ∧ d.product = 18) ∧
    l.foldr (λ x acc, acc + Multiset.card (x.toMultiset)) 0 = 36 := 
  sorry

end number_of_distinct_four_digit_integers_with_product_18_l523_523875


namespace angle_B_eq_3pi_over_10_l523_523025

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523025


namespace expression_eqn_l523_523190

theorem expression_eqn (a : ℝ) (E : ℝ → ℝ)
  (h₁ : -6 * a^2 = 3 * (E a + 2))
  (h₂ : a = 1) : E a = -2 * a^2 - 2 :=
by
  sorry

end expression_eqn_l523_523190


namespace find_w_l523_523322

def sin_func (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 6)

theorem find_w
  (w : ℝ)
  (hw : w > 0)
  (h_dist : ∀ x, ∃ c ∈ Set.Icc (-Real.pi/2) (Real.pi/2), (sin_func w x) = (sin_func w (x + c)) ∧ c = Real.pi / 3) :
  w = 3 / 2 := by
  sorry

end find_w_l523_523322


namespace arithmetic_sqrt_sqrt_16_l523_523588

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523588


namespace number_of_boys_l523_523170

-- Definitions for the conditions
def total_children : ℕ := 13
def is_boy (n : ℕ) : Prop := sorry -- function to determine if the nth child is a boy

-- The sequence of statements follows an alternating pattern:
def statements (n : ℕ) : Prop :=
  if n % 2 = 0 then "Most of us are boys." else "Most of us are girls."

-- The main theorem to prove
theorem number_of_boys (boys_count : ℕ) (girls_count : ℕ) 
  (h_sum : boys_count + girls_count = total_children)
  (h_majority: boys_count ≥ 7 ∨ girls_count ≥ 7) :
  boys_count = 7 :=
by
  sorry

end number_of_boys_l523_523170


namespace large_block_volume_correct_l523_523703

def normal_block_volume (w d l : ℝ) : ℝ := w * d * l

def large_block_volume (w d l : ℝ) : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem large_block_volume_correct (w d l : ℝ) (h : normal_block_volume w d l = 3) :
  large_block_volume w d l = 36 :=
by sorry

end large_block_volume_correct_l523_523703


namespace radius_of_circle_l523_523635

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l523_523635


namespace arithmetic_sqrt_sqrt_16_l523_523593

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523593


namespace simplify_sqrt_seven_pow_six_proof_l523_523517

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523517


namespace angle_of_inclination_of_line_l523_523787

theorem angle_of_inclination_of_line (x y : ℝ) (h : x + y + 3 = 0) : 
  ∃ θ : ℝ, θ = 3 * Real.pi / 4 ∧ ∀ k θ, tan θ = k ∧ k = -1 :=  
by 
  sorry

end angle_of_inclination_of_line_l523_523787


namespace log_infinite_expression_pos_l523_523767

theorem log_infinite_expression_pos :
  let x := real.logb 3 (81 + real.logb 3 (81 + real.logb 3 (81 + ...)))
  in x = 4 :=
sorry

end log_infinite_expression_pos_l523_523767


namespace four_disjoint_subsets_with_equal_sums_l523_523436

theorem four_disjoint_subsets_with_equal_sums :
  ∀ (S : Finset ℕ), 
  (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧ S.card = 117 → 
  ∃ A B C D : Finset ℕ, 
    (A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ D ⊆ S) ∧ 
    (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ A ∩ D = ∅ ∧ B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅) ∧ 
    (A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = D.sum id) := by
  sorry

end four_disjoint_subsets_with_equal_sums_l523_523436


namespace simplify_sqrt_seven_pow_six_proof_l523_523510

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523510


namespace sock_ratio_l523_523439

theorem sock_ratio (b : ℕ) (x : ℕ) (hx_pos : 0 < x)
  (h1 : 5 * x + 3 * b * x = k) -- Original cost is 5x + 3bx
  (h2 : b * x + 15 * x = 2 * k) -- Interchanged cost is doubled
  : b = 1 :=
by sorry

end sock_ratio_l523_523439


namespace complement_N_in_U_l523_523864

open Set

variable U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
variable N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem complement_N_in_U :
  complement N ∩ U = {x | (-3 ≤ x ∧ x < 0) ∨ (2 ≤ x ∧ x ≤ 3)} :=
by
  sorry

end complement_N_in_U_l523_523864


namespace mutter_lagaan_payment_l523_523047

-- Conditions as definitions
def total_lagaan_collected : ℝ := 344000
def mutter_percentage_of_total_taxable_land : ℝ := 0.23255813953488372 / 100

-- Proof statement
theorem mutter_lagaan_payment : (mutter_percentage_of_total_taxable_land * total_lagaan_collected) = 800 := by
  sorry

end mutter_lagaan_payment_l523_523047


namespace intersection_A_B_l523_523823

/-- Definitions for the sets A and B --/
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

-- Theorem statement regarding the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {1} :=
by sorry

end intersection_A_B_l523_523823


namespace lamp_switching_goal_achievable_l523_523064

theorem lamp_switching_goal_achievable (n : ℕ) (h : n > 0) :
  (∃ (rounds : list (fin n → bool)), 
    ∀ i j : fin n, (i ≠ j → rounds.map (λr, r i ≠ r j).count_true % 2 = 1)) ↔ n ≠ 2 :=
by sorry

end lamp_switching_goal_achievable_l523_523064


namespace simplify_sqrt_pow_six_l523_523474

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523474


namespace pythagorean_triple_9_12_15_l523_523207

theorem pythagorean_triple_9_12_15 : ∃ a b c : ℕ, a = 9 ∧ b = 12 ∧ c = 15 ∧ (a * a + b * b = c * c) :=
by 
  existsi (9, 12, 15)
  split
  rfl
  split
  rfl
  split
  rfl
  sorry

end pythagorean_triple_9_12_15_l523_523207


namespace triangle_perimeter_l523_523900

theorem triangle_perimeter (a : ℕ) (h_even : a % 2 = 0) (h1 : 2 < a) (h2 : a < 14) : 
  a = 10 → 6 + 8 + a = 24 :=
by
  intro h_a
  rw h_a
  norm_num

end triangle_perimeter_l523_523900


namespace intersection_at_one_point_l523_523232

theorem intersection_at_one_point (b : ℝ) :
  (∃ x : ℝ, bx^2 + 5x + 3 = -2x - 3) ∧ (∀ x₁ x₂ : ℝ, bx^2 + 5x + 3 = -2x - 3 → x₁ = x₂) →
  b = 49 / 24 := by
  sorry

end intersection_at_one_point_l523_523232


namespace Ivan_expected_shots_l523_523949

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l523_523949


namespace max_ahead_distance_l523_523202

noncomputable def distance_run_by_alex (initial_distance ahead1 ahead_max_runs final_ahead : ℝ) : ℝ :=
  initial_distance + ahead1 + ahead_max_runs + final_ahead

theorem max_ahead_distance :
  let initial_distance := 200
  let ahead1 := 300
  let final_ahead := 440
  let total_road := 5000
  let distance_remaining := 3890
  let distance_run_alex := total_road - distance_remaining
  ∃ X : ℝ, distance_run_by_alex initial_distance ahead1 X final_ahead = distance_run_alex ∧ X = 170 :=
by
  intro initial_distance ahead1 final_ahead total_road distance_remaining distance_run_alex
  use 170
  simp [initial_distance, ahead1, final_ahead, total_road, distance_remaining, distance_run_alex, distance_run_by_alex]
  sorry

end max_ahead_distance_l523_523202


namespace total_cost_for_trip_l523_523883

def cost_of_trip (students : ℕ) (teachers : ℕ) (seats_per_bus : ℕ) (cost_per_bus : ℕ) (toll_per_bus : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_required := (total_people + seats_per_bus - 1) / seats_per_bus -- ceiling division
  let total_rent_cost := buses_required * cost_per_bus
  let total_toll_cost := buses_required * toll_per_bus
  total_rent_cost + total_toll_cost

theorem total_cost_for_trip
  (students : ℕ := 252)
  (teachers : ℕ := 8)
  (seats_per_bus : ℕ := 41)
  (cost_per_bus : ℕ := 300000)
  (toll_per_bus : ℕ := 7500) :
  cost_of_trip students teachers seats_per_bus cost_per_bus toll_per_bus = 2152500 := by
  sorry -- Proof to be filled

end total_cost_for_trip_l523_523883


namespace video_games_spent_l523_523356

def total_allowance : ℝ := 60
def fraction_books : ℝ := 1 / 4
def fraction_snacks : ℝ := 1 / 6
def fraction_toys : ℝ := 2 / 5

theorem video_games_spent :
  total_allowance - (fraction_books * total_allowance + fraction_snacks * total_allowance + fraction_toys * total_allowance) = 11 :=
by
  sorry

end video_games_spent_l523_523356


namespace arithmetic_seq_sum_l523_523982

theorem arithmetic_seq_sum (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S(3) = 9) 
  (h₂ : S(6) = 36) 
  (h₃ : ∀ n, S(n + 1) = S(n) + a(n + 1)) :
  a(7) + a(8) + a(9) = 45 :=
by
  sorry

end arithmetic_seq_sum_l523_523982


namespace no_solution_for_given_m_l523_523152

theorem no_solution_for_given_m (x m : ℝ) (h1 : x ≠ 5) (h2 : x ≠ 8) :
  (∀ y : ℝ, (y - 2) / (y - 5) = (y - m) / (y - 8) → false) ↔ m = 5 :=
by
  sorry

end no_solution_for_given_m_l523_523152


namespace total_spent_l523_523203

def spending (A B C : ℝ) : Prop :=
  (A = (13 / 10) * B) ∧
  (C = (4 / 5) * B) ∧
  (A = C + 15)

theorem total_spent (A B C : ℝ) (h : spending A B C) : A + B + C = 93 :=
by
  sorry

end total_spent_l523_523203


namespace angle_B_eq_3pi_over_10_l523_523026

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523026


namespace max_min_g_l523_523313

noncomputable def f_range := set.Icc (3/8) (4/9)

def g (x : ℝ) : ℝ := x + real.sqrt (1 - 2 * x)

theorem max_min_g :
  ∃ (min max : ℝ), min = 7 / 9 ∧ max = 7 / 8 ∧
  ∀ x ∈ f_range, min ≤ g x ∧ g x ≤ max :=
  sorry

end max_min_g_l523_523313


namespace impossible_coins_l523_523464

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l523_523464


namespace nine_digit_number_conditions_l523_523438

def nine_digit_number := 900900000

def remove_second_digit (n : ℕ) : ℕ := n / 100000000 * 10000000 + n % 10000000
def remove_third_digit (n : ℕ) : ℕ := n / 10000000 * 1000000 + n % 1000000
def remove_ninth_digit (n : ℕ) : ℕ := n / 10

theorem nine_digit_number_conditions :
  (remove_second_digit nine_digit_number) % 2 = 0 ∧
  (remove_third_digit nine_digit_number) % 3 = 0 ∧
  (remove_ninth_digit nine_digit_number) % 9 = 0 :=
by
  -- Proof steps would be included here.
  sorry

end nine_digit_number_conditions_l523_523438


namespace impossible_coins_l523_523465

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l523_523465


namespace find_b_l523_523840

theorem find_b (b : ℝ) (h : ∃ (f_inv : ℝ → ℝ), (∀ x y, f_inv (2^x + b) = y) ∧ f_inv 5 = 2) :
    b = 1 := by
  sorry

end find_b_l523_523840


namespace arithmetic_sqrt_of_sqrt_16_l523_523561

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523561


namespace correct_proposition_l523_523309

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f (-x) = f x)
variable (h_increasing : ∀ x y : ℝ, x < y → y < 0 → f x < f y)

theorem correct_proposition : f (-1) > f 2 :=
  by {
    have h1 : f (-1) = f 1, from h_even 1,
    have h2 : f (-2) = f 2, from h_even 2,
    have : f (-2) < f (-1), from h_increasing (-2) (-1) (by linarith) (by linarith),
    linarith,
  }

end correct_proposition_l523_523309


namespace max_value_proof_l523_523974

def max_value (a b : ℕ) (λ : ℝ) [irrational λ] : ℝ :=
  a * ceil (b * λ) - b * floor (a * λ)

theorem max_value_proof {a b : ℕ} (hpos_a : a > 0) (hpos_b : b > 0) (λ : ℝ) [irrational λ] : 
  ∃ val, val = max_value a b λ ∧ val = a + b - gcd a b := 
by
  sorry

end max_value_proof_l523_523974


namespace namjoons_position_l523_523094

theorem namjoons_position
(seokjins_position : ℕ)
(namjoon_behind : ℕ)
(h1 : seokjins_position = 5)
(h2 : namjoon_behind = 6) :
  seokjins_position + namjoon_behind = 11 :=
by
  rw [h1, h2]
  rfl

end namjoons_position_l523_523094


namespace slope_and_midpoint_l523_523247

theorem slope_and_midpoint (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 1) (hy1 : y1 = -7) (hx2 : x2 = -4) (hy2 : y2 = 3) :
  let m := (y2 - y1) / (x2 - x1),
      Mx := (x1 + x2) / 2,
      My := (y1 + y2) / 2
  in m = -2 ∧ Mx = -3/2 ∧ My = -2 :=
by
  simp [hx1, hy1, hx2, hy2]
  sorry

end slope_and_midpoint_l523_523247


namespace problem_statement_l523_523855

theorem problem_statement (ω : ℝ) (h1 : ω ∈ set.Ioo 0 2):
  (∀ x : ℝ, (f x = 4 * sin (ω * x - π / 4) * cos (ω * x)) 
            ∧ (∀ x : ℝ, has_deriv_at (f x) (f' x) x)
            ∧ (f' (π / 4) = 0)
            ∧ (∃ c : ℝ, f' c = 0 ∧ c ∈ set.Ioo (0 : ℝ) (2 : ℝ))) →
  (smallest_positive_period f = 2 * π / 3) ∧ 
  (∃ α : ℝ, α ∈ set.Ioo 0 (π / 2) ∧ 
   g α = (2 * sin (α - π / 6) - sqrt 2) 
   ∧ g α = 4 / 3 - sqrt 2 
   ∧ cos α = (sqrt 15 - 2) / 6) :=
sorry

end problem_statement_l523_523855


namespace number_of_students_l523_523912

theorem number_of_students (N : ℕ) 
  (h1 : 3 * 95 + (N - 8) * 45 = N * 42) : 
  N = 25 := 
begin
  sorry
end

end number_of_students_l523_523912


namespace arithmetic_square_root_of_sqrt_16_l523_523573

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523573


namespace cos_alpha_solution_l523_523284

variable (α : ℝ)

def condition1 : Prop := sin (α + π/3) + sin α = -4 * sqrt 3 / 5
def condition2 : Prop := -π/2 < α ∧ α < 0
def target : Prop := cos α = (3 * sqrt 3 - 4) / 10

theorem cos_alpha_solution (h1 : condition1 α) (h2 : condition2 α) : target α := 
  sorry

end cos_alpha_solution_l523_523284


namespace triangle_ABC_right_and_perimeter_l523_523374

noncomputable def A : ℝ × ℝ := (-2, 3)
noncomputable def B : ℝ × ℝ := (5, 3)
noncomputable def C : ℝ × ℝ := (5, -2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem triangle_ABC_right_and_perimeter :
  let AB := distance A B
  let BC := distance B C
  let AC := distance A C
  AB^2 + BC^2 = AC^2 ∧ AB + BC + AC = 12 + real.sqrt 74 :=
by
  sorry

end triangle_ABC_right_and_perimeter_l523_523374


namespace function_equality_l523_523406

theorem function_equality (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f n < f (n + 1) )
  (h2 : f 2 = 2)
  (h3 : ∀ m n : ℕ, f (m * n) = f m * f n) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_equality_l523_523406


namespace parabola_properties_l523_523388

noncomputable def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties
  (a b c t m n x₀ : ℝ)
  (ha : a > 0)
  (h1 : parabola a b c 1 = m)
  (h4 : parabola a b c 4 = n)
  (ht : t = -b / (2 * a))
  (h3ab : 3 * a + b = 0) 
  (hmnc : m < c ∧ c < n)
  (hx₀ym : parabola a b c x₀ = m) :
  m < n ∧ (1 / 2) < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 :=
  sorry

end parabola_properties_l523_523388


namespace simplify_sqrt_power_l523_523483

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523483


namespace max_value_of_sum_of_sides_l523_523369

theorem max_value_of_sum_of_sides 
  (a b c A B C : ℝ) 
  (h1 : A + B + C = π)
  (h2 : a^2 = b^2 + c^2 - 2 * b * c * cos A)
  (h3 : b^2 = c^2 + a^2 - 2 * c * a * cos B)
  (h4 : c^2 = a^2 + b^2 - 2 * a * b * cos C)
  (h5 : b * cos C + c * cos B = c * sin A)
  (h6 : sin A ≠ 0)
  : ∃ A B : ℝ, max_value_of_sum_of_sides a b c A B C = sqrt(2) :=
  begin
    sorry
  end

end max_value_of_sum_of_sides_l523_523369


namespace tan_identity_15_30_l523_523741

theorem tan_identity_15_30 :
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  have h1 := Real.tan_pi_div_four,
  have h2 : Real.tan (Real.pi / 12 + Real.pi / 6) = 1 := by rw [Real.add_div, Real.tan_pi_div_four],
  have h3 : Real.pi / 12 + Real.pi / 6 = Real.pi / 4 := by norm_num,
  sorry

end tan_identity_15_30_l523_523741


namespace find_angle_B_l523_523002

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l523_523002


namespace triangle_problem_l523_523932

noncomputable def a (A : ℝ) (b c : ℝ) : ℝ := real.sqrt(b^2 + c^2 - 2*b*c*real.cos A)
noncomputable def area (b c A : ℝ) : ℝ := 0.5 * b * c * real.sin A

theorem triangle_problem
  (A : ℝ) (b : ℝ) (area_value : ℝ)
  (hA : A = real.pi / 3)
  (hb: b = 1)
  (hArea: area b 1 A = real.sqrt 3) :
  (∃ c : ℝ, (∃ a : ℝ, ((a + b + c) / (real.sin A + real.sin (real.asin (a * real.sin A / b)) + real.sin (real.asin (c * real.sin A / b))) = 2 * real.sqrt (39) / 3))) :=
begin
  intros,
  sorry
end

end triangle_problem_l523_523932


namespace max_value_of_geometric_sequence_l523_523295

def geom_arith_seq_max (a : ℕ → ℕ) :=
  (∀ n : ℕ, a n = 2 ^ (5 - n)) ∧
  (∀ n : ℕ, a 2 * a 5 = 2 * a 3) ∧
  (a 4 + 2 * a 7 = 2 * (5 / 4)) →
  ∃ n : ℕ, a 1 * a 2 * a 3 * a 4 * a 5 = 1024

theorem max_value_of_geometric_sequence : 
  geom_arith_seq_max (λ n, 2 ^ (5 - n)) :=
  by
    sorry

end max_value_of_geometric_sequence_l523_523295


namespace Kim_morning_routine_time_l523_523961

theorem Kim_morning_routine_time :
  let senior_employees := 3
  let junior_employees := 3
  let interns := 3

  let senior_overtime := 2
  let junior_overtime := 3
  let intern_overtime := 1
  let senior_not_overtime := senior_employees - senior_overtime
  let junior_not_overtime := junior_employees - junior_overtime
  let intern_not_overtime := interns - intern_overtime

  let coffee_time := 5
  let email_time := 10
  let supplies_time := 8
  let meetings_time := 6
  let reports_time := 5

  let status_update_time := 3 * senior_employees + 2 * junior_employees + 1 * interns
  let payroll_update_time := 
    4 * senior_overtime + 2 * senior_not_overtime +
    3 * junior_overtime + 1 * junior_not_overtime +
    2 * intern_overtime + 0.5 * intern_not_overtime
  let daily_tasks_time :=
    4 * senior_employees + 3 * junior_employees + 2 * interns

  let total_time := coffee_time + status_update_time + payroll_update_time + daily_tasks_time + email_time + supplies_time + meetings_time + reports_time
  total_time = 101 := by
  sorry

end Kim_morning_routine_time_l523_523961


namespace square_shaded_fraction_l523_523910

-- Define the conditions and theorem
theorem square_shaded_fraction (s : ℝ) (h₁ : s > 0) :
  let R := (s / 2, 0)
  let S := (0, s / 2)
  let area_square := s ^ 2
  let area_triangle := 1 / 2 * (s / 2) * (s / 2)
  let area_two_triangles := 2 * area_triangle
  let shaded_area := area_square - area_two_triangles 
in shaded_area / area_square = 3 / 4 :=
by
  sorry

end square_shaded_fraction_l523_523910


namespace simplify_sqrt7_pow6_l523_523499

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523499


namespace simplify_sqrt_power_l523_523481

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523481


namespace S_9_value_l523_523304

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

theorem S_9_value (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : a 3 + a 4 + a 8 = 9) : 
  (a 4 + a 4 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 > 9) false := 
begin
  sorry
end

end S_9_value_l523_523304


namespace vet_donation_correct_l523_523211

def vet_fee (animal : String) : Nat :=
match animal with
| "dog" => 15
| "cat" => 13
| "rabbit" => 10
| "parrot" => 12
| _ => 0

def adoptions (animal : String) : Nat :=
match animal with
| "dog" => 8
| "cat" => 3
| "rabbit" => 5
| "parrot" => 2
| _ => 0

def total_fees : Nat :=
(vet_fee "dog" * adoptions "dog") + 
(vet_fee "cat" * adoptions "cat") + 
(vet_fee "rabbit" * adoptions "rabbit") + 
(vet_fee "parrot" * adoptions "parrot")

def donation_rounded : Nat :=
Nat.round (total_fees / 3 : Float)
noncomputable def veterinarians_donation : Nat :=
Float.round (total_fees / 3 : Float)

theorem vet_donation_correct : veterinarians_donation = 78 :=
by sorry

end vet_donation_correct_l523_523211


namespace angle_ODB_eq_angle_OEC_l523_523227

open Real

def triangle (A B C : Point): Prop := -- Definition of a triangle
  true

def incircle_tangent (ABC : Triangle) (D E : Point) (AB AC : Line) : Prop :=
  -- Conditions for the in-circle being tangent to AB at D and to AC at E
  true

def circumcenter (B C I : Point) (O : Point) : Prop :=
  -- Definition of circumcenter
  true

def angle (A B C : Point) : Real := 
  -- Definition of an angle between three points
  0

variables {A B C I D E O : Point}
variables [triangle A B C]

theorem angle_ODB_eq_angle_OEC :
  incircle_tangent triangle A B C D E AB AC →
  circumcenter B C I O →
  angle O D B = angle O E C :=
by
  sorry

end angle_ODB_eq_angle_OEC_l523_523227


namespace wire_weight_l523_523718

theorem wire_weight (w : ℕ → ℕ) (h_proportional : ∀ (x y : ℕ), w (x + y) = w x + w y) : 
  (w 25 = 5) → w 75 = 15 :=
by
  intro h1
  sorry

end wire_weight_l523_523718


namespace proof_value_l523_523834

noncomputable def a : ℝ := 1 / 2

def f (x : ℝ) : ℝ :=
  if x < 0 then a ^ x else Real.log x / Real.log a

theorem proof_value :
  f (1 / 4) + f (Real.log 1 / Real.log 6) = 8 := by
  sorry

end proof_value_l523_523834


namespace parabola_vertex_l523_523614

-- Define the condition: the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4 * y + 3 * x + 1 = 0

-- Define the statement: prove that the vertex of the parabola is (1, -2)
theorem parabola_vertex :
  parabola_equation 1 (-2) :=
by
  sorry

end parabola_vertex_l523_523614


namespace find_sum_abc_l523_523421

noncomputable def f (x a b c : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + c

theorem find_sum_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (habc_distinct : a ≠ b) (hfa : f a a b c = a^3) (hfb : f b a b c = b^3) : 
  a + b + c = 18 := 
sorry

end find_sum_abc_l523_523421


namespace find_angle_B_l523_523023

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523023


namespace number_of_distinct_four_digit_integers_with_product_18_l523_523877

theorem number_of_distinct_four_digit_integers_with_product_18 : 
  ∃ l : List (List ℕ), (∀ d ∈ l, d.length = 4 ∧ d.product = 18) ∧
    l.foldr (λ x acc, acc + Multiset.card (x.toMultiset)) 0 = 36 := 
  sorry

end number_of_distinct_four_digit_integers_with_product_18_l523_523877


namespace coins_with_specific_probabilities_impossible_l523_523444

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l523_523444


namespace find_x0_l523_523854

noncomputable def f (x : ℝ) : ℝ := (√2) * x^2 - 8 * x + 13
noncomputable def f' (x : ℝ) : ℝ := 2 * (√2) * x - 8

theorem find_x0 : ∃ (x0 : ℝ), f'(x0) = 4 ∧ x0 = 3 * (√2) :=
by
  sorry

end find_x0_l523_523854


namespace measure_angle_RQP_l523_523920

theorem measure_angle_RQP {P Q R S : Type} [EuclideanGeometry P Q R S]
  (hP_on_RS : P ∈ (line RS))
  (hBisect_SQR : angleBisector Q P (angle SQR))
  (hBisect_PRS : angleBisector R P (angle PRS))
  (hPQ_eq_PR : distance PQ = distance PR)
  (hAngle_RSQ : ∠RSQ = 4 * x)
  (hAngle_QRP : ∠QRP = 5 * x)
  : ∠RQP = 720 / 11 :=
sorry

end measure_angle_RQP_l523_523920


namespace intersection_for_m_eq_2_subset_condition_l523_523301

def A (x : ℝ) : Set ℝ := { y | ∃ x, y = Real.sqrt (3 - 2 * x) ∧ x ∈ Set.Icc (-13 / 2) (3 / 2) }
def B (m : ℝ) : Set ℝ := Set.Icc (1 - m) (m + 1)

theorem intersection_for_m_eq_2 : A ∩ B 2 = Set.Icc 0 3 :=
by sorry

theorem subset_condition (m : ℝ) : B m ⊆ A → m ≤ 1 :=
by sorry

end intersection_for_m_eq_2_subset_condition_l523_523301


namespace max_c_value_equality_conditions_for_n_l523_523269

open Real Nat

/-- 
Prove the maximum value of c such that for all natural numbers n, 
the fractional part of n * sqrt(2) is at least c/n, and 
determine n for which the equality holds. 
-/
theorem max_c_value (n : ℕ) (c : ℝ) : 
  (∀ n : ℕ, fract (n * sqrt 2) ≥ c / n) ↔ c = sqrt 2 / 4 :=
sorry

/-- 
For the proven value of c, identify the natural numbers n for 
which the fractional part of n * sqrt(2) equals c / n. 
-/
theorem equality_conditions_for_n (n : ℕ) : 
  fract (n * sqrt 2) = (sqrt 2 / 4) / n ↔ n * sqrt 2 - ⌊n * sqrt 2⌋ = sqrt 2 / (4 * n) :=
sorry

end max_c_value_equality_conditions_for_n_l523_523269


namespace radius_of_circle_l523_523641

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l523_523641


namespace discount_at_original_time_l523_523702

-- Define the principal amount
def principal : ℝ := 110

-- Define the discount amount at double the time
def discount_2T : ℝ := 18.33

-- Define the rate of interest per time period T
def interest_rate (T : ℝ) : ℝ := discount_2T / (principal * 2 * T)

-- Define the discount at the original time
def discount_T (T : ℝ) : ℝ := principal * interest_rate T * T

-- State the theorem we want to prove
theorem discount_at_original_time (T : ℝ) : discount_T T = 9.165 :=
by
  sorry

end discount_at_original_time_l523_523702


namespace conic_section_is_ellipse_l523_523760

theorem conic_section_is_ellipse (x y : ℝ) :
  3 * x^2 + 5 * y^2 - 9 * x + 10 * y + 15 = 0 → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (λ x y, 3 * (x - a)^2 + 5 * (y - b)^2 = 7/4)) := 
by
  sorry

end conic_section_is_ellipse_l523_523760


namespace simplify_sqrt_pow_six_l523_523468

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523468


namespace survey_min_people_l523_523178

theorem survey_min_people (X : ℕ) (N : ℕ) (h1 : X ≥ 23) (h2 : X - 20 ≥ 23) 
(h3 : 23 ≤ 100) (h4 : 100 ≤ 2300) :
  ((X - 23) + (X - 20 - 23) + 23 + 23 = 100) → 
  (least_common_multiple 23 100 = 2300) → 
  N = 2300 := 
by
  sorry

end survey_min_people_l523_523178


namespace number_of_divisors_180_l523_523342

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l523_523342


namespace num_divisors_180_l523_523354

-- Define a positive integer 180
def n : ℕ := 180

-- Define the function to calculate the number of divisors using prime factorization
def num_divisors (n : ℕ) : ℕ :=
  let factors := [(2, 2), (3, 2), (5, 1)] in
  factors.foldl (λ acc (p : ℕ × ℕ), acc * (p.snd + 1)) 1

-- The main theorem statement
theorem num_divisors_180 : num_divisors n = 18 :=
by
  sorry

end num_divisors_180_l523_523354


namespace arithmetic_sqrt_of_sqrt_16_l523_523608

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523608


namespace sum_integers_neg40_to_60_l523_523150

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end sum_integers_neg40_to_60_l523_523150


namespace product_increase_by_13_l523_523390

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end product_increase_by_13_l523_523390


namespace metropolis_hospital_babies_l523_523372

theorem metropolis_hospital_babies 
    (a b d : ℕ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * d) 
    (h3 : 2 * a + 3 * b + 5 * d = 1200) : 
    5 * d = 260 := 
sorry

end metropolis_hospital_babies_l523_523372


namespace impossible_coins_l523_523454

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l523_523454


namespace sin_one_div_x_intercepts_l523_523272

theorem sin_one_div_x_intercepts :
  let f : ℝ → ℝ := λ x, sin (1 / x)
  let interval := set.Ioo (0.00001 : ℝ) (0.0001 : ℝ)
  let intercepts_count := (⌊100_000 / real.pi⌋ - ⌊10_000 / real.pi⌋)
  intercepts_count = 28648 :=
by
  -- Introducing necessary conditions
  have h1 : ∀ x, f x = 0 ↔ (∃ k : ℤ, x = (1 / (k * real.pi))) := sorry
  have interval_pos : 0 < 0.00001 := by norm_num
  have interval_small : 0.00001 < 0.0001 := by norm_num
  have interval_bound : ∀ x, x ∈ interval ↔ (0.00001 < x ∧ x < 0.0001) := sorry
  have int_lemma : ∀ a b : ℝ, (⌊b / real.pi⌋ - ⌊a / real.pi⌋) = intercepts_count := sorry
  -- Concluding the proof
  exact int_lemma 10_000 100_000

end sin_one_div_x_intercepts_l523_523272


namespace max_sum_of_marks_l523_523782

-- Definitions based on conditions
def is_marked_with (n : ℕ) (mark : ℕ) : Prop :=
  mark ∈ {0, 1, 2}

def marking_rule (k j : ℕ) (marks : ℕ → ℕ) : Prop :=
  (marks k = j) → (∀ i ≤ j, marks (k + i) = 0)

def sum_marks (marks : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum marks

-- Main theorem statement
theorem max_sum_of_marks : 
  ∃ marks : ℕ → ℕ, 
  (∀ k, is_marked_with k (marks k)) → 
  (∀ k j, marking_rule k j marks) → 
  sum_marks marks 2019 = 2021 := 
sorry

end max_sum_of_marks_l523_523782


namespace caterer_ordered_sundaes_l523_523701

theorem caterer_ordered_sundaes :
  ∃ S : ℕ, 225 * 0.60 + S * 0.52 = 200 ∧ S = 125 :=
by
  sorry

end caterer_ordered_sundaes_l523_523701


namespace trajectory_of_point_M_l523_523187

theorem trajectory_of_point_M (a x y : ℝ) (h: 0 < a) (A B M : ℝ × ℝ)
    (hA : A = (x, 0)) (hB : B = (0, y)) (hAB_length : Real.sqrt (x^2 + y^2) = 2 * a)
    (h_ratio : ∃ k, k ≠ 0 ∧ ∃ k', k' ≠ 0 ∧ A = k • M + k' • B ∧ (k + k' = 1) ∧ (k / k' = 1 / 2)) :
    (x / (4 / 3 * a))^2 + (y / (2 / 3 * a))^2 = 1 :=
sorry

end trajectory_of_point_M_l523_523187


namespace distinct_points_count_l523_523405

theorem distinct_points_count (p : ℕ) (hp : Nat.Prime p) (h3 : 3 ≤ p) :
  let points := {i | ∃ k : ℕ, 1 ≤ k ∧ k ≤ p ∧ i = (k * (k - 1) / 2) % p}
  points.card = (p + 1) / 2 :=
by
  sorry

end distinct_points_count_l523_523405


namespace arithmetic_sqrt_sqrt_16_l523_523590

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523590


namespace world_cup_teams_count_l523_523930

/-- In the world cup inauguration event, captains and vice-captains of all the teams are invited and awarded welcome gifts. There are some teams participating in the world cup, and 14 gifts are needed for this event. If each team has a captain and a vice-captain, and thus receives 2 gifts, then the number of teams participating is 7. -/
theorem world_cup_teams_count (total_gifts : ℕ) (gifts_per_team : ℕ) (teams : ℕ) 
  (h1 : total_gifts = 14) 
  (h2 : gifts_per_team = 2) 
  (h3 : total_gifts = teams * gifts_per_team) 
: teams = 7 :=
by sorry

end world_cup_teams_count_l523_523930


namespace solve_fractional_eq_l523_523548

-- Defining the fractional equation as a predicate
def fractional_eq (x : ℝ) : Prop :=
  (5 / x) = (7 / (x - 2))

-- The main theorem to be proven
theorem solve_fractional_eq : ∃ x : ℝ, fractional_eq x ∧ x = -5 := by
  sorry

end solve_fractional_eq_l523_523548


namespace perpendicular_foot_on_side_l523_523084

variables {P : Type*} [MetricSpace P] {poly : Set P} {O : P}

def is_convex (s : Set P) : Prop := sorry

def is_interior (p : P) (s : Set P) : Prop := sorry

def foot_of_perpendicular (p : P) (line : Line P) : P := sorry

theorem perpendicular_foot_on_side (h_convex : is_convex poly)
    (h_interior : is_interior O poly) :
    ∃ (side : Line P), (side ∈ sides poly) ∧ 
    let foot := foot_of_perpendicular O side in
    foot ∈ side.segment :=
begin
    -- Proof goes here
    sorry
end

end perpendicular_foot_on_side_l523_523084


namespace total_number_of_marbles_l523_523046

def pink_marbles : ℕ := 13
def orange_marbles : ℕ := pink_marbles - 9
def purple_marbles : ℕ := 4 * orange_marbles
def blue_marbles : ℕ := (3 / 2 : ℚ) * purple_marbles

def total_marbles : ℕ := pink_marbles + orange_marbles + purple_marbles + blue_marbles

theorem total_number_of_marbles : total_marbles = 57 := by
  calc
    total_marbles = pink_marbles + orange_marbles + purple_marbles + blue_marbles := rfl
               ... = 13 + orange_marbles + purple_marbles + blue_marbles := by rw [pink_marbles]
               ... = 13 + (13 - 9) + purple_marbles + blue_marbles := by rw [orange_marbles]
               ... = 13 + 4 + (4 * (13 - 9)) + blue_marbles := by rw [orange_marbles, purple_marbles]
               ... = 13 + 4 + 16 + blue_marbles := by norm_num
               ... = 13 + 4 + 16 + ((3 / 2 : ℚ) * 16) := by rw [blue_marbles, purple_marbles]
               ... = 13 + 4 + 16 + 24 := by norm_num
               ... = 57 := by norm_num

end total_number_of_marbles_l523_523046


namespace arithmetic_sqrt_sqrt_16_l523_523594

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523594


namespace triangle_angle_B_l523_523006

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523006


namespace tens_digit_of_6_pow_2050_l523_523740

theorem tens_digit_of_6_pow_2050 :
  let n := 6^2050 in 
  (n % 100) / 10 % 10 = 5 :=
  sorry

end tens_digit_of_6_pow_2050_l523_523740


namespace math_problem_correct_propositions_l523_523679

theorem math_problem_correct_propositions :
  (∀ x : ℝ, x > 0 → (∃ y : ℝ, y > 0 → (1 / x + 3 / y = 1 → x + 2 * y ≥ 7 + 2 * real.sqrt 6))) ∧
  (1 / real.logb (1/4) (1/9) + 1 / real.logb (1/5) (1/3) = 1 / real.log 3) :=
by
  sorry

end math_problem_correct_propositions_l523_523679


namespace b8_expression_l523_523056

theorem b8_expression (a b : ℕ → ℚ)
  (ha0 : a 0 = 2)
  (hb0 : b 0 = 3)
  (ha : ∀ n, a (n + 1) = (a n) ^ 2 / (b n))
  (hb : ∀ n, b (n + 1) = (b n) ^ 2 / (a n)) :
  b 8 = 3 ^ 3281 / 2 ^ 3280 :=
by
  sorry

end b8_expression_l523_523056


namespace solve_equation_l523_523542

theorem solve_equation (x y z : ℝ) (n m k : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1/(sin^2 x))^3 + (cos^2 y + 1/(cos^2 y))^3 = 16 * sin^2 z) →
  (∃ n m k : ℤ, x = (π / 2) + π * n ∧ y = π * m ∧ z = (π / 2) + π * k) :=
sorry

end solve_equation_l523_523542


namespace nested_logarithm_l523_523770

noncomputable def logarithm_series : ℝ := 
  if h : ∃ x : ℝ, 3^x = x + 81 then classical.some h else 0

theorem nested_logarithm (h : ∃ x : ℝ, 3^x = x + 81) :
  logarithm_series = classical.some h ∧ 
  abs (logarithm_series - 4.5) < 1 :=
by
  sorry

end nested_logarithm_l523_523770


namespace arrangement_count_correct_l523_523651

-- Define sets for teachers and students
def teachers : Finset ℕ := {1, 2, 3}  -- Represent teachers as 1, 2, 3 
def students : Finset ℕ := {1, 2, 3, 4, 5, 6}  -- Represent students as 1 to 6

noncomputable def count_arrangements : ℕ :=
  let C (n k : ℕ) : ℕ := Nat.choose n k  -- Binomial coefficient
  let A (n k : ℕ) : ℕ := Nat.fact n / Nat.fact (n - k)  -- Permutation
  C 6 2 * C 4 2 * C 2 2 * A 3 3  -- Calculating arrangements

theorem arrangement_count_correct : count_arrangements = 540 := by
  -- Expected proof (not provided)
  sorry

end arrangement_count_correct_l523_523651


namespace arithmetic_sqrt_of_sqrt_16_l523_523565

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523565


namespace simplify_sqrt7_pow6_l523_523519

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523519


namespace nested_logarithm_l523_523769

noncomputable def logarithm_series : ℝ := 
  if h : ∃ x : ℝ, 3^x = x + 81 then classical.some h else 0

theorem nested_logarithm (h : ∃ x : ℝ, 3^x = x + 81) :
  logarithm_series = classical.some h ∧ 
  abs (logarithm_series - 4.5) < 1 :=
by
  sorry

end nested_logarithm_l523_523769


namespace A_inter_B_complement_l523_523826

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem A_inter_B_complement :
  A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)} ∧
  {x | x ∉ A ∩ B} = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8 } :=
by
  sorry

end A_inter_B_complement_l523_523826


namespace simplify_sqrt_7_pow_6_l523_523489

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523489


namespace solve_functional_equation_l523_523267

theorem solve_functional_equation (f : ℝ → ℝ) :
  (∀ x y, f(x + y) + f(x) * f(y) = f(x * y) + 2 * x * y + 1) →
    (f = λ x, -x - 1 ∨ f = λ x, 2 * x - 1 ∨ f = λ x, x^2 - 1) := by
  sorry

end solve_functional_equation_l523_523267


namespace impossible_coins_l523_523448

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l523_523448


namespace arithmetic_sqrt_of_sqrt_16_l523_523570

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523570


namespace Mark_bill_total_l523_523696

theorem Mark_bill_total
  (original_bill : ℝ)
  (first_late_charge_rate : ℝ)
  (second_late_charge_rate : ℝ)
  (after_first_late_charge : ℝ)
  (final_total : ℝ) :
  original_bill = 500 ∧
  first_late_charge_rate = 0.02 ∧
  second_late_charge_rate = 0.02 ∧
  after_first_late_charge = original_bill * (1 + first_late_charge_rate) ∧
  final_total = after_first_late_charge * (1 + second_late_charge_rate) →
  final_total = 520.20 := by
  sorry

end Mark_bill_total_l523_523696


namespace Ivan_expected_shots_l523_523951

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l523_523951


namespace binomial_coefficient_third_term_l523_523755

theorem binomial_coefficient_third_term :
  (nat.choose 5 2) = 10 :=
by
  sorry

end binomial_coefficient_third_term_l523_523755


namespace simplify_sqrt7_pow6_l523_523526

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523526


namespace log_4_50_approximation_l523_523303

theorem log_4_50_approximation :
  let log_base_10_2 := 0.301
  let log_base_10_5 := 0.699
  log_base_4_50 := log 50 / log 4 in
  abs (log_base_4_50 - 20 / 7) < abs (log_base_4_50 - 21 / 7) 
   ∧ abs (log_base_4_50 - 20 / 7) < abs (log_base_4_50 - 19 / 7) 
   ∧ abs (log_base_4_50 - 20 / 7) < abs (log_base_4_50 - 22 / 7) := 
begin
  sorry
end

end log_4_50_approximation_l523_523303


namespace valid_ordered_triples_count_l523_523053

def S := {n | 1 ≤ n ∧ n ≤ 23}

def succ (a b : ℕ) : Prop :=
  (∃ n : ℕ, n ∈ {1, 2, ..., 10} ∧ (a - b) % 23 = n) ∨ 
  (∃ m : ℕ, m > 12 ∧ (b - a) % 23 = m)

theorem valid_ordered_triples_count :
  { t : (ℕ × ℕ × ℕ) // t.1 ∈ S ∧ t.2.1 ∈ S ∧ t.2.2 ∈ S ∧ succ t.1 t.2.1 ∧ succ t.2.1 t.2.2 ∧ succ t.2.2 t.1 }.card = 759 :=
sorry

end valid_ordered_triples_count_l523_523053


namespace bridge_length_is_390_meters_l523_523725

noncomputable def train_length : ℕ := 110
noncomputable def train_speed_kmph : ℕ := 60
noncomputable def time_to_cross_bridge : ℝ := 29.997600191984642
noncomputable def speed_conversion_factor : ℝ := 1000 / 3600 -- Converting kmph to m/s
noncomputable def train_speed_mps : ℝ := train_speed_kmph * speed_conversion_factor

theorem bridge_length_is_390_meters :
  let train_length := (110 : ℕ),
      time_to_cross_bridge := (29.9976 : ℝ),
      train_speed_mps := (60 * (1000 / 3600) : ℝ) in
  (train_speed_mps * time_to_cross_bridge) - train_length = 390 :=
by
  sorry

end bridge_length_is_390_meters_l523_523725


namespace max_brownies_l523_523867

theorem max_brownies (m n : ℕ) (h1 : (m-2)*(n-2) = 2*(2*m + 2*n - 4)) : m * n ≤ 294 :=
by sorry

end max_brownies_l523_523867


namespace emily_journey_length_l523_523779

theorem emily_journey_length
  (y : ℝ)
  (h1 : y / 5 + 30 + y / 3 + y / 6 = y) :
  y = 100 :=
by
  sorry

end emily_journey_length_l523_523779


namespace units_digit_sum_l523_523885

theorem units_digit_sum : 
  (∃ (A : ℕ → ℕ), 
    (A 1)^1 = 1 ∧ 
    (A 2)^2 = 4 ∧ 
    (A 3)^3 = 6 ∧ 
    (A 4)^4 = 256 ∧
    (∀ n, 5 ≤ n ∧ n ≤ 100 → (A n)^n % 10 = 0) ∧
    (((A 1)^1 + (A 2)^2 + (A 3)^3 + (A 4)^4 + ∑ i in finset.range 96, (A (i+5))^(i+5)) % 10 = 3)) :=
sorry

end units_digit_sum_l523_523885


namespace sum_of_reversed_base5_base11_numbers_l523_523280

theorem sum_of_reversed_base5_base11_numbers :
  let S := {n : ℕ | (∀ (b5 : ℕ), (n.digits 5).reverse = (b5.digits 11))} in
  S.sum = 10 := 
by
  let S := {n : ℕ | ((n.digits 5).reverse = n.digits 11)}
  sorry

end sum_of_reversed_base5_base11_numbers_l523_523280


namespace simplify_sqrt_seven_pow_six_l523_523535

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523535


namespace num_pos_divisors_180_l523_523344

theorem num_pos_divisors_180 : 
  let n := 180 in
  let prime_factorization := [(2, 2), (3, 2), (5, 1)] in
  (prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1) = 18 :=
by 
  let n := 180
  let prime_factorization := [(2, 2), (3, 2), (5, 1)]
  have num_divisors := prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1 
  show num_divisors = 18
  sorry

end num_pos_divisors_180_l523_523344


namespace value_of_f_sum_positive_l523_523321

variable (f : ℝ → ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the function f
def f (x : ℝ) : ℝ := x + x^3

-- Define the conditions
variable (h₁ : x₁ + x₂ > 0)
variable (h₂ : x₂ + x₃ > 0)
variable (h₃ : x₃ + x₁ > 0)

theorem value_of_f_sum_positive :
  f x₁ + f x₂ + f x₃ > 0 :=
sorry

end value_of_f_sum_positive_l523_523321


namespace ellipse_properties_proof_l523_523317

section ellipse

variables (a b x y : ℝ)

-- Given conditions
def ellipse (a b x y : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def hyperbola (x y : ℝ) : Prop := (y^2 / 2 - x^2 = 1)

def eccentricity (a b : ℝ) : ℝ := sqrt (1 - b^2 / a^2)

def equation_of_ellipse : Prop :=
  let a := 2
  let b := sqrt 3
  (a^2 = 4) ∧ (b^2 = 3) ∧ (x^2 / 4 + y^2 / 3 = 1)

-- The Lean statement for proving the problem
theorem ellipse_properties_proof 
  (a b x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = 0.5) 
  (h4 : (0, b) = (0, sqrt 3)) : 
  equation_of_ellipse 
  ∧ ∃ l P A B, l ≠ P ∧ l ≠ 0 ∧ l ≠ 90 ∧ 
  ∀ A B, (A = (x, y)) ∧ (B = (x, -y)) 
  ∧ (-4 ≤ (|OA| * |OB|)) 
  ∧ ((|OA| * |OB|) < 13 / 4) 
  :=
begin
  sorry
end

end ellipse

end ellipse_properties_proof_l523_523317


namespace angle_between_AC_and_BC1_in_cube_l523_523918

-- Definitions representing the geometric entities 
def Cube (A B C D A1 B1 C1 D1 : Type) := sorry

def angle_between_skew_lines (AC BC1 : Type) : ℝ := sorry

-- Definitions representing the lines in cube
variables {A B C D A1 B1 C1 D1 : Type}

-- The theorem statement
theorem angle_between_AC_and_BC1_in_cube {A B C D A1 B1 C1 D1 : Type} (h : Cube A B C D A1 B1 C1 D1) :
  angle_between_skew_lines AC BC1 = 60 := sorry

end angle_between_AC_and_BC1_in_cube_l523_523918


namespace complex_pure_imaginary_l523_523838

theorem complex_pure_imaginary (a : ℝ) : 
  (z : ℂ) = ((2 + a*complex.I)*complex.I / (1 - complex.I)) 
  → ∃ (a : ℝ), z = 2 * complex.I
 := sorry

end complex_pure_imaginary_l523_523838


namespace impossible_consecutive_naturals_after_operations_l523_523935

theorem impossible_consecutive_naturals_after_operations 
  (initial_numbers : list ℕ)
  (h_initial : ∃ n : ℕ, initial_numbers = list.range' n 10)
  (operations : list (ℕ × ℕ → ℕ × ℕ))  -- List of operations
  (apply_operations : list ℕ → list ℕ → list ℕ)  -- Function to apply operations
  (h_apply_operations : ∀ nums, apply_operations nums operations ≠ initial_numbers) :
  ¬ (∃ final_numbers : list ℕ, final_numbers = list.range' n 10) :=
by
  sorry

end impossible_consecutive_naturals_after_operations_l523_523935


namespace regular_hexagon_area_decrease_l523_523717

noncomputable def area_decrease (original_area : ℝ) (side_decrease : ℝ) : ℝ :=
  let s := (2 * original_area) / (3 * Real.sqrt 3)
  let new_side := s - side_decrease
  let new_area := (3 * Real.sqrt 3 / 2) * new_side ^ 2
  original_area - new_area

theorem regular_hexagon_area_decrease :
  area_decrease (150 * Real.sqrt 3) 3 = 76.5 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_decrease_l523_523717


namespace num_factors_of_N_correct_l523_523416

-- Define the letters in "SUPERCALIFRAGILISTICEXPIALIDOCIOUS"
def S := 3
def U := 2
def P := 2
def E := 2
def R := 2
def C := 3
def A := 3
def L := 3
def I := 7
def F := 1
def G := 1
def T := 1
def X := 1
def D := 1
def O := 2

-- Define the total number of letters
def total_letters := 34

-- Calculate the number of distinct rearrangements
noncomputable def N : Nat :=
  (total_letters.factorial) / (I.factorial * (S.factorial)^4 * (U.factorial)^5)

-- Prime factorization of N provided as per the steps in the solution
def N_prime_factors : List (Nat × Nat) :=
  [(2, 19), (3, 9), (5, 6), (7, 3), (11, 3), (13, 2), (17, 2), (19, 1), (23, 1), (29, 1), (31, 1)]

-- Calculate the number of positive factors of N
noncomputable def num_positive_factors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (λ acc (p : Nat × Nat), acc * (p.snd + 1)) 1

theorem num_factors_of_N_correct :
  num_positive_factors N_prime_factors = 1612800 := by
  sorry

end num_factors_of_N_correct_l523_523416


namespace arithmetic_seq_sum_l523_523980

theorem arithmetic_seq_sum (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S(3) = 9) 
  (h₂ : S(6) = 36) 
  (h₃ : ∀ n, S(n + 1) = S(n) + a(n + 1)) :
  a(7) + a(8) + a(9) = 45 :=
by
  sorry

end arithmetic_seq_sum_l523_523980


namespace tickets_bought_l523_523041

-- Definitions: Jackson's expenditure and cost constraints
def cost_of_game : ℕ := 66
def cost_per_ticket : ℕ := 12
def total_spent : ℕ := 102
def num_tickets (x : ℕ) : Prop := cost_of_game + cost_per_ticket * x = total_spent

-- Theorem: Determine the number of movie tickets Jackson bought
theorem tickets_bought : ∃ x : ℕ, num_tickets x ∧ x = 3 := 
by
  use 3
  constructor
  { calc
      cost_of_game + cost_per_ticket * 3
      = 66 + 12 * 3 : by rfl
      ... = 66 + 36 : by rfl
      ... = 102 : by rfl
  }
sorries

end tickets_bought_l523_523041


namespace total_rent_of_field_is_correct_l523_523695

namespace PastureRental

def cowMonths (cows : ℕ) (months : ℕ) : ℕ := cows * months

def aCowMonths : ℕ := cowMonths 24 3
def bCowMonths : ℕ := cowMonths 10 5
def cCowMonths : ℕ := cowMonths 35 4
def dCowMonths : ℕ := cowMonths 21 3

def totalCowMonths : ℕ := aCowMonths + bCowMonths + cCowMonths + dCowMonths

def rentPerCowMonth : ℕ := 1440 / aCowMonths

def totalRent : ℕ := rentPerCowMonth * totalCowMonths

theorem total_rent_of_field_is_correct :
  totalRent = 6500 :=
by
  sorry

end PastureRental

end total_rent_of_field_is_correct_l523_523695


namespace solve_trig_eqn_l523_523546

theorem solve_trig_eqn (x y z : ℝ) (n m k : ℤ) :
  (\sin x ≠ 0) →
  (\cos y ≠ 0) →
  ((\sin^2 x + 1 / (\sin^2 x))^3 + (\cos^2 y + 1 / (\cos^2 y))^3 = 16 * \sin^2 z) →
  (x = π / 2 + n * π ∧ y = m * π ∧ z = π / 2 + k * π) :=
  sorry

end solve_trig_eqn_l523_523546


namespace impossible_coins_l523_523466

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l523_523466


namespace probability_letter_in_PROBABILITY_l523_523892

theorem probability_letter_in_PROBABILITY :
  let alphabet_size := 26
  let unique_letters_in_PROBABILITY := 9
  (unique_letters_in_PROBABILITY : ℝ) / (alphabet_size : ℝ) = 9 / 26 := by
    sorry

end probability_letter_in_PROBABILITY_l523_523892


namespace linear_regression_time_l523_523841

theorem linear_regression_time (x : ℕ) (h : x = 400) : 
  let y := 0.2 * x + 3 in y = 83 :=
by
  sorry

end linear_regression_time_l523_523841


namespace simplify_sqrt_pow_six_l523_523472

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523472


namespace find_value_l523_523285

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end find_value_l523_523285


namespace find_bd_length_l523_523931

-- Given definitions and conditions
variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables [MetricSpace.angle A B C]
variables [MetricSpace.length AC BC AB]
variables [MetricSpace.length DE BD]

-- Outputs the calculation of BD
theorem find_bd_length
  (H1 : angle C = 90) 
  (H2 : length AC = 9)
  (H3 : length BC = 12)
  (H4 : on_line D AB)
  (H5 : on_line E BC)
  (H6 : angle BED = 90)
  (H7 : length DE = 5) :
  length BD = 25 / 3 :=
sorry

end find_bd_length_l523_523931


namespace train_length_l523_523685

-- Definitions based on conditions
def relative_speed_km_per_hr (v1 v2 : ℝ) : ℝ := v1 - v2

def relative_speed_m_per_s (relative_speed : ℝ) : ℝ := relative_speed * (5 / 18)

def distance_covered (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length 
  (v1 v2 : ℝ) 
  (relative_speed_km_per_hr v1 v2 = 10)
  (version m/s = relative_speed_m_per_s 10)
  (time : ℝ)
  (time = 36) 
  (distance = distance_covered 2.7778 time) : 
  ∃ L : ℝ, 2 * L = distance → L = 50 := 
by
  sorry

end train_length_l523_523685


namespace solve_equation_l523_523543

theorem solve_equation (x y z : ℝ) (n m k : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1/(sin^2 x))^3 + (cos^2 y + 1/(cos^2 y))^3 = 16 * sin^2 z) →
  (∃ n m k : ℤ, x = (π / 2) + π * n ∧ y = π * m ∧ z = (π / 2) + π * k) :=
sorry

end solve_equation_l523_523543


namespace solve_prime_equation_l523_523540

theorem solve_prime_equation (x y z : ℕ) (hpx : nat.prime x) (hpy : nat.prime y) (hpz : nat.prime z) (h : x^y + 1 = z) : x = 2 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end solve_prime_equation_l523_523540


namespace power_of_7_in_expression_l523_523796

namespace PrimeFactors

theorem power_of_7_in_expression (p : ℕ) :
  (4 ^ 13 = (2 : ℕ) ^ 26) ∧ (11 ^ 2 = (11 : ℕ) * 11) ∧ (total_prime_factors : ℕ = 33) →
  (p = 33 - 26 - 2) →
  (p = 5) :=
by
  sorry

end PrimeFactors

end power_of_7_in_expression_l523_523796


namespace car_arrives_first_and_earlier_l523_523799

-- Define the conditions
def total_intersections : ℕ := 11
def total_blocks : ℕ := 12
def green_time : ℕ := 3
def red_time : ℕ := 1
def car_block_time : ℕ := 1
def bus_block_time : ℕ := 2

-- Define the functions that compute the travel times
def car_travel_time (blocks : ℕ) : ℕ :=
  (blocks / 3) * (green_time + red_time) + (blocks % 3 * car_block_time)

def bus_travel_time (blocks : ℕ) : ℕ :=
  blocks * bus_block_time

-- Define the theorem to prove
theorem car_arrives_first_and_earlier :
  car_travel_time total_blocks < bus_travel_time total_blocks ∧
  bus_travel_time total_blocks - car_travel_time total_blocks = 9 := 
by
  sorry

end car_arrives_first_and_earlier_l523_523799


namespace arithmetic_sqrt_of_sqrt_16_l523_523558

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523558


namespace find_a7_a8_a9_l523_523988

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l523_523988


namespace num_four_digit_with_product_18_l523_523879

-- Definitions based on conditions
def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999
def digit_product (n : ℕ) : ℕ := 
  n.digits.foldl (λ acc d => acc * d) 1

-- Main theorem
theorem num_four_digit_with_product_18 : 
  (count (λ n : ℕ, is_four_digit n ∧ digit_product n = 18)) = 48 :=
sorry

end num_four_digit_with_product_18_l523_523879


namespace problem_ab_value_l523_523433

theorem problem_ab_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : (∛(log 10 a)) + (∛(log 10 b)) + (log 10 (∛a)) + (log 10 (∛b)) = 12)
  (h4 : (∛(log 10 a)) ∈ ℤ) (h5 : (∛(log 10 b)) ∈ ℤ) 
  (h6 : (log 10 (∛(a))) ∈ ℤ) (h7 : (log 10 (∛(b))) ∈ ℤ) : 
  a * b = 10^9 := 
  sorry

end problem_ab_value_l523_523433


namespace Al_closest_to_given_mass_percentage_l523_523788

-- Define the conditions
def Al_mass := 26.98  -- g/mol
def Cl_mass := 35.45  -- g/mol
def AlCl3_molar_mass := 133.33  -- g/mol

-- Define the calculation of the mass percentage
def mass_percentage_Al : Float :=
  (Al_mass / AlCl3_molar_mass) * 100

-- Define the mass percentage given in the problem
def given_mass_percentage : Float := 20.45

-- The statement to prove
theorem Al_closest_to_given_mass_percentage :
  |mass_percentage_Al - given_mass_percentage| < 0.5 :=
begin
  sorry
end

end Al_closest_to_given_mass_percentage_l523_523788


namespace solve_equation_l523_523541

theorem solve_equation (x y z : ℝ) (n m k : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1/(sin^2 x))^3 + (cos^2 y + 1/(cos^2 y))^3 = 16 * sin^2 z) →
  (∃ n m k : ℤ, x = (π / 2) + π * n ∧ y = π * m ∧ z = (π / 2) + π * k) :=
sorry

end solve_equation_l523_523541


namespace mass_percentage_O_in_boric_acid_l523_523143

theorem mass_percentage_O_in_boric_acid :
  let H_mass := 1.01
  let B_mass := 10.81
  let O_mass := 16.00
  let boric_acid_formula := (3 * H_mass) + 1 * B_mass + 3 * O_mass
  (3 * O_mass / boric_acid_formula) * 100 ≈ 77.57 := 
by
  let H_mass := 1.01
  let B_mass := 10.81
  let O_mass := 16.00
  let boric_acid_formula := (3 * H_mass) + 1 * B_mass + 3 * O_mass
  have h : (3 * O_mass / boric_acid_formula) * 100 = 77.5733900359 := 
    by norm_num
  sorry

end mass_percentage_O_in_boric_acid_l523_523143


namespace number_of_valid_four_digit_integers_equals_36_l523_523874

def is_valid_digit (d : ℕ) : Prop :=
  d ≥ 1 ∧ d ≤ 9

def valid_digits (num : ℕ) (digits : list ℕ) : Prop :=
  digits.product = 18 ∧ list.length digits = 4 ∧ (∀ d ∈ digits, is_valid_digit d)

def count_valid_numbers : ℕ :=
  -- Combinations and their corresponding permutations calculated from the solution
  12 + 12 + 12

theorem number_of_valid_four_digit_integers_equals_36 :
  count_valid_numbers = 36 := sorry

end number_of_valid_four_digit_integers_equals_36_l523_523874


namespace area_of_region_l523_523754

theorem area_of_region : 
  (∃ A : ℝ, 
    (∀ x y : ℝ, 
      (|4 * x - 20| + |3 * y + 9| ≤ 4) → 
      A = (32 / 3))) :=
by 
  sorry

end area_of_region_l523_523754


namespace find_a2_l523_523917

variable {a_n : ℕ → ℚ}

def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n-1) * d

theorem find_a2 (h_seq : arithmetic_seq a_n) (h3_5 : a_n 3 + a_n 5 = 15) (h6 : a_n 6 = 7) :
  a_n 2 = 8 := 
sorry

end find_a2_l523_523917


namespace simplify_sqrt7_pow6_l523_523520

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523520


namespace min_value_u3_v3_l523_523292

open Complex

theorem min_value_u3_v3 
(u v : ℂ)
(h1 : |u + v| = 2)
(h2 : |u^2 + v^2| = 8) : 
  |u^3 + v^3| = 20 := 
sorry

end min_value_u3_v3_l523_523292


namespace max_volume_cylinder_l523_523106

-- Conditions
def circumference (d m : ℝ) : Prop := 2 * (d + m) = 90

-- Proof statement
theorem max_volume_cylinder (d m : ℝ) (h : circumference d m) : 
  ∃ r h : ℝ, d = 2 * r ∧ m = h ∧
  (∀ V : ℝ, V = π * r^2 * h → V ≤ 3375 * π) ∧ 
  ∃ r : ℝ, d = 2 * r ∧ m = r ―> 3375 * π = π * r^2 * r := 
sorry

end max_volume_cylinder_l523_523106


namespace find_g_of_2_l523_523784

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_2 :
  (∀ x : ℝ, x ≠ 1 / 2 → g(x) + g((2 * x + 1) / (1 - 2 * x)) = 2 * x) →
  g(2) = 22 / 9 := by
  intros h
  sorry

end find_g_of_2_l523_523784


namespace first_player_has_winning_strategy_l523_523128

noncomputable def first_player_wins : Prop :=
  ∀ (turns : ℕ → ℕ) (turn_condition : ∀ n, 0 ≤ turns n ∧ turns n ≤ 9),
  ( ∃ (win_strategy : ℕ → ℕ), 
    (∀ i, 0 ≤ win_strategy i ∧ win_strategy i ≤ 9) ∧ 
    (∑ i in finset.range 101, win_strategy i) % 11 = 0)

theorem first_player_has_winning_strategy : first_player_wins :=
sorry

end first_player_has_winning_strategy_l523_523128


namespace range_of_m_l523_523069

noncomputable def f : ℝ → ℝ
| x if x < 0 := Real.log (-x)
| x if x > 0 := -Real.log x
| _ := 0

theorem range_of_m (m : ℝ) : (f m > f (-m)) ↔ (m < -1 ∨ (0 < m ∧ m < 1)) := by
  sorry

end range_of_m_l523_523069


namespace number_1973_occurrences_l523_523110

theorem number_1973_occurrences :
  ∀ n, Nat.Prime n →
  (∀ k, k ≤ n → ∃ a b c : ℕ, (∀ m, m ≤ k → (a + b = c))) →
  ∀ iter, iter = 1973 →
  EulerTotient (n - 1) = 1972 :=
by
  assume n h_prime h_seq iter h_iter
  sorry

end number_1973_occurrences_l523_523110


namespace problem_area_triangle_PNT_l523_523091

noncomputable def area_triangle_PNT (PQ QR x : ℝ) : ℝ :=
  let PS := Real.sqrt (PQ^2 + QR^2)
  let PN := PS / 2
  let area := (PN * Real.sqrt (61 - x^2)) / 4
  area

theorem problem_area_triangle_PNT :
  ∀ (PQ QR : ℝ) (x : ℝ), PQ = 10 → QR = 12 → 0 ≤ x ∧ x ≤ 10 → area_triangle_PNT PQ QR x = 
  (Real.sqrt (244) * Real.sqrt (61 - x^2)) / 4 :=
by
  intros PQ QR x hPQ hQR hx
  sorry

end problem_area_triangle_PNT_l523_523091


namespace roundness_of_8000000_l523_523759

def roundness (n : ℕ) : ℕ :=
  let factors := (n.factorize)
  factors.foldr (λ (p e) acc => acc + e) 0

theorem roundness_of_8000000 : roundness 8000000 = 15 := by
  sorry

end roundness_of_8000000_l523_523759


namespace triangle_tan_conditions_l523_523901

-- Given the triangle ABC and the conditions of the problem
variables (A B C : ℝ)
variables (AB AC BC : ℝ)

-- Assuming the conditions 
theorem triangle_tan_conditions 
  (h1 : tan A = 1/4)
  (h2 : tan B = 3/5)
  (h3 : C = π - (A + B))
  (h4 : AB = sqrt 17)
  (h5 : ∀ (X : ℝ), 0 < A < π/2 ∧ 0 < B < π/2 ∧ 0 < C < π) :
  -- Prove the statements required
  (C = 3 * π / 4) ∧ (BC = sqrt 2) :=
begin
  sorry
end

end triangle_tan_conditions_l523_523901


namespace arithmetic_sequence_sum_l523_523689

-- Define the arithmetic sequence properties
def seq : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
def first := 81
def last := 99
def common_diff := 2
def n := 10

-- Main theorem statement proving the desired property
theorem arithmetic_sequence_sum :
  2 * (seq.sum) = 1800 := by
  sorry

end arithmetic_sequence_sum_l523_523689


namespace midpoint_in_segment_l523_523973

open Set Metric

theorem midpoint_in_segment (S : Set ℝ^2) (D : Set ℝ^2)
  (hS_nonempty : S.Nonempty) (hS_closed : IsClosed S)
  (hD_closed : IsClosed D) (hS_in_D : S ⊆ D)
  (hD_property : ∀ D', IsClosed D' → (S ⊆ D' → D ⊆ D')) :
  ∀ y ∈ D, ∃ z1 z2 ∈ S, y = (z1 + z2) / 2 :=
sorry

end midpoint_in_segment_l523_523973


namespace janet_jasmine_shampoo_l523_523955

theorem janet_jasmine_shampoo (rose jasmine total : ℚ) :
  rose = 1/3 ∧ total = 7 * (1/12) ∧ total - rose = jasmine → 
  jasmine = 1/4 := 
by 
  intros h 
  obtain ⟨h_rose, h_total, h_total_rose⟩ := h 
  rw [h_rose, h_total, h_total_rose] 
  norm_num
  sorry

end janet_jasmine_shampoo_l523_523955


namespace number_of_divisors_of_180_l523_523349

theorem number_of_divisors_of_180 : 
   (nat.coprime 2 3 ∧ nat.coprime 3 5 ∧ nat.coprime 5 2 ∧ 180 = 2^2 * 3^2 * 5^1) →
   (nat.divisors_count 180 = 18) :=
by
  sorry

end number_of_divisors_of_180_l523_523349


namespace number_of_pieces_correct_l523_523071

-- Define the dimensions of the pan
def pan_length : ℕ := 30
def pan_width : ℕ := 24

-- Define the dimensions of each piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 2

-- Calculate the area of the pan
def pan_area : ℕ := pan_length * pan_width

-- Calculate the area of each piece of brownie
def piece_area : ℕ := piece_length * piece_width

-- The proof problem statement
theorem number_of_pieces_correct : (pan_area / piece_area) = 120 :=
by sorry

end number_of_pieces_correct_l523_523071


namespace circle_radius_l523_523633

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l523_523633


namespace radius_of_circle_l523_523639

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l523_523639


namespace simplify_sqrt7_pow6_l523_523505

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523505


namespace length_of_tank_l523_523723

noncomputable def tank_length : ℝ :=
  let width := 12
  let depth := 6
  let cost_per_sq_m := 75 / 100 -- converting paise to INR
  let total_cost := 558
  let total_area_plastered := total_cost / cost_per_sq_m
  let equation := (2 * (width * depth)) + (2 * (depth * width)) + (width * tank_length)
  total_area_plastered = equation,
  744 = 24 * tank_length + 144,
  tank_length

theorem length_of_tank (width depth : ℝ) (cost_per_sq_m total_cost tank_length : ℝ) :
  width = 12 → depth = 6 → cost_per_sq_m = 0.75 → total_cost = 558 →
  (2 * (tank_length * depth) + 2 * (width * depth) + (tank_length * width) = total_cost / cost_per_sq_m) →
  tank_length = 25 := 
begin
  sorry
end

end length_of_tank_l523_523723


namespace trees_still_left_l523_523658

theorem trees_still_left 
  (initial_trees : ℕ) 
  (trees_died : ℕ) 
  (trees_cut : ℕ) 
  (initial_trees_eq : initial_trees = 86) 
  (trees_died_eq : trees_died = 15) 
  (trees_cut_eq : trees_cut = 23) 
  : initial_trees - (trees_died + trees_cut) = 48 :=
by
  sorry

end trees_still_left_l523_523658


namespace find_YZ_l523_523934

-- Given the conditions
def triangle_ABC (A B C : Type) := 
  ∃ (a b c : ℝ),
    ∠ A = 45 ∧
    ∠ C = 90 ∧
    a = 6 ∧
    -- Correct Answer
    c = 3 * Real.sqrt 2

-- Prove YZ = 3*sqrt(2) is the length of the side in the triangle
theorem find_YZ : ∃ (YZ : ℝ), YZ = 3 * Real.sqrt 2 :=
by {
  let A := 45,
  let C := 90,
  let XZ := 6,
  use 3 * Real.sqrt 2,
  sorry
}

end find_YZ_l523_523934


namespace simplify_sqrt_seven_pow_six_l523_523528

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523528


namespace circle_radius_l523_523631

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l523_523631


namespace find_x_coordinate_of_X_l523_523082

namespace GeometryProblem

def Point : Type := ℝ × ℝ

noncomputable def x_coord_maximizing_angle (A B : Point) : ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  let AB := (Bx - Ax, By - Ay)
  let slope_AB := (By - Ay) / (Bx - Ax)
  let y_intercept := By - slope_AB * Bx
  - (y_intercept / slope_AB) + 5 * Real.sqrt 2

theorem find_x_coordinate_of_X :
  let A := (0, 4) : Point
  let B := (3, 8) : Point
  x_coord_maximizing_angle A B = 5 * Real.sqrt 2 - 3 :=
by
  skip_proof
  sorry

end GeometryProblem

end find_x_coordinate_of_X_l523_523082


namespace theta_value_l523_523239

def complex_number := 2 - complex.i

theorem theta_value : ∃ θ : ℝ, θ = -real.arctan(1 / 2) ∧ 
  complex_number = abs complex_number * complex.exp(θ * complex.i) :=
by
  sorry

end theta_value_l523_523239


namespace arithmetic_sqrt_of_sqrt_16_l523_523562

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523562


namespace carmen_counting_cars_l523_523744

theorem carmen_counting_cars 
  (num_trucks : ℕ)
  (num_cars : ℕ)
  (red_trucks : ℕ)
  (black_trucks : ℕ)
  (white_trucks : ℕ)
  (total_vehicles : ℕ)
  (percent_white_trucks : ℚ) :
  num_trucks = 50 →
  num_cars = 40 →
  red_trucks = num_trucks / 2 →
  black_trucks = (20 * num_trucks) / 100 →
  white_trucks = num_trucks - red_trucks - black_trucks →
  total_vehicles = num_trucks + num_cars →
  percent_white_trucks = (white_trucks : ℚ) / total_vehicles * 100 →
  percent_white_trucks ≈ 17 :=
sorry

end carmen_counting_cars_l523_523744


namespace smallest_common_multiple_5_6_l523_523146

theorem smallest_common_multiple_5_6 (n : ℕ) 
  (h_pos : 0 < n) 
  (h_5 : 5 ∣ n) 
  (h_6 : 6 ∣ n) :
  n = 30 :=
sorry

end smallest_common_multiple_5_6_l523_523146


namespace principal_amount_is_586_l523_523681

-- Definitions of conditions
variables {P r : ℝ} -- Principal amount and annual rate as real numbers
def simple_interest (P r t : ℝ) := P + (P * r * t)
def condition1 := simple_interest P r 2 = 710
def condition2 := simple_interest P r 7 = 1020

-- The main theorem stating the proof problem
theorem principal_amount_is_586 (h1 : condition1) (h2 : condition2) : P = 586 :=
by sorry

end principal_amount_is_586_l523_523681


namespace fraction_of_girls_at_science_fair_l523_523228

theorem fraction_of_girls_at_science_fair
  (PG_total : ℕ) (PG_boys_ratio PG_girls_ratio : ℕ)
  (MT_total : ℕ) (MT_boys_ratio MT_girls_ratio : ℕ) :
  PG_total = 300 →
  PG_boys_ratio = 3 →
  PG_girls_ratio = 2 →
  MT_total = 240 →
  MT_boys_ratio = 5 →
  MT_girls_ratio = 3 →
  (120 + 90) = 210 ∧ (PG_total + MT_total) = 540 →
  210 / 540 = 7 / 18 :=
begin
  -- conditions
  intros PG_total_eq PG_boys_ratio_eq PG_girls_ratio_eq MT_total_eq MT_boys_ratio_eq MT_girls_ratio_eq calculation,
  sorry,
end

end fraction_of_girls_at_science_fair_l523_523228


namespace count_of_irrational_in_given_numbers_l523_523733

def is_rational (x : ℝ) := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def given_numbers : List ℝ := [1 / 3, 1.414, -Real.sqrt 2, Real.pi, (8:ℝ)^(1 / 3)]

def count_irrational (lst : List ℝ) : ℕ :=
  lst.countp (λ x, ¬ is_rational x)

theorem count_of_irrational_in_given_numbers :
  count_irrational given_numbers = 2 :=
by
  sorry

end count_of_irrational_in_given_numbers_l523_523733


namespace arithmetic_sequence_problem_l523_523365

theorem arithmetic_sequence_problem :
  let S := 1001 + 1004 + 1007 + 1010 + 1013 in
  S = 5050 - N → N = 15 :=
by
  intros S h
  have hS : S = 1001 + 1004 + 1007 + 1010 + 1013 := rfl
  rw hS at h
  sorry

end arithmetic_sequence_problem_l523_523365


namespace tangent_line_at_one_extreme_points_and_inequality_l523_523070

noncomputable def f (x a : ℝ) := x^2 - 2*x + a * Real.log x

-- Question 1: Tangent Line
theorem tangent_line_at_one (x a : ℝ) (h_a : a = 2) (hx_pos : x > 0) :
    2*x - Real.log x - (2*x - Real.log 1 - 1) = 0 := by
  sorry

-- Question 2: Extreme Points and Inequality
theorem extreme_points_and_inequality (a x1 x2 : ℝ) (h1 : 2*x1^2 - 2*x1 + a = 0)
    (h2 : 2*x2^2 - 2*x2 + a = 0) (hx12 : x1 < x2) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
    0 < a ∧ a < 1/2 ∧ (f x1 a) / x2 > -3/2 - Real.log 2 := by
  sorry

end tangent_line_at_one_extreme_points_and_inequality_l523_523070


namespace correct_calculation_l523_523677

theorem correct_calculation (a b : ℝ) :
  ((ab)^3 = a^3 * b^3) ∧ 
  ¬(a + 2 * a^2 = 3 * a^3) ∧ 
  ¬(a * (-a)^4 = -a^5) ∧ 
  ¬((a^3)^2 = a^5) :=
  by
  sorry

end correct_calculation_l523_523677


namespace leif_has_more_oranges_than_apples_l523_523971

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l523_523971


namespace find_missing_digit_l523_523622

-- Define the conditions
def digits (n : ℕ) := [5, 3, 6, 8, 7, 0, 9, 1, 2]

def sum_digits := 41

-- Set 0..9
def all_digits := finset.range 10

-- The digit to check missing
def missing_digit := 4

-- Define the theorem to prove the missing digit is 4
theorem find_missing_digit:
  (∑ k in digits 536870912, k) = sum_digits → 
  ∃ x, x ∉ digits 536870912 ∧ x ∈ all_digits ∧ (x + sum_digits = ∑ y in all_digits, y) :=
by { sorry }

end find_missing_digit_l523_523622


namespace Ivan_expected_shots_l523_523952

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l523_523952


namespace sides_equally_inclined_l523_523778

theorem sides_equally_inclined {A B C D M N : Point}
  (h1 : divides A M B)
  (h2 : divides D N C)
  (h3 : (AM / MB) = (DN / NC))
  (h4 : (DN / NC) = (AD / BC)) :
  equally_inclined (line M N) (AD) (BC) := 
sorry

end sides_equally_inclined_l523_523778


namespace factor_expression_l523_523236

variable (x : ℝ)

theorem factor_expression : 
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := 
by 
  sorry

end factor_expression_l523_523236


namespace exponent_multiplication_l523_523691

variable (x : ℤ)

theorem exponent_multiplication :
  (-x^2) * x^3 = -x^5 :=
sorry

end exponent_multiplication_l523_523691


namespace players_in_team_l523_523554

theorem players_in_team (balls_per_player total_balls : ℕ) (h1 : balls_per_player = 11) (h2 : total_balls = 242) :
  total_balls / balls_per_player = 22 :=
by {
  rw [h1, h2],
  exact Nat.div_eq_of_eq_mul_left (by norm_num) rfl sorry }


end players_in_team_l523_523554


namespace rain_volume_on_ground_l523_523379

def volume_of_rain (A : ℝ) (r : ℝ) : ℝ := A * r

theorem rain_volume_on_ground (A : ℝ) (r : ℝ) 
(hA : A = 1.5 * 10000) 
(hr : r = 0.05) : 
volume_of_rain A r = 750 := 
by 
  unfold volume_of_rain 
  rw [hA, hr] 
  norm_num
  sorry

end rain_volume_on_ground_l523_523379


namespace simplify_sqrt_7_pow_6_l523_523491

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523491


namespace find_external_resistance_l523_523656

def internal_resistance : ℝ := 20

def external_resistance_condition (R: ℝ) : Prop :=
  let max_power := (ε / (internal_resistance + internal_resistance))^2 * internal_resistance
  let reduced_power := 0.75 * max_power
  (ε / (R + internal_resistance))^2 * R = reduced_power

theorem find_external_resistance (ε : ℝ) :
  ∃ R : ℝ, external_resistance_condition R :=
begin
  use [60, 6.67],
  sorry
end

end find_external_resistance_l523_523656


namespace exponential_identity_l523_523801

theorem exponential_identity (x : ℝ) (h : 2^x + 2^(-x) = 3) : 4^x + 4^(-x) = 7 := 
by
  sorry

end exponential_identity_l523_523801


namespace find_M_plus_N_l523_523847

theorem find_M_plus_N (M N : ℕ)
  (h1 : 4 * 63 = 7 * M)
  (h2 : 4 * N = 7 * 84) :
  M + N = 183 :=
by sorry

end find_M_plus_N_l523_523847


namespace card_problem_l523_523698

-- Define the variables
variables (x y : ℕ)

-- Conditions given in the problem
theorem card_problem 
  (h1 : x - 1 = y + 1) 
  (h2 : x + 1 = 2 * (y - 1)) : 
  x + y = 12 :=
sorry

end card_problem_l523_523698


namespace simplify_sqrt7_pow6_l523_523504

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523504


namespace range_of_f_l523_523644

def f (x : ℝ) : ℝ := 1 / 2 * Real.exp x * (Real.sin x + Real.cos x)

theorem range_of_f : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
  Set.range f = Set.Icc (1 / 2) (1 / 2 * Real.exp (Real.pi / 2)) :=
sorry

end range_of_f_l523_523644


namespace interval_of_x_l523_523189

theorem interval_of_x (x : ℝ) (h : x = ((-x)^2 / x) + 3) : 3 < x ∧ x ≤ 6 :=
by
  sorry

end interval_of_x_l523_523189


namespace log_3_eq_4_of_infinite_log_pattern_l523_523764

theorem log_3_eq_4_of_infinite_log_pattern (x : ℝ) (h : 0 < x) :
  x = 4 :=
begin
  sorry
end

end log_3_eq_4_of_infinite_log_pattern_l523_523764


namespace rabbitAgeOrder_l523_523426

-- Define the ages of the rabbits as variables
variables (blue black red gray : ℕ)

-- Conditions based on the problem statement
noncomputable def rabbitConditions := 
  (blue ≠ max blue (max black (max red gray))) ∧  -- The blue-eyed rabbit is not the eldest
  (gray ≠ min blue (min black (min red gray))) ∧  -- The gray rabbit is not the youngest
  (red ≠ min blue (min black (min red gray))) ∧  -- The red-eyed rabbit is not the youngest
  (black > red) ∧ (gray > black)  -- The black rabbit is older than the red-eyed rabbit and younger than the gray rabbit

-- Required proof statement
theorem rabbitAgeOrder : rabbitConditions blue black red gray → gray > black ∧ black > red ∧ red > blue :=
by
  intro h
  sorry

end rabbitAgeOrder_l523_523426


namespace arithmetic_sqrt_of_sqrt_16_l523_523567

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523567


namespace find_percentage_l523_523893

theorem find_percentage (P N : ℝ) (h1 : (P / 100) * N = 60) (h2 : 0.80 * N = 240) : P = 20 :=
sorry

end find_percentage_l523_523893


namespace coefficient_of_x3_l523_523756

theorem coefficient_of_x3 : 
  (polynomial.coeff ((1 - X) * (1 + X) ^ 6) 3) = 5 :=
sorry

end coefficient_of_x3_l523_523756


namespace min_value_S_max_value_S_l523_523654

-- Define the problem context.
variable (n : ℕ) (heights : finset ℕ)
variable (h_range : heights = finset.range (n + 1))
variables [n_pos : fact (0 < n)]

-- Define the sum S of absolute differences between adjacent pairs.
def S := finset.sum (finset.range n) (λ i, nat.abs ((heights.lookup (((i + 1) % n).val)) - 
                                                 (heights.lookup (i.val))))

-- To show minimum value of S is 2(n - 1).
theorem min_value_S : S n heights = 2 * (n - 1) :=
sorry

-- To show maximum value of S is ⌊n^2 / 2⌋.
theorem max_value_S : S n heights = nat.floor (n^2 / 2) :=
sorry

end min_value_S_max_value_S_l523_523654


namespace area_of_XPN_l523_523138

noncomputable def triangle_area_XPN (XYZ_area : ℝ) (M_midXY : Bool) (N_midYZ : Bool) (P_midXM : Bool) : ℝ :=
  if XYZ_area = 180 ∧ M_midXY = tt ∧ N_midYZ = tt ∧ P_midXM = tt then 22.5 else 0

theorem area_of_XPN (XYZ_area : ℝ) (M_midXY : Bool) (N_midYZ : Bool) (P_midXM : Bool) (h₁ : XYZ_area = 180) (h₂ : M_midXY = tt) (h₃ : N_midYZ = tt) (h₄ : P_midXM = tt) :
  triangle_area_XPN XYZ_area M_midXY N_midYZ P_midXM = 22.5 :=
by
  -- Lean proof is required to be implemented here.
  sorry

end area_of_XPN_l523_523138


namespace simplify_sqrt_seven_pow_six_proof_l523_523512

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523512


namespace triangle_area_l523_523420

noncomputable def point : Type := (ℝ × ℝ × ℝ)

theorem triangle_area :
  let O : point := (0, 0, 0)
    let A : point := (Real.sqrt 10, 0, 0)
    let B : point := (0, Real.sqrt 10, 0)
    let C : point := (0, 0, Real.sqrt 10)
    let angle_BAC := Real.pi / 4
  in ∃ (area : ℝ), area = 5 * Real.sqrt 2 :=
by
  sorry

end triangle_area_l523_523420


namespace kitten_current_length_l523_523212

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l523_523212


namespace countSelectivePositiveIntegers_l523_523241

-- Define what a selective number is
def isSelective (n : Nat) : Prop :=
  (n < 10) ∨
  (List.Nodup (Nat.digits 10 n) ∧ List.Incl [5] (Nat.digits 10 n) ∧ List.Increases (Nat.digits 10 n)) ∨
  (List.Nodup (Nat.digits 10 n) ∧ List.Decreases (Nat.digits 10 n) ∧ (Nat.digits 10 n).head = 0)

-- Define the concept of increasing and decreasing sequences in digits
def List.Increases : List Nat → Prop
| []       => True
| [_]      => True
| (x::y::xs) => x < y ∧ List.Increases (y::xs)

def List.Decreases : List Nat → Prop
| []       => True
| [_]      => True
| (x::y::xs) => x > y ∧ List.Decreases (y::xs)

-- Translate the problem as a theorem in Lean 4
theorem countSelectivePositiveIntegers : ∑ n in {n | n < 10000 ∧ isSelective n}, 1 = 1015 := sorry

end countSelectivePositiveIntegers_l523_523241


namespace find_angle_B_l523_523012

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523012


namespace class_A_has_neater_scores_l523_523383

-- Definitions for the given problem conditions
def mean_Class_A : ℝ := 120
def mean_Class_B : ℝ := 120
def variance_Class_A : ℝ := 42
def variance_Class_B : ℝ := 56

-- The theorem statement to prove Class A has neater scores
theorem class_A_has_neater_scores : (variance_Class_A < variance_Class_B) := by
  sorry

end class_A_has_neater_scores_l523_523383


namespace sample_data_properties_l523_523815

theorem sample_data_properties :
  let x := List.range' 1 15 
  ∧ let x := List.map (λ i → 2 * i) x
  ∧ let y := List.map (λ xi → xi - 20) x
  in (List.sample_variance x = List.sample_variance y)
  ∧ (List.percentile y 30 = -10) :=
by
  let x := List.range' 1 15 
  let x := List.map (λ i → 2 * i) x
  let y := List.map (λ xi → xi - 20) x
  have h_var : List.sample_variance x = List.sample_variance y := sorry
  have h_percentile : List.percentile y 30 = -10 := sorry
  exact ⟨h_var, h_percentile⟩

end sample_data_properties_l523_523815


namespace sum_sequence_l523_523719

theorem sum_sequence : 
  let b : ℕ → ℚ := λ n, if n = 1 then 2 else if n = 2 then 3 else (1/2) * b (n-1) + (1/5) * b (n-2) in
  ∑' n, b n = 40 / 3 :=
by
  sorry

end sum_sequence_l523_523719


namespace simplify_sqrt_seven_pow_six_l523_523531

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523531


namespace income_calculation_l523_523118

-- Define the conditions
def ratio (i e : ℕ) : Prop := 9 * e = 8 * i
def savings (i e : ℕ) : Prop := i - e = 4000

-- The theorem statement
theorem income_calculation (i e : ℕ) (h1 : ratio i e) (h2 : savings i e) : i = 36000 := by
  sorry

end income_calculation_l523_523118


namespace find_max_value_l523_523418

noncomputable def max_value (x y z : ℝ) : ℝ := (x + y) / (x * y * z)

theorem find_max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 2) :
  max_value x y z ≤ 13.5 :=
sorry

end find_max_value_l523_523418


namespace simplify_sqrt7_pow6_l523_523525

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523525


namespace mutter_lagaan_payment_l523_523048

-- Conditions as definitions
def total_lagaan_collected : ℝ := 344000
def mutter_percentage_of_total_taxable_land : ℝ := 0.23255813953488372 / 100

-- Proof statement
theorem mutter_lagaan_payment : (mutter_percentage_of_total_taxable_land * total_lagaan_collected) = 800 := by
  sorry

end mutter_lagaan_payment_l523_523048


namespace difference_red_green_in_min_diff_basket_l523_523659

-- Defining the counts of marbles in each basket
def basketA_red := 7
def basketA_yellow := 3
def basketA_blue := 5
def basketA_purple := 6

def basketB_green := 10
def basketB_yellow := 1
def basketB_orange := 2
def basketB_red := 5

def basketC_white := 3
def basketC_yellow := 9
def basketC_black := 4
def basketC_green := 2

-- Calculating the difference between the total number of red and green marbles
-- in the basket with the smallest difference in the number of blue and orange marbles.
theorem difference_red_green_in_min_diff_basket : 
  let basketC_red := 0 in -- Since there are no red marbles in Basket C
  let basketC_green := 2 in -- Already defined above, consistent for clarity
  abs (basketC_red - basketC_green) = 2 :=
by
  sorry

end difference_red_green_in_min_diff_basket_l523_523659


namespace coins_with_specific_probabilities_impossible_l523_523445

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l523_523445


namespace arithmetic_square_root_of_sqrt_16_l523_523584

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523584


namespace simplify_sqrt7_pow6_l523_523523

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523523


namespace impossible_coins_l523_523462

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l523_523462


namespace min_value_f_l523_523791

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x)))

theorem min_value_f : ∃ x > 0, ∀ y > 0, f y ≥ f x ∧ f x = 5 / 2 :=
by
  sorry

end min_value_f_l523_523791


namespace largest_subset_A_l523_523978

def isProductOfConsecutiveInts (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * (m + 1)

def validSet (A : Set ℤ) : Prop :=
  (∀ a b k, a ∈ A → b ∈ A → ¬ isProductOfConsecutiveInts (a + b + 30 * k)) ∧
  A ⊆ {x | 0 ≤ x ∧ x ≤ 29}

def maximalValidSet (A : Set ℤ) : Prop :=
  validSet A ∧ ∀ B : Set ℤ, validSet B → Finset.card (A.to_finset) ≥ Finset.card (B.to_finset)

noncomputable def A : Set ℤ := {x | ∃ l, 0 ≤ l ∧ l < 10 ∧ x = 3 * l + 2}

theorem largest_subset_A : maximalValidSet A :=
sorry

end largest_subset_A_l523_523978


namespace measure_angle_ACD_correct_l523_523395

noncomputable def angle_sum (A B C : ℕ) : ℕ := 
  180 - A - B

noncomputable def straight_line_angle (ECD: ℕ) : ℕ :=
  180 - ECD

noncomputable def measure_angle_ACD 
  (BAC BCE ECD : ℕ)
  (h1 : BAC = 50)
  (h2 : BCE = 70)
  (h3 : ECD = 40) : ℕ :=
  let ACB := angle_sum BAC BCE in
  let DCE := straight_line_angle ECD in
  DCE - ACB

theorem measure_angle_ACD_correct 
  (BAC BCE ECD : ℕ) 
  (h1 : BAC = 50)
  (h2 : BCE = 70)
  (h3 : ECD = 40) :
  measure_angle_ACD BAC BCE ECD h1 h2 h3 = 80 := 
by 
  sorry

end measure_angle_ACD_correct_l523_523395


namespace find_a7_a8_a9_l523_523986

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l523_523986


namespace lines_are_perpendicular_l523_523800

variables (O : Point) (l1 l2 l3 : Ray)

-- Non-coplanar rays emanating from point O
axiom non_coplanar : ¬ ∃ P : Plane, O ∈ P ∧ l1 ⊆ P ∧ l2 ⊆ P ∧ l3 ⊆ P

-- Acute triangle condition for points A1 in l1, A2 in l2, A3 in l3
axiom acute_triangle :
  ∀ (A1 : Point) (A2 : Point) (A3 : Point),
    A1 ∈ l1 → A2 ∈ l2 → A3 ∈ l3 → A1 ≠ O → A2 ≠ O → A3 ≠ O →
    Triangle A1 A2 A3 → AcuteTriangle (Triangle A1 A2 A3)

-- Final goal: l1, l2, l3 are pairwise perpendicular
theorem lines_are_perpendicular :
  (angle l1 l2 = 90) ∧ (angle l2 l3 = 90) ∧ (angle l3 l1 = 90) :=
sorry

end lines_are_perpendicular_l523_523800


namespace rectangle_area_l523_523774

noncomputable theory

open Classical

variables {A B C D M N : Type} [metric_space A]

-- Definitions of the points and segments
variables (a : A) (b : A) (c : A) (d : A)

variables (Ac : ℝ) (Ae : ℝ) (Ef : ℝ) (Fc : ℝ)

-- The main theorem we want to prove
theorem rectangle_area (Ac_eq : Ac = 9) (Ae_eq : Ae = 2) (Ef_eq : Ef = 3) (Fc_eq : Fc = 4) :
  let area := 33.7 in
  calc_area a b c d Ae Ef Fc == area := sorry

end rectangle_area_l523_523774


namespace find_angle_B_l523_523014

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523014


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523597

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523597


namespace total_fare_calculation_l523_523251

def peak_hour_surcharge : ℝ := 3
def initial_ride_fee : ℝ := 2
def distance_office_to_michelle : ℝ := 4
def distance_michelle_to_coworker1 : ℝ := 6
def distance_coworker1_to_coworker2 : ℝ := 8
def charge_first_5_miles : ℝ := 2.5
def charge_additional_miles : ℝ := 3
def idle_charge_per_5_minutes : ℝ := 1.5
def idle_time_coworker1 : ℝ := 10
def idle_time_coworker2 : ℝ := 20

noncomputable def total_fare : ℝ :=
  initial_ride_fee + 
  peak_hour_surcharge + 
  (distance_office_to_michelle * charge_first_5_miles) + 
  ((5 * charge_first_5_miles) + ((distance_michelle_to_coworker1 - 5) * charge_additional_miles)) + 
  ((5 * charge_first_5_miles) + ((distance_coworker1_to_coworker2 - 5) * charge_additional_miles)) + 
  (((idle_time_coworker1 + idle_time_coworker2) / 5).ceil * idle_charge_per_5_minutes)

theorem total_fare_calculation : total_fare = 61 := by
  sorry

end total_fare_calculation_l523_523251


namespace exists_arithmetic_sequence_for_natural_numbers_l523_523294

theorem exists_arithmetic_sequence_for_natural_numbers 
  (n : Nat) (hn : n = 5 ∨ n = 1989) : ∃ (a : Nat) (d : Nat), a ≤ d ∧ ∃ (x : Fin n → Nat), 
  (∀ i : Fin n, ∃ k : Nat, x i = a + k * d) ∧ 
  (∃ (count : Nat) (Hcount3or4 : count = 3 ∨ count = 4), 
     count = Finset.card {i : Fin n | ∃ k : Nat, x i = a + k * d}.toFinset) := 
sorry

end exists_arithmetic_sequence_for_natural_numbers_l523_523294


namespace expected_shots_l523_523948

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l523_523948


namespace number_of_10_digit_integers_with_consecutive_twos_l523_523871

open Nat

-- Define the total number of 10-digit integers using only '1' and '2's
def total_10_digit_numbers : ℕ := 2^10

-- Define the Fibonacci function
def fibonacci : ℕ → ℕ
| 0    => 1
| 1    => 2
| n+2  => fibonacci (n+1) + fibonacci n

-- Calculate the 10th Fibonacci number for the problem context
def F_10 : ℕ := fibonacci 9 + fibonacci 8

-- Prove that the number of 10-digit integers with at least one pair of consecutive '2's is 880
theorem number_of_10_digit_integers_with_consecutive_twos :
  total_10_digit_numbers - F_10 = 880 :=
by
  sorry

end number_of_10_digit_integers_with_consecutive_twos_l523_523871


namespace angle_B_eq_3pi_over_10_l523_523027

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523027


namespace simplify_sqrt7_pow6_l523_523522

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523522


namespace length_of_first_train_l523_523727

variable (speed1_kmph : ℝ) (speed2_kmph : ℝ) (length2_m : ℝ) (time_s : ℝ)
variable (L1 : ℝ)

-- Define the speeds in m/s
def speed1 := speed1_kmph * (1000 / 3600)
def speed2 := speed2_kmph * (1000 / 3600)

-- Define the condition: relative speed
def relative_speed := speed1 - speed2

-- Define the equation for L1 given the other parameters
def length_first_train := L1 + length2_m = relative_speed * time_s

-- Define the value of L1
noncomputable def length_first_train_value := 10 * 54.995600351971845 - 300

theorem length_of_first_train
    (h1 : speed1_kmph = 72)
    (h2 : speed2_kmph = 36)
    (h3 : length2_m = 300)
    (h4 : time_s = 54.995600351971845)
    : L1 = 249.95600351971845 :=
by
  unfold speed1 speed2
  unfold relative_speed
  unfold length_first_train
  unfold length_first_train_value
  sorry

end length_of_first_train_l523_523727


namespace minimum_value_of_expression_l523_523055

noncomputable def expr (a b c : ℝ) : ℝ := 8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c)

theorem minimum_value_of_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  expr a b c ≥ 18 * Real.sqrt 3 := 
by
  sorry

end minimum_value_of_expression_l523_523055


namespace students_owning_both_pets_l523_523373

theorem students_owning_both_pets:
  ∀ (students total students_dog students_cat : ℕ),
    total = 45 →
    students_dog = 28 →
    students_cat = 38 →
    -- Each student owning at least one pet means 
    -- total = students_dog ∪ students_cat
    total = students_dog + students_cat - students →
    students = 21 :=
by
  intros students total students_dog students_cat h_total h_dog h_cat h_union
  sorry

end students_owning_both_pets_l523_523373


namespace annual_cereal_cost_l523_523135

theorem annual_cereal_cost :
  let boxA_cost := 3.50
  let boxB_cost := 4.00
  let boxC_cost := 5.25
  let discount := 0.10
  let weeks := 52
  let weekly_cost := boxA_cost * 1 + boxB_cost * 0.5 + boxC_cost * (1/3)
  let discounted_cost := weekly_cost * (1 - discount)
  in discounted_cost * weeks = 339.30 := sorry

end annual_cereal_cost_l523_523135


namespace inequality_solution_set_is_correct_l523_523647

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  (3 * x - 1) / (2 - x) ≥ 1

theorem inequality_solution_set_is_correct :
  { x : ℝ | inequality_solution_set x } = { x : ℝ | 3 / 4 ≤ x ∧ x < 2 } :=
by sorry

end inequality_solution_set_is_correct_l523_523647


namespace area_of_triangle_l523_523666

-- Definitions for conditions in the problem
open_locale classical -- Allow for classical reasoning

def is_inscribed_rectangle (A B C : Type) (AB BC AC : set A) (w : ℝ) : Prop :=
∃ (P Q R S P₁ Q₁ R₁ S₁ : A),
  P ∈ AB ∧ P₁ ∈ AB ∧
  Q ∈ BC ∧ Q₁ ∈ BC ∧
  R ∈ AC ∧ S ∈ AC ∧ 
  R₁ ∈ AC ∧ S₁ ∈ AC ∧
  (P - S : ℝ) = 12 ∧
  (P₁ - S₁ : ℝ) = 3 ∧
  PQRS ≅ P₁Q₁R₁S₁ -- Identical rectangles condition

theorem area_of_triangle (A B C : Type) (AB BC AC : set A) :
  is_inscribed_rectangle A B C AB BC AC →
  ∃ area : ℝ, area = 225 / 2 :=
by
  sorry -- Proof to be provided

end area_of_triangle_l523_523666


namespace molecular_weight_7_moles_correct_l523_523674

noncomputable theory

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def num_atoms_C : ℕ := 6
def num_atoms_H : ℕ := 8
def num_atoms_O : ℕ := 7
def num_moles_C6H8O7 : ℝ := 7.0

def molecular_weight_C6H8O7 : ℝ :=
  (num_atoms_C * atomic_weight_C) + (num_atoms_H * atomic_weight_H) + (num_atoms_O * atomic_weight_O)

def weight_of_7_moles_C6H8O7 : ℝ := num_moles_C6H8O7 * molecular_weight_C6H8O7

theorem molecular_weight_7_moles_correct :
  weight_of_7_moles_C6H8O7 = 1344.868 := by
  sorry

end molecular_weight_7_moles_correct_l523_523674


namespace rational_terms_in_expansion_l523_523798

theorem rational_terms_in_expansion (x : ℝ) (n : ℕ) (h : n = 8)
  (ar_seq : ∀ a b c : ℕ, b - a = c - b) : 
  ∃ k : fin 4, k.val = 3 := 
begin
  sorry
end

end rational_terms_in_expansion_l523_523798


namespace sum_of_four_circles_l523_523731

open Real

theorem sum_of_four_circles:
  ∀ (s c : ℝ), 
  (2 * s + 3 * c = 26) → 
  (3 * s + 2 * c = 23) → 
  (4 * c = 128 / 5) :=
by
  intros s c h1 h2
  sorry

end sum_of_four_circles_l523_523731


namespace max_triangle_area_l523_523844

noncomputable def given_ellipse (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

def is_tangent_line (l : ℝ → ℝ) (t : ℝ) : Prop :=
  ∀ x, (l x) = 0 ↔ x^2 + (l x)^2 = 1

def intersection_points (l : ℝ → ℝ) (t : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let x := p.1, y := p.2 in given_ellipse x y ∧ y = l x }

def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

def area_ΔAOB (t : ℝ) (l : ℝ → ℝ) [hlt : is_tangent_line l t] : ℝ :=
  let A := intersection_points l t in
  triangle_area (0, 0) (A.to_list.nth 0) (A.to_list.nth 1)

theorem max_triangle_area (t : ℝ) (hlt : |t| ≥ 1) :
  ∃ l, is_tangent_line l t ∧ area_ΔAOB t l = 1 :=
sorry

end max_triangle_area_l523_523844


namespace anniversary_celebrated_18_months_ago_l523_523371

-- Definitions for time intervals
def months_to_years (months : ℕ) : ℝ := months / 12

-- Given conditions
def in_6_months_anniversary : ℝ := 4        -- they will celebrate their 4th anniversary in 6 months
def current_years : ℝ := in_6_months_anniversary - months_to_years 6 -- current years into relationship

-- Target condition to prove
def years_ago (months : ℕ) : ℝ := months_to_years months

theorem anniversary_celebrated_18_months_ago :
  ∀ (current : ℝ) (ago : ℝ), current = 3.5 ∧ ago = 1.5 → (current - ago) = 2 :=
by {
  intros current ago h,
  cases h with h_current h_ago,
  rw [h_current, h_ago],
  sorry
}

end anniversary_celebrated_18_months_ago_l523_523371


namespace john_total_payment_in_month_l523_523959

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end john_total_payment_in_month_l523_523959


namespace plot_length_breadth_difference_l523_523555

theorem plot_length_breadth_difference
  (B : ℕ) (h₁ : B = 14)
  (h₂ : 24 * B = (24 : ℕ) * B) :
  24 - B = 10 :=
by 
  have hL : 24 = (24 : ℕ), from rfl,
  have h3 : 24 - 14 = 10, from rfl,
  rw [← h₁, hL],
  exact h3

end plot_length_breadth_difference_l523_523555


namespace correct_product_l523_523092

-- Define the two-digit nature of the number a.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Define the reversing function for a two-digit number a. 
def reverse_digits (a : ℕ) : ℕ := 
  let tens := a / 10 in 
  let units := a % 10 in 
  units * 10 + tens

-- Given conditions for the problem.
variables (a b : ℕ)

-- Condition 1: a is a two-digit number
axiom h1 : is_two_digit a

-- Condition 2: The reversed digits product
axiom h2 : reverse_digits a * b = 187

theorem correct_product : a * b = 187 :=
sorry

end correct_product_l523_523092


namespace simplify_sqrt_pow_six_l523_523469

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523469


namespace distance_between_trees_l523_523171

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (num_spaces : ℕ) (distance : ℕ)
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_spaces = num_trees - 1)
  (h4 : distance = yard_length / num_spaces) :
  distance = 18 :=
by
  sorry

end distance_between_trees_l523_523171


namespace circle_radius_l523_523629

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l523_523629


namespace absValCalculation_l523_523738

-- Define the absolute value function, as the condition requires using it
def absVal (x : Int) : Int :=
  if x < 0 then -x else x

-- Define the main theorem to prove 
theorem absValCalculation : -2 + absVal(-3) = 1 := by
  sorry

end absValCalculation_l523_523738


namespace distance_range_from_circle_to_line_l523_523123

theorem distance_range_from_circle_to_line :
  let circle_center := (0, 2)
  let circle_radius := 1
  let line := (3, -4, -2)
  let d := (abs (3 * 0 + -4 * 2 - 2)) / (sqrt (3^2 + (-4)^2))
  d = 2 →
  ∀ (x y : ℝ), (x^2 + (y - 2)^2 = 1) →
    1 ≤ (abs (3 * x + -4 * y - 2)) / (sqrt (3^2 + (-4)^2)) ≤ 3 :=
by
  intros circle_center circle_radius line d h x y h1
  sorry

end distance_range_from_circle_to_line_l523_523123


namespace years_ago_l523_523549

theorem years_ago (M D X : ℕ) (hM : M = 41) (hD : D = 23) 
  (h_eq : M - X = 2 * (D - X)) : X = 5 := by 
  sorry

end years_ago_l523_523549


namespace first_digit_base_5_of_87_l523_523240

theorem first_digit_base_5_of_87 : (nat.digits 5 87).reverse.head = 3 := by sorry

end first_digit_base_5_of_87_l523_523240


namespace area_of_cross_section_l523_523117

noncomputable def area_cross_section (H α : ℝ) : ℝ :=
  let AC := 2 * H * Real.sqrt 3 * Real.tan (Real.pi / 2 - α)
  let MK := (H / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2)
  (1 / 2) * AC * MK

theorem area_of_cross_section (H α : ℝ) :
  area_cross_section H α = (H^2 * Real.sqrt 3 * Real.tan (Real.pi / 2 - α) / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2) :=
sorry

end area_of_cross_section_l523_523117


namespace arithmetic_square_root_of_sqrt_16_l523_523586

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523586


namespace future_value_account_after_five_years_l523_523121

def futureValueOfAnnuity (R : ℝ) (i : ℝ) (n : ℕ) : ℝ :=
  R * ((1 + i) ^ n - 1) / i

theorem future_value_account_after_five_years :
  let R := 240000
  let i := 0.10
  let n := 5
  futureValueOfAnnuity R i n = 1465224 :=
by
  sorry

end future_value_account_after_five_years_l523_523121


namespace S_range_l523_523282

noncomputable def S (a : Fin 2016 → ℝ) : ℝ :=
  let x := ∑ i in (Finset.univ.filter (λ j, j.1 < j.2)), (Real.sin (a j.1 - a j.2))^2
  let y := ∑ i in (Finset.univ.filter (λ j, j.1 < j.2)), (Real.cos (a j.1 - a j.2))^2
  (7 + 23 * x) / (7 + 24 * y)

theorem S_range :
  ∀ (a : Fin 2016 → ℝ),
  ∃ (L U : ℝ),
  L = 1 / 6963841 ∧
  U = 3338497 / 3480193 ∧
  L ≤ S a ∧ S a ≤ U :=
by
  sorry

end S_range_l523_523282


namespace simplify_sqrt_7_pow_6_l523_523497

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523497


namespace optimal_selling_price_maximizes_profit_l523_523643

/-- The purchase price of a certain product is 40 yuan. -/
def cost_price : ℝ := 40

/-- At a selling price of 50 yuan, 50 units can be sold. -/
def initial_selling_price : ℝ := 50
def initial_quantity_sold : ℝ := 50

/-- If the selling price increases by 1 yuan, the sales volume decreases by 1 unit. -/
def price_increase_effect (x : ℝ) : ℝ := initial_selling_price + x
def quantity_decrease_effect (x : ℝ) : ℝ := initial_quantity_sold - x

/-- The revenue function. -/
def revenue (x : ℝ) : ℝ := (price_increase_effect x) * (quantity_decrease_effect x)

/-- The cost function. -/
def cost (x : ℝ) : ℝ := cost_price * (quantity_decrease_effect x)

/-- The profit function. -/
def profit (x : ℝ) : ℝ := revenue x - cost x

/-- The proof that the optimal selling price to maximize profit is 70 yuan. -/
theorem optimal_selling_price_maximizes_profit : price_increase_effect 20 = 70 :=
by
  sorry

end optimal_selling_price_maximizes_profit_l523_523643


namespace abs_sub_eq_five_l523_523996

theorem abs_sub_eq_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
sorry

end abs_sub_eq_five_l523_523996


namespace number_of_divisors_180_l523_523339

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l523_523339


namespace find_a2_b2_l523_523308

theorem find_a2_b2 (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 :=
by
  sorry

end find_a2_b2_l523_523308


namespace first_player_wins_iff_not_power_of_two_l523_523126

theorem first_player_wins_iff_not_power_of_two :
  ∀ (n : ℕ), n > 1 → (∃ (winning_strategy : ∀ (n : ℕ), n > 1 → bool),
  (winning_strategy n) = true ↔ ¬(∃ (k : ℕ), n = 2 ^ k)) :=
sorry

end first_player_wins_iff_not_power_of_two_l523_523126


namespace train_speeds_l523_523683

-- Definitions used in conditions
def initial_distance : ℝ := 300
def time_elapsed : ℝ := 2
def remaining_distance : ℝ := 40
def speed_difference : ℝ := 10

-- Stating the problem in Lean
theorem train_speeds :
  ∃ (v_fast v_slow : ℝ),
    v_slow + speed_difference = v_fast ∧
    (2 * (v_slow + v_fast)) = (initial_distance - remaining_distance) ∧
    v_slow = 60 ∧
    v_fast = 70 :=
by
  sorry

end train_speeds_l523_523683


namespace arithmetic_square_root_of_sqrt_16_l523_523576

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523576


namespace rhombus_area_l523_523245

theorem rhombus_area 
  (a : ℝ) (d1 d2 : ℝ)
  (h_side : a = Real.sqrt 113)
  (h_diagonal_diff : abs (d1 - d2) = 8)
  (h_geq : d1 ≠ d2) : 
  (a^2 * d1 * d2 / 2 = 194) :=
sorry -- Proof to be completed

end rhombus_area_l523_523245


namespace num_ordered_pairs_xy_eq_2200_l523_523625

/-- There are 24 ordered pairs (x, y) such that xy = 2200. -/
theorem num_ordered_pairs_xy_eq_2200 : 
  ∃ (n : ℕ), n = 24 ∧ (∃ divisors : Finset ℕ, 
    (∀ d ∈ divisors, 2200 % d = 0) ∧ 
    (divisors.card = 24)) := 
sorry

end num_ordered_pairs_xy_eq_2200_l523_523625


namespace max_beam_length_through_corridors_l523_523665

/--
Two corridors, each with a height and width of 1 meter, run perpendicular to each other. The ceiling and floor separating them have been removed to create a 1 meter by 1 meter hole. 
Assume the beam is a rigid segment of zero thickness. Additionally, the separating floor thickness is zero, meaning the floor of the upper corridor and the ceiling of the lower corridor are in the same plane.
-/
theorem max_beam_length_through_corridors
  (height width : ℝ)
  (hole : ℝ)
  (beam_thickness : ℝ)
  (same_plane : Prop)
  (perpendicular : Prop)
  (h_height : height = 1)
  (h_width : width = 1)
  (h_hole : hole = 1)
  (h_beam_thickness : beam_thickness = 0)
  (h_same_plane : same_plane)
  (h_perpendicular : perpendicular) :
  ∃ d, d = (1 / 2) * (2 + real.cbrt 4)^((3 : ℕ) / 2) :=
begin
  use (1 / 2) * (2 + real.cbrt 4)^((3 : ℕ) / 2),
  sorry
end

end max_beam_length_through_corridors_l523_523665


namespace container_volume_ratio_l523_523254

theorem container_volume_ratio (C D : ℝ) (hC: C > 0) (hD: D > 0)
  (h: (3/4) * C = (5/8) * D) : (C / D) = (5 / 6) :=
by
  sorry

end container_volume_ratio_l523_523254


namespace kitten_length_after_4_months_l523_523220

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l523_523220


namespace kitten_current_length_l523_523213

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l523_523213


namespace infinite_pairs_exists_l523_523437

open Nat

def gcd := Nat.gcd

theorem infinite_pairs_exists :
  ∃ (a b : ℕ), ∀ n : ℕ, n > 1 → 
  (gcd (a n) (b n) = 1) ∧ 
  (a n > b n) ∧
  (a n ∣ (b n ^ 2 - 5)) ∧ 
  (b n ∣ (a n ^ 2 - 5)) :=
begin
  sorry
end

end infinite_pairs_exists_l523_523437


namespace appropriate_sampling_method_l523_523130

theorem appropriate_sampling_method :
  ∀ (n1 n2 n3 total partsInSample : ℕ),
  n1 = 400 →
  n2 = 200 →
  n3 = 150 →
  total = n1 + n2 + n3 →
  partsInSample = 50 →
  (∃ method, method = "Stratified Sampling") :=
by
  intros n1 n2 n3 total partsInSample h1 h2 h3 h_total h_partsInSample
  use "Stratified Sampling"
  sorry

end appropriate_sampling_method_l523_523130


namespace tangent_line_at_2_l523_523849

def f (x : ℝ) : ℝ := 2 * x^2 - x * (deriv (f : ℝ → ℝ) 2)

theorem tangent_line_at_2 : 
  let y := deriv f 2 in
  let point := (2, f 2) in
  4 * fst point - snd point - 8 = 0 :=
by
  sorry

end tangent_line_at_2_l523_523849


namespace find_a7_a8_a9_l523_523987

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l523_523987


namespace cubic_polynomial_real_root_satisfies_l523_523843

theorem cubic_polynomial_real_root_satisfies (x p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : 27 * x^3 - 12 * x^2 - 12 * x - 4 = 0)
  (hx : x = (Real.cbrt p + Real.cbrt q + 2) / r) : 
  p + q + r = 10 :=
sorry

end cubic_polynomial_real_root_satisfies_l523_523843


namespace num_pos_divisors_180_l523_523346

theorem num_pos_divisors_180 : 
  let n := 180 in
  let prime_factorization := [(2, 2), (3, 2), (5, 1)] in
  (prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1) = 18 :=
by 
  let n := 180
  let prime_factorization := [(2, 2), (3, 2), (5, 1)]
  have num_divisors := prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1 
  show num_divisors = 18
  sorry

end num_pos_divisors_180_l523_523346


namespace ellipse_equation_triangle_area_l523_523916

-- Define the ellipse and key points
def ellipse (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def F : ℝ × ℝ := (-1, 0)
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (b : ℝ) : ℝ × ℝ := (0, b)
def C (b : ℝ) : ℝ × ℝ := (0, -b)
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define standard equation and area under conditions
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (M : F = (midpoint (A a) (C b))) :
  ellipse 3 (sqrt 8) x y := sorry

theorem triangle_area (a b : ℝ) (h1 : a = sqrt 2) (h2 : b = 1) (slope : 1) 
  (D := ((-4 / 3), (AED.1 / 3))) :
  area B C D = 4 / 3 := sorry

end ellipse_equation_triangle_area_l523_523916


namespace cos_of_angle_alpha_l523_523649

theorem cos_of_angle_alpha (x y r : ℝ) (h1 : x = 4) (h2 : y = -3) (h3 : r = 5) : real.cos (real.atan2 y x) = 4 / 5 :=
by {
  sorry
}

end cos_of_angle_alpha_l523_523649


namespace smallest_n_for_gn_gt_10_l523_523891

def g (n : ℕ) : ℕ :=
  (Real.to_digits (1 / 3^n)).2.sum

theorem smallest_n_for_gn_gt_10 :
  ∃ n : ℕ, n > 0 ∧ g(n) > 10 ∧ ∀ m : ℕ, m > 0 ∧ g(m) > 10 → n ≤ m :=
sorry

end smallest_n_for_gn_gt_10_l523_523891


namespace solution_set_f_geq_3_max_a_if_exists_x_f_leq_negative_a_squared_plus_a_plus_7_l523_523326

def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set_f_geq_3 :
  { x : ℝ | f x ≥ 3 } = { x : ℝ | x ≤ 0 ∨ x ≥ 3 } :=
by
  sorry

theorem max_a_if_exists_x_f_leq_negative_a_squared_plus_a_plus_7 :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → (a : ℝ → a ≤ 3) :=
by
  sorry

end solution_set_f_geq_3_max_a_if_exists_x_f_leq_negative_a_squared_plus_a_plus_7_l523_523326


namespace base_of_first_term_is_two_l523_523159

-- Define h as a positive integer
variable (h : ℕ) (a b c : ℕ)

-- Conditions
variables 
  (h_positive : h > 0)
  (divisor_225 : 225 ∣ h)
  (divisor_216 : 216 ∣ h)

-- Given h can be expressed as specified and a + b + c = 8
variable (h_expression : ∃ k : ℕ, h = k^a * 3^b * 5^c)
variable (sum_eight : a + b + c = 8)

-- Prove the base of the first term in the expression for h is 2.
theorem base_of_first_term_is_two : (∃ k : ℕ, k^a * 3^b * 5^c = h) → k = 2 :=
by 
  sorry

end base_of_first_term_is_two_l523_523159


namespace number_of_divisors_180_l523_523341

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l523_523341


namespace find_prob_Y_ge_1_l523_523836

variable {Ω : Type*} [MeasureSpace Ω]

-- Define binomial random variables X and Y
noncomputable def binomial (n : ℕ) (p : ℝ) : MeasureTheory.ProbabilityMassFunction (fin (n + 1)) :=
sorry

-- Define the random variables X and Y as binomial distributions
constant p : ℝ
constant X : MeasureTheory.ProbabilityMassFunction (fin 3)
constant Y : MeasureTheory.ProbabilityMassFunction (fin 4)

-- Conditions
axiom hX : X = binomial 2 p
axiom hY : Y = binomial 3 p
axiom hPX : X.prob (λ k, k ≥ 1) = 3 / 4

-- Theorem to prove
theorem find_prob_Y_ge_1 : Y.prob (λ k, k ≥ 1) = 7 / 8 :=
by {
  sorry
}

end find_prob_Y_ge_1_l523_523836


namespace find_correct_answer_l523_523962

def AnswerChoice := {A, B, C, D}

def hint1 (answer : AnswerChoice) : Prop := answer = 'A' ∨ answer = 'B'
def hint2 (answer : AnswerChoice) : Prop := answer = 'C' ∨ answer = 'D'
def hint3 (answer : AnswerChoice) : Prop := answer = 'B'
def hint4 (answer : AnswerChoice) : Prop := answer ≠ 'D'

def correct_hint (h1 h2 h3 h4 : Prop) : Prop :=
  (¬h1 ∧ ¬h2 ∧ ¬h3 ∧ h4) ∨
  (¬h1 ∧ ¬h2 ∧ h3 ∧ ¬h4) ∨
  (¬h1 ∧ h2 ∧ ¬h3 ∧ ¬h4) ∨
  (h1 ∧ ¬h2 ∧ ¬h3 ∧ ¬h4)

theorem find_correct_answer (answer : AnswerChoice) :
  correct_hint (hint1 answer) (hint2 answer) (hint3 answer) (hint4 answer) → answer = 'C' :=
by
  sorry

end find_correct_answer_l523_523962


namespace question_a_question_b_l523_523196

noncomputable def cos_to_deg (x : ℝ) : ℝ :=
Real.arccos x * 180 / Real.pi

noncomputable def deg_to_cos (x : ℝ) : ℝ :=
Real.cos (x * Real.pi / 180)

theorem question_a (ε : ℝ) (φ : ℝ) (h : ε = 45) :
  φ = 76 + 10 / 60 :=
begin
  have h_cos : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num,
  have φ_cos_sq := (2 - Real.sqrt 2) / (6 + 3 * Real.sqrt 2),
  have h_φ := Real.cos (76 + 10 / 60) * Real.cos (76 + 10 / 60),
  calc φ
    = cos_to_deg φ_cos_sq : sorry
    . ≈ 76.1667 : sorry,
end

theorem question_b (ε : ℝ) (φ : ℝ) (h : φ = 45) :
  ε = 101 + 32 / 60 :=
begin
  have h_cos : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
    by norm_num,
  have h_cos_ε_neg := -1 / 5,
  have h_cos_adj := Real.arccos (0.2),
  calc ε
    = 180 - cos_to_deg h_cos_ε_neg : sorry
    . ≈ 101 + 32 / 60 : sorry,
end

end question_a_question_b_l523_523196


namespace sequence_bounded_l523_523243

def greatest_prime_factor (x : ℕ) : ℕ := sorry  -- Define the greatest prime factor function.

def sequence (M k : ℕ) (n : ℕ) : ℕ :=
rec (λ a, if nat.prime a then a + k else a - greatest_prime_factor a) M n

theorem sequence_bounded (M k : ℕ) (hM : M > 1) (hk : k > 0) : ∃ B, ∀ n, sequence M k n ≤ B := 
sorry

end sequence_bounded_l523_523243


namespace angle_between_a_and_b_minus_a_l523_523828

-- Define vectors a and b being unit vectors
variables {a b : ℝ^3}
variable unit_a : |a| = 1
variable unit_b : |b| = 1

-- Define the condition |a - 2 * b| = 2
variable condition : |a - 2 * b| = 2

-- Define the theorem to prove
theorem angle_between_a_and_b_minus_a :
  real.arccos (-(real.sqrt 6 / 4)) = angle a (b - a) :=
sorry

end angle_between_a_and_b_minus_a_l523_523828


namespace monochromatic_isosceles_triangles_constant_l523_523376

theorem monochromatic_isosceles_triangles_constant (n k : ℕ) (h := 6 * n + 1 - k) : 
  let x3 := (h * (h-1)) / 2 
  let x0 := (k * (k-1)) / 2 
  let b := k * h in 
  x3 + x0 - (b / 2) = 
  ((6 * n + 1 - k) * (6 * n + 1 - k - 1)) / 2 + (k * (k - 1)) / 2 - ((6 * n + 1 - k) * k) / 2 := 
sorry

end monochromatic_isosceles_triangles_constant_l523_523376


namespace sum_of_terms_l523_523984

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l523_523984


namespace total_rides_correct_l523_523363

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end total_rides_correct_l523_523363


namespace simplify_sqrt_seven_pow_six_proof_l523_523514

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523514


namespace sufficient_condition_monotonically_increasing_l523_523319

def f (x : ℝ) (c : ℝ) : ℝ := 
  if x ≥ 1 then Real.logBase 2 x else x + c

theorem sufficient_condition_monotonically_increasing :
  ∀ (x y : ℝ) (c : ℝ), c = -1 → (x ≤ y → f x c ≤ f y c) :=
by
  sorry

end sufficient_condition_monotonically_increasing_l523_523319


namespace min_entries_to_alter_l523_523618

-- Define the initial 4x4 matrix
def matrix : list (list ℕ) := [[7, 12, 1, 6], [5, 8, 10, 3], [9, 2, 11, 4], [13, 6, 3, 4]]

-- Define the initial row sums and column sums
def row_sums : list ℕ := [26, 26, 26, 26]
def col_sums : list ℕ := [34, 28, 25, 17]

-- Define the final row sums and column sums after altering 4 elements
def altered_row_sums : list ℕ := [19, 18, 15, 22]
def altered_col_sums : list ℕ := [27, 20, 14, 13]

-- Lean statement to prove the minimum number of alterations
theorem min_entries_to_alter (matrix : list (list ℕ)) 
  (row_sums : list ℕ) 
  (col_sums : list ℕ) 
  (altered_row_sums : list ℕ) 
  (altered_col_sums : list ℕ) : 
  (∀ r₁ r₂ ∈ altered_row_sums, r₁ ≠ r₂) → 
  (∀ c₁ c₂ ∈ altered_col_sums, c₁ ≠ c₂) → 
  (Sum (zip_with (λ rs cs, abs (rs - cs)) row_sums col_sums) = 0) →
  matrix.length = 4 → list.length matrix = 4 → 4 :=
by
  sorry

end min_entries_to_alter_l523_523618


namespace radius_of_circle_l523_523640

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l523_523640


namespace max_area_isosceles_trapezoid_l523_523817

theorem max_area_isosceles_trapezoid (d α : ℝ) (hα : 0 < α ∧ α < π) :
  ∃ x : ℝ, (2 * x + d - 2 * x = d) ∧ (x = d / (2 * (2 - cos α))) :=
by
  sorry

end max_area_isosceles_trapezoid_l523_523817


namespace find_value_of_f_l523_523327

-- Define a power function
def power_function (x : ℝ) (a : ℝ) : ℝ := x^a

-- Given condition
def condition (a : ℝ) : Prop := power_function 2 a = 1 / 2

-- Main theorem statement
theorem find_value_of_f (a : ℝ) (h : condition a) : power_function (1 / 2) a = 2 :=
by
  sorry

end find_value_of_f_l523_523327


namespace inverse_sum_l523_523993

def g (x : ℝ) : ℝ :=
if x < 0 then -x else x^2 - 4 * x + 3

noncomputable def g_inv (y : ℝ) : Option ℝ :=
if y = -4 then none
else if y = 0 then Some 1
else if y = 4 then Some (-4)
else none

theorem inverse_sum :
  g_inv (-4) = none ∧
  g_inv (0) = some 1 ∧
  g_inv (4) = some (-4) ∧
  (g_inv 0).get_or_else 0 + (g_inv 4).get_or_else 0 = 0 :=
by
  sorry

end inverse_sum_l523_523993


namespace impossible_coins_l523_523451

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l523_523451


namespace pythagorean_triple_9_12_15_l523_523206

theorem pythagorean_triple_9_12_15 : ∃ a b c : ℕ, a = 9 ∧ b = 12 ∧ c = 15 ∧ (a * a + b * b = c * c) :=
by 
  existsi (9, 12, 15)
  split
  rfl
  split
  rfl
  split
  rfl
  sorry

end pythagorean_triple_9_12_15_l523_523206


namespace divisors_end_with_1_l523_523086

theorem divisors_end_with_1 (n : ℕ) (h : n > 0) :
  ∀ d : ℕ, d ∣ (10^(5^n) - 1) / 9 → d % 10 = 1 :=
sorry

end divisors_end_with_1_l523_523086


namespace arithmetic_sqrt_of_sqrt_16_l523_523559

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523559


namespace area_of_triangle_ABC_is_9_l523_523262

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def dist (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

def triangle_area (A B C : Point3D) : ℝ :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  if AB^2 + BC^2 = AC^2 then
    0.5 * AB * BC
  else if AB^2 + AC^2 = BC^2 then
    0.5 * AB * AC
  else if AC^2 + BC^2 = AB^2 then
    0.5 * AC * BC
  else
    -- If not a right triangle, use an alternative method (Heron's formula or vector cross product)
    sorry  -- This has to be implemented if the triangle is not right-angled.

def A : Point3D := { x := 1, y := 8, z := 11 }
def B : Point3D := { x := 0, y := 7, z := 7 }
def C : Point3D := { x := -3, y := 10, z := 7 }

theorem area_of_triangle_ABC_is_9 : triangle_area A B C = 9 := by
  sorry

end area_of_triangle_ABC_is_9_l523_523262


namespace nora_nuts_problem_l523_523252

theorem nora_nuts_problem :
  ∃ n : ℕ, (∀ (a p c : ℕ), 30 * n = 18 * a ∧ 30 * n = 21 * p ∧ 30 * n = 16 * c) ∧ n = 34 :=
by
  -- Provided conditions and solution steps will go here.
  sorry

end nora_nuts_problem_l523_523252


namespace leif_apples_oranges_l523_523967

theorem leif_apples_oranges : 
  let apples := 14
  let dozens_of_oranges := 2 
  let oranges := dozens_of_oranges * 12
  in oranges - apples = 10 :=
by 
  let apples := 14
  let dozens_of_oranges := 2
  let oranges := dozens_of_oranges * 12
  show oranges - apples = 10
  sorry

end leif_apples_oranges_l523_523967


namespace alpha_value_l523_523288

noncomputable def f (x : ℝ) (α : ℚ) : ℝ := x ^ (α:ℝ)  -- Define the function f(x) = x^α

theorem alpha_value (α : ℚ) (h : derivative (f x α) x = -4) : α = 4 := by
  sorry

end alpha_value_l523_523288


namespace value_of_a_plus_b_l523_523063

theorem value_of_a_plus_b 
  (a b : ℝ) 
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = a * x + b)
  (h₂ : ∀ x, g x = 3 * x - 6)
  (h₃ : ∀ x, g (f x) = 4 * x + 5) : 
  a + b = 5 :=
sorry

end value_of_a_plus_b_l523_523063


namespace sin_double_angle_l523_523315

/-- This Lean statement formulates the given conditions and the target proof about angle θ. --/
theorem sin_double_angle (θ : ℝ) (h₁ : (0, 0) = (0, 0)) (h₂ : ∀ x,  x = 0 → y = 0) (h₃ : ∀ x,  y = -√3 * x) : 
  sin (2 * θ) = -√3 / 2 :=
sorry

end sin_double_angle_l523_523315


namespace distance_from_base_camp_to_summit_l523_523334

-- Define the rates and conditions
def climbing_rate_Hillary := 800
def climbing_rate_Eddy := 500
def descent_rate_Hillary := 1000
def pass_time := 6 -- in hours
def stop_before_summit := 1000

-- Prove the distance from the base camp to the summit
theorem distance_from_base_camp_to_summit : ∃ D, D = 5800 ∧ (∃ t, (t * descent_rate_Hillary + 3000 = 4800)) :=
by
  -- We specify what we need, i.e., the distance D and descent time t
  let D := 4800 + 1000
  let t := (4800 - 3000) / descent_rate_Hillary
  use D
  split
  -- Proof for D
  . exact rfl
  -- Proof for t
  . use t
    have h1 : 4800 - 3000 = 1800 := rfl
    have h2 : 1800 / descent_rate_Hillary = 1.8 := rfl
    rw [h1, h2]
    have h3 : 1000 * 1.8 + 3000 = 4800 := by ring
    exact h3

-- The proof itself is just a verification of the equations and assumptions,
-- hence marked as sorry, indicating it needs to be verified by the above logic.

end distance_from_base_camp_to_summit_l523_523334


namespace platform_length_is_correct_l523_523682

noncomputable def length_of_platform (train1_speed_kmph : ℕ) (train2_speed_kmph : ℕ) (cross_time_s : ℕ) (platform_time_s : ℕ) : ℕ :=
  let train1_speed_mps := train1_speed_kmph * 5 / 18
  let train2_speed_mps := train2_speed_kmph * 5 / 18
  let relative_speed := train1_speed_mps + train2_speed_mps
  let total_distance := relative_speed * cross_time_s
  let train1_length := 2 * total_distance / 3
  let platform_length := train1_speed_mps * platform_time_s
  platform_length

theorem platform_length_is_correct : length_of_platform 48 42 12 45 = 600 :=
by
  sorry

end platform_length_is_correct_l523_523682


namespace polynomial_remainder_x1012_l523_523278

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end polynomial_remainder_x1012_l523_523278


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523596

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523596


namespace radius_of_circle_l523_523634

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l523_523634


namespace incorrect_average_calculated_initially_l523_523104

theorem incorrect_average_calculated_initially 
    (S : ℕ) 
    (h1 : (S + 75) / 10 = 51) 
    (h2 : (S + 25) = a) 
    : a / 10 = 46 :=
by
  sorry

end incorrect_average_calculated_initially_l523_523104


namespace arithmetic_square_root_of_sqrt_16_l523_523575

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523575


namespace non_similar_1500_pointed_stars_l523_523757

def number_of_non_similar_regular_stars (n : ℕ) : ℕ :=
  let phi_n := (Nat.totient n)
  let valid_m_count := phi_n - 2
  valid_m_count / 2

theorem non_similar_1500_pointed_stars :
  number_of_non_similar_regular_stars 1500 = 199 :=
by
  sorry

end non_similar_1500_pointed_stars_l523_523757


namespace functions_correct_l523_523258

noncomputable def f (x : ℝ) : ℝ := 
if h : x > 0 then 3 / x else 0

noncomputable def g (x : ℝ) : ℝ := 
if h : x > 0 then 3 / x else 0

theorem functions_correct (x : ℝ) (hx : x > 0) :
  g(f(x)) = x / (x * f(x) - 2) ∧ f(g(x)) = x / (x * g(x) - 2) :=
by
  have hf : f(x) = 3 / x := sorry
  have hg : g(x) = 3 / x := sorry
  rw [hf, hg]
  split
  · sorry
  · sorry

end functions_correct_l523_523258


namespace pi2_second_intersection_x_l523_523713

-- Define points where parabolas pass through
def pi1_p1 : ℝ × ℝ := (10, 0)
def pi1_p2 : ℝ × ℝ := (13, 0)
def pi2_p : ℝ × ℝ := (13, 0)

-- Define the x-coordinate of the vertex of pi1
def vertex_x_pi1 : ℝ := (pi1_p1.1 + pi1_p2.1) / 2 -- (10 + 13) / 2 = 11.5

-- Define the x-coordinate of the vertex of pi2
def vertex_x_pi2 : ℝ := 2 * vertex_x_pi1 -- 2 * 11.5 = 23

-- The proof problem statement in Lean 4
theorem pi2_second_intersection_x :
  ∃ t : ℝ, t = 33 ∧ (vertex_x_pi2 = (pi2_p.1 + t) / 2) :=
by
  use 33
  simp [vertex_x_pi2, pi2_p]
  sorry

end pi2_second_intersection_x_l523_523713


namespace impossible_coins_l523_523450

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l523_523450


namespace volume_removed_tetrahedra_l523_523195

theorem volume_removed_tetrahedra :
  let prism_length := 2
  let prism_width := 2
  let prism_height := 3
  let total_volume := (22 - 46 * Real.sqrt 2) / 3
  in prism_length = 2 ∧ prism_width = 2 ∧ prism_height = 3 → 
     ∀ (cuts: List (ℝ × ℝ)), -- representing the cuts as a list of pairs (each pair contains a length of the hexagon and the length of the remaining segment)
     -- Assuming the cutting of corners yields the same dimensions for all tetrahedra
     length (cuts) = 8 → 
     (cuts.map (λ cut, let (hex_side, remaining_length) := cut in
     volume_of_tetrahedron hex_side remaining_length)).sum = total_volume :=
sorry

end volume_removed_tetrahedra_l523_523195


namespace find_angle_B_correct_l523_523831

noncomputable def find_angle_B (a b c A B C : ℝ) : ℝ :=
  if (a = 1) ∧ (b = 2 * Real.cos C) ∧ (Real.sin C * Real.cos A - Real.sin (π / 4 - B) * Real.sin (π / 4 + B) = 0)
  then B
  else 0

theorem find_angle_B_correct (a b c A B C : ℝ) (h : (a = 1) ∧ (b = 2 * Real.cos C) ∧ (Real.sin C * Real.cos A - Real.sin (π / 4 - B) * Real.sin (π / 4 + B) = 0)) : B = π / 6 :=
begin
  rw find_angle_B,
  split_ifs,
  exact sorry,
end

end find_angle_B_correct_l523_523831


namespace area_of_triangle_ABC_l523_523919

theorem area_of_triangle_ABC 
  (ABCD_is_trapezoid : ∀ {a b c d : ℝ}, a + d = b + c)
  (area_ABCD : ∀ {a b : ℝ}, a * b = 24)
  (CD_three_times_AB : ∀ {a : ℝ}, a * 3 = 24) :
  ∃ (area_ABC : ℝ), area_ABC = 6 :=
by 
  sorry

end area_of_triangle_ABC_l523_523919


namespace brooke_social_studies_problems_l523_523737

theorem brooke_social_studies_problems :
  ∀ (math_problems science_problems total_minutes : Nat) 
    (math_time_per_problem science_time_per_problem soc_studies_time_per_problem : Nat)
    (soc_studies_problems : Nat),
  math_problems = 15 →
  science_problems = 10 →
  total_minutes = 48 →
  math_time_per_problem = 2 →
  science_time_per_problem = 3 / 2 → -- converting 1.5 minutes to a fraction
  soc_studies_time_per_problem = 1 / 2 → -- converting 30 seconds to a fraction
  math_problems * math_time_per_problem + science_problems * science_time_per_problem + soc_studies_problems * soc_studies_time_per_problem = 48 →
  soc_studies_problems = 6 :=
by
  intros math_problems science_problems total_minutes math_time_per_problem science_time_per_problem soc_studies_time_per_problem soc_studies_problems
  intros h_math_problems h_science_problems h_total_minutes h_math_time_per_problem h_science_time_per_problem h_soc_studies_time_per_problem h_eq
  sorry

end brooke_social_studies_problems_l523_523737


namespace smallest_positive_period_and_max_value_interval_monotonically_increasing_l523_523851

noncomputable def f (x : ℝ) : ℝ :=
  cos (π + x) * cos (3/2 * π - x) - sqrt 3 * cos x ^ 2 + sqrt 3 / 2

theorem smallest_positive_period_and_max_value :
  (∀ x, f (x + π) = f x) ∧ 
  (∃ x, f x = 1) :=
sorry

theorem interval_monotonically_increasing :
  ∀ x, (π/6 ≤ x ∧ x ≤ 2/3 * π → π/6 ≤ x ∧ x ≤ 5/12 * π ∧ ∀ y z, (π/6 ≤ y ∧ y ≤ z ∧ z ≤ 5/12 * π) → f y ≤ f z) :=
sorry

end smallest_positive_period_and_max_value_interval_monotonically_increasing_l523_523851


namespace part1_minimum_value_of_derivative_part2_range_of_a_l523_523848

noncomputable def f (x a : ℝ) : ℝ := (x^2 + 1) * log x - x^2 - a * x

theorem part1_minimum_value_of_derivative (a : ℝ):
  a = 1 → (∀ x > 0, deriv (λ x, f x 1) x = 2 * x * log x - x + (1 / x) - 1) 
  ∧ (∀ x > 0, ∃ x₁, deriv (λ x, f x 1) x₁ = -1) :=
by sorry

theorem part2_range_of_a (a : ℝ):
  (∃ x > 0, f x a = a * x * exp (2 * a * x) - x^2) → a ∈ Iic (1/exp 1) :=
by sorry

end part1_minimum_value_of_derivative_part2_range_of_a_l523_523848


namespace gas_pipe_probability_l523_523191

-- Define the conditions as Lean hypotheses
theorem gas_pipe_probability (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hxy : x + y ≤ 100) (h25x : 25 ≤ x) (h25y : 25 ≤ y)
    (h100xy : 75 ≥ x + y) :
  ∃ (p : ℝ), p = 1/16 :=
by
  sorry

end gas_pipe_probability_l523_523191


namespace arithmetic_sqrt_of_sqrt_16_l523_523563

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523563


namespace skyler_total_songs_l523_523538

open Nat

def totalSongs (hitSongs top100Songs unreleasedSongs : ℕ) : ℕ :=
  hitSongs + top100Songs + unreleasedSongs

theorem skyler_total_songs :
  ∀ (hitSongs top100Songs unreleasedSongs : ℕ), 
  hitSongs = 25 →
  top100Songs = hitSongs + 10 →
  unreleasedSongs = hitSongs - 5 →
  totalSongs hitSongs top100Songs unreleasedSongs = 80 :=
by
  intros hitSongs top100Songs unreleasedSongs h1 h2 h3
  rw [h1, h2, h3]
  unfold totalSongs
  norm_num

end skyler_total_songs_l523_523538


namespace unique_final_configuration_l523_523193

theorem unique_final_configuration (n : ℕ) : 
  ∃! (p : ℕ → ℕ), is_final_configuration p n :=
sorry

end unique_final_configuration_l523_523193


namespace sphere_radius_five_times_surface_area_l523_523650

theorem sphere_radius_five_times_surface_area (R : ℝ) (h₁ : (4 * π * R^3 / 3) = 5 * (4 * π * R^2)) : R = 15 :=
sorry

end sphere_radius_five_times_surface_area_l523_523650


namespace equal_angles_l523_523377

-- Define the right-angled triangle, its altitude, median, and angle bisector.
variable (Δ : Type) [IsRightAngledTriangle BAC]
variable (A B C D E F : Δ)
variable (AD AE AF : Line Δ)
variable (BC : Segment Δ)

-- Provide the conditions stated in the problem.
axiom angle_BAC_right : ∡ BAC = π / 2
axiom altitude_AD : IsAltitude AD BAC
axiom median_AE : IsMedian AE BC
axiom midpoint_E : IsMidpoint E BC
axiom angle_bisector_AF : IsAngleBisector AF BAC

-- State the proof problem in Lean format.
theorem equal_angles : ∡ FAD = ∡ FAE :=
sorry

end equal_angles_l523_523377


namespace simplify_sqrt7_pow6_l523_523503

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523503


namespace isosceles_triangle_area_l523_523366

theorem isosceles_triangle_area 
  (x y : ℝ)
  (h_perimeter : 2*y + 2*x = 32)
  (h_height : ∃ h : ℝ, h = 8 ∧ y^2 = x^2 + h^2) :
  ∃ area : ℝ, area = 48 :=
by
  sorry

end isosceles_triangle_area_l523_523366


namespace simplify_sqrt_seven_pow_six_proof_l523_523515

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523515


namespace num_divisors_180_l523_523351

-- Define a positive integer 180
def n : ℕ := 180

-- Define the function to calculate the number of divisors using prime factorization
def num_divisors (n : ℕ) : ℕ :=
  let factors := [(2, 2), (3, 2), (5, 1)] in
  factors.foldl (λ acc (p : ℕ × ℕ), acc * (p.snd + 1)) 1

-- The main theorem statement
theorem num_divisors_180 : num_divisors n = 18 :=
by
  sorry

end num_divisors_180_l523_523351


namespace number_of_divisors_of_180_l523_523350

theorem number_of_divisors_of_180 : 
   (nat.coprime 2 3 ∧ nat.coprime 3 5 ∧ nat.coprime 5 2 ∧ 180 = 2^2 * 3^2 * 5^1) →
   (nat.divisors_count 180 = 18) :=
by
  sorry

end number_of_divisors_of_180_l523_523350


namespace total_songs_performed_l523_523281

theorem total_songs_performed :
  ∃ N : ℕ, 
  (∃ e d o : ℕ, 
     (e > 3 ∧ e < 9) ∧ (d > 3 ∧ d < 9) ∧ (o > 3 ∧ o < 9)
      ∧ N = (9 + 3 + e + d + o) / 4) ∧ N = 6 :=
sorry

end total_songs_performed_l523_523281


namespace arithmetic_square_root_of_sqrt_16_l523_523582

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523582


namespace incorrect_expression_l523_523550

variable (V : ℚ) (X Y : ℚ) (a b : ℕ)

-- Define the non-repeating and repeating parts
def non_repeating : ℚ := X / 10^a
def repeating : ℚ := Y / (10^b - 1)

-- Define the repeating decimal number V
def V_def : V = non_repeating + repeating := sorry

-- Incorrect expression to identify
theorem incorrect_expression (hV : V = non_repeating + repeating) :
  ¬(10^a * (10^b - 1) * V = X * (Y - 1)) := 
by 
  sorry

end incorrect_expression_l523_523550


namespace find_angle_B_l523_523019

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523019


namespace kitten_length_doubling_l523_523216

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l523_523216


namespace circle_radius_l523_523628

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l523_523628


namespace checkerboard_square_count_l523_523750

theorem checkerboard_square_count : 
  ∀ (rows cols : ℕ), rows = 3 ∧ cols = 4 → 
  (∑ i in finset.range (rows + 1), ∑ j in finset.range (cols + 1), if min (rows - i) (cols - j) > 0 then 1 else 0) = 20 := 
by
  intros rows cols h
  have h1 : rows = 3 := h.1
  have h2 : cols = 4 := h.2
  sorry

end checkerboard_square_count_l523_523750


namespace lower_limit_brother_opinion_l523_523225

variables (w B : ℝ)

-- Conditions
-- Arun's weight is between 61 and 72 kg
def arun_cond := 61 < w ∧ w < 72
-- Arun's brother's opinion: greater than B, less than 70
def brother_cond := B < w ∧ w < 70
-- Arun's mother's view: not greater than 64
def mother_cond :=  w ≤ 64

-- Given the average
def avg_weight := 63

theorem lower_limit_brother_opinion (h_arun : arun_cond w) (h_brother: brother_cond w B) (h_mother: mother_cond w) (h_avg: avg_weight = (B + 64)/2) : 
  B = 62 :=
sorry

end lower_limit_brother_opinion_l523_523225


namespace pythagorean_triple_9_12_15_l523_523208

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 :=
by 
  sorry

end pythagorean_triple_9_12_15_l523_523208


namespace fraction_comparison_l523_523114

theorem fraction_comparison :
  let d := 0.33333333
  let f := (1 : ℚ) / 3
  f > d ∧ f - d = 1 / (3 * (10^8 : ℚ)) :=
by
  sorry

end fraction_comparison_l523_523114


namespace circles_intersect_l523_523246

noncomputable def circle1 (x y : ℝ) := (x - 1) ^ 2 + y ^ 2 = 1
noncomputable def circle2 (x y : ℝ) := x ^ 2 + (y - 1) ^ 2 = 2

theorem circles_intersect : 
  ∃ x y : ℝ, circle1 x y ∧ circle2 x y :=
begin
  sorry
end

end circles_intersect_l523_523246


namespace more_oranges_than_apples_l523_523965

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l523_523965


namespace area_of_closed_figure_l523_523060

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem area_of_closed_figure (f : ℝ → ℝ) 
    (h1 : is_even_function f)
    (h2 : ∀ x : ℝ, f (x + 2) = f x - f 1)
    (h3 : ∀ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x = -2*x^2 + 12*x - 18) : 
  ∫ x in (set.Icc 0 3 : set ℝ), real.abs (f x) = 2 :=
sorry

end area_of_closed_figure_l523_523060


namespace arithmetic_sqrt_of_sqrt_16_l523_523605

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523605


namespace arithmetic_square_root_of_sqrt_16_l523_523587

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523587


namespace num_valid_k_values_l523_523797

theorem num_valid_k_values : 
  ∃ (k_values : Finset ℕ), 
    k_values.card = 3 ∧
    ∀ k ∈ k_values, 0 ≤ k ∧ k < 10 ∧ 
      (∃ n, (7 * 10^3 + k * 10^2 + 5 * 10 + 2) = 12 * n) :=
by
sorr

end num_valid_k_values_l523_523797


namespace arithmetic_sqrt_sqrt_16_l523_523591

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523591


namespace sqrt_sqrt_eq_pm_two_l523_523866

-- Given definitions for the conditions
def cond1 (x y : ℝ) : Prop := (2 * x + 5 * y + 4) ^ 2 + |3 * x - 4 * y - 17| = 0

-- Lean theorem statement to prove the result
theorem sqrt_sqrt_eq_pm_two (x y : ℝ) (h : cond1 x y) : (real.sqrt (real.sqrt (4 * x - 2 * y))) = 2 ∨ (real.sqrt (real.sqrt (4 * x - 2 * y))) = -2 :=
sorry

end sqrt_sqrt_eq_pm_two_l523_523866


namespace sally_total_miles_l523_523440

noncomputable theory

/-- Definition representing Sally's pedometer reset value. --/
def pedometer_reset_value : ℕ := 99999

/-- Number of times pedometer resets. --/
def num_resets : ℕ := 50

/-- Additional steps recorded on December 31. --/
def additional_steps : ℕ := 30000

/-- Steps per mile. --/
def steps_per_mile : ℕ := 2000

/-- Steps recorded during a marathon. --/
def marathon_steps : ℕ := 50000

/-- Total miles walked by Sally during the year, rounded to the nearest whole number. --/
def total_miles_walked : ℕ :=
  let total_steps_from_resets := num_resets * (pedometer_reset_value + 1)
  let total_steps := total_steps_from_resets + additional_steps + marathon_steps
  total_steps / steps_per_mile

theorem sally_total_miles :
  total_miles_walked = 2540 :=
by
  sorry

end sally_total_miles_l523_523440


namespace divisor_problem_l523_523407

-- Defining the given condition
def n : ℕ := 2^31 * 3^19

-- Define the statement to prove
theorem divisor_problem : 
  let num_divisors_n := (31 + 1) * (19 + 1),
      num_divisors_n2 := (62 + 1) * (38 + 1),
      divisors_less_than_n := (num_divisors_n2 - 1) / 2
  in divisors_less_than_n - (num_divisors_n - 1) = 589 :=
by
  sorry

end divisor_problem_l523_523407


namespace combine_like_terms_l523_523153

theorem combine_like_terms : ∀ (x y : ℝ), -2 * x * y^2 + 2 * x * y^2 = 0 :=
by
  intros
  sorry

end combine_like_terms_l523_523153


namespace division_problem_l523_523141

theorem division_problem : 96 / (8 / 4) = 48 := 
by {
  sorry
}

end division_problem_l523_523141


namespace ratio_of_white_to_yellow_balls_l523_523197

theorem ratio_of_white_to_yellow_balls (original_white original_yellow extra_yellow : ℕ) 
(h1 : original_white = 32) 
(h2 : original_yellow = 32) 
(h3 : extra_yellow = 20) : 
(original_white : ℚ) / (original_yellow + extra_yellow) = 8 / 13 := 
by
  sorry

end ratio_of_white_to_yellow_balls_l523_523197


namespace survey_min_people_l523_523177

theorem survey_min_people (X : ℕ) (N : ℕ) (h1 : X ≥ 23) (h2 : X - 20 ≥ 23) 
(h3 : 23 ≤ 100) (h4 : 100 ≤ 2300) :
  ((X - 23) + (X - 20 - 23) + 23 + 23 = 100) → 
  (least_common_multiple 23 100 = 2300) → 
  N = 2300 := 
by
  sorry

end survey_min_people_l523_523177


namespace simplify_sqrt7_pow6_l523_523524

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523524


namespace num_ways_choose_pair_of_diff_color_socks_l523_523357

-- Define the numbers of socks of each color
def num_white := 5
def num_brown := 5
def num_blue := 3
def num_black := 3

-- Define the calculation for pairs of different colored socks
def num_pairs_white_brown := num_white * num_brown
def num_pairs_brown_blue := num_brown * num_blue
def num_pairs_white_blue := num_white * num_blue
def num_pairs_white_black := num_white * num_black
def num_pairs_brown_black := num_brown * num_black
def num_pairs_blue_black := num_blue * num_black

-- Define the total number of pairs
def total_pairs := num_pairs_white_brown + num_pairs_brown_blue + num_pairs_white_blue + num_pairs_white_black + num_pairs_brown_black + num_pairs_blue_black

-- The theorem to be proved
theorem num_ways_choose_pair_of_diff_color_socks : total_pairs = 94 := by
  -- Since we do not need to include the proof steps, we use sorry
  sorry

end num_ways_choose_pair_of_diff_color_socks_l523_523357


namespace reduced_price_per_dozen_l523_523158

variables {P R : ℝ}

theorem reduced_price_per_dozen
  (H1 : R = 0.6 * P)
  (H2 : 40 / P - 40 / R = 64) :
  R = 3 := 
sorry

end reduced_price_per_dozen_l523_523158


namespace ellipse_area_l523_523911

open Real

def ellipse_center : ℝ × ℝ := (-2, 1)

def major_axis_length : ℝ := 14

def semi_major_axis : ℝ := 7

def semi_minor_axis_square : ℝ := 784 / 13

noncomputable def semi_minor_axis : ℝ := sqrt semi_minor_axis_square

noncomputable def area_of_ellipse : ℝ := π * semi_major_axis * semi_minor_axis

theorem ellipse_area : 
  let center := (-2, 1)
  let a := semi_major_axis
  let b := semi_minor_axis
  let area := π * a * b
  in 
    center = (-2, 1) ∧ 
    a = 7 ∧ 
    b = sqrt (784 / 13) ∧ 
    area = 28 * sqrt (196 / 13) * π :=
by
  sorry

end ellipse_area_l523_523911


namespace part_a_part_b_l523_523835

-- Define p as a prime number greater than 3
variables (p : ℕ)
  (prime_p : p.prime)
  (p_gt_3 : p > 3)

-- Prove the first part: p + 1 and p - 1 are even
theorem part_a : even (p + 1) ∧ even (p - 1) := sorry

-- Prove the second part: At least one of p + 1 or p - 1 is divisible by 3
theorem part_b : (p + 1) % 3 = 0 ∨ (p - 1) % 3 = 0 := sorry

end part_a_part_b_l523_523835


namespace num_integer_areas_between_2_and_100_l523_523283

noncomputable def A (n : ℕ) : ℝ :=
  let k := Real.floor (Real.log n)
  in (1 / 2) * 9 * 10^(k : ℝ) * k * 10^(2 * k) * 101

def integer_areas (m : ℕ) (n : ℕ) : ℕ :=
  (List.range' m (n - m + 1)).countp (λ i => (A i).den = 1)

theorem num_integer_areas_between_2_and_100 : integer_areas 2 100 = 91 := sorry

end num_integer_areas_between_2_and_100_l523_523283


namespace minimum_people_surveyed_l523_523180

-- Define the conditions
variables (X : ℝ) (N : ℕ)
def liked_product_A := X / 100 * N
def liked_product_B := (X - 20) / 100 * N
def liked_both := 23 / 100 * N
def liked_neither := 23 / 100 * N

-- State the theorem to prove the minimum number of people surveyed is 2300
theorem minimum_people_surveyed :
  ∃ N : ℕ, liked_product_A N = (X / 100) * N ∧
         liked_product_B N = ((X - 20) / 100) * N ∧
         liked_both N = (23 / 100) * N ∧
         liked_neither N = (23 / 100) * N ∧
         2 * X - 20 = 100 ∧ 
         N = 2300 :=
begin
  sorry
end

end minimum_people_surveyed_l523_523180


namespace units_digit_sum_of_factorials_50_l523_523358

def units_digit (n : Nat) : Nat :=
  n % 10

def sum_of_factorials (n : Nat) : Nat :=
  (List.range' 1 n).map Nat.factorial |>.sum

theorem units_digit_sum_of_factorials_50 :
  units_digit (sum_of_factorials 51) = 3 := 
sorry

end units_digit_sum_of_factorials_50_l523_523358


namespace total_weight_of_hay_bales_l523_523657

theorem total_weight_of_hay_bales
  (initial_bales : Nat) (weight_per_initial_bale : Nat)
  (total_bales_now : Nat) (weight_per_new_bale : Nat) : 
  (initial_bales = 73 ∧ weight_per_initial_bale = 45 ∧ 
   total_bales_now = 96 ∧ weight_per_new_bale = 50) →
  (73 * 45 + (96 - 73) * 50 = 4435) :=
by
  sorry

end total_weight_of_hay_bales_l523_523657


namespace anna_gets_more_candy_l523_523224

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end anna_gets_more_candy_l523_523224


namespace sum_of_terms_l523_523985

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l523_523985


namespace interest_rate_l523_523109

theorem interest_rate (CI SI : ℝ) (P : ℝ) (t : ℕ) (h₁ : CI - SI = 36) (h₂ : P = 3600) (h₃ : t = 2) :
  let r := 0.1 in
  CI = P * (1 + r)^t - P ∧ SI = P * r * t :=
sorry

end interest_rate_l523_523109


namespace arithmetic_sqrt_of_sqrt_16_l523_523557

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523557


namespace single_elimination_games_l523_523909

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = n - 1 :=
by
  have h1 : n = 512 := h
  use 511
  sorry

end single_elimination_games_l523_523909


namespace product_of_midpoint_coordinates_l523_523144

-- Definitions based on the conditions
def p1 : (ℝ × ℝ × ℝ) := (3, -2, 4)
def p2 : (ℝ × ℝ × ℝ) := (-5, 6, -8)

-- Lean 4 theorem statement
theorem product_of_midpoint_coordinates :
  let midpoint := ( (p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2 )
  in midpoint.1 * midpoint.2 * midpoint.3 = 4 :=
by
  sorry

end product_of_midpoint_coordinates_l523_523144


namespace pencils_donated_to_library_l523_523181

theorem pencils_donated_to_library
  (total_pencils : ℕ)
  (classrooms : ℕ)
  (library_pencils : ℕ)
  (h_total : total_pencils = 935)
  (h_classrooms : classrooms = 9)
  : total_pencils % classrooms = library_pencils → library_pencils = 8 := by
  intros h
  rw [h_total, h_classrooms] at h
  exact h

end pencils_donated_to_library_l523_523181


namespace book_shelf_arrangement_l523_523882

-- Definitions for the problem conditions
def math_books := 3
def english_books := 4
def science_books := 2

-- The total number of ways to arrange the books
def total_arrangements :=
  (Nat.factorial (math_books + english_books + science_books - 6)) * -- For the groups
  (Nat.factorial math_books) * -- For math books within the group
  (Nat.factorial english_books) * -- For English books within the group
  (Nat.factorial science_books) -- For science books within the group

theorem book_shelf_arrangement :
  total_arrangements = 1728 := by
  -- Proof starts here
  sorry

end book_shelf_arrangement_l523_523882


namespace sum_of_sides_l523_523807

theorem sum_of_sides (A B C D E F : Point) (AB BC FA : ℝ) (area_ABCDEF : ℝ) :
  AB = 11 → BC = 11 → FA = 6 → area_ABCDEF = 74 →
  (∀ x y z, is_polygon [A, B, C, D, E, F] has_area area_ABCDEF → x + y = 14.4) → 
  DE + EF = 14.4 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry,
end

end sum_of_sides_l523_523807


namespace radius_of_circle_l523_523637

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l523_523637


namespace area_increase_l523_523199

-- Defining original area and the increase in side length
def original_area := 256
def side_increase := 2

-- Defining the theorem to prove
theorem area_increase (A : ℝ) (s_inc : ℝ) : 
  A = original_area ∧ s_inc = side_increase → 
  let s := Real.sqrt A in
  (s + s_inc) ^ 2 - s ^ 2 = 68 := 
by
  sorry

end area_increase_l523_523199


namespace simplify_sqrt7_pow6_l523_523506

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523506


namespace arithmetic_sqrt_sqrt_16_l523_523592

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523592


namespace simplify_sqrt_seven_pow_six_l523_523534

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523534


namespace remainder_u1011_mod_19_l523_523156

-- Define the sequence u_n based on given recurrence relations
def sequence_u : ℕ → ℕ
| 0       := 0  -- Dummy initial value, as the sequence starts from n = 1.
| 1       := 1
| 2       := 4
| (n + 3) := 4 * (sequence_u (n + 2)) - (sequence_u (n + 1))

theorem remainder_u1011_mod_19 : (sequence_u 1011) % 19 = 0 :=
sorry

end remainder_u1011_mod_19_l523_523156


namespace lower_limit_brother_l523_523902

variable (W B : Real)

-- Arun's opinion
def aruns_opinion := 66 < W ∧ W < 72

-- Brother's opinion
def brothers_opinion := B < W ∧ W < 70

-- Mother's opinion
def mothers_opinion := W ≤ 69

-- Given the average probable weight of Arun which is 68 kg
def average_weight := (69 + (max 66 B)) / 2 = 68

theorem lower_limit_brother (h₁ : aruns_opinion W) (h₂ : brothers_opinion W B) (h₃ : mothers_opinion W) (h₄ : average_weight B) :
  B = 67 := sorry

end lower_limit_brother_l523_523902


namespace log_3_eq_4_of_infinite_log_pattern_l523_523763

theorem log_3_eq_4_of_infinite_log_pattern (x : ℝ) (h : 0 < x) :
  x = 4 :=
begin
  sorry
end

end log_3_eq_4_of_infinite_log_pattern_l523_523763


namespace ratio_monkeys_snakes_l523_523729

def parrots : ℕ := 8
def snakes : ℕ := 3 * parrots
def elephants : ℕ := (parrots + snakes) / 2
def zebras : ℕ := elephants - 3
def monkeys : ℕ := zebras + 35

theorem ratio_monkeys_snakes : (monkeys : ℕ) / (snakes : ℕ) = 2 / 1 :=
by
  sorry

end ratio_monkeys_snakes_l523_523729


namespace leif_has_more_oranges_than_apples_l523_523970

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l523_523970


namespace exists_unique_polynomial_l523_523417

theorem exists_unique_polynomial (d : ℕ) (hd : d > 0) :
  ∃! S : ℕ → ℚ, ∀ n : ℕ, n ≥ 0 → 
    S n = (∑ k in Finset.range (n+1), (k : ℕ)^d) ∧
    ∃ (c : Fin (d+2) → ℚ), 
      (S n = (Finset.range (d+2)).sum (λ i, c i * (2*n+1)^(i : ℕ))) ∧
      (∀ i : Fin (d+1), c i * c (i + 1) = 0) :=
sorry

end exists_unique_polynomial_l523_523417


namespace product_of_two_numbers_l523_523648

variable (x y : ℝ)

-- conditions
def condition1 : Prop := x + y = 23
def condition2 : Prop := x - y = 7

-- target
theorem product_of_two_numbers {x y : ℝ} 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x * y = 120 := 
sorry

end product_of_two_numbers_l523_523648


namespace book_gifting_ways_l523_523653

theorem book_gifting_ways
  (types_of_books : ℕ)
  (copies_per_type : ℕ)
  (num_students : ℕ)
  (h_types : types_of_books = 5)
  (h_copies : copies_per_type ≥ 3)
  (h_students : num_students = 3) :
  (∏ i in finset.range num_students, types_of_books) = 125 :=
by
  sorry

end book_gifting_ways_l523_523653


namespace abs_sum_div_diff_sqrt_7_5_l523_523886

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end abs_sum_div_diff_sqrt_7_5_l523_523886


namespace expected_shots_l523_523945

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l523_523945


namespace angle_B_eq_3pi_over_10_l523_523024

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523024


namespace max_path_length_is_32_l523_523697
-- Import the entire Mathlib library to use its definitions and lemmas

-- Definition of the problem setup
def number_of_edges_4x4_grid : Nat := 
  let total_squares := 4 * 4
  let total_edges_per_square := 4
  total_squares * total_edges_per_square

-- Definitions of internal edges shared by adjacent squares
def distinct_edges_4x4_grid : Nat := 
  let horizontal_lines := 5 * 4
  let vertical_lines := 5 * 4
  horizontal_lines + vertical_lines

-- Calculate the maximum length of the path
def max_length_of_path_4x4_grid : Nat := 
  let degree_3_nodes := 8
  distinct_edges_4x4_grid - degree_3_nodes

-- Main statement: Prove that the maximum length of the path is 32
theorem max_path_length_is_32 : max_length_of_path_4x4_grid = 32 := by
  -- Definitions for clarity and correctness
  have h1 : number_of_edges_4x4_grid = 64 := rfl
  have h2 : distinct_edges_4x4_grid = 40 := rfl
  have h3 : max_length_of_path_4x4_grid = 32 := rfl
  exact h3

end max_path_length_is_32_l523_523697


namespace f_diff_l523_523287

-- Define the function f(n)
def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n + 1 + 1)).sum (λ i => 1 / (n + i + 1))

-- The theorem stating the main problem
theorem f_diff (k : ℕ) : 
  f (k + 1) - f k = (1 / (3 * k + 2)) + (1 / (3 * k + 3)) + (1 / (3 * k + 4)) - (1 / (k + 1)) :=
by
  sorry

end f_diff_l523_523287


namespace find_tan_angle_QDE_l523_523081

-- Define the conditions
def PointInsideTriangle (Q D E F : Type) : Prop :=
  ∃ (anglesCong : Prop), anglesCong = ( ∃ (φ : ℝ), φ = ∠QDE ∧ φ = ∠QEF ∧ φ = ∠QFD )

-- Define the lengths of the sides of the triangle
def Sides (DE EF FD : ℝ) : Prop :=
  DE = 15 ∧ EF = 17 ∧ FD = 18

-- Define the proof problem
theorem find_tan_angle_QDE (Q D E F : Type) (φ : ℝ) (a b c : ℝ) :
  PointInsideTriangle Q D E F →
  Sides 15 17 18 →
  (tan φ = 100 / 419) :=
by
  intros _ _
  sorry

end find_tan_angle_QDE_l523_523081


namespace difference_between_new_and_original_l523_523299

variables (x y : ℤ) -- Declaring variables x and y as integers

-- The original number is represented as 10*x + y, and the new number after swapping is 10*y + x.
-- We need to prove that the difference between the new number and the original number is -9*x + 9*y.
theorem difference_between_new_and_original (x y : ℤ) :
  (10 * y + x) - (10 * x + y) = -9 * x + 9 * y :=
by
  sorry -- Proof placeholder

end difference_between_new_and_original_l523_523299


namespace sum_first_15_terms_l523_523380

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def a1_plus_a15_eq_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 15 = 3

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem sum_first_15_terms (a : ℕ → ℝ) (h_arith: arithmetic_sequence a) (h_sum: a1_plus_a15_eq_three a) :
  sum_first_n_terms a 15 = 22.5 := by
  sorry

end sum_first_15_terms_l523_523380


namespace inequality_satisfied_l523_523249

theorem inequality_satisfied (x y : ℝ) : y - x > real.sqrt (x^2 + 9) ↔ y > x + real.sqrt (x^2 + 9) :=
sorry

end inequality_satisfied_l523_523249


namespace impossible_coins_l523_523458

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l523_523458


namespace Ivan_expected_shots_l523_523950

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l523_523950


namespace ZM_perpendicular_to_O1O2_l523_523054

-- Define geometry entities and their properties
structure Circle (α : Type) [RealSpace α] :=
(center : α)
(radius : ℝ)

structure Point (α : Type) :=
(coords : α)

structure Line (α : Type) :=
(points : Point α × Point α)

noncomputable def midpoint {α : Type} [RealSpace α] (A B : Point α) : Point α := sorry

noncomputable def intersection {α : Type} [RealSpace α] (A B : Line α) : Point α := sorry

-- Definitions of disjoint circles, common tangent, midpoint, and intersection points
variables {α : Type} [RealSpace α]
variable (ω Ω : Circle α)
variable (O1 O2 : Point α) (T1 T2 : Point α) (M X Y Z : Point α)
variable (t : Line α)

-- Conditions
axiom circles_disjoint : ω.center ≠ Ω.center -- Ensuring ω and Ω are disjoint circles
axiom tangent_conditions : (T1 ∈ ω) ∧ (T2 ∈ Ω) ∧ (t = Line.mk (T1, T2))
axiom midpoint_condition : M = midpoint T1 T2
axiom segment_intersections : (X ∈ ω) ∧ (Y ∈ Ω) ∧ (Line.mk (O1, O2) = Line.mk (X, Y))
axiom intersection_condition : Z = intersection (Line.mk (T1, X)) (Line.mk (T2, Y))

-- Proof goal
theorem ZM_perpendicular_to_O1O2 : ⊥ Line.mk (Z, M) (Line.mk (O1, O2)) := sorry

end ZM_perpendicular_to_O1O2_l523_523054


namespace number_of_divisors_of_180_l523_523348

theorem number_of_divisors_of_180 : 
   (nat.coprime 2 3 ∧ nat.coprime 3 5 ∧ nat.coprime 5 2 ∧ 180 = 2^2 * 3^2 * 5^1) →
   (nat.divisors_count 180 = 18) :=
by
  sorry

end number_of_divisors_of_180_l523_523348


namespace find_a_plus_b_l523_523253

variable (r a b : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions on the sequence
axiom seq_def : seq 0 = 4096
axiom seq_rule : ∀ n, seq (n + 1) = seq n * r

-- Given value
axiom r_value : r = 1 / 4

-- Given intermediate positions in the sequence
axiom seq_a : seq 3 = a
axiom seq_b : seq 4 = b
axiom seq_5 : seq 5 = 4

-- Theorem to prove
theorem find_a_plus_b : a + b = 80 := by
  sorry

end find_a_plus_b_l523_523253


namespace length_of_DF_l523_523394

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2

noncomputable def vector (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem length_of_DF (t1 t2 : ℝ) (h1 : 0 < t1) (h2 : t1 < 1) (h3 : 0 < t2) (h4 : t2 < 1) (h5 : t1 + 2 * t2 = 1) :
  ∃ df, (1 / Real.sqrt 5) ≤ df ∧ df < 1 :=
by
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let C := (0, 1, 0)
  let A1 := (0, 0, 1)
  let B1 := (1, 0, 1)
  let C1 := (0, 1, 1)
  
  let G := midpoint A1 B1
  let E := midpoint C C1
  let D := (0, t2, 0)
  let F := (t1, 0, 0)
  
  let EF := vector E F
  let GD := vector G D
  
  have h_perp : dot_product EF GD = 0 := by sorry
  let DF := vector D F
  let df := Real.sqrt (DF.1 ^ 2 + DF.2 ^ 2 + DF.3 ^ 2)
  
  sorry

end length_of_DF_l523_523394


namespace Jenson_shirts_per_day_l523_523044

variables (S : ℕ)

/-- Jenson and Kingsley have a tailoring business. Jenson makes some shirts, 
Kingsley makes 5 pairs of pants per day. Each shirt uses 2 yards of fabric 
and a pair of pants uses 5 yards of fabric. They need 93 yards of fabric 
every 3 days. Prove that Jenson makes 3 shirts per day. -/
theorem Jenson_shirts_per_day (h1 : Jenson makes some shirts per day)
                             (h2 : Kingsley makes 5 pairs of pants per day)
                             (h3 : Each shirt uses 2 yards of fabric)
                             (h4 : Each pair of pants uses 5 yards of fabric)
                             (h5 : They need 93 yards of fabric every 3 days) :
  S = 3 :=
by
  -- Define the number of yards used by Jenson and Kingsley per day
  let K_fabric_day : ℕ := 5 * 5
  let J_fabric_day := 2 * S
  let total_fabric_day := K_fabric_day + J_fabric_day
  
  -- Daily fabric needed
  let needed_fabric_day := 93 / 3
  
  -- Given equation: 25 + 2S = 31
  have h : total_fabric_day = needed_fabric_day
    := by sorry  -- Assuming the provided conditions and solving the equation

  -- Solving for S
  have eq : 25 + 2 * S = 31
    := by sorry  -- From the equation above

  -- Isolating S
  have : 2 * S = 6
    := by sorry  -- Isolate the term with S

  -- Solving for S
  have : S = 3
    := by sorry  -- Divide both sides by 2

  -- Thus, Jenson makes 3 shirts per day
  exact sorry -- S will be 3

end Jenson_shirts_per_day_l523_523044


namespace arithmetic_sqrt_of_sqrt_16_l523_523610

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523610


namespace surface_area_of_cube_l523_523684

theorem surface_area_of_cube (V : ℝ) (side : ℝ) (S : ℝ) 
  (h1 : V = 729) 
  (h2 : side = ∛729) 
  (h3 : S = 6 * (side^2)) : 
  S = 486 :=
sorry

end surface_area_of_cube_l523_523684


namespace floor_sum_inequality_l523_523085

theorem floor_sum_inequality {x : ℝ} {n : ℕ} (hx : x ≥ 0) (hn: n > 0) :
  ⌊n * x⌋ ≥ ∑ i in finset.range n, ⌊(i + 1) * x⌋ / (i + 1) := 
by
  sorry

end floor_sum_inequality_l523_523085


namespace chessboard_grains_difference_l523_523714

open BigOperators

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), grains_on_square k

theorem chessboard_grains_difference : 
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := 
by 
  -- Proof of the statement goes here.
  sorry

end chessboard_grains_difference_l523_523714


namespace arithmetic_sqrt_sqrt_16_l523_523595

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l523_523595


namespace diff_of_squares_value_l523_523151

theorem diff_of_squares_value :
  535^2 - 465^2 = 70000 :=
by sorry

end diff_of_squares_value_l523_523151


namespace find_m_decreasing_l523_523859

-- Define the power function y = x^(m^2 + 2 * m - 3)
def powerFunction (x : ℝ) (m : ℕ) : ℝ :=
  x ^ (m^2 + 2 * m - 3)

-- Define the condition that the function is decreasing in the interval (0, +∞)
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → f x > f y

-- Using the above definitions, state the problem as a theorem
theorem find_m_decreasing :
  ∀ (m : ℕ), isDecreasing (powerFunction m) → m = 0 :=
by
  sorry

end find_m_decreasing_l523_523859


namespace simplify_sqrt_seven_pow_six_l523_523536

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523536


namespace initial_pokemon_cards_l523_523399

variables (x : ℕ)

theorem initial_pokemon_cards (h : x - 2 = 1) : x = 3 := 
sorry

end initial_pokemon_cards_l523_523399


namespace estimate_black_pieces_l523_523913

theorem estimate_black_pieces (x : ℕ) : 
  (∃ (x : ℕ), (x + 9) ≠ 0 ∧ 9 / (x + 9) = 0.3) → x = 21 :=
by
  intro hx
  -- Placeholder for proof
  sorry

end estimate_black_pieces_l523_523913


namespace tiles_divisible_by_3_l523_523976

theorem tiles_divisible_by_3 (n : ℕ) 
    (h_tiling : ∀ (G : Type) [fintype G] [decidable_eq G], ∃ T_red T_blue : set (G × G),
        ∀ x : G, ∃ y : G, (x, y) ∈ T_red ∧ (x, y) ∈ T_blue ∧ 
        (∀ a b : ℕ, (a ≠ b) → ¬ ( (x + 1) * x = (y + 2) + a * b))) :
    n % 3 = 0 :=
by
  sorry

end tiles_divisible_by_3_l523_523976


namespace Mell_and_friends_total_payment_l523_523425

-- Define the prices of items
def coffee_price := 4
def cake_price := 7
def sandwich_price := 6
def ice_cream_price := 3
def water_price := 2

-- Define the quantities ordered by Mell
def mell_coffee := 2
def mell_cake := 1
def mell_sandwich := 1

-- Define the quantities ordered by each of her friends
def friend_coffee := 2
def friend_cake := 1
def friend_sandwich := 1
def friend_ice_cream := 1
def friend_water := 1

-- Define the number of friends
def num_friends := 2

-- Define the special promotion
def coffee_promotion := (2 * coffee_price + coffee_price / 2)

-- Define the discount and sales tax percentages
def discount_percentage := 0.15
def tax_percentage := 0.10

-- Define the total cost calculation
noncomputable def total_cost := 
  let mell_total := (mell_coffee * coffee_price) + (mell_cake * cake_price) + (mell_sandwich * sandwich_price)
  let friend_total := (friend_coffee * coffee_price + friend_cake * cake_price + 
                      friend_sandwich * sandwich_price + friend_ice_cream * ice_cream_price + friend_water * water_price)
  let friends_total := num_friends * coffee_promotion + (num_friends * (friend_cake * cake_price + 
                     friend_sandwich * sandwich_price + friend_ice_cream * ice_cream_price + friend_water * water_price))
  let subtotal := mell_total + friends_total
  let discount := subtotal * discount_percentage
  let discounted_total := subtotal - discount
  let tax := discounted_total * tax_percentage
  discounted_total + tax

theorem Mell_and_friends_total_payment : total_cost = 64.52 :=
  by
  -- placeholder for the proof
  sorry

end Mell_and_friends_total_payment_l523_523425


namespace sector_central_angle_l523_523842

theorem sector_central_angle (r l θ : ℝ) (h_perimeter : 2 * r + l = 8) (h_area : (1 / 2) * l * r = 4) : θ = 2 :=
by
  sorry

end sector_central_angle_l523_523842


namespace value_of_f_f_quarter_l523_523850

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 3 ^ x

theorem value_of_f_f_quarter :
  f (f (1 / 4)) = 1 / 9 :=
by
  sorry

end value_of_f_f_quarter_l523_523850


namespace num_pos_divisors_180_l523_523343

theorem num_pos_divisors_180 : 
  let n := 180 in
  let prime_factorization := [(2, 2), (3, 2), (5, 1)] in
  (prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1) = 18 :=
by 
  let n := 180
  let prime_factorization := [(2, 2), (3, 2), (5, 1)]
  have num_divisors := prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1 
  show num_divisors = 18
  sorry

end num_pos_divisors_180_l523_523343


namespace min_radius_for_area_l523_523073

theorem min_radius_for_area (r : ℝ) (π : ℝ) (A : ℝ) (h1 : A = 314) (h2 : A = π * r^2) : r ≥ 10 :=
by
  sorry

end min_radius_for_area_l523_523073


namespace domain_of_f_l523_523367

theorem domain_of_f (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end domain_of_f_l523_523367


namespace number_of_valid_four_digit_integers_equals_36_l523_523872

def is_valid_digit (d : ℕ) : Prop :=
  d ≥ 1 ∧ d ≤ 9

def valid_digits (num : ℕ) (digits : list ℕ) : Prop :=
  digits.product = 18 ∧ list.length digits = 4 ∧ (∀ d ∈ digits, is_valid_digit d)

def count_valid_numbers : ℕ :=
  -- Combinations and their corresponding permutations calculated from the solution
  12 + 12 + 12

theorem number_of_valid_four_digit_integers_equals_36 :
  count_valid_numbers = 36 := sorry

end number_of_valid_four_digit_integers_equals_36_l523_523872


namespace sqrt_mul_power_expr_l523_523235

theorem sqrt_mul_power_expr : ( (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 ) = (Real.sqrt 3 + Real.sqrt 2) := 
  sorry

end sqrt_mul_power_expr_l523_523235


namespace price_reduction_l523_523705

theorem price_reduction (C : ℝ) (h1 : C > 0) :
  let first_discounted_price := 0.7 * C
  let final_discounted_price := 0.8 * first_discounted_price
  let reduction := 1 - final_discounted_price / C
  reduction = 0.44 :=
by
  sorry

end price_reduction_l523_523705


namespace find_angle_B_l523_523015

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523015


namespace calculate_ellipse_and_minimum_area_l523_523845

noncomputable def ellipse_equation (a b : ℝ) : String := 
  s!(frac{x^2}{a^2} + frac{y^2}{b^2} = 1)

noncomputable def minimum_area_triangle : ℝ :=
  4 / 5

theorem calculate_ellipse_and_minimum_area 
  (a b : ℝ) (eccentricity : ℝ) (p : ℝ × ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : eccentricity = (sqrt 3) / 2)
  (h4 : p = (sqrt 2, sqrt 2 / 2))
  (h5 : frac{p.1 ^ 2}{a ^ 2} + frac{p.2 ^ 2}{b ^ 2} = 1) :
  ellipse_equation a b = "frac{x^2}{4} + y^2 = 1" ∧ minimum_area_triangle = 4 / 5 :=
sorry

end calculate_ellipse_and_minimum_area_l523_523845


namespace coins_with_specific_probabilities_impossible_l523_523443

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l523_523443


namespace rectangle_perimeter_l523_523136

theorem rectangle_perimeter {a b c width : ℕ} (h₁: a = 15) (h₂: b = 20) (h₃: c = 25) (w : ℕ) (h₄: w = 5) :
  let area_triangle := (a * b) / 2
  let length := area_triangle / w
  let perimeter := 2 * (length + w)
  perimeter = 70 :=
by
  sorry

end rectangle_perimeter_l523_523136


namespace find_mass_of_aluminum_l523_523210

noncomputable def mass_of_aluminum 
  (rho_A : ℝ) (rho_M : ℝ) (delta_m : ℝ) : ℝ :=
  rho_A * delta_m / (rho_M - rho_A)

theorem find_mass_of_aluminum :
  mass_of_aluminum 2700 8900 0.06 = 26 := by
  sorry

end find_mass_of_aluminum_l523_523210


namespace mp_parallel_bc_l523_523226

variables {A B C E M P : Type} [euclidean_geometry A B C] [point E] [point M] [point P]

-- Given conditions
def triangle_abc (A B C : point) : Prop := triangle A B C
def altitude_ae (A E B C : point) : Prop := altitude A E B C
def altitude_bm (B M A C : point) : Prop := altitude B M A C
def altitude_cp (C P A B : point) : Prop := altitude C P A B
def em_parallel_ab (E M A B : point) [parallel E M A B] : Prop := E M ∥ A B
def ep_parallel_ac (E P A C : point) [parallel E P A C] : Prop := E P ∥ A C

-- Target statement
theorem mp_parallel_bc {A B C E M P : point}
  (h1 : triangle_abc A B C)
  (h2 : altitude_ae A E B C)
  (h3 : altitude_bm B M A C)
  (h4 : altitude_cp C P A B)
  (h5 : em_parallel_ab E M A B)
  (h6 : ep_parallel_ac E P A C)
  : M P ∥ B C :=
sorry

end mp_parallel_bc_l523_523226


namespace total_cost_correct_l523_523424

noncomputable def totalCost : ℝ :=
  let fuel_efficiences := [15, 12, 14, 10, 13, 15]
  let distances := [10, 6, 7, 5, 3, 9]
  let gas_prices := [3.5, 3.6, 3.4, 3.55, 3.55, 3.5]
  let gas_used := distances.zip fuel_efficiences |>.map (λ p => (p.1 : ℝ) / p.2)
  let costs := gas_prices.zip gas_used |>.map (λ p => p.1 * p.2)
  costs.sum

theorem total_cost_correct : abs (totalCost - 10.52884) < 0.01 := by
  sorry

end total_cost_correct_l523_523424


namespace range_of_m_l523_523860

-- Define the conditions
def quadratic_inequality (x m : ℝ) : Prop :=
  2 * x^2 - 2 * m * x + m < 0

def contains_exactly_two_integers (A : Set ℝ) : Prop :=
  ∃ a b : ℤ, a ≠ b ∧ ∀ x : ℤ, x ∈ A → x = a ∨ x = b

-- The problem setup
theorem range_of_m (m : ℝ) (A : Set ℝ) :
  (∀ x : ℝ, quadratic_inequality x m ↔ x ∈ A) ∧
  (m > 0) ∧
  (contains_exactly_two_integers A) →
  (8/3 < m ∧ m ≤ 18/5) :=
by
  sorry

end range_of_m_l523_523860


namespace math_problem_l523_523312

-- Definitions based on conditions
def f (ω : ℝ) (x : ℝ) := (√3) * (Real.cos (ω * x))^2 - (Real.sin (ω * x)) * (Real.cos (ω * x)) - (√3) / 2
def is_tangent_line (m ω : ℝ) (x₁ x₂ x₃ : ℝ) := 
  (f ω x₁ = m) ∧ (f ω x₂ = m) ∧ (f ω x₃ = m) ∧ 
  ((x₁, x₂, x₃) = ((x₁, x₁ + π, x₁ + 2 * π)) ∨ (x₁, x₂, x₃) = ((x₂, x₂ + π, x₂ + 2 * π)) ∨ (x₁, x₂, x₃) = ((x₃, x₃ + π, x₃ + 2 * π)))

-- Problem statement
theorem math_problem (ω m x₁ x₂ x₃ A b c : ℝ) (h1 : is_tangent_line m ω x₁ x₂ x₃)
                    (h2 : (A / 2, 0) = (A / 2, 0)) (h3 : a = 4)
                    : ω = 1 ∧ m = 1 ∧ b + c ≤ 8 :=
by
  sorry

end math_problem_l523_523312


namespace answer_correct_l523_523132

-- Define the centers of the circles
variable (X Y Z : Point)

-- Define the rectangle and its sides
variables (E F G H : Point)

-- Define the circles and their congruency
variable (r : ℝ) -- radius
variable (d : ℝ := 6) -- diameter

-- Assume the circles are congruent and tangent to the sides of the rectangle
axiom circles_tangent_to_rectangle : 
  tangent (circle X r) (line_through E F) ∧
  tangent (circle X r) (line_through E G) ∧
  tangent (circle Y r) (line_through F H) ∧
  tangent (circle Y r) (line_through G H) ∧
  tangent (circle Z r) (line_through F G) ∧
  tangent (circle Z r) (line_through E H)

-- The circle centered at Y has a diameter of 6
axiom diameter_Y : diameter (circle Y r) = d

-- Circle centered at Y passes through points X and Z
axiom Y_passes_through_XZ : (circle Y r).passes_through X ∧ (circle Y r).passes_through Z

-- Calculate the area and perimeter
def calculate_area_perimeter (r : ℝ) (d : ℝ) : Prop :=
  let height := d in
  let width := 2 * d in
  let area := height * width in
  let perimeter := 2 * (height + width) in
  (area = 72) ∧ (perimeter = 36)

-- The mathematical problem in Lean statement
theorem answer_correct :
  calculate_area_perimeter r d :=
sorry

end answer_correct_l523_523132


namespace arithmetic_sequence_sum_l523_523300

variables {α : Type*} [LinearOrderedField α] (a : ℕ → α) (n : ℕ)

-- S_n is the sum of the first n terms of the arithmetic sequence a
def Sn (a : ℕ → α) (n : ℕ) : α := ∑ i in finset.range n, (a i)

-- Statement to be proved
theorem arithmetic_sequence_sum (a : ℕ → α) (n : ℕ) :
  Sn a n = n * (a 0 + a (n - 1)) / 2 :=
sorry

end arithmetic_sequence_sum_l523_523300


namespace simple_interest_is_70_l523_523122

-- Define the principal amount, rate of interest, and time.
def P : ℝ := 400
def R : ℝ := 0.175
def T : ℝ := 2

-- Define the formula for simple interest.
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Prove that the simple interest is equal to 70 when P = 400, R = 0.175, and T = 2 years.
theorem simple_interest_is_70 : simple_interest P R T = 70 :=
by
  sorry

end simple_interest_is_70_l523_523122


namespace quadratic_equation_value_l523_523827

theorem quadratic_equation_value (a : ℝ) (h₁ : a^2 - 2 = 2) (h₂ : a ≠ 2) : a = -2 :=
by
  sorry

end quadratic_equation_value_l523_523827


namespace solve_trig_eqn_l523_523544

theorem solve_trig_eqn (x y z : ℝ) (n m k : ℤ) :
  (\sin x ≠ 0) →
  (\cos y ≠ 0) →
  ((\sin^2 x + 1 / (\sin^2 x))^3 + (\cos^2 y + 1 / (\cos^2 y))^3 = 16 * \sin^2 z) →
  (x = π / 2 + n * π ∧ y = m * π ∧ z = π / 2 + k * π) :=
  sorry

end solve_trig_eqn_l523_523544


namespace probability_sum_geq_l523_523735

theorem probability_sum_geq {a b c d e f g : ℕ} (h : {a, b, c, d, e, f, g} = {1, 2, 3, 4, 5, 6, 7}) :
  let seq := [a, b, c, d, e, f, g]
  let S := seq.take 3.sum
  let T := seq.drop 4.sum
  (∃! perm : list ℕ, perm.perm seq ∧ S ≥ T ∧ S + d + T = 28) →
  (73 : ℚ) / 140 := 
by
  sorry


end probability_sum_geq_l523_523735


namespace fraction_value_l523_523408

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 5) (h3 : (∃ m : ℤ, x = m * y)) : x / y = -2 :=
sorry

end fraction_value_l523_523408


namespace Hallie_net_earnings_l523_523333

def Monday_hours := 7
def Monday_tips := 18
def Tuesday_hours := 5
def Tuesday_tips := 12
def Wednesday_hours := 7
def Wednesday_tips := 20
def Thursday_hours := 8
def Thursday_tips := 25
def Friday_hours := 6
def Friday_tips := 15
def hourly_wage := 10
def discount_rate := 0.05

def daily_earnings (hours : Nat) (tips : Nat) : Nat := hours * hourly_wage + tips

def total_earnings : Nat :=
  daily_earnings Monday_hours Monday_tips +
  daily_earnings Tuesday_hours Tuesday_tips +
  daily_earnings Wednesday_hours Wednesday_tips +
  daily_earnings Thursday_hours Thursday_tips +
  daily_earnings Friday_hours Friday_tips

def discount_amount : Nat := (total_earnings * discount_rate).toNat

def net_earnings : Nat := total_earnings - discount_amount

theorem Hallie_net_earnings :
  net_earnings = 399 := by
  -- Proof omitted
  sorry

end Hallie_net_earnings_l523_523333


namespace max_cylinders_fit_l523_523673

-- Define the conditions
def box_volume : ℝ := 9 * 8 * 6
def cylinder_volume : ℝ := π * (1.5 ^ 2) * 4

-- State the theorem
theorem max_cylinders_fit (h₁ : box_volume = 432) 
                          (h₂ : cylinder_volume = 9 * π)
                          (h₃ : 48 / π ≈ 15.286) :
  ∃ (n : ℕ), n = 15 ∧ (n ≤ ⌊box_volume / cylinder_volume⌋) :=
  sorry

end max_cylinders_fit_l523_523673


namespace probability_at_least_two_same_l523_523956

theorem probability_at_least_two_same (rolls : ℕ) (sides : ℕ) (p : Prop) :
  rolls = 5 ∧ sides = 8 ∧ p = (1 - (6720 / 32768)) ∧ p = (6512 / 8192)  :=
by
  sorry

end probability_at_least_two_same_l523_523956


namespace unique_triple_solution_l523_523244

theorem unique_triple_solution {x y z : ℤ} (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (H1 : x ∣ y * z - 1) (H2 : y ∣ z * x - 1) (H3 : z ∣ x * y - 1) :
  (x, y, z) = (5, 3, 2) :=
sorry

end unique_triple_solution_l523_523244


namespace range_of_m_l523_523324

open Real

def f (x θ : ℝ) : ℝ := (1 - 2 * cos x ^ 2) * sin (3 * π / 2 + θ) - 2 * sin x * cos x * cos (π / 2 - θ)

theorem range_of_m (θ : ℝ) (hθ1 : abs θ ≤ π / 2) 
  (h_mono : ∀ x1 x2, -3 * π / 8 ≤ x1 ∧ x1 ≤ -π / 6 ∧ -3 * π / 8 ≤ x2 ∧ x2 ≤ -π / 6 ∧ x1 < x2 → f x1 θ < f x2 θ)
  (h_m : f (π / 8) θ ≤ m) : 
  m ≥ 1 :=
sorry

end range_of_m_l523_523324


namespace find_ratio_BO_BD_l523_523431

variables {A B C C1 B1 A1 O : Point}
variables {n : ℝ} {S S_triangle ABC_area BO BD : ℝ}

-- Given side ratio and conditions
variables (equilateral_ABC : is_equilateral_triangle A B C)
variables (equilateral_A1B1C1 : is_equilateral_triangle A1 B1 C1)
variables (ratio_A1B1_AB : A1B1 / AB = n)
variables (BD_altitude : is_altitude B D)
variables (O_intersection : ∃ O, BD ∩ A1C1 = O)

-- Proof goal: find the ratio BO / BD
theorem find_ratio_BO_BD :
  BO / BD = (2 / 3) * (1 - n^2) :=
  sorry

end find_ratio_BO_BD_l523_523431


namespace speed_of_second_train_l523_523668

-- Definitions of conditions
def distance_train1 : ℝ := 200
def speed_train1 : ℝ := 50
def distance_train2 : ℝ := 240
def time_train1_and_train2 : ℝ := 4

-- Statement of the problem
theorem speed_of_second_train : (distance_train2 / time_train1_and_train2) = 60 := by
  sorry

end speed_of_second_train_l523_523668


namespace overall_percentage_l523_523721

theorem overall_percentage (p1 p2 p3 : ℝ) (P : ℝ) 
  (h1 : p1 = 60) 
  (h2 : p2 = 70) 
  (h3 : p3 = 80) 
  (H : P = (p1 + p2 + p3) / 3) : 
  P = 70 := 
by 
  rw [h1, h2, h3] at H 
  simp at H 
  exact H

#print overall_percentage -- Verifying the statement

end overall_percentage_l523_523721


namespace find_N_l523_523785

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem find_N (N : ℕ) (hN1 : N < 10000)
  (hN2 : N = 26 * sum_of_digits N) : N = 234 ∨ N = 468 := 
  sorry

end find_N_l523_523785


namespace derivative_at_five_l523_523325

noncomputable def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_five : deriv g 5 = 26 :=
sorry

end derivative_at_five_l523_523325


namespace part1_part2_part3_l523_523320

section Problem1

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem part1 : (∃ x : ℝ, f x = 1) :=
by sorry

end Problem1

section Problem2

noncomputable def g (x : ℝ) : ℝ := (Real.exp x) / x - 1

def M : set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def P (a : ℝ) : set ℝ := {x | f x > a * x}

theorem part2 (a : ℝ) : (∃ x ∈ (M ∩ (P a)), true) → a < (Real.exp 2) / 2 - 1 :=
by sorry

end Problem2

section Problem3

variable (t : ℝ) [Nonneg t]

noncomputable def S_n (n : ℕ) : ℝ := ∫ x in t..n, f x + x

theorem part3 (t : ℝ) [H : Nonneg t] : t = 0 → (∃ b : ℕ → ℝ, ∀ n, ∑ i in finset.range n, b i = S_n t n ∧ b n = (Real.exp 1 - 1) * Real.exp (n - 1)) :=
by sorry

end Problem3

end part1_part2_part3_l523_523320


namespace find_maximum_of_f_l523_523270

def f (x : ℝ) : ℝ := x * sqrt (18 - x) + sqrt (18 * x - x^3)

theorem find_maximum_of_f :
  ( ∃ x ∈ set.Icc 0 18, ∀ y ∈ set.Icc 0 18, f y ≤ f x ) → f 1 = 2 * sqrt 17 :=
by
  sorry

end find_maximum_of_f_l523_523270


namespace single_filter_price_l523_523700

theorem single_filter_price :
  let kit_price : ℝ := 87.50
  let filter_price1 : ℝ := 16.45
  let filter_price2 : ℝ := 14.05
  let total_known_cost : ℝ := 2 * filter_price1 + 2 * filter_price2
  let saved_percentage : ℝ := 0.08
  let total_individual_cost : ℝ := kit_price / (1 - saved_percentage)
  let single_filter_price := total_individual_cost - total_known_cost
  in single_filter_price = 34.11 :=
by
  sorry

end single_filter_price_l523_523700


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523598

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523598


namespace cos_identity_l523_523359

theorem cos_identity (α : ℝ) (h : Real.cos (Real.pi / 8 - α) = 1 / 6) :
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end cos_identity_l523_523359


namespace arithmetic_sequence_S9_l523_523307

theorem arithmetic_sequence_S9 (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n : ℕ, a n = a 1 + ↑n * d)
  (h2 : a 3 + a 4 + a 8 = 9) : 
  (∑ i in finset.range 9, a i) = 27 :=
by
  sorry

end arithmetic_sequence_S9_l523_523307


namespace arithmetic_sqrt_of_sqrt_16_l523_523569

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523569


namespace coprime_boxes_equal_after_operations_not_coprime_boxes_never_equal_l523_523553

-- Defining the conditions
def m : ℕ := 5 -- Or any specific value you want to test
def n : ℕ := 3 -- Or any selected value less than m
def operation (boxes : vector ℕ m) (chosen : fin n → fin m) : vector ℕ m :=
  boxes.map_with_idx (λ i x, x + if i ∈ chosen then 1 else 0)

-- Part 1: Coprime case
theorem coprime_boxes_equal_after_operations
  (h_coprime : nat.coprime m n):
  ∃ t : ℕ, ∀ k : ℕ, ∀ boxes : vector ℕ m,
  ∃ boxes' : vector ℕ m, 
  (boxes' == vector.const (boxes.head + t * n) m) :=
sorry

-- Part 2: Not coprime case
theorem not_coprime_boxes_never_equal
  (h_not_coprime : ¬ nat.coprime m n):
  ∃ (initial_distribution : vector ℕ m),
  ∀ t : ℕ, ∃ boxes' : vector ℕ m, 
  ¬ ∀ i : fin m, boxes'.nth i = boxes'.nth 0 :=
sorry

end coprime_boxes_equal_after_operations_not_coprime_boxes_never_equal_l523_523553


namespace arithmetic_sqrt_of_sqrt_16_l523_523604

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523604


namespace minimum_dot_product_l523_523846

-- Definitions and conditions
def point_on_ellipse (m n : ℝ) : Prop :=
  m^2 / 36 + n^2 / 9 = 1

def fixed_point_E : (ℝ × ℝ) := (3, 0)

def perpendicular_vectors (m n : ℝ) (Q : ℝ × ℝ) : Prop :=
  let EP := ((m - 3), n)
  let EQ := ((Q.fst - 3), (Q.snd - 0)) in
  EP.fst * EQ.fst + EP.snd * EQ.snd = 0

-- Statement to prove
theorem minimum_dot_product (m n : ℝ) (Q : ℝ × ℝ)
  (h₁ : point_on_ellipse m n)
  (h₂ : perpendicular_vectors m n Q) :
  ∃ z : ℝ, z = 6 ∧ (let EP := prod.mk (m - 3) n in
                      let QP := prod.mk (Q.fst - m) (Q.snd - n) in
                      prod.fst EP * prod.fst QP + prod.snd EP * prod.snd QP = z) :=
sorry

end minimum_dot_product_l523_523846


namespace total_rides_correct_l523_523364

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end total_rides_correct_l523_523364


namespace product_increase_by_13_exists_l523_523391

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end product_increase_by_13_exists_l523_523391


namespace circle_radius_l523_523632

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l523_523632


namespace good_sets_count_l523_523183

-- Definition of the deck conditions
def deck_conditions : Prop :=
  ∃ (cards : list ℕ), -- a list of card numbers in the deck
    (∀ (c ∈ cards), c ∈ (0 :: (list.range 1 11))) ∧ -- each card is either 0 (Joker) or in the range 1 to 10
    (cards.filter (λ x, x = 0)).length = 2 ∧ -- there are exactly 2 Jokers
    (∀ n ∈ (list.range 1 11), (cards.filter (λ x, x = n)).length = 3) -- each number from 1 to 10 appears 3 times (one for each color)

-- Definition of the value of a card numbered k
def card_value (k : ℕ) : ℕ := 2^k

-- Definition of a "good" set
def is_good_set (S : list ℕ) : Prop :=
  (S.map card_value).sum = 2004

-- Main theorem statement
theorem good_sets_count : deck_conditions →
  ∃ (good_sets : finset (finset ℕ)), good_sets.card = 1006009 ∧ ∀ S ∈ good_sets, is_good_set S :=
by
  sorry

end good_sets_count_l523_523183


namespace rhombus_shorter_diagonal_l523_523108

variable (d1 d2 : ℝ) (Area : ℝ)

def is_rhombus (Area : ℝ) (d1 d2 : ℝ) : Prop := Area = (d1 * d2) / 2

theorem rhombus_shorter_diagonal
  (h_d2 : d2 = 20)
  (h_Area : Area = 110)
  (h_rhombus : is_rhombus Area d1 d2) :
  d1 = 11 := by
  sorry

end rhombus_shorter_diagonal_l523_523108


namespace inscribed_quad_diagonal_AC_l523_523381

-- Define the conditions
variables {A B C D : Type}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D]
variables (rA rB rC : ℝ) (sideCD sideBC : ℝ)
variables (A B C D : Point)

-- Conditions of the problem
axiom cyclic_quadrilateral : is_cyclic_quadrilateral A B C D
axiom angle_ratio : angle_ratio (angle A) (angle B) (angle C) = 2 / 3 / 4
axiom length_CD : dist C D = 12
axiom length_BC : dist B C = 8 * sqrt 3 - 6

-- The statement to be proven
theorem inscribed_quad_diagonal_AC : dist A C = 20 := sorry

end inscribed_quad_diagonal_AC_l523_523381


namespace number_of_valid_four_digit_integers_equals_36_l523_523873

def is_valid_digit (d : ℕ) : Prop :=
  d ≥ 1 ∧ d ≤ 9

def valid_digits (num : ℕ) (digits : list ℕ) : Prop :=
  digits.product = 18 ∧ list.length digits = 4 ∧ (∀ d ∈ digits, is_valid_digit d)

def count_valid_numbers : ℕ :=
  -- Combinations and their corresponding permutations calculated from the solution
  12 + 12 + 12

theorem number_of_valid_four_digit_integers_equals_36 :
  count_valid_numbers = 36 := sorry

end number_of_valid_four_digit_integers_equals_36_l523_523873


namespace angle_B_eq_3pi_over_10_l523_523028

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523028


namespace log_a_y_eq_half_m_minus_n_l523_523804

open Real

noncomputable theory
variables (a x y m n : ℝ)

-- Defining the conditions
def condition1 : Prop := x^2 + y^2 = 1
def condition2 : Prop := x > 0
def condition3 : Prop := y > 0
def condition4 : Prop := log a (1 + x) = m
def condition5 : Prop := log a (1 / (1 - x)) = n

-- Problem statement
theorem log_a_y_eq_half_m_minus_n 
  (h1 : condition1 a x y m n)
  (h2 : condition2 a x y m n)
  (h3 : condition3 a x y m n)
  (h4 : condition4 a x y m n)
  (h5 : condition5 a x y m n) : 
  log a y = (1 / 2) * (m - n) := 
sorry

end log_a_y_eq_half_m_minus_n_l523_523804


namespace circumscribed_sphere_volume_l523_523928

theorem circumscribed_sphere_volume (SA ABC : ℝ) (h1 : SA = 2 * sqrt 3) (h2 : ABC = sqrt 3) :
  let r := 1
  let d := sqrt 3
  let R := sqrt (r^2 + d^2)
  volume := (4/3) * π * R^3
  volume = (32/3) * π :=
by
  sorry

end circumscribed_sphere_volume_l523_523928


namespace find_k_l523_523083

noncomputable def a : ℚ := sorry -- Represents positive rational number a
noncomputable def b : ℚ := sorry -- Represents positive rational number b

def minimal_period (x : ℚ) : ℕ := sorry -- Function to determine minimal period of a rational number

-- Conditions as definitions
axiom h1 : minimal_period a = 30
axiom h2 : minimal_period b = 30
axiom h3 : minimal_period (a - b) = 15

-- Statement to prove smallest natural number k such that minimal period of (a + k * b) is 15
theorem find_k : ∃ k : ℕ, minimal_period (a + k * b) = 15 ∧ ∀ n < k, minimal_period (a + n * b) ≠ 15 :=
sorry

end find_k_l523_523083


namespace simplify_expression_l523_523467

theorem simplify_expression (r : ℝ) : (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := 
by
  sorry

end simplify_expression_l523_523467


namespace range_of_func_l523_523758

noncomputable def log_base_inv3 (t : ℝ) := Real.log t / Real.log (1/3)

def func (x : ℝ) := log_base_inv3 (x^2 - 6 * x + 18)

theorem range_of_func : set.range func = set.Iic (-2) := by
  sorry

end range_of_func_l523_523758


namespace average_surfers_correct_l523_523101

-- Define the number of surfers for each day
def surfers_first_day : ℕ := 1500
def surfers_second_day : ℕ := surfers_first_day + 600
def surfers_third_day : ℕ := (2 / 5 : ℝ) * surfers_first_day

-- Average number of surfers
def average_surfers : ℝ := (surfers_first_day + surfers_second_day + surfers_third_day) / 3

theorem average_surfers_correct :
  average_surfers = 1400 := 
  by 
    sorry

end average_surfers_correct_l523_523101


namespace geometric_arithmetic_l523_523296

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a 1 * q ^ n

noncomputable def arith_seq (b : ℕ → ℝ) : Prop :=
∀ n, 2 * b (n + 1) = b n + b (n + 2)

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ k in range n, a k

theorem geometric_arithmetic {a : ℕ → ℝ} {q : ℝ} (hq : q > 0 ∧ q ≠ 1)
  (hgs : geom_seq a q)
  (has : arith_seq (λ n, [4, 3, 2][n] * a (n + 1)))
  (hsum : sum_first_n_terms a 4 = 15) :
  (∀ n, b n = 2^n-1) ∧ (∀ n, sum_first_n_terms (λ n, a n + 2 * n) n = 2^n - 1 + n * (n + 1)) :=
sorry

end geometric_arithmetic_l523_523296


namespace least_k_for_squares_l523_523268

theorem least_k_for_squares (k : ℕ) (n : ℕ) : 
  ∃ (x y z : ℤ), x^2 + y^2 + z^2 = n ∧ ¬ ∃ (a b : ℤ), a^2 + b^2 = n := 
by {
  let n := 2010
  let k := 3,
  sorry
}

end least_k_for_squares_l523_523268


namespace gcd_condition_ellipse_equation_correct_main_l523_523734

noncomputable def parametric_ellipse (t : ℝ) : ℝ × ℝ :=
  ( (3 * (Real.sin t + 2)) / (3 - Real.cos t),
    (4 * (Real.cos t - 6)) / (3 - Real.cos t) )

def ellipse_standard_form (x y : ℝ) : ℝ :=
  9 * x ^ 2 + 12 * x * y + 8 * y ^ 2 + 8 * x + 136 * y + 560

theorem gcd_condition : Int.gcd 9 (Int.gcd 12 (Int.gcd 8 (Int.gcd 8 (Int.gcd 136 560)))) = 1 :=
begin
  -- This corresponds to ensuring A, B, C, D, E, F are coprime
  sorry
end

theorem ellipse_equation_correct (x y : ℝ) (t : ℝ)
  (h : (x, y) = parametric_ellipse t) :
  ellipse_standard_form x y = 0 :=
begin
  -- This corresponds to showing the equation is correct for all (x, y) from the parametric form
  sorry
end

theorem main : 9 + 12 + 8 + 8 + 136 + 560 = 733 :=
by norm_num

end gcd_condition_ellipse_equation_correct_main_l523_523734


namespace house_total_volume_l523_523185

def room_volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def bathroom_volume := room_volume 4 2 7
def bedroom_volume := room_volume 12 10 8
def livingroom_volume := room_volume 15 12 9

def total_volume := bathroom_volume + bedroom_volume + livingroom_volume

theorem house_total_volume : total_volume = 2636 := by
  sorry

end house_total_volume_l523_523185


namespace coordinates_of_T_l523_523257

open Real EuclideanGeometry

-- Definitions of points in the Euclidean space
structure Point : Type :=
  (x : ℝ)
  (y : ℝ)

-- Definition of square and calculation properties
def is_square (O P Q R : Point) : Prop :=
  O.x = 0 ∧ O.y = 0 ∧ Q.x = 3 ∧ Q.y = 3 ∧
  P.x = 3 ∧ P.y = 0 ∧ R.x = 0 ∧ R.y = 3 ∧
  dist O Q = dist P R ∧ dist O P = dist O R ∧
  dist P Q = dist O Q / Real.sqrt 2

-- Area of a square given side length
def square_area (O P Q R : Point) : ℝ :=
  let side := dist O P in
  side * side

-- Area of a triangle using base and height
def triangle_area (P Q T : Point) : ℝ :=
  0.5 * dist P Q * abs (T.y - P.y)

-- The theorem to be proven
theorem coordinates_of_T 
  (O P Q R T : Point)
  (h_square : is_square O P Q R)
  (h_area : triangle_area P Q T = 2 * square_area O P Q R) : 
  T = {x := 3, y := 12} :=
by {
  sorry
}

end coordinates_of_T_l523_523257


namespace largest_value_satisfies_abs_equation_l523_523790

theorem largest_value_satisfies_abs_equation (x : ℝ) : |5 - x| = 15 + x → x = -5 := by
  intros h
  sorry

end largest_value_satisfies_abs_equation_l523_523790


namespace max_ab_l523_523830

theorem max_ab
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a / 3 + b / 4 = 1) :
  ab ≤ 3 :=
begin
  -- sorry means we skip the proof.
  sorry,
end

end max_ab_l523_523830


namespace log_equation_solution_l523_523098

-- Problem statement
theorem log_equation_solution (x : ℕ) (h1 : log 2 8 = 3) (h2 : log 2 2 = 1) :
  (log 4 x - 3 * log 2 8 = 1 - log 2 2) → x = 262144 := 
sorry

end log_equation_solution_l523_523098


namespace lina_walk_probability_l523_523168

/-- Total number of gates -/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet -/
def gate_distance : ℕ := 50

/-- Maximum distance in feet Lina can walk to be within the desired range -/
def max_walk_distance : ℕ := 200

/-- Number of gates Lina can move within the max walk distance -/
def max_gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- Total possible gate pairs for initial and new gate selection -/
def total_possible_pairs : ℕ := num_gates * (num_gates - 1)

/-- Total number of favorable gate pairs where walking distance is within the allowed range -/
def total_favorable_pairs : ℕ :=
  let edge_favorable (g : ℕ) := if g = 1 ∨ g = num_gates then 4
                                else if g = 2 ∨ g = num_gates - 1 then 5
                                else if g = 3 ∨ g = num_gates - 2 then 6
                                else if g = 4 ∨ g = num_gates - 3 then 7 else 8
  (edge_favorable 1) + (edge_favorable 2) + (edge_favorable 3) +
  (edge_favorable 4) + (num_gates - 8) * 8

/-- Probability that Lina walks 200 feet or less expressed as a reduced fraction -/
def probability_within_distance : ℚ :=
  (total_favorable_pairs : ℚ) / (total_possible_pairs : ℚ)

/-- p and q components of the fraction representing the probability -/
def p := 7
def q := 19

/-- Sum of p and q -/
def p_plus_q : ℕ := p + q

theorem lina_walk_probability : p_plus_q = 26 := by sorry

end lina_walk_probability_l523_523168


namespace alpha_beta_equiv_l523_523811

theorem alpha_beta_equiv (p : ℕ) (hp : p.prime) :
  (∃ α : ℕ, 0 < α ∧ p ∣ α * (α - 1) + 3) ↔ (∃ β : ℕ, 0 < β ∧ p ∣ β * (β - 1) + 25) := by
  sorry

end alpha_beta_equiv_l523_523811


namespace correct_expression_is_B_l523_523732

theorem correct_expression_is_B :
  let sqrt3_squared := real.sqrt (3 ^ 2)
  let neg_sqrt3_squared := -sqrt3_squared
  sqrt3_squared = 3 ∧ neg_sqrt3_squared = -3 ∧
  sqrt ((-3) ^ 2) ≠ -3 ∧ sqrt ((-3) ^ 2) ≠ ±3 ∧ sqrt (3 ^ 2) ≠ ±3 :=
by
  sorry

end correct_expression_is_B_l523_523732


namespace arithmetic_square_root_of_sqrt_16_l523_523577

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523577


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523599

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523599


namespace evaluate_f_at_2_l523_523889

def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem evaluate_f_at_2 : f 2 = 4 :=
by
  -- Proof goes here
  sorry

end evaluate_f_at_2_l523_523889


namespace find_x_when_f_eq_2_l523_523853

def f (x : ℝ) : ℝ :=
if x ≤ 1 then
  3 * x + 1
else
  -x

theorem find_x_when_f_eq_2 (x : ℝ) (h : f x = 2) : x = 1 / 3 := 
by
  sorry

end find_x_when_f_eq_2_l523_523853


namespace average_difference_sasha_asha_l523_523116

theorem average_difference_sasha_asha 
  (differences : List ℤ)
  (h : differences = [15, -5, 25, 0, -15, 10, 20]) :
  (differences.sum / differences.length : ℚ) = 50 / 7 :=
by
  rw [h, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_nil, 
    List.length_cons, List.length_cons, List.length_cons, List.length_cons, List.length_cons, List.length_cons, List.length_nil]
  norm_num
  sorry

end average_difference_sasha_asha_l523_523116


namespace num_elements_in_intersection_l523_523825

open Set

def setA : Set (ℝ × ℝ) := {p | p.2 = p.1}
def setB : Set (ℝ × ℝ) := {p | |p.1| + |p.2| = 1}

theorem num_elements_in_intersection : (setA ∩ setB).finite.card = 2 :=
by
  sorry

end num_elements_in_intersection_l523_523825


namespace intersect_points_and_calculation_l523_523238

-- Definitions
def f (x : ℝ) : ℝ := (x - 2) ^ 2 - 3
def g (x : ℝ) : ℝ := -f(x)
def h (x : ℝ) : ℝ := f(-x)

-- Problem statement
theorem intersect_points_and_calculation :
  let a := set.count (set_of (λ x, f x = g x))
  let b := set.count (set_of (λ x, f x = h x)) in
  10 * a + b = 22 :=
by
  -- Placeholder for the proof
  sorry

end intersect_points_and_calculation_l523_523238


namespace simplify_sqrt_power_l523_523480

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523480


namespace product_increase_by_13_l523_523389

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end product_increase_by_13_l523_523389


namespace f_odd_inequality_solution_l523_523289

def f (x: Real) : Real := 1 - (2 / (2^x + 1))

theorem f_odd (x : Real) : f (-x) = -f x :=
by
  sorry

theorem inequality_solution (t : Real) : f t + f (t^2 - t - 1) < 0 ↔ -1 < t ∧ t < 1 :=
by
  sorry

end f_odd_inequality_solution_l523_523289


namespace S_9_value_l523_523305

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

theorem S_9_value (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : a 3 + a 4 + a 8 = 9) : 
  (a 4 + a 4 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 > 9) false := 
begin
  sorry
end

end S_9_value_l523_523305


namespace collinear_BQP_l523_523914

variables {A B C D P Q : Type*} [affine_space ℝ A]
variables {a b c d p q : A}
variables [add_comm_group V] [module ℝ V] [affine_space V A]

def parallelogram (ABCD : set A) : Prop :=
  ∃ (a b c d : A), -- corners of the parallelogram
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  a +ᵥ (b -ᵥ a) = b ∧
  a +ᵥ (d -ᵥ a) = d ∧
  c +ᵥ (b -ᵥ d) = b ∧
  c +ᵥ (d -ᵥ c) = d

def ratio_points_segment (A B : A) (ratio : ℝ) (P : A) : Prop :=
  segment_ratio A B ratio P

def collinear (A B C : A) : Prop :=
  ∃ (m : affine_subspace ℝ A), m.affine_span = line_span A ⇹ line_span B ⇹ line_span C

variables (parallelogram ABCD)

theorem collinear_BQP (hP: ratio_points_segment c d (1:2) p)
(hQ: ratio_points_segment c a (1:3) q) :
collinear b q p :=
sorry

end collinear_BQP_l523_523914


namespace simplify_sqrt_seven_pow_six_proof_l523_523508

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523508


namespace profit_function_max_profit_expense_l523_523776

theorem profit_function (a x t y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ a^2 - 3*a + 3) (h3 : a > 0)
    (h4 : t = 5 - 2 / (x+1)) (h5 : y = t * (4 + 20 / t) - (10 + 2 * t) - x) :
    y = 20 - x - 4 / (x + 1) := 
by sorry

theorem max_profit_expense (a x y : ℝ) (h1 : 0 < a ∧ a ≤ 1 ∨ 2 ≤ a ∨ 1 < a ∧ a < 2)
    (h2 : x = if (0 < a ∧ a ≤ 1 ∨ 2 ≤ a) then 1 else a^2 - 3 * a + 3)
    (h3 : y = 20 - x - 4 / (x+1)) :
    ∀ x y, y ≤ 17 :=
by sorry

end profit_function_max_profit_expense_l523_523776


namespace number_of_valid_sequences_l523_523998

def valid_sequence (b : List ℕ) :=
  b.head = 1 ∧ b.getLast! = 12 ∧ (∀ i ∈ b.tail, (i + 1 ∈ b.takeWhile (≠ i) ∨ i - 1 ∈ b.takeWhile (≠ i)))

theorem number_of_valid_sequences : 
  let bs := {b : List ℕ | valid_sequence b ∧ (List.range 12 = b.perm)} in
  bs.size = 256 :=
sorry

end number_of_valid_sequences_l523_523998


namespace michael_made_small_balls_l523_523074

def num_small_balls (total_bands : ℕ) (bands_per_small : ℕ) (bands_per_large : ℕ) (num_large : ℕ) : ℕ :=
  (total_bands - num_large * bands_per_large) / bands_per_small

theorem michael_made_small_balls :
  num_small_balls 5000 50 300 13 = 22 :=
by
  sorry

end michael_made_small_balls_l523_523074


namespace simplify_sqrt_seven_pow_six_proof_l523_523511

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523511


namespace remainder_division_l523_523675

noncomputable def p (x : ℝ) := 4 * x^8 - x^7 + 5 * x^6 - 7 * x^4 + 3 * x^3 - 9
noncomputable def d (x : ℝ) := 3 * x - 6
def remainder : ℝ := p 2

theorem remainder_division :
  remainder = 1119 :=
by
  sorry

end remainder_division_l523_523675


namespace pythagorean_triple_9_12_15_l523_523209

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 :=
by 
  sorry

end pythagorean_triple_9_12_15_l523_523209


namespace arithmetic_sqrt_of_sqrt_16_l523_523611

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523611


namespace nested_logarithm_l523_523771

noncomputable def logarithm_series : ℝ := 
  if h : ∃ x : ℝ, 3^x = x + 81 then classical.some h else 0

theorem nested_logarithm (h : ∃ x : ℝ, 3^x = x + 81) :
  logarithm_series = classical.some h ∧ 
  abs (logarithm_series - 4.5) < 1 :=
by
  sorry

end nested_logarithm_l523_523771


namespace min_distance_curve_C1_C3_l523_523163

-- Define the problem conditions
def polar_eq_curve_C1 := ∀ θ : ℝ, let ρ := 24 / (4 * Real.cos θ + 3 * Real.sin θ) in 
  4 * ρ * Real.cos θ + 3 * ρ * Real.sin θ = 24

def parametric_eq_curve_C2 := ∀ θ : ℝ, let x := Real.cos θ, y := Real.sin θ in 
  x^2 + y^2 = 1

def transformation_C2_to_C3 (x y : ℝ) := (2 * Real.sqrt 2 * x, 2 * y)

def cartesian_eq_curve_C1 := ∀ (x y : ℝ), 
  (4 * x + 3 * y - 24 = 0)

def cartesian_eq_curve_C3 := ∀ (x y : ℝ), 
  (x^2 / 8 + y^2 / 4 = 1)

-- Prove the minimum distance |MN| is (24 - 2*sqrt(41))/5 given the conditions
theorem min_distance_curve_C1_C3 : 
  (polar_eq_curve_C1) → (parametric_eq_curve_C2) → (cartesian_eq_curve_C1) → (cartesian_eq_curve_C3) →
  ∃ (α : ℝ), Real.sin (α + Real.atan (4 * Real.sqrt 2 / 3)) = 1 → 
  let d := (24 - 2 * Real.sqrt 41) / 5 in d = 24 - 2* Real.sqrt 41 :=
sorry

end min_distance_curve_C1_C3_l523_523163


namespace one_eighth_of_two_pow_36_eq_two_pow_y_l523_523895

theorem one_eighth_of_two_pow_36_eq_two_pow_y (y : ℕ) : (2^36 / 8 = 2^y) → (y = 33) :=
by
  sorry

end one_eighth_of_two_pow_36_eq_two_pow_y_l523_523895


namespace irrational_between_3_and_4_l523_523680

def exists_irrational_between_3_and_4 : Prop :=
  ∃ (x : ℝ), irrational x ∧ 3 < x ∧ x < 4

theorem irrational_between_3_and_4 : exists_irrational_between_3_and_4 :=
  sorry

end irrational_between_3_and_4_l523_523680


namespace sequence_count_16_l523_523748

def count_sequences (n : ℕ) : ℕ := 
  if n = 1 then 1
  else if n = 2 then 1
  else count_sequences (n - 2) + count_sequences (n - 1)

theorem sequence_count_16 : count_sequences 16 = 682 := 
  sorry

end sequence_count_16_l523_523748


namespace paving_stone_length_l523_523134

theorem paving_stone_length
  (length_courtyard : ℝ)
  (width_courtyard : ℝ)
  (num_paving_stones : ℝ)
  (width_paving_stone : ℝ)
  (total_area : ℝ := length_courtyard * width_courtyard)
  (area_per_paving_stone : ℝ := (total_area / num_paving_stones))
  (length_paving_stone : ℝ := (area_per_paving_stone / width_paving_stone)) :
  length_courtyard = 20 ∧
  width_courtyard = 16.5 ∧
  num_paving_stones = 66 ∧
  width_paving_stone = 2 →
  length_paving_stone = 2.5 :=
by {
   sorry
}

end paving_stone_length_l523_523134


namespace circle_radius_l523_523626

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l523_523626


namespace arithmetic_sqrt_of_sqrt_16_l523_523609

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523609


namespace greta_hours_worked_l523_523869

-- Define the problem conditions
def greta_hourly_rate := 12
def lisa_hourly_rate := 15
def lisa_hours_to_equal_greta_earnings := 32
def greta_earnings (hours_worked : ℕ) := greta_hourly_rate * hours_worked
def lisa_earnings := lisa_hourly_rate * lisa_hours_to_equal_greta_earnings

-- Problem statement
theorem greta_hours_worked (G : ℕ) (H : greta_earnings G = lisa_earnings) : G = 40 := by
  sorry

end greta_hours_worked_l523_523869


namespace condition_for_sqrt_equality_l523_523362

theorem condition_for_sqrt_equality (x y : ℝ) (h : x * y ≠ 0) : 
  sqrt (4 * x^2 * y^3) = -2 * x * y * sqrt y ↔ x < 0 ∧ y > 0 :=
by
  sorry

end condition_for_sqrt_equality_l523_523362


namespace solve_trig_eqn_l523_523545

theorem solve_trig_eqn (x y z : ℝ) (n m k : ℤ) :
  (\sin x ≠ 0) →
  (\cos y ≠ 0) →
  ((\sin^2 x + 1 / (\sin^2 x))^3 + (\cos^2 y + 1 / (\cos^2 y))^3 = 16 * \sin^2 z) →
  (x = π / 2 + n * π ∧ y = m * π ∧ z = π / 2 + k * π) :=
  sorry

end solve_trig_eqn_l523_523545


namespace leak_empty_time_l523_523704

theorem leak_empty_time
  (R : ℝ) (L : ℝ)
  (hR : R = 1 / 8)
  (hRL : R - L = 1 / 10) :
  1 / L = 40 :=
by
  sorry

end leak_empty_time_l523_523704


namespace impossible_coins_l523_523457

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l523_523457


namespace simplify_sqrt_power_l523_523479

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523479


namespace arithmetic_sqrt_of_sqrt_16_l523_523556

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523556


namespace simplify_sqrt_power_l523_523478

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523478


namespace anna_more_candy_than_billy_l523_523222

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end anna_more_candy_than_billy_l523_523222


namespace solve_alcohol_mixture_problem_l523_523655

theorem solve_alcohol_mixture_problem (x y : ℝ) 
(h1 : x + y = 18) 
(h2 : 0.75 * x + 0.15 * y = 9) 
: x = 10.5 ∧ y = 7.5 :=
by 
  sorry

end solve_alcohol_mixture_problem_l523_523655


namespace total_surface_area_of_pyramid_l523_523612

-- Conditions
variables (a α β : ℝ)
variables [fact (0 < α)] [fact (α < π/2)] [fact (0 < β)] [fact (β < π/2)]

-- Prove the total surface area of the pyramid
theorem total_surface_area_of_pyramid 
  (a α β : ℝ) [fact (0 < α)] [fact (α < π/2)] [fact (0 < β)] [fact (β < π/2)] :
  total_surface_area = (2 * a^2 * sin α * cos^2(β/2)) / cos β := by
  sorry

end total_surface_area_of_pyramid_l523_523612


namespace function_monotonic_range_l523_523898

theorem function_monotonic_range (x1 x2 : ℝ) (a : ℝ) :
  ¬(x1 = x2) →
  (∀ x1 x2, (f x1 - f x2) / (x1 - x2) > 0) →
  f(x) = if x ≥ 1 then a^x else (4 - a / 2) * x + 2 :=
  a ∈ set.Ico 4 8 :=
sorry

end function_monotonic_range_l523_523898


namespace angle_B_in_triangle_l523_523034

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523034


namespace AB_over_BC_equals_l523_523089

-- Definitions for the problem conditions
variables (A B C D E : Type)
variables [add_comm_group A] [affine_space A E] [module ℝ A]
variables (AB BC : ℝ) (x : ℝ) (ratio : ℝ)

-- Right angles at B and C
def right_angle_at_B : Prop := ∃ (B : E), angle B A C = π / 2
def right_angle_at_C : Prop := ∃ (C : E), angle B C D = π / 2

-- Similar triangles
def triangle_sim_ABC_BCD : Prop := 
  ∃ (B C E : E), similarity (triangle B A C) (triangle B C D)
def triangle_sim_ABC_CEB : Prop := 
  ∃ (E : E), similarity (triangle B A C) (triangle C E B)

-- Area condition
def area_condition : Prop :=
  let area_CEB := area (triangle C E B) in
  area (triangle A E D) = 25 * area_CEB

-- Final proof statement
theorem AB_over_BC_equals :
  right_angle_at_B B ∧ right_angle_at_C C ∧ triangle_sim_ABC_BCD B C ∧ 
  AB > BC ∧ triangle_sim_ABC_CEB B C E ∧ area_condition B C E (2 + sqrt(3)) → 
  ratio = 2 + sqrt(3) :=
sorry

end AB_over_BC_equals_l523_523089


namespace num_divisors_180_l523_523353

-- Define a positive integer 180
def n : ℕ := 180

-- Define the function to calculate the number of divisors using prime factorization
def num_divisors (n : ℕ) : ℕ :=
  let factors := [(2, 2), (3, 2), (5, 1)] in
  factors.foldl (λ acc (p : ℕ × ℕ), acc * (p.snd + 1)) 1

-- The main theorem statement
theorem num_divisors_180 : num_divisors n = 18 :=
by
  sorry

end num_divisors_180_l523_523353


namespace find_angle_B_l523_523011

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523011


namespace smallest_integer_n_l523_523062

def a : ℝ := Real.pi / 2010

theorem smallest_integer_n : ∃ n : ℕ, 0 < n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬(1005 ∣ m * (m + 1)) ∧ 1005 ∣ n * (n + 1) :=
by
  use 66
  sorry

end smallest_integer_n_l523_523062


namespace gcd_equation_solutions_l523_523753

theorem gcd_equation_solutions:
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y^2 + Nat.gcd x y ^ 3 = x * y * Nat.gcd x y 
  → (x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3) := 
by
  intros x y h
  sorry

end gcd_equation_solutions_l523_523753


namespace inscribable_circle_in_tangential_quadrilateral_l523_523297

theorem inscribable_circle_in_tangential_quadrilateral
  {A B C D : Type} {a b c d e : ℝ} 
  (h_rectangle : IsRectangle A B C D)
  (h_diag_len : AC = e)
  (h_radii_conditions : a + c = b + d ∧ b + d < e) :
  ∃ O r, IsInscribableCircle (QuadrilateralFormedByTangents A B C D) O r :=
by
  sorry

end inscribable_circle_in_tangential_quadrilateral_l523_523297


namespace simplify_sqrt_pow_six_l523_523475

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523475


namespace y_intercepts_distance_l523_523709

theorem y_intercepts_distance (x₁ y₁ m₁ m₂ : ℝ) (h₁ : m₁ = -2) (h₂ : m₂ = 4) (hx : x₁ = 8) (hy : y₁ = 20) :
  let y_intercept_1 := y₁ - m₁ * x₁
  let y_intercept_2 := y₁ - m₂ * x₁
  abs (y_intercept_1 - y_intercept_2) = 68 :=
by
  let y_intercept_1 := y₁ - m₁ * x₁
  let y_intercept_2 := y₁ - m₂ * x₁
  have h3 : y_intercept_1 = 56 := by sorry -- Calculate y_intercept_1
  have h4 : y_intercept_2 = -12 := by sorry -- Calculate y_intercept_2
  calc 
    abs (y_intercept_1 - y_intercept_2)
    _ = abs (56 - (-12))   : by rw [h3, h4]
    _ = 68                 : by norm_num

end y_intercepts_distance_l523_523709


namespace expected_shots_l523_523946

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l523_523946


namespace count_special_integers_l523_523337

theorem count_special_integers : 
  let is_special (n : ℕ) := 
    1500 ≤ n ∧ n < 2500 ∧ (let d := n % 10 in let sum := (n / 10) % 10 + (n / 100) % 10 + n / 1000 in d = sum)
  in (Finset.filter is_special (Finset.Ico 1500 2500)).card = 81 := 
sorry

end count_special_integers_l523_523337


namespace probability_of_same_color_correct_l523_523884

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_ways_to_choose_two : ℕ := choose total_plates 2
noncomputable def ways_to_choose_two_red : ℕ := choose red_plates 2
noncomputable def ways_to_choose_two_blue : ℕ := choose blue_plates 2

noncomputable def ways_to_choose_two_same_color : ℕ :=
  ways_to_choose_two_red + ways_to_choose_two_blue

noncomputable def probability_same_color : ℚ :=
  ways_to_choose_two_same_color / total_ways_to_choose_two

theorem probability_of_same_color_correct :
  probability_same_color = 4 / 9 := by
  sorry

end probability_of_same_color_correct_l523_523884


namespace part_one_part_two_l523_523323

def f (x : ℝ) : ℝ := 2 * Real.sin (2*x + Math.pi / 6)

def h (x : ℝ) : ℝ := (1 / 2) * f x - Real.cos (2*x)

def g (x : ℝ) : ℝ := h (h x)

theorem part_one {x : ℝ} :
  f x = 2 * Real.sin (2*x + Math.pi / 6) :=
sorry

theorem part_two (a : ℝ) :
  (∃ x ∈ Icc (Math.pi / 12) (Math.pi / 3), 
    g x^2 + (2 - a) * g x + 3 - a ≤ 0) → 
  a ∈ Set.Ici (2 * Real.sqrt 2) :=
sorry

end part_one_part_two_l523_523323


namespace C1_polar_equation_area_triangle_MPQ_l523_523927

-- Definitions (conditions)
def curve_C (t : ℝ) : (ℝ × ℝ) :=
  (1 + Real.cos t, 0.5 * Real.sin t)

def curve_C1 (t : ℝ) : (ℝ × ℝ) :=
  (1 + Real.cos t, Real.sin t)

def curve_C2 (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos (θ - Real.pi / 6) = 3 * Real.sqrt 3

def point_M : (ℝ × ℝ) := (1, 0)

def line_l (θ : ℝ) : Prop := θ = Real.pi / 3

-- Theorem statements (questions with answers)
theorem C1_polar_equation : ∀ (θ : ℝ), 
  ∃ (ρ : ℝ), 
    let xy := curve_C1 θ in
    ρ = Real.sqrt (xy.1^2 + xy.2^2) ∧ 
    xy.1 = ρ * Real.cos θ ∧ 
    xy.2 = ρ * Real.sin θ ∧ 
    ρ = 2 * Real.cos θ := 
  by sorry

theorem area_triangle_MPQ : ∃ (P Q : ℝ × ℝ), 
  (P = (1, Real.pi / 3)) ∧ 
  (Q = (3, Real.pi / 3)) ∧
  let PQ_dist := Real.dist P Q in
  let M_to_l := Real.sqrt 3 / 2 in
  M_to_l = Real.sqrt 3 / 2 ∧
  ∃ (area : ℝ), 
    area = 1 / 2 * M_to_l * PQ_dist ∧
    area = Real.sqrt 3 / 2 :=
  by sorry

end C1_polar_equation_area_triangle_MPQ_l523_523927


namespace man_speed_is_correct_l523_523726

noncomputable def train_length : ℝ := 275
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def time_seconds : ℝ := 14.998800095992323

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def relative_speed_ms : ℝ := train_length / time_seconds
noncomputable def man_speed_ms : ℝ := relative_speed_ms - train_speed_ms
noncomputable def man_speed_kmh : ℝ := man_speed_ms * (3600 / 1000)
noncomputable def expected_man_speed_kmh : ℝ := 6.006

theorem man_speed_is_correct : abs (man_speed_kmh - expected_man_speed_kmh) < 0.001 :=
by
  -- proof goes here
  sorry

end man_speed_is_correct_l523_523726


namespace sum_integers_neg40_to_60_l523_523149

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end sum_integers_neg40_to_60_l523_523149


namespace construct_point_X_on_line_l523_523808

variables {α : Type*} [metric_space α]

-- Definitions for the conditions
def line (l : set α) : Prop := sorry -- Definition of line
def point (A : α) : Prop := sorry -- Definition of point
def on_one_side_of_line (A B : α) (l : set α) : Prop := sorry -- Definition of "on one side of line"

-- Given conditions
variables (l : set α) (A B : α) (a : ℝ)
  (hl : line l)
  (hA : point A)
  (hB : point B)
  (h_on_side : on_one_side_of_line A B l)
  (ha : 0 < a)

-- The statement to prove
theorem construct_point_X_on_line :
  ∃ X ∈ l, dist A X + dist B X = a :=
begin
  sorry -- Proof goes here
end

end construct_point_X_on_line_l523_523808


namespace total_points_of_other_7_team_members_l523_523201

theorem total_points_of_other_7_team_members (x y : ℕ) 
  (h1 : Alexa_points = (1/4 : ℚ) * x)
  (h2 : Brittany_points = (2/7 : ℚ) * x)
  (h3 : Chelsea_points = 15)
  (h4 : Other7_points = y)
  (h5 : Alexa_points + Brittany_points + Chelsea_points + Other7_points = x)
  (h6 : ∀ t ∈ {1, 2, 3, 4, 5, 6, 7}, OtherTeamMember t ≤ 2)
  : y = 11 :=
by
  sorry

end total_points_of_other_7_team_members_l523_523201


namespace fencing_required_l523_523157

theorem fencing_required (L W : ℕ) (hL : L = 10) (hA : L * W = 600) : L + 2 * W = 130 :=
by
  sorry

end fencing_required_l523_523157


namespace figure_100_contains_30301_triangles_l523_523906

def g (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem figure_100_contains_30301_triangles :
  g(100) = 30301 :=
by
  -- g(100) is expected to be 30301, we'll prove it
  sorry

end figure_100_contains_30301_triangles_l523_523906


namespace hardcover_volumes_l523_523777

theorem hardcover_volumes (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 25 * h + 15 * p = 240) : h = 6 :=
by
  -- omitted proof steps for brevity
  sorry

end hardcover_volumes_l523_523777


namespace selene_discount_percentage_l523_523093

theorem selene_discount_percentage :
  let cost_camera := 110
  let cost_frame := 120
  let amount_paid := 551
  let total_cost := (2 * cost_camera) + (3 * cost_frame)
  let discount_amount := total_cost - amount_paid
  let discount_percentage := (discount_amount / total_cost.toFloat) * 100
  discount_percentage = 5 :=
by
  sorry

end selene_discount_percentage_l523_523093


namespace seq_problem_l523_523862

-- Define the sequence and initial conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = -2 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n + a (n + 2)

-- Define the specific propositions to be proven
def problem (a : ℕ → ℤ) :=
  seq a ∧ a 5 = 2 ∧ (∑ i in finset.range 2016, a (i + 1)) = 0

-- State the theorem that the sequence satisfies the required properties
theorem seq_problem (a : ℕ → ℤ) (h : seq a) : problem a := sorry

end seq_problem_l523_523862


namespace mutter_paid_correct_amount_l523_523050

def total_lagaan_collected : ℝ := 344000
def mutter_land_percentage : ℝ := 0.0023255813953488372
def mutter_lagaan_paid : ℝ := 800

theorem mutter_paid_correct_amount : 
  mutter_lagaan_paid = total_lagaan_collected * mutter_land_percentage := by
  sorry

end mutter_paid_correct_amount_l523_523050


namespace impossible_coins_l523_523459

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l523_523459


namespace john_pays_in_30_day_month_l523_523957

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end john_pays_in_30_day_month_l523_523957


namespace find_angle_B_l523_523022

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523022


namespace sum_AC_l523_523430

-- Define the points and their conditions
variables (A B C D : Point)
variables (AB_AD_eq : dist A B = 7 ∧ dist A D = 7)
variables (CB_CD_eq : dist C B = 4 ∧ dist C D = 4)
variables (BD_eq : dist B D = 6)

-- The sum of all possible values of AC
theorem sum_AC (A B C D : Point)
(AB_AD_eq : dist A B = 7 ∧ dist A D = 7)
(CB_CD_eq : dist C B = 4 ∧ dist C D = 4)
(BD_eq : dist B D = 6) :
  ∃ AC_sum, AC_sum = 4 * (real.sqrt 10) :=
sorry

end sum_AC_l523_523430


namespace complement_M_l523_523331

noncomputable def U : Set ℝ := Set.univ

def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M : U \ M = { x | x < -2 ∨ x > 2 } :=
by 
  sorry

end complement_M_l523_523331


namespace abs_sum_div_diff_sqrt_7_5_l523_523887

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end abs_sum_div_diff_sqrt_7_5_l523_523887


namespace simplify_sqrt_7_pow_6_l523_523490

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523490


namespace average_score_in_5_matches_l523_523105

theorem average_score_in_5_matches 
  (avg1 avg2 : ℕ)
  (total_matches1 total_matches2 : ℕ)
  (h1 : avg1 = 27) 
  (h2 : avg2 = 32)
  (h3 : total_matches1 = 2) 
  (h4 : total_matches2 = 3) 
  : 
  (avg1 * total_matches1 + avg2 * total_matches2) / (total_matches1 + total_matches2) = 30 :=
by 
  sorry

end average_score_in_5_matches_l523_523105


namespace line_equation_l523_523789

theorem line_equation (P A B : ℝ × ℝ) (h1 : P = (-1, 3)) (h2 : A = (1, 2)) (h3 : B = (3, 1)) :
  ∃ c : ℝ, (x - 2*y + c = 0) ∧ (4*x - 2*y - 5 = 0) :=
by
  sorry

end line_equation_l523_523789


namespace circle_radius_l523_523627

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l523_523627


namespace find_sequence_term_l523_523314

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  (2 / 3) * n^2 - (1 / 3) * n

def sequence_term (n : ℕ) : ℚ :=
  if n = 1 then (1 / 3) else (4 / 3) * n - 1

theorem find_sequence_term (n : ℕ) : sequence_term n = (sequence_sum n - sequence_sum (n - 1)) :=
by
  unfold sequence_sum
  unfold sequence_term
  sorry

end find_sequence_term_l523_523314


namespace range_of_f_l523_523422

noncomputable def f (θ x : ℝ) : ℝ := (sin θ / 3) * x^3 + (sqrt 3 * cos θ / 2) * x^2 + tan θ
noncomputable def f'' (θ : ℝ) : ℝ := 2 * sin θ + sqrt 3 * cos θ

theorem range_of_f''_at_one (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 5 * π / 12) : sqrt 2 ≤ f'' θ ∧ f'' θ ≤ 2 :=
sorry

end range_of_f_l523_523422


namespace sum_seq_2014_l523_523328

def a : ℕ → ℤ
| 0     := 2008
| 1     := 2009
| n + 2 := a n + a (n + 1)

def S : ℕ → ℤ
| 0 => a 0
| 1 => a 1
| n + 2 => S n + a (n + 2)

theorem sum_seq_2014 :
  S 2013 = 2010 := by
  sorry

end sum_seq_2014_l523_523328


namespace impossible_coins_l523_523460

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l523_523460


namespace european_stamps_cost_l523_523690

def germany_stamps_60s : ℕ := 5
def germany_stamps_70s : ℕ := 10
def germany_stamps_80s : ℕ := 8

def italy_stamps_60s : ℕ := 6
def italy_stamps_70s : ℕ := 12
def italy_stamps_80s : ℕ := 9

def sweden_stamps_60s : ℕ := 7
def sweden_stamps_70s : ℕ := 8
def sweden_stamps_80s : ℕ := 15

def norway_stamps_60s : ℕ := 4
def norway_stamps_70s : ℕ := 6
def norway_stamps_80s : ℕ := 10

def price_per_stamp_germany_italy : ℝ := 0.08
def price_per_stamp_sweden : ℝ := 0.05
def price_per_stamp_norway : ℝ := 0.07

def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10

def stamps_before_80s (g60 g70 g80 i60 i70 i80 : ℕ) : ℕ :=
  g60 + g70 + i60 + i70

noncomputable def cost_of_stamps (count : ℕ) (price : ℝ) (threshold : ℕ) (discount : ℝ) : ℝ :=
  if count > threshold then (count * price) * (1 - discount) else count * price

theorem european_stamps_cost :
  let germany_count := stamps_before_80s germany_stamps_60s germany_stamps_70s germany_stamps_80s italy_stamps_60s italy_stamps_70s italy_stamps_80s in
  let italy_count := stamps_before_80s germany_stamps_60s germany_stamps_70s germany_stamps_80s italy_stamps_60s italy_stamps_70s italy_stamps_80s in
  let germany_cost := cost_of_stamps germany_count price_per_stamp_germany_italy discount_threshold discount_rate in
  let italy_cost := cost_of_stamps italy_count price_per_stamp_germany_italy discount_threshold discount_rate in
  rounded (germany_cost + italy_cost) = 2.50 :=
sorry

end european_stamps_cost_l523_523690


namespace second_company_cheaper_l523_523428

theorem second_company_cheaper (x : ℕ) :
  (250 + 15 * x < 120 + 18 * x) → (x ≥ 44) :=
by
  intro h
  have : 130 < 3 * x := by linarith
  exact Nat.le_of_lt_succ (Int.ofNat_lt.mp (by linarith))

end second_company_cheaper_l523_523428


namespace find_x_minus_y_l523_523805

open Real

theorem find_x_minus_y (x y : ℝ) (h : (sin x ^ 2 - cos x ^ 2 + cos x ^ 2 * cos y ^ 2 - sin x ^ 2 * sin y ^ 2) / sin(x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * (k : ℝ) * π + π / 2 :=
by
  sorry

end find_x_minus_y_l523_523805


namespace sequence_term_is_correct_l523_523861

theorem sequence_term_is_correct : ∀ (n : ℕ), (n = 7) → (2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) :=
by
  sorry

end sequence_term_is_correct_l523_523861


namespace simplify_sqrt_seven_pow_six_l523_523537

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523537


namespace number_of_real_roots_of_f_f_x_equal_4x_is_0_l523_523852

variable {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem number_of_real_roots_of_f_f_x_equal_4x_is_0
  (ha : a ≠ 0)
  (h_no_roots : ¬ ∃ x : ℝ, f x = 2 * x) :
  ¬ ∃ x : ℝ, f (f x) = 4 * x :=
by
  sorry

end number_of_real_roots_of_f_f_x_equal_4x_is_0_l523_523852


namespace no_straight_line_intersects_all_sides_l523_523040

def is2001Gon (P : Type) : Prop :=
  ∃ (n : ℕ), n = 2001 ∧ ∀ (a : P), P

def intersects_sides_exactly_once (P L : Type) : Prop :=
  ∀ (s : P), ∃ (p1 p2 : P), p1 ≠ p2 ∧ p1 ∉ P ∧ p2 ∉ P ∧ p1 ∈ L ∧ p2 ∈ L

theorem no_straight_line_intersects_all_sides :
  ∀ (P L : Type), is2001Gon P →
  ¬(intersects_sides_exactly_once P L) :=
begin
  intros P L h1 h2,
  -- sorry for skipping the proof.
  sorry 
end

end no_straight_line_intersects_all_sides_l523_523040


namespace james_subscribers_before_gift_l523_523043

theorem james_subscribers_before_gift
  (gifted_subscribers : ℕ)
  (monthly_income_per_subscriber : ℕ)
  (total_monthly_income_after_gift : ℕ)
  (H1 : gifted_subscribers = 50)
  (H2 : monthly_income_per_subscriber = 9)
  (H3 : total_monthly_income_after_gift = 1800) :
  ∃ (initial_subscribers : ℕ), (initial_subscribers + gifted_subscribers) * monthly_income_per_subscriber = total_monthly_income_after_gift ∧ initial_subscribers = 150 :=
by
  use 150
  split
  sorry

end james_subscribers_before_gift_l523_523043


namespace perpendicular_condition_l523_523332

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

theorem perpendicular_condition :
  (∃ (k1 k2 : ℝ), k1 = ⟪b, c⟫ ∧ k2 = ⟪a, c⟫ ∧ ⟪c, k1 • a - k2 • b⟫ = 0) :=
by
  use ⟪b, c⟫,
  use ⟪a, c⟫,
  split,
  { refl },
  split,
  { refl },
  -- The following proof step shows that the dot product equals zero
  sorry

end perpendicular_condition_l523_523332


namespace number_of_divisors_180_l523_523340

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l523_523340


namespace calculate_expression_l523_523739

theorem calculate_expression : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 :=
by
  -- Proof steps would be included here if they were needed, but the proof is left as sorry for now.
  sorry

end calculate_expression_l523_523739


namespace number_of_x_intercepts_of_sin_1_over_x_l523_523273

noncomputable def x_intercepts_sin_1_over_x_in_interval (a b : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℕ :=
  let π := Real.pi
  let k₁ := Nat.floor (b⁻¹ * π) in
  let k₂ := Nat.floor (a⁻¹ * π) in
  k₁ - k₂

theorem number_of_x_intercepts_of_sin_1_over_x :
  x_intercepts_sin_1_over_x_in_interval 0.00001 0.0001 (λ x, Real.sin (x⁻¹)) = 28647 :=
by
  sorry

end number_of_x_intercepts_of_sin_1_over_x_l523_523273


namespace simplify_sqrt_seven_pow_six_l523_523533

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523533


namespace proof_problem_l523_523412
-- Import the necessary Mathlib library

-- Define the problem conditions and question
noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := 3 + 4 * i
noncomputable def conjugate_z : ℂ := complex.conj z

-- State the proof problem
theorem proof_problem : (i^2018) * conjugate_z = -3 + 4 * i :=
  sorry

end proof_problem_l523_523412


namespace circle_properties_l523_523858

noncomputable def polar_eq : Prop :=
  ∀ (ρ θ : ℝ), ρ^2 - 4 * sqrt 2 * ρ * cos (θ - π / 4) + 6 = 0 → 
               ∃ x y : ℝ, (x = ρ * cos θ) ∧ (y = ρ * sin θ) ∧ (x - 2)^2 + (y - 2)^2 = 2

noncomputable def parametric_eq : Prop :=
  ∀ α : ℝ, 
  ∃ x y : ℝ, 
  x = 2 + sqrt 2 * cos α ∧ 
  y = 2 + sqrt 2 * sin α

noncomputable def extreme_values : Prop := 
  ∀ α : ℝ, 
  let xy := 2 + sqrt 2 * (sin α + cos α) in 
  2 ≤ xy ∧ xy ≤ 6 

theorem circle_properties : polar_eq ∧ parametric_eq ∧ extreme_values := 
by 
  split; 
  sorry

end circle_properties_l523_523858


namespace equilateral_triangle_side_length_l523_523905

theorem equilateral_triangle_side_length :
  let side_large := 4
  let area_large := (sqrt 3 / 4) * (side_large ^ 2)
  let area_small := (1 / 3) * area_large
  ∃ (s : ℝ), (sqrt 3 / 4) * (s^2) = area_small ∧ s = (4 * sqrt 3 / 3) :=
by
  let side_large := 4
  let area_large := (sqrt 3 / 4) * (side_large ^ 2)
  let area_small := (1 / 3) * area_large
  use (4 * sqrt 3 / 3)
  split
  {
    sorry
  }
  {
    sorry
  }

end equilateral_triangle_side_length_l523_523905


namespace theater_ticket_problem_l523_523231

noncomputable def total_cost_proof (x : ℝ) : Prop :=
  let cost_adult_tickets := 10 * x
  let cost_child_tickets := 8 * (x / 2)
  let cost_senior_tickets := 4 * (0.75 * x)
  cost_adult_tickets + cost_child_tickets + cost_senior_tickets = 58.65

theorem theater_ticket_problem (x : ℝ) (h : 6 * x + 5 * (x / 2) + 3 * (0.75 * x) = 42) : 
  total_cost_proof x :=
by
  sorry

end theater_ticket_problem_l523_523231


namespace term_with_minimum_coefficient_is_4th_l523_523925

noncomputable def binomial_coefficient (n k : ℕ) : ℝ :=
  (nat.choose n k : ℕ)

def expansion_term (n r : ℕ) (x : ℝ) : ℝ :=
  binomial_coefficient n r * (- (1 / 2) : ℝ) ^ r * x ^ (4 - r)

def term_with_minimum_coefficient (n : ℕ) (x : ℝ) :=
  ∃ r : ℕ, (r % 2 = 1) ∧ 0 ≤ r ∧ r ≤ n ∧ 
    ∀ (k : ℕ), (k % 2 = 1) ∧ 0 ≤ k ∧ k ≤ n → expansion_term n r x ≤ expansion_term n k x

theorem term_with_minimum_coefficient_is_4th (n : ℕ) (x : ℝ) :
  n = 8 → term_with_minimum_coefficient n x :=
begin
  intro h,
  rw h,
  use 3,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { linarith },
  { intros k hk1 hk2 hk3,
    rw [h, expansion_term, expansion_term],
    -- Further steps would follow to show minimization
    sorry,
  }
end

end term_with_minimum_coefficient_is_4th_l523_523925


namespace john_reads_bible_in_7_weeks_l523_523402

variable (weekdays_reading_time_per_day : ℝ := 1.5)
variable (weekdays_reading_rate : ℝ := 40)
variable (saturday_reading_time : ℝ := 2.5)
variable (saturday_reading_rate : ℝ := 60)
variable (bible_length : ℝ := 2800)
variable (weekdays_per_week : ℕ := 5)

noncomputable def pages_per_weekday := weekdays_reading_time_per_day * weekdays_reading_rate
noncomputable def pages_per_saturday := saturday_reading_time * saturday_reading_rate
noncomputable def total_pages_per_week := (weekdays_per_week : ℝ) * pages_per_weekday + pages_per_saturday
noncomputable def weeks_to_read_bible := (bible_length / total_pages_per_week).ceil

theorem john_reads_bible_in_7_weeks : weeks_to_read_bible = 7 := by
  sorry

end john_reads_bible_in_7_weeks_l523_523402


namespace john_pays_in_30_day_month_l523_523958

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end john_pays_in_30_day_month_l523_523958


namespace propositions_correct_count_l523_523904

-- Definitions of the propositions in Lean
def proposition_1 : Prop := 
  ∀ (length width height: ℝ), 
  ∃ (x y z: ℝ), 
  (∀ (v: (ℝ × ℝ × ℝ)), v ∈ {(0, 0, 0), (length, 0, 0), (0, width, 0), (0, 0, height),
                           (length, width, 0), (length, 0, height), (0, width, height), (length, width, height)} → 
    dist (x, y, z) v = dist (x, y, z) (0, 0, 0))

def proposition_2 : Prop := 
  ∀ (length width height: ℝ), 
  ∄ (x y z: ℝ), 
  (∀ (e: (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)), 
    (e ∈ {((0,0,0),(length,0,0)), ((0,0,0),(0,width,0)), ((0,0,0),(0,0,height)), 
           ((length,0,0),(length,width,0)), ((length,0,0),(length,0,height)), ((0,width,0),(length,width,0)), 
           ((0,width,0),(0,width,height)), ((0,0,height),(length,0,height)), ((0,0,height),(0,width,height)), 
           ((length,width,0),(length,width,height)), ((length,0,height),(length,width,height)), 
           ((0,width,height),(length,width,height))}) →
      (dist (x, y, z) (e.1) = dist (x, y, z) (0,0,0)))

def proposition_3 : Prop := 
  ∀ (length width height: ℝ), 
  ∄ (x y z: ℝ), 
  (∀ (f: ℝ × ℝ × ℝ), 
    (f ∈ {((0,0,0),(length,0,0),(0,width,0),(0,0,height)), ((length,0,0),(length,width,0),(length,0,height),(length,width,height)), 
           ((0,width,0),(length,width,0),(0,width,height),(length,width,height)), 
           ((0,0,height),(length,0,height),(0,width,height),(length,width,height)), 
           ((0,0,0),(length,0,0),(0,width,0),(length,0,height)), 
           {(length,0,0),(0,width,0),(length,width,0),(length,width,height)}}) →
      (dist (x, y, z) f = dist (x, y, z) (0,0,0)))

theorem propositions_correct_count : 
  (proposition_1 ∧ ¬proposition_2 ∧ ¬proposition_3) → 
  1 :=
by 
  sorry

end propositions_correct_count_l523_523904


namespace max_revenue_l523_523124

-- Definitions of p(t) and Q(t)
def p (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then -t + 100
  else if 25 ≤ t ∧ t ≤ 30 then t + 20
  else 0

def Q (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

-- Definition of y(t) as y = p * Q
def y (t : ℕ) : ℝ :=
  p t * Q t

-- The maximum revenue statement
theorem max_revenue : 
  ∃ t : ℕ, t ∈ (1:ℕ)..30 ∧ y t = 1125 ∧ ∀ u : ℕ, u ∈ (1:ℕ)..30 → y u ≤ 1125 :=
by sorry

end max_revenue_l523_523124


namespace num_pos_divisors_180_l523_523345

theorem num_pos_divisors_180 : 
  let n := 180 in
  let prime_factorization := [(2, 2), (3, 2), (5, 1)] in
  (prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1) = 18 :=
by 
  let n := 180
  let prime_factorization := [(2, 2), (3, 2), (5, 1)]
  have num_divisors := prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1 
  show num_divisors = 18
  sorry

end num_pos_divisors_180_l523_523345


namespace triangle_angle_B_l523_523007

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523007


namespace simplify_sqrt_7_pow_6_l523_523493

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523493


namespace sum_of_terms_l523_523983

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l523_523983


namespace triangle_angle_B_l523_523008

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523008


namespace geom_prog_min_third_term_l523_523720

theorem geom_prog_min_third_term :
  ∃ (d : ℝ), (-4 + 10 * Real.sqrt 6 = d ∨ -4 - 10 * Real.sqrt 6 = d) ∧
  (∀ x, x = 37 + 2 * d → x ≤ 29 - 20 * Real.sqrt 6) := 
sorry

end geom_prog_min_third_term_l523_523720


namespace complex_abs_sum_eq_1_or_3_l523_523990

open Complex

theorem complex_abs_sum_eq_1_or_3 (a b c : ℂ) (ha : abs a = 1) (hb : abs b = 1) (hc : abs c = 1) 
  (h : a^3/(b^2 * c) + b^3/(a^2 * c) + c^3/(a^2 * b) = 1) : abs (a + b + c) = 1 ∨ abs (a + b + c) = 3 := 
by
  sorry

end complex_abs_sum_eq_1_or_3_l523_523990


namespace proof_C_M_N_l523_523329

variables (M N : Set ℤ)

-- M is defined as {m^2, m} for some integer m
def M_def (m : ℤ) : Set ℤ := {m^2, m}

-- N is defined as {1}
def N_def : Set ℤ := {1}

-- The intersection is not empty, so M ∩ N ≠ ∅
variables (m : ℤ) (h_intersect : M_def m ∩ N_def ≠ ∅)

-- The desired result to prove
theorem proof_C_M_N : C_M (M_def m) N_def = {-1} :=
sorry

end proof_C_M_N_l523_523329


namespace simplify_sqrt_pow_six_l523_523473

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523473


namespace exists_four_cities_l523_523127

-- Define the type for cities
structure City :=
  (index : Nat)

-- Define the total number of cities
def numberOfCities := 100

-- Define the conditions
variables (cities : Finset City)
          (directPhone : City → City → Prop)
          (regularFlights : City → City → Prop)

-- State the properties based on the conditions given
axiom phone_or_flight (A B : City) : A ∈ cities ∧ B ∈ cities → (directPhone A B ↔ ¬ regularFlights A B)
axiom phone_connectivity (A B : City) : A ∈ cities ∧ B ∈ cities → (A = B) ∨ (directPhone A B ∨ (∃ intermediates : Finset City, ∀ C ∈ intermediates, directPhone A C ∨ directPhone C B))
axiom flight_connectivity (A B : City) : A ∈ cities ∧ B ∈ cities → (A = B) ∨ (regularFlights A B ∨ (∃ intermediates : Finset City, ∀ C ∈ intermediates, regularFlights A C ∨ regularFlights C B))

-- Define the theorem to prove the equivalence
theorem exists_four_cities :
  ∃ quartet : Finset City, quartet.card = 4 ∧ 
  ∀ A B ∈ quartet, (directPhone A B ∨ (∃ C D ∈ quartet \ {A, B}, (directPhone A C ∧ directPhone C B) ∨ (regularFlights A C ∧ regularFlights C B))) :=
by {
  sorry
}

end exists_four_cities_l523_523127


namespace impossible_coins_l523_523453

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l523_523453


namespace find_angle_B_l523_523001

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l523_523001


namespace janet_better_condition_count_l523_523397

noncomputable def janet_initial := 10
noncomputable def janet_sells := 6
noncomputable def janet_remaining := janet_initial - janet_sells
noncomputable def brother_gives := 2 * janet_remaining
noncomputable def janet_after_brother := janet_remaining + brother_gives
noncomputable def janet_total := 24

theorem janet_better_condition_count : 
  janet_total - janet_after_brother = 12 := by
  sorry

end janet_better_condition_count_l523_523397


namespace simplify_sqrt_7_pow_6_l523_523494

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523494


namespace kitten_current_length_l523_523214

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l523_523214


namespace log_infinite_expression_pos_l523_523766

theorem log_infinite_expression_pos :
  let x := real.logb 3 (81 + real.logb 3 (81 + real.logb 3 (81 + ...)))
  in x = 4 :=
sorry

end log_infinite_expression_pos_l523_523766


namespace midpoint_of_equal_areas_l523_523432

-- Definitions for the points and triangles
structure Triangle (α : Type) :=
  (A B C : α)

structure Point (α : Type) := 
  (x y : α)

noncomputable def area {α : Type} [Field α] (T : Triangle α) (M N P : Point α) : α := sorry

-- The main theorem we aim to prove
theorem midpoint_of_equal_areas 
  {α : Type} [Field α] (T : Triangle α) (M N P : Point α) :
  (area T (Point.mk M.x (M.y)) (Point.mk N.x (N.y)) (Point.mk P.x (P.y)) = area T.A B C * (1 / 4)) ∧
  (area T (Point.mk B.x (B.y)) (Point.mk N.x (N.y)) (Point.mk P.x (P.y)) = area T.A B C * (1 / 4)) ∧
  (area T (Point.mk A.x (A.y)) (Point.mk M.x (M.y)) (Point.mk P.x (P.y)) = area T.A B C * (1 / 4)) ∧
  (area T (Point.mk N.x (N.y)) (Point.mk M.x (M.y)) (Point.mk P.x (P.y)) = area T.A B C * (1 / 4)) → 
  (M.x = (T.A + T.B) / 2) ∧ 
  (N.x = (T.B + T.C) / 2) ∧ 
  (P.x = (T.C + T.A) / 2) := sorry

end midpoint_of_equal_areas_l523_523432


namespace jason_safe_combinations_l523_523400

theorem jason_safe_combinations : 
  let digits := [1, 2, 3, 4, 5, 6] in
  let is_odd := λ x, x = 1 ∨ x = 2 ∨ x = 3 in
  let is_even := λ x, x = 4 ∨ x = 5 ∨ x = 6 in
  let valid_combination := ∀ (d1 d2 d3 d4 d5 d6 : ℕ),
    is_odd d1 → is_even d2 → is_odd d3 → is_even d4 → is_odd d5 → is_even d6 →
    d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧ d6 ∈ digits ∨
    is_even d1 → is_odd d2 → is_even d3 → is_odd d4 → is_even d5 → is_odd d6 →
    d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧ d6 ∈ digits
  in
  3^6 + 3^6 = 1458 :=
begin
  sorry
end

end jason_safe_combinations_l523_523400


namespace solution_count_l523_523338

theorem solution_count (n : ℕ) (hn : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ a b : ℕ, 0 < a → 0 < b → (4 * a - b) * (4 * b - a) = 1770^n → f a b ∈ finset.univ.filter (λ (x : ℕ × ℕ), (x.1 > 0 ∧ x.2 > 0)) ∧
    (f a b).fst = 9 * (n - 1) ^ 2) := 
begin
  sorry
end

end solution_count_l523_523338


namespace max_min_angle_five_points_l523_523810

open Real

noncomputable def max_min_angle (n : ℕ) : ℝ :=
  if (n = 5) then (pi * ((√5 - 1) / 5)) else 0

theorem max_min_angle_five_points (n : ℕ) (h : n = 5) (points : Fin n → ℝ × ℝ) 
  (h_noncollinear : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k →
    ¬Collinear ℝ {points i, points j, points k}) : 
      ∃ α : ℝ, (α = max_min_angle n) :=
by
  rw [h]
  use (pi * ((√5 - 1) / 5))
  sorry

end max_min_angle_five_points_l523_523810


namespace number_sequence_impossible_l523_523686

theorem number_sequence_impossible :
  ¬ ∃ (s : list ℕ), (s.perm {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧ 
  (∀ i, 1 ≤ i → i < s.length → 
        ∃ k : ℕ, s.nth i = ((s.nth (i - 1)).getD 0) + ((s.nth (i - 1)).getD 0) * k / 100 ∨
                  s.nth i = ((s.nth (i - 1)).getD 0) - ((s.nth (i - 1)).getD 0) * k / 100) :=
by sorry

end number_sequence_impossible_l523_523686


namespace angle_B_eq_3pi_over_10_l523_523029

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l523_523029


namespace reflect_point_across_x_axis_l523_523384

theorem reflect_point_across_x_axis {x y : ℝ} (h : (x, y) = (2, 3)) : (x, -y) = (2, -3) :=
by
  sorry

end reflect_point_across_x_axis_l523_523384


namespace gcf_3150_7350_l523_523142

theorem gcf_3150_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end gcf_3150_7350_l523_523142


namespace radius_of_circle_l523_523638

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l523_523638


namespace sum_of_weights_of_all_expansions_eq_q_l523_523977

def is_coprime (a b : ℕ) : Prop := gcd a b = 1

noncomputable def expansion_weight {p q : ℕ} (expansion : List ℤ) : ℕ :=
expansion.tail.foldl (λ acc ai, acc * (nat_abs ai - 1)) 1

noncomputable def calc_weights_sum (p q : ℕ) : ℕ := 
-- This function computes the sum of weights of all expansions of p / q.
-- Place holder implementation to match condition purposes
0

theorem sum_of_weights_of_all_expansions_eq_q (p q : ℕ) (h_coprime: is_coprime p q) (h_p: 2 ≤ p) (h_q: 2 ≤ q) : 
calc_weights_sum p q = q :=
by
  sorry

end sum_of_weights_of_all_expansions_eq_q_l523_523977


namespace ratio_of_areas_l523_523663

open Real EuclideanGeometry

/-- Triangle ABC is a right triangle with AB = AC. -/
variables (A B C D E F G H : Point)
variables (AB AC BC : ℝ)
variables (s : ℝ)

/-- Define the midpoints D, E, F, G, H. -/
def midpoint (A B : Point) : Point := sorry

/-- The right triangle property and isosceles property (AB = AC). -/
axiom h1 : AB = AC
axiom ab : AB = 2 * s
axiom ac : AC = 2 * s
axiom is_right_triangle : ∠ABC = π / 2

/-- Midpoints of the specified segments. -/
axiom def_D : D = midpoint A B
axiom def_E : E = midpoint B C
axiom def_F : F = midpoint C A
axiom def_G : G = midpoint D F
axiom def_H : H = midpoint D E

/-- Triangle similarity and area ratio. -/
theorem ratio_of_areas : 
  let area_ABC := 1 / 2 * AB * AC,
      area_DEF := 1 / 2 * (AB / 2) * (AC / 2) in
  area_DEF / (area_ABC - area_DEF) = 1 / 3 :=
by {
  sorry
}

end ratio_of_areas_l523_523663


namespace series_sum_is_six_correct_statements_l523_523749

-- Define the first term and common ratio of our geometric series
def a : ℝ := 3
def r : ℝ := 1 / 2

-- Define the infinite geometric series
def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ := a / (1 - r)

-- Prove that the above series sum to 6
theorem series_sum_is_six (h : |r| < 1) : infinite_geometric_series_sum a r h = 6 :=
sorry

-- Prove the correct answer (4 and 5)
theorem correct_statements : 
  ¬(infinite_geometric_series_sum a r (by norm_num) > 6) ∧ -- statement (1) is false
  ¬(infinite_geometric_series_sum a r (by norm_num) < 6) ∧ -- statement (2) is false
  ¬(∀ ε > 0, ∃ n, |(infinite_geometric_series_sum a r (by norm_num) - 0)| < ε) ∧ -- statement (3) is false
  (∀ ε > 0, |infinite_geometric_series_sum a r (by norm_num) - 6| < ε) ∧ -- statement (4) is true
  ∃ L, ∀ ε > 0, |infinite_geometric_series_sum a r (by norm_num) - L| < ε -- statement (5) is true
:=
sorry

end series_sum_is_six_correct_statements_l523_523749


namespace largest_variable_l523_523888

theorem largest_variable (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e + 1) :
  c = a + 2 ∧ (∀ x ∈ ({a, b, d, e} : set ℝ), x ≤ c) :=
by
  -- Define variables based on given conditions
  have hb : b = a - 5, by sorry,
  have hc : c = a + 2, by sorry,
  have hd : d = a - 7, by sorry,
  have he : e = a - 3, by sorry,
  -- Prove largest
  sorry

end largest_variable_l523_523888


namespace area_of_shaded_region_l523_523921

noncomputable def pq : ℝ := 5
noncomputable def rs : ℝ := 5
noncomputable def center_angle : ℝ := 60
noncomputable def radius : ℝ := 5

theorem area_of_shaded_region :
  let triangle_area := 2 * (sqrt 3 / 4 * radius^2)
  let sector_area := 2 * ((center_angle / 360) * π * radius^2)
  triangle_area + sector_area = (25 * sqrt 3) / 2 + (25 * π) / 3 :=
by 
  let triangle_area := 2 * (sqrt 3 / 4 * radius^2)
  let sector_area := 2 * ((center_angle / 360) * π * radius^2)
  sorry

end area_of_shaded_region_l523_523921


namespace triangle_angle_B_l523_523009

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523009


namespace algebra_sqrt_identity_l523_523360

theorem algebra_sqrt_identity (x : ℝ) (h : real.sqrt (5 + x) + real.sqrt (25 - x) = 8) :
  (5 + x) * (25 - x) = 289 := 
by 
  sorry

end algebra_sqrt_identity_l523_523360


namespace circle_chords_radius_squared_l523_523175

-- Define the problem
variables {r : ℝ} -- Radius of the circle
variables {AB CD : ℝ} -- Lengths of the chords
variables {P A B C D : ℝ} -- Point P and endpoints of the chords
variables (h_chords_lengths : AB = 12 ∧ CD = 8)
variables (h_intersects_outside : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D)
variables (h_angle_right : angle P A D = π/2)
variables (h_BP : BP = 10)

-- The goal is to prove:
theorem circle_chords_radius_squared (h_chords_lengths : AB = 12 ∧ CD = 8)
    (h_intersects_outside : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D)
    (h_angle_right : angle P A D = π/2)
    (h_BP : BP = 10) :
    r^2 = 317 :=
by
    sorry

end circle_chords_radius_squared_l523_523175


namespace tan_alpha_eq_neg_one_l523_523829

variable (α : Real)

theorem tan_alpha_eq_neg_one 
  (h1 : sin α - cos α = Real.sqrt 2)
  (h2 : 0 < α) (h3 : α < Real.pi) :
  Real.tan α = -1 := 
sorry

end tan_alpha_eq_neg_one_l523_523829


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523602

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523602


namespace find_z2_l523_523311

def z1_satisfies (z1 : ℂ) : Prop :=
  z1 * (2 + complex.I) = 5 * complex.I

def z2_satisfies (z1 z2 : ℂ) : Prop :=
  (z1 + z2).im = 0 ∧ (z1 * z2).re = 0

theorem find_z2 (z1 z2 : ℂ) (h1 : z1_satisfies z1) (h2 : z2_satisfies z1 z2) :
  z2 = -4 - 2 * complex.I :=
sorry

end find_z2_l523_523311


namespace find_value_l523_523286

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end find_value_l523_523286


namespace more_oranges_than_apples_l523_523964

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l523_523964


namespace simplify_sqrt7_pow6_l523_523518

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523518


namespace simplify_sqrt7_pow6_l523_523501

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523501


namespace expected_shots_l523_523947

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l523_523947


namespace sum_at_simple_interest_l523_523722

theorem sum_at_simple_interest (P R : ℝ) (h1 : P * R * 3 / 100 - P * (R + 3) * 3 / 100 = -90) : P = 1000 :=
sorry

end sum_at_simple_interest_l523_523722


namespace simplify_sqrt7_pow6_l523_523507

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523507


namespace susan_weather_probability_l523_523617

theorem susan_weather_probability :
  let P_sunny := 0.4
  let P_rain := 0.6
  let days := 3
  let P_1_sunny := (3.choose 1) * (P_sunny^1) * (P_rain^(days-1))
  let P_2_sunny := (3.choose 2) * (P_sunny^2) * (P_rain^(days-2))
  P_1_sunny + P_2_sunny = 18 / 25 :=
by
  let P_sunny := 2 / 5
  let P_rain := 3 / 5
  let days := 3
  let P_1_sunny := (3.choose 1) * (P_sunny^1) * (P_rain^(days-1))
  let P_2_sunny := (3.choose 2) * (P_sunny^2) * (P_rain^(days-2))
  have h_P_1_sunny : P_1_sunny = 54 / 125 := by sorry
  have h_P_2_sunny : P_2_sunny = 36 / 125 := by sorry
  have h_sum : P_1_sunny + P_2_sunny = 90 / 125 := by sorry
  have h_simplified : 90 / 125 = 18 / 25 := by sorry
  exact h_simplified

end susan_weather_probability_l523_523617


namespace largest_fraction_l523_523678

noncomputable def compare_fractions : List ℚ :=
  [5 / 11, 7 / 16, 9 / 20, 11 / 23, 111 / 245, 145 / 320, 185 / 409, 211 / 465, 233 / 514]

theorem largest_fraction :
  max (5 / 11) (max (7 / 16) (max (9 / 20) (max (11 / 23) (max (111 / 245) (max (145 / 320) (max (185 / 409) (max (211 / 465) (233 / 514)))))))) = 11 / 23 := 
  sorry

end largest_fraction_l523_523678


namespace leif_apples_oranges_l523_523968

theorem leif_apples_oranges : 
  let apples := 14
  let dozens_of_oranges := 2 
  let oranges := dozens_of_oranges * 12
  in oranges - apples = 10 :=
by 
  let apples := 14
  let dozens_of_oranges := 2
  let oranges := dozens_of_oranges * 12
  show oranges - apples = 10
  sorry

end leif_apples_oranges_l523_523968


namespace one_over_a_lt_one_over_b_iff_a_gt_b_l523_523821

theorem one_over_a_lt_one_over_b_iff_a_gt_b (a b : ℝ) (hab : a * b > 0) :
  (1 / a < 1 / b) ↔ (a > b) :=
begin
  sorry
end

end one_over_a_lt_one_over_b_iff_a_gt_b_l523_523821


namespace least_three_digit_product_18_l523_523672

theorem least_three_digit_product_18 : ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ N = 100 * H + 10 * T + U ∧ H * T * U = 18) ∧ ∀ M : ℕ, (100 ≤ M ∧ M ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ M = 100 * H + 10 * T + U ∧ H * T * U = 18)) → N ≤ M :=
    sorry

end least_three_digit_product_18_l523_523672


namespace sum_of_a_is_9_l523_523818

theorem sum_of_a_is_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 5 / 7 = (a2 / 2.factorial) + (a3 / 3.factorial) + (a4 / 4.factorial) + (a5 / 5.factorial) + (a6 / 6.factorial) + (a7 / 7.factorial))
  (h2 : 0 ≤ a2 ∧ a2 < 2)
  (h3 : 0 ≤ a3 ∧ a3 < 3)
  (h4 : 0 ≤ a4 ∧ a4 < 4)
  (h5 : 0 ≤ a5 ∧ a5 < 5)
  (h6 : 0 ≤ a6 ∧ a6 < 6)
  (h7 : 0 ≤ a7 ∧ a7 < 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 :=
sorry

end sum_of_a_is_9_l523_523818


namespace angle_B_in_triangle_l523_523032

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523032


namespace evaluate_expression_l523_523255

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 :=
by
  sorry

end evaluate_expression_l523_523255


namespace students_in_class_l523_523176

theorem students_in_class (b n : ℕ) :
  6 * (b + 1) = n ∧ 9 * (b - 1) = n → n = 36 :=
by
  sorry

end students_in_class_l523_523176


namespace xy_product_l523_523551

theorem xy_product (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := 
sorry

end xy_product_l523_523551


namespace f_even_l523_523803

-- Define E_x^n as specified
def E_x (n : ℕ) (x : ℝ) : ℝ := List.prod (List.map (λ i => x + i) (List.range n))

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * E_x 5 (x - 2)

-- Define the statement to prove f(x) is even
theorem f_even (x : ℝ) : f x = f (-x) := by
  sorry

end f_even_l523_523803


namespace minimum_people_surveyed_l523_523179

-- Define the conditions
variables (X : ℝ) (N : ℕ)
def liked_product_A := X / 100 * N
def liked_product_B := (X - 20) / 100 * N
def liked_both := 23 / 100 * N
def liked_neither := 23 / 100 * N

-- State the theorem to prove the minimum number of people surveyed is 2300
theorem minimum_people_surveyed :
  ∃ N : ℕ, liked_product_A N = (X / 100) * N ∧
         liked_product_B N = ((X - 20) / 100) * N ∧
         liked_both N = (23 / 100) * N ∧
         liked_neither N = (23 / 100) * N ∧
         2 * X - 20 = 100 ∧ 
         N = 2300 :=
begin
  sorry
end

end minimum_people_surveyed_l523_523179


namespace leif_apples_oranges_l523_523966

theorem leif_apples_oranges : 
  let apples := 14
  let dozens_of_oranges := 2 
  let oranges := dozens_of_oranges * 12
  in oranges - apples = 10 :=
by 
  let apples := 14
  let dozens_of_oranges := 2
  let oranges := dozens_of_oranges * 12
  show oranges - apples = 10
  sorry

end leif_apples_oranges_l523_523966


namespace max_true_statements_l523_523058

theorem max_true_statements :
  ∀ (x : ℝ), 
    let S1 := (0 < x^2 ∧ x^2 < 4) in
    let S2 := (x^2 > 4) in
    let S3 := (-2 < x ∧ x < 0) in
    let S4 := (0 < x ∧ x < 2) in
    let S5 := (0 < x - (x^2 / 4) ∧ x - (x^2 / 4) < 1) in
    (S1 → ¬S2) ∧ 
    (S2 → ¬S1) ∧ 
    (S3 → ¬S4) ∧ 
    (S4 → ¬S3) ∧ 
    ((0 < x ∧ x < 2) → S5) →
    ((S1 ∧ S4 ∧ S5) ∨ (S2 ∧ S3 ∧ S5)) := sorry

end max_true_statements_l523_523058


namespace washing_machine_cost_l523_523870

variable (W D : ℝ)
variable (h1 : D = W - 30)
variable (h2 : 0.90 * (W + D) = 153)

theorem washing_machine_cost :
  W = 100 := by
  sorry

end washing_machine_cost_l523_523870


namespace simplify_and_evaluate_l523_523097

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  ( (1 + x) / (1 - x) / (x - (2 * x / (1 - x))) = - (Real.sqrt 2 + 2) / 2) :=
by
  rw [h]
  simp
  sorry

end simplify_and_evaluate_l523_523097


namespace tan_theta_expr_l523_523409

variables {θ x : ℝ}

-- Let θ be an acute angle and let sin(θ/2) = sqrt((x - 2) / (3x)).
theorem tan_theta_expr (h₀ : 0 < θ) (h₁ : θ < (Real.pi / 2)) (h₂ : Real.sin (θ / 2) = Real.sqrt ((x - 2) / (3 * x))) :
  Real.tan θ = (3 * Real.sqrt (7 * x^2 - 8 * x - 16)) / (x + 4) :=
sorry

end tan_theta_expr_l523_523409


namespace irrational_count_is_three_l523_523090

-- Let's define the real numbers given in the problem
def num1 := 2 * Real.pi
def num2 := Real.sqrt 5
def num3 := (4 : ℝ)
def num4 := 4.2121212121 -- repeating 21
def num5 := Real.cbrt 64
def num6 := (8.181181118 : ℝ)  -- with 1 inserted between every two 8's
def num7 := 11 / 7

-- Definition to check if a number is irrational
def is_irrational (x : ℝ) : Prop := ¬ ∃ q : ℚ, (q : ℝ) = x

-- Theorem to prove the number of irrational numbers
theorem irrational_count_is_three :
  let irrationals := [num1, num2, num6]
  ∃ (S : Finset ℝ), S = {num1, num2, num6} ∧ irrationals.card = 3 := by
  sorry

end irrational_count_is_three_l523_523090


namespace berries_to_sell_l523_523039

-- Definitions for the given conditions
def blueberries : ℕ := 30
def cranberries : ℕ := 20
def raspberries : ℕ := 10
def total_berries : ℕ := blueberries + cranberries + raspberries
def rotten_fraction : ℚ := 1 / 3
def keep_fraction : ℚ := 1 / 2

-- Statement of the proof problem
theorem berries_to_sell : 
  let rotten_berries := rotten_fraction * total_berries in
  let fresh_berries := total_berries - rotten_berries in
  let berries_to_keep := keep_fraction * fresh_berries in
  fresh_berries - berries_to_keep = 20 := by
  sorry

end berries_to_sell_l523_523039


namespace simplify_sqrt_seven_pow_six_proof_l523_523513

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523513


namespace exists_difference_at_least_1min51sec_not_always_exactly_2min_l523_523664

-- Define the conditions based on the problem
constants (Player : Type) (Clock : Player → ℕ → ℕ)

-- Assume the clocks show the same time after 40 moves (2 hours and 30 minutes, i.e., 150 minutes)
axiom initial_same_time : Clock p1 0 = Clock p2 0
axiom final_same_time : Clock p1 40 = Clock p2 40
axiom move_change : ∀ (t : ℕ) (n : ℕ), Clock p1 (n + t) ≠ Clock p2 (n + t)
axiom switch_clocks : ∀ (p1 p2 : Player), ∀ t, Clock p1 t = Clock p2 (t + 1)

-- Part (a)
theorem exists_difference_at_least_1min51sec :
  ∃ t : ℕ, Clock p1 t - Clock p2 t ≥ 111 / 60 := sorry

-- Part (b)
theorem not_always_exactly_2min :
  ¬ ∀ t : ℕ, Clock p1 t - Clock p2 t = 2 := sorry

end exists_difference_at_least_1min51sec_not_always_exactly_2min_l523_523664


namespace find_missing_angle_l523_523194

theorem find_missing_angle (n : ℕ) (angles : Fin n → ℕ) (H_sum : ∑ i, angles i = 3420)
  (H_all_but_one_150 : ∃ j, (∀ i, i ≠ j → angles i = 150)) :
  ∃ j, angles j = 420 := by
  sorry

end find_missing_angle_l523_523194


namespace basketball_weight_l523_523076

-- Definitions based on the given conditions
variables (b c : ℕ) -- weights of basketball and bicycle in pounds

-- Condition 1: Nine basketballs weigh the same as six bicycles
axiom condition1 : 9 * b = 6 * c

-- Condition 2: Four bicycles weigh a total of 120 pounds
axiom condition2 : 4 * c = 120

-- The proof statement we need to prove
theorem basketball_weight : b = 20 :=
by
  sorry

end basketball_weight_l523_523076


namespace carmen_counting_cars_l523_523743

theorem carmen_counting_cars 
  (num_trucks : ℕ)
  (num_cars : ℕ)
  (red_trucks : ℕ)
  (black_trucks : ℕ)
  (white_trucks : ℕ)
  (total_vehicles : ℕ)
  (percent_white_trucks : ℚ) :
  num_trucks = 50 →
  num_cars = 40 →
  red_trucks = num_trucks / 2 →
  black_trucks = (20 * num_trucks) / 100 →
  white_trucks = num_trucks - red_trucks - black_trucks →
  total_vehicles = num_trucks + num_cars →
  percent_white_trucks = (white_trucks : ℚ) / total_vehicles * 100 →
  percent_white_trucks ≈ 17 :=
sorry

end carmen_counting_cars_l523_523743


namespace smallest_positive_integer_l523_523145

theorem smallest_positive_integer (m n : ℤ) : ∃ k : ℕ, k > 0 ∧ (∃ m n : ℤ, k = 5013 * m + 111111 * n) ∧ k = 3 :=
by {
  sorry 
}

end smallest_positive_integer_l523_523145


namespace radius_of_circle_l523_523636

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l523_523636


namespace problem_equivalence_l523_523833

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n - 1)
noncomputable def S_n (n : ℕ) : ℕ := n * (3 + 3 * (n - 1)) / 2
noncomputable def T_n (n : ℕ) : ℕ := 3 * (n - 1) * 2^(n - 1)

theorem problem_equivalence (n : ℕ) (n ≠ 0) :
  (a_n 2 * b_n 2 = 12) ∧
  (S_n 3 + b_n 2 = 20) →
  (T_n n = 3 * (n - 1) * 2^(n - 1))
:=
begin
  sorry
end

end problem_equivalence_l523_523833


namespace simplify_sqrt_seven_pow_six_l523_523529

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523529


namespace B_A_equals_expectedBA_l523_523065

noncomputable def MatrixA : Matrix (Fin 2) (Fin 2) ℝ := sorry
noncomputable def MatrixB : Matrix (Fin 2) (Fin 2) ℝ := sorry
def MatrixAB : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 1], ![-2, 4]]
def expectedBA : Matrix (Fin 2) (Fin 2) ℝ := ![![10, 2], ![-4, 8]]

theorem B_A_equals_expectedBA (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = MatrixAB) : 
  B * A = expectedBA := by
  sorry

end B_A_equals_expectedBA_l523_523065


namespace impossible_coins_l523_523463

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l523_523463


namespace simplify_sqrt_power_l523_523482

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523482


namespace find_angle_B_l523_523017

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523017


namespace trajectory_equation_l523_523615

theorem trajectory_equation : ∀ (x y : ℝ),
  (x + 3)^2 + y^2 + (x - 3)^2 + y^2 = 38 → x^2 + y^2 = 10 :=
by
  intros x y h
  sorry

end trajectory_equation_l523_523615


namespace coins_with_specific_probabilities_impossible_l523_523446

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l523_523446


namespace sum_of_legs_of_larger_triangle_l523_523667

-- Define the conditions as hypotheses
def similar_right_triangles (α β : Type) :=
  ∀ (a1 a2 h1 : ℝ), a1 * a2 / 2 = 6 ∧ h1 = 5 → 
  ∀ (A B h2 : ℝ), A * B / 2 = 150 ∧ h2 / h1 = (A * A + B * B) / (a1 * a1 + a2 * a2)

-- Prove the sum of the lengths of the legs of the larger triangle
theorem sum_of_legs_of_larger_triangle :
  ∀ (a1 a2 h1 A B h2: ℝ),
    (h1 = 5) →
    (a1 * a2 / 2 = 6) →
    (A * B / 2 = 150) →
    (h2 = h1 * sqrt(150 / 6)) →
    similar_right_triangles (a1, a2, h1) (A, B, h2) →
    A + B = 35 :=
by
  sorry

end sum_of_legs_of_larger_triangle_l523_523667


namespace cartesian_equation_correct_finding_alpha_l523_523806

noncomputable def polar_to_cartesian_eq (ρ θ : ℝ) (h : ρ * sin θ^2 = 4 * cos θ) : Prop :=
  let x := ρ * cos θ
      y := ρ * sin θ in
  y^2 = 4 * x

noncomputable def intersection_condition (α : ℝ) (hα : 0 < α ∧ α < π)
  (h : ∀ t : ℝ, ((1 + t * cos α) * (t * sin α)^2 = 4 * (t * sin α))) : Prop :=
  let Lx := 1 + t * cos α 
      Ly := t * sin α in
  ∃ k : ℝ, k = 1 ∨ k = -1 ∧ (α = π / 4 ∨ α = 3 * π / 4)

theorem cartesian_equation_correct :
  ∀ (ρ θ : ℝ), ρ * sin θ ^ 2 = 4 * cos θ → polar_to_cartesian_eq ρ θ sorry :=
sorry

theorem finding_alpha :
  ∀ (α : ℝ) (hα : 0 < α ∧ α < π)
  (h : ∀ t : ℝ, (1 + t * cos α) * (t * sin α)^2 = 4 * (t * sin α)), intersection_condition α hα h :=
sorry

end cartesian_equation_correct_finding_alpha_l523_523806


namespace largest_five_digit_negative_int_congruent_mod_23_l523_523671

theorem largest_five_digit_negative_int_congruent_mod_23 :
  ∃ n : ℤ, 23 * n + 1 < -9999 ∧ 23 * n + 1 = -9994 := 
sorry

end largest_five_digit_negative_int_congruent_mod_23_l523_523671


namespace solve_equation_l523_523547

theorem solve_equation : ∀ x : ℚ, 3 * x * (x - 2) = 2 * (x - 2) → (x = 2 ∨ x = 2 / 3) :=
by
  intro x
  intro h
  have h1 : (x - 2) * (3 * x - 2) = 0 := by sorry -- From step 3 of solution
  cases eq_zero_or_eq_zero_of_mul_eq_zero h₁ with hx_minus_2 h3x_minus_2 -- From step 4 of solution
  { left
    exact eq_of_sub_eq_zero hx_minus-2 }
  { right
    exact eq_of_sub_eq_zero h3x_minus_2 }

end solve_equation_l523_523547


namespace impossible_coins_l523_523449

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l523_523449


namespace simplify_sqrt_7_pow_6_l523_523495

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523495


namespace prob_odd_divisor_18_fact_l523_523621

theorem prob_odd_divisor_18_fact : 
  let n := nat.factorial 18,
      total_divisors := (16 + 1) * (8 + 1) * (3 + 1) * (2 + 1) * 2 * 2 * 2,
      odd_divisors := (8 + 1) * (3 + 1) * (2 + 1) * 2 * 2 * 2 in
  total_divisors ≠ 0 → 
  (odd_divisors : ℚ) / (total_divisors : ℚ) = 1 / 17 := 
by admit

end prob_odd_divisor_18_fact_l523_523621


namespace kitten_length_after_4_months_l523_523218

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l523_523218


namespace ivan_expected_shots_l523_523944

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l523_523944


namespace count_multiples_9_ends_with_4_l523_523355

-- Define a function to check if a number ends with a particular digit
def ends_with (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d

-- Define a function to count the valid multiples
def count_valid_multiples (upper_limit : ℕ) (end_digit : ℕ) : ℕ :=
  (1 to upper_limit).count (λ k, (k % 9 = 0) ∧ ends_with k end_digit)

theorem count_multiples_9_ends_with_4 :
  count_valid_multiples 999 4 = 11 := sorry

end count_multiples_9_ends_with_4_l523_523355


namespace nested_logarithm_l523_523772

noncomputable def logarithm_series : ℝ := 
  if h : ∃ x : ℝ, 3^x = x + 81 then classical.some h else 0

theorem nested_logarithm (h : ∃ x : ℝ, 3^x = x + 81) :
  logarithm_series = classical.some h ∧ 
  abs (logarithm_series - 4.5) < 1 :=
by
  sorry

end nested_logarithm_l523_523772


namespace ivan_expected_shots_l523_523943

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l523_523943


namespace time_for_one_essay_l523_523045

-- We need to define the times for questions and paragraphs first.

def time_per_short_answer_question := 3 -- in minutes
def time_per_paragraph := 15 -- in minutes
def total_homework_time := 4 -- in hours
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15

-- Now we need to state the total homework time and define the goal
def computed_homework_time :=
  (time_per_short_answer_question * num_short_answer_questions +
   time_per_paragraph * num_paragraphs) / 60 + num_essays * sorry -- time for one essay in hours

theorem time_for_one_essay :
  (total_homework_time = computed_homework_time) → sorry = 1 :=
by
  sorry

end time_for_one_essay_l523_523045


namespace part1_proof_part2_proof_part3_proof_l523_523387

-- Part (1)
def part1_condition : Prop :=
  ∃ P Q : ℝ × ℝ, P = (-3, 5) ∧ Q = (12, -4) ∧ (Q = (P.1 + 3 * P.2, 3 * P.1 + P.2))

-- Part (2)
def part2_condition : Prop :=
  ∃ c : ℝ, c = -6/5 ∧ (P_1 : c × ℝ × ℝ) ∧ (P_1 == (c - 5))
  ∃ P_1 P_2 : ℝ × ℝ, 
    (P_1 = (c, 2*c + 2)) ∧ 
    ((P_2 = (-5*c - 6, -c + 2)) ∧ (P_2.fst = 0 ∨ P_2.snd = 0))

-- Part (3)
def part3_condition : Prop :=
  ∃ a x : ℝ, x > 0 ∧ (a = 2 ∨ a = -2) ∧ 
    ∃ P P_3 : ℝ × ℝ, P = (x, 0) ∧ P_3 = (x, a*x) ∧ 
      dist P P_3 = 2 * dist (0, 0) P

theorem part1_proof : part1_condition := 
by sorry

theorem part2_proof : part2_condition :=
by sorry

theorem part3_proof : part3_condition :=
by sorry

end part1_proof_part2_proof_part3_proof_l523_523387


namespace determinant_of_binomial_matrix_l523_523164

open Matrix -- To use matrix-related functions and definitions.

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the matrix D using binomial coefficients.
noncomputable def matrix_D (n : ℕ) : Matrix (Fin (n+1)) (Fin (n+1)) ℕ :=
  λ i j, binom (i + j : ℕ) j

-- The theorem statement that we need to prove.
theorem determinant_of_binomial_matrix (n : ℕ) (hn : 0 < n) :
  det (matrix_D n) = 1 := sorry

end determinant_of_binomial_matrix_l523_523164


namespace intersect_circumcircles_on_AB_l523_523999

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def right_triangle (A B C : Point) (angle_ACB : Real) : Prop := sorry
noncomputable def perpendicular (C : Point) (AB : Line) : Prop := sorry
noncomputable def circumcircle (P1 P2 P3 : Point) : Circle := sorry
noncomputable def on_circle (P : Point) (C : Circle) : Prop := sorry
noncomputable def line_intersection (l1 l2 : Line) : Point := sorry
noncomputable def angle (A B C : Point) : Real := sorry

variables {A B C M G P Q H : Point}
variables {MC AG BG : Line}
variables {circumcircle_AQG circumcircle_BPG : Circle}

theorem intersect_circumcircles_on_AB
  (h_right_triangle : right_triangle A B C 90)
  (h_midpoint : M = midpoint A B)
  (h_point_G_MC : G ∈ MC)
  (h_point_P_AG : P ∈ AG ∧ angle C P A = angle B A C)
  (h_point_Q_BG : Q ∈ BG ∧ angle B Q C = angle C B A)
  (h_perpendicular : perpendicular C (Line.mk A B))
  (h_circumcircle_AQG : circumcircle_AQG = circumcircle A Q G)
  (h_circumcircle_BPG : circumcircle_BPG = circumcircle B P G)
  : ∃ H : Point, H ∈ Line.mk A B ∧ on_circle H circumcircle_AQG ∧ on_circle H circumcircle_BPG :=
sorry

end intersect_circumcircles_on_AB_l523_523999


namespace limit_of_f_l523_523291

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem limit_of_f (h : ∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → abs (1 / (2 + x) - 1/2) / abs x < ε) :
  (real.lim (λ Δx, (f (2 + Δx) - f 2) / Δx) Δx → 0) = -1 / 4 :=
by
  sorry

end limit_of_f_l523_523291


namespace sphere_cooling_time_l523_523711

theorem sphere_cooling_time :
  ∃ t : ℝ, t = 15 ∧
  (∀ (k : ℝ) (T : ℝ → ℝ),
    (∀ t, T t = 12 * real.exp (-k * t)) ∧
    (T 0 = 12) ∧ 
    (T 8 = 9) ∧
    (T t = 7) ∧
    ∃ k > 0, (9 = 12 * real.exp (-k * 8)) ∧ (T t = 7)) :=
begin
  sorry
end

end sphere_cooling_time_l523_523711


namespace simplify_sqrt7_pow6_l523_523521

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523521


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523601

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523601


namespace graph_shift_to_sin2x_l523_523661

theorem graph_shift_to_sin2x :
  (∀ x : ℝ, (sin (2 * (x + π / 8) - π / 4)) = (sin (2 * x))) :=
by
  intro x
  sorry

end graph_shift_to_sin2x_l523_523661


namespace coins_with_specific_probabilities_impossible_l523_523442

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l523_523442


namespace trapezoid_area_correct_l523_523816

-- Given sides of the trapezoid
def sides : List ℚ := [4, 6, 8, 10]

-- Definition of the function to calculate the sum of all possible areas.
noncomputable def sumOfAllPossibleAreas (sides : List ℚ) : ℚ :=
  -- Assuming configurations and calculations are correct by problem statement
  let r4 := 21
  let r5 := 7
  let r6 := 0
  let n4 := 3
  let n5 := 15
  r4 + r5 + r6 + n4 + n5

-- Check that the given sides lead to sum of areas equal to 46
theorem trapezoid_area_correct : sumOfAllPossibleAreas sides = 46 := by
  sorry

end trapezoid_area_correct_l523_523816


namespace sum_integers_neg40_to_60_l523_523147

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end sum_integers_neg40_to_60_l523_523147


namespace sale_in_second_month_l523_523184

theorem sale_in_second_month (s2 : ℕ) : 
  let s1 := 800 in
  let s3 := 1000 in
  let s4 := 700 in
  let s5 := 800 in
  let s6 := 900 in
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 850 → s2 = 900 :=
begin
  intros,
  sorry
end

end sale_in_second_month_l523_523184


namespace intersection_is_empty_l523_523824

def A : Set ℝ := { α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3 }
def B : Set ℝ := { β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2 }

theorem intersection_is_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_is_empty_l523_523824


namespace radius_of_sphere_touching_pyramid_base_and_lateral_edge_l523_523813

-- defining the given parameters
variables (a b : ℝ) (h : b > a)

-- Theorem statement
theorem radius_of_sphere_touching_pyramid_base_and_lateral_edge
  (R : ℝ)
  (hr : R = (a * (2 * b - a) * real.sqrt 3) / (2 * real.sqrt (3 * b^2 - a^2))) :
  ∃ r, r = R := 
begin
  use R,
  exact hr,
end

#check radius_of_sphere_touching_pyramid_base_and_lateral_edge

end radius_of_sphere_touching_pyramid_base_and_lateral_edge_l523_523813


namespace compression_force_l523_523780

def T : ℕ := 3
def H : ℕ := 9

def L (T H : ℕ) : ℕ := 30 * T ^ 5 / H ^ 3

theorem compression_force :
  L T H = 10 :=
by
  unfold L
  -- L = 30 * 3 ^ 5 / 9 ^ 3
  -- We ensure this is simplified via calculation
  have h1 : 3 ^ 5 = 243 := by norm_num
  have h2 : 9 ^ 3 = 729 := by norm_num
  rw [h1, h2]
  norm_num
  sorry -- this step will be replaced by a detailed proof showing the simplification

end compression_force_l523_523780


namespace inequality_proof_equality_condition_l523_523066

variable {α : ℝ}
variables (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)

theorem inequality_proof (hα : α ∈ ℝ) :
  abc * (a^α + b^α + c^α) ≥ 
  a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) :=
sorry

theorem equality_condition (hα : α ∈ ℝ) :
  (abc * (a^α + b^α + c^α) = 
  a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c))
  ↔ (a = b ∧ b = c ∧ c = a) :=
sorry

end inequality_proof_equality_condition_l523_523066


namespace samantha_birth_year_l523_523112

theorem samantha_birth_year (first_kangaroo_year birth_year kangaroo_freq : ℕ)
  (h_first_kangaroo: first_kangaroo_year = 1991)
  (h_kangaroo_freq: kangaroo_freq = 1)
  (h_samantha_age: ∃ y, y = (first_kangaroo_year + 9 * kangaroo_freq) ∧ 2000 - 14 = y) :
  birth_year = 1986 :=
by sorry

end samantha_birth_year_l523_523112


namespace integer_solutions_for_xyz_eq_4_l523_523623

theorem integer_solutions_for_xyz_eq_4 :
  {n : ℕ // n = 48} :=
sorry

end integer_solutions_for_xyz_eq_4_l523_523623


namespace eden_initial_bears_l523_523751

theorem eden_initial_bears (d_total : ℕ) (d_favorite : ℕ) (sisters : ℕ) (eden_after : ℕ) (each_share : ℕ)
  (h1 : d_total = 20)
  (h2 : d_favorite = 8)
  (h3 : sisters = 3)
  (h4 : eden_after = 14)
  (h5 : each_share = (d_total - d_favorite) / sisters)
  : (eden_after - each_share) = 10 :=
by
  sorry

end eden_initial_bears_l523_523751


namespace point_on_circle_l523_523822

theorem point_on_circle (s : ℝ) : 
  let x := (2 - 3 * s^2) / (2 + 3 * s^2)
  let y := (6 * s) / (2 + 3 * s^2)
  in x^2 + y^2 = 1 :=
by
  let x := (2 - 3 * s^2) / (2 + 3 * s^2)
  let y := (6 * s) / (2 + 3 * s^2)
  sorry

end point_on_circle_l523_523822


namespace irrational_sqrt_2023_l523_523154

theorem irrational_sqrt_2023 (A B C D : ℝ) :
  A = -2023 → B = Real.sqrt 2023 → C = 0 → D = 1 / 2023 →
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ B = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ A = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ C = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ D = p / q) := 
by
  intro hA hB hC hD
  sorry

end irrational_sqrt_2023_l523_523154


namespace transformable_arrays_count_l523_523102

-- Definitions based on conditions:
def array_2013x2013 := array (fin 2013) (array (fin 2013) (fin 2013.succ))

-- Main theorem statement:
theorem transformable_arrays_count :
  let num_possible_arrays (n : ℕ) := n ^ (2 * n - 1)
  in num_possible_arrays 2013 = 2013 ^ 4025 :=
by
  sorry

end transformable_arrays_count_l523_523102


namespace cube_construction_possible_l523_523172

theorem cube_construction_possible (n : ℕ) : (∃ k : ℕ, n = 12 * k) ↔ ∃ V : ℕ, (n ^ 3) = 12 * V := by
sorry

end cube_construction_possible_l523_523172


namespace total_distance_traveled_l523_523868

/-- Defining the distance Greg travels in each leg of his trip -/
def distance_workplace_to_market : ℕ := 30

def distance_market_to_friend : ℕ := distance_workplace_to_market + 10

def distance_friend_to_aunt : ℕ := 5

def distance_aunt_to_grocery : ℕ := 7

def distance_grocery_to_home : ℕ := 18

/-- The total distance Greg traveled during his entire trip is the sum of all individual distances -/
theorem total_distance_traveled :
  distance_workplace_to_market + distance_market_to_friend + distance_friend_to_aunt + distance_aunt_to_grocery + distance_grocery_to_home = 100 :=
by
  sorry

end total_distance_traveled_l523_523868


namespace find_angle_B_l523_523021

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523021


namespace impossible_coins_l523_523447

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l523_523447


namespace triangle_sides_l523_523646

theorem triangle_sides (A B C : Point) (d : ℝ) (sin_C : ℝ) (AC : ℝ) :
  sin_C = √3 / 2 ∧ AC = 5 ∧ d = 12 →
  ∃ AB BC, AB = √229 ∧ (BC = 12 ∨ BC = (5 + sqrt 501) / 2) ∧
  height_from_b_to_ac B A C = d := 
begin
  intros h,
  sorry
end

end triangle_sides_l523_523646


namespace circle_radius_l523_523630

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l523_523630


namespace simplify_sqrt_power_l523_523485

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523485


namespace conjugate_in_third_quadrant_l523_523413

open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i + i^2

-- Define the conjugate of z
def conj_z := conj z

-- Define the coordinates of the conjugate in the complex plane
def conj_z_coords : ℝ × ℝ := (conj_z.re, conj_z.im)

-- Define the quadrant function
def in_quadrant (coords : ℝ × ℝ) : ℕ :=
  if h₁ : coords.1 > 0 ∧ coords.2 > 0 then 1
  else if h₂ : coords.1 < 0 ∧ coords.2 > 0 then 2
  else if h₃ : coords.1 < 0 ∧ coords.2 < 0 then 3
  else if h₄ : coords.1 > 0 ∧ coords.2 < 0 then 4
  else 0

theorem conjugate_in_third_quadrant : in_quadrant conj_z_coords = 3 := 
by 
  -- Proof will go here
  sorry

end conjugate_in_third_quadrant_l523_523413


namespace find_angle_B_l523_523018

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523018


namespace impossible_coins_l523_523452

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l523_523452


namespace discount_percentage_l523_523042

theorem discount_percentage (number_of_tshirts : ℕ) (cost_per_tshirt amount_paid : ℝ)
  (h1 : number_of_tshirts = 6)
  (h2 : cost_per_tshirt = 20)
  (h3 : amount_paid = 60) : 
  ((number_of_tshirts * cost_per_tshirt - amount_paid) / (number_of_tshirts * cost_per_tshirt) * 100) = 50 := by
  -- The proof will go here
  sorry

end discount_percentage_l523_523042


namespace distance_from_M₀_to_plane_is_correct_l523_523266

-- Define the points
def M₁ : ℝ × ℝ × ℝ := (1, -1, 1)
def M₂ : ℝ × ℝ × ℝ := (-2, 0, 3)
def M₃ : ℝ × ℝ × ℝ := (2, 1, -1)
def M₀ : ℝ × ℝ × ℝ := (-2, 4, 2)

-- Define the function that calculates distance from a point to a plane
noncomputable def distance_from_point_to_plane (p M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=  -- p here represents the point M₀
  let n := cross_product ((M₂.1 - M₁.1, M₂.2 - M₁.2, M₂.3 - M₁.3), (M₃.1 - M₁.1, M₃.2 - M₁.2, M₃.3 - M₁.3))
  let d := -(n.1 * M₁.1 + n.2 * M₁.2 + n.3 * M₁.3)
  abs (n.1 * p.1 + n.2 * p.2 + n.3 * p.3 + d) / sqrt (n.1^2 + n.2^2 + n.3^2)

theorem distance_from_M₀_to_plane_is_correct :
  distance_from_point_to_plane M₀ M₁ M₂ M₃ = 9 / sqrt 101 :=
by
  sorry

end distance_from_M₀_to_plane_is_correct_l523_523266


namespace simplify_sqrt_seven_pow_six_proof_l523_523509

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523509


namespace slower_speed_l523_523192

theorem slower_speed (x : ℝ) :
  (50 / x = 70 / 14) → x = 10 := by
  sorry

end slower_speed_l523_523192


namespace right_triangle_expression_value_l523_523814

theorem right_triangle_expression_value
  (AB BC AC BM: ℝ)
  (h : AB^2 + BC^2 = AC^2)
  (h1 : ∃ d, d = BM / Real.sqrt 2 ∧ 1 / d = 1 / AB + 1 / BC ∧ BM = d * Real.sqrt 2):
  let E := Real.sqrt 1830 * (AC - Real.sqrt (AB^2 + BC^2)) +
           1789 - ((1 / AB + 1 / BC - Real.sqrt 2 / BM) / (1848)^3) in
  E = 1789 :=
by
  sorry

end right_triangle_expression_value_l523_523814


namespace prism_surface_area_l523_523692

theorem prism_surface_area (a : ℝ) : 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  surface_area_cubes - surface_area_shared_faces = 14 * a^2 := 
by 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  have : surface_area_cubes - surface_area_shared_faces = 14 * a^2 := sorry
  exact this

end prism_surface_area_l523_523692


namespace find_b_l523_523933

variable (a b c A B C : ℝ)
variable (h_seq1 : 2 * b = a + c)
variable (h_seq2 : 2 * B = A + C)
variable (h_sum_angles : A + B + C = Real.pi)
variable (h_area : (1/2) * a * c * Real.sin B = sqrt 3 / 2)

theorem find_b (h_seq1 : 2 * b = a + c) (h_seq2 : 2 * B = A + C) 
  (h_sum_angles : A + B + C = Real.pi) (h_area : (1/2) * a * c * Real.sin B = sqrt 3 / 2) :
  b = (3 + sqrt 3) / 3 :=
by
  sorry

end find_b_l523_523933


namespace distribution_number_l523_523775

-- Define the conditions
variables (students : Fin 5) (dorms : Fin 3)
variable (distribution : students → dorms)

-- Conditions:
-- There must be exactly 5 students
-- Each dorm must have at least 1 and at most 2 students
-- Student 0 (representing student A) does not go to dorm 0 (representing dormitory A)

-- Define the premise: Student 0 is not assigned to dorm 0
def student0_not_in_dorm0 : Prop := distribution 0 ≠ 0

-- Define the range of dorm assignments
def dorm_condition : Prop := ∀ d, (1 ≤ d.card) ∧ (d.card ≤ 2)

-- Define the main statement
theorem distribution_number : (student0_not_in_dorm0 ∧ dorm_condition) → (card {f : students → dorms | student0_not_in_dorm0 ∧ dorm_condition f} = 60) := 
by
  sorry

end distribution_number_l523_523775


namespace find_math_marks_l523_523752

theorem find_math_marks :
  ∀ (english marks physics chemistry biology : ℕ) (average : ℕ),
  average = 78 →
  english = 91 →
  physics = 82 →
  chemistry = 67 →
  biology = 85 →
  (english + marks + physics + chemistry + biology) / 5 = average →
  marks = 65 :=
by
  intros english marks physics chemistry biology average h_average h_english h_physics h_chemistry h_biology h_avg_eq
  sorry

end find_math_marks_l523_523752


namespace kitten_length_after_4_months_l523_523219

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l523_523219


namespace john_total_payment_in_month_l523_523960

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end john_total_payment_in_month_l523_523960


namespace find_angle_B_l523_523010

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523010


namespace projection_norm_ratio_l523_523989

theorem projection_norm_ratio 
  {V : Type*} [InnerProductSpace ℝ V]
  (v w : V) 
  (hv_nonzero : v ≠ 0)
  (hw_nonzero : w ≠ 0)
  (p := (⟨(inner v w / inner w w) • w⟩ : V))
  (q := (⟨(inner p v / inner v v) • v⟩ : V))
  (r := (⟨(inner q w / inner w w) • w⟩ : V))
  (hp_ratio : ∥p∥ / ∥v∥ = 3 / 5) 
  : ∥r∥ / ∥v∥ = 27 / 125 :=
by 
  sorry

end projection_norm_ratio_l523_523989


namespace min_value_inequality_l523_523410

theorem min_value_inequality (a b c d e f : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f)
    (h_sum : a + b + c + d + e + f = 9) : 
    1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f ≥ 676 / 9 := 
by 
  sorry

end min_value_inequality_l523_523410


namespace sin_one_div_x_intercepts_l523_523271

theorem sin_one_div_x_intercepts :
  let f : ℝ → ℝ := λ x, sin (1 / x)
  let interval := set.Ioo (0.00001 : ℝ) (0.0001 : ℝ)
  let intercepts_count := (⌊100_000 / real.pi⌋ - ⌊10_000 / real.pi⌋)
  intercepts_count = 28648 :=
by
  -- Introducing necessary conditions
  have h1 : ∀ x, f x = 0 ↔ (∃ k : ℤ, x = (1 / (k * real.pi))) := sorry
  have interval_pos : 0 < 0.00001 := by norm_num
  have interval_small : 0.00001 < 0.0001 := by norm_num
  have interval_bound : ∀ x, x ∈ interval ↔ (0.00001 < x ∧ x < 0.0001) := sorry
  have int_lemma : ∀ a b : ℝ, (⌊b / real.pi⌋ - ⌊a / real.pi⌋) = intercepts_count := sorry
  -- Concluding the proof
  exact int_lemma 10_000 100_000

end sin_one_div_x_intercepts_l523_523271


namespace sum_of_fraction_in_checkerboard_l523_523795

open Nat

noncomputable def num_rectangles_checkerboard (n : ℕ) : ℕ :=
  binomial (n + 1) 2 * binomial (n + 1) 2

noncomputable def num_squares_checkerboard (n : ℕ) : ℕ :=
  (range (n + 1)).sum (λ k => k^2)

theorem sum_of_fraction_in_checkerboard : 
  let s := num_squares_checkerboard 7,
      r := num_rectangles_checkerboard 7,
      reduced_fraction := (Int.gcd s r),
      m := s / reduced_fraction,
      n := r / reduced_fraction
  in m + n = 33 := 
by
  sorry

end sum_of_fraction_in_checkerboard_l523_523795


namespace sum_integers_neg40_to_60_l523_523148

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end sum_integers_neg40_to_60_l523_523148


namespace exists_n_consecutive_lcm_unique_set_n_consecutive_l523_523167

-- Statement for part (a)
theorem exists_n_consecutive_lcm (n : ℕ) (h : n > 2) :
  (∃ a : ℕ, ∃ S : Finset ℕ, S.card = n ∧ (S.max' sorry = some (a + n - 1)) ∧ (a + n - 1 ∣ Finset.lcm (S.erase (a + n - 1)))) ↔ n ≥ 4 :=
begin
  sorry
end

-- Statement for part (b)
theorem unique_set_n_consecutive (n : ℕ) (h : n > 2) :
  (∃! a : ℕ, ∃ S : Finset ℕ, S.card = n ∧ (S.max' sorry = some (a + n - 1)) ∧ (a + n - 1 ∣ Finset.lcm (S.erase (a + n - 1)))) ↔ n = 4 :=
begin
  sorry
end

end exists_n_consecutive_lcm_unique_set_n_consecutive_l523_523167


namespace simplify_sqrt_7_pow_6_l523_523492

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523492


namespace average_surfers_correct_l523_523100

-- Define the number of surfers for each day
def surfers_first_day : ℕ := 1500
def surfers_second_day : ℕ := surfers_first_day + 600
def surfers_third_day : ℕ := (2 / 5 : ℝ) * surfers_first_day

-- Average number of surfers
def average_surfers : ℝ := (surfers_first_day + surfers_second_day + surfers_third_day) / 3

theorem average_surfers_correct :
  average_surfers = 1400 := 
  by 
    sorry

end average_surfers_correct_l523_523100


namespace find_x0_l523_523318

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then x^2 - 4 else 2 * x

theorem find_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 := by
  sorry

end find_x0_l523_523318


namespace inequality_l523_523067

theorem inequality (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
by
  sorry

end inequality_l523_523067


namespace pencils_count_l523_523162

theorem pencils_count (P L : ℕ) (h₁ : 6 * P = 5 * L) (h₂ : L = P + 4) : L = 24 :=
by sorry

end pencils_count_l523_523162


namespace arithmetic_sqrt_of_sqrt_16_l523_523560

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523560


namespace expected_shots_ivan_l523_523937

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l523_523937


namespace constant_term_expansion_l523_523107

theorem constant_term_expansion :
  let a := (x : ℂ) ^ 6
  let b := -(1 : ℂ) / (x * x ^ (1 / 2 : ℝ))
  let n := 5
  ∃ k : ℕ, 6 * (n - k) - (3 / 2 : ℝ) * k = 0 ∧ 
           (a ^ (n - k) * b ^ k).re = 5 :=
by
  sorry

end constant_term_expansion_l523_523107


namespace percentage_spent_on_clothing_l523_523401

variable (T : ℝ) -- total amount Jill spent excluding taxes
variable (C : ℝ) -- percentage spent on clothing

-- Conditions
def food_percentage : ℝ := 0.10
def other_items_percentage : ℝ := 0.30
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0.00
def other_items_tax_rate : ℝ := 0.08
def total_tax_rate : ℝ := 0.048

-- Problem statement
theorem percentage_spent_on_clothing :
  C = 0.6 * T :=
by
  have h1 : C = (1 - food_percentage - other_items_percentage) * T, from sorry
  have h2 : clothing_tax_rate * C + other_items_tax_rate * (other_items_percentage * T) = total_tax_rate * T, from sorry
  rw [h1] at h2
  rw[eq_comm] at h2
  exact sorry

end percentage_spent_on_clothing_l523_523401


namespace cubic_roots_expression_l523_523411

theorem cubic_roots_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -1) (h3 : a * b * c = 2) :
  2 * a * (b - c) ^ 2 + 2 * b * (c - a) ^ 2 + 2 * c * (a - b) ^ 2 = -36 :=
by
  sorry

end cubic_roots_expression_l523_523411


namespace female_officers_number_l523_523429

variables (F : ℕ) -- F represents the total number of female officers
variables (onDutyFemale : ℕ) -- onDutyFemale is the number of female officers on duty
variables (totalOnDuty : ℕ) -- totalOnDuty is the total number of officers on duty

def percent_female_on_duty := 0.23 -- 23 percent of female officers on duty
def percent_total_on_duty := 0.62 -- 62 percent of total officers on duty

-- Conditions
axiom duty_female_calc : onDutyFemale = (percent_total_on_duty * totalOnDuty).round
axiom onDuty_total : totalOnDuty = 225
axiom female_percentage_calc : onDutyFemale = (percent_female_on_duty * F).round

-- Question to Prove
theorem female_officers_number : F = 609 :=
begin
  sorry,
end

end female_officers_number_l523_523429


namespace even_integers_in_form_3k_plus_4_l523_523336

theorem even_integers_in_form_3k_plus_4 (n : ℕ) :
  (20 ≤ n ∧ n ≤ 180 ∧ ∃ k : ℕ, n = 3 * k + 4) → 
  (∃ s : ℕ, s = 27) :=
by
  sorry

end even_integers_in_form_3k_plus_4_l523_523336


namespace digital_earth_application_l523_523205

def digital_electric_earth := sorry

theorem digital_earth_application
    (makes_most_resources : digital_electric_earth -> Prop)
    (obtains_information_certain_ways : digital_electric_earth -> Prop)
    (cannot_control_crimes : digital_electric_earth -> Prop)
    (cannot_control_precipitation : digital_electric_earth -> Prop)
    (cannot_control_geological_disasters : digital_electric_earth -> Prop) :
    (∃ (d : digital_electric_earth), makes_most_resources d ∧ obtains_information_certain_ways d ∧
     cannot_control_crimes d ∧ cannot_control_precipitation d ∧ cannot_control_geological_disasters d) →
    (∃ (d : digital_electric_earth), provides_reference_data d) :=
sorry

end digital_earth_application_l523_523205


namespace simplify_sqrt_power_l523_523484

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523484


namespace angle_B_in_triangle_l523_523035

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523035


namespace largest_number_using_digits_l523_523676

theorem largest_number_using_digits (d1 d2 d3 : ℕ) (h1 : d1 = 7) (h2 : d2 = 1) (h3 : d3 = 0) : 
  ∃ n : ℕ, (n = 710) ∧ (∀ m : ℕ, (m = d1 * 100 + d2 * 10 + d3) ∨ (m = d1 * 100 + d3 * 10 + d2) ∨ (m = d2 * 100 + d1 * 10 + d3) ∨ 
  (m = d2 * 100 + d3 * 10 + d1) ∨ (m = d3 * 100 + d1 * 10 + d2) ∨ (m = d3 * 100 + d2 * 10 + d1) → n ≥ m) := 
by
  sorry

end largest_number_using_digits_l523_523676


namespace arithmetic_sqrt_of_sqrt_16_l523_523571

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523571


namespace range_of_a_l523_523897

theorem range_of_a (a : ℝ) (h : ∅ ⊂ {x : ℝ | x^2 ≤ a}) : 0 ≤ a :=
by
  sorry

end range_of_a_l523_523897


namespace no_real_roots_of_geom_seq_l523_523361

theorem no_real_roots_of_geom_seq (a b c : ℝ) (h_geom_seq : b^2 = a * c) : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  -- You can assume the steps of proving here
  sorry

end no_real_roots_of_geom_seq_l523_523361


namespace sum_cyc_geq_one_l523_523096

theorem sum_cyc_geq_one (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hcond : a * b + b * c + c * a = a * b * c) :
  (a^4 / (b * (b^4 + c^3)) + b^4 / (c * (c^3 + a^4)) + c^4 / (a * (a^4 + b^3))) ≥ 1 :=
sorry

end sum_cyc_geq_one_l523_523096


namespace triangle_inequality_max_sum_l523_523936

theorem triangle_inequality_max_sum {A B C M : Point} (hM : M ∈ triangle A B C) :
  dist M A + dist M B + dist M C ≤ max (dist A B + dist B C) (max (dist B C + dist C A) (dist C A + dist A B)) :=
sorry

end triangle_inequality_max_sum_l523_523936


namespace beautiful_set_exists_N_l523_523052

-- Definition of beautiful set
def is_beautiful (S : Set ℕ) : Prop :=
  ∀ {x y z : ℕ}, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → z ≠ x →
    (x ∣ (x + y + z) ∨ y ∣ (x + y + z) ∨ z ∣ (x + y + z))

-- Main theorem
theorem beautiful_set_exists_N : ∃ (N : ℕ), N = 6 ∧ 
  (∀ (S : Set ℕ), is_beautiful S → ∃ (n_s : ℕ), n_s ≥ 2 → 
    ∃ (count : ℕ), count ≤ N ∧ count = S.filter (λ x, ¬(n_s | x)).size) :=
by
  sorry

end beautiful_set_exists_N_l523_523052


namespace choose_starting_team_and_coach_l523_523078

theorem choose_starting_team_and_coach (team_size coach_position goalie_position players_left : ℕ) 
  (h_team_size : team_size = 15)
  (h_coach_position : coach_position = 1)
  (h_goalie_position : goalie_position = 1)
  (h_players_left : players_left = 5) :
  (team_size * (team_size - coach_position) * nat.choose (team_size - coach_position - goalie_position) players_left) = 270270 :=
by {
  -- Here we would normally provide a proof, but we are omitting the proof as per instructions
  sorry
}

end choose_starting_team_and_coach_l523_523078


namespace decagon_diagonal_relation_l523_523908

-- Define side length, shortest diagonal, and longest diagonal in a regular decagon
variable (a b d : ℝ)
variable (h1 : a > 0) -- Side length must be positive
variable (h2 : b > 0) -- Shortest diagonal length must be positive
variable (h3 : d > 0) -- Longest diagonal length must be positive

theorem decagon_diagonal_relation (ha : d^2 = 5 * a^2) (hb : b^2 = 3 * a^2) : b^2 = a * d :=
sorry

end decagon_diagonal_relation_l523_523908


namespace inscribed_hexagon_area_l523_523687

-- Definitions to represent the hexagon and its circumcircle
structure Hexagon (α β γ : ℝ) :=
  (AB BC CD DE EF FA: ℝ)
  (inscribed: True)
  (AB_eq_BC: AB = BC)
  (CD_eq_DE: CD = DE)
  (EF_eq_FA: EF = FA)

noncomputable def Area (α β γ : ℝ) (R: ℝ) :=
  2 * R^2 * sin(α) * sin(β) * sin(γ)

theorem inscribed_hexagon_area : 
  ∀ (α β γ R : ℝ) (hex : Hexagon α β γ), (Area α β γ R) ≤ (Area (α + β) / 2 (β + γ) / 2 (α + γ) / 2 R) :=
by
  intros α β γ R hex
  sorry

end inscribed_hexagon_area_l523_523687


namespace kho_kho_only_l523_523169

theorem kho_kho_only (kabaddi_total : ℕ) (both_games : ℕ) (total_players : ℕ) (kabaddi_only : ℕ) (kho_kho_only : ℕ) 
  (h1 : kabaddi_total = 10)
  (h2 : both_games = 5)
  (h3 : total_players = 50)
  (h4 : kabaddi_only = 10 - both_games)
  (h5 : kabaddi_only + kho_kho_only + both_games = total_players) :
  kho_kho_only = 40 :=
by
  -- Proof is not required
  sorry

end kho_kho_only_l523_523169


namespace arithmetic_sqrt_of_sqrt_16_l523_523564

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523564


namespace correct_propositions_l523_523414

variables (m n : Line) (α β γ : Plane)

-- Condition for proposition 1 and 3
axiom prop1_condition (h1 : m ⊥ α) (h2 : n ∥ α) : m ⊥ n
axiom prop3_condition (h3 : α ∥ β) (h4 : β ∥ γ) (h5 : m ⊥ α) : m ⊥ γ

theorem correct_propositions (h1 : m ⊥ α)
                            (h2 : n ∥ α)
                            (h3 : α ∥ β)
                            (h4 : β ∥ γ)
                            (h5 : m ⊥ α) :
  (∀ m n α, m ⊥ α ∧ n ∥ α → m ⊥ n) ∧ (∀ α β γ m, α ∥ β ∧ β ∥ γ ∧ m ⊥ α → m ⊥ γ) :=
by {
  split;
  { intros,
    { exact prop1_condition h1 h2 },
    { exact prop3_condition h3 h4 h5 } }
}

end correct_propositions_l523_523414


namespace integral_of_odd_function_l523_523234

theorem integral_of_odd_function (f : ℝ → ℝ) (a : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_symm : a > 0):
  ∫ x in -a..a, f x = 0 := 
by
  sorry

example : ∫ x in (-3 : ℝ)..3, x^3 * Real.cos x = 0 :=
begin
  apply integral_of_odd_function,
  intros x,
  simp,
  ring
end

end integral_of_odd_function_l523_523234


namespace total_cans_per_closet_l523_523953

theorem total_cans_per_closet
  (cans_per_row : ℕ)
  (rows_per_shelf : ℕ)
  (shelves_per_closet : ℕ) :
  cans_per_row = 12 →
  rows_per_shelf = 4 →
  shelves_per_closet = 10 →
  cans_per_row * rows_per_shelf * shelves_per_closet = 480 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_cans_per_closet_l523_523953


namespace problem_l523_523129

def is_acute_angle (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_first_quadrant (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_second_quadrant (θ: ℝ) : Prop := θ > 90 ∧ θ < 180

def cond1 (θ: ℝ) : Prop := θ < 90 → is_acute_angle θ
def cond2 (θ: ℝ) : Prop := in_first_quadrant θ → θ ≥ 0
def cond3 (θ: ℝ) : Prop := is_acute_angle θ → in_first_quadrant θ
def cond4 (θ θ': ℝ) : Prop := in_second_quadrant θ → in_first_quadrant θ' → θ > θ'

theorem problem :
  (¬ ∃ θ, cond1 θ) ∧ (¬ ∃ θ, cond2 θ) ∧ (∃ θ, cond3 θ) ∧ (¬ ∃ θ θ', cond4 θ θ') →
  (number_of_correct_propositions = 1) :=
  by
    sorry

end problem_l523_523129


namespace arithmetic_sqrt_of_sqrt_16_l523_523606

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523606


namespace degree_of_g_at_least_5_l523_523688

open Polynomial

theorem degree_of_g_at_least_5
    (f g : Polynomial ℤ)
    (h_nonzero_deg_f : f.degree ≠ 0)
    (h_nonzero_deg_g : g.degree ≠ 0)
    (h_g_divides_f : g ∣ f)
    (h_f_plus_2009_has_50_int_roots : (f + C 2009).roots_nodup.card = 50) :
    g.degree ≥ 5 := 
  sorry

end degree_of_g_at_least_5_l523_523688


namespace find_eccentricity_l523_523310

noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
noncomputable def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
noncomputable def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x
noncomputable def distance_to_axis (p x y : ℝ) : Prop := x = p

theorem find_eccentricity (p a b : ℝ)
  (hp : 0 < p)
  (ha : 0 < a)
  (hb : 0 < b)
  (hA_parabola : ∃ x y, parabola p x y ∧ asymptote a b x y ∧ hyperbola a b x y)
  (hA_axis : ∃ x y, parabola p x y ∧ distance_to_axis p x y) :
  ∃ e : ℝ, e = Real.sqrt 5 :=
begin
  sorry
end

end find_eccentricity_l523_523310


namespace leif_has_more_oranges_than_apples_l523_523969

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l523_523969


namespace find_angle_B_l523_523016

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523016


namespace arithmetic_sqrt_of_sqrt_16_l523_523607

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523607


namespace find_2u_plus_v_l523_523997

-- Definitions and Conditions
variables (u v : ℤ)
hypothesis (h0 : 0 < v ∧ v < u)
def A := (2 * u, v)
def B := (v, 2 * u)
def C := (-v, 2 * u)
def D := (-v, -2 * u)
def E := (v, -2 * u)
def pentagon_area := 8 * u * v + 2 * u * v = 902


-- Goal
theorem find_2u_plus_v :
  0 < v ∧ v < u ∧ 8 * u * v + 2 * u * v = 902 → 2 * u + v = 29 :=
sorry

end find_2u_plus_v_l523_523997


namespace max_elements_set_M_l523_523857

theorem max_elements_set_M (n : ℕ) (hn : n ≥ 2) (M : Finset (ℕ × ℕ))
  (hM : ∀ {i k}, (i, k) ∈ M → i < k → ∀ {m}, k < m → (k, m) ∉ M) :
  M.card ≤ n^2 / 4 :=
sorry

end max_elements_set_M_l523_523857


namespace decrease_in_average_l523_523188

noncomputable def original_average := 12.4
noncomputable def runs_in_last_match := 26
noncomputable def additional_wickets := 7
noncomputable def initial_wickets := 145

def new_wickets := initial_wickets + additional_wickets
def original_runs := initial_wickets * original_average
def new_runs := original_runs + runs_in_last_match
def new_average := new_runs / new_wickets

theorem decrease_in_average : 
  original_average - new_average ≈ 0.4132 := by sorry

end decrease_in_average_l523_523188


namespace find_third_integer_l523_523642

theorem find_third_integer (a b c : ℕ) (h1 : a * b * c = 42) (h2 : a + b = 9) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : c = 3 :=
sorry

end find_third_integer_l523_523642


namespace percent_loss_on_transaction_l523_523706

variable (cost_per_sheep : ℝ)
variables (sheep_bought sheep_sold_initial sheep_sold_remaining : ℕ)
variables (initial_revenue_multiplier : ℝ) (additional_price_percentage_increase : ℝ)

-- Conditions setup
def cost_sheep (cost_per_sheep : ℝ) (n: ℕ) : ℝ :=
  cost_per_sheep * n

def total_revenue_initial (n : ℕ) (cost_per_sheep : ℝ) (multiplier : ℝ) : ℝ :=
  cost_per_sheep * multiplier / n

def total_revenue_remaining (n : ℕ) (initial_price_per_sheep : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price_per_sheep * (percentage_increase / 100) * n

-- Theorem to prove
theorem percent_loss_on_transaction :
  let cost_per_sheep: ℝ := cost_per_sheep in
  let sheep_bought := 800 in
  let sheep_sold_initial := 600 in
  let sheep_sold_remaining := 200 in
  let initial_revenue_multiplier := 700 in
  let additional_price_percentage_increase := 10 in
  let cost := cost_sheep cost_per_sheep sheep_bought in
  let revenue_initial := total_revenue_initial sheep_sold_initial cost_per_sheep initial_revenue_multiplier in
  let revenue_remaining := total_revenue_remaining sheep_sold_remaining (total_revenue_initial sheep_sold_initial cost_per_sheep initial_revenue_multiplier / sheep_sold_initial) additional_price_percentage_increase in
  let total_revenue := revenue_initial + revenue_remaining in
  let profit_or_loss := total_revenue - cost in
  let percent_loss := (profit_or_loss.abs / cost) * 100 in
  percent_loss = 6.08 :=
sorry

end percent_loss_on_transaction_l523_523706


namespace max_value_expr_l523_523059

theorem max_value_expr (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxyz : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (x - y + z) ≤ 2187 / 216 :=
sorry

end max_value_expr_l523_523059


namespace hurleys_age_l523_523613

-- Definitions and conditions
variable (H R : ℕ)
variable (cond1 : R - H = 20)
variable (cond2 : (R + 40) + (H + 40) = 128)

-- Theorem statement
theorem hurleys_age (H R : ℕ) (cond1 : R - H = 20) (cond2 : (R + 40) + (H + 40) = 128) : H = 14 := 
by
  sorry

end hurleys_age_l523_523613


namespace arithmetic_seq_div_by_7_l523_523435

theorem arithmetic_seq_div_by_7 :
  ∀ (a : ℕ), ∃! x ∈ (list.range 7).map (λ n, a + n * 30), x % 7 = 0 := 
by sorry

end arithmetic_seq_div_by_7_l523_523435


namespace log_infinite_expression_pos_l523_523768

theorem log_infinite_expression_pos :
  let x := real.logb 3 (81 + real.logb 3 (81 + real.logb 3 (81 + ...)))
  in x = 4 :=
sorry

end log_infinite_expression_pos_l523_523768


namespace arithmetic_sequence_sum_q_l523_523103

theorem arithmetic_sequence_sum_q (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 2) + a (n + 1) = 2 * a n)
  (hq : q ≠ 1) :
  S 5 = 11 :=
sorry

end arithmetic_sequence_sum_q_l523_523103


namespace sequence_correct_propositions_l523_523125

-- Define the sequence with given properties
noncomputable def sequence := ℕ → ℝ
def a₁ : ℝ := 1
def a₂ : ℝ := 2
def lambda : ℝ := -1
def an (a : sequence) : ℕ → ℝ
  | 1 := a₁
  | 2 := a₂
  | 2 * n + 1 := (a (2 * n + 1) - a (2 * n)) + 2 * a (2 * n)
  | 2 * n + 2 := (a (2 * n) * a (2 * n + 2)) / a (2 * n + 1)

-- Define propositions
def prop1 (λ : ℝ) (a : sequence) : Prop := λ = 1 → a 3 = 3
def prop2 (λ : ℝ) (a : sequence) : Prop := λ = -1 → a 4 < 0
def prop3 (a : sequence) : Prop := ∃ λ : ℝ, λ > 0 ∧ a 3 = a 4
def prop4 (a : sequence) : Prop := ∀ λ : ℝ, True

-- Number of correct propositions
def num_correct_props (a : sequence) : ℕ :=
if h1 : prop1 lambda a then
  if h2 : prop2 lambda a then
    if h3 : prop3 a then
      if h4 : prop4 a then 4 else 3
    else if h4 : prop4 a then 2 else 1
  else if h3 : prop3 a then
    if h4 : prop4 a then 3 else 2
  else if h4 : prop4 a then 1 else 0
else if h2 : prop2 lambda a then
  if h3 : prop3 a then
    if h4 : prop4 a then 3 else 2
  else if h4 : prop4 a then 1 else 0
else if h3 : prop3 a then
  if h4 : prop4 a then 2 else 1
else if h4 : prop4 a then 1 else 0

theorem sequence_correct_propositions : num_correct_props an = 3 :=
sorry

end sequence_correct_propositions_l523_523125


namespace coefficient_of_x3_in_expansion_l523_523670

theorem coefficient_of_x3_in_expansion :
  let coeff := 56 * 972 * Real.sqrt 2
  coeff = 54432 * Real.sqrt 2 :=
by
  let coeff := 56 * 972 * Real.sqrt 2
  have h : coeff = 54432 * Real.sqrt 2 := sorry
  exact h

end coefficient_of_x3_in_expansion_l523_523670


namespace range_of_a_l523_523131

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, ¬ (x^2 - a * x + 1 ≤ 0)) ↔ -2 < a ∧ a < 2 := 
sorry

end range_of_a_l523_523131


namespace arithmetic_square_root_of_sqrt_16_l523_523579

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523579


namespace simplify_sqrt_pow_six_l523_523471

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523471


namespace find_values_l523_523423

theorem find_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) 
  (h2 : x + y^2 = 45) : 
  x = 7 ∧ y = Real.sqrt 38 :=
by
  sorry

end find_values_l523_523423


namespace median_of_set_condition_l523_523298

-- Variables and conditions
variables (X : ℕ) (data_set1 : List ℕ) (data_set2 : List ℕ)

-- Defining the sets and mode condition
def condition_mode := (data_set1 = [15, 13, 9, X, 7]) ∧ (∃! m, m = 9 ∧ (∃ m == 9, mode data_set1 = m))

-- Defining the expected median condition with X = 9
def condition_median := (X = 9) ∧ (data_set2 = [10, 11, 14, 8, X]) ∧ (median(data_set2) = 10)

-- Main theorem statement
theorem median_of_set_condition : condition_mode X data_set1 → condition_median X data_set2 → median(data_set2) = 10 :=
by
  intros h_mode h_median
  rcases h_mode with ⟨h_set1, h_mode_exists⟩
  sorry

end median_of_set_condition_l523_523298


namespace symmetric_coloring_probability_l523_523386

-- Defining the problem conditions
def num_squares : ℕ := 13
def red_squares : ℕ := 8
def blue_squares : ℕ := 5
def central_square_color : Color := Color.blue

-- Defining the problem statement
theorem symmetric_coloring_probability (n : ℕ) :
  let total_colorings := nat.comb num_squares blue_squares,
      valid_symmetries := 3 in
  (valid_symmetries : ℚ) / total_colorings = 1 / n ↔ n = 429 := by
  sorry

end symmetric_coloring_probability_l523_523386


namespace virginia_initial_sweettarts_l523_523669

theorem virginia_initial_sweettarts :
  ∃ (initial_sweettarts : ℕ),
    (∀ (given_each : ℕ), given_each = 3 →
    let given_to_friends := 4 * given_each in
    let eaten_by_virginia := 3 in
    initial_sweettarts = given_to_friends + eaten_by_virginia) :=
exists.intro 15 (by intros given_each h;
                let given_to_friends := 4 * given_each;
                let eaten_by_virginia := 3;
                rw h;
                rfl)

end virginia_initial_sweettarts_l523_523669


namespace ivan_expected_shots_l523_523941

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l523_523941


namespace wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l523_523140

noncomputable def number_of_cubes (n : ℕ) : ℕ :=
  n + (n - 1) + n

noncomputable def painted_area (n : ℕ) : ℕ :=
  (5 * n) + (3 * (n + 1)) + (2 * (n - 2))

theorem wall_with_5_peaks_has_14_cubes : number_of_cubes 5 = 14 :=
  by sorry

theorem wall_with_2014_peaks_has_6041_cubes : number_of_cubes 2014 = 6041 :=
  by sorry

theorem painted_area_wall_with_2014_peaks : painted_area 2014 = 20139 :=
  by sorry

end wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l523_523140


namespace simplify_sqrt_7_pow_6_l523_523488

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l523_523488


namespace area_MNC_fraction_ABC_l523_523370

variable {α : Type*} [Field α]

-- Defining the points and the triangle
structure Point (α : Type*) [Field α] :=
(x : α)
(y : α)

def Triangle (α : Type*) [Field α] := (Point α) × (Point α) × (Point α)

def midpoint (A B : Point α) : Point α :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

-- Function to define the centroid given two medians AD and CE intersect at M
def centroid (A B C D E M : Point α) : Prop :=
(A.x + C.x + M.x)/3 = M.x ∧ (A.y + C.y + M.y)/3 = M.y

-- Assumptions based on the problem statement
variables (A B C D E M N : Point α)
variables (ABC : Triangle α)
variables (hMedians : centroid A B C D E M)
variables (hNMidpoint : N = midpoint A C)

-- Main theorem
theorem area_MNC_fraction_ABC (hMedians : centroid A B C D E M) (hNMidpoint : N = midpoint A C) :
  area_of_triangle M N C = (1 : α) / 6 * area_of_triangle A B C :=
sorry

end area_MNC_fraction_ABC_l523_523370


namespace linear_function_passing_quadrants_l523_523839

theorem linear_function_passing_quadrants (b : ℝ) :
  (∀ x : ℝ, (y = x + b) ∧ (y > 0 ↔ (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0))) →
  b > 0 :=
sorry

end linear_function_passing_quadrants_l523_523839


namespace dhoni_initial_toys_l523_523773

theorem dhoni_initial_toys (x : ℕ) (T : ℕ) 
    (h1 : T = 10 * x) 
    (h2 : T + 16 = 66) : x = 5 := by
  sorry

end dhoni_initial_toys_l523_523773


namespace adam_change_l523_523166

theorem adam_change (adam_has cost_of_airplane change : ℝ) (h_price : cost_of_airplane = 4.28) (h_money : adam_has = 5.00) (h_change : change = adam_has - cost_of_airplane) :
  change = 0.72 :=
by
  have h1 : change = 5.00 - 4.28, by rw [h_money, h_price, h_change]
  rw h1
  sorry

end adam_change_l523_523166


namespace impossible_coins_l523_523456

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l523_523456


namespace simplify_sqrt_pow_six_l523_523476

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l523_523476


namespace find_monic_polynomial_of_shifted_roots_l523_523991

theorem find_monic_polynomial_of_shifted_roots (a b c : ℝ) (h : ∀ x : ℝ, (x - a) * (x - b) * (x - c) = x^3 - 5 * x + 7) : 
  (x : ℝ) → (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 22 * x + 19 :=
by
  -- Proof will be provided here.
  sorry

end find_monic_polynomial_of_shifted_roots_l523_523991


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523600

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523600


namespace simplify_sqrt7_pow6_l523_523527

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l523_523527


namespace oil_drop_probability_l523_523079

def circle_area (d : ℝ) : ℝ := π * (d / 2) ^ 2
def square_area (side : ℝ) : ℝ := side ^ 2

theorem oil_drop_probability :
  let diameter := 2
  let side_length := 1
  let circle_Area := circle_area diameter
  let square_Area := square_area side_length
  (square_Area / circle_Area) = (1 / π) :=
by
  sorry

end oil_drop_probability_l523_523079


namespace num_divisors_180_l523_523352

-- Define a positive integer 180
def n : ℕ := 180

-- Define the function to calculate the number of divisors using prime factorization
def num_divisors (n : ℕ) : ℕ :=
  let factors := [(2, 2), (3, 2), (5, 1)] in
  factors.foldl (λ acc (p : ℕ × ℕ), acc * (p.snd + 1)) 1

-- The main theorem statement
theorem num_divisors_180 : num_divisors n = 18 :=
by
  sorry

end num_divisors_180_l523_523352


namespace sum_of_roots_l523_523645

noncomputable def phi (k : ℕ) : ℝ := (225 + 360 * k) / 5

theorem sum_of_roots :
  let φ := λ k, phi k in
  (φ 0 + φ 1 + φ 2 + φ 3 + φ 4) = 1125 :=
by
  sorry

end sum_of_roots_l523_523645


namespace area_of_triangle_ABC_is_9_l523_523261

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def dist (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

def triangle_area (A B C : Point3D) : ℝ :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  if AB^2 + BC^2 = AC^2 then
    0.5 * AB * BC
  else if AB^2 + AC^2 = BC^2 then
    0.5 * AB * AC
  else if AC^2 + BC^2 = AB^2 then
    0.5 * AC * BC
  else
    -- If not a right triangle, use an alternative method (Heron's formula or vector cross product)
    sorry  -- This has to be implemented if the triangle is not right-angled.

def A : Point3D := { x := 1, y := 8, z := 11 }
def B : Point3D := { x := 0, y := 7, z := 7 }
def C : Point3D := { x := -3, y := 10, z := 7 }

theorem area_of_triangle_ABC_is_9 : triangle_area A B C = 9 := by
  sorry

end area_of_triangle_ABC_is_9_l523_523261


namespace arithmetic_square_root_of_sqrt_16_l523_523585

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523585


namespace simplify_sqrt_seven_pow_six_l523_523532

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523532


namespace cos_lt_cos_iff_sin_gt_sin_l523_523396

variable {A B C : ℝ}

theorem cos_lt_cos_iff_sin_gt_sin (hA : 0 < A ∧ A < π)
                                 (hB : 0 < B ∧ B < π)
                                 (hABC : A + B + C = π) :
    cos A < cos B ↔ sin A > sin B :=
begin
  sorry
end

end cos_lt_cos_iff_sin_gt_sin_l523_523396


namespace arithmetic_sqrt_of_sqrt_16_l523_523568

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523568


namespace proportion_of_salt_correct_l523_523250

def grams_of_salt := 50
def grams_of_water := 1000
def total_solution := grams_of_salt + grams_of_water
def proportion_of_salt : ℚ := grams_of_salt / total_solution

theorem proportion_of_salt_correct :
  proportion_of_salt = 1 / 21 := 
  by {
    sorry
  }

end proportion_of_salt_correct_l523_523250


namespace impossible_coins_l523_523455

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l523_523455


namespace number_of_divisors_of_180_l523_523347

theorem number_of_divisors_of_180 : 
   (nat.coprime 2 3 ∧ nat.coprime 3 5 ∧ nat.coprime 5 2 ∧ 180 = 2^2 * 3^2 * 5^1) →
   (nat.divisors_count 180 = 18) :=
by
  sorry

end number_of_divisors_of_180_l523_523347


namespace two_colorable_plane_division_l523_523087

-- Define the problem in a Lean environment
theorem two_colorable_plane_division (n : ℕ) :
  ∀ (lines : set ℝ^2) (circles : set ℝ^2), divided_by_lines_and_circles lines circles →  
  ∃ (colors : set ℝ^2 → Prop), 
    two_colorable colors ∧ 
    ∀ (region1 region2 : set ℝ^2), 
      adjacent region1 region2 → colors region1 ≠ colors region2 := 
  sorry

end two_colorable_plane_division_l523_523087


namespace acetic_acid_produced_from_reactants_l523_523792

-- Definitions of conditions
def ethanol := "C2H5OH"
def oxygen := "O2"
def acetic_acid := "CH3COOH"
def water := "H2O"

-- Defining the balanced chemical equation
def balanced_equation := ethanol + oxygen = acetic_acid + water

-- Number of moles initially present
def initial_moles_ethanol := 3
def initial_moles_oxygen := 3

-- Stoichiometric ratio extracted from balanced equation
def ethanol_to_acetic_acid_ratio := 1
def oxygen_to_acetic_acid_ratio := 1

theorem acetic_acid_produced_from_reactants :
  initial_moles_ethanol = 3 → initial_moles_oxygen = 3 → 3 = 3 :=
by
  sorry

end acetic_acid_produced_from_reactants_l523_523792


namespace number_of_cherries_l523_523708

-- Definitions for the problem conditions
def total_fruits : ℕ := 580
def raspberries (b : ℕ) : ℕ := 2 * b
def grapes (c : ℕ) : ℕ := 3 * c
def cherries (r : ℕ) : ℕ := 3 * r

-- Theorem to prove the number of cherries
theorem number_of_cherries (b r g c : ℕ) 
  (H1 : b + r + g + c = total_fruits)
  (H2 : r = raspberries b)
  (H3 : g = grapes c)
  (H4 : c = cherries r) :
  c = 129 :=
by sorry

end number_of_cherries_l523_523708


namespace problem_l523_523832

theorem problem (x : ℂ) (h : x - 1/x = 3 * complex.I) : x^12 - 1/x^12 = 103682 :=
by
  sorry

end problem_l523_523832


namespace bikes_in_parking_lot_l523_523907

theorem bikes_in_parking_lot (n c : ℕ) (h1 : c = 14) (h2 : n = 66) (h3 : ∀ x, x ∈ {c, b} -> x * 4 + x * 2 = n) : b = 5 := by
    sorry

end bikes_in_parking_lot_l523_523907


namespace FayeScores36_l523_523783

variable (totalPoints : Nat) (numPlayers : Nat) (pointsPerOther : Nat)

theorem FayeScores36 (h1 : totalPoints = 68) (h2 : numPlayers = 5) (h3 : pointsPerOther = 8) : 
  let FayePoints := totalPoints - (numPlayers - 1) * pointsPerOther in
  FayePoints = 36 :=
by
  sorry

end FayeScores36_l523_523783


namespace find_four_digit_number_l523_523624

/-- 
  If there exists a positive integer M and M² both end in the same sequence of 
  five digits abcde in base 10 where a ≠ 0, 
  then the four-digit number abcd derived from M = 96876 is 9687.
-/
theorem find_four_digit_number
  (M : ℕ)
  (h_end_digits : (M % 100000) = (M * M % 100000))
  (h_first_digit_nonzero : 10000 ≤ M % 100000  ∧ M % 100000 < 100000)
  : (M = 96876 → (M / 10 % 10000 = 9687)) :=
by { sorry }

end find_four_digit_number_l523_523624


namespace bugs_meet_at_s_l523_523137

noncomputable def length_PQ : ℝ := 8
noncomputable def length_QR : ℝ := 10
noncomputable def length_PR : ℝ := 12
noncomputable def speed_clockwise : ℝ := 2
noncomputable def speed_counterclockwise : ℝ := 3

theorem bugs_meet_at_s (RS : ℝ) :
  ∃ S : ℝ, S ∈ set.Icc 0 length_QR ∧ RS = QR - S := by
begin
  sorry

end bugs_meet_at_s_l523_523137


namespace kitten_length_doubling_l523_523215

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l523_523215


namespace sin_minus_cos_obtuse_triangle_l523_523802

variables (α : ℝ) (sin cos : ℝ → ℝ)

-- Define the main condition
axiom sin_cos_condition : sin α + cos α = 1 / 5

-- Statement for part (1)
theorem sin_minus_cos (h: sin α + cos α = 1 / 5) : sin α - cos α = -7 / 5 :=
sorry

-- Statement for part (2)
theorem obtuse_triangle (h: sin α + cos α = 1 / 5) (internal_angle : 0 < α ∧ α < π) :
  let sin_alpha := 4 / 5 in
  let cos_alpha := -3 / 5 in
  α > π / 2 ∧ α < π :=
sorry

-- End of Lean 4 statements

end sin_minus_cos_obtuse_triangle_l523_523802


namespace arithmetic_square_root_of_sqrt_16_l523_523578

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523578


namespace simplify_sqrt7_pow6_l523_523500

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523500


namespace cost_price_of_watch_l523_523728

/-
Let's state the problem conditions as functions
C represents the cost price
SP1 represents the selling price at 36% loss
SP2 represents the selling price at 4% gain
-/

def cost_price (C : ℝ) : ℝ := C

def selling_price_loss (C : ℝ) : ℝ := 0.64 * C

def selling_price_gain (C : ℝ) : ℝ := 1.04 * C

def price_difference (C : ℝ) : ℝ := (selling_price_gain C) - (selling_price_loss C)

theorem cost_price_of_watch : ∀ C : ℝ, price_difference C = 140 → C = 350 :=
by
   intro C H
   sorry

end cost_price_of_watch_l523_523728


namespace expected_shots_ivan_l523_523940

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l523_523940


namespace bob_password_probability_l523_523736

def num_non_negative_single_digits : ℕ := 10
def num_odd_single_digits : ℕ := 5
def num_even_positive_single_digits : ℕ := 4
def probability_first_digit_odd : ℚ := num_odd_single_digits / num_non_negative_single_digits
def probability_middle_letter : ℚ := 1
def probability_last_digit_even_positive : ℚ := num_even_positive_single_digits / num_non_negative_single_digits

theorem bob_password_probability :
  probability_first_digit_odd * probability_middle_letter * probability_last_digit_even_positive = 1 / 5 :=
by
  sorry

end bob_password_probability_l523_523736


namespace positive_integer_representation_l523_523786

theorem positive_integer_representation (a b c n : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) 
  (h₄ : n = (abc + a * b + a) / (abc + c * b + c)) : n = 1 ∨ n = 2 := 
by
  sorry

end positive_integer_representation_l523_523786


namespace van_distance_covered_l523_523200

theorem van_distance_covered :
  let t₁ := 6 in                         -- Initial time to cover distance
  let t₂ := 3/2 * t₁ in                 -- New time to cover the same distance
  let s₂ := 28 in                       -- New speed
  D = s₂ * t₂ →                         -- Expected distance calculation using the new speed and time
  D = 252 :=                            -- Distance covered by the van
by
  sorry

end van_distance_covered_l523_523200


namespace polygon_eight_sides_l523_523057

variables {b x y : ℝ}

def T (b : ℝ) (x y : ℝ) : Prop :=
  b ≤ x ∧ x ≤ 3 * b ∧
  b ≤ y ∧ y ≤ 3 * b ∧
  x + y ≥ 2 * b ∧
  x + 2 * b ≥ y ∧
  y + 2 * b ≥ x ∧
  (x - 2 * b)^2 + (y - 2 * b)^2 ≤ b^2 

theorem polygon_eight_sides (b : ℝ) (hb : 0 < b) : 
  ∃ (n : ℕ), (n = 8) ∧ (∃ (vertices : fin n → ℝ × ℝ), ∀ i, T b (vertices i).1 (vertices i).2) :=
sorry

end polygon_eight_sides_l523_523057


namespace collinear_B_E_R_l523_523812

-- Definitions

variables {A B C D P Q R E : Point} {Γ : Circle}

-- Conditions
variables (h1 : InscribedQuadrilateral Γ A B C D)
variables (h2 : OnExtensionOfSegment P A C)
variables (h3 : TangentToCircle Γ P B ∧ TangentToCircle Γ P D)
variables (h4 : TangentThroughPoint Γ C intersecting (PD) (AD) Q R)
variables (h5 : SecondIntersectionPoint AQ Γ E)

-- Statement
theorem collinear_B_E_R 
  (h1 : InscribedQuadrilateral Γ A B C D)
  (h2 : OnExtensionOfSegment P A C)
  (h3 : TangentToCircle Γ P B ∧ TangentToCircle Γ P D)
  (h4 : TangentThroughPoint Γ C intersecting (PD) (AD) Q R)
  (h5 : SecondIntersectionPoint AQ Γ E) :
  Collinear B E R :=
sorry

end collinear_B_E_R_l523_523812


namespace find_n_for_conditions_l523_523259

theorem find_n_for_conditions :
  ∀ (n : ℕ), n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 5 →
  ∃ (k : ℕ), k ≥ 2 ∧ ∃ (a : Fin k → ℚ), 
  (∀ i, 0 < a i) ∧
  (∑ i, a i = n) ∧
  (∏ i, a i = n) :=
by
  intros n hn
  sorry

end find_n_for_conditions_l523_523259


namespace factorization_implies_k_l523_523293

theorem factorization_implies_k (x y k : ℝ) (h : ∃ (a b c d e f : ℝ), 
                            x^3 + 3 * x^2 - 2 * x * y - k * x - 4 * y = (a * x + b * y + c) * (d * x^2 + e * xy + f)) :
  k = -2 :=
sorry

end factorization_implies_k_l523_523293


namespace anna_more_candy_than_billy_l523_523221

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end anna_more_candy_than_billy_l523_523221


namespace find_pairs_l523_523972

def matrix_eq {a b : ℝ} (P : Matrix 2 2 ℝ) : Prop :=
  P ⬝ Pᵀ = 25 • (1 : Matrix 2 2 ℝ)

theorem find_pairs (a b : ℝ) (h : matrix_eq (Matrix.vecCons (Matrix.vecCons 3 a Vector.nil) (Matrix.vecCons 4 b Vector.nil))) :
  (a = 4 ∧ b = -3) ∨ (a = -4 ∧ b = 3) :=
sorry

end find_pairs_l523_523972


namespace largest_even_numbers_in_grid_l523_523809

theorem largest_even_numbers_in_grid (n : ℕ) (h_positive : 0 < n) : 
    let N := if n % 2 = 0 then n^2 else n^2 - n + 1 in 
    ∃ f : Π (i j : Fin n), ℤ, 
    ∃ M : (Fin n) × (Fin n) → ((Fin n) → (Fin n) → ℤ) → ((Fin n) → (Fin n) → ℤ), 
    (∀ (i j : Fin n) (g : (Fin n) → (Fin n) → ℤ), 
        let updated_g := M (i, j) g in 
        ∃ (num_even : ℕ), 
        (num_even ≥ N) ∧ (∀ (i j : Fin n), even (updated_g i j) ↔ even (f i j + (if i = j then 1 else 0)))) :=
sorry

end largest_even_numbers_in_grid_l523_523809


namespace dayNumberAppearances_l523_523781

def numOfTimesDayAppears (i n : ℕ) : ℕ :=
  if i <= n then i * (i + 1) / 2
  else if 2 * n - 1 <= i ∧ i <= 3 * n - 2 then numOfTimesDayAppears (3 * n - 1 - i) n
  else
    let s := (i + 1 - n) + (i - n) + ... + (2 * n - i)
    s / 2

theorem dayNumberAppearances
  (n : ℕ) 
  (i : ℕ): 
  ∃ A : ℕ, A = numOfTimesDayAppears i n :=
sorry

end dayNumberAppearances_l523_523781


namespace calculate_markup_l523_523161

-- Definitions of the conditions
def purchase_price : ℝ := 72
def overhead_rate : ℝ := 0.40
def tax_rate : ℝ := 0.08
def net_profit : ℝ := 25
def discount_rate : ℝ := 0.10

-- Computation to prove the required markup
theorem calculate_markup :
  let overhead := overhead_rate * purchase_price,
      taxes := tax_rate * purchase_price,
      total_cost := purchase_price + overhead + taxes + net_profit,
      sp := total_cost / (1 - discount_rate),
      markup := sp - purchase_price
  in
    markup = 74.18 := by
  sorry

end calculate_markup_l523_523161


namespace rectangular_table_sum_squares_l523_523434

-- Define sequences
def a (i : ℕ) : ℕ := sorry -- Placeholder, assume a has properties
def b (j : ℕ) : ℕ := sorry -- Placeholder, assume b has properties

-- Define properties of sequences a and b based on conditions
axiom a_conditions (i : ℕ) (k : ℕ) : 
  a 1 >= 3 ∧
  (∀ i, i >= 2 → a i % 2 = 0) ∧
  (a 1 ∧ ∑ i in range (n - 1), a i ^ 2 = 2 * k + 1)  

axiom b_conditions (j : ℕ) (S : ℕ) : 
  b 1 > 2 * a n ∧
  (∀ j, j >= 2 → b j % 2 = 0) ∧
  (b 1 ∧ ∑ j in range (m - 1), b j ^ 2 = 2 * S + 1)

-- Define the table
def table (i j : ℕ) : ℕ := (a i) ^ 2 * (b j) ^ 2

-- Main theorem
theorem rectangular_table_sum_squares (m n : ℕ) : 
  (∀ i, i < m → ∑ j in range n, table i j = (k+1)^2) ∧
  (∀ j, j < n → ∑ i in range m, table i j = (S+1)^2) :=
sorry

end rectangular_table_sum_squares_l523_523434


namespace number_of_distinct_four_digit_integers_with_product_18_l523_523876

theorem number_of_distinct_four_digit_integers_with_product_18 : 
  ∃ l : List (List ℕ), (∀ d ∈ l, d.length = 4 ∧ d.product = 18) ∧
    l.foldr (λ x acc, acc + Multiset.card (x.toMultiset)) 0 = 36 := 
  sorry

end number_of_distinct_four_digit_integers_with_product_18_l523_523876


namespace evaluate_expression_l523_523256

theorem evaluate_expression :
  ( ( ( 5 / 2 : ℚ ) / ( 7 / 12 : ℚ ) ) - ( 4 / 9 : ℚ ) ) = ( 242 / 63 : ℚ ) :=
by
  sorry

end evaluate_expression_l523_523256


namespace expected_shots_ivan_l523_523938

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l523_523938


namespace square_area_is_256_l523_523133

-- Definitions of the conditions
def rect_width : ℝ := 4
def rect_length : ℝ := 3 * rect_width
def side_of_square : ℝ := rect_length + rect_width

-- Proposition
theorem square_area_is_256 (rect_width : ℝ) (h1 : rect_width = 4) 
                           (rect_length : ℝ) (h2 : rect_length = 3 * rect_width) :
  side_of_square ^ 2 = 256 :=
by 
  sorry

end square_area_is_256_l523_523133


namespace triangle_angle_B_l523_523004

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523004


namespace count_15_letter_arrangements_l523_523335

open Nat

-- Define the conditions
def valid_arrangement (arr : List Char) : Prop :=
  (∀ i, i < 5 → arr[i] ≠ 'B') ∧
  (∀ i, 5 ≤ i ∧ i < 10 → arr[i] ≠ 'C') ∧
  (∀ i, 10 ≤ i ∧ i < 15 → arr[i] ≠ 'A')
  
def count_valid_arrangement : ℕ :=
  (Finset.range (5 + 1)).sum (λ j, Nat.choose 5 j * Nat.choose 5 (4 - j) * Nat.choose 5 (4 - j))

-- Theorem to be proved
theorem count_15_letter_arrangements :
  (∑ j in Finset.range (5 + 1), choose 5 j * choose 5 (4 - j) * choose 5 (4 - j)) = count_valid_arrangement :=
sorry

end count_15_letter_arrangements_l523_523335


namespace patanjali_distance_first_day_l523_523080

theorem patanjali_distance_first_day
  (h : ℕ)
  (H1 : 3 * h + 4 * (h - 1) + 4 * h = 62) :
  3 * h = 18 :=
by
  sorry

end patanjali_distance_first_day_l523_523080


namespace simplify_sqrt_power_l523_523486

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l523_523486


namespace point_on_z_axis_eq_dist_l523_523929

theorem point_on_z_axis_eq_dist (z : ℝ) : 
  let A := (1 : ℝ, -2 : ℝ, 1 : ℝ)
  let B := (2 : ℝ, 1 : ℝ, 3 : ℝ)
  let P := (0 : ℝ, 0 : ℝ, z)
  (A.1 - P.1)^2 + (A.2 - P.2)^2 + (A.3 - P.3)^2 = (B.1 - P.1)^2 + (B.2 - P.2)^2 + (B.3 - P.3)^2 → 
  P = (0, 0, 2) :=
by
  sorry

end point_on_z_axis_eq_dist_l523_523929


namespace fraction_cookies_blue_or_green_l523_523699

theorem fraction_cookies_blue_or_green (C : ℕ) (h1 : 1/C = 1/4) (h2 : 0.5555555555555556 = 5/9) :
  (1/4 + (5/9) * (3/4)) = (2/3) :=
by sorry

end fraction_cookies_blue_or_green_l523_523699


namespace arithmetic_sqrt_of_sqrt_16_l523_523566

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l523_523566


namespace arithmetic_seq_sum_l523_523981

theorem arithmetic_seq_sum (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S(3) = 9) 
  (h₂ : S(6) = 36) 
  (h₃ : ∀ n, S(n + 1) = S(n) + a(n + 1)) :
  a(7) + a(8) + a(9) = 45 :=
by
  sorry

end arithmetic_seq_sum_l523_523981


namespace quadrilateral_area_l523_523715

noncomputable def area_quadrilateral (PQ PR QS k : ℝ) (QPR : ℝ) : ℝ :=
  (1 / 2) * PQ * PR * real.sin (QPR) + (1 / 2) * PR * real.sqrt (k)

theorem quadrilateral_area :
  let PQ := 4
  let PR := 7
  let QS := 5
  let QPR := 60
  let k := PR^2 - PQ^2
  area_quadrilateral PQ PR QS QPR.to_real = 7 * real.sqrt 3 + 2 * real.sqrt 33 :=
by
  sorry

end quadrilateral_area_l523_523715


namespace smallest_n_A_n_eq_I_l523_523794

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![1/2, -Real.sqrt 3 / 2],
    ![Real.sqrt 3 / 2, 1/2]
  ]

def I : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal !![1, 1]

theorem smallest_n_A_n_eq_I :
  ∃ (n : ℕ), 0 < n ∧ (A ^ n = I) ∧ ∀ m : ℕ, (0 < m ∧ m < n) → (A ^ m ≠ I) :=
sorry

end smallest_n_A_n_eq_I_l523_523794


namespace max_number_of_girls_l523_523204

open Nat

theorem max_number_of_girls (total_schoolchildren boys_collecting_stamps ussr_stamps african_stamps american_stamps
  only_ussr_stamps only_african_stamps only_american_stamps ivanov_collects_all
  : ℕ) : total_schoolchildren = 150 ∧ boys_collecting_stamps = (only_ussr_stamps + only_african_stamps + only_american_stamps + (ussr_stamps - only_ussr_stamps + african_stamps - only_african_stamps + american_stamps - only_american_stamps) - 2 * ivanov_collects_all + ivanov_collects_all) → 
  max_number_of_girls = total_schoolchildren - boys_collecting_stamps :=
  
by
  sorry

-- Definitions according to the conditions
variable (total_schoolchildren : ℕ := 150)
variable (ussr_stamps : ℕ := 67)
variable (african_stamps : ℕ := 48)
variable (american_stamps : ℕ := 32)
variable (only_ussr_stamps : ℕ := 11)
variable (only_african_stamps : ℕ := 7)
variable (only_american_stamps : ℕ := 4)
variable (ivanov_collects_all : ℕ := 1)
variable (boys_collecting_stamps : ℕ := 84)
variable (max_number_of_girls : ℕ := 66)

end max_number_of_girls_l523_523204


namespace arithmetic_square_root_of_sqrt_16_l523_523572

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523572


namespace find_angle_B_l523_523013

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l523_523013


namespace product_increase_by_13_exists_l523_523392

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end product_increase_by_13_exists_l523_523392


namespace connect_AB_with_conditions_l523_523237

-- Definitions to assume points, equal segments, and constraints
variables (Point : Type) [Inhabited Point] [DecidableEq Point] 
variables (A B P Q R : Point)
variables (is_marked : Point → Prop)
variables (dist : Point → Point → ℝ)

-- Assumptions based on problem conditions
axiom marked_points_no_interior (p1 p2 : Point) (h : is_marked p1 ∧ is_marked p2) : (∀ (x : Point), x ≠ p1 ∧ x ≠ p2 → ¬ is_marked x)
axiom equal_segments (a b : Point) : dist A P = dist P Q ∧ dist P Q = dist Q R ∧ dist Q R = dist R B
axiom no_collinear_segments (a b c : Point) : ¬ (dist a b = dist b c ∧ dist b c = dist c d)

-- Problem Statement
theorem connect_AB_with_conditions :
  ∃ (A P Q R B : Point),
    (is_marked A) ∧ (is_marked P) ∧ (is_marked Q) ∧ (is_marked R) ∧ (is_marked B) ∧ 
    (dist A P = dist P Q ∧ dist P Q = dist Q R ∧ dist Q R = dist R B) ∧ 
    (¬ (marked_points_no_interior A P ∧ marked_points_no_interior P Q ∧ marked_points_no_interior Q R ∧ marked_points_no_interior R B)) ∧ 
    (¬ (no_collinear_segments A P Q ∧ no_collinear_segments P Q R ∧ no_collinear_segments Q R B)) :=
sorry

end connect_AB_with_conditions_l523_523237


namespace ellipse_foci_on_y_axis_l523_523890

theorem ellipse_foci_on_y_axis (m n : ℝ) (h : m > n > 0) :
  ∃ (f : ℝ) (a b : ℝ), mx^2 + ny^2 = 1 ∧ ( (f = b*a) ∧ b ≠ 0 ) ∧ ∀ x y : ℝ, 
  mx^2 + ny^2 = 1 ↔ (math.sqrt(h * y^2) + math.sqrt((1 - h) * x^2)) = 1 :=
sorry

end ellipse_foci_on_y_axis_l523_523890


namespace g_of_5_l523_523115

noncomputable def g : ℝ → ℝ := sorry

theorem g_of_5 :
  (∀ x y : ℝ, x * g y = y * g x) →
  g 20 = 30 →
  g 5 = 7.5 :=
by
  intros h1 h2
  sorry

end g_of_5_l523_523115


namespace number_four_units_away_from_neg_five_l523_523427

theorem number_four_units_away_from_neg_five (x : ℝ) : 
    abs (x + 5) = 4 ↔ x = -9 ∨ x = -1 :=
by 
  sorry

end number_four_units_away_from_neg_five_l523_523427


namespace derivative_f_at_neg_one_l523_523368

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - (1 / x)

-- State the hypothesis and conclusion for the derivative
theorem derivative_f_at_neg_one : deriv f (-1) = -1 := 
by
  -- This is where the proof would go, substitute sorry for now
  sorry

end derivative_f_at_neg_one_l523_523368


namespace solution_correct_l523_523068

variables (A B C D E F O H : Point)
variables (AC BC : ℝ)
variables (r : ℝ)
variables (ω : Circle)
variables (m n : ℕ)

-- Assume the right triangle conditions
axiom AC_length : AC = 3
axiom BC_length : BC = 4
axiom right_angle_C : ∃ (ABC : Triangle), is_right_triangle ABC

-- Assume the definition of point D as the projection from C to AB
axiom projection_D : D = projection C (Segment A B)

-- Assume circle ω with center D and radius CD
axiom circle_omega : ω = mkCircle D r
axiom radius_omega : r = distance D C

-- E is any variable point on the circumference of ω
axiom E_on_omega : OnCircle E ω

-- F is the reflection of E over D
axiom reflection_F : F = reflection E D

-- O is the center of the circumcircle of △ABE
axiom circumcircle_center_O : ∃ (circ_abe : Circle), O = circumcenter (Triangle A B E)

-- H is the orthocenter of △EFO
axiom orthocenter_H : ∃ (orth_ef_o : Point), H = orthocenter (Triangle E F O)

-- The path traced by H defines a region with area mπ/n
axiom area_R : ∃ (region : Region), area region = (144 * Real.pi) / 35

-- We need to prove that sqrt(m) + sqrt(n) = 12 + sqrt(35)
theorem solution_correct :
  (∃ m n, RelPrime m n ∧ (144 * Real.pi) / 35 = m / n * Real.pi) → 
  (Real.sqrt m + Real.sqrt n = 12 + Real.sqrt 35) := 
sorry

end solution_correct_l523_523068


namespace arithmetic_square_root_of_sqrt_16_l523_523574

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523574


namespace simplify_sqrt_seven_pow_six_proof_l523_523516

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l523_523516


namespace product_of_invertibles_mod_120_l523_523995

theorem product_of_invertibles_mod_120 (n : ℕ) (h : n = 120) : 
  let invertible_mod_n := { k | k < n ∧ Nat.coprime k n }
  let p := ∏ k in invertible_mod_n, k
  p % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l523_523995


namespace count_multiples_of_4_not_8_under_300_l523_523881

theorem count_multiples_of_4_not_8_under_300 : 
  let count := (List.range 300).filter (λ n, n > 0 ∧ n % 4 = 0 ∧ n % 8 ≠ 0) in
  count.length = 37 := 
by
  sorry

end count_multiples_of_4_not_8_under_300_l523_523881


namespace eccentricity_of_ellipse_l523_523419

-- Definitions and conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def tangency_condition (F2 : ℝ × ℝ) (b : ℝ) : Prop :=
  F2.snd = 0 ∧ ∀ E : ℝ × ℝ, E.snd = b → (sqrt ((E.fst - F2.fst)^2 + (E.snd - F2.snd)^2) = b)

def foci_distance_condition (F1 F2 : ℝ × ℝ) (c : ℝ) : Prop :=
  F1.fst = -c ∧ F2.fst = c

-- The proof statement
theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : F1 = (-c, 0)) (h5 : F2 = (c, 0))
  (h6 : ellipse_eq a b x y)
  (h7 : tangency_condition F2 b)
  (h8 : EF2 = b)
  (h9 : EF1 = 2a - b)
  (h10 : F1F2 = 2c)
  (h11 : 4c^2 = (2a - b)^2 + b^2)
  (h12 : c^2 = a^2 - b^2) :
  e = sqrt 5 / 3 := sorry

end eccentricity_of_ellipse_l523_523419


namespace area_of_rectangle_l523_523923

theorem area_of_rectangle (s : ℝ) (h1 : 4 * s = 100) : 2 * s * 2 * s = 2500 := by
  sorry

end area_of_rectangle_l523_523923


namespace simplify_sqrt_seven_pow_six_l523_523530

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l523_523530


namespace poly_remainder_l523_523277

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end poly_remainder_l523_523277


namespace expected_shots_ivan_l523_523939

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l523_523939


namespace one_eighth_of_two_pow_36_eq_two_pow_y_l523_523896

theorem one_eighth_of_two_pow_36_eq_two_pow_y (y : ℕ) : (2^36 / 8 = 2^y) → (y = 33) :=
by
  sorry

end one_eighth_of_two_pow_36_eq_two_pow_y_l523_523896


namespace length_of_BD_l523_523922

theorem length_of_BD (A B C D : Point) (h_triangle : right_triangle A B C)
  (h_perpendicular : perpendicular AD BC) (h_AB : length AB = 45)
  (h_AC : length AC = 60) : length BD = 48 :=
sorry

end length_of_BD_l523_523922


namespace max_area_quadrilateral_ACBD_l523_523155
-- Required Import

-- Statement of the Problem in Lean 4
theorem max_area_quadrilateral_ACBD (AB CD : ℝ) (h_AB : AB = 10) (h_CD : CD = 7) :
  ∃ w : ℝ, 0 ≤ w ∧ w ≤ π ∧ (∃ sin_w : ℝ, sin_w = sin w ∧ (1 / 2) * AB * CD * sin_w = 35) :=
by
  sorry

end max_area_quadrilateral_ACBD_l523_523155


namespace weight_of_B_l523_523160

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

theorem weight_of_B :
  (A + B + C) / 3 = 45 → 
  (A + B) / 2 = 40 → 
  (B + C) / 2 = 43 → 
  B = 31 :=
by
  intros h1 h2 h3
  -- detailed proof steps omitted
  sorry

end weight_of_B_l523_523160


namespace wedge_volume_is_approx_402_l523_523182

-- Definition of the radius and height of the cylinder
def radius : ℝ := 8
def height : ℝ := 8

-- Definition of π approximated to 3.14
def pi_approx : ℝ := 3.14

-- Volume of the cylinder
def volume_cylinder (r h : ℝ) : ℝ := pi * r^2 * h

-- One-fourth of the volume of the cylinder (the wedge)
def volume_wedge (r h : ℝ) : ℝ := (volume_cylinder r h) / 4

-- Approximate volume of the wedge
def volume_wedge_approx (r h pi_approx : ℝ) : ℝ := (volume_cylinder r h) / 4 * (pi_approx / pi)

-- Theorem statement: Prove that the volume of the wedge is closest to 402 cubic centimeters
theorem wedge_volume_is_approx_402 : abs (volume_wedge_approx radius height pi_approx - 402) < 1 :=
  by
    -- Proof steps go here
    sorry

end wedge_volume_is_approx_402_l523_523182


namespace Mater_costs_10_percent_of_Lightning_l523_523072

-- Conditions
def price_Lightning : ℕ := 140000
def price_Sally : ℕ := 42000
def price_Mater : ℕ := price_Sally / 3

-- The theorem we want to prove
theorem Mater_costs_10_percent_of_Lightning :
  (price_Mater * 100 / price_Lightning) = 10 := 
by 
  sorry

end Mater_costs_10_percent_of_Lightning_l523_523072


namespace find_angle_B_l523_523020

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l523_523020


namespace log_infinite_expression_pos_l523_523765

theorem log_infinite_expression_pos :
  let x := real.logb 3 (81 + real.logb 3 (81 + real.logb 3 (81 + ...)))
  in x = 4 :=
sorry

end log_infinite_expression_pos_l523_523765


namespace angle_B_in_triangle_l523_523033

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l523_523033


namespace white_truck_percentage_is_17_l523_523746

-- Define the conditions
def total_trucks : ℕ := 50
def total_cars : ℕ := 40
def total_vehicles : ℕ := total_trucks + total_cars

def red_trucks : ℕ := total_trucks / 2
def black_trucks : ℕ := (total_trucks * 20) / 100
def white_trucks : ℕ := total_trucks - red_trucks - black_trucks

def percentage_white_trucks : ℕ := (white_trucks * 100) / total_vehicles

theorem white_truck_percentage_is_17 :
  percentage_white_trucks = 17 :=
  by sorry

end white_truck_percentage_is_17_l523_523746


namespace simplify_sqrt_product_l523_523233

variable {y : ℝ}

theorem simplify_sqrt_product (hy : 0 ≤ y) :
  sqrt (48 * y) * sqrt (18 * y) * sqrt (50 * y) = 120 * y * sqrt (3 * y) := by
  sorry

end simplify_sqrt_product_l523_523233


namespace infinite_cubes_diff_3p1_infinite_cubes_diff_5q1_l523_523088

theorem infinite_cubes_diff_3p1 : 
  ∀ n : ℕ+, ∃ p : ℕ+, 3 * (n ^ 2 + n) + 1 = (n + 1) ^ 3 - n ^ 3 := 
by 
  intros n 
  use (n ^ 2 + n)
  sorry

theorem infinite_cubes_diff_5q1 :
  ∀ n : ℕ+, ∃ q : ℕ+, 5 * (15 * n ^ 2 + 3 * n) + 1 = (5 * n + 1) ^ 3 - (5 * n) ^ 3 := 
by 
  intros n 
  use (15 * n ^ 2 + 3 * n)
  sorry

end infinite_cubes_diff_3p1_infinite_cubes_diff_5q1_l523_523088


namespace cooking_time_at_least_l523_523174

-- Definitions based on conditions
def total_potatoes : ℕ := 35
def cooked_potatoes : ℕ := 11
def time_per_potato : ℕ := 7 -- in minutes
def salad_time : ℕ := 15 -- in minutes

-- The statement to prove
theorem cooking_time_at_least (oven_capacity : ℕ) :
  ∃ t : ℕ, t ≥ salad_time :=
by
  sorry

end cooking_time_at_least_l523_523174


namespace tangent_line_at_one_curve_symmetric_range_of_a_for_extreme_values_l523_523856

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 / x + a) * Real.log (1 + x)

-- Part 1
theorem tangent_line_at_one (a := -1) :
  let x := 1
  let fx := f x a
  (1 / x - 1) * (Real.log (1 + x)) = 0 → 
  y = -Real.log 2 * (x - 1) :=
by sorry

-- Part 2
noncomputable def f_inv (x : ℝ) (a : ℝ) : ℝ :=
  (x + a) * Real.log ((x + 1) / x)

theorem curve_symmetric (a := 1/2) (b := -1/2) :
  let b := -1/2
  ∃ a b : ℝ, f 1 a = (1 + 1/2) * Real.log 2 ∧ f (-2) (2 - 1/2) * Real.log 2 :=
by sorry

-- Part 3
theorem range_of_a_for_extreme_values :
  let open_interval := (0, ∞)
  0 < a ∧ a < 1/2 → 
  ∃ x ∈ open_interval, f'(x) = 0 :=
by sorry

end tangent_line_at_one_curve_symmetric_range_of_a_for_extreme_values_l523_523856


namespace simplify_sqrt7_pow6_l523_523502

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l523_523502


namespace product_of_c_with_two_real_roots_l523_523793

theorem product_of_c_with_two_real_roots :
  let q : ℕ → Prop := λ c, 9x^2 + 18x + c = 0
  let discriminant_condition : ℕ → Prop := λ c, 324 - 36c > 0
  let positive_integers_lt_9 := { n | n > 0 ∧ n < 9 }
  let product_of_values (S : set ℕ) := S.to_list.foldr (*) 1
  product_of_values positive_integers_lt_9 = 40320 :=
by {
  sorry
}

end product_of_c_with_two_real_roots_l523_523793


namespace triangle_angle_B_l523_523005

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523005


namespace youngest_child_age_l523_523694

variable (Y : ℕ) (O : ℕ) -- Y: the youngest child's present age
variable (P₀ P₁ P₂ P₃ : ℕ) -- P₀, P₁, P₂, P₃: the present ages of the 4 original family members

-- Conditions translated to Lean
variable (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
variable (h₂ : O = Y + 2)
variable (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24)

theorem youngest_child_age (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
                       (h₂ : O = Y + 2)
                       (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24) :
  Y = 3 := by 
  sorry

end youngest_child_age_l523_523694


namespace farmer_percent_gain_l523_523707

theorem farmer_percent_gain :
  ∀ (x : ℝ), x > 0 →
  let cost := 900 * x in
  let revenue_850 := 900 * x in
  let price_per_sheep := (900 * x) / 850 in
  let revenue_50 := 50 * price_per_sheep in
  let total_revenue := revenue_850 + revenue_50 in
  let profit := total_revenue - cost in
  let percent_gain := (profit / cost) * 100 in
  percent_gain = 5.88 :=
by
  intros x hx
  let cost := 900 * x
  let revenue_850 := 900 * x
  let price_per_sheep := (900 * x) / 850
  let revenue_50 := 50 * price_per_sheep
  let total_revenue := revenue_850 + revenue_50
  let profit := total_revenue - cost
  let percent_gain := (profit / cost) * 100
  have h1 : price_per_sheep = (180 / 170) * x := by sorry
  have h2 : revenue_50 = 52.9411764706 * x := by sorry
  have h3 : total_revenue = 952.9411764706 * x := by sorry
  have h4 : profit = 52.9411764706 * x := by sorry
  have h5 : percent_gain = 5.88 := by sorry
  exact h5

end farmer_percent_gain_l523_523707


namespace bottle_caps_left_l523_523404

theorem bottle_caps_left {init_caps given_away_rebecca given_away_michael left_caps : ℝ} 
  (h1 : init_caps = 143.6)
  (h2 : given_away_rebecca = 89.2)
  (h3 : given_away_michael = 16.7)
  (h4 : left_caps = init_caps - (given_away_rebecca + given_away_michael)) :
  left_caps = 37.7 := by
  sorry

end bottle_caps_left_l523_523404


namespace kitten_length_doubling_l523_523217

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l523_523217


namespace triangle_angle_B_l523_523003

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l523_523003


namespace arithmetic_square_root_of_sqrt_16_l523_523580

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l523_523580


namespace area_of_triangle_l523_523264

def point3D := (ℝ × ℝ × ℝ)

def dist (a b : point3D) : ℝ := 
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

theorem area_of_triangle : 
  let A : point3D := (1, 8, 11)
  let B : point3D := (0, 7, 7)
  let C : point3D := (-3, 10, 7)
  dist A B^2 + dist B C^2 = dist A C^2 → 
  1/2 * dist A B * dist B C = 9 := 
by 
  sorry

end area_of_triangle_l523_523264


namespace arithmetic_sqrt_sqrt_16_eq_2_l523_523603

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l523_523603
