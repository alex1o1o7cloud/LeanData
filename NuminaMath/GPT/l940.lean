import Mathlib

namespace spadesuit_proof_l940_94080

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem spadesuit_proof : 
  spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6)) = 3 :=
by
  sorry

end spadesuit_proof_l940_94080


namespace harvest_season_duration_l940_94077

theorem harvest_season_duration (weekly_rent : ℕ) (total_rent_paid : ℕ) : 
    (weekly_rent = 388) →
    (total_rent_paid = 527292) →
    (total_rent_paid / weekly_rent = 1360) :=
by
  intros h1 h2
  sorry

end harvest_season_duration_l940_94077


namespace f_odd_f_increasing_on_2_infty_solve_inequality_f_l940_94040

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = -f x := by
  sorry

theorem f_increasing_on_2_infty (x₁ x₂ : ℝ) (hx₁ : 2 < x₁) (hx₂ : 2 < x₂) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

theorem solve_inequality_f (x : ℝ) (hx : -5 < x ∧ x < -1) : f (2*x^2 + 5*x + 8) + f (x - 3 - x^2) < 0 := by
  sorry

end f_odd_f_increasing_on_2_infty_solve_inequality_f_l940_94040


namespace arithmetic_mean_of_fractions_l940_94078

def mean (a b : ℚ) : ℚ := (a + b) / 2

theorem arithmetic_mean_of_fractions (a b c : ℚ) (h₁ : a = 8/11)
                                      (h₂ : b = 5/6) (h₃ : c = 19/22) :
  mean a c = b :=
by
  sorry

end arithmetic_mean_of_fractions_l940_94078


namespace sasha_study_more_l940_94035

theorem sasha_study_more (d_wkdy : List ℤ) (d_wknd : List ℤ) (h_wkdy : d_wkdy = [5, -5, 15, 25, -15]) (h_wknd : d_wknd = [30, 30]) :
  (d_wkdy.sum + d_wknd.sum) / 7 = 12 := by
  sorry

end sasha_study_more_l940_94035


namespace smallest_positive_integer_b_l940_94064
-- Import the necessary library

-- Define the conditions and problem statement
def smallest_b_factors (r s : ℤ) := r + s

theorem smallest_positive_integer_b :
  ∃ r s : ℤ, r * s = 1800 ∧ ∀ r' s' : ℤ, r' * s' = 1800 → smallest_b_factors r s ≤ smallest_b_factors r' s' :=
by
  -- Declare that the smallest positive integer b satisfying the conditions is 85
  use 45, 40
  -- Check the core condition
  have rs_eq_1800 := (45 * 40 = 1800)
  sorry

end smallest_positive_integer_b_l940_94064


namespace total_worth_of_stock_l940_94002

theorem total_worth_of_stock (total_worth profit_fraction profit_rate loss_fraction loss_rate overall_loss : ℝ) :
  profit_fraction = 0.20 ->
  profit_rate = 0.20 -> 
  loss_fraction = 0.80 -> 
  loss_rate = 0.10 -> 
  overall_loss = 500 ->
  total_worth - (profit_fraction * total_worth * profit_rate) - (loss_fraction * total_worth * loss_rate) = overall_loss ->
  total_worth = 12500 :=
by
  sorry

end total_worth_of_stock_l940_94002


namespace point_above_line_l940_94066

-- Define the point P with coordinates (-2, t)
variable (t : ℝ)

-- Define the line equation
def line_eq (x y : ℝ) : ℝ := 2 * x - 3 * y + 6

-- Proving that t must be greater than 2/3 for the point P to be above the line
theorem point_above_line : (line_eq (-2) t < 0) -> t > 2 / 3 :=
by
  sorry

end point_above_line_l940_94066


namespace hyperbola_m_value_l940_94012

theorem hyperbola_m_value
  (m : ℝ)
  (h1 : 3 * m * x^2 - m * y^2 = 3)
  (focus : ∃ c, (0, c) = (0, 2)) :
  m = -1 :=
sorry

end hyperbola_m_value_l940_94012


namespace intersection_complement_P_CUQ_l940_94065

universe U

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}
def CUQ : Set ℕ := U \ Q

theorem intersection_complement_P_CUQ : 
  (P ∩ CUQ) = {1, 2} :=
by 
  sorry

end intersection_complement_P_CUQ_l940_94065


namespace circle_radius_l940_94075

theorem circle_radius (x y : ℝ) : x^2 - 10*x + y^2 + 4*y + 13 = 0 → ∃ r : ℝ, r = 4 :=
by
  -- sorry here to indicate that the proof is skipped
  sorry

end circle_radius_l940_94075


namespace problem1_problem2_l940_94063

theorem problem1 : |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := 
by {
  sorry
}

theorem problem2 : Real.sqrt 5 * (Real.sqrt 5 - 1 / Real.sqrt 5) = 4 := 
by {
  sorry
}

end problem1_problem2_l940_94063


namespace egyptians_panamanians_l940_94003

-- Given: n + m = 12 and (n(n-1))/2 + (m(m-1))/2 = 31 and n > m
-- Prove: n = 7 and m = 5

theorem egyptians_panamanians (n m : ℕ) (h1 : n + m = 12) (h2 : n > m) 
(h3 : n * (n - 1) / 2 + m * (m - 1) / 2 = 31) :
  n = 7 ∧ m = 5 := 
by
  sorry

end egyptians_panamanians_l940_94003


namespace unpainted_cube_count_is_correct_l940_94033

def unit_cube_count : ℕ := 6 * 6 * 6
def opposite_faces_painted_squares : ℕ := 16 * 2
def remaining_faces_painted_squares : ℕ := 9 * 4
def total_painted_squares (overlap_count : ℕ) : ℕ :=
  opposite_faces_painted_squares + remaining_faces_painted_squares - overlap_count
def overlap_count : ℕ := 4 * 2
def painted_cubes : ℕ := total_painted_squares overlap_count
def unpainted_cubes : ℕ := unit_cube_count - painted_cubes

theorem unpainted_cube_count_is_correct : unpainted_cubes = 156 := by
  sorry

end unpainted_cube_count_is_correct_l940_94033


namespace sin_beta_l940_94016

open Real

theorem sin_beta {α β : ℝ} (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_cosα : cos α = 2 * sqrt 5 / 5)
  (h_sinαβ : sin (α - β) = -3 / 5) :
  sin β = 2 * sqrt 5 / 5 := 
sorry

end sin_beta_l940_94016


namespace general_integral_of_ODE_l940_94021

noncomputable def general_solution (x y : ℝ) (m C : ℝ) : Prop :=
  (x^2 * y - x - m) / (x^2 * y - x + m) = C * Real.exp (2 * m / x)

theorem general_integral_of_ODE (m : ℝ) (y : ℝ → ℝ) (C : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∀ (y' : ℝ → ℝ) (x : ℝ), deriv y x = m^2 / x^4 - (y x)^2) ∧ 
  (y 1 = 1 / x + m / x^2) ∧ 
  (y 2 = 1 / x - m / x^2) →
  general_solution x (y x) m C :=
by 
  sorry

end general_integral_of_ODE_l940_94021


namespace percent_kindergarten_combined_l940_94061

-- Define the constants provided in the problem
def studentsPinegrove : ℕ := 150
def studentsMaplewood : ℕ := 250

def percentKindergartenPinegrove : ℝ := 18.0
def percentKindergartenMaplewood : ℝ := 14.0

-- The proof statement
theorem percent_kindergarten_combined :
  (27.0 + 35.0) / (150.0 + 250.0) * 100.0 = 15.5 :=
by 
  sorry

end percent_kindergarten_combined_l940_94061


namespace total_wet_surface_area_l940_94009

-- Necessary definitions based on conditions
def length : ℝ := 6
def width : ℝ := 4
def water_level : ℝ := 1.25

-- Defining the areas
def bottom_area : ℝ := length * width
def side_area (height : ℝ) (side_length : ℝ) : ℝ := height * side_length

-- Proof statement
theorem total_wet_surface_area :
  bottom_area + 2 * side_area water_level length + 2 * side_area water_level width = 49 := 
sorry

end total_wet_surface_area_l940_94009


namespace three_digit_condition_l940_94079

-- Define the three-digit number and its rotated variants
def num (a b c : ℕ) := 100 * a + 10 * b + c
def num_bca (a b c : ℕ) := 100 * b + 10 * c + a
def num_cab (a b c : ℕ) := 100 * c + 10 * a + b

-- The main statement to prove
theorem three_digit_condition (a b c: ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) (h_c : 0 ≤ c ∧ c ≤ 9) :
  2 * num a b c = num_bca a b c + num_cab a b c ↔ 
  (num a b c = 111 ∨ num a b c = 222 ∨ 
  num a b c = 333 ∨ num a b c = 370 ∨ 
  num a b c = 407 ∨ num a b c = 444 ∨ 
  num a b c = 481 ∨ num a b c = 518 ∨ 
  num a b c = 555 ∨ num a b c = 592 ∨ 
  num a b c = 629 ∨ num a b c = 666 ∨ 
  num a b c = 777 ∨ num a b c = 888 ∨ 
  num a b c = 999) := by
  sorry

end three_digit_condition_l940_94079


namespace gov_addresses_l940_94076

theorem gov_addresses (S H K : ℕ) 
  (H1 : S = 2 * H) 
  (H2 : K = S + 10) 
  (H3 : S + H + K = 40) : 
  S = 12 := 
sorry 

end gov_addresses_l940_94076


namespace question_inequality_l940_94068

theorem question_inequality (m : ℝ) :
  (∀ x : ℝ, ¬ (m * x ^ 2 - m * x - 1 ≥ 0)) ↔ (-4 < m ∧ m ≤ 0) :=
sorry

end question_inequality_l940_94068


namespace four_digit_number_count_l940_94025

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l940_94025


namespace sqrt_four_eq_pm_two_l940_94097

theorem sqrt_four_eq_pm_two : ∀ (x : ℝ), x * x = 4 ↔ x = 2 ∨ x = -2 := 
by
  sorry

end sqrt_four_eq_pm_two_l940_94097


namespace calculate_expr_l940_94046

theorem calculate_expr : 1 - Real.sqrt 9 = -2 := by
  sorry

end calculate_expr_l940_94046


namespace regular_octagon_interior_angle_l940_94028

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l940_94028


namespace similarity_transformation_l940_94086

theorem similarity_transformation (C C' : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : C = (4, 1))
  (h3 : C' = (r * 4, r * 1)) : (C' = (12, 3) ∨ C' = (-12, -3)) := by
  sorry

end similarity_transformation_l940_94086


namespace folded_triangle_square_length_l940_94001

theorem folded_triangle_square_length (side_length folded_distance length_squared : ℚ) 
(h1: side_length = 15) 
(h2: folded_distance = 11) 
(h3: length_squared = 1043281/31109) :
∃ (PQ : ℚ), PQ^2 = length_squared := 
by 
  sorry

end folded_triangle_square_length_l940_94001


namespace bicycle_count_l940_94038

theorem bicycle_count (T : ℕ) (B : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 :=
by {
  sorry
}

end bicycle_count_l940_94038


namespace div_by_self_condition_l940_94093

theorem div_by_self_condition (n : ℤ) (h : n^2 + 1 ∣ n) : n = 0 :=
by sorry

end div_by_self_condition_l940_94093


namespace systematic_sampling_third_group_draw_l940_94095

theorem systematic_sampling_third_group_draw
  (first_draw : ℕ) (second_draw : ℕ) (first_draw_eq : first_draw = 2)
  (second_draw_eq : second_draw = 12) :
  ∃ (third_draw : ℕ), third_draw = 22 :=
by
  sorry

end systematic_sampling_third_group_draw_l940_94095


namespace solve_for_y_l940_94023

-- Given conditions expressed as a Lean definition
def given_condition (y : ℝ) : Prop :=
  (y / 5) / 3 = 15 / (y / 3)

-- Prove the equivalent statement
theorem solve_for_y (y : ℝ) (h : given_condition y) : y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 :=
sorry

end solve_for_y_l940_94023


namespace train_crosses_platform_in_39_seconds_l940_94000

theorem train_crosses_platform_in_39_seconds :
  ∀ (length_train length_platform : ℝ) (time_cross_signal : ℝ),
  length_train = 300 →
  length_platform = 25 →
  time_cross_signal = 36 →
  ((length_train + length_platform) / (length_train / time_cross_signal)) = 39 := by
  intros length_train length_platform time_cross_signal
  intros h_length_train h_length_platform h_time_cross_signal
  rw [h_length_train, h_length_platform, h_time_cross_signal]
  sorry

end train_crosses_platform_in_39_seconds_l940_94000


namespace arithmetic_expression_l940_94037

theorem arithmetic_expression :
  (5^6) / (5^4) + 3^3 - 6^2 = 16 := by
  sorry

end arithmetic_expression_l940_94037


namespace division_result_l940_94005

theorem division_result (x : ℝ) (h : (x - 2) / 13 = 4) : (x - 5) / 7 = 7 := by
  sorry

end division_result_l940_94005


namespace total_distance_l940_94043

-- Definitions for the given problem conditions
def Beka_distance : ℕ := 873
def Jackson_distance : ℕ := 563
def Maria_distance : ℕ := 786

-- Theorem that needs to be proved
theorem total_distance : Beka_distance + Jackson_distance + Maria_distance = 2222 := by
  sorry

end total_distance_l940_94043


namespace function_properties_l940_94024

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x^2)

theorem function_properties : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y) → f x > f y) :=
by
  sorry

end function_properties_l940_94024


namespace fraction_of_2d_nails_l940_94053

theorem fraction_of_2d_nails (x : ℝ) (h1 : x + 0.5 = 0.75) : x = 0.25 :=
by
  sorry

end fraction_of_2d_nails_l940_94053


namespace area_of_fourth_rectangle_l940_94022

theorem area_of_fourth_rectangle
    (x y z w : ℝ)
    (h1 : x * y = 24)
    (h2 : z * y = 15)
    (h3 : z * w = 9) :
    y * w = 15 := 
sorry

end area_of_fourth_rectangle_l940_94022


namespace find_vector_n_l940_94054

variable (a b : ℝ)

def is_orthogonal (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def is_same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_vector_n (m n : ℝ × ℝ) (h1 : is_orthogonal m n) (h2 : is_same_magnitude m n) :
  n = (b, -a) :=
  sorry

end find_vector_n_l940_94054


namespace find_fx_for_negative_x_l940_94083

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem find_fx_for_negative_x (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_given : given_function f) :
  ∀ x, (x < 0) → f x = x + x^2 :=
by
  sorry

end find_fx_for_negative_x_l940_94083


namespace minimum_moves_to_determine_polynomial_l940_94042

-- Define quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define conditions as per the given problem
variables {f g : ℝ → ℝ}
def is_quadratic (p : ℝ → ℝ) := ∃ a b c : ℝ, ∀ x : ℝ, p x = quadratic a b c x

axiom f_is_quadratic : is_quadratic f
axiom g_is_quadratic : is_quadratic g

-- Define the main problem statement
theorem minimum_moves_to_determine_polynomial (n : ℕ) :
  (∀ (t : ℕ → ℝ), (∀ m ≤ n, (f (t m) = g (t m)) ∨ (f (t m) ≠ g (t m))) →
  (∃ a b c: ℝ, ∀ x: ℝ, f x = quadratic a b c x ∨ g x = quadratic a b c x)) ↔ n = 8 :=
sorry -- Proof is omitted

end minimum_moves_to_determine_polynomial_l940_94042


namespace number_of_orders_l940_94090

open Nat

theorem number_of_orders (total_targets : ℕ) (targets_A : ℕ) (targets_B : ℕ) (targets_C : ℕ)
  (h1 : total_targets = 10)
  (h2 : targets_A = 4)
  (h3 : targets_B = 3)
  (h4 : targets_C = 3)
  : total_orders = 80 :=
sorry

end number_of_orders_l940_94090


namespace final_result_is_110_l940_94062

def chosen_number : ℕ := 63
def multiplier : ℕ := 4
def subtracted_value : ℕ := 142

def final_result : ℕ := (chosen_number * multiplier) - subtracted_value

theorem final_result_is_110 : final_result = 110 := by
  sorry

end final_result_is_110_l940_94062


namespace min_odd_integers_l940_94096

theorem min_odd_integers (a b c d e f g h i : ℤ)
  (h1 : a + b + c = 30)
  (h2 : a + b + c + d + e + f = 48)
  (h3 : a + b + c + d + e + f + g + h + i = 69) :
  ∃ k : ℕ, k = 1 ∧
  (∃ (aa bb cc dd ee ff gg hh ii : ℤ), (fun (x : ℤ) => x % 2 = 1 → k = 1) (aa + bb + cc + dd + ee + ff + gg + hh + ii)) :=
by
  intros
  sorry

end min_odd_integers_l940_94096


namespace arithmetic_geo_sum_l940_94047

theorem arithmetic_geo_sum (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →
  (d = 2) →
  (a 3) ^ 2 = (a 1) * (a 4) →
  (a 2 + a 3 = -10) := 
by
  intros h_arith h_d h_geo
  sorry

end arithmetic_geo_sum_l940_94047


namespace increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l940_94027

-- Definitions for conditions
def original_profit_per_set := 40
def original_sets_sold_per_day := 20
def additional_sets_per_dollar_drop := 2

-- The proof problems

-- Part 1: Prove the increase in sets when profit reduces to $38
theorem increase_in_sets_when_profit_38 :
  let decrease_in_profit := (original_profit_per_set - 38)
  additional_sets_per_dollar_drop * decrease_in_profit = 4 :=
by
  sorry

-- Part 2: Prove the price reduction needed for $1200 daily profit
theorem price_reduction_for_1200_profit :
  ∃ x, (original_profit_per_set - x) * (original_sets_sold_per_day + 2 * x) = 1200 ∧ x = 20 :=
by
  sorry

end increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l940_94027


namespace percentage_increase_each_year_is_50_l940_94052

-- Definitions based on conditions
def students_passed_three_years_ago : ℕ := 200
def students_passed_this_year : ℕ := 675

-- The prove statement
theorem percentage_increase_each_year_is_50
    (N3 N0 : ℕ)
    (P : ℚ)
    (h1 : N3 = students_passed_three_years_ago)
    (h2 : N0 = students_passed_this_year)
    (h3 : N0 = N3 * (1 + P)^3) :
  P = 0.5 :=
by
  sorry

end percentage_increase_each_year_is_50_l940_94052


namespace quadratic_factorization_b_value_l940_94058

theorem quadratic_factorization_b_value (b : ℤ) (c d e f : ℤ) (h1 : 24 * c + 24 * d = 240) :
  (24 * (c * e) + b + 24) = 0 →
  (c * e = 24) →
  (c * f + d * e = b) →
  (d * f = 24) →
  (c + d = 10) →
  b = 52 :=
by
  intros
  sorry

end quadratic_factorization_b_value_l940_94058


namespace number_of_cows_l940_94056

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def chicken_cost : ℕ := 100 * 5
def installation_cost : ℕ := 6 * 100
def equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

theorem number_of_cows : 
  (total_cost - (land_cost + house_cost + chicken_cost + installation_cost + equipment_cost)) / 1000 = 20 := by
  sorry

end number_of_cows_l940_94056


namespace simplify_fraction_l940_94050

variable (x y : ℕ)

theorem simplify_fraction (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 :=
by
  sorry

end simplify_fraction_l940_94050


namespace range_of_a_plus_b_l940_94048

variable {a b : ℝ}

-- Assumptions
def are_positive_and_unequal (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b
def equation_holds (a b : ℝ) : Prop := a^2 - a + b^2 - b + a * b = 0

-- Problem Statement
theorem range_of_a_plus_b (h₁ : are_positive_and_unequal a b) (h₂ : equation_holds a b) : 1 < a + b ∧ a + b < 4 / 3 :=
sorry

end range_of_a_plus_b_l940_94048


namespace present_age_of_son_l940_94010

theorem present_age_of_son (S F : ℕ)
  (h1 : F = S + 24)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  -- The proof is omitted, as per instructions.
  sorry
}

end present_age_of_son_l940_94010


namespace circle_center_radius_l940_94074

theorem circle_center_radius (x y : ℝ) :
  (x - 1)^2 + (y - 3)^2 = 4 → (1, 3) = (1, 3) ∧ 2 = 2 :=
by
  intro h
  exact ⟨rfl, rfl⟩

end circle_center_radius_l940_94074


namespace frank_is_15_years_younger_than_john_l940_94049

variables (F J : ℕ)

theorem frank_is_15_years_younger_than_john
  (h1 : J + 3 = 2 * (F + 3))
  (h2 : F + 4 = 16) : J - F = 15 := by
  sorry

end frank_is_15_years_younger_than_john_l940_94049


namespace train_speed_l940_94067

theorem train_speed (l t: ℝ) (h1: l = 441) (h2: t = 21) : l / t = 21 := by
  sorry

end train_speed_l940_94067


namespace english_vocab_related_to_reading_level_l940_94085

theorem english_vocab_related_to_reading_level (N : ℕ) (K_squared : ℝ) (critical_value : ℝ) (p_value : ℝ)
  (hN : N = 100)
  (hK_squared : K_squared = 7)
  (h_critical_value : critical_value = 6.635)
  (h_p_value : p_value = 0.010) :
  p_value <= 0.01 → K_squared > critical_value → true :=
by
  intro h_p_value_le h_K_squared_gt
  sorry

end english_vocab_related_to_reading_level_l940_94085


namespace pencil_and_pen_choice_count_l940_94082

-- Definitions based on the given conditions
def numPencilTypes : Nat := 4
def numPenTypes : Nat := 6

-- Statement we want to prove
theorem pencil_and_pen_choice_count : (numPencilTypes * numPenTypes) = 24 :=
by
  sorry

end pencil_and_pen_choice_count_l940_94082


namespace decrease_percent_in_revenue_l940_94045

theorem decrease_percent_in_revenue 
  (T C : ℝ) 
  (original_revenue : ℝ := T * C)
  (new_tax : ℝ := 0.80 * T)
  (new_consumption : ℝ := 1.15 * C)
  (new_revenue : ℝ := new_tax * new_consumption) :
  ((original_revenue - new_revenue) / original_revenue) * 100 = 8 := 
sorry

end decrease_percent_in_revenue_l940_94045


namespace dhoni_remaining_earnings_l940_94070

theorem dhoni_remaining_earnings (rent_percent dishwasher_percent : ℝ) 
  (h1 : rent_percent = 20) (h2 : dishwasher_percent = 15) : 
  100 - (rent_percent + dishwasher_percent) = 65 := 
by 
  sorry

end dhoni_remaining_earnings_l940_94070


namespace slices_in_loaf_initial_l940_94089

-- Define the total slices used from Monday to Friday
def slices_used_weekdays : Nat := 5 * 2

-- Define the total slices used on Saturday
def slices_used_saturday : Nat := 2 * 2

-- Define the total slices used in the week
def total_slices_used : Nat := slices_used_weekdays + slices_used_saturday

-- Define the slices left
def slices_left : Nat := 6

-- Prove the total slices Tony started with
theorem slices_in_loaf_initial :
  let slices := total_slices_used + slices_left
  slices = 20 :=
by
  sorry

end slices_in_loaf_initial_l940_94089


namespace train_B_time_to_reach_destination_l940_94039

theorem train_B_time_to_reach_destination
    (T t : ℝ)
    (train_A_speed : ℝ) (train_B_speed : ℝ)
    (train_A_extra_hours : ℝ)
    (h1 : train_A_speed = 110)
    (h2 : train_B_speed = 165)
    (h3 : train_A_extra_hours = 9)
    (h_eq : 110 * (T + train_A_extra_hours) = 110 * T + 165 * t) :
    t = 6 := 
by
  sorry

end train_B_time_to_reach_destination_l940_94039


namespace man_l940_94032

theorem man's_rate_in_still_water (speed_with_stream speed_against_stream : ℝ) (h1 : speed_with_stream = 26) (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 19 := 
by
  rw [h1, h2]
  norm_num

end man_l940_94032


namespace value_of_a3_l940_94098

def a_n (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem value_of_a3 : a_n 3 = -10 :=
by
  -- The proof would go here.
  sorry

end value_of_a3_l940_94098


namespace proof_problem_l940_94013

theorem proof_problem 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) : 
  |a / b + b / a| ≥ 2 := 
sorry

end proof_problem_l940_94013


namespace smallest_value_other_integer_l940_94008

noncomputable def smallest_possible_value_b : ℕ :=
  by sorry

theorem smallest_value_other_integer (x : ℕ) (h_pos : x > 0) (b : ℕ) 
  (h_gcd : Nat.gcd 36 b = x + 3) (h_lcm : Nat.lcm 36 b = x * (x + 3)) :
  b = 108 :=
  by sorry

end smallest_value_other_integer_l940_94008


namespace evaluate_polynomial_l940_94036

variable {x y : ℚ}

theorem evaluate_polynomial (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 :=
by
  sorry

end evaluate_polynomial_l940_94036


namespace triangle_area_l940_94044

open Real

-- Define the conditions
variables (a : ℝ) (B : ℝ) (cosA : ℝ)
variable (S : ℝ)

-- Given conditions of the problem
def triangle_conditions : Prop :=
  a = 5 ∧ B = π / 3 ∧ cosA = 11 / 14

-- State the theorem to be proved
theorem triangle_area (h : triangle_conditions a B cosA) : S = 10 * sqrt 3 :=
sorry

end triangle_area_l940_94044


namespace number_of_boys_exceeds_girls_by_l940_94051

theorem number_of_boys_exceeds_girls_by (girls boys: ℕ) (h1: girls = 34) (h2: boys = 841) : boys - girls = 807 := by
  sorry

end number_of_boys_exceeds_girls_by_l940_94051


namespace production_days_l940_94060

theorem production_days (n P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 65) : n = 5 := sorry

end production_days_l940_94060


namespace trig_identity_l940_94014

theorem trig_identity (x : ℝ) (h : Real.cos (x - π / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * π / 3) + Real.sin (π / 3 - x)^2 = 5 / 3 :=
by
  sorry

end trig_identity_l940_94014


namespace find_smallest_x_l940_94088

noncomputable def smallest_pos_real_x : ℝ :=
  55 / 7

theorem find_smallest_x (x : ℝ) (h : x > 0) (hx : ⌊x^2⌋ - x * ⌊x⌋ = 6) : x = smallest_pos_real_x :=
  sorry

end find_smallest_x_l940_94088


namespace find_x_l940_94069

theorem find_x (x : ℕ) (h_odd : x % 2 = 1) (h_pos : 0 < x) :
  (∃ l : List ℕ, l.length = 8 ∧ (∀ n ∈ l, n < 80 ∧ n % 2 = 1) ∧ l.Nodup = true ∧
  (∀ k m, k > 0 → m % 2 = 1 → k * x * m ∈ l)) → x = 5 := by
  sorry

end find_x_l940_94069


namespace max_blue_cubes_visible_l940_94087

def max_visible_blue_cubes (board : ℕ × ℕ × ℕ → ℕ) : ℕ :=
  board (0, 0, 0)

theorem max_blue_cubes_visible (board : ℕ × ℕ × ℕ → ℕ) :
  max_visible_blue_cubes board = 12 :=
sorry

end max_blue_cubes_visible_l940_94087


namespace volume_of_prism_l940_94091

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) :
    a * b * c = 426 :=
sorry

end volume_of_prism_l940_94091


namespace translate_quadratic_function_l940_94020

theorem translate_quadratic_function :
  ∀ x : ℝ, (y = (1 / 3) * x^2) →
          (y₂ = (1 / 3) * (x - 1)^2) →
          (y₃ = y₂ + 3) →
          y₃ = (1 / 3) * (x - 1)^2 + 3 := 
by 
  intros x h₁ h₂ h₃ 
  sorry

end translate_quadratic_function_l940_94020


namespace count_integer_values_not_satisfying_inequality_l940_94055

theorem count_integer_values_not_satisfying_inequality : 
  ∃ n : ℕ, 
  (n = 3) ∧ (∀ x : ℤ, (4 * x^2 + 22 * x + 21 ≤ 25) → (-2 ≤ x ∧ x ≤ 0)) :=
by
  sorry

end count_integer_values_not_satisfying_inequality_l940_94055


namespace incorrect_calculation_l940_94006

theorem incorrect_calculation :
    (5 / 8 + (-7 / 12) ≠ -1 / 24) :=
by
  sorry

end incorrect_calculation_l940_94006


namespace zero_positive_integers_prime_polynomial_l940_94081

noncomputable def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem zero_positive_integers_prime_polynomial :
  ∀ (n : ℕ), ¬ is_prime (n^3 - 7 * n^2 + 16 * n - 12) :=
by
  sorry

end zero_positive_integers_prime_polynomial_l940_94081


namespace exists_n_for_m_l940_94034

def π (x : ℕ) : ℕ := sorry -- Placeholder for the prime counting function

theorem exists_n_for_m (m : ℕ) (hm : m > 1) : ∃ n : ℕ, n > 1 ∧ n / π n = m :=
by sorry

end exists_n_for_m_l940_94034


namespace fred_games_this_year_l940_94092

variable (last_year_games : ℕ)
variable (difference : ℕ)

theorem fred_games_this_year (h1 : last_year_games = 36) (h2 : difference = 11) : 
  last_year_games - difference = 25 := 
by
  sorry

end fred_games_this_year_l940_94092


namespace matrix_problem_l940_94099

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)
variable (I : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = !![2, 1; 4, 3]) :
  B * A = !![2, 1; 4, 3] :=
sorry

end matrix_problem_l940_94099


namespace total_seeds_in_garden_l940_94041

-- Definitions based on the conditions
def top_bed_rows : ℕ := 4
def top_bed_seeds_per_row : ℕ := 25
def num_top_beds : ℕ := 2

def medium_bed_rows : ℕ := 3
def medium_bed_seeds_per_row : ℕ := 20
def num_medium_beds : ℕ := 2

-- Calculation of total seeds in top beds
def seeds_per_top_bed : ℕ := top_bed_rows * top_bed_seeds_per_row
def total_seeds_top_beds : ℕ := num_top_beds * seeds_per_top_bed

-- Calculation of total seeds in medium beds
def seeds_per_medium_bed : ℕ := medium_bed_rows * medium_bed_seeds_per_row
def total_seeds_medium_beds : ℕ := num_medium_beds * seeds_per_medium_bed

-- Proof goal
theorem total_seeds_in_garden : total_seeds_top_beds + total_seeds_medium_beds = 320 :=
by
  sorry

end total_seeds_in_garden_l940_94041


namespace quadratic_distinct_roots_l940_94026

theorem quadratic_distinct_roots (p q₁ q₂ : ℝ) 
  (h_eq : p = q₁ + q₂ + 1) :
  q₁ ≥ 1/4 → 
  (∃ x, x^2 + x + q₁ = 0 ∧ ∃ x', x' ≠ x ∧ x'^2 + x' + q₁ = 0) 
  ∨ 
  (∃ y, y^2 + p*y + q₂ = 0 ∧ ∃ y', y' ≠ y ∧ y'^2 + p*y' + q₂ = 0) :=
by 
  sorry

end quadratic_distinct_roots_l940_94026


namespace students_taking_art_l940_94073

def total_students := 500
def students_taking_music := 40
def students_taking_both := 10
def students_taking_neither := 450

theorem students_taking_art : ∃ A, total_students = students_taking_music - students_taking_both + (A - students_taking_both) + students_taking_both + students_taking_neither ∧ A = 20 :=
by
  sorry

end students_taking_art_l940_94073


namespace complex_numbers_xyz_l940_94015

theorem complex_numbers_xyz (x y z : ℂ) (h1 : x * y + 5 * y = -20) (h2 : y * z + 5 * z = -20) (h3 : z * x + 5 * x = -20) :
  x * y * z = 100 :=
sorry

end complex_numbers_xyz_l940_94015


namespace border_area_l940_94030

theorem border_area (photo_height photo_width border_width : ℕ) (h1 : photo_height = 12) (h2 : photo_width = 16) (h3 : border_width = 3) : 
  let framed_height := photo_height + 2 * border_width 
  let framed_width := photo_width + 2 * border_width 
  let area_of_photo := photo_height * photo_width
  let area_of_framed := framed_height * framed_width 
  let area_of_border := area_of_framed - area_of_photo 
  area_of_border = 204 := 
by
  sorry

end border_area_l940_94030


namespace seafood_noodles_l940_94011

theorem seafood_noodles (total_plates lobster_rolls spicy_hot_noodles : ℕ)
  (h_total : total_plates = 55)
  (h_lobster : lobster_rolls = 25)
  (h_spicy : spicy_hot_noodles = 14) :
  total_plates - (lobster_rolls + spicy_hot_noodles) = 16 :=
by
  sorry

end seafood_noodles_l940_94011


namespace moderate_intensity_pushups_l940_94094

theorem moderate_intensity_pushups :
  let normal_heart_rate := 80
  let k := 7
  let y (x : ℕ) := 80 * (Real.log (Real.sqrt (x / 12)) + 1)
  let t (x : ℕ) := y x / normal_heart_rate
  let f (t : ℝ) := k * Real.exp t
  28 ≤ f (Real.log (Real.sqrt 3)) + 1 ∧ f (Real.log (Real.sqrt 3)) + 1 ≤ 34 :=
sorry

end moderate_intensity_pushups_l940_94094


namespace sum_positive_implies_at_least_one_positive_l940_94019

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l940_94019


namespace intersecting_diagonals_of_parallelogram_l940_94059

theorem intersecting_diagonals_of_parallelogram (A C : ℝ × ℝ) (hA : A = (2, -3)) (hC : C = (14, 9)) :
    ∃ M : ℝ × ℝ, M = (8, 3) ∧ M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) :=
by {
  sorry
}

end intersecting_diagonals_of_parallelogram_l940_94059


namespace total_amount_spent_l940_94057

-- Define the variables B and D representing the amounts Ben and David spent.
variables (B D : ℝ)

-- Define the conditions based on the given problem.
def conditions : Prop :=
  (D = 0.60 * B) ∧ (B = D + 14)

-- The main theorem stating the total amount spent by Ben and David is 56.
theorem total_amount_spent (h : conditions B D) : B + D = 56 :=
sorry  -- Proof omitted.

end total_amount_spent_l940_94057


namespace bowl_capacity_l940_94018

theorem bowl_capacity (C : ℝ) (h1 : (2/3) * C * 5 + (1/3) * C * 4 = 700) : C = 150 := 
by
  sorry

end bowl_capacity_l940_94018


namespace discount_rate_pony_jeans_l940_94029

theorem discount_rate_pony_jeans
  (fox_price pony_price : ℕ)
  (fox_pairs pony_pairs : ℕ)
  (total_savings total_discount_rate : ℕ)
  (F P : ℕ)
  (h1 : fox_price = 15)
  (h2 : pony_price = 20)
  (h3 : fox_pairs = 3)
  (h4 : pony_pairs = 2)
  (h5 : total_savings = 9)
  (h6 : total_discount_rate = 22)
  (h7 : F + P = total_discount_rate)
  (h8 : fox_pairs * fox_price * F / 100 + pony_pairs * pony_price * P / 100 = total_savings) : 
  P = 18 :=
sorry

end discount_rate_pony_jeans_l940_94029


namespace kiana_siblings_ages_l940_94017

/-- Kiana has two twin brothers, one is twice as old as the other, 
and their ages along with Kiana's age multiply to 72. Prove that 
the sum of their ages is 13. -/
theorem kiana_siblings_ages
  (y : ℕ) (K : ℕ) (h1 : 2 * y * K = 72) :
  y + 2 * y + K = 13 := 
sorry

end kiana_siblings_ages_l940_94017


namespace unique_real_solution_l940_94004

theorem unique_real_solution :
  ∀ x : ℝ, (x > 0 → (x ^ 16 + 1) * (x ^ 12 + x ^ 8 + x ^ 4 + 1) = 18 * x ^ 8 → x = 1) :=
by
  introv
  sorry

end unique_real_solution_l940_94004


namespace arithmetic_sequence_general_geometric_sequence_sum_l940_94007

theorem arithmetic_sequence_general (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_a3 : a 3 = -6) 
  (h_a6 : a 6 = 0) :
  ∀ n, a n = 2 * n - 12 := 
sorry

theorem geometric_sequence_sum (a b : ℕ → ℤ) 
  (r : ℤ) 
  (S : ℕ → ℤ)
  (h_geom : ∀ n : ℕ, b (n + 1) = b n * r) 
  (h_b1 : b 1 = -8) 
  (h_b2 : b 2 = a 0 + a 1 + a 2) 
  (h_a1 : a 0 = -10) 
  (h_a2 : a 1 = -8) 
  (h_a3 : a 2 = -6) :
  ∀ n, S n = 4 * (1 - 3 ^ n) := 
sorry

end arithmetic_sequence_general_geometric_sequence_sum_l940_94007


namespace sum_of_roots_eq_6_l940_94031

theorem sum_of_roots_eq_6 : ∀ (x1 x2 : ℝ), (x1 * x1 = x1 ∧ x1 * x2 = x2) → (x1 + x2 = 6) :=
by
   intro x1 x2 hx
   have H : x1 + x2 = 6 := sorry
   exact H

end sum_of_roots_eq_6_l940_94031


namespace base_of_isosceles_triangle_l940_94084

namespace TriangleProblem

def equilateral_triangle_perimeter (s : ℕ) : ℕ := 3 * s
def isosceles_triangle_perimeter (s b : ℕ) : ℕ := 2 * s + b

theorem base_of_isosceles_triangle (s b : ℕ) (h1 : equilateral_triangle_perimeter s = 45) 
    (h2 : isosceles_triangle_perimeter s b = 40) : b = 10 :=
by
  sorry

end TriangleProblem

end base_of_isosceles_triangle_l940_94084


namespace invalid_root_l940_94071

theorem invalid_root (a_1 a_0 : ℤ) : ¬(19 * (1/7 : ℚ)^3 + 98 * (1/7 : ℚ)^2 + a_1 * (1/7 : ℚ) + a_0 = 0) :=
by 
  sorry

end invalid_root_l940_94071


namespace complement_intersection_l940_94072

noncomputable def real_universal_set : Set ℝ := Set.univ

noncomputable def set_A (x : ℝ) : Prop := x + 1 < 0
def A : Set ℝ := {x | set_A x}

noncomputable def set_B (x : ℝ) : Prop := x - 3 < 0
def B : Set ℝ := {x | set_B x}

noncomputable def complement_A : Set ℝ := {x | ¬set_A x}

noncomputable def intersection (S₁ S₂ : Set ℝ) : Set ℝ := {x | x ∈ S₁ ∧ x ∈ S₂}

theorem complement_intersection :
  intersection complement_A B = {x | -1 ≤ x ∧ x < 3} :=
sorry

end complement_intersection_l940_94072
