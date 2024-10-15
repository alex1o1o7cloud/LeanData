import Mathlib

namespace NUMINAMATH_GPT_find_other_endpoint_l897_89717

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ)
  (h_midpoint_x : x_m = (x_1 + x_2) / 2)
  (h_midpoint_y : y_m = (y_1 + y_2) / 2)
  (h_given_midpoint : x_m = 3 ∧ y_m = 0)
  (h_given_endpoint1 : x_1 = 7 ∧ y_1 = -4) :
  x_2 = -1 ∧ y_2 = 4 :=
sorry

end NUMINAMATH_GPT_find_other_endpoint_l897_89717


namespace NUMINAMATH_GPT_exists_c_d_rel_prime_l897_89798

theorem exists_c_d_rel_prime (a b : ℤ) :
  ∃ c d : ℤ, ∀ n : ℤ, gcd (a * n + c) (b * n + d) = 1 :=
sorry

end NUMINAMATH_GPT_exists_c_d_rel_prime_l897_89798


namespace NUMINAMATH_GPT_value_of_x_l897_89710

theorem value_of_x (p q r x : ℝ)
  (h1 : p = 72)
  (h2 : q = 18)
  (h3 : r = 108)
  (h4 : x = 180 - (q + r)) : 
  x = 54 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l897_89710


namespace NUMINAMATH_GPT_distributor_cost_l897_89725

variable (C : ℝ) -- Cost of the item for the distributor
variable (P_observed : ℝ) -- Observed price
variable (commission_rate : ℝ) -- Commission rate
variable (profit_rate : ℝ) -- Desired profit rate

-- Conditions
def is_observed_price_correct (C : ℝ) (P_observed : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) : Prop :=
  let SP := C * (1 + profit_rate)
  let observed := SP * (1 - commission_rate)
  observed = P_observed

-- The proof goal
theorem distributor_cost (h : is_observed_price_correct C 30 0.20 0.20) : C = 31.25 := sorry

end NUMINAMATH_GPT_distributor_cost_l897_89725


namespace NUMINAMATH_GPT_linear_dependent_vectors_l897_89733

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_linear_dependent_vectors_l897_89733


namespace NUMINAMATH_GPT_rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l897_89791

theorem rectangles_on_8x8_chessboard : 
  (Nat.choose 9 2) * (Nat.choose 9 2) = 1296 := by
  sorry

theorem rectangles_on_nxn_chessboard (n : ℕ) : 
  (Nat.choose (n + 1) 2) * (Nat.choose (n + 1) 2) = (n * (n + 1) / 2) * (n * (n + 1) / 2) := by 
  sorry

end NUMINAMATH_GPT_rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l897_89791


namespace NUMINAMATH_GPT_valid_raise_percentage_l897_89734

-- Define the conditions
def raise_between (x : ℝ) : Prop :=
  0.05 ≤ x ∧ x ≤ 0.10

def salary_increase_by_fraction (x : ℝ) : Prop :=
  x = 0.06

-- Define the main theorem 
theorem valid_raise_percentage (x : ℝ) (hx_between : raise_between x) (hx_fraction : salary_increase_by_fraction x) :
  x = 0.06 :=
sorry

end NUMINAMATH_GPT_valid_raise_percentage_l897_89734


namespace NUMINAMATH_GPT_find_side_length_of_cube_l897_89796

theorem find_side_length_of_cube (n : ℕ) :
  (4 * n^2 = (1/3) * 6 * n^3) -> n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_side_length_of_cube_l897_89796


namespace NUMINAMATH_GPT_not_divisible_l897_89706

theorem not_divisible {x y : ℕ} (hx : x > 0) (hy : y > 2) : ¬ (2^y - 1) ∣ (2^x + 1) := sorry

end NUMINAMATH_GPT_not_divisible_l897_89706


namespace NUMINAMATH_GPT_girls_ran_miles_l897_89729

def boys_laps : ℕ := 34
def extra_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6
def girls_laps : ℕ := boys_laps + extra_laps

theorem girls_ran_miles : girls_laps * lap_distance = 9 := 
by 
  sorry

end NUMINAMATH_GPT_girls_ran_miles_l897_89729


namespace NUMINAMATH_GPT_value_of_expression_l897_89793

theorem value_of_expression :
  (43 + 15)^2 - (43^2 + 15^2) = 2 * 43 * 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l897_89793


namespace NUMINAMATH_GPT_imo_inequality_l897_89773

variable {a b c : ℝ}

theorem imo_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : (a + b) * (b + c) * (c + a) = 1) :
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ (1 / 2) := 
sorry

end NUMINAMATH_GPT_imo_inequality_l897_89773


namespace NUMINAMATH_GPT_abs_h_eq_1_div_2_l897_89720

theorem abs_h_eq_1_div_2 {h : ℝ} 
  (h_sum_sq_roots : ∀ (r s : ℝ), (r + s) = 4 * h ∧ (r * s) = -8 → (r ^ 2 + s ^ 2) = 20) : 
  |h| = 1 / 2 :=
sorry

end NUMINAMATH_GPT_abs_h_eq_1_div_2_l897_89720


namespace NUMINAMATH_GPT_five_times_x_plus_four_l897_89712

theorem five_times_x_plus_four (x : ℝ) (h : 4 * x - 3 = 13 * x + 12) : 5 * (x + 4) = 35 / 3 := 
by
  sorry

end NUMINAMATH_GPT_five_times_x_plus_four_l897_89712


namespace NUMINAMATH_GPT_students_suggested_tomatoes_79_l897_89722

theorem students_suggested_tomatoes_79 (T : ℕ)
  (mashed_potatoes : ℕ)
  (h1 : mashed_potatoes = 144)
  (h2 : mashed_potatoes = T + 65) :
  T = 79 :=
by {
  -- Proof steps will go here
  sorry
}

end NUMINAMATH_GPT_students_suggested_tomatoes_79_l897_89722


namespace NUMINAMATH_GPT_length_and_width_of_prism_l897_89784

theorem length_and_width_of_prism (w l h d : ℝ) (h_cond : h = 12) (d_cond : d = 15) (length_cond : l = 3 * w) :
  (w = 3) ∧ (l = 9) :=
by
  -- The proof is omitted as instructed in the task description.
  sorry

end NUMINAMATH_GPT_length_and_width_of_prism_l897_89784


namespace NUMINAMATH_GPT_andy_max_cookies_l897_89776

theorem andy_max_cookies (total_cookies : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ)
  (h1 : total_cookies = 30)
  (h2 : bella_cookies = 2 * andy_cookies)
  (h3 : andy_cookies + bella_cookies = total_cookies) :
  andy_cookies = 10 := by
  sorry

end NUMINAMATH_GPT_andy_max_cookies_l897_89776


namespace NUMINAMATH_GPT_quadrilateral_diagonal_areas_relation_l897_89718

-- Defining the areas of the four triangles and the quadrilateral
variables (A B C D Q : ℝ)

-- Stating the property to be proven
theorem quadrilateral_diagonal_areas_relation 
  (H1 : Q = A + B + C + D) :
  A * B * C * D = ((A + B) * (B + C) * (C + D) * (D + A))^2 / Q^4 :=
by sorry

end NUMINAMATH_GPT_quadrilateral_diagonal_areas_relation_l897_89718


namespace NUMINAMATH_GPT_completing_square_eq_sum_l897_89762

theorem completing_square_eq_sum :
  ∃ (a b c : ℤ), a > 0 ∧ (∀ (x : ℝ), 36 * x^2 - 60 * x + 25 = (a * x + b)^2 - c) ∧ a + b + c = 26 :=
by
  sorry

end NUMINAMATH_GPT_completing_square_eq_sum_l897_89762


namespace NUMINAMATH_GPT_coconut_grove_problem_l897_89723

variable (x : ℝ)

-- Conditions
def trees_yield_40_nuts_per_year : ℝ := 40 * (x + 2)
def trees_yield_120_nuts_per_year : ℝ := 120 * x
def trees_yield_180_nuts_per_year : ℝ := 180 * (x - 2)
def average_yield_per_tree_per_year : ℝ := 100

-- Problem Statement
theorem coconut_grove_problem
  (yield_40_trees : trees_yield_40_nuts_per_year x = 40 * (x + 2))
  (yield_120_trees : trees_yield_120_nuts_per_year x = 120 * x)
  (yield_180_trees : trees_yield_180_nuts_per_year x = 180 * (x - 2))
  (average_yield : average_yield_per_tree_per_year = 100) :
  x = 7 :=
by
  sorry

end NUMINAMATH_GPT_coconut_grove_problem_l897_89723


namespace NUMINAMATH_GPT_find_m_p_pairs_l897_89716

theorem find_m_p_pairs (m p : ℕ) (h_prime : Nat.Prime p) (h_eq : ∃ (x : ℕ), 2^m * p^2 + 27 = x^3) :
  (m, p) = (1, 7) :=
sorry

end NUMINAMATH_GPT_find_m_p_pairs_l897_89716


namespace NUMINAMATH_GPT_cistern_problem_l897_89779

noncomputable def cistern_problem_statement : Prop :=
∀ (x : ℝ),
  (1 / 5 - 1 / x = 1 / 11.25) → x = 9

theorem cistern_problem : cistern_problem_statement :=
sorry

end NUMINAMATH_GPT_cistern_problem_l897_89779


namespace NUMINAMATH_GPT_proof_solution_l897_89795

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  0 < x ∧ 7.61 * log x / log 2 + 2 * log x / log 4 = x ^ (log 16 / log 3 / log x / log 9)

theorem proof_solution : proof_problem (16 / 3) :=
by
  sorry

end NUMINAMATH_GPT_proof_solution_l897_89795


namespace NUMINAMATH_GPT_clock_820_angle_is_130_degrees_l897_89738

def angle_at_8_20 : ℝ :=
  let degrees_per_hour := 30.0
  let degrees_per_minute_hour_hand := 0.5
  let num_hour_sections := 4.0
  let minutes := 20.0
  let hour_angle := num_hour_sections * degrees_per_hour
  let minute_addition := minutes * degrees_per_minute_hour_hand
  hour_angle + minute_addition

theorem clock_820_angle_is_130_degrees :
  angle_at_8_20 = 130 :=
by
  sorry

end NUMINAMATH_GPT_clock_820_angle_is_130_degrees_l897_89738


namespace NUMINAMATH_GPT_symmetric_circle_l897_89740

variable (x y : ℝ)

def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem symmetric_circle :
  (∃ x y, original_circle x y) → (x^2 + (y + 2)^2 = 5) :=
sorry

end NUMINAMATH_GPT_symmetric_circle_l897_89740


namespace NUMINAMATH_GPT_wood_length_equation_l897_89766

-- Define the conditions as hypotheses
def length_of_wood_problem (x : ℝ) :=
  (1 / 2) * (x + 4.5) = x - 1

-- Now we state the theorem we want to prove, which is equivalent to the question == answer
theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end NUMINAMATH_GPT_wood_length_equation_l897_89766


namespace NUMINAMATH_GPT_trigonometric_identity_l897_89754

theorem trigonometric_identity :
  (Real.sqrt 3 / Real.cos (10 * Real.pi / 180) - 1 / Real.sin (170 * Real.pi / 180) = -4) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l897_89754


namespace NUMINAMATH_GPT_min_value_complex_mod_one_l897_89755

/-- Given that the modulus of the complex number \( z \) is 1, prove that the minimum value of
    \( |z - 4|^2 + |z + 3 * Complex.I|^2 \) is \( 17 \). -/
theorem min_value_complex_mod_one (z : ℂ) (h : ‖z‖ = 1) : 
  ∃ α : ℝ, (‖z - 4‖^2 + ‖z + 3 * Complex.I‖^2) = 17 :=
sorry

end NUMINAMATH_GPT_min_value_complex_mod_one_l897_89755


namespace NUMINAMATH_GPT_equation_solution_count_l897_89732

theorem equation_solution_count (n : ℕ) (h_pos : n > 0)
    (h_solutions : ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 28 ∧ ∀ (x y z : ℕ), (x, y, z) ∈ s → 2 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) :
    n = 17 ∨ n = 18 :=
sorry

end NUMINAMATH_GPT_equation_solution_count_l897_89732


namespace NUMINAMATH_GPT_mo_rainy_days_last_week_l897_89724

theorem mo_rainy_days_last_week (R NR n : ℕ) (h1 : n * R + 4 * NR = 26) (h2 : 4 * NR - n * R = 14) (h3 : R + NR = 7) : R = 2 :=
sorry

end NUMINAMATH_GPT_mo_rainy_days_last_week_l897_89724


namespace NUMINAMATH_GPT_work_completion_time_l897_89768

/-
Conditions:
1. A man alone can do the work in 6 days.
2. A woman alone can do the work in 18 days.
3. A boy alone can do the work in 9 days.

Question:
How long will they take to complete the work together?

Correct Answer:
3 days
-/

theorem work_completion_time (M W B : ℕ) (hM : M = 6) (hW : W = 18) (hB : B = 9) : 1 / (1/M + 1/W + 1/B) = 3 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l897_89768


namespace NUMINAMATH_GPT_problem_statement_l897_89778

namespace MathProof

def p : Prop := (2 + 4 = 7)
def q : Prop := ∀ x : ℝ, x = 1 → x^2 ≠ 1

theorem problem_statement : ¬ (p ∧ q) ∧ (p ∨ q) :=
by
  -- To be filled in
  sorry

end MathProof

end NUMINAMATH_GPT_problem_statement_l897_89778


namespace NUMINAMATH_GPT_angle_trig_identity_l897_89780

theorem angle_trig_identity
  (A B C : ℝ)
  (h_sum : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 = Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 - 
                       2 * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2) :=
by
  sorry

end NUMINAMATH_GPT_angle_trig_identity_l897_89780


namespace NUMINAMATH_GPT_problem_l897_89707

-- Definitions
variables {a b : ℝ}
def is_root (p : ℝ → ℝ) (x : ℝ) : Prop := p x = 0

-- Root condition using the given equation
def quadratic_eq (x : ℝ) : ℝ := (x - 3) * (2 * x + 7) - (x^2 - 11 * x + 28)

-- Statement to prove
theorem problem (ha : is_root quadratic_eq a) (hb : is_root quadratic_eq b) (h_distinct : a ≠ b):
  (a + 2) * (b + 2) = -66 :=
sorry

end NUMINAMATH_GPT_problem_l897_89707


namespace NUMINAMATH_GPT_find_x_if_alpha_beta_eq_4_l897_89726

def alpha (x : ℝ) : ℝ := 4 * x + 9
def beta (x : ℝ) : ℝ := 9 * x + 6

theorem find_x_if_alpha_beta_eq_4 :
  (∃ x : ℝ, alpha (beta x) = 4 ∧ x = -29 / 36) :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_alpha_beta_eq_4_l897_89726


namespace NUMINAMATH_GPT_find_largest_number_l897_89792

theorem find_largest_number
  (a b c d e : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h₁ : a + b = 32)
  (h₂ : a + c = 36)
  (h₃ : b + c = 37)
  (h₄ : c + e = 48)
  (h₅ : d + e = 51) :
  (max a (max b (max c (max d e)))) = 27.5 :=
sorry

end NUMINAMATH_GPT_find_largest_number_l897_89792


namespace NUMINAMATH_GPT_modulus_remainder_l897_89730

namespace Proof

def a (n : ℕ) : ℕ := 88134 + n

theorem modulus_remainder :
  (2 * ((a 0)^2 + (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2)) % 11 = 3 := by
  sorry

end Proof

end NUMINAMATH_GPT_modulus_remainder_l897_89730


namespace NUMINAMATH_GPT_profit_per_tire_l897_89727

theorem profit_per_tire
  (fixed_cost : ℝ)
  (variable_cost_per_tire : ℝ)
  (selling_price_per_tire : ℝ)
  (batch_size : ℕ)
  (total_cost : ℝ)
  (total_revenue : ℝ)
  (total_profit : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : variable_cost_per_tire = 8)
  (h3 : selling_price_per_tire = 20)
  (h4 : batch_size = 15000)
  (h5 : total_cost = fixed_cost + variable_cost_per_tire * batch_size)
  (h6 : total_revenue = selling_price_per_tire * batch_size)
  (h7 : total_profit = total_revenue - total_cost)
  (h8 : profit_per_tire = total_profit / batch_size) :
  profit_per_tire = 10.50 :=
sorry

end NUMINAMATH_GPT_profit_per_tire_l897_89727


namespace NUMINAMATH_GPT_fish_pond_estimate_l897_89774

variable (N : ℕ)
variable (total_first_catch total_second_catch marked_in_first_catch marked_in_second_catch : ℕ)

/-- Estimate the total number of fish in the pond -/
theorem fish_pond_estimate
  (h1 : total_first_catch = 100)
  (h2 : total_second_catch = 120)
  (h3 : marked_in_first_catch = 100)
  (h4 : marked_in_second_catch = 15)
  (h5 : (marked_in_second_catch : ℚ) / total_second_catch = (marked_in_first_catch : ℚ) / N) :
  N = 800 := 
sorry

end NUMINAMATH_GPT_fish_pond_estimate_l897_89774


namespace NUMINAMATH_GPT_shirley_cases_needed_l897_89728

-- Define the given conditions
def trefoils_boxes := 54
def samoas_boxes := 36
def boxes_per_case := 6

-- The statement to prove
theorem shirley_cases_needed : trefoils_boxes / boxes_per_case >= samoas_boxes / boxes_per_case ∧ 
                               samoas_boxes / boxes_per_case = 6 :=
by
  let n_cases := samoas_boxes / boxes_per_case
  have h1 : trefoils_boxes / boxes_per_case = 9 := sorry
  have h2 : samoas_boxes / boxes_per_case = 6 := sorry
  have h3 : 9 >= 6 := by linarith
  exact ⟨h3, h2⟩


end NUMINAMATH_GPT_shirley_cases_needed_l897_89728


namespace NUMINAMATH_GPT_parabola_focus_directrix_distance_l897_89763

theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), 
    y^2 = 8 * x → 
    ∃ p : ℝ, 2 * p = 8 ∧ p = 4 := by
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_distance_l897_89763


namespace NUMINAMATH_GPT_total_miles_l897_89760

-- Define the variables and equations as given in the conditions
variables (a b c d e : ℝ)
axiom h1 : a + b = 36
axiom h2 : b + c + d = 45
axiom h3 : c + d + e = 45
axiom h4 : a + c + e = 38

-- The conjecture we aim to prove
theorem total_miles : a + b + c + d + e = 83 :=
sorry

end NUMINAMATH_GPT_total_miles_l897_89760


namespace NUMINAMATH_GPT_wage_increase_l897_89743

-- Definition: Regression line equation
def regression_line (x : ℝ) : ℝ := 80 * x + 50

-- Theorem: On average, when the labor productivity increases by 1000 yuan, the wage increases by 80 yuan
theorem wage_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 80 :=
by
  sorry

end NUMINAMATH_GPT_wage_increase_l897_89743


namespace NUMINAMATH_GPT_math_problem_l897_89737

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0):
  ((2^(2^n) + 1) * (2^(2^m) + 1)) % (m * n) = 0 →
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l897_89737


namespace NUMINAMATH_GPT_arithmetic_example_l897_89777

theorem arithmetic_example : (2468 * 629) / (1234 * 37) = 34 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_example_l897_89777


namespace NUMINAMATH_GPT_angle_sum_around_point_l897_89775

theorem angle_sum_around_point (y : ℕ) (h1 : 210 + 3 * y = 360) : y = 50 := 
by 
  sorry

end NUMINAMATH_GPT_angle_sum_around_point_l897_89775


namespace NUMINAMATH_GPT_solve_a_b_l897_89714

theorem solve_a_b (a b : ℕ) (h₀ : 2 * a^2 = 3 * b^3) : ∃ k : ℕ, a = 18 * k^3 ∧ b = 6 * k^2 := 
sorry

end NUMINAMATH_GPT_solve_a_b_l897_89714


namespace NUMINAMATH_GPT_performance_attendance_l897_89708

theorem performance_attendance (A C : ℕ) (hC : C = 18) (hTickets : 16 * A + 9 * C = 258) : A + C = 24 :=
by
  sorry

end NUMINAMATH_GPT_performance_attendance_l897_89708


namespace NUMINAMATH_GPT_zhou_yu_age_eq_l897_89758

-- Define the conditions based on the problem statement
variable (x : ℕ)  -- x represents the tens digit of Zhou Yu's age

-- Condition: The tens digit is three less than the units digit
def units_digit := x + 3

-- Define Zhou Yu's age based on the tens and units digits
def zhou_yu_age := 10 * x + units_digit x

-- Prove the correct equation representing Zhou Yu's lifespan
theorem zhou_yu_age_eq : zhou_yu_age x = (units_digit x) ^ 2 :=
by sorry

end NUMINAMATH_GPT_zhou_yu_age_eq_l897_89758


namespace NUMINAMATH_GPT_age_sum_l897_89750

variables (A B C : ℕ)

theorem age_sum (h1 : A = 20 + B + C) (h2 : A^2 = 2000 + (B + C)^2) : A + B + C = 100 :=
by
  -- Assume the subsequent proof follows here
  sorry

end NUMINAMATH_GPT_age_sum_l897_89750


namespace NUMINAMATH_GPT_cookies_guests_l897_89745

theorem cookies_guests (cc_cookies : ℕ) (oc_cookies : ℕ) (sc_cookies : ℕ) (cc_per_guest : ℚ) (oc_per_guest : ℚ) (sc_per_guest : ℕ)
    (cc_total : cc_cookies = 45) (oc_total : oc_cookies = 62) (sc_total : sc_cookies = 38) (cc_ratio : cc_per_guest = 1.5)
    (oc_ratio : oc_per_guest = 2.25) (sc_ratio : sc_per_guest = 1) :
    (cc_cookies / cc_per_guest) ≥ 0 ∧ (oc_cookies / oc_per_guest) ≥ 0 ∧ (sc_cookies / sc_per_guest) ≥ 0 → 
    Nat.floor (oc_cookies / oc_per_guest) = 27 :=
by
  sorry

end NUMINAMATH_GPT_cookies_guests_l897_89745


namespace NUMINAMATH_GPT_compare_neg_frac1_l897_89719

theorem compare_neg_frac1 : (-3 / 7 : ℝ) < (-8 / 21 : ℝ) :=
sorry

end NUMINAMATH_GPT_compare_neg_frac1_l897_89719


namespace NUMINAMATH_GPT_find_f_2017_div_2_l897_89764

noncomputable def is_odd_function {X Y : Type*} [AddGroup X] [AddGroup Y] (f : X → Y) :=
  ∀ x : X, f (-x) = -f x

noncomputable def is_periodic_function {X Y : Type*} [AddGroup X] [AddGroup Y] (p : X) (f : X → Y) :=
  ∀ x : X, f (x + p) = f x

noncomputable def f : ℝ → ℝ 
| x => if -1 ≤ x ∧ x ≤ 0 then x * x + x else sorry

theorem find_f_2017_div_2 : f (2017 / 2) = 1 / 4 :=
by
  have h_odd : is_odd_function f := sorry
  have h_period : is_periodic_function 2 f := sorry
  unfold f
  sorry

end NUMINAMATH_GPT_find_f_2017_div_2_l897_89764


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_12_15_16_exists_l897_89709

theorem smallest_positive_integer_divisible_12_15_16_exists :
  ∃ x : ℕ, x > 0 ∧ 12 ∣ x ∧ 15 ∣ x ∧ 16 ∣ x ∧ x = 240 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_integer_divisible_12_15_16_exists_l897_89709


namespace NUMINAMATH_GPT_problem_statement_l897_89751

open Nat

theorem problem_statement (k : ℕ) (hk : k > 0) : 
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * (factorial (k*n) / factorial n) ≤ (factorial (k*n) / factorial n))
  ↔ ∃ a : ℕ, k = 2^a := 
sorry

end NUMINAMATH_GPT_problem_statement_l897_89751


namespace NUMINAMATH_GPT_committee_selection_l897_89759

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection :
  let seniors := 10
  let members := 30
  let non_seniors := members - seniors
  let choices := binom seniors 2 * binom non_seniors 3 +
                 binom seniors 3 * binom non_seniors 2 +
                 binom seniors 4 * binom non_seniors 1 +
                 binom seniors 5
  choices = 78552 :=
by
  sorry

end NUMINAMATH_GPT_committee_selection_l897_89759


namespace NUMINAMATH_GPT_cube_product_l897_89742

/-- A cube is a three-dimensional shape with a specific number of vertices and faces. -/
structure Cube where
  vertices : ℕ
  faces : ℕ

theorem cube_product (C : Cube) (h1: C.vertices = 8) (h2: C.faces = 6) : 
  (C.vertices * C.faces = 48) :=
by sorry

end NUMINAMATH_GPT_cube_product_l897_89742


namespace NUMINAMATH_GPT_total_shirts_correct_l897_89787

def machine_A_production_rate := 6
def machine_A_yesterday_minutes := 12
def machine_A_today_minutes := 10

def machine_B_production_rate := 8
def machine_B_yesterday_minutes := 10
def machine_B_today_minutes := 15

def machine_C_production_rate := 5
def machine_C_yesterday_minutes := 20
def machine_C_today_minutes := 0

def total_shirts_produced : Nat :=
  (machine_A_production_rate * machine_A_yesterday_minutes +
  machine_A_production_rate * machine_A_today_minutes) +
  (machine_B_production_rate * machine_B_yesterday_minutes +
  machine_B_production_rate * machine_B_today_minutes) +
  (machine_C_production_rate * machine_C_yesterday_minutes +
  machine_C_production_rate * machine_C_today_minutes)

theorem total_shirts_correct : total_shirts_produced = 432 :=
by 
  sorry 

end NUMINAMATH_GPT_total_shirts_correct_l897_89787


namespace NUMINAMATH_GPT_graph_not_through_third_quadrant_l897_89757

theorem graph_not_through_third_quadrant (k : ℝ) (h_nonzero : k ≠ 0) (h_decreasing : k < 0) : 
  ¬(∃ x y : ℝ, y = k * x - k ∧ x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_GPT_graph_not_through_third_quadrant_l897_89757


namespace NUMINAMATH_GPT_corrected_mean_l897_89797

theorem corrected_mean :
  let original_mean := 45
  let num_observations := 100
  let observations_wrong := [32, 12, 25]
  let observations_correct := [67, 52, 85]
  let original_total_sum := original_mean * num_observations
  let incorrect_sum := observations_wrong.sum
  let correct_sum := observations_correct.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_total_sum := original_total_sum + adjustment
  let corrected_new_mean := corrected_total_sum / num_observations
  corrected_new_mean = 46.35 := 
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l897_89797


namespace NUMINAMATH_GPT_beth_red_pill_cost_l897_89715

noncomputable def red_pill_cost (blue_pill_cost : ℝ) : ℝ := blue_pill_cost + 3

theorem beth_red_pill_cost :
  ∃ (blue_pill_cost : ℝ), 
  (21 * (red_pill_cost blue_pill_cost + blue_pill_cost) = 966) 
  → 
  red_pill_cost blue_pill_cost = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_beth_red_pill_cost_l897_89715


namespace NUMINAMATH_GPT_work_rate_b_l897_89704

theorem work_rate_b (W : ℝ) (A B C : ℝ) :
  (A = W / 11) → 
  (C = W / 55) →
  (8 * A + 4 * B + 4 * C = W) →
  B = W / (2420 / 341) :=
by
  intros hA hC hWork
  -- We start with the given assumptions and work towards showing B = W / (2420 / 341)
  sorry

end NUMINAMATH_GPT_work_rate_b_l897_89704


namespace NUMINAMATH_GPT_rectangle_difference_length_width_l897_89721

theorem rectangle_difference_length_width (x y p d : ℝ) (h1 : x + y = p / 2) (h2 : x^2 + y^2 = d^2) (h3 : x > y) : 
  x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 := sorry

end NUMINAMATH_GPT_rectangle_difference_length_width_l897_89721


namespace NUMINAMATH_GPT_compute_fraction_product_l897_89786

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_product_l897_89786


namespace NUMINAMATH_GPT_count_monomials_l897_89749

def isMonomial (expr : String) : Bool :=
  match expr with
  | "m+n" => false
  | "2x^2y" => true
  | "1/x" => true
  | "-5" => true
  | "a" => true
  | _ => false

theorem count_monomials :
  let expressions := ["m+n", "2x^2y", "1/x", "-5", "a"]
  (expressions.filter isMonomial).length = 3 :=
by { sorry }

end NUMINAMATH_GPT_count_monomials_l897_89749


namespace NUMINAMATH_GPT_snooker_tournament_l897_89741

theorem snooker_tournament : 
  ∀ (V G : ℝ),
    V + G = 320 →
    40 * V + 15 * G = 7500 →
    V ≥ 80 →
    G ≥ 100 →
    G - V = 104 :=
by
  intros V G h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_snooker_tournament_l897_89741


namespace NUMINAMATH_GPT_inequality_inequality_hold_l897_89752

theorem inequality_inequality_hold (k : ℕ) (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_sum : x + y + z = 1) : 
  (x ^ (k + 2) / (x ^ (k + 1) + y ^ k + z ^ k) 
  + y ^ (k + 2) / (y ^ (k + 1) + z ^ k + x ^ k) 
  + z ^ (k + 2) / (z ^ (k + 1) + x ^ k + y ^ k)) 
  ≥ (1 / 7) :=
sorry

end NUMINAMATH_GPT_inequality_inequality_hold_l897_89752


namespace NUMINAMATH_GPT_arithmetic_difference_l897_89731

variable (S : ℕ → ℤ)
variable (n : ℕ)

-- Definitions as conditions from the problem
def is_arithmetic_sum (s : ℕ → ℤ) :=
  ∀ n : ℕ, s n = 2 * n ^ 2 - 5 * n

theorem arithmetic_difference :
  is_arithmetic_sum S →
  S 10 - S 7 = 87 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_arithmetic_difference_l897_89731


namespace NUMINAMATH_GPT_shelter_cats_l897_89789

theorem shelter_cats (initial_dogs initial_cats additional_cats : ℕ) 
  (h1 : initial_dogs = 75)
  (h2 : initial_dogs * 7 = initial_cats * 15)
  (h3 : initial_dogs * 11 = 15 * (initial_cats + additional_cats)) : 
  additional_cats = 20 :=
by
  sorry

end NUMINAMATH_GPT_shelter_cats_l897_89789


namespace NUMINAMATH_GPT_no_real_solutions_l897_89767

open Real

theorem no_real_solutions :
  ¬(∃ x : ℝ, (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 = 0) := by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l897_89767


namespace NUMINAMATH_GPT_find_s_over_r_l897_89781

-- Define the function
def f (k : ℝ) : ℝ := 9 * k ^ 2 - 6 * k + 15

-- Define constants
variables (d r s : ℝ)

-- Define the main theorem to be proved
theorem find_s_over_r : 
  (∀ k : ℝ, f k = d * (k + r) ^ 2 + s) → s / r = -42 :=
by
  sorry

end NUMINAMATH_GPT_find_s_over_r_l897_89781


namespace NUMINAMATH_GPT_min_value_frac_sum_l897_89770

-- Define the main problem
theorem min_value_frac_sum (a b : ℝ) (h1 : 2 * a + 3 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ x : ℝ, (x = 25) ∧ ∀ y, (y = (2 / a + 3 / b)) → y ≥ x :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l897_89770


namespace NUMINAMATH_GPT_max_value_of_quadratic_expression_l897_89756

theorem max_value_of_quadratic_expression (s : ℝ) : ∃ x : ℝ, -3 * s^2 + 24 * s - 8 ≤ x ∧ x = 40 :=
sorry

end NUMINAMATH_GPT_max_value_of_quadratic_expression_l897_89756


namespace NUMINAMATH_GPT_cyclic_quadrilateral_tangency_l897_89702

theorem cyclic_quadrilateral_tangency (a b c d x y : ℝ) (h_cyclic : a = 80 ∧ b = 100 ∧ c = 140 ∧ d = 120) 
  (h_tangency: x + y = 140) : |x - y| = 5 := 
sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_tangency_l897_89702


namespace NUMINAMATH_GPT_length_of_CD_l897_89761

theorem length_of_CD (x y : ℝ) (h1 : x = (1/5) * (4 + y))
  (h2 : (x + 4) / y = 2 / 3) (h3 : 4 = 4) : x + y + 4 = 17.143 :=
sorry

end NUMINAMATH_GPT_length_of_CD_l897_89761


namespace NUMINAMATH_GPT_probability_of_x_gt_8y_l897_89790

noncomputable def probability_x_gt_8y : ℚ :=
  let rect_area := 2020 * 2030
  let tri_area := (2020 * (2020 / 8)) / 2
  tri_area / rect_area

theorem probability_of_x_gt_8y :
  probability_x_gt_8y = 255025 / 4100600 := by
  sorry

end NUMINAMATH_GPT_probability_of_x_gt_8y_l897_89790


namespace NUMINAMATH_GPT_parabola_equation_l897_89747

theorem parabola_equation (h k : ℝ) (p : ℝ × ℝ) (a b c : ℝ) :
  h = 3 ∧ k = -2 ∧ p = (4, -5) ∧
  (∀ x y : ℝ, y = a * (x - h) ^ 2 + k → p.2 = a * (p.1 - h) ^ 2 + k) →
  -(3:ℝ) = a ∧ 18 = b ∧ -29 = c :=
by sorry

end NUMINAMATH_GPT_parabola_equation_l897_89747


namespace NUMINAMATH_GPT_tangent_line_eq_l897_89713

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

def point : ℝ × ℝ := (2, 5)

theorem tangent_line_eq : ∀ (x y : ℝ), 
  (y = x^2 + 3 * x + 1) ∧ (x = 2 ∧ y = 5) →
  7 * x - y = 9 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l897_89713


namespace NUMINAMATH_GPT_maximize_product_minimize_product_l897_89735

-- Define lists of the digits to be used
def digits : List ℕ := [2, 4, 6, 8]

-- Function to calculate the number from a list of digits
def toNumber (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * 10 + d) 0

-- Function to calculate the product given two numbers represented as lists of digits
def product (digits1 digits2 : List ℕ) : ℕ :=
  toNumber digits1 * toNumber digits2

-- Definitions of specific permutations to be used
def maxDigits1 : List ℕ := [8, 6, 4]
def maxDigit2 : List ℕ := [2]
def minDigits1 : List ℕ := [2, 4, 6]
def minDigit2 : List ℕ := [8]

-- Theorem statements
theorem maximize_product : product maxDigits1 maxDigit2 = 864 * 2 := by
  sorry

theorem minimize_product : product minDigits1 minDigit2 = 246 * 8 := by
  sorry

end NUMINAMATH_GPT_maximize_product_minimize_product_l897_89735


namespace NUMINAMATH_GPT_smallest_number_am_median_l897_89748

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_am_median_l897_89748


namespace NUMINAMATH_GPT_prob_neq_zero_l897_89700

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 
  then (5/6)^4 
  else 0

theorem prob_neq_zero (a b c d : ℕ) :
  (1 ≤ a) ∧ (a ≤ 6) ∧ (1 ≤ b) ∧ (b ≤ 6) ∧ (1 ≤ c) ∧ (c ≤ 6) ∧ (1 ≤ d) ∧ (d ≤ 6) →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  probability_no_one a b c d = 625/1296 :=
by
  sorry

end NUMINAMATH_GPT_prob_neq_zero_l897_89700


namespace NUMINAMATH_GPT_expected_volunteers_2008_l897_89739

theorem expected_volunteers_2008 (initial_volunteers: ℕ) (annual_increase: ℚ) (h1: initial_volunteers = 500) (h2: annual_increase = 1.2) : 
  let volunteers_2006 := initial_volunteers * annual_increase
  let volunteers_2007 := volunteers_2006 * annual_increase
  let volunteers_2008 := volunteers_2007 * annual_increase
  volunteers_2008 = 864 := 
by
  sorry

end NUMINAMATH_GPT_expected_volunteers_2008_l897_89739


namespace NUMINAMATH_GPT_fraction_addition_l897_89705

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end NUMINAMATH_GPT_fraction_addition_l897_89705


namespace NUMINAMATH_GPT_g_88_value_l897_89788

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n m : ℕ) (h : n < m) : g n < g m
axiom g_multiplicative (m n : ℕ) : g (m * n) = g m * g n
axiom g_exponential_condition (m n : ℕ) (h : m ≠ n ∧ m ^ n = n ^ m) : g m = n ∨ g n = m

theorem g_88_value : g 88 = 7744 :=
sorry

end NUMINAMATH_GPT_g_88_value_l897_89788


namespace NUMINAMATH_GPT_number_of_boxes_ordered_l897_89769

-- Definitions based on the conditions
def boxes_contain_matchboxes : Nat := 20
def matchboxes_contain_sticks : Nat := 300
def total_match_sticks : Nat := 24000

-- Statement of the proof problem
theorem number_of_boxes_ordered :
  (total_match_sticks / matchboxes_contain_sticks) / boxes_contain_matchboxes = 4 := 
sorry

end NUMINAMATH_GPT_number_of_boxes_ordered_l897_89769


namespace NUMINAMATH_GPT_complement_A_in_U_l897_89785

def U := {x : ℝ | -4 < x ∧ x < 4}
def A := {x : ℝ | -3 ≤ x ∧ x < 2}

theorem complement_A_in_U :
  {x : ℝ | x ∈ U ∧ x ∉ A} = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_A_in_U_l897_89785


namespace NUMINAMATH_GPT_problem_statement_l897_89753

noncomputable def f1 (x : ℝ) := x + (1 / x)
noncomputable def f2 (x : ℝ) := 1 / (x ^ 2)
noncomputable def f3 (x : ℝ) := x ^ 3 - 2 * x
noncomputable def f4 (x : ℝ) := x ^ 2

theorem problem_statement : ∀ (x : ℝ), f2 (-x) = f2 x := by 
  sorry

end NUMINAMATH_GPT_problem_statement_l897_89753


namespace NUMINAMATH_GPT_product_increase_false_l897_89711

theorem product_increase_false (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬ (a * b = a * (10 * b) / 10 ∧ a * (10 * b) / 10 = 10 * (a * b)) :=
by 
  sorry

end NUMINAMATH_GPT_product_increase_false_l897_89711


namespace NUMINAMATH_GPT_quadruple_solution_l897_89799

theorem quadruple_solution (a b p n : ℕ) (hp: Nat.Prime p) (hp_pos: p > 0) (ha_pos: a > 0) (hb_pos: b > 0) (hn_pos: n > 0) :
    a^3 + b^3 = p^n →
    (∃ k, k ≥ 1 ∧ (
        (a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k-2) ∨ 
        (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k-1) ∨ 
        (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k-1)
    )) := 
sorry

end NUMINAMATH_GPT_quadruple_solution_l897_89799


namespace NUMINAMATH_GPT_practice_problems_total_l897_89744

theorem practice_problems_total :
  let marvin_yesterday := 40
  let marvin_today := 3 * marvin_yesterday
  let arvin_yesterday := 2 * marvin_yesterday
  let arvin_today := 2 * marvin_today
  let kevin_yesterday := 30
  let kevin_today := kevin_yesterday + 10
  let total_problems := (marvin_yesterday + marvin_today) + (arvin_yesterday + arvin_today) + (kevin_yesterday + kevin_today)
  total_problems = 550 :=
by
  sorry

end NUMINAMATH_GPT_practice_problems_total_l897_89744


namespace NUMINAMATH_GPT_divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l897_89772

open Nat

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (factors n).eraseDups.length

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

variables {p q m n : ℕ}
variables (hp : is_prime p) (hq : is_prime q) (hdist : p ≠ q) (hm : 0 ≤ m) (hn : 0 ≤ n)

-- a) Prove the number of divisors of pq is 4
theorem divisors_pq : num_divisors (p * q) = 4 :=
sorry

-- b) Prove the number of divisors of p^2 q is 6
theorem divisors_p2q : num_divisors (p^2 * q) = 6 :=
sorry

-- c) Prove the number of divisors of p^2 q^2 is 9
theorem divisors_p2q2 : num_divisors (p^2 * q^2) = 9 :=
sorry

-- d) Prove the number of divisors of p^m q^n is (m + 1)(n + 1)
theorem divisors_pmqn : num_divisors (p^m * q^n) = (m + 1) * (n + 1) :=
sorry

end NUMINAMATH_GPT_divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l897_89772


namespace NUMINAMATH_GPT_inradius_of_triangle_l897_89765

theorem inradius_of_triangle (P A : ℝ) (hP : P = 40) (hA : A = 50) : 
  ∃ r : ℝ, r = 2.5 ∧ A = r * (P / 2) :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l897_89765


namespace NUMINAMATH_GPT_crayons_problem_l897_89736

theorem crayons_problem
  (S M L : ℕ)
  (hS_condition : (3 / 5 : ℚ) * S = 60)
  (hM_condition : (1 / 4 : ℚ) * M = 98)
  (hL_condition : (4 / 7 : ℚ) * L = 168) :
  S = 100 ∧ M = 392 ∧ L = 294 ∧ ((2 / 5 : ℚ) * S + (3 / 4 : ℚ) * M + (3 / 7 : ℚ) * L = 460) := 
by
  sorry

end NUMINAMATH_GPT_crayons_problem_l897_89736


namespace NUMINAMATH_GPT_fraction_dutch_americans_has_window_l897_89701

variable (P D DA : ℕ)
variable (f_P_d d_P_w : ℚ)
variable (DA_w : ℕ)

-- Total number of people on the bus P 
-- Fraction of people who were Dutch f_P_d
-- Fraction of Dutch Americans who got window seats d_P_w
-- Number of Dutch Americans who sat at windows DA_w
-- Define the assumptions
def total_people_on_bus := P = 90
def fraction_dutch := f_P_d = 3 / 5
def fraction_dutch_americans_window := d_P_w = 1 / 3
def dutch_americans_window := DA_w = 9

-- Prove that fraction of Dutch people who were also American is 1/2
theorem fraction_dutch_americans_has_window (P D DA DA_w : ℕ) (f_P_d d_P_w : ℚ) :
  total_people_on_bus P ∧ fraction_dutch f_P_d ∧
  fraction_dutch_americans_window d_P_w ∧ dutch_americans_window DA_w →
  (DA: ℚ) / D = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_dutch_americans_has_window_l897_89701


namespace NUMINAMATH_GPT_longer_side_of_rectangle_l897_89771

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end NUMINAMATH_GPT_longer_side_of_rectangle_l897_89771


namespace NUMINAMATH_GPT_sum_of_squares_pentagon_greater_icosagon_l897_89782

noncomputable def compare_sum_of_squares (R : ℝ) : Prop :=
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  4 * a_20^2 < a_5^2

theorem sum_of_squares_pentagon_greater_icosagon (R : ℝ) : 
  compare_sum_of_squares R :=
  sorry

end NUMINAMATH_GPT_sum_of_squares_pentagon_greater_icosagon_l897_89782


namespace NUMINAMATH_GPT_jose_tabs_remaining_l897_89794

def initial_tabs : Nat := 400
def step1_tabs_closed (n : Nat) : Nat := n / 4
def step2_tabs_closed (n : Nat) : Nat := 2 * n / 5
def step3_tabs_closed (n : Nat) : Nat := n / 2

theorem jose_tabs_remaining :
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  after_step3 = 90 :=
by
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  have h : after_step3 = 90 := sorry
  exact h

end NUMINAMATH_GPT_jose_tabs_remaining_l897_89794


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l897_89746

theorem sufficient_but_not_necessary (x : ℝ) : 
  (1 < x ∧ x < 2) → (x > 0) ∧ ¬((x > 0) → (1 < x ∧ x < 2)) := 
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l897_89746


namespace NUMINAMATH_GPT_division_correct_l897_89783

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end NUMINAMATH_GPT_division_correct_l897_89783


namespace NUMINAMATH_GPT_kamal_twice_age_in_future_l897_89703

theorem kamal_twice_age_in_future :
  ∃ x : ℕ, (K = 40) ∧ (K - 8 = 4 * (S - 8)) ∧ (K + x = 2 * (S + x)) :=
by {
  sorry 
}

end NUMINAMATH_GPT_kamal_twice_age_in_future_l897_89703
