import Mathlib

namespace NUMINAMATH_GPT_driving_scenario_l1144_114416

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_driving_scenario_l1144_114416


namespace NUMINAMATH_GPT_max_value_A_l1144_114414

theorem max_value_A (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( ( (x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2 ) / 
    ( (x - y)^2 + (y - z)^2 + (z - x)^2 + 2 ) ) ≤ 1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_A_l1144_114414


namespace NUMINAMATH_GPT_simplify_expression_evaluate_expression_l1144_114439

-- Definitions for the first part
variable (a b : ℝ)

theorem simplify_expression (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2)) / (1/3 * a^(1/6) * b^(5/6)) = 6 * a :=
by
  sorry

-- Definitions for the second part
theorem evaluate_expression :
  (9 / 16)^(1 / 2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + Real.log (4 * Real.exp 3) 
  - (Real.log 8 / Real.log 9) * (Real.log 33 / Real.log 4) = 7 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_evaluate_expression_l1144_114439


namespace NUMINAMATH_GPT_job_completion_days_l1144_114428

theorem job_completion_days :
  let days_total := 150
  let workers_initial := 25
  let workers_less_efficient := 15
  let workers_more_efficient := 10
  let days_elapsed := 40
  let efficiency_less := 1
  let efficiency_more := 1.5
  let work_fraction_completed := 1/3
  let workers_fired_less := 4
  let workers_fired_more := 3
  let units_per_day_initial := (workers_less_efficient * efficiency_less) + (workers_more_efficient * efficiency_more)
  let work_completed := units_per_day_initial * days_elapsed
  let total_work := work_completed / work_fraction_completed
  let workers_remaining_less := workers_less_efficient - workers_fired_less
  let workers_remaining_more := workers_more_efficient - workers_fired_more
  let units_per_day_new := (workers_remaining_less * efficiency_less) + (workers_remaining_more * efficiency_more)
  let work_remaining := total_work * (2/3)
  let remaining_days := work_remaining / units_per_day_new
  remaining_days.ceil = 112 :=
by
  sorry

end NUMINAMATH_GPT_job_completion_days_l1144_114428


namespace NUMINAMATH_GPT_karen_cases_picked_up_l1144_114488

theorem karen_cases_picked_up (total_boxes : ℤ) (boxes_per_case : ℤ) (h1 : total_boxes = 36) (h2 : boxes_per_case = 12) : (total_boxes / boxes_per_case) = 3 := by
  sorry

end NUMINAMATH_GPT_karen_cases_picked_up_l1144_114488


namespace NUMINAMATH_GPT_product_of_last_two_digits_l1144_114481

theorem product_of_last_two_digits (n A B : ℤ) 
  (h1 : n % 8 = 0) 
  (h2 : 10 * A + B = n % 100) 
  (h3 : A + B = 14) : 
  A * B = 48 := 
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l1144_114481


namespace NUMINAMATH_GPT_solve_logarithmic_equation_l1144_114443

/-- The solution to the equation log_2(9^x - 5) = 2 + log_2(3^x - 2) is x = 1. -/
theorem solve_logarithmic_equation (x : ℝ) :
  (Real.logb 2 (9^x - 5) = 2 + Real.logb 2 (3^x - 2)) → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_logarithmic_equation_l1144_114443


namespace NUMINAMATH_GPT_problem_statement_l1144_114471

theorem problem_statement (a b c : ℝ) (h : a * c^2 > b * c^2) (hc : c ≠ 0) : 
  a > b :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1144_114471


namespace NUMINAMATH_GPT_flagpole_shadow_length_correct_l1144_114477

noncomputable def flagpole_shadow_length (flagpole_height building_height building_shadow_length : ℕ) :=
  flagpole_height * building_shadow_length / building_height

theorem flagpole_shadow_length_correct :
  flagpole_shadow_length 18 20 50 = 45 :=
by
  sorry

end NUMINAMATH_GPT_flagpole_shadow_length_correct_l1144_114477


namespace NUMINAMATH_GPT_find_C_l1144_114420

theorem find_C (A B C : ℕ) (h1 : (8 + 4 + A + 7 + 3 + B + 2) % 3 = 0)
  (h2 : (5 + 2 + 9 + A + B + 4 + C) % 3 = 0) : C = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l1144_114420


namespace NUMINAMATH_GPT_inequality_areas_l1144_114404

theorem inequality_areas (a b c α β γ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  a / α + b / β + c / γ ≥ 3 / 2 :=
by
  -- Insert the AM-GM inequality application and simplifications
  sorry

end NUMINAMATH_GPT_inequality_areas_l1144_114404


namespace NUMINAMATH_GPT_paper_strip_total_covered_area_l1144_114483

theorem paper_strip_total_covered_area :
  let length := 12
  let width := 2
  let strip_count := 5
  let overlap_per_intersection := 4
  let intersection_count := 10
  let area_per_strip := length * width
  let total_area_without_overlap := strip_count * area_per_strip
  let total_overlap_area := intersection_count * overlap_per_intersection
  total_area_without_overlap - total_overlap_area = 80 := 
by
  sorry

end NUMINAMATH_GPT_paper_strip_total_covered_area_l1144_114483


namespace NUMINAMATH_GPT_value_of_x_l1144_114447

-- Let a and b be real numbers.
variable (a b : ℝ)

-- Given conditions
def cond_1 : 10 * a = 6 * b := sorry
def cond_2 : 120 * a * b = 800 := sorry

theorem value_of_x (x : ℝ) (h1 : 10 * a = x) (h2 : 6 * b = x) (h3 : 120 * a * b = 800) : x = 20 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1144_114447


namespace NUMINAMATH_GPT_compute_expression_l1144_114415

theorem compute_expression : (46 + 15)^2 - (46 - 15)^2 = 2760 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1144_114415


namespace NUMINAMATH_GPT_find_y_of_series_eq_92_l1144_114403

theorem find_y_of_series_eq_92 (y : ℝ) (h : (∑' n, (2 + 5 * n) * y^n) = 92) (converge : abs y < 1) : y = 18 / 23 :=
sorry

end NUMINAMATH_GPT_find_y_of_series_eq_92_l1144_114403


namespace NUMINAMATH_GPT_intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l1144_114418

def U := ℝ
def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def C_U_B : Set ℝ := {x | x < -2 ∨ x ≥ 4}

theorem intersection_A_B_eq : A ∩ B = {x | 0 ≤ x ∧ x < 4} := by
  sorry

theorem union_A_B_eq : A ∪ B = {x | -2 ≤ x ∧ x < 5} := by
  sorry

theorem intersection_A_C_U_B_eq : A ∩ C_U_B = {x | 4 ≤ x ∧ x < 5} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l1144_114418


namespace NUMINAMATH_GPT_total_value_is_correct_l1144_114496

-- We will define functions that convert base 7 numbers to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let digits := (n.digits 7)
  digits.enum.foldr (λ ⟨i, d⟩ acc => acc + d * 7^i) 0

-- Define the specific numbers in base 7
def silver_value_base7 : Nat := 5326
def gemstone_value_base7 : Nat := 3461
def spice_value_base7 : Nat := 656

-- Define the combined total in base 10
def total_value_base10 : Nat := base7_to_base10 silver_value_base7 + base7_to_base10 gemstone_value_base7 + base7_to_base10 spice_value_base7

theorem total_value_is_correct :
  total_value_base10 = 3485 :=
by
  sorry

end NUMINAMATH_GPT_total_value_is_correct_l1144_114496


namespace NUMINAMATH_GPT_find_number_l1144_114429

theorem find_number (x q : ℕ) (h1 : x = 3 * q) (h2 : q + x + 3 = 63) : x = 45 :=
sorry

end NUMINAMATH_GPT_find_number_l1144_114429


namespace NUMINAMATH_GPT_batsman_average_after_17th_match_l1144_114441

theorem batsman_average_after_17th_match 
  (A : ℕ) 
  (h1 : (16 * A + 87) / 17 = A + 3) : 
  A + 3 = 39 := 
sorry

end NUMINAMATH_GPT_batsman_average_after_17th_match_l1144_114441


namespace NUMINAMATH_GPT_min_sum_m_n_l1144_114462

open Nat

theorem min_sum_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m * n - 2 * m - 3 * n - 20 = 0) : m + n = 20 :=
sorry

end NUMINAMATH_GPT_min_sum_m_n_l1144_114462


namespace NUMINAMATH_GPT_circle_tangent_radius_l1144_114476

noncomputable def R : ℝ := 4
noncomputable def r : ℝ := 3
noncomputable def O1O2 : ℝ := R + r
noncomputable def r_inscribed : ℝ := (R * r) / O1O2

theorem circle_tangent_radius :
  r_inscribed = (24 : ℝ) / 7 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_circle_tangent_radius_l1144_114476


namespace NUMINAMATH_GPT_exponent_multiplication_rule_l1144_114455

theorem exponent_multiplication_rule :
  3000 * (3000 ^ 3000) = 3000 ^ 3001 := 
by {
  sorry
}

end NUMINAMATH_GPT_exponent_multiplication_rule_l1144_114455


namespace NUMINAMATH_GPT_average_height_males_l1144_114456

theorem average_height_males
  (M W H_m : ℝ)
  (h₀ : W ≠ 0)
  (h₁ : M = 2 * W)
  (h₂ : (M * H_m + W * 170) / (M + W) = 180) :
  H_m = 185 := 
sorry

end NUMINAMATH_GPT_average_height_males_l1144_114456


namespace NUMINAMATH_GPT_pages_for_15_dollars_l1144_114479

theorem pages_for_15_dollars 
  (cpg : ℚ) -- cost per 5 pages in cents
  (budget : ℚ) -- budget in cents
  (h_cpg_pos : cpg = 7 * 1) -- 7 cents for 5 pages
  (h_budget_pos : budget = 1500 * 1) -- $15 = 1500 cents
  : (budget * (5 / cpg)).floor = 1071 :=
by {
  sorry
}

end NUMINAMATH_GPT_pages_for_15_dollars_l1144_114479


namespace NUMINAMATH_GPT_sum_of_roots_abs_eqn_zero_l1144_114474

theorem sum_of_roots_abs_eqn_zero (x : ℝ) (hx : |x|^2 - 4*|x| - 5 = 0) : (5 + (-5) = 0) :=
  sorry

end NUMINAMATH_GPT_sum_of_roots_abs_eqn_zero_l1144_114474


namespace NUMINAMATH_GPT_intersection_of_lines_l1144_114486

-- Definitions for the lines given by their equations
def line1 (x y : ℝ) : Prop := 5 * x - 3 * y = 9
def line2 (x y : ℝ) : Prop := x^2 + 4 * x - y = 10

-- The statement to prove
theorem intersection_of_lines :
  (line1 2 (1 / 3) ∧ line2 2 (1 / 3)) ∨ (line1 (-3.5) (-8.83) ∧ line2 (-3.5) (-8.83)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1144_114486


namespace NUMINAMATH_GPT_ratio_of_triangle_BFD_to_square_ABCE_l1144_114407

def is_square (ABCE : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a b c e : ℝ, ABCE a b c e → a = b ∧ b = c ∧ c = e

def ratio_of_areas (AF FE CD DE : ℝ) (ratio : ℝ) : Prop :=
  AF = 3 * FE ∧ CD = 3 * DE ∧ ratio = 1 / 2

theorem ratio_of_triangle_BFD_to_square_ABCE (AF FE CD DE ratio : ℝ) (ABCE : ℝ → ℝ → ℝ → ℝ → Prop)
  (h1 : is_square ABCE)
  (h2 : AF = 3 * FE) (h3 : CD = 3 * DE) : ratio_of_areas AF FE CD DE (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_BFD_to_square_ABCE_l1144_114407


namespace NUMINAMATH_GPT_find_a4_plus_b4_l1144_114417

-- Variables representing the given conditions
variables {a b : ℝ}

-- The theorem statement to prove
theorem find_a4_plus_b4 (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 56 :=
sorry

end NUMINAMATH_GPT_find_a4_plus_b4_l1144_114417


namespace NUMINAMATH_GPT_power_function_is_x_cubed_l1144_114468

/-- Define the power function and its property -/
def power_function (a : ℕ) (x : ℝ) : ℝ := x ^ a

/-- The given condition that the function passes through the point (3, 27) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 3 = 27

/-- Prove that the power function is x^3 -/
theorem power_function_is_x_cubed (f : ℝ → ℝ)
  (h : passes_through_point f) : 
  f = fun x => x ^ 3 := 
by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_power_function_is_x_cubed_l1144_114468


namespace NUMINAMATH_GPT_checkered_triangle_division_l1144_114497

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end NUMINAMATH_GPT_checkered_triangle_division_l1144_114497


namespace NUMINAMATH_GPT_mean_of_S_eq_651_l1144_114453

theorem mean_of_S_eq_651 
  (s n : ℝ) 
  (h1 : (s + 1) / (n + 1) = s / n - 13) 
  (h2 : (s + 2001) / (n + 1) = s / n + 27) 
  (hn : n ≠ 0) : s / n = 651 := 
by 
  sorry

end NUMINAMATH_GPT_mean_of_S_eq_651_l1144_114453


namespace NUMINAMATH_GPT_factor_of_quadratic_polynomial_l1144_114480

theorem factor_of_quadratic_polynomial (t : ℚ) :
  (8 * t^2 + 22 * t + 5 = 0) ↔ (t = -1/4) ∨ (t = -5/2) :=
by sorry

end NUMINAMATH_GPT_factor_of_quadratic_polynomial_l1144_114480


namespace NUMINAMATH_GPT_cos_120_eq_neg_half_l1144_114452

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_GPT_cos_120_eq_neg_half_l1144_114452


namespace NUMINAMATH_GPT_symmetric_line_equation_l1144_114464

theorem symmetric_line_equation (x y : ℝ) : 
  (y = 2 * x + 1) → (-y = 2 * (-x) + 1) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1144_114464


namespace NUMINAMATH_GPT_floor_eq_correct_l1144_114423

theorem floor_eq_correct (y : ℝ) (h : ⌊y⌋ + y = 17 / 4) : y = 9 / 4 :=
sorry

end NUMINAMATH_GPT_floor_eq_correct_l1144_114423


namespace NUMINAMATH_GPT_complement_intersection_l1144_114432

-- Definitions
def U : Set ℕ := {x | x ≤ 4 ∧ 0 < x}
def A : Set ℕ := {1, 4}
def B : Set ℕ := {2, 4}
def complement (s : Set ℕ) := {x | x ∈ U ∧ x ∉ s}

-- The theorem to prove
theorem complement_intersection :
  complement (A ∩ B) = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1144_114432


namespace NUMINAMATH_GPT_inequality_proof_l1144_114402

theorem inequality_proof {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (1 / Real.sqrt (1 + x^2)) + (1 / Real.sqrt (1 + y^2)) ≤ (2 / Real.sqrt (1 + x * y)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1144_114402


namespace NUMINAMATH_GPT_problem_l1144_114498

noncomputable def a : Real := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : Real := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : Real := Real.sqrt 6 / 2

theorem problem :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_problem_l1144_114498


namespace NUMINAMATH_GPT_quadrilateral_angle_W_l1144_114451

theorem quadrilateral_angle_W (W X Y Z : ℝ) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) 
  (sum_angles : W + X + Y + Z = 360) : 
  W = 1440 / 7 := by
sorry

end NUMINAMATH_GPT_quadrilateral_angle_W_l1144_114451


namespace NUMINAMATH_GPT_PR_length_right_triangle_l1144_114430

theorem PR_length_right_triangle
  (P Q R : Type)
  (cos_R : ℝ)
  (PQ PR : ℝ)
  (h1 : cos_R = 5 * Real.sqrt 34 / 34)
  (h2 : PQ = Real.sqrt 34)
  (h3 : cos_R = PR / PQ) : PR = 5 := by
  sorry

end NUMINAMATH_GPT_PR_length_right_triangle_l1144_114430


namespace NUMINAMATH_GPT_water_volume_correct_l1144_114458

noncomputable def volume_of_water : ℝ :=
  let r := 4
  let h := 9
  let d := 2
  48 * Real.pi - 36 * Real.sqrt 3

theorem water_volume_correct :
  volume_of_water = 48 * Real.pi - 36 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_water_volume_correct_l1144_114458


namespace NUMINAMATH_GPT_binary_calculation_l1144_114494

-- Binary arithmetic definition
def binary_mul (a b : Nat) : Nat := a * b
def binary_div (a b : Nat) : Nat := a / b

-- Binary numbers in Nat (representing binary literals by their decimal equivalent)
def b110010 := 50   -- 110010_2 in decimal
def b101000 := 40   -- 101000_2 in decimal
def b100 := 4       -- 100_2 in decimal
def b10 := 2        -- 10_2 in decimal
def b10111000 := 184-- 10111000_2 in decimal

theorem binary_calculation :
  binary_div (binary_div (binary_mul b110010 b101000) b100) b10 = b10111000 :=
by
  sorry

end NUMINAMATH_GPT_binary_calculation_l1144_114494


namespace NUMINAMATH_GPT_square_perimeter_ratio_l1144_114413

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end NUMINAMATH_GPT_square_perimeter_ratio_l1144_114413


namespace NUMINAMATH_GPT_range_of_a_l1144_114459

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (cubic_function x1 = a) ∧ (cubic_function x2 = a) ∧ (cubic_function x3 = a)) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1144_114459


namespace NUMINAMATH_GPT_additional_cost_l1144_114466

theorem additional_cost (original_bales : ℕ) (previous_cost_per_bale : ℕ) (better_quality_cost_per_bale : ℕ) :
  let new_bales := original_bales * 2
  let original_cost := original_bales * previous_cost_per_bale
  let new_cost := new_bales * better_quality_cost_per_bale
  new_cost - original_cost = 210 :=
by sorry

end NUMINAMATH_GPT_additional_cost_l1144_114466


namespace NUMINAMATH_GPT_log_proof_l1144_114440

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem log_proof (x : ℝ) (h : log_base 7 (x + 6) = 2) : log_base 13 x = log_base 13 43 :=
by
  sorry

end NUMINAMATH_GPT_log_proof_l1144_114440


namespace NUMINAMATH_GPT_mutually_exclusive_events_l1144_114467

-- Define the bag, balls, and events
def bag := (5, 3) -- (red balls, white balls)

def draws (r w : Nat) := (r + w = 3)

def event_A (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.1 = 3 -- At least one red ball and all red balls
def event_B (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 = 3 -- At least one red ball and all white balls
def event_C (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 ≥ 1 -- At least one red ball and at least one white ball
def event_D (draw : ℕ × ℕ) := (draw.1 = 1 ∨ draw.1 = 2) ∧ draws draw.1 draw.2 -- Exactly one red ball and exactly two red balls

theorem mutually_exclusive_events : 
  ∀ draw : ℕ × ℕ, 
  (event_A draw ∨ event_B draw ∨ event_C draw ∨ event_D draw) → 
  (event_D draw ↔ (draw.1 = 1 ∧ draw.2 = 2) ∨ (draw.1 = 2 ∧ draw.2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_events_l1144_114467


namespace NUMINAMATH_GPT_bowling_ball_weight_l1144_114427

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1144_114427


namespace NUMINAMATH_GPT_irrational_number_among_choices_l1144_114442

theorem irrational_number_among_choices : ∃ x ∈ ({17/6, -27/100, 0, Real.sqrt 2} : Set ℝ), Irrational x ∧ x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_irrational_number_among_choices_l1144_114442


namespace NUMINAMATH_GPT_area_new_rectangle_l1144_114401

-- Define the given rectangle's dimensions
def a : ℕ := 3
def b : ℕ := 4

-- Define the diagonal of the given rectangle
def d : ℕ := Nat.sqrt (a^2 + b^2)

-- Define the new rectangle's dimensions
def length_new : ℕ := d + a
def breadth_new : ℕ := d - b

-- The target area of the new rectangle
def area_new : ℕ := length_new * breadth_new

-- Prove that the area of the new rectangle is 8 square units
theorem area_new_rectangle (h : d = 5) : area_new = 8 := by
  -- Indicate that proof steps are not provided
  sorry

end NUMINAMATH_GPT_area_new_rectangle_l1144_114401


namespace NUMINAMATH_GPT_additional_cost_tv_ad_l1144_114411

theorem additional_cost_tv_ad (in_store_price : ℝ) (payment : ℝ) (shipping : ℝ) :
  in_store_price = 129.95 → payment = 29.99 → shipping = 14.95 → 
  (4 * payment + shipping - in_store_price) * 100 = 496 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_additional_cost_tv_ad_l1144_114411


namespace NUMINAMATH_GPT_arith_seq_fraction_l1144_114461

theorem arith_seq_fraction (a : ℕ → ℝ) (d : ℝ) (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d ≠ 0) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 :=
sorry

end NUMINAMATH_GPT_arith_seq_fraction_l1144_114461


namespace NUMINAMATH_GPT_ratio_accepted_to_rejected_l1144_114454

-- Let n be the total number of eggs processed per day
def eggs_per_day := 400

-- Let accepted_per_batch be the number of accepted eggs per batch
def accepted_per_batch := 96

-- Let rejected_per_batch be the number of rejected eggs per batch
def rejected_per_batch := 4

-- On a particular day, 12 additional eggs were accepted
def additional_accepted_eggs := 12

-- Normalize definitions to make our statements clearer
def accepted_batches := eggs_per_day / (accepted_per_batch + rejected_per_batch)
def normally_accepted_eggs := accepted_per_batch * accepted_batches
def normally_rejected_eggs := rejected_per_batch * accepted_batches
def total_accepted_eggs := normally_accepted_eggs + additional_accepted_eggs
def total_rejected_eggs := eggs_per_day - total_accepted_eggs

theorem ratio_accepted_to_rejected :
  (total_accepted_eggs / gcd total_accepted_eggs total_rejected_eggs) = 99 ∧
  (total_rejected_eggs / gcd total_accepted_eggs total_rejected_eggs) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_accepted_to_rejected_l1144_114454


namespace NUMINAMATH_GPT_total_cost_of_cable_l1144_114408

-- Defining the conditions as constants
def east_west_streets := 18
def east_west_length := 2
def north_south_streets := 10
def north_south_length := 4
def cable_per_mile_street := 5
def cost_per_mile_cable := 2000

-- The theorem contains the problem statement and asserts the answer
theorem total_cost_of_cable :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) * cable_per_mile_street * cost_per_mile_cable = 760000 := 
  sorry

end NUMINAMATH_GPT_total_cost_of_cable_l1144_114408


namespace NUMINAMATH_GPT_tax_rate_correct_l1144_114435

def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 89.6

def taxable_amount : ℝ := total_value - non_taxable_amount

theorem tax_rate_correct : (tax_paid / taxable_amount) * 100 = 8 := by
  sorry

end NUMINAMATH_GPT_tax_rate_correct_l1144_114435


namespace NUMINAMATH_GPT_brother_pays_correct_amount_l1144_114499

-- Definition of constants and variables
def friend_per_day := 5
def cousin_per_day := 4
def total_amount_collected := 119
def days := 7
def brother_per_day := 8

-- Statement of the theorem to be proven
theorem brother_pays_correct_amount :
  friend_per_day * days + cousin_per_day * days + brother_per_day * days = total_amount_collected :=
by {
  sorry
}

end NUMINAMATH_GPT_brother_pays_correct_amount_l1144_114499


namespace NUMINAMATH_GPT_book_cost_l1144_114426

theorem book_cost (x y : ℝ) (h₁ : 2 * y = x) (h₂ : 100 + y = x - 100) : x = 200 := by
  sorry

end NUMINAMATH_GPT_book_cost_l1144_114426


namespace NUMINAMATH_GPT_solve_for_x_l1144_114482

theorem solve_for_x : ∃ x : ℤ, 25 - (4 + 3) = 5 + x ∧ x = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1144_114482


namespace NUMINAMATH_GPT_square_of_hypotenuse_product_eq_160_l1144_114460

noncomputable def square_of_product_of_hypotenuses (x y : ℝ) (h1 h2 : ℝ) : ℝ :=
  (h1 * h2) ^ 2

theorem square_of_hypotenuse_product_eq_160 :
  ∀ (x y h1 h2 : ℝ),
    (1 / 2) * x * (2 * y) = 4 →
    (1 / 2) * x * y = 8 →
    x^2 + (2 * y)^2 = h1^2 →
    x^2 + y^2 = h2^2 →
    square_of_product_of_hypotenuses x y h1 h2 = 160 :=
by
  intros x y h1 h2 area1 area2 pythagorean1 pythagorean2
  -- The detailed proof steps would go here
  sorry

end NUMINAMATH_GPT_square_of_hypotenuse_product_eq_160_l1144_114460


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1144_114436

theorem solution_set_of_inequality {x : ℝ} :
  {x | |x| * (1 - 2 * x) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x < 1 / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1144_114436


namespace NUMINAMATH_GPT_find_y_value_l1144_114472

theorem find_y_value (k c x y : ℝ) (h1 : c = 3) 
                     (h2 : ∀ x : ℝ, y = k * x + c)
                     (h3 : ∃ k : ℝ, 15 = k * 5 + 3) :
  y = -21 :=
by 
  sorry

end NUMINAMATH_GPT_find_y_value_l1144_114472


namespace NUMINAMATH_GPT_cost_of_peaches_eq_2_per_pound_l1144_114412

def initial_money : ℕ := 20
def after_buying_peaches : ℕ := 14
def pounds_of_peaches : ℕ := 3
def cost_per_pound : ℕ := 2

theorem cost_of_peaches_eq_2_per_pound (h: initial_money - after_buying_peaches = pounds_of_peaches * cost_per_pound) :
  cost_per_pound = 2 := by
  sorry

end NUMINAMATH_GPT_cost_of_peaches_eq_2_per_pound_l1144_114412


namespace NUMINAMATH_GPT_major_premise_is_wrong_l1144_114487

-- Definitions of the conditions
def line_parallel_to_plane (l : Type) (p : Type) : Prop := sorry
def line_contained_in_plane (l : Type) (p : Type) : Prop := sorry

-- Stating the main problem: the major premise is wrong
theorem major_premise_is_wrong :
  ∀ (a b : Type) (α : Type), line_contained_in_plane a α → line_parallel_to_plane b α → ¬ (line_parallel_to_plane b a) := 
by 
  intros a b α h1 h2
  sorry

end NUMINAMATH_GPT_major_premise_is_wrong_l1144_114487


namespace NUMINAMATH_GPT_complex_values_l1144_114485

open Complex

theorem complex_values (z : ℂ) (h : z ^ 3 + z = 2 * (abs z) ^ 2) :
  z = 0 ∨ z = 1 ∨ z = -1 + 2 * Complex.I ∨ z = -1 - 2 * Complex.I :=
by sorry

end NUMINAMATH_GPT_complex_values_l1144_114485


namespace NUMINAMATH_GPT_exists_equilateral_triangle_same_color_l1144_114422

-- Define a type for colors
inductive Color
| red : Color
| blue : Color

-- Define our statement
-- Given each point in the plane is colored either red or blue,
-- there exists an equilateral triangle with vertices of the same color.
theorem exists_equilateral_triangle_same_color (coloring : ℝ × ℝ → Color) : 
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ 
    dist p₁ p₂ = dist p₂ p₃ ∧ dist p₂ p₃ = dist p₃ p₁ ∧ 
    (coloring p₁ = coloring p₂ ∧ coloring p₂ = coloring p₃) :=
by
  sorry

end NUMINAMATH_GPT_exists_equilateral_triangle_same_color_l1144_114422


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1144_114491

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : S n = n * (a 1 + a n) / 2)
  (h3 : a 2 + a 5 + a 11 = 6) :
  S 11 = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1144_114491


namespace NUMINAMATH_GPT_original_phone_number_eq_l1144_114425

theorem original_phone_number_eq :
  ∃ (a b c d e f : ℕ), 
    (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 282500) ∧
    (1000000 * 2 + 100000 * a + 10000 * 8 + 1000 * b + 100 * c + 10 * d + e = 81 * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) ∧
    (0 ≤ a ∧ a ≤ 9) ∧
    (0 ≤ b ∧ b ≤ 9) ∧
    (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧
    (0 ≤ e ∧ e ≤ 9) ∧
    (0 ≤ f ∧ f ≤ 9) :=
sorry

end NUMINAMATH_GPT_original_phone_number_eq_l1144_114425


namespace NUMINAMATH_GPT_negation_example_l1144_114433

theorem negation_example : (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1144_114433


namespace NUMINAMATH_GPT_exists_x_l1144_114469

noncomputable def g (x : ℝ) : ℝ := (2 / 7) ^ x + (3 / 7) ^ x + (6 / 7) ^ x

theorem exists_x (x : ℝ) : ∃ c : ℝ, g c = 1 :=
sorry

end NUMINAMATH_GPT_exists_x_l1144_114469


namespace NUMINAMATH_GPT_range_of_a_l1144_114400

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) →
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1144_114400


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1144_114446

theorem quadratic_two_distinct_real_roots 
    (a b c : ℝ)
    (h1 : a > 0)
    (h2 : c < 0) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + bx + c = 0 := 
sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1144_114446


namespace NUMINAMATH_GPT_west_move_7m_l1144_114448

-- Definitions and conditions
def east_move (distance : Int) : Int := distance -- Moving east
def west_move (distance : Int) : Int := -distance -- Moving west is represented as negative

-- Problem: Prove that moving west by 7m is denoted by -7m given the conditions.
theorem west_move_7m : west_move 7 = -7 :=
by
  -- Proof will be handled here normally, but it's omitted as per instruction
  sorry

end NUMINAMATH_GPT_west_move_7m_l1144_114448


namespace NUMINAMATH_GPT_skee_ball_tickets_l1144_114434

-- Represent the given conditions as Lean definitions
def whack_a_mole_tickets : ℕ := 33
def candy_cost_per_piece : ℕ := 6
def candies_bought : ℕ := 7
def total_candy_tickets : ℕ := candies_bought * candy_cost_per_piece

-- Goal: Prove the number of tickets won playing 'skee ball'
theorem skee_ball_tickets (h : 42 = total_candy_tickets): whack_a_mole_tickets + 9 = total_candy_tickets :=
by {
  sorry
}

end NUMINAMATH_GPT_skee_ball_tickets_l1144_114434


namespace NUMINAMATH_GPT_largest_of_eight_consecutive_summing_to_5400_l1144_114438

theorem largest_of_eight_consecutive_summing_to_5400 :
  ∃ (n : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 5400)
  → (n+7 = 678) :=
by 
  sorry

end NUMINAMATH_GPT_largest_of_eight_consecutive_summing_to_5400_l1144_114438


namespace NUMINAMATH_GPT_cos_330_eq_sqrt3_div_2_l1144_114473

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_330_eq_sqrt3_div_2_l1144_114473


namespace NUMINAMATH_GPT_parabola_coefficients_l1144_114419

theorem parabola_coefficients :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, (a * (x - 3)^2 + 2 = 0 → (x = 1) ∧ (a * (1 - 3)^2 + 2 = 0))
    ∧ (a = -1/2 ∧ b = 3 ∧ c = -5/2)) 
    ∧ (∀ x : ℝ, a * x^2 + b * x + c = - 1 / 2 * x^2 + 3 * x - 5 / 2) :=
sorry

end NUMINAMATH_GPT_parabola_coefficients_l1144_114419


namespace NUMINAMATH_GPT_lisa_ratio_l1144_114463

theorem lisa_ratio (L J T : ℝ) 
  (h1 : L + J + T = 60) 
  (h2 : T = L / 2) 
  (h3 : L = T + 15) : 
  L / 60 = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_lisa_ratio_l1144_114463


namespace NUMINAMATH_GPT_correct_option_l1144_114493

theorem correct_option 
  (A_false : ¬ (-6 - (-9)) = -3)
  (B_false : ¬ (-2 * (-5)) = -7)
  (C_false : ¬ (-x^2 + 3 * x^2) = 2)
  (D_true : (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b) :
  (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b :=
by sorry

end NUMINAMATH_GPT_correct_option_l1144_114493


namespace NUMINAMATH_GPT_lending_rate_is_8_percent_l1144_114409

-- Define all given conditions.
def principal₁ : ℝ := 5000
def time₁ : ℝ := 2
def rate₁ : ℝ := 4  -- in percentage
def gain_per_year : ℝ := 200

-- Prove that the interest rate for lending is 8%
theorem lending_rate_is_8_percent :
  ∃ (rate₂ : ℝ), rate₂ = 8 :=
by
  let interest₁ := principal₁ * rate₁ * time₁ / 100
  let interest_per_year₁ := interest₁ / time₁
  let total_interest_received_per_year := gain_per_year + interest_per_year₁
  let rate₂ := (total_interest_received_per_year * 100) / principal₁
  use rate₂
  sorry

end NUMINAMATH_GPT_lending_rate_is_8_percent_l1144_114409


namespace NUMINAMATH_GPT_conor_vegetables_per_week_l1144_114470

theorem conor_vegetables_per_week : 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := 
by 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  show (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 
  sorry

end NUMINAMATH_GPT_conor_vegetables_per_week_l1144_114470


namespace NUMINAMATH_GPT_loaf_slices_l1144_114489

theorem loaf_slices (S : ℕ) (T : ℕ) : 
  (S - 7 = 2 * T + 3) ∧ (S ≥ 20) → S = 20 :=
by
  sorry

end NUMINAMATH_GPT_loaf_slices_l1144_114489


namespace NUMINAMATH_GPT_sequence_sum_l1144_114449

theorem sequence_sum (A B C D E F G H I J : ℤ)
  (h1 : D = 7)
  (h2 : A + B + C = 24)
  (h3 : B + C + D = 24)
  (h4 : C + D + E = 24)
  (h5 : D + E + F = 24)
  (h6 : E + F + G = 24)
  (h7 : F + G + H = 24)
  (h8 : G + H + I = 24)
  (h9 : H + I + J = 24) : 
  A + J = 105 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l1144_114449


namespace NUMINAMATH_GPT_not_recurring_decimal_l1144_114405

-- Definitions based on the provided conditions
def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ d m n : ℕ, d ≠ 0 ∧ (x * d) % 10 ^ n = m

-- Condition: 0.89898989
def number_0_89898989 : ℝ := 0.89898989

-- Proof statement to show 0.89898989 is not a recurring decimal
theorem not_recurring_decimal : ¬ is_recurring_decimal number_0_89898989 :=
sorry

end NUMINAMATH_GPT_not_recurring_decimal_l1144_114405


namespace NUMINAMATH_GPT_four_digit_number_exists_l1144_114431

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), A = B / 3 ∧ C = A + B ∧ D = 3 * B ∧
  A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1349) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_exists_l1144_114431


namespace NUMINAMATH_GPT_gamma_bank_min_savings_l1144_114421

def total_airfare_cost : ℕ := 10200 * 2 * 3
def total_hotel_cost : ℕ := 6500 * 12
def total_food_cost : ℕ := 1000 * 14 * 3
def total_excursion_cost : ℕ := 20000
def total_expenses : ℕ := total_airfare_cost + total_hotel_cost + total_food_cost + total_excursion_cost
def initial_amount_available : ℕ := 150000

def annual_rate_rebs : ℝ := 0.036
def annual_rate_gamma : ℝ := 0.045
def annual_rate_tisi : ℝ := 0.0312
def monthly_rate_btv : ℝ := 0.0025

noncomputable def compounded_amount_rebs : ℝ :=
  initial_amount_available * (1 + annual_rate_rebs / 12) ^ 6

noncomputable def compounded_amount_gamma : ℝ :=
  initial_amount_available * (1 + annual_rate_gamma / 2)

noncomputable def compounded_amount_tisi : ℝ :=
  initial_amount_available * (1 + annual_rate_tisi / 4) ^ 2

noncomputable def compounded_amount_btv : ℝ :=
  initial_amount_available * (1 + monthly_rate_btv) ^ 6

noncomputable def interest_rebs : ℝ := compounded_amount_rebs - initial_amount_available
noncomputable def interest_gamma : ℝ := compounded_amount_gamma - initial_amount_available
noncomputable def interest_tisi : ℝ := compounded_amount_tisi - initial_amount_available
noncomputable def interest_btv : ℝ := compounded_amount_btv - initial_amount_available

noncomputable def amount_needed_from_salary_rebs : ℝ :=
  total_expenses - initial_amount_available - interest_rebs

noncomputable def amount_needed_from_salary_gamma : ℝ :=
  total_expenses - initial_amount_available - interest_gamma

noncomputable def amount_needed_from_salary_tisi : ℝ :=
  total_expenses - initial_amount_available - interest_tisi

noncomputable def amount_needed_from_salary_btv : ℝ :=
  total_expenses - initial_amount_available - interest_btv

theorem gamma_bank_min_savings :
  amount_needed_from_salary_gamma = 47825.00 ∧
  amount_needed_from_salary_rebs = 48479.67 ∧
  amount_needed_from_salary_tisi = 48850.87 ∧
  amount_needed_from_salary_btv = 48935.89 :=
by sorry

end NUMINAMATH_GPT_gamma_bank_min_savings_l1144_114421


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l1144_114484

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l1144_114484


namespace NUMINAMATH_GPT_sum_three_numbers_l1144_114478

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = a + 20) 
  (h2 : (a + b + c) / 3 = c - 30) 
  (h3 : b = 10) :
  sum_of_three_numbers a b c = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_three_numbers_l1144_114478


namespace NUMINAMATH_GPT_point_on_line_eq_l1144_114450

theorem point_on_line_eq (a b : ℝ) (h : b = -3 * a - 4) : b + 3 * a + 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_eq_l1144_114450


namespace NUMINAMATH_GPT_count_triangles_l1144_114465

-- Define the problem conditions
def num_small_triangles : ℕ := 11
def num_medium_triangles : ℕ := 4
def num_large_triangles : ℕ := 1

-- Define the main statement asserting the total number of triangles
theorem count_triangles (small : ℕ) (medium : ℕ) (large : ℕ) :
  small = num_small_triangles →
  medium = num_medium_triangles →
  large = num_large_triangles →
  small + medium + large = 16 :=
by
  intros h_small h_medium h_large
  rw [h_small, h_medium, h_large]
  sorry

end NUMINAMATH_GPT_count_triangles_l1144_114465


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1144_114424

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 4) (h2 : a 4 = 2) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 6 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1144_114424


namespace NUMINAMATH_GPT_fraction_shaded_area_l1144_114495

theorem fraction_shaded_area (PX XQ : ℝ) (PA PR PQ : ℝ) (h1 : PX = 1) (h2 : 3 * XQ = PX) (h3 : PQ = PR) (h4 : PA = 1) (h5 : PA + AR = PR) (h6 : PR = 4):
  (3 / 16 : ℝ) = 0.375 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_fraction_shaded_area_l1144_114495


namespace NUMINAMATH_GPT_find_natural_numbers_l1144_114437

def LCM (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

theorem find_natural_numbers :
  ∃ a b : ℕ, a + b = 54 ∧ LCM a b - Nat.gcd a b = 114 ∧ (a = 24 ∧ b = 30 ∨ a = 30 ∧ b = 24) := by {
  sorry
}

end NUMINAMATH_GPT_find_natural_numbers_l1144_114437


namespace NUMINAMATH_GPT_addition_example_l1144_114492

theorem addition_example : 300 + 2020 + 10001 = 12321 := 
by 
  sorry

end NUMINAMATH_GPT_addition_example_l1144_114492


namespace NUMINAMATH_GPT_pastries_and_cost_correct_l1144_114490

def num_pastries_lola := 13 + 10 + 8 + 6
def cost_lola := 13 * 0.50 + 10 * 1.00 + 8 * 3.00 + 6 * 2.00

def num_pastries_lulu := 16 + 12 + 14 + 9
def cost_lulu := 16 * 0.50 + 12 * 1.00 + 14 * 3.00 + 9 * 2.00

def num_pastries_lila := 22 + 15 + 10 + 12
def cost_lila := 22 * 0.50 + 15 * 1.00 + 10 * 3.00 + 12 * 2.00

def num_pastries_luka := 18 + 20 + 7 + 14 + 25
def cost_luka := 18 * 0.50 + 20 * 1.00 + 7 * 3.00 + 14 * 2.00 + 25 * 1.50

def total_pastries := num_pastries_lola + num_pastries_lulu + num_pastries_lila + num_pastries_luka
def total_cost := cost_lola + cost_lulu + cost_lila + cost_luka

theorem pastries_and_cost_correct :
  total_pastries = 231 ∧ total_cost = 328.00 :=
by
  sorry

end NUMINAMATH_GPT_pastries_and_cost_correct_l1144_114490


namespace NUMINAMATH_GPT_distance_ratio_l1144_114475

-- Defining the conditions
def speedA : ℝ := 50 -- Speed of Car A in km/hr
def timeA : ℝ := 6 -- Time taken by Car A in hours

def speedB : ℝ := 100 -- Speed of Car B in km/hr
def timeB : ℝ := 1 -- Time taken by Car B in hours

-- Calculating the distances
def distanceA : ℝ := speedA * timeA -- Distance covered by Car A
def distanceB : ℝ := speedB * timeB -- Distance covered by Car B

-- Statement to prove the ratio of distances
theorem distance_ratio : (distanceA / distanceB) = 3 :=
by
  -- Calculations here might be needed, but we use sorry to indicate proof is pending
  sorry

end NUMINAMATH_GPT_distance_ratio_l1144_114475


namespace NUMINAMATH_GPT_find_f_minus3_and_f_2009_l1144_114406

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Conditions
axiom h1 : is_odd f
axiom h2 : f 1 = 2
axiom h3 : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Questions
theorem find_f_minus3_and_f_2009 : f (-3) = 0 ∧ f 2009 = -2 :=
by 
  sorry

end NUMINAMATH_GPT_find_f_minus3_and_f_2009_l1144_114406


namespace NUMINAMATH_GPT_sin_identity_l1144_114457

variable (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 3 / 2)

theorem sin_identity : Real.sin (3 * Real.pi / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_identity_l1144_114457


namespace NUMINAMATH_GPT_math_problem_equivalence_l1144_114445

theorem math_problem_equivalence :
  (-3 : ℚ) / (-1 - 3 / 4) * (3 / 4) / (3 / 7) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_equivalence_l1144_114445


namespace NUMINAMATH_GPT_domain_of_sqrt_l1144_114444

theorem domain_of_sqrt (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_l1144_114444


namespace NUMINAMATH_GPT_problem_statement_l1144_114410

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C D : V)
  (BC CD : V)
  (AC AB AD : V)

theorem problem_statement
  (h1 : BC = 2 • CD)
  (h2 : BC = AC - AB) :
  AD = (3 / 2 : ℝ) • AC - (1 / 2 : ℝ) • AB :=
sorry

end NUMINAMATH_GPT_problem_statement_l1144_114410
