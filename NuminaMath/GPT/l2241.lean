import Mathlib

namespace NUMINAMATH_GPT_smallest_value_a2_b2_c2_l2241_224143

theorem smallest_value_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 4 * c = 120) : 
  a^2 + b^2 + c^2 ≥ 14400 / 29 :=
by sorry

end NUMINAMATH_GPT_smallest_value_a2_b2_c2_l2241_224143


namespace NUMINAMATH_GPT_candy_problem_l2241_224111

variable (total_pieces_eaten : ℕ) (pieces_from_sister : ℕ) (pieces_from_neighbors : ℕ)

theorem candy_problem
  (h1 : total_pieces_eaten = 18)
  (h2 : pieces_from_sister = 13)
  (h3 : total_pieces_eaten = pieces_from_sister + pieces_from_neighbors) :
  pieces_from_neighbors = 5 := by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_candy_problem_l2241_224111


namespace NUMINAMATH_GPT_pure_alcohol_added_l2241_224130

theorem pure_alcohol_added (x : ℝ) (h1 : 6 * 0.40 = 2.4)
    (h2 : (2.4 + x) / (6 + x) = 0.50) : x = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_pure_alcohol_added_l2241_224130


namespace NUMINAMATH_GPT_compare_f_m_plus_2_l2241_224191

theorem compare_f_m_plus_2 (a : ℝ) (ha : a > 0) (m : ℝ) 
  (hf : (a * m^2 + 2 * a * m + 1) < 0) : 
  (a * (m + 2)^2 + 2 * a * (m + 2) + 1) > 1 :=
sorry

end NUMINAMATH_GPT_compare_f_m_plus_2_l2241_224191


namespace NUMINAMATH_GPT_q_at_1_is_zero_l2241_224150

-- Define the function q : ℝ → ℝ
-- The conditions imply q(1) = 0
axiom q : ℝ → ℝ

-- Given that (1, 0) is on the graph of y = q(x)
axiom q_condition : q 1 = 0

-- Prove q(1) = 0 given the condition that (1, 0) is on the graph
theorem q_at_1_is_zero : q 1 = 0 :=
by
  exact q_condition

end NUMINAMATH_GPT_q_at_1_is_zero_l2241_224150


namespace NUMINAMATH_GPT_youseff_blocks_l2241_224152

theorem youseff_blocks (x : ℕ) 
  (H1 : (1 : ℚ) * x = (1/3 : ℚ) * x + 8) : 
  x = 12 := 
sorry

end NUMINAMATH_GPT_youseff_blocks_l2241_224152


namespace NUMINAMATH_GPT_simplify_to_linear_binomial_l2241_224193

theorem simplify_to_linear_binomial (k : ℝ) (x : ℝ) : 
  (-3 * k * x^2 + x - 1) + (9 * x^2 - 4 * k * x + 3 * k) = 
  (1 - 4 * k) * x + (3 * k - 1) → 
  k = 3 := by
  sorry

end NUMINAMATH_GPT_simplify_to_linear_binomial_l2241_224193


namespace NUMINAMATH_GPT_min_value_of_expression_l2241_224195

theorem min_value_of_expression : ∀ x : ℝ, ∃ (M : ℝ), (∀ x, 16^x - 4^x - 4^(x+1) + 3 ≥ M) ∧ M = -4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2241_224195


namespace NUMINAMATH_GPT_negation_of_prop_p_l2241_224176

open Classical

theorem negation_of_prop_p:
  (¬ ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≤ 1 / 2) ↔ ∃ x : ℕ, x > 0 ∧ (1 / 2) ^ x > 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_p_l2241_224176


namespace NUMINAMATH_GPT_gasoline_tank_capacity_l2241_224157

-- Given conditions
def initial_fraction_full := 5 / 6
def used_gallons := 15
def final_fraction_full := 2 / 3

-- Mathematical problem statement in Lean 4
theorem gasoline_tank_capacity (x : ℝ)
  (initial_full : initial_fraction_full * x = 5 / 6 * x)
  (final_full : initial_fraction_full * x - used_gallons = final_fraction_full * x) :
  x = 90 := by
  sorry

end NUMINAMATH_GPT_gasoline_tank_capacity_l2241_224157


namespace NUMINAMATH_GPT_total_cost_correct_l2241_224161

-- Define the conditions
def total_employees : ℕ := 300
def emp_12_per_hour : ℕ := 200
def emp_14_per_hour : ℕ := 40
def emp_17_per_hour : ℕ := total_employees - emp_12_per_hour - emp_14_per_hour

def wage_12_per_hour : ℕ := 12
def wage_14_per_hour : ℕ := 14
def wage_17_per_hour : ℕ := 17

def hours_per_shift : ℕ := 8

-- Define the cost calculations
def cost_12 : ℕ := emp_12_per_hour * wage_12_per_hour * hours_per_shift
def cost_14 : ℕ := emp_14_per_hour * wage_14_per_hour * hours_per_shift
def cost_17 : ℕ := emp_17_per_hour * wage_17_per_hour * hours_per_shift

def total_cost : ℕ := cost_12 + cost_14 + cost_17

-- The theorem to be proved
theorem total_cost_correct :
  total_cost = 31840 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l2241_224161


namespace NUMINAMATH_GPT_tan_half_angles_l2241_224197

theorem tan_half_angles (a b : ℝ) (ha : 3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0) :
  ∃ z : ℝ, z = Real.tan (a / 2) * Real.tan (b / 2) ∧ (z = Real.sqrt (6 / 13) ∨ z = -Real.sqrt (6 / 13)) :=
by
  sorry

end NUMINAMATH_GPT_tan_half_angles_l2241_224197


namespace NUMINAMATH_GPT_probability_exceeds_175_l2241_224115

theorem probability_exceeds_175 (P_lt_160 : ℝ) (P_160_to_175 : ℝ) (h : ℝ) :
  P_lt_160 = 0.2 → P_160_to_175 = 0.5 → 1 - P_lt_160 - P_160_to_175 = 0.3 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_probability_exceeds_175_l2241_224115


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2241_224103

theorem sufficient_not_necessary_condition (x : ℝ) : (x ≥ 3 → (x - 2) ≥ 0) ∧ ((x - 2) ≥ 0 → x ≥ 3) = false :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2241_224103


namespace NUMINAMATH_GPT_roots_quadratic_sum_l2241_224144

theorem roots_quadratic_sum (a b : ℝ) (h1 : (-2) + (-(1/4)) = -b/a)
  (h2 : -2 * (-(1/4)) = -2/a) : a + b = -13 := by
  sorry

end NUMINAMATH_GPT_roots_quadratic_sum_l2241_224144


namespace NUMINAMATH_GPT_combined_length_of_legs_is_ten_l2241_224177

-- Define the conditions given in the problem.
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * Real.sqrt 2

def hypotenuse_length (c : ℝ) : Prop :=
  c = 7.0710678118654755

def perimeter_condition (a b c perimeter : ℝ) : Prop :=
  perimeter = a + b + c ∧ perimeter = 10 + c

-- Prove the combined length of the two legs is 10.
theorem combined_length_of_legs_is_ten :
  ∃ (a b c : ℝ), is_isosceles_right_triangle a b c →
  hypotenuse_length c →
  ∀ perimeter : ℝ, perimeter_condition a b c perimeter →
  2 * a = 10 :=
by
  sorry

end NUMINAMATH_GPT_combined_length_of_legs_is_ten_l2241_224177


namespace NUMINAMATH_GPT_opposite_face_of_lime_is_black_l2241_224117

-- Define the colors
inductive Color
| P | C | M | S | K | L

-- Define the problem conditions
def face_opposite (c : Color) : Color := sorry

-- Theorem statement
theorem opposite_face_of_lime_is_black : face_opposite Color.L = Color.K := sorry

end NUMINAMATH_GPT_opposite_face_of_lime_is_black_l2241_224117


namespace NUMINAMATH_GPT_original_number_of_members_l2241_224158

-- Define the initial conditions
variables (x y : ℕ)

-- First condition: if five 9-year-old members leave
def condition1 : Prop := x * y - 45 = (y + 1) * (x - 5)

-- Second condition: if five 17-year-old members join
def condition2 : Prop := x * y + 85 = (y + 1) * (x + 5)

-- The theorem to be proven
theorem original_number_of_members (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 :=
by sorry

end NUMINAMATH_GPT_original_number_of_members_l2241_224158


namespace NUMINAMATH_GPT_root_increases_implies_m_neg7_l2241_224139

theorem root_increases_implies_m_neg7 
  (m : ℝ) 
  (h : ∃ x : ℝ, x ≠ 3 ∧ x = -m - 4 → x = 3) 
  : m = -7 := by
  sorry

end NUMINAMATH_GPT_root_increases_implies_m_neg7_l2241_224139


namespace NUMINAMATH_GPT_range_of_a_given_quadratic_condition_l2241_224171

theorem range_of_a_given_quadratic_condition:
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 3 * a * x + 9 ≥ 0) → (-2 ≤ a ∧ a ≤ 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_given_quadratic_condition_l2241_224171


namespace NUMINAMATH_GPT_slope_probability_l2241_224184

def line_equation (a x y : ℝ) : Prop := a * x + 2 * y - 3 = 0

def in_interval (a : ℝ) : Prop := -5 ≤ a ∧ a ≤ 4

def slope_not_less_than_1 (a : ℝ) : Prop := - a / 2 ≥ 1

noncomputable def probability_slope_not_less_than_1 : ℝ :=
  (2 - (-5)) / (4 - (-5))

theorem slope_probability :
  ∀ (a : ℝ), in_interval a → slope_not_less_than_1 a → probability_slope_not_less_than_1 = 1 / 3 :=
by
  intros a h_in h_slope
  sorry

end NUMINAMATH_GPT_slope_probability_l2241_224184


namespace NUMINAMATH_GPT_probability_dmitry_before_anatoly_l2241_224136

theorem probability_dmitry_before_anatoly (m : ℝ) (non_neg_m : 0 < m) :
  let volume_prism := (m^3) / 2
  let volume_tetrahedron := (m^3) / 3
  let probability := volume_tetrahedron / volume_prism
  probability = (2 : ℝ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_dmitry_before_anatoly_l2241_224136


namespace NUMINAMATH_GPT_alex_singles_percentage_l2241_224198

theorem alex_singles_percentage (total_hits home_runs triples doubles: ℕ) 
  (h1 : total_hits = 50) 
  (h2 : home_runs = 2) 
  (h3 : triples = 3) 
  (h4 : doubles = 10) :
  ((total_hits - (home_runs + triples + doubles)) / total_hits : ℚ) * 100 = 70 := 
by
  sorry

end NUMINAMATH_GPT_alex_singles_percentage_l2241_224198


namespace NUMINAMATH_GPT_positive_divisors_840_multiple_of_4_l2241_224138

theorem positive_divisors_840_multiple_of_4 :
  let n := 840
  let prime_factors := (2^3 * 3^1 * 5^1 * 7^1)
  (∀ k : ℕ, k ∣ n → k % 4 = 0 → ∀ a b c d : ℕ, 2 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 →
  k = 2^a * 3^b * 5^c * 7^d) → 
  (∃ count, count = 16) :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_divisors_840_multiple_of_4_l2241_224138


namespace NUMINAMATH_GPT_problem_inequality_l2241_224110

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

axiom f_pos : ∀ x : ℝ, x > 0 → f x > 0

axiom f_increasing : ∀ x y : ℝ, x > 0 → y > 0 → x ≤ y → (f x / x) ≤ (f y / y)

theorem problem_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
    3 * (f a + f b + f c) / (a + b + c) + (f a / a + f b / b + f c / c) :=
sorry

end NUMINAMATH_GPT_problem_inequality_l2241_224110


namespace NUMINAMATH_GPT_choir_members_correct_l2241_224101

def choir_members_condition (n : ℕ) : Prop :=
  150 < n ∧ n < 250 ∧ 
  n % 3 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3

theorem choir_members_correct : ∃ n, choir_members_condition n ∧ (n = 195 ∨ n = 219) :=
by {
  sorry
}

end NUMINAMATH_GPT_choir_members_correct_l2241_224101


namespace NUMINAMATH_GPT_cone_base_radius_l2241_224192

theorem cone_base_radius (slant_height : ℝ) (central_angle_deg : ℝ) (r : ℝ) 
  (h1 : slant_height = 6) 
  (h2 : central_angle_deg = 120) 
  (h3 : 2 * π * slant_height * (central_angle_deg / 360) = 4 * π) 
  : r = 2 := by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l2241_224192


namespace NUMINAMATH_GPT_abs_neg_two_l2241_224141

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end NUMINAMATH_GPT_abs_neg_two_l2241_224141


namespace NUMINAMATH_GPT_sequence_a2017_l2241_224180

theorem sequence_a2017 (a : ℕ → ℚ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n / (3 * a n + 2)) :
  a 2017 = 1 / 3026 :=
sorry

end NUMINAMATH_GPT_sequence_a2017_l2241_224180


namespace NUMINAMATH_GPT_meetings_percentage_l2241_224116

/-- Define the total work day in hours -/
def total_work_day_hours : ℕ := 10

/-- Define the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60 -- 1 hour = 60 minutes

/-- Define the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Define the break duration in minutes -/
def break_minutes : ℕ := 30

/-- Define the effective work minutes -/
def effective_work_minutes : ℕ := (total_work_day_hours * 60) - break_minutes

/-- Define the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- The percentage of the effective work day spent in meetings -/
def percent_meetings : ℕ := (total_meeting_minutes * 100) / effective_work_minutes

theorem meetings_percentage : percent_meetings = 24 := by
  sorry

end NUMINAMATH_GPT_meetings_percentage_l2241_224116


namespace NUMINAMATH_GPT_prove_positive_a_l2241_224105

variable (a b c n : ℤ)
variable (p : ℤ → ℤ)

-- Conditions given in the problem
def quadratic_polynomial (x : ℤ) : ℤ := a*x^2 + b*x + c

def condition_1 : Prop := a ≠ 0
def condition_2 : Prop := n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))

-- Proof goal
theorem prove_positive_a (h1 : a ≠ 0) (h2 : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
by
  sorry

end NUMINAMATH_GPT_prove_positive_a_l2241_224105


namespace NUMINAMATH_GPT_remainder_calculation_l2241_224182

theorem remainder_calculation : 
  ∀ (dividend divisor quotient remainder : ℕ), 
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 :=
by
  intros dividend divisor quotient remainder hdividend hdivisor hquotient heq
  sorry

end NUMINAMATH_GPT_remainder_calculation_l2241_224182


namespace NUMINAMATH_GPT_train_speed_including_stoppages_l2241_224196

theorem train_speed_including_stoppages (s : ℝ) (t : ℝ) (running_time_fraction : ℝ) :
  s = 48 ∧ t = 1/4 ∧ running_time_fraction = (1 - t) → (s * running_time_fraction = 36) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_including_stoppages_l2241_224196


namespace NUMINAMATH_GPT_hyperbola_equiv_l2241_224145

-- The existing hyperbola
def hyperbola1 (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- The new hyperbola with same asymptotes passing through (2, 2) should have this form
def hyperbola2 (x y : ℝ) : Prop := (x^2 / 3 - y^2 / 12 = 1)

theorem hyperbola_equiv (x y : ℝ) :
  (hyperbola1 2 2) →
  (y^2 / 4 - x^2 / 4 = -3) →
  (hyperbola2 x y) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_hyperbola_equiv_l2241_224145


namespace NUMINAMATH_GPT_eval_polynomial_at_3_l2241_224163

theorem eval_polynomial_at_3 : (3 : ℤ) ^ 3 + (3 : ℤ) ^ 2 + 3 + 1 = 40 := by
  sorry

end NUMINAMATH_GPT_eval_polynomial_at_3_l2241_224163


namespace NUMINAMATH_GPT_solve_system_l2241_224168

theorem solve_system (x y : ℝ) (h1 : x^2 + y^2 + x + y = 50) (h2 : x * y = 20) :
  (x = 5 ∧ y = 4) ∨ (x = 4 ∧ y = 5) ∨ (x = -5 + Real.sqrt 5 ∧ y = -5 - Real.sqrt 5) ∨ (x = -5 - Real.sqrt 5 ∧ y = -5 + Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2241_224168


namespace NUMINAMATH_GPT_probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l2241_224151

-- Definitions and conditions from the problem
def success_probability_each_segment : ℚ := 3 / 4
def num_segments : ℕ := 4

-- Correct answers from the solution
def prob_both_success : ℚ := 9 / 16
def expected_successful_segments : ℚ := 3
def cond_prob_given_3_successful : ℚ := 3 / 4

theorem probability_both_segments_successful :
  (success_probability_each_segment * success_probability_each_segment) = prob_both_success :=
by
  sorry

theorem expected_number_of_successful_segments :
  (num_segments * success_probability_each_segment) = expected_successful_segments :=
by
  sorry

theorem probability_given_3_successful :
  let prob_M := 4 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  let prob_NM := 3 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  (prob_NM / prob_M) = cond_prob_given_3_successful :=
by
  sorry

end NUMINAMATH_GPT_probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l2241_224151


namespace NUMINAMATH_GPT_phase_shift_equivalence_l2241_224194

noncomputable def y_original (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def y_target (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1
noncomputable def phase_shift : ℝ := 5 * Real.pi / 12

theorem phase_shift_equivalence : 
  ∀ x : ℝ, y_original x = y_target (x - phase_shift) :=
sorry

end NUMINAMATH_GPT_phase_shift_equivalence_l2241_224194


namespace NUMINAMATH_GPT_product_of_coordinates_of_D_l2241_224174

theorem product_of_coordinates_of_D (Mx My Cx Cy Dx Dy : ℝ) (M : (Mx, My) = (4, 8)) (C : (Cx, Cy) = (5, 4)) 
  (midpoint : (Mx, My) = ((Cx + Dx) / 2, (Cy + Dy) / 2)) : (Dx * Dy) = 36 := 
by
  sorry

end NUMINAMATH_GPT_product_of_coordinates_of_D_l2241_224174


namespace NUMINAMATH_GPT_max_value_of_f_l2241_224186

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) (h : -2 < a ∧ a ≤ 0) : 
  ∀ x ∈ (Set.Icc 0 (a + 2)), f x ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l2241_224186


namespace NUMINAMATH_GPT_combined_vacations_and_classes_l2241_224127

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end NUMINAMATH_GPT_combined_vacations_and_classes_l2241_224127


namespace NUMINAMATH_GPT_find_x_l2241_224169

theorem find_x (x y: ℤ) (h1: x + 2 * y = 12) (h2: y = 3) : x = 6 := by
  sorry

end NUMINAMATH_GPT_find_x_l2241_224169


namespace NUMINAMATH_GPT_divisible_by_five_solution_exists_l2241_224183

theorem divisible_by_five_solution_exists
  (a b c d : ℤ)
  (h₀ : ∃ k : ℤ, d = 5 * k + d % 5 ∧ d % 5 ≠ 0)
  (h₁ : ∃ n : ℤ, (a * n^3 + b * n^2 + c * n + d) % 5 = 0) :
  ∃ m : ℤ, (a + b * m + c * m^2 + d * m^3) % 5 = 0 := 
sorry

end NUMINAMATH_GPT_divisible_by_five_solution_exists_l2241_224183


namespace NUMINAMATH_GPT_mitchell_pencils_l2241_224149

/-- Mitchell and Antonio have a combined total of 54 pencils.
Mitchell has 6 more pencils than Antonio. -/
theorem mitchell_pencils (A M : ℕ) 
  (h1 : M = A + 6)
  (h2 : M + A = 54) : M = 30 :=
by
  sorry

end NUMINAMATH_GPT_mitchell_pencils_l2241_224149


namespace NUMINAMATH_GPT_power_function_increasing_l2241_224112

   theorem power_function_increasing (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x < y → x^a < y^a) : 0 < a :=
   by
   sorry
   
end NUMINAMATH_GPT_power_function_increasing_l2241_224112


namespace NUMINAMATH_GPT_number_of_distinct_intersections_l2241_224172

theorem number_of_distinct_intersections :
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 16 ∧ 16 * x^2 + 9 * y^2 = 9) →
  (∀ x y₁ y₂ : ℝ, 9 * x^2 + 16 * y₁^2 = 16 ∧ 16 * x^2 + 9 * y₁^2 = 9 ∧
    9 * x^2 + 16 * y₂^2 = 16 ∧ 16 * x^2 + 9 * y₂^2 = 9 → y₁ = y₂) →
  (∃! p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 16 ∧ 16 * p.1^2 + 9 * p.2^2 = 9) :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_intersections_l2241_224172


namespace NUMINAMATH_GPT_moles_of_CO2_formed_l2241_224107

variables (CH4 O2 C2H2 CO2 H2O : Type)
variables (nCH4 nO2 nC2H2 nCO2 : ℕ)
variables (reactsCompletely : Prop)

-- Balanced combustion equations
axiom combustion_methane : ∀ (mCH4 mO2 mCO2 mH2O : ℕ), mCH4 = 1 → mO2 = 2 → mCO2 = 1 → mH2O = 2 → Prop
axiom combustion_acetylene : ∀ (aC2H2 aO2 aCO2 aH2O : ℕ), aC2H2 = 2 → aO2 = 5 → aCO2 = 4 → aH2O = 2 → Prop

-- Given conditions
axiom conditions :
  nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely

-- Prove the number of moles of CO2 formed
theorem moles_of_CO2_formed : 
  (nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely) →
  nCO2 = 3
:= by
  intros h
  sorry

end NUMINAMATH_GPT_moles_of_CO2_formed_l2241_224107


namespace NUMINAMATH_GPT_find_fx_when_x_positive_l2241_224155

def isOddFunction {α : Type} [AddGroup α] [Neg α] (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)
variable (h_odd : isOddFunction f)
variable (h_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + x)

theorem find_fx_when_x_positive : ∀ x : ℝ, x > 0 → f x = x^2 + x :=
by
  sorry

end NUMINAMATH_GPT_find_fx_when_x_positive_l2241_224155


namespace NUMINAMATH_GPT_result_number_of_edges_l2241_224185

-- Define the conditions
def hexagon (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 6 ∧ edges = 6 * side_length)
def triangle (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 3 ∧ edges = 3 * side_length)

-- State the theorem
theorem result_number_of_edges (side_length_hex : ℕ) (side_length_tri : ℕ)
  (h_h : hexagon side_length_hex) (h_t : triangle side_length_tri)
  (aligned_edge_to_edge : side_length_hex = side_length_tri ∧ side_length_hex = 1 ∧ side_length_tri = 1) :
  ∃ edges, edges = 5 :=
by
  -- Proof is not provided, it is marked with sorry
  sorry

end NUMINAMATH_GPT_result_number_of_edges_l2241_224185


namespace NUMINAMATH_GPT_cos_C_in_triangle_l2241_224147

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end NUMINAMATH_GPT_cos_C_in_triangle_l2241_224147


namespace NUMINAMATH_GPT_molecular_weight_is_correct_l2241_224120

noncomputable def molecular_weight_of_compound : ℝ :=
  3 * 39.10 + 2 * 51.996 + 7 * 15.999 + 4 * 1.008 + 1 * 14.007

theorem molecular_weight_is_correct : molecular_weight_of_compound = 351.324 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_is_correct_l2241_224120


namespace NUMINAMATH_GPT_annulus_area_sufficient_linear_element_l2241_224134

theorem annulus_area_sufficient_linear_element (R r : ℝ) (hR : R > 0) (hr : r > 0) (hrR : r < R):
  (∃ d : ℝ, d = R - r ∨ d = R + r) → ∃ A : ℝ, A = π * (R ^ 2 - r ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_annulus_area_sufficient_linear_element_l2241_224134


namespace NUMINAMATH_GPT_union_of_sets_l2241_224146

def setA : Set ℝ := { x | -5 ≤ x ∧ x < 1 }
def setB : Set ℝ := { x | x ≤ 2 }

theorem union_of_sets : setA ∪ setB = { x | x ≤ 2 } :=
by sorry

end NUMINAMATH_GPT_union_of_sets_l2241_224146


namespace NUMINAMATH_GPT_sales_ratio_l2241_224164

def large_price : ℕ := 60
def small_price : ℕ := 30
def last_month_large_paintings : ℕ := 8
def last_month_small_paintings : ℕ := 4
def this_month_sales : ℕ := 1200

theorem sales_ratio :
  (this_month_sales : ℕ) = 2 * (last_month_large_paintings * large_price + last_month_small_paintings * small_price) :=
by
  -- We will just state the proof steps as sorry.
  sorry

end NUMINAMATH_GPT_sales_ratio_l2241_224164


namespace NUMINAMATH_GPT_new_boxes_of_markers_l2241_224175

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end NUMINAMATH_GPT_new_boxes_of_markers_l2241_224175


namespace NUMINAMATH_GPT_max_inscribed_circle_area_of_triangle_l2241_224109

theorem max_inscribed_circle_area_of_triangle
  (a b : ℝ)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (f1 f2 : ℝ × ℝ)
  (F1_coords : f1 = (-1, 0))
  (F2_coords : f2 = (1, 0))
  (P Q : ℝ × ℝ)
  (line_through_F2 : ∀ y : ℝ, x = 1 → y^2 = 9 / 4)
  (P_coords : P = (1, 3/2))
  (Q_coords : Q = (1, -3/2))
  : (π * (3 / 4)^2 = 9 * π / 16) :=
  sorry

end NUMINAMATH_GPT_max_inscribed_circle_area_of_triangle_l2241_224109


namespace NUMINAMATH_GPT_find_g_9_l2241_224170

-- Define the function g
def g (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 7

-- Given conditions
variables (a b c : ℝ)

-- g(-9) = 9
axiom h : g a b c (-9) = 9

-- Prove g(9) = -23
theorem find_g_9 : g a b c 9 = -23 :=
by
  sorry

end NUMINAMATH_GPT_find_g_9_l2241_224170


namespace NUMINAMATH_GPT_find_d_l2241_224173

theorem find_d (d : ℝ) (h : 4 * (3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5) = 3200.0000000000005) : d = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l2241_224173


namespace NUMINAMATH_GPT_find_angle_B_l2241_224126

theorem find_angle_B
  (a : ℝ) (c : ℝ) (A B C : ℝ)
  (h1 : a = 5 * Real.sqrt 2)
  (h2 : c = 10)
  (h3 : A = π / 6) -- 30 degrees in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  : B = 7 * π / 12 ∨ B = π / 12 := -- 105 degrees or 15 degrees in radians
sorry

end NUMINAMATH_GPT_find_angle_B_l2241_224126


namespace NUMINAMATH_GPT_solution_of_equation_l2241_224122

theorem solution_of_equation (a : ℝ) : (∃ x : ℝ, x = 4 ∧ (a * x - 3 = 4 * x + 1)) → a = 5 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_equation_l2241_224122


namespace NUMINAMATH_GPT_final_pen_count_l2241_224178

theorem final_pen_count
  (initial_pens : ℕ := 7) 
  (mike_given_pens : ℕ := 22) 
  (doubled_pens : ℕ := 2)
  (sharon_given_pens : ℕ := 19) :
  let total_after_mike := initial_pens + mike_given_pens
  let total_after_cindy := total_after_mike * doubled_pens
  let final_count := total_after_cindy - sharon_given_pens
  final_count = 39 :=
by
  sorry

end NUMINAMATH_GPT_final_pen_count_l2241_224178


namespace NUMINAMATH_GPT_distance_interval_l2241_224142

theorem distance_interval (d : ℝ) (h₁ : ¬ (d ≥ 8)) (h₂ : ¬ (d ≤ 6)) (h₃ : ¬ (d ≤ 3)) : 6 < d ∧ d < 8 := by
  sorry

end NUMINAMATH_GPT_distance_interval_l2241_224142


namespace NUMINAMATH_GPT_letters_received_per_day_l2241_224199

-- Define the conditions
def packages_per_day := 20
def total_pieces_in_six_months := 14400
def days_in_month := 30
def months := 6

-- Calculate total days in six months
def total_days := months * days_in_month

-- Calculate pieces of mail per day
def pieces_per_day := total_pieces_in_six_months / total_days

-- Define the number of letters per day
def letters_per_day := pieces_per_day - packages_per_day

-- Prove that the number of letters per day is 60
theorem letters_received_per_day : letters_per_day = 60 := sorry

end NUMINAMATH_GPT_letters_received_per_day_l2241_224199


namespace NUMINAMATH_GPT_parallel_lines_m_values_l2241_224179

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (m-2) * x - y - 1 = 0) ∧ (∀ x y : ℝ, 3 * x - m * y = 0) → 
  (m = -1 ∨ m = 3) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_values_l2241_224179


namespace NUMINAMATH_GPT_packs_of_beef_l2241_224121

noncomputable def pounds_per_pack : ℝ := 4
noncomputable def price_per_pound : ℝ := 5.50
noncomputable def total_paid : ℝ := 110
noncomputable def price_per_pack : ℝ := price_per_pound * pounds_per_pack

theorem packs_of_beef (n : ℝ) (h : n = total_paid / price_per_pack) : n = 5 := 
by
  sorry

end NUMINAMATH_GPT_packs_of_beef_l2241_224121


namespace NUMINAMATH_GPT_percentage_of_students_owning_cats_l2241_224113

theorem percentage_of_students_owning_cats (dogs cats total : ℕ) (h_dogs : dogs = 45) (h_cats : cats = 75) (h_total : total = 500) : 
  (cats / total) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_owning_cats_l2241_224113


namespace NUMINAMATH_GPT_polynomial_solution_l2241_224159

theorem polynomial_solution (p : ℝ → ℝ) (h : ∀ x, p (p x) = x * (p x) ^ 2 + x ^ 3) : 
  p = id :=
by {
    sorry
}

end NUMINAMATH_GPT_polynomial_solution_l2241_224159


namespace NUMINAMATH_GPT_parabola_min_area_l2241_224137

-- Definition of the parabola C with vertex at the origin and focus on the positive y-axis
-- (Conditions 1 and 2)
def parabola_eq (x y : ℝ) : Prop := x^2 = 6 * y

-- Line l defined by mx + y - 3/2 = 0 (Condition 3)
def line_eq (m x y : ℝ) : Prop := m * x + y - 3 / 2 = 0

-- Formal statement combining all conditions to prove the equivalent Lean statement
theorem parabola_min_area :
  (∀ x y : ℝ, parabola_eq x y ↔ x^2 = 6 * y) ∧
  (∀ m x y : ℝ, line_eq m x y ↔ m * x + y - 3 / 2 = 0) →
  (parabola_eq 0 0) ∧ (∃ y > 0, parabola_eq 0 y ∧ line_eq 0 0 (y/2) ∧ y = 3 / 2) ∧
  ∀ A B P : ℝ, line_eq 0 A B ∧ line_eq 0 B P ∧ A^2 + B^2 > 0 → 
  ∃ min_S : ℝ, min_S = 9 :=
by
  sorry

end NUMINAMATH_GPT_parabola_min_area_l2241_224137


namespace NUMINAMATH_GPT_matrix_power_difference_l2241_224128

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 4;
     0, 1]

theorem matrix_power_difference :
  B^30 - 3 * B^29 = !![-2, 0;
                       0,  2] := 
by
  sorry

end NUMINAMATH_GPT_matrix_power_difference_l2241_224128


namespace NUMINAMATH_GPT_pencils_to_sell_for_desired_profit_l2241_224118

/-- Definitions based on the conditions provided in the problem. -/
def total_pencils : ℕ := 2000
def cost_per_pencil : ℝ := 0.20
def sell_price_per_pencil : ℝ := 0.40
def desired_profit : ℝ := 160
def total_cost : ℝ := total_pencils * cost_per_pencil

/-- The theorem considers all the conditions and asks to prove the number of pencils to sell -/
theorem pencils_to_sell_for_desired_profit : 
  (desired_profit + total_cost) / sell_price_per_pencil = 1400 :=
by 
  sorry

end NUMINAMATH_GPT_pencils_to_sell_for_desired_profit_l2241_224118


namespace NUMINAMATH_GPT_truck_gas_consumption_l2241_224153

theorem truck_gas_consumption :
  ∀ (initial_gasoline total_distance remaining_gasoline : ℝ),
    initial_gasoline = 12 →
    total_distance = (2 * 5 + 2 + 2 * 2 + 6) →
    remaining_gasoline = 2 →
    (initial_gasoline - remaining_gasoline) ≠ 0 →
    (total_distance / (initial_gasoline - remaining_gasoline)) = 2.2 :=
by
  intros initial_gasoline total_distance remaining_gasoline
  intro h_initial_gas h_total_distance h_remaining_gas h_non_zero
  sorry

end NUMINAMATH_GPT_truck_gas_consumption_l2241_224153


namespace NUMINAMATH_GPT_power_difference_expression_l2241_224129

theorem power_difference_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * (30^1001) :=
by
  sorry

end NUMINAMATH_GPT_power_difference_expression_l2241_224129


namespace NUMINAMATH_GPT_ratio_of_w_to_y_l2241_224181

variables (w x y z : ℚ)

theorem ratio_of_w_to_y:
  (w / x = 5 / 4) →
  (y / z = 5 / 3) →
  (z / x = 1 / 5) →
  (w / y = 15 / 4) :=
by
  intros hwx hyz hzx
  sorry

end NUMINAMATH_GPT_ratio_of_w_to_y_l2241_224181


namespace NUMINAMATH_GPT_max_cos_y_cos_x_l2241_224190

noncomputable def max_cos_sum : ℝ :=
  1 + (Real.sqrt (2 + Real.sqrt 2)) / 2

theorem max_cos_y_cos_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (x y : ℝ), Real.cos y + Real.cos x = max_cos_sum :=
sorry

end NUMINAMATH_GPT_max_cos_y_cos_x_l2241_224190


namespace NUMINAMATH_GPT_common_ratio_of_infinite_geometric_series_l2241_224156

theorem common_ratio_of_infinite_geometric_series 
  (a b : ℚ) 
  (h1 : a = 8 / 10) 
  (h2 : b = -6 / 15) 
  (h3 : b = a * r) : 
  r = -1 / 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_common_ratio_of_infinite_geometric_series_l2241_224156


namespace NUMINAMATH_GPT_percentage_of_boys_currently_l2241_224102

theorem percentage_of_boys_currently (B G : ℕ) (h1 : B + G = 50) (h2 : B + 50 = 95) : (B / 50) * 100 = 90 := by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_currently_l2241_224102


namespace NUMINAMATH_GPT_exist_pairwise_distinct_gcd_l2241_224104

theorem exist_pairwise_distinct_gcd (S : Set ℕ) (h_inf : S.Infinite) 
  (h_gcd : ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ gcd a b ≠ gcd c d) :
  ∃ x y z : ℕ, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x := 
by sorry

end NUMINAMATH_GPT_exist_pairwise_distinct_gcd_l2241_224104


namespace NUMINAMATH_GPT_area_contained_by_graph_l2241_224100

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end NUMINAMATH_GPT_area_contained_by_graph_l2241_224100


namespace NUMINAMATH_GPT_price_increase_decrease_l2241_224108

theorem price_increase_decrease (P : ℝ) (x : ℝ) (h : P > 0) :
  (P * (1 + x / 100) * (1 - x / 100) = 0.64 * P) → (x = 60) :=
by
  sorry

end NUMINAMATH_GPT_price_increase_decrease_l2241_224108


namespace NUMINAMATH_GPT_bacon_suggestion_l2241_224154

theorem bacon_suggestion (x y : ℕ) (h1 : x = 479) (h2 : y = x + 10) : y = 489 := 
by {
  sorry
}

end NUMINAMATH_GPT_bacon_suggestion_l2241_224154


namespace NUMINAMATH_GPT_area_ratio_of_squares_l2241_224123

theorem area_ratio_of_squares (s t : ℝ) (h : 4 * s = 4 * (4 * t)) : (s ^ 2) / (t ^ 2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l2241_224123


namespace NUMINAMATH_GPT_train_speed_before_accident_l2241_224167

theorem train_speed_before_accident (d v : ℝ) (hv_pos : v > 0) (hd_pos : d > 0) :
  (d / ((3/4) * v) - d / v = 35 / 60) ∧
  (d - 24) / ((3/4) * v) - (d - 24) / v = 25 / 60 → 
  v = 64 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_before_accident_l2241_224167


namespace NUMINAMATH_GPT_ExpandedOHaraTripleValue_l2241_224187

/-- Define an Expanded O'Hara triple -/
def isExpandedOHaraTriple (a b x : ℕ) : Prop :=
  2 * (Nat.sqrt a + Nat.sqrt b) = x

/-- Prove that for given a=64 and b=49, x is equal to 30 if (a, b, x) is an Expanded O'Hara triple -/
theorem ExpandedOHaraTripleValue (a b x : ℕ) (ha : a = 64) (hb : b = 49) (h : isExpandedOHaraTriple a b x) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_ExpandedOHaraTripleValue_l2241_224187


namespace NUMINAMATH_GPT_new_cost_percentage_l2241_224188

variable (t b : ℝ)

-- Define the original cost
def original_cost : ℝ := t * b ^ 4

-- Define the new cost when b is doubled
def new_cost : ℝ := t * (2 * b) ^ 4

-- The theorem statement
theorem new_cost_percentage (t b : ℝ) : new_cost t b = 16 * original_cost t b := 
by
  -- Proof steps are skipped
  sorry

end NUMINAMATH_GPT_new_cost_percentage_l2241_224188


namespace NUMINAMATH_GPT_system1_solution_exists_system2_solution_exists_l2241_224140

-- System (1)
theorem system1_solution_exists (x y : ℝ) (h1 : y = 2 * x - 5) (h2 : 3 * x + 4 * y = 2) : 
  x = 2 ∧ y = -1 :=
by
  sorry

-- System (2)
theorem system2_solution_exists (x y : ℝ) (h1 : 3 * x - y = 8) (h2 : (y - 1) / 3 = (x + 5) / 5) : 
  x = 5 ∧ y = 7 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_exists_system2_solution_exists_l2241_224140


namespace NUMINAMATH_GPT_doubled_marks_new_average_l2241_224131

theorem doubled_marks_new_average (avg_marks : ℝ) (num_students : ℕ) (h_avg : avg_marks = 36) (h_num : num_students = 12) : 2 * avg_marks = 72 :=
by
  sorry

end NUMINAMATH_GPT_doubled_marks_new_average_l2241_224131


namespace NUMINAMATH_GPT_line_intersects_circle_l2241_224148

variable (x0 y0 R : ℝ)

theorem line_intersects_circle (h : x0^2 + y0^2 > R^2) :
  ∃ (x y : ℝ), (x^2 + y^2 = R^2) ∧ (x0 * x + y0 * y = R^2) :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l2241_224148


namespace NUMINAMATH_GPT_digit_after_decimal_l2241_224189

theorem digit_after_decimal (n : ℕ) : (n = 123) → (123 % 12 ≠ 0) → (123 % 12 = 3) → (∃ d : ℕ, d = 1 ∧ (43 / 740 : ℚ)^123 = 0 + d / 10^(123)) := 
by
    intros h₁ h₂ h₃
    sorry

end NUMINAMATH_GPT_digit_after_decimal_l2241_224189


namespace NUMINAMATH_GPT_coefficient_of_determination_l2241_224166

-- Define the observations and conditions for the problem
def observations (n : ℕ) := 
  {x : ℕ → ℝ // ∃ b a : ℝ, ∀ i : ℕ, i < n → ∃ y_i : ℝ, y_i = b * x i + a}

/-- 
  Given a set of observations (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) 
  that satisfies the equation y_i = bx_i + a for i = 1, 2, ..., n, 
  prove that the coefficient of determination R² is 1.
-/
theorem coefficient_of_determination (n : ℕ) (obs : observations n) : 
  ∃ R_squared : ℝ, R_squared = 1 :=
sorry

end NUMINAMATH_GPT_coefficient_of_determination_l2241_224166


namespace NUMINAMATH_GPT_correct_answers_count_l2241_224132

-- Define the conditions from the problem
def total_questions : ℕ := 25
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def total_score : ℤ := 85

-- State the theorem
theorem correct_answers_count :
  ∃ x : ℕ, (x ≤ total_questions) ∧ 
           (total_questions - x : ℕ) ≥ 0 ∧ 
           (correct_points * x + incorrect_points * (total_questions - x) = total_score) :=
sorry

end NUMINAMATH_GPT_correct_answers_count_l2241_224132


namespace NUMINAMATH_GPT_slant_height_of_cone_l2241_224135

theorem slant_height_of_cone
  (r : ℝ) (CSA : ℝ) (l : ℝ)
  (hr : r = 14)
  (hCSA : CSA = 1539.3804002589986) :
  CSA = Real.pi * r * l → l = 35 := 
sorry

end NUMINAMATH_GPT_slant_height_of_cone_l2241_224135


namespace NUMINAMATH_GPT_area_of_square_plot_l2241_224124

theorem area_of_square_plot (price_per_foot : ℕ) (total_cost : ℕ) (h_price : price_per_foot = 58) (h_cost : total_cost = 2088) :
  ∃ s : ℕ, s^2 = 81 := by
  sorry

end NUMINAMATH_GPT_area_of_square_plot_l2241_224124


namespace NUMINAMATH_GPT_Cagney_and_Lacey_Cupcakes_l2241_224133

-- Conditions
def CagneyRate := 1 / 25 -- cupcakes per second
def LaceyRate := 1 / 35 -- cupcakes per second
def TotalTimeInSeconds := 10 * 60 -- total time in seconds
def LaceyPrepTimeInSeconds := 1 * 60 -- Lacey's preparation time in seconds
def EffectiveWorkTimeInSeconds := TotalTimeInSeconds - LaceyPrepTimeInSeconds -- effective working time

-- Calculate combined rate
def CombinedRate := 1 / (1 / CagneyRate + 1 / LaceyRate) -- combined rate in cupcakes per second

-- Calculate the total number of cupcakes frosted
def TotalCupcakesFrosted := EffectiveWorkTimeInSeconds * CombinedRate -- total cupcakes frosted

-- We state the theorem that corresponds to our proof problem
theorem Cagney_and_Lacey_Cupcakes : TotalCupcakesFrosted = 37 := by
  sorry

end NUMINAMATH_GPT_Cagney_and_Lacey_Cupcakes_l2241_224133


namespace NUMINAMATH_GPT_opposite_sides_line_l2241_224162

theorem opposite_sides_line (m : ℝ) : 
  (2 * 1 + 3 + m) * (2 * -4 + -2 + m) < 0 ↔ -5 < m ∧ m < 10 :=
by sorry

end NUMINAMATH_GPT_opposite_sides_line_l2241_224162


namespace NUMINAMATH_GPT_solve_ordered_pair_l2241_224125

theorem solve_ordered_pair : ∃ (x y : ℚ), 3*x - 24*y = 3 ∧ x - 3*y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end NUMINAMATH_GPT_solve_ordered_pair_l2241_224125


namespace NUMINAMATH_GPT_ashley_loan_least_months_l2241_224119

theorem ashley_loan_least_months (t : ℕ) (principal : ℝ) (interest_rate : ℝ) (triple_principal : ℝ) : 
  principal = 1500 ∧ interest_rate = 0.06 ∧ triple_principal = 3 * principal → 
  1.06^t > triple_principal → t = 20 :=
by
  intro h h2
  sorry

end NUMINAMATH_GPT_ashley_loan_least_months_l2241_224119


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2241_224165

-- Define the set A as the solutions to the equation x^2 - 4 = 0
def A : Set ℝ := { x | x^2 - 4 = 0 }

-- Define the set B as the explicit set {1, 2}
def B : Set ℝ := {1, 2}

-- Prove that the intersection of sets A and B is {2}
theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  unfold A B
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2241_224165


namespace NUMINAMATH_GPT_range_of_a_l2241_224114

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * Real.cos (Real.pi / 2 - x)

theorem range_of_a (a : ℝ) (h_condition : f (2 * a ^ 2) + f (a - 3) + f 0 < 0) : -3/2 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2241_224114


namespace NUMINAMATH_GPT_yield_percentage_of_stock_l2241_224160

noncomputable def annual_dividend (par_value : ℝ) : ℝ := 0.21 * par_value
noncomputable def market_price : ℝ := 210
noncomputable def yield_percentage (annual_dividend : ℝ) (market_price : ℝ) : ℝ :=
  (annual_dividend / market_price) * 100

theorem yield_percentage_of_stock (par_value : ℝ)
  (h_par_value : par_value = 100) :
  yield_percentage (annual_dividend par_value) market_price = 10 :=
by
  sorry

end NUMINAMATH_GPT_yield_percentage_of_stock_l2241_224160


namespace NUMINAMATH_GPT_nonneg_for_all_x_iff_a_in_range_l2241_224106

def f (x a : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem nonneg_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_nonneg_for_all_x_iff_a_in_range_l2241_224106
