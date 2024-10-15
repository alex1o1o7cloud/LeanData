import Mathlib

namespace NUMINAMATH_GPT_inverse_negative_exchange_l200_20077

theorem inverse_negative_exchange (f1 f2 f3 f4 : ℝ → ℝ) (hx1 : ∀ x, f1 x = x - (1/x))
  (hx2 : ∀ x, f2 x = x + (1/x)) (hx3 : ∀ x, f3 x = Real.log x)
  (hx4 : ∀ x, f4 x = if 0 < x ∧ x < 1 then x else if x = 1 then 0 else -(1/x)) :
  (∀ x, f1 (1/x) = -f1 x) ∧ (∀ x, f2 (1/x) = -f2 x) ∧ (∀ x, f3 (1/x) = -f3 x) ∧
  (∀ x, f4 (1/x) = -f4 x) ↔ True := by 
  sorry

end NUMINAMATH_GPT_inverse_negative_exchange_l200_20077


namespace NUMINAMATH_GPT_combined_size_UK_India_US_l200_20039

theorem combined_size_UK_India_US (U : ℝ)
    (Canada : ℝ := 1.5 * U)
    (Russia : ℝ := (1 + 1/3) * Canada)
    (China : ℝ := (1 / 1.7) * Russia)
    (Brazil : ℝ := (2 / 3) * U)
    (Australia : ℝ := (1 / 2) * Brazil)
    (UK : ℝ := 2 * Australia)
    (India : ℝ := (1 / 4) * Russia)
    (India' : ℝ := 6 * UK)
    (h_India : India = India') :
  UK + India = 7 / 6 * U := 
by
  -- Proof details
  sorry

end NUMINAMATH_GPT_combined_size_UK_India_US_l200_20039


namespace NUMINAMATH_GPT_multiple_of_other_number_l200_20029

theorem multiple_of_other_number (S L k : ℤ) (h₁ : S = 18) (h₂ : L = k * S - 3) (h₃ : S + L = 51) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_other_number_l200_20029


namespace NUMINAMATH_GPT_variance_defect_rate_l200_20015

noncomputable def defect_rate : ℝ := 0.02
noncomputable def number_of_trials : ℕ := 100
noncomputable def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem variance_defect_rate :
  variance_binomial number_of_trials defect_rate = 1.96 :=
by
  sorry

end NUMINAMATH_GPT_variance_defect_rate_l200_20015


namespace NUMINAMATH_GPT_determineFinalCounts_l200_20024

structure FruitCounts where
  plums : ℕ
  oranges : ℕ
  apples : ℕ
  pears : ℕ
  cherries : ℕ

def initialCounts : FruitCounts :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def givenAway : FruitCounts :=
  { plums := 4, oranges := 3, apples := 5, pears := 0, cherries := 0 }

def receivedFromSam : FruitCounts :=
  { plums := 2, oranges := 0, apples := 0, pears := 1, cherries := 0 }

def receivedFromBrother : FruitCounts :=
  { plums := 0, oranges := 1, apples := 2, pears := 0, cherries := 0 }

def receivedFromNeighbor : FruitCounts :=
  { plums := 0, oranges := 0, apples := 0, pears := 3, cherries := 2 }

def finalCounts (initial given receivedSam receivedBrother receivedNeighbor : FruitCounts) : FruitCounts :=
  { plums := initial.plums - given.plums + receivedSam.plums,
    oranges := initial.oranges - given.oranges + receivedBrother.oranges,
    apples := initial.apples - given.apples + receivedBrother.apples,
    pears := initial.pears - given.pears + receivedSam.pears + receivedNeighbor.pears,
    cherries := initial.cherries - given.cherries + receivedNeighbor.cherries }

theorem determineFinalCounts :
  finalCounts initialCounts givenAway receivedFromSam receivedFromBrother receivedFromNeighbor =
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 } :=
by
  sorry

end NUMINAMATH_GPT_determineFinalCounts_l200_20024


namespace NUMINAMATH_GPT_solve_thought_of_number_l200_20007

def thought_of_number (x : ℝ) : Prop :=
  (x / 6) + 5 = 17

theorem solve_thought_of_number :
  ∃ x, thought_of_number x ∧ x = 72 :=
by
  sorry

end NUMINAMATH_GPT_solve_thought_of_number_l200_20007


namespace NUMINAMATH_GPT_find_a5_a7_l200_20032

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom h1 : a 1 + a 3 = 2
axiom h2 : a 3 + a 5 = 4

theorem find_a5_a7 (a : ℕ → ℤ) (d : ℤ) (h_seq : is_arithmetic_sequence a d)
  (h1 : a 1 + a 3 = 2) (h2 : a 3 + a 5 = 4) : a 5 + a 7 = 6 :=
sorry

end NUMINAMATH_GPT_find_a5_a7_l200_20032


namespace NUMINAMATH_GPT_elephant_weight_equivalence_l200_20091

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end NUMINAMATH_GPT_elephant_weight_equivalence_l200_20091


namespace NUMINAMATH_GPT_prop_logic_example_l200_20085

theorem prop_logic_example (p q : Prop) (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by {
  sorry
}

end NUMINAMATH_GPT_prop_logic_example_l200_20085


namespace NUMINAMATH_GPT_max_value_of_expression_l200_20098

-- Define the variables and constraints
variables {a b c d : ℤ}
variables (S : finset ℤ) (a_val b_val c_val d_val : ℤ)

axiom h1 : S = {0, 1, 2, 4, 5}
axiom h2 : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S
axiom h3 : ∀ x ∈ S, x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d
axiom h4 : ∀ x ∈ S, x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d
axiom h5 : ∀ x ∈ S, x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d
axiom h6 : ∀ x ∈ S, x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c

-- The main theorem to be proven
theorem max_value_of_expression : (∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
  (∀ x ∈ S, (x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d) ∧ 
             (x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c)) ∧
  (c * a^b - d = 20)) :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l200_20098


namespace NUMINAMATH_GPT_years_since_mothers_death_l200_20084

noncomputable def jessica_age_at_death (x : ℕ) : ℕ := 40 - x
noncomputable def mother_age_at_death (x : ℕ) : ℕ := 2 * jessica_age_at_death x

theorem years_since_mothers_death (x : ℕ) : mother_age_at_death x + x = 70 ↔ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_years_since_mothers_death_l200_20084


namespace NUMINAMATH_GPT_mrs_hilt_total_payment_l200_20057

noncomputable def total_hotdogs : ℕ := 12
noncomputable def cost_first_4 : ℝ := 4 * 0.60
noncomputable def cost_next_5 : ℝ := 5 * 0.75
noncomputable def cost_last_3 : ℝ := 3 * 0.90
noncomputable def total_cost : ℝ := cost_first_4 + cost_next_5 + cost_last_3

theorem mrs_hilt_total_payment : total_cost = 8.85 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_mrs_hilt_total_payment_l200_20057


namespace NUMINAMATH_GPT_age_difference_l200_20065

def A := 10
def B := 8
def C := B / 2
def total_age (A B C : ℕ) : Prop := A + B + C = 22

theorem age_difference (A B C : ℕ) (hB : B = 8) (hC : B = 2 * C) (h_total : total_age A B C) : A - B = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_l200_20065


namespace NUMINAMATH_GPT_circle_center_distance_travelled_l200_20045

theorem circle_center_distance_travelled :
  ∀ (r : ℝ) (a b c : ℝ), r = 2 ∧ a = 9 ∧ b = 12 ∧ c = 15 → (a^2 + b^2 = c^2) → 
  ∃ (d : ℝ), d = 24 :=
by
  intros r a b c h1 h2
  sorry

end NUMINAMATH_GPT_circle_center_distance_travelled_l200_20045


namespace NUMINAMATH_GPT_total_cookies_l200_20000

theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) : bags * cookies_per_bag = 703 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_l200_20000


namespace NUMINAMATH_GPT_insert_arithmetic_sequence_l200_20090

theorem insert_arithmetic_sequence (d a b : ℤ) 
  (h1 : (-1) + 3 * d = 8) 
  (h2 : a = (-1) + d) 
  (h3 : b = a + d) : 
  a = 2 ∧ b = 5 := by
  sorry

end NUMINAMATH_GPT_insert_arithmetic_sequence_l200_20090


namespace NUMINAMATH_GPT_linear_correlation_test_l200_20025

theorem linear_correlation_test (n1 n2 n3 n4 : ℕ) (r1 r2 r3 r4 : ℝ) :
  n1 = 10 ∧ r1 = 0.9533 →
  n2 = 15 ∧ r2 = 0.3012 →
  n3 = 17 ∧ r3 = 0.9991 →
  n4 = 3  ∧ r4 = 0.9950 →
  abs r1 > abs r2 ∧ abs r3 > abs r4 →
  (abs r1 > abs r2 → abs r1 > abs r4) →
  (abs r3 > abs r2 → abs r3 > abs r4) →
  abs r1 ≠ abs r2 →
  abs r3 ≠ abs r4 →
  true := 
sorry

end NUMINAMATH_GPT_linear_correlation_test_l200_20025


namespace NUMINAMATH_GPT_value_of_expression_l200_20013

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 2 * x + 5 = 9) : 3 * x^2 + 3 * x - 7 = -1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_value_of_expression_l200_20013


namespace NUMINAMATH_GPT_transportation_cost_l200_20068

-- Definitions for the conditions
def number_of_original_bags : ℕ := 80
def weight_of_original_bag : ℕ := 50
def total_cost_original : ℕ := 6000

def scale_factor_bags : ℕ := 3
def scale_factor_weight : ℚ := 3 / 5

-- Derived quantities
def number_of_new_bags : ℕ := scale_factor_bags * number_of_original_bags
def weight_of_new_bag : ℚ := scale_factor_weight * weight_of_original_bag
def cost_per_original_bag : ℚ := total_cost_original / number_of_original_bags
def cost_per_new_bag : ℚ := cost_per_original_bag * (weight_of_new_bag / weight_of_original_bag)

-- Final cost calculation
def total_cost_new : ℚ := number_of_new_bags * cost_per_new_bag

-- The statement that needs to be proved
theorem transportation_cost : total_cost_new = 10800 := sorry

end NUMINAMATH_GPT_transportation_cost_l200_20068


namespace NUMINAMATH_GPT_remainder_3_45_plus_4_mod_5_l200_20040

theorem remainder_3_45_plus_4_mod_5 :
  (3 ^ 45 + 4) % 5 = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_3_45_plus_4_mod_5_l200_20040


namespace NUMINAMATH_GPT_base4_last_digit_390_l200_20021

theorem base4_last_digit_390 : 
  (Nat.digits 4 390).head! = 2 := sorry

end NUMINAMATH_GPT_base4_last_digit_390_l200_20021


namespace NUMINAMATH_GPT_compare_negatives_l200_20037

theorem compare_negatives : -2 > -3 :=
by
  sorry

end NUMINAMATH_GPT_compare_negatives_l200_20037


namespace NUMINAMATH_GPT_grocer_rows_count_l200_20031

theorem grocer_rows_count (n : ℕ) (a d S : ℕ) (h_a : a = 1) (h_d : d = 3) (h_S : S = 225)
  (h_sum : S = n * (2 * a + (n - 1) * d) / 2) : n = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_grocer_rows_count_l200_20031


namespace NUMINAMATH_GPT_frustum_surface_area_l200_20062

noncomputable def total_surface_area_of_frustum
  (R r h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (R - r)^2)
  let A_lateral := Real.pi * (R + r) * s
  let A_top := Real.pi * r^2
  let A_bottom := Real.pi * R^2
  A_lateral + A_top + A_bottom

theorem frustum_surface_area :
  total_surface_area_of_frustum 8 2 5 = 10 * Real.pi * Real.sqrt 61 + 68 * Real.pi :=
  sorry

end NUMINAMATH_GPT_frustum_surface_area_l200_20062


namespace NUMINAMATH_GPT_quadratic_range_l200_20004

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end NUMINAMATH_GPT_quadratic_range_l200_20004


namespace NUMINAMATH_GPT_triangle_a_c_sin_A_minus_B_l200_20019

theorem triangle_a_c_sin_A_minus_B (a b c : ℝ) (A B C : ℝ):
  a + c = 6 → b = 2 → Real.cos B = 7/9 →
  a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_triangle_a_c_sin_A_minus_B_l200_20019


namespace NUMINAMATH_GPT_find_a_l200_20097

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + 1) / (x + 1)

theorem find_a (a : ℝ) (h1 : ∃ t, t = (f a 1 - 1) / (1 - 0) ∧ t = ((3 * a - 1) / 4)) : a = -1 :=
by
  -- Auxiliary steps to frame the Lean theorem precisely
  let f1 := f a 1
  have h2 : f1 = (a + 1) / 2 := sorry
  have slope_tangent : ∀ t : ℝ, t = (3 * a - 1) / 4 := sorry
  have tangent_eq : (∀ (x y : ℝ), y - f1 = ((3 * a - 1) / 4) * (x - 1)) := sorry
  have pass_point : ∀ (x y : ℝ), (x, y) = (0, 1) -> (1 : ℝ) - ((a + 1) / 2) = ((1 - 3 * a) / 4) := sorry
  exact sorry

end NUMINAMATH_GPT_find_a_l200_20097


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l200_20063

/-- Define the conditions. -/
def a : ℕ := 2
def d : ℕ := 5
def a_n : ℕ := 57

/-- Define the proof problem. -/
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, a_n = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l200_20063


namespace NUMINAMATH_GPT_two_digit_numbers_solution_l200_20055

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_solution_l200_20055


namespace NUMINAMATH_GPT_smallest_N_l200_20067

theorem smallest_N (l m n : ℕ) (N : ℕ) (h1 : N = l * m * n) (h2 : (l - 1) * (m - 1) * (n - 1) = 300) : 
  N = 462 :=
sorry

end NUMINAMATH_GPT_smallest_N_l200_20067


namespace NUMINAMATH_GPT_range_of_a_l200_20072

theorem range_of_a :
  (∀ t : ℝ, 0 < t ∧ t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) →
  (2 / 13 ≤ a ∧ a ≤ 1) :=
by
  intro h
  -- Proof of the theorem goes here
  sorry

end NUMINAMATH_GPT_range_of_a_l200_20072


namespace NUMINAMATH_GPT_work_completion_days_l200_20086

open Real

theorem work_completion_days (days_A : ℝ) (days_B : ℝ) (amount_total : ℝ) (amount_C : ℝ) :
  days_A = 6 ∧ days_B = 8 ∧ amount_total = 5000 ∧ amount_C = 625.0000000000002 →
  (1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1)) = 5 / 12 →
  1 / ((1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1))) = 2.4 :=
  sorry

end NUMINAMATH_GPT_work_completion_days_l200_20086


namespace NUMINAMATH_GPT_right_triangle_area_l200_20010

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := 1 / 2 * a * b

theorem right_triangle_area {a b : ℝ} 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 14) : 
  area_of_right_triangle a b = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l200_20010


namespace NUMINAMATH_GPT_cone_sector_central_angle_l200_20081

noncomputable def base_radius := 1
noncomputable def slant_height := 2
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r : ℝ) := circumference r
noncomputable def central_angle (l : ℝ) (s : ℝ) := l / s

theorem cone_sector_central_angle : central_angle (arc_length base_radius) slant_height = Real.pi := 
by 
  -- Here we acknowledge that the proof would go, but it is left out as per instructions.
  sorry

end NUMINAMATH_GPT_cone_sector_central_angle_l200_20081


namespace NUMINAMATH_GPT_mean_of_six_numbers_l200_20044

theorem mean_of_six_numbers (a b c d e f : ℚ) (h : a + b + c + d + e + f = 1 / 3) :
  (a + b + c + d + e + f) / 6 = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_six_numbers_l200_20044


namespace NUMINAMATH_GPT_a_b_product_l200_20093

theorem a_b_product (a b : ℝ) (h1 : 2 * a - b = 1) (h2 : 2 * b - a = 7) : (a + b) * (a - b) = -16 :=
by
  -- The proof would be provided here.
  sorry

end NUMINAMATH_GPT_a_b_product_l200_20093


namespace NUMINAMATH_GPT_correct_option_C_l200_20080

noncomputable def question := "Which of the following operations is correct?"
noncomputable def option_A := (-2)^2
noncomputable def option_B := (-2)^3
noncomputable def option_C := (-1/2)^3
noncomputable def option_D := (-7/3)^3
noncomputable def correct_answer := -1/8

theorem correct_option_C :
  option_C = correct_answer := by
  sorry

end NUMINAMATH_GPT_correct_option_C_l200_20080


namespace NUMINAMATH_GPT_log_9_256_eq_4_log_2_3_l200_20071

noncomputable def logBase9Base2Proof : Prop :=
  (Real.log 256 / Real.log 9 = 4 * (Real.log 3 / Real.log 2))

theorem log_9_256_eq_4_log_2_3 : logBase9Base2Proof :=
by
  sorry

end NUMINAMATH_GPT_log_9_256_eq_4_log_2_3_l200_20071


namespace NUMINAMATH_GPT_f_f_five_eq_five_l200_20088

-- Define the function and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Hypotheses
axiom h1 : ∀ x : ℝ, f (x + 2) = -f x
axiom h2 : f 1 = -5

-- Theorem to prove
theorem f_f_five_eq_five : f (f 5) = 5 :=
sorry

end NUMINAMATH_GPT_f_f_five_eq_five_l200_20088


namespace NUMINAMATH_GPT_two_colonies_same_time_l200_20079

def doubles_in_size_every_day (P : ℕ → ℕ) : Prop :=
∀ n, P (n + 1) = 2 * P n

def reaches_habitat_limit_in (f : ℕ → ℕ) (days limit : ℕ) : Prop :=
f days = limit

theorem two_colonies_same_time (P : ℕ → ℕ) (Q : ℕ → ℕ) (limit : ℕ) (days : ℕ)
  (h1 : doubles_in_size_every_day P)
  (h2 : reaches_habitat_limit_in P days limit)
  (h3 : ∀ n, Q n = 2 * P n) :
  reaches_habitat_limit_in Q days limit :=
sorry

end NUMINAMATH_GPT_two_colonies_same_time_l200_20079


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l200_20026

variables {a b c : ℝ × ℝ}

def nonzero_vector (v : ℝ × ℝ) : Prop := v ≠ (0, 0)

theorem necessary_but_not_sufficient_condition (ha : nonzero_vector a) (hb : nonzero_vector b) (hc : nonzero_vector c) :
  (a.1 * (b.1 - c.1) + a.2 * (b.2 - c.2) = 0) ↔ (b = c) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l200_20026


namespace NUMINAMATH_GPT_gcd_is_3_l200_20073

def gcd_6273_14593 : ℕ := Nat.gcd 6273 14593

theorem gcd_is_3 : gcd_6273_14593 = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_is_3_l200_20073


namespace NUMINAMATH_GPT_johns_profit_is_200_l200_20096

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_johns_profit_is_200_l200_20096


namespace NUMINAMATH_GPT_largest_r_in_subset_l200_20038

theorem largest_r_in_subset (A : Finset ℕ) (hA : A.card = 500) : 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ (B ∩ C).card ≥ 100 := sorry

end NUMINAMATH_GPT_largest_r_in_subset_l200_20038


namespace NUMINAMATH_GPT_vasechkin_result_l200_20082

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end NUMINAMATH_GPT_vasechkin_result_l200_20082


namespace NUMINAMATH_GPT_dan_spent_at_music_store_l200_20016

def cost_of_clarinet : ℝ := 130.30
def cost_of_song_book : ℝ := 11.24
def money_left_in_pocket : ℝ := 12.32
def total_spent : ℝ := 129.22

theorem dan_spent_at_music_store : 
  cost_of_clarinet + cost_of_song_book - money_left_in_pocket = total_spent :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_dan_spent_at_music_store_l200_20016


namespace NUMINAMATH_GPT_color_property_l200_20089

theorem color_property (k : ℕ) (h : k ≥ 1) : k = 1 ∨ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_color_property_l200_20089


namespace NUMINAMATH_GPT_y_share_per_x_l200_20041

theorem y_share_per_x (total_amount y_share : ℝ) (z_share_per_x : ℝ) 
  (h_total : total_amount = 234)
  (h_y_share : y_share = 54)
  (h_z_share_per_x : z_share_per_x = 0.5) :
  ∃ a : ℝ, (forall x : ℝ, y_share = a * x) ∧ a = 9 / 20 :=
by
  use 9 / 20
  intros
  sorry

end NUMINAMATH_GPT_y_share_per_x_l200_20041


namespace NUMINAMATH_GPT_total_area_expanded_dining_area_l200_20070

noncomputable def expanded_dining_area_total : ℝ :=
  let rectangular_area := 35
  let radius := 4
  let semi_circular_area := (1 / 2) * Real.pi * (radius^2)
  rectangular_area + semi_circular_area

theorem total_area_expanded_dining_area :
  expanded_dining_area_total = 60.13272 := by
  sorry

end NUMINAMATH_GPT_total_area_expanded_dining_area_l200_20070


namespace NUMINAMATH_GPT_find_triplets_l200_20046

theorem find_triplets (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a ^ b ∣ b ^ c - 1) ∧ (a ^ c ∣ c ^ b - 1)) ↔ (a = 1 ∨ (b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_GPT_find_triplets_l200_20046


namespace NUMINAMATH_GPT_tan_theta_condition_l200_20002

open Real

theorem tan_theta_condition (k : ℤ) : 
  (∃ θ : ℝ, θ = 2 * k * π + π / 4 ∧ tan θ = 1) ∧ ¬ (∀ θ : ℝ, tan θ = 1 → ∃ k : ℤ, θ = 2 * k * π + π / 4) :=
by sorry

end NUMINAMATH_GPT_tan_theta_condition_l200_20002


namespace NUMINAMATH_GPT_monthly_rent_is_1300_l200_20058

def shop_length : ℕ := 10
def shop_width : ℕ := 10
def annual_rent_per_square_foot : ℕ := 156

def area_of_shop : ℕ := shop_length * shop_width
def annual_rent_for_shop : ℕ := annual_rent_per_square_foot * area_of_shop

def monthly_rent : ℕ := annual_rent_for_shop / 12

theorem monthly_rent_is_1300 : monthly_rent = 1300 := by
  sorry

end NUMINAMATH_GPT_monthly_rent_is_1300_l200_20058


namespace NUMINAMATH_GPT_find_F_l200_20022

theorem find_F (C F : ℝ) 
  (h1 : C = 7 / 13 * (F - 40))
  (h2 : C = 26) :
  F = 88.2857 :=
by
  sorry

end NUMINAMATH_GPT_find_F_l200_20022


namespace NUMINAMATH_GPT_problem_solution_l200_20047

noncomputable def f (A B : ℝ) (x : ℝ) : ℝ := A + B / x + x

theorem problem_solution (A B : ℝ) :
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 →
  (x * f A B (x + 1 / y) + y * f A B y + y / x = y * f A B (y + 1 / x) + x * f A B x + x / y) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l200_20047


namespace NUMINAMATH_GPT_guests_did_not_come_l200_20056

theorem guests_did_not_come 
  (total_cookies : ℕ) 
  (prepared_guests : ℕ) 
  (cookies_per_guest : ℕ) 
  (total_cookies_eq : total_cookies = 18) 
  (prepared_guests_eq : prepared_guests = 10)
  (cookies_per_guest_eq : cookies_per_guest = 18) 
  (total_cookies_computation : total_cookies = cookies_per_guest) :
  prepared_guests - total_cookies / cookies_per_guest = 9 :=
by
  sorry

end NUMINAMATH_GPT_guests_did_not_come_l200_20056


namespace NUMINAMATH_GPT_initial_money_amount_l200_20095

theorem initial_money_amount (x : ℕ) (h : x + 16 = 18) : x = 2 := by
  sorry

end NUMINAMATH_GPT_initial_money_amount_l200_20095


namespace NUMINAMATH_GPT_hotel_rooms_count_l200_20051

theorem hotel_rooms_count
  (TotalLamps : ℕ) (TotalChairs : ℕ) (TotalBedSheets : ℕ)
  (LampsPerRoom : ℕ) (ChairsPerRoom : ℕ) (BedSheetsPerRoom : ℕ) :
  TotalLamps = 147 → 
  TotalChairs = 84 → 
  TotalBedSheets = 210 → 
  LampsPerRoom = 7 → 
  ChairsPerRoom = 4 → 
  BedSheetsPerRoom = 10 →
  (TotalLamps / LampsPerRoom = 21) ∧ 
  (TotalChairs / ChairsPerRoom = 21) ∧ 
  (TotalBedSheets / BedSheetsPerRoom = 21) :=
by
  intros
  sorry

end NUMINAMATH_GPT_hotel_rooms_count_l200_20051


namespace NUMINAMATH_GPT_average_mark_is_correct_l200_20003

-- Define the maximum score in the exam
def max_score := 1100

-- Define the percentages scored by Amar, Bhavan, Chetan, and Deepak
def score_percentage_amar := 64 / 100
def score_percentage_bhavan := 36 / 100
def score_percentage_chetan := 44 / 100
def score_percentage_deepak := 52 / 100

-- Calculate the actual scores based on percentages
def score_amar := score_percentage_amar * max_score
def score_bhavan := score_percentage_bhavan * max_score
def score_chetan := score_percentage_chetan * max_score
def score_deepak := score_percentage_deepak * max_score

-- Define the total score
def total_score := score_amar + score_bhavan + score_chetan + score_deepak

-- Define the number of students
def number_of_students := 4

-- Define the average score
def average_score := total_score / number_of_students

-- The theorem to prove that the average score is 539
theorem average_mark_is_correct : average_score = 539 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_average_mark_is_correct_l200_20003


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l200_20049

theorem infinite_geometric_series_sum : 
  (∃ (a r : ℚ), a = 5/4 ∧ r = 1/3) → 
  ∑' n : ℕ, ((5/4 : ℚ) * (1/3 : ℚ) ^ n) = (15/8 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l200_20049


namespace NUMINAMATH_GPT_new_student_weight_l200_20018

theorem new_student_weight (W_new : ℝ) (W : ℝ) (avg_decrease : ℝ) (num_students : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_decrease = 5 → old_weight = 86 → num_students = 8 →
  W_new = W - old_weight + new_weight → W_new = W - avg_decrease * num_students →
  new_weight = 46 :=
by
  intros avg_decrease_eq old_weight_eq num_students_eq W_new_eq avg_weight_decrease_eq
  rw [avg_decrease_eq, old_weight_eq, num_students_eq] at *
  sorry

end NUMINAMATH_GPT_new_student_weight_l200_20018


namespace NUMINAMATH_GPT_value_of_x_l200_20050

theorem value_of_x (x : ℝ) : (9 - x) ^ 2 = x ^ 2 → x = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l200_20050


namespace NUMINAMATH_GPT_power_function_evaluation_l200_20011

theorem power_function_evaluation (f : ℝ → ℝ) (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = (Real.sqrt 2) / 2) :
  f 4 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_power_function_evaluation_l200_20011


namespace NUMINAMATH_GPT_g_prime_positive_l200_20078

noncomputable def f (a x : ℝ) := a * x - a * x ^ 2 - Real.log x

noncomputable def g (a x : ℝ) := -2 * (a * x - a * x ^ 2 - Real.log x) - (2 * a + 1) * x ^ 2 + a * x

def g_zero (a x1 x2 : ℝ) := g a x1 = 0 ∧ g a x2 = 0

def x1_x2_condition (x1 x2 : ℝ) := x1 < x2 ∧ x2 < 4 * x1

theorem g_prime_positive (a x1 x2 : ℝ) (h1 : g_zero a x1 x2) (h2 : x1_x2_condition x1 x2) :
  (deriv (g a) ((2 * x1 + x2) / 3)) > 0 := by
  sorry

end NUMINAMATH_GPT_g_prime_positive_l200_20078


namespace NUMINAMATH_GPT_olympiad_divisors_l200_20006

theorem olympiad_divisors :
  {n : ℕ | n > 0 ∧ n ∣ (1998 + n)} = {n : ℕ | n > 0 ∧ n ∣ 1998} :=
by {
  sorry
}

end NUMINAMATH_GPT_olympiad_divisors_l200_20006


namespace NUMINAMATH_GPT_total_handshakes_is_72_l200_20034

-- Define the conditions
def number_of_players_per_team := 6
def number_of_teams := 2
def number_of_referees := 3

-- Define the total number of players
def total_players := number_of_teams * number_of_players_per_team

-- Define the total number of handshakes between players of different teams
def team_handshakes := number_of_players_per_team * number_of_players_per_team

-- Define the total number of handshakes between players and referees
def player_referee_handshakes := total_players * number_of_referees

-- Define the total number of handshakes
def total_handshakes := team_handshakes + player_referee_handshakes

-- Prove that the total number of handshakes is 72
theorem total_handshakes_is_72 : total_handshakes = 72 := by
  sorry

end NUMINAMATH_GPT_total_handshakes_is_72_l200_20034


namespace NUMINAMATH_GPT_probability_cooking_is_one_fourth_l200_20023
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_probability_cooking_is_one_fourth_l200_20023


namespace NUMINAMATH_GPT_values_of_x_for_f_l200_20069

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem values_of_x_for_f (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_monotonically_increasing_on_nonneg f) : 
  (∀ x : ℝ, f (2*x - 1) < f 3 ↔ (-1 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_values_of_x_for_f_l200_20069


namespace NUMINAMATH_GPT_vertex_and_maximum_l200_20061

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 9

-- Prove that the vertex of the parabola quadratic is (1, -6) and it is a maximum point
theorem vertex_and_maximum :
  (∃ x y : ℝ, (quadratic x = y) ∧ (x = 1) ∧ (y = -6)) ∧
  (∀ x : ℝ, quadratic x ≤ quadratic 1) :=
sorry

end NUMINAMATH_GPT_vertex_and_maximum_l200_20061


namespace NUMINAMATH_GPT_required_folders_l200_20001

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_count : ℕ := 24
def total_cost : ℝ := 30

theorem required_folders : ∃ (folders : ℕ), folders = 20 ∧ 
  (pencil_count * pencil_cost + folders * folder_cost = total_cost) :=
sorry

end NUMINAMATH_GPT_required_folders_l200_20001


namespace NUMINAMATH_GPT_salon_fingers_l200_20020

theorem salon_fingers (clients non_clients total_fingers cost_per_client total_earnings : Nat)
  (h1 : cost_per_client = 20)
  (h2 : total_earnings = 200)
  (h3 : total_fingers = 210)
  (h4 : non_clients = 11)
  (h_clients : clients = total_earnings / cost_per_client)
  (h_people : total_fingers / 10 = clients + non_clients) :
  10 = total_fingers / (clients + non_clients) :=
by
  sorry

end NUMINAMATH_GPT_salon_fingers_l200_20020


namespace NUMINAMATH_GPT_four_xyz_value_l200_20076

theorem four_xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 4 * x * y * z = 48 := by
  sorry

end NUMINAMATH_GPT_four_xyz_value_l200_20076


namespace NUMINAMATH_GPT_gcd_9011_4379_l200_20053

def a : ℕ := 9011
def b : ℕ := 4379

theorem gcd_9011_4379 : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_9011_4379_l200_20053


namespace NUMINAMATH_GPT_vector_dot_product_l200_20092

def vector := ℝ × ℝ

def collinear (a b : vector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

noncomputable def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (k : ℝ) (h_collinear : collinear (3 / 2, 1) (3, k))
  (h_k : k = 2) :
  dot_product ((3 / 2, 1) - (3, k)) (2 * (3 / 2, 1) + (3, k)) = -13 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l200_20092


namespace NUMINAMATH_GPT_max_silver_coins_l200_20028

theorem max_silver_coins (n : ℕ) : (n < 150) ∧ (n % 15 = 3) → n = 138 :=
by
  sorry

end NUMINAMATH_GPT_max_silver_coins_l200_20028


namespace NUMINAMATH_GPT_total_profit_l200_20027

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end NUMINAMATH_GPT_total_profit_l200_20027


namespace NUMINAMATH_GPT_simplify_expression_l200_20087

theorem simplify_expression (k : ℤ) (c d : ℤ) 
(h1 : (5 * k + 15) / 5 = c * k + d) 
(h2 : ∀ k, d + c * k = k + 3) : 
c / d = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l200_20087


namespace NUMINAMATH_GPT_possible_arrangements_count_l200_20075

-- Define students as a type
inductive Student
| A | B | C | D | E | F

open Student

-- Define Club as a type
inductive Club
| A | B | C

open Club

-- Define the arrangement constraints
structure Arrangement :=
(assignment : Student → Club)
(club_size : Club → Nat)
(A_and_B_same_club : assignment A = assignment B)
(C_and_D_diff_clubs : assignment C ≠ assignment D)
(club_A_size : club_size A = 3)
(all_clubs_nonempty : ∀ c : Club, club_size c > 0)

-- Define the possible number of arrangements
def arrangement_count (a : Arrangement) : Nat := sorry

-- Theorem stating the number of valid arrangements
theorem possible_arrangements_count : ∃ a : Arrangement, arrangement_count a = 24 := sorry

end NUMINAMATH_GPT_possible_arrangements_count_l200_20075


namespace NUMINAMATH_GPT_cube_root_eval_l200_20033

noncomputable def cube_root_nested (N : ℝ) : ℝ := (N * (N * (N * (N)))) ^ (1/81)

theorem cube_root_eval (N : ℝ) (h : N > 1) : 
  cube_root_nested N = N ^ (40 / 81) := 
sorry

end NUMINAMATH_GPT_cube_root_eval_l200_20033


namespace NUMINAMATH_GPT_complement_intersection_l200_20009

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {x | |x| < 1}
def B : Set R := {y | ∃ x, y = 2^x + 1}
def complement_A : Set R := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_intersection (x : R) : 
  x ∈ (complement_A R) ∩ B R ↔ x > 1 :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l200_20009


namespace NUMINAMATH_GPT_min_M_inequality_l200_20014

noncomputable def M_min : ℝ := 9 * Real.sqrt 2 / 32

theorem min_M_inequality :
  ∀ (a b c : ℝ),
    abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
    ≤ M_min * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_min_M_inequality_l200_20014


namespace NUMINAMATH_GPT_find_x_squared_plus_inv_squared_l200_20035

noncomputable def x : ℝ := sorry

theorem find_x_squared_plus_inv_squared (h : x^4 + 1 / x^4 = 240) : x^2 + 1 / x^2 = Real.sqrt 242 := by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_inv_squared_l200_20035


namespace NUMINAMATH_GPT_total_voters_l200_20066

theorem total_voters (x : ℝ)
  (h1 : 0.35 * x + 80 = (0.35 * x + 80) + 0.65 * x - (0.65 * x - 0.45 * (x + 80)))
  (h2 : 0.45 * (x + 80) = 0.65 * x) : 
  x + 80 = 260 := by
  -- We'll provide the proof here
  sorry

end NUMINAMATH_GPT_total_voters_l200_20066


namespace NUMINAMATH_GPT_find_width_l200_20054

variable (a b : ℝ)

def perimeter : ℝ := 6 * a + 4 * b
def length : ℝ := 2 * a + b
def width : ℝ := a + b

theorem find_width (h : perimeter a b = 6 * a + 4 * b)
                   (h₂ : length a b = 2 * a + b) : width a b = (perimeter a b) / 2 - length a b := by
  sorry

end NUMINAMATH_GPT_find_width_l200_20054


namespace NUMINAMATH_GPT_symmetry_propositions_l200_20099

noncomputable def verify_symmetry_conditions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  Prop :=
  -- This defines the propositions to be proven
  (∀ x : ℝ, a^x - 1 = a^(-x) - 1) ∧
  (∀ x : ℝ, a^(x - 2) = a^(2 - x)) ∧
  (∀ x : ℝ, a^(x + 2) = a^(2 - x))

-- Create the problem statement
theorem symmetry_propositions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  verify_symmetry_conditions a h1 h2 :=
sorry

end NUMINAMATH_GPT_symmetry_propositions_l200_20099


namespace NUMINAMATH_GPT_complement_of_M_in_U_l200_20060

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}
def complement_U_M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = complement_U_M :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l200_20060


namespace NUMINAMATH_GPT_distance_between_street_lights_l200_20059

theorem distance_between_street_lights :
  ∀ (n : ℕ) (L : ℝ), n = 18 → L = 16.4 → 8 > 0 →
  (L / (8 : ℕ) = 2.05) :=
by
  intros n L h_n h_L h_nonzero
  sorry

end NUMINAMATH_GPT_distance_between_street_lights_l200_20059


namespace NUMINAMATH_GPT_circle_circumference_ratio_l200_20083

theorem circle_circumference_ratio (q r p : ℝ) (hq : p = q + r) : 
  (2 * Real.pi * q + 2 * Real.pi * r) / (2 * Real.pi * p) = 1 :=
by
  sorry

end NUMINAMATH_GPT_circle_circumference_ratio_l200_20083


namespace NUMINAMATH_GPT_meals_calculation_l200_20036

def combined_meals (k a : ℕ) : ℕ :=
  k + a

theorem meals_calculation :
  ∀ (k a : ℕ), k = 8 → (2 * a = k) → combined_meals k a = 12 :=
  by
    intros k a h1 h2
    rw [h1] at h2
    have ha : a = 4 := by linarith
    rw [h1, ha]
    unfold combined_meals
    sorry

end NUMINAMATH_GPT_meals_calculation_l200_20036


namespace NUMINAMATH_GPT_count_four_digit_numbers_with_thousands_digit_one_l200_20074

theorem count_four_digit_numbers_with_thousands_digit_one : 
  ∃ N : ℕ, N = 1000 ∧ (∀ n : ℕ, 1000 ≤ n ∧ n < 2000 → (n / 1000 = 1)) :=
sorry

end NUMINAMATH_GPT_count_four_digit_numbers_with_thousands_digit_one_l200_20074


namespace NUMINAMATH_GPT_multiplication_of_variables_l200_20012

theorem multiplication_of_variables 
  (a b c d : ℚ)
  (h1 : 3 * a + 2 * b + 4 * c + 6 * d = 48)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : 2 * c - 2 = d) :
  a * b * c * d = -58735360 / 81450625 := 
sorry

end NUMINAMATH_GPT_multiplication_of_variables_l200_20012


namespace NUMINAMATH_GPT_ab_div_c_eq_one_l200_20094

theorem ab_div_c_eq_one (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hne1 : A ≠ B) (hne2 : A ≠ C) (hne3 : B ≠ C) :
  (1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / 1))) → (A + B) / C = 1 :=
by sorry

end NUMINAMATH_GPT_ab_div_c_eq_one_l200_20094


namespace NUMINAMATH_GPT_calc_154_1836_minus_54_1836_l200_20042

-- Statement of the problem in Lean 4
theorem calc_154_1836_minus_54_1836 : 154 * 1836 - 54 * 1836 = 183600 :=
by
  sorry

end NUMINAMATH_GPT_calc_154_1836_minus_54_1836_l200_20042


namespace NUMINAMATH_GPT_arithmetic_sequence_max_sum_l200_20043

-- Condition: first term is 23
def a1 : ℤ := 23

-- Condition: common difference is -2
def d : ℤ := -2

-- Sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Problem Statement: Prove the maximum value of Sn(n)
theorem arithmetic_sequence_max_sum : ∃ n : ℕ, Sn n = 144 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_sum_l200_20043


namespace NUMINAMATH_GPT_num_ways_to_place_balls_in_boxes_l200_20005

theorem num_ways_to_place_balls_in_boxes (num_balls num_boxes : ℕ) (hB : num_balls = 4) (hX : num_boxes = 3) : 
  (num_boxes ^ num_balls) = 81 := by
  rw [hB, hX]
  sorry

end NUMINAMATH_GPT_num_ways_to_place_balls_in_boxes_l200_20005


namespace NUMINAMATH_GPT_determine_x_l200_20048

theorem determine_x
  (w : ℤ) (z : ℤ) (y : ℤ) (x : ℤ)
  (h₁ : w = 90)
  (h₂ : z = w + 25)
  (h₃ : y = z + 12)
  (h₄ : x = y + 7) : x = 134 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l200_20048


namespace NUMINAMATH_GPT_equilateral_triangle_dot_product_l200_20064

noncomputable def dot_product_sum (a b c : ℝ) := 
  a * b + b * c + c * a

theorem equilateral_triangle_dot_product 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A = 1)
  (h2 : B = 1)
  (h3 : C = 1)
  (h4 : a = 1)
  (h5 : b = 1)
  (h6 : c = 1) :
  dot_product_sum a b c = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_equilateral_triangle_dot_product_l200_20064


namespace NUMINAMATH_GPT_correct_weight_misread_l200_20030

theorem correct_weight_misread (initial_avg correct_avg : ℝ) (num_boys : ℕ) (misread_weight : ℝ)
  (h_initial : initial_avg = 58.4) (h_correct : correct_avg = 58.85) (h_num_boys : num_boys = 20)
  (h_misread_weight : misread_weight = 56) :
  ∃ x : ℝ, x = 65 :=
by
  sorry

end NUMINAMATH_GPT_correct_weight_misread_l200_20030


namespace NUMINAMATH_GPT_base8_units_digit_l200_20052

theorem base8_units_digit (n m : ℕ) (h1 : n = 348) (h2 : m = 27) : 
  (n * m % 8) = 4 := sorry

end NUMINAMATH_GPT_base8_units_digit_l200_20052


namespace NUMINAMATH_GPT_marcia_oranges_l200_20008

noncomputable def averageCost
  (appleCost bananaCost orangeCost : ℝ) 
  (numApples numBananas numOranges : ℝ) : ℝ :=
  (numApples * appleCost + numBananas * bananaCost + numOranges * orangeCost) /
  (numApples + numBananas + numOranges)

theorem marcia_oranges : 
  ∀ (appleCost bananaCost orangeCost avgCost : ℝ) 
  (numApples numBananas numOranges : ℝ),
  appleCost = 2 → 
  bananaCost = 1 → 
  orangeCost = 3 → 
  numApples = 12 → 
  numBananas = 4 → 
  avgCost = 2 → 
  averageCost appleCost bananaCost orangeCost numApples numBananas numOranges = avgCost → 
  numOranges = 4 :=
by 
  intros appleCost bananaCost orangeCost avgCost numApples numBananas numOranges
         h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_marcia_oranges_l200_20008


namespace NUMINAMATH_GPT_remainder_of_2357916_div_8_l200_20017

theorem remainder_of_2357916_div_8 : (2357916 % 8) = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2357916_div_8_l200_20017
