import Mathlib

namespace NUMINAMATH_GPT_water_volume_per_minute_l1434_143458

theorem water_volume_per_minute 
  (depth : ℝ) (width : ℝ) (flow_kmph : ℝ)
  (h_depth : depth = 8) (h_width : width = 25) (h_flow_rate : flow_kmph = 8) :
  (width * depth * (flow_kmph * 1000 / 60)) = 26666.67 :=
by 
  have flow_m_per_min := flow_kmph * 1000 / 60
  have area := width * depth
  have volume_per_minute := area * flow_m_per_min
  sorry

end NUMINAMATH_GPT_water_volume_per_minute_l1434_143458


namespace NUMINAMATH_GPT_range_of_j_l1434_143441

def h (x: ℝ) : ℝ := 2 * x + 1
def j (x: ℝ) : ℝ := h (h (h (h (h x))))

theorem range_of_j :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → -1 ≤ j x ∧ j x ≤ 127 :=
by 
  intros x hx
  sorry

end NUMINAMATH_GPT_range_of_j_l1434_143441


namespace NUMINAMATH_GPT_central_angle_l1434_143415

variable (O : Type)
variable (A B C : O)
variable (angle_ABC : ℝ) 

theorem central_angle (h : angle_ABC = 50) : 2 * angle_ABC = 100 := by
  sorry

end NUMINAMATH_GPT_central_angle_l1434_143415


namespace NUMINAMATH_GPT_string_length_l1434_143448

theorem string_length (cylinder_circumference : ℝ)
  (total_loops : ℕ) (post_height : ℝ)
  (height_per_loop : ℝ := post_height / total_loops)
  (hypotenuse_per_loop : ℝ := Real.sqrt (height_per_loop ^ 2 + cylinder_circumference ^ 2))
  : total_loops = 5 → cylinder_circumference = 4 → post_height = 15 → hypotenuse_per_loop * total_loops = 25 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_string_length_l1434_143448


namespace NUMINAMATH_GPT_range_of_m_l1434_143416

theorem range_of_m (x m : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1) (h2 : |x - m| ≤ 2) : -1 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1434_143416


namespace NUMINAMATH_GPT_roots_reciprocal_l1434_143496

theorem roots_reciprocal (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_roots : a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) (h_cond : b^2 = 4 * a * c) : r * s = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_roots_reciprocal_l1434_143496


namespace NUMINAMATH_GPT_weight_difference_l1434_143436

variables (W_A W_B W_C W_D W_E : ℝ)

def condition1 : Prop := (W_A + W_B + W_C) / 3 = 84
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 80
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 79
def condition4 : Prop := W_A = 80

theorem weight_difference (h1 : condition1 W_A W_B W_C) 
                          (h2 : condition2 W_A W_B W_C W_D) 
                          (h3 : condition3 W_B W_C W_D W_E) 
                          (h4 : condition4 W_A) : 
                          W_E - W_D = 8 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_l1434_143436


namespace NUMINAMATH_GPT_solve_a_l1434_143420

def custom_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a :
  ∃ a : ℝ, custom_op a 7 = -20 ∧ a = 29 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_a_l1434_143420


namespace NUMINAMATH_GPT_sum_abs_a_l1434_143421

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem sum_abs_a :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + 
   |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 67) :=
by
  sorry

end NUMINAMATH_GPT_sum_abs_a_l1434_143421


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1434_143402

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l1434_143402


namespace NUMINAMATH_GPT_find_x_l1434_143444

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1434_143444


namespace NUMINAMATH_GPT_compound_bar_chart_must_clearly_indicate_legend_l1434_143440

-- Definitions of the conditions
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_bars_of_different_colors : Bool

-- The theorem stating that a compound bar chart must clearly indicate the legend
theorem compound_bar_chart_must_clearly_indicate_legend 
  (chart : CompoundBarChart)
  (distinguishes_quantities : chart.distinguishes_two_quantities = true)
  (uses_colors : chart.uses_bars_of_different_colors = true) :
  ∃ legend : String, legend ≠ "" := by
  sorry

end NUMINAMATH_GPT_compound_bar_chart_must_clearly_indicate_legend_l1434_143440


namespace NUMINAMATH_GPT_weight_of_fresh_grapes_is_40_l1434_143413

-- Define the weight of fresh grapes and dried grapes
variables (F D : ℝ)

-- Fresh grapes contain 90% water by weight, so 10% is non-water
def fresh_grapes_non_water_content (F : ℝ) : ℝ := 0.10 * F

-- Dried grapes contain 20% water by weight, so 80% is non-water
def dried_grapes_non_water_content (D : ℝ) : ℝ := 0.80 * D

-- Given condition: weight of dried grapes is 5 kg
def weight_of_dried_grapes : ℝ := 5

-- The main theorem to prove
theorem weight_of_fresh_grapes_is_40 :
  fresh_grapes_non_water_content F = dried_grapes_non_water_content weight_of_dried_grapes →
  F = 40 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_fresh_grapes_is_40_l1434_143413


namespace NUMINAMATH_GPT_number_of_ways_to_choose_officers_l1434_143492

open Nat

theorem number_of_ways_to_choose_officers (n : ℕ) (h : n = 8) : 
  n * (n - 1) * (n - 2) = 336 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_officers_l1434_143492


namespace NUMINAMATH_GPT_total_charge_for_2_hours_l1434_143491

theorem total_charge_for_2_hours (A F : ℕ) (h1 : F = A + 35) (h2 : F + 4 * A = 350) : 
  F + A = 161 := 
by 
  sorry

end NUMINAMATH_GPT_total_charge_for_2_hours_l1434_143491


namespace NUMINAMATH_GPT_LCM_of_two_numbers_l1434_143438

theorem LCM_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 14) (h2 : a * b = 2562) : Nat.lcm a b = 183 :=
by
  sorry

end NUMINAMATH_GPT_LCM_of_two_numbers_l1434_143438


namespace NUMINAMATH_GPT_mary_needs_6_cups_l1434_143482
-- We import the whole Mathlib library first.

-- We define the conditions and the question.
def total_cups : ℕ := 8
def cups_added : ℕ := 2
def cups_needed : ℕ := total_cups - cups_added

-- We state the theorem we need to prove.
theorem mary_needs_6_cups : cups_needed = 6 :=
by
  -- We use a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_mary_needs_6_cups_l1434_143482


namespace NUMINAMATH_GPT_first_player_winning_strategy_l1434_143404
noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem first_player_winning_strategy (x1 y1 : ℕ)
    (h1 : x1 > 0) (h2 : y1 > 0) :
    (x1 / y1 = 1) ∨ 
    (x1 / y1 > golden_ratio) ∨ 
    (x1 / y1 < 1 / golden_ratio) :=
sorry

end NUMINAMATH_GPT_first_player_winning_strategy_l1434_143404


namespace NUMINAMATH_GPT_find_two_primes_l1434_143451

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m ≠ n → n % m ≠ 0

-- Prove the existence of two specific prime numbers with the desired properties
theorem find_two_primes :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p = 2 ∧ q = 5 ∧ is_prime (p + q) ∧ is_prime (q - p) :=
by
  exists 2
  exists 5
  repeat {split}
  sorry

end NUMINAMATH_GPT_find_two_primes_l1434_143451


namespace NUMINAMATH_GPT_factorize_expression_l1434_143453

theorem factorize_expression (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l1434_143453


namespace NUMINAMATH_GPT_problem_1_problem_2_l1434_143490

-- Definition f
def f (a x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Problem 1: If a = 1, prove ∀ x, f(1, x) ≤ 2
theorem problem_1 : (∀ x : ℝ, f 1 x ≤ 2) :=
sorry

-- Problem 2: The range of a for which f has a maximum value is -2 ≤ a ≤ 2
theorem problem_2 : (∀ a : ℝ, (∀ x : ℝ, (2 * x - 1 > 0 -> (f a x) ≤ (f a ((4 - a) / (2 * (4 - a))))) 
                        ∧ (2 * x - 1 ≤ 0 -> (f a x) ≤ (f a (1 - 2 / (1 - a))))) 
                        ↔ -2 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1434_143490


namespace NUMINAMATH_GPT_solution_for_x2_l1434_143425

def eq1 (x : ℝ) := 2 * x = 6
def eq2 (x : ℝ) := x + 2 = 0
def eq3 (x : ℝ) := x - 5 = 3
def eq4 (x : ℝ) := 3 * x - 6 = 0

theorem solution_for_x2 : ∀ x : ℝ, x = 2 → ¬eq1 x ∧ ¬eq2 x ∧ ¬eq3 x ∧ eq4 x :=
by 
  sorry

end NUMINAMATH_GPT_solution_for_x2_l1434_143425


namespace NUMINAMATH_GPT_A_and_C_work_together_in_2_hours_l1434_143467

theorem A_and_C_work_together_in_2_hours
  (A_rate : ℚ)
  (B_rate : ℚ)
  (C_rate : ℚ)
  (A_4_hours : A_rate = 1 / 4)
  (B_12_hours : B_rate = 1 / 12)
  (B_and_C_3_hours : B_rate + C_rate = 1 / 3) :
  (A_rate + C_rate = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_A_and_C_work_together_in_2_hours_l1434_143467


namespace NUMINAMATH_GPT_kaashish_problem_l1434_143487

theorem kaashish_problem (x y : ℤ) (h : 2 * x + 3 * y = 100) (k : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_kaashish_problem_l1434_143487


namespace NUMINAMATH_GPT_sum_of_digits_is_3_l1434_143473

-- We introduce variables for the digits a and b, and the number
variables (a b : ℕ)

-- Conditions: a and b must be digits, and the number must satisfy the given equation
-- One half of (10a + b) exceeds its one fourth by 3
def valid_digits (a b : ℕ) : Prop := a < 10 ∧ b < 10
def equation_condition (a b : ℕ) : Prop := 2 * (10 * a + b) = (10 * a + b) + 12

-- The number is two digits number
def two_digits_number (a b : ℕ) : ℕ := 10 * a + b

-- Final statement combining all conditions and proving the desired sum of digits
theorem sum_of_digits_is_3 : 
  ∀ (a b : ℕ), valid_digits a b → equation_condition a b → a + b = 3 := 
by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_digits_is_3_l1434_143473


namespace NUMINAMATH_GPT_exponent_problem_proof_l1434_143499

theorem exponent_problem_proof :
  3 * 3^4 - 27^60 / 27^58 = -486 :=
by
  sorry

end NUMINAMATH_GPT_exponent_problem_proof_l1434_143499


namespace NUMINAMATH_GPT_product_of_successive_numbers_l1434_143433

-- Given conditions
def n : ℝ := 51.49757275833493

-- Proof statement
theorem product_of_successive_numbers : n * (n + 1) = 2703.0000000000005 :=
by
  -- Proof would be supplied here
  sorry

end NUMINAMATH_GPT_product_of_successive_numbers_l1434_143433


namespace NUMINAMATH_GPT_sophia_collection_value_l1434_143468

-- Define the conditions
def stamps_count : ℕ := 24
def partial_stamps_count : ℕ := 8
def partial_value : ℤ := 40
def stamp_value_per_each : ℤ := partial_value / partial_stamps_count
def total_value : ℤ := stamps_count * stamp_value_per_each

-- Statement of the conclusion that needs proving
theorem sophia_collection_value :
  total_value = 120 := by
  sorry

end NUMINAMATH_GPT_sophia_collection_value_l1434_143468


namespace NUMINAMATH_GPT_domain_of_sqrt_function_l1434_143443

theorem domain_of_sqrt_function :
  {x : ℝ | 0 ≤ x + 1} = {x : ℝ | -1 ≤ x} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_sqrt_function_l1434_143443


namespace NUMINAMATH_GPT_parallel_slope_l1434_143449

theorem parallel_slope {x1 y1 x2 y2 : ℝ} (h : x1 = 3 ∧ y1 = -2 ∧ x2 = 1 ∧ y2 = 5) :
    let slope := (y2 - y1) / (x2 - x1)
    slope = -7 / 2 := 
by 
    sorry

end NUMINAMATH_GPT_parallel_slope_l1434_143449


namespace NUMINAMATH_GPT_find_m_range_l1434_143470

-- Defining the function and conditions
variable {f : ℝ → ℝ}
variable {m : ℝ}

-- Prove if given the conditions, then the range of m is as specified
theorem find_m_range (h1 : ∀ x, f (-x) = -f x) 
                     (h2 : ∀ x, -2 < x ∧ x < 2 → f (x) > f (x+1)) 
                     (h3 : -2 < m - 1 ∧ m - 1 < 2) 
                     (h4 : -2 < 2 * m - 1 ∧ 2 * m - 1 < 2) 
                     (h5 : f (m - 1) + f (2 * m - 1) > 0) :
  -1/2 < m ∧ m < 2/3 :=
sorry

end NUMINAMATH_GPT_find_m_range_l1434_143470


namespace NUMINAMATH_GPT_probability_of_same_suit_or_number_but_not_both_l1434_143435

def same_suit_or_number_but_not_both : Prop :=
  let total_outcomes := 52 * 52
  let prob_same_suit := 12 / 51
  let prob_same_number := 3 / 51
  let prob_same_suit_and_number := 1 / 51
  (prob_same_suit + prob_same_number - 2 * prob_same_suit_and_number) = 15 / 52

theorem probability_of_same_suit_or_number_but_not_both :
  same_suit_or_number_but_not_both :=
by sorry

end NUMINAMATH_GPT_probability_of_same_suit_or_number_but_not_both_l1434_143435


namespace NUMINAMATH_GPT_tens_place_of_8_pow_1234_l1434_143430

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_tens_place_of_8_pow_1234_l1434_143430


namespace NUMINAMATH_GPT_largest_prime_factor_problem_l1434_143461

def largest_prime_factor (n : ℕ) : ℕ :=
  -- This function calculates the largest prime factor of n
  sorry

theorem largest_prime_factor_problem :
  largest_prime_factor 57 = 19 ∧
  largest_prime_factor 133 = 19 ∧
  ∀ n, n = 63 ∨ n = 85 ∨ n = 143 → largest_prime_factor n < 19 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_problem_l1434_143461


namespace NUMINAMATH_GPT_mindy_mork_earnings_ratio_l1434_143457

theorem mindy_mork_earnings_ratio (M K : ℝ) (h1 : 0.20 * M + 0.30 * K = 0.225 * (M + K)) : M / K = 3 :=
by
  sorry

end NUMINAMATH_GPT_mindy_mork_earnings_ratio_l1434_143457


namespace NUMINAMATH_GPT_fraction_of_Bs_l1434_143498

theorem fraction_of_Bs 
  (num_students : ℕ)
  (As_fraction : ℚ)
  (Cs_fraction : ℚ)
  (Ds_number : ℕ)
  (total_students : ℕ) 
  (h1 : As_fraction = 1 / 5) 
  (h2 : Cs_fraction = 1 / 2) 
  (h3 : Ds_number = 40) 
  (h4 : total_students = 800) : 
  num_students / total_students = 1 / 4 :=
by
sorry

end NUMINAMATH_GPT_fraction_of_Bs_l1434_143498


namespace NUMINAMATH_GPT_difference_between_x_and_y_l1434_143494

theorem difference_between_x_and_y 
  (x y : ℕ) 
  (h1 : 3 ^ x * 4 ^ y = 531441) 
  (h2 : x = 12) : x - y = 12 := 
by 
  sorry

end NUMINAMATH_GPT_difference_between_x_and_y_l1434_143494


namespace NUMINAMATH_GPT_lcm_36_105_l1434_143427

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_GPT_lcm_36_105_l1434_143427


namespace NUMINAMATH_GPT_find_a₁_l1434_143472

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ n

noncomputable def sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

variables (a₁ q : ℝ)
-- Condition: The common ratio should not be 1.
axiom hq : q ≠ 1
-- Condition: Second term of the sequence a₂ = 1
axiom ha₂ : geometric_sequence a₁ q 1 = 1
-- Condition: 9S₃ = S₆
axiom hsum : 9 * sequence_sum a₁ q 3 = sequence_sum a₁ q 6

theorem find_a₁ : a₁ = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_find_a₁_l1434_143472


namespace NUMINAMATH_GPT_girls_additional_laps_l1434_143476

def distance_per_lap : ℚ := 1 / 6
def boys_laps : ℕ := 34
def boys_distance : ℚ := boys_laps * distance_per_lap
def girls_distance : ℚ := 9
def additional_distance : ℚ := girls_distance - boys_distance
def additional_laps (distance : ℚ) (lap_distance : ℚ) : ℚ := distance / lap_distance

theorem girls_additional_laps :
  additional_laps additional_distance distance_per_lap = 20 := 
by
  sorry

end NUMINAMATH_GPT_girls_additional_laps_l1434_143476


namespace NUMINAMATH_GPT_greatest_integer_with_gcf_5_l1434_143442

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_with_gcf_5_l1434_143442


namespace NUMINAMATH_GPT_grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l1434_143409

def can_jump (x : Int) : Prop :=
  ∃ (k m : Int), x = k * 36 + m * 14

theorem grasshopper_cannot_move_3_cm :
  ¬ can_jump 3 :=
by
  sorry

theorem grasshopper_can_move_2_cm :
  can_jump 2 :=
by
  sorry

theorem grasshopper_can_move_1234_cm :
  can_jump 1234 :=
by
  sorry

end NUMINAMATH_GPT_grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l1434_143409


namespace NUMINAMATH_GPT_circle_radius_c_eq_32_l1434_143486

theorem circle_radius_c_eq_32 :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x-4)^2 + (y+5)^2 = 9) :=
by
  use 32
  sorry

end NUMINAMATH_GPT_circle_radius_c_eq_32_l1434_143486


namespace NUMINAMATH_GPT_solve_for_c_l1434_143478

theorem solve_for_c (a b c : ℝ) (h : 1/a - 1/b = 2/c) : c = (a * b * (b - a)) / 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_c_l1434_143478


namespace NUMINAMATH_GPT_a_sufficient_but_not_necessary_l1434_143477

theorem a_sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → |a| = 1) ∧ (¬ (|a| = 1 → a = 1)) :=
by 
  sorry

end NUMINAMATH_GPT_a_sufficient_but_not_necessary_l1434_143477


namespace NUMINAMATH_GPT__l1434_143429

section BoxProblem

open Nat

def volume_box (l w h : ℕ) : ℕ := l * w * h
def volume_block (l w h : ℕ) : ℕ := l * w * h

def can_fit_blocks (box_l box_w box_h block_l block_w block_h n_blocks : ℕ) : Prop :=
  (volume_box box_l box_w box_h) = (n_blocks * volume_block block_l block_w block_h)

example : can_fit_blocks 4 3 3 3 2 1 6 :=
by
  -- calculation that proves the theorem goes here, but no need to provide proof steps
  sorry

end BoxProblem

end NUMINAMATH_GPT__l1434_143429


namespace NUMINAMATH_GPT_find_x_floor_mul_eq_100_l1434_143426

theorem find_x_floor_mul_eq_100 (x : ℝ) (h1 : 0 < x) (h2 : (⌊x⌋ : ℝ) * x = 100) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_floor_mul_eq_100_l1434_143426


namespace NUMINAMATH_GPT_hyperbola_eccentricity_correct_l1434_143403

open Real

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
    let PF1 := (12 * a / 5)
    let PF2 := PF1 - 2 * a
    let c := (2 * sqrt 37 * a / 5)
    sqrt (1 + (b^2 / a^2))  -- Assuming the geometric properties hold, the eccentricity should match
-- Lean function to verify the result
def verify_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
    hyperbola_eccentricity a b ha hb = sqrt 37 / 5

-- Statement to be verified
theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    verify_eccentricity a b ha hb := sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_correct_l1434_143403


namespace NUMINAMATH_GPT_recorded_instances_l1434_143412

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end NUMINAMATH_GPT_recorded_instances_l1434_143412


namespace NUMINAMATH_GPT_delta_four_equal_zero_l1434_143428

-- Define the sequence u_n
def u (n : ℕ) : ℤ := n^3 + n

-- Define the ∆ operator
def delta1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0   => u
  | k+1 => delta1 (delta k u)

-- The theorem statement
theorem delta_four_equal_zero (n : ℕ) : delta 4 u n = 0 :=
by sorry

end NUMINAMATH_GPT_delta_four_equal_zero_l1434_143428


namespace NUMINAMATH_GPT_find_coordinates_of_B_find_equation_of_BC_l1434_143465

-- Problem 1: Prove that the coordinates of B are (10, 5)
theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0) :
  B = (10, 5) :=
sorry

-- Problem 2: Prove that the equation of line BC is 2x + 9y - 65 = 0
theorem find_equation_of_BC (A B C : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0)
  (coordinates_B : B = (10, 5)) :
  ∃ k : ℝ, ∀ P : ℝ × ℝ, (P.1 - C.1) / (P.2 - C.2) = k → 2 * P.1 + 9 * P.2 - 65 = 0 :=
sorry

end NUMINAMATH_GPT_find_coordinates_of_B_find_equation_of_BC_l1434_143465


namespace NUMINAMATH_GPT_stickers_distribution_l1434_143460

-- Definitions for initial sticker quantities and stickers given to first four friends
def initial_space_stickers : ℕ := 120
def initial_cat_stickers : ℕ := 80
def initial_dinosaur_stickers : ℕ := 150
def initial_superhero_stickers : ℕ := 45

def given_space_stickers : ℕ := 25
def given_cat_stickers : ℕ := 13
def given_dinosaur_stickers : ℕ := 33
def given_superhero_stickers : ℕ := 29

-- Definitions for remaining stickers calculation
def remaining_space_stickers : ℕ := initial_space_stickers - given_space_stickers
def remaining_cat_stickers : ℕ := initial_cat_stickers - given_cat_stickers
def remaining_dinosaur_stickers : ℕ := initial_dinosaur_stickers - given_dinosaur_stickers
def remaining_superhero_stickers : ℕ := initial_superhero_stickers - given_superhero_stickers

def total_remaining_stickers : ℕ := remaining_space_stickers + remaining_cat_stickers + remaining_dinosaur_stickers + remaining_superhero_stickers

-- Definition for number of each type of new sticker
def each_new_type_stickers : ℕ := total_remaining_stickers / 4
def remainder_stickers : ℕ := total_remaining_stickers % 4

-- Statement to be proved
theorem stickers_distribution :
  ∃ X : ℕ, X = 3 ∧ each_new_type_stickers = 73 :=
by
  sorry

end NUMINAMATH_GPT_stickers_distribution_l1434_143460


namespace NUMINAMATH_GPT_usable_area_l1434_143455

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def pond_side : ℕ := 4

theorem usable_area :
  garden_length * garden_width - pond_side * pond_side = 344 :=
by
  sorry

end NUMINAMATH_GPT_usable_area_l1434_143455


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1434_143454

noncomputable def f (x : ℝ) := abs (Real.log (x + 1))

theorem value_of_a_plus_b (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (- (b + 1) / (b + 2))) 
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) : 
  a + b = -11 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1434_143454


namespace NUMINAMATH_GPT_train_speed_120_kmph_l1434_143474

theorem train_speed_120_kmph (t : ℝ) (d : ℝ) (h_t : t = 9) (h_d : d = 300) : 
    (d / t) * 3.6 = 120 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_120_kmph_l1434_143474


namespace NUMINAMATH_GPT_fred_walking_speed_l1434_143406

/-- 
Fred and Sam are standing 55 miles apart and they start walking in a straight line toward each other
at the same time. Fred walks at a certain speed and Sam walks at a constant speed of 5 miles per hour.
Sam has walked 25 miles when they meet.
-/
theorem fred_walking_speed
  (initial_distance : ℕ) 
  (sam_speed : ℕ)
  (sam_distance : ℕ) 
  (meeting_time : ℕ)
  (fred_distance : ℕ) 
  (fred_speed : ℕ)
  (h_initial_distance : initial_distance = 55)
  (h_sam_speed : sam_speed = 5)
  (h_sam_distance : sam_distance = 25)
  (h_meeting_time : meeting_time = 5)
  (h_fred_distance : fred_distance = 30)
  (h_fred_speed : fred_speed = 6)
  : fred_speed = fred_distance / meeting_time :=
by sorry

end NUMINAMATH_GPT_fred_walking_speed_l1434_143406


namespace NUMINAMATH_GPT_combined_garden_area_l1434_143483

def garden_area (length width : ℕ) : ℕ :=
  length * width

def total_area (count length width : ℕ) : ℕ :=
  count * garden_area length width

theorem combined_garden_area :
  let M_length := 16
  let M_width := 5
  let M_count := 3
  let Ma_length := 8
  let Ma_width := 4
  let Ma_count := 2
  total_area M_count M_length M_width + total_area Ma_count Ma_length Ma_width = 304 :=
by
  sorry

end NUMINAMATH_GPT_combined_garden_area_l1434_143483


namespace NUMINAMATH_GPT_solve_for_s_l1434_143446

theorem solve_for_s (k s : ℝ) 
  (h1 : 7 = k * 3^s) 
  (h2 : 126 = k * 9^s) : 
  s = 2 + Real.log 2 / Real.log 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_s_l1434_143446


namespace NUMINAMATH_GPT_volume_removed_percentage_l1434_143452

noncomputable def volume_rect_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def volume_cube (s : ℝ) : ℝ :=
  s * s * s

noncomputable def percent_removed (original_volume removed_volume : ℝ) : ℝ :=
  (removed_volume / original_volume) * 100

theorem volume_removed_percentage :
  let l := 18
  let w := 12
  let h := 10
  let cube_side := 4
  let num_cubes := 8
  let original_volume := volume_rect_prism l w h
  let removed_volume := num_cubes * volume_cube cube_side
  percent_removed original_volume removed_volume = 23.7 := 
sorry

end NUMINAMATH_GPT_volume_removed_percentage_l1434_143452


namespace NUMINAMATH_GPT_reciprocal_sum_l1434_143405

theorem reciprocal_sum (x1 x2 x3 k : ℝ) (h : ∀ x, x^2 + k * x - k * x3 = 0 ∧ x ≠ 0 → x = x1 ∨ x = x2) :
  (1 / x1 + 1 / x2 = 1 / x3) := by
  sorry

end NUMINAMATH_GPT_reciprocal_sum_l1434_143405


namespace NUMINAMATH_GPT_oak_total_after_planting_l1434_143417

-- Let oak_current represent the current number of oak trees in the park.
def oak_current : ℕ := 9

-- Let oak_new represent the number of new oak trees being planted.
def oak_new : ℕ := 2

-- The problem is to prove the total number of oak trees after planting equals 11
theorem oak_total_after_planting : oak_current + oak_new = 11 :=
by
  sorry

end NUMINAMATH_GPT_oak_total_after_planting_l1434_143417


namespace NUMINAMATH_GPT_greene_family_total_spent_l1434_143419

def adm_cost : ℕ := 45

def food_cost : ℕ := adm_cost - 13

def total_cost : ℕ := adm_cost + food_cost

theorem greene_family_total_spent : total_cost = 77 := 
by 
  sorry

end NUMINAMATH_GPT_greene_family_total_spent_l1434_143419


namespace NUMINAMATH_GPT_lamp_turn_off_ways_l1434_143481

theorem lamp_turn_off_ways : 
  ∃ (ways : ℕ), ways = 10 ∧
  (∃ (n : ℕ) (m : ℕ), 
    n = 6 ∧  -- 6 lamps in a row
    m = 2 ∧  -- turn off 2 of them
    ways = Nat.choose (n - m + 1) m) := -- 2 adjacent lamps cannot be turned off
by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_lamp_turn_off_ways_l1434_143481


namespace NUMINAMATH_GPT_determine_triangle_ratio_l1434_143411

theorem determine_triangle_ratio (a d : ℝ) (h : (a + d) ^ 2 = (a - d) ^ 2 + a ^ 2) : a / d = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_determine_triangle_ratio_l1434_143411


namespace NUMINAMATH_GPT_amplitude_of_resultant_wave_l1434_143447

noncomputable def y1 (t : ℝ) := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) := y1 t + y2 t

theorem amplitude_of_resultant_wave :
  ∃ R : ℝ, R = 3 * Real.sqrt 5 ∧ ∀ t : ℝ, y t = R * Real.sin (100 * Real.pi * t - θ) :=
by
  let y_combined := y
  use 3 * Real.sqrt 5
  sorry

end NUMINAMATH_GPT_amplitude_of_resultant_wave_l1434_143447


namespace NUMINAMATH_GPT_investment_time_R_l1434_143485

theorem investment_time_R (x t : ℝ) 
  (h1 : 7 * 5 * x / (5 * 7 * x) = 7 / 9)
  (h2 : 3 * t * x / (5 * 7 * x) = 4 / 9) : 
  t = 140 / 27 :=
by
  -- Placeholder for the proof, which is not required in this step.
  sorry

end NUMINAMATH_GPT_investment_time_R_l1434_143485


namespace NUMINAMATH_GPT_equivalent_angle_l1434_143480

theorem equivalent_angle (θ : ℝ) : 
  (∃ k : ℤ, θ = k * 360 + 257) ↔ θ = -463 ∨ (∃ k : ℤ, θ = k * 360 + 257) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_angle_l1434_143480


namespace NUMINAMATH_GPT_trapezoid_area_l1434_143414

theorem trapezoid_area (h_base : ℕ) (sum_bases : ℕ) (height : ℕ) (hsum : sum_bases = 36) (hheight : height = 15) :
    (sum_bases * height) / 2 = 270 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1434_143414


namespace NUMINAMATH_GPT_probability_of_distinct_divisors_l1434_143450

theorem probability_of_distinct_divisors :
  ∃ (m n : ℕ), (m.gcd n = 1) ∧ (m / n) = 125 / 158081 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_distinct_divisors_l1434_143450


namespace NUMINAMATH_GPT_count_four_digit_integers_l1434_143410

theorem count_four_digit_integers :
    ∃! (a b c d : ℕ), 1 ≤ a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (10 * b + c)^2 = (10 * a + b) * (10 * c + d) := sorry

end NUMINAMATH_GPT_count_four_digit_integers_l1434_143410


namespace NUMINAMATH_GPT_same_grade_percentage_l1434_143418

theorem same_grade_percentage (total_students: ℕ)
  (a_students: ℕ) (b_students: ℕ) (c_students: ℕ) (d_students: ℕ)
  (total: total_students = 30)
  (a: a_students = 2) (b: b_students = 4) (c: c_students = 5) (d: d_students = 1)
  : (a_students + b_students + c_students + d_students) * 100 / total_students = 40 := by
  sorry

end NUMINAMATH_GPT_same_grade_percentage_l1434_143418


namespace NUMINAMATH_GPT_standard_lamp_probability_l1434_143462

-- Define the given probabilities
def P_A1 : ℝ := 0.45
def P_A2 : ℝ := 0.40
def P_A3 : ℝ := 0.15

def P_B_given_A1 : ℝ := 0.70
def P_B_given_A2 : ℝ := 0.80
def P_B_given_A3 : ℝ := 0.81

-- Define the calculation for the total probability of B
def P_B : ℝ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- The statement to prove
theorem standard_lamp_probability : P_B = 0.7565 := by sorry

end NUMINAMATH_GPT_standard_lamp_probability_l1434_143462


namespace NUMINAMATH_GPT_prime_pairs_solution_l1434_143456

def is_prime (n : ℕ) : Prop := Nat.Prime n

def conditions (p q : ℕ) : Prop := 
  p^2 ∣ q^3 + 1 ∧ q^2 ∣ p^6 - 1

theorem prime_pairs_solution :
  ({(p, q) | is_prime p ∧ is_prime q ∧ conditions p q} = {(3, 2), (2, 3)}) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_solution_l1434_143456


namespace NUMINAMATH_GPT_find_constants_l1434_143408

theorem find_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ -2 →
    (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) = 
    P / (x - 1) + Q / (x - 4) + R / (x + 2))
  → (P = 2 / 3 ∧ Q = 8 / 9 ∧ R = -5 / 9) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1434_143408


namespace NUMINAMATH_GPT_bob_questions_created_l1434_143431

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end NUMINAMATH_GPT_bob_questions_created_l1434_143431


namespace NUMINAMATH_GPT_tens_digit_of_19_pow_2023_l1434_143488

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end NUMINAMATH_GPT_tens_digit_of_19_pow_2023_l1434_143488


namespace NUMINAMATH_GPT_hannah_spent_on_dessert_l1434_143439

theorem hannah_spent_on_dessert
  (initial_money : ℕ)
  (money_left : ℕ)
  (half_spent_on_rides : ℕ)
  (total_spent : ℕ)
  (spent_on_dessert : ℕ)
  (H1 : initial_money = 30)
  (H2 : money_left = 10)
  (H3 : half_spent_on_rides = initial_money / 2)
  (H4 : total_spent = initial_money - money_left)
  (H5 : spent_on_dessert = total_spent - half_spent_on_rides) : spent_on_dessert = 5 :=
by
  sorry

end NUMINAMATH_GPT_hannah_spent_on_dessert_l1434_143439


namespace NUMINAMATH_GPT_larger_solution_quadratic_l1434_143489

theorem larger_solution_quadratic :
  (∃ a b : ℝ, a ≠ b ∧ (a = 9) ∧ (b = -2) ∧
              (∀ x : ℝ, x^2 - 7 * x - 18 = 0 → (x = a ∨ x = b))) →
  9 = max a b :=
by
  sorry

end NUMINAMATH_GPT_larger_solution_quadratic_l1434_143489


namespace NUMINAMATH_GPT_entrance_exam_correct_answers_l1434_143479

theorem entrance_exam_correct_answers (c w : ℕ) 
  (h1 : c + w = 70) 
  (h2 : 3 * c - w = 38) : 
  c = 27 := 
sorry

end NUMINAMATH_GPT_entrance_exam_correct_answers_l1434_143479


namespace NUMINAMATH_GPT_selena_taco_packages_l1434_143469

-- Define the problem conditions
def tacos_per_package : ℕ := 4
def shells_per_package : ℕ := 6
def min_tacos : ℕ := 60
def min_shells : ℕ := 60

-- Lean statement to prove the smallest number of taco packages needed
theorem selena_taco_packages :
  ∃ n : ℕ, (n * tacos_per_package ≥ min_tacos) ∧ (∃ m : ℕ, (m * shells_per_package ≥ min_shells) ∧ (n * tacos_per_package = m * shells_per_package) ∧ n = 15) := 
by {
  sorry
}

end NUMINAMATH_GPT_selena_taco_packages_l1434_143469


namespace NUMINAMATH_GPT_c_investment_l1434_143475

theorem c_investment (x : ℝ) (h1 : 5000 / (5000 + 8000 + x) * 88000 = 36000) : 
  x = 20454.5 :=
by
  sorry

end NUMINAMATH_GPT_c_investment_l1434_143475


namespace NUMINAMATH_GPT_meeting_point_2015_is_C_l1434_143484

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end NUMINAMATH_GPT_meeting_point_2015_is_C_l1434_143484


namespace NUMINAMATH_GPT_problem_U_l1434_143424

theorem problem_U :
  ( (1 : ℝ) / (4 - Real.sqrt 15) - (1 / (Real.sqrt 15 - Real.sqrt 14))
  + (1 / (Real.sqrt 14 - 3)) - (1 / (3 - Real.sqrt 12))
  + (1 / (Real.sqrt 12 - Real.sqrt 11)) ) = 10 + Real.sqrt 11 :=
by
  sorry

end NUMINAMATH_GPT_problem_U_l1434_143424


namespace NUMINAMATH_GPT_cos_sin_inequality_inequality_l1434_143466

noncomputable def proof_cos_sin_inequality (a b : ℝ) (cos_x sin_x: ℝ) : Prop :=
  (cos_x ^ 2 = a) → (sin_x ^ 2 = b) → (a + b = 1) → (1 / 4 ≤ a ^ 3 + b ^ 3 ∧ a ^ 3 + b ^ 3 ≤ 1)

theorem cos_sin_inequality_inequality (a b : ℝ) (cos_x sin_x : ℝ) :
  proof_cos_sin_inequality a b cos_x sin_x :=
  by { sorry }

end NUMINAMATH_GPT_cos_sin_inequality_inequality_l1434_143466


namespace NUMINAMATH_GPT_integral_equality_l1434_143459

theorem integral_equality :
  ∫ x in (-1 : ℝ)..(1 : ℝ), (Real.tan x) ^ 11 + (Real.cos x) ^ 21
  = 2 * ∫ x in (0 : ℝ)..(1 : ℝ), (Real.cos x) ^ 21 :=
by
  sorry

end NUMINAMATH_GPT_integral_equality_l1434_143459


namespace NUMINAMATH_GPT_infinite_series_closed_form_l1434_143434

noncomputable def series (a : ℝ) : ℝ :=
  ∑' (k : ℕ), (2 * (k + 1) - 1) / a^k

theorem infinite_series_closed_form (a : ℝ) (ha : 1 < a) : 
  series a = (a^2 + a) / (a - 1)^2 :=
sorry

end NUMINAMATH_GPT_infinite_series_closed_form_l1434_143434


namespace NUMINAMATH_GPT_first_day_revenue_l1434_143437

theorem first_day_revenue :
  ∀ (S : ℕ), (12 * S + 90 = 246) → (4 * S + 3 * 9 = 79) :=
by
  intros S h1
  sorry

end NUMINAMATH_GPT_first_day_revenue_l1434_143437


namespace NUMINAMATH_GPT_area_of_rectangle_l1434_143422

theorem area_of_rectangle (side radius length breadth : ℕ) (h1 : side^2 = 784) (h2 : radius = side) (h3 : length = radius / 4) (h4 : breadth = 5) : length * breadth = 35 :=
by
  -- proof to be filled here
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1434_143422


namespace NUMINAMATH_GPT_evaluate_five_applications_of_f_l1434_143463

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then x + 5 else -x^2 - 3

theorem evaluate_five_applications_of_f :
  f (f (f (f (f (-1))))) = -17554795004 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_five_applications_of_f_l1434_143463


namespace NUMINAMATH_GPT_find_m_value_l1434_143423

-- Definitions of the hyperbola and its focus condition
def hyperbola_eq (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / m) - (y^2 / (3 + m)) = 1

def focus_condition (m : ℝ) : Prop :=
  4 = (m) + (3 + m)

-- Theorem stating the value of m
theorem find_m_value (m : ℝ) : hyperbola_eq m → focus_condition m → m = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_m_value_l1434_143423


namespace NUMINAMATH_GPT_part1_part2_l1434_143432

-- Definition of Set A
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- Definition of Set B
def B : Set ℝ := { x | x ≥ 3 }

-- The Complement of the Intersection of A and B
def C_R (S : Set ℝ) : Set ℝ := { x | ¬ (x ∈ S) }

-- Set C
def C (a : ℝ) : Set ℝ := { x | x ≤ a }

-- Lean statement for part 1
theorem part1 : C_R (A ∩ B) = { x | x < 3 ∨ x > 6 } :=
by sorry

-- Lean statement for part 2
theorem part2 (a : ℝ) (hA_C : A ⊆ C a) : a ≥ 6 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1434_143432


namespace NUMINAMATH_GPT_katie_sold_4_bead_necklaces_l1434_143471

theorem katie_sold_4_bead_necklaces :
  ∃ (B : ℕ), 
    (∃ (G : ℕ), G = 3) ∧ 
    (∃ (C : ℕ), C = 3) ∧ 
    (∃ (T : ℕ), T = 21) ∧ 
    B * 3 + 3 * 3 = 21 :=
sorry

end NUMINAMATH_GPT_katie_sold_4_bead_necklaces_l1434_143471


namespace NUMINAMATH_GPT_probability_odd_divisor_of_15_factorial_l1434_143400

-- Define the factorial function
def fact : ℕ → ℕ
  | 0 => 1
  | (n+1) => (n+1) * fact n

-- Probability function for choosing an odd divisor
noncomputable def probability_odd_divisor (n : ℕ) : ℚ :=
  let prime_factors := [(2, 11), (3, 6), (5, 3), (7, 2), (11, 1), (13, 1)]
  let total_factors := prime_factors.foldr (λ p acc => (p.2 + 1) * acc) 1
  let odd_factors := ((prime_factors.filter (λ p => p.1 ≠ 2)).foldr (λ p acc => (p.2 + 1) * acc) 1)
  (odd_factors : ℚ) / (total_factors : ℚ)

-- Statement to prove the probability of an odd divisor
theorem probability_odd_divisor_of_15_factorial :
  probability_odd_divisor 15 = 1 / 12 :=
by
  -- Proof goes here, which is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_probability_odd_divisor_of_15_factorial_l1434_143400


namespace NUMINAMATH_GPT_hours_learning_english_each_day_l1434_143401

theorem hours_learning_english_each_day (E : ℕ) 
  (h_chinese_each_day : ∀ (d : ℕ), d = 7) 
  (days : ℕ) 
  (h_total_days : days = 5) 
  (h_total_hours : ∀ (t : ℕ), t = 65) 
  (total_learning_time : 5 * (E + 7) = 65) :
  E = 6 :=
by
  sorry

end NUMINAMATH_GPT_hours_learning_english_each_day_l1434_143401


namespace NUMINAMATH_GPT_vet_appointments_cost_l1434_143495

variable (x : ℝ)

def JohnVetAppointments (x : ℝ) : Prop := 
  (x + 0.20 * x + 0.20 * x + 100 = 660)

theorem vet_appointments_cost :
  (∃ x : ℝ, JohnVetAppointments x) → x = 400 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  simp [JohnVetAppointments] at hx
  sorry

end NUMINAMATH_GPT_vet_appointments_cost_l1434_143495


namespace NUMINAMATH_GPT_theresa_hours_l1434_143497

theorem theresa_hours (h1 h2 h3 h4 h5 h6 : ℕ) (avg : ℕ) (x : ℕ) 
  (H_cond : h1 = 10 ∧ h2 = 8 ∧ h3 = 9 ∧ h4 = 11 ∧ h5 = 6 ∧ h6 = 8)
  (H_avg : avg = 9) : 
  (h1 + h2 + h3 + h4 + h5 + h6 + x) / 7 = avg ↔ x = 11 :=
by
  sorry

end NUMINAMATH_GPT_theresa_hours_l1434_143497


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l1434_143493

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l1434_143493


namespace NUMINAMATH_GPT_kelly_peanut_weight_l1434_143407

-- Define the total weight of snacks and the weight of raisins
def total_snacks_weight : ℝ := 0.5
def raisins_weight : ℝ := 0.4

-- Define the weight of peanuts as the remaining part
def peanuts_weight : ℝ := total_snacks_weight - raisins_weight

-- Theorem stating Kelly bought 0.1 pounds of peanuts
theorem kelly_peanut_weight : peanuts_weight = 0.1 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_kelly_peanut_weight_l1434_143407


namespace NUMINAMATH_GPT_area_triangle_ABC_l1434_143445

theorem area_triangle_ABC (AB CD height : ℝ) 
  (h_parallel : AB + CD = 20)
  (h_ratio : CD = 3 * AB)
  (h_height : height = (2 * 20) / (AB + CD)) :
  (1 / 2) * AB * height = 5 := sorry

end NUMINAMATH_GPT_area_triangle_ABC_l1434_143445


namespace NUMINAMATH_GPT_avg_visitors_per_day_l1434_143464

theorem avg_visitors_per_day 
  (avg_visitors_sundays : ℕ) 
  (avg_visitors_other_days : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ)
  (hs : avg_visitors_sundays = 630)
  (ho : avg_visitors_other_days = 240)
  (td : total_days = 30)
  (sd : sundays = 4)
  (od : other_days = 26)
  : (4 * avg_visitors_sundays + 26 * avg_visitors_other_days) / 30 = 292 := 
by
  sorry

end NUMINAMATH_GPT_avg_visitors_per_day_l1434_143464
