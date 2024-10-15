import Mathlib

namespace NUMINAMATH_GPT_aaron_weekly_earnings_l343_34396

def minutes_worked_monday : ℕ := 90
def minutes_worked_tuesday : ℕ := 40
def minutes_worked_wednesday : ℕ := 135
def minutes_worked_thursday : ℕ := 45
def minutes_worked_friday : ℕ := 60
def minutes_worked_saturday1 : ℕ := 90
def minutes_worked_saturday2 : ℕ := 75
def hourly_rate : ℕ := 4

def total_minutes_worked : ℕ :=
  minutes_worked_monday + 
  minutes_worked_tuesday + 
  minutes_worked_wednesday +
  minutes_worked_thursday + 
  minutes_worked_friday +
  minutes_worked_saturday1 + 
  minutes_worked_saturday2

def total_hours_worked : ℕ := total_minutes_worked / 60

def total_earnings : ℕ := total_hours_worked * hourly_rate

theorem aaron_weekly_earnings : total_earnings = 36 := by 
  sorry -- The proof is omitted.

end NUMINAMATH_GPT_aaron_weekly_earnings_l343_34396


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l343_34342

variable (x : ℝ) (p q : Prop)

def p_condition : Prop := 0 < x ∧ x < 1
def q_condition : Prop := x^2 < 2 * x

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p_condition x → q_condition x) ∧
  ¬ (∀ x : ℝ, q_condition x → p_condition x) := by
  sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l343_34342


namespace NUMINAMATH_GPT_current_average_age_of_seven_persons_l343_34318

theorem current_average_age_of_seven_persons (T : ℕ)
  (h1 : T + 12 = 6 * 43)
  (h2 : 69 = 69)
  : (T + 69) / 7 = 45 := by
  sorry

end NUMINAMATH_GPT_current_average_age_of_seven_persons_l343_34318


namespace NUMINAMATH_GPT_line_through_origin_in_quadrants_l343_34331

theorem line_through_origin_in_quadrants (A B C : ℝ) :
  (-A * x - B * y + C = 0) ∧ (0 = 0) ∧ (exists x y, 0 < x * y) →
  (C = 0) ∧ (A * B < 0) :=
sorry

end NUMINAMATH_GPT_line_through_origin_in_quadrants_l343_34331


namespace NUMINAMATH_GPT_total_amount_correct_l343_34372

/-- Meghan has the following cash denominations: -/
def num_100_bills : ℕ := 2
def num_50_bills : ℕ := 5
def num_10_bills : ℕ := 10

/-- Value of each denomination: -/
def value_100_bill : ℕ := 100
def value_50_bill : ℕ := 50
def value_10_bill : ℕ := 10

/-- Meghan's total amount of money: -/
def total_amount : ℕ :=
  (num_100_bills * value_100_bill) +
  (num_50_bills * value_50_bill) +
  (num_10_bills * value_10_bill)

/-- The proof: -/
theorem total_amount_correct : total_amount = 550 :=
by
  -- sorry for now
  sorry

end NUMINAMATH_GPT_total_amount_correct_l343_34372


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l343_34303

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) (h : x > 0) : (∃ y : ℝ, (y < -3 ∨ y > -1) ∧ y > 0) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l343_34303


namespace NUMINAMATH_GPT_correct_transformation_l343_34300

structure Point :=
  (x : ℝ)
  (y : ℝ)

def rotate180 (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def is_rotation_180 (p p' : Point) : Prop :=
  rotate180 p = p'

theorem correct_transformation (C D : Point) (C' D' : Point) 
  (hC : C = Point.mk 3 (-2)) 
  (hC' : C' = Point.mk (-3) 2)
  (hD : D = Point.mk 2 (-5)) 
  (hD' : D' = Point.mk (-2) 5) :
  is_rotation_180 C C' ∧ is_rotation_180 D D' :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l343_34300


namespace NUMINAMATH_GPT_range_function_l343_34326

open Real

noncomputable def function_to_prove (x : ℝ) (a : ℕ) : ℝ := x + 2 * a / x

theorem range_function (a : ℕ) (h1 : a^2 - a < 2) (h2 : a ≠ 0) : 
  Set.range (function_to_prove · a) = {y : ℝ | y ≤ -2 * sqrt 2} ∪ {y : ℝ | y ≥ 2 * sqrt 2} :=
by
  sorry

end NUMINAMATH_GPT_range_function_l343_34326


namespace NUMINAMATH_GPT_problem_divisible_by_factors_l343_34305

theorem problem_divisible_by_factors (n : ℕ) (x : ℝ) : 
  ∃ k : ℝ, (x + 1)^(2 * n) - x^(2 * n) - 2 * x - 1 = k * x * (x + 1) * (2 * x + 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_divisible_by_factors_l343_34305


namespace NUMINAMATH_GPT_total_residents_l343_34347

open Set

/-- 
In a village, there are 912 residents who speak Bashkir, 
653 residents who speak Russian, 
and 435 residents who speak both languages.
Prove the total number of residents in the village is 1130.
-/
theorem total_residents (A B : Finset ℕ) (nA nB nAB : ℕ)
  (hA : nA = 912)
  (hB : nB = 653)
  (hAB : nAB = 435) :
  nA + nB - nAB = 1130 := by
  sorry

end NUMINAMATH_GPT_total_residents_l343_34347


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l343_34393

theorem convert_to_scientific_notation :
  (1670000000 : ℝ) = 1.67 * 10 ^ 9 := 
by
  sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l343_34393


namespace NUMINAMATH_GPT_number_of_digits_if_million_place_l343_34381

theorem number_of_digits_if_million_place (n : ℕ) (h : n = 1000000) : 7 = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_digits_if_million_place_l343_34381


namespace NUMINAMATH_GPT_fraction_simplification_l343_34360

theorem fraction_simplification :
  (1722 ^ 2 - 1715 ^ 2) / (1729 ^ 2 - 1708 ^ 2) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l343_34360


namespace NUMINAMATH_GPT_person_B_correct_probability_l343_34344

-- Define probabilities
def P_A_correct : ℝ := 0.4
def P_A_incorrect : ℝ := 1 - P_A_correct
def P_B_correct_if_A_incorrect : ℝ := 0.5
def P_B_correct : ℝ := P_A_incorrect * P_B_correct_if_A_incorrect

-- Theorem statement
theorem person_B_correct_probability : P_B_correct = 0.3 :=
by
  -- Problem conditions implicitly used in definitions
  sorry

end NUMINAMATH_GPT_person_B_correct_probability_l343_34344


namespace NUMINAMATH_GPT_unused_streetlights_remain_l343_34348

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end NUMINAMATH_GPT_unused_streetlights_remain_l343_34348


namespace NUMINAMATH_GPT_minimum_value_expression_l343_34369

theorem minimum_value_expression {a : ℝ} (h₀ : 1 < a) (h₁ : a < 4) : 
  (∃ m : ℝ, (∀ x : ℝ, 1 < x ∧ x < 4 → m ≤ (x / (4 - x) + 1 / (x - 1))) ∧ m = 2) :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l343_34369


namespace NUMINAMATH_GPT_total_legs_proof_l343_34320

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end NUMINAMATH_GPT_total_legs_proof_l343_34320


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l343_34307

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) : 
  x / y = Real.sqrt (17 / 8) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l343_34307


namespace NUMINAMATH_GPT_production_rate_equation_l343_34358

theorem production_rate_equation (x : ℝ) (h1 : ∀ t : ℝ, t = 600 / (x + 8)) (h2 : ∀ t : ℝ, t = 400 / x) : 
  600/(x + 8) = 400/x :=
by
  sorry

end NUMINAMATH_GPT_production_rate_equation_l343_34358


namespace NUMINAMATH_GPT_pump_B_rate_l343_34346

noncomputable def rate_A := 1 / 2
noncomputable def rate_C := 1 / 6

theorem pump_B_rate :
  ∃ B : ℝ, (rate_A + B - rate_C = 4 / 3) ∧ (B = 1) := by
  sorry

end NUMINAMATH_GPT_pump_B_rate_l343_34346


namespace NUMINAMATH_GPT_length_of_bridge_l343_34355

theorem length_of_bridge (length_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (total_distance : ℝ) (bridge_length : ℝ) :
  length_train = 160 →
  speed_kmh = 45 →
  time_sec = 30 →
  speed_ms = 45 * (1000 / 3600) →
  total_distance = speed_ms * time_sec →
  bridge_length = total_distance - length_train →
  bridge_length = 215 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_length_of_bridge_l343_34355


namespace NUMINAMATH_GPT_degrees_for_cherry_pie_l343_34336

theorem degrees_for_cherry_pie
  (n c a b : ℕ)
  (hc : c = 15)
  (ha : a = 10)
  (hb : b = 9)
  (hn : n = 48)
  (half_remaining_cherry : (n - (c + a + b)) / 2 = 7) :
  (7 / 48 : ℚ) * 360 = 52.5 := 
by sorry

end NUMINAMATH_GPT_degrees_for_cherry_pie_l343_34336


namespace NUMINAMATH_GPT_simplify_expression_l343_34374

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l343_34374


namespace NUMINAMATH_GPT_original_plan_was_to_produce_125_sets_per_day_l343_34302

-- We state our conditions
def plans_to_complete_in_days : ℕ := 30
def produces_sets_per_day : ℕ := 150
def finishes_days_ahead_of_schedule : ℕ := 5

-- Calculations based on conditions
def actual_days_used : ℕ := plans_to_complete_in_days - finishes_days_ahead_of_schedule
def total_production : ℕ := produces_sets_per_day * actual_days_used
def original_planned_production_per_day : ℕ := total_production / plans_to_complete_in_days

-- Claim we want to prove
theorem original_plan_was_to_produce_125_sets_per_day :
  original_planned_production_per_day = 125 :=
by
  sorry

end NUMINAMATH_GPT_original_plan_was_to_produce_125_sets_per_day_l343_34302


namespace NUMINAMATH_GPT_largest_4digit_div_by_35_l343_34356

theorem largest_4digit_div_by_35 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (35 ∣ n) ∧ (∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (35 ∣ m) → m ≤ n) ∧ n = 9985 :=
by
  sorry

end NUMINAMATH_GPT_largest_4digit_div_by_35_l343_34356


namespace NUMINAMATH_GPT_sequence_bound_l343_34321

theorem sequence_bound (n : ℕ) (a : ℝ) (a_seq : ℕ → ℝ) 
  (h1 : a_seq 1 = a) 
  (h2 : a_seq n = a) 
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k < n - 1 → a_seq (k + 1) ≤ (a_seq k + a_seq (k + 2)) / 2) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a_seq k ≤ a := 
by
  sorry

end NUMINAMATH_GPT_sequence_bound_l343_34321


namespace NUMINAMATH_GPT_correct_equation_l343_34316

theorem correct_equation (x : ℝ) : 3 * x + 20 = 4 * x - 25 :=
by sorry

end NUMINAMATH_GPT_correct_equation_l343_34316


namespace NUMINAMATH_GPT_logician1_max_gain_l343_34306

noncomputable def maxCoinsDistribution (logician1 logician2 logician3 : ℕ) := (logician1, logician2, logician3)

theorem logician1_max_gain 
  (total_coins : ℕ) 
  (coins1 coins2 coins3 : ℕ) 
  (H : total_coins = 10)
  (H1 : ¬ (coins1 = 9 ∧ coins2 = 0 ∧ coins3 = 1) → coins1 = 2):
  maxCoinsDistribution coins1 coins2 coins3 = (9, 0, 1) :=
by
  sorry

end NUMINAMATH_GPT_logician1_max_gain_l343_34306


namespace NUMINAMATH_GPT_walking_speed_l343_34394

theorem walking_speed (x : ℝ) (h1 : 20 / x = 40 / (x + 5)) : x + 5 = 10 :=
  by
  sorry

end NUMINAMATH_GPT_walking_speed_l343_34394


namespace NUMINAMATH_GPT_problem1_problem2_l343_34339

open Classical

theorem problem1 (x : ℝ) : -x^2 + 4 * x - 4 < 0 ↔ x ≠ 2 :=
sorry

theorem problem2 (x : ℝ) : (1 - x) / (x - 5) > 0 ↔ 1 < x ∧ x < 5 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l343_34339


namespace NUMINAMATH_GPT_cube_tetrahedron_volume_ratio_l343_34367

theorem cube_tetrahedron_volume_ratio :
  let s := 2
  let v1 := (0, 0, 0)
  let v2 := (2, 2, 0)
  let v3 := (2, 0, 2)
  let v4 := (0, 2, 2)
  let a := Real.sqrt 8 -- Side length of the tetrahedron
  let volume_tetra := (a^3 * Real.sqrt 2) / 12
  let volume_cube := s^3
  volume_cube / volume_tetra = 6 * Real.sqrt 2 := 
by
  -- Proof content skipped
  intros
  sorry

end NUMINAMATH_GPT_cube_tetrahedron_volume_ratio_l343_34367


namespace NUMINAMATH_GPT_largest_number_in_sequence_is_48_l343_34312

theorem largest_number_in_sequence_is_48 
    (a_1 a_2 a_3 a_4 a_5 a_6 : ℕ) 
    (h1 : 0 < a_1) 
    (h2 : a_1 < a_2 ∧ a_2 < a_3 ∧ a_3 < a_4 ∧ a_4 < a_5 ∧ a_5 < a_6)
    (h3 : ∃ k_1 k_2 k_3 k_4 k_5 : ℕ, k_1 > 1 ∧ k_2 > 1 ∧ k_3 > 1 ∧ k_4 > 1 ∧ k_5 > 1 ∧ 
          a_2 = k_1 * a_1 ∧ a_3 = k_2 * a_2 ∧ a_4 = k_3 * a_3 ∧ a_5 = k_4 * a_4 ∧ a_6 = k_5 * a_5)
    (h4 : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 79) 
    : a_6 = 48 := 
by 
    sorry

end NUMINAMATH_GPT_largest_number_in_sequence_is_48_l343_34312


namespace NUMINAMATH_GPT_markup_constant_relationship_l343_34345

variable (C S : ℝ) (k : ℝ)
variable (fractional_markup : k * S = 0.25 * C)
variable (relation : S = C + k * S)

theorem markup_constant_relationship (fractional_markup : k * S = 0.25 * C) (relation : S = C + k * S) :
  k = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_markup_constant_relationship_l343_34345


namespace NUMINAMATH_GPT_determine_digits_l343_34311

def digit (n : Nat) : Prop := n < 10

theorem determine_digits :
  ∃ (A B C D : Nat), digit A ∧ digit B ∧ digit C ∧ digit D ∧
    (1000 * A + 100 * B + 10 * B + B) ^ 2 = 10000 * A + 1000 * C + 100 * D + 10 * B + B ∧
    (1000 * C + 100 * D + 10 * D + D) ^ 3 = 10000 * A + 1000 * C + 100 * B + 10 * D + D ∧
    A = 9 ∧ B = 6 ∧ C = 2 ∧ D = 1 := 
by
  sorry

end NUMINAMATH_GPT_determine_digits_l343_34311


namespace NUMINAMATH_GPT_rebus_solution_l343_34357

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end NUMINAMATH_GPT_rebus_solution_l343_34357


namespace NUMINAMATH_GPT_compute_expression_l343_34332

-- Lean 4 statement for the mathematic equivalence proof problem
theorem compute_expression:
  (1004^2 - 996^2 - 1002^2 + 998^2) = 8000 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l343_34332


namespace NUMINAMATH_GPT_exponent_problem_l343_34313

theorem exponent_problem : (-1 : ℝ)^2003 / (-1 : ℝ)^2004 = -1 := by
  sorry

end NUMINAMATH_GPT_exponent_problem_l343_34313


namespace NUMINAMATH_GPT_cos_105_sub_alpha_l343_34334

variable (α : ℝ)

-- Condition
def condition : Prop := Real.cos (75 * Real.pi / 180 + α) = 1 / 2

-- Statement
theorem cos_105_sub_alpha (h : condition α) : Real.cos (105 * Real.pi / 180 - α) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_105_sub_alpha_l343_34334


namespace NUMINAMATH_GPT_area_of_square_containing_circle_l343_34387

theorem area_of_square_containing_circle (r : ℝ) (hr : r = 4) :
  ∃ (a : ℝ), a = 64 ∧ (∀ (s : ℝ), s = 2 * r → a = s * s) :=
by
  use 64
  sorry

end NUMINAMATH_GPT_area_of_square_containing_circle_l343_34387


namespace NUMINAMATH_GPT_find_f_neg_one_l343_34351

theorem find_f_neg_one (f h : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x)
    (h2 : ∀ x, h x = f x - 9) (h3 : h 1 = 2) : f (-1) = -11 := 
by
  sorry

end NUMINAMATH_GPT_find_f_neg_one_l343_34351


namespace NUMINAMATH_GPT_room_length_l343_34343

/-- Define the conditions -/
def width : ℝ := 3.75
def cost_paving : ℝ := 6187.5
def cost_per_sqm : ℝ := 300

/-- Prove that the length of the room is 5.5 meters -/
theorem room_length : 
  (cost_paving / cost_per_sqm) / width = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_room_length_l343_34343


namespace NUMINAMATH_GPT_cycling_distance_l343_34335

-- Define the conditions
def cycling_time : ℕ := 40  -- Total cycling time in minutes
def time_per_interval : ℕ := 10  -- Time per interval in minutes
def distance_per_interval : ℕ := 2  -- Distance per interval in miles

-- Proof statement
theorem cycling_distance : (cycling_time / time_per_interval) * distance_per_interval = 8 := by
  sorry

end NUMINAMATH_GPT_cycling_distance_l343_34335


namespace NUMINAMATH_GPT_part1_solution_set_part2_minimum_value_l343_34341

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_minimum_value_l343_34341


namespace NUMINAMATH_GPT_largest_sum_of_base8_digits_l343_34308

theorem largest_sum_of_base8_digits (a b c y : ℕ) (h1 : a < 8) (h2 : b < 8) (h3 : c < 8) (h4 : 0 < y ∧ y ≤ 16) (h5 : (a * 64 + b * 8 + c) * y = 512) :
  a + b + c ≤ 5 :=
sorry

end NUMINAMATH_GPT_largest_sum_of_base8_digits_l343_34308


namespace NUMINAMATH_GPT_alice_meets_john_time_l343_34383

-- Definitions according to conditions
def john_speed : ℝ := 4
def bob_speed : ℝ := 6
def alice_speed : ℝ := 3
def initial_distance_alice_john : ℝ := 2

-- Prove the required meeting time
theorem alice_meets_john_time : 2 / (john_speed + alice_speed) * 60 = 17 := 
by
  sorry

end NUMINAMATH_GPT_alice_meets_john_time_l343_34383


namespace NUMINAMATH_GPT_width_of_channel_at_bottom_l343_34301

theorem width_of_channel_at_bottom
    (top_width : ℝ)
    (area : ℝ)
    (depth : ℝ)
    (b : ℝ)
    (H1 : top_width = 12)
    (H2 : area = 630)
    (H3 : depth = 70)
    (H4 : area = 0.5 * (top_width + b) * depth) :
    b = 6 := 
sorry

end NUMINAMATH_GPT_width_of_channel_at_bottom_l343_34301


namespace NUMINAMATH_GPT_intersection_set_l343_34362

def M : Set ℤ := {1, 2, 3, 5, 7}
def N : Set ℤ := {x | ∃ k ∈ M, x = 2 * k - 1}
def I : Set ℤ := {1, 3, 5}

theorem intersection_set :
  M ∩ N = I :=
by sorry

end NUMINAMATH_GPT_intersection_set_l343_34362


namespace NUMINAMATH_GPT_rainfall_in_2011_l343_34329

-- Define the parameters
def avg_rainfall_2010 : ℝ := 37.2
def increase_from_2010_to_2011 : ℝ := 1.8
def months_in_a_year : ℕ := 12

-- Define the total rainfall in 2011
def total_rainfall_2011 : ℝ := 468

-- Prove that the total rainfall in Driptown in 2011 is 468 mm
theorem rainfall_in_2011 :
  avg_rainfall_2010 + increase_from_2010_to_2011 = 39.0 → 
  12 * (avg_rainfall_2010 + increase_from_2010_to_2011) = total_rainfall_2011 :=
by sorry

end NUMINAMATH_GPT_rainfall_in_2011_l343_34329


namespace NUMINAMATH_GPT_letter_ratio_l343_34354

theorem letter_ratio (G B M : ℕ) (h1 : G = B + 10) 
                     (h2 : B = 40) 
                     (h3 : G + B + M = 270) : 
                     M / (G + B) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_letter_ratio_l343_34354


namespace NUMINAMATH_GPT_shanna_initial_tomato_plants_l343_34399

theorem shanna_initial_tomato_plants (T : ℕ) 
  (h1 : 56 = (T / 2) * 7 + 2 * 7 + 3 * 7) : 
  T = 6 :=
by sorry

end NUMINAMATH_GPT_shanna_initial_tomato_plants_l343_34399


namespace NUMINAMATH_GPT_gcd_of_18_and_30_l343_34373

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_18_and_30_l343_34373


namespace NUMINAMATH_GPT_part1_l343_34392

theorem part1 (a : ℤ) (h : a = -2) : 
  ((a^2 + a) / (a^2 - 3 * a)) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2 / 3 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_part1_l343_34392


namespace NUMINAMATH_GPT_largest_n_binary_operation_l343_34376

-- Define the binary operation @
def binary_operation (n : ℤ) : ℤ := n - (n * 5)

-- Define the theorem stating the desired property
theorem largest_n_binary_operation (x : ℤ) (h : x > -8) :
  ∃ (n : ℤ), n = 2 ∧ binary_operation n < x :=
sorry

end NUMINAMATH_GPT_largest_n_binary_operation_l343_34376


namespace NUMINAMATH_GPT_largest_multiple_of_three_l343_34359

theorem largest_multiple_of_three (n : ℕ) (h : 3 * n + (3 * n + 3) + (3 * n + 6) = 117) : 3 * n + 6 = 42 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_three_l343_34359


namespace NUMINAMATH_GPT_fraction_equivalent_to_decimal_l343_34386

theorem fraction_equivalent_to_decimal : 
  ∃ (x : ℚ), x = 0.6 + 0.0037 * (1 / (1 - 0.01)) ∧ x = 631 / 990 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equivalent_to_decimal_l343_34386


namespace NUMINAMATH_GPT_project_contribution_l343_34398

theorem project_contribution (total_cost : ℝ) (num_participants : ℝ) (expected_contribution : ℝ) 
  (h1 : total_cost = 25 * 10^9) 
  (h2 : num_participants = 300 * 10^6) 
  (h3 : expected_contribution = 83) : 
  total_cost / num_participants = expected_contribution := 
by 
  sorry

end NUMINAMATH_GPT_project_contribution_l343_34398


namespace NUMINAMATH_GPT_stratified_sampling_l343_34327

theorem stratified_sampling 
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (H_male_students : male_students = 40)
  (H_female_students : female_students = 30)
  (H_sample_size : sample_size = 7)
  (H_stratified_sample : sample_size = male_students_drawn + female_students_drawn) :
  male_students_drawn = 4 ∧ female_students_drawn = 3  :=
sorry

end NUMINAMATH_GPT_stratified_sampling_l343_34327


namespace NUMINAMATH_GPT_sum_of_roots_eq_three_l343_34395

-- Definitions of the polynomials
def poly1 (x : ℝ) : ℝ := 3 * x^3 + 3 * x^2 - 9 * x + 27
def poly2 (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + 5

-- Theorem stating the sum of the roots of the given equation is 3
theorem sum_of_roots_eq_three : 
  (∀ a b c d e f g h i : ℝ, 
    (poly1 a = 0) → (poly1 b = 0) → (poly1 c = 0) → 
    (poly2 d = 0) → (poly2 e = 0) → (poly2 f = 0) →
    a + b + c + d + e + f = 3) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_three_l343_34395


namespace NUMINAMATH_GPT_fourth_derivative_at_0_l343_34391

noncomputable def f : ℝ → ℝ := sorry

axiom f_at_0 : f 0 = 1
axiom f_prime_at_0 : deriv f 0 = 2
axiom f_double_prime : ∀ t, deriv (deriv f) t = 4 * deriv f t - 3 * f t + 1

-- We want to prove that the fourth derivative of f at 0 equals 54
theorem fourth_derivative_at_0 : deriv (deriv (deriv (deriv f))) 0 = 54 :=
sorry

end NUMINAMATH_GPT_fourth_derivative_at_0_l343_34391


namespace NUMINAMATH_GPT_calculate_expression_l343_34380

theorem calculate_expression : ∀ x y : ℝ, x = 7 → y = 3 → (x - y) ^ 2 * (x + y) = 160 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_calculate_expression_l343_34380


namespace NUMINAMATH_GPT_fixed_point_l343_34365

variable (p : ℝ)

def f (x : ℝ) : ℝ := 9 * x^2 + p * x - 5 * p

theorem fixed_point : ∀ c d : ℝ, (∀ p : ℝ, f p c = d) → (c = 5 ∧ d = 225) :=
by
  intro c d h
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_fixed_point_l343_34365


namespace NUMINAMATH_GPT_transformed_curve_l343_34333

theorem transformed_curve (x y : ℝ) :
  (∃ (x1 y1 : ℝ), x1 = 3*x ∧ y1 = 2*y ∧ (x1^2 / 9 + y1^2 / 4 = 1)) →
  x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_transformed_curve_l343_34333


namespace NUMINAMATH_GPT_repair_cost_l343_34324

theorem repair_cost
  (R : ℝ) -- R is the cost to repair the used shoes
  (new_shoes_cost : ℝ := 30) -- New shoes cost $30.00
  (new_shoes_lifetime : ℝ := 2) -- New shoes last for 2 years
  (percentage_increase : ℝ := 42.857142857142854) 
  (h1 : new_shoes_cost / new_shoes_lifetime = R + (percentage_increase / 100) * R) :
  R = 10.50 :=
by
  sorry

end NUMINAMATH_GPT_repair_cost_l343_34324


namespace NUMINAMATH_GPT_arithmetic_seq_common_diff_l343_34323

theorem arithmetic_seq_common_diff
  (a₃ a₇ S₁₀ : ℤ)
  (h₁ : a₃ + a₇ = 16)
  (h₂ : S₁₀ = 85)
  (a₃_eq : ∃ a₁ d : ℤ, a₃ = a₁ + 2 * d)
  (a₇_eq : ∃ a₁ d : ℤ, a₇ = a₁ + 6 * d)
  (S₁₀_eq : ∃ a₁ d : ℤ, S₁₀ = 10 * a₁ + 45 * d) :
  ∃ d : ℤ, d = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_common_diff_l343_34323


namespace NUMINAMATH_GPT_burger_cost_l343_34319

theorem burger_cost :
  ∃ b s f : ℕ, 4 * b + 2 * s + 3 * f = 480 ∧ 3 * b + s + 2 * f = 360 ∧ b = 80 :=
by
  sorry

end NUMINAMATH_GPT_burger_cost_l343_34319


namespace NUMINAMATH_GPT_base_five_to_ten_3214_l343_34350

theorem base_five_to_ten_3214 : (3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 4 * 5^0) = 434 := by
  sorry

end NUMINAMATH_GPT_base_five_to_ten_3214_l343_34350


namespace NUMINAMATH_GPT_real_roots_of_quadratic_l343_34384

theorem real_roots_of_quadratic (m : ℝ) : ((m - 2) ≠ 0 ∧ (-4 * m + 24) ≥ 0) → (m ≤ 6 ∧ m ≠ 2) := 
by 
  sorry

end NUMINAMATH_GPT_real_roots_of_quadratic_l343_34384


namespace NUMINAMATH_GPT_range_of_a_l343_34309

theorem range_of_a (a : ℝ) (x : ℝ) :
  ((a < x ∧ x < a + 2) → x > 3) ∧ ¬(∀ x, (x > 3) → (a < x ∧ x < a + 2)) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l343_34309


namespace NUMINAMATH_GPT_team_total_points_l343_34352

-- Definition of Wade's average points per game
def wade_avg_points_per_game := 20

-- Definition of teammates' average points per game
def teammates_avg_points_per_game := 40

-- Definition of the number of games
def number_of_games := 5

-- The total points calculation problem
theorem team_total_points 
  (Wade_avg : wade_avg_points_per_game = 20)
  (Teammates_avg : teammates_avg_points_per_game = 40)
  (Games : number_of_games = 5) :
  5 * wade_avg_points_per_game + 5 * teammates_avg_points_per_game = 300 := 
by 
  -- The proof is omitted and marked as sorry
  sorry

end NUMINAMATH_GPT_team_total_points_l343_34352


namespace NUMINAMATH_GPT_expressions_inequivalence_l343_34322

theorem expressions_inequivalence (x : ℝ) (h : x > 0) :
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (x + 1) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (x + 1) ^ (2 * x + 2)) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (0.5 * x + x) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (2 * x + 2) ^ (2 * x + 2)) := by
  sorry

end NUMINAMATH_GPT_expressions_inequivalence_l343_34322


namespace NUMINAMATH_GPT_mean_score_all_students_l343_34353

theorem mean_score_all_students
  (M A E : ℝ) (m a e : ℝ)
  (hM : M = 78)
  (hA : A = 68)
  (hE : E = 82)
  (h_ratio_ma : m / a = 4 / 5)
  (h_ratio_mae : (m + a) / e = 9 / 2)
  : (M * m + A * a + E * e) / (m + a + e) = 74.4 := by
  sorry

end NUMINAMATH_GPT_mean_score_all_students_l343_34353


namespace NUMINAMATH_GPT_trapezoid_area_l343_34328

-- Definitions based on conditions
def CL_div_LD (CL LD : ℝ) : Prop := CL / LD = 1 / 4

-- The main statement we want to prove
theorem trapezoid_area (BC CD : ℝ) (h1 : BC = 9) (h2 : CD = 30) (CL LD : ℝ) (h3 : CL_div_LD CL LD) : 
  1/2 * (BC + AD) * 24 = 972 :=
sorry

end NUMINAMATH_GPT_trapezoid_area_l343_34328


namespace NUMINAMATH_GPT_find_original_cost_of_chips_l343_34314

def original_cost_chips (discount amount_spent : ℝ) : ℝ :=
  discount + amount_spent

theorem find_original_cost_of_chips :
  original_cost_chips 17 18 = 35 := by
  sorry

end NUMINAMATH_GPT_find_original_cost_of_chips_l343_34314


namespace NUMINAMATH_GPT_liquid_X_percentage_correct_l343_34325

noncomputable def percent_liquid_X_in_solution_A := 0.8 / 100
noncomputable def percent_liquid_X_in_solution_B := 1.8 / 100

noncomputable def weight_solution_A := 400.0
noncomputable def weight_solution_B := 700.0

noncomputable def weight_liquid_X_in_A := percent_liquid_X_in_solution_A * weight_solution_A
noncomputable def weight_liquid_X_in_B := percent_liquid_X_in_solution_B * weight_solution_B

noncomputable def total_weight_solution := weight_solution_A + weight_solution_B
noncomputable def total_weight_liquid_X := weight_liquid_X_in_A + weight_liquid_X_in_B

noncomputable def percent_liquid_X_in_mixed_solution := (total_weight_liquid_X / total_weight_solution) * 100

theorem liquid_X_percentage_correct :
  percent_liquid_X_in_mixed_solution = 1.44 :=
by
  sorry

end NUMINAMATH_GPT_liquid_X_percentage_correct_l343_34325


namespace NUMINAMATH_GPT_max_rectangles_in_triangle_l343_34397

theorem max_rectangles_in_triangle : 
  (∃ (n : ℕ), n = 192 ∧ 
  ∀ (i j : ℕ), i + j < 7 → ∀ (a b : ℕ), a ≤ 6 - i ∧ b ≤ 6 - j → 
  ∃ (rectangles : ℕ), rectangles = (6 - i) * (6 - j)) :=
sorry

end NUMINAMATH_GPT_max_rectangles_in_triangle_l343_34397


namespace NUMINAMATH_GPT_abc_eq_zero_l343_34338

variable (a b c : ℝ) (n : ℕ)

theorem abc_eq_zero
  (h1 : a^n + b^n = c^n)
  (h2 : a^(n+1) + b^(n+1) = c^(n+1))
  (h3 : a^(n+2) + b^(n+2) = c^(n+2)) :
  a * b * c = 0 :=
sorry

end NUMINAMATH_GPT_abc_eq_zero_l343_34338


namespace NUMINAMATH_GPT_nina_money_proof_l343_34317

def total_money_nina_has (W M : ℝ) : Prop :=
  (10 * W = M) ∧ (14 * (W - 1.75) = M)

theorem nina_money_proof (W M : ℝ) (h : total_money_nina_has W M) : M = 61.25 :=
by 
  sorry

end NUMINAMATH_GPT_nina_money_proof_l343_34317


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l343_34375

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → false := 
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l343_34375


namespace NUMINAMATH_GPT_sub_three_five_l343_34368

theorem sub_three_five : 3 - 5 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_sub_three_five_l343_34368


namespace NUMINAMATH_GPT_pencils_left_l343_34364

theorem pencils_left (initial_pencils : ℕ := 79) (pencils_taken : ℕ := 4) : initial_pencils - pencils_taken = 75 :=
by
  sorry

end NUMINAMATH_GPT_pencils_left_l343_34364


namespace NUMINAMATH_GPT_smallest_number_diminished_by_35_l343_34340

def lcm_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

def conditions : List ℕ := [5, 10, 15, 20, 25, 30, 35]

def lcm_conditions := lcm_list conditions

theorem smallest_number_diminished_by_35 :
  ∃ n, n - 35 = lcm_conditions :=
sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_35_l343_34340


namespace NUMINAMATH_GPT_sugar_price_difference_l343_34388

theorem sugar_price_difference (a b : ℝ) (h : (3 / 5 * a + 2 / 5 * b) - (2 / 5 * a + 3 / 5 * b) = 1.32) :
  a - b = 6.6 :=
by
  sorry

end NUMINAMATH_GPT_sugar_price_difference_l343_34388


namespace NUMINAMATH_GPT_transformed_polynomial_roots_l343_34310

theorem transformed_polynomial_roots (a b c d : ℝ) 
  (h1 : a + b + c + d = 0)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h3 : a * b * c * d ≠ 0)
  (h4 : Polynomial.eval a (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h5 : Polynomial.eval b (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h6 : Polynomial.eval c (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h7 : Polynomial.eval d (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0):
  Polynomial.eval (-2 / d^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / c^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / b^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / a^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 :=
sorry

end NUMINAMATH_GPT_transformed_polynomial_roots_l343_34310


namespace NUMINAMATH_GPT_tangent_line_find_a_l343_34385

theorem tangent_line_find_a (a : ℝ) (f : ℝ → ℝ) (tangent : ℝ → ℝ) (x₀ : ℝ)
  (hf : ∀ x, f x = x + 1/x - a * Real.log x)
  (h_tangent : ∀ x, tangent x = x + 1)
  (h_deriv : deriv f x₀ = deriv tangent x₀)
  (h_eq : f x₀ = tangent x₀) :
  a = -1 :=
sorry

end NUMINAMATH_GPT_tangent_line_find_a_l343_34385


namespace NUMINAMATH_GPT_length_PC_in_rectangle_l343_34389

theorem length_PC_in_rectangle (PA PB PD: ℝ) (P_inside: True) 
(h1: PA = 5) (h2: PB = 7) (h3: PD = 3) : PC = Real.sqrt 65 := 
sorry

end NUMINAMATH_GPT_length_PC_in_rectangle_l343_34389


namespace NUMINAMATH_GPT_statement_I_l343_34377

section Problem
variable (g : ℝ → ℝ)

-- Conditions
def cond1 : Prop := ∀ x : ℝ, g x > 0
def cond2 : Prop := ∀ a b : ℝ, g a * g b = g (a + 2 * b)

-- Statement I to be proved
theorem statement_I (h1 : cond1 g) (h2 : cond2 g) : g 0 = 1 :=
by
  -- Proof is omitted
  sorry
end Problem

end NUMINAMATH_GPT_statement_I_l343_34377


namespace NUMINAMATH_GPT_sum_of_coefficients_l343_34361

theorem sum_of_coefficients (a b : ℝ)
  (h1 : 15 * a^4 * b^2 = 135)
  (h2 : 6 * a^5 * b = -18) :
  (a + b)^6 = 64 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l343_34361


namespace NUMINAMATH_GPT_false_statement_about_circles_l343_34379

variable (P Q : Type) [MetricSpace P] [MetricSpace Q]
variable (p q : ℝ)
variable (dist_PQ : ℝ)

theorem false_statement_about_circles 
  (hA : p - q = dist_PQ → false)
  (hB : p + q = dist_PQ → false)
  (hC : p + q < dist_PQ → false)
  (hD : p - q < dist_PQ → false) : 
  false :=
by sorry

end NUMINAMATH_GPT_false_statement_about_circles_l343_34379


namespace NUMINAMATH_GPT_total_puppies_count_l343_34363

theorem total_puppies_count (total_cost sale_cost others_cost: ℕ) 
  (three_puppies_on_sale: ℕ) 
  (one_sale_puppy_cost: ℕ)
  (one_other_puppy_cost: ℕ)
  (h1: total_cost = 800)
  (h2: three_puppies_on_sale = 3)
  (h3: one_sale_puppy_cost = 150)
  (h4: others_cost = total_cost - three_puppies_on_sale * one_sale_puppy_cost)
  (h5: one_other_puppy_cost = 175)
  (h6: ∃ other_puppies : ℕ, other_puppies = others_cost / one_other_puppy_cost) :
  ∃ total_puppies : ℕ,
  total_puppies = three_puppies_on_sale + (others_cost / one_other_puppy_cost) := 
sorry

end NUMINAMATH_GPT_total_puppies_count_l343_34363


namespace NUMINAMATH_GPT_intersection_P_Q_l343_34330

def P : Set ℝ := {x : ℝ | x < 1}
def Q : Set ℝ := {x : ℝ | x^2 < 4}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := 
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l343_34330


namespace NUMINAMATH_GPT_log_579_between_consec_ints_l343_34304

theorem log_579_between_consec_ints (a b : ℤ) (h₁ : 2 < Real.log 579 / Real.log 10) (h₂ : Real.log 579 / Real.log 10 < 3) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_log_579_between_consec_ints_l343_34304


namespace NUMINAMATH_GPT_big_joe_height_is_8_l343_34337

variable (Pepe_height Frank_height Larry_height Ben_height BigJoe_height : ℝ)

axiom Pepe_height_def : Pepe_height = 4.5
axiom Frank_height_def : Frank_height = Pepe_height + 0.5
axiom Larry_height_def : Larry_height = Frank_height + 1
axiom Ben_height_def : Ben_height = Larry_height + 1
axiom BigJoe_height_def : BigJoe_height = Ben_height + 1

theorem big_joe_height_is_8 :
  BigJoe_height = 8 :=
sorry

end NUMINAMATH_GPT_big_joe_height_is_8_l343_34337


namespace NUMINAMATH_GPT_decimal_division_l343_34370

theorem decimal_division : (0.05 : ℝ) / (0.005 : ℝ) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_decimal_division_l343_34370


namespace NUMINAMATH_GPT_domain_f_2x_l343_34378

-- Given conditions as definitions
def domain_f_x_minus_1 (x : ℝ) := 3 < x ∧ x ≤ 7

-- The main theorem statement that needs a proof
theorem domain_f_2x : (∀ x : ℝ, domain_f_x_minus_1 (x-1) → (1 < x ∧ x ≤ 3)) :=
by
  -- Proof steps will be here, however, as requested, they are omitted.
  sorry

end NUMINAMATH_GPT_domain_f_2x_l343_34378


namespace NUMINAMATH_GPT_range_of_b_l343_34390

theorem range_of_b (a b : ℝ) (h1 : a ≠ 0) (h2 : a * b^2 > a) (h3 : a > a * b) : b < -1 :=
sorry

end NUMINAMATH_GPT_range_of_b_l343_34390


namespace NUMINAMATH_GPT_correct_statements_count_l343_34371

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end NUMINAMATH_GPT_correct_statements_count_l343_34371


namespace NUMINAMATH_GPT_percent_decrease_in_hours_l343_34366

theorem percent_decrease_in_hours (W H : ℝ) 
  (h1 : W > 0) 
  (h2 : H > 0)
  (new_wage : ℝ := W * 1.25)
  (H_new : ℝ := H / 1.25)
  (total_income_same : W * H = new_wage * H_new) :
  ((H - H_new) / H) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percent_decrease_in_hours_l343_34366


namespace NUMINAMATH_GPT_volume_tetrahedron_formula_l343_34382

-- Definitions of the problem elements
def distance (A B C D : Point) : ℝ := sorry
def angle (A B C D : Point) : ℝ := sorry
def length (A B : Point) : ℝ := sorry

-- The problem states you need to prove the volume of the tetrahedron
noncomputable def volume_tetrahedron (A B C D : Point) : ℝ := sorry

-- Conditions
variable (A B C D : Point)
variable (d : ℝ) (phi : ℝ) -- d = distance between lines AB and CD, phi = angle between lines AB and CD

-- Question reformulated as a proof statement
theorem volume_tetrahedron_formula (h1 : d = distance A B C D)
                                   (h2 : phi = angle A B C D) :
  volume_tetrahedron A B C D = (d * length A B * length C D * Real.sin phi) / 6 :=
sorry

end NUMINAMATH_GPT_volume_tetrahedron_formula_l343_34382


namespace NUMINAMATH_GPT_find_multiple_of_hats_l343_34349

/-
   Given:
   - Fire chief Simpson has 15 hats.
   - Policeman O'Brien now has 34 hats.
   - Before he lost one, Policeman O'Brien had 5 more hats than a certain multiple of Fire chief Simpson's hats.
   Prove:
   The multiple of Fire chief Simpson's hats that Policeman O'Brien had before he lost one is 2.
-/

theorem find_multiple_of_hats :
  ∃ x : ℕ, 34 + 1 = 5 + 15 * x ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_hats_l343_34349


namespace NUMINAMATH_GPT_gas_pressure_inversely_proportional_l343_34315

theorem gas_pressure_inversely_proportional
  (p v k : ℝ)
  (v_i v_f : ℝ)
  (p_i p_f : ℝ)
  (h1 : v_i = 3.5)
  (h2 : p_i = 8)
  (h3 : v_f = 7)
  (h4 : p * v = k)
  (h5 : p_i * v_i = k)
  (h6 : p_f * v_f = k) : p_f = 4 := by
  sorry

end NUMINAMATH_GPT_gas_pressure_inversely_proportional_l343_34315
