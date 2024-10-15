import Mathlib

namespace NUMINAMATH_GPT_largest_constant_inequality_l2404_240483

theorem largest_constant_inequality (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) :=
sorry

end NUMINAMATH_GPT_largest_constant_inequality_l2404_240483


namespace NUMINAMATH_GPT_gcd_104_156_l2404_240431

theorem gcd_104_156 : Nat.gcd 104 156 = 52 :=
by
  -- the proof steps will go here, but we can use sorry to skip it
  sorry

end NUMINAMATH_GPT_gcd_104_156_l2404_240431


namespace NUMINAMATH_GPT_cost_per_pound_mixed_feed_correct_l2404_240465

noncomputable def total_weight_of_feed : ℝ := 17
noncomputable def cost_per_pound_cheaper_feed : ℝ := 0.11
noncomputable def cost_per_pound_expensive_feed : ℝ := 0.50
noncomputable def weight_cheaper_feed : ℝ := 12.2051282051

noncomputable def total_cost_of_feed : ℝ :=
  (cost_per_pound_cheaper_feed * weight_cheaper_feed) + 
  (cost_per_pound_expensive_feed * (total_weight_of_feed - weight_cheaper_feed))

noncomputable def cost_per_pound_mixed_feed : ℝ :=
  total_cost_of_feed / total_weight_of_feed

theorem cost_per_pound_mixed_feed_correct : 
  cost_per_pound_mixed_feed = 0.22 :=
  by
    sorry

end NUMINAMATH_GPT_cost_per_pound_mixed_feed_correct_l2404_240465


namespace NUMINAMATH_GPT_product_positions_8_2_100_100_l2404_240401

def num_at_position : ℕ → ℕ → ℤ
| 0, _ => 0
| n, k => 
  let remainder := k % 3
  if remainder = 1 then 1 
  else if remainder = 2 then 2
  else -3

theorem product_positions_8_2_100_100 : 
  num_at_position 8 2 * num_at_position 100 100 = -3 :=
by
  unfold num_at_position
  -- unfold necessary definition steps
  sorry

end NUMINAMATH_GPT_product_positions_8_2_100_100_l2404_240401


namespace NUMINAMATH_GPT_tram_speed_l2404_240420

/-- 
Given:
1. The pedestrian's speed is 1 km per 10 minutes, which converts to 6 km/h.
2. The speed of the trams is V km/h.
3. The relative speed of oncoming trams is V + 6 km/h.
4. The relative speed of overtaking trams is V - 6 km/h.
5. The ratio of the number of oncoming trams to overtaking trams is 700/300.
Prove:
The speed of the trams V is 15 km/h.
-/
theorem tram_speed (V : ℝ) (h1 : (V + 6) / (V - 6) = 700 / 300) : V = 15 :=
by
  sorry

end NUMINAMATH_GPT_tram_speed_l2404_240420


namespace NUMINAMATH_GPT_add_base8_l2404_240438

-- Define the base 8 numbers 5_8 and 16_8
def five_base8 : ℕ := 5
def sixteen_base8 : ℕ := 1 * 8 + 6

-- Convert the result to base 8 from the sum in base 10
def sum_base8 (a b : ℕ) : ℕ :=
  let sum_base10 := a + b
  let d1 := sum_base10 / 8
  let d0 := sum_base10 % 8
  d1 * 10 + d0 

theorem add_base8 (x y : ℕ) (hx : x = five_base8) (hy : y = sixteen_base8) :
  sum_base8 x y = 23 :=
by
  sorry

end NUMINAMATH_GPT_add_base8_l2404_240438


namespace NUMINAMATH_GPT_percentage_decrease_hours_with_assistant_l2404_240407

theorem percentage_decrease_hours_with_assistant :
  ∀ (B H H_new : ℝ), H_new = 0.9 * H → (H - H_new) / H * 100 = 10 :=
by
  intros B H H_new h_new_def
  sorry

end NUMINAMATH_GPT_percentage_decrease_hours_with_assistant_l2404_240407


namespace NUMINAMATH_GPT_expand_expression_l2404_240434

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := 
by 
  sorry

end NUMINAMATH_GPT_expand_expression_l2404_240434


namespace NUMINAMATH_GPT_find_k_l2404_240402

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 50 < f a b c 7)
  (h3 : f a b c 7 < 60)
  (h4 : 70 < f a b c 8)
  (h5 : f a b c 8 < 80)
  (h6 : 5000 * k < f a b c 100)
  (h7 : f a b c 100 < 5000 * (k + 1)) :
  k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l2404_240402


namespace NUMINAMATH_GPT_subset_implies_a_ge_2_l2404_240479

theorem subset_implies_a_ge_2 (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → x ≤ a) → a ≥ 2 :=
by sorry

end NUMINAMATH_GPT_subset_implies_a_ge_2_l2404_240479


namespace NUMINAMATH_GPT_stuffed_animals_count_l2404_240418

theorem stuffed_animals_count
  (total_prizes : ℕ)
  (frisbees : ℕ)
  (yoyos : ℕ)
  (h1 : total_prizes = 50)
  (h2 : frisbees = 18)
  (h3 : yoyos = 18) :
  (total_prizes - (frisbees + yoyos) = 14) :=
by
  sorry

end NUMINAMATH_GPT_stuffed_animals_count_l2404_240418


namespace NUMINAMATH_GPT_park_area_l2404_240475

variable (length width : ℝ)
variable (cost_per_meter total_cost : ℝ)
variable (ratio_length ratio_width : ℝ)
variable (x : ℝ)

def rectangular_park_ratio (length width : ℝ) (ratio_length ratio_width : ℝ) : Prop :=
  length / width = ratio_length / ratio_width

def fencing_cost (cost_per_meter total_cost : ℝ) (perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

theorem park_area (length width : ℝ) (cost_per_meter total_cost : ℝ)
  (ratio_length ratio_width : ℝ) (x : ℝ)
  (h1 : rectangular_park_ratio length width ratio_length ratio_width)
  (h2 : cost_per_meter = 0.70)
  (h3 : total_cost = 175)
  (h4 : ratio_length = 3)
  (h5 : ratio_width = 2)
  (h6 : length = 3 * x)
  (h7 : width = 2 * x)
  (h8 : fencing_cost cost_per_meter total_cost (2 * (length + width))) :
  length * width = 3750 := by
  sorry

end NUMINAMATH_GPT_park_area_l2404_240475


namespace NUMINAMATH_GPT_general_formula_no_arithmetic_sequence_l2404_240494

-- Given condition
def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n - 3 * n

-- Theorem 1: General formula for the sequence a_n
theorem general_formula (a : ℕ → ℤ) (n : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) : 
  a n = 3 * 2^n - 3 :=
sorry

-- Theorem 2: No three terms of the sequence form an arithmetic sequence
theorem no_arithmetic_sequence (a : ℕ → ℤ) (x y z : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) (hx : x < y) (hy : y < z) :
  ¬ (a x + a z = 2 * a y) :=
sorry

end NUMINAMATH_GPT_general_formula_no_arithmetic_sequence_l2404_240494


namespace NUMINAMATH_GPT_determine_g_function_l2404_240410

theorem determine_g_function (t x : ℝ) (g : ℝ → ℝ) 
  (line_eq : ∀ x y : ℝ, y = 2 * x - 40) 
  (param_eq : ∀ t : ℝ, (x, 20 * t - 14) = (g t, 20 * t - 14)) :
  g t = 10 * t + 13 :=
by 
  sorry

end NUMINAMATH_GPT_determine_g_function_l2404_240410


namespace NUMINAMATH_GPT_isabella_hourly_rate_l2404_240446

def isabella_hours_per_day : ℕ := 5
def isabella_days_per_week : ℕ := 6
def isabella_weeks : ℕ := 7
def isabella_total_earnings : ℕ := 1050

theorem isabella_hourly_rate :
  (isabella_hours_per_day * isabella_days_per_week * isabella_weeks) * x = isabella_total_earnings → x = 5 := by
  sorry

end NUMINAMATH_GPT_isabella_hourly_rate_l2404_240446


namespace NUMINAMATH_GPT_possible_values_of_b_l2404_240467

theorem possible_values_of_b (b : ℝ) : (¬ ∃ x : ℝ, x^2 + b * x + 1 ≤ 0) → -2 < b ∧ b < 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_possible_values_of_b_l2404_240467


namespace NUMINAMATH_GPT_polynomial_real_root_l2404_240485

variable {A B C D E : ℝ}

theorem polynomial_real_root
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_real_root_l2404_240485


namespace NUMINAMATH_GPT_total_tissues_l2404_240428

-- define the number of students in each group
def g1 : Nat := 9
def g2 : Nat := 10
def g3 : Nat := 11

-- define the number of tissues per mini tissue box
def t : Nat := 40

-- state the main theorem
theorem total_tissues : (g1 + g2 + g3) * t = 1200 := by
  sorry

end NUMINAMATH_GPT_total_tissues_l2404_240428


namespace NUMINAMATH_GPT_red_peaches_per_basket_l2404_240499

theorem red_peaches_per_basket (R : ℕ) (green_peaches_per_basket : ℕ) (number_of_baskets : ℕ) (total_peaches : ℕ) (h1 : green_peaches_per_basket = 4) (h2 : number_of_baskets = 15) (h3 : total_peaches = 345) : R = 19 :=
by
  sorry

end NUMINAMATH_GPT_red_peaches_per_basket_l2404_240499


namespace NUMINAMATH_GPT_sum_of_ages_is_12_l2404_240454

-- Let Y be the age of the youngest child
def Y : ℝ := 1.5

-- Let the ages of the other children
def age2 : ℝ := Y + 1
def age3 : ℝ := Y + 2
def age4 : ℝ := Y + 3

-- Define the sum of the ages
def sum_of_ages : ℝ := Y + age2 + age3 + age4

-- The theorem to prove the sum of the ages is 12 years
theorem sum_of_ages_is_12 : sum_of_ages = 12 :=
by
  -- The detailed proof is to be filled in later, currently skipped.
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_12_l2404_240454


namespace NUMINAMATH_GPT_cos_150_eq_neg_sqrt3_div_2_l2404_240497

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  unfold Real.cos
  sorry

end NUMINAMATH_GPT_cos_150_eq_neg_sqrt3_div_2_l2404_240497


namespace NUMINAMATH_GPT_abs_neg_one_ninth_l2404_240442

theorem abs_neg_one_ninth : abs (- (1 / 9)) = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_abs_neg_one_ninth_l2404_240442


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l2404_240421

-- Define the given conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def passes_through (a : ℝ) (p : ℝ × ℝ) : Prop := p.snd = parabola a p.fst

-- Main theorem: Prove the coordinates of the focus
theorem parabola_focus_coordinates (a : ℝ) (h : passes_through a (1, 4)) (ha : a = 4) : (0, 1 / 16) = (0, 1 / (4 * a)) :=
by
  rw [ha] -- substitute the value of a
  simp -- simplify the expression
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l2404_240421


namespace NUMINAMATH_GPT_henry_age_is_20_l2404_240430

open Nat

def sum_ages (H J : ℕ) : Prop := H + J = 33
def age_relation (H J : ℕ) : Prop := H - 6 = 2 * (J - 6)

theorem henry_age_is_20 (H J : ℕ) (h1 : sum_ages H J) (h2 : age_relation H J) : H = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_henry_age_is_20_l2404_240430


namespace NUMINAMATH_GPT_min_overlap_l2404_240437

variable (P : Set ℕ → ℝ)
variable (B M : Set ℕ)

-- Conditions
def P_B_def : P B = 0.95 := sorry
def P_M_def : P M = 0.85 := sorry

-- To Prove
theorem min_overlap : P (B ∩ M) = 0.80 := sorry

end NUMINAMATH_GPT_min_overlap_l2404_240437


namespace NUMINAMATH_GPT_min_troublemakers_29_l2404_240403

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end NUMINAMATH_GPT_min_troublemakers_29_l2404_240403


namespace NUMINAMATH_GPT_hiking_time_l2404_240426

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end NUMINAMATH_GPT_hiking_time_l2404_240426


namespace NUMINAMATH_GPT_nat_power_digit_condition_l2404_240477

theorem nat_power_digit_condition (n k : ℕ) : 
  (10^(k-1) < n^n ∧ n^n < 10^k) → (10^(n-1) < k^k ∧ k^k < 10^n) → 
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end NUMINAMATH_GPT_nat_power_digit_condition_l2404_240477


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l2404_240451

theorem gcd_of_three_numbers (a b c : ℕ) (h1: a = 4557) (h2: b = 1953) (h3: c = 5115) : 
    Nat.gcd a (Nat.gcd b c) = 93 :=
by
  rw [h1, h2, h3]
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l2404_240451


namespace NUMINAMATH_GPT_find_beta_l2404_240436

variable (α β : ℝ)

theorem find_beta 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) : β = Real.pi / 3 := sorry

end NUMINAMATH_GPT_find_beta_l2404_240436


namespace NUMINAMATH_GPT_solve_for_x_l2404_240443

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l2404_240443


namespace NUMINAMATH_GPT_least_number_divisor_l2404_240450

theorem least_number_divisor (d : ℕ) (n m : ℕ) 
  (h1 : d = 1081)
  (h2 : m = 1077)
  (h3 : n = 4)
  (h4 : ∃ k, m + n = k * d) :
  d = 1081 :=
by
  sorry

end NUMINAMATH_GPT_least_number_divisor_l2404_240450


namespace NUMINAMATH_GPT_find_length_l2404_240491

-- Let's define the conditions given in the problem
variables (b l : ℝ)

-- Length is more than breadth by 200%
def length_eq_breadth_plus_200_percent (b l : ℝ) : Prop := l = 3 * b

-- Total cost and rate per square meter
def cost_eq_area_times_rate (total_cost rate area : ℝ) : Prop := total_cost = rate * area

-- Given values
def total_cost : ℝ := 529
def rate_per_sq_meter : ℝ := 3

-- We need to prove that the length l is approximately 23 meters
theorem find_length (h1 : length_eq_breadth_plus_200_percent b l) 
    (h2 : cost_eq_area_times_rate total_cost rate_per_sq_meter (3 * b^2)) : 
    abs (l - 23) < 1 :=
by
  sorry -- Proof to be filled

end NUMINAMATH_GPT_find_length_l2404_240491


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l2404_240452

theorem algebraic_expression_evaluation (a b c : ℝ) 
  (h1 : a^2 + b * c = 14) 
  (h2 : b^2 - 2 * b * c = -6) : 
  3 * a^2 + 4 * b^2 - 5 * b * c = 18 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l2404_240452


namespace NUMINAMATH_GPT_factorization_of_1386_l2404_240447

-- We start by defining the number and the requirements.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def factors_mult (a b : ℕ) : Prop := a * b = 1386
def factorization_count (count : ℕ) : Prop :=
  ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors_mult a b ∧ 
  (∀ c d, is_two_digit c ∧ is_two_digit d ∧ factors_mult c d → 
  (c = a ∧ d = b ∨ c = b ∧ d = a) → c = a ∧ d = b ∨ c = b ∧ d = a) ∧
  count = 4

-- Now, we state the theorem.
theorem factorization_of_1386 : factorization_count 4 :=
sorry

end NUMINAMATH_GPT_factorization_of_1386_l2404_240447


namespace NUMINAMATH_GPT_train_speed_l2404_240478

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (conversion_factor : ℝ) :
  length_of_train = 200 → 
  time_to_cross = 24 → 
  conversion_factor = 3600 → 
  (length_of_train / 1000) / (time_to_cross / conversion_factor) = 30 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_l2404_240478


namespace NUMINAMATH_GPT_response_rate_percentage_l2404_240400

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 240) (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := 
by 
  sorry

end NUMINAMATH_GPT_response_rate_percentage_l2404_240400


namespace NUMINAMATH_GPT_sin_sum_triangle_l2404_240472

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_sin_sum_triangle_l2404_240472


namespace NUMINAMATH_GPT_motorboat_distance_l2404_240405

variable (S v u : ℝ)
variable (V_m : ℝ := 2 * v + u)  -- Velocity of motorboat downstream
variable (V_b : ℝ := 3 * v - u)  -- Velocity of boat upstream

theorem motorboat_distance :
  ( L = (161 / 225) * S ∨ L = (176 / 225) * S) :=
by
  sorry

end NUMINAMATH_GPT_motorboat_distance_l2404_240405


namespace NUMINAMATH_GPT_angle_CAB_in_regular_hexagon_l2404_240495

-- Define a regular hexagon
structure regular_hexagon (A B C D E F : Type) :=
  (interior_angle : ℝ)
  (all_sides_equal : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = F)
  (all_angles_equal : interior_angle = 120)

-- Define the problem of finding the angle CAB
theorem angle_CAB_in_regular_hexagon 
  (A B C D E F : Type)
  (hex : regular_hexagon A B C D E F)
  (diagonal_AC : A = C)
  : ∃ (CAB : ℝ), CAB = 30 :=
sorry

end NUMINAMATH_GPT_angle_CAB_in_regular_hexagon_l2404_240495


namespace NUMINAMATH_GPT_percentage_supports_policy_l2404_240439

theorem percentage_supports_policy
    (men_support_percentage : ℝ)
    (women_support_percentage : ℝ)
    (num_men : ℕ)
    (num_women : ℕ)
    (total_surveyed : ℕ)
    (total_supporters : ℕ)
    (overall_percentage : ℝ) :
    (men_support_percentage = 0.70) →
    (women_support_percentage = 0.75) →
    (num_men = 200) →
    (num_women = 800) →
    (total_surveyed = num_men + num_women) →
    (total_supporters = (men_support_percentage * num_men) + (women_support_percentage * num_women)) →
    (overall_percentage = (total_supporters / total_surveyed) * 100) →
    overall_percentage = 74 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_supports_policy_l2404_240439


namespace NUMINAMATH_GPT_running_speed_l2404_240455

theorem running_speed (walk_speed total_distance walk_time total_time run_distance : ℝ) 
  (h_walk_speed : walk_speed = 4)
  (h_total_distance : total_distance = 4)
  (h_walk_time : walk_time = 0.5)
  (h_total_time : total_time = 0.75)
  (h_run_distance : run_distance = total_distance / 2) :
  (2 / ((total_time - walk_time) - 2 / walk_speed)) = 8 := 
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_running_speed_l2404_240455


namespace NUMINAMATH_GPT_coordinates_of_P_l2404_240492

-- Define the point P with given coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P(3, 5)
def P : Point := ⟨3, 5⟩

-- Define a theorem stating that the coordinates of P are (3, 5)
theorem coordinates_of_P : P = ⟨3, 5⟩ :=
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l2404_240492


namespace NUMINAMATH_GPT_one_fifth_of_5_times_7_l2404_240413

theorem one_fifth_of_5_times_7 : (1 / 5) * (5 * 7) = 7 := by
  sorry

end NUMINAMATH_GPT_one_fifth_of_5_times_7_l2404_240413


namespace NUMINAMATH_GPT_expression_value_l2404_240464

theorem expression_value (x y z : ℤ) (h1 : x = 25) (h2 : y = 30) (h3 : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l2404_240464


namespace NUMINAMATH_GPT_a_7_is_4_l2404_240411

-- Define the geometric sequence and its properties
variable {a : ℕ → ℝ}

-- Given conditions
axiom pos_seq : ∀ n, a n > 0
axiom geom_seq : ∀ n m, a (n + m) = a n * a m
axiom specific_condition : a 3 * a 11 = 16

theorem a_7_is_4 : a 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_a_7_is_4_l2404_240411


namespace NUMINAMATH_GPT_find_PR_in_triangle_l2404_240466

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end NUMINAMATH_GPT_find_PR_in_triangle_l2404_240466


namespace NUMINAMATH_GPT_max_area_triangle_bqc_l2404_240484

noncomputable def triangle_problem : ℝ :=
  let a := 112.5
  let b := 56.25
  let c := 3
  a + b + c

theorem max_area_triangle_bqc : triangle_problem = 171.75 :=
by
  -- The proof would involve validating the steps to ensure the computations
  -- for the maximum area of triangle BQC match the expression 112.5 - 56.25 √3,
  -- and thus confirm that a = 112.5, b = 56.25, c = 3
  -- and verifying that a + b + c = 171.75.
  sorry

end NUMINAMATH_GPT_max_area_triangle_bqc_l2404_240484


namespace NUMINAMATH_GPT_profit_A_after_upgrade_profit_B_constrained_l2404_240419

-- Part Ⅰ
theorem profit_A_after_upgrade (x : ℝ) (h : x^2 - 300 * x ≤ 0) : 0 < x ∧ x ≤ 300 := sorry

-- Part Ⅱ
theorem profit_B_constrained (a x : ℝ) (h1 : a ≤ (x/125 + 500/x + 3/2)) (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := sorry

end NUMINAMATH_GPT_profit_A_after_upgrade_profit_B_constrained_l2404_240419


namespace NUMINAMATH_GPT_area_region_eq_6_25_l2404_240498

noncomputable def area_of_region : ℝ :=
  ∫ x in -0.5..4.5, (5 - |x - 2| - |x - 2|)

theorem area_region_eq_6_25 :
  area_of_region = 6.25 :=
sorry

end NUMINAMATH_GPT_area_region_eq_6_25_l2404_240498


namespace NUMINAMATH_GPT_sum_of_acutes_tan_eq_pi_over_4_l2404_240429

theorem sum_of_acutes_tan_eq_pi_over_4 {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h : (1 + Real.tan α) * (1 + Real.tan β) = 2) : α + β = π / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_acutes_tan_eq_pi_over_4_l2404_240429


namespace NUMINAMATH_GPT_g_1986_l2404_240463

def g : ℕ → ℤ := sorry

axiom g_def : ∀ n : ℕ, g n ≥ 0
axiom g_one : g 1 = 3
axiom g_func_eq : ∀ (a b : ℕ), g (a + b) = g a + g b - 3 * g (a * b)

theorem g_1986 : g 1986 = 0 :=
by
  sorry

end NUMINAMATH_GPT_g_1986_l2404_240463


namespace NUMINAMATH_GPT_gcd_g105_g106_l2404_240448

def g (x : ℕ) : ℕ := x^2 - x + 2502

theorem gcd_g105_g106 : gcd (g 105) (g 106) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_g105_g106_l2404_240448


namespace NUMINAMATH_GPT_attendees_chose_water_l2404_240486

theorem attendees_chose_water
  (total_attendees : ℕ)
  (juice_percentage water_percentage : ℝ)
  (attendees_juice : ℕ)
  (h1 : juice_percentage = 0.7)
  (h2 : water_percentage = 0.3)
  (h3 : attendees_juice = 140)
  (h4 : total_attendees * juice_percentage = attendees_juice)
  : total_attendees * water_percentage = 60 := by
  sorry

end NUMINAMATH_GPT_attendees_chose_water_l2404_240486


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l2404_240433

noncomputable def parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 (m : ℝ) : Prop :=
  let l1_slope := -(1 + m) / 1
  let l2_slope := -m / 2
  l1_slope = l2_slope

theorem find_m_for_parallel_lines :
  parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 m →
  m = 1 :=
by
  intro h_parallel
  -- Here we would present the proof steps to show that m = 1 under the given conditions.
  sorry

end NUMINAMATH_GPT_find_m_for_parallel_lines_l2404_240433


namespace NUMINAMATH_GPT_find_a_l2404_240471

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a^2 = 12) : a = 3 :=
by sorry

end NUMINAMATH_GPT_find_a_l2404_240471


namespace NUMINAMATH_GPT_tommy_first_house_price_l2404_240404

theorem tommy_first_house_price (C : ℝ) (P : ℝ) (loan_rate : ℝ) (interest_rate : ℝ)
  (term : ℝ) (property_tax_rate : ℝ) (insurance_cost : ℝ) 
  (price_ratio : ℝ) (monthly_payment : ℝ) :
  C = 500000 ∧ price_ratio = 1.25 ∧ P * price_ratio = C ∧
  loan_rate = 0.75 ∧ interest_rate = 0.035 ∧ term = 15 ∧
  property_tax_rate = 0.015 ∧ insurance_cost = 7500 → 
  P = 400000 :=
by sorry

end NUMINAMATH_GPT_tommy_first_house_price_l2404_240404


namespace NUMINAMATH_GPT_intersection_of_asymptotes_l2404_240424

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6 * x + 8) / (x^2 - 6 * x + 9)

theorem intersection_of_asymptotes :
  ∃ (p : ℝ × ℝ), p = (3, 1) ∧
    (∀ (x : ℝ), x ≠ 3 → f x ≠ 1) ∧
    ((∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → |f x - 1| < ε) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - 1| ∧ |y - 1| < δ → |f (3 + y) - 1| < ε)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_asymptotes_l2404_240424


namespace NUMINAMATH_GPT_no_natural_number_divides_Q_by_x_squared_minus_one_l2404_240462

def Q (n : ℕ) (x : ℝ) : ℝ := 1 + 5*x^2 + x^4 - (n - 1) * x^(n - 1) + (n - 8) * x^n

theorem no_natural_number_divides_Q_by_x_squared_minus_one :
  ∀ (n : ℕ), n > 0 → ¬ (x^2 - 1 ∣ Q n x) :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_no_natural_number_divides_Q_by_x_squared_minus_one_l2404_240462


namespace NUMINAMATH_GPT_union_of_S_and_T_l2404_240416

-- Definitions of the sets S and T
def S : Set ℝ := { y | ∃ x : ℝ, y = Real.exp x - 2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

-- Lean proof problem statement
theorem union_of_S_and_T : (S ∪ T) = { y | -4 ≤ y } :=
by
  sorry

end NUMINAMATH_GPT_union_of_S_and_T_l2404_240416


namespace NUMINAMATH_GPT_grazing_months_b_l2404_240423

theorem grazing_months_b (a_oxen a_months b_oxen c_oxen c_months total_rent c_share : ℕ) (x : ℕ) 
  (h_a : a_oxen = 10) (h_am : a_months = 7) (h_b : b_oxen = 12) 
  (h_c : c_oxen = 15) (h_cm : c_months = 3) (h_tr : total_rent = 105) 
  (h_cs : c_share = 27) : 
  45 * 105 = 27 * (70 + 12 * x + 45) → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_grazing_months_b_l2404_240423


namespace NUMINAMATH_GPT_square_side_length_l2404_240414

theorem square_side_length (radius : ℝ) (s1 s2 : ℝ) (h1 : s1 = s2) (h2 : radius = 2 - Real.sqrt 2):
  s1 = 1 :=
  sorry

end NUMINAMATH_GPT_square_side_length_l2404_240414


namespace NUMINAMATH_GPT_zeros_of_geometric_sequence_quadratic_l2404_240470

theorem zeros_of_geometric_sequence_quadratic (a b c : ℝ) (h_geometric : b^2 = a * c) (h_pos : a * c > 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_GPT_zeros_of_geometric_sequence_quadratic_l2404_240470


namespace NUMINAMATH_GPT_square_of_binomial_l2404_240488

theorem square_of_binomial (a : ℝ) : 16 * x^2 + 32 * x + a = (4 * x + 4)^2 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l2404_240488


namespace NUMINAMATH_GPT_triangular_pyramid_volume_l2404_240482

theorem triangular_pyramid_volume (a b c : ℝ)
  (h1 : 1/2 * a * b = 1.5)
  (h2 : 1/2 * b * c = 2)
  (h3 : 1/2 * a * c = 6) :
  (1/6 * a * b * c = 2) :=
by {
  -- Here, we would provide the proof steps, but for now we leave it as sorry
  sorry
}

end NUMINAMATH_GPT_triangular_pyramid_volume_l2404_240482


namespace NUMINAMATH_GPT_triangle_angles_30_60_90_l2404_240456

-- Definition of the angles based on the given ratio
def angles_ratio (A B C : ℝ) : Prop :=
  A / B = 1 / 2 ∧ B / C = 2 / 3

-- The main statement to be proved
theorem triangle_angles_30_60_90
  (A B C : ℝ)
  (h1 : angles_ratio A B C)
  (h2 : A + B + C = 180) :
  A = 30 ∧ B = 60 ∧ C = 90 := 
sorry

end NUMINAMATH_GPT_triangle_angles_30_60_90_l2404_240456


namespace NUMINAMATH_GPT_birds_joined_l2404_240481

-- Definitions based on the identified conditions
def initial_birds : ℕ := 3
def initial_storks : ℕ := 2
def total_after_joining : ℕ := 10

-- Theorem statement that follows from the problem setup
theorem birds_joined :
  total_after_joining - (initial_birds + initial_storks) = 5 := by
  sorry

end NUMINAMATH_GPT_birds_joined_l2404_240481


namespace NUMINAMATH_GPT_find_circle_center_value_x_plus_y_l2404_240458

theorem find_circle_center_value_x_plus_y : 
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * x - 6 * y + 9) → 
    x + y = -1 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_find_circle_center_value_x_plus_y_l2404_240458


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l2404_240473

-- Definitions based on conditions
def LCM (x y : ℕ) : ℕ := sorry  -- Assume some definition of LCM
def HCF (x y : ℕ) : ℕ := sorry  -- Assume some definition of HCF

-- Given conditions
axiom cond1 (x y : ℕ) : LCM x y = 600
axiom cond2 (x y : ℕ) : x * y = 18000

-- Statement to prove
theorem hcf_of_two_numbers (x y : ℕ) (h1 : LCM x y = 600) (h2 : x * y = 18000) : HCF x y = 30 :=
by {
  -- Proof omitted, hence we use sorry
  sorry
}

end NUMINAMATH_GPT_hcf_of_two_numbers_l2404_240473


namespace NUMINAMATH_GPT_cakes_in_november_l2404_240457

-- Define the function modeling the number of cakes baked each month
def num_of_cakes (initial: ℕ) (n: ℕ) := initial + 2 * n

-- Given conditions
def cakes_in_october := 19
def cakes_in_december := 23
def cakes_in_january := 25
def cakes_in_february := 27
def monthly_increase := 2

-- Prove that the number of cakes baked in November is 21
theorem cakes_in_november : num_of_cakes cakes_in_october 1 = 21 :=
by
  sorry

end NUMINAMATH_GPT_cakes_in_november_l2404_240457


namespace NUMINAMATH_GPT_time_ratio_upstream_downstream_l2404_240417

theorem time_ratio_upstream_downstream (S_boat S_stream D : ℝ) (h1 : S_boat = 72) (h2 : S_stream = 24) :
  let time_upstream := D / (S_boat - S_stream)
  let time_downstream := D / (S_boat + S_stream)
  (time_upstream / time_downstream) = 2 :=
by
  sorry

end NUMINAMATH_GPT_time_ratio_upstream_downstream_l2404_240417


namespace NUMINAMATH_GPT_trader_sold_40_meters_l2404_240444

noncomputable def meters_of_cloth_sold (profit_per_meter total_profit : ℕ) : ℕ :=
  total_profit / profit_per_meter

theorem trader_sold_40_meters (profit_per_meter total_profit : ℕ) (h1 : profit_per_meter = 35) (h2 : total_profit = 1400) :
  meters_of_cloth_sold profit_per_meter total_profit = 40 :=
by
  sorry

end NUMINAMATH_GPT_trader_sold_40_meters_l2404_240444


namespace NUMINAMATH_GPT_committee_count_l2404_240422

theorem committee_count (total_students : ℕ) (include_students : ℕ) (choose_students : ℕ) :
  total_students = 8 → include_students = 2 → choose_students = 3 →
  Nat.choose (total_students - include_students) choose_students = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_committee_count_l2404_240422


namespace NUMINAMATH_GPT_second_term_of_geometric_series_l2404_240440

theorem second_term_of_geometric_series (a r S: ℝ) (h_r : r = 1/4) (h_S : S = 40) (h_geom_sum : S = a / (1 - r)) : a * r = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_second_term_of_geometric_series_l2404_240440


namespace NUMINAMATH_GPT_find_integer_x_l2404_240406

open Nat

noncomputable def isSquareOfPrime (n : ℤ) : Prop :=
  ∃ p : ℤ, Nat.Prime (Int.natAbs p) ∧ n = p * p

theorem find_integer_x :
  ∃ x : ℤ,
  (x = -360 ∨ x = -60 ∨ x = -48 ∨ x = -40 ∨ x = 8 ∨ x = 20 ∨ x = 32 ∨ x = 332) ∧
  isSquareOfPrime (x^2 + 28*x + 889) :=
sorry

end NUMINAMATH_GPT_find_integer_x_l2404_240406


namespace NUMINAMATH_GPT_square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l2404_240425

theorem square_roots_of_four_ninths : {x : ℚ | x ^ 2 = 4 / 9} = {2 / 3, -2 / 3} :=
by
  sorry

theorem cube_root_of_neg_sixty_four : {y : ℚ | y ^ 3 = -64} = {-4} :=
by
  sorry

end NUMINAMATH_GPT_square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l2404_240425


namespace NUMINAMATH_GPT_minimize_fencing_l2404_240435

def area_requirement (w : ℝ) : Prop :=
  2 * (w * w) ≥ 800

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

def perimeter (w l : ℝ) : ℝ :=
  2 * l + 2 * w

theorem minimize_fencing (w l : ℝ) (h1 : area_requirement w) (h2 : length_twice_width l w) :
  w = 20 ∧ l = 40 :=
by
  sorry

end NUMINAMATH_GPT_minimize_fencing_l2404_240435


namespace NUMINAMATH_GPT_line_passing_through_quadrants_l2404_240474

theorem line_passing_through_quadrants (a : ℝ) :
  (∀ x : ℝ, (3 * a - 1) * x - 1 ≠ 0) →
  (3 * a - 1 > 0) →
  a > 1 / 3 :=
by
  intro h1 h2
  -- proof to be filled
  sorry

end NUMINAMATH_GPT_line_passing_through_quadrants_l2404_240474


namespace NUMINAMATH_GPT_f_a1_a3_a5_positive_l2404_240468

theorem f_a1_a3_a5_positive (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf_odd : ∀ x, f (-x) = - f x)
  (hf_mono : ∀ x y, x < y → f x < f y)
  (ha_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha3_pos : 0 < a 3) :
  0 < f (a 1) + f (a 3) + f (a 5) :=
sorry

end NUMINAMATH_GPT_f_a1_a3_a5_positive_l2404_240468


namespace NUMINAMATH_GPT_number_of_Ca_atoms_in_compound_l2404_240493

theorem number_of_Ca_atoms_in_compound
  (n : ℤ)
  (total_weight : ℝ)
  (ca_weight : ℝ)
  (i_weight : ℝ)
  (n_i_atoms : ℤ)
  (molecular_weight : ℝ) :
  n_i_atoms = 2 →
  molecular_weight = 294 →
  ca_weight = 40.08 →
  i_weight = 126.90 →
  n * ca_weight + n_i_atoms * i_weight = molecular_weight →
  n = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_Ca_atoms_in_compound_l2404_240493


namespace NUMINAMATH_GPT_james_vegetable_intake_l2404_240453

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end NUMINAMATH_GPT_james_vegetable_intake_l2404_240453


namespace NUMINAMATH_GPT_beth_finishes_first_l2404_240459

open Real

noncomputable def andy_lawn_area : ℝ := sorry
noncomputable def beth_lawn_area : ℝ := andy_lawn_area / 3
noncomputable def carlos_lawn_area : ℝ := andy_lawn_area / 4

noncomputable def andy_mowing_rate : ℝ := sorry
noncomputable def beth_mowing_rate : ℝ := andy_mowing_rate
noncomputable def carlos_mowing_rate : ℝ := andy_mowing_rate / 2

noncomputable def carlos_break : ℝ := 10

noncomputable def andy_mowing_time := andy_lawn_area / andy_mowing_rate
noncomputable def beth_mowing_time := beth_lawn_area / beth_mowing_rate
noncomputable def carlos_mowing_time := (carlos_lawn_area / carlos_mowing_rate) + carlos_break

theorem beth_finishes_first :
  beth_mowing_time < andy_mowing_time ∧ beth_mowing_time < carlos_mowing_time := by
  sorry

end NUMINAMATH_GPT_beth_finishes_first_l2404_240459


namespace NUMINAMATH_GPT_monkey_swinging_speed_l2404_240427

namespace LamplighterMonkey

def running_speed : ℝ := 15
def running_time : ℝ := 5
def swinging_time : ℝ := 10
def total_distance : ℝ := 175

theorem monkey_swinging_speed : 
  (total_distance = running_speed * running_time + (running_speed / swinging_time) * swinging_time) → 
  (running_speed / swinging_time = 10) := 
by 
  intros h
  sorry

end LamplighterMonkey

end NUMINAMATH_GPT_monkey_swinging_speed_l2404_240427


namespace NUMINAMATH_GPT_general_solution_of_diff_eq_l2404_240415

theorem general_solution_of_diff_eq
  (f : ℝ → ℝ → ℝ)
  (D : Set (ℝ × ℝ))
  (hf : ∀ x y, f x y = x)
  (hD : D = Set.univ) :
  ∃ C : ℝ, ∀ x : ℝ, ∃ y : ℝ, y = (x^2) / 2 + C :=
by
  sorry

end NUMINAMATH_GPT_general_solution_of_diff_eq_l2404_240415


namespace NUMINAMATH_GPT_valid_arrangements_count_is_20_l2404_240490

noncomputable def count_valid_arrangements : ℕ :=
  sorry

theorem valid_arrangements_count_is_20 :
  count_valid_arrangements = 20 :=
  by
    sorry

end NUMINAMATH_GPT_valid_arrangements_count_is_20_l2404_240490


namespace NUMINAMATH_GPT_comb_12_9_eq_220_l2404_240469

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end NUMINAMATH_GPT_comb_12_9_eq_220_l2404_240469


namespace NUMINAMATH_GPT_age_sum_is_47_l2404_240412

theorem age_sum_is_47 (a b c : ℕ) (b_def : b = 18) 
  (a_def : a = b + 2) (c_def : c = b / 2) : a + b + c = 47 :=
by
  sorry

end NUMINAMATH_GPT_age_sum_is_47_l2404_240412


namespace NUMINAMATH_GPT_greatest_possible_value_of_a_l2404_240487

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ x : ℤ, x * (x + a) = -24 → x * (x + a) = -24) ∧ (∀ b : ℕ, (∀ x : ℤ, x * (x + b) = -24 → x * (x + b) = -24) → b ≤ a) ∧ a = 25 :=
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_a_l2404_240487


namespace NUMINAMATH_GPT_third_even_number_sequence_l2404_240432

theorem third_even_number_sequence (x : ℕ) (h_even : x % 2 = 0) (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) = 180) : x + 4 = 30 :=
by
  sorry

end NUMINAMATH_GPT_third_even_number_sequence_l2404_240432


namespace NUMINAMATH_GPT_size_of_former_apartment_l2404_240480

open Nat

theorem size_of_former_apartment
  (former_rent_rate : ℕ)
  (new_apartment_cost : ℕ)
  (savings_per_year : ℕ)
  (split_factor : ℕ)
  (savings_per_month : ℕ)
  (share_new_rent : ℕ)
  (former_rent : ℕ)
  (apartment_size : ℕ)
  (h1 : former_rent_rate = 2)
  (h2 : new_apartment_cost = 2800)
  (h3 : savings_per_year = 1200)
  (h4 : split_factor = 2)
  (h5 : savings_per_month = savings_per_year / 12)
  (h6 : share_new_rent = new_apartment_cost / split_factor)
  (h7 : former_rent = share_new_rent + savings_per_month)
  (h8 : apartment_size = former_rent / former_rent_rate) :
  apartment_size = 750 :=
by
  sorry

end NUMINAMATH_GPT_size_of_former_apartment_l2404_240480


namespace NUMINAMATH_GPT_C0E_hex_to_dec_l2404_240441

theorem C0E_hex_to_dec : 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  result = 3086 :=
by 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  sorry

end NUMINAMATH_GPT_C0E_hex_to_dec_l2404_240441


namespace NUMINAMATH_GPT_find_three_digit_number_l2404_240461

theorem find_three_digit_number (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (h_sum : 122 * a + 212 * b + 221 * c = 2003) :
  100 * a + 10 * b + c = 345 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l2404_240461


namespace NUMINAMATH_GPT_good_permutations_count_l2404_240445

-- Define the main problem and the conditions
theorem good_permutations_count (n : ℕ) (hn : n > 0) : 
  ∃ P : ℕ → ℕ, 
  (P n = (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2) ^ (n + 1) - ((1 - Real.sqrt 5) / 2) ^ (n + 1))) := 
sorry

end NUMINAMATH_GPT_good_permutations_count_l2404_240445


namespace NUMINAMATH_GPT_jeff_cat_shelter_l2404_240449

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end NUMINAMATH_GPT_jeff_cat_shelter_l2404_240449


namespace NUMINAMATH_GPT_tourists_count_l2404_240460

theorem tourists_count (n k : ℤ) (h1 : 2 * k % n = 1) (h2 : 3 * k % n = 13) : n = 23 := 
by
-- Proof is omitted
sorry

end NUMINAMATH_GPT_tourists_count_l2404_240460


namespace NUMINAMATH_GPT_maria_total_earnings_l2404_240476

noncomputable def total_earnings : ℕ := 
  let tulips_day1 := 30
  let roses_day1 := 20
  let lilies_day1 := 15
  let sunflowers_day1 := 10
  let tulips_day2 := tulips_day1 * 2
  let roses_day2 := roses_day1 * 2
  let lilies_day2 := lilies_day1
  let sunflowers_day2 := sunflowers_day1 * 3
  let tulips_day3 := tulips_day2 / 10
  let roses_day3 := 16
  let lilies_day3 := lilies_day1 / 2
  let sunflowers_day3 := sunflowers_day2
  let price_tulip := 2
  let price_rose := 3
  let price_lily := 4
  let price_sunflower := 5
  let day1_earnings := tulips_day1 * price_tulip + roses_day1 * price_rose + lilies_day1 * price_lily + sunflowers_day1 * price_sunflower
  let day2_earnings := tulips_day2 * price_tulip + roses_day2 * price_rose + lilies_day2 * price_lily + sunflowers_day2 * price_sunflower
  let day3_earnings := tulips_day3 * price_tulip + roses_day3 * price_rose + lilies_day3 * price_lily + sunflowers_day3 * price_sunflower
  day1_earnings + day2_earnings + day3_earnings

theorem maria_total_earnings : total_earnings = 920 := 
by 
  unfold total_earnings
  sorry

end NUMINAMATH_GPT_maria_total_earnings_l2404_240476


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l2404_240408

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 5) 
  (h2 : a * r^3 = 45) : 
  a = 5 / (3^(2/3)) := 
by
  -- proof steps to be filled here
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l2404_240408


namespace NUMINAMATH_GPT_sarah_driving_distance_l2404_240496

def sarah_car_mileage (miles_per_gallon : ℕ) (tank_capacity : ℕ) (initial_drive : ℕ) (refuel : ℕ) (remaining_fraction : ℚ) : Prop :=
  ∃ (total_drive : ℚ),
    (initial_drive / miles_per_gallon + refuel - (tank_capacity * remaining_fraction / 1)) * miles_per_gallon = total_drive ∧
    total_drive = 467

theorem sarah_driving_distance :
  sarah_car_mileage 28 16 280 6 (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sarah_driving_distance_l2404_240496


namespace NUMINAMATH_GPT_min_value_x_y_xy_l2404_240409

theorem min_value_x_y_xy (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  x + y + x * y ≥ -9 / 8 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_xy_l2404_240409


namespace NUMINAMATH_GPT_pentomino_symmetry_count_l2404_240489

noncomputable def num_symmetric_pentominoes : Nat :=
  15 -- This represents the given set of 15 different pentominoes

noncomputable def symmetric_pentomino_count : Nat :=
  -- Here we are asserting that the count of pentominoes with at least one vertical symmetry is 8
  8

theorem pentomino_symmetry_count :
  symmetric_pentomino_count = 8 :=
sorry

end NUMINAMATH_GPT_pentomino_symmetry_count_l2404_240489
