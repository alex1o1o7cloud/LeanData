import Mathlib

namespace NUMINAMATH_GPT_find_e_l1788_178882

theorem find_e (x y e : ℝ) (h1 : x / (2 * y) = 5 / e) (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) : e = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_e_l1788_178882


namespace NUMINAMATH_GPT_evaluate_expression_l1788_178814

theorem evaluate_expression : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1788_178814


namespace NUMINAMATH_GPT_inequality_proof_l1788_178847

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom abc_eq_one : a * b * c = 1

theorem inequality_proof :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1788_178847


namespace NUMINAMATH_GPT_g_five_l1788_178824

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_one : g 1 = 2

theorem g_five : g 5 = 10 :=
by sorry

end NUMINAMATH_GPT_g_five_l1788_178824


namespace NUMINAMATH_GPT_cow_cost_calculation_l1788_178884

theorem cow_cost_calculation (C cow calf : ℝ) 
  (h1 : cow = 8 * calf) 
  (h2 : cow + calf = 990) : 
  cow = 880 :=
by
  sorry

end NUMINAMATH_GPT_cow_cost_calculation_l1788_178884


namespace NUMINAMATH_GPT_expression_bounds_l1788_178852

theorem expression_bounds (a b c d : ℝ) (h0a : 0 ≤ a) (h1a : a ≤ 1) (h0b : 0 ≤ b) (h1b : b ≤ 1)
  (h0c : 0 ≤ c) (h1c : c ≤ 1) (h0d : 0 ≤ d) (h1d : d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ∧
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end NUMINAMATH_GPT_expression_bounds_l1788_178852


namespace NUMINAMATH_GPT_inequality_proof_l1788_178851

-- Define the main theorem with the conditions
theorem inequality_proof 
  (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a = b ∧ b = c ∧ c = d) ↔ (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a)) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1788_178851


namespace NUMINAMATH_GPT_find_prob_real_roots_l1788_178898

-- Define the polynomial q(x)
def q (a : ℝ) (x : ℝ) : ℝ := x^4 + 3*a*x^3 + (3*a - 5)*x^2 + (-6*a + 4)*x - 3

-- Define the conditions for a to ensure all roots of the polynomial are real
noncomputable def all_roots_real_condition (a : ℝ) : Prop :=
  a ≤ -1/3 ∨ 1 ≤ a

-- Define the probability that given a in the interval [-12, 32] all q's roots are real
noncomputable def probability_real_roots : ℝ :=
  let total_length := 32 - (-12)
  let excluded_interval_length := 1 - (-1/3)
  let valid_interval_length := total_length - excluded_interval_length
  valid_interval_length / total_length

-- State the theorem
theorem find_prob_real_roots :
  probability_real_roots = 32 / 33 :=
sorry

end NUMINAMATH_GPT_find_prob_real_roots_l1788_178898


namespace NUMINAMATH_GPT_find_a_l1788_178828

open Real

def is_chord_length_correct (a : ℝ) : Prop :=
  let x_line := fun t : ℝ => 1 + t
  let y_line := fun t : ℝ => a - t
  let x_circle := fun α : ℝ => 2 + 2 * cos α
  let y_circle := fun α : ℝ => 2 + 2 * sin α
  let distance_from_center := abs (3 - a) / sqrt 2
  let chord_length := 2 * sqrt (4 - distance_from_center ^ 2)
  chord_length = 2 * sqrt 2 

theorem find_a (a : ℝ) : is_chord_length_correct a → a = 1 ∨ a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1788_178828


namespace NUMINAMATH_GPT_total_legs_correct_l1788_178871

def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goats : ℕ := 1
def legs_per_animal : ℕ := 4

theorem total_legs_correct :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_correct_l1788_178871


namespace NUMINAMATH_GPT_area_of_perpendicular_triangle_l1788_178841

theorem area_of_perpendicular_triangle 
  (S R d : ℝ) (S' : ℝ) -- defining the variables and constants
  (h1 : S > 0) (h2 : R > 0) (h3 : d ≥ 0) :
  S' = (S / 4) * |1 - (d^2 / R^2)| := 
sorry

end NUMINAMATH_GPT_area_of_perpendicular_triangle_l1788_178841


namespace NUMINAMATH_GPT_new_op_4_3_l1788_178813

def new_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem new_op_4_3 : new_op 4 3 = 13 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_new_op_4_3_l1788_178813


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1788_178896

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) : 
  1/x + 1/y = 14/45 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1788_178896


namespace NUMINAMATH_GPT_fourth_place_points_l1788_178857

variables (x : ℕ)

def points_awarded (place : ℕ) : ℕ :=
  if place = 1 then 11
  else if place = 2 then 7
  else if place = 3 then 5
  else if place = 4 then x
  else 0

theorem fourth_place_points:
  (∃ a b c y u : ℕ, a + b + c + y + u = 7 ∧ points_awarded x 1 ^ a * points_awarded x 2 ^ b * points_awarded x 3 ^ c * points_awarded x 4 ^ y * 1 ^ u = 38500) →
  x = 4 :=
sorry

end NUMINAMATH_GPT_fourth_place_points_l1788_178857


namespace NUMINAMATH_GPT_slope_of_line_passes_through_points_l1788_178872

theorem slope_of_line_passes_through_points :
  let k := (2 + Real.sqrt 3 - 2) / (4 - 1)
  k = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_passes_through_points_l1788_178872


namespace NUMINAMATH_GPT_tetrahedron_labeling_count_l1788_178862

def is_valid_tetrahedron_labeling (labeling : Fin 4 → ℕ) : Prop :=
  let f1 := labeling 0 + labeling 1 + labeling 2
  let f2 := labeling 0 + labeling 1 + labeling 3
  let f3 := labeling 0 + labeling 2 + labeling 3
  let f4 := labeling 1 + labeling 2 + labeling 3
  labeling 0 + labeling 1 + labeling 2 + labeling 3 = 10 ∧ 
  f1 = f2 ∧ f2 = f3 ∧ f3 = f4

theorem tetrahedron_labeling_count : 
  ∃ (n : ℕ), n = 3 ∧ (∃ (labelings: Finset (Fin 4 → ℕ)), 
  ∀ labeling ∈ labelings, is_valid_tetrahedron_labeling labeling) :=
sorry

end NUMINAMATH_GPT_tetrahedron_labeling_count_l1788_178862


namespace NUMINAMATH_GPT_greatest_third_term_of_arithmetic_sequence_l1788_178820

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end NUMINAMATH_GPT_greatest_third_term_of_arithmetic_sequence_l1788_178820


namespace NUMINAMATH_GPT_max_sum_of_digits_l1788_178873

theorem max_sum_of_digits (a b c : ℕ) (x : ℕ) (N : ℕ) :
  N = 100 * a + 10 * b + c →
  100 <= N →
  N < 1000 →
  a ≠ 0 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1730 + x →
  a + b + c = 20 :=
by
  intros hN hN_ge_100 hN_lt_1000 ha_ne_0 hsum
  sorry

end NUMINAMATH_GPT_max_sum_of_digits_l1788_178873


namespace NUMINAMATH_GPT_garden_length_to_width_ratio_l1788_178875

theorem garden_length_to_width_ratio (area : ℕ) (width : ℕ) (h_area : area = 432) (h_width : width = 12) :
  ∃ length : ℕ, length = area / width ∧ (length / width = 3) := 
by
  sorry

end NUMINAMATH_GPT_garden_length_to_width_ratio_l1788_178875


namespace NUMINAMATH_GPT_Jimin_scabs_l1788_178877

theorem Jimin_scabs (total_scabs : ℕ) (days_in_week : ℕ) (daily_scabs: ℕ)
  (h₁ : total_scabs = 220) (h₂ : days_in_week = 7) 
  (h₃ : daily_scabs = (total_scabs + days_in_week - 1) / days_in_week) : 
  daily_scabs ≥ 32 := by
  sorry

end NUMINAMATH_GPT_Jimin_scabs_l1788_178877


namespace NUMINAMATH_GPT_number_of_8_digit_integers_l1788_178883

theorem number_of_8_digit_integers : 
  ∃ n, n = 90000000 ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
     d1 ≠ 0 → 0 ≤ d1 ∧ d1 ≤ 9 ∧ 
     0 ≤ d2 ∧ d2 ≤ 9 ∧ 
     0 ≤ d3 ∧ d3 ≤ 9 ∧ 
     0 ≤ d4 ∧ d4 ≤ 9 ∧ 
     0 ≤ d5 ∧ d5 ≤ 9 ∧ 
     0 ≤ d6 ∧ d6 ≤ 9 ∧ 
     0 ≤ d7 ∧ d7 ≤ 9 ∧ 
     0 ≤ d8 ∧ d8 ≤ 9 →
     ∀ count, count = (if d1 ≠ 0 then 9 * 10^7 else 0)) :=
sorry

end NUMINAMATH_GPT_number_of_8_digit_integers_l1788_178883


namespace NUMINAMATH_GPT_total_distance_traveled_l1788_178894

noncomputable def travel_distance (speed : ℝ) (time : ℝ) (headwind : ℝ) : ℝ :=
  (speed - headwind) * time

theorem total_distance_traveled :
  let headwind := 5
  let eagle_speed := 15
  let eagle_time := 2.5
  let eagle_distance := travel_distance eagle_speed eagle_time headwind

  let falcon_speed := 46
  let falcon_time := 2.5
  let falcon_distance := travel_distance falcon_speed falcon_time headwind

  let pelican_speed := 33
  let pelican_time := 2.5
  let pelican_distance := travel_distance pelican_speed pelican_time headwind

  let hummingbird_speed := 30
  let hummingbird_time := 2.5
  let hummingbird_distance := travel_distance hummingbird_speed hummingbird_time headwind

  let hawk_speed := 45
  let hawk_time := 3
  let hawk_distance := travel_distance hawk_speed hawk_time headwind

  let swallow_speed := 25
  let swallow_time := 1.5
  let swallow_distance := travel_distance swallow_speed swallow_time headwind

  eagle_distance + falcon_distance + pelican_distance + hummingbird_distance + hawk_distance + swallow_distance = 410 :=
sorry

end NUMINAMATH_GPT_total_distance_traveled_l1788_178894


namespace NUMINAMATH_GPT_exponent_multiplication_l1788_178801

theorem exponent_multiplication :
  (10^(3/4)) * (10^(-0.25)) * (10^(1.5)) = 10^2 :=
by sorry

end NUMINAMATH_GPT_exponent_multiplication_l1788_178801


namespace NUMINAMATH_GPT_count_numbers_divisible_by_12_not_20_l1788_178802

theorem count_numbers_divisible_by_12_not_20 : 
  let N := 2017
  let a := Nat.floor (N / 12)
  let b := Nat.floor (N / 60)
  a - b = 135 := by
    -- Definitions used
    let N := 2017
    let a := Nat.floor (N / 12)
    let b := Nat.floor (N / 60)
    -- The desired statement
    show a - b = 135
    sorry

end NUMINAMATH_GPT_count_numbers_divisible_by_12_not_20_l1788_178802


namespace NUMINAMATH_GPT_hexagon_ratio_identity_l1788_178839

theorem hexagon_ratio_identity
  (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (AB BC CD DE EF FA : ℝ)
  (angle_B angle_D angle_F : ℝ)
  (h1 : AB / BC * CD / DE * EF / FA = 1)
  (h2 : angle_B + angle_D + angle_F = 360) :
  (BC / AC * AE / EF * FD / DB = 1) := sorry

end NUMINAMATH_GPT_hexagon_ratio_identity_l1788_178839


namespace NUMINAMATH_GPT_greatest_x_l1788_178854

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_l1788_178854


namespace NUMINAMATH_GPT_count_valid_choices_l1788_178863

open Nat

def base4_representation (N : ℕ) : ℕ := 
  let a3 := N / 64 % 4
  let a2 := N / 16 % 4
  let a1 := N / 4 % 4
  let a0 := N % 4
  64 * a3 + 16 * a2 + 4 * a1 + a0

def base7_representation (N : ℕ) : ℕ := 
  let b3 := N / 343 % 7
  let b2 := N / 49 % 7
  let b1 := N / 7 % 7
  let b0 := N % 7
  343 * b3 + 49 * b2 + 7 * b1 + b0

def S (N : ℕ) : ℕ := base4_representation N + base7_representation N

def valid_choices (N : ℕ) : Prop := 
  (S N % 100) = (2 * N % 100)

theorem count_valid_choices : 
  ∃ (count : ℕ), count = 20 ∧ ∀ (N : ℕ), (N >= 1000 ∧ N < 10000) → valid_choices N ↔ (count = 20) :=
sorry

end NUMINAMATH_GPT_count_valid_choices_l1788_178863


namespace NUMINAMATH_GPT_exists_m_n_for_d_l1788_178887

theorem exists_m_n_for_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) := 
sorry

end NUMINAMATH_GPT_exists_m_n_for_d_l1788_178887


namespace NUMINAMATH_GPT_bisection_method_termination_condition_l1788_178868

theorem bisection_method_termination_condition (x1 x2 e : ℝ) (h : e > 0) :
  |x1 - x2| < e → true :=
sorry

end NUMINAMATH_GPT_bisection_method_termination_condition_l1788_178868


namespace NUMINAMATH_GPT_probability_value_at_least_75_cents_l1788_178809

-- Given conditions
def box_contains (pennies nickels quarters : ℕ) : Prop :=
  pennies = 4 ∧ nickels = 3 ∧ quarters = 5

def draw_without_replacement (total_coins : ℕ) (drawn_coins : ℕ) : Prop :=
  total_coins = 12 ∧ drawn_coins = 5

def equal_probability (chosen_probability : ℚ) (total_coins : ℕ) : Prop :=
  chosen_probability = 1/total_coins

-- Probability that the value of coins drawn is at least 75 cents
theorem probability_value_at_least_75_cents
  (pennies nickels quarters total_coins drawn_coins : ℕ)
  (chosen_probability : ℚ) :
  box_contains pennies nickels quarters →
  draw_without_replacement total_coins drawn_coins →
  equal_probability chosen_probability total_coins →
  chosen_probability = 1/792 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_value_at_least_75_cents_l1788_178809


namespace NUMINAMATH_GPT_value_of_f_1_plus_g_4_l1788_178845

def f (x : Int) : Int := 2 * x - 1
def g (x : Int) : Int := x + 1

theorem value_of_f_1_plus_g_4 : f (1 + g 4) = 11 := by
  sorry

end NUMINAMATH_GPT_value_of_f_1_plus_g_4_l1788_178845


namespace NUMINAMATH_GPT_mixed_fraction_product_example_l1788_178823

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_mixed_fraction_product_example_l1788_178823


namespace NUMINAMATH_GPT_value_of_expression_l1788_178832

variable (a b : ℝ)

theorem value_of_expression : 
  let x := a + b 
  let y := a - b 
  (x - y) * (x + y) = 4 * a * b := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1788_178832


namespace NUMINAMATH_GPT_Avery_builds_in_4_hours_l1788_178864

variable (A : ℝ) (TomTime : ℝ := 2) (TogetherTime : ℝ := 1) (RemainingTomTime : ℝ := 0.5)

-- Conditions:
axiom Tom_builds_in_2_hours : TomTime = 2
axiom Work_together_for_1_hour : TogetherTime = 1
axiom Tom_finishes_in_0_5_hours : RemainingTomTime = 0.5

-- Question:
theorem Avery_builds_in_4_hours : A = 4 :=
by
  sorry

end NUMINAMATH_GPT_Avery_builds_in_4_hours_l1788_178864


namespace NUMINAMATH_GPT_smallest_n_l1788_178890

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3 * n = k ^ 2) (h2 : ∃ m : ℕ, 5 * n = m ^ 5) : n = 151875 := sorry

end NUMINAMATH_GPT_smallest_n_l1788_178890


namespace NUMINAMATH_GPT_student_chose_number_l1788_178838

theorem student_chose_number : ∃ x : ℤ, 2 * x - 152 = 102 ∧ x = 127 :=
by
  sorry

end NUMINAMATH_GPT_student_chose_number_l1788_178838


namespace NUMINAMATH_GPT_sister_height_on_birthday_l1788_178869

theorem sister_height_on_birthday (previous_height : ℝ) (growth_rate : ℝ)
    (h_previous_height : previous_height = 139.65)
    (h_growth_rate : growth_rate = 0.05) :
    previous_height * (1 + growth_rate) = 146.6325 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sister_height_on_birthday_l1788_178869


namespace NUMINAMATH_GPT_cars_in_parking_lot_l1788_178800

theorem cars_in_parking_lot (C : ℕ) (customers_per_car : ℕ) (total_purchases : ℕ) 
  (h1 : customers_per_car = 5)
  (h2 : total_purchases = 50)
  (h3 : C * customers_per_car = total_purchases) : 
  C = 10 := 
by
  sorry

end NUMINAMATH_GPT_cars_in_parking_lot_l1788_178800


namespace NUMINAMATH_GPT_sum_of_digits_divisible_by_45_l1788_178819

theorem sum_of_digits_divisible_by_45 (a b : ℕ) (h1 : b = 0 ∨ b = 5) (h2 : (21 + a + b) % 9 = 0) : a + b = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_divisible_by_45_l1788_178819


namespace NUMINAMATH_GPT_first_team_speed_l1788_178811

theorem first_team_speed:
  ∃ v: ℝ, 
  (∀ (t: ℝ), t = 2.5 → 
  (∀ s: ℝ, s = 125 → 
  (v + 30) * t = s) ∧ v = 20) := 
  sorry

end NUMINAMATH_GPT_first_team_speed_l1788_178811


namespace NUMINAMATH_GPT_syllogism_correct_l1788_178836

-- Hypotheses for each condition
def OptionA := "The first section, the second section, the third section"
def OptionB := "Major premise, minor premise, conclusion"
def OptionC := "Induction, conjecture, proof"
def OptionD := "Dividing the discussion into three sections"

-- Definition of a syllogism in deductive reasoning
def syllogism_def := "A logical argument that applies deductive reasoning to arrive at a conclusion based on two propositions assumed to be true"

-- Theorem stating that a syllogism corresponds to Option B
theorem syllogism_correct :
  syllogism_def = OptionB :=
by
  sorry

end NUMINAMATH_GPT_syllogism_correct_l1788_178836


namespace NUMINAMATH_GPT_train_problem_l1788_178892

theorem train_problem (Sat M S C : ℕ) 
  (h_boarding_day : true)
  (h_arrival_day : true)
  (h_date_matches_car_on_monday : M = C)
  (h_seat_less_than_car : S < C)
  (h_sat_date_greater_than_car : Sat > C) :
  C = 2 ∧ S = 1 :=
by sorry

end NUMINAMATH_GPT_train_problem_l1788_178892


namespace NUMINAMATH_GPT_value_of_expression_l1788_178827

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : 3 * m^2 + 3 * m + 2006 = 2009 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1788_178827


namespace NUMINAMATH_GPT_tv_sales_value_increase_l1788_178867

theorem tv_sales_value_increase (P V : ℝ) :
    let P1 := 0.82 * P
    let V1 := 1.72 * V
    let P2 := 0.75 * P1
    let V2 := 1.90 * V1
    let initial_sales := P * V
    let final_sales := P2 * V2
    final_sales = 2.00967 * initial_sales :=
by
  sorry

end NUMINAMATH_GPT_tv_sales_value_increase_l1788_178867


namespace NUMINAMATH_GPT_solve_equation_l1788_178846

theorem solve_equation : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1788_178846


namespace NUMINAMATH_GPT_smallest_divisor_of_7614_l1788_178826

theorem smallest_divisor_of_7614 (h : Nat) (H_h_eq : h = 1) (n : Nat) (H_n_eq : n = (7600 + 10 * h + 4)) :
  ∃ d, d > 1 ∧ d ∣ n ∧ ∀ x, x > 1 ∧ x ∣ n → d ≤ x :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisor_of_7614_l1788_178826


namespace NUMINAMATH_GPT_Nancy_needs_5_loads_l1788_178870

/-- Definition of the given problem conditions. -/
def pieces_of_clothing (shirts sweaters socks jeans : ℕ) : ℕ :=
  shirts + sweaters + socks + jeans

def washing_machine_capacity : ℕ := 12

def loads_required (total_clothing capacity : ℕ) : ℕ :=
  (total_clothing + capacity - 1) / capacity -- integer division with rounding up

/-- Theorem statement. -/
theorem Nancy_needs_5_loads :
  loads_required (pieces_of_clothing 19 8 15 10) washing_machine_capacity = 5 :=
by
  -- Insert proof here when needed.
  sorry

end NUMINAMATH_GPT_Nancy_needs_5_loads_l1788_178870


namespace NUMINAMATH_GPT_beth_students_proof_l1788_178805

-- Let initial := 150
-- Let joined := 30
-- Let left := 15
-- final := initial + joined - left
-- Prove final = 165

def beth_final_year_students (initial joined left final : ℕ) : Prop :=
  initial = 150 ∧ joined = 30 ∧ left = 15 ∧ final = initial + joined - left

theorem beth_students_proof : ∃ final, beth_final_year_students 150 30 15 final ∧ final = 165 :=
by
  sorry

end NUMINAMATH_GPT_beth_students_proof_l1788_178805


namespace NUMINAMATH_GPT_probability_three_common_books_l1788_178804

-- Defining the total number of books
def total_books : ℕ := 12

-- Defining the number of books each of Harold and Betty chooses
def books_per_person : ℕ := 6

-- Assertion that the probability of choosing exactly 3 common books is 50/116
theorem probability_three_common_books :
  ((Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)) /
  ((Nat.choose 12 6) * (Nat.choose 12 6)) = 50 / 116 := by
  sorry

end NUMINAMATH_GPT_probability_three_common_books_l1788_178804


namespace NUMINAMATH_GPT_simplify_and_evaluate_problem_l1788_178803

noncomputable def problem_expression (a : ℤ) : ℚ :=
  (1 - (3 : ℚ) / (a + 1)) / ((a^2 - 4 * a + 4 : ℚ) / (a + 1))

theorem simplify_and_evaluate_problem :
  ∀ (a : ℤ), -2 ≤ a ∧ a ≤ 2 → a ≠ -1 → a ≠ 2 →
  (problem_expression a = 1 / (a - 2 : ℚ)) ∧
  (a = 0 → problem_expression a = -1 / 2) ∧
  (a = 1 → problem_expression a = -1) :=
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_problem_l1788_178803


namespace NUMINAMATH_GPT_positive_difference_prime_factors_159137_l1788_178853

-- Lean 4 Statement Following the Instructions
theorem positive_difference_prime_factors_159137 :
  (159137 = 11 * 17 * 23 * 37) → (37 - 23 = 14) :=
by
  intro h
  sorry -- Proof will be written here

end NUMINAMATH_GPT_positive_difference_prime_factors_159137_l1788_178853


namespace NUMINAMATH_GPT_math_problem_example_l1788_178825

theorem math_problem_example (m n : ℤ) (h0 : m > 0) (h1 : n > 0)
    (h2 : 3 * m + 2 * n = 225) (h3 : Int.gcd m n = 15) : m + n = 105 :=
sorry

end NUMINAMATH_GPT_math_problem_example_l1788_178825


namespace NUMINAMATH_GPT_vanilla_syrup_cost_l1788_178815

theorem vanilla_syrup_cost :
  ∀ (unit_cost_drip : ℝ) (num_drip : ℕ)
    (unit_cost_espresso : ℝ) (num_espresso : ℕ)
    (unit_cost_latte : ℝ) (num_lattes : ℕ)
    (unit_cost_cold_brew : ℝ) (num_cold_brews : ℕ)
    (unit_cost_cappuccino : ℝ) (num_cappuccino : ℕ)
    (total_cost : ℝ) (vanilla_cost : ℝ),
  unit_cost_drip = 2.25 →
  num_drip = 2 →
  unit_cost_espresso = 3.50 →
  num_espresso = 1 →
  unit_cost_latte = 4.00 →
  num_lattes = 2 →
  unit_cost_cold_brew = 2.50 →
  num_cold_brews = 2 →
  unit_cost_cappuccino = 3.50 →
  num_cappuccino = 1 →
  total_cost = 25.00 →
  vanilla_cost =
    total_cost -
    ((unit_cost_drip * num_drip) +
    (unit_cost_espresso * num_espresso) +
    (unit_cost_latte * (num_lattes - 1)) +
    (unit_cost_cold_brew * num_cold_brews) +
    (unit_cost_cappuccino * num_cappuccino)) →
  vanilla_cost = 0.50 := sorry

end NUMINAMATH_GPT_vanilla_syrup_cost_l1788_178815


namespace NUMINAMATH_GPT_sqrt_three_pow_divisible_l1788_178817

/-- For any non-negative integer n, (1 + sqrt 3)^(2*n + 1) is divisible by 2^(n + 1) -/
theorem sqrt_three_pow_divisible (n : ℕ) :
  ∃ k : ℕ, (⌊(1 + Real.sqrt 3)^(2 * n + 1)⌋ : ℝ) = k * 2^(n + 1) :=
sorry

end NUMINAMATH_GPT_sqrt_three_pow_divisible_l1788_178817


namespace NUMINAMATH_GPT_present_cost_after_discount_l1788_178808

theorem present_cost_after_discount 
  (X : ℝ) (P : ℝ) 
  (h1 : X - 4 = (0.80 * P) / 3) 
  (h2 : P = 3 * X)
  :
  0.80 * P = 48 :=
by
  sorry

end NUMINAMATH_GPT_present_cost_after_discount_l1788_178808


namespace NUMINAMATH_GPT_num_boys_in_class_l1788_178886

-- Definitions based on conditions
def num_positions (p1 p2 : Nat) (total : Nat) : Nat :=
  if h : p1 < p2 then p2 - p1
  else total - (p1 - p2)

theorem num_boys_in_class (p1 p2 : Nat) (total : Nat) :
  p1 = 6 ∧ p2 = 16 ∧ num_positions p1 p2 total = 10 → total = 22 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_num_boys_in_class_l1788_178886


namespace NUMINAMATH_GPT_sum_a_b_c_l1788_178856

theorem sum_a_b_c (a b c : ℕ) (h : a = 5 ∧ b = 10 ∧ c = 14) : a + b + c = 29 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_c_l1788_178856


namespace NUMINAMATH_GPT_student_comprehensive_score_l1788_178879

def comprehensive_score (t_score i_score d_score : ℕ) (t_ratio i_ratio d_ratio : ℕ) :=
  (t_score * t_ratio + i_score * i_ratio + d_score * d_ratio) / (t_ratio + i_ratio + d_ratio)

theorem student_comprehensive_score :
  comprehensive_score 95 88 90 2 5 3 = 90 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_student_comprehensive_score_l1788_178879


namespace NUMINAMATH_GPT_production_increase_percentage_l1788_178859

variable (T : ℝ) -- Initial production
variable (T1 T2 T5 : ℝ) -- Productions at different years
variable (x : ℝ) -- Unknown percentage increase for last three years

-- Conditions
def condition1 : Prop := T1 = T * 1.06
def condition2 : Prop := T2 = T1 * 1.08
def condition3 : Prop := T5 = T * (1.1 ^ 5)

-- Statement to prove
theorem production_increase_percentage :
  condition1 T T1 →
  condition2 T1 T2 →
  (T5 = T2 * (1 + x / 100) ^ 3) →
  x = 12.1 :=
by
  sorry

end NUMINAMATH_GPT_production_increase_percentage_l1788_178859


namespace NUMINAMATH_GPT_tensor_value_l1788_178844

variables (h : ℝ)

def tensor (x y : ℝ) : ℝ := x^2 - y^2

theorem tensor_value : tensor h (tensor h h) = h^2 :=
by 
-- Complete proof body not required, 'sorry' is used for omitted proof
sorry

end NUMINAMATH_GPT_tensor_value_l1788_178844


namespace NUMINAMATH_GPT_third_candidate_votes_l1788_178818

theorem third_candidate_votes
  (total_votes : ℝ)
  (votes_for_two_candidates : ℝ)
  (winning_percentage : ℝ)
  (H1 : votes_for_two_candidates = 4636 + 11628)
  (H2 : winning_percentage = 67.21387283236994 / 100)
  (H3 : total_votes = votes_for_two_candidates / (1 - winning_percentage)) :
  (total_votes - votes_for_two_candidates) = 33336 :=
by
  sorry

end NUMINAMATH_GPT_third_candidate_votes_l1788_178818


namespace NUMINAMATH_GPT_bullet_speed_difference_l1788_178861

theorem bullet_speed_difference
  (horse_speed : ℝ := 20) 
  (bullet_speed : ℝ := 400) : 
  ((bullet_speed + horse_speed) - (bullet_speed - horse_speed) = 40) := by
  sorry

end NUMINAMATH_GPT_bullet_speed_difference_l1788_178861


namespace NUMINAMATH_GPT_max_cells_cut_diagonals_l1788_178806

theorem max_cells_cut_diagonals (board_size : ℕ) (k : ℕ) (internal_cells : ℕ) :
  board_size = 9 →
  internal_cells = (board_size - 2) ^ 2 →
  64 = internal_cells →
  V = internal_cells + k →
  E = 4 * k →
  k ≤ 21 :=
by
  sorry

end NUMINAMATH_GPT_max_cells_cut_diagonals_l1788_178806


namespace NUMINAMATH_GPT_nonoverlapping_unit_squares_in_figure_100_l1788_178840

theorem nonoverlapping_unit_squares_in_figure_100 :
  ∃ f : ℕ → ℕ, (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 15 ∧ f 3 = 27) ∧ f 100 = 20203 :=
by
  sorry

end NUMINAMATH_GPT_nonoverlapping_unit_squares_in_figure_100_l1788_178840


namespace NUMINAMATH_GPT_train_departure_at_10am_l1788_178843

noncomputable def train_departure_time (distance travel_rate : ℕ) (arrival_time_chicago : ℕ) (time_difference : ℤ) : ℕ :=
  let travel_time := distance / travel_rate
  let arrival_time_ny := arrival_time_chicago + 1
  arrival_time_ny - travel_time

theorem train_departure_at_10am :
  train_departure_time 480 60 17 1 = 10 :=
by
  -- implementation of the proof will go here
  -- but we skip the proof as per the instructions
  sorry

end NUMINAMATH_GPT_train_departure_at_10am_l1788_178843


namespace NUMINAMATH_GPT_major_axis_range_l1788_178812

theorem major_axis_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ x M N : ℝ, (x + (1 - x)) = 1 → x * (1 - x) = 0) 
  (e : ℝ) (h4 : (Real.sqrt 3 / 3) ≤ e ∧ e ≤ (Real.sqrt 2 / 2)) :
  ∃ a : ℝ, 2 * (Real.sqrt 5) ≤ 2 * a ∧ 2 * a ≤ 2 * (Real.sqrt 6) := 
sorry

end NUMINAMATH_GPT_major_axis_range_l1788_178812


namespace NUMINAMATH_GPT_mechanical_pencils_and_pens_price_l1788_178866

theorem mechanical_pencils_and_pens_price
    (x y : ℝ)
    (h₁ : 7 * x + 6 * y = 46.8)
    (h₂ : 3 * x + 5 * y = 32.2) :
  x = 2.4 ∧ y = 5 :=
sorry

end NUMINAMATH_GPT_mechanical_pencils_and_pens_price_l1788_178866


namespace NUMINAMATH_GPT_m_1_sufficient_but_not_necessary_l1788_178821

def lines_parallel (m : ℝ) : Prop :=
  let l1_slope := -m
  let l2_slope := (2 - 3 * m) / m
  l1_slope = l2_slope

theorem m_1_sufficient_but_not_necessary (m : ℝ) (h₁ : lines_parallel m) : 
  (m = 1) → (∃ m': ℝ, lines_parallel m' ∧ m' ≠ 1) :=
sorry

end NUMINAMATH_GPT_m_1_sufficient_but_not_necessary_l1788_178821


namespace NUMINAMATH_GPT_quadratic_interlaced_roots_l1788_178895

theorem quadratic_interlaced_roots
  (p1 p2 q1 q2 : ℝ)
  (h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  ∃ (r1 r2 s1 s2 : ℝ),
    (r1^2 + p1 * r1 + q1 = 0) ∧
    (r2^2 + p1 * r2 + q1 = 0) ∧
    (s1^2 + p2 * s1 + q2 = 0) ∧
    (s2^2 + p2 * s2 + q2 = 0) ∧
    (r1 < s1 ∧ s1 < r2 ∨ s1 < r1 ∧ r1 < s2) :=
sorry

end NUMINAMATH_GPT_quadratic_interlaced_roots_l1788_178895


namespace NUMINAMATH_GPT_number_solution_l1788_178891

variable (a : ℝ) (x : ℝ)

theorem number_solution :
  (a^(-x) + 25^(-2*x) + 5^(-4*x) = 11) ∧ (x = 0.25) → a = 625 / 7890481 :=
by 
  sorry

end NUMINAMATH_GPT_number_solution_l1788_178891


namespace NUMINAMATH_GPT_ratio_of_refurb_to_new_tshirt_l1788_178810

def cost_of_new_tshirt : ℤ := 5
def cost_of_pants : ℤ := 4
def cost_of_skirt : ℤ := 6

-- Total income from selling two new T-shirts, one pair of pants, four skirts, and six refurbished T-shirts is $53.
def total_income : ℤ := 53

-- Total income from selling new items.
def income_from_new_items : ℤ :=
  2 * cost_of_new_tshirt + cost_of_pants + 4 * cost_of_skirt

-- Income from refurbished T-shirts.
def income_from_refurb_tshirts : ℤ :=
  total_income - income_from_new_items

-- Number of refurbished T-shirts sold.
def num_refurb_tshirts_sold : ℤ := 6

-- Price of one refurbished T-shirt.
def cost_of_refurb_tshirt : ℤ :=
  income_from_refurb_tshirts / num_refurb_tshirts_sold

-- Prove the ratio of the price of a refurbished T-shirt to a new T-shirt is 0.5
theorem ratio_of_refurb_to_new_tshirt :
  (cost_of_refurb_tshirt : ℚ) / cost_of_new_tshirt = 0.5 := 
sorry

end NUMINAMATH_GPT_ratio_of_refurb_to_new_tshirt_l1788_178810


namespace NUMINAMATH_GPT_total_students_l1788_178899

theorem total_students (T : ℝ) 
  (h1 : 0.28 * T = 280) : 
  T = 1000 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_students_l1788_178899


namespace NUMINAMATH_GPT_correct_operations_result_l1788_178858

theorem correct_operations_result {n : ℕ} (h₁ : n / 8 - 20 = 12) :
  (n * 8 + 20) = 2068 ∧ 1800 < 2068 ∧ 2068 < 2200 :=
by
  sorry

end NUMINAMATH_GPT_correct_operations_result_l1788_178858


namespace NUMINAMATH_GPT_race_track_radius_l1788_178842

theorem race_track_radius (C_inner : ℝ) (width : ℝ) (r_outer : ℝ) : 
  C_inner = 440 ∧ width = 14 ∧ r_outer = (440 / (2 * Real.pi) + 14) → r_outer = 84 :=
by
  intros
  sorry

end NUMINAMATH_GPT_race_track_radius_l1788_178842


namespace NUMINAMATH_GPT_horizontal_distance_l1788_178878

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Condition: y-coordinate of point P is 8
def P_y : ℝ := 8

-- Condition: y-coordinate of point Q is -8
def Q_y : ℝ := -8

-- x-coordinates of points P and Q solve these equations respectively
def P_satisfies (x : ℝ) : Prop := curve x = P_y
def Q_satisfies (x : ℝ) : Prop := curve x = Q_y

-- The horizontal distance between P and Q is 1
theorem horizontal_distance : ∃ (Px Qx : ℝ), P_satisfies Px ∧ Q_satisfies Qx ∧ |Px - Qx| = 1 :=
by
  sorry

end NUMINAMATH_GPT_horizontal_distance_l1788_178878


namespace NUMINAMATH_GPT_solve_quadratic_l1788_178833

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 5 * x ^ 2 + 9 * x - 18 = 0) : x = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1788_178833


namespace NUMINAMATH_GPT_number_of_women_attended_l1788_178876

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end NUMINAMATH_GPT_number_of_women_attended_l1788_178876


namespace NUMINAMATH_GPT_geometric_series_first_term_l1788_178849

noncomputable def first_term_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  S = a / (1 - r)

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (hr : r = 1/6)
  (hS : S = 54) :
  first_term_geometric_series r S a →
  a = 45 :=
by
  intros h
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1788_178849


namespace NUMINAMATH_GPT_periodic_sequences_zero_at_two_l1788_178829

variable {R : Type*} [AddGroup R]

def seq_a (a b : ℕ → R) (n : ℕ) : Prop := a (n + 1) = a n + b n
def seq_b (b c : ℕ → R) (n : ℕ) : Prop := b (n + 1) = b n + c n
def seq_c (c d : ℕ → R) (n : ℕ) : Prop := c (n + 1) = c n + d n
def seq_d (d a : ℕ → R) (n : ℕ) : Prop := d (n + 1) = d n + a n

theorem periodic_sequences_zero_at_two
  (a b c d : ℕ → R)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (ha : ∀ n, seq_a a b n)
  (hb : ∀ n, seq_b b c n)
  (hc : ∀ n, seq_c c d n)
  (hd : ∀ n, seq_d d a n)
  (kra : a (k + m) = a m)
  (krb : b (k + m) = b m)
  (krc : c (k + m) = c m)
  (krd : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := sorry

end NUMINAMATH_GPT_periodic_sequences_zero_at_two_l1788_178829


namespace NUMINAMATH_GPT_five_x_plus_four_is_25_over_7_l1788_178860

theorem five_x_plus_four_is_25_over_7 (x : ℚ) (h : 5 * x - 8 = 12 * x + 15) : 5 * (x + 4) = 25 / 7 := by
  sorry

end NUMINAMATH_GPT_five_x_plus_four_is_25_over_7_l1788_178860


namespace NUMINAMATH_GPT_range_of_function_x_l1788_178834

theorem range_of_function_x (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := sorry

end NUMINAMATH_GPT_range_of_function_x_l1788_178834


namespace NUMINAMATH_GPT_find_number_l1788_178897

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1788_178897


namespace NUMINAMATH_GPT_setC_not_basis_l1788_178822

-- Definitions based on the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e₁ e₂ : V)
variables (v₁ v₂ : V)

-- Assuming e₁ and e₂ are non-collinear
axiom non_collinear : ¬Collinear ℝ {e₁, e₂}

-- The vectors in the set C
def setC_v1 : V := 3 • e₁ - 2 • e₂
def setC_v2 : V := 4 • e₂ - 6 • e₁

-- The proof problem statement
theorem setC_not_basis : Collinear ℝ {setC_v1 e₁ e₂, setC_v2 e₁ e₂} :=
sorry

end NUMINAMATH_GPT_setC_not_basis_l1788_178822


namespace NUMINAMATH_GPT_total_snakes_count_l1788_178807

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end NUMINAMATH_GPT_total_snakes_count_l1788_178807


namespace NUMINAMATH_GPT_laptop_price_difference_l1788_178888

theorem laptop_price_difference :
  let list_price := 59.99
  let tech_bargains_discount := 15
  let budget_bytes_discount_percentage := 0.30
  let tech_bargains_price := list_price - tech_bargains_discount
  let budget_bytes_price := list_price * (1 - budget_bytes_discount_percentage)
  let cheaper_price := min tech_bargains_price budget_bytes_price
  let expensive_price := max tech_bargains_price budget_bytes_price
  (expensive_price - cheaper_price) * 100 = 300 :=
by
  sorry

end NUMINAMATH_GPT_laptop_price_difference_l1788_178888


namespace NUMINAMATH_GPT_length_of_room_l1788_178830

theorem length_of_room {L : ℝ} (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : width = 4)
  (h2 : cost_per_sqm = 750)
  (h3 : total_cost = 16500) :
  L = 5.5 ↔ (L * width) * cost_per_sqm = total_cost := 
by
  sorry

end NUMINAMATH_GPT_length_of_room_l1788_178830


namespace NUMINAMATH_GPT_greatest_integer_third_side_l1788_178848

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_third_side_l1788_178848


namespace NUMINAMATH_GPT_sam_money_left_l1788_178837

-- Assuming the cost per dime and quarter
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Given conditions
def dimes : ℕ := 19
def quarters : ℕ := 6
def cost_per_candy_bar_in_dimes : ℕ := 3
def candy_bars : ℕ := 4
def lollipops : ℕ := 1

-- Calculate the initial money in cents
def initial_money : ℕ := (dimes * dime_value) + (quarters * quarter_value)

-- Calculate the cost of candy bars in cents
def candy_bars_cost : ℕ := candy_bars * cost_per_candy_bar_in_dimes * dime_value

-- Calculate the cost of lollipops in cents
def lollipop_cost : ℕ := lollipops * quarter_value

-- Calculate the total cost of purchases in cents
def total_cost : ℕ := candy_bars_cost + lollipop_cost

-- Calculate the final money left in cents
def final_money : ℕ := initial_money - total_cost

-- Theorem to prove
theorem sam_money_left : final_money = 195 := by
  sorry

end NUMINAMATH_GPT_sam_money_left_l1788_178837


namespace NUMINAMATH_GPT_sum_over_positive_reals_nonnegative_l1788_178865

theorem sum_over_positive_reals_nonnegative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (b + c - 2 * a) / (a^2 + b * c) + 
  (c + a - 2 * b) / (b^2 + c * a) + 
  (a + b - 2 * c) / (c^2 + a * b) ≥ 0 :=
sorry

end NUMINAMATH_GPT_sum_over_positive_reals_nonnegative_l1788_178865


namespace NUMINAMATH_GPT_ratio_ba_in_range_l1788_178889

theorem ratio_ba_in_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h1 : a + 2 * b = 7) (h2 : a^2 + b^2 ≤ 25) : 
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ 4 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_ba_in_range_l1788_178889


namespace NUMINAMATH_GPT_prob_red_or_blue_l1788_178874

open Nat

noncomputable def total_marbles : Nat := 90
noncomputable def prob_white : (ℚ) := 1 / 6
noncomputable def prob_green : (ℚ) := 1 / 5

theorem prob_red_or_blue :
  let prob_total := 1
  let prob_white_or_green := prob_white + prob_green
  let prob_red_blue := prob_total - prob_white_or_green
  prob_red_blue = 19 / 30 := by
    sorry

end NUMINAMATH_GPT_prob_red_or_blue_l1788_178874


namespace NUMINAMATH_GPT_largest_value_of_x_l1788_178855

theorem largest_value_of_x : 
  ∃ x, ( (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ) ∧ x = (19 + Real.sqrt 229) / 22 :=
sorry

end NUMINAMATH_GPT_largest_value_of_x_l1788_178855


namespace NUMINAMATH_GPT_transmitted_word_is_PAROHOD_l1788_178835

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end NUMINAMATH_GPT_transmitted_word_is_PAROHOD_l1788_178835


namespace NUMINAMATH_GPT_matrix_problem_l1788_178880

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![6, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 8], ![3, -5]]
def RHS : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![15, -3]]

theorem matrix_problem : 
  2 • A + B = RHS :=
by
  sorry

end NUMINAMATH_GPT_matrix_problem_l1788_178880


namespace NUMINAMATH_GPT_work_completion_times_l1788_178850

-- Definitions based on conditions
def condition1 (x y : ℝ) : Prop := 2 * (1 / x) + 5 * (1 / y) = 1 / 2
def condition2 (x y : ℝ) : Prop := 3 * (1 / x + 1 / y) = 0.45

-- Main theorem stating the solution
theorem work_completion_times :
  ∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ x = 12 ∧ y = 15 := 
sorry

end NUMINAMATH_GPT_work_completion_times_l1788_178850


namespace NUMINAMATH_GPT_extra_yellow_balls_dispatched_l1788_178831

theorem extra_yellow_balls_dispatched : 
  ∀ (W Y E : ℕ), -- Declare natural numbers W, Y, E
  W = Y →      -- Condition that the number of white balls equals the number of yellow balls
  W + Y = 64 → -- Condition that the total number of originally ordered balls is 64
  W / (Y + E) = 8 / 13 → -- The given ratio involving the extra yellow balls
  E = 20 :=               -- Prove that the extra yellow balls E equals 20
by
  intros W Y E h1 h2 h3
  -- Proof mechanism here
  sorry

end NUMINAMATH_GPT_extra_yellow_balls_dispatched_l1788_178831


namespace NUMINAMATH_GPT_associate_professors_bring_2_pencils_l1788_178816

theorem associate_professors_bring_2_pencils (A B P : ℕ) 
  (h1 : A + B = 5)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 5)
  : P = 2 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_associate_professors_bring_2_pencils_l1788_178816


namespace NUMINAMATH_GPT_prize_behind_door_4_eq_a_l1788_178885

theorem prize_behind_door_4_eq_a :
  ∀ (prize : ℕ → ℕ)
    (h_prizes : ∀ i j, 1 ≤ prize i ∧ prize i ≤ 4 ∧ prize i = prize j → i = j)
    (hA1 : prize 1 = 2)
    (hA2 : prize 3 = 3)
    (hB1 : prize 2 = 2)
    (hB2 : prize 3 = 4)
    (hC1 : prize 4 = 2)
    (hC2 : prize 2 = 3)
    (hD1 : prize 4 = 1)
    (hD2 : prize 3 = 3),
    prize 4 = 1 :=
by
  intro prize h_prizes hA1 hA2 hB1 hB2 hC1 hC2 hD1 hD2
  sorry

end NUMINAMATH_GPT_prize_behind_door_4_eq_a_l1788_178885


namespace NUMINAMATH_GPT_q_range_l1788_178881

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  ∀ y : ℝ, y ∈ Set.range q ↔ 0 ≤ y :=
by sorry

end NUMINAMATH_GPT_q_range_l1788_178881


namespace NUMINAMATH_GPT_top_card_is_joker_probability_l1788_178893

theorem top_card_is_joker_probability :
  let totalCards := 54
  let jokerCards := 2
  let probability := (jokerCards : ℚ) / (totalCards : ℚ)
  probability = 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_top_card_is_joker_probability_l1788_178893
