import Mathlib

namespace NUMINAMATH_GPT_simplify_log_expression_l1034_103403

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end NUMINAMATH_GPT_simplify_log_expression_l1034_103403


namespace NUMINAMATH_GPT_alissa_total_amount_spent_correct_l1034_103492
-- Import necessary Lean library

-- Define the costs of individual items
def football_cost : ℝ := 8.25
def marbles_cost : ℝ := 6.59
def puzzle_cost : ℝ := 12.10
def action_figure_cost : ℝ := 15.29
def board_game_cost : ℝ := 23.47

-- Define the discount rate and the sales tax rate
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.06

-- Define the total cost before discount
def total_cost_before_discount : ℝ :=
  football_cost + marbles_cost + puzzle_cost + action_figure_cost + board_game_cost

-- Define the discount amount
def discount : ℝ := total_cost_before_discount * discount_rate

-- Define the total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount

-- Define the sales tax amount
def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_after_discount + sales_tax

-- Prove that the total amount spent is $62.68
theorem alissa_total_amount_spent_correct : total_amount_spent = 62.68 := 
  by 
    sorry

end NUMINAMATH_GPT_alissa_total_amount_spent_correct_l1034_103492


namespace NUMINAMATH_GPT_find_number_l1034_103410

theorem find_number (x : ℕ) (h : 24 * x = 2376) : x = 99 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1034_103410


namespace NUMINAMATH_GPT_fraction_of_quarters_1840_1849_equals_4_over_15_l1034_103466

noncomputable def fraction_of_states_from_1840s (total_states : ℕ) (states_from_1840s : ℕ) : ℚ := 
  states_from_1840s / total_states

theorem fraction_of_quarters_1840_1849_equals_4_over_15 :
  fraction_of_states_from_1840s 30 8 = 4 / 15 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_quarters_1840_1849_equals_4_over_15_l1034_103466


namespace NUMINAMATH_GPT_kenya_peanuts_eq_133_l1034_103436

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end NUMINAMATH_GPT_kenya_peanuts_eq_133_l1034_103436


namespace NUMINAMATH_GPT_solve_system_l1034_103450

open Real

theorem solve_system :
  (∃ x y : ℝ, (sin x) ^ 2 + (cos y) ^ 2 = y ^ 4 ∧ (sin y) ^ 2 + (cos x) ^ 2 = x ^ 2) → 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) := by
  sorry

end NUMINAMATH_GPT_solve_system_l1034_103450


namespace NUMINAMATH_GPT_larger_group_men_count_l1034_103496

-- Define the conditions
def total_man_days (men : ℕ) (days : ℕ) : ℕ := men * days

-- Define the total work for 36 men in 18 days
def work_by_36_men_in_18_days : ℕ := total_man_days 36 18

-- Define the number of days the larger group takes
def days_for_larger_group : ℕ := 8

-- Problem Statement: Prove that if 36 men take 18 days to complete the work, and a larger group takes 8 days, then the larger group consists of 81 men.
theorem larger_group_men_count : 
  ∃ (M : ℕ), total_man_days M days_for_larger_group = work_by_36_men_in_18_days ∧ M = 81 := 
by
  -- Here would go the proof steps
  sorry

end NUMINAMATH_GPT_larger_group_men_count_l1034_103496


namespace NUMINAMATH_GPT_car_winning_probability_l1034_103454

noncomputable def probability_of_winning (P_X P_Y P_Z : ℚ) : ℚ :=
  P_X + P_Y + P_Z

theorem car_winning_probability :
  let P_X := (1 : ℚ) / 6
  let P_Y := (1 : ℚ) / 10
  let P_Z := (1 : ℚ) / 8
  probability_of_winning P_X P_Y P_Z = 47 / 120 :=
by
  sorry

end NUMINAMATH_GPT_car_winning_probability_l1034_103454


namespace NUMINAMATH_GPT_find_a_value_l1034_103480

theorem find_a_value 
  (A : Set ℝ := {x | x^2 - 4 ≤ 0})
  (B : Set ℝ := {x | 2 * x + a ≤ 0})
  (intersection : A ∩ B = {x | -2 ≤ x ∧ x ≤ 1}) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l1034_103480


namespace NUMINAMATH_GPT_number_of_liars_l1034_103435

/-- Definition of conditions -/
def total_islands : Nat := 17
def population_per_island : Nat := 119

-- Conditions based on the problem description
def islands_yes_first_question : Nat := 7
def islands_no_first_question : Nat := total_islands - islands_yes_first_question

def islands_no_second_question : Nat := 7
def islands_yes_second_question : Nat := total_islands - islands_no_second_question

def minimum_knights_for_no_second_question : Nat := 60  -- At least 60 knights

/-- Main theorem -/
theorem number_of_liars : 
  ∃ x y: Nat, 
    (x + (islands_no_first_question - y) = islands_yes_first_question ∧ 
     y - x = 3 ∧ 
     60 * x + 59 * y + 119 * (islands_no_first_question - y) = 1010 ∧
     (total_islands * population_per_island - 1010 = 1013)) := by
  sorry

end NUMINAMATH_GPT_number_of_liars_l1034_103435


namespace NUMINAMATH_GPT_coins_left_zero_when_divided_by_9_l1034_103468

noncomputable def smallestCoinCount (n: ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_left_zero_when_divided_by_9 (n : ℕ) (h : smallestCoinCount n) (h_min: ∀ m : ℕ, smallestCoinCount m → n ≤ m) :
  n % 9 = 0 :=
sorry

end NUMINAMATH_GPT_coins_left_zero_when_divided_by_9_l1034_103468


namespace NUMINAMATH_GPT_trees_in_garden_l1034_103424

theorem trees_in_garden (yard_length distance_between_trees : ℕ) (h1 : yard_length = 800) (h2 : distance_between_trees = 32) :
  ∃ n : ℕ, n = (yard_length / distance_between_trees) + 1 ∧ n = 26 :=
by
  sorry

end NUMINAMATH_GPT_trees_in_garden_l1034_103424


namespace NUMINAMATH_GPT_find_rectangle_width_l1034_103472

variable (length_square : ℕ) (length_rectangle : ℕ) (width_rectangle : ℕ)

-- Given conditions
def square_side_length := 700
def rectangle_length := 400
def square_perimeter := 4 * square_side_length
def rectangle_perimeter := square_perimeter / 2
def rectangle_perimeter_eq := 2 * length_rectangle + 2 * width_rectangle

-- Statement to prove
theorem find_rectangle_width :
  (square_perimeter = 2800) →
  (rectangle_perimeter = 1400) →
  (length_rectangle = 400) →
  (rectangle_perimeter_eq = 1400) →
  (width_rectangle = 300) :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_rectangle_width_l1034_103472


namespace NUMINAMATH_GPT_purely_imaginary_number_eq_l1034_103434

theorem purely_imaginary_number_eq (z : ℂ) (a : ℝ) (i : ℂ) (h_imag : z.im = 0 ∧ z = 0 ∧ (3 - i) * z = a + i + i) :
  a = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_purely_imaginary_number_eq_l1034_103434


namespace NUMINAMATH_GPT_number_of_cows_on_farm_l1034_103489

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end NUMINAMATH_GPT_number_of_cows_on_farm_l1034_103489


namespace NUMINAMATH_GPT_find_perpendicular_line_l1034_103444

theorem find_perpendicular_line (x y : ℝ) (h₁ : y = (1/2) * x + 1)
    (h₂ : (x, y) = (2, 0)) : y = -2 * x + 4 :=
sorry

end NUMINAMATH_GPT_find_perpendicular_line_l1034_103444


namespace NUMINAMATH_GPT_arithmetic_expression_l1034_103413

theorem arithmetic_expression :
  (30 / (10 + 2 - 5) + 4) * 7 = 58 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l1034_103413


namespace NUMINAMATH_GPT_chromium_percentage_new_alloy_l1034_103482

-- Conditions as definitions
def first_alloy_chromium_percentage : ℝ := 12
def second_alloy_chromium_percentage : ℝ := 8
def first_alloy_weight : ℝ := 10
def second_alloy_weight : ℝ := 30

-- Final proof statement
theorem chromium_percentage_new_alloy : 
  ((first_alloy_chromium_percentage / 100 * first_alloy_weight +
    second_alloy_chromium_percentage / 100 * second_alloy_weight) /
  (first_alloy_weight + second_alloy_weight)) * 100 = 9 :=
by
  sorry

end NUMINAMATH_GPT_chromium_percentage_new_alloy_l1034_103482


namespace NUMINAMATH_GPT_racers_in_final_segment_l1034_103451

def initial_racers := 200

def racers_after_segment_1 (initial: ℕ) := initial - 10
def racers_after_segment_2 (after_segment_1: ℕ) := after_segment_1 - after_segment_1 / 3
def racers_after_segment_3 (after_segment_2: ℕ) := after_segment_2 - after_segment_2 / 4
def racers_after_segment_4 (after_segment_3: ℕ) := after_segment_3 - after_segment_3 / 3
def racers_after_segment_5 (after_segment_4: ℕ) := after_segment_4 - after_segment_4 / 2
def racers_after_segment_6 (after_segment_5: ℕ) := after_segment_5 - (3 * after_segment_5 / 4)

theorem racers_in_final_segment : racers_after_segment_6 (racers_after_segment_5 (racers_after_segment_4 (racers_after_segment_3 (racers_after_segment_2 (racers_after_segment_1 initial_racers))))) = 8 :=
  by
  sorry

end NUMINAMATH_GPT_racers_in_final_segment_l1034_103451


namespace NUMINAMATH_GPT_mean_equal_l1034_103493

theorem mean_equal (y : ℚ) :
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := 
by
  sorry

end NUMINAMATH_GPT_mean_equal_l1034_103493


namespace NUMINAMATH_GPT_total_nephews_l1034_103419

noncomputable def Alden_past_nephews : ℕ := 50
noncomputable def Alden_current_nephews : ℕ := 2 * Alden_past_nephews
noncomputable def Vihaan_current_nephews : ℕ := Alden_current_nephews + 60

theorem total_nephews :
  Alden_current_nephews + Vihaan_current_nephews = 260 := 
by
  sorry

end NUMINAMATH_GPT_total_nephews_l1034_103419


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1034_103461

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymptotes : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  (c / a = 5 / 4) ∨ (c / a = 5 / 3) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1034_103461


namespace NUMINAMATH_GPT_jane_buys_bagels_l1034_103469

variable (b m : ℕ)
variable (h1 : b + m = 7)
variable (h2 : 65 * b + 40 * m % 100 = 80)
variable (h3 : 40 * b + 40 * m % 100 = 0)

theorem jane_buys_bagels : b = 4 := by sorry

end NUMINAMATH_GPT_jane_buys_bagels_l1034_103469


namespace NUMINAMATH_GPT_find_constants_l1034_103460

theorem find_constants (P Q R : ℚ) 
  (h : ∀ x : ℚ, x ≠ 4 → x ≠ 2 → (5 * x + 1) / ((x - 4) * (x - 2) ^ 2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) :
  P = 21 / 4 ∧ Q = 15 ∧ R = -11 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1034_103460


namespace NUMINAMATH_GPT_length_of_ae_l1034_103462

def consecutive_points_on_line (a b c d e : ℝ) : Prop :=
  ∃ (ab bc cd de : ℝ), 
  ab = 5 ∧ 
  bc = 2 * cd ∧ 
  de = 4 ∧ 
  a + ab = b ∧ 
  b + bc = c ∧ 
  c + cd = d ∧ 
  d + de = e ∧
  a + ab + bc = c -- ensuring ac = 11

theorem length_of_ae (a b c d e : ℝ) 
  (h1 : consecutive_points_on_line a b c d e) 
  (h2 : a + 5 = b)
  (h3 : b + 2 * (c - b) = c)
  (h4 : d - c = 3)
  (h5 : d + 4 = e)
  (h6 : a + 5 + 2 * (c - b) = c) :
  e - a = 18 :=
sorry

end NUMINAMATH_GPT_length_of_ae_l1034_103462


namespace NUMINAMATH_GPT_parabola_focus_l1034_103453

theorem parabola_focus (x y : ℝ) :
  (∃ x, y = 4 * x^2 + 8 * x - 5) →
  (x, y) = (-1, -8.9375) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1034_103453


namespace NUMINAMATH_GPT_quadrilateral_area_l1034_103426

def vertex1 : ℝ × ℝ := (2, 1)
def vertex2 : ℝ × ℝ := (4, 3)
def vertex3 : ℝ × ℝ := (7, 1)
def vertex4 : ℝ × ℝ := (4, 6)

noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) -
       (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)) / 2

theorem quadrilateral_area :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1034_103426


namespace NUMINAMATH_GPT_unique_providers_count_l1034_103402

theorem unique_providers_count :
  let num_children := 4
  let num_providers := 25
  (∀ s : Fin num_children, s.val < num_providers)
  → num_providers * (num_providers - 1) * (num_providers - 2) * (num_providers - 3) = 303600
:= sorry

end NUMINAMATH_GPT_unique_providers_count_l1034_103402


namespace NUMINAMATH_GPT_range_of_x_l1034_103446

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_x (x : ℝ) (h₀ : -1 < x ∧ x < 1) (h₁ : f 0 = 0) (h₂ : f (1 - x) + f (1 - x^2) < 0) :
  1 < x ∧ x < Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1034_103446


namespace NUMINAMATH_GPT_compute_seventy_five_squared_minus_thirty_five_squared_l1034_103447

theorem compute_seventy_five_squared_minus_thirty_five_squared :
  75^2 - 35^2 = 4400 := by
  sorry

end NUMINAMATH_GPT_compute_seventy_five_squared_minus_thirty_five_squared_l1034_103447


namespace NUMINAMATH_GPT_current_price_of_soda_l1034_103421

theorem current_price_of_soda (C S : ℝ) (h1 : 1.25 * C = 15) (h2 : C + S = 16) : 1.5 * S = 6 :=
by
  sorry

end NUMINAMATH_GPT_current_price_of_soda_l1034_103421


namespace NUMINAMATH_GPT_distance_from_dormitory_to_city_l1034_103464

theorem distance_from_dormitory_to_city (D : ℝ)
  (h1 : (1 / 5) * D + (2 / 3) * D + 4 = D) : D = 30 := by
  sorry

end NUMINAMATH_GPT_distance_from_dormitory_to_city_l1034_103464


namespace NUMINAMATH_GPT_sum_of_percentages_l1034_103456

theorem sum_of_percentages : (20 / 100 : ℝ) * 40 + (25 / 100 : ℝ) * 60 = 23 := 
by 
  -- Sorry skips the proof
  sorry

end NUMINAMATH_GPT_sum_of_percentages_l1034_103456


namespace NUMINAMATH_GPT_proof_ratio_QP_over_EF_l1034_103465

noncomputable def rectangle_theorem : Prop :=
  ∃ (A B C D E F G P Q : ℝ × ℝ),
    -- Coordinates of the rectangle vertices
    A = (0, 4) ∧ B = (5, 4) ∧ C = (5, 0) ∧ D = (0, 0) ∧
    -- Coordinates of points E, F, and G on the sides of the rectangle
    E = (4, 4) ∧ F = (2, 0) ∧ G = (5, 1) ∧
    -- Coordinates of intersection points P and Q
    P = (20 / 7, 12 / 7) ∧ Q = (40 / 13, 28 / 13) ∧
    -- Ratio of distances PQ and EF
    (dist P Q)/(dist E F) = 10 / 91

theorem proof_ratio_QP_over_EF : rectangle_theorem :=
sorry

end NUMINAMATH_GPT_proof_ratio_QP_over_EF_l1034_103465


namespace NUMINAMATH_GPT_product_closest_value_l1034_103486

theorem product_closest_value (a b : ℝ) (ha : a = 0.000321) (hb : b = 7912000) :
  abs ((a * b) - 2523) < min (abs ((a * b) - 2500)) (min (abs ((a * b) - 2700)) (min (abs ((a * b) - 3100)) (abs ((a * b) - 2000)))) := by
  sorry

end NUMINAMATH_GPT_product_closest_value_l1034_103486


namespace NUMINAMATH_GPT_brick_width_l1034_103478

theorem brick_width (L W : ℕ) (l : ℕ) (b : ℕ) (n : ℕ) (A B : ℕ) 
    (courtyard_area_eq : A = L * W * 10000)
    (brick_area_eq : B = l * b)
    (total_bricks_eq : A = n * B)
    (courtyard_dims : L = 30 ∧ W = 16)
    (brick_len : l = 20)
    (num_bricks : n = 24000) :
    b = 10 := by
  sorry

end NUMINAMATH_GPT_brick_width_l1034_103478


namespace NUMINAMATH_GPT_work_days_together_l1034_103459

theorem work_days_together (A B : Type) (R_A R_B : ℝ) 
  (h1 : R_A = 1/2 * R_B) (h2 : R_B = 1 / 27) : 
  (1 / (R_A + R_B)) = 18 :=
by
  sorry

end NUMINAMATH_GPT_work_days_together_l1034_103459


namespace NUMINAMATH_GPT_range_of_m_empty_solution_set_inequality_l1034_103443

theorem range_of_m_empty_solution_set_inequality (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 ≥ 0 → false) ↔ -4 < m ∧ m < 0 := 
sorry

end NUMINAMATH_GPT_range_of_m_empty_solution_set_inequality_l1034_103443


namespace NUMINAMATH_GPT_polygon_sides_l1034_103471

-- Define the given conditions
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def sum_exterior_angles : ℕ := 360

-- Define the theorem
theorem polygon_sides (n : ℕ) (h : sum_interior_angles n = 3 * sum_exterior_angles + 180) : n = 9 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1034_103471


namespace NUMINAMATH_GPT_range_of_a_l1034_103498

noncomputable def range_of_a_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, |x + 1| + |x - a| ≤ 2

theorem range_of_a : ∀ a : ℝ, range_of_a_condition a → (-3 : ℝ) ≤ a ∧ a ≤ 1 :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_range_of_a_l1034_103498


namespace NUMINAMATH_GPT_sum_of_roots_l1034_103437

theorem sum_of_roots (z1 z2 : ℂ) (h : z1^2 + 5*z1 - 14 = 0 ∧ z2^2 + 5*z2 - 14 = 0) :
  z1 + z2 = -5 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1034_103437


namespace NUMINAMATH_GPT_incorrect_transformation_D_l1034_103499

theorem incorrect_transformation_D (x : ℝ) (hx1 : x + 1 ≠ 0) : 
  (2 - x) / (x + 1) ≠ (x - 2) / (1 + x) := 
by 
  sorry

end NUMINAMATH_GPT_incorrect_transformation_D_l1034_103499


namespace NUMINAMATH_GPT_green_to_blue_ratio_l1034_103483

-- Definition of the problem conditions
variable (G B R : ℕ)
variable (H1 : 2 * G = R)
variable (H2 : B = 80)
variable (H3 : R = 1280)

-- Theorem statement: the ratio of the green car's speed to the blue car's speed is 8:1
theorem green_to_blue_ratio (G B R : ℕ) (H1 : 2 * G = R) (H2 : B = 80) (H3 : R = 1280) :
  G / B = 8 :=
by
  sorry

end NUMINAMATH_GPT_green_to_blue_ratio_l1034_103483


namespace NUMINAMATH_GPT_maximum_abc_value_l1034_103418

theorem maximum_abc_value:
  (∀ (a b c : ℝ), (0 < a ∧ a < 3) ∧ (0 < b ∧ b < 3) ∧ (0 < c ∧ c < 3) ∧ (∀ x : ℝ, (x^4 + a * x^3 + b * x^2 + c * x + 1) ≠ 0) → (abc ≤ 18.75)) :=
sorry

end NUMINAMATH_GPT_maximum_abc_value_l1034_103418


namespace NUMINAMATH_GPT_infections_first_wave_l1034_103442

theorem infections_first_wave (x : ℕ)
  (h1 : 4 * x * 14 = 21000) : x = 375 :=
  sorry

end NUMINAMATH_GPT_infections_first_wave_l1034_103442


namespace NUMINAMATH_GPT_exponent_identity_l1034_103455

variable (x : ℝ) (m n : ℝ)
axiom h1 : x^m = 6
axiom h2 : x^n = 9

theorem exponent_identity : x^(2 * m - n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_exponent_identity_l1034_103455


namespace NUMINAMATH_GPT_inequality_holds_for_triangle_sides_l1034_103497

theorem inequality_holds_for_triangle_sides (a : ℝ) : 
  (∀ (x y z : ℕ), x + y > z ∧ y + z > x ∧ z + x > y → (x^2 + y^2 + z^2 ≤ a * (x * y + y * z + z * x))) ↔ (1 ≤ a ∧ a ≤ 6 / 5) :=
by sorry

end NUMINAMATH_GPT_inequality_holds_for_triangle_sides_l1034_103497


namespace NUMINAMATH_GPT_motorist_routes_birmingham_to_sheffield_l1034_103405

-- Definitions for the conditions
def routes_bristol_to_birmingham : ℕ := 6
def routes_sheffield_to_carlisle : ℕ := 2
def total_routes_bristol_to_carlisle : ℕ := 36

-- The proposition that should be proven
theorem motorist_routes_birmingham_to_sheffield : 
  ∃ x : ℕ, routes_bristol_to_birmingham * x * routes_sheffield_to_carlisle = total_routes_bristol_to_carlisle ∧ x = 3 :=
sorry

end NUMINAMATH_GPT_motorist_routes_birmingham_to_sheffield_l1034_103405


namespace NUMINAMATH_GPT_equivalent_single_reduction_l1034_103439

theorem equivalent_single_reduction :
  ∀ (P : ℝ), P * (1 - 0.25) * (1 - 0.20) = P * (1 - 0.40) :=
by
  intros P
  -- Proof will be skipped
  sorry

end NUMINAMATH_GPT_equivalent_single_reduction_l1034_103439


namespace NUMINAMATH_GPT_bob_tiller_swath_width_l1034_103449

theorem bob_tiller_swath_width
  (plot_width plot_length : ℕ)
  (tilling_rate_seconds_per_foot : ℕ)
  (total_tilling_minutes : ℕ)
  (total_area : ℕ)
  (tilled_length : ℕ)
  (swath_width : ℕ) :
  plot_width = 110 →
  plot_length = 120 →
  tilling_rate_seconds_per_foot = 2 →
  total_tilling_minutes = 220 →
  total_area = plot_width * plot_length →
  tilled_length = (total_tilling_minutes * 60) / tilling_rate_seconds_per_foot →
  swath_width = total_area / tilled_length →
  swath_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_bob_tiller_swath_width_l1034_103449


namespace NUMINAMATH_GPT_max_area_of_triangle_ABC_l1034_103452

noncomputable def max_area_triangle_ABC: ℝ :=
  let QA := 3
  let QB := 4
  let QC := 5
  let BC := 6
  -- Given these conditions, prove the maximum area of triangle ABC
  19

theorem max_area_of_triangle_ABC 
  (QA QB QC BC : ℝ) 
  (h1 : QA = 3) 
  (h2 : QB = 4) 
  (h3 : QC = 5) 
  (h4 : BC = 6) 
  (h5 : QB * QB + BC * BC = QC * QC) -- The right angle condition at Q
  : max_area_triangle_ABC = 19 :=
by sorry

end NUMINAMATH_GPT_max_area_of_triangle_ABC_l1034_103452


namespace NUMINAMATH_GPT_travel_time_without_walking_l1034_103484

-- Definitions based on the problem's conditions
def walking_time_without_escalator (x y : ℝ) : Prop := 75 * x = y
def walking_time_with_escalator (x k y : ℝ) : Prop := 30 * (x + k) = y

-- Main theorem: Time taken to travel the distance with the escalator alone
theorem travel_time_without_walking (x k y : ℝ) (h1 : walking_time_without_escalator x y) (h2 : walking_time_with_escalator x k y) : y / k = 50 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_without_walking_l1034_103484


namespace NUMINAMATH_GPT_line_intersects_ellipse_l1034_103414

theorem line_intersects_ellipse
  (m : ℝ) :
  ∃ P : ℝ × ℝ, P = (3, 2) ∧ ((m + 2) * P.1 - (m + 4) * P.2 + 2 - m = 0) ∧ 
  (P.1^2 / 25 + P.2^2 / 9 < 1) :=
by 
  sorry

end NUMINAMATH_GPT_line_intersects_ellipse_l1034_103414


namespace NUMINAMATH_GPT_smallest_positive_multiple_l1034_103412

theorem smallest_positive_multiple (a : ℕ) :
  (37 * a) % 97 = 7 → 37 * a = 481 :=
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_l1034_103412


namespace NUMINAMATH_GPT_intersection_of_complements_l1034_103476

open Set

variable (U A B : Set Nat)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 4, 5})
variable (hB : B = {2, 4, 6, 8})

theorem intersection_of_complements :
  A ∩ (U \ B) = {3, 5} :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_intersection_of_complements_l1034_103476


namespace NUMINAMATH_GPT_total_distance_travelled_l1034_103445

theorem total_distance_travelled (D : ℝ) (h1 : (D / 2) / 30 + (D / 2) / 25 = 11) : D = 150 :=
sorry

end NUMINAMATH_GPT_total_distance_travelled_l1034_103445


namespace NUMINAMATH_GPT_pond_length_l1034_103494

theorem pond_length (V W D L : ℝ) (hV : V = 1600) (hW : W = 10) (hD : D = 8) :
  L = 20 ↔ V = L * W * D :=
by
  sorry

end NUMINAMATH_GPT_pond_length_l1034_103494


namespace NUMINAMATH_GPT_problem_statement_l1034_103416

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem problem_statement :
  let l := { p : ℝ × ℝ | p.1 - p.2 - 2 = 0 }
  let C := { p : ℝ × ℝ | ∃ θ : ℝ, p = (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ) }
  let A := (-4, -6)
  let B := (4, 2)
  let P := (-2 * Real.sqrt 3, 2)
  let d := (|2 * Real.sqrt 3 * Real.cos (5 * Real.pi / 6) - 2|) / Real.sqrt 2
  distance A B = 8 * Real.sqrt 2 ∧ d = 3 * Real.sqrt 2 ∧
  let max_area := 1 / 2 * 8 * Real.sqrt 2 * 3 * Real.sqrt 2
  P ∈ C ∧ max_area = 24 := by
sorry

end NUMINAMATH_GPT_problem_statement_l1034_103416


namespace NUMINAMATH_GPT_probability_of_selecting_green_ball_l1034_103408

-- Declare the probability of selecting each container
def prob_of_selecting_container := (1 : ℚ) / 4

-- Declare the number of balls in each container
def balls_in_container_A := 10
def balls_in_container_B := 14
def balls_in_container_C := 14
def balls_in_container_D := 10

-- Declare the number of green balls in each container
def green_balls_in_A := 6
def green_balls_in_B := 6
def green_balls_in_C := 6
def green_balls_in_D := 7

-- Calculate the probability of drawing a green ball from each container
def prob_green_from_A := (green_balls_in_A : ℚ) / balls_in_container_A
def prob_green_from_B := (green_balls_in_B : ℚ) / balls_in_container_B
def prob_green_from_C := (green_balls_in_C : ℚ) / balls_in_container_C
def prob_green_from_D := (green_balls_in_D : ℚ) / balls_in_container_D

-- Calculate the total probability of drawing a green ball
def total_prob_green :=
  prob_of_selecting_container * prob_green_from_A +
  prob_of_selecting_container * prob_green_from_B +
  prob_of_selecting_container * prob_green_from_C +
  prob_of_selecting_container * prob_green_from_D

theorem probability_of_selecting_green_ball : total_prob_green = 13 / 28 :=
by sorry

end NUMINAMATH_GPT_probability_of_selecting_green_ball_l1034_103408


namespace NUMINAMATH_GPT_inequality_solution_l1034_103430

theorem inequality_solution (x : ℚ) : (3 * x - 5 ≥ 9 - 2 * x) → (x ≥ 14 / 5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1034_103430


namespace NUMINAMATH_GPT_correct_union_l1034_103423

universe u

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2}
def C_I (A : Set ℕ) : Set ℕ := {x ∈ I | x ∉ A}

-- Theorem statement
theorem correct_union : B ∪ C_I A = {2, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_correct_union_l1034_103423


namespace NUMINAMATH_GPT_flower_shop_ratio_l1034_103415

theorem flower_shop_ratio (V C T R : ℕ) 
(total_flowers : V + C + T + R > 0)
(tulips_ratio : T = V / 4)
(roses_tulips_equal : R = T)
(carnations_fraction : C = 2 / 3 * (V + T + R + C)) 
: V / C = 1 / 3 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_flower_shop_ratio_l1034_103415


namespace NUMINAMATH_GPT_no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l1034_103417

theorem no_triangle_sum_of_any_two_angles_lt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) :=
by
  sorry

theorem no_triangle_sum_of_any_two_angles_gt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120) :=
by
  sorry

end NUMINAMATH_GPT_no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l1034_103417


namespace NUMINAMATH_GPT_expand_polynomial_correct_l1034_103433

open Polynomial

noncomputable def expand_polynomial : Polynomial ℤ :=
  (C 3 * X^3 - C 2 * X^2 + X - C 4) * (C 4 * X^2 - C 2 * X + C 5)

theorem expand_polynomial_correct :
  expand_polynomial = C 12 * X^5 - C 14 * X^4 + C 23 * X^3 - C 28 * X^2 + C 13 * X - C 20 :=
by sorry

end NUMINAMATH_GPT_expand_polynomial_correct_l1034_103433


namespace NUMINAMATH_GPT_ash_cloud_ratio_l1034_103485

theorem ash_cloud_ratio
  (distance_ashes_shot_up : ℕ)
  (radius_ash_cloud : ℕ)
  (h1 : distance_ashes_shot_up = 300)
  (h2 : radius_ash_cloud = 2700) :
  (2 * radius_ash_cloud) / distance_ashes_shot_up = 18 :=
by
  sorry

end NUMINAMATH_GPT_ash_cloud_ratio_l1034_103485


namespace NUMINAMATH_GPT_mean_noon_temperature_l1034_103487

def temperatures : List ℝ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_noon_temperature :
  (List.sum temperatures) / (temperatures.length) = 770 / 9 := by
  sorry

end NUMINAMATH_GPT_mean_noon_temperature_l1034_103487


namespace NUMINAMATH_GPT_sum_seven_terms_l1034_103457

-- Define the arithmetic sequence and sum of first n terms
variable {a : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S : ℕ → ℝ} -- The sum of the first n terms S_n

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition: a_4 = 4
def a_4_eq_4 (a : ℕ → ℝ) : Prop :=
  a 4 = 4

-- Proposition we want to prove: S_7 = 28 given a_4 = 4
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (ha : is_arithmetic_sequence a)
  (hS : sum_of_arithmetic_sequence a S)
  (h : a_4_eq_4 a) : 
  S 7 = 28 := 
sorry

end NUMINAMATH_GPT_sum_seven_terms_l1034_103457


namespace NUMINAMATH_GPT_geom_series_sum_l1034_103420

/-- The sum of the first six terms of the geometric series 
    with first term a = 1 and common ratio r = (1 / 4) is 1365 / 1024. -/
theorem geom_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1 / 4
  let n : ℕ := 6
  (a * (1 - r^n) / (1 - r)) = 1365 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_sum_l1034_103420


namespace NUMINAMATH_GPT_problem_statement_l1034_103407

-- Definitions and conditions
def f (x : ℝ) : ℝ := x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

-- Given the specific condition
def f_symmetric_about_1 : Prop := is_symmetric_about f 1

-- We need to prove that this implies g(x) = 3x - 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f_symmetric_about_1 → ∀ x, g x = 3 * x - 2 := 
by
  intro h
  sorry -- Detailed proof is omitted

end NUMINAMATH_GPT_problem_statement_l1034_103407


namespace NUMINAMATH_GPT_polynomial_is_first_degree_l1034_103425

theorem polynomial_is_first_degree (k m : ℝ) (h : (k - 1) = 0) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_is_first_degree_l1034_103425


namespace NUMINAMATH_GPT_bulb_works_longer_than_4000_hours_l1034_103463

noncomputable def P_X := 0.5
noncomputable def P_Y := 0.3
noncomputable def P_Z := 0.2

noncomputable def P_4000_given_X := 0.59
noncomputable def P_4000_given_Y := 0.65
noncomputable def P_4000_given_Z := 0.70

noncomputable def P_4000 := 
  P_X * P_4000_given_X + P_Y * P_4000_given_Y + P_Z * P_4000_given_Z

theorem bulb_works_longer_than_4000_hours : P_4000 = 0.63 :=
by
  sorry

end NUMINAMATH_GPT_bulb_works_longer_than_4000_hours_l1034_103463


namespace NUMINAMATH_GPT_amount_left_after_spending_l1034_103475

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end NUMINAMATH_GPT_amount_left_after_spending_l1034_103475


namespace NUMINAMATH_GPT_fourth_term_arithmetic_sequence_l1034_103495

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end NUMINAMATH_GPT_fourth_term_arithmetic_sequence_l1034_103495


namespace NUMINAMATH_GPT_difference_of_squares_l1034_103428

def a : ℕ := 601
def b : ℕ := 597

theorem difference_of_squares : a^2 - b^2 = 4792 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_of_squares_l1034_103428


namespace NUMINAMATH_GPT_sum_of_common_ratios_l1034_103422

variable (m x y : ℝ)
variable (h₁ : x ≠ y)
variable (h₂ : a2 = m * x)
variable (h₃ : a3 = m * x^2)
variable (h₄ : b2 = m * y)
variable (h₅ : b3 = m * y^2)
variable (h₆ : a3 - b3 = 3 * (a2 - b2))

theorem sum_of_common_ratios : x + y = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l1034_103422


namespace NUMINAMATH_GPT_functional_equation_solution_l1034_103440

theorem functional_equation_solution {f : ℝ → ℝ}
  (h : ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)) :
  (f = fun x => 0) ∨ (f = id) ∨ (f = fun x => -x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1034_103440


namespace NUMINAMATH_GPT_value_of_m_minus_n_l1034_103429

variables {a b : ℕ}
variables {m n : ℤ}

def are_like_terms (m n : ℤ) : Prop :=
  (m - 2 = 4) ∧ (n + 7 = 4)

theorem value_of_m_minus_n (h : are_like_terms m n) : m - n = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_minus_n_l1034_103429


namespace NUMINAMATH_GPT_trig_identity_l1034_103438

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1034_103438


namespace NUMINAMATH_GPT_product_of_integers_l1034_103404

theorem product_of_integers (a b : ℤ) (h_lcm : Int.lcm a b = 45) (h_gcd : Int.gcd a b = 9) : a * b = 405 :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_l1034_103404


namespace NUMINAMATH_GPT_James_vegetable_intake_in_third_week_l1034_103400

noncomputable def third_week_vegetable_intake : ℝ :=
  let asparagus_per_day_first_week : ℝ := 0.25
  let broccoli_per_day_first_week : ℝ := 0.25
  let cauliflower_per_day_first_week : ℝ := 0.5

  let asparagus_per_day_second_week := 2 * asparagus_per_day_first_week
  let broccoli_per_day_second_week := 3 * broccoli_per_day_first_week
  let cauliflower_per_day_second_week := cauliflower_per_day_first_week * 1.75
  let spinach_per_day_second_week : ℝ := 0.5
  
  let daily_intake_second_week := asparagus_per_day_second_week +
                                  broccoli_per_day_second_week +
                                  cauliflower_per_day_second_week +
                                  spinach_per_day_second_week
  
  let kale_per_day_third_week : ℝ := 0.5
  let zucchini_per_day_third_week : ℝ := 0.15
  
  let daily_intake_third_week := asparagus_per_day_second_week +
                                 broccoli_per_day_second_week +
                                 cauliflower_per_day_second_week +
                                 spinach_per_day_second_week +
                                 kale_per_day_third_week +
                                 zucchini_per_day_third_week
  
  daily_intake_third_week * 7

theorem James_vegetable_intake_in_third_week : 
  third_week_vegetable_intake = 22.925 :=
  by
    sorry

end NUMINAMATH_GPT_James_vegetable_intake_in_third_week_l1034_103400


namespace NUMINAMATH_GPT_exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l1034_103411

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end NUMINAMATH_GPT_exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l1034_103411


namespace NUMINAMATH_GPT_sum_of_squares_greater_than_cubics_l1034_103406

theorem sum_of_squares_greater_than_cubics (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a)
  : 
  (2 * (a + b + c) * (a^2 + b^2 + c^2)) / 3 > a^3 + b^3 + c^3 + a * b * c := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_greater_than_cubics_l1034_103406


namespace NUMINAMATH_GPT_cricket_problem_solved_l1034_103470

noncomputable def cricket_problem : Prop :=
  let run_rate_10 := 3.2
  let target := 252
  let required_rate := 5.5
  let overs_played := 10
  let total_overs := 50
  let runs_scored := run_rate_10 * overs_played
  let runs_remaining := target - runs_scored
  let overs_remaining := total_overs - overs_played
  (runs_remaining / overs_remaining = required_rate)

theorem cricket_problem_solved : cricket_problem :=
by
  sorry

end NUMINAMATH_GPT_cricket_problem_solved_l1034_103470


namespace NUMINAMATH_GPT_equal_distribution_arithmetic_seq_l1034_103479

theorem equal_distribution_arithmetic_seq :
  ∃ (a1 d : ℚ), (a1 + (a1 + d) = (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d)) ∧ 
                (5 * a1 + 10 / 2 * d = 5) ∧ 
                (a1 = 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_equal_distribution_arithmetic_seq_l1034_103479


namespace NUMINAMATH_GPT_find_integer_x_l1034_103431

theorem find_integer_x (x y : ℕ) (h_gt : x > y) (h_gt_zero : y > 0) (h_eq : x + y + x * y = 99) : x = 49 :=
sorry

end NUMINAMATH_GPT_find_integer_x_l1034_103431


namespace NUMINAMATH_GPT_least_possible_perimeter_l1034_103448

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_perimeter_l1034_103448


namespace NUMINAMATH_GPT_maximize_sector_area_l1034_103441

noncomputable def sector_radius_angle (r l α : ℝ) : Prop :=
  2 * r + l = 40 ∧ α = l / r

theorem maximize_sector_area :
  ∃ r α : ℝ, sector_radius_angle r 20 α ∧ r = 10 ∧ α = 2 :=
by
  sorry

end NUMINAMATH_GPT_maximize_sector_area_l1034_103441


namespace NUMINAMATH_GPT_total_sweaters_calculated_l1034_103490

def monday_sweaters := 8
def tuesday_sweaters := monday_sweaters + 2
def wednesday_sweaters := tuesday_sweaters - 4
def thursday_sweaters := tuesday_sweaters - 4
def friday_sweaters := monday_sweaters / 2

def total_sweaters := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem total_sweaters_calculated : total_sweaters = 34 := 
by sorry

end NUMINAMATH_GPT_total_sweaters_calculated_l1034_103490


namespace NUMINAMATH_GPT_initial_friends_count_l1034_103427

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end NUMINAMATH_GPT_initial_friends_count_l1034_103427


namespace NUMINAMATH_GPT_frosting_cupcakes_in_10_minutes_l1034_103474

def speed_Cagney := 1 / 20 -- Cagney frosts 1 cupcake every 20 seconds
def speed_Lacey := 1 / 30 -- Lacey frosts 1 cupcake every 30 seconds
def speed_Jamie := 1 / 15 -- Jamie frosts 1 cupcake every 15 seconds

def combined_speed := speed_Cagney + speed_Lacey + speed_Jamie -- Combined frosting rate (cupcakes per second)

def total_seconds := 10 * 60 -- 10 minutes converted to seconds

def number_of_cupcakes := combined_speed * total_seconds -- Total number of cupcakes frosted in 10 minutes

theorem frosting_cupcakes_in_10_minutes :
  number_of_cupcakes = 90 := by
  sorry

end NUMINAMATH_GPT_frosting_cupcakes_in_10_minutes_l1034_103474


namespace NUMINAMATH_GPT_total_fruit_in_buckets_l1034_103467

theorem total_fruit_in_buckets (A B C : ℕ) 
  (h1 : A = B + 4)
  (h2 : B = C + 3)
  (h3 : C = 9) :
  A + B + C = 37 := by
  sorry

end NUMINAMATH_GPT_total_fruit_in_buckets_l1034_103467


namespace NUMINAMATH_GPT_sum_of_x_coordinates_on_parabola_l1034_103409

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1

-- Define the points P and Q on the parabola
variables {x1 x2 : ℝ}

-- The Lean theorem statement: 
theorem sum_of_x_coordinates_on_parabola 
  (h1 : parabola x1 = 1) 
  (h2 : parabola x2 = 1) : 
  x1 + x2 = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_x_coordinates_on_parabola_l1034_103409


namespace NUMINAMATH_GPT_num_rectangles_in_5x5_grid_l1034_103481

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end NUMINAMATH_GPT_num_rectangles_in_5x5_grid_l1034_103481


namespace NUMINAMATH_GPT_calculate_product_sum_l1034_103458

theorem calculate_product_sum :
  17 * (17/18) + 35 * (35/36) = 50 + 1/12 :=
by sorry

end NUMINAMATH_GPT_calculate_product_sum_l1034_103458


namespace NUMINAMATH_GPT_seq_formula_l1034_103477

def S (n : ℕ) (a : ℕ → ℤ) : ℤ := 2 * a n + 1

theorem seq_formula (a : ℕ → ℤ) (S_n : ℕ → ℤ)
  (hS : ∀ n, S_n n = S n a) :
  a = fun n => -2^(n-1) := by
  sorry

end NUMINAMATH_GPT_seq_formula_l1034_103477


namespace NUMINAMATH_GPT_vector_subtraction_l1034_103491

-- Define the given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- State the theorem that the vector subtraction b - a equals (2, -1)
theorem vector_subtraction : b - a = (2, -1) :=
by
  -- Proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1034_103491


namespace NUMINAMATH_GPT_no_real_a_values_l1034_103432

noncomputable def polynomial_with_no_real_root (a : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 ≠ 0
  
theorem no_real_a_values :
  ∀ a : ℝ, (∃ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 = 0) → false :=
by sorry

end NUMINAMATH_GPT_no_real_a_values_l1034_103432


namespace NUMINAMATH_GPT_problem_statement_l1034_103488

-- Definitions for conditions
def cond_A : Prop := ∃ B : ℝ, B = 45 ∨ B = 135
def cond_B : Prop := ∃ C : ℝ, C = 90
def cond_C : Prop := false
def cond_D : Prop := ∃ B : ℝ, 0 < B ∧ B < 60

-- Prove that only cond_A has two possibilities
theorem problem_statement : cond_A ∧ ¬cond_B ∧ ¬cond_C ∧ ¬cond_D :=
by 
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_problem_statement_l1034_103488


namespace NUMINAMATH_GPT_range_of_a_l1034_103401

open Complex Real

theorem range_of_a (a : ℝ) (h : abs (1 + a * Complex.I) ≤ 2) : a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1034_103401


namespace NUMINAMATH_GPT_arrangement_of_athletes_l1034_103473

def num_arrangements (n : ℕ) (available_tracks_for_A : ℕ) (permutations_remaining : ℕ) : ℕ :=
  n * available_tracks_for_A * permutations_remaining

theorem arrangement_of_athletes :
  num_arrangements 2 3 24 = 144 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_of_athletes_l1034_103473
