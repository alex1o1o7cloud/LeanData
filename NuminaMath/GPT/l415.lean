import Mathlib

namespace NUMINAMATH_GPT_valid_numbers_count_l415_41510

def count_valid_numbers (n : ℕ) : ℕ := 1 / 4 * (5^n + 2 * 3^n + 1)

theorem valid_numbers_count (n : ℕ) : count_valid_numbers n = (1 / 4) * (5^n + 2 * 3^n + 1) :=
by sorry

end NUMINAMATH_GPT_valid_numbers_count_l415_41510


namespace NUMINAMATH_GPT_haley_marbles_l415_41514

theorem haley_marbles (boys marbles_per_boy : ℕ) (h1: boys = 5) (h2: marbles_per_boy = 7) : boys * marbles_per_boy = 35 := 
by 
  sorry

end NUMINAMATH_GPT_haley_marbles_l415_41514


namespace NUMINAMATH_GPT_A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l415_41507

def prob_A_wins_B_one_throw : ℚ := 1 / 3
def prob_tie_one_throw : ℚ := 1 / 3
def prob_A_wins_B_no_more_2_throws : ℚ := 4 / 9

def prob_C_treats_two_throws : ℚ := 2 / 9

def prob_C_treats_exactly_2_days_out_of_3 : ℚ := 28 / 243

theorem A_wins_B_no_more_than_two_throws (P1 : ℚ := prob_A_wins_B_one_throw) (P2 : ℚ := prob_tie_one_throw) :
  P1 + P2 * P1 = prob_A_wins_B_no_more_2_throws := 
by
  sorry

theorem C_treats_after_two_throws : prob_tie_one_throw ^ 2 = prob_C_treats_two_throws :=
by
  sorry

theorem C_treats_exactly_two_days (n : ℕ := 3) (k : ℕ := 2) (p_success : ℚ := prob_C_treats_two_throws) :
  (n.choose k) * (p_success ^ k) * ((1 - p_success) ^ (n - k)) = prob_C_treats_exactly_2_days_out_of_3 :=
by
  sorry

end NUMINAMATH_GPT_A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l415_41507


namespace NUMINAMATH_GPT_polygon_angle_multiple_l415_41567

theorem polygon_angle_multiple (m : ℕ) (h : m ≥ 3) : 
  (∃ k : ℕ, (2 * m - 2) * 180 = k * ((m - 2) * 180)) ↔ (m = 3 ∨ m = 4) :=
by sorry

end NUMINAMATH_GPT_polygon_angle_multiple_l415_41567


namespace NUMINAMATH_GPT_sequence_general_formula_l415_41518

theorem sequence_general_formula {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5 ^ n) 
  : ∀ n : ℕ, a n = 5 ^ n - 3 * 2 ^ (n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l415_41518


namespace NUMINAMATH_GPT_min_value_inequality_l415_41535

theorem min_value_inequality (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = (1 / (2*x + y)^2 + 4 / (x - 2*y)^2) ∧ m = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l415_41535


namespace NUMINAMATH_GPT_find_a_squared_plus_b_squared_l415_41596

theorem find_a_squared_plus_b_squared (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := 
by
  sorry

end NUMINAMATH_GPT_find_a_squared_plus_b_squared_l415_41596


namespace NUMINAMATH_GPT_maximize_profit_l415_41547

noncomputable def profit (x : ℕ) : ℝ :=
  let price := (180 + 10 * x : ℝ)
  let rooms_occupied := (50 - x : ℝ)
  let expenses := 20
  (price - expenses) * rooms_occupied

theorem maximize_profit :
  ∃ x : ℕ, profit x = profit 17 → (180 + 10 * x) = 350 :=
by
  use 17
  sorry

end NUMINAMATH_GPT_maximize_profit_l415_41547


namespace NUMINAMATH_GPT_trace_bag_weight_l415_41559

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end NUMINAMATH_GPT_trace_bag_weight_l415_41559


namespace NUMINAMATH_GPT_range_of_k_intersecting_hyperbola_l415_41594

theorem range_of_k_intersecting_hyperbola :
  (∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1) →
  -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_k_intersecting_hyperbola_l415_41594


namespace NUMINAMATH_GPT_student_difference_l415_41583

theorem student_difference 
  (C1 : ℕ) (x : ℕ)
  (hC1 : C1 = 25)
  (h_total : C1 + (C1 - x) + (C1 - 2 * x) + (C1 - 3 * x) + (C1 - 4 * x) = 105) : 
  x = 2 := 
by
  sorry

end NUMINAMATH_GPT_student_difference_l415_41583


namespace NUMINAMATH_GPT_most_reasonable_sampling_method_l415_41574

-- Definitions based on the conditions in the problem:
def area_divided_into_200_plots : Prop := true
def plan_randomly_select_20_plots : Prop := true
def large_difference_in_plant_coverage : Prop := true
def goal_representative_sample_accurate_estimate : Prop := true

-- Main theorem statement
theorem most_reasonable_sampling_method
  (h1 : area_divided_into_200_plots)
  (h2 : plan_randomly_select_20_plots)
  (h3 : large_difference_in_plant_coverage)
  (h4 : goal_representative_sample_accurate_estimate) :
  Stratified_sampling := 
sorry

end NUMINAMATH_GPT_most_reasonable_sampling_method_l415_41574


namespace NUMINAMATH_GPT_min_value_of_sum_of_powers_l415_41572

theorem min_value_of_sum_of_powers (x y : ℝ) (h : x + 3 * y = 1) : 
  2^x + 8^y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_sum_of_powers_l415_41572


namespace NUMINAMATH_GPT_minimum_value_frac_sum_l415_41536

-- Define the statement problem C and proof outline skipping the steps
theorem minimum_value_frac_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
by
  -- Proof is to be constructed here
  sorry

end NUMINAMATH_GPT_minimum_value_frac_sum_l415_41536


namespace NUMINAMATH_GPT_correct_calculation_l415_41566

variable (a : ℕ)

theorem correct_calculation : 
  ¬(a + a = a^2) ∧ ¬(a^3 * a = a^3) ∧ ¬(a^8 / a^2 = a^4) ∧ ((a^3)^2 = a^6) := 
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l415_41566


namespace NUMINAMATH_GPT_eval_expr_l415_41552

theorem eval_expr :
  - (18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l415_41552


namespace NUMINAMATH_GPT_fill_time_with_conditions_l415_41564

-- Define rates as constants
def pipeA_rate := 1 / 10
def pipeB_rate := 1 / 6
def pipeC_rate := 1 / 5
def tarp_factor := 1 / 2
def leak_rate := 1 / 15

-- Define effective fill rate taking into account the tarp and leak
def effective_fill_rate := ((pipeA_rate + pipeB_rate + pipeC_rate) * tarp_factor) - leak_rate

-- Define the required time to fill the pool
def required_time := 1 / effective_fill_rate

theorem fill_time_with_conditions :
  required_time = 6 :=
by
  sorry

end NUMINAMATH_GPT_fill_time_with_conditions_l415_41564


namespace NUMINAMATH_GPT_min_value_of_expression_l415_41592

theorem min_value_of_expression (x y : ℤ) (h : 4 * x + 5 * y = 7) : ∃ k : ℤ, 
  5 * Int.natAbs (3 + 5 * k) - 3 * Int.natAbs (-1 - 4 * k) = 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l415_41592


namespace NUMINAMATH_GPT_initial_ratio_milk_water_l415_41537

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 45) 
  (h2 : M = 3 * (W + 3)) 
  : M / W = 4 := 
sorry

end NUMINAMATH_GPT_initial_ratio_milk_water_l415_41537


namespace NUMINAMATH_GPT_zero_of_my_function_l415_41549

-- Define the function y = e^(2x) - 1
noncomputable def my_function (x : ℝ) : ℝ :=
  Real.exp (2 * x) - 1

-- Statement that the zero of the function is at x = 0
theorem zero_of_my_function : my_function 0 = 0 :=
by sorry

end NUMINAMATH_GPT_zero_of_my_function_l415_41549


namespace NUMINAMATH_GPT_tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l415_41520

theorem tan_beta_of_tan_alpha_and_tan_alpha_plus_beta (α β : ℝ)
  (h1 : Real.tan α = 2)
  (h2 : Real.tan (α + β) = 1 / 5) :
  Real.tan β = -9 / 7 :=
sorry

end NUMINAMATH_GPT_tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l415_41520


namespace NUMINAMATH_GPT_continuous_stripe_probability_l415_41573

-- Define the conditions of the tetrahedron and stripe orientations
def tetrahedron_faces : ℕ := 4
def stripe_orientations_per_face : ℕ := 2
def total_stripe_combinations : ℕ := stripe_orientations_per_face ^ tetrahedron_faces
def favorable_stripe_combinations : ℕ := 2 -- Clockwise and Counterclockwise combinations for a continuous stripe

-- Define the probability calculation
def probability_of_continuous_stripe : ℚ :=
  favorable_stripe_combinations / total_stripe_combinations

-- Theorem statement
theorem continuous_stripe_probability : probability_of_continuous_stripe = 1 / 8 :=
by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l415_41573


namespace NUMINAMATH_GPT_condition_sufficient_not_necessary_monotonicity_l415_41519

theorem condition_sufficient_not_necessary_monotonicity
  (f : ℝ → ℝ) (a : ℝ) (h_def : ∀ x, f x = 2^(abs (x - a))) :
  (∀ x > 1, x - a ≥ 0) → (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y) ∧
  (∃ a, a ≤ 1 ∧ (∀ x > 1, x - a ≥ 0) ∧ (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y)) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficient_not_necessary_monotonicity_l415_41519


namespace NUMINAMATH_GPT_solution_set_I_range_of_m_l415_41533

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem solution_set_I (x : ℝ) : f x < 8 ↔ -5 / 2 < x ∧ x < 3 / 2 :=
sorry

theorem range_of_m (m : ℝ) (h : ∃ x, f x ≤ |3 * m + 1|) : m ≤ -5 / 3 ∨ m ≥ 1 :=
sorry

end NUMINAMATH_GPT_solution_set_I_range_of_m_l415_41533


namespace NUMINAMATH_GPT_problem_statement_l415_41560

variable {x a : Real}

theorem problem_statement (h1 : x < a) (h2 : a < 0) : x^2 > a * x ∧ a * x > a^2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l415_41560


namespace NUMINAMATH_GPT_age_of_child_l415_41587

theorem age_of_child (H W C : ℕ) (h1 : (H + W) / 2 = 23) (h2 : (H + 5 + W + 5 + C) / 3 = 19) : C = 1 := by
  sorry

end NUMINAMATH_GPT_age_of_child_l415_41587


namespace NUMINAMATH_GPT_total_cupcakes_baked_l415_41532

-- Conditions
def morning_cupcakes : ℕ := 20
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

-- Goal
theorem total_cupcakes_baked :
  (morning_cupcakes + afternoon_cupcakes) = 55 :=
by
  sorry

end NUMINAMATH_GPT_total_cupcakes_baked_l415_41532


namespace NUMINAMATH_GPT_triangle_angle_side_cases_l415_41561

theorem triangle_angle_side_cases
  (b c : ℝ) (B : ℝ)
  (hb : b = 3)
  (hc : c = 3 * Real.sqrt 3)
  (hB : B = Real.pi / 6) :
  (∃ A C a, A = Real.pi / 2 ∧ C = Real.pi / 3 ∧ a = Real.sqrt 21) ∨
  (∃ A C a, A = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧ a = 3) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_side_cases_l415_41561


namespace NUMINAMATH_GPT_tara_additional_stamps_l415_41500

def stamps_needed (current_stamps total_stamps : Nat) : Nat :=
  if total_stamps % 9 == 0 then 0 else 9 - (total_stamps % 9)

theorem tara_additional_stamps :
  stamps_needed 38 45 = 7 := by
  sorry

end NUMINAMATH_GPT_tara_additional_stamps_l415_41500


namespace NUMINAMATH_GPT_melanie_bought_books_l415_41505

-- Defining the initial number of books and final number of books
def initial_books : ℕ := 41
def final_books : ℕ := 87

-- Theorem stating that Melanie bought 46 books at the yard sale
theorem melanie_bought_books : (final_books - initial_books) = 46 := by
  sorry

end NUMINAMATH_GPT_melanie_bought_books_l415_41505


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l415_41542

def setA : Set ℝ := { x | (x - 3) * (x + 1) ≥ 0 }
def setB : Set ℝ := { x | x < -4/5 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | x ≤ -1 } :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l415_41542


namespace NUMINAMATH_GPT_line_eq_l415_41509

theorem line_eq (m b : ℝ) 
  (h_slope : m = (4 + 2) / (3 - 1)) 
  (h_point : -2 = m * 1 + b) :
  m + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_line_eq_l415_41509


namespace NUMINAMATH_GPT_common_roots_product_sum_l415_41540

theorem common_roots_product_sum (C D u v w t p q r : ℝ) (huvw : u^3 + C * u - 20 = 0) (hvw : v^3 + C * v - 20 = 0)
  (hw: w^3 + C * w - 20 = 0) (hut: t^3 + D * t^2 - 40 = 0) (hvw: v^3 + D * v^2 - 40 = 0) 
  (hu: u^3 + D * u^2 - 40 = 0) (h1: u + v + w = 0) (h2: u * v * w = 20) 
  (h3: u * v + u * t + v * t = 0) (h4: u * v * t = 40) :
  p = 4 → q = 3 → r = 5 → p + q + r = 12 :=
by sorry

end NUMINAMATH_GPT_common_roots_product_sum_l415_41540


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_eight_l415_41516

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 2 - a 1)

variable {a : ℕ → ℤ}

theorem arithmetic_sequence_a4_eight (h_arith_sequence : arithmetic_sequence a)
    (h_cond : a 3 + a 5 = 16) : a 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_eight_l415_41516


namespace NUMINAMATH_GPT_probability_between_lines_l415_41593

def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

theorem probability_between_lines 
  (h1 : ∀ x > 0, line_l x ≥ 0) 
  (h2 : ∀ x > 0, line_m x ≥ 0) 
  (h3 : ∀ x > 0, line_l x < line_m x ∨ line_m x ≤ 0) : 
  (1 / 16 : ℝ) * 100 = 0.16 :=
by
  sorry

end NUMINAMATH_GPT_probability_between_lines_l415_41593


namespace NUMINAMATH_GPT_fraction_of_pianists_got_in_l415_41544

-- Define the conditions
def flutes_got_in (f : ℕ) := f = 16
def clarinets_got_in (c : ℕ) := c = 15
def trumpets_got_in (t : ℕ) := t = 20
def total_band_members (total : ℕ) := total = 53
def total_pianists (p : ℕ) := p = 20

-- The main statement we want to prove
theorem fraction_of_pianists_got_in : 
  ∃ (pi : ℕ), 
    flutes_got_in 16 ∧ 
    clarinets_got_in 15 ∧ 
    trumpets_got_in 20 ∧ 
    total_band_members 53 ∧ 
    total_pianists 20 ∧ 
    pi / 20 = 1 / 10 := 
  sorry

end NUMINAMATH_GPT_fraction_of_pianists_got_in_l415_41544


namespace NUMINAMATH_GPT_binom_8_2_eq_28_l415_41555

open Nat

theorem binom_8_2_eq_28 : Nat.choose 8 2 = 28 := by
  sorry

end NUMINAMATH_GPT_binom_8_2_eq_28_l415_41555


namespace NUMINAMATH_GPT_g_at_1_l415_41541

variable (g : ℝ → ℝ)

theorem g_at_1 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 1 = 18 := by
  sorry

end NUMINAMATH_GPT_g_at_1_l415_41541


namespace NUMINAMATH_GPT_digit_B_for_divisibility_by_9_l415_41530

theorem digit_B_for_divisibility_by_9 :
  ∃! (B : ℕ), B < 10 ∧ (5 + B + B + 3) % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_digit_B_for_divisibility_by_9_l415_41530


namespace NUMINAMATH_GPT_average_visitors_per_day_l415_41581

theorem average_visitors_per_day
  (sunday_visitors : ℕ := 540)
  (other_days_visitors : ℕ := 240)
  (days_in_month : ℕ := 30)
  (first_day_is_sunday : Bool := true)
  (result : ℕ := 290) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_days_visitors
  let average_visitors := total_visitors / days_in_month
  average_visitors = result :=
by
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_l415_41581


namespace NUMINAMATH_GPT_average_books_collected_per_day_l415_41598

theorem average_books_collected_per_day :
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  S_n / n = 48 :=
by
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  show S_n / n = 48
  sorry

end NUMINAMATH_GPT_average_books_collected_per_day_l415_41598


namespace NUMINAMATH_GPT_correct_option_l415_41508

theorem correct_option : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := sorry

end NUMINAMATH_GPT_correct_option_l415_41508


namespace NUMINAMATH_GPT_div_eq_of_scaled_div_eq_l415_41531

theorem div_eq_of_scaled_div_eq (h : 29.94 / 1.45 = 17.7) : 2994 / 14.5 = 17.7 := 
by
  sorry

end NUMINAMATH_GPT_div_eq_of_scaled_div_eq_l415_41531


namespace NUMINAMATH_GPT_monotone_f_range_l415_41524

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_f_range (a : ℝ) :
  (∀ x : ℝ, (1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x) ≥ 0) ↔ (-1 / 3 ≤ a ∧ a ≤ 1 / 3) := 
sorry

end NUMINAMATH_GPT_monotone_f_range_l415_41524


namespace NUMINAMATH_GPT_minimum_value_of_f_l415_41586

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem minimum_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 * Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l415_41586


namespace NUMINAMATH_GPT_intersection_A_B_l415_41522

def A (x : ℝ) : Prop := x^2 - 3 * x < 0
def B (x : ℝ) : Prop := x > 2

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l415_41522


namespace NUMINAMATH_GPT_farm_area_l415_41506

theorem farm_area
  (b : ℕ) (l : ℕ) (d : ℕ)
  (h_b : b = 30)
  (h_cost : 15 * (l + b + d) = 1800)
  (h_pythagorean : d^2 = l^2 + b^2) :
  l * b = 1200 :=
by
  sorry

end NUMINAMATH_GPT_farm_area_l415_41506


namespace NUMINAMATH_GPT_find_larger_number_l415_41539

variable (x y : ℕ)

theorem find_larger_number (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 := 
by 
  sorry

end NUMINAMATH_GPT_find_larger_number_l415_41539


namespace NUMINAMATH_GPT_three_gorges_dam_capacity_scientific_notation_l415_41570

theorem three_gorges_dam_capacity_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (16780000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.678 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_three_gorges_dam_capacity_scientific_notation_l415_41570


namespace NUMINAMATH_GPT_side_length_of_S2_l415_41513

-- Define our context and the statements we need to work with
theorem side_length_of_S2
  (r s : ℕ)
  (h1 : 2 * r + s = 2450)
  (h2 : 2 * r + 3 * s = 4000) : 
  s = 775 :=
sorry

end NUMINAMATH_GPT_side_length_of_S2_l415_41513


namespace NUMINAMATH_GPT_polynomial_use_square_of_binomial_form_l415_41551

theorem polynomial_use_square_of_binomial_form (a b x y : ℝ) :
  (1 + x) * (x + 1) = (x + 1) ^ 2 ∧ 
  (2 * a + b) * (b - 2 * a) = b^2 - 4 * a^2 ∧ 
  (-a + b) * (a - b) = - (a - b)^2 ∧ 
  (x^2 - y) * (y^2 + x) ≠ (x + y)^2 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_use_square_of_binomial_form_l415_41551


namespace NUMINAMATH_GPT_probability_of_square_or_circle_is_seven_tenths_l415_41589

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- The number of squares or circles
def num_squares_or_circles : ℕ := num_squares + num_circles

-- The probability of selecting a square or a circle
def probability_square_or_circle : ℚ := num_squares_or_circles / total_figures

-- The theorem stating the required proof
theorem probability_of_square_or_circle_is_seven_tenths :
  probability_square_or_circle = 7/10 :=
sorry -- proof goes here

end NUMINAMATH_GPT_probability_of_square_or_circle_is_seven_tenths_l415_41589


namespace NUMINAMATH_GPT_intersection_point_not_on_x_3_l415_41562

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)
noncomputable def g (x : ℝ) : ℝ := (-1/3 * x^2 + 6*x - 6) / (x - 2)

theorem intersection_point_not_on_x_3 : 
  ∃ x y : ℝ, (x ≠ 3) ∧ (f x = g x) ∧ (y = f x) ∧ (x = 11/3 ∧ y = -11/3) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_not_on_x_3_l415_41562


namespace NUMINAMATH_GPT_op_dot_of_10_5_l415_41501

-- Define the operation \odot
def op_dot (a b : ℕ) : ℕ := a + (2 * a) / b

-- Theorem stating that 10 \odot 5 = 14
theorem op_dot_of_10_5 : op_dot 10 5 = 14 :=
by
  sorry

end NUMINAMATH_GPT_op_dot_of_10_5_l415_41501


namespace NUMINAMATH_GPT_min_PM_PN_l415_41517

noncomputable def C1 (x y : ℝ) : Prop := (x + 6)^2 + (y - 5)^2 = 4
noncomputable def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

theorem min_PM_PN : ∀ (P M N : ℝ × ℝ),
  P.2 = 0 ∧ C1 M.1 M.2 ∧ C2 N.1 N.2 → (|P.1 - M.1| + (P.1 - N.1)^2 + (P.2 - N.2)^2).sqrt = 7 := by
  sorry

end NUMINAMATH_GPT_min_PM_PN_l415_41517


namespace NUMINAMATH_GPT_count_4x4_increasing_arrays_l415_41538

-- Define the notion of a 4x4 grid that satisfies the given conditions
def isInIncreasingOrder (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, i < 3 -> matrix i j < matrix (i+1) j) ∧
  (∀ i j : Fin 4, j < 3 -> matrix i j < matrix i (j+1))

def validGrid (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, 1 ≤ matrix i j ∧ matrix i j ≤ 16) ∧ isInIncreasingOrder matrix

noncomputable def countValidGrids : ℕ :=
  sorry

theorem count_4x4_increasing_arrays : countValidGrids = 13824 :=
  sorry

end NUMINAMATH_GPT_count_4x4_increasing_arrays_l415_41538


namespace NUMINAMATH_GPT_perfect_square_sequence_l415_41526

theorem perfect_square_sequence (k : ℤ) (y : ℕ → ℤ) :
  (y 1 = 1) ∧ (y 2 = 1) ∧
  (∀ n : ℕ, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) →
  (∀ n ≥ 1, ∃ m : ℤ, y n = m^2) ↔ (k = 1 ∨ k = 3) :=
sorry

end NUMINAMATH_GPT_perfect_square_sequence_l415_41526


namespace NUMINAMATH_GPT_number_of_red_balloons_l415_41577

-- Definitions for conditions
def balloons_total : ℕ := 85
def at_least_one_red (red blue : ℕ) : Prop := red ≥ 1 ∧ red + blue = balloons_total
def every_pair_has_blue (red blue : ℕ) : Prop := ∀ r1 r2, r1 < red → r2 < red → red = 1

-- Theorem to be proved
theorem number_of_red_balloons (red blue : ℕ) 
  (total : red + blue = balloons_total)
  (at_least_one : at_least_one_red red blue)
  (pair_condition : every_pair_has_blue red blue) : red = 1 :=
sorry

end NUMINAMATH_GPT_number_of_red_balloons_l415_41577


namespace NUMINAMATH_GPT_reciprocal_difference_decreases_l415_41571

theorem reciprocal_difference_decreases (n : ℕ) (hn : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1 : ℝ)) < (1 / (n * n : ℝ)) :=
by 
  sorry

end NUMINAMATH_GPT_reciprocal_difference_decreases_l415_41571


namespace NUMINAMATH_GPT_find_q_from_min_y_l415_41575

variables (a p q m : ℝ)
variable (a_nonzero : a ≠ 0)
variable (min_y : ∀ x : ℝ, a*x^2 + p*x + q ≥ m)

theorem find_q_from_min_y :
  q = m + p^2 / (4 * a) :=
sorry

end NUMINAMATH_GPT_find_q_from_min_y_l415_41575


namespace NUMINAMATH_GPT_maximum_value_2a_plus_b_l415_41525

variable (a b : ℝ)

theorem maximum_value_2a_plus_b (h : 4 * a^2 + b^2 + a * b = 1) : 2 * a + b ≤ 2 * Real.sqrt (10) / 5 :=
by sorry

end NUMINAMATH_GPT_maximum_value_2a_plus_b_l415_41525


namespace NUMINAMATH_GPT_paint_mixer_days_l415_41521

/-- Making an equal number of drums of paint each day, a paint mixer takes three days to make 18 drums of paint.
    We want to determine how many days it will take for him to make 360 drums of paint. -/
theorem paint_mixer_days (n : ℕ) (h1 : n > 0) 
  (h2 : 3 * n = 18) : 
  360 / n = 60 := by
  sorry

end NUMINAMATH_GPT_paint_mixer_days_l415_41521


namespace NUMINAMATH_GPT_area_of_larger_square_l415_41579

theorem area_of_larger_square (side_length : ℕ) (num_squares : ℕ)
  (h₁ : side_length = 2)
  (h₂ : num_squares = 8) : 
  (num_squares * side_length^2) = 32 :=
by
  sorry

end NUMINAMATH_GPT_area_of_larger_square_l415_41579


namespace NUMINAMATH_GPT_cos_double_angle_of_tan_half_l415_41550

theorem cos_double_angle_of_tan_half (α : ℝ) (h : Real.tan α = 1 / 2) :
  Real.cos (2 * α) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_of_tan_half_l415_41550


namespace NUMINAMATH_GPT_tan_sum_formula_l415_41588

theorem tan_sum_formula {A B : ℝ} (hA : A = 55) (hB : B = 65) (h1 : Real.tan (A + B) = Real.tan 120) 
    (h2 : Real.tan 120 = -Real.sqrt 3) :
    Real.tan 55 + Real.tan 65 - Real.sqrt 3 * Real.tan 55 * Real.tan 65 = -Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_sum_formula_l415_41588


namespace NUMINAMATH_GPT_bananas_left_l415_41554

theorem bananas_left (dozen_bananas : ℕ) (eaten_bananas : ℕ) (h1 : dozen_bananas = 12) (h2 : eaten_bananas = 2) : dozen_bananas - eaten_bananas = 10 :=
sorry

end NUMINAMATH_GPT_bananas_left_l415_41554


namespace NUMINAMATH_GPT_find_angle_B_l415_41557

noncomputable def triangle_sides_and_angles 
(a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def vectors_parallel 
(A B C a b c : ℝ) : Prop :=
  (Real.sin B - Real.sin A) / Real.sin C = (Real.sqrt 3 * a + c) / (a + b)

theorem find_angle_B (A B C a b c : ℝ)
  (h_triangle : triangle_sides_and_angles a b c A B C)
  (h_parallel : vectors_parallel A B C a b c) :
  B = 5 * Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l415_41557


namespace NUMINAMATH_GPT_arrangements_count_l415_41595

-- Definitions of students and grades
inductive Student : Type
| A | B | C | D | E | F
deriving DecidableEq

inductive Grade : Type
| first | second | third
deriving DecidableEq

-- A function to count valid arrangements
def valid_arrangements (assignments : Student → Grade) : Bool :=
  assignments Student.A = Grade.first ∧
  assignments Student.B ≠ Grade.third ∧
  assignments Student.C ≠ Grade.third ∧
  (assignments Student.A = Grade.first) ∧
  ((assignments Student.B = Grade.second ∧ assignments Student.C = Grade.second ∧ 
    (assignments Student.D ≠ Grade.first ∨ assignments Student.E ≠ Grade.first ∨ assignments Student.F ≠ Grade.first)) ∨
   ((assignments Student.B ≠ Grade.second ∨ assignments Student.C ≠ Grade.second) ∧ 
    (assignments Student.B ≠ Grade.first ∨ assignments Student.C ≠ Grade.first)))

theorem arrangements_count : 
  ∃ (count : ℕ), count = 9 ∧
  count = (Nat.card { assign : Student → Grade // valid_arrangements assign } : ℕ) := sorry

end NUMINAMATH_GPT_arrangements_count_l415_41595


namespace NUMINAMATH_GPT_miniature_tower_height_l415_41556

theorem miniature_tower_height
  (actual_height : ℝ)
  (actual_volume : ℝ)
  (miniature_volume : ℝ)
  (actual_height_eq : actual_height = 60)
  (actual_volume_eq : actual_volume = 200000)
  (miniature_volume_eq : miniature_volume = 0.2) :
  ∃ (miniature_height : ℝ), miniature_height = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_miniature_tower_height_l415_41556


namespace NUMINAMATH_GPT_division_identity_l415_41527

theorem division_identity
  (x y : ℕ)
  (h1 : x = 7)
  (h2 : y = 2)
  : (x^3 + y^3) / (x^2 - x * y + y^2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_division_identity_l415_41527


namespace NUMINAMATH_GPT_triangle_angle_C_l415_41548

theorem triangle_angle_C (A B C : ℝ) (h1 : A = 86) (h2 : B = 3 * C + 22) (h3 : A + B + C = 180) : C = 18 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_C_l415_41548


namespace NUMINAMATH_GPT_students_not_in_any_subject_l415_41546

theorem students_not_in_any_subject (total_students mathematics_students chemistry_students biology_students
  mathematics_chemistry_students chemistry_biology_students mathematics_biology_students all_three_students: ℕ)
  (h_total: total_students = 120) 
  (h_m: mathematics_students = 70)
  (h_c: chemistry_students = 50)
  (h_b: biology_students = 40)
  (h_mc: mathematics_chemistry_students = 30)
  (h_cb: chemistry_biology_students = 20)
  (h_mb: mathematics_biology_students = 10)
  (h_all: all_three_students = 5) :
  total_students - ((mathematics_students - mathematics_chemistry_students - mathematics_biology_students + all_three_students) +
    (chemistry_students - chemistry_biology_students - mathematics_chemistry_students + all_three_students) +
    (biology_students - chemistry_biology_students - mathematics_biology_students + all_three_students) +
    (mathematics_chemistry_students + chemistry_biology_students + mathematics_biology_students - 2 * all_three_students)) = 20 :=
by sorry

end NUMINAMATH_GPT_students_not_in_any_subject_l415_41546


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l415_41599

theorem inequality_for_positive_reals (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℕ) (h_k : 2 ≤ k) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a) ≥ 3 / 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l415_41599


namespace NUMINAMATH_GPT_possible_values_of_expression_l415_41558

theorem possible_values_of_expression (x y : ℝ) (hxy : x + 2 * y = 2) (hx_pos : x > 0) (hy_pos : y > 0) :
  ∃ v, v = 21 / 4 ∧ (1 / x + 2 / y) = v :=
sorry

end NUMINAMATH_GPT_possible_values_of_expression_l415_41558


namespace NUMINAMATH_GPT_binary_to_base5_conversion_l415_41545

theorem binary_to_base5_conversion : ∀ (b : ℕ), b = 1101 → (13 : ℕ) % 5 = 3 ∧ (13 / 5) % 5 = 2 → b = 1101 → (1101 : ℕ) = 13 → 13 = 23 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_base5_conversion_l415_41545


namespace NUMINAMATH_GPT_prime_p_prime_p₁₀_prime_p₁₄_l415_41503

theorem prime_p_prime_p₁₀_prime_p₁₄ (p : ℕ) (h₀p : Nat.Prime p) 
  (h₁ : Nat.Prime (p + 10)) (h₂ : Nat.Prime (p + 14)) : p = 3 := by
  sorry

end NUMINAMATH_GPT_prime_p_prime_p₁₀_prime_p₁₄_l415_41503


namespace NUMINAMATH_GPT_max_coins_as_pleases_max_coins_equally_distributed_l415_41568

-- Part a
theorem max_coins_as_pleases {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 31) := 
by
  sorry

-- Part b
theorem max_coins_equally_distributed {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 30) := 
by
  sorry

end NUMINAMATH_GPT_max_coins_as_pleases_max_coins_equally_distributed_l415_41568


namespace NUMINAMATH_GPT_smallest_value_geq_4_l415_41553

noncomputable def smallest_value (a b c d : ℝ) : ℝ :=
  (a + b + c + d) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem smallest_value_geq_4 (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_value a b c d ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_geq_4_l415_41553


namespace NUMINAMATH_GPT_exists_segment_satisfying_condition_l415_41534

theorem exists_segment_satisfying_condition :
  ∃ (x₁ x₂ x₃ : ℚ) (f : ℚ → ℤ), x₃ = (x₁ + x₂) / 2 ∧ f x₁ + f x₂ ≤ 2 * f x₃ :=
sorry

end NUMINAMATH_GPT_exists_segment_satisfying_condition_l415_41534


namespace NUMINAMATH_GPT_find_y_solution_l415_41591

variable (y : ℚ)

theorem find_y_solution (h : (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5) : 
    y = -17/6 :=
by
  sorry

end NUMINAMATH_GPT_find_y_solution_l415_41591


namespace NUMINAMATH_GPT_triangle_side_length_l415_41512

theorem triangle_side_length 
  (a b c : ℝ) 
  (cosA : ℝ) 
  (h1: a = Real.sqrt 5) 
  (h2: c = 2) 
  (h3: cosA = 2 / 3) 
  (h4: a^2 = b^2 + c^2 - 2 * b * c * cosA) : 
  b = 3 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_side_length_l415_41512


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l415_41585

variables {a b : ℝ}

theorem necessary_but_not_sufficient (h : a > 0) (h₁ : a > b) (h₂ : a⁻¹ > b⁻¹) : 
  b < 0 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l415_41585


namespace NUMINAMATH_GPT_integer_solution_existence_l415_41578

theorem integer_solution_existence : ∃ (x y : ℤ), 2 * x + y - 1 = 0 :=
by
  use 1
  use -1
  sorry

end NUMINAMATH_GPT_integer_solution_existence_l415_41578


namespace NUMINAMATH_GPT_fraction_expression_evaluation_l415_41563

theorem fraction_expression_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/4) = 1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_expression_evaluation_l415_41563


namespace NUMINAMATH_GPT_greatest_xy_value_l415_41597

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end NUMINAMATH_GPT_greatest_xy_value_l415_41597


namespace NUMINAMATH_GPT_ratio_of_juice_to_bread_l415_41515

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_of_juice_to_bread_l415_41515


namespace NUMINAMATH_GPT_new_average_doubled_marks_l415_41565

theorem new_average_doubled_marks (n : ℕ) (avg : ℕ) (h_n : n = 11) (h_avg : avg = 36) :
  (2 * avg * n) / n = 72 :=
by
  sorry

end NUMINAMATH_GPT_new_average_doubled_marks_l415_41565


namespace NUMINAMATH_GPT_tip_percentage_l415_41582

theorem tip_percentage
  (total_amount_paid : ℝ)
  (price_of_food : ℝ)
  (sales_tax_rate : ℝ)
  (total_amount : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_amount_paid = 184.80)
  (h2 : price_of_food = 140)
  (h3 : sales_tax_rate = 0.10)
  (h4 : total_amount = price_of_food + (price_of_food * sales_tax_rate))
  (h5 : tip_percentage = ((total_amount_paid - total_amount) / total_amount) * 100) :
  tip_percentage = 20 := sorry

end NUMINAMATH_GPT_tip_percentage_l415_41582


namespace NUMINAMATH_GPT_calculate_g3_l415_41511

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem calculate_g3 : g 3 = 3 / 17 :=
by {
    -- Here we add the proof steps if necessary, but for now we use sorry
    sorry
}

end NUMINAMATH_GPT_calculate_g3_l415_41511


namespace NUMINAMATH_GPT_chromium_atoms_in_compound_l415_41569

-- Definitions of given conditions
def hydrogen_atoms : Nat := 2
def oxygen_atoms : Nat := 4
def compound_molecular_weight : ℝ := 118
def hydrogen_atomic_weight : ℝ := 1
def chromium_atomic_weight : ℝ := 52
def oxygen_atomic_weight : ℝ := 16

-- Problem statement to find the number of Chromium atoms
theorem chromium_atoms_in_compound (hydrogen_atoms : Nat) (oxygen_atoms : Nat) (compound_molecular_weight : ℝ)
    (hydrogen_atomic_weight : ℝ) (chromium_atomic_weight : ℝ) (oxygen_atomic_weight : ℝ) :
  hydrogen_atoms * hydrogen_atomic_weight + 
  oxygen_atoms * oxygen_atomic_weight + 
  chromium_atomic_weight = compound_molecular_weight → 
  chromium_atomic_weight = 52 :=
by
  sorry

end NUMINAMATH_GPT_chromium_atoms_in_compound_l415_41569


namespace NUMINAMATH_GPT_solve_system_equations_l415_41523

-- Define the hypotheses of the problem
variables {a x y : ℝ}
variables (h1 : (0 < a) ∧ (a ≠ 1))
variables (h2 : (0 < x))
variables (h3 : (0 < y))
variables (eq1 : (log a x + log a y - 2) * log 18 a = 1)
variables (eq2 : 2 * x + y - 20 * a = 0)

-- State the theorem to be proved
theorem solve_system_equations :
  (x = a ∧ y = 18 * a) ∨ (x = 9 * a ∧ y = 2 * a) := by
  sorry

end NUMINAMATH_GPT_solve_system_equations_l415_41523


namespace NUMINAMATH_GPT_sum_fifth_powers_l415_41504

theorem sum_fifth_powers (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^5 + b^5 + c^5 = 98 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_sum_fifth_powers_l415_41504


namespace NUMINAMATH_GPT_thirty_five_power_identity_l415_41529

theorem thirty_five_power_identity (m n : ℕ) : 
  let P := 5^m 
  let Q := 7^n 
  35^(m*n) = P^n * Q^m :=
by 
  sorry

end NUMINAMATH_GPT_thirty_five_power_identity_l415_41529


namespace NUMINAMATH_GPT_problem_proof_l415_41580

theorem problem_proof (N : ℤ) (h : N / 5 = 4) : ((N - 10) * 3) - 18 = 12 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_proof_l415_41580


namespace NUMINAMATH_GPT_translate_parabola_l415_41584

noncomputable def f (x : ℝ) : ℝ := 3 * x^2

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

theorem translate_parabola (x : ℝ) : g x = 3 * (x - 1)^2 - 4 :=
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_translate_parabola_l415_41584


namespace NUMINAMATH_GPT_triplet_D_sum_not_one_l415_41590

def triplet_sum_not_equal_to_one : Prop :=
  (1.2 + -0.2 + 0.0 ≠ 1)

theorem triplet_D_sum_not_one : triplet_sum_not_equal_to_one := 
  by
    sorry

end NUMINAMATH_GPT_triplet_D_sum_not_one_l415_41590


namespace NUMINAMATH_GPT_derivative_at_one_l415_41576

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

theorem derivative_at_one : deriv f 1 = -2 :=
by 
  -- Here we would provide the proof that f'(1) = -2
  sorry

end NUMINAMATH_GPT_derivative_at_one_l415_41576


namespace NUMINAMATH_GPT_number_of_red_balls_l415_41528

theorem number_of_red_balls (x : ℕ) (h₀ : 4 > 0) (h₁ : (x : ℝ) / (x + 4) = 0.6) : x = 6 :=
sorry

end NUMINAMATH_GPT_number_of_red_balls_l415_41528


namespace NUMINAMATH_GPT_fractional_eq_nonneg_solution_l415_41502

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end NUMINAMATH_GPT_fractional_eq_nonneg_solution_l415_41502


namespace NUMINAMATH_GPT_sandy_marks_loss_l415_41543

theorem sandy_marks_loss (n m c p : ℕ) (h1 : n = 30) (h2 : m = 65) (h3 : c = 25) (h4 : p = 3) :
  ∃ x : ℕ, (c * p - m) / (n - c) = x ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_sandy_marks_loss_l415_41543
