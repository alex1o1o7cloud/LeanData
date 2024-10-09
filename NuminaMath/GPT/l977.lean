import Mathlib

namespace diet_soda_bottles_l977_97747

-- Define the conditions and then state the problem
theorem diet_soda_bottles (R D : ℕ) (h1 : R = 67) (h2 : R = D + 58) : D = 9 :=
by
  -- The proof goes here
  sorry

end diet_soda_bottles_l977_97747


namespace students_taking_history_but_not_statistics_l977_97799

theorem students_taking_history_but_not_statistics (H S U : ℕ) (total_students : ℕ) 
  (H_val : H = 36) (S_val : S = 30) (U_val : U = 59) (total_students_val : total_students = 90) :
  H - (H + S - U) = 29 := 
by
  sorry

end students_taking_history_but_not_statistics_l977_97799


namespace power_modulo_l977_97737

theorem power_modulo (k : ℕ) : 7^32 % 19 = 1 → 7^2050 % 19 = 11 :=
by {
  sorry
}

end power_modulo_l977_97737


namespace jerusha_and_lottie_earnings_l977_97750

theorem jerusha_and_lottie_earnings :
  let J := 68
  let L := J / 4
  J + L = 85 := 
by
  sorry

end jerusha_and_lottie_earnings_l977_97750


namespace females_with_advanced_degrees_l977_97771

noncomputable def total_employees := 200
noncomputable def total_females := 120
noncomputable def total_advanced_degrees := 100
noncomputable def males_college_degree_only := 40

theorem females_with_advanced_degrees :
  (total_employees - total_females) - males_college_degree_only = 
  total_employees - total_females - males_college_degree_only ∧ 
  total_females = 120 ∧ 
  total_advanced_degrees = 100 ∧ 
  total_employees = 200 ∧ 
  males_college_degree_only = 40 ∧
  total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60 :=
sorry

end females_with_advanced_degrees_l977_97771


namespace cost_per_box_l977_97774

theorem cost_per_box (trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℕ) (box_cost : ℝ) 
  (h1 : trays = 3) 
  (h2 : cookies_per_tray = 80) 
  (h3 : cookies_per_box = 60)
  (h4 : total_cost = 14) 
  (h5 : (trays * cookies_per_tray) = 240)
  (h6 : (240 / cookies_per_box : ℕ) = 4) 
  (h7 : (total_cost / 4 : ℝ) = box_cost) : 
  box_cost = 3.5 := 
by sorry

end cost_per_box_l977_97774


namespace number_of_nickels_l977_97700

variable (n : Nat) -- number of nickels

def value_of_nickels := n * 5 -- value of nickels n in cents
def total_value :=
    2 * 100 +   -- 2 one-dollar bills
    1 * 500 +   -- 1 five-dollar bill
    13 * 25 +   -- 13 quarters
    20 * 10 +   -- 20 dimes
    35 * 1 +    -- 35 pennies
    value_of_nickels n

theorem number_of_nickels :
    total_value n = 1300 ↔ n = 8 :=
by sorry

end number_of_nickels_l977_97700


namespace student_correct_answers_l977_97740

noncomputable def correct_answers : ℕ := 58

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = correct_answers :=
by {
  -- placeholder for actual proof
  sorry
}

end student_correct_answers_l977_97740


namespace antisymmetric_function_multiplication_cauchy_solution_l977_97741

variable (f : ℤ → ℤ)
variable (h : ∀ x y : ℤ, f (x + y) = f x + f y)

theorem antisymmetric : ∀ x : ℤ, f (-x) = -f x := by
  sorry

theorem function_multiplication : ∀ x y : ℤ, f (x * y) = x * f y := by
  sorry

theorem cauchy_solution : ∃ c : ℤ, ∀ x : ℤ, f x = c * x := by
  sorry

end antisymmetric_function_multiplication_cauchy_solution_l977_97741


namespace fraction_question_l977_97720

theorem fraction_question :
  ((3 / 8 + 5 / 6) / (5 / 12 + 1 / 4) = 29 / 16) :=
by
  -- This is where we will put the proof steps 
  sorry

end fraction_question_l977_97720


namespace decreasing_interval_of_function_l977_97787

noncomputable def y (x : ℝ) : ℝ := (3 / Real.pi) ^ (x ^ 2 + 2 * x - 3)

theorem decreasing_interval_of_function :
  ∀ x ∈ Set.Ioi (-1 : ℝ), ∃ ε > 0, ∀ δ > 0, δ ≤ ε → y (x - δ) > y x :=
by
  sorry

end decreasing_interval_of_function_l977_97787


namespace max_f_l977_97775

noncomputable def f (x : ℝ) : ℝ :=
  min (min (2 * x + 2) (1 / 2 * x + 1)) (-3 / 4 * x + 7)

theorem max_f : ∃ x : ℝ, f x = 17 / 5 :=
by
  sorry

end max_f_l977_97775


namespace condition_for_a_pow_zero_eq_one_l977_97709

theorem condition_for_a_pow_zero_eq_one (a : Real) : a ≠ 0 ↔ a^0 = 1 :=
by
  sorry

end condition_for_a_pow_zero_eq_one_l977_97709


namespace horner_rule_polynomial_polynomial_value_at_23_l977_97773

def polynomial (x : ℤ) : ℤ := 7 * x ^ 3 + 3 * x ^ 2 - 5 * x + 11

def horner_polynomial (x : ℤ) : ℤ := x * ((7 * x + 3) * x - 5) + 11

theorem horner_rule_polynomial (x : ℤ) : polynomial x = horner_polynomial x :=
by 
  -- The proof steps would go here,
  -- demonstrating that polynomial x = horner_polynomial x.
  sorry

-- Instantiation of the theorem for a specific value of x
theorem polynomial_value_at_23 : polynomial 23 = horner_polynomial 23 :=
by 
  -- Using the previously established theorem
  apply horner_rule_polynomial

end horner_rule_polynomial_polynomial_value_at_23_l977_97773


namespace analogical_reasoning_correct_l977_97703

theorem analogical_reasoning_correct (a b c : ℝ) (hc : c ≠ 0) : (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c :=
by
  sorry

end analogical_reasoning_correct_l977_97703


namespace smallest_four_digit_divisible_by_53_l977_97762

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l977_97762


namespace distinct_positive_integers_count_l977_97756

-- Define the digits' ranges
def digit (n : ℤ) : Prop := 0 ≤ n ∧ n ≤ 9
def nonzero_digit (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define the 4-digit numbers ABCD and DCBA
def ABCD (A B C D : ℤ) := 1000 * A + 100 * B + 10 * C + D
def DCBA (A B C D : ℤ) := 1000 * D + 100 * C + 10 * B + A

-- Define the difference
def difference (A B C D : ℤ) := ABCD A B C D - DCBA A B C D

-- The theorem to be proven
theorem distinct_positive_integers_count :
  ∃ n : ℤ, n = 161 ∧
  ∀ A B C D : ℤ,
  nonzero_digit A → nonzero_digit D → digit B → digit C → 
  0 < difference A B C D → (∃! x : ℤ, x = difference A B C D) :=
sorry

end distinct_positive_integers_count_l977_97756


namespace problem_part1_problem_part2_l977_97785

noncomputable def f (x a : ℝ) := |x - a| + x

theorem problem_part1 (a : ℝ) (h_a : a = 1) :
  {x : ℝ | f x a ≥ x + 2} = {x : ℝ | x ≥ 3} ∪ {x : ℝ | x ≤ -1} :=
by 
  simp [h_a, f]
  sorry

theorem problem_part2 (a : ℝ) (h_solution : {x : ℝ | f x a ≤ 3 * x} = {x : ℝ | x ≥ 2}) :
  a = 6 :=
by
  simp [f] at h_solution
  sorry

end problem_part1_problem_part2_l977_97785


namespace circle_intersection_area_l977_97705

theorem circle_intersection_area
  (r : ℝ)
  (θ : ℝ)
  (a b c : ℝ)
  (h_r : r = 5)
  (h_θ : θ = π / 2)
  (h_expr : a * Real.sqrt b + c * π = 5 * 5 * π / 2 - 5 * 5 * Real.sqrt 3 / 2 ) :
  a + b + c = -9.5 :=
by
  sorry

end circle_intersection_area_l977_97705


namespace value_of_x_for_zero_expression_l977_97739

theorem value_of_x_for_zero_expression (x : ℝ) (h : (x-5 = 0)) (h2 : (6*x - 12 ≠ 0)) :
  x = 5 :=
by {
  sorry
}

end value_of_x_for_zero_expression_l977_97739


namespace cars_cleaned_per_day_l977_97719

theorem cars_cleaned_per_day
  (money_per_car : ℕ)
  (total_money : ℕ)
  (days : ℕ)
  (h1 : money_per_car = 5)
  (h2 : total_money = 2000)
  (h3 : days = 5) :
  (total_money / (money_per_car * days)) = 80 := by
  sorry

end cars_cleaned_per_day_l977_97719


namespace paving_stone_width_l977_97708

theorem paving_stone_width
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_length : ℝ)
  (courtyard_area : ℝ) (paving_stone_area : ℝ)
  (width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16.5)
  (h3 : num_paving_stones = 99)
  (h4 : paving_stone_length = 2.5)
  (h5 : courtyard_area = courtyard_length * courtyard_width)
  (h6 : courtyard_area = 495)
  (h7 : paving_stone_area = courtyard_area / num_paving_stones)
  (h8 : paving_stone_area = 5)
  (h9 : paving_stone_area = paving_stone_length * width) :
  width = 2 := by
  sorry

end paving_stone_width_l977_97708


namespace simplify_expression_l977_97791

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (x⁻¹ - x + 2) = (1 - (x - 1)^2) / x := 
sorry

end simplify_expression_l977_97791


namespace factor_expression_l977_97752

theorem factor_expression (b : ℤ) : 
  (8 * b ^ 3 + 120 * b ^ 2 - 14) - (9 * b ^ 3 - 2 * b ^ 2 + 14) 
  = -1 * (b ^ 3 - 122 * b ^ 2 + 28) := 
by {
  sorry
}

end factor_expression_l977_97752


namespace number_of_digits_in_product_l977_97733

open Nat

noncomputable def num_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log 10 n + 1

def compute_product : ℕ := 234567 * 123^3

theorem number_of_digits_in_product : num_digits compute_product = 13 := by 
  sorry

end number_of_digits_in_product_l977_97733


namespace sum_first_n_abs_terms_arithmetic_seq_l977_97727

noncomputable def sum_abs_arithmetic_sequence (n : ℕ) (h : n ≥ 3) : ℚ :=
  if n = 1 ∨ n = 2 then (n * (4 + 7 - 3 * n)) / 2
  else (3 * n^2 - 11 * n + 20) / 2

theorem sum_first_n_abs_terms_arithmetic_seq (n : ℕ) (h : n ≥ 3) :
  sum_abs_arithmetic_sequence n h = (3 * n^2) / 2 - (11 * n) / 2 + 10 :=
sorry

end sum_first_n_abs_terms_arithmetic_seq_l977_97727


namespace num_ways_to_factor_2210_l977_97707

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_ways_to_factor_2210 : ∃! (a b : ℕ), a * b = 2210 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end num_ways_to_factor_2210_l977_97707


namespace dayan_sequence_20th_term_l977_97716

theorem dayan_sequence_20th_term (a : ℕ → ℕ) (h1 : a 0 = 0)
    (h2 : a 1 = 2) (h3 : a 2 = 4) (h4 : a 3 = 8) (h5 : a 4 = 12)
    (h6 : a 5 = 18) (h7 : a 6 = 24) (h8 : a 7 = 32) (h9 : a 8 = 40) (h10 : a 9 = 50)
    (h_even : ∀ n : ℕ, a (2 * n) = 2 * n^2) :
  a 20 = 200 :=
  sorry

end dayan_sequence_20th_term_l977_97716


namespace subscriptions_sold_to_parents_l977_97798

-- Definitions for the conditions
variable (P : Nat) -- subscriptions sold to parents
def grandfather := 1
def next_door_neighbor := 2
def other_neighbor := 2 * next_door_neighbor
def subscriptions_other_than_parents := grandfather + next_door_neighbor + other_neighbor
def total_earnings := 55
def earnings_from_others := 5 * subscriptions_other_than_parents
def earnings_from_parents := total_earnings - earnings_from_others
def subscription_price := 5

-- Theorem stating the equivalent math proof
theorem subscriptions_sold_to_parents : P = earnings_from_parents / subscription_price :=
by
  sorry

end subscriptions_sold_to_parents_l977_97798


namespace inequality_of_reals_l977_97714

theorem inequality_of_reals (a b c d : ℝ) : 
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := 
  sorry

end inequality_of_reals_l977_97714


namespace number_consisting_of_11_hundreds_11_tens_and_11_units_l977_97776

theorem number_consisting_of_11_hundreds_11_tens_and_11_units :
  11 * 100 + 11 * 10 + 11 = 1221 :=
by
  sorry

end number_consisting_of_11_hundreds_11_tens_and_11_units_l977_97776


namespace find_a_minus_c_l977_97796

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 80) (h2 : (b + c) / 2 = 180) : a - c = -200 :=
by 
  sorry

end find_a_minus_c_l977_97796


namespace fish_weight_l977_97726

theorem fish_weight (W : ℝ) (h : W = 2 + W / 3) : W = 3 :=
by
  sorry

end fish_weight_l977_97726


namespace jack_lap_time_improvement_l977_97783

/-!
Jack practices running in a stadium. Initially, he completed 15 laps in 45 minutes.
After a month of training, he completed 18 laps in 42 minutes. By how many minutes 
has he improved his lap time?
-/

theorem jack_lap_time_improvement:
  ∀ (initial_laps current_laps : ℕ) 
    (initial_time current_time : ℝ), 
    initial_laps = 15 → 
    current_laps = 18 → 
    initial_time = 45 → 
    current_time = 42 → 
    (initial_time / initial_laps - current_time / current_laps = 2/3) :=
by 
  intros _ _ _ _ h_initial_laps h_current_laps h_initial_time h_current_time
  rw [h_initial_laps, h_current_laps, h_initial_time, h_current_time]
  sorry

end jack_lap_time_improvement_l977_97783


namespace sum_arithmetic_sequence_l977_97734

theorem sum_arithmetic_sequence {a : ℕ → ℤ} (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 13 = S 2000 →
  S 2013 = 0 :=
by
  sorry

end sum_arithmetic_sequence_l977_97734


namespace tangent_line_exists_l977_97781

noncomputable def tangent_line_problem := ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  Int.gcd (Int.gcd a b) c = 1 ∧ 
  (∀ x y : ℝ, a * x + b * (x^2 + 52 / 25) = c ∧ a * (y^2 + 81 / 16) + b * y = c) ∧ 
  a + b + c = 168

theorem tangent_line_exists : tangent_line_problem := by
  sorry

end tangent_line_exists_l977_97781


namespace find_p_l977_97730

noncomputable def f (p : ℝ) : ℝ := 2 * p^2 + 20 * Real.sin p

theorem find_p : ∃ p : ℝ, f (f (f (f p))) = -4 :=
by
  sorry

end find_p_l977_97730


namespace complex_numbers_equation_l977_97790

theorem complex_numbers_equation {a b : ℂ} (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := 
by sorry

end complex_numbers_equation_l977_97790


namespace apples_in_first_group_l977_97763

variable (A O : ℝ) (X : ℕ)

-- Given conditions
axiom h1 : A = 0.21
axiom h2 : X * A + 3 * O = 1.77
axiom h3 : 2 * A + 5 * O = 1.27 

-- Goal: Prove that the number of apples in the first group is 6
theorem apples_in_first_group : X = 6 := 
by 
  sorry

end apples_in_first_group_l977_97763


namespace equilateral_triangle_l977_97749

theorem equilateral_triangle
  (A B C : Type)
  (angle_A : ℝ)
  (side_BC : ℝ)
  (perimeter : ℝ)
  (h1 : angle_A = 60)
  (h2 : side_BC = 1/3 * perimeter)
  (side_AB : ℝ)
  (side_AC : ℝ)
  (h3 : perimeter = side_BC + side_AB + side_AC) :
  (side_AB = side_BC) ∧ (side_AC = side_BC) :=
by
  sorry

end equilateral_triangle_l977_97749


namespace find_base_b4_l977_97742

theorem find_base_b4 (b_4 : ℕ) : (b_4 - 1) * (b_4 - 2) * (b_4 - 3) = 168 → b_4 = 8 :=
by
  intro h
  -- proof goes here
  sorry

end find_base_b4_l977_97742


namespace rationalize_denominator_l977_97706

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
  (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11)) = 
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
  A + B + C + D + E + F = 136 := 
sorry

end rationalize_denominator_l977_97706


namespace all_points_below_line_l977_97770

theorem all_points_below_line (a b : ℝ) (n : ℕ) (x y : ℕ → ℝ)
  (h1 : b > a)
  (h2 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k = a + ((k : ℝ) * (b - a) / (n + 1)))
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k = a * (b / a) ^ (k / (n + 1))) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k < x k := 
sorry

end all_points_below_line_l977_97770


namespace problem1_range_problem2_range_l977_97715

theorem problem1_range (x y : ℝ) (h : y = 2*|x-1| - |x-4|) : -3 ≤ y := sorry

theorem problem2_range (x a : ℝ) (h : ∀ x, 2*|x-1| - |x-a| ≥ -1) : 0 ≤ a ∧ a ≤ 2 := sorry

end problem1_range_problem2_range_l977_97715


namespace range_of_k_l977_97701

theorem range_of_k 
  (h : ∀ x : ℝ, x^2 + 2 * k * x - (k - 2) > 0) : -2 < k ∧ k < 1 := 
sorry

end range_of_k_l977_97701


namespace part1_part2_l977_97795

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x - 4 ≤ 0}

-- Problem 1
theorem part1 (m : ℝ) : 
  (A ∩ B m = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) → m = 3 :=
by sorry

-- Problem 2
theorem part2 (m : ℝ) : 
  (A ⊆ (B m)ᶜ) → (m < -3 ∨ m > 5) :=
by sorry

end part1_part2_l977_97795


namespace f_strictly_increasing_solve_inequality_l977_97710

variable (f : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 3

-- Prove monotonicity
theorem f_strictly_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Solve the inequality
theorem solve_inequality (m : ℝ) : -2/3 < m ∧ m < 2 ↔ f (3 * m^2 - m - 2) < 2 := by
  sorry

end f_strictly_increasing_solve_inequality_l977_97710


namespace combined_time_l977_97718

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l977_97718


namespace smallest_positive_period_max_value_in_interval_min_value_in_interval_l977_97753

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem smallest_positive_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = 2 := by
  sorry

theorem min_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = -1 := by
  sorry

end smallest_positive_period_max_value_in_interval_min_value_in_interval_l977_97753


namespace Okeydokey_should_receive_25_earthworms_l977_97784

def applesOkeydokey : ℕ := 5
def applesArtichokey : ℕ := 7
def totalEarthworms : ℕ := 60
def totalApples : ℕ := applesOkeydokey + applesArtichokey
def okeydokeyProportion : ℚ := applesOkeydokey / totalApples
def okeydokeyEarthworms : ℚ := okeydokeyProportion * totalEarthworms

theorem Okeydokey_should_receive_25_earthworms : okeydokeyEarthworms = 25 := by
  sorry

end Okeydokey_should_receive_25_earthworms_l977_97784


namespace developer_break_even_price_l977_97772

theorem developer_break_even_price :
  let acres := 4
  let cost_per_acre := 1863
  let total_cost := acres * cost_per_acre
  let num_lots := 9
  let cost_per_lot := total_cost / num_lots
  cost_per_lot = 828 :=
by {
  sorry  -- This is where the proof would go.
} 

end developer_break_even_price_l977_97772


namespace combined_total_value_of_items_l977_97768

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end combined_total_value_of_items_l977_97768


namespace walls_per_room_is_8_l977_97767

-- Definitions and conditions
def total_rooms : Nat := 10
def green_rooms : Nat := 3 * total_rooms / 5
def purple_rooms : Nat := total_rooms - green_rooms
def purple_walls : Nat := 32
def walls_per_room : Nat := purple_walls / purple_rooms

-- Theorem to prove
theorem walls_per_room_is_8 : walls_per_room = 8 := by
  sorry

end walls_per_room_is_8_l977_97767


namespace inequality_holds_for_all_x_y_l977_97793

theorem inequality_holds_for_all_x_y (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ x + y + x * y := 
by sorry

end inequality_holds_for_all_x_y_l977_97793


namespace katie_pink_marbles_l977_97725

-- Define variables for the problem
variables (P O R : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  O = P - 9 ∧
  R = 4 * (P - 9) ∧
  P + O + R = 33

-- Desired result
def result : Prop :=
  P = 13

-- Proof statement
theorem katie_pink_marbles : conditions P O R → result P :=
by
  intros h
  sorry

end katie_pink_marbles_l977_97725


namespace total_students_in_lab_l977_97779

def total_workstations : Nat := 16
def workstations_for_2_students : Nat := 10
def students_per_workstation_2 : Nat := 2
def students_per_workstation_3 : Nat := 3

theorem total_students_in_lab :
  let workstations_with_2_students := workstations_for_2_students
  let workstations_with_3_students := total_workstations - workstations_for_2_students
  let students_in_2_student_workstations := workstations_with_2_students * students_per_workstation_2
  let students_in_3_student_workstations := workstations_with_3_students * students_per_workstation_3
  students_in_2_student_workstations + students_in_3_student_workstations = 38 :=
by
  sorry

end total_students_in_lab_l977_97779


namespace largest_element_sum_of_digits_in_E_l977_97758
open BigOperators
open Nat

def E : Set ℕ := { n | ∃ (r₉ r₁₀ r₁₁ : ℕ), 0 < r₉ ∧ r₉ ≤ 9 ∧ 0 < r₁₀ ∧ r₁₀ ≤ 10 ∧ 0 < r₁₁ ∧ r₁₁ ≤ 11 ∧
  r₉ = n % 9 ∧ r₁₀ = n % 10 ∧ r₁₁ = n % 11 ∧
  (r₉ > 1) ∧ (r₁₀ > 1) ∧ (r₁₁ > 1) ∧
  ∃ (a : ℕ) (b : ℕ) (c : ℕ), r₉ = a ∧ r₁₀ = a * b ∧ r₁₁ = a * b * c ∧ b ≠ 1 ∧ c ≠ 1 }

noncomputable def N : ℕ := 
  max (max (74 % 990) (134 % 990)) (526 % 990)

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem largest_element_sum_of_digits_in_E :
  sum_of_digits N = 13 :=
sorry

end largest_element_sum_of_digits_in_E_l977_97758


namespace girls_maple_grove_correct_l977_97722

variables (total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge : ℕ)
variables (girls_maple_grove : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 150 ∧ 
  boys = 82 ∧ 
  girls = 68 ∧ 
  pine_ridge_students = 70 ∧ 
  maple_grove_students = 80 ∧ 
  boys_pine_ridge = 36 ∧ 
  girls_maple_grove = girls - (pine_ridge_students - boys_pine_ridge)

-- Question and Answer translated to a proposition
def proof_problem : Prop :=
  conditions total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove → 
  girls_maple_grove = 34

-- Statement
theorem girls_maple_grove_correct : proof_problem total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove :=
by {
  sorry -- Proof omitted
}

end girls_maple_grove_correct_l977_97722


namespace arithmetic_sequence_m_value_l977_97797

theorem arithmetic_sequence_m_value 
  (a : ℕ → ℝ) (d : ℝ) (h₁ : d ≠ 0) 
  (h₂ : a 3 + a 6 + a 10 + a 13 = 32) 
  (m : ℕ) (h₃ : a m = 8) : 
  m = 8 :=
sorry

end arithmetic_sequence_m_value_l977_97797


namespace find_a_14_l977_97782

variable {α : Type} [LinearOrderedField α]

-- Define the arithmetic sequence sum formula
def arithmetic_seq_sum (a_1 d : α) (n : ℕ) : α :=
  n * a_1 + n * (n - 1) / 2 * d

-- Define the nth term of an arithmetic sequence
def arithmetic_seq_nth (a_1 d : α) (n : ℕ) : α :=
  a_1 + (n - 1 : ℕ) * d

theorem find_a_14
  (a_1 d : α)
  (h1 : arithmetic_seq_sum a_1 d 11 = 55)
  (h2 : arithmetic_seq_nth a_1 d 10 = 9) :
  arithmetic_seq_nth a_1 d 14 = 13 :=
by
  sorry

end find_a_14_l977_97782


namespace quadratic_roots_l977_97780

theorem quadratic_roots (x : ℝ) : 
  (2 * x^2 - 4 * x - 5 = 0) ↔ 
  (x = (2 + Real.sqrt 14) / 2 ∨ x = (2 - Real.sqrt 14) / 2) :=
by
  sorry

end quadratic_roots_l977_97780


namespace total_goals_is_50_l977_97777

def team_a_first_half_goals := 8
def team_b_first_half_goals := team_a_first_half_goals / 2
def team_c_first_half_goals := 2 * team_b_first_half_goals
def team_a_first_half_missed_penalty := 1
def team_c_first_half_missed_penalty := 2

def team_a_second_half_goals := team_c_first_half_goals
def team_b_second_half_goals := team_a_first_half_goals
def team_c_second_half_goals := team_b_second_half_goals + 3
def team_a_second_half_successful_penalty := 1
def team_b_second_half_successful_penalty := 2

def total_team_a_goals := team_a_first_half_goals + team_a_second_half_goals + team_a_second_half_successful_penalty
def total_team_b_goals := team_b_first_half_goals + team_b_second_half_goals + team_b_second_half_successful_penalty
def total_team_c_goals := team_c_first_half_goals + team_c_second_half_goals

def total_goals := total_team_a_goals + total_team_b_goals + total_team_c_goals

theorem total_goals_is_50 : total_goals = 50 := by
  unfold total_goals
  unfold total_team_a_goals total_team_b_goals total_team_c_goals
  unfold team_a_first_half_goals team_b_first_half_goals team_c_first_half_goals
  unfold team_a_second_half_goals team_b_second_half_goals team_c_second_half_goals
  unfold team_a_second_half_successful_penalty team_b_second_half_successful_penalty
  sorry

end total_goals_is_50_l977_97777


namespace find_innings_l977_97745

noncomputable def calculate_innings (A : ℕ) (n : ℕ) : Prop :=
  (n * A + 140 = (n + 1) * (A + 8)) ∧ (A + 8 = 28)

theorem find_innings (n : ℕ) (A : ℕ) :
  calculate_innings A n → n = 14 :=
by
  intros h
  -- Here you would prove that h implies n = 14, but we use sorry to skip the proof steps.
  sorry

end find_innings_l977_97745


namespace fraction_meaningful_range_l977_97711

theorem fraction_meaningful_range (x : ℝ) : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l977_97711


namespace area_shaded_region_in_hexagon_l977_97794

theorem area_shaded_region_in_hexagon (s : ℝ) (r : ℝ) (h_s : s = 4) (h_r : r = 2) :
  let area_hexagon := ((3 * Real.sqrt 3) / 2) * s^2
  let area_semicircle := (π * r^2) / 2
  let total_area_semicircles := 8 * area_semicircle
  let area_shaded_region := area_hexagon - total_area_semicircles
  area_shaded_region = 24 * Real.sqrt 3 - 16 * π :=
by {
  sorry
}

end area_shaded_region_in_hexagon_l977_97794


namespace order_of_abc_l977_97764

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l977_97764


namespace range_of_a_in_fourth_quadrant_l977_97704

-- Define the fourth quadrant condition
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Define the point P(a+1, a-1) and state the theorem
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  in_fourth_quadrant (a + 1) (a - 1) → -1 < a ∧ a < 1 :=
by
  intro h
  have h1 : a + 1 > 0 := h.1
  have h2 : a - 1 < 0 := h.2
  have h3 : a > -1 := by linarith
  have h4 : a < 1 := by linarith
  exact ⟨h3, h4⟩

end range_of_a_in_fourth_quadrant_l977_97704


namespace seashells_total_now_l977_97792

def henry_collected : ℕ := 11
def paul_collected : ℕ := 24
def total_initial : ℕ := 59
def leo_initial (henry_collected paul_collected total_initial : ℕ) : ℕ := total_initial - (henry_collected + paul_collected)
def leo_gave (leo_initial : ℕ) : ℕ := leo_initial / 4
def total_now (total_initial leo_gave : ℕ) : ℕ := total_initial - leo_gave

theorem seashells_total_now :
  total_now total_initial (leo_gave (leo_initial henry_collected paul_collected total_initial)) = 53 :=
sorry

end seashells_total_now_l977_97792


namespace problem1_problem2_l977_97755

-- Definitions of the sets A and B
def set_A (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4
def set_B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

-- Problem 1: If A ∩ B ≠ ∅, find the range of a
theorem problem1 (a : ℝ) : (∃ x : ℝ, set_A x ∧ set_B x a) → a ≤ -1 / 2 ∨ a = 2 :=
sorry

-- Problem 2: If A ∩ B = B, find the value of a
theorem problem2 (a : ℝ) : (∀ x : ℝ, set_B x a → set_A x) → a ≤ -1 / 2 ∨ a ≥ 2 :=
sorry

end problem1_problem2_l977_97755


namespace swimming_pool_radius_l977_97735

theorem swimming_pool_radius 
  (r : ℝ)
  (h1 : ∀ (r : ℝ), r > 0)
  (h2 : π * (r + 4)^2 - π * r^2 = (11 / 25) * π * r^2) :
  r = 20 := 
sorry

end swimming_pool_radius_l977_97735


namespace reduction_percentage_40_l977_97744

theorem reduction_percentage_40 (P : ℝ) : 
  1500 * 1.20 - (P / 100 * (1500 * 1.20)) = 1080 ↔ P = 40 :=
by
  sorry

end reduction_percentage_40_l977_97744


namespace square_window_side_length_is_24_l977_97760

noncomputable def side_length_square_window
  (num_panes_per_row : ℕ) (pane_height_ratio : ℝ) (border_width : ℝ) (x : ℝ) : ℝ :=
  num_panes_per_row * x + (num_panes_per_row + 1) * border_width

theorem square_window_side_length_is_24
  (num_panes_per_row : ℕ)
  (pane_height_ratio : ℝ)
  (border_width : ℝ) 
  (pane_width : ℝ)
  (pane_height : ℝ)
  (window_side_length : ℝ) : 
  (num_panes_per_row = 3) →
  (pane_height_ratio = 3) →
  (border_width = 3) →
  (pane_height = pane_height_ratio * pane_width) →
  (window_side_length = side_length_square_window num_panes_per_row pane_height_ratio border_width pane_width) →
  (window_side_length = 24) :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end square_window_side_length_is_24_l977_97760


namespace monotonicity_of_f_inequality_of_f_l977_97702

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

theorem monotonicity_of_f {a : ℝ}:
(a ≥ 0 → ∀ x y : ℝ, 0 < x ∧ x < y → f x a ≤ f y a) ∧
(a < 0 → ∀ x y : ℝ, 0 < x ∧ x < y ∧ x ≥ -1 + Real.sqrt (1 - 2 * a) → f x a ≤ f y a 
∨ 0 < x ∧ x < -1 + Real.sqrt (1 - 2 * a) → f x a ≥ f y a) := sorry

theorem inequality_of_f {a : ℝ} (h : t ≥ 1) :
(f (2*t-1) a ≥ 2 * f t a - 3) ↔ (a ≤ 2) := sorry

end monotonicity_of_f_inequality_of_f_l977_97702


namespace number_of_dogs_l977_97766

-- Define variables for the number of cats (C) and dogs (D)
variables (C D : ℕ)

-- Define the conditions from the problem statement
def condition1 : Prop := C = D - 6
def condition2 : Prop := C * 3 = D * 2

-- State the theorem that D should be 18 given the conditions
theorem number_of_dogs (h1 : condition1 C D) (h2 : condition2 C D) : D = 18 :=
  sorry

end number_of_dogs_l977_97766


namespace janet_lunch_cost_l977_97754

theorem janet_lunch_cost 
  (num_children : ℕ) (num_chaperones : ℕ) (janet : ℕ) (extra_lunches : ℕ) (cost_per_lunch : ℕ) : 
  num_children = 35 → num_chaperones = 5 → janet = 1 → extra_lunches = 3 → cost_per_lunch = 7 → 
  cost_per_lunch * (num_children + num_chaperones + janet + extra_lunches) = 308 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_lunch_cost_l977_97754


namespace smaller_number_is_270_l977_97746

theorem smaller_number_is_270 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 :=
sorry

end smaller_number_is_270_l977_97746


namespace tangent_x_axis_l977_97728

noncomputable def curve (k : ℝ) : ℝ → ℝ := λ x => Real.log x - k * x + 3

theorem tangent_x_axis (k : ℝ) : 
  ∃ t : ℝ, curve k t = 0 ∧ deriv (curve k) t = 0 → k = Real.exp 2 :=
by
  sorry

end tangent_x_axis_l977_97728


namespace triangle_acute_angle_exists_l977_97712

theorem triangle_acute_angle_exists (a b c d e : ℝ)
  (h_abc : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_abd : a + b > d ∧ a + d > b ∧ b + d > a)
  (h_abe : a + b > e ∧ a + e > b ∧ b + e > a)
  (h_acd : a + c > d ∧ a + d > c ∧ c + d > a)
  (h_ace : a + c > e ∧ a + e > c ∧ c + e > a)
  (h_ade : a + d > e ∧ a + e > d ∧ d + e > a)
  (h_bcd : b + c > d ∧ b + d > c ∧ c + d > b)
  (h_bce : b + c > e ∧ b + e > c ∧ c + e > b)
  (h_bde : b + d > e ∧ b + e > d ∧ d + e > b)
  (h_cde : c + d > e ∧ c + e > d ∧ d + e > c) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
           x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
           x + y > z ∧ x + z > y ∧ y + z > x ∧
           (x * x + y * y > z * z ∧ y * y + z * z > x * x ∧ z * z + x * x > y * y) := 
sorry

end triangle_acute_angle_exists_l977_97712


namespace x_add_one_greater_than_x_l977_97769

theorem x_add_one_greater_than_x (x : ℝ) : x + 1 > x :=
by
  sorry

end x_add_one_greater_than_x_l977_97769


namespace x_plus_y_eq_20_l977_97748

theorem x_plus_y_eq_20 (x y : ℝ) (hxy : x ≠ y) (hdet : (Matrix.det ![
  ![2, 3, 7],
  ![4, x, y],
  ![4, y, x]]) = 0) : x + y = 20 :=
by
  sorry

end x_plus_y_eq_20_l977_97748


namespace find_h_l977_97751

def infinite_sqrt_series (b : ℝ) : ℝ := sorry -- Placeholder for infinite series sqrt(b + sqrt(b + ...))

def diamond (a b : ℝ) : ℝ :=
  a^2 + infinite_sqrt_series b

theorem find_h (h : ℝ) : diamond 3 h = 12 → h = 6 :=
by
  intro h_condition
  -- Further steps will be used during proof
  sorry

end find_h_l977_97751


namespace total_wet_surface_area_l977_97724

def length : ℝ := 8
def width : ℝ := 4
def depth : ℝ := 1.25

theorem total_wet_surface_area : length * width + 2 * (length * depth) + 2 * (width * depth) = 62 :=
by
  sorry

end total_wet_surface_area_l977_97724


namespace sixth_graders_forgot_homework_percentage_l977_97759

-- Definitions of the conditions
def num_students_A : ℕ := 20
def num_students_B : ℕ := 80
def percent_forgot_A : ℚ := 20 / 100
def percent_forgot_B : ℚ := 15 / 100

-- Statement to be proven
theorem sixth_graders_forgot_homework_percentage :
  (num_students_A * percent_forgot_A + num_students_B * percent_forgot_B) /
  (num_students_A + num_students_B) = 16 / 100 :=
by
  sorry

end sixth_graders_forgot_homework_percentage_l977_97759


namespace charity_tickets_l977_97788

theorem charity_tickets (f h p : ℕ) (H1 : f + h = 140) (H2 : f * p + h * (p / 2) = 2001) : f * p = 782 := 
sorry

end charity_tickets_l977_97788


namespace predicted_holiday_shoppers_l977_97723

-- Conditions
def packages_per_bulk_box : Nat := 25
def every_third_shopper_buys_package : Nat := 3
def bulk_boxes_ordered : Nat := 5

-- Number of predicted holiday shoppers
theorem predicted_holiday_shoppers (pbb : packages_per_bulk_box = 25)
                                   (etsbp : every_third_shopper_buys_package = 3)
                                   (bbo : bulk_boxes_ordered = 5) :
  (bulk_boxes_ordered * packages_per_bulk_box * every_third_shopper_buys_package) = 375 :=
by 
  -- Proof steps can be added here
  sorry

end predicted_holiday_shoppers_l977_97723


namespace minimum_value_of_expression_l977_97761

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value_expr a b = -2031948.5 :=
  sorry

end minimum_value_of_expression_l977_97761


namespace car_travels_more_l977_97729

theorem car_travels_more (train_speed : ℕ) (car_speed : ℕ) (time : ℕ)
  (h1 : train_speed = 60)
  (h2 : car_speed = 2 * train_speed)
  (h3 : time = 3) :
  car_speed * time - train_speed * time = 180 :=
by
  sorry

end car_travels_more_l977_97729


namespace litter_patrol_total_l977_97713

theorem litter_patrol_total (glass_bottles : Nat) (aluminum_cans : Nat) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 :=
by
  sorry

end litter_patrol_total_l977_97713


namespace D_is_quadratic_l977_97743

-- Define the equations
def eq_A (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq_B (x : ℝ) : Prop := 2 * x^2 - 3 * x = 2 * (x^2 - 2)
def eq_C (x : ℝ) : Prop := x^3 - 2 * x + 7 = 0
def eq_D (x : ℝ) : Prop := (x - 2)^2 - 4 = 0

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x ↔ a * x^2 + b * x + c = 0)

theorem D_is_quadratic : is_quadratic eq_D :=
sorry

end D_is_quadratic_l977_97743


namespace multiply_exponents_l977_97789

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end multiply_exponents_l977_97789


namespace lowest_score_85_avg_l977_97757

theorem lowest_score_85_avg (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 = 79) (h2 : a2 = 88) (h3 : a3 = 94) 
  (h4 : a4 = 91) (h5 : 75 ≤ a5) (h6 : 75 ≤ a6) 
  (h7 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 85) : (a5 = 75 ∨ a6 = 75) ∧ (a5 = 75 ∨ a5 > 75) := 
by
  sorry

end lowest_score_85_avg_l977_97757


namespace find_smallest_d_l977_97778

noncomputable def smallest_possible_d (c d : ℕ) : ℕ :=
  if c - d = 8 ∧ Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 then d else 0

-- Proving the smallest possible value of d given the conditions
theorem find_smallest_d :
  ∀ c d : ℕ, (0 < c) → (0 < d) → (c - d = 8) → 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 → d = 4 :=
by
  sorry

end find_smallest_d_l977_97778


namespace complement_union_l977_97738

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  U \ (A ∪ B) = {4} :=
by
  sorry

end complement_union_l977_97738


namespace find_x_when_y_30_l977_97731

variable (x y k : ℝ)

noncomputable def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, x * y = k

theorem find_x_when_y_30
  (h_inv_prop : inversely_proportional x y) 
  (h_known_values : x = 5 ∧ y = 15) :
  ∃ x : ℝ, (∃ y : ℝ, y = 30) ∧ x = 5 / 2 := by
  sorry

end find_x_when_y_30_l977_97731


namespace manuscript_pages_l977_97721

theorem manuscript_pages (P : ℕ)
  (h1 : 30 = 30)
  (h2 : 20 = 20)
  (h3 : 50 = 30 + 20)
  (h4 : 710 = 5 * (P - 50) + 30 * 8 + 20 * 11) :
  P = 100 :=
by
  sorry

end manuscript_pages_l977_97721


namespace ratio_Ryn_Nikki_l977_97732

def Joyce_movie_length (M : ℝ) : ℝ := M + 2
def Nikki_movie_length (M : ℝ) : ℝ := 3 * M
def Ryn_movie_fraction (F : ℝ) (Nikki_movie_length : ℝ) : ℝ := F * Nikki_movie_length

theorem ratio_Ryn_Nikki 
  (M : ℝ) 
  (Nikki_movie_is_30 : Nikki_movie_length M = 30) 
  (total_movie_hours_is_76 : M + Joyce_movie_length M + Nikki_movie_length M + Ryn_movie_fraction F (Nikki_movie_length M) = 76) 
  : F = 4 / 5 := 
by 
  sorry

end ratio_Ryn_Nikki_l977_97732


namespace statues_painted_l977_97717

-- Definitions based on the conditions provided in the problem
def paint_remaining : ℚ := 1/2
def paint_per_statue : ℚ := 1/4

-- The theorem that answers the question
theorem statues_painted (h : paint_remaining = 1/2 ∧ paint_per_statue = 1/4) : 
  (paint_remaining / paint_per_statue) = 2 := 
sorry

end statues_painted_l977_97717


namespace solve_g_l977_97765

def g (a b : ℚ) : ℚ :=
if a + b ≤ 4 then (a * b - 2 * a + 3) / (3 * a)
else (a * b - 3 * b - 1) / (-3 * b)

theorem solve_g :
  g 3 1 + g 1 5 = 11 / 15 :=
by
  -- Here we just set up the theorem statement. Proof is not included.
  sorry

end solve_g_l977_97765


namespace complete_the_square_d_l977_97786

theorem complete_the_square_d (x : ℝ) :
  ∃ c d, (x^2 + 10 * x + 9 = 0 → (x + c)^2 = d) ∧ d = 16 :=
sorry

end complete_the_square_d_l977_97786


namespace part1_tangent_line_part2_monotonicity_l977_97736

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (x ^ 2 - 2 * a * x) * Real.log x - x ^ 2 + 4 * a * x + 1

theorem part1_tangent_line (a : ℝ) (h : a = 0) :
  let e := Real.exp 1
  let f_x := f e 0
  let tangent_line := 4 * e - 3 * e ^ 2 + 1
  tangent_line = 4 * e * (x - e) + f_x :=
sorry

theorem part2_monotonicity (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → f (x) a > 0 ↔ a ≤ 0) ∧
  (∀ x : ℝ, 0 < x → x < a → f (x) a > 0 ↔ 0 < a ∧ a < 1) ∧
  (∀ x : ℝ, 1 < x → x < a → f (x) a < 0 ↔ a > 1) ∧
  (∀ x : ℝ, 0 < x → 1 < x → x < a → f (x) a < 0 ↔ (a > 1)) ∧
  (∀ x : ℝ, x > 1 → f (x) a > 0 ↔ (a < 1)) :=
sorry

end part1_tangent_line_part2_monotonicity_l977_97736
