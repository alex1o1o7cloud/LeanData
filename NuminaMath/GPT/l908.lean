import Mathlib

namespace NUMINAMATH_GPT_find_p_l908_90880

noncomputable def f (p : ℝ) : ℝ := 2 * p - 20

theorem find_p : (f ∘ f ∘ f) p = 6 → p = 18.25 := by
  sorry

end NUMINAMATH_GPT_find_p_l908_90880


namespace NUMINAMATH_GPT_exists_duplicate_parenthesizations_l908_90807

def expr : List Int := List.range' 1 (1991 + 1)

def num_parenthesizations : Nat := 2 ^ 995

def num_distinct_results : Nat := 3966067

theorem exists_duplicate_parenthesizations :
  num_parenthesizations > num_distinct_results :=
sorry

end NUMINAMATH_GPT_exists_duplicate_parenthesizations_l908_90807


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_13_l908_90815

theorem smallest_four_digit_multiple_of_13 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 13 = 0) ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 13 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_13_l908_90815


namespace NUMINAMATH_GPT_range_of_f_t_l908_90852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (Real.exp x) + Real.log x - x

theorem range_of_f_t (a : ℝ) (t : ℝ) 
  (h_unique_critical : ∀ x, f a x = 0 → x = t) : 
  ∃ y : ℝ, y ≥ -2 ∧ ∀ z : ℝ, y = f a t :=
sorry

end NUMINAMATH_GPT_range_of_f_t_l908_90852


namespace NUMINAMATH_GPT_expression_evaluation_l908_90857

theorem expression_evaluation : 
  54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l908_90857


namespace NUMINAMATH_GPT_y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l908_90884

def y : ℕ := 36 + 48 + 72 + 144 + 216 + 432 + 1296

theorem y_is_multiple_of_12 : y % 12 = 0 := by
  sorry

theorem y_is_multiple_of_3 : y % 3 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_4 : y % 4 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_6 : y % 6 = 0 := by
  have h := y_is_multiple_of_12
  sorry

end NUMINAMATH_GPT_y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l908_90884


namespace NUMINAMATH_GPT_largest_n_employees_in_same_quarter_l908_90801

theorem largest_n_employees_in_same_quarter (n : ℕ) (h1 : 72 % 4 = 0) (h2 : 72 / 4 = 18) : 
  n = 18 :=
sorry

end NUMINAMATH_GPT_largest_n_employees_in_same_quarter_l908_90801


namespace NUMINAMATH_GPT_find_special_two_digit_numbers_l908_90832

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_special (A : ℕ) : Prop :=
  let sum_A := sum_digits A
  sum_A^2 = sum_digits (A^2)

theorem find_special_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A < 100 ∧ is_special A} = {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by 
  sorry

end NUMINAMATH_GPT_find_special_two_digit_numbers_l908_90832


namespace NUMINAMATH_GPT_coins_in_stack_l908_90859

-- Define the thickness of each coin type
def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75

-- Define the total stack height
def total_stack_height : ℝ := 15

-- The statement to prove
theorem coins_in_stack (pennies nickels dimes quarters : ℕ) :
  pennies * penny_thickness + nickels * nickel_thickness + 
  dimes * dime_thickness + quarters * quarter_thickness = total_stack_height →
  pennies + nickels + dimes + quarters = 9 :=
sorry

end NUMINAMATH_GPT_coins_in_stack_l908_90859


namespace NUMINAMATH_GPT_temperature_on_April_15_and_19_l908_90835

/-
We define the daily temperatures as functions of the temperature on April 15 (T_15) with the given increment of 1.5 degrees each day. 
T_15 represents the temperature on April 15.
-/
theorem temperature_on_April_15_and_19 (T : ℕ → ℝ) (T_avg : ℝ) (inc : ℝ) 
  (h1 : inc = 1.5)
  (h2 : T_avg = 17.5)
  (h3 : ∀ n, T (15 + n) = T 15 + inc * n)
  (h4 : (T 15 + T 16 + T 17 + T 18 + T 19) / 5 = T_avg) :
  T 15 = 14.5 ∧ T 19 = 20.5 :=
by
  sorry

end NUMINAMATH_GPT_temperature_on_April_15_and_19_l908_90835


namespace NUMINAMATH_GPT_lisa_pizza_l908_90808

theorem lisa_pizza (P H S : ℕ) 
  (h1 : H = 2 * P) 
  (h2 : S = P + 12) 
  (h3 : P + H + S = 132) : 
  P = 30 := 
by
  sorry

end NUMINAMATH_GPT_lisa_pizza_l908_90808


namespace NUMINAMATH_GPT_determine_b_l908_90843

theorem determine_b (b : ℤ) : (x - 5) ∣ (x^3 + 3 * x^2 + b * x + 5) → b = -41 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l908_90843


namespace NUMINAMATH_GPT_Louisa_average_speed_l908_90828

theorem Louisa_average_speed : 
  ∀ (v : ℝ), (∀ v, (160 / v) + 3 = (280 / v)) → v = 40 :=
by
  intros v h
  sorry

end NUMINAMATH_GPT_Louisa_average_speed_l908_90828


namespace NUMINAMATH_GPT_no_prime_factor_congruent_to_7_mod_8_l908_90817

open Nat

theorem no_prime_factor_congruent_to_7_mod_8 (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ p : ℕ, p.Prime ∧ p ∣ 2^n + 1 ∧ p % 8 = 7) :=
sorry

end NUMINAMATH_GPT_no_prime_factor_congruent_to_7_mod_8_l908_90817


namespace NUMINAMATH_GPT_max_a_inequality_l908_90872

theorem max_a_inequality (a : ℝ) :
  (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := 
sorry

end NUMINAMATH_GPT_max_a_inequality_l908_90872


namespace NUMINAMATH_GPT_max_marks_l908_90842

theorem max_marks (M : ℝ) (h1 : 0.40 * M = 200) : M = 500 := by
  sorry

end NUMINAMATH_GPT_max_marks_l908_90842


namespace NUMINAMATH_GPT_solve_inequality_l908_90864

theorem solve_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l908_90864


namespace NUMINAMATH_GPT_square_roots_sum_eq_zero_l908_90846

theorem square_roots_sum_eq_zero (x y : ℝ) (h1 : x^2 = 2011) (h2 : y^2 = 2011) : x + y = 0 :=
by sorry

end NUMINAMATH_GPT_square_roots_sum_eq_zero_l908_90846


namespace NUMINAMATH_GPT_value_of_x2_plus_y2_l908_90820

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x2_plus_y2_l908_90820


namespace NUMINAMATH_GPT_determine_a_square_binomial_l908_90812

theorem determine_a_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 16 * x + a) = (2 * x + b)^2) → a = 16 := 
by
  sorry

end NUMINAMATH_GPT_determine_a_square_binomial_l908_90812


namespace NUMINAMATH_GPT_ralph_total_cost_correct_l908_90851

noncomputable def calculate_total_cost : ℝ :=
  let original_cart_cost := 54.00
  let small_issue_item_original := 20.00
  let additional_item_original := 15.00
  let small_issue_discount := 0.20
  let additional_item_discount := 0.25
  let coupon_discount := 0.10
  let sales_tax := 0.07

  -- Calculate the discounted prices
  let small_issue_discounted := small_issue_item_original * (1 - small_issue_discount)
  let additional_item_discounted := additional_item_original * (1 - additional_item_discount)

  -- Total cost before the coupon and tax
  let total_before_coupon := original_cart_cost + small_issue_discounted + additional_item_discounted

  -- Apply the coupon discount
  let total_after_coupon := total_before_coupon * (1 - coupon_discount)

  -- Apply the sales tax
  total_after_coupon * (1 + sales_tax)

-- Define the problem statement
theorem ralph_total_cost_correct : calculate_total_cost = 78.24 :=
by sorry

end NUMINAMATH_GPT_ralph_total_cost_correct_l908_90851


namespace NUMINAMATH_GPT_range_of_a_l908_90896

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l908_90896


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l908_90841

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 < x → (1 / 2 < x → (f (x + 0.1) > f x)) :=
by
  intro x hx h
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l908_90841


namespace NUMINAMATH_GPT_find_length_AB_l908_90838

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line y = x - 1
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection length |AB|
noncomputable def length_AB (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

-- Main theorem statement
theorem find_length_AB (x1 x2 : ℝ)
  (h₁ : parabola x1 (x1 - 1))
  (h₂ : parabola x2 (x2 - 1))
  (hx : x1 + x2 = 6) :
  length_AB x1 x2 = 8 := sorry

end NUMINAMATH_GPT_find_length_AB_l908_90838


namespace NUMINAMATH_GPT_max_four_color_rectangles_l908_90834

def color := Fin 4
def grid := Fin 100 × Fin 100
def colored_grid := grid → color

def count_four_color_rectangles (g : colored_grid) : ℕ := sorry

theorem max_four_color_rectangles (g : colored_grid) :
  count_four_color_rectangles g ≤ 9375000 := sorry

end NUMINAMATH_GPT_max_four_color_rectangles_l908_90834


namespace NUMINAMATH_GPT_most_reasonable_sample_l908_90881

-- Define what it means to be a reasonable sample
def is_reasonable_sample (sample : String) : Prop :=
  sample = "D"

-- Define the conditions for each sample
def sample_A := "A"
def sample_B := "B"
def sample_C := "C"
def sample_D := "D"

-- Define the problem statement
theorem most_reasonable_sample :
  is_reasonable_sample sample_D :=
sorry

end NUMINAMATH_GPT_most_reasonable_sample_l908_90881


namespace NUMINAMATH_GPT_has_only_one_minimum_point_and_no_maximum_point_l908_90823

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

theorem has_only_one_minimum_point_and_no_maximum_point :
  ∃! c : ℝ, (deriv f c = 0 ∧ ∀ x < c, deriv f x < 0 ∧ ∀ x > c, deriv f x > 0) ∧
  ∀ x, f x ≥ f c ∧ (∀ x, deriv f x > 0 ∨ deriv f x < 0) := sorry

end NUMINAMATH_GPT_has_only_one_minimum_point_and_no_maximum_point_l908_90823


namespace NUMINAMATH_GPT_sum_consecutive_integers_l908_90809

theorem sum_consecutive_integers (S : ℕ) (hS : S = 221) :
  ∃ (k : ℕ) (hk : k ≥ 2) (n : ℕ), 
    (S = k * n + (k * (k - 1)) / 2) → k = 2 := sorry

end NUMINAMATH_GPT_sum_consecutive_integers_l908_90809


namespace NUMINAMATH_GPT_mean_of_second_set_l908_90885

theorem mean_of_second_set (x : ℝ)
  (H1 : (28 + x + 70 + 88 + 104) / 5 = 67) :
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
sorry

end NUMINAMATH_GPT_mean_of_second_set_l908_90885


namespace NUMINAMATH_GPT_find_p_if_geometric_exists_p_arithmetic_sequence_l908_90821

variable (a : ℕ → ℝ) (p : ℝ)

-- Condition 1: a_1 = 1
axiom a1_eq_1 : a 1 = 1

-- Condition 2: a_n + a_{n+1} = pn + 1
axiom a_recurrence : ∀ n : ℕ, a n + a (n + 1) = p * n + 1

-- Question 1: If a_1, a_2, a_4 form a geometric sequence, find p
theorem find_p_if_geometric (h_geometric : (a 2)^2 = (a 1) * (a 4)) : p = 2 := by
  -- Proof goes here
  sorry

-- Question 2: Does there exist a p such that the sequence {a_n} is an arithmetic sequence?
theorem exists_p_arithmetic_sequence : ∃ p : ℝ, (∀ n : ℕ, a n + a (n + 1) = p * n + 1) ∧ 
                                         (∀ m n : ℕ, a (m + n) - a n = m * p) := by
  -- Proof goes here
  exists 2
  sorry

end NUMINAMATH_GPT_find_p_if_geometric_exists_p_arithmetic_sequence_l908_90821


namespace NUMINAMATH_GPT_percent_decrease_area_square_l908_90858

/-- 
In a configuration, two figures, an equilateral triangle and a square, are initially given. 
The equilateral triangle has an area of 27√3 square inches, and the square has an area of 27 square inches.
If the side length of the square is decreased by 10%, prove that the percent decrease in the area of the square is 19%.
-/
theorem percent_decrease_area_square 
  (triangle_area : ℝ := 27 * Real.sqrt 3)
  (square_area : ℝ := 27)
  (percentage_decrease : ℝ := 0.10) : 
  let new_square_side := Real.sqrt square_area * (1 - percentage_decrease)
  let new_square_area := new_square_side ^ 2
  let area_decrease := square_area - new_square_area
  let percent_decrease := (area_decrease / square_area) * 100
  percent_decrease = 19 := 
by
  sorry

end NUMINAMATH_GPT_percent_decrease_area_square_l908_90858


namespace NUMINAMATH_GPT_snow_shoveling_l908_90847

noncomputable def volume_of_snow_shoveled (length1 length2 width depth1 depth2 : ℝ) : ℝ :=
  (length1 * width * depth1) + (length2 * width * depth2)

theorem snow_shoveling :
  volume_of_snow_shoveled 15 15 4 1 (1 / 2) = 90 :=
by
  sorry

end NUMINAMATH_GPT_snow_shoveling_l908_90847


namespace NUMINAMATH_GPT_solve_for_x_l908_90876

/-- Let f(x) = 2 - 1 / (2 - x)^3.
Proof that f(x) = 1 / (2 - x)^3 implies x = 1. -/
theorem solve_for_x (x : ℝ) (h : 2 - 1 / (2 - x)^3 = 1 / (2 - x)^3) : x = 1 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l908_90876


namespace NUMINAMATH_GPT_geometric_sequence_sum_l908_90886

theorem geometric_sequence_sum (S : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n : ℕ, n > 0 → S n = 2^n + a) →
  (S 1 = 2 + a) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  (a_n 1 = 1) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l908_90886


namespace NUMINAMATH_GPT_minimum_possible_sum_of_4x4x4_cube_l908_90870

theorem minimum_possible_sum_of_4x4x4_cube: 
  (∀ die: ℕ, (1 ≤ die) ∧ (die ≤ 6) ∧ (∃ opposite, die + opposite = 7)) → 
  (∃ sum, sum = 304) :=
by
  sorry

end NUMINAMATH_GPT_minimum_possible_sum_of_4x4x4_cube_l908_90870


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l908_90879

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l908_90879


namespace NUMINAMATH_GPT_C_completes_work_in_4_days_l908_90825

theorem C_completes_work_in_4_days
  (A_days : ℕ)
  (B_efficiency : ℕ → ℕ)
  (C_efficiency : ℕ → ℕ)
  (hA : A_days = 12)
  (hB : ∀ {x}, B_efficiency x = x * 3 / 2)
  (hC : ∀ {x}, C_efficiency x = x * 2) :
  (1 / (1 / (C_efficiency (B_efficiency A_days)))) = 4 := by
  sorry

end NUMINAMATH_GPT_C_completes_work_in_4_days_l908_90825


namespace NUMINAMATH_GPT_collective_apples_l908_90874

theorem collective_apples :
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8 := by
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  show (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8
  sorry

end NUMINAMATH_GPT_collective_apples_l908_90874


namespace NUMINAMATH_GPT_intersecting_lines_sum_c_d_l908_90837

theorem intersecting_lines_sum_c_d 
  (c d : ℚ)
  (h1 : 2 = 1 / 5 * (3 : ℚ) + c)
  (h2 : 3 = 1 / 5 * (2 : ℚ) + d) : 
  c + d = 4 :=
by sorry

end NUMINAMATH_GPT_intersecting_lines_sum_c_d_l908_90837


namespace NUMINAMATH_GPT_find_b_in_triangle_l908_90824

theorem find_b_in_triangle (a B C A b : ℝ)
  (ha : a = Real.sqrt 3)
  (hB : Real.sin B = 1 / 2)
  (hC : C = Real.pi / 6)
  (hA : A = 2 * Real.pi / 3) :
  b = 1 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_b_in_triangle_l908_90824


namespace NUMINAMATH_GPT_range_g_l908_90839

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2 * x + 2)

theorem range_g : Set.Icc (-(1:ℝ)/2) (1/2) = {y : ℝ | ∃ x : ℝ, g x = y} := 
by
  sorry

end NUMINAMATH_GPT_range_g_l908_90839


namespace NUMINAMATH_GPT_males_listen_l908_90866

theorem males_listen (total_listen : ℕ) (females_listen : ℕ) (known_total_listen : total_listen = 160)
  (known_females_listen : females_listen = 75) : (total_listen - females_listen) = 85 :=
by 
  sorry

end NUMINAMATH_GPT_males_listen_l908_90866


namespace NUMINAMATH_GPT_min_triangular_faces_l908_90804

theorem min_triangular_faces (l c e m n k : ℕ) (h1 : l > c) (h2 : l + c = e + 2) (h3 : l = c + k) (h4 : e ≥ (3 * m + 4 * n) / 2) :
  m ≥ 6 := sorry

end NUMINAMATH_GPT_min_triangular_faces_l908_90804


namespace NUMINAMATH_GPT_quadratic_function_vertex_upwards_exists_l908_90883

theorem quadratic_function_vertex_upwards_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x, f x = a * (x - 1) * (x - 1) - 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_vertex_upwards_exists_l908_90883


namespace NUMINAMATH_GPT_solve_for_x_l908_90867

theorem solve_for_x (x y z : ℕ) 
  (h1 : 3^x * 4^y / 2^z = 59049)
  (h2 : x - y + 2 * z = 10) : 
  x = 10 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l908_90867


namespace NUMINAMATH_GPT_count_multiples_5_or_7_but_not_both_l908_90844

-- Definitions based on the given problem conditions
def multiples_of_five (n : Nat) : Nat :=
  (n - 1) / 5

def multiples_of_seven (n : Nat) : Nat :=
  (n - 1) / 7

def multiples_of_thirty_five (n : Nat) : Nat :=
  (n - 1) / 35

def count_multiples (n : Nat) : Nat :=
  (multiples_of_five n) + (multiples_of_seven n) - 2 * (multiples_of_thirty_five n)

-- The main statement to be proved
theorem count_multiples_5_or_7_but_not_both : count_multiples 101 = 30 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_5_or_7_but_not_both_l908_90844


namespace NUMINAMATH_GPT_set_difference_equals_six_l908_90806

-- Set Operations definitions used
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Problem statement to prove
theorem set_difference_equals_six : set_difference N M = {6} :=
  sorry

end NUMINAMATH_GPT_set_difference_equals_six_l908_90806


namespace NUMINAMATH_GPT_find_base_l908_90897

noncomputable def f (a x : ℝ) := 1 + (Real.log x) / (Real.log a)

theorem find_base (a : ℝ) (hinv_pass : (∀ y : ℝ, (∀ x : ℝ, f a x = y → x = 4 → y = 3))) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_base_l908_90897


namespace NUMINAMATH_GPT_average_of_six_starting_from_d_plus_one_l908_90871

theorem average_of_six_starting_from_d_plus_one (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (c + 6) = ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 6 := 
by 
-- Proof omitted; end with sorry
sorry

end NUMINAMATH_GPT_average_of_six_starting_from_d_plus_one_l908_90871


namespace NUMINAMATH_GPT_ratio_of_radii_l908_90826

theorem ratio_of_radii (r R : ℝ) (hR : R > 0) (hr : r > 0)
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_l908_90826


namespace NUMINAMATH_GPT_range_x_inequality_l908_90893

theorem range_x_inequality (a b x : ℝ) (ha : a ≠ 0) :
  (x ≥ 1/2) ∧ (x ≤ 5/2) →
  |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|) :=
by
  sorry

end NUMINAMATH_GPT_range_x_inequality_l908_90893


namespace NUMINAMATH_GPT_inequality_1_minimum_value_l908_90829

-- Definition for part (1)
theorem inequality_1 (a b m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (a^2 / m + b^2 / n) ≥ ((a + b)^2 / (m + n)) :=
sorry

-- Definition for part (2)
theorem minimum_value (x : ℝ) (hx : 0 < x) (hx' : x < 1) : 
  (∃ (y : ℝ), y = (1 / x + 4 / (1 - x)) ∧ y = 9) :=
sorry

end NUMINAMATH_GPT_inequality_1_minimum_value_l908_90829


namespace NUMINAMATH_GPT_nancy_clay_pots_l908_90899

theorem nancy_clay_pots : 
  ∃ M : ℕ, (M + 2 * M + 14 = 50) ∧ M = 12 :=
sorry

end NUMINAMATH_GPT_nancy_clay_pots_l908_90899


namespace NUMINAMATH_GPT_less_than_reciprocal_l908_90814

theorem less_than_reciprocal (a b c d e : ℝ) (ha : a = -3) (hb : b = -1/2) (hc : c = 0.5) (hd : d = 1) (he : e = 3) :
  (a < 1 / a) ∧ (c < 1 / c) ∧ ¬(b < 1 / b) ∧ ¬(d < 1 / d) ∧ ¬(e < 1 / e) :=
by
  sorry

end NUMINAMATH_GPT_less_than_reciprocal_l908_90814


namespace NUMINAMATH_GPT_irrational_of_sqrt_3_l908_90898

theorem irrational_of_sqrt_3 :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ ↑a / ↑b = Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_irrational_of_sqrt_3_l908_90898


namespace NUMINAMATH_GPT_percent_area_contained_l908_90827

-- Define the conditions as Lean definitions
def side_length_square (s : ℝ) : ℝ := s
def width_rectangle (s : ℝ) : ℝ := 2 * s
def length_rectangle (s : ℝ) : ℝ := 3 * (width_rectangle s)

-- Define areas based on definitions
def area_square (s : ℝ) : ℝ := (side_length_square s) ^ 2
def area_rectangle (s : ℝ) : ℝ := (length_rectangle s) * (width_rectangle s)

-- The main theorem stating the percentage of the rectangle's area contained within the square
theorem percent_area_contained (s : ℝ) (h : s ≠ 0) :
  (area_square s / area_rectangle s) * 100 = 8.33 := by
  sorry

end NUMINAMATH_GPT_percent_area_contained_l908_90827


namespace NUMINAMATH_GPT_math_problem_l908_90850

noncomputable def A (k : ℝ) : ℝ := k - 5
noncomputable def B (k : ℝ) : ℝ := k + 2
noncomputable def C (k : ℝ) : ℝ := k / 2
noncomputable def D (k : ℝ) : ℝ := 2 * k

theorem math_problem (k : ℝ) (h : A k + B k + C k + D k = 100) : 
  (A k) * (B k) * (C k) * (D k) =  (161 * 224 * 103 * 412) / 6561 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l908_90850


namespace NUMINAMATH_GPT_basketball_volleyball_problem_l908_90822

-- Define variables and conditions
variables (x y : ℕ) (m : ℕ)

-- Conditions
def price_conditions : Prop :=
  2 * x + 3 * y = 190 ∧ 3 * x = 5 * y

def price_solutions : Prop :=
  x = 50 ∧ y = 30

def purchase_conditions : Prop :=
  8 ≤ m ∧ m ≤ 10 ∧ 50 * m + 30 * (20 - m) ≤ 800

-- The most cost-effective plan
def cost_efficient_plan : Prop :=
  m = 8 ∧ (20 - m) = 12

-- Conjecture for the problem
theorem basketball_volleyball_problem :
  price_conditions x y ∧ purchase_conditions m →
  price_solutions x y ∧ cost_efficient_plan m :=
by {
  sorry
}

end NUMINAMATH_GPT_basketball_volleyball_problem_l908_90822


namespace NUMINAMATH_GPT_inequality_I_l908_90892

theorem inequality_I (a b x y : ℝ) (hx : x < a) (hy : y < b) : x * y < a * b :=
sorry

end NUMINAMATH_GPT_inequality_I_l908_90892


namespace NUMINAMATH_GPT_nora_third_tree_oranges_l908_90895

theorem nora_third_tree_oranges (a b c total : ℕ)
  (h_a : a = 80)
  (h_b : b = 60)
  (h_total : total = 260)
  (h_sum : total = a + b + c) :
  c = 120 :=
by
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_nora_third_tree_oranges_l908_90895


namespace NUMINAMATH_GPT_june_found_total_eggs_l908_90861

def eggs_in_tree_1 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_tree_2 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_yard (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest

def total_eggs (eggs_tree_1 : ℕ) (eggs_tree_2 : ℕ) (eggs_yard : ℕ) : ℕ :=
eggs_tree_1 + eggs_tree_2 + eggs_yard

theorem june_found_total_eggs :
  total_eggs (eggs_in_tree_1 2 5) (eggs_in_tree_2 1 3) (eggs_in_yard 1 4) = 17 :=
by
  sorry

end NUMINAMATH_GPT_june_found_total_eggs_l908_90861


namespace NUMINAMATH_GPT_quadratic_single_solution_positive_n_l908_90831

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end NUMINAMATH_GPT_quadratic_single_solution_positive_n_l908_90831


namespace NUMINAMATH_GPT_spinner_points_east_l908_90863

-- Definitions for the conditions
def initial_direction := "north"

-- Clockwise and counterclockwise movements as improper fractions
def clockwise_move := (7 : ℚ) / 2
def counterclockwise_move := (17 : ℚ) / 4

-- Compute the net movement (negative means counterclockwise)
def net_movement := clockwise_move - counterclockwise_move

-- Translate net movement into a final direction (using modulo arithmetic with 1 revolution = 360 degrees equivalent)
def final_position : ℚ := (net_movement + 1) % 1

-- The goal is to prove that the final direction is east (which corresponds to 1/4 revolution)
theorem spinner_points_east :
  final_position = (1 / 4 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_spinner_points_east_l908_90863


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l908_90865

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + m * (a 1 - a 0)

theorem arithmetic_sequence_sum
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 = 15) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l908_90865


namespace NUMINAMATH_GPT_snake_alligator_consumption_l908_90894

theorem snake_alligator_consumption :
  (616 / 7) = 88 :=
by
  sorry

end NUMINAMATH_GPT_snake_alligator_consumption_l908_90894


namespace NUMINAMATH_GPT_find_total_tennis_balls_l908_90818

noncomputable def original_white_balls : ℕ := sorry
noncomputable def original_yellow_balls : ℕ := sorry
noncomputable def dispatched_yellow_balls : ℕ := original_yellow_balls + 20

theorem find_total_tennis_balls
  (white_balls_eq : original_white_balls = original_yellow_balls)
  (ratio_eq : original_white_balls / dispatched_yellow_balls = 8 / 13) :
  original_white_balls + original_yellow_balls = 64 := sorry

end NUMINAMATH_GPT_find_total_tennis_balls_l908_90818


namespace NUMINAMATH_GPT_box_dimension_triples_l908_90882

theorem box_dimension_triples (N : ℕ) :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c = 1 / 8) → ∃ k, k = N := sorry 

end NUMINAMATH_GPT_box_dimension_triples_l908_90882


namespace NUMINAMATH_GPT_monthly_pool_cost_is_correct_l908_90813

def cost_of_cleaning : ℕ := 150
def tip_percentage : ℕ := 10
def number_of_cleanings_in_a_month : ℕ := 30 / 3
def cost_of_chemicals_per_use : ℕ := 200
def number_of_chemical_uses_in_a_month : ℕ := 2

def monthly_cost_of_pool : ℕ :=
  let cost_per_cleaning := cost_of_cleaning + (cost_of_cleaning * tip_percentage / 100)
  let total_cleaning_cost := number_of_cleanings_in_a_month * cost_per_cleaning
  let total_chemical_cost := number_of_chemical_uses_in_a_month * cost_of_chemicals_per_use
  total_cleaning_cost + total_chemical_cost

theorem monthly_pool_cost_is_correct : monthly_cost_of_pool = 2050 :=
by
  sorry

end NUMINAMATH_GPT_monthly_pool_cost_is_correct_l908_90813


namespace NUMINAMATH_GPT_area_of_polygon_ABHFGD_l908_90854

noncomputable def total_area_ABHFGD : ℝ :=
  let side_ABCD := 3
  let side_EFGD := 5
  let area_ABCD := side_ABCD * side_ABCD
  let area_EFGD := side_EFGD * side_EFGD
  let area_DBH := 0.5 * 3 * (3 / 2 : ℝ) -- Area of triangle DBH
  let area_DFH := 0.5 * 5 * (5 / 2 : ℝ) -- Area of triangle DFH
  area_ABCD + area_EFGD - (area_DBH + area_DFH)

theorem area_of_polygon_ABHFGD : total_area_ABHFGD = 25.5 := by
  sorry

end NUMINAMATH_GPT_area_of_polygon_ABHFGD_l908_90854


namespace NUMINAMATH_GPT_initial_amount_of_money_l908_90889

-- Definitions based on conditions in a)
variables (n : ℚ) -- Bert left the house with n dollars
def after_hardware_store := (3 / 4) * n
def after_dry_cleaners := after_hardware_store - 9
def after_grocery_store := (1 / 2) * after_dry_cleaners
def after_bookstall := (2 / 3) * after_grocery_store
def after_donation := (4 / 5) * after_bookstall

-- Theorem statement
theorem initial_amount_of_money : after_donation = 27 → n = 72 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_of_money_l908_90889


namespace NUMINAMATH_GPT_find_n_equiv_l908_90849

theorem find_n_equiv :
  ∃ (n : ℕ), 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ (n = 3 ∨ n = 9) :=
by
  sorry

end NUMINAMATH_GPT_find_n_equiv_l908_90849


namespace NUMINAMATH_GPT_ab_product_l908_90819

theorem ab_product (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) * (2 * b + a) = 4752) : a * b = 520 := 
by
  sorry

end NUMINAMATH_GPT_ab_product_l908_90819


namespace NUMINAMATH_GPT_not_odd_not_even_min_value_3_l908_90811

def f (x : ℝ) : ℝ := x^2 + abs (x - 2) - 1

-- Statement 1: Prove that the function is neither odd nor even.
theorem not_odd_not_even : 
  ¬(∀ x, f (-x) = -f x) ∧ ¬(∀ x, f (-x) = f x) :=
sorry

-- Statement 2: Prove that the minimum value of the function is 3.
theorem min_value_3 : ∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≥ 3 :=
sorry

end NUMINAMATH_GPT_not_odd_not_even_min_value_3_l908_90811


namespace NUMINAMATH_GPT_solution_set_quadratic_ineq_all_real_l908_90888

theorem solution_set_quadratic_ineq_all_real (a b c : ℝ) :
  (∀ x : ℝ, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_quadratic_ineq_all_real_l908_90888


namespace NUMINAMATH_GPT_sum_units_digits_3a_l908_90860

theorem sum_units_digits_3a (a : ℕ) (h_pos : 0 < a) (h_units : (2 * a) % 10 = 4) : 
  ((3 * (a % 10) = (6 : ℕ) ∨ (3 * (a % 10) = (21 : ℕ))) → 6 + 1 = 7) := 
by
  sorry

end NUMINAMATH_GPT_sum_units_digits_3a_l908_90860


namespace NUMINAMATH_GPT_temperature_at_midnight_is_minus4_l908_90800

-- Definitions of initial temperature and changes
def initial_temperature : ℤ := -2
def temperature_rise_noon : ℤ := 6
def temperature_drop_midnight : ℤ := 8

-- Temperature at midnight
def temperature_midnight : ℤ :=
  initial_temperature + temperature_rise_noon - temperature_drop_midnight

theorem temperature_at_midnight_is_minus4 :
  temperature_midnight = -4 := by
  sorry

end NUMINAMATH_GPT_temperature_at_midnight_is_minus4_l908_90800


namespace NUMINAMATH_GPT_ducks_remaining_after_three_nights_l908_90802

def initial_ducks : ℕ := 320
def first_night_ducks_eaten (ducks : ℕ) : ℕ := ducks * 1 / 4
def after_first_night (ducks : ℕ) : ℕ := ducks - first_night_ducks_eaten ducks
def second_night_ducks_fly_away (ducks : ℕ) : ℕ := ducks * 1 / 6
def after_second_night (ducks : ℕ) : ℕ := ducks - second_night_ducks_fly_away ducks
def third_night_ducks_stolen (ducks : ℕ) : ℕ := ducks * 30 / 100
def after_third_night (ducks : ℕ) : ℕ := ducks - third_night_ducks_stolen ducks

theorem ducks_remaining_after_three_nights : after_third_night (after_second_night (after_first_night initial_ducks)) = 140 :=
by 
  -- replace the following sorry with the actual proof steps
  sorry

end NUMINAMATH_GPT_ducks_remaining_after_three_nights_l908_90802


namespace NUMINAMATH_GPT_find_divisor_l908_90805

theorem find_divisor (n x : ℕ) (hx : x ≠ 11) (hn : n = 386) 
  (h1 : ∃ k : ℤ, n = k * x + 1) (h2 : ∀ m : ℤ, n = 11 * m + 1 → n = 386) : x = 5 :=
  sorry

end NUMINAMATH_GPT_find_divisor_l908_90805


namespace NUMINAMATH_GPT_obtuse_equilateral_triangle_impossible_l908_90887

-- Define a scalene triangle 
def is_scalene_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ A + B + C = 180

-- Define acute triangles
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

-- Define right triangles
def is_right_triangle (A B C : ℝ) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

-- Define isosceles triangles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

-- Define obtuse triangles
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

-- Define equilateral triangles
def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = a ∧ A = 60 ∧ B = 60 ∧ C = 60

theorem obtuse_equilateral_triangle_impossible :
  ¬ ∃ (a b c A B C : ℝ), is_equilateral_triangle a b c A B C ∧ is_obtuse_triangle A B C :=
by
  sorry

end NUMINAMATH_GPT_obtuse_equilateral_triangle_impossible_l908_90887


namespace NUMINAMATH_GPT_shortest_chord_through_M_l908_90890

noncomputable def point_M : ℝ × ℝ := (1, 0)
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y = 0

theorem shortest_chord_through_M :
  (∀ x y : ℝ, circle_C x y → x + y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_through_M_l908_90890


namespace NUMINAMATH_GPT_min_value_of_x4_y3_z2_l908_90862

noncomputable def min_value_x4_y3_z2 (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem min_value_of_x4_y3_z2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : 1/x + 1/y + 1/z = 9) : 
  min_value_x4_y3_z2 x y z = 1 / 3456 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_x4_y3_z2_l908_90862


namespace NUMINAMATH_GPT_fg_of_3_eq_29_l908_90875

def f (x : ℝ) : ℝ := 2 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_29_l908_90875


namespace NUMINAMATH_GPT_oven_capacity_correct_l908_90855

-- Definitions for the conditions
def dough_time := 30 -- minutes
def bake_time := 30 -- minutes
def pizzas_per_batch := 3
def total_time := 5 * 60 -- minutes (5 hours)
def total_pizzas := 12

-- Calculation of the number of batches
def batches_needed := total_pizzas / pizzas_per_batch

-- Calculation of the time for making dough
def dough_preparation_time := batches_needed * dough_time

-- Calculation of the remaining time for baking
def remaining_baking_time := total_time - dough_preparation_time

-- Calculation of the number of 30-minute baking intervals
def baking_intervals := remaining_baking_time / bake_time

-- Calculation of the capacity of the oven
def oven_capacity := total_pizzas / baking_intervals

theorem oven_capacity_correct : oven_capacity = 2 := by
  sorry

end NUMINAMATH_GPT_oven_capacity_correct_l908_90855


namespace NUMINAMATH_GPT_seven_y_minus_x_eq_three_l908_90868

-- Definitions for the conditions
variables (x y : ℤ)
variables (hx : x > 0)
variables (h1 : x = 11 * y + 4)
variables (h2 : 2 * x = 18 * y + 1)

-- The theorem we want to prove
theorem seven_y_minus_x_eq_three : 7 * y - x = 3 :=
by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_seven_y_minus_x_eq_three_l908_90868


namespace NUMINAMATH_GPT_no_solution_for_a_l908_90869

theorem no_solution_for_a {a : ℝ} :
  (a ∈ Set.Iic (-32) ∪ Set.Ici 0) →
  ¬ ∃ x : ℝ,  9 * |x - 4 * a| + |x - a^2| + 8 * x - 4 * a = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_solution_for_a_l908_90869


namespace NUMINAMATH_GPT_no_solutions_of_pairwise_distinct_l908_90878

theorem no_solutions_of_pairwise_distinct 
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∀ x : ℝ, ¬(x^3 - a * x^2 + b^3 = 0 ∧ x^3 - b * x^2 + c^3 = 0 ∧ x^3 - c * x^2 + a^3 = 0) :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_no_solutions_of_pairwise_distinct_l908_90878


namespace NUMINAMATH_GPT_toms_investment_l908_90830

theorem toms_investment 
  (P : ℝ)
  (rA : ℝ := 0.06)
  (nA : ℝ := 1)
  (tA : ℕ := 4)
  (rB : ℝ := 0.08)
  (nB : ℕ := 2)
  (tB : ℕ := 4)
  (delta : ℝ := 100)
  (A_A := P * (1 + rA / nA) ^ (nA * tA))
  (A_B := P * (1 + rB / nB) ^ (nB * tB))
  (h : A_B - A_A = delta) : 
  P = 942.59 := by
sorry

end NUMINAMATH_GPT_toms_investment_l908_90830


namespace NUMINAMATH_GPT_positive_value_of_A_l908_90840

theorem positive_value_of_A (A : ℝ) (h : A^2 + 3^2 = 130) : A = 11 :=
sorry

end NUMINAMATH_GPT_positive_value_of_A_l908_90840


namespace NUMINAMATH_GPT_absolute_value_property_l908_90873

theorem absolute_value_property (a b c : ℤ) (h : |a - b| + |c - a| = 1) : |a - c| + |c - b| + |b - a| = 2 :=
sorry

end NUMINAMATH_GPT_absolute_value_property_l908_90873


namespace NUMINAMATH_GPT_smallest_5_digit_number_divisible_by_and_factor_of_l908_90856

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

def is_divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

def is_factor_of (x y : ℕ) : Prop := is_divisible_by y x

def is_5_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_5_digit_number_divisible_by_and_factor_of :
  ∃ n : ℕ,
    is_5_digit_number n ∧
    is_divisible_by n 32 ∧
    is_divisible_by n 45 ∧
    is_divisible_by n 54 ∧
    is_factor_of n 30 ∧
    (∀ m : ℕ, is_5_digit_number m → is_divisible_by m 32 → is_divisible_by m 45 → is_divisible_by m 54 → is_factor_of m 30 → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_5_digit_number_divisible_by_and_factor_of_l908_90856


namespace NUMINAMATH_GPT_intersection_eq_singleton_l908_90853

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_eq_singleton :
  A ∩ B = {1} :=
sorry

end NUMINAMATH_GPT_intersection_eq_singleton_l908_90853


namespace NUMINAMATH_GPT_find_missing_number_l908_90833

-- Define the given numbers as a list
def given_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

-- Define the arithmetic mean condition
def arithmetic_mean (xs : List ℕ) (mean : ℕ) : Prop :=
  (xs.sum + mean) / xs.length.succ = 12

-- Define the proof problem
theorem find_missing_number (x : ℕ) (h : arithmetic_mean given_numbers x) : x = 7 := 
sorry

end NUMINAMATH_GPT_find_missing_number_l908_90833


namespace NUMINAMATH_GPT_scientific_notation_40_9_billion_l908_90836

theorem scientific_notation_40_9_billion :
  (40.9 * 10^9) = 4.09 * 10^9 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_40_9_billion_l908_90836


namespace NUMINAMATH_GPT_picked_balls_correct_l908_90803

-- Conditions
def initial_balls := 6
def final_balls := 24

-- The task is to find the number of picked balls
def picked_balls : Nat := final_balls - initial_balls

-- The proof goal
theorem picked_balls_correct : picked_balls = 18 :=
by
  -- We declare, but the proof is not required
  sorry

end NUMINAMATH_GPT_picked_balls_correct_l908_90803


namespace NUMINAMATH_GPT_inequality_solution_range_l908_90816

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, 2 * x - 6 + m < 0 ∧ 4 * x - m > 0) → m < 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l908_90816


namespace NUMINAMATH_GPT_intersection_P_Q_l908_90877

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x < 1}

-- The proof statement
theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l908_90877


namespace NUMINAMATH_GPT_min_value_of_f_l908_90891

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_of_f : ∃ x : ℝ, f x = x^3 - 3 * x^2 + 1 ∧ (∀ y : ℝ, f y ≥ f 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l908_90891


namespace NUMINAMATH_GPT_cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l908_90848

theorem cos_2alpha_plus_pi_div_2_eq_neg_24_div_25
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tanα : Real.tan α = 4 / 3) :
  Real.cos (2 * α + π / 2) = - 24 / 25 :=
by sorry

end NUMINAMATH_GPT_cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l908_90848


namespace NUMINAMATH_GPT_problem1_problem2_l908_90810

-- Problem 1
theorem problem1 (a : ℝ) : 3 * a ^ 2 - 2 * a + 1 + (3 * a - a ^ 2 + 2) = 2 * a ^ 2 + a + 3 :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : x - 2 * (x - 3 / 2 * y) + 3 * (x - x * y) = 2 * x + 3 * y - 3 * x * y :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l908_90810


namespace NUMINAMATH_GPT_cost_per_board_game_is_15_l908_90845

-- Definitions of the conditions
def number_of_board_games : ℕ := 6
def bill_paid : ℕ := 100
def bill_value : ℕ := 5
def bills_received : ℕ := 2

def total_change := bills_received * bill_value
def total_cost := bill_paid - total_change
def cost_per_board_game := total_cost / number_of_board_games

-- The theorem stating that the cost of each board game is $15
theorem cost_per_board_game_is_15 : cost_per_board_game = 15 := 
by
  -- Omitted proof steps
  sorry

end NUMINAMATH_GPT_cost_per_board_game_is_15_l908_90845
