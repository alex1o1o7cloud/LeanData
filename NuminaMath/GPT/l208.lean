import Mathlib

namespace NUMINAMATH_GPT_pieces_given_l208_20811

def pieces_initially := 38
def pieces_now := 54

theorem pieces_given : pieces_now - pieces_initially = 16 := by
  sorry

end NUMINAMATH_GPT_pieces_given_l208_20811


namespace NUMINAMATH_GPT_solve_inequality_l208_20836

theorem solve_inequality (x : ℝ) : (1 / (x + 2) + 4 / (x + 8) ≤ 3 / 4) ↔ ((-8 < x ∧ x ≤ -4) ∨ (-4 ≤ x ∧ x ≤ 4 / 3)) ∧ x ≠ -2 ∧ x ≠ -8 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l208_20836


namespace NUMINAMATH_GPT_sum_of_xyz_l208_20835

theorem sum_of_xyz (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 18) (hxz : x * z = 3) (hyz : y * z = 6) : x + y + z = 10 := 
sorry

end NUMINAMATH_GPT_sum_of_xyz_l208_20835


namespace NUMINAMATH_GPT_triangle_area_bounded_by_lines_l208_20859

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_bounded_by_lines_l208_20859


namespace NUMINAMATH_GPT_distinct_real_solutions_exist_l208_20853

theorem distinct_real_solutions_exist (a : ℝ) (h : a > 3 / 4) : 
  ∃ (x y : ℝ), x ≠ y ∧ x = a - y^2 ∧ y = a - x^2 := 
sorry

end NUMINAMATH_GPT_distinct_real_solutions_exist_l208_20853


namespace NUMINAMATH_GPT_one_third_sugar_amount_l208_20897

-- Define the original amount of sugar as a mixed number
def original_sugar_mixed : ℚ := 6 + 1 / 3

-- Define the fraction representing one-third of the recipe
def one_third : ℚ := 1 / 3

-- Define the expected amount of sugar for one-third of the recipe
def expected_sugar_mixed : ℚ := 2 + 1 / 9

-- The theorem stating the proof problem
theorem one_third_sugar_amount : (one_third * original_sugar_mixed) = expected_sugar_mixed :=
sorry

end NUMINAMATH_GPT_one_third_sugar_amount_l208_20897


namespace NUMINAMATH_GPT_range_f1_l208_20850
open Function

theorem range_f1 (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Ici (-1) → y ∈ Set.Ici (-1) → x ≤ y → (x^2 + 2*a*x + 3) ≤ (y^2 + 2*a*y + 3)) →
  6 ≤ (1^2 + 2*a*1 + 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_f1_l208_20850


namespace NUMINAMATH_GPT_eval_complex_div_l208_20888

theorem eval_complex_div : 
  (i / (Real.sqrt 7 + 3 * I) = (3 / 16) + (Real.sqrt 7 / 16) * I) := 
by 
  sorry

end NUMINAMATH_GPT_eval_complex_div_l208_20888


namespace NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_repeating_decimal_l208_20808

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.45
  let a := 9 -- GCD of 45 and 99
  let numerator := 5
  let denominator := 11
  numerator + denominator = 16 :=
by { 
  sorry 
}

end NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_repeating_decimal_l208_20808


namespace NUMINAMATH_GPT_area_of_shaded_rectangle_l208_20837

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_rectangle_l208_20837


namespace NUMINAMATH_GPT_sum_of_first_2015_digits_l208_20893

noncomputable def repeating_decimal : List ℕ := [1, 4, 2, 8, 5, 7]

def sum_first_n_digits (digits : List ℕ) (n : ℕ) : ℕ :=
  let repeat_length := digits.length
  let full_cycles := n / repeat_length
  let remaining_digits := n % repeat_length
  full_cycles * (digits.sum) + (digits.take remaining_digits).sum

theorem sum_of_first_2015_digits :
  sum_first_n_digits repeating_decimal 2015 = 9065 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_2015_digits_l208_20893


namespace NUMINAMATH_GPT_wrapping_paper_each_present_l208_20866

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_each_present_l208_20866


namespace NUMINAMATH_GPT_remove_parentheses_correct_l208_20809

variable {a b c : ℝ}

theorem remove_parentheses_correct :
  -(a - b) = -a + b :=
by sorry

end NUMINAMATH_GPT_remove_parentheses_correct_l208_20809


namespace NUMINAMATH_GPT_cesaro_sum_51_term_sequence_l208_20804

noncomputable def cesaro_sum (B : List ℝ) : ℝ :=
  let T := List.scanl (· + ·) 0 B
  T.drop 1 |>.sum / B.length

theorem cesaro_sum_51_term_sequence (B : List ℝ) (h_length : B.length = 49)
  (h_cesaro_sum_49 : cesaro_sum B = 500) :
  cesaro_sum (B ++ [0, 0]) = 1441.18 :=
by
  sorry

end NUMINAMATH_GPT_cesaro_sum_51_term_sequence_l208_20804


namespace NUMINAMATH_GPT_fraction_equality_l208_20855

noncomputable def x := (4 : ℚ) / 6
noncomputable def y := (8 : ℚ) / 12

theorem fraction_equality : (6 * x + 8 * y) / (48 * x * y) = (7 : ℚ) / 16 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_equality_l208_20855


namespace NUMINAMATH_GPT_largest_unpayable_soldo_l208_20845

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end NUMINAMATH_GPT_largest_unpayable_soldo_l208_20845


namespace NUMINAMATH_GPT_percentage_increase_l208_20832

variable (E : ℝ) (P : ℝ)
variable (h1 : 1.36 * E = 495)
variable (h2 : (1 + P) * E = 454.96)

theorem percentage_increase :
  P = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l208_20832


namespace NUMINAMATH_GPT_tunnel_length_is_correct_l208_20886

-- Define the conditions given in the problem
def length_of_train : ℕ := 90
def speed_of_train : ℕ := 160
def time_to_pass_tunnel : ℕ := 3

-- Define the length of the tunnel to be proven
def length_of_tunnel : ℕ := 480 - length_of_train

-- Define the statement to be proven
theorem tunnel_length_is_correct : length_of_tunnel = 390 := by
  sorry

end NUMINAMATH_GPT_tunnel_length_is_correct_l208_20886


namespace NUMINAMATH_GPT_min_value_of_fraction_l208_20823

theorem min_value_of_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (4 / (a + 2) + 1 / (b + 1)) = 9 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_l208_20823


namespace NUMINAMATH_GPT_abs_sum_bound_l208_20851

theorem abs_sum_bound (x : ℝ) (a : ℝ) (h : |x - 4| + |x - 3| < a) (ha : 0 < a) : 1 < a :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_bound_l208_20851


namespace NUMINAMATH_GPT_evaluate_at_minus_three_l208_20882

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 9 * x^3 - 6 * x^2 + 15 * x - 210

theorem evaluate_at_minus_three : g (-3) = -1686 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_at_minus_three_l208_20882


namespace NUMINAMATH_GPT_someone_received_grade_D_or_F_l208_20891

theorem someone_received_grade_D_or_F (m x : ℕ) (hboys : ∃ n : ℕ, n = m + 3) 
  (hgrades_B : ∃ k : ℕ, k = x + 2) (hgrades_C : ∃ l : ℕ, l = 2 * (x + 2)) :
  ∃ p : ℕ, p = 1 ∨ p = 2 :=
by
  sorry

end NUMINAMATH_GPT_someone_received_grade_D_or_F_l208_20891


namespace NUMINAMATH_GPT_sandwiches_prepared_l208_20817

variable (S : ℕ)
variable (H1 : S > 0)
variable (H2 : ∃ r : ℕ, r = S / 4)
variable (H3 : ∃ b : ℕ, b = (3 * S / 4) / 6)
variable (H4 : ∃ c : ℕ, c = 2 * b)
variable (H5 : ∃ x : ℕ, 5 * x = 5)
variable (H6 : 3 * S / 8 - 5 = 4)

theorem sandwiches_prepared : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_sandwiches_prepared_l208_20817


namespace NUMINAMATH_GPT_age_sum_is_27_l208_20819

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end NUMINAMATH_GPT_age_sum_is_27_l208_20819


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l208_20887

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ (a : ℝ), a = 2 → (-(a) * (a / 4) = -1)) ∧ ∀ (a : ℝ), (-(a) * (a / 4) = -1 → a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l208_20887


namespace NUMINAMATH_GPT_min_value_x_y_l208_20810

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1 / x + 4 / y + 8) : 
  x + y ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_l208_20810


namespace NUMINAMATH_GPT_car_interval_length_l208_20807

theorem car_interval_length (S1 T : ℝ) (interval_length : ℝ) 
  (h1 : S1 = 39) 
  (h2 : (fun (n : ℕ) => S1 - 3 * n) 4 = 27)
  (h3 : 3.6 = 27 * T) 
  (h4 : interval_length = T * 60) :
  interval_length = 8 :=
by
  sorry

end NUMINAMATH_GPT_car_interval_length_l208_20807


namespace NUMINAMATH_GPT_smallest_positive_period_f_intervals_monotonically_increasing_f_l208_20822

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.sin x + Real.cos x)

-- 1. Proving the smallest positive period is π
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi := 
sorry

-- 2. Proving the intervals where the function is monotonically increasing
theorem intervals_monotonically_increasing_f : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi - (3 * Real.pi / 8)) (k * Real.pi + (Real.pi / 8)) → 
    0 < deriv f x :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_f_intervals_monotonically_increasing_f_l208_20822


namespace NUMINAMATH_GPT_solve_by_completing_square_l208_20847

theorem solve_by_completing_square (x: ℝ) (h: x^2 + 4 * x - 3 = 0) : (x + 2)^2 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_solve_by_completing_square_l208_20847


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l208_20856

theorem solution_set_of_inequality_system (x : ℝ) : (x + 1 > 0) ∧ (-2 * x ≤ 6) ↔ (x > -1) := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l208_20856


namespace NUMINAMATH_GPT_probability_of_smallest_section_l208_20849

-- Define the probabilities for the largest and next largest sections
def P_largest : ℚ := 1 / 2
def P_next_largest : ℚ := 1 / 3

-- Define the total probability constraint
def total_probability (P_smallest : ℚ) : Prop :=
  P_largest + P_next_largest + P_smallest = 1

-- State the theorem to be proved
theorem probability_of_smallest_section : 
  ∃ P_smallest : ℚ, total_probability P_smallest ∧ P_smallest = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_smallest_section_l208_20849


namespace NUMINAMATH_GPT_solve_x_eq_40_l208_20860

theorem solve_x_eq_40 : ∀ (x : ℝ), x + 2 * x = 400 - (3 * x + 4 * x) → x = 40 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_x_eq_40_l208_20860


namespace NUMINAMATH_GPT_min_value_m_plus_n_l208_20802

theorem min_value_m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : 45 * m = n^3) : m + n = 90 :=
sorry

end NUMINAMATH_GPT_min_value_m_plus_n_l208_20802


namespace NUMINAMATH_GPT_domain_sqrt_sin_cos_l208_20813

open Real

theorem domain_sqrt_sin_cos (k : ℤ) :
  {x : ℝ | ∃ k : ℤ, (2 * k * π + π / 4 ≤ x) ∧ (x ≤ 2 * k * π + 5 * π / 4)} = 
  {x : ℝ | sin x - cos x ≥ 0} :=
sorry

end NUMINAMATH_GPT_domain_sqrt_sin_cos_l208_20813


namespace NUMINAMATH_GPT_product_of_two_numbers_l208_20826

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h₁ : x - y = 8) 
  (h₂ : x^2 + y^2 = 160) 
  : x * y = 48 := 
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l208_20826


namespace NUMINAMATH_GPT_find_abc_l208_20838

theorem find_abc (a b c : ℚ) 
  (h1 : a + b + c = 24)
  (h2 : a + 2 * b = 2 * c)
  (h3 : a = b / 2) : 
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_abc_l208_20838


namespace NUMINAMATH_GPT_sqrt_inequality_sum_of_squares_geq_sum_of_products_l208_20800

theorem sqrt_inequality : (Real.sqrt 6) + (Real.sqrt 10) > (2 * Real.sqrt 3) + 2 := by
  sorry

theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
    a^2 + b^2 + c^2 ≥ a * b + b * c + a * c := by
  sorry

end NUMINAMATH_GPT_sqrt_inequality_sum_of_squares_geq_sum_of_products_l208_20800


namespace NUMINAMATH_GPT_problem_1_problem_2_l208_20864

-- Define the function f(x) = |x + a| + |x|
def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) + abs x

-- (Ⅰ) Prove that for a = 1, the solution set for f(x) ≥ 2 is (-∞, -1/2] ∪ [3/2, +∞)
theorem problem_1 : 
  ∀ (x : ℝ), f x 1 ≥ 2 ↔ (x ≤ -1/2 ∨ x ≥ 3/2) :=
by
  intro x
  sorry

-- (Ⅱ) Prove that if there exists x ∈ ℝ such that f(x) < 2, then -2 < a < 2
theorem problem_2 :
  (∃ (x : ℝ), f x a < 2) → -2 < a ∧ a < 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l208_20864


namespace NUMINAMATH_GPT_solve_quadratic_l208_20803

def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem solve_quadratic : (quadratic_eq (-2) 1 3 (-1)) ∧ (quadratic_eq (-2) 1 3 (3/2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l208_20803


namespace NUMINAMATH_GPT_evaluate_f_at_3_l208_20873

-- Function definition
def f (x : ℚ) : ℚ := (x - 2) / (4 * x + 5)

-- Problem statement
theorem evaluate_f_at_3 : f 3 = 1 / 17 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_3_l208_20873


namespace NUMINAMATH_GPT_complex_division_l208_20871

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 - i) = -1 + i :=
by sorry

end NUMINAMATH_GPT_complex_division_l208_20871


namespace NUMINAMATH_GPT_black_white_difference_l208_20878

theorem black_white_difference (m n : ℕ) (h_dim : m = 7 ∧ n = 9) (h_first_black : m % 2 = 1 ∧ n % 2 = 1) :
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  black_count - white_count = 1 := 
by
  -- We start with known dimensions and conditions
  let ⟨hm, hn⟩ := h_dim
  have : m = 7 := by rw [hm]
  have : n = 9 := by rw [hn]
  
  -- Calculate the number of black and white squares 
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  
  -- Use given formulas to calculate the difference
  have diff : black_count - white_count = 1 := by
    sorry -- proof to be provided
  
  exact diff

end NUMINAMATH_GPT_black_white_difference_l208_20878


namespace NUMINAMATH_GPT_tree_placement_impossible_l208_20820

theorem tree_placement_impossible
  (length width : ℝ) (h_length : length = 4) (h_width : width = 1) :
  ¬ (∃ (t1 t2 t3 : ℝ × ℝ), 
       dist t1 t2 ≥ 2.5 ∧ 
       dist t2 t3 ≥ 2.5 ∧ 
       dist t1 t3 ≥ 2.5 ∧ 
       t1.1 ≥ 0 ∧ t1.1 ≤ length ∧ t1.2 ≥ 0 ∧ t1.2 ≤ width ∧ 
       t2.1 ≥ 0 ∧ t2.1 ≤ length ∧ t2.2 ≥ 0 ∧ t2.2 ≤ width ∧ 
       t3.1 ≥ 0 ∧ t3.1 ≤ length ∧ t3.2 ≥ 0 ∧ t3.2 ≤ width) := 
by {
  sorry
}

end NUMINAMATH_GPT_tree_placement_impossible_l208_20820


namespace NUMINAMATH_GPT_printer_cost_comparison_l208_20824

-- Definitions based on the given conditions
def in_store_price : ℝ := 150.00
def discount_rate : ℝ := 0.10
def installment_payment : ℝ := 28.00
def number_of_installments : ℕ := 5
def shipping_handling_charge : ℝ := 12.50

-- Discounted in-store price calculation
def discounted_in_store_price : ℝ := in_store_price * (1 - discount_rate)

-- Total cost from the television advertiser
def tv_advertiser_total_cost : ℝ := (number_of_installments * installment_payment) + shipping_handling_charge

-- Proof statement
theorem printer_cost_comparison :
  discounted_in_store_price - tv_advertiser_total_cost = -17.50 :=
by
  sorry

end NUMINAMATH_GPT_printer_cost_comparison_l208_20824


namespace NUMINAMATH_GPT_tan_value_l208_20863

theorem tan_value (α : ℝ) 
  (h : (2 * Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / (Real.sqrt 2 * Real.sin (2 * α + π / 4)) = 4) : 
  Real.tan (2 * α + π / 4) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_value_l208_20863


namespace NUMINAMATH_GPT_part1_part2_l208_20812

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x ^ 2 + 1

theorem part1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (0 < a ∧ a < 1) :=
sorry

theorem part2 (a : ℝ) :
  (∃ α β m : ℝ, 1 ≤ α ∧ α ≤ 4 ∧ 1 ≤ β ∧ β ≤ 4 ∧ β - α = 1 ∧ f α a = m ∧ f β a = m) ↔ 
  (Real.log 4 / 3 * (2 / 7) ≤ a ∧ a ≤ Real.log 2 * (2 / 3)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l208_20812


namespace NUMINAMATH_GPT_daily_sales_volume_relationship_maximize_daily_sales_profit_l208_20857

variables (x : ℝ) (y : ℝ) (P : ℝ)

-- Conditions
def cost_per_box : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def decrease_in_sales_volume_per_dollar : ℝ := 20

-- The functional relationship between y and x
theorem daily_sales_volume_relationship (hx : min_selling_price ≤ x ∧ x < 80) : y = -20 * x + 1600 := by
  sorry

-- The profit function
def profit_function (x : ℝ) := (x - cost_per_box) * (initial_sales_volume - decrease_in_sales_volume_per_dollar * (x - initial_selling_price))

-- Maximizing the profit
theorem maximize_daily_sales_profit : ∃ x_max, x_max = 60 ∧ P = profit_function 60 ∧ P = 8000 := by
  sorry

end NUMINAMATH_GPT_daily_sales_volume_relationship_maximize_daily_sales_profit_l208_20857


namespace NUMINAMATH_GPT_minimum_value_f_l208_20842

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / 2 + 2 / (Real.sin x)

theorem minimum_value_f (x : ℝ) (h : 0 < x ∧ x ≤ Real.pi / 2) :
  ∃ y, (∀ z, 0 < z ∧ z ≤ Real.pi / 2 → f z ≥ y) ∧ y = 5 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_l208_20842


namespace NUMINAMATH_GPT_angle_B_l208_20805

-- Define the conditions
variables {A B C : ℝ} (a b c : ℝ)
variable (h : a^2 + c^2 = b^2 + ac)

-- State the theorem
theorem angle_B (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  B = π / 3 :=
sorry

end NUMINAMATH_GPT_angle_B_l208_20805


namespace NUMINAMATH_GPT_base_number_of_equation_l208_20870

theorem base_number_of_equation (y : ℕ) (b : ℕ) (h1 : 16 ^ y = b ^ 14) (h2 : y = 7) : b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_base_number_of_equation_l208_20870


namespace NUMINAMATH_GPT_bouquets_needed_to_earn_1000_l208_20801

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end NUMINAMATH_GPT_bouquets_needed_to_earn_1000_l208_20801


namespace NUMINAMATH_GPT_minimum_value_fraction_1_x_plus_1_y_l208_20841

theorem minimum_value_fraction_1_x_plus_1_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) :
  1 / x + 1 / y = 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_fraction_1_x_plus_1_y_l208_20841


namespace NUMINAMATH_GPT_three_gt_sqrt_seven_l208_20815

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := sorry

end NUMINAMATH_GPT_three_gt_sqrt_seven_l208_20815


namespace NUMINAMATH_GPT_twenty_one_less_than_sixty_thousand_l208_20885

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 :=
by
  sorry

end NUMINAMATH_GPT_twenty_one_less_than_sixty_thousand_l208_20885


namespace NUMINAMATH_GPT_Sunzi_problem_correctness_l208_20844

theorem Sunzi_problem_correctness (x y : ℕ) :
  3 * (x - 2) = 2 * x + 9 ∧ (y / 3) + 2 = (y - 9) / 2 :=
by
  sorry

end NUMINAMATH_GPT_Sunzi_problem_correctness_l208_20844


namespace NUMINAMATH_GPT_factorize_expression_l208_20829

theorem factorize_expression (a b : ℝ) :
  4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l208_20829


namespace NUMINAMATH_GPT_proportion_equation_l208_20880

theorem proportion_equation (x y : ℝ) (h : 3 * x = 4 * y) (hy : y ≠ 0) : (x / 4 = y / 3) :=
by
  sorry

end NUMINAMATH_GPT_proportion_equation_l208_20880


namespace NUMINAMATH_GPT_trapezoidal_field_base_count_l208_20814

theorem trapezoidal_field_base_count
  (A : ℕ) (h : ℕ) (b1 b2 : ℕ)
  (hdiv8 : ∃ m n : ℕ, b1 = 8 * m ∧ b2 = 8 * n)
  (area_eq : A = (h * (b1 + b2)) / 2)
  (A_val : A = 1400)
  (h_val : h = 50) :
  (∃ pair1 pair2 pair3, (pair1 + pair2 + pair3 = (b1 + b2))) :=
by
  sorry

end NUMINAMATH_GPT_trapezoidal_field_base_count_l208_20814


namespace NUMINAMATH_GPT_inequality_transformation_l208_20884

variable (x y : ℝ)

theorem inequality_transformation (h : x > y) : x - 2 > y - 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_transformation_l208_20884


namespace NUMINAMATH_GPT_barrels_are_1360_l208_20879

-- Defining the top layer dimensions and properties
def a : ℕ := 2
def b : ℕ := 1
def n : ℕ := 15

-- Defining the dimensions of the bottom layer based on given properties
def c : ℕ := a + n
def d : ℕ := b + n

-- Formula for the total number of barrels
def total_barrels : ℕ := n * ((2 * a + c) * b + (2 * c + a) * d + (d - b)) / 6

-- Theorem to prove
theorem barrels_are_1360 : total_barrels = 1360 :=
by
  sorry

end NUMINAMATH_GPT_barrels_are_1360_l208_20879


namespace NUMINAMATH_GPT_daisy_lunch_vs_breakfast_spending_l208_20869

noncomputable def breakfast_cost : ℝ := 2 + 3 + 4 + 3.5 + 1.5
noncomputable def lunch_base_cost : ℝ := 3.5 + 4 + 5.25 + 6 + 1 + 3
noncomputable def service_charge : ℝ := 0.10 * lunch_base_cost
noncomputable def lunch_cost_with_service_charge : ℝ := lunch_base_cost + service_charge
noncomputable def food_tax : ℝ := 0.05 * lunch_cost_with_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_with_service_charge + food_tax
noncomputable def difference : ℝ := total_lunch_cost - breakfast_cost

theorem daisy_lunch_vs_breakfast_spending :
  difference = 12.28 :=
by 
  sorry

end NUMINAMATH_GPT_daisy_lunch_vs_breakfast_spending_l208_20869


namespace NUMINAMATH_GPT_readers_in_group_l208_20896

theorem readers_in_group (S L B T : ℕ) (hS : S = 120) (hL : L = 90) (hB : B = 60) :
  T = S + L - B → T = 150 :=
by
  intro h₁
  rw [hS, hL, hB] at h₁
  linarith

end NUMINAMATH_GPT_readers_in_group_l208_20896


namespace NUMINAMATH_GPT_sequence_divisible_by_11_l208_20830

theorem sequence_divisible_by_11 {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : a 2 = 3)
    (h_rec : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
    (a 4 % 11 = 0) ∧ (a 8 % 11 = 0) ∧ (a 10 % 11 = 0) ∧ (∀ n, n ≥ 11 → a n % 11 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sequence_divisible_by_11_l208_20830


namespace NUMINAMATH_GPT_original_number_l208_20876

theorem original_number (n : ℕ) (h1 : 100000 ≤ n ∧ n < 1000000) (h2 : n / 100000 = 7) (h3 : (n % 100000) * 10 + 7 = n / 5) : n = 714285 :=
sorry

end NUMINAMATH_GPT_original_number_l208_20876


namespace NUMINAMATH_GPT_sum_of_decimals_l208_20840

theorem sum_of_decimals : (5.76 + 4.29 = 10.05) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l208_20840


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l208_20872

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem sufficient_but_not_necessary_condition (h: ∀ x : ℝ, p x → q x) : (∀ x : ℝ, q x → p x) → false := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l208_20872


namespace NUMINAMATH_GPT_probability_two_dice_sum_gt_8_l208_20831

def num_ways_to_get_sum_at_most_8 := 
  1 + 2 + 3 + 4 + 5 + 6 + 5

def total_outcomes := 36

def probability_sum_greater_than_8 : ℚ := 1 - (num_ways_to_get_sum_at_most_8 / total_outcomes)

theorem probability_two_dice_sum_gt_8 :
  probability_sum_greater_than_8 = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_dice_sum_gt_8_l208_20831


namespace NUMINAMATH_GPT_evaluate_polynomial_at_3_l208_20874

def f (x : ℕ) : ℕ := 3 * x^7 + 2 * x^5 + 4 * x^3 + x

theorem evaluate_polynomial_at_3 : f 3 = 7158 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_3_l208_20874


namespace NUMINAMATH_GPT_second_tap_fills_in_15_hours_l208_20895

theorem second_tap_fills_in_15_hours 
  (r1 r3 : ℝ) 
  (x : ℝ) 
  (H1 : r1 = 1 / 10) 
  (H2 : r3 = 1 / 6) 
  (H3 : r1 + 1 / x + r3 = 1 / 3) : 
  x = 15 :=
sorry

end NUMINAMATH_GPT_second_tap_fills_in_15_hours_l208_20895


namespace NUMINAMATH_GPT_distance_from_edge_to_bottom_l208_20848

theorem distance_from_edge_to_bottom (d x : ℕ) 
  (h1 : 63 + d + 20 = 10 + d + x) : x = 73 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_distance_from_edge_to_bottom_l208_20848


namespace NUMINAMATH_GPT_sampling_probability_equal_l208_20867

theorem sampling_probability_equal :
  let total_people := 2014
  let first_sample := 14
  let remaining_people := total_people - first_sample
  let sample_size := 50
  let probability := sample_size / total_people
  50 / 2014 = 25 / 1007 :=
by
  sorry

end NUMINAMATH_GPT_sampling_probability_equal_l208_20867


namespace NUMINAMATH_GPT_theater_ticket_difference_l208_20843

theorem theater_ticket_difference
  (O B : ℕ)
  (h1 : O + B = 355)
  (h2 : 12 * O + 8 * B = 3320) :
  B - O = 115 :=
sorry

end NUMINAMATH_GPT_theater_ticket_difference_l208_20843


namespace NUMINAMATH_GPT_distance_from_point_to_asymptote_l208_20883

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_asymptote_l208_20883


namespace NUMINAMATH_GPT_extreme_value_proof_l208_20833

noncomputable def extreme_value (x y : ℝ) := 4 * x + 3 * y 

theorem extreme_value_proof 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : x + y = 5 * x * y) : 
  extreme_value x y = 3 :=
sorry

end NUMINAMATH_GPT_extreme_value_proof_l208_20833


namespace NUMINAMATH_GPT_angle_A_range_l208_20862

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_angle_A_range_l208_20862


namespace NUMINAMATH_GPT_banana_count_l208_20816

-- Variables representing the number of bananas, oranges, and apples
variables (B O A : ℕ)

-- Conditions translated from the problem statement
def conditions : Prop :=
  (O = 2 * B) ∧
  (A = 2 * O) ∧
  (B + O + A = 35)

-- Theorem to prove the number of bananas is 5 given the conditions
theorem banana_count (B O A : ℕ) (h : conditions B O A) : B = 5 :=
sorry

end NUMINAMATH_GPT_banana_count_l208_20816


namespace NUMINAMATH_GPT_smallest_b_for_34b_perfect_square_is_4_l208_20858

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end NUMINAMATH_GPT_smallest_b_for_34b_perfect_square_is_4_l208_20858


namespace NUMINAMATH_GPT_barbara_needs_more_weeks_l208_20892

/-
  Problem Statement:
  Barbara wants to save up for a new wristwatch that costs $100. Her parents give her an allowance
  of $5 a week and she can either save it all up or spend it as she wishes. 10 weeks pass and
  due to spending some of her money, Barbara currently only has $20. How many more weeks does she need
  to save for a watch if she stops spending on other things right now?
-/

def wristwatch_cost : ℕ := 100
def allowance_per_week : ℕ := 5
def current_savings : ℕ := 20
def amount_needed : ℕ := wristwatch_cost - current_savings
def weeks_needed : ℕ := amount_needed / allowance_per_week

theorem barbara_needs_more_weeks :
  weeks_needed = 16 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_barbara_needs_more_weeks_l208_20892


namespace NUMINAMATH_GPT_symmetric_points_ab_value_l208_20839

theorem symmetric_points_ab_value
  (a b : ℤ)
  (h₁ : a + 2 = -4)
  (h₂ : 2 = b) :
  a * b = -12 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_ab_value_l208_20839


namespace NUMINAMATH_GPT_sum_distances_between_l208_20865

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

theorem sum_distances_between (A B D : ℝ × ℝ)
  (hB : B = (0, 5))
  (hD : D = (8, 0))
  (hA : A = (20, 0)) :
  21 < distance A D + distance B D ∧ distance A D + distance B D < 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_distances_between_l208_20865


namespace NUMINAMATH_GPT_probability_event_A_probability_event_B_probability_event_C_l208_20834

-- Define the total number of basic events for three dice
def total_basic_events : ℕ := 6 * 6 * 6

-- Define events and their associated basic events
def event_A_basic_events : ℕ := 2 * 3 * 3
def event_B_basic_events : ℕ := 2 * 3 * 6
def event_C_basic_events : ℕ := 6 * 6 * 3

-- Define probabilities for each event
def P_A : ℚ := event_A_basic_events / total_basic_events
def P_B : ℚ := event_B_basic_events / total_basic_events
def P_C : ℚ := event_C_basic_events / total_basic_events

-- Statement to be proven
theorem probability_event_A : P_A = 1 / 12 := by
  sorry

theorem probability_event_B : P_B = 1 / 6 := by
  sorry

theorem probability_event_C : P_C = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_event_A_probability_event_B_probability_event_C_l208_20834


namespace NUMINAMATH_GPT_value_of_a_plus_b_l208_20852

def f (x : ℝ) (a b : ℝ) := x^3 + (a - 1) * x^2 + a * x + b

theorem value_of_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) → a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l208_20852


namespace NUMINAMATH_GPT_sum_primes_upto_20_l208_20890

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end NUMINAMATH_GPT_sum_primes_upto_20_l208_20890


namespace NUMINAMATH_GPT_train_waiting_probability_l208_20846

-- Conditions
def trains_per_hour : ℕ := 1
def total_minutes : ℕ := 60
def wait_time : ℕ := 10

-- Proposition
theorem train_waiting_probability : 
  (wait_time : ℝ) / (total_minutes / trains_per_hour) = 1 / 6 :=
by
  -- Here we assume the proof proceeds correctly
  sorry

end NUMINAMATH_GPT_train_waiting_probability_l208_20846


namespace NUMINAMATH_GPT_determine_speeds_l208_20877

structure Particle :=
  (speed : ℝ)

def distance : ℝ := 3.01 -- meters

def initial_distance (m1_speed : ℝ) : ℝ :=
  301 - 11 * m1_speed -- converted to cm

theorem determine_speeds :
  ∃ (m1 m2 : Particle), 
  m1.speed = 11 ∧ m2.speed = 7 ∧ 
  ∀ t : ℝ, (t = 10 ∨ t = 45) →
  (initial_distance m1.speed) = t * (m1.speed + m2.speed) ∧
  20 * m2.speed = 35 * (m1.speed - m2.speed) :=
by {
  sorry 
}

end NUMINAMATH_GPT_determine_speeds_l208_20877


namespace NUMINAMATH_GPT_measure_exterior_angle_BAC_l208_20894

-- Define the interior angle of a regular nonagon
def nonagon_interior_angle := (180 * (9 - 2)) / 9

-- Define the exterior angle of the nonagon
def nonagon_exterior_angle := 360 - nonagon_interior_angle

-- The square's interior angle
def square_interior_angle := 90

-- The question to be proven
theorem measure_exterior_angle_BAC :
  nonagon_exterior_angle - square_interior_angle = 130 :=
  by
  sorry

end NUMINAMATH_GPT_measure_exterior_angle_BAC_l208_20894


namespace NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l208_20827

-- Problem 1
theorem calc1 : (-2: ℝ) ^ 2 - (7 - Real.pi) ^ 0 - (1 / 3) ^ (-1: ℝ) = 0 := by
  sorry

-- Problem 2
variable (m : ℝ)
theorem calc2 : 2 * m ^ 3 * 3 * m - (2 * m ^ 2) ^ 2 + m ^ 6 / m ^ 2 = 3 * m ^ 4 := by
  sorry

-- Problem 3
variable (a : ℝ)
theorem calc3 : (a + 1) ^ 2 + (a + 1) * (a - 2) = 2 * a ^ 2 + a - 1 := by
  sorry

-- Problem 4
variables (x y : ℝ)
theorem calc4 : (x + y - 1) * (x - y - 1) = x ^ 2 - 2 * x + 1 - y ^ 2 := by
  sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l208_20827


namespace NUMINAMATH_GPT_rational_t_l208_20889

variable (A B t : ℚ)

theorem rational_t (A B : ℚ) (hA : A = 2 * t / (1 + t^2)) (hB : B = (1 - t^2) / (1 + t^2)) : ∃ t' : ℚ, t = t' :=
by
  sorry

end NUMINAMATH_GPT_rational_t_l208_20889


namespace NUMINAMATH_GPT_no_integer_x_square_l208_20821

theorem no_integer_x_square (x : ℤ) : 
  ∀ n : ℤ, x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1 ≠ n^2 :=
by sorry

end NUMINAMATH_GPT_no_integer_x_square_l208_20821


namespace NUMINAMATH_GPT_least_number_of_stamps_l208_20861

def min_stamps (x y : ℕ) : ℕ := x + y

theorem least_number_of_stamps {x y : ℕ} (h : 5 * x + 7 * y = 50) 
  : min_stamps x y = 8 :=
sorry

end NUMINAMATH_GPT_least_number_of_stamps_l208_20861


namespace NUMINAMATH_GPT_average_weight_of_three_l208_20868

theorem average_weight_of_three :
  ∀ A B C : ℝ,
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 :=
by
  intros A B C h1 h2 h3
  sorry

end NUMINAMATH_GPT_average_weight_of_three_l208_20868


namespace NUMINAMATH_GPT_genevieve_drinks_pints_l208_20806

theorem genevieve_drinks_pints (total_gallons : ℝ) (thermoses : ℕ) 
  (gallons_to_pints : ℝ) (genevieve_thermoses : ℕ) 
  (h1 : total_gallons = 4.5) (h2 : thermoses = 18) 
  (h3 : gallons_to_pints = 8) (h4 : genevieve_thermoses = 3) : 
  (total_gallons * gallons_to_pints / thermoses) * genevieve_thermoses = 6 := 
by
  admit

end NUMINAMATH_GPT_genevieve_drinks_pints_l208_20806


namespace NUMINAMATH_GPT_possible_values_of_N_l208_20825

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_N_l208_20825


namespace NUMINAMATH_GPT_apples_in_basket_l208_20899

theorem apples_in_basket
  (total_rotten : ℝ := 12 / 100)
  (total_spots : ℝ := 7 / 100)
  (total_insects : ℝ := 5 / 100)
  (total_varying_rot : ℝ := 3 / 100)
  (perfect_apples : ℝ := 66) :
  (perfect_apples / ((1 - (total_rotten + total_spots + total_insects + total_varying_rot))) = 90) :=
by
  sorry

end NUMINAMATH_GPT_apples_in_basket_l208_20899


namespace NUMINAMATH_GPT_train_crosses_platform_in_25_002_seconds_l208_20854

noncomputable def time_to_cross_platform 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (speed_kmph : ℝ) : ℝ := 
  let total_distance := length_train + length_platform
  let speed_mps := speed_kmph * (1000 / 3600)
  total_distance / speed_mps

theorem train_crosses_platform_in_25_002_seconds :
  time_to_cross_platform 225 400.05 90 = 25.002 := by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_in_25_002_seconds_l208_20854


namespace NUMINAMATH_GPT_alex_ate_more_pears_than_sam_l208_20881

namespace PearEatingContest

def number_of_pears_eaten (Alex Sam : ℕ) : ℕ :=
  Alex - Sam

theorem alex_ate_more_pears_than_sam :
  number_of_pears_eaten 8 2 = 6 := by
  -- proof
  sorry

end PearEatingContest

end NUMINAMATH_GPT_alex_ate_more_pears_than_sam_l208_20881


namespace NUMINAMATH_GPT_total_cost_of_tickets_l208_20818

theorem total_cost_of_tickets (num_family_members num_adult_tickets num_children_tickets : ℕ)
    (cost_adult_ticket cost_children_ticket total_cost : ℝ) 
    (h1 : num_family_members = 7) 
    (h2 : cost_adult_ticket = 21) 
    (h3 : cost_children_ticket = 14) 
    (h4 : num_adult_tickets = 4) 
    (h5 : num_children_tickets = num_family_members - num_adult_tickets) 
    (h6 : total_cost = num_adult_tickets * cost_adult_ticket + num_children_tickets * cost_children_ticket) :
    total_cost = 126 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_tickets_l208_20818


namespace NUMINAMATH_GPT_construct_triangle_l208_20898

variable (h_a h_b h_c : ℝ)

noncomputable def triangle_exists_and_similar :=
  ∃ (a b c : ℝ), (a = h_b) ∧ (b = h_a) ∧ (c = h_a * h_b / h_c) ∧
  (∃ (area : ℝ), area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c)

theorem construct_triangle (h_a h_b h_c : ℝ) :
  ∃ a b c, a = h_b ∧ b = h_a ∧ c = h_a * h_b / h_c ∧
  ∃ area, area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c := 
  sorry

end NUMINAMATH_GPT_construct_triangle_l208_20898


namespace NUMINAMATH_GPT_even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l208_20875

theorem even_not_divisible_by_4_not_sum_of_two_consecutive_odds (x n : ℕ) (h₁ : Even x) (h₂ : ¬ ∃ k, x = 4 * k) : x ≠ (2 * n + 1) + (2 * n + 3) := by
  sorry

end NUMINAMATH_GPT_even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l208_20875


namespace NUMINAMATH_GPT_find_f_of_3_l208_20828

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end NUMINAMATH_GPT_find_f_of_3_l208_20828
