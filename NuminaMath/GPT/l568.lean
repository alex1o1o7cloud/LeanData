import Mathlib

namespace NUMINAMATH_GPT_ones_digit_of_sum_of_powers_l568_56874

theorem ones_digit_of_sum_of_powers :
  (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_sum_of_powers_l568_56874


namespace NUMINAMATH_GPT_no_savings_if_purchased_together_l568_56859

def window_price : ℕ := 120

def free_windows (purchased_windows : ℕ) : ℕ :=
  (purchased_windows / 10) * 2

def total_cost (windows_needed : ℕ) : ℕ :=
  (windows_needed - free_windows windows_needed) * window_price

def separate_cost : ℕ :=
  total_cost 9 + total_cost 11 + total_cost 10

def joint_cost : ℕ :=
  total_cost 30

theorem no_savings_if_purchased_together :
  separate_cost = joint_cost :=
by
  -- Proof will be provided here, currently skipped.
  sorry

end NUMINAMATH_GPT_no_savings_if_purchased_together_l568_56859


namespace NUMINAMATH_GPT_solve_pears_and_fruits_l568_56837

noncomputable def pears_and_fruits_problem : Prop :=
  ∃ (x y : ℕ), x + y = 1000 ∧ (11 * x) * (1/9 : ℚ) + (4 * y) * (1/7 : ℚ) = 999

theorem solve_pears_and_fruits :
  pears_and_fruits_problem := by
  sorry

end NUMINAMATH_GPT_solve_pears_and_fruits_l568_56837


namespace NUMINAMATH_GPT_sahil_selling_price_l568_56831

-- Defining the conditions as variables
def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Defining the total cost
def total_cost : ℕ := purchase_price + repair_cost + transportation_charges

-- Calculating the profit amount
def profit : ℕ := (profit_percentage * total_cost) / 100

-- Calculating the selling price
def selling_price : ℕ := total_cost + profit

-- The Lean statement to prove the selling price is Rs 30,000
theorem sahil_selling_price : selling_price = 30000 :=
by 
  simp [total_cost, profit, selling_price]
  sorry

end NUMINAMATH_GPT_sahil_selling_price_l568_56831


namespace NUMINAMATH_GPT_smallest_a_l568_56806

theorem smallest_a (a x : ℤ) (hx : x^2 + a * x = 30) (ha_pos : a > 0) (product_gt_30 : ∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = -a ∧ x₁ * x₂ > 30) : a = 11 :=
sorry

end NUMINAMATH_GPT_smallest_a_l568_56806


namespace NUMINAMATH_GPT_find_factor_l568_56851

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_factor_l568_56851


namespace NUMINAMATH_GPT_count_4_tuples_l568_56839

theorem count_4_tuples (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  Nat.card {abcd : ℕ × ℕ × ℕ × ℕ // (0 < abcd.1 ∧ abcd.1 < p) ∧ 
                                     (0 < abcd.2.1 ∧ abcd.2.1 < p) ∧ 
                                     (0 < abcd.2.2.1 ∧ abcd.2.2.1 < p) ∧ 
                                     (0 < abcd.2.2.2 ∧ abcd.2.2.2 < p) ∧ 
                                     ((abcd.1 * abcd.2.2.2 - abcd.2.1 * abcd.2.2.1) % p = 0)} = (p - 1) * (p - 1) * (p - 1) :=
by
  sorry

end NUMINAMATH_GPT_count_4_tuples_l568_56839


namespace NUMINAMATH_GPT_house_height_l568_56847

theorem house_height
  (tree_height : ℕ) (tree_shadow : ℕ)
  (house_shadow : ℕ) (h : ℕ) :
  tree_height = 15 →
  tree_shadow = 18 →
  house_shadow = 72 →
  (h / tree_height) = (house_shadow / tree_shadow) →
  h = 60 :=
by
  intros h1 h2 h3 h4
  have h5 : h / 15 = 72 / 18 := by
    rw [h1, h2, h3] at h4
    exact h4
  sorry

end NUMINAMATH_GPT_house_height_l568_56847


namespace NUMINAMATH_GPT_max_value_h3_solve_for_h_l568_56826

-- Definition part for conditions
def quadratic_function (h : ℝ) (x : ℝ) : ℝ :=
  -(x - h) ^ 2

-- Part (1): When h = 3, proving the maximum value of the function within 2 ≤ x ≤ 5 is 0.
theorem max_value_h3 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function 3 x ≤ 0 :=
by
  sorry

-- Part (2): If the maximum value of the function is -1, then the value of h is 6 or 1.
theorem solve_for_h (h : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function h x ≤ -1) ↔ h = 6 ∨ h = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_h3_solve_for_h_l568_56826


namespace NUMINAMATH_GPT_x_intercept_of_perpendicular_line_l568_56828

theorem x_intercept_of_perpendicular_line (x y : ℝ) (h1 : 5 * x - 3 * y = 9) (y_intercept : ℝ) 
  (h2 : y_intercept = 4) : x = 20 / 3 :=
sorry

end NUMINAMATH_GPT_x_intercept_of_perpendicular_line_l568_56828


namespace NUMINAMATH_GPT_B_join_time_l568_56889

theorem B_join_time (x : ℕ) (hx : (45000 * 12) / (27000 * (12 - x)) = 2) : x = 2 :=
sorry

end NUMINAMATH_GPT_B_join_time_l568_56889


namespace NUMINAMATH_GPT_value_two_sd_below_mean_l568_56892

theorem value_two_sd_below_mean :
  let mean := 14.5
  let stdev := 1.7
  mean - 2 * stdev = 11.1 :=
by
  sorry

end NUMINAMATH_GPT_value_two_sd_below_mean_l568_56892


namespace NUMINAMATH_GPT_initial_ratio_l568_56879

theorem initial_ratio (partners associates associates_after_hiring : ℕ)
  (h_partners : partners = 20)
  (h_associates_after_hiring : associates_after_hiring = 20 * 34)
  (h_assoc_equation : associates + 50 = associates_after_hiring) :
  (partners : ℚ) / associates = 2 / 63 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_l568_56879


namespace NUMINAMATH_GPT_asymptote_of_hyperbola_l568_56811

theorem asymptote_of_hyperbola (h : (∀ x y : ℝ, y^2 / 3 - x^2 / 2 = 1)) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (sqrt6 / 2) * x ∨ y = - (sqrt6 / 2) * x) :=
sorry

end NUMINAMATH_GPT_asymptote_of_hyperbola_l568_56811


namespace NUMINAMATH_GPT_part1_part2_l568_56848

-- Part (1) Lean 4 statement
theorem part1 {x : ℕ} (h : 0 < x ∧ 4 * (x + 2) < 18 + 2 * x) : x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 :=
sorry

-- Part (2) Lean 4 statement
theorem part2 (x : ℝ) (h1 : 5 * x + 2 ≥ 4 * x + 1) (h2 : (x + 1) / 4 > (x - 3) / 2 + 1) : -1 ≤ x ∧ x < 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l568_56848


namespace NUMINAMATH_GPT_range_of_a_l568_56808

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 0 → x^2 + 2 * x - 3 + a ≤ 0) ↔ a ≤ -12 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l568_56808


namespace NUMINAMATH_GPT_no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l568_56842

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l568_56842


namespace NUMINAMATH_GPT_advertisements_shown_l568_56832

theorem advertisements_shown (advertisement_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) :
  advertisement_duration = 3 →
  cost_per_minute = 4000 →
  total_cost = 60000 →
  total_cost / (advertisement_duration * cost_per_minute) = 5 :=
by
  sorry

end NUMINAMATH_GPT_advertisements_shown_l568_56832


namespace NUMINAMATH_GPT_determine_a_l568_56849

theorem determine_a
  (h : ∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a * x - 2) ≥ 0) : 
  a = 1 :=
sorry

end NUMINAMATH_GPT_determine_a_l568_56849


namespace NUMINAMATH_GPT_Isabel_reading_pages_l568_56865

def pages_of_math_homework : ℕ := 2
def problems_per_page : ℕ := 5
def total_problems : ℕ := 30

def math_problems : ℕ := pages_of_math_homework * problems_per_page
def reading_problems : ℕ := total_problems - math_problems

theorem Isabel_reading_pages : (reading_problems / problems_per_page) = 4 :=
by
  sorry

end NUMINAMATH_GPT_Isabel_reading_pages_l568_56865


namespace NUMINAMATH_GPT_area_converted_2018_l568_56807

theorem area_converted_2018 :
  let a₁ := 8 -- initial area in ten thousand hectares
  let q := 1.1 -- common ratio
  let a₆ := a₁ * q^5 -- area converted in 2018
  a₆ = 8 * 1.1^5 :=
sorry

end NUMINAMATH_GPT_area_converted_2018_l568_56807


namespace NUMINAMATH_GPT_correct_calculation_l568_56834

theorem correct_calculation (x : ℤ) (h : 20 + x = 60) : 34 - x = -6 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l568_56834


namespace NUMINAMATH_GPT_simplify_and_evaluate_l568_56804

theorem simplify_and_evaluate :
  ∀ (a b : ℤ), a = -1 → b = 4 →
  (a + b)^2 - 2 * a * (a - b) + (a + 2 * b) * (a - 2 * b) = -64 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l568_56804


namespace NUMINAMATH_GPT_log2_3_value_l568_56830

variables (a b log2 log3 : ℝ)

-- Define the conditions
axiom h1 : a = log2 + log3
axiom h2 : b = 1 + log2

-- Define the logarithmic requirement to be proved
theorem log2_3_value : log2 * log3 = (a - b + 1) / (b - 1) :=
sorry

end NUMINAMATH_GPT_log2_3_value_l568_56830


namespace NUMINAMATH_GPT_polygon_interior_exterior_angles_l568_56816

theorem polygon_interior_exterior_angles (n : ℕ) :
  (n - 2) * 180 = 360 + 720 → n = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_polygon_interior_exterior_angles_l568_56816


namespace NUMINAMATH_GPT_find_A_find_B_l568_56898

-- First problem: Prove A = 10 given 100A = 35^2 - 15^2
theorem find_A (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) : A = 10 := by
  sorry

-- Second problem: Prove B = 4 given (A-1)^6 = 27^B and A = 10
theorem find_B (B : ℕ) (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) (h₂ : (A - 1) ^ 6 = 27 ^ B) : B = 4 := by
  have A_is_10 : A = 10 := by
    apply find_A
    assumption
  sorry

end NUMINAMATH_GPT_find_A_find_B_l568_56898


namespace NUMINAMATH_GPT_geometric_sequence_ratio_28_l568_56821

noncomputable def geometric_sequence_sum_ratio (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :=
  S 6 / S 3 = 28

theorem geometric_sequence_ratio_28 (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_GS : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h_increasing : ∀ n m, n < m → a1 * q^n < a1 * q^m) 
  (h_mean : 2 * 6 * a1 * q^6 = a1 * q^7 + a1 * q^8) : 
  geometric_sequence_sum_ratio a1 q S := 
by {
  -- Proof should be completed here
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_ratio_28_l568_56821


namespace NUMINAMATH_GPT_seashells_second_day_l568_56829

theorem seashells_second_day (x : ℕ) (h1 : 5 + x + 2 * (5 + x) = 36) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_seashells_second_day_l568_56829


namespace NUMINAMATH_GPT_difficult_vs_easy_l568_56823

theorem difficult_vs_easy (x y z : ℕ) (h1 : x + y + z = 100) (h2 : x + 3 * y + 2 * z = 180) :
  x - y = 20 :=
by sorry

end NUMINAMATH_GPT_difficult_vs_easy_l568_56823


namespace NUMINAMATH_GPT_weight_of_piece_l568_56818

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_piece_l568_56818


namespace NUMINAMATH_GPT_max_height_l568_56844

-- Define the parabolic function h(t) representing the height of the soccer ball.
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

-- State that the maximum height of the soccer ball is 136 feet.
theorem max_height : ∃ t : ℝ, h t = 136 :=
by
  sorry

end NUMINAMATH_GPT_max_height_l568_56844


namespace NUMINAMATH_GPT_could_not_be_diagonal_lengths_l568_56876

-- Definitions of the diagonal conditions
def diagonal_condition (s : List ℕ) : Prop :=
  match s with
  | [x, y, z] => x^2 + y^2 > z^2 ∧ x^2 + z^2 > y^2 ∧ y^2 + z^2 > x^2
  | _ => false

-- Statement of the problem
theorem could_not_be_diagonal_lengths : 
  ¬ diagonal_condition [5, 6, 8] :=
by 
  sorry

end NUMINAMATH_GPT_could_not_be_diagonal_lengths_l568_56876


namespace NUMINAMATH_GPT_factorize_m_cubed_minus_16m_l568_56872

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end NUMINAMATH_GPT_factorize_m_cubed_minus_16m_l568_56872


namespace NUMINAMATH_GPT_determine_original_price_l568_56866

namespace PriceProblem

variable (x : ℝ)

def final_price (x : ℝ) : ℝ := 0.98175 * x

theorem determine_original_price (h : final_price x = 100) : x = 101.86 :=
by
  sorry

end PriceProblem

end NUMINAMATH_GPT_determine_original_price_l568_56866


namespace NUMINAMATH_GPT_timeAfter2687Minutes_l568_56888

-- We define a structure for representing time in hours and minutes.
structure Time :=
  (hour : Nat)
  (minute : Nat)

-- Define the current time
def currentTime : Time := {hour := 7, minute := 0}

-- Define a function that computes the time after adding a given number of minutes to a given time
noncomputable def addMinutes (t : Time) (minutesToAdd : Nat) : Time :=
  let totalMinutes := t.minute + minutesToAdd
  let extraHours := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  let totalHours := t.hour + extraHours
  let effectiveHours := totalHours % 24
  {hour := effectiveHours, minute := remainingMinutes}

-- The theorem to state that 2687 minutes after 7:00 a.m. is 3:47 a.m.
theorem timeAfter2687Minutes : addMinutes currentTime 2687 = { hour := 3, minute := 47 } :=
  sorry

end NUMINAMATH_GPT_timeAfter2687Minutes_l568_56888


namespace NUMINAMATH_GPT_parabola_intersection_l568_56825

theorem parabola_intersection:
  (∀ x y1 y2 : ℝ, (y1 = 3 * x^2 - 6 * x + 6) ∧ (y2 = -2 * x^2 - 4 * x + 6) → y1 = y2 → x = 0 ∨ x = 2 / 5) ∧
  (∀ a c : ℝ, a = 0 ∧ c = 2 / 5 ∧ c ≥ a → c - a = 2 / 5) :=
by sorry

end NUMINAMATH_GPT_parabola_intersection_l568_56825


namespace NUMINAMATH_GPT_oblique_prism_volume_l568_56877

noncomputable def volume_of_oblique_prism 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : ℝ :=
  a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2)

theorem oblique_prism_volume 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : volume_of_oblique_prism a b c α β hα hβ = a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2) := 
by
  -- Proof will be completed here
  sorry

end NUMINAMATH_GPT_oblique_prism_volume_l568_56877


namespace NUMINAMATH_GPT_value_of_a_star_b_l568_56867

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_value_of_a_star_b_l568_56867


namespace NUMINAMATH_GPT_evaluate_expression_l568_56843

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l568_56843


namespace NUMINAMATH_GPT_student_selection_l568_56855

theorem student_selection (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 4) : a + b + c = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_student_selection_l568_56855


namespace NUMINAMATH_GPT_largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l568_56809

theorem largest_integer_less_than_80_with_remainder_3_when_divided_by_5 : 
  ∃ x : ℤ, x < 80 ∧ x % 5 = 3 ∧ (∀ y : ℤ, y < 80 ∧ y % 5 = 3 → y ≤ x) :=
sorry

end NUMINAMATH_GPT_largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l568_56809


namespace NUMINAMATH_GPT_compute_expression_l568_56875

-- Definition of the operation "minus the reciprocal of"
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement to prove the given problem
theorem compute_expression :
  ((diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5))) = -71 / 380 := 
sorry

end NUMINAMATH_GPT_compute_expression_l568_56875


namespace NUMINAMATH_GPT_rhombus_area_l568_56820

theorem rhombus_area (x y : ℝ)
  (h1 : x^2 + y^2 = 113) 
  (h2 : x = y + 8) : 
  1 / 2 * (2 * y) * (2 * (y + 4)) = 97 := 
by 
  -- Assume x and y are the half-diagonals of the rhombus
  sorry

end NUMINAMATH_GPT_rhombus_area_l568_56820


namespace NUMINAMATH_GPT_functional_eq_solutions_l568_56822

theorem functional_eq_solutions
  (f : ℚ → ℚ)
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  ∀ x : ℚ, f x = x ∨ f x = -x := 
sorry

end NUMINAMATH_GPT_functional_eq_solutions_l568_56822


namespace NUMINAMATH_GPT_cone_slice_ratio_l568_56891

theorem cone_slice_ratio (h r : ℝ) (hb : h > 0) (hr : r > 0) :
    let V1 := (1/3) * π * (5*r)^2 * (5*h) - (1/3) * π * (4*r)^2 * (4*h)
    let V2 := (1/3) * π * (4*r)^2 * (4*h) - (1/3) * π * (3*r)^2 * (3*h)
    V2 / V1 = 37 / 61 := by {
  sorry
}

end NUMINAMATH_GPT_cone_slice_ratio_l568_56891


namespace NUMINAMATH_GPT_minimum_value_inequality_l568_56833

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 64) :
  ∃ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 64 ∧ (x^2 + 8 * x * y + 4 * y^2 + 4 * z^2) = 384 := 
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l568_56833


namespace NUMINAMATH_GPT_find_positive_integers_l568_56812

theorem find_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / m + 1 / n - 1 / (m * n) = 2 / 5) ↔ 
  (m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4) :=
by sorry

end NUMINAMATH_GPT_find_positive_integers_l568_56812


namespace NUMINAMATH_GPT_problem_statement_l568_56801

theorem problem_statement 
  (x y z : ℝ) 
  (hx1 : x ≠ 1) 
  (hy1 : y ≠ 1) 
  (hz1 : z ≠ 1) 
  (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l568_56801


namespace NUMINAMATH_GPT_john_spending_l568_56857

theorem john_spending (X : ℝ) 
  (H1 : X * (1 / 4) + X * (1 / 3) + X * (1 / 6) + 6 = X) : 
  X = 24 := 
sorry

end NUMINAMATH_GPT_john_spending_l568_56857


namespace NUMINAMATH_GPT_find_f_of_7_l568_56878

-- Defining the conditions in the problem.
variables (f : ℝ → ℝ)
variables (odd_f : ∀ x : ℝ, f (-x) = -f x)
variables (periodic_f : ∀ x : ℝ, f (x + 4) = f x)
variables (f_eqn : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = x + 2)

-- The statement of the problem, to prove f(7) = -3.
theorem find_f_of_7 : f 7 = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_7_l568_56878


namespace NUMINAMATH_GPT_find_g_neg_six_l568_56853

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end NUMINAMATH_GPT_find_g_neg_six_l568_56853


namespace NUMINAMATH_GPT_solution_set_of_inequality_l568_56880

variables {R : Type*} [LinearOrderedField R]

-- Define f as an even function
def even_function (f : R → R) := ∀ x : R, f x = f (-x)

-- Define f as an increasing function on [0, +∞)
def increasing_on_nonneg (f : R → R) := ∀ ⦃x y : R⦄, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the hypothesis and the theorem
theorem solution_set_of_inequality (f : R → R)
  (h_even : even_function f)
  (h_inc : increasing_on_nonneg f) :
  { x : R | f x > f 1 } = { x : R | x > 1 ∨ x < -1 } :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_of_inequality_l568_56880


namespace NUMINAMATH_GPT_actual_area_of_region_l568_56850

-- Problem Definitions
def map_scale : ℕ := 300000
def map_area_cm_squared : ℕ := 24

-- The actual area calculation should be 216 km²
theorem actual_area_of_region :
  let scale_factor_distance := map_scale
  let scale_factor_area := scale_factor_distance ^ 2
  let actual_area_cm_squared := map_area_cm_squared * scale_factor_area
  let actual_area_km_squared := actual_area_cm_squared / 10^10
  actual_area_km_squared = 216 := 
by
  sorry

end NUMINAMATH_GPT_actual_area_of_region_l568_56850


namespace NUMINAMATH_GPT_union_sets_l568_56810

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def setB : Set ℝ := { x | x^2 - 7 * x + 10 ≤ 0 }

theorem union_sets : setA ∪ setB = { x | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l568_56810


namespace NUMINAMATH_GPT_total_area_of_sheet_l568_56894

theorem total_area_of_sheet (A B : ℝ) (h1 : A = 4 * B) (h2 : A = B + 2208) : A + B = 3680 :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_sheet_l568_56894


namespace NUMINAMATH_GPT_simplify_fraction_l568_56883

-- Define what it means for a fraction to be in simplest form
def coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Define what it means for a fraction to be reducible
def reducible_fraction (num den : ℕ) : Prop := ∃ d > 1, d ∣ num ∧ d ∣ den

-- Main theorem statement
theorem simplify_fraction 
  (m n : ℕ) (h_coprime : coprime m n) 
  (h_reducible : reducible_fraction (4 * m + 3 * n) (5 * m + 2 * n)) : ∃ d, d = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l568_56883


namespace NUMINAMATH_GPT_monthly_payment_l568_56827

noncomputable def house_price := 280
noncomputable def deposit := 40
noncomputable def mortgage_years := 10
noncomputable def months_per_year := 12

theorem monthly_payment (house_price deposit : ℕ) (mortgage_years months_per_year : ℕ) :
  (house_price - deposit) / mortgage_years / months_per_year = 2 :=
by
  sorry

end NUMINAMATH_GPT_monthly_payment_l568_56827


namespace NUMINAMATH_GPT_min_value_expression_l568_56893

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  ∃ A : ℝ, A = 3 * Real.sqrt 2 ∧ 
  (A = (Real.sqrt (a^6 + b^4 * c^6) / b) + 
       (Real.sqrt (b^6 + c^4 * a^6) / c) + 
       (Real.sqrt (c^6 + a^4 * b^6) / a)) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l568_56893


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l568_56895

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l568_56895


namespace NUMINAMATH_GPT_students_neither_l568_56896

-- Define the conditions
def total_students : ℕ := 60
def students_math : ℕ := 40
def students_physics : ℕ := 35
def students_both : ℕ := 25

-- Define the problem statement
theorem students_neither : total_students - ((students_math - students_both) + (students_physics - students_both) + students_both) = 10 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_l568_56896


namespace NUMINAMATH_GPT_sum_of_intercepts_l568_56899

theorem sum_of_intercepts (x y : ℝ) (h : x / 3 - y / 4 = 1) : (x / 3 = 1 ∧ y / (-4) = 1) → 3 + (-4) = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_intercepts_l568_56899


namespace NUMINAMATH_GPT_trapezoid_diagonal_comparison_l568_56890

variable {A B C D: Type}
variable (α β : Real) -- Representing angles
variable (AB CD BD AC : Real) -- Representing lengths of sides and diagonals
variable (h : Real) -- Height
variable (A' B' : Real) -- Projections

noncomputable def trapezoid (AB CD: Real) := True -- Trapezoid definition placeholder
noncomputable def angle_relation (α β : Real) := α < β -- Angle relationship

theorem trapezoid_diagonal_comparison
  (trapezoid_ABCD: trapezoid AB CD)
  (angle_relation_ABC_DCB : angle_relation α β)
  : BD > AC :=
sorry

end NUMINAMATH_GPT_trapezoid_diagonal_comparison_l568_56890


namespace NUMINAMATH_GPT_total_days_2010_to_2013_l568_56814

theorem total_days_2010_to_2013 :
  let year2010_days := 365
  let year2011_days := 365
  let year2012_days := 366
  let year2013_days := 365
  year2010_days + year2011_days + year2012_days + year2013_days = 1461 := by
  sorry

end NUMINAMATH_GPT_total_days_2010_to_2013_l568_56814


namespace NUMINAMATH_GPT_jill_spent_on_clothing_l568_56870

-- Define the total amount spent excluding taxes, T.
variable (T : ℝ)
-- Define the percentage of T Jill spent on clothing, C.
variable (C : ℝ)

-- Define the conditions based on the problem statement.
def jill_tax_conditions : Prop :=
  let food_percent := 0.20
  let other_items_percent := 0.30
  let clothing_tax := 0.04
  let food_tax := 0
  let other_tax := 0.10
  let total_tax := 0.05
  let food_amount := food_percent * T
  let other_items_amount := other_items_percent * T
  let clothing_amount := C * T
  let clothing_tax_amount := clothing_tax * clothing_amount
  let other_tax_amount := other_tax * other_items_amount
  let total_tax_amount := clothing_tax_amount + food_tax * food_amount + other_tax_amount
  C * T + food_percent * T + other_items_percent * T = T ∧
  total_tax_amount / T = total_tax

-- The goal is to prove that C = 0.50.
theorem jill_spent_on_clothing (h : jill_tax_conditions T C) : C = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_jill_spent_on_clothing_l568_56870


namespace NUMINAMATH_GPT_pink_highlighters_count_l568_56858

-- Definitions for the problem's conditions
def total_highlighters : Nat := 11
def yellow_highlighters : Nat := 2
def blue_highlighters : Nat := 5
def non_pink_highlighters : Nat := yellow_highlighters + blue_highlighters

-- Statement of the problem as a theorem
theorem pink_highlighters_count : total_highlighters - non_pink_highlighters = 4 :=
by
  sorry

end NUMINAMATH_GPT_pink_highlighters_count_l568_56858


namespace NUMINAMATH_GPT_jenna_costume_l568_56819

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end NUMINAMATH_GPT_jenna_costume_l568_56819


namespace NUMINAMATH_GPT_nancy_kept_tortilla_chips_l568_56856

theorem nancy_kept_tortilla_chips (initial_chips : ℕ) (chips_to_brother : ℕ) (chips_to_sister : ℕ) (remaining_chips : ℕ) 
  (h1 : initial_chips = 22) 
  (h2 : chips_to_brother = 7) 
  (h3 : chips_to_sister = 5) 
  (h_total_given : initial_chips - (chips_to_brother + chips_to_sister) = remaining_chips) :
  remaining_chips = 10 :=
sorry

end NUMINAMATH_GPT_nancy_kept_tortilla_chips_l568_56856


namespace NUMINAMATH_GPT_n_squared_plus_n_divisible_by_2_l568_56817

theorem n_squared_plus_n_divisible_by_2 (n : ℤ) : 2 ∣ (n^2 + n) :=
sorry

end NUMINAMATH_GPT_n_squared_plus_n_divisible_by_2_l568_56817


namespace NUMINAMATH_GPT_equivalent_weeks_l568_56835

def hoursPerDay := 24
def daysPerWeek := 7
def hoursPerWeek := daysPerWeek * hoursPerDay
def totalHours := 2016

theorem equivalent_weeks : totalHours / hoursPerWeek = 12 := 
by
  sorry

end NUMINAMATH_GPT_equivalent_weeks_l568_56835


namespace NUMINAMATH_GPT_quoted_value_stock_l568_56863

-- Define the conditions
def face_value : ℕ := 100
def dividend_percentage : ℝ := 0.14
def yield_percentage : ℝ := 0.1

-- Define the computed dividend per share
def dividend_per_share : ℝ := dividend_percentage * face_value

-- State the theorem to prove the quoted value
theorem quoted_value_stock : (dividend_per_share / yield_percentage) * 100 = 140 :=
by
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_quoted_value_stock_l568_56863


namespace NUMINAMATH_GPT_evaluate_expression_l568_56897

theorem evaluate_expression : (2014 - 2013) * (2013 - 2012) = 1 := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l568_56897


namespace NUMINAMATH_GPT_factor_expression_l568_56803

theorem factor_expression (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l568_56803


namespace NUMINAMATH_GPT_distance_between_points_on_parabola_l568_56840

theorem distance_between_points_on_parabola (x1 y1 x2 y2 : ℝ) 
  (h_parabola : ∀ (x : ℝ), 4 * ((x^2)/4) = x^2) 
  (h_focus : F = (0, 1))
  (h_line : y1 = k * x1 + 1 ∧ y2 = k * x2 + 1)
  (h_intersects : x1^2 = 4 * y1 ∧ x2^2 = 4 * y2)
  (h_y_sum : y1 + y2 = 6) :
  |dist (x1, y1) (x2, y2)| = 8 := sorry

end NUMINAMATH_GPT_distance_between_points_on_parabola_l568_56840


namespace NUMINAMATH_GPT_equal_a_b_l568_56869

theorem equal_a_b (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_n : 0 < n) 
  (h_eq : (a + b)^n - (a - b)^n = (a / b) * ((a + b)^n + (a - b)^n)) : a = b :=
sorry

end NUMINAMATH_GPT_equal_a_b_l568_56869


namespace NUMINAMATH_GPT_number_of_monomials_l568_56802

def isMonomial (expr : String) : Bool :=
  match expr with
  | "-(2 / 3) * a^3 * b" => true
  | "(x * y) / 2" => true
  | "-4" => true
  | "0" => true
  | _ => false

def countMonomials (expressions : List String) : Nat :=
  expressions.foldl (fun acc expr => if isMonomial expr then acc + 1 else acc) 0

theorem number_of_monomials : countMonomials ["-(2 / 3) * a^3 * b", "(x * y) / 2", "-4", "-(2 / a)", "0", "x - y"] = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_monomials_l568_56802


namespace NUMINAMATH_GPT_sum_of_square_areas_l568_56846

variable (WX XZ : ℝ)

theorem sum_of_square_areas (hW : WX = 15) (hX : XZ = 20) : WX^2 + XZ^2 = 625 := by
  sorry

end NUMINAMATH_GPT_sum_of_square_areas_l568_56846


namespace NUMINAMATH_GPT_max_marks_l568_56854

theorem max_marks {M : ℝ} (h : 0.90 * M = 550) : M = 612 :=
sorry

end NUMINAMATH_GPT_max_marks_l568_56854


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l568_56845

/- 
  Define the arithmetic sequence in terms of its properties 
  and prove that the 10th term is 18.
-/

theorem arithmetic_sequence_tenth_term (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 8) : a 10 = 18 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l568_56845


namespace NUMINAMATH_GPT_value_of_c_l568_56805

theorem value_of_c (c : ℝ) :
  (∀ x y : ℝ, (x, y) = ((2 + 8) / 2, (6 + 10) / 2) → x + y = c) → c = 13 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_value_of_c_l568_56805


namespace NUMINAMATH_GPT_graph_passes_through_point_l568_56873

theorem graph_passes_through_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
    ∃ y : ℝ, y = a^0 + 1 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l568_56873


namespace NUMINAMATH_GPT_number_of_female_students_l568_56815

theorem number_of_female_students
  (F : ℕ) -- number of female students
  (T : ℕ) -- total number of students
  (h1 : T = F + 8) -- total students = female students + 8 male students
  (h2 : 90 * T = 85 * 8 + 92 * F) -- equation from the sum of scores
  : F = 20 :=
sorry

end NUMINAMATH_GPT_number_of_female_students_l568_56815


namespace NUMINAMATH_GPT_find_value_of_expression_l568_56862

theorem find_value_of_expression (a b c : ℝ) (h : (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0) : a + 2*b + 3*c = 18 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l568_56862


namespace NUMINAMATH_GPT_min_cuts_for_30_sided_polygons_l568_56871

theorem min_cuts_for_30_sided_polygons (n : ℕ) (h : n = 73) : 
  ∃ k : ℕ, (∀ m : ℕ, m < k → (m + 1) ≤ 2 * m - 1972) ∧ (k = 1970) :=
sorry

end NUMINAMATH_GPT_min_cuts_for_30_sided_polygons_l568_56871


namespace NUMINAMATH_GPT_escalator_length_l568_56860

theorem escalator_length
  (escalator_speed : ℕ)
  (person_speed : ℕ)
  (time_taken : ℕ)
  (combined_speed : ℕ)
  (condition1 : escalator_speed = 12)
  (condition2 : person_speed = 2)
  (condition3 : time_taken = 14)
  (condition4 : combined_speed = escalator_speed + person_speed)
  (condition5 : combined_speed * time_taken = 196) :
  combined_speed * time_taken = 196 := 
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_escalator_length_l568_56860


namespace NUMINAMATH_GPT_rectangle_same_color_exists_l568_56881

theorem rectangle_same_color_exists (grid : Fin 3 → Fin 7 → Bool) : 
  ∃ (r1 r2 c1 c2 : Fin 3), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid r1 c1 = grid r1 c2 ∧ grid r1 c1 = grid r2 c1 ∧ grid r1 c1 = grid r2 c2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_same_color_exists_l568_56881


namespace NUMINAMATH_GPT_digit_864_div_5_appending_zero_possibilities_l568_56813

theorem digit_864_div_5_appending_zero_possibilities :
  ∀ X : ℕ, (X * 1000 + 864) % 5 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_digit_864_div_5_appending_zero_possibilities_l568_56813


namespace NUMINAMATH_GPT_sequence_bound_l568_56800

noncomputable def sequenceProperties (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ i, 0 ≤ a i ∧ a i ≤ c) ∧ (∀ i j, i ≠ j → abs (a i - a j) ≥ 1 / (i + j))

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) (h : sequenceProperties a c) : 
  c ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_bound_l568_56800


namespace NUMINAMATH_GPT_sqrt_9_eq_3_and_neg3_l568_56882

theorem sqrt_9_eq_3_and_neg3 : { x : ℝ | x^2 = 9 } = {3, -3} :=
by
  sorry

end NUMINAMATH_GPT_sqrt_9_eq_3_and_neg3_l568_56882


namespace NUMINAMATH_GPT_election_1002nd_k_election_1001st_k_l568_56824

variable (k : ℕ)

noncomputable def election_in_1002nd_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 2001 → -- The conditions include the number of candidates 'n', and specifying that 'k' being the maximum initially means k ≤ 2001.
  true

noncomputable def election_in_1001st_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 1 → -- Similarly, these conditions specify the initial maximum placement as 1 when elected in 1001st round.
  true

-- Definitions specifying the problem to identify max k for given rounds
theorem election_1002nd_k : election_in_1002nd_round_max_k k := sorry

theorem election_1001st_k : election_in_1001st_round_max_k k := sorry

end NUMINAMATH_GPT_election_1002nd_k_election_1001st_k_l568_56824


namespace NUMINAMATH_GPT_white_rabbit_hop_distance_per_minute_l568_56885

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end NUMINAMATH_GPT_white_rabbit_hop_distance_per_minute_l568_56885


namespace NUMINAMATH_GPT_expression_value_l568_56861

noncomputable def evaluate_expression : ℝ :=
  Real.logb 2 (3 * 11 + Real.exp (4 - 8)) + 3 * Real.sin (Real.pi^2 - Real.sqrt ((6 * 4) / 3 - 4))

theorem expression_value : evaluate_expression = 3.832 := by
  sorry

end NUMINAMATH_GPT_expression_value_l568_56861


namespace NUMINAMATH_GPT_chord_length_perpendicular_bisector_l568_56841

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) :
  ∃ (CD : ℝ), CD = 10 * Real.sqrt 3 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_chord_length_perpendicular_bisector_l568_56841


namespace NUMINAMATH_GPT_proof_problem_l568_56886

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l568_56886


namespace NUMINAMATH_GPT_polygon_sides_l568_56864

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end NUMINAMATH_GPT_polygon_sides_l568_56864


namespace NUMINAMATH_GPT_player_A_elimination_after_third_round_at_least_one_player_passes_all_l568_56884

-- Define probabilities for Player A's success in each round
def P_A1 : ℚ := 4 / 5
def P_A2 : ℚ := 3 / 4
def P_A3 : ℚ := 2 / 3

-- Define probabilities for Player B's success in each round
def P_B1 : ℚ := 2 / 3
def P_B2 : ℚ := 2 / 3
def P_B3 : ℚ := 1 / 2

-- Define theorems
theorem player_A_elimination_after_third_round :
  P_A1 * P_A2 * (1 - P_A3) = 1 / 5 := by
  sorry

theorem at_least_one_player_passes_all :
  1 - ((1 - (P_A1 * P_A2 * P_A3)) * (1 - (P_B1 * P_B2 * P_B3))) = 8 / 15 := by
  sorry


end NUMINAMATH_GPT_player_A_elimination_after_third_round_at_least_one_player_passes_all_l568_56884


namespace NUMINAMATH_GPT_at_least_one_variety_has_27_apples_l568_56852

theorem at_least_one_variety_has_27_apples (total_apples : ℕ) (varieties : ℕ) 
  (h_total : total_apples = 105) (h_varieties : varieties = 4) : 
  ∃ v : ℕ, v ≥ 27 := 
sorry

end NUMINAMATH_GPT_at_least_one_variety_has_27_apples_l568_56852


namespace NUMINAMATH_GPT_perpendicular_angles_l568_56838

theorem perpendicular_angles (α β : ℝ) (k : ℤ) : 
  (∃ k : ℤ, β - α = k * 360 + 90 ∨ β - α = k * 360 - 90) →
  β = k * 360 + α + 90 ∨ β = k * 360 + α - 90 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_angles_l568_56838


namespace NUMINAMATH_GPT_seating_arrangements_l568_56868

-- Define the participants
inductive Person : Type
| xiaoMing
| parent1
| parent2
| grandparent1
| grandparent2

open Person

-- Define the function to count seating arrangements
noncomputable def count_seating_arrangements : Nat :=
  let arrangements := [
    -- (Only one parent next to Xiao Ming, parents not next to each other)
    12,
    -- (Only one parent next to Xiao Ming, parents next to each other)
    24,
    -- (Both parents next to Xiao Ming)
    12
  ]
  arrangements.foldr (· + ·) 0

theorem seating_arrangements : count_seating_arrangements = 48 := by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l568_56868


namespace NUMINAMATH_GPT_range_of_f_l568_56836

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x - 1

-- Define the domain
def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

-- Define the range
def range : Set ℕ := {2, 5, 8, 11}

-- Lean 4 theorem statement
theorem range_of_f : 
  {y | ∃ x ∈ domain, y = f x} = range :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l568_56836


namespace NUMINAMATH_GPT_greatest_possible_y_l568_56887

theorem greatest_possible_y (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : y ≤ 18 :=
sorry

end NUMINAMATH_GPT_greatest_possible_y_l568_56887
