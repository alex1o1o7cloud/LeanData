import Mathlib

namespace second_year_growth_rate_l190_190245

variable (initial_investment : ℝ) (first_year_growth : ℝ) (additional_investment : ℝ) (final_value : ℝ) (second_year_growth : ℝ)

def calculate_portfolio_value_after_first_year (initial_investment first_year_growth : ℝ) : ℝ :=
  initial_investment * (1 + first_year_growth)

def calculate_new_value_after_addition (value_after_first_year additional_investment : ℝ) : ℝ :=
  value_after_first_year + additional_investment

def calculate_final_value_after_second_year (new_value second_year_growth : ℝ) : ℝ :=
  new_value * (1 + second_year_growth)

theorem second_year_growth_rate 
  (h1 : initial_investment = 80) 
  (h2 : first_year_growth = 0.15) 
  (h3 : additional_investment = 28) 
  (h4 : final_value = 132) : 
  calculate_final_value_after_second_year
    (calculate_new_value_after_addition
      (calculate_portfolio_value_after_first_year initial_investment first_year_growth)
      additional_investment)
    0.1 = final_value := 
  by
  sorry

end second_year_growth_rate_l190_190245


namespace rationalize_denominator_result_l190_190931

noncomputable def rationalized_denominator : ℚ := 5 / (3 * real.cbrt 7) * real.cbrt (49 / 49)

theorem rationalize_denominator_result :
  ∃ (A B C : ℤ), C > 0 ∧ ¬∃ (p : ℤ), nat.prime p ∧ p^3 ∣ B ∧ rationalized_denominator = (A * real.cbrt B) / C ∧
  A + B + C = 75 :=
begin
  use [5, 49, 21],
  split,
  { exact zero_lt_one.trans_le (le_refl 21), },
  split,
  { rintros ⟨p, hp⟩,
    exfalso,
    cases hp with hp_prime hp_div,
    have hp_not_cube : ¬ p^3 ∣ 49 := λ h, sorry, -- just a placeholder to skip this proof
    exact hp_not_cube hp_div, },
  split,
  { field_simp [rationalized_denominator],
    norm_num [real.cbrt_pow] },
  { ring }
end

end rationalize_denominator_result_l190_190931


namespace amazing_squares_exist_l190_190593

structure Quadrilateral :=
(A B C D : Point)

def diagonals_not_perpendicular (quad : Quadrilateral) : Prop := sorry -- The precise definition will abstractly represent the non-perpendicularity of diagonals.

def amazing_square (quad : Quadrilateral) (square : Square) : Prop :=
  -- Definition stating that the sides of the square (extended if necessary) pass through distinct vertices of the quadrilateral
  sorry

theorem amazing_squares_exist (quad : Quadrilateral) (h : diagonals_not_perpendicular quad) :
  ∃ squares : Finset Square, squares.card ≥ 6 ∧ ∀ square ∈ squares, amazing_square quad square :=
by sorry

end amazing_squares_exist_l190_190593


namespace factorization_result_l190_190954

theorem factorization_result :
  ∃ (c d : ℕ), (c > d) ∧ ((x^2 - 20 * x + 91) = (x - c) * (x - d)) ∧ (2 * d - c = 1) :=
by
  -- Using the conditions and proving the given equation
  sorry

end factorization_result_l190_190954


namespace problem1_problem2_l190_190062

theorem problem1 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) :
  (2 < x1 + x2 ∧ x1 + x2 < 6) ∧ |x1 - x2| < 2 :=
by
  sorry

theorem problem2 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2 - x + 1) :
  |x1 - x2| < |f x1 - f x2| ∧ |f x1 - f x2| < 5 * |x1 - x2| :=
by
  sorry

end problem1_problem2_l190_190062


namespace tangent_angle_inclination_range_l190_190879

noncomputable def tangent_angle_range : set ℝ := 
  (set.Icc (0 : ℝ) (π/4)) ∪ (set.Ico (3*π/4) π)

theorem tangent_angle_inclination_range (x : ℝ) :
  x ∈ set.Icc (0 : ℝ) (2 * π) →
  ∃ θ ∈ tangent_angle_range, θ = real.arctan (real.cos x) := 
by
  sorry

end tangent_angle_inclination_range_l190_190879


namespace quotient_when_divided_by_5_l190_190926

theorem quotient_when_divided_by_5 (N : ℤ) (k : ℤ) (Q : ℤ) 
  (h1 : N = 5 * Q) 
  (h2 : N % 4 = 2) : 
  Q = 2 := 
sorry

end quotient_when_divided_by_5_l190_190926


namespace find_certain_number_multiplied_by_24_l190_190526

-- Define the conditions
theorem find_certain_number_multiplied_by_24 :
  (∃ x : ℤ, 37 - x = 24) →
  ∀ x : ℤ, (37 - x = 24) → (x * 24 = 312) :=
by
  intros h x hx
  -- Here we will have the proof using the assumption and the theorem.
  sorry

end find_certain_number_multiplied_by_24_l190_190526


namespace polynomial_expansion_l190_190860

theorem polynomial_expansion (x : ℝ) :
  (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 :=
by sorry

end polynomial_expansion_l190_190860


namespace abs_h_minus_2k_l190_190722

-- Defining conditions from the problem:
def polynomial (x : ℝ) (h k : ℝ) := 3 * x^3 - h * x + k

-- Given conditions using the Remainder Theorem:
def condition1 (h k : ℝ) := polynomial 3 h k = 0
def condition2 (h k : ℝ) := polynomial (-1) h k = 0

-- Main statement we need to prove
theorem abs_h_minus_2k (h k : ℝ) (h_cond1 : condition1 h k) (h_cond2 : condition2 h k) :
  |h - 2 * k| = 57 :=
by sorry

end abs_h_minus_2k_l190_190722


namespace minimum_discount_l190_190697

theorem minimum_discount (C M : ℝ) (profit_margin : ℝ) (x : ℝ) 
  (hC : C = 800) (hM : M = 1200) (hprofit_margin : profit_margin = 0.2) :
  (M * x - C ≥ C * profit_margin) → (x ≥ 0.8) :=
by
  -- Here, we need to solve the inequality given the conditions
  sorry

end minimum_discount_l190_190697


namespace multiply_decimals_l190_190575

noncomputable def real_num_0_7 : ℝ := 7 * 10⁻¹
noncomputable def real_num_0_3 : ℝ := 3 * 10⁻¹
noncomputable def real_num_0_21 : ℝ := 0.21

theorem multiply_decimals :
  real_num_0_7 * real_num_0_3 = real_num_0_21 :=
sorry

end multiply_decimals_l190_190575


namespace evaluate_expression_l190_190462

noncomputable def absoluteValue (x : ℝ) : ℝ := |x|

noncomputable def ceilingFunction (x : ℝ) : ℤ := ⌈x⌉

theorem evaluate_expression : ceilingFunction (absoluteValue (-52.7)) = 53 :=
by
  sorry

end evaluate_expression_l190_190462


namespace sphere_radius_is_five_l190_190289

theorem sphere_radius_is_five
    (π : ℝ)
    (r r_cylinder h : ℝ)
    (A_sphere A_cylinder : ℝ)
    (h1 : A_sphere = 4 * π * r ^ 2)
    (h2 : A_cylinder = 2 * π * r_cylinder * h)
    (h3 : h = 10)
    (h4 : r_cylinder = 5)
    (h5 : A_sphere = A_cylinder) :
    r = 5 :=
by
  sorry

end sphere_radius_is_five_l190_190289


namespace max_min_value_f_l190_190477

theorem max_min_value_f (x m : ℝ) : ∃ m : ℝ, (∀ x : ℝ, x^2 - 2*m*x + 8*m + 4 ≥ -m^2 + 8*m + 4) ∧ (∀ n : ℝ, -n^2 + 8*n + 4 ≤ 20) :=
  sorry

end max_min_value_f_l190_190477


namespace parameterization_of_line_l190_190119

theorem parameterization_of_line : 
  ∀ t : ℝ, ∃ f : ℝ → ℝ, (f t, 20 * t - 14) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), y = 2 * x - 40 ∧ p = (x, y) } ∧ f t = 10 * t + 13 :=
by
  sorry

end parameterization_of_line_l190_190119


namespace value_of_y_l190_190760

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 :=
by
  sorry

end value_of_y_l190_190760


namespace find_fraction_identity_l190_190515

variable (x y z : ℝ)

theorem find_fraction_identity
 (h1 : 16 * y^2 = 15 * x * z)
 (h2 : y = 2 * x * z / (x + z)) :
 x / z + z / x = 34 / 15 := by
-- proof skipped
sorry

end find_fraction_identity_l190_190515


namespace sum_of_squares_of_roots_eq_zero_l190_190718

theorem sum_of_squares_of_roots_eq_zero :
  let f : Polynomial ℝ := Polynomial.C 50 + Polynomial.monomial 3 (-2) + Polynomial.monomial 7 5 + Polynomial.monomial 10 1
  ∀ (r : ℝ), r ∈ Multiset.toFinset f.roots → r ^ 2 = 0 :=
by
  sorry

end sum_of_squares_of_roots_eq_zero_l190_190718


namespace tom_remaining_balloons_l190_190425

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end tom_remaining_balloons_l190_190425


namespace average_cost_is_2_l190_190648

noncomputable def total_amount_spent (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℕ :=
  apples_quantity * apples_cost + bananas_quantity * bananas_cost + oranges_quantity * oranges_cost

noncomputable def total_number_of_fruits (apples_quantity bananas_quantity oranges_quantity : ℕ) : ℕ :=
  apples_quantity + bananas_quantity + oranges_quantity

noncomputable def average_cost (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℚ :=
  (total_amount_spent apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℚ) /
  (total_number_of_fruits apples_quantity bananas_quantity oranges_quantity : ℚ)

theorem average_cost_is_2 :
  average_cost 12 4 4 2 1 3 = 2 := 
by
  sorry

end average_cost_is_2_l190_190648


namespace product_sum_correct_l190_190849

def product_sum_eq : Prop :=
  let a := 4 * 10^6
  let b := 8 * 10^6
  (a * b + 2 * 10^13) = 5.2 * 10^13

theorem product_sum_correct : product_sum_eq :=
by
  sorry

end product_sum_correct_l190_190849


namespace wendy_first_day_miles_l190_190925

-- Define the variables for the problem
def total_miles : ℕ := 493
def miles_day2 : ℕ := 223
def miles_day3 : ℕ := 145

-- Define the proof problem
theorem wendy_first_day_miles :
  total_miles = miles_day2 + miles_day3 + 125 :=
sorry

end wendy_first_day_miles_l190_190925


namespace necessary_condition_not_sufficient_condition_l190_190792

def P (x : ℝ) := x > 0
def Q (x : ℝ) := x > -2

theorem necessary_condition : ∀ x: ℝ, P x → Q x := 
by sorry

theorem not_sufficient_condition : ∃ x: ℝ, Q x ∧ ¬ P x := 
by sorry

end necessary_condition_not_sufficient_condition_l190_190792


namespace f_inequality_l190_190753

-- Define the function f.
def f (x : ℝ) : ℝ := x^2 - x + 13

-- The main theorem to prove the given inequality.
theorem f_inequality (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2*(|m| + 1) :=
by
  sorry

end f_inequality_l190_190753


namespace cylinder_volume_l190_190500

theorem cylinder_volume (h : ℝ) (H1 : π * h ^ 2 = 4 * π) : (π * (h / 2) ^ 2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l190_190500


namespace find_integer_solutions_l190_190735

theorem find_integer_solutions (k : ℕ) (hk : k > 1) : 
  ∃ x y : ℤ, y^k = x^2 + x ↔ (k = 2 ∧ (x = 0 ∨ x = -1)) ∨ (k > 2 ∧ y^k ≠ x^2 + x) :=
by
  sorry

end find_integer_solutions_l190_190735


namespace geometric_seq_ratio_l190_190387

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l190_190387


namespace distance_between_towns_l190_190004

theorem distance_between_towns (D S : ℝ) (h1 : D = S * 3) (h2 : 200 = S * 5) : D = 120 :=
by
  sorry

end distance_between_towns_l190_190004


namespace jane_earnings_two_weeks_l190_190080

def num_chickens : ℕ := 10
def num_eggs_per_chicken_per_week : ℕ := 6
def dollars_per_dozen : ℕ := 2
def dozens_in_12_eggs : ℕ := 12

theorem jane_earnings_two_weeks :
  (num_chickens * num_eggs_per_chicken_per_week * 2 / dozens_in_12_eggs * dollars_per_dozen) = 20 := by
  sorry

end jane_earnings_two_weeks_l190_190080


namespace linear_equation_in_two_vars_example_l190_190035

def is_linear_equation_in_two_vars (eq : String) : Prop :=
  eq = "x + 4y = 6"

theorem linear_equation_in_two_vars_example :
  is_linear_equation_in_two_vars "x + 4y = 6" :=
by
  sorry

end linear_equation_in_two_vars_example_l190_190035


namespace hotel_loss_l190_190303
  
  -- Conditions
  def operations_expenses : ℝ := 100
  def total_payments : ℝ := (3 / 4) * operations_expenses
  
  -- Theorem to prove
  theorem hotel_loss : operations_expenses - total_payments = 25 :=
  by
    sorry
  
end hotel_loss_l190_190303


namespace part1_part2_l190_190037

-- Problem Part 1
theorem part1 : (-((-8)^(1/3)) - |(3^(1/2) - 2)| + ((-3)^2)^(1/2) + -3^(1/2) = 3) :=
by {
  sorry
}

-- Problem Part 2
theorem part2 (x : ℤ) : (2 * x + 5 ≤ 3 * (x + 2) ∧ 2 * x - (1 + 3 * x) / 2 < 1) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by {
  sorry
}

end part1_part2_l190_190037


namespace subtraction_division_l190_190146

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end subtraction_division_l190_190146


namespace solve_for_x_l190_190105

theorem solve_for_x (x : ℝ) (h : (1/3) + (1/x) = 2/3) : x = 3 :=
by
  sorry

end solve_for_x_l190_190105


namespace jennie_total_rental_cost_l190_190666

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end jennie_total_rental_cost_l190_190666


namespace leo_assignment_third_part_time_l190_190379

-- Define all the conditions as variables
def first_part_time : ℕ := 25
def first_break : ℕ := 10
def second_part_time : ℕ := 2 * first_part_time
def second_break : ℕ := 15
def total_time : ℕ := 150

-- The calculated total time of the first two parts and breaks
def time_spent_on_first_two_parts_and_breaks : ℕ :=
  first_part_time + first_break + second_part_time + second_break

-- The remaining time for the third part of the assignment
def third_part_time : ℕ :=
  total_time - time_spent_on_first_two_parts_and_breaks

-- The theorem to prove that the time Leo took to finish the third part is 50 minutes
theorem leo_assignment_third_part_time : third_part_time = 50 := by
  sorry

end leo_assignment_third_part_time_l190_190379


namespace quartic_root_sum_l190_190919

theorem quartic_root_sum (a n l : ℝ) (h : ∃ (r1 r2 r3 r4 : ℝ), 
  r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧ 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ 
  r1 + r2 + r3 + r4 = 10 ∧
  r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4 = a ∧
  r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 = n ∧
  r1 * r2 * r3 * r4 = l) : 
  a + n + l = 109 :=
sorry

end quartic_root_sum_l190_190919


namespace simplify_and_evaluate_l190_190660

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4 * x - 4 = 0) :
  3 * (x - 2) ^ 2 - 6 * (x + 1) * (x - 1) = 6 :=
by
  sorry

end simplify_and_evaluate_l190_190660


namespace seating_arrangement_l190_190450

def num_ways_to_seat (A B C D E F : Type) (chairs : List (Option Type)) : Nat := sorry

theorem seating_arrangement {A B C D E F : Type} :
  ∀ (chairs : List (Option Type)),
    (A ≠ B ∧ A ≠ C ∧ F ≠ B) → num_ways_to_seat A B C D E F chairs = 28 :=
by
  sorry

end seating_arrangement_l190_190450


namespace sample_size_120_l190_190553

theorem sample_size_120
  (x y : ℕ)
  (h_ratio : x / 2 = y / 3 ∧ y / 3 = 60 / 5)
  (h_max : max x (max y 60) = 60) :
  x + y + 60 = 120 := by
  sorry

end sample_size_120_l190_190553


namespace number_of_women_per_table_l190_190706

theorem number_of_women_per_table
  (tables : ℕ) (men_per_table : ℕ) 
  (total_customers : ℕ) (total_tables : tables = 9) 
  (men_at_each_table : men_per_table = 3) 
  (customers : total_customers = 90) 
  (total_men : 3 * 9 = 27) 
  (total_women : 90 - 27 = 63) :
  (63 / 9 = 7) :=
by
  sorry

end number_of_women_per_table_l190_190706


namespace fraction_proof_l190_190285

-- Define the fractions as constants
def a := 1 / 3
def b := 1 / 4
def c := 1 / 2
def d := 1 / 3

-- Prove the main statement
theorem fraction_proof : (a - b) / (c - d) = 1 / 2 := by
  sorry

end fraction_proof_l190_190285


namespace campaign_meaning_l190_190827

-- Define a function that gives the meaning of "campaign" as a noun
def meaning_of_campaign_noun : String :=
  "campaign, activity"

-- The theorem asserts that the meaning of "campaign" as a noun is "campaign, activity"
theorem campaign_meaning : meaning_of_campaign_noun = "campaign, activity" :=
by
  -- We add sorry here because we are not required to provide the proof
  sorry

end campaign_meaning_l190_190827


namespace toothpick_removal_l190_190523

noncomputable def removalStrategy : ℕ :=
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4

  -- minimum toothpicks to remove to achieve the goal
  15

theorem toothpick_removal :
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4
  removalStrategy = 15 := by
  sorry

end toothpick_removal_l190_190523


namespace sum_even_1_to_200_l190_190687

open Nat

/-- The sum of all even numbers from 1 to 200 is 10100. --/
theorem sum_even_1_to_200 :
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  sum = 10100 :=
by
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  show sum = 10100
  sorry

end sum_even_1_to_200_l190_190687


namespace greatest_number_divides_with_remainders_l190_190547

theorem greatest_number_divides_with_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end greatest_number_divides_with_remainders_l190_190547


namespace factor_y6_plus_64_l190_190040

theorem factor_y6_plus_64 : (y^2 + 4) ∣ (y^6 + 64) :=
sorry

end factor_y6_plus_64_l190_190040


namespace last_three_digits_of_7_to_50_l190_190591

theorem last_three_digits_of_7_to_50 : (7^50) % 1000 = 991 := 
by 
  sorry

end last_three_digits_of_7_to_50_l190_190591


namespace investor_more_money_in_A_l190_190452

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end investor_more_money_in_A_l190_190452


namespace necessary_condition_l190_190793

theorem necessary_condition (x : ℝ) (h : x > 0) : x > -2 :=
by {
  exact lt_trans (by norm_num) h,
}

end necessary_condition_l190_190793


namespace roots_polynomial_sum_products_l190_190641

theorem roots_polynomial_sum_products (p q r : ℂ)
  (h : 6 * p^3 - 5 * p^2 + 13 * p - 10 = 0)
  (h' : 6 * q^3 - 5 * q^2 + 13 * q - 10 = 0)
  (h'' : 6 * r^3 - 5 * r^2 + 13 * r - 10 = 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) :
  p * q + q * r + r * p = 13 / 6 := 
sorry

end roots_polynomial_sum_products_l190_190641


namespace allocation_methods_count_l190_190145

theorem allocation_methods_count :
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  ∃ (allocation_methods : ℕ), allocation_methods = 12 := 
by
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  use doctors * Nat.choose nurses 2
  sorry

end allocation_methods_count_l190_190145


namespace fred_balloons_l190_190741

theorem fred_balloons (T S D F : ℕ) (hT : T = 72) (hS : S = 46) (hD : D = 16) (hTotal : T = F + S + D) : F = 10 := 
by
  sorry

end fred_balloons_l190_190741


namespace noah_billed_amount_l190_190923

theorem noah_billed_amount
  (minutes_per_call : ℕ)
  (cost_per_minute : ℝ)
  (weeks_per_year : ℕ)
  (total_cost : ℝ)
  (h_minutes_per_call : minutes_per_call = 30)
  (h_cost_per_minute : cost_per_minute = 0.05)
  (h_weeks_per_year : weeks_per_year = 52)
  (h_total_cost : total_cost = 78) :
  (minutes_per_call * cost_per_minute * weeks_per_year = total_cost) :=
by
  sorry

end noah_billed_amount_l190_190923


namespace min_value_reciprocal_sum_l190_190512

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end min_value_reciprocal_sum_l190_190512


namespace correct_option_l190_190542

-- Definitions
def option_A (a : ℕ) : Prop := a^2 * a^3 = a^5
def option_B (a : ℕ) : Prop := a^6 / a^2 = a^3
def option_C (a b : ℕ) : Prop := (a * b^3) ^ 2 = a^2 * b^9
def option_D (a : ℕ) : Prop := 5 * a - 2 * a = 3

-- Theorem statement
theorem correct_option :
  (∃ (a : ℕ), option_A a) ∧
  (∀ (a : ℕ), ¬option_B a) ∧
  (∀ (a b : ℕ), ¬option_C a b) ∧
  (∀ (a : ℕ), ¬option_D a) :=
by
  sorry

end correct_option_l190_190542


namespace f_diff_eq_l190_190762

def f (n : ℕ) : ℚ := 1 / 4 * (n * (n + 1) * (n + 3))

theorem f_diff_eq (r : ℕ) : 
  f (r + 1) - f r = 1 / 4 * (3 * r^2 + 11 * r + 8) :=
by {
  sorry
}

end f_diff_eq_l190_190762


namespace darkCubeValidPositions_l190_190241

-- Conditions:
-- 1. The structure is made up of twelve identical cubes.
-- 2. The dark cube must be relocated to a position where the surface area remains unchanged.
-- 3. The cubes must touch each other with their entire faces.
-- 4. The positions of the light cubes cannot be changed.

-- Let's define the structure and the conditions in Lean.

structure Cube :=
  (id : ℕ) -- unique identifier for each cube

structure Position :=
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

structure Configuration :=
  (cubes : List Cube)
  (positions : Cube → Position)

def initialCondition (config : Configuration) : Prop :=
  config.cubes.length = 12

def surfaceAreaUnchanged (config : Configuration) (darkCube : Cube) (newPos : Position) : Prop :=
  sorry -- This predicate should capture the logic that the surface area remains unchanged

def validPositions (config : Configuration) (darkCube : Cube) : List Position :=
  sorry -- This function should return the list of valid positions for the dark cube

-- Main theorem: The number of valid positions for the dark cube to maintain the surface area.
theorem darkCubeValidPositions (config : Configuration) (darkCube : Cube) :
    initialCondition config →
    (validPositions config darkCube).length = 3 :=
  by
  sorry

end darkCubeValidPositions_l190_190241


namespace circle_through_BD_of_parallelogram_ABCD_l190_190149

theorem circle_through_BD_of_parallelogram_ABCD :
  ∀ (A B C D M N P K : Point) (circle : Circle),
    Parallelogram A B C D →
    PointsOnCircle circle [B, D] →
    IntersectCircleAndSides circle A B C D M N P K →
    Parallelogram.AB_parallel_MK circle A B C D M K →
    Parallelogram.CD_parallel_NP circle A B C D N P →
    Parallel MK NP :=
by
  sorry

end circle_through_BD_of_parallelogram_ABCD_l190_190149


namespace hotel_loss_l190_190305

theorem hotel_loss :
  (ops_expenses : ℝ) (payment_frac : ℝ) (total_received : ℝ) (loss : ℝ)
  (h_ops_expenses : ops_expenses = 100)
  (h_payment_frac : payment_frac = 3 / 4)
  (h_total_received : total_received = payment_frac * ops_expenses)
  (h_loss : loss = ops_expenses - total_received) :
  loss = 25 :=
by
  sorry

end hotel_loss_l190_190305


namespace convex_symmetric_polygon_area_ineq_l190_190088

theorem convex_symmetric_polygon_area_ineq 
  (P : Set Point)
  (O : Point)
  (is_convex : ConvexPolygon P)
  (is_symmetric : SymmetricToPoint P O) :
  ∃ (R : Parallelogram), (P ⊆ R) ∧ (area R / area P ≤ Real.sqrt 2) :=
by
  sorry

end convex_symmetric_polygon_area_ineq_l190_190088


namespace cost_of_600_pages_l190_190634

def cost_per_5_pages := 10 -- 10 cents for 5 pages
def pages_to_copy := 600
def expected_cost := 12 * 100 -- 12 dollars in cents

theorem cost_of_600_pages : pages_to_copy * (cost_per_5_pages / 5) = expected_cost := by
  sorry

end cost_of_600_pages_l190_190634


namespace available_seats_l190_190015

/-- Two-fifths of the seats in an auditorium that holds 500 people are currently taken. --/
def seats_taken : ℕ := (2 * 500) / 5

/-- One-tenth of the seats in an auditorium that holds 500 people are broken. --/
def seats_broken : ℕ := 500 / 10

/-- Total seats in the auditorium --/
def total_seats := 500

/-- There are 500 total seats in an auditorium. Two-fifths of the seats are taken and 
one-tenth are broken. Prove that the number of seats still available is 250. --/
theorem available_seats : (total_seats - seats_taken - seats_broken) = 250 :=
by 
  sorry

end available_seats_l190_190015


namespace solve_cubic_eq_l190_190524

theorem solve_cubic_eq (x : ℝ) (h1 : (x + 1)^3 = x^3) (h2 : 0 ≤ x) (h3 : x < 1) : x = 0 :=
by
  sorry

end solve_cubic_eq_l190_190524


namespace Adam_marbles_l190_190842

variable (Adam Greg : Nat)

theorem Adam_marbles (h1 : Greg = 43) (h2 : Greg = Adam + 14) : Adam = 29 := 
by
  sorry

end Adam_marbles_l190_190842


namespace find_expression_value_l190_190864

theorem find_expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end find_expression_value_l190_190864


namespace unique_bijective_function_l190_190254

noncomputable def find_bijective_function {n : ℕ}
  (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ)
  (f : Fin n → ℝ) : Prop :=
∀ i : Fin n, f i = x i

theorem unique_bijective_function (n : ℕ) (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ) (f : Fin n → ℝ)
  (hf_bij : Function.Bijective f)
  (h_abs_diff : ∀ i, |f i - x i| = 0) : find_bijective_function hn hodd x f :=
by
  sorry

end unique_bijective_function_l190_190254


namespace multiply_equality_l190_190140

variable (a b c d e : ℝ)

theorem multiply_equality
  (h1 : a = 2994)
  (h2 : b = 14.5)
  (h3 : c = 173)
  (h4 : d = 29.94)
  (h5 : e = 1.45)
  (h6 : a * b = c) : d * e = 1.73 :=
sorry

end multiply_equality_l190_190140


namespace smallest_positive_period_sin_cos_sin_l190_190961

noncomputable def smallest_positive_period := 2 * Real.pi

theorem smallest_positive_period_sin_cos_sin :
  ∃ T > 0, (∀ x, (Real.sin x - 2 * Real.cos (2 * x) + 4 * Real.sin (4 * x)) = (Real.sin (x + T) - 2 * Real.cos (2 * (x + T)) + 4 * Real.sin (4 * (x + T)))) ∧ T = smallest_positive_period := by
sorry

end smallest_positive_period_sin_cos_sin_l190_190961


namespace major_airlines_wifi_l190_190403

-- Definitions based on conditions
def percentage (x : ℝ) := 0 ≤ x ∧ x ≤ 100

variables (W S B : ℝ)

-- Assume the conditions
axiom H1 : S = 70
axiom H2 : B = 45
axiom H3 : B ≤ S

-- The final proof problem that W = 45
theorem major_airlines_wifi : W = B :=
by
  sorry

end major_airlines_wifi_l190_190403


namespace tom_remaining_balloons_l190_190426

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end tom_remaining_balloons_l190_190426


namespace rationalize_denominator_l190_190946

theorem rationalize_denominator (A B C : ℤ) (hB : ¬ ∃ p : ℤ, p ≥ 2 ∧ p ^ 3 ∣ B) (hC : C > 0) :
  (A = 5) ∧ (B = 49) ∧ (C = 21) → A + B + C = 75 :=
by
  intro h
  rcases h with ⟨hA, hB, hC⟩
  rw [hA, hB, hC]
  simp
  sorry

end rationalize_denominator_l190_190946


namespace automotive_test_l190_190036

theorem automotive_test (D T1 T2 T3 T_total : ℕ) (H1 : 3 * D = 180) 
  (H2 : T1 = D / 4) (H3 : T2 = D / 5) (H4 : T3 = D / 6)
  (H5 : T_total = T1 + T2 + T3) : T_total = 37 :=
  sorry

end automotive_test_l190_190036


namespace algebra_expression_value_l190_190356

theorem algebra_expression_value (a : ℝ) (h : 3 * a ^ 2 + 2 * a - 1 = 0) : 3 * a ^ 2 + 2 * a - 2019 = -2018 := 
by 
  -- Proof goes here
  sorry

end algebra_expression_value_l190_190356


namespace problem_correct_answer_l190_190689

theorem problem_correct_answer :
  (∀ (P L : Type) (passes_through_point : P → L → Prop) (parallel_to : L → L → Prop),
    (∀ (l₁ l₂ : L) (p : P), passes_through_point p l₁ ∧ ¬ passes_through_point p l₂ → (∃! l : L, passes_through_point p l ∧ parallel_to l l₂)) ->
  (∃ (l₁ l₂ : L) (A : P), passes_through_point A l₁ ∧ ¬ passes_through_point A l₂ ∧ ∃ l : L, passes_through_point A l ∧ parallel_to l l₂) ) :=
sorry

end problem_correct_answer_l190_190689


namespace tomatoes_eaten_l190_190124

theorem tomatoes_eaten (initial_tomatoes : ℕ) (remaining_tomatoes : ℕ) (portion_eaten : ℚ)
  (h_init : initial_tomatoes = 21)
  (h_rem : remaining_tomatoes = 14)
  (h_portion : portion_eaten = 1/3) :
  initial_tomatoes - remaining_tomatoes = (portion_eaten * initial_tomatoes) :=
by
  sorry

end tomatoes_eaten_l190_190124


namespace least_k_square_divisible_by_240_l190_190021

theorem least_k_square_divisible_by_240 (k : ℕ) (h : ∃ m : ℕ, k ^ 2 = 240 * m) : k ≥ 60 :=
by
  sorry

end least_k_square_divisible_by_240_l190_190021


namespace problem_statement_l190_190383

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l190_190383


namespace motorist_travel_time_l190_190702

noncomputable def total_time (dist1 dist2 speed1 speed2 : ℝ) : ℝ :=
  (dist1 / speed1) + (dist2 / speed2)

theorem motorist_travel_time (speed1 speed2 : ℝ) (total_dist : ℝ) (half_dist : ℝ) :
  speed1 = 60 → speed2 = 48 → total_dist = 324 → half_dist = total_dist / 2 →
  total_time half_dist half_dist speed1 speed2 = 6.075 :=
by
  intros h1 h2 h3 h4
  simp [total_time, h1, h2, h3, h4]
  sorry

end motorist_travel_time_l190_190702


namespace heads_at_least_once_in_three_tosses_l190_190619

theorem heads_at_least_once_in_three_tosses :
  let total_outcomes := 8
  let all_tails_outcome := 1
  (1 - (all_tails_outcome / total_outcomes) = (7 / 8)) :=
by
  let total_outcomes := 8
  let all_tails_outcome := 1
  sorry

end heads_at_least_once_in_three_tosses_l190_190619


namespace wire_length_after_two_bends_is_three_l190_190770

-- Let's define the initial length and the property of bending the wire.
def initial_length : ℕ := 12

def half_length (length : ℕ) : ℕ :=
  length / 2

-- Define the final length after two bends.
def final_length_after_two_bends : ℕ :=
  half_length (half_length initial_length)

-- The theorem stating that the final length is 3 cm after two bends.
theorem wire_length_after_two_bends_is_three :
  final_length_after_two_bends = 3 :=
by
  -- The proof can be added later.
  sorry

end wire_length_after_two_bends_is_three_l190_190770


namespace proof_problem_l190_190251

-- Given conditions
variables {a b c : ℕ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a > b) (h5 : a^2 - a * b - a * c + b * c = 7)

-- Statement to prove
theorem proof_problem : a - c = 1 ∨ a - c = 7 :=
sorry

end proof_problem_l190_190251


namespace tops_count_l190_190674

def price_eq (C T : ℝ) : Prop := 3 * C + 6 * T = 1500 ∧ C + 12 * T = 1500

def tops_to_buy (C T : ℝ) (num_tops : ℝ) : Prop := 500 = 100 * num_tops

theorem tops_count (C T num_tops : ℝ) (h1 : price_eq C T) (h2 : tops_to_buy C T num_tops) : num_tops = 5 :=
by
  sorry

end tops_count_l190_190674


namespace sum_of_values_of_N_l190_190041

-- Given conditions
variables (N R : ℝ)
-- Condition that needs to be checked
def condition (N R : ℝ) : Prop := N + 3 / N = R ∧ N ≠ 0

-- The statement to prove
theorem sum_of_values_of_N (N R : ℝ) (h: condition N R) : N + (3 / N) = R :=
sorry

end sum_of_values_of_N_l190_190041


namespace problem_solution_l190_190072

variable (y Q : ℝ)

theorem problem_solution
  (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l190_190072


namespace length_more_than_breadth_by_200_l190_190118

-- Definitions and conditions
def rectangular_floor_length := 23
def painting_cost := 529
def painting_rate := 3
def floor_area := painting_cost / painting_rate
def floor_breadth := floor_area / rectangular_floor_length

-- Prove that the length is more than the breadth by 200%
theorem length_more_than_breadth_by_200 : 
  rectangular_floor_length = floor_breadth * (1 + 200 / 100) :=
sorry

end length_more_than_breadth_by_200_l190_190118


namespace value_of_m_l190_190498

theorem value_of_m (x m : ℝ) (h : x ≠ 3) (H : (x / (x - 3) = 2 - m / (3 - x))) : m = 3 :=
sorry

end value_of_m_l190_190498


namespace angle_sum_proof_l190_190629

theorem angle_sum_proof (x α β : ℝ) (h1 : 3 * x + 4 * x + α = 180)
 (h2 : α + 5 * x + β = 180)
 (h3 : 2 * x + 2 * x + 6 * x = 180) :
  x = 18 := by
  sorry

end angle_sum_proof_l190_190629


namespace solve_for_x_l190_190106

theorem solve_for_x (x : ℝ) (h : (1/3) + (1/x) = 2/3) : x = 3 :=
by
  sorry

end solve_for_x_l190_190106


namespace find_linear_function_l190_190344

theorem find_linear_function (a m : ℝ) : 
  (∀ x y : ℝ, (x, y) = (-2, -3) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, m) ∨ (x, y) = (1, 3) ∨ (x, y) = (a, 5) → 
  y = 2 * x + 1) → 
  (m = 1 ∧ a = 2) :=
by
  sorry

end find_linear_function_l190_190344


namespace rationalize_denominator_correct_l190_190942

noncomputable def rationalize_denominator : ℚ := 
  let A := 5
  let B := 49
  let C := 21
  -- Form is (5 * ∛49) / 21
  A + B + C

theorem rationalize_denominator_correct : rationalize_denominator = 75 :=
  by 
    -- The proof steps are omitted, as they are not required for this task
    sorry

end rationalize_denominator_correct_l190_190942


namespace sin_365_1_eq_m_l190_190595

noncomputable def sin_value (θ : ℝ) : ℝ := Real.sin (Real.pi * θ / 180)
variables (m : ℝ) (h : sin_value 5.1 = m)

theorem sin_365_1_eq_m : sin_value 365.1 = m :=
by sorry

end sin_365_1_eq_m_l190_190595


namespace propositions_correct_l190_190164

def vertical_angles (α β : ℝ) : Prop := ∃ γ, α = γ ∧ β = γ

def problem_statement : Prop :=
  (∀ α β, vertical_angles α β → α = β) ∧
  ¬(∀ α β, α = β → vertical_angles α β) ∧
  ¬(∀ α β, ¬vertical_angles α β → ¬(α = β)) ∧
  (∀ α β, ¬(α = β) → ¬vertical_angles α β)

theorem propositions_correct :
  problem_statement :=
by
  sorry

end propositions_correct_l190_190164


namespace ab_value_l190_190065

theorem ab_value 
  (a b : ℕ) 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h1 : a + b = 30)
  (h2 : 3 * a * b + 4 * a = 5 * b + 318) : 
  (a * b = 56) :=
sorry

end ab_value_l190_190065


namespace swimming_pool_time_l190_190158

theorem swimming_pool_time 
  (empty_rate : ℕ) (fill_rate : ℕ) (capacity : ℕ) (final_volume : ℕ) (t : ℕ)
  (h_empty : empty_rate = 120 / 4) 
  (h_fill : fill_rate = 120 / 6) 
  (h_capacity : capacity = 120) 
  (h_final : final_volume = 90) 
  (h_eq : capacity - (empty_rate - fill_rate) * t = final_volume) :
  t = 3 := 
sorry

end swimming_pool_time_l190_190158


namespace find_m_f_monotonicity_l190_190212

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / x - x ^ m

theorem find_m : ∃ (m : ℝ), f 4 m = -7 / 2 := sorry

noncomputable def g (x : ℝ) : ℝ := 2 / x - x

theorem f_monotonicity : ∀ x1 x2 : ℝ, (0 < x2 ∧ x2 < x1) → f x1 1 < f x2 1 := sorry

end find_m_f_monotonicity_l190_190212


namespace onions_left_on_scale_l190_190551

-- Define the given weights and conditions
def total_weight_of_40_onions : ℝ := 7680 -- in grams
def avg_weight_remaining_onions : ℝ := 190 -- grams
def avg_weight_removed_onions : ℝ := 206 -- grams

-- Converting original weight from kg to grams
def original_weight_kg_to_g (w_kg : ℝ) : ℝ := w_kg * 1000

-- Proof problem
theorem onions_left_on_scale (w_kg : ℝ) (n_total : ℕ) (n_removed : ℕ) 
    (total_weight : ℝ) (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ)
    (h1 : original_weight_kg_to_g w_kg = total_weight)
    (h2 : n_total = 40)
    (h3 : n_removed = 5)
    (h4 : avg_weight_remaining = avg_weight_remaining_onions)
    (h5 : avg_weight_removed = avg_weight_removed_onions) : 
    n_total - n_removed = 35 :=
sorry

end onions_left_on_scale_l190_190551


namespace area_of_parallelogram_l190_190979

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 :=
by
  rw [h_base, h_height]
  norm_num

end area_of_parallelogram_l190_190979


namespace large_cartridge_pages_correct_l190_190030

-- Define the conditions
def small_cartridge_pages : ℕ := 600
def medium_cartridge_pages : ℕ := 2 * 3 * small_cartridge_pages / 6
def large_cartridge_pages : ℕ := 2 * 3 * medium_cartridge_pages / 6

-- The theorem to prove
theorem large_cartridge_pages_correct :
  large_cartridge_pages = 1350 :=
by
  sorry

end large_cartridge_pages_correct_l190_190030


namespace triangle_angle_A_triangle_length_b_l190_190632

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (m n : ℝ × ℝ)
variable (S : ℝ)

theorem triangle_angle_A (h1 : a = 7) (h2 : c = 8) (h3 : m = (1, 7 * a)) (h4 : n = (-4 * a, Real.sin C))
  (h5 : m.1 * n.1 + m.2 * n.2 = 0) : 
  A = Real.pi / 6 := 
  sorry

theorem triangle_length_b (h1 : a = 7) (h2 : c = 8) (h3 : (7 * 8 * Real.sin B) / 2 = 16 * Real.sqrt 3) :
  b = Real.sqrt 97 :=
  sorry

end triangle_angle_A_triangle_length_b_l190_190632


namespace triangle_is_right_angled_l190_190010

theorem triangle_is_right_angled
  (a b c : ℝ)
  (h1 : a ≠ c)
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : c > 0)
  (h5 : ∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0 ∧ x ≠ 0) :
  c^2 + b^2 = a^2 :=
by sorry

end triangle_is_right_angled_l190_190010


namespace range_of_m_l190_190878

theorem range_of_m (m : ℝ) (x : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 2023) * x₁ + m + 2023) > ((m - 2023) * x₂ + m + 2023)) → m < 2023 :=
by
  sorry

end range_of_m_l190_190878


namespace inequality_proof_l190_190206

open Real

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_proof_l190_190206


namespace min_L_Trominos_l190_190654

theorem min_L_Trominos (x y : ℕ) :
  (2020 * 2021 % 4 = 0) →
  (4 * x + 4 * y = 2020 * 2021) →
  (2020 * 1010 ≥ 2 * x + y) →
  y = 1010 :=
by
  sorry

end min_L_Trominos_l190_190654


namespace area_new_rectangle_l190_190745

-- Define the given rectangle's dimensions
def a : ℕ := 3
def b : ℕ := 4

-- Define the diagonal of the given rectangle
def d : ℕ := Nat.sqrt (a^2 + b^2)

-- Define the new rectangle's dimensions
def length_new : ℕ := d + a
def breadth_new : ℕ := d - b

-- The target area of the new rectangle
def area_new : ℕ := length_new * breadth_new

-- Prove that the area of the new rectangle is 8 square units
theorem area_new_rectangle (h : d = 5) : area_new = 8 := by
  -- Indicate that proof steps are not provided
  sorry

end area_new_rectangle_l190_190745


namespace hotel_loss_l190_190302

variable (operations_expenses : ℝ)
variable (fraction_payment : ℝ)

theorem hotel_loss :
  operations_expenses = 100 →
  fraction_payment = 3 / 4 →
  let total_payment := fraction_payment * operations_expenses in
  let loss := operations_expenses - total_payment in
  loss = 25 :=
by
  intros h₁ h₂
  have tstp : total_payment = 75 := by
    rw [h₁, h₂]
    norm_num
  have lss : loss = 25 := by
    rw [h₁, tstp]
    norm_num
  exact lss

end hotel_loss_l190_190302


namespace tomatoes_picked_second_week_l190_190990

-- Define the constants
def initial_tomatoes : Nat := 100
def fraction_picked_first_week : Nat := 1 / 4
def remaining_tomatoes : Nat := 15

-- Theorem to prove the number of tomatoes Jane picked in the second week
theorem tomatoes_picked_second_week (x : Nat) :
  let T := initial_tomatoes
  let p := fraction_picked_first_week
  let r := remaining_tomatoes
  let first_week_pick := T * p
  let remaining_after_first := T - first_week_pick
  let total_picked := remaining_after_first - r
  let second_week_pick := total_picked / 3
  second_week_pick = 20 := 
sorry

end tomatoes_picked_second_week_l190_190990


namespace sum_of_distinct_primes_even_l190_190502

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def even (n : ℕ) : Prop := n % 2 = 0

def is_sum_even (a b : ℕ) : Prop := even (a + b)

def num_combinations (n k : ℕ) : ℕ :=
  n.choose k

def num_even_sum_pairs : ℕ :=
  (first_seven_primes.filter even).length * 
  (first_seven_primes.filter (fun p => ¬even p)).length

def total_combinations : ℕ := num_combinations 7 2

def probability_even_sum : ℚ :=
  (total_combinations - num_even_sum_pairs) / total_combinations

theorem sum_of_distinct_primes_even :
  probability_even_sum = 5 / 7 := 
  by
  sorry

end sum_of_distinct_primes_even_l190_190502


namespace each_person_bid_count_l190_190315

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l190_190315


namespace arithmetic_expression_correct_l190_190820

theorem arithmetic_expression_correct :
  let a := 10
  let b := 10
  let c := 4
  let d := 2
  (d + c / a) * a = 24 :=
by
  let a := 10
  let b := 10
  let c := 4
  let d := 2
  calc
    (d + c / a) * a
        = (2 + 4 / 10) * 10 : by rw [←add_mul, mul_comm 10 (4 / 10), div_mul_cancel'] -- real arithmetic correctness
    ... = 24 : by norm_num [div_eq_mul_inv, ←mul_assoc, mul_inv_cancel_left]

end arithmetic_expression_correct_l190_190820


namespace sally_baseball_cards_l190_190263

theorem sally_baseball_cards (initial_cards sold_cards : ℕ) (h1 : initial_cards = 39) (h2 : sold_cards = 24) :
  (initial_cards - sold_cards = 15) :=
by
  -- Proof needed
  sorry

end sally_baseball_cards_l190_190263


namespace igor_number_proof_l190_190162

noncomputable def igor_number (init_lineup : List ℕ) (igor_num : ℕ) : Prop :=
  let after_first_command := [9, 11, 10, 6, 8, 7] -- Results after first command 
  let after_second_command := [9, 11, 10, 8] -- Results after second command
  let after_third_command := [11, 10, 8] -- Results after third command
  ∃ (idx : ℕ), init_lineup.get? idx = some igor_num ∧
    (∀ new_lineup, 
       (new_lineup = after_first_command ∨ new_lineup = after_second_command ∨ new_lineup = after_third_command) →
       igor_num ∉ new_lineup) ∧ 
    after_third_command.length = 3

theorem igor_number_proof : igor_number [9, 1, 11, 2, 10, 3, 6, 4, 8, 5, 7] 5 :=
  sorry 

end igor_number_proof_l190_190162


namespace pairs_of_polygons_with_angle_difference_l190_190759

theorem pairs_of_polygons_with_angle_difference :
  ∃ (pairs : ℕ), pairs = 52 ∧ ∀ (n k : ℕ), n > k ∧ (360 / k - 360 / n = 1) :=
sorry

end pairs_of_polygons_with_angle_difference_l190_190759


namespace surface_dots_sum_l190_190293

-- Define the sum of dots on opposite faces of a standard die
axiom sum_opposite_faces (x y : ℕ) : x + y = 7

-- Define the large cube dimensions
def large_cube_dimension : ℕ := 3

-- Define the total number of small cubes
def num_small_cubes : ℕ := large_cube_dimension ^ 3

-- Calculate the number of faces on the surface of the large cube
def num_surface_faces : ℕ := 6 * large_cube_dimension ^ 2

-- Given the sum of opposite faces, compute the total number of dots on the surface
theorem surface_dots_sum : num_surface_faces / 2 * 7 = 189 := by
  sorry

end surface_dots_sum_l190_190293


namespace sqrt_0_09_eq_0_3_l190_190813

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end sqrt_0_09_eq_0_3_l190_190813


namespace like_terms_sum_l190_190748

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 4) (h2 : 3 - n = 1) : m + n = 4 :=
by
  sorry

end like_terms_sum_l190_190748


namespace age_of_15th_student_l190_190113

theorem age_of_15th_student (T : ℕ) (T8 : ℕ) (T6 : ℕ)
  (avg_15_students : T / 15 = 15)
  (avg_8_students : T8 / 8 = 14)
  (avg_6_students : T6 / 6 = 16) :
  (T - (T8 + T6)) = 17 := by
  sorry

end age_of_15th_student_l190_190113


namespace emir_needs_more_money_l190_190177

def dictionary_cost : ℕ := 5
def dinosaur_book_cost : ℕ := 11
def cookbook_cost : ℕ := 5
def saved_money : ℕ := 19
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost
def additional_money_needed : ℕ := total_cost - saved_money

theorem emir_needs_more_money : additional_money_needed = 2 := by
  sorry

end emir_needs_more_money_l190_190177


namespace eldora_boxes_paper_clips_l190_190461

theorem eldora_boxes_paper_clips (x y : ℝ)
  (h1 : 1.85 * x + 7 * y = 55.40)
  (h2 : 1.85 * 12 + 10 * y = 61.70)
  (h3 : 1.85 = 1.85) : -- Given && Asserting the constant price of one box

  x = 15 :=
by
  sorry

end eldora_boxes_paper_clips_l190_190461


namespace Amanda_tickets_third_day_l190_190708

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l190_190708


namespace lighter_shopping_bag_weight_l190_190537

theorem lighter_shopping_bag_weight :
  ∀ (G : ℕ), (G + 7 = 10) → (G = 3) := by
  intros G h
  sorry

end lighter_shopping_bag_weight_l190_190537


namespace each_person_bids_five_times_l190_190312

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l190_190312


namespace rate_of_interest_per_annum_l190_190153

theorem rate_of_interest_per_annum (R : ℝ) : 
  (5000 * R * 2 / 100) + (3000 * R * 4 / 100) = 1540 → 
  R = 7 := 
by {
  sorry
}

end rate_of_interest_per_annum_l190_190153


namespace sum_of_arithmetic_sequence_6_7_8_l190_190485

theorem sum_of_arithmetic_sequence_6_7_8 {a : ℕ → ℝ} (a1 : ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_13 : ∑ i in Finset.range 13, a (i + 1) = 39) : 
  a 6 + a 7 + a 8 = 39 :=
sorry

end sum_of_arithmetic_sequence_6_7_8_l190_190485


namespace numbers_square_and_cube_root_l190_190534

theorem numbers_square_and_cube_root (x : ℝ) : (x^2 = x ∧ x^3 = x) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by
  sorry

end numbers_square_and_cube_root_l190_190534


namespace sum_mod_9_is_6_l190_190848

noncomputable def sum_modulo_9 : ℤ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

theorem sum_mod_9_is_6 : sum_modulo_9 % 9 = 6 := 
  by
    sorry

end sum_mod_9_is_6_l190_190848


namespace cargo_arrival_in_days_l190_190983

-- Definitions for conditions
def days_navigate : ℕ := 21
def days_customs : ℕ := 4
def days_transport : ℕ := 7
def days_departed : ℕ := 30

-- Calculate the days since arrival in Vancouver
def days_arrival_vancouver : ℕ := days_departed - days_navigate

-- Calculate the days since customs processes finished
def days_since_customs_done : ℕ := days_arrival_vancouver - days_customs

-- Calculate the days for cargo to arrive at the warehouse from today
def days_until_arrival : ℕ := days_transport - days_since_customs_done

-- Expected number of days from today for the cargo to arrive at the warehouse
theorem cargo_arrival_in_days : days_until_arrival = 2 := by
  -- Insert the proof steps here
  sorry

end cargo_arrival_in_days_l190_190983


namespace min_blue_edges_l190_190176

def tetrahedron_min_blue_edges : ℕ := sorry

theorem min_blue_edges (edges_colored : ℕ → Bool) (face_has_blue_edge : ℕ → Bool) 
    (H1 : ∀ face, face_has_blue_edge face)
    (H2 : ∀ edge, face_has_blue_edge edge = True → edges_colored edge = True) : 
    tetrahedron_min_blue_edges = 2 := 
sorry

end min_blue_edges_l190_190176


namespace arccos_gt_arctan_l190_190045

theorem arccos_gt_arctan (x : ℝ) (h : x ∈ set.Icc (-1 : ℝ) 1) : 
  (arccos x > arctan x) ↔ (x < real.sqrt 2 / 2) :=
sorry

end arccos_gt_arctan_l190_190045


namespace problem_arithmetic_l190_190962

variable {α : Type*} [LinearOrderedField α] 

def arithmetic_sum (a d : α) (n : ℕ) : α := n * (2 * a + (n - 1) * d) / 2
def arithmetic_term (a d : α) (k : ℕ) : α := a + (k - 1) * d

theorem problem_arithmetic (a3 a2015 : ℝ) 
  (h_roots : a3 + a2015 = 10) 
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h_sum : ∀ n, S n = arithmetic_sum a3 ((a2015 - a3) / 2012) n) 
  (h_an : ∀ k, a k = arithmetic_term a3 ((a2015 - a3) / 2012) k) :
  (S 2017) / 2017 + a 1009 = 10 := by
sorry

end problem_arithmetic_l190_190962


namespace other_endpoint_product_l190_190411

theorem other_endpoint_product :
  ∀ (x y : ℤ), 
    (3 = (x + 7) / 2) → 
    (-5 = (y - 1) / 2) → 
    x * y = 9 :=
by
  intro x y h1 h2
  sorry

end other_endpoint_product_l190_190411


namespace tom_remaining_balloons_l190_190424

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end tom_remaining_balloons_l190_190424


namespace product_of_two_numbers_l190_190405

theorem product_of_two_numbers (x y : ℝ) (h_diff : x - y = 12) (h_sum_of_squares : x^2 + y^2 = 245) : x * y = 50.30 :=
sorry

end product_of_two_numbers_l190_190405


namespace total_pieces_of_gum_l190_190795

theorem total_pieces_of_gum (packages pieces_per_package : ℕ) 
  (h_packages : packages = 9)
  (h_pieces_per_package : pieces_per_package = 15) : 
  packages * pieces_per_package = 135 := by
  subst h_packages
  subst h_pieces_per_package
  exact Nat.mul_comm 9 15 ▸ rfl

end total_pieces_of_gum_l190_190795


namespace log_base_equality_l190_190731

theorem log_base_equality : log 4 / log 16 = 1 / 2 := 
by sorry

end log_base_equality_l190_190731


namespace ab_difference_l190_190499

theorem ab_difference (a b : ℝ) 
  (h1 : 10 = a * 3 + b)
  (h2 : 22 = a * 7 + b) : 
  a - b = 2 := 
  sorry

end ab_difference_l190_190499


namespace geometric_sum_ratio_l190_190389

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l190_190389


namespace find_a5_l190_190599

theorem find_a5 (a : ℕ → ℤ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1) 
  (h2 : a 2 + a 4 + a 6 = 18) : 
  a 5 = 5 :=
sorry

end find_a5_l190_190599


namespace least_amount_to_add_l190_190546

theorem least_amount_to_add (current_amount : ℕ) (n : ℕ) (divisor : ℕ) [NeZero divisor]
  (current_amount_eq : current_amount = 449774) (n_eq : n = 1) (divisor_eq : divisor = 6) :
  ∃ k : ℕ, (current_amount + k) % divisor = 0 ∧ k = n := by
  sorry

end least_amount_to_add_l190_190546


namespace mat_inv_int_entries_l190_190380

open scoped Matrix

theorem mat_inv_int_entries (A B : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A.det ∈ {-1, 1})
  (h1 : (A + B).det ∈ {-1, 1})
  (h2 : (A + 2 • B).det ∈ {-1, 1})
  (h3 : (A + 3 • B).det ∈ {-1, 1})
  (h4 : (A + 4 • B).det ∈ {-1, 1}) :
  ∃ (C : Matrix (Fin 2) (Fin 2) ℤ), (C.det = 1 ∨ C.det = -1) ∧ C = (A + 5 • B) :=
sorry

end mat_inv_int_entries_l190_190380


namespace find_m_l190_190334

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d 

noncomputable def sum_first_n_terms (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

theorem find_m {a S : ℕ → ℤ} (d : ℤ) (m : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : a 1 = 1)
  (h4 : S 3 = a 5)
  (h5 : a m = 2011) :
  m = 1006 :=
sorry

end find_m_l190_190334


namespace problem_statement_l190_190090

theorem problem_statement (P : ℝ) (h : P = 1 / (Real.log 11 / Real.log 2) + 1 / (Real.log 11 / Real.log 3) + 1 / (Real.log 11 / Real.log 4) + 1 / (Real.log 11 / Real.log 5)) : 1 < P ∧ P < 2 := 
sorry

end problem_statement_l190_190090


namespace area_trapezoid_def_l190_190155

noncomputable def area_trapezoid (a : ℝ) (h : a ≠ 0) : ℝ :=
  let b := 108 / a
  let DE := a / 2
  let FG := b / 3
  let height := b / 2
  (DE + FG) * height / 2

theorem area_trapezoid_def (a : ℝ) (h : a ≠ 0) :
  area_trapezoid a h = 18 + 18 / a :=
by
  sorry

end area_trapezoid_def_l190_190155


namespace calculate_avg_l190_190950

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem calculate_avg :
  avg3 (avg3 1 2 0) (avg2 0 2) 0 = 2 / 3 :=
by
  sorry

end calculate_avg_l190_190950


namespace combined_original_price_of_books_l190_190856

theorem combined_original_price_of_books (p1 p2 : ℝ) (h1 : p1 / 8 = 8) (h2 : p2 / 9 = 9) :
  p1 + p2 = 145 :=
sorry

end combined_original_price_of_books_l190_190856


namespace intersect_is_one_l190_190488

def SetA : Set ℝ := {x | 0 < x ∧ x < 2}

def SetB : Set ℝ := {0, 1, 2, 3}

theorem intersect_is_one : SetA ∩ SetB = {1} :=
by
  sorry

end intersect_is_one_l190_190488


namespace net_percentage_error_l190_190998

noncomputable section
def calculate_percentage_error (true_side excess_error deficit_error : ℝ) : ℝ :=
  let measured_side1 := true_side * (1 + excess_error / 100)
  let measured_side2 := measured_side1 * (1 - deficit_error / 100)
  let true_area := true_side ^ 2
  let calculated_area := measured_side2 * true_side
  let percentage_error := ((true_area - calculated_area) / true_area) * 100
  percentage_error

theorem net_percentage_error 
  (S : ℝ) (h1 : S > 0) : calculate_percentage_error S 3 (-4) = 1.12 := by
  sorry

end net_percentage_error_l190_190998


namespace remainder_expression_l190_190089

theorem remainder_expression (x y u v : ℕ) (hy_pos : y > 0) (h : x = u * y + v) (hv : 0 ≤ v) (hv_lt : v < y) :
  (x + 4 * u * y) % y = v :=
by
  sorry

end remainder_expression_l190_190089


namespace sin_alpha_minus_3pi_l190_190203

theorem sin_alpha_minus_3pi (α : ℝ) (h : Real.sin α = 3/5) : Real.sin (α - 3 * Real.pi) = -3/5 :=
by
  sorry

end sin_alpha_minus_3pi_l190_190203


namespace james_needs_more_marbles_l190_190242

def number_of_additional_marbles (friends marbles : Nat) : Nat :=
  let required_marbles := (friends * (friends + 1)) / 2
  (if marbles < required_marbles then required_marbles - marbles else 0)

theorem james_needs_more_marbles :
  number_of_additional_marbles 15 80 = 40 := by
  sorry

end james_needs_more_marbles_l190_190242


namespace solve_for_x_l190_190738

variable {x : ℝ}

def is_positive (x : ℝ) : Prop := x > 0

def area_of_triangle_is_150 (x : ℝ) : Prop :=
  let base := 2 * x
  let height := 3 * x
  (1/2) * base * height = 150

theorem solve_for_x (hx : is_positive x) (ha : area_of_triangle_is_150 x) : x = 5 * Real.sqrt 2 := by
  sorry

end solve_for_x_l190_190738


namespace rationalize_denominator_l190_190937

theorem rationalize_denominator : 
  ∃ A B C : ℤ, C > 0 ∧ ∃ k : ℕ, B = k ∧ (∀ p, nat.prime p → p^3 ∣ k → false) ∧ 
  (5:ℚ) / (3 * (real.cbrt 7)) = (A * real.cbrt B : ℚ) / C ∧ A + B + C = 75 :=
by 
  sorry

end rationalize_denominator_l190_190937


namespace tax_diminished_by_16_percent_l190_190535

variables (T X : ℝ)

-- Condition: The new revenue is 96.6% of the original revenue
def new_revenue_effect : Prop :=
  (1.15 * (T - X) / 100) = (T / 100) * 0.966

-- Target: Prove that X is 16% of T
theorem tax_diminished_by_16_percent (h : new_revenue_effect T X) : X = 0.16 * T :=
sorry

end tax_diminished_by_16_percent_l190_190535


namespace inequality_correct_l190_190355

variable {a b : ℝ}

theorem inequality_correct (h₁ : a < 1) (h₂ : b > 1) : ab < a + b :=
sorry

end inequality_correct_l190_190355


namespace acute_triangle_integers_count_l190_190057

theorem acute_triangle_integers_count :
  ∃ (x_vals : List ℕ), (∀ x ∈ x_vals, 7 < x ∧ x < 33 ∧ (if x > 20 then x^2 < 569 else x > Int.sqrt 231)) ∧ x_vals.length = 8 :=
by
  sorry

end acute_triangle_integers_count_l190_190057


namespace q_at_14_l190_190252

noncomputable def q (x : ℝ) : ℝ := - (1 / 2) * x^2 + x + 2

theorem q_at_14 : q 14 = -82 := by
  sorry

end q_at_14_l190_190252


namespace garden_area_l190_190244

variable (L W A : ℕ)
variable (H1 : 3000 = 50 * L)
variable (H2 : 3000 = 15 * (2*L + 2*W))

theorem garden_area : A = 2400 :=
by
  sorry

end garden_area_l190_190244


namespace arithmetic_sequence_sum_10_l190_190749

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_10 (a_1 a_3 a_7 a_9 : ℤ)
    (h1 : ∃ a_1, a_3 = a_1 - 4)
    (h2 : a_7 = a_1 - 12)
    (h3 : a_9 = a_1 - 16)
    (h4 : a_7 * a_7 = a_3 * a_9)
    : sum_of_first_n_terms a_1 (-2) 10 = 110 :=
by 
  sorry

end arithmetic_sequence_sum_10_l190_190749


namespace geometric_sum_S6_l190_190639

variable {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Conditions: S_n represents the sum of the first n terms of the geometric sequence {a_n}
-- and we have S_2 = 4 and S_4 = 6
theorem geometric_sum_S6 (S : ℕ → ℝ) (h1 : S 2 = 4) (h2 : S 4 = 6) : S 6 = 7 :=
sorry

end geometric_sum_S6_l190_190639


namespace line_l_equation_symmetrical_line_equation_l190_190064

theorem line_l_equation (x y : ℝ) (h₁ : 3 * x + 4 * y - 2 = 0) (h₂ : 2 * x + y + 2 = 0) :
  2 * x + y + 2 = 0 :=
sorry

theorem symmetrical_line_equation (x y : ℝ) :
  (2 * x + y + 2 = 0) → (2 * x + y - 2 = 0) :=
sorry

end line_l_equation_symmetrical_line_equation_l190_190064


namespace rationalize_denominator_l190_190947

theorem rationalize_denominator (A B C : ℤ) (hB : ¬ ∃ p : ℤ, p ≥ 2 ∧ p ^ 3 ∣ B) (hC : C > 0) :
  (A = 5) ∧ (B = 49) ∧ (C = 21) → A + B + C = 75 :=
by
  intro h
  rcases h with ⟨hA, hB, hC⟩
  rw [hA, hB, hC]
  simp
  sorry

end rationalize_denominator_l190_190947


namespace percentage_of_students_70_79_l190_190556

def tally_90_100 := 6
def tally_80_89 := 9
def tally_70_79 := 8
def tally_60_69 := 6
def tally_50_59 := 3
def tally_below_50 := 1

def total_students := tally_90_100 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50

theorem percentage_of_students_70_79 : (tally_70_79 : ℚ) / total_students = 8 / 33 :=
by
  sorry

end percentage_of_students_70_79_l190_190556


namespace bug_crawl_distance_l190_190436

-- Define the positions visited by the bug
def start_position := -3
def first_stop := 0
def second_stop := -8
def final_stop := 10

-- Define the function to calculate the total distance crawled by the bug
def total_distance : ℤ :=
  abs (first_stop - start_position) + abs (second_stop - first_stop) + abs (final_stop - second_stop)

-- Prove that the total distance is 29 units
theorem bug_crawl_distance : total_distance = 29 :=
by
  -- Definitions are used here to validate the statement
  sorry

end bug_crawl_distance_l190_190436


namespace total_cups_sold_is_46_l190_190264

-- Define the number of cups sold last week
def cups_sold_last_week : ℕ := 20

-- Define the percentage increase
def percentage_increase : ℕ := 30

-- Calculate the number of cups sold this week
def cups_sold_this_week : ℕ := cups_sold_last_week + (cups_sold_last_week * percentage_increase / 100)

-- Calculate the total number of cups sold over both weeks
def total_cups_sold : ℕ := cups_sold_last_week + cups_sold_this_week

-- State the theorem to prove the total number of cups sold
theorem total_cups_sold_is_46 : total_cups_sold = 46 := sorry

end total_cups_sold_is_46_l190_190264


namespace least_cost_of_grass_seed_l190_190984

-- Definitions of the prices and weights
def price_per_bag (size : Nat) : Float :=
  if size = 5 then 13.85
  else if size = 10 then 20.40
  else if size = 25 then 32.25
  else 0.0

-- The conditions for the weights and costs
def valid_weight_range (total_weight : Nat) : Prop :=
  65 ≤ total_weight ∧ total_weight ≤ 80

-- Calculate the total cost given quantities of each bag size
def total_cost (bag5 : Nat) (bag10 : Nat) (bag25 : Nat) : Float :=
  Float.ofNat bag5 * price_per_bag 5 + Float.ofNat bag10 * price_per_bag 10 + Float.ofNat bag25 * price_per_bag 25

-- Correct cost for the minimum possible cost within the given weight range
def min_possible_cost : Float := 98.75

-- Proof statement to be proven
theorem least_cost_of_grass_seed : ∃ (bag5 bag10 bag25 : Nat), 
  valid_weight_range (bag5 * 5 + bag10 * 10 + bag25 * 25) ∧ total_cost bag5 bag10 bag25 = min_possible_cost :=
sorry

end least_cost_of_grass_seed_l190_190984


namespace common_difference_of_arithmetic_seq_l190_190198

-- Definition of arithmetic sequence sum and general term
def arithmetic_sum(n a1 d : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d
def arithmetic_term(n a1 d : ℕ) : ℕ := a1 + (n - 1) * d

theorem common_difference_of_arithmetic_seq (a1 d : ℕ) :
  arithmetic_sum 13 a1 d = 104 ∧ arithmetic_term 6 a1 d = 5 → d = 3 :=
by {
  sorry
}

end common_difference_of_arithmetic_seq_l190_190198


namespace geometric_sum_ratio_l190_190390

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l190_190390


namespace doris_weeks_to_cover_expenses_l190_190728

-- Define the constants and conditions from the problem
def hourly_rate : ℝ := 20
def monthly_expenses : ℝ := 1200
def weekday_hours_per_day : ℝ := 3
def weekdays_per_week : ℝ := 5
def saturday_hours : ℝ := 5

-- Calculate total hours worked per week
def weekly_hours := (weekday_hours_per_day * weekdays_per_week) + saturday_hours

-- Calculate weekly earnings
def weekly_earnings := hourly_rate * weekly_hours

-- Finally, the number of weeks required to meet the monthly expenses
def required_weeks := monthly_expenses / weekly_earnings

-- The theorem to prove
theorem doris_weeks_to_cover_expenses : required_weeks = 3 := by
  -- We skip the proof but indicate it needs to be provided
  sorry

end doris_weeks_to_cover_expenses_l190_190728


namespace real_numbers_inequality_l190_190053

theorem real_numbers_inequality 
  (x : Fin 2017 → ℝ) : 
  (Finset.univ.sum (λ i => Finset.univ.sum (λ j => |x i + x j|))) 
  ≥ 
  2017 * (Finset.univ.sum (λ i => |x i|)) :=
by sorry

end real_numbers_inequality_l190_190053


namespace bus_speed_proof_l190_190992
noncomputable def speed_of_bus (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed_mps := train_length / time_to_pass
  let bus_speed_mps := relative_speed_mps - train_speed_mps
  bus_speed_mps * 3.6

theorem bus_speed_proof : 
  speed_of_bus 220 90 5.279577633789296 = 60 :=
by
  sorry

end bus_speed_proof_l190_190992


namespace service_center_location_l190_190801

theorem service_center_location : 
  ∀ (milepost4 milepost9 : ℕ), 
  milepost4 = 30 → milepost9 = 150 → 
  (∃ milepost_service_center : ℕ, milepost_service_center = milepost4 + ((milepost9 - milepost4) / 2)) → 
  milepost_service_center = 90 :=
by
  intros milepost4 milepost9 h4 h9 hsc
  sorry

end service_center_location_l190_190801


namespace intersection_eq_l190_190339

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end intersection_eq_l190_190339


namespace find_number_l190_190017

theorem find_number (N : ℕ) (h : N / 7 = 12 ∧ N % 7 = 5) : N = 89 := 
by
  sorry

end find_number_l190_190017


namespace probability_same_color_set_l190_190247

theorem probability_same_color_set 
  (black_pairs blue_pairs : ℕ)
  (green_pairs : {g : Finset (ℕ × ℕ) // g.card = 3})
  (total_pairs := 15)
  (total_shoes := total_pairs * 2) :
  2 * black_pairs + 2 * blue_pairs + green_pairs.val.card * 2 = total_shoes →
  ∃ probability : ℚ, 
    probability = 89 / 435 :=
by
  intro h_total_shoes
  let black_shoes := black_pairs * 2
  let blue_shoes := blue_pairs * 2
  let green_shoes := green_pairs.val.card * 2
  
  have h_black_probability : ℚ := (black_shoes / total_shoes) * (black_pairs / (total_shoes - 1))
  have h_blue_probability : ℚ := (blue_shoes / total_shoes) * (blue_pairs / (total_shoes - 1))
  have h_green_probability : ℚ := (green_shoes / total_shoes) * (green_pairs.val.card / (total_shoes - 1))
  
  have h_total_probability : ℚ := h_black_probability + h_blue_probability + h_green_probability
  
  use h_total_probability
  sorry

end probability_same_color_set_l190_190247


namespace sum_of_squares_of_coefficients_l190_190918

theorem sum_of_squares_of_coefficients :
  ∃ a b c d e f : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 :=
by
  sorry

end sum_of_squares_of_coefficients_l190_190918


namespace average_cost_of_fruit_l190_190647

theorem average_cost_of_fruit : 
  (12 * 2 + 4 * 1 + 4 * 3) / (12 + 4 + 4) = 2 := 
by
  -- Given conditions as definitions
  let cost_apple := 2     -- cost per apple
  let cost_banana := 1    -- cost per banana
  let cost_orange := 3    -- cost per orange
  let qty_apples := 12    -- number of apples bought
  let qty_bananas := 4    -- number of bananas bought
  let qty_oranges := 4    -- number of oranges bought
  
  -- Average cost calculation
  have total_cost := qty_apples * cost_apple + qty_bananas * cost_banana + qty_oranges * cost_orange
  have total_qty := qty_apples + qty_bananas + qty_oranges
  have average_cost := total_cost.toRat / total_qty.toRat
  
  show average_cost = 2 by sorry

end average_cost_of_fruit_l190_190647


namespace quinton_cupcakes_l190_190098

theorem quinton_cupcakes (students_Delmont : ℕ) (students_Donnelly : ℕ)
                         (num_teachers_nurse_principal : ℕ) (leftover : ℕ) :
  students_Delmont = 18 → students_Donnelly = 16 →
  num_teachers_nurse_principal = 4 → leftover = 2 →
  students_Delmont + students_Donnelly + num_teachers_nurse_principal + leftover = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end quinton_cupcakes_l190_190098


namespace patio_total_tiles_l190_190563

theorem patio_total_tiles (s : ℕ) (red_tiles : ℕ) (h1 : s % 2 = 1) (h2 : red_tiles = 2 * s - 1) (h3 : red_tiles = 61) :
  s * s = 961 :=
by
  sorry

end patio_total_tiles_l190_190563


namespace coordinates_P_wrt_origin_l190_190272

/-- Define a point P with coordinates we are given. -/
def P : ℝ × ℝ := (-1, 2)

/-- State that the coordinates of P with respect to the origin O are (-1, 2). -/
theorem coordinates_P_wrt_origin : P = (-1, 2) :=
by
  -- Proof would go here
  sorry

end coordinates_P_wrt_origin_l190_190272


namespace identical_numbers_minimum_l190_190060

theorem identical_numbers_minimum (a: Fin 100 → ℕ) (h: ∑ i, a i ≤ 1600) : 
  ∃ x ∈ Finset.univ.image a, 4 ≤ (Finset.univ.filter (λ i, a i = x)).card :=
sorry

end identical_numbers_minimum_l190_190060


namespace find_the_triplet_l190_190873

theorem find_the_triplet (x y z : ℕ) (h : x + y + z = x * y * z) : (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
by
  sorry

end find_the_triplet_l190_190873


namespace third_square_is_G_l190_190858

-- Conditions
-- Define eight 2x2 squares, where the last placed square is E
def squares : List String := ["F", "H", "G", "D", "A", "B", "C", "E"]

-- Let the third square be G
def third_square := "G"

-- Proof statement
theorem third_square_is_G : squares.get! 2 = third_square :=
by
  sorry

end third_square_is_G_l190_190858


namespace min_value_of_expression_l190_190750

-- positive real numbers a and b
variables (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
-- given condition: 1/a + 9/b = 6
variable (h : 1 / a + 9 / b = 6)

theorem min_value_of_expression : (a + 1) * (b + 9) ≥ 16 := by
  sorry

end min_value_of_expression_l190_190750


namespace range_of_a_l190_190876

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := -(x + 1)^2 + a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f x2 ≤ g x1 a) ↔ a ≥ -1 / Real.exp 1 :=
by
  -- proof would go here
  sorry

end range_of_a_l190_190876


namespace oil_bill_january_l190_190828

-- Define the constants and variables
variables (F J : ℝ)

-- Define the conditions
def condition1 : Prop := F / J = 5 / 4
def condition2 : Prop := (F + 45) / J = 3 / 2

-- Define the main theorem stating the proof problem
theorem oil_bill_january 
  (h1 : condition1 F J) 
  (h2 : condition2 F J) : 
  J = 180 :=
sorry

end oil_bill_january_l190_190828


namespace cube_split_l190_190055

theorem cube_split (m : ℕ) (h1 : m > 1)
  (h2 : ∃ (p : ℕ), (p = (m - 1) * (m^2 + m + 1) ∨ p = (m - 1)^2 ∨ p = (m - 1)^2 + 2) ∧ p = 2017) :
  m = 46 :=
by {
    sorry
}

end cube_split_l190_190055


namespace min_pieces_pie_l190_190152

theorem min_pieces_pie (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n : ℕ, n = p + q - 1 ∧ 
    (∀ m, m < n → ¬ (∀ k : ℕ, (k < p → n % p = 0) ∧ (k < q → n % q = 0))) :=
sorry

end min_pieces_pie_l190_190152


namespace scientific_notation_21600_l190_190857

theorem scientific_notation_21600 : ∃ (a : ℝ) (n : ℤ), 21600 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.16 ∧ n = 4 :=
by
  sorry

end scientific_notation_21600_l190_190857


namespace solution_pair_exists_l190_190122

theorem solution_pair_exists :
  ∃ (p q : ℚ), 
    ∀ (x : ℚ), 
      (p * x^4 + q * x^3 + 45 * x^2 - 25 * x + 10 = 
      (5 * x^2 - 3 * x + 2) * 
      ( (5 / 2) * x^2 - 5 * x + 5)) ∧ 
      (p = (25 / 2)) ∧ 
      (q = (-65 / 2)) :=
by
  sorry

end solution_pair_exists_l190_190122


namespace sara_spent_on_movies_l190_190788

def cost_of_movie_tickets : ℝ := 2 * 10.62
def cost_of_rented_movie : ℝ := 1.59
def cost_of_purchased_movie : ℝ := 13.95

theorem sara_spent_on_movies :
  cost_of_movie_tickets + cost_of_rented_movie + cost_of_purchased_movie = 36.78 := by
  sorry

end sara_spent_on_movies_l190_190788


namespace triangle_inequality_l190_190086

theorem triangle_inequality (a b c : ℝ) (h1 : b + c > a) (h2 : c + a > b) (h3 : a + b > c) :
  ab + bc + ca ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (ab + bc + ca) :=
by
  sorry

end triangle_inequality_l190_190086


namespace rationalize_denominator_correct_l190_190943

noncomputable def rationalize_denominator : ℚ := 
  let A := 5
  let B := 49
  let C := 21
  -- Form is (5 * ∛49) / 21
  A + B + C

theorem rationalize_denominator_correct : rationalize_denominator = 75 :=
  by 
    -- The proof steps are omitted, as they are not required for this task
    sorry

end rationalize_denominator_correct_l190_190943


namespace weeks_to_cover_expense_l190_190723

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l190_190723


namespace marts_income_percentage_l190_190142

variable (J T M : ℝ)

theorem marts_income_percentage (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marts_income_percentage_l190_190142


namespace k_range_l190_190215

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  (Real.log x) - x - x * Real.exp (-x) - k

theorem k_range (k : ℝ) : (∀ x > 0, ∃ x > 0, f x k = 0) ↔ k ≤ -1 - (1 / Real.exp 1) :=
sorry

end k_range_l190_190215


namespace find_x_l190_190075

theorem find_x (x : ℝ) : 
  45 - (28 - (37 - (x - 17))) = 56 ↔ x = 15 := 
by
  sorry

end find_x_l190_190075


namespace Jaymee_age_l190_190910

/-- Given that Jaymee is 2 years older than twice the age of Shara,
    and Shara is 10 years old, prove that Jaymee is 22 years old. -/
theorem Jaymee_age (Shara_age : ℕ) (h1 : Shara_age = 10) :
  let Jaymee_age := 2 * Shara_age + 2
  in Jaymee_age = 22 :=
by
  have h2 : 2 * Shara_age + 2 = 22 := sorry
  exact h2

end Jaymee_age_l190_190910


namespace jane_win_probability_l190_190031

-- Define the conditions
def spinner_sectors : ℕ := 8

-- Define the winning condition
def jane_wins (jane_spin sister_spin : ℕ) : Prop := 
  (abs (jane_spin - sister_spin) < 4)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := spinner_sectors * spinner_sectors

-- Define the number of losing combinations
def losing_combinations : ℕ := 20

-- Calculate the probability that Jane wins
def probability_jane_wins : ℚ := 1 - (losing_combinations / total_outcomes: ℚ)

-- Prove the probability that Jane wins is 11/16
theorem jane_win_probability : probability_jane_wins = 11 / 16 := 
by sorry

end jane_win_probability_l190_190031


namespace intersection_complement_eq_l190_190758

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5 * x + 4 < 0}

theorem intersection_complement_eq :
  A ∩ {x | x ≤ 1 ∨ x ≥ 4} = {0, 1} := by
  sorry

end intersection_complement_eq_l190_190758


namespace arithmetic_square_root_of_9_l190_190529

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end arithmetic_square_root_of_9_l190_190529


namespace probability_of_point_in_sphere_l190_190444

noncomputable def probability_inside_sphere : ℝ :=
  let cube_volume := 4 ^ 3 in
  let sphere_volume := (4 / 3) * Real.pi * (2 ^ 3) in
  sphere_volume / cube_volume

theorem probability_of_point_in_sphere :
  ∀ (x y z : ℝ), 
    (-2 ≤ x ∧ x ≤ 2) ∧ 
    (-2 ≤ y ∧ y ≤ 2) ∧ 
    (-2 ≤ z ∧ z ≤ 2) → 
    (probability_inside_sphere = (Real.pi / 6)) := by
  sorry

end probability_of_point_in_sphere_l190_190444


namespace division_equivalent_l190_190685

def division_to_fraction (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 ∧ 0 ≤ a ∧ 0 ≤ b → a / b = (a * 1000) / (b * 1000) :=
by
  intros h
  field_simp
  
theorem division_equivalent (h : 0 ≤ 0.08 ∧ 0 ≤ 0.002 ∧ 0.08 ≠ 0 ∧ 0.002 ≠ 0) :
  0.08 / 0.002 = 40 :=
by
  have := division_to_fraction 0.08 0.002 h
  norm_num at this
  exact this

end division_equivalent_l190_190685


namespace tom_remaining_balloons_l190_190423

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end tom_remaining_balloons_l190_190423


namespace theta_range_l190_190883

noncomputable def f (x θ : ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_range (θ : ℝ) (k : ℤ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x θ > 0) →
  θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
sorry

end theta_range_l190_190883


namespace part_one_part_two_l190_190214

open Real

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

theorem part_one (h1 : ∀ x, f (x + π) = f x)
                 (h2 : ∀ x, f (x + π / 6) = f (-x + π / 6))
                 (h3 : f 0 = 1) :
  f = λ x, 2 * sin (2 * x + π / 6) :=
by sorry

theorem part_two (α β : ℝ)
                 (hαβ1 : α ∈ Ioc 0 (π / 4))
                 (hαβ2 : β ∈ Ioc 0 (π / 4))
                 (h4 : f (α - π / 3) = -10 / 13)
                 (h5 : f (β + π / 6) = 6 / 5) :
  cos (2 * α - 2 * β) = 63 / 65 :=
by sorry

end part_one_part_two_l190_190214


namespace total_heartbeats_during_race_l190_190441

-- Definitions for conditions
def heart_rate_per_minute : ℕ := 120
def pace_minutes_per_km : ℕ := 4
def race_distance_km : ℕ := 120

-- Lean statement of the proof problem
theorem total_heartbeats_during_race :
  120 * (4 * 120) = 57600 := by
  sorry

end total_heartbeats_during_race_l190_190441


namespace handshake_problem_l190_190625

theorem handshake_problem (n : ℕ) (H : (n * (n - 1)) / 2 = 28) : n = 8 := 
sorry

end handshake_problem_l190_190625


namespace investor_difference_l190_190453

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end investor_difference_l190_190453


namespace intersection_coords_perpendicular_line_l190_190200

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := x + y - 2 = 0

theorem intersection_coords : ∃ P : ℝ × ℝ, line1 P.1 P.2 ∧ line2 P.1 P.2 ∧ P = (1, 1) := by
  sorry

theorem perpendicular_line (x y : ℝ) (P : ℝ × ℝ) (hP: P = (1, 1)) : 
  (line2 P.1 P.2) → x - y = 0 := by
  sorry

end intersection_coords_perpendicular_line_l190_190200


namespace distance_from_origin_to_8_15_l190_190899

theorem distance_from_origin_to_8_15 : 
  let origin : ℝ × ℝ := (0, 0)
  let point : ℝ × ℝ := (8, 15)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := 
    real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  distance origin point = 17 :=
by 
  let origin := (0 : ℝ, 0 : ℝ)
  let point := (8 : ℝ, 15 : ℝ)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := 
    real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  sorry

end distance_from_origin_to_8_15_l190_190899


namespace ratio_of_ages_l190_190550

theorem ratio_of_ages (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : a + b = 35) : a / gcd a b = 3 ∧ b / gcd a b = 4 :=
by
  sorry

end ratio_of_ages_l190_190550


namespace part1_m_n_part2_k_l190_190612

-- Definitions of vectors a, b, and c
def veca : ℝ × ℝ := (3, 2)
def vecb : ℝ × ℝ := (-1, 2)
def vecc : ℝ × ℝ := (4, 1)

-- Part (1)
theorem part1_m_n : 
  ∃ (m n : ℝ), (-m + 4 * n = 3) ∧ (2 * m + n = 2) :=
sorry

-- Part (2)
theorem part2_k : 
  ∃ (k : ℝ), (3 + 4 * k) * 2 - (-5) * (2 + k) = 0 :=
sorry

end part1_m_n_part2_k_l190_190612


namespace A_independent_of_beta_l190_190657

noncomputable def A (alpha beta : ℝ) : ℝ :=
  (Real.sin (alpha + beta) ^ 2) + (Real.sin (beta - alpha) ^ 2) - 
  2 * (Real.sin (alpha + beta)) * (Real.sin (beta - alpha)) * (Real.cos (2 * alpha))

theorem A_independent_of_beta (alpha beta : ℝ) : 
  ∃ (c : ℝ), ∀ beta : ℝ, A alpha beta = c :=
by
  sorry

end A_independent_of_beta_l190_190657


namespace intersection_P_Q_l190_190609

open Set

noncomputable def P : Set ℝ := {1, 2, 3, 4}

noncomputable def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := 
by {
  sorry
}

end intersection_P_Q_l190_190609


namespace doris_weeks_to_meet_expenses_l190_190725

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l190_190725


namespace solution_of_system_l190_190418

theorem solution_of_system : ∃ x y : ℝ, (2 * x + y = 2) ∧ (x - y = 1) ∧ (x = 1) ∧ (y = 0) := 
by
  sorry

end solution_of_system_l190_190418


namespace factorization_identity_l190_190862

theorem factorization_identity (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 :=
by
  sorry

end factorization_identity_l190_190862


namespace quadratic_always_positive_l190_190804

theorem quadratic_always_positive (x : ℝ) : x^2 + x + 1 > 0 :=
sorry

end quadratic_always_positive_l190_190804


namespace functional_eq_one_l190_190590

theorem functional_eq_one (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
    (h2 : ∀ x > 0, ∀ y > 0, f x * f (y * f x) = f (x + y)) :
    ∀ x, 0 < x → f x = 1 := 
by
  sorry

end functional_eq_one_l190_190590


namespace find_x_in_interval_l190_190052

theorem find_x_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (h_eq : (2 - Real.sin (2 * x)) * Real.sin (x + π / 4) = 1) : x = π / 4 := 
sorry

end find_x_in_interval_l190_190052


namespace tylenol_intake_proof_l190_190787

noncomputable def calculate_tylenol_intake_grams
  (tablet_mg : ℕ) (tablets_per_dose : ℕ) (hours_per_dose : ℕ) (total_hours : ℕ) : ℕ :=
  let doses := total_hours / hours_per_dose
  let total_mg := doses * tablets_per_dose * tablet_mg
  total_mg / 1000

theorem tylenol_intake_proof : calculate_tylenol_intake_grams 500 2 4 12 = 3 :=
  by sorry

end tylenol_intake_proof_l190_190787


namespace minimum_value_range_of_a_l190_190068

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x a : ℝ) : ℝ := 3/2 - a/x
noncomputable def φ (x : ℝ) : ℝ := f x - g x 1

theorem minimum_value (x : ℝ) (h₀ : x ∈ Set.Ici (4:ℝ)) : 
  φ x ≥ 2 * Real.log 2 - 5 / 4 := 
sorry

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc (1/2:ℝ) 1, x^2 = 3/2 - a/x) ↔
  a ∈ Set.Icc (1/2:ℝ) (Real.sqrt 2 / 2) := 
sorry

end minimum_value_range_of_a_l190_190068


namespace raviraj_cycle_distance_l190_190260

theorem raviraj_cycle_distance :
  ∃ (d : ℝ), d = Real.sqrt ((425: ℝ)^2 + (200: ℝ)^2) ∧ d = 470 := 
by
  sorry

end raviraj_cycle_distance_l190_190260


namespace max_value_of_expression_l190_190915

theorem max_value_of_expression (A M C : ℕ) (hA : 0 < A) (hM : 0 < M) (hC : 0 < C) (hSum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A + A + M + C ≤ 215 :=
sorry

end max_value_of_expression_l190_190915


namespace appropriate_term_for_assessment_l190_190211

-- Definitions
def price : Type := String
def value : Type := String
def cost : Type := String
def expense : Type := String

-- Context for assessment of the project
def assessment_context : Type := Π (word : String), word ∈ ["price", "value", "cost", "expense"] → Prop

-- Main Lean statement
theorem appropriate_term_for_assessment (word : String) (h : word ∈ ["price", "value", "cost", "expense"]) :
  word = "value" :=
sorry

end appropriate_term_for_assessment_l190_190211


namespace lindsey_savings_in_october_l190_190921

-- Definitions based on conditions
def savings_september := 50
def savings_november := 11
def spending_video_game := 87
def final_amount_left := 36
def mom_gift := 25

-- The theorem statement
theorem lindsey_savings_in_october (X : ℕ) 
  (h1 : savings_september + X + savings_november > 75) 
  (total_savings := savings_september + X + savings_november + mom_gift) 
  (final_condition : total_savings - spending_video_game = final_amount_left) : 
  X = 37 :=
by
  sorry

end lindsey_savings_in_october_l190_190921


namespace negation_of_universal_proposition_l190_190410

theorem negation_of_universal_proposition :
  (∃ x : ℤ, x % 5 = 0 ∧ ¬ (x % 2 = 1)) ↔ ¬ (∀ x : ℤ, x % 5 = 0 → (x % 2 = 1)) :=
by sorry

end negation_of_universal_proposition_l190_190410


namespace max_distance_on_highway_l190_190977

-- Assume there are definitions for the context of this problem
def mpg_highway : ℝ := 12.2
def gallons : ℝ := 24
def max_distance (mpg : ℝ) (gal : ℝ) : ℝ := mpg * gal

theorem max_distance_on_highway :
  max_distance mpg_highway gallons = 292.8 :=
sorry

end max_distance_on_highway_l190_190977


namespace geometric_sequence_sum_l190_190385

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l190_190385


namespace find_constant_l190_190621

theorem find_constant (n : ℤ) (c : ℝ) (h1 : ∀ n ≤ 10, c * (n : ℝ)^2 ≤ 12100) : c ≤ 121 :=
sorry

end find_constant_l190_190621


namespace find_y_intercept_l190_190989

def line_y_intercept (m x y : ℝ) (pt : ℝ × ℝ) : ℝ :=
  let y_intercept := pt.snd - m * pt.fst
  y_intercept

theorem find_y_intercept (m x y b : ℝ) (pt : ℝ × ℝ) (h1 : m = 2) (h2 : pt = (498, 998)) :
  line_y_intercept m x y pt = 2 :=
by
  sorry

end find_y_intercept_l190_190989


namespace exists_distinct_nats_with_square_sums_l190_190582

theorem exists_distinct_nats_with_square_sums :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (x y : ℕ), a^2 + 2 * c * d + b^2 = x^2 ∧ c^2 + 2 * a * b + d^2 = y^2 :=
sorry

end exists_distinct_nats_with_square_sums_l190_190582


namespace transformed_sum_of_coordinates_l190_190882

theorem transformed_sum_of_coordinates (g : ℝ → ℝ) (h : g 8 = 5) :
  let x := 8 / 3
  let y := 14 / 9
  3 * y = g (3 * x) / 3 + 3 ∧ (x + y = 38 / 9) :=
by
  sorry

end transformed_sum_of_coordinates_l190_190882


namespace min_value_l190_190207

theorem min_value (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_sum : a + b = 1) : 
  ∃ x : ℝ, (x = 25) ∧ x ≤ (4 / a + 9 / b) :=
by
  sorry

end min_value_l190_190207


namespace shoe_size_combination_l190_190509

theorem shoe_size_combination (J A : ℕ) (hJ : J = 7) (hA : A = 2 * J) : J + A = 21 := by
  sorry

end shoe_size_combination_l190_190509


namespace real_solutions_l190_190186

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end real_solutions_l190_190186


namespace algebraic_expression_value_l190_190365

theorem algebraic_expression_value (x : ℝ) 
  (h : 2 * x^2 + 3 * x + 7 = 8) : 
  4 * x^2 + 6 * x - 9 = -7 := 
by 
  sorry

end algebraic_expression_value_l190_190365


namespace inequality_not_less_than_l190_190861

theorem inequality_not_less_than (y : ℝ) : 2 * y + 8 ≥ -3 := 
sorry

end inequality_not_less_than_l190_190861


namespace distance_light_travels_500_years_l190_190406

def distance_light_travels_one_year : ℝ := 5.87e12
def years : ℕ := 500

theorem distance_light_travels_500_years :
  distance_light_travels_one_year * years = 2.935e15 := 
sorry

end distance_light_travels_500_years_l190_190406


namespace add_in_base_7_l190_190048

def from_base (b : ℕ) (digits : List ℕ) : ℕ := 
  digits.reverse.enum_from 1 |>.map (λ (i, d), d * b^(i-1)).sum

def to_base (b : ℕ) (n : ℕ) : List ℕ :=
  if n = 0 then [0] else 
    List.unfold (λ x, if x = 0 then none else some (x%b, x / b)) n |>.reverse

theorem add_in_base_7 : 
  from_base 7 [6, 6, 6] + from_base 7 [6, 6] + from_base 7 [6] = from_base 7 [1, 4, 0, 0] :=
by 
  unfold from_base 
  have h1 : from_base 7 [6, 6, 6] = 6 * 7^2 + 6 * 7^1 + 6 * 7^0 := by rfl
  have h2 : from_base 7 [6, 6] = 6 * 7^1 + 6 * 7^0 := by rfl
  have h3 : from_base 7 [6] = 6 := by rfl
  have h4 : from_base 7 [1, 4, 0, 0] = 1 * 7^3 + 4 * 7^2 + 0 * 7^1 + 0 * 7^0 := by rfl
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end add_in_base_7_l190_190048


namespace jane_picked_fraction_l190_190991

-- Define the total number of tomatoes initially
def total_tomatoes : ℕ := 100

-- Define the number of tomatoes remaining at the end
def remaining_tomatoes : ℕ := 15

-- Define the number of tomatoes picked in the second week
def second_week_tomatoes : ℕ := 20

-- Define the number of tomatoes picked in the third week
def third_week_tomatoes : ℕ := 2 * second_week_tomatoes

theorem jane_picked_fraction :
  ∃ (f : ℚ), f = 1 / 4 ∧
    (f * total_tomatoes + second_week_tomatoes + third_week_tomatoes + remaining_tomatoes = total_tomatoes) :=
sorry

end jane_picked_fraction_l190_190991


namespace arccos_gt_arctan_l190_190046

theorem arccos_gt_arctan (x : ℝ) (h : -1 ≤ x ∧ x < 1/2) : Real.arccos x > Real.arctan x :=
sorry

end arccos_gt_arctan_l190_190046


namespace range_of_m_l190_190201

noncomputable def range_m (a b : ℝ) (m : ℝ) : Prop :=
  (3 * a + 4 / b = 1) ∧ a > 0 ∧ b > 0 → (1 / a + 3 * b > m)

theorem range_of_m (m : ℝ) : (∀ a b : ℝ, (range_m a b m)) ↔ m < 27 :=
by
  sorry

end range_of_m_l190_190201


namespace range_of_m_l190_190330

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x-3| + |x+4| ≥ |2*m-1|) ↔ (-3 ≤ m ∧ m ≤ 4) := by
  sorry

end range_of_m_l190_190330


namespace sin_double_angle_identity_l190_190202

theorem sin_double_angle_identity 
  (α : ℝ) 
  (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h₂ : Real.sin α = 1 / 5) : 
  Real.sin (2 * α) = - (4 * Real.sqrt 6) / 25 :=
by
  sorry

end sin_double_angle_identity_l190_190202


namespace mrs_li_actual_birthdays_l190_190653
   
   def is_leap_year (year : ℕ) : Prop :=
     (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
   
   def num_leap_years (start end_ : ℕ) : ℕ :=
     (start / 4 - start / 100 + start / 400) -
     (end_ / 4 - end_ / 100 + end_ / 400)
   
   theorem mrs_li_actual_birthdays : num_leap_years 1944 2011 = 16 :=
   by
     -- Calculation logic for the proof
     sorry
   
end mrs_li_actual_birthdays_l190_190653


namespace money_left_correct_l190_190433

-- Define the initial amount of money John had
def initial_money : ℝ := 10.50

-- Define the amount spent on sweets
def sweets_cost : ℝ := 2.25

-- Define the amount John gave to each friend
def gift_per_friend : ℝ := 2.20

-- Define the total number of friends
def number_of_friends : ℕ := 2

-- Calculate the total gifts given to friends
def total_gifts := gift_per_friend * (number_of_friends : ℝ)

-- Calculate the total amount spent
def total_spent := sweets_cost + total_gifts

-- Define the amount of money left
def money_left := initial_money - total_spent

-- The theorem statement
theorem money_left_correct : money_left = 3.85 := 
by 
  sorry

end money_left_correct_l190_190433


namespace employee_pay_l190_190000

variable (X Y Z : ℝ)

-- Conditions
def X_pay (Y : ℝ) := 1.2 * Y
def Z_pay (X : ℝ) := 0.75 * X

-- Proof statement
theorem employee_pay (h1 : X = X_pay Y) (h2 : Z = Z_pay X) (total_pay : X + Y + Z = 1540) : 
  X + Y + Z = 1540 :=
by
  sorry

end employee_pay_l190_190000


namespace complement_N_star_in_N_l190_190846

-- The set of natural numbers
def N : Set ℕ := { n | true }

-- The set of positive integers
def N_star : Set ℕ := { n | n > 0 }

-- The complement of N_star in N is the set {0}
theorem complement_N_star_in_N : { n | n ∈ N ∧ n ∉ N_star } = {0} := by
  sorry

end complement_N_star_in_N_l190_190846


namespace finitely_many_n_divisors_in_A_l190_190087

-- Lean 4 statement
theorem finitely_many_n_divisors_in_A (A : Finset ℕ) (a : ℕ) (hA : ∀ p ∈ A, Nat.Prime p) (ha : a ≥ 2) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ∃ p : ℕ, p ∣ a^n - 1 ∧ p ∉ A := by
  sorry

end finitely_many_n_divisors_in_A_l190_190087


namespace positive_number_sum_square_eq_210_l190_190278

theorem positive_number_sum_square_eq_210 (x : ℕ) (h1 : x^2 + x = 210) (h2 : 0 < x) (h3 : x < 15) : x = 14 :=
by
  sorry

end positive_number_sum_square_eq_210_l190_190278


namespace opposite_of_neg2023_l190_190672

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l190_190672


namespace remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l190_190476

-- Part (a): Remainder of (1989 * 1990 * 1991 + 1992^2) when divided by 7 is 0.
theorem remainder_of_product_and_square_is_zero_mod_7 :
  (1989 * 1990 * 1991 + 1992^2) % 7 = 0 :=
sorry

-- Part (b): Remainder of 9^100 when divided by 8 is 1.
theorem remainder_of_9_pow_100_mod_8 :
  9^100 % 8 = 1 :=
sorry

end remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l190_190476


namespace area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l190_190112

theorem area_of_inscribed_rectangle_not_square (s : ℝ) : 
  (s > 0) ∧ (s < 1 / 2) :=
sorry

theorem area_of_inscribed_rectangle_is_square (s : ℝ) : 
  (s >= 1 / 2) ∧ (s < 1) :=
sorry

end area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l190_190112


namespace jason_needs_201_grams_l190_190908

-- Define the conditions
def rectangular_patch_length : ℕ := 6
def rectangular_patch_width : ℕ := 7
def square_path_side_length : ℕ := 5
def sand_per_square_inch : ℕ := 3

-- Define the areas
def rectangular_patch_area : ℕ := rectangular_patch_length * rectangular_patch_width
def square_path_area : ℕ := square_path_side_length * square_path_side_length

-- Define the total area
def total_area : ℕ := rectangular_patch_area + square_path_area

-- Define the total sand needed
def total_sand_needed : ℕ := total_area * sand_per_square_inch

-- State the proof problem
theorem jason_needs_201_grams : total_sand_needed = 201 := by
    sorry

end jason_needs_201_grams_l190_190908


namespace pq_combined_work_rate_10_days_l190_190694

/-- Conditions: 
1. wr_p = wr_qr, where wr_qr is the combined work rate of q and r
2. wr_r allows completing the work in 30 days
3. wr_q allows completing the work in 30 days

We need to prove that the combined work rate of p and q allows them to complete the work in 10 days.
-/
theorem pq_combined_work_rate_10_days
  (wr_p wr_q wr_r wr_qr : ℝ)
  (h1 : wr_p = wr_qr)
  (h2 : wr_r = 1/30)
  (h3 : wr_q = 1/30) :
  wr_p + wr_q = 1/10 := by
  sorry

end pq_combined_work_rate_10_days_l190_190694


namespace quadruple_pieces_count_l190_190652

theorem quadruple_pieces_count (earned_amount_per_person_in_dollars : ℕ) 
    (total_single_pieces : ℕ) (total_double_pieces : ℕ)
    (total_triple_pieces : ℕ) (single_piece_circles : ℕ) 
    (double_piece_circles : ℕ) (triple_piece_circles : ℕ)
    (quadruple_piece_circles : ℕ) (cents_per_dollar : ℕ) :
    earned_amount_per_person_in_dollars * 2 * cents_per_dollar -
    (total_single_pieces * single_piece_circles + 
    total_double_pieces * double_piece_circles + 
    total_triple_pieces * triple_piece_circles) = 
    165 * quadruple_piece_circles :=
        sorry

#eval quadruple_pieces_count 5 100 45 50 1 2 3 4 100

end quadruple_pieces_count_l190_190652


namespace bob_remaining_corns_l190_190572

theorem bob_remaining_corns (total_bushels : ℕ) (terry_bushels : ℕ) (jerry_bushels : ℕ)
                            (linda_bushels: ℕ) (stacy_ears: ℕ) (ears_per_bushel: ℕ):
                            total_bushels = 50 → terry_bushels = 8 → jerry_bushels = 3 →
                            linda_bushels = 12 → stacy_ears = 21 → ears_per_bushel = 14 →
                            (total_bushels - (terry_bushels + jerry_bushels + linda_bushels + stacy_ears / ears_per_bushel)) * ears_per_bushel = 357 :=
by intros total_cond terry_cond jerry_cond linda_cond stacy_cond ears_cond
   rw [total_cond, terry_cond, jerry_cond, linda_cond, stacy_cond, ears_cond]
   norm_cast
   have : 21 / 14 = (3 / 2 : ℕ) := sorry
   rw this
   linarith
   sorry

end bob_remaining_corns_l190_190572


namespace conclusion1_conclusion2_conclusion3_l190_190358

-- Define the Δ operation
def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

-- 1. Proof that (-2^2) Δ 4 = 0
theorem conclusion1 : delta (-4) 4 = 0 := sorry

-- 2. Proof that (1/3) Δ (1/4) = 3 Δ 4
theorem conclusion2 : delta (1/3) (1/4) = delta 3 4 := sorry

-- 3. Proof that (-m) Δ n = m Δ (-n)
theorem conclusion3 (m n : ℚ) : delta (-m) n = delta m (-n) := sorry

end conclusion1_conclusion2_conclusion3_l190_190358


namespace total_amount_shared_l190_190690

theorem total_amount_shared (a b c : ℝ)
  (h1 : a = 1/3 * (b + c))
  (h2 : b = 2/7 * (a + c))
  (h3 : a = b + 20) : 
  a + b + c = 720 :=
by
  sorry

end total_amount_shared_l190_190690


namespace probability_Ephraim_same_heads_Keiko_l190_190779

theorem probability_Ephraim_same_heads_Keiko :
  let outcomes_Keiko := {0, 1}
  let outcomes_Ephraim := {0, 1, 2}
  let favorable_outcomes := (1, 1) :: (0, 0) :: (1, 2) :: []
  let total_outcomes := [(h1, h2) | h1 ∈ outcomes_Keiko, h2 ∈ outcomes_Ephraim ]
  (favorable_outcomes.length / total_outcomes.length) = 3/8 :=
by
  sorry

end probability_Ephraim_same_heads_Keiko_l190_190779


namespace number_of_ways_to_express_n_as_sum_l190_190785

noncomputable def P (n k : ℕ) : ℕ := sorry
noncomputable def Q (n k : ℕ) : ℕ := sorry

theorem number_of_ways_to_express_n_as_sum (n : ℕ) (k : ℕ) (h : k ≥ 2) : P n k = Q n k := sorry

end number_of_ways_to_express_n_as_sum_l190_190785


namespace fewerEmployeesAbroadThanInKorea_l190_190024

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end fewerEmployeesAbroadThanInKorea_l190_190024


namespace least_integer_value_l190_190133

theorem least_integer_value (x : ℤ) :
  (|3 * x + 4| ≤ 25) → ∃ y : ℤ, x = y ∧ y = -9 :=
by
  sorry

end least_integer_value_l190_190133


namespace seq_nth_term_2009_l190_190115

theorem seq_nth_term_2009 (n x : ℤ) (h : 2 * x - 3 = 5 ∧ 5 * x - 11 = 9 ∧ 3 * x + 1 = 13) :
  n = 502 ↔ 2009 = (2 * x - 3) + (n - 1) * ((5 * x - 11) - (2 * x - 3)) :=
sorry

end seq_nth_term_2009_l190_190115


namespace find_x_l190_190540

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : (∀ (a b c d : ℝ), balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 := 
by
  sorry

end find_x_l190_190540


namespace cos_seven_theta_l190_190618

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (7 * θ) = -83728 / 390625 := 
sorry

end cos_seven_theta_l190_190618


namespace solve_rational_equation_l190_190402

theorem solve_rational_equation : 
  ∀ x : ℝ, x ≠ 1 -> (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) → 
  (x = 6 ∨ x = -2) :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solve_rational_equation_l190_190402


namespace each_person_bids_five_times_l190_190311

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l190_190311


namespace average_age_of_three_l190_190519

theorem average_age_of_three (Kimiko_age : ℕ) (Omi_age : ℕ) (Arlette_age : ℕ) 
  (h1 : Omi_age = 2 * Kimiko_age) 
  (h2 : Arlette_age = (3 * Kimiko_age) / 4) 
  (h3 : Kimiko_age = 28) : 
  (Kimiko_age + Omi_age + Arlette_age) / 3 = 35 := 
  by sorry

end average_age_of_three_l190_190519


namespace find_C_value_l190_190238

theorem find_C_value (A B C : ℕ) 
  (cond1 : A + B + C = 10) 
  (cond2 : B + A = 9)
  (cond3 : A + 1 = 3) :
  C = 1 :=
by
  sorry

end find_C_value_l190_190238


namespace hours_worked_on_saturday_l190_190101

-- Definitions from the problem conditions
def hourly_wage : ℝ := 15
def hours_friday : ℝ := 10
def hours_sunday : ℝ := 14
def total_earnings : ℝ := 450

-- Define number of hours worked on Saturday as a variable
variable (hours_saturday : ℝ)

-- Total earnings can be expressed as the sum of individual day earnings
def total_earnings_eq : Prop := 
  total_earnings = (hours_friday * hourly_wage) + (hours_sunday * hourly_wage) + (hours_saturday * hourly_wage)

-- Prove that the hours worked on Saturday is 6
theorem hours_worked_on_saturday :
  total_earnings_eq hours_saturday →
  hours_saturday = 6 := by
  sorry

end hours_worked_on_saturday_l190_190101


namespace find_remainder_proof_l190_190836

def div_remainder_problem :=
  let number := 220050
  let sum := 555 + 445
  let difference := 555 - 445
  let quotient := 2 * difference
  let divisor := sum
  let quotient_correct := quotient = 220
  let division_formula := number = divisor * quotient + 50
  quotient_correct ∧ division_formula

theorem find_remainder_proof : div_remainder_problem := by
  sorry

end find_remainder_proof_l190_190836


namespace total_travel_time_l190_190008

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end total_travel_time_l190_190008


namespace sqrt_of_0_09_l190_190811

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end sqrt_of_0_09_l190_190811


namespace arithmetic_sequence_count_l190_190613

theorem arithmetic_sequence_count :
  ∃ n : ℕ, 2 + (n-1) * 5 = 2507 ∧ n = 502 :=
by
  sorry

end arithmetic_sequence_count_l190_190613


namespace faster_train_passes_slower_l190_190011

theorem faster_train_passes_slower (v_fast v_slow : ℝ) (length_fast : ℝ) 
  (hv_fast : v_fast = 50) (hv_slow : v_slow = 32) (hl_length_fast : length_fast = 75) :
  ∃ t : ℝ, t = 15 := 
by
  sorry

end faster_train_passes_slower_l190_190011


namespace log_x3y2_eq_2_l190_190354

theorem log_x3y2_eq_2 (x y : ℝ) (h1 : log (x^2 * y^5) = 2) (h2 : log (x^3 * y^2) = 2) :
  log (x^3 * y^2) = 2 :=
sorry

end log_x3y2_eq_2_l190_190354


namespace scientific_notation_eight_million_l190_190630

theorem scientific_notation_eight_million :
  ∃ a n, 8000000 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = 6 :=
by
  use 8
  use 6
  sorry

end scientific_notation_eight_million_l190_190630


namespace number_of_perfect_squares_and_cubes_l190_190222

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l190_190222


namespace inequality_add_six_l190_190074

theorem inequality_add_six (x y : ℝ) (h : x < y) : x + 6 < y + 6 :=
sorry

end inequality_add_six_l190_190074


namespace solve_for_x_l190_190104

theorem solve_for_x (x : ℝ) (h : 1 / 3 + 1 / x = 2 / 3) : x = 3 :=
sorry

end solve_for_x_l190_190104


namespace sampling_method_systematic_l190_190026

theorem sampling_method_systematic 
  (inspect_interval : ℕ := 10)
  (products_interval : ℕ := 10)
  (position : ℕ) :
  inspect_interval = 10 ∧ products_interval = 10 → 
  (sampling_method = "Systematic Sampling") :=
by
  sorry

end sampling_method_systematic_l190_190026


namespace hershey_kisses_to_kitkats_ratio_l190_190574

-- Definitions based on the conditions
def kitkats : ℕ := 5
def nerds : ℕ := 8
def lollipops : ℕ := 11
def baby_ruths : ℕ := 10
def reeses : ℕ := baby_ruths / 2
def candy_total_before : ℕ := kitkats + nerds + lollipops + baby_ruths + reeses
def candy_remaining : ℕ := 49
def lollipops_given : ℕ := 5
def total_candy_before : ℕ := candy_remaining + lollipops_given
def hershey_kisses : ℕ := total_candy_before - candy_total_before

-- Theorem to prove the desired ratio
theorem hershey_kisses_to_kitkats_ratio : hershey_kisses / kitkats = 3 := by
  sorry

end hershey_kisses_to_kitkats_ratio_l190_190574


namespace max_ab_l190_190369

theorem max_ab (a b c : ℝ) (h1 : 3 * a + b = 1) (h2 : 0 ≤ a) (h3 : a < 1) (h4 : 0 ≤ b) 
(h5 : b < 1) (h6 : 0 ≤ c) (h7 : c < 1) (h8 : a + b + c = 1) : 
  ab ≤ 1 / 12 := by
  sorry

end max_ab_l190_190369


namespace cone_lateral_surface_area_l190_190025

theorem cone_lateral_surface_area (r h : ℝ) (h_r : r = 3) (h_h : h = 4) : 
  (1/2) * (2 * Real.pi * r) * (Real.sqrt (r ^ 2 + h ^ 2)) = 15 * Real.pi := 
by
  sorry

end cone_lateral_surface_area_l190_190025


namespace jennie_total_rental_cost_l190_190665

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end jennie_total_rental_cost_l190_190665


namespace Anne_cleaning_time_l190_190456

theorem Anne_cleaning_time (B A C : ℚ) 
  (h1 : B + A + C = 1 / 6) 
  (h2 : B + 2 * A + 3 * C = 1 / 2)
  (h3 : B + A = 1 / 4)
  (h4 : B + C = 1 / 3) : 
  A = 1 / 6 := 
sorry

end Anne_cleaning_time_l190_190456


namespace remainder_of_n_l190_190054

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
by
  sorry

end remainder_of_n_l190_190054


namespace age_difference_l190_190980

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l190_190980


namespace fraction_value_l190_190495

variable (x y : ℝ)

theorem fraction_value (hx : x = 4) (hy : y = -3) : (x - 2 * y) / (x + y) = 10 := by
  sorry

end fraction_value_l190_190495


namespace sin_alpha_value_l190_190600

theorem sin_alpha_value (α : ℝ) (h1 : Real.sin (α + π / 4) = 4 / 5) (h2 : α ∈ Set.Ioo (π / 4) (3 * π / 4)) :
  Real.sin α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end sin_alpha_value_l190_190600


namespace smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l190_190562

/-- Define what it means for a number to be a prime greater than 3 -/
def is_prime_gt_3 (n : ℕ) : Prop :=
  Prime n ∧ 3 < n

/-- Define a scalene triangle with side lengths that are distinct primes greater than 3 -/
def is_scalene_triangle_with_distinct_primes (a b c : ℕ) : Prop :=
  is_prime_gt_3 a ∧ is_prime_gt_3 b ∧ is_prime_gt_3 c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The proof problem statement -/
theorem smallest_possible_perimeter_of_scalene_triangle_with_prime_sides :
  ∃ (a b c : ℕ), is_scalene_triangle_with_distinct_primes a b c ∧ Prime (a + b + c) ∧ (a + b + c = 23) :=
sorry

end smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l190_190562


namespace presidency_meeting_arrangements_l190_190150

theorem presidency_meeting_arrangements :
  (∃ (schools : Fin 4), 
    ∃ (host_representatives : Fin 5 → ℕ), 
    ∃ (other_representatives : Fin 3 → Fin 5 → ℕ),
    (∀ i, host_representatives i ∈ {0, 1, 2, 3}) ∧
    (∀ j k, other_representatives j k ∈ {0, 1})) →
  ∃ (ways_to_arrange_meeting : ℕ), ways_to_arrange_meeting = 5000 :=
by
  sorry

end presidency_meeting_arrangements_l190_190150


namespace max_non_real_roots_l190_190096

theorem max_non_real_roots (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (A B : ℕ → ℕ) (h_turns : ∀ i < 3 * n, A i + B i = 1),
    (∀ i, (A i + B (i + 1)) % 3 = 0) →
    ∃ k, ∀ m, ∃ j < n, j % 2 = 1 → j + m * 2 ≤ 2 * k + j - m)
  → (∃ k, k = (n + 1) / 2) :=
sorry

end max_non_real_roots_l190_190096


namespace intersection_M_N_l190_190644

def M := { x : ℝ | |x| ≤ 1 }
def N := { x : ℝ | x^2 - x < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l190_190644


namespace investor_difference_l190_190454

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end investor_difference_l190_190454


namespace horner_rule_example_l190_190280

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_example : f 2 = 62 := by
  sorry

end horner_rule_example_l190_190280


namespace min_hours_to_pass_message_ge_55_l190_190443

theorem min_hours_to_pass_message_ge_55 : 
  ∃ (n: ℕ), (∀ m: ℕ, m < n → 2^(m+1) - 2 ≤ 55) ∧ 2^(n+1) - 2 > 55 :=
by sorry

end min_hours_to_pass_message_ge_55_l190_190443


namespace correct_sum_rounded_l190_190256

-- Define the conditions: sum and rounding
def sum_58_46 : ℕ := 58 + 46
def round_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 >= 50 then ((n / 100) + 1) * 100 else (n / 100) * 100

-- state the theorem
theorem correct_sum_rounded :
  round_to_nearest_hundred sum_58_46 = 100 :=
by
  sorry

end correct_sum_rounded_l190_190256


namespace hyperbola_eccentricity_asymptotic_lines_l190_190210

-- Define the conditions and the proof goal:

theorem hyperbola_eccentricity_asymptotic_lines {a b c e : ℝ} 
  (h_asym : ∀ x y : ℝ, (y = x ∨ y = -x) ↔ (a = b)) 
  (h_c : c = Real.sqrt (a ^ 2 + b ^ 2))
  (h_e : e = c / a) : e = Real.sqrt 2 := sorry

end hyperbola_eccentricity_asymptotic_lines_l190_190210


namespace perimeter_of_garden_l190_190307

-- Definitions based on conditions
def length : ℕ := 150
def breadth : ℕ := 150
def is_square (l b : ℕ) := l = b

-- Theorem statement proving the perimeter given conditions
theorem perimeter_of_garden : is_square length breadth → 4 * length = 600 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end perimeter_of_garden_l190_190307


namespace bumper_car_line_total_in_both_lines_l190_190570

theorem bumper_car_line (x y Z : ℕ) (hZ : Z = 25 - x + y) : Z = 25 - x + y :=
by
  sorry

theorem total_in_both_lines (x y Z : ℕ) (hZ : Z = 25 - x + y) : 40 - x + y = Z + 15 :=
by
  sorry

end bumper_car_line_total_in_both_lines_l190_190570


namespace parallel_vectors_l190_190886

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (-2, k)

theorem parallel_vectors (k : ℝ) (h : (1, 4) = c k) : k = -8 :=
sorry

end parallel_vectors_l190_190886


namespace parametric_equations_solution_l190_190336

theorem parametric_equations_solution (t₁ t₂ : ℝ) : 
  (1 = 1 + 2 * t₁ ∧ 2 = 2 - 3 * t₁) ∧
  (-1 = 1 + 2 * t₂ ∧ 5 = 2 - 3 * t₂) ↔
  (t₁ = 0 ∧ t₂ = -1) :=
by
  sorry

end parametric_equations_solution_l190_190336


namespace c_completes_in_three_days_l190_190023

variables (r_A r_B r_C : ℝ)
variables (h1 : r_A + r_B = 1/3)
variables (h2 : r_B + r_C = 1/3)
variables (h3 : r_A + r_C = 2/3)

theorem c_completes_in_three_days : 1 / r_C = 3 :=
by sorry

end c_completes_in_three_days_l190_190023


namespace math_students_but_not_science_l190_190901

theorem math_students_but_not_science (total_students : ℕ) (students_math : ℕ) (students_science : ℕ)
  (students_both : ℕ) (students_math_three_times : ℕ) :
  total_students = 30 ∧ students_both = 2 ∧ students_math = 3 * students_science ∧ 
  students_math = students_both + (22 - 2) → (students_math - students_both = 20) :=
by
  sorry

end math_students_but_not_science_l190_190901


namespace rectangle_perimeter_l190_190029

theorem rectangle_perimeter
  (w l P : ℝ)
  (h₁ : l = 2 * w)
  (h₂ : l * w = 400) :
  P = 60 * Real.sqrt 2 :=
by
  sorry

end rectangle_perimeter_l190_190029


namespace revenue_from_full_price_tickets_l190_190299

-- Definitions of the conditions
def total_tickets (f h : ℕ) : Prop := f + h = 180
def total_revenue (f h p : ℕ) : Prop := f * p + h * (p / 2) = 2750

-- Theorem statement
theorem revenue_from_full_price_tickets (f h p : ℕ) 
  (h_total_tickets : total_tickets f h) 
  (h_total_revenue : total_revenue f h p) : 
  f * p = 1000 :=
  sorry

end revenue_from_full_price_tickets_l190_190299


namespace cost_of_basic_calculator_l190_190306

variable (B S G : ℕ)

theorem cost_of_basic_calculator 
  (h₁ : S = 2 * B)
  (h₂ : G = 3 * S)
  (h₃ : B + S + G = 72) : 
  B = 8 :=
by
  sorry

end cost_of_basic_calculator_l190_190306


namespace find_percentage_l190_190764

theorem find_percentage : 
  ∀ (P : ℕ), 
  (50 - 47 = (P / 100) * 15) →
  P = 20 := 
by
  intro P h
  sorry

end find_percentage_l190_190764


namespace Carolyn_wants_to_embroider_l190_190038

theorem Carolyn_wants_to_embroider (s : ℕ) (f : ℕ) (u : ℕ) (g : ℕ) (n_f : ℕ) (t : ℕ) (number_of_unicorns : ℕ) :
  s = 4 ∧ f = 60 ∧ u = 180 ∧ g = 800 ∧ n_f = 50 ∧ t = 1085 ∧ 
  (t * s - (n_f * f) - g) / u = number_of_unicorns ↔ number_of_unicorns = 3 :=
by 
  sorry

end Carolyn_wants_to_embroider_l190_190038


namespace andrew_total_travel_time_l190_190006

theorem andrew_total_travel_time :
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  subway_time + train_time + bike_time = 38 :=
by
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  sorry

end andrew_total_travel_time_l190_190006


namespace running_hours_per_week_l190_190003

theorem running_hours_per_week 
  (initial_days : ℕ) (additional_days : ℕ) (morning_run_time : ℕ) (evening_run_time : ℕ)
  (total_days : ℕ) (total_run_time_per_day : ℕ) (total_run_time_per_week : ℕ)
  (H1 : initial_days = 3)
  (H2 : additional_days = 2)
  (H3 : morning_run_time = 1)
  (H4 : evening_run_time = 1)
  (H5 : total_days = initial_days + additional_days)
  (H6 : total_run_time_per_day = morning_run_time + evening_run_time)
  (H7 : total_run_time_per_week = total_days * total_run_time_per_day) :
  total_run_time_per_week = 10 := 
sorry

end running_hours_per_week_l190_190003


namespace find_c_l190_190514

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end find_c_l190_190514


namespace annual_interest_rate_l190_190999

theorem annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) 
  (hP : P = 5000) 
  (hA : A = 5202) 
  (hn : n = 4) 
  (ht : t = 1 / 2)
  (compound_interest : A = P * (1 + r / n)^ (n * t)) : 
  r = 0.080392 :=
by
  sorry

end annual_interest_rate_l190_190999


namespace maximize_profit_correct_l190_190437

noncomputable def maximize_profit : ℝ × ℝ :=
  let initial_selling_price : ℝ := 50
  let purchase_price : ℝ := 40
  let initial_sales_volume : ℝ := 500
  let sales_volume_decrease_rate : ℝ := 10
  let x := 20
  let optimal_selling_price := initial_selling_price + x
  let maximum_profit := -10 * x^2 + 400 * x + 5000
  (optimal_selling_price, maximum_profit)

theorem maximize_profit_correct :
  maximize_profit = (70, 9000) :=
  sorry

end maximize_profit_correct_l190_190437


namespace hall_area_l190_190536

theorem hall_area 
  (L W : ℝ)
  (h1 : W = 1/2 * L)
  (h2 : L - W = 10) : 
  L * W = 200 := 
sorry

end hall_area_l190_190536


namespace solution_set_of_inequality_l190_190679

theorem solution_set_of_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_inequality_l190_190679


namespace B_pow_99_identity_l190_190637

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_99_identity : (B ^ 99) = 1 := by
  sorry

end B_pow_99_identity_l190_190637


namespace chips_in_bag_l190_190922

theorem chips_in_bag :
  let initial_chips := 5
  let additional_chips := 5
  let daily_chips := 10
  let total_days := 10
  let first_day_chips := initial_chips + additional_chips
  let remaining_days := total_days - 1
  (first_day_chips + remaining_days * daily_chips) = 100 :=
by
  sorry

end chips_in_bag_l190_190922


namespace probability_all_students_get_their_own_lunch_l190_190448

def n : ℕ := 4

theorem probability_all_students_get_their_own_lunch :
  let total_possibilities := 4!,
      favorable_outcomes := 1 in
    (favorable_outcomes : ℚ) / total_possibilities = 1 / 24 :=
by {
  sorry
}

end probability_all_students_get_their_own_lunch_l190_190448


namespace turnip_difference_l190_190095

theorem turnip_difference :
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  (melanie_turnips + benny_turnips) - caroline_turnips = 80 :=
by
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  show (melanie_turnips + benny_turnips) - caroline_turnips = 80
  sorry

end turnip_difference_l190_190095


namespace sum_of_roots_l190_190807

theorem sum_of_roots (r p q : ℝ) 
  (h1 : (3 : ℝ) * r ^ 3 - (9 : ℝ) * r ^ 2 - (48 : ℝ) * r - (12 : ℝ) = 0)
  (h2 : (3 : ℝ) * p ^ 3 - (9 : ℝ) * p ^ 2 - (48 : ℝ) * p - (12 : ℝ) = 0)
  (h3 : (3 : ℝ) * q ^ 3 - (9 : ℝ) * q ^ 2 - (48 : ℝ) * q - (12 : ℝ) = 0)
  (roots_distinct : r ≠ p ∧ r ≠ q ∧ p ≠ q) :
  r + p + q = 3 := 
sorry

end sum_of_roots_l190_190807


namespace circle_tangent_to_parabola_passing_point_center_coordinates_l190_190834

theorem circle_tangent_to_parabola_passing_point_center_coordinates :
    ∃ (a b : ℝ),
    let center := (a, b) in
    ∀ (x y : ℝ), 
    (y = x^2 → ∃ r, 
    (x - 3)^2 + (y - 9)^2 = r^2 ∧ 
    (3 - a)^2 + (9 - b)^2 = r^2 ∧ 
    (a, b) = (-141/11, 128/11) ∧ 
    ((0 - a)^2 + (2 - b)^2 = r^2 ∧ r > 0)) := sorry

end circle_tangent_to_parabola_passing_point_center_coordinates_l190_190834


namespace certain_amount_of_seconds_l190_190830

theorem certain_amount_of_seconds (X : ℕ)
    (cond1 : 12 / X = 16 / 480) :
    X = 360 :=
by
  sorry

end certain_amount_of_seconds_l190_190830


namespace evaluate_complex_fraction_l190_190171

theorem evaluate_complex_fraction : 
  (1 / (2 + (1 / (3 + 1 / 4)))) = (13 / 30) :=
by
  sorry

end evaluate_complex_fraction_l190_190171


namespace correct_choice_is_C_l190_190165

def first_quadrant_positive_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def right_angle_is_axial (θ : ℝ) : Prop :=
  θ = 90

def obtuse_angle_second_quadrant (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

def terminal_side_initial_side_same (θ : ℝ) : Prop :=
  θ = 0 ∨ θ = 360

theorem correct_choice_is_C : obtuse_angle_second_quadrant 120 :=
by
  sorry

end correct_choice_is_C_l190_190165


namespace barney_extra_weight_l190_190169

-- Define the weight of a regular dinosaur
def regular_dinosaur_weight : ℕ := 800

-- Define the combined weight of five regular dinosaurs
def five_regular_dinosaurs_weight : ℕ := 5 * regular_dinosaur_weight

-- Define the total weight of Barney and the five regular dinosaurs together
def total_combined_weight : ℕ := 9500

-- Define the weight of Barney
def barney_weight : ℕ := total_combined_weight - five_regular_dinosaurs_weight

-- The proof statement
theorem barney_extra_weight : barney_weight - five_regular_dinosaurs_weight = 1500 :=
by sorry

end barney_extra_weight_l190_190169


namespace smallest_angle_CBD_l190_190917

-- Definitions for given conditions
def angle_ABC : ℝ := 40
def angle_ABD : ℝ := 15

-- Theorem statement
theorem smallest_angle_CBD : ∃ (angle_CBD : ℝ), angle_CBD = angle_ABC - angle_ABD := by
  use 25
  sorry

end smallest_angle_CBD_l190_190917


namespace cylinder_volume_rotation_l190_190229

theorem cylinder_volume_rotation (length width : ℝ) (π : ℝ) (h : length = 4) (w : width = 2) (V : ℝ) :
  (V = π * (4^2) * width ∨ V = π * (2^2) * length) :=
by
  sorry

end cylinder_volume_rotation_l190_190229


namespace arithmetic_square_root_of_nine_l190_190531

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l190_190531


namespace expression_evaluation_l190_190292

theorem expression_evaluation :
  (0.8 ^ 3) - ((0.5 ^ 3) / (0.8 ^ 2)) + 0.40 + (0.5 ^ 2) = 0.9666875 := 
by 
  sorry

end expression_evaluation_l190_190292


namespace determine_ordered_pair_l190_190578

theorem determine_ordered_pair (s n : ℤ)
    (h1 : ∀ t : ℤ, ∃ x y : ℤ,
        (x, y) = (s + 2 * t, -3 + n * t)) 
    (h2 : ∀ x y : ℤ, y = 2 * x - 7) :
    (s, n) = (2, 4) :=
by
  sorry

end determine_ordered_pair_l190_190578


namespace last_bead_color_is_blue_l190_190078

def bead_color_cycle := ["Red", "Orange", "Yellow", "Yellow", "Green", "Blue", "Purple"]

def bead_color (n : Nat) : String :=
  bead_color_cycle.get! (n % bead_color_cycle.length)

theorem last_bead_color_is_blue :
  bead_color 82 = "Blue" := 
by
  sorry

end last_bead_color_is_blue_l190_190078


namespace solution_to_system_of_equations_l190_190360

def augmented_matrix_system_solution (x y : ℝ) : Prop :=
  (x + 3 * y = 5) ∧ (2 * x + 4 * y = 6)

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), augmented_matrix_system_solution x y ∧ x = -1 ∧ y = 2 :=
by {
  sorry
}

end solution_to_system_of_equations_l190_190360


namespace max_value_A_l190_190875

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem max_value_A (A : ℝ) (hA : A = Real.pi / 6) : 
  ∀ x : ℝ, f x ≤ f A :=
sorry

end max_value_A_l190_190875


namespace M_intersect_N_eq_l190_190877

def M : Set ℝ := { y | ∃ x, y = x ^ 2 }
def N : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 2 + y^2 ≤ 1) }

theorem M_intersect_N_eq : M ∩ { y | (y ∈ Set.univ) } = { y | 0 ≤ y ∧ y ≤ Real.sqrt 2 } :=
by
  sorry

end M_intersect_N_eq_l190_190877


namespace perfect_squares_and_cubes_count_lt_1000_l190_190224

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l190_190224


namespace union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l190_190880

open Set

noncomputable def A := {x : ℝ | -2 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | (m - 2) ≤ x ∧ x ≤ (2 * m + 1)}

-- Part (1):
theorem union_when_m_is_one :
  A ∪ B 1 = {x : ℝ | -2 < x ∧ x ≤ 3} := sorry

-- Part (2):
theorem range_of_m_condition_1 :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ∈ Iic (-3/2) ∪ Ici 4 := sorry

theorem range_of_m_condition_2 :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Iio (-3) ∪ Ioo 0 (1/2) := sorry

end union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l190_190880


namespace height_of_pyramid_l190_190700

-- Define the volumes
def volume_cube (s : ℕ) : ℕ := s^3
def volume_pyramid (b : ℕ) (h : ℕ) : ℕ := (b^2 * h) / 3

-- Given constants
def s := 6
def b := 12

-- Given volume equality
def volumes_equal (s : ℕ) (b : ℕ) (h : ℕ) : Prop :=
  volume_cube s = volume_pyramid b h

-- The statement to prove
theorem height_of_pyramid (h : ℕ) (h_eq : volumes_equal s b h) :
  h = 9 := sorry

end height_of_pyramid_l190_190700


namespace probability_opposite_rooms_l190_190565

theorem probability_opposite_rooms :
  let rooms := {301, 302, 303, 304, 305, 306}
  ∃ A B : ∀ x ∈ rooms, bool,
  let roomPairs := [(301, 302), (303, 304), (305, 306)],
      totalWays := fintype.card (equiv.perm (fin 6)),
      favorableWays := 3 * (fintype.card (equiv.perm (fin 4)) * 2),
      prob := favorableWays / totalWays
  in prob = 1/5 :=
begin
  sorry
end

end probability_opposite_rooms_l190_190565


namespace equality_of_x_and_y_l190_190345

theorem equality_of_x_and_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x^(y^x) = y^(x^y)) : x = y :=
sorry

end equality_of_x_and_y_l190_190345


namespace contrapositive_example_l190_190752

variable (a b : ℝ)

theorem contrapositive_example
  (h₁ : a > 0)
  (h₃ : a + b < 0) :
  b < 0 := 
sorry

end contrapositive_example_l190_190752


namespace tangent_line_of_circle_l190_190375
-- Import the required libraries

-- Define the given condition of the circle in polar coordinates
def polar_circle (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

-- Define the property of the tangent line in polar coordinates
def tangent_line (rho theta : ℝ) : Prop :=
  rho * Real.cos theta = 4

-- State the theorem to be proven
theorem tangent_line_of_circle (rho theta : ℝ) (h : polar_circle rho theta) :
  tangent_line rho theta :=
sorry

end tangent_line_of_circle_l190_190375


namespace number_of_students_who_went_to_church_l190_190816

-- Define the number of chairs and the number of students.
variables (C S : ℕ)

-- Define the first condition: 9 students per chair with one student left.
def condition1 := S = 9 * C + 1

-- Define the second condition: 10 students per chair with one chair vacant.
def condition2 := S = 10 * C - 10

-- The theorem to be proved.
theorem number_of_students_who_went_to_church (h1 : condition1 C S) (h2 : condition2 C S) : S = 100 :=
by
  -- Proof goes here
  sorry

end number_of_students_who_went_to_church_l190_190816


namespace flour_in_cupboard_l190_190636

theorem flour_in_cupboard :
  let flour_on_counter := 100
  let flour_in_pantry := 100
  let flour_per_loaf := 200
  let loaves := 2
  let total_flour_needed := loaves * flour_per_loaf
  let flour_outside_cupboard := flour_on_counter + flour_in_pantry
  let flour_in_cupboard := total_flour_needed - flour_outside_cupboard
  flour_in_cupboard = 200 :=
by
  sorry

end flour_in_cupboard_l190_190636


namespace solution_inequalities_l190_190543

theorem solution_inequalities (x : ℝ) :
  (x^2 - 12 * x + 32 > 0) ∧ (x^2 - 13 * x + 22 < 0) → 2 < x ∧ x < 4 :=
by
  intro h
  sorry

end solution_inequalities_l190_190543


namespace team_team_count_correct_l190_190331

/-- Number of ways to select a team of three students from 20,
    one for each subject: math, Russian language, and informatics. -/
def ways_to_form_team (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

theorem team_team_count_correct : ways_to_form_team 20 = 6840 :=
by sorry

end team_team_count_correct_l190_190331


namespace prob_at_least_two_correct_l190_190777

-- Probability of guessing a question correctly
def prob_correct := 1 / 6

-- Probability of guessing a question incorrectly
def prob_incorrect := 5 / 6

-- Binomial probability mass function for k successes out of n trials
def binom_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate probability P(X = 0)
def prob_X0 := binom_pmf 6 0 prob_correct

-- Calculate probability P(X = 1)
def prob_X1 := binom_pmf 6 1 prob_correct

-- Theorem for the desired probability
theorem prob_at_least_two_correct : 
  1 - (prob_X0 + prob_X1) = 34369 / 58420 := by
  sorry

end prob_at_least_two_correct_l190_190777


namespace marked_price_each_article_l190_190288

noncomputable def pair_price : ℝ := 50
noncomputable def discount : ℝ := 0.60
noncomputable def marked_price_pair : ℝ := 50 / 0.40
noncomputable def marked_price_each : ℝ := marked_price_pair / 2

theorem marked_price_each_article : 
  marked_price_each = 62.50 := by
  sorry

end marked_price_each_article_l190_190288


namespace car_speeds_midpoint_condition_l190_190130

theorem car_speeds_midpoint_condition 
  (v k : ℝ) (h_k : k > 1) 
  (A B C D : ℝ) (AB AD CD : ℝ)
  (h_midpoint : AD = AB / 2) 
  (h_CD_AD : CD / AD = 1 / 2)
  (h_D_midpoint : D = (A + B) / 2) 
  (h_C_on_return : C = D - CD) 
  (h_speeds : (v > 0) ∧ (k * v > v)) 
  (h_AB_AD : AB = 2 * AD) :
  k = 2 :=
by
  sorry

end car_speeds_midpoint_condition_l190_190130


namespace novels_in_shipment_l190_190297

theorem novels_in_shipment (N : ℕ) (H1: 225 = (3/4:ℚ) * N) : N = 300 := 
by
  sorry

end novels_in_shipment_l190_190297


namespace intersection_A_notB_l190_190350

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A according to the given condition
def A : Set ℝ := { x | |x - 1| > 1 }

-- Define set B according to the given condition
def B : Set ℝ := { x | (x - 1) * (x - 4) > 0 }

-- Define the complement of set B in U
def notB : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Lean statement to prove A ∩ notB = { x | 2 < x ∧ x ≤ 4 }
theorem intersection_A_notB :
  A ∩ notB = { x | 2 < x ∧ x ≤ 4 } :=
sorry

end intersection_A_notB_l190_190350


namespace sum_of_squares_of_sines_l190_190924

theorem sum_of_squares_of_sines (α : ℝ) : 
  (Real.sin α)^2 + (Real.sin (α + 60 * Real.pi / 180))^2 + (Real.sin (α + 120 * Real.pi / 180))^2 = 3 / 2 := 
by
  sorry

end sum_of_squares_of_sines_l190_190924


namespace annual_decrease_rate_l190_190960

theorem annual_decrease_rate :
  ∀ (P₀ P₂ : ℕ) (t : ℕ) (rate : ℝ),
    P₀ = 20000 → P₂ = 12800 → t = 2 → P₂ = P₀ * (1 - rate) ^ t → rate = 0.2 :=
by
sorry

end annual_decrease_rate_l190_190960


namespace div_d_a_value_l190_190890

variable {a b c d : ℚ}

theorem div_d_a_value (h1 : a / b = 3) (h2 : b / c = 5 / 3) (h3 : c / d = 2) : d / a = 1 / 10 := by
  sorry

end div_d_a_value_l190_190890


namespace sasha_made_an_error_l190_190821

theorem sasha_made_an_error :
  ∀ (f : ℕ → ℤ), 
  (∀ n, 1 ≤ n → n ≤ 9 → f n = n ∨ f n = -n) →
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 21) →
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 = 20) →
  false :=
by
  intros f h_cons h_volodya_sum h_sasha_sum
  sorry

end sasha_made_an_error_l190_190821


namespace tan_neg_five_pi_over_four_l190_190320

theorem tan_neg_five_pi_over_four : Real.tan (-5 * Real.pi / 4) = -1 :=
  sorry

end tan_neg_five_pi_over_four_l190_190320


namespace complement_of_angleA_is_54_l190_190874

variable (A : ℝ)

-- Condition: \(\angle A = 36^\circ\)
def angleA := 36

-- Definition of complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Proof statement
theorem complement_of_angleA_is_54 (h : angleA = 36) : complement angleA = 54 :=
sorry

end complement_of_angleA_is_54_l190_190874


namespace bob_corn_calc_l190_190573

noncomputable def bob_corn_left (initial_bushels : ℕ) (ears_per_bushel : ℕ) (bushels_taken_by_terry : ℕ) (bushels_taken_by_jerry : ℕ) (bushels_taken_by_linda : ℕ) (ears_taken_by_stacy : ℕ) : ℕ :=
  let initial_ears := initial_bushels * ears_per_bushel
  let ears_given_away := (bushels_taken_by_terry + bushels_taken_by_jerry + bushels_taken_by_linda) * ears_per_bushel + ears_taken_by_stacy
  initial_ears - ears_given_away

theorem bob_corn_calc :
  bob_corn_left 50 14 8 3 12 21 = 357 :=
by
  sorry

end bob_corn_calc_l190_190573


namespace vec_same_direction_l190_190607

theorem vec_same_direction (k : ℝ) : (k = 2) ↔ ∃ m : ℝ, m > 0 ∧ (k, 2) = (m * 1, m * 1) :=
by
  sorry

end vec_same_direction_l190_190607


namespace sum_first_twelve_arithmetic_divisible_by_6_l190_190134

theorem sum_first_twelve_arithmetic_divisible_by_6 
  (a d : ℕ) (h1 : a > 0) (h2 : d > 0) : 
  6 ∣ (12 * a + 66 * d) := 
by
  sorry

end sum_first_twelve_arithmetic_divisible_by_6_l190_190134


namespace original_marketing_pct_correct_l190_190367

-- Define the initial and final percentages of finance specialization students
def initial_finance_pct := 0.88
def final_finance_pct := 0.90

-- Define the final percentage of marketing specialization students
def final_marketing_pct := 0.43333333333333335

-- Define the original percentage of marketing specialization students
def original_marketing_pct := 0.45333333333333335

-- The Lean statement to prove the original percentage of marketing students
theorem original_marketing_pct_correct :
  initial_finance_pct + (final_marketing_pct - initial_finance_pct) = original_marketing_pct := 
sorry

end original_marketing_pct_correct_l190_190367


namespace smallest_three_digit_multiple_of_13_l190_190824

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n ∧ (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l190_190824


namespace blue_paint_quantity_l190_190539

-- Conditions
def paint_ratio (r b y w : ℕ) : Prop := r = 2 * w / 4 ∧ b = 3 * w / 4 ∧ y = 1 * w / 4 ∧ w = 4 * (r + b + y + w) / 10

-- Given
def quart_white_paint : ℕ := 16

-- Prove that Victor should use 12 quarts of blue paint
theorem blue_paint_quantity (r b y w : ℕ) (h : paint_ratio r b y w) (hw : w = quart_white_paint) : 
  b = 12 := by
  sorry

end blue_paint_quantity_l190_190539


namespace simplify_fraction_l190_190953

-- We state the problem as a theorem.
theorem simplify_fraction : (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3 / 5 := by sorry

end simplify_fraction_l190_190953


namespace quadruple_pieces_sold_l190_190651

theorem quadruple_pieces_sold (split_earnings : (2 : ℝ) * 5 = 10) 
  (single_pieces_sold : 100 * (0.01 : ℝ) = 1) 
  (double_pieces_sold : 45 * (0.02 : ℝ) = 0.9) 
  (triple_pieces_sold : 50 * (0.03 : ℝ) = 1.5) : 
  let total_earnings := 10
  let earnings_from_others := 3.4
  let quadruple_piece_price := 0.04
  total_earnings - earnings_from_others = 6.6 → 
  6.6 / quadruple_piece_price = 165 :=
by 
  intros 
  sorry

end quadruple_pieces_sold_l190_190651


namespace max_expression_sum_l190_190107

open Real

theorem max_expression_sum :
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ 
  (2 * x^2 - 3 * x * y + 4 * y^2 = 15 ∧ 
  (3 * x^2 + 2 * x * y + y^2 = 50 * sqrt 3 + 65)) :=
sorry

#eval 65 + 50 + 3 + 1 -- this should output 119

end max_expression_sum_l190_190107


namespace employed_population_percentage_l190_190907

noncomputable def percent_population_employed (total_population employed_males employed_females : ℝ) : ℝ :=
  employed_males + employed_females

theorem employed_population_percentage (population employed_males_percentage employed_females_percentage : ℝ) 
  (h1 : employed_males_percentage = 0.36 * population)
  (h2 : employed_females_percentage = 0.36 * population)
  (h3 : employed_females_percentage + employed_males_percentage = 0.50 * total_population)
  : total_population = 0.72 * population :=
by 
  sorry

end employed_population_percentage_l190_190907


namespace reducedRatesFraction_l190_190545

variable (total_hours_per_week : ℕ := 168)
variable (reduced_rate_hours_weekdays : ℕ := 12 * 5)
variable (reduced_rate_hours_weekends : ℕ := 24 * 2)

theorem reducedRatesFraction
  (h1 : total_hours_per_week = 7 * 24)
  (h2 : reduced_rate_hours_weekdays = 12 * 5)
  (h3 : reduced_rate_hours_weekends = 24 * 2) :
  (reduced_rate_hours_weekdays + reduced_rate_hours_weekends) / total_hours_per_week = 9 / 14 := 
  sorry

end reducedRatesFraction_l190_190545


namespace no_solutions_system_l190_190928

theorem no_solutions_system :
  ∀ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) →
  (y * x^2 + x + y = 0) →
  (y^2 + y - x^2 + 1 = 0) →
  false :=
by
  intro x y h1 h2 h3
  -- Proof goes here
  sorry

end no_solutions_system_l190_190928


namespace total_money_divided_l190_190555

theorem total_money_divided (x y : ℕ) (hx : x = 1000) (ratioxy : 2 * y = 8 * x) : x + y = 5000 := 
by
  sorry

end total_money_divided_l190_190555


namespace complement_union_l190_190351

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  U \ (A ∪ B) = {4} :=
by
  sorry

end complement_union_l190_190351


namespace packs_of_cake_l190_190092

-- Given conditions
def total_grocery_packs : ℕ := 27
def cookie_packs : ℕ := 23

-- Question: How many packs of cake did Lucy buy?
-- Mathematically equivalent problem: Proving that cake_packs is 4
theorem packs_of_cake : (total_grocery_packs - cookie_packs) = 4 :=
by
  -- Proof goes here. Using sorry to skip the proof.
  sorry

end packs_of_cake_l190_190092


namespace smallest_positive_divisible_by_111_has_last_digits_2004_l190_190969

theorem smallest_positive_divisible_by_111_has_last_digits_2004 :
  ∃ (X : ℕ), (∃ (A : ℕ), X = A * 10^4 + 2004) ∧ 111 ∣ X ∧ X = 662004 := by
  sorry

end smallest_positive_divisible_by_111_has_last_digits_2004_l190_190969


namespace sum_six_consecutive_integers_l190_190108

-- Statement of the problem
theorem sum_six_consecutive_integers (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l190_190108


namespace infinite_rational_points_xy_le_12_l190_190957

theorem infinite_rational_points_xy_le_12 :
  ∃ (S : Set (ℚ × ℚ)), (∀ (p : ℚ × ℚ), p ∈ S → 0 < p.fst ∧ 0 < p.snd ∧ p.fst * p.snd ≤ 12) ∧ S.Infinite :=
sorry

end infinite_rational_points_xy_le_12_l190_190957


namespace value_of_expression_is_correct_l190_190814

-- Defining the sub-expressions as Lean terms
def three_squared : ℕ := 3^2
def intermediate_result : ℕ := three_squared - 3
def final_result : ℕ := intermediate_result^2

-- The statement we need to prove
theorem value_of_expression_is_correct : final_result = 36 := by
  sorry

end value_of_expression_is_correct_l190_190814


namespace computer_price_increase_l190_190768

theorem computer_price_increase
  (P : ℝ)
  (h1 : 1.30 * P = 351) :
  (P + 1.30 * P) / P = 2.3 := by
  sorry

end computer_price_increase_l190_190768


namespace lemonade_sales_l190_190265

theorem lemonade_sales (cups_last_week : ℕ) (percent_more : ℕ) 
  (h_last_week : cups_last_week = 20)
  (h_percent_more : percent_more = 30) : 
  let cups_this_week := cups_last_week + (percent_more * cups_last_week / 100)
  in cups_last_week + cups_this_week = 46 := 
by
  -- Definitions and calculation
  let cups_this_week := cups_last_week + (percent_more * cups_last_week / 100)
  have h_this_week : cups_this_week = 26, from calc
    cups_this_week = 20 + (30 * 20 / 100) : by rw [h_last_week, h_percent_more]
    ... = 20 + 6 : by norm_num
    ... = 26 : by norm_num,
  show cups_last_week + cups_this_week = 46, from calc
    20 + 26 = 46 : by norm_num

end lemonade_sales_l190_190265


namespace dress_designs_possible_l190_190300

theorem dress_designs_possible (colors patterns fabric_types : Nat) (color_choices : colors = 5) (pattern_choices : patterns = 6) (fabric_type_choices : fabric_types = 2) : 
  colors * patterns * fabric_types = 60 := by 
  sorry

end dress_designs_possible_l190_190300


namespace series_pattern_l190_190518

theorem series_pattern :
    (3 / (1 * 2) * (1 / 2) + 4 / (2 * 3) * (1 / 2^2) + 5 / (3 * 4) * (1 / 2^3) + 6 / (4 * 5) * (1 / 2^4) + 7 / (5 * 6) * (1 / 2^5)) 
    = (1 - 1 / (6 * 2^5)) :=
  sorry

end series_pattern_l190_190518


namespace swapped_digits_greater_by_18_l190_190160

theorem swapped_digits_greater_by_18 (x : ℕ) : 
  (10 * x + 1) - (10 + x) = 18 :=
  sorry

end swapped_digits_greater_by_18_l190_190160


namespace rachel_total_time_l190_190099

-- Define the conditions
def num_chairs : ℕ := 20
def num_tables : ℕ := 8
def time_per_piece : ℕ := 6

-- Proof statement
theorem rachel_total_time : (num_chairs + num_tables) * time_per_piece = 168 := by
  sorry

end rachel_total_time_l190_190099


namespace qualified_flour_l190_190996

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end qualified_flour_l190_190996


namespace chord_slope_of_ellipse_l190_190884

theorem chord_slope_of_ellipse :
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + x2)/2 = 4 ∧ (y1 + y2)/2 = 2 ∧
    (x1^2)/36 + (y1^2)/9 = 1 ∧ (x2^2)/36 + (y2^2)/9 = 1) →
    (∃ k : ℝ, k = (y1 - y2)/(x1 - x2) ∧ k = -1/2) :=
sorry

end chord_slope_of_ellipse_l190_190884


namespace part1_monotonic_intervals_part2_three_zeros_l190_190195

open Function

noncomputable def f (a : ℝ) (x : ℝ) := x^2 + 2 * x - 4 + a / x

theorem part1_monotonic_intervals :
  (∀ x ∈ (Set.Ioo (-∞ : ℝ) 0), deriv (f 4) x < 0) ∧
  (∀ x ∈ (Set.Ioo (0 : ℝ) 1), deriv (f 4) x < 0) ∧
  (∀ x ∈ (Set.Ioo (1 : ℝ) ∞), deriv (f 4) x > 0) :=
sorry

theorem part2_three_zeros (a : ℝ) :
  (∀ b : ℝ, ∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔ (a ∈ Set.Ioo (-8 : ℝ) 0 ∪ Set.Ioo 0 (40 / 27)) :=
sorry

end part1_monotonic_intervals_part2_three_zeros_l190_190195


namespace james_car_purchase_l190_190508

/-- 
James sold his $20,000 car for 80% of its value, 
then bought a $30,000 sticker price car, 
and he was out of pocket $11,000. 
James bought the new car for 90% of its value. 
-/
theorem james_car_purchase (V_1 P_1 V_2 O P : ℝ)
  (hV1 : V_1 = 20000)
  (hP1 : P_1 = 80)
  (hV2 : V_2 = 30000)
  (hO : O = 11000)
  (hSaleOld : (P_1 / 100) * V_1 = 16000)
  (hDiff : 16000 + O = 27000)
  (hPurchase : (P / 100) * V_2 = 27000) :
  P = 90 := 
sorry

end james_car_purchase_l190_190508


namespace total_people_in_line_l190_190825

theorem total_people_in_line (n : ℕ) (h : n = 5): n + 2 = 7 :=
by
  -- This is where the proof would normally go, but we omit it with "sorry"
  sorry

end total_people_in_line_l190_190825


namespace calculate_brick_quantity_l190_190492

noncomputable def brick_quantity (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := wall_length * wall_height * wall_width
  wall_volume / brick_volume

theorem calculate_brick_quantity :
  brick_quantity 20 10 8 1000 800 2450 = 1225000 := 
by 
  -- Volume calculations are shown but proof is omitted
  sorry

end calculate_brick_quantity_l190_190492


namespace rose_bushes_after_work_l190_190631

def initial_rose_bushes := 2
def planned_rose_bushes := 4
def planting_rate := 3
def removed_rose_bushes := 5

theorem rose_bushes_after_work :
  initial_rose_bushes + (planned_rose_bushes * planting_rate) - removed_rose_bushes = 9 :=
by
  sorry

end rose_bushes_after_work_l190_190631


namespace absolute_value_sum_10_terms_l190_190608

def sequence_sum (n : ℕ) : ℤ := (n^2 - 4 * n + 2)

def term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n - 1)

-- Prove that the sum of the absolute values of the first 10 terms is 66.
theorem absolute_value_sum_10_terms : 
  (|term 1| + |term 2| + |term 3| + |term 4| + |term 5| + 
   |term 6| + |term 7| + |term 8| + |term 9| + |term 10| = 66) := 
by 
  -- Skip the proof
  sorry

end absolute_value_sum_10_terms_l190_190608


namespace non_neg_scalar_product_l190_190194

theorem non_neg_scalar_product (a b c d e f g h : ℝ) : 
  (0 ≤ ac + bd) ∨ (0 ≤ ae + bf) ∨ (0 ≤ ag + bh) ∨ (0 ≤ ce + df) ∨ (0 ≤ cg + dh) ∨ (0 ≤ eg + fh) :=
  sorry

end non_neg_scalar_product_l190_190194


namespace mowing_lawn_time_l190_190776

theorem mowing_lawn_time (pay_mow : ℝ) (rate_hour : ℝ) (time_plant : ℝ) (charge_flowers : ℝ) :
  pay_mow = 15 → rate_hour = 20 → time_plant = 2 → charge_flowers = 45 → 
  (charge_flowers + pay_mow) / rate_hour - time_plant = 1 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- This is an outline, so the actual proof steps are omitted
  sorry

end mowing_lawn_time_l190_190776


namespace volume_small_pyramid_eq_27_60_l190_190987

noncomputable def volume_of_smaller_pyramid (base_edge : ℝ) (slant_edge : ℝ) (height_above_base : ℝ) : ℝ :=
  let total_height := Real.sqrt ((slant_edge ^ 2) - ((base_edge / (2 * Real.sqrt 2)) ^ 2))
  let smaller_pyramid_height := total_height - height_above_base
  let scale_factor := (smaller_pyramid_height / total_height)
  let new_base_edge := base_edge * scale_factor
  let new_base_area := (new_base_edge ^ 2) * 2
  (1 / 3) * new_base_area * smaller_pyramid_height

theorem volume_small_pyramid_eq_27_60 :
  volume_of_smaller_pyramid (10 * Real.sqrt 2) 12 4 = 27.6 :=
by
  sorry

end volume_small_pyramid_eq_27_60_l190_190987


namespace part_a_l190_190691

theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : 
  (a % 10 = b % 10) := 
sorry

end part_a_l190_190691


namespace range_of_m_inequality_a_b_l190_190754

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 2|

theorem range_of_m (m : ℝ) : (∀ x, f x ≥ |m - 1|) → -2 ≤ m ∧ m ≤ 4 :=
sorry

theorem inequality_a_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a^2 + b^2 = 2) : 
  a + b ≥ 2 * a * b :=
sorry

end range_of_m_inequality_a_b_l190_190754


namespace intersection_A_B_l190_190338

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end intersection_A_B_l190_190338


namespace larger_number_is_seventy_two_l190_190620

def five_times_larger_is_six_times_smaller (x y : ℕ) : Prop := 5 * y = 6 * x
def difference_is_twelve (x y : ℕ) : Prop := y - x = 12

theorem larger_number_is_seventy_two (x y : ℕ) 
  (h1 : five_times_larger_is_six_times_smaller x y)
  (h2 : difference_is_twelve x y) : y = 72 :=
sorry

end larger_number_is_seventy_two_l190_190620


namespace smallest_k_base_representation_l190_190721

theorem smallest_k_base_representation :
  ∃ k : ℕ, (k > 0) ∧ (∀ n k, 0 = (42 * (1 - k^(n+1))/(1 - k))) ∧ (0 = (4 * (53 * (1 - k^(n+1))/(1 - k)))) →
  (k = 11) := sorry

end smallest_k_base_representation_l190_190721


namespace expected_number_of_games_probability_B_wins_the_match_l190_190791

-- Conditions
constant Va : ℝ := 0.3
constant Vd : ℝ := 0.5
constant Vb : ℝ := 1 - Va - Vd

constant V2 : ℝ := Va^2 + Vb^2

constant draw_in_3_games : ℝ := Vd^3
constant win_A_win_B_draw : ℝ := 3 * Va * Vb * Vd
constant V4 : ℝ := draw_in_3_games + win_A_win_B_draw

constant V3 : ℝ := 1 - (V2 + V4)

-- Part (a): Expected number of games
noncomputable def M : ℝ := 2 * V2 + 3 * V3 + 4 * V4

theorem expected_number_of_games : M = 3.175 := by
  sorry

-- Part (b): Probability that B wins the match
constant B_wins_2_games : ℝ := Vb^2
constant B_wins_1_and_2_draws : ℝ := 3 * Vb * Vd^2
constant B_wins_and_A_wins_and_draw : ℝ := 2 * Vb^2 * (1 - Vb)
constant B_wins_after_3_draws : ℝ := V4 * Vb

noncomputable def P_B_wins_the_match : ℝ := 
  B_wins_2_games + B_wins_1_and_2_draws + B_wins_and_A_wins_and_draw + B_wins_after_3_draws

theorem probability_B_wins_the_match : P_B_wins_the_match = 0.315 := by
  sorry

end expected_number_of_games_probability_B_wins_the_match_l190_190791


namespace determine_all_functions_l190_190324

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

theorem determine_all_functions (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end determine_all_functions_l190_190324


namespace polygon_sides_l190_190364

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 := sorry

end polygon_sides_l190_190364


namespace right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l190_190458

theorem right_triangle_arithmetic_progression_is_345 (a b c : ℕ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ d, b = a + d ∧ c = a + 2 * d)
  : (a, b, c) = (3, 4, 5) :=
by
  sorry

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

noncomputable def sqrt_golden_ratio_div_2 := Real.sqrt ((1 + Real.sqrt 5) / 2)

theorem right_triangle_geometric_progression 
  (a b c : ℝ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ r, b = a * r ∧ c = a * r * r)
  : (a, b, c) = (1, sqrt_golden_ratio_div_2, golden_ratio) :=
by
  sorry

end right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l190_190458


namespace no_partition_square_isosceles_10deg_l190_190261

theorem no_partition_square_isosceles_10deg :
  ¬ ∃ (P : ℝ → ℝ → Prop), 
    (∀ x y, P x y → ((x = y) ∨ ((10 * x + 10 * y + 160 * (180 - x - y)) = 9 * 10))) ∧
    (∀ x y, P x 90 → P x y) ∧
    (P 90 90 → False) :=
by
  sorry

end no_partition_square_isosceles_10deg_l190_190261


namespace megan_bottles_left_l190_190650

-- Defining the initial conditions
def initial_bottles : Nat := 17
def bottles_drank : Nat := 3

-- Theorem stating that Megan has 14 bottles left
theorem megan_bottles_left : initial_bottles - bottles_drank = 14 := by
  sorry

end megan_bottles_left_l190_190650


namespace farmer_pigs_chickens_l190_190301

-- Defining the problem in Lean 4

theorem farmer_pigs_chickens (p ch : ℕ) (h₁ : 30 * p + 24 * ch = 1200) (h₂ : p > 0) (h₃ : ch > 0) : 
  (p = 4) ∧ (ch = 45) :=
by sorry

end farmer_pigs_chickens_l190_190301


namespace intersection_interval_l190_190870

noncomputable def f (x: ℝ) : ℝ := Real.log x
noncomputable def g (x: ℝ) : ℝ := 7 - 2 * x

theorem intersection_interval : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = g x := 
sorry

end intersection_interval_l190_190870


namespace intersection_union_complement_union_l190_190594

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable [Inhabited (Set ℝ)]

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 2) > 1 }
noncomputable def setB : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection (U : Set ℝ) : 
  (setA ∩ setB) = { x : ℝ | (0 < x ∧ x < 1) ∨ x > 3 } := 
  sorry

theorem union (U : Set ℝ) : 
  (setA ∪ setB) = univ := 
  sorry

theorem complement_union (U : Set ℝ) : 
  ((U \ setA) ∪ setB) = { x : ℝ | x ≥ 0 } := 
  sorry

end intersection_union_complement_union_l190_190594


namespace jane_earnings_in_two_weeks_l190_190082

-- Define the conditions in the lean environment
def number_of_chickens : ℕ := 10
def eggs_per_chicken_per_week : ℕ := 6
def selling_price_per_dozen : ℕ := 2

-- Statement of the proof problem
theorem jane_earnings_in_two_weeks :
  (number_of_chickens * eggs_per_chicken_per_week * 2) / 12 * selling_price_per_dozen = 20 :=
by
  sorry

end jane_earnings_in_two_weeks_l190_190082


namespace distance_to_school_l190_190141

theorem distance_to_school (d : ℝ) (h1 : d / 5 + d / 25 = 1) : d = 25 / 6 :=
by
  sorry

end distance_to_school_l190_190141


namespace pints_in_vat_l190_190997

-- Conditions
def num_glasses : Nat := 5
def pints_per_glass : Nat := 30

-- Problem statement: prove that the total number of pints in the vat is 150
theorem pints_in_vat : num_glasses * pints_per_glass = 150 :=
by
  -- Proof goes here
  sorry

end pints_in_vat_l190_190997


namespace missing_weights_l190_190415

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end missing_weights_l190_190415


namespace andrew_total_travel_time_l190_190005

theorem andrew_total_travel_time :
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  subway_time + train_time + bike_time = 38 :=
by
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  sorry

end andrew_total_travel_time_l190_190005


namespace inequality_transformation_l190_190597

theorem inequality_transformation (x y : ℝ) (h : x > y) : 3 * x > 3 * y :=
by sorry

end inequality_transformation_l190_190597


namespace Jaymee_age_l190_190909

/-- Given that Jaymee is 2 years older than twice the age of Shara,
    and Shara is 10 years old, prove that Jaymee is 22 years old. -/
theorem Jaymee_age (Shara_age : ℕ) (h1 : Shara_age = 10) :
  let Jaymee_age := 2 * Shara_age + 2
  in Jaymee_age = 22 :=
by
  have h2 : 2 * Shara_age + 2 = 22 := sorry
  exact h2

end Jaymee_age_l190_190909


namespace sixty_percent_is_240_l190_190692

variable (x : ℝ)

-- Conditions
def forty_percent_eq_160 : Prop := 0.40 * x = 160

-- Proof problem
theorem sixty_percent_is_240 (h : forty_percent_eq_160 x) : 0.60 * x = 240 :=
sorry

end sixty_percent_is_240_l190_190692


namespace width_of_room_l190_190955

theorem width_of_room (C r l : ℝ) (hC : C = 18700) (hr : r = 850) (hl : l = 5.5) : 
  ∃ w, C / r / l = w ∧ w = 4 :=
by
  use 4
  sorry

end width_of_room_l190_190955


namespace rationalize_denominator_l190_190935

theorem rationalize_denominator (A B C : ℤ) (hA : A = 5) (hB : B = 49) (hC : C = 21)
  (h_pos : 0 < C) (h_not_divisible : ¬ ∃ p : ℤ, prime p ∧ p ^ 3 ∣ B) :
  A + B + C = 75 :=
by
  sorry

end rationalize_denominator_l190_190935


namespace randy_initial_blocks_l190_190100

theorem randy_initial_blocks (used_blocks left_blocks total_blocks : ℕ) (h1 : used_blocks = 19) (h2 : left_blocks = 59) : total_blocks = used_blocks + left_blocks → total_blocks = 78 :=
by 
  intros
  sorry

end randy_initial_blocks_l190_190100


namespace min_xy_positive_real_l190_190603

theorem min_xy_positive_real (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 / (2 + x) + 3 / (2 + y) = 1) :
  ∃ m : ℝ, m = 16 ∧ ∀ xy : ℝ, (xy = x * y) → xy ≥ m :=
by
  sorry

end min_xy_positive_real_l190_190603


namespace find_cost_price_l190_190837

theorem find_cost_price (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (h1 : SP = 715) (h2 : profit_percent = 0.10) (h3 : SP = CP * (1 + profit_percent)) : 
  CP = 650 :=
by
  sorry

end find_cost_price_l190_190837


namespace persimmons_in_Jungkook_house_l190_190421

-- Define the number of boxes and the number of persimmons per box
def num_boxes : ℕ := 4
def persimmons_per_box : ℕ := 5

-- Define the total number of persimmons calculation
def total_persimmons (boxes : ℕ) (per_box : ℕ) : ℕ := boxes * per_box

-- The main theorem statement proving the total number of persimmons
theorem persimmons_in_Jungkook_house : total_persimmons num_boxes persimmons_per_box = 20 := 
by 
  -- We should prove this, but we use 'sorry' to skip proof in this example.
  sorry

end persimmons_in_Jungkook_house_l190_190421


namespace estimated_number_of_red_balls_l190_190505

theorem estimated_number_of_red_balls (total_balls : ℕ) (red_draws : ℕ) (total_draws : ℕ)
    (h_total_balls : total_balls = 8) (h_red_draws : red_draws = 75) (h_total_draws : total_draws = 100) :
    total_balls * (red_draws / total_draws : ℚ) = 6 := 
by
  sorry

end estimated_number_of_red_balls_l190_190505


namespace equilateral_triangle_area_ratio_l190_190798

theorem equilateral_triangle_area_ratio :
  let side_small := 1
  let perim_small := 3 * side_small
  let total_fencing := 6 * perim_small
  let side_large := total_fencing / 3
  let area_small := (Real.sqrt 3) / 4 * side_small ^ 2
  let area_large := (Real.sqrt 3) / 4 * side_large ^ 2
  let total_area_small := 6 * area_small
  total_area_small / area_large = 1 / 6 :=
by
  sorry

end equilateral_triangle_area_ratio_l190_190798


namespace average_of_numbers_l190_190951

theorem average_of_numbers (x : ℝ) (h : (2 + x + 12) / 3 = 8) : x = 10 :=
by sorry

end average_of_numbers_l190_190951


namespace initial_interest_rate_l190_190117

theorem initial_interest_rate 
  (r P : ℝ)
  (h1 : 20250 = P * r)
  (h2 : 22500 = P * (r + 5)) :
  r = 45 :=
by
  sorry

end initial_interest_rate_l190_190117


namespace valid_routes_from_A_to_B_without_C_l190_190887

theorem valid_routes_from_A_to_B_without_C :
  let total_routes := Nat.choose 10 5
  let routes_to_C := Nat.choose 6 3
  let routes_from_C_to_B := Nat.choose 4 2
  total_routes - (routes_to_C * routes_from_C_to_B) = 132 :=
by
  sorry

end valid_routes_from_A_to_B_without_C_l190_190887


namespace triangle_OMN_area_l190_190348

noncomputable def rho (theta : ℝ) : ℝ := 4 * Real.cos theta + 2 * Real.sin theta

theorem triangle_OMN_area :
  let l1 (x y : ℝ) := y = (Real.sqrt 3 / 3) * x
  let l2 (x y : ℝ) := y = Real.sqrt 3 * x
  let C (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5
  let OM := 2 * Real.sqrt 3 + 1
  let ON := 2 + Real.sqrt 3
  let angle_MON := Real.pi / 6
  let area_OMN := (1 / 2) * OM * ON * Real.sin angle_MON
  (4 * (Real.sqrt 3 + 2) + 5 * Real.sqrt 3 = 8 + 5 * Real.sqrt 3) → 
  area_OMN = (8 + 5 * Real.sqrt 3) / 4 :=
sorry

end triangle_OMN_area_l190_190348


namespace solve_for_x_l190_190103

theorem solve_for_x (x : ℝ) (h : 1 / 3 + 1 / x = 2 / 3) : x = 3 :=
sorry

end solve_for_x_l190_190103


namespace hyperbola_range_k_l190_190892

theorem hyperbola_range_k (k : ℝ) : 
  (1 < k ∧ k < 3) ↔ (∃ x y : ℝ, (3 - k > 0) ∧ (k - 1 > 0) ∧ (x * x) / (3 - k) - (y * y) / (k - 1) = 1) :=
by {
  sorry
}

end hyperbola_range_k_l190_190892


namespace solution_exists_real_solution_31_l190_190185

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end solution_exists_real_solution_31_l190_190185


namespace missing_weights_l190_190416

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end missing_weights_l190_190416


namespace infinite_series_eval_l190_190584

open Filter
open Real
open Topology
open BigOperators

-- Define the relevant expression for the infinite sum
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (n / (n^4 - 4 * n^2 + 8))

-- The theorem statement
theorem infinite_series_eval : infinite_series_sum = 5 / 24 :=
by sorry

end infinite_series_eval_l190_190584


namespace number_subtraction_l190_190157

theorem number_subtraction
  (x : ℕ) (y : ℕ)
  (h1 : x = 30)
  (h2 : 8 * x - y = 102) : y = 138 :=
by 
  sorry

end number_subtraction_l190_190157


namespace weight_of_grapes_l190_190128

theorem weight_of_grapes :
  ∀ (weight_of_fruits weight_of_apples weight_of_oranges weight_of_strawberries weight_of_grapes : ℕ),
  weight_of_fruits = 10 →
  weight_of_apples = 3 →
  weight_of_oranges = 1 →
  weight_of_strawberries = 3 →
  weight_of_fruits = weight_of_apples + weight_of_oranges + weight_of_strawberries + weight_of_grapes →
  weight_of_grapes = 3 :=
by
  intros
  sorry

end weight_of_grapes_l190_190128


namespace qualified_flour_l190_190995

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end qualified_flour_l190_190995


namespace avg_speed_4_2_l190_190988

noncomputable def avg_speed_round_trip (D : ℝ) : ℝ :=
  let speed_up := 3
  let speed_down := 7
  let total_distance := 2 * D
  let total_time := D / speed_up + D / speed_down
  total_distance / total_time

theorem avg_speed_4_2 (D : ℝ) (hD : D > 0) : avg_speed_round_trip D = 4.2 := by
  sorry

end avg_speed_4_2_l190_190988


namespace phoebe_dog_peanut_butter_l190_190656

-- Definitions based on the conditions
def servings_per_jar : ℕ := 15
def jars_needed : ℕ := 4
def days : ℕ := 30

-- Problem statement
theorem phoebe_dog_peanut_butter :
  (jars_needed * servings_per_jar) / days / 2 = 1 :=
by sorry

end phoebe_dog_peanut_butter_l190_190656


namespace total_travel_time_l190_190007

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end total_travel_time_l190_190007


namespace sum_circumferences_of_small_circles_l190_190114

theorem sum_circumferences_of_small_circles (R : ℝ) (n : ℕ) (hR : R > 0) (hn : n > 0) :
  let original_circumference := 2 * Real.pi * R
  let part_length := original_circumference / n
  let small_circle_radius := part_length / Real.pi
  let small_circle_circumference := 2 * Real.pi * small_circle_radius
  let total_circumference := n * small_circle_circumference
  total_circumference = 2 * Real.pi ^ 2 * R :=
by {
  sorry
}

end sum_circumferences_of_small_circles_l190_190114


namespace purpose_of_LB_full_nutrient_medium_l190_190675

/--
Given the experiment "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source",
which involves both experimental and control groups with the following conditions:
- The variable in the experiment is the difference in the medium used.
- The experimental group uses a medium with urea as the only nitrogen source (selective medium).
- The control group uses a full-nutrient medium.

Prove that the purpose of preparing LB full-nutrient medium is to observe the types and numbers
of soil microorganisms that can grow under full-nutrient conditions.
-/
theorem purpose_of_LB_full_nutrient_medium
  (experiment: String) (experimental_variable: String) (experimental_group: String) (control_group: String)
  (H1: experiment = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (H2: experimental_variable = "medium")
  (H3: experimental_group = "medium with urea as the only nitrogen source (selective medium)")
  (H4: control_group = "full-nutrient medium") :
  purpose_of_preparing_LB_full_nutrient_medium = "observe the types and numbers of soil microorganisms that can grow under full-nutrient conditions" :=
sorry

end purpose_of_LB_full_nutrient_medium_l190_190675


namespace average_percentage_l190_190020

theorem average_percentage (n1 n2 : ℕ) (s1 s2 : ℕ)
  (h1 : n1 = 15) (h2 : s1 = 80) (h3 : n2 = 10) (h4 : s2 = 90) :
  (n1 * s1 + n2 * s2) / (n1 + n2) = 84 :=
by
  sorry

end average_percentage_l190_190020


namespace orthogonal_trajectories_angle_at_origin_l190_190475

theorem orthogonal_trajectories_angle_at_origin (x y : ℝ) (a : ℝ) :
  ((x + 2 * y) ^ 2 = a * (x + y)) →
  (∃ φ : ℝ, φ = π / 4) :=
by
  sorry

end orthogonal_trajectories_angle_at_origin_l190_190475


namespace rationalize_denominator_l190_190944

theorem rationalize_denominator :
  ∃ (A B C : ℤ), C > 0 ∧ (∀ p : ℤ, prime p → ¬(p^3 ∣ B)) ∧ 
    (5 / (3 * (7 : ℝ)^(1/3)) = (A * (B : ℝ)^(1/3)) / C) ∧ A + B + C = 75 :=
sorry

end rationalize_denominator_l190_190944


namespace perpendicular_tangent_line_l190_190466

theorem perpendicular_tangent_line (a b : ℝ) : 
  let line_slope := 1 / 3,
      perp_slope := -3,
      curve := λ x, x^3 + 3 * x^2 - 5,
      derivative_curve := λ x, 3 * x^2 + 6 * x,
      a_eqn := a = -1,
      b_eqn := b = -3
  in
  (2 * a - 6 * b + 1 = 0) ∧ (a, b) = P ∧ (derivative_curve a = -3) → 
  3 * x + y + 6 = 0 :=
by
  sorry

end perpendicular_tangent_line_l190_190466


namespace find_other_person_weight_l190_190271

noncomputable def other_person_weight (n avg new_avg W1 : ℕ) : ℕ :=
  let total_initial := n * avg
  let new_n := n + 2
  let total_new := new_n * new_avg
  total_new - total_initial - W1

theorem find_other_person_weight:
  other_person_weight 23 48 51 78 = 93 := by
  sorry

end find_other_person_weight_l190_190271


namespace find_age_l190_190662

open Nat

-- Definition of ages
def Teacher_Zhang_age (z : Nat) := z
def Wang_Bing_age (w : Nat) := w

-- Conditions
axiom teacher_zhang_condition (z w : Nat) : z = 3 * w + 4
axiom age_comparison_condition (z w : Nat) : z - 10 = w + 10

-- Proposition to prove
theorem find_age (z w : Nat) (hz : z = 3 * w + 4) (hw : z - 10 = w + 10) : z = 28 ∧ w = 8 := by
  sorry

end find_age_l190_190662


namespace roots_reciprocal_l190_190333

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 4 * x1 - 2 = 0) (h2 : x2^2 - 4 * x2 - 2 = 0) (h3 : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = -2 := 
sorry

end roots_reciprocal_l190_190333


namespace determine_N_l190_190352

theorem determine_N (N : ℕ) : 995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end determine_N_l190_190352


namespace find_correct_average_of_numbers_l190_190664

variable (nums : List ℝ)
variable (n : ℕ) (avg_wrong avg_correct : ℝ) (wrong_val correct_val : ℝ)

noncomputable def correct_average (nums : List ℝ) (wrong_val correct_val : ℝ) : ℝ :=
  let correct_sum := nums.sum - wrong_val + correct_val
  correct_sum / nums.length

theorem find_correct_average_of_numbers
  (h₀ : n = 10)
  (h₁ : avg_wrong = 15)
  (h₂ : wrong_val = 26)
  (h₃ : correct_val = 36)
  (h₄ : avg_correct = 16)
  (nums : List ℝ) :
  avg_wrong * n - wrong_val + correct_val = avg_correct * n := 
sorry

end find_correct_average_of_numbers_l190_190664


namespace correlation_is_1_3_4_l190_190275

def relationship1 := "The relationship between a person's age and their wealth"
def relationship2 := "The relationship between a point on a curve and its coordinates"
def relationship3 := "The relationship between apple production and climate"
def relationship4 := "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"

def isCorrelation (rel: String) : Bool :=
  if rel == relationship1 ∨ rel == relationship3 ∨ rel == relationship4 then true else false

theorem correlation_is_1_3_4 :
  {relationship1, relationship3, relationship4} = {r | isCorrelation r = true} := 
by
  sorry

end correlation_is_1_3_4_l190_190275


namespace micah_water_l190_190257

theorem micah_water (x : ℝ) (h1 : 3 * x + x = 6) : x = 1.5 :=
sorry

end micah_water_l190_190257


namespace doris_weeks_to_cover_expenses_l190_190727

-- Define the constants and conditions from the problem
def hourly_rate : ℝ := 20
def monthly_expenses : ℝ := 1200
def weekday_hours_per_day : ℝ := 3
def weekdays_per_week : ℝ := 5
def saturday_hours : ℝ := 5

-- Calculate total hours worked per week
def weekly_hours := (weekday_hours_per_day * weekdays_per_week) + saturday_hours

-- Calculate weekly earnings
def weekly_earnings := hourly_rate * weekly_hours

-- Finally, the number of weeks required to meet the monthly expenses
def required_weeks := monthly_expenses / weekly_earnings

-- The theorem to prove
theorem doris_weeks_to_cover_expenses : required_weeks = 3 := by
  -- We skip the proof but indicate it needs to be provided
  sorry

end doris_weeks_to_cover_expenses_l190_190727


namespace missed_angle_l190_190897

def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem missed_angle :
  ∃ (n : ℕ), sum_interior_angles n = 3060 ∧ 3060 - 2997 = 63 :=
by {
  sorry
}

end missed_angle_l190_190897


namespace time_to_pass_tree_l190_190566

noncomputable def length_of_train : ℝ := 275
noncomputable def speed_in_kmh : ℝ := 90
noncomputable def speed_in_m_per_s : ℝ := speed_in_kmh * (5 / 18)

theorem time_to_pass_tree : (length_of_train / speed_in_m_per_s) = 11 :=
by {
  sorry
}

end time_to_pass_tree_l190_190566


namespace factorize_expr_l190_190863

-- Define the variables a and b as elements of an arbitrary ring
variables {R : Type*} [CommRing R] (a b : R)

-- Prove the factorization identity
theorem factorize_expr : a^2 * b - b = b * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expr_l190_190863


namespace remainder_2_pow_2015_mod_20_l190_190822

/-- 
  Given that powers of 2 modulo 20 follow a repeating cycle every 4 terms:
  2, 4, 8, 16, 12
  
  Prove that the remainder when \(2^{2015}\) is divided by 20 is 8.
-/
theorem remainder_2_pow_2015_mod_20 : (2 ^ 2015) % 20 = 8 :=
by
  -- The proof is to be filled in.
  sorry

end remainder_2_pow_2015_mod_20_l190_190822


namespace find_k_l190_190677

variables (l w : ℝ) (p A k : ℝ)

def rectangle_conditions : Prop :=
  (l / w = 5 / 2) ∧ (p = 2 * (l + w))

theorem find_k (h : rectangle_conditions l w p) :
  A = (5 / 98) * p^2 :=
sorry

end find_k_l190_190677


namespace henrys_friend_money_l190_190491

theorem henrys_friend_money (h1 h2 : ℕ) (T : ℕ) (f : ℕ) : h1 = 5 → h2 = 2 → T = 20 → h1 + h2 + f = T → f = 13 :=
by
  intros h1_eq h2_eq T_eq total_eq
  rw [h1_eq, h2_eq, T_eq] at total_eq
  sorry

end henrys_friend_money_l190_190491


namespace biased_coin_probability_l190_190688

theorem biased_coin_probability (h : ℚ) (H : 21 * (1 - h) = 35 * h) :
  let p : ℚ := 35 * (3/8)^4 * (5/8)^3 in
  let num_denom_sum := p.num + p.denom in
  num_denom_sum = 2451527 :=
by
  sorry

end biased_coin_probability_l190_190688


namespace rationalize_denominator_l190_190938

theorem rationalize_denominator 
  (A B C : ℤ) 
  (hA : A = 5) 
  (hB : B = 49) 
  (hC : C = 21)
  (hC_positive : C > 0) 
  (hB_not_cubed : ∀ p : ℤ, prime p → ¬ ∃ k : ℤ, B = p^3 * k) :
  A + B + C = 75 := by
  sorry

end rationalize_denominator_l190_190938


namespace solution_exists_real_solution_31_l190_190184

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end solution_exists_real_solution_31_l190_190184


namespace bus_remaining_distance_l190_190833

noncomputable def final_distance (z x : ℝ) : ℝ :=
  z - (z * x / 5)

theorem bus_remaining_distance (z : ℝ) :
  (z / 2) / (z - 19.2) = x ∧ (z - 12) / (z / 2) = x → final_distance z x = 6.4 :=
by
  intro h
  sorry

end bus_remaining_distance_l190_190833


namespace samuel_apples_left_l190_190844

def bonnieApples : ℕ := 8
def extraApples : ℕ := 20
def samuelTotalApples : ℕ := bonnieApples + extraApples
def samuelAte : ℕ := samuelTotalApples / 2
def samuelRemainingAfterEating : ℕ := samuelTotalApples - samuelAte
def samuelUsedForPie : ℕ := samuelRemainingAfterEating / 7
def samuelFinalRemaining : ℕ := samuelRemainingAfterEating - samuelUsedForPie

theorem samuel_apples_left :
  samuelFinalRemaining = 12 := by
  sorry

end samuel_apples_left_l190_190844


namespace find_certain_number_l190_190831

theorem find_certain_number (x y : ℕ) (h1 : x = 19) (h2 : x + y = 36) :
  8 * x + 3 * y = 203 := by
  sorry

end find_certain_number_l190_190831


namespace intersection_eq_l190_190340

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end intersection_eq_l190_190340


namespace triangle_inequality_inequality_l190_190633

theorem triangle_inequality_inequality {a b c : ℝ}
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) :=
sorry

end triangle_inequality_inequality_l190_190633


namespace quadratic_has_one_solution_l190_190865

theorem quadratic_has_one_solution (q : ℚ) (hq : q ≠ 0) : 
  (∃ x, ∀ y, q*y^2 - 18*y + 8 = 0 → x = y) ↔ q = 81 / 8 :=
by
  sorry

end quadratic_has_one_solution_l190_190865


namespace A_inter_B_empty_l190_190349

def setA : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def setB : Set ℝ := {x | Real.log x / Real.log 4 > 1/2}

theorem A_inter_B_empty : setA ∩ setB = ∅ := by
  sorry

end A_inter_B_empty_l190_190349


namespace blue_balls_taken_out_l190_190965

theorem blue_balls_taken_out
  (x : ℕ) 
  (balls_initial : ℕ := 18)
  (blue_initial : ℕ := 6)
  (prob_blue : ℚ := 1/5)
  (total : ℕ := balls_initial - x)
  (blue_current : ℕ := blue_initial - x) :
  (↑blue_current / ↑total = prob_blue) → x = 3 :=
by
  sorry

end blue_balls_taken_out_l190_190965


namespace relation_between_incircle_radius_perimeter_area_l190_190438

theorem relation_between_incircle_radius_perimeter_area (r p S : ℝ) (h : S = (1 / 2) * r * p) : S = (1 / 2) * r * p :=
by {
  sorry
}

end relation_between_incircle_radius_perimeter_area_l190_190438


namespace inequality_holds_l190_190736

theorem inequality_holds (x : ℝ) (hx : 0 < x ∧ x < 4) :
  ∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y) :=
by
  intros y hy_gt_zero
  sorry

end inequality_holds_l190_190736


namespace find_perpendicular_tangent_line_l190_190467

noncomputable def line_eq (a b c x y : ℝ) : Prop :=
a * x + b * y + c = 0

def perp_line (a b c d e f : ℝ) (x y : ℝ) : Prop :=
b * x - a * y = 0  -- Perpendicular condition

def tangent_line (f : ℝ → ℝ) (a b c x : ℝ) : Prop :=
∃ t, f t = a * t + b * (f t) + c ∧ (deriv f t) = -a / b  -- Tangency condition with derivative

theorem find_perpendicular_tangent_line :
  let f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 5 in
  ∃ a b c d e f: ℝ, perp_line 2 (-6) 1 a b c ∧ tangent_line f a b c ∧ line_eq 3 1 6 (x : ℝ) (f x) :=
sorry

end find_perpendicular_tangent_line_l190_190467


namespace min_value_of_f_l190_190667

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x ^ 2

theorem min_value_of_f : ∀ x > 0, f x ≥ 9 ∧ (f x = 9 ↔ x = 2) :=
by
  sorry

end min_value_of_f_l190_190667


namespace problem_statement_l190_190283

theorem problem_statement (a x m : ℝ) (h₀ : |a| ≤ 1) (h₁ : |x| ≤ 1) :
  (∀ x a, |x^2 - a * x - a^2| ≤ m) ↔ m ≥ 5/4 :=
sorry

end problem_statement_l190_190283


namespace sufficient_but_not_necessary_condition_l190_190480

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 > 1) ∧ ¬(x^2 > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l190_190480


namespace rectangle_area_l190_190277

theorem rectangle_area (l : ℝ) (w : ℝ) (h_l : l = 15) (h_ratio : (2 * l + 2 * w) / w = 5) : (l * w) = 150 :=
by
  sorry

end rectangle_area_l190_190277


namespace point_transformation_l190_190154

theorem point_transformation (a b : ℝ) :
  let P := (a, b)
  let P₁ := (2 * 2 - a, 2 * 3 - b) -- Rotate P 180° counterclockwise around (2, 3)
  let P₂ := (P₁.2, P₁.1)           -- Reflect P₁ about the line y = x
  P₂ = (5, -4) → a - b = 7 :=
by
  intros
  sorry

end point_transformation_l190_190154


namespace count_perfect_squares_cubes_under_1000_l190_190221

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l190_190221


namespace count_perfect_squares_and_cubes_l190_190219

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l190_190219


namespace gcd_12_20_l190_190047

theorem gcd_12_20 : Nat.gcd 12 20 = 4 := by
  sorry

end gcd_12_20_l190_190047


namespace sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l190_190341

noncomputable def sin_pi_div_two_plus_2alpha (α : ℝ) : ℝ :=
  Real.sin ((Real.pi / 2) + 2 * α)

def cos_alpha (α : ℝ) := Real.cos α = - (Real.sqrt 2) / 3

theorem sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth (α : ℝ) (h : cos_alpha α) :
  sin_pi_div_two_plus_2alpha α = -5 / 9 :=
sorry

end sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l190_190341


namespace rectangle_not_sum_110_l190_190661

noncomputable def not_sum_110 : Prop :=
  ∀ (w : ℕ), (w > 0) → (2 * w^2 + 6 * w ≠ 110)

theorem rectangle_not_sum_110 : not_sum_110 := 
  sorry

end rectangle_not_sum_110_l190_190661


namespace Erik_ate_pie_l190_190843

theorem Erik_ate_pie (Frank_ate Erik_ate more_than: ℝ) (h1: Frank_ate = 0.3333333333333333)
(h2: more_than = 0.3333333333333333)
(h3: Erik_ate = Frank_ate + more_than) : Erik_ate = 0.6666666666666666 :=
by
  sorry

end Erik_ate_pie_l190_190843


namespace sheep_to_cow_water_ratio_l190_190517

-- Set up the initial conditions
def number_of_cows := 40
def water_per_cow_per_day := 80
def number_of_sheep := 10 * number_of_cows
def total_water_per_week := 78400

-- Calculate total water consumption of cows per week
def water_cows_per_week := number_of_cows * water_per_cow_per_day * 7

-- Calculate total water consumption of sheep per week
def water_sheep_per_week := total_water_per_week - water_cows_per_week

-- Calculate daily water consumption per sheep
def water_sheep_per_day := water_sheep_per_week / 7
def daily_water_per_sheep := water_sheep_per_day / number_of_sheep

-- Define the target ratio
def target_ratio := 1 / 4

-- Statement to prove
theorem sheep_to_cow_water_ratio :
  (daily_water_per_sheep / water_per_cow_per_day) = target_ratio :=
sorry

end sheep_to_cow_water_ratio_l190_190517


namespace arithmetic_progression_25th_term_l190_190019

def arithmetic_progression_nth_term (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_progression_25th_term : arithmetic_progression_nth_term 5 7 25 = 173 := by
  sorry

end arithmetic_progression_25th_term_l190_190019


namespace rationalize_denominator_l190_190945

theorem rationalize_denominator :
  ∃ (A B C : ℤ), C > 0 ∧ (∀ p : ℤ, prime p → ¬(p^3 ∣ B)) ∧ 
    (5 / (3 * (7 : ℝ)^(1/3)) = (A * (B : ℝ)^(1/3)) / C) ∧ A + B + C = 75 :=
sorry

end rationalize_denominator_l190_190945


namespace concurrent_segments_unique_solution_l190_190248

theorem concurrent_segments_unique_solution (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  4^c - 1 = (2^a - 1) * (2^b - 1) ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
by
  sorry

end concurrent_segments_unique_solution_l190_190248


namespace equilateral_triangle_l190_190097

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 0 < α ∧ α < π)
  (h5 : 0 < β ∧ β < π)
  (h6 : 0 < γ ∧ γ < π)
  (h7 : α + β + γ = π)
  (h8 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  α = β ∧ β = γ ∧ γ = α :=
by
  sorry

end equilateral_triangle_l190_190097


namespace oscar_leap_longer_l190_190729

noncomputable def elmer_strides (poles : ℕ) (strides_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_strides := (poles - 1) * strides_per_gap
  total_distance / total_strides

noncomputable def oscar_leaps (poles : ℕ) (leaps_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_leaps := (poles - 1) * leaps_per_gap
  total_distance / total_leaps

theorem oscar_leap_longer (poles : ℕ) (strides_per_gap leaps_per_gap : ℕ) (distance_miles : ℝ) :
  poles = 51 -> strides_per_gap = 50 -> leaps_per_gap = 15 -> distance_miles = 1.25 ->
  let elmer_stride := elmer_strides poles strides_per_gap distance_miles
  let oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  (oscar_leap - elmer_stride) * 12 = 74 :=
by
  intros h_poles h_strides h_leaps h_distance
  have elmer_stride := elmer_strides poles strides_per_gap distance_miles
  have oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  sorry

end oscar_leap_longer_l190_190729


namespace functional_equation_solution_l190_190180

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y * f (x + y) + f x) = 4 * x + 2 * y * f (x + y)) →
  (∀ x : ℝ, f x = 2 * x) :=
sorry

end functional_equation_solution_l190_190180


namespace prime_difference_condition_l190_190131

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_difference_condition :
  ∃ (x y : ℕ), is_prime x ∧ is_prime y ∧ 4 < x ∧ x < 18 ∧ 4 < y ∧ y < 18 ∧ x ≠ y ∧ (x * y - (x + y)) = 119 :=
by
  sorry

end prime_difference_condition_l190_190131


namespace area_of_triangle_DOE_l190_190927

-- Definitions of points D, O, and E
def D (p : ℝ) : ℝ × ℝ := (0, p)
def O : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (15, 0)

-- Theorem statement
theorem area_of_triangle_DOE (p : ℝ) : 
  let base := 15
  let height := p
  let area := (1/2) * base * height
  area = (15 * p) / 2 :=
by sorry

end area_of_triangle_DOE_l190_190927


namespace determine_y_l190_190326

theorem determine_y (y : ℝ) (y_nonzero : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := 
sorry

end determine_y_l190_190326


namespace sigma_algebra_inequality_l190_190291

noncomputable section

open MeasureTheory

-- defining the Bernoulli random variable property
variables {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}

-- Assuming xi_n are independent Bernoulli random variables
variables (xi : ℕ → Ω → ℤ) (X : ℕ → Ω → ℤ)
variable (ℙ : MeasureTheory.ProbabilityMeasure Ω)
variables [∀ n, MeasureTheory.Independent_ (σ (xi n)) (ℙ)]
variables [∀ n, MeasureTheory.isProbabilityMeasure (σ (xi n)) ℙ]

-- Definitions from the problem
def is_bernoulli (xi : Ω → ℤ) : Prop :=
  (ℙ (λ ω, xi ω = -1) = 1/2) ∧ (ℙ (λ ω, xi ω = 1) = 1/2)

def X_n (n : ℕ) (ω : Ω) := ∏ i in finset.range (n + 1), xi i ω

-- Definitions of sigma algebras
def G : MeasurableSpace Ω := MeasurableSpace.generateFrom { s | ∃ n, s = {ω | xi n ω ∈ {-1, 1} }}
def E_n (n : ℕ) : MeasurableSpace Ω := MeasurableSpace.generateFrom { s | ∃ k ≥ n, s = {ω | X k ω ∈ {-1, 1} }}

-- Question: Proving the inequality of sigma algebras
theorem sigma_algebra_inequality (h_bernoulli : ∀ n, is_bernoulli (xi n)) :
  (⋂ n, MeasurableSpace.generateFrom { G, E_n n }) ≠ MeasurableSpace.generateFrom { G, ⋂ n, E_n n } :=
sorry

end sigma_algebra_inequality_l190_190291


namespace sum_of_solutions_l190_190050

theorem sum_of_solutions (x : ℝ) : 
  (∃ y z, x^2 + 2017 * x - 24 = 2017 ∧ y^2 + 2017 * y - 2041 = 0 ∧ z^2 + 2017 * z - 2041 = 0 ∧ y ≠ z) →
  y + z = -2017 := 
by 
  sorry

end sum_of_solutions_l190_190050


namespace solve_ineq_l190_190191

theorem solve_ineq (x : ℝ) : (x > 0 ∧ x < 3 ∨ x > 8) → x^3 - 9 * x^2 + 24 * x > 0 :=
by
  sorry

end solve_ineq_l190_190191


namespace solve_for_y_in_terms_of_x_l190_190616

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3 * x) : y = -2 * x - 2 :=
sorry

end solve_for_y_in_terms_of_x_l190_190616


namespace flour_qualification_l190_190993

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end flour_qualification_l190_190993


namespace molecular_weight_of_benzene_l190_190845

def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def number_of_C_atoms : ℕ := 6
def number_of_H_atoms : ℕ := 6

theorem molecular_weight_of_benzene : 
  (number_of_C_atoms * molecular_weight_C + number_of_H_atoms * molecular_weight_H) = 78.108 :=
by
  sorry

end molecular_weight_of_benzene_l190_190845


namespace geometric_sequence_common_ratio_l190_190771

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 3 = 1/2)
  (h3 : a 1 * (1 + q) = 3) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l190_190771


namespace original_population_l190_190309

theorem original_population (n : ℕ) (h1 : n + 1500 * 85 / 100 = n - 45) : n = 8800 := 
by
  sorry

end original_population_l190_190309


namespace black_lambs_count_l190_190463

def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193
def brown_lambs : ℕ := 527

theorem black_lambs_count :
  total_lambs - white_lambs - brown_lambs = 5328 :=
by
  -- Proof omitted
  sorry

end black_lambs_count_l190_190463


namespace evaluate_expression_l190_190042

theorem evaluate_expression : 
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  e = 3 + 10 * Real.sqrt 3 / 3 :=
by
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  have h : e = 3 + 10 * Real.sqrt 3 / 3 := sorry
  exact h

end evaluate_expression_l190_190042


namespace train_length_is_499_96_l190_190032

-- Define the conditions
def speed_train_kmh : ℕ := 75   -- Speed of the train in km/h
def speed_man_kmh : ℕ := 3     -- Speed of the man in km/h
def time_cross_s : ℝ := 24.998 -- Time taken for the train to cross the man in seconds

-- Define the conversion factors
def km_to_m : ℕ := 1000        -- Conversion from kilometers to meters
def hr_to_s : ℕ := 3600        -- Conversion from hours to seconds

-- Define relative speed in m/s
def relative_speed_ms : ℕ := (speed_train_kmh - speed_man_kmh) * km_to_m / hr_to_s

-- Prove the length of the train in meters
def length_of_train : ℝ := relative_speed_ms * time_cross_s

theorem train_length_is_499_96 : length_of_train = 499.96 := sorry

end train_length_is_499_96_l190_190032


namespace amy_total_distance_equals_168_l190_190712

def amy_biked_monday := 12

def amy_biked_tuesday (monday: ℕ) := 2 * monday - 3

def amy_biked_other_day (previous_day: ℕ) := previous_day + 2

def total_distance_bike_week := 
  let monday := amy_biked_monday
  let tuesday := amy_biked_tuesday monday
  let wednesday := amy_biked_other_day tuesday
  let thursday := amy_biked_other_day wednesday
  let friday := amy_biked_other_day thursday
  let saturday := amy_biked_other_day friday
  let sunday := amy_biked_other_day saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem amy_total_distance_equals_168 : 
  total_distance_bike_week = 168 := by
  sorry

end amy_total_distance_equals_168_l190_190712


namespace find_C_plus_D_l190_190240

theorem find_C_plus_D
  (C D : ℕ)
  (h1 : D = C + 2)
  (h2 : 2 * C^2 + 5 * C + 3 - (7 * D + 5) = (C + D)^2 + 6 * (C + D) + 8)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D) :
  C + D = 26 := by
  sorry

end find_C_plus_D_l190_190240


namespace sampling_interval_l190_190964

theorem sampling_interval 
  (total_population : ℕ) 
  (individuals_removed : ℕ) 
  (population_after_removal : ℕ)
  (sampling_interval : ℕ) :
  total_population = 102 →
  individuals_removed = 2 →
  population_after_removal = total_population - individuals_removed →
  population_after_removal = 100 →
  ∃ s : ℕ, population_after_removal % s = 0 ∧ s = 10 := 
by
  sorry

end sampling_interval_l190_190964


namespace simplify_evaluate_expr_l190_190797

theorem simplify_evaluate_expr (x : ℕ) (h : x = 2023) : (x + 1) ^ 2 - x * (x + 1) = 2024 := 
by 
  sorry

end simplify_evaluate_expr_l190_190797


namespace rationalize_denominator_l190_190939

theorem rationalize_denominator 
  (A B C : ℤ) 
  (hA : A = 5) 
  (hB : B = 49) 
  (hC : C = 21)
  (hC_positive : C > 0) 
  (hB_not_cubed : ∀ p : ℤ, prime p → ¬ ∃ k : ℤ, B = p^3 * k) :
  A + B + C = 75 := by
  sorry

end rationalize_denominator_l190_190939


namespace volume_common_part_equal_quarter_volume_each_cone_l190_190009

theorem volume_common_part_equal_quarter_volume_each_cone
  (r h : ℝ) (V_cone : ℝ)
  (h_cone_volume : V_cone = (1 / 3) * π * r^2 * h) :
  ∃ V_common, V_common = (1 / 4) * V_cone :=
by
  -- Main structure of the proof skipped
  sorry

end volume_common_part_equal_quarter_volume_each_cone_l190_190009


namespace advertisement_broadcasting_methods_l190_190832

/-- A TV station is broadcasting 5 different advertisements.
There are 3 different commercial advertisements.
There are 2 different Olympic promotional advertisements.
The last advertisement must be an Olympic promotional advertisement.
The two Olympic promotional advertisements cannot be broadcast consecutively.
Prove that the total number of different broadcasting methods is 18. -/
theorem advertisement_broadcasting_methods : 
  ∃ (arrangements : ℕ), arrangements = 18 := sorry

end advertisement_broadcasting_methods_l190_190832


namespace water_added_l190_190701

theorem water_added (W x : ℕ) (h₁ : 2 * W = 5 * 10)
                    (h₂ : 2 * (W + x) = 7 * 10) :
  x = 10 :=
by
  sorry

end water_added_l190_190701


namespace division_of_decimals_l190_190684

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l190_190684


namespace sin_x_plus_pi_l190_190596

theorem sin_x_plus_pi {x : ℝ} (hx : Real.sin x = -4 / 5) : Real.sin (x + Real.pi) = 4 / 5 :=
by
  -- Proof steps go here
  sorry

end sin_x_plus_pi_l190_190596


namespace balls_in_boxes_l190_190493

theorem balls_in_boxes : 
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  total_ways - exclude_one_empty = 537 := 
by
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  have h : total_ways - exclude_one_empty = 537 := sorry
  exact h

end balls_in_boxes_l190_190493


namespace number_of_valid_numbers_l190_190071

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def four_digit_number_conditions : Prop :=
  (∀ N : ℕ, 7000 ≤ N ∧ N < 9000 → 
    (N % 5 = 0) →
    (∃ a b c d : ℕ, 
      N = 1000 * a + 100 * b + 10 * c + d ∧
      (a = 7 ∨ a = 8) ∧
      (d = 0 ∨ d = 5) ∧
      3 ≤ b ∧ is_prime b ∧ b < c ∧ c ≤ 7))

theorem number_of_valid_numbers : four_digit_number_conditions → 
  (∃ n : ℕ, n = 24) :=
  sorry

end number_of_valid_numbers_l190_190071


namespace R2_perfect_fit_l190_190746

variables {n : ℕ} (x y : Fin n → ℝ) (b a : ℝ)

-- Condition: Observations \( (x_i, y_i) \) such that \( y_i = bx_i + a \)
def observations (i : Fin n) : Prop :=
  y i = b * x i + a

-- Condition: \( e_i = 0 \) for all \( i \)
def no_error (i : Fin n) : Prop := (b * x i + a + 0 = y i)

theorem R2_perfect_fit (h_obs: ∀ i, observations x y b a i)
                       (h_no_error: ∀ i, no_error x y b a i) : R_squared = 1 := by
  sorry

end R2_perfect_fit_l190_190746


namespace student_l190_190564

-- Definition of the conditions
def mistaken_calculation (x : ℤ) : ℤ :=
  x + 10

def correct_calculation (x : ℤ) : ℤ :=
  x + 5

-- Theorem statement: Prove that the student's result is 10 more than the correct result
theorem student's_error {x : ℤ} : mistaken_calculation x = correct_calculation x + 5 :=
by
  sorry

end student_l190_190564


namespace recycling_points_l190_190577

-- Define the statement
theorem recycling_points : 
  ∀ (C H L I : ℝ) (points_per_six_pounds : ℝ), 
  C = 28 → H = 4.5 → L = 3.25 → I = 8.75 → points_per_six_pounds = 1 / 6 →
  (⌊ C * points_per_six_pounds ⌋ + ⌊ I * points_per_six_pounds ⌋  + ⌊ H * points_per_six_pounds ⌋ + ⌊ L * points_per_six_pounds ⌋ = 5) :=
by
  intros C H L I pps hC hH hL hI hpps
  rw [hC, hH, hL, hI, hpps]
  simp
  sorry

end recycling_points_l190_190577


namespace collinear_points_sum_xy_solution_l190_190002

theorem collinear_points_sum_xy_solution (x y : ℚ)
  (h1 : (B : ℚ × ℚ) = (-2, y))
  (h2 : (A : ℚ × ℚ) = (x, 5))
  (h3 : (C : ℚ × ℚ) = (1, 1))
  (h4 : dist (B.1, B.2) (C.1, C.2) = 2 * dist (A.1, A.2) (C.1, C.2))
  (h5 : (y - 5) / (-2 - x) = (1 - 5) / (1 - x)) :
  x + y = -9 / 2 ∨ x + y = 17 / 2 :=
by sorry

end collinear_points_sum_xy_solution_l190_190002


namespace find_coordinates_of_P_l190_190373

/-- Let the curve C be defined by the equation y = x^3 - 10x + 3 and point P lies on this curve in the second quadrant.
We are given that the slope of the tangent line to the curve at point P is 2. We need to find the coordinates of P.
--/
theorem find_coordinates_of_P :
  ∃ (x y : ℝ), (y = x ^ 3 - 10 * x + 3) ∧ (3 * x ^ 2 - 10 = 2) ∧ (x < 0) ∧ (x = -2) ∧ (y = 15) :=
by
  sorry

end find_coordinates_of_P_l190_190373


namespace opposite_of_neg2023_l190_190673

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l190_190673


namespace root_value_l190_190204

theorem root_value (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) : m * (2 * m - 7) + 5 = 4 := by
  sorry

end root_value_l190_190204


namespace units_digit_div_product_l190_190970

theorem units_digit_div_product :
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 :=
by
  sorry

end units_digit_div_product_l190_190970


namespace problem1_problem2_l190_190885

-- Definitions of sets A and B
def A : Set ℝ := { x | x > 1 }
def B (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Problem 1:
theorem problem1 (a : ℝ) : B a ⊆ A → 1 ≤ a :=
  sorry

-- Problem 2:
theorem problem2 (a : ℝ) : (A ∩ B a).Nonempty → 0 < a :=
  sorry

end problem1_problem2_l190_190885


namespace solution_set_abs_le_one_inteval_l190_190417

theorem solution_set_abs_le_one_inteval (x : ℝ) : |x| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

end solution_set_abs_le_one_inteval_l190_190417


namespace john_shots_l190_190568

theorem john_shots :
  let initial_shots := 30
  let initial_percentage := 0.60
  let additional_shots := 10
  let final_percentage := 0.58
  let made_initial := initial_percentage * initial_shots
  let total_shots := initial_shots + additional_shots
  let made_total := final_percentage * total_shots
  let made_additional := made_total - made_initial
  made_additional = 5 :=
by
  sorry

end john_shots_l190_190568


namespace fibonacci_series_sum_l190_190916

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)

noncomputable def sum_fibonacci_fraction : ℚ :=
  ∑' (n : ℕ), (fibonacci n : ℚ) / (5^n : ℚ)

theorem fibonacci_series_sum : sum_fibonacci_fraction = 5 / 19 := by
  sorry

end fibonacci_series_sum_l190_190916


namespace jane_earnings_two_weeks_l190_190081

def num_chickens : ℕ := 10
def num_eggs_per_chicken_per_week : ℕ := 6
def dollars_per_dozen : ℕ := 2
def dozens_in_12_eggs : ℕ := 12

theorem jane_earnings_two_weeks :
  (num_chickens * num_eggs_per_chicken_per_week * 2 / dozens_in_12_eggs * dollars_per_dozen) = 20 := by
  sorry

end jane_earnings_two_weeks_l190_190081


namespace range_of_a_l190_190623

theorem range_of_a 
  (a x y : ℝ)
  (h1 : 2 * x + y = 3 - a)
  (h2 : x + 2 * y = 4 + 2 * a)
  (h3 : x + y < 1) :
  a < -4 := sorry

end range_of_a_l190_190623


namespace geometric_sequence_ratio_l190_190381

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l190_190381


namespace eval_f_l190_190640

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem eval_f : f (f (1 / 2)) = 1 :=
by
  sorry

end eval_f_l190_190640


namespace distance_origin_to_point_l190_190898

theorem distance_origin_to_point : 
  let origin := (0, 0)
  let point := (8, 15)
  dist origin point = 17 :=
by
  let dist (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  sorry

end distance_origin_to_point_l190_190898


namespace solve_eq1_solve_eq2_l190_190872

-- Definition of the first equation
def eq1 (x : ℝ) : Prop := (1 / 2) * x^2 - 8 = 0

-- Definition of the second equation
def eq2 (x : ℝ) : Prop := (x - 5)^3 = -27

-- Proof statement for the value of x in the first equation
theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 4 ∨ x = -4 := by
  sorry

-- Proof statement for the value of x in the second equation
theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l190_190872


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l190_190223

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l190_190223


namespace money_total_l190_190567

theorem money_total (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 350) (h3 : C = 100) : A + B + C = 450 :=
by {
  sorry
}

end money_total_l190_190567


namespace complement_of_A_l190_190091

/-
Given:
1. Universal set U = {0, 1, 2, 3, 4}
2. Set A = {1, 2}

Prove:
C_U A = {0, 3, 4}
-/

section
  variable (U : Set ℕ) (A : Set ℕ)
  variable (hU : U = {0, 1, 2, 3, 4})
  variable (hA : A = {1, 2})

  theorem complement_of_A (C_UA : Set ℕ) (hCUA : C_UA = {0, 3, 4}) : 
    {x ∈ U | x ∉ A} = C_UA :=
  by
    sorry
end

end complement_of_A_l190_190091


namespace smallest_t_satisfies_equation_l190_190471

def satisfies_equation (t x y : ℤ) : Prop :=
  (x^2 + y^2)^2 + 2 * t * x * (x^2 + y^2) = t^2 * y^2

theorem smallest_t_satisfies_equation : ∃ t x y : ℤ, t > 0 ∧ x > 0 ∧ y > 0 ∧ satisfies_equation t x y ∧
  ∀ t' x' y' : ℤ, t' > 0 ∧ x' > 0 ∧ y' > 0 ∧ satisfies_equation t' x' y' → t' ≥ t :=
sorry

end smallest_t_satisfies_equation_l190_190471


namespace inequality_abc_l190_190399

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := 
sorry

end inequality_abc_l190_190399


namespace edge_length_approx_17_1_l190_190698

-- Define the base dimensions of the rectangular vessel
def length_base : ℝ := 20
def width_base : ℝ := 15

-- Define the rise in water level
def rise_water_level : ℝ := 16.376666666666665

-- Calculate the area of the base
def area_base : ℝ := length_base * width_base

-- Calculate the volume of the cube (which is equal to the volume of water displaced)
def volume_cube : ℝ := area_base * rise_water_level

-- Calculate the edge length of the cube
def edge_length_cube : ℝ := volume_cube^(1/3)

-- Statement: The edge length of the cube is approximately 17.1 cm
theorem edge_length_approx_17_1 : abs (edge_length_cube - 17.1) < 0.1 :=
by sorry

end edge_length_approx_17_1_l190_190698


namespace bushes_for_60_zucchinis_l190_190392

/-- 
Given:
1. Each blueberry bush yields twelve containers of blueberries.
2. Four containers of blueberries can be traded for three pumpkins.
3. Six pumpkins can be traded for five zucchinis.

Prove that eight bushes are needed to harvest 60 zucchinis.
-/
theorem bushes_for_60_zucchinis (bush_to_containers : ℕ) (containers_to_pumpkins : ℕ) (pumpkins_to_zucchinis : ℕ) :
  (bush_to_containers = 12) → (containers_to_pumpkins = 4) → (pumpkins_to_zucchinis = 6) →
  ∃ bushes_needed, bushes_needed = 8 ∧ (60 * pumpkins_to_zucchinis / 5 * containers_to_pumpkins / 3 / bush_to_containers) = bushes_needed :=
by
  intros h1 h2 h3
  sorry

end bushes_for_60_zucchinis_l190_190392


namespace opposite_of_neg_2023_l190_190670

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l190_190670


namespace find_nat_numbers_l190_190465

theorem find_nat_numbers (a b : ℕ) (c : ℕ) (h : ∀ n : ℕ, a^n + b^n = c^(n+1)) : a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_nat_numbers_l190_190465


namespace each_person_bid_count_l190_190313

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l190_190313


namespace nigella_base_salary_is_3000_l190_190258

noncomputable def nigella_base_salary : ℝ :=
  let house_A_cost := 60000
  let house_B_cost := 3 * house_A_cost
  let house_C_cost := (2 * house_A_cost) - 110000
  let commission_A := 0.02 * house_A_cost
  let commission_B := 0.02 * house_B_cost
  let commission_C := 0.02 * house_C_cost
  let total_earnings := 8000
  let total_commission := commission_A + commission_B + commission_C
  total_earnings - total_commission

theorem nigella_base_salary_is_3000 : 
  nigella_base_salary = 3000 :=
by sorry

end nigella_base_salary_is_3000_l190_190258


namespace largest_square_perimeter_is_28_l190_190985

-- Definitions and assumptions
def rect_length : ℝ := 10
def rect_width : ℝ := 7

-- Define the largest possible square
def largest_square_side := rect_width

-- Define the perimeter of a square
def perimeter_of_square (side : ℝ) : ℝ := 4 * side

-- Proving statement
theorem largest_square_perimeter_is_28 :
  perimeter_of_square largest_square_side = 28 := 
  by 
    -- sorry is used to skip the proof
    sorry

end largest_square_perimeter_is_28_l190_190985


namespace B_and_C_have_together_l190_190449

theorem B_and_C_have_together
  (A B C : ℕ)
  (h1 : A + B + C = 700)
  (h2 : A + C = 300)
  (h3 : C = 200) :
  B + C = 600 := by
  sorry

end B_and_C_have_together_l190_190449


namespace valid_integer_pairs_l190_190487

theorem valid_integer_pairs :
  { (x, y) : ℤ × ℤ |
    (∃ α β : ℝ, α^2 + β^2 < 4 ∧ α + β = (-x : ℝ) ∧ α * β = y ∧ x^2 - 4 * y ≥ 0) } =
  {(-2,1), (-1,-1), (-1,0), (0, -1), (0,0), (1,0), (1,-1), (2,1)} :=
sorry

end valid_integer_pairs_l190_190487


namespace tennis_racket_weight_l190_190111

theorem tennis_racket_weight 
  (r b : ℝ)
  (h1 : 10 * r = 8 * b)
  (h2 : 4 * b = 120) :
  r = 24 :=
by
  sorry

end tennis_racket_weight_l190_190111


namespace fraction_comparison_l190_190420

theorem fraction_comparison : (9 / 16) > (5 / 9) :=
by {
  sorry -- the detailed proof is not required for this task
}

end fraction_comparison_l190_190420


namespace units_digit_a_2017_l190_190756

noncomputable def a_n (n : ℕ) : ℝ :=
  (Real.sqrt 2 + 1) ^ n - (Real.sqrt 2 - 1) ^ n

theorem units_digit_a_2017 : (Nat.floor (a_n 2017)) % 10 = 2 :=
  sorry

end units_digit_a_2017_l190_190756


namespace ratio_of_second_to_first_l190_190966

theorem ratio_of_second_to_first:
  ∀ (x y z : ℕ), 
  (y = 90) → 
  (z = 4 * y) → 
  ((x + y + z) / 3 = 165) → 
  (y / x = 2) := 
by 
  intros x y z h1 h2 h3
  sorry

end ratio_of_second_to_first_l190_190966


namespace coordinates_at_5PM_l190_190559

noncomputable def particle_coords_at_5PM : ℝ × ℝ :=
  let t1 : ℝ := 7  -- 7 AM
  let t2 : ℝ := 9  -- 9 AM
  let t3 : ℝ := 17  -- 5 PM in 24-hour format
  let coord1 : ℝ × ℝ := (1, 2)
  let coord2 : ℝ × ℝ := (3, -2)
  let dx : ℝ := (coord2.1 - coord1.1) / (t2 - t1)
  let dy : ℝ := (coord2.2 - coord1.2) / (t2 - t1)
  (coord2.1 + dx * (t3 - t2), coord2.2 + dy * (t3 - t2))

theorem coordinates_at_5PM
  (t1 t2 t3 : ℝ)
  (coord1 coord2 : ℝ × ℝ)
  (h_t1 : t1 = 7)
  (h_t2 : t2 = 9)
  (h_t3 : t3 = 17)
  (h_coord1 : coord1 = (1, 2))
  (h_coord2 : coord2 = (3, -2))
  (h_dx : (coord2.1 - coord1.1) / (t2 - t1) = 1)
  (h_dy : (coord2.2 - coord1.2) / (t2 - t1) = -2)
  : particle_coords_at_5PM = (11, -18) :=
by
  sorry

end coordinates_at_5PM_l190_190559


namespace solve_fractional_equation_l190_190806

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x / (x - 1) = 3) ↔ x = 3 := 
by
  sorry

end solve_fractional_equation_l190_190806


namespace problem_l190_190173

def op (x y : ℝ) : ℝ := x^2 + y^3

theorem problem (k : ℝ) : op k (op k k) = k^2 + k^6 + 6*k^7 + k^9 :=
by
  sorry

end problem_l190_190173


namespace complement_union_l190_190611

open Set

variable (U : Set ℕ := {0, 1, 2, 3, 4}) (A : Set ℕ := {1, 2, 3}) (B : Set ℕ := {2, 4})

theorem complement_union (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) : 
  (U \ A ∪ B) = {0, 2, 4} :=
by
  sorry

end complement_union_l190_190611


namespace flour_qualification_l190_190994

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end flour_qualification_l190_190994


namespace lcm_hcf_product_l190_190767

theorem lcm_hcf_product (A B : ℕ) (h_prod : A * B = 18000) (h_hcf : Nat.gcd A B = 30) : Nat.lcm A B = 600 :=
sorry

end lcm_hcf_product_l190_190767


namespace compare_squares_l190_190851

theorem compare_squares (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ (a + b) / 2 * (a + b) / 2 := 
sorry

end compare_squares_l190_190851


namespace Amanda_tickets_third_day_l190_190709

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l190_190709


namespace group_division_l190_190279

theorem group_division (total_students groups_per_group : ℕ) (h1 : total_students = 30) (h2 : groups_per_group = 5) : 
  (total_students / groups_per_group) = 6 := 
by 
  sorry

end group_division_l190_190279


namespace survey_population_l190_190167

-- Definitions based on conditions
def number_of_packages := 10
def dozens_per_package := 10
def sets_per_dozen := 12

-- Derived from conditions
def total_sets := number_of_packages * dozens_per_package * sets_per_dozen

-- Populations for the proof
def population_quality : ℕ := total_sets
def population_satisfaction : ℕ := total_sets

-- Proof statement
theorem survey_population:
  (population_quality = 1200) ∧ (population_satisfaction = 1200) := by
  sorry

end survey_population_l190_190167


namespace flight_duration_l190_190852

theorem flight_duration (takeoff landing : ℕ) (h : ℕ) (m : ℕ)
  (h0 : takeoff = 11 * 60 + 7)
  (h1 : landing = 2 * 60 + 49 + 12 * 60)
  (h2 : 0 < m) (h3 : m < 60) :
  h + m = 45 := 
sorry

end flight_duration_l190_190852


namespace log_base_16_of_4_eq_half_l190_190733

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l190_190733


namespace joan_gave_sam_43_seashells_l190_190377

def joan_original_seashells : ℕ := 70
def joan_seashells_left : ℕ := 27
def seashells_given_to_sam : ℕ := 43

theorem joan_gave_sam_43_seashells :
  joan_original_seashells - joan_seashells_left = seashells_given_to_sam :=
by
  sorry

end joan_gave_sam_43_seashells_l190_190377


namespace least_number_of_tablets_l190_190298

theorem least_number_of_tablets (tablets_A : ℕ) (tablets_B : ℕ) (hA : tablets_A = 10) (hB : tablets_B = 13) :
  ∃ n, ((tablets_A ≤ 10 → n ≥ tablets_A + 2) ∧ (tablets_B ≤ 13 → n ≥ tablets_B + 2)) ∧ n = 12 :=
by
  sorry

end least_number_of_tablets_l190_190298


namespace find_b_for_intersection_l190_190571

theorem find_b_for_intersection (b : ℝ) :
  (∀ x : ℝ, bx^2 + 2 * x + 3 = 3 * x + 4 → bx^2 - x - 1 = 0) →
  (∀ x : ℝ, x^2 * b - x - 1 = 0 → (1 + 4 * b = 0) → b = -1/4) :=
by
  intros h_eq h_discriminant h_solution
  sorry

end find_b_for_intersection_l190_190571


namespace bike_shop_profit_l190_190084

theorem bike_shop_profit :
  let tire_repair_charge := 20
  let tire_repair_cost := 5
  let tire_repairs_per_month := 300
  let complex_repair_charge := 300
  let complex_repair_cost := 50
  let complex_repairs_per_month := 2
  let retail_profit := 2000
  let fixed_expenses := 4000
  let total_tire_profit := tire_repairs_per_month * (tire_repair_charge - tire_repair_cost)
  let total_complex_profit := complex_repairs_per_month * (complex_repair_charge - complex_repair_cost)
  let total_income := total_tire_profit + total_complex_profit + retail_profit
  let final_profit := total_income - fixed_expenses
  final_profit = 3000 :=
by
  sorry

end bike_shop_profit_l190_190084


namespace angle_HCF_l190_190638

-- Using necessary imports to bring in geometric constructs from Mathlib

open EuclideanGeometry

axiom exists_circle_inter (A B : Point) : ∃ (w : Circle), A ∈ w ∧ B ∈ w
axiom cuts_circle (D E : Point) (w : Circle) : ∃ (w' : Circle), D ∈ w' ∧ E ∈ w' ∧ w ∩ w' = {D, E}
axiom tangent_to_circle (C : Point) (w : Circle) : ∃ (w' : Circle), C ∈ w ∧ is_tangent(C, w')
axiom symmetric_point (F G : Point) : ∃ (H : Point), is_symmetric(F, G, H)

theorem angle_HCF (A B C D E F G H : Point) (w1 w2 w3 : Circle)
    (h1 : A ∈ w1 ∧ B ∈ w1)
    (h2 : C ∈ w2 ∧ is_tangent(C, w3))
    (h3 : D ∈ w3 ∧ E ∈ w3 ∧ D ≠ E)
    (h4 : F ∈ line_through A B ∧ is_tangent(F, line_through A B))
    (h5 : G ∈ line_through D E ∧ G ∈ line_through A B)
    (h6 : H = symmetric_point(F, G)) :
    inner_angle H C F = 90° :=
sorry

end angle_HCF_l190_190638


namespace min_value_fraction_expr_l190_190473

theorem min_value_fraction_expr : ∀ (x : ℝ), x > 0 → (4 + x) * (1 + x) / x ≥ 9 :=
by
  sorry

end min_value_fraction_expr_l190_190473


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l190_190188

-- Define a predicate for consecutive prime numbers
def is_consecutive_primes (a b c d : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧
  (b = a + 1 ∨ b = a + 2) ∧
  (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2)

-- Define the main problem statement
theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (a b c d : ℕ), is_consecutive_primes a b c d ∧ (a + b + c + d) % 5 = 0 ∧ ∀ (w x y z : ℕ), is_consecutive_primes w x y z ∧ (w + x + y + z) % 5 = 0 → a + b + c + d ≤ w + x + y + z :=
sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l190_190188


namespace geometric_sequence_ratio_l190_190382

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l190_190382


namespace square_side_length_l190_190958

theorem square_side_length (s : ℝ) (h : s^2 = 12 * s) : s = 12 :=
by
  sorry

end square_side_length_l190_190958


namespace sector_area_is_2pi_l190_190895

/-- Problem Statement: Prove that the area of a sector of a circle with radius 4 and central
    angle 45° (or π/4 radians) is 2π. -/
theorem sector_area_is_2pi (r : ℝ) (θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_is_2pi_l190_190895


namespace pascal_current_speed_l190_190393

variable (v : ℝ)
variable (h₁ : v > 0) -- current speed is positive

-- Conditions
variable (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16)

-- Proving the speed
theorem pascal_current_speed (h₁ : v > 0) (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16) : v = 8 :=
sorry

end pascal_current_speed_l190_190393


namespace smallest_angle_in_triangle_l190_190120

theorem smallest_angle_in_triangle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180) : 
  3 * k = 45 := 
by sorry

end smallest_angle_in_triangle_l190_190120


namespace rationalize_denominator_correct_l190_190940

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end rationalize_denominator_correct_l190_190940


namespace problem1_problem2_l190_190714

-- Problem 1: Prove that (x + y + z)² - (x + y - z)² = 4z(x + y) for x, y, z ∈ ℝ
theorem problem1 (x y z : ℝ) : (x + y + z)^2 - (x + y - z)^2 = 4 * z * (x + y) := 
sorry

-- Problem 2: Prove that (a + 2b)² - 2(a + 2b)(a - 2b) + (a - 2b)² = 16b² for a, b ∈ ℝ
theorem problem2 (a b : ℝ) : (a + 2 * b)^2 - 2 * (a + 2 * b) * (a - 2 * b) + (a - 2 * b)^2 = 16 * b^2 := 
sorry

end problem1_problem2_l190_190714


namespace jan_clean_car_water_l190_190243

def jan_water_problem
  (initial_water : ℕ)
  (car_water : ℕ)
  (plant_additional : ℕ)
  (plate_clothes_water : ℕ)
  (remaining_water : ℕ)
  (used_water : ℕ)
  (car_cleaning_water : ℕ) : Prop :=
  initial_water = 65 ∧
  plate_clothes_water = 24 ∧
  plant_additional = 11 ∧
  remaining_water = 2 * plate_clothes_water ∧
  used_water = initial_water - remaining_water ∧
  car_water = used_water + plant_additional ∧
  car_cleaning_water = car_water / 4

theorem jan_clean_car_water : jan_water_problem 65 17 11 24 48 17 7 :=
by {
  sorry
}

end jan_clean_car_water_l190_190243


namespace square_ratio_l190_190447

theorem square_ratio (x y : ℝ) (hx : x = 60 / 17) (hy : y = 780 / 169) : 
  x / y = 169 / 220 :=
by
  sorry

end square_ratio_l190_190447


namespace calculate_a3_l190_190197

theorem calculate_a3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S n = 2^n - 1) (h2 : ∀ n, a n = S n - S (n-1)) : 
  a 3 = 4 :=
by
  sorry

end calculate_a3_l190_190197


namespace train_length_calculation_l190_190552

theorem train_length_calculation (len1 : ℝ) (speed1_kmph : ℝ) (speed2_kmph : ℝ) (crossing_time : ℝ) (len2 : ℝ) :
  len1 = 120.00001 → 
  speed1_kmph = 120 → 
  speed2_kmph = 80 → 
  crossing_time = 9 → 
  (len1 + len2) = ((speed1_kmph * 1000 / 3600 + speed2_kmph * 1000 / 3600) * crossing_time) → 
  len2 = 379.99949 :=
by
  intros hlen1 hspeed1 hspeed2 htime hdistance
  sorry

end train_length_calculation_l190_190552


namespace point_A_in_first_quadrant_l190_190076

def point_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_A_in_first_quadrant : point_in_first_quadrant 1 2 := by
  sorry

end point_A_in_first_quadrant_l190_190076


namespace rationalize_denominator_l190_190936

theorem rationalize_denominator : 
  ∃ A B C : ℤ, C > 0 ∧ ∃ k : ℕ, B = k ∧ (∀ p, nat.prime p → p^3 ∣ k → false) ∧ 
  (5:ℚ) / (3 * (real.cbrt 7)) = (A * real.cbrt B : ℚ) / C ∧ A + B + C = 75 :=
by 
  sorry

end rationalize_denominator_l190_190936


namespace length_of_QY_is_31_l190_190695

theorem length_of_QY_is_31
  (P Q R : Point)
  (X Y Z : Point)
  (O₄ O₅ O₆ : Point)
  (h_inscribed : P ∈ Segment YZ ∧ Q ∈ Segment XZ ∧ R ∈ Segment XY)
  (h_circle_centers : circumcenter △PYZ = O₄ ∧ circumcenter △QXR = O₅ ∧ circumcenter △RQP = O₆)
  (XY YZ XZ : ℚ)
  (h_sides_length : XY = 29 ∧ YZ = 35 ∧ XZ = 28)
  (h_arcs : length_arc YR = length_arc QZ ∧ length_arc XR = length_arc PY ∧ length_arc XP = length_arc QY)
  (h_QY_form : ∃ p q : ℕ, p.gcd q = 1 ∧ QY = p / q) :
  ∃ (p q : ℕ), p + q = 31 :=
by
  sorry

end length_of_QY_is_31_l190_190695


namespace arithmetic_sequence_seventh_term_l190_190237

/-- In an arithmetic sequence, the sum of the first three terms is 9 and the third term is 8. 
    Prove that the seventh term is 28. -/
theorem arithmetic_sequence_seventh_term :
  ∃ (a d : ℤ), (a + (a + d) + (a + 2 * d) = 9) ∧ (a + 2 * d = 8) ∧ (a + 6 * d = 28) :=
by
  sorry

end arithmetic_sequence_seventh_term_l190_190237


namespace sufficient_condition_for_inequality_l190_190066

theorem sufficient_condition_for_inequality (m : ℝ) (h : m ≠ 0) : (m > 2) → (m + 4 / m > 4) :=
by
  sorry

end sufficient_condition_for_inequality_l190_190066


namespace rationalize_denominator_l190_190929

theorem rationalize_denominator :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let A := 25
  let B := 20
  let C := 16
  let D := 1
  (1 / (a - b)) = ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / D ∧ (A + B + C + D = 62) := by
  sorry

end rationalize_denominator_l190_190929


namespace find_n_eq_5_l190_190678

variable {a_n b_n : ℕ → ℤ}

def a (n : ℕ) : ℤ := 2 + 3 * (n - 1)
def b (n : ℕ) : ℤ := -2 + 4 * (n - 1)

theorem find_n_eq_5 :
  ∃ n : ℕ, a n = b n ∧ n = 5 :=
by
  sorry

end find_n_eq_5_l190_190678


namespace max_min_values_l190_190853

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_min_values :
  ∃ x_max x_min : ℝ, x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧
    f (-2) = 24 ∧ f 2 = -6 := sorry

end max_min_values_l190_190853


namespace find_unknown_number_l190_190359

theorem find_unknown_number (x n : ℚ) (h1 : n + 7/x = 6 - 5/x) (h2 : x = 12) : n = 5 :=
by
  sorry

end find_unknown_number_l190_190359


namespace no_a_satisfy_quadratic_equation_l190_190740

theorem no_a_satisfy_quadratic_equation :
  ∀ (a : ℕ), (a > 0) ∧ (a ≤ 100) ∧
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ * x₂ = 2 * a^2 ∧ x₁ + x₂ = -(3*a + 1)) → false := by
  sorry

end no_a_satisfy_quadratic_equation_l190_190740


namespace like_terms_sum_l190_190889

theorem like_terms_sum (m n : ℕ) (h1 : m + 1 = 1) (h2 : 3 = n) : m + n = 3 :=
by sorry

end like_terms_sum_l190_190889


namespace find_a_and_b_l190_190761

theorem find_a_and_b (a b : ℚ) (h : ∀ (n : ℕ), 1 / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) : 
  a = 1/2 ∧ b = -1/2 := 
by 
  sorry

end find_a_and_b_l190_190761


namespace Amanda_ticket_sales_goal_l190_190711

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l190_190711


namespace equivalence_cond_indep_l190_190022

variables {Ω : Type*} 
variables [measurable_space Ω]
variables {μ : measure Ω}
variables {𝒜 𝒞 𝒷 : measurable_space Ω}

def cond_indep (𝒜 𝒷 𝒞 : measurable_space Ω) : Prop :=
  ∀ A ∈ 𝒜.sets, ∀ B ∈ 𝒷.sets, condexp (𝒞.measurable_space) (A ∩ B) = condexp (𝒞.measurable_space) A * condexp (𝒞.measurable_space) B

theorem equivalence_cond_indep (𝒜 𝒷 𝒞 : measurable_space Ω) :
  (∀ A ∈ 𝒜.sets, condexp (𝒷.measurable_space ⊔ 𝒞.measurable_space) A = condexp (𝒞.measurable_space) A) ↔
  (∀ X : Ω → ℝ, measurable[𝒜] X → integrable[μ] X → condexp (𝒷.measurable_space ⊔ 𝒞.measurable_space) X = condexp (𝒞.measurable_space) X) ↔
  (∀ A ∈ (generate_from (𝒜.pi_sets)).sets, condexp (𝒷.measurable_space ⊔ 𝒞.measurable_space) A = condexp (𝒞.measurable_space) A) ↔
  (cond_indep 𝒜 𝒷 𝒞) :=
sorry

end equivalence_cond_indep_l190_190022


namespace diagonal_of_rectangular_prism_l190_190704

noncomputable def diagonal_length (l w h : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2 + h^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 15 25 20 = 25 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_rectangular_prism_l190_190704


namespace neither_sufficient_nor_necessary_l190_190490

-- For given real numbers x and y
-- Prove the statement "at least one of x and y is greater than 1" is not necessary and not sufficient for x^2 + y^2 > 2.
noncomputable def at_least_one_gt_one (x y : ℝ) : Prop := (x > 1) ∨ (y > 1)
def sum_of_squares_gt_two (x y : ℝ) : Prop := x^2 + y^2 > 2

theorem neither_sufficient_nor_necessary (x y : ℝ) :
  ¬(at_least_one_gt_one x y → sum_of_squares_gt_two x y) ∧ ¬(sum_of_squares_gt_two x y → at_least_one_gt_one x y) :=
by
  sorry

end neither_sufficient_nor_necessary_l190_190490


namespace find_m_l190_190227

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : (4 - m) / (m - 2) = 2 * m) : 
  m = (3 + Real.sqrt 41) / 4 := by
  sorry

end find_m_l190_190227


namespace part_one_part_two_l190_190332

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem part_one :
  ∀ x m : ℕ, f x ≤ -m^2 + 6 * m → 1 ≤ m ∧ m ≤ 5 := 
by
  sorry

theorem part_two (a b c : ℝ) (h : 3 * a + 4 * b + 5 * c = 1) :
  (a^2 + b^2 + c^2) ≥ (1 / 50) :=
by
  sorry

end part_one_part_two_l190_190332


namespace wendy_percentage_accounting_l190_190427

noncomputable def years_as_accountant : ℕ := 25
noncomputable def years_as_manager : ℕ := 15
noncomputable def total_lifespan : ℕ := 80

def total_years_in_accounting : ℕ := years_as_accountant + years_as_manager

def percentage_of_life_in_accounting : ℝ := (total_years_in_accounting / total_lifespan) * 100

theorem wendy_percentage_accounting : percentage_of_life_in_accounting = 50 := by
  unfold total_years_in_accounting
  unfold percentage_of_life_in_accounting
  sorry

end wendy_percentage_accounting_l190_190427


namespace gcd_triples_l190_190325

theorem gcd_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  gcd a 20 = b ∧ gcd b 15 = c ∧ gcd a c = 5 ↔
  ∃ t : ℕ, t > 0 ∧ 
    ((a = 20 * t ∧ b = 20 ∧ c = 5) ∨ 
     (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨ 
     (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by
  sorry

end gcd_triples_l190_190325


namespace min_length_MN_l190_190803

theorem min_length_MN (a b : ℝ) (H h : ℝ) (MN : ℝ) (midsegment_eq_4 : (a + b) / 2 = 4)
    (area_div_eq_half : (a + MN) / 2 * h = (MN + b) / 2 * H) : MN = 4 :=
by
  sorry

end min_length_MN_l190_190803


namespace max_min_page_difference_l190_190533

-- Define the number of pages in each book
variables (Poetry Documents Rites Changes SpringAndAutumn : ℤ)

-- Define the conditions as given in the problem
axiom h1 : abs (Poetry - Documents) = 24
axiom h2 : abs (Documents - Rites) = 17
axiom h3 : abs (Rites - Changes) = 27
axiom h4 : abs (Changes - SpringAndAutumn) = 19
axiom h5 : abs (SpringAndAutumn - Poetry) = 15

-- Assertion to prove
theorem max_min_page_difference : 
  ∃ a b c d e : ℤ, a = Poetry ∧ b = Documents ∧ c = Rites ∧ d = Changes ∧ e = SpringAndAutumn ∧ 
  abs (a - b) = 24 ∧ abs (b - c) = 17 ∧ abs (c - d) = 27 ∧ abs (d - e) = 19 ∧ abs (e - a) = 15 ∧ 
  (max a (max b (max c (max d e))) - min a (min b (min c (min d e)))) = 34 :=
by {
  sorry
}

end max_min_page_difference_l190_190533


namespace jaymee_is_22_l190_190911

-- Definitions based on the problem conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 2 + 2 * shara_age

-- The theorem we need to prove
theorem jaymee_is_22 : jaymee_age = 22 :=
by
  sorry

end jaymee_is_22_l190_190911


namespace log_10_850_consecutive_integers_l190_190809

theorem log_10_850_consecutive_integers : 
  (2:ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < (3:ℝ) →
  ∃ (a b : ℕ), (a = 2) ∧ (b = 3) ∧ (2 < Real.log 850 / Real.log 10) ∧ (Real.log 850 / Real.log 10 < 3) ∧ (a + b = 5) :=
by
  sorry

end log_10_850_consecutive_integers_l190_190809


namespace cost_of_notebook_l190_190232

theorem cost_of_notebook (s n c : ℕ) 
    (h1 : s > 18) 
    (h2 : n ≥ 2) 
    (h3 : c > n) 
    (h4 : s * c * n = 2376) : 
    c = 11 := 
  sorry

end cost_of_notebook_l190_190232


namespace markup_constant_relationship_l190_190368

variable (C S : ℝ) (k : ℝ)
variable (fractional_markup : k * S = 0.25 * C)
variable (relation : S = C + k * S)

theorem markup_constant_relationship (fractional_markup : k * S = 0.25 * C) (relation : S = C + k * S) :
  k = 1 / 5 :=
by
  sorry

end markup_constant_relationship_l190_190368


namespace total_number_of_sweets_l190_190681

theorem total_number_of_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) (total_sweets : ℕ) 
  (h1 : num_crates = 4) (h2 : sweets_per_crate = 16) : total_sweets = 64 := by
  sorry

end total_number_of_sweets_l190_190681


namespace count_perfect_squares_and_cubes_l190_190225

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l190_190225


namespace library_books_count_l190_190446

def students_per_day : List ℕ := [4, 5, 6, 9]
def books_per_student : ℕ := 5
def total_books_given (students : List ℕ) (books_per_student : ℕ) : ℕ :=
  students.foldl (λ acc n => acc + n * books_per_student) 0

theorem library_books_count :
  total_books_given students_per_day books_per_student = 120 :=
by
  sorry

end library_books_count_l190_190446


namespace tan_value_of_point_on_graph_l190_190230

theorem tan_value_of_point_on_graph (a : ℝ) (h : (4 : ℝ) ^ (1/2) = a) : 
  Real.tan ((a / 6) * Real.pi) = Real.sqrt 3 :=
by 
  sorry

end tan_value_of_point_on_graph_l190_190230


namespace sasha_salt_factor_l190_190138

theorem sasha_salt_factor (x y : ℝ) : 
  (y = 2 * x) →
  (x + y = 2 * x + y / 2) →
  (3 * x / (2 * x) = 1.5) :=
by
  intros h₁ h₂
  sorry

end sasha_salt_factor_l190_190138


namespace more_than_10_weights_missing_l190_190413

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end more_than_10_weights_missing_l190_190413


namespace exp_sum_is_neg_one_l190_190321

noncomputable def sumExpExpressions : ℂ :=
  (Complex.exp (Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 7) +
   Complex.exp (3 * Real.pi * Complex.I / 7) +
   Complex.exp (4 * Real.pi * Complex.I / 7) +
   Complex.exp (5 * Real.pi * Complex.I / 7) +
   Complex.exp (6 * Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 9) +
   Complex.exp (4 * Real.pi * Complex.I / 9) +
   Complex.exp (6 * Real.pi * Complex.I / 9) +
   Complex.exp (8 * Real.pi * Complex.I / 9) +
   Complex.exp (10 * Real.pi * Complex.I / 9) +
   Complex.exp (12 * Real.pi * Complex.I / 9) +
   Complex.exp (14 * Real.pi * Complex.I / 9) +
   Complex.exp (16 * Real.pi * Complex.I / 9))

theorem exp_sum_is_neg_one : sumExpExpressions = -1 := by
  sorry

end exp_sum_is_neg_one_l190_190321


namespace domain_eq_l190_190361

theorem domain_eq (f : ℝ → ℝ) : 
  (∀ x : ℝ, -1 ≤ 3 - 2 * x ∧ 3 - 2 * x ≤ 2) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 :=
by sorry

end domain_eq_l190_190361


namespace solve_system_of_equations_solve_system_of_inequalities_l190_190268

-- Proof for the system of equations
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 32) 
  (h2 : 2 * x - y = 0) :
  x = 8 ∧ y = 16 :=
by
  sorry

-- Proof for the system of inequalities
theorem solve_system_of_inequalities (x : ℝ)
  (h3 : 3 * x - 1 < 5 - 2 * x)
  (h4 : 5 * x + 1 ≥ 2 * x + 3) :
  (2 / 3 : ℝ) ≤ x ∧ x < (6 / 5 : ℝ) :=
by
  sorry

end solve_system_of_equations_solve_system_of_inequalities_l190_190268


namespace coffee_mug_cost_l190_190835

theorem coffee_mug_cost (bracelet_cost gold_heart_necklace_cost total_change total_money_spent : ℤ)
    (bracelets_count gold_heart_necklace_count mugs_count : ℤ)
    (h_bracelet_cost : bracelet_cost = 15)
    (h_gold_heart_necklace_cost : gold_heart_necklace_cost = 10)
    (h_total_change : total_change = 15)
    (h_total_money_spent : total_money_spent = 100)
    (h_bracelets_count : bracelets_count = 3)
    (h_gold_heart_necklace_count : gold_heart_necklace_count = 2)
    (h_mugs_count : mugs_count = 1) :
    mugs_count * ((total_money_spent - total_change) - (bracelets_count * bracelet_cost + gold_heart_necklace_count * gold_heart_necklace_cost)) = 20 :=
by
  sorry

end coffee_mug_cost_l190_190835


namespace ellipse_find_m_l190_190747

theorem ellipse_find_m (a b m e : ℝ) 
  (h1 : a^2 = 4) 
  (h2 : b^2 = m)
  (h3 : e = 1/2) :
  m = 3 := 
by
  sorry

end ellipse_find_m_l190_190747


namespace bids_per_person_l190_190318

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l190_190318


namespace price_of_book_l190_190976

-- Definitions based on the problem conditions
def money_xiaowang_has (p : ℕ) : ℕ := 2 * p - 6
def money_xiaoli_has (p : ℕ) : ℕ := 2 * p - 31

def combined_money (p : ℕ) : ℕ := money_xiaowang_has p + money_xiaoli_has p

-- Lean statement to prove the price of each book
theorem price_of_book (p : ℕ) : combined_money p = 3 * p → p = 37 :=
by
  sorry

end price_of_book_l190_190976


namespace ab_le_1_e2_l190_190479

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.log x - a * x - b

theorem ab_le_1_e2 {a b : ℝ} (h : 0 < a) (hx : ∃ x : ℝ, 0 < x ∧ f x a b ≥ 0) : a * b ≤ 1 / Real.exp 2 :=
sorry

end ab_le_1_e2_l190_190479


namespace no_even_and_increasing_function_l190_190239

-- Definition of a function being even
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Definition of a function being increasing
def is_increasing_function (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem stating the non-existence of a function that is both even and increasing
theorem no_even_and_increasing_function : ¬ ∃ f : ℝ → ℝ, is_even_function f ∧ is_increasing_function f :=
by
  sorry

end no_even_and_increasing_function_l190_190239


namespace willie_bananas_l190_190136

variable (W : ℝ) 

theorem willie_bananas (h1 : 35.0 - 14.0 = 21.0) (h2: W + 35.0 = 83.0) : 
  W = 48.0 :=
by
  sorry

end willie_bananas_l190_190136


namespace initial_velocity_is_three_l190_190166

noncomputable def displacement (t : ℝ) : ℝ :=
  3 * t - t^2

theorem initial_velocity_is_three : 
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_is_three_l190_190166


namespace solve_equation_l190_190525

theorem solve_equation :
  (∃ x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3) → (x = -1) :=
by
  sorry

end solve_equation_l190_190525


namespace solve_problem_l190_190295

-- Define the variables and conditions
def problem_statement : Prop :=
  ∃ x : ℕ, 865 * 48 = 240 * x ∧ x = 173

-- Statement to prove
theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l190_190295


namespace f_2010_eq_0_l190_190346

theorem f_2010_eq_0 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, f (x + 2) = f x) : 
  f 2010 = 0 :=
by sorry

end f_2010_eq_0_l190_190346


namespace rationalize_denominator_correct_l190_190941

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end rationalize_denominator_correct_l190_190941


namespace profit_per_meter_is_35_l190_190841

-- defining the conditions
def meters_sold : ℕ := 85
def selling_price : ℕ := 8925
def cost_price_per_meter : ℕ := 70
def total_cost_price := cost_price_per_meter * meters_sold
def total_selling_price := selling_price
def total_profit := total_selling_price - total_cost_price
def profit_per_meter := total_profit / meters_sold

-- Theorem stating the profit per meter of cloth
theorem profit_per_meter_is_35 : profit_per_meter = 35 := 
by
  sorry

end profit_per_meter_is_35_l190_190841


namespace simplify_expression_l190_190796

theorem simplify_expression : (2 * 3 * b * 4 * (b ^ 2) * 5 * (b ^ 3) * 6 * (b ^ 4)) = 720 * (b ^ 10) :=
by
  sorry

end simplify_expression_l190_190796


namespace problem_a_b_c_ge_neg2_l190_190786

theorem problem_a_b_c_ge_neg2 {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1 / b > -2) ∨ (b + 1 / c > -2) ∨ (c + 1 / a > -2) → False :=
by
  sorry

end problem_a_b_c_ge_neg2_l190_190786


namespace sector_angle_l190_190604

theorem sector_angle (r θ : ℝ) 
  (h1 : r * θ + 2 * r = 6) 
  (h2 : 1/2 * r^2 * θ = 2) : 
  θ = 1 ∨ θ = 4 :=
by 
  sorry

end sector_angle_l190_190604


namespace complement_of_angle_l190_190353

theorem complement_of_angle (α : ℝ) (h : α = 23 + 36 / 60) : 180 - α = 156.4 := 
by
  sorry

end complement_of_angle_l190_190353


namespace find_m_l190_190757

-- Let's define the sets A and B.
def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- We'll state the problem as a theorem
theorem find_m (m : ℝ) (h : B m ⊆ A) : m = 1 ∨ m = -1 :=
by sorry

end find_m_l190_190757


namespace solve_inequalities_l190_190867

theorem solve_inequalities :
  {x : ℝ | 4 ≤ (2*x) / (3*x - 7) ∧ (2*x) / (3*x - 7) < 9} = {x : ℝ | (63 / 25) < x ∧ x ≤ 2.8} :=
by
  sorry

end solve_inequalities_l190_190867


namespace pizza_eaten_after_six_trips_l190_190287

noncomputable def fraction_eaten : ℚ :=
  let first_trip := 1 / 3
  let second_trip := 1 / (3 ^ 2)
  let third_trip := 1 / (3 ^ 3)
  let fourth_trip := 1 / (3 ^ 4)
  let fifth_trip := 1 / (3 ^ 5)
  let sixth_trip := 1 / (3 ^ 6)
  first_trip + second_trip + third_trip + fourth_trip + fifth_trip + sixth_trip

theorem pizza_eaten_after_six_trips : fraction_eaten = 364 / 729 :=
by sorry

end pizza_eaten_after_six_trips_l190_190287


namespace cattle_selling_price_per_pound_correct_l190_190079

def purchase_price : ℝ := 40000
def cattle_count : ℕ := 100
def feed_cost_percentage : ℝ := 0.20
def weight_per_head : ℕ := 1000
def profit : ℝ := 112000

noncomputable def total_feed_cost : ℝ := purchase_price * feed_cost_percentage
noncomputable def total_cost : ℝ := purchase_price + total_feed_cost
noncomputable def total_revenue : ℝ := total_cost + profit
def total_weight : ℕ := cattle_count * weight_per_head
noncomputable def selling_price_per_pound : ℝ := total_revenue / total_weight

theorem cattle_selling_price_per_pound_correct :
  selling_price_per_pound = 1.60 := by
  sorry

end cattle_selling_price_per_pound_correct_l190_190079


namespace inequality_proof_l190_190397

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256 * x * y * z :=
by
  -- Proof goes here
  sorry

end inequality_proof_l190_190397


namespace fraction_remains_unchanged_l190_190501

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) :=
by {
  sorry
}

end fraction_remains_unchanged_l190_190501


namespace gcd_1029_1437_5649_l190_190869

theorem gcd_1029_1437_5649 : Nat.gcd (Nat.gcd 1029 1437) 5649 = 3 := by
  sorry

end gcd_1029_1437_5649_l190_190869


namespace sides_of_polygons_l190_190829

theorem sides_of_polygons (p : ℕ) (γ : ℝ) (n1 n2 : ℕ) (h1 : p = 5) (h2 : γ = 12 / 7) 
    (h3 : n2 = n1 + p) 
    (h4 : 360 / n1 - 360 / n2 = γ) : 
    n1 = 30 ∧ n2 = 35 := 
  sorry

end sides_of_polygons_l190_190829


namespace sqrt_of_0_09_l190_190810

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end sqrt_of_0_09_l190_190810


namespace equilateral_triangle_octagon_area_ratio_l190_190986

theorem equilateral_triangle_octagon_area_ratio
  (s_t s_o : ℝ)
  (h_triangle_area : (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2)) :
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by
  sorry

end equilateral_triangle_octagon_area_ratio_l190_190986


namespace opposite_of_neg_2023_l190_190668

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l190_190668


namespace speed_of_boat_is_15_l190_190680

noncomputable def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 / 5 ∧ (x + 3) * t = 3.6 ∧ x = 15

theorem speed_of_boat_is_15 (x : ℝ) (t : ℝ) (rate_of_current : ℝ) (distance_downstream : ℝ) :
  rate_of_current = 3 →
  distance_downstream = 3.6 →
  t = 1 / 5 →
  (x + rate_of_current) * t = distance_downstream →
  x = 15 :=
by
  intros h1 h2 h3 h4
  -- proof goes here
  sorry

end speed_of_boat_is_15_l190_190680


namespace polynomial_divisibility_l190_190255

theorem polynomial_divisibility (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) : 
    ∃ k : ℕ, k ≥ 1 ∧ (a^2 + 3 * a * b + 3 * b^2 - 1) % k^3 = 0 :=
    sorry

end polynomial_divisibility_l190_190255


namespace correct_f_l190_190646

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom functional_equation (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem correct_f (x : ℝ) : f x = x + 1 := sorry

end correct_f_l190_190646


namespace triangle_side_length_l190_190507

theorem triangle_side_length (A : ℝ) (b : ℝ) (S : ℝ) (hA : A = 120) (hb : b = 4) (hS: S = 2 * Real.sqrt 3) : 
  ∃ c : ℝ, c = 2 := 
by 
  sorry

end triangle_side_length_l190_190507


namespace find_g_26_l190_190116

variable {g : ℕ → ℕ}

theorem find_g_26 (hg : ∀ x, g (x + g x) = 5 * g x) (h1 : g 1 = 5) : g 26 = 120 :=
  sorry

end find_g_26_l190_190116


namespace pens_in_each_pack_l190_190780

-- Given the conditions
def Kendra_packs : ℕ := 4
def Tony_packs : ℕ := 2
def pens_kept_each : ℕ := 2
def friends : ℕ := 14

-- Theorem statement
theorem pens_in_each_pack : ∃ (P : ℕ), Kendra_packs * P + Tony_packs * P - pens_kept_each * 2 - friends = 0 ∧ P = 3 := by
  sorry

end pens_in_each_pack_l190_190780


namespace min_value_reciprocal_sum_l190_190513

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end min_value_reciprocal_sum_l190_190513


namespace midpoint_PQ_on_line_OT_range_TF_PQ_l190_190199

variable {a b : ℝ}
variable {F P Q T : ℝ × ℝ}
variable {O : ℝ × ℝ := (0, 0)}
variable C : ℝ → ℝ → ℝ

noncomputable def ellipse : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def on_line_OT (G T : ℝ × ℝ) : Prop :=
  (T.1 - O.1) * (G.2 - O.2) = (T.2 - O.2) * (G.1 - O.1)

theorem midpoint_PQ_on_line_OT {a b : ℝ} (h : 0 < b ∧ b < a ∧ a > 0) (h₁ : 2 * real.sqrt (a^2 - b^2) = 2)
  (h₂a : midpoint P Q = G) (h₂b : on_line_OT G T) :
  true :=
by
	sorry

theorem range_TF_PQ {F P Q T : ℝ × ℝ} (h : 0 < b ∧ b < a ∧ a > 0) :
  set.range (λ x : ℝ, abs (real.sqrt ((4 - 1)^2 + (x - 0)^2) / (real.sqrt (1 + x^2) * real.sqrt ((-6*x / (3*x^2 + 4))^2 - 4 * (-9 / (3*x^2 + 4)))))) = set.Ici 1 :=
 by
	sorry

end midpoint_PQ_on_line_OT_range_TF_PQ_l190_190199


namespace gcd_problem_l190_190717

def a : ℕ := 101^5 + 1
def b : ℕ := 101^5 + 101^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := by
  sorry

end gcd_problem_l190_190717


namespace arithmetic_square_root_of_nine_l190_190530

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l190_190530


namespace sufficient_but_not_necessary_l190_190494

variable (a : ℝ)

theorem sufficient_but_not_necessary : (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 → False) :=
by
  sorry

end sufficient_but_not_necessary_l190_190494


namespace parabola_behavior_l190_190347

theorem parabola_behavior (x : ℝ) (h : x < 0) : ∃ y, y = 2*x^2 - 1 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 0 ∧ x2 < 0 → (2*x1^2 - 1) > (2*x2^2 - 1) :=
by
  sorry

end parabola_behavior_l190_190347


namespace tan_value_l190_190209

open Real

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

noncomputable def arithmetic_seq (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem tan_value
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : geometric_seq a)
  (hb : arithmetic_seq b)
  (h_geom : a 0 * a 5 * a 10 = -3 * sqrt 3)
  (h_arith : b 0 + b 5 + b 10 = 7 * π) :
  tan ((b 2 + b 8) / (1 - a 3 * a 7)) = -sqrt 3 :=
sorry

end tan_value_l190_190209


namespace find_certain_number_l190_190147

theorem find_certain_number (x : ℝ) (h : 34 = (4/5) * x + 14) : x = 25 :=
by
  sorry

end find_certain_number_l190_190147


namespace translation_result_l190_190903

-- Define the original point A
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 3, y := -2 }

-- Define the translation function
def translate_right (p : Point) (dx : ℤ) : Point :=
  { x := p.x + dx, y := p.y }

-- Prove that translating point A 2 units to the right gives point A'
theorem translation_result :
  translate_right A 2 = { x := 5, y := -2 } :=
by sorry

end translation_result_l190_190903


namespace common_ratio_geometric_series_l190_190868

theorem common_ratio_geometric_series 
  (a : ℚ) (b : ℚ) (r : ℚ)
  (h_a : a = 4 / 5)
  (h_b : b = -5 / 12)
  (h_r : r = b / a) :
  r = -25 / 48 :=
by sorry

end common_ratio_geometric_series_l190_190868


namespace total_points_always_odd_l190_190949

theorem total_points_always_odd (n : ℕ) (h : n ≥ 1) :
  ∀ k : ℕ, ∃ m : ℕ, m = (2 ^ k * (n + 1) - 1) ∧ m % 2 = 1 :=
by
  sorry

end total_points_always_odd_l190_190949


namespace no_real_solution_implies_a_range_l190_190769

noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 4 * x + a^2

theorem no_real_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, quadratic a x ≤ 0 → false) ↔ a < -2 ∨ a > 2 := 
sorry

end no_real_solution_implies_a_range_l190_190769


namespace travel_time_correct_l190_190093

def luke_bus_to_work : ℕ := 70
def paula_bus_to_work : ℕ := (70 * 3) / 5
def jane_train_to_work : ℕ := 120
def michael_cycle_to_work : ℕ := 120 / 4

def luke_bike_back_home : ℕ := 70 * 5
def paula_bus_back_home: ℕ := paula_bus_to_work
def jane_train_back_home : ℕ := 120 * 2
def michael_cycle_back_home : ℕ := michael_cycle_to_work

def luke_total_travel : ℕ := luke_bus_to_work + luke_bike_back_home
def paula_total_travel : ℕ := paula_bus_to_work + paula_bus_back_home
def jane_total_travel : ℕ := jane_train_to_work + jane_train_back_home
def michael_total_travel : ℕ := michael_cycle_to_work + michael_cycle_back_home

def total_travel_time : ℕ := luke_total_travel + paula_total_travel + jane_total_travel + michael_total_travel

theorem travel_time_correct : total_travel_time = 924 :=
by sorry

end travel_time_correct_l190_190093


namespace marissas_sunflower_height_in_meters_l190_190516

-- Define the conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Define the given data
def sister_height_feet : ℝ := 4.15
def additional_height_cm : ℝ := 37
def height_difference_inches : ℝ := 63

-- Calculate the height of Marissa's sunflower in meters
theorem marissas_sunflower_height_in_meters :
  let sister_height_inches := sister_height_feet * inches_per_foot
  let sister_height_cm := sister_height_inches * cm_per_inch
  let total_sister_height_cm := sister_height_cm + additional_height_cm
  let height_difference_cm := height_difference_inches * cm_per_inch
  let marissas_sunflower_height_cm := total_sister_height_cm + height_difference_cm
  let marissas_sunflower_height_m := marissas_sunflower_height_cm / cm_per_meter
  marissas_sunflower_height_m = 3.23512 :=
by
  sorry

end marissas_sunflower_height_in_meters_l190_190516


namespace quadruple_solution_l190_190174

theorem quadruple_solution (x y z w : ℝ) (h1: x + y + z + w = 0) (h2: x^7 + y^7 + z^7 + w^7 = 0) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨ (x = -y ∧ z = -w) ∨ (x = -z ∧ y = -w) ∨ (x = -w ∧ y = -z) :=
by
  sorry

end quadruple_solution_l190_190174


namespace perfect_squares_and_cubes_l190_190218

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l190_190218


namespace ratio_of_speeds_l190_190129

noncomputable def speed_ratios (d t_b t : ℚ) : ℚ × ℚ  :=
  let d_b := t_b * t
  let d_a := d - d_b
  let t_h := t / 60
  let s_a := d_a / t_h
  let s_b := t_b
  (s_a / 15, s_b / 15)

theorem ratio_of_speeds
  (d : ℚ) (s_b : ℚ) (t : ℚ)
  (h : d = 88) (h1 : s_b = 90) (h2 : t = 32) :
  speed_ratios d s_b t = (5, 6) :=
  by
  sorry

end ratio_of_speeds_l190_190129


namespace rationalize_denominator_sum_l190_190933

theorem rationalize_denominator_sum :
  ∃ A B C : ℤ,
  C > 0 ∧
  (∃ p : ℤ, p > 1 ∧ p * p * p ∣ B → false) ∧
  (∃ t : ℝ, t = (5 : ℝ) / (3 * real.cbrt 7) ∧
   t = (A : ℝ) * real.cbrt (B : ℝ) / (C : ℝ)) ∧
  (A + B + C = 75) :=
sorry

end rationalize_denominator_sum_l190_190933


namespace speed_of_boat_l190_190148

-- Given conditions
variables (V_b : ℝ) (V_s : ℝ) (T : ℝ) (D : ℝ)

-- Problem statement in Lean
theorem speed_of_boat (h1 : V_s = 5) (h2 : T = 1) (h3 : D = 45) :
  D = T * (V_b + V_s) → V_b = 40 := 
by
  intro h4
  rw [h1, h2, h3] at h4
  linarith

end speed_of_boat_l190_190148


namespace range_of_m_l190_190067

theorem range_of_m (m x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : 5 * x + 2 * y = 6 * m + 7) 
  (h3 : 2 * x - y < 19) : 
  m < 3 / 2 := 
sorry

end range_of_m_l190_190067


namespace part_I_part_II_part_III_l190_190605

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem part_I (a : ℝ) : (∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f a x ≥ f a 1) ↔ a ≥ -1/2 :=
by
  sorry

theorem part_II : ∀ x : ℝ, f (-Real.exp 1) x + 2 ≤ 0 :=
by
  sorry

theorem part_III : ¬ ∃ x : ℝ, |f (-Real.exp 1) x| = Real.log x / x + 3 / 2 :=
by
  sorry

end part_I_part_II_part_III_l190_190605


namespace ratio_jl_jm_l190_190521

-- Define the side length of the square NOPQ as s
variable (s : ℝ)

-- Define the length (l) and width (m) of the rectangle JKLM
variable (l m : ℝ)

-- Conditions given in the problem
variable (area_overlap : ℝ)
variable (area_condition1 : area_overlap = 0.25 * s * s)
variable (area_condition2 : area_overlap = 0.40 * l * m)

theorem ratio_jl_jm (h1 : area_overlap = 0.25 * s * s) (h2 : area_overlap = 0.40 * l * m) : l / m = 2 / 5 :=
by
  sorry

end ratio_jl_jm_l190_190521


namespace abs_gt_implies_nec_not_suff_l190_190743

theorem abs_gt_implies_nec_not_suff {a b : ℝ} : 
  (|a| > b) → (∀ (a b : ℝ), a > b → |a| > b) ∧ ¬(∀ (a b : ℝ), |a| > b → a > b) :=
by
  sorry

end abs_gt_implies_nec_not_suff_l190_190743


namespace starfish_arms_l190_190715

variable (x : ℕ)

theorem starfish_arms :
  (7 * x + 14 = 49) → (x = 5) := by
  sorry

end starfish_arms_l190_190715


namespace gcd_7_fact_10_fact_div_4_fact_eq_5040_l190_190847

def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

noncomputable def quotient_fact (a b : ℕ) : ℕ := fact a / fact b

theorem gcd_7_fact_10_fact_div_4_fact_eq_5040 :
  Nat.gcd (fact 7) (quotient_fact 10 4) = 5040 := by
sorry

end gcd_7_fact_10_fact_div_4_fact_eq_5040_l190_190847


namespace cylinder_radius_in_cone_l190_190705

-- Define the conditions
def cone_diameter := 18
def cone_height := 20
def cylinder_height_eq_diameter {r : ℝ} := 2 * r

-- Define the theorem to prove
theorem cylinder_radius_in_cone : ∃ r : ℝ, r = 90 / 19 ∧ (20 - 2 * r) / r = 20 / 9 :=
by
  sorry

end cylinder_radius_in_cone_l190_190705


namespace factorize_poly1_factorize_poly2_l190_190585

variable (a b m n : ℝ)

theorem factorize_poly1 : 3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2 :=
sorry

theorem factorize_poly2 : 4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n) :=
sorry

end factorize_poly1_factorize_poly2_l190_190585


namespace each_person_bids_five_times_l190_190310

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l190_190310


namespace simple_interest_principal_l190_190183

theorem simple_interest_principal (A r t : ℝ) (ht_pos : t > 0) (hr_pos : r > 0) (hA_pos : A > 0) :
  (A = 1120) → (r = 0.08) → (t = 2.4) → ∃ (P : ℝ), abs (P - 939.60) < 0.01 :=
by
  intros hA hr ht
  -- Proof would go here
  sorry

end simple_interest_principal_l190_190183


namespace option_B_is_valid_distribution_l190_190284

def is_probability_distribution (p : List ℚ) : Prop :=
  p.sum = 1 ∧ ∀ x ∈ p, 0 < x ∧ x ≤ 1

theorem option_B_is_valid_distribution : is_probability_distribution [1/2, 1/3, 1/6] :=
by
  sorry

end option_B_is_valid_distribution_l190_190284


namespace probability_is_one_over_145_l190_190538

-- Define the domain and properties
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even (n : ℕ) : Prop :=
  n % 2 = 0

-- Total number of ways to pick 2 distinct numbers from 1 to 30
def total_ways_to_pick_two_distinct : ℕ :=
  (30 * 29) / 2

-- Calculate prime numbers between 1 and 30
def primes_from_1_to_30 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Filter valid pairs where both numbers are prime and at least one of them is 2
def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 7), (2, 11), (2, 13), (2, 17), (2, 19), (2, 23), (2, 29)]

def count_valid_pairs (l : List (ℕ × ℕ)) : ℕ :=
  l.length

-- Probability calculation
def probability_prime_and_even : ℚ :=
  count_valid_pairs (valid_pairs primes_from_1_to_30) / total_ways_to_pick_two_distinct

-- Prove that the probability is 1/145
theorem probability_is_one_over_145 : probability_prime_and_even = 1 / 145 :=
by
  sorry

end probability_is_one_over_145_l190_190538


namespace bids_per_person_l190_190317

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l190_190317


namespace find_ordered_pair_l190_190182

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 7 * x - 30 * y = 3) 
  (h2 : 3 * y - x = 5) : 
  x = -53 / 3 ∧ y = -38 / 9 :=
sorry

end find_ordered_pair_l190_190182


namespace mangoes_count_l190_190109

noncomputable def total_fruits : ℕ := 58
noncomputable def pears : ℕ := 10
noncomputable def pawpaws : ℕ := 12
noncomputable def lemons : ℕ := 9
noncomputable def kiwi : ℕ := 9

theorem mangoes_count (mangoes : ℕ) : 
  (pears + pawpaws + lemons + kiwi + mangoes = total_fruits) → 
  mangoes = 18 :=
by
  sorry

end mangoes_count_l190_190109


namespace correct_operation_c_l190_190973

theorem correct_operation_c (a b : ℝ) :
  ¬ (a^2 + a^2 = 2 * a^4)
  ∧ ¬ ((-3 * a * b^2)^2 = -6 * a^2 * b^4)
  ∧ a^6 / (-a)^2 = a^4
  ∧ ¬ ((a - b)^2 = a^2 - b^2) :=
by
  sorry

end correct_operation_c_l190_190973


namespace arithmetic_sequence_formula_and_sum_l190_190773

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h0 : a 1 = 1) 
  (h1 : a 3 = -3)
  (hS : ∃ k, S k = -35):
  (∀ n, a n = 3 - 2 * n) ∧ (∃ k, S k = -35 ∧ k = 7) :=
by
  -- Given an arithmetic sequence where a_1 = 1 and a_3 = -3,
  -- prove that the general formula is a_n = 3 - 2n
  -- and the sum of the first k terms S_k = -35 implies k = 7
  sorry

end arithmetic_sequence_formula_and_sum_l190_190773


namespace smallest_E_of_positive_reals_l190_190395

noncomputable def E (a b c : ℝ) : ℝ :=
  (a^3) / (1 - a^2) + (b^3) / (1 - b^2) + (c^3) / (1 - c^2)

theorem smallest_E_of_positive_reals (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  E a b c = 1 / 8 := 
sorry

end smallest_E_of_positive_reals_l190_190395


namespace weeks_to_cover_expense_l190_190724

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l190_190724


namespace add_fractions_l190_190586

theorem add_fractions : (1 / 4 : ℚ) + (3 / 5) = 17 / 20 := 
by
  sorry

end add_fractions_l190_190586


namespace solution_is_singleton_l190_190805

def solution_set : Set (ℝ × ℝ) := { (x, y) | 2 * x + y = 3 ∧ x - 2 * y = -1 }

theorem solution_is_singleton : solution_set = { (1, 1) } :=
by
  sorry

end solution_is_singleton_l190_190805


namespace seats_still_available_l190_190013

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end seats_still_available_l190_190013


namespace emir_needs_extra_money_l190_190178

def cost_dictionary : ℕ := 5
def cost_dinosaur_book : ℕ := 11
def cost_cookbook : ℕ := 5
def amount_saved : ℕ := 19

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_cookbook
def amount_needed : ℕ := total_cost - amount_saved

theorem emir_needs_extra_money : amount_needed = 2 := by
  rfl -- actual proof that amount_needed equals 2 goes here
  -- Sorry can be used to skip if the proof needs additional steps.
  sorry

end emir_needs_extra_money_l190_190178


namespace parabola_shifting_produces_k_l190_190208

theorem parabola_shifting_produces_k
  (k : ℝ)
  (h1 : -k/2 > 0)
  (h2 : (0 : ℝ) = (((0 : ℝ) - 3) + k/2)^2 - (5*k^2)/4 + 1)
  :
  k = -5 :=
sorry

end parabola_shifting_produces_k_l190_190208


namespace simplify_expression_l190_190400

theorem simplify_expression (x : ℝ) :
  x - 3 * (1 + x) + 4 * (1 - x)^2 - 5 * (1 + 3 * x) = 4 * x^2 - 25 * x - 4 := by
  sorry

end simplify_expression_l190_190400


namespace polynomial_at_one_l190_190482

def f (x : ℝ) : ℝ := x^4 - 7*x^3 - 9*x^2 + 11*x + 7

theorem polynomial_at_one :
  f 1 = 3 := 
by
  sorry

end polynomial_at_one_l190_190482


namespace find_function_satisfying_property_l190_190179

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2 * x * y)

theorem find_function_satisfying_property (f : ℝ → ℝ) (h : ∀ x, 0 ≤ f x) (hf : example_function f) :
  ∃ a : ℝ, 0 ≤ a ∧ ∀ x : ℝ, f x = a * x^2 :=
sorry

end find_function_satisfying_property_l190_190179


namespace bids_per_person_l190_190316

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l190_190316


namespace initial_customers_correct_l190_190034

def initial_customers (remaining : ℕ) (left : ℕ) : ℕ := remaining + left

theorem initial_customers_correct :
  initial_customers 12 9 = 21 :=
by
  sorry

end initial_customers_correct_l190_190034


namespace girls_count_in_leos_class_l190_190503

def leo_class_girls_count (g b : ℕ) :=
  (g / b = 3 / 4) ∧ (g + b = 35) → g = 15

theorem girls_count_in_leos_class (g b : ℕ) :
  leo_class_girls_count g b :=
by
  sorry

end girls_count_in_leos_class_l190_190503


namespace find_a_b_sum_l190_190802

-- Conditions
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f_prime (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem find_a_b_sum (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a + b = 7 := 
sorry

end find_a_b_sum_l190_190802


namespace chess_sequences_l190_190110

def binomial (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_sequences :
  binomial 11 4 = 210 := by
  sorry

end chess_sequences_l190_190110


namespace pharmacy_tubs_needed_l190_190560

theorem pharmacy_tubs_needed 
  (total_tubs_needed : ℕ) 
  (tubs_in_storage : ℕ) 
  (fraction_bought_new_vendor : ℚ) 
  (total_tubs_needed = 100) 
  (tubs_in_storage = 20)
  (fraction_bought_new_vendor = 1 / 4) :
  let tubs_needed_to_buy := total_tubs_needed - tubs_in_storage in
  let tubs_from_new_vendor := (tubs_needed_to_buy / 4 : ℕ) in
  let total_tubs_now := tubs_in_storage + tubs_from_new_vendor in
  let tubs_from_usual_vendor := total_tubs_needed - total_tubs_now in
  tubs_from_usual_vendor = 60 := 
by sorry

end pharmacy_tubs_needed_l190_190560


namespace minimum_value_function_l190_190602

variable {m n x y : ℝ}
variable (h1 : 0 < m) (h2 : 0 < n) (h3 : m ≠ n) (h4 : 0 < x) (h5 : 0 < y) (h6 : x + y ≠ 0)

theorem minimum_value_function (hx : x ∈ Ioo 0 1) :
  ∃ x₀, x₀ ∈ Ioo (0 : ℝ) 1 ∧ (∀ x ∈ Ioo (0 : ℝ) 1, f x₀ ≤ f x) ∧ f x₀ = 25 / 3 :=
by
  let f := λ x:ℝ, 4 / (3 * x) + 3 / (1 - x)
  have h7 : ∀ (x y : ℝ), 0 < x → 0 < y → ℝ, m ≠ n → 
              ( (m^2 / x) + (n^2 / y) ≥ (m+n)^2 / (x+y) ) ∧ ((m^2 / x) + (n^2 / y) = (m+n)^2 / (x+y) ↔ (m / x) = (n / y)),
  sorry

  have h8 : f(x) ≥ 25 / 3,
  sorry

  have h9 : ∃ x₀, x₀ ∈ Ioo 0 1 ∧ (∀ x ∈ Ioo 0 1, f x₀ ≤ f x),
  use 2 / 5
  simp [f]
  -- Here you can show that f(2/5) = 25 / 3 using straightforward calculations.
  sorry

  exact ⟨ 2 / 5, ⟨by norm_num, by norm_num⟩, h8, by norm_num⟩

end minimum_value_function_l190_190602


namespace div_expr_l190_190282

namespace Proof

theorem div_expr (x : ℝ) (h : x = 3.242 * 10) : x / 100 = 0.3242 := by
  sorry

end Proof

end div_expr_l190_190282


namespace point_in_first_quadrant_l190_190881

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := i * (2 - i)

-- Define a predicate that checks if a complex number is in the first quadrant
def isFirstQuadrant (x : ℂ) : Prop := x.re > 0 ∧ x.im > 0

-- State the theorem
theorem point_in_first_quadrant : isFirstQuadrant z := sorry

end point_in_first_quadrant_l190_190881


namespace cyclist_speed_ratio_l190_190127

theorem cyclist_speed_ratio
  (d : ℝ) (t₁ t₂ : ℝ) 
  (v₁ v₂ : ℝ)
  (h1 : d = 8)
  (h2 : t₁ = 4)
  (h3 : t₂ = 1)
  (h4 : d = (v₁ - v₂) * t₁)
  (h5 : d = (v₁ + v₂) * t₂) :
  v₁ / v₂ = 5 / 3 :=
sorry

end cyclist_speed_ratio_l190_190127


namespace boat_speed_is_13_l190_190554

noncomputable def boatSpeedStillWater : ℝ := 
  let Vs := 6 -- Speed of the stream in km/hr
  let time := 3.6315789473684212 -- Time taken in hours to travel 69 km downstream
  let distance := 69 -- Distance traveled in km
  (distance - Vs * time) / time

theorem boat_speed_is_13 : boatSpeedStillWater = 13 := by
  sorry

end boat_speed_is_13_l190_190554


namespace triangle_angle_opposite_c_l190_190231

theorem triangle_angle_opposite_c (a b c : ℝ) (x : ℝ) 
  (ha : a = 2) (hb : b = 2) (hc : c = 4) : x = 180 :=
by 
  -- proof steps are not required as per the instruction
  sorry

end triangle_angle_opposite_c_l190_190231


namespace problem_statement_l190_190384

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l190_190384


namespace real_solutions_l190_190187

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end real_solutions_l190_190187


namespace x_add_y_eq_neg_one_l190_190481

theorem x_add_y_eq_neg_one (x y : ℝ) (h : |x + 3| + (y - 2)^2 = 0) : x + y = -1 :=
by sorry

end x_add_y_eq_neg_one_l190_190481


namespace fraction_of_butterflies_flew_away_l190_190859

theorem fraction_of_butterflies_flew_away (original_butterflies : ℕ) (left_butterflies : ℕ) (h1 : original_butterflies = 9) (h2 : left_butterflies = 6) : (original_butterflies - left_butterflies) / original_butterflies = 1 / 3 :=
by
  sorry

end fraction_of_butterflies_flew_away_l190_190859


namespace robert_cash_spent_as_percentage_l190_190659

theorem robert_cash_spent_as_percentage 
  (raw_material_cost : ℤ) (machinery_cost : ℤ) (total_amount : ℤ) 
  (h_raw : raw_material_cost = 100) 
  (h_machinery : machinery_cost = 125) 
  (h_total : total_amount = 250) :
  ((total_amount - (raw_material_cost + machinery_cost)) * 100 / total_amount) = 10 := 
by 
  -- Proof will be filled here
  sorry

end robert_cash_spent_as_percentage_l190_190659


namespace savings_percentage_l190_190139

variable {I S : ℝ}
variable (h1 : 1.30 * I - 2 * S + I - S = 2 * (I - S))

theorem savings_percentage (h : 1.30 * I - 2 * S + I - S = 2 * (I - S)) : S = 0.30 * I :=
  by
    sorry

end savings_percentage_l190_190139


namespace qualified_products_correct_l190_190635

def defect_rate : ℝ := 0.005
def total_produced : ℝ := 18000

theorem qualified_products_correct :
  total_produced * (1 - defect_rate) = 17910 := by
  sorry

end qualified_products_correct_l190_190635


namespace boys_variance_greater_than_girls_l190_190896

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := (List.sum scores) / (scores.length : ℝ)
  List.sum (scores.map (λ x => (x - mean) ^ 2)) / (scores.length : ℝ)

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end boys_variance_greater_than_girls_l190_190896


namespace intersection_A_B_l190_190337

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end intersection_A_B_l190_190337


namespace part1_part2_part3_l190_190850

-- Part (1)
theorem part1 (m : ℝ) : (2 * m - 3) * (5 - 3 * m) = -6 * m^2 + 19 * m - 15 :=
  sorry

-- Part (2)
theorem part2 (a b : ℝ) : (3 * a^3) ^ 2 * (2 * b^2) ^ 3 / (6 * a * b) ^ 2 = 2 * a^4 * b^4 :=
  sorry

-- Part (3)
theorem part3 (a b : ℝ) : (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
  sorry

end part1_part2_part3_l190_190850


namespace log_base_16_of_4_l190_190732

theorem log_base_16_of_4 : log 16 4 = 1 / 2 := by
  sorry

end log_base_16_of_4_l190_190732


namespace ratio_of_girls_to_boys_l190_190172

-- Define conditions
def num_boys : ℕ := 40
def children_per_counselor : ℕ := 8
def num_counselors : ℕ := 20

-- Total number of children
def total_children : ℕ := num_counselors * children_per_counselor

-- Number of girls
def num_girls : ℕ := total_children - num_boys

-- The ratio of girls to boys
def girls_to_boys_ratio : ℚ := num_girls / num_boys

-- The theorem we need to prove
theorem ratio_of_girls_to_boys : girls_to_boys_ratio = 3 := by
  sorry

end ratio_of_girls_to_boys_l190_190172


namespace find_f_of_7_l190_190601

-- Defining the conditions in the problem.
variables (f : ℝ → ℝ)
variables (odd_f : ∀ x : ℝ, f (-x) = -f x)
variables (periodic_f : ∀ x : ℝ, f (x + 4) = f x)
variables (f_eqn : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = x + 2)

-- The statement of the problem, to prove f(7) = -3.
theorem find_f_of_7 : f 7 = -3 :=
by
  sorry

end find_f_of_7_l190_190601


namespace shirt_tie_combinations_l190_190799

noncomputable def shirts : ℕ := 8
noncomputable def ties : ℕ := 7
noncomputable def forbidden_combinations : ℕ := 2

theorem shirt_tie_combinations :
  shirts * ties - forbidden_combinations = 54 := by
  sorry

end shirt_tie_combinations_l190_190799


namespace largest_sum_ABC_l190_190772

theorem largest_sum_ABC (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_prod : A * B * C = 2401) :
  A + B + C ≤ 351 :=
sorry

end largest_sum_ABC_l190_190772


namespace exist_irreducible_fractions_l190_190855

theorem exist_irreducible_fractions :
  ∃ (a b : ℕ), Nat.gcd a b = 1 ∧ Nat.gcd (a + 1) b = 1 ∧ Nat.gcd (a + 1) (b + 1) = 1 :=
by
  sorry

end exist_irreducible_fractions_l190_190855


namespace cookie_store_expense_l190_190056

theorem cookie_store_expense (B D: ℝ) 
  (h₁: D = (1 / 2) * B)
  (h₂: B = D + 20):
  B + D = 60 := by
  sorry

end cookie_store_expense_l190_190056


namespace correct_operation_l190_190974

theorem correct_operation : 
  (a^2 + a^2 = 2 * a^2) = false ∧ 
  ((-3 * a * b^2)^2 = -6 * a^2 * b^4) = false ∧ 
  (a^6 / (-a)^2 = a^4) = true ∧ 
  ((a - b)^2 = a^2 - b^2) = false :=
sorry

end correct_operation_l190_190974


namespace polygons_sides_l190_190808

def sum_of_angles (x y : ℕ) : ℕ :=
(x - 2) * 180 + (y - 2) * 180

def num_diagonals (x y : ℕ) : ℕ :=
x * (x - 3) / 2 + y * (y - 3) / 2

theorem polygons_sides (x y : ℕ) (hx : x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99) 
(hs : sum_of_angles x y = 21 * (x + y + num_diagonals x y) - 39) :
x = 17 ∧ y = 3 ∨ x = 3 ∧ y = 17 :=
by
  sorry

end polygons_sides_l190_190808


namespace smallest_value_square_l190_190520

theorem smallest_value_square (z : ℂ) (hz : z.re > 0) (A : ℝ) :
  (A = 24 / 25) →
  abs ((Complex.abs z + 1 / Complex.abs z)^2 - (2 - 14 / 25)) = 0 :=
by
  sorry

end smallest_value_square_l190_190520


namespace log_base_16_of_4_l190_190734

theorem log_base_16_of_4 : 
  (16 = 2^4) →
  (4 = 2^2) →
  (∀ (b a c : ℝ), b > 0 → b ≠ 1 → c > 0 → c ≠ 1 → log b a = log c a / log c b) →
  log 16 4 = 1 / 2 :=
by
  intros h1 h2 h3
  sorry

end log_base_16_of_4_l190_190734


namespace birds_flew_up_l190_190294

theorem birds_flew_up (initial_birds new_birds total_birds : ℕ) 
    (h_initial : initial_birds = 29) 
    (h_total : total_birds = 42) : 
    new_birds = total_birds - initial_birds := 
by 
    sorry

end birds_flew_up_l190_190294


namespace hike_up_time_eq_l190_190527

variable (t : ℝ)
variable (h_rate_up : ℝ := 4)
variable (h_rate_down : ℝ := 6)
variable (total_time : ℝ := 3)

theorem hike_up_time_eq (h_rate_up_eq : h_rate_up = 4) 
                        (h_rate_down_eq : h_rate_down = 6) 
                        (total_time_eq : total_time = 3) 
                        (dist_eq : h_rate_up * t = h_rate_down * (total_time - t)) :
  t = 9 / 5 := by
  sorry

end hike_up_time_eq_l190_190527


namespace floor_x_floor_x_eq_44_iff_l190_190866

theorem floor_x_floor_x_eq_44_iff (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 44) ↔ (7.333 ≤ x ∧ x < 7.5) :=
by
  sorry

end floor_x_floor_x_eq_44_iff_l190_190866


namespace available_seats_l190_190014

/-- Two-fifths of the seats in an auditorium that holds 500 people are currently taken. --/
def seats_taken : ℕ := (2 * 500) / 5

/-- One-tenth of the seats in an auditorium that holds 500 people are broken. --/
def seats_broken : ℕ := 500 / 10

/-- Total seats in the auditorium --/
def total_seats := 500

/-- There are 500 total seats in an auditorium. Two-fifths of the seats are taken and 
one-tenth are broken. Prove that the number of seats still available is 250. --/
theorem available_seats : (total_seats - seats_taken - seats_broken) = 250 :=
by 
  sorry

end available_seats_l190_190014


namespace solve_math_problem_l190_190028

theorem solve_math_problem (x : ℕ) (h1 : x > 0) (h2 : x % 3 = 0) (h3 : x % x = 9) : x = 30 := by
  sorry

end solve_math_problem_l190_190028


namespace division_of_decimals_l190_190683

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l190_190683


namespace phillip_remaining_amount_l190_190394

-- Define the initial amount of money
def initial_amount : ℕ := 95

-- Define the amounts spent on various items
def amount_spent_on_oranges : ℕ := 14
def amount_spent_on_apples : ℕ := 25
def amount_spent_on_candy : ℕ := 6

-- Calculate the total amount spent
def total_spent : ℕ := amount_spent_on_oranges + amount_spent_on_apples + amount_spent_on_candy

-- Calculate the remaining amount of money
def remaining_amount : ℕ := initial_amount - total_spent

-- Statement to be proved
theorem phillip_remaining_amount : remaining_amount = 50 :=
by
  sorry

end phillip_remaining_amount_l190_190394


namespace coverable_hook_l190_190580

def is_coverable (m n : ℕ) : Prop :=
  ∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5)

theorem coverable_hook (m n : ℕ) : (∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5))
  ↔ is_coverable m n :=
by
  sorry

end coverable_hook_l190_190580


namespace find_length_PB_l190_190904

-- Define the conditions of the problem
variables (AC AP PB : ℝ) (x : ℝ)

-- Condition: The length of chord AC is x
def length_AC := AC = x

-- Condition: The length of segment AP is x + 1
def length_AP := AP = x + 1

-- Statement of the theorem to prove the length of segment PB
theorem find_length_PB (h_AC : length_AC AC x) (h_AP : length_AP AP x) :
  PB = 2 * x + 1 :=
sorry

end find_length_PB_l190_190904


namespace dad_strawberries_weight_l190_190649

-- Definitions for the problem
def weight_marco := 15
def total_weight := 37

-- Theorem statement
theorem dad_strawberries_weight :
  (total_weight - weight_marco = 22) :=
by
  sorry

end dad_strawberries_weight_l190_190649


namespace rational_sqrts_l190_190658

def is_rational (n : ℝ) : Prop := ∃ (q : ℚ), n = q

theorem rational_sqrts 
  (x y z : ℝ) 
  (hxr : is_rational x) 
  (hyr : is_rational y) 
  (hzr : is_rational z)
  (hw : is_rational (Real.sqrt x + Real.sqrt y + Real.sqrt z)) :
  is_rational (Real.sqrt x) ∧ is_rational (Real.sqrt y) ∧ is_rational (Real.sqrt z) :=
sorry

end rational_sqrts_l190_190658


namespace proof_A_minus_2B_eq_11_l190_190192

theorem proof_A_minus_2B_eq_11 
  (a b : ℤ)
  (hA : ∀ a b, A = 3*b^2 - 2*a^2)
  (hB : ∀ a b, B = ab - 2*b^2 - a^2) 
  (ha : a = 2) 
  (hb : b = -1) : 
  (A - 2*B = 11) :=
by
  sorry

end proof_A_minus_2B_eq_11_l190_190192


namespace sin_double_angle_eq_half_l190_190205

theorem sin_double_angle_eq_half (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : Real.sin (π / 2 + 2 * α) = Real.cos (π / 4 - α)) : 
  Real.sin (2 * α) = 1 / 2 :=
by
  sorry

end sin_double_angle_eq_half_l190_190205


namespace iggy_total_time_correct_l190_190894

noncomputable def total_time_iggy_spends : ℕ :=
  let monday_time := 3 * (10 + 1)
  let tuesday_time := 4 * (9 + 1)
  let wednesday_time := 6 * 12
  let thursday_time := 8 * (8 + 2)
  let friday_time := 3 * 10
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem iggy_total_time_correct : total_time_iggy_spends = 255 :=
by
  -- sorry at the end indicates the skipping of the actual proof elaboration.
  sorry

end iggy_total_time_correct_l190_190894


namespace distance_Q_to_EH_l190_190626

noncomputable def N : ℝ × ℝ := (3, 0)
noncomputable def E : ℝ × ℝ := (0, 6)
noncomputable def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 9
noncomputable def EH_line (y : ℝ) : Prop := y = 6

theorem distance_Q_to_EH :
  ∃ (Q : ℝ × ℝ), circle1 Q.1 Q.2 ∧ circle2 Q.1 Q.2 ∧ Q ≠ (0, 0) ∧ abs (Q.2 - 6) = 19 / 3 := sorry

end distance_Q_to_EH_l190_190626


namespace prove_triangular_cake_volume_surface_area_sum_l190_190840

def triangular_cake_volume_surface_area_sum_proof : Prop :=
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 2
  let base_area : ℝ := (1 / 2) * length * width
  let volume : ℝ := base_area * height
  let top_area : ℝ := base_area
  let side_area : ℝ := (1 / 2) * width * height
  let icing_area : ℝ := top_area + 3 * side_area
  volume + icing_area = 15

theorem prove_triangular_cake_volume_surface_area_sum : triangular_cake_volume_surface_area_sum_proof := by
  sorry

end prove_triangular_cake_volume_surface_area_sum_l190_190840


namespace find_a_l190_190893

variable (m : ℝ)

def root1 := 2 * m - 1
def root2 := m + 4

theorem find_a (h : root1 ^ 2 = root2 ^ 2) : ∃ a : ℝ, a = 9 :=
by
  sorry

end find_a_l190_190893


namespace bg_fg_ratio_l190_190506

open Real

-- Given the lengths AB, BD, AF, DF, BE, CF
def AB : ℝ := 15
def BD : ℝ := 18
def AF : ℝ := 15
def DF : ℝ := 12
def BE : ℝ := 24
def CF : ℝ := 17

-- Prove that the ratio BG : FG = 27 : 17
theorem bg_fg_ratio (BG FG : ℝ)
  (h_BG_FG : BG / FG = 27 / 17) :
  BG / FG = 27 / 17 := by
  sorry

end bg_fg_ratio_l190_190506


namespace middle_dimension_of_crate_l190_190440

theorem middle_dimension_of_crate (middle_dimension : ℝ) : 
    (∀ r : ℝ, r = 5 → ∃ w h l : ℝ, w = 5 ∧ h = 12 ∧ l = middle_dimension ∧
        (diameter = 2 * r ∧ diameter ≤ middle_dimension ∧ h ≥ 12)) → 
    middle_dimension = 10 :=
by
  sorry

end middle_dimension_of_crate_l190_190440


namespace find_quotient_l190_190135

theorem find_quotient (A : ℕ) (h : 41 = (5 * A) + 1) : A = 8 :=
by
  sorry

end find_quotient_l190_190135


namespace quadratic_completing_square_l190_190676

theorem quadratic_completing_square :
  ∃ (a b c : ℚ), a = 12 ∧ b = 6 ∧ c = 1296 ∧ 12 + 6 + 1296 = 1314 ∧
  (12 * (x + b)^2 + c = 12 * x^2 + 144 * x + 1728) :=
by
  sorry

end quadratic_completing_square_l190_190676


namespace number_of_four_digit_numbers_larger_than_2134_l190_190968

open Nat Finset

theorem number_of_four_digit_numbers_larger_than_2134 :
  let digits := {1, 2, 3, 4}
  let all_numbers := (digits.product digits).product (digits.product digits)
  let valid_numbers := all_numbers.filter (λ n, digits.card = 4 ∧ digits.val.forall (λ d, digits.count d = 1))
  let larger_than_2134 := valid_numbers.filter (λ n, n.toNat > 2134)
  larger_than_2134.card = 17 := 
by {
  -- Proof goes here
  sorry
}

end number_of_four_digit_numbers_larger_than_2134_l190_190968


namespace correct_operation_l190_190286

theorem correct_operation :
  (∀ a : ℕ, a ^ 3 * a ^ 2 = a ^ 5) ∧
  (∀ a : ℕ, a + a ^ 2 ≠ a ^ 3) ∧
  (∀ a : ℕ, 6 * a ^ 2 / (2 * a ^ 2) = 3) ∧
  (∀ a : ℕ, (3 * a ^ 2) ^ 3 ≠ 9 * a ^ 6) :=
by
  sorry

end correct_operation_l190_190286


namespace total_voters_l190_190371

theorem total_voters (x : ℝ)
  (h1 : 0.35 * x + 80 = (0.35 * x + 80) + 0.65 * x - (0.65 * x - 0.45 * (x + 80)))
  (h2 : 0.45 * (x + 80) = 0.65 * x) : 
  x + 80 = 260 := by
  -- We'll provide the proof here
  sorry

end total_voters_l190_190371


namespace pyramid_height_value_l190_190699

-- Let s be the edge length of the cube
def cube_edge_length : ℝ := 6

-- Let b be the base edge length of the square-based pyramid
def pyramid_base_edge_length : ℝ := 12

-- Let V_cube be the volume of the cube
def volume_cube (s : ℝ) : ℝ := s ^ 3

-- Let V_pyramid be the volume of the pyramid
def volume_pyramid (b h : ℝ) : ℝ := (1 / 3) * (b ^ 2) * h

-- The given volumes are equal
def volumes_equal : Prop :=
  volume_cube cube_edge_length = volume_pyramid pyramid_base_edge_length h

-- Prove that the height h of the pyramid is 4.5
theorem pyramid_height_value (h : ℝ) (cube_edge_length pyramid_base_edge_length : ℝ) (volumes_equal : Prop) :
  h = 4.5 :=
sorry

end pyramid_height_value_l190_190699


namespace rows_seating_nine_people_l190_190126

theorem rows_seating_nine_people (x y : ℕ) (h : 9 * x + 7 * y = 74) : x = 2 :=
by sorry

end rows_seating_nine_people_l190_190126


namespace part1_part2_l190_190249

-- Definitions based on the conditions
def a_i (i : ℕ) : ℕ := sorry -- Define ai's values based on the given conditions
def f (n : ℕ) : ℕ := sorry  -- Define f(n) as the number of n-digit wave numbers satisfying the given conditions

-- Prove the first part: f(10) = 3704
theorem part1 : f 10 = 3704 := sorry

-- Prove the second part: f(2008) % 13 = 10
theorem part2 : (f 2008) % 13 = 10 := sorry

end part1_part2_l190_190249


namespace drop_in_water_level_l190_190497

theorem drop_in_water_level (rise_level : ℝ) (drop_level : ℝ) 
  (h : rise_level = 1) : drop_level = -2 :=
by
  sorry

end drop_in_water_level_l190_190497


namespace least_possible_value_expression_l190_190430

theorem least_possible_value_expression :
  ∃ min_value : ℝ, ∀ x : ℝ, ((x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019) ≥ min_value ∧ min_value = 2018 :=
by
  sorry

end least_possible_value_expression_l190_190430


namespace find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l190_190429

theorem find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square :
  ∃ n : ℕ, (4^n + 5^n) = k^2 ↔ n = 1 :=
by
  sorry

end find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l190_190429


namespace slope_of_line_l190_190016

variable (x y : ℝ)

def line_equation : Prop := 4 * y = -5 * x + 8

theorem slope_of_line (h : line_equation x y) :
  ∃ m b, y = m * x + b ∧ m = -5/4 :=
by
  sorry

end slope_of_line_l190_190016


namespace total_customers_is_40_l190_190144

-- The number of tables the waiter is attending
def num_tables : ℕ := 5

-- The number of women at each table
def women_per_table : ℕ := 5

-- The number of men at each table
def men_per_table : ℕ := 3

-- The total number of customers at each table
def customers_per_table : ℕ := women_per_table + men_per_table

-- The total number of customers the waiter has
def total_customers : ℕ := num_tables * customers_per_table

theorem total_customers_is_40 : total_customers = 40 :=
by
  -- Proof goes here
  sorry

end total_customers_is_40_l190_190144


namespace trapezoid_ratio_l190_190782

-- Define the isosceles trapezoid properties and the point inside it
noncomputable def isosceles_trapezoid (r s : ℝ) (hr : r > s) (triangle_areas : List ℝ) : Prop :=
  triangle_areas = [2, 3, 4, 5]

-- Define the problem statement
theorem trapezoid_ratio (r s : ℝ) (hr : r > s) (areas : List ℝ) (hareas : isosceles_trapezoid r s hr areas) :
  r / s = 2 + Real.sqrt 2 := sorry

end trapezoid_ratio_l190_190782


namespace arithmetic_square_root_of_9_l190_190528

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end arithmetic_square_root_of_9_l190_190528


namespace least_possible_value_of_c_l190_190818

theorem least_possible_value_of_c (a b c : ℕ) 
  (h1 : a + b + c = 60) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : b = a + 13) : c = 45 :=
sorry

end least_possible_value_of_c_l190_190818


namespace fraction_of_oil_sent_to_production_l190_190168

-- Definitions based on the problem's conditions
def initial_concentration : ℝ := 0.02
def replacement_concentration1 : ℝ := 0.03
def replacement_concentration2 : ℝ := 0.015
def final_concentration : ℝ := 0.02

-- Main theorem stating the fraction x is 1/2
theorem fraction_of_oil_sent_to_production (x : ℝ) (hx : x > 0) :
  (initial_concentration + (replacement_concentration1 - initial_concentration) * x) * (1 - x) +
  replacement_concentration2 * x = final_concentration →
  x = 0.5 :=
  sorry

end fraction_of_oil_sent_to_production_l190_190168


namespace polynomial_value_at_2_l190_190682

def f (x : ℕ) : ℕ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem polynomial_value_at_2 : f 2 = 1397 := by
  sorry

end polynomial_value_at_2_l190_190682


namespace range_of_a_l190_190755

open Real

noncomputable def f (x : ℝ) : ℝ := abs (log x)

noncomputable def g (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 1 then 0 
  else abs (x^2 - 4) - 2

noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |h x| = a → has_four_real_roots : Prop) ↔ (1 ≤ a ∧ a < 2 - log 2) := sorry

end range_of_a_l190_190755


namespace average_power_heater_l190_190234

structure Conditions where
  (M : ℝ)    -- mass of the piston
  (tau : ℝ)  -- time period τ
  (a : ℝ)    -- constant acceleration
  (c : ℝ)    -- specific heat at constant volume
  (R : ℝ)    -- universal gas constant

theorem average_power_heater (cond : Conditions) : 
  let P := cond.M * cond.a^2 * cond.tau / 2 * (1 + cond.c / cond.R)
  P = (cond.M * cond.a^2 * cond.tau / 2) * (1 + cond.c / cond.R) :=
by
  sorry

end average_power_heater_l190_190234


namespace remainder_of_max_6_multiple_no_repeated_digits_l190_190250

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end remainder_of_max_6_multiple_no_repeated_digits_l190_190250


namespace problem_solution_l190_190469

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1)^2

theorem problem_solution :
  (∀ x : ℝ, (0 < x ∧ x ≤ 5) → x ≤ f x ∧ f x ≤ 2 * |x - 1| + 1) →
  (f 1 = 4 * (1 / 4) + 1) →
  (∃ (t m : ℝ), m > 1 ∧ 
               (∀ x : ℝ, (1 ≤ x ∧ x ≤ m) → f t ≤ (1 / 4) * (x + t + 1)^2)) →
  (1 / 4 = 1 / 4) ∧ (m = 2) :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l190_190469


namespace smallest_angle_of_triangle_l190_190121

theorem smallest_angle_of_triangle : ∀ (k : ℝ), (3 * k) + (4 * k) + (5 * k) = 180 →
  3 * (180 / 12 : ℝ) = 45 :=
by
  intro k h
  rw [← h]
  field_simp [k]
  norm_num

end smallest_angle_of_triangle_l190_190121


namespace stack_trays_height_l190_190156

theorem stack_trays_height
  (thickness : ℕ)
  (top_diameter : ℕ)
  (bottom_diameter : ℕ)
  (decrement_step : ℕ)
  (base_height : ℕ)
  (cond1 : thickness = 2)
  (cond2 : top_diameter = 30)
  (cond3 : bottom_diameter = 8)
  (cond4 : decrement_step = 2)
  (cond5 : base_height = 2) :
  (bottom_diameter + decrement_step * (top_diameter - bottom_diameter) / decrement_step * thickness + base_height) = 26 :=
by
  sorry

end stack_trays_height_l190_190156


namespace polynomial_divisibility_l190_190181

theorem polynomial_divisibility (n : ℕ) (h : n > 2) : 
    (∀ k : ℕ, n = 3 * k + 1) ↔ ∃ (k : ℕ), n = 3 * k + 1 := 
sorry

end polynomial_divisibility_l190_190181


namespace number_of_valid_sequences_l190_190196

-- Define the sequence and conditions
def is_valid_sequence (a : Fin 9 → ℝ) : Prop :=
  a 0 = 1 ∧ a 8 = 1 ∧
  ∀ i : Fin 8, (a (i + 1) / a i) ∈ ({2, 1, -1/2} : Set ℝ)

-- The main problem statement
theorem number_of_valid_sequences : ∃ n, n = 491 ∧ ∀ a : Fin 9 → ℝ, is_valid_sequence a ↔ n = 491 := 
sorry

end number_of_valid_sequences_l190_190196


namespace dan_spent_more_on_chocolates_l190_190323

def price_candy_bar : ℝ := 4
def number_of_candy_bars : ℕ := 5
def candy_discount : ℝ := 0.20
def discount_threshold : ℕ := 3
def price_chocolate : ℝ := 6
def number_of_chocolates : ℕ := 4
def chocolate_tax_rate : ℝ := 0.05

def candy_cost_total : ℝ :=
  let cost_without_discount := number_of_candy_bars * price_candy_bar
  if number_of_candy_bars >= discount_threshold
  then cost_without_discount * (1 - candy_discount)
  else cost_without_discount

def chocolate_cost_total : ℝ :=
  let cost_without_tax := number_of_chocolates * price_chocolate
  cost_without_tax * (1 + chocolate_tax_rate)

def difference_in_spending : ℝ :=
  chocolate_cost_total - candy_cost_total

theorem dan_spent_more_on_chocolates :
  difference_in_spending = 9.20 :=
by
  sorry

end dan_spent_more_on_chocolates_l190_190323


namespace coefficient_a9_of_polynomial_l190_190363

theorem coefficient_a9_of_polynomial (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a_0 + 
    a_1 * (x + 1) + 
    a_2 * (x + 1)^2 + 
    a_3 * (x + 1)^3 + 
    a_4 * (x + 1)^4 + 
    a_5 * (x + 1)^5 + 
    a_6 * (x + 1)^6 + 
    a_7 * (x + 1)^7 + 
    a_8 * (x + 1)^8 + 
    a_9 * (x + 1)^9 + 
    a_10 * (x + 1)^10) 
  → a_9 = -10 :=
by
  intro h
  sorry

end coefficient_a9_of_polynomial_l190_190363


namespace pipe_network_renovation_l190_190819

theorem pipe_network_renovation 
  (total_length : Real)
  (efficiency_increase : Real)
  (days_ahead_of_schedule : Nat)
  (days_completed : Nat)
  (total_period : Nat)
  (original_daily_renovation : Real)
  (additional_renovation : Real)
  (h1 : total_length = 3600)
  (h2 : efficiency_increase = 20 / 100)
  (h3 : days_ahead_of_schedule = 10)
  (h4 : days_completed = 20)
  (h5 : total_period = 40)
  (h6 : (3600 / original_daily_renovation) - (3600 / (1.2 * original_daily_renovation)) = 10)
  (h7 : 20 * (72 + additional_renovation) >= 3600 - 1440) :
  (1.2 * original_daily_renovation = 72) ∧ (additional_renovation >= 36) :=
by
  sorry

end pipe_network_renovation_l190_190819


namespace more_than_10_weights_missing_l190_190414

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end more_than_10_weights_missing_l190_190414


namespace jane_earnings_in_two_weeks_l190_190083

-- Define the conditions in the lean environment
def number_of_chickens : ℕ := 10
def eggs_per_chicken_per_week : ℕ := 6
def selling_price_per_dozen : ℕ := 2

-- Statement of the proof problem
theorem jane_earnings_in_two_weeks :
  (number_of_chickens * eggs_per_chicken_per_week * 2) / 12 * selling_price_per_dozen = 20 :=
by
  sorry

end jane_earnings_in_two_weeks_l190_190083


namespace positive_difference_of_fraction_results_l190_190043

theorem positive_difference_of_fraction_results :
  let a := 8
  let expr1 := (a ^ 2 - a ^ 2) / a
  let expr2 := (a ^ 2 * a ^ 2) / a
  expr1 = 0 ∧ expr2 = 512 ∧ (expr2 - expr1) = 512 := 
by
  sorry

end positive_difference_of_fraction_results_l190_190043


namespace stock_percent_change_l190_190398

theorem stock_percent_change (y : ℝ) : 
  let value_after_day1 := 0.85 * y
  let value_after_day2 := 1.25 * value_after_day1
  (value_after_day2 - y) / y * 100 = 6.25 := by
  sorry

end stock_percent_change_l190_190398


namespace hotel_loss_l190_190304

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l190_190304


namespace odd_function_expression_l190_190335

theorem odd_function_expression (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^2 + |x| - 1) : 
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
by
  sorry

end odd_function_expression_l190_190335


namespace nominal_rate_of_interest_correct_l190_190532

noncomputable def nominal_rate_of_interest (EAR : ℝ) (n : ℕ) : ℝ :=
  let i := by 
    sorry
  i

theorem nominal_rate_of_interest_correct :
  nominal_rate_of_interest 0.0609 2 = 0.0598 :=
by 
  sorry

end nominal_rate_of_interest_correct_l190_190532


namespace intersecting_lines_l190_190579

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem intersecting_lines (x y : ℝ) : x ≠ 0 → y ≠ 0 → 
  (diamond x y = diamond y x) ↔ (y = x ∨ y = -x) := 
by
  sorry

end intersecting_lines_l190_190579


namespace job_time_relation_l190_190981

theorem job_time_relation (a b c m n x : ℝ) 
  (h1 : m / a = 1 / b + 1 / c)
  (h2 : n / b = 1 / a + 1 / c)
  (h3 : x / c = 1 / a + 1 / b) :
  x = (m + n + 2) / (m * n - 1) := 
sorry

end job_time_relation_l190_190981


namespace largest_sum_ABC_l190_190236

noncomputable def max_sum_ABC (A B C : ℕ) : ℕ :=
if A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 then
  A + B + C
else
  0

theorem largest_sum_ABC : ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 ∧ max_sum_ABC A B C = 52 :=
sorry

end largest_sum_ABC_l190_190236


namespace eq_op_op_op_92_l190_190719

noncomputable def opN (N : ℝ) : ℝ := 0.75 * N + 2

theorem eq_op_op_op_92 : opN (opN (opN 92)) = 43.4375 :=
by
  sorry

end eq_op_op_op_92_l190_190719


namespace cost_of_hard_lenses_l190_190455

theorem cost_of_hard_lenses (x H : ℕ) (h1 : x + (x + 5) = 11)
    (h2 : 150 * (x + 5) + H * x = 1455) : H = 85 := by
  sorry

end cost_of_hard_lenses_l190_190455


namespace find_length_AB_l190_190800

theorem find_length_AB 
(distance_between_parallels : ℚ)
(radius_of_incircle : ℚ)
(is_isosceles : Prop)
(h_parallel : distance_between_parallels = 18 / 25)
(h_radius : radius_of_incircle = 8 / 3)
(h_isosceles : is_isosceles) :
  ∃ AB : ℚ, AB = 20 := 
sorry

end find_length_AB_l190_190800


namespace words_per_minute_after_break_l190_190713

variable (w : ℕ)

theorem words_per_minute_after_break (h : 10 * 5 - (w * 5) = 10) : w = 8 := by
  sorry

end words_per_minute_after_break_l190_190713


namespace find_value_l190_190742

theorem find_value (a b c : ℝ) (h1 : a + b = 8) (h2 : a * b = c^2 + 16) : a + 2 * b + 3 * c = 12 := by
  sorry

end find_value_l190_190742


namespace smaller_number_between_5_and_8_l190_190716

theorem smaller_number_between_5_and_8 :
  min 5 8 = 5 :=
by
  sorry

end smaller_number_between_5_and_8_l190_190716


namespace quadratic_minimum_val_l190_190703

theorem quadratic_minimum_val (p q x : ℝ) (hp : p > 0) (hq : q > 0) : 
  (∀ x, x^2 - 2 * p * x + 4 * q ≥ p^2 - 4 * q) := 
by
  sorry

end quadratic_minimum_val_l190_190703


namespace seats_still_available_l190_190012

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end seats_still_available_l190_190012


namespace corrected_sum_l190_190627

theorem corrected_sum : 37541 + 43839 ≠ 80280 → 37541 + 43839 = 81380 :=
by
  sorry

end corrected_sum_l190_190627


namespace percentage_of_life_in_accounting_jobs_l190_190428

-- Define the conditions
def years_as_accountant : ℕ := 25
def years_as_manager : ℕ := 15
def lifespan : ℕ := 80

-- Define the proof problem statement
theorem percentage_of_life_in_accounting_jobs :
  (years_as_accountant + years_as_manager) / lifespan * 100 = 50 := 
by sorry

end percentage_of_life_in_accounting_jobs_l190_190428


namespace range_of_m_l190_190343

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2 * m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 1) := by
  sorry

end range_of_m_l190_190343


namespace sufficient_not_necessary_condition_l190_190854

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 4 → x^2 - 4 * x > 0) ∧ ¬ (x^2 - 4 * x > 0 → x > 4) :=
sorry

end sufficient_not_necessary_condition_l190_190854


namespace length_XW_l190_190077

theorem length_XW {XY XZ YZ XW : ℝ}
  (hXY : XY = 15)
  (hXZ : XZ = 17)
  (hAngle : XY^2 + YZ^2 = XZ^2)
  (hYZ : YZ = 8) :
  XW = 15 :=
by
  sorry

end length_XW_l190_190077


namespace Tia_drove_192_more_miles_l190_190789

noncomputable def calculate_additional_miles (s_C t_C : ℝ) : ℝ :=
  let d_C := s_C * t_C
  let d_M := (s_C + 8) * (t_C + 3)
  let d_T := (s_C + 12) * (t_C + 4)
  d_T - d_C

theorem Tia_drove_192_more_miles (s_C t_C : ℝ) (h1 : d_M = d_C + 120) (h2 : d_M = (s_C + 8) * (t_C + 3)) : calculate_additional_miles s_C t_C = 192 :=
by {
  sorry
}

end Tia_drove_192_more_miles_l190_190789


namespace gcd_a_b_l190_190511

def a : ℕ := 333333333
def b : ℕ := 555555555

theorem gcd_a_b : Nat.gcd a b = 111111111 := 
by
  sorry

end gcd_a_b_l190_190511


namespace initial_birds_l190_190269

-- Given conditions
def number_birds_initial (x : ℕ) : Prop :=
  ∃ (y : ℕ), y = 4 ∧ (x + y = 6)

-- Proof statement
theorem initial_birds : ∃ x : ℕ, number_birds_initial x ↔ x = 2 :=
by {
  sorry
}

end initial_birds_l190_190269


namespace complex_multiplication_l190_190342

theorem complex_multiplication :
  ∀ (i : ℂ), i^2 = -1 → (1 - i) * i = 1 + i :=
by
  sorry

end complex_multiplication_l190_190342


namespace number_of_arrangements_with_one_between_A_and_B_l190_190422

theorem number_of_arrangements_with_one_between_A_and_B :
  let people := ["A", "B", "C", "D", "E"] in
  let total_people := people.length in
  list.arrangements people total_people = 36 := sorry

end number_of_arrangements_with_one_between_A_and_B_l190_190422


namespace rationalize_denominator_sum_l190_190932

theorem rationalize_denominator_sum :
  ∃ A B C : ℤ,
  C > 0 ∧
  (∃ p : ℤ, p > 1 ∧ p * p * p ∣ B → false) ∧
  (∃ t : ℝ, t = (5 : ℝ) / (3 * real.cbrt 7) ∧
   t = (A : ℝ) * real.cbrt (B : ℝ) / (C : ℝ)) ∧
  (A + B + C = 75) :=
sorry

end rationalize_denominator_sum_l190_190932


namespace problem_1_simplification_l190_190457

theorem problem_1_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) : 
  (x - 2) / (x ^ 2) / (1 - 2 / x) = 1 / x := 
  sorry

end problem_1_simplification_l190_190457


namespace sum_in_base7_l190_190049

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7_l190_190049


namespace four_genuine_coin_probability_l190_190583

noncomputable def probability_all_genuine_given_equal_weight : ℚ :=
  let total_coins := 20
  let genuine_coins := 12
  let counterfeit_coins := 8

  -- Calculate the probability of selecting two genuine coins from total coins
  let prob_first_pair_genuine := (genuine_coins / total_coins) * 
                                    ((genuine_coins - 1) / (total_coins - 1))

  -- Updating remaining counts after selecting the first pair
  let remaining_genuine_coins := genuine_coins - 2
  let remaining_total_coins := total_coins - 2

  -- Calculate the probability of selecting another two genuine coins
  let prob_second_pair_genuine := (remaining_genuine_coins / remaining_total_coins) * 
                                    ((remaining_genuine_coins - 1) / (remaining_total_coins - 1))

  -- Probability of A ∩ B
  let prob_A_inter_B := prob_first_pair_genuine * prob_second_pair_genuine

  -- Assuming prob_B represents the weighted probabilities including complexities
  let prob_B := (110 / 1077) -- This is an estimated combined probability for the purpose of this definition

  -- Conditional probability P(A | B)
  prob_A_inter_B / prob_B

theorem four_genuine_coin_probability :
  probability_all_genuine_given_equal_weight = 110 / 1077 := sorry

end four_genuine_coin_probability_l190_190583


namespace principal_calc_l190_190051

noncomputable def principal (r : ℝ) : ℝ :=
  (65000 : ℝ) / r

theorem principal_calc (P r : ℝ) (h : 0 < r) :
    (P * 0.10 + P * 1.10 * r / 100 - P * (0.10 + r / 100) = 65) → 
    P = principal r :=
by
  sorry

end principal_calc_l190_190051


namespace rationalize_denominator_result_l190_190930

noncomputable def rationalized_denominator : ℚ := 5 / (3 * real.cbrt 7) * real.cbrt (49 / 49)

theorem rationalize_denominator_result :
  ∃ (A B C : ℤ), C > 0 ∧ ¬∃ (p : ℤ), nat.prime p ∧ p^3 ∣ B ∧ rationalized_denominator = (A * real.cbrt B) / C ∧
  A + B + C = 75 :=
begin
  use [5, 49, 21],
  split,
  { exact zero_lt_one.trans_le (le_refl 21), },
  split,
  { rintros ⟨p, hp⟩,
    exfalso,
    cases hp with hp_prime hp_div,
    have hp_not_cube : ¬ p^3 ∣ 49 := λ h, sorry, -- just a placeholder to skip this proof
    exact hp_not_cube hp_div, },
  split,
  { field_simp [rationalized_denominator],
    norm_num [real.cbrt_pow] },
  { ring }
end

end rationalize_denominator_result_l190_190930


namespace at_least_two_positive_l190_190794

theorem at_least_two_positive (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c > 0) (h5 : a * b + b * c + c * a > 0) :
  (∃ x y : ℝ, (x ≠ y ∧ ((x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c)) ∧ x > 0 ∧ y > 0)) :=
by
  sorry

end at_least_two_positive_l190_190794


namespace work_rate_l190_190435

/-- 
A alone can finish a work in some days which B alone can finish in 15 days. 
If they work together and finish it, then out of a total wages of Rs. 3400, 
A will get Rs. 2040. Prove that A alone can finish the work in 22.5 days. 
-/
theorem work_rate (A : ℚ) (B_rate : ℚ) 
  (total_wages : ℚ) (A_wages : ℚ) 
  (total_rate : ℚ) 
  (hB : B_rate = 1 / 15) 
  (hWages : total_wages = 3400 ∧ A_wages = 2040) 
  (hTotal : total_rate = 1 / A + B_rate)
  (hWorkTogether : 
    (A_wages / (total_wages - A_wages) = 51 / 34) ↔ 
    (A / (A + 15) = 51 / 85)) : 
  A = 22.5 := 
sorry

end work_rate_l190_190435


namespace smallest_odd_number_with_four_prime_factors_l190_190823

theorem smallest_odd_number_with_four_prime_factors (n : ℕ) (hodd : n % 2 = 1) (hp : ∃ p1 p2 p3 p4, nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ p1 * p2 * p3 * p4 = n ∧ (p1 > 11 ∨ p2 > 11 ∨ p3 > 11 ∨ p4 > 11)) : n = 1365 :=
by
  sorry

end smallest_odd_number_with_four_prime_factors_l190_190823


namespace symmetric_line_equation_l190_190408

theorem symmetric_line_equation 
  (L : ℝ → ℝ → Prop)
  (H : ∀ x y, L x y ↔ x - 2 * y + 1 = 0) : 
  ∃ L' : ℝ → ℝ → Prop, 
    (∀ x y, L' x y ↔ x + 2 * y - 3 = 0) ∧ 
    ( ∀ x y, L (2 - x) y ↔ L' x y ) := 
sorry

end symmetric_line_equation_l190_190408


namespace binom_20_17_l190_190459

theorem binom_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end binom_20_17_l190_190459


namespace g_inv_undefined_at_1_l190_190486

noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x - 5)

noncomputable def g_inv (x : ℝ) : ℝ := (5 * x - 3) / (x - 1)

theorem g_inv_undefined_at_1 : ∀ x : ℝ, (g_inv x) = g_inv 1 → x = 1 :=
by
  intro x h
  sorry

end g_inv_undefined_at_1_l190_190486


namespace tan_periodic_n_solution_l190_190470

open Real

theorem tan_periodic_n_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ tan (n * (π / 180)) = tan (1540 * (π / 180)) ∧ n = 40 :=
by
  sorry

end tan_periodic_n_solution_l190_190470


namespace only_natural_number_dividing_power_diff_l190_190589

theorem only_natural_number_dividing_power_diff (n : ℕ) (h : n ∣ (2^n - 1)) : n = 1 :=
by
  sorry

end only_natural_number_dividing_power_diff_l190_190589


namespace quadratic_solutions_l190_190267

theorem quadratic_solutions (x : ℝ) : (2 * x^2 + 5 * x + 3 = 0) → (x = -1 ∨ x = -3 / 2) :=
by {
  sorry
}

end quadratic_solutions_l190_190267


namespace consecutive_odd_numbers_l190_190978

theorem consecutive_odd_numbers (a b c d e : ℤ) (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h4 : e = a + 8) (h5 : a + c = 146) : e = 79 := 
by
  sorry

end consecutive_odd_numbers_l190_190978


namespace count_integers_satisfying_inequalities_l190_190739

theorem count_integers_satisfying_inequalities :
  {n : Int | 3 ≤ n ∧ n ≤ 7 ∧ (Real.sqrt (3 * n - 1) ≤ Real.sqrt (5 * n - 7)) ∧ (Real.sqrt (5 * n - 7) < Real.sqrt (3 * n + 8))}.card = 5 := 
by 
  sorry

end count_integers_satisfying_inequalities_l190_190739


namespace area_of_triangle_l190_190409

theorem area_of_triangle (h : ℝ) (a : ℝ) (b : ℝ) (hypotenuse : h = 13) (side_a : a = 5) (right_triangle : a^2 + b^2 = h^2) : 
  ∃ (area : ℝ), area = 30 := 
by
  sorry

end area_of_triangle_l190_190409


namespace range_of_a_l190_190888

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x = 1 → x > a) : a < 1 := 
by
  sorry

end range_of_a_l190_190888


namespace proof_area_of_squares_l190_190123

noncomputable def area_of_squares : Prop :=
  let side_C := 48
  let side_D := 60
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  (area_C / area_D = (16 / 25)) ∧ 
  ((area_D - area_C) / area_C = (36 / 100))

theorem proof_area_of_squares : area_of_squares := sorry

end proof_area_of_squares_l190_190123


namespace compute_star_difference_l190_190228

def star (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_star_difference : (star 6 3) - (star 3 6) = 45 := by
  sorry

end compute_star_difference_l190_190228


namespace limit_one_minus_reciprocal_l190_190170

theorem limit_one_minus_reciprocal (h : Filter.Tendsto (fun (n : ℕ) => 1 / n) Filter.atTop (nhds 0)) :
  Filter.Tendsto (fun (n : ℕ) => 1 - 1 / n) Filter.atTop (nhds 1) :=
sorry

end limit_one_minus_reciprocal_l190_190170


namespace percentage_exceeds_self_l190_190442

theorem percentage_exceeds_self (N : ℕ) (P : ℝ) (h1 : N = 150) (h2 : N = (P / 100) * N + 126) : P = 16 := by
  sorry

end percentage_exceeds_self_l190_190442


namespace next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l190_190871

-- Part (a)
theorem next_terms_arithmetic_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ), 
  a₀ = 3 → a₁ = 7 → a₂ = 11 → a₃ = 15 → a₄ = 19 → a₅ = 23 → d = 4 →
  (a₅ + d = 27) ∧ (a₅ + 2*d = 31) :=
by intros; sorry


-- Part (b)
theorem next_terms_alternating_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℕ),
  a₀ = 9 → a₁ = 1 → a₂ = 7 → a₃ = 1 → a₄ = 5 → a₅ = 1 →
  a₄ - 2 = 3 ∧ a₁ = 1 :=
by intros; sorry


-- Part (c)
theorem next_terms_interwoven_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ),
  a₀ = 4 → a₁ = 5 → a₂ = 8 → a₃ = 9 → a₄ = 12 → a₅ = 13 → d = 4 →
  (a₄ + d = 16) ∧ (a₅ + d = 17) :=
by intros; sorry


-- Part (d)
theorem next_terms_geometric_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅: ℕ), 
  a₀ = 1 → a₁ = 2 → a₂ = 4 → a₃ = 8 → a₄ = 16 → a₅ = 32 →
  (a₅ * 2 = 64) ∧ (a₅ * 4 = 128) :=
by intros; sorry

end next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l190_190871


namespace unique_valid_quintuple_l190_190474

theorem unique_valid_quintuple :
  ∃! (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
    a^2 + b^2 + c^3 + d^3 + e^3 = 5 ∧
    (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25 :=
sorry

end unique_valid_quintuple_l190_190474


namespace lions_at_sanctuary_l190_190319

variable (L C : ℕ)

noncomputable def is_solution :=
  C = 1 / 2 * (L + 14) ∧
  L + 14 + C = 39 ∧
  L = 12

theorem lions_at_sanctuary : is_solution L C :=
sorry

end lions_at_sanctuary_l190_190319


namespace find_period_l190_190027

-- Definitions based on conditions
def interest_rate_A : ℝ := 0.10
def interest_rate_C : ℝ := 0.115
def principal : ℝ := 4000
def total_gain : ℝ := 180

-- The question to prove
theorem find_period (n : ℝ) : 
  n = 3 :=
by 
  have interest_to_A := interest_rate_A * principal
  have interest_from_C := interest_rate_C * principal
  have annual_gain := interest_from_C - interest_to_A
  have equation := total_gain = annual_gain * n
  sorry

end find_period_l190_190027


namespace product_of_solutions_eq_neg_35_l190_190541

theorem product_of_solutions_eq_neg_35 :
  ∀ (x : ℝ), -35 = -x^2 - 2 * x → ∃ (p : ℝ), p = -35 :=
by
  intro x h
  sorry

end product_of_solutions_eq_neg_35_l190_190541


namespace gray_region_correct_b_l190_190308

-- Define the basic conditions
def square_side_length : ℝ := 3
def small_square_side_length : ℝ := 1

-- Define the triangles resulting from cutting a square
def triangle_area : ℝ := 0.5 * square_side_length * square_side_length

-- Define the gray region area for the second figure (b)
def gray_region_area_b : ℝ := 0.25

-- Lean statement to prove the area of the gray region
theorem gray_region_correct_b : gray_region_area_b = 0.25 := by
  -- Proof is omitted
  sorry

end gray_region_correct_b_l190_190308


namespace intersection_points_l190_190329

noncomputable def line1 (x y : ℝ) : Prop := 3 * x - 2 * y = 12
noncomputable def line2 (x y : ℝ) : Prop := 2 * x + 4 * y = 8
noncomputable def line3 (x y : ℝ) : Prop := -5 * x + 15 * y = 30
noncomputable def line4 (x : ℝ) : Prop := x = -3

theorem intersection_points : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ 
  (∃ (x y : ℝ), line1 x y ∧ x = -3 ∧ y = -10.5) ∧ 
  ¬(∃ (x y : ℝ), line2 x y ∧ line3 x y) ∧
  ∃ (x y : ℝ), line4 x ∧ y = -10.5 :=
  sorry

end intersection_points_l190_190329


namespace f_odd_and_inequality_l190_190645

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem f_odd_and_inequality (x c : ℝ) : ∀ x c, 
  f x < c^2 - 3 * c + 3 := by 
  sorry

end f_odd_and_inequality_l190_190645


namespace find_n_l190_190959

def Point : Type := ℝ × ℝ

def A : Point := (5, -8)
def B : Point := (9, -30)
def C (n : ℝ) : Point := (n, n)

def collinear (p1 p2 p3 : Point) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_n (n : ℝ) (h : collinear A B (C n)) : n = 3 := 
by
  sorry

end find_n_l190_190959


namespace Amanda_ticket_sales_goal_l190_190710

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l190_190710


namespace triangle_height_l190_190826

theorem triangle_height (area base : ℝ) (h_area : area = 9.31) (h_base : base = 4.9) : (2 * area) / base = 3.8 :=
by
  sorry

end triangle_height_l190_190826


namespace impossible_d_values_count_l190_190276

def triangle_rectangle_difference (d : ℕ) : Prop :=
  ∃ (l w : ℕ),
  l = 2 * w ∧
  6 * w > 0 ∧
  (6 * w + 2 * d) - 6 * w = 1236 ∧
  d > 0

theorem impossible_d_values_count : ∀ d : ℕ, d ≠ 618 → ¬triangle_rectangle_difference d :=
by
  sorry

end impossible_d_values_count_l190_190276


namespace alice_and_bob_pies_l190_190163

theorem alice_and_bob_pies (T : ℝ) : (T / 5 = T / 6 + 2) → T = 60 := by
  sorry

end alice_and_bob_pies_l190_190163


namespace xy_zero_iff_x_zero_necessary_not_sufficient_l190_190193

theorem xy_zero_iff_x_zero_necessary_not_sufficient {x y : ℝ} : 
  (x * y = 0) → ((x = 0) ∨ (y = 0)) ∧ ¬((x = 0) → (x * y ≠ 0)) := 
sorry

end xy_zero_iff_x_zero_necessary_not_sufficient_l190_190193


namespace opposite_of_neg2023_l190_190671

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l190_190671


namespace doris_weeks_to_meet_expenses_l190_190726

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l190_190726


namespace correct_exponentiation_l190_190432

theorem correct_exponentiation (x : ℝ) : x^2 * x^3 = x^5 :=
by sorry

end correct_exponentiation_l190_190432


namespace labor_day_to_national_day_l190_190614

theorem labor_day_to_national_day :
  let labor_day := 1 -- Monday is represented as 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  (labor_day + total_days % 7) % 7 = 0 := -- Since 0 corresponds to Sunday modulo 7
by
  let labor_day := 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  have h1 : (labor_day + total_days % 7) % 7 = ((1 + (31 * 3 + 30 * 2) % 7) % 7) := by rfl
  sorry

end labor_day_to_national_day_l190_190614


namespace probability_computation_l190_190445

noncomputable def probability_inside_sphere : ℝ :=
  let volume_of_cube : ℝ := 64
  let volume_of_sphere : ℝ := (4/3) * Real.pi * (2^3)
  volume_of_sphere / volume_of_cube

theorem probability_computation :
  probability_inside_sphere = Real.pi / 6 :=
by
  sorry

end probability_computation_l190_190445


namespace tangent_line_ln_x_xsq_l190_190273

theorem tangent_line_ln_x_xsq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) :
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_ln_x_xsq_l190_190273


namespace correct_operation_l190_190975

theorem correct_operation : 
  (a^2 + a^2 = 2 * a^2) = false ∧ 
  ((-3 * a * b^2)^2 = -6 * a^2 * b^4) = false ∧ 
  (a^6 / (-a)^2 = a^4) = true ∧ 
  ((a - b)^2 = a^2 - b^2) = false :=
sorry

end correct_operation_l190_190975


namespace rationalize_denominator_l190_190934

theorem rationalize_denominator (A B C : ℤ) (hA : A = 5) (hB : B = 49) (hC : C = 21)
  (h_pos : 0 < C) (h_not_divisible : ¬ ∃ p : ℤ, prime p ∧ p ^ 3 ∣ B) :
  A + B + C = 75 :=
by
  sorry

end rationalize_denominator_l190_190934


namespace students_surveyed_l190_190296

theorem students_surveyed (S : ℕ)
  (h1 : (2/3 : ℝ) * 6 + (1/3 : ℝ) * 4 = 16/3)
  (h2 : S * (16/3 : ℝ) = 320) :
  S = 60 :=
sorry

end students_surveyed_l190_190296


namespace units_digit_A_is_1_l190_190217

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

theorem units_digit_A_is_1 : units_digit A = 1 := by
  sorry

end units_digit_A_is_1_l190_190217


namespace find_CD_l190_190464

noncomputable def C : ℝ := 32 / 9
noncomputable def D : ℝ := 4 / 9

theorem find_CD :
  (∀ x, x ≠ 6 ∧ x ≠ -3 → (4 * x + 8) / (x^2 - 3 * x - 18) = 
       C / (x - 6) + D / (x + 3)) →
  C = 32 / 9 ∧ D = 4 / 9 :=
by sorry

end find_CD_l190_190464


namespace opposite_pairs_l190_190569

theorem opposite_pairs :
  ∃ (x y : ℤ), (x = -5 ∧ y = -(-5)) ∧ (x = -y) ∧ (
    (¬ (∃ (a b : ℤ), (a = -2 ∧ b = 1/2) ∧ (a = -b))) ∧ 
    (¬ (∃ (c d : ℤ), (c = | -1 | ∧ d = 1) ∧ (c = -d))) ∧
    (¬ (∃ (e f : ℤ), (e = (-3)^2 ∧ f = 3^2) ∧ (e = -f)))
  ) :=
by
  sorry

end opposite_pairs_l190_190569


namespace eliot_account_balance_l190_190948

variable (A E : ℝ)

theorem eliot_account_balance (h1 : A - E = (1/12) * (A + E)) (h2 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 := 
by 
  sorry

end eliot_account_balance_l190_190948


namespace A_E_not_third_l190_190235

-- Define the runners and their respective positions.
inductive Runner
| A : Runner
| B : Runner
| C : Runner
| D : Runner
| E : Runner
open Runner

variable (position : Runner → Nat)

-- Conditions
axiom A_beats_B : position A < position B
axiom C_beats_D : position C < position D
axiom B_beats_E : position B < position E
axiom D_after_A_before_B : position A < position D ∧ position D < position B

-- Prove that A and E cannot be in third place.
theorem A_E_not_third : position A ≠ 3 ∧ position E ≠ 3 :=
sorry

end A_E_not_third_l190_190235


namespace geom_prog_common_ratio_l190_190588

-- Definition of a geometric progression
def geom_prog (u : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)

-- Statement of the problem
theorem geom_prog_common_ratio (u : ℕ → ℝ) (q : ℝ) (hq : ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)) :
  (q = (1 + Real.sqrt 5) / 2) ∨ (q = (1 - Real.sqrt 5) / 2) :=
sorry

end geom_prog_common_ratio_l190_190588


namespace exists_integer_root_l190_190253

theorem exists_integer_root (a b c d : ℤ) (ha : a ≠ 0)
  (h : ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * (a * x^3 + b * x^2 + c * x + d) = y * (a * y^3 + b * y^2 + c * y + d)) :
  ∃ z : ℤ, a * z^3 + b * z^2 + c * z + d = 0 :=
by
  sorry

end exists_integer_root_l190_190253


namespace unique_solution_j_l190_190058

theorem unique_solution_j (j : ℝ) : (∀ x : ℝ, (2 * x + 7) * (x - 5) = -43 + j * x) → (j = 5 ∨ j = -11) :=
by
  sorry

end unique_solution_j_l190_190058


namespace inequality_solution_l190_190781

noncomputable def f (a b x : ℝ) : ℝ := 1 / Real.sqrt x + 1 / Real.sqrt (a + b - x)

theorem inequality_solution 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (x : ℝ) 
  (hx : x ∈ Set.Ioo (min a b) (max a b)) : 
  f a b x < f a b a ∧ f a b x < f a b b := 
sorry

end inequality_solution_l190_190781


namespace sinA_value_triangle_area_l190_190624

-- Definitions of the given variables
variables (A B C : ℝ)
variables (a b c : ℝ)
variables (sinA sinC cosC : ℝ)

-- Given conditions
axiom h_c : c = Real.sqrt 2
axiom h_a : a = 1
axiom h_cosC : cosC = 3 / 4
axiom h_sinC : sinC = Real.sqrt 7 / 4
axiom h_b : b = 2

-- Question 1: Prove sin A = sqrt 14 / 8
theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
sorry

-- Question 2: Prove the area of triangle ABC is sqrt 7 / 4
theorem triangle_area : 1/2 * a * b * sinC = Real.sqrt 7 / 4 :=
sorry

end sinA_value_triangle_area_l190_190624


namespace polynomial_value_at_n_plus_3_l190_190642

-- Define the polynomial P
noncomputable def P (x : ℕ) : ℕ → ℝ := λ n, if x = n then (2^x) else 0

-- Define the theorem to be proved
theorem polynomial_value_at_n_plus_3 (n : ℕ) : 
  (P (n + 3) n) = 2 * (2^(n + 2) - n - 3) :=
by sorry

end polynomial_value_at_n_plus_3_l190_190642


namespace simplify_and_evaluate_expression_l190_190266

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l190_190266


namespace equation_of_parallel_line_l190_190407

theorem equation_of_parallel_line (c : ℕ) :
  (∃ c, x + 2 * y + c = 0) ∧ (1 + 2 * 1 + c = 0) -> x + 2 * y - 3 = 0 :=
by 
  sorry

end equation_of_parallel_line_l190_190407


namespace opposite_of_neg_2023_l190_190669

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l190_190669


namespace sqrt_0_09_eq_0_3_l190_190812

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end sqrt_0_09_eq_0_3_l190_190812


namespace xy_sum_square_l190_190587

theorem xy_sum_square (x y : ℕ) (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end xy_sum_square_l190_190587


namespace polynomial_simplification_l190_190102

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 9 * x - 8) + (-x^5 + x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 14) = 
  -x^5 + 3 * x^4 + x^3 - x^2 + 3 * x + 6 :=
by
  sorry

end polynomial_simplification_l190_190102


namespace cyclic_sum_nonneg_l190_190061

theorem cyclic_sum_nonneg 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (k : ℝ) (hk1 : 0 ≤ k) (hk2 : k < 2) :
  (a^2 - b * c) / (b^2 + c^2 + k * a^2)
  + (b^2 - c * a) / (c^2 + a^2 + k * b^2)
  + (c^2 - a * b) / (a^2 + b^2 + k * c^2) ≥ 0 :=
sorry

end cyclic_sum_nonneg_l190_190061


namespace range_of_a_l190_190606

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a + 3

theorem range_of_a :
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ g x2 a = 0 ∧ |x1 - x2| ≤ 1) ↔ (a ∈ Set.Icc 2 3) := sorry

end range_of_a_l190_190606


namespace triangle_inequality_l190_190510

noncomputable def p (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def r (a b c : ℝ) : ℝ := 
  let p := p a b c
  let x := p - a
  let y := p - b
  let z := p - c
  Real.sqrt ((x * y * z) / (x + y + z))

noncomputable def x (a b c : ℝ) : ℝ := p a b c - a
noncomputable def y (a b c : ℝ) : ℝ := p a b c - b
noncomputable def z (a b c : ℝ) : ℝ := p a b c - c

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 / (x a b c)^2 + 1 / (y a b c)^2 + 1 / (z a b c)^2 ≥ (x a b c + y a b c + z a b c) / ((x a b c) * (y a b c) * (z a b c)) := by
    sorry

end triangle_inequality_l190_190510


namespace joe_selects_SIMPLER_l190_190246

noncomputable def probability_SIMPLER : ℚ := (3 / 10) * (1 / 3) * (2 / 3)

theorem joe_selects_SIMPLER :
  let THINK := {'T', 'H', 'I', 'N', 'K'} in
  let STREAM := {'S', 'T', 'R', 'E', 'A', 'M'} in
  let PLACES := {'P', 'L', 'A', 'C', 'E', 'S'} in
  let needed_letters := {'S', 'I', 'M', 'P', 'L', 'E', 'R'} in
  (think_subprob : THINK.card = 5) ∧ (stream_subprob : STREAM.card = 6) ∧ (places_subprob : PLACES.card = 6) →
  (p_needed_think : (THINK.choose 3).filter (λ subset, 'I' ∈ subset ∧ 'N' ∈ subset).card / (THINK.choose 3).card = (3 / 10)) →
  (p_needed_stream : (STREAM.choose 5).filter (λ subset, 'S' ∈ subset ∧ 'M' ∈ subset ∧ 'E' ∈ subset ∧ 'R' ∈ subset).card / (STREAM.choose 5).card = (1 / 3)) →
  (p_needed_places : (PLACES.choose 4).filter (λ subset, 'L' ∈ subset).card / (PLACES.choose 4).card = (2 / 3)) →
  probability_SIMPLER = 1 / 15 :=
by
  sorry

end joe_selects_SIMPLER_l190_190246


namespace circle_area_approx_error_exceeds_one_l190_190557

theorem circle_area_approx_error_exceeds_one (r : ℝ) : 
  (3.14159 < Real.pi ∧ Real.pi < 3.14160) → 
  2 * r > 25 →  
  |(r * r * Real.pi - r * r * 3.14)| > 1 → 
  2 * r = 51 := 
by 
  sorry

end circle_area_approx_error_exceeds_one_l190_190557


namespace parent_combinations_for_O_l190_190161

-- Define the blood types
inductive BloodType
| A
| B
| O
| AB

open BloodType

-- Define the conditions given in the problem
def parent_not_AB (p : BloodType) : Prop :=
  p ≠ AB

def possible_parent_types : List BloodType :=
  [A, B, O]

-- The math proof problem
theorem parent_combinations_for_O :
  ∀ (mother father : BloodType),
    parent_not_AB mother →
    parent_not_AB father →
    mother ∈ possible_parent_types →
    father ∈ possible_parent_types →
    (possible_parent_types.length * possible_parent_types.length) = 9 := 
by
  intro mother father h1 h2 h3 h4
  sorry

end parent_combinations_for_O_l190_190161


namespace Jaymee_is_22_l190_190913

-- Define Shara's age
def Shara_age : ℕ := 10

-- Define Jaymee's age according to the problem conditions
def Jaymee_age : ℕ := 2 + 2 * Shara_age

-- The proof statement to show that Jaymee's age is 22
theorem Jaymee_is_22 : Jaymee_age = 22 := by 
  -- The proof is omitted according to the instructions.
  sorry

end Jaymee_is_22_l190_190913


namespace value_of_q_l190_190489

theorem value_of_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 :=
by
  sorry

end value_of_q_l190_190489


namespace find_line_equation_l190_190598

theorem find_line_equation :
  ∃ (a b c : ℝ), (a * -5 + b * -1 = c) ∧ (a * 1 + b * 1 = c + 2) ∧ (b ≠ 0) ∧ (a * 2 + b = 0) → (∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = -5) :=
by
  sorry

end find_line_equation_l190_190598


namespace max_cone_cross_section_area_l190_190063

theorem max_cone_cross_section_area
  (V A B : Type)
  (E : Type)
  (l : ℝ)
  (α : ℝ) :
  0 < l ∧ 0 < α ∧ α < 180 → 
  ∃ (area : ℝ), area = (1 / 2) * l^2 :=
by
  sorry

end max_cone_cross_section_area_l190_190063


namespace Kat_training_hours_l190_190778

theorem Kat_training_hours
  (h_strength_times : ℕ)
  (h_strength_hours : ℝ)
  (h_boxing_times : ℕ)
  (h_boxing_hours : ℝ)
  (h_times : h_strength_times = 3)
  (h_strength : h_strength_hours = 1)
  (b_times : h_boxing_times = 4)
  (b_hours : h_boxing_hours = 1.5) :
  h_strength_times * h_strength_hours + h_boxing_times * h_boxing_hours = 9 :=
by
  sorry

end Kat_training_hours_l190_190778


namespace petya_pencils_l190_190655

theorem petya_pencils (x : ℕ) (promotion : x + 12 = 61) :
  x = 49 :=
by
  sorry

end petya_pencils_l190_190655


namespace inequality_problem_l190_190615

variable (a b c d : ℝ)

theorem inequality_problem (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := sorry

end inequality_problem_l190_190615


namespace calculate_expression_l190_190576

theorem calculate_expression : 2 * (-2) + (-3) = -7 := 
  sorry

end calculate_expression_l190_190576


namespace solve_for_3x2_plus_6_l190_190765

theorem solve_for_3x2_plus_6 (x : ℚ) (h : 5 * x + 3 = 2 * x - 4) : 3 * (x^2 + 6) = 103 / 3 :=
by
  sorry

end solve_for_3x2_plus_6_l190_190765


namespace borya_number_l190_190707

theorem borya_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) 
  (h3 : (n * 2 + 5) * 5 = 715) : n = 69 :=
sorry

end borya_number_l190_190707


namespace stickers_started_with_l190_190137

-- Definitions for the conditions
def stickers_given (Emily : ℕ) : Prop := Emily = 7
def stickers_ended_with (Willie_end : ℕ) : Prop := Willie_end = 43

-- The main proof statement
theorem stickers_started_with (Willie_start : ℕ) :
  stickers_given 7 →
  stickers_ended_with 43 →
  Willie_start = 43 - 7 :=
by
  intros h₁ h₂
  sorry

end stickers_started_with_l190_190137


namespace points_collinear_distance_relation_l190_190001

theorem points_collinear_distance_relation (x y : ℝ) 
  (h1 : (5 - y) * (5 - 1) = -4 * (-2 - x))
  (h2 : real.sqrt ((y - 1)^2 + 9) = 2 * real.sqrt ((x - 1)^2 + 16)) :
  (x + y = -9 / 2) ∨ (x + y = 17 / 2) := 
sorry

end points_collinear_distance_relation_l190_190001


namespace simplify_and_evaluate_expression_l190_190522

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = 1) (h₂ : b = -2) :
  (2 * a + b)^2 - 3 * a * (2 * a - b) = -12 :=
by
  rw [h₁, h₂]
  -- Now the expression to prove transforms to:
  -- (2 * 1 + (-2))^2 - 3 * 1 * (2 * 1 - (-2)) = -12
  -- Subsequent proof steps would follow simplification directly.
  sorry

end simplify_and_evaluate_expression_l190_190522


namespace no_values_of_expression_l190_190412

theorem no_values_of_expression (x : ℝ) (h : x^2 - 4 * x + 4 < 0) :
  ¬ ∃ y, y = x^2 + 4 * x + 5 :=
by
  sorry

end no_values_of_expression_l190_190412


namespace division_equivalent_l190_190686

def division_to_fraction (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 ∧ 0 ≤ a ∧ 0 ≤ b → a / b = (a * 1000) / (b * 1000) :=
by
  intros h
  field_simp
  
theorem division_equivalent (h : 0 ≤ 0.08 ∧ 0 ≤ 0.002 ∧ 0.08 ≠ 0 ∧ 0.002 ≠ 0) :
  0.08 / 0.002 = 40 :=
by
  have := division_to_fraction 0.08 0.002 h
  norm_num at this
  exact this

end division_equivalent_l190_190686


namespace number_times_half_squared_is_eight_l190_190290

noncomputable def num : ℝ := 32

theorem number_times_half_squared_is_eight :
  (num * (1 / 2) ^ 2 = 2 ^ 3) :=
by
  sorry

end number_times_half_squared_is_eight_l190_190290


namespace cycle_reappear_l190_190775

/-- Given two sequences with cycle lengths 6 and 4, prove the sequences will align on line number 12 -/
theorem cycle_reappear (l1 l2 : ℕ) (h1 : l1 = 6) (h2 : l2 = 4) :
  Nat.lcm l1 l2 = 12 := by
  sorry

end cycle_reappear_l190_190775


namespace find_prime_numbers_of_form_p_p_plus_1_l190_190044

def has_at_most_19_digits (n : ℕ) : Prop := n < 10^19

theorem find_prime_numbers_of_form_p_p_plus_1 :
  {n : ℕ | ∃ p : ℕ, n = p^p + 1 ∧ has_at_most_19_digits n ∧ Nat.Prime n} = {2, 5, 257} :=
by
  sorry

end find_prime_numbers_of_form_p_p_plus_1_l190_190044


namespace probability_of_winning_set_l190_190817

def winning_probability : ℚ :=
  let total_cards := 9
  let total_draws := 3
  let same_color_sets := 3
  let same_letter_sets := 3
  let total_ways_to_draw := Nat.choose total_cards total_draws
  let total_favorable_outcomes := same_color_sets + same_letter_sets
  let probability := total_favorable_outcomes / total_ways_to_draw
  probability

theorem probability_of_winning_set :
  winning_probability = 1 / 14 :=
by
  sorry

end probability_of_winning_set_l190_190817


namespace train_speed_l190_190159

/--
Given:
- The speed of the first person \(V_p\) is 4 km/h.
- The train takes 9 seconds to pass the first person completely.
- The length of the train is approximately 50 meters (49.999999999999986 meters).

Prove:
- The speed of the train \(V_t\) is 24 km/h.
-/
theorem train_speed (V_p : ℝ) (t : ℝ) (L : ℝ) (V_t : ℝ) 
  (hV_p : V_p = 4) 
  (ht : t = 9)
  (hL : L = 49.999999999999986)
  (hrel_speed : (L / t) * 3.6 = V_t - V_p) :
  V_t = 24 :=
by
  sorry

end train_speed_l190_190159


namespace coeff_of_term_equal_three_l190_190617

theorem coeff_of_term_equal_three (x : ℕ) (h : x = 13) : 
    2^x - 2^(x - 2) = 3 * 2^(11) :=
by
    rw [h]
    sorry

end coeff_of_term_equal_three_l190_190617


namespace joan_spent_on_toys_l190_190376

theorem joan_spent_on_toys :
  let toy_cars := 14.88
  let toy_trucks := 5.86
  toy_cars + toy_trucks = 20.74 :=
by
  let toy_cars := 14.88
  let toy_trucks := 5.86
  sorry

end joan_spent_on_toys_l190_190376


namespace p_and_q_together_complete_in_10_days_l190_190693

noncomputable def p_time := 50 / 3
noncomputable def q_time := 25
noncomputable def r_time := 50

theorem p_and_q_together_complete_in_10_days 
  (h1 : 1 / p_time = 1 / q_time + 1 / r_time)
  (h2 : r_time = 50)
  (h3 : q_time = 25) :
  (p_time * q_time) / (p_time + q_time) = 10 :=
by
  sorry

end p_and_q_together_complete_in_10_days_l190_190693


namespace value_of_x7_plus_64x2_l190_190784

-- Let x be a real number such that x^3 + 4x = 8.
def x_condition (x : ℝ) : Prop := x^3 + 4 * x = 8

-- We need to determine the value of x^7 + 64x^2.
theorem value_of_x7_plus_64x2 (x : ℝ) (h : x_condition x) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end value_of_x7_plus_64x2_l190_190784


namespace extreme_point_of_f_l190_190274

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - log x

theorem extreme_point_of_f : ∃ x₀ > 0, f x₀ = f (sqrt 3 / 3) ∧ 
  (∀ x < sqrt 3 / 3, f x > f (sqrt 3 / 3)) ∧
  (∀ x > sqrt 3 / 3, f x > f (sqrt 3 / 3)) :=
sorry

end extreme_point_of_f_l190_190274


namespace max_AMC_AM_MC_CA_l190_190085

theorem max_AMC_AM_MC_CA (A M C : ℕ) (h_sum : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_AMC_AM_MC_CA_l190_190085


namespace rachel_age_when_emily_half_age_l190_190730

theorem rachel_age_when_emily_half_age 
  (E_0 : ℕ) (R_0 : ℕ) (h1 : E_0 = 20) (h2 : R_0 = 24) 
  (age_diff : R_0 - E_0 = 4) : 
  ∃ R : ℕ, ∃ E : ℕ, E = R / 2 ∧ R = E + 4 ∧ R = 8 :=
by
  sorry

end rachel_age_when_emily_half_age_l190_190730


namespace goose_eggs_count_l190_190259

theorem goose_eggs_count (E : ℕ) 
  (hatch_ratio : ℝ := 1/4)
  (survival_first_month_ratio : ℝ := 4/5)
  (survival_first_year_ratio : ℝ := 3/5)
  (survived_first_year : ℕ := 120) :
  ((survival_first_year_ratio * (survival_first_month_ratio * hatch_ratio * E)) = survived_first_year) → E = 1000 :=
by
  intro h
  sorry

end goose_eggs_count_l190_190259


namespace ball_bounce_height_l190_190696

theorem ball_bounce_height (initial_height : ℝ) (r : ℝ) (k : ℕ) : 
  initial_height = 1000 → r = 1/2 → (r ^ k * initial_height < 1) → k = 10 := by
sorry

end ball_bounce_height_l190_190696


namespace base_7_divisibility_l190_190175

theorem base_7_divisibility (y : ℕ) :
  (934 + 7 * y) % 19 = 0 ↔ y = 3 :=
by
  sorry

end base_7_divisibility_l190_190175


namespace whiteboards_per_class_is_10_l190_190628

-- Definitions from conditions
def classes : ℕ := 5
def ink_per_whiteboard_ml : ℕ := 20
def cost_per_ml_cents : ℕ := 50
def total_cost_cents : ℕ := 100 * 100  -- converting $100 to cents

-- Following the solution, define other useful constants
def cost_per_whiteboard_cents : ℕ := ink_per_whiteboard_ml * cost_per_ml_cents
def total_cost_all_classes_cents : ℕ := classes * total_cost_cents
def total_whiteboards : ℕ := total_cost_all_classes_cents / cost_per_whiteboard_cents
def whiteboards_per_class : ℕ := total_whiteboards / classes

-- We want to prove that each class uses 10 whiteboards.
theorem whiteboards_per_class_is_10 : whiteboards_per_class = 10 :=
  sorry

end whiteboards_per_class_is_10_l190_190628


namespace range_of_a_l190_190982

def A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a < x ∧ x < a + 1}

theorem range_of_a (a : ℝ)
  (h₀ : a < 1)
  (h₁ : B a ⊆ A) :
  a ∈ {x : ℝ | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
by
  sorry

end range_of_a_l190_190982


namespace fraction_of_primes_is_prime_l190_190391

theorem fraction_of_primes_is_prime
  (p q r : ℕ) 
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (hr : Nat.Prime r)
  (h : ∃ k : ℕ, p * q * r = k * (p + q + r)) :
  Nat.Prime (p * q * r / (p + q + r)) := 
sorry

end fraction_of_primes_is_prime_l190_190391


namespace magnitude_of_z_l190_190496

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end magnitude_of_z_l190_190496


namespace ratio_of_pete_to_susan_l190_190790

noncomputable def Pete_backward_speed := 12 -- in miles per hour
noncomputable def Pete_handstand_speed := 2 -- in miles per hour
noncomputable def Tracy_cartwheel_speed := 4 * Pete_handstand_speed -- in miles per hour
noncomputable def Susan_forward_speed := Tracy_cartwheel_speed / 2 -- in miles per hour

theorem ratio_of_pete_to_susan :
  Pete_backward_speed / Susan_forward_speed = 3 := 
sorry

end ratio_of_pete_to_susan_l190_190790


namespace sum_of_dice_not_in_set_l190_190967

theorem sum_of_dice_not_in_set (a b c : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) (h₃ : 1 ≤ c ∧ c ≤ 6) 
  (h₄ : a * b * c = 72) (h₅ : a = 4 ∨ b = 4 ∨ c = 4) :
  a + b + c ≠ 12 ∧ a + b + c ≠ 14 ∧ a + b + c ≠ 15 ∧ a + b + c ≠ 16 :=
by
  sorry

end sum_of_dice_not_in_set_l190_190967


namespace geometric_seq_ratio_l190_190388

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l190_190388


namespace compute_product_fraction_l190_190322

theorem compute_product_fraction :
  ( ((3 : ℚ)^4 - 1) / ((3 : ℚ)^4 + 1) *
    ((4 : ℚ)^4 - 1) / ((4 : ℚ)^4 + 1) * 
    ((5 : ℚ)^4 - 1) / ((5 : ℚ)^4 + 1) *
    ((6 : ℚ)^4 - 1) / ((6 : ℚ)^4 + 1) *
    ((7 : ℚ)^4 - 1) / ((7 : ℚ)^4 + 1)
  ) = (25 / 210) := 
  sorry

end compute_product_fraction_l190_190322


namespace area_enclosed_by_graph_eq_2pi_l190_190132

theorem area_enclosed_by_graph_eq_2pi :
  (∃ (x y : ℝ), x^2 + y^2 = 2 * |x| + 2 * |y| ) →
  ∀ (A : ℝ), A = 2 * Real.pi :=
sorry

end area_enclosed_by_graph_eq_2pi_l190_190132


namespace find_first_number_l190_190963

theorem find_first_number (x y : ℝ) (h1 : x + y = 50) (h2 : 2 * (x - y) = 20) : x = 30 :=
by
  sorry

end find_first_number_l190_190963


namespace determine_k_for_intersection_l190_190328

theorem determine_k_for_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 3 = 2 * x + 5) ∧ 
  (∀ x₁ x₂ : ℝ, (k * x₁^2 + 2 * x₁ + 3 = 2 * x₁ + 5) ∧ 
                (k * x₂^2 + 2 * x₂ + 3 = 2 * x₂ + 5) → 
              x₁ = x₂) ↔ k = -1/2 :=
by
  sorry

end determine_k_for_intersection_l190_190328


namespace numPerfectSquaresOrCubesLessThan1000_l190_190220

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l190_190220


namespace find_x_plus_y_l190_190357

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 10) : x + y = 26/5 :=
sorry

end find_x_plus_y_l190_190357


namespace cube_of_99999_is_correct_l190_190783

theorem cube_of_99999_is_correct : (99999 : ℕ)^3 = 999970000299999 :=
by
  sorry

end cube_of_99999_is_correct_l190_190783


namespace each_person_bid_count_l190_190314

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l190_190314


namespace max_value_of_y_l190_190226

theorem max_value_of_y (x : ℝ) (h : 0 < x ∧ x < 1 / 2) : (∃ y, y = x^2 * (1 - 2*x) ∧ y ≤ 1 / 27) :=
sorry

end max_value_of_y_l190_190226


namespace tan_alpha_add_pi_over_4_l190_190374

open Real

theorem tan_alpha_add_pi_over_4 
  (α : ℝ)
  (h1 : tan α = sqrt 3) : 
  tan (α + π / 4) = -2 - sqrt 3 :=
by
  sorry

end tan_alpha_add_pi_over_4_l190_190374


namespace ram_pairs_sold_correct_l190_190151

-- Define the costs
def graphics_card_cost := 600
def hard_drive_cost := 80
def cpu_cost := 200
def ram_pair_cost := 60

-- Define the number of items sold
def graphics_cards_sold := 10
def hard_drives_sold := 14
def cpus_sold := 8
def total_earnings := 8960

-- Calculate earnings from individual items
def earnings_graphics_cards := graphics_cards_sold * graphics_card_cost
def earnings_hard_drives := hard_drives_sold * hard_drive_cost
def earnings_cpus := cpus_sold * cpu_cost

-- Calculate total earnings from graphics cards, hard drives, and CPUs
def earnings_other_items := earnings_graphics_cards + earnings_hard_drives + earnings_cpus

-- Calculate earnings from RAM
def earnings_from_ram := total_earnings - earnings_other_items

-- Calculate number of RAM pairs sold
def ram_pairs_sold := earnings_from_ram / ram_pair_cost

-- The theorem to be proven
theorem ram_pairs_sold_correct : ram_pairs_sold = 4 :=
by
  sorry

end ram_pairs_sold_correct_l190_190151


namespace unique_positive_b_solution_exists_l190_190581

theorem unique_positive_b_solution_exists (c : ℝ) (k : ℝ) :
  (∃b : ℝ, b > 0 ∧ ∀x : ℝ, x^2 + (b + 1/b) * x + c = 0 → x = 0) ∧
  (∀b : ℝ, b^4 + (2 - 4 * c) * b^2 + k = 0) → c = 1 :=
by
  sorry

end unique_positive_b_solution_exists_l190_190581


namespace average_bull_weight_l190_190504

def ratioA : ℚ := 7 / 28  -- Ratio of cows to total cattle in section A
def ratioB : ℚ := 5 / 20  -- Ratio of cows to total cattle in section B
def ratioC : ℚ := 3 / 12  -- Ratio of cows to total cattle in section C

def total_cattle : ℕ := 1220  -- Total cattle on the farm
def total_bull_weight : ℚ := 200000  -- Total weight of bulls in kg

theorem average_bull_weight :
  ratioA = 7 / 28 ∧
  ratioB = 5 / 20 ∧
  ratioC = 3 / 12 ∧
  total_cattle = 1220 ∧
  total_bull_weight = 200000 →
  ∃ avg_weight : ℚ, avg_weight = 218.579 :=
sorry

end average_bull_weight_l190_190504


namespace area_of_square_field_l190_190281

theorem area_of_square_field (side_length : ℕ) (h : side_length = 25) :
  side_length * side_length = 625 := by
  sorry

end area_of_square_field_l190_190281


namespace manny_had_3_pies_l190_190094

-- Definitions of the conditions
def number_of_classmates : ℕ := 24
def number_of_teachers : ℕ := 1
def slices_per_pie : ℕ := 10
def slices_left : ℕ := 4

-- Number of people including Manny
def number_of_people : ℕ := number_of_classmates + number_of_teachers + 1

-- Total number of slices eaten
def slices_eaten : ℕ := number_of_people

-- Total number of slices initially
def total_slices : ℕ := slices_eaten + slices_left

-- Number of pies Manny had
def number_of_pies : ℕ := (total_slices / slices_per_pie) + 1

-- Theorem statement
theorem manny_had_3_pies : number_of_pies = 3 := by
  sorry

end manny_had_3_pies_l190_190094


namespace minimum_value_expr_eq_neg6680_25_l190_190472

noncomputable def expr (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200

theorem minimum_value_expr_eq_neg6680_25 : ∃ x : ℝ, (∀ y : ℝ, expr y ≥ expr x) ∧ expr x = -6680.25 :=
sorry

end minimum_value_expr_eq_neg6680_25_l190_190472


namespace correct_operation_c_l190_190972

theorem correct_operation_c (a b : ℝ) :
  ¬ (a^2 + a^2 = 2 * a^4)
  ∧ ¬ ((-3 * a * b^2)^2 = -6 * a^2 * b^4)
  ∧ a^6 / (-a)^2 = a^4
  ∧ ¬ ((a - b)^2 = a^2 - b^2) :=
by
  sorry

end correct_operation_c_l190_190972


namespace sum_T_mod_1000_l190_190190

open Nat

def T (a b : ℕ) : ℕ :=
  if h : a + b ≤ 6 then Nat.choose 6 a * Nat.choose 6 b * Nat.choose 6 (a + b) else 0

def sum_T : ℕ :=
  (Finset.range 7).sum (λ a => (Finset.range (7 - a)).sum (λ b => T a b))

theorem sum_T_mod_1000 : sum_T % 1000 = 564 := by
  sorry

end sum_T_mod_1000_l190_190190


namespace sum_of_possible_values_l190_190905

noncomputable def solution : ℕ :=
  sorry

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| - 4 = 0) : solution = 10 :=
by
  sorry

end sum_of_possible_values_l190_190905


namespace math_problem_l190_190213

theorem math_problem 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (h_def : ∀ x, f x = 2 * Real.sin (2 * x + phi) + 1)
  (h_point : f 0 = 0)
  (h_phi_range : -Real.pi / 2 < phi ∧ phi < 0) : 
  (phi = -Real.pi / 6) ∧ (∃ k : ℤ, ∀ x, f x = 3 ↔ x = k * Real.pi + 2 * Real.pi / 3) :=
sorry

end math_problem_l190_190213


namespace calculate_star_difference_l190_190478

def star (a b : ℕ) : ℕ := a^2 + 2 * a * b + b^2

theorem calculate_star_difference : (star 3 5) - (star 2 4) = 28 := by
  sorry

end calculate_star_difference_l190_190478


namespace initial_welders_count_l190_190404

theorem initial_welders_count (W : ℕ) (h1: (1 + 16 * (W - 9) / W = 8)) : W = 16 :=
by {
  sorry
}

end initial_welders_count_l190_190404


namespace final_total_cost_is_12_70_l190_190431

-- Definitions and conditions
def sandwich_count : ℕ := 2
def sandwich_cost_per_unit : ℝ := 2.45

def soda_count : ℕ := 4
def soda_cost_per_unit : ℝ := 0.87

def chips_count : ℕ := 3
def chips_cost_per_unit : ℝ := 1.29

def sandwich_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

-- Final price after discount and tax
noncomputable def total_cost : ℝ :=
  let sandwiches_total := sandwich_count * sandwich_cost_per_unit
  let discounted_sandwiches := sandwiches_total * (1 - sandwich_discount)
  let sodas_total := soda_count * soda_cost_per_unit
  let chips_total := chips_count * chips_cost_per_unit
  let subtotal := discounted_sandwiches + sodas_total + chips_total
  let final_total := subtotal * (1 + sales_tax)
  final_total

theorem final_total_cost_is_12_70 : total_cost = 12.70 :=
by 
  sorry

end final_total_cost_is_12_70_l190_190431


namespace positive_solution_for_y_l190_190610

theorem positive_solution_for_y (x y z : ℝ) 
  (h1 : x * y = 4 - x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z)
  (h3 : x * z = 40 - 5 * x - 2 * z) : y = 2 := 
sorry

end positive_solution_for_y_l190_190610


namespace area_of_rectangle_l190_190902

theorem area_of_rectangle (AB AC : ℝ) (angle_ABC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) (h_angle_ABC : angle_ABC = 90) :
  ∃ BC : ℝ, (BC = 8) ∧ (AB * BC = 120) :=
by
  sorry

end area_of_rectangle_l190_190902


namespace lesser_of_two_numbers_l190_190419

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 :=
by
  sorry

end lesser_of_two_numbers_l190_190419


namespace positive_number_property_l190_190838

-- Define the problem conditions and the goal
theorem positive_number_property (y : ℝ) (hy : y > 0) (h : y^2 / 100 = 9) : y = 30 := by
  sorry

end positive_number_property_l190_190838


namespace intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l190_190216

theorem intersect_at_point_m_eq_1_3_n_eq_neg_73_9 
  (m : ℚ) (n : ℚ) : 
  (m^2 + 8 + n = 0) ∧ (3*m - 1 = 0) → 
  (m = 1/3 ∧ n = -73/9) := 
by 
  sorry

theorem lines_parallel_pass_through 
  (m : ℚ) (n : ℚ) :
  (m ≠ 0) → (m^2 = 16) ∧ (3*m - 8 + n = 0) → 
  (m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20) :=
by 
  sorry

theorem lines_perpendicular_y_intercept 
  (m : ℚ) (n : ℚ) :
  (m = 0 ∧ 8*(-1) + n = 0) → 
  (m = 0 ∧ n = 8) :=
by 
  sorry

end intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l190_190216


namespace mass_percentage_Ca_in_mixture_l190_190720

theorem mass_percentage_Ca_in_mixture :
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  percentage_Ca = 26.69 :=
by
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  have : percentage_Ca = 26.69 := by sorry
  exact this

end mass_percentage_Ca_in_mixture_l190_190720


namespace problem1_problem2_l190_190018

-- Definitions for the inequalities
def f (x a : ℝ) : ℝ := abs (x - a) - 1

-- Problem 1: Given a = 2, solve the inequality f(x) + |2x - 3| > 0
theorem problem1 (x : ℝ) (h1 : abs (x - 2) + abs (2 * x - 3) > 1) : (x ≥ 2 ∨ x ≤ 4 / 3) := sorry

-- Problem 2: If the inequality f(x) > |x - 3| has solutions, find the range of a
theorem problem2 (a : ℝ) (h2 : ∃ x : ℝ, abs (x - a) - abs (x - 3) > 1) : a < 2 ∨ a > 4 := sorry

end problem1_problem2_l190_190018


namespace barefoot_kids_count_l190_190125

def kidsInClassroom : Nat := 35
def kidsWearingSocks : Nat := 18
def kidsWearingShoes : Nat := 15
def kidsWearingBoth : Nat := 8

def barefootKids : Nat := kidsInClassroom - (kidsWearingSocks - kidsWearingBoth + kidsWearingShoes - kidsWearingBoth + kidsWearingBoth)

theorem barefoot_kids_count : barefootKids = 10 := by
  sorry

end barefoot_kids_count_l190_190125


namespace cindy_age_l190_190039

-- Define the ages involved
variables (C J M G : ℕ)

-- Define the conditions
def jan_age_condition : Prop := J = C + 2
def marcia_age_condition : Prop := M = 2 * J
def greg_age_condition : Prop := G = M + 2
def greg_age_known : Prop := G = 16

-- The statement we need to prove
theorem cindy_age : 
  jan_age_condition C J → 
  marcia_age_condition J M → 
  greg_age_condition M G → 
  greg_age_known G → 
  C = 5 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end cindy_age_l190_190039


namespace tubs_from_usual_vendor_l190_190561

def total_tubs_needed : Nat := 100
def tubs_in_storage : Nat := 20
def fraction_from_new_vendor : Rat := 1 / 4

theorem tubs_from_usual_vendor :
  let remaining_tubs := total_tubs_needed - tubs_in_storage
  let tubs_from_new_vendor := remaining_tubs * fraction_from_new_vendor
  let tubs_from_usual_vendor := remaining_tubs - tubs_from_new_vendor
  tubs_from_usual_vendor = 60 :=
by
  intro remaining_tubs tubs_from_new_vendor
  exact sorry

end tubs_from_usual_vendor_l190_190561


namespace trigonometric_identity_l190_190483

-- Define the conditions and the target statement
theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l190_190483


namespace max_tank_volume_l190_190460

-- Defining the conditions given in the problem.
def side_length : ℝ := 120

-- The function for volume given side length x.
def volume (x : ℝ) : ℝ := - (1 / 2) * x^3 + 60 * x^2

-- The conditions for x.
def valid_x (x : ℝ) : Prop := 0 < x ∧ x < side_length

-- The statement to be proved.
theorem max_tank_volume : ∃ x, valid_x x ∧ 
  ∀ y, valid_x y → volume y ≤ volume x ∧ 
  volume x = 128000 :=
sorry

end max_tank_volume_l190_190460


namespace right_triangle_second_arm_square_l190_190362

theorem right_triangle_second_arm_square :
  ∀ (k : ℤ) (a : ℤ) (c : ℤ) (b : ℤ),
  a = 2 * k + 1 → 
  c = 2 * k + 3 → 
  a^2 + b^2 = c^2 → 
  b^2 ≠ a * c ∧ b^2 ≠ (c / a) ∧ b^2 ≠ (a + c) ∧ b^2 ≠ (c - a) :=
by sorry

end right_triangle_second_arm_square_l190_190362


namespace Jaymee_is_22_l190_190914

-- Define Shara's age
def Shara_age : ℕ := 10

-- Define Jaymee's age according to the problem conditions
def Jaymee_age : ℕ := 2 + 2 * Shara_age

-- The proof statement to show that Jaymee's age is 22
theorem Jaymee_is_22 : Jaymee_age = 22 := by 
  -- The proof is omitted according to the instructions.
  sorry

end Jaymee_is_22_l190_190914


namespace normal_price_of_article_l190_190143

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20) 
  (h3 : sale_price = 72) 
  (h4 : sale_price = (P * (1 - discount1)) * (1 - discount2)) : 
  P = 100 :=
by 
  sorry

end normal_price_of_article_l190_190143


namespace perimeter_original_rectangle_l190_190663

variable {L W : ℕ}

axiom area_original : L * W = 360
axiom area_changed : (L + 10) * (W - 6) = 360

theorem perimeter_original_rectangle : 2 * (L + W) = 76 :=
by
  sorry

end perimeter_original_rectangle_l190_190663


namespace expression_independent_of_alpha_l190_190262

theorem expression_independent_of_alpha
  (α : Real) (n : ℤ) (h : α ≠ (n * (π / 2)) + (π / 12)) :
  (1 - 2 * Real.sin (α - (3 * π / 2))^2 + (Real.sqrt 3) * Real.cos (2 * α + (3 * π / 2))) /
  (Real.sin (π / 6 - 2 * α)) = -2 := 
sorry

end expression_independent_of_alpha_l190_190262


namespace find_number_l190_190763

theorem find_number (x n : ℝ) (h1 : 0.12 / x * n = 12) (h2 : x = 0.1) : n = 10 := by
  sorry

end find_number_l190_190763


namespace horner_v1_value_l190_190059

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

def horner (x : ℝ) (coeffs : List ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_v1_value :
  let x := 5
  let coeffs := [4, -12, 3.5, -2.6, 1.7, -0.8]
  let v0 := coeffs.head!
  let v1 := v0 * x + coeffs.getD 1 0
  v1 = 8 := by
  -- skip the actual proof steps
  sorry

end horner_v1_value_l190_190059


namespace total_wet_surface_area_l190_190439

-- Necessary definitions based on conditions
def length : ℝ := 6
def width : ℝ := 4
def water_level : ℝ := 1.25

-- Defining the areas
def bottom_area : ℝ := length * width
def side_area (height : ℝ) (side_length : ℝ) : ℝ := height * side_length

-- Proof statement
theorem total_wet_surface_area :
  bottom_area + 2 * side_area water_level length + 2 * side_area water_level width = 49 := 
sorry

end total_wet_surface_area_l190_190439


namespace no_solution_value_of_m_l190_190891

theorem no_solution_value_of_m (m : ℤ) : ¬ ∃ x : ℤ, x ≠ 3 ∧ (x - 5) * (x - 3) = (m * (x - 3) + 2 * (x - 3) * (x - 3)) → m = -2 :=
by
  sorry

end no_solution_value_of_m_l190_190891


namespace prob_red_on_fourth_draw_expectation_xi_l190_190233

-- Definitions for the problem conditions
def total_balls : ℕ := 8
def red_balls : ℕ := 5
def white_balls : ℕ := 3
def draws (n : ℕ) : ℕ := n

-- Probability calculation for drawing a red ball on the fourth draw
theorem prob_red_on_fourth_draw : 
  (probability_of_drawing_red_ball_on_nth_draw 4 total_balls red_balls white_balls) = 5 / 14 :=
by
  sorry

-- Let ξ be the number of red balls drawn in the first three draws
def xi : ℕ := number_of_red_balls_in_first_n_draws 3 total_balls red_balls white_balls

-- Expectation calculation of ξ
theorem expectation_xi : expectation_of_xi xi = -- provide the expected value here :=
by
  sorry

end prob_red_on_fourth_draw_expectation_xi_l190_190233


namespace initial_population_is_9250_l190_190372

noncomputable def initial_population : ℝ :=
  let final_population := 6514
  let factor := (1.08 * 0.85 * (1.02)^5 * 0.95 * 0.9)
  final_population / factor

theorem initial_population_is_9250 : initial_population = 9250 := by
  sorry

end initial_population_is_9250_l190_190372


namespace sum_of_selected_terms_l190_190484

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function from natural numbers to rational numbers

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_of_selected_terms (h₁ : sum_first_n_terms a 13 = 39) : a 6 + a 7 + a 8 = 13 :=
sorry

end sum_of_selected_terms_l190_190484


namespace magic_square_sum_l190_190906

theorem magic_square_sum (a b c d e : ℕ) 
    (h1 : a + c + e = 55)
    (h2 : 30 + 10 + a = 55)
    (h3 : 30 + e + 15 = 55)
    (h4 : 10 + 30 + d = 55) :
    d + e = 25 := by
  sorry

end magic_square_sum_l190_190906


namespace geometric_sequence_sum_l190_190386

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l190_190386


namespace exactly_one_even_contradiction_assumption_l190_190396

variable (a b c : ℕ)

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)

def conclusion (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (c % 2 = 0 ∧ a % 2 = 0)

theorem exactly_one_even_contradiction_assumption :
    exactly_one_even a b c ↔ ¬ conclusion a b c :=
by
  sorry

end exactly_one_even_contradiction_assumption_l190_190396


namespace gcd_values_count_l190_190971

noncomputable def count_gcd_values (a b : ℕ) : ℕ :=
  if (a * b = 720 ∧ a + b = 50) then 1 else 0

theorem gcd_values_count : 
  (∃ a b : ℕ, a * b = 720 ∧ a + b = 50) → count_gcd_values a b = 1 :=
by
  sorry

end gcd_values_count_l190_190971


namespace chickens_and_sheep_are_ten_l190_190370

noncomputable def chickens_and_sheep_problem (C S : ℕ) : Prop :=
  (C + 4 * S = 2 * C) ∧ (2 * C + 4 * (S - 4) = 16 * (S - 4)) → (S + 2 = 10)

theorem chickens_and_sheep_are_ten (C S : ℕ) : chickens_and_sheep_problem C S :=
sorry

end chickens_and_sheep_are_ten_l190_190370


namespace f_zero_is_two_l190_190270

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x1 x2 x3 x4 x5 : ℝ) : 
  f (x1 + x2 + x3 + x4 + x5) = f x1 + f x2 + f x3 + f x4 + f x5 - 8

theorem f_zero_is_two : f 0 = 2 := 
by
  sorry

end f_zero_is_two_l190_190270


namespace investor_more_money_in_A_l190_190451

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end investor_more_money_in_A_l190_190451


namespace circle_through_point_and_tangent_to_lines_l190_190744

theorem circle_through_point_and_tangent_to_lines :
  ∃ h k,
     ((h, k) = (4 / 5, 3 / 5) ∨ (h, k) = (4, -1)) ∧ 
     ((x - h)^2 + (y - k)^2 = 5) :=
by
  let P := (3, 1)
  let l1 := fun x y => x + 2 * y + 3 
  let l2 := fun x y => x + 2 * y - 7 
  sorry

end circle_through_point_and_tangent_to_lines_l190_190744


namespace total_wheels_of_four_wheelers_l190_190900

-- Define the number of four-wheelers and wheels per four-wheeler
def number_of_four_wheelers : ℕ := 13
def wheels_per_four_wheeler : ℕ := 4

-- Prove the total number of wheels for the 13 four-wheelers
theorem total_wheels_of_four_wheelers : (number_of_four_wheelers * wheels_per_four_wheeler) = 52 :=
by sorry

end total_wheels_of_four_wheelers_l190_190900


namespace cos_alpha_plus_pi_six_l190_190073

theorem cos_alpha_plus_pi_six (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (α + Real.pi / 6) = - (4 / 5) := 
by 
  sorry

end cos_alpha_plus_pi_six_l190_190073


namespace pyramid_new_volume_l190_190839

-- Define constants
def V : ℝ := 100
def l : ℝ := 3
def w : ℝ := 2
def h : ℝ := 1.20

-- Define the theorem
theorem pyramid_new_volume : (l * w * h) * V = 720 := by
  sorry -- Proof is skipped

end pyramid_new_volume_l190_190839


namespace average_attendance_l190_190549

def monday_attendance := 10
def tuesday_attendance := 15
def wednesday_attendance := 10
def thursday_attendance := 10
def friday_attendance := 10
def total_days := 5

theorem average_attendance :
  (monday_attendance + tuesday_attendance + wednesday_attendance + thursday_attendance + friday_attendance) / total_days = 11 :=
by
  sorry

end average_attendance_l190_190549


namespace gracie_height_l190_190070

open Nat

theorem gracie_height (Griffin_height : ℕ) (Grayson_taller_than_Griffin : ℕ) (Gracie_shorter_than_Grayson : ℕ) 
  (h1 : Griffin_height = 61) (h2 : Grayson_taller_than_Griffin = 2) (h3 : Gracie_shorter_than_Grayson = 7) :
  ∃ Gracie_height, Gracie_height = 56 :=
by 
  let Grayson_height := Griffin_height + Grayson_taller_than_Griffin
  let Gracie_height := Grayson_height - Gracie_shorter_than_Grayson
  have h: Gracie_height = 56 := by
    rw [Grayson_height, Gracie_height, h1, h2, h3]
    simp
  exact ⟨56, h⟩

end gracie_height_l190_190070


namespace problem1_problem2_l190_190548

-- Definitions for permutation and combination
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problems statements
theorem problem1 : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by 
  sorry

theorem problem2 :
  C 200 198 + C 200 196 + 2 * C 200 197 = C 202 4 := by 
  sorry

end problem1_problem2_l190_190548


namespace car_speed_l190_190766

/-- 
If a tire rotates at 400 revolutions per minute, and the circumference of the tire is 6 meters, 
the speed of the car is 144 km/h.
-/
theorem car_speed (rev_per_min : ℕ) (circumference : ℝ) (speed : ℝ) :
  rev_per_min = 400 → circumference = 6 → speed = 144 :=
by
  intro h_rev h_circ
  sorry

end car_speed_l190_190766


namespace student_2005_says_1_l190_190366

def pattern : List ℕ := [1, 2, 3, 4, 3, 2]

def nth_number_in_pattern (n : ℕ) : ℕ :=
  List.nthLe pattern (n % 6) sorry  -- The index is (n-1) % 6 because Lean indices start at 0

theorem student_2005_says_1 : nth_number_in_pattern 2005 = 1 := 
  by
  -- The proof goes here
  sorry

end student_2005_says_1_l190_190366


namespace find_value_of_a_l190_190751

theorem find_value_of_a (x a : ℝ) (h : 2 * x - a + 5 = 0) (h_x : x = -2) : a = 1 :=
by
  sorry

end find_value_of_a_l190_190751


namespace instantaneous_velocity_at_1_l190_190774

noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_at_1 :
  (deriv h 1) = -3.3 :=
by
  sorry

end instantaneous_velocity_at_1_l190_190774


namespace probability_equal_dice_l190_190434

noncomputable def prob_equal_one_two_digit (n : Nat) (p : ℚ) := 
  (Finset.card (Finset.range 10) : ℚ) / n

noncomputable def prob_equal_two_digit (n : Nat) (p : ℚ) := 
  (Finset.card (Finset.Icc 10 15) : ℚ) / n

theorem probability_equal_dice (n : Nat) (k : Nat) : 
  let p1 := prob_equal_one_two_digit 15,
      p2 := prob_equal_two_digit 15,
      bincomb := nat.choose 6 3,
      prob_part := (p1 ^ 3) * (p2 ^ 3)
  in (bincomb * prob_part = (4320 : ℚ) / 15625) :=
by sorry

end probability_equal_dice_l190_190434


namespace bisection_method_root_interval_l190_190920

def f (x : ℝ) : ℝ := x^3 + x - 8

theorem bisection_method_root_interval :
  f 1 < 0 → f 1.5 < 0 → f 1.75 < 0 → f 2 > 0 → ∃ x, (1.75 < x ∧ x < 2 ∧ f x = 0) :=
by
  intros h1 h15 h175 h2
  sorry

end bisection_method_root_interval_l190_190920


namespace students_in_all_classes_l190_190815

theorem students_in_all_classes (total_students : ℕ) (students_photography : ℕ) (students_music : ℕ) (students_theatre : ℕ) (students_dance : ℕ) (students_at_least_two : ℕ) (students_in_all : ℕ) :
  total_students = 30 →
  students_photography = 15 →
  students_music = 18 →
  students_theatre = 12 →
  students_dance = 10 →
  students_at_least_two = 18 →
  students_in_all = 4 :=
by
  intros
  sorry

end students_in_all_classes_l190_190815


namespace simplify_expression_l190_190401

theorem simplify_expression : 
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := 
by 
  sorry

end simplify_expression_l190_190401


namespace problem1_domain_valid_problem2_domain_valid_l190_190737

-- Definition of the domains as sets.

def domain1 (x : ℝ) : Prop := ∃ k : ℤ, x = 2 * k * Real.pi

def domain2 (x : ℝ) : Prop := (-3 ≤ x ∧ x < -Real.pi / 2) ∨ (0 < x ∧ x < Real.pi / 2)

-- Theorem statements for the domains.

theorem problem1_domain_valid (x : ℝ) : (∀ y : ℝ, y = Real.log (Real.cos x) → y ≥ 0) ↔ domain1 x := sorry

theorem problem2_domain_valid (x : ℝ) : 
  (∀ y : ℝ, y = Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x ^ 2) → y ∈ Set.Icc (-3) 3) ↔ domain2 x := sorry

end problem1_domain_valid_problem2_domain_valid_l190_190737


namespace Kaleb_got_rid_of_7_shirts_l190_190378

theorem Kaleb_got_rid_of_7_shirts (initial_shirts : ℕ) (remaining_shirts : ℕ) 
    (h1 : initial_shirts = 17) (h2 : remaining_shirts = 10) : initial_shirts - remaining_shirts = 7 := 
by
  sorry

end Kaleb_got_rid_of_7_shirts_l190_190378


namespace solve_inequalities_l190_190592

theorem solve_inequalities (x : ℝ) (h1 : |4 - x| < 5) (h2 : x^2 < 36) : (-1 < x) ∧ (x < 6) :=
by
  sorry

end solve_inequalities_l190_190592


namespace coloring_theorem_l190_190643

open Real BigOperators

def adjacent (P Q : ℤ × ℤ) : Prop :=
  (↑P.fst - ↑Q.fst)^2 + (↑P.snd - ↑Q.snd)^2 = 2 ∨
  P.fst = Q.fst ∨ P.snd = Q.snd

def colorings (n : ℕ) : ℝ :=
  (6 / Real.sqrt 33) * ((7 + Real.sqrt 33) / 2)^n - 
  (6 / Real.sqrt 33) * ((7 - Real.sqrt 33) / 2)^n

theorem coloring_theorem (n : ℕ) (hn : 0 < n) :
  ∃ f : ℤ × ℤ → ℕ, 
    (∀ P Q, adjacent P Q → f P ≠ f Q) ∧
    ∑ (x, y) in (finset.Icc (-n, -n) (n, n)), colorings n 
    = (6 / Real.sqrt 33) * ((7 + Real.sqrt 33) / 2)^n - 
      (6 / Real.sqrt 33) * ((7 - Real.sqrt 33) / 2)^n := 
sorry

end coloring_theorem_l190_190643


namespace perpendicular_tangent_line_exists_and_correct_l190_190468

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end perpendicular_tangent_line_exists_and_correct_l190_190468


namespace jaymee_is_22_l190_190912

-- Definitions based on the problem conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 2 + 2 * shara_age

-- The theorem we need to prove
theorem jaymee_is_22 : jaymee_age = 22 :=
by
  sorry

end jaymee_is_22_l190_190912


namespace find_multiple_l190_190558

theorem find_multiple:
  let number := 220025
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := number / sum
  let remainder := number % sum
  (remainder = 25) → (quotient = 220) → (quotient / diff = 2) :=
by
  intros number sum diff quotient remainder h1 h2
  sorry

end find_multiple_l190_190558


namespace width_of_door_is_correct_l190_190952

theorem width_of_door_is_correct
  (L : ℝ) (W : ℝ) (H : ℝ := 12)
  (door_height : ℝ := 6) (window_height : ℝ := 4) (window_width : ℝ := 3)
  (cost_per_square_foot : ℝ := 10) (total_cost : ℝ := 9060) :
  (L = 25 ∧ W = 15) →
  2 * (L + W) * H - (door_height * width_door + 3 * (window_height * window_width)) * cost_per_square_foot = total_cost →
  width_door = 3 :=
by
  intros h1 h2
  sorry

end width_of_door_is_correct_l190_190952


namespace train_overtakes_motorbike_time_l190_190033

theorem train_overtakes_motorbike_time :
  let train_speed_kmph := 100
  let motorbike_speed_kmph := 64
  let train_length_m := 120.0096
  let relative_speed_kmph := train_speed_kmph - motorbike_speed_kmph
  let relative_speed_m_s := (relative_speed_kmph : ℝ) * (1 / 3.6)
  let time_seconds := train_length_m / relative_speed_m_s
  time_seconds = 12.00096 :=
sorry

end train_overtakes_motorbike_time_l190_190033


namespace Gracie_height_is_correct_l190_190069

-- Given conditions
def Griffin_height : ℤ := 61
def Grayson_height : ℤ := Griffin_height + 2
def Gracie_height : ℤ := Grayson_height - 7

-- The proof problem: Prove that Gracie's height is 56 inches.
theorem Gracie_height_is_correct : Gracie_height = 56 := by
  sorry

end Gracie_height_is_correct_l190_190069


namespace value_of_A_l190_190956

def random_value (c : Char) : ℤ := sorry

-- Given conditions
axiom H_value : random_value 'H' = 12
axiom MATH_value : random_value 'M' + random_value 'A' + random_value 'T' + random_value 'H' = 40
axiom TEAM_value : random_value 'T' + random_value 'E' + random_value 'A' + random_value 'M' = 50
axiom MEET_value : random_value 'M' + random_value 'E' + random_value 'E' + random_value 'T' = 44

-- Prove that A = 28
theorem value_of_A : random_value 'A' = 28 := by
  sorry

end value_of_A_l190_190956


namespace SarahCansYesterday_l190_190544

variable (S : ℕ)
variable (LaraYesterday : ℕ := S + 30)
variable (SarahToday : ℕ := 40)
variable (LaraToday : ℕ := 70)
variable (YesterdayTotal : ℕ := LaraYesterday + S)
variable (TodayTotal : ℕ := SarahToday + LaraToday)

theorem SarahCansYesterday : 
  TodayTotal + 20 = YesterdayTotal -> 
  S = 50 :=
by
  sorry

end SarahCansYesterday_l190_190544


namespace find_k_series_sum_l190_190189

theorem find_k_series_sum (k : ℝ) :
  (2 + ∑' n : ℕ, (2 + (n + 1) * k) / 2 ^ (n + 1)) = 6 -> k = 1 :=
by 
  sorry

end find_k_series_sum_l190_190189


namespace minimize_expression_is_correct_l190_190327

noncomputable
def minimize_expression : ℝ :=
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (14, 46)
  let dist (P Q : ℝ × ℝ) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let expression (t : ℝ) := dist (t, t^2) A + dist (t, t^2) B
  classical.some (exists_min 'x, ∀ t, expression t ≥ expression x)

theorem minimize_expression_is_correct : minimize_expression = 7 / 2 :=
sorry

end minimize_expression_is_correct_l190_190327


namespace range_of_a_minus_b_l190_190622

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) : -1 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l190_190622
