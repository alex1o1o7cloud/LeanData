import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l2248_224846

theorem solve_for_x (x : ℚ) (h : 5 * (x - 4) = 3 * (3 - 3 * x) + 6) : x = 5 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l2248_224846


namespace NUMINAMATH_GPT_distance_focus_to_asymptote_of_hyperbola_l2248_224838

open Real

noncomputable def distance_from_focus_to_asymptote_of_hyperbola : ℝ :=
  let a := 2
  let b := 1
  let c := sqrt (a^2 + b^2)
  let foci1 := (sqrt (a^2 + b^2), 0)
  let foci2 := (-sqrt (a^2 + b^2), 0)
  let asymptote_slope := a / b
  let distance_formula := (|abs (sqrt 5)|) / (sqrt (1 + asymptote_slope^2))
  distance_formula

theorem distance_focus_to_asymptote_of_hyperbola :
  distance_from_focus_to_asymptote_of_hyperbola = 1 :=
sorry

end NUMINAMATH_GPT_distance_focus_to_asymptote_of_hyperbola_l2248_224838


namespace NUMINAMATH_GPT_intercept_sum_l2248_224836

-- Define the equation of the line and the condition on the intercepts.
theorem intercept_sum (c : ℚ) (x y : ℚ) (h1 : 3 * x + 5 * y + c = 0) (h2 : x + y = 55/4) : 
  c = 825/32 :=
sorry

end NUMINAMATH_GPT_intercept_sum_l2248_224836


namespace NUMINAMATH_GPT_gcd_factorials_l2248_224849

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := by
  sorry

end NUMINAMATH_GPT_gcd_factorials_l2248_224849


namespace NUMINAMATH_GPT_cost_split_difference_l2248_224865

-- Definitions of amounts paid
def SarahPaid : ℕ := 150
def DerekPaid : ℕ := 210
def RitaPaid : ℕ := 240

-- Total paid by all three
def TotalPaid : ℕ := SarahPaid + DerekPaid + RitaPaid

-- Each should have paid:
def EachShouldHavePaid : ℕ := TotalPaid / 3

-- Amount Sarah owes Rita
def SarahOwesRita : ℕ := EachShouldHavePaid - SarahPaid

-- Amount Derek should receive back from Rita
def DerekShouldReceiveFromRita : ℕ := DerekPaid - EachShouldHavePaid

-- Difference between the amounts Sarah and Derek owe/should receive from Rita
theorem cost_split_difference : SarahOwesRita - DerekShouldReceiveFromRita = 60 := by
    sorry

end NUMINAMATH_GPT_cost_split_difference_l2248_224865


namespace NUMINAMATH_GPT_product_increase_by_13_l2248_224850

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end NUMINAMATH_GPT_product_increase_by_13_l2248_224850


namespace NUMINAMATH_GPT_problem_f_2009_plus_f_2010_l2248_224832

theorem problem_f_2009_plus_f_2010 (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (2 * x + 1) = f (2 * (x + 5 / 2) + 1))
  (h_f1 : f 1 = 5) :
  f 2009 + f 2010 = 0 :=
sorry

end NUMINAMATH_GPT_problem_f_2009_plus_f_2010_l2248_224832


namespace NUMINAMATH_GPT_range_of_a_plus_2014b_l2248_224839

theorem range_of_a_plus_2014b (a b : ℝ) (h1 : a < b) (h2 : |(Real.log a) / (Real.log 2)| = |(Real.log b) / (Real.log 2)|) :
  ∃ c : ℝ, c > 2015 ∧ ∀ x : ℝ, a + 2014 * b = x → x > 2015 := by
  sorry

end NUMINAMATH_GPT_range_of_a_plus_2014b_l2248_224839


namespace NUMINAMATH_GPT_triangle_area_l2248_224856

theorem triangle_area (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 54 := by
  -- conditions provided
  sorry

end NUMINAMATH_GPT_triangle_area_l2248_224856


namespace NUMINAMATH_GPT_closest_integer_to_cube_root_of_150_l2248_224802

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_closest_integer_to_cube_root_of_150_l2248_224802


namespace NUMINAMATH_GPT_females_with_advanced_degrees_l2248_224801

theorem females_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_degree_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_degree_only = 40) :
  (total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60) :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_females_with_advanced_degrees_l2248_224801


namespace NUMINAMATH_GPT_inequality_solution_l2248_224823

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem inequality_solution (a b : ℝ) 
  (h1 : ∀ (x : ℝ), f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ (x : ℝ), f a b (-2 * x) < 0 ↔ x < -3 / 2 ∨ x > 1 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2248_224823


namespace NUMINAMATH_GPT_smallest_positive_debt_resolvable_l2248_224898

theorem smallest_positive_debt_resolvable :
  ∃ D : ℤ, D > 0 ∧ (D = 250 * p + 175 * g + 125 * s ∧ 
  (∀ (D' : ℤ), D' > 0 → (∃ p g s : ℤ, D' = 250 * p + 175 * g + 125 * s) → D' ≥ D)) := 
sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolvable_l2248_224898


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2248_224847

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  2 * a + b + c ≥ 4 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2248_224847


namespace NUMINAMATH_GPT_probability_even_distinct_digits_l2248_224882

theorem probability_even_distinct_digits :
  let count_even_distinct := 1960
  let total_numbers := 8000
  count_even_distinct / total_numbers = 49 / 200 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_distinct_digits_l2248_224882


namespace NUMINAMATH_GPT_diameter_of_larger_circle_l2248_224840

theorem diameter_of_larger_circle (R r D : ℝ) 
  (h1 : R^2 - r^2 = 25) 
  (h2 : D = 2 * R) : 
  D = Real.sqrt (100 + 4 * r^2) := 
by 
  sorry

end NUMINAMATH_GPT_diameter_of_larger_circle_l2248_224840


namespace NUMINAMATH_GPT_total_flying_days_l2248_224862

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end NUMINAMATH_GPT_total_flying_days_l2248_224862


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l2248_224818

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l2248_224818


namespace NUMINAMATH_GPT_set_A_enumeration_l2248_224875

-- Define the conditions of the problem.
def A : Set ℕ := { x | ∃ (n : ℕ), 6 = n * (6 - x) }

-- State the theorem to be proved.
theorem set_A_enumeration : A = {0, 2, 3, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_set_A_enumeration_l2248_224875


namespace NUMINAMATH_GPT_maximum_smallest_triplet_sum_l2248_224803

theorem maximum_smallest_triplet_sum (circle : Fin 10 → ℕ) (h : ∀ i : Fin 10, 1 ≤ circle i ∧ circle i ≤ 10 ∧ ∀ j k, j ≠ k → circle j ≠ circle k):
  ∃ (i : Fin 10), ∀ j ∈ ({i, i + 1, i + 2} : Finset (Fin 10)), circle i + circle (i + 1) + circle (i + 2) ≤ 15 :=
sorry

end NUMINAMATH_GPT_maximum_smallest_triplet_sum_l2248_224803


namespace NUMINAMATH_GPT_simple_interest_difference_l2248_224873

/-- The simple interest on a certain amount at a 4% rate for 5 years amounted to a certain amount less than the principal. The principal was Rs 2400. Prove that the difference between the principal and the simple interest is Rs 1920. 
-/
theorem simple_interest_difference :
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  P - SI = 1920 :=
by
  /- We introduce the let definitions for the conditions and then state the theorem
    with the conclusion that needs to be proved. -/
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  /- The final step where we would conclude our theorem. -/
  sorry

end NUMINAMATH_GPT_simple_interest_difference_l2248_224873


namespace NUMINAMATH_GPT_Lynne_bought_3_magazines_l2248_224831

open Nat

def books_about_cats : Nat := 7
def books_about_solar_system : Nat := 2
def book_cost : Nat := 7
def magazine_cost : Nat := 4
def total_spent : Nat := 75

theorem Lynne_bought_3_magazines:
  let total_books := books_about_cats + books_about_solar_system
  let total_cost_books := total_books * book_cost
  let total_cost_magazines := total_spent - total_cost_books
  total_cost_magazines / magazine_cost = 3 :=
by sorry

end NUMINAMATH_GPT_Lynne_bought_3_magazines_l2248_224831


namespace NUMINAMATH_GPT_renovation_days_l2248_224813

/-
Conditions:
1. Cost to hire a company: 50000 rubles
2. Cost of buying materials: 20000 rubles
3. Husband's daily wage: 2000 rubles
4. Wife's daily wage: 1500 rubles
Question:
How many workdays can they spend on the renovation to make it more cost-effective?
-/

theorem renovation_days (cost_hire_company cost_materials : ℕ) 
  (husband_daily_wage wife_daily_wage : ℕ) 
  (more_cost_effective_days : ℕ) :
  cost_hire_company = 50000 → 
  cost_materials = 20000 → 
  husband_daily_wage = 2000 → 
  wife_daily_wage = 1500 → 
  more_cost_effective_days = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_renovation_days_l2248_224813


namespace NUMINAMATH_GPT_unit_digit_product_7858_1086_4582_9783_l2248_224859

-- Define the unit digits of the given numbers
def unit_digit_7858 : ℕ := 8
def unit_digit_1086 : ℕ := 6
def unit_digit_4582 : ℕ := 2
def unit_digit_9783 : ℕ := 3

-- Define a function to calculate the unit digit of a product of two numbers based on their unit digits
def unit_digit_product (a b : ℕ) : ℕ :=
  (a * b) % 10

-- The theorem that states the unit digit of the product of the numbers is 4
theorem unit_digit_product_7858_1086_4582_9783 :
  unit_digit_product (unit_digit_product (unit_digit_product unit_digit_7858 unit_digit_1086) unit_digit_4582) unit_digit_9783 = 4 :=
  by
  sorry

end NUMINAMATH_GPT_unit_digit_product_7858_1086_4582_9783_l2248_224859


namespace NUMINAMATH_GPT_possible_values_n_l2248_224896

theorem possible_values_n (n : ℕ) (h_pos : 0 < n) (h1 : n > 9 / 4) (h2 : n < 14) :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ k ∈ S, k = n :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_possible_values_n_l2248_224896


namespace NUMINAMATH_GPT_find_k_and_a_range_l2248_224869

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^2 + Real.exp x - k * Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem find_k_and_a_range (k a : ℝ) (h_even : ∀ x : ℝ, f x k = f (-x) k) :
  k = -1 ∧ 2 ≤ a := by
    sorry

end NUMINAMATH_GPT_find_k_and_a_range_l2248_224869


namespace NUMINAMATH_GPT_palm_meadows_total_beds_l2248_224897

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end NUMINAMATH_GPT_palm_meadows_total_beds_l2248_224897


namespace NUMINAMATH_GPT_range_of_a_l2248_224830

-- Define the propositions
def Proposition_p (a : ℝ) := ∀ x : ℝ, x > 0 → x + 1/x > a
def Proposition_q (a : ℝ) := ∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0

-- Define the main theorem
theorem range_of_a (a : ℝ) (h1 : ¬ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) 
(h2 : (∀ x : ℝ, x > 0 → x + 1/x > a) ∧ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) :
a ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2248_224830


namespace NUMINAMATH_GPT_moles_of_KOH_combined_l2248_224806

theorem moles_of_KOH_combined (H2O_formed : ℕ) (NH4I_used : ℕ) (ratio_KOH_H2O : ℕ) : H2O_formed = 54 → NH4I_used = 3 → ratio_KOH_H2O = 1 → H2O_formed = NH4I_used := 
by 
  intro H2O_formed_eq NH4I_used_eq ratio_eq 
  sorry

end NUMINAMATH_GPT_moles_of_KOH_combined_l2248_224806


namespace NUMINAMATH_GPT_sequence_is_odd_l2248_224851

theorem sequence_is_odd (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 7) 
  (h3 : ∀ n ≥ 2, -1/2 < (a (n + 1)) - (a n) * (a n) / a (n-1) ∧
                (a (n + 1)) - (a n) * (a n) / a (n-1) ≤ 1/2) :
  ∀ n > 1, (a n) % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_odd_l2248_224851


namespace NUMINAMATH_GPT_correct_statement_l2248_224888

theorem correct_statement (a b : ℝ) (ha : a < b) (hb : b < 0) : |a| / |b| > 1 :=
sorry

end NUMINAMATH_GPT_correct_statement_l2248_224888


namespace NUMINAMATH_GPT_smallest_possible_r_l2248_224889

theorem smallest_possible_r (p q r : ℤ) (hpq: p < q) (hqr: q < r) 
  (hgeo: q^2 = p * r) (harith: 2 * q = p + r) : r = 4 :=
sorry

end NUMINAMATH_GPT_smallest_possible_r_l2248_224889


namespace NUMINAMATH_GPT_find_x_l2248_224805

theorem find_x (x : ℚ) : (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2248_224805


namespace NUMINAMATH_GPT_numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l2248_224894

theorem numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1 :
  (63 ∣ 2^48 - 1) ∧ (65 ∣ 2^48 - 1) := 
by
  sorry

end NUMINAMATH_GPT_numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l2248_224894


namespace NUMINAMATH_GPT_q_alone_time_24_days_l2248_224895

theorem q_alone_time_24_days:
  ∃ (Wq : ℝ), (∀ (Wp Ws : ℝ), 
    Wp = Wq + 1 / 60 → 
    Wp + Wq = 1 / 10 → 
    Wp + 1 / 60 + 2 * Wq = 1 / 6 → 
    1 / Wq = 24) :=
by
  sorry

end NUMINAMATH_GPT_q_alone_time_24_days_l2248_224895


namespace NUMINAMATH_GPT_charlie_cookies_l2248_224800

theorem charlie_cookies (father_cookies mother_cookies total_cookies charlie_cookies : ℕ)
  (h1 : father_cookies = 10) (h2 : mother_cookies = 5) (h3 : total_cookies = 30) :
  father_cookies + mother_cookies + charlie_cookies = total_cookies → charlie_cookies = 15 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_charlie_cookies_l2248_224800


namespace NUMINAMATH_GPT_valid_conditions_x_y_z_l2248_224884

theorem valid_conditions_x_y_z (x y z : ℤ) :
  x = y - 1 ∧ z = y + 1 ∨ x = y ∧ z = y + 1 ↔ x * (x - y) + y * (y - x) + z * (z - y) = 1 :=
sorry

end NUMINAMATH_GPT_valid_conditions_x_y_z_l2248_224884


namespace NUMINAMATH_GPT_dice_sum_probability_l2248_224899

theorem dice_sum_probability :
  let total_outcomes := 36
  let sum_le_8_outcomes := 13
  (sum_le_8_outcomes : ℕ) / (total_outcomes : ℕ) = (13 / 18 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_dice_sum_probability_l2248_224899


namespace NUMINAMATH_GPT_hyperbola_asymptote_l2248_224855

def hyperbola_eqn (m x y : ℝ) := m * x^2 - y^2 = 1

def vertex_distance_condition (m : ℝ) := 2 * Real.sqrt (1 / m) = 4

theorem hyperbola_asymptote (m : ℝ) (h_eq : hyperbola_eqn m x y) (h_dist : vertex_distance_condition m) :
  ∃ k, y = k * x ∧ k = 1 / 2 ∨ k = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l2248_224855


namespace NUMINAMATH_GPT_smallest_sum_is_381_l2248_224880

def is_valid_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def uses_digits_once (n m : ℕ) : Prop :=
  (∀ d ∈ [1, 2, 3, 4, 5, 6], (d ∈ n.digits 10 ∨ d ∈ m.digits 10)) ∧
  (∀ d, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ m.digits 10 → d ∈ [1, 2, 3, 4, 5, 6])

theorem smallest_sum_is_381 :
  ∃ (n m : ℕ), is_valid_3_digit_number n ∧ is_valid_3_digit_number m ∧
  uses_digits_once n m ∧ n + m = 381 :=
sorry

end NUMINAMATH_GPT_smallest_sum_is_381_l2248_224880


namespace NUMINAMATH_GPT_total_cost_of_cultivating_field_l2248_224834

theorem total_cost_of_cultivating_field 
  (base height : ℕ) 
  (cost_per_hectare : ℝ) 
  (base_eq: base = 3 * height) 
  (height_eq: height = 300) 
  (cost_eq: cost_per_hectare = 24.68) 
  : (1/2 : ℝ) * base * height / 10000 * cost_per_hectare = 333.18 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_cultivating_field_l2248_224834


namespace NUMINAMATH_GPT_closest_to_10_l2248_224890

theorem closest_to_10
  (A B C D : ℝ)
  (hA : A = 9.998)
  (hB : B = 10.1)
  (hC : C = 10.09)
  (hD : D = 10.001) :
  abs (10 - D) < abs (10 - A) ∧ abs (10 - D) < abs (10 - B) ∧ abs (10 - D) < abs (10 - C) :=
by
  sorry

end NUMINAMATH_GPT_closest_to_10_l2248_224890


namespace NUMINAMATH_GPT_possible_values_of_a_l2248_224821

-- Declare the sets M and N based on given conditions.
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define a proof where the set of possible values for a is {-1, 0, 2/3}
theorem possible_values_of_a : 
  {a : ℝ | N a ⊆ M} = {-1, 0, 2 / 3} := 
by 
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l2248_224821


namespace NUMINAMATH_GPT_total_time_for_process_l2248_224864

-- Given conditions
def cat_resistance_time : ℕ := 20
def walking_distance : ℕ := 64
def walking_rate : ℕ := 8

-- Prove the total time
theorem total_time_for_process : cat_resistance_time + (walking_distance / walking_rate) = 28 := by
  sorry

end NUMINAMATH_GPT_total_time_for_process_l2248_224864


namespace NUMINAMATH_GPT_sum_of_largest_and_smallest_l2248_224893

theorem sum_of_largest_and_smallest (n : ℕ) (h : 6 * n + 15 = 105) : (n + (n + 5) = 35) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_largest_and_smallest_l2248_224893


namespace NUMINAMATH_GPT_restaurant_total_cost_l2248_224866

theorem restaurant_total_cost (burger_cost pizza_cost : ℕ)
    (h1 : burger_cost = 9)
    (h2 : pizza_cost = 2 * burger_cost) :
    pizza_cost + 3 * burger_cost = 45 := 
by
  sorry

end NUMINAMATH_GPT_restaurant_total_cost_l2248_224866


namespace NUMINAMATH_GPT_scale_drawing_represents_line_segment_l2248_224827

-- Define the given conditions
def scale_factor : ℝ := 800
def line_segment_length_inch : ℝ := 4.75

-- Prove the length in feet
theorem scale_drawing_represents_line_segment :
  line_segment_length_inch * scale_factor = 3800 :=
by
  sorry

end NUMINAMATH_GPT_scale_drawing_represents_line_segment_l2248_224827


namespace NUMINAMATH_GPT_num_solutions_eq_three_l2248_224854

theorem num_solutions_eq_three :
  (∃ n : Nat, (x : ℝ) → (x^2 - 4) * (x^2 - 1) = (x^2 + 3 * x + 2) * (x^2 - 8 * x + 7) → n = 3) :=
sorry

end NUMINAMATH_GPT_num_solutions_eq_three_l2248_224854


namespace NUMINAMATH_GPT_relationship_ab_c_l2248_224829

def a := 0.8 ^ 0.8
def b := 0.8 ^ 0.9
def c := 1.2 ^ 0.8

theorem relationship_ab_c : c > a ∧ a > b := 
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_relationship_ab_c_l2248_224829


namespace NUMINAMATH_GPT_rational_solution_for_quadratic_l2248_224868

theorem rational_solution_for_quadratic (k : ℕ) (h_pos : 0 < k) : 
  ∃ m : ℕ, (18^2 - 4 * k * (2 * k)) = m^2 ↔ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_rational_solution_for_quadratic_l2248_224868


namespace NUMINAMATH_GPT_positive_number_divisible_by_4_l2248_224809

theorem positive_number_divisible_by_4 (N : ℕ) (h1 : N % 4 = 0) (h2 : (2 + 4 + N + 3) % 2 = 1) : N = 4 := 
by 
  sorry

end NUMINAMATH_GPT_positive_number_divisible_by_4_l2248_224809


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l2248_224853

theorem abs_inequality_solution_set (x : ℝ) : 
  (|2 * x - 3| ≤ 1) ↔ (1 ≤ x ∧ x ≤ 2) := 
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l2248_224853


namespace NUMINAMATH_GPT_cubic_roots_l2248_224881

open Real

theorem cubic_roots (x1 x2 x3 : ℝ) (h1 : x1 * x2 = 1)
  (h2 : 3 * x1^3 + 2 * sqrt 3 * x1^2 - 21 * x1 + 6 * sqrt 3 = 0)
  (h3 : 3 * x2^3 + 2 * sqrt 3 * x2^2 - 21 * x2 + 6 * sqrt 3 = 0)
  (h4 : 3 * x3^3 + 2 * sqrt 3 * x3^2 - 21 * x3 + 6 * sqrt 3 = 0) :
  (x1 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x1 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) := 
sorry

end NUMINAMATH_GPT_cubic_roots_l2248_224881


namespace NUMINAMATH_GPT_option_b_correct_l2248_224857

variable (Line Plane : Type)

-- Definitions for perpendicularity and parallelism
variable (perp parallel : Line → Plane → Prop) (parallel_line : Line → Line → Prop)

-- Assumptions reflecting the conditions in the problem
axiom perp_alpha_1 {a : Line} {alpha : Plane} : perp a alpha
axiom perp_alpha_2 {b : Line} {alpha : Plane} : perp b alpha

-- The statement to prove
theorem option_b_correct (a b : Line) (alpha : Plane) :
  perp a alpha → perp b alpha → parallel_line a b :=
by
  intro h1 h2
  -- proof omitted
  sorry

end NUMINAMATH_GPT_option_b_correct_l2248_224857


namespace NUMINAMATH_GPT_students_neither_correct_l2248_224843

-- Define the total number of students and the numbers for chemistry, biology, and both
def total_students := 75
def chemistry_students := 42
def biology_students := 33
def both_subject_students := 18

-- Define a function to calculate the number of students taking neither chemistry nor biology
def students_neither : ℕ :=
  total_students - ((chemistry_students - both_subject_students) 
                    + (biology_students - both_subject_students) 
                    + both_subject_students)

-- Theorem stating that the number of students taking neither chemistry nor biology is as expected
theorem students_neither_correct : students_neither = 18 :=
  sorry

end NUMINAMATH_GPT_students_neither_correct_l2248_224843


namespace NUMINAMATH_GPT_manuscript_fee_3800_l2248_224835

theorem manuscript_fee_3800 (tax_fee manuscript_fee : ℕ) 
  (h1 : tax_fee = 420) 
  (h2 : (0 < manuscript_fee) ∧ 
        (manuscript_fee ≤ 4000) → 
        tax_fee = (14 * (manuscript_fee - 800)) / 100) 
  (h3 : (manuscript_fee > 4000) → 
        tax_fee = (11 * manuscript_fee) / 100) : manuscript_fee = 3800 :=
by
  sorry

end NUMINAMATH_GPT_manuscript_fee_3800_l2248_224835


namespace NUMINAMATH_GPT_remainder_21_pow_2051_mod_29_l2248_224876

theorem remainder_21_pow_2051_mod_29 :
  ∀ (a : ℤ), (21^4 ≡ 1 [MOD 29]) -> (2051 = 4 * 512 + 3) -> (21^3 ≡ 15 [MOD 29]) -> (21^2051 ≡ 15 [MOD 29]) :=
by
  intros a h1 h2 h3
  sorry

end NUMINAMATH_GPT_remainder_21_pow_2051_mod_29_l2248_224876


namespace NUMINAMATH_GPT_least_number_remainder_l2248_224814

theorem least_number_remainder (n : ℕ) (hn : n = 115) : n % 38 = 1 ∧ n % 3 = 1 := by
  sorry

end NUMINAMATH_GPT_least_number_remainder_l2248_224814


namespace NUMINAMATH_GPT_sock_pair_selection_l2248_224828

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 5
def num_blue_socks : Nat := 3

def white_odd_positions : List Nat := [1, 3, 5]
def white_even_positions : List Nat := [2, 4]

def brown_odd_positions : List Nat := [1, 3, 5]
def brown_even_positions : List Nat := [2, 4]

def blue_odd_positions : List Nat := [1, 3]
def blue_even_positions : List Nat := [2]

noncomputable def count_pairs : Nat :=
  let white_brown := (white_odd_positions.length * brown_odd_positions.length) +
                     (white_even_positions.length * brown_even_positions.length)
  
  let brown_blue := (brown_odd_positions.length * blue_odd_positions.length) +
                    (brown_even_positions.length * blue_even_positions.length)

  let white_blue := (white_odd_positions.length * blue_odd_positions.length) +
                    (white_even_positions.length * blue_even_positions.length)

  white_brown + brown_blue + white_blue

theorem sock_pair_selection :
  count_pairs = 29 :=
by
  sorry

end NUMINAMATH_GPT_sock_pair_selection_l2248_224828


namespace NUMINAMATH_GPT_f_periodic_odd_condition_l2248_224858

theorem f_periodic_odd_condition (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 4) = f x) (h_one : f 1 = 5) : f 2015 = -5 :=
by
  sorry

end NUMINAMATH_GPT_f_periodic_odd_condition_l2248_224858


namespace NUMINAMATH_GPT_factorize_expression_l2248_224886

variables {a x y : ℝ}

theorem factorize_expression (a x y : ℝ) : 3 * a * x ^ 2 + 6 * a * x * y + 3 * a * y ^ 2 = 3 * a * (x + y) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2248_224886


namespace NUMINAMATH_GPT_range_of_x_l2248_224808

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) :
  -1 ≤ x ∧ x < 5 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_x_l2248_224808


namespace NUMINAMATH_GPT_simplify_expression_l2248_224815

variable (b : ℝ)

theorem simplify_expression (h : b ≠ 2) : (2 - 1 / (1 + b / (2 - b))) = 1 + b / 2 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l2248_224815


namespace NUMINAMATH_GPT_union_of_A_and_B_l2248_224811

-- Definitions for sets A and B
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- Theorem statement to prove the union of A and B
theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2248_224811


namespace NUMINAMATH_GPT_dan_initial_money_l2248_224844

def initial_amount (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ) : ℕ :=
  spent_candy + spent_chocolate + remaining

theorem dan_initial_money 
  (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ)
  (h_candy : spent_candy = 2)
  (h_chocolate : spent_chocolate = 3)
  (h_remaining : remaining = 2) :
  initial_amount spent_candy spent_chocolate remaining = 7 :=
by
  rw [h_candy, h_chocolate, h_remaining]
  unfold initial_amount
  rfl

end NUMINAMATH_GPT_dan_initial_money_l2248_224844


namespace NUMINAMATH_GPT_find_valid_pairs_l2248_224819

-- Definitions and conditions:
def satisfies_equation (a b : ℤ) : Prop := a^2 + a * b - b = 2018

-- Correct answers:
def valid_pairs : List (ℤ × ℤ) :=
  [(2, 2014), (0, -2018), (2018, -2018), (-2016, 2014)]

-- Statement to prove:
theorem find_valid_pairs :
  ∀ (a b : ℤ), satisfies_equation a b ↔ (a, b) ∈ valid_pairs.toFinset := by
  sorry

end NUMINAMATH_GPT_find_valid_pairs_l2248_224819


namespace NUMINAMATH_GPT_cherry_pies_count_correct_l2248_224879

def total_pies : ℕ := 36

def ratio_ap_bb_ch : (ℕ × ℕ × ℕ) := (2, 3, 4)

def total_ratio_parts : ℕ := 2 + 3 + 4

def pies_per_part (total_pies : ℕ) (total_ratio_parts : ℕ) : ℕ := total_pies / total_ratio_parts

def num_parts_ch : ℕ := 4

def num_cherry_pies (total_pies : ℕ) (total_ratio_parts : ℕ) (num_parts_ch : ℕ) : ℕ :=
  pies_per_part total_pies total_ratio_parts * num_parts_ch

theorem cherry_pies_count_correct : num_cherry_pies total_pies total_ratio_parts num_parts_ch = 16 := by
  sorry

end NUMINAMATH_GPT_cherry_pies_count_correct_l2248_224879


namespace NUMINAMATH_GPT_min_sum_chessboard_labels_l2248_224860

theorem min_sum_chessboard_labels :
  ∃ (r : Fin 9 → Fin 9), 
  (∀ (i j : Fin 9), i ≠ j → r i ≠ r j) ∧ 
  ((Finset.univ : Finset (Fin 9)).sum (λ i => 1 / (r i + i.val + 1)) = 1) :=
by
  sorry

end NUMINAMATH_GPT_min_sum_chessboard_labels_l2248_224860


namespace NUMINAMATH_GPT_probability_of_first_three_red_cards_l2248_224842

theorem probability_of_first_three_red_cards :
  let total_cards := 60
  let red_cards := 36
  let black_cards := total_cards - red_cards
  let total_ways := total_cards * (total_cards - 1) * (total_cards - 2)
  let red_ways := red_cards * (red_cards - 1) * (red_cards - 2)
  (red_ways / total_ways) = 140 / 673 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_first_three_red_cards_l2248_224842


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l2248_224863

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n - 1) = 2) : ∀ n, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l2248_224863


namespace NUMINAMATH_GPT_parabola_circle_intercept_l2248_224883

theorem parabola_circle_intercept (p : ℝ) (h_pos : p > 0) :
  (∃ (x y : ℝ), y^2 = 2 * p * x ∧ x^2 + y^2 + 2 * x - 3 = 0) ∧
  (∃ (y1 y2 : ℝ), (y1 - y2)^2 + (-(p / 2) + 1)^2 = 4^2) → p = 2 :=
by sorry

end NUMINAMATH_GPT_parabola_circle_intercept_l2248_224883


namespace NUMINAMATH_GPT_total_profit_proof_l2248_224878
-- Import the necessary libraries

-- Define the investments and profits
def investment_tom : ℕ := 3000 * 12
def investment_jose : ℕ := 4500 * 10
def profit_jose : ℕ := 3500

-- Define the ratio and profit parts
def ratio_tom : ℕ := investment_tom / Nat.gcd investment_tom investment_jose
def ratio_jose : ℕ := investment_jose / Nat.gcd investment_tom investment_jose
def ratio_total : ℕ := ratio_tom + ratio_jose
def one_part_value : ℕ := profit_jose / ratio_jose
def profit_tom : ℕ := ratio_tom * one_part_value

-- The total profit
def total_profit : ℕ := profit_tom + profit_jose

-- The theorem to prove
theorem total_profit_proof : total_profit = 6300 := by
  sorry

end NUMINAMATH_GPT_total_profit_proof_l2248_224878


namespace NUMINAMATH_GPT_probability_jack_queen_king_l2248_224833

theorem probability_jack_queen_king :
  let deck_size := 52
  let jacks := 4
  let queens := 4
  let kings := 4
  let remaining_after_jack := deck_size - 1
  let remaining_after_queen := deck_size - 2
  (jacks / deck_size) * (queens / remaining_after_jack) * (kings / remaining_after_queen) = 8 / 16575 :=
by
  sorry

end NUMINAMATH_GPT_probability_jack_queen_king_l2248_224833


namespace NUMINAMATH_GPT_perpendicular_slope_l2248_224804

-- Define the line equation and the result we want to prove about its perpendicular slope
def line_eq (x y : ℝ) := 5 * x - 2 * y = 10

theorem perpendicular_slope : ∀ (m : ℝ), 
  (∀ (x y : ℝ), line_eq x y → y = (5 / 2) * x - 5) →
  m = -(2 / 5) :=
by
  intros m H
  -- Additional logical steps would go here
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l2248_224804


namespace NUMINAMATH_GPT_equation_of_line_l2248_224807

theorem equation_of_line (x_intercept slope : ℝ)
  (hx : x_intercept = 2) (hm : slope = 1) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -1 ∧ c = -2 ∧ (∀ x y : ℝ, y = slope * (x - x_intercept) ↔ a * x + b * y + c = 0) := sorry

end NUMINAMATH_GPT_equation_of_line_l2248_224807


namespace NUMINAMATH_GPT_compare_exponential_functions_l2248_224841

theorem compare_exponential_functions (x : ℝ) (hx1 : 0 < x) :
  0.4^4 < 1 ∧ 1 < 4^0.4 :=
by sorry

end NUMINAMATH_GPT_compare_exponential_functions_l2248_224841


namespace NUMINAMATH_GPT_find_a_l2248_224816

theorem find_a (a : ℝ) : 
  (∃ r : ℕ, (10 - 3 * r = 1 ∧ (-a)^r * (Nat.choose 5 r) *  x^(10 - 2 * r - r) = x ∧ -10 = (-a)^3 * (Nat.choose 5 3)))
  → a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l2248_224816


namespace NUMINAMATH_GPT_cos_sum_formula_l2248_224810

open Real

theorem cos_sum_formula (A B C : ℝ) (h1 : sin A + sin B + sin C = 0) (h2 : cos A + cos B + cos C = 0) :
  cos (A - B) + cos (B - C) + cos (C - A) = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_sum_formula_l2248_224810


namespace NUMINAMATH_GPT_inequality_proof_l2248_224891

variable {a b c d : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2248_224891


namespace NUMINAMATH_GPT_set_complement_intersection_l2248_224885

open Set

variable (U M N : Set ℕ)

theorem set_complement_intersection :
  U = {1, 2, 3, 4, 5, 6, 7} →
  M = {3, 4, 5} →
  N = {1, 3, 6} →
  {2, 7} = (U \ M) ∩ (U \ N) :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end NUMINAMATH_GPT_set_complement_intersection_l2248_224885


namespace NUMINAMATH_GPT_player_current_average_l2248_224871

theorem player_current_average (A : ℝ) 
  (h1 : 10 * A + 76 = (A + 4) * 11) : 
  A = 32 :=
sorry

end NUMINAMATH_GPT_player_current_average_l2248_224871


namespace NUMINAMATH_GPT_no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l2248_224820

theorem no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100 :
  ¬ ∃ (a b c d : ℕ), a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 :=
by
  sorry

end NUMINAMATH_GPT_no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l2248_224820


namespace NUMINAMATH_GPT_train_length_l2248_224837

theorem train_length (L : ℝ) 
  (equal_length : ∀ (A B : ℝ), A = B → L = A)
  (same_direction : ∀ (dir1 dir2 : ℤ), dir1 = 1 → dir2 = 1)
  (speed_faster : ℝ := 50) (speed_slower : ℝ := 36)
  (time_to_pass : ℝ := 36)
  (relative_speed := speed_faster - speed_slower)
  (relative_speed_km_per_sec := relative_speed / 3600)
  (distance_covered := relative_speed_km_per_sec * time_to_pass)
  (total_distance := distance_covered)
  (length_per_train := total_distance / 2)
  (length_in_meters := length_per_train * 1000): 
  L = 70 := 
by 
  sorry

end NUMINAMATH_GPT_train_length_l2248_224837


namespace NUMINAMATH_GPT_new_weights_inequality_l2248_224824

theorem new_weights_inequality (W : ℝ) (x y : ℝ) (h_avg_increase : (8 * W - 2 * 68 + x + y) / 8 = W + 5.5)
  (h_sum_new_weights : x + y ≤ 180) : x > W ∧ y > W :=
by {
  sorry
}

end NUMINAMATH_GPT_new_weights_inequality_l2248_224824


namespace NUMINAMATH_GPT_recess_breaks_l2248_224845

theorem recess_breaks (total_outside_time : ℕ) (lunch_break : ℕ) (extra_recess : ℕ) (recess_duration : ℕ) 
  (h1 : total_outside_time = 80)
  (h2 : lunch_break = 30)
  (h3 : extra_recess = 20)
  (h4 : recess_duration = 15) : 
  (total_outside_time - (lunch_break + extra_recess)) / recess_duration = 2 := 
by {
  -- proof starts here
  sorry
}

end NUMINAMATH_GPT_recess_breaks_l2248_224845


namespace NUMINAMATH_GPT_chairs_problem_l2248_224822

theorem chairs_problem (B G W : ℕ) 
  (h1 : G = 3 * B) 
  (h2 : W = B + G - 13) 
  (h3 : B + G + W = 67) : 
  B = 10 :=
by
  sorry

end NUMINAMATH_GPT_chairs_problem_l2248_224822


namespace NUMINAMATH_GPT_find_uv_l2248_224877

open Real

def vec1 : ℝ × ℝ := (3, -2)
def vec2 : ℝ × ℝ := (-1, 2)
def vec3 : ℝ × ℝ := (1, -1)
def vec4 : ℝ × ℝ := (4, -7)
def vec5 : ℝ × ℝ := (-3, 5)

theorem find_uv (u v : ℝ) :
  vec1 + ⟨4 * u, -7 * u⟩ = vec2 + ⟨-3 * v, 5 * v⟩ + vec3 →
  u = 3 / 4 ∧ v = -9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_uv_l2248_224877


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2248_224848

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) * (x + 3) > 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2248_224848


namespace NUMINAMATH_GPT_depth_of_second_hole_l2248_224872

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let total_man_hours1 := workers1 * hours1
  let rate_of_work := depth1 / total_man_hours1
  let workers2 := 45 + 45
  let hours2 := 6
  let total_man_hours2 := workers2 * hours2
  let depth2 := rate_of_work * total_man_hours2
  depth2 = 45 := by
    sorry

end NUMINAMATH_GPT_depth_of_second_hole_l2248_224872


namespace NUMINAMATH_GPT_min_sum_ab_l2248_224861

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end NUMINAMATH_GPT_min_sum_ab_l2248_224861


namespace NUMINAMATH_GPT_thief_distance_l2248_224852

variable (d : ℝ := 250)   -- initial distance in meters
variable (v_thief : ℝ := 12 * 1000 / 3600)  -- thief's speed in m/s (converted from km/hr)
variable (v_policeman : ℝ := 15 * 1000 / 3600)  -- policeman's speed in m/s (converted from km/hr)

noncomputable def distance_thief_runs : ℝ :=
  v_thief * (d / (v_policeman - v_thief))

theorem thief_distance :
  distance_thief_runs d v_thief v_policeman = 990.47 := sorry

end NUMINAMATH_GPT_thief_distance_l2248_224852


namespace NUMINAMATH_GPT_Kayla_total_items_l2248_224817

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end NUMINAMATH_GPT_Kayla_total_items_l2248_224817


namespace NUMINAMATH_GPT_sqrt_x_minus_2_domain_l2248_224892

theorem sqrt_x_minus_2_domain {x : ℝ} : (∃y : ℝ, y = Real.sqrt (x - 2)) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_domain_l2248_224892


namespace NUMINAMATH_GPT_num_adult_tickets_l2248_224826

variables (A C : ℕ)

def num_tickets (A C : ℕ) : Prop := A + C = 900
def total_revenue (A C : ℕ) : Prop := 7 * A + 4 * C = 5100

theorem num_adult_tickets : ∃ A, ∃ C, num_tickets A C ∧ total_revenue A C ∧ A = 500 := 
by
  sorry

end NUMINAMATH_GPT_num_adult_tickets_l2248_224826


namespace NUMINAMATH_GPT_axis_of_symmetry_l2248_224874

variable (f : ℝ → ℝ)

theorem axis_of_symmetry (h : ∀ x, f x = f (5 - x)) :  ∀ x y, y = f x ↔ (x = 2.5 ∧ y = f 2.5) := 
sorry

end NUMINAMATH_GPT_axis_of_symmetry_l2248_224874


namespace NUMINAMATH_GPT_inversely_proportional_y_l2248_224825

theorem inversely_proportional_y (k : ℚ) (x y : ℚ) (hx_neg_10 : x = -10) (hy_5 : y = 5) (hprop : y * x = k) (hx_neg_4 : x = -4) : 
  y = 25 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_y_l2248_224825


namespace NUMINAMATH_GPT_toy_sword_cost_l2248_224812

theorem toy_sword_cost (L S : ℕ) (play_dough_cost total_cost : ℕ) :
    L = 250 →
    play_dough_cost = 35 →
    total_cost = 1940 →
    3 * L + 7 * S + 10 * play_dough_cost = total_cost →
    S = 120 :=
by
  intros hL h_play_dough_cost h_total_cost h_eq
  sorry

end NUMINAMATH_GPT_toy_sword_cost_l2248_224812


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l2248_224867

theorem equation_one_solution (x : ℝ) : 4 * (x - 1)^2 - 9 = 0 ↔ (x = 5 / 2) ∨ (x = - 1 / 2) := 
by sorry

theorem equation_two_solution (x : ℝ) : x^2 - 6 * x - 7 = 0 ↔ (x = 7) ∨ (x = - 1) :=
by sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l2248_224867


namespace NUMINAMATH_GPT_sequence_term_four_l2248_224887

theorem sequence_term_four (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 4 = 7 :=
sorry

end NUMINAMATH_GPT_sequence_term_four_l2248_224887


namespace NUMINAMATH_GPT_tv_cost_solution_l2248_224870

theorem tv_cost_solution (M T : ℝ) 
  (h1 : 2 * M + T = 7000)
  (h2 : M + 2 * T = 9800) : 
  T = 4200 :=
by
  sorry

end NUMINAMATH_GPT_tv_cost_solution_l2248_224870
