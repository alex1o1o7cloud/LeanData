import Mathlib

namespace NUMINAMATH_GPT_combined_motion_properties_l1338_133806

noncomputable def y (x : ℝ) := Real.sin x + (Real.sin x) ^ 2

theorem combined_motion_properties :
  (∀ x: ℝ, - (1/4: ℝ) ≤ y x ∧ y x ≤ 2) ∧ 
  (∃ x: ℝ, y x = 2) ∧
  (∃ x: ℝ, y x = -(1/4: ℝ)) :=
by
  -- The complete proofs for these statements are omitted.
  -- This theorem specifies the required properties of the function y.
  sorry

end NUMINAMATH_GPT_combined_motion_properties_l1338_133806


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1338_133843

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (x - y) * x^4 < 0 → x < y ∧ ¬(x < y → (x - y) * x^4 < 0) := 
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1338_133843


namespace NUMINAMATH_GPT_circles_positional_relationship_l1338_133875

theorem circles_positional_relationship :
  ∃ R r : ℝ, (R * r = 2 ∧ R + r = 3) ∧ 3 = R + r → "externally tangent" = "externally tangent" :=
by
  sorry

end NUMINAMATH_GPT_circles_positional_relationship_l1338_133875


namespace NUMINAMATH_GPT_speed_of_train_l1338_133890

-- Define the given conditions
def length_of_bridge : ℝ := 200
def length_of_train : ℝ := 100
def time_to_cross_bridge : ℝ := 60

-- Define the speed conversion factor
def m_per_s_to_km_per_h : ℝ := 3.6

-- Prove that the speed of the train is 18 km/h
theorem speed_of_train :
  (length_of_bridge + length_of_train) / time_to_cross_bridge * m_per_s_to_km_per_h = 18 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_train_l1338_133890


namespace NUMINAMATH_GPT_max_b_integer_l1338_133830

theorem max_b_integer (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ -10) → b ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_max_b_integer_l1338_133830


namespace NUMINAMATH_GPT_number_of_buses_l1338_133813

theorem number_of_buses (total_students : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) (buses : ℕ)
  (h1 : total_students = 375)
  (h2 : students_per_bus = 53)
  (h3 : students_in_cars = 4)
  (h4 : buses = (total_students - students_in_cars + students_per_bus - 1) / students_per_bus) :
  buses = 8 := by
  -- We will demonstrate that the number of buses indeed equals 8 under the given conditions.
  sorry

end NUMINAMATH_GPT_number_of_buses_l1338_133813


namespace NUMINAMATH_GPT_find_q_l1338_133837

theorem find_q (P J T : ℝ) (Q : ℝ) (q : ℚ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : T = P * (1 - Q))
  (h4 : Q = q / 100) :
  q = 6.25 := 
by
  sorry

end NUMINAMATH_GPT_find_q_l1338_133837


namespace NUMINAMATH_GPT_money_needed_to_finish_collection_l1338_133869

-- Define the conditions
def initial_action_figures : ℕ := 9
def total_action_figures_needed : ℕ := 27
def cost_per_action_figure : ℕ := 12

-- Define the goal
theorem money_needed_to_finish_collection 
  (initial : ℕ) (total_needed : ℕ) (cost_per : ℕ) 
  (h1 : initial = initial_action_figures)
  (h2 : total_needed = total_action_figures_needed)
  (h3 : cost_per = cost_per_action_figure) :
  ((total_needed - initial) * cost_per = 216) := 
by
  sorry

end NUMINAMATH_GPT_money_needed_to_finish_collection_l1338_133869


namespace NUMINAMATH_GPT_total_charge_for_first_4_minutes_under_plan_A_is_0_60_l1338_133868

def planA_charges (X : ℝ) (minutes : ℕ) : ℝ :=
  if minutes <= 4 then X
  else X + (minutes - 4) * 0.06

def planB_charges (minutes : ℕ) : ℝ :=
  minutes * 0.08

theorem total_charge_for_first_4_minutes_under_plan_A_is_0_60
  (X : ℝ)
  (h : planA_charges X 18 = planB_charges 18) :
  X = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_total_charge_for_first_4_minutes_under_plan_A_is_0_60_l1338_133868


namespace NUMINAMATH_GPT_european_fraction_is_one_fourth_l1338_133825

-- Define the total number of passengers
def P : ℕ := 108

-- Define the fractions and the number of passengers from each continent
def northAmerica := (1 / 12) * P
def africa := (1 / 9) * P
def asia := (1 / 6) * P
def otherContinents := 42

-- Define the total number of non-European passengers
def totalNonEuropean := northAmerica + africa + asia + otherContinents

-- Define the number of European passengers
def european := P - totalNonEuropean

-- Define the fraction of European passengers
def europeanFraction := european / P

-- Prove that the fraction of European passengers is 1/4
theorem european_fraction_is_one_fourth : europeanFraction = 1 / 4 := 
by
  unfold europeanFraction european totalNonEuropean northAmerica africa asia P
  sorry

end NUMINAMATH_GPT_european_fraction_is_one_fourth_l1338_133825


namespace NUMINAMATH_GPT_oranges_per_glass_l1338_133894

theorem oranges_per_glass (total_oranges glasses_of_juice oranges_per_glass : ℕ)
    (h_oranges : total_oranges = 12)
    (h_glasses : glasses_of_juice = 6) : 
    total_oranges / glasses_of_juice = oranges_per_glass :=
by 
    sorry

end NUMINAMATH_GPT_oranges_per_glass_l1338_133894


namespace NUMINAMATH_GPT_intersection_M_N_l1338_133871

open Set

noncomputable def M : Set ℕ := {x | x < 6}
noncomputable def N : Set ℕ := {x | x^2 - 11 * x + 18 < 0}

theorem intersection_M_N : M ∩ N = {3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1338_133871


namespace NUMINAMATH_GPT_number_exceeds_35_percent_by_245_l1338_133809

theorem number_exceeds_35_percent_by_245 : 
  ∃ (x : ℝ), (0.35 * x + 245 = x) ∧ x = 376.92 := 
by
  sorry

end NUMINAMATH_GPT_number_exceeds_35_percent_by_245_l1338_133809


namespace NUMINAMATH_GPT_harmonic_mean_pairs_count_l1338_133819

open Nat

theorem harmonic_mean_pairs_count :
  ∃! n : ℕ, (∀ x y : ℕ, x < y ∧ x > 0 ∧ y > 0 ∧ (2 * x * y) / (x + y) = 4^15 → n = 29) :=
sorry

end NUMINAMATH_GPT_harmonic_mean_pairs_count_l1338_133819


namespace NUMINAMATH_GPT_prime_cond_l1338_133872

theorem prime_cond (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1) : 
  (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) → (p = 2 ∧ q = 5 ∧ n = 2) :=
  sorry

end NUMINAMATH_GPT_prime_cond_l1338_133872


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1338_133878

-- Problem 1
theorem problem1 : (-3 / 8) + ((-5 / 8) * (-6)) = 27 / 8 :=
by sorry

-- Problem 2
theorem problem2 : 12 + (7 * (-3)) - (18 / (-3)) = -3 :=
by sorry

-- Problem 3
theorem problem3 : -((2:ℤ)^2) - (4 / 7) * (2:ℚ) - (-((3:ℤ)^2:ℤ) : ℤ) = -99 / 7 :=
by sorry

-- Problem 4
theorem problem4 : -(((-1) ^ 2020 : ℤ)) + ((6 : ℚ) / (-(2 : ℤ) ^ 3)) * (-1 / 3) = -3 / 4 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1338_133878


namespace NUMINAMATH_GPT_tea_customers_count_l1338_133892

theorem tea_customers_count :
  ∃ T : ℕ, 7 * 5 + T * 4 = 67 ∧ T = 8 :=
by
  sorry

end NUMINAMATH_GPT_tea_customers_count_l1338_133892


namespace NUMINAMATH_GPT_math_proof_problem_l1338_133804

noncomputable def problem_statement : Prop :=
  let a_bound := 14
  let b_bound := 7
  let c_bound := 14
  let num_square_divisors := (a_bound / 2 + 1) * (b_bound / 2 + 1) * (c_bound / 2 + 1)
  let num_cube_divisors := (a_bound / 3 + 1) * (b_bound / 3 + 1) * (c_bound / 3 + 1)
  let num_sixth_power_divisors := (a_bound / 6 + 1) * (b_bound / 6 + 1) * (c_bound / 6 + 1)
  
  num_square_divisors + num_cube_divisors - num_sixth_power_divisors = 313

theorem math_proof_problem : problem_statement := by sorry

end NUMINAMATH_GPT_math_proof_problem_l1338_133804


namespace NUMINAMATH_GPT_union_A_B_interval_l1338_133854

def setA (x : ℝ) : Prop := x ≥ -1
def setB (y : ℝ) : Prop := y ≥ 1

theorem union_A_B_interval :
  {x | setA x} ∪ {y | setB y} = {z : ℝ | z ≥ -1} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_interval_l1338_133854


namespace NUMINAMATH_GPT_point_on_y_axis_is_zero_l1338_133814

-- Given conditions
variables (m : ℝ) (y : ℝ)
-- \( P(m, 2) \) lies on the y-axis
def point_on_y_axis (m y : ℝ) : Prop := (m = 0)

-- Proof statement: Prove that if \( P(m, 2) \) lies on the y-axis, then \( m = 0 \)
theorem point_on_y_axis_is_zero (h : point_on_y_axis m 2) : m = 0 :=
by 
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_point_on_y_axis_is_zero_l1338_133814


namespace NUMINAMATH_GPT_swim_back_distance_l1338_133840

variables (swimming_speed_still_water : ℝ) (water_speed : ℝ) (time_back : ℝ) (distance_back : ℝ)

theorem swim_back_distance :
  swimming_speed_still_water = 12 → 
  water_speed = 10 → 
  time_back = 4 →
  distance_back = (swimming_speed_still_water - water_speed) * time_back →
  distance_back = 8 :=
by
  intros swimming_speed_still_water_eq water_speed_eq time_back_eq distance_back_eq
  have swim_speed : (swimming_speed_still_water - water_speed) = 2 := by sorry
  rw [swim_speed, time_back_eq] at distance_back_eq
  sorry

end NUMINAMATH_GPT_swim_back_distance_l1338_133840


namespace NUMINAMATH_GPT_only_one_P_Q_l1338_133802

def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - x + a = 0

theorem only_one_P_Q (a : ℝ) :
  (P a ∧ ¬ Q a) ∨ (Q a ∧ ¬ P a) ↔
  (a < 0) ∨ (1/4 < a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_only_one_P_Q_l1338_133802


namespace NUMINAMATH_GPT_sin_cos_ratio_value_sin_cos_expression_value_l1338_133893

variable (α : ℝ)

-- Given condition
def tan_alpha_eq_3 := Real.tan α = 3

-- Goal (1)
theorem sin_cos_ratio_value 
  (h : tan_alpha_eq_3 α) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 4 / 5 := 
  sorry

-- Goal (2)
theorem sin_cos_expression_value
  (h : tan_alpha_eq_3 α) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 15 := 
  sorry

end NUMINAMATH_GPT_sin_cos_ratio_value_sin_cos_expression_value_l1338_133893


namespace NUMINAMATH_GPT_f_at_neg2_l1338_133822

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - Real.log (x^2 - 3*x + 5) / Real.log 3 
else -2^(-x) + Real.log ((-x)^2 + 3*(-x) + 5) / Real.log 3 

theorem f_at_neg2 : f (-2) = -3 := by
  sorry

end NUMINAMATH_GPT_f_at_neg2_l1338_133822


namespace NUMINAMATH_GPT_domain_of_log_function_l1338_133861

theorem domain_of_log_function (x : ℝ) :
  (-1 < x ∧ x < 1) ↔ (1 - x) / (1 + x) > 0 :=
by sorry

end NUMINAMATH_GPT_domain_of_log_function_l1338_133861


namespace NUMINAMATH_GPT_general_form_of_equation_l1338_133891

theorem general_form_of_equation : 
  ∀ x : ℝ, (x - 1) * (x - 2) = 4 → x^2 - 3 * x - 2 = 0 := by
  sorry

end NUMINAMATH_GPT_general_form_of_equation_l1338_133891


namespace NUMINAMATH_GPT_picked_tomatoes_eq_53_l1338_133855

-- Definitions based on the conditions
def initial_tomatoes : ℕ := 177
def initial_potatoes : ℕ := 12
def items_left : ℕ := 136

-- Define what we need to prove
theorem picked_tomatoes_eq_53 : initial_tomatoes + initial_potatoes - items_left = 53 :=
by sorry

end NUMINAMATH_GPT_picked_tomatoes_eq_53_l1338_133855


namespace NUMINAMATH_GPT_squirrels_in_tree_l1338_133863

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) (h1 : nuts = 2) (h2 : squirrels = nuts + 2) : squirrels = 4 :=
by
    rw [h1] at h2
    exact h2

end NUMINAMATH_GPT_squirrels_in_tree_l1338_133863


namespace NUMINAMATH_GPT_binary_operation_l1338_133899

-- Definitions of the binary numbers.
def a : ℕ := 0b10110      -- 10110_2 in base 10
def b : ℕ := 0b10100      -- 10100_2 in base 10
def c : ℕ := 0b10         -- 10_2 in base 10
def result : ℕ := 0b11011100 -- 11011100_2 in base 10

-- The theorem to be proven
theorem binary_operation : (a * b) / c = result := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_binary_operation_l1338_133899


namespace NUMINAMATH_GPT_fill_question_mark_l1338_133845

def sudoku_grid : Type := 
  List (List (Option ℕ))

def initial_grid : sudoku_grid := 
  [ [some 3, none, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ]

def valid_sudoku (grid : sudoku_grid) : Prop :=
  -- Ensure the grid is a valid 4x4 Sudoku grid
  -- Adding necessary constraints for rows, columns and 2x2 subgrids.
  sorry

def solve_sudoku (grid : sudoku_grid) : sudoku_grid :=
  -- Function that solves the Sudoku (not implemented for this proof statement)
  sorry

theorem fill_question_mark : solve_sudoku initial_grid = 
  [ [some 3, some 2, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ] :=
  sorry

end NUMINAMATH_GPT_fill_question_mark_l1338_133845


namespace NUMINAMATH_GPT_find_f_of_power_function_l1338_133801

theorem find_f_of_power_function (a : ℝ) (alpha : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a ∧ a ≠ 1) 
  (h2 : ∀ x, f x = x^alpha) 
  (h3 : ∀ x, a^(x-2) + 3 = f (2)): 
  f 2 = 4 := 
  sorry

end NUMINAMATH_GPT_find_f_of_power_function_l1338_133801


namespace NUMINAMATH_GPT_marble_cut_in_third_week_l1338_133844

def percentage_cut_third_week := 
  let initial_weight : ℝ := 250 
  let final_weight : ℝ := 105
  let percent_cut_first_week : ℝ := 0.30
  let percent_cut_second_week : ℝ := 0.20
  let weight_after_first_week := initial_weight * (1 - percent_cut_first_week)
  let weight_after_second_week := weight_after_first_week * (1 - percent_cut_second_week)
  (weight_after_second_week - final_weight) / weight_after_second_week * 100 = 25

theorem marble_cut_in_third_week :
  percentage_cut_third_week = true :=
by
  sorry

end NUMINAMATH_GPT_marble_cut_in_third_week_l1338_133844


namespace NUMINAMATH_GPT_common_property_of_rhombus_and_rectangle_l1338_133818

structure Rhombus :=
  (bisect_perpendicular : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_not_equal : ∀ d₁ d₂ : ℝ, ¬(d₁ = d₂))

structure Rectangle :=
  (bisect_each_other : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_equal : ∀ d₁ d₂ : ℝ, d₁ = d₂)

theorem common_property_of_rhombus_and_rectangle (R : Rhombus) (S : Rectangle) :
  ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0) :=
by
  -- Assuming the properties of Rhombus R and Rectangle S
  sorry

end NUMINAMATH_GPT_common_property_of_rhombus_and_rectangle_l1338_133818


namespace NUMINAMATH_GPT_average_of_original_set_l1338_133839

-- Average of 8 numbers is some value A and the average of the new set where each number is 
-- multiplied by 8 is 168. We need to show that the original average A is 21.

theorem average_of_original_set (A : ℝ) (h1 : (64 * A) / 8 = 168) : A = 21 :=
by {
  -- This is the theorem statement, we add the proof next
  sorry -- proof placeholder
}

end NUMINAMATH_GPT_average_of_original_set_l1338_133839


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1338_133848

theorem solve_quadratic_eq : (x : ℝ) → (x^2 - 4 = 0) → (x = 2 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1338_133848


namespace NUMINAMATH_GPT_percentage_error_equals_l1338_133808

noncomputable def correct_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7/8 : ℚ) * 8
  let denom := (3/10 : ℚ) - (1/8 : ℚ)
  num / denom

noncomputable def incorrect_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7 / 8 : ℚ) * 8
  num * (3/5 : ℚ)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem percentage_error_equals :
  percentage_error correct_fraction_calc incorrect_fraction_calc = 89.47 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_equals_l1338_133808


namespace NUMINAMATH_GPT_binary_mul_correct_l1338_133820

def bin_to_nat (l : List ℕ) : ℕ :=
  l.foldl (λ n b => 2 * n + b) 0

def p : List ℕ := [1,0,1,1,0,1]
def q : List ℕ := [1,1,0,1]
def r : List ℕ := [1,0,0,0,1,0,0,0,1,1]

theorem binary_mul_correct :
  bin_to_nat p * bin_to_nat q = bin_to_nat r := by
  sorry

end NUMINAMATH_GPT_binary_mul_correct_l1338_133820


namespace NUMINAMATH_GPT_probability_heart_then_king_of_clubs_l1338_133811

theorem probability_heart_then_king_of_clubs : 
  let deck := 52
  let hearts := 13
  let remaining_cards := deck - 1
  let king_of_clubs := 1
  let first_card_heart_probability := (hearts : ℝ) / deck
  let second_card_king_of_clubs_probability := (king_of_clubs : ℝ) / remaining_cards
  first_card_heart_probability * second_card_king_of_clubs_probability = 1 / 204 :=
by
  sorry

end NUMINAMATH_GPT_probability_heart_then_king_of_clubs_l1338_133811


namespace NUMINAMATH_GPT_triangle_area_is_17_point_5_l1338_133877

-- Define the points A, B, and C as tuples of coordinates
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (7, 2)
def C : (ℝ × ℝ) := (4, 9)

-- Function to calculate the area of a triangle given its vertices
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

-- The theorem statement asserting the area of the triangle is 17.5 square units
theorem triangle_area_is_17_point_5 :
  area_of_triangle A B C = 17.5 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_triangle_area_is_17_point_5_l1338_133877


namespace NUMINAMATH_GPT_mod_remainder_l1338_133874

open Int

theorem mod_remainder (n : ℤ) : 
  (1125 * 1127 * n) % 12 = 3 ↔ n % 12 = 1 :=
by
  sorry

end NUMINAMATH_GPT_mod_remainder_l1338_133874


namespace NUMINAMATH_GPT_remaining_milk_correct_l1338_133865

def arranged_milk : ℝ := 21.52
def sold_milk : ℝ := 12.64
def remaining_milk (total : ℝ) (sold : ℝ) : ℝ := total - sold

theorem remaining_milk_correct :
  remaining_milk arranged_milk sold_milk = 8.88 :=
by
  sorry

end NUMINAMATH_GPT_remaining_milk_correct_l1338_133865


namespace NUMINAMATH_GPT_cookies_in_the_fridge_l1338_133873

-- Define the conditions
def total_baked : ℕ := 256
def tim_cookies : ℕ := 15
def mike_cookies : ℕ := 23
def anna_cookies : ℕ := 2 * tim_cookies

-- Define the proof problem
theorem cookies_in_the_fridge : (total_baked - (tim_cookies + mike_cookies + anna_cookies)) = 188 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_cookies_in_the_fridge_l1338_133873


namespace NUMINAMATH_GPT_valid_digit_cancel_fractions_l1338_133829

def digit_cancel_fraction (a b c d : ℕ) : Prop :=
  10 * a + b == 0 ∧ 10 * c + d == 0 ∧ 
  (b == d ∨ b == c ∨ a == d ∨ a == c) ∧
  (b ≠ a ∨ d ≠ c) ∧
  ((10 * a + b) ≠ (10 * c + d)) ∧
  ((10 * a + b) * d == (10 * c + d) * a)

theorem valid_digit_cancel_fractions :
  ∀ (a b c d : ℕ), 
  digit_cancel_fraction a b c d → 
  (10 * a + b == 26 ∧ 10 * c + d == 65) ∨
  (10 * a + b == 16 ∧ 10 * c + d == 64) ∨
  (10 * a + b == 19 ∧ 10 * c + d == 95) ∨
  (10 * a + b == 49 ∧ 10 * c + d == 98) :=
by {sorry}

end NUMINAMATH_GPT_valid_digit_cancel_fractions_l1338_133829


namespace NUMINAMATH_GPT_number_of_valid_pairs_is_343_l1338_133817

-- Define the given problem conditions
def given_number : Nat := 1003003001

-- Define the expression for LCM calculation
def LCM (x y : Nat) : Nat := (x * y) / (Nat.gcd x y)

-- Define the prime factorization of the given number
def is_prime_factorization_correct : Prop :=
  given_number = 7^3 * 11^3 * 13^3

-- Define x and y form as described
def is_valid_form (x y : Nat) : Prop :=
  ∃ (a b c d e f : ℕ), x = 7^a * 11^b * 13^c ∧ y = 7^d * 11^e * 13^f

-- Define the LCM condition for the ordered pairs
def meets_lcm_condition (x y : Nat) : Prop :=
  LCM x y = given_number

-- State the theorem to prove an equivalent problem
theorem number_of_valid_pairs_is_343 : is_prime_factorization_correct →
  (∃ (n : ℕ), n = 343 ∧ 
    (∀ (x y : ℕ), is_valid_form x y → meets_lcm_condition x y → x > 0 → y > 0 → True)
  ) :=
by
  intros h
  use 343
  sorry

end NUMINAMATH_GPT_number_of_valid_pairs_is_343_l1338_133817


namespace NUMINAMATH_GPT_fred_money_last_week_l1338_133881

-- Definitions for the conditions in the problem
variables {f j : ℕ} (current_fred : ℕ) (current_jason : ℕ) (last_week_jason : ℕ)
variable (earning : ℕ)

-- Conditions
axiom Fred_current_money : current_fred = 115
axiom Jason_current_money : current_jason = 44
axiom Jason_last_week_money : last_week_jason = 40
axiom Earning_amount : earning = 4

-- Theorem statement: prove Fred's money last week
theorem fred_money_last_week (current_fred last_week_jason current_jason earning : ℕ)
  (Fred_current_money : current_fred = 115)
  (Jason_current_money : current_jason = 44)
  (Jason_last_week_money : last_week_jason = 40)
  (Earning_amount : earning = 4)
  : current_fred - earning = 111 :=
sorry

end NUMINAMATH_GPT_fred_money_last_week_l1338_133881


namespace NUMINAMATH_GPT_rancher_cattle_count_l1338_133800

theorem rancher_cattle_count
  (truck_capacity : ℕ)
  (distance_to_higher_ground : ℕ)
  (truck_speed : ℕ)
  (total_transport_time : ℕ)
  (h1 : truck_capacity = 20)
  (h2 : distance_to_higher_ground = 60)
  (h3 : truck_speed = 60)
  (h4 : total_transport_time = 40):
  ∃ (number_of_cattle : ℕ), number_of_cattle = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_rancher_cattle_count_l1338_133800


namespace NUMINAMATH_GPT_find_positive_solution_l1338_133888

-- Defining the variables x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the conditions from the problem statement
def condition1 : Prop := x * y + 3 * x + 4 * y + 10 = 30
def condition2 : Prop := y * z + 4 * y + 2 * z + 8 = 6
def condition3 : Prop := x * z + 4 * x + 3 * z + 12 = 30

-- The theorem that states the positive solution for x is 3
theorem find_positive_solution (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 x z) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_positive_solution_l1338_133888


namespace NUMINAMATH_GPT_total_length_of_scale_l1338_133847

theorem total_length_of_scale (num_parts : ℕ) (length_per_part : ℕ) 
  (h1: num_parts = 4) (h2: length_per_part = 20) : 
  num_parts * length_per_part = 80 := by
  sorry

end NUMINAMATH_GPT_total_length_of_scale_l1338_133847


namespace NUMINAMATH_GPT_amount_of_p_l1338_133838

theorem amount_of_p (p q r : ℝ) (h1 : q = (1 / 6) * p) (h2 : r = (1 / 6) * p) 
  (h3 : p = (q + r) + 32) : p = 48 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_p_l1338_133838


namespace NUMINAMATH_GPT_max_halls_visited_l1338_133887

theorem max_halls_visited (side_len large_tri small_tri: ℕ) 
  (h1 : side_len = 100)
  (h2 : large_tri = 100)
  (h3 : small_tri = 10)
  (div : large_tri = (side_len / small_tri) ^ 2) :
  ∃ m : ℕ, m = 91 → m ≤ large_tri - 9 := 
sorry

end NUMINAMATH_GPT_max_halls_visited_l1338_133887


namespace NUMINAMATH_GPT_different_testing_methods_1_different_testing_methods_2_l1338_133852

-- Definitions used in Lean 4 statement should be derived from the conditions in a).
def total_products := 10
def defective_products := 4
def non_defective_products := total_products - defective_products
def choose (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement (1)
theorem different_testing_methods_1 :
  let first_defective := 5
  let last_defective := 10
  let non_defective_in_first_4 := choose 6 4
  let defective_in_middle_5 := choose 5 3
  let total_methods := non_defective_in_first_4 * defective_in_middle_5 * Nat.factorial 5 * Nat.factorial 4
  total_methods = 103680 := sorry

-- Statement (2)
theorem different_testing_methods_2 :
  let first_defective := 5
  let remaining_defective := 4
  let non_defective_in_first_4 := choose 6 4
  let total_methods := non_defective_in_first_4 * Nat.factorial 5
  total_methods = 576 := sorry

end NUMINAMATH_GPT_different_testing_methods_1_different_testing_methods_2_l1338_133852


namespace NUMINAMATH_GPT_cakes_and_bread_weight_l1338_133832

theorem cakes_and_bread_weight 
  (B : ℕ)
  (cake_weight : ℕ := B + 100)
  (h1 : 4 * cake_weight = 800)
  : 3 * cake_weight + 5 * B = 1100 := by
  sorry

end NUMINAMATH_GPT_cakes_and_bread_weight_l1338_133832


namespace NUMINAMATH_GPT_find_number_l1338_133835

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 126) : x = 5600 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_number_l1338_133835


namespace NUMINAMATH_GPT_range_of_x_l1338_133836

theorem range_of_x (x p : ℝ) (h₀ : 0 ≤ p ∧ p ≤ 4) :
  x^2 + p * x > 4 * x + p - 3 → (x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_GPT_range_of_x_l1338_133836


namespace NUMINAMATH_GPT_empty_pipe_time_l1338_133886

theorem empty_pipe_time (R1 R2 : ℚ) (t1 t2 t_total : ℕ) (h1 : t1 = 60) (h2 : t_total = 180) (H1 : R1 = 1 / t1) (H2 : R1 - R2 = 1 / t_total) :
  1 / R2 = 90 :=
by
  sorry

end NUMINAMATH_GPT_empty_pipe_time_l1338_133886


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l1338_133882

theorem problem_part_1 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) : 
  1 - p_A ^ 3 = 19 / 27 :=
by sorry

theorem problem_part_2 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) 
  (h1 : 3 * (p_A ^ 2) * (1 - p_A) = 4 / 9)
  (h2 : 3 * p_B * ((1 - p_B) ^ 2) = 9 / 64) : 
  (4 / 9) * (9 / 64) = 1 / 16 :=
by sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l1338_133882


namespace NUMINAMATH_GPT_proof_problem_l1338_133824

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1338_133824


namespace NUMINAMATH_GPT_paint_needed_for_snake_l1338_133857

open Nat

def total_paint (paint_per_segment segments additional_paint : Nat) : Nat :=
  paint_per_segment * segments + additional_paint

theorem paint_needed_for_snake :
  total_paint 240 336 20 = 80660 :=
by
  sorry

end NUMINAMATH_GPT_paint_needed_for_snake_l1338_133857


namespace NUMINAMATH_GPT_ratio_of_c_d_l1338_133896

theorem ratio_of_c_d 
  (x y c d : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c)
  (h2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_c_d_l1338_133896


namespace NUMINAMATH_GPT_verify_exact_countries_attended_l1338_133880

theorem verify_exact_countries_attended :
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  (attended_countries = 68) :=
by
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  have : attended_countries = 68 := rfl
  exact this

end NUMINAMATH_GPT_verify_exact_countries_attended_l1338_133880


namespace NUMINAMATH_GPT_area_of_room_in_square_inches_l1338_133859

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end NUMINAMATH_GPT_area_of_room_in_square_inches_l1338_133859


namespace NUMINAMATH_GPT_factorization_of_difference_of_squares_l1338_133856

theorem factorization_of_difference_of_squares (m : ℝ) : 
  m^2 - 16 = (m + 4) * (m - 4) := 
by 
  sorry

end NUMINAMATH_GPT_factorization_of_difference_of_squares_l1338_133856


namespace NUMINAMATH_GPT_smallest_yellow_marbles_l1338_133821

-- Definitions for given conditions
def total_marbles (n : ℕ): Prop := n > 0
def blue_marbles (n : ℕ) : ℕ := n / 4
def red_marbles (n : ℕ) : ℕ := n / 6
def green_marbles : ℕ := 7
def yellow_marbles (n : ℕ) : ℕ := n - (blue_marbles n + red_marbles n + green_marbles)

-- Lean statement that verifies the smallest number of yellow marbles is 0
theorem smallest_yellow_marbles (n : ℕ) (h : total_marbles n) : yellow_marbles n = 0 :=
  sorry

end NUMINAMATH_GPT_smallest_yellow_marbles_l1338_133821


namespace NUMINAMATH_GPT_coterminal_angle_l1338_133867

theorem coterminal_angle (α : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 283 ↔ ∃ k : ℤ, α = k * 360 - 437 :=
sorry

end NUMINAMATH_GPT_coterminal_angle_l1338_133867


namespace NUMINAMATH_GPT_abs_f_sub_lt_abs_l1338_133884

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

theorem abs_f_sub_lt_abs (a b : ℝ) (h : a ≠ b) : 
  |f a - f b| < |a - b| := 
by
  sorry

end NUMINAMATH_GPT_abs_f_sub_lt_abs_l1338_133884


namespace NUMINAMATH_GPT_temperature_difference_l1338_133834

-- Define the temperatures given in the problem.
def T_noon : ℝ := 10
def T_midnight : ℝ := -150

-- State the theorem to prove the temperature difference.
theorem temperature_difference :
  T_noon - T_midnight = 160 :=
by
  -- We skip the proof and add sorry.
  sorry

end NUMINAMATH_GPT_temperature_difference_l1338_133834


namespace NUMINAMATH_GPT_four_digit_numbers_with_3_or_7_l1338_133870

theorem four_digit_numbers_with_3_or_7 : 
  let total_four_digit_numbers := 9000
  let numbers_without_3_or_7 := 3584
  total_four_digit_numbers - numbers_without_3_or_7 = 5416 :=
by
  trivial

end NUMINAMATH_GPT_four_digit_numbers_with_3_or_7_l1338_133870


namespace NUMINAMATH_GPT_digits_of_2_pow_100_last_three_digits_of_2_pow_100_l1338_133841

-- Prove that 2^100 has 31 digits.
theorem digits_of_2_pow_100 : (10^30 ≤ 2^100) ∧ (2^100 < 10^31) :=
by
  sorry

-- Prove that the last three digits of 2^100 are 376.
theorem last_three_digits_of_2_pow_100 : 2^100 % 1000 = 376 :=
by
  sorry

end NUMINAMATH_GPT_digits_of_2_pow_100_last_three_digits_of_2_pow_100_l1338_133841


namespace NUMINAMATH_GPT_number_of_ways_to_choose_committee_l1338_133879

-- Definitions of the conditions
def eligible_members : ℕ := 30
def new_members : ℕ := 3
def committee_size : ℕ := 5
def eligible_pool : ℕ := eligible_members - new_members

-- Problem statement to prove
theorem number_of_ways_to_choose_committee : (Nat.choose eligible_pool committee_size) = 80730 := by
  -- This space is reserved for the proof which is not required per instructions.
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_committee_l1338_133879


namespace NUMINAMATH_GPT_whale_consumption_third_hour_l1338_133810

theorem whale_consumption_third_hour (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 450) → ((x + 6) = 90) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_whale_consumption_third_hour_l1338_133810


namespace NUMINAMATH_GPT_non_degenerate_ellipse_l1338_133815

theorem non_degenerate_ellipse (x y k : ℝ) : (∃ k, (2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k) → k > -135 / 4) := sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_l1338_133815


namespace NUMINAMATH_GPT_value_of_y_l1338_133866

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 :=
sorry

end NUMINAMATH_GPT_value_of_y_l1338_133866


namespace NUMINAMATH_GPT_area_of_rectangle_l1338_133816

theorem area_of_rectangle (a b : ℝ) (area : ℝ) 
(h1 : a = 5.9) 
(h2 : b = 3) 
(h3 : area = a * b) : 
area = 17.7 := 
by 
  -- proof goes here
  sorry

-- Definitions and conditions alignment:
-- a represents one side of the rectangle.
-- b represents the other side of the rectangle.
-- area represents the area of the rectangle.
-- h1: a = 5.9 corresponds to the first condition.
-- h2: b = 3 corresponds to the second condition.
-- h3: area = a * b connects the conditions to the formula to find the area.
-- The goal is to show that area = 17.7, which matches the correct answer.

end NUMINAMATH_GPT_area_of_rectangle_l1338_133816


namespace NUMINAMATH_GPT_range_of_a_l1338_133853

theorem range_of_a (a : ℝ) :
  (∀ x, (3 ≤ x → 2*a*x + 4 ≤ 2*a*(x+1) + 4) ∧ (2 < x ∧ x < 3 → (a + (2*a + 2)/(x-2) ≤ a + (2*a + 2)/(x-1))) ) →
  -1 < a ∧ a ≤ -2/3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l1338_133853


namespace NUMINAMATH_GPT_percentage_water_in_puree_l1338_133807

/-- Given that tomato juice is 90% water and Heinz obtains 2.5 litres of tomato puree from 20 litres of tomato juice,
proves that the percentage of water in the tomato puree is 20%. -/
theorem percentage_water_in_puree (tj_volume : ℝ) (tj_water_content : ℝ) (tp_volume : ℝ) (tj_to_tp_ratio : ℝ) 
  (h1 : tj_water_content = 0.90) 
  (h2 : tj_volume = 20) 
  (h3 : tp_volume = 2.5) 
  (h4 : tj_to_tp_ratio = tj_volume / tp_volume) : 
  ((tp_volume - (1 - tj_water_content) * (tj_volume * (tp_volume / tj_volume))) / tp_volume) * 100 = 20 := 
sorry

end NUMINAMATH_GPT_percentage_water_in_puree_l1338_133807


namespace NUMINAMATH_GPT_range_of_a_l1338_133849

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 0 then -x + 3 * a else x^2 - a * x + 1

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≥ f a x2) ↔ (0 <= a ∧ a <= 1/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1338_133849


namespace NUMINAMATH_GPT_sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l1338_133858

def avg_daily_production := 400
def weekly_planned_production := 2800
def daily_deviations := [15, -5, 21, 16, -7, 0, -8]
def total_weekly_deviation := 80

-- Calculation for sets produced on Saturday
def sat_production_exceeds_plan := total_weekly_deviation - (daily_deviations.take (daily_deviations.length - 1)).sum
def sat_production := avg_daily_production + sat_production_exceeds_plan

-- Calculation for the difference between the max and min production days
def max_deviation := max sat_production_exceeds_plan (daily_deviations.maximum.getD 0)
def min_deviation := min sat_production_exceeds_plan (daily_deviations.minimum.getD 0)
def highest_lowest_diff := max_deviation - min_deviation

-- Calculation for the weekly wage for each worker
def workers := 20
def daily_wage := 200
def basic_weekly_wage := daily_wage * 7
def additional_wage := (15 + 21 + 16 + sat_production_exceeds_plan) * 10 - (5 + 7 + 8) * 15
def total_bonus := additional_wage / workers
def total_weekly_wage := basic_weekly_wage + total_bonus

theorem sat_production_correct : sat_production = 448 := by
  sorry

theorem highest_lowest_diff_correct : highest_lowest_diff = 56 := by
  sorry

theorem total_weekly_wage_correct : total_weekly_wage = 1435 := by
  sorry

end NUMINAMATH_GPT_sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l1338_133858


namespace NUMINAMATH_GPT_ratio_of_diagonals_l1338_133823

theorem ratio_of_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (4 * b) / (4 * a) = 11) : (b * Real.sqrt 2) / (a * Real.sqrt 2) = 11 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_diagonals_l1338_133823


namespace NUMINAMATH_GPT_storage_temperature_overlap_l1338_133812

theorem storage_temperature_overlap (T_A_min T_A_max T_B_min T_B_max : ℝ) 
  (hA : T_A_min = 0)
  (hA' : T_A_max = 5)
  (hB : T_B_min = 2)
  (hB' : T_B_max = 7) : 
  (max T_A_min T_B_min, min T_A_max T_B_max) = (2, 5) := by 
{
  sorry -- The proof is omitted as per instructions.
}

end NUMINAMATH_GPT_storage_temperature_overlap_l1338_133812


namespace NUMINAMATH_GPT_amy_music_files_l1338_133862

-- Define the number of total files on the flash drive
def files_on_flash_drive := 48.0

-- Define the number of video files on the flash drive
def video_files := 21.0

-- Define the number of picture files on the flash drive
def picture_files := 23.0

-- Define the number of music files, derived from the conditions
def music_files := files_on_flash_drive - (video_files + picture_files)

-- The theorem we need to prove
theorem amy_music_files : music_files = 4.0 := by
  sorry

end NUMINAMATH_GPT_amy_music_files_l1338_133862


namespace NUMINAMATH_GPT_smallest_seating_l1338_133826

theorem smallest_seating (N : ℕ) (h: ∀ (chairs : ℕ) (occupants : ℕ), 
  chairs = 100 ∧ occupants = 25 → 
  ∃ (adjacent_occupied: ℕ), adjacent_occupied > 0 ∧ adjacent_occupied < chairs ∧
  adjacent_occupied ≠ occupants) : 
  N = 25 :=
sorry

end NUMINAMATH_GPT_smallest_seating_l1338_133826


namespace NUMINAMATH_GPT_range_of_a_l1338_133805

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1338_133805


namespace NUMINAMATH_GPT_find_m_l1338_133831

theorem find_m (m : ℝ) : (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1338_133831


namespace NUMINAMATH_GPT_problem_sequence_inequality_l1338_133864

def a (n : ℕ) : ℚ := 15 + (n - 1 : ℚ) * (-(2 / 3))

theorem problem_sequence_inequality :
  ∃ k : ℕ, (a k) * (a (k + 1)) < 0 ∧ k = 23 :=
by {
  use 23,
  sorry
}

end NUMINAMATH_GPT_problem_sequence_inequality_l1338_133864


namespace NUMINAMATH_GPT_original_team_players_l1338_133860

theorem original_team_players (n : ℕ) (W : ℝ)
    (h1 : W = n * 76)
    (h2 : (W + 110 + 60) / (n + 2) = 78) : n = 7 :=
  sorry

end NUMINAMATH_GPT_original_team_players_l1338_133860


namespace NUMINAMATH_GPT_solve_inequality_l1338_133828

theorem solve_inequality (x : ℝ) :
  (0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4) ↔
  (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l1338_133828


namespace NUMINAMATH_GPT_units_digit_of_product_l1338_133846

-- Definitions for units digit patterns for powers of 5 and 7
def units_digit (n : ℕ) : ℕ := n % 10

def power5_units_digit := 5
def power7_units_cycle := [7, 9, 3, 1]

-- Statement of the problem
theorem units_digit_of_product :
  units_digit ((5 ^ 3) * (7 ^ 52)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l1338_133846


namespace NUMINAMATH_GPT_molecular_weight_l1338_133889

theorem molecular_weight :
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  (2 * H_weight + 1 * Br_weight + 3 * O_weight + 1 * C_weight + 1 * N_weight + 2 * S_weight) = 220.065 :=
by
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  sorry

end NUMINAMATH_GPT_molecular_weight_l1338_133889


namespace NUMINAMATH_GPT_num_green_balls_l1338_133842

theorem num_green_balls (G : ℕ) (h : (3 * 2 : ℚ) / ((5 + G) * (4 + G)) = 1/12) : G = 4 :=
by
  sorry

end NUMINAMATH_GPT_num_green_balls_l1338_133842


namespace NUMINAMATH_GPT_book_has_125_pages_l1338_133827

-- Define the number of pages in each chapter
def chapter1_pages : ℕ := 66
def chapter2_pages : ℕ := 35
def chapter3_pages : ℕ := 24

-- Define the total number of pages in the book
def total_pages : ℕ := chapter1_pages + chapter2_pages + chapter3_pages

-- State the theorem to prove that the total number of pages is 125
theorem book_has_125_pages : total_pages = 125 := 
by 
  -- The proof is omitted for the purpose of this task
  sorry

end NUMINAMATH_GPT_book_has_125_pages_l1338_133827


namespace NUMINAMATH_GPT_b_2023_equals_one_fifth_l1338_133850

theorem b_2023_equals_one_fifth (b : ℕ → ℚ) (h1 : b 1 = 4) (h2 : b 2 = 5)
    (h_rec : ∀ (n : ℕ), n ≥ 3 → b n = b (n - 1) / b (n - 2)) :
    b 2023 = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_b_2023_equals_one_fifth_l1338_133850


namespace NUMINAMATH_GPT_lisa_needs_change_probability_l1338_133851

theorem lisa_needs_change_probability :
  let quarters := 16
  let toy_prices := List.range' 2 10 |> List.map (fun n => n * 25) -- List of toy costs: (50,75,...,300)
  let favorite_toy_price := 275
  let factorial := Nat.factorial
  let favorable := (factorial 9) + 9 * (factorial 8)
  let total_permutations := factorial 10
  let p_no_change := (favorable.toFloat / total_permutations.toFloat) -- Convert to Float for probability calculations
  let p_change_needed := Float.round ((1.0 - p_no_change) * 100.0) / 100.0
  p_change_needed = 4.0 / 5.0 := sorry

end NUMINAMATH_GPT_lisa_needs_change_probability_l1338_133851


namespace NUMINAMATH_GPT_analytic_expression_of_f_max_min_of_f_on_interval_l1338_133803

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytic_expression_of_f :
  ∀ A ω φ : ℝ, (∀ x, f x = A * Real.sin (ω * x + φ)) →
  A = 2 ∧ ω = 2 ∧ φ = Real.pi / 6 :=
by
  sorry -- Placeholder for the actual proof

theorem max_min_of_f_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≤ Real.sqrt 3) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≥ 1) :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_analytic_expression_of_f_max_min_of_f_on_interval_l1338_133803


namespace NUMINAMATH_GPT_calculate_x_l1338_133883

theorem calculate_x :
  let a := 3
  let b := 5
  let c := 2
  let d := 4
  let term1 := (a ^ 2) * b * 0.47 * 1442
  let term2 := c * d * 0.36 * 1412
  (term1 - term2) + 63 = 26544.74 := by
  sorry

end NUMINAMATH_GPT_calculate_x_l1338_133883


namespace NUMINAMATH_GPT_solution_set_inequality_l1338_133885

theorem solution_set_inequality (x : ℝ) : 3 * x - 2 > x → x > 1 := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1338_133885


namespace NUMINAMATH_GPT_cube_difference_positive_l1338_133898

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end NUMINAMATH_GPT_cube_difference_positive_l1338_133898


namespace NUMINAMATH_GPT_smallest_number_among_given_l1338_133897

theorem smallest_number_among_given :
  ∀ (a b c d : ℚ), a = -2 → b = -5/2 → c = 0 → d = 1/5 →
  (min (min (min a b) c) d) = b :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end NUMINAMATH_GPT_smallest_number_among_given_l1338_133897


namespace NUMINAMATH_GPT_surface_area_of_each_smaller_cube_l1338_133876

theorem surface_area_of_each_smaller_cube
  (L : ℝ) (l : ℝ)
  (h1 : 6 * L^2 = 600)
  (h2 : 125 * l^3 = L^3) :
  6 * l^2 = 24 := by
  sorry

end NUMINAMATH_GPT_surface_area_of_each_smaller_cube_l1338_133876


namespace NUMINAMATH_GPT_ladder_rung_length_l1338_133833

noncomputable def ladder_problem : Prop :=
  let total_height_ft := 50
  let spacing_in := 6
  let wood_ft := 150
  let feet_to_inches(ft : ℕ) : ℕ := ft * 12
  let total_height_in := feet_to_inches total_height_ft
  let wood_in := feet_to_inches wood_ft
  let number_of_rungs := total_height_in / spacing_in
  let length_of_each_rung := wood_in / number_of_rungs
  length_of_each_rung = 18

theorem ladder_rung_length : ladder_problem := sorry

end NUMINAMATH_GPT_ladder_rung_length_l1338_133833


namespace NUMINAMATH_GPT_problem_1_problem_2_l1338_133895

-- Define the given function
def f (x : ℝ) := |x - 1|

-- Problem 1: Prove if f(x) + f(1 - x) ≥ a always holds, then a ≤ 1
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f x + f (1 - x) ≥ a) → a ≤ 1 :=
  sorry

-- Problem 2: Prove if a + 2b = 8, then f(a)^2 + f(b)^2 ≥ 5
theorem problem_2 (a b : ℝ) : 
  (a + 2 * b = 8) → (f a)^2 + (f b)^2 ≥ 5 :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1338_133895
