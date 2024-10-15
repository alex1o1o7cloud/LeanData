import Mathlib

namespace NUMINAMATH_GPT_total_pumpkin_weight_l418_41881

-- Conditions
def weight_first_pumpkin : ℝ := 4
def weight_second_pumpkin : ℝ := 8.7

-- Statement
theorem total_pumpkin_weight :
  weight_first_pumpkin + weight_second_pumpkin = 12.7 :=
by
  -- Proof can be done manually or via some automation here
  sorry

end NUMINAMATH_GPT_total_pumpkin_weight_l418_41881


namespace NUMINAMATH_GPT_shirley_ends_with_106_l418_41830

-- Define the initial number of eggs and the number bought
def initialEggs : Nat := 98
def additionalEggs : Nat := 8

-- Define the final count as the sum of initial eggs and additional eggs
def finalEggCount : Nat := initialEggs + additionalEggs

-- State the theorem with the correct answer
theorem shirley_ends_with_106 :
  finalEggCount = 106 :=
by
  sorry

end NUMINAMATH_GPT_shirley_ends_with_106_l418_41830


namespace NUMINAMATH_GPT_arithmetic_seq_sum_x_y_l418_41815

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_x_y_l418_41815


namespace NUMINAMATH_GPT_converse_proposition_l418_41879

-- Define the propositions p and q
variables (p q : Prop)

-- State the problem as a theorem
theorem converse_proposition (p q : Prop) : (q → p) ↔ ¬p → ¬q ∧ ¬q → ¬p ∧ (p → q) := 
by 
  sorry

end NUMINAMATH_GPT_converse_proposition_l418_41879


namespace NUMINAMATH_GPT_find_value_of_expression_l418_41802

theorem find_value_of_expression (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) : 2 * a + 2 * b - 3 * (a * b) = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l418_41802


namespace NUMINAMATH_GPT_trig_identity_proofs_l418_41838

theorem trig_identity_proofs (α : ℝ) 
  (h : Real.sin α + Real.cos α = 1 / 5) :
  (Real.sin α - Real.cos α = 7 / 5 ∨ Real.sin α - Real.cos α = -7 / 5) ∧
  (Real.sin α ^ 3 + Real.cos α ^ 3 = 37 / 125) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proofs_l418_41838


namespace NUMINAMATH_GPT_value_of_b_l418_41855

def g (x : ℝ) : ℝ := 5 * x - 6

theorem value_of_b (b : ℝ) : g b = 0 ↔ b = 6 / 5 :=
by sorry

end NUMINAMATH_GPT_value_of_b_l418_41855


namespace NUMINAMATH_GPT_legoland_kangaroos_l418_41839

theorem legoland_kangaroos :
  ∃ (K R : ℕ), R = 5 * K ∧ K + R = 216 ∧ R = 180 := by
  sorry

end NUMINAMATH_GPT_legoland_kangaroos_l418_41839


namespace NUMINAMATH_GPT_loss_percentage_l418_41837

theorem loss_percentage (C S : ℕ) (H1 : C = 750) (H2 : S = 600) : (C - S) * 100 / C = 20 := by
  sorry

end NUMINAMATH_GPT_loss_percentage_l418_41837


namespace NUMINAMATH_GPT_positive_integer_solutions_l418_41851

theorem positive_integer_solutions :
  ∀ (a b c : ℕ), (8 * a - 5 * b)^2 + (3 * b - 2 * c)^2 + (3 * c - 7 * a)^2 = 2 → 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 12 ∧ b = 19 ∧ c = 28) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l418_41851


namespace NUMINAMATH_GPT_trapezium_second_side_length_l418_41829

theorem trapezium_second_side_length
  (side1 : ℝ)
  (height : ℝ)
  (area : ℝ) 
  (h1 : side1 = 20) 
  (h2 : height = 13) 
  (h3 : area = 247) : 
  ∃ side2 : ℝ, 0 ≤ side2 ∧ ∀ side2, area = 1 / 2 * (side1 + side2) * height → side2 = 18 :=
by
  use 18
  sorry

end NUMINAMATH_GPT_trapezium_second_side_length_l418_41829


namespace NUMINAMATH_GPT_trajectory_of_center_line_passes_fixed_point_l418_41846

-- Define the conditions
def pointA : ℝ × ℝ := (4, 0)
def chord_length : ℝ := 8
def pointB : ℝ × ℝ := (-3, 0)
def not_perpendicular_to_x_axis (t : ℝ) : Prop := t ≠ 0
def trajectory_eq (x y : ℝ) : Prop := y^2 = 8 * x
def line_eq (t m y x : ℝ) : Prop := x = t * y + m
def x_axis_angle_bisector (y1 x1 y2 x2 : ℝ) : Prop := (y1 / (x1 + 3)) + (y2 / (x2 + 3)) = 0

-- Prove the trajectory of the center of the moving circle is \( y^2 = 8x \)
theorem trajectory_of_center (x y : ℝ) 
  (H1: (x-4)^2 + y^2 = 4^2 + x^2) 
  (H2: trajectory_eq x y) : 
  trajectory_eq x y := sorry

-- Prove the line passes through the fixed point (3, 0)
theorem line_passes_fixed_point (t m y1 x1 y2 x2 : ℝ) 
  (Ht: not_perpendicular_to_x_axis t)
  (Hsys: ∀ y x, line_eq t m y x → trajectory_eq x y)
  (Hangle: x_axis_angle_bisector y1 x1 y2 x2) : 
  (m = 3) ∧ ∃ y, line_eq t 3 y 3 := sorry

end NUMINAMATH_GPT_trajectory_of_center_line_passes_fixed_point_l418_41846


namespace NUMINAMATH_GPT_zeros_of_f_l418_41886

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 2

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_zeros_of_f_l418_41886


namespace NUMINAMATH_GPT_fuel_A_volume_l418_41852

-- Let V_A and V_B be defined as the volumes of fuel A and B respectively.
def V_A : ℝ := sorry
def V_B : ℝ := sorry

-- Given conditions:
axiom h1 : V_A + V_B = 214
axiom h2 : 0.12 * V_A + 0.16 * V_B = 30

-- Prove that the volume of fuel A added, V_A, is 106 gallons.
theorem fuel_A_volume : V_A = 106 := 
by
  sorry

end NUMINAMATH_GPT_fuel_A_volume_l418_41852


namespace NUMINAMATH_GPT_negation_equivalence_l418_41809

theorem negation_equivalence {Triangle : Type} (has_circumcircle : Triangle → Prop) :
  ¬ (∃ (t : Triangle), ¬ has_circumcircle t) ↔ (∀ (t : Triangle), has_circumcircle t) :=
by
  sorry

end NUMINAMATH_GPT_negation_equivalence_l418_41809


namespace NUMINAMATH_GPT_solution_l418_41819

theorem solution 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := 
sorry 

end NUMINAMATH_GPT_solution_l418_41819


namespace NUMINAMATH_GPT_problem_1_problem_2_l418_41872

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem problem_1 : {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
  sorry

theorem problem_2 (m : ℝ) : (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l418_41872


namespace NUMINAMATH_GPT_p_necessary_condition_q_l418_41812

variable (a b : ℝ) (p : ab = 0) (q : a^2 + b^2 ≠ 0)

theorem p_necessary_condition_q : (∀ a b : ℝ, (ab = 0) → (a^2 + b^2 ≠ 0)) ∧ (∃ a b : ℝ, (a^2 + b^2 ≠ 0) ∧ ¬ (ab = 0)) := sorry

end NUMINAMATH_GPT_p_necessary_condition_q_l418_41812


namespace NUMINAMATH_GPT_min_moves_to_balance_stacks_l418_41856

theorem min_moves_to_balance_stacks :
  let stack1 := 9
  let stack2 := 7
  let stack3 := 5
  let stack4 := 10
  let target := 8
  let total_coins := stack1 + stack2 + stack3 + stack4
  total_coins = 31 →
  ∃ moves, moves = 11 ∧
    (stack1 + 3 * moves = target) ∧
    (stack2 + 3 * moves = target) ∧
    (stack3 + 3 * moves = target) ∧
    (stack4 + 3 * moves = target) :=
sorry

end NUMINAMATH_GPT_min_moves_to_balance_stacks_l418_41856


namespace NUMINAMATH_GPT_part_a_part_b_l418_41845

def N := 10^40

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_perfect_square (a : ℕ) : Prop := ∃ m : ℕ, m * m = a

def is_perfect_cube (a : ℕ) : Prop := ∃ m : ℕ, m * m * m = a

def is_perfect_power (a : ℕ) : Prop := ∃ (m n : ℕ), n > 1 ∧ a = m^n

def num_divisors_not_square_or_cube (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that are neither perfect squares nor perfect cubes

def num_divisors_not_in_form_m_n (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that cannot be represented in the form m^n where n > 1

theorem part_a : num_divisors_not_square_or_cube N = 1093 := by
  sorry

theorem part_b : num_divisors_not_in_form_m_n N = 981 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l418_41845


namespace NUMINAMATH_GPT_students_taking_either_but_not_both_l418_41807

-- Definitions to encapsulate the conditions
def students_taking_both : ℕ := 15
def students_taking_mathematics : ℕ := 30
def students_taking_history_only : ℕ := 12

-- The goal is to prove the number of students taking mathematics or history but not both
theorem students_taking_either_but_not_both
  (hb : students_taking_both = 15)
  (hm : students_taking_mathematics = 30)
  (hh : students_taking_history_only = 12) : 
  students_taking_mathematics - students_taking_both + students_taking_history_only = 27 :=
by
  sorry

end NUMINAMATH_GPT_students_taking_either_but_not_both_l418_41807


namespace NUMINAMATH_GPT_sum_three_consecutive_divisible_by_three_l418_41831

theorem sum_three_consecutive_divisible_by_three (n : ℤ) : 3 ∣ ((n - 1) + n + (n + 1)) :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_sum_three_consecutive_divisible_by_three_l418_41831


namespace NUMINAMATH_GPT_probability_of_5_distinct_dice_rolls_is_5_over_54_l418_41836

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_5_distinct_dice_rolls_is_5_over_54_l418_41836


namespace NUMINAMATH_GPT_first_player_wins_l418_41892

def winning_strategy (m n : ℕ) : Prop :=
  if m = 1 ∧ n = 1 then false else true

theorem first_player_wins (m n : ℕ) :
  winning_strategy m n :=
by
  sorry

end NUMINAMATH_GPT_first_player_wins_l418_41892


namespace NUMINAMATH_GPT_decreased_area_of_equilateral_triangle_l418_41897

theorem decreased_area_of_equilateral_triangle 
    (A : ℝ) (hA : A = 100 * Real.sqrt 3) 
    (decrease : ℝ) (hdecrease : decrease = 6) :
    let s := Real.sqrt (4 * A / Real.sqrt 3)
    let s' := s - decrease
    let A' := (s' ^ 2 * Real.sqrt 3) / 4
    A - A' = 51 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_decreased_area_of_equilateral_triangle_l418_41897


namespace NUMINAMATH_GPT_john_total_distance_l418_41859

theorem john_total_distance : 
  let daily_distance := 1700
  let days_run := 6
  daily_distance * days_run = 10200 :=
by
  sorry

end NUMINAMATH_GPT_john_total_distance_l418_41859


namespace NUMINAMATH_GPT_weekly_earnings_l418_41865

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def phone_repairs : ℕ := 5
def laptop_repairs : ℕ := 2
def computer_repairs : ℕ := 2

def total_earnings : ℕ := 
  phone_repairs * phone_repair_cost + 
  laptop_repairs * laptop_repair_cost + 
  computer_repairs * computer_repair_cost

theorem weekly_earnings : total_earnings = 121 := by
  sorry

end NUMINAMATH_GPT_weekly_earnings_l418_41865


namespace NUMINAMATH_GPT_sector_perimeter_l418_41885

noncomputable def radius : ℝ := 2
noncomputable def central_angle_deg : ℝ := 120
noncomputable def expected_perimeter : ℝ := (4 / 3) * Real.pi + 4

theorem sector_perimeter (r : ℝ) (θ : ℝ) (h_r : r = radius) (h_θ : θ = central_angle_deg) :
    let arc_length := θ / 360 * 2 * Real.pi * r
    let perimeter := arc_length + 2 * r
    perimeter = expected_perimeter :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_sector_perimeter_l418_41885


namespace NUMINAMATH_GPT_one_serving_weight_l418_41811

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end NUMINAMATH_GPT_one_serving_weight_l418_41811


namespace NUMINAMATH_GPT_person_a_catch_up_person_b_5_times_l418_41890

theorem person_a_catch_up_person_b_5_times :
  ∀ (num_flags laps_a laps_b : ℕ),
  num_flags = 2015 →
  laps_a = 23 →
  laps_b = 13 →
  (∃ t : ℕ, ∃ n : ℕ, 10 * t = num_flags * n ∧
             23 * t / 10 = k * num_flags ∧
             n % 2 = 0) →
  n = 10 →
  10 / (2 * 1) = 5 :=
by sorry

end NUMINAMATH_GPT_person_a_catch_up_person_b_5_times_l418_41890


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l418_41887

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) (h : a > b + 1) : (a > b) ∧ ¬ (∀ (a b : ℝ), a > b → a > b + 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l418_41887


namespace NUMINAMATH_GPT_operation_result_l418_41806

def star (a b c : ℝ) : ℝ := (a + b + c) ^ 2

theorem operation_result (x : ℝ) : star (x - 1) (1 - x) 1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_operation_result_l418_41806


namespace NUMINAMATH_GPT_minimum_games_for_80_percent_l418_41843

theorem minimum_games_for_80_percent :
  ∃ N : ℕ, ( ∀ N' : ℕ, (1 + N') / (5 + N') * 100 < 80 → N < N') ∧ (1 + N) / (5 + N) * 100 ≥ 80 :=
sorry

end NUMINAMATH_GPT_minimum_games_for_80_percent_l418_41843


namespace NUMINAMATH_GPT_polynomial_roots_geometric_progression_q_l418_41854

theorem polynomial_roots_geometric_progression_q :
    ∃ (a r : ℝ), (a ≠ 0) ∧ (r ≠ 0) ∧
    (a + a * r + a * r ^ 2 + a * r ^ 3 = 0) ∧
    (a ^ 4 * r ^ 6 = 16) ∧
    (a ^ 2 + (a * r) ^ 2 + (a * r ^ 2) ^ 2 + (a * r ^ 3) ^ 2 = 16) :=
by
    sorry

end NUMINAMATH_GPT_polynomial_roots_geometric_progression_q_l418_41854


namespace NUMINAMATH_GPT_symmetric_poly_roots_identity_l418_41884

variable (a b c : ℝ)

theorem symmetric_poly_roots_identity (h1 : a + b + c = 6) (h2 : ab + bc + ca = 5) (h3 : abc = 1) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) = 38 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_poly_roots_identity_l418_41884


namespace NUMINAMATH_GPT_cube_volume_l418_41820

theorem cube_volume (S : ℝ) (hS : S = 294) : ∃ V : ℝ, V = 343 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l418_41820


namespace NUMINAMATH_GPT_sum_possible_m_continuous_l418_41858

noncomputable def g (x m : ℝ) : ℝ :=
if x < m then x^2 + 4 * x + 3 else 3 * x + 9

theorem sum_possible_m_continuous :
  let m₁ := -3
  let m₂ := 2
  m₁ + m₂ = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_m_continuous_l418_41858


namespace NUMINAMATH_GPT_solve_for_x_l418_41848

theorem solve_for_x : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by
  use -5
  sorry

end NUMINAMATH_GPT_solve_for_x_l418_41848


namespace NUMINAMATH_GPT_find_a_l418_41880

def f : ℝ → ℝ := sorry

theorem find_a (x a : ℝ) 
  (h1 : ∀ x, f ((1/2)*x - 1) = 2*x - 5)
  (h2 : f a = 6) : 
  a = 7/4 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l418_41880


namespace NUMINAMATH_GPT_sum_of_primes_146_sum_of_primes_99_l418_41871

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 146
theorem sum_of_primes_146 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 146 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 99
theorem sum_of_primes_99 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 99 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

end NUMINAMATH_GPT_sum_of_primes_146_sum_of_primes_99_l418_41871


namespace NUMINAMATH_GPT_value_of_h_h_2_is_353_l418_41873

def h (x : ℕ) : ℕ := 3 * x^2 - x + 1

theorem value_of_h_h_2_is_353 : h (h 2) = 353 := 
by
  sorry

end NUMINAMATH_GPT_value_of_h_h_2_is_353_l418_41873


namespace NUMINAMATH_GPT_ceil_sqrt_225_eq_15_l418_41857

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end NUMINAMATH_GPT_ceil_sqrt_225_eq_15_l418_41857


namespace NUMINAMATH_GPT_puppies_per_cage_calculation_l418_41818

noncomputable def initial_puppies : ℝ := 18.0
noncomputable def additional_puppies : ℝ := 3.0
noncomputable def total_puppies : ℝ := initial_puppies + additional_puppies
noncomputable def total_cages : ℝ := 4.2
noncomputable def puppies_per_cage : ℝ := total_puppies / total_cages

theorem puppies_per_cage_calculation :
  puppies_per_cage = 5.0 :=
by
  sorry

end NUMINAMATH_GPT_puppies_per_cage_calculation_l418_41818


namespace NUMINAMATH_GPT_find_angle_B_l418_41847

-- Define the parallel lines and angles
variables (l m : ℝ) -- Representing the lines as real numbers for simplicity
variables (A C B : ℝ) -- Representing the angles as real numbers

-- The conditions
def parallel_lines (l m : ℝ) : Prop := l = m
def angle_A (A : ℝ) : Prop := A = 100
def angle_C (C : ℝ) : Prop := C = 60

-- The theorem stating that, given the conditions, the angle B is 120 degrees
theorem find_angle_B (l m : ℝ) (A C B : ℝ) 
  (h1 : parallel_lines l m) 
  (h2 : angle_A A) 
  (h3 : angle_C C) : B = 120 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l418_41847


namespace NUMINAMATH_GPT_mosel_fills_315_boxes_per_week_l418_41800

-- Definitions for the conditions given in the problem.
def hens : ℕ := 270
def eggs_per_hen_per_day : ℕ := 1
def boxes_capacity : ℕ := 6
def days_per_week : ℕ := 7

-- Objective: Prove that the number of boxes filled each week is 315
theorem mosel_fills_315_boxes_per_week :
  let eggs_per_day := hens * eggs_per_hen_per_day
  let boxes_per_day := eggs_per_day / boxes_capacity
  let boxes_per_week := boxes_per_day * days_per_week
  boxes_per_week = 315 := by
  sorry

end NUMINAMATH_GPT_mosel_fills_315_boxes_per_week_l418_41800


namespace NUMINAMATH_GPT_total_height_geometric_solid_l418_41833

-- Definitions corresponding to conditions
def radius_cylinder1 : ℝ := 1
def radius_cylinder2 : ℝ := 3
def height_water_surface_figure2 : ℝ := 20
def height_water_surface_figure3 : ℝ := 28

-- The total height of the geometric solid is 29 cm
theorem total_height_geometric_solid :
  ∃ height_total : ℝ,
    (height_water_surface_figure2 + height_total - height_water_surface_figure3) = 29 :=
sorry

end NUMINAMATH_GPT_total_height_geometric_solid_l418_41833


namespace NUMINAMATH_GPT_composite_product_division_l418_41823

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end NUMINAMATH_GPT_composite_product_division_l418_41823


namespace NUMINAMATH_GPT_max_radius_approx_l418_41895

open Real

def angle_constraint (θ : ℝ) : Prop :=
  π / 4 ≤ θ ∧ θ ≤ 3 * π / 4

def wire_constraint (r θ : ℝ) : Prop :=
  16 = r * (2 + θ)

noncomputable def max_radius (θ : ℝ) : ℝ :=
  16 / (2 + θ)

theorem max_radius_approx :
  ∃ r θ, angle_constraint θ ∧ wire_constraint r θ ∧ abs (r - 3.673) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_max_radius_approx_l418_41895


namespace NUMINAMATH_GPT_interval_length_l418_41894

theorem interval_length (a b m h : ℝ) (h_eq : h = m / |a - b|) : |a - b| = m / h := 
by 
  sorry

end NUMINAMATH_GPT_interval_length_l418_41894


namespace NUMINAMATH_GPT_sum_squares_and_products_of_nonneg_reals_l418_41889

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end NUMINAMATH_GPT_sum_squares_and_products_of_nonneg_reals_l418_41889


namespace NUMINAMATH_GPT_smallest_sum_l418_41860

theorem smallest_sum (a b : ℕ) (h1 : 3^8 * 5^2 = a^b) (h2 : 0 < a) (h3 : 0 < b) : a + b = 407 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l418_41860


namespace NUMINAMATH_GPT_QR_value_l418_41883

-- Given conditions for the problem
def QP : ℝ := 15
def sinQ : ℝ := 0.4

-- Define QR based on the given conditions
noncomputable def QR : ℝ := QP / sinQ

-- The theorem to prove that QR = 37.5
theorem QR_value : QR = 37.5 := 
by
  unfold QR QP sinQ
  sorry

end NUMINAMATH_GPT_QR_value_l418_41883


namespace NUMINAMATH_GPT_triangle_area_l418_41805

theorem triangle_area {a b : ℝ} (h₁ : a = 3) (h₂ : b = 4) (h₃ : Real.sin (C : ℝ) = 1/2) :
  let area := (1 / 2) * a * b * (Real.sin C) 
  area = 3 := 
by
  rw [h₁, h₂, h₃]
  simp [Real.sin, mul_assoc]
  sorry

end NUMINAMATH_GPT_triangle_area_l418_41805


namespace NUMINAMATH_GPT_fractional_eq_solution_range_l418_41878

theorem fractional_eq_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 3 ∧ x > 0) ↔ m < -3 :=
by
  sorry

end NUMINAMATH_GPT_fractional_eq_solution_range_l418_41878


namespace NUMINAMATH_GPT_inequality_solution_set_l418_41803

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1 / 3 ≤ x ∧ x < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l418_41803


namespace NUMINAMATH_GPT_wheat_acres_l418_41801

theorem wheat_acres (x y : ℤ) 
  (h1 : x + y = 4500) 
  (h2 : 42 * x + 35 * y = 165200) : 
  y = 3400 :=
sorry

end NUMINAMATH_GPT_wheat_acres_l418_41801


namespace NUMINAMATH_GPT_sum_is_composite_l418_41893

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ x * y = a + b + c + d :=
sorry

end NUMINAMATH_GPT_sum_is_composite_l418_41893


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l418_41814

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l418_41814


namespace NUMINAMATH_GPT_simplify_expression_l418_41864

theorem simplify_expression : 
  (6^8 - 4^7) * (2^3 - (-2)^3) ^ 10 = 1663232 * 16 ^ 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l418_41864


namespace NUMINAMATH_GPT_toys_total_is_240_l418_41888

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end NUMINAMATH_GPT_toys_total_is_240_l418_41888


namespace NUMINAMATH_GPT_pesto_calculation_l418_41849

def basil_needed_per_pesto : ℕ := 4
def basil_harvest_per_week : ℕ := 16
def weeks : ℕ := 8
def total_basil_harvested : ℕ := basil_harvest_per_week * weeks
def total_pesto_possible : ℕ := total_basil_harvested / basil_needed_per_pesto

theorem pesto_calculation :
  total_pesto_possible = 32 :=
by
  sorry

end NUMINAMATH_GPT_pesto_calculation_l418_41849


namespace NUMINAMATH_GPT_square_of_second_arm_l418_41826

theorem square_of_second_arm (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
sorry

end NUMINAMATH_GPT_square_of_second_arm_l418_41826


namespace NUMINAMATH_GPT_product_173_240_l418_41882

theorem product_173_240 :
  ∃ n : ℕ, n = 3460 ∧ n * 12 = 173 * 240 ∧ 173 * 240 = 41520 :=
by
  sorry

end NUMINAMATH_GPT_product_173_240_l418_41882


namespace NUMINAMATH_GPT_percent_increase_of_income_l418_41891

theorem percent_increase_of_income (original_income new_income : ℝ) 
  (h1 : original_income = 120) (h2 : new_income = 180) :
  ((new_income - original_income) / original_income) * 100 = 50 := 
by 
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_percent_increase_of_income_l418_41891


namespace NUMINAMATH_GPT_aram_fraction_of_fine_l418_41842

theorem aram_fraction_of_fine (F : ℝ) (H1 : Joe_paid = (1/4)*F + 3)
  (H2 : Peter_paid = (1/3)*F - 3)
  (H3 : Aram_paid = (1/2)*F - 4)
  (H4 : Joe_paid + Peter_paid + Aram_paid = F) : 
  Aram_paid / F = 5 / 12 := 
sorry

end NUMINAMATH_GPT_aram_fraction_of_fine_l418_41842


namespace NUMINAMATH_GPT_granola_bars_distribution_l418_41850

theorem granola_bars_distribution
  (total_bars : ℕ)
  (eaten_bars : ℕ)
  (num_children : ℕ)
  (remaining_bars := total_bars - eaten_bars)
  (bars_per_child := remaining_bars / num_children) :
  total_bars = 200 → eaten_bars = 80 → num_children = 6 → bars_per_child = 20 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_granola_bars_distribution_l418_41850


namespace NUMINAMATH_GPT_sin_double_angle_l418_41835

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 4 / 5) : Real.sin (2 * α) = -24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l418_41835


namespace NUMINAMATH_GPT_f_properties_l418_41898

noncomputable def f : ℚ × ℚ → ℚ := sorry

theorem f_properties :
  (∀ (x y z : ℚ), f (x*y, z) = f (x, z) * f (y, z)) →
  (∀ (x y z : ℚ), f (z, x*y) = f (z, x) * f (z, y)) →
  (∀ (x : ℚ), f (x, 1 - x) = 1) →
  (∀ (x : ℚ), f (x, x) = 1) ∧
  (∀ (x : ℚ), f (x, -x) = 1) ∧
  (∀ (x y : ℚ), f (x, y) * f (y, x) = 1) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_f_properties_l418_41898


namespace NUMINAMATH_GPT_remainder_when_eight_n_plus_five_divided_by_eleven_l418_41863

theorem remainder_when_eight_n_plus_five_divided_by_eleven
  (n : ℤ) (h : n % 11 = 4) : (8 * n + 5) % 11 = 4 := 
  sorry

end NUMINAMATH_GPT_remainder_when_eight_n_plus_five_divided_by_eleven_l418_41863


namespace NUMINAMATH_GPT_volume_and_surface_area_of_prism_l418_41853

theorem volume_and_surface_area_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 24)
  (h2 : b * c = 18)
  (h3 : c * a = 12) :
  (a * b * c = 72) ∧ (2 * (a * b + b * c + c * a) = 108) := by
  sorry

end NUMINAMATH_GPT_volume_and_surface_area_of_prism_l418_41853


namespace NUMINAMATH_GPT_neg_p_implies_neg_q_l418_41899

variables {x : ℝ}

def condition_p (x : ℝ) : Prop := |x + 1| > 2
def condition_q (x : ℝ) : Prop := 5 * x - 6 > x^2
def neg_p (x : ℝ) : Prop := |x + 1| ≤ 2
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

theorem neg_p_implies_neg_q : (∀ x, neg_p x → neg_q x) :=
by 
  -- Proof is skipped according to the instructions
  sorry

end NUMINAMATH_GPT_neg_p_implies_neg_q_l418_41899


namespace NUMINAMATH_GPT_number_of_new_students_l418_41896

variable (O N : ℕ)
variable (H1 : 48 * O + 32 * N = 44 * 160)
variable (H2 : O + N = 160)

theorem number_of_new_students : N = 40 := sorry

end NUMINAMATH_GPT_number_of_new_students_l418_41896


namespace NUMINAMATH_GPT_find_f_at_2_l418_41832

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem find_f_at_2 (a b : ℝ) 
  (h1 : 3 + 2 * a + b = 0) 
  (h2 : 1 + a + b + 1 = -2) : 
  f a b 2 = 3 := 
by
  dsimp [f]
  sorry

end NUMINAMATH_GPT_find_f_at_2_l418_41832


namespace NUMINAMATH_GPT_length_of_side_of_largest_square_l418_41866

-- Definitions based on the conditions
def string_length : ℕ := 24

-- The main theorem corresponding to the problem statement.
theorem length_of_side_of_largest_square (h: string_length = 24) : 24 / 4 = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_side_of_largest_square_l418_41866


namespace NUMINAMATH_GPT_range_of_a_l418_41827

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 3| - |x + 2| ≥ Real.log a / Real.log 2) ↔ (0 < a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l418_41827


namespace NUMINAMATH_GPT_quadratic_point_comparison_l418_41868

theorem quadratic_point_comparison (c y1 y2 y3 : ℝ) 
  (h1 : y1 = -(-2:ℝ)^2 + c)
  (h2 : y2 = -(1:ℝ)^2 + c)
  (h3 : y3 = -(3:ℝ)^2 + c) : y2 > y1 ∧ y1 > y3 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_point_comparison_l418_41868


namespace NUMINAMATH_GPT_find_m_l418_41840

theorem find_m (m x1 x2 : ℝ) 
  (h1 : x1 * x1 - 2 * (m + 1) * x1 + m^2 + 2 = 0)
  (h2 : x2 * x2 - 2 * (m + 1) * x2 + m^2 + 2 = 0)
  (h3 : (x1 + 1) * (x2 + 1) = 8) : 
  m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l418_41840


namespace NUMINAMATH_GPT_x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l418_41861

theorem x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0 :
  (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end NUMINAMATH_GPT_x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l418_41861


namespace NUMINAMATH_GPT_angela_action_figures_l418_41834

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end NUMINAMATH_GPT_angela_action_figures_l418_41834


namespace NUMINAMATH_GPT_find_point_M_l418_41808

def parabola (x y : ℝ) := x^2 = 4 * y
def focus_dist (M : ℝ × ℝ) := dist M (0, 1) = 2
def point_on_parabola (M : ℝ × ℝ) := parabola M.1 M.2

theorem find_point_M (M : ℝ × ℝ) (h1 : point_on_parabola M) (h2 : focus_dist M) :
  M = (2, 1) ∨ M = (-2, 1) := by
  sorry

end NUMINAMATH_GPT_find_point_M_l418_41808


namespace NUMINAMATH_GPT_max_f_value_l418_41825

noncomputable def f (x : ℝ) : ℝ := 9 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 15 :=
by
  sorry

end NUMINAMATH_GPT_max_f_value_l418_41825


namespace NUMINAMATH_GPT_negation_example_l418_41874

theorem negation_example (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_example_l418_41874


namespace NUMINAMATH_GPT_trig_identity_l418_41828

theorem trig_identity (α a : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : (Real.tan α) + (1 / (Real.tan α)) = a) : 
    (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt (a^2 + 2 * a) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l418_41828


namespace NUMINAMATH_GPT_measure_of_angle_C_l418_41867

-- Definitions of the angles
def angles (A B C : ℝ) : Prop :=
  -- Conditions: measure of angle A is 1/4 of measure of angle B
  A = (1 / 4) * B ∧
  -- Lines p and q are parallel so alternate interior angles are equal
  C = A ∧
  -- Since angles B and C are supplementary
  B + C = 180

-- The problem in Lean 4 statement: Prove that C = 36 given the conditions
theorem measure_of_angle_C (A B C : ℝ) (h : angles A B C) : C = 36 := sorry

end NUMINAMATH_GPT_measure_of_angle_C_l418_41867


namespace NUMINAMATH_GPT_final_selling_price_l418_41869

def actual_price : ℝ := 9941.52
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

noncomputable def final_price (P : ℝ) : ℝ :=
  P * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem final_selling_price :
  final_price actual_price = 6800.00 :=
by
  sorry

end NUMINAMATH_GPT_final_selling_price_l418_41869


namespace NUMINAMATH_GPT_base7_difference_l418_41841

theorem base7_difference (a b : ℕ) (h₁ : a = 12100) (h₂ : b = 3666) :
  ∃ c, c = 1111 ∧ (a - b = c) := by
sorry

end NUMINAMATH_GPT_base7_difference_l418_41841


namespace NUMINAMATH_GPT_num_students_in_section_A_l418_41862

def avg_weight (total_weight : ℕ) (total_students : ℕ) : ℕ :=
  total_weight / total_students

variables (x : ℕ) -- number of students in section A
variables (weight_A : ℕ := 40 * x) -- total weight of section A
variables (students_B : ℕ := 20)
variables (weight_B : ℕ := 20 * 35) -- total weight of section B
variables (total_weight : ℕ := weight_A + weight_B) -- total weight of the whole class
variables (total_students : ℕ := x + students_B) -- total number of students in the class
variables (avg_weight_class : ℕ := avg_weight total_weight total_students)

theorem num_students_in_section_A :
  avg_weight_class = 38 → x = 30 :=
by
-- The proof will go here
sorry

end NUMINAMATH_GPT_num_students_in_section_A_l418_41862


namespace NUMINAMATH_GPT_original_number_without_10s_digit_l418_41822

theorem original_number_without_10s_digit (h : ℕ) (n : ℕ) 
  (h_eq_1 : h = 1) 
  (n_eq : n = 2 * 1000 + h * 100 + 84) 
  (div_by_6: n % 6 = 0) : n = 2184 → 284 = 284 :=
by
  sorry

end NUMINAMATH_GPT_original_number_without_10s_digit_l418_41822


namespace NUMINAMATH_GPT_parallel_vectors_l418_41810

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 3)) (h₂ : b = (-1, 2)) :
  (m * a.1 + b.1) * (-1) - 4 * (m * a.2 + b.2) = 0 → m = -1 / 2 :=
by
  intro h
  rw [h₁, h₂] at h
  simp at h
  sorry

end NUMINAMATH_GPT_parallel_vectors_l418_41810


namespace NUMINAMATH_GPT_max_value_of_sin2A_tan2B_l418_41844

-- Definitions for the trigonometric functions and angles in triangle ABC
variables {A B C : ℝ}

-- Condition: sin^2 A + sin^2 B = sin^2 C - sqrt 2 * sin A * sin B
def condition (A B C : ℝ) : Prop :=
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 = (Real.sin C) ^ 2 - Real.sqrt 2 * (Real.sin A) * (Real.sin B)

-- Question: Find the maximum value of sin 2A * tan^2 B
noncomputable def target (A B : ℝ) : ℝ :=
  Real.sin (2 * A) * (Real.tan B) ^ 2

-- The proof statement
theorem max_value_of_sin2A_tan2B (h : condition A B C) : ∃ (max_val : ℝ), max_val = 3 - 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), target A x ≤ max_val := 
sorry

end NUMINAMATH_GPT_max_value_of_sin2A_tan2B_l418_41844


namespace NUMINAMATH_GPT_probability_of_white_balls_from_both_boxes_l418_41824

theorem probability_of_white_balls_from_both_boxes :
  let P_white_A := 3 / (3 + 2)
  let P_white_B := 2 / (2 + 3)
  P_white_A * P_white_B = 6 / 25 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_white_balls_from_both_boxes_l418_41824


namespace NUMINAMATH_GPT_option_d_correct_l418_41876

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end NUMINAMATH_GPT_option_d_correct_l418_41876


namespace NUMINAMATH_GPT_odd_and_monotonic_l418_41804

-- Definitions based on the conditions identified
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_monotonic_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x ≤ f y
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement without the proof
theorem odd_and_monotonic :
  is_odd f ∧ is_monotonic_increasing f :=
sorry

end NUMINAMATH_GPT_odd_and_monotonic_l418_41804


namespace NUMINAMATH_GPT_rectangle_width_l418_41813

theorem rectangle_width (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w + l) = w * l) : w = 3 :=
by sorry

end NUMINAMATH_GPT_rectangle_width_l418_41813


namespace NUMINAMATH_GPT_river_current_speed_l418_41816

def motorboat_speed_still_water : ℝ := 20
def distance_between_points : ℝ := 60
def total_trip_time : ℝ := 6.25

theorem river_current_speed : ∃ v_T : ℝ, v_T = 4 ∧ 
  (distance_between_points / (motorboat_speed_still_water + v_T)) + 
  (distance_between_points / (motorboat_speed_still_water - v_T)) = total_trip_time := 
sorry

end NUMINAMATH_GPT_river_current_speed_l418_41816


namespace NUMINAMATH_GPT_percentage_discount_l418_41821

theorem percentage_discount (P D: ℝ) 
  (sale_price: P * (100 - D) / 100 = 78.2)
  (final_price_increase: 78.2 * 1.25 = P - 5.75):
  D = 24.44 :=
by
  sorry

end NUMINAMATH_GPT_percentage_discount_l418_41821


namespace NUMINAMATH_GPT_yura_picture_dimensions_l418_41875

theorem yura_picture_dimensions (a b : ℕ) (h : (a + 2) * (b + 2) - a * b = a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
by
  -- Place your proof here
  sorry

end NUMINAMATH_GPT_yura_picture_dimensions_l418_41875


namespace NUMINAMATH_GPT_lana_average_speed_l418_41877

theorem lana_average_speed (initial_reading : ℕ) (final_reading : ℕ) (time_first_day : ℕ) (time_second_day : ℕ) :
  initial_reading = 1991 → 
  final_reading = 2332 → 
  time_first_day = 5 → 
  time_second_day = 7 → 
  (final_reading - initial_reading) / (time_first_day + time_second_day : ℝ) = 28.4 :=
by
  intros h_init h_final h_first h_second
  rw [h_init, h_final, h_first, h_second]
  norm_num
  sorry

end NUMINAMATH_GPT_lana_average_speed_l418_41877


namespace NUMINAMATH_GPT_smallest_three_digit_number_exists_l418_41870

def is_valid_permutation_sum (x y z : ℕ) : Prop :=
  let perms := [100*x + 10*y + z, 100*x + 10*z + y, 100*y + 10*x + z, 100*z + 10*x + y, 100*y + 10*z + x, 100*z + 10*y + x]
  perms.sum = 2220

theorem smallest_three_digit_number_exists : ∃ (x y z : ℕ), x < y ∧ y < z ∧ x + y + z = 10 ∧ is_valid_permutation_sum x y z ∧ 100 * x + 10 * y + z = 127 :=
by {
  -- proof goal and steps would go here if we were to complete the proof
  sorry
}

end NUMINAMATH_GPT_smallest_three_digit_number_exists_l418_41870


namespace NUMINAMATH_GPT_no_infinite_lines_satisfying_conditions_l418_41817

theorem no_infinite_lines_satisfying_conditions :
  ¬ ∃ (l : ℕ → ℝ → ℝ → Prop)
      (k : ℕ → ℝ)
      (a b : ℕ → ℝ),
    (∀ n, l n 1 1) ∧
    (∀ n, k (n + 1) = a n - b n) ∧
    (∀ n, k n * k (n + 1) ≥ 0) := 
sorry

end NUMINAMATH_GPT_no_infinite_lines_satisfying_conditions_l418_41817
