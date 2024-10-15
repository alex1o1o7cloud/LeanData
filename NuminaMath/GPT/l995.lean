import Mathlib

namespace NUMINAMATH_GPT_not_proportional_l995_99513

theorem not_proportional (x y : ℕ) :
  (∀ k : ℝ, y ≠ 3 * x - 7 ∧ y ≠ (13 - 4 * x) / 3) → 
  ((y = 3 * x - 7 ∨ y = (13 - 4 * x) / 3) → ¬(∃ k : ℝ, (y = k * x) ∨ (y = k / x))) := sorry

end NUMINAMATH_GPT_not_proportional_l995_99513


namespace NUMINAMATH_GPT_airplane_seats_l995_99593

theorem airplane_seats (F : ℕ) (h : F + 4 * F + 2 = 387) : F = 77 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_airplane_seats_l995_99593


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l995_99585

def P (x : ℝ) : Prop := 1 < x ∧ x < 4
def Q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem intersection_of_P_and_Q (x : ℝ) : P x ∧ Q x ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l995_99585


namespace NUMINAMATH_GPT_max_f_eq_find_a_l995_99516

open Real

noncomputable def f (α : ℝ) : ℝ :=
  let a := (sin α, cos α)
  let b := (6 * sin α + cos α, 7 * sin α - 2 * cos α)
  a.1 * b.1 + a.2 * b.2

theorem max_f_eq : 
  ∃ α : ℝ, f α = 4 * sqrt 2 + 2 :=
sorry

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_A : ℝ)

noncomputable def f_triangle (A : ℝ) : ℝ :=
  let a := (sin A, cos A)
  let b := (6 * sin A + cos A, 7 * sin A - 2 * cos A)
  a.1 * b.1 + a.2 * b.2

axiom f_A_eq (A : ℝ) : f_triangle A = 6

theorem find_a (A B C a b c : ℝ) (h₁ : f_triangle A = 6) (h₂ : 1 / 2 * b * c * sin A = 3) (h₃ : b + c = 2 + 3 * sqrt 2) :
  a = sqrt 10 :=
sorry

end NUMINAMATH_GPT_max_f_eq_find_a_l995_99516


namespace NUMINAMATH_GPT_area_KLMQ_l995_99530

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def JR := 2
def RQ := 3
def JL := 8

def JLMR : Rectangle := {length := JL, width := JR}
def JKQR : Rectangle := {length := RQ, width := JR}

def RM : ℝ := JL
def QM : ℝ := RM - RQ
def LM : ℝ := JR

def KLMQ : Rectangle := {length := QM, width := LM}

theorem area_KLMQ : KLMQ.length * KLMQ.width = 10 :=
by
  sorry

end NUMINAMATH_GPT_area_KLMQ_l995_99530


namespace NUMINAMATH_GPT_decimal_palindrome_multiple_l995_99573

def is_decimal_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem decimal_palindrome_multiple (n : ℕ) (h : ¬ (10 ∣ n)) : 
  ∃ m : ℕ, is_decimal_palindrome m ∧ m % n = 0 :=
by sorry

end NUMINAMATH_GPT_decimal_palindrome_multiple_l995_99573


namespace NUMINAMATH_GPT_q_investment_correct_l995_99533

-- Define the conditions
def profit_ratio := (4, 6)
def p_investment := 60000
def expected_q_investment := 90000

-- Define the theorem statement
theorem q_investment_correct (p_investment: ℕ) (q_investment: ℕ) (profit_ratio : ℕ × ℕ)
  (h_ratio: profit_ratio = (4, 6)) (hp_investment: p_investment = 60000) :
  q_investment = 90000 := by
  sorry

end NUMINAMATH_GPT_q_investment_correct_l995_99533


namespace NUMINAMATH_GPT_village_household_count_l995_99562

theorem village_household_count
  (H : ℕ)
  (water_per_household_per_month : ℕ := 20)
  (total_water : ℕ := 2000)
  (duration_months : ℕ := 10)
  (total_consumption_condition : water_per_household_per_month * H * duration_months = total_water) :
  H = 10 :=
by
  sorry

end NUMINAMATH_GPT_village_household_count_l995_99562


namespace NUMINAMATH_GPT_total_students_exam_l995_99546

theorem total_students_exam (N T T' T'' : ℕ) (h1 : T = 88 * N) (h2 : T' = T - 8 * 50) 
  (h3 : T' = 92 * (N - 8)) (h4 : T'' = T' - 100) (h5 : T'' = 92 * (N - 9)) : N = 84 :=
by
  sorry

end NUMINAMATH_GPT_total_students_exam_l995_99546


namespace NUMINAMATH_GPT_walkway_time_against_direction_l995_99532

theorem walkway_time_against_direction (v_p v_w t : ℝ) (h1 : 90 = (v_p + v_w) * 30)
  (h2 : v_p * 48 = 90) 
  (h3 : 90 = (v_p - v_w) * t) :
  t = 120 := by 
  sorry

end NUMINAMATH_GPT_walkway_time_against_direction_l995_99532


namespace NUMINAMATH_GPT_max_x1_squared_plus_x2_squared_l995_99564

theorem max_x1_squared_plus_x2_squared (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = k - 2)
  (h2 : x₁ * x₂ = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) :
  x₁ ^ 2 + x₂ ^ 2 ≤ 18 :=
sorry

end NUMINAMATH_GPT_max_x1_squared_plus_x2_squared_l995_99564


namespace NUMINAMATH_GPT_simplify_expression_l995_99511

theorem simplify_expression :
  64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l995_99511


namespace NUMINAMATH_GPT_upper_bound_y_l995_99582

/-- 
  Theorem:
  For any real numbers x and y such that 3 < x < 6 and 6 < y, 
  if the greatest possible positive integer difference between x and y is 6,
  then the upper bound for y is 11.
 -/
theorem upper_bound_y (x y : ℝ) (h₁ : 3 < x) (h₂ : x < 6) (h₃ : 6 < y) (h₄ : y < some_number) (h₅ : y - x = 6) : y = 11 := 
by
  sorry

end NUMINAMATH_GPT_upper_bound_y_l995_99582


namespace NUMINAMATH_GPT_geometric_series_sum_l995_99549

  theorem geometric_series_sum :
    let a := (1 / 4 : ℚ)
    let r := (1 / 4 : ℚ)
    let n := 4
    let S_n := a * (1 - r^n) / (1 - r)
    S_n = 255 / 768 := by
  sorry
  
end NUMINAMATH_GPT_geometric_series_sum_l995_99549


namespace NUMINAMATH_GPT_negation_of_proposition_l995_99524

open Classical

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l995_99524


namespace NUMINAMATH_GPT_share_of_a_is_240_l995_99583

theorem share_of_a_is_240 (A B C : ℝ) 
  (h1 : A = (2/3) * (B + C)) 
  (h2 : B = (2/3) * (A + C)) 
  (h3 : A + B + C = 600) : 
  A = 240 := 
by sorry

end NUMINAMATH_GPT_share_of_a_is_240_l995_99583


namespace NUMINAMATH_GPT_puzzle_solution_l995_99588

-- Definitions for the digits
def K : ℕ := 3
def O : ℕ := 2
def M : ℕ := 4
def R : ℕ := 5
def E : ℕ := 6

-- The main proof statement
theorem puzzle_solution : (10 * K + O : ℕ) + (M / 10 + K / 10 + O / 100) = (10 * K + R : ℕ) + (O / 10 + M / 100) := 
  by 
  sorry

end NUMINAMATH_GPT_puzzle_solution_l995_99588


namespace NUMINAMATH_GPT_smallest_number_of_hikers_l995_99559

theorem smallest_number_of_hikers (n : ℕ) :
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 4) ↔ n = 154 :=
by sorry

end NUMINAMATH_GPT_smallest_number_of_hikers_l995_99559


namespace NUMINAMATH_GPT_second_train_start_time_l995_99568

theorem second_train_start_time :
  let start_time_first_train := 14 -- 2:00 pm in 24-hour format
  let catch_up_time := 22          -- 10:00 pm in 24-hour format
  let speed_first_train := 70      -- km/h
  let speed_second_train := 80     -- km/h
  let travel_time_first_train := catch_up_time - start_time_first_train
  let distance_first_train := speed_first_train * travel_time_first_train
  let t := distance_first_train / speed_second_train
  let start_time_second_train := catch_up_time - t
  start_time_second_train = 15 := -- 3:00 pm in 24-hour format
by
  sorry

end NUMINAMATH_GPT_second_train_start_time_l995_99568


namespace NUMINAMATH_GPT_smallest_positive_perfect_cube_l995_99512

theorem smallest_positive_perfect_cube (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ m : ℕ, m = (a * b * c^2)^3 ∧ (a^2 * b^3 * c^5 ∣ m)
:=
sorry

end NUMINAMATH_GPT_smallest_positive_perfect_cube_l995_99512


namespace NUMINAMATH_GPT_files_deleted_l995_99515

-- Definitions based on the conditions
def initial_files : ℕ := 93
def files_per_folder : ℕ := 8
def num_folders : ℕ := 9

-- The proof problem
theorem files_deleted : initial_files - (files_per_folder * num_folders) = 21 :=
by
  sorry

end NUMINAMATH_GPT_files_deleted_l995_99515


namespace NUMINAMATH_GPT_arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l995_99506

open Nat

variable (a : ℕ → ℝ)
variable (c : ℕ → ℝ)
variable (k b : ℝ)

-- Condition 1: sequence definition
def sequence_condition := ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + n + 1

-- Condition 2: initial value
def initial_value := a 1 = -1

-- Condition 3: c_n definition
def geometric_sequence_condition := ∀ n : ℕ, 0 < n → c (n + 1) / c n = 2

-- Problem 1: Arithmetic sequence parameters
theorem arith_sequence_parameters (h1 : sequence_condition a) (h2 : initial_value a) : a 1 = -3 ∧ 2 * (a 1 + 2) - a 1 - 7 = -1 :=
by sorry

-- Problem 2: Cannot be a geometric sequence
theorem not_geo_sequence (h1 : sequence_condition a) (h2 : initial_value a) : ¬ (∃ q, ∀ n : ℕ, 0 < n → a n * q = a (n + 1)) :=
by sorry

-- Problem 3: c_n is a geometric sequence and general term for a_n
theorem geo_sequence_and_gen_term (h1 : sequence_condition a) (h2 : initial_value a) 
    (h3 : ∀ n : ℕ, 0 < n → c n = a n + k * n + b)
    (hk : k = 1) (hb : b = 2) : sequence_condition a ∧ initial_value a :=
by sorry

end NUMINAMATH_GPT_arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l995_99506


namespace NUMINAMATH_GPT_range_of_k_if_intersection_empty_l995_99579

open Set

variable (k : ℝ)

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem range_of_k_if_intersection_empty (h : M ∩ N k = ∅) : k ≤ -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_k_if_intersection_empty_l995_99579


namespace NUMINAMATH_GPT_contribution_per_student_l995_99534

theorem contribution_per_student (total_contribution : ℝ) (class_funds : ℝ) (num_students : ℕ) 
(h1 : total_contribution = 90) (h2 : class_funds = 14) (h3 : num_students = 19) : 
  (total_contribution - class_funds) / num_students = 4 :=
by
  sorry

end NUMINAMATH_GPT_contribution_per_student_l995_99534


namespace NUMINAMATH_GPT_simplify_and_evaluate_l995_99519

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) : 
  ( ( (2 * x + 1) / x - 1 ) / ( (x^2 - 1) / x ) ) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l995_99519


namespace NUMINAMATH_GPT_circle_radius_and_circumference_l995_99509

theorem circle_radius_and_circumference (A : ℝ) (hA : A = 64 * Real.pi) :
  ∃ r C : ℝ, r = 8 ∧ C = 2 * Real.pi * r :=
by
  -- statement ensures that with given area A, you can find r and C satisfying the conditions.
  sorry

end NUMINAMATH_GPT_circle_radius_and_circumference_l995_99509


namespace NUMINAMATH_GPT_distance_from_center_of_C_to_line_l995_99554

def circle_center_distance : ℝ :=
  let line1 (x y : ℝ) := x - y - 4
  let circle1 (x y : ℝ) := x^2 + y^2 - 4 * x - 6
  let circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6
  let line2 (x y : ℝ) := 3 * x + 4 * y + 5
  sorry

theorem distance_from_center_of_C_to_line :
  circle_center_distance = 2 := sorry

end NUMINAMATH_GPT_distance_from_center_of_C_to_line_l995_99554


namespace NUMINAMATH_GPT_sum_of_reciprocals_l995_99548

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  (1 / x) + (1 / y) = 3 / 8 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l995_99548


namespace NUMINAMATH_GPT_find_f_of_minus_five_l995_99567

theorem find_f_of_minus_five (a b : ℝ) (f : ℝ → ℝ) (h1 : f 5 = 7) (h2 : ∀ x, f x = a * x + b * Real.sin x + 1) : f (-5) = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_minus_five_l995_99567


namespace NUMINAMATH_GPT_length_dg_l995_99598

theorem length_dg (a b k l S : ℕ) (h1 : S = 47 * (a + b)) 
                   (h2 : S = a * k) (h3 : S = b * l) (h4 : b = S / l) 
                   (h5 : a = S / k) (h6 : k * l = 47 * k + 47 * l + 2209) : 
  k = 2256 :=
by sorry

end NUMINAMATH_GPT_length_dg_l995_99598


namespace NUMINAMATH_GPT_smallest_possible_value_l995_99566

/-
Given:
1. m and n are positive integers.
2. gcd of m and n is (x + 5).
3. lcm of m and n is x * (x + 5).
4. m = 60.
5. x is a positive integer.

Prove:
The smallest possible value of n is 100.
-/

theorem smallest_possible_value 
  (m n x : ℕ) 
  (h1 : m = 60) 
  (h2 : x > 0) 
  (h3 : Nat.gcd m n = x + 5) 
  (h4 : Nat.lcm m n = x * (x + 5)) : 
  n = 100 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l995_99566


namespace NUMINAMATH_GPT_problem_inequality_l995_99577

theorem problem_inequality (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h: (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  a / b + b / c + c / a = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l995_99577


namespace NUMINAMATH_GPT_solve_for_V_l995_99543

open Real

theorem solve_for_V :
  ∃ k V, 
    (U = k * (V / W) ∧ (U = 16 ∧ W = 1 / 4 ∧ V = 2) ∧ (U = 25 ∧ W = 1 / 5 ∧ V = 2.5)) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_V_l995_99543


namespace NUMINAMATH_GPT_height_of_barbed_wire_l995_99547

theorem height_of_barbed_wire (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (total_cost : ℝ) (h : ℝ) :
  area = 3136 →
  cost_per_meter = 1.50 →
  gate_width = 2 →
  total_cost = 999 →
  h = 3 := 
by
  sorry

end NUMINAMATH_GPT_height_of_barbed_wire_l995_99547


namespace NUMINAMATH_GPT_pieces_of_green_candy_l995_99531

theorem pieces_of_green_candy (total_pieces red_pieces blue_pieces : ℝ)
  (h_total : total_pieces = 3409.7)
  (h_red : red_pieces = 145.5)
  (h_blue : blue_pieces = 785.2) :
  total_pieces - red_pieces - blue_pieces = 2479 := by
  sorry

end NUMINAMATH_GPT_pieces_of_green_candy_l995_99531


namespace NUMINAMATH_GPT_solutionToEquations_solutionToInequalities_l995_99507

-- Part 1: Solve the system of equations
def solveEquations (x y : ℝ) : Prop :=
2 * x - y = 3 ∧ x + y = 6

theorem solutionToEquations (x y : ℝ) (h : solveEquations x y) : 
x = 3 ∧ y = 3 :=
sorry

-- Part 2: Solve the system of inequalities
def solveInequalities (x : ℝ) : Prop :=
3 * x > x - 4 ∧ (4 + x) / 3 > x + 2

theorem solutionToInequalities (x : ℝ) (h : solveInequalities x) : 
-2 < x ∧ x < -1 :=
sorry

end NUMINAMATH_GPT_solutionToEquations_solutionToInequalities_l995_99507


namespace NUMINAMATH_GPT_distinct_arithmetic_progression_roots_l995_99537

theorem distinct_arithmetic_progression_roots (a b : ℝ) : 
  (∃ (d : ℝ), d ≠ 0 ∧ ∀ x, x^3 + a * x + b = 0 ↔ x = -d ∨ x = 0 ∨ x = d) → a < 0 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arithmetic_progression_roots_l995_99537


namespace NUMINAMATH_GPT_rectangle_to_square_l995_99517

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end NUMINAMATH_GPT_rectangle_to_square_l995_99517


namespace NUMINAMATH_GPT_find_line_and_intersection_l995_99553

def direct_proportion_function (k : ℝ) (x : ℝ) : ℝ :=
  k * x

def shifted_function (k : ℝ) (x b : ℝ) : ℝ :=
  k * x + b

theorem find_line_and_intersection
  (k : ℝ) (b : ℝ) (h₀ : direct_proportion_function k 1 = 2) (h₁ : b = 5) :
  (shifted_function k 1 b = 7) ∧ (shifted_function k (-5/2) b = 0) :=
by
  -- This is just a placeholder to indicate where the proof would go
  sorry

end NUMINAMATH_GPT_find_line_and_intersection_l995_99553


namespace NUMINAMATH_GPT_brad_red_balloons_l995_99591

theorem brad_red_balloons (total balloons green : ℕ) (h1 : total = 17) (h2 : green = 9) : total - green = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_brad_red_balloons_l995_99591


namespace NUMINAMATH_GPT_fifteenth_term_is_143_l995_99572

noncomputable def first_term : ℕ := 3
noncomputable def second_term : ℕ := 13
noncomputable def third_term : ℕ := 23
noncomputable def common_difference : ℕ := second_term - first_term
noncomputable def nth_term (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

theorem fifteenth_term_is_143 :
  nth_term 15 = 143 := by
  sorry

end NUMINAMATH_GPT_fifteenth_term_is_143_l995_99572


namespace NUMINAMATH_GPT_tangent_line_equation_at_1_range_of_a_l995_99590

noncomputable def f (x a : ℝ) : ℝ := (x+1) * Real.log x - a * (x-1)

-- (I) Tangent line equation when a = 4
theorem tangent_line_equation_at_1 (x : ℝ) (hx : x = 1) :
  let a := 4
  2*x + f 1 a - 2 = 0 :=
sorry

-- (II) Range of values for a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_at_1_range_of_a_l995_99590


namespace NUMINAMATH_GPT_translation_result_l995_99526

-- Define the original point M
def M : ℝ × ℝ := (-10, 1)

-- Define the translation on the y-axis by 4 units
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the resulting point M1 after translation
def M1 : ℝ × ℝ := translate_y M 4

-- The theorem we want to prove: the coordinates of M1 are (-10, 5)
theorem translation_result : M1 = (-10, 5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_translation_result_l995_99526


namespace NUMINAMATH_GPT_harmonic_mean_average_of_x_is_11_l995_99592

theorem harmonic_mean_average_of_x_is_11 :
  let h := (2 * 1008) / (2 + 1008)
  ∃ (x : ℕ), (h + x) / 2 = 11 → x = 18 := by
  sorry

end NUMINAMATH_GPT_harmonic_mean_average_of_x_is_11_l995_99592


namespace NUMINAMATH_GPT_melissa_total_points_l995_99599

-- Definition of the points scored per game and the number of games played.
def points_per_game : ℕ := 7
def number_of_games : ℕ := 3

-- The total points scored by Melissa is defined as the product of points per game and number of games.
def total_points_scored : ℕ := points_per_game * number_of_games

-- The theorem stating the verification of the total points scored by Melissa.
theorem melissa_total_points : total_points_scored = 21 := by
  -- The proof will be given here.
  sorry

end NUMINAMATH_GPT_melissa_total_points_l995_99599


namespace NUMINAMATH_GPT_min_C_over_D_l995_99584

-- Define y + 1/y = D and y^2 + 1/y^2 = C.
theorem min_C_over_D (y C D : ℝ) (hy_pos : 0 < y) (hC : y ^ 2 + 1 / (y ^ 2) = C) (hD : y + 1 / y = D) (hC_pos : 0 < C) (hD_pos : 0 < D) :
  C / D = 2 := by
  sorry

end NUMINAMATH_GPT_min_C_over_D_l995_99584


namespace NUMINAMATH_GPT_sequence_general_formula_l995_99560

theorem sequence_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+2) = 2 * a (n+1) / (2 + a (n+1))) :
  (a 1 = 1) → ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l995_99560


namespace NUMINAMATH_GPT_right_triangular_prism_volume_l995_99594

theorem right_triangular_prism_volume (R a h V : ℝ)
  (h1 : 4 * Real.pi * R^2 = 12 * Real.pi)
  (h2 : h = 2 * R)
  (h3 : (1 / 3) * (Real.sqrt 3 / 2) * a = R)
  (h4 : V = (1 / 2) * a * a * (Real.sin (Real.pi / 3)) * h) :
  V = 54 :=
by sorry

end NUMINAMATH_GPT_right_triangular_prism_volume_l995_99594


namespace NUMINAMATH_GPT_price_per_yellow_stamp_l995_99523

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end NUMINAMATH_GPT_price_per_yellow_stamp_l995_99523


namespace NUMINAMATH_GPT_correct_area_ratio_l995_99535

noncomputable def area_ratio (P : ℝ) : ℝ :=
  let x := P / 6 
  let length := P / 3
  let diagonal := (P * Real.sqrt 5) / 6
  let r := diagonal / 2
  let A := (5 * (P^2) * Real.pi) / 144
  let s := P / 5
  let R := P / (10 * Real.sin (36 * Real.pi / 180))
  let B := (P^2 * Real.pi) / (100 * (Real.sin (36 * Real.pi / 180))^2)
  A / B

theorem correct_area_ratio (P : ℝ) : area_ratio P = 500 * (Real.sin (36 * Real.pi / 180))^2 / 144 := 
  sorry

end NUMINAMATH_GPT_correct_area_ratio_l995_99535


namespace NUMINAMATH_GPT_smallest_abs_sum_l995_99596

open Matrix

noncomputable def matrix_square_eq (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![a, b], ![c, d]] * ![![a, b], ![c, d]]

theorem smallest_abs_sum (a b c d : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
    (M_eq : matrix_square_eq a b c d = ![![9, 0], ![0, 9]]) :
    |a| + |b| + |c| + |d| = 8 :=
sorry

end NUMINAMATH_GPT_smallest_abs_sum_l995_99596


namespace NUMINAMATH_GPT_symmetric_point_origin_l995_99502

theorem symmetric_point_origin (m : ℤ) : 
  (symmetry_condition : (3, m - 2) = (-(-3), -5)) → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l995_99502


namespace NUMINAMATH_GPT_sam_more_than_avg_l995_99552

def bridget_count : ℕ := 14
def reginald_count : ℕ := bridget_count - 2
def sam_count : ℕ := reginald_count + 4
def average_count : ℕ := (bridget_count + reginald_count + sam_count) / 3

theorem sam_more_than_avg 
    (h1 : bridget_count = 14) 
    (h2 : reginald_count = bridget_count - 2) 
    (h3 : sam_count = reginald_count + 4) 
    (h4 : average_count = (bridget_count + reginald_count + sam_count) / 3): 
    sam_count - average_count = 2 := 
  sorry

end NUMINAMATH_GPT_sam_more_than_avg_l995_99552


namespace NUMINAMATH_GPT_jeremy_remaining_money_l995_99551

-- Conditions as definitions
def computer_cost : ℝ := 3000
def accessories_cost : ℝ := 0.1 * computer_cost
def initial_money : ℝ := 2 * computer_cost

-- Theorem statement for the proof problem
theorem jeremy_remaining_money : initial_money - computer_cost - accessories_cost = 2700 := by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_jeremy_remaining_money_l995_99551


namespace NUMINAMATH_GPT_max_marks_l995_99527

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 165): M = 500 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l995_99527


namespace NUMINAMATH_GPT_unique_solution_for_digits_l995_99504

theorem unique_solution_for_digits :
  ∃ (A B C D E : ℕ),
  (A < B ∧ B < C ∧ C < D ∧ D < E) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
   C ≠ D ∧ C ≠ E ∧
   D ≠ E) ∧
  (10 * A + B) * C = 10 * D + E ∧
  (A = 1 ∧ B = 3 ∧ C = 6 ∧ D = 7 ∧ E = 8) :=
sorry

end NUMINAMATH_GPT_unique_solution_for_digits_l995_99504


namespace NUMINAMATH_GPT_median_of_roller_coaster_times_l995_99500

theorem median_of_roller_coaster_times:
  let data := [80, 85, 90, 125, 130, 135, 140, 145, 195, 195, 210, 215, 240, 245, 300, 305, 315, 320, 325, 330, 300]
  ∃ median_time, median_time = 210 ∧
    (∀ t ∈ data, t ≤ median_time ↔ index_of_median = 11) :=
by
  sorry

end NUMINAMATH_GPT_median_of_roller_coaster_times_l995_99500


namespace NUMINAMATH_GPT_count_silver_coins_l995_99586

theorem count_silver_coins 
  (gold_value : ℕ)
  (silver_value : ℕ)
  (num_gold_coins : ℕ)
  (cash : ℕ)
  (total_money : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  num_gold_coins = 3 →
  cash = 30 →
  total_money = 305 →
  ∃ S : ℕ, num_gold_coins * gold_value + S * silver_value + cash = total_money ∧ S = 5 := 
by
  sorry

end NUMINAMATH_GPT_count_silver_coins_l995_99586


namespace NUMINAMATH_GPT_xyz_expr_min_max_l995_99503

open Real

theorem xyz_expr_min_max (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 1) :
  ∃ m M : ℝ, m = 0 ∧ M = 1/4 ∧
    (∀ x y z : ℝ, x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
      xy + yz + zx - 3 * xyz ≥ m ∧ xy + yz + zx - 3 * xyz ≤ M) :=
sorry

end NUMINAMATH_GPT_xyz_expr_min_max_l995_99503


namespace NUMINAMATH_GPT_equilateral_triangle_l995_99576

theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ca) : a = b ∧ b = c := 
by sorry

end NUMINAMATH_GPT_equilateral_triangle_l995_99576


namespace NUMINAMATH_GPT_reciprocal_sum_of_roots_l995_99542

theorem reciprocal_sum_of_roots
  (a b c : ℝ)
  (ha : a^3 - 2022 * a + 1011 = 0)
  (hb : b^3 - 2022 * b + 1011 = 0)
  (hc : c^3 - 2022 * c + 1011 = 0)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (1 / a) + (1 / b) + (1 / c) = 2 :=
sorry

end NUMINAMATH_GPT_reciprocal_sum_of_roots_l995_99542


namespace NUMINAMATH_GPT_esperanza_gross_salary_l995_99587

def rent : ℕ := 600
def food_expenses (rent : ℕ) : ℕ := 3 * rent / 5
def mortgage_bill (food_expenses : ℕ) : ℕ := 3 * food_expenses
def savings : ℕ := 2000
def taxes (savings : ℕ) : ℕ := 2 * savings / 5
def total_expenses (rent food_expenses mortgage_bill taxes : ℕ) : ℕ :=
  rent + food_expenses + mortgage_bill + taxes
def gross_salary (total_expenses savings : ℕ) : ℕ :=
  total_expenses + savings

theorem esperanza_gross_salary : 
  gross_salary (total_expenses rent (food_expenses rent) (mortgage_bill (food_expenses rent)) (taxes savings)) savings = 4840 :=
by
  sorry

end NUMINAMATH_GPT_esperanza_gross_salary_l995_99587


namespace NUMINAMATH_GPT_book_arrangement_l995_99508

theorem book_arrangement :
  let total_books := 6
  let identical_books := 3
  let unique_arrangements := Nat.factorial total_books / Nat.factorial identical_books
  unique_arrangements = 120 := by
  sorry

end NUMINAMATH_GPT_book_arrangement_l995_99508


namespace NUMINAMATH_GPT_average_growth_rate_equation_l995_99571

-- Define the current and target processing capacities
def current_capacity : ℝ := 1000
def target_capacity : ℝ := 1200

-- Define the time period in months
def months : ℕ := 2

-- Define the monthly average growth rate
variable (x : ℝ)

-- The statement to be proven: current capacity increased by the growth rate over 2 months equals the target capacity 
theorem average_growth_rate_equation :
  current_capacity * (1 + x) ^ months = target_capacity :=
sorry

end NUMINAMATH_GPT_average_growth_rate_equation_l995_99571


namespace NUMINAMATH_GPT_inequality_solution_l995_99505

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l995_99505


namespace NUMINAMATH_GPT_constant_term_expansion_l995_99580

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ k : ℝ, k = -21/2 ∧
  (∀ r : ℕ, (9 : ℕ).choose r * (x^(1/2))^(9-r) * ((-(1/(2*x)))^r) = k) :=
sorry

end NUMINAMATH_GPT_constant_term_expansion_l995_99580


namespace NUMINAMATH_GPT_top_card_is_queen_probability_l995_99539

theorem top_card_is_queen_probability :
  let total_cards := 54
  let number_of_queens := 4
  (number_of_queens / total_cards) = (2 / 27) := by
    sorry

end NUMINAMATH_GPT_top_card_is_queen_probability_l995_99539


namespace NUMINAMATH_GPT_cost_of_pure_milk_l995_99558

theorem cost_of_pure_milk (C : ℝ) (total_milk : ℝ) (pure_milk : ℝ) (water : ℝ) (profit : ℝ) :
  total_milk = pure_milk + water → profit = (total_milk * C) - (pure_milk * C) → profit = 35 → C = 7 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_of_pure_milk_l995_99558


namespace NUMINAMATH_GPT_cone_sphere_volume_ratio_l995_99555

theorem cone_sphere_volume_ratio (r h : ℝ) 
  (radius_eq : r > 0)
  (volume_rel : (1 / 3 : ℝ) * π * r^2 * h = (1 / 3 : ℝ) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_sphere_volume_ratio_l995_99555


namespace NUMINAMATH_GPT_find_first_number_l995_99544

noncomputable def x : ℕ := 7981
noncomputable def y : ℕ := 9409
noncomputable def mean_proportional : ℕ := 8665

theorem find_first_number (mean_is_correct : (mean_proportional^2 = x * y)) : x = 7981 := by
-- Given: mean_proportional^2 = x * y
-- Goal: x = 7981
  sorry

end NUMINAMATH_GPT_find_first_number_l995_99544


namespace NUMINAMATH_GPT_P_eq_Q_l995_99597

def P (m : ℝ) : Prop := -1 < m ∧ m < 0

def quadratic_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 + 4 * m * x - 4 < 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, quadratic_inequality m x

theorem P_eq_Q : ∀ m : ℝ, P m ↔ Q m := 
by 
  sorry

end NUMINAMATH_GPT_P_eq_Q_l995_99597


namespace NUMINAMATH_GPT_annie_total_blocks_l995_99561

-- Definitions of the blocks traveled in each leg of Annie's journey
def walk_to_bus_stop := 5
def ride_bus_to_train_station := 7
def train_to_friends_house := 10
def walk_to_coffee_shop := 4
def walk_back_to_friends_house := walk_to_coffee_shop

-- The total blocks considering the round trip and additional walk to/from coffee shop
def total_blocks_traveled :=
  2 * (walk_to_bus_stop + ride_bus_to_train_station + train_to_friends_house) +
  walk_to_coffee_shop + walk_back_to_friends_house

-- Statement to prove
theorem annie_total_blocks : total_blocks_traveled = 52 :=
by
  sorry

end NUMINAMATH_GPT_annie_total_blocks_l995_99561


namespace NUMINAMATH_GPT_distance_home_to_school_l995_99522

theorem distance_home_to_school :
  ∃ (D : ℝ) (T : ℝ), 
    3 * (T + 7 / 60) = D ∧
    6 * (T - 8 / 60) = D ∧
    D = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_home_to_school_l995_99522


namespace NUMINAMATH_GPT_find_n_for_quadratic_roots_l995_99556

noncomputable def quadratic_root_properties (d c e n : ℝ) : Prop :=
  let A := (n + 2)
  let B := -((n + 2) * d + (n - 2) * c)
  let C := e * (n - 2)
  ∃ y1 y2 : ℝ, (A * y1 * y1 + B * y1 + C = 0) ∧ (A * y2 * y2 + B * y2 + C = 0) ∧ (y1 = -y2) ∧ (y1 + y2 = 0)

theorem find_n_for_quadratic_roots (d c e : ℝ) (h : d ≠ c) : 
  (quadratic_root_properties d c e (-2)) :=
sorry

end NUMINAMATH_GPT_find_n_for_quadratic_roots_l995_99556


namespace NUMINAMATH_GPT_seeds_per_plant_l995_99574

theorem seeds_per_plant :
  let trees := 2
  let plants_per_tree := 20
  let total_plants := trees * plants_per_tree
  let planted_trees := 24
  let planting_fraction := 0.60
  exists S : ℝ, planting_fraction * (total_plants * S) = planted_trees ∧ S = 1 :=
by
  sorry

end NUMINAMATH_GPT_seeds_per_plant_l995_99574


namespace NUMINAMATH_GPT_rational_square_of_1_minus_xy_l995_99570

theorem rational_square_of_1_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : ∃ (q : ℚ), 1 - x * y = q^2 :=
by
  sorry

end NUMINAMATH_GPT_rational_square_of_1_minus_xy_l995_99570


namespace NUMINAMATH_GPT_circle_is_axisymmetric_and_centrally_symmetric_l995_99569

structure Shape where
  isAxisymmetric : Prop
  isCentrallySymmetric : Prop

theorem circle_is_axisymmetric_and_centrally_symmetric :
  ∃ (s : Shape), s.isAxisymmetric ∧ s.isCentrallySymmetric :=
by
  sorry

end NUMINAMATH_GPT_circle_is_axisymmetric_and_centrally_symmetric_l995_99569


namespace NUMINAMATH_GPT_statement_A_l995_99520

theorem statement_A (x : ℝ) (h : x < -1) : x^2 > x :=
sorry

end NUMINAMATH_GPT_statement_A_l995_99520


namespace NUMINAMATH_GPT_total_boys_in_class_l995_99510

theorem total_boys_in_class (n : ℕ) (h_circle : ∀ i, 1 ≤ i ∧ i ≤ n -> i ≤ n) 
  (h_opposite : ∀ j k, j = 7 ∧ k = 27 ∧ j < k -> (k - j = n / 2)) : 
  n = 40 :=
sorry

end NUMINAMATH_GPT_total_boys_in_class_l995_99510


namespace NUMINAMATH_GPT_maximum_sum_minimum_difference_l995_99545

-- Definitions based on problem conditions
def is_least_common_multiple (m n lcm: ℕ) : Prop := Nat.lcm m n = lcm
def is_greatest_common_divisor (m n gcd: ℕ) : Prop := Nat.gcd m n = gcd

-- The target theorem to prove
theorem maximum_sum_minimum_difference (x y: ℕ) (h_lcm: is_least_common_multiple x y 2010) (h_gcd: is_greatest_common_divisor x y 2) :
  (x + y = 2012 ∧ x - y = 104 ∨ y - x = 104) :=
by
  sorry

end NUMINAMATH_GPT_maximum_sum_minimum_difference_l995_99545


namespace NUMINAMATH_GPT_inequality_am_gm_l995_99528

theorem inequality_am_gm 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) >= 6 := 
sorry

end NUMINAMATH_GPT_inequality_am_gm_l995_99528


namespace NUMINAMATH_GPT_sum_of_digits_next_perfect_square_222_l995_99536

-- Define the condition for the perfect square that begins with "222"
def starts_with_222 (n: ℕ) : Prop :=
  n / 10^3 = 222

-- Define the sum of the digits function
def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Statement for the Lean 4 statement: 
-- Prove that the sum of the digits of the next perfect square that starts with "222" is 18
theorem sum_of_digits_next_perfect_square_222 : sum_of_digits (492 ^ 2) = 18 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_sum_of_digits_next_perfect_square_222_l995_99536


namespace NUMINAMATH_GPT_sequence_periodicity_l995_99518

noncomputable def a : ℕ → ℚ
| 0       => 0
| (n + 1) => (a n - 2) / ((5/4) * a n - 2)

theorem sequence_periodicity : a 2017 = 0 := by
  sorry

end NUMINAMATH_GPT_sequence_periodicity_l995_99518


namespace NUMINAMATH_GPT_cost_of_notebook_is_12_l995_99540

/--
In a class of 36 students, a majority purchased notebooks. Each student bought the same number of notebooks (greater than 2). The price of a notebook in cents was double the number of notebooks each student bought, and the total expense was 2772 cents.
Prove that the cost of one notebook in cents is 12.
-/
theorem cost_of_notebook_is_12
  (s n c : ℕ) (total_students : ℕ := 36) 
  (h_majority : s > 18) 
  (h_notebooks : n > 2) 
  (h_cost : c = 2 * n) 
  (h_total_cost : s * c * n = 2772) 
  : c = 12 :=
by sorry

end NUMINAMATH_GPT_cost_of_notebook_is_12_l995_99540


namespace NUMINAMATH_GPT_koi_fish_multiple_l995_99589

theorem koi_fish_multiple (n m : ℕ) (h1 : n = 39) (h2 : m * n - 64 < n) : m * n = 78 :=
by
  sorry

end NUMINAMATH_GPT_koi_fish_multiple_l995_99589


namespace NUMINAMATH_GPT_total_tickets_sold_l995_99557

theorem total_tickets_sold 
(adult_ticket_price : ℕ) (child_ticket_price : ℕ) 
(total_revenue : ℕ) (adult_tickets_sold : ℕ) 
(child_tickets_sold : ℕ) (total_tickets : ℕ) : 
adult_ticket_price = 5 → 
child_ticket_price = 2 → 
total_revenue = 275 → 
adult_tickets_sold = 35 → 
(child_tickets_sold * child_ticket_price) + (adult_tickets_sold * adult_ticket_price) = total_revenue →
total_tickets = adult_tickets_sold + child_tickets_sold →
total_tickets = 85 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l995_99557


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l995_99578

theorem arithmetic_progression_sum (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d)
  (h2 : a 0 = 2)
  (h3 : a 1 + a 2 = 13) :
  a 3 + a 4 + a 5 = 42 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l995_99578


namespace NUMINAMATH_GPT_simplify_expression_l995_99525

theorem simplify_expression :
  (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l995_99525


namespace NUMINAMATH_GPT_smallest_positive_integer_l995_99521

theorem smallest_positive_integer (N : ℕ) :
  (N % 2 = 1) ∧
  (N % 3 = 2) ∧
  (N % 4 = 3) ∧
  (N % 5 = 4) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ↔ 
  N = 2519 := by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_integer_l995_99521


namespace NUMINAMATH_GPT_find_the_number_l995_99595

theorem find_the_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end NUMINAMATH_GPT_find_the_number_l995_99595


namespace NUMINAMATH_GPT_parabola_equation_l995_99575

theorem parabola_equation (a b c d e f: ℤ) (ha: a = 2) (hb: b = 0) (hc: c = 0) (hd: d = -16) (he: e = -1) (hf: f = 32) :
  ∃ x y : ℝ, 2 * x ^ 2 - 16 * x + 32 - y = 0 ∧ gcd (abs a) (gcd (abs b) (gcd (abs c) (gcd (abs d) (gcd (abs e) (abs f))))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l995_99575


namespace NUMINAMATH_GPT_max_square_test_plots_l995_99563

theorem max_square_test_plots
    (length : ℕ)
    (width : ℕ)
    (fence : ℕ)
    (fields_measure : length = 30 ∧ width = 45)
    (fence_measure : fence = 2250) :
  ∃ (number_of_plots : ℕ),
    number_of_plots = 150 :=
by
  sorry

end NUMINAMATH_GPT_max_square_test_plots_l995_99563


namespace NUMINAMATH_GPT_fewer_cans_today_l995_99529

variable (nc_sarah_yesterday : ℕ)
variable (nc_lara_yesterday : ℕ)
variable (nc_alex_yesterday : ℕ)
variable (nc_sarah_today : ℕ)
variable (nc_lara_today : ℕ)
variable (nc_alex_today : ℕ)

-- Given conditions
def yesterday_collected_cans : Prop :=
  nc_sarah_yesterday = 50 ∧
  nc_lara_yesterday = nc_sarah_yesterday + 30 ∧
  nc_alex_yesterday = 90

def today_collected_cans : Prop :=
  nc_sarah_today = 40 ∧
  nc_lara_today = 70 ∧
  nc_alex_today = 55

theorem fewer_cans_today :
  yesterday_collected_cans nc_sarah_yesterday nc_lara_yesterday nc_alex_yesterday →
  today_collected_cans nc_sarah_today nc_lara_today nc_alex_today →
  (nc_sarah_yesterday + nc_lara_yesterday + nc_alex_yesterday) -
  (nc_sarah_today + nc_lara_today + nc_alex_today) = 55 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_fewer_cans_today_l995_99529


namespace NUMINAMATH_GPT_quiz_common_difference_l995_99538

theorem quiz_common_difference 
  (x d : ℕ) 
  (h1 : x + 2 * d = 39) 
  (h2 : 8 * x + 28 * d = 360) 
  : d = 4 := 
  sorry

end NUMINAMATH_GPT_quiz_common_difference_l995_99538


namespace NUMINAMATH_GPT_simplify_fraction_l995_99581

-- We state the problem as a theorem.
theorem simplify_fraction : (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3 / 5 := by sorry

end NUMINAMATH_GPT_simplify_fraction_l995_99581


namespace NUMINAMATH_GPT_total_ticket_cost_l995_99565

theorem total_ticket_cost (x y : ℕ) 
  (h1 : x + y = 380) 
  (h2 : y = x + 240) 
  (cost_orchestra : ℕ := 12) 
  (cost_balcony : ℕ := 8): 
  12 * x + 8 * y = 3320 := 
by 
  sorry

end NUMINAMATH_GPT_total_ticket_cost_l995_99565


namespace NUMINAMATH_GPT_obtuse_triangle_has_exactly_one_obtuse_angle_l995_99501

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- Definition of an obtuse angle
def is_obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

-- The theorem statement
theorem obtuse_triangle_has_exactly_one_obtuse_angle {A B C : ℝ} 
  (h1 : is_obtuse_triangle A B C) : 
  (is_obtuse_angle A ∨ is_obtuse_angle B ∨ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle B) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle B ∧ is_obtuse_angle C) :=
sorry

end NUMINAMATH_GPT_obtuse_triangle_has_exactly_one_obtuse_angle_l995_99501


namespace NUMINAMATH_GPT_intersection_point_value_l995_99541

theorem intersection_point_value (c d: ℤ) (h1: d = 2 * -4 + c) (h2: -4 = 2 * d + c) : d = -4 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_value_l995_99541


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l995_99550

theorem problem_1 : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 :=
by sorry

theorem problem_2 : Real.sqrt (2 / 3) / Real.sqrt (8 / 27) = (3 / 2) :=
by sorry

theorem problem_3 : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = (10 * Real.sqrt 2 - 3 * Real.sqrt 3) :=
by sorry

theorem problem_4 : (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1 / 8) - Real.sqrt 24) = (Real.sqrt 2 / 4) + 3 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l995_99550


namespace NUMINAMATH_GPT_draw_3_odd_balls_from_15_is_336_l995_99514

-- Define the problem setting as given in the conditions
def odd_balls : Finset ℕ := {1, 3, 5, 7, 9, 11, 13, 15}

-- Define the function that calculates the number of ways to draw 3 balls
noncomputable def draw_3_odd_balls (S : Finset ℕ) : ℕ :=
  S.card * (S.card - 1) * (S.card - 2)

-- Prove that the drawing of 3 balls results in 336 ways
theorem draw_3_odd_balls_from_15_is_336 : draw_3_odd_balls odd_balls = 336 := by
  sorry

end NUMINAMATH_GPT_draw_3_odd_balls_from_15_is_336_l995_99514
