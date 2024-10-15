import Mathlib

namespace NUMINAMATH_GPT_factor_expression_l756_75641

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l756_75641


namespace NUMINAMATH_GPT_perimeter_of_region_l756_75669

theorem perimeter_of_region : 
  let side := 1
  let diameter := side
  let radius := diameter / 2
  let full_circumference := 2 * Real.pi * radius
  let arc_length := (3 / 4) * full_circumference
  let total_arcs := 4
  let perimeter := total_arcs * arc_length
  perimeter = 3 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_region_l756_75669


namespace NUMINAMATH_GPT_number_of_teams_l756_75685

theorem number_of_teams (n : ℕ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → (games_played : ℕ) = 4) 
  (h2 : ∀ (i j : ℕ), i ≠ j → (count : ℕ) = 760) : 
  n = 20 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_teams_l756_75685


namespace NUMINAMATH_GPT_solve_for_x_l756_75696

theorem solve_for_x 
    (x : ℝ) 
    (h : (4 * x - 2) / (5 * x - 5) = 3 / 4) 
    : x = -7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l756_75696


namespace NUMINAMATH_GPT_value_of_expression_l756_75661

theorem value_of_expression :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_expression_l756_75661


namespace NUMINAMATH_GPT_find_h_l756_75639

def f (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

theorem find_h : ∃ a h k, (h = -3 / 2) ∧ (f x = a * (x - h)^2 + k) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_h_l756_75639


namespace NUMINAMATH_GPT_last_two_digits_square_l756_75675

theorem last_two_digits_square (n : ℕ) (hnz : (n % 10 ≠ 0) ∧ ((n ^ 2) % 100 = n % 10 * 11)): ((n ^ 2) % 100 = 44) :=
sorry

end NUMINAMATH_GPT_last_two_digits_square_l756_75675


namespace NUMINAMATH_GPT_infinite_sqrt_eval_l756_75606

theorem infinite_sqrt_eval {x : ℝ} (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_GPT_infinite_sqrt_eval_l756_75606


namespace NUMINAMATH_GPT_total_worth_of_travelers_checks_l756_75637

theorem total_worth_of_travelers_checks (x y : ℕ) (h1 : x + y = 30) (h2 : 50 * (x - 18) + 100 * y = 900) : 
  50 * x + 100 * y = 1800 := 
by
  sorry

end NUMINAMATH_GPT_total_worth_of_travelers_checks_l756_75637


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l756_75646

theorem quadratic_inequality_solution (x : ℝ) :
    -15 * x^2 + 10 * x + 5 > 0 ↔ (-1 / 3 : ℝ) < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l756_75646


namespace NUMINAMATH_GPT_find_a_cubed_minus_b_cubed_l756_75698

theorem find_a_cubed_minus_b_cubed (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) : a^3 - b^3 = 486 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_cubed_minus_b_cubed_l756_75698


namespace NUMINAMATH_GPT_joe_lift_ratio_l756_75603

theorem joe_lift_ratio (F S : ℕ) 
  (h1 : F + S = 1800) 
  (h2 : F = 700) 
  (h3 : 2 * F = S + 300) : F / S = 7 / 11 :=
by
  sorry

end NUMINAMATH_GPT_joe_lift_ratio_l756_75603


namespace NUMINAMATH_GPT_min_time_to_one_ball_l756_75650

-- Define the problem in Lean
theorem min_time_to_one_ball (n : ℕ) (h : n = 99) : 
  ∃ T : ℕ, T = 98 ∧ ∀ t < T, ∃ ball_count : ℕ, ball_count > 1 :=
by
  -- Since we are not providing the proof, we use "sorry"
  sorry

end NUMINAMATH_GPT_min_time_to_one_ball_l756_75650


namespace NUMINAMATH_GPT_parallel_lines_m_values_l756_75659

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end NUMINAMATH_GPT_parallel_lines_m_values_l756_75659


namespace NUMINAMATH_GPT_range_of_m_l756_75630

noncomputable def f (x m : ℝ) := (1/2) * x^2 + m * x + Real.log x

noncomputable def f_prime (x m : ℝ) := x + 1/x + m

theorem range_of_m (x0 m : ℝ) 
  (h1 : (1/2) ≤ x0 ∧ x0 ≤ 3) 
  (unique_x0 : ∀ y, f_prime y m = 0 → y = x0) 
  (cond1 : f_prime (1/2) m < 0) 
  (cond2 : f_prime 3 m ≥ 0) 
  : -10 / 3 ≤ m ∧ m < -5 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l756_75630


namespace NUMINAMATH_GPT_arithmetic_square_root_of_9_l756_75691

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_9_l756_75691


namespace NUMINAMATH_GPT_product_equation_l756_75667

/-- Given two numbers x and y such that x + y = 20 and x - y = 4,
    the product of three times the larger number and the smaller number is 288. -/
theorem product_equation (x y : ℕ) (h1 : x + y = 20) (h2 : x - y = 4) (h3 : x > y) : 3 * x * y = 288 := 
sorry

end NUMINAMATH_GPT_product_equation_l756_75667


namespace NUMINAMATH_GPT_correct_equation_among_options_l756_75610

theorem correct_equation_among_options
  (a : ℝ) (x : ℝ) :
  (-- Option A
  ¬ ((-1)^3 = -3)) ∧
  (-- Option B
  ¬ (((-2)^2 * (-2)^3) = (-2)^6)) ∧
  (-- Option C
  ¬ ((2 * a - a) = 2)) ∧
  (-- Option D
  ((x - 2)^2 = x^2 - 4*x + 4)) :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_among_options_l756_75610


namespace NUMINAMATH_GPT_petya_can_write_divisible_by_2019_l756_75612

open Nat

theorem petya_can_write_divisible_by_2019 (M : ℕ) (h : ∃ k : ℕ, M = (10^k - 1) / 9) : ∃ N : ℕ, (N = (10^M - 1) / 9) ∧ 2019 ∣ N :=
by
  sorry

end NUMINAMATH_GPT_petya_can_write_divisible_by_2019_l756_75612


namespace NUMINAMATH_GPT_t_sum_max_min_l756_75662

noncomputable def t_max (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry
noncomputable def t_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry

theorem t_sum_max_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) :
  t_max a b h + t_min a b h = 16 / 7 := sorry

end NUMINAMATH_GPT_t_sum_max_min_l756_75662


namespace NUMINAMATH_GPT_angle_A_eq_pi_over_3_perimeter_eq_24_l756_75654

namespace TriangleProof

-- We introduce the basic setup for the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition
axiom condition : 2 * b = 2 * a * Real.cos C + c

-- Part 1: Prove angle A is π/3
theorem angle_A_eq_pi_over_3 (h : 2 * b = 2 * a * Real.cos C + c) :
  A = Real.pi / 3 :=
sorry

-- Part 2: Given a = 10 and the area is 8√3, prove perimeter is 24
theorem perimeter_eq_24 (a_eq_10 : a = 10) (area_eq_8sqrt3 : 8 * Real.sqrt 3 = (1 / 2) * b * c * Real.sin A) :
  a + b + c = 24 :=
sorry

end TriangleProof

end NUMINAMATH_GPT_angle_A_eq_pi_over_3_perimeter_eq_24_l756_75654


namespace NUMINAMATH_GPT_sqrt3_f_pi6_lt_f_pi3_l756_75699

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_derivative_tan_lt (x : ℝ) (h : 0 < x ∧ x < π / 2) : f x < (deriv f x) * tan x

theorem sqrt3_f_pi6_lt_f_pi3 :
  sqrt 3 * f (π / 6) < f (π / 3) :=
by
  sorry

end NUMINAMATH_GPT_sqrt3_f_pi6_lt_f_pi3_l756_75699


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l756_75697

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l756_75697


namespace NUMINAMATH_GPT_exponent_property_l756_75605

theorem exponent_property (a x y : ℝ) (h1 : 0 < a) (h2 : a ^ x = 2) (h3 : a ^ y = 3) : a ^ (x - y) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_exponent_property_l756_75605


namespace NUMINAMATH_GPT_initial_men_count_l756_75652

theorem initial_men_count (x : ℕ) (h : x * 25 = 15 * 60) : x = 36 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_count_l756_75652


namespace NUMINAMATH_GPT_remainder_when_divided_by_13_is_11_l756_75670

theorem remainder_when_divided_by_13_is_11 
  (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : 
  349 % 13 = 11 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_13_is_11_l756_75670


namespace NUMINAMATH_GPT_minimum_value_l756_75645

theorem minimum_value(a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (2 / a + 3 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_l756_75645


namespace NUMINAMATH_GPT_probability_A_mc_and_B_tf_probability_at_least_one_mc_l756_75686

section ProbabilityQuiz

variable (total_questions : ℕ) (mc_questions : ℕ) (tf_questions : ℕ)

def prob_A_mc_and_B_tf (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  (mc_questions * tf_questions : ℚ) / (total_questions * (total_questions - 1))

def prob_at_least_one_mc (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  1 - ((tf_questions * (tf_questions - 1) : ℚ) / (total_questions * (total_questions - 1)))

theorem probability_A_mc_and_B_tf :
  prob_A_mc_and_B_tf 10 6 4 = 4 / 15 := by
  sorry

theorem probability_at_least_one_mc :
  prob_at_least_one_mc 10 6 4 = 13 / 15 := by
  sorry

end ProbabilityQuiz

end NUMINAMATH_GPT_probability_A_mc_and_B_tf_probability_at_least_one_mc_l756_75686


namespace NUMINAMATH_GPT_distance_from_edge_l756_75624

theorem distance_from_edge (wall_width picture_width x : ℕ) (h_wall : wall_width = 24) (h_picture : picture_width = 4) (h_centered : x + picture_width + x = wall_width) : x = 10 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_distance_from_edge_l756_75624


namespace NUMINAMATH_GPT_flag_yellow_area_percentage_l756_75679

theorem flag_yellow_area_percentage (s w : ℝ) (h_flag_area : s > 0)
  (h_width_positive : w > 0) (h_cross_area : 4 * s * w - 3 * w^2 = 0.49 * s^2) :
  (w^2 / s^2) * 100 = 12.25 :=
by
  sorry

end NUMINAMATH_GPT_flag_yellow_area_percentage_l756_75679


namespace NUMINAMATH_GPT_circle_radius_triple_area_l756_75634

/-- Given the area of a circle is tripled when its radius r is increased by n, prove that 
    r = n * (sqrt(3) - 1) / 2 -/
theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_circle_radius_triple_area_l756_75634


namespace NUMINAMATH_GPT_smallest_value_a_plus_b_l756_75619

theorem smallest_value_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 3^7 * 5^3 = a^b) : a + b = 3376 :=
sorry

end NUMINAMATH_GPT_smallest_value_a_plus_b_l756_75619


namespace NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_2_3_5_l756_75695

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_2_3_5_l756_75695


namespace NUMINAMATH_GPT_find_coordinates_of_P_l756_75627

-- Define the problem conditions
def P (m : ℤ) := (2 * m + 4, m - 1)
def A := (2, -4)
def line_l (y : ℤ) := y = -4
def P_on_line_l (m : ℤ) := line_l (m - 1)

theorem find_coordinates_of_P (m : ℤ) (h : P_on_line_l m) : P m = (-2, -4) := 
  by sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l756_75627


namespace NUMINAMATH_GPT_dilation_complex_l756_75671

theorem dilation_complex :
  let c := (1 : ℂ) - (2 : ℂ) * I
  let k := 3
  let z := -1 + I
  (k * (z - c) + c = -5 + 7 * I) :=
by
  sorry

end NUMINAMATH_GPT_dilation_complex_l756_75671


namespace NUMINAMATH_GPT_shaded_area_percentage_is_correct_l756_75616

noncomputable def total_area_of_square : ℕ := 49

noncomputable def area_of_first_shaded_region : ℕ := 2^2

noncomputable def area_of_second_shaded_region : ℕ := 25 - 9

noncomputable def area_of_third_shaded_region : ℕ := 49 - 36

noncomputable def total_shaded_area : ℕ :=
  area_of_first_shaded_region + area_of_second_shaded_region + area_of_third_shaded_region

noncomputable def percent_shaded_area : ℚ :=
  (total_shaded_area : ℚ) / total_area_of_square * 100

theorem shaded_area_percentage_is_correct :
  percent_shaded_area = 67.35 := by
sorry

end NUMINAMATH_GPT_shaded_area_percentage_is_correct_l756_75616


namespace NUMINAMATH_GPT_find_a_b_sum_l756_75628

theorem find_a_b_sum (a b : ℕ) (h : a^2 - b^4 = 2009) : a + b = 47 :=
sorry

end NUMINAMATH_GPT_find_a_b_sum_l756_75628


namespace NUMINAMATH_GPT_gary_chickens_l756_75609

theorem gary_chickens (initial_chickens : ℕ) (multiplication_factor : ℕ) 
  (weekly_eggs : ℕ) (days_in_week : ℕ)
  (h1 : initial_chickens = 4)
  (h2 : multiplication_factor = 8)
  (h3 : weekly_eggs = 1344)
  (h4 : days_in_week = 7) :
  (weekly_eggs / days_in_week) / (initial_chickens * multiplication_factor) = 6 :=
by
  sorry

end NUMINAMATH_GPT_gary_chickens_l756_75609


namespace NUMINAMATH_GPT_range_of_alpha_l756_75615

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 5 * x

theorem range_of_alpha (α : ℝ) (h₀ : -1 < α) (h₁ : α < 1) (h₂ : f (1 - α) + f (1 - α^2) < 0) : 1 < α ∧ α < Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_range_of_alpha_l756_75615


namespace NUMINAMATH_GPT_inequality_solution_l756_75636

theorem inequality_solution (m : ℝ) (h : m < -1) :
  (if m = -3 then
    {x : ℝ | x > 1} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if -3 < m ∧ m < -1 then
    ({x : ℝ | x < m / (m + 3)} ∪ {x : ℝ | x > 1}) =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if m < -3 then
    {x : ℝ | 1 < x ∧ x < m / (m + 3)} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else
    False) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l756_75636


namespace NUMINAMATH_GPT_complement_union_eq_complement_l756_75602

open Set

variable (U : Set ℤ) 
variable (A : Set ℤ) 
variable (B : Set ℤ)

theorem complement_union_eq_complement : 
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} :=
by
  intros hU hA hB
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_complement_union_eq_complement_l756_75602


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l756_75693

-- Definitions based on given conditions
def propA (a b : ℕ) : Prop := a + b ≠ 4
def propB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem statement (proof not required)
theorem neither_sufficient_nor_necessary (a b : ℕ) :
  ¬ (propA a b → propB a b) ∧ ¬ (propB a b → propA a b) := 
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l756_75693


namespace NUMINAMATH_GPT_smallest_positive_value_of_a_minus_b_l756_75622

theorem smallest_positive_value_of_a_minus_b :
  ∃ (a b : ℤ), 17 * a + 6 * b = 13 ∧ a - b = 17 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_value_of_a_minus_b_l756_75622


namespace NUMINAMATH_GPT_cost_of_green_shirts_l756_75690

noncomputable def total_cost_kindergarten : ℝ := 101 * 5.8
noncomputable def total_cost_first_grade : ℝ := 113 * 5
noncomputable def total_cost_second_grade : ℝ := 107 * 5.6
noncomputable def total_cost_all_but_third : ℝ := total_cost_kindergarten + total_cost_first_grade + total_cost_second_grade
noncomputable def total_third_grade : ℝ := 2317 - total_cost_all_but_third
noncomputable def cost_per_third_grade_shirt : ℝ := total_third_grade / 108

theorem cost_of_green_shirts : cost_per_third_grade_shirt = 5.25 := sorry

end NUMINAMATH_GPT_cost_of_green_shirts_l756_75690


namespace NUMINAMATH_GPT_white_ring_weight_l756_75614

def weight_of_orange_ring : ℝ := 0.08
def weight_of_purple_ring : ℝ := 0.33
def total_weight_of_rings : ℝ := 0.83

def weight_of_white_ring (total : ℝ) (orange : ℝ) (purple : ℝ) : ℝ :=
  total - (orange + purple)

theorem white_ring_weight :
  weight_of_white_ring total_weight_of_rings weight_of_orange_ring weight_of_purple_ring = 0.42 :=
by
  sorry

end NUMINAMATH_GPT_white_ring_weight_l756_75614


namespace NUMINAMATH_GPT_union_set_eq_l756_75611

open Set

def P := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | x^2 ≤ 4}

theorem union_set_eq : P ∪ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_union_set_eq_l756_75611


namespace NUMINAMATH_GPT_mul_three_point_six_and_zero_point_twenty_five_l756_75607

theorem mul_three_point_six_and_zero_point_twenty_five : 3.6 * 0.25 = 0.9 := by 
  sorry

end NUMINAMATH_GPT_mul_three_point_six_and_zero_point_twenty_five_l756_75607


namespace NUMINAMATH_GPT_trees_died_in_typhoon_l756_75664

-- Define the total number of trees, survived trees, and died trees
def total_trees : ℕ := 14

def survived_trees (S : ℕ) : ℕ := S

def died_trees (S : ℕ) : ℕ := S + 4

-- The Lean statement that formalizes the proof problem
theorem trees_died_in_typhoon : ∃ S : ℕ, survived_trees S + died_trees S = total_trees ∧ died_trees S = 9 :=
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_trees_died_in_typhoon_l756_75664


namespace NUMINAMATH_GPT_chord_equation_l756_75665

-- Definitions and conditions
def parabola (x y : ℝ) := y^2 = 8 * x
def point_Q := (4, 1)

-- Statement to prove
theorem chord_equation :
  ∃ (m : ℝ) (c : ℝ), m = 4 ∧ c = -15 ∧
    ∀ (x y : ℝ), (parabola x y ∧ x + y = 8 ∧ y + y = 2) →
      4 * x - y = 15 :=
by
  sorry -- Proof elided

end NUMINAMATH_GPT_chord_equation_l756_75665


namespace NUMINAMATH_GPT_marbles_left_mrs_hilt_marbles_left_l756_75623

-- Define the initial number of marbles
def initial_marbles : ℕ := 38

-- Define the number of marbles lost
def marbles_lost : ℕ := 15

-- Define the number of marbles given away
def marbles_given_away : ℕ := 6

-- Define the number of marbles found
def marbles_found : ℕ := 8

-- Use these definitions to calculate the total number of marbles left
theorem marbles_left : ℕ :=
  initial_marbles - marbles_lost - marbles_given_away + marbles_found

-- Prove that total number of marbles left is 25
theorem mrs_hilt_marbles_left : marbles_left = 25 := by 
  sorry

end NUMINAMATH_GPT_marbles_left_mrs_hilt_marbles_left_l756_75623


namespace NUMINAMATH_GPT_determine_p_in_terms_of_q_l756_75618

variable {p q : ℝ}

-- Given the condition in the problem
def log_condition (p q : ℝ) : Prop :=
  Real.log p + 2 * Real.log q = Real.log (2 * p + q)

-- The goal is to prove that under this condition, the following holds
theorem determine_p_in_terms_of_q (h : log_condition p q) :
  p = q / (q^2 - 2) :=
sorry

end NUMINAMATH_GPT_determine_p_in_terms_of_q_l756_75618


namespace NUMINAMATH_GPT_like_terms_sum_l756_75692

theorem like_terms_sum (m n : ℕ) (h1 : 6 * x ^ 5 * y ^ (2 * n) = 6 * x ^ m * y ^ 4) : m + n = 7 := by
  sorry

end NUMINAMATH_GPT_like_terms_sum_l756_75692


namespace NUMINAMATH_GPT_ratio_of_x_intercepts_l756_75600

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_intercepts_l756_75600


namespace NUMINAMATH_GPT_charlie_golden_delicious_bags_l756_75608

theorem charlie_golden_delicious_bags :
  ∀ (total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags : ℝ),
  total_bags = 0.67 →
  macintosh_bags = 0.17 →
  cortland_bags = 0.33 →
  total_bags = golden_delicious_bags + macintosh_bags + cortland_bags →
  golden_delicious_bags = 0.17 := by
  intros total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags
  intros h_total h_macintosh h_cortland h_sum
  sorry

end NUMINAMATH_GPT_charlie_golden_delicious_bags_l756_75608


namespace NUMINAMATH_GPT_possible_measure_of_angle_AOC_l756_75653

-- Given conditions
def angle_AOB : ℝ := 120
def OC_bisects_angle_AOB (x : ℝ) : Prop := x = 60
def OD_bisects_angle_AOB_and_OC_bisects_angle (x y : ℝ) : Prop :=
  (y = 60 ∧ (x = 30 ∨ x = 90))

-- Theorem statement
theorem possible_measure_of_angle_AOC (angle_AOC : ℝ) :
  (OC_bisects_angle_AOB angle_AOC ∨ 
  (OD_bisects_angle_AOB_and_OC_bisects_angle angle_AOC 60)) →
  (angle_AOC = 30 ∨ angle_AOC = 60 ∨ angle_AOC = 90) :=
by
  sorry

end NUMINAMATH_GPT_possible_measure_of_angle_AOC_l756_75653


namespace NUMINAMATH_GPT_curlers_count_l756_75625

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end NUMINAMATH_GPT_curlers_count_l756_75625


namespace NUMINAMATH_GPT_koala_fiber_consumption_l756_75666

theorem koala_fiber_consumption (x : ℝ) (h : 0.40 * x = 8) : x = 20 :=
sorry

end NUMINAMATH_GPT_koala_fiber_consumption_l756_75666


namespace NUMINAMATH_GPT_monthly_salary_l756_75617

theorem monthly_salary (S : ℝ) (E : ℝ) 
  (h1 : S - 1.20 * E = 220)
  (h2 : E = 0.80 * S) :
  S = 5500 :=
by
  sorry

end NUMINAMATH_GPT_monthly_salary_l756_75617


namespace NUMINAMATH_GPT_trajectory_eqn_of_point_Q_l756_75635

theorem trajectory_eqn_of_point_Q 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (A : ℝ × ℝ := (-2, 0))
  (B : ℝ × ℝ := (2, 0))
  (l : ℝ := 10 / 3) 
  (hP_on_l : P.1 = l)
  (hQ_on_AP : (Q.2 * -4) = Q.1 * (P.2 - 0) - (P.2 * -4))
  (hBP_perp_BQ : (Q.2 * 4) = -Q.1 * ((3 * P.2) / 4 - 2))
: (Q.1^2 / 4) + Q.2^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_eqn_of_point_Q_l756_75635


namespace NUMINAMATH_GPT_minimum_value_l756_75647

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem minimum_value (a b c d : ℝ) (h1 : a < (2 / 3) * b) 
  (h2 : ∀ x, 3 * a * x^2 + 2 * b * x + c ≥ 0) : 
  ∃ (x : ℝ), ∀ c, 2 * b - 3 * a ≠ 0 → (c = (b^2 / 3 / a)) → (c / (2 * b - 3 * a) ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l756_75647


namespace NUMINAMATH_GPT_inequality_proof_l756_75626

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l756_75626


namespace NUMINAMATH_GPT_inv_prop_func_point_l756_75655

theorem inv_prop_func_point {k : ℝ} :
  (∃ y x : ℝ, y = k / x ∧ (x = 2 ∧ y = -1)) → k = -2 :=
by
  intro h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_inv_prop_func_point_l756_75655


namespace NUMINAMATH_GPT_expression_equals_neg_eight_l756_75601

variable {a b : ℝ}

theorem expression_equals_neg_eight (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| ≠ |b|) :
  ( (b^2 / a^2 + a^2 / b^2 - 2) * 
    ((a + b) / (b - a) + (b - a) / (a + b)) * 
    (((1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2)) - ((1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)))
  ) = -8 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_neg_eight_l756_75601


namespace NUMINAMATH_GPT_find_original_number_l756_75682

variable (x : ℝ)

def tripled := 3 * x
def doubled := 2 * tripled
def subtracted := doubled - 9
def trebled := 3 * subtracted

theorem find_original_number (h : trebled = 90) : x = 6.5 := by
  sorry

end NUMINAMATH_GPT_find_original_number_l756_75682


namespace NUMINAMATH_GPT_card_worth_l756_75656

theorem card_worth (value_per_card : ℕ) (num_cards_traded : ℕ) (profit : ℕ) (value_traded : ℕ) (worth_received : ℕ) :
  value_per_card = 8 →
  num_cards_traded = 2 →
  profit = 5 →
  value_traded = num_cards_traded * value_per_card →
  worth_received = value_traded + profit →
  worth_received = 21 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_card_worth_l756_75656


namespace NUMINAMATH_GPT_wrongly_entered_mark_l756_75681

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ (correct_mark avg_increase pupils : ℝ), 
  correct_mark = 45 ∧ avg_increase = 0.5 ∧ pupils = 80 ∧
  (avg_increase * pupils = (x - correct_mark)) →
  x = 85) :=
by 
  intro correct_mark avg_increase pupils
  rintro ⟨hc, ha, hp, h⟩
  sorry

end NUMINAMATH_GPT_wrongly_entered_mark_l756_75681


namespace NUMINAMATH_GPT_combined_distance_l756_75684

theorem combined_distance (t1 t2 : ℕ) (s1 s2 : ℝ)
  (h1 : t1 = 30) (h2 : s1 = 9.5) (h3 : t2 = 45) (h4 : s2 = 8.3)
  : (s1 * t1 + s2 * t2) = 658.5 := 
by
  sorry

end NUMINAMATH_GPT_combined_distance_l756_75684


namespace NUMINAMATH_GPT_magic_king_total_episodes_l756_75649

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end NUMINAMATH_GPT_magic_king_total_episodes_l756_75649


namespace NUMINAMATH_GPT_percentage_of_uninsured_part_time_l756_75660

noncomputable def number_of_employees := 330
noncomputable def uninsured_employees := 104
noncomputable def part_time_employees := 54
noncomputable def probability_neither := 0.5606060606060606

theorem percentage_of_uninsured_part_time:
  (13 / 104) * 100 = 12.5 := 
by 
  -- Here you can assume proof steps would occur/assertions to align with the solution found
  sorry

end NUMINAMATH_GPT_percentage_of_uninsured_part_time_l756_75660


namespace NUMINAMATH_GPT_range_of_2a_minus_b_l756_75629

variable (a b : ℝ)
variable (h1 : -2 < a ∧ a < 2)
variable (h2 : 2 < b ∧ b < 3)

theorem range_of_2a_minus_b (a b : ℝ) (h1 : -2 < a ∧ a < 2) (h2 : 2 < b ∧ b < 3) :
  -7 < 2 * a - b ∧ 2 * a - b < 2 := sorry

end NUMINAMATH_GPT_range_of_2a_minus_b_l756_75629


namespace NUMINAMATH_GPT_quadratic_least_value_l756_75658

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_least_value_l756_75658


namespace NUMINAMATH_GPT_sharpener_difference_l756_75688

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end NUMINAMATH_GPT_sharpener_difference_l756_75688


namespace NUMINAMATH_GPT_inverse_property_l756_75678

-- Given conditions
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variable (hf_injective : Function.Injective f)
variable (hf_surjective : Function.Surjective f)
variable (h_inverse : ∀ y : ℝ, f (f_inv y) = y)
variable (hf_property : ∀ x : ℝ, f (-x) + f (x) = 3)

-- The proof goal
theorem inverse_property (x : ℝ) : (f_inv (x - 1) + f_inv (4 - x)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_inverse_property_l756_75678


namespace NUMINAMATH_GPT_max_ahn_achieve_max_ahn_achieve_attained_l756_75668

def is_two_digit_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ahn_achieve :
  ∀ (n : ℕ), is_two_digit_integer n → 3 * (300 - n) ≤ 870 := 
by sorry

theorem max_ahn_achieve_attained :
  3 * (300 - 10) = 870 := 
by norm_num

end NUMINAMATH_GPT_max_ahn_achieve_max_ahn_achieve_attained_l756_75668


namespace NUMINAMATH_GPT_find_C_coordinates_l756_75620

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 11, y := 9 }
def B : Point := { x := 2, y := -3 }
def D : Point := { x := -1, y := 3 }

-- Define the isosceles property
def is_isosceles (A B C : Point) : Prop :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2) = Real.sqrt ((A.x - C.x) ^ 2 + (A.y - C.y) ^ 2)

-- Define the midpoint property
def is_midpoint (D B C : Point) : Prop :=
  D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

theorem find_C_coordinates (C : Point)
  (h_iso : is_isosceles A B C)
  (h_mid : is_midpoint D B C) :
  C = { x := -4, y := 9 } := 
  sorry

end NUMINAMATH_GPT_find_C_coordinates_l756_75620


namespace NUMINAMATH_GPT_work_completion_time_l756_75604

theorem work_completion_time (A B C D : Type) 
  (work_rate_A : ℚ := 1 / 10) 
  (work_rate_AB : ℚ := 1 / 5)
  (work_rate_C : ℚ := 1 / 15) 
  (work_rate_D : ℚ := 1 / 20) 
  (combined_work_rate_AB : work_rate_A + (work_rate_AB - work_rate_A) = 1 / 10) : 
  (1 / (work_rate_A + (work_rate_AB - work_rate_A) + work_rate_C + work_rate_D)) = 60 / 19 := 
sorry

end NUMINAMATH_GPT_work_completion_time_l756_75604


namespace NUMINAMATH_GPT_solve_part_a_solve_part_b_l756_75689

-- Part (a)
theorem solve_part_a (x : ℝ) (h1 : 36 * x^2 - 1 = (6 * x + 1) * (6 * x - 1)) :
  (3 / (1 - 6 * x) = 2 / (6 * x + 1) - (8 + 9 * x) / (36 * x^2 - 1)) ↔ x = 1 / 3 :=
sorry

-- Part (b)
theorem solve_part_b (z : ℝ) (h2 : 1 - z^2 = (1 + z) * (1 - z)) :
  (3 / (1 - z^2) = 2 / (1 + z)^2 - 5 / (1 - z)^2) ↔ z = -3 / 7 :=
sorry

end NUMINAMATH_GPT_solve_part_a_solve_part_b_l756_75689


namespace NUMINAMATH_GPT_number_of_sheep_l756_75643

-- Define the conditions as given in the problem
variables (S H : ℕ)
axiom ratio_condition : S * 7 = H * 3
axiom food_condition : H * 230 = 12880

-- The theorem to prove
theorem number_of_sheep : S = 24 :=
by sorry

end NUMINAMATH_GPT_number_of_sheep_l756_75643


namespace NUMINAMATH_GPT_divide_subtract_result_l756_75638

theorem divide_subtract_result (x : ℕ) (h : (x - 26) / 2 = 37) : 48 - (x / 4) = 23 := 
by
  sorry

end NUMINAMATH_GPT_divide_subtract_result_l756_75638


namespace NUMINAMATH_GPT_total_tickets_l756_75644

theorem total_tickets (A C total_tickets total_cost : ℕ) 
  (adult_ticket_cost : ℕ := 8) (child_ticket_cost : ℕ := 5) 
  (total_cost_paid : ℕ := 201) (child_tickets_count : ℕ := 21) 
  (ticket_cost_eqn : 8 * A + 5 * 21 = 201) 
  (adult_tickets_count : A = total_cost_paid - (child_ticket_cost * child_tickets_count) / adult_ticket_cost) :
  total_tickets = A + child_tickets_count :=
sorry

end NUMINAMATH_GPT_total_tickets_l756_75644


namespace NUMINAMATH_GPT_find_m_correct_l756_75680

noncomputable def find_m (Q : Point) (B : List Point) (m : ℝ) : Prop :=
  let circle_area := 4 * Real.pi
  let radius := 2
  let area_sector_B1B2 := Real.pi / 3
  let area_region_B1B2 := 1 / 8
  let area_triangle_B1B2 := area_sector_B1B2 - area_region_B1B2 * circle_area
  let area_sector_B4B5 := Real.pi / 3
  let area_region_B4B5 := 1 / 10
  let area_triangle_B4B5 := area_sector_B4B5 - area_region_B4B5 * circle_area
  let area_sector_B9B10 := Real.pi / 3
  let area_region_B9B10 := 4 / 15 - Real.sqrt 3 / m
  let area_triangle_B9B10 := area_sector_B9B10 - area_region_B9B10 * circle_area
  m = 3

theorem find_m_correct (Q : Point) (B : List Point) : find_m Q B 3 :=
by
  unfold find_m
  sorry

end NUMINAMATH_GPT_find_m_correct_l756_75680


namespace NUMINAMATH_GPT_simplify_expression_l756_75642

theorem simplify_expression (x y : ℤ) :
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l756_75642


namespace NUMINAMATH_GPT_division_by_power_of_ten_l756_75640

theorem division_by_power_of_ten (a b : ℕ) (h_a : a = 10^7) (h_b : b = 5 * 10^4) : a / b = 200 := by
  sorry

end NUMINAMATH_GPT_division_by_power_of_ten_l756_75640


namespace NUMINAMATH_GPT_find_x_l756_75674

theorem find_x (x : ℝ) (A B : Set ℝ) (hA : A = {1, 4, x}) (hB : B = {1, x^2}) (h_inter : A ∩ B = B) : x = -2 ∨ x = 2 ∨ x = 0 :=
sorry

end NUMINAMATH_GPT_find_x_l756_75674


namespace NUMINAMATH_GPT_contrapositive_of_odd_even_l756_75673

-- Definitions as conditions
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main statement
theorem contrapositive_of_odd_even :
  (∀ a b : ℕ, is_odd a ∧ is_odd b → is_even (a + b)) →
  (∀ a b : ℕ, ¬ is_even (a + b) → ¬ (is_odd a ∧ is_odd b)) := 
by
  intros h a b h1
  sorry

end NUMINAMATH_GPT_contrapositive_of_odd_even_l756_75673


namespace NUMINAMATH_GPT_number_of_ways_to_place_balls_l756_75651

theorem number_of_ways_to_place_balls : 
  let balls := 3 
  let boxes := 4 
  (boxes^balls = 64) :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_place_balls_l756_75651


namespace NUMINAMATH_GPT_chrom_replication_not_in_prophase_I_l756_75663

-- Definitions for the conditions
def chrom_replication (stage : String) : Prop := 
  stage = "Interphase"

def chrom_shortening_thickening (stage : String) : Prop := 
  stage = "Prophase I"

def pairing_homologous_chromosomes (stage : String) : Prop := 
  stage = "Prophase I"

def crossing_over (stage : String) : Prop :=
  stage = "Prophase I"

-- Stating the theorem
theorem chrom_replication_not_in_prophase_I :
  chrom_replication "Interphase" ∧ 
  chrom_shortening_thickening "Prophase I" ∧ 
  pairing_homologous_chromosomes "Prophase I" ∧ 
  crossing_over "Prophase I" → 
  ¬ chrom_replication "Prophase I" := 
by
  sorry

end NUMINAMATH_GPT_chrom_replication_not_in_prophase_I_l756_75663


namespace NUMINAMATH_GPT_simplify_expression_l756_75694

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 5)) + (1 / (1 - Real.sqrt 5)) = 1 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l756_75694


namespace NUMINAMATH_GPT_value_of_4_inch_cube_l756_75677

noncomputable def value_per_cubic_inch (n : ℕ) : ℝ :=
  match n with
  | 1 => 300
  | _ => 1.1 ^ (n - 1) * 300

def cube_volume (n : ℕ) : ℝ :=
  n^3

noncomputable def total_value (n : ℕ) : ℝ :=
  cube_volume n * value_per_cubic_inch n

theorem value_of_4_inch_cube : total_value 4 = 25555 := by
  admit

end NUMINAMATH_GPT_value_of_4_inch_cube_l756_75677


namespace NUMINAMATH_GPT_factor_2310_two_digit_numbers_l756_75683

theorem factor_2310_two_digit_numbers :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2310 ∧ ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c * d = 2310 → (c = a ∧ d = b) ∨ (c = b ∧ d = a) :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_2310_two_digit_numbers_l756_75683


namespace NUMINAMATH_GPT_distinguishable_triangles_count_l756_75632

def count_distinguishable_triangles (colors : ℕ) : ℕ :=
  let corner_cases := colors + (colors * (colors - 1)) + (colors * (colors - 1) * (colors - 2) / 6)
  let edge_cases := colors * colors
  let center_cases := colors
  corner_cases * edge_cases * center_cases

theorem distinguishable_triangles_count :
  count_distinguishable_triangles 8 = 61440 :=
by
  unfold count_distinguishable_triangles
  -- corner_cases = 8 + 8 * 7 + (8 * 7 * 6) / 6 = 120
  -- edge_cases = 8 * 8 = 64
  -- center_cases = 8
  -- Total = 120 * 64 * 8 = 61440
  sorry

end NUMINAMATH_GPT_distinguishable_triangles_count_l756_75632


namespace NUMINAMATH_GPT_union_of_sets_l756_75613

def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }

theorem union_of_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x : ℝ | 2 < x ∧ x < 10 }) :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l756_75613


namespace NUMINAMATH_GPT_solve_quartic_equation_l756_75631

theorem solve_quartic_equation (a b c : ℤ) (x : ℤ) : 
  x^4 + a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_solve_quartic_equation_l756_75631


namespace NUMINAMATH_GPT_fewer_pushups_l756_75657

theorem fewer_pushups (sets: ℕ) (pushups_per_set : ℕ) (total_pushups : ℕ) 
  (h1 : sets = 3) (h2 : pushups_per_set = 15) (h3 : total_pushups = 40) :
  sets * pushups_per_set - total_pushups = 5 :=
by
  sorry

end NUMINAMATH_GPT_fewer_pushups_l756_75657


namespace NUMINAMATH_GPT_find_numbers_between_1000_and_4000_l756_75672

theorem find_numbers_between_1000_and_4000 :
  ∃ (x : ℤ), 1000 ≤ x ∧ x ≤ 4000 ∧
             (x % 11 = 2) ∧
             (x % 13 = 12) ∧
             (x % 19 = 18) ∧
             (x = 1234 ∨ x = 3951) :=
sorry

end NUMINAMATH_GPT_find_numbers_between_1000_and_4000_l756_75672


namespace NUMINAMATH_GPT_algebraic_expression_for_A_l756_75648

variable {x y A : ℝ}

theorem algebraic_expression_for_A
  (h : (3 * x + 2 * y) ^ 2 = (3 * x - 2 * y) ^ 2 + A) :
  A = 24 * x * y :=
sorry

end NUMINAMATH_GPT_algebraic_expression_for_A_l756_75648


namespace NUMINAMATH_GPT_number_of_organizations_in_foundation_l756_75621

def company_raised : ℕ := 2500
def donation_percentage : ℕ := 80
def each_organization_receives : ℕ := 250
def total_donated : ℕ := (donation_percentage * company_raised) / 100

theorem number_of_organizations_in_foundation : total_donated / each_organization_receives = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_organizations_in_foundation_l756_75621


namespace NUMINAMATH_GPT_complex_division_simplification_l756_75676

theorem complex_division_simplification (i : ℂ) (h_i : i * i = -1) : (1 - 3 * i) / (2 - i) = 1 - i := by
  sorry

end NUMINAMATH_GPT_complex_division_simplification_l756_75676


namespace NUMINAMATH_GPT_solve_inequality_l756_75633

theorem solve_inequality (x : ℝ) : 2 * x^2 + 8 * x ≤ -6 ↔ -3 ≤ x ∧ x ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l756_75633


namespace NUMINAMATH_GPT_number_of_pairs_lcm_600_l756_75687

theorem number_of_pairs_lcm_600 :
  ∃ n, n = 53 ∧ (∀ m n : ℕ, (m ≤ n ∧ m > 0 ∧ n > 0 ∧ Nat.lcm m n = 600) ↔ n = 53) := sorry

end NUMINAMATH_GPT_number_of_pairs_lcm_600_l756_75687
