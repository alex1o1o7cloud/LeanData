import Mathlib

namespace NUMINAMATH_GPT_constant_in_price_equation_l857_85777

theorem constant_in_price_equation (x y: ℕ) (h: y = 70 * x) : ∃ c, ∀ (x: ℕ), y = c * x ∧ c = 70 :=
  sorry

end NUMINAMATH_GPT_constant_in_price_equation_l857_85777


namespace NUMINAMATH_GPT_sum_of_values_of_z_l857_85725

def f (x : ℝ) := x^2 - 2*x + 3

theorem sum_of_values_of_z (z : ℝ) (h : f (5 * z) = 7) : z = 2 / 25 :=
sorry

end NUMINAMATH_GPT_sum_of_values_of_z_l857_85725


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_inequalities_l857_85764

theorem necessary_but_not_sufficient_for_inequalities (a b : ℝ) :
  (a + b > 4) ↔ (a > 2 ∧ b > 2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_inequalities_l857_85764


namespace NUMINAMATH_GPT_find_integer_x_l857_85779

theorem find_integer_x (x : ℕ) (pos_x : 0 < x) (ineq : x + 1000 > 1000 * x) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_integer_x_l857_85779


namespace NUMINAMATH_GPT_positive_integer_base_conversion_l857_85740

theorem positive_integer_base_conversion (A B : ℕ) (h1 : A < 9) (h2 : B < 7) 
(h3 : 9 * A + B = 7 * B + A) : 9 * 3 + 4 = 31 :=
by sorry

end NUMINAMATH_GPT_positive_integer_base_conversion_l857_85740


namespace NUMINAMATH_GPT_third_place_prize_is_120_l857_85715

noncomputable def prize_for_third_place (total_prize : ℕ) (first_place_prize : ℕ) (second_place_prize : ℕ) (prize_per_novel : ℕ) (num_novels_receiving_prize : ℕ) : ℕ :=
  let remaining_prize := total_prize - first_place_prize - second_place_prize
  let total_other_prizes := num_novels_receiving_prize * prize_per_novel
  remaining_prize - total_other_prizes

theorem third_place_prize_is_120 : prize_for_third_place 800 200 150 22 15 = 120 := by
  sorry

end NUMINAMATH_GPT_third_place_prize_is_120_l857_85715


namespace NUMINAMATH_GPT_kyle_caught_14_fish_l857_85792

theorem kyle_caught_14_fish (K T C : ℕ) (h1 : K = T) (h2 : C = 8) (h3 : C + K + T = 36) : K = 14 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_kyle_caught_14_fish_l857_85792


namespace NUMINAMATH_GPT_probability_mixed_doubles_l857_85746

def num_athletes : ℕ := 6
def num_males : ℕ := 3
def num_females : ℕ := 3
def num_coaches : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select athletes
def total_ways : ℕ :=
  (choose num_athletes 2) * (choose (num_athletes - 2) 2) * (choose (num_athletes - 4) 2)

-- Number of favorable ways to select mixed doubles teams
def favorable_ways : ℕ :=
  (choose num_males 1) * (choose num_females 1) *
  (choose (num_males - 1) 1) * (choose (num_females - 1) 1) *
  (choose 1 1) * (choose 1 1)

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

theorem probability_mixed_doubles :
  probability = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_probability_mixed_doubles_l857_85746


namespace NUMINAMATH_GPT_parallelogram_area_l857_85711

theorem parallelogram_area (base height : ℕ) (h_base : base = 5) (h_height : height = 3) :
  base * height = 15 :=
by
  -- Here would be the proof, but it is omitted per instructions
  sorry

end NUMINAMATH_GPT_parallelogram_area_l857_85711


namespace NUMINAMATH_GPT_ratio_almonds_to_walnuts_l857_85762

theorem ratio_almonds_to_walnuts (almonds walnuts mixture : ℝ) 
  (h1 : almonds = 116.67)
  (h2 : mixture = 140)
  (h3 : walnuts = mixture - almonds) : 
  (almonds / walnuts) = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_almonds_to_walnuts_l857_85762


namespace NUMINAMATH_GPT_angle_XYZ_of_excircle_circumcircle_incircle_l857_85705

theorem angle_XYZ_of_excircle_circumcircle_incircle 
  (a b c x y z : ℝ) 
  (hA : a = 50)
  (hB : b = 70)
  (hC : c = 60) 
  (triangleABC : a + b + c = 180) 
  (excircle_Omega : Prop) 
  (incircle_Gamma : Prop) 
  (circumcircle_Omega_triangleXYZ : Prop) 
  (X_on_BC : Prop)
  (Y_on_AB : Prop) 
  (Z_on_CA : Prop): 
  x = 115 := 
by 
  sorry

end NUMINAMATH_GPT_angle_XYZ_of_excircle_circumcircle_incircle_l857_85705


namespace NUMINAMATH_GPT_scaled_multiplication_l857_85730

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end NUMINAMATH_GPT_scaled_multiplication_l857_85730


namespace NUMINAMATH_GPT_clara_sells_total_cookies_l857_85745

theorem clara_sells_total_cookies :
  let cookies_per_box_1 := 12
  let cookies_per_box_2 := 20
  let cookies_per_box_3 := 16
  let cookies_per_box_4 := 18
  let cookies_per_box_5 := 22

  let boxes_sold_1 := 50.5
  let boxes_sold_2 := 80.25
  let boxes_sold_3 := 70.75
  let boxes_sold_4 := 65.5
  let boxes_sold_5 := 55.25

  let total_cookies_1 := cookies_per_box_1 * boxes_sold_1
  let total_cookies_2 := cookies_per_box_2 * boxes_sold_2
  let total_cookies_3 := cookies_per_box_3 * boxes_sold_3
  let total_cookies_4 := cookies_per_box_4 * boxes_sold_4
  let total_cookies_5 := cookies_per_box_5 * boxes_sold_5

  let total_cookies := total_cookies_1 + total_cookies_2 + total_cookies_3 + total_cookies_4 + total_cookies_5

  total_cookies = 5737.5 :=
by
  sorry

end NUMINAMATH_GPT_clara_sells_total_cookies_l857_85745


namespace NUMINAMATH_GPT_max_integer_solutions_l857_85729

def quad_func (x : ℝ) : ℝ := x^2 - 6 * x + 1

theorem max_integer_solutions (p : ℝ → ℝ) : 
  (p = quad_func) →
  (∃ n1 n2 n3 n4 : ℤ, 
    ((p n1 = p (n1 ^ 2)) ∧ (p n2 = p (n2 ^ 2)) ∧ 
    (p n3 = p (n3 ^ 2)) ∧ (p n4 = p (n4 ^ 2))) ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ 
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ 
    n3 ≠ n4) :=
by
  sorry

end NUMINAMATH_GPT_max_integer_solutions_l857_85729


namespace NUMINAMATH_GPT_proof_problem_l857_85721

theorem proof_problem (a b : ℝ) (h : a^2 + b^2 + 2*a - 4*b + 5 = 0) : 2*a^2 + 4*b - 3 = 7 :=
sorry

end NUMINAMATH_GPT_proof_problem_l857_85721


namespace NUMINAMATH_GPT_remaining_two_by_two_square_exists_l857_85759

theorem remaining_two_by_two_square_exists (grid_size : ℕ) (cut_squares : ℕ) : grid_size = 29 → cut_squares = 99 → 
  ∃ remaining_square : ℕ, remaining_square = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_two_by_two_square_exists_l857_85759


namespace NUMINAMATH_GPT_max_pN_value_l857_85737

noncomputable def max_probability_units_digit (N: ℕ) (q2 q5 q10: ℚ) : ℚ :=
  let qk (k : ℕ) := (Nat.floor (N / k) : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_pN_value : ∃ (a b : ℕ), (a.gcd b = 1) ∧ (∀ N q2 q5 q10, max_probability_units_digit N q2 q5 q10 ≤  27 / 100) ∧ (100 * 27 + 100 = 2800) :=
by
  sorry

end NUMINAMATH_GPT_max_pN_value_l857_85737


namespace NUMINAMATH_GPT_total_trip_time_l857_85708

noncomputable def speed_coastal := 10 / 20  -- miles per minute
noncomputable def speed_highway := 4 * speed_coastal  -- miles per minute
noncomputable def time_highway := 50 / speed_highway  -- minutes
noncomputable def total_time := 20 + time_highway  -- minutes

theorem total_trip_time : total_time = 45 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_trip_time_l857_85708


namespace NUMINAMATH_GPT_correct_survey_option_l857_85781

-- Definitions for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Predicate that checks if an option is suitable for a comprehensive survey method
def suitable_for_comprehensive_survey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => false
  | SurveyOption.B => false
  | SurveyOption.C => false
  | SurveyOption.D => true

-- Theorem statement
theorem correct_survey_option : suitable_for_comprehensive_survey SurveyOption.D := 
  by sorry

end NUMINAMATH_GPT_correct_survey_option_l857_85781


namespace NUMINAMATH_GPT_min_sum_length_perpendicular_chords_l857_85776

variables {p : ℝ} (h : p > 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * p * (x + p)

theorem min_sum_length_perpendicular_chords (h: p > 0) :
  ∃ (AB CD : ℝ), AB * CD = 1 → |AB| + |CD| = 16 * p := sorry

end NUMINAMATH_GPT_min_sum_length_perpendicular_chords_l857_85776


namespace NUMINAMATH_GPT_vertical_line_divides_triangle_equal_area_l857_85738

theorem vertical_line_divides_triangle_equal_area :
  let A : (ℝ × ℝ) := (1, 2)
  let B : (ℝ × ℝ) := (1, 1)
  let C : (ℝ × ℝ) := (10, 1)
  let area_ABC := (1 / 2 : ℝ) * (C.1 - A.1) * (A.2 - B.2)
  let a : ℝ := 5.5
  let area_left_triangle := (1 / 2 : ℝ) * (a - A.1) * (A.2 - B.2)
  let area_right_triangle := (1 / 2 : ℝ) * (C.1 - a) * (A.2 - B.2)
  area_left_triangle = area_right_triangle :=
by
  sorry

end NUMINAMATH_GPT_vertical_line_divides_triangle_equal_area_l857_85738


namespace NUMINAMATH_GPT_perimeter_of_ghost_l857_85748
open Real

def radius := 2
def angle_degrees := 90
def full_circle_degrees := 360

noncomputable def missing_angle := angle_degrees
noncomputable def remaining_angle := full_circle_degrees - missing_angle
noncomputable def fraction_of_circle := remaining_angle / full_circle_degrees
noncomputable def full_circumference := 2 * π * radius
noncomputable def arc_length := fraction_of_circle * full_circumference
noncomputable def radii_length := 2 * radius

theorem perimeter_of_ghost : arc_length + radii_length = 3 * π + 4 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_ghost_l857_85748


namespace NUMINAMATH_GPT_nancy_earns_more_l857_85735

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nancy_earns_more_l857_85735


namespace NUMINAMATH_GPT_compare_expressions_l857_85794

theorem compare_expressions (n : ℕ) (hn : 0 < n):
  (n ≤ 48 ∧ 99^n + 100^n > 101^n) ∨ (n > 48 ∧ 99^n + 100^n < 101^n) :=
sorry  -- Proof is omitted.

end NUMINAMATH_GPT_compare_expressions_l857_85794


namespace NUMINAMATH_GPT_eq_a2b2_of_given_condition_l857_85778

theorem eq_a2b2_of_given_condition (a b : ℝ) (h : a^4 + b^4 = a^2 - 2 * a^2 * b^2 + b^2 + 6) : a^2 + b^2 = 3 :=
sorry

end NUMINAMATH_GPT_eq_a2b2_of_given_condition_l857_85778


namespace NUMINAMATH_GPT_sum_of_common_ratios_l857_85785

theorem sum_of_common_ratios (k p r : ℝ) (h : k ≠ 0) (h1 : k * p ≠ k * r)
  (h2 : k * p ^ 2 - k * r ^ 2 = 3 * (k * p - k * r)) : p + r = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l857_85785


namespace NUMINAMATH_GPT_find_certain_number_l857_85717

theorem find_certain_number (mystery_number certain_number : ℕ) (h1 : mystery_number = 47) 
(h2 : mystery_number + certain_number = 92) : certain_number = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l857_85717


namespace NUMINAMATH_GPT_smallest_value_of_expression_l857_85718

theorem smallest_value_of_expression :
  ∃ (k l : ℕ), 36^k - 5^l = 11 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_expression_l857_85718


namespace NUMINAMATH_GPT_total_lobster_pounds_l857_85739

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end NUMINAMATH_GPT_total_lobster_pounds_l857_85739


namespace NUMINAMATH_GPT_solve_the_problem_l857_85783

noncomputable def solve_problem : Prop :=
  ∀ (θ t α : ℝ),
    (∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = 4 * Real.sin θ) → 
    (∃ x y : ℝ, x = 1 + t * Real.cos α ∧ y = 2 + t * Real.sin α) →
    (∃ m n : ℝ, m = 1 ∧ n = 2) →
    (-2 = Real.tan α)

theorem solve_the_problem : solve_problem := by
  sorry

end NUMINAMATH_GPT_solve_the_problem_l857_85783


namespace NUMINAMATH_GPT_range_of_m_l857_85706

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (-2 < x ∧ x ≤ 2) → x ≤ m) → m ≥ 2 :=
by
  intro h
  -- insert necessary proof steps here
  sorry

end NUMINAMATH_GPT_range_of_m_l857_85706


namespace NUMINAMATH_GPT_least_possible_value_of_s_l857_85786

theorem least_possible_value_of_s (a b : ℤ) 
(h : a^3 + b^3 - 60 * a * b * (a + b) ≥ 2012) : 
∃ a b, a^3 + b^3 - 60 * a * b * (a + b) = 2015 :=
by sorry

end NUMINAMATH_GPT_least_possible_value_of_s_l857_85786


namespace NUMINAMATH_GPT_calculate_subtraction_l857_85724

def base9_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

def base6_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

theorem calculate_subtraction : base9_to_base10 324 - base6_to_base10 231 = 174 :=
  by sorry

end NUMINAMATH_GPT_calculate_subtraction_l857_85724


namespace NUMINAMATH_GPT_find_speed_l857_85771

variable (d : ℝ) (t : ℝ)
variable (h1 : d = 50 * (t + 1/12))
variable (h2 : d = 70 * (t - 1/12))

theorem find_speed (d t : ℝ)
  (h1 : d = 50 * (t + 1/12))
  (h2 : d = 70 * (t - 1/12)) :
  58 = d / t := by
  sorry

end NUMINAMATH_GPT_find_speed_l857_85771


namespace NUMINAMATH_GPT_complement_intersection_l857_85719

open Set

namespace UniversalSetProof

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {4, 5} :=
by
  sorry

end UniversalSetProof

end NUMINAMATH_GPT_complement_intersection_l857_85719


namespace NUMINAMATH_GPT_children_neither_happy_nor_sad_l857_85798

-- conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10

-- proof problem
theorem children_neither_happy_nor_sad :
  total_children - happy_children - sad_children = 20 := by
  sorry

end NUMINAMATH_GPT_children_neither_happy_nor_sad_l857_85798


namespace NUMINAMATH_GPT_flower_profit_equation_l857_85722

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end NUMINAMATH_GPT_flower_profit_equation_l857_85722


namespace NUMINAMATH_GPT_evaluate_expression_l857_85712

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l857_85712


namespace NUMINAMATH_GPT_least_number_to_add_l857_85713

theorem least_number_to_add (n divisor : ℕ) (h₁ : n = 27306) (h₂ : divisor = 151) : 
  ∃ k : ℕ, k = 25 ∧ (n + k) % divisor = 0 := 
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l857_85713


namespace NUMINAMATH_GPT_derivative_odd_function_l857_85752

theorem derivative_odd_function (a b c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^3 + b * x^2 + c * x + 2) 
    (h_deriv_odd : ∀ x, deriv f (-x) = - deriv f x) : a^2 + c^2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_derivative_odd_function_l857_85752


namespace NUMINAMATH_GPT_fourth_term_geometric_sequence_l857_85701

theorem fourth_term_geometric_sequence (x : ℝ) :
  ∃ r : ℝ, (r > 0) ∧ 
  x ≠ 0 ∧
  (3 * x + 3)^2 = x * (6 * x + 6) →
  x = -3 →
  6 * x + 6 ≠ 0 →
  4 * (6 * x + 6) * (3 * x + 3) = -24 :=
by
  -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_fourth_term_geometric_sequence_l857_85701


namespace NUMINAMATH_GPT_sofa_love_seat_ratio_l857_85703

theorem sofa_love_seat_ratio (L S: ℕ) (h1: L = 148) (h2: S + L = 444): S = 2 * L := by
  sorry

end NUMINAMATH_GPT_sofa_love_seat_ratio_l857_85703


namespace NUMINAMATH_GPT_sphere_radius_l857_85767

theorem sphere_radius (tree_height sphere_shadow tree_shadow : ℝ) 
  (h_tree_shadow_pos : tree_shadow > 0) 
  (h_sphere_shadow_pos : sphere_shadow > 0) 
  (h_tree_height_pos : tree_height > 0)
  (h_tangent : (tree_height / tree_shadow) = (sphere_shadow / 15)) : 
  sphere_shadow = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_l857_85767


namespace NUMINAMATH_GPT_forester_trees_planted_l857_85766

theorem forester_trees_planted (initial_trees : ℕ) (tripled_trees : ℕ) (trees_planted_monday : ℕ) (trees_planted_tuesday : ℕ) :
  initial_trees = 30 ∧ tripled_trees = 3 * initial_trees ∧ trees_planted_monday = tripled_trees - initial_trees ∧ trees_planted_tuesday = trees_planted_monday / 3 →
  trees_planted_monday + trees_planted_tuesday = 80 :=
by
  sorry

end NUMINAMATH_GPT_forester_trees_planted_l857_85766


namespace NUMINAMATH_GPT_smallest_value_of_expression_l857_85787

theorem smallest_value_of_expression (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 - b^2 = 16) : 
  (∃ k : ℚ, k = (a + b) / (a - b) + (a - b) / (a + b) ∧ (∀ x : ℚ, x = (a + b) / (a - b) + (a - b) / (a + b) → x ≥ 9/4)) :=
sorry

end NUMINAMATH_GPT_smallest_value_of_expression_l857_85787


namespace NUMINAMATH_GPT_sandbox_length_l857_85731

theorem sandbox_length (width : ℕ) (area : ℕ) (h_width : width = 146) (h_area : area = 45552) : ∃ length : ℕ, length = 312 :=
by {
  sorry
}

end NUMINAMATH_GPT_sandbox_length_l857_85731


namespace NUMINAMATH_GPT_area_of_equilateral_triangle_l857_85743

theorem area_of_equilateral_triangle
  (A B C D E : Type) 
  (side_length : ℝ) 
  (medians_perpendicular : Prop) 
  (BD CE : ℝ)
  (inscribed_circle : Prop)
  (equilateral_triangle : A = B ∧ B = C) 
  (s : side_length = 18) 
  (BD_len : BD = 15) 
  (CE_len : CE = 9) 
  : ∃ area, area = 81 * Real.sqrt 3
  :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_equilateral_triangle_l857_85743


namespace NUMINAMATH_GPT_AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l857_85799

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ)

-- Condition
def AB2A_eq_AB := A * B ^ 2 * A = A * B * A

-- Part (a): Prove that (AB)^2 = AB
theorem AB_squared_eq_AB (h : AB2A_eq_AB A B) : (A * B) ^ 2 = A * B :=
sorry

-- Part (b): Prove that (AB - BA)^3 = 0
theorem AB_minus_BA_cubed_eq_zero (h : AB2A_eq_AB A B) : (A * B - B * A) ^ 3 = 0 :=
sorry

end NUMINAMATH_GPT_AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l857_85799


namespace NUMINAMATH_GPT_parabola_coefficients_sum_l857_85710

theorem parabola_coefficients_sum (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = a * (x + 3)^2 + 2) ∧
  (-6 = a * (1 + 3)^2 + 2) →
  a + b + c = -11/2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_coefficients_sum_l857_85710


namespace NUMINAMATH_GPT_find_salary_l857_85756

-- Define the conditions
variables (S : ℝ) -- S is the man's monthly salary

def saves_25_percent (S : ℝ) : ℝ := 0.25 * S
def expenses (S : ℝ) : ℝ := 0.75 * S
def increased_expenses (S : ℝ) : ℝ := 0.75 * S + 0.10 * (0.75 * S)
def monthly_savings_after_increase (S : ℝ) : ℝ := S - increased_expenses S

-- Define the problem statement
theorem find_salary
  (h1 : saves_25_percent S = 0.25 * S)
  (h2 : increased_expenses S = 0.825 * S)
  (h3 : monthly_savings_after_increase S = 175) :
  S = 1000 :=
sorry

end NUMINAMATH_GPT_find_salary_l857_85756


namespace NUMINAMATH_GPT_find_a_l857_85780

theorem find_a (a : ℤ) : 
  (∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^3)) ↔ (a = 3^9) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l857_85780


namespace NUMINAMATH_GPT_determine_c_for_inverse_l857_85760

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_c_for_inverse :
  (∀ x : ℝ, x ≠ 0 → f (f_inv x) c = x) ↔ c = 1 :=
sorry

end NUMINAMATH_GPT_determine_c_for_inverse_l857_85760


namespace NUMINAMATH_GPT_amount_of_medication_B_l857_85793

def medicationAmounts (x y : ℝ) : Prop :=
  (x + y = 750) ∧ (0.40 * x + 0.20 * y = 215)

theorem amount_of_medication_B (x y : ℝ) (h : medicationAmounts x y) : y = 425 :=
  sorry

end NUMINAMATH_GPT_amount_of_medication_B_l857_85793


namespace NUMINAMATH_GPT_length_AB_l857_85726

theorem length_AB (r : ℝ) (A B : ℝ) (π : ℝ) : 
  r = 4 ∧ π = 3 ∧ (A = 8 ∧ B = 8) → (A = B ∧ A + B = 24 → AB = 6) :=
by
  intros
  sorry

end NUMINAMATH_GPT_length_AB_l857_85726


namespace NUMINAMATH_GPT_find_years_invested_l857_85733

-- Defining the conditions and theorem
variables (P : ℕ) (r1 r2 D : ℝ) (n : ℝ)

-- Given conditions
def principal := (P : ℝ) = 7000
def rate_1 := r1 = 0.15
def rate_2 := r2 = 0.12
def interest_diff := D = 420

-- Theorem to be proven
theorem find_years_invested (h1 : principal P) (h2 : rate_1 r1) (h3 : rate_2 r2) (h4 : interest_diff D) :
  7000 * 0.15 * n - 7000 * 0.12 * n = 420 → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_years_invested_l857_85733


namespace NUMINAMATH_GPT_non_science_majors_percentage_l857_85791

-- Definitions of conditions
def women_percentage (class_size : ℝ) : ℝ := 0.6 * class_size
def men_percentage (class_size : ℝ) : ℝ := 0.4 * class_size

def women_science_majors (class_size : ℝ) : ℝ := 0.2 * women_percentage class_size
def men_science_majors (class_size : ℝ) : ℝ := 0.7 * men_percentage class_size

def total_science_majors (class_size : ℝ) : ℝ := women_science_majors class_size + men_science_majors class_size

-- Theorem to prove the percentage of the class that are non-science majors is 60%
theorem non_science_majors_percentage (class_size : ℝ) : total_science_majors class_size / class_size = 0.4 → (class_size - total_science_majors class_size) / class_size = 0.6 := 
by
  sorry

end NUMINAMATH_GPT_non_science_majors_percentage_l857_85791


namespace NUMINAMATH_GPT_johns_donation_l857_85768

theorem johns_donation
    (A T D : ℝ)
    (n : ℕ)
    (hA1 : A * 1.75 = 100)
    (hA2 : A = 100 / 1.75)
    (hT : T = 10 * A)
    (hD : D = 11 * 100 - T)
    (hn : n = 10) :
    D = 3700 / 7 := 
sorry

end NUMINAMATH_GPT_johns_donation_l857_85768


namespace NUMINAMATH_GPT_solve_for_phi_l857_85769

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ)

theorem solve_for_phi (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π / 2)
    (h_min_diff : |x1 - x2| = π / 6)
    (h_condition : |f x1 - g x2 φ| = 4) :
    φ = π / 3 := 
    sorry

end NUMINAMATH_GPT_solve_for_phi_l857_85769


namespace NUMINAMATH_GPT_heights_equal_l857_85775

-- Define base areas and volumes
variables {V : ℝ} {S : ℝ}

-- Assume equal volumes and base areas for the prism and cylinder
variables (h_prism h_cylinder : ℝ) (volume_eq : V = S * h_prism) (base_area_eq : S = S)

-- Define a proof goal
theorem heights_equal 
  (equal_volumes : V = S * h_prism) 
  (equal_base_areas : S = S) : 
  h_prism = h_cylinder :=
sorry

end NUMINAMATH_GPT_heights_equal_l857_85775


namespace NUMINAMATH_GPT_triangle_area_l857_85716

theorem triangle_area
  (area_WXYZ : ℝ)
  (side_small_squares : ℝ)
  (AB_eq_AC : ℝ)
  (A_coincides_with_O : ℝ)
  (area : ℝ) :
  area_WXYZ = 49 →  -- The area of square WXYZ is 49 cm^2
  side_small_squares = 2 → -- Sides of the smaller squares are 2 cm long
  AB_eq_AC = AB_eq_AC → -- Triangle ABC is isosceles with AB = AC
  A_coincides_with_O = A_coincides_with_O → -- A coincides with O
  area = 45 / 4 := -- The area of triangle ABC is 45/4 cm^2
by
  sorry

end NUMINAMATH_GPT_triangle_area_l857_85716


namespace NUMINAMATH_GPT_first_caller_to_win_all_prizes_is_900_l857_85761

-- Define the conditions: frequencies of win types
def every_25th_caller_wins_music_player (n : ℕ) : Prop := n % 25 = 0
def every_36th_caller_wins_concert_tickets (n : ℕ) : Prop := n % 36 = 0
def every_45th_caller_wins_backstage_passes (n : ℕ) : Prop := n % 45 = 0

-- Formalize the problem to prove
theorem first_caller_to_win_all_prizes_is_900 :
  ∃ n : ℕ, every_25th_caller_wins_music_player n ∧
           every_36th_caller_wins_concert_tickets n ∧
           every_45th_caller_wins_backstage_passes n ∧
           n = 900 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_caller_to_win_all_prizes_is_900_l857_85761


namespace NUMINAMATH_GPT_complex_power_difference_l857_85789

theorem complex_power_difference (i : ℂ) (hi : i^2 = -1) : (1 + 2 * i)^8 - (1 - 2 * i)^8 = 672 * i := 
by
  sorry

end NUMINAMATH_GPT_complex_power_difference_l857_85789


namespace NUMINAMATH_GPT_evaluate_expression_l857_85702

theorem evaluate_expression :
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l857_85702


namespace NUMINAMATH_GPT_sum_area_triangles_lt_total_area_l857_85728

noncomputable def G : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def A_k (k : ℕ+) : ℝ := sorry -- Assume we've defined A_k's expression correctly
noncomputable def S (S1 S2 : ℝ) : ℝ := 2 * S1 - S2

theorem sum_area_triangles_lt_total_area (k : ℕ+) (S1 S2 : ℝ) :
  (A_k k < S S1 S2) :=
sorry

end NUMINAMATH_GPT_sum_area_triangles_lt_total_area_l857_85728


namespace NUMINAMATH_GPT_ways_to_select_four_doctors_l857_85707

def num_ways_to_select_doctors (num_internists : ℕ) (num_surgeons : ℕ) (team_size : ℕ) : ℕ :=
  (Nat.choose num_internists 1 * Nat.choose num_surgeons (team_size - 1)) + 
  (Nat.choose num_internists 2 * Nat.choose num_surgeons (team_size - 2)) + 
  (Nat.choose num_internists 3 * Nat.choose num_surgeons (team_size - 3))

theorem ways_to_select_four_doctors : num_ways_to_select_doctors 5 6 4 = 310 := 
by
  sorry

end NUMINAMATH_GPT_ways_to_select_four_doctors_l857_85707


namespace NUMINAMATH_GPT_necessary_condition_abs_sq_necessary_and_sufficient_add_l857_85736

theorem necessary_condition_abs_sq (a b : ℝ) : a^2 > b^2 → |a| > |b| :=
sorry

theorem necessary_and_sufficient_add (a b c : ℝ) :
  (a > b) ↔ (a + c > b + c) :=
sorry

end NUMINAMATH_GPT_necessary_condition_abs_sq_necessary_and_sufficient_add_l857_85736


namespace NUMINAMATH_GPT_circle_line_intersection_points_l857_85749

theorem circle_line_intersection_points :
  let circle_eqn : ℝ × ℝ → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 16
  let line_eqn  : ℝ × ℝ → Prop := fun p => p.1 = 4
  ∃ (p₁ p₂ : ℝ × ℝ), 
    circle_eqn p₁ ∧ line_eqn p₁ ∧ circle_eqn p₂ ∧ line_eqn p₂ ∧ p₁ ≠ p₂ 
      → ∀ (p : ℝ × ℝ), circle_eqn p ∧ line_eqn p → 
        p = p₁ ∨ p = p₂ ∧ (p₁ ≠ p ∨ p₂ ≠ p)
 := sorry

end NUMINAMATH_GPT_circle_line_intersection_points_l857_85749


namespace NUMINAMATH_GPT_problem_1_problem_2_l857_85709

theorem problem_1 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2) → (a = 0 ∨ a = 1) :=
by sorry

theorem problem_2 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2 ∨ ¬ ∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a ≥ 1 ∨ a = 0) :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l857_85709


namespace NUMINAMATH_GPT_solutions_of_quadratic_eq_l857_85797

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solutions_of_quadratic_eq_l857_85797


namespace NUMINAMATH_GPT_tangent_y_intercept_l857_85763

theorem tangent_y_intercept :
  let C1 := (2, 4)
  let r1 := 5
  let C2 := (14, 9)
  let r2 := 10
  let m := 120 / 119
  m > 0 → ∃ b, b = 912 / 119 := by
  sorry

end NUMINAMATH_GPT_tangent_y_intercept_l857_85763


namespace NUMINAMATH_GPT_product_of_integers_l857_85758

theorem product_of_integers (a b : ℤ) (h1 : Int.gcd a b = 12) (h2 : Int.lcm a b = 60) : a * b = 720 :=
sorry

end NUMINAMATH_GPT_product_of_integers_l857_85758


namespace NUMINAMATH_GPT_domain_of_g_l857_85755

def f : ℝ → ℝ := sorry

theorem domain_of_g 
  (hf_dom : ∀ x, -2 ≤ x ∧ x ≤ 4 → f x = f x) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 ↔ (∃ y, y = f x + f (-x)) := 
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_g_l857_85755


namespace NUMINAMATH_GPT_ratio_of_volumes_l857_85727

noncomputable def inscribedSphereVolume (s : ℝ) : ℝ := (4 / 3) * Real.pi * (s / 2) ^ 3

noncomputable def cubeVolume (s : ℝ) : ℝ := s ^ 3

theorem ratio_of_volumes (s : ℝ) (h : s > 0) :
  inscribedSphereVolume s / cubeVolume s = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l857_85727


namespace NUMINAMATH_GPT_cosine_theorem_l857_85757

theorem cosine_theorem (a b c : ℝ) (A : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

end NUMINAMATH_GPT_cosine_theorem_l857_85757


namespace NUMINAMATH_GPT_find_f_sqrt2_l857_85790

theorem find_f_sqrt2 (f : ℝ → ℝ)
  (hf : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x * y) = f x + f y)
  (hf8 : f 8 = 3) :
  f (Real.sqrt 2) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_f_sqrt2_l857_85790


namespace NUMINAMATH_GPT_fourth_root_eq_solution_l857_85750

theorem fourth_root_eq_solution (x : ℝ) (h : Real.sqrt (Real.sqrt x) = 16 / (8 - Real.sqrt (Real.sqrt x))) : x = 256 := by
  sorry

end NUMINAMATH_GPT_fourth_root_eq_solution_l857_85750


namespace NUMINAMATH_GPT_range_of_m_l857_85774

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x > m
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * m * x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ m ∈ Set.Ioo (-2:ℝ) (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l857_85774


namespace NUMINAMATH_GPT_circle_center_l857_85723

theorem circle_center (x y : ℝ) :
  x^2 + 4 * x + y^2 - 6 * y + 1 = 0 → (x + 2, y - 3) = (0, 0) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_l857_85723


namespace NUMINAMATH_GPT_gas_pressure_inversely_proportional_l857_85795

theorem gas_pressure_inversely_proportional :
  ∀ (p v : ℝ), (p * v = 27.2) → (8 * 3.4 = 27.2) → (v = 6.8) → p = 4 :=
by
  intros p v h1 h2 h3
  have h4 : 27.2 = 8 * 3.4 := by sorry
  have h5 : p * 6.8 = 27.2 := by sorry
  exact sorry

end NUMINAMATH_GPT_gas_pressure_inversely_proportional_l857_85795


namespace NUMINAMATH_GPT_mono_intervals_range_of_a_l857_85741

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.exp (x - 1)

theorem mono_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x, f x a > 0) ∧ 
  (a > 0 → (∀ x, x < 1 - Real.log a → f x a > 0) ∧ (∀ x, x > 1 - Real.log a → f x a < 0)) :=
sorry

theorem range_of_a (h : ∀ x, f x a ≤ 0) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_mono_intervals_range_of_a_l857_85741


namespace NUMINAMATH_GPT_range_of_a_l857_85732

variable {x a : ℝ}

theorem range_of_a (h1 : 2 * x - a < 0)
                   (h2 : 1 - 2 * x ≥ 7)
                   (h3 : ∀ x, x ≤ -3) : ∀ a, a > -6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l857_85732


namespace NUMINAMATH_GPT_f_1986_l857_85788

noncomputable def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 3 * f (a * b)
axiom f_1 : f 1 = 2

theorem f_1986 : f 1986 = 2 :=
by
  sorry

end NUMINAMATH_GPT_f_1986_l857_85788


namespace NUMINAMATH_GPT_prob_exactly_two_trains_on_time_is_0_398_l857_85765

-- Definitions and conditions
def eventA := true
def eventB := true
def eventC := true

def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B
def P_not_C : ℝ := 1 - P_C

-- Question definition (to be proved)
def exact_two_on_time : ℝ :=
  P_A * P_B * P_not_C + P_A * P_not_B * P_C + P_not_A * P_B * P_C

-- Theorem statement
theorem prob_exactly_two_trains_on_time_is_0_398 :
  exact_two_on_time = 0.398 := sorry

end NUMINAMATH_GPT_prob_exactly_two_trains_on_time_is_0_398_l857_85765


namespace NUMINAMATH_GPT_number_of_books_from_second_shop_l857_85700

theorem number_of_books_from_second_shop (books_first_shop : ℕ) (cost_first_shop : ℕ)
    (books_second_shop : ℕ) (cost_second_shop : ℕ) (average_price : ℕ) :
    books_first_shop = 50 →
    cost_first_shop = 1000 →
    cost_second_shop = 800 →
    average_price = 20 →
    average_price * (books_first_shop + books_second_shop) = cost_first_shop + cost_second_shop →
    books_second_shop = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_number_of_books_from_second_shop_l857_85700


namespace NUMINAMATH_GPT_Vasechkin_result_l857_85720

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_Vasechkin_result_l857_85720


namespace NUMINAMATH_GPT_solve_triangle_problem_l857_85734
noncomputable def triangle_problem (A B C a b c : ℝ) (area : ℝ) : Prop :=
  (2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0) ∧
  area = Real.sqrt 3 ∧ 
  b + c = 5 →
  (A = Real.pi / 3) ∧ (a = Real.sqrt 13)

-- Lean statement for the proof problem
theorem solve_triangle_problem 
  (A B C a b c : ℝ) 
  (h1 : 2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0)
  (h2 : 1/2 * b * c * Real.sin A = Real.sqrt 3)
  (h3 : b + c = 5) :
  A = Real.pi / 3 ∧ a = Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_solve_triangle_problem_l857_85734


namespace NUMINAMATH_GPT_inequality_cannot_hold_l857_85754

theorem inequality_cannot_hold (a b : ℝ) (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_cannot_hold_l857_85754


namespace NUMINAMATH_GPT_value_of_expression_l857_85704

open Real

theorem value_of_expression (m n r t : ℝ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l857_85704


namespace NUMINAMATH_GPT_minimum_a_for_f_leq_one_range_of_a_for_max_value_l857_85744

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * log x - (1 / 3) * a * x^3 + 2 * x

theorem minimum_a_for_f_leq_one :
  ∀ {a : ℝ}, (a > 0) → (∀ x : ℝ, f a x ≤ 1) → (a ≥ 3) :=
sorry

theorem range_of_a_for_max_value :
  ∀ {a : ℝ}, (a > 0) → (∃ B : ℝ, ∀ x : ℝ, f a x ≤ B) ↔ (0 < a ∧ a ≤ (3 / 2) * exp 3) :=
sorry

end NUMINAMATH_GPT_minimum_a_for_f_leq_one_range_of_a_for_max_value_l857_85744


namespace NUMINAMATH_GPT_no_five_distinct_natural_numbers_feasible_l857_85782

theorem no_five_distinct_natural_numbers_feasible :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
  sorry

end NUMINAMATH_GPT_no_five_distinct_natural_numbers_feasible_l857_85782


namespace NUMINAMATH_GPT_baseball_cards_initial_count_unkn_l857_85751

-- Definitions based on the conditions
def cardValue : ℕ := 6
def tradedCards : ℕ := 2
def receivedCardsValue : ℕ := (3 * 2) + 9   -- 3 cards worth $2 each and 1 card worth $9
def profit : ℕ := receivedCardsValue - (tradedCards * cardValue)

-- Lean 4 statement to represent the proof problem
theorem baseball_cards_initial_count_unkn (h_trade : tradedCards * cardValue = 12)
    (h_receive : receivedCardsValue = 15)
    (h_profit : profit = 3) : ∃ n : ℕ, n >= 2 ∧ n = 2 + (n - 2) :=
sorry

end NUMINAMATH_GPT_baseball_cards_initial_count_unkn_l857_85751


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l857_85753

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l857_85753


namespace NUMINAMATH_GPT_intersection_complement_eq_l857_85714

open Set

def U : Set Int := univ
def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 3}

theorem intersection_complement_eq :
  (U \ M) ∩ N = {3} :=
  by sorry

end NUMINAMATH_GPT_intersection_complement_eq_l857_85714


namespace NUMINAMATH_GPT_range_of_a_l857_85772

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → abs (2 * a - 1) ≤ abs (x + 1 / x)) →
  -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l857_85772


namespace NUMINAMATH_GPT_cistern_length_is_four_l857_85747

noncomputable def length_of_cistern (width depth total_area : ℝ) : ℝ :=
  let L := ((total_area - (2 * width * depth)) / (2 * (width + depth)))
  L

theorem cistern_length_is_four
  (width depth total_area : ℝ)
  (h_width : width = 2)
  (h_depth : depth = 1.25)
  (h_total_area : total_area = 23) :
  length_of_cistern width depth total_area = 4 :=
by 
  sorry

end NUMINAMATH_GPT_cistern_length_is_four_l857_85747


namespace NUMINAMATH_GPT_subtraction_decimal_nearest_hundredth_l857_85742

theorem subtraction_decimal_nearest_hundredth : 
  (845.59 - 249.27 : ℝ) = 596.32 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_decimal_nearest_hundredth_l857_85742


namespace NUMINAMATH_GPT_range_of_a_l857_85770

theorem range_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, ∀ x : ℝ, x + a * x0 + 1 < 0) → (a ≥ -2 ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l857_85770


namespace NUMINAMATH_GPT_prism_volume_l857_85796

noncomputable def volume_prism (x y z : ℝ) : ℝ := x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 12) (h2 : y * z = 8) (h3 : z * x = 6) :
  volume_prism x y z = 24 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l857_85796


namespace NUMINAMATH_GPT_lena_muffins_l857_85773

theorem lena_muffins (x y z : Real) 
  (h1 : x + 2 * y + 3 * z = 3 * x + z)
  (h2 : 3 * x + z = 6 * y)
  (h3 : x + 2 * y + 3 * z = 6 * y)
  (lenas_spending : 2 * x + 2 * z = 6 * y) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end NUMINAMATH_GPT_lena_muffins_l857_85773


namespace NUMINAMATH_GPT_find_angle_beta_l857_85784

theorem find_angle_beta (α β : ℝ)
  (h1 : (π / 2) < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19)
  (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := 
sorry

end NUMINAMATH_GPT_find_angle_beta_l857_85784
