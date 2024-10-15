import Mathlib

namespace NUMINAMATH_GPT_original_price_of_cycle_l1343_134327

variable (P : ℝ)

theorem original_price_of_cycle (h1 : 0.75 * P = 1050) : P = 1400 :=
sorry

end NUMINAMATH_GPT_original_price_of_cycle_l1343_134327


namespace NUMINAMATH_GPT_find_x_l1343_134388

-- Definitions to capture angles and triangle constraints
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def perpendicular (A B : ℝ) : Prop := A + B = 90

-- Given conditions
axiom angle_ABC : ℝ
axiom angle_BAC : ℝ
axiom angle_BCA : ℝ
axiom angle_DCE : ℝ
axiom angle_x : ℝ

-- Specific values for the angles provided in the problem
axiom angle_ABC_is_70 : angle_ABC = 70
axiom angle_BAC_is_50 : angle_BAC = 50

-- Angle BCA in triangle ABC
axiom angle_sum_ABC : angle_sum_triangle angle_ABC angle_BAC angle_BCA

-- Conditional relationships in triangle CDE
axiom angle_DCE_equals_BCA : angle_DCE = angle_BCA
axiom angle_sum_CDE : perpendicular angle_DCE angle_x

-- The theorem we need to prove
theorem find_x : angle_x = 30 := sorry

end NUMINAMATH_GPT_find_x_l1343_134388


namespace NUMINAMATH_GPT_slope_acute_l1343_134347

noncomputable def curve (a : ℤ) : ℝ → ℝ := λ x => x^3 - 2 * a * x^2 + 2 * a * x

noncomputable def tangent_slope (a : ℤ) : ℝ → ℝ := λ x => 3 * x^2 - 4 * a * x + 2 * a

theorem slope_acute (a : ℤ) : (∀ x : ℝ, (tangent_slope a x > 0)) ↔ (a = 1) := sorry

end NUMINAMATH_GPT_slope_acute_l1343_134347


namespace NUMINAMATH_GPT_total_hike_time_l1343_134300

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end NUMINAMATH_GPT_total_hike_time_l1343_134300


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1343_134311

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1343_134311


namespace NUMINAMATH_GPT_smallest_three_digit_integer_solution_l1343_134370

theorem smallest_three_digit_integer_solution :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (∃ a b c : ℕ,
      n = 100 * a + 10 * b + c ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧ 
      0 ≤ c ∧ c ≤ 9 ∧
      2 * n = 100 * c + 10 * b + a + 5) ∧ 
    n = 102 := by
{
  sorry
}

end NUMINAMATH_GPT_smallest_three_digit_integer_solution_l1343_134370


namespace NUMINAMATH_GPT_total_spent_on_toys_l1343_134350

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_spent : ℝ := 12.30

theorem total_spent_on_toys : football_cost + marbles_cost = total_spent :=
by sorry

end NUMINAMATH_GPT_total_spent_on_toys_l1343_134350


namespace NUMINAMATH_GPT_taishan_maiden_tea_prices_l1343_134379

theorem taishan_maiden_tea_prices (x y : ℝ) 
  (h1 : 30 * x + 20 * y = 6000)
  (h2 : 24 * x + 18 * y = 5100) :
  x = 100 ∧ y = 150 :=
by
  sorry

end NUMINAMATH_GPT_taishan_maiden_tea_prices_l1343_134379


namespace NUMINAMATH_GPT_polygon_sides_of_interior_angle_l1343_134319

theorem polygon_sides_of_interior_angle (n : ℕ) (h : ∀ i : Fin n, (∃ (x : ℝ), x = (180 - 144) / 1) → (360 / (180 - 144)) = n) : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_sides_of_interior_angle_l1343_134319


namespace NUMINAMATH_GPT_probability_of_selection_l1343_134336

/-- A school selects 80 students for a discussion from a total of 883 students. First, 3 people are eliminated using simple random sampling, and then 80 are selected from the remaining 880 using systematic sampling. Prove that the probability of each person being selected is 80/883. -/
theorem probability_of_selection (total_students : ℕ) (students_eliminated : ℕ) (students_selected : ℕ) 
  (h_total : total_students = 883) (h_eliminated : students_eliminated = 3) (h_selected : students_selected = 80) :
  ((total_students - students_eliminated) * students_selected) / (total_students * (total_students - students_eliminated)) = 80 / 883 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selection_l1343_134336


namespace NUMINAMATH_GPT_solve_system_eq_l1343_134371

theorem solve_system_eq (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : 3 * x + 2 * y = 8) :
  x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eq_l1343_134371


namespace NUMINAMATH_GPT_other_team_members_points_l1343_134329

theorem other_team_members_points :
  ∃ (x : ℕ), ∃ (y : ℕ), (y ≤ 9 * 3) ∧ (x = y + 18 + x / 3 + x / 5) ∧ y = 24 :=
by
  sorry

end NUMINAMATH_GPT_other_team_members_points_l1343_134329


namespace NUMINAMATH_GPT_most_likely_number_of_red_balls_l1343_134323

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end NUMINAMATH_GPT_most_likely_number_of_red_balls_l1343_134323


namespace NUMINAMATH_GPT_find_sum_of_xyz_l1343_134366

theorem find_sum_of_xyz : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  (151 / 44 : ℚ) = 3 + 1 / (x + 1 / (y + 1 / z)) ∧ x + y + z = 11 :=
by 
  sorry

end NUMINAMATH_GPT_find_sum_of_xyz_l1343_134366


namespace NUMINAMATH_GPT_find_x_l1343_134392

variable (x : ℤ)

-- Define the conditions based on the problem
def adjacent_sum_condition := 
  (x + 15) + (x + 8) + (x - 7) = x

-- State the goal, which is to prove x = -8
theorem find_x : x = -8 :=
by
  have h : adjacent_sum_condition x := sorry
  sorry

end NUMINAMATH_GPT_find_x_l1343_134392


namespace NUMINAMATH_GPT_volume_between_concentric_spheres_l1343_134302

theorem volume_between_concentric_spheres
  (r1 r2 : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 10) :
  (4 / 3 * Real.pi * r2^3 - 4 / 3 * Real.pi * r1^3) = (3500 / 3) * Real.pi :=
by
  rw [h_r1, h_r2]
  sorry

end NUMINAMATH_GPT_volume_between_concentric_spheres_l1343_134302


namespace NUMINAMATH_GPT_triangular_region_area_l1343_134330

def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def y (x : ℝ) := x

theorem triangular_region_area : 
  ∀ (x y: ℝ),
  (y = line 1 2 x ∧ y = 3) ∨ 
  (y = line (-1) 8 x ∧ y = 3) ∨ 
  (y = line 1 2 x ∧ y = line (-1) 8 x)
  →
  ∃ (area: ℝ), area = 4.00 := 
by
  sorry

end NUMINAMATH_GPT_triangular_region_area_l1343_134330


namespace NUMINAMATH_GPT_order_wxyz_l1343_134396

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_wxyz : x < y ∧ y < z ∧ z < w := by
  sorry

end NUMINAMATH_GPT_order_wxyz_l1343_134396


namespace NUMINAMATH_GPT_find_abcd_from_N_l1343_134378

theorem find_abcd_from_N (N : ℕ) (hN1 : N ≥ 10000) (hN2 : N < 100000)
  (hN3 : N % 100000 = (N ^ 2) % 100000) : (N / 10) / 10 / 10 / 10 = 2999 := by
  sorry

end NUMINAMATH_GPT_find_abcd_from_N_l1343_134378


namespace NUMINAMATH_GPT_angle_between_clock_hands_at_3_05_l1343_134386

theorem angle_between_clock_hands_at_3_05 :
  let minute_angle := 5 * 6
  let hour_angle := (5 / 60) * 30
  let initial_angle := 3 * 30
  initial_angle - minute_angle + hour_angle = 62.5 := by
  sorry

end NUMINAMATH_GPT_angle_between_clock_hands_at_3_05_l1343_134386


namespace NUMINAMATH_GPT_problem_3_equals_answer_l1343_134325

variable (a : ℝ)

theorem problem_3_equals_answer :
  (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 :=
by
  sorry

end NUMINAMATH_GPT_problem_3_equals_answer_l1343_134325


namespace NUMINAMATH_GPT_total_books_is_correct_l1343_134304

-- Definitions based on the conditions
def initial_books_benny : Nat := 24
def books_given_to_sandy : Nat := 10
def books_tim : Nat := 33

-- Definition based on the computation in the solution
def books_benny_now := initial_books_benny - books_given_to_sandy
def total_books : Nat := books_benny_now + books_tim

-- The statement to be proven
theorem total_books_is_correct : total_books = 47 := by
  sorry

end NUMINAMATH_GPT_total_books_is_correct_l1343_134304


namespace NUMINAMATH_GPT_system_of_equations_solution_l1343_134320

theorem system_of_equations_solution :
  ∀ (a b : ℝ),
  (-2 * a + b^2 = Real.cos (π * a + b^2) - 1 ∧ b^2 = Real.cos (2 * π * a + b^2) - 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 0) :=
by
  intro a b
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1343_134320


namespace NUMINAMATH_GPT_largest_possible_A_l1343_134305

theorem largest_possible_A (A B C : ℕ) (h1 : 10 = A * B + C) (h2 : B = C) : A ≤ 9 :=
by sorry

end NUMINAMATH_GPT_largest_possible_A_l1343_134305


namespace NUMINAMATH_GPT_winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l1343_134314

def game (n : ℕ) : Prop :=
  ∃ A_winning_strategy B_winning_strategy neither_winning_strategy,
    (n ≥ 8 → A_winning_strategy) ∧
    (n ≤ 5 → B_winning_strategy) ∧
    (n = 6 ∨ n = 7 → neither_winning_strategy)

theorem winning_strategy_for_A (n : ℕ) (h : n ≥ 8) :
  game n :=
sorry

theorem winning_strategy_for_B (n : ℕ) (h : n ≤ 5) :
  game n :=
sorry

theorem no_winning_strategy (n : ℕ) (h : n = 6 ∨ n = 7) :
  game n :=
sorry

end NUMINAMATH_GPT_winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l1343_134314


namespace NUMINAMATH_GPT_at_least_one_zero_l1343_134326

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem at_least_one_zero (p q : ℝ) (h_zero : ∃ m : ℝ, f m p q = 0 ∧ f (f (f m p q) p q) p q = 0) :
  f 0 p q = 0 ∨ f 1 p q = 0 :=
sorry

end NUMINAMATH_GPT_at_least_one_zero_l1343_134326


namespace NUMINAMATH_GPT_minimum_choir_members_l1343_134309

def choir_members_min (n : ℕ) : Prop :=
  (n % 8 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 10 = 0) ∧ 
  (n % 11 = 0)

theorem minimum_choir_members : ∃ n, choir_members_min n ∧ (∀ m, choir_members_min m → n ≤ m) :=
sorry

end NUMINAMATH_GPT_minimum_choir_members_l1343_134309


namespace NUMINAMATH_GPT_batsman_average_after_17th_l1343_134357

def runs_17th_inning : ℕ := 87
def increase_in_avg : ℕ := 4
def num_innings : ℕ := 17

theorem batsman_average_after_17th (A : ℕ) (H : A + increase_in_avg = (16 * A + runs_17th_inning) / num_innings) : 
  (A + increase_in_avg) = 23 := sorry

end NUMINAMATH_GPT_batsman_average_after_17th_l1343_134357


namespace NUMINAMATH_GPT_axis_of_symmetry_l1343_134339

theorem axis_of_symmetry (x : ℝ) (h : x = -Real.pi / 12) :
  ∃ k : ℤ, 2 * x - Real.pi / 3 = k * Real.pi + Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1343_134339


namespace NUMINAMATH_GPT_no_real_intersections_l1343_134374

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end NUMINAMATH_GPT_no_real_intersections_l1343_134374


namespace NUMINAMATH_GPT_part1_part2_l1343_134360

def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x * x - 4 * x + 3 ≤ 0

theorem part1 (a : ℝ) (h : a = 2) (hpq : ∀ x : ℝ, p x a ∧ q x) :
  Set.Ico 1 (2 : ℝ) = {x : ℝ | p x a ∧ q x} :=
by {
  sorry
}

theorem part2 (hp : ∀ (x a : ℝ), p x a → ¬ q x) : {a : ℝ | ∀ x : ℝ, q x → p x a} = Set.Ioi 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1343_134360


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l1343_134310

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l1343_134310


namespace NUMINAMATH_GPT_expression_divisible_by_84_l1343_134397

theorem expression_divisible_by_84 (p : ℕ) (hp : p > 0) : (4 ^ (2 * p) - 3 ^ (2 * p) - 7) % 84 = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_84_l1343_134397


namespace NUMINAMATH_GPT_height_difference_l1343_134382

def burj_khalifa_height : ℝ := 830
def sears_tower_height : ℝ := 527

theorem height_difference : burj_khalifa_height - sears_tower_height = 303 := 
by
  sorry

end NUMINAMATH_GPT_height_difference_l1343_134382


namespace NUMINAMATH_GPT_razorback_shop_jersey_revenue_l1343_134348

theorem razorback_shop_jersey_revenue :
  let price_per_tshirt := 67
  let price_per_jersey := 165
  let tshirts_sold := 74
  let jerseys_sold := 156
  jerseys_sold * price_per_jersey = 25740 := by
  sorry

end NUMINAMATH_GPT_razorback_shop_jersey_revenue_l1343_134348


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1343_134358

theorem solution_set_of_inequality :
  { x : ℝ | abs (x - 4) + abs (3 - x) < 2 } = { x : ℝ | 2.5 < x ∧ x < 4.5 } := sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1343_134358


namespace NUMINAMATH_GPT_unique_representation_l1343_134345

theorem unique_representation {p x y : ℕ} 
  (hp : p > 2 ∧ Prime p) 
  (h : 2 * y = p * (x + y)) 
  (hx : x ≠ y) : 
  ∃ x y : ℕ, (1/x + 1/y = 2/p) ∧ x ≠ y := 
sorry

end NUMINAMATH_GPT_unique_representation_l1343_134345


namespace NUMINAMATH_GPT_converse_equivalence_l1343_134375

-- Definition of the original proposition
def original_proposition : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

-- Definition of the converse proposition
def converse_proposition : Prop := ∀ (x : ℝ), x^2 > 0 → x < 0

-- Theorem statement asserting the equivalence
theorem converse_equivalence : (converse_proposition = ¬ original_proposition) :=
sorry

end NUMINAMATH_GPT_converse_equivalence_l1343_134375


namespace NUMINAMATH_GPT_floor_tiling_l1343_134387

-- Define that n can be expressed as 7k for some integer k.
theorem floor_tiling (n : ℕ) (h : ∃ x : ℕ, n^2 = 7 * x) : ∃ k : ℕ, n = 7 * k := by
  sorry

end NUMINAMATH_GPT_floor_tiling_l1343_134387


namespace NUMINAMATH_GPT_simplify_expression_l1343_134385

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) : (a - 2) * (b - 2) = -2 * m := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1343_134385


namespace NUMINAMATH_GPT_sin_range_l1343_134376

theorem sin_range (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2)) : 
  Set.range (fun x => Real.sin x) = Set.Icc (1/2 : ℝ) 1 :=
sorry

end NUMINAMATH_GPT_sin_range_l1343_134376


namespace NUMINAMATH_GPT_intersection_of_sets_l1343_134301

def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }
def setB : Set ℝ := { x | 2*x - 3 > 0 }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | x > 3/2 ∧ x < 3 } :=
  by sorry

end NUMINAMATH_GPT_intersection_of_sets_l1343_134301


namespace NUMINAMATH_GPT_wheel_radius_l1343_134346

theorem wheel_radius 
(D: ℝ) (N: ℕ) (r: ℝ) 
(hD: D = 88 * 1000) 
(hN: N = 1000) 
(hC: 2 * Real.pi * r * N = D) : 
r = 88 / (2 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_wheel_radius_l1343_134346


namespace NUMINAMATH_GPT_liters_per_bottle_l1343_134315

-- Condition statements
def price_per_liter : ℕ := 1
def total_cost : ℕ := 12
def num_bottles : ℕ := 6

-- Desired result statement
theorem liters_per_bottle : (total_cost / price_per_liter) / num_bottles = 2 := by
  sorry

end NUMINAMATH_GPT_liters_per_bottle_l1343_134315


namespace NUMINAMATH_GPT_perfect_square_condition_l1343_134313

-- Definitions from conditions
def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

-- Theorem statement
theorem perfect_square_condition (n : ℤ) (h1 : 0 < n) (h2 : is_integer (2 + 2 * Real.sqrt (1 + 12 * (n: ℝ)^2))) : 
  is_perfect_square n :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_condition_l1343_134313


namespace NUMINAMATH_GPT_negation_implication_l1343_134328

theorem negation_implication (a b c : ℝ) : 
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by 
  sorry

end NUMINAMATH_GPT_negation_implication_l1343_134328


namespace NUMINAMATH_GPT_adam_age_l1343_134369

theorem adam_age (x : ℤ) :
  (∃ m : ℤ, x - 2 = m^2) ∧ (∃ n : ℤ, x + 2 = n^3) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_adam_age_l1343_134369


namespace NUMINAMATH_GPT_problem_statement_l1343_134373

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem problem_statement :
  (∀ x : ℝ, f (x) = 0 → x = - Real.pi / 6) ∧ (∀ x : ℝ, f (x) = 4 * Real.cos (2 * x - Real.pi / 6)) := sorry

end NUMINAMATH_GPT_problem_statement_l1343_134373


namespace NUMINAMATH_GPT_find_x_l1343_134355

theorem find_x
  (x : ℝ)
  (h : (x + 1) / (x + 5) = (x + 5) / (x + 13)) :
  x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1343_134355


namespace NUMINAMATH_GPT_license_plate_count_l1343_134399

-- Formalize the conditions
def is_letter (c : Char) : Prop := 'a' ≤ c ∧ c ≤ 'z'
def is_digit (c : Char) : Prop := '0' ≤ c ∧ c ≤ '9'

-- Define the main proof problem
theorem license_plate_count :
  (26 * (25 + 9) * 26 * 10 = 236600) :=
by sorry

end NUMINAMATH_GPT_license_plate_count_l1343_134399


namespace NUMINAMATH_GPT_chess_tournament_ratio_l1343_134308

theorem chess_tournament_ratio:
  ∃ n : ℕ, (n * (n - 1)) / 2 = 231 ∧ (n - 1) = 21 := 
sorry

end NUMINAMATH_GPT_chess_tournament_ratio_l1343_134308


namespace NUMINAMATH_GPT_amount_c_gets_l1343_134356

theorem amount_c_gets (total_amount : ℕ) (ratio_b ratio_c : ℕ) (h_total_amount : total_amount = 2000) (h_ratio : ratio_b = 4 ∧ ratio_c = 16) : ∃ (c_amount: ℕ), c_amount = 1600 :=
by
  sorry

end NUMINAMATH_GPT_amount_c_gets_l1343_134356


namespace NUMINAMATH_GPT_tangent_line_at_1_1_is_5x_plus_y_minus_6_l1343_134318

noncomputable def f : ℝ → ℝ :=
  λ x => x^3 - 4*x^2 + 4

def tangent_line_equation (x₀ y₀ m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y - y₀ = m * (x - x₀)

theorem tangent_line_at_1_1_is_5x_plus_y_minus_6 : 
  tangent_line_equation 1 1 (-5) = (λ x y => 5 * x + y - 6 = 0) := 
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_1_is_5x_plus_y_minus_6_l1343_134318


namespace NUMINAMATH_GPT_sequence_arithmetic_l1343_134334

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * n^2 - 3 * n)
  (h₀ : S 0 = 0) 
  (h₁ : ∀ n, S (n+1) = S n + a (n+1)) :
  ∀ n, a n = 4 * n - 1 := sorry

end NUMINAMATH_GPT_sequence_arithmetic_l1343_134334


namespace NUMINAMATH_GPT_equal_intercepts_condition_l1343_134317

theorem equal_intercepts_condition (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  (a = b ∨ c = 0) ↔ (c = 0 ∨ (c ≠ 0 ∧ a = b)) :=
by sorry

end NUMINAMATH_GPT_equal_intercepts_condition_l1343_134317


namespace NUMINAMATH_GPT_min_small_bottles_needed_l1343_134343

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end NUMINAMATH_GPT_min_small_bottles_needed_l1343_134343


namespace NUMINAMATH_GPT_initial_pokemon_cards_l1343_134398

variables (x : ℕ)

theorem initial_pokemon_cards (h : x - 2 = 1) : x = 3 := 
sorry

end NUMINAMATH_GPT_initial_pokemon_cards_l1343_134398


namespace NUMINAMATH_GPT_fraction_of_b_eq_three_tenths_a_l1343_134331

theorem fraction_of_b_eq_three_tenths_a (a b : ℝ) (h1 : a + b = 100) (h2 : b = 60) :
  (3 / 10) * a = (1 / 5) * b :=
by 
  have h3 : a = 40 := by linarith [h1, h2]
  rw [h2, h3]
  linarith

end NUMINAMATH_GPT_fraction_of_b_eq_three_tenths_a_l1343_134331


namespace NUMINAMATH_GPT_length_of_one_side_of_square_l1343_134391

variable (total_ribbon_length : ℕ) (triangle_perimeter : ℕ)

theorem length_of_one_side_of_square (h1 : total_ribbon_length = 78)
                                    (h2 : triangle_perimeter = 46) :
  (total_ribbon_length - triangle_perimeter) / 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_one_side_of_square_l1343_134391


namespace NUMINAMATH_GPT_cubes_difference_l1343_134384

theorem cubes_difference :
  let a := 642
  let b := 641
  a^3 - b^3 = 1234567 :=
by
  let a := 642
  let b := 641
  have h : a^3 - b^3 = 264609288 - 263374721 := sorry
  have h_correct : 264609288 - 263374721 = 1234567 := sorry
  exact Eq.trans h h_correct

end NUMINAMATH_GPT_cubes_difference_l1343_134384


namespace NUMINAMATH_GPT_mike_sold_song_book_for_correct_amount_l1343_134352

-- Define the constants for the cost of the trumpet and the net amount spent
def cost_of_trumpet : ℝ := 145.16
def net_amount_spent : ℝ := 139.32

-- Define the amount received from selling the song book
def amount_received_from_selling_song_book : ℝ :=
  cost_of_trumpet - net_amount_spent

-- The theorem stating the amount Mike sold the song book for
theorem mike_sold_song_book_for_correct_amount :
  amount_received_from_selling_song_book = 5.84 :=
sorry

end NUMINAMATH_GPT_mike_sold_song_book_for_correct_amount_l1343_134352


namespace NUMINAMATH_GPT_geometric_sequence_frac_l1343_134322

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
variable (h_decreasing : ∀ n, a (n+1) < a n)
variable (h1 : a 2 * a 8 = 6)
variable (h2 : a 4 + a 6 = 5)

theorem geometric_sequence_frac (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
                                (h_decreasing : ∀ n, a (n+1) < a n)
                                (h1 : a 2 * a 8 = 6)
                                (h2 : a 4 + a 6 = 5) :
                                a 3 / a 7 = 9 / 4 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_frac_l1343_134322


namespace NUMINAMATH_GPT_max_fraction_l1343_134380

theorem max_fraction (x y : ℝ) (h1 : -6 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 5) :
  (∀ x y, -6 ≤ x → x ≤ -3 → 3 ≤ y → y ≤ 5 → (x - y) / y ≤ -2) :=
by
  sorry

end NUMINAMATH_GPT_max_fraction_l1343_134380


namespace NUMINAMATH_GPT_factor_correct_l1343_134394

noncomputable def factor_expression (x : ℝ) : ℝ :=
  66 * x^6 - 231 * x^12

theorem factor_correct (x : ℝ) :
  factor_expression x = 33 * x^6 * (2 - 7 * x^6) :=
by 
  sorry

end NUMINAMATH_GPT_factor_correct_l1343_134394


namespace NUMINAMATH_GPT_triangle_smallest_side_l1343_134344

theorem triangle_smallest_side (a b c : ℝ) (h : b^2 + c^2 ≥ 5 * a^2) : 
    (a ≤ b ∧ a ≤ c) := 
sorry

end NUMINAMATH_GPT_triangle_smallest_side_l1343_134344


namespace NUMINAMATH_GPT_count_valid_orderings_l1343_134365

-- Define the houses and conditions
inductive HouseColor where
  | Green
  | Purple
  | Blue
  | Pink
  | X -- Representing the fifth unspecified house

open HouseColor

def validOrderings : List (List HouseColor) :=
  [
    [Green, Blue, Purple, Pink, X], 
    [Green, Blue, X, Purple, Pink],
    [Green, X, Purple, Blue, Pink],
    [X, Pink, Purple, Blue, Green],
    [X, Purple, Pink, Blue, Green],
    [X, Pink, Blue, Purple, Green]
  ] 

-- Prove that there are exactly 6 valid orderings
theorem count_valid_orderings : (validOrderings.length = 6) :=
by
  -- Since we list all possible valid orderings above, just compute the length
  sorry

end NUMINAMATH_GPT_count_valid_orderings_l1343_134365


namespace NUMINAMATH_GPT_f_value_l1343_134338

noncomputable def f : ℝ → ℝ
| x => if x > 1 then 2^(x-1) else Real.tan (Real.pi * x / 3)

theorem f_value : f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_f_value_l1343_134338


namespace NUMINAMATH_GPT_solve_fractional_equation_l1343_134354

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 2) : 
  (4 * x ^ 2 + 3 * x + 2) / (x - 2) = 4 * x + 5 ↔ x = -2 := by 
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1343_134354


namespace NUMINAMATH_GPT_shift_graph_sin_cos_l1343_134342

open Real

theorem shift_graph_sin_cos :
  ∀ x : ℝ, sin (2 * x + π / 3) = cos (2 * (x + π / 12) - π / 3) :=
by
  sorry

end NUMINAMATH_GPT_shift_graph_sin_cos_l1343_134342


namespace NUMINAMATH_GPT_longest_collection_pages_l1343_134389

theorem longest_collection_pages 
    (pages_per_inch_miles : ℕ := 5) 
    (pages_per_inch_daphne : ℕ := 50) 
    (height_miles : ℕ := 240) 
    (height_daphne : ℕ := 25) : 
  max (pages_per_inch_miles * height_miles) (pages_per_inch_daphne * height_daphne) = 1250 := 
by
  sorry

end NUMINAMATH_GPT_longest_collection_pages_l1343_134389


namespace NUMINAMATH_GPT_part1_part2_l1343_134372

-- The quadratic equation of interest
def quadratic_eq (k x : ℝ) : ℝ :=
  x^2 + (2 * k - 1) * x + k^2 - k

-- Part 1: Proof that the equation has two distinct real roots
theorem part1 (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_eq k x1 = 0 ∧ quadratic_eq k x2 = 0) := 
  sorry

-- Part 2: Given x = 2 is a root, prove the value of the expression
theorem part2 (k : ℝ) (h : quadratic_eq k 2 = 0) : -2 * k^2 - 6 * k - 5 = -1 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1343_134372


namespace NUMINAMATH_GPT_monomial_2023_eq_l1343_134368

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^(n+1) * (2*n - 1), n)

theorem monomial_2023_eq : monomial 2023 = (4045, 2023) :=
by
  sorry

end NUMINAMATH_GPT_monomial_2023_eq_l1343_134368


namespace NUMINAMATH_GPT_find_missing_number_l1343_134335

theorem find_missing_number (n : ℝ) : n * 120 = 173 * 240 → n = 345.6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_missing_number_l1343_134335


namespace NUMINAMATH_GPT_charles_average_speed_l1343_134306

theorem charles_average_speed
  (total_distance : ℕ)
  (half_distance : ℕ)
  (second_half_speed : ℕ)
  (total_time : ℕ)
  (first_half_distance second_half_distance : ℕ)
  (time_for_second_half : ℕ)
  (time_for_first_half : ℕ)
  (first_half_speed : ℕ)
  (h1 : total_distance = 3600)
  (h2 : half_distance = total_distance / 2)
  (h3 : first_half_distance = half_distance)
  (h4 : second_half_distance = half_distance)
  (h5 : second_half_speed = 180)
  (h6 : total_time = 30)
  (h7 : time_for_second_half = second_half_distance / second_half_speed)
  (h8 : time_for_first_half = total_time - time_for_second_half)
  (h9 : first_half_speed = first_half_distance / time_for_first_half) :
  first_half_speed = 90 := by
  sorry

end NUMINAMATH_GPT_charles_average_speed_l1343_134306


namespace NUMINAMATH_GPT_hotdogs_per_hour_l1343_134390

-- Define the necessary conditions
def price_per_hotdog : ℝ := 2
def total_hours : ℝ := 10
def total_sales : ℝ := 200

-- Prove that the number of hot dogs sold per hour equals 10
theorem hotdogs_per_hour : (total_sales / total_hours) / price_per_hotdog = 10 :=
by
  sorry

end NUMINAMATH_GPT_hotdogs_per_hour_l1343_134390


namespace NUMINAMATH_GPT_g_constant_term_l1343_134362

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

-- Conditions from the problem
def f_has_constant_term_5 : f.coeff 0 = 5 := sorry
def h_has_constant_term_neg_10 : h.coeff 0 = -10 := sorry
def g_is_quadratic : g.degree ≤ 2 := sorry

-- Statement of the problem
theorem g_constant_term : g.coeff 0 = -2 :=
by
  have h_eq_fg : h = f * g := rfl
  have f_const := f_has_constant_term_5
  have h_const := h_has_constant_term_neg_10
  have g_quad := g_is_quadratic
  sorry

end NUMINAMATH_GPT_g_constant_term_l1343_134362


namespace NUMINAMATH_GPT_remainder_of_power_modulo_l1343_134340

theorem remainder_of_power_modulo : (3^2048) % 11 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_of_power_modulo_l1343_134340


namespace NUMINAMATH_GPT_friends_lunch_spending_l1343_134332

-- Problem conditions and statement to prove
theorem friends_lunch_spending (x : ℝ) (h1 : x + (x + 15) + (x - 20) + 2 * x = 100) : 
  x = 21 :=
by sorry

end NUMINAMATH_GPT_friends_lunch_spending_l1343_134332


namespace NUMINAMATH_GPT_combined_6th_grade_percentage_l1343_134321

noncomputable def percentage_of_6th_graders 
  (parkPercent : Fin 7 → ℚ) 
  (riversidePercent : Fin 7 → ℚ) 
  (totalParkside : ℕ) 
  (totalRiverside : ℕ) 
  : ℚ := 
    let num6thParkside := parkPercent 6 * totalParkside
    let num6thRiverside := riversidePercent 6 * totalRiverside
    let total6thGraders := num6thParkside + num6thRiverside
    let totalStudents := totalParkside + totalRiverside
    (total6thGraders / totalStudents) * 100

theorem combined_6th_grade_percentage :
  let parkPercent := ![(14.0 : ℚ) / 100, 13 / 100, 16 / 100, 15 / 100, 12 / 100, 15 / 100, 15 / 100]
  let riversidePercent := ![(13.0 : ℚ) / 100, 16 / 100, 13 / 100, 15 / 100, 14 / 100, 15 / 100, 14 / 100]
  percentage_of_6th_graders parkPercent riversidePercent 150 250 = 15 := 
  by
  sorry

end NUMINAMATH_GPT_combined_6th_grade_percentage_l1343_134321


namespace NUMINAMATH_GPT_simplify_expression_l1343_134337

variable (x : ℝ)
variable (h₁ : x ≠ 2)
variable (h₂ : x ≠ 3)
variable (h₃ : x ≠ 4)
variable (h₄ : x ≠ 5)

theorem simplify_expression : 
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) 
  = ( (x - 1) * (x - 5) ) / ( (x - 4) * (x - 2) * (x - 3) ) ) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1343_134337


namespace NUMINAMATH_GPT_find_numbers_l1343_134312

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1343_134312


namespace NUMINAMATH_GPT_mandy_pieces_eq_fifteen_l1343_134303

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end NUMINAMATH_GPT_mandy_pieces_eq_fifteen_l1343_134303


namespace NUMINAMATH_GPT_problem_real_numbers_inequality_l1343_134333

open Real

theorem problem_real_numbers_inequality 
  (a1 b1 a2 b2 : ℝ) :
  a1 * b1 + a2 * b2 ≤ sqrt (a1^2 + a2^2) * sqrt (b1^2 + b2^2) :=
by 
  sorry

end NUMINAMATH_GPT_problem_real_numbers_inequality_l1343_134333


namespace NUMINAMATH_GPT_probability_of_target_destroyed_l1343_134351

theorem probability_of_target_destroyed :
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  (p1 * p2 * p3) + (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) = 0.954 :=
by
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  sorry

end NUMINAMATH_GPT_probability_of_target_destroyed_l1343_134351


namespace NUMINAMATH_GPT_inequality_proof_inequality_equality_conditions_l1343_134367

theorem inequality_proof
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  (x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 ≤ (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2)) :=
sorry

theorem inequality_equality_conditions
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 = (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2))
  ↔ (x1 = x2 ∧ y1 = y2 ∧ z1 = z2)) :=
sorry

end NUMINAMATH_GPT_inequality_proof_inequality_equality_conditions_l1343_134367


namespace NUMINAMATH_GPT_degrees_of_remainder_division_l1343_134341

theorem degrees_of_remainder_division (f g : Polynomial ℝ) (h : g = Polynomial.C 3 * Polynomial.X ^ 3 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X + Polynomial.C (-8)) :
  ∃ r q : Polynomial ℝ, f = g * q + r ∧ (r.degree < 3) := 
sorry

end NUMINAMATH_GPT_degrees_of_remainder_division_l1343_134341


namespace NUMINAMATH_GPT_number_of_adult_dogs_l1343_134361

theorem number_of_adult_dogs (x : ℕ) (h : 2 * 50 + x * 100 + 2 * 150 = 700) : x = 3 :=
by
  -- Definitions from conditions
  have cost_cats := 2 * 50
  have cost_puppies := 2 * 150
  have total_cost := 700
  
  -- Using the provided hypothesis to assert our proof
  sorry

end NUMINAMATH_GPT_number_of_adult_dogs_l1343_134361


namespace NUMINAMATH_GPT_length_of_each_part_l1343_134316

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_of_each_part_l1343_134316


namespace NUMINAMATH_GPT_quadratic_intersects_x_axis_l1343_134307

theorem quadratic_intersects_x_axis (a b : ℝ) (h : a ≠ 0) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 - (b^2 / (4 * a)) = 0 ∧ a * x2^2 + b * x2 - (b^2 / (4 * a)) = 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_intersects_x_axis_l1343_134307


namespace NUMINAMATH_GPT_energy_consumption_correct_l1343_134359

def initial_wattages : List ℕ := [60, 80, 100, 120]

def increased_wattages : List ℕ := initial_wattages.map (λ x => x + (x * 25 / 100))

def combined_wattage (ws : List ℕ) : ℕ := ws.sum

def daily_energy_consumption (cw : ℕ) : ℕ := cw * 6 / 1000

def total_energy_consumption (dec : ℕ) : ℕ := dec * 30

-- Main theorem statement
theorem energy_consumption_correct :
  total_energy_consumption (daily_energy_consumption (combined_wattage increased_wattages)) = 81 := 
sorry

end NUMINAMATH_GPT_energy_consumption_correct_l1343_134359


namespace NUMINAMATH_GPT_abs_sub_eq_five_l1343_134381

theorem abs_sub_eq_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
sorry

end NUMINAMATH_GPT_abs_sub_eq_five_l1343_134381


namespace NUMINAMATH_GPT_p_at_zero_l1343_134383

-- Definitions according to given conditions
def p (x : ℝ) : ℝ := sorry  -- Polynomial of degree 6 with specific values

-- Given condition: Degree of polynomial
def degree_p : Prop := (∀ n : ℕ, (n ≤ 6) → p (3 ^ n) = 1 / 3 ^ n)

-- Theorem that needs to be proved
theorem p_at_zero : degree_p → p 0 = 6560 / 2187 := 
by
  sorry

end NUMINAMATH_GPT_p_at_zero_l1343_134383


namespace NUMINAMATH_GPT_no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l1343_134364

noncomputable def system_discriminant (a b c : ℝ) : ℝ := (b - 1)^2 - 4 * a * c

theorem no_real_solutions_if_discriminant_neg (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c < 0) :
  ¬∃ (x₁ x₂ x₃ : ℝ), (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

theorem one_real_solution_if_discriminant_zero (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c = 0) :
  ∃ (x : ℝ), ∀ (x₁ x₂ x₃ : ℝ), (x₁ = x) ∧ (x₂ = x) ∧ (x₃ = x) ∧
                              (a * x₁^2 + b * x₁ + c = x₂) ∧
                              (a * x₂^2 + b * x₂ + c = x₃) ∧
                              (a * x₃^2 + b * x₃ + c = x₁)  :=
sorry

theorem more_than_one_real_solution_if_discriminant_pos (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c > 0) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

end NUMINAMATH_GPT_no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l1343_134364


namespace NUMINAMATH_GPT_total_weekly_earnings_l1343_134363

-- Define the total weekly hours and earnings
def weekly_hours_weekday : ℕ := 5 * 5
def weekday_rate : ℕ := 3
def weekday_earnings : ℕ := weekly_hours_weekday * weekday_rate

-- Define the total weekend hours and earnings
def weekend_days : ℕ := 2
def weekend_hours_per_day : ℕ := 3
def weekend_rate : ℕ := 3 * 2
def weekend_hours : ℕ := weekend_days * weekend_hours_per_day
def weekend_earnings : ℕ := weekend_hours * weekend_rate

-- Prove that Mitch's total earnings per week are $111
theorem total_weekly_earnings : weekday_earnings + weekend_earnings = 111 := by
  sorry

end NUMINAMATH_GPT_total_weekly_earnings_l1343_134363


namespace NUMINAMATH_GPT_nontrivial_solution_exists_l1343_134349

theorem nontrivial_solution_exists 
  (a b : ℤ) 
  (h_square_a : ∀ k : ℤ, a ≠ k^2) 
  (h_square_b : ∀ k : ℤ, b ≠ k^2) 
  (h_nontrivial : ∃ (x y z w : ℤ), x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) : 
  ∃ (x y z : ℤ), x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) :=
by
  sorry

end NUMINAMATH_GPT_nontrivial_solution_exists_l1343_134349


namespace NUMINAMATH_GPT_tea_in_each_box_initially_l1343_134353

theorem tea_in_each_box_initially (x : ℕ) 
  (h₁ : 4 * (x - 9) = x) : 
  x = 12 := 
sorry

end NUMINAMATH_GPT_tea_in_each_box_initially_l1343_134353


namespace NUMINAMATH_GPT_sin_alpha_plus_beta_alpha_plus_two_beta_l1343_134377

variables {α β : ℝ} (hα_acute : 0 < α ∧ α < π / 2) (hβ_acute : 0 < β ∧ β < π / 2)
          (h_tan_α : Real.tan α = 1 / 7) (h_sin_β : Real.sin β = Real.sqrt 10 / 10)

theorem sin_alpha_plus_beta : 
    Real.sin (α + β) = Real.sqrt 5 / 5 :=
by
  sorry

theorem alpha_plus_two_beta : 
    α + 2 * β = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_beta_alpha_plus_two_beta_l1343_134377


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l1343_134393

theorem arithmetic_expression_eval :
  ((26.3 * 12 * 20) / 3) + 125 = 2229 :=
sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l1343_134393


namespace NUMINAMATH_GPT_kabadi_players_l1343_134395

def people_play_kabadi (Kho_only Both Total : ℕ) : Prop :=
  ∃ K : ℕ, Kho_only = 20 ∧ Both = 5 ∧ Total = 30 ∧ K = Total - Kho_only ∧ (K + Both) = 15

theorem kabadi_players :
  people_play_kabadi 20 5 30 :=
by
  sorry

end NUMINAMATH_GPT_kabadi_players_l1343_134395


namespace NUMINAMATH_GPT_num_children_got_off_l1343_134324

-- Define the original number of children on the bus
def original_children : ℕ := 43

-- Define the number of children left after some got off the bus
def children_left : ℕ := 21

-- Define the number of children who got off the bus as the difference between original_children and children_left
def children_got_off : ℕ := original_children - children_left

-- State the theorem that the number of children who got off the bus is 22
theorem num_children_got_off : children_got_off = 22 :=
by
  -- Proof steps would go here, but are omitted
  sorry

end NUMINAMATH_GPT_num_children_got_off_l1343_134324
