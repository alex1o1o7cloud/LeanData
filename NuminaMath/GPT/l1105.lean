import Mathlib

namespace NUMINAMATH_GPT_ratio_black_white_l1105_110528

-- Definitions of the parameters
variables (B W : ℕ)
variables (h1 : B + W = 200)
variables (h2 : 30 * B + 25 * W = 5500)

theorem ratio_black_white (B W : ℕ) (h1 : B + W = 200) (h2 : 30 * B + 25 * W = 5500) :
  B = W :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ratio_black_white_l1105_110528


namespace NUMINAMATH_GPT_sin_double_angle_l1105_110545

theorem sin_double_angle (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin (2 * x) = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1105_110545


namespace NUMINAMATH_GPT_total_vertical_distance_of_rings_l1105_110556

theorem total_vertical_distance_of_rings :
  let thickness := 2
  let top_outside_diameter := 20
  let bottom_outside_diameter := 4
  let n := (top_outside_diameter - bottom_outside_diameter) / thickness + 1
  let total_distance := n * thickness
  total_distance + thickness = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_vertical_distance_of_rings_l1105_110556


namespace NUMINAMATH_GPT_probability_of_x_gt_3y_is_correct_l1105_110585

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle_width := 2016
  let rectangle_height := 2017
  let triangle_height := 672 -- 2016 / 3
  let triangle_area := 1 / 2 * rectangle_width * triangle_height
  let rectangle_area := rectangle_width * rectangle_height
  triangle_area / rectangle_area

theorem probability_of_x_gt_3y_is_correct :
  probability_x_gt_3y = 336 / 2017 :=
by
  -- Proof will be filled in later
  sorry

end NUMINAMATH_GPT_probability_of_x_gt_3y_is_correct_l1105_110585


namespace NUMINAMATH_GPT_total_weight_of_4_moles_of_ba_cl2_l1105_110575

-- Conditions
def atomic_weight_ba : ℝ := 137.33
def atomic_weight_cl : ℝ := 35.45
def moles_ba_cl2 : ℝ := 4

-- Molecular weight of BaCl2
def molecular_weight_ba_cl2 : ℝ := 
  atomic_weight_ba + 2 * atomic_weight_cl

-- Total weight of 4 moles of BaCl2
def total_weight : ℝ := 
  molecular_weight_ba_cl2 * moles_ba_cl2

-- Theorem stating the total weight of 4 moles of BaCl2
theorem total_weight_of_4_moles_of_ba_cl2 :
  total_weight = 832.92 :=
sorry

end NUMINAMATH_GPT_total_weight_of_4_moles_of_ba_cl2_l1105_110575


namespace NUMINAMATH_GPT_triangle_angle_A_l1105_110587

variable {a b c : ℝ} {A : ℝ}

theorem triangle_angle_A (h : a^2 = b^2 + c^2 - b * c) : A = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_A_l1105_110587


namespace NUMINAMATH_GPT_solve_equation_l1105_110504

theorem solve_equation (a : ℝ) : 
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} :=
sorry

end NUMINAMATH_GPT_solve_equation_l1105_110504


namespace NUMINAMATH_GPT_sum_cubes_first_39_eq_608400_l1105_110537

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

theorem sum_cubes_first_39_eq_608400 : sum_of_cubes 39 = 608400 :=
by
  sorry

end NUMINAMATH_GPT_sum_cubes_first_39_eq_608400_l1105_110537


namespace NUMINAMATH_GPT_eval_f_l1105_110526

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem eval_f : f (f (1 / 2)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_f_l1105_110526


namespace NUMINAMATH_GPT_books_per_shelf_l1105_110567

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ) 
    (h1 : mystery_shelves = 5) (h2 : picture_shelves = 4) (h3 : total_books = 54) : 
    total_books / (mystery_shelves + picture_shelves) = 6 := 
by
  -- necessary preliminary steps and full proof will go here
  sorry

end NUMINAMATH_GPT_books_per_shelf_l1105_110567


namespace NUMINAMATH_GPT_sum_of_odd_integers_15_to_51_l1105_110505

def odd_arithmetic_series_sum (a1 an d : ℤ) (n : ℕ) : ℤ :=
  (n * (a1 + an)) / 2

theorem sum_of_odd_integers_15_to_51 :
  odd_arithmetic_series_sum 15 51 2 19 = 627 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_odd_integers_15_to_51_l1105_110505


namespace NUMINAMATH_GPT_sum_of_coefficients_l1105_110586

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (∀ x : ℝ, (3 * x - 2)^6 = a_0 + a_1 * (2 * x - 1) + a_2 * (2 * x - 1)^2 + a_3 * (2 * x - 1)^3 + a_4 * (2 * x - 1)^4 + a_5 * (2 * x - 1)^5 + a_6 * (2 * x - 1)^6) ->
  a_1 + a_3 + a_5 = -63 / 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1105_110586


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1105_110500

variable (a : ℝ)

theorem necessary_but_not_sufficient_condition (h : 0 ≤ a ∧ a ≤ 4) :
  (∀ x : ℝ, x^2 + a * x + a > 0) → (0 ≤ a ∧ a ≤ 4 ∧ ¬ (∀ x : ℝ, x^2 + a * x + a > 0)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1105_110500


namespace NUMINAMATH_GPT_probability_not_sit_at_ends_l1105_110510

theorem probability_not_sit_at_ends (h1: ∀ M J: ℕ, M ≠ J → M ≠ 1 ∧ M ≠ 8 ∧ J ≠ 1 ∧ J ≠ 8) : 
  (∃ p: ℚ, p = (3 / 7)) :=
by 
  sorry

end NUMINAMATH_GPT_probability_not_sit_at_ends_l1105_110510


namespace NUMINAMATH_GPT_point_on_inverse_graph_and_sum_l1105_110579

-- Definitions
variable (f : ℝ → ℝ)
variable (h : f 2 = 6)

-- Theorem statement
theorem point_on_inverse_graph_and_sum (hf : ∀ x, x = 2 → 3 = (f x) / 2) :
  (6, 1 / 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, (f⁻¹ x) / 2)} ∧
  (6 + (1 / 2) = 13 / 2) :=
by
  sorry

end NUMINAMATH_GPT_point_on_inverse_graph_and_sum_l1105_110579


namespace NUMINAMATH_GPT_rectangle_circle_diameter_l1105_110503

theorem rectangle_circle_diameter:
  ∀ (m n : ℕ), (∃ (x : ℚ), m + n = 47 ∧ (∀ (r : ℚ), r = (20 / 7)) →
  (2 * r = (40 / 7))) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_circle_diameter_l1105_110503


namespace NUMINAMATH_GPT_computer_game_cost_l1105_110583

variable (ticket_cost : ℕ := 12)
variable (num_tickets : ℕ := 3)
variable (total_spent : ℕ := 102)

theorem computer_game_cost (C : ℕ) (h : C + num_tickets * ticket_cost = total_spent) : C = 66 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_computer_game_cost_l1105_110583


namespace NUMINAMATH_GPT_find_a8_l1105_110525

variable (a : ℕ+ → ℕ)

theorem find_a8 (h : ∀ m n : ℕ+, a (m * n) = a m * a n) (h2 : a 2 = 3) : a 8 = 27 := 
by
  sorry

end NUMINAMATH_GPT_find_a8_l1105_110525


namespace NUMINAMATH_GPT_groups_of_four_on_plane_l1105_110566

-- Define the points in the tetrahedron
inductive Point
| vertex : Point
| midpoint : Point

noncomputable def points : List Point :=
  [Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.midpoint,
   Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.vertex]

-- Condition: all 10 points are either vertices or midpoints of the edges of a tetrahedron 
def points_condition : ∀ p ∈ points, p = Point.vertex ∨ p = Point.midpoint := sorry

-- Function to count unique groups of four points lying on the same plane
noncomputable def count_groups : ℕ :=
  33  -- Given as the correct answer in the problem

-- Proof problem stating the count of groups
theorem groups_of_four_on_plane : count_groups = 33 :=
by 
  sorry -- Proof omitted

end NUMINAMATH_GPT_groups_of_four_on_plane_l1105_110566


namespace NUMINAMATH_GPT_marbles_given_by_Joan_l1105_110592

def initial_yellow_marbles : ℝ := 86.0
def final_yellow_marbles : ℝ := 111.0

theorem marbles_given_by_Joan :
  final_yellow_marbles - initial_yellow_marbles = 25 := by
  sorry

end NUMINAMATH_GPT_marbles_given_by_Joan_l1105_110592


namespace NUMINAMATH_GPT_common_factor_of_polynomial_l1105_110508

theorem common_factor_of_polynomial :
  ∀ (x : ℝ), (2 * x^2 - 8 * x) = 2 * x * (x - 4) := by
  sorry

end NUMINAMATH_GPT_common_factor_of_polynomial_l1105_110508


namespace NUMINAMATH_GPT_boys_who_did_not_bring_laptops_l1105_110552

-- Definitions based on the conditions.
def total_boys : ℕ := 20
def students_who_brought_laptops : ℕ := 25
def girls_who_brought_laptops : ℕ := 16

-- Main theorem statement.
theorem boys_who_did_not_bring_laptops : total_boys - (students_who_brought_laptops - girls_who_brought_laptops) = 11 := by
  sorry

end NUMINAMATH_GPT_boys_who_did_not_bring_laptops_l1105_110552


namespace NUMINAMATH_GPT_polynomial_inequality_solution_l1105_110595

theorem polynomial_inequality_solution :
  {x : ℝ | x^3 - 4*x^2 - x + 20 > 0} = {x | x < -4} ∪ {x | 1 < x ∧ x < 5} ∪ {x | x > 5} :=
sorry

end NUMINAMATH_GPT_polynomial_inequality_solution_l1105_110595


namespace NUMINAMATH_GPT_largest_c_for_minus3_in_range_of_quadratic_l1105_110588

theorem largest_c_for_minus3_in_range_of_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 5*x + c = -3) ↔ c ≤ 13/4 :=
sorry

end NUMINAMATH_GPT_largest_c_for_minus3_in_range_of_quadratic_l1105_110588


namespace NUMINAMATH_GPT_find_c_l1105_110548

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem find_c (a b c S : ℝ) (C : ℝ) 
  (ha : a = 3) 
  (hC : C = 120) 
  (hS : S = 15 * Real.sqrt 3 / 4) 
  (hab : a * b = 15)
  (hc2 : c^2 = a^2 + b^2 - 2 * a * b * cos_deg C) :
  c = 7 :=
by 
  sorry

end NUMINAMATH_GPT_find_c_l1105_110548


namespace NUMINAMATH_GPT_marks_in_physics_l1105_110540

def marks_in_english : ℝ := 74
def marks_in_mathematics : ℝ := 65
def marks_in_chemistry : ℝ := 67
def marks_in_biology : ℝ := 90
def average_marks : ℝ := 75.6
def number_of_subjects : ℕ := 5

-- We need to show that David's marks in Physics are 82.
theorem marks_in_physics : ∃ (P : ℝ), P = 82 ∧ 
  ((marks_in_english + marks_in_mathematics + P + marks_in_chemistry + marks_in_biology) / number_of_subjects = average_marks) :=
by sorry

end NUMINAMATH_GPT_marks_in_physics_l1105_110540


namespace NUMINAMATH_GPT_lattice_points_on_hyperbola_l1105_110580

-- Define the problem
def countLatticePoints (n : ℤ) : ℕ :=
  let factoredCount := (2 + 1) * (2 + 1) * (4 + 1) -- Number of divisors of 2^2 * 3^2 * 5^4
  2 * factoredCount -- Each pair has two solutions considering positive and negative values

-- The theorem to be proven
theorem lattice_points_on_hyperbola : countLatticePoints 1800 = 90 := sorry

end NUMINAMATH_GPT_lattice_points_on_hyperbola_l1105_110580


namespace NUMINAMATH_GPT_minutes_sean_played_each_day_l1105_110596

-- Define the given conditions
def t : ℕ := 1512                               -- Total minutes played by Sean and Indira
def i : ℕ := 812                                -- Total minutes played by Indira
def d : ℕ := 14                                 -- Number of days Sean played

-- Define the to-be-proved statement
theorem minutes_sean_played_each_day : (t - i) / d = 50 :=
by
  sorry

end NUMINAMATH_GPT_minutes_sean_played_each_day_l1105_110596


namespace NUMINAMATH_GPT_trees_died_in_typhoon_imply_all_died_l1105_110524

-- Given conditions
def trees_initial := 3
def survived_trees (x : Int) := x
def died_trees (x : Int) := x + 23

-- Prove that the number of died trees is 3
theorem trees_died_in_typhoon_imply_all_died : ∀ x, 2 * survived_trees x + 23 = trees_initial → trees_initial = died_trees x := 
by
  intro x h
  sorry

end NUMINAMATH_GPT_trees_died_in_typhoon_imply_all_died_l1105_110524


namespace NUMINAMATH_GPT_fractional_equation_solution_l1105_110547

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end NUMINAMATH_GPT_fractional_equation_solution_l1105_110547


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1105_110527

theorem maximum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz * (x + y + z)) / ((x + y)^2 * (y + z)^2) ≤ (1 / 4) :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1105_110527


namespace NUMINAMATH_GPT_problem_solution_l1105_110529

theorem problem_solution (x1 x2 : ℝ) (h1 : x1^2 + x1 - 4 = 0) (h2 : x2^2 + x2 - 4 = 0) (h3 : x1 + x2 = -1) : 
  x1^3 - 5 * x2^2 + 10 = -19 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l1105_110529


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1105_110584

theorem perfect_square_trinomial (m : ℝ) : (∃ (a b : ℝ), (a * x + b) ^ 2 = x^2 + m * x + 16) -> (m = 8 ∨ m = -8) :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1105_110584


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1105_110535

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ)
  (d : ℤ) (h_a : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 2 + a 3 = 156)
  (h2 : a 2 + a 3 + a 4 = 147) :
  ∃ n : ℕ, n = 19 ∧ (∀ m : ℕ, S m ≤ S n) :=
sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1105_110535


namespace NUMINAMATH_GPT_find_n_l1105_110570

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 16 = 48) (h_gcf : Nat.gcd n 16 = 4) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1105_110570


namespace NUMINAMATH_GPT_budget_percentage_l1105_110506

-- Define the given conditions
def basic_salary_per_hour : ℝ := 7.50
def commission_rate : ℝ := 0.16
def hours_worked : ℝ := 160
def total_sales : ℝ := 25000
def amount_for_insurance : ℝ := 260

-- Define the basic salary, commission, and total earnings
def basic_salary : ℝ := basic_salary_per_hour * hours_worked
def commission : ℝ := commission_rate * total_sales
def total_earnings : ℝ := basic_salary + commission
def amount_for_budget : ℝ := total_earnings - amount_for_insurance

-- Define the proof problem
theorem budget_percentage : (amount_for_budget / total_earnings) * 100 = 95 := by
  simp [basic_salary, commission, total_earnings, amount_for_budget]
  sorry

end NUMINAMATH_GPT_budget_percentage_l1105_110506


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1105_110565

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x ≥ 2 ∧ y ≥ 2) → x + y ≥ 4 ∧ (¬ (x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1105_110565


namespace NUMINAMATH_GPT_trees_died_more_than_survived_l1105_110549

theorem trees_died_more_than_survived :
  ∀ (initial_trees survived_percent : ℕ),
    initial_trees = 25 →
    survived_percent = 40 →
    (initial_trees * survived_percent / 100) + (initial_trees - initial_trees * survived_percent / 100) -
    (initial_trees * survived_percent / 100) = 5 :=
by
  intro initial_trees survived_percent initial_trees_eq survived_percent_eq
  sorry

end NUMINAMATH_GPT_trees_died_more_than_survived_l1105_110549


namespace NUMINAMATH_GPT_ratio_first_to_second_l1105_110578

theorem ratio_first_to_second (S F T : ℕ) 
  (hS : S = 60)
  (hT : T = F / 3)
  (hSum : F + S + T = 220) :
  F / S = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_first_to_second_l1105_110578


namespace NUMINAMATH_GPT_f_zero_is_118_l1105_110513

theorem f_zero_is_118
  (f : ℕ → ℕ)
  (eq1 : ∀ m n : ℕ, f (m^2 + n^2) = (f m - f n)^2 + f (2 * m * n))
  (eq2 : 8 * f 0 + 9 * f 1 = 2006) :
  f 0 = 118 :=
sorry

end NUMINAMATH_GPT_f_zero_is_118_l1105_110513


namespace NUMINAMATH_GPT_intersection_M_N_l1105_110573

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := U \ complement_U_N

theorem intersection_M_N : M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1105_110573


namespace NUMINAMATH_GPT_camel_cost_is_5200_l1105_110577

-- Definitions of costs in terms of Rs.
variable (C H O E : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : ∃ X : ℕ, X * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 130000

-- Theorem to prove
theorem camel_cost_is_5200 (hC : C = 5200) : C = 5200 :=
by sorry

end NUMINAMATH_GPT_camel_cost_is_5200_l1105_110577


namespace NUMINAMATH_GPT_sandy_age_l1105_110516

theorem sandy_age (S M : ℕ) (h1 : M = S + 18) (h2 : S * 9 = M * 7) : S = 63 := by
  sorry

end NUMINAMATH_GPT_sandy_age_l1105_110516


namespace NUMINAMATH_GPT_good_numbers_l1105_110507

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_odd_prime (n : ℕ) : Prop :=
  Prime n ∧ n % 2 = 1

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_divisor (d + 1) (n + 1)

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ n = 1 ∨ is_odd_prime n :=
sorry

end NUMINAMATH_GPT_good_numbers_l1105_110507


namespace NUMINAMATH_GPT_percentage_reduction_l1105_110541

theorem percentage_reduction (original reduced : ℕ) (h₁ : original = 260) (h₂ : reduced = 195) :
  (original - reduced) / original * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l1105_110541


namespace NUMINAMATH_GPT_graph_passes_through_fixed_point_l1105_110562

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x-1)

theorem graph_passes_through_fixed_point (a : ℝ) : f a 1 = 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_graph_passes_through_fixed_point_l1105_110562


namespace NUMINAMATH_GPT_range_of_a_l1105_110589

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1) → f a x1 ≥ g x2) →
  a ≥ -2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1105_110589


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1105_110512

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) : (∀ x : ℝ, f (x + 1) = x^2 + 4*x + 1) → (∀ x : ℝ, f x = x^2 + 2*x - 2) :=
by
  intro h
  sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) : (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) → (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) → (∀ x : ℝ, f x = x + 3) :=
by
  intros h1 h2
  sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) : (∀ x : ℝ, 2 * f x + f (1 / x) = 3 * x) → (∀ x : ℝ, f x = 2 * x - 1 / x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1105_110512


namespace NUMINAMATH_GPT_max_value_y_l1105_110564

noncomputable def max_y (a b c d : ℝ) : ℝ :=
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2

theorem max_value_y {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 10) : max_y a b c d = 40 := 
  sorry

end NUMINAMATH_GPT_max_value_y_l1105_110564


namespace NUMINAMATH_GPT_james_chess_learning_time_l1105_110514

theorem james_chess_learning_time (R : ℝ) 
    (h1 : R + 49 * R + 100 * (R + 49 * R) = 10100) 
    : R = 2 :=
by 
  sorry

end NUMINAMATH_GPT_james_chess_learning_time_l1105_110514


namespace NUMINAMATH_GPT_evan_books_in_ten_years_l1105_110543

def E4 : ℕ := 400
def E_now : ℕ := E4 - 80
def E2 : ℕ := E_now / 2
def E10 : ℕ := 6 * E2 + 120

theorem evan_books_in_ten_years : E10 = 1080 := by
sorry

end NUMINAMATH_GPT_evan_books_in_ten_years_l1105_110543


namespace NUMINAMATH_GPT_value_of_at_20_at_l1105_110563

noncomputable def left_at (x : ℝ) : ℝ := 9 - x
noncomputable def right_at (x : ℝ) : ℝ := x - 9

theorem value_of_at_20_at : right_at (left_at 20) = -20 := by
  sorry

end NUMINAMATH_GPT_value_of_at_20_at_l1105_110563


namespace NUMINAMATH_GPT_companion_value_4164_smallest_N_satisfies_conditions_l1105_110522

-- Define relevant functions
def G (N : ℕ) : ℕ :=
  let digits := [N / 1000 % 10, N / 100 % 10, N / 10 % 10, N % 10]
  digits.sum

def P (N : ℕ) : ℕ :=
  (N / 1000 % 10) * (N / 100 % 10)

def Q (N : ℕ) : ℕ :=
  (N / 10 % 10) * (N % 10)

def companion_value (N : ℕ) : ℚ :=
  |(G N : ℤ) / ((P N : ℤ) - (Q N : ℤ))|

-- Proof problem for part (1)
theorem companion_value_4164 : companion_value 4164 = 3 / 4 := sorry

-- Proof problem for part (2)
theorem smallest_N_satisfies_conditions :
  ∀ (N : ℕ), N > 1000 ∧ N < 10000 ∧ (∀ d, N / 10^d % 10 ≠ 0) ∧ (N / 1000 % 10 + N % 10) % 9 = 0 ∧ G N = 16 ∧ companion_value N = 4 → N = 2527 := sorry

end NUMINAMATH_GPT_companion_value_4164_smallest_N_satisfies_conditions_l1105_110522


namespace NUMINAMATH_GPT_average_marks_is_75_l1105_110590

-- Define the scores for the four tests based on the given conditions.
def first_test : ℕ := 80
def second_test : ℕ := first_test + 10
def third_test : ℕ := 65
def fourth_test : ℕ := third_test

-- Define the total marks scored in the four tests.
def total_marks : ℕ := first_test + second_test + third_test + fourth_test

-- Number of tests.
def num_tests : ℕ := 4

-- Define the average marks scored in the four tests.
def average_marks : ℕ := total_marks / num_tests

-- Prove that the average marks scored in the four tests is 75.
theorem average_marks_is_75 : average_marks = 75 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_is_75_l1105_110590


namespace NUMINAMATH_GPT_heat_released_is_1824_l1105_110555

def ΔH_f_NH3 : ℝ := -46  -- Enthalpy of formation of NH3 in kJ/mol
def ΔH_f_H2SO4 : ℝ := -814  -- Enthalpy of formation of H2SO4 in kJ/mol
def ΔH_f_NH4SO4 : ℝ := -909  -- Enthalpy of formation of (NH4)2SO4 in kJ/mol

def ΔH_rxn : ℝ :=
  2 * ΔH_f_NH4SO4 - (2 * ΔH_f_NH3 + ΔH_f_H2SO4)  -- Reaction enthalpy change

def heat_released : ℝ := 2 * ΔH_rxn  -- Heat released for 4 moles of NH3

theorem heat_released_is_1824 : heat_released = -1824 :=
by
  -- Theorem statement for proving heat released is 1824 kJ
  sorry

end NUMINAMATH_GPT_heat_released_is_1824_l1105_110555


namespace NUMINAMATH_GPT_percentage_of_material_A_in_second_solution_l1105_110534

theorem percentage_of_material_A_in_second_solution 
  (material_A_first_solution : ℝ)
  (material_B_first_solution : ℝ)
  (material_B_second_solution : ℝ)
  (material_A_mixture : ℝ)
  (percentage_first_solution_in_mixture : ℝ)
  (percentage_second_solution_in_mixture : ℝ)
  (total_mixture: ℝ)
  (hyp1 : material_A_first_solution = 20 / 100)
  (hyp2 : material_B_first_solution = 80 / 100)
  (hyp3 : material_B_second_solution = 70 / 100)
  (hyp4 : material_A_mixture = 22 / 100)
  (hyp5 : percentage_first_solution_in_mixture = 80 / 100)
  (hyp6 : percentage_second_solution_in_mixture = 20 / 100)
  (hyp7 : percentage_first_solution_in_mixture + percentage_second_solution_in_mixture = total_mixture)
  : ∃ (x : ℝ), x = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_of_material_A_in_second_solution_l1105_110534


namespace NUMINAMATH_GPT_smallest_b_factors_l1105_110551

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end NUMINAMATH_GPT_smallest_b_factors_l1105_110551


namespace NUMINAMATH_GPT_triangle_area_l1105_110594

theorem triangle_area :
  let line1 (x : ℝ) := 2 * x + 1
  let line2 (x : ℝ) := (16 + x) / 4
  ∃ (base height : ℝ), height = (16 + 2 * base) / 7 ∧ base * height / 2 = 18 / 7 :=
  by
    sorry

end NUMINAMATH_GPT_triangle_area_l1105_110594


namespace NUMINAMATH_GPT_correct_answers_l1105_110591

-- Definitions
variable (C W : ℕ)
variable (h1 : C + W = 120)
variable (h2 : 3 * C - W = 180)

-- Goal statement
theorem correct_answers : C = 75 :=
by
  sorry

end NUMINAMATH_GPT_correct_answers_l1105_110591


namespace NUMINAMATH_GPT_max_a_value_l1105_110568

theorem max_a_value : ∃ a b : ℕ, 1 < a ∧ a < b ∧
  (∀ x y : ℝ, y = -2 * x + 4033 ∧ y = |x - 1| + |x + a| + |x - b| → 
  a = 4031) := sorry

end NUMINAMATH_GPT_max_a_value_l1105_110568


namespace NUMINAMATH_GPT_find_b_when_a_equals_neg10_l1105_110598

theorem find_b_when_a_equals_neg10 
  (ab_k : ∀ a b : ℝ, (a * b) = 675) 
  (sum_60 : ∀ a b : ℝ, (a + b = 60 → a = 3 * b)) 
  (a_eq_neg10 : ∀ a : ℝ, a = -10) : 
  ∃ b : ℝ, b = -67.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_when_a_equals_neg10_l1105_110598


namespace NUMINAMATH_GPT_John_other_trip_length_l1105_110569

theorem John_other_trip_length :
  ∀ (fuel_per_km total_fuel first_trip_length other_trip_length : ℕ),
    fuel_per_km = 5 →
    total_fuel = 250 →
    first_trip_length = 20 →
    total_fuel / fuel_per_km - first_trip_length = other_trip_length →
    other_trip_length = 30 :=
by
  intros fuel_per_km total_fuel first_trip_length other_trip_length h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_John_other_trip_length_l1105_110569


namespace NUMINAMATH_GPT_polynomial_divisibility_l1105_110560

theorem polynomial_divisibility (m : ℤ) : (3 * (-2)^2 + 5 * (-2) + m = 0) ↔ (m = -2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1105_110560


namespace NUMINAMATH_GPT_intersection_is_singleton_l1105_110518

namespace ProofProblem

def M : Set ℤ := {-3, -2, -1}

def N : Set ℤ := {x : ℤ | (x + 2) * (x - 3) < 0}

theorem intersection_is_singleton : M ∩ N = {-1} :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_is_singleton_l1105_110518


namespace NUMINAMATH_GPT_sequence_general_formula_l1105_110581

theorem sequence_general_formula :
  ∃ (a : ℕ → ℕ), 
    (a 1 = 4) ∧ 
    (∀ n : ℕ, a (n + 1) = a n + 3) ∧ 
    (∀ n : ℕ, a n = 3 * n + 1) :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l1105_110581


namespace NUMINAMATH_GPT_hexagon_side_equalities_l1105_110550

variables {A B C D E F : Type}

-- Define the properties and conditions of the problem
noncomputable def convex_hexagon (A B C D E F : Type) : Prop :=
  True -- Since we neglect geometric properties in this abstract.

def parallel (a b : Type) : Prop := True -- placeholder for parallel condition
def equal_length (a b : Type) : Prop := True -- placeholder for length

-- Given conditions
variables (h1 : convex_hexagon A B C D E F)
variables (h2 : parallel AB DE)
variables (h3 : parallel BC FA)
variables (h4 : parallel CD FA)
variables (h5 : equal_length AB DE)

-- Statement to prove
theorem hexagon_side_equalities : equal_length BC DE ∧ equal_length CD FA := sorry

end NUMINAMATH_GPT_hexagon_side_equalities_l1105_110550


namespace NUMINAMATH_GPT_solve_inequality_l1105_110559

theorem solve_inequality (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 6) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 2) :
  (x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Ioo 2 6)) → 
  ((x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Icc 3 5))) :=
by 
  introv h
  sorry

end NUMINAMATH_GPT_solve_inequality_l1105_110559


namespace NUMINAMATH_GPT_car_speed_is_100_l1105_110531

def avg_speed (d1 d2 t: ℕ) := (d1 + d2) / t = 80

theorem car_speed_is_100 
  (x : ℕ)
  (speed_second_hour : ℕ := 60)
  (total_time : ℕ := 2)
  (h : avg_speed x speed_second_hour total_time):
  x = 100 :=
by
  unfold avg_speed at h
  sorry

end NUMINAMATH_GPT_car_speed_is_100_l1105_110531


namespace NUMINAMATH_GPT_rearranged_number_divisible_by_27_l1105_110593

theorem rearranged_number_divisible_by_27 (n m : ℕ) (hn : m = 3 * n) 
  (hdigits : ∀ a b : ℕ, (a ∈ n.digits 10 ↔ b ∈ m.digits 10)) : 27 ∣ m :=
sorry

end NUMINAMATH_GPT_rearranged_number_divisible_by_27_l1105_110593


namespace NUMINAMATH_GPT_combined_cost_price_l1105_110561

def cost_price_A : ℕ := (120 + 60) / 2
def cost_price_B : ℕ := (200 + 100) / 2
def cost_price_C : ℕ := (300 + 180) / 2

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C

theorem combined_cost_price :
  total_cost_price = 480 := by
  sorry

end NUMINAMATH_GPT_combined_cost_price_l1105_110561


namespace NUMINAMATH_GPT_original_number_is_24_l1105_110544

def number_parts (x y original_number : ℝ) : Prop :=
  7 * x + 5 * y = 146 ∧ x = 13 ∧ original_number = x + y

theorem original_number_is_24 :
  ∃ (x y original_number : ℝ), number_parts x y original_number ∧ original_number = 24 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_24_l1105_110544


namespace NUMINAMATH_GPT_radius_of_circle_l1105_110597

theorem radius_of_circle (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ a) : 
  ∃ R, R = (b - a) / 2 ∨ R = (b + a) / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_of_circle_l1105_110597


namespace NUMINAMATH_GPT_smallest_possible_b_l1105_110501

theorem smallest_possible_b
  (a c b : ℤ)
  (h1 : a < c)
  (h2 : c < b)
  (h3 : c = (a + b) / 2)
  (h4 : b^2 / c = a) :
  b = 2 :=
sorry

end NUMINAMATH_GPT_smallest_possible_b_l1105_110501


namespace NUMINAMATH_GPT_solve_for_x_l1105_110574

theorem solve_for_x (x : ℤ) (h : 13 * x + 14 * x + 17 * x + 11 = 143) : x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1105_110574


namespace NUMINAMATH_GPT_rectangleY_has_tileD_l1105_110553

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define tiles
def TileA : Tile := { top := 6, right := 3, bottom := 5, left := 2 }
def TileB : Tile := { top := 3, right := 6, bottom := 2, left := 5 }
def TileC : Tile := { top := 5, right := 7, bottom := 1, left := 2 }
def TileD : Tile := { top := 2, right := 5, bottom := 6, left := 3 }

-- Define rectangles (positioning)
inductive Rectangle
| W | X | Y | Z

-- Define which tile is in Rectangle Y
def tileInRectangleY : Tile → Prop :=
  fun t => t = TileD

-- Statement to prove
theorem rectangleY_has_tileD : tileInRectangleY TileD :=
by
  -- The final statement to be proven, skipping the proof itself with sorry
  sorry

end NUMINAMATH_GPT_rectangleY_has_tileD_l1105_110553


namespace NUMINAMATH_GPT_factor_squared_of_symmetric_poly_l1105_110554

theorem factor_squared_of_symmetric_poly (P : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ)
  (h_symm : ∀ x y, P x y = P y x)
  (h_factor : ∀ x y, (x - y) ∣ P x y) :
  ∀ x y, (x - y) ^ 2 ∣ P x y := 
sorry

end NUMINAMATH_GPT_factor_squared_of_symmetric_poly_l1105_110554


namespace NUMINAMATH_GPT_find_a100_l1105_110511

theorem find_a100 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, a n = (2 * (S n)^2) / (2 * (S n) - 1))
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 100 = -2 / 39203 := 
sorry

-- Explanation of the statement:
-- 'theorem find_a100': We define a theorem to find a_100.
-- 'a : ℕ → ℝ': a is a sequence of real numbers.
-- 'S : ℕ → ℝ': S is a sequence representing the sum of the first n terms.
-- 'h1' to 'h3': Given conditions from the problem statement.
-- 'a 100 = -2 / 39203' : The statement to prove.

end NUMINAMATH_GPT_find_a100_l1105_110511


namespace NUMINAMATH_GPT_days_worked_per_week_l1105_110546

theorem days_worked_per_week (total_toys_per_week toys_produced_each_day : ℕ) 
  (h1 : total_toys_per_week = 5505)
  (h2 : toys_produced_each_day = 1101)
  : total_toys_per_week / toys_produced_each_day = 5 :=
  by
    sorry

end NUMINAMATH_GPT_days_worked_per_week_l1105_110546


namespace NUMINAMATH_GPT_functional_equation_solution_l1105_110558

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1105_110558


namespace NUMINAMATH_GPT_variable_value_l1105_110576

theorem variable_value (w x v : ℝ) (h1 : 5 / w + 5 / x = 5 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) : v = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_variable_value_l1105_110576


namespace NUMINAMATH_GPT_chocolate_bars_in_box_l1105_110557

theorem chocolate_bars_in_box (x : ℕ) (h1 : 2 * (x - 4) = 18) : x = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_chocolate_bars_in_box_l1105_110557


namespace NUMINAMATH_GPT_smallest_number_condition_l1105_110519

theorem smallest_number_condition :
  ∃ n, 
  (n > 0) ∧ 
  (∀ k, k < n → (n - 3) % 12 = 0 ∧ (n - 3) % 16 = 0 ∧ (n - 3) % 18 = 0 ∧ (n - 3) % 21 = 0 ∧ (n - 3) % 28 = 0 → k = 0) ∧
  (n - 3) % 12 = 0 ∧
  (n - 3) % 16 = 0 ∧
  (n - 3) % 18 = 0 ∧
  (n - 3) % 21 = 0 ∧
  (n - 3) % 28 = 0 ∧
  n = 1011 :=
sorry

end NUMINAMATH_GPT_smallest_number_condition_l1105_110519


namespace NUMINAMATH_GPT_largest_n_divisible_l1105_110523

theorem largest_n_divisible (n : ℕ) (h : (n ^ 3 + 144) % (n + 12) = 0) : n ≤ 84 :=
sorry

end NUMINAMATH_GPT_largest_n_divisible_l1105_110523


namespace NUMINAMATH_GPT_sum_of_a_b_l1105_110520

-- Define the conditions in Lean
def a : ℝ := 1
def b : ℝ := 1

-- Define the proof statement
theorem sum_of_a_b : a + b = 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_a_b_l1105_110520


namespace NUMINAMATH_GPT_temperature_lower_than_freezing_point_is_minus_three_l1105_110502

-- Define the freezing point of water
def freezing_point := 0 -- in degrees Celsius

-- Define the temperature lower by a certain value
def lower_temperature (t: Int) (delta: Int) := t - delta

-- State the theorem to be proved
theorem temperature_lower_than_freezing_point_is_minus_three:
  lower_temperature freezing_point 3 = -3 := by
  sorry

end NUMINAMATH_GPT_temperature_lower_than_freezing_point_is_minus_three_l1105_110502


namespace NUMINAMATH_GPT_max_value_of_exp_l1105_110509

theorem max_value_of_exp (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : 
  a^2 * b^3 * c ≤ 27 / 16 := 
  sorry

end NUMINAMATH_GPT_max_value_of_exp_l1105_110509


namespace NUMINAMATH_GPT_marble_group_size_l1105_110582

-- Define the conditions
def num_marbles : ℕ := 220
def future_people (x : ℕ) : ℕ := x + 2
def marbles_per_person (x : ℕ) : ℕ := num_marbles / x
def marbles_if_2_more (x : ℕ) : ℕ := num_marbles / future_people x

-- Statement of the theorem
theorem marble_group_size (x : ℕ) :
  (marbles_per_person x - 1 = marbles_if_2_more x) ↔ x = 20 :=
sorry

end NUMINAMATH_GPT_marble_group_size_l1105_110582


namespace NUMINAMATH_GPT_number_of_ears_pierced_l1105_110533

-- Definitions for the conditions
def nosePiercingPrice : ℝ := 20
def earPiercingPrice := nosePiercingPrice + 0.5 * nosePiercingPrice
def totalAmountMade : ℝ := 390
def nosesPierced : ℕ := 6
def totalFromNoses := nosesPierced * nosePiercingPrice
def totalFromEars := totalAmountMade - totalFromNoses

-- The proof statement
theorem number_of_ears_pierced : totalFromEars / earPiercingPrice = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_ears_pierced_l1105_110533


namespace NUMINAMATH_GPT_find_missing_number_l1105_110538

theorem find_missing_number (x : ℕ) :
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1105_110538


namespace NUMINAMATH_GPT_evaluate_expression_l1105_110539

theorem evaluate_expression :
  2 - (-3) - 4 + (-5) - 6 + 7 = -3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1105_110539


namespace NUMINAMATH_GPT_final_value_of_S_l1105_110532

theorem final_value_of_S :
  ∀ (S n : ℕ), S = 1 → n = 1 →
  (∀ S n : ℕ, ¬ n > 3 → 
    (∃ S' n' : ℕ, S' = S + 2 * n ∧ n' = n + 1 ∧ 
      (∀ S n : ℕ, n > 3 → S' = 13))) :=
by 
  intros S n hS hn
  simp [hS, hn]
  sorry

end NUMINAMATH_GPT_final_value_of_S_l1105_110532


namespace NUMINAMATH_GPT_num_of_adults_l1105_110517

def students : ℕ := 22
def vans : ℕ := 3
def capacity_per_van : ℕ := 8

theorem num_of_adults : (vans * capacity_per_van) - students = 2 := by
  sorry

end NUMINAMATH_GPT_num_of_adults_l1105_110517


namespace NUMINAMATH_GPT_sales_fraction_l1105_110572

theorem sales_fraction (A D : ℝ) (h : D = 2 * A) : D / (11 * A + D) = 2 / 13 :=
by
  sorry

end NUMINAMATH_GPT_sales_fraction_l1105_110572


namespace NUMINAMATH_GPT_min_value_inequality_l1105_110536

theorem min_value_inequality (p q r : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) (h₂ : 0 < r) :
  ( 3 * r / (p + 2 * q) + 3 * p / (2 * r + q) + 2 * q / (p + r) ) ≥ (29 / 6) := 
sorry

end NUMINAMATH_GPT_min_value_inequality_l1105_110536


namespace NUMINAMATH_GPT_eve_age_l1105_110542

variable (E : ℕ)

theorem eve_age (h1 : ∀ (a : ℕ), a = 9 → (E + 1) = 3 * (9 - 4)) : E = 14 := 
by
  have h2 : 9 - 4 = 5 := by norm_num
  have h3 : 3 * 5 = 15 := by norm_num
  have h4 : (E + 1) = 15 := h1 9 rfl
  linarith

end NUMINAMATH_GPT_eve_age_l1105_110542


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l1105_110515

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  ( ( (a - b) * (a - c) / (a + b + c) ) + 
    ( (b - c) * (b - d) / (b + c + d) ) + 
    ( (c - d) * (c - a) / (c + d + a) ) + 
    ( (d - a) * (d - b) / (d + a + b) ) ) ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l1105_110515


namespace NUMINAMATH_GPT_Jane_buys_three_bagels_l1105_110571

theorem Jane_buys_three_bagels (b m c : ℕ) (h1 : b + m + c = 5) (h2 : 80 * b + 60 * m + 100 * c = 400) : b = 3 := 
sorry

end NUMINAMATH_GPT_Jane_buys_three_bagels_l1105_110571


namespace NUMINAMATH_GPT_parallel_vectors_l1105_110530

def vec_a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem parallel_vectors (x : ℝ) : vec_a x = (2, 4) → x = 2 := by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l1105_110530


namespace NUMINAMATH_GPT_find_n_modulo_l1105_110599

theorem find_n_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 11) (h3 : n ≡ 15827 [ZMOD 12]) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_n_modulo_l1105_110599


namespace NUMINAMATH_GPT_product_of_ab_l1105_110521

theorem product_of_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 13) : a * b = -6 :=
by
  sorry

end NUMINAMATH_GPT_product_of_ab_l1105_110521
