import Mathlib

namespace NUMINAMATH_GPT_henry_friend_fireworks_l1387_138739

-- Definitions of variables and conditions
variable 
  (F : ℕ) -- Number of fireworks Henry's friend bought

-- Main theorem statement
theorem henry_friend_fireworks (h1 : 6 + 2 + F = 11) : F = 3 :=
by
  sorry

end NUMINAMATH_GPT_henry_friend_fireworks_l1387_138739


namespace NUMINAMATH_GPT_trisha_bought_amount_initially_l1387_138755

-- Define the amounts spent on each item
def meat : ℕ := 17
def chicken : ℕ := 22
def veggies : ℕ := 43
def eggs : ℕ := 5
def dogs_food : ℕ := 45
def amount_left : ℕ := 35

-- Define the total amount spent
def total_spent : ℕ := meat + chicken + veggies + eggs + dogs_food

-- Define the amount brought at the beginning
def amount_brought_at_beginning : ℕ := total_spent + amount_left

-- Theorem stating the amount Trisha brought at the beginning is 167
theorem trisha_bought_amount_initially : amount_brought_at_beginning = 167 := by
  -- Formal proof would go here, we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_trisha_bought_amount_initially_l1387_138755


namespace NUMINAMATH_GPT_total_goals_by_other_players_l1387_138754

theorem total_goals_by_other_players (total_players goals_season games_played : ℕ)
  (third_players_goals avg_goals_per_third : ℕ)
  (h1 : total_players = 24)
  (h2 : goals_season = 150)
  (h3 : games_played = 15)
  (h4 : third_players_goals = total_players / 3)
  (h5 : avg_goals_per_third = 1)
  : (goals_season - (third_players_goals * avg_goals_per_third * games_played)) = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_goals_by_other_players_l1387_138754


namespace NUMINAMATH_GPT_book_pages_l1387_138719

theorem book_pages (x : ℕ) : 
  (x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15) - (1/3 * ((x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15)) + 18) = 62 →
  x = 240 :=
by
  -- This is where the proof would go, but it's omitted for this task.
  sorry

end NUMINAMATH_GPT_book_pages_l1387_138719


namespace NUMINAMATH_GPT_triangle_perimeter_l1387_138723

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : a = 3) (h2 : b = 3) 
    (h3 : c^2 = a * Real.cos B + b * Real.cos A) : 
    a + b + c = 7 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1387_138723


namespace NUMINAMATH_GPT_range_of_a_monotonically_decreasing_l1387_138799

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv (f a) x ≤ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_monotonically_decreasing_l1387_138799


namespace NUMINAMATH_GPT_consecutive_arithmetic_sequence_l1387_138721

theorem consecutive_arithmetic_sequence (a b c : ℝ) 
  (h : (2 * b - a)^2 + (2 * b - c)^2 = 2 * (2 * b^2 - a * c)) : 
  2 * b = a + c :=
by
  sorry

end NUMINAMATH_GPT_consecutive_arithmetic_sequence_l1387_138721


namespace NUMINAMATH_GPT_total_sample_any_candy_42_percent_l1387_138717

-- Define percentages as rational numbers to avoid dealing with decimals directly
def percent_of_caught_A : ℚ := 12 / 100
def percent_of_not_caught_A : ℚ := 7 / 100
def percent_of_caught_B : ℚ := 5 / 100
def percent_of_not_caught_B : ℚ := 6 / 100
def percent_of_caught_C : ℚ := 9 / 100
def percent_of_not_caught_C : ℚ := 3 / 100

-- Sum up the percentages for those caught and not caught for each type of candy
def total_percent_A : ℚ := percent_of_caught_A + percent_of_not_caught_A
def total_percent_B : ℚ := percent_of_caught_B + percent_of_not_caught_B
def total_percent_C : ℚ := percent_of_caught_C + percent_of_not_caught_C

-- Sum of the total percentages for all types
def total_percent_sample_any_candy : ℚ := total_percent_A + total_percent_B + total_percent_C

theorem total_sample_any_candy_42_percent :
  total_percent_sample_any_candy = 42 / 100 :=
by
  sorry

end NUMINAMATH_GPT_total_sample_any_candy_42_percent_l1387_138717


namespace NUMINAMATH_GPT_age_problem_l1387_138726

theorem age_problem (A B C D E : ℕ)
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : D = C / 2)
  (h4 : E = D - 3)
  (h5 : A + B + C + D + E = 52) : B = 16 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l1387_138726


namespace NUMINAMATH_GPT_pyramid_surface_area_l1387_138720

-- Definitions based on conditions
def upper_base_edge_length : ℝ := 2
def lower_base_edge_length : ℝ := 4
def side_edge_length : ℝ := 2

-- Problem statement in Lean
theorem pyramid_surface_area :
  let slant_height := Real.sqrt ((side_edge_length ^ 2) - (1 ^ 2))
  let perimeter_base := (4 * upper_base_edge_length) + (4 * lower_base_edge_length)
  let lsa := (perimeter_base * slant_height) / 2
  let total_surface_area := lsa + (upper_base_edge_length ^ 2) + (lower_base_edge_length ^ 2)
  total_surface_area = 10 * Real.sqrt 3 + 20 := sorry

end NUMINAMATH_GPT_pyramid_surface_area_l1387_138720


namespace NUMINAMATH_GPT_rectangle_perimeter_l1387_138748

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b) = 4 * (2 * a + 2 * b) - 12) :
    (2 * (a + b) = 72) ∨ (2 * (a + b) = 100) := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1387_138748


namespace NUMINAMATH_GPT_total_worth_of_stock_l1387_138776

theorem total_worth_of_stock (X : ℝ) :
  (0.30 * 0.10 * X + 0.40 * -0.05 * X + 0.30 * -0.10 * X = -500) → X = 25000 :=
by
  intro h
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_total_worth_of_stock_l1387_138776


namespace NUMINAMATH_GPT_find_value_l1387_138706

theorem find_value
  (x a y b z c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_l1387_138706


namespace NUMINAMATH_GPT_range_of_k_l1387_138789

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x < 3 → x - k < 2 * k) → 1 ≤ k :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1387_138789


namespace NUMINAMATH_GPT_calculate_expression_l1387_138744

theorem calculate_expression : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1387_138744


namespace NUMINAMATH_GPT_Robert_salary_loss_l1387_138715

-- Define the conditions as hypotheses
variable (S : ℝ) (decrease_percent increase_percent : ℝ)
variable (decrease_percent_eq : decrease_percent = 0.6)
variable (increase_percent_eq : increase_percent = 0.6)

-- Define the problem statement to prove that Robert loses 36% of his salary.
theorem Robert_salary_loss (S : ℝ) (decrease_percent increase_percent : ℝ) 
  (decrease_percent_eq : decrease_percent = 0.6) 
  (increase_percent_eq : increase_percent = 0.6) :
  let new_salary := S * (1 - decrease_percent)
  let increased_salary := new_salary * (1 + increase_percent)
  let loss_percentage := (S - increased_salary) / S * 100 
  loss_percentage = 36 := 
by
  sorry

end NUMINAMATH_GPT_Robert_salary_loss_l1387_138715


namespace NUMINAMATH_GPT_total_pounds_of_peppers_l1387_138725

-- Definitions based on the conditions
def greenPeppers : ℝ := 0.3333333333333333
def redPeppers : ℝ := 0.3333333333333333

-- Goal statement expressing the problem
theorem total_pounds_of_peppers :
  greenPeppers + redPeppers = 0.6666666666666666 := 
by
  sorry

end NUMINAMATH_GPT_total_pounds_of_peppers_l1387_138725


namespace NUMINAMATH_GPT_pyramid_height_l1387_138716

def height_of_pyramid (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem pyramid_height (n : ℕ) : height_of_pyramid n = 2 * (n - 1) :=
by
  -- The proof would typically go here
  sorry

end NUMINAMATH_GPT_pyramid_height_l1387_138716


namespace NUMINAMATH_GPT_FC_value_l1387_138713

theorem FC_value (DC CB AD FC : ℝ) (h1 : DC = 10) (h2 : CB = 9)
  (h3 : AB = (1 / 3) * AD) (h4 : ED = (3 / 4) * AD) : FC = 13.875 := by
  sorry

end NUMINAMATH_GPT_FC_value_l1387_138713


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1387_138775

variables {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def increasing_on (f : R → R) (S : Set R) : Prop :=
∀ ⦃x y⦄, x ∈ S → y ∈ S → x < y → f x < f y

theorem solution_set_of_inequality
  {f : R → R}
  (h_odd : odd_function f)
  (h_neg_one : f (-1) = 0)
  (h_increasing : increasing_on f {x : R | x > 0}) :
  {x : R | x * f x > 0} = {x : R | x < -1} ∪ {x : R | x > 1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1387_138775


namespace NUMINAMATH_GPT_games_played_l1387_138749

def total_points : ℝ := 120.0
def points_per_game : ℝ := 12.0
def num_games : ℝ := 10.0

theorem games_played : (total_points / points_per_game) = num_games := 
by 
  sorry

end NUMINAMATH_GPT_games_played_l1387_138749


namespace NUMINAMATH_GPT_roots_equation_1352_l1387_138707

theorem roots_equation_1352 {c d : ℝ} (hc : c^2 - 6 * c + 8 = 0) (hd : d^2 - 6 * d + 8 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 1352 :=
by
  sorry

end NUMINAMATH_GPT_roots_equation_1352_l1387_138707


namespace NUMINAMATH_GPT_trucks_on_lot_l1387_138735

-- We'll state the conditions as hypotheses and then conclude the total number of trucks.
theorem trucks_on_lot (T : ℕ)
  (h₁ : ∀ N : ℕ, 50 ≤ N ∧ N ≤ 20 → N / 2 = 10)
  (h₂ : T ≥ 20 + 10): T = 30 :=
sorry

end NUMINAMATH_GPT_trucks_on_lot_l1387_138735


namespace NUMINAMATH_GPT_calc_value_l1387_138734

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem calc_value :
  ((diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2))) = -13 / 28 :=
by sorry

end NUMINAMATH_GPT_calc_value_l1387_138734


namespace NUMINAMATH_GPT_trig_identity_l1387_138718

theorem trig_identity : 
  Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1387_138718


namespace NUMINAMATH_GPT_range_of_b_l1387_138770

theorem range_of_b (y : ℝ) (b : ℝ) (h1 : |y - 2| + |y - 5| < b) (h2 : b > 1) : b > 3 := 
sorry

end NUMINAMATH_GPT_range_of_b_l1387_138770


namespace NUMINAMATH_GPT_square_side_length_l1387_138763

theorem square_side_length (A : ℝ) (h : A = 625) : ∃ l : ℝ, l^2 = A ∧ l = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_side_length_l1387_138763


namespace NUMINAMATH_GPT_probability_of_selecting_female_l1387_138759

theorem probability_of_selecting_female (total_students female_students male_students : ℕ)
  (h_total : total_students = female_students + male_students)
  (h_female : female_students = 3)
  (h_male : male_students = 1) :
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_female_l1387_138759


namespace NUMINAMATH_GPT_perpendicular_condition_centroid_coordinates_l1387_138777

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 0}
def B : Point := {x := 4, y := 0}
def C (c : ℝ) : Point := {x := 0, y := c}

def vec (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y}

def dot_product (P Q : Point) : ℝ :=
  P.x * Q.x + P.y * Q.y

theorem perpendicular_condition (c : ℝ) (h : dot_product (vec A (C c)) (vec B (C c)) = 0) :
  c = 2 ∨ c = -2 :=
by
  -- proof to be filled in
  sorry

theorem centroid_coordinates (c : ℝ) (h : c = 2 ∨ c = -2) :
  (c = 2 → Point.mk 1 (2 / 3) = Point.mk 1 (2 / 3)) ∧
  (c = -2 → Point.mk 1 (-2 / 3) = Point.mk 1 (-2 / 3)) :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_perpendicular_condition_centroid_coordinates_l1387_138777


namespace NUMINAMATH_GPT_pure_imaginary_condition_fourth_quadrant_condition_l1387_138783

theorem pure_imaginary_condition (m : ℝ) (h1: m * (m - 1) = 0) (h2: m ≠ 1) : m = 0 :=
by
  sorry

theorem fourth_quadrant_condition (m : ℝ) (h3: m + 1 > 0) (h4: m^2 - 1 < 0) : -1 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_condition_fourth_quadrant_condition_l1387_138783


namespace NUMINAMATH_GPT_common_difference_is_one_l1387_138778

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions given in the problem
axiom h1 : a 1 ^ 2 + a 10 ^ 2 = 101
axiom h2 : a 5 + a 6 = 11
axiom h3 : ∀ n m, n < m → a n < a m
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n+1) = a n + d

-- Theorem stating the common difference d is 1
theorem common_difference_is_one : is_arithmetic_sequence a d → d = 1 := 
by
  sorry

end NUMINAMATH_GPT_common_difference_is_one_l1387_138778


namespace NUMINAMATH_GPT_man_l1387_138703

theorem man's_age_twice_son_in_2_years 
  (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = 38) (h3 : M = S + 20) : 
  ∃ X : ℕ, (M + X = 2 * (S + X)) ∧ X = 2 :=
by
  sorry

end NUMINAMATH_GPT_man_l1387_138703


namespace NUMINAMATH_GPT_kevin_watermelons_l1387_138738

theorem kevin_watermelons (w1 w2 w_total : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) (h_total : w_total = 14.02) : 
  w1 + w2 = w_total → 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_kevin_watermelons_l1387_138738


namespace NUMINAMATH_GPT_find_value_at_l1387_138704

-- Defining the function f
variable (f : ℝ → ℝ)

-- Conditions
-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 4
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 4) = f x

-- Condition 3: In the interval [0,1], f(x) = 3x
def definition_on_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- Statement to prove
theorem find_value_at (f : ℝ → ℝ) 
  (odd_f : odd_function f) 
  (periodic_f : periodic_function f) 
  (def_on_interval : definition_on_interval f) :
  f 11.5 = -1.5 := by 
  sorry

end NUMINAMATH_GPT_find_value_at_l1387_138704


namespace NUMINAMATH_GPT_speed_of_train_l1387_138710

-- Define the conditions
def length_of_train : ℕ := 240
def length_of_bridge : ℕ := 150
def time_to_cross : ℕ := 20

-- Compute the expected speed of the train
def expected_speed : ℝ := 19.5

-- The statement that needs to be proven
theorem speed_of_train : (length_of_train + length_of_bridge) / time_to_cross = expected_speed := by
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_speed_of_train_l1387_138710


namespace NUMINAMATH_GPT_diagonal_of_rectangle_l1387_138747

theorem diagonal_of_rectangle (l w d : ℝ) (h_length : l = 15) (h_area : l * w = 120) (h_diagonal : d^2 = l^2 + w^2) : d = 17 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_rectangle_l1387_138747


namespace NUMINAMATH_GPT_tangent_line_hyperbola_eq_l1387_138790

noncomputable def tangent_line_ellipse (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0) 
  (h_ell : x0 ^ 2 / a ^ 2 + y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1

noncomputable def tangent_line_hyperbola (a b x0 y0 x y : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h_hyp : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1

theorem tangent_line_hyperbola_eq (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
  (h_ellipse_tangent : tangent_line_ellipse a b x0 y0 x y h1 h2 h3 (by sorry))
  (h_hyperbola : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : 
  tangent_line_hyperbola a b x0 y0 x y h3 h2 h_hyperbola :=
by sorry

end NUMINAMATH_GPT_tangent_line_hyperbola_eq_l1387_138790


namespace NUMINAMATH_GPT_dvd_cost_l1387_138794

-- Given conditions
def vhs_trade_in_value : Int := 2
def number_of_movies : Int := 100
def total_replacement_cost : Int := 800

-- Statement to prove
theorem dvd_cost :
  ((number_of_movies * vhs_trade_in_value) + (number_of_movies * 6) = total_replacement_cost) :=
by
  sorry

end NUMINAMATH_GPT_dvd_cost_l1387_138794


namespace NUMINAMATH_GPT_solve_system_of_equations_l1387_138779

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1387_138779


namespace NUMINAMATH_GPT_cos2_alpha_plus_2sin2_alpha_l1387_138772

theorem cos2_alpha_plus_2sin2_alpha {α : ℝ} (h : Real.tan α = 3 / 4) : 
    Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_cos2_alpha_plus_2sin2_alpha_l1387_138772


namespace NUMINAMATH_GPT_Cannot_Halve_Triangles_With_Diagonals_l1387_138728

structure Polygon where
  vertices : Nat
  edges : Nat

def is_convex (n : Nat) (P : Polygon) : Prop :=
  P.vertices = n ∧ P.edges = n

def non_intersecting_diagonals (P : Polygon) : Prop :=
  -- Assuming a placeholder for the actual non-intersecting diagonals condition
  true

def count_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  P.vertices - 2 -- This is the simplification used for counting triangles

def count_all_diagonals_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  -- Placeholder for function to count triangles formed exclusively by diagonals
  1000

theorem Cannot_Halve_Triangles_With_Diagonals (P : Polygon) (h : is_convex 2002 P) (d : non_intersecting_diagonals P) :
  count_triangles P d = 2000 → ¬ (count_all_diagonals_triangles P d = 1000) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_Cannot_Halve_Triangles_With_Diagonals_l1387_138728


namespace NUMINAMATH_GPT_mango_coconut_ratio_l1387_138787

open Function

theorem mango_coconut_ratio
  (mango_trees : ℕ)
  (coconut_trees : ℕ)
  (total_trees : ℕ)
  (R : ℚ)
  (H1 : mango_trees = 60)
  (H2 : coconut_trees = R * 60 - 5)
  (H3 : total_trees = 85)
  (H4 : total_trees = mango_trees + coconut_trees) :
  R = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_mango_coconut_ratio_l1387_138787


namespace NUMINAMATH_GPT_melanie_attended_games_l1387_138702

theorem melanie_attended_games 
(missed_games total_games attended_games : ℕ) 
(h1 : total_games = 64) 
(h2 : missed_games = 32)
(h3 : attended_games = total_games - missed_games) 
: attended_games = 32 :=
by sorry

end NUMINAMATH_GPT_melanie_attended_games_l1387_138702


namespace NUMINAMATH_GPT_expression_value_l1387_138745

theorem expression_value : (36 + 9) ^ 2 - (9 ^ 2 + 36 ^ 2) = -1894224 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1387_138745


namespace NUMINAMATH_GPT_slower_train_speed_l1387_138780

-- Define the given conditions
def speed_faster_train : ℝ := 50  -- km/h
def length_faster_train : ℝ := 75.006  -- meters
def passing_time : ℝ := 15  -- seconds

-- Conversion factor
def mps_to_kmph : ℝ := 3.6

-- Define the problem to be proved
theorem slower_train_speed : 
  ∃ speed_slower_train : ℝ, 
    speed_slower_train = speed_faster_train - (75.006 / 15) * mps_to_kmph := 
  by
    exists 31.99856
    sorry

end NUMINAMATH_GPT_slower_train_speed_l1387_138780


namespace NUMINAMATH_GPT_ratio_of_art_to_math_books_l1387_138782

-- The conditions provided
def total_budget : ℝ := 500
def price_math_book : ℝ := 20
def num_math_books : ℕ := 4
def num_art_books : ℕ := num_math_books
def price_art_book : ℝ := 20
def num_science_books : ℕ := num_math_books + 6
def price_science_book : ℝ := 10
def cost_music_books : ℝ := 160

-- Desired proof statement
theorem ratio_of_art_to_math_books : num_art_books / num_math_books = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_art_to_math_books_l1387_138782


namespace NUMINAMATH_GPT_andy_older_than_rahim_l1387_138730

-- Define Rahim's current age
def Rahim_current_age : ℕ := 6

-- Define Andy's age in 5 years
def Andy_age_in_5_years : ℕ := 2 * Rahim_current_age

-- Define Andy's current age
def Andy_current_age : ℕ := Andy_age_in_5_years - 5

-- Define the difference in age between Andy and Rahim right now
def age_difference : ℕ := Andy_current_age - Rahim_current_age

-- Theorem stating the age difference between Andy and Rahim right now is 1 year
theorem andy_older_than_rahim : age_difference = 1 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_andy_older_than_rahim_l1387_138730


namespace NUMINAMATH_GPT_middle_number_is_10_l1387_138758

theorem middle_number_is_10 (A B C : ℝ) (h1 : B - C = A - B) (h2 : A * B = 85) (h3 : B * C = 115) : B = 10 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_is_10_l1387_138758


namespace NUMINAMATH_GPT_line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l1387_138796

theorem line_through_point_parallel_to_given_line :
  ∃ c : ℤ, (∀ x y : ℤ, 2 * x + 3 * y + c = 0 ↔ (x, y) = (2, 1)) ∧ c = -7 :=
sorry

theorem line_through_point_sum_intercepts_is_minus_four :
  ∃ (a b : ℤ), (∀ x y : ℤ, (x / a) + (y / b) = 1 ↔ (x, y) = (-3, 1)) ∧ (a + b = -4) ∧ 
  ((a = -6 ∧ b = 2) ∨ (a = -2 ∧ b = -2)) ∧ 
  ((∀ x y : ℤ, x - 3 * y + 6 = 0 ↔ (x, y) = (-3, 1)) ∨ 
  (∀ x y : ℤ, x + y + 2 = 0 ↔ (x, y) = (-3, 1))) :=
sorry

end NUMINAMATH_GPT_line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l1387_138796


namespace NUMINAMATH_GPT_value_of_a_2015_l1387_138727

def a : ℕ → Int
| 0 => 1
| 1 => 5
| n+2 => a (n+1) - a n

theorem value_of_a_2015 : a 2014 = -5 := by
  sorry

end NUMINAMATH_GPT_value_of_a_2015_l1387_138727


namespace NUMINAMATH_GPT_polygon_perimeter_eq_21_l1387_138788

-- Definitions and conditions from the given problem
def rectangle_side_a := 6
def rectangle_side_b := 4
def triangle_hypotenuse := 5

-- The combined polygon perimeter proof statement
theorem polygon_perimeter_eq_21 :
  let rectangle_perimeter := 2 * (rectangle_side_a + rectangle_side_b)
  let adjusted_perimeter := rectangle_perimeter - rectangle_side_b + triangle_hypotenuse
  adjusted_perimeter = 21 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end NUMINAMATH_GPT_polygon_perimeter_eq_21_l1387_138788


namespace NUMINAMATH_GPT_factorize_expression_l1387_138767

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1387_138767


namespace NUMINAMATH_GPT_expected_value_dodecahedral_die_l1387_138781

-- Define the faces of the die
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the scoring rule
def score (n : ℕ) : ℕ :=
  if n ≤ 6 then 2 * n else n

-- The probability of each face
def prob : ℚ := 1 / 12

-- Calculate the expected value
noncomputable def expected_value : ℚ :=
  prob * (score 1 + score 2 + score 3 + score 4 + score 5 + score 6 + 
          score 7 + score 8 + score 9 + score 10 + score 11 + score 12)

-- State the theorem to be proved
theorem expected_value_dodecahedral_die : expected_value = 8.25 := 
  sorry

end NUMINAMATH_GPT_expected_value_dodecahedral_die_l1387_138781


namespace NUMINAMATH_GPT_index_difference_l1387_138760

noncomputable def index_females (n k1 k2 k3 : ℕ) : ℚ :=
  ((n - k1 + k2 : ℚ) / n) * (1 + k3 / 10)

noncomputable def index_males (n k1 l1 l2 : ℕ) : ℚ :=
  ((n - (n - k1) + l1 : ℚ) / n) * (1 + l2 / 10)

theorem index_difference (n k1 k2 k3 l1 l2 : ℕ)
  (h_n : n = 35) (h_k1 : k1 = 15) (h_k2 : k2 = 5) (h_k3 : k3 = 8)
  (h_l1 : l1 = 6) (h_l2 : l2 = 10) : 
  index_females n k1 k2 k3 - index_males n k1 l1 l2 = 3 / 35 :=
by
  sorry

end NUMINAMATH_GPT_index_difference_l1387_138760


namespace NUMINAMATH_GPT_solve_system_eqns_l1387_138753

theorem solve_system_eqns (x y : ℚ) 
    (h1 : (x - 30) / 3 = (2 * y + 7) / 4)
    (h2 : x - y = 10) :
  x = -81 / 2 ∧ y = -101 / 2 := 
sorry

end NUMINAMATH_GPT_solve_system_eqns_l1387_138753


namespace NUMINAMATH_GPT_quadratic_inequality_range_of_k_l1387_138714

theorem quadratic_inequality (a b x : ℝ) (h1 : a = 1) (h2 : b > 1) :
  (a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
sorry

theorem range_of_k (x y k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1/x) + (2/y) = 1) (h4 : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_range_of_k_l1387_138714


namespace NUMINAMATH_GPT_product_of_six_numbers_l1387_138729

theorem product_of_six_numbers (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x^3 * y^2 = 108) : 
  x * y * (x * y) * (x^2 * y) * (x^3 * y^2) * (x^5 * y^3) = 136048896 := 
by
  sorry

end NUMINAMATH_GPT_product_of_six_numbers_l1387_138729


namespace NUMINAMATH_GPT_first_laptop_cost_l1387_138733

variable (x : ℝ)

def cost_first_laptop (x : ℝ) : ℝ := x
def cost_second_laptop (x : ℝ) : ℝ := 3 * x
def total_cost (x : ℝ) : ℝ := cost_first_laptop x + cost_second_laptop x
def budget : ℝ := 2000

theorem first_laptop_cost : total_cost x = budget → x = 500 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_first_laptop_cost_l1387_138733


namespace NUMINAMATH_GPT_smallest_solution_l1387_138705

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l1387_138705


namespace NUMINAMATH_GPT_distinct_real_numbers_condition_l1387_138764

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)

theorem distinct_real_numbers_condition (a b x1 x2 x3 : ℝ) :
  f a b x1 = x2 → f a b x2 = x3 → f a b x3 = x1 → x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → a = -b^2 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_numbers_condition_l1387_138764


namespace NUMINAMATH_GPT_second_game_score_count_l1387_138742

-- Define the conditions and problem
def total_points (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  A1 + A2 + A3 + B1 + B2 + B3 = 31

def valid_game_1 (A1 B1 : ℕ) : Prop :=
  A1 ≥ 11 ∧ A1 - B1 ≥ 2

def valid_game_2 (A2 B2 : ℕ) : Prop :=
  B2 ≥ 11 ∧ B2 - A2 ≥ 2

def valid_game_3 (A3 B3 : ℕ) : Prop :=
  A3 ≥ 11 ∧ A3 - B3 ≥ 2

def game_sequence (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  valid_game_1 A1 B1 ∧ valid_game_2 A2 B2 ∧ valid_game_3 A3 B3

noncomputable def second_game_score_possibilities : ℕ := 
  8 -- This is derived from calculating the valid scores where B wins the second game.

theorem second_game_score_count (A1 A2 A3 B1 B2 B3 : ℕ) (h_total : total_points A1 A2 A3 B1 B2 B3) (h_sequence : game_sequence A1 A2 A3 B1 B2 B3) :
  second_game_score_possibilities = 8 := sorry

end NUMINAMATH_GPT_second_game_score_count_l1387_138742


namespace NUMINAMATH_GPT_percentage_students_left_in_classroom_l1387_138773

def total_students : ℕ := 250
def fraction_painting : ℚ := 3 / 10
def fraction_field : ℚ := 2 / 10
def fraction_science : ℚ := 1 / 5

theorem percentage_students_left_in_classroom :
  let gone_painting := total_students * fraction_painting
  let gone_field := total_students * fraction_field
  let gone_science := total_students * fraction_science
  let students_gone := gone_painting + gone_field + gone_science
  let students_left := total_students - students_gone
  (students_left / total_students) * 100 = 30 :=
by sorry

end NUMINAMATH_GPT_percentage_students_left_in_classroom_l1387_138773


namespace NUMINAMATH_GPT_LCM_GCD_even_nonnegative_l1387_138786

theorem LCM_GCD_even_nonnegative (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  : ∃ (n : ℕ), (n = Nat.lcm a b + Nat.gcd a b - a - b) ∧ (n % 2 = 0) ∧ (0 ≤ n) := 
sorry

end NUMINAMATH_GPT_LCM_GCD_even_nonnegative_l1387_138786


namespace NUMINAMATH_GPT_digit_150_in_fraction_l1387_138709

-- Define the decimal expansion repeating sequence for the fraction 31/198
def repeat_seq : List Nat := [1, 5, 6, 5, 6, 5]

-- Define a function to get the nth digit of the repeating sequence
def nth_digit (n : Nat) : Nat :=
  repeat_seq.get! ((n - 1) % repeat_seq.length)

-- State the theorem to be proved
theorem digit_150_in_fraction : nth_digit 150 = 5 := 
sorry

end NUMINAMATH_GPT_digit_150_in_fraction_l1387_138709


namespace NUMINAMATH_GPT_sum_of_roots_quadratic_eq_l1387_138765

theorem sum_of_roots_quadratic_eq : ∀ P Q : ℝ, (3 * P^2 - 9 * P + 6 = 0) ∧ (3 * Q^2 - 9 * Q + 6 = 0) → P + Q = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_quadratic_eq_l1387_138765


namespace NUMINAMATH_GPT_conic_sections_of_equation_l1387_138740

theorem conic_sections_of_equation :
  ∀ x y : ℝ, y^4 - 9*x^6 = 3*y^2 - 1 →
  (∃ y, y^2 - 3*x^3 = 4 ∨ y^2 + 3*x^3 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_conic_sections_of_equation_l1387_138740


namespace NUMINAMATH_GPT_area_of_rectangular_garden_l1387_138743

-- Definitions based on conditions
def width : ℕ := 15
def length : ℕ := 3 * width
def area : ℕ := length * width

-- The theorem we want to prove
theorem area_of_rectangular_garden : area = 675 :=
by sorry

end NUMINAMATH_GPT_area_of_rectangular_garden_l1387_138743


namespace NUMINAMATH_GPT_total_steps_l1387_138750

def steps_on_feet (jason_steps : Nat) (nancy_ratio : Nat) : Nat :=
  jason_steps + (nancy_ratio * jason_steps)

theorem total_steps (jason_steps : Nat) (nancy_ratio : Nat) (h1 : jason_steps = 8) (h2 : nancy_ratio = 3) :
  steps_on_feet jason_steps nancy_ratio = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_steps_l1387_138750


namespace NUMINAMATH_GPT_work_days_of_a_l1387_138795

variable (da wa wb wc : ℕ)
variable (hcp : 3 * wc = 5 * wa)
variable (hbw : 4 * wc = 5 * wb)
variable (hwc : wc = 100)
variable (hear : 60 * da + 9 * 80 + 4 * 100 = 1480)

theorem work_days_of_a : da = 6 :=
by
  sorry

end NUMINAMATH_GPT_work_days_of_a_l1387_138795


namespace NUMINAMATH_GPT_mass_of_man_l1387_138741

def boat_length : ℝ := 3 -- boat length in meters
def boat_breadth : ℝ := 2 -- boat breadth in meters
def boat_sink_depth : ℝ := 0.01 -- boat sink depth in meters
def water_density : ℝ := 1000 -- density of water in kg/m^3

/- Theorem: The mass of the man is equal to 60 kg given the parameters defined above. -/
theorem mass_of_man : (water_density * (boat_length * boat_breadth * boat_sink_depth)) = 60 :=
by
  simp [boat_length, boat_breadth, boat_sink_depth, water_density]
  sorry

end NUMINAMATH_GPT_mass_of_man_l1387_138741


namespace NUMINAMATH_GPT_lifespan_histogram_l1387_138774

theorem lifespan_histogram :
  (class_interval = 20) →
  (height_vertical_axis_60_80 = 0.03) →
  (total_people = 1000) →
  (number_of_people_60_80 = 600) :=
by
  intro class_interval height_vertical_axis_60_80 total_people
  -- Perform necessary calculations (omitting actual proof as per instructions)
  sorry

end NUMINAMATH_GPT_lifespan_histogram_l1387_138774


namespace NUMINAMATH_GPT_functional_equation_solution_l1387_138798

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1387_138798


namespace NUMINAMATH_GPT_exists_two_factorizations_in_C_another_number_with_property_l1387_138769

def in_set_C (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 1

def is_prime_wrt_C (k : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, in_set_C a ∧ in_set_C b ∧ k = a * b

theorem exists_two_factorizations_in_C : 
  ∃ (a b a' b' : ℕ), 
  in_set_C 4389 ∧ 
  in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
  (4389 = a * b ∧ 4389 = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

theorem another_number_with_property : 
 ∃ (n a b a' b' : ℕ), 
 n ≠ 4389 ∧ in_set_C n ∧ 
 in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
 (n = a * b ∧ n = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

end NUMINAMATH_GPT_exists_two_factorizations_in_C_another_number_with_property_l1387_138769


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1387_138746

theorem necessary_but_not_sufficient (a : ℝ) : (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) := sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1387_138746


namespace NUMINAMATH_GPT_problem_statement_l1387_138757

open Real

theorem problem_statement (α : ℝ) 
  (h1 : cos (α + π / 4) = (7 * sqrt 2) / 10)
  (h2 : cos (2 * α) = 7 / 25) :
  sin α + cos α = 1 / 5 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1387_138757


namespace NUMINAMATH_GPT_repaired_shoes_last_time_l1387_138761

theorem repaired_shoes_last_time :
  let cost_of_repair := 13.50
  let cost_of_new := 32.00
  let duration_of_new := 2.0
  let surcharge := 0.1852
  let avg_cost_new := cost_of_new / duration_of_new
  let avg_cost_repair (T : ℝ) := cost_of_repair / T
  (avg_cost_new = (1 + surcharge) * avg_cost_repair 1) ↔ T = 1 := 
by
  sorry

end NUMINAMATH_GPT_repaired_shoes_last_time_l1387_138761


namespace NUMINAMATH_GPT_min_air_routes_l1387_138762

theorem min_air_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) : 
  a + b + c ≥ 21 :=
sorry

end NUMINAMATH_GPT_min_air_routes_l1387_138762


namespace NUMINAMATH_GPT_contrapositive_example_l1387_138737

theorem contrapositive_example (a b m : ℝ) :
  (a > b → a * (m^2 + 1) > b * (m^2 + 1)) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end NUMINAMATH_GPT_contrapositive_example_l1387_138737


namespace NUMINAMATH_GPT_power_sum_l1387_138731

theorem power_sum (h : (9 : ℕ) = 3^2) : (2^567 + (9^5 / 3^2) : ℕ) = 2^567 + 6561 := by
  sorry

end NUMINAMATH_GPT_power_sum_l1387_138731


namespace NUMINAMATH_GPT_propositionA_necessary_but_not_sufficient_for_propositionB_l1387_138711

-- Definitions for propositions and conditions
def propositionA (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0
def propositionB (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement for the necessary but not sufficient condition
theorem propositionA_necessary_but_not_sufficient_for_propositionB (a : ℝ) :
  (propositionA a) → (¬ propositionB a) ∧ (propositionB a → propositionA a) :=
by
  sorry

end NUMINAMATH_GPT_propositionA_necessary_but_not_sufficient_for_propositionB_l1387_138711


namespace NUMINAMATH_GPT_cakes_sold_l1387_138732

/-- If a baker made 54 cakes and has 13 cakes left, then the number of cakes he sold is 41. -/
theorem cakes_sold (original_cakes : ℕ) (cakes_left : ℕ) 
  (h1 : original_cakes = 54) (h2 : cakes_left = 13) : 
  original_cakes - cakes_left = 41 := 
by 
  sorry

end NUMINAMATH_GPT_cakes_sold_l1387_138732


namespace NUMINAMATH_GPT_average_production_per_day_for_entire_month_l1387_138724

-- Definitions based on the conditions
def average_first_25_days := 65
def average_last_5_days := 35
def number_of_days_in_first_period := 25
def number_of_days_in_last_period := 5
def total_days_in_month := 30

-- The goal is to prove that the average production per day for the entire month is 60 TVs/day.
theorem average_production_per_day_for_entire_month :
  (average_first_25_days * number_of_days_in_first_period + 
   average_last_5_days * number_of_days_in_last_period) / total_days_in_month = 60 := 
by
  sorry

end NUMINAMATH_GPT_average_production_per_day_for_entire_month_l1387_138724


namespace NUMINAMATH_GPT_proof_of_problem_l1387_138766

noncomputable def problem_statement (a b c x y z : ℝ) : Prop :=
  23 * x + b * y + c * z = 0 ∧
  a * x + 33 * y + c * z = 0 ∧
  a * x + b * y + 52 * z = 0 ∧
  a ≠ 23 ∧
  x ≠ 0 →
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1

theorem proof_of_problem (a b c x y z : ℝ) (h : problem_statement a b c x y z) : 
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 :=
sorry

end NUMINAMATH_GPT_proof_of_problem_l1387_138766


namespace NUMINAMATH_GPT_equilateral_triangle_l1387_138756

variable (A B C A₀ B₀ C₀ : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]

variable (midpoint : ∀ (X₁ X₂ : Type), Type) 
variable (circumcircle : ∀ (X Y Z : Type), Type)

def medians_meet_circumcircle := ∀ (A A₁ B B₁ C C₁ : Type) 
  [AddGroup A] [AddGroup A₁] [AddGroup B] [AddGroup B₁] [AddGroup C] [AddGroup C₁], 
  Prop

def areas_equal := ∀ (ABC₀ AB₀C A₀BC : Type) 
  [AddGroup ABC₀] [AddGroup AB₀C] [AddGroup A₀BC], 
  Prop

theorem equilateral_triangle (A B C A₀ B₀ C₀ A₁ B₁ C₁ : Type)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]
  [AddGroup A₁] [AddGroup B₁] [AddGroup C₁] 
  (midpoint_cond : ∀ (X Y Z : Type), Z = midpoint X Y)
  (circumcircle_cond : ∀ (X Y Z : Type), Z = circumcircle X Y Z)
  (medians_meet_circumcircle : Prop)
  (areas_equal: Prop) :
    A = B ∧ B = C ∧ C = A :=
  sorry

end NUMINAMATH_GPT_equilateral_triangle_l1387_138756


namespace NUMINAMATH_GPT_find_a_l1387_138797

open Real

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line equation passing through P(2,2)
def line_through_P (m b x y : ℝ) : Prop := y = m * x + b ∧ (2, 2) = (x, y)

-- Define the line equation ax - y + 1 = 0
def perpendicular_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a : ∃ a : ℝ, ∀ x y m b : ℝ,
    circle x y ∧ line_through_P m b x y ∧
    (line_through_P m b x y → perpendicular_line a x y) → a = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_a_l1387_138797


namespace NUMINAMATH_GPT_volume_of_cube_for_tetrahedron_l1387_138722

theorem volume_of_cube_for_tetrahedron (h : ℝ) (b1 b2 : ℝ) (V : ℝ) 
  (h_condition : h = 15) (b1_condition : b1 = 8) (b2_condition : b2 = 12)
  (V_condition : V = 3375) : 
  V = (max h (max b1 b2)) ^ 3 := by
  -- To illustrate the mathematical context and avoid concrete steps,
  -- sorry provides the completion of the logical binding to the correct answer
  sorry

end NUMINAMATH_GPT_volume_of_cube_for_tetrahedron_l1387_138722


namespace NUMINAMATH_GPT_liam_total_time_l1387_138736

noncomputable def total_time_7_laps : Nat :=
let time_first_200 := 200 / 5  -- Time in seconds for the first 200 meters
let time_next_300 := 300 / 6   -- Time in seconds for the next 300 meters
let time_per_lap := time_first_200 + time_next_300
let laps := 7
let total_time := laps * time_per_lap
total_time

theorem liam_total_time : total_time_7_laps = 630 := by
sorry

end NUMINAMATH_GPT_liam_total_time_l1387_138736


namespace NUMINAMATH_GPT_jerry_age_is_13_l1387_138752

variable (M J : ℕ)

theorem jerry_age_is_13 (h1 : M = 2 * J - 6) (h2 : M = 20) : J = 13 := by
  sorry

end NUMINAMATH_GPT_jerry_age_is_13_l1387_138752


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1387_138784

-- Define the concept of an angle being in a specific quadrant
def is_in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

-- Prove that -1200° is in the third quadrant
theorem angle_in_third_quadrant :
  is_in_third_quadrant (240) → is_in_third_quadrant (-1200 % 360 + 360 * (if -1200 % 360 ≤ 0 then 1 else 0)) :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1387_138784


namespace NUMINAMATH_GPT_range_of_a_l1387_138751

def p (x : ℝ) : Prop := (1/2 ≤ x ∧ x ≤ 1)

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬ p x) → 
  (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1387_138751


namespace NUMINAMATH_GPT_find_k_l1387_138712

theorem find_k (x y k : ℝ) (h_line : 2 - k * x = -4 * y) (h_point : x = 3 ∧ y = -2) : k = -2 :=
by
  -- Given the conditions that the point (3, -2) lies on the line 2 - kx = -4y, 
  -- we want to prove that k = -2
  sorry

end NUMINAMATH_GPT_find_k_l1387_138712


namespace NUMINAMATH_GPT_total_students_correct_l1387_138701

def num_first_graders : ℕ := 358
def num_second_graders : ℕ := num_first_graders - 64
def total_students : ℕ := num_first_graders + num_second_graders

theorem total_students_correct : total_students = 652 :=
by
  sorry

end NUMINAMATH_GPT_total_students_correct_l1387_138701


namespace NUMINAMATH_GPT_tom_change_l1387_138793

theorem tom_change :
  let SNES_value := 150
  let credit_percent := 0.80
  let amount_given := 80
  let game_value := 30
  let NES_sale_price := 160
  let credit_for_SNES := credit_percent * SNES_value
  let amount_to_pay_for_NES := NES_sale_price - credit_for_SNES
  let effective_amount_paid := amount_to_pay_for_NES - game_value
  let change_received := amount_given - effective_amount_paid
  change_received = 70 :=
by
  sorry

end NUMINAMATH_GPT_tom_change_l1387_138793


namespace NUMINAMATH_GPT_number_of_good_games_l1387_138708

def total_games : ℕ := 11
def bad_games : ℕ := 5
def good_games : ℕ := total_games - bad_games

theorem number_of_good_games : good_games = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_good_games_l1387_138708


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1387_138768

-- Define the conditions
def equation1 (x y : ℝ) : Prop := x^2 + (y^2 / 2) = 1
def equation2 (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def equation3 (x y : ℝ) : Prop := x^2 + (y^2 / 4) = 1
def equation4 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the points
def point1 (x y : ℝ) : Prop := (x = 1 ∧ y = 0)
def point2 (x y : ℝ) : Prop := (x = 0 ∧ y = 2)

-- Define the main theorem
theorem ellipse_standard_equation :
  (equation4 1 0 ∧ equation4 0 2) ↔
  ((equation1 1 0 ∧ equation1 0 2) ∨
   (equation2 1 0 ∧ equation2 0 2) ∨
   (equation3 1 0 ∧ equation3 0 2) ∨
   (equation4 1 0 ∧ equation4 0 2)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1387_138768


namespace NUMINAMATH_GPT_total_phones_in_Delaware_l1387_138771

def population : ℕ := 974000
def phones_per_1000 : ℕ := 673

theorem total_phones_in_Delaware : (population / 1000) * phones_per_1000 = 655502 := by
  sorry

end NUMINAMATH_GPT_total_phones_in_Delaware_l1387_138771


namespace NUMINAMATH_GPT_functional_relationship_l1387_138791

variable (x y k1 k2 : ℝ)

axiom h1 : y = k1 * x + k2 / (x - 2)
axiom h2 : (y = -1) ↔ (x = 1)
axiom h3 : (y = 5) ↔ (x = 3)

theorem functional_relationship :
  (∀ x y, y = k1 * x + k2 / (x - 2) ∧
    ((x = 1) → y = -1) ∧
    ((x = 3) → y = 5) → y = x + 2 / (x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_functional_relationship_l1387_138791


namespace NUMINAMATH_GPT_primary_school_capacity_l1387_138792

variable (x : ℝ)

/-- In a town, there are four primary schools. Two of them can teach 400 students at a time, 
and the other two can teach a certain number of students at a time. These four primary schools 
can teach a total of 1480 students at a time. -/
theorem primary_school_capacity 
  (h1 : 2 * 400 + 2 * x = 1480) : 
  x = 340 :=
sorry

end NUMINAMATH_GPT_primary_school_capacity_l1387_138792


namespace NUMINAMATH_GPT_find_weight_of_b_l1387_138785

variable (a b c d : ℝ)

def average_weight_of_four : Prop := (a + b + c + d) / 4 = 45

def average_weight_of_a_and_b : Prop := (a + b) / 2 = 42

def average_weight_of_b_and_c : Prop := (b + c) / 2 = 43

def ratio_of_d_to_a : Prop := d / a = 3 / 4

theorem find_weight_of_b (h1 : average_weight_of_four a b c d)
                        (h2 : average_weight_of_a_and_b a b)
                        (h3 : average_weight_of_b_and_c b c)
                        (h4 : ratio_of_d_to_a a d) :
    b = 29.43 :=
  by sorry

end NUMINAMATH_GPT_find_weight_of_b_l1387_138785


namespace NUMINAMATH_GPT_maoming_population_scientific_notation_l1387_138700

-- Definitions for conditions
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- The main theorem to prove
theorem maoming_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 6800000 ∧ a = 6.8 ∧ n = 6 :=
sorry

end NUMINAMATH_GPT_maoming_population_scientific_notation_l1387_138700
