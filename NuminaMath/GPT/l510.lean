import Mathlib

namespace NUMINAMATH_GPT_find_age_of_15th_person_l510_51098

-- Define the conditions given in the problem
def total_age_of_18_persons (avg_18 : ℕ) (num_18 : ℕ) : ℕ := avg_18 * num_18
def total_age_of_5_persons (avg_5 : ℕ) (num_5 : ℕ) : ℕ := avg_5 * num_5
def total_age_of_9_persons (avg_9 : ℕ) (num_9 : ℕ) : ℕ := avg_9 * num_9

-- Define the overall question which is the age of the 15th person
def age_of_15th_person (total_18 : ℕ) (total_5 : ℕ) (total_9 : ℕ) : ℕ :=
  total_18 - total_5 - total_9

-- Statement of the theorem to prove
theorem find_age_of_15th_person :
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  age_of_15th_person total_18 total_5 total_9 = 56 :=
by
  -- Definitions for the total ages
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  
  -- Goal: compute the age of the 15th person
  let answer := age_of_15th_person total_18 total_5 total_9

  -- Prove that the computed age is equal to 56
  show answer = 56
  sorry

end NUMINAMATH_GPT_find_age_of_15th_person_l510_51098


namespace NUMINAMATH_GPT_sarah_took_correct_amount_l510_51027

-- Definition of the conditions
def total_cookies : Nat := 150
def neighbors_count : Nat := 15
def correct_amount_per_neighbor : Nat := 10
def remaining_cookies : Nat := 8
def first_neighbors_count : Nat := 14
def last_neighbor : String := "Sarah"

-- Calculations based on conditions
def total_cookies_taken : Nat := total_cookies - remaining_cookies
def correct_cookies_taken : Nat := first_neighbors_count * correct_amount_per_neighbor
def extra_cookies_taken : Nat := total_cookies_taken - correct_cookies_taken
def sarah_cookies : Nat := correct_amount_per_neighbor + extra_cookies_taken

-- Proof statement: Sarah took 12 cookies
theorem sarah_took_correct_amount : sarah_cookies = 12 := by
  sorry

end NUMINAMATH_GPT_sarah_took_correct_amount_l510_51027


namespace NUMINAMATH_GPT_line_parabola_intersection_l510_51068

theorem line_parabola_intersection (k : ℝ) (M A B : ℝ × ℝ) (h1 : ¬ k = 0) 
  (h2 : M = (2, 0))
  (h3 : ∃ x y, (x = k * y + 2 ∧ (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧ (p = A ∨ p = B))) 
  : 1 / |dist M A|^2 + 1 / |dist M B|^2 = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_line_parabola_intersection_l510_51068


namespace NUMINAMATH_GPT_complex_fraction_simplification_l510_51059

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : (2 : ℂ) / (1 + i)^2 = i :=
by 
-- this will be filled when proving the theorem in Lean
sorry

end NUMINAMATH_GPT_complex_fraction_simplification_l510_51059


namespace NUMINAMATH_GPT_area_of_shaded_region_l510_51066

-- Given conditions
def side_length := 8
def area_of_square := side_length * side_length
def area_of_triangle := area_of_square / 4

-- Lean 4 statement for the equivalence
theorem area_of_shaded_region : area_of_triangle = 16 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l510_51066


namespace NUMINAMATH_GPT_remainder_of_8x_minus_5_l510_51096

theorem remainder_of_8x_minus_5 (x : ℕ) (h : x % 15 = 7) : (8 * x - 5) % 15 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_8x_minus_5_l510_51096


namespace NUMINAMATH_GPT_mouse_jump_vs_grasshopper_l510_51046

-- Definitions for jumps
def grasshopper_jump : ℕ := 14
def frog_jump : ℕ := grasshopper_jump + 37
def mouse_jump : ℕ := frog_jump - 16

-- Theorem stating the result
theorem mouse_jump_vs_grasshopper : mouse_jump - grasshopper_jump = 21 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_mouse_jump_vs_grasshopper_l510_51046


namespace NUMINAMATH_GPT_solution_set_inequality_l510_51005

theorem solution_set_inequality (m : ℤ) (h₁ : (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2)) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} :=
by
  -- The detailed proof would be added here.
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l510_51005


namespace NUMINAMATH_GPT_car_travel_distance_l510_51014

-- Define the conditions: speed and time
def speed : ℝ := 160 -- in km/h
def time : ℝ := 5 -- in hours

-- Define the calculation for distance
def distance (s t : ℝ) : ℝ := s * t

-- Prove that given the conditions, the distance is 800 km
theorem car_travel_distance : distance speed time = 800 := by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l510_51014


namespace NUMINAMATH_GPT_not_invited_students_l510_51022

-- Definition of the problem conditions
def students := 15
def direct_friends_of_mia := 4
def unique_friends_of_each_friend := 2

-- Problem statement
theorem not_invited_students : (students - (1 + direct_friends_of_mia + direct_friends_of_mia * unique_friends_of_each_friend) = 2) :=
by
  sorry

end NUMINAMATH_GPT_not_invited_students_l510_51022


namespace NUMINAMATH_GPT_find_y_given_conditions_l510_51053

theorem find_y_given_conditions (x : ℤ) (y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 2) (h2 : x = -5) : y = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l510_51053


namespace NUMINAMATH_GPT_initial_percentage_alcohol_l510_51095

-- Define the initial conditions
variables (P : ℚ) -- percentage of alcohol in the initial solution
variables (V1 V2 : ℚ) -- volumes of the initial solution and added alcohol
variables (C2 : ℚ) -- concentration of the resulting solution

-- Given the initial conditions and additional parameters
def initial_solution_volume : ℚ := 6
def added_alcohol_volume : ℚ := 1.8
def final_solution_volume : ℚ := initial_solution_volume + added_alcohol_volume
def final_solution_concentration : ℚ := 0.5 -- 50%

-- The amount of alcohol initially = (P / 100) * V1
-- New amount of alcohol after adding pure alcohol
-- This should equal to the final concentration of the new volume

theorem initial_percentage_alcohol : 
  (P / 100 * initial_solution_volume) + added_alcohol_volume = final_solution_concentration * final_solution_volume → 
  P = 35 :=
sorry

end NUMINAMATH_GPT_initial_percentage_alcohol_l510_51095


namespace NUMINAMATH_GPT_trip_time_is_correct_l510_51065

noncomputable def total_trip_time : ℝ :=
  let wrong_direction_time := 100 / 60
  let return_time := 100 / 45
  let detour_time := 30 / 45
  let normal_trip_time := 300 / 60
  let stop_time := 2 * (15 / 60)
  wrong_direction_time + return_time + detour_time + normal_trip_time + stop_time

theorem trip_time_is_correct : total_trip_time = 10.06 :=
  by
    -- Proof steps are omitted
    sorry

end NUMINAMATH_GPT_trip_time_is_correct_l510_51065


namespace NUMINAMATH_GPT_cone_height_is_2_sqrt_15_l510_51007

noncomputable def height_of_cone (radius : ℝ) (num_sectors : ℕ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let sector_arc_length := circumference / num_sectors
  let base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := radius
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2)

theorem cone_height_is_2_sqrt_15 :
  height_of_cone 8 4 = 2 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_cone_height_is_2_sqrt_15_l510_51007


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l510_51031

theorem quadratic_has_distinct_real_roots :
  ∀ (x : ℝ), x^2 - 2 * x - 1 = 0 → (∃ Δ > 0, Δ = ((-2)^2 - 4 * 1 * (-1))) := by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l510_51031


namespace NUMINAMATH_GPT_marbles_count_l510_51041

theorem marbles_count (initial_marble: ℕ) (bought_marble: ℕ) (final_marble: ℕ) 
  (h1: initial_marble = 53) (h2: bought_marble = 134) : 
  final_marble = initial_marble + bought_marble -> final_marble = 187 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

-- sorry is omitted as proof is given.

end NUMINAMATH_GPT_marbles_count_l510_51041


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_l510_51088

-- Definitions
def quadratic (a b c x : ℝ) := a * x ^ 2 + b * x + c
def has_two_real_roots (a b c : ℝ) := b ^ 2 - 4 * a * c ≥ 0 
def neighboring_root_equation (a b c : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧ |x₁ - x₂| = 1

-- Proof problem 1: Prove whether x^2 + x - 6 = 0 is a neighboring root equation
theorem problem1_part1 : ¬ neighboring_root_equation 1 1 (-6) := 
sorry

-- Proof problem 2: Prove whether 2x^2 - 2√5x + 2 = 0 is a neighboring root equation
theorem problem1_part2 : neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 := 
sorry

-- Proof problem 3: Prove that m = -1 or m = -3 for x^2 - (m-2)x - 2m = 0 to be a neighboring root equation
theorem problem2 (m : ℝ) (h : neighboring_root_equation 1 (-(m-2)) (-2*m)) : 
  m = -1 ∨ m = -3 := 
sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_l510_51088


namespace NUMINAMATH_GPT_calculate_taxes_l510_51087

def gross_pay : ℝ := 4500
def tax_rate_1 : ℝ := 0.10
def tax_rate_2 : ℝ := 0.15
def tax_rate_3 : ℝ := 0.20
def income_bracket_1 : ℝ := 1500
def income_bracket_2 : ℝ := 2000
def income_bracket_remaining : ℝ := gross_pay - income_bracket_1 - income_bracket_2
def standard_deduction : ℝ := 100

theorem calculate_taxes :
  let tax_1 := tax_rate_1 * income_bracket_1
  let tax_2 := tax_rate_2 * income_bracket_2
  let tax_3 := tax_rate_3 * income_bracket_remaining
  let total_tax := tax_1 + tax_2 + tax_3
  let tax_after_deduction := total_tax - standard_deduction
  tax_after_deduction = 550 :=
by
  sorry

end NUMINAMATH_GPT_calculate_taxes_l510_51087


namespace NUMINAMATH_GPT_jordan_more_novels_than_maxime_l510_51042

def jordan_french_novels : ℕ := 130
def jordan_spanish_novels : ℕ := 20

def alexandre_french_novels : ℕ := jordan_french_novels / 10
def alexandre_spanish_novels : ℕ := 3 * jordan_spanish_novels

def camille_french_novels : ℕ := 2 * alexandre_french_novels
def camille_spanish_novels : ℕ := jordan_spanish_novels / 2

def total_french_novels : ℕ := jordan_french_novels + alexandre_french_novels + camille_french_novels

def maxime_french_novels : ℕ := total_french_novels / 2 - 5
def maxime_spanish_novels : ℕ := 2 * camille_spanish_novels

def jordan_total_novels : ℕ := jordan_french_novels + jordan_spanish_novels
def maxime_total_novels : ℕ := maxime_french_novels + maxime_spanish_novels

def novels_difference : ℕ := jordan_total_novels - maxime_total_novels

theorem jordan_more_novels_than_maxime : novels_difference = 51 :=
sorry

end NUMINAMATH_GPT_jordan_more_novels_than_maxime_l510_51042


namespace NUMINAMATH_GPT_min_xy_value_min_x_plus_y_value_l510_51032

variable {x y : ℝ}

theorem min_xy_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : xy ≥ 64 := 
sorry

theorem min_x_plus_y_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : x + y ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_xy_value_min_x_plus_y_value_l510_51032


namespace NUMINAMATH_GPT_age_difference_l510_51006

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a - c = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l510_51006


namespace NUMINAMATH_GPT_inscribed_rectangle_area_l510_51017

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end NUMINAMATH_GPT_inscribed_rectangle_area_l510_51017


namespace NUMINAMATH_GPT_average_height_of_four_people_l510_51039

theorem average_height_of_four_people (
  h1 h2 h3 h4 : ℕ
) (diff12 : h2 = h1 + 2)
  (diff23 : h3 = h2 + 2)
  (diff34 : h4 = h3 + 6)
  (h4_eq : h4 = 83) :
  (h1 + h2 + h3 + h4) / 4 = 77 :=
by sorry

end NUMINAMATH_GPT_average_height_of_four_people_l510_51039


namespace NUMINAMATH_GPT_change_is_correct_l510_51010

-- Define the cost of the pencil in cents
def cost_of_pencil : ℕ := 35

-- Define the amount paid in cents
def amount_paid : ℕ := 100

-- State the theorem for the change
theorem change_is_correct : amount_paid - cost_of_pencil = 65 :=
by sorry

end NUMINAMATH_GPT_change_is_correct_l510_51010


namespace NUMINAMATH_GPT_four_digit_numbers_divisible_by_5_l510_51057

theorem four_digit_numbers_divisible_by_5 : 
  let smallest_4_digit := 1000
  let largest_4_digit := 9999
  let divisible_by_5 (n : ℕ) := ∃ k : ℕ, n = 5 * k
  ∃ n : ℕ, ( ∀ x : ℕ, smallest_4_digit ≤ x ∧ x ≤ largest_4_digit ∧ divisible_by_5 x ↔ (smallest_4_digit + (n-1) * 5 = x) ) ∧ n = 1800 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_divisible_by_5_l510_51057


namespace NUMINAMATH_GPT_solve_expression_l510_51067

theorem solve_expression (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end NUMINAMATH_GPT_solve_expression_l510_51067


namespace NUMINAMATH_GPT_abs_gt_1_not_sufficient_nor_necessary_l510_51075

theorem abs_gt_1_not_sufficient_nor_necessary (a : ℝ) :
  ¬((|a| > 1) → (a > 0)) ∧ ¬((a > 0) → (|a| > 1)) :=
by
  sorry

end NUMINAMATH_GPT_abs_gt_1_not_sufficient_nor_necessary_l510_51075


namespace NUMINAMATH_GPT_distance_between_points_A_B_l510_51051

theorem distance_between_points_A_B :
  let A := (8, -5)
  let B := (0, 10)
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 17 :=
by
  let A := (8, -5)
  let B := (0, 10)
  sorry

end NUMINAMATH_GPT_distance_between_points_A_B_l510_51051


namespace NUMINAMATH_GPT_find_positive_integer_triples_l510_51072

-- Define the condition for the integer divisibility problem
def is_integer_division (t a b : ℕ) : Prop :=
  (t ^ (a + b) + 1) % (t ^ a + t ^ b + 1) = 0

-- Statement of the theorem
theorem find_positive_integer_triples :
  ∀ (t a b : ℕ), t > 0 → a > 0 → b > 0 → is_integer_division t a b → (t, a, b) = (2, 1, 1) :=
by
  intros t a b t_pos a_pos b_pos h
  sorry

end NUMINAMATH_GPT_find_positive_integer_triples_l510_51072


namespace NUMINAMATH_GPT_concession_stand_total_revenue_l510_51071

theorem concession_stand_total_revenue :
  let hot_dog_price : ℝ := 1.50
  let soda_price : ℝ := 0.50
  let total_items_sold : ℕ := 87
  let hot_dogs_sold : ℕ := 35
  let sodas_sold := total_items_sold - hot_dogs_sold
  let revenue_from_hot_dogs := hot_dogs_sold * hot_dog_price
  let revenue_from_sodas := sodas_sold * soda_price
  revenue_from_hot_dogs + revenue_from_sodas = 78.50 :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_concession_stand_total_revenue_l510_51071


namespace NUMINAMATH_GPT_arithmetic_seq_min_S19_l510_51002

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_seq_min_S19
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_S8 : S a 8 ≤ 6)
  (h_S11 : S a 11 ≥ 27) :
  S a 19 ≥ 133 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_min_S19_l510_51002


namespace NUMINAMATH_GPT_mateen_garden_area_l510_51028

theorem mateen_garden_area :
  ∃ (L W : ℝ), (20 * L = 1000) ∧ (8 * (2 * L + 2 * W) = 1000) ∧ (L * W = 625) :=
by
  sorry

end NUMINAMATH_GPT_mateen_garden_area_l510_51028


namespace NUMINAMATH_GPT_evaluate_expression_l510_51047

theorem evaluate_expression (a b : ℤ) (h_a : a = 1) (h_b : b = -2) : 
  2 * (a^2 - 3 * a * b + 1) - (2 * a^2 - b^2) + 5 * a * b = 8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l510_51047


namespace NUMINAMATH_GPT_intersection_complement_eq_l510_51097

open Set

namespace MathProof

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {3, 4, 5} →
  B = {1, 3, 6} →
  A ∩ (U \ B) = {4, 5} :=
by
  intros hU hA hB
  sorry

end MathProof

end NUMINAMATH_GPT_intersection_complement_eq_l510_51097


namespace NUMINAMATH_GPT_maximum_tied_teams_round_robin_l510_51015

noncomputable def round_robin_tournament_max_tied_teams (n : ℕ) : ℕ := 
  sorry

theorem maximum_tied_teams_round_robin (h : n = 8) : round_robin_tournament_max_tied_teams n = 7 :=
sorry

end NUMINAMATH_GPT_maximum_tied_teams_round_robin_l510_51015


namespace NUMINAMATH_GPT_problem_statement_l510_51003

open Set

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem problem_statement :
  {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l510_51003


namespace NUMINAMATH_GPT_dartboard_area_ratio_l510_51021

theorem dartboard_area_ratio
  (side_length : ℝ)
  (h_side_length : side_length = 2)
  (t : ℝ)
  (q : ℝ)
  (h_t : t = (1 / 2) * (1 / (Real.sqrt 2)) * (1 / (Real.sqrt 2)))
  (h_q : q = ((side_length * side_length) - (8 * t)) / 4) :
  q / t = 2 := by
  sorry

end NUMINAMATH_GPT_dartboard_area_ratio_l510_51021


namespace NUMINAMATH_GPT_suzannes_book_pages_l510_51012

-- Conditions
def pages_read_on_monday : ℕ := 15
def pages_read_on_tuesday : ℕ := 31
def pages_left : ℕ := 18

-- Total number of pages in the book
def total_pages : ℕ := pages_read_on_monday + pages_read_on_tuesday + pages_left

-- Problem statement
theorem suzannes_book_pages : total_pages = 64 :=
by
  -- Proof is not required, only the statement
  sorry

end NUMINAMATH_GPT_suzannes_book_pages_l510_51012


namespace NUMINAMATH_GPT_value_of_a_l510_51026

theorem value_of_a (a : ℚ) (h : a + a / 4 = 6 / 2) : a = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l510_51026


namespace NUMINAMATH_GPT_angle_Z_is_120_l510_51062

-- Define angles and lines
variables {p q : Prop} {X Y Z : ℝ}
variables (h_parallel : p ∧ q)
variables (hX : X = 100)
variables (hY : Y = 140)

-- Proof statement: Given the angles X and Y, we prove that angle Z is 120 degrees.
theorem angle_Z_is_120 (h_parallel : p ∧ q) (hX : X = 100) (hY : Y = 140) : Z = 120 := by 
  -- Here we would add the proof steps
  sorry

end NUMINAMATH_GPT_angle_Z_is_120_l510_51062


namespace NUMINAMATH_GPT_proposition_R_is_converse_negation_of_P_l510_51029

variables (x y : ℝ)

def P : Prop := x + y = 0 → x = -y
def Q : Prop := ¬(x + y = 0) → x ≠ -y
def R : Prop := x ≠ -y → ¬(x + y = 0)

theorem proposition_R_is_converse_negation_of_P : R x y ↔ ¬P x y :=
by sorry

end NUMINAMATH_GPT_proposition_R_is_converse_negation_of_P_l510_51029


namespace NUMINAMATH_GPT_x_in_A_neither_sufficient_nor_necessary_for_x_in_B_l510_51050

def A : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem x_in_A_neither_sufficient_nor_necessary_for_x_in_B : ¬ ((∀ x, x ∈ A → x ∈ B) ∧ (∀ x, x ∈ B → x ∈ A)) := by
  sorry

end NUMINAMATH_GPT_x_in_A_neither_sufficient_nor_necessary_for_x_in_B_l510_51050


namespace NUMINAMATH_GPT_length_breadth_difference_l510_51009

theorem length_breadth_difference (b l : ℕ) (h1 : b = 5) (h2 : l * b = 15 * b) : l - b = 10 :=
by
  sorry

end NUMINAMATH_GPT_length_breadth_difference_l510_51009


namespace NUMINAMATH_GPT_transformed_equation_solutions_l510_51082

theorem transformed_equation_solutions :
  (∀ x : ℝ, x^2 + 2 * x - 3 = 0 → (x = 1 ∨ x = -3)) →
  (∀ x : ℝ, (x + 3)^2 + 2 * (x + 3) - 3 = 0 → (x = -2 ∨ x = -6)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_transformed_equation_solutions_l510_51082


namespace NUMINAMATH_GPT_a_3_eq_5_l510_51084

variable (a : ℕ → ℕ) -- Defines the arithmetic sequence
variable (S : ℕ → ℕ) -- The sum of the first n terms of the sequence

-- Condition: S_5 = 25
axiom S_5_eq_25 : S 5 = 25

-- Define what it means for S to be the sum of the first n terms of the arithmetic sequence
axiom sum_arith_seq : ∀ n, S n = n * (a 1 + a n) / 2

theorem a_3_eq_5 : a 3 = 5 :=
by
  -- Proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_a_3_eq_5_l510_51084


namespace NUMINAMATH_GPT_length_of_BC_l510_51030

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end NUMINAMATH_GPT_length_of_BC_l510_51030


namespace NUMINAMATH_GPT_intersection_A_B_l510_51049

-- Definitions for sets A and B based on the problem conditions
def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.log (2 - x) }

-- Proof problem statement
theorem intersection_A_B : (A ∩ B) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l510_51049


namespace NUMINAMATH_GPT_fraction_irreducible_l510_51056

theorem fraction_irreducible (n : ℤ) : Nat.gcd (18 * n + 3).natAbs (12 * n + 1).natAbs = 1 := 
sorry

end NUMINAMATH_GPT_fraction_irreducible_l510_51056


namespace NUMINAMATH_GPT_largest_k_for_3_in_g_l510_51036

theorem largest_k_for_3_in_g (k : ℝ) :
  (∃ x : ℝ, 2*x^2 - 8*x + k = 3) ↔ k ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_k_for_3_in_g_l510_51036


namespace NUMINAMATH_GPT_average_side_length_of_squares_l510_51090

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end NUMINAMATH_GPT_average_side_length_of_squares_l510_51090


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l510_51094

theorem arithmetic_sequence_sum_ratio (a_n : ℕ → ℕ) (S : ℕ → ℕ) 
  (hS : ∀ n, S n = n * a_n 1 + n * (n - 1) / 2 * (a_n 2 - a_n 1)) 
  (h1 : S 6 / S 3 = 4) : S 9 / S 6 = 9 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l510_51094


namespace NUMINAMATH_GPT_ratio_areas_of_circumscribed_circles_l510_51034

theorem ratio_areas_of_circumscribed_circles (P : ℝ) (A B : ℝ)
  (h1 : ∃ (x : ℝ), P = 8 * x)
  (h2 : ∃ (s : ℝ), s = P / 3)
  (hA : A = (5 * (P^2) * Real.pi) / 128)
  (hB : B = (P^2 * Real.pi) / 27) :
  A / B = 135 / 128 := by
  sorry

end NUMINAMATH_GPT_ratio_areas_of_circumscribed_circles_l510_51034


namespace NUMINAMATH_GPT_smallest_n_l510_51081

/--
Each of \( 2020 \) boxes in a line contains 2 red marbles, 
and for \( 1 \le k \le 2020 \), the box in the \( k \)-th 
position also contains \( k \) white marbles. 

Let \( Q(n) \) be the probability that James stops after 
drawing exactly \( n \) marbles. Prove that the smallest 
value of \( n \) for which \( Q(n) < \frac{1}{2020} \) 
is 31.
-/
theorem smallest_n (Q : ℕ → ℚ) (hQ : ∀ n, Q n = (2 : ℚ) / ((n + 1) * (n + 2)))
  : ∃ n, Q n < 1/2020 ∧ ∀ m < n, Q m ≥ 1/2020 := by
  sorry

end NUMINAMATH_GPT_smallest_n_l510_51081


namespace NUMINAMATH_GPT_line_eq_slope_form_l510_51093

theorem line_eq_slope_form (a b c : ℝ) (h : b ≠ 0) :
    ∃ k l : ℝ, ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (y = k * x + l) := 
sorry

end NUMINAMATH_GPT_line_eq_slope_form_l510_51093


namespace NUMINAMATH_GPT_barney_no_clean_towels_days_l510_51054

theorem barney_no_clean_towels_days
  (wash_cycle_weeks : ℕ := 1)
  (total_towels : ℕ := 18)
  (towels_per_day : ℕ := 2)
  (days_per_week : ℕ := 7)
  (missed_laundry_weeks : ℕ := 1) :
  (days_per_week - (total_towels - (days_per_week * towels_per_day * missed_laundry_weeks)) / towels_per_day) = 5 :=
by
  sorry

end NUMINAMATH_GPT_barney_no_clean_towels_days_l510_51054


namespace NUMINAMATH_GPT_smallest_prime_less_than_square_l510_51035

theorem smallest_prime_less_than_square : ∃ p n : ℕ, Prime p ∧ p = n^2 - 20 ∧ p = 5 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_prime_less_than_square_l510_51035


namespace NUMINAMATH_GPT_polynomial_expansion_l510_51077

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^5 + 5 * t^4 + 2 * t^3 - 26 * t^2 + 25 * t - 12 :=
by sorry

end NUMINAMATH_GPT_polynomial_expansion_l510_51077


namespace NUMINAMATH_GPT_no_prime_divisible_by_77_l510_51000

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_77_l510_51000


namespace NUMINAMATH_GPT_cats_left_l510_51004

theorem cats_left (siamese house persian sold_first sold_second : ℕ) (h1 : siamese = 23) (h2 : house = 17) (h3 : persian = 29) (h4 : sold_first = 40) (h5 : sold_second = 12) :
  siamese + house + persian - sold_first - sold_second = 17 :=
by sorry

end NUMINAMATH_GPT_cats_left_l510_51004


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_of_bn_l510_51013

variable (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ)

theorem arithmetic_sequence (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6) :
  (∀ n, a n = 2 * n) :=
by sorry

theorem sum_of_bn (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6)
                  (h3 : ∀ n, a n = 2 * n)
                  (h4 : ∀ n, b n = 4 / (a n * a (n + 1))) :
  (∀ n, S n = n / (n + 1)) :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_of_bn_l510_51013


namespace NUMINAMATH_GPT_solve_for_y_l510_51060

theorem solve_for_y (y : ℚ) : y - 1 / 2 = 1 / 6 - 2 / 3 + 1 / 4 → y = 1 / 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l510_51060


namespace NUMINAMATH_GPT_integral_value_l510_51076

theorem integral_value (a : ℝ) (h : a = 2) : ∫ x in a..2*Real.exp 1, 1/x = 1 := by
  sorry

end NUMINAMATH_GPT_integral_value_l510_51076


namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l510_51033

-- Define the proposition p
def p : Prop :=
  ∀ x > 0, (x + 1) * Real.exp x > 1

-- Define the negation of the proposition p
def neg_p : Prop :=
  ∃ x > 0, (x + 1) * Real.exp x ≤ 1

-- State the proof problem: negation of p is neg_p
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by
  -- Stating that ¬p is equivalent to neg_p
  sorry

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l510_51033


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l510_51048

open Real

theorem tan_alpha_minus_pi_over_4
  (α : ℝ)
  (a b : ℝ × ℝ)
  (h1 : a = (cos α, -2))
  (h2 : b = (sin α, 1))
  (h3 : ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) :
  tan (α - π / 4) = -3 := 
sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l510_51048


namespace NUMINAMATH_GPT_charlotte_flour_cost_l510_51044

noncomputable def flour_cost 
  (flour_sugar_eggs_butter_cost blueberry_cost cherry_cost total_cost : ℝ)
  (blueberry_weight oz_per_lb blueberry_cost_per_container cherry_weight cherry_cost_per_bag : ℝ)
  (additional_cost : ℝ) : ℝ :=
  total_cost - (blueberry_cost + additional_cost)

theorem charlotte_flour_cost :
  flour_cost 2.5 13.5 14 18 3 16 2.25 4 14 2.5 = 2 :=
by
  unfold flour_cost
  sorry

end NUMINAMATH_GPT_charlotte_flour_cost_l510_51044


namespace NUMINAMATH_GPT_playground_perimeter_l510_51080

theorem playground_perimeter (x y : ℝ) 
  (h1 : x^2 + y^2 = 289) 
  (h2 : x * y = 120) : 
  2 * (x + y) = 46 :=
by 
  sorry

end NUMINAMATH_GPT_playground_perimeter_l510_51080


namespace NUMINAMATH_GPT_polygon_interior_sum_polygon_angle_ratio_l510_51011

-- Part 1: Number of sides based on the sum of interior angles
theorem polygon_interior_sum (n: ℕ) (h: (n - 2) * 180 = 2340) : n = 15 :=
  sorry

-- Part 2: Number of sides based on the ratio of interior to exterior angles
theorem polygon_angle_ratio (n: ℕ) (exterior_angle: ℕ) (ratio: 13 * exterior_angle + 2 * exterior_angle = 180) : n = 15 :=
  sorry

end NUMINAMATH_GPT_polygon_interior_sum_polygon_angle_ratio_l510_51011


namespace NUMINAMATH_GPT_brother_age_l510_51018

variables (M B : ℕ)

theorem brother_age (h1 : M = B + 12) (h2 : M + 2 = 2 * (B + 2)) : B = 10 := by
  sorry

end NUMINAMATH_GPT_brother_age_l510_51018


namespace NUMINAMATH_GPT_farmer_land_l510_51055

noncomputable def farmer_land_example (A : ℝ) : Prop :=
  let cleared_land := 0.90 * A
  let barley_land := 0.70 * cleared_land
  let potatoes_land := 0.10 * cleared_land
  let corn_land := 0.10 * cleared_land
  let tomatoes_bell_peppers_land := 0.10 * cleared_land
  tomatoes_bell_peppers_land = 90 → A = 1000

theorem farmer_land (A : ℝ) (h_cleared_land : 0.90 * A = cleared_land)
  (h_barley_land : 0.70 * cleared_land = barley_land)
  (h_potatoes_land : 0.10 * cleared_land = potatoes_land)
  (h_corn_land : 0.10 * cleared_land = corn_land)
  (h_tomatoes_bell_peppers_land : 0.10 * cleared_land = 90) :
  A = 1000 :=
by
  sorry

end NUMINAMATH_GPT_farmer_land_l510_51055


namespace NUMINAMATH_GPT_proportional_sets_l510_51001

/-- Prove that among the sets of line segments, the ones that are proportional are: -/
theorem proportional_sets : 
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  ∃ a b c d, (a, b, c, d) = C ∧ (a * d = b * c) :=
by
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  sorry

end NUMINAMATH_GPT_proportional_sets_l510_51001


namespace NUMINAMATH_GPT_det_matrix_A_l510_51079

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, 4], ![-2, 3]]

def determinant_2x2 (A : Matrix (Fin 2) (Fin 2) ℤ) : ℤ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

theorem det_matrix_A : determinant_2x2 matrix_A = 32 := by
  sorry

end NUMINAMATH_GPT_det_matrix_A_l510_51079


namespace NUMINAMATH_GPT_number_of_members_l510_51074

-- Definitions based on conditions in the problem
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def cap_cost : ℕ := tshirt_cost

def home_game_cost_per_member : ℕ := sock_cost + tshirt_cost
def away_game_cost_per_member : ℕ := sock_cost + tshirt_cost + cap_cost
def total_cost_per_member : ℕ := home_game_cost_per_member + away_game_cost_per_member

def total_league_cost : ℕ := 4324

-- Statement to be proved
theorem number_of_members (m : ℕ) (h : total_league_cost = m * total_cost_per_member) : m = 85 :=
sorry

end NUMINAMATH_GPT_number_of_members_l510_51074


namespace NUMINAMATH_GPT_quadratic_expression_l510_51023

theorem quadratic_expression (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 6) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := 
by sorry

end NUMINAMATH_GPT_quadratic_expression_l510_51023


namespace NUMINAMATH_GPT_wedding_reception_friends_l510_51025

theorem wedding_reception_friends (total_guests bride_couples groom_couples bride_coworkers groom_coworkers bride_relatives groom_relatives: ℕ)
  (h1: total_guests = 400)
  (h2: bride_couples = 40) 
  (h3: groom_couples = 40)
  (h4: bride_coworkers = 10) 
  (h5: groom_coworkers = 10)
  (h6: bride_relatives = 20)
  (h7: groom_relatives = 20)
  : (total_guests - ((bride_couples + groom_couples) * 2 + (bride_coworkers + groom_coworkers) + (bride_relatives + groom_relatives))) = 180 := 
by 
  sorry

end NUMINAMATH_GPT_wedding_reception_friends_l510_51025


namespace NUMINAMATH_GPT_expression_value_l510_51070

theorem expression_value (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end NUMINAMATH_GPT_expression_value_l510_51070


namespace NUMINAMATH_GPT_angle_between_hands_230_pm_l510_51037

def hour_hand_position (hour minute : ℕ) : ℕ := hour % 12 * 5 + minute / 12
def minute_hand_position (minute : ℕ) : ℕ := minute
def divisions_to_angle (divisions : ℕ) : ℕ := divisions * 30

theorem angle_between_hands_230_pm :
    hour_hand_position 2 30 = 2 * 5 + 30 / 12 ∧
    minute_hand_position 30 = 30 ∧
    divisions_to_angle (minute_hand_position 30 / 5 - hour_hand_position 2 30 / 5) = 105 :=
by {
    sorry
}

end NUMINAMATH_GPT_angle_between_hands_230_pm_l510_51037


namespace NUMINAMATH_GPT_find_k_value_l510_51064

variable (x y z k : ℝ)

theorem find_k_value (h : 7 / (x + y) = k / (x + z) ∧ k / (x + z) = 11 / (z - y)) :
  k = 18 :=
sorry

end NUMINAMATH_GPT_find_k_value_l510_51064


namespace NUMINAMATH_GPT_repeating_decimal_sum_l510_51024

theorem repeating_decimal_sum :
  let x := (1 : ℚ) / 3
  let y := (7 : ℚ) / 33
  x + y = 6 / 11 :=
  by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l510_51024


namespace NUMINAMATH_GPT_combined_salary_l510_51091

theorem combined_salary (S_B : ℝ) (S_A : ℝ) (h1 : S_B = 8000) (h2 : 0.20 * S_A = 0.15 * S_B) : 
S_A + S_B = 14000 :=
by {
  sorry
}

end NUMINAMATH_GPT_combined_salary_l510_51091


namespace NUMINAMATH_GPT_expression_value_l510_51086

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end NUMINAMATH_GPT_expression_value_l510_51086


namespace NUMINAMATH_GPT_staff_members_attended_meeting_l510_51040

theorem staff_members_attended_meeting
  (n_doughnuts_served : ℕ)
  (e_each_staff_member : ℕ)
  (n_doughnuts_left : ℕ)
  (h1 : n_doughnuts_served = 50)
  (h2 : e_each_staff_member = 2)
  (h3 : n_doughnuts_left = 12) :
  (n_doughnuts_served - n_doughnuts_left) / e_each_staff_member = 19 := 
by
  sorry

end NUMINAMATH_GPT_staff_members_attended_meeting_l510_51040


namespace NUMINAMATH_GPT_total_cases_sold_is_correct_l510_51019

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end NUMINAMATH_GPT_total_cases_sold_is_correct_l510_51019


namespace NUMINAMATH_GPT_B_gives_C_100_meters_start_l510_51061

-- Definitions based on given conditions
variables (Va Vb Vc : ℝ) (T : ℝ)

-- Assume the conditions based on the problem statement
def race_condition_1 := Va = 1000 / T
def race_condition_2 := Vb = 900 / T
def race_condition_3 := Vc = 850 / T

-- Theorem stating that B can give C a 100 meter start
theorem B_gives_C_100_meters_start
  (h1 : race_condition_1 Va T)
  (h2 : race_condition_2 Vb T)
  (h3 : race_condition_3 Vc T) :
  (Vb = (1000 - 100) / T) :=
by
  -- Utilize conditions h1, h2, and h3
  sorry

end NUMINAMATH_GPT_B_gives_C_100_meters_start_l510_51061


namespace NUMINAMATH_GPT_no_full_conspiracies_in_same_lab_l510_51078

theorem no_full_conspiracies_in_same_lab
(six_conspiracies : Finset (Finset (Fin 10)))
(h_conspiracies : ∀ c ∈ six_conspiracies, c.card = 3)
(h_total : six_conspiracies.card = 6) :
  ∃ (lab1 lab2 : Finset (Fin 10)), lab1 ∩ lab2 = ∅ ∧ lab1 ∪ lab2 = Finset.univ ∧ ∀ c ∈ six_conspiracies, ¬(c ⊆ lab1 ∨ c ⊆ lab2) :=
by
  sorry

end NUMINAMATH_GPT_no_full_conspiracies_in_same_lab_l510_51078


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l510_51045

theorem sum_of_consecutive_integers (x : ℤ) (h1 : x * (x + 1) + x + (x + 1) = 156) (h2 : x + 1 < 20) : x + (x + 1) = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l510_51045


namespace NUMINAMATH_GPT_constant_k_for_linear_function_l510_51092

theorem constant_k_for_linear_function (k : ℝ) (h : ∀ (x : ℝ), y = x^(k-1) + 2 → y = a * x + b) : k = 2 :=
sorry

end NUMINAMATH_GPT_constant_k_for_linear_function_l510_51092


namespace NUMINAMATH_GPT_no_int_solutions_except_zero_l510_51063

theorem no_int_solutions_except_zero 
  (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_int_solutions_except_zero_l510_51063


namespace NUMINAMATH_GPT_inequality_a2b3c_l510_51020

theorem inequality_a2b3c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end NUMINAMATH_GPT_inequality_a2b3c_l510_51020


namespace NUMINAMATH_GPT_proportion_calculation_l510_51083

theorem proportion_calculation (x y : ℝ) (h1 : 0.75 / x = 5 / y) (h2 : x = 1.2) : y = 8 :=
by
  sorry

end NUMINAMATH_GPT_proportion_calculation_l510_51083


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l510_51099

theorem quadratic_inequality_solution : ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l510_51099


namespace NUMINAMATH_GPT_sequence_general_term_correctness_l510_51052

def sequenceGeneralTerm (n : ℕ) : ℤ :=
  if n % 2 = 1 then
    0
  else
    (-1) ^ (n / 2 + 1)

theorem sequence_general_term_correctness (n : ℕ) :
  (∀ m, sequenceGeneralTerm m = 0 ↔ m % 2 = 1) ∧
  (∀ k, sequenceGeneralTerm k = (-1) ^ (k / 2 + 1) ↔ k % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_correctness_l510_51052


namespace NUMINAMATH_GPT_least_m_value_l510_51069

def recursive_sequence (x : ℕ → ℚ) : Prop :=
  x 0 = 3 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_value (x : ℕ → ℚ) (h : recursive_sequence x) : ∃ m, m > 0 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k, k > 0 → k < m → x k > 3 + 1 / 2^10 :=
sorry

end NUMINAMATH_GPT_least_m_value_l510_51069


namespace NUMINAMATH_GPT_gcd_a_b_l510_51085

noncomputable def a : ℕ := 3333333
noncomputable def b : ℕ := 666666666

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_a_b_l510_51085


namespace NUMINAMATH_GPT_cone_volume_l510_51038

theorem cone_volume (lateral_area : ℝ) (angle : ℝ) 
  (h₀ : lateral_area = 20 * Real.pi)
  (h₁ : angle = Real.arccos (4/5)) : 
  (1/3) * Real.pi * (4^2) * 3 = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l510_51038


namespace NUMINAMATH_GPT_vasya_correct_l510_51008

-- Define the condition of a convex quadrilateral
def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a < 180 ∧ b < 180 ∧ c < 180 ∧ d < 180

-- Define the properties of forming two types of triangles from a quadrilateral
def can_form_two_acute_triangles (a b c d : ℝ) : Prop :=
  a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90

def can_form_two_right_triangles (a b c d : ℝ) : Prop :=
  (a = 90 ∧ b = 90) ∨ (b = 90 ∧ c = 90) ∨ (c = 90 ∧ d = 90) ∨ (d = 90 ∧ a = 90)

def can_form_two_obtuse_triangles (a b c d : ℝ) : Prop :=
  ∃ x y z w, (x > 90 ∧ y < 90 ∧ z < 90 ∧ w < 90 ∧ (x + y + z + w = 360)) ∧
             (x > 90 ∨ y > 90 ∨ z > 90 ∨ w > 90)

-- Prove that Vasya's claim is definitively correct
theorem vasya_correct (a b c d : ℝ) (h : convex_quadrilateral a b c d) :
  can_form_two_obtuse_triangles a b c d ∧
  ¬(can_form_two_acute_triangles a b c d) ∧
  ¬(can_form_two_right_triangles a b c d) ∨
  can_form_two_right_triangles a b c d ∧
  can_form_two_obtuse_triangles a b c d := sorry

end NUMINAMATH_GPT_vasya_correct_l510_51008


namespace NUMINAMATH_GPT_man_l510_51043

-- Define all given conditions using Lean definitions
def speed_with_current_wind : ℝ := 22
def speed_of_current : ℝ := 5
def wind_resistance_factor : ℝ := 0.15
def current_increase_factor : ℝ := 0.10

-- Define the key quantities (man's speed in still water, effective speed in still water, new current speed against)
def speed_in_still_water : ℝ := speed_with_current_wind - speed_of_current
def effective_speed_in_still_water : ℝ := speed_in_still_water - (wind_resistance_factor * speed_in_still_water)
def new_speed_of_current_against : ℝ := speed_of_current + (current_increase_factor * speed_of_current)

-- Proof goal: Prove that the man's speed against the current is 8.95 km/hr considering all the conditions
theorem man's_speed_against_current_is_correct : 
  (effective_speed_in_still_water - new_speed_of_current_against) = 8.95 := 
by
  sorry

end NUMINAMATH_GPT_man_l510_51043


namespace NUMINAMATH_GPT_probability_of_picking_red_ball_l510_51016

theorem probability_of_picking_red_ball (w r : ℕ) 
  (h1 : r > w) 
  (h2 : r < 2 * w) 
  (h3 : 2 * w + 3 * r = 60) : 
  r / (w + r) = 7 / 11 :=
sorry

end NUMINAMATH_GPT_probability_of_picking_red_ball_l510_51016


namespace NUMINAMATH_GPT_range_f_l510_51073

noncomputable def g (x : ℝ) : ℝ := 30 + 14 * Real.cos x - 7 * Real.cos (2 * x)

noncomputable def z (t : ℝ) : ℝ := 40.5 - 14 * (t - 0.5) ^ 2

noncomputable def u (z : ℝ) : ℝ := (Real.pi / 54) * z

noncomputable def f (x : ℝ) : ℝ := Real.sin (u (z (Real.cos x)))

theorem range_f : ∀ x : ℝ, 0.5 ≤ f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_range_f_l510_51073


namespace NUMINAMATH_GPT_purely_imaginary_iff_real_iff_second_quadrant_iff_l510_51089

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_GPT_purely_imaginary_iff_real_iff_second_quadrant_iff_l510_51089


namespace NUMINAMATH_GPT_num_distinct_orders_of_targets_l510_51058

theorem num_distinct_orders_of_targets : 
  let total_targets := 10
  let column_A_targets := 4
  let column_B_targets := 4
  let column_C_targets := 2
  (Nat.factorial total_targets) / 
  ((Nat.factorial column_A_targets) * (Nat.factorial column_B_targets) * (Nat.factorial column_C_targets)) = 5040 := 
by
  sorry

end NUMINAMATH_GPT_num_distinct_orders_of_targets_l510_51058
