import Mathlib

namespace NUMINAMATH_GPT_Brady_average_hours_l1740_174041

-- Definitions based on conditions
def hours_per_day_April : ℕ := 6
def hours_per_day_June : ℕ := 5
def hours_per_day_September : ℕ := 8
def days_in_April : ℕ := 30
def days_in_June : ℕ := 30
def days_in_September : ℕ := 30

-- Definition to prove
def average_hours_per_month : ℕ := 190

-- Theorem statement
theorem Brady_average_hours :
  (hours_per_day_April * days_in_April + hours_per_day_June * days_in_June + hours_per_day_September * days_in_September) / 3 = average_hours_per_month :=
sorry

end NUMINAMATH_GPT_Brady_average_hours_l1740_174041


namespace NUMINAMATH_GPT_tree_ratio_l1740_174044

theorem tree_ratio (native_trees : ℕ) (total_planted : ℕ) (M : ℕ) 
  (h1 : native_trees = 30) 
  (h2 : total_planted = 80) 
  (h3 : total_planted = M + M / 3) :
  (native_trees + M) / native_trees = 3 :=
sorry

end NUMINAMATH_GPT_tree_ratio_l1740_174044


namespace NUMINAMATH_GPT_find_f2_l1740_174003

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (f x) = (x ^ 2 - x) / 2 * f x + 2 - x

theorem find_f2 : f 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l1740_174003


namespace NUMINAMATH_GPT_parallel_perpendicular_implies_l1740_174088

variables {Line : Type} {Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom distinct_lines : m ≠ n
axiom distinct_planes : α ≠ β

-- Parallel and Perpendicular relationships
axiom parallel : Line → Plane → Prop
axiom perpendicular : Line → Plane → Prop

-- Given conditions
axiom parallel_mn : parallel m n
axiom perpendicular_mα : perpendicular m α

-- Proof statement
theorem parallel_perpendicular_implies (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α :=
sorry

end NUMINAMATH_GPT_parallel_perpendicular_implies_l1740_174088


namespace NUMINAMATH_GPT_length_of_arc_correct_l1740_174082

open Real

noncomputable def length_of_arc (r θ : ℝ) := θ * r

theorem length_of_arc_correct (A r θ : ℝ) (hA : A = (θ / (2 * π)) * (π * r^2)) (hr : r = 5) (hA_val : A = 13.75) :
  length_of_arc r θ = 5.5 :=
by
  -- Proof steps are omitted
  sorry

end NUMINAMATH_GPT_length_of_arc_correct_l1740_174082


namespace NUMINAMATH_GPT_distinct_midpoints_at_least_2n_minus_3_l1740_174092

open Set

theorem distinct_midpoints_at_least_2n_minus_3 
  (n : ℕ) 
  (points : Finset (ℝ × ℝ)) 
  (h_points_card : points.card = n) :
  ∃ (midpoints : Finset (ℝ × ℝ)), 
    midpoints.card ≥ 2 * n - 3 := 
sorry

end NUMINAMATH_GPT_distinct_midpoints_at_least_2n_minus_3_l1740_174092


namespace NUMINAMATH_GPT_problem_statement_l1740_174020

theorem problem_statement (x y : ℝ) (h₁ : 2.5 * x = 0.75 * y) (h₂ : x = 20) : y = 200 / 3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1740_174020


namespace NUMINAMATH_GPT_f_specification_l1740_174065

open Function

def f : ℕ → ℕ := sorry -- define function f here

axiom f_involution (n : ℕ) : f (f n) = n

axiom f_functional_property (n : ℕ) : f (f n + 1) = if n % 2 = 0 then n - 1 else n + 3

axiom f_bijective : Bijective f

axiom f_not_two (n : ℕ) : f (f n + 1) ≠ 2

axiom f_one_eq_two : f 1 = 2

theorem f_specification (n : ℕ) : 
  f n = if n % 2 = 1 then n + 1 else n - 1 :=
sorry

end NUMINAMATH_GPT_f_specification_l1740_174065


namespace NUMINAMATH_GPT_find_coordinates_l1740_174021

def point_in_fourth_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 < 0
def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.2| = d
def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.1| = d

theorem find_coordinates :
  ∃ P : ℝ × ℝ, point_in_fourth_quadrant P ∧ distance_to_x_axis P 2 ∧ distance_to_y_axis P 5 ∧ P = (5, -2) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_l1740_174021


namespace NUMINAMATH_GPT_inequality_problem_l1740_174087

theorem inequality_problem
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
sorry

end NUMINAMATH_GPT_inequality_problem_l1740_174087


namespace NUMINAMATH_GPT_schedule_courses_l1740_174033

-- Define the number of courses and periods
def num_courses : Nat := 4
def num_periods : Nat := 8

-- Define the total number of ways to schedule courses without restrictions
def unrestricted_schedules : Nat := Nat.choose num_periods num_courses * Nat.factorial num_courses

-- Define the number of invalid schedules using PIE (approximate value given in problem)
def invalid_schedules : Nat := 1008 + 180 + 120

-- Define the number of valid schedules
def valid_schedules : Nat := unrestricted_schedules - invalid_schedules

theorem schedule_courses : valid_schedules = 372 := sorry

end NUMINAMATH_GPT_schedule_courses_l1740_174033


namespace NUMINAMATH_GPT_winning_percentage_votes_l1740_174001

theorem winning_percentage_votes (P : ℝ) (votes_total : ℝ) (majority_votes : ℝ) (winning_votes : ℝ) : 
  votes_total = 4500 → majority_votes = 900 → 
  winning_votes = (P / 100) * votes_total → 
  majority_votes = winning_votes - ((100 - P) / 100) * votes_total → P = 60 := 
by
  intros h_total h_majority h_winning_votes h_majority_eq
  sorry

end NUMINAMATH_GPT_winning_percentage_votes_l1740_174001


namespace NUMINAMATH_GPT_Tim_younger_than_Jenny_l1740_174064

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end NUMINAMATH_GPT_Tim_younger_than_Jenny_l1740_174064


namespace NUMINAMATH_GPT_average_pastries_per_day_l1740_174059

-- Conditions
def pastries_on_monday := 2

def pastries_on_day (n : ℕ) : ℕ :=
  pastries_on_monday + n

def total_pastries_in_week : ℕ :=
  List.sum (List.map pastries_on_day (List.range 7))

def number_of_days_in_week : ℕ := 7

-- Theorem to prove
theorem average_pastries_per_day : (total_pastries_in_week / number_of_days_in_week) = 5 :=
by
  sorry

end NUMINAMATH_GPT_average_pastries_per_day_l1740_174059


namespace NUMINAMATH_GPT_find_y_value_l1740_174052

theorem find_y_value :
  ∀ (y : ℝ), (dist (1, 3) (7, y) = 13) ∧ (y > 0) → y = 3 + Real.sqrt 133 :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l1740_174052


namespace NUMINAMATH_GPT_range_of_a_l1740_174054

noncomputable def p (a : ℝ) : Prop :=
∀ (x : ℝ), x > -1 → (x^2) / (x + 1) ≥ a

noncomputable def q (a : ℝ) : Prop :=
∃ (x : ℝ), (a*x^2 - a*x + 1 = 0)

theorem range_of_a (a : ℝ) :
  ¬ p a ∧ ¬ q a ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1740_174054


namespace NUMINAMATH_GPT_trigonometric_identity_1_l1740_174010

theorem trigonometric_identity_1 :
  ( (Real.sqrt 3 * Real.sin (-1200 * Real.pi / 180)) / (Real.tan (11 * Real.pi / 3)) 
  - Real.cos (585 * Real.pi / 180) * Real.tan (-37 * Real.pi / 4) = (Real.sqrt 3 / 2) - (Real.sqrt 2 / 2) ) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_1_l1740_174010


namespace NUMINAMATH_GPT_power_multiplication_eq_neg4_l1740_174095

theorem power_multiplication_eq_neg4 :
  (-0.25) ^ 11 * (-4) ^ 12 = -4 := 
  sorry

end NUMINAMATH_GPT_power_multiplication_eq_neg4_l1740_174095


namespace NUMINAMATH_GPT_complete_consoles_production_rate_l1740_174025

-- Define the production rates of each chip
def production_rate_A := 467
def production_rate_B := 413
def production_rate_C := 532
def production_rate_D := 356
def production_rate_E := 494

-- Define the maximum number of consoles that can be produced per day
def max_complete_consoles (A B C D E : ℕ) := min (min (min (min A B) C) D) E

-- Statement
theorem complete_consoles_production_rate :
  max_complete_consoles production_rate_A production_rate_B production_rate_C production_rate_D production_rate_E = 356 :=
by
  sorry

end NUMINAMATH_GPT_complete_consoles_production_rate_l1740_174025


namespace NUMINAMATH_GPT_houses_without_garage_nor_pool_l1740_174081

def total_houses : ℕ := 85
def houses_with_garage : ℕ := 50
def houses_with_pool : ℕ := 40
def houses_with_both : ℕ := 35
def neither_garage_nor_pool : ℕ := 30

theorem houses_without_garage_nor_pool :
  total_houses - (houses_with_garage + houses_with_pool - houses_with_both) = neither_garage_nor_pool :=
by
  sorry

end NUMINAMATH_GPT_houses_without_garage_nor_pool_l1740_174081


namespace NUMINAMATH_GPT_polynomial_divisibility_l1740_174074

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, (x-1)^3 ∣ x^4 + a * x^2 + b * x + c) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1740_174074


namespace NUMINAMATH_GPT_sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l1740_174093

theorem sum_of_two_terms_is_term_iff_a_is_multiple_of_d
    (a d : ℤ) 
    (n k : ℕ) 
    (h : ∀ (p : ℕ), a + d * n + (a + d * k) = a + d * p)
    : ∃ m : ℤ, a = d * m :=
sorry

end NUMINAMATH_GPT_sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l1740_174093


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1740_174019

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 5 = 16)
  (h_pos : ∀ n : ℕ, 0 < a n) :
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1740_174019


namespace NUMINAMATH_GPT_squirrel_travel_distance_l1740_174073

def squirrel_distance (height : ℕ) (circumference : ℕ) (rise_per_circuit : ℕ) : ℕ :=
  let circuits := height / rise_per_circuit
  let horizontal_distance := circuits * circumference
  Nat.sqrt (height * height + horizontal_distance * horizontal_distance)

theorem squirrel_travel_distance :
  (squirrel_distance 16 3 4) = 20 := by
  sorry

end NUMINAMATH_GPT_squirrel_travel_distance_l1740_174073


namespace NUMINAMATH_GPT_digit_agreement_l1740_174048

theorem digit_agreement (N : ℕ) (abcd : ℕ) (h1 : N % 10000 = abcd) (h2 : N ^ 2 % 10000 = abcd) (h3 : ∃ a b c d, abcd = a * 1000 + b * 100 + c * 10 + d ∧ a ≠ 0) : abcd / 10 = 937 := sorry

end NUMINAMATH_GPT_digit_agreement_l1740_174048


namespace NUMINAMATH_GPT_a_81_eq_640_l1740_174046

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 0 -- auxiliary value since sequence begins from n=1
else if n = 1 then 1
else (2 * n - 1) ^ 2 - (2 * n - 3) ^ 2

theorem a_81_eq_640 : sequence_a 81 = 640 :=
by
  sorry

end NUMINAMATH_GPT_a_81_eq_640_l1740_174046


namespace NUMINAMATH_GPT_smallest_x_value_l1740_174098

theorem smallest_x_value {x : ℝ} (h : abs (x + 4) = 15) : x = -19 :=
sorry

end NUMINAMATH_GPT_smallest_x_value_l1740_174098


namespace NUMINAMATH_GPT_concurrent_segments_unique_solution_l1740_174060

theorem concurrent_segments_unique_solution (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  4^c - 1 = (2^a - 1) * (2^b - 1) ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
by
  sorry

end NUMINAMATH_GPT_concurrent_segments_unique_solution_l1740_174060


namespace NUMINAMATH_GPT_Kelly_needs_to_give_away_l1740_174047

variable (n k : Nat)

theorem Kelly_needs_to_give_away (h_n : n = 20) (h_k : k = 12) : n - k = 8 := 
by
  sorry

end NUMINAMATH_GPT_Kelly_needs_to_give_away_l1740_174047


namespace NUMINAMATH_GPT_hyperbola_distance_to_foci_l1740_174089

theorem hyperbola_distance_to_foci
  (E : ∀ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1)
  (F1 F2 : ℝ)
  (P : ℝ)
  (dist_PF1 : P = 5)
  (a : ℝ)
  (ha : a = 3): 
  |P - F2| = 11 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_distance_to_foci_l1740_174089


namespace NUMINAMATH_GPT_expression_value_l1740_174038

open Real

theorem expression_value :
  3 + sqrt 3 + 1 / (3 + sqrt 3) + 1 / (sqrt 3 - 3) = 3 + 2 * sqrt 3 / 3 := 
sorry

end NUMINAMATH_GPT_expression_value_l1740_174038


namespace NUMINAMATH_GPT_calculation_correct_l1740_174099

theorem calculation_correct :
  (-1 : ℝ)^51 + (2 : ℝ)^(4^2 + 5^2 - 7^2) = -(127 / 128) := 
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1740_174099


namespace NUMINAMATH_GPT_find_leak_rate_l1740_174049

-- Conditions in Lean 4
def pool_capacity : ℝ := 60
def hose_rate : ℝ := 1.6
def fill_time : ℝ := 40

-- Define the leak rate calculation
def leak_rate (L : ℝ) : Prop :=
  pool_capacity = (hose_rate - L) * fill_time

-- The main theorem we want to prove
theorem find_leak_rate : ∃ L, leak_rate L ∧ L = 0.1 := by
  sorry

end NUMINAMATH_GPT_find_leak_rate_l1740_174049


namespace NUMINAMATH_GPT_factor_theorem_l1740_174024

noncomputable def polynomial_to_factor : Prop :=
  ∀ x : ℝ, x^4 - 4 * x^2 + 4 = (x^2 - 2)^2

theorem factor_theorem : polynomial_to_factor :=
by
  sorry

end NUMINAMATH_GPT_factor_theorem_l1740_174024


namespace NUMINAMATH_GPT_find_a1_l1740_174035

theorem find_a1 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h2 : a 2 = 2)
  : a 1 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a1_l1740_174035


namespace NUMINAMATH_GPT_cuboid_volume_l1740_174072

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) : a * b * c = 6 := by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l1740_174072


namespace NUMINAMATH_GPT_jars_of_pickled_mangoes_l1740_174080

def total_mangoes := 54
def ratio_ripe := 1/3
def ratio_unripe := 2/3
def kept_unripe_mangoes := 16
def mangoes_per_jar := 4

theorem jars_of_pickled_mangoes : 
  (total_mangoes * ratio_unripe - kept_unripe_mangoes) / mangoes_per_jar = 5 :=
by
  sorry

end NUMINAMATH_GPT_jars_of_pickled_mangoes_l1740_174080


namespace NUMINAMATH_GPT_reciprocal_sum_greater_l1740_174029

theorem reciprocal_sum_greater (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
    (1 / a + 1 / b) > 1 / (a + b) :=
sorry

end NUMINAMATH_GPT_reciprocal_sum_greater_l1740_174029


namespace NUMINAMATH_GPT_tom_age_is_19_l1740_174007

-- Define the ages of Carla, Tom, Dave, and Emily
variable (C : ℕ) -- Carla's age

-- Conditions
def tom_age := 2 * C - 1
def dave_age := C + 3
def emily_age := C / 2

-- Sum of their ages equating to 48
def total_age := C + tom_age C + dave_age C + emily_age C

-- Theorem to be proven
theorem tom_age_is_19 (h : total_age C = 48) : tom_age C = 19 := 
by {
  sorry
}

end NUMINAMATH_GPT_tom_age_is_19_l1740_174007


namespace NUMINAMATH_GPT_solution_to_fraction_problem_l1740_174056

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end NUMINAMATH_GPT_solution_to_fraction_problem_l1740_174056


namespace NUMINAMATH_GPT_workers_to_build_cars_l1740_174079

theorem workers_to_build_cars (W : ℕ) (hW : W > 0) : 
  (∃ D : ℝ, D = 63 / W) :=
by
  sorry

end NUMINAMATH_GPT_workers_to_build_cars_l1740_174079


namespace NUMINAMATH_GPT_solve_inequality_l1740_174084

theorem solve_inequality :
  { x : ℝ | (9 * x^2 + 27 * x - 64) / ((3 * x - 4) * (x + 5) * (x - 1)) < 4 } = 
    { x : ℝ | -5 < x ∧ x < -17 / 3 } ∪ { x : ℝ | 1 < x ∧ x < 4 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1740_174084


namespace NUMINAMATH_GPT_evaluate_expression_l1740_174043

theorem evaluate_expression (a b c : ℚ) (h1 : c = b - 8) (h2 : b = a + 3) (h3 : a = 2) 
  (h4 : a + 1 ≠ 0) (h5 : b - 3 ≠ 0) (h6 : c + 5 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 7) / (c + 5) = 20 / 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1740_174043


namespace NUMINAMATH_GPT_length_rest_of_body_l1740_174008

theorem length_rest_of_body (height legs head arms rest_of_body : ℝ) 
  (hlegs : legs = (1/3) * height)
  (hhead : head = (1/4) * height)
  (harms : arms = (1/5) * height)
  (htotal : height = 180)
  (hr: rest_of_body = height - (legs + head + arms)) : 
  rest_of_body = 39 :=
by
  -- proof is not required
  sorry

end NUMINAMATH_GPT_length_rest_of_body_l1740_174008


namespace NUMINAMATH_GPT_cube_painted_probability_l1740_174058

theorem cube_painted_probability :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 76
  let total_ways := Nat.choose total_cubes 2
  let favorable_ways := cubes_with_3_faces * cubes_with_no_faces
  let probability := (favorable_ways : ℚ) / total_ways
  probability = (2 : ℚ) / 205 :=
by
  sorry

end NUMINAMATH_GPT_cube_painted_probability_l1740_174058


namespace NUMINAMATH_GPT_trigonometric_comparison_l1740_174042

open Real

theorem trigonometric_comparison :
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  a < b ∧ b < c := 
by
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  sorry

end NUMINAMATH_GPT_trigonometric_comparison_l1740_174042


namespace NUMINAMATH_GPT_quadratic_equation_value_l1740_174096

theorem quadratic_equation_value (a : ℝ) (h₁ : a^2 - 2 = 2) (h₂ : a ≠ 2) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_value_l1740_174096


namespace NUMINAMATH_GPT_students_received_B_l1740_174085

theorem students_received_B (charles_ratio : ℚ) (dawsons_class : ℕ) 
  (h_charles_ratio : charles_ratio = 3 / 5) (h_dawsons_class : dawsons_class = 30) : 
  ∃ y : ℕ, (charles_ratio = y / dawsons_class) ∧ y = 18 := 
by 
  sorry

end NUMINAMATH_GPT_students_received_B_l1740_174085


namespace NUMINAMATH_GPT_volume_ratio_of_smaller_snowball_l1740_174063

theorem volume_ratio_of_smaller_snowball (r : ℝ) (k : ℝ) :
  let V₀ := (4/3) * π * r^3
  let S := 4 * π * r^2
  let V_large := (4/3) * π * (2 * r)^3
  let V_large_half := V_large / 2
  let new_r := (V_large_half / ((4/3) * π))^(1/3)
  let reduction := 2*r - new_r
  let remaining_r := r - reduction
  let remaining_V := (4/3) * π * remaining_r^3
  let volume_ratio := remaining_V / V₀ 
  volume_ratio = 1/5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_volume_ratio_of_smaller_snowball_l1740_174063


namespace NUMINAMATH_GPT_sum_of_fractions_l1740_174051

theorem sum_of_fractions :
  (1 / 3) + (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-9 / 20) = -9 / 20 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1740_174051


namespace NUMINAMATH_GPT_present_population_l1740_174076

theorem present_population (P : ℝ)
  (h1 : P + 0.10 * P = 242) :
  P = 220 := 
sorry

end NUMINAMATH_GPT_present_population_l1740_174076


namespace NUMINAMATH_GPT_side_length_estimate_l1740_174078

theorem side_length_estimate (x : ℝ) (h : x^2 = 15) : 3 < x ∧ x < 4 :=
sorry

end NUMINAMATH_GPT_side_length_estimate_l1740_174078


namespace NUMINAMATH_GPT_percentage_students_below_8_years_l1740_174055

theorem percentage_students_below_8_years :
  ∀ (n8 : ℕ) (n_gt8 : ℕ) (n_total : ℕ),
  n8 = 24 →
  n_gt8 = 2 * n8 / 3 →
  n_total = 50 →
  (n_total - (n8 + n_gt8)) * 100 / n_total = 20 :=
by
  intros n8 n_gt8 n_total h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_students_below_8_years_l1740_174055


namespace NUMINAMATH_GPT_problem_expression_value_l1740_174067

variable (m n p q : ℝ)
variable (h1 : m + n = 0) (h2 : m / n = -1)
variable (h3 : p * q = 1) (h4 : m ≠ n)

theorem problem_expression_value : 
  (m + n) / m + 2 * p * q - m / n = 3 :=
by sorry

end NUMINAMATH_GPT_problem_expression_value_l1740_174067


namespace NUMINAMATH_GPT_P_div_by_Q_iff_l1740_174086

def P (x : ℂ) (n : ℕ) : ℂ := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) : ℂ := x^4 + x^3 + x^2 + x + 1

theorem P_div_by_Q_iff (n : ℕ) : (Q x ∣ P x n) ↔ ¬(5 ∣ n) := sorry

end NUMINAMATH_GPT_P_div_by_Q_iff_l1740_174086


namespace NUMINAMATH_GPT_janna_sleep_hours_l1740_174097

-- Define the sleep hours from Monday to Sunday with the specified conditions
def sleep_hours_monday : ℕ := 7
def sleep_hours_tuesday : ℕ := 7 + 1 / 2
def sleep_hours_wednesday : ℕ := 7
def sleep_hours_thursday : ℕ := 7 + 1 / 2
def sleep_hours_friday : ℕ := 7 + 1
def sleep_hours_saturday : ℕ := 8
def sleep_hours_sunday : ℕ := 8

-- Calculate the total sleep hours in a week
noncomputable def total_sleep_hours : ℕ :=
  sleep_hours_monday +
  sleep_hours_tuesday +
  sleep_hours_wednesday +
  sleep_hours_thursday +
  sleep_hours_friday +
  sleep_hours_saturday +
  sleep_hours_sunday

-- The statement we want to prove
theorem janna_sleep_hours : total_sleep_hours = 53 := by
  sorry

end NUMINAMATH_GPT_janna_sleep_hours_l1740_174097


namespace NUMINAMATH_GPT_total_amount_spent_l1740_174004

variables (D B : ℝ)

-- Conditions
def condition1 : Prop := B = 1.5 * D
def condition2 : Prop := D = B - 15

-- Question: Prove that the total amount they spent together is 75.00
theorem total_amount_spent (h1 : condition1 D B) (h2 : condition2 D B) : B + D = 75 :=
sorry

end NUMINAMATH_GPT_total_amount_spent_l1740_174004


namespace NUMINAMATH_GPT_remainder_div_180_l1740_174006

theorem remainder_div_180 {j : ℕ} (h1 : 0 < j) (h2 : 120 % (j^2) = 12) : 180 % j = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_180_l1740_174006


namespace NUMINAMATH_GPT_findLastNames_l1740_174066

noncomputable def peachProblem : Prop :=
  ∃ (a b c d : ℕ),
    2 * a + 3 * b + 4 * c + 5 * d = 32 ∧
    a + b + c + d = 10 ∧
    (a = 3 ∧ b = 4 ∧ c = 1 ∧ d = 2)

theorem findLastNames :
  peachProblem :=
sorry

end NUMINAMATH_GPT_findLastNames_l1740_174066


namespace NUMINAMATH_GPT_participants_with_exactly_five_problems_l1740_174040

theorem participants_with_exactly_five_problems (n : ℕ) 
  (p : Fin 6 → Fin 6 → ℕ)
  (h1 : ∀ i j : Fin 6, i ≠ j → p i j > 2 * n / 5)
  (h2 : ¬ ∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → p i j = n)
  : ∃ k1 k2 : Fin n, k1 ≠ k2 ∧ (∀ i : Fin 6, (p i k1 = 5) ∧ (p i k2 = 5)) :=
sorry

end NUMINAMATH_GPT_participants_with_exactly_five_problems_l1740_174040


namespace NUMINAMATH_GPT_find_loan_amount_l1740_174023

-- Define the conditions
def rate_of_interest : ℝ := 0.06
def time_period : ℝ := 6
def interest_paid : ℝ := 432

-- Define the simple interest formula
def simple_interest (P r t : ℝ) : ℝ := P * r * t

-- State the theorem to prove the loan amount
theorem find_loan_amount (P : ℝ) (h1 : rate_of_interest = 0.06) (h2 : time_period = 6) (h3 : interest_paid = 432) (h4 : simple_interest P rate_of_interest time_period = interest_paid) : P = 1200 :=
by
  -- Here should be the proof, but it's omitted for now
  sorry

end NUMINAMATH_GPT_find_loan_amount_l1740_174023


namespace NUMINAMATH_GPT_max_value_of_2a_plus_b_l1740_174026

variable (a b : ℝ)

def cond1 := 4 * a + 3 * b ≤ 10
def cond2 := 3 * a + 5 * b ≤ 11

theorem max_value_of_2a_plus_b : 
  cond1 a b → 
  cond2 a b → 
  2 * a + b ≤ 48 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_of_2a_plus_b_l1740_174026


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1740_174061

theorem arithmetic_sequence_ninth_term (a d : ℤ) 
    (h5 : a + 4 * d = 23) (h7 : a + 6 * d = 37) : 
    a + 8 * d = 51 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1740_174061


namespace NUMINAMATH_GPT_train_speed_l1740_174028

theorem train_speed (x : ℝ) (v : ℝ) 
  (h1 : (x / 50) + (2 * x / v) = 3 * x / 25) : v = 20 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1740_174028


namespace NUMINAMATH_GPT_verify_incorrect_option_l1740_174032

variable (a : ℕ → ℝ) -- The sequence a_n
variable (S : ℕ → ℝ) -- The sum of the first n terms S_n

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℝ) : Prop := S 5 < S 6

def condition_2 (S : ℕ → ℝ) : Prop := S 6 = S 7 ∧ S 7 > S 8

theorem verify_incorrect_option (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond1 : condition_1 S)
  (h_cond2 : condition_2 S) :
  S 9 ≤ S 5 :=
sorry

end NUMINAMATH_GPT_verify_incorrect_option_l1740_174032


namespace NUMINAMATH_GPT_probability_of_70th_percentile_is_25_over_56_l1740_174091

-- Define the weights of the students
def weights : List ℕ := [90, 100, 110, 120, 140, 150, 150, 160]

-- Define the number of students to select
def n_selected_students : ℕ := 3

-- Define the percentile value
def percentile_value : ℕ := 70

-- Define the corresponding weight for the 70th percentile
def percentile_weight : ℕ := 150

-- Define the combination function
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability calculation
noncomputable def probability_70th_percentile : ℚ :=
  let total_ways := C 8 3
  let favorable_ways := (C 2 2) * (C 5 1) + (C 2 1) * (C 5 2)
  favorable_ways / total_ways

-- Define the theorem to prove the probability
theorem probability_of_70th_percentile_is_25_over_56 :
  probability_70th_percentile = 25 / 56 := by
  sorry

end NUMINAMATH_GPT_probability_of_70th_percentile_is_25_over_56_l1740_174091


namespace NUMINAMATH_GPT_decimal_to_base7_l1740_174030

theorem decimal_to_base7 :
    ∃ k₀ k₁ k₂ k₃ k₄, 1987 = k₀ * 7^4 + k₁ * 7^3 + k₂ * 7^2 + k₃ * 7^1 + k₄ * 7^0 ∧
    k₀ = 0 ∧
    k₁ = 5 ∧
    k₂ = 3 ∧
    k₃ = 5 ∧
    k₄ = 6 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_base7_l1740_174030


namespace NUMINAMATH_GPT_min_value_of_expression_l1740_174016

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 3) : 
  ∃ k : ℝ, k = 4 + 2 * Real.sqrt 3 ∧ ∀ z, (z = (1 / (x - 1) + 3 / (y - 1))) → z ≥ k :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1740_174016


namespace NUMINAMATH_GPT_ab_sum_eq_2_l1740_174015

theorem ab_sum_eq_2 (a b : ℝ) (M : Set ℝ) (N : Set ℝ) (f : ℝ → ℝ) 
  (hM : M = {b / a, 1})
  (hN : N = {a, 0})
  (hf : ∀ x ∈ M, f x ∈ N)
  (f_def : ∀ x, f x = 2 * x) :
  a + b = 2 :=
by
  -- proof goes here.
  sorry

end NUMINAMATH_GPT_ab_sum_eq_2_l1740_174015


namespace NUMINAMATH_GPT_find_root_of_quadratic_equation_l1740_174036

theorem find_root_of_quadratic_equation
  (a b c : ℝ)
  (h1 : 3 * a * (2 * b - 3 * c) ≠ 0)
  (h2 : 2 * b * (3 * c - 2 * a) ≠ 0)
  (h3 : 5 * c * (2 * a - 3 * b) ≠ 0)
  (r : ℝ)
  (h_roots : (r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) ∨ (r = (-2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) * 2)) :
  r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c)) :=
by
  sorry

end NUMINAMATH_GPT_find_root_of_quadratic_equation_l1740_174036


namespace NUMINAMATH_GPT_log_216_eq_3_log_2_add_3_log_3_l1740_174070

theorem log_216_eq_3_log_2_add_3_log_3 (log : ℝ → ℝ) (h1 : ∀ x y, log (x * y) = log x + log y)
  (h2 : ∀ x n, log (x^n) = n * log x) :
  log 216 = 3 * log 2 + 3 * log 3 :=
by
  sorry

end NUMINAMATH_GPT_log_216_eq_3_log_2_add_3_log_3_l1740_174070


namespace NUMINAMATH_GPT_num_real_roots_eq_two_l1740_174068

theorem num_real_roots_eq_two : 
  ∀ x : ℝ, (∃ r : ℕ, r = 2 ∧ (abs (x^2 - 1) = 1/10 * (x + 9/10) → x = r)) := sorry

end NUMINAMATH_GPT_num_real_roots_eq_two_l1740_174068


namespace NUMINAMATH_GPT_max_value_of_a_l1740_174002

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - a * x + 1

theorem max_value_of_a :
  ∃ (a : ℝ), (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) → |f a x| ≤ 1) ∧ a = 8 := by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1740_174002


namespace NUMINAMATH_GPT_parabola_intersects_once_compare_y_values_l1740_174062

noncomputable def parabola (x : ℝ) (m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_intersects_once (m : ℝ) : 
  ∃ x, parabola x m = 0 ↔ m = -2 := 
by 
  sorry

theorem compare_y_values (x1 x2 m : ℝ) (h1 : x1 > x2) (h2 : x2 > 2) : 
  parabola x1 m < parabola x2 m :=
by 
  sorry

end NUMINAMATH_GPT_parabola_intersects_once_compare_y_values_l1740_174062


namespace NUMINAMATH_GPT_daniel_original_noodles_l1740_174027

-- Define the total number of noodles Daniel had originally
def original_noodles : ℕ := 81

-- Define the remaining noodles after giving 1/3 to William
def remaining_noodles (n : ℕ) : ℕ := (2 * n) / 3

-- State the theorem
theorem daniel_original_noodles (n : ℕ) (h : remaining_noodles n = 54) : n = original_noodles := by sorry

end NUMINAMATH_GPT_daniel_original_noodles_l1740_174027


namespace NUMINAMATH_GPT_ratio_of_socks_l1740_174039

variable (b : ℕ)            -- the number of pairs of blue socks
variable (x : ℝ)            -- the price of blue socks per pair

def original_cost : ℝ := 5 * 3 * x + b * x
def interchanged_cost : ℝ := b * 3 * x + 5 * x

theorem ratio_of_socks :
  (5 : ℝ) / b = 5 / 14 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_socks_l1740_174039


namespace NUMINAMATH_GPT_exists_colored_triangle_l1740_174050

structure Point := (x : ℝ) (y : ℝ)
inductive Color
| red
| blue

def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)
  
def same_color_triangle_exists (S : Finset Point) (color : Point → Color) : Prop :=
  ∃ (A B C : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧
                    (color A = color B ∧ color B = color C) ∧
                    ¬ collinear A B C ∧
                    (∃ (X Y Z : Point), 
                      ((X ∈ S ∧ color X ≠ color A ∧ (X ≠ A ∧ X ≠ B ∧ X ≠ C)) ∧ 
                       (Y ∈ S ∧ color Y ≠ color A ∧ (Y ≠ A ∧ Y ≠ B ∧ Y ≠ C)) ∧
                       (Z ∈ S ∧ color Z ≠ color A ∧ (Z ≠ A ∧ Z ≠ B ∧ Z ≠ C)) → 
                       False))

theorem exists_colored_triangle 
  (S : Finset Point) (h1 : 5 ≤ S.card) (color : Point → Color) 
  (h2 : ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → (color A = color B ∧ color B = color C) → ¬ collinear A B C) 
  : same_color_triangle_exists S color :=
sorry

end NUMINAMATH_GPT_exists_colored_triangle_l1740_174050


namespace NUMINAMATH_GPT_max_min_y_l1740_174018

def g (t : ℝ) : ℝ := 80 - 2 * t

def f (t : ℝ) : ℝ := 20 - |t - 10|

def y (t : ℝ) : ℝ := g t * f t

theorem max_min_y (t : ℝ) (h : 0 ≤ t ∧ t ≤ 20) :
  (y t = 1200 → t = 10) ∧ (y t = 400 → t = 20) :=
by
  sorry

end NUMINAMATH_GPT_max_min_y_l1740_174018


namespace NUMINAMATH_GPT_largest_n_l1740_174022

theorem largest_n {x y z n : ℕ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (n:ℤ)^2 = (x:ℤ)^2 + (y:ℤ)^2 + (z:ℤ)^2 + 2*(x:ℤ)*(y:ℤ) + 2*(y:ℤ)*(z:ℤ) + 2*(z:ℤ)*(x:ℤ) + 6*(x:ℤ) + 6*(y:ℤ) + 6*(z:ℤ) - 12
  → n = 13 :=
sorry

end NUMINAMATH_GPT_largest_n_l1740_174022


namespace NUMINAMATH_GPT_count_triangles_in_3x3_grid_l1740_174053

/--
In a 3x3 grid of dots, the number of triangles formed by connecting the dots is 20.
-/
def triangles_in_3x3_grid : Prop :=
  let num_rows := 3
  let num_cols := 3
  let total_triangles := 20
  ∃ (n : ℕ), n = total_triangles ∧ n = 20

theorem count_triangles_in_3x3_grid : triangles_in_3x3_grid :=
by {
  -- Insert the proof here
  sorry
}

end NUMINAMATH_GPT_count_triangles_in_3x3_grid_l1740_174053


namespace NUMINAMATH_GPT_jake_has_3_peaches_l1740_174034

-- Define the number of peaches Steven has.
def steven_peaches : ℕ := 13

-- Define the number of peaches Jake has based on the condition.
def jake_peaches (P_S : ℕ) : ℕ := P_S - 10

-- The theorem that states Jake has 3 peaches.
theorem jake_has_3_peaches : jake_peaches steven_peaches = 3 := sorry

end NUMINAMATH_GPT_jake_has_3_peaches_l1740_174034


namespace NUMINAMATH_GPT_range_of_m_l1740_174090

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) (hineq : 4 / (x + 1) + 1 / y < m^2 + (3 / 2) * m) :
  m < -3 ∨ m > 3 / 2 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1740_174090


namespace NUMINAMATH_GPT_percentage_of_female_officers_on_duty_l1740_174009

theorem percentage_of_female_officers_on_duty :
  ∀ (total_on_duty : ℕ) (half_on_duty : ℕ) (total_female_officers : ℕ), 
  total_on_duty = 204 → half_on_duty = total_on_duty / 2 → total_female_officers = 600 → 
  ((half_on_duty: ℚ) / total_female_officers) * 100 = 17 :=
by
  intro total_on_duty half_on_duty total_female_officers
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_of_female_officers_on_duty_l1740_174009


namespace NUMINAMATH_GPT_rectangular_field_area_l1740_174057

theorem rectangular_field_area (L B : ℝ) (h1 : B = 0.6 * L) (h2 : 2 * L + 2 * B = 800) : L * B = 37500 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l1740_174057


namespace NUMINAMATH_GPT_solve_x_squared_plus_15_eq_y_squared_l1740_174037

theorem solve_x_squared_plus_15_eq_y_squared (x y : ℤ) : x^2 + 15 = y^2 → x = 7 ∨ x = -7 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_GPT_solve_x_squared_plus_15_eq_y_squared_l1740_174037


namespace NUMINAMATH_GPT_exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l1740_174083

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end NUMINAMATH_GPT_exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l1740_174083


namespace NUMINAMATH_GPT_spherical_coordinate_conversion_l1740_174013

theorem spherical_coordinate_conversion (ρ θ φ : ℝ) 
  (h_ρ : ρ > 0) 
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h_φ : 0 ≤ φ): 
  (ρ, θ, φ - 2 * Real.pi * ⌊φ / (2 * Real.pi)⌋) = (5, 3 * Real.pi / 4, Real.pi / 4) :=
  by 
  sorry

end NUMINAMATH_GPT_spherical_coordinate_conversion_l1740_174013


namespace NUMINAMATH_GPT_find_integers_l1740_174005

theorem find_integers 
  (A k : ℕ) 
  (h_sum : A + A * k + A * k^2 = 93) 
  (h_product : A * (A * k) * (A * k^2) = 3375) : 
  (A, A * k, A * k^2) = (3, 15, 75) := 
by 
  sorry

end NUMINAMATH_GPT_find_integers_l1740_174005


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1740_174011

theorem arithmetic_sequence_common_difference {a : ℕ → ℝ} (h₁ : a 1 = 2) (h₂ : a 2 + a 4 = a 6) : ∃ d : ℝ, d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1740_174011


namespace NUMINAMATH_GPT_julian_needs_more_legos_l1740_174077

-- Definitions based on the conditions
def legos_julian_has := 400
def legos_per_airplane := 240
def number_of_airplanes := 2

-- Calculate the total number of legos required for two airplane models
def total_legos_needed := legos_per_airplane * number_of_airplanes

-- Calculate the number of additional legos Julian needs
def additional_legos_needed := total_legos_needed - legos_julian_has

-- Statement that needs to be proven
theorem julian_needs_more_legos : additional_legos_needed = 80 := by
  sorry

end NUMINAMATH_GPT_julian_needs_more_legos_l1740_174077


namespace NUMINAMATH_GPT_vegetables_in_one_serving_l1740_174094

theorem vegetables_in_one_serving
  (V : ℝ)
  (H1 : ∀ servings : ℝ, servings > 0 → servings * (V + 2.5) = 28)
  (H_pints_to_cups : 14 * 2 = 28) :
  V = 1 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_vegetables_in_one_serving_l1740_174094


namespace NUMINAMATH_GPT_smallest_palindrome_not_five_digit_l1740_174069

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

theorem smallest_palindrome_not_five_digit (n : ℕ) :
  (∃ n, is_palindrome n ∧ 100 ≤ n ∧ n < 1000 ∧ ¬is_palindrome (102 * n)) → n = 101 := by
  sorry

end NUMINAMATH_GPT_smallest_palindrome_not_five_digit_l1740_174069


namespace NUMINAMATH_GPT_positive_integer_solutions_of_inequality_l1740_174071

theorem positive_integer_solutions_of_inequality : 
  {x : ℕ | 3 * x - 1 ≤ 2 * x + 3} = {1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_of_inequality_l1740_174071


namespace NUMINAMATH_GPT_triangle_inequality_harmonic_mean_l1740_174045

theorem triangle_inequality_harmonic_mean (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ DP DQ : ℝ, DP + DQ ≤ (2 * a * b) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_harmonic_mean_l1740_174045


namespace NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_l1740_174000

theorem value_of_a_squared_plus_b_squared (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) :
  a^2 + b^2 = 68 :=
sorry

end NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_l1740_174000


namespace NUMINAMATH_GPT_calc_square_difference_and_square_l1740_174075

theorem calc_square_difference_and_square (a b : ℤ) (h1 : a = 7) (h2 : b = 3)
  (h3 : a^2 = 49) (h4 : b^2 = 9) : (a^2 - b^2)^2 = 1600 := by
  sorry

end NUMINAMATH_GPT_calc_square_difference_and_square_l1740_174075


namespace NUMINAMATH_GPT_smallest_fraction_numerator_l1740_174017

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end NUMINAMATH_GPT_smallest_fraction_numerator_l1740_174017


namespace NUMINAMATH_GPT_total_number_of_boys_l1740_174012

-- Define the circular arrangement and the opposite positions
variable (n : ℕ)

theorem total_number_of_boys (h : (40 ≠ 10 ∧ (40 - 10) * 2 = n - 2)) : n = 62 := 
sorry

end NUMINAMATH_GPT_total_number_of_boys_l1740_174012


namespace NUMINAMATH_GPT_length_of_segments_equal_d_l1740_174014

noncomputable def d_eq (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) : ℝ :=
  if h_eq : AB = 550 ∧ BC = 580 ∧ AC = 620 then 342 else 0

theorem length_of_segments_equal_d (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) :
  d_eq AB BC AC h = 342 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segments_equal_d_l1740_174014


namespace NUMINAMATH_GPT_banknotes_combination_l1740_174031

theorem banknotes_combination (a b c d : ℕ) (h : a + b + c + d = 10) (h_val : 2000 * a + 1000 * b + 500 * c + 200 * d = 5000) :
  (a = 0 ∧ b = 0 ∧ c = 10 ∧ d = 0) ∨ 
  (a = 1 ∧ b = 0 ∧ c = 4 ∧ d = 5) ∨ 
  (a = 0 ∧ b = 3 ∧ c = 2 ∧ d = 5) :=
by
  sorry

end NUMINAMATH_GPT_banknotes_combination_l1740_174031
