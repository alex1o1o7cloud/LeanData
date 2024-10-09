import Mathlib

namespace find_integer_n_l2290_229033

theorem find_integer_n (a b : ℕ) (n : ℕ)
  (h1 : n = 2^a * 3^b)
  (h2 : (2^(a+1) - 1) * ((3^(b+1) - 1) / (3 - 1)) = 1815) : n = 648 :=
  sorry

end find_integer_n_l2290_229033


namespace base_of_number_l2290_229044

theorem base_of_number (b : ℕ) : 
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 :=
by
  sorry

end base_of_number_l2290_229044


namespace equation_solution_l2290_229089

variable (x y : ℝ)

theorem equation_solution
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66):
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 :=
  by sorry

end equation_solution_l2290_229089


namespace prime_triplets_satisfy_condition_l2290_229095

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_triplets_satisfy_condition :
  ∀ p q r : ℕ,
    is_prime p → is_prime q → is_prime r →
    (p * (r - 1) = q * (r + 7)) →
    (p = 3 ∧ q = 2 ∧ r = 17) ∨ 
    (p = 7 ∧ q = 3 ∧ r = 7) ∨
    (p = 5 ∧ q = 3 ∧ r = 13) :=
by
  sorry

end prime_triplets_satisfy_condition_l2290_229095


namespace number_of_men_in_first_group_l2290_229007

theorem number_of_men_in_first_group 
    (x : ℕ) (H1 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = 1 / (5 * x))
    (H2 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate 15 12 = 1 / (15 * 12))
    (H3 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = work_rate 15 12) 
    : x = 36 := 
by {
    sorry
}

end number_of_men_in_first_group_l2290_229007


namespace score_not_possible_l2290_229034

theorem score_not_possible (c u i : ℕ) (score : ℤ) :
  c + u + i = 25 ∧ score = 79 → score ≠ 5 * c + 3 * u - 25 := by
  intro h
  sorry

end score_not_possible_l2290_229034


namespace difference_is_1343_l2290_229064

-- Define the larger number L and the relationship with the smaller number S.
def L : ℕ := 1608
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Define the relationship: L = 6S + 15
def relationship (S : ℕ) : Prop := L = quotient * S + remainder

-- The theorem we want to prove: The difference between the larger and smaller number is 1343
theorem difference_is_1343 (S : ℕ) (h_rel : relationship S) : L - S = 1343 :=
by
  sorry

end difference_is_1343_l2290_229064


namespace spaghetti_manicotti_ratio_l2290_229013

-- Define the number of students who were surveyed and their preferences
def total_students := 800
def students_prefer_spaghetti := 320
def students_prefer_manicotti := 160

-- The ratio of students who prefer spaghetti to those who prefer manicotti is 2
theorem spaghetti_manicotti_ratio :
  students_prefer_spaghetti / students_prefer_manicotti = 2 :=
by
  sorry

end spaghetti_manicotti_ratio_l2290_229013


namespace race_length_l2290_229039

variables (L : ℕ)

def distanceCondition1 := L - 70
def distanceCondition2 := L - 100
def distanceCondition3 := L - 163

theorem race_length (h1 : distanceCondition1 = L - 70) 
                    (h2 : distanceCondition2 = L - 100) 
                    (h3 : distanceCondition3 = L - 163)
                    (h4 : (L - 70) / (L - 163) = (L) / (L - 100)) : 
  L = 1000 :=
sorry

end race_length_l2290_229039


namespace benny_spending_l2290_229027

variable (S D V : ℝ)

theorem benny_spending :
  (200 - 45) = S + (D / 110) + (V / 0.75) :=
by
  sorry

end benny_spending_l2290_229027


namespace probability_range_l2290_229041

noncomputable def probability_distribution (K : ℕ) : ℝ :=
  if K > 0 then 1 / (2^K) else 0

theorem probability_range (h2 : 2 < 3) (h3 : 3 ≤ 4) :
  probability_distribution 3 + probability_distribution 4 = 3 / 16 :=
by
  sorry

end probability_range_l2290_229041


namespace Li_age_is_12_l2290_229056

-- Given conditions:
def Zhang_twice_Li (Li: ℕ) : ℕ := 2 * Li
def Jung_older_Zhang (Zhang: ℕ) : ℕ := Zhang + 2
def Jung_age := 26

-- Proof problem:
theorem Li_age_is_12 : ∃ Li: ℕ, Jung_older_Zhang (Zhang_twice_Li Li) = Jung_age ∧ Li = 12 :=
by
  sorry

end Li_age_is_12_l2290_229056


namespace molecular_weight_N2O_correct_l2290_229004

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of N2O
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

-- Prove the statement
theorem molecular_weight_N2O_correct : molecular_weight_N2O = 44.02 := by
  -- We leave the proof as an exercise (or assumption)
  sorry

end molecular_weight_N2O_correct_l2290_229004


namespace tangent_line_through_P_is_correct_l2290_229067

-- Define the circle and the point
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 25
def pointP : ℝ × ℝ := (-1, 7)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 31 = 0

-- State the theorem
theorem tangent_line_through_P_is_correct :
  (circle_eq (-1) 7) → 
  (tangent_line (-1) 7) :=
sorry

end tangent_line_through_P_is_correct_l2290_229067


namespace tax_increase_proof_l2290_229002

variables (old_tax_rate new_tax_rate : ℝ) (old_income new_income : ℝ)

def old_taxes_paid (old_tax_rate old_income : ℝ) : ℝ := old_tax_rate * old_income

def new_taxes_paid (new_tax_rate new_income : ℝ) : ℝ := new_tax_rate * new_income

def increase_in_taxes (old_tax_rate new_tax_rate old_income new_income : ℝ) : ℝ :=
  new_taxes_paid new_tax_rate new_income - old_taxes_paid old_tax_rate old_income

theorem tax_increase_proof :
  increase_in_taxes 0.20 0.30 1000000 1500000 = 250000 := by
  sorry

end tax_increase_proof_l2290_229002


namespace sum_is_272_l2290_229037

-- Define the constant number x
def x : ℕ := 16

-- Define the sum of the number and its square
def sum_of_number_and_its_square (n : ℕ) : ℕ := n + n^2

-- State the theorem that the sum of the number and its square is 272 when the number is 16
theorem sum_is_272 : sum_of_number_and_its_square x = 272 :=
by
  sorry

end sum_is_272_l2290_229037


namespace prove_a_minus_c_l2290_229074

-- Define the given conditions as hypotheses
def condition1 (a b d : ℝ) : Prop := (a + d + b + d) / 2 = 80
def condition2 (b c d : ℝ) : Prop := (b + d + c + d) / 2 = 180

-- State the theorem to be proven
theorem prove_a_minus_c (a b c d : ℝ) (h1 : condition1 a b d) (h2 : condition2 b c d) : a - c = -200 :=
by
  sorry

end prove_a_minus_c_l2290_229074


namespace missing_number_evaluation_l2290_229005

theorem missing_number_evaluation (x : ℝ) (h : |4 + 9 * x| - 6 = 70) : x = 8 :=
sorry

end missing_number_evaluation_l2290_229005


namespace max_b_of_box_volume_l2290_229047

theorem max_b_of_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : Prime c) (h5 : a * b * c = 360) : b = 12 := 
sorry

end max_b_of_box_volume_l2290_229047


namespace cost_per_treat_l2290_229038

def treats_per_day : ℕ := 2
def days_in_month : ℕ := 30
def total_spent : ℝ := 6.0

theorem cost_per_treat : (total_spent / (treats_per_day * days_in_month : ℕ)) = 0.10 :=
by 
  sorry

end cost_per_treat_l2290_229038


namespace river_width_after_30_seconds_l2290_229062

noncomputable def width_of_river (initial_width : ℝ) (width_increase_rate : ℝ) (rowing_rate : ℝ) (time_taken : ℝ) : ℝ :=
  initial_width + (time_taken * rowing_rate * (width_increase_rate / 10))

theorem river_width_after_30_seconds :
  width_of_river 50 2 5 30 = 80 :=
by
  -- it suffices to check the calculations here
  sorry

end river_width_after_30_seconds_l2290_229062


namespace number_of_boats_l2290_229054

theorem number_of_boats (total_people : ℕ) (people_per_boat : ℕ)
  (h1 : total_people = 15) (h2 : people_per_boat = 3) : total_people / people_per_boat = 5 :=
by {
  -- proof steps here
  sorry
}

end number_of_boats_l2290_229054


namespace angelfish_goldfish_difference_l2290_229023

-- Given statements
variables {A G : ℕ}
def goldfish := 8
def total_fish := 44

-- Conditions
axiom twice_as_many_guppies : G = 2 * A
axiom total_fish_condition : A + G + goldfish = total_fish

-- Theorem
theorem angelfish_goldfish_difference : A - goldfish = 4 :=
by
  sorry

end angelfish_goldfish_difference_l2290_229023


namespace exists_factor_between_10_and_20_l2290_229022

theorem exists_factor_between_10_and_20 (n : ℕ) : ∃ k, (10 ≤ k ∧ k ≤ 20) ∧ k ∣ (2^n - 1) → k = 17 :=
by
  sorry

end exists_factor_between_10_and_20_l2290_229022


namespace sum_of_a_and_b_l2290_229092

-- Define conditions
def population_size : ℕ := 55
def sample_size : ℕ := 5
def interval : ℕ := population_size / sample_size
def sample_indices : List ℕ := [6, 28, 50]

-- Assume a and b are such that the systematic sampling is maintained
variable (a b : ℕ)
axiom a_idx : a = sample_indices.head! + interval
axiom b_idx : b = sample_indices.getLast! - interval

-- Define Lean 4 statement to prove
theorem sum_of_a_and_b :
  (a + b) = 56 :=
by
  -- This will be the place where the proof is inserted
  sorry

end sum_of_a_and_b_l2290_229092


namespace necessary_but_not_sufficient_l2290_229094

-- Define conditions P and Q
def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

-- Statement to prove
theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x ∧ ¬ (Q x → P x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l2290_229094


namespace mother_daughter_age_relation_l2290_229055

theorem mother_daughter_age_relation (x : ℕ) (hc1 : 43 - x = 5 * (11 - x)) : x = 3 := 
sorry

end mother_daughter_age_relation_l2290_229055


namespace privateer_overtakes_at_6_08_pm_l2290_229099

noncomputable def time_of_overtake : Bool :=
  let initial_distance := 12 -- miles
  let initial_time := 10 -- 10:00 a.m.
  let privateer_speed_initial := 10 -- mph
  let merchantman_speed := 7 -- mph
  let time_to_sail_initial := 3 -- hours
  let distance_covered_privateer := privateer_speed_initial * time_to_sail_initial
  let distance_covered_merchantman := merchantman_speed * time_to_sail_initial
  let relative_distance_after_three_hours := initial_distance + distance_covered_merchantman - distance_covered_privateer
  let privateer_speed_modified := 13 -- new speed
  let merchantman_speed_modified := 12 -- corresponding merchantman speed

  -- Calculating the new relative speed after the privateer's speed is reduced
  let privateer_new_speed := (13 / 12) * merchantman_speed
  let relative_speed_after_damage := privateer_new_speed - merchantman_speed
  let time_to_overtake_remainder := relative_distance_after_three_hours / relative_speed_after_damage
  let total_time := time_to_sail_initial + time_to_overtake_remainder -- in hours

  let final_time := initial_time + total_time -- converting into the final time of the day
  final_time == 18.1333 -- This should convert to 6:08 p.m., approximately 18 hours and 8 minutes in a 24-hour format

theorem privateer_overtakes_at_6_08_pm : time_of_overtake = true :=
  by
    -- Proof will be provided here
    sorry

end privateer_overtakes_at_6_08_pm_l2290_229099


namespace sum_of_areas_of_circles_l2290_229024

-- Definitions of the conditions given in the problem
def triangle_side1 : ℝ := 6
def triangle_side2 : ℝ := 8
def triangle_side3 : ℝ := 10

-- Definitions of the radii r, s, t
variables (r s t : ℝ)

-- Conditions derived from the problem
axiom rs_eq : r + s = triangle_side1
axiom rt_eq : r + t = triangle_side2
axiom st_eq : s + t = triangle_side3

-- Main theorem to prove
theorem sum_of_areas_of_circles : (π * r^2) + (π * s^2) + (π * t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_circles_l2290_229024


namespace aero_flight_tees_per_package_l2290_229060

theorem aero_flight_tees_per_package {A : ℕ} :
  (∀ (num_people : ℕ), num_people = 4 → 20 * num_people ≤ A * 28 + 2 * 12) →
  A * 28 ≥ 56 →
  A = 2 :=
by
  intros h1 h2
  sorry

end aero_flight_tees_per_package_l2290_229060


namespace vector_addition_scalar_multiplication_l2290_229045

def u : ℝ × ℝ × ℝ := (3, -2, 5)
def v : ℝ × ℝ × ℝ := (-1, 6, -3)
def result : ℝ × ℝ × ℝ := (4, 8, 4)

theorem vector_addition_scalar_multiplication :
  2 • (u + v) = result :=
by
  sorry

end vector_addition_scalar_multiplication_l2290_229045


namespace sequence_recurrence_l2290_229021

noncomputable def a (n : ℕ) : ℤ := Int.floor ((1 + Real.sqrt 2) ^ n)

theorem sequence_recurrence (k : ℕ) (h : 2 ≤ k) : 
  ∀ n : ℕ, 
  (a 2 * k = 2 * a (2 * k - 1) + a (2 * k - 2)) ∧
  (a (2 * k + 1) = 2 * a (2 * k) + a (2 * k - 1) + 2) :=
sorry

end sequence_recurrence_l2290_229021


namespace parallel_lines_a_l2290_229026
-- Import necessary libraries

-- Define the given conditions and the main statement
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 2 = 0 → 3 * x + (a + 2) * y + 1 = 0) →
  (a = -3 ∨ a = 1) :=
by
  -- Place the proof here
  sorry

end parallel_lines_a_l2290_229026


namespace ninety_one_square_friendly_unique_square_friendly_l2290_229082

-- Given conditions
def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18 * m + c = n^2

-- Part (a)
theorem ninety_one_square_friendly : square_friendly 81 :=
sorry

-- Part (b)
theorem unique_square_friendly (c c' : ℤ) (h_c : square_friendly c) (h_c' : square_friendly c') : c = c' :=
sorry

end ninety_one_square_friendly_unique_square_friendly_l2290_229082


namespace problem_statement_l2290_229070

-- Define the sides of the original triangle
def side_5 := 5
def side_12 := 12
def side_13 := 13

-- Define the perimeters of the isosceles triangles
def P := 3 * side_5
def Q := 3 * side_12
def R := 3 * side_13

-- Statement we want to prove
theorem problem_statement : P + R = (3 / 2) * Q := by
  sorry

end problem_statement_l2290_229070


namespace find_v_plus_z_l2290_229010

variable (x u v w z : ℂ)
variable (y : ℂ)
variable (condition1 : y = 2)
variable (condition2 : w = -x - u)
variable (condition3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I)

theorem find_v_plus_z : v + z = -4 :=
by
  have h1 : y = 2 := condition1
  have h2 : w = -x - u := condition2
  have h3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I := condition3
  sorry

end find_v_plus_z_l2290_229010


namespace find_common_difference_l2290_229009

-- Definitions for arithmetic sequences and sums
def S (a1 d : ℕ) (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2
def a (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

theorem find_common_difference (a1 d : ℕ) :
  S a1 d 3 = 6 → a a1 d 3 = 4 → d = 2 :=
by
  intros S3_eq a3_eq
  sorry

end find_common_difference_l2290_229009


namespace find_k_l2290_229091

-- Define the number and compute the sum of its digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem find_k :
  ∃ k : ℕ, sum_of_digits (9 * (10^k - 1)) = 1111 ∧ k = 124 :=
sorry

end find_k_l2290_229091


namespace inner_cube_surface_area_l2290_229031

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l2290_229031


namespace cricket_problem_l2290_229057

theorem cricket_problem
  (x : ℕ)
  (run_rate_initial : ℝ := 3.8)
  (overs_remaining : ℕ := 40)
  (run_rate_remaining : ℝ := 6.1)
  (target_runs : ℕ := 282) :
  run_rate_initial * x + run_rate_remaining * overs_remaining = target_runs → x = 10 :=
by
  -- proof goes here
  sorry

end cricket_problem_l2290_229057


namespace length_of_smaller_cube_edge_is_5_l2290_229042

-- Given conditions
def stacked_cube_composed_of_smaller_cubes (n: ℕ) (a: ℕ) : Prop := a * a * a = n

def volume_of_larger_cube (l: ℝ) (v: ℝ) : Prop := l ^ 3 = v

-- Problem statement: Prove that the length of one edge of the smaller cube is 5 cm
theorem length_of_smaller_cube_edge_is_5 :
  ∃ s: ℝ, stacked_cube_composed_of_smaller_cubes 8 2 ∧ volume_of_larger_cube (2*s) 1000 ∧ s = 5 :=
  sorry

end length_of_smaller_cube_edge_is_5_l2290_229042


namespace triangular_stack_log_count_l2290_229078

theorem triangular_stack_log_count : 
  ∀ (a₁ aₙ d : ℤ) (n : ℤ), a₁ = 15 → aₙ = 1 → d = -2 → 
  (a₁ - aₙ) / (-d) + 1 = n → 
  (n * (a₁ + aₙ)) / 2 = 64 :=
by
  intros a₁ aₙ d n h₁ hₙ hd hn
  sorry

end triangular_stack_log_count_l2290_229078


namespace nonnegative_exists_l2290_229065

theorem nonnegative_exists (a b c : ℝ) (h : a + b + c = 0) : a ≥ 0 ∨ b ≥ 0 ∨ c ≥ 0 :=
by
  sorry

end nonnegative_exists_l2290_229065


namespace b_alone_days_l2290_229000

-- Definitions from the conditions
def work_rate_b (W_b : ℝ) : ℝ := W_b
def work_rate_a (W_b : ℝ) : ℝ := 2 * W_b
def work_rate_c (W_b : ℝ) : ℝ := 6 * W_b
def combined_work_rate (W_b : ℝ) : ℝ := work_rate_a W_b + work_rate_b W_b + work_rate_c W_b
def total_days_together : ℝ := 10
def total_work (W_b : ℝ) : ℝ := combined_work_rate W_b * total_days_together

-- The proof problem
theorem b_alone_days (W_b : ℝ) : 90 = total_work W_b / work_rate_b W_b :=
by
  sorry

end b_alone_days_l2290_229000


namespace intersection_A_B_l2290_229096

-- Definition of sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y > 0 }

-- The proof goal
theorem intersection_A_B : A ∩ B = { x | x > 1 } :=
by sorry

end intersection_A_B_l2290_229096


namespace min_b_minus_2c_over_a_l2290_229019

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (h1 : a ≤ b + c ∧ b + c ≤ 3 * a)
variable (h2 : 3 * b^2 ≤ a * (a + c) ∧ a * (a + c) ≤ 5 * b^2)

theorem min_b_minus_2c_over_a : (∃ u : ℝ, (u = (b - 2 * c) / a) ∧ (∀ v : ℝ, (v = (b - 2 * c) / a) → u ≤ v)) :=
  sorry

end min_b_minus_2c_over_a_l2290_229019


namespace gasoline_price_increase_l2290_229029

theorem gasoline_price_increase :
  let P_initial := 29.90
  let P_final := 149.70
  (P_final - P_initial) / P_initial * 100 = 400 :=
by
  let P_initial := 29.90
  let P_final := 149.70
  sorry

end gasoline_price_increase_l2290_229029


namespace work_completion_days_l2290_229020

theorem work_completion_days (P R: ℕ) (hP: P = 80) (hR: R = 120) : P * R / (P + R) = 48 := by
  -- The proof is omitted as we are only writing the statement
  sorry

end work_completion_days_l2290_229020


namespace find_b_value_l2290_229040

theorem find_b_value :
  ∃ b : ℕ, 70 = (2 * (b + 1)^2 + 3 * (b + 1) + 4) - (2 * (b - 1)^2 + 3 * (b - 1) + 4) ∧ b > 0 ∧ b < 1000 :=
by
  sorry

end find_b_value_l2290_229040


namespace smallest_palindrome_base2_base4_l2290_229059

def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n)
  digits = digits.reverse

theorem smallest_palindrome_base2_base4 : 
  ∃ (x : ℕ), x > 15 ∧ is_palindrome_base x 2 ∧ is_palindrome_base x 4 ∧ x = 17 :=
by
  sorry

end smallest_palindrome_base2_base4_l2290_229059


namespace difference_seven_three_times_l2290_229069

theorem difference_seven_three_times (n : ℝ) (h1 : n = 3) 
  (h2 : 7 * n = 3 * n + (21.0 - 9.0)) :
  7 * n - 3 * n = 12.0 := by
  sorry

end difference_seven_three_times_l2290_229069


namespace problem_xy_squared_and_product_l2290_229087

theorem problem_xy_squared_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 :=
by
  sorry

end problem_xy_squared_and_product_l2290_229087


namespace trig_identity_l2290_229011

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 :=
by
  sorry

end trig_identity_l2290_229011


namespace calculate_ratio_l2290_229075

variables (M Q P N R : ℝ)

-- Definitions of conditions
def M_def : M = 0.40 * Q := by sorry
def Q_def : Q = 0.30 * P := by sorry
def N_def : N = 0.60 * P := by sorry
def R_def : R = 0.20 * P := by sorry

-- Statement of the proof problem
theorem calculate_ratio (hM : M = 0.40 * Q) (hQ : Q = 0.30 * P)
  (hN : N = 0.60 * P) (hR : R = 0.20 * P) : 
  (M + R) / N = 8 / 15 := by
  sorry

end calculate_ratio_l2290_229075


namespace find_number_l2290_229083

theorem find_number (x : ℕ) (h : x + 3 * x = 20) : x = 5 :=
by
  sorry

end find_number_l2290_229083


namespace number_is_a_l2290_229081

theorem number_is_a (x y z a : ℝ) (h1 : x + y + z = a) (h2 : (1 / x) + (1 / y) + (1 / z) = 1 / a) : 
  x = a ∨ y = a ∨ z = a :=
sorry

end number_is_a_l2290_229081


namespace repeating_decimal_as_fraction_l2290_229058

theorem repeating_decimal_as_fraction :
  (0.58207 : ℝ) = 523864865 / 999900 := sorry

end repeating_decimal_as_fraction_l2290_229058


namespace train_length_l2290_229053

theorem train_length (speed_kmph : ℤ) (time_sec : ℤ) (expected_length_m : ℤ) 
    (speed_kmph_eq : speed_kmph = 72)
    (time_sec_eq : time_sec = 7)
    (expected_length_eq : expected_length_m = 140) :
    expected_length_m = (speed_kmph * 1000 / 3600) * time_sec :=
by 
    sorry

end train_length_l2290_229053


namespace geoff_additional_votes_needed_l2290_229028

-- Define the given conditions
def totalVotes : ℕ := 6000
def geoffPercentage : ℕ := 5 -- Represent 0.5% as 5 out of 1000 for better integer computation
def requiredPercentage : ℕ := 505 -- Represent 50.5% as 505 out of 1000 for better integer computation

-- Define the expressions for the number of votes received by Geoff and the votes required to win
def geoffVotes := (geoffPercentage * totalVotes) / 1000
def requiredVotes := (requiredPercentage * totalVotes) / 1000 + 1

-- The proposition to prove the additional number of votes needed for Geoff to win
theorem geoff_additional_votes_needed : requiredVotes - geoffVotes = 3001 := by sorry

end geoff_additional_votes_needed_l2290_229028


namespace inequality_solution_intervals_l2290_229097

theorem inequality_solution_intervals (x : ℝ) (h : x > 2) : 
  (x-2)^(x^2 - 6 * x + 8) > 1 ↔ (2 < x ∧ x < 3) ∨ x > 4 := 
sorry

end inequality_solution_intervals_l2290_229097


namespace most_probable_light_is_green_l2290_229072

def duration_red := 30
def duration_yellow := 5
def duration_green := 40
def total_duration := duration_red + duration_yellow + duration_green

def prob_red := duration_red / total_duration
def prob_yellow := duration_yellow / total_duration
def prob_green := duration_green / total_duration

theorem most_probable_light_is_green : prob_green > prob_red ∧ prob_green > prob_yellow := 
  by
  sorry

end most_probable_light_is_green_l2290_229072


namespace temperature_on_Friday_l2290_229001

variable (M T W Th F : ℝ)

def avg_M_T_W_Th := (M + T + W + Th) / 4 = 48
def avg_T_W_Th_F := (T + W + Th + F) / 4 = 46
def temp_Monday := M = 42

theorem temperature_on_Friday
  (h1 : avg_M_T_W_Th M T W Th)
  (h2 : avg_T_W_Th_F T W Th F) 
  (h3 : temp_Monday M) : F = 34 := by
  sorry

end temperature_on_Friday_l2290_229001


namespace find_a5_l2290_229048

theorem find_a5 (a : ℕ → ℤ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1) 
  (h2 : a 2 + a 4 + a 6 = 18) : 
  a 5 = 5 :=
sorry

end find_a5_l2290_229048


namespace grocer_initial_stock_l2290_229090

theorem grocer_initial_stock 
  (x : ℝ) 
  (h1 : 0.20 * x + 70 = 0.30 * (x + 100)) : 
  x = 400 := by
  sorry

end grocer_initial_stock_l2290_229090


namespace zhou_catches_shuttle_probability_l2290_229030

-- Condition 1: Shuttle arrival time and duration
def shuttle_arrival_start : ℕ := 420 -- 7:00 AM in minutes since midnight
def shuttle_duration : ℕ := 15

-- Condition 2: Zhou's random arrival time window
def zhou_arrival_start : ℕ := 410 -- 6:50 AM in minutes since midnight
def zhou_arrival_end : ℕ := 465 -- 7:45 AM in minutes since midnight

-- Total time available for Zhou to arrive (55 minutes) 
def total_time : ℕ := zhou_arrival_end - zhou_arrival_start

-- Time window when Zhou needs to arrive to catch the shuttle (15 minutes)
def successful_time : ℕ := shuttle_arrival_start + shuttle_duration - shuttle_arrival_start

-- Calculate the probability that Zhou catches the shuttle
theorem zhou_catches_shuttle_probability : 
  (successful_time : ℚ) / total_time = 3 / 11 := 
by 
  -- We don't need the actual proof steps, just the statement
  sorry

end zhou_catches_shuttle_probability_l2290_229030


namespace Zhang_Laoshi_pens_l2290_229068

theorem Zhang_Laoshi_pens (x : ℕ) (original_price new_price : ℝ)
  (discount : new_price = 0.75 * original_price)
  (more_pens : x * original_price = (x + 25) * new_price) :
  x = 75 :=
by
  sorry

end Zhang_Laoshi_pens_l2290_229068


namespace no_solution_m_l2290_229084

noncomputable def fractional_eq (x m : ℝ) : Prop :=
  2 / (x - 2) + m * x / (x^2 - 4) = 3 / (x + 2)

theorem no_solution_m (m : ℝ) : 
  (¬ ∃ x, fractional_eq x m) ↔ (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end no_solution_m_l2290_229084


namespace solve_for_x_l2290_229014

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end solve_for_x_l2290_229014


namespace average_marks_of_failed_boys_l2290_229018

def total_boys : ℕ := 120
def average_marks_all_boys : ℝ := 35
def number_of_passed_boys : ℕ := 100
def average_marks_passed_boys : ℝ := 39
def number_of_failed_boys : ℕ := total_boys - number_of_passed_boys

noncomputable def total_marks_all_boys : ℝ := average_marks_all_boys * total_boys
noncomputable def total_marks_passed_boys : ℝ := average_marks_passed_boys * number_of_passed_boys
noncomputable def total_marks_failed_boys : ℝ := total_marks_all_boys - total_marks_passed_boys
noncomputable def average_marks_failed_boys : ℝ := total_marks_failed_boys / number_of_failed_boys

theorem average_marks_of_failed_boys :
  average_marks_failed_boys = 15 :=
by
  -- The proof can be filled in here
  sorry

end average_marks_of_failed_boys_l2290_229018


namespace total_cost_price_l2290_229025

theorem total_cost_price (P_ct P_ch P_bs : ℝ) (h1 : 8091 = P_ct * 1.24)
    (h2 : 5346 = P_ch * 1.18 * 0.95) (h3 : 11700 = P_bs * 1.30) : 
    P_ct + P_ch + P_bs = 20295 := 
by 
    sorry

end total_cost_price_l2290_229025


namespace combined_average_age_l2290_229035

noncomputable def roomA : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
noncomputable def roomB : Set ℕ := {11, 12, 13, 14}
noncomputable def average_age_A := 55
noncomputable def average_age_B := 35
noncomputable def total_people := (10 + 4)
noncomputable def total_age_A := 10 * average_age_A
noncomputable def total_age_B := 4 * average_age_B
noncomputable def combined_total_age := total_age_A + total_age_B

theorem combined_average_age :
  (combined_total_age / total_people : ℚ) = 49.29 :=
by sorry

end combined_average_age_l2290_229035


namespace difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l2290_229051

-- Exploration 1
theorem difference_in_square_sides (a b : ℝ) (h1 : a + b = 20) (h2 : a^2 - b^2 = 40) : a - b = 2 :=
by sorry

-- Exploration 2
theorem square_side_length (x y : ℝ) : (2 * x + 2 * y) / 4 = (x + y) / 2 :=
by sorry

theorem square_area_greater_than_rectangle (x y : ℝ) (h : x > y) : ( (x + y) / 2 ) ^ 2 > x * y :=
by sorry

end difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l2290_229051


namespace not_exists_k_eq_one_l2290_229043

theorem not_exists_k_eq_one (k : ℝ) : (∃ x y : ℝ, y = k * x + 2 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by sorry

end not_exists_k_eq_one_l2290_229043


namespace polynomial_roots_l2290_229016

theorem polynomial_roots :
  Polynomial.roots (Polynomial.C 4 * Polynomial.X ^ 5 +
                    Polynomial.C 13 * Polynomial.X ^ 4 +
                    Polynomial.C (-30) * Polynomial.X ^ 3 +
                    Polynomial.C 8 * Polynomial.X ^ 2) =
  {0, 0, 1 / 2, -2 + 2 * Real.sqrt 2, -2 - 2 * Real.sqrt 2} :=
by
  sorry

end polynomial_roots_l2290_229016


namespace horner_example_l2290_229049

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end horner_example_l2290_229049


namespace measure_of_angle_B_in_triangle_l2290_229050

theorem measure_of_angle_B_in_triangle
  {a b c : ℝ} {A B C : ℝ} 
  (h1 : a * c = b^2 - a^2)
  (h2 : A = Real.pi / 6)
  (h3 : a / Real.sin A = b / Real.sin B) 
  (h4 : b / Real.sin B = c / Real.sin C)
  (h5 : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
by sorry

end measure_of_angle_B_in_triangle_l2290_229050


namespace total_payment_is_correct_l2290_229073

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end total_payment_is_correct_l2290_229073


namespace sum_of_first_seven_terms_l2290_229012

variable (a : ℕ → ℝ) -- a sequence of real numbers (can be adapted to other types if needed)

-- Given conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a n = a 0 + n * d

def sum_of_three_terms (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  a 2 + a 3 + a 4 = sum

-- Theorem to prove
theorem sum_of_first_seven_terms (a : ℕ → ℝ) (h1 : is_arithmetic_progression a) (h2 : sum_of_three_terms a 12) :
  (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) = 28 :=
sorry

end sum_of_first_seven_terms_l2290_229012


namespace shepherd_flock_l2290_229077

theorem shepherd_flock (x y : ℕ) (h1 : (x - 1) * 5 = 7 * y) (h2 : x * 3 = 5 * (y - 1)) :
  x + y = 25 :=
sorry

end shepherd_flock_l2290_229077


namespace determine_alpha_l2290_229015

variables (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m + n = 1)
variables (α : ℝ)

-- Defining the minimum value condition
def minimum_value_condition : Prop :=
  (1 / m + 16 / n) = 25

-- Defining the curve passing through point P
def passes_through_P : Prop :=
  (m / 5) ^ α = (m / 4)

theorem determine_alpha
  (h_min_value : minimum_value_condition m n)
  (h_passes_through : passes_through_P m α) :
  α = 1 / 2 :=
sorry

end determine_alpha_l2290_229015


namespace not_possible_odd_sum_l2290_229080

theorem not_possible_odd_sum (m n : ℤ) (h : (m ^ 2 + n ^ 2) % 2 = 0) : (m + n) % 2 ≠ 1 :=
sorry

end not_possible_odd_sum_l2290_229080


namespace tangent_of_11pi_over_4_l2290_229046

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l2290_229046


namespace inequality_proof_l2290_229061

theorem inequality_proof
  {x1 x2 x3 x4 : ℝ}
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x4 ≥ 2)
  (h5 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 :=
by
  sorry

end inequality_proof_l2290_229061


namespace even_function_coeff_l2290_229017

theorem even_function_coeff (a : ℝ) (h : ∀ x : ℝ, (a-2)*x^2 + (a-1)*x + 3 = (a-2)*(-x)^2 + (a-1)*(-x) + 3) : a = 1 :=
by {
  -- Proof here
  sorry
}

end even_function_coeff_l2290_229017


namespace cos_C_of_triangle_l2290_229093

theorem cos_C_of_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (h_sine_relation : 3 * Real.sin A = 2 * Real.sin B)
  (h_cosine_law : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.cos C = -1/4 :=
by
  sorry

end cos_C_of_triangle_l2290_229093


namespace E_72_eq_9_l2290_229063

def E (n : ℕ) : ℕ :=
  -- Assume a function definition counting representations
  -- (this function body is a placeholder, as the exact implementation
  -- is not part of the problem statement)
  sorry

theorem E_72_eq_9 :
  E 72 = 9 :=
sorry

end E_72_eq_9_l2290_229063


namespace jasonPears_l2290_229086

-- Define the conditions
def keithPears : Nat := 47
def mikePears : Nat := 12
def totalPears : Nat := 105

-- Define the theorem stating the number of pears Jason picked
theorem jasonPears : (totalPears - keithPears - mikePears) = 46 :=
by 
  sorry

end jasonPears_l2290_229086


namespace intersection_M_N_l2290_229006

open Set

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_M_N : M ∩ N = {-1, 1} :=
by
  sorry

end intersection_M_N_l2290_229006


namespace range_of_m_l2290_229036

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 0 < m) 
  (h4 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : m ≥ 4 :=
sorry

end range_of_m_l2290_229036


namespace original_price_of_cycle_l2290_229066

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h_SP : SP = 1080)
  (h_gain_percent: gain_percent = 60)
  (h_relation : SP = 1.6 * P)
  : P = 675 :=
by {
  sorry
}

end original_price_of_cycle_l2290_229066


namespace intersection_point_l2290_229088

theorem intersection_point (x y : ℚ) (h1 : 8 * x - 5 * y = 40) (h2 : 6 * x + 2 * y = 14) :
  x = 75 / 23 ∧ y = -64 / 23 :=
by
  -- Proof not needed, so we finish with sorry
  sorry

end intersection_point_l2290_229088


namespace probability_calculation_l2290_229032

noncomputable def probability_same_color (pairs_black pairs_brown pairs_gray : ℕ) : ℚ :=
  let total_shoes := 2 * (pairs_black + pairs_brown + pairs_gray)
  let prob_black := (2 * pairs_black : ℚ) / total_shoes * (pairs_black : ℚ) / (total_shoes - 1)
  let prob_brown := (2 * pairs_brown : ℚ) / total_shoes * (pairs_brown : ℚ) / (total_shoes - 1)
  let prob_gray := (2 * pairs_gray : ℚ) / total_shoes * (pairs_gray : ℚ) / (total_shoes - 1)
  prob_black + prob_brown + prob_gray

theorem probability_calculation :
  probability_same_color 7 4 3 = 37 / 189 :=
by
  sorry

end probability_calculation_l2290_229032


namespace intersection_A_B_solution_inequalities_l2290_229052

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def C : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = C :=
by
  sorry

theorem solution_inequalities (x : ℝ) :
  (2 * x^2 + x - 1 > 0) ↔ (x < -1 ∨ x > 1/2) :=
by
  sorry

end intersection_A_B_solution_inequalities_l2290_229052


namespace bricks_in_row_l2290_229071

theorem bricks_in_row 
  (total_bricks : ℕ) 
  (rows_per_wall : ℕ) 
  (num_walls : ℕ)
  (total_rows : ℕ)
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) 
  (h4 : total_rows = rows_per_wall * num_walls) :
  total_bricks / total_rows = 30 :=
by
  sorry

end bricks_in_row_l2290_229071


namespace anne_carries_16point5_kg_l2290_229098

theorem anne_carries_16point5_kg :
  let w1 := 2
  let w2 := 1.5 * w1
  let w3 := 2 * w1
  let w4 := w1 + w2
  let w5 := (w1 + w2) / 2
  w1 + w2 + w3 + w4 + w5 = 16.5 :=
by {
  sorry
}

end anne_carries_16point5_kg_l2290_229098


namespace cube_of_odd_sum_l2290_229085

theorem cube_of_odd_sum (a : ℕ) (h1 : 1 < a) (h2 : ∃ (n : ℕ), (n = (a - 1) + 2 * (a - 1) + 1) ∧ n = 1979) : a = 44 :=
sorry

end cube_of_odd_sum_l2290_229085


namespace intersection_M_N_l2290_229008

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | ∃ y ∈ M, |y| = x}

-- The main theorem to prove M ∩ N = {0, 1, 2}
theorem intersection_M_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_M_N_l2290_229008


namespace total_books_written_l2290_229076

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end total_books_written_l2290_229076


namespace exponent_equality_l2290_229079

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end exponent_equality_l2290_229079


namespace repair_time_and_earnings_l2290_229003

-- Definitions based on given conditions
def cars : ℕ := 10
def cars_repair_50min : ℕ := 6
def repair_time_50min : ℕ := 50 -- minutes per car
def longer_percentage : ℕ := 80 -- 80% longer for the remaining cars
def wage_per_hour : ℕ := 30 -- dollars per hour

-- Remaining cars to repair
def remaining_cars : ℕ := cars - cars_repair_50min

-- Calculate total repair time for each type of cars and total repair time
def repair_time_remaining_cars : ℕ := repair_time_50min + (repair_time_50min * longer_percentage) / 100
def total_repair_time : ℕ := (cars_repair_50min * repair_time_50min) + (remaining_cars * repair_time_remaining_cars)

-- Convert total repair time from minutes to hours
def total_repair_hours : ℕ := total_repair_time / 60

-- Calculate total earnings
def total_earnings : ℕ := wage_per_hour * total_repair_hours

-- The theorem to be proved: total_repair_time == 660 and total_earnings == 330
theorem repair_time_and_earnings :
  total_repair_time = 660 ∧ total_earnings = 330 := by
  sorry

end repair_time_and_earnings_l2290_229003
