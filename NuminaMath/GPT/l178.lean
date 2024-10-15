import Mathlib

namespace NUMINAMATH_GPT_running_to_weightlifting_ratio_l178_17882

-- Definitions for given conditions in the problem
def total_practice_time : ℕ := 120 -- 120 minutes
def shooting_time : ℕ := total_practice_time / 2
def weightlifting_time : ℕ := 20
def running_time : ℕ := shooting_time - weightlifting_time

-- The goal is to prove that the ratio of running time to weightlifting time is 2:1
theorem running_to_weightlifting_ratio : running_time / weightlifting_time = 2 :=
by
  /- use the given problem conditions directly -/
  exact sorry

end NUMINAMATH_GPT_running_to_weightlifting_ratio_l178_17882


namespace NUMINAMATH_GPT_part1_part2_l178_17836

-- Definition of the function f
def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

-- Part 1: For m = 1, the solution set of f(x) >= 6
theorem part1 (x : ℝ) : f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := 
by 
  sorry

-- Part 2: If the inequality f(x) ≤ 2m - 5 has a solution with respect to x, then m ≥ 8
theorem part2 (m : ℝ) (h : ∃ x, f x m ≤ 2 * m - 5) : m ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l178_17836


namespace NUMINAMATH_GPT_find_point_P_l178_17815

def f (x : ℝ) : ℝ := x^4 - 2 * x

def tangent_line_perpendicular (x y : ℝ) : Prop :=
  (f x) = y ∧ (4 * x^3 - 2 = 2)

theorem find_point_P :
  ∃ (x y : ℝ), tangent_line_perpendicular x y ∧ x = 1 ∧ y = -1 :=
sorry

end NUMINAMATH_GPT_find_point_P_l178_17815


namespace NUMINAMATH_GPT_NumberOfRootsForEquation_l178_17837

noncomputable def numRootsAbsEq : ℕ :=
  let f := (fun x : ℝ => abs (abs (abs (abs (x - 1) - 9) - 9) - 3))
  let roots : List ℝ := [27, -25, 11, -9, 9, -7]
  roots.length

theorem NumberOfRootsForEquation : numRootsAbsEq = 6 := by
  sorry

end NUMINAMATH_GPT_NumberOfRootsForEquation_l178_17837


namespace NUMINAMATH_GPT_minimum_value_of_l178_17854

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem minimum_value_of (x y z : ℝ) (hxyz : x > 0 ∧ y > 0 ∧ z > 0) (h : 1/x + 1/y + 1/z = 9) :
  minimum_value x y z = 1 / 3456 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_l178_17854


namespace NUMINAMATH_GPT_line_intercepts_of_3x_minus_y_plus_6_eq_0_l178_17802

theorem line_intercepts_of_3x_minus_y_plus_6_eq_0 :
  (∃ y, 3 * 0 - y + 6 = 0 ∧ y = 6) ∧ (∃ x, 3 * x - 0 + 6 = 0 ∧ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_line_intercepts_of_3x_minus_y_plus_6_eq_0_l178_17802


namespace NUMINAMATH_GPT_gcd_of_three_l178_17894

theorem gcd_of_three (a b c : ℕ) (h₁ : a = 9242) (h₂ : b = 13863) (h₃ : c = 34657) :
  Nat.gcd (Nat.gcd a b) c = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_three_l178_17894


namespace NUMINAMATH_GPT_greatest_third_side_l178_17808

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end NUMINAMATH_GPT_greatest_third_side_l178_17808


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l178_17858

def M : Set ℤ := {x : ℤ | -4 < x ∧ x < 2}
def N : Set ℤ := {x : ℤ | x^2 < 4}

theorem intersection_of_M_and_N : (M ∩ N) = { -1, 0, 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l178_17858


namespace NUMINAMATH_GPT_initial_pups_per_mouse_l178_17833

-- Definitions from the problem's conditions
def initial_mice : ℕ := 8
def stress_factor : ℕ := 2
def second_round_pups : ℕ := 6
def total_mice : ℕ := 280

-- Define a variable for the initial number of pups each mouse had
variable (P : ℕ)

-- Lean statement to prove the number of initial pups per mouse
theorem initial_pups_per_mouse (P : ℕ) (initial_mice stress_factor second_round_pups total_mice : ℕ) :
  total_mice = initial_mice + initial_mice * P + (initial_mice + initial_mice * P) * second_round_pups - stress_factor * (initial_mice + initial_mice * P) → 
  P = 6 := 
by
  sorry

end NUMINAMATH_GPT_initial_pups_per_mouse_l178_17833


namespace NUMINAMATH_GPT_geometric_common_ratio_l178_17844

noncomputable def geo_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_common_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) 
  (h : 2 * geo_sum a₁ q n = geo_sum a₁ q (n + 1) + geo_sum a₁ q (n + 2)) : q = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_l178_17844


namespace NUMINAMATH_GPT_fraction_decimal_equivalent_l178_17822

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end NUMINAMATH_GPT_fraction_decimal_equivalent_l178_17822


namespace NUMINAMATH_GPT_total_students_accommodated_l178_17860

def num_columns : ℕ := 4
def num_rows : ℕ := 10
def num_buses : ℕ := 6

theorem total_students_accommodated : num_columns * num_rows * num_buses = 240 := by
  sorry

end NUMINAMATH_GPT_total_students_accommodated_l178_17860


namespace NUMINAMATH_GPT_find_c_for_circle_radius_5_l178_17866

theorem find_c_for_circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 + 8 * y + c = 0 
    → x^2 + 4 * x + y^2 + 8 * y = 5^2 - 25) 
  → c = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_c_for_circle_radius_5_l178_17866


namespace NUMINAMATH_GPT_coplanar_condition_l178_17824

-- Definitions representing points A, B, C, D and the origin O in a vector space over the reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (O A B C D : V)

-- The main statement of the problem
theorem coplanar_condition (h : (2 : ℝ) • (A - O) - (3 : ℝ) • (B - O) + (7 : ℝ) • (C - O) + k • (D - O) = 0) :
  k = -6 :=
sorry

end NUMINAMATH_GPT_coplanar_condition_l178_17824


namespace NUMINAMATH_GPT_sum_of_four_primes_div_by_60_l178_17829

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_four_primes_div_by_60
  (p q r s : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (horder : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  (p + q + r + s) % 60 = 0 :=
by
  sorry


end NUMINAMATH_GPT_sum_of_four_primes_div_by_60_l178_17829


namespace NUMINAMATH_GPT_selectedParticipants_correct_l178_17888

-- Define the random number table portion used in the problem
def randomNumTable := [
  [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76]
]

-- Define the conditions
def totalStudents := 247
def selectedStudentsCount := 4
def startingIndexRow := 4
def startingIndexCol := 9
def startingNumber := randomNumTable[0][8]

-- Define the expected selected participants' numbers
def expectedParticipants := [050, 121, 014, 218]

-- The Lean statement that needs to be proved
theorem selectedParticipants_correct : expectedParticipants = [050, 121, 014, 218] := by
  sorry

end NUMINAMATH_GPT_selectedParticipants_correct_l178_17888


namespace NUMINAMATH_GPT_solve_prime_equation_l178_17826

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem solve_prime_equation (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0) :
  (p = 17 ∧ q = 3) :=
sorry

end NUMINAMATH_GPT_solve_prime_equation_l178_17826


namespace NUMINAMATH_GPT_exponential_inequality_l178_17881

variable (a b : ℝ)

theorem exponential_inequality (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b :=
by
  sorry

end NUMINAMATH_GPT_exponential_inequality_l178_17881


namespace NUMINAMATH_GPT_who_is_first_l178_17893

def positions (A B C D : ℕ) : Prop :=
  A + B + D = 6 ∧ B + C = 6 ∧ B < A ∧ A + B + C + D = 10

theorem who_is_first (A B C D : ℕ) (h : positions A B C D) : D = 1 :=
sorry

end NUMINAMATH_GPT_who_is_first_l178_17893


namespace NUMINAMATH_GPT_sum_of_coeffs_l178_17865

theorem sum_of_coeffs (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5, (2 - x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5)
  → (a_0 = 32 ∧ 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5)
  → a_1 + a_2 + a_3 + a_4 + a_5 = -31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_l178_17865


namespace NUMINAMATH_GPT_payment_methods_20_yuan_l178_17804

theorem payment_methods_20_yuan :
  let ten_yuan_note := 10
  let five_yuan_note := 5
  let one_yuan_note := 1
  ∃ (methods : Nat), 
    methods = 9 ∧ 
    ∃ (num_10 num_5 num_1 : Nat),
      (num_10 * ten_yuan_note + num_5 * five_yuan_note + num_1 * one_yuan_note = 20) →
      methods = 9 :=
sorry

end NUMINAMATH_GPT_payment_methods_20_yuan_l178_17804


namespace NUMINAMATH_GPT_part1_part2_l178_17877

-- Step 1: Define necessary probabilities
def P_A1 : ℚ := 5 / 6
def P_A2 : ℚ := 2 / 3
def P_B1 : ℚ := 3 / 5
def P_B2 : ℚ := 3 / 4

-- Step 2: Winning event probabilities for both participants
def P_A_wins := P_A1 * P_A2
def P_B_wins := P_B1 * P_B2

-- Step 3: Problem statement: Comparing probabilities
theorem part1 (P_A_wins P_A_wins : ℚ) : P_A_wins > P_B_wins := 
  by sorry

-- Step 4: Complement probabilities for not winning the competition
def P_not_A_wins := 1 - P_A_wins
def P_not_B_wins := 1 - P_B_wins

-- Step 5: Probability at least one wins
def P_at_least_one_wins := 1 - (P_not_A_wins * P_not_B_wins)

-- Step 6: Problem statement: At least one wins
theorem part2 : P_at_least_one_wins = 34 / 45 := 
  by sorry

end NUMINAMATH_GPT_part1_part2_l178_17877


namespace NUMINAMATH_GPT_greatest_y_value_l178_17850

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end NUMINAMATH_GPT_greatest_y_value_l178_17850


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l178_17810

theorem solve_equation1 (x : ℝ) (h1 : 2 * x - 9 = 4 * x) : x = -9 / 2 :=
by
  sorry

theorem solve_equation2 (x : ℝ) (h2 : 5 / 2 * x - 7 / 3 * x = 4 / 3 * 5 - 5) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l178_17810


namespace NUMINAMATH_GPT_find_m_l178_17884

theorem find_m (m : ℝ) (A : Set ℝ) (hA : A = {0, m, m^2 - 3 * m + 2}) (h2 : 2 ∈ A) : m = 3 :=
  sorry

end NUMINAMATH_GPT_find_m_l178_17884


namespace NUMINAMATH_GPT_find_circle_center_l178_17892

theorem find_circle_center :
  ∃ (a b : ℝ), a = 1 / 2 ∧ b = 7 / 6 ∧
  (0 - a)^2 + (1 - b)^2 = (1 - a)^2 + (1 - b)^2 ∧
  (1 - a) * 3 = b - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_circle_center_l178_17892


namespace NUMINAMATH_GPT_length_of_field_l178_17870

variable (w : ℕ) (l : ℕ)

def length_field_is_double_width (w l : ℕ) : Prop :=
  l = 2 * w

def pond_area_equals_one_eighth_field_area (w l : ℕ) : Prop :=
  36 = 1 / 8 * (l * w)

theorem length_of_field (w l : ℕ) (h1 : length_field_is_double_width w l) (h2 : pond_area_equals_one_eighth_field_area w l) : l = 24 := 
by
  sorry

end NUMINAMATH_GPT_length_of_field_l178_17870


namespace NUMINAMATH_GPT_brianne_savings_in_may_l178_17855

-- Definitions based on conditions from a)
def initial_savings_jan : ℕ := 20
def multiplier : ℕ := 3
def additional_income : ℕ := 50

-- Savings in successive months
def savings_feb : ℕ := multiplier * initial_savings_jan
def savings_mar : ℕ := multiplier * savings_feb + additional_income
def savings_apr : ℕ := multiplier * savings_mar + additional_income
def savings_may : ℕ := multiplier * savings_apr + additional_income

-- The main theorem to verify
theorem brianne_savings_in_may : savings_may = 2270 :=
sorry

end NUMINAMATH_GPT_brianne_savings_in_may_l178_17855


namespace NUMINAMATH_GPT_average_weight_of_B_C_D_E_l178_17821

theorem average_weight_of_B_C_D_E 
    (W_A W_B W_C W_D W_E : ℝ)
    (h1 : (W_A + W_B + W_C)/3 = 60)
    (h2 : W_A = 87)
    (h3 : (W_A + W_B + W_C + W_D)/4 = 65)
    (h4 : W_E = W_D + 3) :
    (W_B + W_C + W_D + W_E)/4 = 64 :=
by {
    sorry
}

end NUMINAMATH_GPT_average_weight_of_B_C_D_E_l178_17821


namespace NUMINAMATH_GPT_num_ordered_pairs_solutions_l178_17862

theorem num_ordered_pairs_solutions :
  ∃ (n : ℕ), n = 18 ∧
    (∀ (a b : ℝ), (∃ x y : ℤ , a * (x : ℝ) + b * (y : ℝ) = 1 ∧ (x * x + y * y = 50))) :=
sorry

end NUMINAMATH_GPT_num_ordered_pairs_solutions_l178_17862


namespace NUMINAMATH_GPT_find_x_l178_17803

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_x_l178_17803


namespace NUMINAMATH_GPT_percentage_students_camping_trip_l178_17891

theorem percentage_students_camping_trip 
  (total_students : ℝ)
  (camping_trip_with_more_than_100 : ℝ) 
  (camping_trip_without_more_than_100_ratio : ℝ) :
  camping_trip_with_more_than_100 / (camping_trip_with_more_than_100 / 0.25) = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_percentage_students_camping_trip_l178_17891


namespace NUMINAMATH_GPT_members_who_didnt_show_up_l178_17867

theorem members_who_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : total_members = 5) (h2 : points_per_member = 6) (h3 : total_points = 18) : 
  total_members - total_points / points_per_member = 2 :=
by
  sorry

end NUMINAMATH_GPT_members_who_didnt_show_up_l178_17867


namespace NUMINAMATH_GPT_correct_calculation_l178_17879

theorem correct_calculation (x y : ℝ) : -x^2 * y + 3 * x^2 * y = 2 * x^2 * y :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l178_17879


namespace NUMINAMATH_GPT_total_length_of_wire_l178_17857

-- Definitions based on conditions
def num_squares : ℕ := 15
def length_of_grid : ℕ := 10
def width_of_grid : ℕ := 5
def height_of_grid : ℕ := 3
def side_length : ℕ := length_of_grid / width_of_grid -- 2 units
def num_horizontal_wires : ℕ := height_of_grid + 1    -- 4 wires
def num_vertical_wires : ℕ := width_of_grid + 1      -- 6 wires
def total_length_horizontal_wires : ℕ := num_horizontal_wires * length_of_grid -- 40 units
def total_length_vertical_wires : ℕ := num_vertical_wires * (height_of_grid * side_length) -- 36 units

-- The theorem to prove the total length of wire needed
theorem total_length_of_wire : total_length_horizontal_wires + total_length_vertical_wires = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_wire_l178_17857


namespace NUMINAMATH_GPT_smallest_angle_in_triangle_l178_17873

theorem smallest_angle_in_triangle (x : ℝ) 
  (h_ratio : 4 * x < 5 * x ∧ 5 * x < 9 * x) 
  (h_sum : 4 * x + 5 * x + 9 * x = 180) : 
  4 * x = 40 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_in_triangle_l178_17873


namespace NUMINAMATH_GPT_min_value_expression_l178_17869

open Real

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 60 := 
  sorry

end NUMINAMATH_GPT_min_value_expression_l178_17869


namespace NUMINAMATH_GPT_remainder_n_plus_2023_l178_17876

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 :=
by sorry

end NUMINAMATH_GPT_remainder_n_plus_2023_l178_17876


namespace NUMINAMATH_GPT_giftWrapperPerDay_l178_17880

variable (giftWrapperPerBox : ℕ)
variable (boxesPer3Days : ℕ)

def giftWrapperUsedIn3Days := giftWrapperPerBox * boxesPer3Days

theorem giftWrapperPerDay (h_giftWrapperPerBox : giftWrapperPerBox = 18)
  (h_boxesPer3Days : boxesPer3Days = 15) : giftWrapperUsedIn3Days giftWrapperPerBox boxesPer3Days / 3 = 90 :=
by
  sorry

end NUMINAMATH_GPT_giftWrapperPerDay_l178_17880


namespace NUMINAMATH_GPT_radical_axis_eq_l178_17806

-- Definitions of the given circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- The theorem proving that the equation of the radical axis is 3x - y - 9 = 0
theorem radical_axis_eq (x y : ℝ) :
  (circle1_eq x y) ∧ (circle2_eq x y) → 3 * x - y - 9 = 0 :=
sorry

end NUMINAMATH_GPT_radical_axis_eq_l178_17806


namespace NUMINAMATH_GPT_solve_for_x_l178_17817

-- Define the variables and conditions
variable (x : ℚ)

-- Define the given condition
def condition : Prop := (x + 4)/(x - 3) = (x - 2)/(x + 2)

-- State the theorem that x = -2/11 is a solution to the condition
theorem solve_for_x (h : condition x) : x = -2 / 11 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l178_17817


namespace NUMINAMATH_GPT_irreducible_fraction_iff_not_congruent_mod_5_l178_17851

theorem irreducible_fraction_iff_not_congruent_mod_5 (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := 
by 
  sorry

end NUMINAMATH_GPT_irreducible_fraction_iff_not_congruent_mod_5_l178_17851


namespace NUMINAMATH_GPT_min_k_plus_p_is_19199_l178_17856

noncomputable def find_min_k_plus_p : ℕ :=
  let D := 1007
  let domain_len := 1 / D
  let min_k : ℕ := 19  -- Minimum k value for which domain length condition holds, found via problem conditions
  let p_for_k (k : ℕ) : ℕ := (D * (k^2 - 1)) / k
  let k_plus_p (k : ℕ) : ℕ := k + p_for_k k
  k_plus_p min_k

theorem min_k_plus_p_is_19199 : find_min_k_plus_p = 19199 :=
  sorry

end NUMINAMATH_GPT_min_k_plus_p_is_19199_l178_17856


namespace NUMINAMATH_GPT_area_new_rectangle_l178_17830

theorem area_new_rectangle (a b : ℝ) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
sorry

end NUMINAMATH_GPT_area_new_rectangle_l178_17830


namespace NUMINAMATH_GPT_Tigers_Sharks_min_games_l178_17842

open Nat

theorem Tigers_Sharks_min_games (N : ℕ) : 
  (let total_games := 3 + N
   let sharks_wins := 1 + N
   sharks_wins * 20 ≥ total_games * 19) ↔ N ≥ 37 := 
by
  sorry

end NUMINAMATH_GPT_Tigers_Sharks_min_games_l178_17842


namespace NUMINAMATH_GPT_integer_solutions_count_l178_17875

theorem integer_solutions_count :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 15 ∧
    ∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ (∃ x y, pair = (x, y) ∧ (Nat.sqrt x + Nat.sqrt y = 14))) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l178_17875


namespace NUMINAMATH_GPT_intersection_point_parabola_l178_17868

theorem intersection_point_parabola :
  ∃ k : ℝ, (∀ x : ℝ, (3 * (x - 4)^2 + k = 0 ↔ x = 2 ∨ x = 6)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_parabola_l178_17868


namespace NUMINAMATH_GPT_find_a_in_terms_of_y_l178_17805

theorem find_a_in_terms_of_y (a b y : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * y^3) (h3 : a - b = 3 * y) :
  a = 3 * y :=
sorry

end NUMINAMATH_GPT_find_a_in_terms_of_y_l178_17805


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l178_17859

-- Define the sides of the triangle
def AB : ℝ := 12
def BC : ℝ := 9

-- Define the expected area of the triangle
def expectedArea : ℝ := 54

-- Prove the area of the triangle using the given conditions
theorem area_of_triangle_ABC : (1/2) * AB * BC = expectedArea := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l178_17859


namespace NUMINAMATH_GPT_min_value_SN64_by_aN_is_17_over_2_l178_17887

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_SN64_by_aN_is_17_over_2_l178_17887


namespace NUMINAMATH_GPT_pies_from_apples_l178_17823

theorem pies_from_apples 
  (initial_apples : ℕ) (handed_out_apples : ℕ) (apples_per_pie : ℕ) 
  (remaining_apples := initial_apples - handed_out_apples) 
  (pies := remaining_apples / apples_per_pie) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out_apples = 19) 
  (h3 : apples_per_pie = 8) : 
  pies = 7 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_pies_from_apples_l178_17823


namespace NUMINAMATH_GPT_value_of_x_plus_y_pow_2023_l178_17832

theorem value_of_x_plus_y_pow_2023 (x y : ℝ) (h : abs (x - 2) + abs (y + 3) = 0) : 
  (x + y) ^ 2023 = -1 := 
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_pow_2023_l178_17832


namespace NUMINAMATH_GPT_function_range_ge_4_l178_17814

variable {x : ℝ}

theorem function_range_ge_4 (h : x > 0) : 2 * x + 2 * x⁻¹ ≥ 4 :=
sorry

end NUMINAMATH_GPT_function_range_ge_4_l178_17814


namespace NUMINAMATH_GPT_solution_set_of_inequality_l178_17839

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set_of_inequality (H1 : f 1 = 1)
  (H2 : ∀ x : ℝ, x * f' x < 1 / 2) :
  {x : ℝ | f (Real.log x ^ 2) < (Real.log x ^ 2) / 2 + 1 / 2} = 
  {x : ℝ | 0 < x ∧ x < 1 / 10} ∪ {x : ℝ | x > 10} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l178_17839


namespace NUMINAMATH_GPT_find_a_l178_17846

theorem find_a (f g : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, f x = 2 * x / 3 + 4) 
  (h₂ : ∀ x, g x = 5 - 2 * x) 
  (h₃ : f (g a) = 7) : 
  a = 1 / 4 := 
sorry

end NUMINAMATH_GPT_find_a_l178_17846


namespace NUMINAMATH_GPT_smallest_angle_satisfying_trig_eqn_l178_17831

theorem smallest_angle_satisfying_trig_eqn :
  ∃ x : ℝ, 0 < x ∧ 8 * (Real.sin x)^2 * (Real.cos x)^4 - 8 * (Real.sin x)^4 * (Real.cos x)^2 = 1 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_satisfying_trig_eqn_l178_17831


namespace NUMINAMATH_GPT_ahn_largest_number_l178_17809

def largest_number_ahn_can_get : ℕ :=
  let n := 10
  2 * (200 - n)

theorem ahn_largest_number :
  (10 ≤ 99) →
  (10 ≤ 99) →
  largest_number_ahn_can_get = 380 := 
by
-- Conditions: n is a two-digit integer with range 10 ≤ n ≤ 99
-- Proof is skipped
  sorry

end NUMINAMATH_GPT_ahn_largest_number_l178_17809


namespace NUMINAMATH_GPT_sides_of_triangle_inequality_l178_17807

theorem sides_of_triangle_inequality (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_sides_of_triangle_inequality_l178_17807


namespace NUMINAMATH_GPT_expression_simplification_l178_17861

theorem expression_simplification (x y : ℝ) :
  20 * (x + y) - 19 * (x + y) = x + y :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l178_17861


namespace NUMINAMATH_GPT_profit_per_metre_l178_17853

/-- 
Given:
1. A trader sells 85 meters of cloth for Rs. 8925.
2. The cost price of one metre of cloth is Rs. 95.

Prove:
The profit per metre of cloth is Rs. 10.
-/
theorem profit_per_metre 
  (SP : ℕ) (CP : ℕ)
  (total_SP : SP = 8925)
  (total_meters : ℕ := 85)
  (cost_per_meter : CP = 95) :
  (SP - total_meters * CP) / total_meters = 10 :=
by
  sorry

end NUMINAMATH_GPT_profit_per_metre_l178_17853


namespace NUMINAMATH_GPT_cannot_fill_box_exactly_l178_17872

def box_length : ℝ := 70
def box_width : ℝ := 40
def box_height : ℝ := 25
def cube_side : ℝ := 4.5

theorem cannot_fill_box_exactly : 
  ¬ (∃ n : ℕ, n * cube_side^3 = box_length * box_width * box_height ∧
               (∃ x y z : ℕ, x * cube_side = box_length ∧ 
                             y * cube_side = box_width ∧ 
                             z * cube_side = box_height)) :=
by sorry

end NUMINAMATH_GPT_cannot_fill_box_exactly_l178_17872


namespace NUMINAMATH_GPT_constant_fraction_condition_l178_17835

theorem constant_fraction_condition 
    (a1 b1 c1 a2 b2 c2 : ℝ) : 
    (∀ x : ℝ, (a1 * x^2 + b1 * x + c1) / (a2 * x^2 + b2 * x + c2) = k) ↔ 
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :=
by
  sorry

end NUMINAMATH_GPT_constant_fraction_condition_l178_17835


namespace NUMINAMATH_GPT_martin_goldfish_count_l178_17849

-- Define the initial number of goldfish
def initial_goldfish := 18

-- Define the number of goldfish that die each week
def goldfish_die_per_week := 5

-- Define the number of goldfish purchased each week
def goldfish_purchased_per_week := 3

-- Define the number of weeks
def weeks := 7

-- Calculate the expected number of goldfish after 7 weeks
noncomputable def final_goldfish := initial_goldfish - (goldfish_die_per_week * weeks) + (goldfish_purchased_per_week * weeks)

-- State the theorem and the proof target
theorem martin_goldfish_count : final_goldfish = 4 := 
sorry

end NUMINAMATH_GPT_martin_goldfish_count_l178_17849


namespace NUMINAMATH_GPT_power_set_card_greater_l178_17898

open Set

variables {A : Type*} (α : ℕ) [Fintype A] (hA : Fintype.card A = α)

theorem power_set_card_greater (h : Fintype.card A = α) :
  2 ^ α > α :=
sorry

end NUMINAMATH_GPT_power_set_card_greater_l178_17898


namespace NUMINAMATH_GPT_faye_age_l178_17897

variables (C D E F G : ℕ)
variables (h1 : D = E - 2)
variables (h2 : E = C + 6)
variables (h3 : F = C + 4)
variables (h4 : G = C - 5)
variables (h5 : D = 16)

theorem faye_age : F = 16 :=
by
  -- Proof will be placed here
  sorry

end NUMINAMATH_GPT_faye_age_l178_17897


namespace NUMINAMATH_GPT_real_roots_iff_l178_17801

theorem real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 + 2 * k = 0) ↔ (-1 ≤ k ∧ k ≤ 0) :=
by sorry

end NUMINAMATH_GPT_real_roots_iff_l178_17801


namespace NUMINAMATH_GPT_stratified_sampling_correct_l178_17890

-- Define the conditions
def num_freshmen : ℕ := 900
def num_sophomores : ℕ := 1200
def num_seniors : ℕ := 600
def total_sample_size : ℕ := 135
def total_students := num_freshmen + num_sophomores + num_seniors

-- Proportions
def proportion_freshmen := (num_freshmen : ℚ) / total_students
def proportion_sophomores := (num_sophomores : ℚ) / total_students
def proportion_seniors := (num_seniors : ℚ) / total_students

-- Expected samples count
def expected_freshmen_samples := (total_sample_size : ℚ) * proportion_freshmen
def expected_sophomores_samples := (total_sample_size : ℚ) * proportion_sophomores
def expected_seniors_samples := (total_sample_size : ℚ) * proportion_seniors

-- Statement to be proven
theorem stratified_sampling_correct :
  expected_freshmen_samples = (45 : ℚ) ∧
  expected_sophomores_samples = (60 : ℚ) ∧
  expected_seniors_samples = (30 : ℚ) := by
  -- Provide the necessary proof or calculation
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l178_17890


namespace NUMINAMATH_GPT_correct_student_mark_l178_17838

theorem correct_student_mark :
  ∀ (total_marks total_correct_marks incorrect_mark correct_average students : ℝ)
  (h1 : total_marks = students * 100)
  (h2 : incorrect_mark = 60)
  (h3 : correct_average = 95)
  (h4 : total_correct_marks = students * correct_average),
  total_marks - incorrect_mark + (total_correct_marks - (total_marks - incorrect_mark)) = 10 :=
by
  intros total_marks total_correct_marks incorrect_mark correct_average students h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_correct_student_mark_l178_17838


namespace NUMINAMATH_GPT_original_inhabitants_7200_l178_17828

noncomputable def original_inhabitants (X : ℝ) : Prop :=
  let initial_decrease := 0.9 * X
  let final_decrease := 0.75 * initial_decrease
  final_decrease = 4860

theorem original_inhabitants_7200 : ∃ X : ℝ, original_inhabitants X ∧ X = 7200 := by
  use 7200
  unfold original_inhabitants
  simp
  sorry

end NUMINAMATH_GPT_original_inhabitants_7200_l178_17828


namespace NUMINAMATH_GPT_part1_part2_part3_l178_17841

noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.exp x + 1

theorem part1 :
  ∃ x0 > 0, f x0 = 0 :=
sorry

theorem part2 (x0 : ℝ) (h1 : f x0 = 0) :
  ∀ x, f x ≤ (3 - Real.exp x0) * (x - x0) :=
sorry

theorem part3 (m x1 x2 : ℝ) (h1 : m > 0) (h2 : x1 < x2) (h3 : f x1 = m) (h4 : f x2 = m):
  x2 - x1 < 2 - 3 * m / 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l178_17841


namespace NUMINAMATH_GPT_value_of_expression_l178_17848

theorem value_of_expression : (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = (27 / 89) :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l178_17848


namespace NUMINAMATH_GPT_find_ratio_l178_17845

theorem find_ratio (x y c d : ℝ) (h1 : 8 * x - 6 * y = c) (h2 : 12 * y - 18 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = -1 := by
  sorry

end NUMINAMATH_GPT_find_ratio_l178_17845


namespace NUMINAMATH_GPT_transformation_correct_l178_17883

variables {x y : ℝ}

theorem transformation_correct (h : x = y) : x - 2 = y - 2 := by
  sorry

end NUMINAMATH_GPT_transformation_correct_l178_17883


namespace NUMINAMATH_GPT_expected_value_equals_51_l178_17840

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end NUMINAMATH_GPT_expected_value_equals_51_l178_17840


namespace NUMINAMATH_GPT_elena_probability_at_least_one_correct_l178_17825

-- Conditions
def total_questions := 30
def choices_per_question := 4
def guessed_questions := 6
def incorrect_probability_single := 3 / 4

-- Expression for the probability of missing all guessed questions
def probability_all_incorrect := (incorrect_probability_single) ^ guessed_questions

-- Calculation from the solution
def probability_at_least_one_correct := 1 - probability_all_incorrect

-- Problem statement to prove
theorem elena_probability_at_least_one_correct : probability_at_least_one_correct = 3367 / 4096 :=
by sorry

end NUMINAMATH_GPT_elena_probability_at_least_one_correct_l178_17825


namespace NUMINAMATH_GPT_basketball_price_l178_17863

variable (P : ℝ)

def coachA_cost : ℝ := 10 * P
def coachB_baseball_cost : ℝ := 14 * 2.5
def coachB_bat_cost : ℝ := 18
def coachB_total_cost : ℝ := coachB_baseball_cost + coachB_bat_cost
def coachA_excess_cost : ℝ := 237

theorem basketball_price (h : coachA_cost P = coachB_total_cost + coachA_excess_cost) : P = 29 :=
by
  sorry

end NUMINAMATH_GPT_basketball_price_l178_17863


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l178_17886

-- Condition: Let M be the number of men in the first group
variable (M : ℕ)

-- Condition: M men can complete the work in 20 hours
-- Condition: 15 men can complete the same work in 48 hours
-- We want to prove that if M * 20 = 15 * 48, then M = 36
theorem number_of_men_in_first_group (h : M * 20 = 15 * 48) : M = 36 := by
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l178_17886


namespace NUMINAMATH_GPT_intersection_M_N_l178_17819

-- Define the sets based on the given conditions
def M : Set ℝ := {x | x + 2 < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {x | x < -2} := by
sorry

end NUMINAMATH_GPT_intersection_M_N_l178_17819


namespace NUMINAMATH_GPT_weight_loss_percentage_l178_17878

theorem weight_loss_percentage (W : ℝ) (hW : W > 0) : 
  let new_weight := 0.89 * W
  let final_weight_with_clothes := new_weight * 1.02
  (W - final_weight_with_clothes) / W * 100 = 9.22 := by
  sorry

end NUMINAMATH_GPT_weight_loss_percentage_l178_17878


namespace NUMINAMATH_GPT_mary_keep_warm_hours_l178_17885

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end NUMINAMATH_GPT_mary_keep_warm_hours_l178_17885


namespace NUMINAMATH_GPT_worker_surveys_per_week_l178_17827

theorem worker_surveys_per_week :
  let regular_rate := 30
  let cellphone_rate := regular_rate + 0.20 * regular_rate
  let surveys_with_cellphone := 50
  let earnings := 3300
  cellphone_rate = regular_rate + 0.20 * regular_rate →
  earnings = surveys_with_cellphone * cellphone_rate →
  regular_rate = 30 →
  surveys_with_cellphone = 50 →
  earnings = 3300 →
  surveys_with_cellphone = 50 := sorry

end NUMINAMATH_GPT_worker_surveys_per_week_l178_17827


namespace NUMINAMATH_GPT_boxes_A_B_cost_condition_boxes_B_profit_condition_l178_17820

/-
Part 1: Prove the number of brand A boxes is 60 and number of brand B boxes is 40 given the cost condition.
-/
theorem boxes_A_B_cost_condition (x : ℕ) (y : ℕ) :
  80 * x + 130 * y = 10000 ∧ x + y = 100 → x = 60 ∧ y = 40 :=
by sorry

/-
Part 2: Prove the number of brand B boxes should be at least 54 given the profit condition.
-/
theorem boxes_B_profit_condition (y : ℕ) :
  40 * (100 - y) + 70 * y ≥ 5600 → y ≥ 54 :=
by sorry

end NUMINAMATH_GPT_boxes_A_B_cost_condition_boxes_B_profit_condition_l178_17820


namespace NUMINAMATH_GPT_Y_tagged_value_l178_17874

variables (W X Y Z : ℕ)
variables (tag_W : W = 200)
variables (tag_X : X = W / 2)
variables (tag_Z : Z = 400)
variables (total : W + X + Y + Z = 1000)

theorem Y_tagged_value : Y = 300 :=
by sorry

end NUMINAMATH_GPT_Y_tagged_value_l178_17874


namespace NUMINAMATH_GPT_min_value_of_a_l178_17818

theorem min_value_of_a : 
  ∃ (a : ℤ), ∃ x y : ℤ, x ≠ y ∧ |x| ≤ 10 ∧ (x - y^2 = a) ∧ (y - x^2 = a) ∧ a = -111 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_l178_17818


namespace NUMINAMATH_GPT_sum_of_remainders_eq_24_l178_17800

theorem sum_of_remainders_eq_24 (a b c : ℕ) 
  (h1 : a % 30 = 13) (h2 : b % 30 = 19) (h3 : c % 30 = 22) :
  (a + b + c) % 30 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_eq_24_l178_17800


namespace NUMINAMATH_GPT_system_solution_unique_l178_17811

theorem system_solution_unique (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x ^ 3 + 2 * y ^ 2 + 1 / (4 * z) = 1)
  (eq2 : y ^ 3 + 2 * z ^ 2 + 1 / (4 * x) = 1)
  (eq3 : z ^ 3 + 2 * x ^ 2 + 1 / (4 * y) = 1) :
  (x, y, z) = ( ( (-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2) ) := 
by
  sorry

end NUMINAMATH_GPT_system_solution_unique_l178_17811


namespace NUMINAMATH_GPT_find_integer_mul_a_l178_17852

noncomputable def integer_mul_a (a b : ℤ) (n : ℤ) : Prop :=
  n * a * (-8 * b) + a * b = 89 ∧ n < 0 ∧ n * a < 0 ∧ -8 * b < 0

theorem find_integer_mul_a (a b : ℤ) (n : ℤ) (h : integer_mul_a a b n) : n = -11 :=
  sorry

end NUMINAMATH_GPT_find_integer_mul_a_l178_17852


namespace NUMINAMATH_GPT_candy_mixture_cost_l178_17847

/-- 
A club mixes 15 pounds of candy worth $8.00 per pound with 30 pounds of candy worth $5.00 per pound.
We need to find the cost per pound of the mixture.
-/
theorem candy_mixture_cost :
    (15 * 8 + 30 * 5) / (15 + 30) = 6 := 
by
  sorry

end NUMINAMATH_GPT_candy_mixture_cost_l178_17847


namespace NUMINAMATH_GPT_tan_eq_860_l178_17864

theorem tan_eq_860 (n : ℤ) (hn : -180 < n ∧ n < 180) : 
  n = -40 ↔ (Real.tan (n * Real.pi / 180) = Real.tan (860 * Real.pi / 180)) := 
sorry

end NUMINAMATH_GPT_tan_eq_860_l178_17864


namespace NUMINAMATH_GPT_minor_premise_is_wrong_l178_17895

theorem minor_premise_is_wrong (a : ℝ) : ¬ (0 < a^2) := by
  sorry

end NUMINAMATH_GPT_minor_premise_is_wrong_l178_17895


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l178_17889

noncomputable def vertex_angle_of_isosceles (a b : ℝ) : ℝ :=
  if a = b then 40 else 100

theorem isosceles_triangle_vertex_angle (a : ℝ) (interior_angle : ℝ)
  (h_isosceles : a = 40 ∨ a = interior_angle ∧ interior_angle = 40 ∨ interior_angle = 100) :
  vertex_angle_of_isosceles a interior_angle = 40 ∨ vertex_angle_of_isosceles a interior_angle = 100 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l178_17889


namespace NUMINAMATH_GPT_crit_value_expr_l178_17896

theorem crit_value_expr : 
  ∃ x : ℝ, -4 < x ∧ x < 1 ∧ (x^2 - 2*x + 2) / (2*x - 2) = -1 :=
sorry

end NUMINAMATH_GPT_crit_value_expr_l178_17896


namespace NUMINAMATH_GPT_base7_calculation_result_l178_17813

-- Define the base 7 addition and multiplication
def base7_add (a b : ℕ) := (a + b)
def base7_mul (a b : ℕ) := (a * b)

-- Represent the given numbers in base 10 for calculations:
def num1 : ℕ := 2 * 7 + 5 -- 25 in base 7
def num2 : ℕ := 3 * 7^2 + 3 * 7 + 4 -- 334 in base 7
def mul_factor : ℕ := 2 -- 2 in base 7

-- Addition result
def sum : ℕ := base7_add num1 num2

-- Multiplication result
def result : ℕ := base7_mul sum mul_factor

-- Proving the result is equal to the final answer in base 7
theorem base7_calculation_result : result = 6 * 7^2 + 6 * 7 + 4 := 
by sorry

end NUMINAMATH_GPT_base7_calculation_result_l178_17813


namespace NUMINAMATH_GPT_inequality_for_positive_a_b_n_l178_17871

theorem inequality_for_positive_a_b_n (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end NUMINAMATH_GPT_inequality_for_positive_a_b_n_l178_17871


namespace NUMINAMATH_GPT_journey_total_distance_l178_17843

theorem journey_total_distance (D : ℝ) (h_train : D * (3 / 5) = t) (h_bus : D * (7 / 20) = b) (h_walk : D * (1 - ((3 / 5) + (7 / 20))) = 6.5) : D = 130 :=
by
  sorry

end NUMINAMATH_GPT_journey_total_distance_l178_17843


namespace NUMINAMATH_GPT_price_range_of_book_l178_17899

variable (x : ℝ)

theorem price_range_of_book (h₁ : ¬(x ≥ 15)) (h₂ : ¬(x ≤ 12)) (h₃ : ¬(x ≤ 10)) : 12 < x ∧ x < 15 := 
by
  sorry

end NUMINAMATH_GPT_price_range_of_book_l178_17899


namespace NUMINAMATH_GPT_fold_point_area_sum_l178_17834

noncomputable def fold_point_area (AB AC : ℝ) (angle_B : ℝ) : ℝ :=
  let BC := Real.sqrt (AB ^ 2 + AC ^ 2)
  -- Assuming the fold point area calculation as per the problem's solution
  let q := 270
  let r := 324
  let s := 3
  q * Real.pi - r * Real.sqrt s

theorem fold_point_area_sum (AB AC : ℝ) (angle_B : ℝ) (hAB : AB = 36) (hAC : AC = 72) (hangle_B : angle_B = π / 2) :
  let S := fold_point_area AB AC angle_B
  ∃ q r s : ℕ, S = q * Real.pi - r * Real.sqrt s ∧ q + r + s = 597 :=
by
  sorry

end NUMINAMATH_GPT_fold_point_area_sum_l178_17834


namespace NUMINAMATH_GPT_correct_proposition_l178_17812

-- Definitions of the propositions p and q
def p : Prop := ∀ x : ℝ, (x > 1 → x > 2)
def q : Prop := ∀ x y : ℝ, (x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1)

-- The proof problem statement
theorem correct_proposition : ¬p ∧ q :=
by
  -- Assuming p is false (i.e., ¬p is true) and q is true
  sorry

end NUMINAMATH_GPT_correct_proposition_l178_17812


namespace NUMINAMATH_GPT_increasing_interval_of_f_l178_17816

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * Real.pi / 3 - 2 * x)

theorem increasing_interval_of_f :
  ∃ a b : ℝ, f x = 3 * Real.sin (2 * Real.pi / 3 - 2 * x) ∧ (a = 7 * Real.pi / 12) ∧ (b = 13 * Real.pi / 12) ∧ ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := 
sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l178_17816
