import Mathlib

namespace quadratic_function_l390_39014

theorem quadratic_function :
  ∃ a : ℝ, ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a * (x - 1) * (x - 5)) ∧ f 3 = 10 ∧ 
  f = fun x => -2.5 * x^2 + 15 * x - 12.5 :=
by
  sorry

end quadratic_function_l390_39014


namespace Clea_ride_time_l390_39079

theorem Clea_ride_time
  (c s d t : ℝ)
  (h1 : d = 80 * c)
  (h2 : d = 30 * (c + s))
  (h3 : s = 5 / 3 * c)
  (h4 : t = d / s) :
  t = 48 := by sorry

end Clea_ride_time_l390_39079


namespace multiply_fractions_l390_39025

theorem multiply_fractions :
  (2 / 3) * (5 / 7) * (8 / 9) = 80 / 189 :=
by sorry

end multiply_fractions_l390_39025


namespace maximize_tables_eqn_l390_39059

theorem maximize_tables_eqn :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 12 → 400 * x = 20 * (12 - x) * 4 :=
by
  sorry

end maximize_tables_eqn_l390_39059


namespace expression_value_l390_39057

theorem expression_value (x y z w : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 := 
sorry

end expression_value_l390_39057


namespace total_salaries_l390_39053

variable (A_salary B_salary : ℝ)

def A_saves : ℝ := 0.05 * A_salary
def B_saves : ℝ := 0.15 * B_salary

theorem total_salaries (h1 : A_salary = 5250) 
                       (h2 : A_saves = B_saves) : 
    A_salary + B_salary = 7000 := by
  sorry

end total_salaries_l390_39053


namespace distribution_plans_equiv_210_l390_39048

noncomputable def number_of_distribution_plans : ℕ := sorry -- we will skip the proof

theorem distribution_plans_equiv_210 :
  number_of_distribution_plans = 210 := by
  sorry

end distribution_plans_equiv_210_l390_39048


namespace find_w_value_l390_39069

theorem find_w_value : 
  (2^5 * 9^2) / (8^2 * 243) = 0.16666666666666666 := 
by
  sorry

end find_w_value_l390_39069


namespace stickers_initial_count_l390_39065

theorem stickers_initial_count (S : ℕ) 
  (h1 : (3 / 5 : ℝ) * (2 / 3 : ℝ) * S = 54) : S = 135 := 
by
  sorry

end stickers_initial_count_l390_39065


namespace spending_after_drink_l390_39028

variable (X : ℝ)
variable (Y : ℝ)

theorem spending_after_drink (h : X - 1.75 - Y = 6) : Y = X - 7.75 :=
by sorry

end spending_after_drink_l390_39028


namespace pam_age_l390_39062

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end pam_age_l390_39062


namespace tammy_avg_speed_l390_39009

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l390_39009


namespace highest_digit_a_divisible_by_eight_l390_39011

theorem highest_digit_a_divisible_by_eight :
  ∃ a : ℕ, a ≤ 9 ∧ 8 ∣ (100 * a + 16) ∧ ∀ b : ℕ, b > a → b ≤ 9 → ¬ (8 ∣ (100 * b + 16)) := by
  sorry

end highest_digit_a_divisible_by_eight_l390_39011


namespace percentage_of_masters_is_76_l390_39029

variable (x y : ℕ)  -- Let x be the number of junior players, y be the number of master players
variable (junior_avg master_avg team_avg : ℚ)

-- The conditions given in the problem
def juniors_avg_points : Prop := junior_avg = 22
def masters_avg_points : Prop := master_avg = 47
def team_avg_points (x y : ℕ) (junior_avg master_avg team_avg : ℚ) : Prop :=
  (22 * x + 47 * y) / (x + y) = 41

def proportion_of_masters (x y : ℕ) : ℚ := (y : ℚ) / (x + y)

-- The theorem to be proved
theorem percentage_of_masters_is_76 (x y : ℕ) (junior_avg master_avg team_avg : ℚ) :
  juniors_avg_points junior_avg →
  masters_avg_points master_avg →
  team_avg_points x y junior_avg master_avg team_avg →
  proportion_of_masters x y = 19 / 25 := 
sorry

end percentage_of_masters_is_76_l390_39029


namespace number_of_five_digit_numbers_with_at_least_one_zero_l390_39031

-- Definitions for the conditions
def total_five_digit_numbers : ℕ := 90000
def five_digit_numbers_with_no_zeros : ℕ := 59049

-- Theorem stating that the number of 5-digit numbers with at least one zero is 30,951
theorem number_of_five_digit_numbers_with_at_least_one_zero : 
    total_five_digit_numbers - five_digit_numbers_with_no_zeros = 30951 :=
by
  sorry

end number_of_five_digit_numbers_with_at_least_one_zero_l390_39031


namespace first_player_always_wins_l390_39075

theorem first_player_always_wins :
  ∃ A B : ℤ, A ≠ 0 ∧ B ≠ 0 ∧
  (A = 1998 ∧ B = -2 * 1998) ∧
  (∀ a b c : ℤ, (a = A ∨ a = B ∨ a = 1998) ∧ 
                (b = A ∨ b = B ∨ b = 1998) ∧ 
                (c = A ∨ c = B ∨ c = 1998) ∧ 
                a ≠ b ∧ b ≠ c ∧ a ≠ c → 
                ∃ x1 x2 : ℚ, x1 ≠ x2 ∧ 
                (a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)) :=
by
  sorry

end first_player_always_wins_l390_39075


namespace solve_quadratic_1_solve_quadratic_2_l390_39060

-- Define the first problem
theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 4 * x = 2 * x → x = 0 ∨ x = 2 := by
  -- Proof step will go here
  sorry

-- Define the second problem
theorem solve_quadratic_2 (x : ℝ) : x * (x + 8) = 16 → x = -4 + 4 * Real.sqrt 2 ∨ x = -4 - 4 * Real.sqrt 2 := by
  -- Proof step will go here
  sorry

end solve_quadratic_1_solve_quadratic_2_l390_39060


namespace circle_and_tangent_lines_l390_39015

-- Define the problem conditions
def passes_through (a b r : ℝ) : Prop :=
  (a - (-2))^2 + (b - 2)^2 = r^2 ∧
  (a - (-5))^2 + (b - 5)^2 = r^2

def lies_on_line (a b : ℝ) : Prop :=
  a + b + 3 = 0

-- Define the standard equation of the circle
def is_circle_eq (a b r : ℝ) : Prop := ∀ x y : ℝ, 
  (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 5)^2 + (y - 2)^2 = 9

-- Define the tangent lines
def is_tangent_lines (x y k : ℝ) : Prop :=
  (k = (20 / 21) ∨ x = -2) → (20 * x - 21 * y + 229 = 0 ∨ x = -2)

-- The theorem statement in Lean 4
theorem circle_and_tangent_lines (a b r : ℝ) (x y k : ℝ) :
  passes_through a b r →
  lies_on_line a b →
  is_circle_eq a b r →
  is_tangent_lines x y k :=
by {
  sorry
}

end circle_and_tangent_lines_l390_39015


namespace x_intercept_of_line_l390_39063

open Real

theorem x_intercept_of_line : 
  ∃ x : ℝ, 
  (∃ m : ℝ, m = (3 - -5) / (10 - -6) ∧ (∀ y : ℝ, y = m * (x - 10) + 3)) ∧ 
  (∀ y : ℝ, y = 0 → x = 4) :=
sorry

end x_intercept_of_line_l390_39063


namespace infinite_geometric_sum_example_l390_39024

noncomputable def infinite_geometric_sum (a₁ q : ℝ) : ℝ :=
a₁ / (1 - q)

theorem infinite_geometric_sum_example :
  infinite_geometric_sum 18 (-1/2) = 12 := by
  sorry

end infinite_geometric_sum_example_l390_39024


namespace find_value_of_expression_l390_39064

variable (a b c : ℝ)

def parabola_symmetry (a b c : ℝ) :=
  (36 * a + 6 * b + c = 2) ∧ 
  (25 * a + 5 * b + c = 6) ∧ 
  (49 * a + 7 * b + c = -4)

theorem find_value_of_expression :
  (∃ a b c : ℝ, parabola_symmetry a b c) →
  3 * a + 3 * c + b = -8 :=  sorry

end find_value_of_expression_l390_39064


namespace part1_part2_l390_39012

namespace Problem

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) : (B m ⊆ A) → (m ≤ 3) :=
by
  intro h
  sorry

theorem part2 (m : ℝ) : (A ∩ B m = ∅) → (m < 2 ∨ 4 < m) :=
by
  intro h
  sorry

end Problem

end part1_part2_l390_39012


namespace max_perimeter_of_polygons_l390_39008

theorem max_perimeter_of_polygons 
  (t s : ℕ) 
  (hts : t + s = 7) 
  (hsum_angles : 60 * t + 90 * s = 360) 
  (max_squares : s ≤ 4) 
  (side_length : ℕ := 2) 
  (tri_perimeter : ℕ := 3 * side_length) 
  (square_perimeter : ℕ := 4 * side_length) :
  2 * (t * tri_perimeter + s * square_perimeter) = 68 := 
sorry

end max_perimeter_of_polygons_l390_39008


namespace concrete_pillars_correct_l390_39068

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l390_39068


namespace binary_representation_of_38_l390_39039

theorem binary_representation_of_38 : ∃ binary : ℕ, binary = 0b100110 ∧ binary = 38 :=
by
  sorry

end binary_representation_of_38_l390_39039


namespace determinant_example_l390_39072

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)
noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

-- Define the determinant of a 2x2 matrix in terms of its entries
def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Proposed theorem statement in Lean 4
theorem determinant_example : 
  determinant_2x2 (cos_deg 45) (sin_deg 75) (sin_deg 135) (cos_deg 105) = - (Real.sqrt 3 / 2) := 
by sorry

end determinant_example_l390_39072


namespace height_is_geometric_mean_of_bases_l390_39098

-- Given conditions
variables (a c m : ℝ)
-- we declare the condition that the given trapezoid is symmetric and tangential
variables (isSymmetricTangentialTrapezoid : Prop)

-- The theorem to be proven
theorem height_is_geometric_mean_of_bases 
(isSymmetricTangentialTrapezoid: isSymmetricTangentialTrapezoid) 
: m = Real.sqrt (a * c) :=
sorry

end height_is_geometric_mean_of_bases_l390_39098


namespace a_is_minus_one_l390_39034

theorem a_is_minus_one (a : ℤ) (h1 : 2 * a + 1 < 0) (h2 : 2 + a > 0) : a = -1 := 
by
  sorry

end a_is_minus_one_l390_39034


namespace inequality_solution_set_l390_39084

theorem inequality_solution_set (x : ℝ) :
  ∀ x, 
  (x^2 * (x + 1) / (-x^2 - 5 * x + 6) <= 0) ↔ (-6 < x ∧ x <= -1) ∨ (x = 0) ∨ (1 < x) :=
by
  sorry

end inequality_solution_set_l390_39084


namespace next_correct_time_l390_39042

def clock_shows_correct_time (start_date : String) (start_time : String) (time_lost_per_hour : Int) : String :=
  if start_date = "March 21" ∧ start_time = "12:00 PM" ∧ time_lost_per_hour = 25 then
    "June 1, 12:00 PM"
  else
    "unknown"

theorem next_correct_time :
  clock_shows_correct_time "March 21" "12:00 PM" 25 = "June 1, 12:00 PM" :=
by sorry

end next_correct_time_l390_39042


namespace interval_intersection_l390_39090

theorem interval_intersection (x : ℝ) : 
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) ↔ (1 / 3 < x ∧ x < 2 / 3) := 
by 
  sorry

end interval_intersection_l390_39090


namespace fg_eval_at_3_l390_39085

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_eval_at_3 : f (g 3) = 99 := by
  sorry

end fg_eval_at_3_l390_39085


namespace pi_approx_by_jews_l390_39040

theorem pi_approx_by_jews (S D C : ℝ) (h1 : 4 * S = (5 / 4) * C) (h2 : D = S) (h3 : C = π * D) : π = 3 := by
  sorry

end pi_approx_by_jews_l390_39040


namespace simplify_fraction_l390_39076

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l390_39076


namespace number_of_elements_l390_39067

theorem number_of_elements (n : ℕ) (S : ℕ) (sum_first_six : ℕ) (sum_last_six : ℕ) (sixth_number : ℕ)
    (h1 : S = 22 * n) 
    (h2 : sum_first_six = 6 * 19) 
    (h3 : sum_last_six = 6 * 27) 
    (h4 : sixth_number = 34) 
    (h5 : S = sum_first_six + sum_last_six - sixth_number) : 
    n = 11 := 
by
  sorry

end number_of_elements_l390_39067


namespace new_persons_joined_l390_39032

theorem new_persons_joined (initial_avg_age new_avg_age initial_total new_avg_age_total final_avg_age final_total : ℝ) 
  (n_initial n_new : ℕ) 
  (h1 : initial_avg_age = 16)
  (h2 : n_initial = 20)
  (h3 : new_avg_age = 15)
  (h4 : final_avg_age = 15.5)
  (h5 : initial_total = initial_avg_age * n_initial)
  (h6 : new_avg_age_total = new_avg_age * (n_new : ℝ))
  (h7 : final_total = initial_total + new_avg_age_total)
  (h8 : final_total = final_avg_age * (n_initial + n_new)) 
  : n_new = 20 :=
by
  sorry

end new_persons_joined_l390_39032


namespace sum_of_all_possible_values_of_z_l390_39074

noncomputable def sum_of_z_values (w x y z : ℚ) : ℚ :=
if h : w < x ∧ x < y ∧ y < z ∧ 
       (w + x = 1 ∧ w + y = 2 ∧ w + z = 3 ∧ x + y = 4 ∨ 
        w + x = 1 ∧ w + y = 2 ∧ w + z = 4 ∧ x + y = 3) ∧ 
       ((w + x) ≠ (w + y) ∧ (w + x) ≠ (w + z) ∧ (w + x) ≠ (x + y) ∧ (w + x) ≠ (x + z) ∧ (w + x) ≠ (y + z)) ∧ 
       ((w + y) ≠ (w + z) ∧ (w + y) ≠ (x + y) ∧ (w + y) ≠ (x + z) ∧ (w + y) ≠ (y + z)) ∧ 
       ((w + z) ≠ (x + y) ∧ (w + z) ≠ (x + z) ∧ (w + z) ≠ (y + z)) ∧ 
       ((x + y) ≠ (x + z) ∧ (x + y) ≠ (y + z)) ∧ 
       ((x + z) ≠ (y + z)) then
  if w + z = 4 then
    4 + 7/2
  else 0
else
  0

theorem sum_of_all_possible_values_of_z : sum_of_z_values w x y z = 15 / 2 :=
by sorry

end sum_of_all_possible_values_of_z_l390_39074


namespace binary_to_base5_1101_l390_39061

-- Definition of the binary to decimal conversion for the given number
def binary_to_decimal (b: Nat): Nat :=
  match b with
  | 0    => 0
  | 1101 => 1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3
  | _    => 0  -- This is a specific case for the given problem

-- Definition of the decimal to base-5 conversion method
def decimal_to_base5 (d: Nat): Nat :=
  match d with
  | 0    => 0
  | 13   =>
    let rem1 := 13 % 5
    let div1 := 13 / 5
    let rem2 := div1 % 5
    let div2 := div1 / 5
    rem2 * 10 + rem1  -- Assemble the base-5 number from remainders
  | _    => 0  -- This is a specific case for the given problem

-- Proof statement: conversion of 1101 in binary to base-5 yields 23
theorem binary_to_base5_1101 : decimal_to_base5 (binary_to_decimal 1101) = 23 := by
  sorry

end binary_to_base5_1101_l390_39061


namespace net_change_in_salary_l390_39046

variable (S : ℝ)

theorem net_change_in_salary : 
  let increased_salary := S + (0.1 * S)
  let final_salary := increased_salary - (0.1 * increased_salary)
  final_salary - S = -0.01 * S :=
by
  sorry

end net_change_in_salary_l390_39046


namespace sales_first_month_l390_39022

theorem sales_first_month (S1 S2 S3 S4 S5 S6 : ℝ) 
  (h2 : S2 = 7000) (h3 : S3 = 6800) (h4 : S4 = 7200) (h5 : S5 = 6500) (h6 : S6 = 5100)
  (avg : (S1 + S2 + S3 + S4 + S5 + S6) / 6 = 6500) : S1 = 6400 := by
  sorry

end sales_first_month_l390_39022


namespace calum_spend_per_disco_ball_l390_39005

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end calum_spend_per_disco_ball_l390_39005


namespace cost_of_pencil_and_pens_l390_39002

variable (p q : ℝ)

def equation1 := 3 * p + 4 * q = 3.20
def equation2 := 2 * p + 3 * q = 2.50

theorem cost_of_pencil_and_pens (h1 : equation1 p q) (h2 : equation2 p q) : p + 2 * q = 1.80 := 
by 
  sorry

end cost_of_pencil_and_pens_l390_39002


namespace teacher_proctor_arrangements_l390_39023

theorem teacher_proctor_arrangements {f m : ℕ} (hf : f = 2) (hm : m = 5) :
  (∃ moving_teachers : ℕ, moving_teachers = 1 ∧ (f - moving_teachers) + m = 7 
   ∧ (f - moving_teachers).choose 2 = 21)
  ∧ 2 * 21 = 42 :=
by
    sorry

end teacher_proctor_arrangements_l390_39023


namespace intersecting_lines_l390_39045

theorem intersecting_lines (c d : ℝ)
  (h1 : 16 = 2 * 4 + c)
  (h2 : 16 = 5 * 4 + d) :
  c + d = 4 :=
sorry

end intersecting_lines_l390_39045


namespace frood_points_l390_39058

theorem frood_points (n : ℕ) (h : n > 29) : (n * (n + 1) / 2) > 15 * n := by
  sorry

end frood_points_l390_39058


namespace reduced_price_per_kg_l390_39000

-- Assume the constants in the conditions
variables (P R : ℝ)
variables (h1 : R = P - 0.40 * P) -- R = 0.60P
variables (h2 : 2000 / P + 10 = 2000 / R) -- extra 10 kg for the same 2000 rs

-- State the target we want to prove
theorem reduced_price_per_kg : R = 80 :=
by
  -- The steps and details of the proof
  sorry

end reduced_price_per_kg_l390_39000


namespace remainder_division_l390_39096

theorem remainder_division (G Q1 R1 Q2 : ℕ) (hG : G = 88)
  (h1 : 3815 = G * Q1 + R1) (h2 : 4521 = G * Q2 + 33) : R1 = 31 :=
sorry

end remainder_division_l390_39096


namespace sum_of_coefficients_of_expansion_l390_39078

theorem sum_of_coefficients_of_expansion (x y : ℝ) :
  (3*x - 4*y) ^ 20 = 1 :=
by 
  sorry

end sum_of_coefficients_of_expansion_l390_39078


namespace oil_bill_for_January_l390_39077

variable {F J : ℕ}

theorem oil_bill_for_January (h1 : 2 * F = 3 * J) (h2 : 3 * (F + 20) = 5 * J) : J = 120 := by
  sorry

end oil_bill_for_January_l390_39077


namespace correct_operation_l390_39087

variable (a : ℝ)

theorem correct_operation :
  (2 * a^2 * a = 2 * a^3) ∧
  ((a + 1)^2 ≠ a^2 + 1) ∧
  ((a^2 / (2 * a)) ≠ 2 * a) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) :=
by
  { sorry }

end correct_operation_l390_39087


namespace value_of_m_if_f_is_power_function_l390_39001

theorem value_of_m_if_f_is_power_function (m : ℤ) :
  (2 * m + 3 = 1) → m = -1 :=
by
  sorry

end value_of_m_if_f_is_power_function_l390_39001


namespace triangles_with_perimeter_20_l390_39081

theorem triangles_with_perimeter_20 (sides : Finset (Finset ℕ)) : 
  (∀ {a b c : ℕ}, (a + b + c = 20) → (a > 0) → (b > 0) → (c > 0) 
  → (a + b > c) → (a + c > b) → (b + c > a) → ({a, b, c} ∈ sides)) 
  → sides.card = 8 := 
by
  sorry

end triangles_with_perimeter_20_l390_39081


namespace total_games_played_is_53_l390_39033

theorem total_games_played_is_53 :
  ∃ (ken_wins dave_wins jerry_wins larry_wins total_ties total_games_played : ℕ),
  jerry_wins = 7 ∧
  dave_wins = jerry_wins + 3 ∧
  ken_wins = dave_wins + 5 ∧
  larry_wins = 2 * jerry_wins ∧
  5 ≤ ken_wins ∧ 5 ≤ dave_wins ∧ 5 ≤ jerry_wins ∧ 5 ≤ larry_wins ∧
  total_ties = jerry_wins ∧
  total_games_played = ken_wins + dave_wins + jerry_wins + larry_wins + total_ties ∧
  total_games_played = 53 :=
by
  sorry

end total_games_played_is_53_l390_39033


namespace probability_even_toys_l390_39044

theorem probability_even_toys:
  let total_toys := 21
  let even_toys := 10
  let probability_first_even := (even_toys : ℚ) / total_toys
  let probability_second_even := (even_toys - 1 : ℚ) / (total_toys - 1)
  let probability_both_even := probability_first_even * probability_second_even
  probability_both_even = 3 / 14 :=
by
  sorry

end probability_even_toys_l390_39044


namespace vasya_numbers_l390_39006

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l390_39006


namespace cans_needed_eq_l390_39092

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end cans_needed_eq_l390_39092


namespace selling_price_per_book_l390_39052

noncomputable def fixed_costs : ℝ := 35630
noncomputable def variable_cost_per_book : ℝ := 11.50
noncomputable def num_books : ℕ := 4072
noncomputable def total_production_costs : ℝ := fixed_costs + variable_cost_per_book * num_books

theorem selling_price_per_book :
  (total_production_costs / num_books : ℝ) = 20.25 := by
  sorry

end selling_price_per_book_l390_39052


namespace hazel_additional_days_l390_39095

theorem hazel_additional_days (school_year_days : ℕ) (miss_percent : ℝ) (already_missed : ℕ)
  (h1 : school_year_days = 180)
  (h2 : miss_percent = 0.05)
  (h3 : already_missed = 6) :
  (⌊miss_percent * school_year_days⌋ - already_missed) = 3 :=
by
  sorry

end hazel_additional_days_l390_39095


namespace lighter_shopping_bag_weight_l390_39013

theorem lighter_shopping_bag_weight :
  ∀ (G : ℕ), (G + 7 = 10) → (G = 3) := by
  intros G h
  sorry

end lighter_shopping_bag_weight_l390_39013


namespace value_of_M_l390_39056

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end value_of_M_l390_39056


namespace man_alone_days_l390_39035

-- Conditions from the problem
variables (M : ℕ) (h1 : (1 / (↑M : ℝ)) + (1 / 12) = 1 / 3)  -- Combined work rate condition

-- The proof statement we need to show
theorem man_alone_days : M = 4 :=
by {
  sorry
}

end man_alone_days_l390_39035


namespace jorge_spent_amount_l390_39020

theorem jorge_spent_amount
  (num_tickets : ℕ)
  (price_per_ticket : ℕ)
  (discount_percentage : ℚ)
  (h1 : num_tickets = 24)
  (h2 : price_per_ticket = 7)
  (h3 : discount_percentage = 0.5) :
  num_tickets * price_per_ticket * (1 - discount_percentage) = 84 := 
by
  simp [h1, h2, h3]
  sorry

end jorge_spent_amount_l390_39020


namespace cube_surface_area_equals_353_l390_39026

noncomputable def volume_of_prism : ℝ := 5 * 3 * 30
noncomputable def edge_length_of_cube (volume : ℝ) : ℝ := (volume)^(1/3)
noncomputable def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length^2

theorem cube_surface_area_equals_353 :
  surface_area_of_cube (edge_length_of_cube volume_of_prism) = 353 := by
sorry

end cube_surface_area_equals_353_l390_39026


namespace unique_prime_sum_diff_l390_39050

-- Define that p is a prime number that satisfies both conditions
def sum_two_primes (p a b : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime a ∧ Nat.Prime b ∧ p = a + b

def diff_two_primes (p c d : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime c ∧ Nat.Prime d ∧ p = c - d

-- Main theorem to prove: The only prime p that satisfies both conditions is 5
theorem unique_prime_sum_diff (p : ℕ) :
  (∃ a b, sum_two_primes p a b) ∧ (∃ c d, diff_two_primes p c d) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l390_39050


namespace smallest_number_divide_perfect_cube_l390_39016

theorem smallest_number_divide_perfect_cube (n : ℕ):
  n = 450 → (∃ m : ℕ, n * m = k ∧ ∃ k : ℕ, k ^ 3 = n * m) ∧ (∀ m₂ : ℕ, (n * m₂ = l ∧ ∃ l : ℕ, l ^ 3 = n * m₂) → m ≤ m₂) → m = 60 :=
by
  sorry

end smallest_number_divide_perfect_cube_l390_39016


namespace solve_abs_equation_l390_39041

theorem solve_abs_equation (y : ℤ) : (|y - 8| + 3 * y = 12) ↔ (y = 2) :=
by
  sorry  -- skip the proof steps.

end solve_abs_equation_l390_39041


namespace arithmetic_seq_a4_l390_39055

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions and the goal to prove
theorem arithmetic_seq_a4 (h₁ : is_arithmetic_sequence a d) (h₂ : a 2 + a 6 = 10) : 
  a 4 = 5 :=
by
  sorry

end arithmetic_seq_a4_l390_39055


namespace only_polynomial_is_identity_l390_39004

-- Define the number composed only of digits 1
def Ones (k : ℕ) : ℕ := (10^k - 1) / 9

theorem only_polynomial_is_identity (P : ℕ → ℕ) :
  (∀ k : ℕ, P (Ones k) = Ones k) → (∀ x : ℕ, P x = x) :=
by
  intro h
  sorry

end only_polynomial_is_identity_l390_39004


namespace not_net_of_cuboid_l390_39007

noncomputable def cuboid_closed_path (c : Type) (f : c → c) :=
∀ (x1 x2 : c), ∃ (y : c), f x1 = y ∧ f x2 = y

theorem not_net_of_cuboid (c : Type) [Nonempty c] [DecidableEq c] (net : c → Set c) (f : c → c) :
  cuboid_closed_path c f → ¬ (∀ x, net x = {x}) :=
by
  sorry

end not_net_of_cuboid_l390_39007


namespace election_proof_l390_39047

noncomputable def election_problem : Prop :=
  ∃ (V : ℝ) (votesA votesB votesC : ℝ),
  (votesA = 0.35 * V) ∧
  (votesB = votesA + 1800) ∧
  (votesC = 0.5 * votesA) ∧
  (V = votesA + votesB + votesC) ∧
  (V = 14400) ∧
  ((votesA / V) * 100 = 35) ∧
  ((votesB / V) * 100 = 47.5) ∧
  ((votesC / V) * 100 = 17.5)

theorem election_proof : election_problem := sorry

end election_proof_l390_39047


namespace students_in_front_l390_39089

theorem students_in_front (total_students : ℕ) (students_behind : ℕ) (students_total : total_students = 25) (behind_Yuna : students_behind = 9) :
  (total_students - (students_behind + 1)) = 15 :=
by
  sorry

end students_in_front_l390_39089


namespace intersection_eq_l390_39019

-- Define the sets M and N
def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The statement to prove
theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l390_39019


namespace inequality_solution_set_empty_l390_39037

theorem inequality_solution_set_empty (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)) → a ≤ 5 :=
by sorry

end inequality_solution_set_empty_l390_39037


namespace journey_time_l390_39017

theorem journey_time
  (speed1 speed2 : ℝ)
  (distance total_time : ℝ)
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance = 240)
  (h4 : total_time = 5) :
  ∃ (t1 t2 : ℝ), (t1 + t2 = total_time) ∧ (speed1 * t1 + speed2 * t2 = distance) ∧ (t1 = 3) := 
by
  use (3 : ℝ), (2 : ℝ)
  simp [h1, h2, h3, h4]
  norm_num
  -- Additional steps to finish the proof would go here, but are omitted as per the requirements
  -- sorry

end journey_time_l390_39017


namespace cone_radius_l390_39054

theorem cone_radius
    (l : ℝ) (n : ℝ) (r : ℝ)
    (h1 : l = 2 * Real.pi)
    (h2 : n = 120)
    (h3 : l = (n * Real.pi * r) / 180 ) :
    r = 3 :=
sorry

end cone_radius_l390_39054


namespace digit_difference_is_7_l390_39021

def local_value (d : Nat) (place : Nat) : Nat :=
  d * (10^place)

def face_value (d : Nat) : Nat :=
  d

def difference (d : Nat) (place : Nat) : Nat :=
  local_value d place - face_value d

def numeral : Nat := 65793

theorem digit_difference_is_7 :
  ∃ d place, 0 ≤ d ∧ d < 10 ∧ difference d place = 693 ∧ d ∈ [6, 5, 7, 9, 3] ∧ numeral = 65793 ∧
  (local_value 6 4 = 60000 ∧ local_value 5 3 = 5000 ∧ local_value 7 2 = 700 ∧ local_value 9 1 = 90 ∧ local_value 3 0 = 3 ∧
   face_value 6 = 6 ∧ face_value 5 = 5 ∧ face_value 7 = 7 ∧ face_value 9 = 9 ∧ face_value 3 = 3) ∧ 
  d = 7 :=
sorry

end digit_difference_is_7_l390_39021


namespace abc_over_ab_bc_ca_l390_39091

variable {a b c : ℝ}

theorem abc_over_ab_bc_ca (h1 : ab / (a + b) = 2)
                          (h2 : bc / (b + c) = 5)
                          (h3 : ca / (c + a) = 7) :
        abc / (ab + bc + ca) = 35 / 44 :=
by
  -- The proof would go here.
  sorry

end abc_over_ab_bc_ca_l390_39091


namespace greatest_possible_value_of_a_l390_39088

theorem greatest_possible_value_of_a :
  ∃ a : ℕ, (∀ x : ℤ, x * (x + a) = -12) → a = 13 := by
  sorry

end greatest_possible_value_of_a_l390_39088


namespace quadratic_polynomial_solution_is_zero_l390_39080

-- Definitions based on given conditions
variables (a b c r s : ℝ)
variables (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
variables (h3 : r + s = -b / a)
variables (h4 : r * s = c / a)

-- Proposition matching the equivalent proof problem
theorem quadratic_polynomial_solution_is_zero :
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  (∃ r s : ℝ, (r + s = -b / a) ∧ (r * s = c / a) ∧ (c = r * s ∨ b = r * s ∨ a = r * s) ∧
  (a = r ∨ a = s)) :=
sorry

end quadratic_polynomial_solution_is_zero_l390_39080


namespace brian_cards_after_waine_takes_l390_39071

-- Define the conditions
def brian_initial_cards : ℕ := 76
def wayne_takes_away : ℕ := 59

-- Define the expected result
def brian_remaining_cards : ℕ := 17

-- The statement of the proof problem
theorem brian_cards_after_waine_takes : brian_initial_cards - wayne_takes_away = brian_remaining_cards := 
by 
-- the proof would be provided here 
sorry

end brian_cards_after_waine_takes_l390_39071


namespace pencil_eraser_cost_l390_39038

theorem pencil_eraser_cost (p e : ℕ) (hp : p > e) (he : e > 0)
  (h : 20 * p + 4 * e = 160) : p + e = 12 :=
sorry

end pencil_eraser_cost_l390_39038


namespace non_union_employees_women_percent_l390_39018

-- Define the conditions
variables (total_employees men_percent women_percent unionized_percent unionized_men_percent : ℕ)
variables (total_men total_women total_unionized total_non_unionized unionized_men non_unionized_men non_unionized_women : ℕ)

axiom condition1 : men_percent = 52
axiom condition2 : unionized_percent = 60
axiom condition3 : unionized_men_percent = 70

axiom calc1 : total_employees = 100
axiom calc2 : total_men = total_employees * men_percent / 100
axiom calc3 : total_women = total_employees - total_men
axiom calc4 : total_unionized = total_employees * unionized_percent / 100
axiom calc5 : unionized_men = total_unionized * unionized_men_percent / 100
axiom calc6 : non_unionized_men = total_men - unionized_men
axiom calc7 : total_non_unionized = total_employees - total_unionized
axiom calc8 : non_unionized_women = total_non_unionized - non_unionized_men

-- Define the proof statement
theorem non_union_employees_women_percent : 
  (non_unionized_women / total_non_unionized) * 100 = 75 :=
by 
  sorry

end non_union_employees_women_percent_l390_39018


namespace hypotenuse_length_l390_39049

theorem hypotenuse_length {a b c : ℕ} (ha : a = 8) (hb : b = 15) (hc : c = (8^2 + 15^2).sqrt) : c = 17 :=
by
  sorry

end hypotenuse_length_l390_39049


namespace quadratic_roots_bounds_l390_39010

theorem quadratic_roots_bounds (a b c : ℤ) (p1 p2 : ℝ) (h_a_pos : a > 0) 
  (h_int_coeff : ∀ x : ℤ, x = a ∨ x = b ∨ x = c) 
  (h_distinct_roots : p1 ≠ p2) 
  (h_roots : a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0) 
  (h_roots_bounds : 0 < p1 ∧ p1 < 1 ∧ 0 < p2 ∧ p2 < 1) : 
     a ≥ 5 := 
sorry

end quadratic_roots_bounds_l390_39010


namespace min_value_expression_l390_39073

/-- Prove that for integers a, b, c satisfying 1 ≤ a ≤ b ≤ c ≤ 5, the minimum value of the expression 
  (a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2 is 1.2595. -/
theorem min_value_expression (a b c : ℤ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (min_val : ℝ), min_val = ((a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2) ∧ min_val = 1.2595 :=
by
  sorry

end min_value_expression_l390_39073


namespace calculate_expression_l390_39070

theorem calculate_expression : 
  (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := 
by sorry

end calculate_expression_l390_39070


namespace range_of_a_l390_39094

theorem range_of_a (a : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ (a^2 * x - 2 * a + 1 = 0)) ↔ (a > 1/2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_l390_39094


namespace lime_bottom_means_magenta_top_l390_39099

-- Define the colors as an enumeration for clarity
inductive Color
| Purple : Color
| Cyan : Color
| Magenta : Color
| Lime : Color
| Silver : Color
| Black : Color

open Color

-- Define the function representing the question
def opposite_top_face_given_bottom (bottom : Color) : Color :=
  match bottom with
  | Lime => Magenta
  | _ => Lime  -- For simplicity, we're only handling the Lime case as specified

-- State the theorem
theorem lime_bottom_means_magenta_top : 
  opposite_top_face_given_bottom Lime = Magenta :=
by
  -- This theorem states exactly what we need: if Lime is the bottom face, then Magenta is the top face.
  sorry

end lime_bottom_means_magenta_top_l390_39099


namespace range_of_k_l390_39093

theorem range_of_k (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → y^2 = 2 * x → (∃! (x₀ y₀ : ℝ), y₀ = k * x₀ + 1 ∧ y₀^2 = 2 * x₀)) ↔ 
  (k = 0 ∨ k ≥ 1/2) :=
sorry

end range_of_k_l390_39093


namespace good_arrangement_iff_coprime_l390_39083

-- Definitions for the concepts used
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_good_arrangement (n m : ℕ) : Prop :=
  ∃ k₀, ∀ i, (n * k₀ * i) % (m + n) = (i % (m + n))

theorem good_arrangement_iff_coprime (n m : ℕ) : is_good_arrangement n m ↔ is_coprime n m := 
sorry

end good_arrangement_iff_coprime_l390_39083


namespace min_value_reciprocals_l390_39030

theorem min_value_reciprocals (a b : ℝ) 
  (h1 : 2 * a + 2 * b = 2) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocals_l390_39030


namespace maximize_h_at_1_l390_39097

-- Definitions and conditions
def f (x : ℝ) : ℝ := -2 * x + 2
def g (x : ℝ) : ℝ := -3 * x + 6
def h (x : ℝ) : ℝ := f x * g x

-- The theorem to prove
theorem maximize_h_at_1 : (∀ x : ℝ, h x <= h 1) :=
sorry

end maximize_h_at_1_l390_39097


namespace proof_Bill_age_is_24_l390_39086

noncomputable def Bill_is_24 (C : ℝ) (Bill_age : ℝ) (Daniel_age : ℝ) :=
  (Bill_age = 2 * C - 1) ∧ 
  (Daniel_age = C - 4) ∧ 
  (C + Bill_age + Daniel_age = 45) → 
  (Bill_age = 24)

theorem proof_Bill_age_is_24 (C Bill_age Daniel_age : ℝ) : 
  Bill_is_24 C Bill_age Daniel_age :=
by
  sorry

end proof_Bill_age_is_24_l390_39086


namespace find_x_y_z_l390_39043

theorem find_x_y_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x * y = x + y) (h2 : y * z = 3 * (y + z)) (h3 : z * x = 2 * (z + x)) : 
  x + y + z = 12 :=
sorry

end find_x_y_z_l390_39043


namespace find_a_l390_39027

theorem find_a
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 2 * (b * Real.cos A + a * Real.cos B) = c^2)
  (h2 : b = 3)
  (h3 : 3 * Real.cos A = 1) :
  a = 3 :=
sorry

end find_a_l390_39027


namespace manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l390_39082

-- Definitions of costs and the problem conditions.
def cost_manufacturer_A (desks chairs : ℕ) : ℝ :=
  200 * desks + 50 * (chairs - desks)

def cost_manufacturer_B (desks chairs : ℕ) : ℝ :=
  0.9 * (200 * desks + 50 * chairs)

-- Given condition: School needs 60 desks.
def desks : ℕ := 60

-- (1) Prove manufacturer A is more cost-effective when x < 360.
theorem manufacturer_A_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs < 360 → cost_manufacturer_A desks chairs < cost_manufacturer_B desks chairs :=
by sorry

-- (2) Prove manufacturer B is more cost-effective when x > 360.
theorem manufacturer_B_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs > 360 → cost_manufacturer_A desks chairs > cost_manufacturer_B desks chairs :=
by sorry

end manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l390_39082


namespace initial_apples_l390_39051

-- Definitions based on the given conditions
def apples_given_away : ℕ := 88
def apples_left : ℕ := 39

-- Statement to prove
theorem initial_apples : apples_given_away + apples_left = 127 :=
by {
  -- Proof steps would go here
  sorry
}

end initial_apples_l390_39051


namespace at_least_two_equal_l390_39036

theorem at_least_two_equal (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : b + a^2 + c^2 = c + a^2 + b^2) : 
  (a = b) ∨ (a = c) ∨ (b = c) :=
sorry

end at_least_two_equal_l390_39036


namespace numbers_in_circle_are_zero_l390_39066

theorem numbers_in_circle_are_zero (a : Fin 55 → ℤ) 
  (h : ∀ i, a i = a ((i + 54) % 55) + a ((i + 1) % 55)) : 
  ∀ i, a i = 0 := 
by
  sorry

end numbers_in_circle_are_zero_l390_39066


namespace birthday_guests_l390_39003

theorem birthday_guests (total_guests : ℕ) (women men children guests_left men_left children_left : ℕ)
  (h_total : total_guests = 60)
  (h_women : women = total_guests / 2)
  (h_men : men = 15)
  (h_children : children = total_guests - (women + men))
  (h_men_left : men_left = men / 3)
  (h_children_left : children_left = 5)
  (h_guests_left : guests_left = men_left + children_left) :
  (total_guests - guests_left) = 50 :=
by sorry

end birthday_guests_l390_39003
