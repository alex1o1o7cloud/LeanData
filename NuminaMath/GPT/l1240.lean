import Mathlib

namespace max_value_of_a_l1240_124017

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (x + 1) * (1 + Real.log (x + 1)) - a * x

theorem max_value_of_a (a : ℤ) : 
  (∀ x : ℝ, x ≥ -1 → (a : ℝ) * x ≤ (x + 1) * (1 + Real.log (x + 1))) → a ≤ 3 := sorry

end max_value_of_a_l1240_124017


namespace sum_ends_in_zero_squares_end_same_digit_l1240_124072

theorem sum_ends_in_zero_squares_end_same_digit (a b : ℕ) (h : (a + b) % 10 = 0) : (a^2 % 10) = (b^2 % 10) := 
sorry

end sum_ends_in_zero_squares_end_same_digit_l1240_124072


namespace solve_system_equations_l1240_124029

variable (x y z : ℝ)

theorem solve_system_equations (h1 : 3 * x = 20 + (20 - x))
    (h2 : y = 2 * x - 5)
    (h3 : z = Real.sqrt (x + 4)) :
  x = 10 ∧ y = 15 ∧ z = Real.sqrt 14 :=
by
  sorry

end solve_system_equations_l1240_124029


namespace check_interval_of_quadratic_l1240_124081

theorem check_interval_of_quadratic (z : ℝ) : (z^2 - 40 * z + 344 ≤ 0) ↔ (20 - 2 * Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2 * Real.sqrt 14) :=
sorry

end check_interval_of_quadratic_l1240_124081


namespace a_plus_b_eq_neg1_l1240_124011

theorem a_plus_b_eq_neg1 (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : a + b = -1 :=
by
  sorry

end a_plus_b_eq_neg1_l1240_124011


namespace sum_of_inserted_numbers_in_arithmetic_sequence_l1240_124035

theorem sum_of_inserted_numbers_in_arithmetic_sequence :
  ∃ a2 a3 : ℤ, 2015 > a2 ∧ a2 > a3 ∧ a3 > 131 ∧ (2015 - a2) = (a2 - a3) ∧ (a2 - a3) = (a3 - 131) ∧ (a2 + a3) = 2146 := 
by
  sorry

end sum_of_inserted_numbers_in_arithmetic_sequence_l1240_124035


namespace mul_value_proof_l1240_124013

theorem mul_value_proof :
  ∃ x : ℝ, (8.9 - x = 3.1) ∧ ((x * 3.1) * 2.5 = 44.95) :=
by
  sorry

end mul_value_proof_l1240_124013


namespace solution_system_linear_eqns_l1240_124091

theorem solution_system_linear_eqns
    (a1 b1 c1 a2 b2 c2 : ℝ)
    (h1: a1 * 6 + b1 * 3 = c1)
    (h2: a2 * 6 + b2 * 3 = c2) :
    (4 * a1 * 22 + 3 * b1 * 33 = 11 * c1) ∧
    (4 * a2 * 22 + 3 * b2 * 33 = 11 * c2) :=
by
    sorry

end solution_system_linear_eqns_l1240_124091


namespace football_team_throwers_l1240_124061

theorem football_team_throwers
    (total_players : ℕ)
    (right_handed_players : ℕ)
    (one_third : ℚ)
    (number_throwers : ℕ)
    (number_non_throwers : ℕ)
    (right_handed_non_throwers : ℕ)
    (left_handed_non_throwers : ℕ)
    (h1 : total_players = 70)
    (h2 : right_handed_players = 63)
    (h3 : one_third = 1 / 3)
    (h4 : number_non_throwers = total_players - number_throwers)
    (h5 : right_handed_non_throwers = right_handed_players - number_throwers)
    (h6 : left_handed_non_throwers = one_third * number_non_throwers)
    (h7 : 2 * left_handed_non_throwers = right_handed_non_throwers)
    : number_throwers = 49 := 
by
  sorry

end football_team_throwers_l1240_124061


namespace water_force_on_dam_l1240_124097

-- Given conditions
def density : Real := 1000  -- kg/m^3
def gravity : Real := 10    -- m/s^2
def a : Real := 5.7         -- m
def b : Real := 9.0         -- m
def h : Real := 4.0         -- m

-- Prove that the force is 544000 N under the given conditions
theorem water_force_on_dam : ∃ (F : Real), F = 544000 :=
by
  sorry  -- proof goes here

end water_force_on_dam_l1240_124097


namespace factor_expression_l1240_124079

theorem factor_expression (y : ℝ) :
  5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) :=
by
  sorry

end factor_expression_l1240_124079


namespace cube_volume_l1240_124030

variable (V_sphere : ℝ)
variable (V_cube : ℝ)
variable (R : ℝ)
variable (a : ℝ)

theorem cube_volume (h1 : V_sphere = (32 / 3) * Real.pi)
    (h2 : V_sphere = (4 / 3) * Real.pi * R^3)
    (h3 : R = 2)
    (h4 : R = (Real.sqrt 3 / 2) * a)
    (h5 : a = 4 * Real.sqrt 3 / 3) :
    V_cube = (4 * Real.sqrt 3 / 3) ^ 3 :=
  by
    sorry

end cube_volume_l1240_124030


namespace proof_problem_l1240_124053

noncomputable def polar_to_cartesian_O1 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = 4 * Real.cos θ → (ρ^2 = 4 * ρ * Real.cos θ)

noncomputable def cartesian_O1 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = 4 * x → x^2 + y^2 - 4 * x = 0

noncomputable def polar_to_cartesian_O2 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = -4 * Real.sin θ → (ρ^2 = -4 * ρ * Real.sin θ)

noncomputable def cartesian_O2 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = -4 * y → x^2 + y^2 + 4 * y = 0

noncomputable def intersections_O1_O2 : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 + 4 * y = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)

noncomputable def line_through_intersections : Prop :=
  ∀ (x y : ℝ), ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)) → y = -x

theorem proof_problem : polar_to_cartesian_O1 ∧ cartesian_O1 ∧ polar_to_cartesian_O2 ∧ cartesian_O2 ∧ intersections_O1_O2 ∧ line_through_intersections :=
  sorry

end proof_problem_l1240_124053


namespace max_product_partition_l1240_124039

theorem max_product_partition (k n : ℕ) (hkn : k ≥ n) 
  (q r : ℕ) (hqr : k = n * q + r) (h_r : 0 ≤ r ∧ r < n) : 
  ∃ (F : ℕ → ℕ), F k = q^(n-r) * (q+1)^r :=
by
  sorry

end max_product_partition_l1240_124039


namespace student_rank_from_right_l1240_124015

theorem student_rank_from_right (n m : ℕ) (h1 : n = 8) (h2 : m = 20) : m - (n - 1) = 13 :=
by
  sorry

end student_rank_from_right_l1240_124015


namespace max_projection_area_l1240_124003

noncomputable def maxProjectionArea (a : ℝ) : ℝ :=
  if a > (Real.sqrt 3 / 3) ∧ a <= (Real.sqrt 3 / 2) then
    Real.sqrt 3 / 4
  else if a >= (Real.sqrt 3 / 2) then
    a / 2
  else 
    0  -- if the condition for a is not met, it's an edge case which shouldn't logically occur here

theorem max_projection_area (a : ℝ) (h1 : a > Real.sqrt 3 / 3) (h2 : a <= Real.sqrt 3 / 2 ∨ a >= Real.sqrt 3 / 2) :
  maxProjectionArea a = 
    if a > Real.sqrt 3 / 3 ∧ a <= Real.sqrt 3 / 2 then Real.sqrt 3 / 4
    else if a >= Real.sqrt 3 / 2 then a / 2
    else
      sorry :=
by sorry

end max_projection_area_l1240_124003


namespace proof_x_eq_y_l1240_124000

variable (x y z : ℝ)

theorem proof_x_eq_y (h1 : x = 6 - y) (h2 : z^2 = x * y - 9) : x = y := 
  sorry

end proof_x_eq_y_l1240_124000


namespace problem1_l1240_124067

theorem problem1 (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (Real.pi / 2 - α)^2 + 3 * Real.sin (α + Real.pi) * Real.sin (α + Real.pi / 2) = -1 :=
sorry

end problem1_l1240_124067


namespace addition_amount_first_trial_l1240_124044

theorem addition_amount_first_trial :
  ∀ (a b : ℝ),
  20 ≤ a ∧ a ≤ 30 ∧ 20 ≤ b ∧ b ≤ 30 → (a = 20 + (30 - 20) * 0.618 ∨ b = 30 - (30 - 20) * 0.618) :=
by {
  sorry
}

end addition_amount_first_trial_l1240_124044


namespace completing_square_correct_l1240_124058

-- Define the initial equation
def eq1 : Prop := ∀ x : ℝ, x^2 - 4*x - 1 = 0

-- Define the condition after moving the constant term
def eq2 : Prop := ∀ x : ℝ, x^2 - 4*x = 1

-- Define the condition after adding 4 to both sides
def eq3 : Prop := ∀ x : ℝ, x^2 - 4*x + 4 = 5

-- Define the final transformed equation
def final_eq : Prop := ∀ x : ℝ, (x - 2)^2 = 5

-- State the theorem
theorem completing_square_correct : 
  (eq1 → eq2) ∧ 
  (eq2 → eq3) ∧ 
  (eq3 → final_eq) :=
by
  sorry

end completing_square_correct_l1240_124058


namespace smallest_number_is_111111_2_l1240_124023

def base9_to_decimal (n : Nat) : Nat :=
  (n / 10) * 9 + (n % 10)

def base6_to_decimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n % 100) / 10) * 6 + (n % 10)

def base4_to_decimal (n : Nat) : Nat :=
  (n / 1000) * 64

def base2_to_decimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n % 100000) / 10000) * 16 + ((n % 10000) / 1000) * 8 + ((n % 1000) / 100) * 4 + ((n % 100) / 10) * 2 + (n % 10)

theorem smallest_number_is_111111_2 :
  let n1 := base9_to_decimal 85
  let n2 := base6_to_decimal 210
  let n3 := base4_to_decimal 1000
  let n4 := base2_to_decimal 111111
  n4 < n1 ∧ n4 < n2 ∧ n4 < n3 := by
    sorry

end smallest_number_is_111111_2_l1240_124023


namespace find_C_given_eq_statement_max_area_triangle_statement_l1240_124073

open Real

noncomputable def find_C_given_eq (a b c A : ℝ) (C : ℝ) : Prop :=
  (2 * a = sqrt 3 * c * sin A - a * cos C) → 
  C = 2 * π / 3

noncomputable def max_area_triangle (a b c : ℝ) (C : ℝ) : Prop :=
  C = 2 * π / 3 →
  c = sqrt 3 →
  ∃ S, S = (sqrt 3 / 4) * a * b ∧ 
  ∀ a b : ℝ, a * b ≤ 1 → S = (sqrt 3 / 4)

-- Lean statements
theorem find_C_given_eq_statement (a b c A C : ℝ) : find_C_given_eq a b c A C := 
by sorry

theorem max_area_triangle_statement (a b c : ℝ) (C : ℝ) : max_area_triangle a b c C := 
by sorry

end find_C_given_eq_statement_max_area_triangle_statement_l1240_124073


namespace div_polynomial_not_div_l1240_124048

theorem div_polynomial_not_div (n : ℕ) : ¬ (n + 2) ∣ (n^3 - 2 * n^2 - 5 * n + 7) := by
  sorry

end div_polynomial_not_div_l1240_124048


namespace number_of_marbles_removed_and_replaced_l1240_124006

def bag_contains_red_marbles (r : ℕ) : Prop := r = 12
def total_marbles (t : ℕ) : Prop := t = 48
def probability_not_red_twice (r t : ℕ) : Prop := ((t - r) / t : ℝ) * ((t - r) / t) = 9 / 16

theorem number_of_marbles_removed_and_replaced (r t : ℕ)
  (hr : bag_contains_red_marbles r)
  (ht : total_marbles t)
  (hp : probability_not_red_twice r t) :
  2 = 2 := by
  sorry

end number_of_marbles_removed_and_replaced_l1240_124006


namespace prime_in_A_l1240_124060

def A (n : ℕ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ n = a^2 + 2 * b^2

theorem prime_in_A {p : ℕ} (h_prime : Nat.Prime p) (h_p2_in_A : A (p^2)) : A p :=
sorry

end prime_in_A_l1240_124060


namespace cost_of_graphing_calculator_l1240_124076

/-
  Everton college paid $1625 for an order of 45 calculators.
  Each scientific calculator costs $10.
  The order included 20 scientific calculators and 25 graphing calculators.
  We need to prove that each graphing calculator costs $57.
-/

namespace EvertonCollege

theorem cost_of_graphing_calculator
  (total_cost : ℕ)
  (cost_scientific : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (cost_graphing : ℕ)
  (h_order : total_cost = 1625)
  (h_cost_scientific : cost_scientific = 10)
  (h_num_scientific : num_scientific = 20)
  (h_num_graphing : num_graphing = 25)
  (h_total_calc : num_scientific + num_graphing = 45)
  (h_pay : total_cost = num_scientific * cost_scientific + num_graphing * cost_graphing) :
  cost_graphing = 57 :=
by
  sorry

end EvertonCollege

end cost_of_graphing_calculator_l1240_124076


namespace lines_intersect_at_l1240_124049

noncomputable def L₁ (t : ℝ) : ℝ × ℝ := (2 - t, -3 + 4 * t)
noncomputable def L₂ (u : ℝ) : ℝ × ℝ := (-1 + 5 * u, 6 - 7 * u)
noncomputable def point_of_intersection : ℝ × ℝ := (2 / 13, 69 / 13)

theorem lines_intersect_at :
  ∃ t u : ℝ, L₁ t = point_of_intersection ∧ L₂ u = point_of_intersection := 
sorry

end lines_intersect_at_l1240_124049


namespace ferris_wheel_capacity_l1240_124080

-- Define the conditions
def number_of_seats : ℕ := 14
def people_per_seat : ℕ := 6

-- Theorem to prove the total capacity is 84
theorem ferris_wheel_capacity : number_of_seats * people_per_seat = 84 := sorry

end ferris_wheel_capacity_l1240_124080


namespace symmetric_circle_l1240_124090

-- Define given circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 8 * y + 12 = 0

-- Define the line of symmetry
def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 5 = 0

-- Define the symmetric circle equation we need to prove
def symm_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- Lean 4 theorem statement
theorem symmetric_circle (x y : ℝ) :
  (∃ a b : ℝ, circle_equation 2 4 ∧ line_equation a b ∧ (a, b) = (0, 0)) →
  symm_circle_equation x y :=
by sorry

end symmetric_circle_l1240_124090


namespace emma_average_speed_l1240_124071

-- Define the given conditions
def distance1 : ℕ := 420     -- Distance traveled in the first segment
def time1 : ℕ := 7          -- Time taken in the first segment
def distance2 : ℕ := 480    -- Distance traveled in the second segment
def time2 : ℕ := 8          -- Time taken in the second segment

-- Define the total distance and total time
def total_distance : ℕ := distance1 + distance2
def total_time : ℕ := time1 + time2

-- Define the expected average speed
def expected_average_speed : ℕ := 60

-- Prove that the average speed is 60 miles per hour
theorem emma_average_speed : (total_distance / total_time) = expected_average_speed := by
  sorry

end emma_average_speed_l1240_124071


namespace find_x_if_perpendicular_l1240_124074

-- Define vectors a and b in the given conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x - 5, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The Lean theorem statement equivalent to the math problem
theorem find_x_if_perpendicular (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : x = 2 := by
  sorry

end find_x_if_perpendicular_l1240_124074


namespace first_group_number_l1240_124019

theorem first_group_number (x : ℕ) (h1 : x + 120 = 126) : x = 6 :=
by
  sorry

end first_group_number_l1240_124019


namespace new_pressure_of_transferred_gas_l1240_124064

theorem new_pressure_of_transferred_gas (V1 V2 : ℝ) (p1 k : ℝ) 
  (h1 : V1 = 3.5) (h2 : p1 = 8) (h3 : k = V1 * p1) (h4 : V2 = 7) :
  ∃ p2 : ℝ, p2 = 4 ∧ k = V2 * p2 :=
by
  use 4
  sorry

end new_pressure_of_transferred_gas_l1240_124064


namespace tuesday_rainfall_l1240_124045

-- Condition: average rainfall for the whole week is 3 cm
def avg_rainfall_week : ℝ := 3

-- Condition: number of days in a week
def days_in_week : ℕ := 7

-- Condition: total rainfall for the week
def total_rainfall_week : ℝ := avg_rainfall_week * days_in_week

-- Condition: total rainfall is twice the rainfall on Tuesday
def total_rainfall_equals_twice_T (T : ℝ) : ℝ := 2 * T

-- Theorem: Prove that the rainfall on Tuesday is 10.5 cm
theorem tuesday_rainfall : ∃ T : ℝ, total_rainfall_equals_twice_T T = total_rainfall_week ∧ T = 10.5 := by
  sorry

end tuesday_rainfall_l1240_124045


namespace general_term_formula_l1240_124010

def sequence_sum (n : ℕ) : ℕ := 3 * n^2 - 2 * n

def general_term (n : ℕ) : ℕ := if n = 0 then 0 else 6 * n - 5

theorem general_term_formula (n : ℕ) (h : n > 0) :
  general_term n = sequence_sum n - sequence_sum (n - 1) := by
  sorry

end general_term_formula_l1240_124010


namespace fraction_of_unoccupied_chairs_is_two_fifths_l1240_124062

noncomputable def fraction_unoccupied_chairs (total_chairs : ℕ) (chair_capacity : ℕ) (attended_board_members : ℕ) : ℚ :=
  let total_capacity := total_chairs * chair_capacity
  let total_board_members := total_capacity
  let unoccupied_members := total_board_members - attended_board_members
  let unoccupied_chairs := unoccupied_members / chair_capacity
  unoccupied_chairs / total_chairs

theorem fraction_of_unoccupied_chairs_is_two_fifths :
  fraction_unoccupied_chairs 40 2 48 = 2 / 5 :=
by
  sorry

end fraction_of_unoccupied_chairs_is_two_fifths_l1240_124062


namespace complex_number_properties_l1240_124036

theorem complex_number_properties (z : ℂ) (h : z^2 = 3 + 4 * Complex.I) : 
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_number_properties_l1240_124036


namespace find_initial_pomelos_l1240_124098

theorem find_initial_pomelos (g w w' g' : ℕ) 
  (h1 : w = 3 * g)
  (h2 : w' = w - 90)
  (h3 : g' = g - 60)
  (h4 : w' = 4 * g' - 26) 
  : g = 176 :=
by
  sorry

end find_initial_pomelos_l1240_124098


namespace average_price_per_dvd_l1240_124032

-- Define the conditions
def num_movies_box1 : ℕ := 10
def price_per_movie_box1 : ℕ := 2
def num_movies_box2 : ℕ := 5
def price_per_movie_box2 : ℕ := 5

-- Define total calculations based on conditions
def total_cost_box1 : ℕ := num_movies_box1 * price_per_movie_box1
def total_cost_box2 : ℕ := num_movies_box2 * price_per_movie_box2

def total_cost : ℕ := total_cost_box1 + total_cost_box2
def total_movies : ℕ := num_movies_box1 + num_movies_box2

-- Define the average price per DVD and prove it to be 3
theorem average_price_per_dvd : total_cost / total_movies = 3 := by
  sorry

end average_price_per_dvd_l1240_124032


namespace dilution_problem_l1240_124077

theorem dilution_problem
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (desired_concentration : ℝ)
  (initial_alcohol_content : initial_concentration * initial_volume / 100 = 4.8)
  (desired_alcohol_content : 4.8 = desired_concentration * (initial_volume + N) / 100)
  (N : ℝ) :
  N = 11.2 :=
sorry

end dilution_problem_l1240_124077


namespace find_moles_of_NaOH_l1240_124050

-- Define the conditions
def reaction (NaOH HClO4 NaClO4 H2O : ℕ) : Prop :=
  NaOH = HClO4 ∧ NaClO4 = HClO4 ∧ H2O = 1

def moles_of_HClO4 := 3
def moles_of_NaClO4 := 3

-- Problem statement
theorem find_moles_of_NaOH : ∃ (NaOH : ℕ), NaOH = moles_of_HClO4 ∧ moles_of_NaClO4 = 3 ∧ NaOH = 3 :=
by sorry

end find_moles_of_NaOH_l1240_124050


namespace moles_of_water_formed_l1240_124099

-- Definitions
def moles_of_H2SO4 : Nat := 3
def moles_of_NaOH : Nat := 3
def moles_of_NaHSO4 : Nat := 3
def moles_of_H2O := moles_of_NaHSO4

-- Theorem
theorem moles_of_water_formed :
  moles_of_H2SO4 = 3 →
  moles_of_NaOH = 3 →
  moles_of_NaHSO4 = 3 →
  moles_of_H2O = 3 :=
by
  intros h1 h2 h3
  rw [moles_of_H2O]
  exact h3

end moles_of_water_formed_l1240_124099


namespace pow_mod_l1240_124055

theorem pow_mod (h : 3^3 ≡ 1 [MOD 13]) : 3^21 ≡ 1 [MOD 13] :=
by
sorry

end pow_mod_l1240_124055


namespace sqrt_ab_eq_18_l1240_124085

noncomputable def a := Real.log 9 / Real.log 4
noncomputable def b := 108 * (Real.log 8 / Real.log 3)

theorem sqrt_ab_eq_18 : Real.sqrt (a * b) = 18 := by
  sorry

end sqrt_ab_eq_18_l1240_124085


namespace find_middle_part_length_l1240_124022

theorem find_middle_part_length (a b c : ℝ) 
  (h1 : a + b + c = 28) 
  (h2 : (a - 0.5 * a) + b + 0.5 * c = 16) :
  b = 4 :=
by
  sorry

end find_middle_part_length_l1240_124022


namespace initial_walnuts_l1240_124008

theorem initial_walnuts (W : ℕ) (boy_effective : ℕ) (girl_effective : ℕ) (total_walnuts : ℕ) :
  boy_effective = 5 → girl_effective = 3 → total_walnuts = 20 → W + boy_effective + girl_effective = total_walnuts → W = 12 :=
by
  intros h_boy h_girl h_total h_eq
  rw [h_boy, h_girl, h_total] at h_eq
  linarith

end initial_walnuts_l1240_124008


namespace sqrt_three_irrational_l1240_124089

theorem sqrt_three_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a:ℝ) / b = Real.sqrt 3 :=
sorry

end sqrt_three_irrational_l1240_124089


namespace find_value_l1240_124070

-- Definitions of the curve and the line
def curve (a b : ℝ) (P : ℝ × ℝ) : Prop := (P.1*P.1) / a - (P.2*P.2) / b = 1
def line (P : ℝ × ℝ) : Prop := P.1 + P.2 - 1 = 0

-- Definition of the dot product condition
def dot_product_zero (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

-- Theorem statement
theorem find_value (a b : ℝ) (P Q : ℝ × ℝ)
  (hc1 : curve a b P)
  (hc2 : curve a b Q)
  (hl1 : line P)
  (hl2 : line Q)
  (h_dot : dot_product_zero P Q) :
  1 / a - 1 / b = 2 :=
sorry

end find_value_l1240_124070


namespace fill_in_the_blank_l1240_124051

-- Definitions of the problem conditions
def parent := "being a parent"
def parent_with_special_needs := "being the parent of a child with special needs"

-- The sentence describing two situations of being a parent
def sentence1 := "Being a parent is not always easy"
def sentence2 := "being the parent of a child with special needs often carries with ___ extra stress."

-- The correct word to fill in the blank.
def correct_answer := "it"

-- Proof problem
theorem fill_in_the_blank : correct_answer = "it" :=
by
  sorry

end fill_in_the_blank_l1240_124051


namespace quadratic_inequality_solution_set_conclusions_l1240_124043

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set_conclusions (h1 : ∀ x, -1 ≤ x ∧ x ≤ 2 → ax^2 + bx + c ≥ 0)
(h2 : ∀ x, x < -1 ∨ x > 2 → ax^2 + bx + c < 0) :
(a + b = 0) ∧ (a + b + c > 0) ∧ (c > 0) ∧ ¬ (b < 0) := by
sorry

end quadratic_inequality_solution_set_conclusions_l1240_124043


namespace ab_value_l1240_124025

   variable (log2_3 : Real) (b : Real) (a : Real)

   -- Hypotheses
   def log_condition : Prop := log2_3 = 1
   def exp_condition (b : Real) : Prop := (4:Real) ^ b = 3
   
   -- Final statement to prove
   theorem ab_value (h_log2_3 : log_condition log2_3) (h_exp : exp_condition b) 
   (ha : a = 1) : a * b = 1 / 2 := sorry
   
end ab_value_l1240_124025


namespace percentage_of_third_number_l1240_124014

variable (T F S : ℝ)

-- Declare the conditions from step a)
def condition_one : Prop := S = 0.25 * T
def condition_two : Prop := F = 0.20 * S

-- Define the proof problem, proving that F is 5% of T given the conditions
theorem percentage_of_third_number
  (h1 : condition_one T S)
  (h2 : condition_two F S) :
  F = 0.05 * T := by
  sorry

end percentage_of_third_number_l1240_124014


namespace find_all_pairs_l1240_124016

def is_solution (m n : ℕ) : Prop := 200 * m + 6 * n = 2006

def valid_pairs : List (ℕ × ℕ) := [(1, 301), (4, 201), (7, 101), (10, 1)]

theorem find_all_pairs :
  ∀ (m n : ℕ), is_solution m n ↔ (m, n) ∈ valid_pairs := by sorry

end find_all_pairs_l1240_124016


namespace incorrect_comparison_tan_138_tan_143_l1240_124052

theorem incorrect_comparison_tan_138_tan_143 :
  ¬ (Real.tan (Real.pi * 138 / 180) > Real.tan (Real.pi * 143 / 180)) :=
by sorry

end incorrect_comparison_tan_138_tan_143_l1240_124052


namespace identify_incorrect_calculation_l1240_124033

theorem identify_incorrect_calculation : 
  (∀ x : ℝ, x^2 * x^3 = x^5) ∧ 
  (∀ x : ℝ, x^3 + x^3 = 2 * x^3) ∧ 
  (∀ x : ℝ, x^6 / x^2 = x^4) ∧ 
  ¬ (∀ x : ℝ, (-3 * x)^2 = 6 * x^2) := 
by
  sorry

end identify_incorrect_calculation_l1240_124033


namespace megatek_manufacturing_percentage_l1240_124027

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ) 
    (h1 : total_degrees = 360) 
    (h2 : manufacturing_degrees = 126) : 
    (manufacturing_degrees / total_degrees) * 100 = 35 := by
  sorry

end megatek_manufacturing_percentage_l1240_124027


namespace original_chairs_count_l1240_124002

theorem original_chairs_count (n : ℕ) (m : ℕ) :
  (∀ k : ℕ, (k % 4 = 0 → k * (2 * n / 4) = k * (3 * n / 4) ) ∧ 
  (m = (4 / 2) * 15) ∧ (n = (4 * m / (2 * m)) - ((2 * m) / m)) ∧ 
  n + (n + 9) = 72) → n = 63 :=
by
  sorry

end original_chairs_count_l1240_124002


namespace sum_of_coordinates_of_D_is_12_l1240_124087

theorem sum_of_coordinates_of_D_is_12 :
  (exists (x y : ℝ), (5 = (11 + x) / 2) ∧ (9 = (5 + y) / 2) ∧ (x + y = 12)) :=
by
  sorry

end sum_of_coordinates_of_D_is_12_l1240_124087


namespace product_of_real_solutions_l1240_124040

theorem product_of_real_solutions :
  (∀ x : ℝ, (x + 1) / (3 * x + 3) = (3 * x + 2) / (8 * x + 2)) →
  x = -1 ∨ x = -4 →
  (-1) * (-4) = 4 := 
sorry

end product_of_real_solutions_l1240_124040


namespace fraction_divisible_by_n_l1240_124020

theorem fraction_divisible_by_n (a b n : ℕ) (h1 : a ≠ b) (h2 : n > 0) (h3 : n ∣ (a^n - b^n)) : n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end fraction_divisible_by_n_l1240_124020


namespace largest_among_trig_expressions_l1240_124082

theorem largest_among_trig_expressions :
  let a := Real.tan 48 + 1 / Real.tan 48
  let b := Real.sin 48 + Real.cos 48
  let c := Real.tan 48 + Real.cos 48
  let d := 1 / Real.tan 48 + Real.sin 48
  a > b ∧ a > c ∧ a > d :=
by
  sorry

end largest_among_trig_expressions_l1240_124082


namespace speed_of_canoe_downstream_l1240_124046

-- Definition of the problem conditions
def speed_of_canoe_in_still_water (V_c : ℝ) (V_s : ℝ) (upstream_speed : ℝ) : Prop :=
  V_c - V_s = upstream_speed

def speed_of_stream (V_s : ℝ) : Prop :=
  V_s = 4

-- The statement we want to prove
theorem speed_of_canoe_downstream (V_c V_s : ℝ) (upstream_speed : ℝ) 
  (h1 : speed_of_canoe_in_still_water V_c V_s upstream_speed)
  (h2 : speed_of_stream V_s)
  (h3 : upstream_speed = 4) :
  V_c + V_s = 12 :=
by
  sorry

end speed_of_canoe_downstream_l1240_124046


namespace maximum_take_home_pay_l1240_124012

noncomputable def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - ((x + 10) / 100 * 1000 * x)

theorem maximum_take_home_pay : 
  ∃ x : ℝ, (take_home_pay x = 20250) ∧ (45000 = 1000 * x) :=
by
  sorry

end maximum_take_home_pay_l1240_124012


namespace third_day_sales_correct_l1240_124094

variable (a : ℕ)

def firstDaySales := a
def secondDaySales := a + 4
def thirdDaySales := 2 * (a + 4) - 7
def expectedSales := 2 * a + 1

theorem third_day_sales_correct : thirdDaySales a = expectedSales a :=
by
  -- Main proof goes here
  sorry

end third_day_sales_correct_l1240_124094


namespace min_rectangles_needed_l1240_124018

theorem min_rectangles_needed 
  (type1_corners type2_corners : ℕ)
  (rectangles_cover : ℕ → ℕ)
  (h1 : type1_corners = 12)
  (h2 : type2_corners = 12)
  (h3 : ∀ n, rectangles_cover (3 * n) = n) : 
  (rectangles_cover type2_corners) + (rectangles_cover type1_corners) = 12 := 
sorry

end min_rectangles_needed_l1240_124018


namespace percent_of_x_l1240_124069

variable (x : ℝ) (h : x > 0)

theorem percent_of_x (p : ℝ) : 
  (p * x = 0.21 * x + 10) → 
  p = 0.21 + 10 / x :=
sorry

end percent_of_x_l1240_124069


namespace geometric_sequence_ratio_l1240_124068

noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio 
  (S : ℕ → ℝ) 
  (hS12 : S 12 = 1)
  (hS6 : S 6 = 2)
  (geom_property : ∀ a r, (S n = a * (1 - r^n) / (1 - r))) :
  S 18 / S 6 = 3 / 4 :=
by
  sorry

end geometric_sequence_ratio_l1240_124068


namespace truffles_more_than_caramels_l1240_124059

-- Define the conditions
def chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def peanut_clusters := (64 * chocolates) / 100
def truffles := chocolates - (caramels + nougats + peanut_clusters)

-- Define the claim
theorem truffles_more_than_caramels : (truffles - caramels) = 6 := by
  sorry

end truffles_more_than_caramels_l1240_124059


namespace solve_fraction_l1240_124021

theorem solve_fraction (a b : ℝ) (hab : 3 * a = 2 * b) : (a + b) / b = 5 / 3 :=
by
  sorry

end solve_fraction_l1240_124021


namespace tangent_line_slope_angle_l1240_124093

theorem tangent_line_slope_angle (θ : ℝ) : 
  (∃ k : ℝ, (∀ x y, k * x - y = 0) ∧ ∀ x y, x^2 + y^2 - 4 * x + 3 = 0) →
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end tangent_line_slope_angle_l1240_124093


namespace percentage_neither_bp_nor_ht_l1240_124037

noncomputable def percentage_teachers_neither_condition (total: ℕ) (high_bp: ℕ) (heart_trouble: ℕ) (both: ℕ) : ℚ :=
  let either_condition := high_bp + heart_trouble - both
  let neither_condition := total - either_condition
  (neither_condition * 100 : ℚ) / total

theorem percentage_neither_bp_nor_ht :
  percentage_teachers_neither_condition 150 90 50 30 = 26.67 :=
by
  sorry

end percentage_neither_bp_nor_ht_l1240_124037


namespace statement_1_statement_2_statement_3_statement_4_l1240_124084

variables (a b c x0 : ℝ)
noncomputable def P (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Statement ①
theorem statement_1 (h : a - b + c = 0) : P a b c (-1) = 0 := sorry

-- Statement ②
theorem statement_2 (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + c = 0 ∧ a*x2^2 + c = 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 := sorry

-- Statement ③
theorem statement_3 (h : P a b c c = 0) : a*c + b + 1 = 0 := sorry

-- Statement ④
theorem statement_4 (h : P a b c x0 = 0) : b^2 - 4*a*c = (2*a*x0 + b)^2 := sorry

end statement_1_statement_2_statement_3_statement_4_l1240_124084


namespace profit_share_ratio_l1240_124096

theorem profit_share_ratio (P Q : ℝ) (hP : P = 40000) (hQ : Q = 60000) : P / Q = 2 / 3 :=
by
  rw [hP, hQ]
  norm_num

end profit_share_ratio_l1240_124096


namespace route_down_distance_l1240_124054

theorem route_down_distance
  (rate_up : ℕ)
  (time_up : ℕ)
  (rate_down_rate_factor : ℚ)
  (time_down : ℕ)
  (h1 : rate_up = 4)
  (h2 : time_up = 2)
  (h3 : rate_down_rate_factor = (3 / 2))
  (h4 : time_down = time_up) :
  rate_down_rate_factor * rate_up * time_up = 12 := 
by
  rw [h1, h2, h3]
  sorry

end route_down_distance_l1240_124054


namespace solution_set_of_inequality_l1240_124042

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_def : ∀ x : ℝ, x ≤ 0 → f x = x^2 + 2 * x) :
  {x : ℝ | f (x + 2) < 3} = {x : ℝ | -5 < x ∧ x < 1} :=
by sorry

end solution_set_of_inequality_l1240_124042


namespace count_non_integer_angles_l1240_124066

open Int

def interior_angle (n : ℕ) : ℕ := 180 * (n - 2) / n

def is_integer_angle (n : ℕ) : Prop := 180 * (n - 2) % n = 0

theorem count_non_integer_angles : ∃ (count : ℕ), count = 2 ∧ ∀ n, 3 ≤ n ∧ n < 12 → is_integer_angle n ↔ ¬ (count = count + 1) :=
sorry

end count_non_integer_angles_l1240_124066


namespace round_trip_time_l1240_124063

variable (dist : ℝ)
variable (speed_to_work : ℝ)
variable (speed_to_home : ℝ)

theorem round_trip_time (h_dist : dist = 24) (h_speed_to_work : speed_to_work = 60) (h_speed_to_home : speed_to_home = 40) :
    (dist / speed_to_work + dist / speed_to_home) = 1 := 
by 
  sorry

end round_trip_time_l1240_124063


namespace cakes_to_make_l1240_124065

-- Define the conditions
def packages_per_cake : ℕ := 2
def cost_per_package : ℕ := 3
def total_cost : ℕ := 12

-- Define the proof problem
theorem cakes_to_make (h1 : packages_per_cake = 2) (h2 : cost_per_package = 3) (h3 : total_cost = 12) :
  (total_cost / cost_per_package) / packages_per_cake = 2 :=
by sorry

end cakes_to_make_l1240_124065


namespace Robin_hair_initial_length_l1240_124057

theorem Robin_hair_initial_length (x : ℝ) (h1 : x + 8 - 20 = 2) : x = 14 :=
by
  sorry

end Robin_hair_initial_length_l1240_124057


namespace tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l1240_124047

theorem tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2 (alpha : ℝ) 
  (h1 : Real.sin alpha = - (Real.sqrt 3) / 2) 
  (h2 : 3 * π / 2 < alpha ∧ alpha < 2 * π) : 
  Real.tan alpha = - Real.sqrt 3 := 
by 
  sorry

end tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l1240_124047


namespace expansive_sequence_in_interval_l1240_124007

-- Definition of an expansive sequence
def expansive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (i j : ℕ), (i < j) → (|a i - a j| ≥ 1 / j)

-- Upper bound condition for C
def upper_bound_C (C : ℝ) : Prop :=
  C ≥ 2 * Real.log 2

-- The main statement combining both definitions into a proof problem
theorem expansive_sequence_in_interval (C : ℝ) (a : ℕ → ℝ) 
  (h_exp : expansive_sequence a) (h_bound : upper_bound_C C) :
  ∀ n, 0 ≤ a n ∧ a n ≤ C :=
sorry

end expansive_sequence_in_interval_l1240_124007


namespace cube_surface_area_correct_l1240_124028

noncomputable def total_surface_area_of_reassembled_cube : ℝ :=
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let top_bottom_area := 3 * 1 -- Each slab contributes 1 square foot for the top and bottom
  let side_area := 2 * 1 -- Each side slab contributes 1 square foot
  let front_back_area := 2 * 1 -- Each front and back contributes 1 square foot
  top_bottom_area + side_area + front_back_area

theorem cube_surface_area_correct :
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let total_surface_area := total_surface_area_of_reassembled_cube
  total_surface_area = 10 :=
by
  sorry

end cube_surface_area_correct_l1240_124028


namespace cost_per_piece_l1240_124026

-- Definitions based on the problem conditions
def total_cost : ℕ := 80         -- Total cost is $80
def num_pizzas : ℕ := 4          -- Luigi bought 4 pizzas
def pieces_per_pizza : ℕ := 5    -- Each pizza was cut into 5 pieces

-- Main theorem statement proving the cost per piece
theorem cost_per_piece :
  (total_cost / (num_pizzas * pieces_per_pizza)) = 4 :=
by
  sorry

end cost_per_piece_l1240_124026


namespace sin_cos_eq_l1240_124095

theorem sin_cos_eq (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := sorry

end sin_cos_eq_l1240_124095


namespace incorrect_statement_D_l1240_124078

theorem incorrect_statement_D 
  (population : Set ℕ)
  (time_spent_sample : ℕ → ℕ)
  (sample_size : ℕ)
  (individual : ℕ)
  (h1 : ∀ s, s ∈ population → s ≤ 24)
  (h2 : ∀ i, i < sample_size → population (time_spent_sample i))
  (h3 : sample_size = 300)
  (h4 : ∀ i, i < 300 → time_spent_sample i = individual):
  ¬ (∀ i, i < 300 → time_spent_sample i = individual) :=
sorry

end incorrect_statement_D_l1240_124078


namespace find_height_of_box_l1240_124009

-- Definitions for the problem conditions
def numCubes : ℕ := 24
def volumeCube : ℕ := 27
def lengthBox : ℕ := 8
def widthBox : ℕ := 9
def totalVolumeBox : ℕ := numCubes * volumeCube

-- Problem statement in Lean 4
theorem find_height_of_box : totalVolumeBox = lengthBox * widthBox * 9 :=
by sorry

end find_height_of_box_l1240_124009


namespace first_grade_muffins_total_l1240_124092

theorem first_grade_muffins_total :
  let muffins_brier : ℕ := 218
  let muffins_macadams : ℕ := 320
  let muffins_flannery : ℕ := 417
  let muffins_smith : ℕ := 292
  let muffins_jackson : ℕ := 389
  muffins_brier + muffins_macadams + muffins_flannery + muffins_smith + muffins_jackson = 1636 :=
by
  apply sorry

end first_grade_muffins_total_l1240_124092


namespace hannah_highest_score_l1240_124075

-- Definitions based on conditions
def total_questions : ℕ := 40
def wrong_questions : ℕ := 3
def correct_percent_student_1 : ℝ := 0.95

-- The Lean statement representing the proof problem
theorem hannah_highest_score :
  ∃ q : ℕ, (q > (total_questions - wrong_questions) ∧ q > (total_questions * correct_percent_student_1)) ∧ q = 39 :=
by
  sorry

end hannah_highest_score_l1240_124075


namespace solution_l1240_124088

noncomputable def prove_a_greater_than_3 : Prop :=
  ∀ (x : ℝ) (a : ℝ), (a > 0) → (|x - 2| + |x - 3| + |x - 4| < a) → a > 3

theorem solution : prove_a_greater_than_3 :=
by
  intros x a h_pos h_ineq
  sorry

end solution_l1240_124088


namespace total_units_is_34_l1240_124083

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end total_units_is_34_l1240_124083


namespace price_increase_equivalence_l1240_124041

theorem price_increase_equivalence (P : ℝ) : 
  let increase_35 := P * 1.35
  let increase_40 := increase_35 * 1.40
  let increase_20 := increase_40 * 1.20
  let final_increase := increase_20
  final_increase = P * 2.268 :=
by
  -- proof skipped
  sorry

end price_increase_equivalence_l1240_124041


namespace negation_equiv_l1240_124004

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 + 1 ≥ 1

-- Negation of the original proposition
def negated_prop : Prop := ∃ x : ℝ, x^2 + 1 < 1

-- Main theorem stating the equivalence
theorem negation_equiv :
  (¬ (∀ x : ℝ, original_prop x)) ↔ negated_prop :=
by sorry

end negation_equiv_l1240_124004


namespace geometric_sequence_17th_term_l1240_124056

variable {α : Type*} [Field α]

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

theorem geometric_sequence_17th_term :
  ∀ (a r : α),
    a * r ^ 4 = 9 →  -- Fifth term condition
    a * r ^ 12 = 1152 →  -- Thirteenth term condition
    a * r ^ 16 = 36864 :=  -- Seventeenth term conclusion
by
  intros a r h5 h13
  sorry

end geometric_sequence_17th_term_l1240_124056


namespace minimum_value_function_equality_holds_at_two_thirds_l1240_124024

noncomputable def f (x : ℝ) : ℝ := 4 / x + 1 / (1 - x)

theorem minimum_value_function (x : ℝ) (hx : 0 < x ∧ x < 1) : f x ≥ 9 := sorry

theorem equality_holds_at_two_thirds : f (2 / 3) = 9 := sorry

end minimum_value_function_equality_holds_at_two_thirds_l1240_124024


namespace division_multiplication_order_l1240_124038

theorem division_multiplication_order : 1100 / 25 * 4 / 11 = 16 := by
  sorry

end division_multiplication_order_l1240_124038


namespace group_sum_180_in_range_1_to_60_l1240_124031

def sum_of_arithmetic_series (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem group_sum_180_in_range_1_to_60 :
  ∃ (a n : ℕ), 1 ≤ a ∧ a + n - 1 ≤ 60 ∧ sum_of_arithmetic_series a 1 n = 180 :=
by
  sorry

end group_sum_180_in_range_1_to_60_l1240_124031


namespace percentage_increase_l1240_124005

theorem percentage_increase (P : ℝ) (h : 200 * (1 + P/100) * 0.70 = 182) : 
  P = 30 := 
sorry

end percentage_increase_l1240_124005


namespace solution_set_of_fx_eq_zero_l1240_124034

noncomputable def f (x : ℝ) : ℝ :=
if hx : x = 0 then 0 else if 0 < x then Real.log x / Real.log 2 else - (Real.log (-x) / Real.log 2)

lemma f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by sorry

lemma f_is_log_for_positive : ∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2 :=
by sorry

theorem solution_set_of_fx_eq_zero :
  {x : ℝ | f x = 0} = {-1, 0, 1} :=
by sorry

end solution_set_of_fx_eq_zero_l1240_124034


namespace min_sum_areas_of_triangles_l1240_124001

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1 / 4, 0)

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

def O := (0, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def on_opposite_sides_x_axis (p q : ℝ × ℝ) : Prop := p.2 * q.2 < 0

theorem min_sum_areas_of_triangles 
  (A B : ℝ × ℝ)
  (hA : parabola A)
  (hB : parabola B)
  (hAB : on_opposite_sides_x_axis A B)
  (h_dot : dot_product A B = 2) :
  ∃ m : ℝ, m = 3 := by
  sorry

end min_sum_areas_of_triangles_l1240_124001


namespace total_net_worth_after_2_years_l1240_124086

def initial_value : ℝ := 40000
def depreciation_rate : ℝ := 0.05
def initial_maintenance_cost : ℝ := 2000
def inflation_rate : ℝ := 0.03
def years : ℕ := 2

def value_at_end_of_year (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc _ => acc * (1 - rate)) initial_value (List.range years)

def cumulative_maintenance_cost (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc year => acc + initial_maintenance_cost * ((1 + inflation_rate) ^ year)) 0 (List.range years)

def total_net_worth (initial_value : ℝ) (depreciation_rate : ℝ) (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  value_at_end_of_year initial_value depreciation_rate years - cumulative_maintenance_cost initial_maintenance_cost inflation_rate years

theorem total_net_worth_after_2_years : total_net_worth initial_value depreciation_rate initial_maintenance_cost inflation_rate years = 32040 :=
  by
    sorry

end total_net_worth_after_2_years_l1240_124086
