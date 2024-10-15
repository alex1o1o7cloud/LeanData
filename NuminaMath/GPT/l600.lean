import Mathlib

namespace NUMINAMATH_GPT_balcony_height_l600_60021

-- Definitions for conditions given in the problem

def final_position := 0 -- y, since the ball hits the ground
def initial_velocity := 5 -- v₀ in m/s
def time_elapsed := 3 -- t in seconds
def gravity := 10 -- g in m/s²

theorem balcony_height : 
  ∃ h₀ : ℝ, final_position = h₀ + initial_velocity * time_elapsed - (1/2) * gravity * time_elapsed^2 ∧ h₀ = 30 := 
by 
  sorry

end NUMINAMATH_GPT_balcony_height_l600_60021


namespace NUMINAMATH_GPT_total_milks_taken_l600_60001

def total_milks (chocolateMilk strawberryMilk regularMilk : Nat) : Nat :=
  chocolateMilk + strawberryMilk + regularMilk

theorem total_milks_taken :
  total_milks 2 15 3 = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_milks_taken_l600_60001


namespace NUMINAMATH_GPT_function_inequality_l600_60058

noncomputable def f : ℝ → ℝ
| x => if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

theorem function_inequality : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end NUMINAMATH_GPT_function_inequality_l600_60058


namespace NUMINAMATH_GPT_hockey_league_games_l600_60051

def num_teams : ℕ := 18
def encounters_per_pair : ℕ := 10
def num_games (n : ℕ) (k : ℕ) : ℕ := (n * (n - 1)) / 2 * k

theorem hockey_league_games :
  num_games num_teams encounters_per_pair = 1530 :=
by
  sorry

end NUMINAMATH_GPT_hockey_league_games_l600_60051


namespace NUMINAMATH_GPT_center_of_circle_eq_l600_60071

theorem center_of_circle_eq {x y : ℝ} : (x - 2)^2 + (y - 3)^2 = 1 → (x, y) = (2, 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_center_of_circle_eq_l600_60071


namespace NUMINAMATH_GPT_alpha_value_l600_60060

theorem alpha_value (m : ℝ) (α : ℝ) (h : m * 8 ^ α = 1 / 4) : α = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l600_60060


namespace NUMINAMATH_GPT_parallelogram_area_l600_60087

noncomputable def angle_ABC : ℝ := 30
noncomputable def AX : ℝ := 20
noncomputable def CY : ℝ := 22

theorem parallelogram_area (angle_ABC_eq : angle_ABC = 30)
    (AX_eq : AX = 20)
    (CY_eq : CY = 22)
    : ∃ (BC : ℝ), (BC * AX = 880) := sorry

end NUMINAMATH_GPT_parallelogram_area_l600_60087


namespace NUMINAMATH_GPT_sum_remainder_l600_60075

theorem sum_remainder (n : ℕ) (h : n = 102) :
  ((n * (n + 1) / 2) % 5250) = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_l600_60075


namespace NUMINAMATH_GPT_sin_2B_value_l600_60065

-- Define the triangle's internal angles and the tangent of angles
variables (A B C : ℝ) 

-- Given conditions from the problem
def tan_sequence (tanA tanB tanC : ℝ) : Prop :=
  tanA = (1/2) * tanB ∧
  tanC = (3/2) * tanB ∧
  2 * tanB = tanC + tanB + (tanC - tanA)

-- The statement to be proven
theorem sin_2B_value (h : tan_sequence (Real.tan A) (Real.tan B) (Real.tan C)) :
  Real.sin (2 * B) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_sin_2B_value_l600_60065


namespace NUMINAMATH_GPT_curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l600_60019

noncomputable def curve (x y a : ℝ) : ℝ :=
  x^2 + y^2 - 2 * a * x + 2 * (a - 2) * y + 2 

theorem curve_trajectory_a_eq_1 :
  ∃! (x y : ℝ), curve x y 1 = 0 ∧ x = 1 ∧ y = 1 := by
  sorry

theorem curve_fixed_point_a_ne_1 (a : ℝ) (ha : a ≠ 1) :
  curve 1 1 a = 0 := by
  sorry

end NUMINAMATH_GPT_curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l600_60019


namespace NUMINAMATH_GPT_correct_number_of_conclusions_l600_60039

def y (x : ℝ) := -5 * x + 1

def conclusion1 := y (-1) = 5
def conclusion2 := ∃ x1 x2 x3 : ℝ, y x1 > 0 ∧ y x2 > 0 ∧ y (x3) < 0 ∧ (x1 < 0) ∧ (x2 > 0) ∧ (x3 < x2)
def conclusion3 := ∀ x : ℝ, x > 1 → y x < 0
def conclusion4 := ∀ x1 x2 : ℝ, x1 < x2 → y x1 < y x2

-- We want to prove that exactly 2 of these conclusions are correct
theorem correct_number_of_conclusions : (¬ conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬ conclusion4) :=
by
  sorry

end NUMINAMATH_GPT_correct_number_of_conclusions_l600_60039


namespace NUMINAMATH_GPT_triangle_third_side_l600_60037

theorem triangle_third_side (x : ℝ) (h1 : x > 2) (h2 : x < 6) : x = 5 :=
sorry

end NUMINAMATH_GPT_triangle_third_side_l600_60037


namespace NUMINAMATH_GPT_imaginary_part_of_z_l600_60003

open Complex -- open complex number functions

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) * (2 - I) = 5 * I) :
  z.im = 2 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l600_60003


namespace NUMINAMATH_GPT_remainder_of_1234567_div_123_l600_60068

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_1234567_div_123_l600_60068


namespace NUMINAMATH_GPT_john_has_22_dimes_l600_60084

theorem john_has_22_dimes (d q : ℕ) (h1 : d = q + 4) (h2 : 10 * d + 25 * q = 680) : d = 22 :=
by
sorry

end NUMINAMATH_GPT_john_has_22_dimes_l600_60084


namespace NUMINAMATH_GPT_total_students_l600_60053

-- Define the problem statement in Lean 4
theorem total_students (n : ℕ) (h1 : n < 400)
  (h2 : n % 17 = 15) (h3 : n % 19 = 10) : n = 219 :=
sorry

end NUMINAMATH_GPT_total_students_l600_60053


namespace NUMINAMATH_GPT_seashells_total_correct_l600_60064

-- Define the initial counts for Henry, John, and Adam.
def initial_seashells_Henry : ℕ := 11
def initial_seashells_John : ℕ := 24
def initial_seashells_Adam : ℕ := 17

-- Define the total initial seashells collected by all.
def total_initial_seashells : ℕ := 83

-- Calculate Leo's initial seashells.
def initial_seashells_Leo : ℕ := total_initial_seashells - (initial_seashells_Henry + initial_seashells_John + initial_seashells_Adam)

-- Define the changes occurred when they returned home.
def extra_seashells_Henry : ℕ := 3
def given_away_seashells_John : ℕ := 5
def percentage_given_away_Leo : ℕ := 40
def extra_seashells_Leo : ℕ := 5

-- Define the final number of seashells each person has.
def final_seashells_Henry : ℕ := initial_seashells_Henry + extra_seashells_Henry
def final_seashells_John : ℕ := initial_seashells_John - given_away_seashells_John
def given_away_seashells_Leo : ℕ := (initial_seashells_Leo * percentage_given_away_Leo) / 100
def final_seashells_Leo : ℕ := initial_seashells_Leo - given_away_seashells_Leo + extra_seashells_Leo
def final_seashells_Adam : ℕ := initial_seashells_Adam

-- Define the total number of seashells they have now.
def total_final_seashells : ℕ := final_seashells_Henry + final_seashells_John + final_seashells_Leo + final_seashells_Adam

-- Proposition that asserts the total number of seashells is 74.
theorem seashells_total_correct :
  total_final_seashells = 74 :=
sorry

end NUMINAMATH_GPT_seashells_total_correct_l600_60064


namespace NUMINAMATH_GPT_xyz_poly_identity_l600_60040

theorem xyz_poly_identity (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
  (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
  (x^6 + y^6 + z^6) / (xyz * (xy + xz + yz)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_xyz_poly_identity_l600_60040


namespace NUMINAMATH_GPT_disease_given_positive_l600_60033

-- Definitions and conditions extracted from the problem
def Pr_D : ℚ := 1 / 200
def Pr_Dc : ℚ := 1 - Pr_D
def Pr_T_D : ℚ := 1
def Pr_T_Dc : ℚ := 0.05

-- Derived probabilites from given conditions
def Pr_T : ℚ := Pr_T_D * Pr_D + Pr_T_Dc * Pr_Dc

-- Statement for the probability using Bayes' theorem
theorem disease_given_positive :
  (Pr_T_D * Pr_D) / Pr_T = 20 / 219 :=
sorry

end NUMINAMATH_GPT_disease_given_positive_l600_60033


namespace NUMINAMATH_GPT_arithmetic_sequence_example_l600_60069

theorem arithmetic_sequence_example 
    (a : ℕ → ℤ) 
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) 
    (h2 : a 1 + a 4 + a 7 = 45) 
    (h3 : a 2 + a 5 + a 8 = 39) :
    a 3 + a 6 + a 9 = 33 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_example_l600_60069


namespace NUMINAMATH_GPT_length_of_segment_P_to_P_l600_60074

/-- Point P is given as (-4, 3) and P' is the reflection of P over the x-axis. 
    We need to prove that the length of the segment connecting P to P' is 6. -/
theorem length_of_segment_P_to_P' :
  let P := (-4, 3)
  let P' := (-4, -3)
  dist P P' = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_P_to_P_l600_60074


namespace NUMINAMATH_GPT_smallest_integer_l600_60067

theorem smallest_integer (k : ℕ) : 
  (∀ (n : ℕ), n = 2^2 * 3^1 * 11^1 → 
  (∀ (f : ℕ), (f = 2^4 ∨ f = 3^3 ∨ f = 13^3) → f ∣ (n * k))) → 
  k = 79092 :=
  sorry

end NUMINAMATH_GPT_smallest_integer_l600_60067


namespace NUMINAMATH_GPT_frac_subtraction_simplified_l600_60008

-- Definitions of the fractions involved.
def frac1 : ℚ := 8 / 19
def frac2 : ℚ := 5 / 57

-- The primary goal is to prove the equality.
theorem frac_subtraction_simplified : frac1 - frac2 = 1 / 3 :=
by {
  -- Proof of the statement.
  sorry
}

end NUMINAMATH_GPT_frac_subtraction_simplified_l600_60008


namespace NUMINAMATH_GPT_no_naturals_satisfy_divisibility_condition_l600_60028

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end NUMINAMATH_GPT_no_naturals_satisfy_divisibility_condition_l600_60028


namespace NUMINAMATH_GPT_area_of_right_triangle_l600_60088

variable (a b : ℝ)

theorem area_of_right_triangle (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ (S : ℝ), S = a * b :=
sorry

end NUMINAMATH_GPT_area_of_right_triangle_l600_60088


namespace NUMINAMATH_GPT_decreased_revenue_l600_60016

variable (T C : ℝ)
def Revenue (tax consumption : ℝ) : ℝ := tax * consumption

theorem decreased_revenue (hT_new : T_new = 0.9 * T) (hC_new : C_new = 1.1 * C) :
  Revenue T_new C_new = 0.99 * (Revenue T C) := 
sorry

end NUMINAMATH_GPT_decreased_revenue_l600_60016


namespace NUMINAMATH_GPT_geometric_sequence_l600_60089

theorem geometric_sequence (a : ℝ) (h1 : a > 0)
  (h2 : ∃ r : ℝ, 210 * r = a ∧ a * r = 63 / 40) :
  a = 18.1875 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_l600_60089


namespace NUMINAMATH_GPT_cell_phones_sold_l600_60057

theorem cell_phones_sold (init_samsung init_iphone final_samsung final_iphone defective_samsung defective_iphone : ℕ)
    (h1 : init_samsung = 14) 
    (h2 : init_iphone = 8) 
    (h3 : final_samsung = 10) 
    (h4 : final_iphone = 5) 
    (h5 : defective_samsung = 2) 
    (h6 : defective_iphone = 1) : 
    init_samsung - defective_samsung - final_samsung + 
    init_iphone - defective_iphone - final_iphone = 4 := 
by
  sorry

end NUMINAMATH_GPT_cell_phones_sold_l600_60057


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l600_60002

variable {a₁ d : ℝ} (S : ℕ → ℝ)

axiom Sum_of_terms (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_problem
  (h : S 10 = 4 * S 5) :
  (a₁ / d) = 1 / 2 :=
by
  -- definitional expansion and algebraic simplification would proceed here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l600_60002


namespace NUMINAMATH_GPT_find_smallest_n_l600_60099

-- Define costs and relationships
def cost_red (r : ℕ) : ℕ := 10 * r
def cost_green (g : ℕ) : ℕ := 18 * g
def cost_blue (b : ℕ) : ℕ := 20 * b
def cost_purple (n : ℕ) : ℕ := 24 * n

-- Define the mathematical problem
theorem find_smallest_n (r g b : ℕ) :
  ∃ n : ℕ, 24 * n = Nat.lcm (cost_red r) (Nat.lcm (cost_green g) (cost_blue b)) ∧ n = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_n_l600_60099


namespace NUMINAMATH_GPT_volume_of_prism_l600_60093

-- Define the conditions
variables {a b c : ℝ}
-- Areas of the faces
def ab := 50
def ac := 72
def bc := 45

-- Theorem stating the volume of the prism
theorem volume_of_prism : a * b * c = 180 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l600_60093


namespace NUMINAMATH_GPT_initial_speed_l600_60018

variable (v : ℝ)
variable (h1 : (v / 2) + 2 * v = 75)

theorem initial_speed (v : ℝ) (h1 : (v / 2) + 2 * v = 75) : v = 30 :=
sorry

end NUMINAMATH_GPT_initial_speed_l600_60018


namespace NUMINAMATH_GPT_correct_operation_l600_60076

variable (a b : ℝ)

theorem correct_operation : 2 * (a - 1) = 2 * a - 2 :=
sorry

end NUMINAMATH_GPT_correct_operation_l600_60076


namespace NUMINAMATH_GPT_fraction_to_decimal_l600_60036

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end NUMINAMATH_GPT_fraction_to_decimal_l600_60036


namespace NUMINAMATH_GPT_total_production_l600_60025

variable (x : ℕ) -- total units produced by 4 machines in 6 days
variable (R : ℕ) -- rate of production per machine per day

-- Condition 1: 4 machines can produce x units in 6 days
axiom rate_definition : 4 * R * 6 = x

-- Question: Prove the total amount of product produced by 16 machines in 3 days is 2x
theorem total_production : 16 * R * 3 = 2 * x :=
by 
  sorry

end NUMINAMATH_GPT_total_production_l600_60025


namespace NUMINAMATH_GPT_number_of_girls_l600_60063

theorem number_of_girls (B G: ℕ) 
  (ratio : 8 * G = 5 * B) 
  (total : B + G = 780) :
  G = 300 := 
sorry

end NUMINAMATH_GPT_number_of_girls_l600_60063


namespace NUMINAMATH_GPT_factorize_expression_l600_60031

theorem factorize_expression (x y : ℝ) : 
  x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l600_60031


namespace NUMINAMATH_GPT_probability_at_least_two_defective_probability_at_most_one_defective_l600_60017

variable (P_no_defective : ℝ)
variable (P_one_defective : ℝ)
variable (P_two_defective : ℝ)
variable (P_all_defective : ℝ)

theorem probability_at_least_two_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_two_defective + P_all_defective = 0.29 :=
  by sorry

theorem probability_at_most_one_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_no_defective + P_one_defective = 0.71 :=
  by sorry

end NUMINAMATH_GPT_probability_at_least_two_defective_probability_at_most_one_defective_l600_60017


namespace NUMINAMATH_GPT_total_rainfall_l600_60042

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_l600_60042


namespace NUMINAMATH_GPT_square_measurement_error_l600_60023

theorem square_measurement_error (S S' : ℝ) (error_percentage : ℝ)
  (area_error_percentage : ℝ) (h1 : area_error_percentage = 2.01) :
  error_percentage = 1 :=
by
  sorry

end NUMINAMATH_GPT_square_measurement_error_l600_60023


namespace NUMINAMATH_GPT_jackie_free_time_correct_l600_60014

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end NUMINAMATH_GPT_jackie_free_time_correct_l600_60014


namespace NUMINAMATH_GPT_complement_U_M_l600_60000

theorem complement_U_M :
  let U := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  let M := {x : ℤ | ∃ k : ℤ, x = 4 * k}
  {x | x ∈ U ∧ x ∉ M} = {x : ℤ | ∃ k : ℤ, x = 4 * k - 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_M_l600_60000


namespace NUMINAMATH_GPT_trapezoid_area_is_correct_l600_60029

noncomputable def isosceles_trapezoid_area : ℝ :=
  let a : ℝ := 12
  let b : ℝ := 24 - 12 * Real.sqrt 2
  let h : ℝ := 6 * Real.sqrt 2
  (24 + b) / 2 * h

theorem trapezoid_area_is_correct :
  let a := 12
  let b := 24 - 12 * Real.sqrt 2
  let h := 6 * Real.sqrt 2
  (24 + b) / 2 * h = 144 * Real.sqrt 2 - 72 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_is_correct_l600_60029


namespace NUMINAMATH_GPT_teal_more_blue_l600_60009

theorem teal_more_blue (total : ℕ) (green : ℕ) (both_green_blue : ℕ) (neither_green_blue : ℕ)
  (h1 : total = 150) (h2 : green = 90) (h3 : both_green_blue = 40) (h4 : neither_green_blue = 25) :
  ∃ (blue : ℕ), blue = 75 :=
by
  sorry

end NUMINAMATH_GPT_teal_more_blue_l600_60009


namespace NUMINAMATH_GPT_cos_in_third_quadrant_l600_60072

theorem cos_in_third_quadrant (B : ℝ) (h_sin_B : Real.sin B = -5/13) (h_quadrant : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 :=
by
  sorry

end NUMINAMATH_GPT_cos_in_third_quadrant_l600_60072


namespace NUMINAMATH_GPT_james_muffins_baked_l600_60047

theorem james_muffins_baked (arthur_muffins : ℝ) (factor : ℝ) (h1 : arthur_muffins = 115.0) (h2 : factor = 12.0) :
  (arthur_muffins / factor) = 9.5833 :=
by 
  -- using the conditions given, we would proceed to prove the result:
  -- sorry is used to indicate that the proof is omitted here
  sorry

end NUMINAMATH_GPT_james_muffins_baked_l600_60047


namespace NUMINAMATH_GPT_pool_filling_water_amount_l600_60085

theorem pool_filling_water_amount (Tina_pail Tommy_pail Timmy_pail Trudy_pail : ℕ) 
  (h1 : Tina_pail = 4)
  (h2 : Tommy_pail = Tina_pail + 2)
  (h3 : Timmy_pail = 2 * Tommy_pail)
  (h4 : Trudy_pail = (3 * Timmy_pail) / 2)
  (Timmy_trips Trudy_trips Tommy_trips Tina_trips: ℕ)
  (h5 : Timmy_trips = 4)
  (h6 : Trudy_trips = 4)
  (h7 : Tommy_trips = 6)
  (h8 : Tina_trips = 6) :
  Timmy_trips * Timmy_pail + Trudy_trips * Trudy_pail + Tommy_trips * Tommy_pail + Tina_trips * Tina_pail = 180 := by
  sorry

end NUMINAMATH_GPT_pool_filling_water_amount_l600_60085


namespace NUMINAMATH_GPT_find_initial_red_balloons_l600_60030

-- Define the initial state of balloons and the assumption.
def initial_blue_balloons : ℕ := 4
def red_balloons_after_inflation (R : ℕ) : ℕ := R + 2
def blue_balloons_after_inflation : ℕ := initial_blue_balloons + 2
def total_balloons (R : ℕ) : ℕ := red_balloons_after_inflation R + blue_balloons_after_inflation

-- Define the likelihood condition.
def likelihood_red (R : ℕ) : Prop := (red_balloons_after_inflation R : ℚ) / (total_balloons R : ℚ) = 0.4

-- Statement of the problem.
theorem find_initial_red_balloons (R : ℕ) (h : likelihood_red R) : R = 2 := by
  sorry

end NUMINAMATH_GPT_find_initial_red_balloons_l600_60030


namespace NUMINAMATH_GPT_total_profit_amount_l600_60054

-- Definitions representing the conditions:
def ratio_condition (P_X P_Y : ℝ) : Prop :=
  P_X / P_Y = (1 / 2) / (1 / 3)

def difference_condition (P_X P_Y : ℝ) : Prop :=
  P_X - P_Y = 160

-- The proof problem statement:
theorem total_profit_amount (P_X P_Y : ℝ) (h1 : ratio_condition P_X P_Y) (h2 : difference_condition P_X P_Y) :
  P_X + P_Y = 800 := by
  sorry

end NUMINAMATH_GPT_total_profit_amount_l600_60054


namespace NUMINAMATH_GPT_added_number_is_five_l600_60070

def original_number := 19
def final_resultant := 129
def doubling_expression (x : ℕ) (y : ℕ) := 3 * (2 * x + y)

theorem added_number_is_five:
  ∃ y, doubling_expression original_number y = final_resultant ↔ y = 5 :=
sorry

end NUMINAMATH_GPT_added_number_is_five_l600_60070


namespace NUMINAMATH_GPT_youngest_person_age_l600_60011

theorem youngest_person_age (total_age_now : ℕ) (total_age_when_born : ℕ) (Y : ℕ) (h1 : total_age_now = 210) (h2 : total_age_when_born = 162) : Y = 48 :=
by
  sorry

end NUMINAMATH_GPT_youngest_person_age_l600_60011


namespace NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l600_60012

variable (a b : ℝ)

theorem sufficient_condition (hab : (a - b) * a^2 < 0) : a < b :=
by
  sorry

theorem not_necessary_condition (h : a < b) : (a - b) * a^2 < 0 ∨ (a - b) * a^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l600_60012


namespace NUMINAMATH_GPT_sharon_trip_distance_l600_60097

noncomputable def usual_speed (x : ℝ) : ℝ := x / 180
noncomputable def reduced_speed (x : ℝ) : ℝ := usual_speed x - 25 / 60
noncomputable def increased_speed (x : ℝ) : ℝ := usual_speed x + 10 / 60
noncomputable def pre_storm_time : ℝ := 60
noncomputable def total_time : ℝ := 300

theorem sharon_trip_distance : 
  ∀ (x : ℝ), 
  60 + (x / 3) / reduced_speed x + (x / 3) / increased_speed x = 240 → 
  x = 135 :=
sorry

end NUMINAMATH_GPT_sharon_trip_distance_l600_60097


namespace NUMINAMATH_GPT_smallest_square_contains_five_disks_l600_60081

noncomputable def smallest_side_length := 2 + 2 * Real.sqrt 2

theorem smallest_square_contains_five_disks :
  ∃ (a : ℝ), a = smallest_side_length ∧ (∃ (d : ℕ → ℝ × ℝ), 
    (∀ i, 0 ≤ i ∧ i < 5 → (d i).fst ^ 2 + (d i).snd ^ 2 < (a / 2 - 1) ^ 2) ∧ 
    (∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 ∧ i ≠ j → 
      (d i).fst ^ 2 + (d i).snd ^ 2 + (d j).fst ^ 2 + (d j).snd ^ 2 ≥ 4)) :=
sorry

end NUMINAMATH_GPT_smallest_square_contains_five_disks_l600_60081


namespace NUMINAMATH_GPT_no_adjacent_standing_prob_l600_60094

def coin_flip_probability : ℚ :=
  let a2 := 3
  let a3 := 4
  let a4 := a3 + a2
  let a5 := a4 + a3
  let a6 := a5 + a4
  let a7 := a6 + a5
  let a8 := a7 + a6
  let a9 := a8 + a7
  let a10 := a9 + a8
  let favorable_outcomes := a10
  favorable_outcomes / (2 ^ 10)

theorem no_adjacent_standing_prob :
  coin_flip_probability = (123 / 1024 : ℚ) :=
by sorry

end NUMINAMATH_GPT_no_adjacent_standing_prob_l600_60094


namespace NUMINAMATH_GPT_stratified_sampling_l600_60026

-- Conditions
def total_students : ℕ := 1200
def freshmen : ℕ := 300
def sophomores : ℕ := 400
def juniors : ℕ := 500
def sample_size : ℕ := 60
def probability : ℚ := sample_size / total_students

-- Number of students to be sampled from each grade
def freshmen_sampled : ℚ := freshmen * probability
def sophomores_sampled : ℚ := sophomores * probability
def juniors_sampled : ℚ := juniors * probability

-- Theorem to prove
theorem stratified_sampling :
  freshmen_sampled = 15 ∧ sophomores_sampled = 20 ∧ juniors_sampled = 25 :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_stratified_sampling_l600_60026


namespace NUMINAMATH_GPT_joshua_skittles_l600_60038

theorem joshua_skittles (eggs : ℝ) (skittles_per_friend : ℝ) (friends : ℝ) (h1 : eggs = 6.0) (h2 : skittles_per_friend = 40.0) (h3 : friends = 5.0) : skittles_per_friend * friends = 200.0 := 
by 
  sorry

end NUMINAMATH_GPT_joshua_skittles_l600_60038


namespace NUMINAMATH_GPT_math_expression_evaluation_l600_60095

theorem math_expression_evaluation :
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - (1/2)⁻¹ + (3 - Real.pi)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_math_expression_evaluation_l600_60095


namespace NUMINAMATH_GPT_y_completes_work_in_seventy_days_l600_60020

def work_days (mahesh_days : ℕ) (mahesh_work_days : ℕ) (rajesh_days : ℕ) (y_days : ℕ) : Prop :=
  let mahesh_rate := (1:ℝ) / mahesh_days
  let rajesh_rate := (1:ℝ) / rajesh_days
  let work_done_by_mahesh := mahesh_rate * mahesh_work_days
  let remaining_work := (1:ℝ) - work_done_by_mahesh
  let rajesh_remaining_work_days := remaining_work / rajesh_rate
  let y_rate := (1:ℝ) / y_days
  y_rate = rajesh_rate

theorem y_completes_work_in_seventy_days :
  work_days 35 20 30 70 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_y_completes_work_in_seventy_days_l600_60020


namespace NUMINAMATH_GPT_interval_length_correct_l600_60032

def sin_log_interval_sum : ℝ := sorry

theorem interval_length_correct :
  sin_log_interval_sum = 2^π / (1 + 2^π) :=
by
  -- Definitions
  let is_valid_x (x : ℝ) := x < 1 ∧ x > 0 ∧ (Real.sin (Real.log x / Real.log 2)) < 0
  
  -- Assertion
  sorry

end NUMINAMATH_GPT_interval_length_correct_l600_60032


namespace NUMINAMATH_GPT_simplify_sqrt_expr_l600_60073

-- We need to prove that simplifying √(5 - 2√6) is equal to √3 - √2.
theorem simplify_sqrt_expr : 
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expr_l600_60073


namespace NUMINAMATH_GPT_days_before_reinforcement_l600_60006

theorem days_before_reinforcement
    (garrison_1 : ℕ)
    (initial_days : ℕ)
    (reinforcement : ℕ)
    (additional_days : ℕ)
    (total_men_after_reinforcement : ℕ)
    (man_days_initial : ℕ)
    (man_days_after : ℕ)
    (x : ℕ) :
    garrison_1 * (initial_days - x) = total_men_after_reinforcement * additional_days →
    garrison_1 = 2000 →
    initial_days = 54 →
    reinforcement = 1600 →
    additional_days = 20 →
    total_men_after_reinforcement = garrison_1 + reinforcement →
    man_days_initial = garrison_1 * initial_days →
    man_days_after = total_men_after_reinforcement * additional_days →
    x = 18 :=
by
  intros h_eq g_1 i_days r_f a_days total_men m_days_i m_days_a
  sorry

end NUMINAMATH_GPT_days_before_reinforcement_l600_60006


namespace NUMINAMATH_GPT_Maurice_current_age_l600_60066

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end NUMINAMATH_GPT_Maurice_current_age_l600_60066


namespace NUMINAMATH_GPT_proportion_option_B_true_l600_60048

theorem proportion_option_B_true {a b c d : ℚ} (h : a / b = c / d) : 
  (a + c) / c = (b + d) / d := 
by 
  sorry

end NUMINAMATH_GPT_proportion_option_B_true_l600_60048


namespace NUMINAMATH_GPT_avg_speed_in_mph_l600_60086

/-- 
Given conditions:
1. The man travels 10,000 feet due north.
2. He travels 6,000 feet due east in 1/4 less time than he took heading north, traveling at 3 miles per minute.
3. He returns to his starting point by traveling south at 1 mile per minute.
4. He travels back west at the same speed as he went east.
We aim to prove that the average speed for the entire trip is 22.71 miles per hour.
-/
theorem avg_speed_in_mph :
  let distance_north_feet := 10000
  let distance_east_feet := 6000
  let speed_east_miles_per_minute := 3
  let speed_south_miles_per_minute := 1
  let feet_per_mile := 5280
  let distance_north_mil := (distance_north_feet / feet_per_mile : ℝ)
  let distance_east_mil := (distance_east_feet / feet_per_mile : ℝ)
  let time_north_min := distance_north_mil / (1 / 3)
  let time_east_min := time_north_min * 0.75
  let time_south_min := distance_north_mil / speed_south_miles_per_minute
  let time_west_min := time_east_min
  let total_time_hr := (time_north_min + time_east_min + time_south_min + time_west_min) / 60
  let total_distance_miles := 2 * (distance_north_mil + distance_east_mil)
  let avg_speed_mph := total_distance_miles / total_time_hr
  avg_speed_mph = 22.71 := by
sorry

end NUMINAMATH_GPT_avg_speed_in_mph_l600_60086


namespace NUMINAMATH_GPT_ratio_problem_l600_60007

theorem ratio_problem
  (w x y z : ℝ)
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 2 / 3)
  (h3 : w / z = 3 / 5) :
  (x + y) / z = 27 / 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l600_60007


namespace NUMINAMATH_GPT_pq_identity_l600_60098

theorem pq_identity (p q : ℝ) (h1 : p * q = 20) (h2 : p + q = 10) : p^2 + q^2 = 60 :=
sorry

end NUMINAMATH_GPT_pq_identity_l600_60098


namespace NUMINAMATH_GPT_ajay_walks_distance_l600_60034

theorem ajay_walks_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h_speed : speed = 3) 
  (h_time : time = 16.666666666666668) : 
  distance = speed * time :=
by
  sorry

end NUMINAMATH_GPT_ajay_walks_distance_l600_60034


namespace NUMINAMATH_GPT_rectangle_width_squared_l600_60083

theorem rectangle_width_squared (w l : ℝ) (h1 : w^2 + l^2 = 400) (h2 : 4 * w^2 + l^2 = 484) : w^2 = 28 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_width_squared_l600_60083


namespace NUMINAMATH_GPT_plane_divides_space_into_two_parts_l600_60078

def divides_space : Prop :=
  ∀ (P : ℝ → ℝ → ℝ → Prop), (∀ x y z, P x y z → P x y z) →
  (∃ region1 region2 : ℝ → ℝ → ℝ → Prop,
    (∀ x y z, P x y z → (region1 x y z ∨ region2 x y z)) ∧
    (∀ x y z, region1 x y z → ¬region2 x y z) ∧
    (∃ x1 y1 z1 x2 y2 z2, region1 x1 y1 z1 ∧ region2 x2 y2 z2))

theorem plane_divides_space_into_two_parts (P : ℝ → ℝ → ℝ → Prop) (hP : ∀ x y z, P x y z → P x y z) : 
  divides_space :=
  sorry

end NUMINAMATH_GPT_plane_divides_space_into_two_parts_l600_60078


namespace NUMINAMATH_GPT_term_is_18_minimum_value_l600_60080

-- Define the sequence a_n
def a_n (n : ℕ) : ℤ := n^2 - 5 * n + 4

-- Prove that a_n = 18 implies n = 7
theorem term_is_18 (n : ℕ) (h : a_n n = 18) : n = 7 := 
by 
  sorry

-- Prove that the minimum value of a_n is -2 and it occurs at n = 2 or n = 3
theorem minimum_value (n : ℕ) : n = 2 ∨ n = 3 ∧ a_n n = -2 :=
by 
  sorry

end NUMINAMATH_GPT_term_is_18_minimum_value_l600_60080


namespace NUMINAMATH_GPT_solve_for_y_l600_60091

theorem solve_for_y : ∃ y : ℕ, 8^4 = 2^y ∧ y = 12 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l600_60091


namespace NUMINAMATH_GPT_total_area_of_rug_l600_60024

theorem total_area_of_rug :
  let length_rect := 6
  let width_rect := 4
  let base_parallelogram := 3
  let height_parallelogram := 4
  let area_rect := length_rect * width_rect
  let area_parallelogram := base_parallelogram * height_parallelogram
  let total_area := area_rect + 2 * area_parallelogram
  total_area = 48 := by sorry

end NUMINAMATH_GPT_total_area_of_rug_l600_60024


namespace NUMINAMATH_GPT_remainder_y_div_13_l600_60061

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end NUMINAMATH_GPT_remainder_y_div_13_l600_60061


namespace NUMINAMATH_GPT_lunch_customers_is_127_l600_60044

-- Define the conditions based on the given problem
def breakfast_customers : ℕ := 73
def dinner_customers : ℕ := 87
def total_customers_on_saturday : ℕ := 574
def total_customers_on_friday : ℕ := total_customers_on_saturday / 2

-- Define the variable representing the lunch customers
variable (L : ℕ)

-- State the proposition we want to prove
theorem lunch_customers_is_127 :
  breakfast_customers + L + dinner_customers = total_customers_on_friday → L = 127 := by {
  sorry
}

end NUMINAMATH_GPT_lunch_customers_is_127_l600_60044


namespace NUMINAMATH_GPT_ratio_of_areas_of_triangles_l600_60045

-- Define the given conditions
variables {X Y Z T : Type}
variable (distance_XY : ℝ)
variable (distance_XZ : ℝ)
variable (distance_YZ : ℝ)
variable (is_angle_bisector : Prop)

-- Define the correct answer as a goal
theorem ratio_of_areas_of_triangles (h1 : distance_XY = 15)
    (h2 : distance_XZ = 25)
    (h3 : distance_YZ = 34)
    (h4 : is_angle_bisector) : 
    -- Ratio of the areas of triangle XYT to triangle XZT
    ∃ (ratio : ℝ), ratio = 3 / 5 :=
by
  -- This is where the proof would go, omitted with "sorry"
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_triangles_l600_60045


namespace NUMINAMATH_GPT_Sandy_phone_bill_expense_l600_60005
noncomputable def Sandy_age_now : ℕ := 34
noncomputable def Kim_age_now : ℕ := 10
noncomputable def Sandy_phone_bill : ℕ := 10 * Sandy_age_now

theorem Sandy_phone_bill_expense :
  (Sandy_age_now - 2 = 36 - 2) ∧ (Kim_age_now + 2 = 12) ∧ (36 = 3 * 12) ∧ (Sandy_phone_bill = 340) := by
sorry

end NUMINAMATH_GPT_Sandy_phone_bill_expense_l600_60005


namespace NUMINAMATH_GPT_max_value_of_y_l600_60059

noncomputable def max_y (x y : ℝ) : ℝ :=
  if h : x^2 + y^2 = 20*x + 54*y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  max_y x y ≤ 27 + Real.sqrt 829 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l600_60059


namespace NUMINAMATH_GPT_mountain_bike_cost_l600_60046

theorem mountain_bike_cost (savings : ℕ) (lawns : ℕ) (lawn_rate : ℕ) (newspapers : ℕ) (paper_rate : ℕ) (dogs : ℕ) (dog_rate : ℕ) (remaining : ℕ) (total_earned : ℕ) (total_before_purchase : ℕ) (cost : ℕ) : 
  savings = 1500 ∧ lawns = 20 ∧ lawn_rate = 20 ∧ newspapers = 600 ∧ paper_rate = 40 ∧ dogs = 24 ∧ dog_rate = 15 ∧ remaining = 155 ∧ 
  total_earned = (lawns * lawn_rate) + (newspapers * paper_rate / 100) + (dogs * dog_rate) ∧
  total_before_purchase = savings + total_earned ∧
  cost = total_before_purchase - remaining →
  cost = 2345 := by
  sorry

end NUMINAMATH_GPT_mountain_bike_cost_l600_60046


namespace NUMINAMATH_GPT_remainder_n_l600_60055

-- Definitions for the conditions
/-- m is a positive integer leaving a remainder of 2 when divided by 6 -/
def m (m : ℕ) : Prop := m % 6 = 2

/-- The remainder when m - n is divided by 6 is 5 -/
def mn_remainder (m n : ℕ) : Prop := (m - n) % 6 = 5

-- Theorem statement
theorem remainder_n (m n : ℕ) (h1 : m % 6 = 2) (h2 : (m - n) % 6 = 5) (h3 : m > n) :
  n % 6 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_n_l600_60055


namespace NUMINAMATH_GPT_greatest_number_of_problems_missed_l600_60056

theorem greatest_number_of_problems_missed 
    (total_problems : ℕ) (passing_percentage : ℝ) (max_missed : ℕ) :
    total_problems = 40 →
    passing_percentage = 0.85 →
    max_missed = total_problems - ⌈total_problems * passing_percentage⌉ →
    max_missed = 6 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_greatest_number_of_problems_missed_l600_60056


namespace NUMINAMATH_GPT_combined_tennis_percentage_l600_60010

variable (totalStudentsNorth totalStudentsSouth : ℕ)
variable (percentTennisNorth percentTennisSouth : ℕ)

def studentsPreferringTennisNorth : ℕ := totalStudentsNorth * percentTennisNorth / 100
def studentsPreferringTennisSouth : ℕ := totalStudentsSouth * percentTennisSouth / 100

def totalStudentsBothSchools : ℕ := totalStudentsNorth + totalStudentsSouth
def studentsPreferringTennisBothSchools : ℕ := studentsPreferringTennisNorth totalStudentsNorth percentTennisNorth
                                            + studentsPreferringTennisSouth totalStudentsSouth percentTennisSouth

def combinedPercentTennis : ℕ := studentsPreferringTennisBothSchools totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth
                                 * 100 / totalStudentsBothSchools totalStudentsNorth totalStudentsSouth

theorem combined_tennis_percentage :
  (totalStudentsNorth = 1800) →
  (totalStudentsSouth = 2700) →
  (percentTennisNorth = 25) →
  (percentTennisSouth = 35) →
  combinedPercentTennis totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth = 31 :=
by
  intros
  sorry

end NUMINAMATH_GPT_combined_tennis_percentage_l600_60010


namespace NUMINAMATH_GPT_find_x_l600_60050

theorem find_x (h : 0.60 / x = 6 / 2) : x = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l600_60050


namespace NUMINAMATH_GPT_loan_to_scholarship_ratio_l600_60082

noncomputable def tuition := 22000
noncomputable def parents_contribution := tuition / 2
noncomputable def scholarship := 3000
noncomputable def wage_per_hour := 10
noncomputable def working_hours := 200
noncomputable def earnings := wage_per_hour * working_hours
noncomputable def total_scholarship_and_work := scholarship + earnings
noncomputable def remaining_tuition := tuition - parents_contribution - total_scholarship_and_work
noncomputable def student_loan := remaining_tuition

theorem loan_to_scholarship_ratio :
  (student_loan / scholarship) = 2 := 
by
  sorry

end NUMINAMATH_GPT_loan_to_scholarship_ratio_l600_60082


namespace NUMINAMATH_GPT_range_of_a_l600_60062

variable (f : ℝ → ℝ)

-- f is an odd function
def odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 1: f is an odd function
axiom h_odd : odd_function f

-- Condition 2: f(x) + f(x + 3 / 2) = 0 for any real number x
axiom h_periodicity : ∀ x : ℝ, f x + f (x + 3 / 2) = 0

-- Condition 3: f(1) > 1
axiom h_f1 : f 1 > 1

-- Condition 4: f(2) = a for some real number a
variable (a : ℝ)
axiom h_f2 : f 2 = a

-- Goal: Prove that a < -1
theorem range_of_a : a < -1 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l600_60062


namespace NUMINAMATH_GPT_range_of_a_not_empty_solution_set_l600_60015

theorem range_of_a_not_empty_solution_set :
  {a : ℝ | ∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0} =
  {a : ℝ | a ∈ {a : ℝ | a < -2} ∪ {a : ℝ | a ≥ 6 / 5}} :=
sorry

end NUMINAMATH_GPT_range_of_a_not_empty_solution_set_l600_60015


namespace NUMINAMATH_GPT_find_a_b_l600_60049

noncomputable def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x : ℝ | -3 < x ∧ x < 2 }
noncomputable def sol_set (a b : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + b < 0 }

theorem find_a_b :
  (sol_set (-2) (3 - 6)) = A ∩ B → (-1) + (-2) = -3 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_find_a_b_l600_60049


namespace NUMINAMATH_GPT_intersection_A_complement_B_range_of_a_l600_60052

-- Define sets A and B with their respective conditions
def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Question 1: Prove the intersection when a = 2
theorem intersection_A_complement_B (a : ℝ) (h : a = 2) : 
  A a ∩ (U \ B a) = {x | 2 < x ∧ x ≤ 4} ∪ {x | 5 ≤ x ∧ x < 7} :=
by sorry

-- Question 2: Find the range of a such that A ∪ B = A given a ≠ 1
theorem range_of_a (a : ℝ) (h : a ≠ 1) : 
  (A a ∪ B a = A a) ↔ (1 < a ∧ a ≤ 3 ∨ a = -1) :=
by sorry

end NUMINAMATH_GPT_intersection_A_complement_B_range_of_a_l600_60052


namespace NUMINAMATH_GPT_right_triangle_and_inverse_l600_60013

theorem right_triangle_and_inverse :
  30 * 30 + 272 * 272 = 278 * 278 ∧ (∃ (n : ℕ), 0 ≤ n ∧ n < 4079 ∧ (550 * n) % 4079 = 1) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_and_inverse_l600_60013


namespace NUMINAMATH_GPT_find_c_values_l600_60092

noncomputable def line_intercept_product (c : ℝ) : Prop :=
  let x_intercept := -c / 8
  let y_intercept := -c / 5
  x_intercept * y_intercept = 24

theorem find_c_values :
  ∃ c : ℝ, (line_intercept_product c) ∧ (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) :=
by
  sorry

end NUMINAMATH_GPT_find_c_values_l600_60092


namespace NUMINAMATH_GPT_expr_containing_x_to_y_l600_60041

theorem expr_containing_x_to_y (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  -- proof steps would be here
  sorry

end NUMINAMATH_GPT_expr_containing_x_to_y_l600_60041


namespace NUMINAMATH_GPT_monthly_savings_correct_l600_60027

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end NUMINAMATH_GPT_monthly_savings_correct_l600_60027


namespace NUMINAMATH_GPT_ratio_of_speeds_l600_60004

theorem ratio_of_speeds (L V : ℝ) (R : ℝ) (h1 : L > 0) (h2 : V > 0) (h3 : R ≠ 0)
  (h4 : (1.48 * L) / (R * V) = (1.40 * L) / V) : R = 37 / 35 :=
by
  -- Proof would be inserted here
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l600_60004


namespace NUMINAMATH_GPT_woman_work_rate_l600_60022

theorem woman_work_rate (M W : ℝ) (h1 : 10 * M + 15 * W = 1 / 8) (h2 : M = 1 / 100) : W = 1 / 600 :=
by 
  sorry

end NUMINAMATH_GPT_woman_work_rate_l600_60022


namespace NUMINAMATH_GPT_vector_solution_l600_60090

theorem vector_solution
  (x y : ℝ)
  (h1 : (2*x - y = 0))
  (h2 : (x^2 + y^2 = 20)) :
  (x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end NUMINAMATH_GPT_vector_solution_l600_60090


namespace NUMINAMATH_GPT_polynomial_real_roots_abs_c_geq_2_l600_60043

-- Definition of the polynomial P(x)
def P (x : ℝ) (a b c : ℝ) : ℝ := x^6 + a*x^5 + b*x^4 + c*x^3 + b*x^2 + a*x + 1

-- Statement of the problem: Given that P(x) has six distinct real roots, prove |c| ≥ 2
theorem polynomial_real_roots_abs_c_geq_2 (a b c : ℝ) :
  (∃ r1 r2 r3 r4 r5 r6 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
                           r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
                           r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
                           r4 ≠ r5 ∧ r4 ≠ r6 ∧
                           r5 ≠ r6 ∧
                           P r1 a b c = 0 ∧ P r2 a b c = 0 ∧ P r3 a b c = 0 ∧
                           P r4 a b c = 0 ∧ P r5 a b c = 0 ∧ P r6 a b c = 0) →
  |c| ≥ 2 := by
  sorry

end NUMINAMATH_GPT_polynomial_real_roots_abs_c_geq_2_l600_60043


namespace NUMINAMATH_GPT_expected_sectors_pizza_l600_60079

/-- Let N be the total number of pizza slices and M be the number of slices taken randomly.
    Given N = 16 and M = 5, the expected number of sectors formed is 11/3. -/
theorem expected_sectors_pizza (N M : ℕ) (hN : N = 16) (hM : M = 5) :
  (N - M) * M / (N - 1) = 11 / 3 :=
  sorry

end NUMINAMATH_GPT_expected_sectors_pizza_l600_60079


namespace NUMINAMATH_GPT_g_difference_l600_60077

def g (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

theorem g_difference (m : ℕ) : 
  g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_g_difference_l600_60077


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l600_60035

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l600_60035


namespace NUMINAMATH_GPT_book_set_cost_l600_60096

theorem book_set_cost (charge_per_sqft : ℝ) (lawn_length lawn_width : ℝ) (num_lawns : ℝ) (additional_area : ℝ) (total_cost : ℝ) :
  charge_per_sqft = 0.10 ∧ lawn_length = 20 ∧ lawn_width = 15 ∧ num_lawns = 3 ∧ additional_area = 600 ∧ total_cost = 150 →
  (num_lawns * (lawn_length * lawn_width) * charge_per_sqft + additional_area * charge_per_sqft = total_cost) :=
by
  sorry

end NUMINAMATH_GPT_book_set_cost_l600_60096
