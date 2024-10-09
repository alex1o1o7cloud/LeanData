import Mathlib

namespace find_a_l1910_191081

noncomputable def log_a (a: ℝ) (x: ℝ) : ℝ := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : log_a a 2 - log_a a 4 = 2) :
  a = Real.sqrt 2 / 2 :=
sorry

end find_a_l1910_191081


namespace distance_traveled_l1910_191099

theorem distance_traveled (speed1 speed2 hours1 hours2 : ℝ)
  (h1 : speed1 = 45) (h2 : hours1 = 2) (h3 : speed2 = 50) (h4 : hours2 = 3) :
  speed1 * hours1 + speed2 * hours2 = 240 := by
  sorry

end distance_traveled_l1910_191099


namespace eval_expression_l1910_191014

theorem eval_expression : 1999^2 - 1998 * 2002 = -3991 := 
by
  sorry

end eval_expression_l1910_191014


namespace necessary_but_not_sufficient_condition_l1910_191080

open Real

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 ≤ a ∧ a ≤ 4) → (a^2 - 4 * a < 0) := 
by
  sorry

end necessary_but_not_sufficient_condition_l1910_191080


namespace shuttle_speed_conversion_l1910_191063

-- Define the speed of the space shuttle in kilometers per second
def shuttle_speed_km_per_sec : ℕ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the expected speed in kilometers per hour
def expected_speed_km_per_hour : ℕ := 21600

-- Prove that the speed converted to kilometers per hour is equal to the expected speed
theorem shuttle_speed_conversion : shuttle_speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hour :=
by
    sorry

end shuttle_speed_conversion_l1910_191063


namespace smallest_possible_odd_b_l1910_191056

theorem smallest_possible_odd_b 
    (a b : ℕ) 
    (h1 : a + b = 90) 
    (h2 : Nat.Prime a) 
    (h3 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ b) 
    (h4 : a > b) 
    (h5 : b % 2 = 1) 
    : b = 85 := 
sorry

end smallest_possible_odd_b_l1910_191056


namespace algebraic_expression_value_l1910_191035

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 1 + Real.sqrt 2) (h2 : b = Real.sqrt 3) : 
  a^2 + b^2 - 2 * a + 1 = 5 := 
by
  sorry

end algebraic_expression_value_l1910_191035


namespace moles_HCl_formed_l1910_191022

-- Define the initial moles of CH4 and Cl2
def CH4_initial : ℕ := 2
def Cl2_initial : ℕ := 4

-- Define the balanced chemical equation in terms of the number of moles
def balanced_equation (CH4 : ℕ) (Cl2 : ℕ) : Prop :=
  CH4 + 4 * Cl2 = 1 * CH4 + 4 * Cl2

-- Theorem statement: Given the conditions, prove the number of moles of HCl formed is 4
theorem moles_HCl_formed (CH4_initial Cl2_initial : ℕ) (h_CH4 : CH4_initial = 2) (h_Cl2 : Cl2_initial = 4) :
  ∃ (HCl : ℕ), HCl = 4 :=
  sorry

end moles_HCl_formed_l1910_191022


namespace Tamika_hours_l1910_191064

variable (h : ℕ)

theorem Tamika_hours :
  (45 * h = 55 * 5 + 85) → h = 8 :=
by 
  sorry

end Tamika_hours_l1910_191064


namespace total_emails_received_l1910_191016

theorem total_emails_received (emails_morning emails_afternoon : ℕ) 
  (h1 : emails_morning = 3) 
  (h2 : emails_afternoon = 5) : 
  emails_morning + emails_afternoon = 8 := 
by 
  sorry

end total_emails_received_l1910_191016


namespace total_points_scored_l1910_191087

def num_members : ℕ := 12
def num_absent : ℕ := 4
def points_per_member : ℕ := 8

theorem total_points_scored : 
  (num_members - num_absent) * points_per_member = 64 := by
  sorry

end total_points_scored_l1910_191087


namespace sufficient_but_not_necessary_l1910_191033

-- Definitions of propositions p and q
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2
def q (a b : ℝ) : Prop := a < b

-- Problem statement as a Lean theorem
theorem sufficient_but_not_necessary (a b m : ℝ) : 
  (p a b m → q a b) ∧ (¬ (q a b → p a b m)) :=
by
  sorry

end sufficient_but_not_necessary_l1910_191033


namespace final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l1910_191028

-- Define the movements as a list of integers
def movements : List ℤ := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

-- Define the function to calculate the final position
def final_position (movements : List ℤ) : ℤ :=
  movements.foldl (· + ·) 0

-- Define the function to find the total distance walked (absolute sum)
def total_distance (movements : List ℤ) : ℕ :=
  movements.foldl (fun acc x => acc + x.natAbs) 0

-- Calorie consumption rate per kilometer (1000 meters)
def calories_per_kilometer : ℕ := 7000

-- Calculate the calories consumed
def calories_consumed (total_meters : ℕ) : ℕ :=
  (total_meters / 1000) * calories_per_kilometer

-- Lean 4 theorem statements

theorem final_position_west_of_bus_stop : final_position movements = -400 := by
  sorry

theorem distance_from_bus_stop : |final_position movements| = 400 := by
  sorry

theorem total_calories_consumed : calories_consumed (total_distance movements) = 44800 := by
  sorry

end final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l1910_191028


namespace solve_inequality_l1910_191078

theorem solve_inequality (x : ℝ) : 
  (-1 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧ 
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 1) ↔ (1 < x) := 
by 
  sorry

end solve_inequality_l1910_191078


namespace area_of_triangle_intercepts_l1910_191084

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end area_of_triangle_intercepts_l1910_191084


namespace number_of_ways_to_assign_shifts_l1910_191062

def workers : List String := ["A", "B", "C"]

theorem number_of_ways_to_assign_shifts :
  let shifts := ["day", "night"]
  (workers.length * (workers.length - 1)) = 6 := by
  sorry

end number_of_ways_to_assign_shifts_l1910_191062


namespace proof_not_necessarily_15_points_l1910_191068

-- Define the number of teams
def teams := 14

-- Define a tournament where each team plays every other exactly once
def games := (teams * (teams - 1)) / 2

-- Define a function calculating the total points by summing points for each game
def total_points (wins draws : ℕ) := (3 * wins) + (1 * draws)

-- Define a statement that total points is at least 150
def scores_sum_at_least_150 (wins draws : ℕ) : Prop :=
  total_points wins draws ≥ 150

-- Define a condition that a score could be less than 15
def highest_score_not_necessarily_15 : Prop :=
  ∃ (scores : Finset ℕ), scores.card = teams ∧ ∀ score ∈ scores, score < 15

theorem proof_not_necessarily_15_points :
  ∃ (wins draws : ℕ), wins + draws = games ∧ scores_sum_at_least_150 wins draws ∧ highest_score_not_necessarily_15 :=
by
  sorry

end proof_not_necessarily_15_points_l1910_191068


namespace tom_first_part_speed_l1910_191036

theorem tom_first_part_speed 
  (total_distance : ℕ)
  (distance_first_part : ℕ)
  (speed_second_part : ℕ)
  (average_speed : ℕ)
  (total_time : ℕ)
  (distance_remaining : ℕ)
  (T2 : ℕ)
  (v : ℕ) :
  total_distance = 80 →
  distance_first_part = 30 →
  speed_second_part = 50 →
  average_speed = 40 →
  total_time = 2 →
  distance_remaining = 50 →
  T2 = 1 →
  total_time = distance_first_part / v + T2 →
  v = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, we need to prove that v = 30 given the above conditions.
  sorry

end tom_first_part_speed_l1910_191036


namespace fifty_third_card_is_A_s_l1910_191000

def sequence_position (n : ℕ) : String :=
  let cycle_length := 26
  let pos_in_cycle := (n - 1) % cycle_length + 1
  if pos_in_cycle <= 13 then
    "A_s"
  else
    "A_h"

theorem fifty_third_card_is_A_s : sequence_position 53 = "A_s" := by
  sorry  -- proof placeholder

end fifty_third_card_is_A_s_l1910_191000


namespace regular_pentagons_similar_l1910_191093

-- Define a regular pentagon
structure RegularPentagon :=
  (side_length : ℝ)
  (internal_angle : ℝ)
  (angle_eq : internal_angle = 108)
  (side_positive : side_length > 0)

-- The theorem stating that two regular pentagons are always similar
theorem regular_pentagons_similar (P Q : RegularPentagon) : 
  ∀ P Q : RegularPentagon, P.internal_angle = Q.internal_angle ∧ P.side_length * Q.side_length ≠ 0 := 
sorry

end regular_pentagons_similar_l1910_191093


namespace system_of_equations_solution_l1910_191076

theorem system_of_equations_solution :
  ∃ x y : ℝ, (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) :=
by
  sorry

end system_of_equations_solution_l1910_191076


namespace coordinates_of_point_P_l1910_191021

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end coordinates_of_point_P_l1910_191021


namespace part_a_part_b_l1910_191088

-- Part (a)
theorem part_a (n : ℕ) (h : n > 0) :
  (2 * n ∣ n * (n + 1) / 2) ↔ ∃ k : ℕ, n = 4 * k - 1 :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n > 0) :
  (2 * n + 1 ∣ n * (n + 1) / 2) ↔ (2 * n + 1 ≡ 1 [MOD 4]) ∨ (2 * n + 1 ≡ 3 [MOD 4]) :=
by sorry

end part_a_part_b_l1910_191088


namespace num_4digit_special_integers_l1910_191004

noncomputable def count_valid_4digit_integers : ℕ :=
  let first_two_options := 3 * 3 -- options for the first two digits
  let valid_last_two_pairs := 4 -- (6,9), (7,8), (8,7), (9,6)
  first_two_options * valid_last_two_pairs

theorem num_4digit_special_integers : count_valid_4digit_integers = 36 :=
by
  sorry

end num_4digit_special_integers_l1910_191004


namespace least_common_multiple_first_ten_l1910_191013

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l1910_191013


namespace boxes_per_class_l1910_191092

variable (boxes : ℕ) (classes : ℕ)

theorem boxes_per_class (h1 : boxes = 3) (h2 : classes = 4) : 
  (boxes : ℚ) / (classes : ℚ) = 3 / 4 :=
by
  rw [h1, h2]
  norm_num

end boxes_per_class_l1910_191092


namespace problem1_problem2_problem3_l1910_191048

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b - 2^x) / (2^(x + 1) + a)

-- Problem 1
theorem problem1 (h_odd : ∀ x, f x a b = -f (-x) a b) : a = 2 ∧ b = 1 :=
sorry

-- Problem 2
theorem problem2 : (∀ x, f x 2 1 = -f (-x) 2 1) → ∀ x y, x < y → f x 2 1 > f y 2 1 :=
sorry

-- Problem 3
theorem problem3 (h_pos : ∀ x ≥ 1, f (k * 3^x) 2 1 + f (3^x - 9^x + 2) 2 1 > 0) : k < 4 / 3 :=
sorry

end problem1_problem2_problem3_l1910_191048


namespace mod_equiv_22_l1910_191051

theorem mod_equiv_22 : ∃ m : ℕ, (198 * 864) % 50 = m ∧ 0 ≤ m ∧ m < 50 ∧ m = 22 := by
  sorry

end mod_equiv_22_l1910_191051


namespace boat_trip_l1910_191018

variable {v v_T : ℝ}

theorem boat_trip (d_total t_total : ℝ) (h1 : d_total = 10) (h2 : t_total = 5) (h3 : 2 / (v - v_T) = 3 / (v + v_T)) :
  v_T = 5 / 12 ∧ (5 / (v - v_T)) = 3 ∧ (5 / (v + v_T)) = 2 :=
by
  have h4 : 1 / (d_total / t_total) = v - v_T := sorry
  have h5 : 1 / (d_total / t_total) = v + v_T := sorry
  have h6 : v = 5 * v_T := sorry
  have h7 : v_T = 5 / 12 := sorry
  have t_upstream : 5 / (v - v_T) = 3 := sorry
  have t_downstream : 5 / (v + v_T) = 2 := sorry
  exact ⟨h7, t_upstream, t_downstream⟩

end boat_trip_l1910_191018


namespace g_property_l1910_191006

theorem g_property (g : ℝ → ℝ) (h : ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y) :
  let n := 2
  let s := 14 / 3
  n = 2 ∧ s = 14 / 3 ∧ n * s = 28 / 3 :=
by {
  sorry
}

end g_property_l1910_191006


namespace sqrt_five_gt_two_l1910_191031

theorem sqrt_five_gt_two : Real.sqrt 5 > 2 :=
by
  -- Proof goes here
  sorry

end sqrt_five_gt_two_l1910_191031


namespace thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l1910_191071

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_number : triangular_number 30 = 465 := 
by
  sorry

theorem sum_of_thirtieth_and_twentyninth_triangular_numbers : triangular_number 30 + triangular_number 29 = 900 := 
by
  sorry

end thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l1910_191071


namespace find_union_of_sets_l1910_191009

-- Define the sets A and B in terms of a
def A (a : ℤ) : Set ℤ := { n | n = |a + 1| ∨ n = 3 ∨ n = 5 }
def B (a : ℤ) : Set ℤ := { n | n = 2 * a + 1 ∨ n = a^2 + 2 * a ∨ n = a^2 + 2 * a - 1 }

-- Given condition: A ∩ B = {2, 3}
def condition (a : ℤ) : Prop := A a ∩ B a = {2, 3}

-- The correct answer: A ∪ B = {-5, 2, 3, 5}
theorem find_union_of_sets (a : ℤ) (h : condition a) : A a ∪ B a = {-5, 2, 3, 5} :=
sorry

end find_union_of_sets_l1910_191009


namespace rectangle_tileable_iff_divisible_l1910_191083

def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def tileable_with_0b_tiles (m n b : ℕ) : Prop :=
  ∃ t : ℕ, t * (2 * b) = m * n  -- This comes from the total area divided by the area of one tile

theorem rectangle_tileable_iff_divisible (m n b : ℕ) :
  tileable_with_0b_tiles m n b ↔ divisible_by (2 * b) m ∨ divisible_by (2 * b) n := 
sorry

end rectangle_tileable_iff_divisible_l1910_191083


namespace original_radius_of_cylinder_l1910_191011

theorem original_radius_of_cylinder (r z : ℝ) (h : ℝ := 3) :
  z = 3 * π * ((r + 8)^2 - r^2) → z = 8 * π * r^2 → r = 8 :=
by
  intros hz1 hz2
  -- Translate given conditions into their equivalent expressions and equations
  sorry

end original_radius_of_cylinder_l1910_191011


namespace brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l1910_191089

noncomputable def brocard_vertex_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(a * b * c, c^3, b^3)

theorem brocard_vertex_coordinates_correct (a b c : ℝ) :
  brocard_vertex_trilinear_coordinates a b c = (a * b * c, c^3, b^3) :=
sorry

noncomputable def steiner_point_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(1 / (a * (b^2 - c^2)),
  1 / (b * (c^2 - a^2)),
  1 / (c * (a^2 - b^2)))

theorem steiner_point_coordinates_correct (a b c : ℝ) :
  steiner_point_trilinear_coordinates a b c = 
  (1 / (a * (b^2 - c^2)),
   1 / (b * (c^2 - a^2)),
   1 / (c * (a^2 - b^2))) :=
sorry

end brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l1910_191089


namespace maximum_distinct_numbers_l1910_191094

theorem maximum_distinct_numbers (n : ℕ) (hsum : n = 250) : 
  ∃ k ≤ 21, k = 21 :=
by
  sorry

end maximum_distinct_numbers_l1910_191094


namespace find_perpendicular_vector_l1910_191046

def vector_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_magnitude_equal (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 ^ 2 + v1.2 ^ 2) = (v2.1 ^ 2 + v2.2 ^ 2)

theorem find_perpendicular_vector (a b : ℝ) :
  ∃ n : ℝ × ℝ, vector_perpendicular (a, b) n ∧ vector_magnitude_equal (a, b) n ∧ n = (b, -a) :=
by
  sorry

end find_perpendicular_vector_l1910_191046


namespace smallest_yummy_integer_l1910_191060

theorem smallest_yummy_integer :
  ∃ (n A : ℤ), 4046 = n * (2 * A + n - 1) ∧ A ≥ 0 ∧ (∀ m, 4046 = m * (2 * A + m - 1) ∧ m ≥ 0 → A ≤ 1011) :=
sorry

end smallest_yummy_integer_l1910_191060


namespace boss_spends_7600_per_month_l1910_191001

def hoursPerWeekFiona : ℕ := 40
def hoursPerWeekJohn : ℕ := 30
def hoursPerWeekJeremy : ℕ := 25
def hourlyRate : ℕ := 20
def weeksPerMonth : ℕ := 4

def weeklyEarnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def monthlyEarnings (weekly : ℕ) (weeks : ℕ) : ℕ := weekly * weeks

def totalMonthlyExpenditure : ℕ :=
  monthlyEarnings (weeklyEarnings hoursPerWeekFiona hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJohn hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJeremy hourlyRate) weeksPerMonth

theorem boss_spends_7600_per_month :
  totalMonthlyExpenditure = 7600 :=
by
  sorry

end boss_spends_7600_per_month_l1910_191001


namespace no_positive_integer_solution_l1910_191017

theorem no_positive_integer_solution (p x y : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) 
  (h_p_div_x : p ∣ x) (hx_pos : 0 < x) (hy_pos : 0 < y) : x^2 - 1 ≠ y^p :=
sorry

end no_positive_integer_solution_l1910_191017


namespace find_max_value_l1910_191085

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end find_max_value_l1910_191085


namespace problem_1_problem_2_l1910_191010

-- Condition for Question 1
def f (x : ℝ) (a : ℝ) := |x - a|

-- Proof Problem for Question 1
theorem problem_1 (a : ℝ) (h : a = 1) : {x : ℝ | f x a > 1/2 * (x + 1)} = {x | x > 3 ∨ x < 1/3} :=
sorry

-- Condition for Question 2
def g (x : ℝ) (a : ℝ) := |x - a| + |x - 2|

-- Proof Problem for Question 2
theorem problem_2 (a : ℝ) : (∃ x : ℝ, g x a ≤ 3) → (-1 ≤ a ∧ a ≤ 5) :=
sorry

end problem_1_problem_2_l1910_191010


namespace monkey_slip_distance_l1910_191025

theorem monkey_slip_distance
  (height : ℕ)
  (climb_per_hour : ℕ)
  (hours : ℕ)
  (s : ℕ)
  (total_hours : ℕ)
  (final_climb : ℕ)
  (reach_top : height = hours * (climb_per_hour - s) + final_climb)
  (total_hours_constraint : total_hours = 17)
  (climb_per_hour_constraint : climb_per_hour = 3)
  (height_constraint : height = 19)
  (final_climb_constraint : final_climb = 3)
  (hours_constraint : hours = 16) :
  s = 2 := sorry

end monkey_slip_distance_l1910_191025


namespace prove_y_l1910_191040

theorem prove_y (x y : ℝ) (h1 : 3 * x^2 - 4 * x + 7 * y + 3 = 0) (h2 : 3 * x - 5 * y + 6 = 0) :
  25 * y^2 - 39 * y + 69 = 0 := sorry

end prove_y_l1910_191040


namespace roots_of_polynomial_l1910_191023

-- Define the polynomial
def poly := fun (x : ℝ) => x^3 - 7 * x^2 + 14 * x - 8

-- Define the statement
theorem roots_of_polynomial : (poly 1 = 0) ∧ (poly 2 = 0) ∧ (poly 4 = 0) :=
  by
  sorry

end roots_of_polynomial_l1910_191023


namespace coeff_x3y5_in_expansion_l1910_191065

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l1910_191065


namespace investment_amount_first_rate_l1910_191019

theorem investment_amount_first_rate : ∀ (x y : ℝ) (r : ℝ),
  x + y = 15000 → -- Condition 1 (Total investments)
  8200 * r + 6800 * 0.075 = 1023 → -- Condition 2 (Interest yield)
  x = 8200 → -- Condition 3 (Amount invested at first rate)
  x = 8200 := -- Question (How much was invested)
by
  intros x y r h₁ h₂ h₃
  exact h₃

end investment_amount_first_rate_l1910_191019


namespace union_sets_l1910_191045

namespace Proof

def setA : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
sorry

end Proof

end union_sets_l1910_191045


namespace points_lie_on_hyperbola_l1910_191024

noncomputable
def point_on_hyperbola (t : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ 
    (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) }

theorem points_lie_on_hyperbola : 
  ∀ t : ℝ, ∀ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) → (x^2 / 16) - (y^2 / 1) = 1 :=
by 
  intro t x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end points_lie_on_hyperbola_l1910_191024


namespace quadratic_equation_solutions_l1910_191015

theorem quadratic_equation_solutions (x : ℝ) : x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 :=
by
  sorry

end quadratic_equation_solutions_l1910_191015


namespace Nina_has_16dollars65_l1910_191038

-- Definitions based on given conditions
variables (W M : ℝ)

-- Condition 1: Nina has exactly enough money to purchase 5 widgets
def condition1 : Prop := 5 * W = M

-- Condition 2: If the cost of each widget were reduced by $1.25, Nina would have exactly enough money to purchase 8 widgets
def condition2 : Prop := 8 * (W - 1.25) = M

-- Statement: Proving the amount of money Nina has is $16.65
theorem Nina_has_16dollars65 (h1 : condition1 W M) (h2 : condition2 W M) : M = 16.65 :=
sorry

end Nina_has_16dollars65_l1910_191038


namespace systematic_sampling_method_l1910_191049

def num_rows : ℕ := 50
def num_seats_per_row : ℕ := 30

def is_systematic_sampling (select_interval : ℕ) : Prop :=
  ∀ n, select_interval = n * num_seats_per_row + 8

theorem systematic_sampling_method :
  is_systematic_sampling 30 :=
by
  sorry

end systematic_sampling_method_l1910_191049


namespace find_first_term_and_common_difference_l1910_191003

variable (n : ℕ)
variable (a_1 d : ℚ)

-- Definition of the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (n : ℕ) (a_1 d : ℚ) : ℚ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2

-- Theorem to prove
theorem find_first_term_and_common_difference 
  (a_1 d : ℚ) 
  (sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2) 
: a_1 = 1/2 ∧ d = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end find_first_term_and_common_difference_l1910_191003


namespace max_surface_area_of_rectangular_solid_l1910_191044

theorem max_surface_area_of_rectangular_solid {r a b c : ℝ} (h_sphere : 4 * π * r^2 = 4 * π)
  (h_diagonal : a^2 + b^2 + c^2 = (2 * r)^2) :
  2 * (a * b + a * c + b * c) ≤ 8 :=
by
  sorry

end max_surface_area_of_rectangular_solid_l1910_191044


namespace area_of_square_with_given_diagonal_l1910_191069

-- Definition of the conditions
def diagonal := 12
def s := Real
def area (s : Real) := s^2
def diag_relation (d s : Real) := d^2 = 2 * s^2

-- The proof statement
theorem area_of_square_with_given_diagonal :
  ∃ s : Real, diag_relation diagonal s ∧ area s = 72 :=
by
  sorry

end area_of_square_with_given_diagonal_l1910_191069


namespace area_of_OPF_eq_sqrt_2_div_2_l1910_191061

noncomputable def area_of_triangle_OPF : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  if (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) then
    let base := dist O F
    let height := Real.sqrt 2
    (1 / 2) * base * height
  else
    0

theorem area_of_OPF_eq_sqrt_2_div_2 : 
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) →
  let base := dist O F
  let height := Real.sqrt 2
  area_of_triangle_OPF = Real.sqrt 2 / 2 := 
by 
  sorry

end area_of_OPF_eq_sqrt_2_div_2_l1910_191061


namespace number_of_8th_graders_l1910_191075

variable (x y : ℕ)
variable (y_valid : 0 ≤ y)

theorem number_of_8th_graders (h : x * (x + 3 - 2 * y) = 14) :
  x = 7 :=
by 
  sorry

end number_of_8th_graders_l1910_191075


namespace arithmetic_sequence_a5_l1910_191030

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ a_1 d, ∀ n, a (n + 1) = a_1 + n * d

theorem arithmetic_sequence_a5 (a : ℕ → α) (h_seq : is_arithmetic_sequence a) (h_cond : a 1 + a 7 = 12) :
  a 4 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l1910_191030


namespace problem_solution_l1910_191091

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x < -6 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 ))
  (h2 : a < b)
  : a + 2 * b + 3 * c = 74 := 
sorry

end problem_solution_l1910_191091


namespace lines_proportional_l1910_191082

variables {x y : ℝ} {p q : ℝ}

theorem lines_proportional (h1 : p * x + 2 * y = 7) (h2 : 3 * x + q * y = 5) :
  p = 21 / 5 := 
sorry

end lines_proportional_l1910_191082


namespace total_dollars_l1910_191072

theorem total_dollars (john emma lucas : ℝ) 
  (h_john : john = 4 / 5) 
  (h_emma : emma = 2 / 5) 
  (h_lucas : lucas = 1 / 2) : 
  john + emma + lucas = 1.7 := by
  sorry

end total_dollars_l1910_191072


namespace proposition_false_l1910_191052

theorem proposition_false : ¬ ∀ x ∈ ({1, -1, 0} : Set ℤ), 2 * x + 1 > 0 := by
  sorry

end proposition_false_l1910_191052


namespace line_points_sum_slope_and_intercept_l1910_191043

-- Definition of the problem
theorem line_points_sum_slope_and_intercept (a b : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 3) ∨ (x = 10 ∧ y = 19) → y = a * x + b) →
  a + b = 1 :=
by
  intro h
  sorry

end line_points_sum_slope_and_intercept_l1910_191043


namespace intersection_always_exists_minimum_chord_length_and_equation_l1910_191050

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 4 * x - 8 * y - 11 = 0

noncomputable def line_eq (m x y : ℝ) : Prop :=
  (m - 1) * x + m * y = m + 1

theorem intersection_always_exists :
  ∀ (m : ℝ), ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
by
  sorry

theorem minimum_chord_length_and_equation :
  ∃ (k : ℝ) (x y : ℝ), k = sqrt 3 ∧ (3 * x - 2 * y + 7 = 0) ∧
    ∀ m, ∃ (xp yp : ℝ), line_eq m xp yp ∧ ∃ (l1 l2 : ℝ), line_eq m l1 l2 ∧ 
    (circle_eq xp yp ∧ circle_eq l1 l2)  :=
by
  sorry

end intersection_always_exists_minimum_chord_length_and_equation_l1910_191050


namespace number_of_slices_per_pizza_l1910_191055

-- Given conditions as definitions in Lean 4
def total_pizzas := 2
def total_slices_per_pizza (S : ℕ) : ℕ := total_pizzas * S
def james_portion : ℚ := 2 / 3
def james_ate_slices (S : ℕ) : ℚ := james_portion * (total_slices_per_pizza S)
def james_ate_exactly := 8

-- The main theorem to prove
theorem number_of_slices_per_pizza (S : ℕ) (h : james_ate_slices S = james_ate_exactly) : S = 6 :=
sorry

end number_of_slices_per_pizza_l1910_191055


namespace find_x_l1910_191090

theorem find_x (x : ℤ) (h : 7 * x - 18 = 66) : x = 12 :=
  sorry

end find_x_l1910_191090


namespace Olivia_house_height_l1910_191073

variable (h : ℕ)
variable (flagpole_height : ℕ := 35)
variable (flagpole_shadow : ℕ := 30)
variable (house_shadow : ℕ := 70)
variable (bush_height : ℕ := 14)
variable (bush_shadow : ℕ := 12)

theorem Olivia_house_height :
  (house_shadow / flagpole_shadow) * flagpole_height = 81 ∧
  (house_shadow / bush_shadow) * bush_height = 81 :=
by
  sorry

end Olivia_house_height_l1910_191073


namespace transylvanian_convinces_l1910_191067

theorem transylvanian_convinces (s : Prop) (t : Prop) (h : s ↔ (¬t ∧ ¬s)) : t :=
by
  -- Leverage the existing equivalence to prove the desired result
  sorry

end transylvanian_convinces_l1910_191067


namespace fraction_to_decimal_l1910_191053

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end fraction_to_decimal_l1910_191053


namespace simplify_polynomial_l1910_191066

theorem simplify_polynomial (q : ℤ) :
  (4*q^4 - 2*q^3 + 3*q^2 - 7*q + 9) + (5*q^3 - 8*q^2 + 6*q - 1) =
  4*q^4 + 3*q^3 - 5*q^2 - q + 8 :=
sorry

end simplify_polynomial_l1910_191066


namespace find_p_l1910_191027

/-- Given the points Q(0, 15), A(3, 15), B(15, 0), O(0, 0), and C(0, p).
The area of triangle ABC is given as 45.
We need to prove that p = 11.25. -/
theorem find_p (ABC_area : ℝ) (p : ℝ) (h : ABC_area = 45) :
  p = 11.25 :=
by
  sorry

end find_p_l1910_191027


namespace eliminate_all_evil_with_at_most_one_good_l1910_191096

-- Defining the problem setting
structure Wizard :=
  (is_good : Bool)

-- The main theorem
theorem eliminate_all_evil_with_at_most_one_good (wizards : List Wizard) (h_wizard_count : wizards.length = 2015) :
  ∃ (banish_sequence : List Wizard), 
    (∀ w ∈ banish_sequence, w.is_good = false) ∨ (∃ (g : Wizard), g.is_good = true ∧ g ∉ banish_sequence) :=
sorry

end eliminate_all_evil_with_at_most_one_good_l1910_191096


namespace solve_fractional_equation_l1910_191058

theorem solve_fractional_equation (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (3 / (x^2 - x) + 1 = x / (x - 1)) → x = 3 :=
by
  sorry -- Placeholder for the actual proof

end solve_fractional_equation_l1910_191058


namespace unique_function_satisfying_conditions_l1910_191012

theorem unique_function_satisfying_conditions (f : ℤ → ℤ) :
  (∀ n : ℤ, f (f n) + f n = 2 * n + 3) → 
  (f 0 = 1) → 
  (∀ n : ℤ, f n = n + 1) :=
by
  intro h1 h2
  sorry

end unique_function_satisfying_conditions_l1910_191012


namespace Bernoulli_inequality_l1910_191041

theorem Bernoulli_inequality (p : ℝ) (k : ℚ) (hp : 0 < p) (hk : 1 < k) : 
  (1 + p) ^ (k : ℝ) > 1 + p * (k : ℝ) := by
sorry

end Bernoulli_inequality_l1910_191041


namespace inequality_solution_set_l1910_191034

noncomputable def solution_set : Set ℝ := { x : ℝ | x > 5 ∨ x < -2 }

theorem inequality_solution_set (x : ℝ) :
  x^2 - 3 * x - 10 > 0 ↔ x > 5 ∨ x < -2 :=
by
  sorry

end inequality_solution_set_l1910_191034


namespace fraction_is_one_fourth_l1910_191008

-- Defining the numbers
def num1 : ℕ := 16
def num2 : ℕ := 8

-- Conditions
def difference_correct : Prop := num1 - num2 = 8
def sum_of_numbers : ℕ := num1 + num2
def fraction_of_sum (f : ℚ) : Prop := f * sum_of_numbers = 6

-- Theorem stating the fraction
theorem fraction_is_one_fourth (f : ℚ) (h1 : difference_correct) (h2 : fraction_of_sum f) : f = 1 / 4 :=
by {
  -- This will use the conditions and show that f = 1/4
  sorry
}

end fraction_is_one_fourth_l1910_191008


namespace geometric_sum_eight_terms_l1910_191042

noncomputable def geometric_series_sum_8 (a r : ℝ) : ℝ :=
  a * (1 - r^8) / (1 - r)

theorem geometric_sum_eight_terms
  (a r : ℝ) (h_geom_pos : r > 0)
  (h_sum_two : a + a * r = 2)
  (h_sum_eight : a * r^2 + a * r^3 = 8) :
  geometric_series_sum_8 a r = 170 := 
sorry

end geometric_sum_eight_terms_l1910_191042


namespace marcy_minimum_avg_score_l1910_191097

variables (s1 s2 s3 : ℝ)
variable (qualified_avg : ℝ := 90)
variable (required_total : ℝ := 5 * qualified_avg)
variable (first_three_total : ℝ := s1 + s2 + s3)
variable (needed_points : ℝ := required_total - first_three_total)
variable (required_avg : ℝ := needed_points / 2)

/-- The admission criteria for a mathematics contest require a contestant to 
    achieve an average score of at least 90% over five rounds to qualify for the final round.
    Marcy scores 87%, 92%, and 85% in the first three rounds. 
    Prove that Marcy must average at least 93% in the next two rounds to qualify for the final. --/
theorem marcy_minimum_avg_score 
    (h1 : s1 = 87) (h2 : s2 = 92) (h3 : s3 = 85)
    : required_avg ≥ 93 :=
sorry

end marcy_minimum_avg_score_l1910_191097


namespace odd_positive_integer_minus_twenty_l1910_191032

theorem odd_positive_integer_minus_twenty (x : ℕ) (h : x = 53) : (2 * x - 1) - 20 = 85 := by
  subst h
  rfl

end odd_positive_integer_minus_twenty_l1910_191032


namespace simplify_parentheses_l1910_191007

theorem simplify_parentheses (a b c x y : ℝ) : (3 * a - (2 * a - c) = 3 * a - 2 * a + c) := 
by 
  sorry

end simplify_parentheses_l1910_191007


namespace min_students_participating_l1910_191059

def ratio_9th_to_10th (n9 n10 : ℕ) : Prop := n9 * 4 = n10 * 3
def ratio_10th_to_11th (n10 n11 : ℕ) : Prop := n10 * 6 = n11 * 5

theorem min_students_participating (n9 n10 n11 : ℕ) 
    (h1 : ratio_9th_to_10th n9 n10) 
    (h2 : ratio_10th_to_11th n10 n11) : 
    n9 + n10 + n11 = 59 :=
sorry

end min_students_participating_l1910_191059


namespace relationship_a_b_l1910_191086

theorem relationship_a_b (a b : ℝ) :
  (∃ (P : ℝ × ℝ), P ∈ {Q : ℝ × ℝ | Q.snd = -3 * Q.fst + b} ∧
                   ∃ (R : ℝ × ℝ), R ∈ {S : ℝ × ℝ | S.snd = -a * S.fst + 3} ∧
                   R = (-P.snd, -P.fst)) →
  a = 1 / 3 ∧ b = -9 :=
by
  intro h
  sorry

end relationship_a_b_l1910_191086


namespace trivia_competition_points_l1910_191070

theorem trivia_competition_points 
  (total_members : ℕ := 120) 
  (absent_members : ℕ := 37) 
  (points_per_member : ℕ := 24) : 
  (total_members - absent_members) * points_per_member = 1992 := 
by
  sorry

end trivia_competition_points_l1910_191070


namespace estimate_white_balls_l1910_191095

-- Statements for conditions
variables (black_balls white_balls : ℕ)
variables (draws : ℕ := 40)
variables (black_draws : ℕ := 10)

-- Define total white draws
def white_draws := draws - black_draws

-- Ratio of black to white draws
def draw_ratio := black_draws / white_draws

-- Given condition on known draws
def black_ball_count := 4
def known_draw_ratio := 1 / 3

-- Lean 4 statement to prove the number of white balls
theorem estimate_white_balls (h : black_ball_count / white_balls = known_draw_ratio) : white_balls = 12 :=
sorry -- Proof omitted

end estimate_white_balls_l1910_191095


namespace GCF_75_135_l1910_191079

theorem GCF_75_135 : Nat.gcd 75 135 = 15 :=
by
sorry

end GCF_75_135_l1910_191079


namespace phil_cards_left_l1910_191002

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end phil_cards_left_l1910_191002


namespace max_min_x_sub_2y_l1910_191026

theorem max_min_x_sub_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 0 ≤ x - 2*y ∧ x - 2*y ≤ 10 :=
sorry

end max_min_x_sub_2y_l1910_191026


namespace cube_of_composite_as_diff_of_squares_l1910_191077

theorem cube_of_composite_as_diff_of_squares (n : ℕ) (h : ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    n^3 = A₁^2 - B₁^2 ∧ 
    n^3 = A₂^2 - B₂^2 ∧ 
    n^3 = A₃^2 - B₃^2 ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) := sorry

end cube_of_composite_as_diff_of_squares_l1910_191077


namespace solveSystem1_solveFractionalEq_l1910_191047

-- Definition: system of linear equations
def system1 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ x - 4 * y = 9

-- Theorem: solution to the system of equations
theorem solveSystem1 : ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = -1 :=
by
  sorry
  
-- Definition: fractional equation
def fractionalEq (x : ℝ) : Prop :=
  (x + 2) / (x^2 - 2 * x + 1) + 3 / (x - 1) = 0

-- Theorem: solution to the fractional equation
theorem solveFractionalEq : ∃ x : ℝ, fractionalEq x ∧ x = 1 / 4 :=
by
  sorry

end solveSystem1_solveFractionalEq_l1910_191047


namespace inequality_hold_l1910_191005

theorem inequality_hold (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - |c| > b - |c| :=
sorry

end inequality_hold_l1910_191005


namespace Hiram_age_l1910_191074

theorem Hiram_age (H A : ℕ) (h₁ : H + 12 = 2 * A - 4) (h₂ : A = 28) : H = 40 :=
by
  sorry

end Hiram_age_l1910_191074


namespace g_of_square_sub_one_l1910_191057

variable {R : Type*} [LinearOrderedField R]

def g (x : R) : R := 3

theorem g_of_square_sub_one (x : R) : g ((x - 1)^2) = 3 := 
by sorry

end g_of_square_sub_one_l1910_191057


namespace bear_weight_gain_l1910_191039

theorem bear_weight_gain :
  let total_weight := 1000
  let weight_from_berries := total_weight / 5
  let weight_from_acorns := 2 * weight_from_berries
  let weight_from_salmon := (total_weight - weight_from_berries - weight_from_acorns) / 2
  let weight_from_small_animals := total_weight - (weight_from_berries + weight_from_acorns + weight_from_salmon)
  weight_from_small_animals = 200 :=
by sorry

end bear_weight_gain_l1910_191039


namespace ordered_pairs_of_positive_integers_l1910_191037

theorem ordered_pairs_of_positive_integers (x y : ℕ) (h : x * y = 2800) :
  2^4 * 5^2 * 7 = 2800 → ∃ (n : ℕ), n = 30 ∧ (∃ x y : ℕ, x * y = 2800 ∧ n = 30) :=
by
  sorry

end ordered_pairs_of_positive_integers_l1910_191037


namespace cannot_tile_remaining_with_dominoes_l1910_191029

def can_tile_remaining_board (pieces : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j

theorem cannot_tile_remaining_with_dominoes : 
  ∃ (pieces : List (ℕ × ℕ)), (∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 10) ∧ (1 ≤ j ∧ j ≤ 10) → ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j) ∧ ¬ can_tile_remaining_board pieces :=
sorry

end cannot_tile_remaining_with_dominoes_l1910_191029


namespace find_antonym_word_l1910_191054

-- Defining the condition that the word means "rarely" or "not often."
def means_rarely_or_not_often (word : String) : Prop :=
  word = "seldom"

-- Theorem statement: There exists a word such that it meets the given condition.
theorem find_antonym_word : 
  ∃ word : String, means_rarely_or_not_often word :=
by
  use "seldom"
  unfold means_rarely_or_not_often
  rfl

end find_antonym_word_l1910_191054


namespace johns_total_expenditure_l1910_191098

-- Conditions
def treats_first_15_days : ℕ := 3 * 15
def treats_next_15_days : ℕ := 4 * 15
def total_treats : ℕ := treats_first_15_days + treats_next_15_days
def cost_per_treat : ℝ := 0.10
def discount_threshold : ℕ := 50
def discount_rate : ℝ := 0.10

-- Intermediate calculations
def total_cost_without_discount : ℝ := total_treats * cost_per_treat
def discounted_cost_per_treat : ℝ := cost_per_treat * (1 - discount_rate)
def total_cost_with_discount : ℝ := total_treats * discounted_cost_per_treat

-- Main theorem statement
theorem johns_total_expenditure : total_cost_with_discount = 9.45 :=
by
  -- Place proof here
  sorry

end johns_total_expenditure_l1910_191098


namespace intersection_complement_B_l1910_191020

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - 3 * x < 0 }
def B : Set ℝ := { x | abs x > 2 }

-- Complement of B
def complement_B : Set ℝ := { x | x ≥ -2 ∧ x ≤ 2 }

-- Final statement to prove the intersection equals the given set
theorem intersection_complement_B :
  A ∩ complement_B = { x : ℝ | 0 < x ∧ x ≤ 2 } := 
by 
  -- Proof omitted
  sorry

end intersection_complement_B_l1910_191020
