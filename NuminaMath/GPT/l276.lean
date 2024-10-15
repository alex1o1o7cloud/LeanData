import Mathlib

namespace NUMINAMATH_GPT_find_n_l276_27651

theorem find_n (n : ℚ) : 1 / 2 + 2 / 3 + 3 / 4 + n / 12 = 2 ↔ n = 1 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_find_n_l276_27651


namespace NUMINAMATH_GPT_investment_period_l276_27653

theorem investment_period (x t : ℕ) (p_investment q_investment q_time : ℕ) (profit_ratio : ℚ):
  q_investment = 5 * x →
  p_investment = 7 * x →
  q_time = 16 →
  profit_ratio = 7 / 10 →
  7 * x * t = profit_ratio * 5 * x * q_time →
  t = 8 := sorry

end NUMINAMATH_GPT_investment_period_l276_27653


namespace NUMINAMATH_GPT_find_a_decreasing_l276_27680

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem find_a_decreasing : 
  (∀ x : ℝ, x < 6 → f x a ≤ f (x - 1) a) → a ≥ 6 := 
sorry

end NUMINAMATH_GPT_find_a_decreasing_l276_27680


namespace NUMINAMATH_GPT_number_of_possible_values_for_c_l276_27662

theorem number_of_possible_values_for_c : 
  (∃ c_values : Finset ℕ, (∀ c ∈ c_values, c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) 
  ∧ c_values.card = 10) :=
sorry

end NUMINAMATH_GPT_number_of_possible_values_for_c_l276_27662


namespace NUMINAMATH_GPT_digit_sum_equality_l276_27605

-- Definitions for the conditions
def is_permutation_of_digits (a b : ℕ) : Prop :=
  -- Assume implementation that checks if b is a permutation of the digits of a
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  -- Assume implementation that computes the sum of digits of n
  sorry

-- The theorem statement
theorem digit_sum_equality (a b : ℕ)
  (h : is_permutation_of_digits a b) :
  sum_of_digits (5 * a) = sum_of_digits (5 * b) :=
sorry

end NUMINAMATH_GPT_digit_sum_equality_l276_27605


namespace NUMINAMATH_GPT_find_a_l276_27647

theorem find_a (a : ℝ) (h : a ≠ 0) :
  (∀ x, -1 ≤ x ∧ x ≤ 4 → ax - a + 2 ≤ 7) →
  (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ ax - a + 2 = 7) →
  (a = 5/3 ∨ a = -5/2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l276_27647


namespace NUMINAMATH_GPT_squido_oysters_l276_27616

theorem squido_oysters (S C : ℕ) (h1 : C ≥ 2 * S) (h2 : S + C = 600) : S = 200 :=
sorry

end NUMINAMATH_GPT_squido_oysters_l276_27616


namespace NUMINAMATH_GPT_graph_passes_through_point_l276_27661

theorem graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
    ∀ x y : ℝ, (y = a^(x-2) + 2) → (x = 2) → (y = 3) :=
by
    intros x y hxy hx
    rw [hx] at hxy
    simp at hxy
    sorry

end NUMINAMATH_GPT_graph_passes_through_point_l276_27661


namespace NUMINAMATH_GPT_max_xy_on_line_AB_l276_27676

noncomputable def pointA : ℝ × ℝ := (3, 0)
noncomputable def pointB : ℝ × ℝ := (0, 4)

-- Define the line passing through points A and B
def on_line_AB (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P.1 = 3 - 3 * t ∧ P.2 = 4 * t

theorem max_xy_on_line_AB : ∃ (P : ℝ × ℝ), on_line_AB P ∧ P.1 * P.2 = 3 := 
sorry

end NUMINAMATH_GPT_max_xy_on_line_AB_l276_27676


namespace NUMINAMATH_GPT_proof_problem_l276_27683

def h (x : ℝ) : ℝ := 2 * x + 4
def k (x : ℝ) : ℝ := 4 * x + 6

theorem proof_problem : h (k 3) - k (h 3) = -6 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l276_27683


namespace NUMINAMATH_GPT_sin_neg_pi_div_two_l276_27608

theorem sin_neg_pi_div_two : Real.sin (-π / 2) = -1 := by
  -- Define the necessary conditions
  let π_in_deg : ℝ := 180 -- π radians equals 180 degrees
  have sin_neg_angle : ∀ θ : ℝ, Real.sin (-θ) = -Real.sin θ := sorry -- sin(-θ) = -sin(θ) for any θ
  have sin_90_deg : Real.sin (π_in_deg / 2) = 1 := sorry -- sin(90 degrees) = 1

  -- The main statement to prove
  sorry

end NUMINAMATH_GPT_sin_neg_pi_div_two_l276_27608


namespace NUMINAMATH_GPT_example_problem_l276_27663

-- Define vectors a and b with the given conditions
def a (k : ℝ) : ℝ × ℝ := (2, k)
def b : ℝ × ℝ := (6, 4)

-- Define the condition that vectors are perpendicular
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Calculate the sum of two vectors
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Check if a vector is collinear
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- The main theorem with the given conditions
theorem example_problem (k : ℝ) (hk : perpendicular (a k) b) :
  collinear (vector_add (a k) b) (-16, -2) :=
by
  sorry

end NUMINAMATH_GPT_example_problem_l276_27663


namespace NUMINAMATH_GPT_find_value_l276_27659

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l276_27659


namespace NUMINAMATH_GPT_candy_mixture_solution_l276_27632

theorem candy_mixture_solution :
  ∃ x y : ℝ, 18 * x + 10 * y = 1500 ∧ x + y = 100 ∧ x = 62.5 ∧ y = 37.5 := by
  sorry

end NUMINAMATH_GPT_candy_mixture_solution_l276_27632


namespace NUMINAMATH_GPT_carlos_initial_blocks_l276_27620

theorem carlos_initial_blocks (g : ℕ) (l : ℕ) (total : ℕ) : g = 21 → l = 37 → total = g + l → total = 58 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_carlos_initial_blocks_l276_27620


namespace NUMINAMATH_GPT_expected_hit_targets_correct_expected_hit_targets_at_least_half_l276_27691

noncomputable def expected_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n)^n)

theorem expected_hit_targets_correct (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n = n * (1 - (1 - (1 : ℝ) / n)^n) :=
by
  unfold expected_hit_targets
  sorry

theorem expected_hit_targets_at_least_half (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n >= n / 2 :=
by
  unfold expected_hit_targets
  sorry

end NUMINAMATH_GPT_expected_hit_targets_correct_expected_hit_targets_at_least_half_l276_27691


namespace NUMINAMATH_GPT_domain_is_all_real_l276_27682

-- Definitions and conditions
def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 18

def domain_of_f (x : ℝ) : Prop := ∃ (y : ℝ), y = 1 / (⌊quadratic_expression x⌋)

-- Theorem statement
theorem domain_is_all_real : ∀ x : ℝ, domain_of_f x :=
by
  sorry

end NUMINAMATH_GPT_domain_is_all_real_l276_27682


namespace NUMINAMATH_GPT_smallest_integer_geq_l276_27686

theorem smallest_integer_geq : ∃ (n : ℤ), (n^2 - 9*n + 18 ≥ 0) ∧ ∀ (m : ℤ), (m^2 - 9*m + 18 ≥ 0) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_geq_l276_27686


namespace NUMINAMATH_GPT_infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l276_27684

theorem infinite_nat_sum_of_squares_and_cubes_not_sixth_powers :
  ∃ (N : ℕ) (k : ℕ), N > 0 ∧
  (N = 250 * 3^(6 * k)) ∧
  (∃ (x y : ℕ), N = x^2 + y^2) ∧
  (∃ (a b : ℕ), N = a^3 + b^3) ∧
  (∀ (u v : ℕ), N ≠ u^6 + v^6) :=
by
  sorry

end NUMINAMATH_GPT_infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l276_27684


namespace NUMINAMATH_GPT_cos_double_angle_l276_27655

theorem cos_double_angle (α β : ℝ) (h1 : Real.sin (α - β) = 1 / 3) (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l276_27655


namespace NUMINAMATH_GPT_Gwen_money_left_l276_27669

theorem Gwen_money_left (received spent : ℕ) (h_received : received = 14) (h_spent : spent = 8) : 
  received - spent = 6 := 
by 
  sorry

end NUMINAMATH_GPT_Gwen_money_left_l276_27669


namespace NUMINAMATH_GPT_eugene_swim_time_l276_27627

-- Define the conditions
variable (S : ℕ) -- Swim time on Sunday
variable (swim_time_mon : ℕ := 30) -- Swim time on Monday
variable (swim_time_tue : ℕ := 45) -- Swim time on Tuesday
variable (average_swim_time : ℕ := 34) -- Average swim time over three days

-- The total swim time over three days
def total_swim_time := S + swim_time_mon + swim_time_tue

-- The problem statement: Prove that given the conditions, Eugene swam for 27 minutes on Sunday.
theorem eugene_swim_time : total_swim_time S = 3 * average_swim_time → S = 27 := by
  -- Proof process will follow here
  sorry

end NUMINAMATH_GPT_eugene_swim_time_l276_27627


namespace NUMINAMATH_GPT_tree_planting_activity_l276_27671

theorem tree_planting_activity (x y : ℕ) 
  (h1 : y = 2 * x + 15)
  (h2 : x = y / 3 + 6) : 
  y = 81 ∧ x = 33 := 
by sorry

end NUMINAMATH_GPT_tree_planting_activity_l276_27671


namespace NUMINAMATH_GPT_range_of_a_l276_27621

open Real

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem range_of_a (a : ℝ) (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) :
  (f (a * sin θ) + f (1 - a) > 0) → a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l276_27621


namespace NUMINAMATH_GPT_friends_travelled_distance_l276_27642

theorem friends_travelled_distance :
  let lionel_distance : ℝ := 4 * 5280
  let esther_distance : ℝ := 975 * 3
  let niklaus_distance : ℝ := 1287
  let isabella_distance : ℝ := 18 * 1000 * 3.28084
  let sebastian_distance : ℝ := 2400 * 3.28084
  let total_distance := lionel_distance + esther_distance + niklaus_distance + isabella_distance + sebastian_distance
  total_distance = 91261.136 := 
by
  sorry

end NUMINAMATH_GPT_friends_travelled_distance_l276_27642


namespace NUMINAMATH_GPT_event_B_is_certain_l276_27615

-- Define the event that the sum of two sides of a triangle is greater than the third side
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the term 'certain event'
def certain_event (E : Prop) : Prop := E

/-- Prove that the event "the sum of two sides of a triangle is greater than the third side" is a certain event -/
theorem event_B_is_certain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  certain_event (triangle_inequality a b c) :=
sorry

end NUMINAMATH_GPT_event_B_is_certain_l276_27615


namespace NUMINAMATH_GPT_no_real_solution_l276_27603

theorem no_real_solution (x y : ℝ) : x^3 + y^2 = 2 → x^2 + x * y + y^2 - y = 0 → false := 
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_no_real_solution_l276_27603


namespace NUMINAMATH_GPT_tom_reads_700_pages_in_7_days_l276_27656

theorem tom_reads_700_pages_in_7_days
  (total_hours : ℕ)
  (total_days : ℕ)
  (pages_per_hour : ℕ)
  (reads_same_amount_every_day : Prop)
  (h1 : total_hours = 10)
  (h2 : total_days = 5)
  (h3 : pages_per_hour = 50)
  (h4 : reads_same_amount_every_day) :
  (total_hours / total_days) * (pages_per_hour * 7) = 700 :=
by
  -- Begin and skip proof with sorry
  sorry

end NUMINAMATH_GPT_tom_reads_700_pages_in_7_days_l276_27656


namespace NUMINAMATH_GPT_not_divisible_by_n_l276_27643

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬ (n ∣ (2^n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_n_l276_27643


namespace NUMINAMATH_GPT_cone_base_radius_l276_27633

noncomputable def sector_radius : ℝ := 9
noncomputable def central_angle_deg : ℝ := 240

theorem cone_base_radius :
  let arc_length := (central_angle_deg * Real.pi * sector_radius) / 180
  let base_circumference := arc_length
  let base_radius := base_circumference / (2 * Real.pi)
  base_radius = 6 :=
by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l276_27633


namespace NUMINAMATH_GPT_solve_for_y_l276_27601

theorem solve_for_y (x y : ℝ) (h : 3 * x - 5 * y = 7) : y = (3 * x - 7) / 5 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l276_27601


namespace NUMINAMATH_GPT_a3_mul_a7_eq_36_l276_27693

-- Definition of a geometric sequence term
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions
def a (n : ℕ) : ℤ := sorry  -- Placeholder for the geometric sequence

axiom a5_eq_6 : a 5 = 6  -- Given that a_5 = 6

axiom geo_seq : geometric_sequence a  -- The sequence is geometric

-- Problem statement: Prove that a_3 * a_7 = 36
theorem a3_mul_a7_eq_36 : a 3 * a 7 = 36 :=
  sorry

end NUMINAMATH_GPT_a3_mul_a7_eq_36_l276_27693


namespace NUMINAMATH_GPT_area_excluding_hole_l276_27622

open Polynomial

theorem area_excluding_hole (x : ℝ) : 
  ((x^2 + 7) * (x^2 + 5)) - ((2 * x^2 - 3) * (x^2 - 2)) = -x^4 + 19 * x^2 + 29 :=
by
  sorry

end NUMINAMATH_GPT_area_excluding_hole_l276_27622


namespace NUMINAMATH_GPT_total_cost_calculation_l276_27667

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end NUMINAMATH_GPT_total_cost_calculation_l276_27667


namespace NUMINAMATH_GPT_tickets_used_correct_l276_27644

def ferris_wheel_rides : ℕ := 7
def bumper_car_rides : ℕ := 3
def cost_per_ride : ℕ := 5

def total_rides : ℕ := ferris_wheel_rides + bumper_car_rides
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_correct : total_tickets_used = 50 := by
  sorry

end NUMINAMATH_GPT_tickets_used_correct_l276_27644


namespace NUMINAMATH_GPT_solve_equation_l276_27618

theorem solve_equation :
  ∀ x : ℝ, (101 * x ^ 2 - 18 * x + 1) ^ 2 - 121 * x ^ 2 * (101 * x ^ 2 - 18 * x + 1) + 2020 * x ^ 4 = 0 ↔ 
    x = 1 / 18 ∨ x = 1 / 9 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_l276_27618


namespace NUMINAMATH_GPT_largest_possible_red_socks_l276_27614

theorem largest_possible_red_socks (r b : ℕ) (h1 : 0 < r) (h2 : 0 < b)
  (h3 : r + b ≤ 2500) (h4 : r > b) :
  r * (r - 1) + b * (b - 1) = 3/5 * (r + b) * (r + b - 1) → r ≤ 1164 :=
by sorry

end NUMINAMATH_GPT_largest_possible_red_socks_l276_27614


namespace NUMINAMATH_GPT_y_relationship_range_of_x_l276_27636

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end NUMINAMATH_GPT_y_relationship_range_of_x_l276_27636


namespace NUMINAMATH_GPT_min_y_value_l276_27648

noncomputable def y (x : ℝ) : ℝ := x^2 + 16 * x + 20

theorem min_y_value : ∀ (x : ℝ), x ≥ -3 → y x ≥ -19 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_min_y_value_l276_27648


namespace NUMINAMATH_GPT_problem_statement_l276_27635

theorem problem_statement 
  (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := 
    sorry

end NUMINAMATH_GPT_problem_statement_l276_27635


namespace NUMINAMATH_GPT_time_difference_l276_27628

-- Definitions for the conditions
def blocks_to_office : Nat := 12
def walk_time_per_block : Nat := 1 -- time in minutes
def bike_time_per_block : Nat := 20 / 60 -- time in minutes, converted from seconds

-- Definitions for the total times
def walk_time : Nat := blocks_to_office * walk_time_per_block
def bike_time : Nat := blocks_to_office * bike_time_per_block

-- Theorem statement
theorem time_difference : walk_time - bike_time = 8 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_time_difference_l276_27628


namespace NUMINAMATH_GPT_volume_of_soil_removal_l276_27624

theorem volume_of_soil_removal {a b m c d : ℝ} :
  (∃ (K : ℝ), K = (m / 6) * (2 * a * c + 2 * b * d + a * d + b * c)) :=
sorry

end NUMINAMATH_GPT_volume_of_soil_removal_l276_27624


namespace NUMINAMATH_GPT_reflection_points_reflection_line_l276_27681

-- Definitions of given points and line equation
def original_point : ℝ × ℝ := (2, 3)
def reflected_point : ℝ × ℝ := (8, 7)

-- Definitions of line parameters for y = mx + b
variable {m b : ℝ}

-- Statement of the reflection condition
theorem reflection_points_reflection_line : m + b = 9.5 := by
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_reflection_points_reflection_line_l276_27681


namespace NUMINAMATH_GPT_cubes_with_odd_neighbors_in_5x5x5_l276_27650

theorem cubes_with_odd_neighbors_in_5x5x5 (unit_cubes : Fin 125 → ℕ) 
  (neighbors : ∀ (i : Fin 125), Fin 125 → Prop) : ∃ n, n = 62 := 
by
  sorry

end NUMINAMATH_GPT_cubes_with_odd_neighbors_in_5x5x5_l276_27650


namespace NUMINAMATH_GPT_votes_for_Crow_l276_27689

theorem votes_for_Crow 
  (J : ℕ)
  (P V K : ℕ)
  (ε1 ε2 ε3 ε4 : ℤ)
  (h₁ : P + V = 15 + ε1)
  (h₂ : V + K = 18 + ε2)
  (h₃ : K + P = 20 + ε3)
  (h₄ : P + V + K = 59 + ε4)
  (bound₁ : |ε1| ≤ 13)
  (bound₂ : |ε2| ≤ 13)
  (bound₃ : |ε3| ≤ 13)
  (bound₄ : |ε4| ≤ 13)
  : V = 13 :=
sorry

end NUMINAMATH_GPT_votes_for_Crow_l276_27689


namespace NUMINAMATH_GPT_plant_cost_and_max_green_lily_students_l276_27674

-- Given conditions
def two_green_lily_three_spider_plants_cost (x y : ℕ) : Prop :=
  2 * x + 3 * y = 36

def one_green_lily_two_spider_plants_cost (x y : ℕ) : Prop :=
  x + 2 * y = 21

def total_students := 48

def cost_constraint (x y m : ℕ) : Prop :=
  9 * m + 6 * (48 - m) ≤ 378

-- Prove that x = 9, y = 6 and m ≤ 30
theorem plant_cost_and_max_green_lily_students :
  ∃ x y m : ℕ, two_green_lily_three_spider_plants_cost x y ∧ 
               one_green_lily_two_spider_plants_cost x y ∧ 
               cost_constraint x y m ∧ 
               x = 9 ∧ y = 6 ∧ m ≤ 30 :=
by
  sorry

end NUMINAMATH_GPT_plant_cost_and_max_green_lily_students_l276_27674


namespace NUMINAMATH_GPT_arnold_danny_age_l276_27639

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 17 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_arnold_danny_age_l276_27639


namespace NUMINAMATH_GPT_problem_inequality_l276_27664

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) : 
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l276_27664


namespace NUMINAMATH_GPT_tiles_required_for_floor_l276_27619

def tileDimensionsInFeet (width_in_inches : ℚ) (length_in_inches : ℚ) : ℚ × ℚ :=
  (width_in_inches / 12, length_in_inches / 12)

def area (length : ℚ) (width : ℚ) : ℚ :=
  length * width

noncomputable def numberOfTiles (floor_length : ℚ) (floor_width : ℚ) (tile_length : ℚ) (tile_width : ℚ) : ℚ :=
  (area floor_length floor_width) / (area tile_length tile_width)

theorem tiles_required_for_floor : numberOfTiles 10 15 (5/12) (2/3) = 540 := by
  sorry

end NUMINAMATH_GPT_tiles_required_for_floor_l276_27619


namespace NUMINAMATH_GPT_jane_spent_75_days_reading_l276_27638

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end NUMINAMATH_GPT_jane_spent_75_days_reading_l276_27638


namespace NUMINAMATH_GPT_option_C_is_different_l276_27660

def cause_and_effect_relationship (description: String) : Prop :=
  description = "A: Great teachers produce outstanding students" ∨
  description = "B: When the water level rises, the boat goes up" ∨
  description = "D: The higher you climb, the farther you see"

def not_cause_and_effect_relationship (description: String) : Prop :=
  description = "C: The brighter the moon, the fewer the stars"

theorem option_C_is_different :
  ∀ (description: String),
  (not_cause_and_effect_relationship description) →
  ¬ cause_and_effect_relationship description :=
by intros description h1 h2; sorry

end NUMINAMATH_GPT_option_C_is_different_l276_27660


namespace NUMINAMATH_GPT_exists_n_for_pn_consecutive_zeros_l276_27675

theorem exists_n_for_pn_consecutive_zeros (p : ℕ) (hp : Nat.Prime p) (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, (∃ k : ℕ, (p^n) / 10^(k+m) % 10^m = 0) := sorry

end NUMINAMATH_GPT_exists_n_for_pn_consecutive_zeros_l276_27675


namespace NUMINAMATH_GPT_fractional_shaded_area_l276_27690

noncomputable def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)

theorem fractional_shaded_area :
  let a := (7 : ℚ) / 16
  let r := (1 : ℚ) / 16
  geometric_series_sum a r = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fractional_shaded_area_l276_27690


namespace NUMINAMATH_GPT_smallest_number_in_sample_l276_27692

theorem smallest_number_in_sample :
  ∀ (N : ℕ) (k : ℕ) (n : ℕ), 
  0 < k → 
  N = 80 → 
  k = 5 →
  n = 42 →
  ∃ (a : ℕ), (0 ≤ a ∧ a < k) ∧
  42 = (N / k) * (42 / (N / k)) + a ∧
  ∀ (m : ℕ), (0 ≤ m ∧ m < k) → 
    (∀ (j : ℕ), (j = (N / k) * m + 10)) → 
    m = 0 → a = 10 := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_in_sample_l276_27692


namespace NUMINAMATH_GPT_decrease_by_150_percent_l276_27688

theorem decrease_by_150_percent (x : ℝ) (h : x = 80) : x - 1.5 * x = -40 :=
by
  sorry

end NUMINAMATH_GPT_decrease_by_150_percent_l276_27688


namespace NUMINAMATH_GPT_part_I_part_II_l276_27606

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

theorem part_I (α : ℝ) (hα : ∃ (P : ℝ × ℝ), P = (Real.sqrt 3, -1) ∧
  (Real.tan α = -1 / Real.sqrt 3 ∨ Real.tan α = - (Real.sqrt 3) / 3)) :
  f α = -3 := by
  sorry

theorem part_II (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  -2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l276_27606


namespace NUMINAMATH_GPT_mileage_per_gallon_l276_27697

-- Define the conditions
def miles_driven : ℝ := 100
def gallons_used : ℝ := 5

-- Define the question as a theorem to be proven
theorem mileage_per_gallon : (miles_driven / gallons_used) = 20 := by
  sorry

end NUMINAMATH_GPT_mileage_per_gallon_l276_27697


namespace NUMINAMATH_GPT_problem_1_problem_2_l276_27640

noncomputable def f (x : ℝ) : ℝ :=
  (Real.logb 3 (x / 27)) * (Real.logb 3 (3 * x))

theorem problem_1 (h₁ : 1 / 27 ≤ x)
(h₂ : x ≤ 1 / 9) :
    (∀ x, f x ≤ 12) ∧ (∃ x, f x = 5) := 
sorry

theorem problem_2
(m α β : ℝ)
(h₁ : f α + m = 0)
(h₂ : f β + m = 0) :
    α * β = 9 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l276_27640


namespace NUMINAMATH_GPT_number_of_days_woman_weaves_l276_27646

theorem number_of_days_woman_weaves
  (a_1 : ℝ) (a_n : ℝ) (S_n : ℝ) (n : ℝ)
  (h1 : a_1 = 5)
  (h2 : a_n = 1)
  (h3 : S_n = 90)
  (h4 : S_n = n * (a_1 + a_n) / 2) :
  n = 30 :=
by
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_number_of_days_woman_weaves_l276_27646


namespace NUMINAMATH_GPT_spinner_probability_l276_27687

theorem spinner_probability :
  let p_A := (1 / 4)
  let p_B := (1 / 3)
  let p_C := (5 / 12)
  let p_D := 1 - (p_A + p_B + p_C)
  p_D = 0 :=
by
  sorry

end NUMINAMATH_GPT_spinner_probability_l276_27687


namespace NUMINAMATH_GPT_value_of_N_l276_27629

theorem value_of_N (a b c N : ℚ) (h1 : a + b + c = 120) (h2 : a - 10 = N) (h3 : 10 * b = N) (h4 : c - 10 = N) : N = 1100 / 21 := 
sorry

end NUMINAMATH_GPT_value_of_N_l276_27629


namespace NUMINAMATH_GPT_intersection_value_l276_27658

theorem intersection_value (x y : ℝ) (h₁ : y = 10 / (x^2 + 5)) (h₂ : x + 2 * y = 5) : 
  x = 1 :=
sorry

end NUMINAMATH_GPT_intersection_value_l276_27658


namespace NUMINAMATH_GPT_min_distance_point_curve_to_line_l276_27695

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem min_distance_point_curve_to_line :
  ∀ (P : ℝ × ℝ), 
  curve P.1 = P.2 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_point_curve_to_line_l276_27695


namespace NUMINAMATH_GPT_greatest_y_least_y_greatest_integer_y_l276_27666

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end NUMINAMATH_GPT_greatest_y_least_y_greatest_integer_y_l276_27666


namespace NUMINAMATH_GPT_smallest_value_of_a_l276_27657

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end NUMINAMATH_GPT_smallest_value_of_a_l276_27657


namespace NUMINAMATH_GPT_inverse_proportion_function_l276_27679

theorem inverse_proportion_function (x y : ℝ) (h : y = 6 / x) : x * y = 6 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_function_l276_27679


namespace NUMINAMATH_GPT_houston_firewood_l276_27670

theorem houston_firewood (k e h : ℕ) (k_collected : k = 10) (e_collected : e = 13) (total_collected : k + e + h = 35) : h = 12 :=
by
  sorry

end NUMINAMATH_GPT_houston_firewood_l276_27670


namespace NUMINAMATH_GPT_quadratic_passes_through_l276_27630

def quadratic_value_at_point (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_passes_through (a b c : ℝ) :
  quadratic_value_at_point a b c 1 = 5 ∧ 
  quadratic_value_at_point a b c 3 = n ∧ 
  a * (-2)^2 + b * (-2) + c = -8 ∧ 
  (-4*a + b = 0) → 
  n = 253/9 := 
sorry

end NUMINAMATH_GPT_quadratic_passes_through_l276_27630


namespace NUMINAMATH_GPT_nested_fraction_evaluation_l276_27610

theorem nested_fraction_evaluation : 
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))))) = (21 / 55) :=
by
  sorry

end NUMINAMATH_GPT_nested_fraction_evaluation_l276_27610


namespace NUMINAMATH_GPT_point_on_x_axis_l276_27672

theorem point_on_x_axis (m : ℝ) (h : (m, m - 1).snd = 0) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l276_27672


namespace NUMINAMATH_GPT_samples_from_workshop_l276_27694

theorem samples_from_workshop (T S P : ℕ) (hT : T = 2048) (hS : S = 128) (hP : P = 256) : 
  (s : ℕ) → (s : ℕ) = (256 * 128 / 2048) → s = 16 :=
by
  intros s hs
  rw [Nat.div_eq (256 * 128) 2048] at hs
  sorry

end NUMINAMATH_GPT_samples_from_workshop_l276_27694


namespace NUMINAMATH_GPT_katie_total_expenditure_l276_27665

-- Define the conditions
def flower_cost : ℕ := 6
def roses_bought : ℕ := 5
def daisies_bought : ℕ := 5

-- Define the total flowers bought
def total_flowers_bought : ℕ := roses_bought + daisies_bought

-- Calculate the total cost
def total_cost (flower_cost : ℕ) (total_flowers_bought : ℕ) : ℕ :=
  total_flowers_bought * flower_cost

-- Prove that Katie spent 60 dollars
theorem katie_total_expenditure : total_cost flower_cost total_flowers_bought = 60 := sorry

end NUMINAMATH_GPT_katie_total_expenditure_l276_27665


namespace NUMINAMATH_GPT_one_greater_others_less_l276_27613

theorem one_greater_others_less {a b c : ℝ} (h1 : a > 0 ∧ b > 0 ∧ c > 0) (h2 : a * b * c = 1) (h3 : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
by
  sorry

end NUMINAMATH_GPT_one_greater_others_less_l276_27613


namespace NUMINAMATH_GPT_jameson_badminton_medals_l276_27678

theorem jameson_badminton_medals :
  ∃ (b : ℕ),  (∀ (t s : ℕ), t = 5 → s = 2 * t → t + s + b = 20) ∧ b = 5 :=
by {
sorry
}

end NUMINAMATH_GPT_jameson_badminton_medals_l276_27678


namespace NUMINAMATH_GPT_intersection_of_sets_l276_27604

theorem intersection_of_sets :
  let A := {1, 2}
  let B := {x : ℝ | x^2 - 3 * x + 2 = 0}
  A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l276_27604


namespace NUMINAMATH_GPT_expand_binomials_l276_27602

theorem expand_binomials (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_binomials_l276_27602


namespace NUMINAMATH_GPT_Ming_initial_ladybugs_l276_27607

-- Define the conditions
def Sami_spiders : Nat := 3
def Hunter_ants : Nat := 12
def insects_remaining : Nat := 21
def ladybugs_flew_away : Nat := 2

-- Formalize the proof problem
theorem Ming_initial_ladybugs : Sami_spiders + Hunter_ants + (insects_remaining + ladybugs_flew_away) - (Sami_spiders + Hunter_ants) = 8 := by
  sorry

end NUMINAMATH_GPT_Ming_initial_ladybugs_l276_27607


namespace NUMINAMATH_GPT_scientific_notation_50000000000_l276_27698

theorem scientific_notation_50000000000 :
  ∃ (a : ℝ) (n : ℤ), 50000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ (a = 5.0 ∨ a = 5) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_50000000000_l276_27698


namespace NUMINAMATH_GPT_two_card_draw_probability_l276_27612

open ProbabilityTheory

def card_values (card : ℕ) : ℕ :=
  if card = 1 ∨ card = 11 ∨ card = 12 ∨ card = 13 then 10 else card

def deck_size := 52

def total_prob : ℚ :=
  let cards := (1, deck_size)
  let case_1 := (card_values 6 * card_values 9 / (deck_size * (deck_size - 1))) + 
                (card_values 7 * card_values 8 / (deck_size * (deck_size - 1)))
  let case_2 := (3 * 4 / (deck_size * (deck_size - 1))) + 
                (4 * 3 / (deck_size * (deck_size - 1)))
  case_1 + case_2

theorem two_card_draw_probability :
  total_prob = 16 / 331 :=
by
  sorry

end NUMINAMATH_GPT_two_card_draw_probability_l276_27612


namespace NUMINAMATH_GPT_three_a_greater_three_b_l276_27649

variable (a b : ℝ)

theorem three_a_greater_three_b (h : a > b) : 3 * a > 3 * b :=
  sorry

end NUMINAMATH_GPT_three_a_greater_three_b_l276_27649


namespace NUMINAMATH_GPT_min_value_of_function_l276_27625

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  ∃ x, (x > -1) ∧ (x = 0) ∧ ∀ y, (y = x + (1 / (x + 1))) → y ≥ 1 := 
sorry

end NUMINAMATH_GPT_min_value_of_function_l276_27625


namespace NUMINAMATH_GPT_problem_solution_l276_27623

theorem problem_solution (k x1 x2 y1 y2 : ℝ) 
  (h₁ : k ≠ 0) 
  (h₂ : y1 = k * x1) 
  (h₃ : y1 = -5 / x1) 
  (h₄ : y2 = k * x2) 
  (h₅ : y2 = -5 / x2) 
  (h₆ : x1 = -x2) 
  (h₇ : y1 = -y2) : 
  x1 * y2 - 3 * x2 * y1 = 10 := 
sorry

end NUMINAMATH_GPT_problem_solution_l276_27623


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l276_27634

noncomputable def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem area_of_triangle_ABC (A B C O : ℝ × ℝ)
  (h_isosceles_right : ∃ d: ℝ, distance A B = d ∧ distance A C = d ∧ distance B C = Real.sqrt (2 * d^2))
  (h_A_right : A = (0, 0))
  (h_OA : distance O A = 5)
  (h_OB : distance O B = 7)
  (h_OC : distance O C = 3) :
  ∃ S : ℝ, S = (29 / 2) + (5 / 2) * Real.sqrt 17 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l276_27634


namespace NUMINAMATH_GPT_pairs_count_l276_27677

noncomputable def count_pairs (n : ℕ) : ℕ :=
  3^n

theorem pairs_count (A : Finset ℕ) (h : A.card = n) :
  ∃ f : Finset ℕ × Finset ℕ → Finset ℕ, ∀ B C, (B ≠ ∅ ∧ B ⊆ C ∧ C ⊆ A) → (f (B, C)).card = count_pairs n :=
sorry

end NUMINAMATH_GPT_pairs_count_l276_27677


namespace NUMINAMATH_GPT_production_profit_range_l276_27699

theorem production_profit_range (x : ℝ) (t : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) (h3 : 0 ≤ t) :
  (200 * (5 * x + 1 - 3 / x) ≥ 3000) → (3 ≤ x ∧ x ≤ 10) :=
sorry

end NUMINAMATH_GPT_production_profit_range_l276_27699


namespace NUMINAMATH_GPT_old_supervisor_salary_correct_l276_27685

def old_supervisor_salary (W S_old : ℝ) : Prop :=
  let avg_old := (W + S_old) / 9
  let avg_new := (W + 510) / 9
  avg_old = 430 ∧ avg_new = 390 → S_old = 870

theorem old_supervisor_salary_correct (W : ℝ) :
  old_supervisor_salary W 870 :=
by
  unfold old_supervisor_salary
  intro h
  sorry

end NUMINAMATH_GPT_old_supervisor_salary_correct_l276_27685


namespace NUMINAMATH_GPT_total_number_of_students_l276_27652

theorem total_number_of_students 
    (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat) 
    (h1 : group1 = 5) (h2 : group2 = 8) (h3 : group3 = 7) (h4 : group4 = 4) : 
    group1 + group2 + group3 + group4 = 24 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_students_l276_27652


namespace NUMINAMATH_GPT_minimum_soldiers_to_add_l276_27600

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_soldiers_to_add_l276_27600


namespace NUMINAMATH_GPT_sum_of_ages_l276_27696

theorem sum_of_ages (X_c Y_c : ℕ) (h1 : X_c = 45) 
  (h2 : X_c - 3 = 2 * (Y_c - 3)) : 
  (X_c + 7) + (Y_c + 7) = 83 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l276_27696


namespace NUMINAMATH_GPT_number_100_in_row_15_l276_27645

theorem number_100_in_row_15 (A : ℕ) (H1 : 1 ≤ A)
  (H2 : ∀ n : ℕ, n > 0 → n ≤ 100 * A)
  (H3 : ∃ k : ℕ, 4 * A + 1 ≤ 31 ∧ 31 ≤ 5 * A ∧ k = 5):
  ∃ r : ℕ, (14 * A + 1 ≤ 100 ∧ 100 ≤ 15 * A ∧ r = 15) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_100_in_row_15_l276_27645


namespace NUMINAMATH_GPT_sale_in_fifth_month_l276_27637

-- Define the sale amounts and average sale required.
def sale_first_month : ℕ := 7435
def sale_second_month : ℕ := 7920
def sale_third_month : ℕ := 7855
def sale_fourth_month : ℕ := 8230
def sale_sixth_month : ℕ := 6000
def average_sale_required : ℕ := 7500

-- State the theorem to determine the sale in the fifth month.
theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 avg : ℕ)
  (h1 : s1 = sale_first_month)
  (h2 : s2 = sale_second_month)
  (h3 : s3 = sale_third_month)
  (h4 : s4 = sale_fourth_month)
  (h6 : s6 = sale_sixth_month)
  (havg : avg = average_sale_required) :
  s1 + s2 + s3 + s4 + s6 + x = 6 * avg →
  x = 7560 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l276_27637


namespace NUMINAMATH_GPT_function_neither_even_nor_odd_l276_27626

noncomputable def f (x : ℝ) : ℝ := (4 * x ^ 3 - 3) / (x ^ 6 + 2)

theorem function_neither_even_nor_odd : 
  (∀ x : ℝ, f (-x) ≠ f x) ∧ (∀ x : ℝ, f (-x) ≠ -f x) :=
by
  sorry

end NUMINAMATH_GPT_function_neither_even_nor_odd_l276_27626


namespace NUMINAMATH_GPT_math_problem_l276_27673

theorem math_problem : (-4)^2 * ((-1)^2023 + (3 / 4) + (-1 / 2)^3) = -6 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l276_27673


namespace NUMINAMATH_GPT_five_alpha_plus_two_beta_is_45_l276_27617

theorem five_alpha_plus_two_beta_is_45
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (tan_β : Real.tan β = 3 / 79) :
  5 * α + 2 * β = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_five_alpha_plus_two_beta_is_45_l276_27617


namespace NUMINAMATH_GPT_tax_rate_correct_l276_27631

/-- The tax rate in dollars per $100.00 is $82.00, given that the tax rate as a percent is 82%. -/
theorem tax_rate_correct (x : ℝ) (h : x = 82) : (x / 100) * 100 = 82 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_tax_rate_correct_l276_27631


namespace NUMINAMATH_GPT_harry_walks_9_dogs_on_thursday_l276_27609

-- Define the number of dogs Harry walks on specific days
def dogs_monday : Nat := 7
def dogs_wednesday : Nat := 7
def dogs_friday : Nat := 7
def dogs_tuesday : Nat := 12

-- Define the payment per dog
def payment_per_dog : Nat := 5

-- Define total weekly earnings
def total_weekly_earnings : Nat := 210

-- Define the number of dogs Harry walks on Thursday
def dogs_thursday : Nat := 9

-- Define the total earnings for Monday, Wednesday, Friday, and Tuesday
def earnings_first_four_days : Nat := (dogs_monday + dogs_wednesday + dogs_friday + dogs_tuesday) * payment_per_dog

-- Now we state the theorem that we need to prove
theorem harry_walks_9_dogs_on_thursday :
  (total_weekly_earnings - earnings_first_four_days) / payment_per_dog = dogs_thursday :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_harry_walks_9_dogs_on_thursday_l276_27609


namespace NUMINAMATH_GPT_sin_double_angle_of_tan_l276_27641

-- Given condition: tan(alpha) = 2
-- To prove: sin(2 * alpha) = 4/5
theorem sin_double_angle_of_tan (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
  sorry

end NUMINAMATH_GPT_sin_double_angle_of_tan_l276_27641


namespace NUMINAMATH_GPT_outfits_count_l276_27611

def num_outfits (n : Nat) (total_colors : Nat) : Nat :=
  let total_combinations := n * n * n
  let undesirable_combinations := total_colors
  total_combinations - undesirable_combinations

theorem outfits_count : num_outfits 5 5 = 120 :=
  by
  sorry

end NUMINAMATH_GPT_outfits_count_l276_27611


namespace NUMINAMATH_GPT_annual_interest_rate_is_approx_14_87_percent_l276_27654

-- Let P be the principal amount, r the annual interest rate, and n the number of years
-- Given: A = P(1 + r)^n, where A is the amount of money after n years
-- In this problem: A = 2P, n = 5

theorem annual_interest_rate_is_approx_14_87_percent
    (P : Real) (r : Real) (n : Real) (A : Real) (condition1 : n = 5)
    (condition2 : A = 2 * P)
    (condition3 : A = P * (1 + r)^n) :
  r = 2^(1/5) - 1 := 
  sorry

end NUMINAMATH_GPT_annual_interest_rate_is_approx_14_87_percent_l276_27654


namespace NUMINAMATH_GPT_juice_spilled_l276_27668

def initial_amount := 1.0
def Youngin_drank := 0.1
def Narin_drank := Youngin_drank + 0.2
def remaining_amount := 0.3

theorem juice_spilled :
  initial_amount - (Youngin_drank + Narin_drank) - remaining_amount = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_juice_spilled_l276_27668
