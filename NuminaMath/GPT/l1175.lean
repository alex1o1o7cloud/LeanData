import Mathlib

namespace NUMINAMATH_GPT_number_of_refills_l1175_117531

variable (totalSpent costPerRefill : ℕ)
variable (h1 : totalSpent = 40)
variable (h2 : costPerRefill = 10)

theorem number_of_refills (h1 h2 : totalSpent = 40) (h2 : costPerRefill = 10) :
  totalSpent / costPerRefill = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_refills_l1175_117531


namespace NUMINAMATH_GPT_rate_percent_calculation_l1175_117598

theorem rate_percent_calculation (SI P T : ℝ) (R : ℝ) : SI = 640 ∧ P = 4000 ∧ T = 2 → SI = P * R * T / 100 → R = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rate_percent_calculation_l1175_117598


namespace NUMINAMATH_GPT_biased_die_expected_value_is_neg_1_5_l1175_117509

noncomputable def biased_die_expected_value : ℚ :=
  let prob_123 := (1 / 6 : ℚ) + (1 / 6) + (1 / 6)
  let prob_456 := (1 / 2 : ℚ)
  let gain := prob_123 * 2
  let loss := prob_456 * -5
  gain + loss

theorem biased_die_expected_value_is_neg_1_5 :
  biased_die_expected_value = - (3 / 2 : ℚ) :=
by
  -- We skip the detailed proof steps here.
  sorry

end NUMINAMATH_GPT_biased_die_expected_value_is_neg_1_5_l1175_117509


namespace NUMINAMATH_GPT_geometric_sequence_a5_eq_neg1_l1175_117549

-- Definitions for the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def roots_of_quadratic (a3 a7 : ℝ) : Prop :=
  a3 + a7 = -4 ∧ a3 * a7 = 1

-- The statement to prove
theorem geometric_sequence_a5_eq_neg1 {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_roots : roots_of_quadratic (a 3) (a 7)) :
  a 5 = -1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_eq_neg1_l1175_117549


namespace NUMINAMATH_GPT_no_adjacent_teachers_l1175_117560

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

theorem no_adjacent_teachers (students teachers : ℕ)
  (h_students : students = 4)
  (h_teachers : teachers = 3) :
  ∃ (arrangements : ℕ), arrangements = (factorial students) * (permutation (students + 1) teachers) :=
by
  sorry

end NUMINAMATH_GPT_no_adjacent_teachers_l1175_117560


namespace NUMINAMATH_GPT_find_a_l1175_117542

theorem find_a (x : ℝ) (a : ℝ)
  (h1 : 3 * x - 4 = a)
  (h2 : (x + a) / 3 = 1)
  (h3 : (x = (a + 4) / 3) → (x = 3 - a → ((a + 4) / 3 = 2 * (3 - a)))) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1175_117542


namespace NUMINAMATH_GPT_regular_polygon_interior_angle_160_l1175_117564

theorem regular_polygon_interior_angle_160 (n : ℕ) (h : 160 * n = 180 * (n - 2)) : n = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_regular_polygon_interior_angle_160_l1175_117564


namespace NUMINAMATH_GPT_part1_part2_l1175_117508

noncomputable def f (x a : ℝ) := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

noncomputable def g (x a : ℝ) := f x a + Real.log (x + 1) + 1/2 * x

theorem part1 (a : ℝ) (x : ℝ) (h : x > 0) : 
  (a ≤ 2 → ∀ x, g x a > 0) ∧ 
  (a > 2 → ∀ x, x < Real.exp (a - 2) - 1 → g x a < 0) ∧
  (a > 2 → ∀ x, x > Real.exp (a - 2) - 1 → g x a > 0) :=
sorry

theorem part2 (a : ℤ) : 
  (∃ x ≥ 0, f x a < 0) → a ≥ 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1175_117508


namespace NUMINAMATH_GPT_driving_time_eqn_l1175_117596

open Nat

-- Define the variables and constants
def avg_speed_before := 80 -- km/h
def stop_time := 1 / 3 -- hour
def avg_speed_after := 100 -- km/h
def total_distance := 250 -- km
def total_time := 3 -- hours

variable (t : ℝ) -- the time in hours before the stop

-- State the main theorem
theorem driving_time_eqn :
  avg_speed_before * t + avg_speed_after * (total_time - stop_time - t) = total_distance := by
  sorry

end NUMINAMATH_GPT_driving_time_eqn_l1175_117596


namespace NUMINAMATH_GPT_least_positive_x_multiple_l1175_117583

theorem least_positive_x_multiple (x : ℕ) : 
  (∃ k : ℕ, (2 * x + 41) = 53 * k) → 
  x = 6 :=
sorry

end NUMINAMATH_GPT_least_positive_x_multiple_l1175_117583


namespace NUMINAMATH_GPT_find_widgets_l1175_117597

theorem find_widgets (a b c d e f : ℕ) : 
  (3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255) →
  (3 ^ a * 11 ^ b * 5 ^ c * 7 ^ d * 13 ^ e * 17 ^ f = 351125648000) →
  c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_widgets_l1175_117597


namespace NUMINAMATH_GPT_unit_digit_product_is_zero_l1175_117551

-- Definitions based on conditions in (a)
def a_1 := 6245
def a_2 := 7083
def a_3 := 9137
def a_4 := 4631
def a_5 := 5278
def a_6 := 3974

-- Helper function to get the unit digit of a number
def unit_digit (n : Nat) : Nat := n % 10

-- Main theorem to prove
theorem unit_digit_product_is_zero :
  unit_digit (a_1 * a_2 * a_3 * a_4 * a_5 * a_6) = 0 := by
  sorry

end NUMINAMATH_GPT_unit_digit_product_is_zero_l1175_117551


namespace NUMINAMATH_GPT_triangle_internal_angles_external_angle_theorem_l1175_117554

theorem triangle_internal_angles {A B C : ℝ}
 (mA : A = 64) (mB : B = 33) (mC_ext : C = 120) :
  180 - A - B = 83 :=
by
  sorry

theorem external_angle_theorem {A C D : ℝ}
 (mA : A = 64) (mC_ext : C = 120) :
  C = A + D → D = 56 :=
by
  sorry

end NUMINAMATH_GPT_triangle_internal_angles_external_angle_theorem_l1175_117554


namespace NUMINAMATH_GPT_birdhouse_volume_difference_l1175_117579

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_birdhouse_volume_difference_l1175_117579


namespace NUMINAMATH_GPT_initial_population_l1175_117529

-- Define the initial population
variable (P : ℝ)

-- Define the conditions
theorem initial_population
  (h1 : P * 1.25 * 0.8 * 1.1 * 0.85 * 1.3 + 150 = 25000) :
  P = 24850 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l1175_117529


namespace NUMINAMATH_GPT_proportional_value_l1175_117559

theorem proportional_value :
  ∃ (x : ℝ), 18 / 60 / (12 / 60) = x / 6 ∧ x = 9 := sorry

end NUMINAMATH_GPT_proportional_value_l1175_117559


namespace NUMINAMATH_GPT_contrapositive_property_l1175_117555

def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def is_divisible_by_2 (n : ℤ) : Prop := n % 2 = 0

theorem contrapositive_property :
  (∀ n : ℤ, is_divisible_by_6 n → is_divisible_by_2 n) ↔ (∀ n : ℤ, ¬ is_divisible_by_2 n → ¬ is_divisible_by_6 n) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_property_l1175_117555


namespace NUMINAMATH_GPT_trig_identity_l1175_117557

theorem trig_identity : 2 * Real.sin (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 1 = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1175_117557


namespace NUMINAMATH_GPT_smallest_number_satisfying_conditions_l1175_117501

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n % 6 = 2 ∧ n % 7 = 3 ∧ n % 8 = 4 ∧ ∀ m, (m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) :=
  sorry

end NUMINAMATH_GPT_smallest_number_satisfying_conditions_l1175_117501


namespace NUMINAMATH_GPT_solve_fabric_price_l1175_117545

-- Defining the variables
variables (x y : ℕ)

-- Conditions as hypotheses
def condition1 := 7 * x = 9 * y
def condition2 := x - y = 36

-- Theorem statement to prove the system of equations
theorem solve_fabric_price (h1 : condition1 x y) (h2 : condition2 x y) :
  (7 * x = 9 * y) ∧ (x - y = 36) :=
by
  -- No proof is provided
  sorry

end NUMINAMATH_GPT_solve_fabric_price_l1175_117545


namespace NUMINAMATH_GPT_apples_for_juice_l1175_117567

def totalApples : ℝ := 6
def exportPercentage : ℝ := 0.25
def juicePercentage : ℝ := 0.60

theorem apples_for_juice : 
  let remainingApples := totalApples * (1 - exportPercentage)
  let applesForJuice := remainingApples * juicePercentage
  applesForJuice = 2.7 :=
by
  sorry

end NUMINAMATH_GPT_apples_for_juice_l1175_117567


namespace NUMINAMATH_GPT_cross_area_l1175_117533

variables (R : ℝ) (A : ℝ × ℝ) (φ : ℝ)
  -- Radius R of the circle, Point A inside the circle, and angle φ in radians

-- Define the area of the cross formed by rotated lines
def area_of_cross (R : ℝ) (φ : ℝ) : ℝ :=
  2 * φ * R^2

theorem cross_area (R : ℝ) (A : ℝ × ℝ) (φ : ℝ) (hR : 0 < R) (hA : dist A (0, 0) < R) :
  area_of_cross R φ = 2 * φ * R^2 := 
sorry

end NUMINAMATH_GPT_cross_area_l1175_117533


namespace NUMINAMATH_GPT_simplify_expression_l1175_117578

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) : 
  (x^2 - x) / (x^2 - 2 * x + 1) = 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1175_117578


namespace NUMINAMATH_GPT_number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l1175_117547

-- Defining the conditions
def wooden_boards_type_A := 400
def wooden_boards_type_B := 500
def desk_needs_type_A := 2
def desk_needs_type_B := 1
def chair_needs_type_A := 1
def chair_needs_type_B := 2
def total_students := 30
def desk_assembly_time := 10
def chair_assembly_time := 7

-- Theorem for the number of assembled desks and chairs
theorem number_of_assembled_desks_and_chairs :
  ∃ x y : ℕ, 2 * x + y = wooden_boards_type_A ∧ x + 2 * y = wooden_boards_type_B ∧ x = 100 ∧ y = 200 :=
by {
  sorry
}

-- Theorem for the feasibility of students completing the tasks simultaneously
theorem students_cannot_complete_tasks_simultaneously :
  ¬ ∃ a : ℕ, (a ≤ total_students) ∧ (total_students - a > 0) ∧ 
  (100 / a) * desk_assembly_time = (200 / (total_students - a)) * chair_assembly_time :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l1175_117547


namespace NUMINAMATH_GPT_probability_of_matching_pair_l1175_117513

def total_socks : ℕ := 12 + 6 + 9
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

def black_pairs : ℕ := choose_two 12
def white_pairs : ℕ := choose_two 6
def blue_pairs : ℕ := choose_two 9

def total_pairs : ℕ := choose_two total_socks
def matching_pairs : ℕ := black_pairs + white_pairs + blue_pairs

def probability : ℚ := matching_pairs / total_pairs

theorem probability_of_matching_pair :
  probability = 1 / 3 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_probability_of_matching_pair_l1175_117513


namespace NUMINAMATH_GPT_one_third_of_nine_times_seven_l1175_117550

theorem one_third_of_nine_times_seven : (1 / 3) * (9 * 7) = 21 := 
by
  sorry

end NUMINAMATH_GPT_one_third_of_nine_times_seven_l1175_117550


namespace NUMINAMATH_GPT_defect_rate_probability_l1175_117537

theorem defect_rate_probability (p : ℝ) (n : ℕ) (ε : ℝ) (q : ℝ) : 
  p = 0.02 →
  n = 800 →
  ε = 0.01 →
  q = 1 - p →
  1 - (p * q) / (n * ε^2) = 0.755 :=
by
  intro hp hn he hq
  rw [hp, hn, he, hq]
  -- Calculation steps can be verified here
  sorry

end NUMINAMATH_GPT_defect_rate_probability_l1175_117537


namespace NUMINAMATH_GPT_system_of_equations_solution_l1175_117576

theorem system_of_equations_solution :
  ∃ x y : ℝ, (x + y = 3) ∧ (2 * x - 3 * y = 1) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1175_117576


namespace NUMINAMATH_GPT_inequality_solution_l1175_117585

theorem inequality_solution (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1175_117585


namespace NUMINAMATH_GPT_winston_initial_quarters_l1175_117538

-- Defining the conditions
def spent_candy := 50 -- 50 cents spent on candy
def remaining_cents := 300 -- 300 cents left

-- Defining the value of a quarter in cents
def value_of_quarter := 25

-- Calculating the number of quarters Winston initially had
def initial_quarters := (spent_candy + remaining_cents) / value_of_quarter

-- Proof statement
theorem winston_initial_quarters : initial_quarters = 14 := 
by sorry

end NUMINAMATH_GPT_winston_initial_quarters_l1175_117538


namespace NUMINAMATH_GPT_scientific_notation_of_0_000000032_l1175_117536

theorem scientific_notation_of_0_000000032 :
  0.000000032 = 3.2 * 10^(-8) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_000000032_l1175_117536


namespace NUMINAMATH_GPT_rectangle_area_eq_l1175_117510

theorem rectangle_area_eq (a b c d x y z w : ℝ)
  (h1 : a = x + y) (h2 : b = y + z) (h3 : c = z + w) (h4 : d = w + x) :
  a + c = b + d :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_eq_l1175_117510


namespace NUMINAMATH_GPT_intersection_A_B_at_1_range_of_a_l1175_117519

-- Problem definitions
def set_A (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def set_B (x a : ℝ) : Prop := x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0

-- Question (I) If a = 1, find A ∩ B
theorem intersection_A_B_at_1 : (∀ x : ℝ, set_A x ∧ set_B x 1 ↔ (1 < x ∧ x ≤ 1 + Real.sqrt 2)) := sorry

-- Question (II) If A ∩ B contains exactly one integer, find the range of a.
theorem range_of_a (h : ∃ x : ℤ, set_A x ∧ set_B x 2) : 3 / 4 ≤ 2 ∧ 2 < 4 / 3 := sorry

end NUMINAMATH_GPT_intersection_A_B_at_1_range_of_a_l1175_117519


namespace NUMINAMATH_GPT_larger_segment_length_l1175_117518

theorem larger_segment_length (a b c : ℕ) (h : ℝ) (x : ℝ)
  (ha : a = 50) (hb : b = 90) (hc : c = 110)
  (hyp1 : a^2 = x^2 + h^2)
  (hyp2 : b^2 = (c - x)^2 + h^2) :
  110 - x = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_segment_length_l1175_117518


namespace NUMINAMATH_GPT_solve_inequality_l1175_117552

noncomputable def solution_set (a b : ℝ) (x : ℝ) : Prop :=
x < -1 / b ∨ x > 1 / a

theorem solve_inequality (a b : ℝ) (x : ℝ)
  (h_a : a > 0) (h_b : b > 0) :
  (-b < 1 / x ∧ 1 / x < a) ↔ solution_set a b x :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1175_117552


namespace NUMINAMATH_GPT_arithmetic_square_root_16_l1175_117511

theorem arithmetic_square_root_16 : ∃ (x : ℝ), x * x = 16 ∧ x ≥ 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_16_l1175_117511


namespace NUMINAMATH_GPT_rate_up_the_mountain_l1175_117507

noncomputable def mountain_trip_rate (R : ℝ) : ℝ := 1.5 * R

theorem rate_up_the_mountain : 
  ∃ R : ℝ, (2 * 1.5 * R = 18) ∧ (1.5 * R = 9) → R = 6 :=
by
  sorry

end NUMINAMATH_GPT_rate_up_the_mountain_l1175_117507


namespace NUMINAMATH_GPT_soccer_ball_cost_l1175_117568

theorem soccer_ball_cost (x : ℝ) (soccer_balls basketballs : ℕ) 
  (soccer_ball_cost basketball_cost : ℝ) 
  (h1 : soccer_balls = 2 * basketballs)
  (h2 : 5000 = soccer_balls * soccer_ball_cost)
  (h3 : 4000 = basketballs * basketball_cost)
  (h4 : basketball_cost = soccer_ball_cost + 30)
  (eqn : 5000 / soccer_ball_cost = 2 * (4000 / basketball_cost)) :
  soccer_ball_cost = x :=
by
  sorry

end NUMINAMATH_GPT_soccer_ball_cost_l1175_117568


namespace NUMINAMATH_GPT_sum_of_digits_of_2010_l1175_117548

noncomputable def sum_of_base6_digits (n : ℕ) : ℕ :=
  (n.digits 6).sum

theorem sum_of_digits_of_2010 : sum_of_base6_digits 2010 = 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_2010_l1175_117548


namespace NUMINAMATH_GPT_solve_for_x_l1175_117572

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_solve_for_x_l1175_117572


namespace NUMINAMATH_GPT_total_pencils_l1175_117500

theorem total_pencils  (a b c : Nat) (total : Nat) 
(h₀ : a = 43) 
(h₁ : b = 19) 
(h₂ : c = 16) 
(h₃ : total = a + b + c) : 
total = 78 := 
by
  sorry

end NUMINAMATH_GPT_total_pencils_l1175_117500


namespace NUMINAMATH_GPT_probability_sum_5_l1175_117535

theorem probability_sum_5 :
  let total_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_probability_sum_5_l1175_117535


namespace NUMINAMATH_GPT_systematic_sampling_eighth_group_l1175_117584

theorem systematic_sampling_eighth_group (total_students : ℕ) (groups : ℕ) (group_size : ℕ)
(start_number : ℕ) (group_number : ℕ)
(h1 : total_students = 480)
(h2 : groups = 30)
(h3 : group_size = 16)
(h4 : start_number = 5)
(h5 : group_number = 8) :
  (group_number - 1) * group_size + start_number = 117 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_eighth_group_l1175_117584


namespace NUMINAMATH_GPT_trigonometric_inequality_proof_l1175_117587

theorem trigonometric_inequality_proof : 
  ∀ (sin cos : ℝ → ℝ), 
  (∀ θ, 0 ≤ θ ∧ θ ≤ π/2 → sin θ = cos (π/2 - θ)) → 
  sin (π * 11 / 180) < sin (π * 12 / 180) ∧ sin (π * 12 / 180) < sin (π * 80 / 180) :=
by 
  intros sin cos identity
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_proof_l1175_117587


namespace NUMINAMATH_GPT_eleonora_age_l1175_117543

-- Definitions
def age_eleonora (e m : ℕ) : Prop :=
m - e = 3 * (2 * e - m) ∧ 3 * e + (m + 2 * e) = 100

-- Theorem stating that Eleonora's age is 15
theorem eleonora_age (e m : ℕ) (h : age_eleonora e m) : e = 15 :=
sorry

end NUMINAMATH_GPT_eleonora_age_l1175_117543


namespace NUMINAMATH_GPT_percentage_discount_l1175_117539

theorem percentage_discount (C S S' : ℝ) (h1 : S = 1.14 * C) (h2 : S' = 2.20 * C) :
  (S' - S) / S' * 100 = 48.18 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_discount_l1175_117539


namespace NUMINAMATH_GPT_min_q_difference_l1175_117573

theorem min_q_difference (p q : ℕ) (hpq : 0 < p ∧ 0 < q) (ineq1 : (7:ℚ)/12 < p/q) (ineq2 : p/q < (5:ℚ)/8) (hmin : ∀ r s : ℕ, 0 < r ∧ 0 < s ∧ (7:ℚ)/12 < r/s ∧ r/s < (5:ℚ)/8 → q ≤ s) : q - p = 2 :=
sorry

end NUMINAMATH_GPT_min_q_difference_l1175_117573


namespace NUMINAMATH_GPT_tan_of_acute_angle_l1175_117544

theorem tan_of_acute_angle (A : ℝ) (hA1 : 0 < A ∧ A < π / 2)
  (hA2 : 4 * (Real.sin A)^2 - 4 * Real.sin A * Real.cos A + (Real.cos A)^2 = 0) :
  Real.tan A = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_acute_angle_l1175_117544


namespace NUMINAMATH_GPT_find_a_range_l1175_117592

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := 3 * Real.exp x + a

theorem find_a_range (a : ℝ) :
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x > g x a) → a < Real.exp 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l1175_117592


namespace NUMINAMATH_GPT_maria_mushrooms_l1175_117565

theorem maria_mushrooms (potatoes carrots onions green_beans bell_peppers mushrooms : ℕ) 
  (h1 : carrots = 6 * potatoes)
  (h2 : onions = 2 * carrots)
  (h3 : green_beans = onions / 3)
  (h4 : bell_peppers = 4 * green_beans)
  (h5 : mushrooms = 3 * bell_peppers)
  (h0 : potatoes = 3) : 
  mushrooms = 144 :=
by
  sorry

end NUMINAMATH_GPT_maria_mushrooms_l1175_117565


namespace NUMINAMATH_GPT_subtraction_of_fractions_l1175_117540

theorem subtraction_of_fractions : (5 / 9) - (1 / 6) = 7 / 18 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_of_fractions_l1175_117540


namespace NUMINAMATH_GPT_convert_length_convert_area_convert_time_convert_mass_l1175_117512

theorem convert_length (cm : ℕ) : cm = 7 → (cm : ℚ) / 100 = 7 / 100 :=
by sorry

theorem convert_area (dm2 : ℕ) : dm2 = 35 → (dm2 : ℚ) / 100 = 7 / 20 :=
by sorry

theorem convert_time (min : ℕ) : min = 45 → (min : ℚ) / 60 = 3 / 4 :=
by sorry

theorem convert_mass (g : ℕ) : g = 2500 → (g : ℚ) / 1000 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_convert_length_convert_area_convert_time_convert_mass_l1175_117512


namespace NUMINAMATH_GPT_thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l1175_117589

theorem thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five :
  (35 * 99 ≠ 35 * 100 + 35) :=
by
  sorry

end NUMINAMATH_GPT_thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l1175_117589


namespace NUMINAMATH_GPT_bus_trip_times_l1175_117563

/-- Given two buses traveling towards each other from points A and B which are 120 km apart.
The first bus stops for 10 minutes and the second bus stops for 5 minutes. The first bus reaches 
its destination 25 minutes before the second bus. The first bus travels 20 km/h faster than the 
second bus. Prove that the travel times for the buses are 
1 hour 40 minutes and 2 hours 5 minutes respectively. -/
theorem bus_trip_times (d : ℕ) (v1 v2 : ℝ) (t1 t2 t : ℝ) (h1 : d = 120) (h2 : v1 = v2 + 20) 
(h3 : t1 = d / v1 + 10) (h4 : t2 = d / v2 + 5) (h5 : t2 - t1 = 25) :
t1 = 100 ∧ t2 = 125 := 
by 
  sorry

end NUMINAMATH_GPT_bus_trip_times_l1175_117563


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_find_k_l1175_117561

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (k : ℝ) : 
  let a := 1
  let b := 2 * k - 1
  let c := -k - 2
  let Δ := b^2 - 4 * a * c
  (Δ > 0) :=
by
  sorry

-- Part 2: Given the roots condition, find k
theorem find_k (x1 x2 k : ℝ)
  (h1 : x1 + x2 = -(2 * k - 1))
  (h2 : x1 * x2 = -k - 2)
  (h3 : x1 + x2 - 4 * x1 * x2 = 1) : 
  k = -4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_find_k_l1175_117561


namespace NUMINAMATH_GPT_cement_mixture_weight_l1175_117520

theorem cement_mixture_weight 
  (W : ℝ)
  (h1 : W = (2/5) * W + (1/6) * W + (1/10) * W + (1/8) * W + 12) :
  W = 57.6 := by
  sorry

end NUMINAMATH_GPT_cement_mixture_weight_l1175_117520


namespace NUMINAMATH_GPT_max_value_f_at_a1_f_div_x_condition_l1175_117580

noncomputable def f (a x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem max_value_f_at_a1 :
  ∀ x : ℝ, (f 1 0) = 0 ∧ ( ∀ y : ℝ, y ≠ 0 → f 1 y < f 1 0) := 
sorry

theorem f_div_x_condition :
  ∀ x : ℝ, x ≠ 0 → (((f 1 x) / x) < 1) :=
sorry

end NUMINAMATH_GPT_max_value_f_at_a1_f_div_x_condition_l1175_117580


namespace NUMINAMATH_GPT_book_cost_is_2_l1175_117591

-- Define initial amount of money
def initial_amount : ℕ := 48

-- Define the number of books purchased
def num_books : ℕ := 5

-- Define the amount of money left after purchasing the books
def amount_left : ℕ := 38

-- Define the cost per book
def cost_per_book (initial amount_left : ℕ) (num_books : ℕ) : ℕ := (initial - amount_left) / num_books

-- The theorem to prove
theorem book_cost_is_2
    (initial_amount : ℕ := 48) 
    (amount_left : ℕ := 38) 
    (num_books : ℕ := 5) :
    cost_per_book initial_amount amount_left num_books = 2 :=
by
  sorry

end NUMINAMATH_GPT_book_cost_is_2_l1175_117591


namespace NUMINAMATH_GPT_speed_of_stream_l1175_117570

theorem speed_of_stream
  (D : ℝ) (v : ℝ)
  (h : D / (72 - v) = 2 * D / (72 + v)) :
  v = 24 := by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1175_117570


namespace NUMINAMATH_GPT_find_numbers_l1175_117595

theorem find_numbers (a b : ℝ) (h1 : a - b = 7.02) (h2 : a = 10 * b) : a = 7.8 ∧ b = 0.78 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1175_117595


namespace NUMINAMATH_GPT_price_after_discounts_l1175_117577

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * (1 - 0.10)
  let second_discount := first_discount * (1 - 0.20)
  second_discount

theorem price_after_discounts (initial_price : ℝ) (h : final_price initial_price = 174.99999999999997) : 
  final_price initial_price = 175 := 
by {
  sorry
}

end NUMINAMATH_GPT_price_after_discounts_l1175_117577


namespace NUMINAMATH_GPT_equilateral_triangle_t_gt_a_squared_l1175_117588

theorem equilateral_triangle_t_gt_a_squared {a x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ a) :
  2 * x^2 - 2 * a * x + 3 * a^2 > a^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_equilateral_triangle_t_gt_a_squared_l1175_117588


namespace NUMINAMATH_GPT_compare_logs_l1175_117593

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log (1 / 2)

theorem compare_logs : c < a ∧ a < b := by
  have h0 : a = Real.log 2 / Real.log 3 := rfl
  have h1 : b = Real.log 3 / Real.log 2 := rfl
  have h2 : c = Real.log 5 / Real.log (1 / 2) := rfl
  sorry

end NUMINAMATH_GPT_compare_logs_l1175_117593


namespace NUMINAMATH_GPT_last_four_digits_5_pow_2017_l1175_117503

theorem last_four_digits_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end NUMINAMATH_GPT_last_four_digits_5_pow_2017_l1175_117503


namespace NUMINAMATH_GPT_toy_value_l1175_117541

theorem toy_value
  (t : ℕ)                 -- total number of toys
  (W : ℕ)                 -- total worth in dollars
  (v : ℕ)                 -- value of one specific toy
  (x : ℕ)                 -- value of one of the other toys
  (h1 : t = 9)            -- condition 1: total number of toys
  (h2 : W = 52)           -- condition 2: total worth
  (h3 : v = 12)           -- condition 3: value of one specific toy
  (h4 : (t - 1) * x + v = W) -- condition 4: equation based on the problem
  : x = 5 :=              -- theorem statement: other toy's value
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_toy_value_l1175_117541


namespace NUMINAMATH_GPT_Yi_visited_city_A_l1175_117562

variable (visited : String -> String -> Prop) -- denote visited "Student" "City"
variables (Jia Yi Bing : String) (A B C : String)

theorem Yi_visited_city_A
  (h1 : visited Jia A ∧ visited Jia C ∧ ¬ visited Jia B)
  (h2 : ¬ visited Yi C)
  (h3 : visited Jia A ∧ visited Yi A ∧ visited Bing A) :
  visited Yi A :=
by
  sorry

end NUMINAMATH_GPT_Yi_visited_city_A_l1175_117562


namespace NUMINAMATH_GPT_number_of_customers_l1175_117504

theorem number_of_customers 
  (total_cartons : ℕ) 
  (damaged_cartons : ℕ) 
  (accepted_cartons : ℕ) 
  (customers : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : damaged_cartons = 60)
  (h3 : accepted_cartons = 160)
  (h_eq_per_customer : (total_cartons / customers) - damaged_cartons = accepted_cartons / customers) :
  customers = 4 :=
sorry

end NUMINAMATH_GPT_number_of_customers_l1175_117504


namespace NUMINAMATH_GPT_orthogonal_vectors_l1175_117574

open Real

variables (r s : ℝ)

def a : ℝ × ℝ × ℝ := (5, r, -3)
def b : ℝ × ℝ × ℝ := (-1, 2, s)

theorem orthogonal_vectors
  (orthogonality : 5 * (-1) + r * 2 + (-3) * s = 0)
  (magnitude_condition : 34 + r^2 = 4 * (5 + s^2)) :
  ∃ (r s : ℝ), (2 * r - 3 * s = 5) ∧ (r^2 - 4 * s^2 = -14) :=
  sorry

end NUMINAMATH_GPT_orthogonal_vectors_l1175_117574


namespace NUMINAMATH_GPT_probability_of_region_C_l1175_117514

theorem probability_of_region_C (pA pB pC : ℚ) 
  (h1 : pA = 1/2) 
  (h2 : pB = 1/5) 
  (h3 : pA + pB + pC = 1) : 
  pC = 3/10 := 
sorry

end NUMINAMATH_GPT_probability_of_region_C_l1175_117514


namespace NUMINAMATH_GPT_digit_sum_9_l1175_117558

def digits := {n : ℕ // n < 10}

theorem digit_sum_9 (a b : digits) 
  (h1 : (4 * 100) + (a.1 * 10) + 3 + 984 = (1 * 1000) + (3 * 100) + (b.1 * 10) + 7) 
  (h2 : (1 + b.1) - (3 + 7) % 11 = 0) 
: a.1 + b.1 = 9 :=
sorry

end NUMINAMATH_GPT_digit_sum_9_l1175_117558


namespace NUMINAMATH_GPT_original_rent_eq_l1175_117553

theorem original_rent_eq (R : ℝ)
  (h1 : 4 * 800 = 3200)
  (h2 : 4 * 850 = 3400)
  (h3 : 3400 - 3200 = 200)
  (h4 : 200 = 0.25 * R) : R = 800 := by
  sorry

end NUMINAMATH_GPT_original_rent_eq_l1175_117553


namespace NUMINAMATH_GPT_art_gallery_total_pieces_l1175_117521

theorem art_gallery_total_pieces :
  ∃ T : ℕ, 
    (1/3 : ℝ) * T + (2/3 : ℝ) * (1/3 : ℝ) * T + 400 + 3 * (1/18 : ℝ) * T + 2 * (1/18 : ℝ) * T = T :=
sorry

end NUMINAMATH_GPT_art_gallery_total_pieces_l1175_117521


namespace NUMINAMATH_GPT_water_formed_l1175_117599

theorem water_formed (CaOH2 CO2 CaCO3 H2O : Nat) 
  (h_balanced : ∀ n, n * CaOH2 + n * CO2 = n * CaCO3 + n * H2O)
  (h_initial : CaOH2 = 2 ∧ CO2 = 2) : 
  H2O = 2 :=
by
  sorry

end NUMINAMATH_GPT_water_formed_l1175_117599


namespace NUMINAMATH_GPT_find_x_coordinate_l1175_117534

theorem find_x_coordinate 
  (x : ℝ)
  (h1 : (0, 0) = (0, 0))
  (h2 : (0, 4) = (0, 4))
  (h3 : (x, 4) = (x, 4))
  (h4 : (x, 0) = (x, 0))
  (h5 : 0.4 * (4 * x) = 8)
  : x = 5 := 
sorry

end NUMINAMATH_GPT_find_x_coordinate_l1175_117534


namespace NUMINAMATH_GPT_bridge_toll_fees_for_annie_are_5_l1175_117569

-- Conditions
def start_fee : ℝ := 2.50
def cost_per_mile : ℝ := 0.25
def mike_miles : ℕ := 36
def annie_miles : ℕ := 16
def total_cost_mike : ℝ := start_fee + cost_per_mile * mike_miles

-- Hypothesis from conditions
axiom both_charged_same : ∀ (bridge_fees : ℝ), total_cost_mike = start_fee + cost_per_mile * annie_miles + bridge_fees

-- Proof problem
theorem bridge_toll_fees_for_annie_are_5 : ∃ (bridge_fees : ℝ), bridge_fees = 5 :=
by
  existsi 5
  sorry

end NUMINAMATH_GPT_bridge_toll_fees_for_annie_are_5_l1175_117569


namespace NUMINAMATH_GPT_pentagon_area_l1175_117516

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end NUMINAMATH_GPT_pentagon_area_l1175_117516


namespace NUMINAMATH_GPT_running_speed_is_24_l1175_117526

def walk_speed := 8 -- km/h
def walk_time := 3 -- hours
def run_time := 1 -- hour

def walk_distance := walk_speed * walk_time

def run_speed := walk_distance / run_time

theorem running_speed_is_24 : run_speed = 24 := 
by
  sorry

end NUMINAMATH_GPT_running_speed_is_24_l1175_117526


namespace NUMINAMATH_GPT_monthly_fixed_cost_is_correct_l1175_117517

-- Definitions based on the conditions in the problem
def production_cost_per_component : ℕ := 80
def shipping_cost_per_component : ℕ := 5
def components_per_month : ℕ := 150
def minimum_price_per_component : ℕ := 195

-- Monthly fixed cost definition based on the provided solution
def monthly_fixed_cost := components_per_month * (minimum_price_per_component - (production_cost_per_component + shipping_cost_per_component))

-- Theorem stating that the calculated fixed cost is correct.
theorem monthly_fixed_cost_is_correct : monthly_fixed_cost = 16500 :=
by
  unfold monthly_fixed_cost
  norm_num
  sorry

end NUMINAMATH_GPT_monthly_fixed_cost_is_correct_l1175_117517


namespace NUMINAMATH_GPT_proof_range_of_a_l1175_117522

/-- p is the proposition that for all x in [1,2], x^2 - a ≥ 0 --/
def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

/-- q is the proposition that there exists an x0 in ℝ such that x0^2 + (a-1)x0 + 1 < 0 --/
def q (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + (a-1)*x0 + 1 < 0

theorem proof_range_of_a (a : ℝ) : (p a ∨ q a) ∧ (¬p a ∧ ¬q a) → (a ≥ -1 ∧ a ≤ 1) ∨ a > 3 :=
by
  sorry -- proof will be filled out here

end NUMINAMATH_GPT_proof_range_of_a_l1175_117522


namespace NUMINAMATH_GPT_time_45_minutes_after_10_20_is_11_05_l1175_117546

def time := Nat × Nat -- Represents time as (hours, minutes)

noncomputable def add_minutes (t : time) (m : Nat) : time :=
  let (hours, minutes) := t
  let total_minutes := minutes + m
  let new_hours := hours + total_minutes / 60
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_45_minutes_after_10_20_is_11_05 :
  add_minutes (10, 20) 45 = (11, 5) :=
  sorry

end NUMINAMATH_GPT_time_45_minutes_after_10_20_is_11_05_l1175_117546


namespace NUMINAMATH_GPT_mike_cards_remaining_l1175_117575

-- Define initial condition
def mike_initial_cards : ℕ := 87

-- Define the cards bought by Sam
def sam_bought_cards : ℕ := 13

-- Define the expected remaining cards
def mike_final_cards := mike_initial_cards - sam_bought_cards

-- Theorem to prove the final count of Mike's baseball cards
theorem mike_cards_remaining : mike_final_cards = 74 := by
  sorry

end NUMINAMATH_GPT_mike_cards_remaining_l1175_117575


namespace NUMINAMATH_GPT_geom_cos_sequence_l1175_117582

open Real

theorem geom_cos_sequence (b : ℝ) (hb : 0 < b ∧ b < 360) (h : cos (2*b) / cos b = cos (3*b) / cos (2*b)) : b = 180 :=
by
  sorry

end NUMINAMATH_GPT_geom_cos_sequence_l1175_117582


namespace NUMINAMATH_GPT_complex_exp_power_cos_angle_l1175_117523

theorem complex_exp_power_cos_angle (z : ℂ) (h : z + 1/z = 2 * Complex.cos (Real.pi / 36)) :
    z^1000 + 1/(z^1000) = 2 * Complex.cos (Real.pi * 2 / 9) :=
by
  sorry

end NUMINAMATH_GPT_complex_exp_power_cos_angle_l1175_117523


namespace NUMINAMATH_GPT_solve_quadratic_l1175_117515

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l1175_117515


namespace NUMINAMATH_GPT_base_angle_isosceles_triangle_l1175_117590

theorem base_angle_isosceles_triangle
  (sum_angles : ∀ (α β γ : ℝ), α + β + γ = 180)
  (isosceles : ∀ (α β : ℝ), α = β)
  (one_angle_forty : ∃ α : ℝ, α = 40) :
  ∃ β : ℝ, β = 70 ∨ β = 40 :=
by
  sorry

end NUMINAMATH_GPT_base_angle_isosceles_triangle_l1175_117590


namespace NUMINAMATH_GPT_number_of_cows_l1175_117571

variable {D C : ℕ}

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 :=
by sorry

end NUMINAMATH_GPT_number_of_cows_l1175_117571


namespace NUMINAMATH_GPT_simple_interest_proof_l1175_117581

def simple_interest (P R T: ℝ) : ℝ :=
  P * R * T

theorem simple_interest_proof :
  simple_interest 810 (4.783950617283951 / 100) 4 = 154.80 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_proof_l1175_117581


namespace NUMINAMATH_GPT_ratio_of_two_numbers_l1175_117530

variable {a b : ℝ}

theorem ratio_of_two_numbers
  (h1 : a + b = 7 * (a - b))
  (h2 : 0 < b)
  (h3 : a > b) :
  a / b = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_two_numbers_l1175_117530


namespace NUMINAMATH_GPT_rabbit_count_l1175_117566

theorem rabbit_count (r1 r2 : ℕ) (h1 : r1 = 8) (h2 : r2 = 5) : r1 + r2 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_rabbit_count_l1175_117566


namespace NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l1175_117532

variable (a1 a15 : ℚ)
variable (n : ℕ) (a7 : ℚ)

-- Given conditions
def first_term (a1 : ℚ) : Prop := a1 = 3
def last_term (a15 : ℚ) : Prop := a15 = 72
def total_terms (n : ℕ) : Prop := n = 15

-- Arithmetic sequence formula
def common_difference (d : ℚ) : Prop := d = (72 - 3) / (15 - 1)
def nth_term (a_n : ℚ) (a1 : ℚ) (n : ℕ) (d : ℚ) : Prop := a_n = a1 + (n - 1) * d

-- Prove that the 7th term is approximately 33
theorem arithmetic_sequence_seventh_term :
  ∀ (a1 a15 : ℚ) (n : ℕ), first_term a1 → last_term a15 → total_terms n → ∃ a7 : ℚ, 
  nth_term a7 a1 7 ((a15 - a1) / (n - 1)) ∧ (33 - 0.5) < a7 ∧ a7 < (33 + 0.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l1175_117532


namespace NUMINAMATH_GPT_class_ratio_and_percentage_l1175_117524

theorem class_ratio_and_percentage:
  ∀ (female male : ℕ), female = 15 → male = 25 →
  (∃ ratio_n ratio_d : ℕ, gcd ratio_n ratio_d = 1 ∧ ratio_n = 5 ∧ ratio_d = 8 ∧
  ratio_n / ratio_d = male / (female + male))
  ∧
  (∃ percentage : ℕ, percentage = 40 ∧ percentage = 100 * (male - female) / male) :=
by
  intros female male hf hm
  have h1 : female = 15 := hf
  have h2 : male = 25 := hm
  sorry

end NUMINAMATH_GPT_class_ratio_and_percentage_l1175_117524


namespace NUMINAMATH_GPT_lines_coplanar_parameter_l1175_117525

/-- 
  Two lines are given in parametric form: 
  L1: (2 + 2s, 4s, -3 + rs)
  L2: (-1 + 3t, 2t, 1 + 2t)
  Prove that if these lines are coplanar, then r = 4.
-/
theorem lines_coplanar_parameter (s t r : ℝ) :
  ∃ (k : ℝ), 
  (∀ s t, 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0
      ∧
      (2 + 2 * s, 4 * s, -3 + r * s) = (k * (-1 + 3 * t), k * 2 * t, k * (1 + 2 * t))
  ) → r = 4 := sorry

end NUMINAMATH_GPT_lines_coplanar_parameter_l1175_117525


namespace NUMINAMATH_GPT_length_of_shorter_train_l1175_117505

noncomputable def relativeSpeedInMS (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  (speed1_kmh + speed2_kmh) * (5 / 18)

noncomputable def totalDistanceCovered (relativeSpeed_ms time_s : ℝ) : ℝ :=
  relativeSpeed_ms * time_s

noncomputable def lengthOfShorterTrain (longerTrainLength_m time_s : ℝ) (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relativeSpeed_ms := relativeSpeedInMS speed1_kmh speed2_kmh
  let totalDistance := totalDistanceCovered relativeSpeed_ms time_s
  totalDistance - longerTrainLength_m

theorem length_of_shorter_train :
  lengthOfShorterTrain 160 10.07919366450684 60 40 = 117.8220467912412 := 
sorry

end NUMINAMATH_GPT_length_of_shorter_train_l1175_117505


namespace NUMINAMATH_GPT_secant_length_l1175_117502

theorem secant_length
  (A B C D E : ℝ)
  (AB : A - B = 7)
  (BC : B - C = 7)
  (AD : A - D = 10)
  (pos : A > E ∧ D > E):
  E - D = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_secant_length_l1175_117502


namespace NUMINAMATH_GPT_vote_difference_l1175_117506

-- Definitions of initial votes for and against the policy
def vote_initial_for (x y : ℕ) : Prop := x + y = 450
def initial_margin (x y m : ℕ) : Prop := y > x ∧ y - x = m

-- Definitions of votes for and against in the second vote
def vote_second_for (x' y' : ℕ) : Prop := x' + y' = 450
def second_margin (x' y' m : ℕ) : Prop := x' - y' = 3 * m
def second_vote_ratio (x' y : ℕ) : Prop := x' = 10 * y / 9

-- Theorem to prove the increase in votes
theorem vote_difference (x y x' y' m : ℕ)
  (hi : vote_initial_for x y)
  (hm : initial_margin x y m)
  (hs : vote_second_for x' y')
  (hsm : second_margin x' y' m)
  (hr : second_vote_ratio x' y) : 
  x' - x = 52 :=
sorry

end NUMINAMATH_GPT_vote_difference_l1175_117506


namespace NUMINAMATH_GPT_n_cubed_minus_n_plus_one_is_square_l1175_117556

theorem n_cubed_minus_n_plus_one_is_square (n : ℕ) (h : (n^5 + n^4 + 1).divisors.card = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
sorry

end NUMINAMATH_GPT_n_cubed_minus_n_plus_one_is_square_l1175_117556


namespace NUMINAMATH_GPT_sum_invested_eq_2000_l1175_117594

theorem sum_invested_eq_2000 (P : ℝ) (R1 R2 T : ℝ) (H1 : R1 = 18) (H2 : R2 = 12) 
  (H3 : T = 2) (H4 : (P * R1 * T / 100) - (P * R2 * T / 100) = 240): 
  P = 2000 :=
by 
  sorry

end NUMINAMATH_GPT_sum_invested_eq_2000_l1175_117594


namespace NUMINAMATH_GPT_cost_per_person_l1175_117586

theorem cost_per_person 
  (total_cost : ℕ) 
  (total_people : ℕ) 
  (total_cost_in_billion : total_cost = 40000000000) 
  (total_people_in_million : total_people = 200000000) :
  total_cost / total_people = 200 := 
sorry

end NUMINAMATH_GPT_cost_per_person_l1175_117586


namespace NUMINAMATH_GPT_ratio_of_nuts_to_raisins_l1175_117527

theorem ratio_of_nuts_to_raisins 
  (R N : ℝ) 
  (h_ratio : 3 * R = 0.2727272727272727 * (3 * R + 4 * N)) : 
  N = 2 * R := 
sorry

end NUMINAMATH_GPT_ratio_of_nuts_to_raisins_l1175_117527


namespace NUMINAMATH_GPT_relationship_between_y_values_l1175_117528

theorem relationship_between_y_values 
  (m : ℝ) 
  (y1 y2 y3 : ℝ)
  (h1 : y1 = (-1 : ℝ) ^ 2 + 2 * (-1 : ℝ) + m) 
  (h2 : y2 = (3 : ℝ) ^ 2 + 2 * (3 : ℝ) + m) 
  (h3 : y3 = ((1 / 2) : ℝ) ^ 2 + 2 * ((1 / 2) : ℝ) + m) : 
  y2 > y3 ∧ y3 > y1 := 
by 
  sorry

end NUMINAMATH_GPT_relationship_between_y_values_l1175_117528
