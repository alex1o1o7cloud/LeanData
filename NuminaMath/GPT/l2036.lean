import Mathlib

namespace NUMINAMATH_GPT_exists_triang_and_square_le_50_l2036_203623

def is_triang_num (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem exists_triang_and_square_le_50 : ∃ n : ℕ, n ≤ 50 ∧ is_triang_num n ∧ is_perfect_square n :=
by
  sorry

end NUMINAMATH_GPT_exists_triang_and_square_le_50_l2036_203623


namespace NUMINAMATH_GPT_shapeB_is_symmetric_to_original_l2036_203614

-- Assume a simple type to represent our shapes
inductive Shape
| shapeA
| shapeB
| shapeC
| shapeD
| shapeE
| originalShape

-- Define the symmetry condition
def is_symmetric (s1 s2 : Shape) : Prop := sorry  -- this would be the condition to check symmetry

-- The theorem to prove that shapeB is symmetric to the original shape
theorem shapeB_is_symmetric_to_original :
  is_symmetric Shape.shapeB Shape.originalShape :=
sorry

end NUMINAMATH_GPT_shapeB_is_symmetric_to_original_l2036_203614


namespace NUMINAMATH_GPT_each_person_received_5_l2036_203693

theorem each_person_received_5 (S n : ℕ) (hn₁ : n > 5) (hn₂ : 5 * S = 2 * n * (n - 5)) (hn₃ : 4 * S = n * (n + 4)) :
  S / (n + 4) = 5 :=
by
  sorry

end NUMINAMATH_GPT_each_person_received_5_l2036_203693


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2036_203637

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2036_203637


namespace NUMINAMATH_GPT_least_possible_value_m_n_l2036_203652

theorem least_possible_value_m_n :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd (m + n) 330 = 1 ∧ n ∣ m^m ∧ ¬(m % n = 0) ∧ (m + n = 377) :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_m_n_l2036_203652


namespace NUMINAMATH_GPT_find_x_l2036_203650

def perpendicular_vectors_solution (x : ℝ) : Prop :=
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (3, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 / 3

theorem find_x (x : ℝ) : perpendicular_vectors_solution x := sorry

end NUMINAMATH_GPT_find_x_l2036_203650


namespace NUMINAMATH_GPT_nobody_but_angela_finished_9_problems_l2036_203618

theorem nobody_but_angela_finished_9_problems :
  ∀ (total_problems martha_problems : ℕ)
    (jenna_problems : ℕ → ℕ)
    (mark_problems : ℕ → ℕ),
    total_problems = 20 →
    martha_problems = 2 →
    jenna_problems martha_problems = 4 * martha_problems - 2 →
    mark_problems (jenna_problems martha_problems) = (jenna_problems martha_problems) / 2 →
    total_problems - (martha_problems + jenna_problems martha_problems + mark_problems (jenna_problems martha_problems)) = 9 :=
by
  intros total_problems martha_problems jenna_problems mark_problems h_total h_martha h_jenna h_mark
  sorry

end NUMINAMATH_GPT_nobody_but_angela_finished_9_problems_l2036_203618


namespace NUMINAMATH_GPT_distance_after_rest_l2036_203620

-- Define the conditions
def distance_before_rest := 0.75
def total_distance := 1.0

-- State the theorem
theorem distance_after_rest :
  total_distance - distance_before_rest = 0.25 :=
by sorry

end NUMINAMATH_GPT_distance_after_rest_l2036_203620


namespace NUMINAMATH_GPT_find_number_l2036_203667

theorem find_number (X : ℝ) (h : 50 = 0.20 * X + 47) : X = 15 :=
sorry

end NUMINAMATH_GPT_find_number_l2036_203667


namespace NUMINAMATH_GPT_find_x_l2036_203674

-- Define the condition variables
variables (y z x : ℝ) (Y Z X : ℝ)
-- Primary conditions given in the problem
variable (h_y : y = 7)
variable (h_z : z = 6)
variable (h_cosYZ : Real.cos (Y - Z) = 15 / 16)

-- The main theorem to prove
theorem find_x (h_y : y = 7) (h_z : z = 6) (h_cosYZ : Real.cos (Y - Z) = 15 / 16) :
  x = Real.sqrt 22 :=
sorry

end NUMINAMATH_GPT_find_x_l2036_203674


namespace NUMINAMATH_GPT_joe_initial_paint_amount_l2036_203610

theorem joe_initial_paint_amount (P : ℝ) 
  (h1 : (2/3) * P + (1/15) * P = 264) : P = 360 :=
sorry

end NUMINAMATH_GPT_joe_initial_paint_amount_l2036_203610


namespace NUMINAMATH_GPT_fraction_is_half_l2036_203678

variable (N : ℕ) (F : ℚ)

theorem fraction_is_half (h1 : N = 90) (h2 : 3 + F * (1/3) * (1/5) * N = (1/15) * N) : F = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_half_l2036_203678


namespace NUMINAMATH_GPT_cubic_intersection_unique_point_l2036_203690

-- Define the cubic functions f and g
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def g (a b c d x : ℝ) : ℝ := -a * x^3 + b * x^2 - c * x + d

-- Translate conditions into Lean conditions
variables (a b c d : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Lean statement to prove the intersection point
theorem cubic_intersection_unique_point :
  ∀ x y : ℝ, (f a b c d x = y) ↔ (g a b c d x = y) → (x = 0 ∧ y = d) :=
by
  -- Mathematical steps would go here (omitted with sorry)
  sorry

end NUMINAMATH_GPT_cubic_intersection_unique_point_l2036_203690


namespace NUMINAMATH_GPT_average_speed_l2036_203698

-- Define the given conditions as Lean variables and constants
variables (v : ℕ)

-- The average speed problem in Lean
theorem average_speed (h : 8 * v = 528) : v = 66 :=
sorry

end NUMINAMATH_GPT_average_speed_l2036_203698


namespace NUMINAMATH_GPT_number_of_propositions_is_4_l2036_203636

def is_proposition (s : String) : Prop :=
  s = "The Earth is a planet in the solar system" ∨ 
  s = "{0} ∈ ℕ" ∨ 
  s = "1+1 > 2" ∨ 
  s = "Elderly people form a set"

theorem number_of_propositions_is_4 : 
  (is_proposition "The Earth is a planet in the solar system" ∨ 
   is_proposition "{0} ∈ ℕ" ∨ 
   is_proposition "1+1 > 2" ∨ 
   is_proposition "Elderly people form a set") → 
  4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_propositions_is_4_l2036_203636


namespace NUMINAMATH_GPT_num_ordered_pairs_eq_1728_l2036_203669

theorem num_ordered_pairs_eq_1728 (x y : ℕ) (h1 : 1728 = 2^6 * 3^3) (h2 : x * y = 1728) : 
  ∃ (n : ℕ), n = 28 := 
sorry

end NUMINAMATH_GPT_num_ordered_pairs_eq_1728_l2036_203669


namespace NUMINAMATH_GPT_function_properties_l2036_203644

variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, 2 * f x * f y = f (x + y) + f (x - y))
variable (h2 : f 1 = -1)

theorem function_properties :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f x + f (1 - x) = 0) :=
sorry

end NUMINAMATH_GPT_function_properties_l2036_203644


namespace NUMINAMATH_GPT_find_a1_l2036_203670

variable {q a1 a2 a3 a4 : ℝ}
variable (S : ℕ → ℝ)

axiom common_ratio_pos : q > 0
axiom S2_eq : S 2 = 3 * a2 + 2
axiom S4_eq : S 4 = 3 * a4 + 2

theorem find_a1 (h1 : S 2 = 3 * a2 + 2) (h2 : S 4 = 3 * a4 + 2) (common_ratio_pos : q > 0) : a1 = -1 :=
sorry

end NUMINAMATH_GPT_find_a1_l2036_203670


namespace NUMINAMATH_GPT_cuboid_height_l2036_203686

-- Definition of variables
def length := 4  -- in cm
def breadth := 6  -- in cm
def surface_area := 120  -- in cm²

-- The formula for the surface area of a cuboid: S = 2(lb + lh + bh)
def surface_area_formula (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

-- Given these values, we need to prove that the height h is 3.6 cm
theorem cuboid_height : 
  ∃ h : ℝ, surface_area = surface_area_formula length breadth h ∧ h = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_height_l2036_203686


namespace NUMINAMATH_GPT_largest_part_of_proportional_division_l2036_203608

theorem largest_part_of_proportional_division :
  ∀ (x y z : ℝ),
    x + y + z = 120 ∧
    x / (1 / 2) = y / (1 / 4) ∧
    x / (1 / 2) = z / (1 / 6) →
    max x (max y z) = 60 :=
by sorry

end NUMINAMATH_GPT_largest_part_of_proportional_division_l2036_203608


namespace NUMINAMATH_GPT_independent_sum_of_projections_l2036_203602

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem independent_sum_of_projections (A1 A2 A3 P P1 P2 P3 : ℝ × ℝ) 
  (h_eq_triangle : distance A1 A2 = distance A2 A3 ∧ distance A2 A3 = distance A3 A1)
  (h_proj_P1 : P1 = (P.1, A2.2))
  (h_proj_P2 : P2 = (P.1, A3.2))
  (h_proj_P3 : P3 = (P.1, A1.2)) :
  distance A1 P2 + distance A2 P3 + distance A3 P1 = (3 / 2) * distance A1 A2 := 
sorry

end NUMINAMATH_GPT_independent_sum_of_projections_l2036_203602


namespace NUMINAMATH_GPT_defect_rate_product_l2036_203600

theorem defect_rate_product (P1_defect P2_defect : ℝ) (h1 : P1_defect = 0.10) (h2 : P2_defect = 0.03) : 
  ((1 - P1_defect) * (1 - P2_defect)) = 0.873 → (1 - ((1 - P1_defect) * (1 - P2_defect)) = 0.127) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_defect_rate_product_l2036_203600


namespace NUMINAMATH_GPT_a_range_l2036_203601

open Set

variable (A B : Set Real) (a : Real)

def A_def : Set Real := {x | 3 * x + 1 < 4}
def B_def : Set Real := {x | x - a < 0}
def intersection_eq : A ∩ B = A := sorry

theorem a_range : a ≥ 1 :=
  by
  have hA : A = {x | x < 1} := sorry
  have hB : B = {x | x < a} := sorry
  have h_intersection : (A ∩ B) = A := sorry
  sorry

end NUMINAMATH_GPT_a_range_l2036_203601


namespace NUMINAMATH_GPT_intersection_eq_l2036_203691

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x < 2}

theorem intersection_eq : M ∩ N = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l2036_203691


namespace NUMINAMATH_GPT_max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l2036_203632

-- Define the traffic flow function
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

-- Condition: v > 0
axiom v_pos (v : ℝ) : v > 0 → traffic_flow v ≥ 0

-- Prove that the average speed v = 40 results in the maximum traffic flow y = 920/83 ≈ 11.08
theorem max_traffic_flow_at_v_40 : traffic_flow 40 = 920 / 83 :=
sorry

-- Prove that to ensure the traffic flow is at least 10 thousand vehicles per hour,
-- the average speed v should be in the range [25, 64]
theorem traffic_flow_at_least_10_thousand (v : ℝ) (h : traffic_flow v ≥ 10) : 25 ≤ v ∧ v ≤ 64 :=
sorry

end NUMINAMATH_GPT_max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l2036_203632


namespace NUMINAMATH_GPT_evaluate_expression_l2036_203616

variable (a b c : ℝ)

theorem evaluate_expression 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l2036_203616


namespace NUMINAMATH_GPT_find_f_l2036_203617

def f : ℝ → ℝ := sorry

theorem find_f (x : ℝ) : f (x + 2) = 2 * x + 3 → f x = 2 * x - 1 :=
by
  intro h
  -- Proof goes here 
  sorry

end NUMINAMATH_GPT_find_f_l2036_203617


namespace NUMINAMATH_GPT_algebraic_fraction_l2036_203643

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end NUMINAMATH_GPT_algebraic_fraction_l2036_203643


namespace NUMINAMATH_GPT_building_height_l2036_203635

noncomputable def height_of_building (flagpole_height shadow_of_flagpole shadow_of_building : ℝ) : ℝ :=
  (flagpole_height / shadow_of_flagpole) * shadow_of_building

theorem building_height : height_of_building 18 45 60 = 24 := by {
  sorry
}

end NUMINAMATH_GPT_building_height_l2036_203635


namespace NUMINAMATH_GPT_simplify_expression_l2036_203625

theorem simplify_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2036_203625


namespace NUMINAMATH_GPT_math_books_count_l2036_203642

theorem math_books_count (M H : ℕ) (h1 : M + H = 80) (h2 : 4 * M + 5 * H = 373) : M = 27 :=
by
  sorry

end NUMINAMATH_GPT_math_books_count_l2036_203642


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l2036_203606

theorem equilateral_triangle_perimeter (x : ℕ) (h : 2 * x = x + 15) : 
  3 * (2 * x) = 90 :=
by
  -- Definitions & hypothesis
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l2036_203606


namespace NUMINAMATH_GPT_total_exercise_hours_l2036_203627

theorem total_exercise_hours (natasha_minutes_per_day : ℕ) (natasha_days : ℕ)
  (esteban_minutes_per_day : ℕ) (esteban_days : ℕ)
  (h_n : natasha_minutes_per_day = 30) (h_nd : natasha_days = 7)
  (h_e : esteban_minutes_per_day = 10) (h_ed : esteban_days = 9) :
  (natasha_minutes_per_day * natasha_days + esteban_minutes_per_day * esteban_days) / 60 = 5 :=
by
  sorry

end NUMINAMATH_GPT_total_exercise_hours_l2036_203627


namespace NUMINAMATH_GPT_total_drivers_l2036_203640

theorem total_drivers (N : ℕ) (A : ℕ) (sA sB sC sD : ℕ) (total_sampled : ℕ)
  (hA : A = 96) (hsA : sA = 12) (hsB : sB = 21) (hsC : sC = 25) (hsD : sD = 43) (htotal : total_sampled = sA + sB + sC + sD)
  (hsA_proportion : (sA : ℚ) / A = (total_sampled : ℚ) / N) : N = 808 := by
  sorry

end NUMINAMATH_GPT_total_drivers_l2036_203640


namespace NUMINAMATH_GPT_tires_in_parking_lot_l2036_203687

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end NUMINAMATH_GPT_tires_in_parking_lot_l2036_203687


namespace NUMINAMATH_GPT_product_and_divisibility_l2036_203659

theorem product_and_divisibility (n : ℕ) (h : n = 3) :
  (n-1) * n * (n+1) * (n+2) * (n+3) = 720 ∧ ¬ (720 % 11 = 0) :=
by
  sorry

end NUMINAMATH_GPT_product_and_divisibility_l2036_203659


namespace NUMINAMATH_GPT_angle_bc_l2036_203612

variables (a b c : ℝ → ℝ → Prop) (theta : ℝ)

-- Definitions of parallelism and angle conditions
def parallel (x y : ℝ → ℝ → Prop) : Prop := ∀ p q r s : ℝ, x p q → y r s → p - q = r - s

def angle_between (x y : ℝ → ℝ → Prop) (θ : ℝ) : Prop := sorry  -- Assume we have a definition for angle between lines

-- Given conditions
axiom parallel_ab : parallel a b
axiom angle_ac : angle_between a c theta

-- Theorem statement
theorem angle_bc : angle_between b c theta :=
sorry

end NUMINAMATH_GPT_angle_bc_l2036_203612


namespace NUMINAMATH_GPT_cubed_multiplication_identity_l2036_203621

theorem cubed_multiplication_identity : 3^3 * 6^3 = 5832 := by
  sorry

end NUMINAMATH_GPT_cubed_multiplication_identity_l2036_203621


namespace NUMINAMATH_GPT_present_age_of_father_l2036_203681

/-- The present age of the father is 3 years more than 3 times the age of his son, 
    and 3 years hence, the father's age will be 8 years more than twice the age of the son. 
    Prove that the present age of the father is 27 years. -/
theorem present_age_of_father (F S : ℕ) (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 8) : F = 27 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_father_l2036_203681


namespace NUMINAMATH_GPT_rationalize_fraction_l2036_203633

theorem rationalize_fraction :
  (5 : ℚ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = 
  (5 * Real.sqrt 2) / 36 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_fraction_l2036_203633


namespace NUMINAMATH_GPT_product_of_roots_abs_eq_l2036_203685

theorem product_of_roots_abs_eq (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  x = 5 ∨ x = -5 ∧ ((5 : ℝ) * (-5 : ℝ) = -25) := 
sorry

end NUMINAMATH_GPT_product_of_roots_abs_eq_l2036_203685


namespace NUMINAMATH_GPT_more_perfect_squares_with_7_digit_17th_l2036_203611

noncomputable def seventeenth_digit (n : ℕ) : ℕ :=
  (n / 10^16) % 10

theorem more_perfect_squares_with_7_digit_17th
  (h_bound : ∀ n, n < 10^10 → (n * n) < 10^20)
  (h_representation : ∀ m, m < 10^20 → ∃ n, n < 10^10 ∧ m = n * n) :
  (∃ majority_digit_7 : ℕ,
    (∃ majority_digit_8 : ℕ,
      ∀ n, seventeenth_digit (n * n) = 7 → majority_digit_7 > majority_digit_8)
  ) :=
sorry

end NUMINAMATH_GPT_more_perfect_squares_with_7_digit_17th_l2036_203611


namespace NUMINAMATH_GPT_train_return_time_l2036_203634

open Real

theorem train_return_time
  (C_small : Real := 1.5)
  (C_large : Real := 3)
  (speed : Real := 10)
  (initial_connection : String := "A to C")
  (switch_interval : Real := 1) :
  (126 = 2.1 * 60) :=
sorry

end NUMINAMATH_GPT_train_return_time_l2036_203634


namespace NUMINAMATH_GPT_train_ride_length_l2036_203654

noncomputable def totalMinutesUntil0900 (leaveTime : Nat) (arrivalTime : Nat) : Nat :=
  arrivalTime - leaveTime

noncomputable def walkTime : Nat := 10

noncomputable def rideTime (totalTime : Nat) (walkTime : Nat) : Nat :=
  totalTime - walkTime

theorem train_ride_length (leaveTime : Nat) (arrivalTime : Nat) :
  leaveTime = 450 → arrivalTime = 540 → rideTime (totalMinutesUntil0900 leaveTime arrivalTime) walkTime = 80 :=
by
  intros h_leaveTime h_arrivalTime
  rw [h_leaveTime, h_arrivalTime]
  unfold totalMinutesUntil0900
  unfold rideTime
  unfold walkTime
  sorry

end NUMINAMATH_GPT_train_ride_length_l2036_203654


namespace NUMINAMATH_GPT_add_hex_numbers_l2036_203628

theorem add_hex_numbers : (7 * 16^2 + 10 * 16^1 + 3) + (1 * 16^2 + 15 * 16^1 + 4) = 9 * 16^2 + 9 * 16^1 + 7 := by sorry

end NUMINAMATH_GPT_add_hex_numbers_l2036_203628


namespace NUMINAMATH_GPT_length_of_each_piece_l2036_203653

theorem length_of_each_piece :
  ∀ (ribbon_length remaining_length pieces : ℕ),
  ribbon_length = 51 →
  remaining_length = 36 →
  pieces = 100 →
  (ribbon_length - remaining_length) / pieces * 100 = 15 :=
by
  intros ribbon_length remaining_length pieces h1 h2 h3
  sorry

end NUMINAMATH_GPT_length_of_each_piece_l2036_203653


namespace NUMINAMATH_GPT_joan_balloons_l2036_203619

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_joan_balloons_l2036_203619


namespace NUMINAMATH_GPT_total_players_l2036_203694

theorem total_players (K Kho_only Both : Nat) (hK : K = 10) (hKho_only : Kho_only = 30) (hBoth : Both = 5) : 
  (K - Both) + Kho_only + Both = 40 := by
  sorry

end NUMINAMATH_GPT_total_players_l2036_203694


namespace NUMINAMATH_GPT_abs_ineq_range_l2036_203688

theorem abs_ineq_range (x : ℝ) : |x - 3| + |x + 1| ≥ 4 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_GPT_abs_ineq_range_l2036_203688


namespace NUMINAMATH_GPT_ratio_x_y_l2036_203651

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end NUMINAMATH_GPT_ratio_x_y_l2036_203651


namespace NUMINAMATH_GPT_curve_is_line_l2036_203648

-- Define the polar equation as a condition
def polar_eq (r θ : ℝ) : Prop := r = 2 / (2 * Real.sin θ - Real.cos θ)

-- Define what it means for a curve to be a line
def is_line (x y : ℝ) : Prop := x + 2 * y = 2

-- The main statement to prove
theorem curve_is_line (r θ : ℝ) (x y : ℝ) (hr : polar_eq r θ) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  is_line x y :=
sorry

end NUMINAMATH_GPT_curve_is_line_l2036_203648


namespace NUMINAMATH_GPT_John_needs_more_days_l2036_203638

theorem John_needs_more_days (days_worked : ℕ) (amount_earned : ℕ) :
  days_worked = 10 ∧ amount_earned = 250 ∧ 
  (∀ d : ℕ, d < days_worked → amount_earned / days_worked = amount_earned / 10) →
  ∃ more_days : ℕ, more_days = 10 ∧ amount_earned * 2 = (days_worked + more_days) * (amount_earned / days_worked) :=
sorry

end NUMINAMATH_GPT_John_needs_more_days_l2036_203638


namespace NUMINAMATH_GPT_largest_fraction_l2036_203641

noncomputable def compare_fractions : List ℚ :=
  [5 / 11, 7 / 16, 9 / 20, 11 / 23, 111 / 245, 145 / 320, 185 / 409, 211 / 465, 233 / 514]

theorem largest_fraction :
  max (5 / 11) (max (7 / 16) (max (9 / 20) (max (11 / 23) (max (111 / 245) (max (145 / 320) (max (185 / 409) (max (211 / 465) (233 / 514)))))))) = 11 / 23 := 
  sorry

end NUMINAMATH_GPT_largest_fraction_l2036_203641


namespace NUMINAMATH_GPT_problem1_problem2_l2036_203677

-- Definitions used directly from conditions
def inequality (m x : ℝ) : Prop := m * x ^ 2 - 2 * m * x - 1 < 0

-- Proof problem (1)
theorem problem1 (m : ℝ) (h : ∀ x : ℝ, inequality m x) : -1 < m ∧ m ≤ 0 :=
sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) (h : ∀ m : ℝ, |m| ≤ 1 → inequality m x) :
  (1 - Real.sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2036_203677


namespace NUMINAMATH_GPT_students_algebra_or_drafting_not_both_not_geography_l2036_203671

variables (A D G : Finset ℕ)
-- Condition 1: Fifteen students are taking both algebra and drafting
variable (h1 : (A ∩ D).card = 15)
-- Condition 2: There are 30 students taking algebra
variable (h2 : A.card = 30)
-- Condition 3: There are 12 students taking drafting only
variable (h3 : (D \ A).card = 12)
-- Condition 4: There are eight students taking a geography class
variable (h4 : G.card = 8)
-- Condition 5: Two students are also taking both algebra and drafting and geography
variable (h5 : ((A ∩ D) ∩ G).card = 2)

-- Question: Prove the final count of students taking algebra or drafting but not both, and not taking geography is 25
theorem students_algebra_or_drafting_not_both_not_geography :
  ((A \ D) ∪ (D \ A)).card - ((A ∩ D) ∩ G).card = 25 :=
by
  sorry

end NUMINAMATH_GPT_students_algebra_or_drafting_not_both_not_geography_l2036_203671


namespace NUMINAMATH_GPT_Barbara_Mike_ratio_is_one_half_l2036_203607

-- Define the conditions
def Mike_age_current : ℕ := 16
def Mike_age_future : ℕ := 24
def Barbara_age_future : ℕ := 16

-- Define Barbara's current age based on the conditions
def Barbara_age_current : ℕ := Mike_age_current - (Mike_age_future - Barbara_age_future)

-- Define the ratio of Barbara's age to Mike's age
def ratio_Barbara_Mike : ℚ := Barbara_age_current / Mike_age_current

-- Prove that the ratio is 1:2
theorem Barbara_Mike_ratio_is_one_half : ratio_Barbara_Mike = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_Barbara_Mike_ratio_is_one_half_l2036_203607


namespace NUMINAMATH_GPT_cooler_capacity_l2036_203695

theorem cooler_capacity (C : ℝ) (h1 : 3.25 * C = 325) : C = 100 :=
sorry

end NUMINAMATH_GPT_cooler_capacity_l2036_203695


namespace NUMINAMATH_GPT_expand_expression_l2036_203697

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_expand_expression_l2036_203697


namespace NUMINAMATH_GPT_average_temperature_l2036_203663

def temperatures : List ℝ := [-36, 13, -15, -10]

theorem average_temperature : (List.sum temperatures) / (temperatures.length) = -12 := by
  sorry

end NUMINAMATH_GPT_average_temperature_l2036_203663


namespace NUMINAMATH_GPT_original_side_length_l2036_203679

theorem original_side_length (x : ℝ) 
  (h1 : (x - 4) * (x - 3) = 120) : x = 12 :=
sorry

end NUMINAMATH_GPT_original_side_length_l2036_203679


namespace NUMINAMATH_GPT_jake_present_weight_l2036_203631

theorem jake_present_weight :
  ∃ (J K L : ℕ), J = 194 ∧ J + K = 287 ∧ J - L = 2 * K ∧ J = 194 := by
  sorry

end NUMINAMATH_GPT_jake_present_weight_l2036_203631


namespace NUMINAMATH_GPT_no_common_real_root_l2036_203630

theorem no_common_real_root (a b : ℚ) : 
  ¬ ∃ (r : ℝ), (r^5 - r - 1 = 0) ∧ (r^2 + a * r + b = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_common_real_root_l2036_203630


namespace NUMINAMATH_GPT_hyperbola_focal_length_range_l2036_203682

theorem hyperbola_focal_length_range (m : ℝ) (h1 : m > 0)
    (h2 : ∀ x y, x^2 - y^2 / m^2 ≠ 1 → y ≠ m * x ∧ y ≠ -m * x)
    (h3 : ∀ x y, x^2 + (y + 2)^2 = 1 → x^2 + y^2 / m^2 ≠ 1) :
    ∃ c : ℝ, 2 < 2 * Real.sqrt (1 + m^2) ∧ 2 * Real.sqrt (1 + m^2) < 4 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focal_length_range_l2036_203682


namespace NUMINAMATH_GPT_pythagorean_theorem_special_cases_l2036_203692

open Nat

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem pythagorean_theorem_special_cases (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (is_even a ∨ is_even b) ∧ 
  (is_multiple_of_3 a ∨ is_multiple_of_3 b) ∧ 
  (is_multiple_of_5 a ∨ is_multiple_of_5 b ∨ is_multiple_of_5 c) :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_theorem_special_cases_l2036_203692


namespace NUMINAMATH_GPT_cube_root_1728_simplified_l2036_203699

theorem cube_root_1728_simplified :
  let a := 12
  let b := 1
  a + b = 13 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_1728_simplified_l2036_203699


namespace NUMINAMATH_GPT_crackers_given_to_friends_l2036_203604

theorem crackers_given_to_friends (crackers_per_friend : ℕ) (number_of_friends : ℕ) (h1 : crackers_per_friend = 6) (h2 : number_of_friends = 6) : (crackers_per_friend * number_of_friends) = 36 :=
by
  sorry

end NUMINAMATH_GPT_crackers_given_to_friends_l2036_203604


namespace NUMINAMATH_GPT_ratio_of_tetrahedron_to_cube_volume_l2036_203624

theorem ratio_of_tetrahedron_to_cube_volume (x : ℝ) (hx : 0 < x) :
  let V_cube := x^3
  let a_tetrahedron := (x * Real.sqrt 3) / 2
  let V_tetrahedron := (a_tetrahedron^3 * Real.sqrt 2) / 12
  (V_tetrahedron / V_cube) = (Real.sqrt 6 / 32) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_tetrahedron_to_cube_volume_l2036_203624


namespace NUMINAMATH_GPT_range_of_a_l2036_203626

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x = 1 → x > a) : a < 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2036_203626


namespace NUMINAMATH_GPT_maximum_profit_l2036_203649

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end NUMINAMATH_GPT_maximum_profit_l2036_203649


namespace NUMINAMATH_GPT_seats_needed_on_bus_l2036_203665

variable (f t tr dr c h : ℕ)

def flute_players := 5
def trumpet_players := 3 * flute_players
def trombone_players := trumpet_players - 8
def drummers := trombone_players + 11
def clarinet_players := 2 * flute_players
def french_horn_players := trombone_players + 3

theorem seats_needed_on_bus :
  f = 5 →
  t = 3 * f →
  tr = t - 8 →
  dr = tr + 11 →
  c = 2 * f →
  h = tr + 3 →
  f + t + tr + dr + c + h = 65 :=
by
  sorry

end NUMINAMATH_GPT_seats_needed_on_bus_l2036_203665


namespace NUMINAMATH_GPT_count_positive_integers_satisfy_l2036_203668

theorem count_positive_integers_satisfy :
  ∃ (S : Finset ℕ), (∀ n ∈ S, (n + 5) * (n - 3) * (n - 12) * (n - 17) < 0) ∧ S.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_satisfy_l2036_203668


namespace NUMINAMATH_GPT_inequality_proof_l2036_203666

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = a * b) : 
  (a / (b^2 + 4) + b / (a^2 + 4) >= 1 / 2) := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2036_203666


namespace NUMINAMATH_GPT_joe_travel_time_l2036_203629

theorem joe_travel_time
  (d : ℝ) -- Total distance
  (rw : ℝ) (rr : ℝ) -- Walking and running rates
  (tw : ℝ) -- Walking time
  (tr : ℝ) -- Running time
  (h1 : tw = 9)
  (h2 : rr = 4 * rw)
  (h3 : rw * tw = d / 3)
  (h4 : rr * tr = 2 * d / 3) :
  tw + tr = 13.5 :=
by 
  sorry

end NUMINAMATH_GPT_joe_travel_time_l2036_203629


namespace NUMINAMATH_GPT_quadrilateral_circumscribed_l2036_203639

structure ConvexQuad (A B C D : Type) := 
  (is_convex : True)
  (P : Type)
  (interior : True)
  (angle_APB_angle_CPD_eq_angle_BPC_angle_DPA : True)
  (angle_PAD_angle_PCD_eq_angle_PAB_angle_PCB : True)
  (angle_PDC_angle_PBC_eq_angle_PDA_angle_PBA : True)

theorem quadrilateral_circumscribed (A B C D : Type) (quad : ConvexQuad A B C D) : True := 
sorry

end NUMINAMATH_GPT_quadrilateral_circumscribed_l2036_203639


namespace NUMINAMATH_GPT_maurice_rides_l2036_203605

theorem maurice_rides (M : ℕ) 
    (h1 : ∀ m_attended : ℕ, m_attended = 8)
    (h2 : ∀ matt_other : ℕ, matt_other = 16)
    (h3 : ∀ total_matt : ℕ, total_matt = matt_other + m_attended)
    (h4 : total_matt = 3 * M) : M = 8 :=
by 
  sorry

end NUMINAMATH_GPT_maurice_rides_l2036_203605


namespace NUMINAMATH_GPT_remainder_polynomial_2047_l2036_203675

def f (r : ℤ) : ℤ := r ^ 11 - 1

theorem remainder_polynomial_2047 : f 2 = 2047 :=
by
  sorry

end NUMINAMATH_GPT_remainder_polynomial_2047_l2036_203675


namespace NUMINAMATH_GPT_three_sum_eq_nine_seven_five_l2036_203613

theorem three_sum_eq_nine_seven_five {a b c : ℝ} 
    (h1 : b + c = 15 - 2 * a)
    (h2 : a + c = -10 - 4 * b)
    (h3 : a + b = 8 - 2 * c) : 
    3 * a + 3 * b + 3 * c = 9.75 := 
by
    sorry

end NUMINAMATH_GPT_three_sum_eq_nine_seven_five_l2036_203613


namespace NUMINAMATH_GPT_trigonometric_identity_l2036_203683

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_trigonometric_identity_l2036_203683


namespace NUMINAMATH_GPT_triangle_inequality_l2036_203622

variable (a b c : ℝ) -- sides of the triangle
variable (h_a h_b h_c S r R : ℝ) -- heights, area of the triangle, inradius, circumradius

-- Definitions of conditions
axiom h_def : h_a + h_b + h_c = (a + b + c) -- express heights sum in terms of sides sum (for illustrative purposes)
axiom S_def : S = 0.5 * a * h_a  -- area definition (adjust as needed)
axiom r_def : 9 * r ≤ h_a + h_b + h_c -- given in solution
axiom R_def : h_a + h_b + h_c ≤ 9 * R / 2 -- given in solution

theorem triangle_inequality :
  9 * r / (2 * S) ≤ (1 / a) + (1 / b) + (1 / c) ∧ (1 / a) + (1 / b) + (1 / c) ≤ 9 * R / (4 * S) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2036_203622


namespace NUMINAMATH_GPT_milton_books_l2036_203615

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end NUMINAMATH_GPT_milton_books_l2036_203615


namespace NUMINAMATH_GPT_jessica_mother_age_l2036_203696

theorem jessica_mother_age
  (mother_age_when_died : ℕ)
  (jessica_age_when_died : ℕ)
  (jessica_current_age : ℕ)
  (years_since_mother_died : ℕ)
  (half_age_condition : jessica_age_when_died = mother_age_when_died / 2)
  (current_age_condition : jessica_current_age = 40)
  (years_since_death_condition : years_since_mother_died = 10)
  (age_at_death_condition : jessica_age_when_died = jessica_current_age - years_since_mother_died) :
  mother_age_when_died + years_since_mother_died = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_jessica_mother_age_l2036_203696


namespace NUMINAMATH_GPT_probability_exactly_one_correct_l2036_203689

def P_A := 0.7
def P_B := 0.8

def P_A_correct_B_incorrect := P_A * (1 - P_B)
def P_A_incorrect_B_correct := (1 - P_A) * P_B

theorem probability_exactly_one_correct :
  P_A_correct_B_incorrect + P_A_incorrect_B_correct = 0.38 :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_one_correct_l2036_203689


namespace NUMINAMATH_GPT_total_fish_l2036_203609

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end NUMINAMATH_GPT_total_fish_l2036_203609


namespace NUMINAMATH_GPT_students_without_A_l2036_203645

theorem students_without_A 
  (total_students : ℕ) 
  (A_in_literature : ℕ) 
  (A_in_science : ℕ) 
  (A_in_both : ℕ) 
  (h_total_students : total_students = 35)
  (h_A_in_literature : A_in_literature = 10)
  (h_A_in_science : A_in_science = 15)
  (h_A_in_both : A_in_both = 5) :
  total_students - (A_in_literature + A_in_science - A_in_both) = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_without_A_l2036_203645


namespace NUMINAMATH_GPT_simplify_expression_l2036_203661

theorem simplify_expression :
  (210 / 18) * (6 / 150) * (9 / 4) = 21 / 20 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2036_203661


namespace NUMINAMATH_GPT_monotonic_intervals_and_non_negative_f_l2036_203673

noncomputable def f (m x : ℝ) : ℝ := m / x - m + Real.log x

theorem monotonic_intervals_and_non_negative_f (m : ℝ) : 
  (∀ x > 0, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_and_non_negative_f_l2036_203673


namespace NUMINAMATH_GPT_parallel_lines_slope_m_l2036_203676

theorem parallel_lines_slope_m (m : ℝ) : (∀ (x y : ℝ), (x - 2 * y + 5 = 0) ↔ (2 * x + m * y - 5 = 0)) → m = -4 :=
by
  intros h
  -- Add the necessary calculative steps here
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_m_l2036_203676


namespace NUMINAMATH_GPT_max_sector_area_l2036_203603

theorem max_sector_area (r θ : ℝ) (S : ℝ) (h_perimeter : 2 * r + θ * r = 16)
  (h_max_area : S = 1 / 2 * θ * r^2) :
  r = 4 ∧ θ = 2 ∧ S = 16 := by
  -- sorry, the proof is expected to go here
  sorry

end NUMINAMATH_GPT_max_sector_area_l2036_203603


namespace NUMINAMATH_GPT_max_value_of_f_value_of_f_given_tan_half_alpha_l2036_203672

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * (Real.sin x)

theorem max_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ 3) ∧ (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 ∧ f x = 3) :=
sorry

theorem value_of_f_given_tan_half_alpha (α : ℝ) (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_value_of_f_given_tan_half_alpha_l2036_203672


namespace NUMINAMATH_GPT_negation_of_p_l2036_203647

theorem negation_of_p : (¬ ∃ x : ℕ, x^2 > 4^x) ↔ (∀ x : ℕ, x^2 ≤ 4^x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l2036_203647


namespace NUMINAMATH_GPT_Jurassic_Zoo_Total_l2036_203684

theorem Jurassic_Zoo_Total
  (C : ℕ) (A : ℕ)
  (h1 : C = 161)
  (h2 : 8 * A + 4 * C = 964) :
  A + C = 201 := by
  sorry

end NUMINAMATH_GPT_Jurassic_Zoo_Total_l2036_203684


namespace NUMINAMATH_GPT_problem_condition_l2036_203658

theorem problem_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 5 * x - 3) → (∀ x : ℝ, |x + 0.4| < b → |f x + 1| < a) ↔ (0 < a ∧ 0 < b ∧ b ≤ a / 5) := by
  sorry

end NUMINAMATH_GPT_problem_condition_l2036_203658


namespace NUMINAMATH_GPT_find_third_number_in_proportion_l2036_203646

theorem find_third_number_in_proportion (x : ℝ) (third_number : ℝ) (h1 : x = 0.9) (h2 : 0.75 / 6 = x / third_number) : third_number = 5 := by
  sorry

end NUMINAMATH_GPT_find_third_number_in_proportion_l2036_203646


namespace NUMINAMATH_GPT_number_of_children_at_matinee_l2036_203680

-- Definitions of constants based on conditions
def children_ticket_price : ℝ := 4.50
def adult_ticket_price : ℝ := 6.75
def total_receipts : ℝ := 405
def additional_children : ℕ := 20

-- Variables for number of adults and children
variable (A C : ℕ)

-- Assertions based on conditions
axiom H1 : C = A + additional_children
axiom H2 : children_ticket_price * (C : ℝ) + adult_ticket_price * (A : ℝ) = total_receipts

-- Theorem statement: Prove that the number of children is 48
theorem number_of_children_at_matinee : C = 48 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_at_matinee_l2036_203680


namespace NUMINAMATH_GPT_find_m_l2036_203657

def A : Set ℤ := {-1, 1}
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

theorem find_m (m : ℤ) (h : B m ⊆ A) : m = 0 ∨ m = 1 ∨ m = -1 := 
sorry

end NUMINAMATH_GPT_find_m_l2036_203657


namespace NUMINAMATH_GPT_min_value_expression_l2036_203655

theorem min_value_expression : ∀ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l2036_203655


namespace NUMINAMATH_GPT_max_xy_l2036_203656

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x + 6 * y < 90) :
  xy * (90 - 5 * x - 6 * y) ≤ 900 := by
  sorry

end NUMINAMATH_GPT_max_xy_l2036_203656


namespace NUMINAMATH_GPT_smallest_denominator_of_sum_of_irreducible_fractions_l2036_203660

theorem smallest_denominator_of_sum_of_irreducible_fractions :
  ∀ (a b : ℕ),
  Nat.Coprime a 600 → Nat.Coprime b 700 →
  (∃ c d : ℕ, Nat.Coprime c d ∧ d < 168 ∧ (7 * a + 6 * b) / Nat.gcd (7 * a + 6 * b) 4200 = c / d) →
  False :=
by
  sorry

end NUMINAMATH_GPT_smallest_denominator_of_sum_of_irreducible_fractions_l2036_203660


namespace NUMINAMATH_GPT_correctProduct_l2036_203664

-- Define the digits reverse function
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- Main theorem statement
theorem correctProduct (a b : ℕ) (h1 : 9 < a ∧ a < 100) (h2 : reverseDigits a * b = 143) : a * b = 341 :=
  sorry -- proof to be provided

end NUMINAMATH_GPT_correctProduct_l2036_203664


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2036_203662

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ a₁ q : ℝ, ∀ n, a n = a₁ * q^n

axiom a_3_eq_2 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2
axiom a_4a_6_eq_16 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 4 * a 6 = 16

theorem geometric_sequence_problem :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2 ∧ a 4 * a 6 = 16 →
  (a 9 - a 11) / (a 5 - a 7) = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2036_203662
