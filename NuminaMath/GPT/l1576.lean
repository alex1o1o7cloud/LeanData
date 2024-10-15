import Mathlib

namespace NUMINAMATH_GPT_maximum_distance_correct_l1576_157651

noncomputable def maximum_distance 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  ℝ :=
3 + Real.sqrt 5

theorem maximum_distance_correct 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  maximum_distance m θ P intersection distance = 3 + Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_maximum_distance_correct_l1576_157651


namespace NUMINAMATH_GPT_infinite_solutions_eq_a_l1576_157658

variable (a x y: ℝ)

-- Define the two equations
def eq1 : Prop := a * x + y - 1 = 0
def eq2 : Prop := 4 * x + a * y - 2 = 0

theorem infinite_solutions_eq_a (h : ∃ x y, eq1 a x y ∧ eq2 a x y) :
  a = 2 := 
sorry

end NUMINAMATH_GPT_infinite_solutions_eq_a_l1576_157658


namespace NUMINAMATH_GPT_jennifer_fruits_left_l1576_157687

open Nat

theorem jennifer_fruits_left :
  (p o a g : ℕ) → p = 10 → o = 20 → a = 2 * p → g = 2 → (p - g) + (o - g) + (a - g) = 44 :=
by
  intros p o a g h_p h_o h_a h_g
  rw [h_p, h_o, h_a, h_g]
  sorry

end NUMINAMATH_GPT_jennifer_fruits_left_l1576_157687


namespace NUMINAMATH_GPT_distance_between_stripes_l1576_157689

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end NUMINAMATH_GPT_distance_between_stripes_l1576_157689


namespace NUMINAMATH_GPT_tangent_expression_l1576_157623

theorem tangent_expression :
  (Real.tan (10 * Real.pi / 180) + Real.tan (50 * Real.pi / 180) + Real.tan (120 * Real.pi / 180))
  / (Real.tan (10 * Real.pi / 180) * Real.tan (50 * Real.pi / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tangent_expression_l1576_157623


namespace NUMINAMATH_GPT_purchasing_methods_count_l1576_157622

def material_cost : ℕ := 40
def instrument_cost : ℕ := 60
def budget : ℕ := 400
def min_materials : ℕ := 4
def min_instruments : ℕ := 2

theorem purchasing_methods_count : 
  (∃ (n_m m : ℕ), 
    n_m ≥ min_materials ∧ m ≥ min_instruments ∧ 
    n_m * material_cost + m * instrument_cost ≤ budget) → 
  (∃ (count : ℕ), count = 7) :=
by 
  sorry

end NUMINAMATH_GPT_purchasing_methods_count_l1576_157622


namespace NUMINAMATH_GPT_power_function_no_origin_l1576_157607

theorem power_function_no_origin (m : ℝ) : 
  (m^2 - m - 1 <= 0) ∧ (m^2 - 3 * m + 3 = 1) → m = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_power_function_no_origin_l1576_157607


namespace NUMINAMATH_GPT_power_function_increasing_is_3_l1576_157601

theorem power_function_increasing_is_3 (m : ℝ) :
  (∀ x : ℝ, x > 0 → (m^2 - m - 5) * (x^(m)) > 0) ∧ (m^2 - m - 5 = 1) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_power_function_increasing_is_3_l1576_157601


namespace NUMINAMATH_GPT_radio_price_rank_l1576_157648

theorem radio_price_rank (total_items : ℕ) (radio_position_highest : ℕ) (radio_position_lowest : ℕ) 
  (h1 : total_items = 40) (h2 : radio_position_highest = 17) : 
  radio_position_lowest = total_items - radio_position_highest + 1 :=
by
  sorry

end NUMINAMATH_GPT_radio_price_rank_l1576_157648


namespace NUMINAMATH_GPT_volume_of_prism_l1576_157625

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 15) (hwh : w * h = 20) (hlh : l * h = 24) : l * w * h = 60 := 
sorry

end NUMINAMATH_GPT_volume_of_prism_l1576_157625


namespace NUMINAMATH_GPT_solve_x_l1576_157695

noncomputable def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

theorem solve_x (x : ℝ) (h : op (x - 1) 2 = 1) : x = -1 := 
by {
  -- proof outline here...
  sorry
}

end NUMINAMATH_GPT_solve_x_l1576_157695


namespace NUMINAMATH_GPT_arithmetic_progression_a6_l1576_157612

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_a6_l1576_157612


namespace NUMINAMATH_GPT_bacteria_growth_rate_l1576_157614

theorem bacteria_growth_rate (r : ℝ) :
  (1 + r)^6 = 64 → r = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bacteria_growth_rate_l1576_157614


namespace NUMINAMATH_GPT_inequality_true_l1576_157654

noncomputable def f : ℝ → ℝ := sorry -- f is a function defined on (0, +∞)

axiom f_derivative (x : ℝ) (hx : 0 < x) : ∃ f'' : ℝ → ℝ, f'' x * x + 2 * f x = 1 / x^2

theorem inequality_true : (f 2) / 9 < (f 3) / 4 :=
  sorry

end NUMINAMATH_GPT_inequality_true_l1576_157654


namespace NUMINAMATH_GPT_Zilla_savings_l1576_157660

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end NUMINAMATH_GPT_Zilla_savings_l1576_157660


namespace NUMINAMATH_GPT_proof_problem_l1576_157661

theorem proof_problem
  (x y : ℤ)
  (hx : ∃ m : ℤ, x = 6 * m)
  (hy : ∃ n : ℤ, y = 12 * n) :
  (x + y) % 2 = 0 ∧ (x + y) % 6 = 0 ∧ ¬ (x + y) % 12 = 0 → ¬ (x + y) % 12 = 0 :=
  sorry

end NUMINAMATH_GPT_proof_problem_l1576_157661


namespace NUMINAMATH_GPT_average_chemistry_mathematics_l1576_157676

noncomputable def marks (P C M B : ℝ) : Prop := 
  P + C + M + B = (P + B) + 180 ∧ P = 1.20 * B

theorem average_chemistry_mathematics 
  (P C M B : ℝ) (h : marks P C M B) : (C + M) / 2 = 90 :=
by
  sorry

end NUMINAMATH_GPT_average_chemistry_mathematics_l1576_157676


namespace NUMINAMATH_GPT_unique_10_tuple_solution_l1576_157686

noncomputable def condition (x : Fin 10 → ℝ) : Prop :=
  (1 - x 0)^2 +
  (x 0 - x 1)^2 + 
  (x 1 - x 2)^2 + 
  (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + 
  (x 4 - x 5)^2 + 
  (x 5 - x 6)^2 + 
  (x 6 - x 7)^2 + 
  (x 7 - x 8)^2 + 
  (x 8 - x 9)^2 + 
  x 9^2 + 
  (1/2) * (x 9 - x 0)^2 = 1/10

theorem unique_10_tuple_solution : 
  ∃! (x : Fin 10 → ℝ), condition x := 
sorry

end NUMINAMATH_GPT_unique_10_tuple_solution_l1576_157686


namespace NUMINAMATH_GPT_avg_weight_difference_l1576_157677

-- Define the weights of the boxes following the given conditions.
def box1_weight : ℕ := 200
def box3_weight : ℕ := box1_weight + (25 * box1_weight / 100)
def box2_weight : ℕ := box3_weight + (20 * box3_weight / 100)
def box4_weight : ℕ := 350
def box5_weight : ℕ := box4_weight * 100 / 70

-- Define the average weight of the four heaviest boxes.
def avg_heaviest : ℕ := (box2_weight + box3_weight + box4_weight + box5_weight) / 4

-- Define the average weight of the four lightest boxes.
def avg_lightest : ℕ := (box1_weight + box2_weight + box3_weight + box4_weight) / 4

-- Define the difference between the average weights of the heaviest and lightest boxes.
def avg_difference : ℕ := avg_heaviest - avg_lightest

-- State the theorem with the expected result.
theorem avg_weight_difference : avg_difference = 75 :=
by
  -- Proof is not provided.
  sorry

end NUMINAMATH_GPT_avg_weight_difference_l1576_157677


namespace NUMINAMATH_GPT_smallest_consecutive_integer_sum_l1576_157696

-- Definitions based on conditions
def consecutive_integer_sum (n : ℕ) := 20 * n + 190

-- Theorem statement
theorem smallest_consecutive_integer_sum : 
  ∃ (n k : ℕ), (consecutive_integer_sum n = k^3) ∧ (∀ m l : ℕ, (consecutive_integer_sum m = l^3) → k^3 ≤ l^3) :=
sorry

end NUMINAMATH_GPT_smallest_consecutive_integer_sum_l1576_157696


namespace NUMINAMATH_GPT_bjorn_cannot_prevent_vakha_l1576_157613

-- Define the primary settings and objects involved
def n_points : ℕ := 99
inductive Color
| red 
| blue 

structure GameState :=
  (turn : ℕ)
  (points : Fin n_points → Option Color)

-- Define the valid states of the game where turn must be within the range of points
def valid_state (s : GameState) : Prop :=
  s.turn ≤ n_points ∧ ∀ p, s.points p ≠ none

-- Define what it means for an equilateral triangle to be monochromatically colored
def monochromatic_equilateral_triangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Fin n_points), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (p1.val + (n_points/3) % n_points) = p2.val ∧
    (p2.val + (n_points/3) % n_points) = p3.val ∧
    (p3.val + (n_points/3) % n_points) = p1.val ∧
    (state.points p1 = state.points p2) ∧ 
    (state.points p2 = state.points p3)

-- Vakha's winning condition
def vakha_wins (state : GameState) : Prop := 
  monochromatic_equilateral_triangle state

-- Bjorn's winning condition prevents Vakha from winning
def bjorn_can_prevent_vakha (initial_state : GameState) : Prop :=
  ¬ vakha_wins initial_state

-- Main theorem stating Bjorn cannot prevent Vakha from winning
theorem bjorn_cannot_prevent_vakha : ∀ (initial_state : GameState),
  valid_state initial_state → ¬ bjorn_can_prevent_vakha initial_state :=
sorry

end NUMINAMATH_GPT_bjorn_cannot_prevent_vakha_l1576_157613


namespace NUMINAMATH_GPT_cos_angle_identity_l1576_157644

theorem cos_angle_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_angle_identity_l1576_157644


namespace NUMINAMATH_GPT_sin_alpha_given_cos_alpha_plus_pi_over_3_l1576_157645

theorem sin_alpha_given_cos_alpha_plus_pi_over_3 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := 
by 
  sorry

end NUMINAMATH_GPT_sin_alpha_given_cos_alpha_plus_pi_over_3_l1576_157645


namespace NUMINAMATH_GPT_drying_time_l1576_157679

theorem drying_time
  (time_short : ℕ := 10) -- Time to dry a short-haired dog in minutes
  (time_full : ℕ := time_short * 2) -- Time to dry a full-haired dog in minutes, which is twice as long
  (num_short : ℕ := 6) -- Number of short-haired dogs
  (num_full : ℕ := 9) -- Number of full-haired dogs
  : (time_short * num_short + time_full * num_full) / 60 = 4 := 
by
  sorry

end NUMINAMATH_GPT_drying_time_l1576_157679


namespace NUMINAMATH_GPT_area_difference_l1576_157638

-- Definitions of the given conditions
structure Triangle :=
(base : ℝ)
(height : ℝ)

def area (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

-- Conditions of the problem
def EFG : Triangle := {base := 8, height := 4}
def EFG' : Triangle := {base := 4, height := 2}

-- Proof statement
theorem area_difference :
  area EFG - area EFG' = 12 :=
by
  sorry

end NUMINAMATH_GPT_area_difference_l1576_157638


namespace NUMINAMATH_GPT_minimum_omega_l1576_157688

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)
noncomputable def h (ω : ℝ) (x : ℝ) : ℝ := f ω x + g ω x

theorem minimum_omega (ω : ℝ) (m : ℝ) 
  (h1 : 0 < ω)
  (h2 : ∀ x : ℝ, h ω m ≤ h ω x ∧ h ω x ≤ h ω (m + 1)) :
  ω = π :=
by
  sorry

end NUMINAMATH_GPT_minimum_omega_l1576_157688


namespace NUMINAMATH_GPT_seating_arrangements_l1576_157609

/-- 
Given seven seats in a row, with four people sitting such that exactly two adjacent seats are empty,
prove that the number of different seating arrangements is 480.
-/
theorem seating_arrangements (seats people : ℕ) (adj_empty : ℕ) : 
  seats = 7 → people = 4 → adj_empty = 2 → 
  (∃ count : ℕ, count = 480) :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l1576_157609


namespace NUMINAMATH_GPT_average_minutes_run_is_44_over_3_l1576_157665

open BigOperators

def average_minutes_run (s : ℕ) : ℚ :=
  let sixth_graders := 3 * s
  let seventh_graders := s
  let eighth_graders := s / 2
  let total_students := sixth_graders + seventh_graders + eighth_graders
  let total_minutes_run := 20 * sixth_graders + 12 * eighth_graders
  total_minutes_run / total_students

theorem average_minutes_run_is_44_over_3 (s : ℕ) (h1 : 0 < s) : 
  average_minutes_run s = 44 / 3 := 
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_is_44_over_3_l1576_157665


namespace NUMINAMATH_GPT_factor_polynomial_l1576_157669

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1576_157669


namespace NUMINAMATH_GPT_solve_system_of_equations_l1576_157667

theorem solve_system_of_equations :
  ∀ (x1 x2 x3 x4 x5: ℝ), 
  (x3 + x4 + x5)^5 = 3 * x1 ∧ 
  (x4 + x5 + x1)^5 = 3 * x2 ∧ 
  (x5 + x1 + x2)^5 = 3 * x3 ∧ 
  (x1 + x2 + x3)^5 = 3 * x4 ∧ 
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨ 
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨ 
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) := 
by 
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1576_157667


namespace NUMINAMATH_GPT_third_set_candies_l1576_157615

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end NUMINAMATH_GPT_third_set_candies_l1576_157615


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l1576_157650

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180)
(h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l1576_157650


namespace NUMINAMATH_GPT_find_rth_term_l1576_157620

def S (n : ℕ) : ℕ := 2 * n + 3 * (n^3)

def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem find_rth_term (r : ℕ) : a r = 9 * r^2 - 9 * r + 5 := by
  sorry

end NUMINAMATH_GPT_find_rth_term_l1576_157620


namespace NUMINAMATH_GPT_cuboid_third_edge_length_l1576_157619

theorem cuboid_third_edge_length
  (l w : ℝ)
  (A : ℝ)
  (h : ℝ)
  (hl : l = 4)
  (hw : w = 5)
  (hA : A = 148)
  (surface_area_formula : A = 2 * (l * w + l * h + w * h)) :
  h = 6 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_third_edge_length_l1576_157619


namespace NUMINAMATH_GPT_smallest_number_of_coins_l1576_157610

theorem smallest_number_of_coins :
  ∃ pennies nickels dimes quarters half_dollars : ℕ,
    pennies + nickels + dimes + quarters + half_dollars = 6 ∧
    (∀ amount : ℕ, amount < 100 →
      ∃ p n d q h : ℕ,
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧ h ≤ half_dollars ∧
        1 * p + 5 * n + 10 * d + 25 * q + 50 * h = amount) :=
sorry

end NUMINAMATH_GPT_smallest_number_of_coins_l1576_157610


namespace NUMINAMATH_GPT_midpoint_coordinates_l1576_157697

theorem midpoint_coordinates (A B M : ℝ × ℝ) (hx : A = (2, -4)) (hy : B = (-6, 2)) (hm : M = (-2, -1)) :
  let (x1, y1) := A
  let (x2, y2) := B
  M = ((x1 + x2) / 2, (y1 + y2) / 2) :=
  sorry

end NUMINAMATH_GPT_midpoint_coordinates_l1576_157697


namespace NUMINAMATH_GPT_find_min_max_value_l1576_157626

open Real

theorem find_min_max_value (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) (h_det : b^2 - 4 * a * c < 0) :
  ∃ (min_val max_val: ℝ),
    min_val = (2 * d * sqrt (a * c)) / (b + 2 * sqrt (a * c)) ∧ 
    max_val = (2 * d * sqrt (a * c)) / (b - 2 * sqrt (a * c)) ∧
    (∀ x y : ℝ, a * x^2 + c * y^2 ≥ min_val ∧ a * x^2 + c * y^2 ≤ max_val) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_min_max_value_l1576_157626


namespace NUMINAMATH_GPT_person_B_reads_more_than_A_l1576_157608

-- Assuming people are identifiers for Person A and Person B.
def pages_read_A (days : ℕ) (daily_read : ℕ) : ℕ := days * daily_read

def pages_read_B (days : ℕ) (daily_read : ℕ) (rest_cycle : ℕ) : ℕ := 
  let full_cycles := days / rest_cycle
  let remainder_days := days % rest_cycle
  let active_days := days - full_cycles
  active_days * daily_read

-- Given conditions
def daily_read_A := 8
def daily_read_B := 13
def rest_cycle_B := 3
def total_days := 7

-- The main theorem to prove
theorem person_B_reads_more_than_A : 
  (pages_read_B total_days daily_read_B rest_cycle_B) - (pages_read_A total_days daily_read_A) = 9 :=
by
  sorry

end NUMINAMATH_GPT_person_B_reads_more_than_A_l1576_157608


namespace NUMINAMATH_GPT_find_number_l1576_157675

theorem find_number (x : ℕ) (h : 5 + 2 * (8 - x) = 15) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_number_l1576_157675


namespace NUMINAMATH_GPT_less_than_subtraction_l1576_157655

-- Define the numbers as real numbers
def a : ℝ := 47.2
def b : ℝ := 0.5

-- Theorem statement
theorem less_than_subtraction : a - b = 46.7 :=
by
  sorry

end NUMINAMATH_GPT_less_than_subtraction_l1576_157655


namespace NUMINAMATH_GPT_not_sufficient_nor_necessary_l1576_157671

theorem not_sufficient_nor_necessary (a b : ℝ) : ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) := 
by 
  sorry

end NUMINAMATH_GPT_not_sufficient_nor_necessary_l1576_157671


namespace NUMINAMATH_GPT_find_a_l1576_157602

-- Conditions as definitions:
variable (a : ℝ) (b : ℝ)
variable (A : ℝ × ℝ := (0, 0)) (B : ℝ × ℝ := (a, 0)) (C : ℝ × ℝ := (0, b))
noncomputable def area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Given conditions:
axiom h1 : b = 4
axiom h2 : area a b = 28
axiom h3 : a > 0

-- The proof goal:
theorem find_a : a = 14 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_a_l1576_157602


namespace NUMINAMATH_GPT_complex_number_calculation_l1576_157680

theorem complex_number_calculation (i : ℂ) (h : i * i = -1) : i^7 - 2/i = i := 
by 
  sorry

end NUMINAMATH_GPT_complex_number_calculation_l1576_157680


namespace NUMINAMATH_GPT_snail_total_distance_l1576_157624

-- Conditions
def initial_pos : ℤ := 0
def pos1 : ℤ := 4
def pos2 : ℤ := -3
def pos3 : ℤ := 6

-- Total distance traveled by the snail
def distance_traveled : ℤ :=
  abs (pos1 - initial_pos) +
  abs (pos2 - pos1) +
  abs (pos3 - pos2)

-- Theorem statement
theorem snail_total_distance : distance_traveled = 20 :=
by
  -- Proof is omitted, as per request
  sorry

end NUMINAMATH_GPT_snail_total_distance_l1576_157624


namespace NUMINAMATH_GPT_inequality_holds_l1576_157692

theorem inequality_holds (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 :=
by sorry

end NUMINAMATH_GPT_inequality_holds_l1576_157692


namespace NUMINAMATH_GPT_cyclic_quadrilaterals_count_l1576_157616

theorem cyclic_quadrilaterals_count :
  ∃ n : ℕ, n = 568 ∧
  ∀ (a b c d : ℕ), 
    a + b + c + d = 32 ∧
    a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c > d) ∧ (b + c + d > a) ∧ (c + d + a > b) ∧ (d + a + b > c) ∧
    (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2
      → n = 568 := 
sorry

end NUMINAMATH_GPT_cyclic_quadrilaterals_count_l1576_157616


namespace NUMINAMATH_GPT_number_of_whole_numbers_between_roots_l1576_157693

theorem number_of_whole_numbers_between_roots :
  let sqrt_18 := Real.sqrt 18
  let sqrt_98 := Real.sqrt 98
  Nat.card { x : ℕ | sqrt_18 < x ∧ x < sqrt_98 } = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_whole_numbers_between_roots_l1576_157693


namespace NUMINAMATH_GPT_rowing_problem_l1576_157636

theorem rowing_problem (R S x y : ℝ) 
  (h1 : R = y + x) 
  (h2 : S = y - x) : 
  x = (R - S) / 2 ∧ y = (R + S) / 2 :=
by
  sorry

end NUMINAMATH_GPT_rowing_problem_l1576_157636


namespace NUMINAMATH_GPT_symmetric_points_tangent_line_l1576_157682

theorem symmetric_points_tangent_line (k : ℝ) (hk : 0 < k) :
  (∃ P Q : ℝ × ℝ, P.2 = Real.exp P.1 ∧ ∃ x₀ : ℝ, 
    Q.2 = k * Q.1 ∧ Q = (P.2, P.1) ∧ 
    Q.1 = x₀ ∧ k = 1 / x₀ ∧ x₀ = Real.exp 1) → k = 1 / Real.exp 1 := 
by 
  sorry

end NUMINAMATH_GPT_symmetric_points_tangent_line_l1576_157682


namespace NUMINAMATH_GPT_substance_same_number_of_atoms_l1576_157618

def molecule (kind : String) (atom_count : ℕ) := (kind, atom_count)

def H3PO4 := molecule "H₃PO₄" 8
def H2O2 := molecule "H₂O₂" 4
def H2SO4 := molecule "H₂SO₄" 7
def NaCl := molecule "NaCl" 2 -- though it consists of ions, let's denote it as 2 for simplicity
def HNO3 := molecule "HNO₃" 5

def mol_atoms (mol : ℝ) (molecule : ℕ) : ℝ := mol * molecule

theorem substance_same_number_of_atoms :
  mol_atoms 0.2 H3PO4.2 = mol_atoms 0.4 H2O2.2 :=
by
  unfold H3PO4 H2O2 mol_atoms
  sorry

end NUMINAMATH_GPT_substance_same_number_of_atoms_l1576_157618


namespace NUMINAMATH_GPT_total_laundry_time_correct_l1576_157617

-- Define the washing and drying times for each load
def whites_washing_time : Nat := 72
def whites_drying_time : Nat := 50
def darks_washing_time : Nat := 58
def darks_drying_time : Nat := 65
def colors_washing_time : Nat := 45
def colors_drying_time : Nat := 54

-- Define total times for each load
def whites_total_time : Nat := whites_washing_time + whites_drying_time
def darks_total_time : Nat := darks_washing_time + darks_drying_time
def colors_total_time : Nat := colors_washing_time + colors_drying_time

-- Define the total time for all three loads
def total_laundry_time : Nat := whites_total_time + darks_total_time + colors_total_time

-- The proof statement
theorem total_laundry_time_correct : total_laundry_time = 344 := by
  unfold total_laundry_time
  unfold whites_total_time darks_total_time colors_total_time
  unfold whites_washing_time whites_drying_time
  unfold darks_washing_time darks_drying_time
  unfold colors_washing_time colors_drying_time
  sorry

end NUMINAMATH_GPT_total_laundry_time_correct_l1576_157617


namespace NUMINAMATH_GPT_max_a_for_necessary_not_sufficient_condition_l1576_157642

theorem max_a_for_necessary_not_sufficient_condition {x a : ℝ} (h : ∀ x, x^2 > 1 → x < a) : a = -1 :=
by sorry

end NUMINAMATH_GPT_max_a_for_necessary_not_sufficient_condition_l1576_157642


namespace NUMINAMATH_GPT_whiteboard_ink_cost_l1576_157672

/-- 
There are 5 classes: A, B, C, D, E
Class A: 3 whiteboards
Class B: 2 whiteboards
Class C: 4 whiteboards
Class D: 1 whiteboard
Class E: 3 whiteboards
The ink usage per whiteboard in each class:
Class A: 20ml per whiteboard
Class B: 25ml per whiteboard
Class C: 15ml per whiteboard
Class D: 30ml per whiteboard
Class E: 20ml per whiteboard
The cost of ink is 50 cents per ml
-/
def total_cost_in_dollars : ℕ :=
  let ink_usage_A := 3 * 20
  let ink_usage_B := 2 * 25
  let ink_usage_C := 4 * 15
  let ink_usage_D := 1 * 30
  let ink_usage_E := 3 * 20
  let total_ink_usage := ink_usage_A + ink_usage_B + ink_usage_C + ink_usage_D + ink_usage_E
  let total_cost_in_cents := total_ink_usage * 50
  total_cost_in_cents / 100

theorem whiteboard_ink_cost : total_cost_in_dollars = 130 := 
  by 
    sorry -- Proof needs to be implemented

end NUMINAMATH_GPT_whiteboard_ink_cost_l1576_157672


namespace NUMINAMATH_GPT_value_of_x_l1576_157698

theorem value_of_x (x c m n : ℝ) (hne: m≠n) (hneq : c ≠ 0) 
  (h1: c = 3) (h2: m = 2) (h3: n = 5)
  (h4: (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) : 
  x = -11 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1576_157698


namespace NUMINAMATH_GPT_geometric_sequence_general_term_arithmetic_sequence_sum_l1576_157605

noncomputable def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 4 * (n - 1)

def S (n : ℕ) : ℕ := 2 * n^2 - 2 * n

theorem geometric_sequence_general_term
    (a1 : ℕ := 2)
    (a4 : ℕ := 16)
    (h1 : a 1 = a1)
    (h2 : a 4 = a4)
    : ∀ n : ℕ, a n = a 1 * 2^(n-1) :=
by
  sorry

theorem arithmetic_sequence_sum
    (a2 : ℕ := 4)
    (a5 : ℕ := 32)
    (b2 : ℕ := a 2)
    (b9 : ℕ := a 5)
    (h1 : b 2 = b2)
    (h2 : b 9 = b9)
    : ∀ n : ℕ, S n = n * (n - 1) * 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_arithmetic_sequence_sum_l1576_157605


namespace NUMINAMATH_GPT_original_sheets_count_is_115_l1576_157699

def find_sheets_count (S P : ℕ) : Prop :=
  -- Ann's condition: all papers are used leaving 100 flyers
  S - P = 100 ∧
  -- Bob's condition: all bindings used leaving 35 sheets of paper
  5 * P = S - 35

theorem original_sheets_count_is_115 (S P : ℕ) (h : find_sheets_count S P) : S = 115 :=
by
  sorry

end NUMINAMATH_GPT_original_sheets_count_is_115_l1576_157699


namespace NUMINAMATH_GPT_value_of_a_plus_c_l1576_157653

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def f_inv (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem value_of_a_plus_c : a + c = -1 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_c_l1576_157653


namespace NUMINAMATH_GPT_max_sum_value_l1576_157603

noncomputable def maxSum (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : ℤ :=
  i + j + k

theorem max_sum_value (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  maxSum i j k h ≤ 77 :=
  sorry

end NUMINAMATH_GPT_max_sum_value_l1576_157603


namespace NUMINAMATH_GPT_inverse_of_matrix_l1576_157633

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 9], ![2, 5]]

def inv_mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5/2, -9/2], ![-1, 2]]

theorem inverse_of_matrix :
  ∃ (inv : Matrix (Fin 2) (Fin 2) ℚ), 
    inv * mat = 1 ∧ mat * inv = 1 :=
  ⟨inv_mat, by
    -- Providing the proof steps here is beyond the scope
    sorry⟩

end NUMINAMATH_GPT_inverse_of_matrix_l1576_157633


namespace NUMINAMATH_GPT_sin_cos_identity_l1576_157681

theorem sin_cos_identity (a : ℝ) (h : Real.sin (π - a) = -2 * Real.sin (π / 2 + a)) : 
  Real.sin a * Real.cos a = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l1576_157681


namespace NUMINAMATH_GPT_min_value_of_quadratic_l1576_157628

theorem min_value_of_quadratic (x y : ℝ) : (x^2 + 2*x*y + y^2) ≥ 0 ∧ ∃ x y, x = -y ∧ x^2 + 2*x*y + y^2 = 0 := by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1576_157628


namespace NUMINAMATH_GPT_rectangular_board_area_l1576_157656

variable (length width : ℕ)

theorem rectangular_board_area
  (h1 : length = 2 * width)
  (h2 : 2 * length + 2 * width = 84) :
  length * width = 392 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_board_area_l1576_157656


namespace NUMINAMATH_GPT_domain_h_l1576_157637

noncomputable def h (x : ℝ) : ℝ := (x^2 + 5 * x + 6) / (|x - 2| + |x + 2|)

theorem domain_h : ∀ x : ℝ, ∃ y : ℝ, y = h x :=
by
  sorry

end NUMINAMATH_GPT_domain_h_l1576_157637


namespace NUMINAMATH_GPT_diophantine_equation_solvable_l1576_157674

theorem diophantine_equation_solvable (a : ℕ) (ha : 0 < a) : 
  ∃ (x y : ℤ), x^2 - y^2 = a^3 :=
by
  let x := (a * (a + 1)) / 2
  let y := (a * (a - 1)) / 2
  have hx : x^2 = (a * (a + 1) / 2 : ℤ)^2 := sorry
  have hy : y^2 = (a * (a - 1) / 2 : ℤ)^2 := sorry
  use x
  use y
  sorry

end NUMINAMATH_GPT_diophantine_equation_solvable_l1576_157674


namespace NUMINAMATH_GPT_fourth_number_unit_digit_l1576_157683

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit (a b c d : ℕ) (h₁ : a = 7858) (h₂: b = 1086) (h₃ : c = 4582) (h₄ : unit_digit (a * b * c * d) = 8) :
  unit_digit d = 4 :=
sorry

end NUMINAMATH_GPT_fourth_number_unit_digit_l1576_157683


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1576_157664

theorem solve_eq1 (x : ℝ) : (x+1)^2 = 4 ↔ x = 1 ∨ x = -3 := 
by sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 2*x - 1 = 0 ↔ x = 1 ∨ x = -1/3 := 
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1576_157664


namespace NUMINAMATH_GPT_person_B_winning_strategy_l1576_157684

-- Definitions for the problem conditions
def winning_strategy_condition (L a b : ℕ) : Prop := 
  b = 2 * a ∧ ∃ k : ℕ, L = k * a

-- Lean theorem statement for the given problem
theorem person_B_winning_strategy (L a b : ℕ) (hL_pos : 0 < L) (ha_lt_hb : a < b) 
(hpos_a : 0 < a) (hpos_b : 0 < b) : 
  (∃ B_strat : Type, winning_strategy_condition L a b) :=
sorry

end NUMINAMATH_GPT_person_B_winning_strategy_l1576_157684


namespace NUMINAMATH_GPT_find_root_interval_l1576_157629

noncomputable def f : ℝ → ℝ := sorry

theorem find_root_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 < 0 ∧ f 2.75 > 0 ∧ f 2.625 > 0 ∧ f 2.5625 > 0 →
  ∃ x, 2.5 < x ∧ x < 2.5625 ∧ f x = 0 := sorry

end NUMINAMATH_GPT_find_root_interval_l1576_157629


namespace NUMINAMATH_GPT_inverse_proportionality_l1576_157627

theorem inverse_proportionality:
  (∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = k / x) ∧ y = 1 ∧ x = 2 →
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = 2 / x :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportionality_l1576_157627


namespace NUMINAMATH_GPT_cost_of_each_pack_l1576_157634

theorem cost_of_each_pack (num_packs : ℕ) (total_paid : ℝ) (change_received : ℝ) 
(h1 : num_packs = 3) (h2 : total_paid = 20) (h3 : change_received = 11) : 
(total_paid - change_received) / num_packs = 3 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_pack_l1576_157634


namespace NUMINAMATH_GPT_moles_of_C2H6_formed_l1576_157690

-- Define the initial conditions
def initial_moles_H2 : ℕ := 3
def initial_moles_C2H4 : ℕ := 3
def reaction_ratio_C2H4_H2_C2H6 (C2H4 H2 C2H6 : ℕ) : Prop :=
  C2H4 = H2 ∧ C2H4 = C2H6

-- State the theorem to prove
theorem moles_of_C2H6_formed : reaction_ratio_C2H4_H2_C2H6 initial_moles_C2H4 initial_moles_H2 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_moles_of_C2H6_formed_l1576_157690


namespace NUMINAMATH_GPT_q_investment_l1576_157643

theorem q_investment (p_investment : ℝ) (profit_ratio_p : ℝ) (profit_ratio_q : ℝ) (q_investment : ℝ) 
  (h1 : p_investment = 40000) 
  (h2 : profit_ratio_p / profit_ratio_q = 2 / 3) 
  : q_investment = 60000 := 
sorry

end NUMINAMATH_GPT_q_investment_l1576_157643


namespace NUMINAMATH_GPT_sum_of_all_three_digit_positive_even_integers_l1576_157611

def sum_of_three_digit_even_integers : ℕ :=
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_all_three_digit_positive_even_integers :
  sum_of_three_digit_even_integers = 247050 :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_sum_of_all_three_digit_positive_even_integers_l1576_157611


namespace NUMINAMATH_GPT_B_and_C_together_l1576_157685

-- Defining the variables and conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 500)
variable (h2 : A + C = 200)
variable (h3 : C = 50)

-- The theorem to prove that B + C = 350
theorem B_and_C_together : B + C = 350 :=
by
  -- Replacing with the actual proof steps
  sorry

end NUMINAMATH_GPT_B_and_C_together_l1576_157685


namespace NUMINAMATH_GPT_perfect_square_condition_l1576_157631

theorem perfect_square_condition (n : ℤ) : 
    ∃ k : ℤ, n^2 + 6*n + 1 = k^2 ↔ n = 0 ∨ n = -6 := by
  sorry

end NUMINAMATH_GPT_perfect_square_condition_l1576_157631


namespace NUMINAMATH_GPT_equivalent_operation_l1576_157652

theorem equivalent_operation (x : ℚ) :
  (x / (5 / 6) * (4 / 7)) = x * (24 / 35) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_operation_l1576_157652


namespace NUMINAMATH_GPT_remaining_slices_correct_l1576_157621

-- Define initial slices of pie and cake
def initial_pie_slices : Nat := 2 * 8
def initial_cake_slices : Nat := 12

-- Define slices eaten on Friday
def friday_pie_slices_eaten : Nat := 2
def friday_cake_slices_eaten : Nat := 2

-- Define slices eaten on Saturday
def saturday_pie_slices_eaten (remaining: Nat) : Nat := remaining / 2 -- 50%
def saturday_cake_slices_eaten (remaining: Nat) : Nat := remaining / 4 -- 25%

-- Define slices eaten on Sunday morning
def sunday_morning_pie_slices_eaten : Nat := 2
def sunday_morning_cake_slices_eaten : Nat := 3

-- Define slices eaten on Sunday evening
def sunday_evening_pie_slices_eaten : Nat := 4
def sunday_evening_cake_slices_eaten : Nat := 1

-- Function to calculate remaining slices
def remaining_slices : Nat × Nat :=
  let after_friday_pies := initial_pie_slices - friday_pie_slices_eaten
  let after_friday_cake := initial_cake_slices - friday_cake_slices_eaten
  let after_saturday_pies := after_friday_pies - saturday_pie_slices_eaten after_friday_pies
  let after_saturday_cake := after_friday_cake - saturday_cake_slices_eaten after_friday_cake
  let after_sunday_morning_pies := after_saturday_pies - sunday_morning_pie_slices_eaten
  let after_sunday_morning_cake := after_saturday_cake - sunday_morning_cake_slices_eaten
  let final_pies := after_sunday_morning_pies - sunday_evening_pie_slices_eaten
  let final_cake := after_sunday_morning_cake - sunday_evening_cake_slices_eaten
  (final_pies, final_cake)

theorem remaining_slices_correct :
  remaining_slices = (1, 4) :=
  by {
    sorry -- Proof is omitted
  }

end NUMINAMATH_GPT_remaining_slices_correct_l1576_157621


namespace NUMINAMATH_GPT_sachin_age_l1576_157641

theorem sachin_age (S R : ℕ) (h1 : R = S + 18) (h2 : S * 9 = R * 7) : S = 63 := 
by
  sorry

end NUMINAMATH_GPT_sachin_age_l1576_157641


namespace NUMINAMATH_GPT_problem_l1576_157678

theorem problem (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : p + 5 < q)
  (h2 : (p + (p + 2) + (p + 5) + q + (q + 1) + (2 * q - 1)) / 6 = q)
  (h3 : (p + 5 + q) / 2 = q) : p + q = 11 :=
by sorry

end NUMINAMATH_GPT_problem_l1576_157678


namespace NUMINAMATH_GPT_heptagon_diagonals_l1576_157600

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end NUMINAMATH_GPT_heptagon_diagonals_l1576_157600


namespace NUMINAMATH_GPT_bus_initial_count_l1576_157635

theorem bus_initial_count (x : ℕ) (got_off : ℕ) (remained : ℕ) (h1 : got_off = 47) (h2 : remained = 43) (h3 : x - got_off = remained) : x = 90 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_bus_initial_count_l1576_157635


namespace NUMINAMATH_GPT_divide_gray_area_l1576_157647

-- The conditions
variables {A_rectangle A_square : ℝ} (h : 0 ≤ A_square ∧ A_square ≤ A_rectangle)

-- The main statement
theorem divide_gray_area : ∃ l : ℝ → ℝ → Prop, (∀ (x : ℝ), l x (A_rectangle / 2)) ∧ (∀ (y : ℝ), l (A_square / 2) y) ∧ (A_rectangle - A_square) / 2 = (A_rectangle - A_square) / 2 := by sorry

end NUMINAMATH_GPT_divide_gray_area_l1576_157647


namespace NUMINAMATH_GPT_largest_square_area_correct_l1576_157640

noncomputable def area_of_largest_square (x y z : ℝ) : Prop := 
  ∃ (area : ℝ), (z^2 = area) ∧ 
                 (x^2 + y^2 = z^2) ∧ 
                 (x^2 + y^2 + 2*z^2 = 722) ∧ 
                 (area = 722 / 3)

theorem largest_square_area_correct (x y z : ℝ) :
  area_of_largest_square x y z :=
  sorry

end NUMINAMATH_GPT_largest_square_area_correct_l1576_157640


namespace NUMINAMATH_GPT_geo_sequence_sum_l1576_157604

theorem geo_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 2)
  (h2 : a 4 + a 5 = 4)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  a 10 + a 11 = 16 := by
  -- Insert proof here
  sorry  -- skipping the proof

end NUMINAMATH_GPT_geo_sequence_sum_l1576_157604


namespace NUMINAMATH_GPT_angle_AXC_angle_ACB_l1576_157646

-- Definitions of the problem conditions
variables (A B C D X : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty X]
variables (AD DC: Type) [Nonempty AD] [Nonempty DC]
variables (angleB angleXDC angleAXC angleACB : ℝ)
variables (AB BX: ℝ)

-- Given conditions
axiom equal_sides: AD = DC
axiom pointX: BX = AB
axiom given_angleB: angleB = 34
axiom given_angleXDC: angleXDC = 52

-- Proof goals (no proof included, only the statements)
theorem angle_AXC: angleAXC = 107 :=
sorry

theorem angle_ACB: angleACB = 47 :=
sorry

end NUMINAMATH_GPT_angle_AXC_angle_ACB_l1576_157646


namespace NUMINAMATH_GPT_smallest_period_of_f_l1576_157639

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x) ^ 2 + 1

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_smallest_period_of_f_l1576_157639


namespace NUMINAMATH_GPT_odd_factors_count_l1576_157649

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end NUMINAMATH_GPT_odd_factors_count_l1576_157649


namespace NUMINAMATH_GPT_quadratic_square_binomial_l1576_157659

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end NUMINAMATH_GPT_quadratic_square_binomial_l1576_157659


namespace NUMINAMATH_GPT_range_of_m_l1576_157673

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (6 - 3 * (x + 1) < x - 9) ∧ (x - m > -1) ↔ (x > 3)) → (m ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1576_157673


namespace NUMINAMATH_GPT_calculate_total_parts_l1576_157694

theorem calculate_total_parts (sample_size : ℕ) (draw_probability : ℚ) (N : ℕ) 
  (h_sample_size : sample_size = 30) 
  (h_draw_probability : draw_probability = 0.25) 
  (h_relation : sample_size = N * draw_probability) : 
  N = 120 :=
by
  rw [h_sample_size, h_draw_probability] at h_relation
  sorry

end NUMINAMATH_GPT_calculate_total_parts_l1576_157694


namespace NUMINAMATH_GPT_commute_days_l1576_157630

-- Definitions of the variables
variables (a b c x : ℕ)

-- Given conditions
def condition1 : Prop := a + c = 12
def condition2 : Prop := b + c = 20
def condition3 : Prop := a + b = 14

-- The theorem to prove
theorem commute_days (h1 : condition1 a c) (h2 : condition2 b c) (h3 : condition3 a b) : a + b + c = 23 :=
sorry

end NUMINAMATH_GPT_commute_days_l1576_157630


namespace NUMINAMATH_GPT_exists_y_less_than_half_p_l1576_157657

theorem exists_y_less_than_half_p (p : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) :
  ∃ (y : ℕ), y < p / 2 ∧ ∀ (a b : ℕ), p * y + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by sorry

end NUMINAMATH_GPT_exists_y_less_than_half_p_l1576_157657


namespace NUMINAMATH_GPT_distance_to_first_museum_l1576_157662

theorem distance_to_first_museum (x : ℝ) 
  (dist_second_museum : ℝ) 
  (total_distance : ℝ) 
  (h1 : dist_second_museum = 15) 
  (h2 : total_distance = 40) 
  (h3 : 2 * x + 2 * dist_second_museum = total_distance) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_distance_to_first_museum_l1576_157662


namespace NUMINAMATH_GPT_mul_point_five_point_three_l1576_157666

theorem mul_point_five_point_three : 0.5 * 0.3 = 0.15 := 
by  sorry

end NUMINAMATH_GPT_mul_point_five_point_three_l1576_157666


namespace NUMINAMATH_GPT_travel_remaining_distance_l1576_157632

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end NUMINAMATH_GPT_travel_remaining_distance_l1576_157632


namespace NUMINAMATH_GPT_slope_of_line_l1576_157670

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (4, 8)) :
  (y2 - y1) / (x2 - x1) = 2 := 
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l1576_157670


namespace NUMINAMATH_GPT_cistern_total_wet_surface_area_l1576_157668

-- Define the length, width, and depth of water in the cistern
def length : ℝ := 9
def width : ℝ := 4
def depth : ℝ := 1.25

-- Define the bottom surface area
def bottom_surface_area : ℝ := length * width

-- Define the longer side surface area
def longer_side_surface_area_each : ℝ := depth * length

-- Define the shorter end surface area
def shorter_end_surface_area_each : ℝ := depth * width

-- Calculate the total wet surface area
def total_wet_surface_area : ℝ := bottom_surface_area + 2 * longer_side_surface_area_each + 2 * shorter_end_surface_area_each

-- The theorem to be proved
theorem cistern_total_wet_surface_area :
  total_wet_surface_area = 68.5 :=
by
  -- since bottom_surface_area = 36,
  -- 2 * longer_side_surface_area_each = 22.5, and
  -- 2 * shorter_end_surface_area_each = 10
  -- the total will be equal to 68.5
  sorry

end NUMINAMATH_GPT_cistern_total_wet_surface_area_l1576_157668


namespace NUMINAMATH_GPT_find_c_l1576_157691

theorem find_c (c : ℝ) :
  (∃ (infinitely_many_y : ℝ → Prop), (∀ y, infinitely_many_y y ↔ 3 * (5 + 2 * c * y) = 18 * y + 15))
  → c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1576_157691


namespace NUMINAMATH_GPT_customer_payment_probability_l1576_157606

theorem customer_payment_probability :
  let total_customers := 100
  let age_40_50_non_mobile := 13
  let age_50_60_non_mobile := 27
  let total_40_60_non_mobile := age_40_50_non_mobile + age_50_60_non_mobile
  let probability := (total_40_60_non_mobile : ℚ) / total_customers
  probability = 2 / 5 := by
sorry

end NUMINAMATH_GPT_customer_payment_probability_l1576_157606


namespace NUMINAMATH_GPT_expand_polynomial_l1576_157663

theorem expand_polynomial (x : ℝ) : (5 * x + 3) * (6 * x ^ 2 + 2) = 30 * x ^ 3 + 18 * x ^ 2 + 10 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1576_157663
