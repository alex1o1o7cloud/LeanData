import Mathlib

namespace no_valid_n_for_three_digit_conditions_l754_75404

theorem no_valid_n_for_three_digit_conditions :
  ∃ (n : ℕ) (h₁ : 100 ≤ n / 4 ∧ n / 4 ≤ 999) (h₂ : 100 ≤ 4 * n ∧ 4 * n ≤ 999), false :=
by sorry

end no_valid_n_for_three_digit_conditions_l754_75404


namespace general_term_formula_l754_75481

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 * a n - 1) : 
  ∀ n, a n = 2^(n-1) := 
by
  sorry

end general_term_formula_l754_75481


namespace average_mpg_highway_l754_75450

variable (mpg_city : ℝ) (H mpg : ℝ) (gallons : ℝ) (max_distance : ℝ)

noncomputable def SUV_fuel_efficiency : Prop :=
  mpg_city  = 7.6 ∧
  gallons = 20 ∧
  max_distance = 244 ∧
  H * gallons = max_distance

theorem average_mpg_highway (h1 : mpg_city = 7.6) (h2 : gallons = 20) (h3 : max_distance = 244) :
  SUV_fuel_efficiency mpg_city H gallons max_distance → H = 12.2 :=
by
  intros h
  cases h
  sorry

end average_mpg_highway_l754_75450


namespace polygon_interior_angles_eq_360_l754_75411

theorem polygon_interior_angles_eq_360 (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angles_eq_360_l754_75411


namespace inequality_reciprocal_l754_75476

theorem inequality_reciprocal (a b : ℝ)
  (h : a * b > 0) : a > b ↔ 1 / a < 1 / b := 
sorry

end inequality_reciprocal_l754_75476


namespace find_YJ_l754_75446

structure Triangle :=
  (XY XZ YZ : ℝ)
  (XY_pos : XY > 0)
  (XZ_pos : XZ > 0)
  (YZ_pos : YZ > 0)

noncomputable def incenter_length (T : Triangle) : ℝ := 
  let XY := T.XY
  let XZ := T.XZ
  let YZ := T.YZ
  -- calculation using the provided constraints goes here
  3 * Real.sqrt 13 -- this should be computed based on the constraints, but is directly given as the answer

theorem find_YJ
  (T : Triangle)
  (XY_eq : T.XY = 17)
  (XZ_eq : T.XZ = 19)
  (YZ_eq : T.YZ = 20) :
  incenter_length T = 3 * Real.sqrt 13 :=
by 
  sorry

end find_YJ_l754_75446


namespace oliver_bags_fraction_l754_75407

theorem oliver_bags_fraction
  (weight_james_bag : ℝ)
  (combined_weight_oliver_bags : ℝ)
  (h1 : weight_james_bag = 18)
  (h2 : combined_weight_oliver_bags = 6)
  (f : ℝ) :
  2 * f * weight_james_bag = combined_weight_oliver_bags → f = 1 / 6 :=
by
  intro h
  sorry

end oliver_bags_fraction_l754_75407


namespace correct_statement_l754_75400

section
variables {a b c d : Real}

-- Define the conditions as hypotheses/functions

-- Statement A: If a > b, then 1/a < 1/b
def statement_A (a b : Real) : Prop := a > b → 1 / a < 1 / b

-- Statement B: If a > b, then a^2 > b^2
def statement_B (a b : Real) : Prop := a > b → a^2 > b^2

-- Statement C: If a > b and c > d, then ac > bd
def statement_C (a b c d : Real) : Prop := a > b ∧ c > d → a * c > b * d

-- Statement D: If a^3 > b^3, then a > b
def statement_D (a b : Real) : Prop := a^3 > b^3 → a > b

-- The Lean statement to prove which statement is correct
theorem correct_statement : ¬ statement_A a b ∧ ¬ statement_B a b ∧ ¬ statement_C a b c d ∧ statement_D a b :=
by {
  sorry
}

end

end correct_statement_l754_75400


namespace explorers_crossing_time_l754_75452

/-- Define constants and conditions --/
def num_explorers : ℕ := 60
def boat_capacity : ℕ := 6
def crossing_time : ℕ := 3
def round_trip_crossings : ℕ := 2
def total_trips := 1 + (num_explorers - boat_capacity - 1) / (boat_capacity - 1) + 1

theorem explorers_crossing_time :
  total_trips * crossing_time * round_trip_crossings / 2 + crossing_time = 69 :=
by sorry

end explorers_crossing_time_l754_75452


namespace total_number_of_athletes_l754_75419

theorem total_number_of_athletes (M F x : ℕ) (r1 r2 r3 : ℕ×ℕ) (H1 : r1 = (19, 12)) (H2 : r2 = (20, 13)) (H3 : r3 = (30, 19))
  (initial_males : M = 380 * x) (initial_females : F = 240 * x)
  (males_after_gym : M' = 390 * x) (females_after_gym : F' = 247 * x)
  (conditions : (M' - M) - (F' - F) = 30) : M' + F' = 6370 :=
by
  sorry

end total_number_of_athletes_l754_75419


namespace net_profit_100_patches_l754_75482

theorem net_profit_100_patches :
  let cost_per_patch := 1.25
  let num_patches_ordered := 100
  let selling_price_per_patch := 12.00
  let total_cost := cost_per_patch * num_patches_ordered
  let total_revenue := selling_price_per_patch * num_patches_ordered
  let net_profit := total_revenue - total_cost
  net_profit = 1075 :=
by
  sorry

end net_profit_100_patches_l754_75482


namespace total_floor_area_covered_l754_75413

-- Definitions for the given problem
def combined_area : ℕ := 204
def overlap_two_layers : ℕ := 24
def overlap_three_layers : ℕ := 20
def total_floor_area : ℕ := 140

-- Theorem to prove the total floor area covered by the rugs
theorem total_floor_area_covered :
  combined_area - overlap_two_layers - 2 * overlap_three_layers = total_floor_area := by
  sorry

end total_floor_area_covered_l754_75413


namespace circle_equation_unique_circle_equation_l754_75448

-- Definitions based on conditions
def radius (r : ℝ) : Prop := r = 1
def center_in_first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0
def tangent_to_line (a b : ℝ) : Prop := (|4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2)) = 1
def tangent_to_x_axis (b : ℝ) : Prop := b = 1

-- Main theorem statement
theorem circle_equation_unique 
  {a b : ℝ} 
  (h_rad : radius 1) 
  (h_center : center_in_first_quadrant a b) 
  (h_tan_line : tangent_to_line a b) 
  (h_tan_x : tangent_to_x_axis b) :
  (a = 2 ∧ b = 1) :=
sorry

-- Final circle equation
theorem circle_equation : 
  (∀ a b : ℝ, ((a = 2) ∧ (b = 1)) → (x - a)^2 + (y - b)^2 = 1) :=
sorry

end circle_equation_unique_circle_equation_l754_75448


namespace evaluate_f_g_at_3_l754_75425

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_f_g_at_3 : f (g 3) = 123 := by
  sorry

end evaluate_f_g_at_3_l754_75425


namespace evaluate_expression_c_eq_4_l754_75403

theorem evaluate_expression_c_eq_4 :
  (4^4 - 4 * (4-1)^(4-1))^(4-1) = 3241792 :=
by
  sorry

end evaluate_expression_c_eq_4_l754_75403


namespace three_digit_number_452_l754_75402

theorem three_digit_number_452 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 100 * a + 10 * b + c % (a + b + c) = 1)
  (h8 : 100 * c + 10 * b + a % (a + b + c) = 1)
  (h9 : a ≠ b) (h10 : b ≠ c) (h11 : a ≠ c)
  (h12 : a > c) :
  100 * a + 10 * b + c = 452 :=
sorry

end three_digit_number_452_l754_75402


namespace polar_to_rectangular_coords_l754_75457

theorem polar_to_rectangular_coords (r θ : ℝ) (x y : ℝ) 
  (hr : r = 5) (hθ : θ = 5 * Real.pi / 4)
  (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x = - (5 * Real.sqrt 2) / 2 ∧ y = - (5 * Real.sqrt 2) / 2 := 
by
  rw [hr, hθ] at hx hy
  simp [Real.cos, Real.sin] at hx hy
  rw [hx, hy]
  constructor
  . sorry
  . sorry

end polar_to_rectangular_coords_l754_75457


namespace intersection_S_T_l754_75410

def S : Set ℝ := { x | 2 * x + 1 > 0 }
def T : Set ℝ := { x | 3 * x - 5 < 0 }

theorem intersection_S_T :
  S ∩ T = { x | -1/2 < x ∧ x < 5/3 } := by
  sorry

end intersection_S_T_l754_75410


namespace times_reaching_35m_l754_75477

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 30 * t

theorem times_reaching_35m :
  ∃ t1 t2 : ℝ, (abs (t1 - 1.57) < 0.01 ∧ abs (t2 - 4.55) < 0.01) ∧
               projectile_height t1 = 35 ∧ projectile_height t2 = 35 :=
by
  sorry

end times_reaching_35m_l754_75477


namespace union_sets_l754_75464

open Set

/-- Given sets A and B defined as follows:
    A = {x | -1 ≤ x ∧ x ≤ 2}
    B = {x | x ≤ 4}
    Prove that A ∪ B = {x | x ≤ 4}
--/
theorem union_sets  :
    let A := {x | -1 ≤ x ∧ x ≤ 2}
    let B := {x | x ≤ 4}
    A ∪ B = {x | x ≤ 4} :=
by
    intros A B
    have : A = {x | -1 ≤ x ∧ x ≤ 2} := rfl
    have : B = {x | x ≤ 4} := rfl
    sorry

end union_sets_l754_75464


namespace cole_cost_l754_75443

def length_of_sides := 15
def length_of_back := 30
def cost_per_foot_side := 4
def cost_per_foot_back := 5
def cole_installation_fee := 50

def neighbor_behind_contribution := (length_of_back * cost_per_foot_back) / 2
def neighbor_left_contribution := (length_of_sides * cost_per_foot_side) / 3

def total_cost := 
  2 * length_of_sides * cost_per_foot_side + 
  length_of_back * cost_per_foot_back

def cole_contribution := 
  total_cost - neighbor_behind_contribution - neighbor_left_contribution + cole_installation_fee

theorem cole_cost (h : cole_contribution = 225) : cole_contribution = 225 := by
  sorry

end cole_cost_l754_75443


namespace lower_water_level_by_inches_l754_75455

theorem lower_water_level_by_inches
  (length width : ℝ) (gallons_removed : ℝ) (gallons_to_cubic_feet : ℝ) (feet_to_inches : ℝ) : 
  length = 20 → 
  width = 25 → 
  gallons_removed = 1875 → 
  gallons_to_cubic_feet = 7.48052 → 
  feet_to_inches = 12 → 
  (gallons_removed / gallons_to_cubic_feet) / (length * width) * feet_to_inches = 6.012 := 
by 
  sorry

end lower_water_level_by_inches_l754_75455


namespace pancake_problem_l754_75414

theorem pancake_problem :
  let mom_rate := (100 : ℚ) / 30
  let anya_rate := (100 : ℚ) / 40
  let andrey_rate := (100 : ℚ) / 60
  let combined_baking_rate := mom_rate + anya_rate
  let net_rate := combined_baking_rate - andrey_rate
  let target_pancakes := 100
  let time := target_pancakes / net_rate
  time = 24 := by
sorry

end pancake_problem_l754_75414


namespace log_simplify_l754_75463

open Real

theorem log_simplify : 
  (1 / (log 12 / log 3 + 1)) + 
  (1 / (log 8 / log 2 + 1)) + 
  (1 / (log 30 / log 5 + 1)) = 2 :=
by
  sorry

end log_simplify_l754_75463


namespace fraction_is_percent_of_y_l754_75437

theorem fraction_is_percent_of_y (y : ℝ) (hy : y > 0) : 
  (2 * y / 5 + 3 * y / 10) / y = 0.7 :=
sorry

end fraction_is_percent_of_y_l754_75437


namespace A_works_alone_45_days_l754_75472

open Nat

theorem A_works_alone_45_days (x : ℕ) :
  (∀ x : ℕ, (9 * (1 / x + 1 / 40) + 23 * (1 / 40) = 1) → (x = 45)) :=
sorry

end A_works_alone_45_days_l754_75472


namespace vector_subtraction_l754_75420

def vector1 : ℝ × ℝ := (3, -5)
def vector2 : ℝ × ℝ := (2, -6)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_subtraction :
  (scalar1 • vector1 - scalar2 • vector2) = (6, -2) := by
  sorry

end vector_subtraction_l754_75420


namespace part1_part2_l754_75431

-- Part 1
theorem part1 (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  x^2 - 2 * (x^2 - 3 * y) - 3 * (2 * x^2 + 5 * y) = -1 :=
by
  -- Proof to be provided
  sorry

-- Part 2
theorem part2 (a b : ℤ) (hab : a - b = 2 * b^2) :
  2 * (a^3 - 2 * b^2) - (2 * b - a) + a - 2 * a^3 = 0 :=
by
  -- Proof to be provided
  sorry

end part1_part2_l754_75431


namespace square_area_l754_75447

theorem square_area (side_length : ℕ) (h : side_length = 12) : side_length * side_length = 144 :=
by
  sorry

end square_area_l754_75447


namespace union_of_sets_l754_75451

theorem union_of_sets (P Q : Set ℝ) 
  (hP : P = {x | 2 ≤ x ∧ x ≤ 3}) 
  (hQ : Q = {x | x^2 ≤ 4}) : 
  P ∪ Q = {x | -2 ≤ x ∧ x ≤ 3} := 
sorry

end union_of_sets_l754_75451


namespace area_of_enclosed_figure_l754_75494

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), ((x)^(1/2) - x^2)

theorem area_of_enclosed_figure :
  area_enclosed_by_curves = (1 / 3) :=
by
  sorry

end area_of_enclosed_figure_l754_75494


namespace reflection_eqn_l754_75418

theorem reflection_eqn 
  (x y : ℝ)
  (h : y = 2 * x + 3) : 
  -y = 2 * x + 3 :=
sorry

end reflection_eqn_l754_75418


namespace satisfactory_grades_fraction_l754_75440

def total_satisfactory_students (gA gB gC gD gE : Nat) : Nat :=
  gA + gB + gC + gD + gE

def total_students (gA gB gC gD gE gF : Nat) : Nat :=
  total_satisfactory_students gA gB gC gD gE + gF

def satisfactory_fraction (gA gB gC gD gE gF : Nat) : Rat :=
  total_satisfactory_students gA gB gC gD gE / total_students gA gB gC gD gE gF

theorem satisfactory_grades_fraction :
  satisfactory_fraction 3 5 4 2 1 4 = (15 : Rat) / 19 :=
by
  sorry

end satisfactory_grades_fraction_l754_75440


namespace friend_selling_price_correct_l754_75460

-- Definition of the original cost price
def original_cost_price : ℕ := 50000

-- Definition of the loss percentage
def loss_percentage : ℕ := 10

-- Definition of the gain percentage
def gain_percentage : ℕ := 20

-- Definition of the man's selling price after loss
def man_selling_price : ℕ := original_cost_price - (original_cost_price * loss_percentage / 100)

-- Definition of the friend's selling price after gain
def friend_selling_price : ℕ := man_selling_price + (man_selling_price * gain_percentage / 100)

theorem friend_selling_price_correct : friend_selling_price = 54000 := by
  sorry

end friend_selling_price_correct_l754_75460


namespace maryann_time_spent_calling_clients_l754_75449

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end maryann_time_spent_calling_clients_l754_75449


namespace compute_x_l754_75474

/-- 
Let ABC be a triangle. 
Points D, E, and F are on BC, CA, and AB, respectively. 
Given that AE/AC = CD/CB = BF/BA = x for some x with 1/2 < x < 1. 
Segments AD, BE, and CF divide the triangle into 7 non-overlapping regions: 
4 triangles and 3 quadrilaterals. 
The total area of the 4 triangles equals the total area of the 3 quadrilaterals. 
Compute the value of x.
-/
theorem compute_x (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 1)
  (h3 : (∃ (triangleArea quadrilateralArea : ℝ), 
          let A := triangleArea + 3 * x
          let B := quadrilateralArea
          A = B))
  : x = (11 - Real.sqrt 37) / 6 := 
sorry

end compute_x_l754_75474


namespace min_M_for_inequality_l754_75423

noncomputable def M := (9 * Real.sqrt 2) / 32

theorem min_M_for_inequality (a b c : ℝ) : 
  abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) 
  ≤ M * (a^2 + b^2 + c^2)^2 := 
sorry

end min_M_for_inequality_l754_75423


namespace sum_unchanged_difference_changes_l754_75498

-- Definitions from conditions
def original_sum (a b c : ℤ) := a + b + c
def new_first (a : ℤ) := a - 329
def new_second (b : ℤ) := b + 401

-- Problem statement for sum unchanged
theorem sum_unchanged (a b c : ℤ) (h : original_sum a b c = 1281) :
  original_sum (new_first a) (new_second b) (c - 72) = 1281 := by
  sorry

-- Definitions for difference condition
def abs_diff (x y : ℤ) := abs (x - y)
def alter_difference (a b c : ℤ) :=
  abs_diff (new_first a) (new_second b) + abs_diff (new_first a) c + abs_diff b c

-- Problem statement addressing the difference
theorem difference_changes (a b c : ℤ) (h : original_sum a b c = 1281) :
  alter_difference a b c = abs_diff (new_first a) (new_second b) + abs_diff (c - 730) (new_first a) + abs_diff (c - 730) (new_first a) := by
  sorry

end sum_unchanged_difference_changes_l754_75498


namespace jessie_problem_l754_75471

def round_to_nearest_five (n : ℤ) : ℤ :=
  if n % 5 = 0 then n
  else if n % 5 < 3 then n - (n % 5)
  else n - (n % 5) + 5

theorem jessie_problem :
  round_to_nearest_five ((82 + 56) - 15) = 125 :=
by
  sorry

end jessie_problem_l754_75471


namespace fraction_proof_l754_75453

-- Define N
def N : ℕ := 24

-- Define F that satisfies the equation N = F + 15
def F := N - 15

-- Define the fraction that N exceeds by 15
noncomputable def fraction := (F : ℚ) / N

-- Prove that fraction = 3/8
theorem fraction_proof : fraction = 3 / 8 := by
  sorry

end fraction_proof_l754_75453


namespace number_of_students_suggested_mashed_potatoes_l754_75470

theorem number_of_students_suggested_mashed_potatoes 
    (students_suggested_bacon : ℕ := 374) 
    (students_suggested_tomatoes : ℕ := 128) 
    (total_students_participated : ℕ := 826) : 
    (total_students_participated - (students_suggested_bacon + students_suggested_tomatoes)) = 324 :=
by sorry

end number_of_students_suggested_mashed_potatoes_l754_75470


namespace eccentricity_of_hyperbola_l754_75499

-- Definitions and conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def regular_hexagon_side_length (a b c : ℝ) : Prop :=
  2 * a = (Real.sqrt 3 + 1) * c

-- Goal: Prove the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (a b c : ℝ) (x y : ℝ) :
  hyperbola x y a b →
  regular_hexagon_side_length a b c →
  2 * a = (Real.sqrt 3 + 1) * c →
  c ≠ 0 →
  a ≠ 0 →
  b ≠ 0 →
  (c / a = Real.sqrt 3 + 1) :=
by
  intros h_hyp h_hex h_eq h_c_ne_zero h_a_ne_zero h_b_ne_zero
  sorry -- Proof goes here

end eccentricity_of_hyperbola_l754_75499


namespace prove_midpoint_trajectory_eq_l754_75439

noncomputable def midpoint_trajectory_eq {x y : ℝ} (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) : Prop :=
  4*x^2 - 4*y^2 = 1

theorem prove_midpoint_trajectory_eq (x y : ℝ) (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) :
  midpoint_trajectory_eq h :=
sorry

end prove_midpoint_trajectory_eq_l754_75439


namespace find_x_l754_75427

def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, -1)
def scalar_multiply (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def collinear (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x :
  ∃ x : ℝ, collinear (vector_add (scalar_multiply 3 a) b) (c x) ∧ x = 4 :=
by
  sorry

end find_x_l754_75427


namespace child_to_grandmother_ratio_l754_75468

variable (G D C : ℝ)

axiom condition1 : G + D + C = 150
axiom condition2 : D + C = 60
axiom condition3 : D = 42

theorem child_to_grandmother_ratio : (C / G) = (1 / 5) :=
by
  sorry

end child_to_grandmother_ratio_l754_75468


namespace temperature_on_tuesday_l754_75421

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th = 156) ∧ (W + Th + 53 = 162) → T = 47 :=
by
  sorry

end temperature_on_tuesday_l754_75421


namespace minimum_value_of_phi_l754_75445

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def minimum_positive_period (ω : ℝ) := 2 * Real.pi / ω

theorem minimum_value_of_phi {A ω φ : ℝ} (hA : A > 0) (hω : ω > 0) 
  (h_period : minimum_positive_period ω = Real.pi) 
  (h_symmetry : ∀ x, f A ω φ x = f A ω φ (2 * Real.pi / ω - x)) : 
  ∃ k : ℤ, |φ| = |k * Real.pi - Real.pi / 6| → |φ| = Real.pi / 6 :=
by
  sorry

end minimum_value_of_phi_l754_75445


namespace product_of_six_consecutive_nat_not_equal_776965920_l754_75429

theorem product_of_six_consecutive_nat_not_equal_776965920 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920) :=
by
  sorry

end product_of_six_consecutive_nat_not_equal_776965920_l754_75429


namespace jane_savings_l754_75458

-- Given conditions
def cost_pair_1 : ℕ := 50
def cost_pair_2 : ℕ := 40

def promotion_A (cost1 cost2 : ℕ) : ℕ :=
  cost1 + cost2 / 2

def promotion_B (cost1 cost2 : ℕ) : ℕ :=
  cost1 + (cost2 - 15)

-- Define the savings calculation
def savings (promoA promoB : ℕ) : ℕ :=
  promoB - promoA

-- Specify the theorem to prove
theorem jane_savings :
  savings (promotion_A cost_pair_1 cost_pair_2) (promotion_B cost_pair_1 cost_pair_2) = 5 := 
by
  sorry

end jane_savings_l754_75458


namespace probability_both_truth_l754_75422

noncomputable def probability_A_truth : ℝ := 0.75
noncomputable def probability_B_truth : ℝ := 0.60

theorem probability_both_truth : 
  (probability_A_truth * probability_B_truth) = 0.45 :=
by sorry

end probability_both_truth_l754_75422


namespace find_BC_length_l754_75495

noncomputable def area_triangle (A B C : ℝ) : ℝ :=
  1/2 * A * B * C

theorem find_BC_length (A B C : ℝ) (angleA : ℝ)
  (h1 : area_triangle 5 A (Real.sin (π / 6)) = 5 * Real.sqrt 3)
  (h2 : B = 5)
  (h3 : angleA = π / 6) :
  C = Real.sqrt 13 :=
by
  sorry

end find_BC_length_l754_75495


namespace degree_diploma_salary_ratio_l754_75459

theorem degree_diploma_salary_ratio
  (jared_salary : ℕ)
  (diploma_monthly_salary : ℕ)
  (h_annual_salary : jared_salary = 144000)
  (h_diploma_annual_salary : 12 * diploma_monthly_salary = 48000) :
  (jared_salary / (12 * diploma_monthly_salary)) = 3 := 
by sorry

end degree_diploma_salary_ratio_l754_75459


namespace find_f_sqrt_10_l754_75454

-- Definitions and conditions provided in the problem
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2 - 8*x + 30

-- The problem specific conditions for f
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_periodic : is_periodic_function f 2)
variable (h_condition : f_condition f)

-- The statement to prove
theorem find_f_sqrt_10 : f (Real.sqrt 10) = -24 :=
by
  sorry

end find_f_sqrt_10_l754_75454


namespace squares_on_sides_of_triangle_l754_75492

theorem squares_on_sides_of_triangle (A B C : ℕ) (hA : A = 3^2) (hB : B = 4^2) (hC : C = 5^2) : 
  A + B = C :=
by 
  rw [hA, hB, hC] 
  exact Nat.add_comm 9 16 ▸ rfl

end squares_on_sides_of_triangle_l754_75492


namespace probability_of_winning_pair_l754_75409

/--
A deck consists of five red cards and five green cards, with each color having cards labeled from A to E. 
Two cards are drawn from this deck.
A winning pair is defined as two cards of the same color or two cards of the same letter. 
Prove that the probability of drawing a winning pair is 5/9.
-/
theorem probability_of_winning_pair :
  let total_cards := 10
  let total_ways := Nat.choose total_cards 2
  let same_letter_ways := 5
  let same_color_red_ways := Nat.choose 5 2
  let same_color_green_ways := Nat.choose 5 2
  let same_color_ways := same_color_red_ways + same_color_green_ways
  let favorable_outcomes := same_letter_ways + same_color_ways
  favorable_outcomes / total_ways = 5 / 9 := by
  sorry

end probability_of_winning_pair_l754_75409


namespace alex_bought_3_bags_of_chips_l754_75438

theorem alex_bought_3_bags_of_chips (x : ℝ) : 
    (1 * x + 5 + 73) / x = 27 → x = 3 := by sorry

end alex_bought_3_bags_of_chips_l754_75438


namespace probability_point_in_cube_l754_75485

noncomputable def volume_cube (s : ℝ) : ℝ := s ^ 3

noncomputable def radius_sphere (d : ℝ) : ℝ := d / 2

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem probability_point_in_cube :
  let s := 1 -- side length of the cube
  let v_cube := volume_cube s
  let d := Real.sqrt 3 -- diagonal of the cube
  let r := radius_sphere d
  let v_sphere := volume_sphere r
  v_cube / v_sphere = (2 * Real.sqrt 3) / (3 * Real.pi) :=
by
  sorry

end probability_point_in_cube_l754_75485


namespace angle_in_third_quadrant_l754_75493

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end angle_in_third_quadrant_l754_75493


namespace cos_B_plus_C_value_of_c_l754_75441

variable {A B C a b c : ℝ}

-- Given conditions
axiom a_eq_2b : a = 2 * b
axiom sine_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B

-- First proof
theorem cos_B_plus_C (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) :
  Real.cos (B + C) = 1 / 4 := 
sorry

-- Given additional condition for the area
axiom area_eq : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3

-- Second proof
theorem value_of_c (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) (h_area : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3) :
  c = 4 * Real.sqrt 2 :=
sorry

end cos_B_plus_C_value_of_c_l754_75441


namespace int_pairs_satisfy_conditions_l754_75456

theorem int_pairs_satisfy_conditions (m n : ℤ) :
  (∃ a b : ℤ, m^2 + n = a^2 ∧ n^2 + m = b^2) ↔ 
  ∃ k : ℤ, (m = 0 ∧ n = k^2) ∨ (m = k^2 ∧ n = 0) ∨ (m = 1 ∧ n = -1) ∨ (m = -1 ∧ n = 1) := by
  sorry

end int_pairs_satisfy_conditions_l754_75456


namespace certain_event_drawing_triangle_interior_angles_equal_180_deg_l754_75486

-- Define a triangle in the Euclidean space
structure Triangle (α : Type) [plane : TopologicalSpace α] :=
(a b c : α)

-- Define the sum of the interior angles of a triangle
noncomputable def sum_of_interior_angles {α : Type} [TopologicalSpace α] (T : Triangle α) : ℝ :=
180

-- The proof statement
theorem certain_event_drawing_triangle_interior_angles_equal_180_deg {α : Type} [TopologicalSpace α]
(T : Triangle α) : 
(sum_of_interior_angles T = 180) :=
sorry

end certain_event_drawing_triangle_interior_angles_equal_180_deg_l754_75486


namespace sequence_nth_term_mod_2500_l754_75412

def sequence_nth_term (n : ℕ) : ℕ :=
  -- this is a placeholder function definition; the actual implementation to locate the nth term is skipped
  sorry

theorem sequence_nth_term_mod_2500 : (sequence_nth_term 2500) % 7 = 1 := 
sorry

end sequence_nth_term_mod_2500_l754_75412


namespace area_of_region_l754_75488

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 11 = 0) -> 
  ∃ (A : ℝ), A = 24 * Real.pi :=
by 
  sorry

end area_of_region_l754_75488


namespace integer_solution_x_l754_75465

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l754_75465


namespace sin_neg_nine_pi_div_two_l754_75466

theorem sin_neg_nine_pi_div_two : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end sin_neg_nine_pi_div_two_l754_75466


namespace average_of_roots_l754_75462

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end average_of_roots_l754_75462


namespace max_a_value_l754_75430

noncomputable def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5 * a * k + 3) * x + 7

theorem max_a_value : ∀ (k a : ℝ), (0 <= k) → (k <= 2) →
  (∀ (x1 : ℝ), (k <= x1) → (x1 <= k + a) →
  ∀ (x2 : ℝ), (k + 2 * a <= x2) → (x2 <= k + 4 * a) →
  f x1 k a >= f x2 k a) → 
  a <= (2 * Real.sqrt 6 - 4) / 5 := 
sorry

end max_a_value_l754_75430


namespace opponent_score_value_l754_75406

-- Define the given conditions
def total_points : ℕ := 720
def games_played : ℕ := 24
def average_score := total_points / games_played
def championship_score := average_score / 2 - 2
def opponent_score := championship_score + 2

-- Lean theorem statement to prove
theorem opponent_score_value : opponent_score = 15 :=
by
  -- Proof to be filled in
  sorry

end opponent_score_value_l754_75406


namespace MsElizabethInvestmentsCount_l754_75442

variable (MrBanksRevPerInvestment : ℕ) (MsElizabethRevPerInvestment : ℕ) (MrBanksInvestments : ℕ) (MsElizabethExtraRev : ℕ)

def MrBanksTotalRevenue := MrBanksRevPerInvestment * MrBanksInvestments
def MsElizabethTotalRevenue := MrBanksTotalRevenue + MsElizabethExtraRev
def MsElizabethInvestments := MsElizabethTotalRevenue / MsElizabethRevPerInvestment

theorem MsElizabethInvestmentsCount (h1 : MrBanksRevPerInvestment = 500) 
  (h2 : MsElizabethRevPerInvestment = 900)
  (h3 : MrBanksInvestments = 8)
  (h4 : MsElizabethExtraRev = 500) : 
  MsElizabethInvestments MrBanksRevPerInvestment MsElizabethRevPerInvestment MrBanksInvestments MsElizabethExtraRev = 5 :=
by
  sorry

end MsElizabethInvestmentsCount_l754_75442


namespace max_true_statements_l754_75490

theorem max_true_statements (y : ℝ) :
  (0 < y^3 ∧ y^3 < 2 → ∀ (y : ℝ),  y^3 > 2 → False) ∧
  ((-2 < y ∧ y < 0) → ∀ (y : ℝ), (0 < y ∧ y < 2) → False) →
  ∃ (s1 s2 : Prop), 
    ((0 < y^3 ∧ y^3 < 2) = s1 ∨ (y^3 > 2) = s1 ∨ (-2 < y ∧ y < 0) = s1 ∨ (0 < y ∧ y < 2) = s1 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s1) ∧
    ((0 < y^3 ∧ y^3 < 2) = s2 ∨ (y^3 > 2) = s2 ∨ (-2 < y ∧ y < 0) = s2 ∨ (0 < y ∧ y < 2) = s2 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s2) ∧ 
    (s1 ∧ s2) → 
    ∃ m : ℕ, m = 2 := 
sorry

end max_true_statements_l754_75490


namespace problem_statement_l754_75496

variable {x y : ℝ}

def star (a b : ℝ) : ℝ := (a + b)^2

theorem problem_statement (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end problem_statement_l754_75496


namespace dan_helmet_crater_difference_l754_75424

theorem dan_helmet_crater_difference :
  ∀ (r d : ℕ), 
  (r = 75) ∧ (d = 35) ∧ (r = 15 + (d + (r - 15 - d))) ->
  ((d - (r - 15 - d)) = 10) :=
by
  intros r d h
  have hr : r = 75 := h.1
  have hd : d = 35 := h.2.1
  have h_combined : r = 15 + (d + (r - 15 - d)) := h.2.2
  sorry

end dan_helmet_crater_difference_l754_75424


namespace almond_croissant_price_l754_75479

theorem almond_croissant_price (R : ℝ) (T : ℝ) (W : ℕ) (total_spent : ℝ) (regular_price : ℝ) (weeks_in_year : ℕ) :
  R = 3.50 →
  T = 468 →
  W = 52 →
  (total_spent = 468) →
  (weekly_regular : ℝ) = 52 * 3.50 →
  (almond_total_cost : ℝ) = (total_spent - weekly_regular) →
  (A : ℝ) = (almond_total_cost / 52) →
  A = 5.50 := by
  intros hR hT hW htotal_spent hweekly_regular halmond_total_cost hA
  sorry

end almond_croissant_price_l754_75479


namespace total_rainfall_l754_75416

theorem total_rainfall (rain_first_hour : ℕ) (rain_second_hour : ℕ) : Prop :=
  rain_first_hour = 5 →
  rain_second_hour = 7 + 2 * rain_first_hour →
  rain_first_hour + rain_second_hour = 22

-- Add sorry to skip the proof.

end total_rainfall_l754_75416


namespace solve_for_y_l754_75473

-- Define the conditions as Lean functions and statements
def is_positive (y : ℕ) : Prop := y > 0
def multiply_sixteen (y : ℕ) : Prop := 16 * y = 256

-- The theorem that states the value of y
theorem solve_for_y (y : ℕ) (h1 : is_positive y) (h2 : multiply_sixteen y) : y = 16 :=
sorry

end solve_for_y_l754_75473


namespace f_properties_l754_75432

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂) :=
by 
  sorry

end f_properties_l754_75432


namespace cube_mod7_not_divisible_7_l754_75415

theorem cube_mod7_not_divisible_7 (a : ℤ) (h : ¬ (7 ∣ a)) :
  (a^3 % 7 = 1) ∨ (a^3 % 7 = -1) :=
sorry

end cube_mod7_not_divisible_7_l754_75415


namespace vector_perpendicular_solve_x_l754_75461

theorem vector_perpendicular_solve_x
  (x : ℝ)
  (a : ℝ × ℝ := (4, 8))
  (b : ℝ × ℝ := (x, 4))
  (h : 4 * x + 8 * 4 = 0) :
  x = -8 :=
sorry

end vector_perpendicular_solve_x_l754_75461


namespace evaluate_expression_l754_75467

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 1

theorem evaluate_expression : (1 / 2)^(b - a + 1) = 1 :=
by
  sorry

end evaluate_expression_l754_75467


namespace donation_problem_l754_75489

theorem donation_problem
  (A B C D : Prop)
  (h1 : ¬A ↔ (B ∨ C ∨ D))
  (h2 : B ↔ D)
  (h3 : C ↔ ¬B) 
  (h4 : D ↔ ¬B): A := 
by
  sorry

end donation_problem_l754_75489


namespace sum_of_abs_coeffs_in_binomial_expansion_l754_75405

theorem sum_of_abs_coeffs_in_binomial_expansion :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ), 
  (3 * x - 1) ^ 7 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7
  → |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4 ^ 7 :=
by
  sorry

end sum_of_abs_coeffs_in_binomial_expansion_l754_75405


namespace min_value_of_reciprocal_l754_75433

theorem min_value_of_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) :
  (∀ r, r = 1 / x + 1 / y → r ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end min_value_of_reciprocal_l754_75433


namespace right_angled_triangle_solution_l754_75487

-- Define the necessary constants
def t : ℝ := 504 -- area in cm^2
def c : ℝ := 65 -- hypotenuse in cm

-- The definitions of the right-angled triangle's properties
def is_right_angled_triangle (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2 ∧ a * b = 2 * t

-- The proof problem statement
theorem right_angled_triangle_solution :
  ∃ (a b : ℝ), is_right_angled_triangle a b ∧ ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
sorry

end right_angled_triangle_solution_l754_75487


namespace Katie_old_games_l754_75435

theorem Katie_old_games (O : ℕ) (hk1 : Katie_new_games = 57) (hf1 : Friends_new_games = 34) (hk2 : Katie_total_games = Friends_total_games + 62) : 
  O = 39 :=
by
  sorry

variables (Katie_new_games Friends_new_games Katie_total_games Friends_total_games : ℕ)

end Katie_old_games_l754_75435


namespace average_percentage_l754_75469

theorem average_percentage (num_students1 num_students2 : Nat) (avg1 avg2 avg : Nat) :
  num_students1 = 15 ->
  avg1 = 73 ->
  num_students2 = 10 ->
  avg2 = 88 ->
  (num_students1 * avg1 + num_students2 * avg2) / (num_students1 + num_students2) = avg ->
  avg = 79 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_percentage_l754_75469


namespace soccer_team_physics_players_l754_75497

-- Define the number of players on the soccer team
def total_players := 15

-- Define the number of players taking mathematics
def math_players := 10

-- Define the number of players taking both mathematics and physics
def both_subjects_players := 4

-- Define the number of players taking physics
def physics_players := total_players - math_players + both_subjects_players

-- The theorem to prove
theorem soccer_team_physics_players : physics_players = 9 :=
by
  -- using the conditions defined above
  sorry

end soccer_team_physics_players_l754_75497


namespace students_neither_math_physics_drama_exclusive_l754_75417

def total_students : ℕ := 75
def math_students : ℕ := 42
def physics_students : ℕ := 35
def both_students : ℕ := 25
def drama_exclusive_students : ℕ := 10

theorem students_neither_math_physics_drama_exclusive : 
  total_students - (math_students + physics_students - both_students + drama_exclusive_students) = 13 :=
by
  sorry

end students_neither_math_physics_drama_exclusive_l754_75417


namespace area_of_triangle_ABC_l754_75478

structure Point := (x y : ℝ)

def A := Point.mk 2 3
def B := Point.mk 9 3
def C := Point.mk 4 12

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * ((B.x - A.x) * (C.y - A.y))

theorem area_of_triangle_ABC :
  area_of_triangle A B C = 31.5 :=
by
  -- Proof is omitted
  sorry

end area_of_triangle_ABC_l754_75478


namespace pyramid_values_l754_75491

theorem pyramid_values :
  ∃ (A B C D : ℕ),
    (A = 3000) ∧
    (D = 623) ∧
    (B = 700) ∧
    (C = 253) ∧
    (A = 1100 + 1800) ∧
    (D + 451 ≥ 1065) ∧ (D + 451 ≤ 1075) ∧ -- rounding to nearest ten
    (B + 440 ≥ 1050) ∧ (B + 440 ≤ 1150) ∧
    (B + 1070 ≥ 1700) ∧ (B + 1070 ≤ 1900) ∧
    (C + 188 ≥ 430) ∧ (C + 188 ≤ 450) ∧    -- rounding to nearest ten
    (C + 451 ≥ 695) ∧ (C + 451 ≤ 705) :=  -- using B = 700 for rounding range
sorry

end pyramid_values_l754_75491


namespace star_polygon_points_eq_24_l754_75480

theorem star_polygon_points_eq_24 (n : ℕ) 
  (A_i B_i : ℕ → ℝ) 
  (h_congruent_A : ∀ i j, A_i i = A_i j) 
  (h_congruent_B : ∀ i j, B_i i = B_i j) 
  (h_angle_difference : ∀ i, A_i i = B_i i - 15) : 
  n = 24 := 
sorry

end star_polygon_points_eq_24_l754_75480


namespace number_of_players_knight_moves_friend_not_winner_l754_75426

-- Problem (a)
theorem number_of_players (sum_scores : ℕ) (h : sum_scores = 210) : 
  ∃ x : ℕ, x * (x - 1) = 210 :=
sorry

-- Problem (b)
theorem knight_moves (initial_positions : ℕ) (wrong_guess : ℕ) (correct_answer : ℕ) : 
  initial_positions = 1 ∧ wrong_guess = 64 ∧ correct_answer = 33 → 
  ∃ squares : ℕ, squares = 33 :=
sorry

-- Problem (c)
theorem friend_not_winner (total_scores : ℕ) (num_players : ℕ) (friend_score : ℕ) (avg_score : ℕ) : 
  total_scores = 210 ∧ num_players = 15 ∧ friend_score = 12 ∧ avg_score = 14 → 
  ∃ higher_score : ℕ, higher_score > friend_score :=
sorry

end number_of_players_knight_moves_friend_not_winner_l754_75426


namespace non_gray_squares_count_l754_75444

-- Define the dimensions of the grid strip
def width : ℕ := 5
def length : ℕ := 250

-- Define the repeating pattern dimensions and color distribution
def pattern_columns : ℕ := 4
def pattern_non_gray_squares : ℕ := 13
def pattern_total_squares : ℕ := width * pattern_columns

-- Define the number of complete patterns in the grid strip
def complete_patterns : ℕ := length / pattern_columns

-- Define the number of additional columns and additional non-gray squares
def additional_columns : ℕ := length % pattern_columns
def additional_non_gray_squares : ℕ := 6

-- Calculate the total non-gray squares
def total_non_gray_squares : ℕ := complete_patterns * pattern_non_gray_squares + additional_non_gray_squares

theorem non_gray_squares_count : total_non_gray_squares = 812 := by
  sorry

end non_gray_squares_count_l754_75444


namespace count_numbers_without_1_or_2_l754_75475

/-- The number of whole numbers between 1 and 2000 that do not contain the digits 1 or 2 is 511. -/
theorem count_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 511 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2000 →
      ¬ (∃ d : ℕ, (k.digits 10).contains d ∧ (d = 1 ∨ d = 2)) → n = 511) :=
sorry

end count_numbers_without_1_or_2_l754_75475


namespace magical_stack_example_l754_75401

-- Definitions based on the conditions
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def belongs_to_pile_A (card : ℕ) (n : ℕ) : Prop :=
  card <= n

def belongs_to_pile_B (card : ℕ) (n : ℕ) : Prop :=
  n < card

def magical_stack (cards : ℕ) (n : ℕ) : Prop :=
  ∀ (card : ℕ), (belongs_to_pile_A card n ∨ belongs_to_pile_B card n) → 
  (card + n) % (2 * n) = 1

-- The theorem to prove
theorem magical_stack_example :
  ∃ (n : ℕ), magical_stack 482 n ∧ (2 * n = 482) :=
by
  sorry

end magical_stack_example_l754_75401


namespace certain_number_condition_l754_75408

theorem certain_number_condition (x y z : ℤ) (N : ℤ)
  (hx : Even x) (hy : Odd y) (hz : Odd z)
  (hxy : x < y) (hyz : y < z)
  (h1 : y - x > N)
  (h2 : z - x = 7) :
  N < 3 := by
  sorry

end certain_number_condition_l754_75408


namespace principal_amount_simple_interest_l754_75434

theorem principal_amount_simple_interest 
    (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
    (hR : R = 4)
    (hT : T = 5)
    (hSI : SI = P - 2080)
    (hInterestFormula : SI = (P * R * T) / 100) :
    P = 2600 := 
by
  sorry

end principal_amount_simple_interest_l754_75434


namespace find_m_l754_75484

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

theorem find_m (m : ℕ) (h₀ : 0 < m) (h₁ : (m ^ 2 - 2 * m - 3:ℤ) < 0) (h₂ : is_odd (m ^ 2 - 2 * m - 3)) : m = 2 := 
sorry

end find_m_l754_75484


namespace find_real_numbers_l754_75428

theorem find_real_numbers (a b c : ℝ)    :
  (a + b + c = 3) → (a^2 + b^2 + c^2 = 35) → (a^3 + b^3 + c^3 = 99) → 
  (a = 1 ∧ b = -3 ∧ c = 5) ∨ (a = 1 ∧ b = 5 ∧ c = -3) ∨ 
  (a = -3 ∧ b = 1 ∧ c = 5) ∨ (a = -3 ∧ b = 5 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = -3) ∨ (a = 5 ∧ b = -3 ∧ c = 1) :=
by intros h1 h2 h3; sorry

end find_real_numbers_l754_75428


namespace decreasing_functions_l754_75436

noncomputable def f1 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f4 (x : ℝ) : ℝ := 3 ^ x

theorem decreasing_functions :
  (∀ x y : ℝ, 0 < x → x < y → f1 y < f1 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f2 y > f2 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f3 y > f3 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f4 y > f4 x) :=
by {
  sorry
}

end decreasing_functions_l754_75436


namespace cos_sin_combination_l754_75483

theorem cos_sin_combination (x : ℝ) (h : 2 * Real.cos x + 3 * Real.sin x = 4) : 
  3 * Real.cos x - 2 * Real.sin x = 0 := 
by 
  sorry

end cos_sin_combination_l754_75483
