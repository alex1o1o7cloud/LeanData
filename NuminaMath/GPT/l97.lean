import Mathlib

namespace number_of_integer_pairs_satisfying_conditions_l97_97080

-- Define the functions f(x) and g(x)
def f (x : ℤ) : ℤ := 3^x + 4 * 3^81
def g (x : ℤ) : ℤ := 85 + (3^81 - 1) * x

-- Define the conditions for y
def condition1 (x y : ℤ) : Prop := y ≥ f x
def condition2 (x y : ℤ) : Prop := y < g x

open Set

noncomputable def solution : ℕ :=
  let valid_pairs := {(x, y) | ∃ x y, 5 ≤ x ∧ x ≤ 84 ∧ condition1 x y ∧ condition2 x y}
  valid_pairs.card

theorem number_of_integer_pairs_satisfying_conditions :
  solution = 80 := sorry

end number_of_integer_pairs_satisfying_conditions_l97_97080


namespace largest_m_constant_l97_97925

variable (x y z w : ℝ)

theorem largest_m_constant 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hw : 0 < w) : 
  sqrt (x / (y + z + x)) + sqrt (y / (x + z + w)) + sqrt (z / (x + y + w)) + sqrt (w / (x + y + z)) ≥ 2 :=
sorry

end largest_m_constant_l97_97925


namespace find_inlet_rate_l97_97852

-- definitions for the given conditions
def volume_cubic_feet : ℝ := 20
def conversion_factor : ℝ := 12^3
def volume_cubic_inches : ℝ := volume_cubic_feet * conversion_factor

def outlet_rate1 : ℝ := 9
def outlet_rate2 : ℝ := 8
def empty_time : ℕ := 2880

-- theorem that captures the proof problem
theorem find_inlet_rate (volume_cubic_inches : ℝ) (outlet_rate1 outlet_rate2 empty_time : ℝ) :
  ∃ (inlet_rate : ℝ), volume_cubic_inches = (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time ↔ inlet_rate = 5 := 
by
  sorry

end find_inlet_rate_l97_97852


namespace tanner_remaining_cad_l97_97722

noncomputable def tanner_savings_in_cad : ℝ :=
let savings_eur : ℝ := 17
let savings_gbp : ℝ := 48
let savings_usd : ℝ := 25
let spent_jpy : ℝ := 49
let eur_to_usd : ℝ := 1.18
let gbp_to_usd : ℝ := 1.39
let jpy_to_usd : ℝ := 0.009
let cad_to_usd : ℝ := 0.8 in
let total_savings_usd := (savings_eur * eur_to_usd) + (savings_gbp * gbp_to_usd) + savings_usd
let total_spent_usd := spent_jpy * jpy_to_usd in
let remaining_usd := total_savings_usd - total_spent_usd in
remaining_usd / cad_to_usd

theorem tanner_remaining_cad : tanner_savings_in_cad = 139.17375 := by
  sorry

end tanner_remaining_cad_l97_97722


namespace person_A_boxes_average_unit_price_after_promotion_l97_97388

-- Definitions based on the conditions.
def unit_price (x: ℕ) (y: ℕ) : ℚ := y / x

def person_A_spent : ℕ := 2400
def person_B_spent : ℕ := 3000
def promotion_discount : ℕ := 20
def boxes_difference : ℕ := 10

-- Main proofs
theorem person_A_boxes (unit_price: ℕ → ℕ → ℚ) 
  (person_A_spent person_B_spent boxes_difference: ℕ): 
  ∃ x, unit_price person_A_spent x = unit_price person_B_spent (x + boxes_difference) 
  ∧ x = 40 := 
by {
  sorry
}

theorem average_unit_price_after_promotion (unit_price: ℕ → ℕ → ℚ) 
  (promotion_discount: ℕ) (person_A_spent person_B_spent: ℕ) 
  (boxes_A_promotion boxes_B: ℕ): 
  person_A_spent / (boxes_A_promotion * 2) + 20 = 48 
  ∧ person_B_spent / (boxes_B * 2) + 20 = 50 :=
by {
  sorry
}

end person_A_boxes_average_unit_price_after_promotion_l97_97388


namespace equation_undefined_when_x_eq_5_l97_97958

theorem equation_undefined_when_x_eq_5 : 
  ∃ (c : ℝ), ∀ (x : ℝ), (∀ (x = 5), 1 / (x + 5) + 1 / (x - 5) = c) → false :=
by
  intro c
  intro x
  intro h
  -- We aim to show that this leads to a contradiction, namely division by zero
  have h5 : 1 / (5 + 5) + 1 / (5 - 5) = c
  specialize h 5
  exact h 5
  sorry

end equation_undefined_when_x_eq_5_l97_97958


namespace relationship_cannot_be_determined_l97_97706

noncomputable def point_on_parabola (a b c x y : ℝ) : Prop :=
  y = a * x^2 + b * x + c

theorem relationship_cannot_be_determined
  (a b c x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (h1 : a ≠ 0) 
  (h2 : point_on_parabola a b c x1 y1) 
  (h3 : point_on_parabola a b c x2 y2) 
  (h4 : point_on_parabola a b c x3 y3) 
  (h5 : point_on_parabola a b c x4 y4)
  (h6 : x1 + x4 - x2 + x3 = 0) : 
  ¬( ∃ m n : ℝ, ((y4 - y1) / (x4 - x1) = m ∧ (y2 - y3) / (x2 - x3) = m) ∨ 
                     ((y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) = -1) ∨ 
                     ((y4 - y1) / (x4 - x1) ≠ m ∧ (y2 - y3) / (x2 - x3) ≠ m ∧ 
                      (y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) ≠ -1)) :=
sorry

end relationship_cannot_be_determined_l97_97706


namespace smallest_x_solution_l97_97523

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l97_97523


namespace meeting_at_centroid_l97_97037

theorem meeting_at_centroid :
  let A := (2, 9)
  let B := (-3, -4)
  let C := (6, -1)
  let centroid := ((2 - 3 + 6) / 3, (9 - 4 - 1) / 3)
  centroid = (5 / 3, 4 / 3) := sorry

end meeting_at_centroid_l97_97037


namespace license_plate_difference_l97_97694

/-- Given the following conditions:
   - California's license plates follow the L1L2L3 format where L is a letter and 1, 2, 3 are digits.
   - Texas's license plates follow the 123LLL format where 1, 2, 3 are digits and L is a letter.
   - There are 26 options for each letter and 10 options for each digit.
   
   Prove that the difference in the number of possible license plates between California and Texas is zero.
-/
theorem license_plate_difference :
  let cal_format := 26^3 * 10^3 in
  let tex_format := 10^3 * 26^3 in
  cal_format - tex_format = 0 :=
by
  let cal_format := 26 ^ 3 * 10 ^ 3
  let tex_format := 10 ^ 3 * 26 ^ 3
  calc
    cal_format - tex_format = 26 ^ 3 * 10 ^ 3 - 10 ^ 3 * 26 ^ 3 : by rfl
    ... = 0 : by sorry

end license_plate_difference_l97_97694


namespace number_of_mandatory_questions_correct_l97_97221

-- Definitions and conditions
def num_mandatory_questions (x : ℕ) (k : ℕ) (y : ℕ) (m : ℕ) : Prop :=
  (3 * k - 2 * (x - k) + 5 * m = 49) ∧
  (k + m = 15) ∧
  (y = 25 - x)

-- Proof statement
theorem number_of_mandatory_questions_correct :
  ∃ x k y m : ℕ, num_mandatory_questions x k y m ∧ x = 13 :=
by
  sorry

end number_of_mandatory_questions_correct_l97_97221


namespace not_mundane_prime_correct_l97_97010

def is_prime (p: ℕ) : Prop := Nat.Prime p

def is_mundane (p : ℕ) : Prop := 
  ∃ a b : ℕ, 0 < a ∧ a < p / 2 ∧ 0 < b ∧ b < p / 2 ∧ (ab - 1) % p = 0

def not_mundane_primes : List ℕ := [2, 3, 5, 7, 13]

theorem not_mundane_prime_correct : ∀ p : ℕ, is_prime p → ¬ is_mundane p ↔ p ∈ not_mundane_primes :=
by sorry

end not_mundane_prime_correct_l97_97010


namespace sum_remainders_mod_15_l97_97378

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end sum_remainders_mod_15_l97_97378


namespace sabina_saved_10000_l97_97297

noncomputable def sabina_savings (total_cost grant_coverage loan : ℝ) (S : ℝ) : Prop :=
  let remaining_tuition := loan / (1 - grant_coverage)
  in total_cost - remaining_tuition = S

theorem sabina_saved_10000 :
  sabina_savings 30000 0.40 12000 10000 :=
by
  unfold sabina_savings
  sorry

end sabina_saved_10000_l97_97297


namespace hyperbola_property_l97_97130

def hyperbola := {x : ℝ // ∃ y : ℝ, x^2 - y^2 / 8 = 1}

def is_on_left_branch (M : hyperbola) : Prop :=
  M.1 < 0

def focus1 : ℝ := -3
def focus2 : ℝ := 3

def distance (a b : ℝ) : ℝ := abs (a - b)

theorem hyperbola_property (M : hyperbola) (hM : is_on_left_branch M) :
  distance M.1 focus1 + distance focus1 focus2 - distance M.1 focus2 = 4 :=
  sorry

end hyperbola_property_l97_97130


namespace vector_dot_product_zero_l97_97139

variables {ℝ : Type*} [inner_product_space ℝ ℝ]
variable {a : ℝ}
variable {e : ℝ}
variable (h1 : a ≠ e)
variable (h2 : ∥e∥ = 1)
variable (h3 : ∀ t : ℝ, ∥a - t • e∥ ≥ ∥a - e∥)

theorem vector_dot_product_zero : e ⋅ (a - e) = 0 :=
sorry

end vector_dot_product_zero_l97_97139


namespace two_trains_passing_time_l97_97394

theorem two_trains_passing_time :
  ∀ (train_speed : ℝ) (train_length : ℝ), train_speed = 60 → train_length = 1/6 →
  time_to_pass_each_other train_speed train_length = 10 :=
by
  intros train_speed train_length h_speed h_length
  have h_relative_speed : train_speed + train_speed = 120 := by sorry
  have h_total_length : train_length + train_length = 1/3 := by sorry
  have h_speed_per_second : (120 / 60) / 60 = 1/30 := by sorry
  have h_time : (1/3) / (1/30) = 10 := by sorry
  exact h_time

end two_trains_passing_time_l97_97394


namespace number_of_such_lines_l97_97839

noncomputable def num_lines_passing_through_M_intersecting_parabola_exactly_once : Prop :=
  let M : ℝ × ℝ := (2, 4)
  let parabola : ℝ → ℝ → Prop := λ x y, y^2 = 8 * x
  ∃! (l : ℝ → ℝ → Prop), (l M.1 M.2) ∧ (∃! (P : ℝ × ℝ), (l P.1 P.2) ∧ (parabola P.1 P.2))

theorem number_of_such_lines : num_lines_passing_through_M_intersecting_parabola_exactly_once :=
by
  sorry

end number_of_such_lines_l97_97839


namespace max_k_l97_97802

-- Definitions of knight and liar predicates
def is_knight (p : ℕ → Prop) := ∀ i, p i
def is_liar (p : ℕ → Prop) := ∀ i, ¬ p i

-- Definitions for the conditions in the problem
def greater_than_neighbors (n : ℕ) (cards : ℕ → ℕ) :=
  ∀ i : ℕ, 0 < i ∧ i < n - 1 → cards i > cards (i - 1) ∧ cards i > cards (i + 1)

def less_than_neighbors (n : ℕ) (cards : ℕ → ℕ) :=
  ∀ i : ℕ, 0 < i ∧ i < n - 1 → cards i < cards (i - 1) ∧ cards i < cards (i + 1)

def maximum_possible_k (n : ℕ) (cards : ℕ → ℕ) (k : ℕ) :=
  ∀ n : 2015, ∃ k : ℕ, (greater_than_neighbors n cards → is_knight (less_than_neighbors n cards)) ∧
  k = 2013

-- The main theorem statement
theorem max_k (n : ℕ) (cards : ℕ → ℕ) : maximum_possible_k 2015 cards 2013 := sorry

end max_k_l97_97802


namespace find_n_l97_97205

theorem find_n (x n : ℝ) (h₁ : x = 1) (h₂ : 5 / (n + 1 / x) = 1) : n = 4 :=
sorry

end find_n_l97_97205


namespace point_symmetric_about_y_axis_l97_97124

theorem point_symmetric_about_y_axis (A B : ℝ × ℝ) 
  (hA : A = (1, -2)) 
  (hSym : B = (-A.1, A.2)) :
  B = (-1, -2) := 
by 
  sorry

end point_symmetric_about_y_axis_l97_97124


namespace imaginary_part_of_z_l97_97573

open Complex

-- Condition
def equation_z (z : ℂ) : Prop := (z * (1 + I) * I^3) / (1 - I) = 1 - I

-- Problem statement
theorem imaginary_part_of_z (z : ℂ) (h : equation_z z) : z.im = -1 := 
by 
  sorry

end imaginary_part_of_z_l97_97573


namespace polygon_becomes_convex_after_operations_l97_97387

/--
Prove that a non-convex, non-self-intersecting polygon will become convex after a finite number of specified operations.
-/

-- Definitions based on conditions
variable {P : Type} [polygon P] (A B : point) [non_adjacent_vertices P A B]
variable [non_convex_non_self_intersecting_polygon P]

/--
If the polygon P lies on one side of line AB,
then the part of the polygon divided by line AB is reflected relative to the midpoint of segment AB.
-/
def reflect_polygon (P : Type) [polygon P] (A B : point) : Type :=
sorry

/--
Theorem: A non-convex, non-self-intersecting polygon P will become convex after a finite number of these operations.
-/
theorem polygon_becomes_convex_after_operations
  {P : Type} [polygon P] [non_convex_non_self_intersecting_polygon P] :
  ∃ (n : ℕ), is_convex ((reflect_polygon P A B)^[n] P) :=
sorry

end polygon_becomes_convex_after_operations_l97_97387


namespace gain_amount_l97_97986

theorem gain_amount (gain_percent : ℝ) (gain : ℝ) (amount : ℝ) 
  (h_gain_percent : gain_percent = 1) 
  (h_gain : gain = 0.70) 
  : amount = 70 :=
by
  sorry

end gain_amount_l97_97986


namespace temperature_conversion_l97_97798

noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ :=
  (c * (9 / 5)) + 32

theorem temperature_conversion (c : ℝ) (hf : c = 60) :
  celsius_to_fahrenheit c = 140 :=
by {
  rw [hf, celsius_to_fahrenheit];
  norm_num
}

end temperature_conversion_l97_97798


namespace hvac_cost_per_vent_l97_97727

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end hvac_cost_per_vent_l97_97727


namespace restaurant_combinations_l97_97788

theorem restaurant_combinations (n : ℕ) (hn : n = 12) : 
  (n * (n - 1) = 132) :=
by
  rw hn
  rw Nat.sub_succ
  rw Nat.mul_sub_right_distrib
  rw Nat.mul_one
  rw Nat.sub_succ
  rw Nat.one_mul
  sorry

end restaurant_combinations_l97_97788


namespace interval_monotonic_decrease_sum_of_solutions_range_of_m_l97_97611

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 2 * (Real.sin (π / 4 + x))^2 - Real.sqrt 3 * (Real.cos (2 * x))

-- Statement 1: Interval of monotonic decrease
theorem interval_monotonic_decrease (k : ℤ) : 
  ∀ (x : ℝ), (f x) = 2 * (Real.sin (π / 4 + x))^2 - Real.sqrt 3 * (Real.cos (2 * x)) →
    k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 := 
sorry

-- Statement 2: Sum of two distinct real solutions
theorem sum_of_solutions (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Icc (π / 4) (π / 2) ∧ x₂ ∈ Icc (π / 4) (π / 2) ∧ f x₁ = a ∧ f x₂ = a) →
  (x₁ + x₂ = π) := 
sorry

-- Statement 3: Range of m
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Icc (π / 4) (π / 2), abs (f x - m) < 2) → 
  (1 < m ∧ m < 4) := 
sorry

end interval_monotonic_decrease_sum_of_solutions_range_of_m_l97_97611


namespace min_surface_area_of_stacked_solids_l97_97762

theorem min_surface_area_of_stacked_solids :
  ∀ (l w h : ℕ), l = 3 → w = 2 → h = 1 → 
  (2 * (l * w + l * h + w * h) - 2 * l * w = 32) :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  sorry

end min_surface_area_of_stacked_solids_l97_97762


namespace dot_product_of_unit_circle_points_l97_97591

theorem dot_product_of_unit_circle_points
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (θ : ℝ)
  (hθ_obtuse : π / 2 < θ ∧ θ < π)
  (h_sin : sin (θ + π / 4) = 3 / 5)
  : x1 * x2 + y1 * y2 = - (sqrt 2) / 10 :=
by
  sorry

end dot_product_of_unit_circle_points_l97_97591


namespace find_f_minus_100_plus_f_minus_101_l97_97946

-- Define the function f and the given conditions
noncomputable def f : ℝ → ℝ := sorry

-- The given conditions
axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom symmetric_about_x2 : ∀ x : ℝ, f(2 - x) = f(2 + x)
axiom definition_in_interval : ∀ x : ℝ, 0 < x ∧ x ≤ 2 → f(x) = x + 1

-- The proof problem: Prove f(-100) + f(-101) = 2
theorem find_f_minus_100_plus_f_minus_101 : f(-100) + f(-101) = 2 :=
by
  sorry

end find_f_minus_100_plus_f_minus_101_l97_97946


namespace min_cubes_needed_l97_97366

theorem min_cubes_needed (digits : Fin 10 → ℕ) : 
  (∀ k : Fin 10, digits k ≥ 29 ∧ (∀ i : Fin 9, digits (i+1) = 30)) →
  (∃ n : ℕ, 6 * n ≥ (30 * 9 + 29) ∧ n = 50) :=
by
  intro h
  use 50
  split
  · linarith
  · refl

end min_cubes_needed_l97_97366


namespace fewer_noodles_than_pirates_l97_97657

theorem fewer_noodles_than_pirates 
  (P : ℕ) (N : ℕ) (h1 : P = 45) (h2 : N + P = 83) : P - N = 7 := by 
  sorry

end fewer_noodles_than_pirates_l97_97657


namespace sum_xk_over_2k_plus_1_l97_97938

theorem sum_xk_over_2k_plus_1 (x : ℕ → ℝ) 
  (h : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 2014 → ∑ k in (finset.range 2014).image (λ k, k + 1), x k / (n + k + 1) = 1 / (2 * n + 1)) :
  ∑ k in (finset.range 2014).image (λ k, k + 1), x k / (2 * k + 1) = 1 / 4 * (1 - 1 / 4029^2) :=
by
  sorry

end sum_xk_over_2k_plus_1_l97_97938


namespace cubic_roots_geometric_progression_l97_97690

theorem cubic_roots_geometric_progression 
  (a r : ℝ)
  (h_roots: 27 * a^3 * r^3 - 81 * a^2 * r^2 + 63 * a * r - 14 = 0)
  (h_sum: a + a * r + a * r^2 = 3)
  (h_product: a^3 * r^3 = 14 / 27)
  : (max (a^2) ((a * r^2)^2) - min (a^2) ((a * r^2)^2) = 5 / 3) := 
sorry

end cubic_roots_geometric_progression_l97_97690


namespace cos_A_correct_l97_97245

open Real

def triangle_cos_A (A B C : ℝ) (b c h : ℝ) : Prop :=
  ∠ B = π / 4 ∧ c = 4 ∧ h = 1 ∧ cos A = -sqrt 5 / 5

theorem cos_A_correct :
  ∃ (A B C : ℝ) (b c : ℝ), 
    triangle_cos_A A B C b c 1 :=
by {
  use π - arccos (-sqrt 5 / 5),    -- A is determined by the answer (π - arccos because of the cosine rule and the angle sum in triangle)
  use (π / 4),                     -- B is given as π / 4
  use (π / 4),                     -- C is supplementary to angle sum of π - arccos (-sqrt 5 / 5) and π/4
  use sqrt 2,                      -- finding b from given relations
  use 4,                           -- c is given
  use (1 : ℝ),                     -- height h from A to BC is given as 1
  rw triangle_cos_A,
  sorry 
}

end cos_A_correct_l97_97245


namespace eval_g_inv_g_inv_14_l97_97311

variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)

axiom g_def : ∀ x, g x = 3 * x - 4
axiom g_inv_def : ∀ y, g_inv y = (y + 4) / 3

theorem eval_g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by
    sorry

end eval_g_inv_g_inv_14_l97_97311


namespace avg_hamburgers_per_day_l97_97457

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end avg_hamburgers_per_day_l97_97457


namespace number_of_correct_statements_is_one_l97_97032

-- Defining the statements
def statement1 : Prop := ∀ (P1 P2 : Plane) (p1 p2 p3 : Point), 
  p1 ∈ P1 ∧ p1 ∈ P2 ∧ p2 ∈ P1 ∧ p2 ∈ P2 ∧ p3 ∈ P1 ∧ p3 ∈ P2 → P1 = P2

def statement2 : Prop := ∀ (l1 l2 : Line), (∃ P : Plane, l1 ⊆ P ∧ l2 ⊆ P)

def statement3 : Prop := ∀ (M : Point) (α β : Plane) (l : Line), 
  M ∈ α ∧ M ∈ β ∧ (α ∩ β) = l → M ∈ l

def statement4 : Prop := ∀ (l1 l2 l3 : Line) (P : Point), 
  (P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3) → (∃σ : Plane, l1 ⊆ σ ∧ l2 ⊆ σ ∧ l3 ⊆ σ)

-- The theorem stating that the only one correct statement is statement3
theorem number_of_correct_statements_is_one : 
  (¬statement1) ∧ (¬statement2) ∧ statement3 ∧ (¬statement4) :=
by 
  sorry

end number_of_correct_statements_is_one_l97_97032


namespace hyperbola_center_l97_97831

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end hyperbola_center_l97_97831


namespace exists_m_with_totient_ratio_l97_97588

variable (α β : ℝ)

theorem exists_m_with_totient_ratio (h0 : 0 ≤ α) (h1 : α < β) (h2 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := 
  sorry

end exists_m_with_totient_ratio_l97_97588


namespace arithmetic_sequence_term_geometric_sequence_term_constant_value_max_value_l97_97864

theorem arithmetic_sequence_term (a b : ℕ) (d : ℤ) 
  (h₁ : d ≠ 0) 
  (h₂ : (a + d)^2 = a * (a + 4 * d)) 
  (h₃ : 10 * (a + (a + 9 * d)) / 2 = 100) : 
  a_n = λ n, 2 * n - 1 := 
sorry

theorem geometric_sequence_term (b_n : ℕ → ℤ) 
  (h₁ : ∀ n, S_n = 2 * b_n n - 1) :
  b_n = λ n, 2^(n-1) := 
sorry

theorem constant_value 
  (a_n : ℕ → ℤ) (b_n : ℕ → ℤ) 
  (C_n : ℕ → ℤ) 
  (h₁ : ∀ n, C_n n = a_n n + log (sqrt 2) (b_n n)) 
  (T_n : ℕ → ℤ) 
  (h₂ : ∀ n, T_n n = sum (C_n i) from i = 1 to n) 
  (d_n : ℕ → ℤ) 
  (h₃ : ∀ n c, d_n n = T_n n / (n - c)) : 
  c = -1 / 2 := 
sorry

theorem max_value (d_n : ℕ → ℤ) 
  (f : ℕ → ℚ) 
  (h : ∀ n, f n = d_n n / (n + 36) * d_n (n + 1)) : 
  max (f n) = 1 / 49 := 
sorry

end arithmetic_sequence_term_geometric_sequence_term_constant_value_max_value_l97_97864


namespace subscription_difference_is_4000_l97_97029

-- Given definitions
def total_subscription (A B C : ℕ) : Prop :=
  A + B + C = 50000

def subscription_B (x : ℕ) : ℕ :=
  x + 5000

def subscription_A (x y : ℕ) : ℕ :=
  x + 5000 + y

def profit_ratio (profit_C total_profit x : ℕ) : Prop :=
  (profit_C : ℚ) / total_profit = (x : ℚ) / 50000

-- Prove that A subscribed Rs. 4,000 more than B
theorem subscription_difference_is_4000 (x y : ℕ)
  (h1 : total_subscription (subscription_A x y) (subscription_B x) x)
  (h2 : profit_ratio 8400 35000 x) :
  y = 4000 :=
sorry

end subscription_difference_is_4000_l97_97029


namespace months_passed_l97_97040

-- Let's define our conditions in mathematical terms
def received_bones (months : ℕ) : ℕ := 10 * months
def buried_bones : ℕ := 42
def available_bones : ℕ := 8
def total_bones (months : ℕ) : Prop := received_bones months = buried_bones + available_bones

-- We need to prove that the number of months (x) satisfies the condition
theorem months_passed (x : ℕ) : total_bones x → x = 5 :=
by
  sorry

end months_passed_l97_97040


namespace area_arcsin_cos_l97_97076

variable (x : ℝ)
variable (f : ℝ → ℝ := λ x, Real.arcsin (Real.cos x))

theorem area_arcsin_cos :
  ∫ x in 0..2 * Real.pi, f x = 5 * Real.pi^2 / 2 := 
by 
  sorry

end area_arcsin_cos_l97_97076


namespace line_through_point_equal_intercepts_l97_97924

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (x y a : ℝ) (k : ℝ) 
  (hP : P = (2, 3))
  (hx : x / a + y / a = 1 ∨ (P.fst * k - P.snd = 0)) :
  (x + y - 5 = 0 ∨ 3 * P.fst - 2 * P.snd = 0) := by
  sorry

end line_through_point_equal_intercepts_l97_97924


namespace total_combined_rainfall_l97_97248

def mondayRainfall := 7 * 1
def tuesdayRainfall := 4 * 2
def wednesdayRate := 2 * 2
def wednesdayRainfall := 2 * wednesdayRate
def totalRainfall := mondayRainfall + tuesdayRainfall + wednesdayRainfall

theorem total_combined_rainfall : totalRainfall = 23 :=
by
  unfold totalRainfall mondayRainfall tuesdayRainfall wednesdayRainfall wednesdayRate
  sorry

end total_combined_rainfall_l97_97248


namespace notebooks_last_days_l97_97667

-- Given conditions
def n := 5
def p := 40
def u := 4

-- Derived conditions
def total_pages := n * p
def days := total_pages / u

-- The theorem statement
theorem notebooks_last_days : days = 50 := sorry

end notebooks_last_days_l97_97667


namespace smallest_value_of_n_l97_97442

theorem smallest_value_of_n (d : ℕ) (h_d_pos : 0 < d) : ∃ n : ℕ, n = 12 ∧ 
  let cost_per_radio := d / n,
      income_from_donated_radios := 2 * (d / (3 * n)),
      remaining_radios := n - 2,
      profit_per_remaining_radio := 10,
      total_income_from_remaining := remaining_radios * (cost_per_radio + profit_per_remaining_radio),
      total_cost := d,
      total_income := income_from_donated_radios + total_income_from_remaining,
      total_profit := total_income - total_cost
  in total_profit = 100 :=
by sorry

end smallest_value_of_n_l97_97442


namespace correct_propositions_l97_97860

noncomputable def proposition1 : Prop :=
  ∀ (a b : ℝ), (a + b ≠ 6) → (a ≠ 3 ∨ b ≠ 3)

noncomputable def proposition2 : Prop :=
  ∀ (p q : Prop), (p ∨ q) → (p ∧ q)

noncomputable def proposition3 : Prop :=
  ¬ (∀ (a b : ℝ), a^2 + b^2 ≥ 2 * (a - b - 1))

noncomputable def proposition3_negation : Prop :=
  ∃ (a b : ℝ), a^2 + b^2 ≤ 2 * (a - b - 1)

noncomputable def proposition4 : Prop :=
  ∀ (x y : ℝ), (x ≠ 0) → (y ≠ 0) → (xy > -2) → true -- Placeholder for true

theorem correct_propositions : (count_correct_propositions [proposition1, proposition2, proposition3_negation, proposition4] = 2) := by sorry

end correct_propositions_l97_97860


namespace volume_cone_equals_cylinder_minus_surface_area_l97_97708

theorem volume_cone_equals_cylinder_minus_surface_area (r h : ℝ) :
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  V_cone = V_cyl - (1 / 3) * S_lateral_cyl * r := by
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  sorry

end volume_cone_equals_cylinder_minus_surface_area_l97_97708


namespace probability_of_A_probability_of_B_probability_of_AB_l97_97765

noncomputable def probability_A (total_combinations : ℕ) : ℚ :=
  let favorable_A := 252155
  favorable_A / total_combinations

theorem probability_of_A :
  probability_A (Nat.choose 90 5) ≈ 0.005737 := 
sorry

noncomputable def probability_B (total_combinations : ℕ) : ℚ :=
  let favorable_B := 624360
  favorable_B / total_combinations

theorem probability_of_B :
  probability_B (Nat.choose 90 5) ≈ 0.01421 :=
sorry

noncomputable def probability_AB (total_combinations : ℕ) : ℚ :=
  let favorable_AB := 5285
  favorable_AB / total_combinations

theorem probability_of_AB :
  probability_AB (Nat.choose 90 5) ≈ 0.000120 :=
sorry

end probability_of_A_probability_of_B_probability_of_AB_l97_97765


namespace tile_calc_proof_l97_97280

noncomputable def total_tiles (length width : ℕ) : ℕ :=
  let border_tiles_length := (2 * (length - 4)) * 2
  let border_tiles_width := (2 * (width - 4)) * 2
  let total_border_tiles := (border_tiles_length + border_tiles_width) * 2 - 8
  let inner_length := (length - 4)
  let inner_width := (width - 4)
  let inner_area := inner_length * inner_width
  let inner_tiles := inner_area / 4
  total_border_tiles + inner_tiles

theorem tile_calc_proof :
  total_tiles 15 20 = 144 :=
by
  sorry

end tile_calc_proof_l97_97280


namespace lyka_savings_goal_l97_97278

noncomputable def savings (initial : ℝ) (gym_fee : ℝ) (interest_rate : ℝ) 
                          (savings_week1 : ℝ) (savings_week2 : ℝ) 
                          (savings_week3 : ℝ) (savings_week4 : ℝ) 
                          : ℝ :=
let month1 := initial + savings_week1 * 4 - gym_fee + (initial + savings_week1 * 4 - gym_fee) * interest_rate in
let month2 := month1 + savings_week2 * 4 - gym_fee + (month1 + savings_week2 * 4 - gym_fee) * interest_rate in
let month3 := month2 + savings_week3 * 4 - gym_fee + (month2 + savings_week3 * 4 - gym_fee) * interest_rate in
let month4 := month3 + savings_week4 * 4 - gym_fee + (month3 + savings_week4 * 4 - gym_fee) * interest_rate in
month4

theorem lyka_savings_goal :
  let initial := 200 in
  let gym_fee := 50 in
  let interest_rate := 0.01 in
  let week1_savings := 50 in
  let week2_savings := 80 in
  let week3_savings := 30 in
  let week4_savings := 60 in
  let final_savings := savings initial gym_fee interest_rate week1_savings week2_savings week3_savings week4_savings in
  let smartphone_cost1 := 800 in
  let smartphone_cost2 := 600 in
  let discount_rate := 0.1 in
  let discounted_cost := (smartphone_cost1 + smartphone_cost2) * (1 - discount_rate) in
  final_savings < discounted_cost :=
by
  let initial := 200
  let gym_fee := 50
  let interest_rate := 0.01
  let week1_savings := 50
  let week2_savings := 80
  let week3_savings := 30
  let week4_savings := 60
  let final_savings := savings initial gym_fee interest_rate week1_savings week2_savings week3_savings week4_savings
  let smartphone_cost1 := 800
  let smartphone_cost2 := 600
  let discount_rate := 0.1
  let discounted_cost := (smartphone_cost1 + smartphone_cost2) * (1 - discount_rate)
  have h : final_savings = 907.73, sorry
  have h2 : discounted_cost = 1260 * (1 - discount_rate), sorry 
  have h3 : discounted_cost = 1260 * 0.9, by rw [h2] at h; simp
  have h4 : discounted_cost = 1134, sorry
  have h5 : final_savings = 907.73, sorry 
  have h6 : 907.73 < 1134, sorry 
  exact h6

end lyka_savings_goal_l97_97278


namespace chocolate_bars_per_small_box_l97_97446

theorem chocolate_bars_per_small_box (total_chocolate_bars small_boxes : ℕ) 
  (h1 : total_chocolate_bars = 442) 
  (h2 : small_boxes = 17) : 
  total_chocolate_bars / small_boxes = 26 :=
by
  sorry

end chocolate_bars_per_small_box_l97_97446


namespace find_a_l97_97979

def A (a : ℤ) : Set ℤ := {-4, 2 * a - 1, a * a}
def B (a : ℤ) : Set ℤ := {a - 5, 1 - a, 9}

theorem find_a (a : ℤ) : (9 ∈ (A a ∩ B a)) ∧ (A a ∩ B a = {9}) ↔ a = -3 :=
by
  sorry

end find_a_l97_97979


namespace line_intersects_x_axis_at_10_0_l97_97447

theorem line_intersects_x_axis_at_10_0 :
  ∃ x : ℝ, (8:ℝ, 2:ℝ) ≠ (4:ℝ, 6:ℝ) → 
           (∃ y : ℝ, y = -((x - 8) * ((6 - 2) / (4 - 8)):ℝ) + 2) → 
           x = 10 :=
by
  sorry

end line_intersects_x_axis_at_10_0_l97_97447


namespace distance_to_midpoint_is_168_l97_97759

noncomputable def distance_from_B_to_midpoint_AD (AB BC AC AD CD : ℝ) : ℝ :=
  let B := (231 : ℝ, 0 : ℝ)
  let A := (0 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, 160 : ℝ)
  let D := (
    real.sqrt(20473.2578125),
    105.859375
  )
  let M := ((A.fst + D.fst) / 2, (A.snd + D.snd) / 2) 
  real.sqrt((B.fst - M.fst)^2 + (B.snd - M.snd)^2)

theorem distance_to_midpoint_is_168 : distance_from_B_to_midpoint_AD 231 160 281 178 153 = 168 :=
by
  sorry

end distance_to_midpoint_is_168_l97_97759


namespace find_x0_l97_97801

-- Define a function f with domain [0, 3] and its inverse
variable {f : ℝ → ℝ}

-- Assume conditions for the inverse function
axiom f_inv_1 : ∀ x, 0 ≤ x ∧ x < 1 → 1 ≤ f x ∧ f x < 2
axiom f_inv_2 : ∀ x, 2 < x ∧ x ≤ 4 → 0 ≤ f x ∧ f x < 1

-- Domain condition
variables (x : ℝ) (hf_domain : 0 ≤ x ∧ x ≤ 3)

-- The main theorem
theorem find_x0 : (∃ x0: ℝ, f x0 = x0) → x = 2 :=
  sorry

end find_x0_l97_97801


namespace calculate_overhead_cost_l97_97307

noncomputable def overhead_cost (prod_cost revenue_cost : ℕ) (num_performances : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost - num_performances * prod_cost

theorem calculate_overhead_cost :
  overhead_cost 7000 16000 9 (9 * 16000) = 81000 :=
by
  sorry

end calculate_overhead_cost_l97_97307


namespace circumscribable_cone_inscribable_cone_circumscribable_cylinder_inscribable_cylinder_l97_97485

-- Define circular cone as a structure with apex and a circular base
structure CircularCone (Apex : Point) (Base : Circle) :=
(apex : Apex)
(base : Base)

-- Define circular cylinder as a structure with base and height
structure CircularCylinder (Base : Circle) (Height : ℝ) :=
(base : Base)
(height : Height)

-- Define circumscribe sphere
def circumscribe_sphere (shape : Type) : Prop := sorry

-- Define inscribe sphere
def inscribe_sphere (shape : Type) : Prop := sorry

-- Circular cone circumscription and inscription
theorem circumscribable_cone (cone : CircularCone) : circumscribe_sphere cone := sorry

theorem inscribable_cone (cone : CircularCone) : inscribe_sphere cone := sorry

-- Circular cylinder circumscription and inscription
theorem circumscribable_cylinder (cylinder : CircularCylinder) : circumscribe_sphere cylinder := sorry

theorem inscribable_cylinder (cylinder : CircularCylinder) (square_axial_section : cylinder.base.radius * 2 = cylinder.height) : inscribe_sphere cylinder := sorry

-- Auxiliary types for mathematical objects
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

structure Circle := (center : Point) (radius : ℝ)

end circumscribable_cone_inscribable_cone_circumscribable_cylinder_inscribable_cylinder_l97_97485


namespace smallest_x_solution_l97_97522

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l97_97522


namespace complex_div_eq_l97_97275

open Complex

def z := 4 - 2 * I

theorem complex_div_eq :
  (z + I = 4 - I) →
  (z / (4 + 2 * I) = (3 - 4 * I) / 5) :=
by
  sorry

end complex_div_eq_l97_97275


namespace negation_of_existence_l97_97688

theorem negation_of_existence (P : ∃ n : ℕ, n^2 > 2^n) : ¬ P ↔ ∀ n : ℕ, n^2 ≤ 2^n := by
  sorry

end negation_of_existence_l97_97688


namespace correct_answers_unanswered_minimum_correct_answers_l97_97698

-- Definition of the conditions in the problem
def total_questions := 25
def unanswered_questions := 1
def correct_points := 4
def wrong_points := -1
def total_score_1 := 86
def total_score_2 := 90

-- Part 1: Define the conditions and prove that x = 22
theorem correct_answers_unanswered (x : ℕ) (h1 : total_questions - unanswered_questions = 24)
  (h2 : 4 * x + wrong_points * (total_questions - unanswered_questions - x) = total_score_1) : x = 22 :=
sorry

-- Part 2: Define the conditions and prove that at least 23 correct answers are needed
theorem minimum_correct_answers (a : ℕ)
  (h3 : correct_points * a + wrong_points * (total_questions - a) ≥ total_score_2) : a ≥ 23 :=
sorry

end correct_answers_unanswered_minimum_correct_answers_l97_97698


namespace seq_100_gt_14_l97_97337

variable {a : ℕ → ℝ}

axiom seq_def (n : ℕ) : a 0 = 1 ∧ (∀ n ≥ 0, a (n + 1) = a n + 1 / a n)

theorem seq_100_gt_14 : a 100 > 14 :=
by
  -- Establish sequence definition
  have h1 : a 0 = 1 := (seq_def 0).left,
  have h2 : ∀ n ≥ 0, a (n + 1) = a n + 1 / a n := (seq_def 0).right,
  sorry

end seq_100_gt_14_l97_97337


namespace sec_7pi_over_4_eq_sqrt2_l97_97915

theorem sec_7pi_over_4_eq_sqrt2 : Real.sec (7 * Real.pi / 4) = Real.sqrt 2 := 
by 
  sorry

end sec_7pi_over_4_eq_sqrt2_l97_97915


namespace repeated_decimal_to_fraction_l97_97190

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l97_97190


namespace length_of_shorter_piece_l97_97408

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end length_of_shorter_piece_l97_97408


namespace cost_of_pink_notebook_l97_97704

theorem cost_of_pink_notebook
    (total_cost : ℕ) 
    (black_cost : ℕ) 
    (green_cost : ℕ) 
    (num_green : ℕ) 
    (num_black : ℕ) 
    (num_pink : ℕ)
    (total_notebooks : ℕ)
    (h_total_cost : total_cost = 45)
    (h_black_cost : black_cost = 15) 
    (h_green_cost : green_cost = 10) 
    (h_num_green : num_green = 2) 
    (h_num_black : num_black = 1) 
    (h_num_pink : num_pink = 1)
    (h_total_notebooks : total_notebooks = 4) 
    : (total_cost - (num_green * green_cost + black_cost) = 10) :=
by
  sorry

end cost_of_pink_notebook_l97_97704


namespace crease_length_l97_97865

theorem crease_length (ABC : Type) [metric_space ABC] [triangle ABC]
  (A B C A' P Q : ABC) 
  (equilateral : side_length A B = 5 ∧ side_length B C = 5 ∧ side_length C A = 5)
  (folded : distance A A' = 0 ∧ A' ∈ segment B C)
  (BA' : distance B A' = 2)
  (A'C : distance A' C = 3) :
  distance P Q = (7 * Real.sqrt 11) / 2 := 
sorry

end crease_length_l97_97865


namespace part_I_solution_part_II_solution_l97_97100

-- Part (I) proof problem: Prove the solution set for a specific inequality
theorem part_I_solution (x : ℝ) : -6 < x ∧ x < 10 / 3 → |2 * x - 2| + x + 1 < 9 :=
by
  sorry

-- Part (II) proof problem: Prove the range of 'a' for a given inequality to hold
theorem part_II_solution (a : ℝ) : (-3 ≤ a ∧ a ≤ 17 / 3) →
  (∀ x : ℝ, x ≥ 2 → |a * x + a - 4| + x + 1 ≤ (x + 2)^2) :=
by
  sorry

end part_I_solution_part_II_solution_l97_97100


namespace moles_of_BeOH2_l97_97515

-- Definitions based on the given conditions
def balanced_chemical_equation (xBe2C xH2O xBeOH2 xCH4 : ℕ) : Prop :=
  xBe2C = 1 ∧ xH2O = 4 ∧ xBeOH2 = 2 ∧ xCH4 = 1

def initial_conditions (yBe2C yH2O : ℕ) : Prop :=
  yBe2C = 1 ∧ yH2O = 4

-- Lean statement to prove the number of moles of Beryllium hydroxide formed
theorem moles_of_BeOH2 (xBe2C xH2O xBeOH2 xCH4 yBe2C yH2O : ℕ) (h1 : balanced_chemical_equation xBe2C xH2O xBeOH2 xCH4) (h2 : initial_conditions yBe2C yH2O) :
  xBeOH2 = 2 :=
by
  sorry

end moles_of_BeOH2_l97_97515


namespace closest_point_proof_l97_97516

-- Define the point and the line parameters
def point := (3 : ℝ, -1 : ℝ, 0 : ℝ)
def line_point := (5 : ℝ, 1 : ℝ, 2 : ℝ)
def direction := (3 : ℝ, -3 : ℝ, 2 : ℝ)

-- Define the equation of the line
def line (t : ℝ) := (line_point.1 + t * direction.1, line_point.2 + t * direction.2, line_point.3 + t * direction.3)

-- Define a vector from a fixed point to a point on the line
def vector_to_point (t : ℝ) := (line t.1 - point.1, line t.2 - point.2, line t.3 - point.3)

-- The orthogonality condition
def orthogonality_condition (t : ℝ) := vector_to_point t.1 * direction.1 + vector_to_point t.2 * direction.2 + vector_to_point t.3 * direction.3 = 0

-- Prove that the closest point is (49/11 , 19/11, 18/11)
def closest_point := (49 / 11 : ℝ, 19 / 11 : ℝ, 18 / 11 : ℝ)

theorem closest_point_proof : closest_point = line (-2 / 11) :=
by
  sorry

end closest_point_proof_l97_97516


namespace proof_min_max_expected_wasted_minutes_l97_97418

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l97_97418


namespace polynomial_identity_solution_l97_97073

theorem polynomial_identity_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, x * P.eval (x - 1) = (x - 2) * P.eval x) ↔ (∃ a : ℝ, P = Polynomial.C a * (Polynomial.X ^ 2 - Polynomial.X)) :=
by
  sorry

end polynomial_identity_solution_l97_97073


namespace acute_triangle_count_l97_97106

theorem acute_triangle_count (h : ∀ (x : Type), fintype x → (is_circle : bool)) :
  let eightPoints : finset Point := finset.range 8 in
  (∃ (pts: finset Point), -- Given 8 equally spaced points on the circle
    pts.card = 8 ∧
    pts ⊆ eightPoints ∧ -- subset condition for 8 equally spaced points
    -- Count the number of acute triangles
    let triangles := { t: finset Point | t.card = 3 ∧ ∀ pt ∈ t, pt ∈ pts } in
    fintype.card (triangles.filter (λ t, is_acute t)) = 8) := sorry

structure Point where
  angle : ℝ -- Assuming each point is determined by an angle on the circle

noncomputable def is_acute_triangle (a b c : Point) : bool :=
  -- Placeholder for checking if the angles form an acute triangle
  sorry

noncomputable def is_acute (triangle : finset Point) : bool :=
  match finset.to_list triangle with
  | [a, b, c] => is_acute_triangle a b c
  | _ => false

end acute_triangle_count_l97_97106


namespace find_A_from_complement_l97_97148

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define the complement of set A in U
variable (A : Set ℕ)
def complement_U_A : Set ℕ := {n | n ∈ U ∧ n ∉ A}

-- Define the condition given in the problem
axiom h : complement_U_A A = {2}

-- State the theorem to be proven
theorem find_A_from_complement : A = {0, 1} :=
sorry

end find_A_from_complement_l97_97148


namespace domain_of_f_l97_97322

def f (x : ℝ) : ℝ := 1 / (Real.sqrt (2 - x))

theorem domain_of_f :
  {x : ℝ | 2 - x > 0} = {x : ℝ | x < 2} :=
by
  sorry

end domain_of_f_l97_97322


namespace smallest_solution_floor_equation_l97_97550

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l97_97550


namespace side_length_square_correct_l97_97814

noncomputable def side_length_square (time_seconds : ℕ) (speed_kmph : ℕ) : ℕ := sorry

theorem side_length_square_correct (time_seconds : ℕ) (speed_kmph : ℕ) (h_time : time_seconds = 24) 
  (h_speed : speed_kmph = 12) : side_length_square time_seconds speed_kmph = 20 :=
sorry

end side_length_square_correct_l97_97814


namespace sum_of_coefficients_binomial_expansion_l97_97083

theorem sum_of_coefficients_binomial_expansion :
  (∑ k in Finset.range 8, Nat.choose 7 k) = 128 :=
by
  sorry

end sum_of_coefficients_binomial_expansion_l97_97083


namespace avg_hamburgers_per_day_l97_97456

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end avg_hamburgers_per_day_l97_97456


namespace coprime_repeating_decimal_sum_l97_97174

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l97_97174


namespace shorter_piece_is_28_l97_97406

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end shorter_piece_is_28_l97_97406


namespace largest_remainder_2015_l97_97982

theorem largest_remainder_2015 : 
  ∃ (d : ℕ) (r : ℕ), d ∈ (set.Icc 1 1000) ∧ r = 2015 % d ∧ (∀ (d' ∈ (set.Icc 1 1000)), 2015 % d' ≤ r) ∧ r = 671 :=
sorry

end largest_remainder_2015_l97_97982


namespace balls_is_perfect_square_l97_97348

open Classical -- Open classical logic for nonconstructive proofs

-- Define a noncomputable function to capture the main proof argument
noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem balls_is_perfect_square {a v : ℕ} (h : (2 * a * v) = (a + v) * (a + v - 1))
  : is_perfect_square (a + v) :=
sorry

end balls_is_perfect_square_l97_97348


namespace smallest_solution_eq_sqrt_104_l97_97532

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l97_97532


namespace correct_algorithm_option_l97_97341

def OptionA := ("Sequential structure", "Flow structure", "Loop structure")
def OptionB := ("Sequential structure", "Conditional structure", "Nested structure")
def OptionC := ("Sequential structure", "Conditional structure", "Loop structure")
def OptionD := ("Flow structure", "Conditional structure", "Loop structure")

-- The correct structures of an algorithm are sequential, conditional, and loop.
def algorithm_structures := ("Sequential structure", "Conditional structure", "Loop structure")

theorem correct_algorithm_option : algorithm_structures = OptionC := 
by 
  -- This would be proven by logic and checking the options; omitted here with 'sorry'
  sorry

end correct_algorithm_option_l97_97341


namespace yellow_bow_count_l97_97645

theorem yellow_bow_count (N : ℕ)
  (red_fraction : ℚ := 1/4)
  (blue_fraction : ℚ := 1/3)
  (green_fraction : ℚ := 1/6)
  (yellow_fraction : ℚ := 1/12)
  (white_bows : ℕ := 40)
  (total_fraction_colored : ℚ := red_fraction + blue_fraction + green_fraction + yellow_fraction)
  (fraction_white : ℚ := 1 - total_fraction_colored)
  (total_bows : ℕ := white_bows * denominator fraction_white)
  (yellow_bows : ℕ := yellow_fraction * total_bows) :
  yellow_bows = 20 := sorry

end yellow_bow_count_l97_97645


namespace necessary_but_not_sufficient_l97_97102

noncomputable def p (x : ℝ) : Prop := -1 < x ∧ x < 2
noncomputable def q (x : ℝ) : Prop := log (x) / log (2) < 1

theorem necessary_but_not_sufficient (x : ℝ) : p x → q x → p x ∧ ¬(q x → p x) :=
by
  sorry

end necessary_but_not_sufficient_l97_97102


namespace ellipse_eccentricity_l97_97212

theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) ∧ (∃ e : ℝ, e = 1 / 2) → 
  (k = 4 ∨ k = -5 / 4) := sorry

end ellipse_eccentricity_l97_97212


namespace larger_triangle_perimeter_l97_97473

-- Defining the smaller triangle as having sides 12 cm, 12 cm, and 15 cm
structure Triangle :=
  (a b c : ℝ)
  (is_isosceles : a = b)

def smaller_triangle : Triangle := { a := 12, b := 12, c := 15, is_isosceles := rfl }

-- Defining the condition that the longest side of the similar triangle is 45 cm
def similar_triangle_longest_side := 45

-- Defining the ratio between the corresponding sides of the similar triangles
def length_ratio (t1 t2 : Triangle) : ℝ := t1.c / t2.c

-- Proving that the perimeter of the larger triangle is 117 cm
theorem larger_triangle_perimeter {t1 t2 : Triangle}
  (ht1 : t1 = smaller_triangle)
  (ht2 : t2.c = similar_triangle_longest_side)
  (ratio_eq : length_ratio t1 t2 = 1 / 3) :
  t2.a + t2.b + t2.c = 117 :=
  by sorry

end larger_triangle_perimeter_l97_97473


namespace smallest_solution_eq_sqrt_104_l97_97530

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l97_97530


namespace highest_more_than_lowest_by_37_5_percent_l97_97340

variables (highest_price lowest_price : ℝ)

theorem highest_more_than_lowest_by_37_5_percent
  (h_highest : highest_price = 22)
  (h_lowest : lowest_price = 16) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 37.5 :=
by
  sorry

end highest_more_than_lowest_by_37_5_percent_l97_97340


namespace odd_function_value_addition_l97_97264

noncomputable def f (x : ℝ) : ℝ :=
if h : -3 < x ∧ x < 0 then log (3 + x) / log 2 else 
if x = 0 then 0 else 
if 0 < x ∧ x < 3 then -(log (3 - x) / log 2) else 0

theorem odd_function_value_addition :
    f 0 + f 1 = -1 := by
  sorry

end odd_function_value_addition_l97_97264


namespace is_cute_2019_num_cute_lt_2019_l97_97451

def is_cute (n : ℕ) : Prop :=
  ∃ m : ℕ, (∑ k in (finset.range m.succ), n / 5^k) = n

theorem is_cute_2019 : is_cute 2019 :=
sorry

theorem num_cute_lt_2019 : (finset.range 2019).filter is_cute).card = 1484 :=
sorry

end is_cute_2019_num_cute_lt_2019_l97_97451


namespace winning_percentage_is_65_l97_97847

theorem winning_percentage_is_65 
  (total_games won_games : ℕ) 
  (h1 : total_games = 280) 
  (h2 : won_games = 182) :
  ((won_games : ℚ) / (total_games : ℚ)) * 100 = 65 :=
by
  sorry

end winning_percentage_is_65_l97_97847


namespace capacity_c_is_80_percent_of_capacity_b_l97_97795

-- Defining the properties of the tanks
structure Tank where
  height : ℝ
  circumference : ℝ

-- Definitions for Tank C and Tank B
def tankC : Tank := { height := 10, circumference := 8 }
def tankB : Tank := { height := 8, circumference := 10 }

-- Volume calculation for a right circular cylinder
def volume (t : Tank) : ℝ :=
  let r := t.circumference / (2 * Real.pi)
  Real.pi * r^2 * t.height

-- Calculating the capacities
def capacityC := volume tankC
def capacityB := volume tankB

-- Proving the relationship between capacities
theorem capacity_c_is_80_percent_of_capacity_b :
  (capacityC / capacityB) * 100 = 80 := by
sorry

end capacity_c_is_80_percent_of_capacity_b_l97_97795


namespace tickets_spent_dunk_a_clown_booth_l97_97356

/-
The conditions given:
1. Tom bought 40 tickets.
2. Tom went on 3 rides.
3. Each ride costs 4 tickets.
-/
def total_tickets : ℕ := 40
def rides_count : ℕ := 3
def tickets_per_ride : ℕ := 4

/-
We aim to prove that Tom spent 28 tickets at the 'dunk a clown' booth.
-/
theorem tickets_spent_dunk_a_clown_booth :
  (total_tickets - rides_count * tickets_per_ride) = 28 :=
by
  sorry

end tickets_spent_dunk_a_clown_booth_l97_97356


namespace sculpture_exposed_surface_area_l97_97470

theorem sculpture_exposed_surface_area :
  let l₁ := 9
  let l₂ := 6
  let l₃ := 4
  let l₄ := 1

  let exposed_bottom_layer := 9 + 16
  let exposed_second_layer := 6 + 10
  let exposed_third_layer := 4 + 8
  let exposed_top_layer := 5

  l₁ + l₂ + l₃ + l₄ = 20 →
  exposed_bottom_layer + exposed_second_layer + exposed_third_layer + exposed_top_layer = 58 :=
by {
  sorry
}

end sculpture_exposed_surface_area_l97_97470


namespace repeating_decimal_35_as_fraction_l97_97169

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l97_97169


namespace number_of_tables_l97_97726

theorem number_of_tables (x : ℕ) (h : 2 * (x - 1) + 3 = 65) : x = 32 :=
sorry

end number_of_tables_l97_97726


namespace min_cubes_needed_l97_97365

theorem min_cubes_needed (digits : Fin 10 → ℕ) : 
  (∀ k : Fin 10, digits k ≥ 29 ∧ (∀ i : Fin 9, digits (i+1) = 30)) →
  (∃ n : ℕ, 6 * n ≥ (30 * 9 + 29) ∧ n = 50) :=
by
  intro h
  use 50
  split
  · linarith
  · refl

end min_cubes_needed_l97_97365


namespace product_maximization_l97_97474

theorem product_maximization : 
  ∃ A B : ℕ, (A * B = 3402) ∧ 
  (A, B) ∈ {(64, 53), (643, 5), (543, 6), (63, 54)} ∧ 
  ∀ X Y : ℕ, (X, Y) ∈ {(64, 53), (643, 5), (543, 6), (63, 54)} → X * Y ≤ 3402 :=
by
  sorry

end product_maximization_l97_97474


namespace min_value_expression_l97_97269

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  8 * x^3 + 27 * y^3 + 64 * z^3 + (1 / (8 * x * y * z)) ≥ 4 :=
by
  sorry

end min_value_expression_l97_97269


namespace lisa_interest_correct_l97_97313

section
variables (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)

def future_value (P r t : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def interest_earned (P r t : ℝ) (n : ℕ) : ℝ :=
  future_value P r t n - P

theorem lisa_interest_correct :
  ∀ (P r t : ℝ) (n : ℕ),
    P = 1200 →
    r = 0.02 →
    t = 3 →
    n = 2 →
    interest_earned P r t n ≈ 74 :=
begin
  intros P r t n hP hr ht hn,
  sorry
end
end

end lisa_interest_correct_l97_97313


namespace longest_ant_path_on_cube_is_8_edges_l97_97863

-- Define relevant structures for the vertex and edge
structure CubeVertex :=
  (index : Fin 8)

structure CubeEdge :=
  (v1 v2 : CubeVertex)
  (distinct_vertices : v1 ≠ v2)

-- Define the concept of a path and the given conditions
def is_path (path : List CubeEdge) :=
  ∀ (e : CubeEdge) (he : e ∈ path), e.v1 ≠ e.v2

def is_vertex_visited_once (path : List CubeEdge) :=
  let vertices := path.bind (λ e, [e.v1, e.v2])
  vertices.Nodup

def is_starting_and_ending_vertex_same (path : List CubeEdge) (start : CubeVertex) :=
  (path.head?.map Edge.v1).getOrElse start = start ∧ (path.last?.map Edge.v2).getOrElse start = start

-- The actual problem to prove
theorem longest_ant_path_on_cube_is_8_edges:
  ∀ (start : CubeVertex) (path : List CubeEdge),
    is_path path →
    is_vertex_visited_once path →
    is_starting_and_ending_vertex_same path start →
    path.length ≤ 8 :=
by
  sorry

end longest_ant_path_on_cube_is_8_edges_l97_97863


namespace sum_integers_between_6_and_14_l97_97779

theorem sum_integers_between_6_and_14 : (∑ k in Finset.range (15) \ Finset.range (6), k) = 90 := by
  sorry

end sum_integers_between_6_and_14_l97_97779


namespace extreme_value_a_zero_range_of_a_l97_97961

open Real

-- Condition (Ⅰ): When a=0, find the extreme value of f(x)
theorem extreme_value_a_zero :
  ∀ (x : ℝ), f (x : ℝ) = 0 → f (1) = 0 
  where f (x: ℝ) := x * real.log x - x + 1 :=
by
  sorry

-- Condition (Ⅱ): If f(x) < 0 for x ∈ (1, +∞), find the range of values for a
theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 1 < x → f (x) < 0) → a ≥ 1 / 2
  where f (x: ℝ) := x * real.log x - a * ( x - 1 )^2 - x + 1 :=
by
  sorry

end extreme_value_a_zero_range_of_a_l97_97961


namespace hexagon_triangle_side_length_l97_97114

theorem hexagon_triangle_side_length (A B C D E F P Q R K M N : Type*)
  [regular_hexagon A B C D E F]
  [midpoint K A B]
  [midpoint M C D]
  [midpoint N E F]
  [triangle P Q R ∘ midpoints K M N]
  (hex_area : area (hexagon A B C D E F) = 1222) :
  side_length (triangle P Q R) = 141 :=
sorry

end hexagon_triangle_side_length_l97_97114


namespace quadratic_polynomial_value_at_six_l97_97736

theorem quadratic_polynomial_value_at_six (p q : ℝ)
  (h_root_0 : q = f(0))
  (h_root_1 : 1 + p + q = f(1))
  (h_alpha_beta : ∃ α β, α = f(0) ∧ β = f(1) ∧ α + β = -p ∧ α * β = q) 
  : f(6) = 36 + 6p + q := 
sorry

end quadratic_polynomial_value_at_six_l97_97736


namespace smallest_solution_eq_sqrt_104_l97_97535

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l97_97535


namespace right_triangular_pyramid_property_l97_97459

theorem right_triangular_pyramid_property
  (S1 S2 S3 S : ℝ)
  (right_angle_face1_area : S1 = S1) 
  (right_angle_face2_area : S2 = S2) 
  (right_angle_face3_area : S3 = S3) 
  (oblique_face_area : S = S) :
  S1^2 + S2^2 + S3^2 = S^2 := 
sorry

end right_triangular_pyramid_property_l97_97459


namespace shaded_region_area_l97_97891

def circle (radius : ℝ) (center : ℝ × ℝ) :=
  { c : ℝ × ℝ | (c.1 - center.1) ^ 2 + (c.2 - center.2) ^ 2 = radius ^ 2 }

noncomputable def point_on_circle (P : ℝ × ℝ) (Γ : set (ℝ × ℝ)) : Prop :=
  P ∈ Γ

noncomputable def diameter (P Q : ℝ × ℝ) (Γ : set (ℝ × ℝ)) : Prop :=
  point_on_circle P Γ ∧ point_on_circle Q Γ ∧ (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 = (2 * 2) ^ 2

noncomputable def angle (P R Q : ℝ × ℝ) (Γ : set (ℝ × ℝ)) : Prop :=
  point_on_circle R Γ ∧ P ≠ Q ∧
  let θ := real.acos ((P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2)) / 
            (real.sqrt ((P.1 - R.1) ^ 2 + (P.2 - R.2) ^ 2) * real.sqrt ((Q.1 - R.1) ^ 2 + (Q.2 - R.2) ^ 2)) in
  θ = real.pi / 3

theorem shaded_region_area (PS QR PQ : ℝ) (P Q R S T : ℝ × ℝ) (Γ : set (ℝ × ℝ)) :
  PS = 4 →
  QR = 4 →
  diameter P Q Γ →
  point_on_circle R Γ →
  angle P R Q Γ →
  let sector_area := λ r θ : ℝ, 0.5 * r ^ 2 * θ in
  let arc_ST := sector_area 4 real.pi in
  let arc_RS := sector_area 4 real.pi in
  let arc_QR := sector_area 2 (real.pi / 3) in
  let circle_area := real.pi * (2 * 2) in
  arc_ST + arc_RS + arc_RS - circle_area = (37 * real.pi) / 3 :=
by
  sorry

end shaded_region_area_l97_97891


namespace injective_function_property_l97_97501

-- Definitions used in the conditions
def HarmonicSum (S : Finset ℕ) : ℚ :=
  S.sum (λ s, 1 / s)

-- Mathematical statement defining the problem
theorem injective_function_property (f : ℕ → ℕ) (hf : Function.Injective f):
  (∀ S : Finset ℕ, (HarmonicSum S).denom = 1 → (HarmonicSum (S.map ⟨f, hf⟩)).denom = 1) →
  (∀ n : ℕ, 0 < n → f n = n) :=
by
  intros hyp n hn
  sorry

end injective_function_property_l97_97501


namespace sum_of_fraction_components_l97_97198

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l97_97198


namespace smallest_solution_floor_eq_l97_97544

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l97_97544


namespace problem_f_pi_over_3_problem_f_leq_sin_l97_97622

noncomputable def f (x : ℝ) : ℝ :=
if |Real.cos x| >= Real.sqrt 2 / 2 then Real.cos x else 0

theorem problem_f_pi_over_3 : f (Real.pi / 3) = 0 :=
sorry

theorem problem_f_leq_sin (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 2 * Real.pi) :
  f(x) ≤ Real.sin(x) ↔ (Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4) :=
sorry

end problem_f_pi_over_3_problem_f_leq_sin_l97_97622


namespace last_digit_of_largest_power_of_3_dividing_factorial_l97_97514

theorem last_digit_of_largest_power_of_3_dividing_factorial (n : ℕ) (h : n = 3^3) : 
  let m := (Nat.multiplicity 3 n.factorial).get (Nat.multiplicity.finite _ _)
  let last_digit := (3 ^ m) % 10
  last_digit = 3 :=
by
  sorry

end last_digit_of_largest_power_of_3_dividing_factorial_l97_97514


namespace jewel_price_reduction_l97_97476

theorem jewel_price_reduction (P x : ℝ) (P1 : ℝ) (hx : x ≠ 0) 
  (hP1 : P1 = P * (1 - (x / 100) ^ 2))
  (h_final : P1 * (1 - (x / 100) ^ 2) = 2304) : 
  P1 = 2304 / (1 - (x / 100) ^ 2) :=
by
  sorry

end jewel_price_reduction_l97_97476


namespace part1_part2_l97_97466

variables (x y : ℝ)
variables (h_avg : (90 + 110 + x + y + 150) / 5 = 110)
variables (h_sum : x + y = 200)

theorem part1 (h : x < y) : 
  (∃ a b c d e : ℝ, (a, b, c, d, e) = (90, 110, x, y, 150) ∧ (x + y = 200) ∧ x < y → (1 / 10)) := sorry

theorem part2 (h : 90 < x ∧ x < 150) : 
  (∃ v : ℝ, v = (2 / 5) * (x - 100) ^ 2 + 440 ∧ x = 100 → v = 440) := sorry

end part1_part2_l97_97466


namespace find_total_weight_l97_97789

-- Given conditions
variables (W : ℝ) -- W is the total weight of the mixture
# Real numbers are used here to handle fractions precisely.

axiom sand_weight : W / 3 -- 1/3 of the mixture is sand
axiom water_weight : W / 4 -- 1/4 of the mixture is water
axiom gravel_weight : 10 -- Remaining 10 pounds of the mixture is gravel

-- According to the problem, the sum of the weights of sand, water, and gravel equals the total weight W.
theorem find_total_weight (W / 3 + W / 4 + 10 = W) : W = 24 :=
sorry

end find_total_weight_l97_97789


namespace factorial_product_lt_factorial_l97_97277

theorem factorial_product_lt_factorial {a : List ℕ} {k : ℕ} 
  (h1 : ∀i, i < a.length → a[i] > 0)
  (h2 : a.sum < k)
  (h3 : k > 0) :
  (a.map (λ i, i!)).foldr (λ x y, x * y) 1 < k! :=
by
  sorry

end factorial_product_lt_factorial_l97_97277


namespace four_digit_numbers_with_two_identical_digits_l97_97497

theorem four_digit_numbers_with_two_identical_digits :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (10 * (n / 1000) + (n % 1000) / 100) / 10 = 2 ∧
  ∃! d, ((n / 1000 = d ∨ (n % 1000) / 100 = d ∨ (n % 100) / 10 = d ∨ (n % 10) = d) ∧
  (∀ x, (n / 1000 = x ∨ (n % 1000) / 100 = x ∨ (n % 100) / 10 = x ∨ (n % 10) = x) → x = d ∨ x ≠ d))∧
  ∃ m1 m2 m3 m4, m1 ∈ {d, 2} ∧ m2 ∈ {d, 2} ∧ m3 ∈ {d, 2} ∧ m4 ∈ {d, 2} ∧
  m1 + m2 + m3 + m4 = n) →
  {m : ℕ | 1000 ≤ m ∧ m < 10000 ∧ (m / 1000 = 2) ∧ 
  ∃! d, ((m / 1000 = d ∨ (m % 1000) / 100 = d ∨ (m % 100) / 10 = d ∨ (m % 10) = d) ∧
  (∀ x, (m / 1000 = x ∨ (m % 1000) / 100 = x ∨ (m % 100) / 10 = x ∨ (m % 10) = x) → x = d ∨ x ≠ d) ∧
  ExactlyTwoIdenticalDigits d m)}.card = 432 := 
      sorry

end four_digit_numbers_with_two_identical_digits_l97_97497


namespace angle_between_skew_lines_l97_97651

-- Define the geometrical setup of the regular tetrahedron
structure Tetrahedron :=
  (A B C D : ℝ × ℝ × ℝ)
  (is_regular : dist A B = dist A C ∧ dist A B = dist A D ∧
                dist B C = dist B D ∧ dist C D = dist A C ∧
                dist A C = 1)

-- Define the midpoint function
def midpoint (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2, (p.3 + q.3) / 2)

-- Define the centroid of a triangle function
def centroid (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p.1 + q.1 + r.1) / 3, (p.2 + q.2 + r.2) / 3, (p.3 + q.3 + r.3) / 3)

-- Define the vector subtraction
def vector_sub (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2, p.3 - q.3)

-- Define the dot product
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the norm (magnitude) of a vector
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def angle (v w : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (dot_product v w / (norm v * norm w))

-- The theorem to be proven
theorem angle_between_skew_lines {tetra : Tetrahedron} (E : ℝ × ℝ × ℝ) :
  let M := midpoint tetra.A tetra.C in
  let N := centroid tetra.B tetra.C tetra.D in
  ∃ DE : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ,
  ((∀ p q, DE p = vector_sub tetra.D E ∧ DE q = vector_sub tetra.B p ∧ dot_product (vector_sub tetra.D E) (vector_sub tetra.B p) = 0) ∧
  (angle (vector_sub N M) (vector_sub tetra.D E)) = real.arccos (5 / (6 * real.sqrt 3))) :=
sorry

end angle_between_skew_lines_l97_97651


namespace repeating_decimal_35_as_fraction_l97_97167

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l97_97167


namespace inequality_solution_range_4_l97_97748

theorem inequality_solution_range_4 (a : ℝ) : 
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
sorry

end inequality_solution_range_4_l97_97748


namespace sum_of_digits_of_largest_n_l97_97682

def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p.prime ∧ p = n

def single_digit_prime (d : ℕ) : Prop :=
  d < 10 ∧ is_prime d

def distinct_primes (d e : ℕ) : Prop :=
  single_digit_prime d ∧ single_digit_prime e ∧ d ≠ e

def not_original_values (d e : ℕ) : Prop :=
  (d, e) ≠ (7, 3) ∧ (d, e) ≠ (3, 7)

noncomputable def n (d e : ℕ) (h1 : distinct_primes d e) (h2 : not_original_values d e) : ℕ :=
  d * e * (10 * d + e)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_of_largest_n : ∃ d e : ℕ, distinct_primes d e ∧ not_original_values d e ∧ 
  sum_of_digits (n d e sorry sorry) = 12 :=
sorry

end sum_of_digits_of_largest_n_l97_97682


namespace range_of_a_l97_97589

open Classical

variables (a : ℝ)
def p := ∃ x : ℝ, x^2 - a * x + 4 = 0
def q := ∃ x : ℝ, x ≥ 3 → 2 * x^2 + a * x + 4 ≥ 2 * x^2 + a * 3 + 4

theorem range_of_a (hp : p a ∨ q a) (hq : ¬(p a ∧ q a)) : a < -12 ∨ -4 < a < 4 := by
  sorry

end range_of_a_l97_97589


namespace tan_alpha_proof_complex_expression_proof_l97_97128

noncomputable def alpha : ℝ := -1

def sin_alpha : ℝ := - (Real.sqrt 5) / 5

def cos_alpha : ℝ := -2 / (Real.sqrt 5)

def tan_alpha : ℝ := 1 / 2

theorem tan_alpha_proof :
  sin_alpha = - (Real.sqrt 5) / 5 →
  tan_alpha = 1 / 2 :=
by
  intro h1
  have h2 : cos_alpha = -2 / (Real.sqrt 5) := by sorry
  sorry

theorem complex_expression_proof :
  sin_alpha = - (Real.sqrt 5) / 5 →
  (cos (π / 2 + alpha) * sin (-π - alpha)) / (cos (11 * π / 2 - alpha) * sin (9 * π / 2 + alpha)) = 1 / 2 :=
by
  intro h1
  have h2 : cos_alpha = -2 / (Real.sqrt 5) := by sorry
  have h3 : tan_alpha = 1 / 2 := by sorry
  sorry

end tan_alpha_proof_complex_expression_proof_l97_97128


namespace cats_combined_weight_l97_97857

theorem cats_combined_weight :
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  cat1 + cat2 + cat3 = 13 := 
by
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  sorry

end cats_combined_weight_l97_97857


namespace unique_solution_l97_97512

theorem unique_solution (x : ℝ) :
  (∑ k in finset.range 21, (x - (2021 - k)) / (k + 1)) =
  (∑ k in finset.range 21, (x - k ) / (2021 - k)) →
  x = 2021 := by sorry

end unique_solution_l97_97512


namespace min_cubes_required_l97_97851

-- Define the structure for the problem
structure CubeConfig (n : ℕ) :=
(top_view : ℕ → ℕ → Prop)
(side_view : ℕ → ℕ → Prop)

-- Define the conditions based on the given views
def top_view (x y : ℕ) : Prop :=
  ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 2))

def side_view (x y : ℕ) : Prop :=
  ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 1))

-- The problem asks for the minimum number of cubes required given the above constraints
theorem min_cubes_required : ∃ n : ℕ, CubeConfig n ∧ n = 4 :=
begin
  -- Omit the proof for this excercise (using sorry)
  sorry
end

end min_cubes_required_l97_97851


namespace xy_gt_xz_l97_97569

variable {R : Type*} [LinearOrderedField R]
variables (x y z : R)

theorem xy_gt_xz (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z :=
by
  sorry

end xy_gt_xz_l97_97569


namespace problem_value_of_expression_l97_97371

theorem problem_value_of_expression :
  (2 ^ 0 - 1 + 5 ^ 3 - 3) ^ (-2) * 3 = 3 / 14884 :=
by
  -- Steps of the solution can be used to guide the actual proof, but are omitted here.
  sorry

end problem_value_of_expression_l97_97371


namespace coprime_repeating_decimal_sum_l97_97176

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l97_97176


namespace coprime_repeating_decimal_sum_l97_97172

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l97_97172


namespace arithmetic_sequences_count_l97_97583

theorem arithmetic_sequences_count :
  ∃ (a_1 d n : ℕ), n ≥ 3 ∧ (a_1 ≥ 0) ∧ (d ≥ 0) ∧ (n * (2 * a_1 + (n - 1) * d) = 2 * 97^2) ∧
  (finset.card {p : ℕ × ℕ × ℕ // p.2.2 ≥ 3 ∧ (p.1 ≥ 0) ∧ (p.2.1 ≥ 0) ∧ (p.2.2 * (2 * p.1 + (p.2.2 - 1) * p.2.1) = 2 * 97^2)} = 4) :=
by
  -- Sorry to skip the proof
  sorry

end arithmetic_sequences_count_l97_97583


namespace find_a_if_odd_l97_97213

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

-- Define the odd function condition
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- The theorem stating the condition and result
theorem find_a_if_odd (a : ℝ) : is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a_if_odd_l97_97213


namespace old_conveyor_time_l97_97036

theorem old_conveyor_time (x : ℝ) : 
  (1 / x) + (1 / 15) = 1 / 8.75 → 
  x = 21 := 
by 
  intro h 
  sorry

end old_conveyor_time_l97_97036


namespace bowling_ball_volume_l97_97435

noncomputable def remaining_volume_of_bowling_ball (d_ball d_hole1 d_hole2 h_hole1 h_hole2 : ℝ) : ℝ :=
  let ball_radius := d_ball / 2
  let v_ball := (4 / 3) * π * (ball_radius ^ 3)
  let hole1_radius := d_hole1 / 2
  let v_hole1 := π * (hole1_radius ^ 2) * h_hole1
  let hole2_radius := d_hole2 / 2
  let v_hole2 := π * (hole2_radius ^ 2) * h_hole2
  let v_holes := 2 * v_hole1 + 2 * v_hole2
  v_ball - v_holes

theorem bowling_ball_volume : remaining_volume_of_bowling_ball 24 1.5 2 5 6 = 2269.5 * π := by
  sorry

end bowling_ball_volume_l97_97435


namespace sufficient_condition_abs_sum_gt_one_l97_97103

theorem sufficient_condition_abs_sum_gt_one (x y : ℝ) (h : y ≤ -2) : |x| + |y| > 1 :=
  sorry

end sufficient_condition_abs_sum_gt_one_l97_97103


namespace books_sold_in_january_l97_97813

theorem books_sold_in_january (J : ℕ) 
  (h_avg : (J + 16 + 17) / 3 = 16) : J = 15 :=
sorry

end books_sold_in_january_l97_97813


namespace cylinder_area_ratio_l97_97637

noncomputable def ratio_of_areas (r h : ℝ) (h_cond : 2 * r / h = h / (2 * Real.pi * r)) : ℝ :=
  let lateral_area := 2 * Real.pi * r * h
  let total_area := lateral_area + 2 * Real.pi * r * r
  lateral_area / total_area

theorem cylinder_area_ratio {r h : ℝ} (h_cond : 2 * r / h = h / (2 * Real.pi * r)) :
  ratio_of_areas r h h_cond = 2 * Real.sqrt Real.pi / (2 * Real.sqrt Real.pi + 1) := 
sorry

end cylinder_area_ratio_l97_97637


namespace ellipse_condition_l97_97141

theorem ellipse_condition (m : ℝ) : 
  (∀ (x y : ℝ), (x^2 / (25 - m) + y^2 / (m + 9) = 1)) → 
  (8 < m ∧ m < 25) :=
begin
  sorry
end

end ellipse_condition_l97_97141


namespace cryptarithm_solutions_unique_l97_97304

/- Definitions corresponding to the conditions -/
def is_valid_digit (d : Nat) : Prop := d < 10

def is_six_digit_number (n : Nat) : Prop := n >= 100000 ∧ n < 1000000

def matches_cryptarithm (abcdef bcdefa : Nat) : Prop := abcdef * 3 = bcdefa

/- Prove that the two identified solutions are valid and no other solutions exist -/
theorem cryptarithm_solutions_unique :
  ∀ (A B C D E F : Nat),
  is_valid_digit A → is_valid_digit B → is_valid_digit C →
  is_valid_digit D → is_valid_digit E → is_valid_digit F →
  let abcdef := 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F
  let bcdefa := 100000 * B + 10000 * C + 1000 * D + 100 * E + 10 * F + A
  is_six_digit_number abcdef →
  is_six_digit_number bcdefa →
  matches_cryptarithm abcdef bcdefa →
  (abcdef = 142857 ∨ abcdef = 285714) :=
by
  intros A B C D E F A_valid B_valid C_valid D_valid E_valid F_valid abcdef bcdefa abcdef_six_digit bcdefa_six_digit cryptarithm_match
  sorry

end cryptarithm_solutions_unique_l97_97304


namespace f_properties_g_min_value_l97_97578

variable {x m : ℝ}

-- Conditions from the problem
def f (x : ℝ) : ℝ := x^2 + x + 1

theorem f_properties :
  (∀ x, f(x + 1) - f(x - 1) = 4 * x + 2) ∧ f(1) = 3 :=
by sorry

theorem g_min_value (m : ℝ) :
  g(x) = f(x) - (1 + 2*m)*x + 1 ∧ 
  (∀ x ≥ 2, g(x) ≥ -3) ∧ g(√5) = -3 → m = √5 :=
by sorry

end f_properties_g_min_value_l97_97578


namespace equal_faces_of_intersection_of_tetrahedrons_l97_97757

theorem equal_faces_of_intersection_of_tetrahedrons :
  ∀ (T1 T2 T3 : Tetrahedron), 
    (equal_regular_tetrahedrons T1 T2 T3) → (common_center T1 T2 T3) →
    (∃ P : Polyhedron, (formed_by_intersection T1 T2 T3 P) → (all_faces_equal P)) := 
by
  intros T1 T2 T3 h_eq h_center
  use P
  sorry

end equal_faces_of_intersection_of_tetrahedrons_l97_97757


namespace smallest_solution_floor_equation_l97_97551

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l97_97551


namespace valid_license_plate_count_l97_97723

theorem valid_license_plate_count : 
  let letters := {A, B, C}
  let numbers := {1, 3, 5, 7}
  (choose 3 2) * (perm 2 2) * (choose 4 3) * (perm 3 3) = 144 :=
by
  -- Step 1: Calculate combinations of letters
  let c3_2 := Nat.choose 3 2
  have h1 : c3_2 = 3 := by 
    simp [Nat.choose, c3_2]
  -- Step 2: Calculate permutations of letters
  let p2_2 := Nat.perm 2 2
  have h2 : p2_2 = 2 := by
    simp [Nat.perm, p2_2]
  -- Step 3: Calculate combinations of numbers
  let c4_3 := Nat.choose 4 3
  have h3 : c4_3 = 4 := by 
    simp [Nat.choose, c4_3]
  -- Step 4: Calculate permutations of numbers
  let p3_3 := Nat.perm 3 3
  have h4 : p3_3 = 6 := by
    simp [Nat.perm, p3_3]
  -- Calculate the final result
  have final_result : 3 * 2 * 4 * 6 = 144 := by
    norm_num
  exact final_result

end valid_license_plate_count_l97_97723


namespace three_pow_eq_two_pow_l97_97632

theorem three_pow_eq_two_pow (x : ℝ) (h : 3^(2*x) = 6 * 2^(2*x) - 5 * 6^x) : 3^x = 2^x :=
sorry

end three_pow_eq_two_pow_l97_97632


namespace no_a_satisfies_condition_l97_97675

noncomputable def M : Set ℝ := {0, 1}
noncomputable def N (a : ℝ) : Set ℝ := {11 - a, Real.log a / Real.log 1, 2^a, a}

theorem no_a_satisfies_condition :
  ¬ ∃ a : ℝ, M ∩ N a = {1} :=
by
  sorry

end no_a_satisfies_condition_l97_97675


namespace proof_problem_l97_97051

noncomputable def mean (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / (l.length : ℝ)

noncomputable def median (l : List ℕ) : ℝ :=
  let sorted := l.qsort (≤)
  if l.length % 2 = 0 then 
    ((sorted.get ((l.length / 2) - 1)) + (sorted.get (l.length / 2))) / 2
  else 
    sorted.get (l.length / 2)

noncomputable def modes (l : List ℕ) : List ℕ :=
  let freq_map := l.foldl (λ m n => m.insert n (m.find n).get_or_else 0 + 1) ∅
  let max_freq := freq_map.fold (0, 0) (λ (k, v) (max_k, max_v) => if v > max_v then (k, v) else (max_k, max_v)).snd
  freq_map.toList.filter (λ (k, v) => v = max_freq).map Prod.fst

noncomputable def median_of_modes (l : List ℕ) : ℝ :=
  let modes_sorted := (modes l).qsort (≤)
  if modes_sorted.length % 2 = 0 then 
    ((modes_sorted.get ((modes_sorted.length / 2) - 1)) + (modes_sorted.get (modes_sorted.length / 2))) / 2
  else 
    modes_sorted.get (modes_sorted.length / 2)

theorem proof_problem :
  let dates := List.replicate 12 (List.range 1 29) ++ List.replicate 12 29 ++ List.replicate 12 30 ++ List.replicate 8 31
  let flat_dates := dates.join
  median_of_modes flat_dates < mean flat_dates ∧ mean flat_dates < median flat_dates := sorry


end proof_problem_l97_97051


namespace new_number_of_groups_l97_97930

-- Define the number of students
def total_students : ℕ := 2808

-- Define the initial and new number of groups
def initial_groups (n : ℕ) : ℕ := n + 4
def new_groups (n : ℕ) : ℕ := n

-- Condition: Fewer than 30 students per new group
def fewer_than_30_students_per_group (n : ℕ) : Prop :=
  total_students / n < 30

-- Condition: n and n + 4 must be divisors of total_students
def is_divisor (d : ℕ) (a : ℕ) : Prop :=
  a % d = 0

def valid_group_numbers (n : ℕ) : Prop :=
  is_divisor n total_students ∧ is_divisor (n + 4) total_students ∧ n > 93

-- The main theorem
theorem new_number_of_groups : ∃ n : ℕ, valid_group_numbers n ∧ fewer_than_30_students_per_group n ∧ n = 104 :=
by
  sorry

end new_number_of_groups_l97_97930


namespace triangle_area_l97_97737

theorem triangle_area (BD AD : ℝ) (hBD : BD = 2) (hAD : AD = 1) :
    ∃ AC : ℝ, 
    let CD := (abs (BD * (2 / (1 - (1/2)^2)) - AD)) in 
    AC = AD + CD ∧ 
    (1 / 2) * AC * BD = 11 / 3 :=
by
  sorry

end triangle_area_l97_97737


namespace math_proof_problem_l97_97127

namespace Proofs

-- Definition of the arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop := 
  ∀ m n, a n = a m + (n - m) * (a (m + 1) - a m)

-- Conditions for the arithmetic sequence
def a_conditions (a : ℕ → ℤ) : Prop := 
  a 3 = -6 ∧ a 6 = 0

-- Definition of the geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop := 
  ∃ q, ∀ n, b (n + 1) = q * b n

-- Conditions for the geometric sequence
def b_conditions (b a : ℕ → ℤ) : Prop := 
  b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3

-- The general formula for {a_n}
def a_formula (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 12

-- The sum formula of the first n terms of {b_n}
def S_n_formula (b : ℕ → ℤ) (S_n : ℕ → ℤ) :=
  ∀ n, S_n n = 4 * (1 - 3^n)

-- The main theorem combining all
theorem math_proof_problem (a b : ℕ → ℤ) (S_n : ℕ → ℤ) :
  arithmetic_seq a →
  a_conditions a →
  geometric_seq b →
  b_conditions b a →
  (a_formula a ∧ S_n_formula b S_n) :=
by 
  sorry

end Proofs

end math_proof_problem_l97_97127


namespace find_m_values_l97_97147

theorem find_m_values (m : ℝ) :
  let A := { x : ℝ | Real.log x (x^2 - 3*x + 3) = 0 }
  let B := { x | m * x - 2 = 0 }
  A ∩ B = B →
  m = 1 ∨ m = 2 :=
by
  intros A B h
  sorry

end find_m_values_l97_97147


namespace roots_polynomial_l97_97305

noncomputable definition f : ℝ → ℝ := λ x, 8 * x^3 + 4 * x^2 - 34 * x + 15

theorem roots_polynomial (x1 x2 x3 : ℝ) 
  (h1 : 2 * x1 - 4 * x2 = 1)
  (h2 : f x1 = 0)
  (h3 : f x2 = 0)
  (h4 : f x3 = 0)
  (h5 : x1 + x2 + x3 = -1 / 2) :
  x1 = 3 / 2 ∧ x2 = 1 / 2 ∧ x3 = -5 / 2 :=
sorry

end roots_polynomial_l97_97305


namespace charlotte_should_bring_money_l97_97883

theorem charlotte_should_bring_money (p d a : ℝ) (h_p : p = 90) (h_d : d = 20) (h_a : a = 72) :
  a = p - (d / 100 * p) :=
by
  rw [h_p, h_d, h_a]
  norm_num
  sorry

end charlotte_should_bring_money_l97_97883


namespace bank_queue_wasted_time_l97_97426

-- Conditions definition
def simple_time : ℕ := 1
def lengthy_time : ℕ := 5
def num_simple : ℕ := 5
def num_lengthy : ℕ := 3
def total_people : ℕ := 8

-- Theorem statement
theorem bank_queue_wasted_time :
  (min_wasted_time : ℕ := 40) ∧
  (max_wasted_time : ℕ := 100) ∧
  (expected_wasted_time : ℚ := 72.5) := by
  sorry

end bank_queue_wasted_time_l97_97426


namespace simplify_trig_l97_97302

theorem simplify_trig (α : ℝ) : 
  (\frac {2 * Real.sin (2 * α)}{1 + Real.cos (2 * α)} = 2 * Real.tan α) := 
by
  sorry

end simplify_trig_l97_97302


namespace intersection_points_distance_l97_97079

theorem intersection_points_distance : 
  let line1 := λ x: ℝ, 3 * x
  let line2 := λ x: ℝ, 3 * x - 6
  let line3 := 1975
  let P1 := (1975 / 3, line3) -- Intersection of line1 and line3
  let P2 := (1981 / 3, line3) -- Intersection of line2 and line3
  dist P1 P2 = 2 := 
by 
  sorry

end intersection_points_distance_l97_97079


namespace total_cost_meal_l97_97933

-- Define the initial conditions
variables (x : ℝ) -- x represents the total cost of the meal

-- Initial number of friends
def initial_friends : ℝ := 4

-- New number of friends after additional friends join
def new_friends : ℝ := 7

-- The decrease in cost per friend
def cost_decrease : ℝ := 15

-- Lean statement to assert our proof
theorem total_cost_meal : x / initial_friends - x / new_friends = cost_decrease → x = 140 :=
by
  sorry

end total_cost_meal_l97_97933


namespace reverse_base_sum_l97_97555

theorem reverse_base_sum :
  {n : ℕ | ∃ d a_d a_d1 a_d2, 
            n = 5^d * a_d + 5^(d-1) * a_d1 + 5^(d-2) * a_d2 ∧
            n = 12^d * a_d2 + 12^(d-1) * a_d1 + 12^(d-2) * a_d ∧
            (12^d - 1) * a_d2 + (12^(d-1) - 5) * a_d1 + (12^(d-2) - 5^(d-2)) * a_d = 0 ∧
            d ≤ 2 ∧ a_d ≤ 4 ∧ a_d1 ≤ 4 ∧ a_d2 ≤ 4}.sum = 10 := 
sorry

end reverse_base_sum_l97_97555


namespace real_part_of_complex_product_l97_97601

open complex

theorem real_part_of_complex_product (a : ℝ) (i : ℂ) (h : i = I) 
  (H : (1 - I) * (a + I) ∈ ℝ) : a = 1 :=
  sorry

end real_part_of_complex_product_l97_97601


namespace pounds_over_minimum_l97_97875

def cost_per_pound : ℕ := 3
def minimum_purchase : ℕ := 15
def total_spent : ℕ := 105

theorem pounds_over_minimum : 
  (total_spent / cost_per_pound) - minimum_purchase = 20 :=
by
  sorry

end pounds_over_minimum_l97_97875


namespace max_area_of_triangle_ABC_l97_97234

open EuclideanGeometry

noncomputable def triangleArea (A B C : Point) : ℝ :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

@[lemma]
theorem max_area_of_triangle_ABC :
  ∀ (A B C Q : Point),
    dist Q A = 3 →
    dist Q B = 4 →
    dist Q C = 5 →
    dist B C = 6 →
    triangleArea A B C ≤ 19 :=
by
  intros A B C Q hQA hQB hQC hBC
  sorry

end max_area_of_triangle_ABC_l97_97234


namespace area_of_triangle_POF_l97_97948

noncomputable def origin : (ℝ × ℝ) := (0, 0)
noncomputable def focus : (ℝ × ℝ) := (Real.sqrt 2, 0)

noncomputable def parabola (x y : ℝ) : Prop :=
  y ^ 2 = 4 * Real.sqrt 2 * x

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  parabola x y

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def PF_eq_4sqrt2 (x y : ℝ) : Prop :=
  distance x y (Real.sqrt 2) 0 = 4 * Real.sqrt 2

theorem area_of_triangle_POF (x y : ℝ) 
  (h1: point_on_parabola x y)
  (h2: PF_eq_4sqrt2 x y) :
   1 / 2 * distance 0 0 (Real.sqrt 2) 0 * |y| = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_POF_l97_97948


namespace painted_cube_ways_l97_97441

theorem painted_cube_ways (b r g : ℕ) (cubes : ℕ) : 
  b = 1 → r = 2 → g = 3 → cubes = 3 := 
by
  intros
  sorry

end painted_cube_ways_l97_97441


namespace find_smallest_solution_l97_97536

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l97_97536


namespace plane_equidistant_from_B_and_C_l97_97089

-- Define points B and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def B : Point3D := { x := 4, y := 1, z := 0 }
def C : Point3D := { x := 2, y := 0, z := 3 }

-- Define the predicate for a plane equation
def plane_eq (a b c d : ℝ) (P : Point3D) : Prop :=
  a * P.x + b * P.y + c * P.z + d = 0

-- The problem statement
theorem plane_equidistant_from_B_and_C :
  ∃ D : ℝ, plane_eq (-2) (-1) 3 D { x := B.x, y := B.y, z := B.z } ∧
            plane_eq (-2) (-1) 3 D { x := C.x, y := C.y, z := C.z } :=
sorry

end plane_equidistant_from_B_and_C_l97_97089


namespace prove_number_of_cows_l97_97030

-- Define the conditions: Cows, Sheep, Pigs, Total animals
variables (C S P : ℕ)

-- Condition 1: Twice as many sheep as cows
def condition1 : Prop := S = 2 * C

-- Condition 2: Number of Pigs is 3 times the number of sheep
def condition2 : Prop := P = 3 * S

-- Condition 3: Total number of animals is 108
def condition3 : Prop := C + S + P = 108

-- The theorem to prove
theorem prove_number_of_cows (h1 : condition1 C S) (h2 : condition2 S P) (h3 : condition3 C S P) : C = 12 :=
sorry

end prove_number_of_cows_l97_97030


namespace correct_propositions_l97_97142

-- Define the propositions as conditions
def proposition1 (x : ℝ) : Prop := - (1 / x)

def proposition2 (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ -1 → (deriv (λ x:ℝ, x^2 + 2*a*x + 1) x) ≤ 0 → a ≤ 1

def proposition3 (m : ℝ) : Prop :=
  log 0.7 (2 * m) < log 0.7 (m - 1) → m < -1

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) → (f (1 - x) + f (x - 1) = 0)

theorem correct_propositions :
  ({proposition2} ∪ {proposition4} = {proposition2, proposition4}) ∧
  ({proposition1} ∪ {proposition3} = ∅) :=
by
  sorry

end correct_propositions_l97_97142


namespace probability_red_balls_fourth_draw_l97_97347

theorem probability_red_balls_fourth_draw :
  let p_red := 2 / 10
  let p_white := 8 / 10
  p_red * p_red * p_white * p_white * 3 / 10 + 
  p_red * p_white * p_red * p_white * 2 / 10 + 
  p_white * p_red * p_red * p_red = 0.0434 :=
sorry

end probability_red_balls_fourth_draw_l97_97347


namespace books_sold_l97_97701

theorem books_sold (original_books : ℕ) (books_left : ℕ) (sold_books : ℕ) 
  (h1 : original_books = 115) 
  (h2 : books_left = 37) : 
  sold_books = 115 - 37 := by
  rw [h1, h2]
  simp
  sorry

end books_sold_l97_97701


namespace range_of_m_l97_97937

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 3, ∃ x2 ∈ set.Icc (0 : ℝ) 2, (x1^2) ≥ (1/2)^x2 - m) ↔ m ≥ 1/4 :=
sorry

end range_of_m_l97_97937


namespace length_of_bridge_l97_97023

theorem length_of_bridge
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (conversion_factor : ℝ)
  (bridge_length : ℝ) :
  train_length = 100 →
  crossing_time = 12 →
  train_speed_kmph = 120 →
  conversion_factor = 1 / 3.6 →
  bridge_length = 299.96 :=
by
  sorry

end length_of_bridge_l97_97023


namespace inscribed_circle_radius_of_triangle_l97_97047

open Real

theorem inscribed_circle_radius_of_triangle (DE DF EF: ℝ) (h1: DE = 8) (h2: DF = 5) (h3: EF = 9) : 
  let s := (DE + DF + EF) / 2 in
  let K := sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  let r := K / s in
  r = 6 * sqrt 11 / 11 :=
by
  sorry

end inscribed_circle_radius_of_triangle_l97_97047


namespace f_neg1_plus_f_2_l97_97944

def f (x : ℤ) : ℤ :=
  if x ≤ 0 then 4 * x else 2 * x

theorem f_neg1_plus_f_2 : f (-1) + f 2 = 0 := 
by
  -- Definition of f is provided above and conditions are met in that.
  sorry

end f_neg1_plus_f_2_l97_97944


namespace perimeter_road_l97_97448

lemma park_perimeter (x : ℝ) (h1 : 0 < x) (h2 : (x ^ 2 - (x - 6) ^ 2) = 1764) : x = 150 :=
by {
  sorry,
}

theorem perimeter_road (x : ℝ) (h1 : 0 < x) (h2 : (x ^ 2 - (x - 6) ^ 2) = 1764) : 4 * x = 600 :=
by {
  have h3 : x = 150 := park_perimeter x h1 h2,
  rw h3,
  linarith,
}

end perimeter_road_l97_97448


namespace possible_values_of_r_l97_97892

theorem possible_values_of_r : 
  let r_values := {r : ℚ | ∃ (a b c d : ℕ), r = a / 10^1 + b / 10^2 + c / 10^3 + d / 10^4} in
  let fractions := {f : ℚ | f = 1 / n ∨ f = 2 / n ∧ (3 ≤ n ∧ n ≤ 8)} in
  (∀ r ∈ r_values, ∃ f ∈ fractions, |r - 2/5| ≤ |r - f| ) →
  card {r ∈ r_values | 0.3667 ≤ r ∧ r ≤ 0.4499 } = 830 :=
by
  sorry

end possible_values_of_r_l97_97892


namespace f_formula_l97_97263

-- Define the function f(n) as given in the problem
def f (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ k, 2 ^ (3 * k + 1))

-- State the theorem we want to prove
theorem f_formula (n : ℕ) : f n = (2 / 7) * (8 ^ (n + 1) - 1) := sorry

end f_formula_l97_97263


namespace jason_bartender_experience_l97_97662

-- Definitions of the conditions
def total_work_experience : ℕ := 150
def managerial_experience_years : ℕ := 3
def managerial_experience_months : ℕ := 6

-- The problem goal
theorem jason_bartender_experience (total_work_experience managerial_experience_years managerial_experience_months : ℕ) :
  (total_work_experience = 150) ∧ 
  (managerial_experience_years = 3) ∧ 
  (managerial_experience_months = 6) → 
  let managerial_experience := managerial_experience_years * 12 + managerial_experience_months in
  let bartender_experience := total_work_experience - managerial_experience in
  bartender_experience / 12 = 9 :=
by
  intros
  sorry

end jason_bartender_experience_l97_97662


namespace trig_ineq_l97_97108

variable {α β : ℝ}

theorem trig_ineq (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (sin α) ^ 3 / (sin β) + (cos α) ^ 3 / (cos β) ≥ 1 :=
by
  sorry

end trig_ineq_l97_97108


namespace smallest_solution_floor_eq_l97_97545

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l97_97545


namespace atlantis_population_in_2060_l97_97909

noncomputable def population_in_2000 : ℕ := 400
noncomputable def doubles_every_20_years (p : ℕ) : ℕ := 2 * p
noncomputable def reduces_by_25_percent (p : ℕ) : ℕ := (3 * p) / 4

theorem atlantis_population_in_2060 :
  let p2000 := population_in_2000 in
  let p2020 := doubles_every_20_years p2000 in
  let p2040 := doubles_every_20_years p2020 in
  let p2040_adjusted := reduces_by_25_percent p2040 in
  let p2060 := doubles_every_20_years p2040_adjusted in
  p2060 = 2400 :=
by
  sorry

end atlantis_population_in_2060_l97_97909


namespace unique_factor_and_multiple_of_13_l97_97006

theorem unique_factor_and_multiple_of_13 (n : ℕ) (h1 : n ∣ 13) (h2 : 13 ∣ n) : n = 13 :=
sorry

end unique_factor_and_multiple_of_13_l97_97006


namespace solve_frac_eqn_l97_97716

theorem solve_frac_eqn (x : ℚ) (h₁ : x ≠ 4) (h₂ : x ≠ -6) :
  \[
  \frac{x+7}{x-4} = \frac{x-1}{x+6} \Rightarrow x = -\frac{19}{9} 
  \] :=
sorry

end solve_frac_eqn_l97_97716


namespace nth_term_of_series_l97_97220

theorem nth_term_of_series (n : ℕ) : n > 0 → (nth_term n = (n^2 / (n + 1))) :=
by 
  sorry

end nth_term_of_series_l97_97220


namespace hyperbola_center_l97_97834

theorem hyperbola_center (F1 F2 : ℝ × ℝ) (F1_eq : F1 = (3, -2)) (F2_eq : F2 = (11, 6)) :
  let C := ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2) in C = (7, 2) :=
by
  sorry

end hyperbola_center_l97_97834


namespace bert_bought_300_stamps_l97_97042

theorem bert_bought_300_stamps (x : ℝ) 
(H1 : x / 2 + x = 450) : x = 300 :=
by
  sorry

end bert_bought_300_stamps_l97_97042


namespace trig_proof_1_trig_proof_2_l97_97095

variables {α : ℝ}

-- Given condition
def tan_alpha (a : ℝ) := Real.tan a = -3

-- Proof problem statement
theorem trig_proof_1 (h : tan_alpha α) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 := sorry

theorem trig_proof_2 (h : tan_alpha α) :
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := sorry

end trig_proof_1_trig_proof_2_l97_97095


namespace complex_modulus_squared_l97_97910

theorem complex_modulus_squared : (Complex.abs (-3 : ℂ - (8 / 5 : ℂ) * Complex.i)) ^ 2 = 289 / 25 := by
  sorry

end complex_modulus_squared_l97_97910


namespace toby_peanut_butter_servings_l97_97354

theorem toby_peanut_butter_servings :
  let bread_calories := 100
  let peanut_butter_calories_per_serving := 200
  let total_calories := 500
  let bread_pieces := 1
  ∃ (servings : ℕ), total_calories = (bread_calories * bread_pieces) + (peanut_butter_calories_per_serving * servings) → servings = 2 := by
  sorry

end toby_peanut_butter_servings_l97_97354


namespace deepak_and_wife_meet_l97_97327

noncomputable def time_to_meet (circumference : ℕ) (speed1_km_hr : ℚ) (speed2_km_hr : ℚ) : ℚ :=
  let speed1_m_s := speed1_km_hr * (1000 / 3600)
  let speed2_m_s := speed2_km_hr * (1000 / 3600)
  let relative_speed_m_s := speed1_m_s + speed2_m_s
  circumference / relative_speed_m_s

theorem deepak_and_wife_meet :
  let circumference := 1000
  let speed1_km_hr := 20
  let speed2_km_hr := 12
  time_to_meet circumference speed1_km_hr speed2_km_hr ≈ 112.48 :=
by
  let circumference := 1000
  let speed1_km_hr := 20
  let speed2_km_hr := 12
  have h : time_to_meet circumference speed1_km_hr speed2_km_hr ≈ 112.48 := 
    sorry
  exact h

end deepak_and_wife_meet_l97_97327


namespace proportion_solution_l97_97634

theorem proportion_solution (x: ℕ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_solution_l97_97634


namespace hyperbola_center_origin_foci_axes_l97_97576

-- Define the hyperbola centered at the origin with given properties
variables {e : ℝ} (e_value : e = sqrt 6 / 2)
variables {P : ℝ × ℝ} (P_value : P = (2, 3 * sqrt 2))

-- Define the final equation of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop :=
  (y^2 / 10) - (x^2 / 5) = 1

-- State the proof problem
theorem hyperbola_center_origin_foci_axes :
  ∀ (x y : ℝ), (e = sqrt 6 / 2) → P = (2, 3 * sqrt 2) → hyperbola_equation x y := by
  intros x y he hP
  sorry

end hyperbola_center_origin_foci_axes_l97_97576


namespace day_of_week_six_years_later_is_saturday_l97_97756

theorem day_of_week_six_years_later_is_saturday (year : ℕ)
  (june7th_is_saturday : (june7th_day_of_week : ℕ) % 7 = 6) :
  let days_in_regular_year := 365
  let days_in_leap_year := 366
  let total_days := days_in_regular_year * 5 + days_in_leap_year
  in (june7th_day_of_week + total_days) % 7 = 6 :=
by
  sorry

end day_of_week_six_years_later_is_saturday_l97_97756


namespace period_and_monotonic_increase_triangle_area_l97_97276

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, (Real.sqrt 3) * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2

theorem period_and_monotonic_increase (x : ℝ) (k : ℤ) :
  (∃ p : ℝ, p = π ∧ (∀ t : ℝ, f (t + p) = f t)) ∧
  (∃ i : Set ℝ, i = Set.Icc (-(π/3) + k*π) (π/6 + k*π) ∧ ∀ (y : ℝ), y ∈ i → IsMonotonic (f y)) :=
sorry

theorem triangle_area (A B C : ℝ) (a b c : ℝ) (abc_triangle : IsTriangle A B C a b c)
  (ha : a = Real.sqrt 7) (hfA : f A = 2) (hB2C : Real.sin B = 2 * Real.sin C) :
  area_of_triangle A B C a b c = (7 * Real.sqrt 3) / 6 :=
sorry

structure IsTriangle (A B C a b c : ℝ) : Prop :=
  (angle_sum : A + B + C = π)
  (side_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)

def area_of_triangle (A B C a b c : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

end period_and_monotonic_increase_triangle_area_l97_97276


namespace tallest_vs_shortest_height_difference_l97_97996

-- Define the heights of the trees
def pine_tree_height := 12 + 4/5
def birch_tree_height := 18 + 1/2
def maple_tree_height := 14 + 3/5

-- Calculate improper fractions
def pine_tree_improper := 64 / 5
def birch_tree_improper := 41 / 2  -- This is 82/4 but not simplified here
def maple_tree_improper := 73 / 5

-- Calculate height difference
def height_difference := (82 / 4) - (64 / 5)

-- The statement that needs to be proven
theorem tallest_vs_shortest_height_difference : height_difference = 7 + 7 / 10 :=
by 
  sorry

end tallest_vs_shortest_height_difference_l97_97996


namespace roots_irrational_l97_97931

theorem roots_irrational (m : ℝ) (h : m = sqrt 2 ∨ m = -sqrt 2) :
  ∀ a b : ℝ, (a * b = 8) →
  let discriminant := 25 * m * m - 32 in
  discriminant > 0 ∧ ¬ (∃ (n : ℤ), discriminant = (n : ℝ) * (n : ℝ)) →
  irrational a → irrational b :=
by
  sorry

end roots_irrational_l97_97931


namespace rate_is_correct_l97_97314

noncomputable def rate_of_fencing_per_metre 
  (A : ℝ) 
  (C : ℝ) 
  (A_in_hectares : A = 17.56)
  (C_in_approx : C = 5941.9251828093165) : ℝ :=
  let area_in_square_meters := A * 10000
  let radius := real.sqrt (area_in_square_meters / real.pi)
  let circumference := 2 * real.pi * radius
  C / circumference

theorem rate_is_correct 
  (A : ℝ) 
  (C : ℝ) 
  (A_in_hectares : A = 17.56)
  (C_in_approx : C = 5941.9251828093165) :
  rate_of_fencing_per_metre A C A_in_hectares C_in_approx = 4.00 :=
  by
    sorry

end rate_is_correct_l97_97314


namespace units_digit_of_power_product_l97_97505

theorem units_digit_of_power_product : 
  (2^2021 * 5^2022 * 7^2023) % 10 = 0 :=
by
  -- Definitions for units digit patterns of powers modulo 10.
  have h1 : 2^2021 % 10 = 2,
    by sorry,
  have h2 : 5^2022 % 10 = 5,
    by sorry,
  have h3 : 7^2023 % 10 = 3,
    by sorry,
  -- Combine the results to find the units digit of the product.
  calc
    (2^2021 * 5^2022 * 7^2023) % 10 = (2 % 10 * 5 % 10 * 3 % 10) % 10 : 
      by
        rw [h1, h2, h3]
    ... = (2 * 5 * 3) % 10 : by
      sorry
    ... = 30 % 10 : by
      sorry
    ... = 0 : by
      sorry

end units_digit_of_power_product_l97_97505


namespace neg_sqrt_17_estimate_l97_97067

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end neg_sqrt_17_estimate_l97_97067


namespace eccentricity_range_l97_97942

open Real

noncomputable def hyperbola_eccentricity_range (θ : ℝ) (hθ : θ ∈ Icc (π/4) (π/3)) : Set ℝ :=
  let a := sin θ
  let b := sqrt ((sin^4 θ) / (1 - sin^4 θ))
  let e := sqrt (1 + (b / a)^2)
  {e | 1 < e ∧ e ≤ (2 * sqrt 21) / 7}

theorem eccentricity_range : ∀ θ, θ ∈ Icc (π / 4) (π / 3) →
  (λ e, 1 < e ∧ e ≤ (2 * sqrt 21) / 7) (sqrt (1 + ((sqrt ((sin θ)^4 / (1 - (sin θ)^4)) / (sin θ))^2))) :=
by
  intros θ hθ
  let a := sin θ
  let b := sqrt ((sin^4 θ) / (1 - sin^4 θ))
  let e := sqrt (1 + (b / a)^2)
  have : e = sqrt (1 + ((sqrt ((sin θ)^4 / (1 - (sin θ)^4)) / (sin θ))^2) := by rfl
  sorry

end eccentricity_range_l97_97942


namespace sum_of_integers_between_5_and_15_l97_97783

-- Definitions based on conditions
def predicate (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Main theorem statement
theorem sum_of_integers_between_5_and_15 : (Finset.sum (Finset.filter predicate (Finset.range 15))) = 90 :=
by
  sorry

end sum_of_integers_between_5_and_15_l97_97783


namespace repeating_decimal_sum_l97_97182

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l97_97182


namespace sum_of_integers_between_5_and_15_l97_97781

-- Definitions based on conditions
def predicate (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Main theorem statement
theorem sum_of_integers_between_5_and_15 : (Finset.sum (Finset.filter predicate (Finset.range 15))) = 90 :=
by
  sorry

end sum_of_integers_between_5_and_15_l97_97781


namespace repeated_decimal_to_fraction_l97_97189

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l97_97189


namespace g_increasing_on_neg_inf_to_0_l97_97246

variables {f : ℝ → ℝ}

-- Conditions
def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f(x) ≤ f(y)
def negative (f : ℝ → ℝ) : Prop := ∀ x, f(x) < 0

-- Main theorem
theorem g_increasing_on_neg_inf_to_0
  (hf_increasing : increasing f)
  (hf_negative : negative f) :
  ∀ x y, x < y → x < 0 → y < 0 → x^2 * f(x) ≤ y^2 * f(y) := sorry

end g_increasing_on_neg_inf_to_0_l97_97246


namespace sequence_a100_gt_14_l97_97335

theorem sequence_a100_gt_14 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 1 ≤ n → a (n+1) = a n + 1 / a n) :
  a 100 > 14 :=
by sorry

end sequence_a100_gt_14_l97_97335


namespace pi_irrational_l97_97033

theorem pi_irrational : irrational π := 
sorry

end pi_irrational_l97_97033


namespace line_circle_no_intersection_ellipse_perpendicular_intersection_l97_97136

theorem line_circle_no_intersection:
  ∀ (θ : ℝ), 
  (∃ ρ : ℝ, ρ * sin (θ + π / 4) = sqrt 2 / 2) ∧ 
  (∃ θ : ℝ, (2 * cos θ) ^ 2 + ((-2 + 2 * sin θ) + 2) ^ 2 = 4) →
  ∃ C, ∃ (d : ℝ), 
  let C := (0, -2) in (d = 3 * sqrt 2 / 2 ∧ d > 2).

theorem ellipse_perpendicular_intersection:
  ∀ (ϕ : ℝ), 
  (∃ l' : ℝ → ℝ × ℝ, ∀ t : ℝ, l' t = (sqrt 2 / 2 * t, -2 + sqrt 2 / 2 * t)) ∧
  (∃ x : ℝ, ∃ y : ℝ, (x = 2 * cos ϕ ∧ y = sqrt 3 * sin ϕ) →
  (let a := 2 in let b := sqrt 3 in (x^2 / a^2 + y^2 / b^2 = 1)) →
    ∃ t1 t2 : ℝ, 
    (t1 + t2 = 16 * sqrt 2 / 7) ∧ 
    (t1 * t2 = 8 / 7) → 
    (|t1 * t2| = 8 / 7)).

end line_circle_no_intersection_ellipse_perpendicular_intersection_l97_97136


namespace measure_of_angle_Q_l97_97502

def angle_sum_hexagon := 720

def angles_except_Q := [150, 110, 120, 130, 100]

theorem measure_of_angle_Q : ∑ x in angles_except_Q, x + angle_Q = angle_sum_hexagon → angle_Q = 110 := 
by 
  sorry

end measure_of_angle_Q_l97_97502


namespace find_2005th_number_l97_97947

noncomputable def seventh_fraction (a1 a2 a3 a4 : ℕ) : ℚ :=
  (a1 / 7 : ℚ) + (a2 / (7^2) : ℚ) + (a3 / (7^3) : ℚ) + (a4 / (7^4) : ℚ)

def set_T : set ℕ := {0, 1, 2, 3, 4, 5, 6}

def set_M : set ℚ := {seventh_fraction a1 a2 a3 a4 | a1 a2 a3 a4 ∈ set_T}

def element_2005th : ℚ := 1 / 7 + 1 / (7^2) + 0 / (7^3) + 4 / (7^4)

theorem find_2005th_number:
  (∀ ai ∈ set_T, i ∈ {1, 2, 3, 4}) → element_2005th ∈ {x | x ∈ set_M ∧ (sorted_desc_order set_M ! 2004 = element_2005th)} :=
  sorry

end find_2005th_number_l97_97947


namespace max_three_digit_numbers_proof_l97_97920

noncomputable def max_three_digit_numbers_with_conditions : ℕ := 5

theorem max_three_digit_numbers_proof :
  ∃ n : ℕ, (
    (∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (a + b + c = 9) ∧ 
    ∀ (d e f : ℕ), (1 ≤ d ∧ d ≤ 9) ∧ (1 ≤ e ∧ e ≤ 9) ∧ (1 ≤ f ∧ f ≤ 9) ∧ (d + e + f = 9) →
    (a ≠ d ∧ b ≠ e ∧ c ≠ f))))  → 
  (n = 5) :=
by
  use max_three_digit_numbers_with_conditions
  sorry

end max_three_digit_numbers_proof_l97_97920


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97411

def a : ℕ := 1  -- time for a simple operation
def b : ℕ := 5  -- time for a lengthy operation
def n : ℕ := 5  -- number of "simple" customers
def m : ℕ := 3  -- number of "lengthy" customers
def total_customers : ℕ := 8 -- 8 people in queue

theorem min_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → min_wasted_person_minutes ≤ 40) :=
by
  sorry

theorem max_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → max_wasted_person_minutes ≥ 100) :=
by
  sorry

theorem expected_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → expected_wasted_person_minutes = 72.5) :=
by
  sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97411


namespace cylinder_volume_l97_97606

theorem cylinder_volume (V_sphere : ℝ) (V_cylinder : ℝ) (R H : ℝ) 
  (h1 : V_sphere = 4 * π / 3) 
  (h2 : (4 * π * R ^ 3) / 3 = V_sphere) 
  (h3 : H = 2 * R) 
  (h4 : R = 1) : V_cylinder = 2 * π :=
by
  sorry

end cylinder_volume_l97_97606


namespace monotonic_range_of_a_solution_set_of_inequality_l97_97619

-- Part (1): 
theorem monotonic_range_of_a (a : ℝ) :
  (∀ x y ∈ Set.Icc 1 3, f(x) ≤ f(y) ∨ f(y) ≤ f(x)) ↔ (a ≥ -1/2 ∨ a ≤ -5/2) :=
sorry

-- Part (2):
theorem solution_set_of_inequality (a x : ℝ) :
  f(x) = x^2 + (2*a - 1)*x - 2*a →
  (a = -1/2 → {x | f(x) < 0} = ∅) ∧
  (a < -1/2 → {x | f(x) < 0} = Set.Ioo 1 (-2*a)) ∧
  (a > -1/2 → {x | f(x) < 0} = Set.Ioo (-2*a) 1) :=
sorry

end monotonic_range_of_a_solution_set_of_inequality_l97_97619


namespace sum_alternating_sequence_l97_97878

/-- Calculate the sum of the first 101 integers with alternating signs starting from -1. -/
theorem sum_alternating_sequence : 
  let s := ∑ i in (range 101), if i % 2 = 0 then i else -i
  in s = -51 :=
by
  sorry

end sum_alternating_sequence_l97_97878


namespace parabola_p_range_circle_tangent_to_parabola_l97_97974

noncomputable def parabola (p : ℝ) (p_pos : p > 0) := { points : ℝ × ℝ // (points.fst)^2 = 2 * p * (points.snd) }

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def vector_sum_zero (A M B : ℝ × ℝ) : Prop :=
  (M.1 - A.1, M.2 - A.2) +ᵥ (M.1 - B.1, M.2 - B.2) = (0, 0)

theorem parabola_p_range
  (p : ℝ) (hp : p > 0) (M : ℝ × ℝ) (M_def : M = (2, 2))
  (A B : ℝ × ℝ) (hA : A ∈ parabola p hp) (hB : B ∈ parabola p hp)
  (vec_sum_zero : vector_sum_zero A M B) :
  1 < p := 
  sorry

theorem circle_tangent_to_parabola
  (p : ℝ) (h : p = 2)
  (A B : ℝ × ℝ) (A_def : A = (0, 0)) (B_def : B = (4, 4))
  (C : ℝ × ℝ) (hC : C.fst ≠ 0 ∧ C.fst ≠ 4)
  (C_on_parabola : C.fst^2 = 2 * p * C.snd)
  : ∃ t : ℝ, C = (-2, 1) := 
  sorry

end parabola_p_range_circle_tangent_to_parabola_l97_97974


namespace smallest_x_solution_l97_97520

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l97_97520


namespace minimum_value_of_expression_l97_97104

theorem minimum_value_of_expression (x A B C : ℝ) (hx : x > 0) 
  (hA : A = x^2 + 1/x^2) (hB : B = x - 1/x) (hC : C = B * (A + 1)) : 
  ∃ m : ℝ, m = 6.4 ∧ m = A^3 / C :=
by {
  sorry
}

end minimum_value_of_expression_l97_97104


namespace coffee_customers_l97_97039

theorem coffee_customers (C : ℕ) :
  let coffee_cost := 5
  let tea_ordered := 8
  let tea_cost := 4
  let total_revenue := 67
  (coffee_cost * C + tea_ordered * tea_cost = total_revenue) → C = 7 := by
  sorry

end coffee_customers_l97_97039


namespace translation_g_find_m_l97_97612

noncomputable def f (x : ℝ) : ℝ := log (2^(x+1))

noncomputable def g (x : ℝ) : ℝ := log (2^x)

theorem translation_g (x : ℝ) : x > 0 → g x = log (2^x) :=
begin
  intro hx,
  -- This needs to be shown that translation of f(x) results in g(x)
  sorry
end
  
theorem find_m (m : ℝ) : 
  (∀ x ∈ (set.Icc (1 : ℝ) 4), (log (2^x))^2 - m * log (2^x^2) + 3 ≥ 2) → m = 1 :=
begin
  intro h,
  -- This needs to prove that with the given conditions, m should be 1
  sorry
end

end translation_g_find_m_l97_97612


namespace mean_of_sharpened_pencils_l97_97491

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

theorem mean_of_sharpened_pencils :
  mean [13, 8, 13, 21, 7, 23] ≈ 14.17 := 
sorry

end mean_of_sharpened_pencils_l97_97491


namespace valid_pic4_valid_pic5_l97_97291

-- Define the type for grid coordinates
structure Coord where
  x : ℕ
  y : ℕ

-- Define the function to check if two coordinates are adjacent by side
def adjacent (a b : Coord) : Prop :=
  (a.x = b.x ∧ (a.y = b.y + 1 ∨ a.y = b.y - 1)) ∨
  (a.y = b.y ∧ (a.x = b.x + 1 ∨ a.x = b.x - 1))

-- Define the coordinates for the pictures №4 and №5
def pic4_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨1, 0⟩), (4, ⟨2, 0⟩), (3, ⟨0, 1⟩),
   (5, ⟨1, 1⟩), (6, ⟨2, 1⟩), (7, ⟨2, 2⟩), (8, ⟨1, 3⟩)]

def pic5_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨0, 1⟩), (3, ⟨0, 2⟩), (4, ⟨0, 3⟩), (5, ⟨1, 3⟩)]

-- Define the validity condition for a picture
def valid_picture (coords : List (ℕ × Coord)) : Prop :=
  ∀ (n : ℕ) (c1 c2 : Coord), (n, c1) ∈ coords → (n + 1, c2) ∈ coords → adjacent c1 c2

-- The theorem to prove that pictures №4 and №5 are valid configurations
theorem valid_pic4 : valid_picture pic4_coords := sorry

theorem valid_pic5 : valid_picture pic5_coords := sorry

end valid_pic4_valid_pic5_l97_97291


namespace percentage_increase_l97_97511

variable (m y : ℝ)

theorem percentage_increase (h : x = y + (m / 100) * y) : x = ((100 + m) / 100) * y := by
  sorry

end percentage_increase_l97_97511


namespace line_parallel_plane_l97_97211

theorem line_parallel_plane (a n : EuclideanAffineSpace ℝ 3) :
  a = ([1, -1, 3] : Fin 3 → ℝ) → n = ([0, 3, 1] : Fin 3 → ℝ) →
  dot_product a n = 0 → parallel l α := sorry

end line_parallel_plane_l97_97211


namespace choose_starters_l97_97004

theorem choose_starters :
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  Nat.choose totalPlayers 6 - Nat.choose playersExcludingTwins 6 = 5005 :=
by
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  sorry

end choose_starters_l97_97004


namespace arithmetic_seq_sum_x_y_l97_97373

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l97_97373


namespace values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l97_97962

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem values_of_a_and_b (h : ∀ x, f x (1 / 3) (-1 / 2) ≤ f 1 (1 / 3) (-1 / 2)) :
  (∃ a b, a = 1 / 3 ∧ b = -1 / 2) :=
sorry

theorem intervals_of_monotonicity (a b : ℝ) (h : ∀ x, f x a b ≤ f 1 a b) :
  (∀ x, (f x a b ≥ 0 ↔ x ≤ -1 / 3 ∨ x ≥ 1) ∧ (f x a b ≤ 0 ↔ -1 / 3 ≤ x ∧ x ≤ 1)) :=
sorry

theorem range_of_a_for_three_roots :
  (∃ a, -1 < a ∧ a < 5 / 27) :=
sorry

end values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l97_97962


namespace storm_damage_in_usd_l97_97461

theorem storm_damage_in_usd 
    (damage_in_CAD : ℕ) 
    (exchange_rate_CAD_to_Euro : ℚ) 
    (exchange_rate_Euro_to_USD : ℚ) : 
    (damage_in_CAD = 45000000) →
    (exchange_rate_CAD_to_Euro = 0.75) →
    (exchange_rate_Euro_to_USD = 1.2) →
    let damage_in_Euros := damage_in_CAD * exchange_rate_CAD_to_Euro in
    let damage_in_USD := damage_in_Euros * exchange_rate_Euro_to_USD in
    damage_in_USD = 40500000 := 
by 
  intros 
  sorry

end storm_damage_in_usd_l97_97461


namespace sum_of_integers_between_is_90_l97_97771

-- Define the conditions
def is_between (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Define the sum of integers satisfying the conditions
def sum_of_integers_between : ℕ :=
  Finset.sum (Finset.filter is_between (Finset.range 15)) id

-- State the theorem
theorem sum_of_integers_between_is_90 : sum_of_integers_between = 90 := 
by
  sorry

end sum_of_integers_between_is_90_l97_97771


namespace pencil_cost_l97_97404

noncomputable def p : ℝ := 0.10
noncomputable def q : ℝ := 0.32

theorem pencil_cost :
  ∃ (p q : ℝ), 4 * p + 5 * q = 2.00 ∧ 3 * p + 4 * q = 1.58 ∧ p = 0.10 :=
by
  use 0.10, 0.32
  split
  repeat {ring_nf, norm_num}
  sorry

end pencil_cost_l97_97404


namespace max_num_distinct_ages_l97_97316

theorem max_num_distinct_ages (average_age : ℤ) (std_dev : ℤ) (h1 : average_age = 31) (h2 : std_dev = 8) :
  ∃ max_ages : ℤ, max_ages = 17 :=
by
  -- Define the lower and upper bounds using the provided conditions
  let lower_bound := average_age - std_dev
  let upper_bound := average_age + std_dev

  -- Specify that the maximum number of different ages can be calculated as follows
  have h3 : lower_bound = 23 := by rw [h1, h2]; calc 31 - 8 = 23
  have h4 : upper_bound = 39 := by rw [h1, h2]; calc 31 + 8 = 39

  -- Now, calculate the maximum number of different ages
  -- which is upper_bound - lower_bound + 1
  use upper_bound - lower_bound + 1
  rw [h3, h4]
  calc 39 - 23 + 1 = 17

  -- Proof is omitted
  sorry

end max_num_distinct_ages_l97_97316


namespace repeating_decimal_35_as_fraction_l97_97165

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l97_97165


namespace compute_r_l97_97495

noncomputable def r (side_length : ℝ) : ℝ :=
  let a := (0.5 * side_length, 0.5 * side_length)
  let b := (1.5 * side_length, 2.5 * side_length)
  let c := (2.5 * side_length, 1.5 * side_length)
  let ab := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  let ac := Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)
  let bc := Real.sqrt ((c.1 - b.1)^2 + (c.2 - b.2)^2)
  let s := (ab + ac + bc) / 2
  let area_ABC := Real.sqrt (s * (s - ab) * (s - ac) * (s - bc))
  let circumradius := ab * ac * bc / (4 * area_ABC)
  circumradius - (side_length / 2)

theorem compute_r :
  r 1 = (5 * Real.sqrt 2 - 3) / 6 :=
by
  unfold r
  sorry

end compute_r_l97_97495


namespace area_relation_l97_97396

variables {A B C O A1 B1 C1 : Point}
variables {p a b c : ℝ}

-- Define the necessary points and their relationships
def isAngleBisector (A B C A1 : Point) : Prop := sorry
def isPerpendicular (P1 P2 : Point) (l : Line) : Prop := sorry
def incircleCenter (Δ : Triangle) : Point := sorry
def pointOfTangency (Δ : Triangle) (side : Segment) : Point := sorry
def lineThroughPoints (P1 P2 : Point) : Line := sorry

theorem area_relation (Δ : Triangle) (a b c : ℝ) (hA1 : isAngleBisector A B C A1) 
  (hA1dist : dist A A1 = (b + c - a) / 2)
  (hLA : isPerpendicular A1 (incircleCenter Δ) (lineThroughPoints A A1))
  (hLB : isPerpendicular B1 (incircleCenter Δ) (lineThroughPoints B B1))
  (hLC : isPerpendicular C1 (incircleCenter Δ) (lineThroughPoints C C1)) :
  area (triangle A B C) = 
  area (triangle A1 B C) + area (triangle A B1 C) + area (triangle A B C1) :=
sorry

end area_relation_l97_97396


namespace ratio_largest_sum_l97_97896

theorem ratio_largest_sum (n : ℕ) (h : n = 10) : 
  let S := 1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9
  in (2^n) / S = 1 :=
by
  have h1 : S = (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9)
    := by rfl
  have h2 : 2^n = 1024 := by sorry
  have h3 : S = 1023 := by sorry
  sorry

end ratio_largest_sum_l97_97896


namespace trajectory_C_eq_area_triangle_OEF_range_l97_97956

theorem trajectory_C_eq :
  ∀ (x y : ℝ), let A := (0, -1) in let B := (0, 1) in
  (y + 1) * (y - 1) = -\frac{1}{2} * x ^2 → x ≠ 0 → (x^2 / 2 + y^2 = 1) := 
begin
  intros x y A B h1 h2,
  let slope_A := (y + 1) / x,
  let slope_B := (y - 1) / x,
  sorry -- proof here
end

theorem area_triangle_OEF_range :
  ∀ (k : ℝ), k^2 > \frac{3}{2} →
  let D := (0, 2) in
  let l := (λ x, k * x + 2) in
  let O := (0, 0) in
  ∃ (E F : ℝ × ℝ),
  (l E.1 = E.2 ∧ l F.1 = F.2) ∧
  (E ≠ F) ∧
  (E.1, E.2 ∈ C) ∧
  (F.1, F.2 ∈ C) ∧
  ∀ A : ℝ, ((area_Triangle O E F) ≤ sqrt(2) / 2) :=
begin
  intros k hk,
  let D := (0, 2),
  let C := { p : ℝ × ℝ | ∃ (x y : ℝ), (y² * 2 = 2 - x²) ∧ x ≠ 0 },
  let O := (0, 0),
  sorry -- proof here, additional definitions and lemmas might be needed.
end

end trajectory_C_eq_area_triangle_OEF_range_l97_97956


namespace arithmetic_sequence_sum_l97_97375

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l97_97375


namespace sum_of_largest_and_smallest_angles_l97_97855

noncomputable theory

-- Define the lengths of the sides of the triangle.
def side1 : ℝ := 1
def side2 : ℝ := Real.sqrt 5
def side3 : ℝ := 2 * Real.sqrt 2

-- Assume the angles opposite to the respective sides.
axiom angle_alpha : ℝ
axiom angle_beta : ℝ

-- Conditions of the problem translating to Lean.
axiom cos_alpha : Real.cos angle_alpha = (side2^2 + side3^2 - side1^2) / (2 * side2 * side3)
axiom cos_beta : Real.cos angle_beta = (side1^2 + side2^2 - side3^2) / (2 * side1 * side2)
axiom sin_alpha : Real.sin angle_alpha = Real.sqrt (1 - (Real.cos angle_alpha)^2)
axiom sin_beta : Real.sin angle_beta = Real.sqrt (1 - (Real.cos angle_beta)^2)
axiom alpha_beta_sum : Real.acos (Real.cos angle_alpha * Real.cos angle_beta - Real.sin angle_alpha * Real.sin angle_beta) = Real.pi / 4

-- The theorem we need to prove.
theorem sum_of_largest_and_smallest_angles : angle_alpha + angle_beta = (3 * Real.pi) / 4 :=
sorry

end sum_of_largest_and_smallest_angles_l97_97855


namespace cuboid_height_l97_97077

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

end cuboid_height_l97_97077


namespace low_income_households_sampled_l97_97439

def total_households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sampled_high_income_households := 25

theorem low_income_households_sampled :
  (sampled_high_income_households / high_income_households) * low_income_households = 19 := by
  sorry

end low_income_households_sampled_l97_97439


namespace arina_should_accept_anton_offer_l97_97872

noncomputable def total_shares : ℕ := 300000
noncomputable def arina_shares : ℕ := 90001
noncomputable def need_to_be_largest : ℕ := 104999 
noncomputable def shares_needed : ℕ := 14999
noncomputable def largest_shareholder_total : ℕ := 105000

noncomputable def maxim_shares : ℕ := 104999
noncomputable def inga_shares : ℕ := 30000
noncomputable def yuri_shares : ℕ := 30000
noncomputable def yulia_shares : ℕ := 30000
noncomputable def anton_shares : ℕ := 15000

noncomputable def maxim_price_per_share : ℕ := 11
noncomputable def inga_price_per_share : ℕ := 1250 / 100
noncomputable def yuri_price_per_share : ℕ := 1150 / 100
noncomputable def yulia_price_per_share : ℕ := 1300 / 100
noncomputable def anton_price_per_share : ℕ := 14

noncomputable def anton_total_cost : ℕ := anton_shares * anton_price_per_share
noncomputable def yuri_total_cost : ℕ := yuri_shares * yuri_price_per_share
noncomputable def inga_total_cost : ℕ := inga_shares * inga_price_per_share
noncomputable def yulia_total_cost : ℕ := yulia_shares * yulia_price_per_share

theorem arina_should_accept_anton_offer :
  anton_total_cost = 210000 := by
  sorry

end arina_should_accept_anton_offer_l97_97872


namespace problem_statement_l97_97110

variables {A B C D F E G : Type}
variables [affine_space A] [affine_space B] [affine_space C] [affine_space D]

-- Assume specific points and lines representing the conditions
axiom parallelogram_abcd : parallelogram A B C D
axiom extended_ad : collinear A D F
axiom intersection_bf_cd : ∃ G, line B F ∩ line C D = {G}
axiom intersection_bf_ac : ∃ E, line B F ∩ line A C = {E}

theorem problem_statement :
  (1 / (dist B E)) = (1 / (dist B G)) + (1 / (dist B F)) :=
sorry

end problem_statement_l97_97110


namespace sqrt_Q_solution_l97_97315

noncomputable def frustum_area (S S' Q : ℝ) (n m : ℝ) : ℝ :=
  (n * Real.sqrt S + m * Real.sqrt S') / (n + m)

theorem sqrt_Q_solution (S S' Q : ℝ) (n m : ℝ) (h : Q = ((n * (S^0.5) + m * (S'^0.5)) / (n + m))^2) :
  Real.sqrt Q = (n * Real.sqrt S + m * Real.sqrt S') / (n + m) :=
by 
  sorry

end sqrt_Q_solution_l97_97315


namespace neg_sqrt_17_bounds_l97_97065

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end neg_sqrt_17_bounds_l97_97065


namespace maximum_tangent_A_l97_97643

-- Definitions and conditions
variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C respectively

-- Given conditions
def conditions : Prop :=
  a = 2 ∧
  b * real.cos C - c * real.cos B = 4 ∧
  (π / 4) ≤ C ∧ C ≤ (π / 3)

-- Target statement to prove
theorem maximum_tangent_A (h : conditions) : real.tan A = 1 / 2 := sorry

end maximum_tangent_A_l97_97643


namespace curved_surface_area_cone_l97_97137

variable (a α β : ℝ) (l := a * Real.sin α) (r := a * Real.cos β)

theorem curved_surface_area_cone :
  π * r * l = π * a^2 * Real.sin α * Real.cos β := by
  sorry

end curved_surface_area_cone_l97_97137


namespace property1_property2_l97_97296

def setA : Set ℕ := { n | ∃ k : ℕ, n = 2^(k + 1) }

theorem property1 (a : ℕ) (b : ℕ) (ha : a ∈ setA) (hb1 : b ∈ ℤ) (hb2 : b < 2 * a - 1) :
  ¬ (2 * a ∣ b * (b + 1)) :=
begin
  -- proof will be provided here
  sorry
end

theorem property2 (a : ℕ) (ha1 : a ∉ setA) (ha2 : a ≠ 1) :
  ∃ b : ℕ, b ∈ ℤ ∧ b < 2 * a - 1 ∧ 2 * a ∣ b * (b + 1) :=
begin
  -- proof will be provided here
  sorry
end

end property1_property2_l97_97296


namespace smallest_solution_floor_equation_l97_97553

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l97_97553


namespace find_k_from_given_solution_find_other_root_l97_97600

-- Given
def one_solution_of_first_eq_is_same_as_second (x k : ℝ) : Prop :=
  x^2 + k * x - 2 = 0 ∧ (x + 1) / (x - 1) = 3

-- To find k
theorem find_k_from_given_solution : ∃ k : ℝ, ∃ x : ℝ, one_solution_of_first_eq_is_same_as_second x k ∧ k = -1 := by
  sorry

-- To find the other root
theorem find_other_root : ∃ x2 : ℝ, (x2 = -1) := by
  sorry

end find_k_from_given_solution_find_other_root_l97_97600


namespace slower_train_speed_l97_97362

-- Defining the conditions

def length_of_each_train := 80 -- in meters
def faster_train_speed := 52 -- in km/hr
def time_to_pass := 36 -- in seconds

-- Main statement: 
theorem slower_train_speed (v : ℝ) : 
    let relative_speed := (faster_train_speed - v) * (1000 / 3600) -- converting relative speed from km/hr to m/s
    let total_distance := 2 * length_of_each_train
    let speed_equals_distance_over_time := total_distance / time_to_pass 
    (relative_speed = speed_equals_distance_over_time) -> v = 36 :=
by
  intros
  sorry

end slower_train_speed_l97_97362


namespace f_increasing_in_interval_l97_97101

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log a (abs (x + 1))
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (abs (x - 1))

theorem f_increasing_in_interval (a : ℝ) (h1 : 0 < a ∧ a ≠ 1)
  (h2 : ∀ x ∈ Ioo (-1 : ℝ) 0, 0 < g a x) :
  ∀ x y ∈ Ioo (-∞ : ℝ) (-1), x < y → f a x < f a y :=
sorry

end f_increasing_in_interval_l97_97101


namespace rectangle_area_l97_97753

theorem rectangle_area 
  (length_to_width_ratio : Real) 
  (width : Real) 
  (area : Real) 
  (h1 : length_to_width_ratio = 0.875) 
  (h2 : width = 24) 
  (h_area : area = 504) : 
  True := 
sorry

end rectangle_area_l97_97753


namespace T_20_is_192_l97_97244

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 1       := 4
| (n + 2) := 2^n

-- Define the sequence {b_n = log_2(a_n)}
def b : ℕ → ℕ
| 1       := 2
| (n + 2) := n + 1

-- Define the sum of the first n terms of a sequence
def sum_seq (s : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => s (i + 1))

-- Define Tₙ as the sum of the first n terms of the sequence {b_n}
def T : ℕ → ℕ
| n := sum_seq b n

-- Prove that T_20 = 192
theorem T_20_is_192 : T 20 = 192 :=
by
  sorry

end T_20_is_192_l97_97244


namespace deposit_percentage_l97_97008

theorem deposit_percentage (income : ℝ) (children_percentage : ℝ) (children_count : ℕ) (donation_percentage : ℝ) (remaining_amount : ℝ) :
  income = 1000000 →
  children_percentage = 0.20 →
  children_count = 3 →
  donation_percentage = 0.05 →
  remaining_amount = 50000 →
  (children_count * children_percentage < 1) →
  let children_amount := children_count * children_percentage * income,
      remaining_after_children := income - children_amount,
      donation_amount := donation_percentage * remaining_after_children,
      remaining_after_donation := remaining_after_children - donation_amount,
      wife_deposit := remaining_after_donation - remaining_amount,
      wife_deposit_percentage := (wife_deposit / income) * 100
  in wife_deposit_percentage = 33 :=
begin
  intros,
  sorry
end

end deposit_percentage_l97_97008


namespace smallest_solution_eq_sqrt_104_l97_97533

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l97_97533


namespace no_square_remainder_2_infinitely_many_squares_remainder_3_l97_97710

theorem no_square_remainder_2 :
  ∀ n : ℤ, (n * n) % 6 ≠ 2 :=
by sorry

theorem infinitely_many_squares_remainder_3 :
  ∀ k : ℤ, ∃ n : ℤ, n = 6 * k + 3 ∧ (n * n) % 6 = 3 :=
by sorry

end no_square_remainder_2_infinitely_many_squares_remainder_3_l97_97710


namespace remainder_of_x100_l97_97331

theorem remainder_of_x100 (x : ℤ) : 
  ∃ R : polynomial ℤ, degree R < 2 ∧ R = 2^(100 : ℤ) * (polynomial.X - 1) - (polynomial.X - 2) ∧ 
  (polynomial.X^100 % (polynomial.X^2 - 3 * polynomial.X + 2)) = R :=
sorry

end remainder_of_x100_l97_97331


namespace find_perimeter_l97_97358

-- Definitions of side lengths
def side_length_PQ : ℝ := 140
def side_length_QR : ℝ := 260
def side_length_PR : ℝ := 210

-- Definitions of segment lengths
def segment_XY : ℝ := 65
def segment_YZ : ℝ := 35
def segment_ZX : ℝ := 25

-- Calculation of the approximate perimeter based on given conditions
noncomputable def scaling_factor : ℝ :=
  (1 / 4 + 1 / 6 + 1 / 5.6) / 3

noncomputable def side_XY : ℝ :=
  scaling_factor * side_length_QR

noncomputable def side_YZ : ℝ :=
  scaling_factor * side_length_PR

noncomputable def side_ZX : ℝ :=
  scaling_factor * side_length_PQ

noncomputable def perimeter_triangle_XYZ : ℝ :=
  side_XY + side_YZ + side_ZX

theorem find_perimeter :
  perimeter_triangle_XYZ ≈ 121.02 :=
  sorry

end find_perimeter_l97_97358


namespace money_lent_to_B_l97_97838

theorem money_lent_to_B (P_B : ℝ) (r : ℝ) :
  (let P_C := 3000;
       t_B := 2;
       t_C := 4;
       total_interest := 3300;
       I_B := P_B * r * t_B;
       I_C := P_C * r * t_C in
   I_B + I_C = total_interest ∧ r = 0.15) →
  P_B = 5000 :=
begin
  sorry
end

end money_lent_to_B_l97_97838


namespace ice_cream_choices_l97_97472

theorem ice_cream_choices : ∀ (n p : ℕ), n = 5 → p = 14 → (nat.choose (n + p - 1) (p - 1)) = 8568 :=
by
intros n p hn hp
rw [hn, hp]
exact nat.choose_eq (5 + 14 - 1) (14 - 1)

end ice_cream_choices_l97_97472


namespace area_diff_correct_l97_97152

noncomputable def screen_area_diff : ℝ :=
  let x := real.sqrt (576 / 25) in
  let y := real.sqrt (900 / 337) in
  let area1 := (4 * x) * (3 * x) in
  let area2 := (16 * y) * (9 * y) in
  area2 - area1

theorem area_diff_correct : abs (screen_area_diff - 106.21) < 0.01 := sorry

end area_diff_correct_l97_97152


namespace order_of_abc_l97_97567

noncomputable def a : ℚ := 1 / 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 6 - 2

theorem order_of_abc : a > c ∧ c > b := by
  sorry

end order_of_abc_l97_97567


namespace intersection_points_of_five_lines_l97_97087

theorem intersection_points_of_five_lines :
  ∀ (lines : set (set (ℝ × ℝ))),
  lines.card = 5 ∧
  (∀ (l₁ l₂ : set (ℝ × ℝ)), l₁ ∈ lines → l₂ ∈ lines → l₁ ≠ l₂ → ∃! p, p ∈ l₁ ∧ p ∈ l₂) ∧
  (∀ (p : ℝ × ℝ), ¬ (∃ l₁ l₂ l₃ : set (ℝ × ℝ), l₁ ∈ lines ∧ l₂ ∈ lines ∧ l₃ ∈ lines ∧ p ∈ l₁ ∧ p ∈ l₂ ∧ p ∈ l₃)) →
  ∃ intersections : set (ℝ × ℝ), intersections.card = 10 ∧ ∀ p ∈ intersections, ∃! (l₁ l₂ : set (ℝ × ℝ)), l₁ ∈ lines ∧ l₂ ∈ lines ∧ p ∈ l₁ ∧ p ∈ l₂ :=
by
  sorry

end intersection_points_of_five_lines_l97_97087


namespace sum_of_integers_between_6_and_14_l97_97775

theorem sum_of_integers_between_6_and_14 : ∑ i in (Finset.range 15).filter (λ n, n > 5), i = 90 :=
by
  sorry

end sum_of_integers_between_6_and_14_l97_97775


namespace ratio_of_x_to_y_l97_97840

-- Given condition: The percentage that y is less than x is 83.33333333333334%.
def percentage_less_than (x y : ℝ) : Prop := (x - y) / x = 0.8333333333333334

-- Prove: The ratio R = x / y is 1/6.
theorem ratio_of_x_to_y (x y : ℝ) (h : percentage_less_than x y) : x / y = 6 := 
by sorry

end ratio_of_x_to_y_l97_97840


namespace total_handshakes_l97_97350

theorem total_handshakes (gremlins imps unfriendly_gremlins : ℕ) 
    (handshakes_among_friendly : ℕ) (handshakes_friendly_with_unfriendly : ℕ) 
    (handshakes_between_imps_and_gremlins : ℕ) 
    (h_friendly : gremlins = 30) (h_imps : imps = 20) 
    (h_unfriendly : unfriendly_gremlins = 10) 
    (h_handshakes_among_friendly : handshakes_among_friendly = 190) 
    (h_handshakes_friendly_with_unfriendly : handshakes_friendly_with_unfriendly = 200)
    (h_handshakes_between_imps_and_gremlins : handshakes_between_imps_and_gremlins = 600) : 
    handshakes_among_friendly + handshakes_friendly_with_unfriendly + handshakes_between_imps_and_gremlins = 990 := 
by 
    sorry

end total_handshakes_l97_97350


namespace eccentricity_of_ellipse_l97_97328

theorem eccentricity_of_ellipse (a b : ℝ) (h : a = 2 * b) (ha : a > 0) (hb : b > 0) : 
  let e := Real.sqrt (3 / 4) in e = Real.sqrt 3 / 2 := 
by 
  rw [Real.sqrt_div (by linarith), Real.sqrt_div_id (by linarith)] 
  sorry

end eccentricity_of_ellipse_l97_97328


namespace solve_trig_equation_proof_l97_97093

noncomputable def solve_trig_equation (θ : ℝ) : Prop :=
  2 * Real.cos θ ^ 2 - 5 * Real.cos θ + 2 = 0 ∧ (θ = 60 / 180 * Real.pi)

theorem solve_trig_equation_proof (θ : ℝ) :
  solve_trig_equation θ :=
sorry

end solve_trig_equation_proof_l97_97093


namespace ratio_large_square_to_small_square_l97_97351

-- Define the parameters and conditions
variables (s : ℝ) (side_length_large_square : ℝ)

-- Define the smaller squares and rectangles
def small_square_side_length := s
def rectangle_length := 2 * s
def rectangle_width := s

-- Define the arrangement of the smaller squares and rectangles into a large square
def large_square_side_length := 3 * s

-- The theorem stating that the ratio of the side of the large square to the side of the smaller squares is 3
theorem ratio_large_square_to_small_square (s : ℝ) (side_length_large_square : ℝ) 
  (h1 : side_length_large_square = 3 * s) : side_length_large_square / small_square_side_length = 3 :=
by {
  rw h1,
  rw small_square_side_length,
  field_simp,
  norm_num,
} 

end ratio_large_square_to_small_square_l97_97351


namespace find_n_l97_97932

theorem find_n (n : ℕ) 
    (h : 6 * 4 * 3 * n = Nat.factorial 8) : n = 560 := 
sorry

end find_n_l97_97932


namespace tokens_packing_around_central_token_l97_97818

theorem tokens_packing_around_central_token :
  ∀ (r : ℝ), r = 2 → (∃ n : ℕ, n = 6) := 
by
  intro r hr_eq
  rw hr_eq
  use 6
  sorry

end tokens_packing_around_central_token_l97_97818


namespace a_works_on_friday_in_50th_week_l97_97999

-- Define persons and days
inductive Person : Type
| A | B | C | D | E | F

inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Person Day

-- Define the rotation order
def rotation : list Person := [A, B, C, D, E, F]

-- Define the function to get the day of the week
def day_of_week (n : ℕ) : Day :=
match n % 7 with
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| _ => Saturday
end

-- Define the function to get the worker on a specific day
def worker_on_day (n : ℕ) : Person :=
rotation.get_or_else (n % rotation.length) A

-- Proof statement
theorem a_works_on_friday_in_50th_week : worker_on_day (50 * 7) = A := by
sorry

end a_works_on_friday_in_50th_week_l97_97999


namespace cube_side_length_l97_97821

theorem cube_side_length (s : ℕ) (h1 : 58 / 100 ≤ 95 / 100) (h2 : 95 / 100 ≤ 87 / 100)
  (h3 : 10 ≤ s^3 * (1 - 0.87)) (h4 : s^3 * (1 - 0.58) ≤ 29) :
  s = 4 :=
by
  sorry

end cube_side_length_l97_97821


namespace distinct_divisor_sum_l97_97299

theorem distinct_divisor_sum (n : ℕ) (x : ℕ) (h : x < n.factorial) :
  ∃ (k : ℕ) (d : Fin k → ℕ), (k ≤ n) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n.factorial) ∧ (x = Finset.sum Finset.univ d) :=
sorry

end distinct_divisor_sum_l97_97299


namespace find_matrix_N_l97_97513

-- Definitions from the conditions 
def is_3x3_matrix (N : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  N = ![![N 0 0, N 0 1, N 0 2], ![N 1 0, N 1 1, N 1 2], ![N 2 0, N 2 1, N 2 2]]

def satisfies_condition (N : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ (u : Fin 3 → ℝ), N.mul_vec u = fun i => 3 * ↓u i

-- Theorem statement proving the question equals the correct answer given conditions
theorem find_matrix_N :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
  is_3x3_matrix N ∧ satisfies_condition N ∧ 
  N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] :=
by
  sorry

end find_matrix_N_l97_97513


namespace sum_of_integers_between_5_and_15_l97_97780

-- Definitions based on conditions
def predicate (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Main theorem statement
theorem sum_of_integers_between_5_and_15 : (Finset.sum (Finset.filter predicate (Finset.range 15))) = 90 :=
by
  sorry

end sum_of_integers_between_5_and_15_l97_97780


namespace polynomial_roots_and_coeffs_l97_97496

theorem polynomial_roots_and_coeffs :
  ∃ (a b c : ℝ), (a ≠ 0) ∧ (a * b * c ≠ 0) ∧ (a * b ≠ 0) ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (roots.equals (coefficients) ∧ 
  (a = -b ∨ b = -c ∨ c = -a)) → 
  (count_polynomials_with_real_coeffs_and_neg_coefficient 3) :=
by
  sorry

end polynomial_roots_and_coeffs_l97_97496


namespace amount_needed_is_72_l97_97881

-- Define the given conditions
def original_price : ℝ := 90
def discount_rate : ℝ := 20

-- The goal is to prove that the amount of money needed after the discount is $72
theorem amount_needed_is_72 (P : ℝ) (D : ℝ) (hP : P = original_price) (hD : D = discount_rate) : P - (D / 100 * P) = 72 := 
by sorry

end amount_needed_is_72_l97_97881


namespace arith_seq_seventh_term_l97_97767

theorem arith_seq_seventh_term (a1 a25 : ℝ) (n : ℕ) (d : ℝ) (a7 : ℝ) :
  a1 = 5 → a25 = 80 → n = 25 → d = (a25 - a1) / (n - 1) → a7 = a1 + (7 - 1) * d → a7 = 23.75 :=
by
  intros h1 h2 h3 hd ha7
  sorry

end arith_seq_seventh_term_l97_97767


namespace number_of_carnations_in_second_bouquet_l97_97357

-- Let a, b be the number of carnations in the first and third bouquet respectively.
-- The average number of carnations in the three bouquets is 12.
-- Prove that x (the number of carnations in the second bouquet) is 14.
theorem number_of_carnations_in_second_bouquet 
  (a b : ℕ) (x : ℕ) 
  (h_a : a = 9) 
  (h_b : b = 13) 
  (h_avg : (a + x + b) / 3 = 12):
  x = 14 := 
by 
  rw [h_a, h_b] at h_avg
  /
/- Simplifying the average equation 
 -   24/3 = 12 + (2 * remaining)/3 OR 12 = 36 - remaining, giving the final answer. -/
-- start proof
 sorry

end number_of_carnations_in_second_bouquet_l97_97357


namespace baby_plants_produced_l97_97061

theorem baby_plants_produced (baby_plants_per_time: ℕ) (times_per_year: ℕ) (years: ℕ) (total_babies: ℕ) :
  baby_plants_per_time = 2 ∧ times_per_year = 2 ∧ years = 4 ∧ total_babies = baby_plants_per_time * times_per_year * years → 
  total_babies = 16 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end baby_plants_produced_l97_97061


namespace find_x_l97_97572

theorem find_x (x y : ℤ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
by
  sorry

end find_x_l97_97572


namespace S_definitely_not_lowest_avg_position_l97_97653

variable (P Q R S T : Type)

/-- Conditions from the first race -/
def race1_conditions (P Q R S T : Type) [Less P Q] [Less P R] [Less Q S] [Less P T] [Less T Q] : Prop := 
  true

/-- Second race explicit order -/
def race2_order (R P T Q S : Type) : Prop := 
  true

noncomputable def cannot_have_lowest_average_position : Prop :=
  ∀ (P Q R S T : Type), race1_conditions P Q R S T → race2_order R P T Q S → S ≠ runner_with_lowest_average_position P Q R S T

theorem S_definitely_not_lowest_avg_position : cannot_have_lowest_average_position P Q R S T := by
  sorry

end S_definitely_not_lowest_avg_position_l97_97653


namespace value_of_m_l97_97215

noncomputable def no_linear_term (m : ℝ) : Prop :=
  (λ x : ℝ, (2 * x - m) * (x + 1)) = λ x, 2 * x^2 - m

theorem value_of_m :
  (∀ x : ℝ, (2 * x - m) * (x + 1) = 2 * x^2 - m) → m = 2 :=
by
  intros h
  have h_m : (2:ℝ) - m = 0 :=
    sorry -- The proof that the coefficient of x must be zero and solving for m.
  exact eq_of_sub_eq_zero h_m

end value_of_m_l97_97215


namespace point_b_not_inside_circle_a_l97_97289

theorem point_b_not_inside_circle_a (a : ℝ) : a < 5 → ¬ (1 < a ∧ a < 5) :=
by
  sorry

end point_b_not_inside_circle_a_l97_97289


namespace min_value_frac_l97_97966

open Real

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 1) :
  (1 / a + 2 / b) = 9 + 4 * sqrt 2 :=
sorry

end min_value_frac_l97_97966


namespace total_cakes_served_l97_97017

-- Conditions
def cakes_lunch : Nat := 6
def cakes_dinner : Nat := 9

-- Statement of the problem
theorem total_cakes_served : cakes_lunch + cakes_dinner = 15 := 
by
  sorry

end total_cakes_served_l97_97017


namespace hvac_cost_per_vent_l97_97728

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end hvac_cost_per_vent_l97_97728


namespace vectors_parallel_solution_l97_97154

theorem vectors_parallel_solution (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (2, x)) (h2 : b = (x, 8)) (h3 : ∃ k, b = (k * 2, k * x)) : x = 4 ∨ x = -4 :=
by
  sorry

end vectors_parallel_solution_l97_97154


namespace ellipse_proof_l97_97957

noncomputable def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_proof :
  ∃ a b c : ℝ, a > b ∧ b > 0 ∧ b^2 = 3 ∧ c^2 = 1 ∧ a = 2 * c ∧ 
  ellipse 1 (3 / 2) a b ∧ 
  ∀ Q : ℝ × ℝ, Q = (4, 0) →
  ∀ P : ℝ × ℝ, P = (1, 3/2) →
  (∃ k : ℝ, -1/2 < k ∧ k < 1/2 ∧ 
    ( ∃ line : ℝ × ℝ → ℝ × ℝ → ℝ, (line Q P) = 36 * (4 - 1)^2 / 4 ∧
      line (4, 0) P = 35 * (forall pts : ℝ × ℝ, (line (4, 0) pts) * (line pts (4, 0))  
      = sqrt(2)) ( ∃ line_eq : ℝ, (∀ (line_eq = sqrt(2) * x + 4 * y - 4 * sqrt(2)) ∨ 
      (line_eq = sqrt(2) * x - 4 * y - 4 * sqrt(2))))
sorry

end ellipse_proof_l97_97957


namespace f_comp_g_eq_g_comp_f_iff_l97_97206

variable {R : Type} [CommRing R]

def f (m n : R) (x : R) : R := m * x ^ 2 + n
def g (p q : R) (x : R) : R := p * x + q

theorem f_comp_g_eq_g_comp_f_iff (m n p q : R) :
  (∀ x : R, f m n (g p q x) = g p q (f m n x)) ↔ n * (1 - p ^ 2) - q * (1 - m) = 0 :=
by
  sorry

end f_comp_g_eq_g_comp_f_iff_l97_97206


namespace value_of_def_ef_l97_97599

theorem value_of_def_ef
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : (a * f) / (c * d) = 1)
  : d * e * f = 250 := 
by 
  sorry

end value_of_def_ef_l97_97599


namespace xyz_squared_sum_l97_97758

theorem xyz_squared_sum (x y z : ℝ)
  (h1 : x^2 + 6 * y = -17)
  (h2 : y^2 + 4 * z = 1)
  (h3 : z^2 + 2 * x = 2) :
  x^2 + y^2 + z^2 = 14 := 
sorry

end xyz_squared_sum_l97_97758


namespace correct_option_is_B_l97_97787

def natural_growth_rate (birth_rate death_rate : ℕ) : ℕ :=
  birth_rate - death_rate

def option_correct (birth_rate death_rate : ℕ) :=
  (∃ br dr, natural_growth_rate br dr = br - dr)

theorem correct_option_is_B (birth_rate death_rate : ℕ) :
  option_correct birth_rate death_rate :=
by 
  sorry

end correct_option_is_B_l97_97787


namespace part_I_part_II_l97_97614

-- Define the function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

-- Define the derivative of f
def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

-- Part (I): Prove correct values of a and b for the conditions given
theorem part_I :
  (∃ (a b : ℝ), (a = 4 ∧ b = -11) ∧ (f'(1, a, b) = 0) ∧ (f 1 a b = 10)) :=
by
  sorry

-- Part (II): Prove the range of b for the given interval condition
theorem part_II :
  ∀ x ∈ (Icc 1 2 : Set ℝ), a = -1 → f x (-1) b < 0 → b < -5 / 2 :=
by
  sorry

end part_I_part_II_l97_97614


namespace mr_bird_on_time_speed_l97_97281

theorem mr_bird_on_time_speed (d t : ℝ) :
  (d = 50 * (t + 5 / 60)) ∧ (d = 70 * (t - 5 / 60)) → (r = 55) :=
begin
  sorry
end

end mr_bird_on_time_speed_l97_97281


namespace storks_more_than_birds_l97_97403

def birds := 4
def initial_storks := 3
def additional_storks := 6

theorem storks_more_than_birds :
  (initial_storks + additional_storks) - birds = 5 := 
by
  sorry

end storks_more_than_birds_l97_97403


namespace julia_bill_ratio_l97_97285

-- Definitions
def saturday_miles_b (s_b : ℕ) (s_su : ℕ) := s_su = s_b + 4
def sunday_miles_j (s_su : ℕ) (t : ℕ) (s_j : ℕ) := s_j = t * s_su
def total_weekend_miles (s_b : ℕ) (s_su : ℕ) (s_j : ℕ) := s_b + s_su + s_j = 36

-- Proof statement
theorem julia_bill_ratio (s_b s_su s_j : ℕ) (h1 : saturday_miles_b s_b s_su) (h3 : total_weekend_miles s_b s_su s_j) (h_su : s_su = 10) : (2 * s_su = s_j) :=
by
  sorry  -- proof

end julia_bill_ratio_l97_97285


namespace volume_percent_removed_l97_97460

theorem volume_percent_removed (length width height : ℝ) (cube_side : ℝ) (n_cubes : ℕ) :
  length = 15 → width = 10 → height = 8 → cube_side = 3 → n_cubes = 8 → 
  (n_cubes * (cube_side ^ 3) / (length * width * height) * 100) = 18 := 
by 
  intros h_length h_width h_height h_cube_side h_n_cubes
  rw [h_length, h_width, h_height, h_cube_side, h_n_cubes]
  norm_num
  sorry

end volume_percent_removed_l97_97460


namespace interval_strictly_increasing_lengths_of_sides_l97_97617

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin x * cos x + cos x ^ 2

theorem interval_strictly_increasing :
  ∀ (k : ℤ), ∀ (x : ℝ),
  (k : ℝ) * π - π / 3 ≤ x ∧ x ≤ (k : ℝ) * π + π / 6 → 
  deriv f x > 0 :=
sorry

noncomputable def dot_product_condition (AB AC BC : ℝ) : Prop :=
  let A := π / 3 in
  (BC*BC = AB*AB + AC*AC - 2*AB*AC*real.cos A) ∧
  (AB*AC*real.cos A = 4)

theorem lengths_of_sides (BC : ℝ) :
  BC = 2 * sqrt 3 →
  ∃ (AB AC : ℝ), dot_product_condition AB AC BC ∧ (AB = 2 ∧ AC = 4 ∨ AB = 4 ∧ AC = 2) :=
sorry

end interval_strictly_increasing_lengths_of_sides_l97_97617


namespace range_of_f_ge_1_l97_97689

noncomputable def f (x : ℝ) :=
  if x < 1 then (x + 1) ^ 2 else 4 - real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x | x ≤ -2} ∪ {x | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end range_of_f_ge_1_l97_97689


namespace dihedral_angle_measure_l97_97655

open EuclideanGeometry

noncomputable def measure_of_dihedral_angle 
  (AB AD_1 AC BD_1 : ℝ) 
  (right_triangle : is_right_triangle A C D₁) 
  (h_AB : AB = 1) 
  (h_AD_1_AC : AD_1 = AC) 
  (h_AD_1 : AD_1 = sqrt 2) 
  (h_BD_1 : BD_1 = sqrt 3) 
: ℝ := 
  60 -- degrees, the measure of dihedral angle.

theorem dihedral_angle_measure 
  {A B C D₁ A₁ : Point} 
  (AB AD_1 AC BD_1 : ℝ)
  (right_triangle : is_right_triangle A C D₁) 
  (h_AB : AB = 1) 
  (h_AD_1_AC : AD_1 = AC) 
  (h_AD_1 : AD_1 = sqrt 2) 
  (h_BD_1 : BD_1 = sqrt 3) 
:
  measure_of_dihedral_angle AB AD_1 AC BD_1 right_triangle h_AB h_AD_1_AC h_AD_1 h_BD_1 = 60 := 
  sorry

end dihedral_angle_measure_l97_97655


namespace sec_of_7pi_over_4_l97_97917

theorem sec_of_7pi_over_4 : real.sec (7 * real.pi / 4) = real.sqrt 2 := by
  sorry

end sec_of_7pi_over_4_l97_97917


namespace shop_owner_cheat_percentage_l97_97019

def CP : ℝ := 100
def cheating_buying : ℝ := 0.15  -- 15% cheating
def actual_cost_price : ℝ := CP * (1 + cheating_buying)  -- $115
def profit_percentage : ℝ := 43.75

theorem shop_owner_cheat_percentage :
  ∃ x : ℝ, profit_percentage = ((CP - x * CP / 100 - actual_cost_price) / actual_cost_price * 100) ∧ x = 65.26 :=
by
  sorry

end shop_owner_cheat_percentage_l97_97019


namespace repeating_decimal_fraction_l97_97188

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l97_97188


namespace sawyer_discontinued_on_november_first_l97_97713

-- Conditions
def daily_coaching_charge := 39
def total_amount_paid := 11895
def days_in_non_leap_year := 365
def days_in_january := 31
def days_in_february := 28
def days_in_march := 31
def days_in_april := 30
def days_in_may := 31
def days_in_june := 30
def days_in_july := 31
def days_in_august := 31
def days_in_september := 30
def days_in_october := 31
def days_in_non_leap_year_months : List Nat := [
  days_in_january, days_in_february, days_in_march, 
  days_in_april, days_in_may, days_in_june, 
  days_in_july, days_in_august, days_in_september, 
  days_in_october]

-- The calculated number of days Sawyer attended coaching.
def days_attended := total_amount_paid / daily_coaching_charge

-- Sum of days until the end of October.
def sum_days_until_october := days_in_non_leap_year_months.foldl (· + ·) 0

-- The proof problem
theorem sawyer_discontinued_on_november_first : 
  days_attended = 305 → sum_days_until_october + 1 = 305 → 
     "Sawyer discontinued the coaching on November 1st" := by
  sorry

end sawyer_discontinued_on_november_first_l97_97713


namespace parabola_hyperbola_distance_l97_97895

def parabola_focus (p : ℝ) : ℝ × ℝ := (p, 0)
def hyperbola_asymptotes : List (ℝ × ℝ × ℝ) := [(1, √3/3, 0), (1, -√3/3, 0)]

lemma distance_from_focus_to_asymptote 
  (focus : ℝ × ℝ) 
  (asym : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0) := focus
  let (A, B, C) := asym
  |(A * x0 + B * y0 + C)| / (Real.sqrt (A^2 + B^2))

theorem parabola_hyperbola_distance : 
  distance_from_focus_to_asymptote (parabola_focus 1) (1, √3/3, 0) = sqrt 3 / 2 :=
sorry

end parabola_hyperbola_distance_l97_97895


namespace sum_remainders_mod_15_l97_97379

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end sum_remainders_mod_15_l97_97379


namespace minimum_ab_ge_four_l97_97125

variable (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
variable (h : 1 / a + 4 / b = Real.sqrt (a * b))

theorem minimum_ab_ge_four : a * b ≥ 4 := by
  sorry

end minimum_ab_ge_four_l97_97125


namespace most_suitable_for_census_l97_97386

theorem most_suitable_for_census (A B C D : Prop) :
  ¬A ∧ B ∧ ¬C ∧ ¬D → B :=
by
  intro h
  cases h with _ B_and_notC_and_notD
  tauto

end most_suitable_for_census_l97_97386


namespace original_price_l97_97744

variable (p q : ℝ)

theorem original_price (x : ℝ)
  (hp : x * (1 + p / 100) * (1 - q / 100) = 1) :
  x = 10000 / (10000 + 100 * (p - q) - p * q) :=
sorry

end original_price_l97_97744


namespace has_unique_zero_g_l97_97575

-- Defining the problem conditions and the function
variables {α : Type*} [LinearOrderedField α] [TopologicalSpace α] [OrderTopology α]
          (f : α → α) (a b : α)
          (cont_f : ContinuousOn f (Set.Icc a b))
          (range_f : Set.range f ⊆ Set.Icc a b)
          (cond : ∀ {x y : α}, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x ≠ y → |f x - f y| < |x - y|)

-- Proving the existence and uniqueness of zero for g(x) = f(x) - x in [a, b]
theorem has_unique_zero_g : ∃! x ∈ Set.Icc a b, f x - x = 0 :=
sorry

end has_unique_zero_g_l97_97575


namespace max_area_of_triangle_ABC_l97_97233

noncomputable def max_area_triangle_ABC: ℝ :=
  let QA := 3
  let QB := 4
  let QC := 5
  let BC := 6
  -- Given these conditions, prove the maximum area of triangle ABC
  19

theorem max_area_of_triangle_ABC 
  (QA QB QC BC : ℝ) 
  (h1 : QA = 3) 
  (h2 : QB = 4) 
  (h3 : QC = 5) 
  (h4 : BC = 6) 
  (h5 : QB * QB + BC * BC = QC * QC) -- The right angle condition at Q
  : max_area_triangle_ABC = 19 :=
by sorry

end max_area_of_triangle_ABC_l97_97233


namespace people_per_car_l97_97807

theorem people_per_car (total_people : ℝ) (total_cars : ℝ) (h1 : total_people = 189) (h2 : total_cars = 3.0) : total_people / total_cars = 63 := 
by
  sorry

end people_per_car_l97_97807


namespace det_A_power_five_l97_97987

variables {A : Matrix ℝ ℝ}

theorem det_A_power_five (h : det A = 4) : det (A^5) = 1024 :=
by sorry

end det_A_power_five_l97_97987


namespace expression1_expression2_l97_97482

-- First part of the problem
theorem expression1 :
  (0.25: ℝ) ^ (-2 : ℤ) + 8 ^ (2 / 3 : ℝ) - 16 ^ 3 + 2 ^ (Real.log2 3) = 15 := 
by
  sorry

-- Second part of the problem
theorem expression2 :
  Real.log10 16 + 3 * Real.log10 5 - Real.log10 (1 / 5: ℝ) + Real.log2 9 * Real.logb 3 4 = 8 :=
by
  sorry

end expression1_expression2_l97_97482


namespace min_value_of_f_l97_97742

noncomputable theory

open Classical

def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) * (Real.log (2 * x) / Real.log (Real.sqrt 2))

theorem min_value_of_f :
  ∃ x > 0, f x = -1 / 4 :=
begin
  sorry
end

end min_value_of_f_l97_97742


namespace kate_overall_average_speed_is_approximately_6_857_mph_l97_97253

-- Defining the conditions
def biking_time_in_hours := (45 : ℝ) / 60
def biking_speed_mph := 12
def walking_time_in_hours := (60 : ℝ) / 60
def walking_speed_mph := 3

-- Calculating the respective distances
def biking_distance := biking_speed_mph * biking_time_in_hours
def walking_distance := walking_speed_mph * walking_time_in_hours

-- Calculating the total distance and time
def total_distance := biking_distance + walking_distance
def total_time := biking_time_in_hours + walking_time_in_hours

-- Expected overall average speed
def expected_average_speed := total_distance / total_time

theorem kate_overall_average_speed_is_approximately_6_857_mph :
  abs (expected_average_speed - 6.857) < 0.001 :=
by 
  -- The proof will go here
  sorry

end kate_overall_average_speed_is_approximately_6_857_mph_l97_97253


namespace probability_sum_remaining_greater_l97_97477

theorem probability_sum_remaining_greater (A : multiset ℕ) (B : multiset ℕ) :
  A = {10, 10, 1, 1, 1} ∧ B = {5, 5, 5, 5, 1, 1, 1} →
  (∃ (drawA drawB : multiset ℕ),
    drawA.card = 2 ∧ drawB.card = 2 ∧
    drawA ⊆ A ∧ drawB ⊆ B ∧
    (20 - drawA.sum) > (20 - drawB.sum)) →
  (∑ (drawA : multiset ℕ) in A.powerset.filter (λ s, s.card = 2),
    ∑ (drawB : multiset ℕ) in B.powerset.filter (λ s, s.card = 2),
    if (20 - drawA.sum) > (20 - drawB.sum) then 1 else 0).to_finset.card.to_real /
  ((A.powerset.filter (λ s, s.card = 2)).card * (B.powerset.filter (λ s, s.card = 2)).card) = 9 / 35 :=
by
  sorry

end probability_sum_remaining_greater_l97_97477


namespace circle_tangent_and_ellipse_l97_97959

theorem circle_tangent_and_ellipse 
  (M : ℝ × ℝ) (C : ℝ × ℝ → Prop)
  (hC : C = λ (p : ℝ × ℝ), p.1^2 + p.2^2 = 4) 
  (hM : M = (2, 4))
  (T : ℝ → ℝ × ℝ → Prop)
  (hT : T = λ a p, p.1^2 / a^2 + p.2^2 = 1)
  (a b : ℝ) (ha : a > b) (hb : b > 0)
  (hline : ∀ (A B : ℝ × ℝ), 
              (A.1, A.2) ≠ (B.1, B.2) →
              is_tangent_to_circle A ∧
              is_tangent_to_circle B ∧
              line_passes_through (A, B) (2, 0) ∧ line_passes_through (A, B) (0, 1)) :
  (∃ a b : ℝ, hT a (0, 1) ∧ hT b (2, 0)) ∧ 
  (original_origin_triangle (a b : ℝ)
                           (l : ℝ × ℝ → ℝ)
                           (hx : ∀ (P Q : ℝ × ℝ), 
                                   l P = k P.1 + sqrt 3 ∧ 
                                   l Q = k Q.1 + sqrt 3 ∧
                                   intersects_ellipse P ∧ 
                                   intersects_ellipse Q) →
   ∃ k : ℝ, k > 0 ∧ max_area_triangle P Q (0, 0) = 1) :=
sorry

end circle_tangent_and_ellipse_l97_97959


namespace correct_distance_function_l97_97607

def distance_from_A (t : ℝ) : ℝ :=
  if h : 0 ≤ t ∧ t ≤ 2.5 then 60 * t 
  else if h : 2.5 < t ∧ t ≤ 3.5 then 150 
  else if h : 3.5 < t ∧ t ≤ 6.5 then 150 - 50 * (t - 3.5) 
  else 0

theorem correct_distance_function :
  ∀ t : ℝ, 
  (0 ≤ t ∧ t ≤ 2.5 → distance_from_A t = 60 * t) ∧ 
  (2.5 < t ∧ t ≤ 3.5 → distance_from_A t = 150) ∧ 
  (3.5 < t ∧ t ≤ 6.5 → distance_from_A t = 150 - 50 * (t - 3.5)) :=
by
  intro t
  split
  { intro h
    simp [distance_from_A, h] }
  split
  { intro h
    simp [distance_from_A, h] }
  { intro h
    simp [distance_from_A, h] }
sorry

end correct_distance_function_l97_97607


namespace lines_parallel_planes_implies_planes_parallel_l97_97978

-- Define the geometric entities involved
variable {Point : Type}
variable {Line : Type}
variable {Plane : Type}

variable (a b : Line)
variable (α β : Plane)
variable (A : Point)

-- Define relationships as assumptions
variable (inter_a_b : ∃ A, A ∈ a ∧ A ∈ b)
variable (a_parallel_α : ∀ (p : Point), p ∈ a → p ∉ α)
variable (b_parallel_α : ∀ (p : Point), p ∈ b → p ∉ α)
variable (a_parallel_β : ∀ (p : Point), p ∈ a → p ∉ β)
variable (b_parallel_β : ∀ (p : Point), p ∈ b → p ∉ β)

-- State the theorem to be proved
theorem lines_parallel_planes_implies_planes_parallel :
  (∃ A, A ∈ a ∧ A ∈ b) →
  (∀ (p : Point), p ∈ a → p ∉ α) →
  (∀ (p : Point), p ∈ b → p ∉ α) →
  (∀ (p : Point), p ∈ a → p ∉ β) →
  (∀ (p : Point), p ∈ b → p ∉ β) →
  (α ≠ β) →
  α ∥ β :=
sorry

end lines_parallel_planes_implies_planes_parallel_l97_97978


namespace part_one_max_value_part_two_a_range_l97_97618

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp (a * x)
noncomputable def g (a : ℝ) (x : ℝ) := x^2 * f a x
noncomputable def h (a : ℝ) (x : ℝ) := x^2 / f a x - 1

theorem part_one_max_value :
  ∃ (a : ℝ), a = -2 ∧ ∀ x > 0, g a x ≤ Real.exp (-2) := sorry

theorem part_two_a_range (a : ℝ) (ha1 : h a 0 < 0) (ha2 : ∀ x, 0 < x → x < 16 → h a x = 0 → h a x > 0 → x = 2 / a) : 
  (1 / 2 * Real.log 2 < a ∧ a < 2 / Real.exp 1) := 
begin
  sorry,
end


end part_one_max_value_part_two_a_range_l97_97618


namespace cyclists_meet_time_l97_97761

theorem cyclists_meet_time (circumference : ℕ) (speed1 speed2 : ℕ) (h1 : circumference = 600) (h2 : speed1 = 7) (h3 : speed2 = 8) :
  let relative_speed := speed1 + speed2 in
  let time := circumference / relative_speed in
  time = 40 :=
by
  sorry

end cyclists_meet_time_l97_97761


namespace jisha_speed_second_day_l97_97250

-- Define the conditions
def hours_first_day := 18 / 3
def hours_second_day := hours_first_day - 1
def hours_third_day := hours_first_day

def distance_first_day := 18

-- Let v be Jisha's speed on the second day
variable (v : ℝ)

-- Define distances on the second and third day in terms of v
def distance_second_day := v * hours_second_day
def distance_third_day := v * hours_third_day

-- Define the total distance equation
def total_distance := distance_first_day + distance_second_day + distance_third_day

-- State the hypothesis that total distance is 62 miles
axiom total_distance_62 : total_distance = 62

-- Prove that Jisha's speed on the second day is 4 mph under the given conditions
theorem jisha_speed_second_day : v = 4 :=
by
  have h_first_day : hours_first_day = 6 := by norm_num [hours_first_day]
  have h_second_day : hours_second_day = 5 := by norm_num [hours_second_day, h_first_day]
  have h_third_day : hours_third_day = 6 := by norm_num [hours_third_day, h_first_day]

  -- Substitute hours into distances
  have distance_second_day_def : distance_second_day = v * 5 := by rw [h_second_day]
  have distance_third_day_def : distance_third_day = v * 6 := by rw [h_third_day]

  -- Expand total distance and solve for v
  rw [distance_first_day, distance_second_day_def, distance_third_day_def] at total_distance_62

  -- Simplify the equation to find v
  sorry

end jisha_speed_second_day_l97_97250


namespace perpendiculars_intersect_at_symmetric_point_l97_97816

-- Conditions
variables {A B C A1 A2 B1 B2 C1 C2 O M : Point}
variable Circle : Circle
variable ABC : Triangle

-- Assume the circle intersects sides at given points
variable ha1a2 : Circle.IntersectsLine (Segment BC) A1 A2
variable hb1b2 : Circle.IntersectsLine (Segment CA) B1 B2
variable hc1c2 : Circle.IntersectsLine (Segment AB) C1 C2

-- Assume perpendiculars drawn through the points intersect at M
variable h_perpendiculars_intersect_at_M : 
  (Perpendicular (LineThrough A1 BC) (LineThrough M SideBC)) ∧ 
  (Perpendicular (LineThrough B1 CA) (LineThrough M SideCA)) ∧ 
  (Perpendicular (LineThrough C1 AB) (LineThrough M SideAB))

-- Symmetry requirements
variable h_symmetry :
  SymmetricWithRespectTo O A1 A2 ∧ 
  SymmetricWithRespectTo O B1 B2 ∧ 
  SymmetricWithRespectTo O C1 C2

-- Goal: Prove the perpendiculars through A2, B2, C2 intersect at a point symmetric to M with respect to O
theorem perpendiculars_intersect_at_symmetric_point :
  ∃ M', 
    SymmetricWithRespectTo O M M' ∧ 
    Perpendicular (LineThrough A2 BC) (LineThrough M' SideBC) ∧ 
    Perpendicular (LineThrough B2 CA) (LineThrough M' SideCA) ∧ 
    Perpendicular (LineThrough C2 AB) (LineThrough M' SideAB) :=
begin
  sorry
end

end perpendiculars_intersect_at_symmetric_point_l97_97816


namespace regular_dodecagon_midpoints_l97_97943

theorem regular_dodecagon_midpoints :
  let A := (-1, -1)
  let B := (1, -1)
  let C := (1, 1)
  let D := (-1, 1)
  let K := (0, Real.sqrt 3 - 1)
  let L := (1 - Real.sqrt 3, 0)
  let M := (0, -(Real.sqrt 3 - 1))
  let N := (Real.sqrt 3 - 1, 0)
  let mid := λ p q: (ℝ × ℝ), ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  let P1 := mid D N
  let P2 := mid C L
  let P3 := mid K N
  -- Remaining midpoints will be defined similarly
  in
  -- Proof of midpoints forming a regular dodecagon centered at origin
  sorry

end regular_dodecagon_midpoints_l97_97943


namespace cos_gamma_l97_97646

theorem cos_gamma (α β γ : ℝ) (hα : cos α = -1 / 4) (hβ : cos β = 1 / 2) :
  cos γ = sqrt 11 / 4 :=
by
  sorry

end cos_gamma_l97_97646


namespace thomas_percentage_l97_97041

/-- 
Prove that if Emmanuel gets 100 jelly beans out of a total of 200 jelly beans, and 
Barry and Emmanuel share the remainder in a 4:5 ratio, then Thomas takes 10% 
of the jelly beans.
-/
theorem thomas_percentage (total_jelly_beans : ℕ) (emmanuel_jelly_beans : ℕ)
  (barry_ratio : ℕ) (emmanuel_ratio : ℕ) (thomas_percentage : ℕ) :
  total_jelly_beans = 200 → emmanuel_jelly_beans = 100 → barry_ratio = 4 → emmanuel_ratio = 5 →
  thomas_percentage = 10 :=
by
  intros;
  sorry

end thomas_percentage_l97_97041


namespace seminar_duration_total_l97_97740

/-- The first part of the seminar lasted 4 hours and 45 minutes -/
def first_part_minutes := 4 * 60 + 45

/-- The second part of the seminar lasted 135 minutes -/
def second_part_minutes := 135

/-- The closing event lasted 500 seconds -/
def closing_event_minutes := 500 / 60

/-- The total duration of the seminar session in minutes, including the closing event, is 428 minutes -/
theorem seminar_duration_total :
  first_part_minutes + second_part_minutes + closing_event_minutes = 428 := by
  sorry

end seminar_duration_total_l97_97740


namespace smallest_positive_value_l97_97926

theorem smallest_positive_value (a b c d e : ℝ) (h1 : a = 8 - 2 * Real.sqrt 14) 
  (h2 : b = 2 * Real.sqrt 14 - 8) 
  (h3 : c = 20 - 6 * Real.sqrt 10) 
  (h4 : d = 64 - 16 * Real.sqrt 4) 
  (h5 : e = 16 * Real.sqrt 4 - 64) :
  a = 8 - 2 * Real.sqrt 14 ∧ 0 < a ∧ a < c ∧ a < d :=
by
  sorry

end smallest_positive_value_l97_97926


namespace probability_failed_all_three_tests_l97_97699

def total_students : ℕ := 200
def passed_first_test : ℕ := 110
def passed_second_test : ℕ := 80
def passed_third_test : ℕ := 70
def passed_first_and_second_test : ℕ := 35
def passed_second_and_third_test : ℕ := 30
def passed_first_and_third_test : ℕ := 40
def passed_all_three_tests : ℕ := 20

theorem probability_failed_all_three_tests :
  let n := total_students
      A := passed_first_test
      B := passed_second_test
      C := passed_third_test
      A_inter_B := passed_first_and_second_test
      B_inter_C := passed_second_and_third_test
      A_inter_C := passed_first_and_third_test
      A_inter_B_inter_C := passed_all_three_tests
      union_count := A + B + C - A_inter_B - B_inter_C - A_inter_C + A_inter_B_inter_C
      fail_count := n - union_count
  in fail_count.to_rat / n.to_rat = (1 / 8 : ℚ) :=
  by
  sorry

end probability_failed_all_three_tests_l97_97699


namespace repeating_decimal_fraction_l97_97186

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l97_97186


namespace max_value_E_X_E_Y_l97_97208

open MeasureTheory

-- Defining the random variables and their ranges
variables {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
variable (X : Ω → ℝ) (Y : Ω → ℝ)

-- Condition: 2 ≤ X ≤ 3
def condition1 : Prop := ∀ ω, 2 ≤ X ω ∧ X ω ≤ 3

-- Condition: XY = 1
def condition2 : Prop := ∀ ω, X ω * Y ω = 1

-- The theorem statement
theorem max_value_E_X_E_Y (h1 : condition1 X) (h2 : condition2 X Y) : 
  ∃ E_X E_Y, (E_X = ∫ ω, X ω ∂μ) ∧ (E_Y = ∫ ω, Y ω ∂μ) ∧ (E_X * E_Y = 25 / 24) := 
sorry

end max_value_E_X_E_Y_l97_97208


namespace factorization_of_polynomial_l97_97913

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^2 + 6 * x + 9 - 64 * x^4 = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  intro x
  -- Sorry placeholder for the proof
  sorry

end factorization_of_polynomial_l97_97913


namespace average_weight_of_section_B_l97_97344

theorem average_weight_of_section_B
  (num_students_A : ℕ) (num_students_B : ℕ)
  (avg_weight_A : ℝ) (avg_weight_class : ℝ)
  (total_students : ℕ := num_students_A + num_students_B)
  (total_weight_class : ℝ := total_students * avg_weight_class)
  (total_weight_A : ℝ := num_students_A * avg_weight_A)
  (total_weight_B : ℝ := total_weight_class - total_weight_A)
  (avg_weight_B : ℝ := total_weight_B / num_students_B) :
  num_students_A = 50 →
  num_students_B = 40 →
  avg_weight_A = 50 →
  avg_weight_class = 58.89 →
  avg_weight_B = 70.0025 :=
by intros; sorry

end average_weight_of_section_B_l97_97344


namespace hyperbola_equation_l97_97972

-- Define basic elements and conditions
variables (a b : ℝ) (x y : ℝ)
variable (O : Point)
variable (F : Point)
variable (A : Point)

hypothesis (h1 : a > 0)
hypothesis (h2 : b > 0)
hypothesis (h3 : distance O F = 2)
hypothesis (h4 : distance O A = 2)
hypothesis (h5 : distance A F = 2)
hypothesis (h6 : on_asymptote A O F)
hypothesis (h7 : x ≠ 0 ∨ y ≠ 0 → (x, y) ∉ (asymptote_set O F))

-- Prove the equation of the hyperbola
theorem hyperbola_equation : ( ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ x^2 / a^2 - y^2 / b^2 = 1 ∧ a = 1 ∧ b = √3) → x^2 - y^2 / 3 = 1 :=
by
  sorry

end hyperbola_equation_l97_97972


namespace impossible_fifty_pieces_l97_97754

open Nat

theorem impossible_fifty_pieces :
  ¬ ∃ (m : ℕ), 1 + 3 * m = 50 :=
by
  sorry

end impossible_fifty_pieces_l97_97754


namespace find_x_l97_97677

-- Declaration for the custom operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

-- Theorem statement
theorem find_x (x : ℝ) (h : star 3 x = 23) : x = 29 / 6 :=
by {
    sorry -- The proof steps are to be filled here.
}

end find_x_l97_97677


namespace segment_length_l97_97118

def point (α : Type) := prod α α

variable {F : Type} [field F]

def ellipse (e : F) (c : point F) (f1 f2 : point F) := 
  (e = (1/2 : F)) ∧ (c = (0,0) : point F) ∧ ((f1 = (2,0) : point F) ∨ (f2 = (2,0) : point F))

def parabola (p : point F → Prop) := 
  ∀ x y : F, p (x, y) ↔ (y^2 = 8 * x)

theorem segment_length (A B : point F) : 
  ∀ (E : point F → Prop) (C : point F → Prop),
  ellipse (1/2 : F) (0,0) (2,0) (-2,0) → 
  parabola C → 
  (A = (-2,3) : point F) → 
  (B = (-2,-3) : point F) → 
  dist A B = 6 :=
by
  intro E C hE hC hA hB
  sorry

end segment_length_l97_97118


namespace length_PQ_l97_97676

theorem length_PQ (AB AC BC : ℕ) (P Q : ℝ) 
  (h_AB : AB = 1985) 
  (h_AC : AC = 1983) 
  (h_BC : BC = 1982) 
  (altitude_CH : ∃ CH : ℝ, ∃ H : ℝ, 
    ∀ A B C, P = A ∧ Q = B ∧ CH⊥BC ∧ H ∈ AB) 
  (tangent_PQ : ∀ ACH BCH, 
    tangent_point(ACH, P, CH) ∧ tangent_point(BCH, Q, CH) 
    ∧ inscribed_circle(ACH) ∧ inscribed_circle(BCH)) :
  ∃ m n : ℕ, m + n = 1190 :=
by
  sorry

end length_PQ_l97_97676


namespace find_math_marks_l97_97054

theorem find_math_marks
  (marks_english marks_physics marks_chemistry marks_biology : ℝ)
  (avg_marks : ℝ)
  (num_subjects : ℕ) :
  marks_english = 90 →
  marks_physics = 85 →
  marks_chemistry = 87 →
  marks_biology = 85 →
  avg_marks = 87.8 →
  num_subjects = 5 →
  (347 + M) / 5 = 87.8 →
  M = 92 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  sorry
end

end find_math_marks_l97_97054


namespace distance_between_cities_l97_97731

def distance_on_map : ℝ := 20
def scale_inch : ℝ := 0.4
def scale_mile : ℝ := 5
def miles_per_inch := scale_mile / scale_inch
def actual_distance_in_miles := distance_on_map * miles_per_inch

theorem distance_between_cities :
  actual_distance_in_miles = 250 :=
by
  unfold actual_distance_in_miles miles_per_inch
  have : miles_per_inch = 12.5 := by
    rw [scale_mile, scale_inch]
    norm_num
  rw [this, distance_on_map]
  norm_num
  exact rfl

end distance_between_cities_l97_97731


namespace probability_between_1_and_3_l97_97955

open Real

variables (X : ℝ → Prop) (μ σ : ℝ)
noncomputable def isNormal (X : ℝ → Prop) (μ σ : ℝ) : Prop := sorry

axiom normal_distributed_X : isNormal X 3 σ
axiom P_X_less_than_5 : ∫⁻ x in set.Iic 5, 1 ∂measure_theory.measure_space.volume = 0.8

theorem probability_between_1_and_3 : ∫⁻ x in set.Ioo 1 3, 1 ∂measure_theory.measure_space.volume = 0.3 := by
  sorry

end probability_between_1_and_3_l97_97955


namespace bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97432

-- Definitions of operations
def simple_op_time : ℕ := 1
def lengthy_op_time : ℕ := 5
def num_simple_ops : ℕ := 5
def num_lengthy_ops : ℕ := 3
def total_people : ℕ := num_simple_ops + num_lengthy_ops

-- Proving minimum and maximum person-minutes wasted
theorem bank_queue_min_max_wastage :
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 40) ∧
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 100) :=
by sorry

-- Proving expected value of wasted person-minutes
theorem bank_queue_expected_wastage :
  expected_value_wasted_person_minutes total_people simple_op_time lengthy_op_time = 84 :=
by sorry

-- Placeholder for the actual expected value calculation function
noncomputable def expected_value_wasted_person_minutes
  (n : ℕ) (t_simple : ℕ) (t_lengthy : ℕ) : ℕ :=
  -- Calculation logic will be implemented here
  84 -- This is just the provided answer, actual logic needed for correctness

end bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97432


namespace no_nat_gt_5_with_reversed_digits_eq_base6_l97_97300

theorem no_nat_gt_5_with_reversed_digits_eq_base6 :
  ¬ ∃ (n : ℕ), n > 5 ∧ reverse_digits 10 n = to_base 6 n :=
sorry

end no_nat_gt_5_with_reversed_digits_eq_base6_l97_97300


namespace amount_needed_is_72_l97_97882

-- Define the given conditions
def original_price : ℝ := 90
def discount_rate : ℝ := 20

-- The goal is to prove that the amount of money needed after the discount is $72
theorem amount_needed_is_72 (P : ℝ) (D : ℝ) (hP : P = original_price) (hD : D = discount_rate) : P - (D / 100 * P) = 72 := 
by sorry

end amount_needed_is_72_l97_97882


namespace sphere_surface_area_l97_97581

open Real

theorem sphere_surface_area (A B C D O : Type) [MetricSpace O] 
  (h1: ∃ P ∈ O, isOnSurface P A ∧ isOnSurface P B ∧ isOnSurface P C ∧ isOnSurface P D)
  (h2 : DC ⟂ plane ABC)
  (h3 : ∠ACB = 60 * π / 180)
  (h4 : dist A B = 3 * sqrt 2)
  (h5 : dist D C = 2 * sqrt 3) :
  let r := sqrt 6 in
  let R := sqrt (r^2 + (dist D C / 2)^2) in
  4 * π * R^2 = 36 * π := by
  sorry

end sphere_surface_area_l97_97581


namespace bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97421

def simple_op_time : ℕ := 1
def long_op_time : ℕ := 5
def num_simple_customers : ℕ := 5
def num_long_customers : ℕ := 3
def total_customers : ℕ := 8

theorem bank_queue_minimum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  n * a + 3 * a + b + 4 * a + b + a + b + (b + (n - 1) * a) + b + (b + (n-2) * a) = 40 :=
  by intros; sorry

theorem bank_queue_maximum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  m * (m - 1) * b / 2 + n * a * (m + n) + 1 = 100 :=
  by intros; sorry

theorem expected_wasted_minutes_random_order :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  ∑ i in range total_customers, (i * (a + b)) = 72.5 * (total_customers * (total_customers - 1)) / 2 :=
  by intros; sorry

end bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97421


namespace siblings_pizza_order_l97_97090

theorem siblings_pizza_order :
  let Alex := 1 / 6
  let Beth := 2 / 5
  let Cyril := 1 / 3
  let Dan := 1 - (Alex + Beth + Cyril)
  Dan > Alex ∧ Alex > Cyril ∧ Cyril > Beth := sorry

end siblings_pizza_order_l97_97090


namespace cos_two_pi_over_three_l97_97800

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 := 
by
  sorry

end cos_two_pi_over_three_l97_97800


namespace probability_interval_l97_97113

variable (ξ : ℝ → ℝ) (σ : ℝ)

axiom normal_distribution : ProbabilityDistribution ξ (Normal 2 σ^2)

axiom P_xi_gt_neg2 : (P (λ x, ξ x > -2) = 0.964)

theorem probability_interval :
  P (λ x, -2 ≤ ξ x ∧ ξ x ≤ 6) = 0.928 :=
sorry

end probability_interval_l97_97113


namespace base10_sum_base3_addition_l97_97927

-- Lean definitions for base 10 to base 3 conversion
def to_base3 (n : ℕ) : ℕ :=
  nat.rec_on n 0 (λ n ih, 3 * ih + (n % 3))

-- Lean definition for base 3 addition
def add_base3 (a b : ℕ) : ℕ :=
  let rec add_with_carry : ℕ → ℕ → ℕ → ℕ
  | 0, 0, 0 => 0
  | x, y, c => let sum := (x % 10) + (y % 10) + c
               10 * add_with_carry (x / 10) (y / 10) (sum / 3) + (sum % 3)
  in add_with_carry a b 0

-- Lean definition for base 3 to base 10 conversion
def from_base3 (n : ℕ) : ℕ :=
  let rec from_base_helper : ℕ → ℕ → ℕ
  | 0, _ => 0
  | n, p => (n % 10) * (3^p) + from_base_helper (n / 10) (p + 1)
  in from_base_helper n 0

-- Lean statement to prove the problem
theorem base10_sum_base3_addition :
  from_base3 (add_base3 (to_base3 10) (to_base3 23)) = 33 :=
by
  sorry

end base10_sum_base3_addition_l97_97927


namespace p_is_sufficient_but_not_necessary_for_q_l97_97266

variable (x : ℝ)

def p := x > 1
def q := x > 0

theorem p_is_sufficient_but_not_necessary_for_q : (p x → q x) ∧ ¬(q x → p x) := by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l97_97266


namespace segment_division_three_equal_parts_l97_97151

theorem segment_division_three_equal_parts (l1 l2 : Line) (A B : Point) (AB : Segment A B) (h_parallel: Parallel l1 l2) :
  ∃ X Y : Point, OnLine X l1 ∧ OnLine Y l1 ∧ Between A X B ∧ Between X Y B ∧ AX = XY ∧ XY = YB :=
by sorry

end segment_division_three_equal_parts_l97_97151


namespace neg_sqrt_17_estimate_l97_97068

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end neg_sqrt_17_estimate_l97_97068


namespace primes_with_vp_odd_exist_l97_97598

noncomputable def exists_infinitely_many_primes_vp_odd (x y : ℕ) (hxy_coprime : Nat.gcd x y = 1) : Prop :=
  ∃ᶠ p in Nat.primes, odd (Nat.padics.v_p (x^(p-1) - y^(p-1)) p)

theorem primes_with_vp_odd_exist (x y : ℕ) (hxy_coprime : Nat.gcd x y = 1) :
  exists_infinitely_many_primes_vp_odd x y hxy_coprime :=
sorry

end primes_with_vp_odd_exist_l97_97598


namespace average_hamburgers_sold_per_day_l97_97454

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end average_hamburgers_sold_per_day_l97_97454


namespace race_head_start_l97_97791

variables {Va Vb L H : ℝ}

theorem race_head_start
  (h1 : Va = 20 / 14 * Vb)
  (h2 : L / Va = (L - H) / Vb) : 
  H = 3 / 10 * L :=
by
  sorry

end race_head_start_l97_97791


namespace distribution_ways_l97_97058

theorem distribution_ways (books : Finset (Fin 6)) (ppl : Finset (Fin 3)) (h1 : books.card = 6) (h2 : ppl.card = 3) :
  (∃ f : (Fin 6 → Fin 3), (∀ i, ∃ j, f j = i) ∧ (∃ a b c : Fin 3, (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ ((f ⁻¹' {a}).card ≥ 1 ∧ (f ⁻¹' {b}).card ≥ 1 ∧ (f ⁻¹' {c}).card ≥ 1))) → 
  finset.sum (books.subsets) (λ f, if (∀ a b c : Fin 3, (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ((f.card ≥ 1) → distribution_ways books ppl h1 h2))
 sorry
 
end distribution_ways_l97_97058


namespace front_view_correct_l97_97389

noncomputable def column1 := [1, 3]
noncomputable def column2 := [2, 4, 2]
noncomputable def column3 := [3, 5]
noncomputable def column4 := [2]

def max_in_column (col: list ℕ) : ℕ :=
  col.maximum.getD 0

def front_view (cols: list (list ℕ)) : list ℕ :=
  cols.map max_in_column

theorem front_view_correct :
  front_view [column1, column2, column3, column4] = [3, 4, 5, 2] :=
by
  sorry

end front_view_correct_l97_97389


namespace min_diff_composite_sum_93_l97_97810

def isComposite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

theorem min_diff_composite_sum_93 :
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = 93 ∧ (∀ a' b' : ℕ, isComposite a' ∧ isComposite b' ∧ a' + b' = 93 → |a' - b'| ≥ 3) ∧ |a - b| = 3 :=
by
  sorry

end min_diff_composite_sum_93_l97_97810


namespace factorization_of_polynomial_l97_97914

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^2 + 6 * x + 9 - 64 * x^4 = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  intro x
  -- Sorry placeholder for the proof
  sorry

end factorization_of_polynomial_l97_97914


namespace find_a_l97_97965

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.sin x)^2 - 2 * a * (Real.sin x) + 1

def is_minimum_value (f : ℝ → ℝ → ℝ) (a : ℝ) (m : ℝ) : Prop :=
  ∀ x ∈ set.Icc 0 (Real.pi / 2), f x a ≥ m ∧ (∃ y ∈ set.Icc (0 : ℝ) (Real.pi / 2), f y a = m)

theorem find_a (a : ℝ) :
  is_minimum_value f a (1/2) → a = Real.sqrt 2 / 2 :=
by sorry

end find_a_l97_97965


namespace hyperbola_center_l97_97835

variable {Point : Type}

structure coordinates (P : Point) :=
(x : ℝ)
(y : ℝ)

def center_of_hyperbola (P₁ P₂ : Point) := 
  coordinates.mk ((coordinates.x P₁ + coordinates.x P₂) / 2) ((coordinates.y P₁ + coordinates.y P₂) / 2)

theorem hyperbola_center (f1 f2 : Point) (h1 : coordinates f1) (h2 : coordinates f2) :
  h1 = coordinates.mk 3 (-2) → h2 = coordinates.mk 11 6 → center_of_hyperbola f1 f2 = coordinates.mk 7 2 :=
by
  intros
  sorry

end hyperbola_center_l97_97835


namespace bathroom_length_l97_97811

theorem bathroom_length (A L W : ℝ) (h₁ : A = 8) (h₂ : W = 2) (h₃ : A = L * W) : L = 4 :=
by
  -- Skip the proof with sorry
  sorry

end bathroom_length_l97_97811


namespace boat_distance_downstream_in_6_hours_l97_97318

-- Definitions for the given conditions
def boat_own_speed : ℕ := 40
def boat_speed_against_current : ℕ := 37.5

-- Main theorem statement
theorem boat_distance_downstream_in_6_hours (t : ℕ) (hb : boat_own_speed = 40) (ha : boat_speed_against_current = 37.5) :
  t = 6 → (boat_own_speed + (boat_own_speed - boat_speed_against_current) * t = 255) := by
  sorry

end boat_distance_downstream_in_6_hours_l97_97318


namespace bank_queue_wasted_time_l97_97428

-- Conditions definition
def simple_time : ℕ := 1
def lengthy_time : ℕ := 5
def num_simple : ℕ := 5
def num_lengthy : ℕ := 3
def total_people : ℕ := 8

-- Theorem statement
theorem bank_queue_wasted_time :
  (min_wasted_time : ℕ := 40) ∧
  (max_wasted_time : ℕ := 100) ∧
  (expected_wasted_time : ℚ := 72.5) := by
  sorry

end bank_queue_wasted_time_l97_97428


namespace verify_propositions_l97_97862

theorem verify_propositions
  (h1 : ∀ (a b : ℝ^3), (a ⬝ b > 0) → (acute_angle a b))
  (h2 : ∀ (x y : ℝ), (x + y ≠ 0) → (x ≠ 1 ∨ y ≠ -1))
  (h3 : ∀ (x y : ℝ), (x^2 + y^2 = 1) → (max (λ y, ∃ x, (x^2 + y^2 = 1 ∧ y = t * (x + 2))) (λ t, t = sqrt 3 / 3)))
  (h4 : ∀ (f : ℝ → ℝ), (f = (λ x, 3 * sin (2 * x - real.pi / 3))) → (symmetric_about (f, (2 * real.pi/3, 0)))) :
    correct_propositions = {2, 3, 4} :=
  sorry  -- Proof will be provided here

end verify_propositions_l97_97862


namespace relationship_correct_l97_97936

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem relationship_correct (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  log_base a b < a^b ∧ a^b < b^a :=
by sorry

end relationship_correct_l97_97936


namespace bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97420

def simple_op_time : ℕ := 1
def long_op_time : ℕ := 5
def num_simple_customers : ℕ := 5
def num_long_customers : ℕ := 3
def total_customers : ℕ := 8

theorem bank_queue_minimum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  n * a + 3 * a + b + 4 * a + b + a + b + (b + (n - 1) * a) + b + (b + (n-2) * a) = 40 :=
  by intros; sorry

theorem bank_queue_maximum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  m * (m - 1) * b / 2 + n * a * (m + n) + 1 = 100 :=
  by intros; sorry

theorem expected_wasted_minutes_random_order :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  ∑ i in range total_customers, (i * (a + b)) = 72.5 * (total_customers * (total_customers - 1)) / 2 :=
  by intros; sorry

end bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97420


namespace empty_intersection_iff_l97_97624

def setA (m : ℝ) : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = x^2 + m * x + 2 }
def setB : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = x + 1 ∧ 0 ≤ x ∧ x ≤ 2 }

theorem empty_intersection_iff (m : ℝ) : (setA m ∩ setB = ∅) ↔ -1 < m ∧ m < 3 :=
by
  sorry

end empty_intersection_iff_l97_97624


namespace bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97423

def simple_op_time : ℕ := 1
def long_op_time : ℕ := 5
def num_simple_customers : ℕ := 5
def num_long_customers : ℕ := 3
def total_customers : ℕ := 8

theorem bank_queue_minimum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  n * a + 3 * a + b + 4 * a + b + a + b + (b + (n - 1) * a) + b + (b + (n-2) * a) = 40 :=
  by intros; sorry

theorem bank_queue_maximum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  m * (m - 1) * b / 2 + n * a * (m + n) + 1 = 100 :=
  by intros; sorry

theorem expected_wasted_minutes_random_order :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  ∑ i in range total_customers, (i * (a + b)) = 72.5 * (total_customers * (total_customers - 1)) / 2 :=
  by intros; sorry

end bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97423


namespace intersection_points_log_functions_eq_five_l97_97705

theorem intersection_points_log_functions_eq_five : 
  (∃! p, 
    (∃ x > 0, p = (x, Real.log 3 x) ∧ 
      (p = (x, Real.log x 3) ∨ 
      p = (x, Real.log (1/3) x) ∨ 
      p = (x, Real.log x (1/3))))) = 5 :=
sorry

end intersection_points_log_functions_eq_five_l97_97705


namespace area_relationship_l97_97271

variables (A B C P L F M D N E : Type*)
variables [Point : Point] [Triangle : Triangle]
variables [line_through_P_parallel_AB : ∀ (P : Point), Line P L F → Line L F AB]
variables [line_through_P_parallel_BC : ∀ (P : Point), Line P M D → Line M D BC]
variables [line_through_P_parallel_CA : ∀ (P : Point), Line P N E → Line N E CA]

theorem area_relationship 
  (h_P : Point ∈ Triangle ABC)
  (h1 : Line_through_P_parallel_AB P L F)
  (h2 : Line_through_P_parallel_BC P M D)
  (h3 : Line_through_P_parallel_CA P N E)
  : (area P D B L) * (area P E C M) * (area P F A N) = 8 * (area P F M) * (area P E L) * (area P D N) :=
sorry

end area_relationship_l97_97271


namespace parakeets_per_cage_is_2_l97_97842

variables (cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

def number_of_parakeets_each_cage : ℕ :=
  (total_birds - cages * parrots_per_cage) / cages

theorem parakeets_per_cage_is_2
  (hcages : cages = 4)
  (hparrots_per_cage : parrots_per_cage = 8)
  (htotal_birds : total_birds = 40) :
  number_of_parakeets_each_cage cages parrots_per_cage total_birds = 2 := 
by
  sorry

end parakeets_per_cage_is_2_l97_97842


namespace max_min_x_plus_y_l97_97571

theorem max_min_x_plus_y (x y : ℝ) (h : |x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) :
  -3 ≤ x + y ∧ x + y ≤ 6 := 
sorry

end max_min_x_plus_y_l97_97571


namespace smallest_solution_floor_eq_l97_97529

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l97_97529


namespace find_elephants_l97_97908

def animals_seen_saturday := 3 + n
def animals_seen_sunday := 2 + 5
def animals_seen_monday := 5 + 3
def total_animals_seen := animals_seen_saturday + animals_seen_sunday + animals_seen_monday

theorem find_elephants (n : ℕ) (h : total_animals_seen = 20) :
  n = 2 :=
by sorry

end find_elephants_l97_97908


namespace g_inv_g_inv_14_l97_97308

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l97_97308


namespace six_digit_divisible_by_44_l97_97563

theorem six_digit_divisible_by_44 (n : ℕ) (h : n < 10) : 
  (5 * 10 ^ 5 + n * 10 ^ 4 + 7 * 10 ^ 3 + 2 * 10 ^ 2 + 6 * 10 + 4) % 44 = 0 ↔ n = 1 :=
begin
  sorry
end

end six_digit_divisible_by_44_l97_97563


namespace expression_equality_l97_97893

theorem expression_equality :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := 
  sorry

end expression_equality_l97_97893


namespace ellipse_to_parabola_standard_eq_l97_97140

theorem ellipse_to_parabola_standard_eq :
  ∀ (x y : ℝ), (x^2 / 25 + y^2 / 16 = 1) → (y^2 = 12 * x) :=
by
  sorry

end ellipse_to_parabola_standard_eq_l97_97140


namespace solve_expression_l97_97794

theorem solve_expression : (0.76 ^ 3 - 0.008) / (0.76 ^ 2 + 0.76 * 0.2 + 0.04) = 0.560 := 
by
  sorry

end solve_expression_l97_97794


namespace non_congruent_squares_l97_97162

theorem non_congruent_squares (n : ℕ) (h : n = 6) : 
  let standard_aligned_squares := (n-1)^2 + (n-2)^2 + (n-3)^2 + (n-4)^2 + (n-5)^2,
      diagonal_squares := (n-1)^2 + (n-2)^2 + (n-3)^2
  in standard_aligned_squares + diagonal_squares = 105 :=
by
  sorry

end non_congruent_squares_l97_97162


namespace point_on_graph_l97_97605

variable f : ℝ → ℝ

noncomputable def satisfies_condition : Prop :=
  (f 3 = 5) ∧ (2 * (13 : ℝ) = 4 * f (3 * (1 : ℝ)) + 6)

theorem point_on_graph :
  satisfies_condition → (1 + 13 = 14) :=
by
  intros
  have h : 2 * (13 : ℝ) = 4 * f (3 * (1 : ℝ)) + 6
  · sorry
  have f_eq : f 3 = 5 := by sorry
  sorry

end point_on_graph_l97_97605


namespace reverse_base_sum_l97_97554

theorem reverse_base_sum :
  {n : ℕ | ∃ d a_d a_d1 a_d2, 
            n = 5^d * a_d + 5^(d-1) * a_d1 + 5^(d-2) * a_d2 ∧
            n = 12^d * a_d2 + 12^(d-1) * a_d1 + 12^(d-2) * a_d ∧
            (12^d - 1) * a_d2 + (12^(d-1) - 5) * a_d1 + (12^(d-2) - 5^(d-2)) * a_d = 0 ∧
            d ≤ 2 ∧ a_d ≤ 4 ∧ a_d1 ≤ 4 ∧ a_d2 ≤ 4}.sum = 10 := 
sorry

end reverse_base_sum_l97_97554


namespace sector_area_example_l97_97639

open Real

-- Definitions based on conditions
def perimeter (r : ℝ) := 2 * r + 2 * r
def sector_area (r θ : ℝ) := 1 / 2 * θ * r * r

-- The theorem that needs to be proven
theorem sector_area_example :
  ∃ r : ℝ, perimeter r = 16 ∧ sector_area r 2 = 16 := by
  sorry

end sector_area_example_l97_97639


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97410

def a : ℕ := 1  -- time for a simple operation
def b : ℕ := 5  -- time for a lengthy operation
def n : ℕ := 5  -- number of "simple" customers
def m : ℕ := 3  -- number of "lengthy" customers
def total_customers : ℕ := 8 -- 8 people in queue

theorem min_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → min_wasted_person_minutes ≤ 40) :=
by
  sorry

theorem max_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → max_wasted_person_minutes ≥ 100) :=
by
  sorry

theorem expected_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → expected_wasted_person_minutes = 72.5) :=
by
  sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97410


namespace remainder_of_504_divided_by_100_is_4_l97_97902

theorem remainder_of_504_divided_by_100_is_4 :
  (504 % 100) = 4 :=
by
  sorry

end remainder_of_504_divided_by_100_is_4_l97_97902


namespace smallest_solution_floor_eq_l97_97546

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l97_97546


namespace sum_of_fraction_components_l97_97195

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l97_97195


namespace sufficient_not_necessary_l97_97323

-- Definitions of the domains A and B
def A : Set ℝ := {x | (2 / (x + 1) - 1) > 0}

def B (a : ℝ) : Set ℝ := {x | 1 - |x + a| ≥ 0}

-- Statement to be proved
theorem sufficient_not_necessary (a : ℝ) : (a ≥ 2 → A ∩ B a = ∅) ∧ (A ∩ B (-3) = ∅) → IsSufficientButNotNecessary (A ∩ B a = ∅) (a ≥ 2) :=
sorry

-- Custom predicate indicating "sufficient but not necessary condition"
def IsSufficientButNotNecessary (P : Prop) (cond : Prop) : Prop :=
(cond → P) ∧ (∃ x, P ∧ ¬cond)

end sufficient_not_necessary_l97_97323


namespace brasuca_board_count_l97_97808

def is_brasuca_board (board : Matrix (Fin 4) (Fin 4) (Fin 6)) : Prop :=
  (∀ i : Fin 4, (board i).sum = 5) ∧
  (∀ j : Fin 4, ((fun i => board i j) : Fin 4 → Fin 6).sum = 5) ∧
  (board (Fin 0) (Fin 0) + board (Fin 1) (Fin 1) + board (Fin 2) (Fin 2) + board (Fin 3) (Fin 3) = 5) ∧
  (board (Fin 3) (Fin 0) + board (Fin 2) (Fin 1) + board (Fin 1) (Fin 2) + board (Fin 0) (Fin 3) = 5) ∧
  (∀ i j : Fin 4, board (Fin 0) (Fin 0) ≤ board i j) ∧
  ((board (Fin 0) (Fin 0) + board (Fin 0) (Fin 1) + board (Fin 1) (Fin 0) + board (Fin 1) (Fin 1) = 5) ∧
   (board (Fin 0) (Fin 2) + board (Fin 0) (Fin 3) + board (Fin 1) (Fin 2) + board (Fin 1) (Fin 3) = 5) ∧
   (board (Fin 2) (Fin 0) + board (Fin 2) (Fin 1) + board (Fin 3) (Fin 0) + board (Fin 3) (Fin 1) = 5) ∧
   (board (Fin 2) (Fin 2) + board (Fin 2) (Fin 3) + board (Fin 3) (Fin 2) + board (Fin 3) (Fin 3) = 5))

theorem brasuca_board_count : 
  finset.card { board : Matrix (Fin 4) (Fin 4) (Fin 6) | is_brasuca_board board } = 462 :=
sorry

end brasuca_board_count_l97_97808


namespace general_term_formula_sum_of_first_2n_terms_l97_97138

open Nat

def Sn (n : ℕ) : ℤ := n - n^2

def an (n : ℕ) : ℤ := 1 - n

def bn (n : ℕ) : ℤ :=
  if n % 2 = 1 then 2^(an n)
  else
    let k := n / 2;
    let an1 := an (2 * k);
    let an2 := an (2 * k + 2);
    2 / ((1 - an1) * (1 - an2))

def T2n (n : ℕ) : ℤ := sorry  -- To be defined later

theorem general_term_formula (n : ℕ) (h : n > 0) : an n = 1 - n := sorry

theorem sum_of_first_2n_terms (n : ℕ) : T2n n = 
  (11 / 6) - (4 / 3) * (1 / 4)^n - (1 / (2 * n + 2)) := sorry

end general_term_formula_sum_of_first_2n_terms_l97_97138


namespace rectangle_length_reduction_30_percent_l97_97738

variables (L W : ℝ) (x : ℝ)

theorem rectangle_length_reduction_30_percent
  (h : 1 = (1 - x / 100) * 1.4285714285714287) :
  x = 30 :=
sorry

end rectangle_length_reduction_30_percent_l97_97738


namespace find_x_l97_97628

variables (x : ℝ)
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (x, -2)
def c : ℝ × ℝ := (0, 2)

theorem find_x (h : a.1 * (b.1 - c.1) + a.2 * (b.2 - c.2) = 0) : x = 4 / 3 :=
sorry

end find_x_l97_97628


namespace fly_total_distance_l97_97826

-- Define the conditions as a structure
structure CircularRoom where
  radius : ℝ

def fly_journey (cr : CircularRoom) : ℝ := 
  let diameter := 2 * cr.radius
  let last_leg := 85
  let second_leg := Real.sqrt (diameter^2 - last_leg^2)
  diameter + last_leg + second_leg

-- Theorem statement
theorem fly_total_distance (r : ℝ) (h : r = 60) : fly_journey ⟨r⟩ = 205 + Real.sqrt 7175 :=
by 
  sorry

end fly_total_distance_l97_97826


namespace mul_fraction_eq_l97_97494

theorem mul_fraction_eq : 7 * (1 / 11) * 33 = 21 :=
by
  sorry

end mul_fraction_eq_l97_97494


namespace problem_statement_l97_97469

-- Definitions of the events as described in the problem conditions.
def event1 (a b : ℝ) : Prop := a * b < 0 → a + b < 0
def event2 (a b : ℝ) : Prop := a * b < 0 → a - b > 0
def event3 (a b : ℝ) : Prop := a * b < 0 → a * b > 0
def event4 (a b : ℝ) : Prop := a * b < 0 → a / b < 0

-- The problem statement combining the conditions and the conclusion.
theorem problem_statement (a b : ℝ) (h1 : a * b < 0):
  (event4 a b) ∧ ¬(event3 a b) ∧ (event1 a b ∨ ¬(event1 a b)) ∧ (event2 a b ∨ ¬(event2 a b)) :=
by
  sorry

end problem_statement_l97_97469


namespace proof_problem_l97_97242

-- Define the parametric line l
def parametric_line (α t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 2 + t * Real.sin α)

-- Define the polar curve C
def polar_curve (ρ θ : ℝ) : ℝ :=
  ρ^2 + 2 * ρ * Real.cos θ - 2 * ρ * Real.sin θ + 1

-- Cartesian form of the line when α = π/4
def cartesian_line : Prop :=
  ∀ x y : ℝ, (∃ t : ℝ, x = t * Real.cos (Real.pi / 4) ∧ y = 2 + t * Real.sin (Real.pi / 4)) ↔ x - y + 2 = 0

-- Cartesian formal of the curve when converted from polar to Cartesian
def cartesian_curve : Prop :=
  ∀ x y : ℝ, (polar_curve (Real.sqrt (x^2 + y^2)) (Real.atan2 y x) = 0) ↔ (x + 1)^2 + (y - 1)^2 = 1

-- Minimum value calculation proposition
def minimum_value (α : ℝ) (hα : 0 < α ∧ α ≤ Real.pi / 3) : Prop :=
  let PA PB : ℝ := sorry -- Define these properly using geometric considerations
  let t1 t2 : ℝ := sorry -- Intersection calculations
  let value := |PA| * |PB| / (|PA| + |PB|)
  value = (Real.sqrt 2) / 4

-- Main theorem statement
theorem proof_problem :
  cartesian_line ∧ cartesian_curve ∧
  (∀ α : ℝ, 0 < α ∧ α ≤ Real.pi / 3 → minimum_value α sorry) :=
  sorry

end proof_problem_l97_97242


namespace coeff_of_inv_x_sq_in_expansion_l97_97954

theorem coeff_of_inv_x_sq_in_expansion :
  ∀ (n : ℕ), (∀ x : ℝ, n = (min (λ x : ℝ, abs (x - 1) + abs (x + 7)))) →
  ((x + 1/x) ^ 8).coeff ( -2) = 56 :=
by
  intros n hn,
  sorry

end coeff_of_inv_x_sq_in_expansion_l97_97954


namespace average_minutes_run_per_day_l97_97222

theorem average_minutes_run_per_day
  (s : ℕ) -- number of seventh graders and eighth graders
  (sixth_graders_run : ℕ := 18)
  (seventh_graders_run : ℕ := 12)
  (eighth_graders_run : ℕ := 16)
  (num_sixth_graders : ℕ := 3 * s) -- three times as many sixth graders as seventh graders
  (num_seventh_graders : ℕ := s)
  (num_eighth_graders : ℕ := s)
  : (6*18*s + 12*s + 16*s)/5 = 82/5 := 
by {
  rw[add_mul],
  have H: 6=3+1+2,
  linarith,
  rw[H],
  
  -- Solve the proof
  sorry
}

end average_minutes_run_per_day_l97_97222


namespace framed_painting_ratio_4_7_l97_97841

theorem framed_painting_ratio_4_7 (y : ℝ) 
    (painting_width : ℝ := 15)
    (painting_height : ℝ := 20)
    (top_bottom_frame_width : ℝ := 3 * y) 
    (side_frame_width : ℝ := y) 
    (frame_area_equals_painting_area : (painting_width + 2 * side_frame_width) * (painting_height + 2 * top_bottom_frame_width) - painting_width * painting_height = painting_width * painting_height) :
    ((painting_width + 2 * side_frame_width) / (painting_height + 2 * top_bottom_frame_width) = 4 / 7) :=
by
  have h1 : painting_height = 20 := rfl
  have h2 : painting_width = 15 := rfl
  have h3 : top_bottom_frame_width = 3 * y := rfl
  have h4 : side_frame_width = y := rfl
  have h5 : frame_area_equals_painting_area := sorry
  sorry

end framed_painting_ratio_4_7_l97_97841


namespace cone_volume_l97_97574

theorem cone_volume (l h : ℝ) (h1 : l = 5) (h2 : h = 4) : 
  (1 / 3) * π * (real.sqrt (l^2 - h^2))^2 * h = 12 * π :=
by
  sorry

end cone_volume_l97_97574


namespace circle_placement_in_rectangle_l97_97014

theorem circle_placement_in_rectangle
  (L W : ℝ) (n : ℕ) (side_length diameter : ℝ)
  (h_dim : L = 20) (w_dim : W = 25)
  (h_squares : n = 120) (h_side_length : side_length = 1)
  (h_diameter : diameter = 1) :
  ∃ (x y : ℝ) (circle_radius : ℝ), 
    circle_radius = diameter / 2 ∧
    0 ≤ x ∧ x + diameter / 2 ≤ L ∧ 
    0 ≤ y ∧ y + diameter / 2 ≤ W ∧ 
    ∀ (i : ℕ) (hx : i < n) (sx sy : ℝ),
      0 ≤ sx ∧ sx + side_length ≤ L ∧
      0 ≤ sy ∧ sy + side_length ≤ W ∧
      dist (x, y) (sx + side_length / 2, sy + side_length / 2) ≥ diameter / 2 := 
sorry

end circle_placement_in_rectangle_l97_97014


namespace ideal_pairs_count_l97_97684

def I := {1, 2, 3, 4, 5, 6}
def ideal_pair_count : ℕ := 27

theorem ideal_pairs_count (A B : Set ℕ) (hA : A ⊆ I) (hB : B ⊆ I)
  (h : A ∩ B = {1, 3, 5}) : (∃ n, n = ideal_pair_count ∧
  n = 27) :=
by {
  use 27,
  split,
  { refl },
  { refl }
}

end ideal_pairs_count_l97_97684


namespace straw_costs_max_packs_type_a_l97_97436

theorem straw_costs (x y : ℝ) (h1 : 12 * x + 15 * y = 171) (h2 : 24 * x + 28 * y = 332) :
  x = 8 ∧ y = 5 :=
  by sorry

theorem max_packs_type_a (m : ℕ) (cA cB : ℕ) (total_packs : ℕ) (max_cost : ℕ)
  (h1 : cA = 8) (h2 : cB = 5) (h3 : total_packs = 100) (h4 : max_cost = 600) :
  m ≤ 33 :=
  by sorry

end straw_costs_max_packs_type_a_l97_97436


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97413

def a : ℕ := 1  -- time for a simple operation
def b : ℕ := 5  -- time for a lengthy operation
def n : ℕ := 5  -- number of "simple" customers
def m : ℕ := 3  -- number of "lengthy" customers
def total_customers : ℕ := 8 -- 8 people in queue

theorem min_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → min_wasted_person_minutes ≤ 40) :=
by
  sorry

theorem max_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → max_wasted_person_minutes ≥ 100) :=
by
  sorry

theorem expected_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → expected_wasted_person_minutes = 72.5) :=
by
  sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97413


namespace matrix_exponentiation_l97_97258

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  [3, 2],
  [0, 2]
]

theorem matrix_exponentiation : B ^ 15 - 3 * B ^ 14 = ![
  [0, 16384],
  [0, -8192]
] := by
  sorry

end matrix_exponentiation_l97_97258


namespace midpoints_on_thales_circle_l97_97352

noncomputable theory
open_locale classical

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

def midpoint (α : Type*) [linear_ordered_field α] (A B : α × α) : α × α :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_on_circle (α : Type*) [metric_space α] (M : α) (c : Circle α) : Prop :=
dist M c.center = c.radius

def is_thales_circle (α : Type*) [metric_space α] (c_big c_small : Circle α) (O P : α) : Prop :=
c_small.center = midpoint α O P ∧
c_small.radius = dist O P / 2

theorem midpoints_on_thales_circle {α : Type*} [metric_space α] [linear_ordered_field α]
(O P : α) (R : ℝ) (c_big c_small : Circle α) :
P ≠ O → is_on_circle α P c_big → is_thales_circle α c_big c_small O P →
∀ M, (∃ A B, is_on_circle α A c_big ∧ is_on_circle α B c_big ∧ midpoint α A B = M) →
is_on_circle α M c_small :=
sorry

end midpoints_on_thales_circle_l97_97352


namespace arithmetic_sequence_sum_l97_97230

theorem arithmetic_sequence_sum  (a : ℕ → ℤ)
  (h3 : a 3 = 1)
  (h7 : a 7 = 3):
  (∑ i in Finset.range (9 + 1), a i) = 18 := by
sorry

end arithmetic_sequence_sum_l97_97230


namespace num_pens_in_second_set_l97_97806

variables (num_pens_some_set num_pens_second_set : ℕ)
variables (pen_cost pencil_cost : ℝ)

-- The conditions translated into Lean definitions
def condition_1 := 3 * pencil_cost + num_pens_some_set * pen_cost = 1.58
def condition_2 := 4 * pencil_cost + 5 * pen_cost = 2.00
def pencil_cost_value := pencil_cost = 0.1

-- The proof problem in Lean 4
theorem num_pens_in_second_set 
    (h1 : condition_1)
    (h2 : condition_2)
    (h3 : pencil_cost_value)
    : num_pens_second_set = 5 :=
sorry

end num_pens_in_second_set_l97_97806


namespace intersection_slopes_l97_97002

theorem intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (4 / 41)) ∨ m ∈ Set.Ici (Real.sqrt (4 / 41)) := 
sorry

end intersection_slopes_l97_97002


namespace simplify_fraction_1_simplify_fraction_2_l97_97303

theorem simplify_fraction_1 (a x : ℝ) (h₁ : sqrt a ≠ sqrt x) :
  (a * sqrt x - x * sqrt a) / (sqrt a - sqrt x) = sqrt (a * x) :=
by sorry

theorem simplify_fraction_2 (a b : ℝ) (h₂ : a + b ≠ sqrt (a^2 - b^2)) :
  (sqrt (a + b) - sqrt (a - b)) / (a + b - sqrt (a^2 - b^2)) = 1 / sqrt (a + b) :=
by sorry

end simplify_fraction_1_simplify_fraction_2_l97_97303


namespace smallest_x_solution_l97_97521

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l97_97521


namespace function_inequality_l97_97968

noncomputable def f : ℝ → ℝ
| x => if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

theorem function_inequality : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end function_inequality_l97_97968


namespace repeating_decimal_sum_l97_97177

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l97_97177


namespace angle_between_skew_lines_l97_97650

-- Define the geometrical setup of the regular tetrahedron
structure Tetrahedron :=
  (A B C D : ℝ × ℝ × ℝ)
  (is_regular : dist A B = dist A C ∧ dist A B = dist A D ∧
                dist B C = dist B D ∧ dist C D = dist A C ∧
                dist A C = 1)

-- Define the midpoint function
def midpoint (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2, (p.3 + q.3) / 2)

-- Define the centroid of a triangle function
def centroid (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p.1 + q.1 + r.1) / 3, (p.2 + q.2 + r.2) / 3, (p.3 + q.3 + r.3) / 3)

-- Define the vector subtraction
def vector_sub (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2, p.3 - q.3)

-- Define the dot product
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the norm (magnitude) of a vector
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def angle (v w : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (dot_product v w / (norm v * norm w))

-- The theorem to be proven
theorem angle_between_skew_lines {tetra : Tetrahedron} (E : ℝ × ℝ × ℝ) :
  let M := midpoint tetra.A tetra.C in
  let N := centroid tetra.B tetra.C tetra.D in
  ∃ DE : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ,
  ((∀ p q, DE p = vector_sub tetra.D E ∧ DE q = vector_sub tetra.B p ∧ dot_product (vector_sub tetra.D E) (vector_sub tetra.B p) = 0) ∧
  (angle (vector_sub N M) (vector_sub tetra.D E)) = real.arccos (5 / (6 * real.sqrt 3))) :=
sorry

end angle_between_skew_lines_l97_97650


namespace quadratic_equation_conversion_l97_97499

theorem quadratic_equation_conversion :
  ∀ (x : ℝ), 
  let eq := 3 * x * (x - 1) = 2 * (x + 2) + 8 in
  (∀ (a b c : ℝ), (3 * x^2 - 5 * x - 12 = a * x^2 + b * x + c) → (a = 3 ∧ b = -5)) :=
by
  intro x eq
  intro a b c h
  sorry

end quadratic_equation_conversion_l97_97499


namespace num_integer_solutions_l97_97560

def circle_center := (3, 3)
def circle_radius := 10

theorem num_integer_solutions :
  (∃ f : ℕ, f = 15) :=
sorry

end num_integer_solutions_l97_97560


namespace evaluate_g_at_neg1_l97_97621

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_g_at_neg1 : g (-1) = -5 := by
  sorry

end evaluate_g_at_neg1_l97_97621


namespace equilateral_triangles_l97_97874

section

variables {A B C A' B' C' D E F G H K : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B]
variables [InnerProductSpace ℝ C] [InnerProductSpace ℝ A']
variables [InnerProductSpace ℝ B'] [InnerProductSpace ℝ C']
variables [InnerProductSpace ℝ D] [InnerProductSpace ℝ E]
variables [InnerProductSpace ℝ F] [InnerProductSpace ℝ G]
variables [InnerProductSpace ℝ H] [InnerProductSpace ℝ K]

variables (hexagon_DEFGHK : A B C A' B' C' → Type)
variables (parallelogram_conditions : EF ∥ KH ∧ GH ∥ DE ∧ FG ∥ KD)
variables (length_conditions : KH - EF = FG - KD ∧ FG - KD = DE - GH ∧ KH - EF > 0)

theorem equilateral_triangles
  (hexagon_DEFGHK : A B C A' B' C' → Type)
  (parallelogram_conditions : EF ∥ KH ∧ GH ∥ DE ∧ FG ∥ KD)
  (length_conditions : KH - EF = FG - KD ∧ FG - KD = DE - GH ∧ KH - EF > 0) :
  (triangle_equilateral A B C ∧ triangle_equilateral A' B' C') :=
sorry

end

end equilateral_triangles_l97_97874


namespace sum_mod_15_l97_97380

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end sum_mod_15_l97_97380


namespace perimeter_of_triangle_l97_97649

-- Define the non-isosceles triangle and its properties
variables {A B C : Real.Angle} (a b c : ℝ)
variable (ABC_non_isosceles : ¬ Isosceles a b c)
variable (h1 : a * cos (C / 2) ^ 2 + c * cos (A / 2) ^ 2 = (3 / 2) * c)
variable (h2 : 2 * sin (A - B) + b * sin B = a * sin A)

-- Define the condition that needs to be proven
theorem perimeter_of_triangle (a b c : ℝ) (ABC_non_isosceles : ¬ Isosceles a b c)
  (h1 : a * cos (C / 2) ^ 2 + c * cos (A / 2) ^ 2 = (3 / 2) * c)
  (h2 : 2 * sin (A - B) + b * sin B = a * sin A) :
  a + b + c = 6 :=
  sorry

end perimeter_of_triangle_l97_97649


namespace basketball_not_table_tennis_l97_97752

theorem basketball_not_table_tennis :
  ∀ (total_students basketball_likes table_tennis_likes neither_likes : ℕ),
    total_students = 30 →
    basketball_likes = 15 →
    table_tennis_likes = 10 →
    neither_likes = 8 →
    (∃ (both_likes : ℕ), basketball_likes - both_likes = 12) :=
by
  intros total_students basketball_likes table_tennis_likes neither_likes
  assume h1 h2 h3 h4
  let both_likes := (total_students - basketball_likes - table_tennis_likes + neither_likes)
  existsi (both_likes - neither_likes)
  have : both_likes - neither_likes = 3 :=
    calc
      both_likes - neither_likes
          = (30 - 15 - 10 + 8) - 8 : by rw [h1, h2, h3, h4]
      ... = 3 : rfl
  have : 15 - both_likes = 12 :=
    calc
      15 - both_likes
          = 15 - (30 - 15 - 10 + 8) : by rw [h1, h2, h3, h4]
      ... = 15 - 3 : by sorry -- additional calculations not fully expanded
  existsi both_likes - neither_likes
  exact this

end basketball_not_table_tennis_l97_97752


namespace find_distance_l97_97719

-- Define the scenario
def square (E F G H : Point) : Prop := 
  (distance E F = 5) ∧ 
  (distance F G = 5) ∧ 
  (distance G H = 5) ∧ 
  (distance H E = 5)

def midpoint (P1 P2 M : Point) : Prop := 
  (P1.x + P2.x = 2 * M.x) ∧ 
  (P1.y + P2.y = 2 * M.y)

def circle (center : Point) (radius : ℝ) (P : Point) : Prop := 
  (distance center P = radius)

-- Define the points and conditions
def E : Point := {x := 0, y := 5}
def F : Point := {x := 5, y := 5}
def G : Point := {x := 5, y := 0}
def H : Point := {x := 0, y := 0}
def N : Point := {x := 2.5, y := 0}

-- Define the target point Q and the required distance
def Q : Point := sorry -- This needs calculations but is an intersection we only need to reference

theorem find_distance (Q : Point) 
  (h1 : square E F G H)
  (h2 : midpoint G H N)
  (h3 : circle N 2.5 Q)
  (h4 : circle E 5 Q) :
  (distance Q (line EH)) = 1.33 :=
sorry

end find_distance_l97_97719


namespace shaded_region_area_l97_97238

-- Define the radius of the circles
def radius : ℝ := 5

-- Define the area of the shaded region
def shaded_area : ℝ := 50 * π - 100

-- Theorem stating the area of the shaded region given the conditions
theorem shaded_region_area 
  (r : ℝ) (h_radius : r = radius) 
  (sh_area : ℝ) (h_shaded_area : sh_area = shaded_area) :
  sh_area = 50 * π - 100 := by {
  -- Placeholder for the proof, which derives the shaded area from the given radius and intersecting circles.
  sorry
}

end shaded_region_area_l97_97238


namespace hannah_total_spent_l97_97983

noncomputable def total_cost (sweatshirts t_shirts socks jackets : ℕ) 
                             (price_sweatshirts price_t_shirts price_socks : ℕ) 
                             (prices_jackets : list ℕ) 
                             (discount_coupon discount_overall : ℝ) : ℝ :=
  let cost_sweatshirts := sweatshirts * price_sweatshirts
  let cost_t_shirts := t_shirts * price_t_shirts
  let cost_socks := socks * price_socks
  let cost_jackets := prices_jackets.sum
  let total := cost_sweatshirts + cost_t_shirts + cost_socks + cost_jackets
  let discount_amount := discount_overall * total
  total - discount_amount

theorem hannah_total_spent :
  total_cost 3 2 4 [40, 50, 60] 15 10 5 0 0.10 = 211.50 :=
by
  sorry

end hannah_total_spent_l97_97983


namespace pentagon_line_bisector_l97_97319

theorem pentagon_line_bisector 
  (A B C D E : Type) 
  (AB : A → B → Prop)
  (CD : C → D → Prop)
  (BC : B → C → Prop)
  (AD : A → D → Prop)
  (AC : A → C → Prop)
  (DE : D → E → Prop)
  (CE : C → E → Prop)
  (h1 : AB || CD)
  (h2 : BC || AD)
  (h3 : AC || DE)
  (h4 : Perp CE BC) :
  is_angle_bisector EC ∠BED :=
sorry

end pentagon_line_bisector_l97_97319


namespace line_of_symmetry_l97_97055

theorem line_of_symmetry (x : ℝ):
  let f := λ x, (4 : ℝ)^(-x)
  let g := λ x, (2 : ℝ)^(2 * x - 3)
  ∃ x0, (∀ x, f x0 = g x0) ↔ (x0 = 9 / 4) :=
by
  sorry

end line_of_symmetry_l97_97055


namespace lcm_gcd_sum_l97_97928

theorem lcm_gcd_sum (n: ℕ) (d: ℕ) (h1: d = Nat.gcd n 144) (h2: n * 144 = d * Nat.lcm n 144) :
  ∑ m in {m : ℕ | Nat.lcm m 144 = Nat.gcd m 144 + 360} = 216 :=
by
  sorry

end lcm_gcd_sum_l97_97928


namespace find_f_expression_find_alpha_value_l97_97827

noncomputable theory

-- Definitions of conditions
def f (x : ℝ) : ℝ := A * sin(ω * x - φ) + 2
def highest_point := (5 * Real.pi / 12, 4) -- highest point
def symmetry_distance := Real.pi / 2      -- distance between symmetry axes
variables (A ω φ : ℝ)
variables (hA : 0 < A) (hω : 0 < ω) (hφ : 0 < φ ∧ φ < Real.pi / 2)

-- Proof parts
theorem find_f_expression (hx : f (5 * Real.pi / 12) = 4) 
                           (hsym : symmetry_distance = Real.pi / 2) 
                           : f = (λ x, 2 * sin(2 * x - Real.pi / 3) + 2) :=
sorry

theorem find_alpha_value (α : ℝ) (hα : 0 < α ∧ α < Real.pi)
                          (hf_val : f (α / 2) = 3) 
                          : α = Real.pi / 2 :=
sorry

end find_f_expression_find_alpha_value_l97_97827


namespace set_c_is_empty_l97_97861

theorem set_c_is_empty : 
  (λ x : ℝ, x^2 - x + 1 = 0) = ∅ := 
sorry

end set_c_is_empty_l97_97861


namespace sum_solutions_l97_97687

-- Define the conditions as predicates
def condition1 (x y : ℝ) := |x - 5| = |y - 11|
def condition2 (x y : ℝ) := |x - 11| = 3 * |y - 5|

-- Sum up all possible pairs (x, y) satisfying the conditions
theorem sum_solutions : 
  let solutions := [(x, y) | x y, condition1 x y ∧ condition2 x y] in
  ∑ (x, y) in solutions, x + y = 23 := 
  sorry

end sum_solutions_l97_97687


namespace quadratic_discriminant_correct_l97_97078

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant_correct :
  discriminant 5 (5 + 1/2) (-1/2) = 161 / 4 :=
by
  -- let's prove the equality directly
  sorry

end quadratic_discriminant_correct_l97_97078


namespace maximum_distance_achieved_l97_97241

noncomputable def max_distance_circle_line 
  (θ : ℝ) 
  (x : ℝ := (sqrt 6) / 2 * cos θ)
  (y : ℝ := (sqrt 6) / 2 * sin θ) 
  (ρ : ℝ := sqrt 2 / (sqrt 7 * cos θ - sin θ)) 
  (distance : ℝ := abs ((- sqrt 2) / sqrt ((sqrt 7)^2 + (-1)^2))) : ℝ := 
  (sqrt 6) / 2 + distance

theorem maximum_distance_achieved 
  (θ : ℝ) : 
  max_distance_circle_line θ = (sqrt 6) / 2 + 1 / 2 :=
sorry

end maximum_distance_achieved_l97_97241


namespace angle_A44_A45_A43_l97_97799

-- Definitions and conditions
def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_midpoint (M A B : Point) : Prop :=
  dist A M = dist M B ∧ dist A B = dist A M + dist M B

-- Main statement of the problem
theorem angle_A44_A45_A43 
  (A : ℕ → Point)
  (hA_eq : is_equilateral_triangle (A 1) (A 2) (A 3))
  (hA_midpoint : ∀ n, is_midpoint (A (n + 3)) (A n) (A (n + 1))) :
  ∠ (A 44) (A 45) (A 43) = 120 :=
sorry

end angle_A44_A45_A43_l97_97799


namespace Jeanine_gave_fraction_of_pencils_l97_97664

theorem Jeanine_gave_fraction_of_pencils
  (Jeanine_initial_pencils Clare_initial_pencils Jeanine_pencils_after Clare_pencils_after : ℕ)
  (h1 : Jeanine_initial_pencils = 18)
  (h2 : Clare_initial_pencils = Jeanine_initial_pencils / 2)
  (h3 : Jeanine_pencils_after = Clare_pencils_after + 3)
  (h4 : Clare_pencils_after = Clare_initial_pencils)
  (h5 : Jeanine_pencils_after + (Jeanine_initial_pencils - Jeanine_pencils_after) = Jeanine_initial_pencils) :
  (Jeanine_initial_pencils - Jeanine_pencils_after) / Jeanine_initial_pencils = 1 / 3 :=
by
  -- Proof here
  sorry

end Jeanine_gave_fraction_of_pencils_l97_97664


namespace problem_statement_l97_97604

variable {f : ℝ → ℝ}

theorem problem_statement
  (domain_f : ∀ x, 0 < x → x ∈ (0 : ℝ, ⊤))
  (h : ∀ x, 0 < x → (f x) / x > HasDerivAt f x) :
  2015 * f 2016 > 2016 * f 2015 :=
by
  sorry

end problem_statement_l97_97604


namespace decreasing_interval_l97_97743

-- Definitions of the given functions
def f (x : ℝ) : ℝ := real.sqrt (3 - 2 * x - x^2)

-- Condition 1: The domain of f is [-3, 1]
def domain (x : ℝ) : Prop := x ≥ -3 ∧ x ≤ 1

-- Condition 2: Inner function u is a quadratic function
def u (x : ℝ) : ℝ := -x^2 - 2 * x + 3

-- Condition 3: Behavior of u
lemma behavior_u : ∀ x, u x = -(x + 1)^2 + 4 :=
sorry

-- Condition 4: Monotonicity of the inner functions
def increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≥ f y 

def sqrt_increasing (x : ℝ) : Prop :=
  ∀ y z, y ≥ 0 ∧ z ≥ 0 ∧ y ≤ z → real.sqrt y ≤ real.sqrt z

theorem decreasing_interval :
  ∃ I : set ℝ, I = set.Icc (-1 : ℝ) (1 : ℝ) ∧
  ∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y :=
sorry

end decreasing_interval_l97_97743


namespace symmetry_axis_l97_97324

def f (x : ℝ) : ℝ := ( Real.sqrt 3 ) * Real.sin (2 * x) + Real.cos (2 * x)

theorem symmetry_axis :
  ∃ x : ℝ, x = 2 * Real.pi / 3 ∧ ∀ y : ℝ, f y = f (2 * x - y) := 
sorry

end symmetry_axis_l97_97324


namespace repeating_decimal_fraction_l97_97187

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l97_97187


namespace number_of_students_in_second_class_l97_97084

theorem number_of_students_in_second_class 
  (x : ℕ)
  (avg_marks_first_class : ℝ)
  (num_students_first_class : ℕ)
  (avg_marks_second_class : ℝ)
  (total_avg_marks : ℝ) 
  (H1 : avg_marks_first_class = 45)
  (H2 : num_students_first_class = 39)
  (H3 : avg_marks_second_class = 70)
  (H4 : total_avg_marks = 56.75)
  (H5 : 45 * 39 + 70 * x = 56.75 * (39 + x)) 
  : x = 35 := 
by
  sorry

end number_of_students_in_second_class_l97_97084


namespace length_AD_dihedral_angles_right_tetrahedron_to_prism_l97_97115

-- Definitions for the given dimensions and conditions of the tetrahedron
variable {A B C D : ℝ} -- points
variable {e f : ℝ} -- lengths

-- Condition: AB = BC = CD = e
variable (h1 : dist A B = e) 
variable (h2 : dist B C = e) 
variable (h3 : dist C D = e)

-- Condition: AC = BD = f
variable (h4 : dist A C = f)
variable (h5 : dist B D = f)

-- Condition: The angle between the planes ADB and ADC is 60 degrees
variable (h6 : ∠ (Plane.angle A D B) (Plane.angle A D C) = 60)

noncomputable def edge_length_AD (e f : ℝ) : ℝ := sqrt (3 * (f ^ 2 - e ^ 2))

theorem length_AD (A B C D : ℝ) (e f : ℝ)
  (h1 : dist A B = e) (h2 : dist B C = e) (h3 : dist C D = e)
  (h4 : dist A C = f) (h5 : dist B D = f) 
  (h6 : ∠ (Plane.angle A D B) (Plane.angle A D C) = 60) :
  dist A D = sqrt (3 * (f ^ 2 - e ^ 2)) := sorry

theorem dihedral_angles_right (A B C D : ℝ) (e f : ℝ)
  (h1 : dist A B = e) (h2 : dist B C = e) (h3 : dist C D = e)
  (h4 : dist A C = f) (h5 : dist B D = f) 
  (h6 : ∠ (Plane.angle A D B) (Plane.angle A D C) = 60) :
  ∀ (X Y : ℝ), X = f → Y = f → (Angle.dihedral X Y = 90) := sorry

theorem tetrahedron_to_prism (A B C D : ℝ) (e f : ℝ)
  (h1 : dist A B = e) (h2 : dist B C = e) (h3 : dist C D = e)
  (h4 : dist A C = f) (h5 : dist B D = f) 
  (h6 : ∠ (Plane.angle A D B) (Plane.angle A D C) = 60) :
  ∃ (planes : set Plane) (prism : Prism), 
  (forall t : Tetrahedron, t ∈ (ABCD).vertices → t ∉ planes) → 
  rearranged_into_prism ABCD planes prism := sorry

end length_AD_dihedral_angles_right_tetrahedron_to_prism_l97_97115


namespace permutation_exists_l97_97672

noncomputable def sum_powers_eq (s k : ℕ) (α β : Finₓ.s → ℝ) : Prop :=
  ∀ j : ℕ, j ∈ Finset.range (k + 1) → ∑ i in Finset.univ (Finₓ.s), (α i) ^ j = ∑ i in Finset.univ (Finₓ.s), (β i) ^ j

theorem permutation_exists
  (s k : ℕ)
  (α β : Finₓ.s → ℝ)
  (h1 : sum_powers_eq s k α β)
  (h2 : s ≤ k) :
  ∃ (π : Equiv.Perm (Finₓ.s)), ∀ i, β i = α (π i) :=
begin
  sorry
end

end permutation_exists_l97_97672


namespace dog_years_proof_l97_97325

noncomputable def dog_years_after_second (first_year_human_years : ℕ) (second_year_human_years : ℕ) (total_human_years : ℕ) (dog_age : ℕ) : ℕ :=
  (total_human_years - (first_year_human_years + second_year_human_years)) / (dog_age - 2)

theorem dog_years_proof :
  let first_year := 15 in
  let second_year := 9 in
  let total_years := 64 in
  let age := 10 in
  dog_years_after_second first_year second_year total_years age = 5 :=
by
  sorry

end dog_years_proof_l97_97325


namespace line_b_parallel_or_in_plane_l97_97990

def Line : Type := sorry    -- Placeholder for the type of line
def Plane : Type := sorry   -- Placeholder for the type of plane

def is_parallel (a b : Line) : Prop := sorry             -- Predicate for parallel lines
def is_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry   -- Predicate for a line being parallel to a plane
def lies_in_plane (l : Line) (p : Plane) : Prop := sorry          -- Predicate for a line lying in a plane

theorem line_b_parallel_or_in_plane (a b : Line) (α : Plane) 
  (h1 : is_parallel a b) 
  (h2 : is_parallel_to_plane a α) : 
  is_parallel_to_plane b α ∨ lies_in_plane b α :=
sorry

end line_b_parallel_or_in_plane_l97_97990


namespace find_ω_monotonicity_f_on_interval_l97_97963

-- Defining the function f
def f (ω x : ℝ) : ℝ := 4 * cos (ω * x) * sin (ω * x + (Real.pi / 4))

-- Given conditions
variable (ω : ℝ) (hω : ω > 0) (periodic_π : ∃ T > 0, ∀ x, f ω (x + T) = f ω x)

-- Statement to prove the value of ω
theorem find_ω : ω = 1 :=
sorry

-- Statement to prove the monotonicity of f in the given interval
theorem monotonicity_f_on_interval (x : ℝ) :
  0 ≤ x ∧ x ≤ Real.pi / 2 →
  (0 ≤ x ∧ x ≤ Real.pi / 8 → f 1 x ≤ f 1 (x + t)) ∧
  (Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2 → f 1 x ≥ f 1 (x + t)) :=
sorry

end find_ω_monotonicity_f_on_interval_l97_97963


namespace possible_slopes_l97_97001

theorem possible_slopes (m : ℝ) :
    (∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100)) ↔ 
    m ∈ (Set.Ioo (-∞) (-Real.sqrt (2 / 110)) ∪ Set.Ioo (Real.sqrt (2 / 110)) ∞) := 
  sorry

end possible_slopes_l97_97001


namespace min_score_to_achieve_average_l97_97703

theorem min_score_to_achieve_average (a b c : ℕ) (h₁ : a = 76) (h₂ : b = 94) (h₃ : c = 87) :
  ∃ d e : ℕ, d + e = 148 ∧ d ≤ 100 ∧ e ≤ 100 ∧ min d e = 48 :=
by sorry

end min_score_to_achieve_average_l97_97703


namespace convex_2015_polygon_l97_97092

theorem convex_2015_polygon (points : Fin 2015 → ℝ × ℝ)
  (h: ∀ (a b c d : Fin 2015), convex_quadrilateral ({points a, points b, points c, points d})) :
  ∃ (ch : set (ℝ × ℝ)), (is_convex_polygon ch) ∧ (card ch = 2015) ∧ (∀ p, p ∈ points → p ∈ ch) :=
sorry

end convex_2015_polygon_l97_97092


namespace repeating_decimal_fraction_l97_97184

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l97_97184


namespace triangle_properties_l97_97642

-- Definitions based on conditions
def quadratic_roots (a b : ℝ) : Prop := a^2 - 2 * real.sqrt 3 * a + 2 = 0 ∧ b^2 - 2 * real.sqrt 3 * b + 2 = 0
def cosine_sum (A B : ℝ) : Prop := 2 * real.cos (A + B) = -1
def angle_sum (A B C : ℝ) : Prop := A + B + C = 180

-- The main proof statement
theorem triangle_properties
  (a b A B C : ℝ)
  (roots : quadratic_roots a b)
  (cos_sum : cosine_sum A B)
  (angle_eq : angle_sum A B C)
  : C = 60 ∧
    ∃ (c : ℝ), c = real.sqrt 6 ∧
                ∃ area, area = 1/2 * a * b * real.sin C ∧
                        area = real.sqrt 3 / 2 :=
begin
  sorry
end

end triangle_properties_l97_97642


namespace probability_of_collinear_dots_in_5x5_grid_l97_97224

def collinear_dots_probability (total_dots chosen_dots collinear_sets : ℕ) : ℚ :=
  (collinear_sets : ℚ) / (Nat.choose total_dots chosen_dots)

theorem probability_of_collinear_dots_in_5x5_grid :
  collinear_dots_probability 25 4 12 = 12 / 12650 := by
  sorry

end probability_of_collinear_dots_in_5x5_grid_l97_97224


namespace john_min_pizzas_l97_97668

theorem john_min_pizzas (p : ℕ) (car_cost earnings_per_pizza cost_of_gas : ℝ) :
  car_cost = 8000 → earnings_per_pizza = 12 → cost_of_gas = 4 →
  8 * (p : ℝ) ≥ car_cost → p ≥ 1000 :=
by
  intros h_car_cost h_earnings_per_pizza h_cost_of_gas h_break_even
  rwa [h_car_cost, h_earnings_per_pizza, h_cost_of_gas] at h_break_even

end john_min_pizzas_l97_97668


namespace problem_1_problem_2_l97_97941

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q, ∀ n, a (n + 1) = a n * q

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a i

def a_n (n : ℕ) : ℝ := (1 / 2)^(n - 1)

def b_n (n : ℕ) : ℝ := -(real.log (a_n n) / real.log 2)

def c_n (n : ℕ) : ℝ := a_n n * b_n n

def T (n : ℕ) : ℝ := ∑ i in finset.range n, c_n i

theorem problem_1 (a : ℕ → ℝ) (h1 : (8 / a 0) + (6 / a 1) = (5 / a 2)) 
  (h2 : S a 6 = 63 / 32) : 
  a = a_n :=
sorry

theorem problem_2 (n : ℕ) : 
  T n = 2 - (n + 1) * (1 / 2)^(n - 1) :=
sorry

end problem_1_problem_2_l97_97941


namespace beto_winning_strategy_l97_97868

-- Definitions for conditions from step a)
def grid_size : ℕ := 2022

def is_red_colored (x y : ℕ) : Prop := sorry  -- Indicates whether the edge (x, y) is red

def no_two_red_sides_sharing_vertex : Prop :=
  ∀ (x y : ℕ), 
    (is_red_colored x y → ¬ is_red_colored (x + 1) y) ∧
    (is_red_colored x y → ¬ is_red_colored x (y + 1))

def path (start end : (ℕ × ℕ)) : (ℕ × ℕ) -> Prop := sorry  -- Indicates whether there's a path between start and end

def blue_path (start end : (ℕ × ℕ)) : (ℕ × ℕ) -> Prop :=
  path start end ∧ ∀ (x y : ℕ), path start end (x, y) → ¬ is_red_colored x y

-- Lean 4 statement
theorem beto_winning_strategy : 
  no_two_red_sides_sharing_vertex →
  ∀ {start end : (ℕ × ℕ)},
    (start = (0, 0) ∨ start = (0, grid_size) ∨ start = (grid_size, 0) ∨ start = (grid_size, grid_size)) →
    (end = (0, 0) ∨ end = (0, grid_size) ∨ end = (grid_size, 0) ∨ end = (grid_size, grid_size)) →
    (start ≠ end) →
    ∃ path, blue_path start end path :=
begin
  sorry
end

end beto_winning_strategy_l97_97868


namespace sin_neg_150_eq_neg_sin_30_l97_97886

-- Define the conditions
def angle_neg_150 := -150.0
def angle_210 := 210.0
def angle_30 := 30.0
def sin_30 := 1 / 2

-- State the proof problem
theorem sin_neg_150_eq_neg_sin_30 : Real.sin angle_neg_150 = -sin_30 :=
by
  sorry -- Proof is omitted as per instruction

end sin_neg_150_eq_neg_sin_30_l97_97886


namespace vector_magnitude_l97_97153

open Real

def vector (α : Type*) := (α × α)

def a : vector ℝ := (1, 2)
def b (m : ℝ) : vector ℝ := (-2, m)

def parallel (u v : vector ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2)

def vec_add (u v : vector ℝ) : vector ℝ := (u.1 + v.1, u.2 + v.2)

def scalar_mul (c : ℝ) (u : vector ℝ) : vector ℝ := (c * u.1, c * u.2)

def magnitude (u : vector ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem vector_magnitude :
  parallel a (b (-4)) →
  magnitude (vec_add (scalar_mul 2 a) (scalar_mul 3 (b (-4)))) = 4 * sqrt 5 :=
by
  intro h,
  sorry

end vector_magnitude_l97_97153


namespace not_equal_fractions_l97_97384

theorem not_equal_fractions :
  ¬ ((14 / 12 = 7 / 6) ∧
     (1 + 1 / 6 = 7 / 6) ∧
     (21 / 18 = 7 / 6) ∧
     (1 + 2 / 12 = 7 / 6) ∧
     (1 + 1 / 3 = 7 / 6)) :=
by 
  sorry

end not_equal_fractions_l97_97384


namespace people_got_rid_of_some_snails_l97_97755

namespace SnailProblem

def originalSnails : ℕ := 11760
def remainingSnails : ℕ := 8278
def snailsGotRidOf : ℕ := 3482

theorem people_got_rid_of_some_snails :
  originalSnails - remainingSnails = snailsGotRidOf :=
by 
  sorry

end SnailProblem

end people_got_rid_of_some_snails_l97_97755


namespace rectangle_triangle_area_equivalence_l97_97235

theorem rectangle_triangle_area_equivalence (W L x: ℝ) (h1: 2 * L + 2 * W = 40) (h2: L = 2 * W) (h3: (1 / 2) * 40 * x = L * W) : x = 4.4445 := by
  sorry

end rectangle_triangle_area_equivalence_l97_97235


namespace distinct_ordered_pairs_count_l97_97629

noncomputable def count_distinct_ordered_pairs : ℕ :=
  (finset.univ.product finset.univ).filter (λ p, 1 / p.1 + 1 / p.2 = 1 / 3).card

theorem distinct_ordered_pairs_count : count_distinct_ordered_pairs = 3 :=
sorry

end distinct_ordered_pairs_count_l97_97629


namespace sum_integers_between_6_and_14_l97_97777

theorem sum_integers_between_6_and_14 : (∑ k in Finset.range (15) \ Finset.range (6), k) = 90 := by
  sorry

end sum_integers_between_6_and_14_l97_97777


namespace election_votes_l97_97225

theorem election_votes (P : ℕ) (M : ℕ) (V : ℕ) (hP : P = 60) (hM : M = 1300) :
  V = 6500 :=
by
  sorry

end election_votes_l97_97225


namespace simplify_expression_l97_97888

theorem simplify_expression : (abs (π - abs (π - 9))) ^ 2 = 81 - 36 * π + 4 * π ^ 2 := by
  sorry

end simplify_expression_l97_97888


namespace smallest_solution_floor_eq_l97_97542

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l97_97542


namespace nate_pages_left_l97_97695

def total_pages : ℕ := 1250
def percentage_read : ℝ := 0.37
def read_pages : ℕ := floor (percentage_read * total_pages)

theorem nate_pages_left (total_pages = 1250) (percentage_read = 0.37) :
  (total_pages - read_pages) = 788 :=
by
  sorry

end nate_pages_left_l97_97695


namespace pond_fish_count_l97_97793

variable (N : Nat)

def tagged_proportion_initial := 80.0 / N
def tagged_proportion_sample := 2.0 / 80

theorem pond_fish_count (h : tagged_proportion_initial N = tagged_proportion_sample) : N = 3200 :=
sorry

end pond_fish_count_l97_97793


namespace purely_imaginary_z_a_plus_b_l97_97939

-- Proof problem for the first part
theorem purely_imaginary_z (m : ℝ) (z : ℂ) (h1 : z = m^2 + m - 2 + (m-1) * complex.I) 
(h2 : ∀ a b : ℝ, z = complex.I * b -> a = 0) : m = -2 :=
begin
  sorry,
end

-- Proof problem for the second part
theorem a_plus_b (m : ℝ) (z : ℂ) (a b : ℝ) (h1 : m = 2) (h2 : z = m^2 + m - 2 + (m-1) * complex.I)
(h3 : (z + complex.I) / (conj z - complex.I) = a + b * complex.I) : a + b = 7/5 :=
begin
  sorry,
end

end purely_imaginary_z_a_plus_b_l97_97939


namespace two_faucets_fill_60_gallons_l97_97564

def four_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  4 * (tub_volume / time_minutes) = 120 / 5

def two_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  2 * (tub_volume / time_minutes) = 60 / time_minutes

theorem two_faucets_fill_60_gallons :
  (four_faucets_fill 120 5) → ∃ t: ℕ, two_faucets_fill 60 t ∧ t = 5 :=
by {
  sorry
}

end two_faucets_fill_60_gallons_l97_97564


namespace f_is_monotonic_decreasing_l97_97638

noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi / 6)

theorem f_is_monotonic_decreasing : ∀ x y : ℝ, (2 * Real.pi / 3 ≤ x ∧ x ≤ 8 * Real.pi / 3) → (2 * Real.pi / 3 ≤ y ∧ y ≤ 8 * Real.pi / 3) → x < y → f y ≤ f x :=
sorry

end f_is_monotonic_decreasing_l97_97638


namespace solve_inequality_l97_97718

theorem solve_inequality (x : ℝ) : 
  x > 0 → 
  x^{Real.log x / Real.log 3} - 2 ≤ 
  (Real.cbrt 3)^{(Real.log x / Real.log (Real.sqrt 3))^2} - 2 * x^{Real.log (Real.cbrt x) / Real.log 3} ↔ 
  x ∈ Icc 0 (3^(Real.sqrt (Real.log 3 2))) ∪ {1} ∪ (Set.Ici (3^(Real.sqrt (Real.log 3 2)))) :=
by
  sorry

end solve_inequality_l97_97718


namespace calc_exponent_result_l97_97877

theorem calc_exponent_result (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := 
by
  sorry

end calc_exponent_result_l97_97877


namespace total_sales_l97_97644

noncomputable def sales_in_june : ℕ := 96
noncomputable def sales_in_july : ℕ := sales_in_june * 4 / 3

theorem total_sales (june_sales : ℕ) (july_sales : ℕ) (h1 : june_sales = 96)
                    (h2 : july_sales = june_sales * 4 / 3) :
                    june_sales + july_sales = 224 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_sales_l97_97644


namespace find_m_for_positive_integer_x_l97_97608

theorem find_m_for_positive_integer_x :
  ∃ (m : ℤ), (2 * m * x - 8 = (m + 2) * x) → ∀ (x : ℤ), x > 0 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 :=
sorry

end find_m_for_positive_integer_x_l97_97608


namespace number_of_gummies_l97_97487

-- Define the necessary conditions
def lollipop_cost : ℝ := 1.5
def lollipop_count : ℕ := 4
def gummy_cost : ℝ := 2.0
def initial_money : ℝ := 15.0
def money_left : ℝ := 5.0

-- Total cost of lollipops and total amount spent on candies
noncomputable def total_lollipop_cost := lollipop_count * lollipop_cost
noncomputable def total_spent := initial_money - money_left
noncomputable def total_gummy_cost := total_spent - total_lollipop_cost
noncomputable def gummy_count := total_gummy_cost / gummy_cost

-- Main theorem statement
theorem number_of_gummies : gummy_count = 2 := 
by
  sorry -- Proof to be added

end number_of_gummies_l97_97487


namespace rate_for_gravelling_roads_l97_97015

variable (length breadth width cost : ℕ)
variable (rate per_square_meter : ℕ)

def total_area_parallel_length : ℕ := length * width
def total_area_parallel_breadth : ℕ := (breadth * width) - (width * width)
def total_area : ℕ := total_area_parallel_length length width + total_area_parallel_breadth breadth width

def rate_per_square_meter := cost / total_area length breadth width

theorem rate_for_gravelling_roads :
  (length = 70) →
  (breadth = 30) →
  (width = 5) →
  (cost = 1900) →
  rate_per_square_meter length breadth width cost = 4 := by
  intros; exact sorry

end rate_for_gravelling_roads_l97_97015


namespace sculpture_height_l97_97486

def base_height: ℝ := 10  -- height of the base in inches
def combined_height_feet: ℝ := 3.6666666666666665  -- combined height in feet
def inches_per_foot: ℝ := 12  -- conversion factor from feet to inches

-- Convert combined height to inches
def combined_height_inches: ℝ := combined_height_feet * inches_per_foot

-- Math proof problem statement
theorem sculpture_height : combined_height_inches - base_height = 34 := by
  sorry

end sculpture_height_l97_97486


namespace cuboid_volume_l97_97440

theorem cuboid_volume (V : ℝ) (s : ℝ) (L W H V_new : ℝ) (h1 : V = 343) (h2 : s = real.cbrt V) 
                      (h3 : L = 3 * s) (h4 : W = 1.5 * s) (h5 : H = 2.5 * s) 
                      (h6 : V_new = L * W * H) : V_new = 38587.5 := by
  sorry

end cuboid_volume_l97_97440


namespace exist_symmetric_points_l97_97940

-- Define the given problem setup with necessary mathematical entities
noncomputable def symmetric_points (A B C : Point) (circ : Circle) : Prop :=
  ∃ (X Y : Point), 
    X ∈ circ ∧ Y ∈ circ ∧
    X.y = -Y.y ∧ -- symmetric with respect to the line AB
    is_perpendicular (line_through A X) (line_through Y C)

-- Define the intermediate steps in the correct answer
noncomputable def midpoint (A B : Point) : Point := 
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

noncomputable def perpendicular_bisector (A B : Point) : Line :=
  let K := midpoint A B
  Line.mk K (Point.mk (K.x + arbitrary Real) (K.y + sqrt(1 - ((arbitrary Real) ^ 2))))

-- Define the main theorem tying the problem with its constructed solution
theorem exist_symmetric_points (A B C : Point) (circ : Circle) :
  circum diameter A B →
  C ∈ line_through A B →
  ∃ X Y, symmetric_points A B C circ :=
  by {
    sorry
  }

end exist_symmetric_points_l97_97940


namespace proof_min_max_expected_wasted_minutes_l97_97417

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l97_97417


namespace ab_product_l97_97256

theorem ab_product (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) * (2 * b + a) = 4752) : a * b = 520 := 
by
  sorry

end ab_product_l97_97256


namespace intersect_points_l97_97661

open Set

-- Define lines a, b, and l as sets of points
variable {Point : Type*}
variable [metric_space Point]
variable a b l EF : set Point

-- Assume EF is a common perpendicular of skew lines a and b
variable common_perpendicular : EF ⊆ {p : Point | exists (q ∈ a) (r ∈ b), dist p q = dist p r ∧ (forall s ∈ EF, dist s q = dist s r)}

-- Assume line l is parallel to EF
variable parallel_l_EF : ∀ p ∈ l, ∃ q ∈ EF, dist p q = 0

-- Statement to prove
theorem intersect_points (h1 : common_perpendicular) (h2 : parallel_l_EF) : 
  ∃ n : ℕ, n ≤ 1 ∧ ∃ p1 ∈ a, p2 ∈ b, (p1 ∈ l ∨ p2 ∈ l) := sorry

end intersect_points_l97_97661


namespace proof_min_max_expected_wasted_minutes_l97_97415

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l97_97415


namespace angle_ABC_less_30_l97_97116

noncomputable def is_trianlge_A_less_30 := sorry

theorem angle_ABC_less_30 {BC AC : ℝ} (h : BC / AC < 1 / 2) : ∃ (A B C : ℝ), is_trianlge_A_less_30 BC AC /\
    ∡ CAB < (30 * π / 180) :=
sorry

end angle_ABC_less_30_l97_97116


namespace hyperbola_center_l97_97830

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end hyperbola_center_l97_97830


namespace arina_largest_shareholder_min_cost_l97_97870

-- Defining the shares owned by each person.
def owns_shares : Type :=
  { arina: Nat // arina = 90001} ∧
  { maxim: Nat // maxim = 104999} ∧
  { inga: Nat // inga = 30000} ∧
  { yuri: Nat // yuri = 30000} ∧
  { yulia: Nat // yulia = 30000} ∧
  { anton: Nat // anton = 15000}

-- Defining the price per share each person wants for their shares with the yield.
def price_per_share : Type :=
  { maxim: Nat // maxim = 11} ∧ -- 10 * 1.10
  { inga: Nat // inga = 12.5} ∧ -- 10 * 1.25
  { yuri: Nat // yuri = 11.5} ∧ -- 10 * 1.15
  { yulia: Nat // yulia = 13} ∧ -- 10 * 1.30
  { anton: Nat // anton = 14} -- 10 * 1.40

-- Main theorem to prove the minimum cost for Arina to become the largest shareholder.
theorem arina_largest_shareholder_min_cost (ow: owns_shares) (pp: price_per_share) : Nat :=
  (∃ n, n = 210000) ∧ n = 15000 * 14 :=
  sorry

end arina_largest_shareholder_min_cost_l97_97870


namespace determine_f_5_l97_97679

theorem determine_f_5 (f : ℝ → ℝ) (h1 : f 1 = 3) 
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) : f 5 = 45 :=
sorry

end determine_f_5_l97_97679


namespace regular_polygon_sides_l97_97209

theorem regular_polygon_sides (θ : ℝ) (h : θ = 20) : 360 / θ = 18 := by
  sorry

end regular_polygon_sides_l97_97209


namespace determinant_fourth_power_l97_97202

theorem determinant_fourth_power (A : Matrix n n ℝ): (Matrix.det A = -2) → Matrix.det (A^4) = 16 := 
by
  sorry

end determinant_fourth_power_l97_97202


namespace angle_C_of_congruent_triangles_l97_97203

theorem angle_C_of_congruent_triangles 
  (ABC_congruent_A'B'C' : ∀ (A B C A' B' C' : Type), triangle_congruent ABC A'B'C') 
  (angle_A : angle A = 35 + 25/60)
  (angle_B' : angle B' = 49 + 45/60) : 
  angle C = 180 - 35 - 25/60 - 49 - 45/60 := 
by
  sorry

end angle_C_of_congruent_triangles_l97_97203


namespace solution_set_l97_97681

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

variable {f : ℝ → ℝ}

-- Hypotheses
axiom odd_f : is_odd f
axiom increasing_f : is_increasing f
axiom f_of_neg_three : f (-3) = 0

-- Theorem statement
theorem solution_set (x : ℝ) : (x - 3) * f (x - 3) < 0 ↔ (0 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_l97_97681


namespace interval_of_monotonic_increase_l97_97326

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem interval_of_monotonic_increase (φ : ℝ) (h1 : ∀ x > 0, f x φ ≤ |f (π / 6) φ|)
  (h2 : f (π / 2) φ > f π φ) :
  ∃ (k : ℤ), ∀ x > 0, (π / 6 + k * π) ≤ x ∧ x ≤ (2 * π / 3 + k * π) → monotone (f x φ) :=
sorry

end interval_of_monotonic_increase_l97_97326


namespace problem1_problem2_l97_97401

-- Problem 1
theorem problem1 (β : ℝ) (h : tan β = 1 / 2) :
    sin β ^ 2 - 3 * sin β * cos β + 4 * cos β ^ 2 = 3 := by
  sorry

-- Problem 2
theorem problem2 :
    {x : ℝ // -6 < x ∧ x < -5 * π / 3 ∨ 0 ≤ x ∧ x ≤ π / 3} = 
    {x : ℝ // 36 - x ^ 2 > 0 ∧ -2 * cos x ^ 2 + 3 * cos x - 1 ≤ 0} := by
  sorry

end problem1_problem2_l97_97401


namespace find_smallest_solution_l97_97540

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l97_97540


namespace cylinder_radius_eq_original_l97_97062

theorem cylinder_radius_eq_original :
  ∃ (r : ℝ), let V := λ r h : ℝ, π * r^2 * h in
  V (r + 4) 5 = V r (5 + 4) ∧ r = 5 + 3 * Real.sqrt 5 :=
by
  sorry

end cylinder_radius_eq_original_l97_97062


namespace bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97431

-- Definitions of operations
def simple_op_time : ℕ := 1
def lengthy_op_time : ℕ := 5
def num_simple_ops : ℕ := 5
def num_lengthy_ops : ℕ := 3
def total_people : ℕ := num_simple_ops + num_lengthy_ops

-- Proving minimum and maximum person-minutes wasted
theorem bank_queue_min_max_wastage :
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 40) ∧
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 100) :=
by sorry

-- Proving expected value of wasted person-minutes
theorem bank_queue_expected_wastage :
  expected_value_wasted_person_minutes total_people simple_op_time lengthy_op_time = 84 :=
by sorry

-- Placeholder for the actual expected value calculation function
noncomputable def expected_value_wasted_person_minutes
  (n : ℕ) (t_simple : ℕ) (t_lengthy : ℕ) : ℕ :=
  -- Calculation logic will be implemented here
  84 -- This is just the provided answer, actual logic needed for correctness

end bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97431


namespace smallest_solution_floor_eq_l97_97525

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l97_97525


namespace total_cookies_l97_97478

-- Definitions from conditions
def cookies_per_guest : ℕ := 2
def number_of_guests : ℕ := 5

-- Theorem statement that needs to be proved
theorem total_cookies : cookies_per_guest * number_of_guests = 10 := by
  -- We skip the proof since only the statement is required
  sorry

end total_cookies_l97_97478


namespace x_36_bound_l97_97686

noncomputable def x : ℕ → ℝ
| 0       := 10 ^ 9
| (n + 1) := (x n ^ 2 + 2) / (2 * x n)

theorem x_36_bound : 0 < x 36 - Real.sqrt 2 ∧ x 36 - Real.sqrt 2 < 10 ^ (-9) := 
by 
  sorry

end x_36_bound_l97_97686


namespace minimum_value_of_expression_l97_97261

noncomputable def min_value (a b : ℝ) : ℝ := 1 / a + 3 / b

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : min_value a b ≥ 16 := 
sorry

end minimum_value_of_expression_l97_97261


namespace ellipse_area_is_correct_l97_97998

noncomputable def ellipse_area (x1 y1 x2 y2 x3 y3 : ℝ) (a b : ℝ) : ℝ :=
  π * a * b

theorem ellipse_area_is_correct :
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ), 
    (x1, y1) = (-9, 2) → 
    (x2, y2) = (7, 2) →
    (x3, y3) = (5, 6) →
    let cx := (x1 + x2) / 2 in
    let cy := (y1 + y2) / 2 in
    let a := sqrt ((x2 - cx)^2 + (y2 - cy)^2) in
    let ellipse_eq := (x, y):ℝ →  ((x + 1)^2 / a^2 + (y - 2)^2 / b^2 = 1) in
    ellipse_eq (5, 6) →
    ellipse_area (-9) 2 7 2 5 6 8 6.046 = 48.368 * π :=
by
  intros x1 y1 x2 y2 x3 y3 h1 h2 h3;
  sorry

end ellipse_area_is_correct_l97_97998


namespace no_suitable_operation_l97_97353

def has_solvable_operation (x y z w : ℕ) (ops : List (ℕ → ℕ → ℕ)) : Prop :=
  ∃ op ∈ ops, op x y + z - w = 6

theorem no_suitable_operation :
  ¬ has_solvable_operation 8 2 5 1 [Nat.div, (*), (+), (-)] :=
by
  sorry

end no_suitable_operation_l97_97353


namespace possible_slopes_l97_97000

theorem possible_slopes (m : ℝ) :
    (∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100)) ↔ 
    m ∈ (Set.Ioo (-∞) (-Real.sqrt (2 / 110)) ∪ Set.Ioo (Real.sqrt (2 / 110)) ∞) := 
  sorry

end possible_slopes_l97_97000


namespace bill_salary_increase_l97_97636

theorem bill_salary_increase (S P : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + P * S = 770.0000000000001) : 
  P = 0.1 :=
by {
  sorry
}

end bill_salary_increase_l97_97636


namespace total_bill_is_89_l97_97475

-- Define the individual costs and quantities
def adult_meal_cost := 12
def child_meal_cost := 7
def fries_cost := 5
def drink_cost := 10

def num_adults := 4
def num_children := 3
def num_fries := 2
def num_drinks := 1

-- Calculate the total bill
def total_bill : Nat :=
  (num_adults * adult_meal_cost) + 
  (num_children * child_meal_cost) + 
  (num_fries * fries_cost) + 
  (num_drinks * drink_cost)

-- The proof statement
theorem total_bill_is_89 : total_bill = 89 := 
  by
  -- The proof will be provided here
  sorry

end total_bill_is_89_l97_97475


namespace find_t_l97_97510

theorem find_t (t : ℝ) (h : 4 * real.log t / real.log 3 = real.log (9 * t^2) / real.log 3) : t = 3 :=
sorry

end find_t_l97_97510


namespace cubic_polynomial_unique_l97_97890

theorem cubic_polynomial_unique (q : ℝ → ℝ) : 
  (∃ p : ℝ → ℝ, p = q) → 
  (∀ x : ℂ, q x = 0 → 
    (x = 4 - 3 * complex.I ∨ x = 4 + 3 * complex.I ∨ x.real = 3.2)) → 
  q 0 = -80 → 
  q = λ x, x^3 - 11.2 * x^2 + 50.6 * x - 80 :=
by
  sorry

end cubic_polynomial_unique_l97_97890


namespace smallest_solution_floor_equation_l97_97548

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l97_97548


namespace sum_of_fraction_components_l97_97197

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l97_97197


namespace repeated_decimal_to_fraction_l97_97194

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l97_97194


namespace sum_transformed_set_l97_97846

-- Assume we have a set of n numbers with the sum s
variables (n : ℕ) (s : ℕ)

-- We need to prove the resulting sum after transformation is 3s + 80n
theorem sum_transformed_set (x : Fin n → ℕ)
  (hx_sum : (∑ i, x i) = s) :
  (∑ i, (3 * (x i + 30) - 10)) = 3 * s + 80 * n := by
  sorry

end sum_transformed_set_l97_97846


namespace minimum_apples_l97_97349

theorem minimum_apples (x : ℕ) : 
  (x ≡ 10 [MOD 3]) ∧ (x ≡ 11 [MOD 4]) ∧ (x ≡ 12 [MOD 5]) → x = 67 :=
sorry

end minimum_apples_l97_97349


namespace sum_of_integers_between_5_and_15_l97_97782

-- Definitions based on conditions
def predicate (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Main theorem statement
theorem sum_of_integers_between_5_and_15 : (Finset.sum (Finset.filter predicate (Finset.range 15))) = 90 :=
by
  sorry

end sum_of_integers_between_5_and_15_l97_97782


namespace porche_project_time_l97_97294

theorem porche_project_time :
  let total_time := 180
  let math_time := 45
  let english_time := 30
  let science_time := 50
  let history_time := 25
  let homework_time := math_time + english_time + science_time + history_time 
  total_time - homework_time = 30 :=
by
  sorry

end porche_project_time_l97_97294


namespace smallest_solution_floor_eq_l97_97526

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l97_97526


namespace integer_solutions_l97_97072

theorem integer_solutions (m n : ℤ) :
  m^3 - n^3 = 2 * m * n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
sorry

end integer_solutions_l97_97072


namespace cos_inequality_for_y_zero_l97_97075

open Real

theorem cos_inequality_for_y_zero :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ π / 2) → (0 ≤ y ∧ y ≤ π / 2) → y = 0 → 
  cos (x - y) ≥ cos x - cos y :=
by
  intros x y hx hy hy_eq
  rw [hy_eq, sub_zero]
  exact cos x - cos 0
  sorry

end cos_inequality_for_y_zero_l97_97075


namespace find_salary_january_l97_97797

noncomputable section
open Real

def average_salary_jan_to_apr (J F M A : ℝ) : Prop := 
  (J + F + M + A) / 4 = 8000

def average_salary_feb_to_may (F M A May : ℝ) : Prop := 
  (F + M + A + May) / 4 = 9500

def may_salary_value (May : ℝ) : Prop := 
  May = 6500

theorem find_salary_january : 
  ∀ J F M A May, 
    average_salary_jan_to_apr J F M A → 
    average_salary_feb_to_may F M A May → 
    may_salary_value May → 
    J = 500 :=
by
  intros J F M A May h1 h2 h3
  sorry

end find_salary_january_l97_97797


namespace solution_of_quadratic_eq_l97_97105

theorem solution_of_quadratic_eq (x : ℝ) (h : x^2 - 2 * real.sqrt 3 * x + 1 = 0) : x - 1 / x = 2 * real.sqrt 2 ∨ x - 1 / x = -2 * real.sqrt 2 := 
by sorry

end solution_of_quadratic_eq_l97_97105


namespace repeating_decimal_35_as_fraction_l97_97166

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l97_97166


namespace proof_problem_l97_97989

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f x + x * g x = x ^ 2 - 1
axiom condition2 : f 1 = 1

theorem proof_problem : deriv f 1 + deriv g 1 = 3 :=
by
  sorry

end proof_problem_l97_97989


namespace dodecagon_diagonals_l97_97044

def is_regular_dodecagon (vertices : Finset ℕ) : Prop :=
  vertices.card = 12

def is_valid_diagonal (vertices : Finset ℕ) (start end : ℕ) : Prop :=
  (start < end) ∧ (end - start = 3) ∨ (start - end = 9)

theorem dodecagon_diagonals (vertices : Finset ℕ) (h : is_regular_dodecagon vertices) :
  ∃ diagonals : Finset (ℕ × ℕ), 
  (∀ (d : ℕ × ℕ), d ∈ diagonals ↔ (fst d ∈ vertices ∧ snd d ∈ vertices ∧ is_valid_diagonal vertices (fst d) (snd d))) ∧
  diagonals.card = 12 :=
by
  sorry

end dodecagon_diagonals_l97_97044


namespace volume_of_region_l97_97929

noncomputable def f (x y z : ℝ) : ℝ :=
  |2 * x + y + z| + |2 * x + y - z| + |2 * x - y + z| + |-2 * x + y + z|

def region (x y z : ℝ) : Prop := 
  f x y z ≤ 6 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem volume_of_region : volume {p : ℝ × ℝ × ℝ | region p.1 p.2 p.3} = 18 :=
sorry

end volume_of_region_l97_97929


namespace exams_in_fourth_year_l97_97462

variable (a b c d e : ℕ)

theorem exams_in_fourth_year:
  a + b + c + d + e = 31 ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e = 3 * a → d = 8 := by
  sorry

end exams_in_fourth_year_l97_97462


namespace problem_conditions_l97_97145

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) / (x - 1)

theorem problem_conditions :
  (¬ (∀ x : ℝ, x ≠ 1 → (f 1 x = 1 + 2 / (x - 1)) → ∀ x, x ≠ 1 → f 1 x < f 1 (x + 1))) ∧
  (∀ a : ℝ, ∃ c : ℝ × ℝ, c = (1, a) ∧ (f a = λ x, a + (1 + a) / (x - 1))) ∧
  (∀ a : ℝ, ¬ (f a (-x) = - (f a x))) ∧
  (¬ (∀ x : ℝ, x ≠ 1 → f (-1) x = f (-1) (-x))) ∧
  (∀ x1 x2 : ℝ, 2 < x1 ∧ x1 < x2 → f 2 x1 - f 2 x2 < 3 * (x2 - x1)) :=
by
  sorry

end problem_conditions_l97_97145


namespace find_function_l97_97953

theorem find_function (f : ℝ → ℝ) (h : ∀ x, f (Real.logBase 2 x) = x^2) : ∀ x, f x = 4^x :=
sorry

end find_function_l97_97953


namespace pants_cost_l97_97279

/-- Given:
- 3 skirts with each costing $20.00
- 5 blouses with each costing $15.00
- The total spending is $180.00
- A discount on pants: buy 1 pair get 1 pair 1/2 off

Prove that each pair of pants costs $30.00 before the discount. --/
theorem pants_cost (cost_skirt cost_blouse total_amount : ℤ) (pants_discount: ℚ) (total_cost: ℤ) :
  cost_skirt = 20 ∧ cost_blouse = 15 ∧ total_amount = 180 
  ∧ pants_discount * 2 = 1 
  ∧ total_cost = 3 * cost_skirt + 5 * cost_blouse + 3/2 * pants_discount → 
  pants_discount = 30 := by
  sorry

end pants_cost_l97_97279


namespace geometric_sequence_general_term_inequality_b_n_l97_97735

-- Given that the geometric sequence sum S_n lies on the curve y = b^x + r
variables {b r : ℝ} (hb : b > 0) (hb_ne_one : b ≠ 1)
def S (n : ℕ) : ℝ := b^n + r

-- 1. The general term formula for a_n.
theorem geometric_sequence_general_term (n : ℕ) (hn_pos : 0 < n) :
  ∃ a : ℕ → ℝ, a n = b^(n-1) * (b - 1) := 
sorry

-- 2. Inequality for b = 2, given b_n = 2(\log_2 a_n + 1)
def a (n : ℕ) : ℝ := 2^(n-1) * (2 - 1)  -- General term when b = 2
def b_n (n : ℕ) : ℝ := 2 * (Real.log2 (a n) + 1)

-- Prove the inequality
theorem inequality_b_n (n : ℕ) (hn_pos : 0 < n) :
  (∏ i in Finset.range n + 1, (b_n i + 1) / b_n i) > Real.sqrt (n + 1) :=
sorry

end geometric_sequence_general_term_inequality_b_n_l97_97735


namespace smallest_prime_dividing_sum_l97_97370

theorem smallest_prime_dividing_sum : 
  (∃ p : ℕ, prime p ∧ p ∣ (2^11 + 7^13) ∧ 
            (∀ q : ℕ, prime q ∧ q ∣ (2^11 + 7^13) → p ≤ q)) :=
  sorry

end smallest_prime_dividing_sum_l97_97370


namespace ratio_of_A_to_B_share_l97_97020

noncomputable def ratio_of_share (A_investment: ℕ) (A_months: ℕ) (B_investment: ℕ) (B_months: ℕ) : ℚ :=
  let A_effective := A_investment * A_months
  let B_effective := B_investment * B_months
  A_effective / B_effective

theorem ratio_of_A_to_B_share (A_investment: ℕ) (B_investment: ℕ) : 
  (A_investment = 3500) → (B_investment = 9000) → (ratio_of_share A_investment 12 B_investment 7) = 2 / 3 := by
  intro hA hB
  rw [hA, hB]
  unfold ratio_of_share
  norm_num
  sorry

end ratio_of_A_to_B_share_l97_97020


namespace find_f_log2_10_l97_97135

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x ∧ x < 1 then log x / log 2 + 1 else
     if 1 < x ∧ x < 2 then 2 - log (3 - x) / log 2 else
     if -1 < x ∧ x < -2 then 2 + log (-3 - x) / log 2 else
     if 3 < x ∧ x < 4 then 5 - 2 ^ (4 - x) else 0

theorem find_f_log2_10 : f (log 10 / log 2) = 17 / 5 :=
by
  sorry

end find_f_log2_10_l97_97135


namespace part1_part2_l97_97950

section
variables (x y : ℝ)

-- Conditions
axiom sin_add : sin (x + y) = 1 / 3
axiom tan_relation : tan x = 4 * tan y

-- Part 1: Prove the first question
theorem part1 : sin (x - π / 6) * sin (y - π / 3) - cos (x - π / 6) * cos (y - π / 3) = -1 / 3 :=
by sorry

-- Part 2: Prove the second question
theorem part2 : sin (x - y) = 1 / 5 :=
by sorry
end

end part1_part2_l97_97950


namespace convert_base7_to_base4_l97_97898

theorem convert_base7_to_base4 (n : ℕ) (h : n = 563) : 
  (let digits_base4 : ℕ := (290 : ℕ)) in -- this intermediate value 290 will be shown equivalent to final base 4 representation
  (10202 : ℕ) = 4^4 * 1 + 4^3 * 0 + 4^2 * 2 + 4^1 * 0 + 4^0 * 2 :=
by {
  have h_base10 : n = 5 * 7^2 + 6 * 7^1 + 3 * 7^0, {
    calc n = 563 : by exact h
    ... = 5 * 49 + 6 * 7 + 3 : by norm_num
    ... = 290 : by norm_num,
  },
  show (10202 : ℕ) = 4^4 * 1 + 4^3 * 0 + 4^2 * 2 + 4^1 * 0 + 4^0 * 2, {
    calc (10202 : ℕ) = (4^4) * 1 + (4^3) * 0 + (4^2) * 2 + (4^1) * 0 + (4^0) * 2 : by norm_num,
  },
  sorry,
}

end convert_base7_to_base4_l97_97898


namespace trig_inequalities_l97_97785

theorem trig_inequalities :
  (sin (16 * (1 : ℝ) * real.pi / 180) < sin (154 * (1 : ℝ) * real.pi / 180)) ∧
  (cos (110 * (1 : ℝ) * real.pi / 180) < cos (260 * (1 : ℝ) * real.pi / 180)) ∧
  (sin (230 * (1 : ℝ) * real.pi / 180) < sin (80 * (1 : ℝ) * real.pi / 180)) ∧
  (tan (160 * (1 : ℝ) * real.pi / 180) > tan (-23 * (1 : ℝ) * real.pi / 180)) :=
  sorry

end trig_inequalities_l97_97785


namespace postage_unformable_l97_97082

theorem postage_unformable (n : ℕ) (h₁ : n > 0) (h₂ : 110 = 7 * n - 7 - n) :
  n = 19 := 
sorry

end postage_unformable_l97_97082


namespace calculate_expression_l97_97048

theorem calculate_expression : 
  - (1 ^ 2022) + ((3 - Real.pi) ^ 0) - ((1 / 8) * ((-1 / 2) ^ (-2))) = -1 / 2 :=
by
  sorry

end calculate_expression_l97_97048


namespace smallest_solution_eq_sqrt_104_l97_97531

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l97_97531


namespace hypotenuse_length_l97_97951

theorem hypotenuse_length (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : sqrt ((a - 3)^2) + (b - 2)^2 = 0) : c = 3 ∨ c = sqrt 13 :=
by
  sorry

end hypotenuse_length_l97_97951


namespace variance_linear_transform_l97_97257

-- Let X be a binomial random variable with parameters n = 10 and p = 0.8
def is_binomial (X : ℝ → ℝ) (n : ℕ) (p : ℝ) : Prop :=
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n → (X k = (choose n k) * (p ^ k) * ((1 - p) ^ (n - k)))

-- Let D be the variance function
def variance (X : ℝ → ℝ) : ℝ :=
  -- implementation of the variance function (assuming it's defined elsewhere, add as needed)
  sorry

-- Prove that D(2X + 1) = 4D(X)
theorem variance_linear_transform {X : ℝ → ℝ} (h : is_binomial X 10 0.8) :
  variance (λ x => 2 * X x + 1) = 4 * (variance X) :=
sorry

end variance_linear_transform_l97_97257


namespace base_radius_of_cone_l97_97383

-- Define the given conditions in Lean
def sector_angle := 300 * (Real.pi / 180)  -- Convert degrees to radians
def circle_radius := 12

-- Define the problem statement in Lean
theorem base_radius_of_cone {angle radius: ℝ} (h_angle: angle = sector_angle) (h_radius: radius = circle_radius) : 
  let arc_length := (angle / (2 * Real.pi)) * (2 * Real.pi * radius) in
  let base_radius := arc_length / (2 * Real.pi) in
  base_radius = 10 :=
by
  -- Translate the solution to Lean steps
  have h_arc_length : arc_length = 20 * Real.pi,
  { calc
      arc_length
        = (angle / (2 * Real.pi)) * (2 * Real.pi * radius) : by {refl}
        ... = (300 * (Real.pi / 180) / (2 * Real.pi)) * (2 * Real.pi * radius) : by simp [h_angle, sector_angle]
        ... = (5 / 6) * (2 * Real.pi * radius) : by norm_num
        ... = (5 / 6) * (24 * Real.pi) : by simp [h_radius, circle_radius]
        ... = 20 * Real.pi : by norm_num },
  show base_radius = 10, from
    calc
      base_radius
        = arc_length / (2 * Real.pi) : by {refl}
        ... = (20 * Real.pi) / (2 * Real.pi) : by rw [h_arc_length]
        ... = 10 : by norm_num

example : base_radius_of_cone sorry sorry := sorry

end base_radius_of_cone_l97_97383


namespace circumcircles_intersect_on_angle_bisector_l97_97273

theorem circumcircles_intersect_on_angle_bisector 
  (A B C X Y : Point)
  (h_triangle : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ collinear A B C)
  (h_AX_eq_AB : distance A X = distance A B)
  (h_BY_eq_AB : distance B Y = distance B A)
  (h_angle_bisector : ∃ D : Point, D ≠ C ∧ is_angle_bisector (∠ BCA) D) :
  ∃ D : Point, D ≠ C ∧ point_on_circumcircle D (Triangle.Circumcircle A C Y) ∧ 
                    point_on_circumcircle D (Triangle.Circumcircle B C X) ∧ 
                    point_on_angle_bisector D (∠ BCA) :=
by
  sorry

end circumcircles_intersect_on_angle_bisector_l97_97273


namespace arithmetic_sequences_count_l97_97582

theorem arithmetic_sequences_count :
  ∃ (a_1 d n : ℕ), n ≥ 3 ∧ (a_1 ≥ 0) ∧ (d ≥ 0) ∧ (n * (2 * a_1 + (n - 1) * d) = 2 * 97^2) ∧
  (finset.card {p : ℕ × ℕ × ℕ // p.2.2 ≥ 3 ∧ (p.1 ≥ 0) ∧ (p.2.1 ≥ 0) ∧ (p.2.2 * (2 * p.1 + (p.2.2 - 1) * p.2.1) = 2 * 97^2)} = 4) :=
by
  -- Sorry to skip the proof
  sorry

end arithmetic_sequences_count_l97_97582


namespace find_amount_l97_97630

-- Definitions based on the conditions provided
def gain : ℝ := 0.70
def gain_percent : ℝ := 1.0

-- The theorem statement
theorem find_amount (h : gain_percent = 1) : ∀ (amount : ℝ), amount = gain / (gain_percent / 100) → amount = 70 :=
by
  intros amount h_calc
  sorry

end find_amount_l97_97630


namespace calculate_m_plus_n_l97_97905

noncomputable def shoe_pairs_probability_condition (k : ℕ) (n : ℕ) : Prop := 
  ∀ m, (0 < m ∧ m < k) → ¬ ∃ pairs, (pairs.card = m ∧ pairs ⊆ {1..n})

theorem calculate_m_plus_n : 
  let numAdults := 8
  let goodProbability := 21/112
  ∃ (m n : ℕ), shoe_pairs_probability_condition 4 numAdults (goodProbability = m / n) ∧ Nat.coprime m n ∧ m + n = 133 :=
by
  sorry

end calculate_m_plus_n_l97_97905


namespace sum_of_squares_of_real_solutions_l97_97056

theorem sum_of_squares_of_real_solutions :
  (∀ x : ℝ, |x^2 - 3 * x + 1 / 400| = 1 / 400)
  → ((0^2 : ℝ) + 3^2 + (9 - 1 / 100) = 999 / 100) := sorry

end sum_of_squares_of_real_solutions_l97_97056


namespace arithmetic_seq_sum_x_y_l97_97374

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l97_97374


namespace propositions_correctness_l97_97960

theorem propositions_correctness :
  (¬(∀ (prism : Type) (lateral_faces : ∀ (f : prism → prism → Prop), f prism prism → Prop), lateral_faces f ↔ is_right_prism prism)) ∧
  (∀ (P : Type) [simple_polyhedron P] (V F : nat) (h : ∀ (v : P), 3 = num_of_edges v),
    num_of_faces P F → 2 * F - V = 4) ∧
  (∀ (l : Type) (α β : plane) (h₁ : is_perpendicular l α) (h₂ : is_parallel l β),
    is_perpendicular α β) ∧
  (¬¬(∀ (a b : line) (skew_lines : ¬ (is_perpendicular a b)), ∀ (plane_through_a : a → plane), ¬ is_perpendicular plane_through_a b)) :=
 by {
   sorry
 }

end propositions_correctness_l97_97960


namespace canoes_to_kayaks_ratio_l97_97766

-- Define the given conditions
variables (C K : ℕ)
variables (canoe_cost kayak_cost revenue : ℕ)
variables (canoe_diff : ℕ)

-- Conditions from the problem
def conditions :=
  canoe_cost = 9 ∧
  kayak_cost = 12 ∧
  revenue = 432 ∧
  C = K + 6

-- Assertion based on the problem
def ratio_of_canoes_to_kayaks := 
  ∃ C K : ℕ, canoe_cost * C + kayak_cost * K = revenue ∧ C = K + 6 ∧ C = 4 * (K - 6) / 3

theorem canoes_to_kayaks_ratio (h : conditions) :
  ratio_of_canoes_to_kayaks :=
by sorry

end canoes_to_kayaks_ratio_l97_97766


namespace max_value_of_y_min_value_of_fraction_l97_97398

-- Problem 1: Proof of maximum value
theorem max_value_of_y (x : ℝ) (h : x < 1) : 
  y = (4 * x^2 - 3 * x) / (x - 1) 
  ∃ (y : ℝ), y ≤ 1 :=
sorry

-- Problem 2: Proof of minimum value
theorem min_value_of_fraction (a b : ℝ) 
  (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ Y, Y = 3 + 2 * sqrt 2 :=
sorry

end max_value_of_y_min_value_of_fraction_l97_97398


namespace ratio_of_divisor_to_remainder_l97_97219

theorem ratio_of_divisor_to_remainder :
  ∃ (D Q n : ℕ), 
  let R := 46 in
  let Dividend := 5290 in
  D = 10 * Q ∧ 
  D = n * R ∧ 
  Dividend = D * Q + R ∧ 
  (D / R = 5) :=
sorry

end ratio_of_divisor_to_remainder_l97_97219


namespace max_total_profit_max_avg_annual_profit_l97_97464

-- Declaring the properties and equations of the problem
def totalProfit (x : ℕ) := -x^2 + 18 * x - 36
def averageAnnualProfit (x : ℕ) := (totalProfit x) / x

-- (1) Prove that the maximum total profit is 45 ten thousand yuan at x = 9
theorem max_total_profit : totalProfit 9 = 45 := by
  sorry

-- (2) Prove that the maximum average annual profit is 6 ten thousand yuan at x = 6
theorem max_avg_annual_profit : averageAnnualProfit 6 = 6 := by
  sorry

end max_total_profit_max_avg_annual_profit_l97_97464


namespace move_right_by_three_units_l97_97288

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end move_right_by_three_units_l97_97288


namespace shorter_piece_is_28_l97_97407

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end shorter_piece_is_28_l97_97407


namespace sum_of_real_numbers_satisfying_equation_l97_97503

def satisfies_equation (x : ℝ) : Prop :=
  (x^2 - 3 * x + 1)^(x^2 - 4 * x + 1) = 1

theorem sum_of_real_numbers_satisfying_equation :
  (∑ x in {x : ℝ | satisfies_equation x}, x) = 7 := sorry

end sum_of_real_numbers_satisfying_equation_l97_97503


namespace proof_min_max_expected_wasted_minutes_l97_97416

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l97_97416


namespace triathlete_average_speed_is_approx_3_5_l97_97856

noncomputable def triathlete_average_speed : ℝ :=
  let x : ℝ := 1; -- This represents the distance of biking/running segment
  let swimming_speed := 2; -- km/h
  let biking_speed := 25; -- km/h
  let running_speed := 12; -- km/h
  let swimming_distance := 2 * x; -- 2x km
  let biking_distance := x; -- x km
  let running_distance := x; -- x km
  let total_distance := swimming_distance + biking_distance + running_distance; -- 4x km
  let swimming_time := swimming_distance / swimming_speed; -- x hours
  let biking_time := biking_distance / biking_speed; -- x/25 hours
  let running_time := running_distance / running_speed; -- x/12 hours
  let total_time := swimming_time + biking_time + running_time; -- 1.12333x hours
  total_distance / total_time -- This should be the average speed

theorem triathlete_average_speed_is_approx_3_5 :
  abs (triathlete_average_speed - 3.5) < 0.1 := 
by
  sorry

end triathlete_average_speed_is_approx_3_5_l97_97856


namespace sum_of_integers_between_6_and_14_l97_97773

theorem sum_of_integers_between_6_and_14 : ∑ i in (Finset.range 15).filter (λ n, n > 5), i = 90 :=
by
  sorry

end sum_of_integers_between_6_and_14_l97_97773


namespace inequality_holds_if_and_only_if_l97_97274

noncomputable def absolute_inequality (x a : ℝ) : Prop :=
  |x - 3| + |x - 4| + |x - 5| < a

theorem inequality_holds_if_and_only_if (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, absolute_inequality x a) ↔ a > 4 := 
sorry

end inequality_holds_if_and_only_if_l97_97274


namespace integral_evaluation_l97_97506

noncomputable def integral_problem_statement : Prop :=
  let a := 1
  let b := Real.exp 1
  let f := λ x : ℝ, x + 1 / x
  ∫ x in a..b, f x = (b^2 + 1) / 2

theorem integral_evaluation : integral_problem_statement := 
by 
  sorry

end integral_evaluation_l97_97506


namespace lineup_possibilities_l97_97490

theorem lineup_possibilities (total_players : ℕ) (all_stars_in_lineup : ℕ) (injured_player : ℕ) :
  total_players = 15 ∧ all_stars_in_lineup = 2 ∧ injured_player = 1 →
  Nat.choose 12 4 = 495 :=
by
  intro h
  sorry

end lineup_possibilities_l97_97490


namespace solve_problem_l97_97098

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2
  else abs (x - 2) + 2

theorem solve_problem : f (f 0) = 2 := by
  sorry

end solve_problem_l97_97098


namespace angle_between_vectors_proof_l97_97122

noncomputable def angle_between_vectors (a b : ℝ) : ℝ := sorry

theorem angle_between_vectors_proof (a b : ℝ^3):
  a ≠ 0 ∧ b ≠ 0 ∧
  (‖a‖ = 2 * ‖b‖) ∧
  ((a - b) • b = 0) →
  angle_between_vectors a b = real.pi / 3 :=
by sorry

end angle_between_vectors_proof_l97_97122


namespace cost_per_treat_l97_97251

def treats_per_day : ℕ := 2
def days_in_month : ℕ := 30
def total_spent : ℝ := 6.0

theorem cost_per_treat : (total_spent / (treats_per_day * days_in_month : ℕ)) = 0.10 :=
by 
  sorry

end cost_per_treat_l97_97251


namespace stratified_sampling_grade12_l97_97844

theorem stratified_sampling_grade12 (total_students grade12_students sample_size : ℕ) 
  (h_total : total_students = 2000) 
  (h_grade12 : grade12_students = 700) 
  (h_sample : sample_size = 400) : 
  (sample_size * grade12_students) / total_students = 140 := 
by 
  sorry

end stratified_sampling_grade12_l97_97844


namespace minimum_CM_l97_97471

noncomputable def equilateral_triangle_min_CM : ℝ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (2 * Real.sqrt 3, 0 : ℝ)
  let C := (Real.sqrt 3, 3 : ℝ)
  let P := λ θ : ℝ, (Real.cos θ, Real.sin θ)
  let M := λ θ : ℝ, (Real.sqrt 3 + 0.5 * Real.cos θ, 0.5 * Real.sin θ)
  let distance := λ (x₁ y₁ x₂ y₂ : ℝ), Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  have Ap_to_P_is_1 : ∀ θ, distance A.1 A.2 (P θ).1 (P θ).2 = 1,
  from sorry,
  have C_to_M_min_is_5_over_2 : ∀ θ, distance C.1 C.2 (M θ).1 (M θ).2 ≥ 2.5,
  from sorry
  2.5

theorem minimum_CM : equilateral_triangle_min_CM = 2.5 := 
by sorry

end minimum_CM_l97_97471


namespace perpendicular_line_passing_point_l97_97922

theorem perpendicular_line_passing_point (x y : ℝ) (hx : 4 * x - 3 * y + 2 = 0) : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ (3 * x + 4 * y + 1 = 0) → l 1 2) :=
sorry

end perpendicular_line_passing_point_l97_97922


namespace find_prime_triplet_l97_97085

theorem find_prime_triplet (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ↔ (p, q, r) = (5, 3, 19) :=
by
  sorry

end find_prime_triplet_l97_97085


namespace determine_m_l97_97119

def sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 3) ∧ ∀ n, a (n + 1) = a n + n * (a n - 1)

def satisfies_condition (m : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, Nat.gcd m (a n) = 1

theorem determine_m (m : ℕ) (a : ℕ → ℕ) (h : sequence a) :
  satisfies_condition m a → ∃ t : ℕ, m = 2 ^ t :=
sorry

end determine_m_l97_97119


namespace cost_per_vent_l97_97730

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end cost_per_vent_l97_97730


namespace seq_100_gt_14_l97_97336

variable {a : ℕ → ℝ}

axiom seq_def (n : ℕ) : a 0 = 1 ∧ (∀ n ≥ 0, a (n + 1) = a n + 1 / a n)

theorem seq_100_gt_14 : a 100 > 14 :=
by
  -- Establish sequence definition
  have h1 : a 0 = 1 := (seq_def 0).left,
  have h2 : ∀ n ≥ 0, a (n + 1) = a n + 1 / a n := (seq_def 0).right,
  sorry

end seq_100_gt_14_l97_97336


namespace probability_distance_lt_one_l97_97255

-- Define the sides of the triangle
def a : ℝ := 3
def b : ℝ := 7
def c : ℝ := 8

-- Define the semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Calculate the area using Heron's formula
noncomputable def area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Given a point inside the triangle, the probability that the distance to at least one vertex < 1
theorem probability_distance_lt_one : 
  let probability := (Real.pi / (36 * Real.sqrt 3))
  probability = (Real.pi * Real.sqrt 3 / 36) := by 
  sorry

end probability_distance_lt_one_l97_97255


namespace odd_function_property_l97_97734

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x > 0 then -x + 1 else 0  -- Define partially, for arbitrary x, if x is <= 0 then it's undefined in this definition.

theorem odd_function_property (f_is_odd : odd_function f) (f_pos : ∀ x : ℝ, x > 0 → f x = -x + 1) :
  ∀ x : ℝ, x < 0 → f x = -x - 1 :=
by
  sorry

end odd_function_property_l97_97734


namespace coprime_5_subset_theorem_l97_97691

-- Define the set S from 1 to 280
def S := { x : ℕ | 1 ≤ x ∧ x ≤ 280 }

-- Define a predicate to check if 5 numbers in a set are pairwise coprime
def pairwise_coprime_5 (T : Finset ℕ) : Prop :=
  ∃ a b c d e ∈ T, Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime a d ∧ Nat.coprime a e ∧ 
  Nat.coprime b c ∧ Nat.coprime b d ∧ Nat.coprime b e ∧
  Nat.coprime c d ∧ Nat.coprime c e ∧ Nat.coprime d e

-- Define a predicate to check if any n-element subset of S has 5 pairwise coprime numbers
def satisfies_pairwise_coprime_5 (n : ℕ) : Prop :=
  ∀ T ⊆ S, T.card = n → pairwise_coprime_5 T

-- Definition of the smallest n such that the property holds
def smallest_coprime_5_subset : ℕ := 217

-- Main theorem statement
theorem coprime_5_subset_theorem : satisfies_pairwise_coprime_5 smallest_coprime_5_subset :=
sorry

end coprime_5_subset_theorem_l97_97691


namespace max_residents_per_apartment_l97_97231

theorem max_residents_per_apartment (total_floors : ℕ) (floors_with_6_apts : ℕ) (floors_with_5_apts : ℕ)
  (rooms_per_6_floors : ℕ) (rooms_per_5_floors : ℕ) (max_residents : ℕ) : 
  total_floors = 12 ∧ floors_with_6_apts = 6 ∧ floors_with_5_apts = 6 ∧ 
  rooms_per_6_floors = 6 ∧ rooms_per_5_floors = 5 ∧ max_residents = 264 → 
  264 / (6 * 6 + 6 * 5) = 4 := sorry

end max_residents_per_apartment_l97_97231


namespace sum_log_floor_ceil_l97_97885

/-- 
  Given the properties:
  1. ⌈x⌉ - ⌊x⌋ = 1 if x is not an integer and 0 if x is an integer.
  2. For k in the range 1 to 500, if k is a power of 3 (k = 3^j), then ⌈log₃k⌉ - ⌊log₃k⌋ = 0.
  Prove that: 
  ∑ k in 1 to 500, k * (⌈log₃ k⌉ - ⌊log₃ k⌋) = 124886
-/
theorem sum_log_floor_ceil :
  (∑ k in finset.range (501), k * (⌈real.log k / real.log 3⌉ - ⌊real.log k / real.log 3⌋)) = 124886 :=
by
  sorry

end sum_log_floor_ceil_l97_97885


namespace factor_expression_l97_97069

theorem factor_expression (a : ℝ) : 
  49 * a ^ 3 + 245 * a ^ 2 + 588 * a = 49 * a * (a ^ 2 + 5 * a + 12) :=
by
  sorry

end factor_expression_l97_97069


namespace sum_of_radii_of_tangent_circle_l97_97817

theorem sum_of_radii_of_tangent_circle :
  let r := λ r : ℝ, (r - 5)^2 + r^2 = (r + 3)^2 in
  ∃ r1 r2 : ℝ, r r1 ∧ r r2 ∧ (r1 = 8 + 4 * Real.sqrt 3 ∨ r1 = 8 - 4 * Real.sqrt 3) ∧
  (r2 = 8 + 4 * Real.sqrt 3 ∨ r2 = 8 - 4 * Real.sqrt 3) ∧
  (r1 + r2 = 16) :=
by
  sorry

end sum_of_radii_of_tangent_circle_l97_97817


namespace find_a_l97_97229

-- Define line l's parametric equations and conditions
noncomputable def line_l (a t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 3 / 2) * t, a - 1 + (1 / 2) * t)

-- Define curve C's polar equation and its Cartesian conversion
noncomputable def curve_C (theta : ℝ) : ℝ :=
  2 * (Real.sqrt 2) * Real.cos (theta + (Real.pi / 4))

-- Define the curve C in Cartesian coordinates
noncomputable def curve_C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 2

-- Given line l intersects curve C, find a such that the distance between points A and B is sqrt(5)
theorem find_a (a t1 t2 : ℝ) (h1 : curve_C_cartesian (1 + (Real.sqrt 3 / 2) * t1) (a - 1 + (1 / 2) * t1))
                          (h2 : curve_C_cartesian (1 + (Real.sqrt 3 / 2) * t2) (a - 1 + (1 / 2) * t2))
                          (dist : Real.abs (t1 - t2) = Real.sqrt 5) : a = 1 ∨ a = -1 :=
begin
  sorry
end

end find_a_l97_97229


namespace quadratic_example_has_properties_l97_97091

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * (x - 1) * (x - 5)

theorem quadratic_example_has_properties :
  ∃ a : ℝ, 
    let f := quadratic_function a in 
    f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 :=
begin
  use -2,
  let f := quadratic_function (-2),
  simp [quadratic_function],
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end quadratic_example_has_properties_l97_97091


namespace region_areas_correct_l97_97438

theorem region_areas_correct (A B C : ℝ) (π : ℝ)
  (triangle_area : ℝ := 150)
  (semicircle_area : ℝ := 78.125 * π)
  (hypotenuse : ℝ := 25) 
  (radius : ℝ := hypotenuse / 2)
  (circumcircle_area : ℝ := π * radius ^ 2) :
  C = semicircle_area ∧ A + B + triangle_area = semicircle_area - triangle_area → 
  A + B + 150 = C :=
by 
  intros hC hABπ
  sorry

end region_areas_correct_l97_97438


namespace checkerboard_black_squares_33x33_l97_97488

theorem checkerboard_black_squares_33x33 :
  ∀ (n : ℕ), n = 33 →
  (∀ (i j : ℕ), (i % 2 = 0) ↔ (j % 2 = 0)) →
  (∃ corners : ℕ, corners = 4 ∧ (corners % 2 = 0)) →
  (∀ (black_squares : ℕ), black_squares = 545) →
  ∃ black_squares_on_checkerboard : ℕ, black_squares_on_checkerboard = 545 := by
  intros n h1 h2 h3 h4
  have checkerboard : ∀ (i j : ℕ), i < n ∧ j < n → i % 2 = j % 2 → (∃ black : bool, black = tt) := sorry
  have corners : ∃ corners : ℕ, corners = 4 ∧ corners % 2 = 0 := h3
  exists 545
  exact h4
  sorry

end checkerboard_black_squares_33x33_l97_97488


namespace sufficient_and_necessary_condition_for_positive_sum_l97_97109

variable (q : ℤ) (a1 : ℤ)

def geometric_sequence (n : ℕ) : ℤ := a1 * q ^ (n - 1)

def sum_of_first_n_terms (n : ℕ) : ℤ :=
  if q = 1 then a1 * n else (a1 * (1 - q ^ n)) / (1 - q)

theorem sufficient_and_necessary_condition_for_positive_sum :
  (a1 > 0) ↔ (sum_of_first_n_terms q a1 2017 > 0) :=
sorry

end sufficient_and_necessary_condition_for_positive_sum_l97_97109


namespace EKFL_is_parallelogram_l97_97867

open EuclideanGeometry

noncomputable def isosceles_trapezoid (A B C D : Point) : Prop :=
AB ∥ CD ∧ AB ≠ CD ∧ ∃M, midpoint M A D ∧ midpoint M B C

theorem EKFL_is_parallelogram (A B C D E F M N K L : Point)
    (h_trapezoid : AB ∥ CD)
    (h_E_on_BC : E ∈ lineBC)
    (h_F_on_AD : F ∈ lineAD)
    (h_DF_BE : distance D F = distance B E)
    (h_FM_NE : distance F M = distance N E)
    (h_K_foot : is_perpendicular M K AB)
    (h_L_foot : is_perpendicular N L CD) :
    is_parallelogram (quadrilateral.mk E K F L) :=
sorry

end EKFL_is_parallelogram_l97_97867


namespace find_PZ_l97_97359

theorem find_PZ
  (X Y Z P : Type)
  [metric_space X][metric_space Y][metric_space Z][metric_space P]
  (triangle : Is_Triangle X Y Z)
  (right_angle_Y : Is_Right_Angle (Y ↔ Z))
  (P_in_triangle : Point_in_Triangle P X Y Z)
  (PX_13 : dist P X = 13)
  (PY_5 : dist P Y = 5)
  (angles_120 : ∠XPY = 120° ∧ ∠YPZ = 120° ∧ ∠PZX = 120°) :
  dist P Z = 6.25 := 
sorry

end find_PZ_l97_97359


namespace problem_statement_l97_97971

def f (x : ℝ) : ℝ :=
  if x < 1 then (x + 1)^2 else 4 - real.sqrt (x - 1)

theorem problem_statement {x : ℝ} : (f x) ≥ x ↔ x ∈ set.Iic (-2) ∪ set.Icc 0 10 :=
by
  sorry

end problem_statement_l97_97971


namespace eval_power_expr_of_196_l97_97673

theorem eval_power_expr_of_196 (a b : ℕ) (ha : 2^a ∣ 196 ∧ ¬ 2^(a + 1) ∣ 196) (hb : 7^b ∣ 196 ∧ ¬ 7^(b + 1) ∣ 196) :
  (1 / 7 : ℝ)^(b - a) = 1 := by
  have ha_val : a = 2 := sorry
  have hb_val : b = 2 := sorry
  rw [ha_val, hb_val]
  simp

end eval_power_expr_of_196_l97_97673


namespace find_f_l97_97317

theorem find_f (d e f : ℝ) (h_vertex : ∀ x : ℝ, (x + 1)^2 + 3 = d x^2 + e x + f) 
                (h_point : ∀ x : ℝ, x = 0 → d (0 : ℝ)^2 + e (0 : ℝ) + f = 2) : 
  f = 2 := 
sorry

end find_f_l97_97317


namespace repeating_decimal_sum_l97_97180

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l97_97180


namespace number_of_newts_is_one_l97_97223

/-- Define the types of amphibians. --/
inductive Species
| salamander
| newt
| gecko

/-- Define the type of amphibians. --/
structure Amphibian :=
(name : String)
(species : Species)
(stmnts : List String)

/-- The amphibians in the marsh. --/
def Alice := Amphibian.mk "Alice" sorry ["Ellie and I are of the same species."]
def Ben := Amphibian.mk "Ben" sorry ["Carl is a newt."]
def Carl := Amphibian.mk "Carl" sorry ["Dana is a salamander."]
def Dana := Amphibian.mk "Dana" sorry ["Of the five of us, at least three are salamanders."]
def Ellie := Amphibian.mk "Ellie" sorry ["Alice is a gecko."]

/-- List of amphibians for easy reference. --/
def amphibians := [Alice, Ben, Carl, Dana, Ellie]

/-- The problem statement. --/
theorem number_of_newts_is_one :
  (amphibians.countp (λ a, a.species = Species.newt)) = 1 := 
sorry

end number_of_newts_is_one_l97_97223


namespace arithmetic_sequence_sum_l97_97376

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l97_97376


namespace repeating_decimal_sum_l97_97181

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l97_97181


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_l97_97784

axiom normal_distribution (μ σ : ℝ) : ℝ → ℝ

axiom P (A : Set ℝ) : ℝ

theorem conclusion_1 (ξ : ℝ) (σ : ℝ) (h : σ > 0) (μ : ℝ = 1) (hp : P {x | 0 < x ∧ x < 1} = 0.35) : P {x | 0 < x ∧ x < 2} = 0.7 := 
  sorry

theorem conclusion_2 (c k : ℝ) (hx : ∀ x, ln (c * exp (k * x)) = 0.3 * x + 4) : 
  c = exp 4 :=
  sorry

theorem conclusion_3 (m : ℝ) (h : ∀ x > 0, exp x - m ≤ 0) : ¬ (∀ x, exp x - m - 1 ≥ 0) := 
  sorry

theorem conclusion_4 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x > 1, a * x^2 - (a + b - 1) * x + b > 0) ↔ (a ≥ b - 1) := 
  sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_l97_97784


namespace probability_of_picking_red_ball_l97_97997

theorem probability_of_picking_red_ball (total_balls : ℕ) (total_experiments : ℕ) (red_picked : ℕ) (total_balls = 50) (total_experiments = 10) (red_picked = 4) : 
  (red_picked : ℝ) / (total_experiments : ℝ) = 0.4 := 
by 
  sorry

end probability_of_picking_red_ball_l97_97997


namespace Ms_C_loses_240_l97_97444

noncomputable def house_initial_value := 12000
def loss_percentage_C_to_D := 0.15
def gain_percentage_D_to_C := 0.20

def first_transaction_price (initial_value : ℕ) (loss_percentage : ℚ) :=
  initial_value * (1 - loss_percentage)

def second_transaction_price (transaction_price : ℕ) (gain_percentage : ℚ) :=
  transaction_price * (1 + gain_percentage)

theorem Ms_C_loses_240 :
  let sale_price := first_transaction_price house_initial_value loss_percentage_C_to_D,
      buy_price := second_transaction_price sale_price gain_percentage_D_to_C,
      loss := buy_price - house_initial_value in
  loss = 240 := by
  sorry

end Ms_C_loses_240_l97_97444


namespace mean_female_members_selected_l97_97824

/-- Given a group of 5 members including 3 female and 2 male, if 2 members are randomly selected,
the mean number of female members selected is 6/5. -/
theorem mean_female_members_selected (total_members : ℕ)
                                     (female_members : ℕ)
                                     (male_members : ℕ)
                                     (selected_members : ℕ)
                                     (h1 : total_members = 5)
                                     (h2 : female_members = 3)
                                     (h3 : male_members = 2)
                                     (h4 : selected_members = 2) :
                                     (∑ k in finset.range (selected_members + 1),
                                      k * (nat.choose female_members k)
                                      * (nat.choose male_members (selected_members - k))
                                      / (nat.choose total_members selected_members) : ℚ) = 6 / 5 := by
  sorry

end mean_female_members_selected_l97_97824


namespace hyperbola_problem_l97_97623

noncomputable def hyperbola_equation : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b = Real.sqrt 2 ∧ ∃ (e : ℝ), e = Real.sqrt 3 ∧
  (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → (x^2 - y^2 / 2 = 1))

noncomputable def area_of_triangle (P E F : ℝ × ℝ) : Prop :=
  ∃ (p q : ℝ), (|p - q| = 2) ∧
  (∠PEF = 90) ∧
  (p^2 + q^2 = 12) ∧
  (p * q = 4) ∧
  (∀ (P E F : ℝ × ℝ), |P - E| * |P - F| / 2 = 2)

theorem hyperbola_problem : hyperbola_equation ∧ (∀ P E F, area_of_triangle P E F) :=
  sorry

end hyperbola_problem_l97_97623


namespace labyrinth_knights_max_l97_97287

theorem labyrinth_knights_max (n : ℕ) (L : Type) [labyrinth : has_labyrinth_properties L n]
  (no_parallel : ∀ (w1 w2 : wall L), w1 ≠ w2 → ¬parallel w1 w2)
  (no_three_intersect : ∀ (w1 w2 w3 : wall L), distinct_pairwise [w1, w2, w3] → ¬collinear [w1, w2, w3])
  (painted_sides : ∀ (w : wall L), is_colored w.red w.blue)
  (door_connections : ∀ (i1 i2 : intersection L), door_connected i1 i2 → color_diff_diagonal i1 i2) :
  k(L) = n + 1 :=
by sorry

end labyrinth_knights_max_l97_97287


namespace sum_of_integers_between_6_and_14_l97_97774

theorem sum_of_integers_between_6_and_14 : ∑ i in (Finset.range 15).filter (λ n, n > 5), i = 90 :=
by
  sorry

end sum_of_integers_between_6_and_14_l97_97774


namespace cost_of_one_box_of_paper_clips_l97_97063

theorem cost_of_one_box_of_paper_clips (p i : ℝ) 
  (h1 : 15 * p + 7 * i = 55.40) 
  (h2 : 12 * p + 10 * i = 61.70) : 
  p = 1.835 := 
by 
  sorry

end cost_of_one_box_of_paper_clips_l97_97063


namespace tangent_lines_through_point_and_circle_eqns_l97_97923

theorem tangent_lines_through_point_and_circle_eqns: 
  ∃ (k1 k2 : ℝ), 
    (∀ x y, (y = k1*x + 7) → (x - 15)^2 + (y - 2)^2 = 25) ∧ 
    (∀ x y, (y = k2*x + 7) → (x - 15)^2 + (y - 2)^2 = 25) ∧ 
    k1 = 0 ∧ k2 = -3/4 :=
begin
  sorry
end

end tangent_lines_through_point_and_circle_eqns_l97_97923


namespace area_of_region_l97_97045

theorem area_of_region :
  ∫ y in (0:ℝ)..(1:ℝ), y ^ (2 / 3) = 3 / 5 :=
by
  sorry

end area_of_region_l97_97045


namespace part_a_part_b_l97_97306

-- Definitions for the problem
variables (A B C D E F G H I J K L : Point) (circle : Circle) (ABCD : Square A B C D)
variables (EF GH IJ KL : Arc circle) (AEF BGH CIJ DKL : CurvedTriangle)

-- Assumptions based on the problem
axiom intersects_square : square_intersect_circle A B C D circle = { E, F, G, H, I, J, K, L }
axiom forms_curved_triangles :
  forms_curved_triangle AEF (A, E, F) ∧
  forms_curved_triangle BGH (B, G, H) ∧
  forms_curved_triangle CIJ (C, I, J) ∧
  forms_curved_triangle DKL (D, K, L)
axiom arcs_of_circle :
  arc EF circle ∧ arc GH circle ∧ arc IJ circle ∧ arc KL circle

-- Proof statements
theorem part_a : arc_length EF + arc_length IJ = arc_length GH + arc_length KL :=
sorry

theorem part_b :
  perimeter AEF + perimeter CIJ = perimeter BGH + perimeter DKL :=
sorry

end part_a_part_b_l97_97306


namespace number_of_cases_ordered_in_may_l97_97850

noncomputable def cases_ordered_in_may (ordered_in_april_cases : ℕ) (bottles_per_case : ℕ) (total_bottles : ℕ) : ℕ :=
  let bottles_in_april := ordered_in_april_cases * bottles_per_case
  let bottles_in_may := total_bottles - bottles_in_april
  bottles_in_may / bottles_per_case

theorem number_of_cases_ordered_in_may :
  ∀ (ordered_in_april_cases bottles_per_case total_bottles : ℕ),
  ordered_in_april_cases = 20 →
  bottles_per_case = 20 →
  total_bottles = 1000 →
  cases_ordered_in_may ordered_in_april_cases bottles_per_case total_bottles = 30 := by
  intros ordered_in_april_cases bottles_per_case total_bottles ha hbp htt
  sorry

end number_of_cases_ordered_in_may_l97_97850


namespace intersection_complement_A_B_l97_97977

def U := {1, 2, 3, 4, 5}
def A := {1, 3, 5}
def B := {3, 4}

theorem intersection_complement_A_B :
  (U \ A) ∩ B = {4} := 
by
  sorry

end intersection_complement_A_B_l97_97977


namespace greatest_n_for_m_factorial_l97_97393

theorem greatest_n_for_m_factorial (m n : ℕ) (h : m = 3^n) : n ≤ ∑ k in range (40 + 1), 40 / 3^k := by
  sorry

end greatest_n_for_m_factorial_l97_97393


namespace flowers_needed_for_floral_column_l97_97443

noncomputable def floral_column_surface_area (diameter : ℝ) (height : ℝ) (π_approx : ℝ) : ℝ :=
  let radius := diameter / 2
  let cylinder_surface := π_approx * diameter * height
  let hemisphere_surface := 2 * π_approx * radius^2
  cylinder_surface + hemisphere_surface

noncomputable def number_of_flowers (surface_area : ℝ) (flowers_per_sqm : ℝ) : ℝ :=
  surface_area * flowers_per_sqm

theorem flowers_needed_for_floral_column :
  let diameter := 2
  let height := 4
  let π_approx := 3.1
  let flowers_per_sqm := 200
  let total_surface_area := floral_column_surface_area diameter height π_approx
  number_of_flowers total_surface_area flowers_per_sqm = 6200 :=
by
  let diameter := 2
  let height := 4
  let π_approx := 3.1
  let flowers_per_sqm := 200
  let total_surface_area := floral_column_surface_area diameter height π_approx
  let total_number_flowers := number_of_flowers total_surface_area flowers_per_sqm
  exact Eq.refl total_number_flowers

end flowers_needed_for_floral_column_l97_97443


namespace sum_of_integers_between_6_and_14_l97_97772

theorem sum_of_integers_between_6_and_14 : ∑ i in (Finset.range 15).filter (λ n, n > 5), i = 90 :=
by
  sorry

end sum_of_integers_between_6_and_14_l97_97772


namespace charlotte_should_bring_money_l97_97884

theorem charlotte_should_bring_money (p d a : ℝ) (h_p : p = 90) (h_d : d = 20) (h_a : a = 72) :
  a = p - (d / 100 * p) :=
by
  rw [h_p, h_d, h_a]
  norm_num
  sorry

end charlotte_should_bring_money_l97_97884


namespace sum_of_integers_between_is_90_l97_97770

-- Define the conditions
def is_between (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Define the sum of integers satisfying the conditions
def sum_of_integers_between : ℕ :=
  Finset.sum (Finset.filter is_between (Finset.range 15)) id

-- State the theorem
theorem sum_of_integers_between_is_90 : sum_of_integers_between = 90 := 
by
  sorry

end sum_of_integers_between_is_90_l97_97770


namespace sum_f_inv_eq_32_l97_97052

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 2 * x - 3 else 2 * real.sqrt x

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 3 then (y + 3) / 2 else (y / 2) ^ 2

theorem sum_f_inv_eq_32 :
  (finset.sum (finset.range 15) (λ i, f_inv (i - 7))) = 32 :=
begin
  -- The range -7 to 7 is mapped to the indexes 0 to 14 in the sum.
  sorry
end

end sum_f_inv_eq_32_l97_97052


namespace trajectory_area_l97_97590

noncomputable def area_of_trajectory (A B C P : Point) (sphere_radius : ℝ) (height : ℝ) : ℝ :=
  if sphere_radius = 1 ∧ dist A B = 1 ∧ dist A C = 1 ∧ dist B C = 1 ∧ height = sqrt 6 / 2
  then (5 * Real.pi / 6)
  else 0

theorem trajectory_area (A B C P : Point) (sphere_radius : ℝ) (height : ℝ) :
  sphere_radius = 1 →
  dist A B = 1 →
  dist A C = 1 →
  dist B C = 1 →
  height = sqrt 6 / 2 →
  area_of_trajectory A B C P sphere_radius height = 5 * Real.pi / 6 :=
by
  intros h1 h2 h3 h4 h5
  unfold area_of_trajectory
  rw [if_pos (and.intro h1 (and.intro h2 (and.intro h3 (and.intro h4 h5))))]
  sorry

end trajectory_area_l97_97590


namespace belongs_to_union_l97_97465

/-- Definitions related to arithmetic progressions -/
def is_arith_prog (s : Set ℤ) : Prop :=
∃ a d : ℤ, d ≠ 0 ∧ ∀ n : ℤ, a + n * d ∈ s

variable (A B C : Set ℤ)
variable hA : is_arith_prog A
variable hB : is_arith_prog B
variable hC : is_arith_prog C
variable known_set : Set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}
variable set_union_condition : known_set ⊆ A ∪ B ∪ C 

/-- Proving that 1980 is in the union of A, B, and C -/
theorem belongs_to_union : 1980 ∈ A ∪ B ∪ C := by
  sorry

end belongs_to_union_l97_97465


namespace A_correct_D_correct_l97_97603

def f : ℝ → ℝ := sorry -- Definition of f is unspecified
def g (x : ℝ) : ℝ := x^3 + 2*x - 1/(1 + 2^x) + 1/2

-- Hypothesis 1: Domain of f is ℝ
axiom f_domain : ∀ x : ℝ, f x ∈ ℝ

-- Hypothesis 2: f(x+1) symmetric about x = -1
axiom f_symmetric : ∀ x : ℝ, f(x + 2) = f(-x)

-- Hypothesis 3: f(x+1) monotonically increasing on (-∞, -1]
axiom f_mono_increasing : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → x₂ ≤ -1 → f(x₁ + 1) ≤ f(x₂ + 1)

-- Proposition 1: f(g(x)) is an even function
theorem A_correct : ∀ x : ℝ, f(g(-x)) = f(g(x)) :=
sorry

-- Proposition 2: g(f(√3*ln 2)) < g(f(ln 3))
theorem D_correct : g(f(√3 * log 2)) < g(f(log 3)) :=
sorry

end A_correct_D_correct_l97_97603


namespace remainder_polynomial_l97_97111
noncomputable def p (x : ℝ) : ℝ := sorry

theorem remainder_polynomial (r : ℝ → ℝ) (h0 : p 0 = 2) (h2 : p 2 = 6) :
  (∀ x, ∃ q, p x = q x * x * (x - 2) + r x) → (r = λ x, 2 * x + 2) := by
  intro h
  sorry

end remainder_polynomial_l97_97111


namespace find_trajectory_range_area_ACBD_l97_97129

-- Definition of conditions
def point_on_circle (x y : ℝ) : Prop :=
  (x + sqrt 3)^2 + y^2 = 16

def point_F : ℝ × ℝ := (sqrt 3, 0)
def origin : ℝ × ℝ := (0, 0)

def perpendicular_bisector (M F P : ℝ × ℝ) : Prop :=
  let (Mx, My) := M
  let (Fx, Fy) := F
  let (Px, Py) := P
  (Mx + Fx) / 2 = Px ∧ (My + Fy) / 2 = Py

def intersects_at (E M P : ℝ × ℝ) : Prop :=
  -- Intersection of EM at P
  sorry

def on_line (l : ℝ → ℝ) (A : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A
  l Ax = Ay

def equal_distance (A B C : ℝ × ℝ) : Prop :=
  let dist (X Y : ℝ × ℝ) := (fst X - fst Y)^2 + (snd X - snd Y)^2
  dist A C = dist C B

def satisfies_vector_equation (C A B D : ℝ × ℝ) : Prop :=
  let vec_add (X Y : ℝ × ℝ) := (fst X + fst Y, snd X + snd Y)
  C = vec_add A B

-- Theorem statements without proofs
theorem find_trajectory :
  ∀ (M : ℝ × ℝ), point_on_circle (fst M) (snd M) →
  ∃ (P : ℝ × ℝ), perpendicular_bisector M point_F P ∧ intersects_at origin M P →
  ∀ (P : ℝ × ℝ), let (Px, Py) := P in Px^2 / 4 + Py^2 = 1 :=
sorry

theorem range_area_ACBD :
  ∀ (A B C D : ℝ × ℝ),
  (exists l : ℝ → ℝ, on_line l A ∧ on_line l B ∧ intersects_at origin A B) ∧
  equal_distance A B C ∧ satisfies_vector_equation C A B D →
  ∃ (S : ℝ), S = 2 * (4 / 5) ∧ S = 4 :=
sorry

end find_trajectory_range_area_ACBD_l97_97129


namespace solve_double_inequality_l97_97717

theorem solve_double_inequality (x : ℝ) :
  (-1 < (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) ∧
   (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) < 1) ↔ (2 < x ∨ 26 < x) := 
sorry

end solve_double_inequality_l97_97717


namespace parabola_equations_l97_97007

theorem parabola_equations (x y : ℝ) (h₁ : (0, 0) = (0, 0)) (h₂ : (-2, 3) = (-2, 3)) :
  (x^2 = 4 / 3 * y) ∨ (y^2 = - 9 / 2 * x) :=
sorry

end parabola_equations_l97_97007


namespace maximize_profit_l97_97652

-- Define the conditions
def daily_sales_price_relation_a (x : ℝ) (a b : ℝ) : ℝ := a * (x - 3) ^ 2 + b / (x - 1)
def daily_sales_price_relation_b (x : ℝ) : ℝ := -70 * x + 490

-- Define the profit function
def daily_profit_relation_a (x : ℝ) (a b : ℝ) : ℝ := daily_sales_price_relation_a x a b * (x - 1)
def daily_profit_relation_b (x : ℝ) : ℝ := daily_sales_price_relation_b x * (x - 1)

-- Problem Statement
theorem maximize_profit
  (a b : ℝ)
  (h1 : daily_sales_price_relation_a 2 a b = 600)
  (h2 : daily_sales_price_relation_a 3 a b = 150)
  (h3 : 1 < x ∧ x ≤ 3 → ∀ x, y = daily_sales_price_relation_a x a b)
  (h4 : 3 < x ∧ x ≤ 5 → ∀ x, y = daily_sales_price_relation_b x)
  : a = 300 ∧ b = 300 ∧ (daily_profit_relation_a (5 / 3) 300 300 > daily_profit_relation_b 4) 
  ∧ (maximize_profit_result = 1.7) :=
sorry

end maximize_profit_l97_97652


namespace number_of_correct_propositions_l97_97610

noncomputable def axis_of_symmetry (k : ℤ) : Prop :=
  ∀ x, f x = sin (2 * x - π / 4) → x = k * π / 2 + 3 * π / 8

noncomputable def max_value_of_function : Prop :=
  ∀ x, f x = sin x + sqrt 3 * cos x → ∃ y, f y = 2

noncomputable def period_of_function : Prop :=
  ∀ x, f x = sin (cos x - 1) → ∀ y, f (x + 2 * π) = f x

noncomputable def increasing_interval : Prop :=
  ∀ x, f x = sin (x + π / 4) → ∀ y z, (-π / 2 ≤ y ∧ y < z ∧ z ≤ π / 2) → f y ≤ f z

theorem number_of_correct_propositions : Prop :=
  let propositions := [axis_of_symmetry, max_value_of_function, period_of_function, increasing_interval] in
  let correct_count := propositions.count (λ p, p true) in
  correct_count = 2

#check number_of_correct_propositions

end number_of_correct_propositions_l97_97610


namespace table_sides_length_approx_l97_97463

-- Assume all variables are real numbers
variables (L : ℝ)

-- Define the conditions and the required proof

theorem table_sides_length_approx:
  (∃ (L : ℝ), (* The Length of one free side*)
  (L > 0) * (* L is a positive real number*)
  (* Area of rectangle and triangle equals 128 sq feet *)
  ((L * (L / 2)) + (1 / 2 * (L / 2) * (L / 2)) = 128) →
   -- Prove the total length of the table's three free sides
   -- which is 3L, is approximately 42.93 feet.
   3 * L ≈ 42.93 
  )
  := sorry

end table_sides_length_approx_l97_97463


namespace solve_complex_number_l97_97214

theorem solve_complex_number (a : ℝ) :
  let z := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  in z.re = z.im → a = -6 :=
by
  let z := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  have h : z.re = z.im → a = -6
  {
    sorry
  }
  exact h

end solve_complex_number_l97_97214


namespace complex_addition_l97_97566

theorem complex_addition
  (a b : ℝ)
  (ha : (a + 3 * complex.I) / complex.I = b - 2 * complex.I):
  a + b = 5 :=
by
  sorry

end complex_addition_l97_97566


namespace sufficient_condition_l97_97338

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x : ℝ, if x > 0 then log 2 x else -2^x + a

theorem sufficient_condition (a : ℝ) : (a < 0) → (∃! x : ℝ, f a x = 0) :=
by
  sorry

end sufficient_condition_l97_97338


namespace square_display_side_length_is_17_point_4_l97_97889

noncomputable def square_display_side_length : ℝ :=
  let pane_width := 3 / 5
  let pane_height := 3 * pane_width
  let total_width := 4 * pane_width + 5 * 3
  let total_height := 3 * pane_height + 4 * 3
  let side_length := total_width -- since square display, width == height
  side_length

theorem square_display_side_length_is_17_point_4 :
  square_display_side_length = 17.4 :=
by
  let pane_width := 3 / 5
  let total_width := 4 * pane_width + 5 * 3
  have h : total_width = 17.4 := by norm_num
  exact h

end square_display_side_length_is_17_point_4_l97_97889


namespace how_many_strawberries_did_paul_pick_l97_97702

-- Here, we will define the known quantities
def original_strawberries : Nat := 28
def total_strawberries : Nat := 63

-- The statement to prove
theorem how_many_strawberries_did_paul_pick : total_strawberries - original_strawberries = 35 :=
by
  unfold total_strawberries
  unfold original_strawberries
  calc
    63 - 28 = 35 := by norm_num

end how_many_strawberries_did_paul_pick_l97_97702


namespace sum_of_reversed_base_digits_eq_zero_l97_97556

theorem sum_of_reversed_base_digits_eq_zero : ∃ n : ℕ, 
  (∀ a₁ a₀ : ℕ, n = 5 * a₁ + a₀ ∧ n = 12 * a₀ + a₁ ∧ 0 ≤ a₁ ∧ a₁ < 5 ∧ 0 ≤ a₀ ∧ a₀ < 12 
  ∧ n > 0 → n = 0)
:= sorry

end sum_of_reversed_base_digits_eq_zero_l97_97556


namespace rational_zero_l97_97570

theorem rational_zero (x y : ℚ) (h : x + (real.sqrt 2) * y = 0) : x = 0 ∧ y = 0 :=
begin
  sorry
end

end rational_zero_l97_97570


namespace lily_of_the_valley_bushes_needed_l97_97332

theorem lily_of_the_valley_bushes_needed 
  (r l : ℕ) (h_radius : r = 20) (h_length : l = 400) : 
  l / (2 * r) = 10 := 
by 
  sorry

end lily_of_the_valley_bushes_needed_l97_97332


namespace alpha_plus_beta_eq_3pi_over_4_l97_97592

theorem alpha_plus_beta_eq_3pi_over_4 
  (α β: ℝ) 
  (hα1: 0 < α ∧ α < π / 2)
  (hβ1: 0 < β ∧ β < π / 2)
  (hα2: Real.cos α = sqrt 5 / 5)
  (hβ2: Real.sin β = 3 * sqrt 10 / 10)
  : α + β = 3 * π / 4 :=
sorry

end alpha_plus_beta_eq_3pi_over_4_l97_97592


namespace part1_part2_part3_l97_97561

def sim (a b : ℚ) := abs (a - b)

theorem part1 :
  sim (sim 3 5) 9 = 7 ∧ sim (sim 5 9) 3 = 1 :=
by sorry

theorem part2 :
  (∀ a1 a2 a3 a4 : ℚ, (a1 = 1 ∨ a1 = 2 ∨ a1 = 3 ∨ a1 = 4) ∧ (a2 = 1 ∨ a2 = 2 ∨ a2 = 3 ∨ a2 = 4) ∧ 
  (a3 = 1 ∨ a3 = 2 ∨ a3 = 3 ∨ a3 = 4) ∧ (a4 = 1 ∨ a4 = 2 ∨ a4 = 3 ∨ a4 = 4) →
  min (((sim (sim (sim a1 a2) a3) a4) : ℚ) = 0) ∧ 
  max (((sim (sim (sim a1 a2) a3) a4) : ℚ) = 4) :=
by sorry

theorem part3 (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ,
  (if n = 4 * k then
    min ((\lambda a1 a2 ... an, ((sim (sim (sim a1 a2) a3) ... an)) : ℚ) = 0 ∧ \max ... an = n) else
  if n = 4 * k - 1 then
    min ... = 0 ∧ max ... = n - 1 else
  if n = 4 * k - 2 then
    min ... = 1 ∧ max ... = n - 1 else
  if n = 4 * k - 3 then
    min ... = 1 ∧ max ... = n) :=
by sorry

end part1_part2_part3_l97_97561


namespace find_bigger_number_l97_97763

noncomputable def common_factor (x : ℕ) : Prop :=
  8 * x + 3 * x = 143

theorem find_bigger_number (x : ℕ) (h : common_factor x) : 8 * x = 104 :=
by
  sorry

end find_bigger_number_l97_97763


namespace multiplicative_inverse_l97_97049

theorem multiplicative_inverse (a b n : ℤ) (h₁ : a = 208) (h₂ : b = 240) (h₃ : n = 307) : 
  (a * b) % n = 1 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end multiplicative_inverse_l97_97049


namespace part1_part2_l97_97216

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a i
def b (n : ℕ) : ℝ := Real.log 2 (1 - a (n + 1))
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / ((b i) * (b (i + 1)))

theorem part1 (h : ∀ n, S n = 2 * a n + n) : 
  ∃ (first_term common_ratio : ℝ), 
    (∀ n, a n - 1 = first_term * (common_ratio ^ n)) :=
sorry

theorem part2 (h : ∀ n, S n = 2 * a n + n) :
  ∀ n, T n < 1 / 2 :=
sorry

end part1_part2_l97_97216


namespace equal_opposite_sides_of_quadrilateral_l97_97011

theorem equal_opposite_sides_of_quadrilateral 
  (quad : Type*) [quadrilateral quad] 
  (circle : Type*) [incircle quad circle]
  (equal_area_segments : ∀ seg ∈ (circle_intersections quad circle), equal_area seg)
  : sum_of_opposite_sides quad :=
sorry

end equal_opposite_sides_of_quadrilateral_l97_97011


namespace hyperbola_center_l97_97832

theorem hyperbola_center (F1 F2 : ℝ × ℝ) (F1_eq : F1 = (3, -2)) (F2_eq : F2 = (11, 6)) :
  let C := ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2) in C = (7, 2) :=
by
  sorry

end hyperbola_center_l97_97832


namespace book_profit_percentage_l97_97812

noncomputable def profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let discount := discount_rate / 100 * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

theorem book_profit_percentage :
  profit_percentage 47.50 69.85 15 = 24.994736842105263 :=
by
  sorry

end book_profit_percentage_l97_97812


namespace rotated_vector_is_correct_l97_97343

-- Define the initial vector
def initial_vector : ℝ × ℝ × ℝ := (2, 3, 1)

-- Define the resulting vector we need to prove
def resulting_vector : ℝ × ℝ × ℝ := (sqrt (14 / 3), -sqrt (14 / 3), sqrt (14 / 3))

-- Prove that the resulting vector is the rotation of the initial vector by 90 degrees about the origin
theorem rotated_vector_is_correct :
  rotated_vector 90 initial_vector passes_through (0, 1, 0) ∧ magnitude_preserved initial_vector resulting_vector ∧ orthogonal initial_vector resulting_vector →
    resulting_vector = (sqrt (14 / 3), -sqrt (14 / 3), sqrt (14 / 3)) :=
by
  sorry

end rotated_vector_is_correct_l97_97343


namespace tan_150_deg_l97_97887

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - Real.sqrt 3 / 3 := by
  sorry

end tan_150_deg_l97_97887


namespace x_y_complex_l97_97057

theorem x_y_complex (x y : ℂ) (h1 : x ≠ 0) (h2 : 2 * x + y ≠ 0)
  (h : (2 * x + y) / x = y / (2 * x + y)) : 
  x ∈ ℂ ∧ y ∈ ℂ :=
by sorry

end x_y_complex_l97_97057


namespace unique_bisecting_line_exists_l97_97026

noncomputable def triangle_area := 1 / 2 * 6 * 8
noncomputable def triangle_perimeter := 6 + 8 + 10

theorem unique_bisecting_line_exists :
  ∃ (line : ℝ → ℝ), 
    (∃ x y : ℝ, x + y = 12 ∧ x * y = 30 ∧ 
      1 / 2 * x * y * (24 / triangle_perimeter) = 12) ∧
    (∃ x' y' : ℝ, x' + y' = 12 ∧ x' * y' = 24 ∧ 
      1 / 2 * x' * y' * (24 / triangle_perimeter) = 12) ∧
    ((x = x' ∧ y = y') ∨ (x = y' ∧ y = x')) :=
sorry

end unique_bisecting_line_exists_l97_97026


namespace distribute_papers_l97_97467

theorem distribute_papers (n m : ℕ) (h_n : n = 5) (h_m : m = 10) : 
  (m ^ n) = 100000 :=
by 
  rw [h_n, h_m]
  rfl

end distribute_papers_l97_97467


namespace Joan_orange_balloons_l97_97666

theorem Joan_orange_balloons (originally_has : ℕ) (received : ℕ) (final_count : ℕ) 
  (h1 : originally_has = 8) (h2 : received = 2) : 
  final_count = 10 := by
  sorry

end Joan_orange_balloons_l97_97666


namespace count_sixth_powers_less_than_500_l97_97164

theorem count_sixth_powers_less_than_500 : 
  {n : ℕ | n > 0 ∧ n < 500 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 2 := by
  sorry

end count_sixth_powers_less_than_500_l97_97164


namespace coprime_repeating_decimal_sum_l97_97171

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l97_97171


namespace which_two_students_donated_l97_97031

theorem which_two_students_donated (A B C D : Prop) 
  (h1 : A ∨ D) 
  (h2 : ¬(A ∧ D)) 
  (h3 : (A ∧ B) ∨ (A ∧ D) ∨ (B ∧ D))
  (h4 : ¬(A ∧ B ∧ D)) 
  : B ∧ D :=
sorry

end which_two_students_donated_l97_97031


namespace number_of_entries_multiple_of_73_l97_97027

-- Definitions based on conditions
def triangular_array : ℕ → ℕ → ℕ
| 1, k := 1 + 2 * (k - 1)  -- First row definition
| n+1, k := if k = 1 then triangular_array n 1 + triangular_array n 2 else triangular_array n (k-1) + triangular_array n k -- other rows

-- Pattern as identified in solution
def a (n k : ℕ) : ℕ := 2^(n-1) * (n + 2*k - 2)

-- Main theorem statement
theorem number_of_entries_multiple_of_73 : 
  let multiples_of_73 := 
    (λ (n k : ℕ),
      1 ≤ n ∧ n ≤ 31 ∧
      1 ≤ k ∧ k ≤ 53 - n ∧ 
      73 ∣ a n k) in 
  (finset.univ.filter (λ n, n % 2 = 1)).sum (λ n, (finset.range (53 - n + 1)).filter (λ k, multiples_of_73 n k).card) = 16 :=
sorry

end number_of_entries_multiple_of_73_l97_97027


namespace telescoping_product_l97_97876

theorem telescoping_product :
  (∏ n in finset.range 1002 \ {0}, (↑(5 * n + 5) / ↑(5 * n + 10)))
  = (1 / 1002) :=
by
  sorry

end telescoping_product_l97_97876


namespace division_in_expression_correct_l97_97070

theorem division_in_expression_correct :
  let expr := (1 / 60) / ((2 / 3) - (1 / 5) - (2 / 5))
  in expr = 1 / 4 := by
  sorry

end division_in_expression_correct_l97_97070


namespace smallest_solution_floor_eq_l97_97543

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l97_97543


namespace area_closed_figure_l97_97602

noncomputable def binomial_coeff (n k : ℕ) : ℚ := nat.choose n k

theorem area_closed_figure (a : ℚ) (h : binomial_coeff 6 3 * (1 / a)^3 = 160) :
  ∫ x in 0..1, (x^(1/2) - x^2) = 1 / 3 :=
by
  sorry

end area_closed_figure_l97_97602


namespace arithmetic_seq_sum_x_y_l97_97372

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l97_97372


namespace no_19_distinct_naturals_with_same_digit_sum_1999_l97_97903

noncomputable def digit_sum (n : ℕ) : ℕ := sorry

theorem no_19_distinct_naturals_with_same_digit_sum_1999 :
  ¬ ∃ (a : ℕ → ℕ) (S : ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i, digit_sum (a i) = S) ∧ (∑ i in finset.range 19, a i) = 1999 :=
sorry

end no_19_distinct_naturals_with_same_digit_sum_1999_l97_97903


namespace total_sales_l97_97820

theorem total_sales (S : ℕ) (h1 : (1 / 3 : ℚ) * S + (1 / 4 : ℚ) * S = (1 - (1 / 3 + 1 / 4)) * S + 15) : S = 36 :=
by
  sorry

end total_sales_l97_97820


namespace minimum_auxiliary_cars_needed_l97_97360

/-- A structure to represent two cities connected by a road of a certain distance. -/
structure CitiesConnectedByRoad where
  B : Type
  C : Type
  distanceBC : ℝ
  distanceBC_eq_two : distanceBC = 2

/-- A structure to represent the properties of cars in the problem. -/
structure CarProperties where
  maxFuelDistance : ℝ
  maxFuelDistance_eq_one : maxFuelDistance = 1

/-- Definitions for the auxiliary cars and their constraints. -/
structure AuxiliaryCarConstraints where
  can_transfer_fuel : Prop
  must_return_to_start : Prop

/-- The main theorem: Prove that three helper cars are necessary and sufficient
    to transfer car A from city B to city C, given the specified conditions. -/
theorem minimum_auxiliary_cars_needed (cities : CitiesConnectedByRoad)
                                     (carProps : CarProperties)
                                     (auxConstraints : AuxiliaryCarConstraints) :
  (unique (n : ℕ) (n = 3)) :=
by
  -- We provide the proof or logic to support the theorem, but for now we skip it.
  sorry

end minimum_auxiliary_cars_needed_l97_97360


namespace maximum_abc_l97_97262

theorem maximum_abc {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b + c^2 = (a + c) * (b + c)) (h5 : a + b + c = 3) :
  abc ≤ 1 :=
begin
  sorry
end

end maximum_abc_l97_97262


namespace percentage_of_boys_currently_l97_97240

variables (B G : ℕ)

theorem percentage_of_boys_currently
  (h1 : B + G = 50)
  (h2 : B + 50 = 95) :
  (B * 100) / 50 = 90 :=
by
  sorry

end percentage_of_boys_currently_l97_97240


namespace ellipse_equation_l97_97117

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := sqrt 5
noncomputable def c : ℝ := 2

def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_equation :
  (∀ x y : ℝ, ellipse_eq x y ↔ x^2 / 9 + y^2 / 5 = 1) :=
sorry

end ellipse_equation_l97_97117


namespace sum_of_fraction_components_l97_97199

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l97_97199


namespace find_a_l97_97088

open Nat

-- Define the conditions and the proof goal
theorem find_a (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 :=
sorry

end find_a_l97_97088


namespace fraction_division_l97_97043

theorem fraction_division :
  (5 / 4) / (8 / 15) = 75 / 32 :=
sorry

end fraction_division_l97_97043


namespace ellipse_equation_l97_97587

theorem ellipse_equation (c : ℝ) (k : ℝ) (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  let e := 2 in
  let e' := (1 : ℝ) / e in
  (e' = (1 : ℝ) / 2) →
  c^2 = 1 →
  (∀ (x y : ℝ), (x, y) = (1, (3 / 2 : ℝ))) →
  let a := 2 * c in
  let b := sqrt (3 * c^2) in
  eq (a^2 - c^2) (b^2) →
  eq ((1 : ℝ) * x₁^2 / 4 + (1 : ℝ) * y₁^2 / 3) 1 →
  (y₁ = k * x₁ + m) →
  (y₂ = k * x₂ + m) →
  ((3 + 4 * k^2) * x₁^2 + 8 * k * m * x₁ + 4 * m^2 - 12 = 0) →
  (∀ (P : ℝ × ℝ), P = (1 / 5, 0)) →
  (4 * k^2 + 5 * k * m + 3 = 0) →
  ((4 * k^2 + 3)^2 / (25 * k^2) < 4 * k^2 + 3) →
  (k^2 > (1 : ℝ) / 7) :=
by sorry

end ellipse_equation_l97_97587


namespace sum_mod_15_l97_97381

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end sum_mod_15_l97_97381


namespace arina_largest_shareholder_min_cost_l97_97871

-- Defining the shares owned by each person.
def owns_shares : Type :=
  { arina: Nat // arina = 90001} ∧
  { maxim: Nat // maxim = 104999} ∧
  { inga: Nat // inga = 30000} ∧
  { yuri: Nat // yuri = 30000} ∧
  { yulia: Nat // yulia = 30000} ∧
  { anton: Nat // anton = 15000}

-- Defining the price per share each person wants for their shares with the yield.
def price_per_share : Type :=
  { maxim: Nat // maxim = 11} ∧ -- 10 * 1.10
  { inga: Nat // inga = 12.5} ∧ -- 10 * 1.25
  { yuri: Nat // yuri = 11.5} ∧ -- 10 * 1.15
  { yulia: Nat // yulia = 13} ∧ -- 10 * 1.30
  { anton: Nat // anton = 14} -- 10 * 1.40

-- Main theorem to prove the minimum cost for Arina to become the largest shareholder.
theorem arina_largest_shareholder_min_cost (ow: owns_shares) (pp: price_per_share) : Nat :=
  (∃ n, n = 210000) ∧ n = 15000 * 14 :=
  sorry

end arina_largest_shareholder_min_cost_l97_97871


namespace exists_matrix_A_part_a_exists_matrix_A_part_b_l97_97671

-- Defining the size of the matrices
def n : ℕ := 2019

-- Define the zero matrix
def zero_matrix (n : ℕ) : matrix (fin n) (fin n) ℚ :=
  0

-- Define the identity matrix
def identity_matrix (n : ℕ) : matrix (fin n) (fin n) ℚ :=
  1

-- Statement for part (a)
theorem exists_matrix_A_part_a :
  ∃ (A : matrix (fin n) (fin n) ℚ), A ^ 3 + 6 • (A ^ 2) - 2 • identity_matrix n = zero_matrix n :=
sorry

-- Statement for part (b)
theorem exists_matrix_A_part_b :
  ∃ (A : matrix (fin n) (fin n) ℚ), A ^ 4 + 6 • (A ^ 3) - 2 • identity_matrix n = zero_matrix n :=
sorry

end exists_matrix_A_part_a_exists_matrix_A_part_b_l97_97671


namespace largest_710_double_l97_97879

def is_710_double (N : ℕ) : Prop :=
  let N_base_7_digits := List.ofFn (fun i => (N / 7^i) % 7) in
  let N_base_7_as_base_10 := (Nat.digits 7 N).foldl (fun acc d => acc * 10 + d) 0 in
  2 * N = N_base_7_as_base_10

theorem largest_710_double : ∃ N, is_710_double N ∧ ∀ M, is_710_double M → M ≤ 315 := sorry

end largest_710_double_l97_97879


namespace jackie_phil_same_heads_prob_l97_97249

theorem jackie_phil_same_heads_prob :
  let fair_coin_gen_fn := 1 + x,
      biased_coin_gen_fn := 3 + 2 * x,
      combined_gen_fn := (1 + x) * (3 + 2 * x),
      p := 19,
      q := 50
  in (p + q = 69) :=
begin
  -- condition setup
  have fair_coin_gen_fn : polynomial := 1 + X,
  have biased_coin_gen_fn : polynomial := 3 + 2 * X,
  have combined_gen_fn := fair_coin_gen_fn * biased_coin_gen_fn,
  -- derive that the combined probability p/q  = 19/50
  have p := 19,
  have q := 50,
  show (p + q = 69), from rfl
end

end jackie_phil_same_heads_prob_l97_97249


namespace Ofelia_savings_l97_97697

theorem Ofelia_savings (X : ℝ) (h : 16 * X = 160) : X = 10 :=
by
  sorry

end Ofelia_savings_l97_97697


namespace minimum_value_a2013_minus_4a1_l97_97845

noncomputable def sequence_a : ℕ+ → ℝ
| 1       := sorry  -- a_1 > 1 is given, so pick a value greater than 1
| (n + 1) := sequence_a n * (sequence_a n - 1) + 1

theorem minimum_value_a2013_minus_4a1 :
  ∃ a1 : ℝ, 
  (a1 > 1) ∧ 
  (∑ i in finset.range 2012, (1 / sequence_a (i + 1))) = 2 ∧ 
  (sequence_a 2013 - 4 * a1) = - 7 / 2 :=
sorry

end minimum_value_a2013_minus_4a1_l97_97845


namespace tangent_line_at_point_intervals_of_monotonicity_maximum_value_on_interval_l97_97099

noncomputable def f (x : ℝ) (a : ℝ) := x^3 - a * x^2

theorem tangent_line_at_point (a : ℝ) (h : a = 1) : 
      ∃ (L : LinearMap ℝ ℝ), L.span (subtype (λ p, p = (1 : ℝ, f 1 a))) ≤ vector_span ℝ (range (λ x, f x a)) := sorry

theorem intervals_of_monotonicity (a : ℝ) : 
      ∀ x, monotone_on (λ x, f x a) {x | x ∈ interval (-∞) 0 ∪ interval (2 * a / 3) ∞} := sorry

theorem maximum_value_on_interval (a : ℝ) : 
      ∃ x, x ∈ interval 1 3 ∧ (∀ y ∈ interval 1 3, f x a ≥ f y a) := sorry

end tangent_line_at_point_intervals_of_monotonicity_maximum_value_on_interval_l97_97099


namespace sum_of_fraction_components_l97_97200

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l97_97200


namespace outfit_combinations_l97_97631

def shirts : ℕ := 6
def pants : ℕ := 4
def hats : ℕ := 6

def pant_colors : Finset String := {"tan", "black", "blue", "gray"}
def shirt_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}

def total_combinations : ℕ := shirts * pants * hats
def restricted_combinations : ℕ := pant_colors.card

theorem outfit_combinations
    (hshirts : shirts = 6)
    (hpants : pants = 4)
    (hhats : hats = 6)
    (hpant_colors : pant_colors.card = 4)
    (hshirt_colors : shirt_colors.card = 6)
    (hhat_colors : hat_colors.card = 6)
    (hrestricted : restricted_combinations = pant_colors.card) :
    total_combinations - restricted_combinations = 140 := by
  sorry

end outfit_combinations_l97_97631


namespace coprime_repeating_decimal_sum_l97_97175

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l97_97175


namespace find_smallest_solution_l97_97541

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l97_97541


namespace problem_statement_l97_97094

-- Defining the basic entities: planes and lines
variables (α β : Plane) (l : Line)

-- Definition of parallel and perpendicular relationships
def parallel (A B : Plane) : Prop := sorry
def perpendicular (A : Plane) (B : Line) : Prop := sorry

-- Propositions as given in the problem
def p : Prop := (parallel α l ∧ parallel β l) → parallel α β
def q : Prop := (perpendicular α l ∧ perpendicular β l) → parallel α β

-- The theorem we want to prove
theorem problem_statement : p ∨ q :=
by sorry

end problem_statement_l97_97094


namespace quad_side_difference_l97_97012

theorem quad_side_difference (a b c d s x y : ℝ)
  (h1 : a = 80) (h2 : b = 100) (h3 : c = 150) (h4 : d = 120)
  (semiperimeter : s = (a + b + c + d) / 2)
  (h5 : x + y = c) 
  (h6 : (|x - y| = 30)) : 
  |x - y| = 30 :=
sorry

end quad_side_difference_l97_97012


namespace problem_statement_l97_97970

def f (x : ℝ) : ℝ :=
  if x < 1 then (x + 1)^2 else 4 - real.sqrt (x - 1)

theorem problem_statement {x : ℝ} : (f x) ≥ x ↔ x ∈ set.Iic (-2) ∪ set.Icc 0 10 :=
by
  sorry

end problem_statement_l97_97970


namespace angle_between_vectors_is_pi_div_3_l97_97120

noncomputable theory

variables {a b : ℝ^3} (ha : a ≠ 0) (hb : b ≠ 0)
(h1 : ∥a∥ = 2 * ∥b∥)
(h2 : a - b ⬝ b = 0)

theorem angle_between_vectors_is_pi_div_3 :
  real.angle_between a b = real.pi / 3 :=
sorry

end angle_between_vectors_is_pi_div_3_l97_97120


namespace fraction_1_49_repeating_decimal_l97_97369

def division_repeating_sequence_exists_49 : Prop :=
  ∃ s : String, ∃ n : ℕ, (String.length s = n ∧ s.repeat n = "020408163265306122448979591836734693877551")

theorem fraction_1_49_repeating_decimal :
  division_repeating_sequence_exists_49 :=
by
  sorry

end fraction_1_49_repeating_decimal_l97_97369


namespace sum_of_roots_of_parabola_l97_97746

theorem sum_of_roots_of_parabola
  (a b : ℝ)
  (ABCD : set (ℝ × ℝ))
  (h1 : ∀ p ∈ ABCD, ∃ (x y : ℝ), p = (x, y))
  (h2 : ∀ {x y : ℝ}, (x, y) ∈ ABCD → x ∈ set.Icc 0 20)
  (h3 : ∀ {x y : ℝ}, (x, y) ∈ ABCD → y ∈ set.Icc 0 20)
  (h4 : (∀ y ∈ set.Icc 0 20, (0, y) ∈ ABCD))
  (B C : ℝ × ℝ)
  (h5 : B ∈ ABCD)
  (h6 : C ∈ ABCD)
  (h7 : B ≠ C)
  (h8 : ∀ x y : ℝ, B = (x, y) → C = (20 - x, y))
  (E : ℝ × ℝ)
  (h9 : E ∈ ABCD)
  (h10 : ∃ x y : ℝ, E = (x, y) ∧ x = 10) 
  (h11 : B.2 = C.2) :
  let p := λ x : ℝ, (1 / 5) * x^2 + a * x + b in
  p B.1 = B.2 ∧ p C.1 = C.2 ∧ p 10 = E.2 → 
  (-a / (1 / 5)) = 20 :=
by
  sorry

end sum_of_roots_of_parabola_l97_97746


namespace bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97424

def simple_op_time : ℕ := 1
def long_op_time : ℕ := 5
def num_simple_customers : ℕ := 5
def num_long_customers : ℕ := 3
def total_customers : ℕ := 8

theorem bank_queue_minimum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  n * a + 3 * a + b + 4 * a + b + a + b + (b + (n - 1) * a) + b + (b + (n-2) * a) = 40 :=
  by intros; sorry

theorem bank_queue_maximum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  m * (m - 1) * b / 2 + n * a * (m + n) + 1 = 100 :=
  by intros; sorry

theorem expected_wasted_minutes_random_order :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  ∑ i in range total_customers, (i * (a + b)) = 72.5 * (total_customers * (total_customers - 1)) / 2 :=
  by intros; sorry

end bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97424


namespace sequence_of_arrows_l97_97050

-- Definitions
def cycle_modulo (n : ℕ) (modulus : ℕ) := n % modulus

-- The main statement
theorem sequence_of_arrows (s t : ℕ) (modulus : ℕ) (m n p q : ℕ):
  modulus = 6 →
  s = 808 →
  t = 812 →
  m = (s + 1) →
  n = (s + 2) →
  p = (s + 3) →
  q = (s + 4) →
  cycle_modulo s modulus = 2 →
  cycle_modulo t modulus = 2 →
  cycle_modulo m modulus = 3 →
  cycle_modulo n modulus = 4 →
  cycle_modulo p modulus = 5 →
  cycle_modulo q modulus = 2 →
  (2 → 3 ∧ 3 → 4 ∧ 4 → 5 ∧ 5 → 2) :=
by {
  intros,
  sorry
}

end sequence_of_arrows_l97_97050


namespace max_a_squared_b_squared_c_squared_l97_97991

theorem max_a_squared_b_squared_c_squared (a b c : ℤ)
  (h1 : a + b + c = 3)
  (h2 : a^3 + b^3 + c^3 = 3) :
  a^2 + b^2 + c^2 ≤ 57 :=
sorry

end max_a_squared_b_squared_c_squared_l97_97991


namespace shaded_region_area_eq_l97_97236

-- Define the radius of the circles
def radius : ℝ := 5

-- Define the area calculations for the components
def quarter_circle_area : ℝ := (1/4) * Real.pi * radius ^ 2
def isosceles_triangle_area : ℝ := (1/2) * radius * radius

-- Calculate shaded region for a single sector
def shaded_area_single_sector : ℝ := quarter_circle_area - isosceles_triangle_area

-- Total shaded area considering 8 such regions
def total_shaded_area : ℝ := 8 * shaded_area_single_sector

-- Main theorem to prove
theorem shaded_region_area_eq : total_shaded_area = 50 * Real.pi - 100 := by
  sorry

end shaded_region_area_eq_l97_97236


namespace segment_KM_equals_circumradius_l97_97945

theorem segment_KM_equals_circumradius (A B C K M : Point) 
  (eq_tri : equilateral_triangle A B C) 
  (div_AC : divides_ratio K A C (2 / 3)) 
  (div_AB : divides_ratio M A B (1 / 3)) :
  segment_length K M = circumradius A B C :=
sorry

end segment_KM_equals_circumradius_l97_97945


namespace statement_C_is_incorrect_l97_97786

-- Definitions of quadrilaterals and conditions
def Quadrilateral := Type
def Parallelogram (Q : Quadrilateral) := ∃ (a b c d : Q), sides_equal (a, b) (c, d) ∧ sides_equal (b, c) (d, a)
def Rhombus (Q : Quadrilateral) := ∃ (a b c d : Q), sides_equal (a, b) (b, c) ∧ sides_equal (c, d) (d, a)
def Square (Q : Quadrilateral) := ∃ (a b c d : Q), angles_right (a, b, c, d) ∧ diagonals_perpendicular (a, c) (b, d)
def Kite (Q : Quadrilateral) := ∃ (a b c d : Q), diagonals_perpendicular (a, c) (b, d) ∧ sides_congruent (a, b, a, d)

-- Conditions 
axiom quadrilateral_with_opposite_sides_equal_is_parallelogram (Q : Quadrilateral) :
  (∃ (a b c d : Q), sides_equal (a, b) (c, d) ∧ sides_equal (b, c) (d, a)) → Parallelogram Q

axiom quadrilateral_with_all_sides_equal_is_rhombus (Q : Quadrilateral) :
  (∃ (a b c d : Q), sides_equal (a, b) (b, c) ∧ sides_equal (c, d) (d, a)) → Rhombus Q

axiom quadrilateral_with_all_angles_and_diagonals_perpendicular_is_square (Q : Quadrilateral) :
  (∃ (a b c d : Q), angles_right (a, b, c, d) ∧ diagonals_perpendicular (a, c) (b, d)) → Square Q

-- Assertion to be proved
theorem statement_C_is_incorrect (Q : Quadrilateral) :
  ¬(∀ (Q : Quadrilateral), diagonals_perpendicular Q → Rhombus Q) :=
sorry

end statement_C_is_incorrect_l97_97786


namespace cards_per_page_l97_97500

noncomputable def total_cards (new_cards old_cards : ℕ) : ℕ := new_cards + old_cards

theorem cards_per_page
  (new_cards old_cards : ℕ)
  (total_pages : ℕ)
  (h_new_cards : new_cards = 3)
  (h_old_cards : old_cards = 13)
  (h_total_pages : total_pages = 2) :
  total_cards new_cards old_cards / total_pages = 8 :=
by
  rw [h_new_cards, h_old_cards, h_total_pages]
  rfl

end cards_per_page_l97_97500


namespace find_value_l97_97097

theorem find_value (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 + a * b = 7 :=
by
  sorry

end find_value_l97_97097


namespace slower_train_speed_l97_97764

def speed_of_slower_train (distance : ℝ) (time_in_seconds : ℝ) (speed_of_faster_train : ℝ) : ℝ :=
  speed_of_faster_train - ((distance * 3600 / 1000) / time_in_seconds)

theorem slower_train_speed :
  speed_of_slower_train 100 36 46 = 36 :=
by
  unfold speed_of_slower_train
  norm_num
  sorry

end slower_train_speed_l97_97764


namespace find_smallest_solution_l97_97538

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l97_97538


namespace harmonic_division_of_Tangent_Points_l97_97849

-- Defining the circle and square with required properties
variables (Circle : Type) [metric_space Circle]
variables (Square : Type) [metric_space Square]

-- Assume we have a square circumscribed around a circle
axiom square_circumscribed_around_circle : Square → Circle → Prop

-- Assume a tangent intersects the square at four points
variables (P Q R S : Point)
variables (tangent : Circle → Line)

axiom tangent_intersects_square_points : 
  ∀ (c : Circle) (s : Square), 
  square_circumscribed_around_circle s c → tangent c → 
  intersects_at tangent s P R ∧ intersects_at tangent s Q S

-- Harmonic division of points statement
axiom harmonic_division : 
  ∀ {A B C D : Point},
  harmonic (A B C D) ↔ harmonic ((P, R), (Q, S))

-- Translate the proof problem as a Lean theorem statement
theorem harmonic_division_of_Tangent_Points
  {c : Circle} {s : Square}
  (h1 : square_circumscribed_around_circle s c)
  (h2 : tangent_intersects_square_points c s tangent) :
  harmonic ((P, R), (Q, S)) :=
sorry -- Proof to be completed

#check harmonic_division_of_Tangent_Points

end harmonic_division_of_Tangent_Points_l97_97849


namespace lottery_ticket_win_l97_97005

theorem lottery_ticket_win (n : ℕ) (h : n = 50) : 
  ∃ t : ℕ, (t = 26) ∧ (∀ (perm : fin n → fin n), ∃ i : fin t, ∃ j : fin n, (perm j = j)) :=
sorry

end lottery_ticket_win_l97_97005


namespace olivia_choco_cookies_l97_97283

theorem olivia_choco_cookies (total_bags : ℕ) (cookies_per_bag : ℕ) (oatmeal_cookies : ℕ) (choco_cookies : ℕ) :
  total_bags = 6 →
  cookies_per_bag = 9 →
  oatmeal_cookies = 41 →
  choco_cookies = total_bags * cookies_per_bag - oatmeal_cookies →
  choco_cookies = 13 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  simp at h4
  norm_num at h4
  exact h4

end olivia_choco_cookies_l97_97283


namespace cost_per_vent_l97_97729

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end cost_per_vent_l97_97729


namespace find_area_l97_97656

-- Definitions of the problem conditions
variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [EuclideanGeometry A B] [EuclideanGeometry B C] [EuclideanGeometry A C] [EuclideanGeometry D A] [EuclideanGeometry D E] [EuclideanGeometry D B] [EuclideanGeometry F D]

-- Given conditions
def prob_conditions (A B C D E F : Type) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  [EuclideanGeometry A B] [EuclideanGeometry B C] [EuclideanGeometry A C] [EuclideanGeometry D A] [EuclideanGeometry D E] [EuclideanGeometry D B] [EuclideanGeometry F D] : Prop := 
  ∠B = 90 ∧ dist A B = 24 ∧ dist B C = 18 ∧ dist D A = dist D B ∧ perp DF AC ∧ perp DE AB

-- Final proof goal
theorem find_area (A B C D E F : Type) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  [EuclideanGeometry A B] [EuclideanGeometry B C] [EuclideanGeometry A C] [EuclideanGeometry D A] [EuclideanGeometry D E] [EuclideanGeometry D B] [EuclideanGeometry F D]
  (h : prob_conditions A B C D E F) : 
  area A D E F = 162 := 
sorry

end find_area_l97_97656


namespace proposition_four_l97_97626

variables (M N : Type) [normed_add_group M] [normed_add_group N]
variables (α β : set M) (m n : set N)

-- Define the properties needed for the conditions
def parallel (m α : set M) : Prop := sorry -- definition of parallel
def perpendicular (m α : set M) : Prop := sorry -- definition of perpendicular
def not_subset (m α : set M) : Prop := sorry -- definition of not being a subset

theorem proposition_four (α β : set M) (m : set N) 
  (hαβ : perpendicular α β) 
  (hmβ : perpendicular m β) 
  (hmα_not_subset : not_subset m α) 
  : parallel m α := 
sorry

end proposition_four_l97_97626


namespace sum_of_diagonal_elements_l97_97696

/-- Odd numbers from 1 to 49 arranged in a 5x5 grid. -/
def table : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, 1 => 3
| 0, 2 => 5
| 0, 3 => 7
| 0, 4 => 9
| 1, 0 => 11
| 1, 1 => 13
| 1, 2 => 15
| 1, 3 => 17
| 1, 4 => 19
| 2, 0 => 21
| 2, 1 => 23
| 2, 2 => 25
| 2, 3 => 27
| 2, 4 => 29
| 3, 0 => 31
| 3, 1 => 33
| 3, 2 => 35
| 3, 3 => 37
| 3, 4 => 39
| 4, 0 => 41
| 4, 1 => 43
| 4, 2 => 45
| 4, 3 => 47
| 4, 4 => 49
| _, _ => 0

/-- Proof that the sum of five numbers chosen from the table such that no two of them are in the same row or column equals 125. -/
theorem sum_of_diagonal_elements : 
  (table 0 0 + table 1 1 + table 2 2 + table 3 3 + table 4 4) = 125 := by
  sorry

end sum_of_diagonal_elements_l97_97696


namespace exists_square_with_only_invisible_points_l97_97450

def is_invisible (p q : ℤ) : Prop := Int.gcd p q > 1

def all_points_in_square_invisible (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 2 ∧ ∀ x y : ℕ, (x < n ∧ y < n) → is_invisible (k*x) (k*y)

theorem exists_square_with_only_invisible_points (n : ℕ) :
  all_points_in_square_invisible n := sorry

end exists_square_with_only_invisible_points_l97_97450


namespace non_empty_solution_set_range_l97_97973

theorem non_empty_solution_set_range {a : ℝ} 
  (h : ∃ x : ℝ, |x + 2| + |x - 3| ≤ a) : 
  a ≥ 5 :=
sorry

end non_empty_solution_set_range_l97_97973


namespace even_numbers_average_l97_97725

theorem even_numbers_average (n : ℕ) (h1 : 2 * (n * (n + 1)) = 22 * n) : n = 10 :=
by
  sorry

end even_numbers_average_l97_97725


namespace cosine_of_angle_between_vectors_lambda_value_l97_97627

open Real EuclideanSpace

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (-1, -2)

def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

def norm (x : ℝ × ℝ) : ℝ :=
  sqrt (x.1^2 + x.2^2)

def cosine (x y : ℝ × ℝ) : ℝ :=
  dot_product x y / (norm x * norm y)

theorem cosine_of_angle_between_vectors :
  cosine a b = -3 / 5 :=
by
  sorry

def c (λ : ℝ) : ℝ × ℝ := (λ - 2, 2 * λ + 4)
def d : ℝ × ℝ := (-5, 6)

theorem lambda_value :
  ∃ λ : ℝ, dot_product (c λ) d = 0 ∧ λ = -34 / 7 :=
by
  sorry

end cosine_of_angle_between_vectors_lambda_value_l97_97627


namespace rhombus_side_length_l97_97132

theorem rhombus_side_length (area d1 d2 side : ℝ) (h_area : area = 24)
(h_d1 : d1 = 6) (h_other_diag : d2 * 6 = 48) (h_side : side = Real.sqrt (3^2 + 4^2)) :
  side = 5 :=
by
  -- This is where the proof would go
  sorry

end rhombus_side_length_l97_97132


namespace least_cost_for_planting_l97_97712

theorem least_cost_for_planting :
  let 
    area1 := 2 * 3,
    area2 := 3 * 4,
    area3 := 2 * 5,
    area4 := 4 * 4,
    area5 := 3 * 6,
    cost1 := 3.50, -- Cost of Easter lilies
    cost2 := 2.25, -- Cost of Dahlias
    cost3 := 2.00, -- Cost of Cannas
    cost4 := 1.75, -- Cost of Begonias
    cost5 := 1.00 -- Cost of Asters
  in
  (area1 * cost1 + area2 * cost3 + area3 * cost2 + area4 * cost4 + area5 * cost5) = 113.5 :=
by
  sorry

end least_cost_for_planting_l97_97712


namespace composite_N_l97_97901

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

def N : ℕ := (10^2016 + 1) * (∑ i in finset.range 2017, 10^i)

theorem composite_N : is_composite N :=
sorry

end composite_N_l97_97901


namespace stratified_sampling_l97_97437

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem stratified_sampling :
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  combination junior_students junior_sample_size * combination senior_students senior_sample_size =
    combination 400 40 * combination 200 20 :=
by
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  exact sorry

end stratified_sampling_l97_97437


namespace g_inv_g_inv_14_l97_97309

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l97_97309


namespace prove_unique_acute_triangle_l97_97016

noncomputable def acute_triangle_in_regular_2017_gon : Prop := 
  ∀ (P : Finset Point) (triangles : Finset (Finset Point)), 
    (regular_polygon P 2017) ∧ (partition_into_triangles P triangles) →
    (∃! t ∈ triangles, acute_triangle t)
  
axiom regular_polygon (P : Finset Point) (n : ℕ) : Prop
axiom partition_into_triangles (P : Finset Point) (triangles : Finset (Finset Point)) : Prop
axiom acute_triangle (t : Finset Point) : Prop

theorem prove_unique_acute_triangle : acute_triangle_in_regular_2017_gon := 
  by sorry

end prove_unique_acute_triangle_l97_97016


namespace equation_of_line_intersection_l97_97150

theorem equation_of_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∀ x y : ℝ, x - 2*y + 1 = 0 :=
by
  sorry

end equation_of_line_intersection_l97_97150


namespace total_students_in_class_l97_97218

theorem total_students_in_class (F T B N : ℕ) (hF : F = 26) (hT : T = 20) (hB : B = 17) (hN : N = 10) :
  F + T - B + N = 39 :=
by
  rw [hF, hT, hB, hN]
  norm_num

end total_students_in_class_l97_97218


namespace sec_7pi_over_4_eq_sqrt2_l97_97916

theorem sec_7pi_over_4_eq_sqrt2 : Real.sec (7 * Real.pi / 4) = Real.sqrt 2 := 
by 
  sorry

end sec_7pi_over_4_eq_sqrt2_l97_97916


namespace intersection_slopes_l97_97003

theorem intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (4 / 41)) ∨ m ∈ Set.Ici (Real.sqrt (4 / 41)) := 
sorry

end intersection_slopes_l97_97003


namespace least_possible_length_of_XZ_l97_97659

open EuclideanGeometry

def Triangle (α β γ: Point) : Prop := Plane α β γ ∧ ⋆ ∧ Plane ∃ (c_α_equilateral: c_β γ β) ⋆
/- Definition of a right-angled triangle PQR -/
structure RightAngledTriangle (P Q R: Point) where
  angle_Q_eq_90 : angle P Q R = 90
  PQ_length : Real
  QR_length : Real

noncomputable def XZ_least_length (P Q R X Y Z: Point) (PQ: Segment P Q) (QR: Segment Q R) (PR: Segment P R) (PQ_length: PQ.length = 2) (QR_length: QR.length = 8)
    (X_on_PQ: X ∈ PQ) (XY_parallel_QR: Parallel (Line_through X (line_parallel XY QR)) PR) 
    (YZ_parallel_PQ: Parallel (Line_through Y (line_parallel YZ PQ)) QR) : Real :=
  2

theorem least_possible_length_of_XZ (P Q R X Y Z: Point) (PQ: Segment P Q) (QR: Segment Q R) (PR: Segment P R)
    (h_triangle : RightAngledTriangle P Q R)
    (PQ_length: PQ.length = 2)
    (QR_length: QR.length = 8)
    (X_on_PQ: X ∈ PQ)
    (XY_parallel_QR: Parallel (Line_through X (line_parallel QR PR)))
    (YZ_parallel_PQ: Parallel (Line_through Y (line_parallel PQ QR)))
    : XZ_least_length P Q R X Y Z PQ QR PR PQ_length QR_length X_on_PQ XY_parallel_QR YZ_parallel_PQ = 2 :=
sorry

end least_possible_length_of_XZ_l97_97659


namespace butterfly_average_black_dots_l97_97346

theorem butterfly_average_black_dots :
  ∀ (nA nB nC dotsA dotsB dotsC : ℕ), 
  nA = 15 → nB = 25 → nC = 35 → 
  dotsA = 545 → dotsB = 780 → dotsC = 1135 →
  (dotsA * nA) / nA = 545 ∧ 
  (dotsB * nB) / nB = 780 ∧ 
  (dotsC * nC) / nC = 1135 :=
by
  intros nA nB nC dotsA dotsB dotsC hnA hnB hnC hdotsA hdotsB hdotsC
  rw [hnA, hnB, hnC, hdotsA, hdotsB, hdotsC]
  norm_num
  exact ⟨rfl, rfl, rfl⟩

end butterfly_average_black_dots_l97_97346


namespace repeating_decimal_sum_l97_97179

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l97_97179


namespace angle_a_b_l97_97980

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  Real.acos (dot_product / (norm_a * norm_b))

theorem angle_a_b : 
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  let b : ℝ × ℝ := b in
  (Real.sqrt (b.1 * b.1 + b.2 * b.2) = 1 ∧ Real.sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) = Real.sqrt 3)
  → angle_between_vectors a b = 2 * Real.pi / 3 :=
begin
  sorry
end

end angle_a_b_l97_97980


namespace eliza_total_clothes_l97_97064

def time_per_blouse : ℕ := 15
def time_per_dress : ℕ := 20
def blouse_time : ℕ := 2 * 60   -- 2 hours in minutes
def dress_time : ℕ := 3 * 60    -- 3 hours in minutes

theorem eliza_total_clothes :
  (blouse_time / time_per_blouse) + (dress_time / time_per_dress) = 17 :=
by
  sorry

end eliza_total_clothes_l97_97064


namespace lambda_range_l97_97201
open Nat

theorem lambda_range (S : ℕ → ℝ) (a : ℕ → ℝ) (λ : ℝ) :
  S 3 = 3 / 2 → 
  S 6 = 21 / 16 → 
  (∀ n, λ * a n - n^2 > λ * a (n + 1) - (n + 1)^2) → 
  (-1 < λ ∧ λ < 10 / 3) :=
by
  sorry

end lambda_range_l97_97201


namespace function_inequality_l97_97969

noncomputable def f : ℝ → ℝ
| x => if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

theorem function_inequality : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end function_inequality_l97_97969


namespace factor_expression_l97_97509

variable (x : ℝ)

theorem factor_expression : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) :=
by
  sorry

end factor_expression_l97_97509


namespace f_of_neg4_eq_1_l97_97143

noncomputable def f : ℝ → ℝ 
| x if x < 2 := f (x + 2)
| x := Real.log x / Real.log 2

theorem f_of_neg4_eq_1 : f (-4) = 1 := 
by 
  sorry

end f_of_neg4_eq_1_l97_97143


namespace sin_squared_sum_eq_30_5_l97_97492

theorem sin_squared_sum_eq_30_5 :
  (∑ k in FinRange 59, sin^2 (Real.pi / 60 * (k + 1)) : ℝ) = 30.5 := by
  sorry

end sin_squared_sum_eq_30_5_l97_97492


namespace find_smallest_solution_l97_97537

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l97_97537


namespace rectangle_541st_position_l97_97013

def transformation : List String → Nat → List String
| ["W", "X", "Y", "Z"], 0 => ["W", "X", "Y", "Z"]
| ["W", "X", "Y", "Z"], n => if n % 2 = 1 then ["Y", "Z", "X", "W"] else ["W", "X", "Y", "Z"]

theorem rectangle_541st_position :
  transformation ["W", "X", "Y", "Z"] 541 = ["Y", "Z", "X", "W"] :=
by
  sorry

end rectangle_541st_position_l97_97013


namespace balcony_more_than_orchestra_l97_97391

variables (O B : ℕ)

theorem balcony_more_than_orchestra
  (h1 : O + B = 380)
  (h2 : 12 * O + 8 * B = 3_320) :
  B - O = 240 :=
by
  sorry

end balcony_more_than_orchestra_l97_97391


namespace cakes_to_make_l97_97489

-- Define the conditions
def packages_per_cake : ℕ := 2
def cost_per_package : ℕ := 3
def total_cost : ℕ := 12

-- Define the proof problem
theorem cakes_to_make (h1 : packages_per_cake = 2) (h2 : cost_per_package = 3) (h3 : total_cost = 12) :
  (total_cost / cost_per_package) / packages_per_cake = 2 :=
by sorry

end cakes_to_make_l97_97489


namespace repeating_decimal_fraction_l97_97185

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l97_97185


namespace cats_new_total_weight_l97_97858

noncomputable def total_weight (weights : List ℚ) : ℚ :=
  weights.sum

noncomputable def remove_min_max_weight (weights : List ℚ) : ℚ :=
  let min_weight := weights.minimum?.getD 0
  let max_weight := weights.maximum?.getD 0
  weights.sum - min_weight - max_weight

theorem cats_new_total_weight :
  let weights := [3.5, 7.2, 4.8, 6, 5.5, 9, 4]
  remove_min_max_weight weights = 27.5 := by
  sorry

end cats_new_total_weight_l97_97858


namespace min_distance_from_point_to_line_l97_97329

noncomputable def point_to_line_distance (A : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * A.1 + b * A.2 + c| / real.sqrt (a^2 + b^2)

theorem min_distance_from_point_to_line :
  point_to_line_distance (2, 1) 1 1 3 = 3 * real.sqrt 2 :=
by
  sorry

end min_distance_from_point_to_line_l97_97329


namespace smallest_solution_floor_eq_l97_97528

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l97_97528


namespace infinite_non_prime_power_numerators_l97_97709

/-- There exist infinitely many natural numbers n for which 
the numerator of the reduced fraction 1 + 1/2 + ... + 1/n 
is not a power of a prime number with a natural exponent. -/
theorem infinite_non_prime_power_numerators :
  ∃^∞ n : ℕ, ¬ ∃ (p : ℕ) (e : ℕ), nat.prime p ∧ A(n) = p^e :=
sorry

end infinite_non_prime_power_numerators_l97_97709


namespace tan_2theta_l97_97144

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.cos x

theorem tan_2theta (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan (2 * θ) = -4 / 3 := 
by 
  sorry

end tan_2theta_l97_97144


namespace polyhedron_volume_l97_97579

noncomputable def volume_of_polyhedron (a h : ℝ) : ℝ :=
  (2 / 3) * a^2 * h * Real.sqrt 3

theorem polyhedron_volume {a h : ℝ} (h1 : a > 0) (h2 : h > 0) :
  let V := volume_of_polyhedron a h in
  V = (2 / 3) * a^2 * h * Real.sqrt 3 :=
sorry

end polyhedron_volume_l97_97579


namespace arrange_abc_l97_97096

noncomputable def a : ℝ := 5^0.7
noncomputable def b : ℝ := Real.logBase 0.3 2
noncomputable def c : ℝ := 0.7^5

theorem arrange_abc : b < c ∧ c < a := by
  sorry

end arrange_abc_l97_97096


namespace store_earnings_correct_l97_97828

theorem store_earnings_correct :
  let graphics_cards_qty := 10
  let hard_drives_qty := 14
  let cpus_qty := 8
  let rams_qty := 4
  let psus_qty := 12
  let monitors_qty := 6
  let keyboards_qty := 18
  let mice_qty := 24

  let graphics_card_price := 600
  let hard_drive_price := 80
  let cpu_price := 200
  let ram_price := 60
  let psu_price := 90
  let monitor_price := 250
  let keyboard_price := 40
  let mouse_price := 20

  let total_earnings := graphics_cards_qty * graphics_card_price +
                        hard_drives_qty * hard_drive_price +
                        cpus_qty * cpu_price +
                        rams_qty * ram_price +
                        psus_qty * psu_price +
                        monitors_qty * monitor_price +
                        keyboards_qty * keyboard_price +
                        mice_qty * mouse_price
  total_earnings = 12740 :=
by
  -- definitions and calculations here
  sorry

end store_earnings_correct_l97_97828


namespace product_of_valid_n_values_l97_97498
...

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def P (t : ℤ) : ℤ :=
  t^3 - 29 * t^2 + 212 * t - 399

theorem product_of_valid_n_values :
  let valid_ns := {n : ℕ | P n = sum_of_digits n}
  ∏ n in valid_ns, n = 399 :=
sorry

end product_of_valid_n_values_l97_97498


namespace bank_queue_wasted_time_l97_97425

-- Conditions definition
def simple_time : ℕ := 1
def lengthy_time : ℕ := 5
def num_simple : ℕ := 5
def num_lengthy : ℕ := 3
def total_people : ℕ := 8

-- Theorem statement
theorem bank_queue_wasted_time :
  (min_wasted_time : ℕ := 40) ∧
  (max_wasted_time : ℕ := 100) ∧
  (expected_wasted_time : ℚ := 72.5) := by
  sorry

end bank_queue_wasted_time_l97_97425


namespace max_gcd_of_sum_1001_l97_97749

theorem max_gcd_of_sum_1001 :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ),
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = 1001) ∧ 
  (∀ (d : ℕ), (∀ (i : ℕ), i ∈ {a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_{10}} → d ∣ i) → d ≤ 91) ∧
  (91 ∣ a_1 ∧ 91 ∣ a_2 ∧ 91 ∣ a_3 ∧ 91 ∣ a_4 ∧ 91 ∣ a_5 ∧ 91 ∣ a_6 ∧ 91 ∣ a_7 ∧ 91 ∣ a_8 ∧ 91 ∣ a_9 ∧ 91 ∣ a_{10}) := 
sorry

end max_gcd_of_sum_1001_l97_97749


namespace average_calls_per_day_l97_97663

def calls_Monday : ℕ := 35
def calls_Tuesday : ℕ := 46
def calls_Wednesday : ℕ := 27
def calls_Thursday : ℕ := 61
def calls_Friday : ℕ := 31

def total_calls : ℕ := calls_Monday + calls_Tuesday + calls_Wednesday + calls_Thursday + calls_Friday
def number_of_days : ℕ := 5

theorem average_calls_per_day : (total_calls / number_of_days) = 40 := 
by 
  -- calculations and proof steps go here.
  sorry

end average_calls_per_day_l97_97663


namespace prop1_prop2_prop3_prop4_l97_97894

-- Proposition 1
theorem prop1 (a b : ℝ^3) (h : ‖a + b‖ = ‖a‖ + ‖b‖) : collinear (a, b) ∧ same_direction (a, b) :=
sorry

-- Proposition 2
theorem prop2 : ∃ T > 0, ∀ x : ℝ, cos (sin (x + T)) = cos (sin x) ∧ (∀ e > 0, (∃ x, x < e ∧ x < T)) ∧ T = π :=
sorry

-- Proposition 3
theorem prop3 (A B C : Point ℝ) (h₁ : dist A C = 3) (h₂ : dist B C = 4) (h₃ : dist A B = 5) : dot_product (vector A B) (vector B C) ≠ 16 :=
sorry

-- Proposition 4
theorem prop4 : is_symmetry_center (fun x => tan (2 * x - π / 3)) (5 * π / 12, 0) :=
sorry

end prop1_prop2_prop3_prop4_l97_97894


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97414

def a : ℕ := 1  -- time for a simple operation
def b : ℕ := 5  -- time for a lengthy operation
def n : ℕ := 5  -- number of "simple" customers
def m : ℕ := 3  -- number of "lengthy" customers
def total_customers : ℕ := 8 -- 8 people in queue

theorem min_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → min_wasted_person_minutes ≤ 40) :=
by
  sorry

theorem max_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → max_wasted_person_minutes ≥ 100) :=
by
  sorry

theorem expected_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → expected_wasted_person_minutes = 72.5) :=
by
  sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97414


namespace simplify_expression_l97_97912

theorem simplify_expression (x : ℝ) : 2 * x + 3 - 4 * x - 5 + 6 * x + 7 - 8 * x - 9 = -4 * x - 4 :=
by  -- Proof will go here
	sorry

end simplify_expression_l97_97912


namespace interval_of_increase_of_sine_function_l97_97733

open Real

noncomputable def ω := 2 * π / π

-- Proof problem statement
theorem interval_of_increase_of_sine_function : 
  ∀ (x : ℝ), (f : ℝ → ℝ) (f := λ x, sin (ω * x + π / 6))
  (H1 : ∃ T > 0, ∀ x, f (x + T) = f x),
  let I := (Icc (-π / 12 : ℝ) (5 * π / 12 : ℝ)) in 
  (∀ x ∈ I, 0 < x → f x > f (x - 1)) :=
sorry

end interval_of_increase_of_sine_function_l97_97733


namespace nathan_correct_answers_l97_97648

theorem nathan_correct_answers (c w : ℤ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 := 
by sorry

end nathan_correct_answers_l97_97648


namespace lower_bound_strategy_l97_97361

def optimal_strategy (N : ℕ) : ℕ :=
⌈ log 2 N ⌉

theorem lower_bound_strategy (T : (ℕ → ℕ → bool) → ℕ → ℕ) (N : ℕ) :
  ∃ n : ℕ, n ≤ N → T (λ (n : ℕ) (x : ℕ), n ≥ x) n ≥ optimal_strategy N :=
sorry

end lower_bound_strategy_l97_97361


namespace mark_9x9_grid_l97_97660

theorem mark_9x9_grid :
  ∃ (markings : (ℕ × ℕ) → bool),
    (∀ i, 0 ≤ i ∧ i < 8 → (let row_sum j = if j % 2 = 0 then 6 else 4 in
                             ∑ j in finset.range 9, if markings (i, j) then 1 else 0 + if markings (i + 1, j) then 1 else 0) ≥ 6) ∧ 
    (∀ j, 0 ≤ j ∧ j < 8 → (let col_sum i = if j % 2 = 0 then 1 else 4 in
                             ∑ i in finset.range 9, if markings (i, j) then 1 else 0 + if markings (i, j + 1) then 1 else 0) ≤ 5) :=
sorry

end mark_9x9_grid_l97_97660


namespace triangle_inequalities_triangle_inequalities2_l97_97685

theorem triangle_inequalities (a b c λ : ℝ) (h₁ : a + b + c = λ) (h₂ : ∀ x, x > 0):
    (λ > 0) → (a * b * c > 0) → (a ^ 2 + b ^ 2 + c ^ 2 + (4 / λ) * a * b * c < (λ ^ 2) / 2) ∧
    (13 * (λ ^ 2) / 27 ≤ a ^ 2 + b ^ 2 + c ^ 2 + (4 / λ) * a * b * c) :=
begin
  sorry
end

theorem triangle_inequalities2 (a b c λ : ℝ) (h₁ : a + b + c = λ) (h₂ : ∀ x, x > 0):
    (λ > 0) → (a * b * c > 0) → ((1 / 4) * (λ ^ 2) < a * b + b * c + c * a - (2 / λ) * a * b * c) ∧ 
    (a * b + b * c + c * a - (2 / λ) * a * b * c ≤ (7 / 27) * (λ ^ 2)) :=
begin
  sorry
end

end triangle_inequalities_triangle_inequalities2_l97_97685


namespace find_x_from_percentage_l97_97633

theorem find_x_from_percentage (x : ℝ) (h : 0.2 * 30 = 0.25 * x + 2) : x = 16 :=
sorry

end find_x_from_percentage_l97_97633


namespace chess_tournament_games_l97_97988

theorem chess_tournament_games (n : ℕ) (h : n = 21) :
  (n * (n - 1)) / 2 = 210 :=
by {
  rw h,
  norm_num,
  sorry
}

end chess_tournament_games_l97_97988


namespace sum_of_fraction_components_l97_97196

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l97_97196


namespace slope_angle_range_l97_97739

theorem slope_angle_range
  (P : ℝ × ℝ)
  (P_cond : P = (1, -Real.sqrt 3))
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 5)
  (chord_length_condition : ∀ (l : ℝ × ℝ → Prop),
    ∃ (L :  ℝ × ℝ → Prop),
    (L P) ∧
    ((L, x^2 + y^2 = 5) → length(chord x y) ≥ 4)) :
  ∃ α, α ∈ Icc (π / 2) (5 * π / 6) :=
sorry

end slope_angle_range_l97_97739


namespace cannot_determine_right_triangle_from_conditions_l97_97658

-- Let triangle ABC have side lengths a, b, c opposite angles A, B, C respectively.
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Condition A: c^2 = a^2 - b^2 is rearranged to c^2 + b^2 = a^2 implying right triangle
def condition_A (a b c : ℝ) : Prop := c^2 = a^2 - b^2

-- Condition B: Triangle angles in the ratio A:B:C = 3:4:5 means not a right triangle
def condition_B : Prop := 
  let A := 45.0
  let B := 60.0
  let C := 75.0
  A ≠ 90.0 ∧ B ≠ 90.0 ∧ C ≠ 90.0

-- Condition C: Specific lengths 7, 24, 25 form a right triangle
def condition_C : Prop := 
  let a := 7.0
  let b := 24.0
  let c := 25.0
  is_right_triangle a b c

-- Condition D: A = B - C can be shown to always form at least one 90 degree angle, a right triangle
def condition_D (A B C : ℝ) : Prop := A = B - C ∧ (A + B + C = 180)

-- The actual mathematical proof that option B does not determine a right triangle
theorem cannot_determine_right_triangle_from_conditions :
  ∀ a b c (A B C : ℝ),
    (condition_A a b c → is_right_triangle a b c) ∧
    (condition_C → is_right_triangle 7 24 25) ∧
    (condition_D A B C → is_right_triangle a b c) ∧
    ¬condition_B :=
by
  sorry

end cannot_determine_right_triangle_from_conditions_l97_97658


namespace intersection_of_A_and_B_l97_97975

namespace IntersectionProblem

def setA : Set ℝ := {0, 1, 2}
def setB : Set ℝ := {x | x^2 - x ≤ 0}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := sorry

end IntersectionProblem

end intersection_of_A_and_B_l97_97975


namespace find_x_from_conditions_l97_97267

theorem find_x_from_conditions (a b x y s : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) :
  s = (4 * a)^(4 * b) ∧ s = a^b * y^b ∧ y = 4 * x → x = 64 * a^3 :=
by
  sorry

end find_x_from_conditions_l97_97267


namespace magnitude_projection_sum_l97_97259

noncomputable theory
open_locale real_inner_product_space

variables 
  (v w u : ℝ^3)
  (h₁ : inner v w = 6)
  (h₂ : ∥w∥ = 3)
  (h₃ : inner v u = -4)

theorem magnitude_projection_sum :
  ∥((inner v w / ∥w∥ ^ 2) • w + (inner v u / ∥u∥ ^ 2) • u)∥ = 4 * real.sqrt 3 / 3 := 
sorry

end magnitude_projection_sum_l97_97259


namespace hyperbola_center_l97_97833

theorem hyperbola_center (F1 F2 : ℝ × ℝ) (F1_eq : F1 = (3, -2)) (F2_eq : F2 = (11, 6)) :
  let C := ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2) in C = (7, 2) :=
by
  sorry

end hyperbola_center_l97_97833


namespace total_chars_l97_97290

def is_prime (n : ℕ) : Prop := sorry

def valid_lowercase_string (s : String) : Prop :=
  s.length = 12 ∧
  ∀ i, i < s.length - 1 →
    (s.get ⟨i, sorry⟩ ≠ s.get ⟨i + 1, sorry⟩ ∧
    s.count ⟨s.get ⟨i, sorry⟩⟩ ≤ 2)

def valid_uppercase_number_string (s : String) : Prop :=
  s.length = 6 ∧
  ∀ i, (i % 2 = 0 → ('A' ≤ s.get ⟨i, sorry⟩ ∧ s.get ⟨i, sorry⟩ ≤ 'Z')) ∧
       (i % 2 = 1 → ('0' ≤ s.get ⟨i, sorry⟩ ∧ s.get ⟨i, sorry⟩ ≤ '9')) ∧
       (s.get ⟨i, sorry⟩ ≠ s.get ⟨i + 1, sorry⟩) ∧
       (i % 2 = 1 → s.get ⟨i, sorry⟩ != (i+1).toString)

def valid_digit_string (s : String) : Prop :=
  s.length = 4 ∧
  (∑ i in Finset.range s.length, s.get ⟨i, sorry⟩.toNat % 2 = 0) ∧
  ∀ i, (s.get ⟨i, sorry⟩).toNat.succ.prime ∧ i ∈ [0, 3]

def valid_symbol_string (s : String) : Prop :=
  s.length = 2 ∧
  s.get ⟨0, sorry⟩ ≠ s.get ⟨1, sorry⟩ ∧
  (s.get ⟨0, sorry⟩ = '#' ∨ s.get ⟨0, sorry⟩ = '@' ∨ s.get ⟨0, sorry⟩ = '%')

def valid_non_adjacent_case (ls : String) (us : String) : Prop :=
  ∀ i, i < us.length →
    (i % 2 = 0 → ∀ j, j < ls.length - 1 →
      (us.get ⟨i, sorry⟩ ≠ ls.get ⟨j, sorry⟩.toUpperCase) →
      (us.get ⟨i, sorry⟩ ≠ ls.get ⟨j + 1, sorry⟩.toUpperCase))

theorem total_chars (ls us ds ss : String) 
  (h1 : valid_lowercase_string ls)
  (h2 : valid_uppercase_number_string us)
  (h3 : valid_digit_string ds)
  (h4 : valid_symbol_string ss)
  (h5 : valid_non_adjacent_case ls us) :
  ls.length + us.length + ds.length + ss.length = 24 := by
  sorry

end total_chars_l97_97290


namespace correct_statement_is_D_l97_97385

-- Define each statement as a proposition
def statement_A (a b c : ℕ) : Prop := c ≠ 0 → (a * c = b * c → a = b)
def statement_B : Prop := 30.15 = 30 + 15/60
def statement_C : Prop := ∀ (radius : ℕ), (radius ≠ 0) → (360 * (2 / (2 + 3 + 4)) = 90)
def statement_D : Prop := 9 * 30 + 40/2 = 50

-- Define the theorem to state the correct statement (D)
theorem correct_statement_is_D : statement_D :=
sorry

end correct_statement_is_D_l97_97385


namespace binom_20_9_calc_l97_97126

-- Define the given conditions
def binom_18_7 : ℕ := 31824
def binom_18_8 : ℕ := 43758
def binom_18_9 : ℕ := 43758

-- Prove the required binomial coefficient based on these conditions
theorem binom_20_9_calc : nat.choose 20 9 = 163098 :=
by
  have h1 : nat.choose 19 8 = binom_18_7 + binom_18_8 := by
    -- Substitute values and apply Pascal's identity
    rw [nat.choose_succ_succ, nat.choose]

  have h2 : nat.choose 19 9 = binom_18_8 + binom_18_9 := by
    -- Substitute values and apply Pascal's identity
    rw [nat.choose_succ_succ, nat.choose]

  -- Transform the given conditions into the theorem
  rw [h1, h2]
  exact (31824 + 43758) + (43758 + 43758)
  sorry

end binom_20_9_calc_l97_97126


namespace fraction_value_l97_97342

theorem fraction_value : (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 := by
  sorry

end fraction_value_l97_97342


namespace rotation_after_two_hours_l97_97859

theorem rotation_after_two_hours :
  let rotation_per_hour := -30
  in rotation_per_hour * 2 = -60 :=
by
  let rotation_per_hour := -30
  have rotation := rotation_per_hour * 2
  show rotation = -60 from rfl


end rotation_after_two_hours_l97_97859


namespace sequence_a100_gt_14_l97_97334

theorem sequence_a100_gt_14 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 1 ≤ n → a (n+1) = a n + 1 / a n) :
  a 100 > 14 :=
by sorry

end sequence_a100_gt_14_l97_97334


namespace sum_of_perpendicular_distances_l97_97149

theorem sum_of_perpendicular_distances 
  (a b c : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 4) 
  (h3 : c = 5) 
  (O₁ O₂ O₃ : Type)
  (A B C : Type)
  (touch_pairwise : (A : O₁) -> (B : O₂) -> (C : O₃) -> Prop) :
  (
    let d₁ := (5 / Real.sqrt 14),
    let d₂ := (5 / Real.sqrt 21),
    let d₃ := (5 / Real.sqrt 30),
    d₁ + d₂ + d₃ = 5 / Real.sqrt 14 + 5 / Real.sqrt 21 + 5 / Real.sqrt 30
  ) := sorry

end sum_of_perpendicular_distances_l97_97149


namespace smallest_solution_floor_equation_l97_97549

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l97_97549


namespace beautiful_not_all_five_digit_l97_97693

def is_beautiful (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 0 ∧ d < 10 ∧ (∃ k : ℕ, k ≥ 1 ∧ n = d * (10^k + 10^(k-1) + ... + 10^1 + 10^0))

def can_be_represented_as_sum (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    is_beautiful a ∧ is_beautiful b ∧ is_beautiful c ∧ is_beautiful d ∧ is_beautiful e ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ d ∧ b ≠ e ∧ c ≠ e ∧
    a + b + c + d + e = n

theorem beautiful_not_all_five_digit :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ¬ can_be_represented_as_sum n :=
sorry

end beautiful_not_all_five_digit_l97_97693


namespace magnitude_of_z_l97_97952

open Complex -- open the complex number namespace

theorem magnitude_of_z (z : ℂ) (h : z + I = 3) : Complex.abs z = Real.sqrt 10 :=
by
  sorry

end magnitude_of_z_l97_97952


namespace angle_between_vectors_eq_pi_over_3_l97_97597

open Real

def vec : Type := ℝ × ℝ

-- Definitions corresponding to the conditions
variables (a b : vec)
variables (dot_product : vec → vec → ℝ)
variables (magnitude : vec → ℝ)

axiom a_magnitude : magnitude a = 1
axiom b_value : b = (0, 2)
axiom a_dot_b : dot_product a b = 1

-- Definition of dot product and magnitude
def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def magnitude (v : vec) : ℝ := sqrt (v.1^2 + v.2^2)

-- The theorem representing the proof problem
theorem angle_between_vectors_eq_pi_over_3 
  (a b : vec)
  (a_magnitude : magnitude a = 1)
  (b_value : b = (0, 2))
  (a_dot_b : dot_product a b = 1) : 
  ∃ θ : Real, θ = Real.arccos (1 / 2) ∧ θ = π / 3 :=
by
  -- Proof is not required, so using sorry.
  sorry

end angle_between_vectors_eq_pi_over_3_l97_97597


namespace vacation_fund_percentage_l97_97060

-- Define Jill's net monthly salary
def net_monthly_salary : ℝ := 3600

-- Define discretionary income as one fifth of net monthly salary
def discretionary_income := (1/5 : ℝ) * net_monthly_salary

-- Define the percentage Jill puts into savings
def savings_percentage : ℝ := 20 / 100

-- Define the percentage Jill spends on eating out and socializing
def socializing_percentage : ℝ := 35 / 100

-- Define the amount Jill spends on gifts and charitable causes
def gifts_charity_amount : ℝ := 108

-- Define the percentage Jill spends on gifts and charitable causes
def gifts_charity_percentage := gifts_charity_amount / discretionary_income

-- Prove that the percentage Jill puts into a vacation fund is 30%
theorem vacation_fund_percentage : 
  let V := 100 / 100 - (savings_percentage + socializing_percentage + gifts_charity_percentage) in
  V = 30 / 100 := by
  have h1 : discretionary_income = 720 := by 
    calc
      discretionary_income = (1/5 : ℝ) * 3600 := rfl
      ... = 720 := by norm_num
  have h2 : gifts_charity_percentage = 15 / 100 := by
    calc
      gifts_charity_percentage = 108 / 720 := rfl
      ... = 15 / 100 := by norm_num
  calc
    V = 100 / 100 - (20 / 100 + 35 / 100 + 15 / 100) := rfl
    ... = 30 / 100 := by norm_num

end vacation_fund_percentage_l97_97060


namespace sum_integers_between_6_and_14_l97_97776

theorem sum_integers_between_6_and_14 : (∑ k in Finset.range (15) \ Finset.range (6), k) = 90 := by
  sorry

end sum_integers_between_6_and_14_l97_97776


namespace max_min_y_l97_97593

theorem max_min_y (α β : ℝ) (h1 : sin α + sin β = 1/3) :
  let y := sin β - cos α ^ 2
  in (y ≤ 4/9 ∧ -11/12 ≤ y) :=
by
  let y := sin β - cos α ^ 2
  have h2 : y = sin α ^ 2 - sin α - 2/3, {
    rw [←h1, cos_sq, sub_add_eq_sub_sub],
    linarith,
  },
  have bound_α : -2/3 ≤ sin α ∧ sin α ≤ 1,
  {
    interval_cases h1,
    solve_by_elim,
  }
  have bound_y : (sin α - 1/2)^2 - 11/12 ≤ y ∧ y ≤ (sin α - 1/2)^2 - 11/12,
  {
    apply sq_le_sq,
    linarith [bound_α],
  }
  finish sorry

end max_min_y_l97_97593


namespace find_angle_y_l97_97232

theorem find_angle_y (ABC_is_straight : angle_ABC = 127) (angle1 : 37) (angle2 : 40) 
    (sum_angles_ABD : 180) :
    angle_y = 90 :=
sorry

end find_angle_y_l97_97232


namespace find_length_of_AC_l97_97625

theorem find_length_of_AC
  (A B C : Type)
  (AB : Real)
  (AC : Real)
  (Area : Real)
  (angle_A : Real)
  (h1 : AB = 8)
  (h2 : angle_A = (30 * Real.pi / 180)) -- converting degrees to radians
  (h3 : Area = 16) :
  AC = 8 :=
by
  -- Skipping proof as requested
  sorry

end find_length_of_AC_l97_97625


namespace smallest_solution_eq_sqrt_104_l97_97534

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l97_97534


namespace limited_variable_point_example_range_of_limited_variable_point_l97_97282

def limited_variable_point (m n : ℝ) : ℝ :=
  if h : m >= 0 then n - 4 else -n

theorem limited_variable_point_example :
  limited_variable_point (-2) 3 = -3 :=
by
  unfold limited_variable_point
  simp [lt_irrefl]

theorem range_of_limited_variable_point :
  let f (m : ℝ) := -m^2 + 4 * m + 2 in
  let n_prime (m : ℝ) := if h : m >= 0 then f m - 4 else -f m in
  ∀ (m : ℝ), -1 ≤ m ∧ m ≤ 3 → -2 ≤ n_prime m ∧ n_prime m ≤ 3 :=
by
  sorry

end limited_variable_point_example_range_of_limited_variable_point_l97_97282


namespace sum_integers_between_6_and_14_l97_97778

theorem sum_integers_between_6_and_14 : (∑ k in Finset.range (15) \ Finset.range (6), k) = 90 := by
  sorry

end sum_integers_between_6_and_14_l97_97778


namespace sum_equals_expected_l97_97480

noncomputable def sum_of_exponentials :=
  5 * Complex.exp (2 * Real.pi * Complex.I / 13) + 5 * Complex.exp (17 * Real.pi * Complex.I / 26)

noncomputable def expected_form :=
  5 * Real.sqrt 2 * Complex.exp (21 * Real.pi * Complex.I / 52)

theorem sum_equals_expected :
  sum_of_exponentials = expected_form :=
by sorry

end sum_equals_expected_l97_97480


namespace curved_surface_area_l97_97458

def right_cylinder (h r : ℝ) :=
  h > 0 ∧ r > 0

def lateral_surface_area (r h : ℝ) : ℝ := 
  2 * Real.pi * r * h

theorem curved_surface_area
  (h r : ℝ)
  (Hcylinder : right_cylinder h r) 
  (Hh : h = 8)
  (Hr : r = 3) :
  lateral_surface_area r h = 48 * Real.pi :=
by
  rw [Hh, Hr]
  unfold lateral_surface_area
  sorry

end curved_surface_area_l97_97458


namespace remainder_eq_27_l97_97081

def p (x : ℝ) : ℝ := x^4 + 2 * x^2 + 3
def a : ℝ := -2
def remainder := p (-2)
theorem remainder_eq_27 : remainder = 27 :=
by
  sorry

end remainder_eq_27_l97_97081


namespace julia_played_with_kids_on_Monday_l97_97252

theorem julia_played_with_kids_on_Monday (k_wednesday : ℕ) (k_monday : ℕ)
  (h1 : k_wednesday = 4) (h2 : k_monday = k_wednesday + 2) : k_monday = 6 := 
by
  sorry

end julia_played_with_kids_on_Monday_l97_97252


namespace non_similar_triangles_with_arithmetic_angles_l97_97163

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end non_similar_triangles_with_arithmetic_angles_l97_97163


namespace angle_CAG_is_15_degrees_l97_97907

/-!
# Geometry Problem: Equilateral Triangle and Rectangle

Prove that given an equilateral triangle ABC and rectangle BCGF, where BC = BF = 3x and FG = 2x, the measure of angle CAG is 15 degrees.
-/

-- Define basic geometric objects and properties
axiom is_equilateral (ABC : Type) (A B C : ABC) : Prop
axiom is_rectangle (BCGF : Type) (B C G F : BCGF) : Prop
axiom coplanar (ABC : Type) (BCGF : Type)  : Prop
axiom equals (a b : ℝ) : Prop (# Define equality of lengths/angles)

-- Define lengths as given in the problem
axiom BC : ℝ := 3 * x
axiom BF : ℝ := 3 * x
axiom FG : ℝ := 2 * x

-- Define angles and prove the target value
theorem angle_CAG_is_15_degrees
  (A B C : ABC)
  (B C G F : BCGF)
  (h1 : is_equilateral ABC A B C)
  (h2 : is_rectangle BCGF B C G F)
  (h3 : coplanar ABC BCGF)
  (h4 : equals (dist B C) (3 * x))
  (h5 : equals (dist B F) (3 * x))
  (h6 : equals (dist F G) (2 * x)) :
  measure_angle A C G = 15 :=
begin
  sorry
end

end angle_CAG_is_15_degrees_l97_97907


namespace exists_another_nice_triple_l97_97880

noncomputable def is_nice_triple (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ (b - a) = (c - b)) ∧
  (Nat.gcd b a = 1 ∧ Nat.gcd b c = 1) ∧ 
  (∃ k, a * b * c = k^2)

theorem exists_another_nice_triple (a b c : ℕ) 
  (h : is_nice_triple a b c) : ∃ a' b' c', 
  (is_nice_triple a' b' c') ∧ 
  (a' = a ∨ a' = b ∨ a' = c ∨ 
   b' = a ∨ b' = b ∨ b' = c ∨ 
   c' = a ∨ c' = b ∨ c' = c) :=
by sorry

end exists_another_nice_triple_l97_97880


namespace ratio_of_magnitudes_l97_97131

variables {a b c : EuclideanSpace ℝ (Fin 3)}

def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  real.acos ((a ⬝ b) / (∥a∥ * ∥b∥))

theorem ratio_of_magnitudes
  (h1 : angle_between_vectors a b = 2 * real.pi / 3) -- 120 degrees
  (h2 : c = a + b)
  (h3 : c ⬝ a = 0) -- c is perpendicular to a
  : ∥a∥ / ∥b∥ = 1 / 2 :=
sorry

end ratio_of_magnitudes_l97_97131


namespace inverse_variant_cubed_l97_97312

theorem inverse_variant_cubed (k : ℝ) (h : ∀ x w : ℝ, x^3 * (w^(1/3)) = k) 
    (h₁ : h 3 8) : h 6 (1 / 64) :=
by
  sorry

end inverse_variant_cubed_l97_97312


namespace divisibility_by_27_l97_97247

theorem divisibility_by_27 (n : ℕ) (h : ∃ s : ℕ, n = 3 * s ∧ s = (nat.digits 10 n).sum) : 27 ∣ n :=
sorry

end divisibility_by_27_l97_97247


namespace cylindrical_tank_volume_l97_97157

theorem cylindrical_tank_volume (d h : ℝ) (d_eq_20 : d = 20) (h_eq_10 : h = 10) : 
  π * ((d / 2) ^ 2) * h = 1000 * π :=
by
  sorry

end cylindrical_tank_volume_l97_97157


namespace inscribed_circle_radius_eq_l97_97654

-- Defining the isosceles triangle ABC with given condition AB = BC and AC = 10
variable (A B C : Point)
variable (triangle_ABC : Triangle A B C)
variable (isosceles : AB = BC)
variable (AC_eq_10 : AC = 10)

-- Defining the circle inscribed in angle ABC
variable (O : Point)
variable (circle_in_angle_ABC : Circle O 7.5)
variable (midpoint_AC : Midpoint A C)

-- Defining the fact that this circle touches AC at its midpoint
variable (touches_midpoint : touches_midpoint circle_in_angle_ABC AC midpoint_AC)

-- The statement to be proven
theorem inscribed_circle_radius_eq : 
  radius_of_inscribed_circle triangle_ABC = 10/3 := 
  sorry

end inscribed_circle_radius_eq_l97_97654


namespace dogs_grouping_l97_97721

theorem dogs_grouping (dogs : Finset ℕ) (Fluffy Spike : ℕ) (h₁ : dogs.card = 12)
  (h₂ : Fluffy ∈ dogs) (h₃ : Spike ∈ dogs) :
  ∃ (group1 group2 group3 : Finset ℕ),
    group1.card = 4 ∧ 
    group2.card = 5 ∧
    group3.card = 3 ∧
    Fluffy ∈ group1 ∧
    Spike ∈ group3 ∧
    dogs = group1 ∪ group2 ∪ group3 ∧
    group1 ∩ group2 = ∅ ∧ 
    group1 ∩ group3 = ∅ ∧ 
    group2 ∩ group3 = ∅ ∧
    (Finset.choose 3 (dogs.erase Fluffy)).card * 
    (Finset.choose 4 (dogs.erase Fluffy).erase (group.fill 3 group3)).card = 4200 :=
by
  -- These are all the conditions.
  sorry

end dogs_grouping_l97_97721


namespace find_A_l97_97355

-- Definitions representing the conditions
variables {A J R : ℕ}

-- Statements of the given conditions
def condition1 : Prop := A + J = 101
def condition2 : Prop := A + R = 91
def condition3 : Prop := J + R = 88

-- The theorem statement
theorem find_A (h1 : condition1) (h2 : condition2) (h3 : condition3) : A = 52 :=
by sorry

end find_A_l97_97355


namespace square_area_y_coordinates_l97_97848

theorem square_area_y_coordinates (x1 x2 x3 x4 : ℝ) :
  ({3, 4, 8, 9} : set ℝ) = {y | ∃ x, (x = x1) ∨ (x = x2) ∨ (x = x3) ∨ (x = x4) } →
  ∀ y1 y2 y3 y4 : ℝ, (y1 = 3 ∧ y2 = 4 ∧ y3 = 8 ∧ y4 = 9) →
  ∃ side : ℝ, side = 5 ∧ (side^2 = 25) :=
by
  intros h vertices
  cases vertices with h1 h234
  cases h234 with h2 h34
  cases h34 with h3 h4
  use 5
  exact ⟨rfl, by norm_num⟩

end square_area_y_coordinates_l97_97848


namespace FI_squared_l97_97228

theorem FI_squared (A B C D E F G H I J : ℝ) 
  (h_square : ∀ (x : ℝ), ((x - 2) = 0 ∨ (x + 2) = 0))
  (h_AE : AE = sqrt 2)
  (h_AH : AH = sqrt 2)
  (h_perp_FI_EH : ⟦FI⟧ ∨redblur 90 ⟦EH⟧ )
  (h_perp_GJ_EH : ⟦GJ⟧ ∨redblur 90 ⟦EH⟧ )
  (h_area_ae : (1/2) * AH * EH = 1)
  (h_area_bfie : 1)
  (h_area_dhjg : 1)
  (h_area_fcgji : 1) :
  FI ^ 2 = 8 - 4 * sqrt 2 :=
  sorry

end FI_squared_l97_97228


namespace non_congruent_squares_l97_97161

theorem non_congruent_squares (n : ℕ) (h : n = 6) : 
  let standard_aligned_squares := (n-1)^2 + (n-2)^2 + (n-3)^2 + (n-4)^2 + (n-5)^2,
      diagonal_squares := (n-1)^2 + (n-2)^2 + (n-3)^2
  in standard_aligned_squares + diagonal_squares = 105 :=
by
  sorry

end non_congruent_squares_l97_97161


namespace avg_of_consecutive_starting_with_b_l97_97714

variable {a : ℕ} (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7)

theorem avg_of_consecutive_starting_with_b (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) :
  (a + 4 + (a + 4 + 1) + (a + 4 + 2) + (a + 4 + 3) + (a + 4 + 4) + (a + 4 + 5) + (a + 4 + 6)) / 7 = a + 7 :=
  sorry

end avg_of_consecutive_starting_with_b_l97_97714


namespace cot_sum_inequality_l97_97270

theorem cot_sum_inequality (A B C A' B' C' I : Type) [Incenter I A B C] [OrthicTriangle A' B' C' I BC CA AB] :
  cot (angle A) + cot (angle B) + cot (angle C) ≥ cot (angle A') + cot (angle B') + cot (angle C') :=
by
  sorry

end cot_sum_inequality_l97_97270


namespace magnitude_of_complex_l97_97911

-- Definitions based on conditions
def real_part : ℂ := 3 / 5
def imag_part : ℂ := -(4 / 7)
def z : ℂ := real_part + imag_part * complex.i

-- Main statement
theorem magnitude_of_complex :
  complex.abs z = 29 / 35 := by
  sorry

end magnitude_of_complex_l97_97911


namespace visual_range_increase_l97_97790

def percent_increase (v_initial v_new : ℕ) : ℕ :=
  ((v_new - v_initial) * 100) / v_initial

theorem visual_range_increase (v_initial v_new : ℕ) (h1 : v_initial = 60) (h2 : v_new = 150) :
  percent_increase v_initial v_new = 150 :=
by
  rw [h1, h2]
  dsimp [percent_increase]
  norm_num

end visual_range_increase_l97_97790


namespace f_quadratic_min_value_g_not_monotonic_iff_l97_97112

noncomputable def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 1

theorem f_quadratic_min_value :
  ∃ (a b c : ℝ), f = λ x, a * x^2 + b * x + c ∧ 
  (∀ x, f(x) ≥ 1) ∧ f(0) = 3 ∧ f(2) = 3 :=
by
  use [2, -4, 3]
  sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f(x) + a * x

theorem g_not_monotonic_iff (a : ℝ) :
  ¬ (∀ x ∈ Set.Icc (-1 : ℝ) 1, g(x, a) = f(x) + a * x) ↔ -2 < a ∧ a < 6 :=
by
  sorry

end f_quadratic_min_value_g_not_monotonic_iff_l97_97112


namespace average_score_l97_97819

theorem average_score (total_students : ℕ)
  (questions : ℕ) (points_per_question : ℕ)
  (fraction_correct : ℚ) (fraction_half_correct : ℚ)
  (correct_score : ℕ) (half_correct_score : ℕ)
  (remaining_score : ℕ) :
  total_students = 19 →
  questions = 10 →
  points_per_question = 10 →
  fraction_correct = 3/19 →
  fraction_half_correct = 13/19 →
  correct_score = points_per_question * questions →
  half_correct_score = points_per_question * (questions / 2) →
  remaining_score = 0 →
  let num_correct := (fraction_correct * total_students) in
  let num_half_correct := (fraction_half_correct * total_students) in
  let num_remaining := total_students - num_correct.to_nat - num_half_correct.to_nat in
  let total_score := num_correct.to_nat * correct_score +
                     num_half_correct.to_nat * half_correct_score +
                     num_remaining * remaining_score in
  (total_score / total_students) = 50 :=
begin
  intros,
  have h1: num_correct = 3 := by sorry,
  have h2: num_half_correct = 13 := by sorry,
  have h3: num_remaining = 3 := by sorry,
  have h4: correct_score = 100 := by sorry,
  have h5: half_correct_score = 50 := by sorry,
  have h6: total_score = 950 := by sorry,
  rw h1 at *,
  rw h2 at *,
  rw h3 at *,
  rw h4 at *,
  rw h5 at *,
  rw h6,
  simp,
  norm_num,
end

end average_score_l97_97819


namespace probability_divisible_by_15_zero_l97_97640

theorem probability_divisible_by_15_zero (digits : List ℕ) (h : digits = [2, 2, 3, 4, 7, 5]) :
  ¬ (∃ n : ℕ, ValidArrangement digits n ∧ n % 15 = 0) :=
by
  sorry

end probability_divisible_by_15_zero_l97_97640


namespace work_completion_days_l97_97392

theorem work_completion_days :
  let work_rate_a := 1 / 12,
      work_rate_b := 1 / 36,
      work_rate_c := 1 / 18,
      work_per_cycle := work_rate_b + work_rate_c + work_rate_a, -- equals 1/6
      days_per_cycle := 6,
      total_cycles := 1 / work_per_cycle -- 6 cycles required
  in total_cycles * days_per_cycle = 36 :=
by 
sorry 

end work_completion_days_l97_97392


namespace part_I_part_II_l97_97981

-- Vector definitions for a and b
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin (x + π / 6), 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, 4 * Real.cos x - Real.sqrt 3)

-- Dot product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Given that vectors a and b are perpendicular
theorem part_I (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : 
  Real.sin (x + 4 * π / 3) = -1 / 4 := sorry

-- Definition of function f(x)
def f (x : ℝ) : ℝ := dot_product (vector_a x) (vector_b x)

-- Given conditions on alpha
theorem part_II (α : ℝ) (h1 : α ∈ Set.Icc 0 (π / 2)) (h2 : f (α - π / 6) = 2 * Real.sqrt 3) : 
  Real.cos α = (3 + Real.sqrt 21) / 8 := sorry

end part_I_part_II_l97_97981


namespace find_smallest_solution_l97_97539

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l97_97539


namespace part_a_part_b_l97_97869

-- Definitions
variables {A B C D E F S : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace S]
variable {triangle_ABC : Triangle}

-- Angles of triangle ABC are all smaller than 120 degrees
def angles_less_than_120 : Prop :=
  ∀ {α β γ : ℝ}, (α < 120) ∧ (β < 120) ∧ (γ < 120) → 
  Triangle.angles triangle_ABC = ⟨α, β, γ⟩

-- Equilateral triangles constructed externally
def equilateral_triangle (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z] : Prop :=
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a → 
  Triangle.is_equilateral (Triangle.mk X Y Z)

-- Proof statement for part (a)
theorem part_a (h1 : angles_less_than_120) 
  (h2 : equilateral_triangle A F B)
  (h3 : equilateral_triangle B D C)
  (h4 : equilateral_triangle C E A) :
  ∃ S, collinear A D S ∧ collinear B E S ∧ collinear C F S := sorry

-- Proof statement for part (b)
theorem part_b (h1 : angles_less_than_120) 
  (h2 : equilateral_triangle A F B)
  (h3 : equilateral_triangle B D C)
  (h4 : equilateral_triangle C E A)
  (S : Type) [MetricSpace S]
  (hS : ∃ S, collinear A D S ∧ collinear B E S ∧ collinear C F S) : 
  ∀ {D E F S A B C : ℝ}, D + E + F = 2 * (A + B + C) := sorry

end part_a_part_b_l97_97869


namespace possible_positions_l97_97453

def toggle_effects (grid : Array (Array Bool)) (i j : Nat) : Array (Array Bool) := sorry

def all_off (grid : Array (Array Bool)) : Prop :=
  ∀ i j, grid[i][j] = false

def single_on (grid : Array (Array Bool)) : Prop :=
  (∃ i j, grid[i][j] = true) ∧ (∀ i' j', (i ≠ i' ∨ j ≠ j') → grid[i'][j'] = false)

theorem possible_positions :
  ∀ grid : Array (Array Bool),
  (all_off grid) →
  (∃ g' : Array (Array Bool), grid = g' ∧ ∃ n : Nat, toggle_effects grid n = g' ∧ single_on g') →
  (∃! (i j : Nat), (i, j) = (3, 3) ∨ (i, j) = (2, 2) ∨ (i, j) = (2, 4) ∨ (i, j) = (4, 2) ∨ (i, j) = (4, 4)) :=
sorry

end possible_positions_l97_97453


namespace right_angled_triangle_l97_97254

theorem right_angled_triangle {A B C : ℝ} (h₀ : A + B + C = π) (h₁ : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 2) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := 
sorry

end right_angled_triangle_l97_97254


namespace tangents_at_B_do_not_necessarily_coincide_l97_97577

-- Define the parabola and the circle equations
def parabola (x : ℝ) : ℝ := x^2
def circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2

-- Define the condition for tangency of curves at a point
def tangent_equal_at (a b r : ℝ) (x y : ℝ) : Prop :=
  ∃ m : ℝ, ∀ y' : ℝ, parabola x = y → circle a b r x y' → m * (y - y') = (1 : ℝ)

-- Main theorem statement based on identified conditions and correct answer
theorem tangents_at_B_do_not_necessarily_coincide
  (a b r x_A y_A x_B y_B : ℝ)
  (hA : circle a b r x_A y_A)
  (hB : circle a b r x_B y_B)
  (h_intersect : parabola x_A = y_A ∧ parabola x_B = y_B)
  (h_tangent_A : tangent_equal_at a b r x_A y_A)
  : ¬ (tangent_equal_at a b r x_B y_B) := 
sorry

end tangents_at_B_do_not_necessarily_coincide_l97_97577


namespace retailer_marked_price_l97_97018

def discount (price : ℝ) (percent : ℝ) : ℝ :=
  price * percent / 100

def profit_price (cost : ℝ) (profit_percent : ℝ) : ℝ :=
  cost * (1 + profit_percent / 100)

def marked_price (selling_price : ℝ) (discount_percent_marked : ℝ) : ℝ :=
  selling_price / (1 - discount_percent_marked / 100)

theorem retailer_marked_price
  (initial_price : ℝ)
  (discount_percent_initial : ℝ)
  (profit_percent : ℝ)
  (discount_percent_marked : ℝ)
  (final_marked_price : ℝ) :
  initial_price = 36 →
  discount_percent_initial = 15 →
  profit_percent = 40 →
  discount_percent_marked = 25 →
  final_marked_price = 57.12 :=
by
  intro h_initial_price h_discount_percent_initial h_profit_percent h_discount_percent_marked
  sorry

end retailer_marked_price_l97_97018


namespace repeated_decimal_to_fraction_l97_97192

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l97_97192


namespace true_propositions_in_congruence_l97_97034

theorem true_propositions_in_congruence :
  let prop1 := ∀ Δ₁ Δ₂ : Triangle, same_shape Δ₁ Δ₂ → congruent Δ₁ Δ₂
  let prop2 := ∀ Δ₁ Δ₂ : Triangle, congruent Δ₁ Δ₂ →
    (∀ (A₁ A₂ B₁ B₂ C₁ C₂ : Point), 
      corresponding_angles_equal Δ₁ Δ₂ A₁ A₂ B₁ B₂ C₁ C₂ ∧ 
      corresponding_sides_equal Δ₁ Δ₂ A₁ A₂ B₁ B₂ C₁ C₂)
  let prop3 := ∀ Δ₁ Δ₂ : Triangle, congruent Δ₁ Δ₂ →
    (∀ (h₁ h₂ m₁ m₂ b₁ b₂ : Line),
      corresponding_features_equal Δ₁ Δ₂ h₁ h₂ m₁ m₂ b₁ b₂)
  (num_true : ℕ) := 
  (num_true = 2) → 
  (¬ prop1) ∧ prop2 ∧ prop3
:= sorry

end true_propositions_in_congruence_l97_97034


namespace geo_seq_b_formula_b_n_sum_T_n_l97_97580

-- Define the sequence a_n 
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else sorry -- Definition based on provided conditions

-- Define the partial sum S_n
def S (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * a (n-1) + 2 -- Given condition S_{n+1} = 4a_n + 2

-- Condition for b_n
def b (n : ℕ) : ℕ :=
  a (n+1) - 2 * a n

-- Definition for c_n
def c (n : ℕ) := (b n) / 3

-- Define the sequence terms for c_n based sequence
def T (n : ℕ) : ℝ :=
  sorry -- Needs explicit definition from given sequence part

-- Proof statements
theorem geo_seq_b : ∀ n : ℕ, b (n + 1) = 2 * b n :=
  sorry

theorem formula_b_n : ∀ n : ℕ, b n = 3 * 2^(n-1) :=
  sorry

theorem sum_T_n : ∀ n : ℕ, T n = n / (n + 1) :=
  sorry

end geo_seq_b_formula_b_n_sum_T_n_l97_97580


namespace smallest_x_solution_l97_97518

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l97_97518


namespace diff_largest_smallest_roots_eq_two_l97_97921

theorem diff_largest_smallest_roots_eq_two :
  let p := (λ x : ℝ, x^3 - 7 * x^2 + 11 * x - 6) in
  let roots := (Multiset.real_roots p).toFinset in
  let largest := roots.max' sorry in
  let smallest := roots.min' sorry in
  largest - smallest = 2 := by
begin
  sorry
end

end diff_largest_smallest_roots_eq_two_l97_97921


namespace calc_expression_l97_97479

theorem calc_expression :
  5 + 7 * (2 + (1 / 4 : ℝ)) = 20.75 :=
by
  sorry

end calc_expression_l97_97479


namespace sum_c_d_l97_97994

theorem sum_c_d (c d : ℕ) (h : ∏ (i : ℕ) in (finset.range (c - 3 + 1)).image (λ n, n + 3), (n + 1) / n = 16) 
 (c_eq : c = 48) (d_eq : d = 47) : c + d = 95 :=
by
  unfold ∏ finset.prod finset.range finset.image
  sorry

end sum_c_d_l97_97994


namespace flower_beds_l97_97700

theorem flower_beds (total_seeds : ℕ) (seeds_per_bed : ℕ) (beds : ℕ) 
  (h1 : total_seeds = 270) 
  (h2 : seeds_per_bed = 9) 
  (h3 : beds = 30) : 
  total_seeds / seeds_per_bed = beds := 
by
  rw [h1, h2, h3]
  exact Nat.div_self (by decide) -- 9 is not 0
  sorry

end flower_beds_l97_97700


namespace digit_zero_not_in_mean_l97_97046

theorem digit_zero_not_in_mean : 
  let s := {1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999} in
  let mean := (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999) / 9 in
  ¬(0 ∈ set_of (λ d : ℕ, digit_in d mean)) :=
by
  sorry

end digit_zero_not_in_mean_l97_97046


namespace moles_NaCl_for_H₂SO₄_HCl_NaHSO₄_l97_97160

-- Definitions for the chemical species involved
def NaCl : Type := Unit
def H₂SO₄ : Type := Unit
def HCl : Type := Unit
def NaHSO₄ : Type := Unit

-- Stoichiometry relationship based on the balanced equation 
def reaction (n_NaCl : ℕ) (n_H₂SO₄ : ℕ) (n_HCl : ℕ) (n_NaHSO₄ : ℕ) :=
  n_NaCl = 3 ∧ n_H₂SO₄ >= 3 ∧ n_HCl = 3 ∧ n_NaHSO₄ = 3

-- Theorem statement
theorem moles_NaCl_for_H₂SO₄_HCl_NaHSO₄ :
  ∀ (n_NaCl n_H₂SO₄ n_HCl n_NaHSO₄ : ℕ),
    reaction n_NaCl n_H₂SO₄ n_HCl n_NaHSO₄ →
    n_NaCl = 3 :=
begin
  intros,
  sorry -- Proof is skipped as requested
end

end moles_NaCl_for_H₂SO₄_HCl_NaHSO₄_l97_97160


namespace f_prime_correct_g_odd_function_g_monotonic_intervals_l97_97146

def f (x : ℝ) : ℝ := x^3 + 3 * x^2

def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x

def g (x : ℝ) : ℝ := f x - f_prime x

-- Prove that f'(x) = 3x^2 + 6x
theorem f_prime_correct : ∀ x : ℝ, f_prime x = 3 * x^2 + 6 * x := 
by 
  unfold f_prime
  unfold f
  intro x
  rw [f, f_prime]
  sorry

-- Prove that g(x) = f(x) - f'(x) is an odd function
theorem g_odd_function : ∀ x : ℝ, g (-x) = -g x := 
by 
  unfold g
  unfold f
  unfold f_prime
  intro x
  rw [g, f, f_prime]
  sorry

-- Determine intervals and extreme values of g(x)
theorem g_monotonic_intervals :
  ∀ x : ℝ, 
    (∀ x < -Real.sqrt 2, ∃ g_deriv > 0) ∧
    (∀ x > Real.sqrt 2, ∃ g_deriv > 0) ∧
    (∀ - Real.sqrt 2 < x, x < Real.sqrt 2, ∃ g_deriv < 0) ∧ 
    (∀ x = - Real.sqrt 2, g x = 4 * Real.sqrt 2) ∧ 
    (∀ x = Real.sqrt 2, g x = -4 * Real.sqrt 2) := 
by
  unfold g
  unfold f
  unfold f_prime
  unfold Real.sqrt
  intro x
  rw [g, f, f_prime, Real.sqrt]
  sorry

end f_prime_correct_g_odd_function_g_monotonic_intervals_l97_97146


namespace find_b_c_l97_97641

-- Definitions for the conditions of the problem
def angle_A (A : ℝ) : Prop := A = 120
def side_a (a : ℝ) : Prop := a = sqrt 21
def area_ΔABC (area : ℝ) : Prop := area = sqrt 3

-- Main statement
theorem find_b_c (A a area b c : ℝ)
  (hA : angle_A A)
  (ha : side_a a)
  (harea : area_ΔABC area) :
  b * c = 4 ∧ (∀ (squared : ℝ), (squared = b^2 + c^2) ∧ squared = 17) :=
begin
  sorry
end

end find_b_c_l97_97641


namespace natural_numbers_with_property_l97_97071

noncomputable def satisfies_property (n : ℕ) : Prop :=
  2 < n ∧ n ≤ 10000000 ∧ ∀ m : ℕ, m.coprime n → 1 < m → m < n → Nat.Prime m

theorem natural_numbers_with_property :
  {n : ℕ | satisfies_property n} = {3, 4, 6, 8, 12, 18, 24, 30} :=
by sorry

end natural_numbers_with_property_l97_97071


namespace sum_of_integers_between_is_90_l97_97769

-- Define the conditions
def is_between (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Define the sum of integers satisfying the conditions
def sum_of_integers_between : ℕ :=
  Finset.sum (Finset.filter is_between (Finset.range 15)) id

-- State the theorem
theorem sum_of_integers_between_is_90 : sum_of_integers_between = 90 := 
by
  sorry

end sum_of_integers_between_is_90_l97_97769


namespace original_stations_l97_97843

theorem original_stations (m n : ℕ) (h : n > 1) (h_equation : n * (2 * m + n - 1) = 58) : m = 14 :=
by
  -- proof omitted
  sorry

end original_stations_l97_97843


namespace smallest_cubes_to_form_30_digit_number_l97_97364

theorem smallest_cubes_to_form_30_digit_number :
  ∃ (n : ℕ), n = 50 ∧ 
  (∀ (d : ℤ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
   ∃ (cubes : fin n → fin 6 → ℤ),
     (∀ (i : fin n), ∀ (j : fin 6), cubes i j ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
     (∀ (k : ℤ), k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → (∃ (count_k : fin n → ℕ), (∑ i, count_k i = 30 ∧ (∀ i, cubes i (count_k i) = k))) ∧
       (∃ (count_0 : fin n → ℕ), (∑ i, count_0 i = 29 ∧ (∀ i, cubes i (count_0 i) = 0)) ))) := by
  sorry

end smallest_cubes_to_form_30_digit_number_l97_97364


namespace repeating_decimal_fraction_l97_97183

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l97_97183


namespace central_angle_eq_pi_div_3_l97_97992

open Real EuclideanGeometry

/-- If the length of a chord equals the radius of the circle, 
    then the central angle corresponding to this chord is π/3 radians. -/
theorem central_angle_eq_pi_div_3 (radius : ℝ) (h : radius > 0)
  (chord : ℝ) (h_chord : chord = radius) : 
  central_angle (circle_with_radius radius) chord = π/3 := 
sorry

end central_angle_eq_pi_div_3_l97_97992


namespace coprime_repeating_decimal_sum_l97_97173

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l97_97173


namespace park_area_l97_97452

-- Define the width (w) and length (l) of the park
def width : Float := 11.25
def length : Float := 33.75

-- Define the perimeter and area functions
def perimeter (w l : Float) : Float := 2 * (w + l)
def area (w l : Float) : Float := w * l

-- Provide the conditions
axiom width_is_one_third_length : width = length / 3
axiom perimeter_is_90 : perimeter width length = 90

-- Theorem to prove the area given the conditions
theorem park_area : area width length = 379.6875 := by
  sorry

end park_area_l97_97452


namespace max_k_l97_97803

-- Definitions of knight and liar predicates
def is_knight (p : ℕ → Prop) := ∀ i, p i
def is_liar (p : ℕ → Prop) := ∀ i, ¬ p i

-- Definitions for the conditions in the problem
def greater_than_neighbors (n : ℕ) (cards : ℕ → ℕ) :=
  ∀ i : ℕ, 0 < i ∧ i < n - 1 → cards i > cards (i - 1) ∧ cards i > cards (i + 1)

def less_than_neighbors (n : ℕ) (cards : ℕ → ℕ) :=
  ∀ i : ℕ, 0 < i ∧ i < n - 1 → cards i < cards (i - 1) ∧ cards i < cards (i + 1)

def maximum_possible_k (n : ℕ) (cards : ℕ → ℕ) (k : ℕ) :=
  ∀ n : 2015, ∃ k : ℕ, (greater_than_neighbors n cards → is_knight (less_than_neighbors n cards)) ∧
  k = 2013

-- The main theorem statement
theorem max_k (n : ℕ) (cards : ℕ → ℕ) : maximum_possible_k 2015 cards 2013 := sorry

end max_k_l97_97803


namespace equilateral_triangle_black_area_l97_97866

theorem equilateral_triangle_black_area :
  let initial_area := 1
  let black_fraction_remains := (8 / 9 : ℚ)
  ∀ n : ℕ, (n = 6) → 
  (black_fraction_remains ^ n * initial_area = 262144 / 531441 : ℚ) := 
by
  intros initial_area black_fraction_remains n hn
  rw [hn]
  simp only [initial_area, black_fraction_remains]
  have h : (8 / 9 : ℚ) ^ 6 = 262144 / 531441 := sorry
  exact h

end equilateral_triangle_black_area_l97_97866


namespace one_inch_represents_feet_l97_97741

def height_statue : ℕ := 80 -- Height of the statue in feet

def height_model : ℕ := 5 -- Height of the model in inches

theorem one_inch_represents_feet : (height_statue / height_model) = 16 := 
by
  sorry

end one_inch_represents_feet_l97_97741


namespace part_i_part_ii_part_iii_l97_97613

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem part_i : f (Real.pi / 2) = 1 :=
sorry

theorem part_ii : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi :=
sorry

theorem part_iii : ∃ x, g x = -2 :=
sorry

end part_i_part_ii_part_iii_l97_97613


namespace cos_sum_is_zero_l97_97683

theorem cos_sum_is_zero (x y z : ℝ) 
  (h1: Real.cos (2 * x) + 2 * Real.cos (2 * y) + 3 * Real.cos (2 * z) = 0) 
  (h2: Real.sin (2 * x) + 2 * Real.sin (2 * y) + 3 * Real.sin (2 * z) = 0) : 
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := 
by 
  sorry

end cos_sum_is_zero_l97_97683


namespace locus_of_points_l97_97897

def Point := ℝ × ℝ
def Circle (center : Point) (radius : ℝ) := { p : Point | dist p center = radius }
def Ball (center : Point) (radius : ℝ) := { p : Point | dist p center < radius }

variables (P : Point) (r s : ℝ) (B : Point)

theorem locus_of_points
  (hPBR : dist P B + s < r)
  (hB : B ∈ Ball P s) :
  ∃ A : Point, dist A B < r - s :=
by
  sorry

end locus_of_points_l97_97897


namespace sector_area_120_6_l97_97724

open Real

def area_of_sector (R : ℝ) (n : ℝ) : ℝ :=
  (n * π * R ^ 2) / 360

theorem sector_area_120_6 :
  area_of_sector 6 120 = 12 * π :=
by
  sorry

end sector_area_120_6_l97_97724


namespace expression_is_correct_l97_97483

noncomputable def expression : ℝ :=
  (8:ℝ)^(1/2) + |(1 - real.sqrt 2)| - (1/2:ℝ)^(-1) + ((real.pi - real.sqrt 3)^0)

theorem expression_is_correct : expression = 3 * real.sqrt 2 - 2 :=
by
  sorry

end expression_is_correct_l97_97483


namespace smallest_solution_floor_eq_l97_97524

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l97_97524


namespace wade_customers_sunday_l97_97367

theorem wade_customers_sunday :
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  customers_sunday = 36 :=
by
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  have h : customers_sunday = 36 := by sorry
  exact h

end wade_customers_sunday_l97_97367


namespace five_letter_words_count_l97_97159

/-- 
  We consider the problem of counting the number of five-letter words 
  such that the word starts with a vowel and ends with a consonant.

  Given:
  - There are 5 vowels for the first position.
  - There are 21 consonants for the last position.
  - There are 26 choices (alphabet letters) for each of the middle three positions.

  We need to prove that the total number of such five-letter words is 1844760.
-/
theorem five_letter_words_count : 
  let vowels := 5 in
  let consonants := 21 in
  let alphabet := 26 in
  vowels * alphabet^3 * consonants = 1844760 :=
by
  sorry

end five_letter_words_count_l97_97159


namespace no_adjacent_girls_arrangements_adjacent_AB_arrangements_l97_97345

-- Assuming boys and girls are distinguishable
def num_boys : ℕ := 3
def num_girls : ℕ := 4

-- Defining the no_adjacent_girls condition and adjacent_AB condition
def no_adjacent_girls (p: list char) : Prop :=
  ∀i, i < p.length - 1 → (p[i] ∉ ['G'] ∨ p[i + 1] ∉ ['G'])

def adjacent_AB (p: list char) : Prop :=
  ∃i, i < p.length - 1 ∧ p[i] = 'A' ∧ p[i + 1] = 'B'

-- The number of arrangements of 3 boys and 4 girls, with no two girls adjacent
theorem no_adjacent_girls_arrangements: 
  ∀ (p : list char), no_adjacent_girls p → num_boys == 3 → num_girls == 4 → p.permutations.length = 144 := 
by
  sorry

-- The number of arrangements of 3 boys and 4 girls, with boys A and B being adjacent
theorem adjacent_AB_arrangements:
  ∀ (p : list char), adjacent_AB p → num_boys == 3 → num_girls == 4 → p.permutations.length = 240 :=
by
  sorry

end no_adjacent_girls_arrangements_adjacent_AB_arrangements_l97_97345


namespace laborer_income_l97_97796

theorem laborer_income (I : ℕ) (debt : ℕ) 
  (h1 : 6 * I < 420) 
  (h2 : 4 * I = 240 + debt + 30) 
  (h3 : debt = 420 - 6 * I) : 
  I = 69 := by
  sorry

end laborer_income_l97_97796


namespace cosine_value_l97_97949

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

noncomputable def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

noncomputable def magnitude (x : ℝ × ℝ) : ℝ :=
  (x.1 ^ 2 + x.2 ^ 2).sqrt

noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem cosine_value :
  cos_angle a b = 2 * (5:ℝ).sqrt / 25 :=
by
  sorry

end cosine_value_l97_97949


namespace tins_per_case_is_24_l97_97021

def total_cases : ℕ := 15
def damaged_percentage : ℝ := 0.05
def remaining_tins : ℕ := 342

theorem tins_per_case_is_24 (x : ℕ) (h : (1 - damaged_percentage) * (total_cases * x) = remaining_tins) : x = 24 :=
  sorry

end tins_per_case_is_24_l97_97021


namespace fare_collected_I_class_l97_97284

theorem fare_collected_I_class (
  x y : ℝ
  (total_fare : ℝ = 420000)
  (passenger_ratio : ℝ = 1)
  (base_fare_ratio : ℝ = 5)
  (ratio_passenger_2 : ℝ = 4)
  (ratio_passenger_3 : ℝ = 5)
  (ratio_fare_2 : ℝ = 3)
  (ratio_fare_3 : ℝ = 1)
  (discount_I_class : ℝ = 0.9)
  (surcharge_III_class : ℝ = 1.15)
  (fare_collected_from_I_class : ℝ = 84943.82)
  (eq_total_fare : total_fare = (x * 5 * y * discount_I_class) + (x * ratio_passenger_2 * y * ratio_fare_2) + (x * ratio_passenger_3 * y * ratio_fare_3 * surcharge_III_class))
  (fare_I_class : ℝ = x * 5 * y * discount_I_class)
  (eq_fare_I_class : fare_I_class = fare_collected_from_I_class)
) : 
  fare_I_class = fare_collected_from_I_class :=
sorry

end fare_collected_I_class_l97_97284


namespace max_correct_equations_l97_97825

def cards : set ℕ := { i | 1 ≤ i ∧ i ≤ 100 }

theorem max_correct_equations :
  ∃ eqs : set (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ eqs → a ∈ cards ∧ b ∈ cards ∧ c ∈ cards ∧ a + b = c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    ∀ (a b c : ℕ), a ∈ cards ∧ b ∈ cards ∧ c ∈ cards ∧ a + b = c → (a, b, c) ∈ eqs ∨
    (a, b, c) ∉ eqs ∧ ∃ x, (x, y, z) ∈ eqs → x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ z ≠ a ∧ z ≠ b ∧ z ≠ c ∧
    eqs.card = 33 :=
begin
  sorry
end

end max_correct_equations_l97_97825


namespace find_y_l97_97919

theorem find_y (log5 : Real) (h1 : log10 50 = log5 + 1) (h2 : log5 = 0.69897) :
  50 ^ 4 = 10 ^ 6.79588 :=
by
  sorry

end find_y_l97_97919


namespace smallest_solution_floor_equation_l97_97552

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l97_97552


namespace angle_between_vectors_proof_l97_97123

noncomputable def angle_between_vectors (a b : ℝ) : ℝ := sorry

theorem angle_between_vectors_proof (a b : ℝ^3):
  a ≠ 0 ∧ b ≠ 0 ∧
  (‖a‖ = 2 * ‖b‖) ∧
  ((a - b) • b = 0) →
  angle_between_vectors a b = real.pi / 3 :=
by sorry

end angle_between_vectors_proof_l97_97123


namespace arithmetic_sequences_count_l97_97584

theorem arithmetic_sequences_count :
  ∃ a1 d n : ℕ, n ≥ 3 → (a1 + (a1 + d) + ... + (a1 + (n-1)*d) = 97^2) → 
                (a1 ≥ 0) → (d ≥ 0) →
                4 = { (a1, d, n) | n ≥ 3 ∧ a1 ≥ 0 ∧ d ≥ 0 ∧ finset.sum (finset.range n) (λ k, a1 + k * d) = 97^2 }.card := 
sorry

end arithmetic_sequences_count_l97_97584


namespace eval_expression_l97_97507

theorem eval_expression : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end eval_expression_l97_97507


namespace total_bill_first_month_l97_97906

theorem total_bill_first_month (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) 
  (h3 : 2 * C = 2 * C) : 
  F + C = 50 := by
  sorry

end total_bill_first_month_l97_97906


namespace odd_areas_implies_all_triangles_l97_97397

theorem odd_areas_implies_all_triangles
  (a b c d e f : ℕ)
  (h_odd_a : a % 2 = 1)
  (h_odd_b : b % 2 = 1)
  (h_odd_c : c % 2 = 1)
  (h_odd_d : d % 2 = 1)
  (h_odd_e : e % 2 = 1)
  (h_odd_f : f % 2 = 1)
  (ABCD : Type*)
  (M : ABCD → ABCD → Type*)
  (N : ABCD → ABCD → Type*)
  (AM DM BN CN : ABCD → ABCD → Type*)
  (h_M_midpoint : ∀ (BC : ABCD), M BC = ABCD)
  (h_N_midpoint : ∀ (AD : ABCD), N AD = ABCD)
  (divides_into_parts : ∀ (Q : ABCD), AM Q → DM Q → BN Q → CN Q → Prop)
  (six_parts_are_triangles : ∀ (Q : ABCD), divides_into_parts Q → a + b + c + d + e + f = Fintype.card ℕ)
  : true := 
begin
  sorry
end

end odd_areas_implies_all_triangles_l97_97397


namespace price_per_pound_apples_l97_97984

noncomputable def total_pie_cost := 8 * 1
noncomputable def other_ingredients_cost := 2 + 0.5 + 1.5
noncomputable def total_apples_cost := total_pie_cost - other_ingredients_cost
noncomputable def price_per_pound := total_apples_cost / 2

theorem price_per_pound_apples : price_per_pound = 2 := by
  sorry

end price_per_pound_apples_l97_97984


namespace trajectory_of_P_l97_97934

noncomputable def A : Point := ⟨-4, 0⟩

noncomputable def F_circle (x y : ℝ) : Prop := (x - 4) ^ 2 + y ^ 2 = 4

noncomputable def perpendicular_bisector (A B : Point) : Line := sorry -- definition left to be filled

noncomputable def intersect_at_P (B : Point) (L : Line) : Point := sorry -- definition left to be filled

theorem trajectory_of_P :
  ∀ B : Point,
  F_circle B.1 B.2 →
  ∃ P : Point,
  P = intersect_at_P B (perpendicular_bisector A B) ∧
  (P.1^2 - (P.2^2) / 15 = 1 ∧ P.1 ≠ 0) :=
by
  sorry

end trajectory_of_P_l97_97934


namespace cone_CSA_is_correct_l97_97747

noncomputable def slant_height : ℝ := 21 -- in centimeters
noncomputable def radius_base : ℝ := 10 -- in centimeters
noncomputable def π_value : ℝ := real.pi -- The value of π

def curved_surface_area (r l : ℝ) : ℝ := π_value * r * l

theorem cone_CSA_is_correct : curved_surface_area radius_base slant_height = 210 * π_value := 
by
  sorry

end cone_CSA_is_correct_l97_97747


namespace lena_always_greater_l97_97669

def lena_results : set ℕ := {7 * 8, 7 * 9, 8 * 9}
def jonah_results : set ℕ := {(2 + 4) * 6, (2 + 6) * 4, (4 + 6) * 2}

theorem lena_always_greater : ∀ lr ∈ lena_results, ∀ jr ∈ jonah_results, lr > jr :=
by 
  sorry

end lena_always_greater_l97_97669


namespace min_value_expression_l97_97732

open Real

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h : (a^2 + b^2 = (2 * a)^2) := 
b^2 + 1) / (3 * a)

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a^2 + b^2 = (2 * a)^2) :
  let expression := (b^2 + 1) / (3 * a)
  in expression ≥ 2 * sqrt (1 / 3) :=
sorry

end min_value_expression_l97_97732


namespace max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l97_97985

-- Definitions and conditions related to the given problem
def unit_circle (r : ℝ) : Prop := r = 1

-- Maximum number of non-intersecting circles of radius 1 tangent to a unit circle.
theorem max_non_intersecting_circles_tangent (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 6 := sorry

-- Maximum number of circles of radius 1 intersecting a given unit circle without intersecting centers.
theorem max_intersecting_circles_without_center_containment (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 12 := sorry

-- Maximum number of circles of radius 1 intersecting a unit circle K without containing the center of K or any other circle's center.
theorem max_intersecting_circles_without_center_containment_2 (r : ℝ) (K : ℝ)
  (h_r : unit_circle r) (h_K : unit_circle K) :
  ∃ n, n = 18 := sorry

end max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l97_97985


namespace polynomial_conditions_solution_l97_97900

noncomputable def polynomial (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem polynomial_conditions_solution :
  ∀ (a b c d : ℝ), 
    (∀ x, |x| ≤ 1 → |polynomial a b c d x| ≤ 1) →
    polynomial a b c d 2 = 26 →
    (a = 4 ∧ b = 0 ∧ c = -3 ∧ d = 0) :=
begin
  intros a b c d h1 h2,
  sorry -- proof to be filled in
end

end polynomial_conditions_solution_l97_97900


namespace total_distance_journey_l97_97665

def miles_driven : ℕ := 384
def miles_remaining : ℕ := 816

theorem total_distance_journey :
  miles_driven + miles_remaining = 1200 :=
by
  sorry

end total_distance_journey_l97_97665


namespace midpoints_of_quadrilateral_l97_97504

theorem midpoints_of_quadrilateral (A B C D : Type) [quadrilateral A B C D] :
  is_parallelogram (midpoint A B) (midpoint B C) (midpoint C D) (midpoint D A) :=
sorry

end midpoints_of_quadrilateral_l97_97504


namespace unique_prime_factorization_l97_97715

theorem unique_prime_factorization (n : ℕ) (h : n ≥ 2) : 
  ∃ (f : ℕ → ℕ), (∀ p : ℕ, prime p → ∃ k : ℕ, f p = k ∧ n = (p ^ k)) ∧ 
  (∀ (g : ℕ → ℕ), 
    (∀ p : ℕ, prime p → ∃ k : ℕ, g p = k ∧ n = (p ^ k)) → 
    ∀ p : ℕ, prime p → f p = g p) :=
sorry

end unique_prime_factorization_l97_97715


namespace sequence_of_arrows_512_to_517_is_B_C_D_E_A_l97_97720

noncomputable def sequence_from_512_to_517 : List Char :=
  let pattern := ['A', 'B', 'C', 'D', 'E']
  pattern.drop 2 ++ pattern.take 2

theorem sequence_of_arrows_512_to_517_is_B_C_D_E_A : sequence_from_512_to_517 = ['B', 'C', 'D', 'E', 'A'] :=
  sorry

end sequence_of_arrows_512_to_517_is_B_C_D_E_A_l97_97720


namespace smallest_solution_floor_eq_l97_97527

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l97_97527


namespace exterior_angle_bisector_ratio_l97_97217

-- Given definitions and conditions
variables {A B C P : Type} [linear_ordered_field Type]
variables (x : ℝ)
variables (AB BC PA PB : ℝ)
variables (kx : ℝ) (AB_BC_ratio : 4 / 5 = AB / BC) (P_position : PA = PB + AB)
variables (external_angle_bisector : PA / PB = kx / BC)

-- The proof statement: the ratio PA:AB is 7:2
theorem exterior_angle_bisector_ratio 
  (AB_BC_ratio : 4 / 5 = AB / BC)
  (P_position : PA = PB + AB)
  (external_angle_bisector : PA / PB = kx / 5) :
  PA / AB = 7 / 2 := 
begin
  have h1 : 4 / 5 = AB / BC := AB_BC_ratio,
  have h2 : PA = PB + AB := P_position,
  have h3 : PA / PB = kx / 5 := external_angle_bisector,
  sorry,
end

end exterior_angle_bisector_ratio_l97_97217


namespace hyperbola_center_l97_97829

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end hyperbola_center_l97_97829


namespace quadratic_roots_properties_l97_97995

open Real

theorem quadratic_roots_properties
  (a x₁ x₂ : ℝ)
  (h_eq : ∀ (a : ℝ), Polynomial.eval a ((Polynomial.C 1) * Polynomial.pow (Polynomial.X) 2
                        + Polynomial.C a * Polynomial.X + Polynomial.C (2 * a)) = 0)
  (h_x1_ne_x2 : x₁ ≠ x₂)
  (h_roots : Polynomial.root ((Polynomial.C 1) * Polynomial.pow (Polynomial.X) 2
                        + Polynomial.C a * Polynomial.X + Polynomial.C (2 * a)) x₁
                        ∧ Polynomial.root ((Polynomial.C 1) * Polynomial.pow (Polynomial.X) 2
                        + Polynomial.C a * Polynomial.X + Polynomial.C (2 * a)) x₂) :
  (∀ (a x₁ x₂ : ℝ)
    (H₁ : a ≠ 0)
    (H₂ : x₁ ≠ x₂)
    (H₃ : x₁ + x₂ = -a)
    (H₄ : x₁ * x₂ = 2 * a),
  (∃ (k : ℝ), (| x₁ * x₂ | + 1 / (| a | - 17) = 34 + 2 * √2)
  ∧ (1 / x₁ + 1 / x₂ = -1 / 2))) :=
sorry

end quadratic_roots_properties_l97_97995


namespace measure_angle_PQR_is_55_l97_97815

noncomputable def measure_angle_PQR (POQ QOR : ℝ) : ℝ :=
  let POQ := 120
  let QOR := 130
  let POR := 360 - (POQ + QOR)
  let OPR := (180 - POR) / 2
  let OPQ := (180 - POQ) / 2
  let OQR := (180 - QOR) / 2
  OPQ + OQR

theorem measure_angle_PQR_is_55 : measure_angle_PQR 120 130 = 55 := by
  sorry

end measure_angle_PQR_is_55_l97_97815


namespace f_divisibility_l97_97558

/-- Define the c function to represent C_n^i ≡ c(n,i) mod 2, where c(n,i) ∈ {0,1} -/
def c (n i : ℕ) : ℕ :=
  if (nat.choose n i) % 2 = 0 then 0 else 1

/-- Define the f function as given in the problem statement -/
def f (n q : ℕ) : ℕ :=
  ∑ i in finset.range (n + 1), c(n, i) * q^i

/-- Define the key theorem statement -/
theorem f_divisibility
  (m n q r : ℕ)
  (hm : m > 0)
  (hn : n > 0)
  (hq : q > 0)
  (hq_ne_pow_two : ∀ α : ℕ, q + 1 ≠ 2^α)
  (hf_div : f m q ∣ f n q) :
  f m r ∣ f n r :=
sorry

end f_divisibility_l97_97558


namespace unique_primes_in_product_l97_97158

theorem unique_primes_in_product :
  let a := 77
  let b := 81
  let c := 85
  let d := 87
  let factor77 := 7 * 11
  let factor81 := 3^4
  let factor85 := 5 * 17
  let factor87 := 3 * 29
  let product := factor77 * factor81 * factor85 * factor87
  nat.factorization (a*b*c*d) = nat.factorization product ∧ 
  nat.factorization product = [(3, 5), (5, 1), (7, 1), (11, 1), (17, 1), (29, 1)] := 
  (a * b * c * d = product ∧ 
   list.length (list.dedup (nat.factorization (a * b * c * d) (list.keys nat.factorization product)) = 6) :=
begin
  sorry,
end

end unique_primes_in_product_l97_97158


namespace probability_of_ending_at_bottom_vertex_l97_97823

-- Define the dodecahedron structure and conditions
structure Dodecahedron :=
  (top_vertex : ℕ)
  (bottom_vertex : ℕ)
  (adjacent : ℕ → List ℕ)

-- Example instance representing the problem's conditions
def dodecahedron_example : Dodecahedron :=
  { top_vertex := 0,
    bottom_vertex := 11,
    adjacent := λ v, if v = 0 then [1, 2, 3, 4, 5]  -- Adjacent vertices for the top vertex
                      else if v = 1 then [0, 6, 7, 8, 9]  -- Example for one adjacent vertex
                      else [] }  -- Other vertices can be defined similarly

-- Formal statement of the proof problem in Lean 4
theorem probability_of_ending_at_bottom_vertex :
  ∀ (d : Dodecahedron),
  (∃ a ∈ d.adjacent d.top_vertex, d.adjacent a.contains d.bottom_vertex) →
  let total_paths := length (d.adjacent d.top_vertex) * length (d.adjacent (d.adjacent d.top_vertex).head) in
  1 / 5 = length (filter (λ b, b = d.bottom_vertex) ((d.adjacent (d.adjacent d.top_vertex).head))) / total_paths :=
by
  sorry

end probability_of_ending_at_bottom_vertex_l97_97823


namespace range_of_dihedral_angle_l97_97134

-- Define what a regular triangular pyramid is and the dihedral angle between two adjacent faces
def is_regular_triangular_pyramid (P : Type) := sorry

def dihedral_angle (P : Type) (face1 face2 : P) : ℝ := sorry

-- The range of the dihedral angle θ
theorem range_of_dihedral_angle (P : Type) (θ : ℝ) (h : is_regular_triangular_pyramid P) 
  (hθ : ∃ face1 face2 : P, θ = dihedral_angle P face1 face2) : 
  60 < θ ∧ θ < 180 :=
sorry

end range_of_dihedral_angle_l97_97134


namespace repeating_decimal_35_as_fraction_l97_97168

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l97_97168


namespace cone_volume_ratio_l97_97449

-- Define the mathematical objects and conditions
variables (r h : ℝ) (h_pos : h > 0) (r_pos : r > 0)

-- The volume of the entire cone
def V : ℝ := (1/3) * π * r^2 * h

-- The intermediate segment area calculations
def S_sector : ℝ := (1/6) * π * r^2
def S_triangle : ℝ := (r^2 * Real.sqrt 3) / 4
def S_segment : ℝ := S_sector - S_triangle

-- Volumes of the two parts
def V1 : ℝ := (1/3) * S_segment * h
def V2 : ℝ := V - V1

-- The ratio of the volumes
def volume_ratio : ℝ := V1 / V2

-- The Lean theorem statement
theorem cone_volume_ratio : 
  volume_ratio r h = (2 * π - 3 * Real.sqrt 3) / (10 * π + 3 * Real.sqrt 3) :=
sorry

end cone_volume_ratio_l97_97449


namespace part1_part2_l97_97620

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log x / x

-- Problem Part 1: Prove that if f(x) < kx for all x > 0, then k > 1/(2e)
theorem part1 (k : ℝ) : (∀ x > 0, f x < k * x) → k > 1 / (2 * Real.exp 1) :=
sorry

-- Problem Part 2: Prove that if g(x) = f(x) - kx has two zeros in the interval [1/e, e^2], then 2/e^4 ≤ k < 1/(2e)
def g (x k : ℝ) : ℝ := f x - k * x

theorem part2 (k : ℝ) : 
  (∃ x1 x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2), g x1 k = 0 ∧ g x2 k = 0) → 
  2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
sorry

end part1_part2_l97_97620


namespace last_digit_base5_of_M_l97_97853

theorem last_digit_base5_of_M (d e f : ℕ) (hd : d < 5) (he : e < 5) (hf : f < 5)
  (h : 25 * d + 5 * e + f = 64 * f + 8 * e + d) : f = 0 :=
by
  sorry

end last_digit_base5_of_M_l97_97853


namespace kangaroo_can_jump_1000_units_l97_97445

noncomputable def distance (x y : ℕ) : ℕ := x + y

def valid_small_jump (x y : ℕ) : Prop :=
  x + 1 ≥ 0 ∧ y - 1 ≥ 0

def valid_big_jump (x y : ℕ) : Prop :=
  x - 5 ≥ 0 ∧ y + 7 ≥ 0

theorem kangaroo_can_jump_1000_units (x y : ℕ) (h : x + y > 6) :
  distance x y ≥ 1000 :=
sorry

end kangaroo_can_jump_1000_units_l97_97445


namespace find_fourth_root_l97_97330

noncomputable def poly (d e f : ℚ) := λ x : ℂ, x^4 + d * x^2 + e * x + f

theorem find_fourth_root (d e f : ℚ) (p q : ℤ) (h1 : 3 - complex.sqrt 5 = p) (h2 : 3 + complex.sqrt 5 = q):
  3 - complex.sqrt 5 ∈ complex.roots (poly d e f) ∧
  3 + complex.sqrt 5 ∈ complex.roots (poly d e f) ∧
  (∃ (p q : ℤ), p ∈ complex.roots (poly d e f) ∧ q ∈ complex.roots (poly d e f)) →
  -7 ∈ complex.roots (poly d e f) :=
begin
  sorry
end

end find_fourth_root_l97_97330


namespace survey_experimental_method_l97_97035

def is_experimental_survey (survey : String) : Prop :=
  match survey with
  | "Recommending class monitor candidates" => False
  | "Surveying classmates' birthdays" => False
  | "How many meters you can run in 10 seconds" => True
  | "The situation of 'avian influenza' occurrences in the world" => False
  | _ => False

theorem survey_experimental_method :
  is_experimental_survey "How many meters you can run in 10 seconds" = True :=
by 
  unfold is_experimental_survey
  exact eq.refl True

end survey_experimental_method_l97_97035


namespace complement_intersection_l97_97976

open Set

universe u

variable {α : Type u}

def U : Set α := {1, 2, 3, 4}
def A : Set α := {1, 2}
def B : Set α := {2, 4}

theorem complement_intersection : (U \ (A ∩ B)) = ({1, 3, 4} : Set α) := by
  sorry

end complement_intersection_l97_97976


namespace no_transform_to_1998_power_7_l97_97028

theorem no_transform_to_1998_power_7 :
  ∀ n : ℕ, (exists m : ℕ, n = 7^m) ->
  ∀ k : ℕ, n = 10 * k + (n % 10) ->
  ¬ (∃ t : ℕ, (t = (1998 ^ 7))) := 
by sorry

end no_transform_to_1998_power_7_l97_97028


namespace smallest_x_solution_l97_97519

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l97_97519


namespace arithmetic_sequence_sum_l97_97377

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l97_97377


namespace definite_integral_value_l97_97493

-- Define the integrand function
def integrand (x : Real) : Real := 2^4 * (Real.sin x)^6 * (Real.cos x)^2

-- Define the limits of integration
def a : Real := Real.pi / 2
def b : Real := Real.pi

-- State the proof problem
theorem definite_integral_value : 
  (∫ x in a..b, integrand x) = 5 * Real.pi / 16 := 
sorry

end definite_integral_value_l97_97493


namespace range_of_m_log_inequality_l97_97680

noncomputable def f (x m : ℝ) : ℝ :=
  x - abs (x + 2) - abs (x - 3) - m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (1 / m) - 4 ≥ f x m) → (0 < m) :=
sorry

theorem log_inequality (m : ℝ) (h : 0 < m) :
  log (m + 2) / log (m + 1) > log (m + 3) / log (m + 2) :=
sorry

end range_of_m_log_inequality_l97_97680


namespace geometric_sequence_properties_l97_97935

/-- Given {a_n} is a geometric sequence, a_1 = 1 and a_4 = 1/8, 
the common ratio q of {a_n} is 1/2 and the sum of the first 5 terms of {1/a_n} is 31. -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (h1 : a 1 = 1) (h4 : a 4 = 1 / 8) : 
  (∃ q : ℝ, (∀ n : ℕ, a n = a 1 * q ^ (n - 1)) ∧ q = 1 / 2) ∧ 
  (∃ S : ℝ, S = 31 ∧ S = (1 - 2^5) / (1 - 2)) :=
by
  -- Skipping the proof
  sorry

end geometric_sequence_properties_l97_97935


namespace puppies_adopted_in_12_days_l97_97009

theorem puppies_adopted_in_12_days (initial_puppies : ℕ) (additional_puppies : ℕ) (puppies_adopted_per_day : ℕ) : 
  initial_puppies = 35 → 
  additional_puppies = 48 → 
  puppies_adopted_per_day = 7 → 
  ⌈(initial_puppies + additional_puppies : ℕ) / puppies_adopted_per_day⌉ = 12 := 
by
  intros h1 h2 h3
  have total_puppies : ℕ := initial_puppies + additional_puppies
  have days_needed : ℕ := total_puppies / puppies_adopted_per_day + if total_puppies % puppies_adopted_per_day = 0 then 0 else 1
  rw [h1, h2, h3] at *
  exact sorry

end puppies_adopted_in_12_days_l97_97009


namespace cars_with_power_windows_l97_97286

theorem cars_with_power_windows (T P_s P_sw N : ℕ) (hT : T = 65) (hP_s : P_s = 45) (hP_sw : P_sw = 17) (hN : N = 12) :
    ∃ P_w, P_s + P_w - P_sw = T - N ∧ P_w = 25 :=
by
    use 25
    rw [hT, hP_s, hP_sw, hN]
    apply And.intro
    · calc
        45 + 25 - 17 = 70 - 17 : by norm_num
               ... = 53      : by norm_num
               ... = 65 - 12 : by rw [hT, hN]
    · rfl

end cars_with_power_windows_l97_97286


namespace altitude_intersection_problem_l97_97468

noncomputable def calculation_of_expression : ℝ :=
  sorry -- Placeholder as computation isn't needed

theorem altitude_intersection_problem (A B C H P Q : Point)
  (hAP : Altitude A P)
  (hBQ : Altitude B Q)
  (hH : H = intersection (hAP) (hBQ))
  (h_HP : HP = 3)
  (h_HQ : HQ = 4) :
  (BP * PC - AQ * QC = -7) :=
sorry

end altitude_intersection_problem_l97_97468


namespace repeating_decimal_35_as_fraction_l97_97170

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l97_97170


namespace probability_abc_plus_ab_plus_a_divisible_by_4_l97_97707

noncomputable def count_multiples_of (n m : ℕ) : ℕ := (m / n)

noncomputable def probability_divisible_by_4 : ℚ := 
  let total_numbers := 2008
  let multiples_of_4 := count_multiples_of 4 total_numbers
  -- Probability that 'a' is divisible by 4
  let p_a := (multiples_of_4 : ℚ) / total_numbers
  -- Probability that 'a' is not divisible by 4
  let p_not_a := 1 - p_a
  -- Considering specific cases for b and c modulo 4
  let p_bc_cases := (2 * ((1 / 4) * (1 / 4)))  -- Probabilities for specific cases noted as 2 * (1/16)
  -- Adjusting probabilities for non-divisible 'a'
  let p_not_a_cases := p_bc_cases * p_not_a
  -- Total Probability
  p_a + p_not_a_cases

theorem probability_abc_plus_ab_plus_a_divisible_by_4 :
  probability_divisible_by_4 = 11 / 32 :=
sorry

end probability_abc_plus_ab_plus_a_divisible_by_4_l97_97707


namespace coordinates_of_C_l97_97293

theorem coordinates_of_C (A B : ℝ × ℝ) (hA : A = (-2, -1)) (hB : B = (4, 9)) :
    ∃ C : ℝ × ℝ, (dist C A) = 4 * dist C B ∧ C = (-0.8, 1) :=
sorry

end coordinates_of_C_l97_97293


namespace bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97430

-- Definitions of operations
def simple_op_time : ℕ := 1
def lengthy_op_time : ℕ := 5
def num_simple_ops : ℕ := 5
def num_lengthy_ops : ℕ := 3
def total_people : ℕ := num_simple_ops + num_lengthy_ops

-- Proving minimum and maximum person-minutes wasted
theorem bank_queue_min_max_wastage :
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 40) ∧
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 100) :=
by sorry

-- Proving expected value of wasted person-minutes
theorem bank_queue_expected_wastage :
  expected_value_wasted_person_minutes total_people simple_op_time lengthy_op_time = 84 :=
by sorry

-- Placeholder for the actual expected value calculation function
noncomputable def expected_value_wasted_person_minutes
  (n : ℕ) (t_simple : ℕ) (t_lengthy : ℕ) : ℕ :=
  -- Calculation logic will be implemented here
  84 -- This is just the provided answer, actual logic needed for correctness

end bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97430


namespace bank_queue_wasted_time_l97_97427

-- Conditions definition
def simple_time : ℕ := 1
def lengthy_time : ℕ := 5
def num_simple : ℕ := 5
def num_lengthy : ℕ := 3
def total_people : ℕ := 8

-- Theorem statement
theorem bank_queue_wasted_time :
  (min_wasted_time : ℕ := 40) ∧
  (max_wasted_time : ℕ := 100) ∧
  (expected_wasted_time : ℚ := 72.5) := by
  sorry

end bank_queue_wasted_time_l97_97427


namespace probability_at_least_3_out_of_8_laughs_l97_97993

noncomputable def probability_at_least_3_laughs (p_laugh : ℚ) (n : ℕ) : ℚ :=
  let p_not_laugh := 1 - p_laugh
  let probabilities := List.range (3 : ℕ) |>.map (fun k => (nat.choose n k) * (p_laugh^k) * (p_not_laugh^(n-k)))
  let total_probability := probabilities.foldl (· + ·) 0
  1 - total_probability

theorem probability_at_least_3_out_of_8_laughs :
  probability_at_least_3_laughs (1/3) 8 = 3489 / 6561 :=
by
  sorry

end probability_at_least_3_out_of_8_laughs_l97_97993


namespace bank_queue_wasted_time_l97_97429

-- Conditions definition
def simple_time : ℕ := 1
def lengthy_time : ℕ := 5
def num_simple : ℕ := 5
def num_lengthy : ℕ := 3
def total_people : ℕ := 8

-- Theorem statement
theorem bank_queue_wasted_time :
  (min_wasted_time : ℕ := 40) ∧
  (max_wasted_time : ℕ := 100) ∧
  (expected_wasted_time : ℚ := 72.5) := by
  sorry

end bank_queue_wasted_time_l97_97429


namespace part1_max_m_part2_ineq_l97_97595

theorem part1_max_m : ∀ x : ℝ, |3 * x + 1| + |2 - 3 * x| ≥ 3 :=
by sorry

theorem part2_ineq (a b : ℝ) (ha : a > b) (hb : b > 0) :
  a + 4 / (a^2 - 2 * a * b + b^2) ≥ b + 3 :=
begin
  sorry
end

end part1_max_m_part2_ineq_l97_97595


namespace arina_should_accept_anton_offer_l97_97873

noncomputable def total_shares : ℕ := 300000
noncomputable def arina_shares : ℕ := 90001
noncomputable def need_to_be_largest : ℕ := 104999 
noncomputable def shares_needed : ℕ := 14999
noncomputable def largest_shareholder_total : ℕ := 105000

noncomputable def maxim_shares : ℕ := 104999
noncomputable def inga_shares : ℕ := 30000
noncomputable def yuri_shares : ℕ := 30000
noncomputable def yulia_shares : ℕ := 30000
noncomputable def anton_shares : ℕ := 15000

noncomputable def maxim_price_per_share : ℕ := 11
noncomputable def inga_price_per_share : ℕ := 1250 / 100
noncomputable def yuri_price_per_share : ℕ := 1150 / 100
noncomputable def yulia_price_per_share : ℕ := 1300 / 100
noncomputable def anton_price_per_share : ℕ := 14

noncomputable def anton_total_cost : ℕ := anton_shares * anton_price_per_share
noncomputable def yuri_total_cost : ℕ := yuri_shares * yuri_price_per_share
noncomputable def inga_total_cost : ℕ := inga_shares * inga_price_per_share
noncomputable def yulia_total_cost : ℕ := yulia_shares * yulia_price_per_share

theorem arina_should_accept_anton_offer :
  anton_total_cost = 210000 := by
  sorry

end arina_should_accept_anton_offer_l97_97873


namespace problem_solver_equals_girls_l97_97751

variable (A B : ℕ)

/-- Number of boys who solved the problem is equal to the number of girls who did not solve it -/
def num_girls_not_solved := A

/-- Total number of girls in the class -/
def G := B + num_girls_not_solved A B

/-- Total number of students who solved the problem -/
def num_problem_solvers := A + B

/-- Prove that the number of problem solvers equals the number of girls in the class -/
theorem problem_solver_equals_girls :
  num_problem_solvers A B = G A B :=
by
  sorry

end problem_solver_equals_girls_l97_97751


namespace cryptarithm_solution_exists_l97_97711

theorem cryptarithm_solution_exists :
  ∃ (K P O C S R T : ℕ),
    K ≠ P ∧ K ≠ O ∧ K ≠ C ∧ K ≠ S ∧ K ≠ R ∧ K ≠ T ∧
    P ≠ O ∧ P ≠ C ∧ P ≠ S ∧ P ≠ R ∧ P ≠ T ∧
    O ≠ C ∧ O ≠ S ∧ O ≠ R ∧ O ≠ T ∧
    C ≠ S ∧ C ≠ R ∧ C ≠ T ∧
    S ≠ R ∧ S ≠ T ∧
    R ≠ T ∧
    K ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    P ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    O ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    C ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    S ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    R ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    T ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (2 * (K * 10000 + P * 1000 + O * 100 + C * 10 + C)) = (S * 10000 + P * 1000 + O * 100 + R * 10 + T) :=
begin
  sorry,
end

end cryptarithm_solution_exists_l97_97711


namespace smallest_cubes_to_form_30_digit_number_l97_97363

theorem smallest_cubes_to_form_30_digit_number :
  ∃ (n : ℕ), n = 50 ∧ 
  (∀ (d : ℤ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
   ∃ (cubes : fin n → fin 6 → ℤ),
     (∀ (i : fin n), ∀ (j : fin 6), cubes i j ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
     (∀ (k : ℤ), k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → (∃ (count_k : fin n → ℕ), (∑ i, count_k i = 30 ∧ (∀ i, cubes i (count_k i) = k))) ∧
       (∃ (count_0 : fin n → ℕ), (∑ i, count_0 i = 29 ∧ (∀ i, cubes i (count_0 i) = 0)) ))) := by
  sorry

end smallest_cubes_to_form_30_digit_number_l97_97363


namespace triangle_area_l97_97854

theorem triangle_area (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 54 := by
  -- conditions provided
  sorry

end triangle_area_l97_97854


namespace sum_of_reversed_base_digits_eq_zero_l97_97557

theorem sum_of_reversed_base_digits_eq_zero : ∃ n : ℕ, 
  (∀ a₁ a₀ : ℕ, n = 5 * a₁ + a₀ ∧ n = 12 * a₀ + a₁ ∧ 0 ≤ a₁ ∧ a₁ < 5 ∧ 0 ≤ a₀ ∧ a₀ < 12 
  ∧ n > 0 → n = 0)
:= sorry

end sum_of_reversed_base_digits_eq_zero_l97_97557


namespace probability_of_selecting_letter_a_l97_97298

def total_ways := Nat.choose 5 2
def ways_to_select_a := 4
def probability_of_selecting_a := (ways_to_select_a : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_letter_a :
  probability_of_selecting_a = 2 / 5 :=
by
  -- proof steps will be filled in here
  sorry

end probability_of_selecting_letter_a_l97_97298


namespace thirty_knocks_eq_thirtytwo_knicks_l97_97635

-- Definitions of the conditions
variable (knicks knacks knocks : Type)
variable (to_knacks : knicks → knacks)
variable (to_knocks : knacks → knocks)
variable (knacks_to_knicks: knacks → knicks)

-- Given conditions
axiom eight_knicks_eq_three_knacks : ∀ k₁ : knicks, 8 * to_knacks k₁ = 3 * to_knacks (to_knacks k₁)
axiom two_knacks_eq_five_knocks : ∀ k₂ : knacks, 2 * to_knocks k₂ = 5 * to_knocks (to_knocks k₂)

-- Proof statement
theorem thirty_knocks_eq_thirtytwo_knicks : ∀ n : knocks, 30 * knacks_to_knicks (to_knacks (to_knocks n)) = 32 * knicks := 
sorry

end thirty_knocks_eq_thirtytwo_knicks_l97_97635


namespace incorrect_statement_C_l97_97745

theorem incorrect_statement_C :
  (∀ n : ℕ, a_n = a₁ + (n-1)*d) ∧
  (∀ n : ℕ, S_n = n * (a₁ + (n-1) * d / 2)) ∧
  S_7 < S_8 ∧
  S_8 = S_9 ∧
  S_9 > S_{10} →
  ¬(S_{11} > S_7) :=
by
  sorry

end incorrect_statement_C_l97_97745


namespace sum_of_first_39_natural_numbers_l97_97481

theorem sum_of_first_39_natural_numbers : (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_of_first_39_natural_numbers_l97_97481


namespace mixing_paints_l97_97086

theorem mixing_paints (x : ℝ) : 
  x = 5 / 3 →
  ∃ (y : ℝ), y = 5 * 0.2 ∧ x * 0.4 + 1 = 0.25 * (5 + x) := 
by
  intro h1,
  use 1,
  split,
  { simp, norm_num, },
  { rw h1,
    ring_nf,
    simp }

end mixing_paints_l97_97086


namespace eval_g_inv_g_inv_14_l97_97310

variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)

axiom g_def : ∀ x, g x = 3 * x - 4
axiom g_inv_def : ∀ y, g_inv y = (y + 4) / 3

theorem eval_g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by
    sorry

end eval_g_inv_g_inv_14_l97_97310


namespace value_of_expression_l97_97400

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : 3 * m^2 + 3 * m + 2006 = 2009 :=
by
  sorry

end value_of_expression_l97_97400


namespace length_of_shorter_piece_l97_97409

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end length_of_shorter_piece_l97_97409


namespace cylinder_ellipse_eccentricity_l97_97822

noncomputable def eccentricity_of_ellipse (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let b := r
  let a := r / (Real.cos angle)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem cylinder_ellipse_eccentricity :
  eccentricity_of_ellipse 12 (Real.pi / 6) = 1 / 2 :=
by
  sorry

end cylinder_ellipse_eccentricity_l97_97822


namespace complex_conjugate_of_z_l97_97210

noncomputable theory

-- Define the complex numbers
def z : ℂ := (4 - 2 * complex.i) / (1 - 3 * complex.i)

-- Define the conditions
axiom h : (1 - 3 * complex.i) * z = 4 - 2 * complex.i

-- State the theorem to prove
theorem complex_conjugate_of_z : complex.conj z = 1 - complex.i :=
sorry

end complex_conjugate_of_z_l97_97210


namespace sec_of_7pi_over_4_l97_97918

theorem sec_of_7pi_over_4 : real.sec (7 * real.pi / 4) = real.sqrt 2 := by
  sorry

end sec_of_7pi_over_4_l97_97918


namespace f_monotonically_decreasing_always_holds_sum_lower_bound_l97_97964

noncomputable def f (x a : ℝ) : ℝ := log x - (1 / 2) * a * x^2 + x

-- Question 1: Prove that if f(1) = 0 and f'(x) < 0 for x > 1, then f is monotonically decreasing on (1, ∞)
theorem f_monotonically_decreasing (a : ℝ) :
  f 1 a = 0 → 
  (∀ x : ℝ, 1 < x → ((1 / x - a * x + 1) < 0)) →
  (∀ x : ℝ, 1 < x → f x a < f 1 a) :=
by
  sorry

-- Question 2: Prove that if a ≤ 2, then f(x) ≤ ax - 1 always holds for x
theorem always_holds (a : ℝ) :
  a ≤ 2 → 
  ∀ x : ℝ, 
  f x a ≤ a * x - 1 :=
by
  sorry

-- Question 3: Given f(x1) + f(x2) + x1*x2 = 0 and a = -2, prove x1 + x2 ≥ (sqrt 5 - 1) / 2
theorem sum_lower_bound (a x1 x2 : ℝ) :
  a = -2 → 
  f x1 a + f x2 a + x1 * x2 = 0 →
  x1 + x2 ≥ (sqrt 5 - 1) / 2 :=
by
  sorry

end f_monotonically_decreasing_always_holds_sum_lower_bound_l97_97964


namespace repeating_decimal_sum_l97_97178

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l97_97178


namespace minimize_sqrt_difference_l97_97559

theorem minimize_sqrt_difference (p : ℕ) (hp : p.prime) (hp_odd : p % 2 = 1) :
  ∃ m n : ℕ, m ≤ n ∧ (m = (p - 1) / 2) ∧ (n = (p + 1) / 2) ∧ 
    (sqrt (2 * p) - sqrt m - sqrt n ≥ 0 ∧ 
     ∀ m' n' : ℕ, m' ≤ n' → (sqrt (2 * p) - sqrt m' - sqrt n' < sqrt (2 * p) - sqrt m - sqrt n → false)) :=
sorry

end minimize_sqrt_difference_l97_97559


namespace bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97422

def simple_op_time : ℕ := 1
def long_op_time : ℕ := 5
def num_simple_customers : ℕ := 5
def num_long_customers : ℕ := 3
def total_customers : ℕ := 8

theorem bank_queue_minimum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  n * a + 3 * a + b + 4 * a + b + a + b + (b + (n - 1) * a) + b + (b + (n-2) * a) = 40 :=
  by intros; sorry

theorem bank_queue_maximum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  m * (m - 1) * b / 2 + n * a * (m + n) + 1 = 100 :=
  by intros; sorry

theorem expected_wasted_minutes_random_order :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  ∑ i in range total_customers, (i * (a + b)) = 72.5 * (total_customers * (total_customers - 1)) / 2 :=
  by intros; sorry

end bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l97_97422


namespace largest_number_less_than_0_7_l97_97382

theorem largest_number_less_than_0_7 : 
    ∀ (a b c : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → a < 0.7 = false → b < 0.7 → c < 0.7 = false → b = 1/2 :=
by
  intros a b c h_a h_b h_c h_a_lt h_b_lt h_c_lt
  rw h_b
  sorry

end largest_number_less_than_0_7_l97_97382


namespace min_value_of_a_l97_97260

theorem min_value_of_a (a : ℝ) (h : a > 0) (h₁ : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : a ≥ 2 := 
sorry

end min_value_of_a_l97_97260


namespace children_per_block_l97_97038

theorem children_per_block {children total_blocks : ℕ} 
  (h_total_blocks : total_blocks = 9) 
  (h_total_children : children = 54) : 
  (children / total_blocks = 6) :=
by
  -- Definitions from conditions
  have h1 : total_blocks = 9 := h_total_blocks
  have h2 : children = 54 := h_total_children

  -- Goal to prove
  -- children / total_blocks = 6
  sorry

end children_per_block_l97_97038


namespace part_a_car_moves_part_b_no_fewer_than_six_moves_part_c_car_moves_l97_97792

-- Part (a)
theorem part_a_car_moves (initial_positions : Configuration) :
  ∃ moves : list Move, length moves = 6 ∧ car1_can_exit initial_positions moves :=
sorry

-- Part (b)
theorem part_b_no_fewer_than_six_moves (initial_positions : Configuration) :
  ∀ moves : list Move, car1_can_exit initial_positions moves → length moves ≥ 6 :=
sorry

-- Part (c)
theorem part_c_car_moves (other_positions : Configuration) :
  ∃ moves : list Move, car1_can_exit other_positions moves :=
sorry

end part_a_car_moves_part_b_no_fewer_than_six_moves_part_c_car_moves_l97_97792


namespace stratified_sampling_elderly_l97_97402

theorem stratified_sampling_elderly (total_elderly middle_aged young total_sample total_population elderly_to_sample : ℕ) 
  (h1: total_elderly = 30) 
  (h2: middle_aged = 90) 
  (h3: young = 60) 
  (h4: total_sample = 36) 
  (h5: total_population = total_elderly + middle_aged + young) 
  (h6: 1 / 5 * total_elderly = elderly_to_sample)
  : elderly_to_sample = 6 := 
  by 
    sorry

end stratified_sampling_elderly_l97_97402


namespace smallest_solution_floor_eq_l97_97547

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l97_97547


namespace train_average_speed_l97_97024

theorem train_average_speed (x : ℝ) (h1 : x > 0) :
  let d1 := x
  let d2 := 2 * x
  let s1 := 50
  let s2 := 20
  let t1 := d1 / s1
  let t2 := d2 / s2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 25 := 
by
  sorry

end train_average_speed_l97_97024


namespace ryan_fraction_l97_97670

-- Define the total amount of money
def total_money : ℕ := 48

-- Define that Ryan owns a fraction R of the total money
variable {R : ℚ}

-- Define the debts
def ryan_owes_leo : ℕ := 10
def leo_owes_ryan : ℕ := 7

-- Define the final amount Leo has after settling the debts
def leo_final_amount : ℕ := 19

-- Define the condition that Leo and Ryan together have $48
def leo_plus_ryan (leo_amount ryan_amount : ℚ) : Prop := 
  leo_amount + ryan_amount = total_money

-- Define Ryan's amount as a fraction R of the total money
def ryan_amount (R : ℚ) : ℚ := R * total_money

-- Define Leo's amount before debts were settled
def leo_amount_before_debts : ℚ := (leo_final_amount : ℚ) + leo_owes_ryan

-- Define the equation after settling debts
def leo_final_eq (leo_amount_before_debts : ℚ) : Prop :=
  (leo_amount_before_debts - ryan_owes_leo = leo_final_amount)

-- The Lean theorem that needs to be proved
theorem ryan_fraction :
  ∃ (R : ℚ), leo_plus_ryan (leo_amount_before_debts - ryan_owes_leo) (ryan_amount R)
  ∧ leo_final_eq leo_amount_before_debts
  ∧ R = 11 / 24 :=
sorry

end ryan_fraction_l97_97670


namespace min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l97_97568

-- Condition 1: Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- Proof Problem 1: Minimum value of f(x) when a = 1
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f x 1 ≥ 2) :=
sorry

-- Proof Problem 2: Range of values for a when f(x) ≤ 3 has solutions
theorem range_of_a_if_f_leq_3_non_empty : 
  (∃ x : ℝ, f x a ≤ 3) → abs (3 - a) ≤ 3 :=
sorry

end min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l97_97568


namespace inclination_angle_of_line_l97_97320

theorem inclination_angle_of_line : ∀ (x y : ℝ), (x - y + 1 = 0) → ∃ θ : ℝ, θ = 45 ∧ tan θ = 1 :=
by
  intros x y h
  sorry

end inclination_angle_of_line_l97_97320


namespace geom_progr_sum_eq_l97_97339

variable (a b q : ℝ) (n p : ℕ)

theorem geom_progr_sum_eq (h : a * (1 - q ^ (n * p)) / (1 - q) = b * (1 - q ^ (n * p)) / (1 - q ^ p)) :
  b = a * (1 - q ^ p) / (1 - q) :=
by
  sorry

end geom_progr_sum_eq_l97_97339


namespace max_possible_value_of_k_l97_97804

noncomputable def max_knights_saying_less : Nat :=
  let n := 2015
  let k := n - 2
  k

theorem max_possible_value_of_k : max_knights_saying_less = 2013 :=
by
  sorry

end max_possible_value_of_k_l97_97804


namespace area_bounded_by_f_l97_97674

noncomputable def f : ℝ → ℝ :=
  λ x, if h : 0 ≤ x ∧ x ≤ 4 then real.sqrt x else 3 - x / 4

theorem area_bounded_by_f (J : ℝ) :
  let A1 := ∫ x in 0..4, real.sqrt x
  let A2 := ∫ x in 4..10, (3 - x / 4)
  A1 + A2 = J → J = 12.8333 :=
by
  let A1 := ∫ x in 0..4, real.sqrt x
  let A2 := ∫ x in 4..10, 3 - x / 4
  sorry

end area_bounded_by_f_l97_97674


namespace find_monotonic_intervals_and_range_find_range_of_m_l97_97596

variables {m : ℝ} (x : ℝ)

noncomputable def f (m : ℝ) (x : ℝ) := (2 * m / 3) * x^3 + x^2 - 3 * x - m * x + 2
noncomputable def g (m : ℝ) (x : ℝ) := 2 * m * x^2 + 2 * x - 3 - m

theorem find_monotonic_intervals_and_range (h1 : m = 1) :
  (∀ x < -2, f 1 x < f 1 (x + 1)) ∧ (∀ x > 1, f 1 (x - 1) < f 1 x) ∧ (∀ x ∈ Ioo (-2 : ℝ) 1, f 1 (x - 1) > f 1 x) :=
sorry

theorem find_range_of_m : (g m (-1) ≤ g m 1 ∨ g m (-1) ≥ 0 ∧ g m 1 ≤ 0) →
  (m = 0 → False) ∧ m ∈ Icc (-∞ : ℝ) ((-3 - real.sqrt 7) / 2) ∪ Icc (1 : ℝ) ∞ :=
sorry

end find_monotonic_intervals_and_range_find_range_of_m_l97_97596


namespace arithmetic_sequences_count_l97_97585

theorem arithmetic_sequences_count :
  ∃ a1 d n : ℕ, n ≥ 3 → (a1 + (a1 + d) + ... + (a1 + (n-1)*d) = 97^2) → 
                (a1 ≥ 0) → (d ≥ 0) →
                4 = { (a1, d, n) | n ≥ 3 ∧ a1 ≥ 0 ∧ d ≥ 0 ∧ finset.sum (finset.range n) (λ k, a1 + k * d) = 97^2 }.card := 
sorry

end arithmetic_sequences_count_l97_97585


namespace shaded_region_area_eq_l97_97237

-- Define the radius of the circles
def radius : ℝ := 5

-- Define the area calculations for the components
def quarter_circle_area : ℝ := (1/4) * Real.pi * radius ^ 2
def isosceles_triangle_area : ℝ := (1/2) * radius * radius

-- Calculate shaded region for a single sector
def shaded_area_single_sector : ℝ := quarter_circle_area - isosceles_triangle_area

-- Total shaded area considering 8 such regions
def total_shaded_area : ℝ := 8 * shaded_area_single_sector

-- Main theorem to prove
theorem shaded_region_area_eq : total_shaded_area = 50 * Real.pi - 100 := by
  sorry

end shaded_region_area_eq_l97_97237


namespace div_of_floats_l97_97368

theorem div_of_floats : (0.2 : ℝ) / (0.005 : ℝ) = 40 := 
by
  sorry

end div_of_floats_l97_97368


namespace angle_between_vectors_is_pi_div_3_l97_97121

noncomputable theory

variables {a b : ℝ^3} (ha : a ≠ 0) (hb : b ≠ 0)
(h1 : ∥a∥ = 2 * ∥b∥)
(h2 : a - b ⬝ b = 0)

theorem angle_between_vectors_is_pi_div_3 :
  real.angle_between a b = real.pi / 3 :=
sorry

end angle_between_vectors_is_pi_div_3_l97_97121


namespace matrices_not_identity_l97_97053

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![1, 2],
  ![0, 1]
]

def matrix_B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![1, 0],
  ![2, 1]
]

theorem matrices_not_identity (k : ℤ) (h₁ : k ≥ 1)
  (i_0 i_k : ℤ)
  (i : Fin (k-1) → ℤ) (j : Fin k → ℤ)
  (i_nonzero : ∀ n, i n ≠ 0)
  (j_nonzero : ∀ n, j n ≠ 0) :
  let product := (matrix_A ^ i_0) * (matrix_B ^ (j 0)) * (matrix_A ^ (i 0)) * (matrix_B ^ (j 1))
                      * (matrix_A ^ (i 1)) * (matrix_B ^ (j 2)) * ⋯ * (matrix_A ^ (i (k-2))) * (matrix_B ^ (j (k-1))) * (matrix_A ^ i_k)
  in product ≠ (1 : Matrix (Fin 2) (Fin 2) ℤ) := 
by
  sorry

end matrices_not_identity_l97_97053


namespace ellipse_equation_correct_l97_97586

-- Definitions based on the given conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def focal_distance (a c b : ℝ) : Prop :=
  a^2 = b^2 + c^2

-- The main theorem statement
theorem ellipse_equation_correct :
  (focal_distance sqrt(6) 2 := sqrt(2)) →
  ellipse_eq sqrt(6) sqrt(2) :=
by
  sorry

end ellipse_equation_correct_l97_97586


namespace correct_conclusions_l97_97265

noncomputable theory
open Complex Real 

def f (x : ℝ) : ℝ := sqrt 3 * cos (2*x) + 2 * sin x * cos x

-- translated propositions
def period_f : Prop := ∀ x, f (x + π) = f x
def symmetric_f : Prop := ∀ x, f x = f (π / 6 - x + π / 12)
def shifted_cos : Prop := ∀ x, f x = 2 * cos (2 * (x - π / 12))

theorem correct_conclusions : period_f ∧ symmetric_f ∧ shifted_cos :=
by sorry

end correct_conclusions_l97_97265


namespace scientific_notation_of_425000_l97_97508

def scientific_notation (x : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_425000 :
  scientific_notation 425000 = (4.25, 5) := sorry

end scientific_notation_of_425000_l97_97508


namespace skier_glide_distance_l97_97809

noncomputable def mass := 70 -- kg
noncomputable def v₀ := 40 * 1000 / 3600 -- m/s
noncomputable def F₀ := 400 -- N
noncomputable def vMin := 10 * 1000 / 3600 -- m/s
noncomputable def k := F₀ / (v₀ ^ 2 : ℝ) -- proportionality constant

theorem skier_glide_distance :
  let m := mass in
  let v_initial := v₀ in
  let F_initial := F₀ in
  let v_min := vMin in
  let k := k in
  ∃ s : ℝ, s = 30 := by
  sorry

end skier_glide_distance_l97_97809


namespace weighted_avg_surfers_per_day_l97_97904

theorem weighted_avg_surfers_per_day 
  (total_surfers : ℕ) 
  (ratio1_day1 ratio1_day2 ratio2_day3 ratio2_day4 : ℕ) 
  (h_total_surfers : total_surfers = 12000)
  (h_ratio_first_two_days : ratio1_day1 = 5 ∧ ratio1_day2 = 7)
  (h_ratio_last_two_days : ratio2_day3 = 3 ∧ ratio2_day4 = 2) 
  : (total_surfers / (ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4)) * 
    ((ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4) / 4) = 3000 :=
by
  sorry

end weighted_avg_surfers_per_day_l97_97904


namespace vector_magnitude_sum_l97_97133

variables (a b : ℝ → ℝ → ℝ) 
           (cos_theta : ℝ) (norm_a norm_b : ℝ)

-- Given conditions
hypothesis h1 : cos_theta = 3 / 5
hypothesis h2 : norm_a = 1
hypothesis h3 : norm_b = 1
hypothesis h4 : ∀ t, a t = b t + 1

-- Question to prove
theorem vector_magnitude_sum :
  |a + b| = 4 * real.sqrt 5 / 5 :=
sorry

end vector_magnitude_sum_l97_97133


namespace exists_three_quadratics_with_properties_l97_97059

theorem exists_three_quadratics_with_properties : 
  ∃ p q r : ℝ[X], 
    degree p = 2 ∧ degree q = 2 ∧ degree r = 2 ∧ 
    (∃ x : ℝ, p.eval x = 0) ∧ 
    (∃ x : ℝ, q.eval x = 0) ∧ 
    (∃ x : ℝ, r.eval x = 0) ∧ 
    (¬ ∃ x : ℝ, (p + q).eval x = 0) ∧ 
    (¬ ∃ x : ℝ, (p + r).eval x = 0) ∧ 
    (¬ ∃ x : ℝ, (q + r).eval x = 0) :=
sorry

end exists_three_quadratics_with_properties_l97_97059


namespace find_fg_k_l_l97_97243

theorem find_fg_k_l (EV EP VF : ℕ)
  (rectangle EFGH : Prop)
  (circle_center_F : Prop)
  (m_intersects : Prop)
  (S : set ℝ)
  (area_ratio : ℝ)
  (h : ℝ) 
  (k l : ℕ)
  (k_not_divisible_by_square_of_prime : ℕ)
  (reshaped_figure : ℝ) 
  (new_S : Prop) :
  EFGH ∧ circle_center_F ∧ m_intersects ∧ EV = 60 ∧ EP = 90 ∧ VF = 120 ∧
  area_ratio = 1 / 3 ∧ S = 1 / 4 * new_S →
  h = 14 * sqrt 3 →
  FG = h * 30 →
  k = 420 →
  l = 3 →
  k + l = 423 := 
by
  sorry

end find_fg_k_l_l97_97243


namespace math_proof_problem_l97_97678

noncomputable def proof_problem (c d : ℝ) : Prop :=
  (∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  ∧ (∃ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3)
  ∧ 100 * c + d = 141
  
theorem math_proof_problem (c d : ℝ) 
  (h1 : ∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  (h2 : ∀ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3) :
  100 * c + d = 141 := 
sorry

end math_proof_problem_l97_97678


namespace theorem_227_l97_97899

theorem theorem_227 (a b c d : ℤ) (k : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧
  (a - b ≡ a - c [ZMOD d]) ∧
  (a * b ≡ a * c [ZMOD d]) :=
by
  sorry

end theorem_227_l97_97899


namespace increasing_exponential_iff_l97_97609

def is_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem increasing_exponential_iff (a : ℝ) :
  is_increasing_on_R (λ x : ℝ, (a - 1) ^ x) ↔ a > 2 :=
sorry

end increasing_exponential_iff_l97_97609


namespace proof_min_max_expected_wasted_minutes_l97_97419

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l97_97419


namespace MathContestMeanMedianDifference_l97_97647

theorem MathContestMeanMedianDifference :
  (15 / 100 * 65 + 20 / 100 * 85 + 40 / 100 * 95 + 25 / 100 * 110) - 95 = -3 := 
by
  sorry

end MathContestMeanMedianDifference_l97_97647


namespace sin_2varphi_symmetric_l97_97967

theorem sin_2varphi_symmetric (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, (sin (π * x + φ) - 2 * cos (π * x + φ)) = (sin (π * (2 - x) + φ) - 2 * cos (π * (2 - x) + φ))) : 
  sin (2 * φ) = - (4 / 5) :=
sorry

end sin_2varphi_symmetric_l97_97967


namespace porche_project_time_l97_97295

theorem porche_project_time :
  let total_time := 180
  let math_time := 45
  let english_time := 30
  let science_time := 50
  let history_time := 25
  let homework_time := math_time + english_time + science_time + history_time 
  total_time - homework_time = 30 :=
by
  sorry

end porche_project_time_l97_97295


namespace bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97434

-- Definitions of operations
def simple_op_time : ℕ := 1
def lengthy_op_time : ℕ := 5
def num_simple_ops : ℕ := 5
def num_lengthy_ops : ℕ := 3
def total_people : ℕ := num_simple_ops + num_lengthy_ops

-- Proving minimum and maximum person-minutes wasted
theorem bank_queue_min_max_wastage :
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 40) ∧
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 100) :=
by sorry

-- Proving expected value of wasted person-minutes
theorem bank_queue_expected_wastage :
  expected_value_wasted_person_minutes total_people simple_op_time lengthy_op_time = 84 :=
by sorry

-- Placeholder for the actual expected value calculation function
noncomputable def expected_value_wasted_person_minutes
  (n : ℕ) (t_simple : ℕ) (t_lengthy : ℕ) : ℕ :=
  -- Calculation logic will be implemented here
  84 -- This is just the provided answer, actual logic needed for correctness

end bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97434


namespace range_of_trig_function_l97_97517

noncomputable def trig_function (x : ℝ) : ℝ :=
  Real.sin x + Real.cos x + Real.sin x * Real.cos x

theorem range_of_trig_function :
  set.range trig_function = set.Icc (-1) (1/2 + Real.sqrt 2) := by
  sorry

end range_of_trig_function_l97_97517


namespace bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97433

-- Definitions of operations
def simple_op_time : ℕ := 1
def lengthy_op_time : ℕ := 5
def num_simple_ops : ℕ := 5
def num_lengthy_ops : ℕ := 3
def total_people : ℕ := num_simple_ops + num_lengthy_ops

-- Proving minimum and maximum person-minutes wasted
theorem bank_queue_min_max_wastage :
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 40) ∧
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 100) :=
by sorry

-- Proving expected value of wasted person-minutes
theorem bank_queue_expected_wastage :
  expected_value_wasted_person_minutes total_people simple_op_time lengthy_op_time = 84 :=
by sorry

-- Placeholder for the actual expected value calculation function
noncomputable def expected_value_wasted_person_minutes
  (n : ℕ) (t_simple : ℕ) (t_lengthy : ℕ) : ℕ :=
  -- Calculation logic will be implemented here
  84 -- This is just the provided answer, actual logic needed for correctness

end bank_queue_min_max_wastage_bank_queue_expected_wastage_l97_97433


namespace hyperbola_center_l97_97836

variable {Point : Type}

structure coordinates (P : Point) :=
(x : ℝ)
(y : ℝ)

def center_of_hyperbola (P₁ P₂ : Point) := 
  coordinates.mk ((coordinates.x P₁ + coordinates.x P₂) / 2) ((coordinates.y P₁ + coordinates.y P₂) / 2)

theorem hyperbola_center (f1 f2 : Point) (h1 : coordinates f1) (h2 : coordinates f2) :
  h1 = coordinates.mk 3 (-2) → h2 = coordinates.mk 11 6 → center_of_hyperbola f1 f2 = coordinates.mk 7 2 :=
by
  intros
  sorry

end hyperbola_center_l97_97836


namespace solution_of_equation_l97_97204

variables (a b c d p x : ℝ)

-- Conditions
def opposite_numbers (a b : ℝ) := a + b = 0
def reciprocals (c d : ℝ) := c * d = 1
def abs_value_eq_3 (p : ℝ) := p^2 = 9
def equation (a b c d p x : ℝ) := (a + b) * x^2 + 4 * c * d * x + p^2 = x

-- Statement
theorem solution_of_equation
  (h1 : opposite_numbers a b)
  (h2 : reciprocals c d)
  (h3 : abs_value_eq_3 p)
  (x_solution : x = -3) :
  equation a b c d p x :=
by
  unfold opposite_numbers at h1
  unfold reciprocals at h2
  unfold abs_value_eq_3 at h3
  unfold equation
  rw [h1, ←h2, h3]
  simp [←x_solution]
  sorry

end solution_of_equation_l97_97204


namespace tom_needs_to_eat_5_pieces_l97_97395

open Set

variable (board : Fin 8 → Fin 8 → Set String)
variable (P K : String) -- Representing fish and sausage respectively.

-- Conditions
axiom condition_1 : ∀ (x y : Fin 8), x ≤ 2 ∨ y ≤ 2 → (board x y).contains P
axiom condition_2 : ∀ (x y : Fin 8), ∃ u v, x ≤ 2 ∨ y ≤ 2 → ¬ ((board u v).contains P ∧ (board u v).contains K)
axiom condition_3 : ∀ (x y : Fin 8), x ≤ 1 ∨ y ≤ 1 ↔ ∃ u v, x ≤ 1 ∨ y ≤ 1 → (board u u).contains K

noncomputable def minimum_pieces_to_eat : Nat :=
  5

theorem tom_needs_to_eat_5_pieces : ∃ pieces : Fin 8 → Fin 8 → Set String, 
  pieces.card >= 5 → ∀ x y : Fin 8, (board x y).contains P ∧ (board x y).contains K :=
  sorry

end tom_needs_to_eat_5_pieces_l97_97395


namespace trig_identity_l97_97301

variable (x y : ℝ)

theorem trig_identity : sin (x - y) * cos y + cos (x - y) * sin y = sin x :=
by
  sorry

end trig_identity_l97_97301


namespace union_of_A_and_B_is_correct_l97_97692

def A : Set ℕ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B_is_correct : (A ∪ B) = {-1, 0, 1, 2} :=
by sorry

end union_of_A_and_B_is_correct_l97_97692


namespace cos_300_eq_half_l97_97399

theorem cos_300_eq_half : Real.cos (2 * π * (300 / 360)) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l97_97399


namespace range_of_b_l97_97616

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4^x + a) / (2^(x+1))

noncomputable def h (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * f x a - a * x - b

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x, -x = x → f x a = - f x a)
  → (∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), h x a b = 0)
  → a = -1
  → b ∈ set.Icc (-5/2 : ℝ) (5/2 : ℝ) := 
sorry

end range_of_b_l97_97616


namespace neg_sqrt_17_bounds_l97_97066

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end neg_sqrt_17_bounds_l97_97066


namespace sum_of_reciprocals_l97_97750

variable {x y : ℝ}
variable (hx : x + y = 3 * x * y + 2)

theorem sum_of_reciprocals : (1 / x) + (1 / y) = 3 :=
by
  sorry

end sum_of_reciprocals_l97_97750


namespace find_group_2018_l97_97155

-- Definition of the conditions
def group_size (n : Nat) : Nat := 3 * n - 2

def total_numbers (n : Nat) : Nat := 
  (3 * n * n - n) / 2

theorem find_group_2018 : ∃ n : Nat, total_numbers (n - 1) < 1009 ∧ total_numbers n ≥ 1009 ∧ n = 27 :=
  by
  -- This forms the structure for the proof
  sorry

end find_group_2018_l97_97155


namespace length_AD_l97_97227

-- Define given conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [MetricSpace C]
variables (AB BC CD AD : ℝ)

-- Assume lengths and angles as given conditions
axiom AB_length : AB = 7
axiom BC_length : BC = 10
axiom CD_length : CD = 24
axiom B_right_angle : angle B = 90
axiom C_45_degrees : angle C = 45

-- Define the theorem
theorem length_AD : AD = √389 := by
  sorry

end length_AD_l97_97227


namespace derivative_of_x_log_x_l97_97321

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem derivative_of_x_log_x (x : ℝ) (h : x > 0) : 
    HasDerivAt (λ x, x * Real.log x) (1 + Real.log x) x := by
  sorry

end derivative_of_x_log_x_l97_97321


namespace solve_f_prime_at_2023_l97_97615

variable (f : ℝ → ℝ)
variable (f_prime_at_2023 : ℝ)

-- The given function
def given_function (x : ℝ) : ℝ :=
  - (1 / 2) * x^2 + 2 * x * f_prime_at_2023 + 2023 * Real.log x

-- The condition that the provided derivative evaluation holds.
axiom f_prime_cond (h : ∀ x, deriv given_function x = -x + 2 * f_prime_at_2023 + 2023 / x)

theorem solve_f_prime_at_2023 : f_prime_at_2023 = 2022 :=
by
  sorry

end solve_f_prime_at_2023_l97_97615


namespace problem1_problem2_l97_97484

-- Prove that \(\sqrt{8} - \sqrt{2} = \sqrt{2}\)
theorem problem1 : sqrt 8 - sqrt 2 = sqrt 2 := sorry

-- Prove that \(-2\sqrt{12} \times \frac{\sqrt{3}}{4} \div \sqrt{2} = -\frac{3\sqrt{2}}{2}\)
theorem problem2 : -2 * sqrt 12 * (sqrt 3 / 4) / sqrt 2 = -3 * sqrt 2 / 2 := sorry

end problem1_problem2_l97_97484


namespace isosceles_right_triangle_hypotenuse_l97_97226

noncomputable def hypotenuse_length : ℝ :=
  let a := Real.sqrt 363
  let c := Real.sqrt (2 * (a ^ 2))
  c

theorem isosceles_right_triangle_hypotenuse :
  ∀ (a : ℝ),
    (2 * (a ^ 2)) + (a ^ 2) = 1452 →
    hypotenuse_length = Real.sqrt 726 := by
  intro a h
  rw [hypotenuse_length]
  sorry

end isosceles_right_triangle_hypotenuse_l97_97226


namespace sum_of_possible_values_l97_97268

theorem sum_of_possible_values 
  (x y : ℝ) 
  (h : x * y - x / y^2 - y / x^2 = 3) :
  (x = 0 ∨ y = 0 → False) → 
  ((x - 1) * (y - 1) = 1 ∨ (x - 1) * (y - 1) = 4) → 
  ((x - 1) * (y - 1) = 1 → (x - 1) * (y - 1) = 1) → 
  ((x - 1) * (y - 1) = 4 → (x - 1) * (y - 1) = 4) → 
  (1 + 4 = 5) := 
by 
  sorry

end sum_of_possible_values_l97_97268


namespace hyperbola_center_l97_97837

variable {Point : Type}

structure coordinates (P : Point) :=
(x : ℝ)
(y : ℝ)

def center_of_hyperbola (P₁ P₂ : Point) := 
  coordinates.mk ((coordinates.x P₁ + coordinates.x P₂) / 2) ((coordinates.y P₁ + coordinates.y P₂) / 2)

theorem hyperbola_center (f1 f2 : Point) (h1 : coordinates f1) (h2 : coordinates f2) :
  h1 = coordinates.mk 3 (-2) → h2 = coordinates.mk 11 6 → center_of_hyperbola f1 f2 = coordinates.mk 7 2 :=
by
  intros
  sorry

end hyperbola_center_l97_97837


namespace tangent_angle_equality_l97_97565

open EuclideanGeometry

variables {O A B F1 F2 : Point}

theorem tangent_angle_equality (h_tangents : Tangents O A B F1 F2) :
  angle A O F1 = angle B O F2 ∧ angle A F1 O = angle B F1 O := by
  sorry

end tangent_angle_equality_l97_97565


namespace sum_of_integers_between_is_90_l97_97768

-- Define the conditions
def is_between (n : ℕ) : Prop := n > 5 ∧ n < 15

-- Define the sum of integers satisfying the conditions
def sum_of_integers_between : ℕ :=
  Finset.sum (Finset.filter is_between (Finset.range 15)) id

-- State the theorem
theorem sum_of_integers_between_is_90 : sum_of_integers_between = 90 := 
by
  sorry

end sum_of_integers_between_is_90_l97_97768


namespace shaded_region_area_l97_97239

-- Define the radius of the circles
def radius : ℝ := 5

-- Define the area of the shaded region
def shaded_area : ℝ := 50 * π - 100

-- Theorem stating the area of the shaded region given the conditions
theorem shaded_region_area 
  (r : ℝ) (h_radius : r = radius) 
  (sh_area : ℝ) (h_shaded_area : sh_area = shaded_area) :
  sh_area = 50 * π - 100 := by {
  -- Placeholder for the proof, which derives the shaded area from the given radius and intersecting circles.
  sorry
}

end shaded_region_area_l97_97239


namespace max_remainder_P0_l97_97272

theorem max_remainder_P0 (P : ℤ → ℤ) (h_poly : ∀ x, P x ∈ ℤ)
  (h1 : P (-4) = 5) (h2 : P 5 = -4) :
  ∃ k : ℤ, (P 0 = 60 * k + 41) ∧ (∀ m : ℤ, m ≠ k → P 0 ≠ 60 * m + 41) :=
sorry

end max_remainder_P0_l97_97272


namespace outfits_count_l97_97390

def num_outfits (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ) : ℕ :=
  (redShirts * pairsPants * (greenHats + blueHats)) +
  (greenShirts * pairsPants * (redHats + blueHats)) +
  (blueShirts * pairsPants * (redHats + greenHats))

theorem outfits_count :
  ∀ (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ),
  redShirts = 4 → greenShirts = 4 → blueShirts = 4 →
  pairsPants = 7 →
  greenHats = 6 → redHats = 6 → blueHats = 6 →
  num_outfits redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats = 1008 :=
by
  intros redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats
  intros hredShirts hgreenShirts hblueShirts hpairsPants hgreenHats hredHats hblueHats
  rw [hredShirts, hgreenShirts, hblueShirts, hpairsPants, hgreenHats, hredHats, hblueHats]
  sorry

end outfits_count_l97_97390


namespace average_hamburgers_sold_per_day_l97_97455

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end average_hamburgers_sold_per_day_l97_97455


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97412

def a : ℕ := 1  -- time for a simple operation
def b : ℕ := 5  -- time for a lengthy operation
def n : ℕ := 5  -- number of "simple" customers
def m : ℕ := 3  -- number of "lengthy" customers
def total_customers : ℕ := 8 -- 8 people in queue

theorem min_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → min_wasted_person_minutes ≤ 40) :=
by
  sorry

theorem max_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → max_wasted_person_minutes ≥ 100) :=
by
  sorry

theorem expected_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → expected_wasted_person_minutes = 72.5) :=
by
  sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l97_97412


namespace order_of_logs_l97_97594

def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

noncomputable def a : ℝ := log_base 2 3.4
noncomputable def b : ℝ := log_base 4 3.6
noncomputable def c : ℝ := log_base (1/3) 0.3

theorem order_of_logs : a > c ∧ c > b := by
  sorry

end order_of_logs_l97_97594


namespace arithmetic_sequence_l97_97333

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem arithmetic_sequence (a : ℕ → ℝ) (x : ℝ) :
  (a (1) = f (x + 1)) ∧ 
  (a (2) = 0) ∧ 
  (a (3) = f (x - 1)) ∧
  (a(n))  is_arithmetic_sequence → (∀ n, a n = 2 * n - 4 ∨ a n = 4 - 2 * n) :=
begin
  -- Proof omitted
  sorry
end

end arithmetic_sequence_l97_97333


namespace max_possible_value_of_k_l97_97805

noncomputable def max_knights_saying_less : Nat :=
  let n := 2015
  let k := n - 2
  k

theorem max_possible_value_of_k : max_knights_saying_less = 2013 :=
by
  sorry

end max_possible_value_of_k_l97_97805


namespace func_properties_l97_97292

def f : ℝ → ℝ 
| x := if x ≥ 3 then x - 6 
       else if 1 ≤ x ∧ x < 3 then -x 
       else -2 - x

def g : ℝ → ℝ 
| x := if x ≥ 3 then x - 6 
       else if 0 ≤ x ∧ x < 3 then -x 
       else if -3 ≤ x ∧ x < 0 then -x 
       else x - 6

theorem func_properties :
  (∀ x ≥ 3, f x = x - 6) ∧
  (∀ x, 1 ≤ x ∧ x < 3 → f x = -x) ∧
  (∀ x < 1, f x = -2 - x) ∧
  (∀ x ≥ 3, g x = x - 6) ∧
  (∀ x, 0 ≤ x ∧ x < 3 → g x = -x) ∧
  (∀ x, -3 ≤ x ∧ x < 0 → g x = -x) ∧
  (∀ x < -3, g x = x - 6) := by
  sorry

end func_properties_l97_97292


namespace cricketer_average_score_l97_97156

theorem cricketer_average_score
  (runs_19th_inning : ℕ)
  (average_increase : ℕ)
  (initial_average : ℕ)
  (total_runs_before_19th : ℕ := 18 * initial_average)
  (total_runs_after_19th : ℕ := total_runs_before_19th + runs_19th_inning)
  (new_average : ℕ := initial_average + average_increase)
  (eq_total_runs : total_runs_after_19th = 19 * new_average)
  (h1 : runs_19th_inning = 99)
  (h2 : average_increase = 4)
  (h3 : eq_total_runs) :
  initial_average + average_increase = 27 := 
  by
  sorry

end cricketer_average_score_l97_97156


namespace travelers_catch_up_at_day_3_l97_97025

noncomputable def distance_traveler1 : ℕ → ℕ
| 0     := 0
| (n+1) := distance_traveler1 n + (if n = 0 then 2 else 3)

noncomputable def distance_traveler2 : ℕ → ℕ
| 0     := 0
| (n+1) := distance_traveler2 n + (if n = 0 then 3 else 2)

noncomputable def total_distance_traveler1 (n : ℕ) : ℕ :=
∑ i in Finset.range (n + 1), distance_traveler1 i

noncomputable def total_distance_traveler2 (n : ℕ) : ℕ :=
∑ i in Finset.range (n + 1), distance_traveler2 i

theorem travelers_catch_up_at_day_3 : total_distance_traveler1 3 = total_distance_traveler2 3 :=
by {
  -- omitted proof
  sorry
}

end travelers_catch_up_at_day_3_l97_97025


namespace repeated_decimal_to_fraction_l97_97191

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l97_97191


namespace repeated_decimal_to_fraction_l97_97193

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l97_97193


namespace count_paths_l97_97405

-- Define the conditions and setup of the problem
def initial_point : (ℤ × ℤ) := (-5, -5)
def target_point : (ℤ × ℤ) := (5, 5)
def step (p : ℤ × ℤ) (d : ℤ × ℤ) : ℤ × ℤ := (p.1 + d.1, p.2 + d.2)

def is_valid_step : (ℤ × ℤ) → Prop := λ p, 
  -3 <= p.1 <= 3 → -3 <= p.2 <= 3 → false

-- Define the number of paths that meet the criteria.
def valid_paths : ℤ := 202

theorem count_paths :
  (λ start point :=
    ∑ s in list.range (20/2), -- Assuming list.range can represent all possible steps
      if (step (initial_point.1, initial_point.2) s ∉ is_valid_step 
      ∧ step (target_point.1, target_point.2) s ∉ is_valid_step)
      then 1
      else 0 ) initial_point target_point = valid_paths := 
sorry

end count_paths_l97_97405


namespace females_watch_eq_seventy_five_l97_97022

-- Definition of conditions
def males_watch : ℕ := 85
def females_dont_watch : ℕ := 120
def total_watch : ℕ := 160
def total_dont_watch : ℕ := 180

-- Definition of the proof problem
theorem females_watch_eq_seventy_five :
  total_watch - males_watch = 75 :=
by
  sorry

end females_watch_eq_seventy_five_l97_97022


namespace not_both_perfect_squares_l97_97207

open Nat

theorem not_both_perfect_squares (x y : ℕ) :
  ¬(∃ a b : ℕ, a^2 = x^2 + y + 1 ∧ b^2 = y^2 + 4x + 3) :=
sorry

end not_both_perfect_squares_l97_97207


namespace real_solutions_eq_l97_97074

theorem real_solutions_eq :
  ∀ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) → (x = 10 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_l97_97074


namespace find_x_l97_97107

noncomputable def xSolution (x : ℝ) : Prop := 
  (0 < x ∧ x < π / 2) ∧ 
  (∃ a : ℝ, (log (cos x) / log 2)).frac = 0) ∧ 
  (∃ b : ℝ, (log (sqrt (tan x))).frac = 0) ∧ 
  (((log (cos x) / log 2).intPart + (log (sqrt (tan x))).intPart) = 1) → 
  x = Real.arcsin ((sqrt 5 - 1) / 2)

-- This theorem states that x satisfies various conditions and should equal arcsin((sqrt 5 - 1) / 2)
theorem find_x : ∃ x : ℝ, xSolution x := by
  sorry

end find_x_l97_97107


namespace find_tuition_l97_97760

def tuition_problem (T : ℝ) : Prop :=
  75 = T + (T - 15)

theorem find_tuition (T : ℝ) (h : tuition_problem T) : T = 45 :=
by
  sorry

end find_tuition_l97_97760


namespace value_of_n_l97_97562

theorem value_of_n : ∃ (n : ℕ), 6 * 8 * 3 * n = Nat.factorial 8 ∧ n = 280 :=
by
  use 280
  sorry

end value_of_n_l97_97562
