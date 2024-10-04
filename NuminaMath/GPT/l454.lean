import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Polyhedron.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.ProbabilityTheory.Independent
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import analysis.special_functions.exp_log
import data.real.sqrt

namespace evaluate_expression_l454_454902

theorem evaluate_expression : (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 :=
by
  sorry

end evaluate_expression_l454_454902


namespace min_value_ineq_solve_ineq_l454_454936

theorem min_value_ineq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a^3 + 1 / b^3 + 1 / c^3 + 3 * a * b * c) ≥ 6 :=
sorry

theorem solve_ineq (x : ℝ) (h : |x + 1| - 2 * x < 6) : x > -7/3 :=
sorry

end min_value_ineq_solve_ineq_l454_454936


namespace minEmployees_correct_l454_454523

noncomputable def minEmployees (seaTurtles birdMigration bothTurtlesBirds turtlesPlants allThree : ℕ) : ℕ :=
  let onlySeaTurtles := seaTurtles - (bothTurtlesBirds + turtlesPlants - allThree)
  let onlyBirdMigration := birdMigration - (bothTurtlesBirds + allThree - turtlesPlants)
  onlySeaTurtles + onlyBirdMigration + bothTurtlesBirds + turtlesPlants + allThree

theorem minEmployees_correct :
  minEmployees 120 90 30 50 15 = 245 := by
  sorry

end minEmployees_correct_l454_454523


namespace repeating_decimal_fraction_equiv_l454_454151

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454151


namespace fraction_simplification_l454_454112

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end fraction_simplification_l454_454112


namespace sequence_k_l454_454333

theorem sequence_k (a : ℕ → ℕ) (k : ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ m n, a (m + n) = a m * a n) 
    (h3 : (finset.range 10).sum (λ i, a (k + 1 + i)) = 2^15 - 2^5) : 
    k = 4 := 
by sorry

end sequence_k_l454_454333


namespace athlete_time_to_twelfth_flag_l454_454206

theorem athlete_time_to_twelfth_flag
  (equal_distance : Prop)
  (time_to_eighth_flag : ℝ)
  (h_time_to_eighth_flag : time_to_eighth_flag = 8)
  (intervals_to_eighth_flag : ℕ)
  (h_intervals_to_eighth_flag : intervals_to_eighth_flag = 7)
  :
  let time_per_interval := time_to_eighth_flag / intervals_to_eighth_flag in
  let intervals_to_twelfth_flag := 11 in
  let total_time_to_twelfth_flag := intervals_to_twelfth_flag * time_per_interval in
  total_time_to_twelfth_flag = 88 / 7 :=
by
  -- Ensure that the properties hold true
  unfold time_per_interval intervals_to_twelfth_flag total_time_to_twelfth_flag,
  -- Use the given details and compute
  have h1 : time_per_interval = 8 / 7 := by rw [h_time_to_eighth_flag, h_intervals_to_eighth_flag],
  have h2 : intervals_to_twelfth_flag = 11 := rfl,
  have h3 : total_time_to_twelfth_flag = 11 * (8 / 7) := by rw [h1, h2],
  -- Simplify and prove the final statement
  rw [h3],
  norm_num,
  sorry

end athlete_time_to_twelfth_flag_l454_454206


namespace sum_of_areas_of_disks_l454_454554

-- Define the problem conditions
def eight_congruent_disks_on_circle (C : Circle) (r₁ : ℝ) (r₂ : ℝ) : Prop :=
  C.radius = 1 ∧
  r₁ = 1 ∧
  ∀ (i j : ℕ), (0 ≤ i ∧ i < 8) → (0 ≤ j ∧ j < 8) → i ≠ j → 
    (disks_cover_circle C r₂ ∧ no_overlaps r₂ ∧ tangent_neighbors r₂)

-- Prove the sum of the areas
theorem sum_of_areas_of_disks (C : Circle) (r₁ r₂ : ℝ) (h : eight_congruent_disks_on_circle C r₁ r₂) :
  8 * π * r₂^2 = π * (48 - 32 * sqrt 2) :=
by sorry

end sum_of_areas_of_disks_l454_454554


namespace magnitude_a_minus_2b_l454_454639

-- Definitions for vectors, their magnitudes, and angle between them
variables (a b : EuclideanSpace ℝ (Fin 2)) (theta : ℝ)
def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ := sqrt (v ⬝ v)

-- Assumptions
axiom angle_ab : angle a b = π / 6
axiom mag_a : magnitude a = 2
axiom mag_b : magnitude b = sqrt 3

-- Problem Statement
theorem magnitude_a_minus_2b : magnitude (a - 2 • b) = 2 := by
  sorry

end magnitude_a_minus_2b_l454_454639


namespace quadrilateral_longest_side_length_l454_454779

noncomputable def longest_side_length (x y : ℝ) : ℝ :=
  if h : x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0 then
    max (max (real.sqrt ((4 - 0)^2 + (0 - 4)^2)) (real.sqrt ((4 - 1/2)^2 + (0 - 0)^2)))
        (real.sqrt ((1/2 - 0)^2 + (4 - 0)^2))
  else
    0

theorem quadrilateral_longest_side_length : 
  longest_side_length = 4 * real.sqrt 2 :=
sorry

end quadrilateral_longest_side_length_l454_454779


namespace sum_powers_of_neg1_l454_454462

theorem sum_powers_of_neg1 : ∑ i in (Finset.range 2011).map (Finset.elems 1), (-1 : ℤ) ^ i = -1 := 
sorry

end sum_powers_of_neg1_l454_454462


namespace audrey_not_dreaming_time_l454_454083

theorem audrey_not_dreaming_time (total_sleep_time : ℝ) (dreaming_fraction : ℝ) :
  (total_sleep_time = 10) ∧ (dreaming_fraction = 2/5) →
  (total_sleep_time - (dreaming_fraction * total_sleep_time) = 6) :=
by
  intros h
  cases h with Htotal Hdreaming
  rw [Htotal, Hdreaming]
  sorry

end audrey_not_dreaming_time_l454_454083


namespace capture_probability_correct_l454_454395

structure ProblemConditions where
  rachel_speed : ℕ -- seconds per lap
  robert_speed : ℕ -- seconds per lap
  rachel_direction : Bool -- true if counterclockwise, false if clockwise
  robert_direction : Bool -- true if counterclockwise, false if clockwise
  start_time : ℕ -- 0 seconds
  end_time_start : ℕ -- 900 seconds
  end_time_end : ℕ -- 1200 seconds
  photo_coverage_fraction : ℚ -- fraction of the track covered by the photo

noncomputable def probability_capture_in_photo (pc : ProblemConditions) : ℚ :=
  sorry -- define and prove the exact probability

-- Given the conditions in the problem
def problem_instance : ProblemConditions :=
{
  rachel_speed := 120,
  robert_speed := 100,
  rachel_direction := true,
  robert_direction := false,
  start_time := 0,
  end_time_start := 900,
  end_time_end := 1200,
  photo_coverage_fraction := 1/3
}

-- The theorem statement we are asked to prove
theorem capture_probability_correct :
  probability_capture_in_photo problem_instance = 1/9 :=
sorry

end capture_probability_correct_l454_454395


namespace cylinder_from_sector_l454_454014

noncomputable def circle_radius : ℝ := 12
noncomputable def sector_angle : ℝ := 300
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def is_valid_cylinder (base_radius height : ℝ) : Prop :=
  2 * Real.pi * base_radius = arc_length circle_radius sector_angle ∧ height = circle_radius

theorem cylinder_from_sector :
  is_valid_cylinder 10 12 :=
by
  -- here, the proof will be provided
  sorry

end cylinder_from_sector_l454_454014


namespace repeating_decimal_fraction_equiv_l454_454149

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454149


namespace smallest_angle_of_convex_polygon_with_arithmetic_sequence_l454_454506

-- Given conditions
def is_convex (polygon : Type) : Prop := sorry  -- Definition of convexity
def is_arithmetic_sequence (angles : List ℕ) : Prop := sorry  -- Definition of arithmetic sequence with integer values

-- Lean statement
theorem smallest_angle_of_convex_polygon_with_arithmetic_sequence :
  ∀ (polygon : Type) (angles : List ℕ),
    polygon.has_sides 30 → is_convex polygon → is_arithmetic_sequence angles → 
    (angles.length = 30) ∧ (angles.head = 154) :=
by
  -- Proof will be here
  sorry

end smallest_angle_of_convex_polygon_with_arithmetic_sequence_l454_454506


namespace value_of_expression_l454_454297

theorem value_of_expression (x y z : ℕ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 1) : 
  3 * x - 2 * y + 4 * z = 9 := 
by
  sorry

end value_of_expression_l454_454297


namespace repeating_decimal_fraction_eq_l454_454162

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454162


namespace number_of_integers_satisfying_inequality_l454_454280

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℚ) / |n| ≥ 1 / 5}.to_finset.card = 10 :=
by {
  sorry
}

end number_of_integers_satisfying_inequality_l454_454280


namespace jessica_correct_percentage_l454_454356

theorem jessica_correct_percentage :
  let n₁ := 30 in let p₁ := 0.85 in let c₁ := p₁ * n₁ in
  let n₂ := 20 in let p₂ := 0.75 in let c₂ := p₂ * n₂ in
  let n₃ := 40 in let p₃ := 0.65 in let c₃ := p₃ * n₃ in
  let n₄ := 15 in let p₄ := 0.80 in let c₄ := p₄ * n₄ in
  let total_correct := c₁ + c₂ + c₃ + c₄ in
  let total_problems := n₁ + n₂ + n₃ + n₄ in
  (total_correct / total_problems) * 100 = 75 :=
by
  sorry

end jessica_correct_percentage_l454_454356


namespace sequence_k_l454_454334

theorem sequence_k (a : ℕ → ℕ) (k : ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ m n, a (m + n) = a m * a n) 
    (h3 : (finset.range 10).sum (λ i, a (k + 1 + i)) = 2^15 - 2^5) : 
    k = 4 := 
by sorry

end sequence_k_l454_454334


namespace simplify_expression_l454_454727

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l454_454727


namespace circles_tangent_to_both_l454_454359

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Prop := 
  True -- Circles are defined by their center and radius

-- Define the tangency condition
def tangent (C1 C2 : (ℝ × ℝ) × ℝ) : Prop := 
  let ((x1, y1), r1) := C1 in
  let ((x2, y2), r2) := C2 in
  dist (x1, y1) (x2, y2) = r1 + r2

-- The main theorem
theorem circles_tangent_to_both (C1 C2 : (ℝ × ℝ) × ℝ) (r : ℝ):
  Circle C1.1 C1.2 → 
  Circle C2.1 C2.2 → 
  tangent C1 C2 →
  r = 2 →
  ∃ C3 C4 : (ℝ × ℝ) × ℝ, 
    Circle C3.1 C3.2 ∧ 
    Circle C4.1 C4.2 ∧ 
    tangent C3 C1 ∧ 
    tangent C3 C2 ∧ 
    tangent C4 C1 ∧ 
    tangent C4 C2 ∧
    r = 2 :=
by sorry

end circles_tangent_to_both_l454_454359


namespace base_of_524_l454_454061

theorem base_of_524 : 
  ∀ (b : ℕ), (5 * b^2 + 2 * b + 4 = 340) → b = 8 :=
by
  intros b h
  sorry

end base_of_524_l454_454061


namespace range_of_a_l454_454222

theorem range_of_a (a : ℝ) : 
  {x : ℝ | x^2 - 4 * x + 3 < 0} ⊆ {x : ℝ | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0} → 
  -4 ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l454_454222


namespace rational_solutions_quadratic_eq_l454_454212

theorem rational_solutions_quadratic_eq (k : ℕ) (h_pos : k > 0) :
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) :=
by sorry

end rational_solutions_quadratic_eq_l454_454212


namespace no_rational_root_l454_454373

theorem no_rational_root (p q : ℤ) (hp : odd p) (hq : odd q) : ¬ ∃ x : ℚ, x^2 + 2 * p * x + 2 * q = 0 :=
by { sorry }

end no_rational_root_l454_454373


namespace alley_width_l454_454662

noncomputable def width_of_alley (BD AC h : ℝ) (BD_length AC_length h_height : Prop) : ℝ :=
  sorry

theorem alley_width (BD AC : ℝ) (h : ℝ) 
  (BD_length : BD = 3) 
  (AC_length : AC = 2) 
  (h_height : h = 1) :
  width_of_alley BD AC h BD_length AC_length h_height = 1.2311857 :=
sorry

end alley_width_l454_454662


namespace sum_of_three_consecutive_integers_is_21_l454_454477

theorem sum_of_three_consecutive_integers_is_21 (n : ℤ) :
    n ∈ {17, 11, 25, 21, 8} →
    (∃ a, n = a + (a + 1) + (a + 2)) →
    n = 21 :=
by
  intro h
  intro h_consec
  cases h_consec with a ha
  have sum_eq_three_a : n = 3 * a + 3 :=
    by linarith
  -- Verify that 21 is the only possible sum value.
  have h_n_values : n = 17 ∨ n = 11 ∨ n = 25 ∨ n = 21 ∨ n = 8 :=
    by simp at h; exact h
  cases h_n_values
  { rw h_n_values at sum_eq_three_a; contradiction }
  { rw h_n_values at sum_eq_three_a; contradiction }
  { rw h_n_values at sum_eq_three_a; contradiction }
  { exact h_n_values }
  { rw h_n_values at sum_eq_three_a; contradiction }
sorry

end sum_of_three_consecutive_integers_is_21_l454_454477


namespace maximum_PM_PN_value_is_nine_l454_454606

noncomputable def maximum_distance_difference (P M N F1 F2 : ℝ × ℝ) : ℝ :=
  let dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)
  in (dist P M) - (dist P N)

theorem maximum_PM_PN_value_is_nine :
  ∀ (P M N F1 F2 : ℝ × ℝ),
    (P.1^2 / 9 - P.2^2 / 16 = 1) → 
    ((M.1 + 5)^2 + M.2^2 = 4) → 
    ((N.1 - 5)^2 + N.2^2 = 1) → 
    (F1 = (-5, 0)) → 
    (F2 = (5, 0)) → 
    abs ((maximum_distance_difference P M N F1 F2)) ≤ 9 :=
by sorry

end maximum_PM_PN_value_is_nine_l454_454606


namespace intersection_points_calculation_l454_454597

-- Define the quadratic function and related functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def u (a b c x : ℝ) : ℝ := - f a b c (-x)
def v (a b c x : ℝ) : ℝ := f a b c (x + 1)

-- Define the number of intersection points
def m : ℝ := 1
def n : ℝ := 0

-- The proof goal
theorem intersection_points_calculation (a b c : ℝ) : 7 * m + 3 * n = 7 :=
by sorry

end intersection_points_calculation_l454_454597


namespace product_of_t_factors_l454_454548

theorem product_of_t_factors (t : ℤ) :
  (∃ a b : ℤ, (x^2 + t * x - 24 = (x + a) * (x + b) ∧ a * b = -24 ∧ t = a + b)) →
  ∏ (t : set {a | ∃ b : ℤ, a * b = -24})  = 5290000 :=
by
  sorry

end product_of_t_factors_l454_454548


namespace sam_bought_14_boxes_l454_454397

-- Defining the conditions
variables (B : ℕ) (highlighters_per_box highlighter_cost : ℕ) (packages_per_box package_price : ℕ) (pen_price : ℕ) (profit : ℤ)
def conditions :=
  highlighters_per_box = 30 ∧
  highlighter_cost = 10 ∧
  packages_per_box = 6 ∧
  package_price = 3 ∧
  pen_price = 2 ∧
  profit = 115

-- Theorem to prove the number of boxes is 14
theorem sam_bought_14_boxes (B : ℕ) : 
  conditions B 30 10 6 3 2 115 →
  let r1 := 5 * 30 / 6 * 3 in
  let r2 := (B - 5) * 30 / 3 * 2 in
  let total_revenue := r1 + r2 in
  let total_cost := B * 10 in
  total_revenue - total_cost = 115 →
  B = 14 :=
by intro h; -- sorry until the proof is completed.
 sorry

end sam_bought_14_boxes_l454_454397


namespace vacation_costs_shared_evenly_l454_454863

def total_paid (Alice Bob Carol Dave : ℕ) : ℕ :=
  Alice + Bob + Carol + Dave

def cost_per_person (total : ℕ) (people : ℕ) : ℕ :=
  total / people

def amount_to_receive (paid share : ℕ) : Int :=
  Int.ofNat paid - Int.ofNat share

def amount_to_pay (share paid : ℕ) : Int :=
  Int.ofNat share - Int.ofNat paid

theorem vacation_costs_shared_evenly :
  ∀ (Alice Bob Carol Dave : ℕ) (total_share : ℕ)
    (Alice_share Bob_share Carol_share Dave_share : Int)
    (a b : ℕ),
    total_paid Alice Bob Carol Dave = 620 →
    cost_per_person 620 4 = 155 →
    Alice_share = amount_to_receive Alice 155 →
    Bob_share = amount_to_pay 155 Bob →
    Carol_share = amount_to_pay 155 Carol →
    Dave_share = amount_to_receive Dave 155 →
    a = 0 →
    b = 35 →
    a - b = -35 :=
by
  intros Alice Bob Carol Dave total_share Alice_share Bob_share Carol_share Dave_share a b
  intro h1
  intro h2
  intro h3
  intro h4
  intro h5
  intro h6
  intro h7
  intro h8
  rw [h6, h7]
  simp
  sorry

end vacation_costs_shared_evenly_l454_454863


namespace product_of_constants_l454_454547

theorem product_of_constants (t : ℤ) :
  (∃ (a b : ℤ), (x^2 + t*x - 24) = (x + a) * (x + b) ∧ t = a + b) →
  ∏ (t ∈ {23, 10, 5, 2, -2, -5, -10, -23}) = -5290000 :=
sorry

end product_of_constants_l454_454547


namespace find_larger_number_l454_454003

theorem find_larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : 2 * (x + y) = 40) : x = 12.5 :=
by 
  sorry

end find_larger_number_l454_454003


namespace number_of_messages_l454_454435

theorem number_of_messages (n : ℕ) (choices : ℕ) (k : ℕ) (f : Finset (Fin n) → ℕ → Prop): 
  n = 7 ∧ choices = 2 ∧ k = 3 ∧ (∀ (s : Finset (Fin n)), f s k → s.card = k ∧ ∀ x y ∈ s, |x - y| > 1) →
  (let possible_positions := {s ∈ (Finset.powerset (Finset.univ : Finset (Fin n))) | f s k} in
  possible_positions.card * choices^k = 80) :=
by
  intros
  skip -- This is a placeholder indicating the proof should be filled in
  sorry

end number_of_messages_l454_454435


namespace even_product_probability_l454_454918

theorem even_product_probability
    (die_faces : Finset ℕ)
    (h_die : ∀ n ∈ die_faces, 1 ≤ n ∧ n ≤ 8)
    (h_size : die_faces.card = 8):
    let outcomes := (die_faces.product die_faces).filter (λ p, (p.1 * p.2) % 2 = 0)
    in (outcomes.card / (die_faces.card * die_faces.card) : ℚ) = 3 / 4 :=
by
  sorry

end even_product_probability_l454_454918


namespace min_value_expression_l454_454591

theorem min_value_expression (x : ℚ) : ∃ x : ℚ, (2 * x - 5)^2 + 18 = 18 :=
by {
  use 2.5,
  sorry
}

end min_value_expression_l454_454591


namespace Tim_marbles_l454_454915

theorem Tim_marbles (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 :=
by
  sorry

end Tim_marbles_l454_454915


namespace time_for_P_to_finish_job_alone_l454_454391

variable (T : ℝ)

theorem time_for_P_to_finish_job_alone (h1 : 0 < T) (h2 : 3 * (1 / T + 1 / 20) + 0.4 * (1 / T) = 1) : T = 4 :=
by
  sorry

end time_for_P_to_finish_job_alone_l454_454391


namespace fraction_repeating_decimal_l454_454187

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454187


namespace equal_internal_angles_l454_454669

variable (α : ℝ)
variable (A B C D E : Type)
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variable [InnerProductSpace ℝ D] [InnerProductSpace ℝ E]

-- Conditions
variable (angle_BAE : ℝ) (len_BC len_CD len_DE : ℝ)
variable (angle_BCD angle_CDE : ℝ)

-- Assumptions
def convex_pentagon (A B C D E : Type) [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] :=
  angle_BAE = 3 * α ∧
  len_BC = len_CD ∧ len_CD = len_DE ∧
  angle_BCD = 180 - 2 * α ∧ angle_CDE = 180 - 2 * α

-- Proof statement
theorem equal_internal_angles (A B C D E : Type) [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ D] [InnerProductSpace ℝ E]
  (h : convex_pentagon A B C D E) :
  let β := α in
  angle BAC = β ∧ angle CAD = β ∧ angle DAE = β :=
by
  sorry

end equal_internal_angles_l454_454669


namespace length_PQ_l454_454360

noncomputable def midpoint (P Q S : (ℝ × ℝ)) :=
  (S.1 = (P.1 + Q.1) / 2) ∧ 
  (S.2 = (P.2 + Q.2) / 2)

def on_line_1 (P : (ℝ × ℝ)) :=
  P.2 = (12 / 5) * P.1

def on_line_2 (Q : (ℝ × ℝ)) :=
  Q.2 = (4 / 15) * Q.1

def distance (P Q : (ℝ × ℝ)) :=
  (real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2))

def relatively_prime (m n : ℕ) : Prop :=
  nat.gcd m n = 1

theorem length_PQ (m n : ℕ) 
  (hP : (P : (ℝ × ℝ)) (on_line_1 P))
  (hQ : (Q : (ℝ × ℝ)) (on_line_2 Q))
  (hS_mid : (midpoint P Q (10, 8)))
  (hmn_rel_prime : relatively_prime m n)
  (h_eq_length : distance P Q = m / n) :
  m + n = 337 :=
sorry

end length_PQ_l454_454360


namespace find_a1_a10_value_l454_454251

variable {α : Type} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a1_a10_value (a : ℕ → α) (h1 : is_geometric_sequence a)
    (h2 : a 4 + a 7 = 2) (h3 : a 5 * a 6 = -8) : a 1 + a 10 = -7 := by
  sorry

end find_a1_a10_value_l454_454251


namespace ratio_perimeter_l454_454724

-- Definitions
variables (x y : ℝ)

-- Definitions of perimeters based on given conditions
def perimeter_A := 2 * (2 * x + 3 * x)
def perimeter_B := 2 * (2 * x + 4 * x)

-- Theorem stating the correct ratio
theorem ratio_perimeter (x y : ℝ) : (perimeter_A x y) / (perimeter_B x y) = 5 / 6 :=
by
  sorry

end ratio_perimeter_l454_454724


namespace incorrect_inequality_l454_454579

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end incorrect_inequality_l454_454579


namespace num_integers_such_that_n_plus_2i_pow_6_is_integer_l454_454892

theorem num_integers_such_that_n_plus_2i_pow_6_is_integer :
  {n : ℤ | ∃ (n : ℤ), (n + 2 * complex.I)^6 ∈ ℤ}.to_finset.card = 1 := 
sorry

end num_integers_such_that_n_plus_2i_pow_6_is_integer_l454_454892


namespace men_in_business_class_l454_454400

theorem men_in_business_class (total_passengers : ℕ) (percentage_men : ℝ)
  (fraction_business_class : ℝ) (num_men_in_business_class : ℕ) 
  (h1 : total_passengers = 160) 
  (h2 : percentage_men = 0.75) 
  (h3 : fraction_business_class = 1 / 4) 
  (h4 : num_men_in_business_class = total_passengers * percentage_men * fraction_business_class) : 
  num_men_in_business_class = 30 := 
  sorry

end men_in_business_class_l454_454400


namespace min_sum_of_distances_l454_454609

noncomputable def M (x : ℝ) : ℝ × ℝ := (x, sqrt (8 * x))  -- Point on the parabola
noncomputable def F : ℝ × ℝ := (2, 0)  -- Focus of the parabola (a derived knowledge)
def A : set (ℝ × ℝ) := { p | (p.1 - 3)^2 + (p.2 + 1)^2 = 1 }  -- Points on the circle

-- |AM| denotes the Euclidean distance between points A and M
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_sum_of_distances :
  ∃ A ∈ A, ∃ x : ℝ, distance A (M x) + distance (M x) F = 4 :=
by sorry

end min_sum_of_distances_l454_454609


namespace weight_of_new_person_l454_454758

-- Definition of the problem
def average_weight_increases (W : ℝ) (N : ℝ) : Prop :=
  let increase := 2.5
  W - 45 + N = W + 8 * increase

-- The main statement we need to prove
theorem weight_of_new_person (W : ℝ) : ∃ N, average_weight_increases W N ∧ N = 65 := 
by
  use 65
  unfold average_weight_increases
  sorry

end weight_of_new_person_l454_454758


namespace part1_part2_l454_454613

open Real

/-- Part I: Given f(x) = x^2 - a ln(x) with a > 0,
  if the minimum value of f(x) is 1, then a = 2 -/
theorem part1 (a x : ℝ) (h_a : a > 0) (h_min : ∃ x > 0, x^2 - a * log x = 1) : a = 2 :=
sorry

/-- Part II: If f^2(x) * e^(2x) - 6m * f(x) * e^x + 9m = 0
  has a unique real root in the interval [1, +∞), find the range of values for m -/
theorem part2 (m x : ℝ) (h_eq : ∀ x ≥ 1, (x^2 - 2 * log x)^2 * exp(2 * x) - 6 * m * (x^2 - 2 * log x) * exp(x) + 9 * m = 0)
  (h_unique : ∃! x, x ≥ 1 ∧ (x^2 - 2 * log x)^2 * exp(2 * x) - 6 * m * (x^2 - 2 * log x) * exp(x) + 9 * m = 0) :
  m = 1 ∨ m ≥ exp(2) / (6 * exp(1) - 9) :=
sorry

end part1_part2_l454_454613


namespace fraction_equivalent_of_repeating_decimal_l454_454192

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454192


namespace fraction_equivalent_of_repeating_decimal_l454_454194

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454194


namespace repeating_decimal_equals_fraction_l454_454138

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454138


namespace sequence_value_a1_l454_454325

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → a n = -3 * a (n + 1)

def sum_even_terms_limit (a : ℕ → ℝ) : Prop :=
  filter.tendsto (λ n : ℕ, ∑ i in finset.range (n + 1).filter (λ m, even (2 * m)), a (2 * (m+1))) filter.at_top (𝓝 (9 / 2))

theorem sequence_value_a1 {a : ℕ → ℝ} (h1 : sequence a) (h2 : sum_even_terms_limit a) : a 1 = -12 :=
begin
  sorry
end

end sequence_value_a1_l454_454325


namespace audrey_not_dreaming_l454_454084

theorem audrey_not_dreaming :
  ∀ (total_sleep : ℕ) (dream_fraction : ℚ),
  total_sleep = 10 → dream_fraction = 2 / 5 →
  (total_sleep - (dream_fraction * total_sleep).toInt) = 6 :=
by
  intros total_sleep dream_fraction h1 h2
  sorry

end audrey_not_dreaming_l454_454084


namespace number_of_members_l454_454020

theorem number_of_members (n : ℕ) (h1 : ∀ k, k ∈ finset.range n → k = n * 1) (h2 : n * n = 1369) : n = 37 :=
by sorry

end number_of_members_l454_454020


namespace repeating_decimal_to_fraction_l454_454123

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454123


namespace quadratic_coefficient_nonzero_l454_454294

theorem quadratic_coefficient_nonzero (a : ℝ) (x : ℝ) :
  (a - 3) * x^2 - 3 * x - 4 = 0 → a ≠ 3 :=
sorry

end quadratic_coefficient_nonzero_l454_454294


namespace stan_water_intake_l454_454756

-- Define the constants and parameters given in the conditions
def words_per_minute : ℕ := 50
def pages : ℕ := 5
def words_per_page : ℕ := 400
def water_per_hour : ℚ := 15  -- use rational numbers for precise division

-- Define the derived quantities from the conditions
def total_words : ℕ := pages * words_per_page
def total_minutes : ℕ := total_words / words_per_minute
def water_per_minute : ℚ := water_per_hour / 60

-- State the theorem
theorem stan_water_intake : 10 = total_minutes * water_per_minute := by
  sorry

end stan_water_intake_l454_454756


namespace find_p_find_triangle_area_minimum_l454_454636

-- Define the parabola and line equations and their intersection points.
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
noncomputable def line_eq (x y : ℝ) := x - 2 * y + 1 = 0
noncomputable def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (p : ℝ) :
  (∃ (A B : ℝ × ℝ), parabola_eq p A.1 A.2 ∧ parabola_eq p B.1 B.2 ∧ 
   line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ 
   distance A.1 A.2 B.1 B.2 = 4 * real.sqrt 15) → p = 2 := sorry

-- Define conditions for the second part of the problem
noncomputable def focus_x := 1
noncomputable def focus_y := 0
noncomputable def mf_dot_nf_eq_zero (x1 y1 x2 y2 : ℝ) := 
  (x1 - focus_x) * (x2 - focus_x) + y1 * y2 = 0
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) := 
  1 / 2 * abs ((x1 - focus_x) * (y2 - focus_y) - (x2 - focus_x) * (y1 - focus_y))

theorem find_triangle_area_minimum (M N : ℝ × ℝ) :
  (parabola_eq 2 M.1 M.2 ∧ parabola_eq 2 N.1 N.2 ∧ 
  mf_dot_nf_eq_zero M.1 M.2 N.1 N.2) → 
  ∃ (Amin : ℝ), Amin = 12 - 8 * real.sqrt 2 ∧ triangle_area M.1 M.2 N.1 N.2 = Amin := sorry

end find_p_find_triangle_area_minimum_l454_454636


namespace area_triangle_AOM_l454_454322

-- Given definitions and conditions
variables (A B C D M O: Type)
variable [parallelogram ABCD]
variable (midpoint_M : M = midpoint AD)
variable (area_MOCD : area (quadrilateral M O C D) = 5)

-- Proof statement
theorem area_triangle_AOM : 
  area (triangle A O M) = 1 := sorry

end area_triangle_AOM_l454_454322


namespace planes_parallel_l454_454002

def is_collinear (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2, k * v1.3)

theorem planes_parallel : 
  is_collinear (1, 2, 1) (2, 4, 2) :=
by 
  use 2 
  simp
  sorry

end planes_parallel_l454_454002


namespace correct_proposition_l454_454961

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Defining proposition p
def p : Prop := ∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x < 0

-- Defining proposition q
def q : Prop := ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2

-- Theorem statement to prove the correct answer
theorem correct_proposition : (¬ p) ∧ (¬ q) :=
by
  sorry

end correct_proposition_l454_454961


namespace pizza_slices_l454_454888

-- Definitions of conditions
def slices (H C : ℝ) : Prop :=
  (H / 2 - 3 + 2 * C / 3 = 11) ∧ (H = C)

-- Stating the theorem to prove
theorem pizza_slices (H C : ℝ) (h : slices H C) : H = 12 :=
sorry

end pizza_slices_l454_454888


namespace equilateral_triangle_isosceles_triangle_l454_454346

variable {A B C a b c : ℝ}

-- Condition 1
axiom cond1 : (a / Real.cos A) = (b / Real.cos B) = (c / Real.cos C)

-- Condition 2
axiom cond2 : Real.sin A = 2 * Real.sin B * Real.cos C

-- Proof statement: Triangle is equilateral given condition 1
theorem equilateral_triangle : cond1 → (A = B ∧ B = C ∧ A = C) :=
  by sorry

-- Proof statement: Triangle is isosceles given condition 2
theorem isosceles_triangle : cond2 → (B = C) :=
  by sorry

end equilateral_triangle_isosceles_triangle_l454_454346


namespace faster_train_speed_l454_454447

theorem faster_train_speed (V_s : ℝ := 36) (time_crossing : ℝ := 37) (len_faster_train : ℝ := 370) :
  ∃ V_f : ℝ, V_f = 72 :=
by
  -- Given conditions
  have h1 : V_s = 36 := rfl
  have h2 : time_crossing = 37 := rfl
  have h3 : len_faster_train = 370 := rfl
  
  -- Calculations (skipped, as we are not providing proof here)
  sorry

end faster_train_speed_l454_454447


namespace repeating_decimal_eq_fraction_l454_454176

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454176


namespace problem_statement_l454_454716

noncomputable def dates : List ℕ := [121, 122, ..., 1228, 1129, 1130, 731]

def mean (l : List ℕ) : ℝ := (l.sum : ℝ) / l.length

def median (l : List ℕ) : ℕ := 
  let sorted := l.sort
  sorted.sorted_nth (l.length / 2)

def median_mod_remainder (l : List ℕ) (modulus : ℕ) : ℕ := 
  let remainders := l.map (λ x => x % modulus)
  let sorted   := remainders.sort
  sorted.sorted_nth (remainders.length / 2)

theorem problem_statement :
  let μ := mean dates
  let M := median dates
  let d := median_mod_remainder dates 365
  d < μ ∧ μ < M :=
by
  sorry

end problem_statement_l454_454716


namespace water_height_in_cylinder_l454_454079

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * Math.pi * r^2 * h
noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := Math.pi * r^2 * h

theorem water_height_in_cylinder (r_cone h_cone r_cylinder h_cylinder : ℝ) (h1 : r_cone = 15) (h2 : h_cone = 25) (h3 : r_cylinder = 30) :
  h_cylinder = 1875 / 900 → h_cylinder ≈ 2.1 := by
  sorry

end water_height_in_cylinder_l454_454079


namespace q_is_false_of_pq_false_and_notp_false_l454_454303

variables (p q : Prop)

theorem q_is_false_of_pq_false_and_notp_false (hpq_false : ¬(p ∧ q)) (hnotp_false : ¬(¬p)) : ¬q := 
by 
  sorry

end q_is_false_of_pq_false_and_notp_false_l454_454303


namespace fraction_rep_finite_geom_series_036_l454_454129

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454129


namespace exists_4_div_3_no_3_div_4_l454_454021

-- Definitions for numbers composed of digit '4' and digit '3'
def is_composed_digit_4 (n : ℕ) : Prop :=
  ∀ d ∈ (nat.digits 10 n), d = 4
    
def is_composed_digit_3 (n : ℕ) : Prop :=
  ∀ d ∈ (nat.digits 10 n), d = 3

-- Proof statement for Question (a)
theorem exists_4_div_3 : ∃ n m : ℕ, is_composed_digit_4 n ∧ is_composed_digit_3 m ∧ m ∣ n :=
by {
  -- Placeholder for the proof
  sorry
}

-- Proof statement for Question (b)
theorem no_3_div_4 : ∀ n m : ℕ, is_composed_digit_3 n → is_composed_digit_4 m → ¬ m ∣ n :=
by {
  -- Placeholder for the proof
  sorry
}

end exists_4_div_3_no_3_div_4_l454_454021


namespace relationship_among_a_b_c_l454_454219

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 7 / (2 * Real.log 2)
noncomputable def c : ℝ := 0.3 ^ (-3 / 2)

theorem relationship_among_a_b_c :
  c > a ∧ a > b := 
sorry

end relationship_among_a_b_c_l454_454219


namespace ratio_identity_l454_454569

-- Given system of equations
def system_of_equations (k : ℚ) (x y z : ℚ) :=
  x + k * y + 2 * z = 0 ∧
  2 * x + k * y + 3 * z = 0 ∧
  3 * x + 5 * y + 4 * z = 0

-- Prove that for k = -7/5, the system has a nontrivial solution and 
-- that the ratio xz / y^2 equals -25
theorem ratio_identity (x y z : ℚ) (k : ℚ) (h : system_of_equations k x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  k = -7 / 5 → x * z / y^2 = -25 :=
by
  sorry

end ratio_identity_l454_454569


namespace minimum_students_for_200_candies_l454_454997

theorem minimum_students_for_200_candies (candies : ℕ) (students : ℕ) (h_candies : candies = 200) : students = 21 :=
by
  sorry

end minimum_students_for_200_candies_l454_454997


namespace repeating_decimals_subtraction_l454_454533

/--
Calculate the value of 0.\overline{234} - 0.\overline{567} - 0.\overline{891}.
Express your answer as a fraction in its simplest form.

Shown that:
Let x = 0.\overline{234}, y = 0.\overline{567}, z = 0.\overline{891},
Then 0.\overline{234} - 0.\overline{567} - 0.\overline{891} = -1224/999
-/
theorem repeating_decimals_subtraction : 
  let x : ℚ := 234 / 999
  let y : ℚ := 567 / 999
  let z : ℚ := 891 / 999
  x - y - z = -1224 / 999 := 
by
  sorry

end repeating_decimals_subtraction_l454_454533


namespace number_of_combinations_of_blocks_l454_454065

theorem number_of_combinations_of_blocks : 
  (∃ (grid : fin 6 → fin 6 → Prop), 
    (∀ i, ∃! j, grid i j) ∧ 
    (∀ j, ∃! i, grid i j)) → 
  ∃ (n : ℕ), n = 5400 :=
by
  sorry

end number_of_combinations_of_blocks_l454_454065


namespace heptagonal_tiling_exists_l454_454679

def heptagonal_tiling_possible : Prop := 
  ∃ (partition : Set (Polygon R)),
    (∀ poly ∈ partition, poly.sides > 1) ∧
    (∀ poly ∈ partition, ∃ (center : Point R), ∃ (angle : R), angle = 360 / 7 ∧ poly.rotate(center, angle) = poly) ∧
    (is_tiling partition)

theorem heptagonal_tiling_exists : heptagonal_tiling_possible := 
  sorry

end heptagonal_tiling_exists_l454_454679


namespace repeating_decimal_to_fraction_l454_454116

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454116


namespace max_area_triangle_l454_454948
-- Broader imports for necessary libraries

-- Starting the statement of the theorem
theorem max_area_triangle
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a * cos C + c * cos A = 3)
  (h2 : a^2 + c^2 = 9 + a * c)
  (triangle_cond : A + B + C = π)
  (angles_nonneg : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) :
  ∃ (area : ℝ), area = (9 * sqrt 3) / 4 :=
by
  sorry

-- A simple statement confirming the maximum area of the triangle

end max_area_triangle_l454_454948


namespace position_of_2018_2187_in_sequence_l454_454962

theorem position_of_2018_2187_in_sequence :
  let sequence := λ n m, (m : ℕ) / (3^n : ℕ)
  let numerator := λ n, list.range (3^n) |>.filter (λ k, k.gcd (3^n) = 1) in
  let pos_all_before := (nat.sum (λ n, (3^n - 1))) in
  (2018 / 2187) = sequence 7 2018 -> ∃ pos, pos = pos_all_before + 1009 :=
sorry -- This is the proof placeholder.

end position_of_2018_2187_in_sequence_l454_454962


namespace repeating_decimal_eq_fraction_l454_454178

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454178


namespace triangle_side_and_expression_l454_454965

noncomputable def a (b c A : ℝ) : ℝ :=
real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)

theorem triangle_side_and_expression (A : ℝ) (b : ℝ) (c : ℝ) (ha : A = real.pi / 3) (hb : b = 1) (hc : c = 4) :
  let a := a b c A in
  (a = real.sqrt 13) ∧
  (a + b + c) / (real.sin A + real.sin (real.asin (b * real.sin A / a)) + real.sin (real.acos ((a^2 + c^2 - b^2) / (2 * a * c)))) = (2 * real.sqrt 39) / 3 :=
by {sorry}

end triangle_side_and_expression_l454_454965


namespace fraction_repeating_decimal_l454_454182

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454182


namespace ladder_base_l454_454044

theorem ladder_base (h : ℝ) (b : ℝ) (l : ℝ)
  (h_eq : h = 12) (l_eq : l = 15) : b = 9 :=
by
  have hypotenuse := l
  have height := h
  have base := b
  have pythagorean_theorem : height^2 + base^2 = hypotenuse^2 := by sorry 
  sorry

end ladder_base_l454_454044


namespace stable_subsets_even_count_l454_454375

def is_stable_subset (S : Finset (ℕ × ℕ)) (T : Finset (ℕ × ℕ)) : Prop :=
  ∀ {x y : ℕ}, (x, y) ∈ T → (x, y) ∈ S →
    ∀ {x' y' : ℕ}, x' ≤ x → y' ≤ y → (x', y') ∈ S → (x', y') ∈ T

theorem stable_subsets_even_count (S : Finset (ℕ × ℕ)) (hS : ∀ {x y : ℕ}, (x, y) ∈ S →
  ∀ {x' y' : ℕ}, x' ≤ x → y' ≤ y → (x', y') ∈ S) :
  ∃ T : Finset (ℕ × ℕ), is_stable_subset S T ∧
  (∑ T in (S.powerset.filter (λ T, is_stable_subset S T)), if T.card % 2 = 0 then 1 else 0) * 2 ≥ 2 ^ S.card :=
sorry

end stable_subsets_even_count_l454_454375


namespace train_pass_time_l454_454645

-- Define the conditions based on the problem statement
def train_length : ℝ := 150 -- in meters
def train_speed_kmph : ℝ := 90 -- in km/hr
def km_to_meter : ℝ := 1000 -- 1 km = 1000 meters
def hour_to_second : ℝ := 3600 -- 1 hour = 3600 seconds

-- Convert speed from km/hr to m/s
def train_speed_mps : ℝ := train_speed_kmph * (km_to_meter / hour_to_second) -- in m/s

-- Define the expected time to pass the pole
def expected_time : ℝ := 6 -- in seconds

-- The theorem to prove:
theorem train_pass_time : (train_length / train_speed_mps) = expected_time :=
by
  sorry -- Proof to be provided

end train_pass_time_l454_454645


namespace probability_of_prime_sum_less_than_30_l454_454553

open scoped BigOperators

noncomputable def ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def is_prime_sum_less_than_30 (a b : ℕ) : Prop :=
  Nat.Prime (a + b) ∧ (a + b) < 30

theorem probability_of_prime_sum_less_than_30 :
  (ten_primes.card.choose 2) = 45 ∧
  ((ten_primes.filter (λ ab : ℕ × ℕ, is_prime_sum_less_than_30 ab.1 ab.2)).card : ℚ) / ten_primes.card.choose 2 = 4 / 45 :=
sorry

end probability_of_prime_sum_less_than_30_l454_454553


namespace regression_analysis_proof_l454_454951
-- Import necessary libraries

-- Definitions and conditions used in the problem
noncomputable def empirical_regression (x y : List ℝ) : ℝ → ℝ := sorry

def correct_statements (A B C D : Prop) : set Prop :=
  { p | (B ∧ C ∧ D) ∧ ¬ A }

-- Lean 4 statement expressing the proof problem
theorem regression_analysis_proof (A B C D : Prop) 
  (empirical_regression_correct : ∀ (x y : List ℝ), empirical_regression x y (sum x / length x) = sum y / length y) :
  correct_statements A B C D :=
by {
  sorry
}

end regression_analysis_proof_l454_454951


namespace no_real_solutions_l454_454644

noncomputable def f (x : ℝ) : ℝ :=
  (x^2000) / 2001 + 2 * real.sqrt 3 * x^2 - 2 * real.sqrt 5 * x + real.sqrt 3

theorem no_real_solutions : ¬∃ x : ℝ, f x = 0 :=
by
  let f : ℝ → ℝ := λ x, (x^2000) / 2001 + 2 * real.sqrt 3 * x^2 - 2 * real.sqrt 5 * x + real.sqrt 3
  sorry

end no_real_solutions_l454_454644


namespace sequence_k_eq_4_l454_454330

theorem sequence_k_eq_4 {a : ℕ → ℕ} (h1 : a 1 = 2) (h2 : ∀ m n, a (m + n) = a m * a n)
    (h3 : ∑ i in finset.range 10, a (4 + i) = 2^15 - 2^5) : 4 = 4 :=
by
  sorry

end sequence_k_eq_4_l454_454330


namespace fraction_rep_finite_geom_series_036_l454_454125

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454125


namespace profit_without_discount_l454_454482

-- Definitions based on conditions
def cost_price : ℝ := 100
def profit_percentage_with_discount : ℝ := 34.9
def discount_percentage : ℝ := 5

-- The main statement
theorem profit_without_discount (CP : ℝ) (profit_perc : ℝ) (discount_perc : ℝ) :
  profit_perc = 34.9 → discount_perc = 5 → 
  let SP := CP + (profit_perc / 100) * CP in
  let profit_perc_without_discount := ((SP - CP) / CP) * 100 in
  profit_perc_without_discount = 34.9 :=
by
  sorry

end profit_without_discount_l454_454482


namespace greatest_k_inequality_l454_454563

theorem greatest_k_inequality :
  ∃ k : ℕ, k = 13 ∧ ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a * b * c = 1 → 
  (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4) :=
sorry

end greatest_k_inequality_l454_454563


namespace Polyas_probability_relation_l454_454451

variable (Z : ℕ → ℤ → ℝ)

theorem Polyas_probability_relation (n : ℕ) (k : ℤ) :
  Z n k = (1/2) * (Z (n-1) (k-1) + Z (n-1) (k+1)) :=
by
  sorry

end Polyas_probability_relation_l454_454451


namespace simplify_fraction_l454_454732

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454732


namespace total_right_handed_players_l454_454819

-- Defining the conditions and the given values
def total_players : ℕ := 61
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers

-- The proof goal
theorem total_right_handed_players 
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : non_throwers = total_players - throwers)
  (h4 : left_handed_non_throwers = non_throwers / 3)
  (h5 : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h6 : left_handed_non_throwers * 3 = non_throwers)
  : throwers + right_handed_non_throwers = 53 :=
sorry

end total_right_handed_players_l454_454819


namespace percentage_decrease_10_l454_454091

def stocks_decrease (F J M : ℝ) (X : ℝ) : Prop :=
  J = F * (1 - X / 100) ∧
  J = M * 1.20 ∧
  M = F * 0.7500000000000007

theorem percentage_decrease_10 {F J M X : ℝ} (h : stocks_decrease F J M X) :
  X = 9.99999999999992 :=
by
  sorry

end percentage_decrease_10_l454_454091


namespace estimate_formula_integral_l454_454103

noncomputable def estimate_integral (a b : ℝ) (φ f : ℝ → ℝ) (n : ℕ) (x_i : ℕ → ℝ) : ℝ :=
  (1 : ℝ) / n * ∑ i in finset.range n, (φ (x_i i)) / (f (x_i i))

theorem estimate_formula_integral
  (a b : ℝ)
  (φ f : ℝ → ℝ)
  (I : ℝ)
  (n : ℕ)
  (x_i : ℕ → ℝ)
  (h_f_nonneg : ∀ x, a ≤ x → x ≤ b → 0 ≤ f x)
  (h_f_integral : ∫ x in set.Icc a b, f x = 1)
  (h_I_def : I = ∫ x in set.Icc a b, φ x) :
  I = estimate_integral a b φ f n x_i :=
sorry

end estimate_formula_integral_l454_454103


namespace bijective_equation1_neither_injective_surjective_equation2_neither_injective_surjective_equation3_l454_454824

-- Definition and initial conditions for the first equation
def equation1 (f : ℝ → ℝ) := ∀ x y : ℝ, f(x + f(y)) = 2 * f(x) + y

-- Prove that f is bijective
theorem bijective_equation1 (f : ℝ → ℝ) (h : equation1 f) : function.bijective f :=
sorry

-- Definition and initial conditions for the second equation
def equation2 (f : ℝ → ℝ) := ∀ x y : ℝ, f(x + y) = f(x) * f(y)

-- Prove that f is neither injective nor surjective
theorem neither_injective_surjective_equation2 (f : ℝ → ℝ) (h : equation2 f) :
  ¬ function.injective f ∧ ¬ function.surjective f :=
sorry

-- Definition and initial conditions for the third equation
def equation3 (f : ℝ → ℝ) := ∀ x : ℝ, f(f(x)) = Real.sin x

-- Prove that f is neither injective nor surjective
theorem neither_injective_surjective_equation3 (f : ℝ → ℝ) (h : equation3 f) :
  ¬ function.injective f ∧ ¬ function.surjective f :=
sorry

end bijective_equation1_neither_injective_surjective_equation2_neither_injective_surjective_equation3_l454_454824


namespace stratified_sampling_senior_titles_count_l454_454051

theorem stratified_sampling_senior_titles_count :
  ∀ (total_staff senior_titles intermediate_titles junior_titles sample_size : ℕ),
  total_staff = 150 →
  senior_titles = 15 →
  intermediate_titles = 45 →
  junior_titles = 90 →
  sample_size = 30 →
  (senior_titles.to_rat / total_staff.to_rat * sample_size.to_rat).natAbs = 3 :=
by
  intros total_staff senior_titles intermediate_titles junior_titles sample_size
  intros h_total h_senior h_intermediate h_junior h_sample
  rw [h_total, h_senior, h_intermediate, h_junior, h_sample]
  sorry

end stratified_sampling_senior_titles_count_l454_454051


namespace positive_rational_sum_of_sequence_l454_454100

theorem positive_rational_sum_of_sequence (a b : ℕ) (h : 0 < a ∧ 0 < b) :
  ∃ (s : Finset ℕ), (∀ n ∈ s, 0 < n) ∧ (∑ n in s, 1/(n : ℚ)) = (a : ℚ) / (b : ℚ) :=
by
  sorry

end positive_rational_sum_of_sequence_l454_454100


namespace oblique_drawing_parallelogram_correct_l454_454796

theorem oblique_drawing_parallelogram_correct :
  (∀ (T : Type) [h : PlaneFigure T],
    (obliqueDiagram (equilateralTriangle T) ≠ (equilateralTriangle T))
    ∧ (obliqueDiagram (parallelogram T) = (parallelogram T))
    ∧ (obliqueDiagram (square T) ≠ (square T))
    ∧ (obliqueDiagram (circle T) ≠ (circle T))
  ) → 
  (correctOption = OptionB) :=
by
  sorry

end oblique_drawing_parallelogram_correct_l454_454796


namespace determine_a_value_l454_454721

noncomputable def rectangle_value (a : ℝ) : ℝ := (1 + real.sqrt 33) / 2 + 8

theorem determine_a_value :
  ∃ a : ℝ, 
    -- Conditions
    (∃ x : ℝ, ∃ y : ℝ, y = real.log a x ∧ x * 4 = 32 ∧ (x + 8) * real.log a (x + 8) = 2 * real.log a x ∧ (x + 8, y + 4)) ∧ 
    -- Conclusion
    a = real.root 4 (rectangle_value a) :=
begin
  sorry
end

end determine_a_value_l454_454721


namespace longest_side_quadrilateral_l454_454778

theorem longest_side_quadrilateral :
  let A := (4, 0)
  let B := (3, 1)
  let C := (0, 1)
  let D := (0, 4)
  let dist (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  max (dist A B) (max (dist B C) (max (dist C D) (dist D A))) = 4 * real.sqrt 2 :=
by
  sorry

end longest_side_quadrilateral_l454_454778


namespace daniel_age_is_correct_l454_454004

open Nat

-- Define Uncle Ben's age
def uncleBenAge : ℕ := 50

-- Define Edward's age as two-thirds of Uncle Ben's age
def edwardAge : ℚ := (2 / 3) * uncleBenAge

-- Define that Daniel is 7 years younger than Edward
def danielAge : ℚ := edwardAge - 7

-- Assert that Daniel's age is 79/3 years old
theorem daniel_age_is_correct : danielAge = 79 / 3 := by
  sorry

end daniel_age_is_correct_l454_454004


namespace repeating_decimal_eq_fraction_l454_454175

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454175


namespace length_of_BC_l454_454675

theorem length_of_BC (A B C M : Type) 
  [AddGroup A] 
  [Module ℝ A]
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)
  (AM : ℝ)
  (hAB : AB = 1)
  (hAC : AC = 2)
  (hMedian : BC = AM) :
  BC = Real.sqrt 2 :=
by
  sorry

end length_of_BC_l454_454675


namespace M_is_set_of_positive_rationals_le_one_l454_454689

def M : Set ℚ := {x | 0 < x ∧ x ≤ 1}

axiom contains_one (M : Set ℚ) : 1 ∈ M

axiom closed_under_operations (M : Set ℚ) :
  ∀ x ∈ M, (1 / (1 + x) ∈ M) ∧ (x / (1 + x) ∈ M)

theorem M_is_set_of_positive_rationals_le_one :
  M = {x | 0 < x ∧ x ≤ 1} :=
sorry

end M_is_set_of_positive_rationals_le_one_l454_454689


namespace basketball_starting_lineups_l454_454497

noncomputable def choose (n k : ℕ) : ℕ :=
nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem basketball_starting_lineups :
  let total_players := 12 in
  let choose_forwards := choose total_players 2 in
  let choose_guards := choose (total_players - 2) 2 in
  let choose_center := choose (total_players - 4) 1 in
  choose_forwards * choose_guards * choose_center = 23760 := by
  sorry

end basketball_starting_lineups_l454_454497


namespace sum_of_real_solutions_eq_zero_l454_454295

open Real

theorem sum_of_real_solutions_eq_zero (b : ℝ) (hb : b > 1) : 
  let solutions := {x : ℝ | (sqrt (b - sqrt (b^2 + x)) = 2 * x)} in
  ∑ x in solutions.to_finset, x = 0 :=
sorry

end sum_of_real_solutions_eq_zero_l454_454295


namespace problem1_problem2_l454_454632

-- Definitions according to the problem conditions
def parabola (p : ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def line (a b c : ℝ) := {xy : ℝ × ℝ | a * xy.1 + b * xy.2 + c = 0}

-- Problem (1): Prove that p is 2 given |AB| = 4sqrt15.
theorem problem1 (p : ℝ) (h1 : p > 0) (A B : ℝ × ℝ)
  (hA : A ∈ parabola p) (hB : B ∈ parabola p) (hL : A ∈ line 1 (-2) 1)
  (hL' : B ∈ line 1 (-2) 1) (hAB : dist A B = 4 * real.sqrt 15) :
  p = 2 :=
sorry

-- Problem (2): Prove that the minimum area of the triangle MNF is 12 - 8sqrt2.
theorem problem2 (M N : ℝ × ℝ) (p : ℝ) (h1 : p > 0)
  (hM : M ∈ parabola p) (hN : N ∈ parabola p)
  (F : ℝ × ℝ) (hF : is_focus F p)
  (hMF_NF : (M.1 - F.1) * (N.1 - F.1) + M.2 * N.2 = 0) :
  ∃ area : ℝ, (∀ a', triangle_area M N F a' → a' ≥ area) ∧ area = 12 - 8 * real.sqrt 2 :=
sorry

end problem1_problem2_l454_454632


namespace space_diagonal_of_prism_l454_454598

theorem space_diagonal_of_prism (l w h : ℝ) (hl : l = 2) (hw : w = 3) (hh : h = 4) :
  (l ^ 2 + w ^ 2 + h ^ 2).sqrt = Real.sqrt 29 :=
by
  rw [hl, hw, hh]
  sorry

end space_diagonal_of_prism_l454_454598


namespace f_is_monotonically_increasing_max_perimeter_of_triangle_ADC_l454_454618

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x - π / 3) * cos x + sin x * cos x + sqrt 3 * (sin x)^2

def intervals_monotonically_increasing : Set ℝ :=
  {I | ∃ k : ℤ, I = Icc (-(π / 12) + k * π) (5 * (π / 12) + k * π)}

theorem f_is_monotonically_increasing :
  ∀ (x ∈ intervals_monotonically_increasing), MonotoneOn f x := sorry

variables (AC AB AD : ℝ) (B : ℝ) (BC : ℝ) (angle_BAC : ℝ)

def triangle_ABC_conditions (AC : ℝ) (B : ℝ) (AB : ℝ) (AD : ℝ) : Prop :=
  AC = 4 * sqrt 3 ∧
  f B = sqrt 3 ∧
  AB = AD ∧
  0 < B ∧ B < π / 2

def perimeter_ADC : ℝ := AC + AD + BC

theorem max_perimeter_of_triangle_ADC :
  triangle_ABC_conditions AC B AB AD →
  ∀ (D : Point) (on_BC : D.on BC),
  perimeter_ADC = 8 + 4 * sqrt 3 := sorry

end f_is_monotonically_increasing_max_perimeter_of_triangle_ADC_l454_454618


namespace correct_statement_is_D_l454_454015

-- Define relevant polynomial expressions and their properties
noncomputable def polyA : Polynomial ℚ := Polynomial.C (-3^2) * X * Y
noncomputable def polyB : Polynomial ℚ := Polynomial.C 1 * X^2 + Polynomial.C 1 * X + Polynomial.C (-1)
noncomputable def polyC : Polynomial ℚ := Polynomial.C (2^3) * X^2 * Y
noncomputable def polyD : Polynomial ℚ := Polynomial.C 4 * X^2 + Polynomial.C (-3) * X + Polynomial.C 1

-- Define properties for statements A, B, C, and D
def statementA : Prop := ite (Polynomial.coeff polyA 1 1 = -3) True False
def statementB : Prop := ite (Polynomial.coeff polyB 0 0 = 1) True False
def statementC : Prop := ite (Polynomial.natDegree polyC = 6) True False
def statementD : Prop := Polynomial.degree polyD = 2 ∧ Polynomial.card_terms polyD = 3

-- Proof that among these, only statement D is correct
theorem correct_statement_is_D : statementD ∧ ¬statementA ∧ ¬statementB ∧ ¬statementC :=
by sorry

end correct_statement_is_D_l454_454015


namespace complex_evaluation_l454_454556

theorem complex_evaluation : 
  (3 * (i ^ 6) + (i ^ 33) + 5 = 2 + i) :=
by
  have i1 : i ^ 1 = i := sorry
  have i2 : i ^ 2 = -1 := sorry
  have i3 : i ^ 3 = -i := sorry
  have i4 : i ^ 4 = 1 := sorry
  have i6 : i ^ 6 = -1 := by
    calc i ^ 6 = (i ^ 4 * i ^ 2) : sorry
           ... = 1 * (-1)       : sorry
           ... = -1             : sorry
  have i33 : i ^ 33 = i := by
    calc i ^ 33 = i ^ (32+1)       : sorry
           ... = (i ^ 4) ^ 8 * i   : sorry
           ... = 1 * i             : sorry
           ... = i                 : sorry
  calc  3 * i ^ 6 + i ^ 33 + 5 
       = 3 * (-1) + i + 5 : sorry
    ... = -3 + i + 5      : sorry
    ... = 2 + i           : sorry

end complex_evaluation_l454_454556


namespace fixed_point_line_passes_through_range_of_t_l454_454925

-- Definition for first condition: Line with slope k (k ≠ 0)
variables {k : ℝ} (hk : k ≠ 0)

-- Definition for second condition: Ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Third condition: Intersections M and N
variables (M N : ℝ × ℝ)
variables (intersection_M : ellipse_C M.1 M.2)
variables (intersection_N : ellipse_C N.1 N.2)

-- Fourth condition: Slopes are k1 and k2
variables {k1 k2 : ℝ}
variables (hk1 : k1 = M.2 / M.1)
variables (hk2 : k2 = N.2 / N.1)

-- Fifth condition: Given equation 3(k1 + k2) = 8k
variables (h_eq : 3 * (k1 + k2) = 8 * k)

-- Proof for question 1: Line passes through a fixed point
theorem fixed_point_line_passes_through 
    (h_eq : 3 * (k1 + k2) = 8 * k) : 
    ∃ n : ℝ, n = 1/2 ∨ n = -1/2 := sorry

-- Additional conditions for question 2
variables {D : ℝ × ℝ} (hD : D = (1, 0))
variables (t : ℝ)
variables (area_ratio : (M.2 / N.2) = t)
variables (h_ineq : k^2 < 5 / 12)

-- Proof for question 2: Range for t
theorem range_of_t
    (hD : D = (1, 0))
    (area_ratio : (M.2 / N.2) = t)
    (h_ineq : k^2 < 5 / 12) : 
    2 < t ∧ t < 3 ∨ 1 / 3 < t ∧ t < 1 / 2 := sorry

end fixed_point_line_passes_through_range_of_t_l454_454925


namespace find_p_min_area_triangle_MNF_l454_454629

-- Definitions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def line (x y : ℝ) := x - 2 * y + 1 = 0
def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def focus (p : ℝ) := (p / 2, 0)
def dot_product (M N F : ℝ × ℝ) := (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2)

-- Part 1: Finding p
theorem find_p (p : ℝ) (h : p > 0) :
  (∃ A B : ℝ × ℝ, parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  distance A B = 4 * real.sqrt 15) → p = 2 :=
sorry

-- Part 2: Minimum area of triangle MNF
theorem min_area_triangle_MNF (M N : ℝ × ℝ) (h1 : parabola 2 M.1 M.2) (h2 : parabola 2 N.1 N.2) (h3 : dot_product M N (focus 2) = 0) :
  ∃ (area : ℝ), area = 12 - 8 * real.sqrt 2 :=
sorry

end find_p_min_area_triangle_MNF_l454_454629


namespace minimum_degree_f_eq_c_l454_454097

variable (f g : Polynomial ℝ) (c : ℝ)

theorem minimum_degree_f_eq_c (h_c_gt_zero : 0 < c) :
  (c >= 2 → ¬ ∃ (f g : Polynomial ℝ), (∀ x : ℝ, x^2 - c * x + 1 = f.eval x / g.eval x) 
    ∧ f.coeff 0 ≠ 0 ∧ g.coeff 0 ≠ 0 ∧ (∀ n, f.coeff n ≥ 0) ∧ (∀ n, g.coeff n ≥ 0)) ∧
  (c < 2 → ∃ (f g : Polynomial ℝ), (∀ x : ℝ, x^2 - c * x + 1 = f.eval x / g.eval x) 
    ∧ f.coeff 0 ≠ 0 ∧ g.coeff 0 ≠ 0 ∧ (∀ n, f.coeff n ≥ 0) ∧ (∀ n, g.coeff n ≥ 0) 
    ∧ f.degree = ⌈π / Real.arccos (c / 2)⌉) :=
by
  sorry

end minimum_degree_f_eq_c_l454_454097


namespace option_C_is_correct_l454_454811

theorem option_C_is_correct :
  (-3 - (-2) ≠ -5) ∧
  (-|(-1:ℝ)/3| + 1 ≠ 4/3) ∧
  (4 - 4 / 2 = 2) ∧
  (3^2 / 6 * (1/6) ≠ 9) :=
by
  -- Proof omitted
  sorry

end option_C_is_correct_l454_454811


namespace gcd_problem_l454_454240

theorem gcd_problem 
  (b : ℤ) 
  (hb_odd : b % 2 = 1) 
  (hb_multiples_of_8723 : ∃ (k : ℤ), b = 8723 * k) : 
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 15) = 3 := 
by 
  sorry

end gcd_problem_l454_454240


namespace negation_log2_property_l454_454427

theorem negation_log2_property :
  ¬(∃ x₀ : ℝ, Real.log x₀ / Real.log 2 ≤ 0) ↔ ∀ x : ℝ, Real.log x / Real.log 2 > 0 :=
by
  sorry

end negation_log2_property_l454_454427


namespace solve_for_m_l454_454300

theorem solve_for_m (x m : ℝ) (hx : 0 < x) (h_eq : m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) : m = 6 :=
sorry

end solve_for_m_l454_454300


namespace centroid_path_area_l454_454381

-- Define the points and circle
variables {A B C : Point}
variable  circle : Circle

-- Define the conditions: A and B are endpoints of the diameter of the circle
def diameter_AB (circle : Circle) (A B : Point) : Prop :=
  circle.diameter A B

variable  (h_diameter : diameter_AB circle A B)
variable  (AB_length : dist A B = 36)
variable  (C_on_circle : circle.contains C)

-- Prove the area of the region bounded by the centroid's path
theorem centroid_path_area (h_diameter : diameter_AB circle A B)
              (AB_length : dist A B = 36)
              (C_on_circle : circle.contains C) : 
              area (centroid_path circle A B C) = 36 * Real.pi := 
sorry

end centroid_path_area_l454_454381


namespace std_dev_samples_l454_454654

def sample_A := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def sample_B := [84, 86, 86, 88, 88, 88, 90, 90, 90, 90]

noncomputable def std_dev (l : List ℕ) :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  let variance := (l.map (λ x => (x - mean) * (x - mean))).sum / n
  variance.sqrt

theorem std_dev_samples :
  std_dev sample_A = std_dev sample_B := 
sorry

end std_dev_samples_l454_454654


namespace complete_square_b_l454_454013

theorem complete_square_b (a b x : ℝ) (h : x^2 + 6 * x - 3 = 0) : (x + a)^2 = b → b = 12 := by
  sorry

end complete_square_b_l454_454013


namespace min_b1_b2_l454_454543

-- Define the sequence recurrence relation
def sequence_recurrence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2011) / (1 + b (n + 1))

-- Problem statement: Prove the minimum value of b₁ + b₂ is 2012
theorem min_b1_b2 (b : ℕ → ℕ) (h : ∀ n ≥ 1, 0 < b n) (rec : sequence_recurrence b) :
  b 1 + b 2 ≥ 2012 :=
sorry

end min_b1_b2_l454_454543


namespace trader_profit_l454_454069

-- Definitions and conditions
def original_price (P : ℝ) := P
def discounted_price (P : ℝ) := 0.70 * P
def marked_up_price (P : ℝ) := 0.84 * P
def sale_price (P : ℝ) := 0.714 * P
def final_price (P : ℝ) := 1.2138 * P

-- Proof statement
theorem trader_profit (P : ℝ) : ((final_price P - original_price P) / original_price P) * 100 = 21.38 := by
  sorry

end trader_profit_l454_454069


namespace area_diff_of_IJKL_eq_zero_l454_454755

theorem area_diff_of_IJKL_eq_zero 
  (center : Point) 
  (parallel_sides : parallel AB EF) 
  (area_ABCD : area ABCD = 2016) 
  (area_EFGH : area EFGH < 2016)
  (IJ_on_side_ABC : ∀ I, I ∈ IJKL.vertices → I ∈ ABCD.sides)
  (EF_on_side_IJKL : ∀ E, E ∈ EFGH.vertices → E ∈ IJKL.sides) :
  (max_area IJKL - min_area IJKL) = 0 := 
sorry

end area_diff_of_IJKL_eq_zero_l454_454755


namespace cycle_exists_max_cycles_l454_454899

-- Defining the conditions
variables (Player : Type) [fintype Player] -- Finite type of Players
variable (matches : Player → Player → Prop) -- Relation: Player A beats Player B

noncomputable def at_least_one_win (P : Player) : Prop :=
  ∃ Q : Player, matches P Q

-- Problem Statements
theorem cycle_exists {n : ℕ} [h : fact (n ≥ 3)] (h₁ : ∀ P : Player, at_least_one_win matches P) 
  (h₂ : ∀ (P Q : Player), P ≠ Q → (matches P Q ∨ matches Q P))
  (h₃ : fin n ≃ Player) : 
∃ A B C : Player, matches A B ∧ matches B C ∧ matches C A :=
sorry

theorem max_cycles {n : ℕ} (h₁ : ∀ P : Player, at_least_one_win matches P) 
  (h₂ : ∀ i j : fin n, i ≠ j → (matches (h₃ i) (h₃ j) ∨ matches (h₃ j) (h₃ i))) 
  (h₃ : fin n ≃ Player) [fact (n ≥ 3)] :
  let total_cycles := (n - 1) * n * (n + 1) / 24 in 
  total_cycles =
    ((finset.univ : finset (fin n)).card.choose 3) - 
      (finset.univ.sum (λ i, i.val*(i.val-1)*(i.val-3)/2)) :=
sorry

end cycle_exists_max_cycles_l454_454899


namespace incenters_form_rectangle_l454_454228
open EuclideanGeometry

-- Definitions for the four concyclic points
variable {A B C D : Point}
variable [circular_points : Concyclic A B C D]

-- Definitions for the incenters
def incenter_ABC := incenter A B C
def incenter_ABD := incenter A B D
def incenter_BCD := incenter B C D
def incenter_ACD := incenter A C D

-- Define the four incenters
variable (I : Point) (hI : I = incenter_ABC)
variable (J : Point) (hJ : J = incenter_ABD)
variable (K : Point) (hK : K = incenter_BCD)
variable (L : Point) (hL : L = incenter_ACD)

-- Prove that IJKL forms a rectangle
theorem incenters_form_rectangle : rectangle I J K L := sorry

end incenters_form_rectangle_l454_454228


namespace find_k_l454_454210

theorem find_k : 
  ∀ (k : ℤ), 2^4 - 6 = 3^3 + k ↔ k = -17 :=
by sorry

end find_k_l454_454210


namespace terminal_side_of_minus_330_in_first_quadrant_l454_454016

def angle_quadrant (angle : ℤ) : ℕ :=
  let reduced_angle := ((angle % 360) + 360) % 360
  if reduced_angle < 90 then 1
  else if reduced_angle < 180 then 2
  else if reduced_angle < 270 then 3
  else 4

theorem terminal_side_of_minus_330_in_first_quadrant :
  angle_quadrant (-330) = 1 :=
by
  -- We need a proof to justify the theorem, so we leave it with 'sorry' as instructed.
  sorry

end terminal_side_of_minus_330_in_first_quadrant_l454_454016


namespace integer_solution_count_l454_454288

theorem integer_solution_count :
  {n : ℤ | n ≠ 0 ∧ (1 / |(n:ℚ)| ≥ 1 / 5)}.finite.card = 10 :=
by
  sorry

end integer_solution_count_l454_454288


namespace independence_testing_correctness_l454_454478

def independence_testing_statement_A (x y: Type) : Prop := 
  ∀ (x y : Type), ∃ (P : x × y → ℝ), ¬ is_linear P

def independence_testing_statement_B (x y : Type) : Prop := 
  ∀ (x y : Type), ∃ (P : x × y → ℝ), ¬ (certainty (P) = 1)

def independence_testing_statement_C (x y: Type) : Prop := 
  ∀ (P: x → Prop) (Q: y → Prop) (confidence : ℕ → ℕ → ℝ), 
  ¬ (confidence 100 99 = 0.99)

def independence_testing_statement_D (χ²: Type) : Prop := 
  ∀ ( χ²: Type) (k: χ²), 
  smaller_k_probability_error (k) > 0

theorem independence_testing_correctness (x y: Type) (χ² : Type): 
  (independence_testing_statement_A x y) ∧ 
  (independence_testing_statement_B x y) ∧ 
  (independence_testing_statement_C x y) ∧ 
  (independence_testing_statement_D χ²) := 
  by
    exact ⟨sorry, sorry, sorry, sorry⟩

end independence_testing_correctness_l454_454478


namespace find_possible_d_l454_454396

theorem find_possible_d (u v : ℝ) (h1 : (x^3 + u * x + v).is_root)
    (h2 : ((u + 3) * x + (v - 2)).is_root) :
    (d = -13.125 ∨ d = 39.448) :=
sorry

end find_possible_d_l454_454396


namespace standard_form_range_of_x_plus_y_l454_454254

open Real

-- Definition of the parametric equations for the curve C
def parametric_curve (φ : ℝ) : ℝ × ℝ :=
  (4 * cos φ, 3 * sin φ)

-- Proof Problem 1: Standard form of the ellipse
theorem standard_form (x y φ : ℝ) (h1 : x = 4 * cos φ) (h2 : y = 3 * sin φ) :
  (x ^ 2) / 16 + (y ^ 2) / 9 = 1 :=
by {
  rw [h1, h2],
  sorry  -- Proof goes here
}

-- Proof Problem 2: Range of x + y on the curve
theorem range_of_x_plus_y (x y φ : ℝ) (h1 : x = 4 * cos φ) (h2 : y = 3 * sin φ) :
  -5 ≤ x + y ∧ x + y ≤ 5 :=
by {
  rw [h1, h2],
  sorry  -- Proof goes here
}

end standard_form_range_of_x_plus_y_l454_454254


namespace number_of_trees_l454_454442

def playground_length : ℕ := 150
def playground_width : ℕ := 60
def distance_between_trees : ℕ := 10
def perimeter : ℕ := 2 * (playground_length + playground_width)
def total_positions : ℕ := perimeter / distance_between_trees
def trees_each_type : ℕ := total_positions / 2

theorem number_of_trees :
  trees_each_type = 21 ∧ 2 * trees_each_type = total_positions :=
by {
  have h1 : playground_length = 150 := rfl,
  have h2 : playground_width = 60 := rfl,
  have h3 : distance_between_trees = 10 := rfl,
  have h4 : perimeter = 420 := by simp [perimeter, h1, h2],
  have h5 : total_positions = 42 := by simp [total_positions, h3, h4],
  have h6 : trees_each_type = 21 := by simp [trees_each_type, h5],
  exact ⟨h6, by simp [h6]⟩
}

end number_of_trees_l454_454442


namespace integer_expression_l454_454720

theorem integer_expression (x y z : ℤ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) (n : ℕ) :
  (x^n / ((x-y)*(x-z)).toRat + y^n / ((y-x)*(y-z)).toRat + z^n / ((z-x)*(z-y)).toRat).denom = 1 := by
  sorry

end integer_expression_l454_454720


namespace circle_equation_line_tangent_to_circle_line_intersecting_circle_l454_454245

-- Part (I)
theorem circle_equation 
    (C : Type) [MetricSpace C] [InnerProductSpace ℝ C] 
    (centerC : C) (a b : ℝ) 
    (hx : ∥a - 2∥^2 + ∥b - 0∥^2 = 2) 
    : (∀ x y, ∥x - 2∥^2 + ∥y - 0∥^2 = 2) :=
sorry

-- Part (II)
theorem line_tangent_to_circle 
    (C : Type) [MetricSpace C] [InnerProductSpace ℝ C] 
    (centerC : C) (A B : ℝ) 
    (hx : ∥A - 2∥^2 + ∥B - 0∥^2 = 2)
    (hx : ∥1 - (-1)∥^2 + (2 - (-4)) - (2) = sqrt(2))
    : ∥A - B + C∥ = sqrt(2) :=
sorry

-- Part (III)
theorem line_intersecting_circle 
    (C : Type) [MetricSpace C] [InnerProductSpace ℝ C] 
    (p : C) (a₁ b₁ a₂ b₂ : ℝ) 
    (hx : ∥a₁ - 1∥ + ∥b₁ - 3∥ = 2)
    : (p : C → (a - b₁) = ∥(x = 1 or y = -4/3 x + 13/3) :=
sorry

end circle_equation_line_tangent_to_circle_line_intersecting_circle_l454_454245


namespace repeating_decimal_to_fraction_l454_454124

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454124


namespace green_sequins_per_row_correct_l454_454352

def total_blue_sequins : ℕ := 6 * 8
def total_purple_sequins : ℕ := 5 * 12
def total_green_sequins : ℕ := 162 - (total_blue_sequins + total_purple_sequins)
def green_sequins_per_row : ℕ := total_green_sequins / 9

theorem green_sequins_per_row_correct : green_sequins_per_row = 6 := 
by 
  sorry

end green_sequins_per_row_correct_l454_454352


namespace regular_pentagon_l454_454719

variables {A B C D E : Type*} -- vertex types
variables [convex_pentagon A B C D E] -- pentagon is convex
variables [equal_sides A B C D E] -- all sides are equal
variables (angle_A angle_B angle_C angle_D angle_E : ℝ) -- angles
variables [angle_order : angle_A ≥ angle_B ∧ angle_B ≥ angle_C ∧ angle_C ≥ angle_D ∧ angle_D ≥ angle_E] -- angle order

theorem regular_pentagon (h : convex_pentagon A B C D E) (equal_sides : equal_sides A B C D E)  
    (angle_order : angle_A ≥ angle_B ∧ angle_B ≥ angle_C ∧ angle_C ≥ angle_D ∧ angle_D ≥ angle_E) : 
    angle_A = angle_B ∧ angle_B = angle_C ∧ angle_C = angle_D ∧ angle_D = angle_E := 
by 
  sorry

end regular_pentagon_l454_454719


namespace inequality_proof_l454_454981

variable {m n : ℝ}

theorem inequality_proof (h1 : m < n) (h2 : n < 0) : (n / m + m / n > 2) := 
by
  sorry

end inequality_proof_l454_454981


namespace repeating_decimal_fraction_eq_l454_454166

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454166


namespace fraction_power_rule_example_l454_454008

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end fraction_power_rule_example_l454_454008


namespace graph_of_f_transformed_is_D_l454_454624

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then real.sqrt(4 - (x - 2) ^ 2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

-- Transformed function f(x + 2)
def f_transformed (x : ℝ) : ℝ := f (x + 2)

-- Graph D: Expected graph after shifting f(x) two units to the left
def graph_D (x : ℝ) : ℝ := f (x + 2)

theorem graph_of_f_transformed_is_D :
  ∀ x, f_transformed x = graph_D x :=
by
  intro x
  sorry  -- The proof would be provided here

end graph_of_f_transformed_is_D_l454_454624


namespace simplify_fraction_l454_454737

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454737


namespace cylinder_volume_from_rectangle_l454_454449

theorem cylinder_volume_from_rectangle (L W h : ℝ) (hL: L = 12) (hW: W = 8) (h₁: h = 12) (h₂: h = 8) :
  (π * (4 / π) ^ 2 * h₁ = 192 / π) ∨ (π * (6 / π) ^ 2 * h₂ = 288 / π) :=
by {
  sorry
}

end cylinder_volume_from_rectangle_l454_454449


namespace total_hours_worked_l454_454832

-- Define the hourly rate for the bricklayer
def hourly_rate_bricklayer : ℝ := 12

-- Define the hourly rate for the electrician
def hourly_rate_electrician : ℝ := 16

-- Define the total payment from the owner
def total_payment : ℝ := 1350

-- Define the expected total hours they worked together
def expected_hours_together : ℝ := 1350 / (hourly_rate_bricklayer + hourly_rate_electrician)

-- Proof statement
theorem total_hours_worked (H : total_payment = 1350) : 
  let hours := 1350 / (hourly_rate_bricklayer + hourly_rate_electrician)
  hours ≈ 48.21 := 
by
  -- Proof steps
  sorry

end total_hours_worked_l454_454832


namespace geometric_series_sum_l454_454806

theorem geometric_series_sum :
  let a_0 := 1 / 2
  let r := 1 / 2
  let n := 5
  let sum := ∑ i in Finset.range(n), a_0 * r^i
  sum = 31 / 32 :=
by
  sorry

end geometric_series_sum_l454_454806


namespace circle_center_radius_l454_454908

theorem circle_center_radius (x y : ℝ) :
  x^2 - 6*x + y^2 + 2*y - 9 = 0 ↔ (x-3)^2 + (y+1)^2 = 19 :=
sorry

end circle_center_radius_l454_454908


namespace square_root_of_latter_natural_number_l454_454001

-- Define the two consecutive natural numbers
def former_natural_number := 9
def latter_natural_number := former_natural_number + 1

-- The main proof goal
theorem square_root_of_latter_natural_number :
  sqrt latter_natural_number = 3 + sqrt 1 :=
by
  -- Insert the proof steps here
  sorry

end square_root_of_latter_natural_number_l454_454001


namespace max_sum_of_cubes_l454_454700

theorem max_sum_of_cubes (x : Fin 1970 → ℕ) (hpos : ∀ i, 0 < x i) (hsum : (∑ i, x i) = 2007) : 
  ∑ i, (x i)^3 ≤ 56841 :=
sorry

end max_sum_of_cubes_l454_454700


namespace trigonometric_identity_l454_454537

theorem trigonometric_identity :
  (cos (10 * (Real.pi / 180)) + Real.sqrt 3 * sin (10 * (Real.pi / 180))) / 
  Real.sqrt (1 - cos (80 * (Real.pi / 180))) = Real.sqrt 2 := 
sorry

end trigonometric_identity_l454_454537


namespace find_number_of_triples_l454_454199

theorem find_number_of_triples :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b + a * c = 56 ∧ a * b + b * c = 45) →
  (∃ (S : Finset (ℕ × ℕ × ℕ)), S.card = 3 ∧ ∀ (x : ℕ × ℕ × ℕ), x ∈ S → 
    (∃ (a b c : ℕ), x = (a, b, c) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b + a * c = 56 ∧ a * b + b * c = 45)) :=
begin
  sorry
end

end find_number_of_triples_l454_454199


namespace richard_touchdowns_per_game_l454_454872

theorem richard_touchdowns_per_game (Archie_record Richard_avg_touchdowns Richard_games_played : ℕ) (total_games : ℕ) : 
  Archie_record = 89 ∧ Richard_avg_touchdowns = 6 ∧ Richard_games_played = 14 ∧ total_games = 16 → 
  (Richard_avg_touchdowns * Richard_games_played + 2 * 3 > Archie_record) :=
by 
  intros h
  cases h with a ha
  cases ha with b hb
  cases hb with c d
  sorry

end richard_touchdowns_per_game_l454_454872


namespace balcony_more_than_orchestra_l454_454522

variables (O B : ℕ) (H1 : O + B = 380) (H2 : 12 * O + 8 * B = 3320)

theorem balcony_more_than_orchestra : B - O = 240 :=
by sorry

end balcony_more_than_orchestra_l454_454522


namespace fraction_rep_finite_geom_series_036_l454_454132

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454132


namespace valid_arrangements_count_l454_454312
noncomputable def arrangements_leader_vice_leader : ℕ := 288

theorem valid_arrangements_count 
  (total_people : ℕ) 
  (people_chosen : ℕ) 
  (possible_arrangements : ℕ) 
  (invalid_arrangements : ℕ)
  (leader : ℕ)
  (vice_leader : ℕ) 
  (valid_arrangements : ℕ) : 
  total_people = 6 → 
  people_chosen = 4 →
  leader = 1 →
  vice_leader = 2 → 
  possible_arrangements = 360 → 
  invalid_arrangements = 72 →
  valid_arrangements = possible_arrangements - invalid_arrangements →
  valid_arrangements = arrangements_leader_vice_leader :=
by
  intros total_people_eq people_chosen_eq leader_eq vice_leader_eq.
  intros possible_arrangements_eq invalid_arrangements_eq valid_arrangements_eq.
  sorry

end valid_arrangements_count_l454_454312


namespace grandmother_total_payment_l454_454081

theorem grandmother_total_payment
  (senior_discount : Real := 0.30)
  (children_discount : Real := 0.40)
  (num_seniors : Nat := 2)
  (num_children : Nat := 2)
  (num_regular : Nat := 2)
  (senior_ticket_price : Real := 7.50)
  (regular_ticket_price : Real := senior_ticket_price / (1 - senior_discount))
  (children_ticket_price : Real := regular_ticket_price * (1 - children_discount))
  : (num_seniors * senior_ticket_price + num_regular * regular_ticket_price + num_children * children_ticket_price) = 49.27 := 
by
  sorry

end grandmother_total_payment_l454_454081


namespace range_of_x1_l454_454623

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp (x + 1) - exp x + x^2 + 2 * m * (x - 1)

theorem range_of_x1 (m : ℝ) (hₘ : m > 0) (x1 x2 : ℝ) (h₁ : x1 + x2 = 1) (h₂ : f m x1 ≥ f m x2) :
  x1 ∈ Set.Ici (1 / 2) := 
sorry

end range_of_x1_l454_454623


namespace exists_partition_of_nats_l454_454795

noncomputable def partition_congruent_subsets (S : Set (Set ℕ)) : Prop :=
  (∀ A ∈ S, ∃ t : ℤ, (A = (λ n : ℕ, n + t) '' (A \ ∅))) ∧               -- A is congruent to some shift of the empty set
  (∀ A ∈ S, (Set.Infinite A)) ∧                                         -- Each subset A is infinite
  (∀ A B ∈ S, A ≠ B → A ∩ B = ∅)                                         -- Subsets are non-overlapping
  (Set.Infinite S)                                                      -- S is an infinite set

theorem exists_partition_of_nats : ∃ S : Set (Set ℕ), partition_congruent_subsets S :=
begin
    sorry -- Proof is beyond the scope
end

end exists_partition_of_nats_l454_454795


namespace rectangle_pair_81_squares_l454_454436

theorem rectangle_pair_81_squares:
  ∃ (w1 h1 w2 h2 : ℕ), 
    w1 * h1 + w2 * h2 = 81 
    ∧ w1 + h1 = w2 + h2
    ∧ ({(w1, h1), (w2, h2)} = {(3, 11), (6, 8)}) :=
by
  sorry

end rectangle_pair_81_squares_l454_454436


namespace password_configurations_count_l454_454501

-- Define the problem statement.
def valid_password_configurations (n : ℕ) : ℕ :=
  3^n + 2 + (-1)^n

-- The theorem we want to prove given the conditions.
theorem password_configurations_count (n : ℕ) (h : n ≥ 3) :
  valid_password_configurations n = 3^n + 2 + (-1)^n :=
by
  -- The actual proof would go here.
  sorry

end password_configurations_count_l454_454501


namespace repeating_decimal_eq_fraction_l454_454171

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454171


namespace find_a_l454_454920

noncomputable def roots_of_quadratic : Set ℝ :=
  { x | 2 * x^2 - 7 * x - 4 = 0 }

def B (a : ℝ) : Set ℝ :=
  { x | a * x = 1 }

theorem find_a (a : ℝ) (h : B a ⊆ roots_of_quadratic) :
  a = 0 ∨ a = -2 ∨ a = 1 / 4 :=
by {
  have h_b_empty : B a = ∅ ↔ a = 0,
  {
    -- Proof that B(a) is empty iff a = 0
    sorry
  },
  have h_b_not_empty : B a ≠ ∅ ↔ a ≠ 0,
  {
    -- Proof that B(a) is non-empty iff a ≠ 0
    sorry
  },
  cases (h_b_empty ⟨_⟩),
  {
    exact or.inl h_1,
  },
  {
    have h_a_nonzero : a ≠ 0,
    {
      sorry
    },
    have : ∃ x, B a = {x},
    {
      -- Proof that B(a) contains exactly one element when a ≠ 0
      sorry
    },
    cases this with x hx,
    have x_in_A : x ∈ roots_of_quadratic,
    {
      -- Proof that x is in A
      sorry
    },
    have a_satisfies : 1/a = x ∨ 1/a = y,
    {
      -- Proof that 1 / a must be one of the roots of the quadratic equation
      sorry
    },
    exact or.inr
    (
      or.inl
      (
        -- Proof that a = -2 or a = 1 / 4
        sorry
      )
    ),
  }
}

end find_a_l454_454920


namespace triangle_side_c_length_l454_454994

theorem triangle_side_c_length (A : ℝ) (b : ℝ) (area : ℝ) (c : ℝ) (hA : A = 60) (hb : b = 1) (harea : area = √3) :
  (c = 4) :=
sorry

end triangle_side_c_length_l454_454994


namespace marina_more_fudge_l454_454705

theorem marina_more_fudge (h1 : 4.5 * 16 = 72)
                          (h2 : 4 * 16 - 6 = 58) :
                          72 - 58 = 14 := by
  sorry

end marina_more_fudge_l454_454705


namespace simplify_fraction_l454_454744

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l454_454744


namespace max_gold_coins_l454_454816

theorem max_gold_coins (k : ℕ) (n : ℕ) (h : n = 13 * k + 3 ∧ n < 150) : n = 146 :=
by 
  sorry

end max_gold_coins_l454_454816


namespace dogs_not_eating_any_food_l454_454999

noncomputable def kennel : Type := { dogs : ℕ // dogs = 75 }
def watermelon_eaters : { dogs : ℕ // dogs = 18 }
def salmon_eaters : { dogs : ℕ // dogs = 58 }
def salmon_watermelon_eaters : { dogs : ℕ // dogs = 10 }
def chicken_eaters : { dogs : ℕ // dogs = 15 }
def chicken_watermelon_eaters : { dogs : ℕ // dogs = 4 }
def chicken_salmon_eaters : { dogs : ℕ // dogs = 8 }
def all_three_eaters : { dogs : ℕ // dogs = 2 }

theorem dogs_not_eating_any_food:
∃ dogs_not_eating_any_food : ℕ, 
dogs_not_eating_any_food = kennel.val - 73 ∧
watermelon_eaters.val = 18 ∧
salmon_eaters.val = 58 ∧
salmon_watermelon_eaters.val = 10 ∧
chicken_eaters.val = 15 ∧
chicken_watermelon_eaters.val = 4 ∧
chicken_salmon_eaters.val = 8 ∧
all_three_eaters.val = 2 :=
begin
  sorry
end

end dogs_not_eating_any_food_l454_454999


namespace taxi_ride_cost_l454_454857

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end taxi_ride_cost_l454_454857


namespace minimum_distinct_midpoints_l454_454885

-- Define the points and the conditions
variable {α : Type} [EuclideanSpace α] {n : ℕ} (A : Fin n → α)

-- Define a function to count distinct midpoints
def countMidpoints (A : Fin n → α) : ℕ :=
  (2 * n) - 3

-- The main theorem to prove
theorem minimum_distinct_midpoints (h : 2 ≤ n) : 
  ∃ k, k = countMidpoints A ∧ ∀ m, (m ≤ k) → m = k :=
by
  sorry

end minimum_distinct_midpoints_l454_454885


namespace simplify_fraction_l454_454748

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l454_454748


namespace colorings_formula_correct_l454_454448

noncomputable def valid_colorings (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ m) : ℕ :=
  ∑ k in finset.range (m + 1), (-1)^(m - k) * (nat.choose m k) * ((k-1)^n + (-1)^n * (k-1))

theorem colorings_formula_correct (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ m) :
  valid_colorings m n h1 h2 = ∑ k in finset.range (m + 1), (-1)^(m - k) * (nat.choose m k) * ((k-1)^n + (-1)^n * (k-1)) :=
sorry

end colorings_formula_correct_l454_454448


namespace percent_of_y_eq_l454_454469

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end percent_of_y_eq_l454_454469


namespace rice_difference_on_15th_and_first_10_squares_l454_454711

-- Definitions
def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ := 
  (3 * (3^n - 1)) / (3 - 1)

-- Theorem statement
theorem rice_difference_on_15th_and_first_10_squares :
  grains_on_square 15 - sum_first_n_squares 10 = 14260335 :=
by
  sorry

end rice_difference_on_15th_and_first_10_squares_l454_454711


namespace prove_altSum_2011_l454_454457

def altSum (n : ℕ) : ℤ :=
  ∑ k in Finset.range n.succ, (-1) ^ k

theorem prove_altSum_2011 : altSum 2011 = 1 := by
  sorry

end prove_altSum_2011_l454_454457


namespace center_of_symmetry_l454_454076

open Real

theorem center_of_symmetry:
  (∃ x: ℝ, ∃ y: ℝ, y = sin (2 * x) + cos (2 * x) ∧ y = 0) → 
  ∃ x: ℝ, x = (3 * π / 8) :=
by
  sorry

end center_of_symmetry_l454_454076


namespace radius_of_circumscribed_circle_area_of_triangle_l454_454502

noncomputable theory

open Real

-- Terms denoting conditions
variables {B C D : Point} (Omega : Circle) (M_BD M_CD : Point)
variables (R : ℝ)
variables (distance_M_BD_BD : ℝ) (distance_M_CD_CD : ℝ)

-- assume the distances are given as follows:
axiom H1 : distance M_BD B D = 3
axiom H2 : distance M_CD C D = 0.5

-- main theorem statements:
theorem radius_of_circumscribed_circle (h1 : R = 4) : 
  ∃ (radius : ℝ), radius = R := by
  use R
  exact h1

theorem area_of_triangle (A : ℝ) (h2 : A = 15 * Real.sqrt 15 / 4) : 
  ∃ (area : ℝ), area = A := by
  use A
  exact h2

-- sorry is added to indicate unproven statements as this is a statement only exercise

end radius_of_circumscribed_circle_area_of_triangle_l454_454502


namespace repeating_decimal_fraction_l454_454158

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454158


namespace intersection_M_N_l454_454269

namespace proof_problem

def M : Set ℤ := {x | x^2 - 3 * x - 4 ≤ 0}

def N : Set ℕ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {1, 2, 3} :=
by
  -- Proof will go here
  sorry

end proof_problem

end intersection_M_N_l454_454269


namespace min_value_64_l454_454366

noncomputable def min_value_expr (a b c d e f g h : ℝ) : ℝ :=
  (a * e) ^ 2 + (b * f) ^ 2 + (c * g) ^ 2 + (d * h) ^ 2

theorem min_value_64 
  (a b c d e f g h : ℝ) 
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  min_value_expr a b c d e f g h = 64 := 
sorry

end min_value_64_l454_454366


namespace larger_sphere_radius_l454_454797

-- Let r be the radius of the smaller spheres
variable (r : ℝ)

-- The centers of the 8 smaller spheres form a regular octagon
-- and the spheres are tangent to each other

-- Radius of the larger sphere that touches the plane and all smaller spheres
def R (r : ℝ) := ((2 + Real.sqrt 2) * r) / 2

-- Theorem: Correct radius of the larger sphere R
theorem larger_sphere_radius (r : ℝ) : 
  ∃ (R : ℝ), R = ((2 + Real.sqrt 2) * r) / 2 :=
begin
  use ((2 + Real.sqrt 2) * r) / 2,
  sorry
end

end larger_sphere_radius_l454_454797


namespace distance_from_pole_to_line_l454_454105

-- Definitions based on the problem condition
def polar_equation_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = Real.sqrt 3

-- The statement of the proof problem
theorem distance_from_pole_to_line (ρ θ : ℝ) (h : polar_equation_line ρ θ) :
  ρ = Real.sqrt 6 / 2 := sorry

end distance_from_pole_to_line_l454_454105


namespace repeating_decimal_eq_fraction_l454_454172

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454172


namespace time_to_cross_pole_l454_454071

-- Define given conditions
def train_speed_kmh : ℝ := 56
def train_length_m : ℝ := 140

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def train_speed_ms : ℝ := (train_speed_kmh * km_to_m) / hr_to_s

-- Define the expected time to cross the pole
def expected_time : ℝ := 9

-- Prove that the time to cross the pole is 9 seconds
theorem time_to_cross_pole : (train_length_m / train_speed_ms) = expected_time := by
  sorry

end time_to_cross_pole_l454_454071


namespace f_of_pi_over_6_l454_454444

-- Define g(x) as given in the problem
def g (x : Real) : Real := Real.sin (x - Real.pi / 6)

-- Define f(x) by shifting g(x) to the right by π/6 units
def f (x : Real) : Real := g (x - Real.pi / 6)

-- State the theorem
theorem f_of_pi_over_6 : f (Real.pi / 6) = -1 / 2 :=
by
  -- The actual proof steps would go here
  sorry

end f_of_pi_over_6_l454_454444


namespace intersection_empty_l454_454968

noncomputable def M : Set ℂ := {z : ℂ | ∃ (t : ℝ), t ≠ -1 ∧ t ≠ 0 ∧ z = (t / (1 + t)) + (1 + t) / t * Complex.I}

noncomputable def N : Set ℂ := {z : ℂ | ∃ (t : ℝ), |t| ≤ 1 ∧ z = sqrt 2 * (Real.cos (Real.arcsin t) + Real.cos (Real.arccos t) * Complex.I)}

theorem intersection_empty : ∀ z : ℂ, z ∈ M ∩ N → False :=
by
  sorry

end intersection_empty_l454_454968


namespace enthalpy_of_formation_C6H6_l454_454490

theorem enthalpy_of_formation_C6H6 :
  ∀ (enthalpy_C2H2 : ℝ) (enthalpy_C6H6 : ℝ)
  (enthalpy_C6H6_C6H6 : ℝ) (Hess_law : Prop),
  (enthalpy_C2H2 = 226.7) →
  (enthalpy_C6H6 = 631.1) →
  (enthalpy_C6H6_C6H6 = -33.9) →
  Hess_law →
  -- Using the given conditions to accumulate the enthalpy change for the formation of C6H6.
  ∃ Q_formation : ℝ, Q_formation = -82.9 := by
  sorry

end enthalpy_of_formation_C6H6_l454_454490


namespace point_not_in_second_quadrant_l454_454666

theorem point_not_in_second_quadrant (m : ℝ) : ¬ (m^2 + m ≤ 0 ∧ m - 1 ≥ 0) :=
by
  sorry

end point_not_in_second_quadrant_l454_454666


namespace hexagon_triangle_ratio_l454_454516

theorem hexagon_triangle_ratio (P : ℝ) (hexagon_perimeter : 6 * (P / 6) = P) (triangle_perimeter : 3 * (P / 3) = P) : 
  let A := π * (P / 6)^2,
      B := π * (P * real.sqrt 3 / 9)^2
  in A / B = (3 / 4) :=
by 
  sorry

end hexagon_triangle_ratio_l454_454516


namespace token_permutation_possible_l454_454415

-- Definitions of colors
inductive Color
| asparagus
| byzantium
| citrine

open Color

-- Definition of the board and the corresponding color assignment
def color_of_square (x y : ℕ) : Color :=
  match (x + y) % 3 with
  | 0 => asparagus
  | 1 => byzantium
  | 2 => citrine
  | _ => asparagus  -- This case is unreachable

-- The Lean statement for the problem
theorem token_permutation_possible :
  ∀ (n d : ℕ), 
  (∀ x y, 1 ≤ x ∧ x ≤ 3*n ∧ 1 ≤ y ∧ y ≤ 3*n → 
    (∃ color : Color, true)) → 
  (∀ x y, distance (x, y) (permuted_position (x, y)) ≤ d) →
  ∃ permuted_position : ℕ × ℕ → ℕ × ℕ,
  (∀ (x y : ℕ), distance (x, y) (permuted_position (x, y)) ≤ d + 2 ∧
    color_of_square x y = color_of_square (permuted_position x y)) :=
sorry

end token_permutation_possible_l454_454415


namespace paths_A_to_E_l454_454207

theorem paths_A_to_E : 
  let paths_A_B := 2 in
  let paths_B_C := 2 in
  let paths_C_D := 2 in
  let paths_D_E := 2 in
  let direct_path_A_E := 1 in
  paths_A_B * paths_B_C * paths_C_D * paths_D_E + direct_path_A_E = 17 :=
by
  let paths_A_B := 2
  let paths_B_C := 2
  let paths_C_D := 2
  let paths_D_E := 2
  let direct_path_A_E := 1
  sorry

end paths_A_to_E_l454_454207


namespace tangency_implies_concyclic_l454_454363

noncomputable def circles_concyclic (Γ1 Γ2 Γ3 Γ4 : Circle) (A B C D : Point) : Prop :=
tangent Γ1 Γ2 A ∧
tangent Γ2 Γ3 B ∧
tangent Γ3 Γ4 C ∧
tangent Γ4 Γ1 D →

concyclic A B C D

theorem tangency_implies_concyclic {Γ1 Γ2 Γ3 Γ4 : Circle} {A B C D : Point} :
  tangent Γ1 Γ2 A →
  tangent Γ2 Γ3 B →
  tangent Γ3 Γ4 C →
  tangent Γ4 Γ1 D →
  concyclic A B C D :=
sorry

end tangency_implies_concyclic_l454_454363


namespace question1_question2_l454_454619

-- Definitions for the given function and its conditions
def f (k x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1
def fp (k x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

-- Theorem for question 1
theorem question1 (k : ℝ) (h : 0 < k) : 
  ((∀ x ∈ Ioo (0 : ℝ) 4, fp k x < 0) → k = 1 / 3) :=
by
  sorry

-- Theorem for question 2
theorem question2 (k : ℝ) (h1 : 0 < k) : 
  ((∀ x ∈ Icc (0 : ℝ) 4, fp k x ≤ 0) → k ≤ 1 / 3) :=
by
  sorry

end question1_question2_l454_454619


namespace fraction_repeating_decimal_l454_454180

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454180


namespace sum_of_fraction_numerator_denominator_is_146_l454_454809

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  let x := a / (10^b - 1)
  x

theorem sum_of_fraction_numerator_denominator_is_146 :
  let x : ℚ := repeating_decimal_to_fraction 47 99
  x = 47 / 99 → ∃ n d : ℕ, (n + d = 146) ∧ (x.num = n ∧ x.denom = d) ∧ nat.coprime n d :=
by
  intro hx
  use 47
  use 99
  split
  · exact rfl
  · split
    · exact ⟨hx.symm, rfl⟩
    · exact nat.coprime_prime_of_prime_pow 47 (dec_trivial)

end sum_of_fraction_numerator_denominator_is_146_l454_454809


namespace x_represents_price_soccer_ball_l454_454443

variable (num_soccer_balls num_basketballs cost_soccer_balls cost_basketballs price_basketball price_soccer_ball : ℝ)

-- Conditions
def condition1 : Prop := num_soccer_balls = 2 * num_basketballs
def condition2 : Prop := cost_soccer_balls = 5000
def condition3 : Prop := cost_basketballs = 4000
def condition4 : Prop := price_basketball = price_soccer_ball + 30
def condition5 : Prop := 5000 / price_soccer_ball = 2 * (4000 / (price_soccer_ball + 30))

-- Proof Problem
theorem x_represents_price_soccer_ball 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (h5 : condition5) :
  price_soccer_ball = x :=
sorry

end x_represents_price_soccer_ball_l454_454443


namespace rectangle_area_l454_454768

-- Defining the conditions
def area_square : ℕ := 784
def side_square : ℕ := Math.sqrt area_square
def radius_circle : ℕ := side_square
def length_rectangle : ℕ := radius_circle / 4
def breadth_rectangle : ℕ := 5

-- The theorem to prove the area of the rectangle 
theorem rectangle_area : length_rectangle * breadth_rectangle = 35 := by
  -- We denote the required values corresponding to the given problem definitions
  have side_square_eq : side_square = 28 := 
    by norm_num; exact Math.sqrt 784
  have radius_circle_eq : radius_circle = 28 := 
    by rw [radius_circle, side_square_eq]
  have length_rectangle_eq : length_rectangle = 7 := 
    by rw [length_rectangle, radius_circle_eq]; norm_num
  have breadth_rectangle_eq : breadth_rectangle = 5 := 
    rfl -- Given as a condition

  -- Use all calculated and defined variables to prove the theorem
  calc
    length_rectangle * breadth_rectangle
    = 7 * 5 : by rw [length_rectangle_eq, breadth_rectangle_eq]
    ... = 35 : by norm_num

end rectangle_area_l454_454768


namespace roots_of_polynomial_l454_454698

theorem roots_of_polynomial (p q r s : ℝ) (h : (2 * (10 * x + 13)^2 * (5 * x + 8) * (x + 1) = 1 ∧ pq + rs ∈ ℝ)) : pq + rs = 329 / 100 :=
by
  intro p q r s h
  -- provide the appropriate proof if available
  sorry

end roots_of_polynomial_l454_454698


namespace number_of_integers_satisfying_inequality_l454_454285

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℝ) / |n| ≥ 1 / 5}.finite.card = 10 :=
by
  -- Proof goes here
  sorry

end number_of_integers_satisfying_inequality_l454_454285


namespace parameter_a_val_density_func_first_moment_second_moment_third_moment_fourth_moment_variance_skewness_kurtosis_l454_454570

/-- Define the cumulative distribution function -/
def F (x : ℝ) : ℝ :=
  if x < 2 then 0
  else if x ≤ 4 then 0.25 * (x - 2) ^ 2
  else 1

/-- Define the density function f -/
def f (x : ℝ) : ℝ :=
  if x ≤ 2 ∨ x ≥ 4 then 0
  else 0.5 * (x - 2)

/-- Prove the value of parameter a -/
theorem parameter_a_val : (0.25 : ℝ) = 0.25 := 
  sorry

/-- Prove the density function f(x) -/
theorem density_func :
  f = (λ x : ℝ, if x ≤ 2 ∨ x ≥ 4 then 0 else 0.5 * (x - 2)) :=
  sorry

/-- Prove the first moment M1 -/
theorem first_moment : (∫ x in 2..4, x * f x) = (10 / 3 : ℝ) :=
  sorry

/-- Prove the second moment M2 -/
theorem second_moment : (∫ x in 2..4, x^2 * f x) = (34 / 3 : ℝ) :=
  sorry

/-- Prove the third moment M3 -/
theorem third_moment : (∫ x in 2..4, x^3 * f x) = (39.2 : ℝ) :=
  sorry

/-- Prove the fourth moment M4 -/
theorem fourth_moment : (∫ x in 2..4, x^4 * f x) = (-67.2 : ℝ) :=
  sorry

/-- Prove the variance D(X) -/
theorem variance : 
  let M1 := (10 / 3 : ℝ),
      M2 := (34 / 3 : ℝ)
  in (M2 - M1^2) = (2 / 9 : ℝ) :=
  sorry

/-- Prove the skewness -/
theorem skewness : (-0.56 : ℝ) = -(2 * real.sqrt 2 / 5) := 
  sorry

/-- Prove the kurtosis -/
theorem kurtosis : (-0.6 : ℝ) = (-3 / 5) :=
  sorry

end parameter_a_val_density_func_first_moment_second_moment_third_moment_fourth_moment_variance_skewness_kurtosis_l454_454570


namespace defective_pencil_count_l454_454996

theorem defective_pencil_count (total_pencils : ℕ) (selected_pencils : ℕ) 
  (prob_non_defective : ℚ) (D N : ℕ) (h_total : total_pencils = 6) 
  (h_selected : selected_pencils = 3)
  (h_prob : prob_non_defective = 0.2)
  (h_sum : D + N = 6) 
  (h_comb : (N.choose 3 : ℚ) / (total_pencils.choose 3) = prob_non_defective) : 
  D = 2 := 
sorry

end defective_pencil_count_l454_454996


namespace side_length_of_T3_is_15_l454_454077

noncomputable def side_length_third_triangle (a : ℕ) (sum_perimeters : ℕ) : ℕ :=
  let a1 := 3 * a in
  let r  := 1 / 2 in
  have series_sum : a1 / (1 - r) = sum_perimeters, from sorry,
  a / 4

theorem side_length_of_T3_is_15 :
  let a := 60
  let sum_perimeters := 360 in
  side_length_third_triangle a sum_perimeters = 15 := sorry

end side_length_of_T3_is_15_l454_454077


namespace sum_of_powers_specific_sum_l454_454842

theorem sum_of_powers (n : ℕ) (x : ℝ) (h : 1 ≤ n) : 
  (x - 1) * (∑ i in (finset.range n).map (λ i, x^(n-1-i))) = x^n - 1 :=
by sorry

theorem specific_sum : 6^2023 + 6^2022 + 6^2021 + ... + 6 + 1 = (6^2024 - 1) / 5 :=
by {
  -- Use the generalized theorem with specific values
  have h := sum_of_powers 2024 6 (by linarith),
  calc  6^2023 + 6^2022 + 6^2021 + ... + 6 + 1 = (6 - 1) * (∑ i in finset.range 2024, 6^(2023 - i)) : by exact h
                                        ... = (6 - 1) * (6^2023 + 6^2022 + 6^2021 + ... + 6 + 1) : by rfl
                                        ... = (6^2024 - 1) : by sorry,
  -- Simplify to get the desired answer
  calc (6^2024 - 1) / 5 = (6^2024 - 1) / 5 : rfl
}

end sum_of_powers_specific_sum_l454_454842


namespace right_triangle_AD_expression_l454_454062

theorem right_triangle_AD_expression (ABC : Type) [right_triangle ABC]
  (A B C D : point ABC)
  (theta : ℝ)
  (s : ℝ) (c : ℝ)
  (h1 : angle BAC = 90)
  (h2 : angle BCA = theta)
  (h3 : on_line AC D)
  (h4 : bisects BD (angle ABC))
  (hs : s = sin theta)
  (hc : c = cos theta) :
  AD = s / (1 + s) :=
sorry

end right_triangle_AD_expression_l454_454062


namespace repeating_decimal_equals_fraction_l454_454135

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454135


namespace range_of_a_l454_454956

-- Define the quadratic function y = x^2 + 2(a-1)x + 5
def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 5

-- Define the condition that the function is increasing in the interval (4, +∞)
def increasing_interval (a : ℝ) : Prop := ∀ x > 4, ∀ y > x, quadratic_function a y > quadratic_function a x

theorem range_of_a (a : ℝ) : increasing_interval a → a ∈ set.Ici (-3) := 
sorry

end range_of_a_l454_454956


namespace check_scenario_possible_l454_454494

variable (α : ℝ) (n m : ℕ)
def difficult_questions (α m r : ℕ) : Prop := r ≥ (α * m)
def passing_students (α n t : ℕ) : Prop := t ≥ (α * n)
noncomputable def scenario_possible := 
  (difficult_questions α m ∧ passing_students α n -> α = 2 / 3) ∧ 
  (difficult_questions α m ∧ passing_students α n -> ¬(α = 7 / 10))

theorem check_scenario_possible (α : ℝ) (n m : ℕ) :
  scenario_possible α n m := 
by 
  unfold scenario_possible 
  unfold difficult_questions 
  unfold passing_students 
  intro α n m 
  split
  case left => 
    intros h1 h2
    sorry -- Proof required
  case right => 
    intros h1 h2
    sorry -- Proof required

end check_scenario_possible_l454_454494


namespace min_PM_PN_l454_454966

noncomputable def C1 (x y : ℝ) : Prop := (x + 6)^2 + (y - 5)^2 = 4
noncomputable def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

theorem min_PM_PN : ∀ (P M N : ℝ × ℝ),
  P.2 = 0 ∧ C1 M.1 M.2 ∧ C2 N.1 N.2 → (|P.1 - M.1| + (P.1 - N.1)^2 + (P.2 - N.2)^2).sqrt = 7 := by
  sorry

end min_PM_PN_l454_454966


namespace find_principal_and_rate_l454_454023

variables (P R : ℝ)

theorem find_principal_and_rate
  (h1 : 20 = P * R * 2 / 100)
  (h2 : 22 = P * ((1 + R / 100) ^ 2 - 1)) :
  P = 50 ∧ R = 20 :=
by
  sorry

end find_principal_and_rate_l454_454023


namespace fraction_equivalent_of_repeating_decimal_l454_454193

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454193


namespace num_squares_within_bounds_is_22_l454_454974

noncomputable def num_squares_within_bounds : ℕ :=
  let boundary_y := λ x : ℝ, -π * x + 10
  let lower_y := 0.1
  let max_x := 4.9
  (∑ i in finset.range (⌊max_x⌋.nat_abs + 1), 
    if boundary_y i ≥ lower_y then (⌊boundary_y i⌋.nat_abs) else 0)

theorem num_squares_within_bounds_is_22 :
  num_squares_within_bounds = 22 :=
by
  sorry

end num_squares_within_bounds_is_22_l454_454974


namespace mean_of_set_l454_454426

theorem mean_of_set (n : ℤ) (h_median : n + 7 = 14) : (n + (n + 4) + (n + 7) + (n + 10) + (n + 14)) / 5 = 14 := by
  sorry

end mean_of_set_l454_454426


namespace ac_le_bc_l454_454584

theorem ac_le_bc (a b c : ℝ) (h: a > b): ∃ c, ac * c ≤ bc * c := by
  sorry

end ac_le_bc_l454_454584


namespace sequence_k_eq_4_l454_454328

theorem sequence_k_eq_4 {a : ℕ → ℕ} (h1 : a 1 = 2) (h2 : ∀ m n, a (m + n) = a m * a n)
    (h3 : ∑ i in finset.range 10, a (4 + i) = 2^15 - 2^5) : 4 = 4 :=
by
  sorry

end sequence_k_eq_4_l454_454328


namespace fraction_rep_finite_geom_series_036_l454_454128

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454128


namespace repeating_decimal_fraction_l454_454159

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454159


namespace polynomial_coefficients_sum_l454_454216

theorem polynomial_coefficients_sum :
  let p := (1 - (2 : ℝ) * x)^(10 : ℕ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ)
    (h : p = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + 
      a_9 * x^9 + a_10 * x^(10 : ℕ)) : 
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 + 7 * a_7 + 8 * a_8 + 9 * a_9 + 10 * a_10 = 20 :=
by
  sorry

end polynomial_coefficients_sum_l454_454216


namespace sue_answer_is_106_l454_454521

-- Definitions based on conditions
def ben_step1 (x : ℕ) : ℕ := x * 3
def ben_step2 (x : ℕ) : ℕ := ben_step1 x + 2
def ben_step3 (x : ℕ) : ℕ := ben_step2 x * 2

def sue_step1 (y : ℕ) : ℕ := y + 3
def sue_step2 (y : ℕ) : ℕ := sue_step1 y - 2
def sue_step3 (y : ℕ) : ℕ := sue_step2 y * 2

-- Ben starts with the number 8
def ben_number : ℕ := 8

-- Ben gives the number to Sue
def given_to_sue : ℕ := ben_step3 ben_number

-- Lean statement to prove
theorem sue_answer_is_106 : sue_step3 given_to_sue = 106 :=
by
  sorry

end sue_answer_is_106_l454_454521


namespace find_i_plus_j_l454_454810

-- Define binomial coefficient helper function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of getting k heads in n flips
def prob_heads_in_n_flips (n k : ℕ) (h : ℚ) : ℚ := (binom n k : ℚ) * (h^k) * ((1 - h)^(n - k))

lemma probability_equal_condition (h : ℚ) (h_pos : 0 < h) (h_lt : h < 1) :
  prob_heads_in_n_flips 6 2 h = prob_heads_in_n_flips 6 3 h :=
by {
  unfold prob_heads_in_n_flips,
  calc
    (binom 6 2 : ℚ) * (h^2) * ((1 - h)^4) = 15 * (h^2) * ((1 - h)^4) : by norm_num,
    ...                                      = 20 * (h^3) * ((1 - h)^3) : sorry
}

lemma correct_heads_probability (h := (3 : ℚ) / 7) :
  prob_heads_in_n_flips 6 4 h = (19440 : ℚ) / 117649 :=
by {
  unfold prob_heads_in_n_flips,
  calc
    (binom 6 4 : ℚ) * (h^4) * ((1 - h)^2) = 15 * ((3/7)^4) * ((4/7)^2) : by norm_num,
    ...                                    = (19440 : ℚ) / 117649        : sorry
}

theorem find_i_plus_j :
  let i : ℕ := 19440,
      j : ℕ := 117649 in
  i + j = 137089 :=
by {
  rfl
}

end find_i_plus_j_l454_454810


namespace solve_equation_l454_454907

-- Definitions based on the conditions
def equation (a b c d : ℕ) : Prop :=
  2^a * 3^b - 5^c * 7^d = 1

def nonnegative_integers (a b c d : ℕ) : Prop := 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof to show the exact solutions
theorem solve_equation :
  (∃ (a b c d : ℕ), nonnegative_integers a b c d ∧ equation a b c d) ↔ 
  ( (1, 0, 0, 0) = (1, 0, 0, 0) ∨ (3, 0, 0, 1) = (3, 0, 0, 1) ∨ 
    (1, 1, 1, 0) = (1, 1, 1, 0) ∨ (2, 2, 1, 1) = (2, 2, 1, 1) ) := by
  sorry

end solve_equation_l454_454907


namespace longest_side_quadrilateral_l454_454777

theorem longest_side_quadrilateral :
  let A := (4, 0)
  let B := (3, 1)
  let C := (0, 1)
  let D := (0, 4)
  let dist (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  max (dist A B) (max (dist B C) (max (dist C D) (dist D A))) = 4 * real.sqrt 2 :=
by
  sorry

end longest_side_quadrilateral_l454_454777


namespace sin_alpha_plus_beta_value_l454_454218

theorem sin_alpha_plus_beta_value (α β : ℝ) (h1 : 0 < β ∧ β < α ∧ α < π / 2)
  (h2 : cos(α - β) = 12 / 13) (h3 : cos(2 * β) = 3 / 5) :
  sin (α + β) = 63 / 65 := 
sorry

end sin_alpha_plus_beta_value_l454_454218


namespace evaluate_sum_sine_l454_454687

noncomputable def a (n : ℕ) : ℤ := n * (2 * n + 1)

theorem evaluate_sum_sine : 
  ∣ ∑ 1 ≤ j < k ≤ 36, Real.sin (π / 6 * (a k - a j)) ∣ = 18 := 
sorry

end evaluate_sum_sine_l454_454687


namespace product_of_constants_l454_454546

theorem product_of_constants (t : ℤ) :
  (∃ (a b : ℤ), (x^2 + t*x - 24) = (x + a) * (x + b) ∧ t = a + b) →
  ∏ (t ∈ {23, 10, 5, 2, -2, -5, -10, -23}) = -5290000 :=
sorry

end product_of_constants_l454_454546


namespace cars_count_l454_454869

-- Define the number of cars as x
variable (x : ℕ)

-- The conditions for the problem
def condition1 := 3 * (x - 2)
def condition2 := 2 * x + 9

-- The main theorem stating that under the given conditions, x = 15
theorem cars_count : condition1 x = condition2 x → x = 15 := by
  sorry

end cars_count_l454_454869


namespace daily_practice_hours_l454_454510

-- Define the conditions as given in the problem
def total_hours_practiced_this_week : ℕ := 36
def total_days_in_week : ℕ := 7
def days_could_not_practice : ℕ := 1
def actual_days_practiced := total_days_in_week - days_could_not_practice

-- State the theorem including the question and the correct answer, given the conditions
theorem daily_practice_hours :
  total_hours_practiced_this_week / actual_days_practiced = 6 := 
by
  sorry

end daily_practice_hours_l454_454510


namespace fraction_power_rule_example_l454_454007

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end fraction_power_rule_example_l454_454007


namespace people_needed_to_mow_lawn_in_4_hours_l454_454214

-- Define the given constants and conditions
def n := 4
def t := 6
def c := n * t -- The total work that can be done in constant hours
def t' := 4

-- Define the new number of people required to complete the work in t' hours
def n' := c / t'

-- Define the problem statement
theorem people_needed_to_mow_lawn_in_4_hours : n' - n = 2 := 
sorry

end people_needed_to_mow_lawn_in_4_hours_l454_454214


namespace cars_count_l454_454870

-- Define the number of cars as x
variable (x : ℕ)

-- The conditions for the problem
def condition1 := 3 * (x - 2)
def condition2 := 2 * x + 9

-- The main theorem stating that under the given conditions, x = 15
theorem cars_count : condition1 x = condition2 x → x = 15 := by
  sorry

end cars_count_l454_454870


namespace no_such_function_l454_454896

noncomputable def no_such_function_exists : Prop :=
  ¬∃ f : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
    (∀ x : ℝ, f (f x) = (x - 1) * f x + 2)

-- Here's the theorem statement to be proved
theorem no_such_function : no_such_function_exists :=
sorry

end no_such_function_l454_454896


namespace ladder_base_l454_454043

theorem ladder_base (h : ℝ) (b : ℝ) (l : ℝ)
  (h_eq : h = 12) (l_eq : l = 15) : b = 9 :=
by
  have hypotenuse := l
  have height := h
  have base := b
  have pythagorean_theorem : height^2 + base^2 = hypotenuse^2 := by sorry 
  sorry

end ladder_base_l454_454043


namespace find_y_l454_454671

/-
  Definitions:
  Let α = ∠ABC = 70°
  Let β = ∠BAC = 50°
  Let γ = 180° - α - β = ∠BCA = 60°
  Let δ = 90° - γ = y
-/

def α : ℝ := 70
def β : ℝ := 50
def γ : ℝ := 180 - α - β
def δ : ℝ := 90 - γ

theorem find_y (h1 : α = 70) (h2 : β = 50) (h3 : γ = 180 - α - β) (h4 : δ = 90 - γ) : δ = 30 := 
by
  rw [h1, h2] at h3
  simp at h3
  rw [h3] at h4
  simp at h4
  exact h4

end find_y_l454_454671


namespace grid_max_sum_l454_454108

open Finset

-- Definitions for given conditions
variables {α : Type*} [LinearOrder α] [CommSemiring α]

-- Sum of a sequence of integers.
def max_sum (a b : Fin n → α) : α :=
  ∑ i, a i * b i

-- The main statement of the problem.
theorem grid_max_sum (n : ℕ) (grid : Fin n → Fin n → Bool)
  (a b : Fin n → ℕ) (h_a : ∀ i, a i = (univ.filter (λ j, grid i j)).card)
  (h_b : ∀ j, b j = (univ.filter (λ i, ¬ grid i j)).card) :
  max_sum a b ≤ 2 * (n * (n-1) * (n+1) / 6) :=
by
  sorry

end grid_max_sum_l454_454108


namespace find_m_n_l454_454559

theorem find_m_n (m n : ℕ) : 3 ^ m - 2 ^ n ∈ {-1, 5, 7} ↔ (m, n) = (0, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 2) ∨ (m, n) = (2, 1) :=
by
  sorry

end find_m_n_l454_454559


namespace repeating_decimal_fraction_l454_454157

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454157


namespace incenter_of_triangle_is_midpoint_of_tangent_points_l454_454608

-- Define the circle and triangle setup
variables {O O1 : Type} [circle O] [circle O1]
variables {A B C P Q : Type} [point A] [point B] [point C] [point P] [point Q]

-- Define tangent relationships
variable (tangent1 : tangency O1 O)
variable (tangent2 : tangent_to_side O1 A B P)
variable (tangent3 : tangent_to_side O1 A C Q)

-- Prove the main statement
theorem incenter_of_triangle_is_midpoint_of_tangent_points
  (circumcircle_triangle : circumcircle O (triangle A B C))
  (tangent_circles : tangent_circle O1 circumcircle_triangle)
  (tangent_AB : tangent_to_side O1 A B P)
  (tangent_AC : tangent_to_side O1 A C Q) :
  is_incenter (midpoint P Q) (triangle A B C) := 
begin
  -- Proof omitted
  sorry,
end

end incenter_of_triangle_is_midpoint_of_tangent_points_l454_454608


namespace matrix_eqn_l454_454685

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![\[1, 0, 3\], \[0, 1, 0\], \[3, 0, 1\]]

def I : Matrix (Fin 3) (Fin 3) ℤ :=
  1

def ZeroMatrix : Matrix (Fin 3) (Fin 3) ℤ :=
  0

theorem matrix_eqn :
  let p := -3
  let q := -6
  let r := 8 in
  B^3 + p • B^2 + q • B + r • I = ZeroMatrix :=
sorry

end matrix_eqn_l454_454685


namespace gcd_between_35_and_7_l454_454422

theorem gcd_between_35_and_7 {n : ℕ} (h1 : 65 < n) (h2 : n < 75) (h3 : gcd 35 n = 7) : n = 70 := 
sorry

end gcd_between_35_and_7_l454_454422


namespace exists_non_intersecting_side_l454_454350

theorem exists_non_intersecting_side (n : ℕ) (P : Point) (polygon : ConvexPolygon) (h : polygon.sides = 2 * n) (P_in_polygon : P ∈ polygon) :
  ∃ side, ∀ vertex, (line_through vertex P).interior ∩ side.interior = ∅ :=
sorry

end exists_non_intersecting_side_l454_454350


namespace percentage_increase_purchase_l454_454431

theorem percentage_increase_purchase :
  let original_price_book := 300
  let new_price_book := 480
  let original_price_album := 15
  let new_price_album := 20
  let original_price_poster := 5
  let new_price_poster := 10
  let total_original_price := original_price_book + original_price_album + original_price_poster
  let total_new_price := new_price_book + new_price_album + new_price_poster
  let total_increased_price := total_new_price - total_original_price
  let percentage_increase := (total_increased_price / total_original_price.toFloat) * 100
  percentage_increase = 59.375 := 
begin 
  intros,
  sorry 
end

end percentage_increase_purchase_l454_454431


namespace find_second_interest_rate_l454_454498

theorem find_second_interest_rate (
  initial_investment : ℝ := 20000,
  first_interest_rate : ℝ := 0.08,
  final_amount : ℝ := 21242,
  period_fraction : ℝ := 1/4)
  (h1 : initial_investment * (1 + first_interest_rate * period_fraction) = 20400)
  (s : ℝ)
  (h2 : 20400 * (1 + s * period_fraction / 100) = final_amount)
  : s = 16.549 :=
sorry

end find_second_interest_rate_l454_454498


namespace sheets_borrowed_l454_454971

theorem sheets_borrowed (total_pages : ℕ) (total_sheets : ℕ) (average_remaining : ℚ)
  (h1 : total_pages = 50)
  (h2 : total_sheets = 25)
  (h3 : average_remaining = 19) :
  ∃ n : ℕ, n = 13 :=
by
  use 13
  sorry

end sheets_borrowed_l454_454971


namespace total_reams_of_paper_l454_454970

def reams_for_haley : ℕ := 2
def reams_for_sister : ℕ := 3

theorem total_reams_of_paper : reams_for_haley + reams_for_sister = 5 := by
  sorry

end total_reams_of_paper_l454_454970


namespace elizabeth_net_profit_l454_454555

noncomputable section

def net_profit : ℝ :=
  let cost_bag_1 := 2.5
  let cost_bag_2 := 3.5
  let total_cost := 10 * cost_bag_1 + 10 * cost_bag_2
  let selling_price := 6.0
  let sold_bags_1_no_discount := 7 * selling_price
  let sold_bags_2_no_discount := 8 * selling_price
  let discount_1 := 0.2
  let discount_2 := 0.3
  let discounted_price_1 := selling_price * (1 - discount_1)
  let discounted_price_2 := selling_price * (1 - discount_2)
  let sold_bags_1_with_discount := 3 * discounted_price_1
  let sold_bags_2_with_discount := 2 * discounted_price_2
  let total_revenue := sold_bags_1_no_discount + sold_bags_2_no_discount + sold_bags_1_with_discount + sold_bags_2_with_discount
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.8 := by
  sorry

end elizabeth_net_profit_l454_454555


namespace sum_of_squares_distances_leq_9_l454_454923

theorem sum_of_squares_distances_leq_9
  (points : Fin 1997 → ℝ × ℝ)
  (center : points 0 = (0, 0))
  (in_circle : ∀ i, (points i).fst^2 + (points i).snd^2 ≤ 1) :
  let distances := λ i : Fin 1997, 
    let others := {j : Fin 1997 | j ≠ i} in 
    let d := λ j, (points i).fst - (points j).fst ^ 2 + (points i).snd - (points j).snd ^ 2 in 
    Inf (d '' others) in  
  ∑ i, distances i ^ 2 ≤ 9 :=
begin
  sorry
end

end sum_of_squares_distances_leq_9_l454_454923


namespace simplify_expression_l454_454729

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l454_454729


namespace square_diff_proof_l454_454252

variable (y : ℤ)
hypothesis (h : y^2 = 1849)

theorem square_diff_proof (h : y^2 = 1849) : (y + 2) * (y - 2) = 1845 :=
by
  -- Proof here
  sorry

end square_diff_proof_l454_454252


namespace proof_strictly_increasing_sequence_l454_454406

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end proof_strictly_increasing_sequence_l454_454406


namespace min_value_of_function_l454_454770

theorem min_value_of_function : 
  ∃ x > 2, ∀ y > 2, (y + 1 / (y - 2)) ≥ 4 ∧ (x + 1 / (x - 2)) = 4 := 
by sorry

end min_value_of_function_l454_454770


namespace geometry_problem_l454_454674

/-- 
Proof problem equivalent to computing the area of the locus 
where the sum of distances from any point P to the sides of triangle ABC 
is less than or equal to 12 when the sides of the triangle are given.
-/
theorem geometry_problem :
  ∀ (BC CA AB : ℕ), BC = 3 → CA = 4 → AB = 5 →
  let f (P : Type _) := sorry in -- Explicit definition of f(P) not provided; it's complex sum-difference
  let area_of_locus := by sorry in -- Placeholder for actual calculation of area
  100 * 858 + 1 = 85801 :=
by
  intros BC CA AB hBC hCA hAB f area_of_locus
  rw [hBC, hCA, hAB]
  exact sorry

end geometry_problem_l454_454674


namespace option_A_option_B_option_D_l454_454641

-- Definitions for vectors.
noncomputable def a : ℝ × ℝ × ℝ := (1, 1, -1)
noncomputable def b : ℝ × ℝ × ℝ := (2, -1, 0)
noncomputable def c : ℝ × ℝ × ℝ := (0, 1, -2)

-- Dot product of two vectors.
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Proof that a • (b + c) = 4
theorem option_A : dot_product a (b.1 + c.1, b.2 + c.2, b.3 + c.3) = 4 := sorry

-- Proof that (a - b) • (b - c) = -8
theorem option_B : dot_product (a.1 - b.1, a.2 - b.2, a.3 - b.3) (b.1 - c.1, b.2 - c.2, b.3 - c.3) = -8 := sorry

-- Proof that if (a + λb) ⊥ c, then λ = 3.
theorem option_D (λ : ℝ) (h : dot_product (a.1 + λ * b.1, a.2 + λ * b.2, a.3 + λ * b.3) c = 0) : λ = 3 := sorry

end option_A_option_B_option_D_l454_454641


namespace moles_of_HCl_combined_l454_454911

/-- Prove the number of moles of Hydrochloric acid combined is 1, given that 
1 mole of Sodium hydroxide and some moles of Hydrochloric acid react to produce 
1 mole of Water, based on the balanced chemical equation: NaOH + HCl → NaCl + H2O -/
theorem moles_of_HCl_combined (moles_NaOH : ℕ) (moles_HCl : ℕ) (moles_H2O : ℕ)
  (h1 : moles_NaOH = 1) (h2 : moles_H2O = 1) 
  (balanced_eq : moles_NaOH = moles_HCl ∧ moles_HCl = moles_H2O) : 
  moles_HCl = 1 :=
by
  sorry

end moles_of_HCl_combined_l454_454911


namespace prob_at_least_two_diamonds_or_aces_in_three_draws_l454_454834

noncomputable def prob_at_least_two_diamonds_or_aces: ℚ :=
  580 / 2197

def cards_drawn (draws: ℕ) : Prop :=
  draws = 3

def cards_either_diamonds_or_aces: ℕ :=
  16

theorem prob_at_least_two_diamonds_or_aces_in_three_draws:
  cards_drawn 3 →
  cards_either_diamonds_or_aces = 16 →
  prob_at_least_two_diamonds_or_aces = 580 / 2197 :=
  by
  intros
  sorry

end prob_at_least_two_diamonds_or_aces_in_three_draws_l454_454834


namespace units_digit_199_factorial_l454_454012

theorem units_digit_199_factorial : (199.factorial % 10) = 0 := by
  sorry

end units_digit_199_factorial_l454_454012


namespace number_of_integers_satisfying_inequality_l454_454278

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℚ) / |n| ≥ 1 / 5}.to_finset.card = 10 :=
by {
  sorry
}

end number_of_integers_satisfying_inequality_l454_454278


namespace probability_equal_white_black_probability_white_ge_black_l454_454048

/-- Part (a) -/
theorem probability_equal_white_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (2 * m) / (n + m)) := 
  sorry

/-- Part (b) -/
theorem probability_white_ge_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (n - m + 1) / (n + 1)) := 
  sorry

end probability_equal_white_black_probability_white_ge_black_l454_454048


namespace catch_up_time_l454_454388

noncomputable def velocity_A (t : ℝ) : ℝ := 3 * t^2 + 1
noncomputable def velocity_B (t : ℝ) : ℝ := 10 * t

noncomputable def distance_A (t : ℝ) : ℝ := ∫ s in 0..t, velocity_A s
noncomputable def distance_B (t : ℝ) : ℝ := 5 + ∫ s in 0..t, velocity_B s

theorem catch_up_time :
  ∃ t : ℝ, t ≥ 0 ∧ distance_A t = distance_B t ∧ t = 5 :=
by
  sorry

end catch_up_time_l454_454388


namespace intersection_M_N_l454_454270

namespace proof_problem

def M : Set ℤ := {x | x^2 - 3 * x - 4 ≤ 0}

def N : Set ℕ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {1, 2, 3} :=
by
  -- Proof will go here
  sorry

end proof_problem

end intersection_M_N_l454_454270


namespace constant_term_in_expansion_l454_454881

theorem constant_term_in_expansion (n k : ℕ) (x : ℝ) (choose : ℕ → ℕ → ℕ):
  (choose 12 3) * (6 ^ 3) = 47520 :=
by
  sorry

end constant_term_in_expansion_l454_454881


namespace clock_hand_positions_l454_454540

theorem clock_hand_positions : ∃ n : ℕ, n = 143 ∧ 
  (∀ t : ℝ, let hour_pos := t / 12
            let min_pos := t
            let switched_hour_pos := t
            let switched_min_pos := t / 12
            hour_pos = switched_min_pos ∧ min_pos = switched_hour_pos ↔
            ∃ k : ℤ, t = k / 11) :=
by sorry

end clock_hand_positions_l454_454540


namespace ladder_base_distance_l454_454046

theorem ladder_base_distance (h : real) (L : real) (H : real) (B : real) 
  (h_eq : h = 12) (L_eq : L = 15) (h_sq_plus_b_sq : h^2 + B^2 = L^2) : 
  B = 9 :=
by
  rw [h_eq, L_eq] at h_sq_plus_b_sq
  sorry

end ladder_base_distance_l454_454046


namespace range_of_function_l454_454774

theorem range_of_function (y : ℝ) (t: ℝ) (x : ℝ) (h_t : t = x^2 - 1) (h_domain : t ∈ Set.Ici (-1)) :
  ∃ (y_set : Set ℝ), ∀ y ∈ y_set, y = (1/3)^t ∧ y_set = Set.Ioo 0 3 ∨ y_set = Set.Icc 0 3 := by
  sorry

end range_of_function_l454_454774


namespace rice_uniformity_l454_454791

theorem rice_uniformity (sA sB : ℝ) (hA : sA^2 = 11) (hB : sB^2 = 3.4) : (sA^2 > sB^2) -> "Type B rice is more uniform in tillering than type A rice" :=
by {
  sorry  
}

end rice_uniformity_l454_454791


namespace trajectory_equation_find_x0_value_l454_454230

-- Given conditions
variables {H P Q M A B T D E : Point}
variables {a b x y x0 : ℝ}
variables (k : ℝ) (trajectory : Set Point)
variables (line_l : Point → ℝ → Line)
variables (is_midpoint : Point → Point → Point → Prop)
variables (point_on_y_axis : Point → Prop)
variables (is_perpendicular : Vector → Vector → Prop)
variables (on_line : Point → Line → Prop)
variables (line_intersect : Line → Set Point → Set Point)

-- Definition of the conditions
def point_H := H = Point.mk (-6) 0
def point_P_on_y_axis := ∀ b : ℝ, P = Point.mk 0 b ∧ point_on_y_axis P
def point_Q := ∀ a : ℝ, Q = Point.mk a 0 ∧ a > 0
def perpendicular_vectors := is_perpendicular (Vector.mk 6 b) (Vector.mk a (-b))
def point_M := ∀ x y, M = Point.mk x y
def M_on_PQ := on_line M (Line.mk P Q ∧ Vector.mk x (y-b) = 2 * (Vector.mk (a-x) (-y)))
def line_intersects_trajectory := ∀ k : ℝ, ∃ A B : Point, on_line A (line_l T k) ∧ on_line B (line_l T k) ∧ (trajectory C).mem A ∧ (trajectory C).mem B
def midpoint_D := is_midpoint A B D
def perpendicular_bisector_AB := ∃ E : Point, E = Point.mk x0 0 ∧ Line.mk D E ∧ Vector.mk (x0 - x_d) (-y_d/k) = 2 * sqrt 3 * (Vector.mk (x1 - x2) (y1 - y2))

-- Proving the correct answers
theorem trajectory_equation :
  (∀ P, point_P_on_y_axis P → P ∈ trajectory → y^2 = x) :=
sorry

theorem find_x0_value :
  point_H ∧ point_P_on_y_axis P ∧ point_Q Q ∧ perpendicular_vectors → 
  line_intersects_trajectory line_l k → 
  midpoint_D → perpendicular_bisector_AB → 
  x0 = 5 / 3 :=
sorry

end trajectory_equation_find_x0_value_l454_454230


namespace pyramid_height_is_correct_l454_454852

noncomputable def pyramid_height (perimeter : ℝ) (apex_distance : ℝ) : ℝ :=
  let side_length := perimeter / 4
  let half_diagonal := side_length * Real.sqrt 2 / 2
  Real.sqrt (apex_distance ^ 2 - half_diagonal ^ 2)

theorem pyramid_height_is_correct :
  pyramid_height 40 15 = 5 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_is_correct_l454_454852


namespace area_of_region_eq_30_l454_454969

def regionArea (x y : ℝ) : Prop :=
  (0 ≤ x) ∧ (0 ≤ y) ∧ (5 * x + 12 * y ≤ 60)

theorem area_of_region_eq_30 : (∃ x y : ℝ, (regionArea x y) ∧ (x = 12 ∧ y = 0 ∨ x = 0 ∧ y = 5 ∨ x = 0 ∧ y = 0) ∧ 
  (area = (1 / 2) * |12 * (5 - 0) + 0 * (0 - 0) + 0 * (0 - 5)|)) → area = 30 :=
by
  sorry

end area_of_region_eq_30_l454_454969


namespace plane_divides_diagonal_ratio_l454_454053

/--
In a cuboid structure, where \(A, B, C, D\) are vertices originating from point \(A\),
and \(A'\) is the point diagonally opposite to \(A\) within the cuboid,
the plane passing through the points \(B, C, D\) divides the diagonal \(A A'\) in the ratio 1:2.
-/
theorem plane_divides_diagonal_ratio (A B C D A' : Point) (cuboid_structure : Cuboid A B C D A')
  (diagonal : Line A A')
  (plane : Plane B C D)
  (intersection_point : Point)
  (h : intersection_point ∈ diagonal ∧ intersection_point ∈ plane) :
  ratio A intersection_point A' = 1 / 2 :=
sorry

end plane_divides_diagonal_ratio_l454_454053


namespace value_of_a_plus_b_l454_454620

variable {F : Type} [Field F]

theorem value_of_a_plus_b (a b : F) (h1 : ∀ x, x ≠ 0 → a + b / x = 2 ↔ x = -2)
                                      (h2 : ∀ x, x ≠ 0 → a + b / x = 6 ↔ x = -6) :
  a + b = 20 :=
sorry

end value_of_a_plus_b_l454_454620


namespace monotonic_exponential_decreasing_l454_454588

variable (a : ℝ) (f : ℝ → ℝ)

theorem monotonic_exponential_decreasing {m n : ℝ}
  (h0 : a = (Real.sqrt 5 - 1) / 2)
  (h1 : ∀ x, f x = a^x)
  (h2 : 0 < a ∧ a < 1)
  (h3 : f m > f n) :
  m < n :=
sorry

end monotonic_exponential_decreasing_l454_454588


namespace find_room_length_l454_454425

variable (width : ℝ) (cost rate : ℝ) (length : ℝ)

theorem find_room_length (h_width : width = 4.75)
  (h_cost : cost = 34200)
  (h_rate : rate = 900)
  (h_area : cost / rate = length * width) :
  length = 8 :=
sorry

end find_room_length_l454_454425


namespace tan_alpha_l454_454982

theorem tan_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 1 / 5) : Real.tan α = -2 / 3 :=
by
  sorry

end tan_alpha_l454_454982


namespace fish_shop_max_people_l454_454658

theorem fish_shop_max_people :
  let n : ℕ := 28
  in ∃ k : ℕ, k = 2^n - n ∧ 
    ∀ (people : Finset (Fin n → bool)),
      people.card = k → 
      ∀ (p1 p2 ∈ people), p1 ≠ p2 → 
      ∃ i : Fin n, p1 i ≠ p2 i :=
by
  sorry

end fish_shop_max_people_l454_454658


namespace mistaken_fraction_l454_454313

theorem mistaken_fraction (n correct_result student_result : ℕ) (h1 : n = 384)
  (h2 : correct_result = (5 * n) / 16) (h3 : student_result = correct_result + 200) : 
  (student_result / n : ℚ) = 5 / 6 :=
by
  sorry

end mistaken_fraction_l454_454313


namespace solve_abs_eq_real_l454_454648

theorem solve_abs_eq_real (x z : ℝ) (hx : 3 * x = real.exp (2 * real.log z)) :
  |3 * x - 2 * real.log z| = 3 * x + 2 * real.log z → x = 0 ∧ z = 1 :=
by
  sorry

end solve_abs_eq_real_l454_454648


namespace M_inter_N_eq_123_l454_454272

def M : Set ℤ := {x | x^2 - 3 * x - 4 ≤ 0}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem M_inter_N_eq_123 : M ∩ N = {1, 2, 3} := sorry

end M_inter_N_eq_123_l454_454272


namespace sum_powers_of_minus_one_l454_454459

/-- 
  Prove that the sum of powers of -1 from 1 to 2011 equals -1:
  ∑_{i=1}^{2011} (-1)^i = -1
-/
theorem sum_powers_of_minus_one:
  ∑ i in Finset.range 2011.succ, (-1) ^ (i + 1) = -1 := 
by
  sorry

end sum_powers_of_minus_one_l454_454459


namespace problem_statement_l454_454430

-- Define circle 1
def circle1 := ∀ x y : ℝ, x^2 + y^2 - 2 * x = 0

-- Define circle 2
def circle2 := ∀ x y : ℝ, x^2 + y^2 + 4 * y = 0

-- Define the positional relationship as intersect
def circles_intersect : Prop :=
  let center1 := (1, 0) in
  let center2 := (0, -2) in
  let r1 := 1 in
  let r2 := 2 in
  let d := Real.sqrt (1^2 + (-2)^2) in
  r2 - r1 < d ∧ d < r2 + r1

-- Problem statement
theorem problem_statement : circle1 ∧ circle2 → circles_intersect :=
by sorry

end problem_statement_l454_454430


namespace find_k_l454_454773

-- The function that computes the sum of the digits for the known form of the product (9 * 999...9) with k digits.
def sum_of_digits (k : ℕ) : ℕ :=
  8 + 9 * (k - 1) + 1

theorem find_k (k : ℕ) : sum_of_digits k = 2000 ↔ k = 222 := by
  sorry

end find_k_l454_454773


namespace rectangle_fraction_l454_454767

noncomputable def side_of_square : ℝ := Real.sqrt 900
noncomputable def radius_of_circle : ℝ := side_of_square
noncomputable def area_of_rectangle : ℝ := 120
noncomputable def breadth_of_rectangle : ℝ := 10
noncomputable def length_of_rectangle : ℝ := area_of_rectangle / breadth_of_rectangle
noncomputable def fraction : ℝ := length_of_rectangle / radius_of_circle

theorem rectangle_fraction :
  (length_of_rectangle / radius_of_circle) = (2 / 5) :=
by
  sorry

end rectangle_fraction_l454_454767


namespace conclusion_l454_454241

def f : ℕ → ℕ → ℕ := sorry  -- f will eventually satisfy the conditions

axiom f_initial : f 1 1 = 1

axiom f_nat_star (m n : ℕ) : f m n ∈ ℕ*

axiom f_arith_seq (m n : ℕ) : f m (n + 1) = f m n + 2

axiom f_geom_seq (m : ℕ) : f (m + 1) 1 = 2 * f m 1

theorem conclusion :
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 :=
by { sorry }

end conclusion_l454_454241


namespace sin_has_property_T_l454_454992

def property_T (f : ℝ → ℝ) : Prop :=
∃ x1 x2, x1 ≠ x2 ∧ (fderiv ℝ f x1).canonical_module_structure.smul_right 1 * (fderiv ℝ f x2).canonical_module_structure.smul_right 1 = -1

theorem sin_has_property_T : property_T sin := 
by sorry

end sin_has_property_T_l454_454992


namespace series_largest_prime_factor_of_111_l454_454518

def series := [368, 689, 836]  -- given sequence series

def div_condition (n : Nat) := 
  ∃ k : Nat, n = 111 * k

def largest_prime_factor (n : Nat) (p : Nat) := 
  Prime p ∧ ∀ q : Nat, Prime q → q ∣ n → q ≤ p

theorem series_largest_prime_factor_of_111 :
  largest_prime_factor 111 37 := 
by
  sorry

end series_largest_prime_factor_of_111_l454_454518


namespace solution_set_empty_l454_454781

-- Define the quadratic polynomial
def quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem that the solution set of the given inequality is empty
theorem solution_set_empty : ∀ x : ℝ, quadratic x < 0 → false :=
by
  intro x
  unfold quadratic
  sorry

end solution_set_empty_l454_454781


namespace quadrilateral_area_is_28_l454_454087

/-
Define the vertices of the quadrilateral
-/
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, -2⟩
def C : Point := ⟨4, 0⟩
def E : Point := ⟨4, 8⟩

/-
Define the function to calculate the area of a trapezoid based on given vertices.
-/
def trapezoid_area (AB DE height : ℝ) : ℝ := 1 / 2 * (AB + DE) * height

/-
Let's use the vertices to calculate the lengths and height, and assert the area.
-/
noncomputable def quadrilateral_area : ℝ :=
  let AB := abs(B.y - A.y)
  let DE := abs(E.y - C.y)
  let height := abs(C.x - A.x)
  trapezoid_area AB DE height

/-
State our theorem to be proved.
-/
theorem quadrilateral_area_is_28 :
  quadrilateral_area = 28 := by
  sorry

end quadrilateral_area_is_28_l454_454087


namespace max_value_l454_454694

-- Define the problem conditions
def conditions (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧
  x ≥ -1/2 ∧
  y ≥ -2 ∧
  z ≥ -3

-- Define the expression to be maximized
def expression (x y z : ℝ) : ℝ :=
  Real.sqrt (4 * x + 2) + Real.sqrt (4 * y + 8) + Real.sqrt (4 * z + 12)

-- Define the statement of the problem
theorem max_value (x y z : ℝ) (h : conditions x y z) : 
  expression x y z ≤ 3 * Real.sqrt 10 :=
sorry

end max_value_l454_454694


namespace log_eq_solve_for_x_l454_454024

theorem log_eq_solve_for_x (x : ℝ) (h : log 10 5 + log 10 (5 * x + 1) = log 10 (x + 5) + 1) : x = 3 :=
sorry

end log_eq_solve_for_x_l454_454024


namespace integer_solution_count_l454_454286

theorem integer_solution_count :
  {n : ℤ | n ≠ 0 ∧ (1 / |(n:ℚ)| ≥ 1 / 5)}.finite.card = 10 :=
by
  sorry

end integer_solution_count_l454_454286


namespace smaller_disk_radius_ratio_l454_454574

theorem smaller_disk_radius_ratio (r rs : ℝ)
  (h1 : ∀ d1 d2 d3 d4 : ℝ, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d4)
  (h2 : 2 * r = 2 * r)
  (h3 : 2 * r = 2 * r)
  (h4 : let inradius := (2 * r * real.sqrt 3) / 3 in
        rs + r = inradius) :
  rs / r = (2 * real.sqrt 3 - 3) / 3 :=
by
  sorry

end smaller_disk_radius_ratio_l454_454574


namespace scientific_notation_of_1400000000_l454_454830

theorem scientific_notation_of_1400000000 : ∃ n : ℕ, 1.4 * 10^n = 1400000000 :=
by {
  use 9,
  sorry
}

end scientific_notation_of_1400000000_l454_454830


namespace fraction_equivalent_of_repeating_decimal_l454_454190

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454190


namespace revenue_from_full_price_tickets_l454_454512

noncomputable def full_price_ticket_revenue (f h p : ℕ) : ℕ := f * p

theorem revenue_from_full_price_tickets (f h p : ℕ) (total_tickets total_revenue : ℕ) 
  (tickets_eq : f + h = total_tickets)
  (revenue_eq : f * p + h * (p / 2) = total_revenue) 
  (total_tickets_value : total_tickets = 180)
  (total_revenue_value : total_revenue = 2652) :
  full_price_ticket_revenue f h p = 984 :=
by {
  sorry
}

end revenue_from_full_price_tickets_l454_454512


namespace base_b_not_divisible_by_5_l454_454211

theorem base_b_not_divisible_by_5 (b : ℕ) : b = 4 ∨ b = 7 ∨ b = 8 → ¬ (5 ∣ (2 * b^2 * (b - 1))) :=
by
  sorry

end base_b_not_divisible_by_5_l454_454211


namespace simplify_fraction_l454_454736

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454736


namespace number_of_integers_satisfying_inequality_l454_454284

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℝ) / |n| ≥ 1 / 5}.finite.card = 10 :=
by
  -- Proof goes here
  sorry

end number_of_integers_satisfying_inequality_l454_454284


namespace exists_unique_pair_l454_454104

theorem exists_unique_pair (X : Set ℤ) :
  (∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n) :=
sorry

end exists_unique_pair_l454_454104


namespace area_under_curve_l454_454086

noncomputable def area_enclosed_by_curve : Real := 
  ∫ x in 1..5, (6 * x - x^2 - 5)

theorem area_under_curve :
  area_enclosed_by_curve = (32 / 3) := 
by
  sorry

end area_under_curve_l454_454086


namespace simplify_fraction_l454_454743

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454743


namespace min_value_a2_b2_l454_454247

theorem min_value_a2_b2 {a b : ℝ} 
  (h : 2 * a + b + 1 = 0) : (∃ (a b : ℝ), a^2 + b^2 = 1) ∧ 
  (∀ (a b : ℝ), h → a^2 + b^2 ≥ 1) :=
by 
  sorry

end min_value_a2_b2_l454_454247


namespace graphs_intersect_at_one_point_l454_454886

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x) / Real.log 2

theorem graphs_intersect_at_one_point : ∃! x, f x = g x :=
by {
  sorry
}

end graphs_intersect_at_one_point_l454_454886


namespace inequality_hold_l454_454262

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l454_454262


namespace repeating_decimal_fraction_l454_454808

theorem repeating_decimal_fraction :
  ( ∃ x : ℚ, x = 511734 / 99900 ∧ x = (((51 : ℚ) / 100) + 246 / 9990) ∨ sorry : ℚ) := 
  sorry

end repeating_decimal_fraction_l454_454808


namespace distance_between_foci_of_ellipse_l454_454615

theorem distance_between_foci_of_ellipse :
  ∀ (x y : ℝ), 25 * x^2 + 100 * x + 9 * y^2 - 36 * y = 225 → 
  let a := real.sqrt(40.11) in
  let b := real.sqrt(14.44) in
  let c := real.sqrt(a^2 - b^2) in
  2 * c = 10.134 :=
by
  intros x y h a b c
  sorry

end distance_between_foci_of_ellipse_l454_454615


namespace sum_of_solutions_eq_neg9_l454_454376

def f : ℝ → ℝ :=
  λ x, if x < -3 then 3 * x + 9 else -x^2 - 2 * x + 2

theorem sum_of_solutions_eq_neg9 :
  let solutions := {x : ℝ | f x = -6} in 
  ∑ x in solutions, x = -9 :=
by
  sorry

end sum_of_solutions_eq_neg9_l454_454376


namespace percent_equivalence_l454_454466

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end percent_equivalence_l454_454466


namespace a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l454_454937

theorem a_sq_greater_than_b_sq_neither_sufficient_nor_necessary 
  (a b : ℝ) : ¬ ((a^2 > b^2) → (a > b)) ∧  ¬ ((a > b) → (a^2 > b^2)) := sorry

end a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l454_454937


namespace find_m_l454_454616

theorem find_m (m : ℝ) (x : ℝ) (y : ℝ) (h_eq_parabola : y = m * x^2)
  (h_directrix : y = 1 / 8) : m = -2 :=
by
  sorry

end find_m_l454_454616


namespace no_such_prime_pair_l454_454114

open Prime

theorem no_such_prime_pair :
  ∀ (p q : ℕ), Prime p → Prime q → (p > 5) → (q > 5) →
  (p * q) ∣ ((5^p - 2^p) * (5^q - 2^q)) → false :=
by
  intros p q hp hq hp_gt5 hq_gt5 hdiv
  sorry

end no_such_prime_pair_l454_454114


namespace find_N_mod_1000_l454_454913

def sum_digits (n : ℕ) (base : ℕ) : ℕ :=
  (to_digits base n).sum

theorem find_N_mod_1000 :
  ∃ n : ℕ, sum_digits (sum_digits n 5) 7 ≥ 10 ∧ n % 1000 = 781 :=
sorry

end find_N_mod_1000_l454_454913


namespace leap_day_2040_monday_l454_454684

-- given conditions
def leap_day_1996_thursday : Prop := 
  day_of_week 1996 2 29 = "Thursday"

def leap_years_1996_to_2040 : List Nat := 
  [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]

def days_between_leap_days : Nat :=
  32 * 365 + 12 * 366  -- total number of days

def days_mod_7 : Nat :=
  days_between_leap_days % 7

-- proof to be constructed
theorem leap_day_2040_monday 
  (h1 : leap_day_1996_thursday) 
  (h2 : leap_years_1996_to_2040.length = 12) 
  (h3 : days_mod_7 = 4) : 
  day_of_week 2040 2 29 = "Monday" :=
  sorry

end leap_day_2040_monday_l454_454684


namespace break_even_handles_l454_454386

variables (X : ℕ)
constants (fixed_cost : ℝ) (variable_cost_per_handle : ℝ) (selling_price_per_handle : ℝ)

def break_even (fixed_cost variable_cost_per_handle selling_price_per_handle : ℝ) (X : ℕ) : Prop :=
  fixed_cost + variable_cost_per_handle * X = selling_price_per_handle * X

theorem break_even_handles :
  break_even 7640 0.60 4.60 1910 :=
by
  unfold break_even
  norm_num
  sorry

end break_even_handles_l454_454386


namespace geometric_sequence_fraction_l454_454323

-- Define the conditions
variable {a : ℕ → ℝ} {q : ℝ} (h_geometric : ∀ n, a (n+1) = a n * q)
variable (S : ℕ → ℝ)
variable (h_seq_sum2 : S 2 = a 1 + a 2)
variable (h_seq_sum4 : S 4 = a 1 + a 2 + a 3 + a 4)

-- State the given condition
variable (h_condition : S 4 = 5 * S 2)

-- Define the question and result conditions
theorem geometric_sequence_fraction :
  ∀ (n : ℕ), h_condition →
  (q ≠ 1 → (q^2 = 4 → (a 1 - a 5) / (a 3 + a 5) = -3 / 4)) ∧
  (q^2 = 1 → (a 1 - a 5) / (a 3 + a 5) = 0) :=
by sorry

end geometric_sequence_fraction_l454_454323


namespace repeating_decimal_equals_fraction_l454_454142

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454142


namespace barrel_arrangements_42_l454_454384

theorem barrel_arrangements_42 :
  ∃ (arrangements : Finset (Fin 9 × Fin 9 → ℕ)), 
      arrangements.card = 42 ∧ 
      ∀ (grid ∈ arrangements), 
        (∀ i j : Fin 3, grid i j < grid i (j + 1) ∧ grid i j < grid (i + 1) j) := 
sorry

end barrel_arrangements_42_l454_454384


namespace fraction_power_computation_l454_454005

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end fraction_power_computation_l454_454005


namespace hot_dogs_total_l454_454528

theorem hot_dogs_total (D : ℕ)
  (h1 : 9 = 2 * D + D + 3) :
  (2 * D + 9 + D = 15) :=
by sorry

end hot_dogs_total_l454_454528


namespace solve_for_x_l454_454903

theorem solve_for_x : 
  ∃ (x : ℝ), (13 + real.sqrt (-4 + 5 * x * 3) = 14) ∧ x = 1 / 3 :=
begin
  use 1 / 3,
  split,
  { 
    calc 
      13 + real.sqrt (-4 + 5 * (1 / 3) * 3) 
        = 13 + real.sqrt (-4 + 5 * (1 / 3) * 3) : by refl
    ... = 13 + real.sqrt (-4 + 5 * 1) : by norm_num
    ... = 13 + real.sqrt (-4 + 5) : by norm_num
    ... = 13 + real.sqrt 1 : by norm_num
    ... = 13 + 1 : by rw real.sqrt_one
    ... = 14 : by norm_num
  },
  exact rfl
end

end solve_for_x_l454_454903


namespace num_valid_combinations_l454_454917

-- Definitions of the problem's parameters
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Predicate for a valid four-digit number meeting the problem's criteria
def is_valid_number (num : List ℕ) : Prop :=
  num.length = 4 ∧
  (∀ x ∈ num, x ∈ digits) ∧
  num.nodup ∧
  (num.nth 0 ≠ 0) ∧  -- ensure it's a four-digit number, i.e., the first digit is not zero
  (num.index_of 1 + 1 ≠ num.index_of 2) ∧  -- ensure 1 and 2 are not adjacent
  (num.index_of 1 + 1 ≠ num.index_of 2 + 1)

-- Theorem statement
theorem num_valid_combinations : {num : List ℕ | is_valid_number num}.to_finset.card = 68 :=
by sorry -- we skip the proof.

end num_valid_combinations_l454_454917


namespace shortest_line_body_diagonal_lateral_surface_l454_454095

-- Define the condition that the radius of the base's circumscribed circle is smaller
-- than the radii of the circumscribed circles around its lateral faces.
def circumscribed_circle_radius_base_lt_lateral (P : TruncatedSquarePyramid) : Prop :=
  P.base_circumscribed_radius < P.lateral_circumscribed_radius

-- Define the main theorem stating that the shortest line connecting two
-- endpoints of a body diagonal lies on the lateral surface of the pyramid.
theorem shortest_line_body_diagonal_lateral_surface (P : TruncatedSquarePyramid)
  (h : circumscribed_circle_radius_base_lt_lateral P) :
  ∀ (A G : Point), (A ∈ P.vertices ∧ G ∈ P.vertices ∧ A ≠ G) →
  shortest_path_on_surface A G P ∈ P.lateral_surfaces :=
sorry

end shortest_line_body_diagonal_lateral_surface_l454_454095


namespace simplify_fraction_l454_454747

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l454_454747


namespace repeating_decimal_fraction_equiv_l454_454150

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454150


namespace problem1_solution_problem2_solution_problem3_solution_l454_454090

noncomputable def problem1 : Real :=
  sqrt 32 - sqrt 18 + sqrt 8

noncomputable def problem2 : Real :=
  6 * sqrt 2 * sqrt 3 + 3 * sqrt 30 / sqrt 5

noncomputable def problem3 : Real :=
  4 / sqrt 2 + 2 * sqrt 18 - sqrt 24 * sqrt (1 / 3)

theorem problem1_solution : problem1 = 3 * sqrt 2 := by
  sorry

theorem problem2_solution : problem2 = 9 * sqrt 6 := by
  sorry

theorem problem3_solution : problem3 = 6 * sqrt 2 := by
  sorry

end problem1_solution_problem2_solution_problem3_solution_l454_454090


namespace problem1_problem2_l454_454256

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a/x

theorem problem1 (a : ℝ) (x : ℝ) (h : a = 2) : 
  f x a - f (x - 1) a > 2 * x - 1 ↔ 0 < x ∧ x < 1 := sorry

theorem problem2 (a : ℝ) :
  (a = 0 → ∀ x : ℝ, f x a = f (-x) a) ∧
  (a ≠ 0 → ¬ (∀ x : ℝ, f x a = f (-x) a ∨ f x a = -f (-x) a)) := sorry

end problem1_problem2_l454_454256


namespace solve_equation_roots_l454_454754

theorem solve_equation_roots :
  ∀ x : ℝ, 
  x^2 + 2 * x - 3 - sqrt((x^2 + 2 * x - 3) / (x^2 - 2 * x - 3)) = 2 / (x^2 - 2 * x - 3)
  ↔ x = real.sqrt(5 + 2 * real.sqrt(5)) ∨ 
     x = -real.sqrt(5 + 2 * real.sqrt(5)) ∨ 
     x = real.sqrt(5 - real.sqrt(17)) ∨ 
     x = -real.sqrt(5 - real.sqrt(17)) := 
sorry

end solve_equation_roots_l454_454754


namespace max_distance_to_line_l454_454665

noncomputable def curve_param_eq := (x y : ℝ) (θ : ℝ) :=
  x = 1 + Real.cos θ ∧ y = 2 + Real.sin θ

theorem max_distance_to_line (θ : ℝ) :
  let x := 1 + Real.cos θ
  let y := 2 + Real.sin θ
  let d := abs (Real.sin (Real.pi / 4 - θ) - Real.sqrt 2)
  ∃ P : ℝ × ℝ, (P = (x, y) ∧ (x - y - 1 = 0) ∧ d = 1 + Real.sqrt 2) :=
  sorry

end max_distance_to_line_l454_454665


namespace bob_pennies_l454_454984

variable (a b : ℕ)

theorem bob_pennies : 
  (b + 2 = 4 * (a - 2)) →
  (b - 3 = 3 * (a + 3)) →
  b = 78 :=
by
  intros h1 h2
  sorry

end bob_pennies_l454_454984


namespace part1_part2_combined_proof_l454_454621

open Real

-- Define the curve G with the given function.
def curve (x : ℝ) := sqrt (2 * x^2 + 2)

-- Define the focus F.
def focus : ℝ × ℝ := (0, sqrt 3)

-- Define the lines l1 and l2 passing through focus F.
def line1 (k x : ℝ) := k * x + sqrt 3
def line2 (k x : ℝ) := (-1 / k) * x + sqrt 3

-- Define the condition for perpendicular diagonals.
def perpendicular_diagonals (AC BD : ℝ × ℝ) : Prop :=
  AC.1 * BD.1 + AC.2 * BD.2 = 0

-- Proof of the equation of G and the coordinates of its focus.
theorem part1 : (∀ x y, curve x = y ↔ (y^2)/2 - x^2 = 1) ∧ focus = (0, sqrt 3) := 
begin
  split,
  { intros x y,
    split,
    { intro h,
      rw [curve, h],
      sorry, -- Details of converting and transforming the function
    },
    { intro h,
      sorry, -- Transforming back the standard form to the original function
    }
  },
  { refl }
end

-- Proof of the minimum area of quadrilateral ABCD.
theorem part2 (k : ℝ) (h₁ : 1 / 2 < k^2) (h₂ : k^2 < 2) : 
  ∀ (A B C D : ℝ × ℝ), perpendicular_diagonals (A, C) (B, D) →
  let S := (4 * (1 + k^2)^2) / ((2 - k^2) * (2 * k^2 - 1)) in
  S = 16 :=
sorry -- Insert a valid proof here

-- Combine Parts 1 and 2 into a single proof statement
theorem combined_proof (F : ℝ × ℝ) (k : ℝ) (A B C D : ℝ × ℝ) :
  F = (0, sqrt 3) → perpendicular_diagonals (A, C) (B, D) →
  (∀ x y, curve x = y ↔ (y^2)/2 - x^2 = 1) ∧ 
  let S := (4 * (1 + k^2)^2) / ((2 - k^2) * (2 * k^2 - 1)) in
  (1 / 2 < k^2 ∧ k^2 < 2) → S = 16 :=
sorry -- Proof to show that given conditions lead to correct formulas and area

end part1_part2_combined_proof_l454_454621


namespace tangent_line_through_origin_l454_454650

theorem tangent_line_through_origin (α : ℝ) (h : deriv (λ x : ℝ, x^α + 1) 1 = 2) : α = 2 :=
by
  sorry

end tangent_line_through_origin_l454_454650


namespace bisection_second_iteration_value_l454_454351

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_second_iteration_value :
  f 0.25 = -0.234375 :=
by
  -- The proof steps would go here
  sorry

end bisection_second_iteration_value_l454_454351


namespace smallest_x_comp_abs_polynomial_l454_454802

def is_composite (n : ℤ) : Prop :=
  n > 1 ∧ ¬is_prime n

theorem smallest_x_comp_abs_polynomial : 
  ∃ x : ℤ, |5 * x^2 - 38 * x + 7| ∈ {n : ℤ | is_composite n} ∧ 
           ∀ y : ℤ, |5 * y^2 - 38 * y + 7| ∈ {n : ℤ | is_composite n} → x ≤ y :=
  sorry

end smallest_x_comp_abs_polynomial_l454_454802


namespace tidy_up_time_l454_454975

theorem tidy_up_time (A B C : ℕ) (tidyA : A = 5 * 3600) (tidyB : B = 5 * 60) (tidyC : C = 5) :
  B < A ∧ B > C :=
by
  sorry

end tidy_up_time_l454_454975


namespace sum_of_distinct_3_digit_integers_l454_454106

theorem sum_of_distinct_3_digit_integers (digits : List ℕ) (digits_spec : digits = [1, 2, 3, 5, 7]) : 
  let n := 5 in
  let k := 3 in
  let num_combinations := n ^ k in
  let frequency := num_combinations / n in
  let sum_digits := digits.sum in
  let contribution_hundreds := frequency * sum_digits * 100 in
  let contribution_tens := frequency * sum_digits * 10 in
  let contribution_ones := frequency * sum_digits in
  (contribution_hundreds + contribution_tens + contribution_ones) = 49950 := by
  sorry

end sum_of_distinct_3_digit_integers_l454_454106


namespace repeating_decimal_equals_fraction_l454_454137

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454137


namespace repeating_decimal_eq_fraction_l454_454173

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454173


namespace cos_alpha_minus_pi_over_2_l454_454933

variable α : ℝ 

theorem cos_alpha_minus_pi_over_2 (h : Real.sin (Real.pi + α) = Real.sqrt 3 / 2) :
  Real.cos (α - Real.pi / 2) = -Real.sqrt 3 / 2 := by
  sorry

end cos_alpha_minus_pi_over_2_l454_454933


namespace prove_altSum_2011_l454_454456

def altSum (n : ℕ) : ℤ :=
  ∑ k in Finset.range n.succ, (-1) ^ k

theorem prove_altSum_2011 : altSum 2011 = 1 := by
  sorry

end prove_altSum_2011_l454_454456


namespace crushing_load_example_l454_454420

noncomputable def crushing_load (T H : ℝ) : ℝ :=
  (30 * T^5) / H^3

theorem crushing_load_example : crushing_load 5 10 = 93.75 := by
  sorry

end crushing_load_example_l454_454420


namespace simplify_fraction_l454_454738

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454738


namespace percent_of_y_eq_l454_454468

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end percent_of_y_eq_l454_454468


namespace triangle_angle_A_l454_454348

-- Define the angles and sides of the triangle
def angle_C := 30 -- In degrees
def side_a := 3 * Real.sqrt 2
def side_c := 3

-- The angles A we need to prove
def angle_A1 := 45 -- In degrees
def angle_A2 := 135 -- In degrees

-- The condition we need to prove
theorem triangle_angle_A (a c : ℝ) (C : ℝ) (h_a : a = 3 * Real.sqrt 2) (h_c : c = 3) (h_C : C = 30):
  A = 45 ∨ A = 135 := 
  sorry 

end triangle_angle_A_l454_454348


namespace polynomial_coeff_sum_neg_33_l454_454292

theorem polynomial_coeff_sum_neg_33
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 - 3 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -33 :=
by sorry

end polynomial_coeff_sum_neg_33_l454_454292


namespace find_x_l454_454807

noncomputable def value_of_x (x : ℝ) := (5 * x) ^ 4 = (15 * x) ^ 3

theorem find_x : ∀ (x : ℝ), (value_of_x x) ∧ (x ≠ 0) → x = 27 / 5 :=
by
  intro x
  intro h
  sorry

end find_x_l454_454807


namespace part_I_part_II_l454_454954

noncomputable def f (x : ℝ) := (Real.sin x) * (Real.cos x) + (Real.sin x)^2

-- Part I: Prove that f(π / 4) = 1
theorem part_I : f (Real.pi / 4) = 1 := sorry

-- Part II: Prove that the maximum value of f(x) for x ∈ [0, π / 2] is (√2 + 1) / 2
theorem part_II : ∃ x ∈ Set.Icc 0 (Real.pi / 2), (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧ f x = (Real.sqrt 2 + 1) / 2 := sorry

end part_I_part_II_l454_454954


namespace a_2017_eq_l454_454704

noncomputable def a : ℕ → ℚ
| 1 := 3 / 8
| n := sorry  -- Placeholder for the rest of the definition, derived from recursive relations

lemma a_recursive_ineq1 (n : ℕ) (h : n > 0) : a (n + 2) - a n ≤ 3^n :=
sorry -- This will need to be derived from the condition

lemma a_recursive_ineq2 (n : ℕ) (h : n > 0) : a (n + 4) - a n ≥ 10 * 3^n :=
sorry -- This will need to be derived from the condition

theorem a_2017_eq : a 2017 = 3^2017 / 8 :=
sorry -- This will need to show that the derived sequence's 2017th term is as required

end a_2017_eq_l454_454704


namespace good_subset_count_l454_454371

noncomputable def numberOfGoodSubsets (S : Finset ℕ) (n : ℕ) :=
  (Finset.card S).choose n

theorem good_subset_count :
  let S : Finset ℕ := Finset.range 2026 \ Finset.singleton 0
  in numberOfGoodSubsets S 51 % 5 = 0 →
  numberOfGoodSubsets S 51 / 5 = (1 / 5) * (Finset.card S).choose 51 :=
sorry

end good_subset_count_l454_454371


namespace valid_k_range_l454_454418

noncomputable def valid_k_values (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 + 2 * k * x + 1 ≠ 0

theorem valid_k_range : {k : ℝ | valid_k_values k} = set.Ico 0 1 :=
begin
  sorry
end

end valid_k_range_l454_454418


namespace cone_volume_correct_l454_454837

-- Define the conditions
def slant_height (sector_radius : ℝ) := 5
def central_angle (sector_angle : ℝ) := (6 * Real.pi / 5)

-- Define the volume of the cone
def cone_volume (SectorRadius : ℝ) (SectorAngle : ℝ) :=
  let l := slant_height SectorRadius
  let θ := central_angle SectorAngle
  let s := θ * SectorRadius
  let r := s / (2 * Real.pi)
  let h := Math.sqrt (l^2 - r^2)
  (1 / 3) * Real.pi * r^2 * h

-- Prove that the volume of the cone is 12π
theorem cone_volume_correct : cone_volume 5 (6 * Real.pi / 5) = 12 * Real.pi := by
  sorry

end cone_volume_correct_l454_454837


namespace matching_jelly_bean_probability_l454_454529

-- define Ava's jelly beans
def ava_jelly_beans : List (String × ℕ) := [
  ("green", 2),
  ("red", 2)
]

-- define Ben's jelly beans
def ben_jelly_beans : List (String × ℕ) := [
  ("green", 2),
  ("yellow", 3),
  ("red", 3)
]

-- calculate total jelly beans Ava has
def ava_total : ℕ := ava_jelly_beans.sum (fun (_, n) => n)

-- calculate total jelly beans Ben has
def ben_total : ℕ := ben_jelly_beans.sum (fun (_, n) => n)

-- calculate the probability of seeing "green" for Ava
def ava_green_probability : ℚ := (ava_jelly_beans.find_exc (fun (c, _) => c = "green")).2  / ava_total

-- calculate the probability of seeing "red" for Ava
def ava_red_probability : ℚ := (ava_jelly_beans.find_exc (fun (c, _) => c = "red")).2  / ava_total

-- calculate the probability of seeing "green" for Ben
def ben_green_probability : ℚ := (ben_jelly_beans.find_exc (fun (c, _) => c = "green")).2  / ben_total

-- calculate the probability of seeing "red" for Ben
def ben_red_probability : ℚ := (ben_jelly_beans.find_exc (fun (c, _) => c = "red")).2  / ben_total

-- define the event of color matching
def probability_of_matching_colors : ℚ := (ava_green_probability * ben_green_probability) + (ava_red_probability * ben_red_probability)

--state the theorem to be proved
theorem matching_jelly_bean_probability : 
  probability_of_matching_colors = 5 / 16 := 
by
  sorry -- Proof goes here

end matching_jelly_bean_probability_l454_454529


namespace ratio_of_triangle_areas_l454_454310

theorem ratio_of_triangle_areas (A B C D E : Point) (circle : Circle)
  (diameter : Diameter circle A B)
  (chord : Chord circle C D)
  (parallel : Parallel AB CD)
  (intersection : AC ∩ BD = E)
  (angle_AED : ∠AED = β) :
  (area (triangle A B E)) / (area (triangle C D E)) = sin β ^ 2 := by
  sorry

end ratio_of_triangle_areas_l454_454310


namespace degrees_to_radians_l454_454541

theorem degrees_to_radians (deg: ℝ) (h : deg = 120) : deg * (π / 180) = 2 * π / 3 :=
by
  simp [h]
  sorry

end degrees_to_radians_l454_454541


namespace mondays_in_first_70_days_l454_454009

theorem mondays_in_first_70_days (days : ℕ) (h1 : days = 70) (mondays_per_week : ℕ) (h2 : mondays_per_week = 1) : 
  ∃ (mondays : ℕ), mondays = 10 := 
by
  sorry

end mondays_in_first_70_days_l454_454009


namespace length_AC_l454_454926

theorem length_AC {AB BC : ℝ} (h1: AB = 6) (h2: BC = 4) : (AC = 2 ∨ AC = 10) :=
sorry

end length_AC_l454_454926


namespace collinear_points_l454_454934

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {e₁ e₂ : V} (h_non_collinear : e₁ ≠ e₂)
variables (A B C D : V)
variables (λ : ℝ)

open_locale classical

theorem collinear_points (h₁ : B - A = 2 • e₁ + e₂)
    (h₂ : C - B = -e₁ + 3 • e₂)
    (h₃ : D - C = λ • e₁ - e₂)
    (h_collinear : ∃ μ : ℝ, B - A = μ • (D - A)) :
    λ = 5 :=
sorry

end collinear_points_l454_454934


namespace analytical_form_of_f_is_correct_solution_set_of_inequality_correct_l454_454929

def h (x : ℝ) : ℝ := 2^x

noncomputable def f (x : ℝ) : ℝ := (1 - h x) / (1 + h x)

theorem analytical_form_of_f_is_correct : 
  ∀ x : ℝ, f x = (1 - 2^x) / (1 + 2^x) :=
sorry

theorem solution_set_of_inequality_correct {x : ℝ} :
  f (2 * x - 1) > f (x + 1) ↔ x < (2 / 3) :=
sorry

end analytical_form_of_f_is_correct_solution_set_of_inequality_correct_l454_454929


namespace amelia_wins_l454_454864

noncomputable def amelia_wins_probability : ℚ := 21609 / 64328

theorem amelia_wins (h_am_heads : ℚ) (h_bl_heads : ℚ) (game_starts : Prop) (game_alternates : Prop) (win_condition : Prop) :
  h_am_heads = 3/7 ∧ h_bl_heads = 1/3 ∧ game_starts ∧ game_alternates ∧ win_condition →
  amelia_wins_probability = 21609 / 64328 :=
sorry

end amelia_wins_l454_454864


namespace inhabitants_even_l454_454790

def knight (x : α) : Prop
def liar (x : α) : Prop
def even_knights (n : ℕ) : Prop := n % 2 = 0
def odd_liars (n : ℕ) : Prop := n % 2 = 1

theorem inhabitants_even (n_knights n_liars : ℕ) (H1 : ∀ x, knight x → even_knights n_knights)
                        (H2 : ∀ x, liar x → odd_liars n_liars) : (n_knights + n_liars) % 2 = 0 := 
sorry

end inhabitants_even_l454_454790


namespace train_crosses_pole_in_l454_454072

/-
Problem Statement: A train running at the speed of 56 km/hr crosses a pole. 
The length of the train is 140 meters. Prove that it takes 9 seconds for the 
train to cross the pole.
-/

def train_speed_kmph : ℝ := 56
def train_length_m : ℝ := 140
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

theorem train_crosses_pole_in : 
  (train_length_m / train_speed_mps) = 9 := 
sorry

end train_crosses_pole_in_l454_454072


namespace find_k_l454_454336

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l454_454336


namespace percent_equivalence_l454_454467

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end percent_equivalence_l454_454467


namespace number_of_members_l454_454840

theorem number_of_members (n : ℕ) (h : n * n = 8649) : n = 93 :=
by
  sorry

end number_of_members_l454_454840


namespace modulus_of_z_l454_454379

variable (z : ℂ)
variable (h : conj z * (1 - I) = 2 * I)

theorem modulus_of_z : abs z = real.sqrt 2 := 
sorry

end modulus_of_z_l454_454379


namespace count_rationals_in_list_l454_454865

def is_rational (x : ℚ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

theorem count_rationals_in_list :
  let numbers := [4, π, -1/3, 0, -3.142, -0.5, 0.4, -3.2626626662] in
  (numbers.filter is_rational).length = 6 :=
by sorry

end count_rationals_in_list_l454_454865


namespace proof_lambda_l454_454964

noncomputable def vector_a := (-4 : ℝ, 3 : ℝ)
noncomputable def point_A := (-1 : ℝ, 1 : ℝ)
noncomputable def point_B := (0 : ℝ, -1 : ℝ)

noncomputable def A1 := vector.projection vector_a point_A
noncomputable def B1 := vector.projection vector_a point_B

def question_lambda : ℝ := ∥(B1 - A1)∥ / ∥vector_a∥

theorem proof_lambda :
  ∀ (vector_a point_A point_B A1 B1 : ℝ × ℝ),
    A1 = vector.projection vector_a point_A →
    B1 = vector.projection vector_a point_B →
    (∃ λ : ℝ, (B1 - A1) = λ *ᵥ vector_a) →
    question_lambda = -2 / 5 :=
by
  intros,
  sorry

end proof_lambda_l454_454964


namespace cyclic_quadrilateral_sum_of_squares_l454_454829

theorem cyclic_quadrilateral_sum_of_squares (A B C D E O : Point) 
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_perp : Perpendicular AC BD)
  (h_intersect : meets_at E AC BD)
  (h_circumradius : isCircumRadius O R)
  : EA^2 + EB^2 + EC^2 + ED^2 = 4 * R^2 := 
by
  sorry

end cyclic_quadrilateral_sum_of_squares_l454_454829


namespace area_triangle_F1MF2_l454_454098

-- Define the ellipse equation
def ellipse (x y : ℝ) := x^2 / 72 + y^2 / 36 = 1

-- Define the focal distance and the distance |F1F2|
def focal_distance (a b : ℝ) := sqrt (a^2 - b^2)
def distance_F1F2 (c : ℝ) := 2 * c

-- Given parameters for the ellipse
def a := 6 * sqrt 2
def b := 6

-- Given angle condition
def angle_F1MF2 := 60 * (Real.pi / 180) -- 60 degrees in radians

-- The point M must be on the ellipse
def point_on_ellipse (M : (ℝ × ℝ)) :=
  let (Mx, My) := M in ellipse Mx My

-- The main theorem statement
theorem area_triangle_F1MF2 
  (M : (ℝ × ℝ))
  (hM : point_on_ellipse M) 
  (t1 t2 : ℝ) 
  (h_t1t2_sum : t1 + t2 = 12 * sqrt 2) 
  (h_cosine_rule : t1^2 + t2^2 - 2 * t1 * t2 * Real.cos angle_F1MF2 = (12 * sqrt 2)^2)
  : 1 / 2 * t1 * t2 * Real.sin angle_F1MF2 = 12 * sqrt 3 :=
sorry

end area_triangle_F1MF2_l454_454098


namespace solution_l454_454378

open Set

def problem_statement : Prop :=
  let A := {1, 2, 3, 4, 5, 6}
  let B := {4, 5, 6, 7, 8}
  let subsetsA := {S | S ⊆ A}
  let validSubsets := {S | S ⊆ A ∧ S ∩ B ≠ ∅}
  (finite.to_finset validSubsets).card = 56

theorem solution : problem_statement :=
by
  sorry

end solution_l454_454378


namespace taxi_ride_cost_l454_454859

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end taxi_ride_cost_l454_454859


namespace solve_factor_l454_454039

theorem solve_factor 
  (x : ℝ) (h : x = 22.142857142857142) 
  (h_eq : (λ f : ℝ, (f * (x + 5)) / 5 - 5 = 33) ∃ f : ℝ, f ≈ 7) : 
  ∃ f : ℝ, f ≈ 7 := 
sorry

end solve_factor_l454_454039


namespace sequence_k_eq_4_l454_454329

theorem sequence_k_eq_4 {a : ℕ → ℕ} (h1 : a 1 = 2) (h2 : ∀ m n, a (m + n) = a m * a n)
    (h3 : ∑ i in finset.range 10, a (4 + i) = 2^15 - 2^5) : 4 = 4 :=
by
  sorry

end sequence_k_eq_4_l454_454329


namespace required_pumps_l454_454846

-- Define the conditions in Lean
variables (x a b n : ℝ)

-- Condition 1: x + 40a = 80b
def condition1 : Prop := x + 40 * a = 2 * 40 * b

-- Condition 2: x + 16a = 64b
def condition2 : Prop := x + 16 * a = 4 * 16 * b

-- Main theorem: Given the conditions, prove that n >= 6 satisfies the remaining requirement
theorem required_pumps (h1 : condition1 x a b) (h2 : condition2 x a b) : n >= 6 :=
by
  sorry

end required_pumps_l454_454846


namespace find_k_l454_454343

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else sorry

axiom seq_base : seq 1 = 2
axiom seq_recur (m n : ℕ) : seq (m + n) = seq m * seq n

-- Given condition for sum
axiom sum_condition (k : ℕ) : 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5

-- The statement to prove
theorem find_k : ∃ k : ℕ, 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5 ∧ k = 4 :=
by {
  sorry -- proof is not required
}

end find_k_l454_454343


namespace rectangle_ratio_l454_454317

theorem rectangle_ratio (a b c d : ℝ) (h₀ : a = 4)
  (h₁ : b = (4 / 3)) (h₂ : c = (8 / 3)) (h₃ : d = 4) :
  (∃ XY YZ, XY * YZ = a * a ∧ XY / YZ = 0.9) :=
by
  -- Proof to be filled
  sorry

end rectangle_ratio_l454_454317


namespace smallest_number_of_marbles_l454_454471

theorem smallest_number_of_marbles (M : ℕ) (h1 : M ≡ 2 [MOD 5]) (h2 : M ≡ 2 [MOD 6]) (h3 : M ≡ 2 [MOD 7]) (h4 : 1 < M) : M = 212 :=
by sorry

end smallest_number_of_marbles_l454_454471


namespace size_of_deleted_folder_l454_454398

theorem size_of_deleted_folder (free_space_before : ℝ) (used_space_before : ℝ) (new_file_size : ℝ) (new_drive_total_size : ℝ) (new_drive_free_space : ℝ) (x : ℝ) :
  free_space_before = 2.4 ∧
  used_space_before = 12.6 ∧
  new_file_size = 2 ∧
  new_drive_total_size = 20 ∧
  new_drive_free_space = 10 →
  (used_space_before - x + new_file_size = new_drive_total_size - new_drive_free_space) →
  x = 4.6 :=
by
  intros h1 h2
  cases h1 with hfree hrest
  cases hrest with hused hrest2
  cases hrest2 with hnewfile hrest3
  cases hrest3 with hnewdrive hfreedrive
  rw [hfree, hused, hnewfile, hnewdrive, hfreedrive] at h2
  have eq1 := h2
  linarith

end size_of_deleted_folder_l454_454398


namespace prove_triangle_property_l454_454713

-- Definitions of the points and lines as per the given problem conditions
variables (A B C A1 B1 C1 C2 B2 : Type)
variables [add_group A] [add_group B] [add_group C]
variables [add_group A1] [add_group B1] [add_group C1]
variables [add_group C2] [add_group B2]
variables (AA1 BB1 CC1 A1B1 A1C1 : Type)
variables (line_passing_through_A parallel_BC : Type)
variables (segments_intersect : AA1 → BB1 → CC1 → Prop)
variables (lines_intersect : A1B1 → A1C1 → line_passing_through_A → parallel_BC → Prop)

-- The main statement to prove
def triangle_property : Prop :=
  ∀ (A B C A1 B1 C1 C2 B2 : Type)
    [add_group A] [add_group B] [add_group C]
    [add_group A1] [add_group B1] [add_group C1]
    [add_group C2] [add_group B2],
  let AA1 := AA1 in
  let BB1 := BB1 in
  let CC1 := CC1 in
  let A1B1 := A1B1 in
  let A1C1 := A1C1 in
  let line_passing_through_A := line_passing_through_A in
  let parallel_BC := parallel_BC in
  let segments_intersect := segments_intersect AA1 BB1 CC1 in
  let lines_intersect := lines_intersect A1B1 A1C1 line_passing_through_A parallel_BC in
  segments_intersect → lines_intersect → (AB2 = AC2)

-- Proving the property
theorem prove_triangle_property : triangle_property :=
begin
  sorry,
end

end prove_triangle_property_l454_454713


namespace sum_of_three_consecutive_integers_is_21_l454_454474

theorem sum_of_three_consecutive_integers_is_21 : 
  ∃ (n : ℤ), 3 * n = 21 :=
by
  sorry

end sum_of_three_consecutive_integers_is_21_l454_454474


namespace fraction_equivalent_of_repeating_decimal_l454_454196

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454196


namespace abs_diff_of_diagonal_sums_l454_454041

def initial_matrix : matrix (fin 5) (fin 5) ℕ :=
  ![[1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]]

def reversed_matrix : matrix (fin 5) (fin 5) ℕ :=
  ![[1, 2, 3, 4, 5],
    [10, 9, 8, 7, 6],
    [15, 14, 13, 12, 11],
    [16, 17, 18, 19, 20],
    [25, 24, 23, 22, 21]]

def main_diag_sum (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  let indices := [0, 1, 2, 3, 4].map fin.of_nat in
  indices.sum (λ i, m i i)

def antidiag_sum (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  let indices := [0, 1, 2, 3, 4].map fin.of_nat in
  indices.sum (λ i, m i (4 - i))

theorem abs_diff_of_diagonal_sums : 
  |main_diag_sum reversed_matrix - antidiag_sum reversed_matrix| = 4 :=
by
  sorry

end abs_diff_of_diagonal_sums_l454_454041


namespace monotonic_increasing_condition_l454_454761

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → y a x₁ ≤ y a x₂) ↔ 
  (a = 0 ∨ a > 0) :=
sorry

end monotonic_increasing_condition_l454_454761


namespace sum_of_box_weights_l454_454991

theorem sum_of_box_weights (heavy_box_weight : ℚ) (difference : ℚ) 
  (h1 : heavy_box_weight = 14 / 15) (h2 : difference = 1 / 10) :
  heavy_box_weight + (heavy_box_weight - difference) = 53 / 30 := 
  by
  sorry

end sum_of_box_weights_l454_454991


namespace no_real_x_for_sqrt_l454_454572

theorem no_real_x_for_sqrt :
  ¬ ∃ x : ℝ, - (x^2 + 2 * x + 5) ≥ 0 :=
sorry

end no_real_x_for_sqrt_l454_454572


namespace a5_value_l454_454217

-- Defining the conditions
def binomial_expansion (n : ℕ) (a : ℕ → ℤ) : Prop := 
  ∀ x : ℤ, (1 - x)^n = (1 + ∑ i in finset.range (n+1), a i * x^i)

def ratio_a1_a3 (a : ℕ → ℤ) : Prop := 
  a 1 * 7 = a 3

-- The theorem to prove a5 = -56 given the conditions
theorem a5_value (n : ℕ) (a : ℕ → ℤ) 
  (h1 : binomial_expansion n a) 
  (h2 : ratio_a1_a3 a) : 
  n = 8 → a 5 = -56 :=
by 
sorry

end a5_value_l454_454217


namespace inverse_proportional_p_q_l454_454409

theorem inverse_proportional_p_q (k : ℚ)
  (h1 : ∀ p q : ℚ, p * q = k)
  (h2 : (30 : ℚ) * (4 : ℚ) = k) :
  p = 12 ↔ (10 : ℚ) * p = k :=
by
  sorry

end inverse_proportional_p_q_l454_454409


namespace cade_marbles_left_l454_454531

variable (initial_marbles : ℕ) (given_away : ℕ) : ℕ := 87
variable (given_away : ℕ) : ℕ := 8

theorem cade_marbles_left : initial_marbles - given_away = 79 :=
by {
  -- Conditions
  let initial_marbles := 87
  let given_away := 8

  -- Think of this theorem essentially as expressing the condition and answer in a way Lean can check it
  sorry
}

end cade_marbles_left_l454_454531


namespace insphere_touches_centers_of_faces_l454_454539

/--
Given a regular tetrahedron whose faces are all equilateral triangles, 
prove that the insphere touches each triangular face at its center.
-/
theorem insphere_touches_centers_of_faces (T : regular_tetrahedron) :
  ∀ face ∈ T.faces, T.insphere.touches face (T.face_center face) :=
sorry

end insphere_touches_centers_of_faces_l454_454539


namespace value_of_b_l454_454298

theorem value_of_b (x b : ℝ) (h₁ : x = 0.3) 
  (h₂ : (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : 
  b = 2 :=
by
  sorry

end value_of_b_l454_454298


namespace percentage_of_fescue_in_Y_l454_454399

-- Definitions for the given percentages in the problem
def ryegrass_in_X : ℝ := 0.4 -- 40%
def ryegrass_in_Y : ℝ := 0.25 -- 25%
def mixture_percentage_X : ℝ := 1 / 3 -- 33.33333333333333%
def mixture_ryegrass_percentage : ℝ := 0.3 -- 30%

-- Definition of the problem statement in Lean 4
theorem percentage_of_fescue_in_Y : 
  (1 - ryegrass_in_Y) = 0.75 :=
  by
  -- Set up hypotheses and assertions for the proof
  have h1 : mixture_percentage_X * ryegrass_in_X + (1 - mixture_percentage_X) * ryegrass_in_Y = mixture_ryegrass_percentage := sorry
  
  sorry

end percentage_of_fescue_in_Y_l454_454399


namespace cyclic_quadrilateral_count_24_l454_454643

theorem cyclic_quadrilateral_count_24 :
  let valid_quadrilateral (a b c d : ℕ) := a + b + c + d = 24 ∧ 
                                            (a ∣ b + c + d) ∧
                                            (b ∣ a + c + d) ∧
                                            (c ∣ a + b + d) ∧
                                            (d ∣ a + b + c) ∧
                                            (a,b,c,d ≠ 0)
  in (∃ (sides : finset (ℕ × ℕ × ℕ × ℕ)), 
            sides.card = 225 ∧ 
            ∀ (a b c d : ℕ), (a, b, c, d) ∈ sides → valid_quadrilateral a b c d :=
                ∣ {a b c d ∈ (range 25) | a + b + c + d = 24 ∧ (a ∣ b + c + d) ∧
                                                             (b ∣ a + c + d) ∧
                                                             (c ∣ a + b + d) ∧
                                                             (d ∣ a + b + c) ∧
                                                             (a,b,c,d ≠ 0)}.card = 225)) :=
sorry

end cyclic_quadrilateral_count_24_l454_454643


namespace dot_product_a_b_l454_454932

-- Define the vectors and their properties
def i : ℝ^3 := ⟨1, 0, 0⟩  -- Unit vector i
def j : ℝ^3 := ⟨0, 1, 0⟩  -- Unit vector j
def k : ℝ^3 := ⟨0, 0, 1⟩  -- Unit vector k

-- Define the vectors a and b
def a : ℝ^3 := 3 • i + 2 • j - k
def b : ℝ^3 := i - j + 2 • k

-- State the theorem for the dot product
theorem dot_product_a_b : (a ⬝ b) = -1 := by
  sorry  -- Proof placeholder

end dot_product_a_b_l454_454932


namespace probability_sum_30_l454_454055

open Classical

noncomputable def first_die_faces : Finset (Option ℕ) := 
  insert_none (Finset.range 19) + 1

noncomputable def second_die_faces : Finset (Option ℕ) :=
  (insert_none ((Finset.range 10) + 1) ∪ (insert_none ((Finset.range 9) + 12)))

theorem probability_sum_30 : 
  (∃ (outcomes : Finset (Option ℕ × Option ℕ)),
    outcomes.card = (first_die_faces.product second_die_faces).card ∧
    (outcomes.filter (λ x, x.1.isSome ∧ x.2.isSome ∧ x.1.getOrElse 0 + x.2.getOrElse 0 = 30)).card = 9) →
    ((9 : ℚ) / (first_die_faces.card * second_die_faces.card) = 9 / 400) :=
by
  sorry

end probability_sum_30_l454_454055


namespace vector_magnitude_range_l454_454640

variable (α : Real)
def vector_a : Real × Real := (Real.cos α, 0)
def vector_b : Real × Real := (1, Real.sin α)
def magnitude (v : Real × Real) : Real := Real.sqrt (v.1^2 + v.2^2)

theorem vector_magnitude_range : 
  -1 ≤ Real.cos α ∧ Real.cos α ≤ 1 → 
  0 ≤ magnitude (vector_a α + vector_b α) ∧ magnitude (vector_a α + vector_b α) ≤ 2 :=
by
  sorry

end vector_magnitude_range_l454_454640


namespace croissant_combined_cost_l454_454904

-- Definitions for the problem conditions
def cost_plain : ℝ := 3
def cost_focaccia : ℝ := 4
def cost_latte : ℝ := 2.5
def total_spent : ℝ := 21

def total_known_costs : ℝ := cost_plain + cost_focaccia + 2 * cost_latte

theorem croissant_combined_cost (A S : ℝ) (h1 : total_spent = total_known_costs + A + S) : 
  A + S = 9 := by
  have h2 : total_known_costs = 12 := 
    by
    dsimp [total_known_costs, cost_plain, cost_focaccia, cost_latte]
    norm_num
  rw [h2] at h1
  linarith

end croissant_combined_cost_l454_454904


namespace red_tetrahedron_volume_is_172_l454_454838

   def cube_side_length := 8

   def cube_volume := cube_side_length ^ 3

   def base_area (a : ℝ) := (1 / 2) * a * a

   def tetrahedron_volume (base_area : ℝ) (height : ℝ) := (1 / 3) * base_area * height

   def clear_tetrahedron_volume := tetrahedron_volume (base_area cube_side_length) cube_side_length

   def total_clear_tetrahedra_volume := 4 * clear_tetrahedron_volume

   def red_tetrahedron_volume := cube_volume - total_clear_tetrahedra_volume

   theorem red_tetrahedron_volume_is_172 :
       red_tetrahedron_volume = 172 :=
   by
       -- necessary computations can be done within the proof
       sorry
   
end red_tetrahedron_volume_is_172_l454_454838


namespace repeating_decimal_to_fraction_l454_454120

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454120


namespace f_transform_l454_454617

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 4 * x - 5

theorem f_transform (x h : ℝ) : 
  f (x + h) - f x = 6 * x ^ 2 - 6 * x + 6 * x * h + 2 * h ^ 2 - 3 * h + 4 := 
by
  sorry

end f_transform_l454_454617


namespace polygon_interior_angle_sum_l454_454052

theorem polygon_interior_angle_sum (n : ℕ) (h : (n-1) * 180 = 2400 + 120) : n = 16 :=
by
  sorry

end polygon_interior_angle_sum_l454_454052


namespace fraction_power_computation_l454_454006

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end fraction_power_computation_l454_454006


namespace geometric_sequence_partial_sum_T_l454_454695

-- Define the sequence a_n and its partial sum S_n
def sequence_a : ℕ → ℝ
| 0       := 1
| (n + 1) := ((n + 2) / n) * (∑ i in range n, sequence_a i)

def partial_sum_S (n : ℕ) : ℕ := 
∑ i in range n, sequence_a i

-- Goal 1: Prove that the sequence {S_n / n} is geometric
theorem geometric_sequence (n : ℕ) (h : n > 0) : 
  ∃ r : ℝ, 
  (∑ i in range n, (partial_sum_S i) / (i+1)) = (partial_sum_S n) * r := 
  sorry

-- Goal 2: Find the partial sum T_n of the sequence {S_n}
def T (n : ℕ) : ℝ := 
∑ i in range n, partial_sum_S i

theorem partial_sum_T (n : ℕ) :
  T n = (n-1) * 2 ^ n + 1 := 
  sorry

end geometric_sequence_partial_sum_T_l454_454695


namespace sum_angles_triangle_complement_l454_454306

theorem sum_angles_triangle_complement (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) : A + B = 130 :=
by
  have hC : C = 50 := by linarith
  linarith

end sum_angles_triangle_complement_l454_454306


namespace solution_l454_454610

noncomputable def find_m (m : ℝ) : Prop :=
  let direction_vector := (2, m, 1)
  let normal_vector := (1, 0.5, 2)
  let dot_product := direction_vector.1 * normal_vector.1 + direction_vector.2 * normal_vector.2 + direction_vector.3 * normal_vector.3
  dot_product = 0

theorem solution : find_m (-8) :=
by
  let m := -8
  let direction_vector := (2, m, 1)
  let normal_vector := (1, 0.5, 2)
  let dot_product := direction_vector.1 * normal_vector.1 + direction_vector.2 * normal_vector.2 + direction_vector.3 * normal_vector.3
  show dot_product = 0
  sorry

end solution_l454_454610


namespace arithmetic_sequence_sum_l454_454362

  variable (a : ℕ → ℝ)
  variable (S : ℕ → ℝ)

  -- Condition 1: a_2 = 3
  def condition1 : a 2 = 3 := sorry

  -- Condition 2: a_6 = 11
  def condition2 : a 6 = 11 := sorry

  -- Sum of first n terms of arithmetic sequence
  def arithmetic_sum (n : ℕ) : ℝ := (n * (a 1 + a n) / 2)

  theorem arithmetic_sequence_sum : S 7 = 49 :=
  by
    -- Using the conditions to conclude the proof
    have h1 : a 2 = 3 := condition1
    have h2 : a 6 = 11 := condition2
    -- We need to show S_7 = 49 based on these
    sorry
  
end arithmetic_sequence_sum_l454_454362


namespace Worker2_time_on_DE_l454_454825

-- Definitions for the paving problem
variable (v : ℝ) -- Speed of Worker 1
variable (x : ℝ) -- Total distance covered by Worker 1 on path A-B-C
variable (total_time_hours : ℝ) -- Total time in hours both workers spend
variable (speed_ratio : ℝ) -- Speed ratio of Worker 2 to Worker 1
variable (DE_ratio_to_total : ℝ) -- DE distance ratio to total distance

-- Assumptions as per problem statement
axiom Worker1_speed : ∀ t, total_time_hours = x / v

axiom Worker2_speed : ∀ t, total_time_hours = (DE_ratio_to_total * x + x) / (speed_ratio * v)

axiom total_time : total_time_hours = 9
axiom speed_ratio_is_1point2 : speed_ratio = 1.2
axiom DE_ratio : DE_ratio_to_total = 0.1

-- Required result
theorem Worker2_time_on_DE : ∀ (t : ℝ), Worker2_speed t = 45 * 60 := by
  sorry

end Worker2_time_on_DE_l454_454825


namespace average_salary_l454_454028

theorem average_salary :
  let a_salary := 10000 + 2000 in
  let b_salary := 5000 - 1000 in
  let c_salary := 11000 + 3000 in
  let d_salary := 7000 - 500 in
  let e_salary := 9000 + 1500 in
  let f_salary := 12000 - 2000 in
  let g_salary := 8000 in
  let total_salary := a_salary + b_salary + c_salary + d_salary + e_salary + f_salary + g_salary in
  let average_salary := total_salary / 7 in
  average_salary = 9428.57 :=
by
  sorry

end average_salary_l454_454028


namespace max_xy_value_l454_454592

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) : xy ≤ 1 / 2 := 
by
  sorry

end max_xy_value_l454_454592


namespace intersection_of_A_and_B_l454_454605

def A := {x : ℝ | x^2 < 9}
def B := {x : ℕ | -1 < (x : ℝ) ∧ (x : ℝ) < 5}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} :=
by 
  sorry

end intersection_of_A_and_B_l454_454605


namespace tangent_line_at_1_1_l454_454421

noncomputable def power_index_function (f : ℝ → ℝ) (φ : ℝ → ℝ) : ℝ → ℝ := λ x, f(x) ^ φ(x)

theorem tangent_line_at_1_1 : 
  let f : ℝ → ℝ := λ x, x,
      φ : ℝ → ℝ := λ x, x in
  ∀ x (h : x > 0),
  (differentiable_at ℝ (λ x, power_index_function f φ x) 1) ∧ 
  (∃ m : ℝ, m = deriv (λ x, power_index_function f φ x) 1) ∧ 
  (m = 1) →
  ∀ y, (y - 1) = (x - 1) → y = x := 
by
  sorry

end tangent_line_at_1_1_l454_454421


namespace christopher_time_correct_l454_454534

def christopher_total_time 
  (laps : Nat) 
  (dist1 dist2 dist3 : Nat) 
  (speed1 speed2 speed3 : Nat) 
  (s1 s2 s3 : Rat): Nat :=
  (s1 + s2 + s3) * laps

theorem christopher_time_correct :
  christopher_total_time 3 200 200 100 5 4 6 (200 / 5) (200 / 4) (100 / 6) = 320 := 
sorry

end christopher_time_correct_l454_454534


namespace range_of_p_add_q_l454_454949

theorem range_of_p_add_q (p q : ℝ) :
  (∀ x : ℝ, ¬(x^2 + 2 * p * x - (q^2 - 2) = 0)) → 
  (p + q) ∈ Set.Ioo (-2 : ℝ) (2 : ℝ) :=
by
  intro h
  sorry

end range_of_p_add_q_l454_454949


namespace find_p_find_triangle_area_minimum_l454_454634

-- Define the parabola and line equations and their intersection points.
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
noncomputable def line_eq (x y : ℝ) := x - 2 * y + 1 = 0
noncomputable def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (p : ℝ) :
  (∃ (A B : ℝ × ℝ), parabola_eq p A.1 A.2 ∧ parabola_eq p B.1 B.2 ∧ 
   line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ 
   distance A.1 A.2 B.1 B.2 = 4 * real.sqrt 15) → p = 2 := sorry

-- Define conditions for the second part of the problem
noncomputable def focus_x := 1
noncomputable def focus_y := 0
noncomputable def mf_dot_nf_eq_zero (x1 y1 x2 y2 : ℝ) := 
  (x1 - focus_x) * (x2 - focus_x) + y1 * y2 = 0
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) := 
  1 / 2 * abs ((x1 - focus_x) * (y2 - focus_y) - (x2 - focus_x) * (y1 - focus_y))

theorem find_triangle_area_minimum (M N : ℝ × ℝ) :
  (parabola_eq 2 M.1 M.2 ∧ parabola_eq 2 N.1 N.2 ∧ 
  mf_dot_nf_eq_zero M.1 M.2 N.1 N.2) → 
  ∃ (Amin : ℝ), Amin = 12 - 8 * real.sqrt 2 ∧ triangle_area M.1 M.2 N.1 N.2 = Amin := sorry

end find_p_find_triangle_area_minimum_l454_454634


namespace simplify_fraction_l454_454746

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l454_454746


namespace not_all_x_heart_x_eq_0_l454_454542

def heartsuit (x y : ℝ) : ℝ := abs (x + y)

theorem not_all_x_heart_x_eq_0 :
  ¬ (∀ x : ℝ, heartsuit x x = 0) :=
by sorry

end not_all_x_heart_x_eq_0_l454_454542


namespace trains_crossing_time_l454_454446

/-- Define the length of the first train in meters -/
def length_train1 : ℚ := 200

/-- Define the length of the second train in meters -/
def length_train2 : ℚ := 150

/-- Define the speed of the first train in kilometers per hour -/
def speed_train1_kmph : ℚ := 40

/-- Define the speed of the second train in kilometers per hour -/
def speed_train2_kmph : ℚ := 46

/-- Define conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 1000 / 3600

/-- Calculate the relative speed in meters per second assuming both trains are moving in the same direction -/
def relative_speed_mps : ℚ := (speed_train2_kmph - speed_train1_kmph) * kmph_to_mps

/-- Calculate the combined length of both trains in meters -/
def combined_length : ℚ := length_train1 + length_train2

/-- Prove the time in seconds for the two trains to cross each other when moving in the same direction is 210 seconds -/
theorem trains_crossing_time :
  (combined_length / relative_speed_mps) = 210 := by
  sorry

end trains_crossing_time_l454_454446


namespace find_other_num_l454_454424

variables (a b : ℕ)

theorem find_other_num (h_gcd : Nat.gcd a b = 12) (h_lcm : Nat.lcm a b = 5040) (h_a : a = 240) :
  b = 252 :=
  sorry

end find_other_num_l454_454424


namespace polynomial_solution_l454_454113

theorem polynomial_solution (P : ℝ[X]) : 
  (∀ x : ℝ, (x - 1) * P.eval (x + 1) - (x + 2) * P.eval x = 0) ↔ 
  ∃ a : ℝ, P = a • (X^3 - X) :=
by 
  sorry

end polynomial_solution_l454_454113


namespace roll_even_sum_probability_l454_454215

/-- Probability that the sum of the numbers rolled by two players on a fair ten-sided die is even. --/
theorem roll_even_sum_probability :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
  let possible_rolls := outcomes.product outcomes in
  let even_sum (pair : Nat × Nat) := (pair.1 + pair.2) % 2 = 0 in
  (possible_rolls.count even_sum).toRat / possible_rolls.length.toRat = 1/2 :=
by
  sorry

end roll_even_sum_probability_l454_454215


namespace mu_is_integer_l454_454764

noncomputable def mu (n : ℕ) : ℂ :=
  ∑ k in (Finset.filter (λ k, Nat.gcd k n = 1) (Finset.range (n + 1))),
    Complex.exp (2 * k * Real.pi * Complex.I / n)

theorem mu_is_integer (n : ℕ) (hn : 0 < n) : ∃ m : ℤ, (mu n).re = m := 
by
  sorry

end mu_is_integer_l454_454764


namespace smallest_k_for_checkerboard_l454_454202

theorem smallest_k_for_checkerboard :
  ∃ (k : ℕ), k = 1006 ∧ 
  (∀ (board : ℕ → ℕ → Prop),
    (∀ (i j : ℕ), i < 2011 → j < 2011 → board i j → (∃ (r c : ℕ), r < 2011 ∧ c < 2011 ∧ board r c)) →
    (∃ (rows cols : Finset ℕ), rows.card = k ∧ cols.card = k ∧ 
     (∀ (i j : ℕ), board i j → ∃ (r c : ℕ), r ∈ rows ∧ c ∈ cols ∧ board r c))) :=
begin
  use 1006,
  split,
  { exact rfl },
  { intros board board_cond,
    -- theorem proof goes here
    sorry, 
  }

end smallest_k_for_checkerboard_l454_454202


namespace sum_of_diagonals_squared_l454_454019

-- Define the geometric setup and necessary conditions
variables {A B C D O P : Type} [inst : EuclideanGeometry O] 
          (cyclic_quad : CyclicQuadrilateral A B C D)
          (perp_diagonals : Perpendicular (AC) (BD))
          (center_O : Circumcenter O A B C D)
          (intersect_P : Intersection P (AC) (BD))
          (OP_length : ℝ)
          (R : ℝ)

-- Prove the sum of the squares of the diagonals equal to the desired expression
theorem sum_of_diagonals_squared : 
  ∀ (A B C D O P : Point) 
    (cyclic_quad : CyclicQuadrilateral A B C D)
    (perp_diagonals : Perpendicular (AC) (BD))
    (center_O : Circumcenter O A B C D)
    (intersect_P : Intersection P (AC) (BD))
    (OP_length R : ℝ), 
  AC^2 + BD^2 = 8 * R^2 - 4 * (OP_length)^2 := by
  sorry

end sum_of_diagonals_squared_l454_454019


namespace focus_through_line_l454_454943

def point (x y : ℝ) := (x, y)

def ellipse (a b c : ℝ) (e : ℝ) (center : point) (P : point) : Prop :=
  c = sqrt (a^2 - b^2) ∧
  e = c / a ∧
  center = (0, 0) ∧
  P = (0, -2)

def line_through (P A : point) (k : ℝ) : Prop :=
  ∃ x y, A = (x, k * x - 2)

def intersection_in_line (P A : point) (a b : ℝ) (k : ℝ) : Prop :=
  line_through P A k ∧
  ∃ x, A = (x, (2 * k^2 - 8)/(4 + k^2))

theorem focus_through_line (a b k1 k2 : ℝ) (P A B : point):
  ellipse a b (sqrt (a^2 - b^2)) (sqrt 3 / 2) (0, 0) P →
  intersection_in_line (0, -2) A a b k1 →
  intersection_in_line (0, -2) B a b k2 →
  k1 * k2 = 2 →
  let Q := (0, -6) in
  ∃ m, ∃ c, 
    (m * 0 + c = -6) ∧ 
    (∀ x y, (x, y) ∈ line_through P A k1 ∧ (x, y) ∈ line_through P B k2 → y = m * x + c) :=
sorry

end focus_through_line_l454_454943


namespace constants_solution_l454_454561

theorem constants_solution : ∀ (x : ℝ), x ≠ 0 ∧ x^2 ≠ 2 →
  (2 * x^2 - 5 * x + 1) / (x^3 - 2 * x) = (-1 / 2) / x + (2.5 * x - 5) / (x^2 - 2) := by
  intros x hx
  sorry

end constants_solution_l454_454561


namespace genuine_product_probability_l454_454850

-- Define the probabilities as constants
def P_second_grade := 0.03
def P_third_grade := 0.01

-- Define the total probability (outcome must be either genuine or substandard)
def P_substandard := P_second_grade + P_third_grade
def P_genuine := 1 - P_substandard

-- The statement to be proved
theorem genuine_product_probability :
  P_genuine = 0.96 :=
sorry

end genuine_product_probability_l454_454850


namespace total_plants_l454_454680

-- Define the initial number of each type of plant
def ferns : ℕ := 3
def palms : ℕ := 5
def succulents : ℕ := 7
def additional_plants : ℕ := 9

-- Define the total number of plants Justice initially has
def current_plants : ℕ := ferns + palms + succulents

-- Define the total number of plants Justice wants
def total_plants_desired : ℕ := current_plants + additional_plants

-- Prove that the total number of plants Justice wants in her home is 24
theorem total_plants (ferns palms succulents additional_plants : ℕ) (h1 : ferns = 3) (h2 : palms = 5) (h3 : succulents = 7) (h4 : additional_plants = 9) : total_plants_desired = 24 := by
  dsimp [total_plants_desired, current_plants]
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_plants_l454_454680


namespace statement_B_statement_C_statement_D_l454_454676

-- Definitions for the conditions
variables {A B C : ℝ} -- Angles in the triangle

-- Conditions: A triangle ABC
def is_triangle (A B C : ℝ) : Prop := A + B + C = π

-- Condition: Acute triangle (All angles less than 90 degrees)
def is_acute_triangle (A B C : ℝ) : Prop := A < π / 2 ∧ B < π / 2 ∧ C < π / 2

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

-- Theorem statements based on our translation
theorem statement_B (h : A > B) : sin A > sin B := sorry
theorem statement_C (h_triangle : is_triangle A B C) (h_acute : is_acute_triangle A B C): sin A > cos B := sorry
theorem statement_D (h : sin A > sin B) : A > B := sorry

#check statement_B
#check statement_C
#check statement_D

end statement_B_statement_C_statement_D_l454_454676


namespace scheduling_methods_l454_454319

theorem scheduling_methods:
  let arts := ["rites", "music", "archery", "charioteering", "calligraphy", "mathematics"] in
  (∃ (adj_case methods : ℕ), 
    (let combined_units := 5! * 2 in adj_case = combined_units) ∧
    (∃ (between_case methods : ℕ),
      let between_ways := 4 * 4! * 2 in between_case = between_ways) ∧
    (let total_ways := adj_case + between_case in methods = total_ways) ∧
    methods = 432) :=
sorry

end scheduling_methods_l454_454319


namespace no_diff_22_for_prime_products_l454_454208

theorem no_diff_22_for_prime_products (p1 p2 p3 p4 : ℕ) 
  (hp1_prime : Nat.Prime p1) (hp2_prime : Nat.Prime p2) (hp3_prime : Nat.Prime p3) (hp4_prime : Nat.Prime p4)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4)
  (h_ordered : p1 < p2 ∧ p2 < p3 ∧ p3 < p4) 
  (h_bound : p1 * p2 * p3 * p4 < 1995) :
  ∀ (d8 d9 : ℕ), d8 = p1 * p4 → d9 = p2 * p3 → d9 - d8 ≠ 22 := by
    sorry

end no_diff_22_for_prime_products_l454_454208


namespace semicircle_perimeter_calc_l454_454772

noncomputable def radius : ℝ := 21.005164601010506

noncomputable def perimeter_of_semicircle (r : ℝ) : ℝ :=
  real.pi * r + 2 * r

theorem semicircle_perimeter_calc : 
  perimeter_of_semicircle radius = 108.01915941002101 := 
sorry

end semicircle_perimeter_calc_l454_454772


namespace range_of_f_l454_454955

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1)

theorem range_of_f : set.range (f) = set.Icc (-1 : ℝ) (0 : ℝ) := 
sorry

end range_of_f_l454_454955


namespace geometric_ratio_of_arithmetic_sequence_l454_454691

theorem geometric_ratio_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h_nonzero : d ≠ 0) (h_geometric : (a 1 + 2 * d) * (a 1 + 2 * d) = a 1 * (a 1 + 6 * d)) :
  (a 3) / (a 1) = 2 :=
begin
  sorry
end

end geometric_ratio_of_arithmetic_sequence_l454_454691


namespace apples_purchased_l454_454793

variable (A : ℕ) -- Let A be the number of kg of apples purchased.

-- Conditions
def cost_of_apples (A : ℕ) : ℕ := 70 * A
def cost_of_mangoes : ℕ := 45 * 9
def total_amount_paid : ℕ := 965

-- Theorem to prove that A == 8
theorem apples_purchased
  (h : cost_of_apples A + cost_of_mangoes = total_amount_paid) :
  A = 8 := by
sorry

end apples_purchased_l454_454793


namespace population_net_increase_in_one_day_l454_454486

-- Define the birth rate and death rate conditions.
def birth_rate_per_2s : ℕ := 6
def death_rate_per_2s : ℕ := 3

-- Calculate the net increase per second.
def net_increase_per_second : ℝ := (birth_rate_per_2s - death_rate_per_2s) / 2

-- Define the number of seconds in one day.
def seconds_in_one_day : ℕ := 24 * 60 * 60

-- Prove the net increase in one day.
theorem population_net_increase_in_one_day :
  net_increase_per_second * seconds_in_one_day = 64800 := by
  sorry

end population_net_increase_in_one_day_l454_454486


namespace spratilish_9_letters_mod_1000_l454_454308

/--
In Spratilish, all words consist only of the letters M, P, Z, and O.
M and P are consonants, while O and Z are vowels.
A valid Spratilish word must have at least three consonants between any two vowels.
Prove that the number of valid 9-letter Spratilish words ≡ 704 (mod 1000).
-/
theorem spratilish_9_letters_mod_1000 : 
  let a := (fun n => 2 * (a_n n + c_n n)),
      b := (fun n => a_n n),
      c := (fun n => 2 * b_n n),
      a_3 := 8,
      b_3 := 0,
      c_3 := 4,
      a_4 := 2 * (a_3 + c_3),
      b_4 := a_3,
      c_4 := 2 * b_3,
      a_5 := 2 * (a_4 + c_4),
      b_5 := a_4,
      c_5 := 2 * b_4,
      a_6 := 2 * (a_5 + c_5),
      b_6 := a_5,
      c_6 := 2 * b_5,
      a_7 := 2 * (a_6 + c_6),
      b_7 := a_6,
      c_7 := 2 * b_6,
      a_8 := 2 * (a_7 + c_7),
      b_8 := a_7,
      c_8 := 2 * b_7,
      a_9 := 2 * (a_8 + c_8),
      b_9 := a_8,
      c_9 := 2 * b_8,
      res := a_9 % 1000 + b_9 % 1000 + c_9 % 1000
  in res % 1000 = 704 :=
by sorry

end spratilish_9_letters_mod_1000_l454_454308


namespace strictly_decreasing_interval_l454_454909

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem strictly_decreasing_interval :
  ∀ x y : ℝ, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y < x → f y < f x :=
by
  sorry

end strictly_decreasing_interval_l454_454909


namespace simplify_fraction_l454_454749

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l454_454749


namespace obtuse_angle_condition_l454_454930

def vector_length (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

theorem obtuse_angle_condition
  (a b : ℝ × ℝ × ℝ)
  (h1 : vector_length a = 2)
  (h2 : vector_length b = 1)
  (h3 : dot_product a b = 1) :
  ∀ λ : ℝ, (λ ∈ set.Ioo (-1 - real.sqrt 3) (-1 + real.sqrt 3)) ↔
            (dot_product (a + (λ, λ, λ) ⬝ b) ((λ, λ, λ) ⬝ a - 2 ⬝ b) < 0) :=
sorry

end obtuse_angle_condition_l454_454930


namespace call_duration_l454_454682

def initial_credit : ℝ := 30
def cost_per_minute : ℝ := 0.16
def remaining_credit : ℝ := 26.48

theorem call_duration :
  (initial_credit - remaining_credit) / cost_per_minute = 22 := 
sorry

end call_duration_l454_454682


namespace integer_triplets_sum_6_l454_454973

noncomputable def count_integer_triplets (sum : ℤ) (a_range b_range c_range : set ℤ) :=
  { t | t.1 + t.2 + t.3 = sum ∧ t.1 ∈ a_range ∧ t.2 ∈ b_range ∧ t.3 ∈ c_range }.to_finset.card

theorem integer_triplets_sum_6 : count_integer_triplets 6 {a | -1 ≤ a ∧ a ≤ 2} {b | 1 ≤ b ∧ b ≤ 4} {c | 1 ≤ c ∧ c ≤ 4} = 12 :=
by
  sorry

end integer_triplets_sum_6_l454_454973


namespace alcohol_solution_volume_l454_454495

theorem alcohol_solution_volume (V : ℝ) (h1 : 0.42 * V = 0.33 * (V + 3)) : V = 11 :=
by
  sorry

end alcohol_solution_volume_l454_454495


namespace count_digit_7_in_range_20_to_119_l454_454678

-- Define a function that counts the occurrences of digit 7 in a given range
def count_digit_7_in_range (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).map (λ n => a + n).sum (λ n =>
    (n.digits 10).count 7)

-- The actual theorem we want to prove
theorem count_digit_7_in_range_20_to_119 : count_digit_7_in_range 20 119 = 19 :=
  sorry

end count_digit_7_in_range_20_to_119_l454_454678


namespace non_juniors_play_sport_l454_454655

theorem non_juniors_play_sport (j n : ℕ) 
  (h1 : j + n = 600)
  (h2 : 0.5 * j.to_rat + 0.4 * n.to_rat = 312)
  (h3 : ∀ k, 0.5 * k.to_rat = 0.5 * k.to_rat)
  (h4 : ∀ m, 0.4 * m.to_rat = 0.4 * m.to_rat)
  (h5 : 52 * 600 = 31200) :
  (0.6 * n.to_rat).nat_abs = 72 :=
by sorry

end non_juniors_play_sport_l454_454655


namespace locus_of_vertex_l454_454927

-- The problem involves the geometric configuration of a parabola and tangent lines

-- Given a parabola y^2 = 4ax
def parabola (a x y : ℝ) : Prop := y^2 = 4 * a * x

-- Given two tangents intersecting at an angle θ
def tangents_intersecting_at_angle (k1 k2 θ : ℝ) : Prop :=
  θ ≠ 0 ∧ θ ≠ π ∧ tan θ = |k2 - k1| / (1 + k1 * k2)

-- The locus of the vertex of the angle of the tangents satisfies:
def locus_vertex_of_tangents (x y a θ : ℝ) : Prop :=
  (tan θ)^2 * x^2 - y^2 + 2 * a * (2 + (tan θ)^2) * x + a^2 * (tan θ)^2 = 0

-- Main theorem to state the final proof problem
theorem locus_of_vertex (a θ x y : ℝ) (h1 : parabola a x y) 
  (h2 : ∃ k1 k2 : ℝ, tangents_intersecting_at_angle k1 k2 θ) :
  locus_vertex_of_tangents x y a θ := 
  sorry

end locus_of_vertex_l454_454927


namespace cos_alpha_eq_three_fifths_max_min_values_f_l454_454237

theorem cos_alpha_eq_three_fifths (α : ℝ) (hα1 : α ∈ Ioo (π / 4) (π / 2)) 
  (hα2 : sin (α + π / 4) = (7 * sqrt 2) / 10) : 
  cos α = 3 / 5 :=
by sorry

theorem max_min_values_f (α : ℝ) (hα : sin (α + π / 4) = (7 * sqrt 2) / 10)
  (hα1 : α ∈ Ioo (π / 4) (π / 2)) : 
  (∃ x, (cos (2 * x) + (5 / 2) * sin α * sin x) = 3 / 2) ∧
  (∃ x, (cos (2 * x) + (5 / 2) * sin α * sin x) = -3) :=
by sorry

end cos_alpha_eq_three_fifths_max_min_values_f_l454_454237


namespace f_is_small_o_as_x_to_0_l454_454479

noncomputable def f : (Set.Ioo 0 1) → ℝ := sorry

lemma limit_f_zero (ε : ℝ) (hε : 0 < ε) : 
  ∃ δ > 0, ∀ ⦃x⦄, 0 < x → x < δ → |f ⟨x, sorry, sorry⟩| < ε := sorry

lemma f_minus_f_over_2 (ε : ℝ) (hε : 0 < ε) : 
  ∃ δ > 0, ∀ ⦃x⦄, 0 < x → x < δ → |(f ⟨x, sorry, sorry⟩ - f ⟨x / 2, sorry, sorry⟩) / x| < ε := sorry

theorem f_is_small_o_as_x_to_0 : 
  (∀ ε > 0, ∃ δ > 0, ∀ ⦃x⦄, 0 < x → x < δ → |f ⟨x, sorry, sorry⟩ / x| < ε) := 
begin
  sorry
end

end f_is_small_o_as_x_to_0_l454_454479


namespace math_problem_l454_454601

-- Define the conditions for the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms
def sum_first_n_terms (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * 2

-- Define the geometric sequence condition
def geometric_seq (S : ℕ → ℤ) : Prop :=
  (S 2) ^ 2 = S 1 * S 4

-- Define b_n
def b_seq (a : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, b n = (-1) ^ (n - 1) * 4 * n / (a n * a (n + 1))

-- Define T_n
def T_seq (b T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = if n % 2 = 0 then 2 * n / (2 * n + 1)
                  else (2 * n + 2) / (2 * n + 1)

-- Lean statement with the given conditions and goals
theorem math_problem (a S : ℕ → ℤ) (b T : ℕ → ℤ) :
  arithmetic_seq a 2 →
  sum_first_n_terms a S →
  geometric_seq S →
  b_seq a b →
  T_seq b T :=
begin
  sorry
end

end math_problem_l454_454601


namespace total_weight_of_three_packages_l454_454708

theorem total_weight_of_three_packages (a b c d : ℝ)
  (h1 : a + b = 162)
  (h2 : b + c = 164)
  (h3 : c + a = 168) :
  a + b + c = 247 :=
sorry

end total_weight_of_three_packages_l454_454708


namespace fraction_equivalent_of_repeating_decimal_l454_454189

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454189


namespace number_of_people_entered_l454_454841

theorem number_of_people_entered (total_placards : ℕ) (placards_per_person : ℕ) (num_people : ℕ) 
  (h0 : total_placards = 4634) (h1 : placards_per_person = 2) : num_people = total_placards / placards_per_person :=
by
  have h2 : num_people = 4634 / 2, from sorry,
  exact h2

end number_of_people_entered_l454_454841


namespace value_of_M_after_subtracting_10_percent_l454_454293

-- Define the given conditions and desired result formally in Lean 4
theorem value_of_M_after_subtracting_10_percent (M : ℝ) (h : 0.25 * M = 0.55 * 2500) :
  M - 0.10 * M = 4950 :=
by
  sorry

end value_of_M_after_subtracting_10_percent_l454_454293


namespace num_integers_satisfying_inequality_l454_454274

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l454_454274


namespace smallest_three_star_is_102_l454_454491

def is_three_star (n : ℕ) : Prop :=
  (n >= 100) ∧ (n <= 999) ∧ (∃ (p1 p2 p3 : ℕ), (nat.prime p1) ∧ (nat.prime p2) ∧ (nat.prime p3) ∧ (p1 ≠ p2) ∧ (p2 ≠ p3) ∧ (p1 ≠ p3) ∧ (n = p1 * p2 * p3))

theorem smallest_three_star_is_102 : (∀ n, is_three_star n → n ≥ 102) ∧ is_three_star 102 :=
by sorry

end smallest_three_star_is_102_l454_454491


namespace problem_x_sq_plus_y_sq_l454_454296

variables {x y : ℝ}

theorem problem_x_sq_plus_y_sq (h₁ : x - y = 12) (h₂ : x * y = 9) : x^2 + y^2 = 162 := 
sorry

end problem_x_sq_plus_y_sq_l454_454296


namespace max_sphere_radius_in_glass_cross_section_l454_454661

theorem max_sphere_radius_in_glass_cross_section :
  let r := (3 * (2^(1/3))) / 4 in
  ∃ (x : ℝ), (r = 1/2 * (x^4 + 1/x^2) ∧ x = 1/(2^(1/6))) :=
begin
  sorry
end

end max_sphere_radius_in_glass_cross_section_l454_454661


namespace position_of_12340_in_permutations_of_01234_l454_454866

theorem position_of_12340_in_permutations_of_01234 : 
  let digits := [0, 1, 2, 3, 4];
  let perms := list.permutations digits;
  let five_digit_perms := list.filter (λ l, list.length l = 5) perms;
  let sorted_perms := list.sort (λ l1 l2, nat.lt (nat_of_digits 10 l1) (nat_of_digits 10 l2)) five_digit_perms;
  list.index_of [1, 2, 3, 4, 0] sorted_perms + 1 = 10 := 
  sorry

end position_of_12340_in_permutations_of_01234_l454_454866


namespace parabola_min_perimeter_l454_454301

noncomputable def focus_of_parabola (p : ℝ) (hp : p > 0) : ℝ × ℝ :=
(1, 0)

noncomputable def A : ℝ × ℝ := (3, 2)

noncomputable def is_on_parabola (P : ℝ × ℝ) (p : ℝ) : Prop :=
P.2 ^ 2 = 2 * p * P.1

noncomputable def area_of_triangle (A P F : ℝ × ℝ) : ℝ :=
0.5 * abs (A.1 * (P.2 - F.2) + P.1 * (F.2 - A.2) + F.1 * (A.2 - P.2))

noncomputable def perimeter (A P F : ℝ × ℝ) : ℝ := 
abs (A.1 - P.1) + abs (A.1 - F.1) + abs (P.1 - F.1)

theorem parabola_min_perimeter 
  {p : ℝ} (hp : p > 0)
  (A : ℝ × ℝ) (ha : A = (3,2))
  (P : ℝ × ℝ) (hP : is_on_parabola P p)
  {F : ℝ × ℝ} (hF : F = focus_of_parabola p hp)
  (harea : area_of_triangle A P F = 1)
  (hmin : ∀ P', is_on_parabola P' p → 
    perimeter A P' F ≥ perimeter A P F) :
  abs (P.1 - F.1) = 5/2 :=
sorry

end parabola_min_perimeter_l454_454301


namespace find_circle_equation_l454_454253

-- Define coordinates of the center and the radius of the circle
variables {a : ℝ}
def C := (a, 3 * a)
def r := abs (3 * a)

-- Define the conditions
def tangent_to_x_axis (r : ℝ) : Prop := r = abs (3 * a)

def center_on_line_3x_minus_y (C : ℝ × ℝ) : Prop := 3 * C.1 = C.2

def intersection_area (C : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, 
  A = (0, -a) ∧ B = (2a, a) ∧
  1 / 2 * abs ((A.1 - C.1) * (B.2 - C.2) - (A.2 - C.2) * (B.1 - C.1)) = sqrt 14

-- Theorem stating the equation of the circle
theorem find_circle_equation (h1 : tangent_to_x_axis r)
  (h2 : center_on_line_3x_minus_y C)
  (h3 : intersection_area C r) :
  (C = (1, 3) ∧ r = 3 ∧ ((x - 1)^2 + (y - 3)^2 = 9)) ∨ 
  (C = (-1, -3) ∧ r = 3 ∧ ((x + 1)^2 + (y + 3)^2 = 9)) :=
by
s sorry

end find_circle_equation_l454_454253


namespace a_increases_on_interval_l454_454017

def f (x : ℝ) : ℝ := sorry  -- Define f(x) according to the problem context
def φ (x : ℝ) : ℝ := sorry  -- Define φ(x) according to the problem context
def a (x : ℝ) : ℝ := sorry  -- Define a(x) probably in terms of f(x) and φ(x)

theorem a_increases_on_interval (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 < x2) (h3 : x2 ≤ 0.5) :
  a x1 < a x2 :=
begin
  sorry  -- The proof will go here
end

end a_increases_on_interval_l454_454017


namespace particle_returns_to_origin_l454_454513

/-- A particle undergoes a specific transformation defined by rotation and translation. 
    We prove that after 120 moves, the particle's position is (6, 0). -/
theorem particle_returns_to_origin :
  let ω := Complex.cis (Real.pi / 6) in
  let move (z : ℂ) := ω * z + 8 in
  (nat.iterate move 120 6) = 6 :=
by
  let ω := Complex.cis (Real.pi / 6)
  let move := λ z, ω * z + 8
  sorry

end particle_returns_to_origin_l454_454513


namespace inequality_hold_l454_454260

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l454_454260


namespace kx2_kx_1_pos_l454_454492

theorem kx2_kx_1_pos (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end kx2_kx_1_pos_l454_454492


namespace sum_of_primes_eq_24_l454_454939

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

variable (a b c : ℕ)

theorem sum_of_primes_eq_24 (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
    (h4 : a * b + b * c = 119) : a + b + c = 24 :=
sorry

end sum_of_primes_eq_24_l454_454939


namespace eight_digit_palindromic_count_l454_454845

-- Define a palindromic number according to the problem conditions
def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

-- Definition of the count of k-digit palindromic numbers
def count_palindromic_numbers (k : ℕ) : ℕ :=
  if k % 2 == 0 then 9 * 10 ^ (k / 2 - 1)
  else if k == 1 then 10
  else 9 * 10 ^ (k / 2)

-- Theorem statement
theorem eight_digit_palindromic_count : count_palindromic_numbers 8 = 9000 :=
by
  sorry

end eight_digit_palindromic_count_l454_454845


namespace min_value_collinear_l454_454232

theorem min_value_collinear (x y : ℝ) (h₁ : 2 * x + 3 * y = 3) (h₂ : 0 < x) (h₃ : 0 < y) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_collinear_l454_454232


namespace pizza_slices_l454_454357

theorem pizza_slices (num_people : ℕ) (slices_per_person : ℕ) (num_pizzas : ℕ) (total_slices : ℕ) :
  num_people = 6 → 
  slices_per_person = 4 → 
  num_pizzas = 3 → 
  total_slices = num_people * slices_per_person →
  total_slices / num_pizzas = 8 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have h : total_slices = 24 := h4
  rw h
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slices_l454_454357


namespace mat_is_closer_l454_454826

variable {c m p : ℝ}

-- Condition for Mat
def mat_condition (c m : ℝ) : Prop :=
  m + (3/4) * m = c

-- Condition for Pat
def pat_condition (c p : ℝ) : Prop :=
  2 * p + (2/3) * p = c

-- Verification that Mat is closer to the cottage
theorem mat_is_closer (hc1 : mat_condition c m) (hc2 : pat_condition c p) : m > p :=
by
  have h1 : m = (4/7) * c, from sorry,
  have h2 : p = (3/8) * c, from sorry,
  rw [h1, h2],
  exact sorry

end mat_is_closer_l454_454826


namespace break_even_handles_l454_454385

variables (X : ℕ)
constants (fixed_cost : ℝ) (variable_cost_per_handle : ℝ) (selling_price_per_handle : ℝ)

def break_even (fixed_cost variable_cost_per_handle selling_price_per_handle : ℝ) (X : ℕ) : Prop :=
  fixed_cost + variable_cost_per_handle * X = selling_price_per_handle * X

theorem break_even_handles :
  break_even 7640 0.60 4.60 1910 :=
by
  unfold break_even
  norm_num
  sorry

end break_even_handles_l454_454385


namespace value_of_fraction_l454_454757

theorem value_of_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (b / c = 2005) ∧ (c / b = 2005)) : (b + c) / (a + b) = 2005 :=
by
  sorry

end value_of_fraction_l454_454757


namespace sum_of_cubes_of_roots_l454_454593

theorem sum_of_cubes_of_roots (r1 r2 r3 : ℂ) (h1 : r1 + r2 + r3 = 3) (h2 : r1 * r2 + r1 * r3 + r2 * r3 = 0) (h3 : r1 * r2 * r3 = -1) : 
  r1^3 + r2^3 + r3^3 = 24 :=
  sorry

end sum_of_cubes_of_roots_l454_454593


namespace problem_scores_ordering_l454_454914

variable {J K L R : ℕ}

theorem problem_scores_ordering (h1 : J > K) (h2 : J > L) (h3 : J > R)
                                (h4 : L > min K R) (h5 : R > min K L)
                                (h6 : (J ≠ K) ∧ (J ≠ L) ∧ (J ≠ R) ∧ (K ≠ L) ∧ (K ≠ R) ∧ (L ≠ R)) :
                                K < L ∧ L < R :=
sorry

end problem_scores_ordering_l454_454914


namespace find_initial_investment_l454_454707

def investment_problem (x : ℝ) (final_amount : ℝ) (years : ℕ) (tripling_period : ℕ) : Prop :=
  x = 8 / 100 → -- Interest rate of 8%
  final_amount = 13500 → -- Final amount after 28 years is $13500
  years = 28 → -- Investment duration of 28 years
  tripling_period = 112 / x.toNat → -- Formula for tripling period
  ∃ (P : ℝ), P * (3:ℝ)^(years / tripling_period) = final_amount

theorem find_initial_investment : investment_problem 0.08 13500 28 14 :=
by {
  -- The statement above needs the proof. For the question, you end with sorry.
  sorry
}

end find_initial_investment_l454_454707


namespace problem1_problem2_l454_454633

-- Definitions according to the problem conditions
def parabola (p : ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def line (a b c : ℝ) := {xy : ℝ × ℝ | a * xy.1 + b * xy.2 + c = 0}

-- Problem (1): Prove that p is 2 given |AB| = 4sqrt15.
theorem problem1 (p : ℝ) (h1 : p > 0) (A B : ℝ × ℝ)
  (hA : A ∈ parabola p) (hB : B ∈ parabola p) (hL : A ∈ line 1 (-2) 1)
  (hL' : B ∈ line 1 (-2) 1) (hAB : dist A B = 4 * real.sqrt 15) :
  p = 2 :=
sorry

-- Problem (2): Prove that the minimum area of the triangle MNF is 12 - 8sqrt2.
theorem problem2 (M N : ℝ × ℝ) (p : ℝ) (h1 : p > 0)
  (hM : M ∈ parabola p) (hN : N ∈ parabola p)
  (F : ℝ × ℝ) (hF : is_focus F p)
  (hMF_NF : (M.1 - F.1) * (N.1 - F.1) + M.2 * N.2 = 0) :
  ∃ area : ℝ, (∀ a', triangle_area M N F a' → a' ≥ area) ∧ area = 12 - 8 * real.sqrt 2 :=
sorry

end problem1_problem2_l454_454633


namespace incorrect_inequality_l454_454580

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end incorrect_inequality_l454_454580


namespace prove_altSum_2011_l454_454455

def altSum (n : ℕ) : ℤ :=
  ∑ k in Finset.range n.succ, (-1) ^ k

theorem prove_altSum_2011 : altSum 2011 = 1 := by
  sorry

end prove_altSum_2011_l454_454455


namespace ladder_base_distance_l454_454045

theorem ladder_base_distance (h : real) (L : real) (H : real) (B : real) 
  (h_eq : h = 12) (L_eq : L = 15) (h_sq_plus_b_sq : h^2 + B^2 = L^2) : 
  B = 9 :=
by
  rw [h_eq, L_eq] at h_sq_plus_b_sq
  sorry

end ladder_base_distance_l454_454045


namespace repeating_decimal_product_l454_454536

theorem repeating_decimal_product :
  let s := 0.123123123123 : ℚ in
  s * 9 = (41 / 37 : ℚ) :=
by sorry

end repeating_decimal_product_l454_454536


namespace ratio_MB_to_ABQ_l454_454318

-- Define the relevant angles and conditions in the problem
variables (ABC P B Q M : Type) [angleABC : angle ABC] [angleABQ : angle ABQ] [anglePBQ : angle PBQ] [angleMBQ : angle MBQ]

-- Define the bisectors
def is_bisector (ray1 ray2 angle : Type) : Prop :=
sorry -- Define the concept of bisecting an angle

-- Given conditions
axiom bisects_BP_ABC : is_bisector BP B (angle ABC)
axiom bisects_BQ_ABC : is_bisector BQ B (angle ABC)
axiom bisects_BM_PBQ : is_bisector BM B (angle PBQ)

-- Main theorem to prove
theorem ratio_MB_to_ABQ : (measure angleMBQ) / (measure angleABQ) = 1 / 4 :=
sorry

end ratio_MB_to_ABQ_l454_454318


namespace snowman_total_volume_l454_454080

-- Define the radius of each snowball
def r1 : ℝ := 4
def r2 : ℝ := 6
def r3 : ℝ := 8
def r4 : ℝ := 10

-- Define the volume formula for a sphere
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the total volume of the snowman consisting of four snowballs
def total_volume : ℝ := volume r1 + volume r2 + volume r3 + volume r4

-- State the theorem that proves the total volume is as calculated in the solution
theorem snowman_total_volume :
  total_volume = (7168 / 3) * Real.pi := by
  sorry

end snowman_total_volume_l454_454080


namespace rainfall_second_week_l454_454022

theorem rainfall_second_week (x : ℝ) 
  (h1 : x + 1.5 * x = 25) :
  1.5 * x = 15 :=
by
  sorry

end rainfall_second_week_l454_454022


namespace cylinder_surface_area_ratio_l454_454302

theorem cylinder_surface_area_ratio (A : ℝ) (hA : A > 0) :
    let S_lateral := A^2
    let r := A / (2 * real.pi)
    let S_bases := 2 * real.pi * r^2
    let S_total := S_lateral + S_bases
  in S_total / S_lateral = (1 + 2 * real.pi) / (2 * real.pi) :=
sorry

end cylinder_surface_area_ratio_l454_454302


namespace fraction_repeating_decimal_l454_454181

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454181


namespace degree_of_resulting_poly_l454_454111

-- Define the polynomials involved in the problem
noncomputable def poly_1 : Polynomial ℝ := 3 * Polynomial.X ^ 5 + 2 * Polynomial.X ^ 3 - Polynomial.X - 16
noncomputable def poly_2 : Polynomial ℝ := 4 * Polynomial.X ^ 11 - 8 * Polynomial.X ^ 8 + 6 * Polynomial.X ^ 5 + 35
noncomputable def poly_3 : Polynomial ℝ := (Polynomial.X ^ 2 + 4) ^ 8

-- Define the resulting polynomial
noncomputable def resulting_poly : Polynomial ℝ :=
  poly_1 * poly_2 - poly_3

-- The goal is to prove that the degree of the resulting polynomial is 16
theorem degree_of_resulting_poly : resulting_poly.degree = 16 := 
sorry

end degree_of_resulting_poly_l454_454111


namespace determine_analytical_expression_of_f_maximum_and_minimum_of_g_l454_454612

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*a^2*x - a^3

noncomputable def g (x : ℝ) : ℝ := (x - 1)^3 - 2*x^2

theorem determine_analytical_expression_of_f :
  ∀ (a : ℝ), (f a (1 + x) = -f a (1 - x)) → a = 1 :=
by
  intros a h
  have symm : ∀ x, f a (1 + x) = -f a (1 - x) := by assumption
  sorry

theorem maximum_and_minimum_of_g : 
  ∀ (c : ℝ), c = (-1) ∨ c = 1 ∨ c = (1 / 3) →
  g (-1) = -10 ∧ g (1) = -2 ∧ g (1 / 3) = - 14 / 27 :=
by
  intros c hc
  cases hc
  . simp [g]; linarith
  . cases hc
    . simp [g]; linarith
    . simp [g]; linarith
  sorry

end determine_analytical_expression_of_f_maximum_and_minimum_of_g_l454_454612


namespace angles_sum_132_l454_454672

theorem angles_sum_132
  (D E F p q : ℝ)
  (hD : D = 38)
  (hE : E = 58)
  (hF : F = 36)
  (five_sided_angle_sum : D + E + (360 - p) + 90 + (126 - q) = 540) : 
  p + q = 132 := 
by
  sorry

end angles_sum_132_l454_454672


namespace problem_statement_l454_454812

theorem problem_statement : 
  (∃ n : ℕ, ∃ n_fact : n = 17, (n! * (n + 1)!) / 2 = k * k) := 
by 
  -- Definitions
  let n := 17
  have h1 : (n! * (n + 1)!) / 2 = (17! * 18!) / 2 := by refl
  let k := 3  -- Because 9 = 3^2
  use 17, 3
  -- The statement asserts we have such n and k
  sorry

end problem_statement_l454_454812


namespace calculate_average_speed_l454_454642

-- Define the conditions
def segment1_distance : ℚ := 40
def segment1_speed : ℚ := 15

def segment2_distance : ℚ := 60
def segment2_speed : ℚ := 30

def segment3_distance : ℚ := 100
def segment3_speed : ℚ := 45

def segment4_distance : ℚ := 50
def segment4_speed : ℚ := 60

-- Define the expected average speed
def expected_average_speed : ℚ := 4500 / 139

theorem calculate_average_speed :
  let total_distance := segment1_distance + segment2_distance + segment3_distance + segment4_distance in
  let total_time :=
    (segment1_distance / segment1_speed) +
    (segment2_distance / segment2_speed) +
    (segment3_distance / segment3_speed) +
    (segment4_distance / segment4_speed) in
  total_time = 139 / 18 →  -- This helps to match the sum of time fractions correctly
  total_distance / total_time ≈ expected_average_speed := by
  intros total_distance total_time h_total_time
  sorry

end calculate_average_speed_l454_454642


namespace true_propositions_l454_454437

theorem true_propositions :
  (∀ a b : ℝ, (a^2 + b^2 = 0 → a = 0 ∧ b = 0)) ∧
  (∀ q : ℝ, (q ≤ 1 → ∃ x : ℝ, x^2 + 2 * x + q = 0)) →
  (true_prop := 1 ∧ true_prop := 3) :=
by
  sorry

end true_propositions_l454_454437


namespace find_p_find_triangle_area_minimum_l454_454635

-- Define the parabola and line equations and their intersection points.
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
noncomputable def line_eq (x y : ℝ) := x - 2 * y + 1 = 0
noncomputable def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (p : ℝ) :
  (∃ (A B : ℝ × ℝ), parabola_eq p A.1 A.2 ∧ parabola_eq p B.1 B.2 ∧ 
   line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ 
   distance A.1 A.2 B.1 B.2 = 4 * real.sqrt 15) → p = 2 := sorry

-- Define conditions for the second part of the problem
noncomputable def focus_x := 1
noncomputable def focus_y := 0
noncomputable def mf_dot_nf_eq_zero (x1 y1 x2 y2 : ℝ) := 
  (x1 - focus_x) * (x2 - focus_x) + y1 * y2 = 0
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) := 
  1 / 2 * abs ((x1 - focus_x) * (y2 - focus_y) - (x2 - focus_x) * (y1 - focus_y))

theorem find_triangle_area_minimum (M N : ℝ × ℝ) :
  (parabola_eq 2 M.1 M.2 ∧ parabola_eq 2 N.1 N.2 ∧ 
  mf_dot_nf_eq_zero M.1 M.2 N.1 N.2) → 
  ∃ (Amin : ℝ), Amin = 12 - 8 * real.sqrt 2 ∧ triangle_area M.1 M.2 N.1 N.2 = Amin := sorry

end find_p_find_triangle_area_minimum_l454_454635


namespace speed_against_current_l454_454060

theorem speed_against_current (V_m V_c : ℕ) (h1 : V_m + V_c = 20) (h2 : V_c = 3) : V_m - V_c = 14 :=
by 
  sorry

end speed_against_current_l454_454060


namespace find_b_l454_454099

theorem find_b (a u v w : ℝ) (b : ℝ)
  (h1 : ∀ x : ℝ, 12 * x^3 + 7 * a * x^2 + 6 * b * x + b = 0 → (x = u ∨ x = v ∨ x = w))
  (h2 : 0 < u ∧ 0 < v ∧ 0 < w)
  (h3 : u ≠ v ∧ v ≠ w ∧ u ≠ w)
  (h4 : Real.log u / Real.log 3 + Real.log v / Real.log 3 + Real.log w / Real.log 3 = 3):
  b = -324 := 
sorry

end find_b_l454_454099


namespace find_ff_third_find_zeros_f_l454_454952

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then real.log x / real.log 3 else x^2 + 2 * x

theorem find_ff_third :
  f (f (1 / 3)) = -1 :=
by
  sorry

theorem find_zeros_f :
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 1 :=
by
  sorry

end find_ff_third_find_zeros_f_l454_454952


namespace find_angle_A_find_b_and_c_l454_454993

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : ∀ (u v : ℝ × ℝ), u = (sqrt 3 * a, c) → v = (1 + cos A, sin C) → u = k • v)
  : A = π / 3 := 
by
  sorry

theorem find_b_and_c (a b c : ℝ)
  (h1 : 3 * b * c = 16 - a ^ 2)
  (h2 : 1 / 2 * b * c * sin (π / 3) = sqrt 3 / 2)
  : b = 2 ∧ c = 2 :=
by
  sorry


end find_angle_A_find_b_and_c_l454_454993


namespace probability_four_or_more_same_value_l454_454203

theorem probability_four_or_more_same_value :
  let dice := [1, 2, 3, 4, 5, 6] in
  let num_dice := 5 in
  let total_outcomes := 6 ^ num_dice in
  let prob_case1 := 1 * (1 / 6) ^ 4 in
  let prob_case2 := 5 * (1 / 6 ^ 3) * (5 / 6) in
  let total_prob := prob_case1 + prob_case2 in
  total_prob = (13 / 648) :=
by
  sorry

end probability_four_or_more_same_value_l454_454203


namespace complex_problem_1_complex_problem_2_l454_454037

theorem complex_problem_1 : 
  (1 - complex.i + (2 + real.sqrt 5 * complex.i)) / complex.i = 
  real.sqrt 5 - 1 - 3 * complex.i :=
sorry

theorem complex_problem_2 :
  ∀ (m : ℝ), 
  (let Z := (2 * m^2 + m - 1 : ℂ) + (4 * m^2 - 8 * m + 3) * complex.i
   in (m > 1 / 2 ∧ m < 3 / 2) ↔ (2 * m^2 + m - 1 > 0 ∧ 4 * m^2 - 8 * m + 3 < 0)) :=
sorry

end complex_problem_1_complex_problem_2_l454_454037


namespace find_original_number_l454_454844

variable (x : ℝ)

def tripled := 3 * x
def doubled := 2 * tripled
def subtracted := doubled - 9
def trebled := 3 * subtracted

theorem find_original_number (h : trebled = 90) : x = 6.5 := by
  sorry

end find_original_number_l454_454844


namespace shanghai_world_expo_l454_454861

theorem shanghai_world_expo (n : ℕ) (total_cost : ℕ) 
  (H1 : total_cost = 4000)
  (H2 : n ≤ 30 → total_cost = n * 120)
  (H3 : n > 30 → total_cost = n * (120 - 2 * (n - 30)) ∧ (120 - 2 * (n - 30)) ≥ 90) :
  n = 40 := 
sorry

end shanghai_world_expo_l454_454861


namespace main_theorem_l454_454603

variable (A B C K L M O : Type)
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space K]
variable [metric_space L]
variable [metric_space M]
variable [metric_space O]

variables (P Q R : A → A → A → Prop) -- Point P is the orthocenter for triangle defined by points (A,B,C)

variables (orthocenters_coincide : Prop → Prop → Prop) -- Definition of coinciding orthocenters

-- Definitions of collinear points and equal ratios
def collinear (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ r1 r2 r3 : ℝ, AK / KB = r1 ∧ BL / LC = r2 ∧ CM / MA = r3 ∧ r1 = r2 ∧ r2 = r3

-- Definition stating necessary and sufficient condition for the orthocenters of the two triangles to coincide is that the triangle ABC is equilateral.
def ortho_condition_iff_equal_triangle : Prop :=
  (orthocenters_coincide (P A B C) (P K L M)) ↔ (A = B ∧ B = C ∧ C = A) -- Equivalent to triangle ABC being equilateral

theorem main_theorem (H : ortho_condition_iff_equal_triangle) : collinear (A B C) →
  (orthocenters_coincide (P A B C) (P K L M)) ↔ (A = B ∧ B = C ∧ C = A) :=
sorry

end main_theorem_l454_454603


namespace fraction_of_women_married_l454_454875

theorem fraction_of_women_married
  (total_employees : ℕ)
  (women_pct : ℚ)
  (married_pct : ℚ)
  (single_men_fraction : ℚ) :
  women_pct = 64 / 100 →
  married_pct = 60 / 100 →
  single_men_fraction = 2 / 3 →
  (let men = total_employees * (1 - women_pct),
       single_men = men * single_men_fraction,
       married_men = men - single_men,
       married_employees = total_employees * married_pct,
       married_women = married_employees - married_men,
       women = total_employees * women_pct in
   married_women / women = 3 / 4) :=
begin
  intros h_women_pct h_married_pct h_single_men_fraction,
  have h1 : women_pct * total_employees = 64,
  { rw h_women_pct, ring },
  have h2 : married_pct * total_employees = 60,
  { rw h_married_pct, ring },
  have h3 : (1 - women_pct) * total_employees = 36,
  { rw h_women_pct, ring },
  have h4 : (1 - women_pct) * total_employees * single_men_fraction = 24,
  { rw [h_women_pct, h_single_men_fraction], ring },
  have h5 : (1 - women_pct) * total_employees - 
            ((1 - women_pct) * total_employees * single_men_fraction) = 12,
  { rw [h_women_pct, h_single_men_fraction], ring },
  have h6 : 60 - 12 = 48,
  { ring },
  have h7 : 48 / 64 = 3 / 4,
  { norm_num },
  exact h7,
end

end fraction_of_women_married_l454_454875


namespace equation_has_four_integer_solutions_l454_454018

theorem equation_has_four_integer_solutions :
  ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℤ,
    (x1 - 5) * (y1 - 2) = 11.11 ∧
    (x2 - 5) * (y2 - 2) = 11.11 ∧
    (x3 - 5) * (y3 - 2) = 11.11 ∧
    (x4 - 5) * (y4 - 2) = 11.11 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧
    (x3, y3) ≠ (x4, y4) :=
by sorry

end equation_has_four_integer_solutions_l454_454018


namespace correct_operation_l454_454813

theorem correct_operation : 
  ∀ (m n a : ℝ),
  ¬ (a ^ (-2) * a ^ (-3) = a ^ (-6)) ∧ 
  ¬ ((m - n) ^ 2 = m ^ 2 - n ^ 2) ∧
  ¬ ((2 * a ^ 3) ^ 3 = 8 * a ^ 6) ∧
  (2 * m + 1 / 2) * (2 * m - 1 / 2) = 4 * m ^ 2 - 1 / 4 := 
by {
  intro m n a,
  split,
  -- Proof for ¬ (a ^ (-2) * a ^ (-3) = a ^ (-6))
  { intro h, sorry },
  split,
  -- Proof for ¬ ((m - n) ^ 2 = m ^ 2 - n ^ 2)
  { intro h, sorry },
  split,
  -- Proof for ¬ ((2 * a ^ 3) ^ 3 = 8 * a ^ 6)
  { intro h, sorry },
  -- Proof for (2 * m + 1 / 2) * (2 * m - 1 / 2) = 4 * m ^ 2 - 1 / 4
  { exact sorry }
}

end correct_operation_l454_454813


namespace simplify_fraction_l454_454739

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454739


namespace parallel_lines_sufficient_but_not_necessary_l454_454238

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 ↔ ((ax + y - 1 = 0) ∧ (x + ay + 1 = 0) → False)) := 
sorry

end parallel_lines_sufficient_but_not_necessary_l454_454238


namespace dot_product_eq_six_angle_between_vectors_eq_sixty_magnitude_of_difference_eq_sqrt_thirteen_l454_454273

variable (a b : ℝ^3)

axiom magnitude_a : ‖a‖ = 4
axiom magnitude_b : ‖b‖ = 3
axiom condition : (2 • a + 3 • b) ⬝ (2 • a - b) = 61

theorem dot_product_eq_six : a ⬝ b = 6 := sorry

theorem angle_between_vectors_eq_sixty : real.angle_between a b = real.pi / 3 := sorry

theorem magnitude_of_difference_eq_sqrt_thirteen : ‖a - b‖ = real.sqrt 13 := sorry

end dot_product_eq_six_angle_between_vectors_eq_sixty_magnitude_of_difference_eq_sqrt_thirteen_l454_454273


namespace plane_through_KL_divides_tetrahedron_l454_454998

variables (A B C D : Type) [Point A] [Point B] [Point C] [Point D]

def midpoint (P Q : Type) [Point P] [Point Q] : Type := sorry

def is_midpoint (X P Q : Type) [Point X] [Point P] [Point Q] : Prop := sorry

variables (K L : Type) [Point K] [Point L]

axiom K_midpoint : is_midpoint K A B
axiom L_midpoint : is_midpoint L C D

theorem plane_through_KL_divides_tetrahedron
  (P Q T T' : Type) [Point P] [Point Q] [Tetrahedron T] [Set T] [Tetrahedron T'] [Set T']
  (plane_contains_KL : ∀ (p : Plane), contains p K → contains p L → divides p T T')
  : volume T = volume T' :=
sorry

end plane_through_KL_divides_tetrahedron_l454_454998


namespace unit_circle_inequality_l454_454714

noncomputable def points_on_unit_circle (n : ℕ) (h_n : n ≥ 2) : ℕ → ℂ :=
-- Assume this function is a valid representation of n distinct points on the unit circle.
sorry

def distance_product (z_k : ℂ) (points : ℕ → ℂ) (n : ℕ) : ℂ := 
if h : n > 0 then
  ∏ i in (finset.univ : finset (fin n)).erase (finite_idx z_k points), (z_k - points i)
else
  1

theorem unit_circle_inequality (n : ℕ) (h_n : n ≥ 2) :
  let points := points_on_unit_circle n h_n in
  let d := λ k, distance_product (points k) points n in 
  1 ≤ (finset.univ : finset (fin n)).sum (λ k, (1 / (d k)).re) :=
sorry

end unit_circle_inequality_l454_454714


namespace find_k_l454_454344

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else sorry

axiom seq_base : seq 1 = 2
axiom seq_recur (m n : ℕ) : seq (m + n) = seq m * seq n

-- Given condition for sum
axiom sum_condition (k : ℕ) : 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5

-- The statement to prove
theorem find_k : ∃ k : ℕ, 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5 ∧ k = 4 :=
by {
  sorry -- proof is not required
}

end find_k_l454_454344


namespace find_matrix_N_l454_454564

noncomputable def N : Matrix (fin 2) (fin 2) ℚ := ![
  #[46 / 7, -58 / 7],
  #[-28 / 7, 35 / 7]
]

def A : Matrix (fin 2) (fin 2) ℚ := ![
  #[2, -5],
  #[4, -3]
]

def B : Matrix (fin 2) (fin 2) ℚ := ![
  #[-20, -8],
  #[12, 5]
]

theorem find_matrix_N : N * A = B :=
by {
  sorry
}

end find_matrix_N_l454_454564


namespace div_pow_eq_l454_454921

theorem div_pow_eq {a : ℝ} (h : a ≠ 0) : a^3 / a^2 = a :=
sorry

end div_pow_eq_l454_454921


namespace find_z_and_ratio_find_trajectory_l454_454614

variable (z : Complex) (z1 : Complex)
variable (x y : Real)

def is_conjugate (z : Complex) (z_conjugate : Complex) : Prop :=
  z_conjugate.re = z.re ∧ z_conjugate.im = -z.im

def given_condition (z : Complex) (z_conjugate : Complex) : Prop :=
  (1 + 2 * Complex.i) * z_conjugate = 4 + 3 * Complex.i

theorem find_z_and_ratio
  (z_conjugate : Complex)
  (h1 : is_conjugate z z_conjugate)
  (h2 : given_condition z z_conjugate) :
  z = 2 + Complex.i ∧ (z / z_conjugate) = (3 / 5 : ℂ) + (4 / 5 : ℂ) * Complex.i := 
  sorry

theorem find_trajectory 
  (h : Complex.normSq (z - 1) = Complex.normSq z) :
  (x - 1)^2 + y^2 = 5 :=
  sorry

end find_z_and_ratio_find_trajectory_l454_454614


namespace repeating_decimal_fraction_l454_454160

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454160


namespace inequality_proof_l454_454265

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l454_454265


namespace repeating_decimal_fraction_equiv_l454_454148

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454148


namespace Archibald_brother_won_games_l454_454871

theorem Archibald_brother_won_games
  (games_won_by_archibald : ℕ)
  (archibald_win_percentage : ℝ)
  (total_games : ℕ) 
  (brother_win_games : ℕ) :
  games_won_by_archibald = 12 →
  archibald_win_percentage = 0.40 →
  total_games = (games_won_by_archibald / archibald_win_percentage).toNat →
  brother_win_games = total_games - games_won_by_archibald →
  brother_win_games = 18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h3
  have : total_games = 30 := by norm_num [h3]
  rw this at h4
  exact h4

end Archibald_brother_won_games_l454_454871


namespace find_p_find_triangle_area_minimum_l454_454637

-- Define the parabola and line equations and their intersection points.
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
noncomputable def line_eq (x y : ℝ) := x - 2 * y + 1 = 0
noncomputable def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (p : ℝ) :
  (∃ (A B : ℝ × ℝ), parabola_eq p A.1 A.2 ∧ parabola_eq p B.1 B.2 ∧ 
   line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ 
   distance A.1 A.2 B.1 B.2 = 4 * real.sqrt 15) → p = 2 := sorry

-- Define conditions for the second part of the problem
noncomputable def focus_x := 1
noncomputable def focus_y := 0
noncomputable def mf_dot_nf_eq_zero (x1 y1 x2 y2 : ℝ) := 
  (x1 - focus_x) * (x2 - focus_x) + y1 * y2 = 0
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) := 
  1 / 2 * abs ((x1 - focus_x) * (y2 - focus_y) - (x2 - focus_x) * (y1 - focus_y))

theorem find_triangle_area_minimum (M N : ℝ × ℝ) :
  (parabola_eq 2 M.1 M.2 ∧ parabola_eq 2 N.1 N.2 ∧ 
  mf_dot_nf_eq_zero M.1 M.2 N.1 N.2) → 
  ∃ (Amin : ℝ), Amin = 12 - 8 * real.sqrt 2 ∧ triangle_area M.1 M.2 N.1 N.2 = Amin := sorry

end find_p_find_triangle_area_minimum_l454_454637


namespace audrey_not_dreaming_l454_454085

theorem audrey_not_dreaming :
  ∀ (total_sleep : ℕ) (dream_fraction : ℚ),
  total_sleep = 10 → dream_fraction = 2 / 5 →
  (total_sleep - (dream_fraction * total_sleep).toInt) = 6 :=
by
  intros total_sleep dream_fraction h1 h2
  sorry

end audrey_not_dreaming_l454_454085


namespace expected_difference_zero_l454_454530

/-- Define the probabilities for rolling a prime number leading to unsweetened cereal and a composite
    number leading to sweetened cereal on an 8-sided die.
    - Primes (unsweetened): {2, 3, 5}
    - Composites (sweetened): {4, 6, 8}
    - No cereal: 7
    - Reroll: 1
--/

def prob_unsweetened := 3 / 7  -- probability of rolling 2, 3, or 5
def prob_sweetened := 3 / 7    -- probability of rolling 4, 6, or 8
def days_in_year := 365
def expected_days_unsweetened := prob_unsweetened * days_in_year
def expected_days_sweetened := prob_sweetened * days_in_year

theorem expected_difference_zero : expected_days_unsweetened - expected_days_sweetened = 0 :=
by
  sorry

end expected_difference_zero_l454_454530


namespace sum_powers_of_neg1_l454_454463

theorem sum_powers_of_neg1 : ∑ i in (Finset.range 2011).map (Finset.elems 1), (-1 : ℤ) ^ i = -1 := 
sorry

end sum_powers_of_neg1_l454_454463


namespace find_circle_center_l454_454416

theorem find_circle_center : ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x + 12*y + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = 36) ∧ h = 1 ∧ k = -6 := 
sorry

end find_circle_center_l454_454416


namespace minimum_number_of_adventurers_l454_454057

theorem minimum_number_of_adventurers : 
  ∀ (R E S D : Finset ℕ),
  |R| = 4 → |E| = 10 → |S| = 6 → |D| = 14 →
  (∀ (x : ℕ), x ∈ R → x ∈ E ∨ x ∈ D) → ¬ (x ∈ E ∧ x ∈ D) →
  (∀ (y : ℕ), y ∈ E → y ∈ R ∨ y ∈ S) → ¬ (y ∈ R ∧ y ∈ S) →
  (R ∩ E).card = 4 → (E ∩ S).card = 6 →
  ∃ (A : Finset ℕ), |A| = 18 := 
by {
  sorry
}

end minimum_number_of_adventurers_l454_454057


namespace simplify_expression_l454_454726

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l454_454726


namespace zero_point_in_interval_l454_454423

noncomputable def f : ℝ → ℝ := λ x, 3^x + (1 / 2) * x - 2

theorem zero_point_in_interval :
  (∃ x ∈ Ioo (0 : ℝ) (1 : ℝ), f x = 0) :=
begin
  -- Given conditions
  have h0 : f 0 < 0, by {
    have : f 0 = -1,
    norm_num,
    exact this
  },
  have h1 : f 1 > 0, by {
    have : f 1 = 3 / 2,
    norm_num,
    exact this
  },
  -- Applying Intermediate Value Theorem directly
  apply_existsIntermediateValue,        
  -- preface by the necessary properties of f
  exact continuous_on_f,
  exact fun_incr_f,
  -- And the interval bounds and known function values there
  exact h0,
  exact h1
end

end zero_point_in_interval_l454_454423


namespace mean_median_difference_l454_454311

noncomputable def percentage_scores : ℕ → ℕ
| 60 := 15
| 75 := 20
| 85 := 40
| 95 := 20
| 100 := 5
| _ := 0

noncomputable def mean : ℝ :=
(0.15 * 60) + (0.20 * 75) + (0.40 * 85) + (0.20 * 95) + (0.05 * 100)

def median : ℝ := 85

def difference : ℝ := median - mean

theorem mean_median_difference : difference = 3 :=
by
  sorry

end mean_median_difference_l454_454311


namespace minimum_y_value_l454_454374

noncomputable def minimum_value_of_y (x y : ℝ) : ℝ :=
  if h : x^2 + y^2 = 16 * x + 40 * y then
    20 - Real.sqrt 464
  else
    0

theorem minimum_y_value : 
  ∀ x y : ℝ, (x^2 + y^2 = 16 * x + 40 * y) → y = 20 - Real.sqrt 464 :=
by
  intros x y hxy
  suffices : y = 20 - Real.sqrt 464, from this
  sorry -- Proof is omitted, provided problem statement only.

end minimum_y_value_l454_454374


namespace div_exp_eq_25_l454_454220

theorem div_exp_eq_25 (m n : ℕ) (h : m - n = 2) : 5^m ∕ 5^n = 25 :=
by sorry

end div_exp_eq_25_l454_454220


namespace repeating_decimal_to_fraction_l454_454121

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454121


namespace placements_for_nine_squares_l454_454392

-- Define the parameters and conditions of the problem
def countPlacements (n : ℕ) : ℕ := sorry

theorem placements_for_nine_squares : countPlacements 9 = 25 := sorry

end placements_for_nine_squares_l454_454392


namespace intersection_points_condition_l454_454988

-- Definitions of the given functions and the conditions
def f (m : ℝ) (x : ℝ) : ℝ := (1 - m) * Real.log x
def g (m : ℝ) (n : ℝ) (x : ℝ) : ℝ := -m * x^2 / 2 - (m^2 - m - 1) * x - n

-- Definition of the range condition
def range_condition (m n : ℝ) : Prop := 0 < m ∧ m < 1 ∧ n > 0

-- Main theorem statement
theorem intersection_points_condition (m n : ℝ) (h_intersect : 
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
  f m x1 = g m n x1 ∧ f m x2 = g m n x2 ∧ f m x3 = g m n x3) :
  range_condition m n :=
sorry

end intersection_points_condition_l454_454988


namespace percentage_error_in_area_l454_454817

theorem percentage_error_in_area (s : ℝ) (h : s > 0) : 
  let s' := 1.06 * s in
  let A := s ^ 2 in
  let A' := s' ^ 2 in
  let error_in_area := (A' - A) / A * 100 in
  error_in_area = 12.36 := by 
  sorry

end percentage_error_in_area_l454_454817


namespace solution_set_fxx_l454_454611

theorem solution_set_fxx (f : ℝ → ℝ) (hf_incr : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) < f(x₂))
  (hA : f 0 = -1) (hB : f 3 = 1) : {x : ℝ | |f x| < 1} = Ioo 0 3 :=
by
  sorry

end solution_set_fxx_l454_454611


namespace find_f_five_l454_454702

noncomputable def f : ℝ → ℝ :=
λ x, if x < 2 then 2 - Real.logb 2 x else 2 ^ (3 - x)

theorem find_f_five : f (f 5) = 4 :=
by
  sorry

end find_f_five_l454_454702


namespace log_sum_is_two_l454_454787

theorem log_sum_is_two : log 10 0.01 + log 2 16 = 2 := 
by
  -- Since log 10 0.01 = -2 and log 2 16 = 4, we prove that -2 + 4 = 2
  have h1 : log 10 0.01 = -2 := sorry
  have h2 : log 2 16 = 4 := sorry
  rw [h1, h2]
  norm_num
  done

end log_sum_is_two_l454_454787


namespace problem_correct_l454_454089

def decimal_to_fraction_eq_80_5 : Prop :=
  ( (0.5 + 0.25 + 0.125) / (0.5 * 0.25 * 0.125) * ((7 / 18 * (9 / 2) + 1 / 6) / (13 + 1 / 3 - (15 / 4 * 16 / 5))) = 80.5 )

theorem problem_correct : decimal_to_fraction_eq_80_5 :=
  sorry

end problem_correct_l454_454089


namespace WorldCup_group_stage_matches_l454_454407

theorem WorldCup_group_stage_matches
  (teams : ℕ)
  (groups : ℕ)
  (teams_per_group : ℕ)
  (matches_per_group : ℕ)
  (total_matches : ℕ) :
  teams = 32 ∧ 
  groups = 8 ∧ 
  teams_per_group = 4 ∧ 
  matches_per_group = teams_per_group * (teams_per_group - 1) / 2 ∧ 
  total_matches = matches_per_group * groups →
  total_matches = 48 :=
by 
  -- sorry lets Lean skip the proof.
  sorry

end WorldCup_group_stage_matches_l454_454407


namespace sum_powers_of_neg1_l454_454461

theorem sum_powers_of_neg1 : ∑ i in (Finset.range 2011).map (Finset.elems 1), (-1 : ℤ) ^ i = -1 := 
sorry

end sum_powers_of_neg1_l454_454461


namespace correct_answer_l454_454958

-- Define the function 
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Proposition p
def p : Prop := ∃ x : ℝ, (0 < x) ∧ (x < 2) ∧ f(x) < 0

-- Proposition q
def q : Prop := ∀ x y : ℝ, (x + y > 4) → (x > 2) ∧ (y > 2)

-- Correct answer based on the solution
theorem correct_answer : ¬ p ∧ ¬ q = true :=
by
  sorry

end correct_answer_l454_454958


namespace distance_DE_zero_l454_454445

theorem distance_DE_zero
  (A B C D E P: Point)
  (hAB : distance A B = 7)
  (hBC : distance B C = 24)
  (hAC : distance A C = 25)
  (hPC : distance P C = 15)
  (hP_on_AC : P ∈ line_segment A C)
  (hAD_parallel_BC : AD_parallel_B C)
  (hAE_parallel_BC : AE_parallel_B C)
  (hD_on_BP : D ∈ line_B P)
  (hE_on_BP : E ∈ line_B P)
  : distance D E = 0 :=
sorry

end distance_DE_zero_l454_454445


namespace range_of_set_l454_454853

theorem range_of_set (a b c : ℕ) (h1 : a = 2) (h2 : b = 6) (h3 : 2 ≤ c ∧ c ≤ 10) (h4 : (a + b + c) / 3 = 6) : (c - a) = 8 :=
by
  sorry

end range_of_set_l454_454853


namespace marina_more_fudge_l454_454706

theorem marina_more_fudge (h1 : 4.5 * 16 = 72)
                          (h2 : 4 * 16 - 6 = 58) :
                          72 - 58 = 14 := by
  sorry

end marina_more_fudge_l454_454706


namespace exist_nonnegative_int_a_l454_454897

theorem exist_nonnegative_int_a :
  ∃ a : ℕ, ∃ m_count : ℕ, (
    m_count > 10^6 ∧ ∀ i : ℕ, i ≤ m_count →
      ∃ m n : ℕ, m > 0 ∧ n > 0 ∧
      (∑ i in finset.range m.succ, ⌊m / (i + 1)⌋) = n^2 + a
  ) :=
sorry

end exist_nonnegative_int_a_l454_454897


namespace pythagorean_right_triangle_l454_454453

theorem pythagorean_right_triangle (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (right_triangle : c^2 = a^2 + b^2) :
  c^2 = 5 :=
by
  -- Use conditions
  rw [h1, h2] at *
  -- Conclude the proof
  simp at *
  exact right_triangle

end pythagorean_right_triangle_l454_454453


namespace digits_product_inequality_l454_454940

theorem digits_product_inequality (n : ℕ) (a : Fin 10 → ℕ)
  (h_digits : ∀ i: Fin 10, (a i) = nat.digits 10 n |> list.count i) :
  2^(a 1) * 3^(a 2) * 4^(a 3) * 5^(a 4) * 6^(a 5) * 7^(a 6) * 8^(a 7) * 9^(a 8) * 10^(a 9) ≤ n + 1 :=
sorry

end digits_product_inequality_l454_454940


namespace exists_good_permutations_l454_454454

def is_good_permutation (n : ℕ) (perm : list ℕ) : Prop :=
  let prefix_sums := list.scanl (+) 0 perm in
  (prefix_sums.tail.get_different_indices n).length = n

def num_good_permutations (n : ℕ) : ℕ :=
  (list.permutations (list.range n)).count (is_good_permutation n)

theorem exists_good_permutations : ∃ n : ℕ, (phi n ≥ 2020) ∧ (num_good_permutations n ≥ 2020) :=
sorry

end exists_good_permutations_l454_454454


namespace profit_difference_l454_454480

-- Define the initial capitals of A, B, and C
def capital_A := 8000
def capital_B := 10000
def capital_C := 12000

-- Define B's profit share
def profit_share_B := 3500

-- Define the total number of parts
def total_parts := 15

-- Define the number of parts for each person
def parts_A := 4
def parts_B := 5
def parts_C := 6

-- Define the total profit
noncomputable def total_profit := profit_share_B * (total_parts / parts_B)

-- Define the profit shares of A and C
noncomputable def profit_share_A := (parts_A / total_parts) * total_profit
noncomputable def profit_share_C := (parts_C / total_parts) * total_profit

-- Define the difference between the profit shares of A and C
noncomputable def profit_share_difference := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_difference :
  profit_share_difference = 1400 := by
  sorry

end profit_difference_l454_454480


namespace probability_two_consecutive_heads_l454_454054

theorem probability_two_consecutive_heads : 
  let fair_coin_tosses := 4
  let total_outcomes := (2^fair_coin_tosses)
  let favorable_outcomes := 10 -- This needs to be computed based on the complement as shown in the solution.
  favorable_outcomes / total_outcomes = 5 / 8 :=
by
  let fair_coin_tosses := 4
  let total_outcomes := (2^fair_coin_tosses)
  
  -- List unfavorable outcomes
  let unfavorable_outcomes := 6 -- TTTT, TTTH, TTHT, THTT, HTTT, THTH
  
  -- Calculate probability of unfavorable outcomes
  let unfavorable_prob := unfavorable_outcomes / total_outcomes
  
  -- Use complement rule
  let probability_two_consecutive_heads := 1 - unfavorable_prob
  have : probability_two_consecutive_heads = 5 / 8 := sorry
  exact this

end probability_two_consecutive_heads_l454_454054


namespace convex_polygon_area_increase_l454_454075

theorem convex_polygon_area_increase (P h : ℝ) (isConvex : isConvexPolygon) (isPerimeter : Perimeter of isConvex = P) :
  ∃ ΔA, ΔA > P * h + π * h^2 :=
sorry

end convex_polygon_area_increase_l454_454075


namespace correct_answer_l454_454959

-- Define the function 
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Proposition p
def p : Prop := ∃ x : ℝ, (0 < x) ∧ (x < 2) ∧ f(x) < 0

-- Proposition q
def q : Prop := ∀ x y : ℝ, (x + y > 4) → (x > 2) ∧ (y > 2)

-- Correct answer based on the solution
theorem correct_answer : ¬ p ∧ ¬ q = true :=
by
  sorry

end correct_answer_l454_454959


namespace evaluation_of_expression_l454_454557

theorem evaluation_of_expression: 
  (3^10 + 3^7) / (3^10 - 3^7) = 14 / 13 := 
  sorry

end evaluation_of_expression_l454_454557


namespace max_distance_eq_of_l1_l454_454638

noncomputable def equation_of_l1 (l1 l2 : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (1, 3) ∧ B = (2, 4) ∧ -- Points A and B
  l1 A.1 = A.2 ∧ l2 B.1 = B.2 ∧ -- l1 passes through A and l2 passes through B
  (∀ (x : ℝ), l1 x - l2 x = 1) ∧ -- l1 and l2 are parallel (constant difference in y-values)
  (∃ (c : ℝ), ∀ (x : ℝ), l1 x = -x + c ∧ l2 x = -x + c + 1) -- distance maximized

theorem max_distance_eq_of_l1 : 
  ∃ (l1 l2 : ℝ → ℝ), equation_of_l1 l1 l2 (1, 3) (2, 4) ∧
  ∀ (x : ℝ), l1 x = -x + 4 := 
sorry

end max_distance_eq_of_l1_l454_454638


namespace f_zero_plus_f_three_l454_454947

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_mul (a b : ℝ) : f(a + b) = f(a) * f(b)
axiom f_at_one : f(1) = 2

theorem f_zero_plus_f_three : f(0) + f(3) = 9 :=
by
  sorry

end f_zero_plus_f_three_l454_454947


namespace num_integers_satisfying_inequality_l454_454275

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l454_454275


namespace vy_length_l454_454663

/-- Rectangle WXYZ with additional geometric properties and relations. -/
structure RectangleWXYZ :=
  (point_W : Point)
  (point_X : Point)
  (point_Y : Point)
  (point_Z : Point)
  (is_rectangle : IsRectangle point_W point_X point_Y point_Z)
  (point_M : Point)
  (point_N : Point)
  (angle_WMN_right : Angle point_W point_M point_N = 90)
  (point_U : Point)
  (point_V : Point)
  (perpendicular_UV_WY : IsPerpendicular point_U point_V point_W point_Y)
  (WU_eq_UM : Distance point_W point_U = Distance point_U point_M)
  (point_R : Point)
  (U_M_meets_NV_at_R : MeetsAt point_U point_V point_N point_R)
  (point_T : Point)
  (WT_passes_through_R : PassesThrough point_W point_T point_R)
  (MU_length : Distance point_M point_U = 18)
  (RU_length : Distance point_R point_U = 27)
  (MR_length : Distance point_M point_R = 15)


/-- Prove VY = 18 given the conditions in RectangleWXYZ structure. -/
theorem vy_length (r : RectangleWXYZ) : Distance r.point_V r.point_Y = 18 := by
  sorry

end vy_length_l454_454663


namespace arc_length_of_sector_l454_454944

-- Define the given conditions
def α : ℝ := (2 / 3) * Real.pi
def S : ℝ := (25 / 3) * Real.pi

-- State the theorem to be proved
theorem arc_length_of_sector (l : ℝ) (r : ℝ) : 
  r = Real.sqrt (2 * S / α) → 
  l = r * α → 
  l = (10 / 3) * Real.pi :=
begin
  intros h_r h_l,
  rw h_r at h_l,
  rw h_l,
  sorry
end

end arc_length_of_sector_l454_454944


namespace find_erased_number_l454_454064

theorem find_erased_number (n x : ℕ) 
  (h1 : n > 3) 
  (h2 : (rat.ofInt ((n - 2) * (n + 3) / 2) - x) / (n - 2) = rat.ofInt 454 / 9) 
  : x = 107 :=
sorry

end find_erased_number_l454_454064


namespace average_price_per_bottle_l454_454027

/-
  Given:
  * Number of large bottles: 1300
  * Price per large bottle: 1.89
  * Number of small bottles: 750
  * Price per small bottle: 1.38
  
  Prove:
  The approximate average price per bottle is 1.70
-/
theorem average_price_per_bottle : 
  let num_large_bottles := 1300
  let price_per_large_bottle := 1.89
  let num_small_bottles := 750
  let price_per_small_bottle := 1.38
  let total_cost_large_bottles := num_large_bottles * price_per_large_bottle
  let total_cost_small_bottles := num_small_bottles * price_per_small_bottle
  let total_number_bottles := num_large_bottles + num_small_bottles
  let overall_total_cost := total_cost_large_bottles + total_cost_small_bottles
  let average_price := overall_total_cost / total_number_bottles
  average_price = 1.70 :=
by
  sorry

end average_price_per_bottle_l454_454027


namespace area_triangle_ABL_l454_454320

-- Define the entities in the problem
structure Rectangle where
  A B C D J K L : Point
  AB BC CD DA AJ BK : Line
  AB_eq : length AB = 7
  BC_eq : length BC = 4
  DJ_eq : length (segment D J) = 2
  KC_eq : length (segment K C) = 3
  AJ_meets_J : J ∈ AJ
  BK_meets_K : K ∈ BK
  AJ_inters_BK_L : intersection AJ BK = L

-- The requirement: prove area of triangle ABL
theorem area_triangle_ABL (R : Rectangle) : 
  area (triangle R.A R.B R.L) = 98 / 5 :=
begin
  -- Formal proof required
  sorry
end

end area_triangle_ABL_l454_454320


namespace integer_solution_count_l454_454289

theorem integer_solution_count :
  {n : ℤ | n ≠ 0 ∧ (1 / |(n:ℚ)| ≥ 1 / 5)}.finite.card = 10 :=
by
  sorry

end integer_solution_count_l454_454289


namespace number_of_distinct_non_consecutive_integers_between_3000_and_8000_l454_454290

def is_between (n : ℕ) : Prop :=
  3000 ≤ n ∧ n ≤ 8000

def distinct_digits (n : ℕ) : Prop :=
  let digits := List.ofDigits (Nat.digits 10 n) in
  digits.Nodup

def non_consecutive_digits (n : ℕ) : Prop :=
  let digits := List.ofDigits (Nat.digits 10 n) in
  ∀ (d₁ d₂ : ℕ), (d₁ ∈ digits) ∧ (d₂ ∈ digits) → ∣ d₁ - d₂ ∣ > 1

theorem number_of_distinct_non_consecutive_integers_between_3000_and_8000 :
  (Finset.filter (λ n, is_between n ∧ distinct_digits n ∧ non_consecutive_digits n) (Finset.range 8000)).card = 1050 :=
by
  sorry

end number_of_distinct_non_consecutive_integers_between_3000_and_8000_l454_454290


namespace sin_C_in_right_triangle_l454_454660

theorem sin_C_in_right_triangle 
  (A B C : Type)
  [inhabited A]
  [inhabited B]
  [inhabited C]
  (h_right_triangle : ∀ a b c : Real, ∠ A = 90) 
  (h_AB : Real) 
  (h_BC : Real) 
  (h_AB_eq : h_AB = 8) 
  (h_BC_eq : h_BC = 10) :
  sin (Real.pi / 2 - C) = 4 / 5 := 
by 
  sorry

end sin_C_in_right_triangle_l454_454660


namespace least_number_of_coins_l454_454470

theorem least_number_of_coins :
  ∃ n : ℕ, (n % 7 = 3) ∧ (n % 5 = 4) ∧ (∀ m : ℕ, (m % 7 = 3) ∧ (m % 5 = 4) → n ≤ m) := by
  use 24
  split
  - exact rfl
  split
  - exact rfl
  sorry

end least_number_of_coins_l454_454470


namespace repeating_decimal_to_fraction_l454_454118

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454118


namespace count_terminating_decimals_l454_454571

theorem count_terminating_decimals :
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 21 * k)}.finite.card = 23 :=
by sorry

end count_terminating_decimals_l454_454571


namespace positive_integer_divisibility_l454_454213

theorem positive_integer_divisibility (n : ℕ) (h_pos : n > 0) (h_div : (n^2 + 1) ∣ (n + 1)) : n = 1 := 
sorry

end positive_integer_divisibility_l454_454213


namespace train_speed_in_kmh_l454_454860

theorem train_speed_in_kmh (length_of_train : ℕ) (time_to_cross : ℕ) (speed_in_m_per_s : ℕ) (speed_in_km_per_h : ℕ) :
  length_of_train = 300 →
  time_to_cross = 12 →
  speed_in_m_per_s = length_of_train / time_to_cross →
  speed_in_km_per_h = speed_in_m_per_s * 3600 / 1000 →
  speed_in_km_per_h = 90 :=
by
  sorry

end train_speed_in_kmh_l454_454860


namespace find_p_min_area_triangle_MNF_l454_454627

-- Definitions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def line (x y : ℝ) := x - 2 * y + 1 = 0
def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def focus (p : ℝ) := (p / 2, 0)
def dot_product (M N F : ℝ × ℝ) := (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2)

-- Part 1: Finding p
theorem find_p (p : ℝ) (h : p > 0) :
  (∃ A B : ℝ × ℝ, parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  distance A B = 4 * real.sqrt 15) → p = 2 :=
sorry

-- Part 2: Minimum area of triangle MNF
theorem min_area_triangle_MNF (M N : ℝ × ℝ) (h1 : parabola 2 M.1 M.2) (h2 : parabola 2 N.1 N.2) (h3 : dot_product M N (focus 2) = 0) :
  ∃ (area : ℝ), area = 12 - 8 * real.sqrt 2 :=
sorry

end find_p_min_area_triangle_MNF_l454_454627


namespace time_to_prepare_and_cook_omelets_l454_454877

theorem time_to_prepare_and_cook_omelets :
  let chop_pepper := 3
  let chop_onion := 4
  let slice_mushroom := 2
  let dice_tomato := 3
  let grate_cheese := 1
  let saute_veggies := 4
  let cook_eggs_cheese := 6
  let total_peppers := 8
  let total_onions := 4
  let total_mushrooms := 6
  let total_tomatoes := 6
  let total_omelets := 10
  let prep_time := (total_peppers * chop_pepper) + (total_onions * chop_onion) + (total_mushrooms * slice_mushroom) + (total_tomatoes * dice_tomato) + (total_omelets * grate_cheese)
  let cook_time_one_omelet := saute_veggies + cook_eggs_cheese
  let num_omelets_during_prep := prep_time / cook_time_one_omelet
  let remaining_omelets := total_omelets - num_omelets_during_prep
  let total_cook_time := remaining_omelets * cook_time_one_omelet
  total_time := prep_time + total_cook_time
  total_time = 100 :=
by
  sorry

end time_to_prepare_and_cook_omelets_l454_454877


namespace max_sum_of_S_l454_454782

open Set

def disjoint_subset_sum_condition (S : Set ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ⊆ S → B ⊆ S → Disjoint A B → (A.sum id ≠ B.sum id)

theorem max_sum_of_S :
  ∃ S : Set ℕ, (∀ x ∈ S, x ≤ 15) ∧ disjoint_subset_sum_condition S ∧ S.sum id = 61 :=
  sorry

end max_sum_of_S_l454_454782


namespace hexagon_circle_properties_l454_454851

noncomputable def radius : ℝ := 5
def side_length : ℝ := 5
def num_sides : ℕ := 6
def hexagon_perimeter : ℝ := num_sides * side_length

theorem hexagon_circle_properties :
  let circumference := 2 * Real.pi * radius,
      arc_length := (1 / num_sides) * circumference,
      perimeter := hexagon_perimeter
  in circumference = 10 * Real.pi ∧
     arc_length = 5 * Real.pi / 3 ∧
     perimeter = 30 :=
by
  sorry

end hexagon_circle_properties_l454_454851


namespace repeating_decimal_fraction_eq_l454_454169

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454169


namespace evaluate_function_l454_454980

theorem evaluate_function(value: ℝ, sqrt_three_div_two : ℝ, answer : ℝ) 
  (h0: value = 1 - 2*(sqrt_three_div_two)^2) 
  (h1: sqrt_three_div_two = (real.sqrt 3)/2) 
  (h2: answer = -1/2) :
  f(value) = answer := by
  sorry

end evaluate_function_l454_454980


namespace minimum_n_l454_454931

-- Assume the sequence a_n is defined as part of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

-- Define S_n as the sum of the first n terms in the sequence
def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2 * d

-- Given conditions
def a1 := 2
def d := 1  -- Derived from the condition a1 + a4 = a5

-- Problem Statement
theorem minimum_n (n : ℕ) :
  (sum_arithmetic_sequence a1 d n > 32) ↔ n = 6 :=
sorry

end minimum_n_l454_454931


namespace find_pqr_l454_454236

theorem find_pqr (t : ℝ) (p q r : ℕ) (hpq_coprime : Nat.coprime p q)
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) :
  p + q + r = 79 :=
sorry

end find_pqr_l454_454236


namespace CM_parallel_EF_l454_454760

variables {A B C D E F M : Type*} [AddCommGroup A] [Module ℝ A]
variables (ABCD : Parallelogram A B C D)
variables (E : A) (F : A)
variables {B_bisector_meets_AC_at_E : B_bisector ABCD meets AC at E}
variables {B_external_bisector_meets_AD_at_F : B_external_bisector ABCD meets AD at F}
variable {M_midpoint_BE : M = midpoint B E}

theorem CM_parallel_EF
    (parallelogram_ABC : Parallelogram ABCD)
    (angle_B_bisector : Bisector ∠B ABCD meets E)
    (external_angle_B_bisector : ExternalBisector ∠B ABCD meets F)
    (midpoint_M_BE : Midpoint M B E):
  Parallel (CM) (EF) := 
sorry

end CM_parallel_EF_l454_454760


namespace num_consecutive_pairs_of_zeros_l454_454517

namespace SequenceProblem

def fn : ℕ → ℕ → ℕ
| 0, _ := 0
| n, 0 := 0
| n, 1 := 0
| n, 2 := 1
| n, m := 2^(m-3) + fn n (m-2)

theorem num_consecutive_pairs_of_zeros (n : ℕ) :
  fn n n = (1 / 3 : ℝ) * (2 ^ (n - 1) - (-1) ^ (n - 1)) := sorry

end SequenceProblem

end num_consecutive_pairs_of_zeros_l454_454517


namespace sahil_selling_price_l454_454487

def initial_cost : ℝ := 14000
def repair_cost : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percent : ℝ := 50

noncomputable def total_cost : ℝ := initial_cost + repair_cost + transportation_charges
noncomputable def profit : ℝ := profit_percent / 100 * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem sahil_selling_price :
  selling_price = 30000 := by
  sorry

end sahil_selling_price_l454_454487


namespace solve_for_a_l454_454983

theorem solve_for_a (a x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
by sorry

end solve_for_a_l454_454983


namespace find_lengths_and_angles_of_quadrilateral_l454_454656

variables (A B C D : Type) [EuclideanGeometry A B C D]
variables [convex_quadrilateral A B C D] [center_coincide A B C D (inscribed_circle A B C) (circumscribed_circle A D C)]
variables (AB_len := 1 : ℝ) 

theorem find_lengths_and_angles_of_quadrilateral 
  (h1: convex_quadrilateral A B C D)
  (h2: center_coincidence (inscribed_circle A B C) (circumscribed_circle A D C))
  (h3: length AB = 1) :
  length BC = 1 ∧ length CD = 1 ∧ length DA = 1 ∧ 
  angle A = 45 ∧ angle B = 135 ∧ angle C = 45 ∧ angle D = 135 :=
by
  sorry

end find_lengths_and_angles_of_quadrilateral_l454_454656


namespace fraction_rep_finite_geom_series_036_l454_454130

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454130


namespace problem_condition_l454_454231

noncomputable def trajectory_of_M : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 2 }

theorem problem_condition (x y : ℝ) (P : ℝ × ℝ := (2, 2)) (C : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.2 = 0 }) :
  ∀ (M : ℝ × ℝ), M ∈ trajectory_of_M ↔
  (M.1 - 1)^2 + (M.2 - 3)^2 = 2 ∧
  |(0 - P.1) * (0 - M.1) + (0 - P.2) * (0 - M.2)| = |(0 - M.1) * (0 - P.1) + (0 - M.2) * (0 - P.2)| ∧
  ∃ (l : ℝ × ℝ → Prop), ∀ A B : ℝ × ℝ, (A ≠ B) ∧ l (A) ∧ l (B) ∧ l (P) ∧ (A ∈ C) ∧ (B ∈ C) ∧ (M.1 = (A.1 + B.1) / 2) ∧ (M.2 = (A.2 + B.2) / 2) →
  (∀ x y : ℝ, (x + 3 * y - 8 = 0) = l) ∧
  (1/2 * | 0 - 2| * | 0 - (2 - y) | = 16 / 5) :=
begin
  sorry
end

end problem_condition_l454_454231


namespace percentile_sum_correct_l454_454225

def data : List ℕ := [56, 62, 63, 63, 65, 66, 68, 69, 71, 74, 76, 76, 77, 78, 79, 79, 82, 85, 87, 88, 95, 98]

def percentile (p : ℚ) (data : List ℕ) : ℕ :=
  let n := data.length
  let pos := p * n
  data.sorted.nth_le (pos.ceil - 1) sorry -- assuming 1-based indices and ceil for non-integers

theorem percentile_sum_correct :
  percentile (25 / 100) data + percentile (75 / 100) data = 148 := by
  sorry

end percentile_sum_correct_l454_454225


namespace intersection_in_six_points_l454_454226

noncomputable def triangle := ℕ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def is_acute_angle (A B C : ℝ × ℝ) : Prop :=
  let α := angle A B C
  let β := angle B C A
  let γ := angle C A B
  α < 90 ∧ β < 90 ∧ γ < 90

def centers_of_squares_on_sides (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let A1 := center_of_square_on_side B C
  let B1 := center_of_square_on_side C A
  let C1 := center_of_square_on_side A B
  (A1, B1, C1)

theorem intersection_in_six_points (T : triangle) (h : ∀ n, is_acute_angle (T n).1 (T n).2.1 (T n).2.2) :
  ∀ n, (exists p1 p2 p3 p4 p5 p6, six_distinct_points_in (T (n + 1)) (T n)) :=
by
  sorry

end intersection_in_six_points_l454_454226


namespace intersect_on_ac_l454_454314

variables {A B C A1 C1 K L P Q M : Point}
variables [Triangle A B C]

-- Given conditions
axiom angle_bisector_aa1 : Bisector (Line A A1) (Angle BAC)
axiom angle_bisector_cc1 : Bisector (Line C C1) (Angle BCA)
axiom midpoint_k : Midpoint K A B
axiom midpoint_l : Midpoint L B C
axiom perp_ap : Perpendicular (Line A P) (Line C C1)
axiom perp_cq : Perpendicular (Line C Q) (Line A A1)

-- Need to prove
theorem intersect_on_ac : Intersect (Line K P) (Line L Q) M ∧ Midpoint M A C :=
sorry

end intersect_on_ac_l454_454314


namespace solve_complex_z_l454_454751

noncomputable def solve_z : ℂ :=
  let z := (-1) / (8 * complex.I)
  z

theorem solve_complex_z :
  ∀ z : ℂ, 5 + 2 * complex.I * z = 4 - 6 * complex.I * z → z = -1 / (8 * complex.I) := 
by
  intro z
  intro h
  sorry

end solve_complex_z_l454_454751


namespace find_a_l454_454368

noncomputable def pure_imaginary_simplification (a : ℝ) (i : ℂ) (hi : i * i = -1) : Prop :=
  let denom := (3 : ℂ) - (4 : ℂ) * i
  let numer := (15 : ℂ)
  let complex_num := a + numer / denom
  let simplified_real := a + (9 : ℝ) / (5 : ℝ)
  simplified_real = 0

theorem find_a (i : ℂ) (hi : i * i = -1) : pure_imaginary_simplification (- 9 / 5 : ℝ) i hi :=
by
  sorry

end find_a_l454_454368


namespace repeating_decimal_fraction_l454_454155

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454155


namespace last_four_digits_5_to_2019_l454_454389

theorem last_four_digits_5_to_2019 :
  ∃ (x : ℕ), (5^2019) % 10000 = x ∧ x = 8125 :=
by
  sorry

end last_four_digits_5_to_2019_l454_454389


namespace geometric_series_sum_l454_454550

theorem geometric_series_sum :
  let a := (3 : ℝ) / 2
  let r := (3 : ℝ) / 2
  let n := 15
  let S := ∑ k in finset.range (n + 1), a * r ^ k
  S = 42948417 / 32768 :=
by
  let a := (3 : ℝ) / 2
  let r := (3 : ℝ) / 2
  let n := 15
  let S := finset.sum (finset.range (n + 1)) (λ k, a * r ^ k)
  have h : S = a * (1 - r ^ (n + 1)) / (1 - r)
  calc
    S = -3 * (1 - 14348907 / 32768) := by sorry  -- This step should include the actual computation steps corresponding to the solution
        ... = 42948417 / 32768 := by sorry        -- Same here
  exact h

end geometric_series_sum_l454_454550


namespace angle_B_values_l454_454653

noncomputable theory

variables (A B : ℝ)

def angles_parallel (A B : ℝ) : Prop := A = B ∨ A + B = 180

theorem angle_B_values (h_parallel : angles_parallel A B) (h_condition : 3 * A - B = 80) :
  B = 40 ∨ B = 115 :=
by sorry

end angle_B_values_l454_454653


namespace train_crosses_pole_in_l454_454073

/-
Problem Statement: A train running at the speed of 56 km/hr crosses a pole. 
The length of the train is 140 meters. Prove that it takes 9 seconds for the 
train to cross the pole.
-/

def train_speed_kmph : ℝ := 56
def train_length_m : ℝ := 140
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

theorem train_crosses_pole_in : 
  (train_length_m / train_speed_mps) = 9 := 
sorry

end train_crosses_pole_in_l454_454073


namespace average_of_first_10_multiples_of_11_l454_454030

theorem average_of_first_10_multiples_of_11 : 
  let multiples := [11, 22, 33, 44, 55, 66, 77, 88, 99, 110] in
  (List.sum multiples) / (List.length multiples : ℝ) = 60.5 := by
  sorry

end average_of_first_10_multiples_of_11_l454_454030


namespace ratio_of_distances_from_circumcenter_to_sides_is_cos_l454_454786

-- Definitions for the conditions
variables (A B C : Angle)
variables (a b c R m n p : Real)
variables (ABC : Triangle A B C)

-- Hypotheses for the proof
hypothesis h1 : Acute (triangle_ABC)
hypothesis h2 : OA = OB ∧ OB = OC ∧ OC = R
hypothesis h3 : Dist(O, Line(BC)) = m
hypothesis h4 : Dist(O, Line(AC)) = n
hypothesis h5 : Dist(O, Line(AB)) = p

-- Proof statement
theorem ratio_of_distances_from_circumcenter_to_sides_is_cos :
  m / R = cos A ∧ n / R = cos B ∧ p / R = cos C :=
sorry

end ratio_of_distances_from_circumcenter_to_sides_is_cos_l454_454786


namespace percent_students_two_novels_l454_454038

theorem percent_students_two_novels :
  let total_students := 240
  let students_three_or_more := (1/6 : ℚ) * total_students
  let students_one := (5/12 : ℚ) * total_students
  let students_none := 16
  let students_two := total_students - students_three_or_more - students_one - students_none
  (students_two / total_students) * 100 = 35 := 
by
  sorry

end percent_students_two_novels_l454_454038


namespace a_equals_b_l454_454377

def U_n (n : ℕ) : Set (ℕ × ℕ) := { p | let ⟨x, y⟩ := p in x + y % 2 = 0 ∧ 1 ≤ x ∧ x ≤ y ∧ y ≤ n }
def V_n (n : ℕ) : Set (ℕ × ℕ) := { p | let ⟨x, y⟩ := p in x + y ≤ n + 1 ∧ 1 ≤ x ∧ x ≤ y ∧ y ≤ n }

theorem a_equals_b (n : ℕ) (hn : n > 0) : 
  (U_n n).card = (V_n n).card := sorry

end a_equals_b_l454_454377


namespace correct_propositions_l454_454763

def dirichlet_function (x : ℝ) : ℝ :=
  if isRat x then 1 else 0

def proposition_1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = 1

def proposition_2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def proposition_3 (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ¬isRat T → ∀ x : ℝ, f (x + T) = f x

def proposition_4 (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℝ),
    x1 = - (real.sqrt 3 / 3) ∧ x2 = 0 ∧ x3 = real.sqrt 3 / 3 ∧
    f x1 = 0 ∧ f x2 = 1 ∧ f x3 = 0 ∧
    ∀ (A B C : ℝ × ℝ),
    A = (x1, f x1) ∧ B = (x2, f x2) ∧ C = (x3, f x3) →
    (dist A B = dist B C ∧ dist B C = dist C A)

theorem correct_propositions : 
  (proposition_1 dirichlet_function) ∧ 
  ¬(proposition_2 dirichlet_function) ∧ 
  ¬(proposition_3 dirichlet_function 1) ∧ 
  (proposition_4 dirichlet_function) :=
by
  sorry

end correct_propositions_l454_454763


namespace probability_non_positive_product_l454_454794

noncomputable def set_of_integers := {-3, -7, 0, 5, 3, -1}

theorem probability_non_positive_product :
  let total_combinations := Nat.choose 6 2,
      positive_cases := 11 in
  6.total_combinations ≠ 0 ∧ total_combinations = 15 ∧ 
  probability_non_positive_product.set_of_integers = positive_cases ->
  (positive_cases = 0) ∨ (probability_non_positive_product.set_of_integers ≤ total_combinations - positive_cases) →
  (positive_cases / total_combinations = (2 ∧ 3 * 5 / 3 / 15)) :=
sorry

end probability_non_positive_product_l454_454794


namespace square_area_l454_454525

noncomputable def side_length_x (x : ℚ) : Prop :=
5 * x - 20 = 30 - 4 * x

noncomputable def side_length_s : ℚ :=
70 / 9

noncomputable def area_of_square : ℚ :=
(side_length_s)^2

theorem square_area (x : ℚ) (h : side_length_x x) : area_of_square = 4900 / 81 :=
sorry

end square_area_l454_454525


namespace exclude_invalid_three_digits_l454_454291

def is_valid_three_digit (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ 0 ∧ ¬ (d1 = d2 ∧ d2 ≠ d3) ∧ ¬ (d2 = d3 ∧ d1 ≠ d2)

theorem exclude_invalid_three_digits : 
  { n : ℕ | n ≥ 100 ∧ n < 1000 ∧ is_valid_three_digit n }.toFinset.card = 738 :=
by
  sorry

end exclude_invalid_three_digits_l454_454291


namespace simplify_fraction_l454_454735

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454735


namespace angle_FDE_60_l454_454596

variables {ABC : Type} [triangle ABC]
variables {P : Point} {A B C D E F Q : Point} 

-- Conditions
variables (h1 : P ∈ interior_triangle A B C)
variables (h2 : collinear A P D) (h3 : collinear B P E) (h4 : collinear C P F)
variables (h5 : collinear E D Q) (h6 : ∠ EDQ = ∠ FDB)
variables (h7 : ⊥ BE AD)
variables (h8 : DQ = 2 * BD)

-- Goal
theorem angle_FDE_60 (h1 : P ∈ interior_triangle A B C)
 (h2 : collinear A P D) (h3 : collinear B P E) (h4 : collinear C P F)
 (h5 : collinear E D Q) (h6 : ∠ EDQ = ∠ FDB)
 (h7 : ⊥ BE AD) (h8 : DQ = 2 * BD) : 
 ∠ FDE = 60 := 
 sorry

end angle_FDE_60_l454_454596


namespace length_of_AC_l454_454249

-- Given conditions as Lean definitions
def AB : ℝ := 10
def φ : ℝ := (1 + real.sqrt 5) / 2
def golden_section_point (AC BC : ℝ) : Prop := (AC > BC) ∧ (AB = AC + BC) ∧ (φ = AC / BC)

-- The Lean 4 statement to prove the length of AC is approximately 6.18
theorem length_of_AC {AC BC : ℝ}
  (h : golden_section_point AC BC) :
  AC ≈ 6.18 :=
sorry

end length_of_AC_l454_454249


namespace problem1_problem2_l454_454630

-- Definitions according to the problem conditions
def parabola (p : ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def line (a b c : ℝ) := {xy : ℝ × ℝ | a * xy.1 + b * xy.2 + c = 0}

-- Problem (1): Prove that p is 2 given |AB| = 4sqrt15.
theorem problem1 (p : ℝ) (h1 : p > 0) (A B : ℝ × ℝ)
  (hA : A ∈ parabola p) (hB : B ∈ parabola p) (hL : A ∈ line 1 (-2) 1)
  (hL' : B ∈ line 1 (-2) 1) (hAB : dist A B = 4 * real.sqrt 15) :
  p = 2 :=
sorry

-- Problem (2): Prove that the minimum area of the triangle MNF is 12 - 8sqrt2.
theorem problem2 (M N : ℝ × ℝ) (p : ℝ) (h1 : p > 0)
  (hM : M ∈ parabola p) (hN : N ∈ parabola p)
  (F : ℝ × ℝ) (hF : is_focus F p)
  (hMF_NF : (M.1 - F.1) * (N.1 - F.1) + M.2 * N.2 = 0) :
  ∃ area : ℝ, (∀ a', triangle_area M N F a' → a' ≥ area) ∧ area = 12 - 8 * real.sqrt 2 :=
sorry

end problem1_problem2_l454_454630


namespace fraction_repeating_decimal_l454_454186

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454186


namespace solve_abs_equation_l454_454753

theorem solve_abs_equation (x : ℝ) :
  (|2 * x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10 / 3) :=
by sorry

end solve_abs_equation_l454_454753


namespace fraction_repeating_decimal_l454_454184

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454184


namespace analogical_reasoning_example_l454_454815

theorem analogical_reasoning_example :
  (∃ (option : List String) (C : String), 
    option = ["All even numbers are divisible by 2, 2^100 is an even number, therefore 2^100 is divisible by 2", 
              "Inducing the general term formula a_n from the sequence a_1, a_2, a_3, ...", 
              "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle", 
              "If a > b and c > d, then a - d > b - c"]
    ∧ C = "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle") :=
begin
  let options := ["All even numbers are divisible by 2, 2^100 is an even number, therefore 2^100 is divisible by 2", 
                  "Inducing the general term formula a_n from the sequence a_1, a_2, a_3, ...", 
                  "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle", 
                  "If a > b and c > d, then a - d > b - c"],
  have example_of_analogical_reasoning : "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" =
          "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" := rfl,
  exact ⟨options, "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle", rfl, example_of_analogical_reasoning⟩,
  sorry
end

end analogical_reasoning_example_l454_454815


namespace find_k_l454_454340

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l454_454340


namespace X_on_perpendicular_to_AB_l454_454862

-- Define the points and segments
variables (A B C D E H1 H2 X : Type) 

-- Define the conditions given in the problem
variables [convex_quadrilateral A B C D]
variables [parallel_lines A B C D]
variables [intersection E A C B D]
variables [orthocenter H1 B E C]
variables [orthocenter H2 A E D]
variables [midpoint X H1 H2]

-- State the goal theorem
theorem X_on_perpendicular_to_AB :
  on_perpendicular X E A B := 
sorry

end X_on_perpendicular_to_AB_l454_454862


namespace sequence_k_l454_454335

theorem sequence_k (a : ℕ → ℕ) (k : ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ m n, a (m + n) = a m * a n) 
    (h3 : (finset.range 10).sum (λ i, a (k + 1 + i)) = 2^15 - 2^5) : 
    k = 4 := 
by sorry

end sequence_k_l454_454335


namespace sum_of_roots_eq_3_l454_454894

def f (x : ℝ) : ℝ := 3 * x / (2 * x^3 - 6 * x^2 + 4 * x)

theorem sum_of_roots_eq_3 : 
  let A := 0
  let B := 1
  let C := 2
  (A + B + C = 3) → (∀ x, (2 * x^3 - 6 * x^2 + 4 * x ≠ 0) ↔ x ∉ {0, 1, 2}) :=
by 
  intro h 
  sorry

end sum_of_roots_eq_3_l454_454894


namespace distance_between_4th_and_22nd_red_lights_l454_454404

-- Define the repeating pattern of the lights
def light_pattern : list string := ["red", "red", "red", "green", "green"]

-- Function to compute the nth light in the pattern
def nth_light (n : ℕ) : string :=
  light_pattern.nth! (n % light_pattern.length)

-- Find the position of the nth occurrence of a specific light color
def nth_occurrence (color : string) (n : ℕ) : ℕ :=
  let rec find_position (count index : ℕ) :=
    if count = n then index
    else if nth_light index = color then find_position (count + 1) (index + 1)
    else find_position count (index + 1)
  find_position 0 0

-- Define distance calculation
def distance_between_lights (pos1 pos2 : ℕ) : ℕ :=
  (pos2 - pos1) * 8

-- Convert inches to feet
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

-- Define the proof statement
theorem distance_between_4th_and_22nd_red_lights :
  inches_to_feet (distance_between_lights (nth_occurrence "red" 4) (nth_occurrence "red" 22)) = 20 := by
  sorry

end distance_between_4th_and_22nd_red_lights_l454_454404


namespace middle_marble_radius_l454_454204

theorem middle_marble_radius (r_1 r_5 : ℝ) (h1 : r_1 = 8) (h5 : r_5 = 18) : 
  ∃ r_3 : ℝ, r_3 = 12 :=
by
  let r_3 := Real.sqrt (r_1 * r_5)
  have h : r_3 = 12 := sorry
  exact ⟨r_3, h⟩

end middle_marble_radius_l454_454204


namespace repeating_decimal_fraction_equiv_l454_454145

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454145


namespace time_to_cross_pole_l454_454070

-- Define given conditions
def train_speed_kmh : ℝ := 56
def train_length_m : ℝ := 140

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def train_speed_ms : ℝ := (train_speed_kmh * km_to_m) / hr_to_s

-- Define the expected time to cross the pole
def expected_time : ℝ := 9

-- Prove that the time to cross the pole is 9 seconds
theorem time_to_cross_pole : (train_length_m / train_speed_ms) = expected_time := by
  sorry

end time_to_cross_pole_l454_454070


namespace fraction_repeating_decimal_l454_454185

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454185


namespace max_perimeter_trapezoid_l454_454874

-- Definitions from the conditions
variables {A B C D O : Point}
variable (r : ℝ)
noncomputable theory

-- Conditions in the problem
def semicircle (O : Point) (A B : Point) (r : ℝ) := ∀ P : Point, dist O P = r → ∃ θ : ℝ, 0 < θ ∧ θ < π ∧ P = O + (r * cos θ, r * sin θ)
def diameter (A B : Point) (r : ℝ) := dist A B = 2 * r
def parallel_to_diameter (A B C D : Point) : Prop := dist A B > 0 ∧ (A.x = D.x ∨ B.x = C.x)

-- Main statement to prove
theorem max_perimeter_trapezoid (h_semi : semicircle O A B 1) (h_diam : diameter A B 1) (h_parallel : parallel_to_diameter A B C D) :
  ∃ P : ℝ, P = 5 :=
sorry

end max_perimeter_trapezoid_l454_454874


namespace repeating_decimal_equals_fraction_l454_454141

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454141


namespace rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l454_454560

variable (x : ℚ)

-- Polynomial 1
def polynomial1 := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_polynomial1 :
  (polynomial1 (-1) = 0) ∧
  (polynomial1 2 = 0) ∧
  (polynomial1 (-2) = 0) ∧
  (polynomial1 4 = 0) :=
sorry

-- Polynomial 2
def polynomial2 := 8*x^3 - 20*x^2 - 2*x + 5

theorem rational_roots_polynomial2 :
  (polynomial2 (1/2) = 0) ∧
  (polynomial2 (-1/2) = 0) ∧
  (polynomial2 (5/2) = 0) :=
sorry

-- Polynomial 3
def polynomial3 := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem rational_roots_polynomial3 :
  (polynomial3 (-1/2) = 0) ∧
  (polynomial3 (1/2) = 0) ∧
  (polynomial3 1 = 0) ∧
  (polynomial3 3 = 0) :=
sorry

end rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l454_454560


namespace compare_values_l454_454693

noncomputable def a := Real.log 0.9 / Real.log 0.8
noncomputable def b := Real.log 0.9 / Real.log 1.1
noncomputable def c := 1.1 ^ 0.9

theorem compare_values : b < a ∧ a < c := by
  sorry

end compare_values_l454_454693


namespace simplify_fraction_l454_454740

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454740


namespace find_k_l454_454342

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else sorry

axiom seq_base : seq 1 = 2
axiom seq_recur (m n : ℕ) : seq (m + n) = seq m * seq n

-- Given condition for sum
axiom sum_condition (k : ℕ) : 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5

-- The statement to prove
theorem find_k : ∃ k : ℕ, 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5 ∧ k = 4 :=
by {
  sorry -- proof is not required
}

end find_k_l454_454342


namespace relationship_among_abc_l454_454239

variable (a b c : ℝ)

noncomputable def a := 2^0.3
noncomputable def b := Real.log 3 / Real.log 0.2
noncomputable def c := Real.log 4 / Real.log 0.2

theorem relationship_among_abc : a > b ∧ b > c :=
by
  -- Given conditions
  let a := 2^0.3
  let b := Real.log 3 / Real.log 0.2
  let c := Real.log 4 / Real.log 0.2
  -- Prove the relationship among a, b, and c
  sorry

end relationship_among_abc_l454_454239


namespace ac_le_bc_l454_454586

theorem ac_le_bc (a b c : ℝ) (h: a > b): ∃ c, ac * c ≤ bc * c := by
  sorry

end ac_le_bc_l454_454586


namespace range_of_a_l454_454987

theorem range_of_a (a : ℝ) 
  (h : ∀ (f : ℝ → ℝ), 
    (∀ x ≤ a, f x = -x^2 - 2*x) ∧ 
    (∀ x > a, f x = -x) ∧ 
    ¬ ∃ M, ∀ x, f x ≤ M) : 
  a < -1 :=
by
  sorry

end range_of_a_l454_454987


namespace simplify_fraction_l454_454741

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454741


namespace value_of_card_l454_454383

/-- For this problem: 
    1. Matt has 8 baseball cards worth $6 each.
    2. He trades two of them to Jane in exchange for 3 $2 cards and a card of certain value.
    3. He makes a profit of $3.
    We need to prove that the value of the card that Jane gave to Matt apart from the $2 cards is $9. -/
theorem value_of_card (value_per_card traded_cards received_dollar_cards profit received_total_value : ℤ)
  (h1 : value_per_card = 6)
  (h2 : traded_cards = 2)
  (h3 : received_dollar_cards = 6)
  (h4 : profit = 3)
  (h5 : received_total_value = 15) :
  received_total_value - received_dollar_cards = 9 :=
by {
  -- This is just left as a placeholder to signal that the proof needs to be provided.
  sorry
}

end value_of_card_l454_454383


namespace quadrilateral_longest_side_length_l454_454780

noncomputable def longest_side_length (x y : ℝ) : ℝ :=
  if h : x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0 then
    max (max (real.sqrt ((4 - 0)^2 + (0 - 4)^2)) (real.sqrt ((4 - 1/2)^2 + (0 - 0)^2)))
        (real.sqrt ((1/2 - 0)^2 + (4 - 0)^2))
  else
    0

theorem quadrilateral_longest_side_length : 
  longest_side_length = 4 * real.sqrt 2 :=
sorry

end quadrilateral_longest_side_length_l454_454780


namespace gross_income_increase_l454_454481

theorem gross_income_increase (X P : ℝ) 
  (discount_percent increase_percent: ℝ) 
  (h1 : discount_percent = 0.1) 
  (h2 : increase_percent = 0.3) : 
  let initial_income := X * P,
      discounted_price := (1 - discount_percent) * P,
      increased_sales := (1 + increase_percent) * X,
      new_income := increased_sales * discounted_price in
  ((new_income - initial_income) / initial_income) * 100 = 17 := 
sorry

end gross_income_increase_l454_454481


namespace diagonals_of_parallelogram_bisect_each_other_true_l454_454814

-- Definitions based on conditions
def all_prime_numbers_are_odd := ∀ p : ℕ, prime p → ¬ even p
def some_trapezoids_are_isosceles := ∃ t : Type, isosceles_trapezoid t
def diagonals_bisect := ∀ (P : Type) [IsParallelogram P], diagonals_bisect_each_other P
def exists_x_squared_less_zero := ∃ (x : ℝ), x^2 < 0

-- Theorem based on the correct answer
theorem diagonals_of_parallelogram_bisect_each_other_true : diagonals_bisect :=
by sorry

end diagonals_of_parallelogram_bisect_each_other_true_l454_454814


namespace least_number_of_stamps_l454_454683

def min_stamps (x y : ℕ) : ℕ := x + y

theorem least_number_of_stamps {x y : ℕ} (h : 5 * x + 7 * y = 50) 
  : min_stamps x y = 8 :=
sorry

end least_number_of_stamps_l454_454683


namespace sequence_decreasing_l454_454369

variables (x y : ℝ) (c : ℝ)
variables (h_distinct : x ≠ y) (h_pos : 0 < x ∧ 0 < y) (h_c : 1 < c)

def A (n : ℕ) : ℝ :=
  if n = 1 then (c * x + y) / 2
  else (A x y c (n - 1) + H x y c (n - 1)) / 2

def G (n : ℕ) : ℝ :=
  if n = 1 then Real.sqrt (x * y)
  else Real.sqrt (A x y c (n - 1) * H x y c (n - 1))

def H (n : ℕ) : ℝ :=
  if n = 1 then 2 * x * y / (x + y)
  else 2 / (1 / A x y c (n - 1) + 1 / H x y c (n - 1))

theorem sequence_decreasing :
  ∀ n : ℕ, 1 ≤ n → A x y c n > A x y c (n + 1) :=
by
  sorry

end sequence_decreasing_l454_454369


namespace b_integer_iff_a_special_form_l454_454372

theorem b_integer_iff_a_special_form (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h2 : b = (a + Real.sqrt (a ^ 2 + 1)) ^ (1 / 3) + (a - Real.sqrt (a ^ 2 + 1)) ^ (1 / 3)) : 
  (∃ (n : ℕ), a = 1 / 2 * (n * (n^2 + 3))) ↔ (∃ (n : ℕ), b = n) :=
sorry

end b_integer_iff_a_special_form_l454_454372


namespace part_a_probability_of_equal_roots_part_b_probability_of_finite_decimal_l454_454575

-- Problem Statement for Part (a)
theorem part_a_probability_of_equal_roots : 
  let total_outcomes := 9 * 8 * 7 in
  let favorable_outcomes := 2 in
  let single_trial_prob := (favorable_outcomes : ℝ) / total_outcomes in
  let ten_trials_prob := 1 - (1 - single_trial_prob) ^ 10 in
  abs (ten_trials_prob - 0.039) < 0.001 := by
  sorry

-- Problem Statement for Part (b)
theorem part_b_probability_of_finite_decimal : 
  let total_combinations := 9 * 8 in
  let favorable_combinations := 16 in
  let single_trial_prob := (favorable_combinations : ℝ) / total_combinations in
  let five_trials_prob := (5 * single_trial_prob * (1 - single_trial_prob) ^ 4) in
  abs (five_trials_prob - 0.407) < 0.001 := by
  sorry

end part_a_probability_of_equal_roots_part_b_probability_of_finite_decimal_l454_454575


namespace inequality_proof_l454_454717

variable (a b : ℝ)

theorem inequality_proof (a_pos : 0 < a) (b_pos : 0 < b) : 
  (a / real.sqrt b) + (b / real.sqrt a) ≥ (real.sqrt a) + (real.sqrt b) :=
sorry

end inequality_proof_l454_454717


namespace find_k_l454_454338

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l454_454338


namespace leftmost_vertex_x_coordinate_l454_454433

theorem leftmost_vertex_x_coordinate :
  ∃ (n : ℕ), 
  (let A := λ n : ℕ, 0.5 * |(ln (n + 1) * (n + 1) + ln (n + 2) * (n + 2) + ln (n + 3) * (n + 3) + ln (n + 4) * n - (ln (n + 2) * n + ln (n + 3) * (n + 1) + ln (n + 4) * (n + 2) + ln (n + 1) * (n + 3)))| in 
  ln (32 / 30) = ln ((n + 1) * (n + 2) / (n * (n + 4)))) ∧ 
  n = 4 :=
exists.intro 4 sorry

end leftmost_vertex_x_coordinate_l454_454433


namespace sequence_k_eq_4_l454_454327

theorem sequence_k_eq_4 {a : ℕ → ℕ} (h1 : a 1 = 2) (h2 : ∀ m n, a (m + n) = a m * a n)
    (h3 : ∑ i in finset.range 10, a (4 + i) = 2^15 - 2^5) : 4 = 4 :=
by
  sorry

end sequence_k_eq_4_l454_454327


namespace inequality_proof_l454_454263

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l454_454263


namespace quadratics_with_real_roots_count_l454_454096

open Set Classical

noncomputable def count_real_root_quadratics : ℕ :=
  let B := {2, 4, 6}
  let C := {1, 3, 5}
  (B ×ˢ C).count (λ ⟨b, c⟩, b^2 - 4 * c ≥ 0)

theorem quadratics_with_real_roots_count :
  count_real_root_quadratics = 6 :=
by sorry

end quadratics_with_real_roots_count_l454_454096


namespace p_range_in_triangle_l454_454349

section
noncomputable def triangle_side_values {a c : ℝ} (h1 : 4 * a * c = 1) (h2 : a + c = 5 / 4) :
  (a = 1 ∧ c = 1 / 4) ∨ (a = 1 / 4 ∧ c = 1) :=
by
  sorry

theorem p_range_in_triangle {a b c p : ℝ} (h1 : sin A + sin C = p * sin B) (h2 : 4 * a * c = b^2)
  (h3 : 0 < (\frac{a ^ 2 + c ^ 2 - b ^ 2}{2 * a * c})) 
  (h4 : (\frac{a ^ 2 + c ^ 2 - b ^ 2}{2 * a * c}) < 1) :
  (sqrt 6 / 2) < p ∧ p < (sqrt 2) :=
by
  sorry
end

end p_range_in_triangle_l454_454349


namespace circle_area_l454_454649

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = 2 * r) : π * r^2 = 2 := by
  sorry

end circle_area_l454_454649


namespace problem_solution_l454_454701

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) : 
  ∀ val, val ∈ set_of_all_values (λ a b c, 1 / a + 1 / b + 1 / c) ↔ val ∈ set.Ici 9 :=
sorry

end problem_solution_l454_454701


namespace find_k_l454_454345

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else sorry

axiom seq_base : seq 1 = 2
axiom seq_recur (m n : ℕ) : seq (m + n) = seq m * seq n

-- Given condition for sum
axiom sum_condition (k : ℕ) : 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5

-- The statement to prove
theorem find_k : ∃ k : ℕ, 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5 ∧ k = 4 :=
by {
  sorry -- proof is not required
}

end find_k_l454_454345


namespace seq_all_terms_integers_l454_454094

def sequence_integers (a : ℕ → ℤ) (b : ℤ) : Prop :=
  (∀ n : ℕ, a n ≠ 0) ∧ 
  (a 1, a 2, (a 1)^2 + (a 2)^2 + b) ∈ ℤ ∧
  (∀ n : ℕ, a (n+2) = (a (n+1))^2 + b / a n)

theorem seq_all_terms_integers (a : ℕ → ℤ) (b : ℤ) 
  (h_non_zero : ∀ n : ℕ, a n ≠ 0)
  (h_initial_int : a 1 ∈ ℤ ∧ a 2 ∈ ℤ ∧ ((a 1)^2 + (a 2)^2 + b) / (a 1 * a 2) ∈ ℤ)
  (h_recurrence_relation : ∀ n : ℕ, a (n+2) = (a (n+1)^2 + b) / a n) :
  ∀ n, a n ∈ ℤ := 
sorry

end seq_all_terms_integers_l454_454094


namespace number_of_teams_in_tournament_l454_454659

theorem number_of_teams_in_tournament :
  ∃ n : ℕ, 
    (∀ (team_points : ℕ → ℕ), 
      (team_points 1 = 26 ∧ (team_points 2 = 20 ∧ team_points 3 = 20) ∧ 
       (∀ i, team_points i = 2 * wins i) ∧ 
       (∀ i, wins i = ∑ j in (finset.range n), victories i j) ∧ 
       (∀ t1 t2 : ℕ, t1 ≠ t2 → victories t1 t2 + victories t2 t1 = 2) ∧
       (∀ t, victories t t = 0) ∧
       (∑ i in (finset.range n), team_points i = n * (n - 1))
    ) → n = 12)
:= sorry

end number_of_teams_in_tournament_l454_454659


namespace repeating_decimal_fraction_equiv_l454_454144

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454144


namespace second_divisor_of_n_plus_three_l454_454568

theorem second_divisor_of_n_plus_three 
  (n : ℕ) 
  (h1 : n ≡ -3 [MOD 12]) 
  (h2 : n ≡ -3 [MOD 35]) 
  (h3 : n ≡ -3 [MOD 40]) : 
  ∃ d : ℕ, d = 3 ∧ d ∣ (n + 3) ∧ d ≠ 1 ∧ d ≠ 12 ∧ d ≠ 35 ∧ d ≠ 40 :=
sorry

end second_divisor_of_n_plus_three_l454_454568


namespace sum_of_a_l454_454589

def f (n : ℕ) : ℝ := n^2 * Real.cos (n * Real.pi)

def a (n : ℕ) : ℝ := f n + f (n + 1)

theorem sum_of_a {n : ℕ} (h : n = 100) : ∑ i in Finset.range 100, a (i + 1) = -550 :=
by
  sorry

end sum_of_a_l454_454589


namespace power_division_calculation_l454_454532

theorem power_division_calculation :
  ( ( 5^13 / 5^11 )^2 * 5^2 ) / 2^5 = 15625 / 32 :=
by
  sorry

end power_division_calculation_l454_454532


namespace expression_value_l454_454088

theorem expression_value : 
  (Nat.factorial 10) / (2 * (Finset.sum (Finset.range 11) id)) = 33080 := by
  sorry

end expression_value_l454_454088


namespace percent_exceed_l454_454025

theorem percent_exceed (x y : ℝ) (h : x = 0.75 * y) : ((y - x) / x) * 100 = 33.33 :=
by
  sorry

end percent_exceed_l454_454025


namespace ellipse_properties_l454_454941

-- Define the properties of ellipse and points
variables {a b : ℝ}
variable (h1 : 0 < b ∧ b < a)
variable (h2 : (1 : ℝ) * 1 / (a^2) + (1 : ℝ) * 1 / (b^2) = 1)
variable (h3 : ∃ (F1 F2 : ℝ × ℝ), ∀ A : ℝ × ℝ, (abs ((sqrt ((F1.1 - 1) ^ 2 + (F1.2 - 1) ^ 2))) + abs ((sqrt ((F2.1 - 1) ^ 2 + (F2.2 - 1 ) ^ 2))) = 4))

-- The goal is to prove the following three properties
theorem ellipse_properties :
(∃ (a : ℝ), (2 * a = 4)) ∧
(∃ (b : ℝ), let eq1 := (1 / (4 : ℝ)) + (1 / (b^2)) = 1 in
  eq1) ∧
(∃ (kcd : ℝ), kcd = 1 / 3) :=
begin
  sorry
end

end ellipse_properties_l454_454941


namespace min_value_of_expression_l454_454565

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (2 * sin x + (1 / sin x))^2 + (2 * cos x + (1 / cos x))^2

theorem min_value_of_expression : 
  ∃ x ∈ Ioo 0 (π / 2), f x = 28 := sorry

end min_value_of_expression_l454_454565


namespace alternating_sum_l454_454535

theorem alternating_sum : (∑ i in Finset.range 98, if i % 2 = 0 then i + 1 else -(i + 1)) = -49 :=
by
  sorry

end alternating_sum_l454_454535


namespace increasing_function_l454_454473

def f_A (x : ℝ) : ℝ := -x
def f_B (x : ℝ) : ℝ := (2 / 3) ^ x
def f_C (x : ℝ) : ℝ := -1 / x
def f_D (x : ℝ) : ℝ := Real.sqrt x

theorem increasing_function :
  ∀ f ∈ {f_A, f_B, f_C, f_D}, (f = f_D ↔ ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry

end increasing_function_l454_454473


namespace tangent_line_eq_at_1_e_l454_454257

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_eq_at_1_e : 
  let x_0 := 1
  let y_0 := e
  let f' := λ x, (Real.exp x + x * Real.exp x)
  let slope := f' x_0
  x_0 = 1 → y_0 = Real.exp 1 → slope = (Real.exp 1 + 1 * Real.exp 1) →
  ∃ a b, (∀ x, (a * x + b = slope * (x - x_0) + y_0) ∧ a = 2 * Real.exp 1 ∧ b = - Real.exp 1) := 
by 
  sorry

end tangent_line_eq_at_1_e_l454_454257


namespace no_intersection_min_a_l454_454590

def f (a x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem no_intersection_min_a (h : ∀ x, 0 < x ∧ x < 1 / 2 → f a x > 0) : 
  a ≥ 2 - 4 * Real.log 2 :=
by
  sorry

end no_intersection_min_a_l454_454590


namespace trapezoid_perimeter_l454_454673

theorem trapezoid_perimeter (A B C D : Type) 
  [trapezoid ABCD AB CD AD BC] 
  (h_parallel: AB || CD) 
  (h_eq: AB = CD) 
  (h_perp: AD ⊥ BC) 
  (AD_length: AD = 5) 
  (DC_length: DC = 7) 
  (BC_length: BC = 12) :
  perimeter ABCD = 41 := 
sorry

end trapezoid_perimeter_l454_454673


namespace repeating_decimal_equals_fraction_l454_454134

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454134


namespace fraction_equivalent_of_repeating_decimal_l454_454191

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454191


namespace range_of_a_l454_454651

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, x ≥ 1 → x^2 + a * x + 9 ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l454_454651


namespace vector_identity_l454_454408

variable {Point : Type*} [AddCommGroup Point]

variables (A B C M : Point)

-- Condition
axiom h : (AC(M A C) + AB(M A B) = 2 • AM(M A))

-- Question we need to prove
theorem vector_identity : (MC(M C) + MB(M B) = 0) :=
by
  sorry

end vector_identity_l454_454408


namespace num_integer_solutions_l454_454891

-- Definitions based on conditions identified in part a)
def valid_integer_solutions (n : ℤ) : Prop := (n - 2) * (n + 4) * (n - 3) < 0

-- The main statement of the proof problem
theorem num_integer_solutions : ∃ (unique_n : ℤ), valid_integer_solutions unique_n ∧ 
  (∀ (n : ℤ), valid_integer_solutions n → n = unique_n) :=
begin
  sorry
end

end num_integer_solutions_l454_454891


namespace solve_for_x_l454_454752

theorem solve_for_x (x : ℝ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -(2 / 11) :=
by
  sorry

end solve_for_x_l454_454752


namespace fraction_repeating_decimal_l454_454179

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454179


namespace intersection_complement_l454_454963

def U : set ℝ := set.univ
def A : set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : set ℝ := {x | x > 2}
def C : set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem intersection_complement :
  A ∩ (U \ B) = C :=
by
  sorry

end intersection_complement_l454_454963


namespace triangle_centroid_property_l454_454677

def distance_sq (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem triangle_centroid_property
  (A B C P : ℝ × ℝ)
  (G : ℝ × ℝ)
  (hG : G = ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )) :
  distance_sq A P + distance_sq B P + distance_sq C P = 
  distance_sq A G + distance_sq B G + distance_sq C G + 3 * distance_sq G P :=
by
  sorry

end triangle_centroid_property_l454_454677


namespace angle_theorem_l454_454036

noncomputable def angle_proof (EF AB DKP BPK : Type) [geometry EF AB DKP BPK] : Prop :=
  (Parallel EF AB) → 
  (Angle DKP = 105) → 
  (Angle BPK = 75)

theorem angle_theorem {EF AB DKP BPK : Type} [geometry EF AB DKP BPK] :
  angle_proof EF AB DKP BPK := by
  intro h_parallel h_DKP
  sorry

end angle_theorem_l454_454036


namespace cost_per_hardcover_book_l454_454709

-- Define the constants and variables
def snacks_fee_per_member : ℕ := 150
def paperback_fee_per_book : ℕ := 12
def number_of_books : ℕ := 6
def total_collected : ℕ := 2412

-- Define the total fee collected by each member
def total_fee_per_member (H : ℕ) : ℕ :=
  snacks_fee_per_member + number_of_books * H + number_of_books * paperback_fee_per_book

-- Define the total fee collected by Niles
def total_fee_collected (H : ℕ) : ℕ :=
  6 * total_fee_per_member H

-- Prove that the cost per hardcover book is $30
theorem cost_per_hardcover_book : ∃ H : ℕ, total_fee_collected H = total_collected ∧ H = 30 :=
by
  -- We use the conditions to derive the solution
  let H := 30
  have h1 : total_fee_collected H = total_collected :=
    calc
      total_fee_collected H
        = 6 * total_fee_per_member H : by rfl
    ... = 6 * (snacks_fee_per_member + number_of_books * H + number_of_books * paperback_fee_per_book) : by rfl
    ... = 6 * (150 + 6 * 30 + 6 * 12) : by simp [snacks_fee_per_member, number_of_books, paperback_fee_per_book]
    ... = 6 * (150 + 180 + 72) : by norm_num
    ... = 6 * 402 : by norm_num
    ... = 2412 : by norm_num
  use H
  split <;> try { assumption }
  sorry

end cost_per_hardcover_book_l454_454709


namespace mary_needs_more_flour_proof_mary_needs_more_flour_l454_454382

theorem mary_needs_more_flour (total_flour_needed : ℕ) (flour_already_put_in : ℕ) : ℕ :=
  if h : total_flour_needed = 12 ∧ flour_already_put_in = 10 then
    2
  else
    0

theorem proof_mary_needs_more_flour : mary_needs_more_flour 12 10 = 2 :=
by {
  unfold mary_needs_more_flour,
  split_ifs,
  { simp },
  { contradiction }
}

end mary_needs_more_flour_proof_mary_needs_more_flour_l454_454382


namespace smallest_square_area_l454_454567

theorem smallest_square_area (n : ℕ) (h : ∃ m : ℕ, 14 * n = m ^ 2) : n = 14 :=
sorry

end smallest_square_area_l454_454567


namespace tim_interest_rate_l454_454710

theorem tim_interest_rate
  (r : ℝ)
  (h1 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000))
  (h2 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000) + 23.5) : 
  r = 0.1 :=
by
  sorry

end tim_interest_rate_l454_454710


namespace sum_even_integers_102_to_200_l454_454821

theorem sum_even_integers_102_to_200 : 
    let first_term := 102
    let last_term := 200
    let common_difference := 2
    let n := (last_term - first_term) / common_difference + 1
    let sum := (n / 2) * (first_term + last_term)
    sum = 7550 :=
by
  let first_term := 102
  let last_term := 200
  let common_difference := 2
  let n := (last_term - first_term) / common_difference + 1
  let sum := (n / 2) * (first_term + last_term)
  show sum = 7550_from rfl
  sorry

end sum_even_integers_102_to_200_l454_454821


namespace f_f_neg1_l454_454953

def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 - 1 else -x + 2

theorem f_f_neg1 : f (f (-1)) = 8 := by
  sorry

end f_f_neg1_l454_454953


namespace repeating_decimal_fraction_equiv_l454_454143

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454143


namespace problem1_problem2_l454_454631

-- Definitions according to the problem conditions
def parabola (p : ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def line (a b c : ℝ) := {xy : ℝ × ℝ | a * xy.1 + b * xy.2 + c = 0}

-- Problem (1): Prove that p is 2 given |AB| = 4sqrt15.
theorem problem1 (p : ℝ) (h1 : p > 0) (A B : ℝ × ℝ)
  (hA : A ∈ parabola p) (hB : B ∈ parabola p) (hL : A ∈ line 1 (-2) 1)
  (hL' : B ∈ line 1 (-2) 1) (hAB : dist A B = 4 * real.sqrt 15) :
  p = 2 :=
sorry

-- Problem (2): Prove that the minimum area of the triangle MNF is 12 - 8sqrt2.
theorem problem2 (M N : ℝ × ℝ) (p : ℝ) (h1 : p > 0)
  (hM : M ∈ parabola p) (hN : N ∈ parabola p)
  (F : ℝ × ℝ) (hF : is_focus F p)
  (hMF_NF : (M.1 - F.1) * (N.1 - F.1) + M.2 * N.2 = 0) :
  ∃ area : ℝ, (∀ a', triangle_area M N F a' → a' ≥ area) ∧ area = 12 - 8 * real.sqrt 2 :=
sorry

end problem1_problem2_l454_454631


namespace inequality_positive_reals_l454_454587

theorem inequality_positive_reals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8) ∧ (1 / a^2 + 1 / b^2 + 8 * a * b = 8 → a = b ∧ a = 1/2) :=
by
  sorry

end inequality_positive_reals_l454_454587


namespace cubic_root_form_l454_454201

-- Let x^3 + p * x + q = 0 be the given cubic equation
theorem cubic_root_form (p q : ℝ) :
  ∀ (x : ℝ),
    x^3 + p * x + q = 0 ↔
      x = (∛(- q / 2 + √(q^2 / 4 + p^3 / 27))) + (∛(- q / 2 - √(q^2 / 4 + p^3 / 27))) :=
by
  sorry

end cubic_root_form_l454_454201


namespace no_integers_exist_l454_454895

theorem no_integers_exist (m n : ℤ) : ¬(m^2 + 1954 = n^2) :=
begin
  sorry
end

end no_integers_exist_l454_454895


namespace min_edges_for_flight_routes_l454_454500

noncomputable def minEdges (v : ℕ) (e : ℕ) : Prop :=
  ∀ (G : Type) [simple_graph G], G.vertices = v → 
    G.is_connected ∧ ∀ {A B : G.vertex}, G.adj A B → ∃ C : G.vertex, G.adj A C ∧ G.adj B C → e ≥ 3030

theorem min_edges_for_flight_routes (v : ℕ) (e : ℕ) (h : v = 2021) : minEdges v e :=
by
  intros G
  rw h
  sorry

end min_edges_for_flight_routes_l454_454500


namespace find_a_ge_3_l454_454573

open Real Function

noncomputable def trigonometric_inequality (a : ℝ) :=
  ∀ θ ∈ Ico 0 (π / 2),
    sqrt 2 * (2 * a + 3) * cos (θ - π / 4) / (6 / (sin θ + cos θ)) <
    2 * sin (2 * θ) - 3 * a + 6

theorem find_a_ge_3 : 
  ∀ (a : ℝ), trigonometric_inequality a → a > 3 :=
sorry

end find_a_ge_3_l454_454573


namespace center_of_circle_intersection_condition_l454_454950

noncomputable def circle_center (C : ∀ x y : ℝ, x^2 + y^2 - 3*x = 0 ∧ (5/3) < x ∧ x ≤ 3) : ℝ × ℝ :=
(3/2, 0)

theorem center_of_circle : circle_center (λ x y, x^2 + y^2 - 3*x = 0 ∧ (5/3) < x ∧ x ≤ 3) = (3/2, 0) :=
by
  sorry

theorem intersection_condition (L : ∀ k : ℝ, k ∈ ( [-2*(Real.sqrt 5)/7, 2*(Real.sqrt 5)/7] ∪ {-3/4, 3/4} ) ↔ (∀ x y : ℝ, (x^2 + y^2 - 3*x = 0 ∧ (5/3) < x ∧ x ≤ 3) → (y = k*(x - 4) → unique (λ p, (x, y) = p))) :=
by
  sorry

end center_of_circle_intersection_condition_l454_454950


namespace max_a_for_no_lattice_point_l454_454059

theorem max_a_for_no_lattice_point (a : ℝ) (hm : ∀ m : ℝ, 1 / 2 < m ∧ m < a → ¬ ∃ x y : ℤ, 0 < x ∧ x ≤ 200 ∧ y = m * x + 3) : 
  a = 101 / 201 :=
sorry

end max_a_for_no_lattice_point_l454_454059


namespace distance_sum_27_consec_eq_1926_distance_sum_27_consec_a2_eq_1932_find_a_l454_454380

noncomputable def a : ℝ := 2 / 3

theorem distance_sum_27_consec_eq_1926 (k : ℕ) :
  (∑ i in (range 27), abs ((a : ℝ) - (k + i))) = 1926 := sorry

theorem distance_sum_27_consec_a2_eq_1932 (k : ℕ) :
  (∑ i in (range 27), abs ((a ^ 2 : ℝ) - (k + i))) = 1932 := sorry

theorem find_a (a : ℝ) (h1 : (∑ i in (range 27), abs ((a : ℝ) - (k + i))) = 1926)
  (h2 : (∑ i in (range 27), abs ((a ^ 2 : ℝ) - (k + i))) = 1932) : a = 2 / 3 := sorry

end distance_sum_27_consec_eq_1926_distance_sum_27_consec_a2_eq_1932_find_a_l454_454380


namespace converse_not_true_D_l454_454692

-- Definitions for use in the proof
variables {a b c : Line} {α β : Plane}

-- Conditions
axiom diff_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom non_coincident_planes : α ≠ β

-- Definitions for sublines of planes
def line_in_plane (l : Line) (p : Plane) : Prop :=
  l ⊆ p

-- Definitions for perpendicularity and parallelism
def perp_line_plane (l : Line) (p : Plane) : Prop :=
  ∃ p1 p2, p = p1 ∪ p2 ∧ l ⟂ p1 ∧ l ⟂ p2

def perp_plane_plane (p1 p2 : Plane) : Prop :=
  ∃ l, l ∈ p1 ∧ l ⟂ p2

def parallel_plane_plane (p1 p2 : Plane) : Prop :=
  ∀ (l ∈ p1), l ∥ p2

-- Problem statement
theorem converse_not_true_D  (b_in_α : line_in_plane b α)
                            (b_perp_β : perp_line_plane b β)
                            (α_perp_β : perp_plane_plane α β) :
  ¬ (perp_plane_plane α β → perp_line_plane b β) :=
sorry

end converse_not_true_D_l454_454692


namespace problem_a_even_triangles_problem_b_even_triangles_l454_454828

-- Definition for problem (a)
def square_divided_by_triangles_3_4_even (a : ℕ) : Prop :=
  let area_triangle := 3 * 4 / 2
  let area_square := a * a
  let k := area_square / area_triangle
  (k % 2 = 0)

-- Definition for problem (b)
def rectangle_divided_by_triangles_1_2_even (l w : ℕ) : Prop :=
  let area_triangle := 1 * 2 / 2
  let area_rectangle := l * w
  let k := area_rectangle / area_triangle
  (k % 2 = 0)

-- Theorem for problem (a)
theorem problem_a_even_triangles {a : ℕ} (h : a > 0) :
  square_divided_by_triangles_3_4_even a :=
sorry

-- Theorem for problem (b)
theorem problem_b_even_triangles {l w : ℕ} (hl : l > 0) (hw : w > 0) :
  rectangle_divided_by_triangles_1_2_even l w :=
sorry

end problem_a_even_triangles_problem_b_even_triangles_l454_454828


namespace part1_part2_l454_454221

noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem part1 : x^2 + 2 * x * y + y^2 = 12 := 
by 
  let s := sqrt (3:ℝ)
  let t := sqrt (2:ℝ)
  have hx : x = s + t := by rfl
  have hy : y = s - t := by rfl
  rw [hx, hy]
  calc
    (s + t)^2 + 2 * (s + t) * (s - t) + (s - t)^2
    = (s + t)^2 + 2 * ((s * s) - (t * t)) + (s - t)^2 : by rw mul_assoc
    ... = (s + t)^2 + 2 * (3 - 2) + (s - t)^2 : by { rw [sqrt_mul_self 3, sqrt_mul_self 2] }
    ... = (s + t)^2 + 2 * 1 + (s - t)^2 : by simp
    ... = 12 : by
      { 
        calc (s + t)^2 + (s - t)^2 + 2
        = (3 + 2 * s * t + 2) +  (3 - 2 * s * t + 2) - 2
        = 12 : by { rw sqrt_mul_self 2, rw sqrt_mul_self 3, ring }
      }

theorem part2 : (1 / y - 1 / x = 2 * sqrt 2) :=
by 
  let s := sqrt (3:ℝ)
  let t := sqrt (2:ℝ)
  have hx : x = s + t := by rfl
  have hy : y = s - t := by rfl
  rw [hx, hy]
  have xy : (s + t) * (s - t) = 1 := by calc (s + t) * (s - t) = 3 - 2 : by { rw [sqrt_mul_self 3, sqrt_mul_self 2] }
  rw [xy, sub_div, div_one]
  calc (s - t)/1 - 1/(s+t)
  = (2 * sqrt 2) : by 
    { have : 2 * sqrt (2:ℝ) =  (s * s - t * t) / 1, rw ←sqrt_mul_self 
      rw [←sqrt_mul, sqrt_mul_self] }

end part1_part2_l454_454221


namespace incorrect_inequality_l454_454581

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ∃ c : ℝ, ac = bc :=
by
  have h1 : ¬(ac > bc) := by
    let c := 0
    show ac = bc 
    sorry

  exact ⟨0, h1⟩

end incorrect_inequality_l454_454581


namespace repeating_decimal_fraction_eq_l454_454163

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454163


namespace value_of_expression_l454_454428

variable {α : Type*} [LinearOrder α]

def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing (f : α → α) (a b : α) : Prop :=
  ∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x < y → f x < f y

theorem value_of_expression (f : ℝ → ℝ) (h_odd : is_odd_function f)
    (h_increasing : is_increasing f 3 7)
    (h_max : f 6 = 8)
    (h_min : f 3 = -1) :
  2 * f (-6) + f (-3) = -15 :=
by
  sorry

end value_of_expression_l454_454428


namespace incenter_intersects_square_side_l454_454712

-- Definitions of the geometric entities stated in conditions:
-- Semicircle, points on diameter, points on semicircle forming a square with area equal to triangle

variables {R r : ℝ}  -- R is the radius of the semicircle, r is the radius of the incenter of triangle ABC

noncomputable def KLMN_square_area_eq_ABC_area (a : ℝ) (K L M N C : Point) (A B : Point) : Prop :=
  is_square K L M N ∧
  area K L M N = area A B C

-- The main theorem to prove
theorem incenter_intersects_square_side (AB KLMN_condition : Prop) (K L M N A B C : Point) :
  (∃ P, intersects_square_side_and_bisector K L M N A B KLMN_condition P 
    ∧ is_incenter P A B C) ↔ KLMN_square_area_eq_ABC_area A B C :=
  sorry

end incenter_intersects_square_side_l454_454712


namespace integral_curves_l454_454452

theorem integral_curves (y x : ℝ) : 
  (∃ k : ℝ, (y - x) / (y + x) = k) → 
  (∃ c : ℝ, y = x * (c + 1) / (c - 1)) ∨ (y = 0) ∨ (y = x) ∨ (x = 0) :=
by
  sorry

end integral_curves_l454_454452


namespace range_of_m_l454_454234
noncomputable theory

def proposition_p (m : ℝ) : Prop :=
  -((m - 2) / 2) ≤ 1

def proposition_q (m : ℝ) : Prop :=
  m + 1 > 0 ∧ 9 - m > m + 1

def both_false (p q : Prop) : Prop :=
  ¬p ∧ ¬q

def conditions (p q : Prop) : Prop :=
  (p ∨ q) ∧ both_false p q ∧ ¬¬p

theorem range_of_m (m : ℝ) : conditions (proposition_p m) (proposition_q m) → m ≤ -1 ∨ m = 4 :=
by
  sorry

end range_of_m_l454_454234


namespace find_k_l454_454337

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l454_454337


namespace number_of_cars_l454_454868

theorem number_of_cars (x : ℕ) (h : 3 * (x - 2) = 2 * x + 9) : x = 15 :=
by {
  sorry
}

end number_of_cars_l454_454868


namespace max_value_no_min_value_l454_454765

noncomputable def y := λ x : ℝ, x^3 - 3 * x^2 - 9 * x

theorem max_value_no_min_value (h : -2 < 2) :
  ∃ x : ℝ, -2 < x ∧ x < 2 ∧ y x = 5 ∧
  (∀ x : ℝ, -2 < x → x < 2 → y x ≥ y (-1)) ∧
  ¬(∃ x : ℝ, -2 < x ∧ x < 2 ∧ ∀ y' : ℝ, -2 < y' ∧ y' < 2 → y x ≤ y y') :=
by
  change y = λ x : ℝ, x^3 - 3 * x^2 - 9 * x
  sorry

end max_value_no_min_value_l454_454765


namespace valley_number_count_l454_454102

def is_valley_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  let d1 := n / 100 in let d2 := (n / 10) % 10 in let d3 := n % 10
  in d2 < d1 ∧ d2 < d3

def count_valley_numbers : ℕ :=
  Finset.card ((Finset.filter is_valley_number (Finset.range 900)).image (λ n, n + 100))

theorem valley_number_count : count_valley_numbers = 201 :=
  sorry

end valley_number_count_l454_454102


namespace smallest_four_digit_multiple_of_8_with_digit_sum_20_l454_454010

def sum_of_digits (n : Nat) : Nat :=
  Nat.digits 10 n |>.foldl (· + ·) 0

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20:
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ sum_of_digits n = 20 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 8 = 0 ∧ sum_of_digits m = 20 → n ≤ m :=
by { sorry }

end smallest_four_digit_multiple_of_8_with_digit_sum_20_l454_454010


namespace find_annual_interest_rate_l454_454520

theorem find_annual_interest_rate (A P : ℝ) (n t : ℕ) (r : ℝ) :
  A = P * (1 + r / n)^(n * t) →
  A = 5292 →
  P = 4800 →
  n = 1 →
  t = 2 →
  r = 0.05 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end find_annual_interest_rate_l454_454520


namespace distance_from_A_to_B_l454_454800

-- We declare the points A and B in Cartesian coordinates
def point_A := (2 : ℝ, -2 : ℝ)
def point_B := (8 : ℝ, 8 : ℝ)

-- We calculate the expected distance manually
def expected_distance := Real.sqrt 136

-- Define a function to calculate the distance between two points in 2D Cartesian coordinates
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- The theorem statement that proves the distance between the points is sqrt(136)
theorem distance_from_A_to_B : distance point_A point_B = expected_distance :=
by
  -- This is the place to write the proof, but we'll skip it for this statement
  sorry

end distance_from_A_to_B_l454_454800


namespace phone_answer_prob_within_four_rings_l454_454058

def prob_first_ring : ℚ := 1/10
def prob_second_ring : ℚ := 1/5
def prob_third_ring : ℚ := 3/10
def prob_fourth_ring : ℚ := 1/10

theorem phone_answer_prob_within_four_rings :
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring = 7/10 :=
by
  sorry

end phone_answer_prob_within_four_rings_l454_454058


namespace repeating_decimal_fraction_l454_454156

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454156


namespace percentage_of_400_equals_100_l454_454200

def part : ℝ := 100
def whole : ℝ := 400

theorem percentage_of_400_equals_100 : (part / whole) * 100 = 25 := by
  sorry

end percentage_of_400_equals_100_l454_454200


namespace probability_of_pulling_blue_ball_l454_454788

def given_conditions (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ) :=
  total_balls = 15 ∧ initial_blue_balls = 7 ∧ blue_balls_removed = 3

theorem probability_of_pulling_blue_ball
  (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ)
  (hc : given_conditions total_balls initial_blue_balls blue_balls_removed) :
  ((initial_blue_balls - blue_balls_removed) / (total_balls - blue_balls_removed) : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_pulling_blue_ball_l454_454788


namespace g_at_10_l454_454938

-- Definitions and conditions
def f : ℝ → ℝ := sorry
axiom f_at_1 : f 1 = 10
axiom f_inequality_1 : ∀ x : ℝ, f (x + 20) ≥ f x + 20
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1
def g (x : ℝ) : ℝ := f x - x + 1

-- Proof statement (no proof required)
theorem g_at_10 : g 10 = 10 := sorry

end g_at_10_l454_454938


namespace first_number_Harold_says_l454_454109

/-
  Define each student's sequence of numbers.
  - Alice skips every 4th number.
  - Barbara says numbers that Alice didn't say, skipping every 4th in her sequence.
  - Subsequent students follow the same rule.
  - Harold picks the smallest prime number not said by any student.
-/

def is_skipped_by_Alice (n : Nat) : Prop :=
  n % 4 ≠ 0

def is_skipped_by_Barbara (n : Nat) : Prop :=
  is_skipped_by_Alice n ∧ (n / 4) % 4 ≠ 3

def is_skipped_by_Candice (n : Nat) : Prop :=
  is_skipped_by_Barbara n ∧ (n / 16) % 4 ≠ 3

def is_skipped_by_Debbie (n : Nat) : Prop :=
  is_skipped_by_Candice n ∧ (n / 64) % 4 ≠ 3

def is_skipped_by_Eliza (n : Nat) : Prop :=
  is_skipped_by_Debbie n ∧ (n / 256) % 4 ≠ 3

def is_skipped_by_Fatima (n : Nat) : Prop :=
  is_skipped_by_Eliza n ∧ (n / 1024) % 4 ≠ 3

def is_skipped_by_Grace (n : Nat) : Prop :=
  is_skipped_by_Fatima n

def is_skipped_by_anyone (n : Nat) : Prop :=
  ¬ is_skipped_by_Alice n ∨ ¬ is_skipped_by_Barbara n ∨ ¬ is_skipped_by_Candice n ∨
  ¬ is_skipped_by_Debbie n ∨ ¬ is_skipped_by_Eliza n ∨ ¬ is_skipped_by_Fatima n ∨
  ¬ is_skipped_by_Grace n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ (m : Nat), m ∣ n → m = 1 ∨ m = n

theorem first_number_Harold_says : ∃ n : Nat, is_prime n ∧ ¬ is_skipped_by_anyone n ∧ n = 11 := by
  sorry

end first_number_Harold_says_l454_454109


namespace sum_powers_of_minus_one_l454_454458

/-- 
  Prove that the sum of powers of -1 from 1 to 2011 equals -1:
  ∑_{i=1}^{2011} (-1)^i = -1
-/
theorem sum_powers_of_minus_one:
  ∑ i in Finset.range 2011.succ, (-1) ^ (i + 1) = -1 := 
by
  sorry

end sum_powers_of_minus_one_l454_454458


namespace sector_area_l454_454223

theorem sector_area (radius : ℝ) (central_angle : ℝ) (h1 : radius = 3) (h2 : central_angle = 2 * Real.pi / 3) : 
    (1 / 2) * radius^2 * central_angle = 6 * Real.pi :=
by
  rw [h1, h2]
  sorry

end sector_area_l454_454223


namespace least_area_of_square_l454_454029

theorem least_area_of_square :
  ∀ (s : ℝ), (3.5 ≤ s ∧ s < 4.5) → (s * s ≥ 12.25) :=
by
  intro s
  intro hs
  sorry

end least_area_of_square_l454_454029


namespace events_are_mutually_exclusive_but_not_opposite_l454_454916

noncomputable def pencil_case : Finset String := {"pen", "pen", "pencil", "pencil"}

def event_exactly_1_pen (selection: Finset String) : Prop := 
  selection.filter (fun x => x = "pen").card = 1

def event_exactly_2_pencils (selection: Finset String) : Prop := 
  selection.filter (fun x => x = "pencil").card = 2

def mutually_exclusive (e1 e2 : Finset String → Prop) : Prop := 
  ∀ s, ¬ (e1 s ∧ e2 s)

def opposite_events (e1 e2 : Finset String → Prop) : Prop := 
  ∀ s, e1 s = ¬ e2 s

theorem events_are_mutually_exclusive_but_not_opposite :
  mutually_exclusive event_exactly_1_pen event_exactly_2_pencils ∧ 
  ¬ opposite_events event_exactly_1_pen event_exactly_2_pencils :=
by
  sorry

end events_are_mutually_exclusive_but_not_opposite_l454_454916


namespace hiker_distance_l454_454511

theorem hiker_distance : 
  let y_final := 15 - 3 in
  let x_final := 8 - 4 in
  let d := Real.sqrt (y_final^2 + x_final^2)
  in d = 4 * Real.sqrt 10 := 
by
  let y_final := 15 - 3
  let x_final := 8 - 4
  let d := Real.sqrt (y_final^2 + x_final^2)
  sorry

end hiker_distance_l454_454511


namespace fraction_rep_finite_geom_series_036_l454_454127

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454127


namespace repeating_decimal_fraction_l454_454154

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454154


namespace repeating_decimal_fraction_eq_l454_454165

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454165


namespace domain_sqrt_function_l454_454946

noncomputable def quadratic_nonneg_for_all_x (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0

theorem domain_sqrt_function (a : ℝ) :
  quadratic_nonneg_for_all_x a ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end domain_sqrt_function_l454_454946


namespace repeating_decimal_eq_fraction_l454_454170

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454170


namespace repeating_decimal_fraction_equiv_l454_454146

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454146


namespace simplify_expression_l454_454728

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l454_454728


namespace angleBAC_l454_454686

noncomputable def triangleCondition (A B C X: Type) [GeoObject A] [GeoObject B] [GeoObject C] [GeoObject X] : Prop :=
  ∃ (ABC : Triangle A B C) (x : X),
  B.angle C = 67 ∧
  A.distanceTo B = x.distanceTo C ∧
  xA.angle C = 32 ∧
  xA.angle x = 35

theorem angleBAC (A B C X: Type) [GeoObject A] [GeoObject B] [GeoObject C] [GeoObject X] (H : triangleCondition A B C X) : A.angle (B.angle C) (X.angle A) = 81 := 
by
  sorry

end angleBAC_l454_454686


namespace problem_statement_l454_454688

/-- 
Prove that the number M of positive integers less than or equal to 1500 
whose base-2 representation has more 1's than 0's, when divided by 1000, 
gives a remainder of 182.
-/
def countSpecialBinaryNumbers : ℕ :=
  (Finset.filter (λ n : ℕ, (nat.bit_count 1 n > nat.bit_count 0 n)) (Finset.range 1501)).card

theorem problem_statement : countSpecialBinaryNumbers % 1000 = 182 := 
sorry

end problem_statement_l454_454688


namespace dawn_annual_salary_l454_454101

variable (M : ℝ)

theorem dawn_annual_salary (h1 : 0.10 * M = 400) : M * 12 = 48000 := by
  sorry

end dawn_annual_salary_l454_454101


namespace log_arithmetic_sequence_l454_454978

theorem log_arithmetic_sequence (x : ℝ) (h : log 10 2 + log 10 (2^x + 3) = 2 * log 10 (2^x - 1)) : 
  x = log 2 5 :=
  sorry

end log_arithmetic_sequence_l454_454978


namespace arithmetic_sequence_sum_of_cubes_l454_454432

theorem arithmetic_sequence_sum_of_cubes (y : ℤ) (n : ℕ) 
  (h_seq : ∀ k : ℕ, k ≤ n → (y + 3 * k) ^ 3 ≠ 0) 
  (h_sum : ∑ k in finset.range (n + 1), (y + 3 * k) ^ 3 = 2187) 
  (h_n_ineq : n > 4) 
  : n = 5 := 
sorry

end arithmetic_sequence_sum_of_cubes_l454_454432


namespace omega_sum_l454_454690

open Complex

theorem omega_sum (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) :
  ω^{17} + ω^{21} + ω^{25} + ω^{29} + ω^{33} + ω^{37} + ω^{41} + ω^{45} + ω^{49} + ω^{53} + ω^{57} + ω^{61} + ω^{65} = ω :=
by
  -- Proof goes here
  sorry

end omega_sum_l454_454690


namespace sum_of_arithmetic_sequence_l454_454784

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) (S5 : S 5 = 30) (S10 : S 10 = 110) : S 15 = 240 :=
by
  sorry

end sum_of_arithmetic_sequence_l454_454784


namespace fraction_rep_finite_geom_series_036_l454_454133

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454133


namespace cars_distance_at_least_diameter_half_time_l454_454031

theorem cars_distance_at_least_diameter_half_time :
  ∀ (t : ℝ), (0 ≤ t ∧ t ≤ 1) →
  let radius := 1
  let center_α := (0, -Real.sqrt 3)
  let center_β := (1, 0)
  let pos_A := (Real.sin (2 * Real.pi * t), -Real.sqrt 3 + Real.cos (2 * Real.pi * t))
  let pos_B := (1 - Real.cos (2 * Real.pi * t), -Real.sin (2 * Real.pi * t))
  let dist_sq := (pos_A.1 - pos_B.1)^2 + (pos_A.2 - pos_B.2)^2
  let diameter_sq := (2 * radius)^2
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (dist_sq ≥ diameter_sq) ↔ (1 / 2) :=
sorry

end cars_distance_at_least_diameter_half_time_l454_454031


namespace ratio_of_triangle_areas_eq_one_l454_454551

open EuclideanGeometry

-- Definitions of the cyclic quadrilateral and intersections
variables {A B C D O K L M : Point}

-- Defining our conditions
def cyclic_quad (A B C D : Point) : Prop :=
  ∃ (circumcircle : Circle), ∀ (P ∈ {A, B, C, D}), P ∈ circumcircle

def diagonals_intersect_at (A B C D O : Point) : Prop :=
  Line_through A C = Line_through B D ∧ O ∈ Line_through A C

def choose_K_inside_AOB_angle_bisector (A B O K C : Point) : Prop :=
  K ∈ Triangle A O B ∧ KO.bisects_angle (angle CKO)

def ray_intersect_circumcircle_again (P Q R : Point) (circumcircle : Circle) : Point :=
  ∃ x, Point_on_ray P Q x ∧ x ∈ circumcircle ∧ x ≠ R

-- The theorem statement
theorem ratio_of_triangle_areas_eq_one 
  (h1 : cyclic_quad A B C D)
  (h2 : diagonals_intersect_at A B C D O)
  (h3 : choose_K_inside_AOB_angle_bisector A B O K C) 
  (h4 : ray_intersect_circumcircle_again D K O L (circumcircle (Triangle C O K)))
  (h5 : ray_intersect_circumcircle_again C K O M (circumcircle (Triangle D O K)))
  : area (Triangle A L O) / area (Triangle B M O) = 1 :=
sorry

end ratio_of_triangle_areas_eq_one_l454_454551


namespace hyperbola_equation_l454_454246

theorem hyperbola_equation (c : ℝ) (b a : ℝ) 
  (h₁ : c = 2 * Real.sqrt 5) 
  (h₂ : a^2 + b^2 = c^2) 
  (h₃ : b / a = 1 / 2) : 
  (x y : ℝ) → (x^2 / 16) - (y^2 / 4) = 1 :=
by
  sorry

end hyperbola_equation_l454_454246


namespace max_sum_n_of_arithmetic_sequence_l454_454361

/-- Let \( S_n \) be the sum of the first \( n \) terms of an arithmetic sequence \( \{a_n\} \) with 
a non-zero common difference, and \( a_1 > 0 \). If \( S_5 = S_9 \), then when \( S_n \) is maximum, \( n = 7 \). -/
theorem max_sum_n_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (a1_pos : a 1 > 0) (common_difference : ∀ n, a (n + 1) = a n + d)
  (s5_eq_s9 : S 5 = S 9) :
  ∃ n, (∀ m, m ≤ n → S m ≤ S n) ∧ n = 7 :=
sorry

end max_sum_n_of_arithmetic_sequence_l454_454361


namespace only_fifteen_remains_l454_454387

theorem only_fifteen_remains (initial_set : Set ℕ) (h : initial_set = {1, 2, 4, 8, 16, 32}) :
  ∃ seq : List ℕ, ((∀ (x y : ℕ), x ∈ initial_set → y ∈ initial_set → x ≠ y → 
   ∃ z : ℕ, z = abs (x - y) ∧ z ∉ initial_set ∧ seq.nth_le (abs (x - y)) z) ∧ seq.length = 5 →
   ∀ n ∈ seq, n = 15 ) → false :=
by
-- Proof would be provided here
  sorry

end only_fifteen_remains_l454_454387


namespace angle_QSP_in_triangle_PQR_l454_454304

theorem angle_QSP_in_triangle_PQR
  (P Q R S : Type) [AffineSpace P] [has_segment P]
  (triangle : AffineTriangle P Q R)
  (S_on_PR : S ∈ LineSegment P R)
  (HS : Distance P S = Distance S R)
  (angle_PSR : angle P S R = (80 : ℝ)) :
  angle Q S P = (160 : ℝ) :=
sorry

end angle_QSP_in_triangle_PQR_l454_454304


namespace sector_area_is_correct_l454_454250

-- Define the given conditions
def radius : ℝ := 10
def central_angle_degrees : ℝ := 120
def central_angle_radians : ℝ := 120 * Real.pi / 180  -- converting degrees to radians

-- Define a structure to encapsulate the problem statement
structure Sector :=
  (radius : ℝ)
  (central_angle : ℝ)

-- Define a function to calculate the area of a sector
def sector_area (s : Sector) : ℝ :=
  0.5 * s.radius * s.radius * s.central_angle

-- Define an equivalent theorem to the problem statement
theorem sector_area_is_correct (s : Sector) (h_radius : s.radius = radius) (h_angle : s.central_angle = central_angle_radians) :
  sector_area s = (100 * Real.pi) / 3 := by
  sorry

end sector_area_is_correct_l454_454250


namespace sum_of_distances_l454_454209

theorem sum_of_distances : (∑ n in Finset.range 1992, (1 / (n + 1) - 1 / (n + 2))) = 1992 / 1993 :=
by
  sorry

end sum_of_distances_l454_454209


namespace cars_apart_half_time_l454_454034

-- Definitions based on conditions
def radius : ℝ := 1
def angular_velocity : ℝ := 2 * Real.pi

-- Positions of the cars at time t (in hours)
def car_A_position (t : ℝ) : ℝ × ℝ :=
  (Real.sin (angular_velocity * t), -Real.sqrt 3 + Real.cos (angular_velocity * t))

def car_B_position (t : ℝ) : ℝ × ℝ :=
  (1 - Real.cos (angular_velocity * t), -Real.sin (angular_velocity * t))

-- Squared distance between cars
def distance_squared (t : ℝ) : ℝ :=
  let pA := car_A_position t
  let pB := car_B_position t
  (pA.1 - pB.1)^2 + (pA.2 - pB.2)^2

-- Prove the time during which the distance between the cars is at least the diameter
theorem cars_apart_half_time :
  (∫ t in 0..1, if distance_squared t ≥ 4 then 1 else 0) = 1 / 2 :=
by
  sorry

end cars_apart_half_time_l454_454034


namespace first_player_wins_optimal_play_l454_454434

theorem first_player_wins_optimal_play :
  (∃ (A B : Set ℕ) (turns : ℕ → ℕ), 
    A ∪ B = {0, 1, 2, 3, 4, 5, 6} ∧ 
    ∀ n, turns n ∈ {0, 1, 2, 3, 4, 5, 6} ∧
    ∀ i, i < n → (turns i ∈ A ↔ turns i % 2 = 0) ∧
    turns n ∉ A ∪ B → B ⊂ A →
    ∃ x ∈ A, ∃ y ∈ B, (10 * x + y) % 17 = 0 ∨ (x + 10 * y) % 17 = 0 ) :=
sorry

end first_player_wins_optimal_play_l454_454434


namespace point_in_second_quadrant_l454_454321

def imaginary_unit : Type := {i : ℂ // i^2 = -1}
noncomputable def z (i : imaginary_unit) : ℂ := i.val * (1 + i.val)

theorem point_in_second_quadrant (i : imaginary_unit) : 
  let p := z i in
  p.re < 0 ∧ p.im > 0 :=
by
  let p := z i
  sorry

end point_in_second_quadrant_l454_454321


namespace find_invested_sum_l454_454049

-- Define the constants specific to the problem
constant interest_rate1 : ℝ := 0.15  -- 15% per annum
constant interest_rate2 : ℝ := 0.12  -- 12% per annum
constant time_period : ℝ := 2  -- 2 years
constant interest_difference : ℝ := 720  -- Rs. 720 difference

theorem find_invested_sum (P : ℝ) :
  (P * interest_rate1 * time_period - P * interest_rate2 * time_period = interest_difference) →
  P = 12000 :=
by
  sorry

end find_invested_sum_l454_454049


namespace chess_tournament_participants_l454_454485

/-- If each participant of a chess tournament plays exactly one game with each of the remaining participants, and 231 games are played during the tournament, then the number of participants is 22. -/
theorem chess_tournament_participants (n : ℕ) (h : (n - 1) * n / 2 = 231) : n = 22 :=
sorry

end chess_tournament_participants_l454_454485


namespace sum_powers_of_minus_one_l454_454460

/-- 
  Prove that the sum of powers of -1 from 1 to 2011 equals -1:
  ∑_{i=1}^{2011} (-1)^i = -1
-/
theorem sum_powers_of_minus_one:
  ∑ i in Finset.range 2011.succ, (-1) ^ (i + 1) = -1 := 
by
  sorry

end sum_powers_of_minus_one_l454_454460


namespace ring_speed_vertical_position_l454_454854

-- Definitions for the problem conditions
def smooth_horizontal_rod := true
def weightless_ring := true
def light_rod_length (l : ℝ) := true
def point_masses (m : ℝ) := true
def middle_mass := true
def end_mass := true
def initial_horizontal_position := true
def rod_mass_zero := true

-- The theorem to be proved
theorem ring_speed_vertical_position (m l g : ℝ) (H1 : smooth_horizontal_rod) 
                                      (H2 : weightless_ring) (H3 : light_rod_length l) 
                                      (H4 : point_masses m) (H5 : middle_mass) 
                                      (H6 : end_mass) (H7 : initial_horizontal_position)
                                      (H8 : rod_mass_zero) :
  let v := sqrt (15 * g * l) in
  true := sorry

end ring_speed_vertical_position_l454_454854


namespace sum_of_areas_S5_l454_454667

-- Definition of the points and their coordinates
def point (n : ℕ) (hn : n > 0) : ℝ × ℝ := (n, 2 / n)

-- Area of the triangle formed by the line passing through P_n and P_{n+1} with the coordinate axes
def triangleArea (n : ℕ) (hn : n > 0) : ℝ :=
  let y_intercept := (2 / n) + (2 / (n + 1))
  let x_intercept := 2 * n + 1
  (1 / 2) * y_intercept * x_intercept

-- Sum of the areas of the first n triangles
def sumOfAreas (n : ℕ) : ℝ :=
  ∑ i in finset.range n, triangleArea (i + 1) (nat.succ_pos i)

-- Theorem statement
theorem sum_of_areas_S5 : sumOfAreas 5 = 125 / 6 := sorry

end sum_of_areas_S5_l454_454667


namespace find_k_l454_454339

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l454_454339


namespace percentage_difference_in_gain_l454_454985

/-- 
If a certain percentage more is gained by selling an article for Rs. 360 than by selling 
it for Rs. 340, given that the cost of the article is Rs. 400. 
Then, the percentage difference in gain between the two selling prices is 33.33%.
-/
theorem percentage_difference_in_gain : 
  let C := 400
  let SP1 := 360
  let SP2 := 340
  let Gain1 := SP1 - C
  let Gain2 := SP2 - C
  let Difference_in_Gain := Gain1 - Gain2
  let Percentage_Difference_in_Gain := (Difference_in_Gain / |Gain2|) * 100
  Percentage_Difference_in_Gain = 33.33 := 
  by
  sorry

end percentage_difference_in_gain_l454_454985


namespace complex_shape_perimeter_l454_454670

theorem complex_shape_perimeter :
  ∃ h : ℝ, 12 * h - 20 = 95 ∧
  (24 + ((230 / 12) - 2) + 10 : ℝ) = 51.1667 :=
by
  sorry

end complex_shape_perimeter_l454_454670


namespace taxi_fare_distance_l454_454299

-- Define the initial fare and cost per additional mile segment
def initialFare : ℝ := 1.00
def costPerSegment : ℝ := 0.40
def initialDistance : ℝ := 1/5
def totalFare : ℝ := 6.60

-- Given the total fare, prove the total distance
theorem taxi_fare_distance :
  ∃ (distance : ℝ), distance = 3 ∧ totalFare = initialFare + (distance - initialDistance) / (1/5) * costPerSegment :=
by
  sorry

end taxi_fare_distance_l454_454299


namespace number_of_integers_satisfying_inequality_l454_454279

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℚ) / |n| ≥ 1 / 5}.to_finset.card = 10 :=
by {
  sorry
}

end number_of_integers_satisfying_inequality_l454_454279


namespace equation_of_hyperbola_l454_454989

theorem equation_of_hyperbola
  (x y λ : ℝ)
  (P : ℝ × ℝ)
  (hx : P.1 = 6)
  (hy : P.2 = sqrt 3)
  (asymptote_pos : y = (1 / 3) * x)
  (asymptote_neg : y = -(1 / 3) * x) :
  λ ≠ 0 → 
  (P.1 ^ 2 / 9 - P.2 ^ 2 = λ) → 
  (λ = 1) → 
  (x ^ 2 / 9 - y ^ 2 = 1) := 
  sorry

end equation_of_hyperbola_l454_454989


namespace find_p_min_area_triangle_MNF_l454_454628

-- Definitions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def line (x y : ℝ) := x - 2 * y + 1 = 0
def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def focus (p : ℝ) := (p / 2, 0)
def dot_product (M N F : ℝ × ℝ) := (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2)

-- Part 1: Finding p
theorem find_p (p : ℝ) (h : p > 0) :
  (∃ A B : ℝ × ℝ, parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  distance A B = 4 * real.sqrt 15) → p = 2 :=
sorry

-- Part 2: Minimum area of triangle MNF
theorem min_area_triangle_MNF (M N : ℝ × ℝ) (h1 : parabola 2 M.1 M.2) (h2 : parabola 2 N.1 N.2) (h3 : dot_product M N (focus 2) = 0) :
  ∃ (area : ℝ), area = 12 - 8 * real.sqrt 2 :=
sorry

end find_p_min_area_triangle_MNF_l454_454628


namespace distance_from_origin_to_point_l454_454315

theorem distance_from_origin_to_point : ∀ (x y : ℝ), x = -12 → y = 5 → sqrt (x^2 + y^2) = 13 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end distance_from_origin_to_point_l454_454315


namespace number_of_cars_l454_454867

theorem number_of_cars (x : ℕ) (h : 3 * (x - 2) = 2 * x + 9) : x = 15 :=
by {
  sorry
}

end number_of_cars_l454_454867


namespace find_integer_n_l454_454488

def s : List ℤ := [8, 11, 12, 14, 15]

theorem find_integer_n (n : ℤ) (h : (s.sum + n) / (s.length + 1) = (25 / 100) * (s.sum / s.length) + (s.sum / s.length)) : n = 30 := by
  sorry

end find_integer_n_l454_454488


namespace polynomial_never_takes_values_l454_454847

theorem polynomial_never_takes_values (P : ℤ[X]) (hP : ∀ x : ℤ, (P.coeff x).nat_degree > 0)
  (a b c d : ℤ) (h1 : P.eval a = 2) (h2 : P.eval b = 2) (h3 : P.eval c = 2) (h4 : P.eval d = 2) :
  ∀ x : ℤ, P.eval x ≠ 1 ∧ P.eval x ≠ 3 ∧ P.eval x ≠ 5 ∧ P.eval x ≠ 7 ∧ P.eval x ≠ 9 := 
by
  sorry

end polynomial_never_takes_values_l454_454847


namespace abc_ineq_l454_454244

theorem abc_ineq (a b c : ℝ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : c > 0) (h₄ : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27 / 8 :=
sorry

end abc_ineq_l454_454244


namespace more_stable_workshop_A_l454_454056

def weightsA : List ℝ := [102, 101, 99, 98, 103, 98, 99]
def weightsB : List ℝ := [110, 115, 90, 85, 75, 115, 110]

-- Function to calculate the mean of a list of weights
def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Function to calculate the variance of a list of weights
def variance (l : List ℝ) : ℝ :=
  let μ := mean l in
  (l.map (λ x => (x - μ) ^ 2)).sum / (l.length)

-- Definitions for the means and variances of weights from workshops A and B
def meanA : ℝ := mean weightsA
def meanB : ℝ := mean weightsB

def varianceA : ℝ := variance weightsA
def varianceB : ℝ := variance weightsB

theorem more_stable_workshop_A :
  varianceA < varianceB := by
  sorry

end more_stable_workshop_A_l454_454056


namespace more_action_figures_than_books_l454_454355

/-
Definitions corresponding to the conditions:
1. Jerry had 7 action figures.
2. Jerry had 2 books.
3. Jerry added 4 more books to the shelf.
-/

def jerry_action_figures : ℕ := 7
def initial_books : ℕ := 2
def added_books : ℕ := 4

-- Total number of books after adding more books
def total_books : ℕ := initial_books + added_books

-- Mathematical proof goal
theorem more_action_figures_than_books : jerry_action_figures - total_books = 1 := by
  -- initial_books + added_books = 6
  have total_books_eq : total_books = 6 := rfl
  -- jerry_action_figures = 7
  have action_figures_eq : jerry_action_figures = 7 := rfl
  -- 7 - 6 = 1
  rw [total_books_eq, action_figures_eq]
  exact Nat.sub_self 1


end more_action_figures_than_books_l454_454355


namespace third_side_not_twelve_l454_454493

theorem third_side_not_twelve (x : ℕ) (h1 : x > 5) (h2 : x < 11) (h3 : x % 2 = 0) : x ≠ 12 :=
by
  -- The proof is omitted
  sorry

end third_side_not_twelve_l454_454493


namespace repeating_decimal_fraction_equiv_l454_454147

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l454_454147


namespace tan_double_angle_l454_454576

theorem tan_double_angle (α : ℝ) (h : tan (α / 2) = 2) : tan α = -4 / 3 :=
by
  sorry

end tan_double_angle_l454_454576


namespace Y_lies_on_midline_l454_454836

variables (A B C X Y M : Point)
variables (Omega : Circle)
variables (g : Angle)
variables [RightTriangle ABC]
variables [CirclePassesThroughBC Omega B C]
variables [CircleIntersectsACAtX Omega A C X]
variables [TangentsIntersectAtY Omega X B Y]

-- To prove: Point Y lies on the midline of triangle ABC parallel to side BC or its extension
theorem Y_lies_on_midline
  (h1 : Midpoint M A C)
  (h2 : CircleTangentsIntersectAt M Y BC) :
  ∃ m : Point, Midline m Y B C :=
sorry

end Y_lies_on_midline_l454_454836


namespace locus_of_centers_is_ellipse_l454_454928

-- Definitions of points A, B, and the plane
variables (A B : point) (P : plane)

-- The intersection point of line AB with plane P and midpoint of AB 
def C : point := sorry -- (intersection point of line AB with plane P)
def M : point := sorry -- (tangency point with plane P)
def mid_AB : point := midpoint A B -- Midpoint of segment AB

-- The key theorem involving power of point theorem
axiom power_of_point (A B C M : point) :
  |C.distance M|^2 = |C.distance A| * |C.distance B|

-- The main statement proving the locus is an ellipse
theorem locus_of_centers_is_ellipse {O : point} :
  (O ∈ locus) ↔ 
  (O.distance C) = sqrt(|C.distance A| * |C.distance B|) ∧
  (O ∈ plane_of_mid_AB) →
  (locus.shape O ∧ P.perpendicular_line (line_through_mid_AB C)) ↔
  is_ellipse O C :=
sorry

end locus_of_centers_is_ellipse_l454_454928


namespace fraction_equivalent_of_repeating_decimal_l454_454195

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454195


namespace audrey_not_dreaming_time_l454_454082

theorem audrey_not_dreaming_time (total_sleep_time : ℝ) (dreaming_fraction : ℝ) :
  (total_sleep_time = 10) ∧ (dreaming_fraction = 2/5) →
  (total_sleep_time - (dreaming_fraction * total_sleep_time) = 6) :=
by
  intros h
  cases h with Htotal Hdreaming
  rw [Htotal, Hdreaming]
  sorry

end audrey_not_dreaming_time_l454_454082


namespace inequality_proof_l454_454924

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := 
by
  sorry

end inequality_proof_l454_454924


namespace student_weekly_allowance_l454_454483

-- Definitions from conditions
def weekly_allowance (A: ℝ) : Prop :=
  let spent_at_arcade := (3/5) * A in
  let remaining_after_arcade := (2/5) * A in
  let spent_at_toy_store := (1/3) * remaining_after_arcade in
  let remaining_after_toy_store := (2/3) * remaining_after_arcade in
  let spent_at_candy_store := remaining_after_toy_store in
  spent_at_candy_store = 0.60

-- Prove that weekly allowance is $2.25 given the conditions
theorem student_weekly_allowance : ∃ A, weekly_allowance A ∧ A = 2.25 :=
by
  use 2.25
  unfold weekly_allowance
  simp
  sorry

end student_weekly_allowance_l454_454483


namespace parallel_line_eq_l454_454957

theorem parallel_line_eq (b : ℝ) :
  (∃ l : ℝ → ℝ, l = (λ x, (3/4) * x + b) ∧
   ∃ d : ℝ, d = 4 ∧ 
   ∃ c1 c2 : ℝ, c1 = 6 ∧ 
   abs (c1 - b) = 5 / (real.sqrt (1 + (3/4:ℝ)^2)) ) →
  (b = 1) :=
by
  sorry

end parallel_line_eq_l454_454957


namespace inequality_proof_l454_454393

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ Real.cbrt ((ab * (c + d) + cd * (a + b)) / 4) :=
by
  sorry

end inequality_proof_l454_454393


namespace sum_of_digits_exists_l454_454403

theorem sum_of_digits_exists :
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a + b + c = 25 ∧
    d + e + f = 15 ∧
    a + b + c + d + e + f = 24 ∧
    a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    e ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    f ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry -- Proof steps are omitted

end sum_of_digits_exists_l454_454403


namespace cars_apart_half_time_l454_454033

-- Definitions based on conditions
def radius : ℝ := 1
def angular_velocity : ℝ := 2 * Real.pi

-- Positions of the cars at time t (in hours)
def car_A_position (t : ℝ) : ℝ × ℝ :=
  (Real.sin (angular_velocity * t), -Real.sqrt 3 + Real.cos (angular_velocity * t))

def car_B_position (t : ℝ) : ℝ × ℝ :=
  (1 - Real.cos (angular_velocity * t), -Real.sin (angular_velocity * t))

-- Squared distance between cars
def distance_squared (t : ℝ) : ℝ :=
  let pA := car_A_position t
  let pB := car_B_position t
  (pA.1 - pB.1)^2 + (pA.2 - pB.2)^2

-- Prove the time during which the distance between the cars is at least the diameter
theorem cars_apart_half_time :
  (∫ t in 0..1, if distance_squared t ≥ 4 then 1 else 0) = 1 / 2 :=
by
  sorry

end cars_apart_half_time_l454_454033


namespace part_I_part_II_l454_454267

-- Define the sequence a_n with given initial condition and recurrence relation
def a : ℕ → ℕ
  | 0 => 0  -- Dummy value because natural numbers in Lean start from 0
  | 1 => 2
  | (n + 1) => 2 * (a n) - n + 1

-- Part I: Prove a_2, a_3 and geometric property
theorem part_I :
  (a 2 = 4) ∧
  (a 3 = 7) ∧
  (∀ n ≥ 2, (a n - n) = 2 * (a (n - 1) - (n - 1))) :=
by
  sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (a n) / 2^(n-1)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := ∑ i in finset.range n, b (i + 1)

-- Part II: Prove the sum of the first n terms of b_n
theorem part_II (n : ℕ) :
  S n = n + 4 - (2 + n) / 2^(n-1) :=
by
  sorry

end part_I_part_II_l454_454267


namespace problem1_problem2_l454_454882

-- Proof Problem 1
theorem problem1 : 0.027 ^ (-1 / 3 : ℝ) - (-1 / 7 : ℝ) ^ (-2) + 256 ^ (3 / 4 : ℝ) - 3 ^ (-1 : ℝ) + (real.sqrt 2 - 1) ^ 0 = 19 := by
  sorry

-- Proof Problem 2
theorem problem2 : (real.logb 10 8 + real.logb 10 125 - real.logb 10 2 - real.logb 10 5) / (real.logb 10 (real.sqrt 10) * real.logb 10 0.1) = -4 := by
  sorry

end problem1_problem2_l454_454882


namespace problem_v2010_l454_454699

def sequence (n: ℕ): ℕ :=
  if n = 0 then 1
  else 
    let k := (n: ℤ).sqrt
    let k := if (k * k) < n then k + 1 else k
    let m := n - (k * (k - 1) / 2).to_nat
    4 * k + (m - 1)

theorem problem_v2010 : sequence 2010 = 7948 :=
by sorry

end problem_v2010_l454_454699


namespace sequence_k_l454_454331

theorem sequence_k (a : ℕ → ℕ) (k : ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ m n, a (m + n) = a m * a n) 
    (h3 : (finset.range 10).sum (λ i, a (k + 1 + i)) = 2^15 - 2^5) : 
    k = 4 := 
by sorry

end sequence_k_l454_454331


namespace min_φ_for_odd_function_l454_454622

noncomputable def f (x : ℝ) := 2 * cos (π * x - π / 6) ^ 2 - 1

def transformed_f (φ : ℝ) (x : ℝ) := cos (π * x - π * φ - π / 3)

def is_odd (φ : ℝ) : Prop :=
  ∀ x : ℝ, transformed_f(φ)(x) = -transformed_f(φ)(-x)

theorem min_φ_for_odd_function :
  ∃ φ > 0, φ = 1 / 6 ∧ is_odd(φ) :=
begin
  use [1 / 6, by norm_num],
  split,
  { refl },
  sorry
end

end min_φ_for_odd_function_l454_454622


namespace playground_girls_count_l454_454439

theorem playground_girls_count (boys : ℕ) (total_children : ℕ) 
  (h_boys : boys = 35) (h_total : total_children = 63) : 
  ∃ girls : ℕ, girls = 28 ∧ girls = total_children - boys := 
by 
  sorry

end playground_girls_count_l454_454439


namespace cost_of_TOP_book_l454_454527

theorem cost_of_TOP_book (T : ℝ) (h1 : T = 8)
  (abc_cost : ℝ := 23)
  (top_books_sold : ℝ := 13)
  (abc_books_sold : ℝ := 4)
  (earnings_difference : ℝ := 12)
  (h2 : top_books_sold * T - abc_books_sold * abc_cost = earnings_difference) :
  T = 8 := 
by
  sorry

end cost_of_TOP_book_l454_454527


namespace find_c_max_perimeter_l454_454305

-- Define the conditions
variables {a b c : ℝ} {θ : ℝ}

-- Condition: angle ACB is 2π/3
def angle_ACB_eq_2pi_over_3 : Prop := c = 2 * π / 3

-- Condition: sides form an arithmetic sequence with common difference 2
def sides_arithmetic_sequence : Prop := b - a = 2 ∧ c - b = 2

-- Problem 1: Find the value of c
theorem find_c (h1 : angle_ACB_eq_2pi_over_3) (h2 : sides_arithmetic_sequence) : c = 7 :=
sorry

-- Condition: area of the circumcircle is π
def area_of_circumcircle_eq_pi : Prop := π * 1^2 = π -- which means R = 1

-- Problem 2: Maximum perimeter
theorem max_perimeter (h3 : area_of_circumcircle_eq_pi) : ∃ θ ∈ (0, π / 3), 2 * sin (θ + π / 3) + sqrt 3 = 2 + sqrt 3 :=
sorry

end find_c_max_perimeter_l454_454305


namespace problem_solution_l454_454364

theorem problem_solution
  (N1 N2 : ℤ)
  (h : ∀ x : ℝ, 50 * x - 42 ≠ 0 → x ≠ 2 → x ≠ 3 → 
    (50 * x - 42) / (x ^ 2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) : 
  N1 * N2 = -6264 :=
sorry

end problem_solution_l454_454364


namespace option_B_is_same_function_option_A_is_not_same_function_option_C_is_not_same_function_option_D_is_not_same_function_l454_454785

def same_function (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x

theorem option_B_is_same_function :
  same_function (λ x, x^2 - 2 * x - 1) (λ t, t^2 - 2 * t - 1) :=
by
  sorry

theorem option_A_is_not_same_function :
  ¬ same_function (λ x, (x^2 - 1) / (x - 1)) (λ x, x + 1) :=
by
  sorry

theorem option_C_is_not_same_function :
  ¬ same_function (λ x, real.sqrt (x^2)) (λ x, x) :=
by
  sorry

theorem option_D_is_not_same_function :
  ¬ same_function (λ x, x^0 + 1) (λ x, 2) :=
by
  sorry

end option_B_is_same_function_option_A_is_not_same_function_option_C_is_not_same_function_option_D_is_not_same_function_l454_454785


namespace sufficient_not_necessary_condition_l454_454647

theorem sufficient_not_necessary_condition (x : ℝ) : (x^2 - 2 * x < 0) → (|x - 1| < 2) ∧ ¬( (|x - 1| < 2) → (x^2 - 2 * x < 0)) :=
by sorry

end sufficient_not_necessary_condition_l454_454647


namespace max_transfer_employees_range_of_a_l454_454504

variable (x : ℕ) (a : ℝ) (hx : 0 < a)
def initial_profit := 1000000  -- 1000 employees * 100,000 yuan

def remaining_profit := (1000 - x) * 1000000 * (1 + 0.002 * x) / 100000

def transferred_profit := 10000 * a * x

theorem max_transfer_employees :
  1000 - x) * 10 * (1 + 0.002 * x) ≥ 10000 → x ≤ 500 := sorry

theorem range_of_a :
  x ≤ 500 → transferred_profit ≤ remaining_profit → (0 < a) ∧ (a ≤ 5) := sorry

end max_transfer_employees_range_of_a_l454_454504


namespace find_x_l454_454465

theorem find_x 
  (x : ℝ)
  (h : 0.4 * x + (0.6 * 0.8) = 0.56) : 
  x = 0.2 := sorry

end find_x_l454_454465


namespace part1_part2_l454_454889

namespace RationalOp
  -- Define the otimes operation
  def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

  -- Part 1: Prove (-2) ⊗ 4 = -50
  theorem part1 : otimes (-2) 4 = -50 := sorry

  -- Part 2: Given x ⊗ 3 = y ⊗ (-3), prove 8x - 2y + 5 = 5
  theorem part2 (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 8*x - 2*y + 5 = 5 := sorry
end RationalOp

end part1_part2_l454_454889


namespace math_problem_l454_454324

/--
Problem: Given the conditions:
1. The parametric equation of line l is
   x = t * cos α, y = t * sin α, (t is the parameter, 0 ≤ α < π)

2. The polar equation of curve C is ρ^2 - 4 = 4ρcosθ - 2ρsinθ.

We need to prove the following:
1. The rectangular coordinate equation of curve C is 
   (x - 2)^2 + (y + 1)^2 = 9.

2. If line l intersects curve C at points A and B, 
   and the length of AB is 2√5,
   then the equation of line l is x = 0 or y = ¾x.
-/
theorem math_problem 
  (α : ℝ)
  (t : ℝ)
  (h1 : 0 ≤ α ∧ α < π)
  (h2 : ∃ θ ρ, (ρ^2 - 4 = 4 * ρ * Real.cos θ - 2 * ρ * Real.sin θ))
  (intersect_points : ∃ t1 t2, (t1 * t2 = -4) ∧ (|t1 + t2| = 2 * sqrt 5))
  :
  ∃ l : ℝ → ℝ × ℝ, l = (λ t, (t * Real.cos α, t * Real.sin α)) ∧ 
    ((α = π / 2 ∨ Real.tan α = 3 / 4) → 
      (l = (λ t, (0, t)) ∨ l = (λ t, (t, 3 / 4 * t)))) :=
by
  sorry

end math_problem_l454_454324


namespace sufficient_condition_for_neg_p_to_neg_q_l454_454604

variable (a x : ℝ)

def p : Prop := x^2 + 2 * x - 3 > 0

def q : Prop := x > a

theorem sufficient_condition_for_neg_p_to_neg_q
  (h₁ : (-3 ≤ x ∧ x ≤ 1) → (x ≤ a))
  : 1 ≤ a :=
begin
  -- The proof steps would go here, but they are omitted as instructed.
  sorry
end

end sufficient_condition_for_neg_p_to_neg_q_l454_454604


namespace maximal_segment_number_l454_454229

theorem maximal_segment_number (n : ℕ) (h : n > 4) : 
  ∃ k, k = if n % 2 = 0 then 2 * n - 4 else 2 * n - 3 :=
sorry

end maximal_segment_number_l454_454229


namespace triangle_volume_ratio_lemma_l454_454074

noncomputable def triangle_volume_ratio (a b c S : ℝ) (h_a h_b h_c : ℝ) :=
  (S = 1 / 2 * a * h_a) ∧ (S = 1 / 2 * b * h_b) ∧ (S = 1 / 2 * c * h_c) →
  (1 / a) : (1 / b) : (1 / c)

theorem triangle_volume_ratio_lemma (a b c S : ℝ) (h_a h_b h_c : ℝ) :
  (S = 1 / 2 * a * h_a) ∧ (S = 1 / 2 * b * h_b) ∧ (S = 1 / 2 * c * h_c) →
    triangle_volume_ratio a b c S h_a h_b h_c = (1 / a) : (1 / b) : (1 / c) :=
by
  sorry

end triangle_volume_ratio_lemma_l454_454074


namespace number_of_integers_satisfying_inequality_l454_454281

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℚ) / |n| ≥ 1 / 5}.to_finset.card = 10 :=
by {
  sorry
}

end number_of_integers_satisfying_inequality_l454_454281


namespace ratio_surface_area_l454_454652

-- Define the side lengths of the cubes
def side_length_larger (x : ℝ) : ℝ := 5 * x
def side_length_smaller (x : ℝ) : ℝ := x

-- Define the surface area formula for a cube
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Prove the ratio of their surface areas
theorem ratio_surface_area (x : ℝ) (hx : x > 0) :
  (surface_area (side_length_larger x)) / (surface_area (side_length_smaller x)) = 25 :=
by
  -- Define the surface areas
  let sa_larger := surface_area (side_length_larger x)
  let sa_smaller := surface_area (side_length_smaller x)

  -- Simplifying the expressions
  have sa_larger_eq : sa_larger = 6 * (5 * x)^2 := rfl
  have sa_smaller_eq : sa_smaller = 6 * (x)^2 := rfl

  -- Calculate the ratio
  calc
    sa_larger / sa_smaller
      = (6 * (5 * x)^2) / (6 * x^2) : by rw [sa_larger_eq, sa_smaller_eq]
  ... = (6 * 25 * x^2) / (6 * x^2)  : by congr; ring
  ... = 25 : by ring

end ratio_surface_area_l454_454652


namespace simplify_expression_l454_454730

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l454_454730


namespace find_number_l454_454715

theorem find_number (x : ℝ) (h : 0.3 * x - (1 / 3) * (0.3 * x) = 36) : x = 180 :=
sorry

end find_number_l454_454715


namespace shampoo_usage_l454_454524

theorem shampoo_usage (x : ℝ) (H : 0 ≤ x) : 
  (1 : ℝ) = x :=
by
  have h1 : 10 - 4 * x + 2 = (8 : ℝ) := sorry,
  have h2 : 25% of the bottle being hot sauce implies 2 = 0.25 × (10 - 4 * x + 2) := sorry,
  have h3 : 2 = 0.25 * 8 := sorry,
  have h4 : 10-4 * x + 2 = 8 := by linarith,
  have h5 : 2/0.25 = 8 := by norm_num,
  have h6 : 4 * x = 4 := sorry,
  have h7 : x = 1 := sorry,
  exact h7

end shampoo_usage_l454_454524


namespace ac_le_bc_l454_454585

theorem ac_le_bc (a b c : ℝ) (h: a > b): ∃ c, ac * c ≤ bc * c := by
  sorry

end ac_le_bc_l454_454585


namespace correct_proposition_l454_454960

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Defining proposition p
def p : Prop := ∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x < 0

-- Defining proposition q
def q : Prop := ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2

-- Theorem statement to prove the correct answer
theorem correct_proposition : (¬ p) ∧ (¬ q) :=
by
  sorry

end correct_proposition_l454_454960


namespace transformation_matrix_eq_l454_454464

noncomputable def R_90 : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]
noncomputable def S_2 : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]

theorem transformation_matrix_eq : 
  S_2.mul R_90 = !![0, 2; -2, 0] :=
by 
  sorry

end transformation_matrix_eq_l454_454464


namespace EF_eq_21_l454_454664

theorem EF_eq_21 (EFGH_square : square E F G H)
  (FR_eq_12 : FR = 12)
  (RG_eq_24 : RG = 24)
  (tan_ERS_eq_5 : tan (angle E R S) = 5)
  (S_on_EH_projection : S_projection R EH S) :
  EF = 21 := by
  sorry

end EF_eq_21_l454_454664


namespace janets_total_pockets_l454_454354

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end janets_total_pockets_l454_454354


namespace crayons_problem_l454_454508

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end crayons_problem_l454_454508


namespace sum_of_three_consecutive_integers_is_21_l454_454475

theorem sum_of_three_consecutive_integers_is_21 : 
  ∃ (n : ℤ), 3 * n = 21 :=
by
  sorry

end sum_of_three_consecutive_integers_is_21_l454_454475


namespace odd_function_sufficient_condition_odd_function_not_necessary_condition_sufficient_but_not_necessary_l454_454607

theorem odd_function_sufficient_condition 
  (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (x1 x2 : ℝ) (h_eq : x1 + x2 = 0) : 
  f(x1) + f(x2) = 0 :=
by sorry

theorem odd_function_not_necessary_condition 
  (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  ¬ (∀ x1 x2 : ℝ, f(x1) + f(x2) = 0 → x1 + x2 = 0) :=
by sorry

theorem sufficient_but_not_necessary 
  (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  (∀ x1 x2 : ℝ, x1 + x2 = 0 → f(x1) + f(x2) = 0) ∧ 
  ¬ (∀ x1 x2 : ℝ, f(x1) + f(x2) = 0 → x1 + x2 = 0) :=
by
  constructor
  . exact odd_function_sufficient_condition f h_odd
  . exact odd_function_not_necessary_condition f h_odd

end odd_function_sufficient_condition_odd_function_not_necessary_condition_sufficient_but_not_necessary_l454_454607


namespace quadrant_of_alpha_l454_454979

theorem quadrant_of_alpha (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end quadrant_of_alpha_l454_454979


namespace bacteria_growth_rate_l454_454499

theorem bacteria_growth_rate (P : ℝ) (r : ℝ) : 
  (P * r ^ 25 = 2 * (P * r ^ 24) ) → r = 2 :=
by sorry

end bacteria_growth_rate_l454_454499


namespace A_finish_time_l454_454833

-- Given definitions based on conditions
def B_work_rate_per_day := 1 / 15
def B_work_in_10_days := 10 * B_work_rate_per_day
def remaining_work := 1 - B_work_in_10_days
def A_work_in_4_days := remaining_work
def A_work_rate_per_day := A_work_in_4_days / 4

-- The statement to prove
theorem A_finish_time : 
  B_work_rate_per_day = 1 / 15 → 
  B_work_in_10_days = 10 / 15 → 
  remaining_work = 1 / 3 → 
  A_work_in_4_days = 1 / 3 → 
  A_work_rate_per_day = 1 / 12 → A_finish_time = 12 :=
begin
  -- Definitions from conditions
  intros,
  unfold B_work_rate_per_day at *,
  unfold B_work_in_10_days at *,
  unfold remaining_work at *,
  unfold A_work_in_4_days at *,
  unfold A_work_rate_per_day at *,
  sorry
end

end A_finish_time_l454_454833


namespace max_sum_x1_x2_x3_l454_454370

open Nat

theorem max_sum_x1_x2_x3 (x : ℕ → ℕ) (h_nat : ∀ n, x n ∈ ℕ)
  (h_inc : ∀ n, x n < x (n + 1)) (h_sum : sum (range 7) x = 159) :
  ∃ x1 x2 x3, x1 ∈ ℕ ∧ x2 ∈ ℕ ∧ x3 ∈ ℕ ∧ x1 < x2 ∧ x2 < x3 ∧ x1 + x2 + x3 = 61 := 
by
  sorry

end max_sum_x1_x2_x3_l454_454370


namespace movie_ticket_distribution_l454_454107

theorem movie_ticket_distribution : 
  ∃ (n : ℕ), n = 10 * 9 * 8 ∧ ∀ (tickets students : Type), 
  fintype tickets ∧ fintype students → (fintype.card tickets = 3 ∧ fintype.card students = 10) → 
  n = 720 :=
by
  sorry

end movie_ticket_distribution_l454_454107


namespace calculate_percentages_l454_454412

-- Define the conditions
def total_degrees : ℝ := 360
def manufacturing_degrees : ℝ := 252
def administration_degrees : ℝ := 68
def research_degrees : ℝ := 40

-- Define the percentage calculation
def calculate_percentage (degrees : ℝ) : ℝ :=
  (degrees / total_degrees) * 100

-- State the theorem
theorem calculate_percentages :
  calculate_percentage manufacturing_degrees = 70 ∧
  calculate_percentage administration_degrees = 18.89 ∧
  calculate_percentage research_degrees = 11.11 :=
by
  sorry

end calculate_percentages_l454_454412


namespace minimum_value_of_x_l454_454843

variable (x : ℝ)

noncomputable def total_sales_from_january_to_may := 38.6
noncomputable def sales_june := 5
noncomputable def increase_percentage_july := (1 + x / 100)
noncomputable def increase_percentage_august := (1 + x / 100)^2
noncomputable def sales_july := sales_june * increase_percentage_july
noncomputable def sales_august := sales_june * increase_percentage_august
noncomputable def total_sales_july_august := sales_july + sales_august
noncomputable def total_sales_september_october := total_sales_july_august
noncomputable def total_sales_january_to_october := total_sales_from_january_to_may + sales_june + 2 * total_sales_july_august

theorem minimum_value_of_x (h : total_sales_january_to_october x ≥ 70) : x ≥ 20 := by
  sorry

end minimum_value_of_x_l454_454843


namespace probability_of_300_points_probability_of_at_least_300_points_l454_454855

theorem probability_of_300_points :
  let P_A1 := 0.8
  let P_A2 := 0.7
  let P_A3 := 0.6
  let P_not_A2 := 1 - P_A2
  let P_not_A1 := 1 - P_A1 in
  (P_A1 * P_not_A2 * P_A3 + P_not_A1 * P_A2 * P_A3 = 0.228) := by
  let P_A1 := 0.8
  let P_A2 := 0.7
  let P_A3 := 0.6
  let P_not_A2 := 1 - P_A2
  let P_not_A1 := 1 - P_A1
  sorry

theorem probability_of_at_least_300_points :
  let P_A1 := 0.8
  let P_A2 := 0.7
  let P_A3 := 0.6
  let P_not_A2 := 1 - P_A2
  let P_not_A1 := 1 - P_A1
  let P_exact_300 := P_A1 * P_not_A2 * P_A3 + P_not_A1 * P_A2 * P_A3 in
  (P_exact_300 + P_A1 * P_A2 * P_A3 = 0.564) := by
  let P_A1 := 0.8
  let P_A2 := 0.7
  let P_A3 := 0.6
  let P_not_A2 := 1 - P_A2
  let P_not_A1 := 1 - P_A1
  let P_exact_300 := P_A1 * P_not_A2 * P_A3 + P_not_A1 * P_A2 * P_A3
  sorry

end probability_of_300_points_probability_of_at_least_300_points_l454_454855


namespace geometric_series_sum_eq_l454_454803

-- Given conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 5

-- Define the geometric series sum formula
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The main theorem to prove
theorem geometric_series_sum_eq : geometric_series_sum a r n = 31 / 32 := by
  sorry

end geometric_series_sum_eq_l454_454803


namespace effective_percentage_change_l454_454505

def original_price (P : ℝ) : ℝ := P
def annual_sale_discount (P : ℝ) : ℝ := 0.70 * P
def clearance_event_discount (P : ℝ) : ℝ := 0.80 * (annual_sale_discount P)
def sales_tax (P : ℝ) : ℝ := 1.10 * (clearance_event_discount P)

theorem effective_percentage_change (P : ℝ) :
  (sales_tax P) = 0.616 * P := by
  sorry

end effective_percentage_change_l454_454505


namespace cyclic_quadrilateral_of_proportional_division_l454_454429

theorem cyclic_quadrilateral_of_proportional_division
  (A B C D P : Point)
  (hAB : LineSegment A B)
  (hCD : LineSegment C D)
  (hP : IntersectionPoint hAB hCD P)
  (hProp : (P.dist A / P.dist B) = (P.dist C / P.dist D)) :
  CyclicQuadrilateral A B C D := 
sorry

end cyclic_quadrilateral_of_proportional_division_l454_454429


namespace fraction_rep_finite_geom_series_036_l454_454131

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454131


namespace geometric_sum_formula_not_geometric_seq_l454_454595

noncomputable def geometric_sum {α : Type*} [Field α] (a1 q : α) (n : ℕ) : α :=
  if q = 1 then
    n * a1
  else
    a1 * (1 - q^n) / (1 - q)

theorem geometric_sum_formula {α : Type*} [Field α] (a1 q : α) (n : ℕ) :
  geometric_sum a1 q n = if q = 1 then
                            n * a1
                         else
                            a1 * (1 - q^n) / (1 - q) :=
by
  sorry

theorem not_geometric_seq {α : Type*} [Field α] (a1 q : α) (n : ℕ) (hq : q ≠ 1) :
  ¬ ∀ r : α, ∀ m : ℕ, { k : ℕ // k < m} → a1 * (q^(k.val)) + 1 = a1 * (q^(k.val)) * (r^(k.val)) + 1 :=
by
  sorry

end geometric_sum_formula_not_geometric_seq_l454_454595


namespace fourth_root_difference_l454_454879

theorem fourth_root_difference : (81 : ℝ) ^ (1 / 4 : ℝ) - (1296 : ℝ) ^ (1 / 4 : ℝ) = -3 :=
by
  sorry

end fourth_root_difference_l454_454879


namespace gcd_123456_789012_l454_454197

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := sorry

end gcd_123456_789012_l454_454197


namespace max_volume_triangle_pyramid_l454_454066

theorem max_volume_triangle_pyramid
  (r : ℝ)
  (S A B C O : EuclideanGeometry.Point ℝ)
  (surface_area_sphere : real.pi * r^2 = 60 * real.pi)
  (points_on_sphere : ∀ X ∈ [S, A, B, C], dist O X = r)
  (equilateral_triangle : EuclideanGeometry.equilateral_triangle A B C)
  (dist_center_to_plane : EuclideanGeometry.dist_point_to_plane O (EuclideanGeometry.plane_of_triangle A B C) = real.sqrt 3)
  (plane_perpendicular : EuclideanGeometry.perpendicular (EuclideanGeometry.plane_of_points S A B) (EuclideanGeometry.plane_of_triangle A B C))
  : ( ∃ V : ℝ, V = 1/3 * (3 * real.sqrt 3) * (9 * real.sqrt 3) ∧ V = 27) :=
sorry

end max_volume_triangle_pyramid_l454_454066


namespace M_inter_N_eq_123_l454_454271

def M : Set ℤ := {x | x^2 - 3 * x - 4 ≤ 0}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem M_inter_N_eq_123 : M ∩ N = {1, 2, 3} := sorry

end M_inter_N_eq_123_l454_454271


namespace repeating_decimal_to_fraction_l454_454122

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454122


namespace eve_distance_ran_more_l454_454558

variable (ran walked : ℝ)

def eve_distance_difference (ran walked : ℝ) : ℝ :=
  ran - walked

theorem eve_distance_ran_more :
  eve_distance_difference 0.7 0.6 = 0.1 :=
by
  sorry

end eve_distance_ran_more_l454_454558


namespace find_p_min_area_triangle_MNF_l454_454626

-- Definitions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def line (x y : ℝ) := x - 2 * y + 1 = 0
def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def focus (p : ℝ) := (p / 2, 0)
def dot_product (M N F : ℝ × ℝ) := (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2)

-- Part 1: Finding p
theorem find_p (p : ℝ) (h : p > 0) :
  (∃ A B : ℝ × ℝ, parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  distance A B = 4 * real.sqrt 15) → p = 2 :=
sorry

-- Part 2: Minimum area of triangle MNF
theorem min_area_triangle_MNF (M N : ℝ × ℝ) (h1 : parabola 2 M.1 M.2) (h2 : parabola 2 N.1 N.2) (h3 : dot_product M N (focus 2) = 0) :
  ∃ (area : ℝ), area = 12 - 8 * real.sqrt 2 :=
sorry

end find_p_min_area_triangle_MNF_l454_454626


namespace taxi_ride_cost_l454_454856

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end taxi_ride_cost_l454_454856


namespace simplify_expression_l454_454731

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l454_454731


namespace simplify_fraction_l454_454733

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454733


namespace repeating_decimal_fraction_eq_l454_454168

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454168


namespace AB_perpendicular_AD_l454_454657

open EuclideanGeometry

variables {A B C D P Q : Point}

-- Convex quadrilateral ABCD
axiom convex_ABCD : ConvexQuadrilateral A B C D
-- Points P and Q on sides AB and CD respectively
axiom P_on_AB : LiesOn P A B
axiom Q_on_CD : LiesOn Q C D
-- Given conditions
axiom AQ_parallel_CP : parallel AQ CP
axiom BQ_parallel_DP : parallel BQ DP
axiom AB_perpendicular_BC : Perpendicular A B B C
axiom CD_perpendicular_DP : Perpendicular C D D P

theorem AB_perpendicular_AD :
  Perpendicular A B A D :=
by exact sorry

end AB_perpendicular_AD_l454_454657


namespace cosine_greater_sine_cosine_cos_greater_sine_sin_l454_454831

variable {f g : ℝ → ℝ}

-- Problem 1
theorem cosine_greater_sine (h : ∀ x, - (Real.pi / 2) < f x + g x ∧ f x + g x < Real.pi / 2
                            ∧ - (Real.pi / 2) < f x - g x ∧ f x - g x < Real.pi / 2) :
  ∀ x, Real.cos (f x) > Real.sin (g x) :=
sorry

-- Problem 2
theorem cosine_cos_greater_sine_sin (x : ℝ) :  Real.cos (Real.cos x) > Real.sin (Real.sin x) :=
sorry

end cosine_greater_sine_cosine_cos_greater_sine_sin_l454_454831


namespace minimum_n_S_n_l454_454783

noncomputable def a1 : ℝ := sorry -- Derived from the conditions in the solution
noncomputable def d : ℝ := sorry -- Derived from the conditions in the solution

def S (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

theorem minimum_n_S_n : ∃ n : ℕ, n > 0 ∧ nS_n n = -49 :=
by
  let S_n := λ n : ℕ, n * S n
  sorry

end minimum_n_S_n_l454_454783


namespace part1_part2_part3_l454_454646

-- Define the given triangle properties
variables {b h : ℝ} (n : ℝ)
def area_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

-- Part 1: Prove height of smaller triangle is h / 2 when area is 1/4 of the given triangle
theorem part1 (A : ℝ) (h' : ℝ) :
  A = area_triangle b h →
  h' = h / 2 →
  area_triangle b h' = A / 4 :=
sorry

-- Part 2: Prove height of smaller triangle is h / sqrt(n) when area is 1/n of the given triangle
theorem part2 (A : ℝ) (h_n : ℝ) :
  A = area_triangle b h →
  h_n = h / real.sqrt n →
  area_triangle b h_n = A / n :=
sorry

-- Part 3: Prove removal of triangle of area (n-1)/n * A forms quadrilateral with area 1/n * A
theorem part3 (A remove : ℝ) :
  A = area_triangle b h →
  remove = (n - 1) / n * A →
  A - remove = 1 / n * A :=
sorry

end part1_part2_part3_l454_454646


namespace vector_sum_zero_l454_454224

noncomputable def is_long {n: ℕ} {d: Type} [inner_product_space ℝ d] (v: fin n → d) (i: fin n) : Prop :=
  ∥v i∥ ≥ ∥(∑ j: fin n, if i ≠ j then v j else 0)∥

theorem vector_sum_zero (n: ℕ) (v: fin n → ℝ^2) (h_n : 2 < n) 
  (h_long : ∀i, is_long v i) : (∑ i in finset.univ, v i) = 0 := 
by
  sorry

end vector_sum_zero_l454_454224


namespace inequality_proof_l454_454264

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l454_454264


namespace shaded_area_difference_l454_454419

theorem shaded_area_difference (A1 A3 A4 : ℚ) (h1 : 4 = 2 * 2) (h2 : A1 + 5 * A1 + 7 * A1 = 6) (h3 : p + q = 49) : 
  ∃ p q : ℕ, p + q = 49 ∧ p = 36 ∧ q = 13 :=
by {
  sorry
}

end shaded_area_difference_l454_454419


namespace describe_S_l454_454441

-- Define points A, B given the conditions
def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 0⟩

-- Define what it means for C to create triangle ABC with area 2
def area_triangle_ABC (C : Point) : ℝ := (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- Define the set S' of points C
def S' : Set Point := {C | area_triangle_ABC C = 2}

-- Define the mathematical description of S' as two parallel lines
def two_parallel_lines (l₁ l₂ : Line) : Prop :=
  l₁ ≠ l₂ ∧ ∀ (P : Point), (P ∈ l₁ ↔ distance_from_line l₂ P = 2)

-- Statement of the problem in Lean 4
theorem describe_S' : ∃ (l₁ l₂ : Line), two_parallel_lines l₁ l₂ ∧ ∀ (C : Point), C ∈ S' ↔ (C ∈ l₁ ∨ C ∈ l₂) :=
by
  sorry

end describe_S_l454_454441


namespace divisible_by_4_l454_454402

theorem divisible_by_4
  (n : ℕ)
  (a : ℕ → ℤ)
  (h_range : ∀ i, a i = 1 ∨ a i = -1)
  (h_sum : ∑ i in finset.range n, a i * a (i + 1) * a (i + 2) * a (i + 3) = 0) :
  4 ∣ n :=
sorry

end divisible_by_4_l454_454402


namespace derivative_f1_derivative_f2_l454_454562

-- Define the first function f(x) = 2 * ln x
def f1 (x : ℝ) : ℝ := 2 * Real.log x

-- State a theorem to prove the derivative of f1(x)
theorem derivative_f1 (x : ℝ) (hx : 0 < x) : deriv f1 x = 2 / x :=
by {
  sorry
}

-- Define the second function f(x) = exp x / x
def f2 (x : ℝ) : ℝ := Real.exp x / x

-- State a theorem to prove the derivative of f2(x)
theorem derivative_f2 (x : ℝ) (hx : 0 < x) : deriv f2 x = (x * Real.exp x - Real.exp x) / (x^2) :=
by {
  sorry
}

end derivative_f1_derivative_f2_l454_454562


namespace sum_of_integers_l454_454011

theorem sum_of_integers (n : ℕ) (h : 1 ≤ n) : (finset.range n).sum id = n * (n + 1) / 2 :=
sorry

end sum_of_integers_l454_454011


namespace incorrect_inequality_l454_454582

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ∃ c : ℝ, ac = bc :=
by
  have h1 : ¬(ac > bc) := by
    let c := 0
    show ac = bc 
    sorry

  exact ⟨0, h1⟩

end incorrect_inequality_l454_454582


namespace repeating_decimal_to_fraction_l454_454117

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454117


namespace when_did_B_join_l454_454047

theorem when_did_B_join
  (A_investment : ℝ := 27000)
  (B_investment : ℝ := 36000)
  (profit_ratio : ℝ := 2 / 1)
  (A_time : ℝ := 12)
  (B_time : ℝ)
  (profit_A : ℝ)
  (profit_B : ℝ) :
  (profit_A / profit_B = profit_ratio) → 
  (A_investment * A_time / (B_investment * B_time) = profit_ratio) →
  B_time = (12 - 7.5) :=
by
  intro hp ha
  sorry

end when_did_B_join_l454_454047


namespace repeating_decimal_eq_fraction_l454_454174

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454174


namespace same_days_to_dig_scenario_l454_454484

def volume (depth length breadth : ℝ) : ℝ :=
  depth * length * breadth

def days_to_dig (depth length breadth days : ℝ) : Prop :=
  ∃ (labors : ℝ), 
    (volume depth length breadth) * days = (volume 100 25 30) * 12

theorem same_days_to_dig_scenario :
  days_to_dig 75 20 50 12 :=
sorry

end same_days_to_dig_scenario_l454_454484


namespace solveEquation_correct_l454_454750

noncomputable def solveEquation : ℝ :=
  let x := (992 : ℝ) / 15
  x

theorem solveEquation_correct :
  ∃ x : ℝ, (real.cbrt (15 * x + real.cbrt (15 * x + 8))) = 10 ∧ x = solveEquation :=
by
  let x := (992 : ℝ) / 15
  use x
  sorry -- Here we would provide the proof, which is omitted as per instruction

end solveEquation_correct_l454_454750


namespace inequality_solution_set_l454_454248

noncomputable def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem inequality_solution_set (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_A : f 0 = -2)
  (h_B : f 3 = 2) :
  {x : ℝ | |f (x+1)| ≥ 2} = {x | x ≤ -1} ∪ {x | x ≥ 2} :=
sorry

end inequality_solution_set_l454_454248


namespace fraction_repeating_decimal_l454_454183

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l454_454183


namespace inequality_hold_l454_454261

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l454_454261


namespace new_ratio_of_boarders_to_day_students_l454_454775
-- Import the necessary library

-- Define the conditions stated
variable (b_initial : ℕ) (d_initial : ℕ) (b_ratio : ℕ) (d_ratio : ℕ) (b_new_added : ℕ)
variable (b_initial = 220) (b_ratio = 5) (d_ratio = 12) (b_new_added = 44)

-- Define the hypothesis based on the given conditions
def initial_conditions := 
  b_initial / d_initial = b_ratio / d_ratio ∧ 
  b_initial = 220 ∧ 
  b_new_added = 44

-- Define the statement to prove, the new ratio being 1 to 2
theorem new_ratio_of_boarders_to_day_students :
  initial_conditions b_initial d_initial b_ratio d_ratio b_new_added →
  (b_initial + b_new_added) / d_initial = 1 / 2 := 
by
  -- proof can be constructed here
  sorry

end new_ratio_of_boarders_to_day_students_l454_454775


namespace least_possible_integer_l454_454839

theorem least_possible_integer (N : ℕ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ N) ∧
  (∀ m : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ m) → N ≤ m) →
  N = 2329089562800 :=
sorry

end least_possible_integer_l454_454839


namespace rod_total_length_l454_454063

theorem rod_total_length (n : ℕ) (piece_length : ℝ) (total_length : ℝ) 
  (h1 : n = 50) 
  (h2 : piece_length = 0.85) 
  (h3 : total_length = n * piece_length) : 
  total_length = 42.5 :=
by
  -- Proof steps will go here
  sorry

end rod_total_length_l454_454063


namespace vector_magnitude_subtraction_l454_454316

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_magnitude_subtraction
  (a b c : V)
  (ha : ∥a∥ = 1)
  (hc : ∥c∥ = real.sqrt 2)
  (habc : a + b = c) :
  ∥b - a - c∥ = 2 :=
sorry

end vector_magnitude_subtraction_l454_454316


namespace determine_angles_l454_454545

theorem determine_angles 
  (small_angle1 : ℝ) 
  (small_angle2 : ℝ) 
  (large_angle1 : ℝ) 
  (large_angle2 : ℝ) 
  (triangle_sum_property : ∀ a b c : ℝ, a + b + c = 180) 
  (exterior_angle_property : ∀ a c : ℝ, a + c = 180) :
  (small_angle1 = 70) → 
  (small_angle2 = 180 - 130) → 
  (large_angle1 = 45) → 
  (large_angle2 = 50) → 
  ∃ α β : ℝ, α = 120 ∧ β = 85 :=
by
  intros h1 h2 h3 h4
  sorry

end determine_angles_l454_454545


namespace energy_increase_l454_454440

def distance (a b : ℕ) : ℝ := a / b

def energy_inversely_proportional (E d : ℝ) : ℝ :=
  E / d

theorem energy_increase (E_initial : ℝ) (d : ℝ) (charge_pairs : ℕ) (new_dist_1 new_dist_2 : ℝ) :
  E_initial = 18 →
  charge_pairs = 3 →
  new_dist_1 = distance 2 3 * d →
  new_dist_2 = distance 4 3 * d →
  let E_initial_pair := E_initial / charge_pairs in
  let E_new1 := energy_inversely_proportional E_initial_pair new_dist_1 in
  let E_new2 := energy_inversely_proportional E_initial_pair new_dist_2 in
  let E_other := E_initial_pair in
  let E_new_total := E_new1 + E_new2 + E_other in
  (E_new_total - E_initial) = 1.5 :=
by
  intros
  sorry
  -- Proof logic goes here

end energy_increase_l454_454440


namespace invitations_per_pack_l454_454883

/--
Carol was sending out birthday invitations to 12 friends. She bought 3 packs each one having the same number of invitations.
How many invitations are in each pack?
-/
theorem invitations_per_pack (total_invitations packs : ℕ) (h1 : total_invitations = 12) (h2 : packs = 3) : total_invitations / packs = 4 :=
by
  rw [h1, h2]
  norm_num

end invitations_per_pack_l454_454883


namespace point_S_traverses_constant_radius_arc_l454_454309

theorem point_S_traverses_constant_radius_arc
  (O : Point) (AB : Line) (circle : Circle O)
  (PQ : Line) (R : Point)
  (h_diameter : is_diameter O AB)
  (h_chord : is_chord PQ circle)
  (h_parallel : is_parallel PQ AB)
  (h_R_on_PQ : R ∈ PQ)
  (S : Point)
  (h_OR_meets_circle : ∃ S, meets_extension_at_circle OR circle) :
  traverses_constant_radius_arc S circle :=
sorry

end point_S_traverses_constant_radius_arc_l454_454309


namespace percent_preferred_strawberries_l454_454759

theorem percent_preferred_strawberries 
(hApples : ℕ := 80)
(hBananas : ℕ := 70)
(hStrawberries : ℕ := 90)
(hOranges : ℕ := 60)
(hGrapes : ℕ := 50) :
  (hStrawberries.toRat / (hApples + hBananas + hStrawberries + hOranges + hGrapes).toRat) * 100 = 25.71 := 
sorry

end percent_preferred_strawberries_l454_454759


namespace final_price_after_adjustments_l454_454413

theorem final_price_after_adjustments (p : ℝ) :
  let increased_price := p * 1.30
  let discounted_price := increased_price * 0.75
  let final_price := discounted_price * 1.10
  final_price = 1.0725 * p :=
by
  sorry

end final_price_after_adjustments_l454_454413


namespace time_taken_by_A_l454_454026

theorem time_taken_by_A (v_A v_B D t_A t_B : ℚ) (h1 : v_A / v_B = 3 / 4) 
  (h2 : t_A = t_B + 30) (h3 : t_A = D / v_A) (h4 : t_B = D / v_B) 
  : t_A = 120 := 
by 
  sorry

end time_taken_by_A_l454_454026


namespace chris_money_before_birthday_l454_454092

-- Define the given amounts of money from each source
def money_from_grandmother : ℕ := 25
def money_from_aunt_and_uncle : ℕ := 20
def money_from_parents : ℕ := 75
def total_money_now : ℕ := 279

-- Calculate the total birthday money
def total_birthday_money := money_from_grandmother + money_from_aunt_and_uncle + money_from_parents

-- Define the amount of money Chris had before his birthday
def money_before_birthday := total_money_now - total_birthday_money

-- The proof statement
theorem chris_money_before_birthday : money_before_birthday = 159 :=
by
  sorry

end chris_money_before_birthday_l454_454092


namespace figure_F12_diamonds_l454_454599

-- Definitions based on conditions
def initial_diamonds : ℕ := 3

def added_diamonds (n : ℕ) : ℕ := 8 * n

def total_diamonds : ℕ → ℕ
| 1 := initial_diamonds
| n + 1 := total_diamonds n + added_diamonds (n + 1)

-- Proposition to be proved
theorem figure_F12_diamonds : total_diamonds 12 = 619 :=
sorry

end figure_F12_diamonds_l454_454599


namespace find_x_floor_l454_454905

theorem find_x_floor : ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 29 / 4 ∧ x = 29 / 4 := 
by
  sorry

end find_x_floor_l454_454905


namespace correct_derivative_is_D_l454_454472

noncomputable def verify_derivatives :=
  (∂ sin x ≠ - cos x) ∧
  (∂ cos x ≠ sin x) ∧
  (∂ (2^x) ≠ x * 2^(x-1)) ∧
  (∂ (1/x) = -(1/x^2))

theorem correct_derivative_is_D (x : ℝ) : verify_derivatives :=
by {
  sorry
}

end correct_derivative_is_D_l454_454472


namespace player_placing_third_won_against_seventh_l454_454040

theorem player_placing_third_won_against_seventh :
  ∃ (s : Fin 8 → ℚ),
    -- Condition 1: Scores are different
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    -- Condition 2: Second place score equals the sum of the bottom four scores
    (s 1 = s 4 + s 5 + s 6 + s 7) ∧
    -- Result: Third player won against the seventh player
    (s 2 > s 6) :=
sorry

end player_placing_third_won_against_seventh_l454_454040


namespace unique_solution_for_function_l454_454906

theorem unique_solution_for_function (f : ℝ → ℝ) :
  (∀ x y : ℝ, 2 * f x + 2 * f y - f x * f y ≥ 4) → (∀ x : ℝ, f x = 2) :=
by
  intro h
  funext x
  -- the proof will follow the steps in the solution
  -- sorry is used here to skip the proof
  sorry

end unique_solution_for_function_l454_454906


namespace maximal_triangle_area_l454_454696

open Real

def parabola (p : ℝ) := -p^2 + 6 * p - 5

def area_ABC (p : ℝ) : ℝ :=
  abs (-4 * p^2 + 7 * p) / 2

theorem maximal_triangle_area :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 5 ∧ area_ABC p = 16.125 :=
by {
  use 5,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
}

end maximal_triangle_area_l454_454696


namespace probability_of_point_closer_to_six_than_two_l454_454514

noncomputable def probability_point_closer_to_six_than_two (x : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 8) then
  if (|x - 6| < |x - 2|) then
    1
  else
    0
else
  0

theorem probability_of_point_closer_to_six_than_two : 
  ((∫ x in 0..8, probability_point_closer_to_six_than_two x) / 8) = 0.5 :=
by
  sorry

end probability_of_point_closer_to_six_than_two_l454_454514


namespace constant_term_in_expansion_l454_454566

-- Statement: Prove that for n = 5, the expansion of (3x + 1 / (x * sqrt x))^n contains a constant term given that n is a positive natural number.
theorem constant_term_in_expansion (n : ℕ) (h : n = 5) : 
  ∃ k : ℕ, (3 * x + 1 / (x * real.sqrt x)) ^ n = 3^k * (3 * x + 1 / (x * real.sqrt x))^(n - k) * (x^(5 / 2) * (x * real.sqrt x)^(-k)) ∧ x^(n - 5 / 2 * k) = 1 :=
by {
  sorry
}

end constant_term_in_expansion_l454_454566


namespace solve_for_k_l454_454977

theorem solve_for_k : ∀ k : ℝ, (∫ x in 0..k, x - 2*x^2) = 0 → k > 0 → k = 3/4 :=
by
  intro k
  intro integral_eq_zero
  intro k_positive
  sorry

end solve_for_k_l454_454977


namespace selection_methods_with_atleast_one_girl_l454_454050

open Nat

theorem selection_methods_with_atleast_one_girl (n k b g : ℕ) (h1 : n = b + g)
    (h2 : b = 4) (h3 : g = 2) (h4 : k = 4) :
    (choose n k) - (choose b b) = 14 :=
by
  rw [h1, h2, h3, h4]
  -- We simplify our terms using the properties of combinations.
  -- choose 6 4 - choose 4 4 = 14
  have h_total : choose (4 + 2) 4 = 15 := by norm_num
  have h_boys : choose 4 4 = 1 := by norm_num
  rw [h_total, h_boys]
  norm_num
  sorry

end selection_methods_with_atleast_one_girl_l454_454050


namespace repeating_decimal_fraction_l454_454152

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454152


namespace parallelogram_area_l454_454390

-- Given conditions
variables (angle1 : ℝ) (side1 : ℝ) (side2 : ℝ)
-- Defining given values
def angle_ABC : ℝ := 150
def side_AB : ℝ := 18
def side_BC : ℝ := 10

-- Correct answer
def area_parallelogram : ℝ := 60 * Real.sqrt 3

-- Lean 4 statement for the proof problem
theorem parallelogram_area :
  ∀ (angle_ABC = 150) (side_AB = 18) (side_BC = 10), parallelogram_area = 60 * Real.sqrt 3 :=
by sorry

end parallelogram_area_l454_454390


namespace ratio_of_perimeters_l454_454822

theorem ratio_of_perimeters (s : ℝ) (h : s > 0) :
  let diagonal_small := s * Real.sqrt 2,
      diagonal_large := 7 * diagonal_small,
      side_large := diagonal_large / Real.sqrt 2,
      perimeter_small := 4 * s,
      perimeter_large := 4 * side_large
  in perimeter_large / perimeter_small = 7 :=
by
  let diagonal_small := s * Real.sqrt 2
  let diagonal_large := 7 * diagonal_small
  let side_large := diagonal_large / Real.sqrt 2
  let perimeter_small := 4 * s
  let perimeter_large := 4 * side_large
  have h₁ : side_large = 7 * s := by
    sorry
  have h₂ : perimeter_large = 28 * s := by
    sorry
  have h₃ : perimeter_small = 4 * s := by
    sorry
  show (perimeter_large / perimeter_small = 7)
  by
    sorry

end ratio_of_perimeters_l454_454822


namespace part1_part2_l454_454792

open ProbabilityTheory MeasureTheory

noncomputable theory

namespace FireCompetition

variables {Ω : Type*} [ProbabilitySpace Ω]
variables {A1 A2 B1 B2 : Event Ω}
variables (PA1 PA2 PB1 PB2 : ℚ)

-- Given conditions
def P_A1 : Prop := PA1 = 4 / 5
def P_B1 : Prop := PB1 = 3 / 5
def P_A2 : Prop := PA2 = 2 / 3
def P_B2 : Prop := PB2 = 3 / 4

-- Independence condition
def independence : Prop := Independent (A1 :: B1 :: A2 :: B2 :: [])

-- Proof of Part 1: Probability of A winning exactly one round
theorem part1 (h1 : P_A1) (h2 : P_A2) (ind : independence) :
  (Prob (A1 \ (A2 ∩ ¬(A2)))).Val + (Prob ((¬A1) ∩ A2)).Val = 2 / 5 :=
sorry

-- Proof of Part 2: Probability that at least one of A or B wins the competition
theorem part2 (h1 : P_A1) (h2 : P_A2) (h3 : P_B1) (h4 : P_B2) (ind : independence) :
  (1 - Prob ((¬A1 ∩ ¬A2) ∩ (¬B1 ∩ ¬B2))).Val = 223 / 300 :=
sorry

end FireCompetition

end part1_part2_l454_454792


namespace distance_of_complex_points_l454_454668

noncomputable def distance_complex_points : ℂ → ℂ → ℝ
| z1, z2 => 
  let x1 := z1.re
  let y1 := z1.im
  let x2 := z2.re
  let y2 := z2.im
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_of_complex_points :
  distance_complex_points (-3 + Complex.i) (1 - Complex.i) = 2 * Real.sqrt 5 := 
sorry

end distance_of_complex_points_l454_454668


namespace function_monotonically_decreasing_interval_l454_454910

theorem function_monotonically_decreasing_interval (k : ℤ) :
  ∀ x, k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12 → 
  (λ x, sqrt 3 * cos (2 * x) - sin (2 * x)) x ≤ (λ x, sqrt 3 * cos (2 * x) - sin (2 * x)) (x + 1) → 
  ∀ x > k * Real.pi + 5 * Real.pi / 12, (λ x, sqrt 3 * cos (2 * x) - sin (2 * x)) x < (λ x, sqrt 3 * cos (2 * x) - sin (2 * x)) (x - 1) :=
begin
  sorry
end

end function_monotonically_decreasing_interval_l454_454910


namespace ron_eats_24_l454_454722

variable (S : ℕ) (T : ℕ) (R : ℕ)

axiom hS : S = 15
axiom hT : T = 2 * S
axiom hR : R = T - (T / 5)

theorem ron_eats_24 : R = 24 := by
  rw [hS, hT]
  sorry

end ron_eats_24_l454_454722


namespace incorrect_inequality_l454_454578

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end incorrect_inequality_l454_454578


namespace fraction_equivalent_of_repeating_decimal_l454_454188

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l454_454188


namespace total_participants_l454_454401

theorem total_participants (x : ℕ) (h1 : 800 / x + 60 = 800 / (x - 3)) : x = 8 :=
sorry

end total_participants_l454_454401


namespace octagonal_lattice_triangle_count_l454_454900

def is_lattice_point (p : ℕ × ℕ) : Prop :=
  (p.1 ^ 2 + p.2 ^ 2 = 1) ∨ 
  (p.1 ^ 2 + p.2 ^ 2 = 2) -- include points forming the octagon and those at √2 distance

def is_equilateral_triangle (a b c : ℕ × ℕ) : Prop :=
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (dist a b = dist b c ∧ dist b c = dist c a ∧ dist c a = dist a b)

-- Function to determine distance between two points
def dist (p1 p2: ℕ × ℕ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem octagonal_lattice_triangle_count :
  ∃ t : finset ((ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)),
    (∀ ⦃tr⦄, tr ∈ t → is_lattice_point (tr.fst.fst) ∧ 
                      is_lattice_point (tr.fst.snd) ∧ 
                      is_lattice_point tr.snd) ∧
    (∀ ⦃tr⦄, tr ∈ t → is_equilateral_triangle tr.fst.fst tr.fst.snd tr.snd) ∧
    t.card = 8 :=
sorry

end octagonal_lattice_triangle_count_l454_454900


namespace repeating_decimal_fraction_eq_l454_454161

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454161


namespace find_pairs_l454_454890

theorem find_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k m : ℕ, k ≠ 0 ∧ m ≠ 0 ∧ x + 1 = k * y ∧ y + 1 = m * x) ↔
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3) :=
by
  sorry

end find_pairs_l454_454890


namespace repeating_decimal_fraction_eq_l454_454167

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454167


namespace find_k_l454_454341

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else sorry

axiom seq_base : seq 1 = 2
axiom seq_recur (m n : ℕ) : seq (m + n) = seq m * seq n

-- Given condition for sum
axiom sum_condition (k : ℕ) : 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5

-- The statement to prove
theorem find_k : ∃ k : ℕ, 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5 ∧ k = 4 :=
by {
  sorry -- proof is not required
}

end find_k_l454_454341


namespace total_weekly_allowance_l454_454496

theorem total_weekly_allowance
  (total_students : ℕ)
  (students_6dollar : ℕ)
  (students_4dollar : ℕ)
  (students_7dollar : ℕ)
  (allowance_6dollar : ℕ)
  (allowance_4dollar : ℕ)
  (allowance_7dollar : ℕ)
  (days_in_week : ℕ) :
  total_students = 100 →
  students_6dollar = 60 →
  students_4dollar = 25 →
  students_7dollar = 15 →
  allowance_6dollar = 6 →
  allowance_4dollar = 4 →
  allowance_7dollar = 7 →
  days_in_week = 7 →
  (students_6dollar * allowance_6dollar + students_4dollar * allowance_4dollar + students_7dollar * allowance_7dollar) * days_in_week = 3955 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_weekly_allowance_l454_454496


namespace form_numbers_with_five_sevens_l454_454450

theorem form_numbers_with_five_sevens :
  ∀ n, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22} →
  ∃ (expression : Expr),
    eval_expr expression = n ∧ (occurrences_of_digit 7 expression = 5 ∧ valid_operations expression) :=
begin
  sorry
end

end form_numbers_with_five_sevens_l454_454450


namespace sum_remainder_l454_454801

theorem sum_remainder (n : ℕ) (h : n = 102) :
  ((n * (n + 1) / 2) % 5250) = 3 :=
by
  sorry

end sum_remainder_l454_454801


namespace repeating_decimal_equals_fraction_l454_454140

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454140


namespace product_of_t_factors_l454_454549

theorem product_of_t_factors (t : ℤ) :
  (∃ a b : ℤ, (x^2 + t * x - 24 = (x + a) * (x + b) ∧ a * b = -24 ∧ t = a + b)) →
  ∏ (t : set {a | ∃ b : ℤ, a * b = -24})  = 5290000 :=
by
  sorry

end product_of_t_factors_l454_454549


namespace number_of_integers_satisfying_inequality_l454_454282

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℝ) / |n| ≥ 1 / 5}.finite.card = 10 :=
by
  -- Proof goes here
  sorry

end number_of_integers_satisfying_inequality_l454_454282


namespace num_integers_satisfying_inequality_l454_454277

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l454_454277


namespace units_digit_of_m_l454_454000

def is_sequence (seq : List Nat) : Prop :=
  seq.head = some 21 ∧
  seq.last = some 0 ∧
  seq.length = 7 ∧
  ∀ i, i < seq.length.pred → seq.get? (i + 1) = some (seq.get! i - Nat.sqrt (seq.get! i) * Nat.sqrt (seq.get! i))

theorem units_digit_of_m :
  ∃ (m : ℕ), m % 10 = 1 ∧
  ∃ seq, is_sequence seq ∧ seq.head = some m :=
by {
  use 21,
  split,
  { refl },
  use [21, 12, 8, 4, 3, 2, 1, 0],
  split,
  {
    split,
    { refl },
    split,
    { refl },
    {
      split,
      { refl },
      {
        intros i hi,
        repeat { cases i; try {refl}, try {sorry} }
      }
    }
  },
  { refl }
}

end units_digit_of_m_l454_454000


namespace crayons_count_l454_454353

theorem crayons_count (eaten remaining start : ℕ) (h1 : eaten = 7) (h2 : remaining = 80) (h3 : remaining + eaten = start) :
  start = 87 := by
  rw [h1, h2] at h3
  exact h3

end crayons_count_l454_454353


namespace num_valid_pairs_l454_454198

theorem num_valid_pairs
  (condition1 : ∀ m n, -2014 ≤ m ∧ m ≤ 2014 ∧ -2014 ≤ n ∧ n ≤ 2014) :
  ∃ (S : set (ℤ × ℤ)), S.card = 25 ∧
    ∀ p ∈ S, let (m, n) := p in ∃ x y, x^3 + y^3 = m + 3 * n * x * y := 
sorry

end num_valid_pairs_l454_454198


namespace f_evaluation_at_2pi_over_3_smallest_positive_period_of_f_monotonically_decreasing_interval_of_f_l454_454258

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 2 - cos x ^ 2 - 2 * sqrt 3 * sin x * cos x

theorem f_evaluation_at_2pi_over_3 : f (2 * π / 3) = 2 :=
sorry

theorem smallest_positive_period_of_f : 0 < T ∧ ∀ x, f (x + T) = f x :=
sorry

theorem monotonically_decreasing_interval_of_f (k : ℤ) :
  ∀ x, x ∈ set.Icc (k * π - π / 3) (k * π + π / 6) ↔ f' x ≤ 0 :=
sorry

end f_evaluation_at_2pi_over_3_smallest_positive_period_of_f_monotonically_decreasing_interval_of_f_l454_454258


namespace tangent_through_points_l454_454703

theorem tangent_through_points :
  ∀ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ, y₁ = x₁^2 + 1 → y₂ = x₂^2 + 1 → 
    (2 * x₁ * (x₂ - x₁) + y₁ = 0 → x₂ = -x₁) ∧ 
    (2 * x₂ * (x₁ - x₂) + y₂ = 0 → x₁ = -x₂)) →
  (x₁ = 1 / Real.sqrt 3 ∧ x₂ = -1 / Real.sqrt 3 ∧
   (x₁^2 + 1 = (1 / 3) + 1) ∧ (x₂^2 + 1 = (1 / 3) + 1)) :=
by
  sorry

end tangent_through_points_l454_454703


namespace distinct_dress_designs_l454_454509

theorem distinct_dress_designs : 
  let num_colors := 5
  let num_patterns := 6
  num_colors * num_patterns = 30 :=
by
  sorry

end distinct_dress_designs_l454_454509


namespace fixed_point_of_tangents_l454_454942

-- Define the line x + 2y = 4
def on_line (P : ℝ × ℝ) : Prop :=
  P.1 + 2 * P.2 = 4

-- Define the ellipse x^2 + 4y^2 = 4
def on_ellipse (A : ℝ × ℝ) : Prop :=
  A.1^2 + 4 * A.2^2 = 4

-- Define the point P (moving point)
variable {P : ℝ × ℝ}

-- Define the fixed point
def fixed_point : ℝ × ℝ := (1, 1/2)

-- The main theorem statement
theorem fixed_point_of_tangents (P : ℝ × ℝ) (A B : ℝ × ℝ) 
  (hP : on_line P) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (tangent1 : P.1 * A.1 + 4 * P.2 * A.2 = 4)
  (tangent2 : P.1 * B.1 + 4 * P.2 * B.2 = 4)
  : (∃ k : ℝ, A = (k, (4 - k^2) / 8)) ∧ 
    (∃ k' : ℝ, B = (k', (4 - k'^2) / 8)) ∧
    ∃ (l : ℝ), ∀ x y : ℝ, (x, y) ∈ {A, B} → x * l + y * l = 1 :=
    sorry

end fixed_point_of_tangents_l454_454942


namespace maximize_oc_length_l454_454827

noncomputable def chord_max_oc : ℝ :=
  let r : ℝ := 1 in
  let a : ℝ := 1 / 2 in
  let ab := 2 * real.sqrt(1 - a) in
  ab

theorem maximize_oc_length : chord_max_oc = real.sqrt 2 :=
begin
  sorry
end

end maximize_oc_length_l454_454827


namespace repeating_decimal_fraction_eq_l454_454164

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l454_454164


namespace caterpillar_reaches_top_in_16_days_l454_454835

-- Define the constants for the problem
def pole_height : ℕ := 20
def daytime_climb : ℕ := 5
def nighttime_slide : ℕ := 4

-- Define the final result we want to prove
theorem caterpillar_reaches_top_in_16_days :
  ∃ days : ℕ, days = 16 ∧ 
  ((20 - 5) / (daytime_climb - nighttime_slide) + 1) = 16 := by
  sorry

end caterpillar_reaches_top_in_16_days_l454_454835


namespace base_subtraction_l454_454878

-- Define the base 8 number 765432_8 and its conversion to base 10
def base8Number : ℕ := 7 * (8^5) + 6 * (8^4) + 5 * (8^3) + 4 * (8^2) + 3 * (8^1) + 2 * (8^0)

-- Define the base 9 number 543210_9 and its conversion to base 10
def base9Number : ℕ := 5 * (9^5) + 4 * (9^4) + 3 * (9^3) + 2 * (9^2) + 1 * (9^1) + 0 * (9^0)

-- Lean 4 statement for the proof problem
theorem base_subtraction : (base8Number : ℤ) - (base9Number : ℤ) = -67053 := by
    sorry

end base_subtraction_l454_454878


namespace crayons_problem_l454_454507

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end crayons_problem_l454_454507


namespace first_shaded_square_ensuring_all_columns_l454_454519

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def shaded_squares_in_columns (k : ℕ) : Prop :=
  ∀ j : ℕ, j < 7 → ∃ n : ℕ, triangular_number n % 7 = j ∧ triangular_number n ≤ k

theorem first_shaded_square_ensuring_all_columns:
  shaded_squares_in_columns 55 :=
by
  sorry

end first_shaded_square_ensuring_all_columns_l454_454519


namespace length_of_each_fence_l454_454110

-- Definitions for conditions
def rate_per_meter : ℝ := 0.20
def num_fences : ℕ := 50
def total_earnings : ℝ := 5000

-- The theorem to prove
theorem length_of_each_fence :
  let total_length := total_earnings / rate_per_meter in
  let length_per_fence := total_length / (num_fences : ℝ) in
  length_per_fence = 500 :=
by
  sorry

end length_of_each_fence_l454_454110


namespace fraction_rep_finite_geom_series_036_l454_454126

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l454_454126


namespace num_integers_satisfying_inequality_l454_454276

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l454_454276


namespace gauss_family_mean_is_correct_l454_454411

-- Defining the list of ages
def ages : List ℝ := [7, 7, 8, 14, 12, 15, 16]

-- Defining the mean calculation
def mean (lst : List ℝ) : ℝ :=
  lst.sum / lst.length

-- Stating the theorem to prove that the mean is 11.2857
theorem gauss_family_mean_is_correct :
  mean ages = 11.2857 := 
by
  sorry

end gauss_family_mean_is_correct_l454_454411


namespace three_correct_deliveries_probability_l454_454205

theorem three_correct_deliveries_probability (n : ℕ) (h1 : n = 5) :
  (∃ p : ℚ, p = 1/6 ∧ 
   (∃ choose3 : ℕ, choose3 = Nat.choose n 3) ∧ 
   (choose3 * 1/5 * 1/4 * 1/3 = p)) :=
by 
  sorry

end three_correct_deliveries_probability_l454_454205


namespace cookies_in_jar_l454_454307

-- Let C be the total number of cookies in the jar.
def C : ℕ := sorry

-- Conditions
def adults_eat_one_third (C : ℕ) : ℕ := C / 3
def children_get_each (C : ℕ) : ℕ := 20
def num_children : ℕ := 4

-- Proof statement
theorem cookies_in_jar (C : ℕ) (h1 : C / 3 = adults_eat_one_third C)
  (h2 : children_get_each C * num_children = 80)
  (h3 : 2 * (C / 3) = 80) :
  C = 120 :=
sorry

end cookies_in_jar_l454_454307


namespace hyperbola_focus_value_of_a_l454_454625

theorem hyperbola_focus_value_of_a :
  ∃ a : ℝ, (∃ b : ℝ, (a^2 = 9 ∧ b^2 = a ∧ sqrt(13) ^ 2 = 13 ∧ 13 = a^2 + b^2)) → a = 4 :=
by
  sorry

end hyperbola_focus_value_of_a_l454_454625


namespace fixed_point_through_graph_l454_454766

def symmetric_property (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = a^x

def fixed_point_property (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f (1 - 1) = 1

theorem fixed_point_through_graph (a : ℝ) (f : ℝ → ℝ) (h : symmetric_property a f) : 
  fixed_point_property a (λ x, f (x - 1)) :=
  sorry

end fixed_point_through_graph_l454_454766


namespace sequence_k_eq_4_l454_454326

theorem sequence_k_eq_4 {a : ℕ → ℕ} (h1 : a 1 = 2) (h2 : ∀ m n, a (m + n) = a m * a n)
    (h3 : ∑ i in finset.range 10, a (4 + i) = 2^15 - 2^5) : 4 = 4 :=
by
  sorry

end sequence_k_eq_4_l454_454326


namespace circles_intersect_at_two_distinct_points_sector_area_proportion_l454_454093

-- Given definitions and conditions
variable (O1 O2 : Point)
variable (r1 r2 : ℝ)
variable (C1 C2 : Circle)
variable (H1 : C2.passes_through O1)
variable (H2 : r1 = r2 * sqrt(2 + sqrt(3)))

-- Proving the circles intersect at two distinct points
theorem circles_intersect_at_two_distinct_points
    (hC1 : C1.center = O1)
    (hC2 : C2.center = O2)
    (h_r1 : C1.radius = r1)
    (h_r2 : C2.radius = r2) :
    ∃ P Q, P ≠ Q ∧ C1.touches P ∧ C1.touches Q ∧ C2.touches P ∧ C2.touches Q :=
sorry

-- Proving the area proportion of the sector AO1B to circle C1
theorem sector_area_proportion
    (A B : Point)
    (H_intersect: C1.touches A ∧ C1.touches B ∧ C2.touches A ∧ C2.touches B)
    (HAO1B : angle A O1 B = 5 * π / 3) :
    sector_area C1 A O1 B / circle_area C1 = 5 / 6 :=
sorry

end circles_intersect_at_two_distinct_points_sector_area_proportion_l454_454093


namespace value_of_phi_l454_454922

theorem value_of_phi (φ : ℝ) 
  (h : ∀ x : ℝ, (sin (x + φ) + √3 * cos (x + φ)) = sin (-x + φ) + √3 * cos (-x + φ))
  : φ = π / 6 := 
sorry

end value_of_phi_l454_454922


namespace isosceles_triangle_of_condition_l454_454347

theorem isosceles_triangle_of_condition
  (A B C : ℝ)
  (h : 2 * sin A * cos B = sin C)
  (hABC : A + B + C = π) :
  A = B :=
by
  sorry

end isosceles_triangle_of_condition_l454_454347


namespace sequence_k_l454_454332

theorem sequence_k (a : ℕ → ℕ) (k : ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ m n, a (m + n) = a m * a n) 
    (h3 : (finset.range 10).sum (λ i, a (k + 1 + i)) = 2^15 - 2^5) : 
    k = 4 := 
by sorry

end sequence_k_l454_454332


namespace last_two_digits_of_sum_of_factorials_l454_454798

theorem last_two_digits_of_sum_of_factorials : 
  (∑ n in finset.range 10, (fact n) % 100) % 100 = 13 := 
by 
  -- Sum the relevant factorials reduced modulo 100 and take the modulo 100 of the sum
  sorry

end last_two_digits_of_sum_of_factorials_l454_454798


namespace incorrect_inequality_l454_454583

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ∃ c : ℝ, ac = bc :=
by
  have h1 : ¬(ac > bc) := by
    let c := 0
    show ac = bc 
    sorry

  exact ⟨0, h1⟩

end incorrect_inequality_l454_454583


namespace coefficient_x8_in_expansion_l454_454893

theorem coefficient_x8_in_expansion :
    ∀ (x : ℝ), 
    (x^3 + 1 / (2 * Real.sqrt x)) ^ 5 = 
    polynomial.monomial 8 (2 / 5) :=
by
    sorry

end coefficient_x8_in_expansion_l454_454893


namespace math_problem_l454_454365

-- Definitions of the real numbers a and b
variables {a b : ℝ}

-- The set condition given in the problem
def set_condition (a b : ℝ) : Prop := {0, b, b / a} = {1, a, a + b}

-- The goal is to prove a + 2b = 1 given the set condition
theorem math_problem (h : set_condition a b) : a + 2b = 1 :=
sorry

end math_problem_l454_454365


namespace exists_multiple_with_all_digits_l454_454725

theorem exists_multiple_with_all_digits (n : ℕ) :
  ∃ m : ℕ, (m % n = 0) ∧ (∀ d : ℕ, d < 10 → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9) := 
sorry

end exists_multiple_with_all_digits_l454_454725


namespace range_a_l454_454235

noncomputable def A (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}

noncomputable def B : Set ℝ := {x | x < -1 ∨ x > 16}

theorem range_a (a : ℝ) : (A a ∩ B = A a) → (a < 6 ∨ a > 7.5) :=
by
  intro h
  sorry

end range_a_l454_454235


namespace min_value_2x_plus_y_l454_454243

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/(y + 1) = 2) : 2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l454_454243


namespace geometric_series_sum_eq_l454_454804

-- Given conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 5

-- Define the geometric series sum formula
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The main theorem to prove
theorem geometric_series_sum_eq : geometric_series_sum a r n = 31 / 32 := by
  sorry

end geometric_series_sum_eq_l454_454804


namespace system_solutions_a_l454_454405

theorem system_solutions_a (x y z : ℝ) :
  (2 * x = (y + z) ^ 2) ∧ (2 * y = (z + x) ^ 2) ∧ (2 * z = (x + y) ^ 2) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_a_l454_454405


namespace brick_height_l454_454912

theorem brick_height (l w SA : ℝ) (h : ℝ) (h_l : l = 8) (h_w : w = 4) (h_SA : SA = 112) :
  2 * l * w + 2 * l * h + 2 * w * h = SA → h = 2 :=
by {
  intro h_surface_area,
  sorry
}

end brick_height_l454_454912


namespace slope_intercept_product_l454_454255

theorem slope_intercept_product (b m : ℤ) (h1 : b = -3) (h2 : m = 3) : m * b = -9 := by
  sorry

end slope_intercept_product_l454_454255


namespace find_m_when_power_function_decreasing_l454_454990

theorem find_m_when_power_function_decreasing :
  ∃ m : ℝ, (m^2 - 2 * m - 2 = 1) ∧ (-4 * m - 2 < 0) ∧ (m = 3) :=
by
  sorry

end find_m_when_power_function_decreasing_l454_454990


namespace distance_between_lines_l454_454967

def distance_parallel_lines (A B C1 C2 : ℝ) (h_parallel: A = 1 ∧ B = -1 ∧ C1 = 1 ∧ C2 = -3): ℝ :=
  real.abs (C1 - C2) / real.sqrt (A^2 + B^2)

theorem distance_between_lines :
  distance_parallel_lines 1 (-1) 1 (-3) (by simp) = 2 * real.sqrt 2 :=
sorry

end distance_between_lines_l454_454967


namespace minimum_red_squares_l454_454538

noncomputable def minRedSquares (m n : ℕ) : ℕ :=
  if m < 3 ∨ n < 3 then 0 else 2 * (m / 3) * (n / 3) + min (m / 3) (n / 3)

theorem minimum_red_squares (m n : ℕ) (hm : 3 ≤ m) (hn : 3 ≤ n) :
  ∃ r, r = 2 * (m / 3) * (n / 3) + min (m / 3) (n / 3) ∧
  ∀ grid : list (list ℕ), (∀ i j, i < m → j < n → grid.nth_le i (by blah) = 0) →
  (∃ grid', (∀ i j, (i % 3 = 2 ∧ j % 3 = 2) → grid'.nth_le i (by blah') = 1) →
  ∀ i j, i + 3 ≤ m ∧ j + 3 ≤ n → 
  (∃ x y, x < 3 ∧ y < 3 ∧ grid'.nth_le (i + x) (by blah) = 1 ∧ grid'.nth_le (i + y) (by blah) = 1)) :=
sorry

end minimum_red_squares_l454_454538


namespace total_number_of_choices_l454_454789

theorem total_number_of_choices : 
  let n := 5 in
  let choose_one_from_n := Nat.factorial n / Nat.factorial (n - 3) in
  let choose_two_from_n :=
    (Nat.combination n 2 * Nat.factorial 3 / Nat.factorial (3 - 2)) * 3 + 
    (Nat.combination n 2 * Nat.combination 3 2) * 3 in
  choose_one_from_n + choose_two_from_n = 330 :=
by
  sorry

end total_number_of_choices_l454_454789


namespace last_digit_fib2020_l454_454410

def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem last_digit_fib2020 : (fibonacci 2020) % 10 = 5 :=
by sorry

end last_digit_fib2020_l454_454410


namespace inequality_hold_l454_454259

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l454_454259


namespace problem_solution_set_l454_454367

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) is an odd function
noncomputable def g (x : ℝ) : ℝ := sorry -- g(x) is an even function

theorem problem_solution_set :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = g x) ∧
  (∀ x < 0, f' x * g x + f x * g' x > 0) ∧
  (g (-3) = 0) →
  { x : ℝ | f x * g x < 0 } = set.Ioo (-∞) (-3) ∪ set.Ioo 0 3 :=
by
  sorry

end problem_solution_set_l454_454367


namespace repeating_decimal_eq_fraction_l454_454177

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l454_454177


namespace num_integers_satisfying_inequality_l454_454544

theorem num_integers_satisfying_inequality : 
  {x : ℤ | -6 ≤ 3*x + 2 ∧ 3*x + 2 ≤ 9}.finite.card = 5 := 
sorry

end num_integers_satisfying_inequality_l454_454544


namespace possible_scores_count_l454_454068

def proof_problem (x y : ℕ) : Prop :=
  0 ≤ x ∧ x ≤ 100 ∧
  0 ≤ y ∧ y ≤ 100 ∧
  4 * x + y = 410 ∧
  y > x

theorem possible_scores_count : {y : ℕ // ∃ x, proof_problem x y}.card = 4 :=
by
  sorry

end possible_scores_count_l454_454068


namespace find_eccentricity_l454_454227

noncomputable def ellipse_eccentricity (a : ℝ) (b : ℝ) (c : ℝ) (x y : ℝ) : ℝ := 
  c / a

variable (a b c : ℝ)
variable (h₁ : 0 < a)
variable (h₂ : 0 < b)
variable (h₃ : 0 < c)
variable (ellipse_eq : a^2 - b^2 = 4)
variable (focus_eq : c = 2)

theorem find_eccentricity : 
  ellipse_eccentricity a b c (x : ℝ) (y : ℝ) = (sqrt 2) / 2 :=
by
  sorry

end find_eccentricity_l454_454227


namespace parabola_focus_l454_454945

-- Definitions based on the given conditions
def parabola_eq (y : ℝ) (p : ℝ) (x : ℝ) : Prop := y^2 = 2 * p * x

def distance_focus_directrix (p : ℝ) : ℝ :=
p

def focus_coordinates (p : ℝ) : ℝ × ℝ :=
(⟨p/2, 0⟩)

-- The theorem to be proved
theorem parabola_focus (p : ℝ) (h1 : p > 0) (h2 : distance_focus_directrix p = 4) :
  focus_coordinates p = (2, 0) :=
by
-- the proof steps would go here
  sorry

end parabola_focus_l454_454945


namespace range_of_x_l454_454602

variable (f : ℝ → ℝ)

/-- 
This is an even function and monotonically decreasing in the interval (-∞, 0].
-/
axiom even_and_monotone : (∀ x, f x = f (-x)) ∧ (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y)

/-- 
Given that f is an even and monotonically decreasing function in (-∞, 0], 
the range of x such that f(2 * x + 1) < f(3) is (-2, 1).
-/
theorem range_of_x : ∀ {x : ℝ}, f(2 * x + 1) < f(3) → (-2 < x ∧ x < 1) :=
by
  -- Proof goes here
  sorry

end range_of_x_l454_454602


namespace three_exp_div_twentysix_squared_l454_454799

theorem three_exp_div_twentysix_squared :
  27 = 3^3 → 3^{18} / (27^2) = 531441 :=
by
  assume h : 27 = 3^3
  sorry

end three_exp_div_twentysix_squared_l454_454799


namespace correct_propositions_count_l454_454242

variables (m l : Line) (α β : Plane)
def prop1 : Prop := (l ⊥ α ∧ m ∥ α) → l ⊥ m
def prop2 : Prop := (m ∥ l ∧ m ⊆ α) → l ∥ α
def prop3 : Prop := (α ⊥ β ∧ m ⊆ α ∧ l ⊆ β) → m ⊥ l
def prop4 : Prop := (m ⊥ l ∧ m ⊆ α ∧ l ⊆ β) → α ⊥ β

theorem correct_propositions_count : (prop1 m l α β ∧ ¬ prop2 m l α β ∧ ¬ prop3 m l α β ∧ ¬ prop4 m l α β) → ¬ (prop1 m l α β ∧ prop2 m l α β ∧ prop3 m l α β ∧ prop4 m l α β) := 
by 
  sorry

end correct_propositions_count_l454_454242


namespace repeating_decimal_equals_fraction_l454_454136

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454136


namespace arithmetic_sequence_identity_l454_454935

theorem arithmetic_sequence_identity (a : ℕ → ℝ) (d : ℝ)
    (h_arith : ∀ n, a (n + 1) = a 1 + n * d)
    (h_sum : a 4 + a 7 + a 10 = 30) :
    a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 :=
sorry

end arithmetic_sequence_identity_l454_454935


namespace ratio_cards_eaten_l454_454681

-- Definitions according to the conditions
def initial_cards : ℕ := 84
def new_cards : ℕ := 8
def cards_left_after_dog : ℕ := 46

-- Proof statement
theorem ratio_cards_eaten (initial_cards new_cards cards_left_after_dog : ℕ) :
  let total_cards_before := initial_cards + new_cards in
  let cards_eaten := total_cards_before - cards_left_after_dog in
  (cards_eaten : ℚ) / (total_cards_before : ℚ) = 1 / 2 :=
by
  sorry

end ratio_cards_eaten_l454_454681


namespace number_of_integers_satisfying_inequality_l454_454283

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℝ) / |n| ≥ 1 / 5}.finite.card = 10 :=
by
  -- Proof goes here
  sorry

end number_of_integers_satisfying_inequality_l454_454283


namespace inscribed_circle_radius_l454_454873

-- Define the radii of the circles
variables (r1 r2 r3 : ℝ)
-- Define the conditions
axiom no_overlap (c1 c2 c3 : ℝ) : c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3
axiom radii_condition (r1 r2 r3 : ℝ) : r1 > r2 ∧ r1 > r3

-- Define the points A and B where external tangents intersect
variables (A B : ℝ)

-- Main theorem statement
theorem inscribed_circle_radius (c1 c2 c3 : ℝ) (A B : ℝ) :
    no_overlap c1 c2 c3 ∧ radii_condition r1 r2 r3 ∧ A ≠ B
    → ∃ r : ℝ, r = (r1 * r2 * r3) / (r1 * r2 - r1 * r3 - r2 * r3) :=
begin
    sorry
end

end inscribed_circle_radius_l454_454873


namespace domain_function_l454_454762

noncomputable def domain_of_function : set ℝ := {x | x < 0 ∧ x ≠ -3 / 2}

theorem domain_function (x : ℝ) : 
  (y = (2 * x + 3) ^ 0 / (Real.sqrt (abs x - x))) → x ∈ domain_of_function :=
begin
  sorry  -- Proof omitted
end

end domain_function_l454_454762


namespace repeating_decimal_fraction_l454_454153

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l454_454153


namespace simplify_fraction_l454_454745

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l454_454745


namespace calculate_investment_ratio_approx_l454_454776

-- Definitions based on the given conditions
def investment_ratio_pq := (7 : ℝ) / x
def profit_ratio_pq := (7.00001 : ℝ) / 10
def time_p := 5
def time_q := 9.999965714374696

-- Ratio of investment calculation based on the given information
def investment_calculation (x : ℝ) : ℝ := 
  (7 * 5) / (x * 9.999965714374696)

-- The lean statement to prove that the ratio of Q's investment to P's investment is approximately 5:7.
theorem calculate_investment_ratio_approx (x : ℝ) (h : investment_calculation (x : ℝ) = profit_ratio_pq) : 
  (350 / (7.00001 * 9.999965714374696)) ≈ 5  := 
sorry

end calculate_investment_ratio_approx_l454_454776


namespace ellipse_parameters_l454_454526

theorem ellipse_parameters :
  ∃ (a b : ℝ) (h k : ℝ),
    ellipse_in_standard_form (0, 0) (0, 4) (10, 3) a b h k ∧
    a = (Real.sqrt 109 + Real.sqrt 101) / 2 ∧
    b = Real.sqrt ((Real.sqrt 109 + Real.sqrt 101) / 2)^2 - 4 ∧
    h = 0 ∧
    k = 2 :=
by
  -- Definitions and conditions
  sorry

end ellipse_parameters_l454_454526


namespace original_rectangle_area_l454_454515

-- Define the original rectangle sides, square side, and perimeters of rectangles adjacent to the square
variables {a b x : ℝ}
variable (h1 : a + x = 10)
variable (h2 : b + x = 8)

-- Define the area calculation
def area (a b : ℝ) := a * b

-- The area of the original rectangle should be 80 cm²
theorem original_rectangle_area : area (10 - x) (8 - x) = 80 := by
  sorry

end original_rectangle_area_l454_454515


namespace solution_to_system_l454_454115

noncomputable def solve_system (n : ℕ) (a : ℝ) (x : Fin n → ℝ) : Prop :=
  (∀ i : Fin n−1, x i * |x i| - (x i - a) * |x i - a| = x (i + 1) * |x (i + 1)|) ∧
  x (n - 1) * |x (n - 1)| - (x (n - 1) - a) * |x (n - 1) - a| = x 0 * |x 0|

theorem solution_to_system (n : ℕ) (a : ℝ) (x : Fin n → ℝ) :
  solve_system n a x → (∀ i, x i = a) :=
by
  sorry

end solution_to_system_l454_454115


namespace geometric_series_sum_l454_454805

theorem geometric_series_sum :
  let a_0 := 1 / 2
  let r := 1 / 2
  let n := 5
  let sum := ∑ i in Finset.range(n), a_0 * r^i
  sum = 31 / 32 :=
by
  sorry

end geometric_series_sum_l454_454805


namespace coefficient_of_x_cubed_in_expansion_l454_454414

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  n.choose k

-- Definition of the nth term in the binomial expansion of (x-2)^5
def binomial_expansion_term (r : ℕ) : ℕ :=
  binomial 5 r * (-2)^r

-- Prove that the coefficient of the x^3 term is 40
theorem coefficient_of_x_cubed_in_expansion : binomial_expansion_term 2 = 40 := by
  sorry

end coefficient_of_x_cubed_in_expansion_l454_454414


namespace lotus_leaves_not_odd_l454_454035

theorem lotus_leaves_not_odd (n : ℕ) (h1 : n > 1) (h2 : ∀ t : ℕ, ∃ r : ℕ, 0 ≤ r ∧ r < n ∧ (t * (t + 1) / 2 - 1) % n = r) : ¬ Odd n :=
sorry

end lotus_leaves_not_odd_l454_454035


namespace percent_females_employed_l454_454818

noncomputable def employed_percent (population: ℕ) : ℚ := 0.60
noncomputable def employed_males_percent (population: ℕ) : ℚ := 0.48

theorem percent_females_employed (population: ℕ) : ((employed_percent population) - (employed_males_percent population)) / (employed_percent population) = 0.20 :=
by
  sorry

end percent_females_employed_l454_454818


namespace avg_visitors_per_day_l454_454898

theorem avg_visitors_per_day :
  let visitors := [583, 246, 735, 492, 639]
  (visitors.sum / visitors.length) = 539 := by
  sorry

end avg_visitors_per_day_l454_454898


namespace max_value_ineq_l454_454594

-- Define the conditions
variables {x y z : ℝ}
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)
variables (h_condition : (x^2 / (1 + x^2)) + (y^2 / (1 + y^2)) + (z^2 / (1 + z^2)) = 2)

-- Define the theorem statement for the proof problem
theorem max_value_ineq :
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ sqrt 2 :=
sorry

end max_value_ineq_l454_454594


namespace repeating_decimal_equals_fraction_l454_454139

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l454_454139


namespace polynomial_simp_l454_454848

theorem polynomial_simp (d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16 : ℕ)
  (h : (1 - z) ^ d1 * (1 - z ^ 2) ^ d2 * (1 - z ^ 3) ^ d3 * (1 - z ^ 4) ^ d4 * 
       (1 - z ^ 5) ^ d5 * (1 - z ^ 6) ^ d6 * (1 - z ^ 7) ^ d7 * (1 - z ^ 8) ^ d8 * 
       (1 - z ^ 9) ^ d9 * (1 - z ^ 10) ^ d10 * (1 - z ^ 11) ^ d11 * (1 - z ^ 12) ^ d12 * 
       (1 - z ^ 13) ^ d13 * (1 - z ^ 14) ^ d14 * (1 - z ^ 15) ^ d15 * (1 - z ^ 16) ^ d16 
       ≡ (1 : ℤ[z]) - 4 * z [z^17]) : d16 = 128 := 
  sorry

end polynomial_simp_l454_454848


namespace paperclip_production_l454_454820

def machines_paperclips_production (machines_producing_rate paperclips_per_minute: ℕ) : ℕ := paperclips_per_minute / machines_producing_rate

def multiple_machines_production (total_machines rate_per_machine minutes : ℕ) : ℕ := total_machines * rate_per_machine * minutes

theorem paperclip_production (machines_producing_rate total_paperclips_per_minute total_machines minutes : ℕ) :
    machines_producing_rate = 8 ∧ total_paperclips_per_minute = 560 ∧ total_machines = 15 ∧ minutes = 6 →
    multiple_machines_production total_machines (machines_paperclips_production machines_producing_rate total_paperclips_per_minute) minutes = 6300 :=
begin
    sorry
end

end paperclip_production_l454_454820


namespace find_ratio_l454_454718

theorem find_ratio (r AB BC : ℝ)
  (h1 : AB = AC)
  (h2 : AB > r)
  (h3 : BC = π * r / 3)
  (h4 : ∀ angles : Type, radian angles = (π : ℝ) / 3) :
  AB / BC = 3 / π := by
  sorry

end find_ratio_l454_454718


namespace kostya_table_prime_l454_454358

theorem kostya_table_prime {n : ℕ} (hn : n > 3)
  (h : ∀ r s : ℕ, r ≥ 3 → s ≥ 3 → rs - (r + s) ≠ n) : Prime (n + 1) := 
sorry

end kostya_table_prime_l454_454358


namespace taxi_ride_cost_l454_454858

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end taxi_ride_cost_l454_454858


namespace inequality_proof_l454_454266

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l454_454266


namespace sequence_never_arithmetic_progression_l454_454887

def b (n : ℕ) : ℕ
| 0 := 1
| 1 := 2
| (n+2) := b n * b (n+1) + 1

theorem sequence_never_arithmetic_progression :
  ¬ (∃ d : ℕ, ∀ n : ℕ, b (n+1) - b n = d) :=
sorry

end sequence_never_arithmetic_progression_l454_454887


namespace calculate_total_customers_l454_454876

theorem calculate_total_customers 
    (num_no_tip : ℕ) 
    (total_tip_amount : ℕ) 
    (tip_per_customer : ℕ) 
    (number_tipped_customers : ℕ) 
    (number_total_customers : ℕ)
    (h1 : num_no_tip = 5) 
    (h2 : total_tip_amount = 15) 
    (h3 : tip_per_customer = 3) 
    (h4 : number_tipped_customers = total_tip_amount / tip_per_customer) :
    number_total_customers = number_tipped_customers + num_no_tip := 
by {
    sorry
}

end calculate_total_customers_l454_454876


namespace cars_distance_at_least_diameter_half_time_l454_454032

theorem cars_distance_at_least_diameter_half_time :
  ∀ (t : ℝ), (0 ≤ t ∧ t ≤ 1) →
  let radius := 1
  let center_α := (0, -Real.sqrt 3)
  let center_β := (1, 0)
  let pos_A := (Real.sin (2 * Real.pi * t), -Real.sqrt 3 + Real.cos (2 * Real.pi * t))
  let pos_B := (1 - Real.cos (2 * Real.pi * t), -Real.sin (2 * Real.pi * t))
  let dist_sq := (pos_A.1 - pos_B.1)^2 + (pos_A.2 - pos_B.2)^2
  let diameter_sq := (2 * radius)^2
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (dist_sq ≥ diameter_sq) ↔ (1 / 2) :=
sorry

end cars_distance_at_least_diameter_half_time_l454_454032


namespace simplify_fraction_l454_454734

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454734


namespace vector_sum_inequality_l454_454394

theorem vector_sum_inequality (n : ℕ) (a b : Fin n → ℝ) :
  (Real.sqrt ((Finset.univ.sum (λ j, a j)) ^ 2 + (Finset.univ.sum (λ j, b j)) ^ 2)) ≤
  Finset.univ.sum (λ j, Real.sqrt ((a j) ^ 2 + (b j) ^ 2)) :=
by
  sorry

end vector_sum_inequality_l454_454394


namespace no_perfect_square_in_A_odd_product_not_perfect_square_l454_454723

def A (n : ℕ) : Prop := ∃ x y : ℤ, n = 2 * x^2 + 3 * y^2 ∧ x^2 + y^2 ≠ 0

theorem no_perfect_square_in_A : ∀ n : ℕ, n ∈ A → ∀ m : ℕ, m * m ≠ n := by
  intro n hA m hSquare
  -- proof goes here
  sorry

theorem odd_product_not_perfect_square :
  ∀ ns : list ℕ, (∀ n : ℕ, n ∈ ns → A n) → ns.length % 2 = 1 → ∀ m : ℕ, m * m ≠ ns.prod := by
  intro ns hA hOddLength m hSquare
  -- proof goes here
  sorry

end no_perfect_square_in_A_odd_product_not_perfect_square_l454_454723


namespace floor_root_sequence_product_div_l454_454884

def floor_root_sequence_product : ℚ :=
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 1 * 
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 2 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 3 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 5 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 6 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 7 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 8 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 9 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 10 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 11 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 12 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 13 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 14 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 15 / 
((↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 2 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 4 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 6 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 8 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 10 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 12 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 14 *
  (↿(λ n : ℕ, ((int.floor (real.sqrt n : ℝ)).to_nat))) 16)

theorem floor_root_sequence_product_div :
  floor_root_sequence_product = 3 / 8 :=
by
  -- skipping proof
  sorry

end floor_root_sequence_product_div_l454_454884


namespace find_point_P_l454_454233

theorem find_point_P (M N : ℝ × ℝ) (hM : M = (2, 5)) (hN : N = (3, -2))
  (P : ℝ × ℝ) (hP : ∃ λ : ℝ, λ = 3 ∧ P = ((2 + 3 * 3) / (1 + 3), (5 + (-2) * 3) / (1 + 3))) :
  P = (11 / 4, -1 / 4) :=
by
  sorry

end find_point_P_l454_454233


namespace pool_fill_time_l454_454849

theorem pool_fill_time:
  ∀ (A B C D : ℚ),
  (A + B - D = 1 / 6) →
  (A + C - D = 1 / 5) →
  (B + C - D = 1 / 4) →
  (A + B + C - D = 1 / 3) →
  (1 / (A + B + C) = 60 / 23) :=
by intros A B C D h1 h2 h3 h4; sorry

end pool_fill_time_l454_454849


namespace exterior_angle_BAC_l454_454067

/-- Given a coplanar square and regular octagon sharing a common side AD,
    prove that the measure of the exterior angle BAC is 135 degrees. -/
theorem exterior_angle_BAC (square_angle : ℕ) (octagon_angle : ℕ) (shared_side : ℕ)
  (interior_square_angle : square_angle = 90) (interior_octagon_angle : octagon_angle = 135)
  (exterior_angle : shared_side = 360 - square_angle - octagon_angle) :
  shared_side = 135 :=
begin
  rw [interior_square_angle, interior_octagon_angle],
  exact exterior_angle,
  sorry
end

end exterior_angle_BAC_l454_454067


namespace coronavirus_diameter_in_centimeters_l454_454417

noncomputable def diameter_nanometers : ℕ := 120
noncomputable def nanometer_to_meter : ℝ := 10 ^ (-9)
noncomputable def meter_to_centimeter : ℝ := 100

theorem coronavirus_diameter_in_centimeters :
  (diameter_nanometers : ℝ) * nanometer_to_meter * meter_to_centimeter = 1.2 * 10 ^ (-5) :=
by
  sorry

end coronavirus_diameter_in_centimeters_l454_454417


namespace page_number_counted_twice_l454_454771

theorem page_number_counted_twice {n x : ℕ} (h₁ : n = 70) (h₂ : x > 0) (h₃ : x ≤ n) (h₄ : 2550 = n * (n + 1) / 2 + x) : x = 65 :=
by {
  sorry
}

end page_number_counted_twice_l454_454771


namespace annular_region_area_l454_454880

theorem annular_region_area (D_outer D_inner : ℝ) (h_outer : D_outer = 10) (h_inner : D_inner = 6) :
  ∃ (A : ℝ), A = 16 * Real.pi :=
  by
  let R := D_outer / 2
  let r := D_inner / 2
  have h_R : R = 5 := by rw [h_outer]; exact (by norm_num : 10 / 2 = 5)
  have h_r : r = 3 := by rw [h_inner]; exact (by norm_num : 6 / 2 = 3)
  let A := Real.pi * (R^2 - r^2)
  use A
  rw [h_R, h_r]
  have h_annulus_area : A = 16 * Real.pi := by
    rw [sq, sq]
    norm_num
  exact h_annulus_area
  sorry

end annular_region_area_l454_454880


namespace f_continuous_on_interval_f_not_bounded_variation_l454_454697

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x * Real.sin (1 / x)

theorem f_continuous_on_interval : ContinuousOn f (Set.Icc 0 1) :=
sorry

theorem f_not_bounded_variation : ¬ BoundedVariationOn f (Set.Icc 0 1) :=
sorry

end f_continuous_on_interval_f_not_bounded_variation_l454_454697


namespace find_h_of_root_l454_454976

theorem find_h_of_root :
  ∀ h : ℝ, (-3)^3 + h * (-3) - 10 = 0 → h = -37/3 := by
  sorry

end find_h_of_root_l454_454976


namespace paperback_copies_sold_l454_454438

theorem paperback_copies_sold 
(H : ℕ)
(hardback_sold : H = 36000)
(P : ℕ)
(paperback_relation : P = 9 * H)
(total_copies : H + P = 440000) :
P = 324000 :=
sorry

end paperback_copies_sold_l454_454438


namespace find_n_l454_454577

theorem find_n (n : ℕ) (a : ℕ → ℕ) (h : (1 + 1)^n = a 0 + a 1 + a 2 + ... + a n) : n = 5 :=
  sorry

end find_n_l454_454577


namespace integer_solution_count_l454_454287

theorem integer_solution_count :
  {n : ℤ | n ≠ 0 ∧ (1 / |(n:ℚ)| ≥ 1 / 5)}.finite.card = 10 :=
by
  sorry

end integer_solution_count_l454_454287


namespace general_term_T_2013_value_l454_454600

variable {a : ℕ → ℕ}
variable {S : ℕ → ℚ}

-- Definition of arithmetic sequence conditions
def arithmetic_seq (a : ℕ → ℕ) (a1 a3 : ℕ) (d : ℕ) := 
  (a 1 = a1) ∧ (a 3 = a1 + 2 * d)

-- Conditions
def sequence_conditions (a1 a3 : ℕ) : Prop :=
  a1 = 2 ∧ a3 = 6

-- General term formula statement
theorem general_term (a1 a3 : ℕ) (d : ℕ) (h : sequence_conditions a1 a3) : 
  ∃ a : ℕ → ℕ, arithmetic_seq a a1 a3 d ∧ ∀ n, a n = 2 * n :=
sorry

-- Sum of the sequence conditions
def S_n := λ (n : ℕ), n * (n + 1)

-- Define the sum of the first n terms of the sequence {1/S_n}
def sum_sequence (T : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  ∀ n, T n = ∑ i in finset.range (n + 1), 1 / S_n i

-- Statement of the problem 2
theorem T_2013_value : 
  let T : ℕ → ℚ := λ (n : ℕ), ∑ i in finset.range (n + 1), 1 / S_n (i + 1)
  in T 2013 = 2013 / 2014 :=
sorry

end general_term_T_2013_value_l454_454600


namespace flea_can_visit_every_nat_point_l454_454078

theorem flea_can_visit_every_nat_point (flea_pos : ℕ → ℤ) :
  (∀ n : ℕ, ∃ (k : ℕ → ℤ), (flea_pos 0 = 0) ∧ (∀ i, (flea_pos (i + 1) = flea_pos i + (2^i + 1) * k i - (2^i) * (1 - k i))) ∧
  ∀ (N : ℕ), ∃ (n : ℕ), flea_pos n = N) → (∀ m : ℕ, ∃ n : ℕ, flea_pos n = m) :=
begin
  sorry
end

end flea_can_visit_every_nat_point_l454_454078


namespace length_of_curve_l454_454769

noncomputable def length_of_parametric_curve : ℝ :=
let f₁ θ := 2 * (Real.cos θ)^2 in
let f₂ θ := 3 * (Real.sin θ)^2 in
let dx_dθ θ := Real.derivative (λ θ, f₁ θ) θ in
let dy_dθ θ := Real.derivative (λ θ, f₂ θ) θ in
∫ measurable_integral Ioc 0 π (λ θ, Real.sqrt ((dx_dθ θ)^2 + (dy_dθ θ)^2))

theorem length_of_curve {θ : ℝ} :
  (∀ θ ∈ set.Ioc (0 : ℝ) (π : ℝ), 
    x = 2 * (Real.cos θ) ^ 2 
    ∧ y = 3 * (Real.sin θ) ^ 2) →
  length_of_parametric_curve = Real.sqrt 13 :=
begin
  sorry
end

end length_of_curve_l454_454769


namespace upper_limit_for_a_l454_454986

theorem upper_limit_for_a :
  ∃ A : ℤ, (∀ a : ℤ, 6 < a ∧ a < A → (∀ b : ℤ, 3 < b ∧ b < 29 → (A - 1) / 4 - 7 / 28 = 3.75)) ∧ A = 17 :=
by
  sorry

end upper_limit_for_a_l454_454986


namespace relationship_Sn_Tn_l454_454919

theorem relationship_Sn_Tn (n : ℕ) (h_pos : n ≠ 0) : 
  let S_n := 2^n 
  let T_n := 2^n - (-1)^n
  in (n % 2 = 1 -> S_n < T_n) ∧ (n % 2 = 0 -> S_n > T_n) :=
by
  sorry

end relationship_Sn_Tn_l454_454919


namespace trisectors_of_rectangle_form_rhombus_l454_454552

-- Problem Statement: Each angle of a rectangle is trisected and the intersections 
-- of the pairs of trisectors adjacent to the same side of the rectangle form a shape. 
-- Prove that this shape is a rhombus.

theorem trisectors_of_rectangle_form_rhombus
  (A B C D P Q R S : Point)
  (h_rectangle: rectangle A B C D)
  (h_trisected: ∀ (angle: Angle), angle ∈ {∠ A B, ∠ B C, ∠ C D, ∠ D A} → angle.trisected)
  (trisect_intersections: 
    intersections_of_trisectors [ ∠ A B , ∠ B C , ∠ C D , ∠ D A ]
                                [(A, B), (B, C), (C, D), (D, A)] = [P, Q, R, S])
  : is_rhombus P Q R S :=
sorry

end trisectors_of_rectangle_form_rhombus_l454_454552


namespace sequence_difference_l454_454268

theorem sequence_difference (a : ℕ → ℤ) (h_rec : ∀ n : ℕ, a (n + 1) + a n = n) (h_a1 : a 1 = 2) :
  a 4 - a 2 = 1 :=
sorry

end sequence_difference_l454_454268


namespace proof_problem_l454_454995

noncomputable def angle_and_range (a b c : ℝ) (B C : ℝ) : Prop :=
  let A := real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  a * real.cos B - (1/2) * b = (a^2 / c) - (b * real.sin B / real.sin C) ∧
  (a = real.sqrt 3) → (real.sqrt 3 < b + c ∧ b + c ≤ 2 * real.sqrt 3)

theorem proof_problem (a b c B C : ℝ) :
  angle_and_range a b c B C ↔
  let A := real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  A = real.pi / 3 ∧
  (a = real.sqrt 3) → (real.sqrt 3 < b + c ∧ b + c ≤ 2 * real.sqrt 3) :=
by
  sorry

end proof_problem_l454_454995


namespace cube_color_faces_l454_454042

theorem cube_color_faces (n : ℕ) (r b : Set (ℕ × ℕ × ℕ)) 
  (h1 : n = 8)
  (h2 : ∀ i j k, 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n ∧ 0 ≤ k ∧ k < n →
         ((i = 0 ∨ i = n-1) ∨ (j = 0 ∨ j = n-1) ∨ (k = 0 ∨ k = n-1)) →
         ((if (i = 0 ∨ i = n-1) then (0,0,i) ∈ r else true) ∧ 
          (if (j = 0 ∨ j = n-1) then (0,j,0) ∈ r else true) ∧
          (if (k = 0 ∨ k = n-1) then (k,0,0) ∈ r else true) ∧
          (if (i = 0 ∨ i = n-1) then (0,0,i) ∈ b else true) ∧
          (if (j = 0 ∨ j = n-1) then (0,j,0) ∈ b else true) ∧
          (if (k = 0 ∨ k = n-1) then (k,0,0) ∈ b else true))) :
  (∃ c : ℕ, c = 56 ∧ (∀ x, x ∈ r ∩ b ↔ 1 ≤ x.1 ∧ x.1 < n∧ 1 ≤ x.2 ∧ x.2 < n ∧ 1 ≤ x.3 ∧ x.3 < n)) :=
sorry

end cube_color_faces_l454_454042


namespace simplify_fraction_l454_454742

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l454_454742


namespace abs_subtract_abs_l454_454823

-- Define the absolute value function
def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

-- Statement of the problem
theorem abs_subtract_abs : abs (16 - 5) - abs (5 - 12) = 4 :=
by
  sorry -- Proof goes here

end abs_subtract_abs_l454_454823


namespace sum_of_three_consecutive_integers_is_21_l454_454476

theorem sum_of_three_consecutive_integers_is_21 (n : ℤ) :
    n ∈ {17, 11, 25, 21, 8} →
    (∃ a, n = a + (a + 1) + (a + 2)) →
    n = 21 :=
by
  intro h
  intro h_consec
  cases h_consec with a ha
  have sum_eq_three_a : n = 3 * a + 3 :=
    by linarith
  -- Verify that 21 is the only possible sum value.
  have h_n_values : n = 17 ∨ n = 11 ∨ n = 25 ∨ n = 21 ∨ n = 8 :=
    by simp at h; exact h
  cases h_n_values
  { rw h_n_values at sum_eq_three_a; contradiction }
  { rw h_n_values at sum_eq_three_a; contradiction }
  { rw h_n_values at sum_eq_three_a; contradiction }
  { exact h_n_values }
  { rw h_n_values at sum_eq_three_a; contradiction }
sorry

end sum_of_three_consecutive_integers_is_21_l454_454476


namespace nonagon_diagonals_not_parallel_l454_454972

theorem nonagon_diagonals_not_parallel (n : ℕ) (h : n = 9) : 
  ∃ k : ℕ, k = 18 ∧ 
    ∀ v₁ v₂, v₁ ≠ v₂ → (n : ℕ).choose 2 = 27 → 
    (v₂ - v₁) % n ≠ 4 ∧ (v₂ - v₁) % n ≠ n-4 :=
by
  sorry

end nonagon_diagonals_not_parallel_l454_454972


namespace playerA_winning_strategy_l454_454489

theorem playerA_winning_strategy (k : ℕ) : k > 10 → ∃ strategy_for_A, ∀ strategy_for_B, wins(strategy_for_A, strategy_for_B) :=
sorry

end playerA_winning_strategy_l454_454489


namespace volume_of_solid_l454_454901

-- Definitions for the conditions
def vertices_on_same_side (A B C D : Point) (S : Plane) : Prop :=
  (distance_to_plane A S > 0) ∧ (distance_to_plane B S > 0) ∧ (distance_to_plane C S > 0) ∧ (distance_to_plane D S > 0)

def distance_from_plane (A : Point) (S : Plane) (d : ℝ) : Prop :=
  distance_to_plane A S = d

def projection_area (A B C D : Point) (S : Plane) (area : ℝ) : Prop :=
  quadrilateral_area (project_to_plane A S) (project_to_plane B S) (project_to_plane C S) (project_to_plane D S) = area

-- Problem statement in Lean
theorem volume_of_solid (A B C D : Point) (S : Plane) :
  vertices_on_same_side A B C D S →
  distance_from_plane A S 4 →
  distance_from_plane B S 6 →
  distance_from_plane C S 8 →
  projection_area A B C D S 10 →
  volume_of_parallelepiped A B C D (project_to_plane A S) (project_to_plane B S) (project_to_plane C S) (project_to_plane D S) = 60 :=
by sorry

end volume_of_solid_l454_454901


namespace repeating_decimal_to_fraction_l454_454119

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l454_454119


namespace circle_area_circumscribed_about_square_l454_454503

theorem circle_area_circumscribed_about_square (s : ℝ) (h : s = 12) :
  let r := s * (Real.sqrt 2) / 2 in
  let area := (Real.pi * r^2) in
  area = 72 * Real.pi := by
  sorry

end circle_area_circumscribed_about_square_l454_454503
