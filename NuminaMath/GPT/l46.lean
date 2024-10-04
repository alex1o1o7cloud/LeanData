import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Divisors
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Compositions
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Graph.Tree
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.GCD
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import tactic

namespace length_of_median_on_hypotenuse_l46_46917

theorem length_of_median_on_hypotenuse :
  ∀ (a b : ℝ), a = 6 → b = 8 → (c = real.sqrt (a^2 + b^2)) → (m = c / 2) → m = 5 := by
  intros a b h1 h2 h3 h4
  sorry

-- Definitions for conditions
def length_of_hypotenuse (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

def length_of_median (c : ℝ) : ℝ := c / 2

-- Applying the definitions to the known values
example : 
  ∀ (a b c m : ℝ), a = 6 → b = 8 → c = length_of_hypotenuse a b → 
                  m = length_of_median c → m = 5 := by
  intros a b c m ha hb hc hm
  rw [ha, hb] at hc
  rw [hc, hm]
  sorry

end length_of_median_on_hypotenuse_l46_46917


namespace train_length_is_correct_l46_46817

noncomputable def length_of_train (speed_train_kmh : ℝ) (speed_man_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := relative_speed_kmh * (5/18)
  let length := relative_speed_ms * time_s
  length

theorem train_length_is_correct (h1 : 84 = 84) (h2 : 6 = 6) (h3 : 4.399648028157747 = 4.399648028157747) :
  length_of_train 84 6 4.399648028157747 = 110.991201 := by
  dsimp [length_of_train]
  norm_num
  sorry

end train_length_is_correct_l46_46817


namespace hyperbola_asymptotes_angle_45_l46_46482

/-- For a hyperbola defined by the equation x^2/a^2 - y^2/b^2 = 1 with a > b,
     and the angle between the asymptotes is 45 degrees, prove that a/b = sqrt(2) + 1. -/
theorem hyperbola_asymptotes_angle_45 (a b : ℝ) (ha : a > b)
  (h : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x = y * (2 - √2) ∨ x = -y * (2 - √2))) :
  a / b = √2 + 1 :=
sorry

end hyperbola_asymptotes_angle_45_l46_46482


namespace card_combination_sum_divisible_by_100_l46_46496

/--
Given 2414 cards, each with a unique natural number from 1 to 2414,
prove that there are 29112 ways to choose two cards such that the sum
of their numbers is divisible by 100.
-/
theorem card_combination_sum_divisible_by_100 :
  ∃ (n : ℕ), n = 29112 ∧
  ∀ (cards : Finset ℕ), (cards.card = 2414) → 
  (∀ x ∈ cards, x ≥ 1 ∧ x ≤ 2414) → 
  (∃ (count : ℕ), 
    count = 
    (cards.powerset.filter (λ s, s.card = 2 ∧ ∃ a b, a ∈ s ∧ b ∈ s ∧ a + b ≡ 0 [MOD 100])).card 
    ∧ count = n) :=
by
  sorry

end card_combination_sum_divisible_by_100_l46_46496


namespace translation_coordinates_l46_46661

-- Define starting point
def initial_point : ℤ × ℤ := (-2, 3)

-- Define the point moved up by 2 units
def move_up (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ :=
  (p.fst, p.snd + d)

-- Define the point moved right by 2 units
def move_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ :=
  (p.fst + d, p.snd)

-- Expected results after movements
def point_up : ℤ × ℤ := (-2, 5)
def point_right : ℤ × ℤ := (0, 3)

-- Proof statement
theorem translation_coordinates :
  move_up initial_point 2 = point_up ∧
  move_right initial_point 2 = point_right :=
by
  sorry

end translation_coordinates_l46_46661


namespace value_of_a_l46_46973

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2 * x

theorem value_of_a (a : ℝ) (h : f a = 10) : a = -3 ∨ a = 5 :=
by
  sorry

end value_of_a_l46_46973


namespace find_log3_iterative_limit_approx_l46_46011

noncomputable def log3_iterative_limit (x : ℝ) : Prop :=
  x = Real.logb 3 (50 + x)

theorem find_log3_iterative_limit_approx :
  ∃ x : ℝ, x > 0 ∧ log3_iterative_limit x ∧ x ≈ 3.9 :=
by
  sorry

end find_log3_iterative_limit_approx_l46_46011


namespace injective_func_form_l46_46872

-- Define the conditions
def polynomial (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.enum.sum (λ (i, a_i), a_i * x^i)

theorem injective_func_form 
  (f : ℕ → ℕ)
  (injective_f : Function.Injective f)
  (root_condition : ∀ (n : ℕ) (a : Fin n.succ → ℝ),
    (∃ x : ℝ, polynomial (List.ofFn a) x = 0) ↔ 
    (∃ x : ℝ, polynomial (List.ofFn (λ i, a i)) (x ^ (f i)) = 0)) :
  ∃ t : ℕ, (t % 2 = 1) ∧ (∀ n : ℕ, f n = t * n) :=
sorry

end injective_func_form_l46_46872


namespace find_b_and_c_intervals_of_monotonicity_solve_inequality_l46_46621

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def has_maximum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
∀ y, f x ≥ f y

theorem find_b_and_c (b c : ℝ) :
  is_odd (λ x, x^3 + b * x^2 + c * x) ∧ has_maximum_at (λ x, x^3 + b * x^2 + c * x) (-1) →
  b = 0 ∧ c = -3 := sorry

theorem intervals_of_monotonicity :
  ( ∀ x, f x = x^3 - 3x ) →
  ( ∀ x, f' x = 3 * x^2 - 3 ) →
  ( { x | x < -1 } ∪ { x | x > 1 } ) ∧ ( { x | -1 < x ∧ x < 1 }) := sorry

theorem solve_inequality :
  ( ∀ x, f x = x^3 - 3x ) →
  ( ∀ x, |f x| ≤ 2 ) →
  { x | -2 ≤ x ∧ x ≤ 2 } := sorry

end find_b_and_c_intervals_of_monotonicity_solve_inequality_l46_46621


namespace median_length_triangle_l46_46702

theorem median_length_triangle 
    (a b c : ℝ)
    (h₁ : a = 10) 
    (h₂ : b = 8) 
    (h₃ : c = 7) : 
    sqrt ((2*b^2 + 2*c^2 - a^2) / 4) = 3 * sqrt 14 / 2 :=
by 
  rw [h₁, h₂, h₃]
  sorry

end median_length_triangle_l46_46702


namespace sum_of_faces_edges_vertices_l46_46370

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46370


namespace sum_of_integral_solutions_l46_46895

open Real

theorem sum_of_integral_solutions :
  (∑ x in Finset.filter (λ x : ℤ, (∃ y : ℝ, y = sqrt (x - 1) ∧ sqrt (x + 3 - 4 * y) + sqrt (x + 8 - 6 * y) = 1)) (Finset.Icc 5 10)) = 45 :=
by
  sorry

end sum_of_integral_solutions_l46_46895


namespace lcm_48_180_value_l46_46029

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l46_46029


namespace largest_four_digit_sum_20_l46_46708

theorem largest_four_digit_sum_20 : ∃ n : ℕ, (999 < n ∧ n < 10000 ∧ (sum (nat.digits 10 n) = 20 ∧ ∀ m, 999 < m ∧ m < 10000 ∧ sum (nat.digits 10 m) = 20 → m ≤ n)) :=
by
  sorry

end largest_four_digit_sum_20_l46_46708


namespace sum_faces_edges_vertices_eq_26_l46_46340

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46340


namespace collinear_vectors_x_value_l46_46545

theorem collinear_vectors_x_value :
  ∃ x : ℝ, let a := (2, x) in let b := (3, 6) in collinear a b ∧ x = 4 :=
by
  -- Definitions of the vectors
  let a : ℝ × ℝ := (2, _)
  let b : ℝ × ℝ := (3, 6)
  -- Collinearity condition (using proportionality)
  have collinear_cond : 2 / 3 = _ / 6 := sorry
  -- Solve for x
  have x_value : _ = 4 := sorry
  -- Existential quantifier
  exact ⟨4, collinear_cond, x_value⟩

end collinear_vectors_x_value_l46_46545


namespace solve_for_p_plus_s_l46_46909

theorem solve_for_p_plus_s (p q r s : ℝ) (h1 : pqrs ≠ 0)
  (h2 : ∀ x, (g (g x)) = x) : 
  p + s = 0 :=
sorry

def g (x : ℝ) := (p * x + q) / (r * x + s)

def g2 (x : ℝ) := g (g (x))

noncomputable def pqrs := p * q * r * s

lemma g_g_x_eq_x {p q r s : ℝ} (h1 : pqrs ≠ 0) (h2 : ∀ x, g (g x) = x) :
  p + s = 0 :=
sorry

end solve_for_p_plus_s_l46_46909


namespace smallest_positive_period_intervals_monotonic_increase_max_min_values_l46_46950

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem smallest_positive_period (x : ℝ) : (f (x + π)) = f x :=
sorry

theorem intervals_monotonic_increase (k : ℤ) (x : ℝ) : (k * π - π/3) ≤ x ∧ x ≤ (k * π + π/6) → ∃ a b : ℝ, a < b ∧ ∀ x : ℝ, (a ≤ x ∧ x ≤ b) →
  (f x < f (x + 1)) :=
sorry

theorem max_min_values (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π/4) : (∃ y : ℝ, y = max (f 0) (f (π/6)) ∧ y = 1) ∧ (∃ z : ℝ, z = min (f 0) (f (π/6)) ∧ z = 0) :=
sorry

end smallest_positive_period_intervals_monotonic_increase_max_min_values_l46_46950


namespace line_intersects_ellipse_max_chord_length_l46_46128

theorem line_intersects_ellipse (m : ℝ) :
  (-2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), (9 * x^2 + 6 * m * x + 2 * m^2 - 8 = 0) ∧ (y = (3 / 2) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) :=
sorry

theorem max_chord_length (m : ℝ) :
  m = 0 → (∃ (A B : ℝ × ℝ),
  ((A.1^2 / 4 + A.2^2 / 9 = 1) ∧ (A.2 = (3 / 2) * A.1 + m)) ∧
  ((B.1^2 / 4 + B.2^2 / 9 = 1) ∧ (B.2 = (3 / 2) * B.1 + m)) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 26 / 3)) :=
sorry

end line_intersects_ellipse_max_chord_length_l46_46128


namespace prob_A_and_B_same_l46_46180

def prob_choose_question := (1/2 : ℝ)

def prob_same_question : ℝ :=
  let pA_B_choose_14 := prob_choose_question * prob_choose_question
  let pA_B_choose_15 := prob_choose_question * prob_choose_question
  pA_B_choose_14 + pA_B_choose_15

theorem prob_A_and_B_same : prob_same_question = 1/2 := by
  sorry

end prob_A_and_B_same_l46_46180


namespace incorrect_rounding_conclusion_l46_46700

theorem incorrect_rounding_conclusion (x : ℝ) (h : x = 0.05018) : 
  ¬ (round_nearest_tenth x = 0.1) :=
by sorry

def round_nearest_tenth (x : ℝ) : ℝ :=
  let t := 10.0 * x in
  round t / 10.0

end incorrect_rounding_conclusion_l46_46700


namespace fraction_multiplier_l46_46156

theorem fraction_multiplier (x y : ℝ) :
  (3 * x * 3 * y) / (3 * x + 3 * y) = 3 * (x * y) / (x + y) :=
by
  sorry

end fraction_multiplier_l46_46156


namespace cakes_served_for_lunch_l46_46805

theorem cakes_served_for_lunch (total_cakes: ℕ) (dinner_cakes: ℕ) (lunch_cakes: ℕ) 
  (h1: total_cakes = 15) 
  (h2: dinner_cakes = 9) 
  (h3: total_cakes = lunch_cakes + dinner_cakes) : 
  lunch_cakes = 6 := 
by 
  sorry

end cakes_served_for_lunch_l46_46805


namespace min_value_of_function_l46_46673

theorem min_value_of_function :
  ∃ (x : ℝ) (h : x ∈ set.Icc (0 : ℝ) 2), (x^3 - 3 * x) = -2 :=
sorry

end min_value_of_function_l46_46673


namespace sum_faces_edges_vertices_l46_46358

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46358


namespace total_stars_l46_46313

/-- Let n be the number of students, and s be the number of stars each student makes.
    We need to prove that the total number of stars is n * s. --/
theorem total_stars (n : ℕ) (s : ℕ) (h_n : n = 186) (h_s : s = 5) : n * s = 930 :=
by {
  sorry
}

end total_stars_l46_46313


namespace greatest_integer_b_for_no_real_roots_l46_46887

theorem greatest_integer_b_for_no_real_roots (b : ℤ) :
  (∀ x : ℝ, x^2 + (b:ℝ)*x + 10 ≠ 0) ↔ b ≤ 6 :=
sorry

end greatest_integer_b_for_no_real_roots_l46_46887


namespace product_equals_32_l46_46725

theorem product_equals_32 :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_equals_32_l46_46725


namespace possible_forms_of_g_l46_46650

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ

axiom g_def : f (g x) = 9 * x^4 + 6 * x^2 + 1

theorem possible_forms_of_g (x : ℝ) :
  (g x = 3 * x^2 + 1) ∨ (g x = -(3 * x^2 + 1)) :=
sorry

end possible_forms_of_g_l46_46650


namespace correct_multiplication_l46_46775

theorem correct_multiplication :
  ∃ x : ℤ, 136 * x - 1224 = 136 * 34 ∧ x = 43 :=
by
  use 43
  split
  sorry

end correct_multiplication_l46_46775


namespace buckets_needed_to_fill_tank_l46_46594

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem buckets_needed_to_fill_tank :
  let radius_tank := 8
  let height_tank := 32
  let radius_bucket := 8
  let volume_bucket := volume_of_sphere radius_bucket
  let volume_tank := volume_of_cylinder radius_tank height_tank
  volume_tank / volume_bucket = 3 :=
by sorry

end buckets_needed_to_fill_tank_l46_46594


namespace alternating_draw_probability_l46_46755

theorem alternating_draw_probability :
  (∃ (S : finset (fin 8)), S.card = 3 ∧
   ∀ (i ∈ S), ∀ (j ∈ S), i ≠ j → (|i - j| > 1 ∧ i ≠ 0 ∧ i ≠ 7)) →
  (5 / 14 : ℝ) :=
sorry

end alternating_draw_probability_l46_46755


namespace determine_a_l46_46005

theorem determine_a (a b c : ℤ) (h : (b + 11) * (c + 11) = 2) (hb : b + 11 = -2) (hc : c + 11 = -1) :
  a = 13 := by
  sorry

end determine_a_l46_46005


namespace length_of_shorter_side_l46_46804

theorem length_of_shorter_side 
  (W : ℝ := 50) -- Width of the rectangular plot in meters
  (poles : ℕ := 32) -- Number of poles
  (distance_between_poles : ℝ := 5) -- Distance between each pole in meters
  (P : ℝ := (poles - 1) * distance_between_poles) -- Perimeter calculation
  : (2 * (27.5 + W) = P) := 
by 
  have W_def : W = 50 := rfl
  have poles_def : poles = 32 := rfl
  have distance_def : distance_between_poles = 5 := rfl
  have perimeter_calculation : P = 155 := by 
    unfold P
    rw [poles_def, distance_def]
    simp
  rw [W_def, perimeter_calculation]
  simp
  norm_num

end length_of_shorter_side_l46_46804


namespace find_x_y_sum_squared_l46_46511

theorem find_x_y_sum_squared (x y : ℝ) (h1 : x * y = 6) (h2 : (1 / x^2) + (1 / y^2) = 7) (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := sorry

end find_x_y_sum_squared_l46_46511


namespace solve_eq1_solve_eq2_solve_eq3_l46_46646

-- Problem 1: Solving the equation x^2 - 25 = 0
theorem solve_eq1 (x : ℝ) : x^2 - 25 = 0 ↔ x = 5 ∨ x = -5 := by
  sorry

-- Problem 2: Solving the equation (2x - 1)^3 = -8
theorem solve_eq2 (x : ℝ) : (2x - 1)^3 = -8 ↔ x = -1 / 2 := by
  sorry

-- Problem 3: Solving the equation 4(x + 1)^2 = 8
theorem solve_eq3 (x : ℝ) : 4 * (x + 1)^2 = 8 ↔ x = -1 - real.sqrt 2 ∨ x = -1 + real.sqrt 2 := by
  sorry

end solve_eq1_solve_eq2_solve_eq3_l46_46646


namespace quadratic_has_real_roots_l46_46486

theorem quadratic_has_real_roots (k : ℝ) : (k * x^2 - 4 * x + 2 = 0) → (k ≤ 2 ∧ k ≠ 0) :=
by
  -- Consider the discriminant condition for real roots: Δ = b^2 - 4ac ≥ 0, where a = k, b = -4, c = 2
  let Δ := (-4)^2 - 4 * k * 2
  have h1 : Δ ≥ 0 → 16 - 8 * k ≥ 0 := sorry,
  have h2 : 16 - 8 * k ≥ 0 → k ≤ 2 := sorry,
  have h3 : k ≠ 0,
  exact ⟨h2 h1, h3⟩

end quadratic_has_real_roots_l46_46486


namespace flagpole_break_height_proof_l46_46774

-- Define the conditions
variable (AB : ℝ) (BC : ℝ)
-- The original height of the flagpole AB is 7 meters
def flagpole_original_height : ℝ := 7
-- The distance from the base where the top part touches the ground is 2 meters
def flagpole_top_distance_from_base : ℝ := 2

-- Define the point where the pole breaks AD
def flagpole_break_height (AD : ℝ) : Prop :=
  AC = sqrt (flagpole_original_height ^ 2 + flagpole_top_distance_from_base ^ 2) ∧
  AC = 2 * AD

-- The theorem to be proved
theorem flagpole_break_height_proof : ∃ x : ℝ, flagpole_break_height x ∧ x = sqrt 53 / 2 :=
begin
  sorry
end

end flagpole_break_height_proof_l46_46774


namespace bm_eq_cd_l46_46210

structure Triangle :=
(a b c : ℝ)

structure IncircleTangentPoints (T : Triangle) :=
(D : ℝ)  -- D is a point on side b-c
(E : ℝ)  -- E is a point of tangency of the tangent parallel to b-c

-- Define point M
def point_M {T : Triangle} (A E BC : ℝ): ℝ := sorry

-- Prove the equality BM = CD
theorem bm_eq_cd (T : Triangle) (I : IncircleTangentPoints T) (A BC M : ℝ) :
  let B := T.b in
  let C := T.c in
  BM = (I.D) := sorry

end bm_eq_cd_l46_46210


namespace pyramid_volume_of_encompassing_sphere_l46_46939

theorem pyramid_volume_of_encompassing_sphere :
  (∀ (a b c : ℝ) (R : ℝ),
    a = 1 →
    b = 1 →
    c = √2 →
    R = 1 →
    (4 / 3) * Real.pi * R^3 = (4 / 3) * Real.pi) :=
by
  intros a b c R ha hb hc hR
  rw [ha, hb, hc, hR]
  simp
  sorry

end pyramid_volume_of_encompassing_sphere_l46_46939


namespace find_a_l46_46197

-- Define the lines as given
def line1 (x y : ℝ) := 2 * x + y - 5 = 0
def line2 (x y : ℝ) := x - y - 1 = 0
def line3 (a x y : ℝ) := a * x + y - 3 = 0

-- Define the condition that they intersect at a single point
def lines_intersect_at_point (x y a : ℝ) := line1 x y ∧ line2 x y ∧ line3 a x y

-- To prove: If lines intersect at a certain point, then a = 1
theorem find_a (a : ℝ) : (∃ x y, lines_intersect_at_point x y a) → a = 1 :=
by
  sorry

end find_a_l46_46197


namespace total_people_after_one_hour_l46_46439

variable (x y Z : ℕ)

def ferris_wheel_line_initial := 50
def bumper_cars_line_initial := 50
def roller_coaster_line_initial := 50

def ferris_wheel_line_after_half_hour := ferris_wheel_line_initial - x
def bumper_cars_line_after_half_hour := bumper_cars_line_initial + y

axiom Z_eq : Z = ferris_wheel_line_after_half_hour + bumper_cars_line_after_half_hour

theorem total_people_after_one_hour : (Z = (50 - x) + (50 + y)) -> (Z + 100) = ((50 - x) + (50 + y) + 100) :=
by {
  sorry
}

end total_people_after_one_hour_l46_46439


namespace tenth_term_is_six_l46_46015

def sequence : ℕ → ℝ
| 0 => 3
| n+1 => if n % 2 = 0 then 6 else 3

theorem tenth_term_is_six :
  sequence 9 = 6 := sorry

end tenth_term_is_six_l46_46015


namespace pair_70th_l46_46918

def sequence (n : ℕ) : List (ℕ × ℕ) :=
  (List.range n).map (λ i => (i + 1, n - i))

def pair (k : ℕ) : ℕ × ℕ :=
  let ⟨n, r⟩ := (List.range (k + 1)).find (λ s => (s * (s + 1)) / 2 >= k)
  let idx := k - (n * (n - 1)) / 2
  (sequence n)!!.get (idx - 1)

theorem pair_70th : pair 70 = (4, 9) :=
by
  unfold pair
  unfold sequence
  sorry

end pair_70th_l46_46918


namespace cos_45_degree_l46_46452

theorem cos_45_degree : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degree_l46_46452


namespace inscribed_circle_radius_l46_46587

-- Defining the conditions
structure TrapezoidABCDE (A B C D E: ℝ) : Prop :=
(AD_eq_4 : A  - D = 4)
(BC_eq_1 : B - C = 1)
(angle_A : angle A = arctan 2)
(angle_D : angle D = arctan 3)
(E_intersection : intersect AC BD = E)

-- The theorem statement
theorem inscribed_circle_radius {A B C D E: ℝ}
  (T: TrapezoidABCDE A B C D E):
  let r := (18 / (25 + 2 * real.sqrt 130 + real.sqrt 445)) in
  radius (inscribed_circle (triangle C B E)) = r :=
by
  sorry

end inscribed_circle_radius_l46_46587


namespace incorrect_regression_statement_l46_46059

theorem incorrect_regression_statement (x y : Type)
  (data : list (x × y))
  (linear_regression : ∀ data : list (x × y), ℝ → ℝ → Prop)
  (sum_squared_residuals : ∀ data : list (x × y), ℝ)
  (correlation_index_R2 : ∀ data : list (x × y), ℝ)
  (correlation_coefficient : ∀ data : list (x × y), ℝ)
  (center_of_sample : ∀ data : list (x × y), (ℝ × ℝ)) :
  -- Statements
  (∀ data, linear_regression data (center_of_sample data).1 (center_of_sample data).2) →
  (∀ data, sum_squared_residuals data ≥ 0) →
  (∀ data, 0 ≤ correlation_index_R2 data ∧ correlation_index_R2 data ≤ 1) →
  (∀ data, -1 ≤ correlation_coefficient data ∧ correlation_coefficient data ≤ 1) →
  -- Incorrect statement
  (∃ data, correlation_index_R2 data < 0) = false →
  (∃ data, correlation_index_R2 data < correlation_index_R2 data) → false :=
by
  sorry

end incorrect_regression_statement_l46_46059


namespace nonzero_fraction_exponent_zero_l46_46445

theorem nonzero_fraction_exponent_zero (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : (a / b : ℚ)^0 = 1 := 
by 
  sorry

end nonzero_fraction_exponent_zero_l46_46445


namespace scientific_notation_of_130944000000_l46_46464

theorem scientific_notation_of_130944000000 :
  130944000000 = 1.30944 * 10^11 :=
by sorry

end scientific_notation_of_130944000000_l46_46464


namespace sum_faces_edges_vertices_eq_26_l46_46339

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46339


namespace train_cross_pole_in_time_l46_46388

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_pole_in_time :
  time_to_cross_pole 100 126 = 100 / (126 * (1000 / 3600)) :=
by
  -- this will unfold the calculation step-by-step
  unfold time_to_cross_pole
  sorry

end train_cross_pole_in_time_l46_46388


namespace ellen_needs_thirteen_golf_carts_l46_46862

theorem ellen_needs_thirteen_golf_carts :
  ∀ (patrons_from_cars patrons_from_bus patrons_per_cart : ℕ), 
  patrons_from_cars = 12 → 
  patrons_from_bus = 27 → 
  patrons_per_cart = 3 →
  (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := 
by 
  intros patrons_from_cars patrons_from_bus patrons_per_cart h1 h2 h3 
  have h: patrons_from_cars + patrons_from_bus = 39 := by 
    rw [h1, h2] 
    norm_num
  rw[h, h3]
  norm_num
  sorry

end ellen_needs_thirteen_golf_carts_l46_46862


namespace phone_number_exists_l46_46396

theorem phone_number_exists :
  ∃ (phone_number : ℕ),
    phone_number >= 1000000 ∧ phone_number < 10000000 ∧
    (let last_three_digits := phone_number % 1000 in
     (∃ n : ℕ, last_three_digits = 100 * n + 10 * (n + 1) + (n + 2))) ∧
    let first_five_digits := phone_number / 1000 in
    (let a := first_five_digits / 1000,
         b := (first_five_digits / 100) % 10,
         c := (first_five_digits / 10) % 10 in
     (first_five_digits = 10000 * a + 1000 * b + 100 * c + 10 * b + a) ∧
     (c = 1) ∧
     (b = 1 ∨ a = 1)) ∧
    let three_digit_number := phone_number / 1000 in
    (three_digit_number % 9 = 0) ∧
    (phone_number / 100 = 7111 ∨ (phone_number / 10) % 1000 = 111) ∧
    let first_two_digits := (phone_number / 10000) % 100,
        second_two_digits := (phone_number / 100) % 100 in
    (Nat.Prime first_two_digits ∨ Nat.Prime second_two_digits) ∧
    phone_number = 7111765 := sorry

end phone_number_exists_l46_46396


namespace axis_of_symmetry_l46_46136

def f (w x : ℝ) : ℝ := sin (w * x) + (sqrt 3) * cos (w * x)

theorem axis_of_symmetry (w : ℝ) (h₁ : ∃ a b : ℝ, f w a = 2 ∧ f w b = 2 ∧ abs (a - b) = π)
  (h₂ : f 2 = λ x, 2 * sin (2 * x + π / 3)) :
  ∃ k : ℤ, let x := k * (π / 2) + π / 12 in x = π / 12 :=
by
  sorry

end axis_of_symmetry_l46_46136


namespace sum_faces_edges_vertices_eq_26_l46_46338

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46338


namespace millet_exceeds_half_l46_46624

noncomputable def seeds_millet_day (n : ℕ) : ℝ :=
  0.2 * (1 - 0.7 ^ n) / (1 - 0.7) + 0.2 * 0.7 ^ n

noncomputable def seeds_other_day (n : ℕ) : ℝ :=
  0.3 * (1 - 0.1 ^ n) / (1 - 0.1) + 0.3 * 0.1 ^ n

noncomputable def prop_millet (n : ℕ) : ℝ :=
  seeds_millet_day n / (seeds_millet_day n + seeds_other_day n)

theorem millet_exceeds_half : ∃ n : ℕ, prop_millet n > 0.5 ∧ n = 3 :=
by sorry

end millet_exceeds_half_l46_46624


namespace range_k_for_intersection_l46_46945

theorem range_k_for_intersection {k : ℝ} :
  (∀ x y : ℝ, (x^2 - y^2 + 1 = 0) → (y^2 = (k-1)*x) → x*x - (k+1)*x + 1 = 0) →
  k ∈ Icc (-1 : ℝ) (3 : ℝ) :=
by
  intros x y hyp_hyperbola hyp_parabola
  sorry

end range_k_for_intersection_l46_46945


namespace ring_sequence_total_distance_l46_46810

theorem ring_sequence_total_distance : 
  let top_ring_outer_diameter := 40
  let decrement_per_ring := 2
  let bottom_ring_outer_diameter := 4
  let ring_thickness := 2
  let inside_diameter_seq (n : ℕ) := top_ring_outer_diameter - n * decrement_per_ring - ring_thickness
  let n := (top_ring_outer_diameter - bottom_ring_outer_diameter) / decrement_per_ring + 1 
  (2 * n * (top_ring_outer_diameter - ring_thickness) / 2 + (n - 1) * (-decrement_per_ring)) / 2 = 342 :=
begin
  sorry
end

end ring_sequence_total_distance_l46_46810


namespace john_small_planks_l46_46595

theorem john_small_planks (L S : ℕ) (h1 : L = 12) (h2 : L + S = 29) : S = 17 :=
by {
  sorry
}

end john_small_planks_l46_46595


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46344

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46344


namespace sequence_gt_one_l46_46497

variable {b : ℝ} (n : ℕ)

-- Conditions
def a : ℕ → ℝ
| 0 => 1 + b
| (n + 1) => (1 / a n) + b

-- Theorem
theorem sequence_gt_one (h₀ : 0 < b) (h₁ : b < 1) : ∀ n : ℕ, a n > 1 := 
sorry

end sequence_gt_one_l46_46497


namespace count_b_k_divisible_by_9_l46_46610

/--
Let b (n : ℕ) be the number obtained by concatenating the squares of integers from 1 to n.
For example, b 4 is 1491625 since 1^2 = 1, 2^2 = 4, 3^2 = 9, and 4^2 = 16, concatenating these gives 1491625.
We want to prove that there are 11 such b k's which are divisible by 9 for 1 ≤ k ≤ 50.
-/
theorem count_b_k_divisible_by_9 : (finset.range 50).filter (λ k, (∑ i in finset.range (k + 1), i^2) % 9 = 0).card = 11 :=
sorry

end count_b_k_divisible_by_9_l46_46610


namespace polynomial_remainders_l46_46069

variable {R : Type*} [Field R]

theorem polynomial_remainders
  (a b c : R) (f : Polynomial R)
  (p q r l m n : R)
  (h_abc_neq_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_div_remainder_1 : ∀ x, f(x) ≡ p * x + l [MOD (x - a)*(x - b)])
  (h_div_remainder_2 : ∀ x, f(x) ≡ q * x + m [MOD (x - b)*(x - c)])
  (h_div_remainder_3 : ∀ x, f(x) ≡ r * x + n [MOD (x - c)*(x - a)]) :
  l * (1 / a - 1 / b) + m * (1 / b - 1 / c) + n * (1 / c - 1 / a) = 0 :=
by
  sorry

end polynomial_remainders_l46_46069


namespace upper_base_length_l46_46095

structure Trapezoid (A B C D M : Type) :=
  (on_lateral_side : ∀ {AB DM}, DM ⊥ AB)
  (perpendicular : ∀ {DM AB}, DM ⊥ AB)
  (equal_segments : MC = CD)
  (AD_length : AD = d)


theorem upper_base_length {A B C D M : Type} [trapezoid : Trapezoid A B C D M] :
  BC = d / 2 := 
begin
  sorry
end

end upper_base_length_l46_46095


namespace graph_is_empty_l46_46009

theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + y^2 - 9 * x - 4 * y + 17 ≠ 0 :=
by
  intros x y
  sorry

end graph_is_empty_l46_46009


namespace upper_base_length_l46_46094

structure Trapezoid (A B C D M : Type) :=
  (on_lateral_side : ∀ {AB DM}, DM ⊥ AB)
  (perpendicular : ∀ {DM AB}, DM ⊥ AB)
  (equal_segments : MC = CD)
  (AD_length : AD = d)


theorem upper_base_length {A B C D M : Type} [trapezoid : Trapezoid A B C D M] :
  BC = d / 2 := 
begin
  sorry
end

end upper_base_length_l46_46094


namespace linear_equation_solution_l46_46165

theorem linear_equation_solution (x y b : ℝ) (h1 : x - 2*y + b = 0) (h2 : y = (1/2)*x + b - 1) :
  b = 2 :=
by
  sorry

end linear_equation_solution_l46_46165


namespace combined_selling_price_correctness_l46_46812

def cost_price_first := 70
def cost_price_second := 120
def cost_price_third := 150

def selling_price_first := ((0.85 * cost_price_first) * 3 / 2)
def selling_price_second := cost_price_second + (0.3 * cost_price_second)
def selling_price_third := cost_price_third - (0.2 * cost_price_third)

def combined_selling_price := selling_price_first + selling_price_second + selling_price_third

theorem combined_selling_price_correctness : combined_selling_price = 365.25 := by
  sorry

end combined_selling_price_correctness_l46_46812


namespace ellipse_h_k_a_b_sum_l46_46228

noncomputable def h := 3
noncomputable def k := 2
noncomputable def a := 5
noncomputable def b := 4

theorem ellipse_h_k_a_b_sum :
  F1 : (ℝ × ℝ) := (0, 2),
  F2 : (ℝ × ℝ) := (6, 2),
  set_of_points_P_property : ∀ P : (ℝ × ℝ), dist P F1 + dist P F2 = 10 → ∃ h k a b : ℝ, 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1,
  h + k + a + b = 14 :=
by 
  exact sorry

end ellipse_h_k_a_b_sum_l46_46228


namespace length_of_BC_l46_46572

theorem length_of_BC 
  {O A B C D : Point}  -- Points in the circle
  (h1 : is_circle_center O)  -- Center of the circle
  (h2 : is_diameter A D O)  -- AD is the diameter
  (h3 : is_chord A B C)  -- ABC is a chord
  (h4 : distance O B = 13)  -- BO = 13
  (h5 : angle A B O = 90)  -- ∠ABO = 90°
  : distance B C = 13 := 
sorry

end length_of_BC_l46_46572


namespace Carlson_max_candies_l46_46314

theorem Carlson_max_candies :
  ∀ n : ℕ, n ≥ 2 → ((λ n, n * (n - 1) / 2) 38 = 703) :=
by
  intro n hn
  have h : (λ n, n * (n - 1) / 2) 38 = 703
  exact calc (38 * 37) / 2 = 703
  sorry -- Proof omitted

end Carlson_max_candies_l46_46314


namespace volume_spherical_segment_l46_46612

-- Defining the conditions
variables {R h : ℝ}

-- Stating the theorem
theorem volume_spherical_segment (h R : ℝ) : 
  (h > 0) → (R > 0) → 
  volume_of_spherical_segment h R = (π * h^2 * (3 * R - h)) / 3 :=
sorry

end volume_spherical_segment_l46_46612


namespace simplify_fraction_when_b_equals_4_l46_46266

theorem simplify_fraction_when_b_equals_4 (b : ℕ) (h : b = 4) : (18 * b^4) / (27 * b^3) = 8 / 3 :=
by {
  -- we use the provided condition to state our theorem goals.
  sorry
}

end simplify_fraction_when_b_equals_4_l46_46266


namespace possible_values_of_k_l46_46298

theorem possible_values_of_k (n : ℕ) (hn : n ≥ 3) :
  ∃ k : ℕ, (k ≥ 4 ∧ ∀ k is a power of 2) ∧
  ∀ m ∈ {1, 2, ..., n}, last k (perform_moves m) :=
sorry

end possible_values_of_k_l46_46298


namespace number_of_terms_in_arithmetic_sequence_l46_46148

def arithmetic_sequence (a d : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := (arithmetic_sequence a d n) + d

def first_term := 17
def second_term := 23
def last_term := 101
def common_difference := second_term - first_term

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, arithmetic_sequence first_term common_difference n = last_term ∧ n + 1 = 15 :=
by
  -- Proof omitted
  sorry

end number_of_terms_in_arithmetic_sequence_l46_46148


namespace evaluate_expression_l46_46869

theorem evaluate_expression : (4 * nat.factorial 7 + 28 * nat.factorial 6) / nat.factorial 8 = 1 := by
  sorry

end evaluate_expression_l46_46869


namespace part1_monotonicity_part2_inequality_l46_46966

variable (a : ℝ) (x : ℝ)
variable (h : a > 0)

def f := a * (Real.exp x + a) - x

theorem part1_monotonicity :
  if a ≤ 0 then ∀ x : ℝ, (f a x) < (f a (x + 1)) else
  if a > 0 then ∀ x : ℝ, (x < Real.log (1 / a) → (f a x) > (f a (x + 1))) ∧ 
                       (x > Real.log (1 / a) → (f a x) < (f a (x - 1))) else 
  (False) := sorry

theorem part2_inequality :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3 / 2 := sorry

end part1_monotonicity_part2_inequality_l46_46966


namespace largest_common_number_in_range_l46_46828

theorem largest_common_number_in_range (n1 d1 n2 d2 : ℕ) (h1 : n1 = 2) (h2 : d1 = 4) (h3 : n2 = 5) (h4 : d2 = 6) :
  ∃ k : ℕ, k ≤ 200 ∧ (∀ n3 : ℕ, n3 = n1 + d1 * k) ∧ (∀ n4 : ℕ, n4 = n2 + d2 * k) ∧ n3 = 190 ∧ n4 = 190 := 
by {
  sorry
}

end largest_common_number_in_range_l46_46828


namespace coefficient_x3_y2_z3_in_expansion_l46_46381

theorem coefficient_x3_y2_z3_in_expansion :
  (∃ n a b c : ℕ, n = 8 ∧ a = 3 ∧ b = 2 ∧ c = 3 ∧ a + b + c = n ∧
      (Nat.factorial n) / ((Nat.factorial a) * (Nat.factorial b) * (Nat.factorial c)) = 560) :=
by
  use 8, 3, 2, 3
  split; try { exact rfl }
  split; try { exact rfl }
  split; try { exact rfl }
  split; try { exact rfl }
  have h_eq : 3 + 2 + 3 = 8 := by rfl
  split; exact h_eq
  have h_factorial : (Nat.factorial 8) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 3)) = 560 := sorry
  exact h_factorial

end coefficient_x3_y2_z3_in_expansion_l46_46381


namespace morning_prob_exactly_two_teachers_avg_number_computers_used_prob_demand_not_met_l46_46579

theorem morning_prob_exactly_two_teachers (pA pB pC : ℚ) (hpA : pA = 1/4) (hpB : pB = 2/3) (hpC : pC = 2/5) :
  let p := pA * pB * (1 - pC) + pA * (1 - pB) * pC + (1 - pA) * pB * pC in
  p = 1/3 :=
by
  sorry

theorem avg_number_computers_used (p : ℚ) (hp : p = 1/3) (n : ℕ) (hn : n = 5) :
  let E := n * p in
  E = 5/3 :=
by
  sorry

theorem prob_demand_not_met (p : ℚ) (hp : p = 1/3) (n : ℕ) (hn : n = 5) :
  let P4 := combinatorial 5 4 * p^4 * (1 - p) in
  let P5 := p^5 in
  let P := P4 + P5 in
  P = 11/243 :=
by
  sorry

end morning_prob_exactly_two_teachers_avg_number_computers_used_prob_demand_not_met_l46_46579


namespace problem_conditions_l46_46557

variables (a b : ℝ)
open Real

theorem problem_conditions (ha : a < 0) (hb : 0 < b) (hab : a + b > 0) :
  (a / b > -1) ∧ (abs a < abs b) ∧ (1 / a + 1 / b ≤ 0) ∧ ((a - 1) * (b - 1) < 1) := sorry

end problem_conditions_l46_46557


namespace sum_of_a_approx_l46_46878

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∃ x1 x2 x3 : ℝ,
  (x1 = -4 * π * a / (4 - a)) ∧ (x2 = -4 * π * a / (4 - a)) ∧ (x3 = -4 * π * a / (4 - a)) ∧
  0 <= x1 ∧ x1 < π ∧ 0 <= x2 ∧ x2 < π ∧ 0 <= x3 ∧ x3 < π

theorem sum_of_a_approx : ∀ a : ℝ,
  problem_statement a →
  a = 1 ∨ a = 3 ∨ a = 16/3 →
  1 + 3 + 5.33 ≈ 9.33 := 
sorry

end sum_of_a_approx_l46_46878


namespace monotonicity_case1_monotonicity_case2_lower_bound_l46_46972

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 (a : ℝ) (h : a ≤ 0) : 
  ∀ x y : ℝ, x < y → f a x > f a y := 
by
  sorry

theorem monotonicity_case2 (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, x < Real.log (1 / a) → f a x > f a (Real.log (1 / a)) ∧ 
           x > Real.log (1 / a) → f a x < f a (Real.log (1 / a)) := 
by
  sorry

theorem lower_bound (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ,  f a x > 2 * Real.log a + 3 / 2 := 
by
  sorry

end monotonicity_case1_monotonicity_case2_lower_bound_l46_46972


namespace binomial_12_5_l46_46448

def binomial_coefficient : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binomial_12_5 : binomial_coefficient 12 5 = 792 := by
  sorry

end binomial_12_5_l46_46448


namespace trajectory_eq_circle_line_eq_conditions_l46_46924

-- Definitions for points M1 and M2
def M1 := (26 : ℝ, 1 : ℝ)
def M2 := (2 : ℝ, 1 : ℝ)

-- Definitions for midpoint and distance
def dist (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Given conditions in Lean
def condition1 (M : ℝ × ℝ) : Prop := dist M M1 = 5 * dist M M2
def condition2 (M := (-2, 3) : ℝ × ℝ) : Prop := True
def condition3 (C : ℝ × ℝ → Prop) (l : ℝ × ℝ → ℝ) : Prop := ∃ (A B : ℝ × ℝ), C(A) ∧ C(B) ∧ l(A) = 0 ∧ l(B) = 0 ∧ dist A B = 8

-- The trajectory is a circle
def C (M : ℝ × ℝ) : Prop := (M.1 - 1)^2 + (M.2 - 1)^2 = 25

-- The lines to be proven
def l1 (M : ℝ × ℝ) : ℝ := M.1 + 2
def l2 (M : ℝ × ℝ) : ℝ := 5 * M.1 - 12 * M.2 + 46

-- Main goal statements
theorem trajectory_eq_circle (M : ℝ × ℝ) : condition1 M → C M :=
by sorry

theorem line_eq_conditions : (∃ l, ∀ M, condition2 → condition3 C l) → (∀ M, (l1 M = 0 ∨ l2 M = 0)) :=
by sorry

end trajectory_eq_circle_line_eq_conditions_l46_46924


namespace num_distinct_triangles_factors_2001_l46_46518

theorem num_distinct_triangles_factors_2001 : 
  let factors := [1, 3, 23, 29, 69, 87, 667, 2001]
  ∃ tris : list (ℕ × ℕ × ℕ), 
    (∀ (a b c : ℕ), (a, b, c) ∈ tris → a ∈ factors ∧ b ∈ factors ∧ c ∈ factors 
      ∧ a + b > c ∧ a + c > b ∧ b + c > a) 
      ∧ tris.length = 7 :=
by
  sorry

end num_distinct_triangles_factors_2001_l46_46518


namespace integer_solution_exists_l46_46920

theorem integer_solution_exists (p : ℕ) (q : ℕ) (a : Fin p → Fin q → ℤ) 
  (h : ∀ (i : Fin p) (j : Fin q), a i j ∈ {-1, 0, 1}) : 
  q = 2 * p → 
  ∃ (x : Fin q → ℤ), 
    (∀ i : Fin p, ∑ j in Finset.univ, a i j * x j = 0) ∧ 
    (∃ j : Fin q, x j ≠ 0) ∧ 
    (∀ j : Fin q, |x j| ≤ q) :=
by {
  let q := 2 * p,
  sorry
}

end integer_solution_exists_l46_46920


namespace Q_is_234_l46_46539

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {z | ∃ x y : ℕ, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_is_234 : Q = {2, 3, 4} :=
by
  sorry

end Q_is_234_l46_46539


namespace problem1_problem2_l46_46903

variables {m n : ℝ}

def vector := ℝ × ℝ

def AB : vector := (-1, 3)
def BC : vector := (3, m)
def CD : vector := (1, n)
def AD := (AB.1 + BC.1 + CD.1, AB.2 + BC.2 + CD.2)

theorem problem1 (h_parallel : ∃ λ : ℝ, AD = (λ * BC.1, λ * BC.2)) :
  n = -3 :=
by sorry

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def AC := (AB.1 + BC.1, AB.2 + BC.2)
def BD := (BC.1 + CD.1, BC.2 + CD.2)

theorem problem2 (h_perpendicular : dot_product AC BD = 0) (h_n_eq : n = -3) :
  m = 1 ∨ m = -1 :=
by sorry

end problem1_problem2_l46_46903


namespace work_duration_B_l46_46767

theorem work_duration_B (x : ℕ) (h : x = 10) : 
  (x * (1 / 15 : ℚ)) + (2 * (1 / 6 : ℚ)) = 1 := 
by 
  rw [h]
  sorry

end work_duration_B_l46_46767


namespace determine_quadrant_l46_46201

def pointInWhichQuadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On axis or origin"

theorem determine_quadrant : pointInWhichQuadrant (-7) 3 = "Second quadrant" :=
by
  sorry

end determine_quadrant_l46_46201


namespace sum_faces_edges_vertices_l46_46356

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46356


namespace bus_overloaded_l46_46758

theorem bus_overloaded : 
  ∀ (capacity : ℕ) (first_pickup_ratio : ℚ) (next_pickup : ℕ) (bus_full : capacity = 80) (entered_first : first_pickup_ratio = 3/5) (next_pickup_point_waiting : next_pickup = 50), 
  let entered := (first_pickup_ratio * capacity).to_nat in -- people entered at first pickup
  let available_seats := capacity - entered in -- available seats after first pickup
  let could_not_take_bus := next_pickup - available_seats in -- people who could not take the bus
  could_not_take_bus = 18 := 
by 
  intros capacity first_pickup_ratio next_pickup bus_full entered_first next_pickup_point_waiting 
  let entered := (first_pickup_ratio * capacity).to_nat 
  let available_seats := capacity - entered 
  let could_not_take_bus := next_pickup - available_seats 
  sorry

end bus_overloaded_l46_46758


namespace parabola_equation_l46_46293

theorem parabola_equation (k : ℝ) :
  (∃ h : ℝ, ∃ k : ℝ, 
    (2 = k * (7 - h)^2) ∧ -- parabola passing through (2,7)
    (h = 5) ∧ -- vertex y-coordinate when focus' y-coordinate is 5
    (axis_of_symmetry_parallel_to_x : Prop) ∧ -- axis of symmetry condition
    (vertex_on_y_axis : Prop) ∧ -- the vertex lies on y-axis
    (k = 1 / 2) ∧ -- specific value of k from point (2,7)
    (y - 5) has a specific distance from the vertex
  ) → 
  (x = k * (y - 5)^2) → 
  (c ≠ 0)  →  -- |c| positive integer condition
  (gcd (0 : ℕ) (0 : ℕ) 1 (-2) (-10) 25 = 1) := -- gcd condition
suffices : x = 1 / 2 * (y - 5)^2 →  y^2 - 2x - 10y + 25 = 0, -- final form of the solution
sorry -- proof of the theorem

end parabola_equation_l46_46293


namespace digit_at_57_is_1_l46_46159

noncomputable def digit_at_position (n: ℕ) : ℕ :=
if n ≤ 56 then
  (let idx := (n - 1) / 2 in -- get the index of the two-digit number
  let num := 40 - idx in    -- get the two-digit number itself
  if (n - 1) % 2 = 0 then num / 10 else num % 10) -- return the correct digit
else
  (let rem := n - 56 in
  let num := 12 - (rem - 1) / 2 in
  if (rem - 1) % 2 = 0 then num / 10 else num % 10)

theorem digit_at_57_is_1 : digit_at_position 57 = 1 :=
sorry

end digit_at_57_is_1_l46_46159


namespace new_prism_volume_l46_46806

theorem new_prism_volume (L W H : ℝ) 
  (h_volume : L * W * H = 54)
  (L_new : ℝ := 2 * L)
  (W_new : ℝ := 3 * W)
  (H_new : ℝ := 1.5 * H) :
  L_new * W_new * H_new = 486 := 
by
  sorry

end new_prism_volume_l46_46806


namespace percent_increase_of_semicircle_areas_l46_46793

def area_of_semicircle (radius : ℝ) : ℝ :=
  (1 / 2) * real.pi * (radius ^ 2)

theorem percent_increase_of_semicircle_areas :
  let r_large := 6
  let r_small := 4
  let large_area := 2 * area_of_semicircle r_large
  let small_area := 2 * area_of_semicircle r_small
  (large_area / small_area - 1) * 100 = 125 :=
by
  sorry

end percent_increase_of_semicircle_areas_l46_46793


namespace inscribed_circles_touch_l46_46633

theorem inscribed_circles_touch
    (ABCD_convex : Prop)
    (circumscribed_ABC : Prop)
    (circumscribed_ADC : Prop)
    (AD_BC_eq_AB_CD : AD + BC = AB + CD) :
    (∃ K : Point, touches_diagonal K ABC) ∧ (∃ K : Point, touches_diagonal K ADC) → K1 = K2 :=
begin
  sorry
end

end inscribed_circles_touch_l46_46633


namespace harmonic_sum_not_integer_l46_46739

theorem harmonic_sum_not_integer (k n : ℕ) : ¬(∃ (m : ℤ), (∑ i in Finset.range (n + 1), (1 : ℚ) / (k + i)) = m) :=
sorry

end harmonic_sum_not_integer_l46_46739


namespace percent_increase_of_semicircle_areas_l46_46791

def area_of_semicircle (radius : ℝ) : ℝ :=
  (1 / 2) * real.pi * (radius ^ 2)

theorem percent_increase_of_semicircle_areas :
  let r_large := 6
  let r_small := 4
  let large_area := 2 * area_of_semicircle r_large
  let small_area := 2 * area_of_semicircle r_small
  (large_area / small_area - 1) * 100 = 125 :=
by
  sorry

end percent_increase_of_semicircle_areas_l46_46791


namespace bike_route_length_l46_46629

theorem bike_route_length (u1 u2 u3 l1 l2 : ℕ) (h1 : u1 = 4) (h2 : u2 = 7) (h3 : u3 = 2) (h4 : l1 = 6) (h5 : l2 = 7) :
  u1 + u2 + u3 + u1 + u2 + u3 + l1 + l2 + l1 + l2 = 52 := 
by
  sorry

end bike_route_length_l46_46629


namespace triangle_ABC_properties_l46_46569

-- Necessary definitions and conditions
variables {A B C a b c : ℝ}
noncomputable def is_acute_triangle := 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
noncomputable def sides : ℝ := 2

-- Given condition
axiom given_condition : sqrt 3 * tan A * tan B - tan A - tan B = sqrt 3

-- Problem statement
theorem triangle_ABC_properties (h : is_acute_triangle) (hc : c = sides) :
  C = π / 3 ∧ (20 / 3 < a^2 + b^2 ∧ a^2 + b^2 ≤ 8) :=
by
  sorry

end triangle_ABC_properties_l46_46569


namespace range_of_ab_l46_46675

noncomputable def a_b_range (a b : ℝ) : Prop :=
  2^a + 2^b = 1 → a + b ≤ -2

-- Statement of the theorem to be proved
theorem range_of_ab (a b : ℝ) : a_b_range a b :=
  sorry

end range_of_ab_l46_46675


namespace distance_interval_l46_46821

theorem distance_interval (d : ℝ) :
  (d < 8) ∧ (d > 7) ∧ (d > 5) ∧ (d ≠ 3) ↔ (7 < d ∧ d < 8) :=
by
  sorry

end distance_interval_l46_46821


namespace transmission_time_calc_l46_46018

theorem transmission_time_calc
  (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) (time_in_minutes : ℕ)
  (h_blocks : blocks = 80)
  (h_chunks_per_block : chunks_per_block = 640)
  (h_transmission_rate : transmission_rate = 160) 
  (h_time_in_minutes : time_in_minutes = 5) : 
  (blocks * chunks_per_block / transmission_rate) / 60 = time_in_minutes := 
by
  sorry

end transmission_time_calc_l46_46018


namespace rectangular_prism_faces_edges_vertices_sum_l46_46333

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46333


namespace num_ways_choose_7_starters_l46_46249

theorem num_ways_choose_7_starters : 
  (nat.choose 16 7) = 11440 := 
by
  sorry

end num_ways_choose_7_starters_l46_46249


namespace actual_number_of_children_l46_46246

theorem actual_number_of_children (N : ℕ) (B : ℕ) 
  (h1 : B = 2 * N)
  (h2 : ∀ k : ℕ, k = N - 330)
  (h3 : B = 4 * (N - 330)) : 
  N = 660 :=
by 
  sorry

end actual_number_of_children_l46_46246


namespace correct_statement_when_estimating_population_l46_46728

theorem correct_statement_when_estimating_population (A B C D : Prop)
  (hA : A = "The result of the sample is the result of the population")
  (hB : B = "The larger the sample size, the more accurate the estimate")
  (hC : C = "The standard deviation of the sample can approximately reflect the average state of the population")
  (hD : D = "The larger the variance of the data, the more stable the data")
  (not_hA : ¬A)
  (not_hC : ¬C)
  (not_hD : ¬D) :
  B := by
  sorry

end correct_statement_when_estimating_population_l46_46728


namespace probability_X_greater_than_2_l46_46947

-- Define the conditions
def normal_distribution (μ σ²) := sorry

def X : ℝ := sorry
def σ2 : ℝ := sorry

-- Assume X follows a normal distribution N(1, σ²)
axiom normal_X : normal_distribution 1 σ2

-- Given P(0 ≤ X ≤ 1) = 0.35
axiom prob_0_le_X_le_1 : 0.35 = sorry

-- Statement to prove
theorem probability_X_greater_than_2 : P(X > 2) = 0.15 :=
by {
  -- The proof will follow from the stated conditions, equivalence, and properties of the normal distribution
  sorry
}

end probability_X_greater_than_2_l46_46947


namespace question1_question2_l46_46071

section CircleEquations

variable {x y m : ℝ}

def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + m = 0

def line_eq (x y : ℝ) : Prop := 2 * x + y - 3 = 0

def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A in
  let (Bx, By) := B in
  let (Cx, Cy) := C in
  dist (Ax, Ay) (Bx, By) = dist (Bx, By) (Cx, Cy) ∧ dist (Bx, By) (Cx, Cy) = dist (Cx, Cy) (Ax, Ay)

theorem question1 (m : ℝ) :
  (∀ A B : ℝ × ℝ, circle_eq A.1 A.2 m ∧ circle_eq B.1 B.2 m ∧
  line_eq A.1 A.2 ∧ line_eq B.1 B.2) ∧
  ((-1, 2) = C ∧ is_equilateral_triangle A B C) →
  m = 13 / 5 :=
by
  sorry

theorem question2 :
  (∀ A B : ℝ × ℝ, circle_eq A.1 A.2 (-18 / 5) ∧ circle_eq B.1 B.2 (-18 / 5) ∧
  line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧
  passes_through_origin C)&& 
circle_eq (-sqrt(18/5)) ((-18/5)) m) :=
by
  sorry

end CircleEquations


end question1_question2_l46_46071


namespace product_equals_32_l46_46726

theorem product_equals_32 :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_equals_32_l46_46726


namespace consecutive_sunny_days_l46_46412

theorem consecutive_sunny_days (n_sunny_days : ℕ) (n_days_year : ℕ) (days_to_stay : ℕ) (condition1 : n_sunny_days = 350) (condition2 : n_days_year = 365) :
  days_to_stay = 32 :=
by
  sorry

end consecutive_sunny_days_l46_46412


namespace milo_number_of_5s_l46_46625

-- Definition of the problem constraints
def total_grades := 3 + 4 + 1 + x

def sum_grades := 3 * 2 + 4 * 3 + 1 * 4 + 5 * x

def average_grade (x : ℕ) := (22 + 5 * x) / (8 + x)

def cash_reward (x : ℕ) := 5 * average_grade x

-- The statement to prove
theorem milo_number_of_5s (x : ℕ) (h : cash_reward x = 15) : x = 1 :=
by { sorry }

end milo_number_of_5s_l46_46625


namespace evaluate_expr_right_to_left_l46_46586

variable (a b c d : ℝ)

theorem evaluate_expr_right_to_left :
  (a - b * c + d) = a - b * (c + d) :=
sorry

end evaluate_expr_right_to_left_l46_46586


namespace sum_faces_edges_vertices_eq_26_l46_46341

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46341


namespace cos_theta_terminal_point_l46_46567

theorem cos_theta_terminal_point (θ : ℝ) (x y : ℝ) (h₁ : x = - real.sqrt 3) (h₂ : y = 1) (h₃ : real.sqrt (x^2 + y^2) = 2) : 
  real.cos θ = - real.sqrt 3 / 2 :=
sorry

end cos_theta_terminal_point_l46_46567


namespace incorrect_option_D_l46_46157

variable {p q : Prop}

theorem incorrect_option_D (hp : ¬p) (hq : q) : ¬(¬q) := 
by 
  sorry  

end incorrect_option_D_l46_46157


namespace inv_49_mod_102_l46_46512

theorem inv_49_mod_102 : (49 : ℤ)⁻¹ ≡ 67 [MOD 102] :=
by
  have h : (7 : ℤ)⁻¹ ≡ 55 [MOD 102] := sorry
  exact sorry

end inv_49_mod_102_l46_46512


namespace trapezoid_bc_length_l46_46084

theorem trapezoid_bc_length
  (A B C D M : Point)
  (d : Real)
  (h_trapezoid : IsTrapezoid A B C D)
  (h_M_on_AB : OnLine M A B)
  (h_DM_perp_AB : Perpendicular D M A B)
  (h_MC_eq_CD : Distance M C = Distance C D)
  (h_AD_eq_d : Distance A D = d) :
  Distance B C = d / 2 := by
  sorry

end trapezoid_bc_length_l46_46084


namespace geometric_series_common_ratio_l46_46434

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 500) (hS : S = 2500) (h_series : S = a / (1 - r)) : r = 4 / 5 :=
by
  sorry

end geometric_series_common_ratio_l46_46434


namespace sequence_a_10_eq_e_32_l46_46502

-- Define the sequence a_n according to the given condition
def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 →
  (∏ k in Finset.range n, ln (a (k+1)) / (3 * (k+1) - 1)) = (3 * n + 2) / 2

-- State the main theorem we need to prove
theorem sequence_a_10_eq_e_32 (a : ℕ → ℝ) (h : seq a) : a 10 = Real.exp 32 :=
  sorry

end sequence_a_10_eq_e_32_l46_46502


namespace largest_four_digit_sum_20_l46_46713

theorem largest_four_digit_sum_20 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n.digits 10).sum = 20 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m.digits 10).sum = 20 → n ≥ m :=
begin
  sorry
end

end largest_four_digit_sum_20_l46_46713


namespace fraction_B_A_C_l46_46770

theorem fraction_B_A_C (A B C : ℕ) (x : ℚ) 
  (h1 : A = (1 / 3) * (B + C)) 
  (h2 : A = B + 10) 
  (h3 : A + B + C = 360) : 
  x = 2 / 7 ∧ B = x * (A + C) :=
by
  sorry -- The proof steps can be filled in

end fraction_B_A_C_l46_46770


namespace right_triangle_OQ_OR_BC_l46_46831

noncomputable def ellipse (a b : ℝ) (h : a > b > 0) : set (ℝ × ℝ) :=
{ P | (P.1^2 / a^2) + (P.2^2 / b^2) = 1 }

theorem right_triangle_OQ_OR_BC (a b : ℝ) (h : a > b > 0)
    (P Q R : ℝ × ℝ)
    (hP : P ∈ ellipse a b h)
    (hQ : Q ∈ ellipse a b h)
    (hR : R ∈ ellipse a b h)
    (hPQparallel : Q.2 * P.1 = P.2 * Q.1)
    (M : ℝ × ℝ := ((P.1 + 0) / 2, (P.2 + 0) / 2))
    (hM : M = (M.1, M.2))
    (hOMR : ∃ m : ℝ, R = (m * M.1, m * M.2)) : 
    (∃ A B C D : ℝ × ℝ, 
        A = (-a, 0) ∧ B = (a, 0) ∧ C = (0, b) ∧ D = (0, -b) ∧ 
        (Q.1 - 0)^2 + (Q.2 - 0)^2 + (R.1 - 0)^2 + (R.2 - 0)^2 = b^2 + a^2) := 
begin
  sorry
end

end right_triangle_OQ_OR_BC_l46_46831


namespace remainder_of_modified_expression_l46_46144

theorem remainder_of_modified_expression (x y u v : ℕ) (h : x = u * y + v) (hy_pos : y > 0) (hv_bound : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y + 4) % y = v + 4 :=
by sorry

end remainder_of_modified_expression_l46_46144


namespace simplify_expression_l46_46267

-- Define the conditions as parameters
variable (x y : ℕ)

-- State the theorem with the required conditions and proof goal
theorem simplify_expression (hx : x = 2) (hy : y = 3) :
  (8 * x * y^2) / (6 * x^2 * y) = 2 := by
  -- We'll provide the outline and leave the proof as sorry
  sorry

end simplify_expression_l46_46267


namespace part_one_part_two_l46_46975

-- Definitions based on problem conditions
def y (x a : ℝ) : ℝ := x + a / x
def f (x c : ℝ) : ℝ := x + c / x

-- Prove the properties
theorem part_one (b : ℝ) :
  (∀ x : ℝ, x > 0 → (if 0 < x ∧ x ≤ 3 then y x (3^b) else if 3 ≤ x then y x (3^b)) ≤ (if (0 < x ∧ x ≤ 3) then y 3 (3^b) else if (3 ≤ x) then y 3 (3^b)))
  ↔ b = 2 := 
sorry

theorem part_two (c x : ℝ) (h1 : 1 <= x) (h2 : x <= 2) :
  (∀ c : ℝ, 1 <= c ∧ c <= 4 → 
    (f (sqrt c) c = 2 * sqrt c) ∧ 
    (if 1 <= c ∧ c <= 2 then f 2 c = 2 + c / 2 else if 2 < c ∧ c <= 4 then f 1 c = 1 + c)) :=
sorry

end part_one_part_two_l46_46975


namespace range_of_k_non_monotonic_l46_46999

def f (x : ℝ) : ℝ := x^3 - 12 * x

def is_monotonic (g : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → g x ≤ g y

theorem range_of_k_non_monotonic :
  let k : ℝ := sorry
  let I := set.Ioo (k - 1) (k + 1)
  ¬ is_monotonic f I →
  (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end range_of_k_non_monotonic_l46_46999


namespace people_cannot_take_bus_l46_46762

theorem people_cannot_take_bus 
  (carrying_capacity : ℕ) 
  (fraction_entered : ℚ) 
  (next_pickup : ℕ) 
  (carrying_capacity = 80) 
  (fraction_entered = 3 / 5) 
  (next_pickup = 50) : 
  let first_pickup := (fraction_entered * carrying_capacity : ℚ).to_nat in
  let available_seats := carrying_capacity - first_pickup in
  let cannot_board := next_pickup - available_seats in
  cannot_board = 18 :=
by 
  sorry

end people_cannot_take_bus_l46_46762


namespace equation_of_circle_C_range_of_AB_l46_46070

-- Definitions for the given conditions
def circle_C : Set (ℝ × ℝ) := {p | (p.1 + 1) ^ 2 + p.2 ^ 2 = 1}
def circle_D : Set (ℝ × ℝ) := {p | (p.1 - 4) ^ 2 + p.2 ^ 2 = 4}

-- The question to prove
theorem equation_of_circle_C :
  ∃ c : ℝ × ℝ, (c.1 = -1) ∧ (c.2 = 0) ∧ (circle_C (0, 0)) ∧ (circle_C (-1, 1)) := sorry

theorem range_of_AB :
  ∃ (a b : ℝ), (a ≠ b) ∧ ∀ P ∈ circle_D, ∃ (A B : ℝ), 
  (A = 0) ∧ (B = 0) ∧ (A ≠ B) ∧ (a ≤ abs (A - B) ∧ abs (A - B) ≤ b)
  ∧ (a = sqrt 2) ∧ (b = (5 * sqrt 2) / 4) := sorry

end equation_of_circle_C_range_of_AB_l46_46070


namespace max_T_l46_46536

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- Given conditions for the sequence {a_n}
def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ k : ℕ, 0 < k → (a k, a (k+1), a (k+2)).triangle_inequality)

-- Sum of sequence condition
def sum_gt (a : ℕ → ℝ) (T : ℝ) : Prop :=
  (∑ k in finset.range 2020, a (k.succ)) > T

-- Main theorem
theorem max_T (a : ℕ → ℝ) (h_seq : sequence a) :
  sum_gt a (1 / (fib 2018) * ∑ i in finset.range 2018, fib (i + 1)) :=
sorry

end max_T_l46_46536


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46367

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46367


namespace find_x_between_0_and_180_l46_46480

theorem find_x_between_0_and_180 :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 60 := by
  sorry

end find_x_between_0_and_180_l46_46480


namespace distance_between_planes_correct_l46_46474

-- Define the planes
def plane1 (x y z : ℝ) := x + 2 * y - 2 * z + 1
def plane2 (x y z : ℝ) := 2 * x + 5 * y - 4 * z + 8

-- Define the normal vectors of the planes
def normal_vector1 : ℝ × ℝ × ℝ := (1, 2, -2)
def normal_vector2 : ℝ × ℝ × ℝ := (2, 5, -4)

-- Define a point on the first plane
def point_on_plane1 : ℝ × ℝ × ℝ := (1, 0, 1)

-- The distance between the planes
noncomputable def distance_between_planes : ℝ :=
  let a := 2
  let x1 := point_on_plane1.1
  let b := 5
  let y1 := point_on_plane1.2
  let c := -4
  let z1 := point_on_plane1.3
  let d := 8
  (abs (a * x1 + b * y1 + c * z1 + d)) / (sqrt (a^2 + b^2 + c^2))

-- Theorem statement with the correct answer
theorem distance_between_planes_correct : distance_between_planes = (2 * sqrt 5) / 5 :=
by sorry

end distance_between_planes_correct_l46_46474


namespace positive_integer_solutions_x_plus_2y_eq_5_l46_46893

theorem positive_integer_solutions_x_plus_2y_eq_5 :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x + 2 * y = 5) ∧ ((x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 1)) :=
by
  sorry

end positive_integer_solutions_x_plus_2y_eq_5_l46_46893


namespace largest_four_digit_sum_20_l46_46709

theorem largest_four_digit_sum_20 : ∃ n : ℕ, (999 < n ∧ n < 10000 ∧ (sum (nat.digits 10 n) = 20 ∧ ∀ m, 999 < m ∧ m < 10000 ∧ sum (nat.digits 10 m) = 20 → m ≤ n)) :=
by
  sorry

end largest_four_digit_sum_20_l46_46709


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46346

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46346


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46347

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46347


namespace female_officers_on_police_force_l46_46248

theorem female_officers_on_police_force (on_duty_percent : ℝ) (on_duty_total : ℕ) (half_on_duty_female : ℕ)
    (h1 : on_duty_percent = 0.18)
    (h2 : on_duty_total = 180)
    (h3 : half_on_duty_female = on_duty_total / 2)
    (h4 : half_on_duty_female = 90) :
    (total_female_officers : ℕ) (H : 0.18 * total_female_officers = 90) : total_female_officers = 500 := 
begin
    sorry
end

end female_officers_on_police_force_l46_46248


namespace girls_dropped_out_l46_46306

theorem girls_dropped_out (B_initial G_initial B_dropped G_remaining S_remaining : ℕ)
  (hB_initial : B_initial = 14)
  (hG_initial : G_initial = 10)
  (hB_dropped : B_dropped = 4)
  (hS_remaining : S_remaining = 17)
  (hB_remaining : B_initial - B_dropped = B_remaining)
  (hG_remaining : G_remaining = S_remaining - B_remaining) :
  (G_initial - G_remaining) = 3 := 
by 
  sorry

end girls_dropped_out_l46_46306


namespace total_population_l46_46176

theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t :=
by
  -- We state our theorem including assumptions and goal
  sorry -- placeholder for the proof

end total_population_l46_46176


namespace proof_problem_l46_46846

def problem_statement := 
  let m : ℕ := 2022 in 
  ⌊ (2023^3 / (2021 * 2022) - (2021^3 / (2022 * 2023)) ) ⌋ = 8

theorem proof_problem : problem_statement :=
by sorry

end proof_problem_l46_46846


namespace sum_faces_edges_vertices_l46_46354

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46354


namespace difference_of_fractions_l46_46391

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h1 : a = 700) (h2 : b = 7) : a - b = 693 :=
by
  rw [h1, h2]
  norm_num

end difference_of_fractions_l46_46391


namespace probability_even_number_l46_46684

-- Definition of the problem, defining the set of cards and the conditions
def cards := {0, 1, 2, 3}

-- Calculate the probability that the formed two-digit number is even
theorem probability_even_number: 
  ∃ (n m : ℕ), 
  n = 9 ∧ m = 5 ∧ (m : ℚ) / (n : ℚ) = 5 / 9 := 
sorry

end probability_even_number_l46_46684


namespace general_term_formula_l46_46538

theorem general_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 3^n - 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n - 1)) →
  a 1 = 2 →
  ∀ n, a n = 2 * 3^(n - 1) :=
by
    intros hS ha h1 n
    sorry

end general_term_formula_l46_46538


namespace reduction_percentage_l46_46303

-- Variables and constants
variable (original_price reduction_amount : ℝ)
hypothesis h1 : original_price = 500
hypothesis h2 : reduction_amount = 150

-- Definition of the percentage reduction concept
def percent_reduction (original_price reduction_amount : ℝ) : ℝ := (reduction_amount / original_price) * 100

-- Goal statement
theorem reduction_percentage : percent_reduction original_price reduction_amount = 30 := by
  simp [percent_reduction, h1, h2]
  sorry

end reduction_percentage_l46_46303


namespace total_percent_decrease_cardA_l46_46690

noncomputable def cardA_initial_value : ℝ := 150
noncomputable def cardA_decrease_year1 : ℝ := 0.20
noncomputable def cardA_decrease_year2 : ℝ := 0.30
noncomputable def cardA_decrease_year3 : ℝ := 0.15

theorem total_percent_decrease_cardA : 
  let value_after_year1 := cardA_initial_value * (1 - cardA_decrease_year1),
      value_after_year2 := value_after_year1 * (1 - cardA_decrease_year2),
      value_after_year3 := value_after_year2 * (1 - cardA_decrease_year3),
      total_decrease := (cardA_initial_value - value_after_year3) / cardA_initial_value * 100 in
  total_decrease = 52.4 :=
by
  sorry

end total_percent_decrease_cardA_l46_46690


namespace arithmetic_prog_solution_l46_46921

-- Define the terms in the arithmetic progression
def arithmetic_prog (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

-- Define the largest term
def largest_term (a d : ℤ) : ℤ := arithmetic_prog a d 3

-- Define the condition that the largest term is the sum of the squares of the other three terms
def condition (a d : ℤ) : Prop :=
  largest_term a d = (arithmetic_prog a d 0)^2 + (arithmetic_prog a d 1)^2 + (arithmetic_prog a d 2)^2

-- Define the proof problem
theorem arithmetic_prog_solution :
  ∃ (a d : ℤ), condition a d ∧
  {arithmetic_prog a d 3, arithmetic_prog a d 2, arithmetic_prog a d 1, arithmetic_prog a d 0} = {2, 1, 0, -1} :=
sorry

end arithmetic_prog_solution_l46_46921


namespace saras_birdhouse_height_l46_46637

theorem saras_birdhouse_height :
  let jake_width_feet := 16 / 12 
  let jake_height_feet := 20 / 12
  let jake_depth_feet := 18 / 12
  let jake_volume_feet := jake_width_feet * jake_height_feet * jake_depth_feet
  let diff_volume_cubic_feet := 1152 / 1728
  ∃ h : ℝ, 
  (1 * h * 2) - jake_volume_feet = diff_volume_cubic_feet ∧
  h = 2 :=
begin
  sorry
end

end saras_birdhouse_height_l46_46637


namespace expression_evaluation_l46_46067

theorem expression_evaluation (x y z : ℝ) (h : x = y + z) (h' : x = 2) :
  x^3 + 2 * y^3 + 2 * z^3 + 6 * x * y * z = 24 :=
by
  sorry

end expression_evaluation_l46_46067


namespace number_of_ordered_pairs_l46_46857

theorem number_of_ordered_pairs :
  let p := { (m, n) : ℤ × ℤ | m * n ≥ 0 ∧ m^3 + n^3 + 105 * m * n = 35^3 } in
  p.card = 37 :=
by
  -- Proof goes here
  sorry

end number_of_ordered_pairs_l46_46857


namespace factor_exp_l46_46840

theorem factor_exp (k : ℕ) : 3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 :=
by
  intro h
  sorry

end factor_exp_l46_46840


namespace eval_floor_ceil_addition_l46_46466

theorem eval_floor_ceil_addition : ⌊-3.67⌋ + ⌈34.2⌉ = 31 := by
  -- Condition 1: Definition of floor function
  have h1 : ⌊-3.67⌋ = -4 := by sorry
  -- Condition 2: Definition of ceiling function
  have h2 : ⌈34.2⌉ = 35 := by sorry
  -- Combining the results
  calc
    ⌊-3.67⌋ + ⌈34.2⌉ = -4 + 35 : by rw [h1, h2]
                ... = 31 : by sorry

end eval_floor_ceil_addition_l46_46466


namespace log_diff_values_count_l46_46062

-- Define the set of numbers
def numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Define the predicate that checks if two numbers are different and belong to the set
def valid_pair (a b : ℕ) : Prop := a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b

-- Define the function that gives the difference of logarithms
noncomputable def log_diff (a b : ℕ) : ℝ := Real.log a - Real.log b

-- Define the main theorem statement
theorem log_diff_values_count : 
  ∃ count : ℕ, count = 18 ∧ 
  (count = (Set.toFinset {log_diff a b | a b ∈ numbers ∧ a ≠ b}).card) :=
sorry

end log_diff_values_count_l46_46062


namespace find_CF_length_l46_46184

-- Definitions of geometric entities and conditions
def right_angle_triangle (A B C : Type) (hypotenuse : A -> B -> C -> Prop) (right_angle : Prop) : Prop :=
  ∃ A B C : Type, hypotenuse A B C ∧ right_angle

def point_on_line (D : Type) (BC : Type) : Prop := ∃ D ∈ BC

def angle_eq_45 (A D C : Type) : Prop := ∃ angle, angle = 45

def line_perimeter_minimized (triangle : Type) (vertices : Type) (perimeter : ℝ) : Prop :=
  ∃ E ∈ triangle, perimeter = min_perimeter

def find_CF (A B C D E F : Type) (triangle : Type) : ℝ :=
  ∃ length_CF, length_CF = 3.6

-- Theorem to be proved
theorem find_CF_length :
  ∀ (A B C D E F : Type),
    right_angle_triangle A B C (triang ABC) (angle_eq ACB = 90) →
    triangle_side_length AC = 6 →
    triangle_side_length BC = 4 →
    point_on_line D (line BC) ∧ CD > BD →
    angle_eq_45 A D C →
    line_perimeter_minimized (triangle CBE) vertices ABC →
    line_perimeter_minimized (triangle AFE) vertices DEF →
    find_CF A B C D E F (triangle AFE) = 3.6 := sorry

end find_CF_length_l46_46184


namespace median_proof_l46_46991

variable (a : Int) (b : Real)

def conditions : Prop :=
  a ≠ 0 ∧ 0 < b ∧ a * (b ^ 2) = Real.exp b ∧ b < Real.exp b

def median : Real :=
  if conditions a b then 0.5 else sorry

theorem median_proof :
  conditions a b → median a b = 0.5 :=
by
  intro h
  have h1 : a ≠ 0 := h.1
  have h2 : 0 < b := h.2.1
  have h3 : a * (b ^ 2) = Real.exp b := h.2.2.1
  have h4 : b < Real.exp b := h.2.2.2
  sorry

end median_proof_l46_46991


namespace unique_intersection_points_l46_46670

theorem unique_intersection_points : ∀ x : ℝ, x > 0 → 
  (∃ y : ℝ, (y = log x 4 ∨ y = log 4 x ∨ y = log (1/4) x ∨ y = log x (1/4)) ∧
  ∀ z1 z2 : ℝ, (z1 = log 4 (exp z1) ∨ z1 = 4^(1 / z1) ∨ z1 = 4^(-1 / 2) ^ z1 ∨ z1 = (4^-1)^(1 / z1)) ∧
               (z2 = log 4 (exp z2) ∨ z2 = 4^(1 / z2) ∨ z2 = 4^(-1 / 2) ^ z2 ∨ z2 = (4^-1)^(1 / z2)) ∧
               (z1 = z2)) :=
sorry

end unique_intersection_points_l46_46670


namespace percentage_water_in_fresh_fruit_l46_46404

theorem percentage_water_in_fresh_fruit
  (P : ℝ)
  (weight_fresh : ℝ := 81.00000000000001)
  (weight_dried : ℝ := 9)
  (water_fraction_dried : ℝ := 0.19) :
  P = 0.02111111111111111 :=
by
  have eqn : P * weight_fresh = water_fraction_dried * weight_dried,
  sorry

end percentage_water_in_fresh_fruit_l46_46404


namespace sum_not_divisible_by_5_l46_46398

theorem sum_not_divisible_by_5 (n : ℕ) : 
  ¬(5 ∣ ∑ k in Finset.range (n + 1), (Nat.choose (2 * n + 1) (2 * k + 1) * 2 ^ (3 * k))) :=
sorry

end sum_not_divisible_by_5_l46_46398


namespace three_digit_numbers_sum_l46_46989

theorem three_digit_numbers_sum:
  let count_three_digit_numbers := 
    finset.sum (finset.range 10) (λ h, -- iterate over possible hundreds digit 1 to 9
      finset.sum (finset.range 10) (λ t, -- iterate over possible tens digit 0 to 9
        finset.sum (finset.range 10) (λ o, -- iterate over possible ones digit 0 to 9
          if (1 ≤ h) ∧ (h ≤ 9) ∧ (h > t + o) then 1 else 0 )
      )
    )
  in count_three_digit_numbers = 165 :=
by
  sorry

end three_digit_numbers_sum_l46_46989


namespace num_ordered_pairs_eq_three_l46_46856

theorem num_ordered_pairs_eq_three :
  {m n : ℕ // 0 < m ∧ 0 < n ∧ 
  (6 / m + 3 / n : ℝ) = 1}.card = 3 := 
begin
  sorry
end

end num_ordered_pairs_eq_three_l46_46856


namespace wilsons_theorem_l46_46234

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) : 
  (p = 2 ∨ Prime p) ↔ (factorial (p - 1) ≡ -1 [MOD p]) :=
sorry

end wilsons_theorem_l46_46234


namespace tetrahedron_cross_section_area_l46_46691

theorem tetrahedron_cross_section_area (a : ℝ) : 
  ∃ (S : ℝ), 
    let AB := a; 
    let AC := a;
    let AD := a;
    S = (3 * a^2) / 8 
    := sorry

end tetrahedron_cross_section_area_l46_46691


namespace sum_of_integer_solutions_l46_46720

theorem sum_of_integer_solutions (n_values : List ℤ) : 
  (∀ n ∈ n_values, ∃ (k : ℤ), 2 * n - 3 = k ∧ k ∣ 18) → (n_values.sum = 11) := 
by
  sorry

end sum_of_integer_solutions_l46_46720


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46351

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46351


namespace sum_of_fractions_l46_46479

theorem sum_of_fractions : 
  (∑ a in {a | ∃ (n: ℕ), a = n ∧ ¬ Divides 3 a ∧ 30 < a ∧ a < 300 }.to_finset, (a / 3 : ℚ)) = 9900 := 
by
  sorry

end sum_of_fractions_l46_46479


namespace area_LOM_l46_46577
open Real

noncomputable def area_triangle : ℝ → ℝ → ℝ → ℝ
| a, b, c => sqrt (s * (s - a) * (s - b) * (s - c))
  where s = (a + b + c) / 2

theorem area_LOM (α β γ : ℝ)
    (h1 : β = 2 * γ)
    (h2 : α = β - γ)
    (area_ABC : ℝ)
    (h_area_ABC : area_ABC = 8) :
    let LOM_area := 11
    (area L O M ≈ 11) := 
    sorry

end area_LOM_l46_46577


namespace coefficient_x_squared_l46_46281

def poly := (x + 1)^5 * (x - 2)

theorem coefficient_x_squared (x : ℝ) :
  polynomial.coeff (expand poly) 2 = -15 :=
begin
  sorry  -- proof is omitted
end

end coefficient_x_squared_l46_46281


namespace color_graph_with_lists_l46_46602
-- Import the entire Mathlib library to ensure all necessary functionalities are included.

-- Definitions of the problem conditions:
variables (V : Type) [fintype V] (H : simple_graph V)
variables (S : V → list ℕ)
variables (D : V → multiset V)
variables (d_plus : V → ℕ)
variables (kernel : set V)

-- Basic properties for conditions
variable [directed: ∀ v : V, d_plus v < (S v).length]
variable [kernel_condition: ∀ (E : set V), simple_graph.induced_subgraph H D E → ∃ (k : set V), kernel k]

-- Statement of the theorem: H can be colored using the lists S_v
theorem color_graph_with_lists 
  (H : simple_graph V)
  (S : V → list ℕ)
  (D : V → multiset V)
  (d_plus : V → ℕ)
  (kernel : set V)
  [directed: ∀ v : V, d_plus v < (S v).length]
  [kernel_condition: ∀ (E : set V), simple_graph.induced_subgraph H D E → ∃ (k : set V), kernel k] :
  ∃f : V → ℕ, ∀(v : V), f v ∈ S v := 
sorry

end color_graph_with_lists_l46_46602


namespace man_late_minutes_l46_46321

theorem man_late_minutes (v t t' : ℝ) (hv : v' = 3 / 4 * v) (ht : t = 2) (ht' : t' = 4 / 3 * t) :
  t' * 60 - t * 60 = 40 :=
by
  sorry

end man_late_minutes_l46_46321


namespace flour_percentage_remaining_l46_46900

theorem flour_percentage_remaining {initial_percent remaining_after_60 percent_taken_from_40 percent_remaining : ℝ} :
  initial_percent = 100 → remaining_after_60 = 40 → percent_taken_from_40 = 10 →
  percent_remaining = initial_percent - (initial_percent * 0.6 + remaining_after_60 * 0.25) →
  percent_remaining = 30 :=
by
  intros h_init h_rem60 h_10 h_final
  have h1 : remaining_after_60 = initial_percent - (initial_percent * 0.6), by assumption
  have h2 : percent_taken_from_40 = remaining_after_60 * 0.25, by assumption
  rw [h1, h2] at h_final
  exact h_final

end flour_percentage_remaining_l46_46900


namespace problem1_problem2_l46_46530

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - (1 / 2) * (x - a)^2 + 4

theorem problem1 (a : ℝ) : (∀ x : ℝ, Real.exp x - (x - a) ≥ 0) ↔ a ≥ -1 := 
sorry

theorem problem2 (a : ℝ) : (∀ x ≥ 0, f a x ≥ 0) ↔ ln 4 - 4 ≤ a ∧ a ≤ Real.sqrt 10 := 
sorry

end problem1_problem2_l46_46530


namespace rectangle_diagonals_equal_rhombus_not_l46_46304

/-- Define the properties for a rectangle -/
structure Rectangle :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- Define the properties for a rhombus -/
structure Rhombus :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- The property that distinguishes a rectangle from a rhombus is that the diagonals are equal. -/
theorem rectangle_diagonals_equal_rhombus_not
  (R : Rectangle)
  (H : Rhombus)
  (hR1 : R.sides_parallel)
  (hR2 : R.diagonals_equal)
  (hR3 : R.diagonals_bisect)
  (hR4 : R.angles_equal)
  (hH1 : H.sides_parallel)
  (hH2 : ¬H.diagonals_equal)
  (hH3 : H.diagonals_bisect)
  (hH4 : H.angles_equal) :
  (R.diagonals_equal) := by
  sorry

end rectangle_diagonals_equal_rhombus_not_l46_46304


namespace find_q_l46_46932

variable (p q : ℝ)
variable (a : ℂ)
variable (h : a = 1 + complex.I * real.sqrt 3)

theorem find_q (hp : real.of_complex (a * conj a) = q) : q = 4 := by
  sorry

end find_q_l46_46932


namespace triangle_DEF_area_l46_46588

theorem triangle_DEF_area {D E F R S T G H : Type}
    [EuclideanGeometry] (triangle_DEF : Triangle D E F)
    (R_mid_EF : Midpoint R E F) (S_mid_DF : Midpoint S D F) (T_mid_DF : Midpoint T D F)
    (G_cent : Centroid G D E F) (H_intersection : Intersection H (Line R T) (Line E S))
    (area_RGH : ℝ) (m : ℝ) : 
    (area (triangle R G H) = m) →
    (area (triangle D E F) = 12 * m) := by
  sorry

end triangle_DEF_area_l46_46588


namespace pyramid_height_proof_l46_46316

open Real

def right_pyramid (base_area : ℝ) (area1 : ℝ) (area2 : ℝ) (distance : ℝ) :=
  ∃ (h1 h_total : ℝ), 
    (area1 = 300 * sqrt 2) ∧ 
    (area2 = 675 * sqrt 2) ∧ 
    (distance = 10) ∧ 
    (base_area = 1200) ∧ 
    (h1 = 30) ∧ 
    (h_total = 40)

theorem pyramid_height_proof :
  right_pyramid 1200 (300 * sqrt 2) (675 * sqrt 2) 10 :=
  sorry

end pyramid_height_proof_l46_46316


namespace interval_satisfies_inequality_l46_46882

theorem interval_satisfies_inequality :
  { x : ℝ | x ∈ [-1, -1/3) ∪ (-1/3, 0) ∪ (0, 1) ∪ (1, ∞) } =
  { x : ℝ | x^2 + 2*x^3 - 3*x^4 ≠ 0 ∧ x + 2*x^2 - 3*x^3 ≠ 0 ∧ (x >= -1 ∧ (x < 1 ∨ x > -1/3)) ∧ 
            x^2 + 2*x^3 - 3*x^4 / (x + 2*x^2 - 3*x^3) ≥ -1 } := sorry

end interval_satisfies_inequality_l46_46882


namespace range_of_a_l46_46135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then -x^2 + a * x + a / 4 else a^x

theorem range_of_a (a : ℝ) (monotonic_f : ∀ x y, x ≤ y → f a x ≤ f a y) : a ∈ set.Icc 2 4 :=
sorry

end range_of_a_l46_46135


namespace power_function_at_4_l46_46139

noncomputable def power_function (x : ℝ) : ℝ := x^(1/2)

theorem power_function_at_4 :
  (power_function 2 = (sqrt 2)) →
  (power_function 4 = 2) :=
by
  intros h
  -- Omitted proof
  sorry

end power_function_at_4_l46_46139


namespace badminton_members_count_l46_46578

-- Definitions of the conditions
def total_members : ℕ := 40
def tennis_players : ℕ := 18
def neither_sport : ℕ := 5
def both_sports : ℕ := 3
def badminton_players : ℕ := 20 -- The answer we need to prove

-- The proof statement
theorem badminton_members_count :
  total_members = (badminton_players + tennis_players - both_sports) + neither_sport :=
by
  -- The proof is outlined here
  sorry

end badminton_members_count_l46_46578


namespace lcm_48_180_value_l46_46030

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l46_46030


namespace num_values_of_median_l46_46229

-- Define the set S with seven specified elements
def S : Set ℤ := {3, 5, 7, 8, 12, 15, 18}

-- Define a function to count the number of possible medians
noncomputable def num_possible_medians (S : Set ℤ) : ℕ :=
  if h : S.card = 7 then
    let additional_elements : ℤ → Prop := λ x, x ∉ S
    let all_elements := Finset.image id (Finset.filter additional_elements (Finset.range 100)) -- Assuming an upper limit of 100 for simplicity
    let all_possible_S := 
      (all_elements.powerset.filter (λ fs, fs.card = 4)).image (λ fs, fs ∪ S.to_finset)
    (all_possible_S.filter (λ fs, ((fs.sort Finset.preorder_le 6).nth (5)) ≠ none)).card
  else 0

-- The problem statement
theorem num_values_of_median : num_possible_medians S = 6 :=
sorry

end num_values_of_median_l46_46229


namespace incorrect_conclusion_l46_46282

theorem incorrect_conclusion
  (a b c : ℝ)
  (h₀ : c = 6)
  (h₁ : 4*a - 2*b + c = 0)
  (h₂ : a - b + c = 4)
  (h₃ : -a = 1) -- From solving, we get -a = 1 hence a = -1
  (h₄ : b = 1)  -- From solving equations
  (h₅ : ∀ x, -a * x ^ 2 + b * x + c = -x^2 + x + 6)
  : ¬ ∃ x, x^2 = 4 where x = 2 → -x^2 + x + c = 0 := 
sorry

end incorrect_conclusion_l46_46282


namespace ratio_BE_ED_l46_46692

variables {K : Type*} [Field K] [Invertible (4 : K)]
variables (A B C D E F G : K)

-- Given conditions
def is_parallelogram (A B C D : K) : Prop := sorry
def intersects (A B : K) (P : K) : Prop := sorry
def ratio (x y : K) (r : K) : Prop := x / y = r

-- Parallelogram ABCD and intersecting points with specified ratio
axiom parA (hA : is_parallelogram A B C D)
axiom intBD (hE : intersects A B E)
axiom intCD (hF : intersects A D F)
axiom intBC (hG : intersects A C G)
axiom ratioFG_FE (h_ratio : ratio F G 4)

-- Prove that the ratio BE:ED is sqrt(5)
theorem ratio_BE_ED : ratio E D (Real.sqrt 5) := 
  sorry

end ratio_BE_ED_l46_46692


namespace alex_birth_year_l46_46666

theorem alex_birth_year :
  (∃ y : ℕ, y = 1991) ∧
  (∀ n : ℕ, n > 0 → n + 1990 = y) ∧
  (∃ age : ℕ, age = 10) ∧
  (∃ n : ℕ, n = 9) ∧
  y = 1991 → 
  (∀ n : ℕ, n = 9 → y + n - 1 = 1999) →
  (∀ n : ℕ, n = 9 → age = 10 → y + n - 1 = 1999 → 1999 - age = 1989) →
  (∃ birth_year : ℕ, birth_year = 1989) :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  existsi 1989,
  split,
    { linarith, },
    { sorry, }
end

end alex_birth_year_l46_46666


namespace ratio_shorter_to_longer_l46_46754

-- Constants for the problem
def total_length : ℝ := 49
def shorter_piece_length : ℝ := 14

-- Definition of longer piece length based on the given conditions
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- The theorem to be proved
theorem ratio_shorter_to_longer : 
  shorter_piece_length / longer_piece_length = 2 / 5 :=
by
  -- This is where the proof would go
  sorry

end ratio_shorter_to_longer_l46_46754


namespace max_sin_sum_max_sin_sum_achievable_l46_46822

theorem max_sin_sum (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (hABC : A + B + C = π) :
  (sin A + sin B * sin C) ≤ (1 + sqrt 5) / 2 :=
sorry

theorem max_sin_sum_achievable (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (hABC : A + B + C = π)
  (hAchieve : A = π/2 - θ ∧ B = (π - A) / 2 ∧ C = (π - A) / 2)
  (θ : ℝ) (hCosTheta : cos θ = 2 / sqrt 5) :
  (sin A + sin B * sin C) = (1 + sqrt 5) / 2 :=
sorry

end max_sin_sum_max_sin_sum_achievable_l46_46822


namespace intersection_A_B_l46_46927

def A : set ℝ := {x | x^2 - x - 6 < 0}
def B : set ℝ := {x | x^2 + 2x - 8 > 0}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∈ 3} :=
sorry

end intersection_A_B_l46_46927


namespace find_wrongly_written_height_l46_46278

variable (n : ℕ := 35)
variable (average_height_incorrect : ℚ := 184)
variable (actual_height_one_boy : ℚ := 106)
variable (actual_average_height : ℚ := 182)
variable (x : ℚ)

theorem find_wrongly_written_height
  (h_incorrect_total : n * average_height_incorrect = 6440)
  (h_correct_total : n * actual_average_height = 6370) :
  6440 - x + actual_height_one_boy = 6370 ↔ x = 176 := by
  sorry

end find_wrongly_written_height_l46_46278


namespace g_symmetry_l46_46944

def g (x : ℝ) : ℝ := log x / log 3  -- Define the inverse of 3^x, i.e., log base 3 of x

theorem g_symmetry (h : ∀ x, g (3^x) = x) : g 2 = log 2 / log 3 :=
by sorry

end g_symmetry_l46_46944


namespace henry_age_sum_of_digits_next_multiple_l46_46146

-- Definitions used in conditions
variables {J H M n : ℕ} (t : ℕ) -- t is for today's ages

-- Conditions
def mike_age_today (t : ℕ) : Prop := M = 2
def age_relationship (t : ℕ) : Prop := H = J + 4
def henry_double_julia (t : ℕ) : Prop := ∀ (t' : ℕ), H = 2 * J := t'
def julia_today (t : ℕ) : Prop := J = 4
def henry_today (t : ℕ) : Prop := H = 8

-- Question: Sum of the digits of Henry's age when it's a multiple of Mike's age
def next_multiple_of_mike : ℕ := 6

theorem henry_age_sum_of_digits_next_multiple :
  mike_age_today t → age_relationship t → henry_double_julia t → julia_today t → 
  henry_today t → 
  (∃ n : ℕ, H + n = next_multiple_of_mike * (M + n) ∧ ∑_digits (H + n) = 5) := 
sorry

end henry_age_sum_of_digits_next_multiple_l46_46146


namespace circumcircle_through_fixed_point_l46_46072

open Affine

variables {P : Type*} [affine_space P ℝ]

def is_cyclic (A B C D : P) : Prop :=
∃ (O : P) (R : ℝ), ∀ (X : P), X ∈ {A, B, C, D} → dist O X = R

-- Define the assumptions
variables (A B C D E F P Q R : P)
variables (BC AD : P → P → ℝ) -- Distance functions
hypothesis (h_convex : convex ℝ ({A, B, C, D} : set P))
hypothesis (h_eq_dist : BC B C = AD A D)
hypothesis (h_not_parallel : ¬ parallel (Q - '\R), (C - 'D))
hypothesis (E_on_BC : E ∈ open_segment ℝ B C)
hypothesis (F_on_AD : F ∈ open_segment ℝ A D)
hypothesis (BE_eq_DF : BC B E = AD D F)
hypothesis (AC_BD_inter_P : ∃ P : P, line_through A C ∩ line_through B D = {P})
hypothesis (BD_EF_inter_Q : ∃ Q : P, line_through B D ∩ line_through E F = {Q})
hypothesis (EF_AC_inter_R : ∃ R : P, line_through E F ∩ line_through A C = {R})

-- Prove the question
theorem circumcircle_through_fixed_point :
  ∀ E F, E ∈ open_segment ℝ B C → F ∈ open_segment ℝ A D → BE_eq_DF → is_cyclic P Q R R :=
sorry

end circumcircle_through_fixed_point_l46_46072


namespace internal_angle_of_equilateral_triangle_l46_46508

open_locale euclidean_geometry

noncomputable def vectors_add_to_zero (A B C O : Point) [euclidean_space ℝ] : Prop :=
  ((O -ᵥ A) + (O -ᵥ B) + (O -ᵥ C) = 0)

noncomputable def center_of_circumcircle (A B C O : Point) [euclidean_space ℝ] : Prop :=
  (∃ (circle : Circle), circle.center = O ∧ circle.radius = dist O A ∧ dist O A = dist O B ∧ dist O B = dist O C)

theorem internal_angle_of_equilateral_triangle (A B C O : Point) [euclidean_space ℝ]
  (H1 : center_of_circumcircle A B C O)
  (H2 : vectors_add_to_zero A B C O) :
  ∠A = 60 :=
sorry

end internal_angle_of_equilateral_triangle_l46_46508


namespace solution_exists_l46_46858

variable (x y : ℝ)

noncomputable def condition (x y : ℝ) : Prop :=
  (3 + 5 * x = -4 + 6 * y) ∧ (2 + (-6) * x = 6 + 8 * y)

theorem solution_exists : ∃ (x y : ℝ), condition x y ∧ x = -20 / 19 ∧ y = 11 / 38 := 
  by
  sorry

end solution_exists_l46_46858


namespace population_scientific_notation_l46_46627

noncomputable def round_to_nearest_thousand (n : ℝ) : ℕ :=
  Int.toNat (Real.round (n / 1000) * 1000)

theorem population_scientific_notation :
  (round_to_nearest_thousand 322819 = 323000) → (322819 : ℝ) = 3.23 * 10^5 :=
by
  sorry

end population_scientific_notation_l46_46627


namespace per_capita_GDP_exceeds_16000_in_5_years_l46_46171

theorem per_capita_GDP_exceeds_16000_in_5_years:
  ∃ n: ℕ, (8000 * (1 + 0.1)^n > 16000) ∧ (∀ m: ℕ, 8000 * (1 + 0.1)^m > 16000 → n ≤ m) → n = 5 := sorry

end per_capita_GDP_exceeds_16000_in_5_years_l46_46171


namespace num_elements_in_S_l46_46226

noncomputable def f (x : ℝ) : ℝ := (x + 6) / x

def f_seq : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := f ∘ f_seq n

def S : Set ℝ := {x | ∃ n > 0, f_seq n x = x}

theorem num_elements_in_S : S.to_finset.card = 2 := sorry

end num_elements_in_S_l46_46226


namespace palindrome_count_300_to_999_l46_46147

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

noncomputable def count_palindromes (lo hi : ℕ) : ℕ :=
  (List.range' lo (hi - lo + 1)).filter is_palindrome |>.length

theorem palindrome_count_300_to_999 : count_palindromes 300 999 = 70 :=
  sorry

end palindrome_count_300_to_999_l46_46147


namespace find_pairs_l46_46471

open Nat

-- m and n are odd natural numbers greater than 2009
def is_odd_gt_2009 (x : ℕ) : Prop := (x % 2 = 1) ∧ (x > 2009)

-- condition: m divides n^2 + 8
def divides_m_n_squared_plus_8 (m n : ℕ) : Prop := m ∣ (n ^ 2 + 8)

-- condition: n divides m^2 + 8
def divides_n_m_squared_plus_8 (m n : ℕ) : Prop := n ∣ (m ^ 2 + 8)

-- Final statement
theorem find_pairs :
  ∃ m n : ℕ, is_odd_gt_2009 m ∧ is_odd_gt_2009 n ∧ divides_m_n_squared_plus_8 m n ∧ divides_n_m_squared_plus_8 m n ∧ ((m, n) = (881, 89) ∨ (m, n) = (3303, 567)) :=
sorry

end find_pairs_l46_46471


namespace train_crosses_signal_pole_in_18_sec_l46_46753

theorem train_crosses_signal_pole_in_18_sec
  (length_train : ℝ) (time_to_cross_platform : ℝ) (length_platform : ℝ) : 
  length_train = 300 ∧ 
  time_to_cross_platform = 27 ∧ 
  length_platform = 150 → 
  let speed := (length_train + length_platform) / time_to_cross_platform in
  let time_to_cross_signal_pole := length_train / speed in
  time_to_cross_signal_pole = 18 := by
  intros h
  have h_lt : length_train = 300 := h.1
  have h_tp : time_to_cross_platform = 27 := h.2.1
  have h_lp : length_platform = 150 := h.2.2
  let speed := (300 + 150) / 27
  have speed_eq : speed = 450 / 27 := rfl
  let time_to_cross_signal_pole := 300 / speed
  have t_eq : time_to_cross_signal_pole = 300 / (450 / 27) := by rw speed_eq
  have t_eq_18 : time_to_cross_signal_pole = 18 := by
    rw t_eq
    norm_num
  exact t_eq_18

end train_crosses_signal_pole_in_18_sec_l46_46753


namespace lcm_48_180_value_l46_46032

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l46_46032


namespace f_abs_x_is_even_l46_46515

variables {R : Type*} [CommRing R]

def is_odd (f : R → R) : Prop :=
∀ x, f (-x) = -f x

def is_even (f : R → R) : Prop :=
∀ x, f (-x) = f x

-- Given:
variables (f g : R → R)
hypothesis hf : is_odd f
hypothesis hg : is_even g

-- Conclusion:
theorem f_abs_x_is_even : is_even (λ x, f (|x|)) :=
sorry

end f_abs_x_is_even_l46_46515


namespace perimeter_of_semicircle_is_approx_33_93_l46_46740

noncomputable def semicircle_perimeter (r : ℝ) : ℝ :=
  let half_circumference := π * r
  let diameter := 2 * r
  half_circumference + diameter

theorem perimeter_of_semicircle_is_approx_33_93 :
  semicircle_perimeter 6.6 ≈ 33.93 := by
  sorry

end perimeter_of_semicircle_is_approx_33_93_l46_46740


namespace verify_area_of_triangleABC_l46_46570

noncomputable def area_of_triangleABC  (B C A D E F: Type) [has_add A] [has_mul A] [has_one B]: Type :=
  ∃ (triangle ABC : Type) (BC: B) (AC: B) (AD: B) (DEF: B) (k: A) [add_comm_group A] 
  [vector_space A B] [division_ring B],
  midpoint BC D →
  (∀ A E C, AE / EC = 2 / 3) →
  (∀ A F D, AF / FD = 2 / 1) →
  (∀ Δ DEF, area Δ DEF = 10) →
  (∀ A B C, area Δ ABC = 150)
  

theorem verify_area_of_triangleABC : 
  area_of_triangleABC A B C D E F :=
sorry

end verify_area_of_triangleABC_l46_46570


namespace factorial_division_l46_46456

theorem factorial_division :
  11! / 9! = 110 := sorry

end factorial_division_l46_46456


namespace semicircle_area_percentage_difference_l46_46789

-- Define the rectangle dimensions
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 8

-- Define the diameters and radii of the semicircles
def large_semicircle_radius : ℝ := rectangle_length / 2
def small_semicircle_radius : ℝ := rectangle_width / 2

-- Define the areas of the full circles made from the semicircles
def large_circle_area : ℝ := real.pi * (large_semicircle_radius ^ 2)
def small_circle_area : ℝ := real.pi * (small_semicircle_radius ^ 2)

-- Define the percentage larger question
def percent_larger (a b : ℝ) : ℝ := ((a - b) / b) * 100

-- Formal proof statement
theorem semicircle_area_percentage_difference : 
  percent_larger large_circle_area small_circle_area = 125 := 
by
  sorry

end semicircle_area_percentage_difference_l46_46789


namespace cos_45_degree_l46_46454

theorem cos_45_degree : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degree_l46_46454


namespace solve_speed_of_second_train_l46_46318

open Real

noncomputable def speed_of_second_train
  (L1 : ℝ) (L2 : ℝ) (S1 : ℝ) (T : ℝ) : ℝ :=
  let D := (L1 + L2) / 1000   -- Total distance in kilometers
  let H := T / 3600           -- Time in hours
  let relative_speed := D / H -- Relative speed in km/h
  relative_speed - S1         -- Speed of the second train

theorem solve_speed_of_second_train :
  speed_of_second_train 100 220 42 15.99872010239181 = 30 := by
  sorry

end solve_speed_of_second_train_l46_46318


namespace trapezoid_bc_length_l46_46085

theorem trapezoid_bc_length
  (A B C D M : Point)
  (d : Real)
  (h_trapezoid : IsTrapezoid A B C D)
  (h_M_on_AB : OnLine M A B)
  (h_DM_perp_AB : Perpendicular D M A B)
  (h_MC_eq_CD : Distance M C = Distance C D)
  (h_AD_eq_d : Distance A D = d) :
  Distance B C = d / 2 := by
  sorry

end trapezoid_bc_length_l46_46085


namespace gangster_movement_speed_l46_46630

theorem gangster_movement_speed
  (a v : ℝ) (h1 : 0 < a) (h2 : 0 < v) :
  ∃ (g_speed : ℝ),
    (g_speed = 2 * v ∨ g_speed = v / 2) ∧ 
    gangster_remains_unnoticed a v g_speed :=
sorry

end gangster_movement_speed_l46_46630


namespace max_value_cauchy_schwarz_l46_46563

theorem max_value_cauchy_schwarz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : 
  x + 2 * y + 3 * z ≤ 3 * Real.sqrt 14 :=
begin
  sorry   -- proof omitted
end

end max_value_cauchy_schwarz_l46_46563


namespace area_increase_l46_46800

-- Defining the shapes and areas
def radius_large_side := 6
def radius_small_side := 4

def area_large_semicircles : ℝ := real.pi * (radius_large_side^2)
def area_small_semicircles : ℝ := real.pi * (radius_small_side^2)

-- The theorem statement
theorem area_increase : (area_large_semicircles / area_small_semicircles) = 2.25 → 
                         ((2.25 - 1) * 100) = 125 :=
by sorry

end area_increase_l46_46800


namespace correct_option_A_l46_46729

theorem correct_option_A : (sqrt (18) / sqrt (2) = 3) ∧ 
                           (sqrt (2) + sqrt (3) ≠ sqrt (5)) ∧ 
                           (sqrt (3) * 3 * sqrt (3) ≠ 6) ∧ 
                           (sqrt (18) - sqrt (12) ≠ sqrt (6)) :=
by {
  repeat {
    split,
    {
      rw [←sqrt_div, div_self],
      norm_num, exact zero_lt_two,
    },
    exact fun h => by linarith,
    exact fun h => by linarith,
    exact fun h => by linarith,
  }
}  

#eval correct_option_A

end correct_option_A_l46_46729


namespace floor_ceil_sum_l46_46467

theorem floor_ceil_sum : (Int.floor (-3.67) + Int.ceil (34.2) = 31) := 
by
  sorry

end floor_ceil_sum_l46_46467


namespace lcm_48_180_value_l46_46031

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l46_46031


namespace jogging_problem_l46_46697

theorem jogging_problem (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : ¬ ∃ p : ℕ, Prime p ∧ p^2 ∣ z) : 
  (x - y * Real.sqrt z) = 60 - 30 * Real.sqrt 2 → x + y + z = 92 :=
by
  intro h5
  have h6 : (60 - (60 - 30 * Real.sqrt 2))^2 = 1800 :=
    by sorry
  sorry

end jogging_problem_l46_46697


namespace simplify_f_increasing_intervals_l46_46525

-- Step 1: Define the given function f(x).
def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

-- Step 2: Prove the simplification.
theorem simplify_f : f x = 2 * Real.sin (2 * x + π / 6) :=
by
  sorry

-- Step 3: Prove the intervals where the function is increasing.
theorem increasing_intervals (k : ℤ) (x : ℝ) :
    (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) ↔ Function.increasing_on f (Icc (k * π - π / 3) (k * π + π / 6)) :=
by
  sorry

end simplify_f_increasing_intervals_l46_46525


namespace optimal_inscribed_square_area_l46_46190

theorem optimal_inscribed_square_area (ABC : Triangle) 
  (isosceles_right_triangle : isosceles_right ABC)
  (inscribed_square_area_1 : ∃ (s1 : ℝ), s1^2 = 400 ∧ inscribed_along_hypotenuse ABC s1)
  (inscribed_optimal_square : ∃ (s2 : ℝ), inscribed_touching_leg_and_hypotenuse ABC s2) :
  ∃ (a : ℝ), a = (40/3) ∧ a^2 = (1600/9) :=
sorry

end optimal_inscribed_square_area_l46_46190


namespace enclosed_area_of_laser_beam_path_l46_46919

theorem enclosed_area_of_laser_beam_path (square : Type) [metric_space square] 
  (A B C D M N : square) (h_square : is_square A B C D)
  (AB_length : dist A B = 1)
  (h_M : midpoint A B M)
  (h_N : midpoint B C N) :
  let enclosed_area := calculate_laser_path_area M N (sides := [BC, CD, DA, AB]) in
  enclosed_area = 1 / 2 :=
sorry

end enclosed_area_of_laser_beam_path_l46_46919


namespace find_ellipse_equation_max_area_difference_l46_46096

section EllipseProofs

variables {a b c x y e : ℝ}

/-- Conditions for the given ellipse problem -/
def is_ellipse (a b : ℝ) := a > b > 0 ∧ e = 1 / 2

/-- Statement I: Prove the equation of the ellipse M -/
theorem find_ellipse_equation (a b : ℝ) (h : is_ellipse a b) (h_focus : c = 1) (h_eccentricity : e = 1/2) :
  (a = 2 ∧ b ^ 2 = a ^ 2 - c ^ 2 ∧ b = sqrt 3) →
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1) :=
begin
  sorry
end

/-- Statement II: Prove the maximum value of |S1 - S2| and the equation of line l -/
theorem max_area_difference (a b : ℝ) (h : is_ellipse a b) (h_focus : c = 1) (h_eccentricity : e = 1/2) :
  (∀ m : ℝ, let y1 := (6 * m) / (3 * m^2 + 4), 
                y2 := -9 / (3 * m^2 + 4) in
                (|S1 - S2| = 12 / sqrt(3)) → 
                (l = sqrt(3)x + 2y + sqrt(3) = 0 ∨ l = sqrt(3)x - 2y + sqrt(3) = 0)) :=
begin
  sorry
end

end EllipseProofs

end find_ellipse_equation_max_area_difference_l46_46096


namespace ratio_correctness_l46_46283

noncomputable def ratio_spent_on_food_vs_fuel (fuel_cost : ℕ) (distance_per_tank : ℕ) (total_distance : ℕ) (total_spent : ℕ) : ℕ × ℕ :=
let tanks_required := total_distance / distance_per_tank in
let total_fuel_cost := tanks_required * fuel_cost in
let amount_spent_on_food := total_spent - total_fuel_cost in
let gcd_of_values := Int.gcd amount_spent_on_food total_fuel_cost in
(amount_spent_on_food / gcd_of_values, total_fuel_cost / gcd_of_values)

theorem ratio_correctness : ratio_spent_on_food_vs_fuel 45 500 2000 288 = (3, 5) :=
by
    let fuel_cost := 45
    let distance_per_tank := 500
    let total_distance := 2000
    let total_spent := 288
    let tanks_required := total_distance / distance_per_tank
    let total_fuel_cost := tanks_required * fuel_cost
    let amount_spent_on_food := total_spent - total_fuel_cost
    let gcd_of_values := Int.gcd amount_spent_on_food total_fuel_cost
    have : ratio_spent_on_food_vs_fuel 45 500 2000 288 = (amount_spent_on_food / gcd_of_values, total_fuel_cost / gcd_of_values) := rfl
    show (amount_spent_on_food / gcd_of_values, total_fuel_cost / gcd_of_values) = (3, 5) from sorry

end ratio_correctness_l46_46283


namespace score_for_june_is_86_l46_46655

theorem score_for_june_is_86
  (A_avg : ℕ)
  (AMJ_avg : ℕ)
  (h1 : A_avg = 89)
  (h2 : AMJ_avg = 88) :
  let A_sum := A_avg * 2 in
  let AMJ_sum := AMJ_avg * 3 in
  AMJ_sum - A_sum = 86 :=
by
  sorry

end score_for_june_is_86_l46_46655


namespace sum_faces_edges_vertices_eq_26_l46_46336

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46336


namespace sin_double_angle_l46_46929

theorem sin_double_angle (α : ℝ) (h1 : sin (π / 2 - α) = -3 / 5) (h2 : 0 < α ∧ α < π) : 
  sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l46_46929


namespace root_equation_a_gt_2sqrt2_is_largest_root_question_l46_46607

noncomputable def T : ℕ → ℤ
| 0     := 3
| 1     := 3
| 2     := 9
| (n+3) := 3 * T (n+2) - T n

def is_divisible_by_17 (n : ℕ) : Prop := ∃ k : ℤ, n = 17 * k

theorem root_equation_a_gt_2sqrt2_is_largest_root :
  ∀ (x : ℝ), x ^ 3 - 3 * x ^ 2 + 1 = 0 → x > 2 * Real.sqrt 2 → a = x := sorry

theorem question :
  let a := largest_root_of_cubic 1 w with a := root_equation_a_gt_2sqrt2_is_largest_root x in
  is_divisible_by_17 (T 1788 - 1) ∧ is_divisible_by_17 (T 1988 - 1) :=
begin
  sorry
end

end root_equation_a_gt_2sqrt2_is_largest_root_question_l46_46607


namespace lcm_48_180_l46_46042

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l46_46042


namespace at_most_two_points_l46_46505

variables {R : Type*} [ordered_ring R] 

structure Sphere (R : Type*) [ordered_ring R] :=
(center : R^3)
(radius : R)

def on_sphere (s : Sphere R) (P : R^3) :=
  dist P s.center = s.radius

variables {A B C D P : R^3}

def second_intersection (s : Sphere R) (P Q : R^3) : R^3 :=
  let line := line_through P Q
  (line.point_at (2 * (s.radius / dist P s.center)))

def tetrahedron_equilateral (A_Q B_Q C_Q D_Q : R^3) : Prop :=
  dist A_Q B_Q = dist B_Q C_Q ∧
  dist B_Q C_Q = dist C_Q D_Q ∧
  dist C_Q D_Q = dist D_Q A_Q

theorem at_most_two_points {s : Sphere R}
    (hA : on_sphere s A) (hB : on_sphere s B) (hC : on_sphere s C) (hD : on_sphere s D)
    (h_not_coplanar : ¬coplanar {A, B, C, D}) :
  ∃ at_most_two_Q : set R^3, ∀ Q, Q ∈ at_most_two_Q ↔
    tetrahedron_equilateral (second_intersection s A Q) (second_intersection s B Q)
    (second_intersection s C Q) (second_intersection s D Q) := sorry

end at_most_two_points_l46_46505


namespace opposite_of_2023_l46_46732

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l46_46732


namespace equation_1_solution_equation_2_solution_l46_46647

theorem equation_1_solution (x : ℝ) : 
  x * (x + 2) = 2 * x + 4 → (x = -2 ∨ x = 2) :=
by 
sory

theorem equation_2_solution (x : ℝ) : 
  3 * x^2 - x - 2 = 0 → (x = 1 ∨ x = -(2/3)) :=
by 
sory

end equation_1_solution_equation_2_solution_l46_46647


namespace length_YW_l46_46211

-- Define the points X, Y, Z, N, B, W
variable (X Y Z N B W : Type)
variable (coord : X → Y → Z → N → B → W → ℝ × ℝ)

-- Define conditions
def is_angle_YXZ_60 (X Y Z : X) : Prop :=
∠YXZ = 60

def is_angle_XYZ_30 (X Y Z : X) : Prop :=
∠XYZ = 30

def distance_XZ_1 (X Z : X) : Prop :=
dist X Z = 1

def midpoint_N_XZ (N X Z : X) : Prop :=
N = midpoint X Z

def perpendicular_XB_ZN (X B N Z : Type) : Prop :=
XB ⊥ ZN

def extended_YZ_BW_WX (Y Z W B X : Type) : Prop :=
YZ □ Z W = BW = WX

-- Define the theorem to prove
theorem length_YW (X Y Z N B W : Type) 
(coord : X → Y → Z → N → B → W → ℝ × ℝ) 
[is_angle_YXZ_60 X Y Z] [is_angle_XYZ_30 X Y Z] 
[distance_XZ_1 X Z] [midpoint_N_XZ N X Z] 
[perpendicular_XB_ZN X B N Z] [extended_YZ_BW_WX Y Z W B X] : 
  dist Y W = sqrt(3) / 4 :=
sorry

end length_YW_l46_46211


namespace tan_A_in_triangle_l46_46212

noncomputable def triangle (a b c A B C : ℝ) :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A + B + C = π

theorem tan_A_in_triangle (a b c A B C : ℝ)
  (h_triangle : triangle a b c A B C)
  (h1 : a / b = (b + sqrt 3 * c) / a)
  (h2 : sin C = 2 * sqrt 3 * sin B) :
  tan A = sqrt 3 / 3 :=
  sorry

end tan_A_in_triangle_l46_46212


namespace rectangular_prism_faces_edges_vertices_sum_l46_46330

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46330


namespace problem_l46_46599

variable (A : set ℝ)

axiom cond1 : (1 : ℝ) ∈ A
axiom cond2 : ∀ x : ℝ, x ∈ A → x^2 ∈ A
axiom cond3 : ∀ x : ℝ, x^2 - 4 * x + 4 ∈ A → x ∈ A

theorem problem : (2000 + Real.sqrt 2001) ∈ A :=
by
  sorry

end problem_l46_46599


namespace spaceship_finds_alien_l46_46823

-- Definitions of the conditions
variables (u v : ℝ) 

-- Condition that the spaceship's speed is greater than 10 times the alien's speed
def spaceship_faster_than_alien (u v : ℝ) : Prop := v > 10 * u

-- Main statement to be proven
theorem spaceship_finds_alien (u v : ℝ) (h : spaceship_faster_than_alien u v) : 
  ∃ (strategy : ℝ → ℝ → ℝ → Prop), ∀ (alien_path : ℝ → ℝ → Prop), strategy u v alien_path :=
sorry

end spaceship_finds_alien_l46_46823


namespace floor_sqrt_eq_floor_min_l46_46232

theorem floor_sqrt_eq_floor_min (n : ℕ) (h : 0 < n) : 
  (⌊Real.sqrt (4 * n + 1)⌋ : ℤ) = ⌊⇑(Finset.inf (Finset.image (λ k : ℕ, k + n / k) (Finset.range (n + 1)) : ℝ))⌋ := by
  sorry

end floor_sqrt_eq_floor_min_l46_46232


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46345

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46345


namespace prob_interval_0_1_l46_46576

-- Define the random variable ξ following normal distribution N(1, σ²)
noncomputable def xi (σ : ℝ) (hσ : σ > 0) : MeasureTheory.Measure ℝ :=
  MeasureTheory.Measure.normdist 1 σ

-- Given conditions
variables (σ : ℝ) (hσ : σ > 0) 
variable h1 : MeasureTheory.Measure.probability (xi σ hσ) (Set.Icc 0 2) = 0.6

-- Objective: Prove the probability for interval (0, 1)
theorem prob_interval_0_1 :
  MeasureTheory.Measure.probability (xi σ hσ) (Set.Icc 0 1) = 0.3 :=
by
  -- Due to the symmetry of the normal distribution around the mean (μ = 1),
  -- and since the given probability for (0, 2) is 0.6,
  -- by symmetry, (0, 1) and (1, 2) should both be half of 0.6.
  sorry

end prob_interval_0_1_l46_46576


namespace trapezoid_length_property_l46_46311

noncomputable def trapezoid_properties (A B C D X : ℝ) :=
  -- Defining the conditions
  let α₁ := 6 * π / 180 -- ∠DAB in radians
  let α₂ := 42 * π / 180 -- ∠ABC in radians
  let α₃ := 78 * π / 180 -- ∠AXD in radians
  let α₄ := 66 * π / 180 -- ∠CXB in radians
  let h := 1 -- Distance between AB and CD 
  ∃ (AD DX BC CX : ℝ),
    (AD = 1 / (Real.sin α₁)) ∧ 
    (DX = 1 / (Real.sin α₃)) ∧
    (BC = 1 / (Real.sin α₂)) ∧
    (CX = 1 / (Real.sin α₄)) ∧
    (AD + DX - (BC + CX) = 8)

-- Final theorem statement
theorem trapezoid_length_property : ∀ (A B C D X : ℝ), 
  trapezoid_properties A B C D X :=
begin
  intros A B C D X,
  unfold trapezoid_properties,
  -- Proof skipped
  sorry
end

end trapezoid_length_property_l46_46311


namespace find_M_l46_46483

theorem find_M (M : ℕ) : (∃ M, 8! * 10! = 20 * M!) → M = 13 :=
by
  sorry

end find_M_l46_46483


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46362

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46362


namespace apple_cost_l46_46819

theorem apple_cost
  (price_of_orange : ℝ)
  (num_apples : ℕ)
  (num_oranges : ℕ)
  (total_cost : ℝ) :
  price_of_orange = 2 →
  num_oranges = 2 →
  num_apples = 5 →
  total_cost = 9 →
  ∃ price_of_apple : ℝ, 5 * price_of_apple + 2 * price_of_orange = 9 ∧ price_of_apple = 1 :=
by
  intros h_orange h_num_oranges h_num_apples h_total_cost
  use 1
  split
  · rw [h_num_apples, h_num_oranges, h_orange]
    norm_num
  · rfl

-- Proof is omitted by using "sorry"

end apple_cost_l46_46819


namespace cos_45_degree_l46_46453

theorem cos_45_degree : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degree_l46_46453


namespace B_subset_A_l46_46601

noncomputable def A : set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : set ℝ := {x | a * x - 2 = 0}

theorem B_subset_A {a : ℝ} : B a ⊆ A ↔ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end B_subset_A_l46_46601


namespace product_value_l46_46723

theorem product_value :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
    -- Skipping the actual proof
    sorry

end product_value_l46_46723


namespace monotonicity_case1_monotonicity_case2_lower_bound_l46_46971

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 (a : ℝ) (h : a ≤ 0) : 
  ∀ x y : ℝ, x < y → f a x > f a y := 
by
  sorry

theorem monotonicity_case2 (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, x < Real.log (1 / a) → f a x > f a (Real.log (1 / a)) ∧ 
           x > Real.log (1 / a) → f a x < f a (Real.log (1 / a)) := 
by
  sorry

theorem lower_bound (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ,  f a x > 2 * Real.log a + 3 / 2 := 
by
  sorry

end monotonicity_case1_monotonicity_case2_lower_bound_l46_46971


namespace leonardo_extra_cents_needed_l46_46217

theorem leonardo_extra_cents_needed 
  (chocolate_cost_dollars : ℕ)
  (leonardo_dollars : ℕ)
  (borrowed_cents : ℕ)
  (chocolate_cost_cents := chocolate_cost_dollars * 100)
  (leonardo_cents := leonardo_dollars * 100)
  (total_leonardo_cents := leonardo_cents + borrowed_cents):
  chocolate_cost_dollars = 5 → leonardo_dollars = 4 → borrowed_cents = 59 → 
  chocolate_cost_cents - total_leonardo_cents = 41 :=
by
  intros hc hl hb
  rw [hc, hl, hb]
  simp
  sorry

end leonardo_extra_cents_needed_l46_46217


namespace equal_angles_dfe_cfe_l46_46207

-- Define the geometric setup
variables (A B C D E F : Point)
variables (h_trapezoid : IsTrapezoid A D B C) -- A, D, B, and C form a trapezoid with bases AD and BC
variables (h_right_angle : ∠A == 90°) -- angle at vertex A is a right angle
variables (h_diagonals_intersect : IntersectPoint AC BD E) -- E is the intersection of the diagonals
variables (h_projection : Projection E A B F) -- F is the foot of the perpendicular from E to AB

-- The theorem statement that the angles are equal
theorem equal_angles_dfe_cfe :
  ∠DFE = ∠CFE := by
  sorry

end equal_angles_dfe_cfe_l46_46207


namespace red_marble_count_l46_46698

theorem red_marble_count (x y : ℕ) (total_yellow : ℕ) (total_diff : ℕ) 
  (jar1_ratio_red jar1_ratio_yellow : ℕ) (jar2_ratio_red jar2_ratio_yellow : ℕ) 
  (h1 : jar1_ratio_red = 7) (h2 : jar1_ratio_yellow = 2) 
  (h3 : jar2_ratio_red = 5) (h4 : jar2_ratio_yellow = 3) 
  (h5 : 2 * x + 3 * y = 50) (h6 : 8 * y = 9 * x + 20) :
  7 * x + 2 = 5 * y :=
sorry

end red_marble_count_l46_46698


namespace jeff_remaining_laps_l46_46250

theorem jeff_remaining_laps
  (total_laps : ℕ)
  (fri_morning : ℕ)
  (fri_afternoon : ℕ)
  (fri_evening : ℕ)
  (sat_morning : ℕ)
  (sat_afternoon : ℕ)
  (sun_morning : ℕ)
  (total_weekend_laps : ℕ)
  (H : total_laps = 198)
  (H_fri_morning : fri_morning = 23)
  (H_fri_afternoon : fri_afternoon = 12)
  (H_fri_evening : fri_evening = 28)
  (H_sat_morning : sat_morning = 35)
  (H_sat_afternoon : sat_afternoon = 27)
  (H_sun_morning : sun_morning = 15)
  (H_total_weekend_laps : total_weekend_laps = total_laps) :
  total_weekend_laps - (fri_morning + fri_afternoon + fri_evening + sat_morning + sat_afternoon + sun_morning) = 58 :=
by
  subst H
  subst H_fri_morning
  subst H_fri_afternoon
  subst H_fri_evening
  subst H_sat_morning
  subst H_sat_afternoon
  subst H_sun_morning
  subst H_total_weekend_laps
  simp
  sorry

end jeff_remaining_laps_l46_46250


namespace geometric_sequence_log_sum_l46_46118

theorem geometric_sequence_log_sum (a : ℕ → ℝ) 
  (hpos : ∀ n, 0 < a n)
  (hseq : a 5 * a 6 + a 4 * a 7 = 8) :
  (∑ i in Finset.range 10, Real.logb 2 (a (i + 1))) = 10 :=
sorry

end geometric_sequence_log_sum_l46_46118


namespace find_integer_n_satisfying_conditions_l46_46874

open BigOperators

def sigma (n : ℕ) : ℕ :=
  ∑ i in (Finset.range (n + 1)).filter (λ d => n % d = 0), i

def p (n : ℕ) : ℕ :=
  (Nat.factors n).erase_dup.maximum'

theorem find_integer_n_satisfying_conditions (n : ℕ) (h_n_ge_2 : n ≥ 2)
  (h_condition : σ n / (p n - 1) = n) : n = 6 :=
  sorry

end find_integer_n_satisfying_conditions_l46_46874


namespace solution_set_f_x_lt_0_x_lt_0_l46_46160

def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 4 else 2^(-x) - 4

theorem solution_set_f_x_lt_0_x_lt_0 :
  (∀ x, f x = f (-x)) → (∀ x, x ≥ 0 → f x = 2^x - 4) →
  {x : ℝ | f x < 0 ∧ x < 0} = set.Ioo (-2 : ℝ) 0 :=
by
  intros h_even h_def
  unfold f
  sorry

end solution_set_f_x_lt_0_x_lt_0_l46_46160


namespace YQ_over_QG_eq_3_25_l46_46590

theorem YQ_over_QG_eq_3_25
  (X Y Z F G Q : Type)
  (d1 : Dist X Y = 9)
  (d2 : Dist X Z = 6)
  (d3 : Dist Y Z = 4)
  (b1 : angle_bisector X F Q Y G Z)
  : YQ / QG = 3.25 :=
sorry

end YQ_over_QG_eq_3_25_l46_46590


namespace integral_correct_avg_correct_l46_46841

noncomputable def integral_value : ℝ :=
  ∫ x in 5..12, (Real.sqrt (x + 4)) / x

theorem integral_correct :
  integral_value = 2 * Real.log ((5 * Real.exp 1) / 3) :=
sorry

noncomputable def average_value : ℝ :=
  (1 / (12 - 5)) * integral_value

theorem avg_correct :
  average_value = (2 / 7) * Real.log ((5 * Real.exp 1) / 3) :=
sorry

end integral_correct_avg_correct_l46_46841


namespace percent_increase_of_semicircle_areas_l46_46792

def area_of_semicircle (radius : ℝ) : ℝ :=
  (1 / 2) * real.pi * (radius ^ 2)

theorem percent_increase_of_semicircle_areas :
  let r_large := 6
  let r_small := 4
  let large_area := 2 * area_of_semicircle r_large
  let small_area := 2 * area_of_semicircle r_small
  (large_area / small_area - 1) * 100 = 125 :=
by
  sorry

end percent_increase_of_semicircle_areas_l46_46792


namespace find_A_l46_46168

noncomputable def angle_A (a b c : ℝ) (h : b^2 + c^2 - a^2 = bc) : ℝ := Real.angle_of_cos (1/2)

theorem find_A (a b c : ℝ) (h : b^2 + c^2 - a^2 = bc) : angle_A a b c h = π / 3 := 
by 
  -- Angle is found using cosine rule and given condition that b^2 + c^2 - a^2 = bc
  have : Real.angle_of_cos (1/2) = π / 3 := Real.angle_of_cos_half
  rw this 
  apply Real.angle_of_cos_half

end find_A_l46_46168


namespace trig_identity_solution_l46_46065

noncomputable def solve_trig_identity (α : ℝ) : Prop :=
  (cos (π - 2 * α) / sin (α - π / 4) = -√2 / 2 → sin (2 * α) = -3 / 4)

theorem trig_identity_solution (α : ℝ) :
  solve_trig_identity α :=
by
  intro h
  sorry

end trig_identity_solution_l46_46065


namespace lcm_48_180_l46_46036

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l46_46036


namespace area_increase_l46_46799

-- Defining the shapes and areas
def radius_large_side := 6
def radius_small_side := 4

def area_large_semicircles : ℝ := real.pi * (radius_large_side^2)
def area_small_semicircles : ℝ := real.pi * (radius_small_side^2)

-- The theorem statement
theorem area_increase : (area_large_semicircles / area_small_semicircles) = 2.25 → 
                         ((2.25 - 1) * 100) = 125 :=
by sorry

end area_increase_l46_46799


namespace largest_four_digit_sum_20_l46_46707

theorem largest_four_digit_sum_20 : ∃ n : ℕ, (999 < n ∧ n < 10000 ∧ (sum (nat.digits 10 n) = 20 ∧ ∀ m, 999 < m ∧ m < 10000 ∧ sum (nat.digits 10 m) = 20 → m ≤ n)) :=
by
  sorry

end largest_four_digit_sum_20_l46_46707


namespace x_value_l46_46560

theorem x_value (x : ℝ) (h : (x / 3 / 3) = (9 / (x / 3))) : x = 9 * (√3) ∨ x = -9 * (√3) := 
by
  sorry

end x_value_l46_46560


namespace tiles_difference_eighth_sixth_l46_46784

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Define the number of tiles given the side length
def number_of_tiles (n : ℕ) : ℕ := n * n

-- State the theorem about the difference in tiles between the 8th and 6th squares
theorem tiles_difference_eighth_sixth :
  number_of_tiles (side_length 8) - number_of_tiles (side_length 6) = 28 :=
by
  -- skipping the proof
  sorry

end tiles_difference_eighth_sixth_l46_46784


namespace smart_integers_divisible_by_18_ratio_l46_46004

def is_smart_integer (n : ℕ) : Prop :=
  n > 100 ∧ n < 300 ∧ n % 2 = 0 ∧ (n.digits 10).sum = 12

theorem smart_integers_divisible_by_18_ratio :
  (setOf (λ n, is_smart_integer n ∧ n % 18 = 0)).card = 
  (setOf (λ n, is_smart_integer n)).card :=
sorry

end smart_integers_divisible_by_18_ratio_l46_46004


namespace cosine_alpha_plus_beta_l46_46492

-- Definitions using the conditions
variables (α β : ℝ)

-- Conditions
axiom h1 : 0 < β ∧ β < π / 2
axiom h2 : (π / 2) < α ∧ α < π
axiom h3 : cos (α - β / 2) = - 1 / 9
axiom h4 : sin (α / 2 - β) = 2 / 3

-- Statement of the problem
theorem cosine_alpha_plus_beta :
  cos (α + β) = -239 / 729 :=
sorry  -- Proof omitted

end cosine_alpha_plus_beta_l46_46492


namespace solution_to_quadratic_solution_to_cubic_l46_46896

-- Problem 1: x^2 = 4
theorem solution_to_quadratic (x : ℝ) : x^2 = 4 -> x = 2 ∨ x = -2 := by
  sorry

-- Problem 2: 64x^3 + 27 = 0
theorem solution_to_cubic (x : ℝ) : 64 * x^3 + 27 = 0 -> x = -3 / 4 := by
  sorry

end solution_to_quadratic_solution_to_cubic_l46_46896


namespace geometric_sequence_general_term_l46_46585

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ q : ℝ, q > 0 ∧ (∀ n, a (n + 1) = q * a n)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_S3 : a 1 * (1 + (a 2 / a 1) + (a 3 / a 1)) = 21) 
  (h_condition : 2 * a 2 = a 3) :
  ∃ c : ℝ, c = 3 ∧ ∀ n, a n = 3 * 2^(n - 1) := sorry

end geometric_sequence_general_term_l46_46585


namespace line_intercepts_l46_46291

-- Definitions
def point_on_axis (a b : ℝ) : Prop := a = b
def passes_through_point (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 1

theorem line_intercepts (a b x y : ℝ) (hx : x = -1) (hy : y = 2) (intercept_property : point_on_axis a b) (point_property : passes_through_point a b x y) :
  (2 * x + y = 0) ∨ (x + y - 1 = 0) :=
sorry

end line_intercepts_l46_46291


namespace value_of_f_at_4_l46_46516

theorem value_of_f_at_4 (a : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, x^a) (h2 : f 2 = 1) : f 4 = 1 :=
by
  sorry

end value_of_f_at_4_l46_46516


namespace lcm_48_180_eq_720_l46_46046

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l46_46046


namespace converse_triangle_inequality_negation_nonzero_imp_nonzero_contraposition_nonzero_imp_nonzero_l46_46860

theorem converse_triangle_inequality (ABC : Triangle) 
(h : ABC.AB > ABC.AC) : ABC.angleC > ABC.angleB :=
sorry

theorem negation_nonzero_imp_nonzero (a b : ℝ) 
(h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

theorem contraposition_nonzero_imp_nonzero (a b : ℝ) 
(h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end converse_triangle_inequality_negation_nonzero_imp_nonzero_contraposition_nonzero_imp_nonzero_l46_46860


namespace no_tangent_line_l46_46125

theorem no_tangent_line (a : ℝ) :
  (a < -1 ∨ a > 0) → ∀ x : ℝ, 2 * sin x * cos x + 2 * a ≠ -1 :=
by
  sorry

end no_tangent_line_l46_46125


namespace solve_system_of_equations_l46_46241

theorem solve_system_of_equations
  (x y : ℚ)
  (h1 : 5 * x - 3 * y = -7)
  (h2 : 4 * x + 6 * y = 34) :
  x = 10 / 7 ∧ y = 33 / 7 :=
by
  sorry

end solve_system_of_equations_l46_46241


namespace problem_statement_l46_46238

variable {x : ℝ} 

def p : Prop := ∀ x : ℝ, 2^x < 3^x
def q : Prop := ∃ x_0 : ℝ, x_0^3 = 1 - x_0^2

theorem problem_statement : (¬p) ∧ q := 
by 
  sorry

end problem_statement_l46_46238


namespace rectangle_area_l46_46803

theorem rectangle_area :
  ∃ (a b : ℕ), a ≠ b ∧ Even a ∧ (a * b = 3 * (2 * a + 2 * b)) ∧ (a * b = 162) :=
by
  sorry

end rectangle_area_l46_46803


namespace denominator_divisible_by_2_or_5_l46_46562

theorem denominator_divisible_by_2_or_5 
  (b : ℕ → ℕ) (a : ℕ → ℕ) (m k : ℕ) (h1 : m ≥ 1) (h2 : b m ≠ a k) :
  ∃ (p q : ℕ), (q ≠ 0) ∧ (∃ r, 0 < r ∧ p / q = 0.b₁ b₂ ... bₘ a₁ a₂ ... āₖ ... ) ∧ (2 ∣ q ∨ 5 ∣ q) ∨ (∃ s t, s ≠ 0 ∧ t ≠ 0 ∧ p / t = 0.b₁ b₂ ... bₘ a₁ a₂ ... āₖ ... ) (2 ∣ t ∨ 5 ∣ t) :=
sorry

end denominator_divisible_by_2_or_5_l46_46562


namespace largest_four_digit_sum_20_l46_46711

-- Defining the four-digit number and conditions.
def is_four_digit_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  ∃ a b c d : ℕ, a + b + c + d = s ∧ n = 1000 * a + 100 * b + 10 * c + d

-- Proof problem statement.
theorem largest_four_digit_sum_20 : ∃ n, is_four_digit_number n ∧ digits_sum_to n 20 ∧ ∀ m, is_four_digit_number m ∧ digits_sum_to m 20 → m ≤ n :=
  sorry

end largest_four_digit_sum_20_l46_46711


namespace tan_A_right_triangle_l46_46195

theorem tan_A_right_triangle (A B C : Type) [RealField A] [RealField B] [RealField C]
  (h1 : ∠ABC = 90) (h2 : sin B = 3 / 5) : tan A = 4 / 3 :=
sorry

end tan_A_right_triangle_l46_46195


namespace value_of_m_plus_n_l46_46631

-- Conditions
variables (m n : ℤ)
def P_symmetric_Q_x_axis := (m - 1 = 2 * m - 4) ∧ (n + 2 = -2)

-- Proof Problem Statement
theorem value_of_m_plus_n (h : P_symmetric_Q_x_axis m n) : (m + n) ^ 2023 = -1 := sorry

end value_of_m_plus_n_l46_46631


namespace total_widgets_sold_after_10_days_l46_46654

theorem total_widgets_sold_after_10_days :
  let a := 2
  let r := 2
  let n := 10
  ∑ i in Finset.range n, (a * r^i) = 2046 := 
by
  sorry

end total_widgets_sold_after_10_days_l46_46654


namespace monotonicity_2_distinct_zeros_range_l46_46137

def f (x a : ℝ) : ℝ := x^2 + a * x - a^2 * Real.log x

-- Part (1) statement
theorem monotonicity_2 :
  ∀ x > 0, f x 2 decreasing_on Ioo (0:ℝ) 1 ∧ f x 2 increasing_on Ioi 1 :=
sorry

-- Part (2) statement
theorem distinct_zeros_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > 2 * Real.exp (3 / 4) :=
sorry

end monotonicity_2_distinct_zeros_range_l46_46137


namespace spaceship_finds_alien_l46_46825

-- Define the radius of the planet (R), speed of alien (u), speed of spaceship (v), 
-- and the condition that the spaceship speed is greater than 10 times the alien's speed
variables (R u v : ℝ) (h1 : v > 10 * u)

-- The proof problem: Prove that the spaceship can always find the alien given the above conditions
theorem spaceship_finds_alien (h2 : ∀ t: ℝ, alien_position t ∈ surface(R)) 
                               (h3 : ∀ t: ℝ, spaceship_speed t = v)
                               (h4 : ∃ α > 0, ∀ t, alien_speed t ≤ α): 
                                ∃ t, spaceship_position t = alien_position t :=
  sorry

end spaceship_finds_alien_l46_46825


namespace cube_side_length_and_combined_volume_l46_46719

theorem cube_side_length_and_combined_volume
  (surface_area_large_cube : ℕ)
  (h_surface_area : surface_area_large_cube = 864)
  (side_length_large_cube : ℕ)
  (combined_volume : ℕ) :
  side_length_large_cube = 12 ∧ combined_volume = 1728 :=
by
  -- Since we only need the statement, the proof steps are not included.
  sorry

end cube_side_length_and_combined_volume_l46_46719


namespace perpendicular_line_slope_l46_46161

theorem perpendicular_line_slope (a : ℝ) : 
  (∀ a : ℝ, (∃ m₁ m₂ : ℝ,  m₁ = -1/a ∧ m₂ = -2/3 ∧ m₁ * m₂ = -1)) → a = 2/3 :=
by
  intros h
  rcases h a with ⟨m₁, m₂, h₁, h₂, h₃⟩
  rw [h₁, h₂] at h₃
  linarith

end perpendicular_line_slope_l46_46161


namespace rectangle_length_l46_46216

theorem rectangle_length (side_of_square : ℕ) (width_of_rectangle : ℕ) (same_wire_length : ℕ) 
(side_eq : side_of_square = 12) (width_eq : width_of_rectangle = 6) 
(square_perimeter : same_wire_length = 4 * side_of_square) :
  ∃ (length_of_rectangle : ℕ), 2 * (length_of_rectangle + width_of_rectangle) = same_wire_length ∧ length_of_rectangle = 18 :=
by
  sorry

end rectangle_length_l46_46216


namespace boxwoods_shaped_into_spheres_l46_46245

theorem boxwoods_shaped_into_spheres :
  ∀ (total_boxwoods : ℕ) (cost_trimming : ℕ) (cost_shaping : ℕ) (total_charge : ℕ) (x : ℕ),
    total_boxwoods = 30 →
    cost_trimming = 5 →
    cost_shaping = 15 →
    total_charge = 210 →
    30 * 5 + x * 15 = 210 →
    x = 4 :=
by
  intros total_boxwoods cost_trimming cost_shaping total_charge x
  rintro rfl rfl rfl rfl h
  sorry

end boxwoods_shaped_into_spheres_l46_46245


namespace enthalpy_change_l46_46741

def DeltaH_prods : Float := -286.0 - 297.0
def DeltaH_reacts : Float := -20.17
def HessLaw (DeltaH_prods DeltaH_reacts : Float) : Float := DeltaH_prods - DeltaH_reacts

theorem enthalpy_change : HessLaw DeltaH_prods DeltaH_reacts = -1125.66 := by
  -- Lean needs a proof, which is not needed per instructions
  sorry

end enthalpy_change_l46_46741


namespace evaluate_tensor_expression_l46_46994

-- Define the tensor operation
def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- The theorem we want to prove
theorem evaluate_tensor_expression : tensor (tensor 5 3) 2 = 293 / 15 := by
  sorry

end evaluate_tensor_expression_l46_46994


namespace sum_of_a_approx_l46_46877

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∃ x1 x2 x3 : ℝ,
  (x1 = -4 * π * a / (4 - a)) ∧ (x2 = -4 * π * a / (4 - a)) ∧ (x3 = -4 * π * a / (4 - a)) ∧
  0 <= x1 ∧ x1 < π ∧ 0 <= x2 ∧ x2 < π ∧ 0 <= x3 ∧ x3 < π

theorem sum_of_a_approx : ∀ a : ℝ,
  problem_statement a →
  a = 1 ∨ a = 3 ∨ a = 16/3 →
  1 + 3 + 5.33 ≈ 9.33 := 
sorry

end sum_of_a_approx_l46_46877


namespace number_of_non_integer_terms_in_expansion_l46_46558

noncomputable def f (x : ℝ) : ℝ := x + |x|

def integral_value : ℝ := 2 * ∫ x in -3..3, f x

theorem number_of_non_integer_terms_in_expansion :
  integral_value = 18 →
  ∃ n : ℕ, n = 15 ∧
      (∀ a = integral_value, 
          let T_r := λ r : ℕ, (-1)^r * (Nat.choose 18 r) * x^(9 - 5*r/6) in
          {r : ℕ | 0 ≤ r ∧ r ≤ 18 ∧ ¬(∃ k : ℤ, 9 - (5 * ↑r / 6) = k)}.to_finset.card = n) :=
by sorry

end number_of_non_integer_terms_in_expansion_l46_46558


namespace range_of_x_l46_46731

theorem range_of_x:
  (∀ (a : Fin 25 → ℝ), (∀ i, a i = 0 ∨ a i = 2) →
    let x := ∑ i in Finset.range 25, (a i) / 3^(i + 1) in 
    (0 ≤ x ∧ x < 1/3) ∨ (2/3 ≤ x ∧ x < 1)) :=
begin
  sorry
end

end range_of_x_l46_46731


namespace min_expr_value_min_expr_value_iff_l46_46926

theorem min_expr_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 :=
by {
  sorry
}

theorem min_expr_value_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2) = 4 / 9) ↔ (x = 2.5 ∧ y = 2.5) :=
by {
  sorry
}

end min_expr_value_min_expr_value_iff_l46_46926


namespace similar_graphs_l46_46074

def P (x : ℝ) : ℝ := sorry -- Placeholder for the actual polynomial definition

-- Given conditions
def distinct_positive_coeffs (p : ℝ → ℝ) : Prop :=
  sorry -- Placeholder condition stating p has distinct positive nonzero coefficients

def replace_with_median (p : ℝ → ℝ) : ℝ → ℝ :=
  sorry -- Placeholder for the function replacing coefficients with their median

-- Definition of Q(x) as replacing coefficients of P(x) with their median
def Q (x : ℝ) : ℝ := replace_with_median P

-- The proof problem statement
theorem similar_graphs (P : ℝ → ℝ) (Q : ℝ → ℝ)
  (h1 : distinct_positive_coeffs P)
  (h2 : Q = replace_with_median P) :
  sorry -- Here we write the equivalent property that the graphs are very similar
:= sorry

end similar_graphs_l46_46074


namespace probability_of_edge_endpoint_l46_46320

/-- A regular icosahedron has 20 vertices, and each vertex is connected to 5 others. -/
def icosahedron_vertices : nat := 20
def edges_per_vertex : nat := 5

/-- Total number of ways to choose 2 vertices from 20 vertices -/
def total_combinations : nat := Nat.choose icosahedron_vertices 2

/-- Total number of successful edge-forming pairs -/
def successful_pairs : nat := icosahedron_vertices * edges_per_vertex

/-- The desired probability that two randomly chosen vertices form an edge -/
theorem probability_of_edge_endpoint : (successful_pairs : ℚ) / total_combinations = 10 / 19 :=
by
  sorry

end probability_of_edge_endpoint_l46_46320


namespace pants_per_pair_l46_46016

def number_of_pants_per_pair (initial_pants : ℕ) (final_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  (final_pants - initial_pants) / (pairs_per_year * years)

theorem pants_per_pair (h_initial : 50) (h_final : 90) (h_pairs_per_year : 4) (h_years : 5) :
  number_of_pants_per_pair h_initial h_final h_pairs_per_year h_years = 2 :=
by
  sorry

end pants_per_pair_l46_46016


namespace arithmetic_sequence_sum_remainder_l46_46718

theorem arithmetic_sequence_sum_remainder 
  (a d : ℕ) (n : ℤ) 
  (h₁ : a = 3) 
  (h₂ : d = 8) 
  (h₃ : n ≥ 1) 
  (h₄ : a + (n - 1) * d = 283) : 
  (∑ i in finset.range n, a + i * d) % 8 = 4 := 
sorry

end arithmetic_sequence_sum_remainder_l46_46718


namespace no_such_function_exists_l46_46593

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (x : ℕ), (nat.iterate f (f x) x) = x + 1 :=
by sorry

end no_such_function_exists_l46_46593


namespace number_of_ordered_pairs_l46_46477

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), (∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ x < y ∧ (2 * x * y) / (x + y) = 4^30) ↔ n = 59) :=
begin
  sorry
end

end number_of_ordered_pairs_l46_46477


namespace spot_reach_area_l46_46269

-- Defining the given conditions
def side_length : ℝ := 1 -- side of the pentagon
def tether_length : ℝ := 3 -- length of the tether

-- Defining the angles and areas
def internal_angle : ℝ := 108 -- internal angle of the pentagon in degrees
def accessible_sector_angle1 : ℝ := 288 -- angle of the large sector Spot can reach
def accessible_sector_angle2 : ℝ := 72 -- angle of the small sector Spot can reach

-- Convert degree measures to radians for area calculations
def deg_to_rad (deg : ℝ) : ℝ := deg * (real.pi / 180)

-- Areas of the sectors Spot can reach
def area_large_sector : ℝ := real.pi * (tether_length ^ 2) * (accessible_sector_angle1 / 360)
def area_small_sector : ℝ := real.pi * (side_length ^ 2) * (accessible_sector_angle2 / 360)

-- Total area accessible by Spot
def total_accessible_area : ℝ := area_large_sector + 2 * area_small_sector

-- Statement to prove
theorem spot_reach_area : total_accessible_area = 7.6 * real.pi := by
  sorry

end spot_reach_area_l46_46269


namespace yuna_survey_l46_46383

theorem yuna_survey :
  let M := 27
  let K := 28
  let B := 22
  M + K - B = 33 :=
by
  sorry

end yuna_survey_l46_46383


namespace valid_pairs_l46_46024

-- Define the target function and condition
def satisfies_condition (k l : ℤ) : Prop :=
  (7 * k - 5) * (4 * l - 3) = (5 * k - 3) * (6 * l - 1)

-- The theorem stating the exact pairs that satisfy the condition
theorem valid_pairs :
  ∀ (k l : ℤ), satisfies_condition k l ↔
    (k = 0 ∧ l = 6) ∨
    (k = 1 ∧ l = -1) ∨
    (k = 6 ∧ l = -6) ∨
    (k = 13 ∧ l = -7) ∨
    (k = -2 ∧ l = -22) ∨
    (k = -3 ∧ l = -15) ∨
    (k = -8 ∧ l = -10) ∨
    (k = -15 ∧ l = -9) :=
by
  sorry

end valid_pairs_l46_46024


namespace real_values_of_x_l46_46883

theorem real_values_of_x :
  {x : ℝ | (∃ y, y = (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ∧ y ≥ -1)} =
  {x | -1 ≤ x ∧ x < -1/3 ∨ -1/3 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 1 < x} := 
sorry

end real_values_of_x_l46_46883


namespace floor_expression_equality_l46_46845

theorem floor_expression_equality :
  ⌊((2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023))⌋ = 8 := 
sorry

end floor_expression_equality_l46_46845


namespace find_scalars_r_s_l46_46224

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, -4; 5, 2]

theorem find_scalars_r_s :
  ∃ (r s : ℚ), (N ⬝ N = r • N + s • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧ (r = 5) ∧ (s = -26) :=
by
  use [5, -26]
  sorry

end find_scalars_r_s_l46_46224


namespace min_value_of_expression_l46_46236

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x = a^2 + b^2 + (1 / (a + b)^2) + (1 / (a * b)) ∧ x = Real.sqrt 10 :=
sorry

end min_value_of_expression_l46_46236


namespace weight_of_replaced_person_l46_46656

theorem weight_of_replaced_person (W : ℕ) (avg_increase : ℝ) (new_person_weight : ℕ) (num_persons : ℕ) :
  num_persons = 10 →
  avg_increase = 6.3 →
  new_person_weight = 128 →
  W = new_person_weight - num_persons * avg_increase.toNat →
  W = 65 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4
  sorry

end weight_of_replaced_person_l46_46656


namespace S17_value_l46_46001

variable (a1 d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def arithmetic_sum (a1 d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a1 + (n - 1) * d)

theorem S17_value :
  (arithmetic_sum a1 d 7 = 14) →
  (arithmetic_sum a1 d 10 = 13) →
  arithmetic_sum a1 d 17 = -17 / 3 := by
  sorry

end S17_value_l46_46001


namespace ana_salary_after_raise_and_cut_l46_46436

theorem ana_salary_after_raise_and_cut (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) :
  initial_salary = 2500 → raise_percentage = 0.10 → cut_percentage = 0.15 →
  (initial_salary * (1 + raise_percentage) * (1 - cut_percentage)) = 2337.50 :=
by
  intros h_initial_salary h_raise_percentage h_cut_percentage
  rw [h_initial_salary, h_raise_percentage, h_cut_percentage]
  norm_num
  sorry

end ana_salary_after_raise_and_cut_l46_46436


namespace smaller_octagon_fraction_of_area_l46_46297

-- Defining the larger and smaller octagons
def regular_octagon (A B C D E F G H : Point) (O : Point) : Prop :=
  is_regular_octagon A B C D E F G H O

def smaller_octagon (A B C D E F G H : Point) (P Q R S T U V W : Point) : Prop :=
  is_midpoint P A B ∧ is_midpoint Q B C ∧ is_midpoint R C D ∧ is_midpoint S D E ∧
  is_midpoint T E F ∧ is_midpoint U F G ∧ is_midpoint V G H ∧ is_midpoint W H A

-- Fraction of the area enclosed
theorem smaller_octagon_fraction_of_area (A B C D E F G H P Q R S T U V W O : Point) :
  regular_octagon A B C D E F G H O →
  smaller_octagon A B C D E F G H P Q R S T U V W →
  area (smaller_octagon A B C D E F G H P Q R S T U V W) = 
  (3 / 4) * area (regular_octagon A B C D E F G H O) :=
sorry

end smaller_octagon_fraction_of_area_l46_46297


namespace exists_multiple_2003_no_restricted_digits_l46_46258

theorem exists_multiple_2003_no_restricted_digits : ∃ n : ℕ, 
  (∃ k : ℕ, n = 2003 * k) ∧ 
  (n < 10^11) ∧ 
  (∀ d ∈ (n.digits 10), d ∈ {0, 1, 8, 9}) :=
by sorry

end exists_multiple_2003_no_restricted_digits_l46_46258


namespace num_prime_divisors_50_fact_l46_46552
open Nat -- To simplify working with natural numbers

-- We define the prime numbers less than or equal to 50.
def primes_le_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- The problem statement: Prove that the number of prime divisors of 50! which are less than or equal to 50 is 15.
theorem num_prime_divisors_50_fact : (primes_le_50.length = 15) :=
by 
  -- Here we use sorry to skip the proof.
  sorry

end num_prime_divisors_50_fact_l46_46552


namespace extremum_of_f_inequality_for_f_l46_46521

-- Define the function f and its properties
def f (a b x : ℝ) : ℝ := (1 / 3) * a * x^3 - b * (Real.log x)

-- Given conditions
def tangent_condition (a b : ℝ) : Prop :=
  let y := (-2) * 1 + (8 / 3) in
  let f1 := (1 / 3) * a in
  let f1' := a - b in
  f1' * (1 - 1) + f1 == y ∧ a - b == -2 ∧ (-2 / 3) * a + b == 8 / 3

-- Statement for (I) Extremum
theorem extremum_of_f {a b : ℝ} (h : tangent_condition a b) :
  let x_min := Real.cbrt 2 in
  f a b x_min = (4 / 3) * (1 - Real.log 2) := sorry

-- Statement for (II) Inequality
theorem inequality_for_f {a b : ℝ} (h : tangent_condition a b) (x : ℝ) (hx : x > 0) :
  (x * f a b x) / 4 + x / Real.exp x < (x^4) / 6 + 2 / Real.exp 1 := sorry

end extremum_of_f_inequality_for_f_l46_46521


namespace solve_for_a_b_c_l46_46107

-- Conditions and necessary context
def m_angle_A : ℝ := 60  -- In degrees
def BC_length : ℝ := 12  -- Length of BC in units
def angle_DBC_eq_three_times_angle_ECB (DBC ECB : ℝ) : Prop := DBC = 3 * ECB

-- Definitions for perpendicularity could be checked by defining angles
-- between lines, but we can assert these as properties.
axiom BD_perpendicular_AC : Prop
axiom CE_perpendicular_AB : Prop

-- The proof problem
theorem solve_for_a_b_c :
  ∃ (EC a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  b ≠ c ∧ 
  (∀ d, b ∣ d → d = b ∨ d = 1) ∧ 
  (∀ d, c ∣ d → d = c ∨ d = 1) ∧
  EC = a * (Real.sqrt b + Real.sqrt c) ∧ 
  a + b + c = 11 :=
by
  sorry

end solve_for_a_b_c_l46_46107


namespace probability_S2_ne_0_and_S8_eq_2_l46_46695

-- Define the sequence a_n based on coin tosses
def a_n (n : ℕ) (outcome : Fin n → Bool) : ℤ :=
  if outcome n then 1 else -1

-- Define the sum S_n
def S (n : ℕ) (outcome : Fin n → Bool) : ℤ :=
  (Finset.range n).sum (λ i, a_n i outcome)

-- Define the probability calculation for the sequence constraints
noncomputable def probability_of_event : ℚ :=
  (1 / 2)^2 * Nat.choose 6 3 * (1 / 2)^6 + 
  (1 / 2)^2 * Nat.choose 6 5 * (1 / 2)^6

-- Statement of the problem in Lean
theorem probability_S2_ne_0_and_S8_eq_2 :
  probability_of_event = 13 / 128 :=
  sorry

end probability_S2_ne_0_and_S8_eq_2_l46_46695


namespace tan_alpha_eq_two_l46_46498

theorem tan_alpha_eq_two (α : ℝ) (h1 : α ∈ Set.Ioc 0 (Real.pi / 2))
    (h2 : Real.sin ((Real.pi / 4) - α) * Real.sin ((Real.pi / 4) + α) = -3 / 10) :
    Real.tan α = 2 := by
  sorry

end tan_alpha_eq_two_l46_46498


namespace lcm_48_180_l46_46037

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l46_46037


namespace graduation_ceremony_chairs_l46_46178

theorem graduation_ceremony_chairs : 
  (∃ (graduates parents additional_families teachers administrators total_chairs : ℕ),
  graduates = 75 ∧ 
  parents = 2 * graduates ∧ 
  additional_families = (75 * 30 / 100).ceil ∧ 
  teachers = 25 ∧ 
  administrators = (25 / 5) * 2 ∧ 
  total_chairs = graduates + parents + additional_families + teachers + administrators ∧ 
  total_chairs = 283) := 
sorry

end graduation_ceremony_chairs_l46_46178


namespace area_of_yellow_square_l46_46063

theorem area_of_yellow_square (edge_length : ℝ) (purple_paint_total : ℝ) (num_faces : ℝ) (face_paint_purple : ℝ) :
  edge_length = 15 ∧ purple_paint_total = 900 ∧ num_faces = 6 ∧ face_paint_purple = purple_paint_total / num_faces →
  let face_area := edge_length * edge_length in
  let yellow_square_area := face_area - face_paint_purple in
  yellow_square_area = 75 :=
begin
  sorry
end

end area_of_yellow_square_l46_46063


namespace area_semicircles_percent_increase_l46_46795

noncomputable def radius_large_semicircle (length: ℝ) : ℝ := length / 2
noncomputable def radius_small_semicircle (width: ℝ) : ℝ := width / 2

noncomputable def area_semicircle (radius: ℝ) : ℝ := (real.pi * radius^2) / 2

theorem area_semicircles_percent_increase
  (length: ℝ) (width: ℝ)
  (h_length: length = 12) (h_width: width = 8) :
  let 
    large_radius := radius_large_semicircle length,
    small_radius := radius_small_semicircle width,
    area_large := 2 * area_semicircle large_radius,
    area_small := 2 * area_semicircle small_radius
  in
  (area_large / area_small - 1) * 100 = 125 :=
by
  sorry

end area_semicircles_percent_increase_l46_46795


namespace path_count_from_neg6_neg6_to_6_6_avoiding_square_l46_46402

theorem path_count_from_neg6_neg6_to_6_6_avoiding_square :
  let start := (-6, -6)
  let end := (6, 6)
  let condition := λ (p : Int × Int), p.1 ∈ [-6,6] ∧ p.2 ∈ [-6,6] ∧ (p.1 < -3 ∨ p.1 > 3 ∨ p.2 < -3 ∨ p.2 > 3)
  ∀ steps : List (Int × Int),
    steps.head = start →
    steps.last (0, 0) = end →
    (∀ i < steps.length - 1, ((steps.nth i).getOrElse (0, 0)).1 + 1 = ((steps.nth (i+1)).getOrElse (0, 0)).1 ∨ 
                           ((steps.nth i).getOrElse (0, 0)).2 + 1 = ((steps.nth (i+1)).getOrElse (0, 0)).2) →
    (∀ p ∈ steps, condition p) →
    steps.length = 25 →
    steps.countₐ start = 1 →
    steps.countₐ end = 1 →
    (∑ v in steps, if v = start ∨ v = end then 1 else 0) = 11594 :=
by
  sorry

end path_count_from_neg6_neg6_to_6_6_avoiding_square_l46_46402


namespace rectangular_prism_faces_edges_vertices_sum_l46_46332

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46332


namespace sparrows_below_threshold_l46_46020

def sparrow_population (initial_pop : ℕ) (year : ℕ) : ℝ :=
  initial_pop * (0.7 ^ (year - 2010))

theorem sparrows_below_threshold (initial_pop : ℕ) (threshold : ℝ) :
  initial_pop = 1200 →
  threshold = 180 →
  ∃ year, year ≥ 2010 ∧ sparrow_population initial_pop year < threshold :=
by {
  intros h1 h2,
  use 2016,
  split,
  { linarith },
  { rw [h1, h2],
    norm_num,
    sorry
  }
}

end sparrows_below_threshold_l46_46020


namespace optimal_usage_life_l46_46773

noncomputable def total_cost (x : ℕ) : ℕ :=
  x^2 + 2 * x + 100

noncomputable def average_annual_cost (x : ℕ) : ℝ :=
  (x^2 + 2 * x + 100 : ℝ) / (x: ℝ)

theorem optimal_usage_life :
  argmin average_annual_cost { x : ℕ | 0 < x } = 10 :=
sorry

end optimal_usage_life_l46_46773


namespace value_to_be_subtracted_l46_46162

theorem value_to_be_subtracted (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 24) / 10 = 3) : x = 5 := by
  sorry

end value_to_be_subtracted_l46_46162


namespace smallest_positive_integer_satisfying_conditions_l46_46325

open Nat

def smallest_n : ℕ := 6

def satisfies_conditions (n : ℕ) : Prop :=
  ∃ p, p < n ∧ Prime p ∧ Odd p ∧ (n^2 - n + 4) % p = 0 ∧
  ∃ q, q < n ∧ Prime q ∧ (n^2 - n + 4) % q ≠ 0

theorem smallest_positive_integer_satisfying_conditions : smallest_n = 6 ∧ satisfies_conditions smallest_n :=
by
  sorry

end smallest_positive_integer_satisfying_conditions_l46_46325


namespace range_of_a_plus_b_l46_46225

theorem range_of_a_plus_b (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) : 
  0 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l46_46225


namespace vector_properties_l46_46985

open Real

def a : ℝ × ℝ × ℝ := (-2, -3, 1)
def b : ℝ × ℝ × ℝ := (2, 0, 4)
def c : ℝ × ℝ × ℝ := (-4, -6, 2)

theorem vector_properties : (∃ k : ℝ, c = (k • a)) ∧ (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) :=
by
  sorry

end vector_properties_l46_46985


namespace minimum_value_of_norm_sub_vectors_l46_46520

open Real

namespace Proof

variables {V : Type*} [inner_product_space ℝ V]

def orthogonal_unit_vectors (a b : V) : Prop :=
  inner a a = 1 ∧ inner b b = 1 ∧ inner a b = 0

def vector_conditions (a b c : V) (len_c : ℝ) (ca dot cb : ℝ) : Prop :=
  orthogonal_unit_vectors a b ∧ 
  ∥c∥ = len_c ∧ 
  inner c a = dot_ca ∧ 
  inner c b = dot_cb

theorem minimum_value_of_norm_sub_vectors 
  {a b c : V} {t1 t2 : ℝ} 
  (hv : orthogonal_unit_vectors a b) 
  (hc : ∥c∥ = 13) 
  (hca : inner c a = 3) 
  (hcb : inner c b = 4) : 
  ∃ t1 t2 : ℝ, ∥c - t1 • a - t2 • b∥ = 12 := 
begin
  sorry
end

end Proof

end minimum_value_of_norm_sub_vectors_l46_46520


namespace floor_ceil_sum_l46_46468

theorem floor_ceil_sum : (Int.floor (-3.67) + Int.ceil (34.2) = 31) := 
by
  sorry

end floor_ceil_sum_l46_46468


namespace shape_is_cone_l46_46057

-- Constants and definitions
constant c : ℝ
constant π : ℝ

-- Definition of the spherical coordinate condition
def spherical_coordinates_shape (ρ θ φ : ℝ) := φ = (π / 2) - c

-- Statement to prove:
theorem shape_is_cone (ρ θ φ : ℝ) (h : spherical_coordinates_shape ρ θ φ) : 
  ∃ k : ℝ, k = c ∧ 
    (ρ = k * sin(θ) ∧ φ = k * cos(θ)) := sorry

end shape_is_cone_l46_46057


namespace limit_sum_binom_inverse_l46_46855

noncomputable def binom (n k : ℕ) : ℚ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def sum_binom_inverse (n : ℕ) : ℚ :=
  ∑ i in finset.range (n+1), 1 / binom n i

theorem limit_sum_binom_inverse :
  filter.tendsto sum_binom_inverse filter.at_top (filter.tendsto.const_nhds 2) :=
sorry

end limit_sum_binom_inverse_l46_46855


namespace Sasha_greatest_number_of_dimes_l46_46262

noncomputable def greatest_number_of_dimes (total_value : ℚ) (num_quarters : ℕ) (num_nickels : ℕ) (num_dimes : ℕ) :=
  total_value = 5.50 ∧ num_quarters = num_nickels ∧ num_dimes ≥ 3 * num_quarters ∧
  0.25 * num_quarters + 0.05 * num_nickels + 0.10 * num_dimes = 5.50

theorem Sasha_greatest_number_of_dimes :
  ∃ d : ℕ, greatest_number_of_dimes 5.50 9 9 d ∧ d = 28 :=
by 
  existsi 28
  sorry

end Sasha_greatest_number_of_dimes_l46_46262


namespace grandfather_yoongi_age_ratio_l46_46598

theorem grandfather_yoongi_age_ratio :
  ∀ (last_year_yoongi_age last_year_grandfather_age : ℕ),
    (last_year_yoongi_age = 6) →
    (last_year_grandfather_age = 62) →
    (last_year_grandfather_age + 1) = 9 * (last_year_yoongi_age + 1) :=
by
  intros last_year_yoongi_age last_year_grandfather_age h1 h2
  rw [h1, h2]
  sorry

end grandfather_yoongi_age_ratio_l46_46598


namespace graph_length_squared_l46_46270

def f (x : ℝ) := x + 1
def g (x : ℝ) := -x + 5
def h (x : ℝ) := (4 : ℝ)
def p (x : ℝ) := (1 : ℝ)

def j (x : ℝ) := max (f x) (max (g x) (max (h x) (p x)))
def k (x : ℝ) := min (f x) (min (g x) (min (h x) (p x)))

theorem graph_length_squared :
  (let ℓ := 8 in ℓ^2) = 64 :=
by
  sorry

end graph_length_squared_l46_46270


namespace length_upper_base_eq_half_d_l46_46079

variables {A B C D M: Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables {d : ℝ}

def trapezoid (A B C D : Type*) : Prop :=
  ∃ p : B, ∃ q : C, ∃ r : D, A ≠ p ∧ p ≠ q ∧ q ≠ r ∧ r ≠ A

def midpoint (A D : Type*) (N : Type*) (d : ℝ) : Prop :=
  dist A N = d / 2 ∧ dist N D = d / 2

axiom dm_perp_ab : ∀ (M : Type*), dist D M ∧ D ≠ M → dist M (id D) ≠ 0

axiom mc_eq_cd : dist M C = dist C D

theorem length_upper_base_eq_half_d
  (A B C D M : Type*)
  (h1 : trapezoid A B C D)
  (h2 : dist A D = d)
  (h3 : dm_perp_ab M)
  (h4 : mc_eq_cd) :
  dist B C = d / 2 :=
sorry

end length_upper_base_eq_half_d_l46_46079


namespace log_sqrt_five_base_five_l46_46019

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_sqrt_five_base_five : log_base 5 (sqrt[4](5)) = 1 / 4 := 
by
  sorry

end log_sqrt_five_base_five_l46_46019


namespace problem1_problem2_l46_46984

-- Declare the given conditions as definitions
variable (a b : EuclideanSpace ℝ (Fin 3))
variable (h_norm_a : ∥a∥ = 2)
variable (h_norm_b : ∥b∥ = Real.sqrt 2)
variable (h_angle : Real.angle a b = Real.pi * (3/4))

-- Mathematical proof problem 1
theorem problem1 : ∥a - (2:ℝ) • b∥ = 2 * Real.sqrt 5 := sorry

-- Declare additional condition for problem 2
variable (k : ℝ)
variable (h_perp : InnerProductSpace.inner (a - (2:ℝ) • b) (k • a + b) = 0)

-- Mathematical proof problem 2
theorem problem2 : k = 3 / 4 := sorry

end problem1_problem2_l46_46984


namespace max_value_l46_46619

noncomputable theory

open Real

-- Define the problem in Lean 4

def conditions (x : Fin 1997 → ℝ) : Prop :=
  (∀ i, -1/√3 ≤ x i ∧ x i ≤ √3) ∧
  (∑ i, x i = -318 * √3)

theorem max_value (x : Fin 1997 → ℝ) (h : conditions x) :
  (∑ i, (x i)^12) ≤ 189548 :=
sorry

end max_value_l46_46619


namespace sample_and_probability_l46_46315

-- (I) Calculation part
def total_parents (parents1 parents2 parents3 : ℕ) := parents1 + parents2 + parents3

def sample_ratio (total_samples : ℕ) (total_individuals : ℕ) := (total_samples : ℝ) / (total_individuals : ℝ)

def number_sampled (num_parents : ℕ) (ratio : ℝ) := (num_parents : ℝ) * ratio

-- (II) Probability part
def total_combinations (n k : ℕ) := Nat.choose n k

def favorable_combinations := [
  ("A_1", "C_1"), ("A_1", "C_2"), ("A_2", "C_1"), ("A_2", "C_2"),
  ("A_3", "C_1"), ("A_3", "C_2"), ("B_1", "C_1"), ("B_1", "C_2"),
  ("C_1", "C_2")
]

def probability (favorable total : ℕ) := (favorable : ℝ) / (total : ℝ)

theorem sample_and_probability :
  let parents1 := 54;
  let parents2 := 18;
  let parents3 := 36;
  let totalSamples := 6;
  let totalParents := total_parents parents1 parents2 parents3;
  let ratio := sample_ratio totalSamples totalParents;
  let sampled1 := number_sampled parents1 ratio;
  let sampled2 := number_sampled parents2 ratio;
  let sampled3 := number_sampled parents3 ratio;
  let totalComb := total_combinations 6 2;
  let favComb := favorable_combinations.length;
  let prob := probability favComb totalComb 
  in sampled1 = 3 ∧ sampled2 = 1 ∧ sampled3 = 2 ∧ prob = 3 / 5 :=
by
  sorry

end sample_and_probability_l46_46315


namespace largest_n_satisfying_conditions_l46_46892

open Nat

-- Define Euler's totient function φ
noncomputable def totient (n : ℕ) : ℕ := 
  if n = 0 then 0 else finset.card { m ∈ finset.range n | gcd m n = 1 }

-- Define the main theorem statement
theorem largest_n_satisfying_conditions : 
  ∃ (n : ℕ), (∑ m in finset.range (n + 1), ((n / m) - ((n - 1) / m))) = 1992 ∧ totient n ∣ n ∧ 
  (∀ m : ℕ, ( (∑ k in finset.range (m + 1), (m / k) - ((m - 1) / k) = 1992) 
            → (totient m ∣ m) 
            → m ≤ n)) ∧ 
  (n = 2^(1991)) :=
begin
  sorry
end

end largest_n_satisfying_conditions_l46_46892


namespace find_alpha_l46_46506

theorem find_alpha (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 360)
    (h_point : (Real.sin 215) = (Real.sin α) ∧ (Real.cos 215) = (Real.cos α)) :
    α = 235 :=
sorry

end find_alpha_l46_46506


namespace determine_Q_neg_half_l46_46662

variable (R : Type) [CommRing R] [CommRing R] 
variable {n : ℕ}
variable (P Q : Polynomial R)

theorem determine_Q_neg_half 
  (h_degP : P.degree ≤ n)
  (h_degQ : Q.degree ≤ n)
  (h_identity : ∀ x : R, P.eval x * x^(n + 1) + Q.eval x * (x + 1)^(n + 1) = 1) :
  Q.eval (-1 / 2) = 2^n :=
sorry

end determine_Q_neg_half_l46_46662


namespace correct_propositions_l46_46668

-- Definitions of the conditions
def prop1 (p q : Prop) : Prop := ¬(p ∧ q) → ¬p ∧ ¬q
def regression_property (x : Real) : Real := -0.5 * x + 3
def abs_sin_period (x : Real) : Real := abs (sin (x + 1))

-- Propositions
def proposition1 (p q : Prop) : Prop := prop1 p q
def proposition2 : Prop := ∀ x, regression_property (x + 1) - regression_property x = -0.5
def proposition3 : Prop := ∃ T, ∀ x, abs_sin_period (x + T) = abs_sin_period x

-- Lean 4 statement
theorem correct_propositions :
  proposition1 p q = False ∧
  proposition2 = True ∧
  proposition3 = True :=
by sorry

end correct_propositions_l46_46668


namespace geom_seq_sum_of_terms_l46_46574

theorem geom_seq_sum_of_terms
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geometric: ∀ n, a (n + 1) = a n * q)
  (h_q : q = 2)
  (h_sum : a 0 + a 1 + a 2 = 21)
  (h_pos : ∀ n, a n > 0) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end geom_seq_sum_of_terms_l46_46574


namespace rectangular_prism_faces_edges_vertices_sum_l46_46331

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46331


namespace find_m_range_l46_46565

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)
variable (m : R)

-- Define that the function f is monotonically increasing
def monotonically_increasing (f : R → R) : Prop :=
  ∀ ⦃x y : R⦄, x ≤ y → f x ≤ f y

-- Lean statement for the proof problem
theorem find_m_range (h1 : monotonically_increasing f) (h2 : f (2 * m - 3) > f (-m)) : m > 1 :=
by
  sorry

end find_m_range_l46_46565


namespace triangle_inequality_power_sum_l46_46639

theorem triangle_inequality_power_sum
  (a b c : ℝ) (n : ℕ)
  (h_a_bc : a + b + c = 1)
  (h_a_b_c : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_a_triangl : a + b > c)
  (h_b_triangl : b + c > a)
  (h_c_triangl : c + a > b)
  (h_n : n > 1) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + (2^(1/n : ℝ)) / 2 :=
by
  sorry

end triangle_inequality_power_sum_l46_46639


namespace S_bounds_l46_46302

noncomputable def S (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, 
    x k / if k = 0 then real.sqrt (∑ i in Finset.range n, x i) 
           else real.sqrt ((∑ i in Finset.range k, x i + 1) * (∑ i in Finset.range (n - k), x (k + i)))

theorem S_bounds (x : ℕ → ℝ) (n : ℕ) (h_pos : ∀ i < n, x i > 0) (h_sum : ∑ i in Finset.range n, x i = 1) :
  1 ≤ S x n ∧ S x n < real.pi / 2 :=
by
  sorry

end S_bounds_l46_46302


namespace domain_of_sqrt_function_l46_46664

theorem domain_of_sqrt_function :
  {x : ℝ | 0 ≤ x + 1} = {x : ℝ | -1 ≤ x} :=
by {
  sorry
}

end domain_of_sqrt_function_l46_46664


namespace max_distance_from_P_to_ABC_l46_46121

noncomputable def sphereO_surface_area : ℝ := 36 * Real.pi

noncomputable def tetrahedron_vertices : Prop :=
  ∃ (P A B C : ℝ × ℝ × ℝ), 
    let s := 3 in
    let radius_of_ABC_circumcircle := 1 in
    let radius_of_sphereO := 3 in
    let center_O := (0, 0, 0) in
    let side_length_of_ABC := Real.sqrt 3 in
    -- Conditions
    (P ∈ set_of_points_on_sphere center_O radius_of_sphereO) ∧
    (A ∈ set_of_points_on_sphere center_O radius_of_sphereO) ∧
    (B ∈ set_of_points_on_sphere center_O radius_of_sphereO) ∧
    (C ∈ set_of_points_on_sphere center_O radius_of_sphereO) ∧
    (distance_in_plane A B = side_length_of_ABC) ∧
    (distance_in_plane A C = side_length_of_ABC) ∧
    (distance_in_plane B C = side_length_of_ABC)

-- Define a function for the computation of distances in 3D space
noncomputable def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2 + (a.3 - b.3) ^ 2)

-- Define a function for the computation of distances in the plane
noncomputable def distance_in_plane (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (ax, ay, _) := a in
  let (bx, by, _) := b in
  Real.sqrt ((ax - bx) ^ 2 + (ay - by) ^ 2)

-- Statement of the maximum distance problem
theorem max_distance_from_P_to_ABC : tetrahedron_vertices → max_distance P to plane ABC = 3 + 2 * Real.sqrt 2 := by
  sorry

end max_distance_from_P_to_ABC_l46_46121


namespace min_a_plus_b_l46_46230

theorem min_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -145 := sorry

end min_a_plus_b_l46_46230


namespace two_solutions_to_congruence_count_solutions_to_congruence_l46_46510

theorem two_solutions_to_congruence (x : ℕ) (h1 : x > 0) (h2 : x < 70) :
  (x + 20) % 26 = 45 % 26 → x = 25 ∨ x = 51 :=
by sorry

theorem count_solutions_to_congruence :
  (finset.univ.filter (λ x, (x + 20) % 26 = 45 % 26 ∧ x > 0 ∧ x < 70)).card = 2 :=
by sorry

end two_solutions_to_congruence_count_solutions_to_congruence_l46_46510


namespace total_combinations_l46_46815

def varieties_of_wrapping_paper : Nat := 10
def colors_of_ribbon : Nat := 4
def types_of_gift_cards : Nat := 5
def kinds_of_decorative_stickers : Nat := 2

theorem total_combinations : varieties_of_wrapping_paper * colors_of_ribbon * types_of_gift_cards * kinds_of_decorative_stickers = 400 := by
  sorry

end total_combinations_l46_46815


namespace sin_identity_l46_46934

noncomputable def cos_rule {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) : ℝ :=
  (b^2 + c^2 - a^2) / (2 * b * c)

theorem sin_identity
  (a b c : ℝ) 
  (ha : a = 4) 
  (hb : b = 5) 
  (hc : c = 6) 
  (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (sin (A + B) / sin (2 * A)) = 1 :=
by
  sorry

end sin_identity_l46_46934


namespace truck_needs_additional_gallons_l46_46425

-- Definitions based on the given conditions
def miles_per_gallon : ℝ := 3
def total_miles_needed : ℝ := 90
def current_gallons : ℝ := 12

-- Function to calculate the additional gallons needed
def additional_gallons_needed (mpg : ℝ) (total_miles : ℝ) (current_gas : ℝ) : ℝ :=
  (total_miles - current_gas * mpg) / mpg

-- The main theorem to prove
theorem truck_needs_additional_gallons :
  additional_gallons_needed miles_per_gallon total_miles_needed current_gallons = 18 := 
by
  sorry

end truck_needs_additional_gallons_l46_46425


namespace reconstruct_phone_number_l46_46395

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def consecutive_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 3 ∧
  digits.nth 0 + 1 = digits.nth 1 ∧
  digits.nth 1 + 1 = digits.nth 2

def three_consecutive_ones (n: ℕ) : Prop :=
  let digits := n.digits 10 in
  let seq := [1, 1, 1] in
  ∃ k, seq = (digits.drop k).take 3

def one_two_digit_prime (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  let two_digit_numbers := [digits.take 2, digits.drop 3].map (λ l, l.foldl (λ b a, b * 10 + a) 0) in
  two_digit_numbers.any Nat.Prime

def construct_phone_number : ℕ :=
  7111765

theorem reconstruct_phone_number :
  let phone_number := construct_phone_number in
  phone_number.digits 10.length = 7 ∧
  consecutive_digits ((phone_number.digits 10).drop 4 ← 0).take 3) ∧
  is_palindrome ((phone_number.digits 10).take 5 ← 0) ∧
  Nat.sum (phone_number.digits 10).take 3 % 9 = 0 ∧
  three_consecutive_ones phone_number ∧
  one_two_digit_prime phone_number :=
by
  sorry

end reconstruct_phone_number_l46_46395


namespace number_of_valid_x0_l46_46902

-- Define the sequence according to the problem condition.
def sequence (x0 : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x0 else
  let rec aux : ℕ → ℝ → ℝ
    | 0, x => x
    | n, x => if 2 * x < 1 then aux (n - 1) (2 * x) else aux (n - 1) (2 * x - 1)
  in aux n x0

theorem number_of_valid_x0 : 
  {x0 : ℝ // 0 ≤ x0 ∧ x0 < 1 ∧ sequence x0 10 = x0}.card = 1023 :=
sorry

end number_of_valid_x0_l46_46902


namespace ratio_new_breadth_to_original_l46_46671

-- Define the original and new dimensions of the rectangle with the conditions
variables (L B B' : ℝ)

-- Define the original area A and new area A'
def original_area := L * B
def new_area := (L / 2) * B'

-- The percentage change in area is given as 50% less
def area_condition := new_area = 0.5 * original_area

-- Statement to prove the ratio of the new breadth to the original breadth
theorem ratio_new_breadth_to_original (L B B' : ℝ) (h : area_condition L B B') :
  B' / B = 0.5 :=
sorry

end ratio_new_breadth_to_original_l46_46671


namespace Mortdecai_egg_donation_l46_46188

theorem Mortdecai_egg_donation :
  let collected_dozen := 8 * 2 in
  let delivered_dozen := 3 + 5 in
  let remaining_after_delivery := collected_dozen - delivered_dozen in
  let baked_dozen := 4 in
  let remaining_after_baking := remaining_after_delivery - baked_dozen in
  let dozen_to_eggs := 12 in
  remaining_after_baking * dozen_to_eggs = 48 :=
by
  intros
  sorry

end Mortdecai_egg_donation_l46_46188


namespace seven_power_product_prime_count_l46_46632

theorem seven_power_product_prime_count (n : ℕ) :
  ∃ primes: List ℕ, (∀ p ∈ primes, Prime p) ∧ primes.prod = 7^(7^n) + 1 ∧ primes.length ≥ 2*n + 3 :=
by
  sorry

end seven_power_product_prime_count_l46_46632


namespace sum_faces_edges_vertices_l46_46352

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46352


namespace fireworks_display_fireworks_proof_l46_46421

theorem fireworks_display (x : ℕ) : 
  (6 * 4 + 12 * x + 50 * 8 = 484) ↔ (x = 5) :=
by
  simp only [Nat.mul_add, Nat.add_sub_associative]

theorem fireworks_proof : Exists x, 6 * 4 + 12 * x + 50 * 8 = 484 :=
begin
  use 5,
  sorry
end

end fireworks_display_fireworks_proof_l46_46421


namespace federal_guideline_daily_minimum_l46_46021

def total_cups_sunday_to_thursday : ℕ := 8
def daily_requirement_cups : ℕ := 3
def days_sunday_to_thursday : ℕ := 5
def total_required_cups : ℕ := daily_requirement_cups * days_sunday_to_thursday

theorem federal_guideline_daily_minimum 
    (total_cups_sunday_to_thursday = 8) 
    (daily_requirement_cups = 3) 
    (days_sunday_to_thursday = 5):
    daily_requirement_cups = 3 := by
  sorry

end federal_guideline_daily_minimum_l46_46021


namespace fraction_of_rotten_is_one_third_l46_46592

def total_berries (blueberries cranberries raspberries : Nat) : Nat :=
  blueberries + cranberries + raspberries

def fresh_berries (berries_to_sell berries_to_keep : Nat) : Nat :=
  berries_to_sell + berries_to_keep

def rotten_berries (total fresh : Nat) : Nat :=
  total - fresh

def fraction_rot (rotten total : Nat) : Rat :=
  (rotten : Rat) / (total : Rat)

theorem fraction_of_rotten_is_one_third :
  ∀ (blueberries cranberries raspberries berries_to_sell : Nat),
    blueberries = 30 →
    cranberries = 20 →
    raspberries = 10 →
    berries_to_sell = 20 →
    fraction_rot (rotten_berries (total_berries blueberries cranberries raspberries) 
                  (fresh_berries berries_to_sell berries_to_sell))
                  (total_berries blueberries cranberries raspberries) = 1 / 3 :=
by
  intros blueberries cranberries raspberries berries_to_sell
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end fraction_of_rotten_is_one_third_l46_46592


namespace correct_multiplication_result_l46_46751

theorem correct_multiplication_result :
  ∃ x : ℕ, (x * 9 = 153) ∧ (x * 6 = 102) :=
by
  sorry

end correct_multiplication_result_l46_46751


namespace distance_from_point_to_line_circle_tangent_to_line_distance_between_parallel_lines_l46_46535

/-
Problem 1:
Prove that the distance from point P(2,-3) to the line y = -x + 3 is 2√2.
-/
theorem distance_from_point_to_line (x₀ y₀ k b : ℝ) (h_point: x₀ = 2 ∧ y₀ = -3) (h_line: k = -1 ∧ b = 3) :
  let d := (abs (k * x₀ - y₀ + b)) / (sqrt (1 + k^2))
  in d = 2 * sqrt 2 := by
  sorry

/-
Problem 2:
Prove that the circle Q with center (0,5) and radius 2 is tangent to the line y = √3x + 9.
-/
theorem circle_tangent_to_line (x₀ y₀ r k b : ℝ) 
  (h_center: x₀ = 0 ∧ y₀ = 5 ∧ r = 2) (h_line: k = sqrt 3 ∧ b = 9) :
  let d := (abs (k * x₀ - y₀ + b)) / (sqrt (1 + k^2))
  in d = r := by
  sorry

/-
Problem 3:
Prove that the distance between the lines y = -3x - 2 and y = -3x + 6 is 4√10 / 5.
-/
theorem distance_between_parallel_lines (k : ℝ) (b₁ b₂ : ℝ) (h_parallel: k = -3 ∧ b₁ = -2 ∧ b₂ = 6) :
  let d := abs (b₂ - b₁) / (sqrt (1 + k^2))
  in d = 4 * sqrt 10 / 5 := by
  sorry

end distance_from_point_to_line_circle_tangent_to_line_distance_between_parallel_lines_l46_46535


namespace complement_of_M_in_U_l46_46541

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U : (U \ M) = {2, 4, 6} :=
by
  sorry

end complement_of_M_in_U_l46_46541


namespace range_of_x_l46_46100

variable {x : ℝ}

def P := x^2 - 2 * x - 3 ≥ 0
def Q := abs (1 - x / 2) < 1

theorem range_of_x (hP : P) (hQ_false : ¬ Q) : x ≥ 4 ∨ x ≤ -1 :=
sorry

end range_of_x_l46_46100


namespace apples_used_l46_46305

theorem apples_used (x : ℕ) 
  (initial_apples : ℕ := 23) 
  (bought_apples : ℕ := 6) 
  (final_apples : ℕ := 9) 
  (h : (initial_apples - x) + bought_apples = final_apples) : 
  x = 20 :=
by
  sorry

end apples_used_l46_46305


namespace functional_equation_l46_46022

noncomputable def f {p : ℕ → ℕ} (x : ℚ) : ℚ :=
  if (∃ k, x = p (2 * k)) then 1 / p (2 * (exists_unique.intro (λ k h, and.left h)))
  else if (∃ k, x = p (2 * k + 1)) then 1 / p (2 * k) else 0

theorem functional_equation (x y : ℚ) (p : ℕ → ℕ) (h_p_distinct: ∀ i j, i ≠ j → p i ≠ p j) :
  ∀ (x y : ℚ), f(p, x * f(p, y)) = f(p, x) / y := sorry

end functional_equation_l46_46022


namespace cosine_of_45_degrees_l46_46451

theorem cosine_of_45_degrees : Real.cos (π / 4) = √2 / 2 := by
  sorry

end cosine_of_45_degrees_l46_46451


namespace min_perimeter_l46_46317

theorem min_perimeter :
  ∃ (a b c : ℕ), 
  (2 * a + 18 * c = 2 * b + 20 * c) ∧ 
  (9 * Real.sqrt (a^2 - (9 * c)^2) = 10 * Real.sqrt (b^2 - (10 * c)^2)) ∧ 
  (10 * (a - b) = 9 * c) ∧ 
  2 * a + 18 * c = 362 := 
sorry

end min_perimeter_l46_46317


namespace length_of_BC_l46_46937

theorem length_of_BC (AB : ℝ) (A B C : ℝ) (h1 : C = (B - A) * (sqrt 5 - 1) / 2) (h2 : AB = 4) : AB * (sqrt 5 - 1) / 2 = 2 * sqrt 5 - 2 :=
by
  -- Given points A, B, and C, with C being the golden section point such that AC < BC and AB = 4,
  -- we need to prove that BC = 2 * sqrt 5 - 2.
  sorry

end length_of_BC_l46_46937


namespace probability_part_1_probability_part_2_l46_46688

-- Definition of car models and types
inductive CarType : Type
| A | B | C

-- Labels for each type
def label : CarType -> ℕ
| CarType.A => [1, 2, 3]
| CarType.B => [1, 2]
| CarType.C => [0]

-- Combinations for the first scenario
def valid_combinations_1 : List (CarType × ℕ × CarType × ℕ) := [
    (CarType.A, 1, CarType.B, 1),
    (CarType.A, 1, CarType.B, 2),
    (CarType.A, 2, CarType.B, 1)
]

-- Combinations for the second scenario
def valid_combinations_2 : List (CarType × ℕ × CarType × ℕ) := [
    (CarType.A, 1, CarType.B, 1),
    (CarType.A, 1, CarType.B, 2),
    (CarType.A, 2, CarType.B, 1),
    (CarType.A, 1, CarType.C, 0),
    (CarType.A, 2, CarType.C, 0),
    (CarType.A, 3, CarType.C, 0),
    (CarType.B, 1, CarType.C, 0),
    (CarType.B, 2, CarType.C, 0)
]

-- Theorem for the first result
theorem probability_part_1 :
    (valid_combinations_1.length : ℚ) / (Nat.choose 5 2 : ℚ) = 3 / 10 :=
  by sorry

-- Theorem for the second result
theorem probability_part_2 :
    (valid_combinations_2.length : ℚ) / (Nat.choose 6 2 : ℚ) = 8 / 15 :=
  by sorry

end probability_part_1_probability_part_2_l46_46688


namespace triangle_circumradius_condition_l46_46202

theorem triangle_circumradius_condition (AC AB : ℝ) (A : ℝ)
  (R : ℝ) (hA : A = 60) (hAC : AC = 1) 
  (hR : R ≤ 1) :
  1 / 2 < AB ∧ AB < 2 := 
begin
  sorry
end

end triangle_circumradius_condition_l46_46202


namespace largest_four_digit_sum_20_l46_46715

theorem largest_four_digit_sum_20 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n.digits 10).sum = 20 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m.digits 10).sum = 20 → n ≥ m :=
begin
  sorry
end

end largest_four_digit_sum_20_l46_46715


namespace arithmetic_sequence_40th_term_diff_l46_46827

theorem arithmetic_sequence_40th_term_diff :
  ∀ (a : ℕ → ℚ), (∀ n, a n ≤ 90) ∧ (∀ n, a n ≥ 20) ∧ (∑ i in finset.range 150, a i) = 9000 
  ∧ (a 40 = 60 - 111 * d ∨ a 40 = 60 + 111 * d) → 
  ∃ G L, G - L = (6660 : ℚ) / 149 := 
sorry

end arithmetic_sequence_40th_term_diff_l46_46827


namespace general_formula_expression_of_k_l46_46503

noncomputable def sequence_a : ℕ → ℤ
| 0     => 0 
| 1     => 0 
| 2     => -6
| n + 2 => 2 * (sequence_a (n + 1)) - (sequence_a n)

theorem general_formula :
  ∀ n, sequence_a n = 2 * n - 10 := sorry

def sequence_k : ℕ → ℕ
| 0     => 0 
| 1     => 8 
| n + 1 => 3 * 2 ^ n + 5

theorem expression_of_k (n : ℕ) :
  sequence_k (n + 1) = 3 * 2 ^ n + 5 := sorry

end general_formula_expression_of_k_l46_46503


namespace find_y_l46_46897

open Real

def vecV (y : ℝ) : ℝ × ℝ := (1, y)
def vecW : ℝ × ℝ := (6, 4)

noncomputable def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dotProduct v w) / (dotProduct w w)
  (scalar * w.1, scalar * w.2)

theorem find_y (y : ℝ) (h : projection (vecV y) vecW = (3, 2)) : y = 5 := by
  sorry

end find_y_l46_46897


namespace polynomial_factorization_l46_46734

theorem polynomial_factorization : 
  (x : ℤ) → x ^ 12 + x ^ 6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) :=
by {
  intro x,
  sorry -- Proof omitted
}

end polynomial_factorization_l46_46734


namespace cos_double_angle_l46_46930

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
sorry

end cos_double_angle_l46_46930


namespace ribbons_at_start_l46_46626

theorem ribbons_at_start (morning_ribbons : ℕ) (afternoon_ribbons : ℕ) (left_ribbons : ℕ)
  (h_morning : morning_ribbons = 14) (h_afternoon : afternoon_ribbons = 16) (h_left : left_ribbons = 8) :
  morning_ribbons + afternoon_ribbons + left_ribbons = 38 :=
by
  sorry

end ribbons_at_start_l46_46626


namespace arithmetic_geometric_sequence_ratio_l46_46923

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
variable (a_n : ℕ → ℝ) 

-- a_n is an arithmetic-geometric sequence
-- S_n represents the sum of the first n terms
-- Given: 27a_{3} - a_{6} = 0
-- To Prove: (S_{6} / S_{3}) = 28

theorem arithmetic_geometric_sequence_ratio (H1 : ∀ n, a n = a 1 * q ^ (n-1))
  (H2 : ∀ n, S n = finset.sum (finset.range n) (λ k, a k))
  (H3 : 27 * a 3 - a 6 = 0) :
  (S 6 / S 3) = 28 :=
sorry

end arithmetic_geometric_sequence_ratio_l46_46923


namespace total_cost_tom_has_to_pay_l46_46694

theorem total_cost_tom_has_to_pay :
  let vaccine_cost := 45
  let number_of_vaccines := 10
  let doctor_visit_cost := 250
  let insurance_coverage := 0.80
  let trip_cost := 1200
  let medical_expenses := number_of_vaccines * vaccine_cost + doctor_visit_cost
  let insurance_paid := insurance_coverage * medical_expenses
  let tom_pays := trip_cost + (medical_expenses - insurance_paid)
  tom_pays = 1340 :=
by
  let vaccine_cost := 45
  let number_of_vaccines := 10
  let doctor_visit_cost := 250
  let insurance_coverage := 0.80
  let trip_cost := 1200
  let medical_expenses := number_of_vaccines * vaccine_cost + doctor_visit_cost
  let insurance_paid := insurance_coverage * medical_expenses
  let tom_pays := trip_cost + (medical_expenses - insurance_paid)
  show tom_pays = 1340 from sorry

end total_cost_tom_has_to_pay_l46_46694


namespace f_increasing_max_b_g_positive_ln_2_approx_l46_46527

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) - 2

-- State that f is increasing on ℝ
theorem f_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y := sorry

-- Define the function g with parameter b
def g (b : ℝ) (x : ℝ) : ℝ := f (2 * x) - 4 * b * f x

-- Maximum value of b such that g(x) > 0 for x > 0
theorem max_b_g_positive : ∃ b : ℝ, (∀ x : ℝ, x > 0 → g b x > 0) ∧ (∀ b' : ℝ, (∀ x : ℝ, x > 0 → g b' x > 0) → b' ≤ 2) := sorry

-- Estimate of ln 2 given sqrt(2) bounds
theorem ln_2_approx : (1.4142 < Real.sqrt 2 ∧ Real.sqrt 2 < 1.4143) → 0.692 ≤ Real.log 2 ∧ Real.log 2 ≤ 0.694 := sorry

end f_increasing_max_b_g_positive_ln_2_approx_l46_46527


namespace afternoon_eggs_correct_l46_46243

-- Define the given conditions
def total_eggs_used : ℕ := 1339
def eggs_used_in_morning : ℕ := 816

-- Define the number of eggs used in the afternoon
def eggs_used_in_afternoon : ℕ := total_eggs_used - eggs_used_in_morning

-- The theorem to prove the number of eggs used in the afternoon
theorem afternoon_eggs_correct : eggs_used_in_afternoon = 523 :=
by
  simp [eggs_used_in_afternoon, total_eggs_used, eggs_used_in_morning]
  sorry

end afternoon_eggs_correct_l46_46243


namespace find_n_divisible_by_11_l46_46386

theorem find_n_divisible_by_11 : ∃ n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 :=
by
  use 1
  -- proof steps would go here, but we're only asked for the statement
  sorry

end find_n_divisible_by_11_l46_46386


namespace length_AD_is_two_thirds_l46_46591

noncomputable def length_of_AD
  (A : ℝ) (c : ℝ) (area : ℝ) (ratio : ℝ) (b : ℝ)
  (lengths : ℝ × ℝ × ℝ) (cos_B : ℝ) : ℝ :=
let D := by sorry in
let BC := by sorry in
let AD := by sorry in
AD

theorem length_AD_is_two_thirds :
  let A := (2 * Real.pi) / 3,
      c := 2,
      area := Real.sqrt 3 / 2,
      b := 1,
      ratio := 2 in
  (length_of_AD A c area ratio b (4, 1, Real.sqrt 7) (5 / (2 * Real.sqrt 7))) = 2 / 3 :=
by sorry

end length_AD_is_two_thirds_l46_46591


namespace no_universal_card_survivor_l46_46685

def cards := Finset.range 1000

theorem no_universal_card_survivor :
  ¬ ∃ A ∈ cards, ∀ (init_card ∈ cards), init_card ≠ A →
    let sequence := remove_process init_card in sequence.last = A :=
sorry

def remove_process (init_card : ℕ) : list ℕ :=
sorry

end no_universal_card_survivor_l46_46685


namespace no_real_solution_l46_46257

theorem no_real_solution (x y : ℝ) : x^3 + y^2 = 2 → x^2 + x * y + y^2 - y = 0 → false := 
by 
  intro h1 h2
  sorry

end no_real_solution_l46_46257


namespace complex_number_modulus_and_argument_l46_46891

noncomputable def z : ℂ := -complex.sin (real.pi / 8) - complex.i * complex.cos (real.pi / 8)

theorem complex_number_modulus_and_argument :
  (complex.abs z = 1) ∧ (complex.arg z = -5 * real.pi / 8) := by
  sorry

end complex_number_modulus_and_argument_l46_46891


namespace probability_of_factors_less_than_9_is_7_over_16_l46_46323

/-- Define the positive factors of 120 and count the number of factors less than 9 -/
def positive_factors_120 : Finset ℕ := 
  {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}

/-- Define the factors of 120 that are less than 9 -/
def factors_less_than_9 : Finset ℕ := 
  {1, 2, 3, 4, 5, 6, 8}

/-- The total number of positive factors of 120 -/
def num_positive_factors_120 : ℕ := positive_factors_120.card

/-- The number of factors less than 9 -/
def num_factors_less_than_9 : ℕ := factors_less_than_9.card

/-- The probability that a randomly drawn positive factor of 120 is less than 9 -/
def probability_factors_less_than_9 : ℚ := 
  num_factors_less_than_9 / num_positive_factors_120

/-- The main theorem to prove the probability is 7/16 -/
theorem probability_of_factors_less_than_9_is_7_over_16 :
  probability_factors_less_than_9 = 7 / 16 := 
begin
  sorry
end

end probability_of_factors_less_than_9_is_7_over_16_l46_46323


namespace parametric_eqn_line_l_PA_PB_product_value_l46_46199

-- Define given conditions
def circleC_param (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 4 * Real.sin θ)

def P := (1, 2 : ℝ)

def α := Real.pi / 6

def line_l_param (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

-- Prove first part
theorem parametric_eqn_line_l :
  ∃ t : ℝ, (∀ θ : ℝ, (circleC_param θ).fst = (line_l_param t).fst ∧ (circleC_param θ).snd = (line_l_param t).snd) :=
  sorry

-- Prove second part
theorem PA_PB_product_value :
  let A := ((1 + (Real.sqrt 3 / 2) * t1), (2 + (1 / 2) * t1))
  let B := ((1 + (Real.sqrt 3 / 2) * t2), (2 + (1 / 2) * t2))
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  PA * PB = 11 :=
  sorry

end parametric_eqn_line_l_PA_PB_product_value_l46_46199


namespace counters_coincide_l46_46247

def counter := (ℤ × ℤ)

variable (A B C D : counter)

-- Conditions
def can_displace (m n : ℕ) (c : counter) : counter :=
   let (x1, y1) := m;
   let (x2, y2) := n;
   let (x3, y3) := c;
   (x3 + (x2 - x1), y3 + (y2 - y1))

-- Assert the theorem
theorem counters_coincide (A B C D : counter) :
  ∃ (f : ℕ → ℕ → counter → counter), 
  ∃ (n : ℕ),
    (∃ (m1 m2 : ℕ),
    f m1 m2 A = B) ∨ 
    (∃ (m1 m2 : ℕ),
    f m1 m2 B = A) :=
sorry

end counters_coincide_l46_46247


namespace rational_sqrt_of_rational_xy_l46_46600

theorem rational_sqrt_of_rational_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) :
  ∃ k : ℚ, k^2 = 1 - x * y := 
sorry

end rational_sqrt_of_rational_xy_l46_46600


namespace main_problem_l46_46203

def arithmetic_sequence (a : ℕ → ℕ) : Prop := ∃ a₁ d, ∀ n, a (n + 1) = a₁ + n * d

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2

def another_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop := ∀ n, b n = 1 / (a n * a (n + 1))

theorem main_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : a_3 = 5) 
  (h2 : S_3 = 9) 
  (h3 : arithmetic_sequence a)
  (h4 : sequence_sum a S)
  (h5 : another_sequence b a) : 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n = n / (2 * n + 1)) := sorry

end main_problem_l46_46203


namespace floor_add_example_l46_46854

def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_add_example :
  floor (-3.73) + floor (1.4) = -3 :=
by
  sorry

end floor_add_example_l46_46854


namespace integer_satisfaction_l46_46876

noncomputable def sigma (n : ℕ) : ℕ := nat.divisors n |> List.sum

def p (n : ℕ) : ℕ := nat.factors n |> List.maximum' (by simp)

theorem integer_satisfaction (n : ℕ) (h_n_ge_2 : n ≥ 2) (h_eq : sigma n / (p n - 1) = n) : n = 6 :=
sorry

end integer_satisfaction_l46_46876


namespace people_could_not_take_bus_l46_46764

theorem people_could_not_take_bus
  (carrying_capacity : ℕ)
  (first_pickup_ratio : ℚ)
  (first_pickup_people : ℕ)
  (people_waiting : ℕ)
  (total_on_bus : ℕ)
  (additional_can_carry : ℕ)
  (people_could_not_take : ℕ)
  (h1 : carrying_capacity = 80)
  (h2 : first_pickup_ratio = 3/5)
  (h3 : first_pickup_people = carrying_capacity * first_pickup_ratio.to_nat)
  (h4 : first_pickup_people = 48)
  (h5 : total_on_bus = first_pickup_people)
  (h6 : additional_can_carry = carrying_capacity - total_on_bus)
  (h7 : additional_can_carry = 32)
  (h8 : people_waiting = 50)
  (h9 : people_could_not_take = people_waiting - additional_can_carry)
  (h10 : people_could_not_take = 18) : 
  people_could_not_take = 18 :=
by
  sorry -- proof is left for another step

end people_could_not_take_bus_l46_46764


namespace katherine_has_4_apples_l46_46596

variable (A P : ℕ)

theorem katherine_has_4_apples
  (h1 : P = 3 * A)
  (h2 : A + P = 16) :
  A = 4 := 
sorry

end katherine_has_4_apples_l46_46596


namespace determine_AC_squared_l46_46696

-- Define the data points and conditions given in the problem
variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables {x1 x2 : A} (BC AC : B) (AB r : C)

-- Given conditions
variables (BD DC : D) (hBD : BD = 20) (hDC : DC = 16)

-- Triangle ABC is right-angled at A
variable (right_angle : is_right_angle ABC A)

-- Circle centered at A with radius AB intersects BC at D and AC at E

-- Goal: Prove that AC^2 = 936
theorem determine_AC_squared (hBC : BC = BD + DC) (h_circle: is_circle_centered A AB):
  AC^2 = 936 :=
by
  -- Proof omitted
  sorry

end determine_AC_squared_l46_46696


namespace quadratic_has_real_roots_l46_46485

theorem quadratic_has_real_roots (k : ℝ) : (k * x^2 - 4 * x + 2 = 0) → (k ≤ 2 ∧ k ≠ 0) :=
by
  -- Consider the discriminant condition for real roots: Δ = b^2 - 4ac ≥ 0, where a = k, b = -4, c = 2
  let Δ := (-4)^2 - 4 * k * 2
  have h1 : Δ ≥ 0 → 16 - 8 * k ≥ 0 := sorry,
  have h2 : 16 - 8 * k ≥ 0 → k ≤ 2 := sorry,
  have h3 : k ≠ 0,
  exact ⟨h2 h1, h3⟩

end quadratic_has_real_roots_l46_46485


namespace trapezoid_bc_length_l46_46082

theorem trapezoid_bc_length
  (A B C D M : Point)
  (d : Real)
  (h_trapezoid : IsTrapezoid A B C D)
  (h_M_on_AB : OnLine M A B)
  (h_DM_perp_AB : Perpendicular D M A B)
  (h_MC_eq_CD : Distance M C = Distance C D)
  (h_AD_eq_d : Distance A D = d) :
  Distance B C = d / 2 := by
  sorry

end trapezoid_bc_length_l46_46082


namespace soccer_score_combinations_l46_46177

theorem soccer_score_combinations :
  ∃ (x y z : ℕ), x + y + z = 14 ∧ 3 * x + y = 19 ∧ x + y + z ≥ 0 ∧ 
    ({ (3, 10, 1), (4, 7, 3), (5, 4, 5), (6, 1, 7) } = 
      { (x, y, z) | x + y + z = 14 ∧ 3 * x + y = 19 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 }) :=
by 
  sorry

end soccer_score_combinations_l46_46177


namespace percentage_decrease_in_breadth_l46_46443

-- Define the conditions
variables (L B : ℝ) (x : ℝ)
def L' := 1.40 * L
def B' := B * (1 - x / 100)
def A := L * B
def A' := L' * B'

-- Proof statement
theorem percentage_decrease_in_breadth :
  A' = 1.05 * A → x = 25 :=
by
  sorry

end percentage_decrease_in_breadth_l46_46443


namespace sum_of_a_is_9_33_l46_46880

noncomputable def sum_of_a : Real :=
  -- Conditions of the problem
  let equation_holds (a x : Real) : Prop :=
    (4 * Real.pi * a + Real.arcsin (Real.sin x) + 3 * Real.arccos (Real.cos x) - a * x) / (2 + Real.tan x ^ 2) = 0

  -- Function to check the number of solutions for a given a
  let has_three_solutions (a : Real) : Prop :=
    ∃ x1 x2 x3 : Real, 0 ≤ x1 ∧ x1 < Real.pi ∧ 0 ≤ x2 ∧ x2 < Real.pi ∧ 0 ≤ x3 ∧ x3 < Real.pi ∧
                      x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ 
                      equation_holds a x1 ∧ equation_holds a x2 ∧ equation_holds a x3
  
  -- Generate all positive a's satisfying the condition and sum them
  -- a values are known directly from the solution analysis done.
  let valid_as : List Real := [1, 3, 16 / 3]

  -- Sum the values and round to nearest hundredth
  (valid_as.map (λa, Real.round_to (Real.to_float a) 2)).sum

theorem sum_of_a_is_9_33 : sum_of_a = 9.33 :=
  sorry

end sum_of_a_is_9_33_l46_46880


namespace proof_statement_l46_46378

noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

noncomputable def problem_statement : ℕ :=
  (nat.sqrt (factorial 6 * factorial 4)) ^ 4

theorem proof_statement : problem_statement = 298598400 := 
by 
  -- Proof should go here
  sorry

end proof_statement_l46_46378


namespace sum_of_two_digit_integers_l46_46327

theorem sum_of_two_digit_integers :
  let a := 10
  let l := 99
  let d := 1
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = 4905 :=
by
  sorry

end sum_of_two_digit_integers_l46_46327


namespace max_sum_x_eq_n_sq_minus_n_l46_46501

theorem max_sum_x_eq_n_sq_minus_n (n : ℕ) (h1 : 1 < n) (x : Fin n.succ → ℝ) 
  (h2 : ∀ i, i ∈ Fin n.succ → (0 ≤ x i ∧ x i ≤ n)) 
  (h3 : ∏ i in Finset.finRange n.succ, x i = ∏ i in Finset.finRange n.succ, (n - x i)) :
  ∑ i in Finset.finRange n.succ, x i ≤ n^2 - n := 
sorry

end max_sum_x_eq_n_sq_minus_n_l46_46501


namespace find_f3_minus_f4_l46_46651

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = - f x
axiom h_periodic : ∀ x : ℝ, f (x + 5) = f x
axiom h_f1 : f 1 = 1
axiom h_f2 : f 2 = 2

theorem find_f3_minus_f4 : f 3 - f 4 = -1 := by
  sorry

end find_f3_minus_f4_l46_46651


namespace relationship_l46_46935

def a : ℝ := 4 ^ 0.3
def b : ℝ := 8 ^ (1/4)
def c : ℝ := 3 ^ 0.75

theorem relationship among a b c : a < b ∧ b < c :=
by sorry

end relationship_l46_46935


namespace circle_eq_of_points_value_of_m_l46_46936

-- Define the points on the circle
def P : ℝ × ℝ := (0, -4)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (3, -1)

-- Statement 1: The equation of the circle passing through P, Q, and R
theorem circle_eq_of_points (C : ℝ × ℝ → Prop) :
  (C P ∧ C Q ∧ C R) ↔ ∀ x y : ℝ, C (x, y) ↔ (x - 1)^2 + (y + 2)^2 = 5 := sorry

-- Define the line intersecting the circle and the chord length condition |AB| = 4
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

-- Statement 2: The value of m such that the chord length |AB| is 4
theorem value_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) → m = 4 / 3 := sorry

end circle_eq_of_points_value_of_m_l46_46936


namespace simplify_and_evaluate_expr_l46_46641

noncomputable def expr (x : Real) : Real :=
  (1 / (x^2 + 2 * x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1))

theorem simplify_and_evaluate_expr :
  let x := 2 * Real.sqrt 5 - 1 in
  expr x = Real.sqrt 5 / 10 := by
  sorry

end simplify_and_evaluate_expr_l46_46641


namespace alpha_range_l46_46522

theorem alpha_range (f : ℝ → ℝ) (x_0 α : ℝ) (hx0 : 0 < x_0 ∧ x_0 < 1) (hα : 0 < α ∧ α < π / 2)
  (hf : ∀ x, f x = log x + tan α)
  (hf'_x0_eq_f_x0 : deriv f x_0 = f x_0) :
  α ∈ Ioo (π / 4) (π / 2) :=
sorry

end alpha_range_l46_46522


namespace go_stones_perimeter_count_l46_46274

def stones_per_side : ℕ := 6
def sides_of_square : ℕ := 4
def corner_stones : ℕ := 4

theorem go_stones_perimeter_count :
  (stones_per_side * sides_of_square) - corner_stones = 20 := 
by
  sorry

end go_stones_perimeter_count_l46_46274


namespace sin_sum_less_than_sum_of_sins_l46_46933

theorem sin_sum_less_than_sum_of_sins (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : 
  let a := sin (α + β)
  let b := sin α + sin β 
  in a < b :=
by 
  sorry

end sin_sum_less_than_sum_of_sins_l46_46933


namespace inspector_rejects_l46_46435

theorem inspector_rejects (examined_meters rejected_percentage : ℝ) (rejected_meters : ℝ) :
  examined_meters = 66.67 →
  rejected_percentage = 0.15 →
  rejected_meters = (rejected_percentage * examined_meters) →
  rejected_meters = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  calc
    0.15 * 66.67 = 9.999 ...  : by norm_num -- Simplifies to 10
    ... = 10                 : by ring
  sorry

end inspector_rejects_l46_46435


namespace set_intersection_l46_46978

def M := {x : ℝ | x^2 > 4}
def N := {x : ℝ | 1 < x ∧ x ≤ 3}
def complement_M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def intersection := N ∩ complement_M

theorem set_intersection : intersection = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end set_intersection_l46_46978


namespace permutations_remainder_l46_46222

def permutations_count (str : String) : Nat := 
  -- Function that counts the number of valid permutations, implementation is omitted.
  sorry

theorem permutations_remainder :
  let str := "AAAABBBBBCCCCCDDDD"
  let N := permutations_count str
  N % 1000 = 755 :=
begin
  sorry
end

end permutations_remainder_l46_46222


namespace Ah_tribe_count_l46_46265

-- Define the total number of inhabitants
variables (p : ℕ)

-- Define tribe membership predicates
variables (is_Ah : (ℕ → Prop))
variables (is_Uh : (ℕ → Prop))

-- Define the statements made by the first, second, and third person
variables (s1 s2 s3 : Prop)

-- Define the conditions
variables (C1 : s1 → p ≤ 16) (C2 : s1 → ∀ n, is_Uh n)
variables (C3 : s2 → p ≤ 17) (C4 : s2 → ∃ n, is_Ah n)
variables (C5 : s3 → p = 5) (C6 : s3 → ∃ m n o, is_Uh m ∧ is_Uh n ∧ is_Uh o)

-- Define the truth-telling and lying condition for members
variables (Ah_truth : ∀ n, is_Ah n → (∀ (stmt : Prop), stmt → stmt))
variables (Uh_lie : ∀ n, is_Uh n → (∀ (stmt : Prop), stmt → ¬stmt))

-- Define the goal to prove
theorem Ah_tribe_count : ∃ (a : ℕ), a = 15 :=
by
  have : ∃ a, a + 2 = 17 := sorry
  use 15
  sorry

end Ah_tribe_count_l46_46265


namespace projection_eq_self_l46_46481

variables (v w : ℝ × ℝ)
def v := (4, -6)
def w := (-12, 18)

theorem projection_eq_self : (v.proj w) = v :=
sorry

end projection_eq_self_l46_46481


namespace transformed_ellipse_area_l46_46138

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![2, 3]]
noncomputable def matrixB : Matrix (Fin 2) (Fin 2) ℚ := ![![(-3/2 : ℚ), 2], ![1, (-1 : ℚ)]]

def ellipse (x y : ℚ) : Prop := x^2 / 4 + y^2 = 1

theorem transformed_ellipse_area :
  let AB := matrixA ⬝ matrixB in
  (AB = ![![1/2, 0], ![0, 1]]) →
  (∀ (x y : ℚ), ellipse x y →
    let x' := 2 * x in
    let y' := y in
    (x' ^ 2 + y' ^ 2 = 1)) →
  ( ∀ (F : ℚ → ℚ → Prop), (∀ x y, F x y ↔ (x ^ 2 + y ^ 2 = 1)) → (let area := π in (area = π))) :=
by sorry

end transformed_ellipse_area_l46_46138


namespace comics_sold_l46_46638

theorem comics_sold (initial_count sold_count remaining_count : ℕ) 
  (h1 : initial_count = 90)
  (h2 : remaining_count = 25)
  (h3 : sold_count = initial_count - remaining_count) : 
  sold_count = 65 := 
by 
  rw [h1, h2] 
  simp 
  exact h3

end comics_sold_l46_46638


namespace area_of_field_l46_46413

noncomputable def area_square_field (speed_kmh : ℕ) (time_min : ℕ) : ℝ :=
  let speed_m_per_min := (speed_kmh * 1000) / 60
  let distance := speed_m_per_min * time_min
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

-- Given conditions
theorem area_of_field : area_square_field 4 3 = 20000 := by
  sorry

end area_of_field_l46_46413


namespace minimum_steps_to_catch_thief_l46_46830

-- Definitions of positions A, B, C, D, etc., along the board
-- Assuming the positions and movement rules are predefined somewhere in the environment.
-- For a simple abstract model, we assume the following:
-- The positions are nodes in a graph, and each move is one step along the edges of this graph.

def Position : Type := String -- This can be refined to reflect the actual chessboard structure.
def neighbor (p1 p2 : Position) : Prop := sorry -- Predicate defining that p1 and p2 are neighbors.

-- Positions are predefined for simplicity.
def A : Position := "A"
def B : Position := "B"
def C : Position := "C"
def D : Position := "D"
def F : Position := "F"

-- Condition: policeman and thief take turns moving, starting with the policeman.
-- Initial positions of the policeman and the thief.
def policemanStart : Position := A
def thiefStart : Position := B

-- Statement: Prove that the policeman can catch the thief in a minimum of 4 moves.
theorem minimum_steps_to_catch_thief (policeman thief : Position) (turns : ℕ) :
  policeman = policemanStart →
  thief = thiefStart →
  (∀ t < turns, (neighbor policeman thief)) →
  (turns = 4) :=
sorry

end minimum_steps_to_catch_thief_l46_46830


namespace calculate_bridge_length_l46_46672

-- Define the conditions
def train_length : ℕ := 125
def train_speed_kmh : ℕ := 45
def crossing_time_sec : ℕ := 30

-- Convert speed from km/hr to m/s
def speed_m_per_s (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

-- Calculate the total distance covered in 30 seconds
def distance_covered (speed : ℚ) (time : ℕ) : ℚ := speed * time

-- Calculate the bridge length
def bridge_length (total_distance : ℚ) (train_length : ℕ) : ℚ := total_distance - train_length

-- The actual proof goal
theorem calculate_bridge_length :
  let speed := speed_m_per_s train_speed_kmh in
  let total_distance := distance_covered speed crossing_time_sec in
  bridge_length total_distance train_length = 250 := by
  sorry

end calculate_bridge_length_l46_46672


namespace purely_imaginary_condition_l46_46748

variables (a : ℝ)
noncomputable def i : ℂ := complex.I

theorem purely_imaginary_condition : (∀ a : ℝ, (1 - 2 * i) * (a + i)).re = 0 → a = -2 :=
begin
  intro h,
  -- We would provide the proof here to show that a = -2 when the real part of the product is zero.
  sorry
end

end purely_imaginary_condition_l46_46748


namespace largest_expression_l46_46746

-- Defining all the necessary expressions
def tg (x : ℝ) : ℝ := Real.tan x
def ctg (x : ℝ) : ℝ := 1 / Real.tan x
def sin (x : ℝ) : ℝ := Real.sin x
def cos (x : ℝ) : ℝ := Real.cos x

-- Defining the expressions at 48 degrees
noncomputable def tg_48 : ℝ := tg (48 * Real.pi / 180)
noncomputable def ctg_48 : ℝ := ctg (48 * Real.pi / 180)
noncomputable def sin_48 : ℝ := sin (48 * Real.pi / 180)
noncomputable def cos_48 : ℝ := cos (48 * Real.pi / 180)

-- The expressions to compare
noncomputable def expression_A : ℝ := tg_48 + ctg_48
noncomputable def expression_B : ℝ := sin_48 + cos_48
noncomputable def expression_C : ℝ := tg_48 + cos_48
noncomputable def expression_D : ℝ := ctg_48 + sin_48

-- The theorem stating that expression_A is the largest
theorem largest_expression : 
    expression_A > expression_B ∧ 
    expression_A > expression_C ∧ 
    expression_A > expression_D := 
by 
  sorry

end largest_expression_l46_46746


namespace total_beds_in_hotel_l46_46836

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end total_beds_in_hotel_l46_46836


namespace line_joining_complex_eqn_l46_46289

def complex_product (a b : ℂ) : ℂ := a * b

theorem line_joining_complex_eqn 
  (a b : ℂ)
  (u v : ℂ)
  (hu : u = -1 + 2 * complex.I)
  (hv : v = 2 + 2 * complex.I)
  (h_eq : ∀ (z : ℂ), a * z + b * complex.conj(z) = 8) :
  complex_product a b = 1 := 
sorry

end line_joining_complex_eqn_l46_46289


namespace operation_value_l46_46300

variable (a b : ℤ)

theorem operation_value (h : (21 - 1) * (9 - 1) = 160) : a = 21 :=
by
  sorry

end operation_value_l46_46300


namespace find_f_8_l46_46290

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom f_6_eq_8 : f(6) = 8

theorem find_f_8 : f(8) = 32 / 3 := 
by 
  sorry

end find_f_8_l46_46290


namespace no_nonconstant_prime_polynomial_l46_46635

open Polynomial

theorem no_nonconstant_prime_polynomial (P : Polynomial ℤ) (h1 : ¬ degree P = 0)
    (h2 : ∀ n : ℕ, is_prime (P.eval n)) : false :=
by sorry

end no_nonconstant_prime_polynomial_l46_46635


namespace sum_of_prime_values_of_h_l46_46484

theorem sum_of_prime_values_of_h (n : ℕ) (h : ℕ → ℤ) (prime_values_sum : ℤ) :
  (∀ n, n > 0 → h n = n^4 - 500 * n^2 + 625) →
  (prime_values_sum = ∑ p in (finset.filter (λ x, is_prime x) (finset.range (n + 1))), h p) →
  prime_values_sum = 0 :=
by
  sorry

end sum_of_prime_values_of_h_l46_46484


namespace find_x0_l46_46141

-- Definitions based on the problem
def A := Set.Ico 0 1
def B := Set.Icc 1 2

noncomputable def f : ℝ → ℝ := λ x, if x ∈ A then x + 0.5 else 2 * (1 - x)

-- Statement of the problem
theorem find_x0 (x0 : ℝ) (hx0A : x0 ∈ A) (hfA : f (f x0) ∈ A) : x0 = 0.5 :=
sorry

end find_x0_l46_46141


namespace original_class_size_l46_46614

/-- Let A be the average age of the original adult class, which is 40 years. -/
def A : ℕ := 40

/-- Let B be the average age of the 8 new students, which is 32 years. -/
def B : ℕ := 32

/-- Let C be the decreased average age of the class after the new students join, which is 36 years. -/
def C : ℕ := 36

/-- The original number of students in the adult class is N. -/
def N : ℕ := 8

/-- The equation representing the total age of the class after the new students join. -/
theorem original_class_size :
  (A * N) + (B * 8) = C * (N + 8) ↔ N = 8 := by
  sorry

end original_class_size_l46_46614


namespace solve_inequalities_l46_46980

theorem solve_inequalities (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 3 → x - a < 1 ∧ x - 2 * b > 3) ↔ (a = 2 ∧ b = -2) := 
  by 
    sorry

end solve_inequalities_l46_46980


namespace gcf_75_90_l46_46706

theorem gcf_75_90 : Nat.gcd 75 90 = 15 :=
by
  sorry

end gcf_75_90_l46_46706


namespace units_digit_of_x_l46_46055

theorem units_digit_of_x 
  (a x : ℕ) 
  (h1 : a * x = 14^8) 
  (h2 : a % 10 = 9) : 
  x % 10 = 4 := 
by 
  sorry

end units_digit_of_x_l46_46055


namespace f_plus_2012_odd_l46_46852

def f : ℝ → ℝ → ℝ := sorry

lemma f_property (α β : ℝ) : f α β = 2012 := sorry

theorem f_plus_2012_odd : ∀ x : ℝ, f (-x) + 2012 = -(f x + 2012) :=
by
  sorry

end f_plus_2012_odd_l46_46852


namespace binomial_expansion_coefficient_l46_46461

theorem binomial_expansion_coefficient :
  let n := 7
  let a := 2
  let b := 1 / (Real.sqrt 1)
  ∑ i in range (n+1), (Nat.choose n i) * (a ^ i) * (b ^ (n - i)) = (280 : ℕ) := 
by
  sorry

end binomial_expansion_coefficient_l46_46461


namespace number_of_oranges_l46_46811

variable (bananas : ℕ) (oranges : ℕ)
variable (percentGoodOranges percentGoodBananas percentGoodFruits : ℝ)

def conditions := 
  (bananas = 400) ∧ 
  (percentGoodOranges = 0.85) ∧ 
  (percentGoodBananas = 0.92) ∧ 
  (percentGoodFruits = 0.878)

theorem number_of_oranges (h : conditions bananas oranges percentGoodOranges percentGoodBananas percentGoodFruits) : 
  oranges = 629 :=
by
  sorry

end number_of_oranges_l46_46811


namespace trajectory_of_P_is_ellipse_l46_46075

/-- Given a tetrahedron S-ABC with point P within the side face SBC that is 
perpendicular to the base face ABC, if the distance from the moving point P 
to the base face ABC is equal to its distance to point S, then the trajectory 
of P within the side face SBC is a part of an ellipse. -/
theorem trajectory_of_P_is_ellipse (S A B C P : Point) (ABC SBC : Plane) 
  (h1 : P ∈ SBC) (h2 : Perpendicular P ABC) (h3 : Distance P ABC = Distance P S):
  is_ellipse (trajectory P SBC) :=
sorry

end trajectory_of_P_is_ellipse_l46_46075


namespace all_points_in_circle_of_radius_1_l46_46405

open Set

theorem all_points_in_circle_of_radius_1
  (S : Set (ℝ × ℝ))
  (h : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S →
    ∃ (center : ℝ × ℝ), ∃ (r : ℝ), r = 1 ∧ ∀ P ∈ {A, B, C}, (dist center P) ≤ r) :
  ∃ (center : ℝ × ℝ), ∃ R : ℝ, R ≤ 1 ∧ ∀ P ∈ S, (dist center P) ≤ R := 
sorry

end all_points_in_circle_of_radius_1_l46_46405


namespace sum_of_solutions_l46_46721

theorem sum_of_solutions (x : ℝ) :
  (4 * x + 6) * (3 * x - 12) = 0 → (x = -3 / 2 ∨ x = 4) →
  (-3 / 2 + 4) = 5 / 2 :=
by
  intros Hsol Hsols
  sorry

end sum_of_solutions_l46_46721


namespace range_of_a_l46_46273

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0)
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l46_46273


namespace length_AB_l46_46981

variables {x y m : ℝ}

-- Define the circles
def O1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 5}
def O2 : set (ℝ × ℝ) := {p | (p.1 + m)^2 + p.2^2 = 20}

-- Define the points of intersection A and B
def intersects_at_A_and_B (A B : ℝ × ℝ) : Prop :=
  A ∈ O1 ∧ A ∈ O2 ∧ B ∈ O1 ∧ B ∈ O2

-- Define the tangents being perpendicular
def tangents_perpendicular_at_A (A : ℝ × ℝ) : Prop :=
  let O1_to_A := (A.1, A.2)
  let O2_to_A := (A.1 + m, A.2) in
  O1_to_A.1 * O2_to_A.1 + O1_to_A.2 * O2_to_A.2 = 0

-- Theorem statement
theorem length_AB {A B : ℝ × ℝ} (h_intersect : intersects_at_A_and_B A B)
  (h_tangent_perp : tangents_perpendicular_at_A A) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

end length_AB_l46_46981


namespace factorization_count_is_correct_l46_46192

noncomputable def count_factorizations (n : Nat) (k : Nat) : Nat :=
  (Nat.choose (n + k - 1) (k - 1))

noncomputable def factor_count : Nat :=
  let alpha_count := count_factorizations 6 3
  let beta_count := count_factorizations 6 3
  let total_count := alpha_count * beta_count
  let unordered_factorizations := total_count - 15 * 3 - 1
  1 + 15 + unordered_factorizations / 6

theorem factorization_count_is_correct :
  factor_count = 139 := by
  sorry

end factorization_count_is_correct_l46_46192


namespace solve_for_x_l46_46268

noncomputable def numerator : ℝ := Real.sqrt (8^2 + 15^2)
noncomputable def denominator : ℝ := Real.sqrt (25 + 36)
noncomputable def x : ℝ := numerator / denominator

theorem solve_for_x : x = 17 * Real.sqrt 61 / 61 :=
by
  unfold numerator denominator x
  sorry

end solve_for_x_l46_46268


namespace plane_passing_through_point_and_parallel_plane_passing_through_point_and_parallel_l46_46027

theorem plane_passing_through_point_and_parallel (
  (x y z : ℝ) (h : x = 2 ∧ y = -1 ∧ z = 3) : 
  (3 * x + 2 * y - 4 * z + 8 = 0)) : 
  (∀ (A B C D : ℤ), A = 3 ∧ B = 2 ∧ C = -4 ∧ A * x + B * y + C * z + D = 0 :=
begin 
  sorry 
end 

theorem plane_passing_through_point_and_parallel : true :=
begin
  sorry
end

end plane_passing_through_point_and_parallel_plane_passing_through_point_and_parallel_l46_46027


namespace BC_length_l46_46090

variable (A B C D M : Type)
variable [IsTrapezoid A B C D] [IsPointOnLateralSide M A B]
variable (DM_perpendicular_AB : Perpendicular D M A B)
variable (MC_eq_CD : MC = CD)
variable (AD_eq_d : AD = d)

theorem BC_length (d : ℝ) (A B C D M : Point)
  (H1 : IsTrapezoid A B C D)
  (H2 : IsPointOnLateralSide M A B)
  (H3 : Perpendicular D M A B)
  (H4 : MC = CD)
  (H5 : AD = d) :
  BC = d / 2 :=
sorry

end BC_length_l46_46090


namespace sum_faces_edges_vertices_eq_26_l46_46343

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46343


namespace mortdecai_donates_eggs_l46_46186

constant collects_eggs : Nat := 8 * 2  -- dozen eggs
constant delivers_to_market : Nat := 3  -- dozen eggs
constant delivers_to_mall : Nat := 5  -- dozen eggs
constant makes_pie : Nat := 4  -- dozen eggs
constant dozen_to_eggs : Nat := 12  -- eggs per dozen

-- Prove that Mortdecai donates 48 eggs to the charity
theorem mortdecai_donates_eggs : 
  (collects_eggs - (delivers_to_market + delivers_to_mall) - makes_pie) * dozen_to_eggs = 48 := by
  sorry

end mortdecai_donates_eggs_l46_46186


namespace compute_g_sum_l46_46060

noncomputable def g (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - 5/12

theorem compute_g_sum : 
    ∑ k in Finset.range 2015, g ((k + 1) / 2016) = 2015 := 
    sorry

end compute_g_sum_l46_46060


namespace parabola_equation_l46_46532

theorem parabola_equation (a b p : ℝ) (hp : 0 < p) (ha : 0 < a) (hb : 0 < b)
    (h1 : ∀ x y : ℝ, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)
    (eccentricity : (Real.sqrt (a ^ 2 + b ^ 2)) / a = 2)
    (dist_focus_asymp : 2 = (abs (p / (2 * b))) / (Real.sqrt ((1 / a) ^ 2 + (1 / b) ^ 2))) :
    ∀ y : ℝ, y = (x : ℝ) ^ 2 / 16 :=
begin
  sorry
end

end parabola_equation_l46_46532


namespace parabola_intersect_l46_46517

theorem parabola_intersect (b c m p q x1 x2 : ℝ)
  (h_intersect1 : x1^2 + b * x1 + c = 0)
  (h_intersect2 : x2^2 + b * x2 + c = 0)
  (h_order : m < x1)
  (h_middle : x1 < x2)
  (h_range : x2 < m + 1)
  (h_valm : p = m^2 + b * m + c)
  (h_valm1 : q = (m + 1)^2 + b * (m + 1) + c) :
  p < 1 / 4 ∧ q < 1 / 4 :=
sorry

end parabola_intersect_l46_46517


namespace average_daily_visitors_l46_46776

theorem average_daily_visitors
    (avg_sun : ℕ)
    (avg_other : ℕ)
    (days : ℕ)
    (starts_sun : Bool)
    (H1 : avg_sun = 630)
    (H2 : avg_other = 240)
    (H3 : days = 30)
    (H4 : starts_sun = true) :
    (5 * avg_sun + 25 * avg_other) / days = 305 :=
by
  sorry

end average_daily_visitors_l46_46776


namespace lcm_48_180_eq_720_l46_46049

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l46_46049


namespace part_a_part_b_l46_46733

namespace TrihedralAngle

-- Part (a)
theorem part_a (α β γ : ℝ) (h1 : β = 70) (h2 : γ = 100) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    30 < α ∧ α < 170 := 
sorry

-- Part (b)
theorem part_b (α β γ : ℝ) (h1 : β = 130) (h2 : γ = 150) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    20 < α ∧ α < 80 := 
sorry

end TrihedralAngle

end part_a_part_b_l46_46733


namespace pyramid_vol_and_lat_surf_l46_46657

noncomputable def pyramid_properties (m : ℝ) : ℝ × ℝ :=
let vol := (m^3 * real.sqrt2^0.25) / 6
let lat_surf := (m^2 * (2 + real.sqrt2)) / 2
(vol, lat_surf)

theorem pyramid_vol_and_lat_surf {m : ℝ} (h : 0 < m) :
  ∃ vol lat_surf : ℝ, 
    (vol, lat_surf) = (m^3 * real.sqrt2^0.25 / 6, m^2 * (2 + real.sqrt2) / 2) :=
by
  use pyramid_properties m
  sorry

end pyramid_vol_and_lat_surf_l46_46657


namespace parallel_lines_distance_l46_46982

theorem parallel_lines_distance (a b : ℝ) 
  (hl1_par : ∀ (x y : ℝ), a * x + 2 * y + b = 0) 
  (hl2_par : ∀ (x y : ℝ), (a - 1) * x + y + b = 0) 
  (hl_parallel : ∀ (x1 y1 x2 y2 : ℝ), hl1_par x1 y1 → hl2_par x2 y2 → (a * x1 + 2 * y1) / sqrt(5) = (a - 1) * x2 + y2 + b / sqrt(5)) 
  (hl_distance : distance_between_lines hl1_par hl2_par = sqrt(2) / 2) :
  a * b = 4 ∨ a * b = -4 :=
by
  sorry

end parallel_lines_distance_l46_46982


namespace intersection_A_B_l46_46620

def A : Set ℝ := {x | log 2 x < 1}
def B : Set ℝ := {y | ∃ x : ℝ, y = 3^x - 1}

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x < 2} := 
sorry

end intersection_A_B_l46_46620


namespace additional_plates_added_l46_46429

def initial_plates : ℕ := 27
def added_plates : ℕ := 37
def total_plates : ℕ := 83

theorem additional_plates_added :
  total_plates - (initial_plates + added_plates) = 19 :=
by
  sorry

end additional_plates_added_l46_46429


namespace sunday_dogs_count_l46_46782

-- Define initial conditions
def initial_dogs : ℕ := 2
def monday_dogs : ℕ := 3
def total_dogs : ℕ := 10
def sunday_dogs (S : ℕ) : Prop :=
  initial_dogs + S + monday_dogs = total_dogs

-- State the theorem
theorem sunday_dogs_count : ∃ S : ℕ, sunday_dogs S ∧ S = 5 := by
  sorry

end sunday_dogs_count_l46_46782


namespace soda_cost_proof_l46_46820

theorem soda_cost_proof (b s : ℤ) (h1 : 4 * b + 3 * s = 440) (h2 : 3 * b + 2 * s = 310) : s = 80 :=
by
  sorry

end soda_cost_proof_l46_46820


namespace line_parallel_y_intercept_l46_46946

theorem line_parallel_y_intercept (m n : ℝ) (h1 : ∃ k : ℝ, mx + n*y + 2 = 0 = k*(x - 2*y + 5 = 0)) (h2 : y_intercept(mx + n*y + 2 = 0) = (0,1)) :
  (m = 1) ∧ (n = -2) := sorry

end line_parallel_y_intercept_l46_46946


namespace x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l46_46104

theorem x_is_sufficient_but_not_necessary_for_x_squared_eq_one : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
by
  sorry

end x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l46_46104


namespace angle_YOZ_in_incircle_triangle_l46_46589

theorem angle_YOZ_in_incircle_triangle (X Y Z O : Type) [Intriangle X Y Z O] (angle_XYZ : ∠ XYZ = 72) (angle_YXZ : ∠ YXZ = 55) : 
  ∠ YOZ = 26.5 := 
sorry

end angle_YOZ_in_incircle_triangle_l46_46589


namespace correct_product_of_a_and_b_l46_46189

theorem correct_product_of_a_and_b (a a' b : ℕ) (h1 : a' = nat.reverse a) 
  (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ a' ∧ a' < 100) (h4 : a' * b + 2 = 240) 
  (h5 : 0 < b) : a * b = 301 :=
sorry

end correct_product_of_a_and_b_l46_46189


namespace brian_cards_after_waine_takes_l46_46839

-- Define the conditions
def brian_initial_cards : ℕ := 76
def wayne_takes_away : ℕ := 59

-- Define the expected result
def brian_remaining_cards : ℕ := 17

-- The statement of the proof problem
theorem brian_cards_after_waine_takes : brian_initial_cards - wayne_takes_away = brian_remaining_cards := 
by 
-- the proof would be provided here 
sorry

end brian_cards_after_waine_takes_l46_46839


namespace range_of_f_l46_46942

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * sin (ω * x - π / 6)

def range_f (x : ℝ) (cond : 0 ≤ x ∧ x ≤ π / 2) (ω_pos : ω > 0) (ω_eq_2 : ω = 2) : set ℝ :=
  set_of (λ y, ∃ x ∈ Icc 0 (π / 2), f ω x = y)

theorem range_of_f : 
  range_f x (by split; linarith) (by linarith) (by norm_num : ω = 2) = Icc (-3 / 2) 3 :=
sorry

end range_of_f_l46_46942


namespace instantaneous_rate_of_change_of_area_with_respect_to_perimeter_at_3cm_l46_46158

open Real

theorem instantaneous_rate_of_change_of_area_with_respect_to_perimeter_at_3cm :
  let s := 3
  let P := 4 * s
  let A := s^2
  let dAdP := deriv (fun P => (1 / 16) * P^2) P
  P = 12 →
  dAdP = 3 / 2 :=
by
  intros
  rw [←h]
  have s_eq := show 3 = s by rfl
  have P_eq := show 12 = 4 * s by rw [s_eq]; norm_num
  rw [P_eq] at this
  exact this

end instantaneous_rate_of_change_of_area_with_respect_to_perimeter_at_3cm_l46_46158


namespace triangle_angle_bisector_ratio_l46_46169

theorem triangle_angle_bisector_ratio 
  (a b c : ℝ) (A B C : ℝ) (x y : ℝ) (AD BC : ℝ)
  (h_triangle: AD ∈ triangle ABC) 
  (h_angle_bisector : angle_bisector AD A)
  (h_meets_D : meets_at D BC AD)
  (h_x_eq_CD : x = length_of_segment CD) 
  (h_y_eq_BD : y = length_of_segment BD)
  (h_perpendicular : is_perpendicular AD BC) :
  x / y = c / b := 
sorry

end triangle_angle_bisector_ratio_l46_46169


namespace find_coordinates_of_P_l46_46781

-- Define the coordinates of the vertex and the focus
def V : Prod ℝ ℝ := (-2, 3)
def F : Prod ℝ ℝ := (-2, 4)

-- Define the distance condition
def PF {P : Prod ℝ ℝ} : ℝ := Real.sqrt ((P.fst + 2)^2 + (P.snd - 4)^2)

-- Define the parabola equation condition
def parabola (P : Prod ℝ ℝ) : Prop := (P.fst + 2)^2 = 4 * (P.snd - 3)

-- Define the point P satisfying all conditions
def P : Prod ℝ ℝ := (48, 628)

-- Define the first quadrant condition
def first_quadrant (P : Prod ℝ ℝ) : Prop := P.fst > 0 ∧ P.snd > 0

-- Theorem stating the coordinates of point P
theorem find_coordinates_of_P :
  PF P = 51 ∧ parabola P ∧ first_quadrant P → P = (48, 628) :=
by
  sorry

end find_coordinates_of_P_l46_46781


namespace sum_of_faces_edges_vertices_l46_46368

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46368


namespace cube_painting_ways_l46_46152

theorem cube_painting_ways :
  ∃ (colors : Finset ℕ) (cube_faces : Finset (Fin 6)),
    colors.card = 6 ∧
    ∀ face1 face2 ∈ cube_faces, face1 ≠ face2 → face_has_different_color face1 face2 →
      count_distinct_coloring_ways colors cube_faces = 230 :=
begin
  sorry
end

end cube_painting_ways_l46_46152


namespace ABC_concyclic_l46_46583

theorem ABC_concyclic (ABC : Triangle) (A B C E F M N D : Point) 
  (h1 : ∠ E A B = ∠ A C B) (h2 : ∠ C A F = ∠ A B C)
  (h3 : E = midpoint(A M)) (h4 : F = midpoint(A N))
  (h5 : Line.through(B M)) (h6 : Line.through(C N)) (h7 : M = intersection(B M))
  (h8 : N = intersection(C N)) (h9 : D = intersection(B M C N)) 
  (acute : ∀ A B C, acute_triangle A B C) :
  concyclic A B D C :=
by
  sorry

end ABC_concyclic_l46_46583


namespace gcd_1681_1705_l46_46704

theorem gcd_1681_1705 : Nat.gcd 1681 1705 = 1 := 
by 
  sorry

end gcd_1681_1705_l46_46704


namespace product_value_l46_46722

theorem product_value :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
    -- Skipping the actual proof
    sorry

end product_value_l46_46722


namespace f_inequality_l46_46958

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l46_46958


namespace silverware_probability_l46_46555

-- Define the contents of the drawer
def forks := 6
def spoons := 6
def knives := 6

-- Total number of pieces of silverware
def total_silverware := forks + spoons + knives

-- Combinations formula for choosing r items out of n
def choose (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Total number of ways to choose 3 pieces out of 18
def total_ways := choose total_silverware 3

-- Number of ways to choose 1 fork, 1 spoon, and 1 knife
def specific_ways := forks * spoons * knives

-- Calculated probability
def probability := specific_ways / total_ways

theorem silverware_probability : probability = 9 / 34 := 
  sorry
 
end silverware_probability_l46_46555


namespace upper_base_length_l46_46092

structure Trapezoid (A B C D M : Type) :=
  (on_lateral_side : ∀ {AB DM}, DM ⊥ AB)
  (perpendicular : ∀ {DM AB}, DM ⊥ AB)
  (equal_segments : MC = CD)
  (AD_length : AD = d)


theorem upper_base_length {A B C D M : Type} [trapezoid : Trapezoid A B C D M] :
  BC = d / 2 := 
begin
  sorry
end

end upper_base_length_l46_46092


namespace rectangular_prism_faces_edges_vertices_sum_l46_46334

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46334


namespace proof_problem_l46_46847

def problem_statement := 
  let m : ℕ := 2022 in 
  ⌊ (2023^3 / (2021 * 2022) - (2021^3 / (2022 * 2023)) ) ⌋ = 8

theorem proof_problem : problem_statement :=
by sorry

end proof_problem_l46_46847


namespace sum_of_faces_edges_vertices_l46_46375

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46375


namespace true_proposition_l46_46102

def proposition_p : Prop :=
  ∃ x : ℝ, Real.exp x = 0.1

def line_l1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x - a * y = 0

def line_l2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, 2 * x + a * y - 1 = 0

def proposition_q : Prop :=
  line_l1 (sqrt 2) ∧ line_l2 (sqrt 2)

theorem true_proposition : proposition_p ∧ ¬ proposition_q := by
  sorry

end true_proposition_l46_46102


namespace smallest_y_value_l46_46326

theorem smallest_y_value (y : ℝ) : (12 * y^2 - 56 * y + 48 = 0) → y = 2 :=
by
  sorry

end smallest_y_value_l46_46326


namespace invariant_lines_trajectory_of_Q_value_of_m_and_expressions_l46_46124

def complex_transformation (x y : ℝ) : ℝ × ℝ :=
  let m := Real.sqrt 3 in
  let x' := x + m * y in
  let y' := m * x - y in
  (x', y')

theorem invariant_lines :
  ∃ k b : ℝ, (k ≠ 0 ∧ (∀ x y : ℝ, y = k * x + b → let (x', y') := complex_transformation x y in y' = k * x' + b)) :=
begin
  sorry
end

theorem trajectory_of_Q (x y : ℝ) (hline : y = x + 1) : 
  let (x' q_y') := complex_transformation x y in 
  q_y' = (2 - Real.sqrt 3) * x' - 2 * Real.sqrt 3 + 2 :=
begin
  sorry
end

theorem value_of_m_and_expressions (x y : ℝ) :
  let m := Real.sqrt 3 in
  let (x', y') := complex_transformation x y in
  m = Real.sqrt 3 ∧ x' = x + m * y ∧ y' = m * x - y :=
begin
  sorry
end

end invariant_lines_trajectory_of_Q_value_of_m_and_expressions_l46_46124


namespace lcm_48_180_l46_46035

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l46_46035


namespace vertical_asymptote_exactly_one_l46_46006

def vertical_asymptotes (f : ℝ → ℝ) : Set ℝ := {x | ∃ ε > 0, ∀ δ > 0, ∃ y ∈ ball x δ, |f y| > 1/ε}

theorem vertical_asymptote_exactly_one :
  vertical_asymptotes (λ x : ℝ, (x + 2) / (x^2 - 2*x - 8)) = {4} :=
sorry

end vertical_asymptote_exactly_one_l46_46006


namespace length_upper_base_eq_half_d_l46_46078

variables {A B C D M: Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables {d : ℝ}

def trapezoid (A B C D : Type*) : Prop :=
  ∃ p : B, ∃ q : C, ∃ r : D, A ≠ p ∧ p ≠ q ∧ q ≠ r ∧ r ≠ A

def midpoint (A D : Type*) (N : Type*) (d : ℝ) : Prop :=
  dist A N = d / 2 ∧ dist N D = d / 2

axiom dm_perp_ab : ∀ (M : Type*), dist D M ∧ D ≠ M → dist M (id D) ≠ 0

axiom mc_eq_cd : dist M C = dist C D

theorem length_upper_base_eq_half_d
  (A B C D M : Type*)
  (h1 : trapezoid A B C D)
  (h2 : dist A D = d)
  (h3 : dm_perp_ab M)
  (h4 : mc_eq_cd) :
  dist B C = d / 2 :=
sorry

end length_upper_base_eq_half_d_l46_46078


namespace remainder_13_pow_51_mod_5_l46_46324

theorem remainder_13_pow_51_mod_5 : 13^51 % 5 = 2 := by
  sorry

end remainder_13_pow_51_mod_5_l46_46324


namespace monotonic_increase_intervals_g_min_max_l46_46524

-- Define the original function f
def f (x : ℝ) : ℝ := 1 + 2 * sqrt 3 * sin x * cos x - 2 * sin x ^ 2

-- Define the translated function g
def g (x : ℝ) : ℝ := f (x - π / 6)

-- Prove the intervals of monotonic increase for f
theorem monotonic_increase_intervals (k : ℤ) :
  ∀ x, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) ↔
  f.1 < f.2 → ∀ x y, x < y → f x < f y :=
sorry

-- Prove the minimum and maximum values of g in the specified interval
theorem g_min_max :
  ∀ x, (- π / 2 ≤ x ∧ x ≤ 0) →
  -2 ≤ g x ∧ g x ≤ 1 :=
sorry

end monotonic_increase_intervals_g_min_max_l46_46524


namespace rectangular_prism_faces_edges_vertices_sum_l46_46335

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46335


namespace num_150_ray_not_50_ray_partitional_l46_46604

def is_n_ray_partitional (R : set (ℝ × ℝ)) (Y : ℝ × ℝ) (n : ℕ) : Prop :=
  n ≥ 4 ∧ Y ∈ R ∧ ∃ (rays : set (ℝ × ℝ) → ℝ × ℝ → list (ℝ × ℝ)),
  (∀ i < n, ∃ S : set (set (ℝ × ℝ)), (Y, rays i) forms S) ∧ equal_area (S, n)

noncomputable def num_n_ray_partitional_points (R : set (ℝ × ℝ)) (n : ℕ) : ℕ := sorry

theorem num_150_ray_not_50_ray_partitional :
  let R := set.univ : set (ℝ × ℝ),
  num_n_ray_partitional_points R 150 - num_n_ray_partitional_points R 50 = 5000 :=
by sorry

end num_150_ray_not_50_ray_partitional_l46_46604


namespace distance_Xiaolan_to_Xiaohong_reverse_l46_46437

def Xiaohong_to_Xiaolan := 30
def Xiaolu_to_Xiaohong := 26
def Xiaolan_to_Xiaolu := 28

def total_perimeter : ℕ := Xiaohong_to_Xiaolan + Xiaolan_to_Xiaolu + Xiaolu_to_Xiaohong

theorem distance_Xiaolan_to_Xiaohong_reverse : total_perimeter - Xiaohong_to_Xiaolan = 54 :=
by
  rw [total_perimeter]
  norm_num
  sorry

end distance_Xiaolan_to_Xiaohong_reverse_l46_46437


namespace always_negative_product_when_exists_t_l46_46908

noncomputable def f (x : ℝ) (a m : ℝ) : ℝ :=
  (1/3) * x^3 - x^2 + a * x + m

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 2 * x + a

theorem always_negative_product_when_exists_t (a m : ℝ) (h_a : 0 < a) (h_a1 : a < 1) :
  (∃ t : ℝ, f' t a < 0) →
  ∀ t : ℝ, f' (t + 2) a * f' ((2 * t + 1) / 3) a < 0 :=
begin
  sorry
end

end always_negative_product_when_exists_t_l46_46908


namespace point_minimizing_sum_of_distances_l46_46052

theorem point_minimizing_sum_of_distances
    (A B C M : Point)
    (a b c : ℝ) (x y z : ℝ)
    (h1 : scalene_triangle A B C)
    (h2 : a = dist B C)
    (h3 : b = dist C A)
    (h4 : c = dist A B)
    (h5 : M ∈ interior_of_triangle A B C)
    (h6 : x = dist_from_point_to_line M (line_through B C))
    (h7 : y = dist_from_point_to_line M (line_through C A))
    (h8 : z = dist_from_point_to_line M (line_through A B)) :
    x + y + z = min_sum_distance ↔ M = vertex_opposite_largest_side A B C :=
sorry

end point_minimizing_sum_of_distances_l46_46052


namespace quadratic_real_roots_condition_l46_46487

open Real

theorem quadratic_real_roots_condition (k : ℝ) (h : k ≠ 0) :
  (let Δ := (-4)^2 - 4 * k * 2 in Δ ≥ 0) ↔ (k ≤ 2) :=
by
  -- We need to compute discriminant Δ and show this condition is equivalent to k ≤ 2
  let Δ := (-4)^2 - 4 * k * 2
  have : Δ = 16 - 8 * k,
  sorry
  -- Prove the inequality
  show Δ ≥ 0 ↔ k ≤ 2,
  sorry

end quadratic_real_roots_condition_l46_46487


namespace length_of_AD_l46_46747

theorem length_of_AD (A D B C M : Point) (h1 : trisect B C A D) (h2 : midpoint M A D) (h3 : distance M C = 10) :
  distance A D = 60 :=
sorry

end length_of_AD_l46_46747


namespace largest_number_sum13_product36_l46_46849

-- helper definitions for sum and product of digits
def sum_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.sum
def mul_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.foldr (· * ·) 1

theorem largest_number_sum13_product36 : 
  ∃ n : ℕ, sum_digits n = 13 ∧ mul_digits n = 36 ∧ ∀ m : ℕ, sum_digits m = 13 ∧ mul_digits m = 36 → m ≤ n :=
sorry

end largest_number_sum13_product36_l46_46849


namespace sum_of_faces_edges_vertices_l46_46371

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46371


namespace vehicle_count_expression_l46_46181

variable (C B M : ℕ)

-- Given conditions
axiom wheel_count : 4 * C + 2 * B + 2 * M = 196
axiom bike_to_motorcycle : B = 2 * M

-- Prove that the number of cars can be expressed in terms of the number of motorcycles
theorem vehicle_count_expression : C = (98 - 3 * M) / 2 :=
by
  sorry

end vehicle_count_expression_l46_46181


namespace area_of_EFGH_l46_46489

def shorter_side := 6
def ratio := 2
def longer_side := shorter_side * ratio
def width := 2 * longer_side
def length := shorter_side

theorem area_of_EFGH : length * width = 144 := by
  sorry

end area_of_EFGH_l46_46489


namespace angle_in_third_quadrant_l46_46151

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin (2 * α) > 0) (h2 : Real.cos α < 0) :  -- lean 4 detects max depth exception without statement
-- hence changing statements to Real.sin statements
  ( π / 2 < α ∧ α < π) : 
sorry

end angle_in_third_quadrant_l46_46151


namespace max_sum_products_cube_faces_l46_46430

theorem max_sum_products_cube_faces : 
  ∃ (a b c d e f : ℕ), (a + b + c + d + e + f = 21) ∧ 
    ({a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5, 6}) ∧ 
    ∀ (a' b' c' d' e' f' : ℕ), 
    (a' + b' + c' + d' + e' + f' = 21) ∧ 
    ({a', b', c', d', e', f'} ⊆ {1, 2, 3, 4, 5, 6}) →
    (a + b) * (c + d) * (e + f) ≥ (a' + b') * (c' + d') * (e' + f') :=
begin
  sorry
end

end max_sum_products_cube_faces_l46_46430


namespace tan_identity_l46_46056

theorem tan_identity :
  let θ1 := 10 * Real.pi / 180
      θ2 := 20 * Real.pi / 180
      θ3 := 60 * Real.pi / 180
      θ4 := 30 * Real.pi / 180 in
  (Real.tan θ1 * Real.tan θ2 + Real.tan θ2 * Real.tan θ3 + Real.tan θ3 * Real.tan θ1) = 1 :=
by
  have h1 : Real.tan θ4 = (Real.tan θ1 + Real.tan θ2) / (1 - Real.tan θ1 * Real.tan θ2),
  from Real.tan_add θ1 θ2,
  have h2 : Real.tan θ4 = Real.sqrt 3 / 3, -- known value of tan(30°)
  sorry,
  have h3 : Real.sqrt 3 * (Real.tan θ2 + Real.tan θ1) = 1 - Real.tan θ1 * Real.tan θ2,
  sorry,
  have h4 : Real.tan θ1 * Real.tan θ2 + Real.sqrt 3 * (Real.tan θ2 + Real.tan θ1) = 1,
  sorry,
  have h5 : Real.tan θ1 * Real.tan θ2 + Real.tan θ3 * (Real.tan θ2 + Real.tan θ1) = 1,
  from congr_arg (Real.tan θ3 *·) h4,
  sorry


end tan_identity_l46_46056


namespace example_proof_l46_46938

variables (p q : Prop)
variable (h₁ : p = true)
variable (h₂ : q = false)

theorem example_proof : ¬(p ∧ q) :=
begin
  sorry
end

end example_proof_l46_46938


namespace min_value_of_m_l46_46134

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_value_of_m {m : ℝ} (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ m) : m = 5 := 
sorry

end min_value_of_m_l46_46134


namespace cos_double_angle_neg_seven_over_twenty_five_l46_46490

theorem cos_double_angle_neg_seven_over_twenty_five 
  (α : ℝ) 
  (h1 : α ∈ (Real.pi / 2, Real.pi)) 
  (h2: Real.sin α + Real.cos α = 1/5) : 
  Real.cos (2 * α) = -7 / 25 := 
sorry

end cos_double_angle_neg_seven_over_twenty_five_l46_46490


namespace sin_B_value_l46_46167

theorem sin_B_value (A : ℝ) (AB BC : ℝ) (B C : ℝ)
    (hA : A = 120) (hAB : AB = 5) (hBC : BC = 7) 
    (triangle_sum : A + B + C = 180) :
    sin B = (3 * sqrt 3) / 14 := 
by
  sorry

end sin_B_value_l46_46167


namespace solution_set_xf_lt_zero_l46_46109

-- Define f : ℝ → ℝ, which is given to be an even function.
variable {f : ℝ → ℝ}
-- The condition f(2) = 0 when x > 0
def f_even (x : ℝ) : Prop := f x = f (-x)
axiom f_two_zero : f 2 = 0
axiom f_prime_condition : ∀ x > 0, (x * (f' x) - f x) / x^2 < 0

theorem solution_set_xf_lt_zero :
  { x : ℝ | x * f x < 0 } = { x : ℝ | (-2 < x ∧ x < 0) ∨ (2 < x) } :=
sorry

end solution_set_xf_lt_zero_l46_46109


namespace possible_values_of_a_l46_46974

/-- Given the function y = sqrt(x^2 - ax + 4), prove that the set of all possible values of a such that the function is monotonically decreasing on the interval [1, 2] is {4}. -/
theorem possible_values_of_a :
  {a : ℝ | ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 2 ∧ 1 ≤ x2 ∧ x2 ≤ 2 ∧ x1 < x2 → sqrt(x1^2 - a*x1 + 4) ≥ sqrt(x2^2 - a*x2 + 4)} = {4} :=
sorry

end possible_values_of_a_l46_46974


namespace find_a_l46_46529

-- Define the function f
def f (a x : ℝ) : ℝ := a*x^3 + 3*x^2 + 2

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 6*x

-- Define the slope of the tangent line at x = 1
def tangent_slope (a : ℝ) : ℝ := f' a 1

-- Define the slope of the given line x + 3y + 3 = 0
def given_line_slope : ℝ := -(1 / 3)

-- State the proof problem
theorem find_a :
  ∃ a : ℝ, (tangent_slope a) * given_line_slope = -1 ∧ a = -1 :=
by
  sorry

end find_a_l46_46529


namespace points_P_and_Q_harmonically_divide_AB_l46_46235

-- Define the arbitrary triangle
variables {A B C P Q: Type}

-- Define the conditions
variable [euclidean_geometry.Triangle A B C]

-- Define the angle bisectors CP (internal) and CQ (external)
-- Note: In Euclidean geometry, the notation for bisectors and segment conditions
-- might differ but we assume basic definitions provided here
variable [internal_angle_bisector C P A B]
variable [external_angle_bisector C Q A B]

-- State the main theorem
theorem points_P_and_Q_harmonically_divide_AB :
  cross_ratio A B P Q = -1 :=
sorry

end points_P_and_Q_harmonically_divide_AB_l46_46235


namespace inv_matrix_l46_46476

open Matrix
open_locale matrix

def mat_A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![2, -5, 0],
    ![-3, 6, 0],
    ![0, 0, 2]
  ]

def mat_I : Matrix (Fin 3) (Fin 3) ℚ :=
  1

def mat_N : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![-2, -5/3, 0],
    ![-1, -2/3, 0],
    ![0, 0, 1]
  ]

theorem inv_matrix :
  mat_N ⬝ mat_A = mat_I :=
by {
  sorry
}

end inv_matrix_l46_46476


namespace effective_weight_lowered_l46_46693

theorem effective_weight_lowered 
    (num_weight_plates : ℕ) 
    (weight_per_plate : ℝ) 
    (increase_percentage : ℝ) 
    (total_weight_without_technology : ℝ) 
    (additional_weight : ℝ) 
    (effective_weight_lowering : ℝ) 
    (h1 : num_weight_plates = 10)
    (h2 : weight_per_plate = 30)
    (h3 : increase_percentage = 0.20)
    (h4 : total_weight_without_technology = num_weight_plates * weight_per_plate)
    (h5 : additional_weight = increase_percentage * total_weight_without_technology)
    (h6 : effective_weight_lowering = total_weight_without_technology + additional_weight) :
    effective_weight_lowering = 360 := 
by
  sorry

end effective_weight_lowered_l46_46693


namespace Mortdecai_egg_donation_l46_46187

theorem Mortdecai_egg_donation :
  let collected_dozen := 8 * 2 in
  let delivered_dozen := 3 + 5 in
  let remaining_after_delivery := collected_dozen - delivered_dozen in
  let baked_dozen := 4 in
  let remaining_after_baking := remaining_after_delivery - baked_dozen in
  let dozen_to_eggs := 12 in
  remaining_after_baking * dozen_to_eggs = 48 :=
by
  intros
  sorry

end Mortdecai_egg_donation_l46_46187


namespace range_of_a_l46_46115

theorem range_of_a
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_deriv : ∀ x, f' x)
  (h_ineq1 : ∀ x, x ≤ 0 → (f x + x * f' x < 0))
  (h_ineq2 : ∀ θ, -π/2 ≤ θ ∧ θ ≤ π/2 → (|a + 1| * f (|a + 1|) ≥ sin θ * f (sin θ)))
  (a : ℝ) :
  a ≤ -2 ∨ a ≥ 0 :=
begin
  sorry
end

end range_of_a_l46_46115


namespace f_double_neg_one_l46_46130

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 else x + 1

theorem f_double_neg_one : f (f (-1)) = 2 :=
by
  sorry

end f_double_neg_one_l46_46130


namespace area_semicircles_percent_increase_l46_46796

noncomputable def radius_large_semicircle (length: ℝ) : ℝ := length / 2
noncomputable def radius_small_semicircle (width: ℝ) : ℝ := width / 2

noncomputable def area_semicircle (radius: ℝ) : ℝ := (real.pi * radius^2) / 2

theorem area_semicircles_percent_increase
  (length: ℝ) (width: ℝ)
  (h_length: length = 12) (h_width: width = 8) :
  let 
    large_radius := radius_large_semicircle length,
    small_radius := radius_small_semicircle width,
    area_large := 2 * area_semicircle large_radius,
    area_small := 2 * area_semicircle small_radius
  in
  (area_large / area_small - 1) * 100 = 125 :=
by
  sorry

end area_semicircles_percent_increase_l46_46796


namespace frosting_cupcakes_l46_46441

theorem frosting_cupcakes (R_Cagney R_Lacey R_Jamie : ℕ)
  (H1 : R_Cagney = 1 / 20)
  (H2 : R_Lacey = 1 / 30)
  (H3 : R_Jamie = 1 / 40)
  (TotalTime : ℕ)
  (H4 : TotalTime = 600) :
  (R_Cagney + R_Lacey + R_Jamie) * TotalTime = 65 :=
by
  sorry

end frosting_cupcakes_l46_46441


namespace orthogonal_trihedral_angle_area_sum_l46_46616

theorem orthogonal_trihedral_angle_area_sum (O A B C : Point) (a b c : ℝ)
  (hOa : ∥O - A∥ = a) (hOb : ∥O - B∥ = b) (hOc : ∥O - C∥ = c) (orthogonal : ∃P:Triangle, ⟂ (side_of P O A) (side_of P O B) (side_of P O C)) :
  let S := area_of_triangle A B C in
  S^2 = (1/4) * (b^2 * c^2 + c^2 * a^2 + a^2 * b^2) := 
sorry

end orthogonal_trihedral_angle_area_sum_l46_46616


namespace circle_polar_equation_and_segment_length_l46_46002

theorem circle_polar_equation_and_segment_length :
  (∀ ϕ : ℝ, let x := 1 + cos ϕ, let y := sin ϕ in (x - 1)^2 + y^2 = 1) →
  (∀ θ : ℝ, let ρ := 2 * cos θ in (ρ * cos θ - 1)^2 + (ρ * sin θ)^2 = 1) →
  ∀ θ1 θ2 ρ1 ρ2,
    θ1 = π / 3 →
    θ2 = π / 3 →
    ρ1 = 2 * cos θ1 →
    ρ2 * (sin θ2 + sqrt 3 * cos θ2) = 3 * sqrt 3 →
    |ρ1 - ρ2| = 2 :=
by 
  intros h_circle_equation h_polar_circle θ1 θ2 ρ1 ρ2 h_theta1 h_theta2 h_ρ1 h_ρ2
  -- Proof omitted
  sorry

end circle_polar_equation_and_segment_length_l46_46002


namespace functional_form_l46_46025

theorem functional_form {f : ℤ → ℤ} (h1 : ∀ m : ℤ, f(m + 8) ≤ f(m) + 8) 
                                      (h2 : ∀ m : ℤ, f(m + 11) ≥ f(m) + 11) :
  ∃ a : ℤ, ∀ m : ℤ, f(m) = m + a :=
begin
  sorry
end

end functional_form_l46_46025


namespace sequence_a_100_l46_46997

def seq (n : ℕ) : ℕ :=
  if n = 1 then 2
  else seq (n - 1) + 2 * (n - 1)

theorem sequence_a_100 :
  seq 100 = 9902 := 
  sorry

end sequence_a_100_l46_46997


namespace general_eq_line_l_cartesian_eq_curve_C_sum_PA_PB_l46_46198

section
variables {x y t ρ θ : Real}
constants (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ)

-- Definitions of the given conditions
def point_P : ℝ × ℝ := (1, -2)

def line_l (t : ℝ) : ℝ × ℝ := (1 + t, -2 + t)

def curve_C_polar (ρ θ : ℝ) : ℝ := ρ * (sin θ)^2 - 2 * cos θ

def curve_C_cartesian (x y : ℝ) : ℝ := y^2 - 2 * x

-- Proving general equation of the line l
theorem general_eq_line_l : ∀ t, ∃ x y, (1 + t = x) ∧ (-2 + t = y) → (x - y - 3 = 0) :=
by { sorry }

-- Proving Cartesian equation of the curve C
theorem cartesian_eq_curve_C : ∀ (ρ θ : ℝ), (curve_C_polar ρ θ = 0) → 
  ∃ x y, (ρ = real.sqrt (x^2 + y^2)) ∧ (θ = real.atan2 y x) ∧ (y^2 = 2 * x) :=
by { sorry }

-- Proving the calculation of |PA| + |PB|
theorem sum_PA_PB : ∀ (A B : ℝ → ℝ × ℝ), 
  let m := sorry in -- derived from quadratic solution step, omitted here
  (A = line_l m) ∧ (B = line_l m) ∧ (|PA| + |PB| = 6 * sqrt 2) :=
by { sorry }

end

end general_eq_line_l_cartesian_eq_curve_C_sum_PA_PB_l46_46198


namespace weekly_earnings_before_rent_l46_46242

theorem weekly_earnings_before_rent (EarningsAfterRent : ℝ) (weeks : ℕ) (rentPerWeek : ℝ) :
  EarningsAfterRent = 93899 → weeks = 233 → rentPerWeek = 49 →
  ((EarningsAfterRent + rentPerWeek * weeks) / weeks) = 451.99 :=
by
  intros H1 H2 H3
  -- convert the assumptions to the required form
  rw [H1, H2, H3]
  -- provide the objective statement
  change ((93899 + 49 * 233) / 233) = 451.99
  -- leave the final proof details as a sorry for now
  sorry

end weekly_earnings_before_rent_l46_46242


namespace min_value_of_f_l46_46129

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  2 * x^3 - 6 * x^2 + m

theorem min_value_of_f :
  ∀ (m : ℝ),
    f 0 m = 3 →
    ∃ x min, x ∈ Set.Icc (-2:ℝ) (2:ℝ) ∧ min = f x m ∧ min = -37 :=
by
  intros m h
  have h' : f 0 m = 3 := h
  -- Proof omitted.
  sorry

end min_value_of_f_l46_46129


namespace number_of_real_solutions_eq_l46_46008

theorem number_of_real_solutions_eq (x : ℝ) :
  (∃ x : ℝ, 9 * x^2 - 45 * (⌊x⌋ : ℤ) + 63 = 0) ↔ --Statement implying real solutions exist
  count_real_solutions_eq (9 * x^2 - 45 * (⌊x⌋ : ℤ) + 63) = correct_number_of_solutions :=
by 
  sorry

noncomputable def count_real_solutions_eq (f : ℝ → ℝ) : ℤ := 
  -- implementation to count real solutions
  sorry

end number_of_real_solutions_eq_l46_46008


namespace percentage_increase_l46_46780

theorem percentage_increase (original_value : ℕ) (percentage_increase : ℚ) :  
  original_value = 1200 → 
  percentage_increase = 0.40 →
  original_value * (1 + percentage_increase) = 1680 :=
by
  intros h1 h2
  sorry

end percentage_increase_l46_46780


namespace total_beds_in_hotel_l46_46835

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end total_beds_in_hotel_l46_46835


namespace math_proof_problem_l46_46580

-- Definitions derived from the problem 

variables (O A B C : Type) [AdditiveGroup O] [VectorSpace ℝ O]
variables (OA OB OC : O) (x m : ℝ)
variables A_cord : ℝ x -> (ℝ × ℝ) 
variables B_cord : ℝ x -> (ℝ × ℝ) 
variables C_cord : ℝ x -> (ℝ × ℝ)

-- Conditions
axiom h1 : OC = (1 / 3) • OA + (2 / 3) • OB
axiom h2 : A_cord = λ x => (1, cos x)
axiom h3 : B_cord = λ x => (1 + cos x, cos x)
axiom h4 : f x : ℝ := (OC.1 + OC.2 • OC.1) - (2 • m + 2 / 3) * abs_vector_sub O B
axiom h5 : x ∈ [0, π/2]
axiom h6 : f x = -3 / 2

-- Proof statements
noncomputable def prove_collinearity : Prop := ∃ μ : ℝ, (OC - OA) = μ • (OB - OA)
noncomputable def ratio_AC_CB : Prop := |abs_vector_sub_open A C / abs_vector_sub_open C B| = 2
noncomputable def find_m : Prop := m = 7 / 4

-- Lean statement encapsulating the above proof
theorem math_proof_problem :
  (prove_collinearity OA OB OC OA) ∧ (ratio_AC_CB x m) ∧ (find_m m) :=
by sorry

end math_proof_problem_l46_46580


namespace waiting_time_probability_l46_46172

theorem waiting_time_probability :
  (∀ (t : ℝ), 0 ≤ t ∧ t < 30 → (1 / 30) * (if t < 25 then 5 else 5 - (t - 25)) = 1 / 6) :=
by
  sorry

end waiting_time_probability_l46_46172


namespace polar_to_rect_coordinates_l46_46406

theorem polar_to_rect_coordinates :
  let x := 15
      y := 8
      r := Real.sqrt (x^2 + y^2)
      θ := Real.atan (y / x)
      x' := 2 * r * Real.cos (3 * θ)
      y' := 2 * r * Real.sin (3 * θ)
  in (x', y') = (78, -14) :=
by
  have h1 : r = 17 := by sorry
  have h2 : θ = Real.atan (8 / 15) := by sorry
  have h3 : 2 * r = 34 := by sorry
  have h4 : Real.cos θ = 15 / 17 := by sorry
  have h5 : Real.sin θ = 8 / 17 := by sorry
  have h6 : Real.cos (3 * θ) = 11295 / 4913 := by sorry
  have h7 : Real.sin (3 * θ) = -2024 / 4913 := by sorry
  have hx' : x' = 34 * (11295 / 4913) := by sorry
  have hy' : y' = 34 * (-2024 / 4913) := by sorry
  exact (hx', hy')

end polar_to_rect_coordinates_l46_46406


namespace algebraic_expression_equality_l46_46384

variable {x : ℝ}

theorem algebraic_expression_equality (h : x^2 + 3*x + 8 = 7) : 3*x^2 + 9*x - 2 = -5 := 
by
  sorry

end algebraic_expression_equality_l46_46384


namespace product_value_l46_46724

theorem product_value :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
    -- Skipping the actual proof
    sorry

end product_value_l46_46724


namespace real_values_of_x_l46_46884

theorem real_values_of_x :
  {x : ℝ | (∃ y, y = (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ∧ y ≥ -1)} =
  {x | -1 ≤ x ∧ x < -1/3 ∨ -1/3 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 1 < x} := 
sorry

end real_values_of_x_l46_46884


namespace derivative_of_function_l46_46287

theorem derivative_of_function :
  ∀ (x : ℝ), deriv (λ x, -2 * exp x * sin x) x = -2 * exp x * (sin x + cos x) :=
by
  intro x
  sorry

end derivative_of_function_l46_46287


namespace compute_XY_squared_l46_46191

noncomputable def compute_square_distance {AB BC CD DA : ℕ} (angle_D : ℝ) (XY_distance : ℝ) :=
  let X := 15
  let Y := 28
  60 * π / 180 -- converting degrees to radians for calculations

theorem compute_XY_squared :
  ∀ {AB BC CD DA : ℕ} (angle_D : ℝ) (X Y : ℕ), 
  AB = 15 → BC = 15 → CD = 28 → DA = 28 → angle_D = π / 3 →
  let XY := 350.25 + 7 * real.sqrt 3 in XY^2 = XY :=
sorry

end compute_XY_squared_l46_46191


namespace lengths_of_legs_l46_46183

def is_right_triangle (a b c : ℕ) := a^2 + b^2 = c^2

theorem lengths_of_legs (a b : ℕ) 
  (h1 : is_right_triangle a b 60)
  (h2 : a + b = 84) 
  : (a = 48 ∧ b = 36) ∨ (a = 36 ∧ b = 48) :=
  sorry

end lengths_of_legs_l46_46183


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46361

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46361


namespace exists_divisor_l46_46213

open Classical

variables {n : ℕ} (circ : Finset (Fin n)) (color : Fin n → Bool)
variables (S : Finset (Fin n))

-- Conditions
def circle_conditions (n : ℕ) (circ : Finset (Fin n)) (color : Fin n → Bool) (S : Finset (Fin n)) :=
  circ = Finset.univ ∧
  color.range \ {0, 1} ≠ ∅ ∧
  ∃ r b, r ≠ b ∧ r ∈ circ ∧ b ∈ circ ∧ color r ≠ color b ∧ 
  ∃ S, S ⊆ Finset.range n ∧ S.card ≥ 2 ∧ (∀ (x y : Fin n), x < y → color x ≠ color y → (y - x) ∈ S → y ∈ S)

-- The theorem based on the conditions
theorem exists_divisor (h : circle_conditions n circ color S) :
  ∃ (d : ℕ), d ∣ n ∧ d ≠ 1 ∧ d ≠ n ∧
  (∀ (x y : Fin n), x ≠ y → color x ≠ color y → (y - x) % d = 0 → x % d = 0 ∧ y % d = 0) :=
sorry

end exists_divisor_l46_46213


namespace base_k_representation_of_c_l46_46653

variable (k : ℕ) (b c : ℕ)

noncomputable def is_valid_c (k b c : ℕ) :=
  k > 9 ∧
  b = 2 * k + 1 ∧
  Exists (λ x : ℕ, (x = k ∨ x = k - 7) ∧ k * (k - 7) = c)

-- Statement: Prove that the base-k representation of c is 30_k given the conditions
theorem base_k_representation_of_c (h : is_valid_c k b c) : sorry :=
begin
  sorry,
end

end base_k_representation_of_c_l46_46653


namespace general_term_formula_l46_46470

-- Define the sequence as given in the conditions
def seq (n : ℕ) : ℚ := 
  match n with 
  | 0       => 1
  | 1       => 2 / 3
  | 2       => 1 / 2
  | 3       => 2 / 5
  | (n + 1) => sorry   -- This is just a placeholder, to be proved

-- State the theorem
theorem general_term_formula (n : ℕ) : seq n = 2 / (n + 1) := 
by {
  -- Proof will be provided here
  sorry
}

end general_term_formula_l46_46470


namespace confetti_left_correct_l46_46868

-- Define the number of pieces of red and green confetti collected by Eunji
def red_confetti : ℕ := 1
def green_confetti : ℕ := 9

-- Define the total number of pieces of confetti collected by Eunji
def total_confetti : ℕ := red_confetti + green_confetti

-- Define the number of pieces of confetti given to Yuna
def given_to_Yuna : ℕ := 4

-- Define the number of pieces of confetti left with Eunji
def confetti_left : ℕ :=  red_confetti + green_confetti - given_to_Yuna

-- Goal to prove
theorem confetti_left_correct : confetti_left = 6 := by
  -- Here the steps proving the equality would go, but we add sorry to skip the proof
  sorry

end confetti_left_correct_l46_46868


namespace lcm_48_180_l46_46038

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l46_46038


namespace distance_between_trees_l46_46431

theorem distance_between_trees :
  ∀ (yard_length : ℕ) (num_trees : ℕ), 
  yard_length = 273 → 
  num_trees = 14 → 
  (yard_length / (num_trees - 1)) = 21 :=
by
  intros yard_length num_trees h_yard_length h_num_trees
  rw [h_yard_length, h_num_trees]
  exact (273 / 13) = 21
  sorry

end distance_between_trees_l46_46431


namespace house_number_is_fourteen_l46_46768

theorem house_number_is_fourteen (a b c n : ℕ) (h1 : a * b * c = 40) (h2 : a + b + c = n) (h3 : 
  ∃ (a b c : ℕ), a * b * c = 40 ∧ (a = 1 ∧ b = 5 ∧ c = 8) ∨ (a = 2 ∧ b = 2 ∧ c = 10) ∧ n = 14) :
  n = 14 :=
sorry

end house_number_is_fourteen_l46_46768


namespace plane_divides_CD_l46_46253

-- Definitions based on conditions
variables (A B C D M N K : Point)
variables (β : ℝ)

-- Assume rational points and ratios - unspecified point structure
def AM_MD := ratio (AM M D) 2 3
def BN_AN := ratio (BN N A) 1 2
def BK_KC := ratio (BK K C) 1 1

theorem plane_divides_CD :
  (∃ P : Point, ∃ x y : unitInterval,
  Plane_of M N K ∈ Tetrahedron A B C D ∧
  P ∈ Line CD ∧
  divides_line_at P C D 3 1) :=
sorry

end plane_divides_CD_l46_46253


namespace soccer_league_points_l46_46581

structure Team :=
  (name : String)
  (regular_wins : ℕ)
  (losses : ℕ)
  (draws : ℕ)
  (bonus_wins : ℕ)

def total_points (t : Team) : ℕ :=
  3 * t.regular_wins + t.draws + 2 * t.bonus_wins

def Team_Soccer_Stars : Team :=
  { name := "Team Soccer Stars", regular_wins := 18, losses := 5, draws := 7, bonus_wins := 6 }

def Lightning_Strikers : Team :=
  { name := "Lightning Strikers", regular_wins := 15, losses := 8, draws := 7, bonus_wins := 5 }

def Goal_Grabbers : Team :=
  { name := "Goal Grabbers", regular_wins := 21, losses := 5, draws := 4, bonus_wins := 4 }

def Clever_Kickers : Team :=
  { name := "Clever Kickers", regular_wins := 11, losses := 10, draws := 9, bonus_wins := 2 }

theorem soccer_league_points :
  total_points Team_Soccer_Stars = 73 ∧
  total_points Lightning_Strikers = 62 ∧
  total_points Goal_Grabbers = 75 ∧
  total_points Clever_Kickers = 46 ∧
  [Goal_Grabbers, Team_Soccer_Stars, Lightning_Strikers, Clever_Kickers].map total_points =
  [75, 73, 62, 46] := 
by
  sorry

end soccer_league_points_l46_46581


namespace cora_reading_ratio_l46_46003

variable (P : Nat) 
variable (M T W Th F : Nat)

-- Conditions
def conditions (P M T W Th F : Nat) : Prop := 
  P = 158 ∧ 
  M = 23 ∧ 
  T = 38 ∧ 
  W = 61 ∧ 
  Th = 12 ∧ 
  F = Th

-- The theorem statement
theorem cora_reading_ratio (h : conditions P M T W Th F) : F / Th = 1 / 1 :=
by
  -- We use the conditions to apply the proof
  obtain ⟨hp, hm, ht, hw, hth, hf⟩ := h
  rw [hf]
  norm_num
  sorry

end cora_reading_ratio_l46_46003


namespace q1_monotonic_intervals_and_extreme_value_q2_fx_gt_gx_q3_x1_x2_gt4_l46_46955

noncomputable def f (x : ℝ) : ℝ := (x - 1) / Real.exp (x - 1)

def g (x : ℝ) : ℝ := f (4 - x)

theorem q1_monotonic_intervals_and_extreme_value :
  (∀ x, x < 2 → ∃ (f' : ℝ), f' > 0) ∧
  (∀ x, x > 2 → ∃ (f' : ℝ), f' < 0) ∧
  (∀ c, ∃ (fc : ℝ), f c = 1 / Real.exp 1) :=
sorry

theorem q2_fx_gt_gx (x : ℝ) (h : x > 2) :
  f(x) > g(x) :=
sorry

theorem q3_x1_x2_gt4 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 = f x2) :
  x1 + x2 > 4 :=
sorry

end q1_monotonic_intervals_and_extreme_value_q2_fx_gt_gx_q3_x1_x2_gt4_l46_46955


namespace upper_base_length_l46_46093

structure Trapezoid (A B C D M : Type) :=
  (on_lateral_side : ∀ {AB DM}, DM ⊥ AB)
  (perpendicular : ∀ {DM AB}, DM ⊥ AB)
  (equal_segments : MC = CD)
  (AD_length : AD = d)


theorem upper_base_length {A B C D M : Type} [trapezoid : Trapezoid A B C D M] :
  BC = d / 2 := 
begin
  sorry
end

end upper_base_length_l46_46093


namespace sum_of_faces_edges_vertices_l46_46372

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46372


namespace isosceles_triangle_k_value_l46_46573

theorem isosceles_triangle_k_value 
  (α β γ θ : ℝ) 
  (h1 : α = 3 * θ) 
  (h2 : β = 2 * θ) 
  (h3 : α + β + γ = Real.pi)
  (h4 : α + β = Real.pi - γ)
  : γ = Real.pi / 7 ↔ θ = Real.pi / 7:=
begin
  sorry
end

end isosceles_triangle_k_value_l46_46573


namespace BC_length_l46_46087

variable (A B C D M : Type)
variable [IsTrapezoid A B C D] [IsPointOnLateralSide M A B]
variable (DM_perpendicular_AB : Perpendicular D M A B)
variable (MC_eq_CD : MC = CD)
variable (AD_eq_d : AD = d)

theorem BC_length (d : ℝ) (A B C D M : Point)
  (H1 : IsTrapezoid A B C D)
  (H2 : IsPointOnLateralSide M A B)
  (H3 : Perpendicular D M A B)
  (H4 : MC = CD)
  (H5 : AD = d) :
  BC = d / 2 :=
sorry

end BC_length_l46_46087


namespace exists_convex_quadrilateral_and_point_gt_perimeter_l46_46012

theorem exists_convex_quadrilateral_and_point_gt_perimeter :
  ∃ (A B C D P : Type) 
    (d : A → B → ℝ) (x y : ℝ) 
    (h1 : d A D = x) (h2 : d B D = x) (h3 : d C D = x)
    (h4 : d A B = y) (h5 : d B C = y) (h6 : y < x / 4) 
    (h7 : d P D = y), 
  d P A + d P B + d P C + d P D > d A B + d B C + d C D + d D A :=
begin 
  sorry 
end

end exists_convex_quadrilateral_and_point_gt_perimeter_l46_46012


namespace remainder_sum_of_powers_of_3_mod_101_l46_46605

theorem remainder_sum_of_powers_of_3_mod_101 :
  let T := {n | ∃ k, k < 100 ∧ n = 3^k % 101}
  let U := ∑ n in T, n
  U % 101 = 50 :=
by
  let T : Finset ℕ := (Finset.range 100).image (λ k, (3^k % 101))
  let U : ℕ := T.sum id
  have h : U % 101 = 50 := sorry
  exact h

end remainder_sum_of_powers_of_3_mod_101_l46_46605


namespace lcm_48_180_eq_720_l46_46051

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l46_46051


namespace bus_overloaded_l46_46759

theorem bus_overloaded : 
  ∀ (capacity : ℕ) (first_pickup_ratio : ℚ) (next_pickup : ℕ) (bus_full : capacity = 80) (entered_first : first_pickup_ratio = 3/5) (next_pickup_point_waiting : next_pickup = 50), 
  let entered := (first_pickup_ratio * capacity).to_nat in -- people entered at first pickup
  let available_seats := capacity - entered in -- available seats after first pickup
  let could_not_take_bus := next_pickup - available_seats in -- people who could not take the bus
  could_not_take_bus = 18 := 
by 
  intros capacity first_pickup_ratio next_pickup bus_full entered_first next_pickup_point_waiting 
  let entered := (first_pickup_ratio * capacity).to_nat 
  let available_seats := capacity - entered 
  let could_not_take_bus := next_pickup - available_seats 
  sorry

end bus_overloaded_l46_46759


namespace number_of_set_B_l46_46687

theorem number_of_set_B (U A B : Finset ℕ) (hU : U.card = 193) (hA_inter_B : (A ∩ B).card = 25) (hA : A.card = 110) (h_not_in_A_or_B : 193 - (A ∪ B).card = 59) : B.card = 49 := 
by
  sorry

end number_of_set_B_l46_46687


namespace triangle_area_tan_l46_46219

theorem triangle_area_tan (A B C : Point) (area_ABC : ℝ)
  (tan_ABC : ℝ) (h_area: area_ABC = 10) (h_tan: tan_ABC = 5) :
  ∃ (a b c : ℕ), (AC^2 (a b c)) = -a + b * sqrt(c) ∧ a + b + c = 42 :=
by
  -- Definitions for the conditions
  let area : ℝ := 10
  let tan : ℝ := 5
  -- Use the conditions provided
  have h1 : area = area_ABC := by rw h_area
  have h2 : tan = tan_ABC := by rw h_tan
  -- To be proved
  sorry

end triangle_area_tan_l46_46219


namespace sam_pam_ratio_is_2_l46_46310

-- Definition of given conditions
def min_assigned_pages : ℕ := 25
def harrison_extra_read : ℕ := 10
def pam_extra_read : ℕ := 15
def sam_read : ℕ := 100

-- Calculations based on the given conditions
def harrison_read : ℕ := min_assigned_pages + harrison_extra_read
def pam_read : ℕ := harrison_read + pam_extra_read

-- Prove the ratio of the number of pages Sam read to the number of pages Pam read is 2
theorem sam_pam_ratio_is_2 : sam_read / pam_read = 2 := 
by
  sorry

end sam_pam_ratio_is_2_l46_46310


namespace integer_values_count_l46_46007

theorem integer_values_count : 
  ∃ n : ℕ, n = (finset.Icc 26 48).card ∧ n = 23 :=
by
  sorry

end integer_values_count_l46_46007


namespace geometric_means_form_l46_46886

noncomputable def geometric_means_sequence (a b : ℝ) (p : ℕ) (hma : a < b) : List ℝ :=
  let r := (b / a) ^ (1 / (p + 1))
  List.range (p + 1) ++ [p]
  |>.map (λ i => a * r ^ (i + 1))

theorem geometric_means_form (a b : ℝ) (p : ℕ) (hma : a < b) :
  List.zip (geometric_means_sequence a b p hma) (List.range (p + 1))
  = (List.range (p + 1)).map (λ n => (a * (b / a) ^ (n / (p + 1)))) :=
sorry

end geometric_means_form_l46_46886


namespace unique_solution_l46_46472

open Real IntervalIntegral

noncomputable def f (x : ℝ) : ℝ := sqrt (2 * x + 2 / (exp 2 - 1))

lemma f_is_continuously_differentiable : ContDiff ℝ 1 (λ x, f x) := sorry

lemma f_positive (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 0 < f x := 
begin
  -- Proof that f(x) > 0 for all x in [0,1]
  sorry
end

lemma f_condition_at_1 : (f 1) / (f 0) = exp 1 :=
begin
  -- Proof that f(1) / f(0) = e
  sorry
end

lemma integral_condition : 
  ∫ x in (0:ℝ)..1, (1 / (f x)^2) + (f'(x))^2 ≤ 2 :=
begin
  -- Proof that ∫ (1/f(x)^2) dx + ∫ (f'(x)^2) dx ≤ 2
  sorry
end

theorem unique_solution (g : ℝ → ℝ) (hg1 : ContDiff ℝ 1 g) 
    (hg2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 < g x) 
    (hg3 : (g 1) / (g 0) = exp 1)
    (hg4 : ∫ x in (0:ℝ)..1, (1 / (g x)^2) + (g' x)^2 ≤ 2) : 
  (∀ x, f x = g x) :=
begin
  -- Proof that f(x) is the unique solution that satisfies all conditions
  sorry
end

end unique_solution_l46_46472


namespace joey_pills_l46_46215

-- Definitions for the initial conditions
def TypeA_initial := 2
def TypeA_increment := 1

def TypeB_initial := 3
def TypeB_increment := 2

def TypeC_initial := 4
def TypeC_increment := 3

def days := 42

-- Function to calculate the sum of an arithmetic series
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- The theorem to be proved
theorem joey_pills :
  arithmetic_sum TypeA_initial TypeA_increment days = 945 ∧
  arithmetic_sum TypeB_initial TypeB_increment days = 1848 ∧
  arithmetic_sum TypeC_initial TypeC_increment days = 2751 :=
by sorry

end joey_pills_l46_46215


namespace piecewise_function_range_l46_46054

def piecewise_function (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2 * x - x ^ 2
  else if -2 ≤ x ∧ x ≤ 0 then x ^ 2 + 6 * x
  else 0

theorem piecewise_function_range : 
  {y : ℝ | ∃ x : ℝ, (0 ≤ x ∧ x ≤ 3 ∧ y = 2 * x - x^2) ∨ (-2 ≤ x ∧ x ≤ 0 ∧ y = x^2 + 6 * x)} = set.Icc (-8) 1 := 
by
  sorry

end piecewise_function_range_l46_46054


namespace inequality_problem_l46_46122

theorem inequality_problem (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by {
  sorry
}

end inequality_problem_l46_46122


namespace max_distance_on_tabletop_l46_46808

theorem max_distance_on_tabletop (d : ℝ) (rect_l : ℝ) (rect_w : ℝ) :
  d = 1 →  -- diameter of the round table is 1 meter
  rect_l = 1 → -- length of the rectangular part is 1 meter
  rect_w = 0.5 → -- width of the rectangular part is 0.5 meter
  ∀ (p q : ℝ × ℝ), 
  (p.fst^2 + p.snd^2 ≤ (d/2)^2 ∧ q.fst^2 + q.snd^2 ≤ (d/2)^2) → 
   dist p q ≤ 1.5 := 
by
  intros d rect_l rect_w hd hl hw 
  intro p q hpq
  sorry

end max_distance_on_tabletop_l46_46808


namespace sum_of_prime_factors_of_186_l46_46376

theorem sum_of_prime_factors_of_186 : 
  let p1 := 2
  let p2 := 3
  let p3 := 31
  p3 = 186,
  93 = 3 * p3,
  Prime p3
  ∑ (p : ℕ) in {2, 3, 31}, p = 36 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 31
  have hp3 : p3 = 31 := rfl
  have hp_div : 93 = 3 * 31 := by norm_num
  have prime_p3 : Prime 31 := by norm_num
  have sum_factors : ∑ (p : ℕ) in {2, 3, 31}, p = 36 := by norm_num
  exact sum_factors

end sum_of_prime_factors_of_186_l46_46376


namespace concurrency_of_lines_l46_46252

-- Define the tetrahedron and points
variables (S A B C A' B' C' : Point)
variables (SA : Line S A) (SB : Line S B) (SC : Line S C)
variables (SA' : Line S A') (SB' : Line S B') (SC' : Line S C')

-- Define planes and their intersection line
variables (π : Plane A B C) (ρ : Plane A' B' C') (d : Line π ρ)

-- Define lines from vertices to points on the opposite edges
variables (AA' : Line A A') (BB' : Line B B') (CC' : Line C C')

theorem concurrency_of_lines :
  ∃ P, ∀ t, remaining_concurrent (rotated_plane ρ d t) (lines_tetrahedron S A B C ρ A' B' C' d) = true → P = S :=
sorry

end concurrency_of_lines_l46_46252


namespace base3_to_base9_first_digit_correct_l46_46280

def base3_to_base10 (digits : List ℕ) : ℕ :=
  digits.foldr (λ (d : ℕ) (acc : ℕ) → d + 3 * acc) 0

def base10_to_base9 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec divide (n : ℕ) : List ℕ :=
      if n = 0 then []
      else (divide (n / 9)) ++ [n % 9]
    divide n

def first_digit_base9 (digits : List ℕ) : ℕ :=
  digits.headI

def problem_statement : Prop :=
  first_digit_base9 (base10_to_base9 (base3_to_base10 [2, 0, 2, 2, 2, 1, 2, 0, 2, 1, 1, 1, 2, 2, 2, 1, 2, 0, 2, 1])) 
  = 5

theorem base3_to_base9_first_digit_correct : problem_statement :=
  by 
    -- Proof skipped, this is just the statement
    sorry

end base3_to_base9_first_digit_correct_l46_46280


namespace largest_degree_horizontal_asymptote_l46_46460

theorem largest_degree_horizontal_asymptote (p : ℚ[X]) :
  (∃ k, (∀ (x : ℚ), p.degree ≤ k) ∧ k = 6) :=
by
  sorry

end largest_degree_horizontal_asymptote_l46_46460


namespace cut_square_into_50_squares_l46_46986

theorem cut_square_into_50_squares
  (cuts : ℕ)
  (assemble_squares : ℕ)
  (five_by_five_square : ℕ × ℕ)
  (equal_squares : ℕ × ℕ)
  (no_unused_pieces : Bool)
  (no_overlap_pieces : Bool)
  (result : ℕ) :
  five_by_five_square = (5, 5) →
  equal_squares = (1, 1) →
  no_unused_pieces = true →
  no_overlap_pieces = true →
  result = 50 :=
by
  intros five_by_five_square_eq equal_squares_eq no_unused_pieces_true no_overlap_pieces_true
  have cuts_def : cuts = 25 * 4 := sorry
  have assemble_squares_def : assemble_squares = cuts / 2 := sorry
  have result_def : result = assemble_squares := sorry
  have result_correct : result = 50 := by
    rw [result_def, assemble_squares_def, cuts_def]
    norm_num
  exact result_correct

end cut_square_into_50_squares_l46_46986


namespace range_of_a_l46_46495

theorem range_of_a (x a : ℝ) (p : Prop) (q : Prop) (H₁ : p ↔ (x < -3 ∨ x > 1))
  (H₂ : q ↔ (x > a))
  (H₃ : ¬p → ¬q) (H₄ : ¬q → ¬p → false) : a ≥ 1 :=
sorry

end range_of_a_l46_46495


namespace smallest_a_value_l46_46652

theorem smallest_a_value 
  (a b c : ℚ) 
  (a_pos : a > 0)
  (vertex_condition : ∃(x₀ y₀ : ℚ), x₀ = -1/3 ∧ y₀ = -4/3 ∧ y = a * (x + x₀)^2 + y₀)
  (integer_condition : ∃(n : ℤ), a + b + c = n)
  : a = 3/16 := 
sorry

end smallest_a_value_l46_46652


namespace opening_night_customers_l46_46779

theorem opening_night_customers
  (matinee_tickets : ℝ := 5)
  (evening_tickets : ℝ := 7)
  (opening_night_tickets : ℝ := 10)
  (popcorn_cost : ℝ := 10)
  (num_matinee_customers : ℝ := 32)
  (num_evening_customers : ℝ := 40)
  (total_revenue : ℝ := 1670) :
  ∃ x : ℝ, 
    (matinee_tickets * num_matinee_customers + 
    evening_tickets * num_evening_customers + 
    opening_night_tickets * x + 
    popcorn_cost * (num_matinee_customers + num_evening_customers + x) / 2 = total_revenue) 
    ∧ x = 58 := 
by
  use 58
  sorry

end opening_night_customers_l46_46779


namespace warthogs_seen_on_Monday_l46_46017

-- Condition definitions
def saw_animals_Sat : ℕ := 3 + 2 -- 3 lions + 2 elephants
def saw_animals_Sun : ℕ := 2 + 5 -- 2 buffaloes + 5 leopards
def total_animals : ℕ := 20 -- total animals over the three days
def saw_rhinos_Mon : ℕ := 5 -- rhinos seen on Monday

theorem warthogs_seen_on_Monday : saw_animals_Sat + saw_animals_Sun + (saw_rhinos_Mon + 3) = total_animals :=
by
  simp [saw_animals_Sat, saw_animals_Sun, saw_rhinos_Mon, total_animals]
  -- this line simplifies the expressions and should yield the correct equality
  sorry

end warthogs_seen_on_Monday_l46_46017


namespace BC_length_l46_46086

variable (A B C D M : Type)
variable [IsTrapezoid A B C D] [IsPointOnLateralSide M A B]
variable (DM_perpendicular_AB : Perpendicular D M A B)
variable (MC_eq_CD : MC = CD)
variable (AD_eq_d : AD = d)

theorem BC_length (d : ℝ) (A B C D M : Point)
  (H1 : IsTrapezoid A B C D)
  (H2 : IsPointOnLateralSide M A B)
  (H3 : Perpendicular D M A B)
  (H4 : MC = CD)
  (H5 : AD = d) :
  BC = d / 2 :=
sorry

end BC_length_l46_46086


namespace ellen_needs_thirteen_golf_carts_l46_46863

theorem ellen_needs_thirteen_golf_carts :
  ∀ (patrons_from_cars patrons_from_bus patrons_per_cart : ℕ), 
  patrons_from_cars = 12 → 
  patrons_from_bus = 27 → 
  patrons_per_cart = 3 →
  (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := 
by 
  intros patrons_from_cars patrons_from_bus patrons_per_cart h1 h2 h3 
  have h: patrons_from_cars + patrons_from_bus = 39 := by 
    rw [h1, h2] 
    norm_num
  rw[h, h3]
  norm_num
  sorry

end ellen_needs_thirteen_golf_carts_l46_46863


namespace find_m_l46_46112

variables (a b : EuclideanSpace ℝ (Fin 3))
variable (m : ℝ)

axiom angle_between_a_b : real.angle a b = real.pi * (2 / 3)
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 4
axiom perpendicular_condition : ((m • a) + b) ⬝ a = 0

theorem find_m : m = 1 := by
  sorry

end find_m_l46_46112


namespace length_upper_base_eq_half_d_l46_46080

variables {A B C D M: Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables {d : ℝ}

def trapezoid (A B C D : Type*) : Prop :=
  ∃ p : B, ∃ q : C, ∃ r : D, A ≠ p ∧ p ≠ q ∧ q ≠ r ∧ r ≠ A

def midpoint (A D : Type*) (N : Type*) (d : ℝ) : Prop :=
  dist A N = d / 2 ∧ dist N D = d / 2

axiom dm_perp_ab : ∀ (M : Type*), dist D M ∧ D ≠ M → dist M (id D) ≠ 0

axiom mc_eq_cd : dist M C = dist C D

theorem length_upper_base_eq_half_d
  (A B C D M : Type*)
  (h1 : trapezoid A B C D)
  (h2 : dist A D = d)
  (h3 : dm_perp_ab M)
  (h4 : mc_eq_cd) :
  dist B C = d / 2 :=
sorry

end length_upper_base_eq_half_d_l46_46080


namespace integer_satisfaction_l46_46875

noncomputable def sigma (n : ℕ) : ℕ := nat.divisors n |> List.sum

def p (n : ℕ) : ℕ := nat.factors n |> List.maximum' (by simp)

theorem integer_satisfaction (n : ℕ) (h_n_ge_2 : n ≥ 2) (h_eq : sigma n / (p n - 1) = n) : n = 6 :=
sorry

end integer_satisfaction_l46_46875


namespace polynomial_no_value_of_3_l46_46916

variable {R : Type*} [CommRing R] [IsDomain R] 

noncomputable def polynomial_with_values (P : R[X]) (a1 a2 a3 : R) : Prop :=
  P.eval a1 = 2 ∧ P.eval a2 = 2 ∧ P.eval a3 = 2

theorem polynomial_no_value_of_3 (P : ℤ[X]) (a1 a2 a3 : ℤ) (ha_distinct: a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3) :
  (polynomial_with_values P a1 a2 a3) → ¬ ∃ b : ℤ, P.eval b = 3 :=
by
  sorry

end polynomial_no_value_of_3_l46_46916


namespace lcm_48_180_l46_46040

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l46_46040


namespace fixed_point_plane_l46_46543

-- Definitions of points and segments
variables {Point : Type} [metric_space_point : metric_space Point]
variables {A B C M N L : Point}
variables {AM BN CL : ℝ}

-- Claim of the proof problem
theorem fixed_point_plane (h1 : AM + BN + CL = k1 ∨ area AMNB + area BNLC + area CLMA = k2) :
  ∃ P : Point, plane_through_points [M, N, L] P :=
by 
  sorry

end fixed_point_plane_l46_46543


namespace find_g_expression_max_triangle_area_l46_46131

noncomputable def f (x ω : ℝ) : ℝ := 2 * Real.sin (ω * x)

noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin ((1/2) * x - (1/2) * φ)

theorem find_g_expression :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 → 0 < ω ∧ ω < 1 → f x ω ≤ sqrt 2) →
  (0 < φ ∧ φ < π / 2) →
  (∀ (x : ℝ), g x φ = 2 * Real.sin ((1/2) * (x - φ))) →
  (2 * Real.sin ((1/2) * (7 * π / 6 - φ)) = k * π + π / 2) →
     (g x (π / 6) = 2 * Real.sin ((1/2) * x - π / 12)) :=
sorry

theorem max_triangle_area :
  (c = 4) →
  (C = π / 6) →
  (16 ≤ a^2 + b^2 - 2 * a * b * Real.cos C) →
  (∀ (S : ℝ), S = 1/2 * a * b * Real.sin C → S ≤ 8 + 4 * Real.sqrt 3) :=
sorry

end find_g_expression_max_triangle_area_l46_46131


namespace people_cannot_take_bus_l46_46760

theorem people_cannot_take_bus 
  (carrying_capacity : ℕ) 
  (fraction_entered : ℚ) 
  (next_pickup : ℕ) 
  (carrying_capacity = 80) 
  (fraction_entered = 3 / 5) 
  (next_pickup = 50) : 
  let first_pickup := (fraction_entered * carrying_capacity : ℚ).to_nat in
  let available_seats := carrying_capacity - first_pickup in
  let cannot_board := next_pickup - available_seats in
  cannot_board = 18 :=
by 
  sorry

end people_cannot_take_bus_l46_46760


namespace eval_floor_ceil_addition_l46_46465

theorem eval_floor_ceil_addition : ⌊-3.67⌋ + ⌈34.2⌉ = 31 := by
  -- Condition 1: Definition of floor function
  have h1 : ⌊-3.67⌋ = -4 := by sorry
  -- Condition 2: Definition of ceiling function
  have h2 : ⌈34.2⌉ = 35 := by sorry
  -- Combining the results
  calc
    ⌊-3.67⌋ + ⌈34.2⌉ = -4 + 35 : by rw [h1, h2]
                ... = 31 : by sorry

end eval_floor_ceil_addition_l46_46465


namespace f_inequality_l46_46959

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l46_46959


namespace BC_length_l46_46088

variable (A B C D M : Type)
variable [IsTrapezoid A B C D] [IsPointOnLateralSide M A B]
variable (DM_perpendicular_AB : Perpendicular D M A B)
variable (MC_eq_CD : MC = CD)
variable (AD_eq_d : AD = d)

theorem BC_length (d : ℝ) (A B C D M : Point)
  (H1 : IsTrapezoid A B C D)
  (H2 : IsPointOnLateralSide M A B)
  (H3 : Perpendicular D M A B)
  (H4 : MC = CD)
  (H5 : AD = d) :
  BC = d / 2 :=
sorry

end BC_length_l46_46088


namespace solvable_2x5_mazes_l46_46433

-- Define the sequences a_n and b_n via mutual recursion
def a : ℕ → ℕ
| 0       := 0  -- this is a dummy value since we start from a_1
| 1       := 1
| 2       := 3
| (n + 3) := 2 * a (n + 2) + b n

and b : ℕ → ℕ
| 0       := 0  -- this is a dummy value since we start from b_1
| 1       := 2
| 2       := 4
| (n + 3) := 2 * b (n + 2) + a n

-- State that the number of solvable 2 x 5 mazes is 49
theorem solvable_2x5_mazes : a 5 = 49 :=
by
  sorry

end solvable_2x5_mazes_l46_46433


namespace area_increase_l46_46802

-- Defining the shapes and areas
def radius_large_side := 6
def radius_small_side := 4

def area_large_semicircles : ℝ := real.pi * (radius_large_side^2)
def area_small_semicircles : ℝ := real.pi * (radius_small_side^2)

-- The theorem statement
theorem area_increase : (area_large_semicircles / area_small_semicircles) = 2.25 → 
                         ((2.25 - 1) * 100) = 125 :=
by sorry

end area_increase_l46_46802


namespace book_price_net_change_l46_46163

theorem book_price_net_change (P : ℝ) :
  let decreased_price := P * 0.70
  let increased_price := decreased_price * 1.20
  let net_change := (increased_price - P) / P * 100
  net_change = -16 := 
by
  sorry

end book_price_net_change_l46_46163


namespace people_could_not_take_bus_l46_46765

theorem people_could_not_take_bus
  (carrying_capacity : ℕ)
  (first_pickup_ratio : ℚ)
  (first_pickup_people : ℕ)
  (people_waiting : ℕ)
  (total_on_bus : ℕ)
  (additional_can_carry : ℕ)
  (people_could_not_take : ℕ)
  (h1 : carrying_capacity = 80)
  (h2 : first_pickup_ratio = 3/5)
  (h3 : first_pickup_people = carrying_capacity * first_pickup_ratio.to_nat)
  (h4 : first_pickup_people = 48)
  (h5 : total_on_bus = first_pickup_people)
  (h6 : additional_can_carry = carrying_capacity - total_on_bus)
  (h7 : additional_can_carry = 32)
  (h8 : people_waiting = 50)
  (h9 : people_could_not_take = people_waiting - additional_can_carry)
  (h10 : people_could_not_take = 18) : 
  people_could_not_take = 18 :=
by
  sorry -- proof is left for another step

end people_could_not_take_bus_l46_46765


namespace derivative_at_zero_l46_46286

def polynomial (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * ... * (x - 2009)

theorem derivative_at_zero :
  derivative polynomial 0 = -2009! :=
sorry

end derivative_at_zero_l46_46286


namespace find_x_l46_46459

theorem find_x (x : ℚ) (h1 : x > 9) (h2 : (47 + x) / 5 = median [3, 9, 15, 20, x]) : x = 47 / 4 :=
sorry

end find_x_l46_46459


namespace work_together_days_l46_46766

theorem work_together_days
  (A_rate : ℚ := 1 / 30)
  (B_rate : ℚ := 1 / 40)
  (fraction_left : ℚ := 5 / 12) :
  let combined_rate := A_rate + B_rate in
  let work_completed := 1 - fraction_left in
  let days_worked := work_completed / combined_rate in
  days_worked = 10 := sorry

end work_together_days_l46_46766


namespace divisible_by_7_iff_l46_46618

variable {x y : ℤ}

theorem divisible_by_7_iff :
  7 ∣ (2 * x + 3 * y) ↔ 7 ∣ (5 * x + 4 * y) :=
by
  sorry

end divisible_by_7_iff_l46_46618


namespace iron_aluminum_weight_difference_l46_46254

theorem iron_aluminum_weight_difference :
  let iron_weight := 11.17
  let aluminum_weight := 0.83
  iron_weight - aluminum_weight = 10.34 :=
by
  sorry

end iron_aluminum_weight_difference_l46_46254


namespace distance_inequalities_l46_46915

variables (P Q : Point) -- Represents points P and Q
variables (α : Plane) -- Represents plane α
variables (l : Line) -- Represents line l such that l ⊆ α
variables (a b c : ℝ) -- Represents distances a, b, and c

-- Conditions
def point_outside_plane (P : Point) (α : Plane) : Prop := ¬ (P ∈ α)
def line_in_plane (l : Line) (α : Plane) : Prop := l ⊆ α
def point_on_line (Q : Point) (l : Line) : Prop := Q ∈ l
def distance_point_to_plane (P : Point) (α : Plane) : ℝ := a
def distance_point_to_line (P : Point) (l : Line) : ℝ := b
def distance_between_points (P Q : Point) : ℝ := c

-- Question as a theorem
theorem distance_inequalities (h1 : point_outside_plane P α)
                             (h2 : line_in_plane l α)
                             (h3 : point_on_line Q l)
                             (h4 : distance_point_to_plane P α = a)
                             (h5 : distance_point_to_line P l = b)
                             (h6 : distance_between_points P Q = c) :
                             a ≤ b ∧ b ≤ c :=
begin
  sorry
end

end distance_inequalities_l46_46915


namespace area_single_square_l46_46848

theorem area_single_square :
  let side_lengths := λ n, 3 * n in
  let areas := λ n, (side_lengths n) ^ 2 in
  let total_area := ∑ n in (Finset.range 11).map (λ n, n + 1), areas n in
  let A := total_area / 11 in
  A = 414 :=
by
  let side_lengths := λ n, 3 * n
  let areas := λ n, (side_lengths n) ^ 2
  let total_area := ∑ n in (Finset.range 11).map (λ n, n + 1), areas n
  let A := total_area / 11
  sorry

end area_single_square_l46_46848


namespace joe_initial_tests_l46_46214

noncomputable def initial_tests (n : ℕ) : Prop :=
  let initial_sum := 90 * n
  let new_sum := initial_sum - 75
  let new_avg := new_sum / (n - 1)
  new_avg = 85

theorem joe_initial_tests : ∃ n : ℕ, initial_tests n ∧ n = 13 :=
begin
  use 13,
  unfold initial_tests,
  -- Validation skipped 
    
  sorry -- proof skipped
end

end joe_initial_tests_l46_46214


namespace original_faculty_members_l46_46736

theorem original_faculty_members (reduced_faculty : ℕ) (percentage : ℝ) : 
  reduced_faculty = 195 → percentage = 0.80 → 
  (∃ (original_faculty : ℕ), (original_faculty : ℝ) = reduced_faculty / percentage ∧ original_faculty = 244) :=
by
  sorry

end original_faculty_members_l46_46736


namespace rectangle_vertex_x_value_l46_46182

noncomputable def rectangle_x_value (x : ℝ) : Prop :=
  let width := (1 : ℝ) - (-2)
  let area := 12
  let length := area / width
  x = 1 - length

theorem rectangle_vertex_x_value : rectangle_x_value (-3) :=
by
  let width := (1 : ℝ) - (-2)
  have h1 : width = 3 := by sorry
  let area := (12 : ℝ)
  let length := area / width
  have h2 : length = 4 := by sorry
  have h3 : (-3 : ℝ) = 1 - length := by sorry
  rw [←h3]
  exact Eq.trans rfl rfl

end rectangle_vertex_x_value_l46_46182


namespace red_tint_percentage_new_mixture_l46_46778

open Real

def original_volume : ℝ := 30
def original_red_percent : ℝ := 0.20
def added_red_volume : ℝ := 8

def original_red_volume : ℝ := original_red_percent * original_volume
def total_red_volume : ℝ := original_red_volume + added_red_volume
def new_volume : ℝ := original_volume + added_red_volume
def new_red_percent : ℝ := (total_red_volume / new_volume) * 100

theorem red_tint_percentage_new_mixture :
  new_red_percent ≈ 37 := by sorry

end red_tint_percentage_new_mixture_l46_46778


namespace functional_equation_solution_l46_46473

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) → (∀ x : ℝ, f x = 0) :=
by
  sorry

end functional_equation_solution_l46_46473


namespace max_area_triangle_PMN_l46_46097

noncomputable def ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  { p | p.1^2 / a^2 + p.2^2 / b^2 = 1 }

theorem max_area_triangle_PMN :
  (∀ (m : ℝ), m ≠ 0 →
  let b := sqrt 3 in let a := 2 * sqrt 3 in let C := ellipse a b in
  let l (m : ℝ) := { p : ℝ × ℝ | p.1 = m * p.2 + 3 } in
  let P (p : ℝ × ℝ) := (4, 0) in
  let area_PMN (y1 y2 : ℝ) := 2 * sqrt 3 * sqrt ((m^2 + 1) / ((m^2 + 4)^2)) in
  ∃ y1 y2 : ℝ, (y1, y2) ∈ C ∧ P (4, 0) ∧ area_PMN y1 y2 = 1) ∧
  ∃ m, m = sqrt 2 := 
begin
  sorry
end

end max_area_triangle_PMN_l46_46097


namespace spaceship_finds_alien_l46_46826

-- Define the radius of the planet (R), speed of alien (u), speed of spaceship (v), 
-- and the condition that the spaceship speed is greater than 10 times the alien's speed
variables (R u v : ℝ) (h1 : v > 10 * u)

-- The proof problem: Prove that the spaceship can always find the alien given the above conditions
theorem spaceship_finds_alien (h2 : ∀ t: ℝ, alien_position t ∈ surface(R)) 
                               (h3 : ∀ t: ℝ, spaceship_speed t = v)
                               (h4 : ∃ α > 0, ∀ t, alien_speed t ≤ α): 
                                ∃ t, spaceship_position t = alien_position t :=
  sorry

end spaceship_finds_alien_l46_46826


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46366

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46366


namespace dirichlet_expectation_Xj_dirichlet_variance_Xj_dirichlet_covariance_Xj_Xk_dirichlet_expectation_X_beta_l46_46742

noncomputable theory

def Delta (n : ℕ) : Set (Vector ℝ (n - 1)) :=
  {x | (∀ i, 0 ≤ x i) ∧ (0 ≤ x.toList.sum) ∧ (x.toList.sum ≤ 1)}

def dirichlet_density (x : Vector ℝ (n - 1)) (α : Vector ℝ n) : ℝ :=
  let A_n := α.toList.sum
  (Real.Gamma A_n) / ((α.toList.map Real.Gamma).prod) *
  (x.toList.mapWithIndex (λ i xi, xi^(α[i] - 1))).prod *
  (1 - x.toList.sum) ^ (α[n - 1] - 1)

def expectation_Xj (X : Vector ℝ n) (α : Vector ℝ n) (j : Fin n) : ℝ :=
  α[j] / α.toList.sum

def variance_Xj (X : Vector ℝ n) (α : Vector ℝ n) (j : Fin n) : ℝ :=
  let A_n := α.toList.sum
  (α[j] * (A_n - α[j])) / (A_n^2 * (A_n + 1))

def covariance_Xj_Xk (X : Vector ℝ n) (α : Vector ℝ n) (j k : Fin n) : ℝ :=
  let A_n := α.toList.sum
  -(α[j] * α[k]) / (A_n^2 * (A_n + 1))

theorem dirichlet_expectation_Xj (X : Vector ℝ n) (α : Vector ℝ n) (j : Fin n) :
  expectation_Xj X α j = α[j] / α.toList.sum := sorry

theorem dirichlet_variance_Xj (X : Vector ℝ n) (α : Vector ℝ n) (j : Fin n) :
  variance_Xj X α j = α[j] * (α.toList.sum - α[j]) / (α.toList.sum^2 * (α.toList.sum + 1)) := sorry

theorem dirichlet_covariance_Xj_Xk (X : Vector ℝ n) (α : Vector ℝ n) (j k : Fin n) (h : j ≠ k) :
  covariance_Xj_Xk X α j k = -(α[j] * α[k]) / (α.toList.sum^2 * (α.toList.sum + 1)) := sorry

theorem dirichlet_expectation_X_beta (X : Vector ℝ n) (α β : Vector ℝ n) :
  ∑ i, (X[i] ^ β[i]) =
  (Real.Gamma (α.toList.sum)) * ((List.map  (λ i, Real.Gamma (α[i] + β[i]) ) α.toList)).prod /
  ((List.map Real.Gamma α.toList).prod * Real.Gamma (α.toList.sum + β.toList.sum)) := sorry

end dirichlet_expectation_Xj_dirichlet_variance_Xj_dirichlet_covariance_Xj_Xk_dirichlet_expectation_X_beta_l46_46742


namespace angle_through_point_terminal_side_l46_46948

noncomputable def angle_set (k : ℤ) : Set ℝ :=
  {x | ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 6}

theorem angle_through_point_terminal_side (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (3, Real.sqrt 3) ∧ tan α = Real.sqrt 3 / 3) →
  (α ∈ angle_set) :=
by
  sorry

end angle_through_point_terminal_side_l46_46948


namespace sum_base_49_l46_46240

-- Definitions of base b numbers and their base 10 conversion
def num_14_in_base (b : ℕ) : ℕ := b + 4
def num_17_in_base (b : ℕ) : ℕ := b + 7
def num_18_in_base (b : ℕ) : ℕ := b + 8
def num_6274_in_base (b : ℕ) : ℕ := 6 * b^3 + 2 * b^2 + 7 * b + 4

-- The question: Compute 14 + 17 + 18 in base b
def sum_in_base (b : ℕ) : ℕ := 14 + 17 + 18

-- The main statement to prove
theorem sum_base_49 (b : ℕ) (h : (num_14_in_base b) * (num_17_in_base b) * (num_18_in_base b) = num_6274_in_base (b)) :
  sum_in_base b = 49 :=
by sorry

end sum_base_49_l46_46240


namespace f_inequality_l46_46960

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l46_46960


namespace seq_converges_to_zero_l46_46220

-- Define the sequence according to the given conditions
def seq (a : ℝ) : ℕ → ℝ
| 0     := a
| (n+1) := abs (seq n - (1 / (n+1)))

-- Prove that the sequence converges to 0
theorem seq_converges_to_zero (a : ℝ) (h : a > 0) : 
  filter.tendsto (seq a) filter.at_top (nhds 0) := 
sorry

end seq_converges_to_zero_l46_46220


namespace part1_monotonicity_part2_inequality_l46_46968

variable (a : ℝ) (x : ℝ)
variable (h : a > 0)

def f := a * (Real.exp x + a) - x

theorem part1_monotonicity :
  if a ≤ 0 then ∀ x : ℝ, (f a x) < (f a (x + 1)) else
  if a > 0 then ∀ x : ℝ, (x < Real.log (1 / a) → (f a x) > (f a (x + 1))) ∧ 
                       (x > Real.log (1 / a) → (f a x) < (f a (x - 1))) else 
  (False) := sorry

theorem part2_inequality :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3 / 2 := sorry

end part1_monotonicity_part2_inequality_l46_46968


namespace maximize_profit_marginal_profit_monotonic_decreasing_l46_46418

-- Definition of revenue function R
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Definition of cost function C
def C (x : ℕ) : ℤ := 460 * x + 500

-- Definition of profit function p
def p (x : ℕ) : ℤ := R x - C x

-- Lemma for the solution
theorem maximize_profit (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 20) : 
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧ 
  (∀ y, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) :=
by
  sorry

-- Definition of marginal profit function Mp
def Mp (x : ℕ) : ℤ := p (x + 1) - p x

-- Lemma showing Mp is monotonically decreasing
theorem marginal_profit_monotonic_decreasing (x : ℕ) (h2 : 1 ≤ x ∧ x ≤ 19) : 
  Mp x = -30 * x^2 + 60 * x + 3275 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 19 → (Mp y ≥ Mp (y + 1)) :=
by
  sorry

end maximize_profit_marginal_profit_monotonic_decreasing_l46_46418


namespace seq_formula_l46_46140

-- Define the sequence {a_n} using initial condition and recurrence relation
def a : ℕ → ℝ
| 1 := 0
| (n + 1) := a n + 4 * Real.sqrt (a n + 1) + 4

-- State the theorem we want to prove
theorem seq_formula (n : ℕ) (hn : n ≥ 1) : a n = 4 * n^2 - 4 * n := 
by
  sorry

end seq_formula_l46_46140


namespace simplify_and_evaluate_expression_l46_46643

   variable (x : ℝ)

   theorem simplify_and_evaluate_expression (h : x = 2 * Real.sqrt 5 - 1) :
     (1 / (x ^ 2 + 2 * x + 1) * (1 + 3 / (x - 1)) / ((x + 2) / (x ^ 2 - 1))) = Real.sqrt 5 / 10 :=
   sorry
   
end simplify_and_evaluate_expression_l46_46643


namespace semicircle_area_percentage_difference_l46_46790

-- Define the rectangle dimensions
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 8

-- Define the diameters and radii of the semicircles
def large_semicircle_radius : ℝ := rectangle_length / 2
def small_semicircle_radius : ℝ := rectangle_width / 2

-- Define the areas of the full circles made from the semicircles
def large_circle_area : ℝ := real.pi * (large_semicircle_radius ^ 2)
def small_circle_area : ℝ := real.pi * (small_semicircle_radius ^ 2)

-- Define the percentage larger question
def percent_larger (a b : ℝ) : ℝ := ((a - b) / b) * 100

-- Formal proof statement
theorem semicircle_area_percentage_difference : 
  percent_larger large_circle_area small_circle_area = 125 := 
by
  sorry

end semicircle_area_percentage_difference_l46_46790


namespace volume_of_pyramid_proof_l46_46457

open Real

-- Definitions of vertices A, B, and C
def A := (0,0 : ℝ × ℝ)
def B := (26,0 : ℝ × ℝ)
def C := (10,18 : ℝ × ℝ)

-- Definition of midpoints M, N, and P
def M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def P := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)

-- Definition of centroid G
def G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Definition of height h
def h := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Definition of area of triangle MNP using shoelace formula
def area_MNP := (1 / 2 : ℝ) * abs (M.1 * N.2 + N.1 * P.2 + P.1 * M.2 - (M.2 * N.1 + N.2 * P.1 + P.2 * M.1))

-- Definition of the volume of the pyramid
def volume_pyramid := (1 / 3 : ℝ) * area_MNP * h

-- Theorem stating the volume of the triangular pyramid
theorem volume_of_pyramid_proof : volume_pyramid = 7.5 * Real.sqrt 580 := 
by sorry

end volume_of_pyramid_proof_l46_46457


namespace inequality_proof_l46_46513

variable {a b c : ℝ}

theorem inequality_proof (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (sqrt (a + b + c) + sqrt a) / (b + c) +
  (sqrt (a + b + c) + sqrt b) / (c + a) +
  (sqrt (a + b + c) + sqrt c) / (a + b) ≥
    (9 + 3 * sqrt 3) / (2 * sqrt (a + b + c)) :=
by
  sorry

end inequality_proof_l46_46513


namespace reflection_over_line_l46_46414

-- Defining the matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![9 / 41, 40 / 41], ![40 / 41, -9 / 41]]

-- Defining a vector in two dimensions
structure Vector2D :=
  (a : ℤ)
  (b : ℤ)

-- Defining the conditions
def is_valid_vector (v : Vector2D) : Prop :=
  v.a > 0 ∧ Int.gcd v.a v.b = 1

def is_reflection_image (v : Vector2D) : Prop :=
  reflection_matrix.mul_vec ![↑v.a, ↑v.b] = ![↑v.a, ↑v.b]

-- Defining the direction vector
def direction_vector := Vector2D.mk 5 4

-- The proof statement
theorem reflection_over_line :
  is_valid_vector direction_vector ∧ is_reflection_image direction_vector :=
by
  sorry

end reflection_over_line_l46_46414


namespace find_t_l46_46142

theorem find_t (t : ℝ) : (B ⊆ A) → t = 2 :=
  let A := {1, t, 2 * t}
  let B := {1, t^2}
  fun hyp : B ⊆ A => sorry

end find_t_l46_46142


namespace trapezoid_ratio_sum_l46_46206

-- Definitions based on the given conditions
structure Trapezoid (A B C D : Type) where
  parallel : B ∥ D → A ∥ C

variables {A B C D : Type} [Trapezoid A B C D]

def isMidpoint (E : A) (A D : A) : Prop :=
  E.dist A = E.dist D

def pointOnSeg (F : B) (B C : B) : Prop :=
  B.dist F = 2 * F.dist C

-- Proof goal
theorem trapezoid_ratio_sum (A B C D E F : Type) [Trapezoid A B C D] 
  (h1 : isMidpoint E A D) (h2 : pointOnSeg F B C) :
  (E.dist F / F.dist A + B.dist E / E.dist D = 7 / 10) :=
sorry

end trapezoid_ratio_sum_l46_46206


namespace length_of_OM_l46_46914

-- Conditions
def parabola_symmetric_about_y_axis : Prop := true
def vertex_at_origin : Prop := true
def passes_through_M (x₀ : ℝ) : Prop := x₀^2 = 12 ∧ 3 + 1 = 4
def distance_from_M_to_focus_is_4 : Prop := true

-- Question: Prove length of OM is sqrt(21)
theorem length_of_OM (x₀ : ℝ) (h1 : parabola_symmetric_about_y_axis)
  (h2 : vertex_at_origin) (h3 : passes_through_M x₀) 
  (h4 : distance_from_M_to_focus_is_4) :
  real.sqrt (x₀^2 + 9) = real.sqrt 21 :=
by
  rw [h3],
  apply real.sqrt_eq,
  exact (add_right_cancel_iff 9).mp,
  sorry

end length_of_OM_l46_46914


namespace measure_of_angle_C_l46_46568

variables (A B C : ℝ)
variables (sin cos : ℝ → ℝ)

-- Define the conditions as hypotheses
hypothesis h1 : 3 * sin A + 4 * cos B = 6
hypothesis h2 : 3 * cos A + 4 * sin B = 1

-- Lean statement to prove the measure of angle C
theorem measure_of_angle_C (sin cos : ℝ → ℝ) (A B C : ℝ)
  (h1 : 3 * sin A + 4 * cos B = 6)
  (h2 : 3 * cos A + 4 * sin B = 1) :
  C = π / 6 :=
begin
  sorry
end

end measure_of_angle_C_l46_46568


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46348

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46348


namespace lcm_48_180_l46_46043

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l46_46043


namespace mark_spends_amount_l46_46871

-- Definitions based on the conditions in the problem
def notebook_price : ℝ := 2
def pen_price : ℝ := 1.5
def book_price : ℝ := 12
def magazine_price : ℝ := 3
def magazine_discount : ℝ := 0.25
def coupon_threshold : ℝ := 20
def coupon_amount : ℝ := 3

-- The assumptions (conditions) given in the problem
def notebooks_bought : ℕ := 4
def pens_bought : ℕ := 3
def books_bought : ℕ := 1
def magazines_bought : ℕ := 2

-- The total amount spent by Mark with the discount and coupon applied
theorem mark_spends_amount :
  let total_notebooks := (notebooks_bought * notebook_price)
  let total_pens := (pens_bought * pen_price)
  let total_books := (books_bought * book_price)
  let discounted_magazine_price := (magazine_price * (1 - magazine_discount))
  let total_magazines := (magazines_bought * discounted_magazine_price)
  let total_before_coupon := total_notebooks + total_pens + total_books + total_magazines
  let total_with_coupon := if total_before_coupon >= coupon_threshold then total_before_coupon - coupon_amount else total_before_coupon
  in total_with_coupon = 26 :=
by
  -- This is where the proof would go. We use 'sorry' to indicate missing proof.
  sorry

end mark_spends_amount_l46_46871


namespace people_cannot_take_bus_l46_46761

theorem people_cannot_take_bus 
  (carrying_capacity : ℕ) 
  (fraction_entered : ℚ) 
  (next_pickup : ℕ) 
  (carrying_capacity = 80) 
  (fraction_entered = 3 / 5) 
  (next_pickup = 50) : 
  let first_pickup := (fraction_entered * carrying_capacity : ℚ).to_nat in
  let available_seats := carrying_capacity - first_pickup in
  let cannot_board := next_pickup - available_seats in
  cannot_board = 18 :=
by 
  sorry

end people_cannot_take_bus_l46_46761


namespace find_f_log2_5_l46_46507

variable {f g : ℝ → ℝ}

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- g is an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom f_g_equation : ∀ x, f x + g x = (2:ℝ)^x + x

-- Proof goal: Compute f(log_2 5) and show it equals 13/5
theorem find_f_log2_5 : f (Real.log 5 / Real.log 2) = (13:ℝ) / 5 := by
  sorry

end find_f_log2_5_l46_46507


namespace reconstruct_phone_number_l46_46394

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def consecutive_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 3 ∧
  digits.nth 0 + 1 = digits.nth 1 ∧
  digits.nth 1 + 1 = digits.nth 2

def three_consecutive_ones (n: ℕ) : Prop :=
  let digits := n.digits 10 in
  let seq := [1, 1, 1] in
  ∃ k, seq = (digits.drop k).take 3

def one_two_digit_prime (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  let two_digit_numbers := [digits.take 2, digits.drop 3].map (λ l, l.foldl (λ b a, b * 10 + a) 0) in
  two_digit_numbers.any Nat.Prime

def construct_phone_number : ℕ :=
  7111765

theorem reconstruct_phone_number :
  let phone_number := construct_phone_number in
  phone_number.digits 10.length = 7 ∧
  consecutive_digits ((phone_number.digits 10).drop 4 ← 0).take 3) ∧
  is_palindrome ((phone_number.digits 10).take 5 ← 0) ∧
  Nat.sum (phone_number.digits 10).take 3 % 9 = 0 ∧
  three_consecutive_ones phone_number ∧
  one_two_digit_prime phone_number :=
by
  sorry

end reconstruct_phone_number_l46_46394


namespace value_of_f_neg_two_l46_46533
noncomputable theory

def f (x : ℝ) : ℝ :=
if x >= 0 then 3^x - 1 else -3^(-x) + 1

theorem value_of_f_neg_two : f (-2) = -8 :=
sorry

end value_of_f_neg_two_l46_46533


namespace gavin_has_17_green_shirts_l46_46064

noncomputable def gavin_shirts (t b : ℕ) : ℕ :=
  t - b

theorem gavin_has_17_green_shirts :
  ∀ (t b : ℕ), t = 23 → b = 6 → gavin_shirts t b = 17 :=
by
  intros t b ht hb
  rw [ht, hb]
  simp only [gavin_shirts]
  norm_num

end gavin_has_17_green_shirts_l46_46064


namespace find_q_l46_46145

variable (p q : ℝ)

theorem find_q (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end find_q_l46_46145


namespace bernardo_win_sum_digits_l46_46838

theorem bernardo_win_sum_digits :
  ∃ M : ℕ, 0 ≤ M ∧ M ≤ 1999 ∧
  (∀ (x : ℕ), x = M → 81 * x + 3120 > 1999) ∧
  (M = 34) ∧ (nat.digits 10 M).sum = 7 :=
sorry

end bernardo_win_sum_digits_l46_46838


namespace percent_larger_semicircles_l46_46417

theorem percent_larger_semicircles (r1 r2 : ℝ) (d1 d2 : ℝ)
  (hr1 : r1 = d1 / 2) (hr2 : r2 = d2 / 2)
  (hd1 : d1 = 12) (hd2 : d2 = 8) : 
  (2 * (1/2) * Real.pi * r1^2) = (9/4 * (2 * (1/2) * Real.pi * r2^2)) :=
by
  sorry

end percent_larger_semicircles_l46_46417


namespace max_sum_permutation_correct_l46_46218

noncomputable def max_sum_permutation (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6 - (2 * n - 3)

theorem max_sum_permutation_correct (n : ℕ) (S_n : finset (fin n → fin n)) 
  (hS : ∀ σ ∈ S_n, σ ∘ (shift n 1) = σ) :
  ∃ σ ∈ S_n, ∑ i in range n, σ i * σ (i + 1) = max_sum_permutation n :=
sorry

end max_sum_permutation_correct_l46_46218


namespace MN_parallel_AD_l46_46658

variable {A B C D P Q E F M N : Type}

variable (ABCD : Trapezoid A B C D) (BC_parallel_AD : BC ∥ AD)
variable (P : Point) (Q : Point)
variable (external_angle_bisectors_B_C : intersection (external_angle_bisector B) (external_angle_bisector C) = P)
variable (external_angle_bisectors_A_D : intersection (external_angle_bisector A) (external_angle_bisector D) = Q)
variable (E : Point) (F : Point) (M : Point) (N : Point)
variable (PB_intersects_AD_at_E : intersection (line P B) AD = E)
variable (PC_intersects_AD_at_F : intersection (line P C) AD = F)
variable (AP_intersects_EQ_at_M : intersection (line A P) (line E Q) = M)
variable (PD_intersects_FQ_at_N : intersection (line P D) (line F Q) = N)

theorem MN_parallel_AD
  (H_BC_parallel_AD : BC_parallel_AD)
  (H_external_angle_bisectors_B_C : external_angle_bisectors_B_C)
  (H_external_angle_bisectors_A_D : external_angle_bisectors_A_D)
  (H_PB_intersects_AD_at_E : PB_intersects_AD_at_E)
  (H_PC_intersects_AD_at_F : PC_intersects_AD_at_F)
  (H_AP_intersects_EQ_at_M : AP_intersects_EQ_at_M)
  (H_PD_intersects_FQ_at_N : PD_intersects_FQ_at_N) :
  MN ∥ AD :=
  sorry

end MN_parallel_AD_l46_46658


namespace cos4_x_minus_sin4_x_l46_46993

theorem cos4_x_minus_sin4_x (x : ℝ) (h : x = π / 12) : (Real.cos x) ^ 4 - (Real.sin x) ^ 4 = (Real.sqrt 3) / 2 := by
  sorry

end cos4_x_minus_sin4_x_l46_46993


namespace angle_between_vectors_l46_46546

-- Define the problem using vectors and conditions
theorem angle_between_vectors (a b : ℝ → ℝ) (θ : ℝ)
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 1)
  (ineq : ∀ x : ℝ, ∥a + x • b∥ ≥ ∥a + b∥) :
  θ = 2 * π / 3 :=
by
  sorry

end angle_between_vectors_l46_46546


namespace special_dice_probability_l46_46699

def probability_of_sum_37 : ℚ := 1 / 18

theorem special_dice_probability :
  let face_numbers := [1, 4, 9, 16, 25, 36]
  ∃ (dice1 dice2 : List ℕ), 
    dice1.length = 6 ∧ dice2.length = 6 ∧
    (∀ n ∈ dice1, n ∈ face_numbers) ∧
    (∀ n ∈ dice2, n ∈ face_numbers) ∧
    (∀ n ∈ face_numbers, n ∈ dice1 ∨ n ∈ dice2) ∧
    ((1, 36) ∈ (dice1.bind (λ x, dice2.map (λ y, (x, y)))) ∨
     (36, 1) ∈ (dice1.bind (λ x, dice2.map (λ y, (x, y))))) ∧
    (2 = dice1.bind (λ x, dice2.map (λ y, (x, y))).count (λ p, p.1 + p.2 = 37)) →
  probability_of_sum_37 = 1 / 18 :=
by sorry

end special_dice_probability_l46_46699


namespace BC_length_l46_46089

variable (A B C D M : Type)
variable [IsTrapezoid A B C D] [IsPointOnLateralSide M A B]
variable (DM_perpendicular_AB : Perpendicular D M A B)
variable (MC_eq_CD : MC = CD)
variable (AD_eq_d : AD = d)

theorem BC_length (d : ℝ) (A B C D M : Point)
  (H1 : IsTrapezoid A B C D)
  (H2 : IsPointOnLateralSide M A B)
  (H3 : Perpendicular D M A B)
  (H4 : MC = CD)
  (H5 : AD = d) :
  BC = d / 2 :=
sorry

end BC_length_l46_46089


namespace perimeter_of_triangle_AF2B_l46_46531

theorem perimeter_of_triangle_AF2B (a : ℝ) (m n : ℝ) (F1 F2 A B : ℝ × ℝ) 
  (h_hyperbola : ∀ x y : ℝ, (x^2 - 4*y^2 = 4) ↔ (x^2 / 4 - y^2 = 1)) 
  (h_mn : m + n = 3) 
  (h_AF1 : dist A F1 = m) 
  (h_BF1 : dist B F1 = n) 
  (h_AF2 : dist A F2 = 4 + m) 
  (h_BF2 : dist B F2 = 4 + n) 
  : dist A F1 + dist A F2 + dist B F2 + dist B F1 = 14 :=
by
  sorry

end perimeter_of_triangle_AF2B_l46_46531


namespace angle_equality_l46_46772

variables {ABC : Type} [triangle ABC]
variables (I : point) (incircle_center : I)
variables (B1 A1 : point) (on_AC : line AC) (on_BC : line BC)
variables (O : point) (circumcircle_center_AIB : O)
variables (OB1A OA1B : angle)

-- Conditions
def is_incircle (I : point) [incircle_center : I] (B1 A1 : point) (AC BC : line) : Prop :=
  circle I touches AC at B1 ∧ circle I touches BC at A1

def is_circumcircle_center (O : point) (A I B : point) : Prop :=
  circumcircle O A I B

-- Statement to prove
theorem angle_equality 
  (is_incircle I incircle_center B1 A1 AC BC)
  (is_circumcircle_center O A I B)
  : OB1A = OA1B :=
sorry

end angle_equality_l46_46772


namespace line_parallel_eq_l46_46322

theorem line_parallel_eq (x y : ℝ) (h1 : 3 * x - y = 6) (h2 : x = -2 ∧ y = 3) :
  ∃ m b, m = 3 ∧ b = 9 ∧ y = m * x + b :=
by
  sorry

end line_parallel_eq_l46_46322


namespace intersection_A_B_l46_46928

variable (x : ℝ)

def A : set ℝ := { x | |x| < 3 }
def B : set ℝ := { x | x^2 - 4 * x + 3 < 0 }

theorem intersection_A_B :
  A ∩ B = { x | 1 < x ∧ x < 3 } :=
sorry

end intersection_A_B_l46_46928


namespace even_numbers_with_divisors_condition_l46_46023

open Nat

def is_even (n : ℕ) : Prop := n % 2 = 0

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, n % d = 0).length

theorem even_numbers_with_divisors_condition :
  ∀ n : ℕ, is_even n ∧ number_of_divisors n = n / 2 ↔ n = 8 ∨ n = 12 :=
by
  sorry

end even_numbers_with_divisors_condition_l46_46023


namespace lcm_48_180_l46_46044

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l46_46044


namespace lcm_48_180_eq_720_l46_46050

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l46_46050


namespace num_prime_divisors_50_fact_l46_46551
open Nat -- To simplify working with natural numbers

-- We define the prime numbers less than or equal to 50.
def primes_le_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- The problem statement: Prove that the number of prime divisors of 50! which are less than or equal to 50 is 15.
theorem num_prime_divisors_50_fact : (primes_le_50.length = 15) :=
by 
  -- Here we use sorry to skip the proof.
  sorry

end num_prime_divisors_50_fact_l46_46551


namespace magical_stack_266_l46_46659

theorem magical_stack_266 (n : ℕ) (A B : list ℕ) :
  let k := 89 in
  let m := k - 1 in
  let p := m / 2 in
  let n := k + p in
  2 * n = 266 ∧ n + 1 = B.head ∧ A.head = 1 ∧ A.mk_nth 45 = k := 
sorry

end magical_stack_266_l46_46659


namespace calculation_result_l46_46446

theorem calculation_result :
  abs (-8) + (-2011)^0 - 2 * Real.cos (Real.pi / 3) + (1 / 2)⁻¹ = 10 :=
by
  -- The statements from the conditions
  have h1 : abs (-8) = 8 := by sorry,
  have h2 : (-2011)^0 = 1 := by sorry,
  have h3 : Real.cos (Real.pi / 3) = 1 / 2 := by sorry,
  have h4 : (1 / 2)⁻¹ = 2 := by sorry,
  -- Using these statements to prove the final result
  calc abs (-8) + (-2011)^0 - 2 * Real.cos (Real.pi / 3) + (1 / 2)⁻¹
      = 8 + 1 - 2 * (1 / 2) + 2 : by rw [h1, h2, h3, h4]
  ... = 8 + 1 - 1 + 2 : by sorry
  ... = 10 : by sorry

end calculation_result_l46_46446


namespace collinear_projections_l46_46738

theorem collinear_projections 
  {α : Real}
  {A B C P A1 B1 C1 : Point}
  (hP_on_circumcircle : on_circumcircle P (triangle A B C))
  (hA1 : projection A1 P (line B C) α)
  (hB1 : projection B1 P (line C A) α)
  (hC1 : projection C1 P (line A B) α) : 
  collinear A1 B1 C1 := 
sorry

end collinear_projections_l46_46738


namespace find_fx_find_range_of_p_find_range_of_w_l46_46943

variables {a m n p w : ℝ}
variables {f g F : ℝ → ℝ}
hypothesis h₀ : a > 1
hypothesis h₁ : ∀ x, f(x) = log a (x + 1)
hypothesis h₂ : m > -1
hypothesis h₃ : (∀ x ∈ Icc m n, f(x) ∈ Icc (log a (p / m)) (log a (p / n)))
hypothesis h₄ : g(x) = log a (x^2 - 3x + 3)
hypothesis h₅ : F(x) = a^(f(x) - g(x))
hypothesis h₆ : ∀ x ∈ Ioi (-1 : ℝ), w ≥ F(x)

-- Prove #1: the analytic expression of f(x)
theorem find_fx : ∀ x, f(x) = log a (x + 1) := by sorry

-- Prove #2: the range of p
theorem find_range_of_p (h_discriminant : 1 + 4 * p > 0) (h_interval : -1/4 < p ∧ p < 0) :
  -1/4 < p ∧ p < 0 := by sorry

-- Prove #3: the range of w
theorem find_range_of_w (h_F_max : F (sqrt 7 - 1) = (2 * sqrt 7 + 5) / 3) :
  w ≥ (2 * sqrt 7 + 5) / 3 := by sorry

end find_fx_find_range_of_p_find_range_of_w_l46_46943


namespace inequality_2_pow_n_plus_2_gt_n_squared_l46_46559

theorem inequality_2_pow_n_plus_2_gt_n_squared (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := sorry

end inequality_2_pow_n_plus_2_gt_n_squared_l46_46559


namespace expression_is_integer_l46_46256

theorem expression_is_integer (a : ℤ) (n : ℕ) (h : n ≠ 1) :
  let expr := (a ^ (3 * n) / (a ^ n - 1) + 1 / (a ^ n + 1)) - (a ^ (2 * n) / (a ^ n + 1) + 1 / (a ^ n - 1))
  in expr ∈ ℤ := by
sorry

end expression_is_integer_l46_46256


namespace trapezoid_bc_length_l46_46081

theorem trapezoid_bc_length
  (A B C D M : Point)
  (d : Real)
  (h_trapezoid : IsTrapezoid A B C D)
  (h_M_on_AB : OnLine M A B)
  (h_DM_perp_AB : Perpendicular D M A B)
  (h_MC_eq_CD : Distance M C = Distance C D)
  (h_AD_eq_d : Distance A D = d) :
  Distance B C = d / 2 := by
  sorry

end trapezoid_bc_length_l46_46081


namespace find_constant_k_eq_l46_46026

theorem find_constant_k_eq : ∃ k : ℤ, (-x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4)) ↔ (k = -17) :=
by
  sorry

end find_constant_k_eq_l46_46026


namespace probability_coin_heads_and_die_two_l46_46150

namespace CoinDieProbability

def fair_coin := {head, tail} -- Define the possible outcomes for a fair coin
def die := {1, 2, 3, 4, 5, 6} -- Define the possible outcomes for a regular six-sided die

def successful_outcome := (head, 2) -- Define the successful outcome

def total_outcomes := (fair_coin × die) -- Define the sample space of total outcomes

def probability_successful_outcome (s : total_outcomes) : Prop :=
  (s = successful_outcome)

theorem probability_coin_heads_and_die_two :
  (MeasureTheory.Probability (s in total_outcomes, probability_successful_outcome s) = 1 / 12) :=
sorry

end CoinDieProbability

end probability_coin_heads_and_die_two_l46_46150


namespace problem_1_problem_2_l46_46117

theorem problem_1 (n : ℕ) (a : ℕ → ℤ)
  (h1 : ∑ i in finset.range (n+1), nat.choose n i = 64)
  (h2 : (2 * (x:ℝ) - 3)^n = ∑ i in finset.range (n+1), a i * (x - 1)^i) :
  a 2 = 60 :=
sorry

theorem problem_2 (n : ℕ) (a : ℕ → ℤ)
  (h1 : ∑ i in finset.range (n + 1), nat.choose n i = 64)
  (h2 : (2 * (x : ℝ) - 3)^n = ∑ i in finset.range (n + 1), a i * (x - 1)^i) :
  ∑ i in finset.range (n + 1), |a i| = 729 :=
sorry

end problem_1_problem_2_l46_46117


namespace arithmetic_and_geometric_sequence_l46_46575

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_seq {α : Type*} [Field α] (seq : ℕ → α) : Prop :=
  ∀ n m k : ℕ,
    seq m * seq k = seq n * seq (m + k - n)

theorem arithmetic_and_geometric_sequence (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith: arithmetic_seq a 1)
  (h_sum: a 3 + a 10 = 15)
  (h_geom: is_geometric_seq (fun n => a n) 1 4 11)
  : (∀ n : ℕ, a n = n + 1) ∧
    (∀ n : ℕ, let b n := 1 / (a n * a (n + 1))
              in ∑ k in finset.range n, b k = n / (2 * (n + 2))) :=
by
  sorry

end arithmetic_and_geometric_sequence_l46_46575


namespace lcm_48_180_l46_46045

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l46_46045


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46349

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46349


namespace arithmetic_sequence_sum_remainder_l46_46717

theorem arithmetic_sequence_sum_remainder 
  (a d : ℕ) (n : ℤ) 
  (h₁ : a = 3) 
  (h₂ : d = 8) 
  (h₃ : n ≥ 1) 
  (h₄ : a + (n - 1) * d = 283) : 
  (∑ i in finset.range n, a + i * d) % 8 = 4 := 
sorry

end arithmetic_sequence_sum_remainder_l46_46717


namespace cheryl_used_material_l46_46389

theorem cheryl_used_material
    (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
    (h1 : material1 = 5/9)
    (h2 : material2 = 1/3)
    (h_lf : leftover = 8/24) :
    material1 + material2 - leftover = 5/9 :=
by
  sorry

end cheryl_used_material_l46_46389


namespace cube_sphere_surface_area_l46_46911

open Real

noncomputable def cube_edge_length := 1
noncomputable def cube_space_diagonal := sqrt 3
noncomputable def sphere_radius := cube_space_diagonal / 2
noncomputable def sphere_surface_area := 4 * π * (sphere_radius ^ 2)

theorem cube_sphere_surface_area :
  sphere_surface_area = 3 * π :=
by
  sorry

end cube_sphere_surface_area_l46_46911


namespace tangent_line_at_one_is_correct_l46_46665

theorem tangent_line_at_one_is_correct :
  let y := λ x : ℝ, x^3 - 1 in
  ∀ x : ℝ, 
  ∀ y' : ℝ, 
  (∀ x : ℝ, y' = 3 * x^2) → -- definition of the derivative
  let m := y' 1 in
  let point := (1, y 1) in
  (m = 3) → (point = (1, 0)) → -- slope and point at x = 1
  (y = 3 * x - 3) := 
by
  intros y x y' h_deriv m point h_m h_point
  rw h_point
  sorry

end tangent_line_at_one_is_correct_l46_46665


namespace Youngbin_combinations_l46_46309

theorem Youngbin_combinations : nat.choose 3 2 = 3 := by
  -- Proof steps will be provided here
  sorry

end Youngbin_combinations_l46_46309


namespace female_managers_count_l46_46170

variable (E M F FM : ℕ)

-- Conditions
def female_employees : Prop := F = 750
def fraction_managers : Prop := (2 / 5 : ℚ) * E = FM + (2 / 5 : ℚ) * M
def total_employees : Prop := E = M + F

-- Proof goal
theorem female_managers_count (h1 : female_employees F) 
                              (h2 : fraction_managers E M FM) 
                              (h3 : total_employees E M F) : 
  FM = 300 := 
sorry

end female_managers_count_l46_46170


namespace a_parallel_b_a_perp_cond_l46_46983

variables {α : Type*}

def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)

-- The parallel condition proof statement
theorem a_parallel_b (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 2 * Real.pi) 
    (h_parallel : ∃ k : ℝ, vector_a α = (k * (-1/2), k * (Real.sqrt 3 / 2))) :
    α = 2 * Real.pi / 3 ∨ α = 5 * Real.pi / 3 :=
sorry

-- The perpendicular condition proof statement
theorem a_perp_cond (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 2 * Real.pi) 
    (h_perpendicular : by exact ((Real.sqrt 3 * vector_a α + vector_b) ⬝ (vector_a α - Real.sqrt 3 * vector_b) = 0)) :
    Real.tan α = Real.sqrt 3 / 3 :=
sorry

end a_parallel_b_a_perp_cond_l46_46983


namespace yoonseok_handshakes_l46_46750

-- Conditions
def totalFriends : ℕ := 12
def yoonseok := "Yoonseok"
def adjacentFriends (i : ℕ) : Prop := i = 1 ∨ i = (totalFriends - 1)

-- Problem Statement
theorem yoonseok_handshakes : 
  ∀ (totalFriends : ℕ) (adjacentFriends : ℕ → Prop), 
    totalFriends = 12 → 
    (∀ i, adjacentFriends i ↔ i = 1 ∨ i = (totalFriends - 1)) → 
    (totalFriends - 1 - 2 = 9) := by
  intros totalFriends adjacentFriends hTotal hAdjacent
  have hSub : totalFriends - 1 - 2 = 9 := by sorry
  exact hSub

end yoonseok_handshakes_l46_46750


namespace apple_selling_price_l46_46809

noncomputable def cost_price : ℝ := 16
noncomputable def loss_fraction : ℝ := 1/6
noncomputable def loss : ℝ := loss_fraction * cost_price
noncomputable def selling_price : ℝ := cost_price - loss

theorem apple_selling_price :
  selling_price = 13.33 :=
by
  have CP := cost_price
  have Loss := loss
  have SP := selling_price
  rw [←SP, ←CP, ←Loss]
  calc
    SP = CP - Loss : by rw SP
    ... = 16 - (1 / 6 * 16) : by rw [CP, Loss]
    ... = 16 - 2.67 : by norm_num
    ... = 13.33 : by norm_num

end apple_selling_price_l46_46809


namespace minimum_value_l46_46949

open Real

noncomputable def curve (x : ℝ) : ℝ := x^3 - 2 * x^2 + 2
noncomputable def tangent (x : ℝ) : ℝ := 4 * x - 6
def line (m n l : ℝ) (x y : ℝ) : Prop := m * x + n * y = l

theorem minimum_value (m n l : ℝ) (A : ℝ × ℝ)
  (A_on_curve : A.2 = curve A.1)
  (tangent_eq : ∀ x, ∃ k, tangent x = curve x + k * (x - A.1))
  (A_on_line : line m n l A.1 A.2)
  (m_pos : 0 < m) (n_pos : 0 < n) :
  ∃ k : ℝ, (m * A.1 + n * A.2 = l) ∧ ((1 / m) + (2 / n) = k) ∧ k = 6 + 4 * sqrt 2 :=
sorry

end minimum_value_l46_46949


namespace no_perpendicular_moments_l46_46000

def angle_minute_hand (m : ℕ) : ℝ :=
  6 * m

def angle_hour_hand (h m : ℕ) : ℝ :=
  30 * h + m / 2

theorem no_perpendicular_moments :
  ∀ (h m : ℕ), (h < 12) → (m < 60) →
  (|angle_hour_hand h m - angle_minute_hand m| = 90 ∨ 
   |angle_hour_hand h m - angle_minute_hand m| = 270) →
  False :=
begin
  intros h m h_lt_12 m_lt_60 angle_condition,
  sorry
end

end no_perpendicular_moments_l46_46000


namespace upper_base_length_l46_46091

structure Trapezoid (A B C D M : Type) :=
  (on_lateral_side : ∀ {AB DM}, DM ⊥ AB)
  (perpendicular : ∀ {DM AB}, DM ⊥ AB)
  (equal_segments : MC = CD)
  (AD_length : AD = d)


theorem upper_base_length {A B C D M : Type} [trapezoid : Trapezoid A B C D M] :
  BC = d / 2 := 
begin
  sorry
end

end upper_base_length_l46_46091


namespace least_number_remainder_l46_46390

theorem least_number_remainder (n : ℕ) :
  (∀ d ∈ [12, 15, 20, 54], n % d = 5) → n = 545 :=
by
  intro h
  have h₁ : n % 12 = 5 := (List.mem_cons_self 12 _).mp (h 12 (List.mem_cons_self 12 _))
  have h₂ : n % 15 = 5 := (List.mem_cons_self 15 _).mpr (List.mem_cons_cons_self 12 15 54).mpr (h 15 (List.mem_cons_cons_self 12 15 54).mpr (List.mem_cons_self 12 15))
  have h₃ : n % 20 = 5 := (List.mem_cons_self 20 _).mpr (List.mem_cons_cons_self 12 15 54).mpr (List.mem_cons_cons_self 12 15 54).mpr (h 20 (List.mem_cons_cons_self 12 20 54).mpr (List.mem_cons_cons_self 15 20 54).mpr (List.mem_cons_cons_self 12 15 20)))
  have h₄ : n % 54 = 5 := (List.mem_cons_self 54 _).mpr (List.mem_cons_cons_self 12 15 54).mpr (List.mem_cons_cons_self 20 54).mpr (h 54 (List.mem_cons_cons_self 54 54).mpr (List.mem_cons_self 54 54))
  let m := Nat.lcm (Nat.lcm (Nat.lcm 12 15) 20) 54
  have h₅ : 540 = m := by
    rw [Nat.lcm_assoc, Nat.lcm_assoc, Nat.lcm_comm 20 54, ← Nat.lcm_assoc, Nat.lcm_comm 12 (Nat.lcm 15 54)]
    rfl
  rw [h₁, h₂, h₃, h₄] at this
  exact Nat.add_left_inj 5 540 545 (h m).symm
  rfl

end least_number_remainder_l46_46390


namespace one_horse_one_bag_l46_46385

theorem one_horse_one_bag {d: ℕ} (h: d = 15) : 
  (λ h : d = 15, (15 * d - 15 * d / 15 = 1)) = true :=
by
  intros
  -- proof omitted
  sorry

end one_horse_one_bag_l46_46385


namespace interval_satisfies_inequality_l46_46881

theorem interval_satisfies_inequality :
  { x : ℝ | x ∈ [-1, -1/3) ∪ (-1/3, 0) ∪ (0, 1) ∪ (1, ∞) } =
  { x : ℝ | x^2 + 2*x^3 - 3*x^4 ≠ 0 ∧ x + 2*x^2 - 3*x^3 ≠ 0 ∧ (x >= -1 ∧ (x < 1 ∨ x > -1/3)) ∧ 
            x^2 + 2*x^3 - 3*x^4 / (x + 2*x^2 - 3*x^3) ≥ -1 } := sorry

end interval_satisfies_inequality_l46_46881


namespace palm_meadows_total_beds_l46_46833

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end palm_meadows_total_beds_l46_46833


namespace equilateral_triangle_in_pentagon_l46_46634

theorem equilateral_triangle_in_pentagon (P : Type) [pentagon P] :
  (∀ (p : P), convex p ∧ equilateral p → ∃ (T : Type) [triangle T], equilateral T ∧ (∃ (s : side T) (s' : side p), s = s' ∧ inside T p)) :=
by sorry

end equilateral_triangle_in_pentagon_l46_46634


namespace rotated_angle_new_measure_l46_46296

theorem rotated_angle_new_measure (initial_angle : ℝ) (rotation : ℝ) (final_angle : ℝ) :
  initial_angle = 60 ∧ rotation = 300 → final_angle = 120 :=
by
  intros h
  sorry

end rotated_angle_new_measure_l46_46296


namespace smallest_d_for_inverse_l46_46617

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 1

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≠ x2 → (d ≤ x1) → (d ≤ x2) → g x1 ≠ g x2) ∧ d = 3 :=
by
  sorry

end smallest_d_for_inverse_l46_46617


namespace abs_a1_plus_abs_a2_to_abs_a6_l46_46491

theorem abs_a1_plus_abs_a2_to_abs_a6 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ)
  (h : (2 - x) ^ 6 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6) :
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 :=
sorry

end abs_a1_plus_abs_a2_to_abs_a6_l46_46491


namespace sum_of_n_with_max_f_eq_18_l46_46233

def is_valid_three_digit (n : ℕ) : Prop :=
  n / 100 > 0 ∧ n / 100 < 10 ∧ (n % 10 ≠ (n / 10) % 10 ∨ (n / 100) ≠ (n / 10) % 10)

def f (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd
    (100 * a + 10 * b + c)
    (100 * a + 10 * c + b))
    (100 * b + 10 * a + c))
    (100 * b + 10 * c + a))
    (100 * c + 10 * a + b))
    (100 * c + 10 * b + a)

theorem sum_of_n_with_max_f_eq_18 : 
  (∀ n : ℕ, is_valid_three_digit n → f(n) = 18) → 
  (Finset.sum (Finset.filter (λ n, is_valid_three_digit n ∧ f(n) = 18) (Finset.range 1000)) = 5994) := 
sorry

end sum_of_n_with_max_f_eq_18_l46_46233


namespace impurities_removed_l46_46108

def initial_mass : ℝ := 1000 -- kg
def impurity_percentage : ℝ := 5.95 / 100
def final_iron_percentage : ℝ := 99 / 100
def initial_impurities_mass : ℝ := initial_mass * impurity_percentage
def initial_iron_mass : ℝ := initial_mass - initial_impurities_mass

def final_mass : ℝ := initial_iron_mass / final_iron_percentage
def final_impurities_mass : ℝ := final_mass - initial_iron_mass

theorem impurities_removed : final_iron_percentage = 99 /. 100 -> impurity_percentage = 5.95 / 100 -> initial_mass = 1000 ->
  (initial_impurities_mass - final_impurities_mass) = 50 :=
by
  -- Sorry to skip the proof steps
  sorry

end impurities_removed_l46_46108


namespace sum_faces_edges_vertices_eq_26_l46_46337

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46337


namespace kay_age_l46_46597

/-- Let K be Kay's age. If the youngest sibling is 5 less 
than half of Kay's age, the oldest sibling is four times 
as old as the youngest sibling, and the oldest sibling 
is 44 years old, then Kay is 32 years old. -/
theorem kay_age (K : ℕ) (youngest oldest : ℕ) 
  (h1 : youngest = (K / 2) - 5)
  (h2 : oldest = 4 * youngest)
  (h3 : oldest = 44) : K = 32 := 
by
  sorry

end kay_age_l46_46597


namespace trajectory_equation_line_equation_l46_46123

theorem trajectory_equation (E : Set (ℝ × ℝ)) :
  (∀ x y : ℝ, ((x + sqrt 3)^2 + y^2 = 16) ∧ ((√3, 0) ∈ E) ∧ (E ∈ Ellipse) → 
    (∃ a, a = 4 ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 = 1))) :=
sorry

theorem line_equation (A B : ℝ × ℝ) (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  (∀ P₁ P₂: ℝ × ℝ, 
    (P₁.1 + √3, P₁.2 = 0) ∧ (P₁ ∈ Circle(O, 16)) ∧
    (P₂.1 ≠ P₁.1 ∧ P₂.2 ≠ P₁.2) ∧ (O = (0,0)) ∧
    (Area (Triangle A O B) = 4 / 5) →
    ∃ m b, (l = λ x y, y = m * x + b) ∧ 
    ((m = 1 ∨ m = -1) ∧ (l = λ x y, x + y - 1 = 0 ∨ l = λ x y, x - y - 1 = 0))) :=
sorry

end trajectory_equation_line_equation_l46_46123


namespace trajectory_midpoint_parabola_l46_46977

theorem trajectory_midpoint_parabola : 
  (∀ (p q : ℝ), q^2 = 8 * p →
  ∀ (x y : ℝ), x = (p + 2) / 2 ∧ y = q / 2 →
  y^2 = 4 * x - 4) := 
begin
  intros p q h₁ x y h₂,
  obtain ⟨h₃, h₄⟩ := h₂,
  have h₅ : q = 2 * y := by linarith,
  have h₆ : p = 2 * x - 2 := by linarith,
  rw [h₅, h₆] at h₁,
  simp_rw ← sub_eq_zero at *,
  ring at *,
  sorry,
end

end trajectory_midpoint_parabola_l46_46977


namespace complement_intersection_l46_46143

open Set

noncomputable def M : Set ℝ := {x | ∃ y, y = Real.ln (1 - x)}
noncomputable def N : Set ℝ := {x | 2^(x * (x - 2)) < 1}
noncomputable def CU_M : Set ℝ := compl M

theorem complement_intersection :
  (CU_M ∩ N) = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end complement_intersection_l46_46143


namespace triangle_area_computation_l46_46885

-- Define the lines as functions
def line1 (x y : ℝ) : Prop := y = 2 * x + 2
def line2 (x y : ℝ) : Prop := 2 * y = x + 1

-- Define the intersection points of the lines with the axes
def intercepts : set (ℝ × ℝ) :=
  {p | (p = (0, 2)) ∨ (p = (-1, 0)) ∨ (p = (0, 1/2))}

-- Define the vertices of the triangle
def vertices : set (ℝ × ℝ) :=
  {p | (p = (-1, 0)) ∨ (p = (0, 0)) ∨ (p = (0, 2))}

-- Define the area calculation
def triangle_area (a b h : ℝ) : ℝ := 1 / 2 * a * b

theorem triangle_area_computation :
  triangle_area 2 1 (line1 0 2) = 1 := 
therefore the triangle area is known.
by
  sorry

end triangle_area_computation_l46_46885


namespace term_2019_no_squares_l46_46415

/-- Define the sequence of natural numbers excluding perfect squares. -/
def no_perfect_squares_seq : ℕ → ℕ :=
  λ n, n + Nat.sqrt_floor(n)

/-- Prove that the 2019th term of the sequence obtained by removing all perfect squares is 2063. -/
theorem term_2019_no_squares : no_perfect_squares_seq 2019 = 2063 :=
  sorry

end term_2019_no_squares_l46_46415


namespace value_range_of_f_l46_46312

noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

theorem value_range_of_f : Set.range (fun x => f x) ∩ Set.Icc 3 6 = Set.Icc 1 4 :=
by
  sorry

end value_range_of_f_l46_46312


namespace truck_driver_needs_more_gallons_l46_46426

-- Define the conditions
def miles_per_gallon : ℕ := 3
def total_distance : ℕ := 90
def current_gallons : ℕ := 12
def can_cover_distance : ℕ := miles_per_gallon * current_gallons
def additional_distance_needed : ℕ := total_distance - can_cover_distance

-- Define the main theorem
theorem truck_driver_needs_more_gallons :
  additional_distance_needed / miles_per_gallon = 18 :=
by
  -- Placeholder for the proof
  sorry

end truck_driver_needs_more_gallons_l46_46426


namespace problem_I_problem_II_l46_46952

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |2 * x + a|

-- Problem (I): Inequality solution when a = 1
theorem problem_I (x : ℝ) : f x 1 ≥ 5 ↔ x ∈ (Set.Iic (-4 / 3) ∪ Set.Ici 2) :=
sorry

-- Problem (II): Range of a given the conditions
theorem problem_II (x₀ : ℝ) (a : ℝ) (h : f x₀ a + |x₀ - 2| < 3) : -7 < a ∧ a < -1 :=
sorry

end problem_I_problem_II_l46_46952


namespace inequality_proof_l46_46609

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 :=
sorry

end inequality_proof_l46_46609


namespace collinear_points_l46_46200

-- Vector definitions and collinearity condition
structure Point (α : Type*) := (x : α) (y : α)
def O := Point ℝ 0 0
def A (x : ℝ) := Point ℝ 1 (Real.cos x)
def B (x : ℝ) := Point ℝ (1 + Real.sin x) (Real.cos x)
def C (x : ℝ) := Point ℝ (1 + (2/3) * Real.sin x) (Real.cos x)

def vectorO_to (P : Point ℝ) : (ℝ × ℝ) := (P.x, P.y)
def vector_diff (P Q : Point ℝ) : (ℝ × ℝ) := (P.x - Q.x, P.y - Q.y)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Function as defined in the problem
def f (x m : ℝ) := 
  let OA := vectorO_to (A x)
  let OB := vectorO_to (B x)
  let OC := vectorO_to (C x)
  let AB := vector_diff (B x) (A x)
  (dot_product OA OC) + (2 * m + (1/3)) * (vector_length AB) + m^2

noncomputable def find_m (x : ℝ) : ℝ → Prop :=
  λ m, f x m = 5

-- Lean theorem statement
theorem collinear_points (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π/2) : 
  ∃ m, find_m x m :=
sorry

end collinear_points_l46_46200


namespace probability_at_least_one_of_A_or_B_selected_l46_46263

/-- 
Given 6 experts including A and B, and selecting 2 out of these 6 experts,
prove that the probability that at least one of A and B is selected is 3/5.
-/

theorem probability_at_least_one_of_A_or_B_selected :
  let total_ways := Nat.choose 6 2 in
  let ways_neither_AB := Nat.choose 4 2 in
  1 - (ways_neither_AB / total_ways : ℚ) = 3 / 5 :=
by
  sorry

end probability_at_least_one_of_A_or_B_selected_l46_46263


namespace sum_of_elements_in_10th_set_is_correct_l46_46103

noncomputable def sum_of_elements_in_10th_set : Nat :=
  let n := 10
  let previous_sets_sum := n * (n - 1) * (n + 1) / 6
  let first_element := previous_sets_sum + 1
  let number_of_elements := 10 + 2 * 9
  let last_element := first_element + number_of_elements - 1
  let s_10 := number_of_elements * (first_element + last_element) / 2
  s_10

theorem sum_of_elements_in_10th_set_is_correct : sum_of_elements_in_10th_set = 5026 := by
  -- Definitions used in the conditions
  have sum_triangular_numbers : ∀ k, (k * (k + 1) * (k + 2)) / 6 = k * (k + 1) * (k + 2) / 6 :=
    λ k, rfl

  have number_of_elements_10th_set : 10 + 2 * 9 = 28 := by norm_num
  have sum_previous_sets : (9 * 10 * 11 / 6) = 165 := by norm_num
  have first_element_10th_set := 166
  have last_element_10th_set := 193
  have sum_10th_set := 14 * 359 = 5026 := by norm_num

  sorry

end sum_of_elements_in_10th_set_is_correct_l46_46103


namespace radius_of_inner_tangent_circle_l46_46420

theorem radius_of_inner_tangent_circle (side_length : ℝ) (num_semicircles_per_side : ℝ) (semicircle_radius : ℝ)
  (h_side_length : side_length = 4) (h_num_semicircles_per_side : num_semicircles_per_side = 3) 
  (h_semicircle_radius : semicircle_radius = side_length / (2 * num_semicircles_per_side)) :
  ∃ (inner_circle_radius : ℝ), inner_circle_radius = 7 / 6 :=
by
  sorry

end radius_of_inner_tangent_circle_l46_46420


namespace find_c_value_l46_46463

theorem find_c_value (c : ℝ) :
  (∀ x : ℝ, x ∈ set.Ioo (-9 / 2) 1 → x * (2 * x + 3) < c) ↔ c = -6 :=
by sorry

end find_c_value_l46_46463


namespace monotonicity_case1_monotonicity_case2_lower_bound_l46_46969

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 (a : ℝ) (h : a ≤ 0) : 
  ∀ x y : ℝ, x < y → f a x > f a y := 
by
  sorry

theorem monotonicity_case2 (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, x < Real.log (1 / a) → f a x > f a (Real.log (1 / a)) ∧ 
           x > Real.log (1 / a) → f a x < f a (Real.log (1 / a)) := 
by
  sorry

theorem lower_bound (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ,  f a x > 2 * Real.log a + 3 / 2 := 
by
  sorry

end monotonicity_case1_monotonicity_case2_lower_bound_l46_46969


namespace day_of_week_Jan_1_2000_l46_46628

theorem day_of_week_Jan_1_2000 :
  ∀ (days_in_year : ℕ) (leap_years : ℕ),
  let days := 7 * days_in_year + 3 * (days_in_year + 1) in
  days % 7 = 1 → 
  "Monday" + 1 = "Tuesday" :=
by
  intro days_in_year leap_years days h
  cases days % 7 with
  | 0 => sorry
  | 1 => sorry
  | 2 => sorry
  | 3 => sorry
  | 4 => sorry
  | 5 => sorry
  | 6 => sorry

end day_of_week_Jan_1_2000_l46_46628


namespace sum_faces_edges_vertices_l46_46359

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46359


namespace BE_eq_AH_l46_46615

variables {A B C O D E Z H : Point}
variables {circumcircle : Circle}
variables {arcAB : Arc}

-- Conditions
axiom AB_lt_AC_lt_BC : Distance(A, B) < Distance(A, C) ∧ Distance(A, C) < Distance(B, C)
axiom O_circumcenter : is_circumcenter O (Triangle.mk A B C)
axiom D_midpoint_arcAB : is_midpoint_of_arc D (arcAB not_containing C)
axiom E_intersection_AD_BC : is_intersection_point E (Line.mk A D) (Line.mk B C)
axiom Z_circumcircle_BDE : lies_on Z (circumcircle_of (Triangle.mk B D E)) (Line.mk A B)
axiom H_circumcircle_ADZ : lies_on H (circumcircle_of (Triangle.mk A D Z)) (Line.mk A C)

-- Proof goal
theorem BE_eq_AH : Distance(B, E) = Distance(A, H) :=
  sorry

end BE_eq_AH_l46_46615


namespace ratio_surface_area_of_cube_and_tetrahedron_l46_46410

theorem ratio_surface_area_of_cube_and_tetrahedron 
  (s : ℝ) 
  (h1 : s = 2) 
  (h2 : ∀ v1 v2 v3 v4 : ℝ × ℝ × ℝ, is_regular_tetrahedron v1 v2 v3 v4) : 
  ratio_surface_area (cube_surface_area s) (tetrahedron_surface_area (2 * real.sqrt 2)) = real.sqrt 3 := 
by 
  sorry

end ratio_surface_area_of_cube_and_tetrahedron_l46_46410


namespace volume_change_l46_46554

def initial_volume (a b c : ℝ) : ℝ := a * b * c

def differential_volume (a b c da db dc : ℝ) : ℝ :=
  b * c * da + a * c * db + a * b * dc

theorem volume_change (a b c da db dc : ℝ)
  (ha : a = 8) (hb : b = 6) (hc : c = 3)
  (hda : da = 0.1) (hdb : db = 0.05) (hdc : dc = -0.15) :
  differential_volume a b c da db dc = -4.2 :=
by
  rw [ha, hb, hc, hda, hdb, hdc]
  show 6 * 3 * 0.1 + 8 * 3 * 0.05 + 8 * 6 * -0.15 = -4.2
  calc
    6 * 3 * 0.1 + 8 * 3 * 0.05 + 8 * 6 * -0.15
    = 1.8 + 1.2 - 7.2 : by norm_num
    = -4.2 : by norm_num

end volume_change_l46_46554


namespace monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46964

def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_when_non_positive (a : ℝ) (h_a : a ≤ 0) : ∀ x : ℝ, (a * Real.exp x - 1) ≤ 0 := 
by 
  intro x
  sorry

theorem monotonicity_when_positive (a : ℝ) (h_a : a > 0) : 
  ∀ x : ℝ, 
    (x < Real.log (1 / a) → (a * Real.exp x - 1) < 0) ∧ 
    (x > Real.log (1 / a) → (a * Real.exp x - 1) > 0) := 
by 
  intro x
  sorry

theorem inequality_when_positive (a : ℝ) (h_a : a > 0) : ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := 
by 
  intro x
  sorry

end monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46964


namespace range_satisfying_inequality_l46_46912

noncomputable def f : ℝ → ℝ := sorry 

axiom f_monotone_inc (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x < y → f(x) < f(y)
axiom f_2_eq_1 : f 2 = 1
axiom f_mul (x y : ℝ) : f(x * y) = f(x) + f(y)

theorem range_satisfying_inequality :
  {x : ℝ | 0 < x ∧ f(x) + f(x - 3) ≤ 2} = {x : ℝ | 3 < x ∧ x < 4} :=
by
  sorry

end range_satisfying_inequality_l46_46912


namespace people_could_not_take_bus_l46_46763

theorem people_could_not_take_bus
  (carrying_capacity : ℕ)
  (first_pickup_ratio : ℚ)
  (first_pickup_people : ℕ)
  (people_waiting : ℕ)
  (total_on_bus : ℕ)
  (additional_can_carry : ℕ)
  (people_could_not_take : ℕ)
  (h1 : carrying_capacity = 80)
  (h2 : first_pickup_ratio = 3/5)
  (h3 : first_pickup_people = carrying_capacity * first_pickup_ratio.to_nat)
  (h4 : first_pickup_people = 48)
  (h5 : total_on_bus = first_pickup_people)
  (h6 : additional_can_carry = carrying_capacity - total_on_bus)
  (h7 : additional_can_carry = 32)
  (h8 : people_waiting = 50)
  (h9 : people_could_not_take = people_waiting - additional_can_carry)
  (h10 : people_could_not_take = 18) : 
  people_could_not_take = 18 :=
by
  sorry -- proof is left for another step

end people_could_not_take_bus_l46_46763


namespace smallest_value_of_sum_l46_46995

theorem smallest_value_of_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 3 * a = 4 * b ∧ 4 * b = 7 * c) : a + b + c = 61 :=
sorry

end smallest_value_of_sum_l46_46995


namespace symmetric_line_eq_l46_46925

theorem symmetric_line_eq (x y : ℝ) :
  (∀ x y, 2 * x - y + 1 = 0 → y = -x) → (∀ x y, x - 2 * y + 1 = 0) :=
by sorry

end symmetric_line_eq_l46_46925


namespace sum_of_faces_edges_vertices_of_rect_prism_l46_46350

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l46_46350


namespace find_k_range_l46_46499

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0
def line_eq (x y k : ℝ) : Prop := 3*x - 4*y + k = 0

theorem find_k_range :
  (∃ x1 y1, circle_eq x1 y1 ∧ distance_point_line x1 y1 k = 1) ∧
  (∃ x2 y2, circle_eq x2 y2 ∧ distance_point_line x2 y2 k = 1) →
  k ∈ Ioo (-17) (-7) ∪ Ioo (3) (13) :=
by {
  sorry
}

end find_k_range_l46_46499


namespace monotonicity_case1_monotonicity_case2_lower_bound_l46_46970

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 (a : ℝ) (h : a ≤ 0) : 
  ∀ x y : ℝ, x < y → f a x > f a y := 
by
  sorry

theorem monotonicity_case2 (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, x < Real.log (1 / a) → f a x > f a (Real.log (1 / a)) ∧ 
           x > Real.log (1 / a) → f a x < f a (Real.log (1 / a)) := 
by
  sorry

theorem lower_bound (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ,  f a x > 2 * Real.log a + 3 / 2 := 
by
  sorry

end monotonicity_case1_monotonicity_case2_lower_bound_l46_46970


namespace palm_meadows_total_beds_l46_46834

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end palm_meadows_total_beds_l46_46834


namespace length_QF_l46_46534

-- Define parabola C as y^2 = 8x
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 * P.2 = 8 * P.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the condition that Q is on the parabola and the line PF in the first quadrant
def is_intersection_and_in_first_quadrant (Q : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  is_on_parabola Q ∧ Q.1 - Q.2 - 2 = 0 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the vector relation between P, Q, and F
def vector_relation (P Q F : ℝ × ℝ) : Prop :=
  let vPQ := (Q.1 - P.1, Q.2 - P.2)
  let vQF := (F.1 - Q.1, F.2 - Q.2)
  (vPQ.1^2 + vPQ.2^2) = 2 * (vQF.1^2 + vQF.2^2)

-- Lean 4 statement of the proof problem
theorem length_QF (Q : ℝ × ℝ) (P : ℝ × ℝ) :
  is_on_parabola Q ∧ is_intersection_and_in_first_quadrant Q P ∧ vector_relation P Q focus → 
  dist Q focus = 8 + 4 * Real.sqrt 2 :=
by
  sorry

end length_QF_l46_46534


namespace maximum_value_of_f_eq_2_l46_46523

noncomputable def f (a x : ℝ) : ℝ := -x^2 + 2 * a * x + 1 - a

theorem maximum_value_of_f_eq_2 (a : ℝ) :
  (∃ x ∈ set.Icc (0 : ℝ) (1 : ℝ), ∀ y ∈ set.Icc (0 : ℝ) (1 : ℝ), f a y ≤ f a x) ∧ f a (classical.some (exists_max f 0 1)) = 2 → (a = -1 ∨ a = 2) :=
by
  sorry

end maximum_value_of_f_eq_2_l46_46523


namespace sum_of_a_is_9_33_l46_46879

noncomputable def sum_of_a : Real :=
  -- Conditions of the problem
  let equation_holds (a x : Real) : Prop :=
    (4 * Real.pi * a + Real.arcsin (Real.sin x) + 3 * Real.arccos (Real.cos x) - a * x) / (2 + Real.tan x ^ 2) = 0

  -- Function to check the number of solutions for a given a
  let has_three_solutions (a : Real) : Prop :=
    ∃ x1 x2 x3 : Real, 0 ≤ x1 ∧ x1 < Real.pi ∧ 0 ≤ x2 ∧ x2 < Real.pi ∧ 0 ≤ x3 ∧ x3 < Real.pi ∧
                      x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ 
                      equation_holds a x1 ∧ equation_holds a x2 ∧ equation_holds a x3
  
  -- Generate all positive a's satisfying the condition and sum them
  -- a values are known directly from the solution analysis done.
  let valid_as : List Real := [1, 3, 16 / 3]

  -- Sum the values and round to nearest hundredth
  (valid_as.map (λa, Real.round_to (Real.to_float a) 2)).sum

theorem sum_of_a_is_9_33 : sum_of_a = 9.33 :=
  sorry

end sum_of_a_is_9_33_l46_46879


namespace sqrt_abs_inequality_l46_46996

theorem sqrt_abs_inequality (x : ℝ) (h : -2 * x^2 + 5 * x - 2 > 0) : 
  sqrt (4 * x^2 - 4 * x + 1) + 2 * |x - 2| = 3 :=
sorry

end sqrt_abs_inequality_l46_46996


namespace brenda_ends_with_15_skittles_l46_46440

def initial_skittles : ℕ := 7
def skittles_bought : ℕ := 8

theorem brenda_ends_with_15_skittles : initial_skittles + skittles_bought = 15 := 
by {
  sorry
}

end brenda_ends_with_15_skittles_l46_46440


namespace area_of_square_with_diagonal_l46_46998

theorem area_of_square_with_diagonal (c : ℝ) : 
  (∃ (s : ℝ), 2 * s^2 = c^4) → (∃ (A : ℝ), A = (c^4 / 2)) :=
  by
    sorry

end area_of_square_with_diagonal_l46_46998


namespace ellen_golf_cart_trips_l46_46867

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l46_46867


namespace monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46962

def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_when_non_positive (a : ℝ) (h_a : a ≤ 0) : ∀ x : ℝ, (a * Real.exp x - 1) ≤ 0 := 
by 
  intro x
  sorry

theorem monotonicity_when_positive (a : ℝ) (h_a : a > 0) : 
  ∀ x : ℝ, 
    (x < Real.log (1 / a) → (a * Real.exp x - 1) < 0) ∧ 
    (x > Real.log (1 / a) → (a * Real.exp x - 1) > 0) := 
by 
  intro x
  sorry

theorem inequality_when_positive (a : ℝ) (h_a : a > 0) : ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := 
by 
  intro x
  sorry

end monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46962


namespace sum_of_faces_edges_vertices_l46_46369

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46369


namespace count_factors_that_are_multiples_of_3_l46_46561

theorem count_factors_that_are_multiples_of_3 :
  let n := 2^4 * 3^3 * 7 in
  ∃ (count : ℕ), (∀ d, d ∣ n → (∃ a b c, 0 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1 ∧ d = 2^a * 3^b * 7^c) → count = 30) :=
by {
  let n := 2^4 * 3^3 * 7,
  sorry
}

end count_factors_that_are_multiples_of_3_l46_46561


namespace intersection_point_and_distance_l46_46636

/-- Define the points A, B, C, D, and M based on the specified conditions. --/
def A := (0, 3)
def B := (6, 3)
def C := (6, 0)
def D := (0, 0)
def M := (3, 0)

/-- Define the equations of the circles. --/
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 2.25
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 25

/-- The point P that is one of the intersection points of the two circles. --/
def P := (2, 1.5)

/-- Define the line AD as the y-axis. --/
def AD := 0

/-- Calculate the distance from point P to the y-axis (AD). --/
def distance_to_ad (x : ℝ) := |x|

theorem intersection_point_and_distance :
  circle1 (2 : ℝ) (1.5 : ℝ) ∧ circle2 (2 : ℝ) (1.5 : ℝ) ∧ distance_to_ad 2 = 2 :=
by
  unfold circle1 circle2 distance_to_ad
  norm_num
  sorry

end intersection_point_and_distance_l46_46636


namespace integer_sum_19_l46_46271

variable (p q r s : ℤ)

theorem integer_sum_19 (h1 : p - q + r = 4) 
                       (h2 : q - r + s = 5) 
                       (h3 : r - s + p = 7) 
                       (h4 : s - p + q = 3) :
                       p + q + r + s = 19 :=
by
  sorry

end integer_sum_19_l46_46271


namespace area_increase_l46_46801

-- Defining the shapes and areas
def radius_large_side := 6
def radius_small_side := 4

def area_large_semicircles : ℝ := real.pi * (radius_large_side^2)
def area_small_semicircles : ℝ := real.pi * (radius_small_side^2)

-- The theorem statement
theorem area_increase : (area_large_semicircles / area_small_semicircles) = 2.25 → 
                         ((2.25 - 1) * 100) = 125 :=
by sorry

end area_increase_l46_46801


namespace petya_wins_l46_46393

def convex_100gon : Prop := ∃(P : Fin 100 → ℝ × ℝ),
  @Function.Injective _ _ P ∧
  ∀ i j k: Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k → ConvexHull ℝ (P '' {i, j, k}) = Polygon

def point_X_inside (P : Fin 100 → ℝ × ℝ) (X : ℝ × ℝ) : Prop :=
  X ∈ ConvexHull ℝ (Set.Range P)

def marking_strategy (P: Fin 100 → ℝ × ℝ) (X: ℝ × ℝ) : Prop :=
  ∀ strategy_Vasya : ℕ → Fin 100 → Fin 100,
  ∃ strategy_Petya : ℕ → Fin 100 → Fin 100,
  ∀ n : ℕ, ¬ (point_X_inside (λ i, if i ≤ n then P (strategy_Petya n i) else P (strategy_Vasya n i)) X)

theorem petya_wins 
  (P: Fin 100 → ℝ × ℝ) 
  (hP: convex_100gon)
  (X : ℝ × ℝ) 
  (hX: point_X_inside P X) : 
  marking_strategy P X := 
sorry

end petya_wins_l46_46393


namespace find_function_values_l46_46528

variable (m θ : ℝ)
variable (f : ℝ → ℝ)
variable (C a b c : ℝ)
variable (k : ℤ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_def : Prop := f x = (m + 2 * (cos x) ^ 2) * cos (2 * x + θ)

def f_pi_over_4_zero : Prop := f (π / 4) = 0

def theta_in_range : Prop := 0 < θ ∧ θ < π

def f_at_C_condition : Prop := f (C / 2 + π / 24) = -1 / 2

def sides_conditions : Prop := c = 1 ∧ a * b = 2 * sqrt 3

-- Prove
theorem find_function_values
  (f_odd : is_odd_function f)
  (f_def : f_def)
  (f_pi_over_4_zero : f_pi_over_4_zero)
  (theta_range : theta_in_range)
  (f_at_C_cond : f_at_C_condition)
  (sides_cond : sides_conditions) :
  f x = -1 / 2 * sin (4 * x) ∧
  ∃ k : ℤ, ∀ x, (f x = -1 / 2 * sin (4 * x)) ∧ (∃ k, x = k * π / 4) ∧ -1 / 2 * sin (4 * x) = 0 ∧
  ∃ k : ℤ, (∀ L, ∀ M, L = k * π / 2 + π / 8 ∧ M = k * π / 2 + 3 * π / 8) ∧
  ∀ π, ∀ sqrt_3, (π = 3 + sqrt 3) := 
  sorry

end find_function_values_l46_46528


namespace a_work_days_alone_l46_46735

-- Definitions based on conditions
def work_days_a   (a: ℝ)    : Prop := ∃ (x:ℝ), a = x
def work_days_b   (b: ℝ)    : Prop := b = 36
def alternate_work (a b W x: ℝ) : Prop := 9 * (W / 36 + W / x) = W ∧ x > 0

-- The main theorem to prove
theorem a_work_days_alone (x W: ℝ) (b: ℝ) (h_work_days_b: work_days_b b)
                          (h_alternate_work: alternate_work a b W x) : 
                          work_days_a a → a = 12 :=
by sorry

end a_work_days_alone_l46_46735


namespace focal_distance_of_curve_l46_46667

theorem focal_distance_of_curve : 
  (∀ θ : ℝ, let x := 5 * Real.cos θ in let y := 4 * Real.sin θ in true) → 
  ∃ d : ℝ, d = 6 := 
by 
  intro h 
  use 6 
  sorry

end focal_distance_of_curve_l46_46667


namespace distinct_divisors_count_l46_46913

theorem distinct_divisors_count (n : ℕ) (h_n : n ≥ 2)
  (p : ℕ → Prop) (H_prime : ∀ i, p i → nat.prime i)
  (k : ℕ) (prime_factors : fin k → ℕ) (exponents : fin k → ℕ)
  (prime_factor_cond : ∀ i, p (prime_factors i))
  (factorization_cond : n = (finset.range k).prod (λ i, (prime_factors i) ^ (exponents i))) :
  finset.card ((finset.range k).pi (λ i, finset.range (exponents i + 1))) = (finset.range k).prod (λ i, exponents i + 1) :=
by
  sorry

end distinct_divisors_count_l46_46913


namespace find_sum_xyz_l46_46979

-- Define the problem
def system_of_equations (x y z : ℝ) : Prop :=
  x^2 + x * y + y^2 = 27 ∧
  y^2 + y * z + z^2 = 9 ∧
  z^2 + z * x + x^2 = 36

-- The main theorem to be proved
theorem find_sum_xyz (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 18 :=
sorry

end find_sum_xyz_l46_46979


namespace sum_solutions_eq_sum_solutions_of_eq_zero_problem_solution_l46_46859

theorem sum_solutions_eq (x : ℚ) (h : (4 * x + 6) * (3 * x - 8) = 0) : x = -3/2 ∨ x = 8/3 :=
begin
  have h1 : 4 * x + 6 = 0 ∨ 3 * x - 8 = 0,
  { apply eq_zero_or_eq_zero_of_mul_eq_zero h },
  cases h1 with h2 h3,
  { left,
    linarith },
  { right,
    linarith }
end

theorem sum_solutions_of_eq_zero : (-3/2 : ℚ) + (8/3) = 7/6 :=
begin
  norm_num
end

theorem problem_solution :
  (let solutions := [(-3/2 : ℚ), (8/3 : ℚ)] in solutions.sum) = (7/6) :=
by
  have h1 : (-3/2 : ℚ) + (8/3) = 7/6, from sum_solutions_of_eq_zero,
  exact h1

end sum_solutions_eq_sum_solutions_of_eq_zero_problem_solution_l46_46859


namespace maximize_area_triangle_l46_46098

-- Define the given ellipse and the point P on it
structure Ellipse (a b : ℝ) := (center : ℝ × ℝ) (semi_major : ℝ) (semi_minor : ℝ)

-- Define the point on the ellipse
structure Point (x y : ℝ)

-- Define a function that checks if a point is on the ellipse
def is_on_ellipse (e : Ellipse a b) (p : Point) : Prop :=
  let Ellipse.center := (cx, cy)
  (p.x - cx)^2 / a^2 + (p.y - cy)^2 / b^2 = 1

-- Formalization of the problem statement
theorem maximize_area_triangle 
  (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (e : Ellipse a b)
  (P : Point)
  (hP : is_on_ellipse e P) :
  ∃ Q R : Point, 
    is_on_ellipse e Q ∧ is_on_ellipse e R ∧
    ∀ S T : Point, is_on_ellipse e S → is_on_ellipse e T →
      area_of_triangle P Q R ≥ area_of_triangle P S T :=
sorry

end maximize_area_triangle_l46_46098


namespace area_semicircles_percent_increase_l46_46797

noncomputable def radius_large_semicircle (length: ℝ) : ℝ := length / 2
noncomputable def radius_small_semicircle (width: ℝ) : ℝ := width / 2

noncomputable def area_semicircle (radius: ℝ) : ℝ := (real.pi * radius^2) / 2

theorem area_semicircles_percent_increase
  (length: ℝ) (width: ℝ)
  (h_length: length = 12) (h_width: width = 8) :
  let 
    large_radius := radius_large_semicircle length,
    small_radius := radius_small_semicircle width,
    area_large := 2 * area_semicircle large_radius,
    area_small := 2 * area_semicircle small_radius
  in
  (area_large / area_small - 1) * 100 = 125 :=
by
  sorry

end area_semicircles_percent_increase_l46_46797


namespace part_a_part_b_part_c_part_d_l46_46743

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem part_a (n : ℕ) (h1 : n < 1000) :
  (sum_of_digits n)^2 < 730 :=
sorry

theorem part_b (n : ℕ) (h1 : 1000 ≤ n) (h2: n < 10000) :
  (sum_of_digits n)^2 < n :=
sorry

theorem part_c (n : ℕ) (h1 : 10000 ≤ n) :
  (sum_of_digits n)^2 < n :=
sorry

theorem part_d (n : ℕ) :
  (sum_of_digits n)^2 = n → n = 1 ∨ n = 81 :=
sorry

end part_a_part_b_part_c_part_d_l46_46743


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l46_46494

variable (a b : ℝ)
variable (h : a < b)

theorem option_A_correct : a + 2 < b + 2 := by
  sorry

theorem option_B_correct : 3 * a < 3 * b := by
  sorry

theorem option_C_correct : (1 / 2) * a < (1 / 2) * b := by
  sorry

theorem option_D_incorrect : ¬(-2 * a < -2 * b) := by
  sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l46_46494


namespace students_just_passed_l46_46387

theorem students_just_passed (total_students first_div_percent second_div_percent : ℝ)
  (h_total_students: total_students = 300)
  (h_first_div_percent: first_div_percent = 0.29)
  (h_second_div_percent: second_div_percent = 0.54)
  (h_no_failures : total_students = 300) :
  ∃ passed_students, passed_students = total_students - (first_div_percent * total_students + second_div_percent * total_students) ∧ passed_students = 51 :=
by
  sorry

end students_just_passed_l46_46387


namespace sin_alpha_square_minus_sin_2alpha_l46_46547

theorem sin_alpha_square_minus_sin_2alpha (α : ℝ) (a b : ℝ) 
  (ha : a = (real.cos α, real.sin α)) 
  (hb : b = (2, 3)) 
  (hab : ∃ k : ℝ, a = k • b): 
  real.sin α ^ 2 - real.sin (2 * α) = -3 / 13 := 
by 
  sorry

end sin_alpha_square_minus_sin_2alpha_l46_46547


namespace f_inequality_l46_46957

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l46_46957


namespace find_second_number_l46_46682

theorem find_second_number (x y z : ℚ) (h₁ : x + y + z = 150) (h₂ : x = (3 / 4) * y) (h₃ : z = (7 / 5) * y) : 
  y = 1000 / 21 :=
by sorry

end find_second_number_l46_46682


namespace sum_first_10_terms_sequence_l46_46566

variable (a : ℕ → ℝ)

def sum_first_n_terms (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

theorem sum_first_10_terms_sequence :
  (∀ n : ℕ, a (n + 1) + a n = 2^n) →
  sum_first_n_terms a 10 = 682 :=
by
  sorry

end sum_first_10_terms_sequence_l46_46566


namespace spaceship_finds_alien_l46_46824

-- Definitions of the conditions
variables (u v : ℝ) 

-- Condition that the spaceship's speed is greater than 10 times the alien's speed
def spaceship_faster_than_alien (u v : ℝ) : Prop := v > 10 * u

-- Main statement to be proven
theorem spaceship_finds_alien (u v : ℝ) (h : spaceship_faster_than_alien u v) : 
  ∃ (strategy : ℝ → ℝ → ℝ → Prop), ∀ (alien_path : ℝ → ℝ → Prop), strategy u v alien_path :=
sorry

end spaceship_finds_alien_l46_46824


namespace lcm_48_180_l46_46034

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l46_46034


namespace option_A_correct_l46_46906

variables (a b: Line) (α β: Plane)

-- Definitions of parallel and perpendicular lines and planes
def is_parallel (x y : Line) : Prop :=
  sorry  -- Definition here

def intersect (p q : Plane) : Line :=
  sorry  -- Intersection definition here

def is_parallel_plane (p q : Plane) : Prop :=
  sorry  -- Definition here

def is_perpendicular (x y: Line) : Prop :=
  sorry  -- Definition here

def is_perpendicular_plane (p q : Plane) : Prop :=
  sorry  -- Definition here

theorem option_A_correct (h1: is_parallel a α) (h2: is_parallel a β) (h3: intersect α β = b) : is_parallel a b :=
by
  sorry

end option_A_correct_l46_46906


namespace tan_A_in_right_triangle_l46_46194

theorem tan_A_in_right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (angle_A angle_B angle_C : ℝ) 
  (sin_B : ℚ) (tan_A : ℚ) :
  angle_C = 90 ∧ sin_B = 3 / 5 → tan_A = 4 / 3 := by
  sorry

end tan_A_in_right_triangle_l46_46194


namespace rhombus_area_112_5_l46_46703

def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_112_5 :
  ∀ (d1 d2 : ℝ), d1 = 15 → d2 = 15 → area_of_rhombus d1 d2 = 112.5 :=
by intros d1 d2 h1 h2; simp [area_of_rhombus, h1, h2]; norm_num; sorry

end rhombus_area_112_5_l46_46703


namespace checkerboard_sum_l46_46407

theorem checkerboard_sum :
  let f (i j : ℕ) := 15 * (i - 1) + j
  let g (i j : ℕ) := 11 * (j - 1) + i
  -- List valid pairs (i, j) based on the problem constraints.
  let valid_pairs :=
    [(1, 2), (5, 9)] 
  -- Calculate the sum of matching numbers.
  \sum (i, j) in valid_pairs, f i j = 117 :=
begin
  sorry
end

end checkerboard_sum_l46_46407


namespace ellen_golf_cart_trips_l46_46866

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l46_46866


namespace evaluateExpression_at_1_l46_46379

noncomputable def evaluateExpression (x : ℝ) : ℝ :=
  (x^2 - 3 * x - 10) / (x - 5)

theorem evaluateExpression_at_1 : evaluateExpression 1 = 3 :=
by
  sorry

end evaluateExpression_at_1_l46_46379


namespace unique_ones_digits_divisible_by_8_l46_46244

/-- Carla likes numbers that are divisible by 8.
    We want to show that there are 5 unique ones digits for such numbers. -/
theorem unique_ones_digits_divisible_by_8 : 
  (Finset.card 
    (Finset.image (fun n => n % 10) 
                  (Finset.filter (fun n => n % 8 = 0) (Finset.range 100)))) = 5 := 
by
  sorry

end unique_ones_digits_divisible_by_8_l46_46244


namespace cos_alpha_value_l46_46120

theorem cos_alpha_value :
  let α := angle(O, (-3/5, 4/5)),
      origin := (0, 0),
      P := (-3/5, 4/5) in
    dist origin P = 1 →
    cos α = -3/5 :=
by
  intro α origin P h
  sorry

end cos_alpha_value_l46_46120


namespace midpoint_chord_intersection_eq_l46_46438

theorem midpoint_chord_intersection_eq (circle : Set Point) (M N A B C D E F P : Point) 
  (chord_MN : M ∈ circle ∧ N ∈ circle) 
  (midpoint_P : segment M N / 2 = P) 
  (chord_AB : A ∈ circle ∧ B ∈ circle)
  (chord_CD : C ∈ circle ∧ D ∈ circle)
  (E_intersection : on_line E (segment M N) ∧ intersect (line B C) (line M N) = Some E)
  (F_intersection : on_line F (segment M N) ∧ intersect (line A D) (line M N) = Some F) :
  dist P E = dist P F := 
sorry

end midpoint_chord_intersection_eq_l46_46438


namespace relationship_between_a_and_b_l46_46110

-- Define the objects and their relationships
noncomputable def α_parallel_β : Prop := sorry
noncomputable def a_parallel_α : Prop := sorry
noncomputable def b_perpendicular_β : Prop := sorry

-- Define the relationship we want to prove
noncomputable def a_perpendicular_b : Prop := sorry

-- The statement we want to prove
theorem relationship_between_a_and_b (h1 : α_parallel_β) (h2 : a_parallel_α) (h3 : b_perpendicular_β) : a_perpendicular_b :=
sorry

end relationship_between_a_and_b_l46_46110


namespace two_digit_squares_product_of_digits_is_square_l46_46990

def is_square (n : ℕ) : Prop :=
  ∃ m, m * m = n

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def two_digit_squares_with_square_digit_product_count : ℕ :=
  let squares := [16, 25, 36, 49, 64, 81]
  (squares.filter (λ n => is_square (digit_product n))).length

theorem two_digit_squares_product_of_digits_is_square :
  two_digit_squares_with_square_digit_product_count = 1 :=
by
  simp [two_digit_squares_with_square_digit_product_count, digit_product, is_square]
  -- Removed proof steps for clarity
  -- The proof is omitted
  sorry

end two_digit_squares_product_of_digits_is_square_l46_46990


namespace correct_diff_operation_l46_46730

/--
  Given the following differentiation operations:
  1. (x + 1/x)' = 1 + 1/x^2
  2. (log₂(x))' = 1 / (x * ln 2)
  3. (3^x)' = 3^x * log₃(e)
  4. (x^2 / e^x)' = (2x + x^2) / e^x

  Prove that:
  The differentiation of log₂(x) with respect to x gives the correct result.
-/
theorem correct_diff_operation: 
  ∀ (x : ℝ), 
  differentiable ℝ (fun x => Real.log x / Real.log 2) →
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
by
  -- proof goes here
  sorry

end correct_diff_operation_l46_46730


namespace count_numbers_seven_times_sum_of_digits_l46_46988

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc, d + acc) 0

def is_seven_times_sum_of_digits (n : ℕ) : Prop :=
  n = 7 * (sum_of_digits n)

theorem count_numbers_seven_times_sum_of_digits : (finset.range 1000).filter (λ n, is_seven_times_sum_of_digits n).card = 3 :=
by
  sorry

end count_numbers_seven_times_sum_of_digits_l46_46988


namespace mortdecai_donates_eggs_l46_46185

constant collects_eggs : Nat := 8 * 2  -- dozen eggs
constant delivers_to_market : Nat := 3  -- dozen eggs
constant delivers_to_mall : Nat := 5  -- dozen eggs
constant makes_pie : Nat := 4  -- dozen eggs
constant dozen_to_eggs : Nat := 12  -- eggs per dozen

-- Prove that Mortdecai donates 48 eggs to the charity
theorem mortdecai_donates_eggs : 
  (collects_eggs - (delivers_to_market + delivers_to_mall) - makes_pie) * dozen_to_eggs = 48 := by
  sorry

end mortdecai_donates_eggs_l46_46185


namespace solution_of_inequality_l46_46940

namespace math_proof

variables {ℝ : Type*} [linear_ordered_field ℝ] {f : ℝ → ℝ}

-- Conditions
def domain_f (f : ℝ → ℝ) := ∀ x : ℝ, true
def symmetric_about_point_1 (f : ℝ → ℝ) := ∀ x : ℝ, f(x-1) = 2 - f(3-x)  -- ensuring symmetry about (1,0)
def f_3_eq_0 (f : ℝ → ℝ) := f 3 = 0
def monotonic_lessthan_zero (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 → (f(x2) < f(x1) ↔ x2 < x1)

-- Inequality to solve
def inequality (f : ℝ → ℝ) (x : ℝ) := (x - 1) * f (x + 1) ≥ 0

-- Correct answer set
def correct_solution_set (x : ℝ) := (x ∈ Icc (-4:ℝ) (-1) ∨ x ∈ Icc (1:ℝ) (2))

-- The proof problem statement
theorem solution_of_inequality 
  (df : domain_f f) 
  (symmetric : symmetric_about_point_1 f) 
  (f3 : f_3_eq_0 f) 
  (monotonic : monotonic_lessthan_zero f) :
  ∀ x : ℝ, inequality f x ↔ correct_solution_set x := 
sorry

end math_proof

end solution_of_inequality_l46_46940


namespace limit_S_n_as_n_approaches_infinity_l46_46408

noncomputable def S_n (a b : ℝ) (n : ℕ) : ℝ := (finset.range n).sum (λ k, (π * (a^2) * (1 / 2) ^ k) / 4)

theorem limit_S_n_as_n_approaches_infinity (a b : ℝ) (h : a ≤ b) : 
  filter.tendsto (S_n a b) filter.at_top (nhds (π * a^2 / 2)) :=
begin
  sorry
end

end limit_S_n_as_n_approaches_infinity_l46_46408


namespace log_expression_l46_46842

section log_problem

variable (log : ℝ → ℝ)
variable (m n : ℝ)

-- Assume the properties of logarithms:
-- 1. log(m^n) = n * log(m)
axiom log_pow (m : ℝ) (n : ℝ) : log (m ^ n) = n * log m
-- 2. log(m * n) = log(m) + log(n)
axiom log_mul (m n : ℝ) : log (m * n) = log m + log n
-- 3. log(1) = 0
axiom log_one : log 1 = 0

theorem log_expression : log 5 * log 2 + log (2 ^ 2) - log 2 = 0 := by
  sorry

end log_problem

end log_expression_l46_46842


namespace find_pair_s_m_l46_46295

theorem find_pair_s_m :
  ∃ s m : ℝ, s = -10 ∧ m = -7 / 3 ∧
    (∀ (t : ℝ), ∃ x y : ℝ,
      (x, y) = ( -4 + t * m, s + t * (-7)) ∧ y = 3 * x + 2) :=
begin
  use [-10, -7 / 3],
  split, 
  { refl },
  split, 
  { refl },
  intros t,
  use [-4 + t * (-7 / 3), -10 + t * (-7)],
  split,
  { simp },
  {
    calc
    -10 + t * (-7) = 3 * (-4 + t * (-7 / 3)) + 2 : by sorry
  }
end


end find_pair_s_m_l46_46295


namespace rectangular_prism_faces_edges_vertices_sum_l46_46328

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46328


namespace cos_negative_half_l46_46149

theorem cos_negative_half :
  {x : ℝ | 0 ≤ x ∧ x < 360 ∧ Real.cos (x * Real.pi / 180) = -0.5}.card = 2 :=
sorry

end cos_negative_half_l46_46149


namespace percent_increase_of_semicircle_areas_l46_46794

def area_of_semicircle (radius : ℝ) : ℝ :=
  (1 / 2) * real.pi * (radius ^ 2)

theorem percent_increase_of_semicircle_areas :
  let r_large := 6
  let r_small := 4
  let large_area := 2 * area_of_semicircle r_large
  let small_area := 2 * area_of_semicircle r_small
  (large_area / small_area - 1) * 100 = 125 :=
by
  sorry

end percent_increase_of_semicircle_areas_l46_46794


namespace exists_nonneg_partial_sum_l46_46504

open Nat Classical

noncomputable theory

def periodic_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, a (n + k) = a k

def sum_to_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, a (k + 1))

theorem exists_nonneg_partial_sum
  (a : ℕ → ℝ)
  (n : ℕ)
  (h_sum : sum_to_n a n = 0)
  (h_periodic : periodic_seq a n) :
  ∃ N : ℕ, ∀ k : ℕ, (finset.range (k + 1)).sum (λ j, a (N + j)) ≥ 0 := sorry

end exists_nonneg_partial_sum_l46_46504


namespace rafael_total_net_pay_is_878_l46_46260

noncomputable def rafaelNetPay 
  (mondayHours tuesdayHours wednesdayHours thursdayHours fridayHours: ℕ)
  (totalHoursBonus taxDeduction taxCredit: ℝ)
  (wagePerHour overtimeWagePerHour: ℝ): ℝ :=
let weeklyHours := mondayHours + tuesdayHours + wednesdayHours + thursdayHours + fridayHours 
in let regularHoursMonday := min mondayHours 8 
in let overtimeHoursMonday := max (mondayHours - 8) 0 
in let regularPayMonday := regularHoursMonday * wagePerHour 
in let overtimePayMonday := overtimeHoursMonday * overtimeWagePerHour 
in let regularPayTWF := tuesdayHours * wagePerHour + wednesdayHours * wagePerHour + thursdayHours * wagePerHour + fridayHours * wagePerHour 
in let totalPayPreBonus := regularPayMonday + overtimePayMonday + regularPayTWF 
in let totalPayWithBonus := totalPayPreBonus + totalHoursBonus 
in let taxOwed := totalPayWithBonus * taxDeduction 
in let taxAfterCredit := max (taxOwed - taxCredit) 0 
in totalPayWithBonus - taxAfterCredit

theorem rafael_total_net_pay_is_878:
rafaelNetPay 10 8 8 8 6 100 0.1 50 20 30 = 878 := 
sorry

end rafael_total_net_pay_is_878_l46_46260


namespace solve_for_x_l46_46380

theorem solve_for_x : ∃ x : ℝ, 3 * x + 6 = |(-5) * 4 + 2| ∧ x = 4 :=
by
  have h : |(-5) * 4 + 2| = 18 := by sorry
  use 4
  rw [h]
  norm_num
  sorry

end solve_for_x_l46_46380


namespace original_number_is_14_l46_46623

theorem original_number_is_14 (x : ℝ) (h : (2 * x + 2) / 3 = 10) : x = 14 := by
  sorry

end original_number_is_14_l46_46623


namespace min_value_fraction_l46_46771

variable (a b : ℝ)
variable (h1 : 2 * a - 2 * b + 2 = 0) -- This corresponds to a + b = 1 based on the given center (-1, 2)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_fraction (h1 : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (4 / a) + (1 / b) ≥ 9 :=
  sorry

end min_value_fraction_l46_46771


namespace complex_number_solution_l46_46564

theorem complex_number_solution (Z : ℂ) (h : Z = complex.i * (2 + Z)) : Z = -1 + complex.i := by
  sorry

end complex_number_solution_l46_46564


namespace hexagon_area_l46_46549

namespace Geometry

def equilateral_triangle_area (s : ℚ) : ℚ :=
  (sqrt 3 / 4) * s^2

def rotated_triangle_overlap_area (s : ℚ) : ℚ :=
  1 / 2 * equilateral_triangle_area s

theorem hexagon_area (s : ℚ) (h_eq : s = 2) : 
  equilateral_triangle_area s + equilateral_triangle_area s - rotated_triangle_overlap_area s = sqrt 3 :=
by
  -- Proof steps would go here
  sorry

end Geometry

end hexagon_area_l46_46549


namespace find_abc_l46_46154

theorem find_abc :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ (100 * a + 10 * b + c = a.factorial + b.factorial + c.factorial) :=
by {
  use [1, 4, 5],
  split, exact ne_of_lt (by norm_num),
  split, exact ne_of_lt (by norm_num),
  split, exact ne_of_lt (by norm_num),
  split, exact by norm_num,
  split, exact by norm_num,
  split, exact by norm_num,
  norm_num,
  exact dec_trivial,
}

end find_abc_l46_46154


namespace log_base_4_evaluation_l46_46870

theorem log_base_4_evaluation :
  let x := (81 : ℝ) * (16 : ℝ)^(1/3) * (16 : ℝ)^(1/6)
  in log 4 x = 4 + 17 / 100 :=
by
  sorry

end log_base_4_evaluation_l46_46870


namespace part1_monotonicity_part2_inequality_l46_46967

variable (a : ℝ) (x : ℝ)
variable (h : a > 0)

def f := a * (Real.exp x + a) - x

theorem part1_monotonicity :
  if a ≤ 0 then ∀ x : ℝ, (f a x) < (f a (x + 1)) else
  if a > 0 then ∀ x : ℝ, (x < Real.log (1 / a) → (f a x) > (f a (x + 1))) ∧ 
                       (x > Real.log (1 / a) → (f a x) < (f a (x - 1))) else 
  (False) := sorry

theorem part2_inequality :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3 / 2 := sorry

end part1_monotonicity_part2_inequality_l46_46967


namespace limit_ln_eq_half_l46_46956

theorem limit_ln_eq_half :
  (∀ (f : ℝ → ℝ), (∀ x, f x = Real.log x) → (Real.log_deriv : (ℝ → ℝ)) = (1 / 2) := by
  sorry

end limit_ln_eq_half_l46_46956


namespace zero_point_order_l46_46542

noncomputable def f (x : ℝ) : ℝ := 2^x + x
noncomputable def g (x : ℝ) : ℝ := x - 1
noncomputable def h (x : ℝ) : ℝ := log 3 x + x

def zero_point_f (a : ℝ) : Prop := f a = 0
def zero_point_g (b : ℝ) : Prop := g b = 0
def zero_point_h (c : ℝ) : Prop := h c = 0

theorem zero_point_order (a b c : ℝ) 
  (h1 : zero_point_f a) 
  (h2 : zero_point_g b) 
  (h3 : zero_point_h c)
  (ha : a < 0)
  (hb : b = 1) 
  (hc : 1/3 < c ∧ c < 1) : 
  a < c ∧ c < b := 
sorry

end zero_point_order_l46_46542


namespace square_center_trajectory_l46_46419

theorem square_center_trajectory (a : ℝ) (h_a : 0 < a) :
  ∃ C : ℝ × ℝ, ∀ (x y : ℝ), 
    (C = (x, y)) ∧ (a ≤ x ∧ x ≤ sqrt (2) * a) ∧ (y = x) ∧
    (∃ A B : ℝ × ℝ, 
       A = (x, 0) ∧ B = (0, y) ∧
       dist A B = 2 * a ∧
       A.fst = x ∧ A.snd = 0 ∧
       B.fst = 0 ∧ B.snd = y) :=
begin
  sorry
end

end square_center_trajectory_l46_46419


namespace y_odd_and_period_pi_over_two_l46_46669

noncomputable def y (x : ℝ) : ℝ := sin (2 * x) * cos (2 * x)

theorem y_odd_and_period_pi_over_two : 
  (∀ x, y (-x) = - y x) ∧ (∀ x, y (x + π/2) = y x) := 
by 
  sorry

end y_odd_and_period_pi_over_two_l46_46669


namespace find_integer_n_satisfying_conditions_l46_46873

open BigOperators

def sigma (n : ℕ) : ℕ :=
  ∑ i in (Finset.range (n + 1)).filter (λ d => n % d = 0), i

def p (n : ℕ) : ℕ :=
  (Nat.factors n).erase_dup.maximum'

theorem find_integer_n_satisfying_conditions (n : ℕ) (h_n_ge_2 : n ≥ 2)
  (h_condition : σ n / (p n - 1) = n) : n = 6 :=
  sorry

end find_integer_n_satisfying_conditions_l46_46873


namespace problem_statement_l46_46101

theorem problem_statement (x m : ℝ) :
  (¬ (x > m) → ¬ (x^2 + x - 2 > 0)) ∧ (¬ (x > m) ↔ ¬ (x^2 + x - 2 > 0)) → m ≥ 1 :=
sorry

end problem_statement_l46_46101


namespace tan_A_right_triangle_l46_46196

theorem tan_A_right_triangle (A B C : Type) [RealField A] [RealField B] [RealField C]
  (h1 : ∠ABC = 90) (h2 : sin B = 3 / 5) : tan A = 4 / 3 :=
sorry

end tan_A_right_triangle_l46_46196


namespace domain_of_f_l46_46663

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (x - 2)

theorem domain_of_f : {x : ℝ | x > -1 ∧ x ≠ 2} = {x : ℝ | x ∈ Set.Ioo (-1) 2 ∪ Set.Ioi 2} :=
by {
  sorry
}

end domain_of_f_l46_46663


namespace probability_f_a_gt_0_l46_46132

def f (x : ℝ) : ℝ := 2 * sin (2 * x - (π / 3)) - 1

theorem probability_f_a_gt_0 : 
  (1 / 2 : ℝ) = (measure_theory.volume (set.Ioo (π / 6) (5 * π / 12)) / measure_theory.volume (set.Icc 0 (π / 2))) :=
by 
  let interval := set.Icc (0 : ℝ) (π / 2)
  have measurable_set_interval : measure_theory.measurable_space.measure_Set interval := sorry
  have f_is_measurable : measure_theory.measurable f := sorry
  have probability_range : set.Ioo (π / 6) (5 * π / 12) = {x | f x > 0} ∩ interval := sorry
  have length_set1 : measure_theory.volume (set.Ioo (π / 6) (5 * π / 12)) = (π / 4) := sorry
  have length_set2 : measure_theory.volume (set.Icc (0) (π / 2)) = (π / 2) := sorry
  have prob_eq : (π / 4) / (π / 2) = (1 / 2 : ℝ) := sorry
  exact prob_eq

end probability_f_a_gt_0_l46_46132


namespace power_equivalence_l46_46377

theorem power_equivalence (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 :=
by sorry

end power_equivalence_l46_46377


namespace cosine_of_45_degrees_l46_46449

theorem cosine_of_45_degrees : Real.cos (π / 4) = √2 / 2 := by
  sorry

end cosine_of_45_degrees_l46_46449


namespace truck_driver_needs_more_gallons_l46_46427

-- Define the conditions
def miles_per_gallon : ℕ := 3
def total_distance : ℕ := 90
def current_gallons : ℕ := 12
def can_cover_distance : ℕ := miles_per_gallon * current_gallons
def additional_distance_needed : ℕ := total_distance - can_cover_distance

-- Define the main theorem
theorem truck_driver_needs_more_gallons :
  additional_distance_needed / miles_per_gallon = 18 :=
by
  -- Placeholder for the proof
  sorry

end truck_driver_needs_more_gallons_l46_46427


namespace right_triangle_perimeter_l46_46976

theorem right_triangle_perimeter
  (a b : ℝ) (c : ℝ)
  (ha : a = 3) (hb : b = 4)
  (hc1 : c = real.sqrt (a^2 + b^2))
  (hc2 : c = real.sqrt (b^2 - a^2) ∨ c = real.sqrt (a^2 + b^2)) :
  (a + b + c = 12) ∨ (a + b + c = 7 + real.sqrt 7) :=
by sorry

end right_triangle_perimeter_l46_46976


namespace train_speed_is_accurate_l46_46816

-- Define the given conditions
def train_length_km : ℝ := 140 / 1000 -- converting 140 meters to kilometers
def time_to_pass_pole : ℝ := 5.142857142857143 -- time in seconds

-- Define the desired speed in km/h
def expected_speed_kmh : ℝ := 97.96

-- The theorem we want to prove
theorem train_speed_is_accurate :
  (train_length_km / time_to_pass_pole) * 3600 = expected_speed_kmh :=
by
  sorry

end train_speed_is_accurate_l46_46816


namespace sqrt_one_plus_cos_l46_46105

theorem sqrt_one_plus_cos (α : ℝ) (h : 180 < α ∧ α < 360) :
  sqrt (1 + Real.cos α) = -sqrt 2 * Real.cos (α / 2) :=
sorry

end sqrt_one_plus_cos_l46_46105


namespace least_positive_integer_exists_l46_46716

theorem least_positive_integer_exists :
  ∃ (x : ℕ), 
    (x % 6 = 5) ∧
    (x % 8 = 7) ∧
    (x % 7 = 6) ∧
    x = 167 :=
by {
  sorry
}

end least_positive_integer_exists_l46_46716


namespace log_sum_of_squares_l46_46239

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

variables {a : ℝ} (ha1 : a > 0) (ha2 : a ≠ 1)
variables {x : ℝ} (hx : x > 0)
variables {x1 x2 ... x2003 : ℝ} (hx1 : x1 > 0) (hx2 : x2 > 0) ... (hx2003 : x2003 > 0)
variables (h : f a (x1 * x2 * ... * x2003) = 8)

theorem log_sum_of_squares :
  f a (x1^2) + f a (x2^2) + ... + f a (x2003^2) = 16 := 
sorry

end log_sum_of_squares_l46_46239


namespace slices_left_for_tomorrow_is_four_l46_46013

def initial_slices : ℕ := 12
def lunch_slices : ℕ := initial_slices / 2
def remaining_slices_after_lunch : ℕ := initial_slices - lunch_slices
def dinner_slices : ℕ := remaining_slices_after_lunch / 3
def slices_left_for_tomorrow : ℕ := remaining_slices_after_lunch - dinner_slices

theorem slices_left_for_tomorrow_is_four : slices_left_for_tomorrow = 4 := by
  sorry

end slices_left_for_tomorrow_is_four_l46_46013


namespace trapezoid_bc_length_l46_46083

theorem trapezoid_bc_length
  (A B C D M : Point)
  (d : Real)
  (h_trapezoid : IsTrapezoid A B C D)
  (h_M_on_AB : OnLine M A B)
  (h_DM_perp_AB : Perpendicular D M A B)
  (h_MC_eq_CD : Distance M C = Distance C D)
  (h_AD_eq_d : Distance A D = d) :
  Distance B C = d / 2 := by
  sorry

end trapezoid_bc_length_l46_46083


namespace total_opponent_score_l46_46428

-- Definitions based on the conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def lost_by_one_point (scores : List ℕ) : Bool :=
  scores = [3, 4, 5]

def scored_twice_as_many (scores : List ℕ) : Bool :=
  scores = [2, 3]

def scored_three_times_as_many (scores : List ℕ) : Bool :=
  scores = [2, 3, 3]

-- Proof problem:
theorem total_opponent_score :
  ∀ (lost_scores twice_scores thrice_scores : List ℕ),
    lost_by_one_point lost_scores →
    scored_twice_as_many twice_scores →
    scored_three_times_as_many thrice_scores →
    (lost_scores.sum + twice_scores.sum + thrice_scores.sum) = 25 :=
by
  intros
  sorry

end total_opponent_score_l46_46428


namespace tangent_line_at_1_l46_46752

def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

def f_derivative (x : ℝ) : ℝ := 2 * x * (x - 2) + x^2

theorem tangent_line_at_1 : 
  let p : ℝ × ℝ := (1, 0) in 
  let slope : ℝ := -1 in 
  ∀ x y : ℝ, 
  y = slope * (x - p.1) + p.2 ↔ x + y - 1 = 0 :=
by 
  sorry

end tangent_line_at_1_l46_46752


namespace number_of_arrangements_l46_46689

def placement (board : Array (Array Char)) (i j : Nat) (c : Char) : Prop :=
  board[i]![j] = c

def row_col_constraints (board : Array (Array Char)) : Prop :=
  ∀ i, (∃ a : Nat, placement board i a 'A') ∧ (∃ b : Nat, placement board i b 'B') ∧ (∃ c : Nat, placement board i c 'C') ∧
       ∀ j, (∃ a : Nat, placement board a j 'A') ∧ (∃ b : Nat, placement board a j 'B') ∧ (∃ c : Nat, placement board a j 'C')

theorem number_of_arrangements :
  ∃! (board : Array (Array Char)), 
    row_col_constraints board ∧
    placement board 0 0 'A' ∧ 
    placement board 1 0 'B' :=
by 
  sorry

end number_of_arrangements_l46_46689


namespace rectangular_prism_faces_edges_vertices_sum_l46_46329

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l46_46329


namespace smallest_positive_period_l46_46526

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period : ∃ T > 0, (∀ x, f(x + T) = f(x)) ∧ (∀ T' > 0, (∀ x, f(x + T') = f(x)) → T ≤ T') ∧ T = Real.pi :=
by
  sorry

end smallest_positive_period_l46_46526


namespace decreasing_range_of_a_l46_46114

noncomputable def f (a x : ℝ) : ℝ := (Real.sqrt (2 - a * x)) / (a - 1)

theorem decreasing_range_of_a (a : ℝ) :
    (∀ x y : ℝ, 0 ≤ x → x ≤ 1/2 → 0 ≤ y → y ≤ 1/2 → x < y → f a y < f a x) ↔ (a < 0 ∨ (1 < a ∧ a ≤ 4)) :=
by
  sorry

end decreasing_range_of_a_l46_46114


namespace parity_of_b_and_c_l46_46894

theorem parity_of_b_and_c (a b c : ℕ) (h_prime : Nat.Prime a) (h_eq : a^2 + b^2 = c^2) : 
  (Nat.odd b ∧ Nat.even c) ∨ (Nat.even b ∧ Nat.odd c) :=
sorry

end parity_of_b_and_c_l46_46894


namespace bus_total_distance_l46_46756

theorem bus_total_distance
  (distance40 : ℝ)
  (distance60 : ℝ)
  (speed40 : ℝ)
  (speed60 : ℝ)
  (total_time : ℝ)
  (distance40_eq : distance40 = 100)
  (speed40_eq : speed40 = 40)
  (speed60_eq : speed60 = 60)
  (total_time_eq : total_time = 5)
  (time40 : ℝ)
  (time40_eq : time40 = distance40 / speed40)
  (time_equation : time40 + distance60 / speed60 = total_time) :
  distance40 + distance60 = 250 := sorry

end bus_total_distance_l46_46756


namespace lcm_48_180_value_l46_46028

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l46_46028


namespace length_CK_angle_BCA_l46_46208

variables {A B C O O₁ O₂ K K₁ K₂ K₃ : Point}
variables {r R : ℝ}
variables {AC CK AK₁ AK₂ : ℝ}

-- Definitions and conditions
def triangle_ABC (A B C : Point) : Prop := True
def incenter (A B C O : Point) : Prop := True
def in_radius_is_equal (O₁ O₂ : Point) (r : ℝ) : Prop := True
def circle_touches_side (circle_center : Point) (side_point : Point) (distance : ℝ) : Prop := True
def circumcenter (A C B O₁ : Point) : Prop := True
def angle (A B C : Point) (θ : ℝ) : Prop := True

-- Conditions from the problem
axiom cond1 : triangle_ABC A B C
axiom cond2 : in_radius_is_equal O₁ O₂ r
axiom cond3 : incenter A B C O
axiom cond4 : circle_touches_side O₁ K₁ 6
axiom cond5 : circle_touches_side O₂ K₂ 8
axiom cond6 : AC = 21
axiom cond7 : circle_touches_side O K 9
axiom cond8 : circumcenter O K₁ K₃ O₁

-- Statements to prove
theorem length_CK : CK = 9 := by
  sorry

theorem angle_BCA : angle B C A 60 := by
  sorry

end length_CK_angle_BCA_l46_46208


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46364

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46364


namespace expression_equals_base10_l46_46469

-- Define numbers in various bases
def base7ToDec (n : ℕ) : ℕ := 1 * (7^2) + 6 * (7^1) + 5 * (7^0)
def base2ToDec (n : ℕ) : ℕ := 1 * (2^1) + 1 * (2^0)
def base6ToDec (n : ℕ) : ℕ := 1 * (6^2) + 2 * (6^1) + 1 * (6^0)
def base3ToDec (n : ℕ) : ℕ := 2 * (3^1) + 1 * (3^0)

-- Prove the given expression equals 39 in base 10
theorem expression_equals_base10 :
  (base7ToDec 165 / base2ToDec 11) + (base6ToDec 121 / base3ToDec 21) = 39 :=
by
  -- Convert the base n numbers to base 10
  let num1 := base7ToDec 165
  let den1 := base2ToDec 11
  let num2 := base6ToDec 121
  let den2 := base3ToDec 21
  
  -- Simplify the expression (skipping actual steps for brevity, replaced by sorry)
  sorry

end expression_equals_base10_l46_46469


namespace largest_four_digit_sum_20_l46_46710

-- Defining the four-digit number and conditions.
def is_four_digit_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  ∃ a b c d : ℕ, a + b + c + d = s ∧ n = 1000 * a + 100 * b + 10 * c + d

-- Proof problem statement.
theorem largest_four_digit_sum_20 : ∃ n, is_four_digit_number n ∧ digits_sum_to n 20 ∧ ∀ m, is_four_digit_number m ∧ digits_sum_to m 20 → m ≤ n :=
  sorry

end largest_four_digit_sum_20_l46_46710


namespace value_of_x_in_terms_of_z_l46_46613

variable {z : ℝ} {x y : ℝ}
  
theorem value_of_x_in_terms_of_z (h1 : y = z + 50) (h2 : x = 0.70 * y) : x = 0.70 * z + 35 := 
  sorry

end value_of_x_in_terms_of_z_l46_46613


namespace calculate_c_from_law_of_cosines_l46_46209

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

theorem calculate_c_from_law_of_cosines 
  (a b c : ℝ) (B : ℝ)
  (ha : a = 8) (hb : b = 7) (hB : B = Real.pi / 3) : 
  (c = 3) ∨ (c = 5) :=
sorry

end calculate_c_from_law_of_cosines_l46_46209


namespace total_managers_l46_46837

def total_employees : ℕ := 250
def female_employees : ℕ := 90
def male_associates : ℕ := 160
def female_managers : ℕ := 40

theorem total_managers (E : ℕ) (F : ℕ) (MA : ℕ) (FM : ℕ) : E = 250 ∧ F = 90 ∧ MA = 160 ∧ FM = 40 → FM = 40 :=
by
  intros h
  cases h with hE hFMA
  cases hFMA with hFMA_hF hFM
  cases hFMA_hF with hF hMA
  exact hFM

end total_managers_l46_46837


namespace solve_for_s_l46_46851

def E (a b c : ℕ) : ℕ := a * b^c

theorem solve_for_s : ∃ s : ℕ, E(s, s, 4) = 1024 ∧ s > 0 ∧ s = 4 :=
by
  sorry

end solve_for_s_l46_46851


namespace solve_linear_equation_one_variable_with_parentheses_l46_46155

/--
Theorem: Solving a linear equation in one variable that contains parentheses
is equivalent to the process of:
1. Removing the parentheses,
2. Moving terms,
3. Combining like terms, and
4. Making the coefficient of the unknown equal to 1.

Given: a linear equation in one variable that contains parentheses
Prove: The process of solving it is to remove the parentheses, move terms, combine like terms, and make the coefficient of the unknown equal to 1.
-/
theorem solve_linear_equation_one_variable_with_parentheses
  (eq : String) :
  ∃ instructions : String,
    instructions = "remove the parentheses; move terms; combine like terms; make the coefficient of the unknown equal to 1" :=
by
  sorry

end solve_linear_equation_one_variable_with_parentheses_l46_46155


namespace floor_expression_equality_l46_46844

theorem floor_expression_equality :
  ⌊((2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023))⌋ = 8 := 
sorry

end floor_expression_equality_l46_46844


namespace largest_four_digit_sum_20_l46_46712

-- Defining the four-digit number and conditions.
def is_four_digit_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  ∃ a b c d : ℕ, a + b + c + d = s ∧ n = 1000 * a + 100 * b + 10 * c + d

-- Proof problem statement.
theorem largest_four_digit_sum_20 : ∃ n, is_four_digit_number n ∧ digits_sum_to n 20 ∧ ∀ m, is_four_digit_number m ∧ digits_sum_to m 20 → m ≤ n :=
  sorry

end largest_four_digit_sum_20_l46_46712


namespace hcf_lcm_fraction_l46_46275

theorem hcf_lcm_fraction (m n : ℕ) (HCF : Nat.gcd m n = 6) (LCM : Nat.lcm m n = 210) (sum_mn : m + n = 72) : 
  (1 / m : ℚ) + (1 / n : ℚ) = 2 / 35 :=
by
  sorry

end hcf_lcm_fraction_l46_46275


namespace ratio_c_over_a_l46_46099

theorem ratio_c_over_a (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : log (a * b) + log (c * b) = 2 * log (a * c))
  (h5 : 4 * (a + c) = 17 * b) :
  (c / a = 16) ∨ (c / a = 1 / 16) := 
sorry

end ratio_c_over_a_l46_46099


namespace sum_of_faces_edges_vertices_l46_46373

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46373


namespace cylinder_line_intersection_l46_46889

noncomputable def intersection_points (R x₀ y₀ a b : ℝ) : Set ℝ :=
  {t | let delta := (x₀ * a + y₀ * b) ^ 2 - (a ^ 2 + b ^ 2) * (x₀ ^ 2 + y₀ ^ 2 - R ^ 2) in
       let numerator₁ := - (x₀ * a + y₀ * b) + Real.sqrt delta in
       let numerator₂ := - (x₀ * a + y₀ * b) - Real.sqrt delta in
       let denominator := a ^ 2 + b ^ 2 in
       t = numerator₁ / denominator ∨ t = numerator₂ / denominator }

theorem cylinder_line_intersection
  (R x₀ y₀ z₀ a b c : ℝ) :
  ∀ t ∈ intersection_points R x₀ y₀ a b,
  (let Lx := x₀ + a * t in
   let Ly := y₀ + b * t in
   let Lz := z₀ + c * t in
   Lx ^ 2 + Ly ^ 2 = R ^ 2) :=
by intros t ht
   obtain ⟨delta, numerator₁, numerator₂, denominator, ht₁, ht₂⟩ := ht
   sorry

end cylinder_line_intersection_l46_46889


namespace tangent_condition_l46_46745

theorem tangent_condition (a b : ℝ) :
  (4 * a^2 + b^2 = 1) ↔ 
  ∀ x y : ℝ, (y = 2 * x + 1) → ((x^2 / a^2) + (y^2 / b^2) = 1) → (∃! y, y = 2 * x + 1 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

end tangent_condition_l46_46745


namespace find_log3_iterative_limit_approx_l46_46010

noncomputable def log3_iterative_limit (x : ℝ) : Prop :=
  x = Real.logb 3 (50 + x)

theorem find_log3_iterative_limit_approx :
  ∃ x : ℝ, x > 0 ∧ log3_iterative_limit x ∧ x ≈ 3.9 :=
by
  sorry

end find_log3_iterative_limit_approx_l46_46010


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46363

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46363


namespace remainder_of_sum_l46_46237

theorem remainder_of_sum (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 :=
by
  sorry

end remainder_of_sum_l46_46237


namespace length_upper_base_eq_half_d_l46_46076

variables {A B C D M: Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables {d : ℝ}

def trapezoid (A B C D : Type*) : Prop :=
  ∃ p : B, ∃ q : C, ∃ r : D, A ≠ p ∧ p ≠ q ∧ q ≠ r ∧ r ≠ A

def midpoint (A D : Type*) (N : Type*) (d : ℝ) : Prop :=
  dist A N = d / 2 ∧ dist N D = d / 2

axiom dm_perp_ab : ∀ (M : Type*), dist D M ∧ D ≠ M → dist M (id D) ≠ 0

axiom mc_eq_cd : dist M C = dist C D

theorem length_upper_base_eq_half_d
  (A B C D M : Type*)
  (h1 : trapezoid A B C D)
  (h2 : dist A D = d)
  (h3 : dm_perp_ab M)
  (h4 : mc_eq_cd) :
  dist B C = d / 2 :=
sorry

end length_upper_base_eq_half_d_l46_46076


namespace semicircle_area_percentage_difference_l46_46788

-- Define the rectangle dimensions
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 8

-- Define the diameters and radii of the semicircles
def large_semicircle_radius : ℝ := rectangle_length / 2
def small_semicircle_radius : ℝ := rectangle_width / 2

-- Define the areas of the full circles made from the semicircles
def large_circle_area : ℝ := real.pi * (large_semicircle_radius ^ 2)
def small_circle_area : ℝ := real.pi * (small_semicircle_radius ^ 2)

-- Define the percentage larger question
def percent_larger (a b : ℝ) : ℝ := ((a - b) / b) * 100

-- Formal proof statement
theorem semicircle_area_percentage_difference : 
  percent_larger large_circle_area small_circle_area = 125 := 
by
  sorry

end semicircle_area_percentage_difference_l46_46788


namespace arithmetic_seq_sum_l46_46922

theorem arithmetic_seq_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a_n 1 = -2010)
  (h2 : (S 2009 / 2009) - (S 2007 / 2007) = 2) :
  S 2011 = 0 := sorry

end arithmetic_seq_sum_l46_46922


namespace remaining_shaded_area_l46_46832
-- Import the entire Mathlib library to ensure all necessary definitions are available

-- Define the conditions as constants
def length_large_rectangle : ℝ := 19
def width_large_rectangle : ℝ := 11
def number_of_squares : ℕ := 4

-- State the problem to prove
theorem remaining_shaded_area :
  let total_area := length_large_rectangle * width_large_rectangle
  in ∃ (area_shaded : ℝ), 
     (area_shaded = 6 ∧ number_of_squares = 4 ∧ 
      length_large_rectangle = 19 ∧ width_large_rectangle = 11) := by
    -- Skip the proof with sorry
    sorry

end remaining_shaded_area_l46_46832


namespace ax5_by5_eq_6200_div_29_l46_46153

variables (a b x y : ℝ)

-- Given conditions
axiom h1 : a * x + b * y = 5
axiom h2 : a * x^2 + b * y^2 = 11
axiom h3 : a * x^3 + b * y^3 = 30
axiom h4 : a * x^4 + b * y^4 = 80

-- Statement to prove
theorem ax5_by5_eq_6200_div_29 : a * x^5 + b * y^5 = 6200 / 29 :=
by
  sorry

end ax5_by5_eq_6200_div_29_l46_46153


namespace prob_X_greater_than_5_l46_46116

noncomputable def X := Normal 3 1

axiom P_1_le_X_le_5 : P (λ x : ℝ, 1 ≤ x ∧ x ≤ 5) = 0.6826

theorem prob_X_greater_than_5 : P (λ x : ℝ, x > 5) = 0.1587 :=
by
  -- Proof goes here
  sorry

end prob_X_greater_than_5_l46_46116


namespace find_k_l46_46442

-- Define the matrix M
def M (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 3], ![0, 4, -k], ![3, -1, 2]]

-- Define the problem statement
theorem find_k (k : ℝ) (h : Matrix.det (M k) = -20) : k = 0 := by
  sorry

end find_k_l46_46442


namespace sum_first_100_terms_l46_46676

theorem sum_first_100_terms : 
  let seq : ℕ → ℤ := λ n, 
    let k := (n + 1) / 2 in 
    if even n then -k else k in
  (∑ i in Finset.range 100, seq i) = -7 :=
by sorry

end sum_first_100_terms_l46_46676


namespace length_upper_base_eq_half_d_l46_46077

variables {A B C D M: Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables {d : ℝ}

def trapezoid (A B C D : Type*) : Prop :=
  ∃ p : B, ∃ q : C, ∃ r : D, A ≠ p ∧ p ≠ q ∧ q ≠ r ∧ r ≠ A

def midpoint (A D : Type*) (N : Type*) (d : ℝ) : Prop :=
  dist A N = d / 2 ∧ dist N D = d / 2

axiom dm_perp_ab : ∀ (M : Type*), dist D M ∧ D ≠ M → dist M (id D) ≠ 0

axiom mc_eq_cd : dist M C = dist C D

theorem length_upper_base_eq_half_d
  (A B C D M : Type*)
  (h1 : trapezoid A B C D)
  (h2 : dist A D = d)
  (h3 : dm_perp_ab M)
  (h4 : mc_eq_cd) :
  dist B C = d / 2 :=
sorry

end length_upper_base_eq_half_d_l46_46077


namespace simplify_and_evaluate_expression_l46_46642

   variable (x : ℝ)

   theorem simplify_and_evaluate_expression (h : x = 2 * Real.sqrt 5 - 1) :
     (1 / (x ^ 2 + 2 * x + 1) * (1 + 3 / (x - 1)) / ((x + 2) / (x ^ 2 - 1))) = Real.sqrt 5 / 10 :=
   sorry
   
end simplify_and_evaluate_expression_l46_46642


namespace equal_sum_division_possible_l46_46899

theorem equal_sum_division_possible (n : ℕ) : 
  (∃ G1 G2 G3 : Finset ℕ, (G1 ∪ G2 ∪ G3 = Finset.range (n + 1)) ∧ (∀ x ∈ G1, x <= n) ∧ (∀ x ∈ G2, x <= n) ∧ (∀ x ∈ G3, x <= n) ∧ 
  (G1.sum (λ i, i) = G2.sum (λ i, i)) ∧ (G2.sum (λ i, i) = G3.sum (λ i, i)) ∧ (G1 ∩ G2 = ∅) ∧ (G2 ∩ G3 = ∅) ∧ (G1 ∩ G3 = ∅)) ↔ (n % 3 = 0 ∨ n % 3 = 2) := 
by sorry

end equal_sum_division_possible_l46_46899


namespace length_MN_is_3_point_75_l46_46166

-- Given a triangle △ABC
variables {A B C M N : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space N]

-- and points in the space
variable {triangle : simplex A B C}

-- and midpoints and bisectors
variables (M_midpoint : midpoint A B M)
variables (CN_bisects : bisector C N A B)
variables (CN_perpendicular : perpendicular C N A M)
variables (AB_length : distance A B = 15)
variables (AC_length : distance A C = 22)

-- prove that the length of MN is 3.75
theorem length_MN_is_3_point_75 : distance M N = 3.75 :=
sorry

end length_MN_is_3_point_75_l46_46166


namespace average_daily_difference_l46_46681

theorem average_daily_difference
  (day1 : ℤ) (day2 : ℤ) (day3 : ℤ) (day4 : ℤ) (day5 : ℤ) (day6 : ℤ) (day7 : ℤ)
  (h1 : day1 = 15) (h2 : day2 = -5) (h3 : day3 = 25) (h4 : day4 = -15) 
  (h5 : day5 = 35) (h6 : day6 = 0) (h7 : day7 = 20) :
  (day1 + day2 + day3 + day4 + day5 + day6 + day7) / 7 = 10 := 
by
  rw [h1, h2, h3, h4, h5, h6, h7]
  norm_num
  sorry

end average_daily_difference_l46_46681


namespace insufficient_data_to_answer_l46_46400

def TotalMovieTheatres (X : Type) : Type := X → Prop
def TheatreHasLessOrEqualScreens (X : Type) := X → Prop
def TheatreSellsMoreThan300PopcornShowing (X : Type) := X → Prop
def TheatreSellsLessOrEqual300PopcornShowing (X : Type) := X → Prop
def TheatreHasMoreScreens (X : Type) := X → Prop
def TheatreSellsMoreThan100PopcornDay (X : Type) := X → Prop

variables 
  (X : Type)
  (total_theatres : TotalMovieTheatres X)
  (theatres_3_screens_or_less : TheatreHasLessOrEqualScreens X)
  (theatres_more_than_300 : TheatreSellsMoreThan300PopcornShowing X)
  (theatres_less_than_equal_300 : TheatreSellsLessOrEqual300PopcornShowing X)
  (theatres_4_or_more_screens : TheatreHasMoreScreens X)
  (theatres_more_than_100_day : TheatreSellsMoreThan100PopcornDay X)

-- Conditions given in the problem
axiom condition_1 : ∀ x, total_theatres x → theatres_3_screens_or_less x → 60% -- (This would ideally be written as a probability but vague)
axiom condition_2 : ∀ x, theatres_3_screens_or_less x → theatres_more_than_300 x → 20%
axiom condition_3 : ∀ x, total_theatres x → theatres_less_than_equal_300 x → 50%

-- The question translates into proving percentage query with given conditions
-- Note: This essentially represents the problem statement without providing a direct solution because of insufficient data.
theorem insufficient_data_to_answer :
  ¬(∃ p, p = 4 → theatres_4_or_more_screens p ∧ theatres_more_than_100_day p) :=
sorry

end insufficient_data_to_answer_l46_46400


namespace convex_ngon_diagonal_bound_l46_46255

def convex_ngon (n : ℕ) : Prop := n ≥ 3

def diagonal_shares_common_vertex {n : ℕ} (diagonals : set (ℕ × ℕ)) : Prop :=
  ∃ (v : ℕ), ∀ (d ∈ diagonals), v ∈ d

theorem convex_ngon_diagonal_bound (n : ℕ) (hn : convex_ngon n) (diagonals : set (ℕ × ℕ)) :
  finset.card diagonals > n → ¬ (diagonal_shares_common_vertex diagonals) :=
by
  sorry

end convex_ngon_diagonal_bound_l46_46255


namespace cosine_of_45_degrees_l46_46450

theorem cosine_of_45_degrees : Real.cos (π / 4) = √2 / 2 := by
  sorry

end cosine_of_45_degrees_l46_46450


namespace brand_tangyuan_purchase_l46_46276

theorem brand_tangyuan_purchase (x y : ℕ) 
  (h1 : x + y = 1000) 
  (h2 : x = 2 * y + 20) : 
  x = 670 ∧ y = 330 := 
sorry

end brand_tangyuan_purchase_l46_46276


namespace min_value_of_sum_l46_46493

theorem min_value_of_sum (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) : 
  (min (λ m n : ℝ, 2 / m + 1 / n) {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 0 < p.1 ∧ 0 < p.2 }) = 2 * sqrt 2 + 3 :=
by
  sorry

end min_value_of_sum_l46_46493


namespace quadratic_roots_property_l46_46068

/-- 
If x₁ and x₂ are the roots of the quadratic equation x² - 5x - 3 = 0,
then x₁² + x₂² = 31 and 1/x₁ - 1/x₂ = √37 / 3.
-/
theorem quadratic_roots_property (x₁ x₂ : ℝ) (h_roots : ∀ x, x ^ 2 - 5 * x - 3 = (x - x₁) * (x - x₂)) :
  x₁^2 + x₂^2 = 31 ∧ (1 / x₁ - 1 / x₂ = sqrt 37 / 3) :=
  sorry

end quadratic_roots_property_l46_46068


namespace minimum_value_of_expression_l46_46608

open Real

theorem minimum_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  ∃ s : ℝ, (a + b + c = s) ∧ (a^2 + b^2 + c^2 + 1/(s^3) ≥ real.cbrt (1/12)) := sorry

end minimum_value_of_expression_l46_46608


namespace exists_group_of_50_schools_l46_46179

-- Definitions for the problem conditions
variables (schools : Type) (male_contestants female_contestants : schools -> ℕ)

-- There are 99 schools
def number_of_schools : Nat := 99

-- The count of schools should be equal to 99
variable [fintype_schools : Fintype schools]
variable (school_count : nat.card schools = number_of_schools)


-- Statement for the existence of a subset of 50 schools
theorem exists_group_of_50_schools 
  (total_male total_female : ℕ)
  (hmale : (Σ s : schools, male_contestants s) = total_male)
  (hfemale : (Σ s : schools, female_contestants s) = total_female) :
  ∃ (group : Finset schools),
    group.card = 50 ∧
    (Σ s in group, male_contestants s) ≥ total_male / 2 ∧
    (Σ s in group, female_contestants s) ≥ total_female / 2 :=
sorry

end exists_group_of_50_schools_l46_46179


namespace max_value_sequence_l46_46679

theorem max_value_sequence (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a (n + 1) = (-1 : ℝ)^n * n - a n)
  (h2 : a 10 = a 1) :
  ∃ n, a n * a (n + 1) = 33 / 4 :=
sorry

end max_value_sequence_l46_46679


namespace graph_three_lines_no_common_point_l46_46458

theorem graph_three_lines_no_common_point :
  ∀ x y : ℝ, x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3) →
    x + 2*y - 3 = 0 ∨ x = y ∨ x = -y :=
by sorry

end graph_three_lines_no_common_point_l46_46458


namespace polynomial_root_arithmetic_sequence_l46_46288

theorem polynomial_root_arithmetic_sequence :
  (∃ (a d : ℝ), 
    (64 * (a - d)^3 + 144 * (a - d)^2 + 92 * (a - d) + 15 = 0) ∧
    (64 * a^3 + 144 * a^2 + 92 * a + 15 = 0) ∧
    (64 * (a + d)^3 + 144 * (a + d)^2 + 92 * (a + d) + 15 = 0) ∧
    (2 * d = 1)) := sorry

end polynomial_root_arithmetic_sequence_l46_46288


namespace domain_single_point_at_64_l46_46227

noncomputable def g1 (x : ℝ) := (2 - x)^(1/3)
noncomputable def gn : ℕ → (ℝ → ℝ)
| 1 := g1
| (n + 1) := λ x, gn n (real.sqrt(n^3 - x))

theorem domain_single_point_at_64 :
  ∃ M d, M = 4 ∧ d = 64 ∧ ∀ x, (gn M x).undefined ↔ x = d :=
by 
  sorry

end domain_single_point_at_64_l46_46227


namespace cos_double_angle_identity_l46_46992

theorem cos_double_angle_identity : 
  (∀ α : ℝ, sin (α - (Real.pi / 6)) = 2 / 3 → cos (2 * α + 2 * Real.pi / 3) = -1 / 9) :=
begin
  intro α,
  intro h,
  sorry
end

end cos_double_angle_identity_l46_46992


namespace largest_four_digit_sum_20_l46_46714

theorem largest_four_digit_sum_20 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n.digits 10).sum = 20 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m.digits 10).sum = 20 → n ≥ m :=
begin
  sorry
end

end largest_four_digit_sum_20_l46_46714


namespace num_integers_divisors_l46_46853

-- Defining the sequence \( \{a_n\}_{n \geq 1} \)
def seq (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else (n^(seq (n - 1)))

-- Prove the main statement
theorem num_integers_divisors : 
  (finset.filter (λ k, (k + 1) ∣ (seq k) - 1) (finset.Icc 2 2020)).card = 1009 :=
by
  sorry

end num_integers_divisors_l46_46853


namespace oil_bill_january_l46_46674

theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 :=
by
  sorry

end oil_bill_january_l46_46674


namespace arithmetic_sequence_sum_l46_46307

theorem arithmetic_sequence_sum (a d x y : ℤ) 
  (h1 : a = 3) (h2 : d = 5) 
  (h3 : x = a + d) 
  (h4 : y = x + d) 
  (h5 : y = 18) 
  (h6 : x = 13) : x + y = 31 := by
  sorry

end arithmetic_sequence_sum_l46_46307


namespace ellen_needs_thirteen_golf_carts_l46_46864

theorem ellen_needs_thirteen_golf_carts :
  ∀ (patrons_from_cars patrons_from_bus patrons_per_cart : ℕ), 
  patrons_from_cars = 12 → 
  patrons_from_bus = 27 → 
  patrons_per_cart = 3 →
  (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := 
by 
  intros patrons_from_cars patrons_from_bus patrons_per_cart h1 h2 h3 
  have h: patrons_from_cars + patrons_from_bus = 39 := by 
    rw [h1, h2] 
    norm_num
  rw[h, h3]
  norm_num
  sorry

end ellen_needs_thirteen_golf_carts_l46_46864


namespace quadratic_graph_value_at_3_l46_46294

theorem quadratic_graph_value_at_3 :
  ∃ (a b c n : ℚ), (∀ x : ℚ, y = a * x^2 + b * x + c) ∧ 
  (∃ v : ℚ, y = a * (x + 2)^2 + v - 3) ∧ 
  (y = -3) ∧ 
  (x = -2) ∧ 
  (y = 10) ∧ 
  (x = 1) ∧ 
  (x = 3) :=
begin
  sorry,
end

end quadratic_graph_value_at_3_l46_46294


namespace probability_of_continuous_stripe_pattern_l46_46861

def tetrahedron_stripes := 
  let faces := 4
  let configurations_per_face := 2
  2 ^ faces

def continuous_stripe_probability := 
  let total_configurations := tetrahedron_stripes
  1 / total_configurations * 4 -- Since final favorable outcomes calculation is already given and inferred to be 1/4.
  -- or any other logic that follows here based on problem description but this matches problem's derivation

theorem probability_of_continuous_stripe_pattern : continuous_stripe_probability = 1 / 4 := by
  sorry

end probability_of_continuous_stripe_pattern_l46_46861


namespace pyramid_dihedral_angle_l46_46259

/-- 
Given a pyramid PQRST with a square base QRST, where PR = PQ = PS = PT, and angle QRT = 60 degrees. 
Let φ be the measure of the dihedral angle formed by faces PQR and PQS. 
The cosine of the dihedral angle is given as cos φ = x + sqrt y, where x and y are integers. 
We need to prove that x + y = 3.
-/
theorem pyramid_dihedral_angle (P Q R S T : Point)
  (h_base_sq : square QRST)
  (h_congruent_edges : PR = PQ ∧ PQ = PS ∧ PS = PT)
  (h_angle_QRT : ∠QRT = 60) :
  let φ := dihedral_angle (plane3 P Q R) (plane3 P Q S) in
  ∃ (x y : ℤ), (cos φ = x + sqrt y) ∧ (x + y = 3) :=
sorry

end pyramid_dihedral_angle_l46_46259


namespace slope_of_line_l46_46127

def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 3 = 1

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def line_slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

def line_intersects_ellipse (A B : Point) : Prop :=
  ellipse A.x A.y ∧ ellipse B.x B.y

theorem slope_of_line {A B M : Point} (h_inter : line_intersects_ellipse A B)
  (h_mid : midpoint A B M) (hm : M.x = 1) (hm : M.y = 1) : 
  line_slope A B = -1 / 2 := 
sorry

end slope_of_line_l46_46127


namespace transformation_incorrect_l46_46204

def abs (x : Int) : Int := if x < 0 then -x else x

theorem transformation_incorrect (a b : Int) (hb : b ≠ 0) :
  abs (-a) / b ≠ a / (-b) := by
  sorry

end transformation_incorrect_l46_46204


namespace waiter_customers_l46_46818

theorem waiter_customers (n_tables : ℕ) (women_per_table men_per_table : ℕ) 
  (h_tables : n_tables = 9) (h_women_per_table : women_per_table = 7) 
  (h_men_per_table : men_per_table = 3) : 
  n_tables * (women_per_table + men_per_table) = 90 :=
by {
  rw [h_tables, h_women_per_table, h_men_per_table],
  norm_num,
  sorry
}

end waiter_customers_l46_46818


namespace f_value_l46_46907

open Real

variables (α : ℝ)

def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * sin (-α + (3 * π / 2))) /
  (tan (-α - π) * sin (-π - α) * cos (-π + α))

theorem f_value {α : ℝ} (hα2 : π / 2 < α ∧ α < π) (hα : sin α = 3 / 5) :
  f α = -16 / 15 :=
by
  sorry

end f_value_l46_46907


namespace range_of_a_l46_46941

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 < x → x < y → f x < f y

-- Main statement
theorem range_of_a (h1 : is_odd_function f) (h2 : is_monotonically_increasing f) :
  f (Real.exp (| a / 2 - 1 |)) + f (-Real.sqrt Real.exp 1) < 0 ↔ 1 < a ∧ a < 3 := by
  sorry

end range_of_a_l46_46941


namespace general_term_of_arithmetic_sequence_sum_of_absolute_terms_l46_46677

-- Define the arithmetic sequence and the sum of the first n terms
variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)

-- Given conditions
axiom a4_eq_3 : a_n 4 = 3
axiom S5_eq_25 : S_n 5 = 25

-- Define arithmetic sequence property and sum of first n terms
noncomputable def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

noncomputable def sum_of_first_n_terms (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  (1+n) * a_n 1 + (n*(n+1)) / 2

-- Prove the general formula for a_n
theorem general_term_of_arithmetic_sequence (a_n : ℕ → ℤ) (a1 : ℤ) (d : ℤ)
    (h1 : a_n 4 = 3) (h2 : S_n 5 = 25) : a_n n = 11 - 2 * n :=
sorry

-- Prove the sum of absolute values of terms
theorem sum_of_absolute_terms (a_n : ℕ → ℤ) (b_n : ℕ → ℤ) (n : ℕ)
    (a1 : ℤ) (d : ℤ)
    (h1 : a_n 4 = 3) (h2 : S_n 5 = 25)
    (b_def : ∀ n, b_n n = abs (a_n n)) :
  S_n n = if n ≤ 5 then 10 * n - n^2 else n^2 - 9 * n + 50 :=
sorry

end general_term_of_arithmetic_sequence_sum_of_absolute_terms_l46_46677


namespace decagon_partition_impossible_l46_46205

theorem decagon_partition_impossible :
  ∀ (n m : ℕ), (n - m = 10) ∧ (n % 3 = 0) ∧ (m % 3 = 0) → false :=
by {
  intro n m,
  rintro ⟨h1, ⟨h2, h3⟩⟩,
  have mod_eq : n - m ≡ 0 [MOD 3] := by {
    rw [nat.modeq.sub], assumption,
    },
  have contr : 10 ≡ 0 [MOD 3] := by {
    exact modeq.symm mod_eq,
    },
  have not_mod_eq : 10 % 3 = 1 := by norm_num,
  contradiction,
}

end decagon_partition_impossible_l46_46205


namespace sum_faces_edges_vertices_eq_26_l46_46342

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l46_46342


namespace lcm_48_180_value_l46_46033

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l46_46033


namespace maximum_value_of_function_l46_46133

theorem maximum_value_of_function :
  (∀ f: ℝ → ℝ, (∀ x: ℝ, f(x) = 2 * f' 1 * real.log x - x) → 
  ∃ x_max: ℝ, x_max = 2 ∧ f 2 = 2 * real.log 2 - 2) :=
by
  sorry

end maximum_value_of_function_l46_46133


namespace perpendicular_vectors_have_magnitude_five_l46_46548

-- Definitions of the vectors and perpendicular condition
variables {t : ℝ}
def a : ℝ × ℝ := (t, 1)
def b : ℝ × ℝ := (-2, t + 2)

theorem perpendicular_vectors_have_magnitude_five (h : t - 2 * t + 2 = 0) :
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2)) = 5 :=
by
  sorry

end perpendicular_vectors_have_magnitude_five_l46_46548


namespace sum_faces_edges_vertices_l46_46353

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46353


namespace find_interval_l46_46954

noncomputable def f (a b c d x : ℝ) : ℝ :=
  (1/3) * a * x^3 + (1/2) * b * x^2 + c * x + d

noncomputable def f_deriv (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_interval (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h_tangent : f_deriv a b c 1 = 0) :
    (∃ m n : ℝ, (m < n) ∧ n - m ∈ set.Ioo (3/2 : ℝ) (3 : ℝ)) :=
sorry

end find_interval_l46_46954


namespace integral_sin4_cos2_l46_46888

theorem integral_sin4_cos2 (x : ℝ) :
  ∫ (t : ℝ) in 0..x, (sin t) ^ 4 * (cos t) ^ 2 = (1 / 16) * x - (1 / 64) * sin(4 * x) - (1 / 48) * (sin(2 * x)) ^ 3 + C :=
by sorry

end integral_sin4_cos2_l46_46888


namespace monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46963

def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_when_non_positive (a : ℝ) (h_a : a ≤ 0) : ∀ x : ℝ, (a * Real.exp x - 1) ≤ 0 := 
by 
  intro x
  sorry

theorem monotonicity_when_positive (a : ℝ) (h_a : a > 0) : 
  ∀ x : ℝ, 
    (x < Real.log (1 / a) → (a * Real.exp x - 1) < 0) ∧ 
    (x > Real.log (1 / a) → (a * Real.exp x - 1) > 0) := 
by 
  intro x
  sorry

theorem inequality_when_positive (a : ℝ) (h_a : a > 0) : ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := 
by 
  intro x
  sorry

end monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46963


namespace lcm_48_180_l46_46041

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l46_46041


namespace feasibility_of_Q_l46_46500

noncomputable def median (a b c : ℝ) : ℝ :=
if a ≤ b then
  if b ≤ c then b else (if a ≤ c then c else a)
else
  if a ≤ c then a else (if b ≤ c then c else b)

theorem feasibility_of_Q (a_2 a_1 a_0 : ℝ)
  (h_distinct : a_2 ≠ a_1 ∧ a_1 ≠ a_0 ∧ a_0 ≠ a_2)
  (h_P_minus1 : a_2 - a_1 + a_0 = 1)
  (h_P_0 : a_0 = 2)
  (h_P_1 : a_2 + a_1 + a_0 = 3) :
    let m := median a_2 a_1 a_0 in
    ¬(m * (-1) + m = 3 ∧ m = 1 ∧ m * 1 + m = 2) :=
by
sorry

end feasibility_of_Q_l46_46500


namespace sin_of_alpha_in_fourth_quadrant_l46_46514

-- Definitions for the conditions
def inFourthQuadrant (α : Real) : Prop := 
  ∃ n : ℤ, α = 2 * n * π + (7 * π / 4) ∧ α < 2 * π ∧ α > (3 * π / 2)

theorem sin_of_alpha_in_fourth_quadrant : 
  ∀ (α : ℝ), inFourthQuadrant α → cos α = 1 / 3 → sin α = - (2 * Real.sqrt 2 / 3) := 
by 
  sorry

end sin_of_alpha_in_fourth_quadrant_l46_46514


namespace similar_triangle_perimeter_l46_46829

noncomputable def is_similar_triangles (a b c a' b' c' : ℝ) := 
  ∃ (k : ℝ), k > 0 ∧ (a = k * a') ∧ (b = k * b') ∧ (c = k * c')

noncomputable def is_isosceles (a b c : ℝ) := (a = b) ∨ (a = c) ∨ (b = c)

theorem similar_triangle_perimeter :
  ∀ (a b c a' b' c' : ℝ),
    is_isosceles a b c → 
    is_similar_triangles a b c a' b' c' →
    c' = 42 →
    (a = 12) → 
    (b = 12) → 
    (c = 14) →
    (b' = 36) →
    (a' = 36) →
    a' + b' + c' = 114 :=
by
  intros
  sorry

end similar_triangle_perimeter_l46_46829


namespace total_marks_proof_l46_46261

variable (total_marks : ℝ)

def Ram_marks := 450
def percentage := 0.90
def given_condition := Ram_marks = percentage * total_marks

theorem total_marks_proof (h : given_condition) : total_marks = 500 :=
by
  sorry

end total_marks_proof_l46_46261


namespace length_of_one_string_l46_46382

theorem length_of_one_string (total_length : ℕ) (num_strings : ℕ) (h_total_length : total_length = 98) (h_num_strings : num_strings = 7) : total_length / num_strings = 14 := by
  sorry

end length_of_one_string_l46_46382


namespace number_of_solutions_l46_46478

theorem number_of_solutions : 
  (∃ x : ℝ, (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2 ∧ x ≠ 0 ∧ x ≠ 4) ↔ 1 :=
by
  sorry

end number_of_solutions_l46_46478


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46360

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46360


namespace tan_A_in_right_triangle_l46_46193

theorem tan_A_in_right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (angle_A angle_B angle_C : ℝ) 
  (sin_B : ℚ) (tan_A : ℚ) :
  angle_C = 90 ∧ sin_B = 3 / 5 → tan_A = 4 / 3 := by
  sorry

end tan_A_in_right_triangle_l46_46193


namespace b_geometric_a_n_formula_and_sum_l46_46537

-- Given sequence definitions and conditions
def a : ℕ → ℕ
| 0       := 2 -- since n ∈ N*, a₁ = 2 is mapped to a 0 = 2 in Lean because indexing starts from 0
| (n + 1) := 3 * a n + 2

def b (n : ℕ) : ℕ := a n + 1

-- Prove that bₙ is a geometric sequence
theorem b_geometric : ∃ q, ∀ n, b (n + 1) = q * b n := 
sorry

-- Prove the general term formula for aₙ and the sum of the first n terms of {aₙ}
theorem a_n_formula_and_sum (n : ℕ) :
  (a n = 3 ^ n - 1) ∧ (∑ k in Finset.range n.succ, a k = (3 / 2) * (3 ^ n - 1) - n) := 
sorry

end b_geometric_a_n_formula_and_sum_l46_46537


namespace circumcenter_locus_l46_46843

noncomputable def tangent_circles (ω Ω : Circle) (X Y T : Point) (P S : Point) (hx : Center ω = X) (hy : Center Ω = Y)
  (ht : IsTangentInternally ω Ω T) (on_omega : OnCircle P Ω) (on_ω : OnCircle S ω)
  (tan : TangentToLine PS ω S) : Set Point :=
{O | IsCircumcenter O P S T}

theorem circumcenter_locus (ω Ω : Circle) (X Y T : Point) (P S : Point) (hx : Center ω = X) (hy : Center Ω = Y)
  (ht : IsTangentInternally ω Ω T) (on_omega : OnCircle P Ω) (on_ω : OnCircle S ω)
  (tan : TangentToLine PS ω S) :
  ∀ O ∈ tangent_circles ω Ω X Y T P S hx hy ht on_omega on_ω tan,
  (Distance O Y = sqrt (Distance X Y * Distance X T)) ∧ O ∉ LineIntersectCircle XY (CircleCenterRadius Y (sqrt (Distance X Y * Distance X T))) :=
sorry

end circumcenter_locus_l46_46843


namespace can_color_numbers_with_only_0_and_1_digits_l46_46850

open Nat

/-- 
  Consider the set of all natural numbers whose digits do not exceed 1.
  It is possible to color each natural number either blue or red such that
  the sum of any two different numbers of the same color contains at least
  two '1's in its decimal representation.
-/
theorem can_color_numbers_with_only_0_and_1_digits : 
  ∃ (coloring : ℕ → bool), ∀ m n : ℕ, (∀ i, (Nat.digits 10 m).nth i ∈ {0, 1}) → 
  (∀ j, (Nat.digits 10 n).nth j ∈ {0, 1}) → 
  coloring m = coloring n → 
  m ≠ n → 
  (Nat.digits 10 (m + n)).count (λ x => x = 1) ≥ 2 := 
sorry

end can_color_numbers_with_only_0_and_1_digits_l46_46850


namespace area_multiple_l46_46014

-- Definitions based on given conditions
variables {A B C A' B' C' : Type} [metric_space A] [metric_space B] [metric_space C]
{AB AC BC : ℝ} {A'B' A'C' B'C' : Type}

-- Defining the original triangle and extended vertices
def original_triangle (ABC : Triangle A B C) :=
 ∀ (A B C : Type) (AB AC BC : ℝ),
 extend (A B C) A' B' C' AB AC BC

-- Defining the new larger triangle
def new_triangle (A'B'C' : Triangle A' B' C') :=
 ∀ (A' B' C' : Type), 
    extended_side A' B' = original_side A B ∧
    extended_side B' C' = original_side B C ∧
    extended_side C' A' = original_side C A

-- The proof that the area of the new triangle is 7 times the area of the original triangle
theorem area_multiple 
  (ABC : Triangle A B C) 
  (A'B'C' : Triangle A' B' C')
  (h1 : original_triangle ABC)
  (h2 : new_triangle A'B'C') :
  area A'B'C' = 7 * area ABC :=
begin
  sorry
end

end area_multiple_l46_46014


namespace correctFractions_equivalence_l46_46701

def correctFractions: List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]

def isValidCancellation (num den: ℕ): Prop :=
  ∃ n₁ n₂ n₃ d₁ d₂ d₃: ℕ, 
    num = 10 * n₁ + n₂ ∧
    den = 10 * d₁ + d₂ ∧
    ((n₁ = d₁ ∧ n₂ = d₂) ∨ (n₁ = d₃ ∧ n₃ = d₂)) ∧
    n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ d₁ ≠ 0 ∧ d₂ ≠ 0

theorem correctFractions_equivalence : 
  ∀ (frac : ℕ × ℕ), frac ∈ correctFractions → 
    ∃ a b: ℕ, correctFractions = [(a, b)] ∧ 
      isValidCancellation a b := sorry

end correctFractions_equivalence_l46_46701


namespace p_eq_r_lt_q_l46_46951

noncomputable def log (x: ℝ) : ℝ := Real.log10 x

variables (a b: ℝ) (h: 0 < a ∧ a < b)

def f (x : ℝ) : ℝ := log x

def p : ℝ := f (Real.sqrt (a * b))
def q : ℝ := f ((a + b) / 2)
def r : ℝ := (f a + f b) / 2

theorem p_eq_r_lt_q 
  (h_ineq: (a + b) / 2 > Real.sqrt (a * b)) 
  (log_mono: ∀ x y, x < y → log x < log y) :
  p = r ∧ p < q :=
by sorry

end p_eq_r_lt_q_l46_46951


namespace eva_hits_10_l46_46644

def unique_scores (s : List ℕ) : Prop :=
  ∀ i j, i ≠ j → s.getD i 0 ≠ s.getD j 0

def total_score (s : List ℕ) (total : ℕ) : Prop :=
  s.sum = total

def all_scores : List (ℕ × ℕ) := [(21, 1), (12, 2), (18, 3), (26, 4), (28, 5), (20, 6)]

theorem eva_hits_10 :
  ∃ (s1 s2 s3 : ℕ), [s1, s2, s3] ~ (eva_scores : List ℕ) ∧
  unique_scores eva_scores ∧
  total_score eva_scores 28 ∧
  10 ∈ eva_scores :=
by
  sorry

end eva_hits_10_l46_46644


namespace reachability_in_subregion_l46_46409

theorem reachability_in_subregion (cities : Set ℕ) (roads : ℕ → ℕ → Prop) :
  cities.card = 1001 →
  (∀ c, cities c → ∀ (d : ℕ), cities d → (roads c).card = 500) →
  ∀ (region : Set ℕ), region.card = 668 →
    (∀ (x y : ℕ), x ∈ region → y ∈ region → is_reachable roads region x y) :=
by
  sorry

end reachability_in_subregion_l46_46409


namespace circle_circumference_ratio_l46_46277

theorem circle_circumference_ratio (A₁ A₂ : ℝ) (h : A₁ / A₂ = 16 / 25) :
  ∃ C₁ C₂ : ℝ, (C₁ / C₂ = 4 / 5) :=
by
  -- Definitions and calculations to be done here
  sorry

end circle_circumference_ratio_l46_46277


namespace propositionC_incorrect_l46_46432

-- conditions
def Parallelogram (P : Type) :=
  ∀ p1 p2 p3 p4 : P, (diagonal p1 p3 = diagonal p2 p4) → isRectangle p1 p2 p3 p4

def IsoscelesTriangle (T : Type) :=
  ∀ t1 t2 t3 : T, (angle t1 t2 t3 = 60) → (angle t1 t3 t2 = angle t2 t3 t1) → isEquilateralTriangle t1 t2 t3

def Square (S : Type) :=
  ∀ s1 s2 s3 s4 : S, (diagonal s1 s3 = diagonal s2 s4) ∧ (perpendicularBisector s1 s2 s3 s4)

-- proposition C
def PropositionCI (P : Type) :=
  ∀ p1 p2 p3 : P, isRightTriangle p1 p2 p3 → (height p1 p2 p3 = halfHypotenuse p1 p2 p3)

theorem propositionC_incorrect (P : Type) : 
  Parallelogram P → IsoscelesTriangle P → Square P → ¬PropositionCI P :=
by
  sorry

end propositionC_incorrect_l46_46432


namespace system1_solution_l46_46648

theorem system1_solution (x y : ℝ) 
  (h1 : x + y = 10^20) 
  (h2 : x - y = 10^19) :
  x = 55 * 10^18 ∧ y = 45 * 10^18 := 
by
  sorry

end system1_solution_l46_46648


namespace lcm_48_180_eq_720_l46_46048

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l46_46048


namespace gcf_75_90_l46_46705

theorem gcf_75_90 : Nat.gcd 75 90 = 15 :=
by
  sorry

end gcf_75_90_l46_46705


namespace infinite_T_with_two_distinct_counts_l46_46898

def d_i (T : ℕ) (i : ℕ) : ℕ := sorry  -- Definition of d_i(T) based on the count of digit i in multiples of 1829 up to T

noncomputable def n : ℕ := 1829  -- Define n

theorem infinite_T_with_two_distinct_counts :
  ∃ᶠ T in at_top, ∃ d1 d2, d1 ≠ d2 ∧ {d_i T 1, d_i T 2, d_i T 3, d_i T 4, d_i T 5, d_i T 6, d_i T 7, d_i T 8, d_i T 9} = {d1, d2} :=
sorry

end infinite_T_with_two_distinct_counts_l46_46898


namespace quadrilateral_x_y_difference_proof_l46_46785

noncomputable def quadrilateral_x_y_difference : Prop :=
  let a := 70
  let b := 90
  let c := 130
  let d := 110

  (inscribed_in_circle a b c d) ∧
  (has_incicle a b c d) →
  ∃ x y, touches_incircle c x y ∧ |x - y| = 13

theorem quadrilateral_x_y_difference_proof : quadrilateral_x_y_difference :=
by
  sorry

end quadrilateral_x_y_difference_proof_l46_46785


namespace num_elements_in_set_S_l46_46058

theorem num_elements_in_set_S (n : ℕ) (hn : n ≥ 1) :
  let S (n : ℕ) := {k : ℕ | k > n ∧ k ∣ (30 * n - 1)}
  let S_union := ⋃ i : ℕ, S i
  (S_union.filter (< 2016)).card = 536 :=
sorry

end num_elements_in_set_S_l46_46058


namespace distinct_valid_sets_count_l46_46987

-- Define non-negative powers of 2 and 3
def is_non_neg_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a ∨ n = 3^b

-- Define the condition for sum of elements in set S to be 2014
def valid_sets (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, is_non_neg_power x) ∧ (S.sum id = 2014)

theorem distinct_valid_sets_count : ∃ (number_of_distinct_sets : ℕ), number_of_distinct_sets = 64 :=
  sorry

end distinct_valid_sets_count_l46_46987


namespace sum_of_faces_edges_vertices_l46_46374

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l46_46374


namespace cuboid_length_l46_46890

theorem cuboid_length (A b h : ℝ) (A_eq : A = 2400) (b_eq : b = 10) (h_eq : h = 16) :
    ∃ l : ℝ, 2 * (l * b + b * h + h * l) = A ∧ l = 40 := by
  sorry

end cuboid_length_l46_46890


namespace car_speed_first_hour_l46_46308

theorem car_speed_first_hour (x : ℝ) (h_second_hour_speed : x + 80 / 2 = 85) : x = 90 :=
sorry

end car_speed_first_hour_l46_46308


namespace problem_statement_l46_46073

noncomputable def circle_F1 : set (ℝ × ℝ) := { p | (p.1 + 2)^2 + p.2^2 = 49 }
noncomputable def circle_F2 : set (ℝ × ℝ) := { p | (p.1 - 2)^2 + p.2^2 = 1 }
noncomputable def curve_C : set (ℝ × ℝ) := {p | p.1 ^ 2 / 9 + p.2 ^ 2 / 5 = 1}

theorem problem_statement :
  let F1 := (⟨-2, 0⟩ : ℝ × ℝ),
      F2 := (⟨2, 0⟩ : ℝ × ℝ) in
  ∃ (R : ℝ) (P : ℝ × ℝ),
    P ∈ curve_C ∧
    ∀ Q : ℝ × ℝ, Q ∈ curve_C ∧ Q.2 ≠ 0 →
      let
        line_OQ := { p : ℝ × ℝ | ∃ m : ℝ, p.1 - Q.1 = m * (p.2 - Q.2) },
        line_through_F2 := { p : ℝ × ℝ | ∃ m : ℝ, p.1 - F2.1 = m * (p.2 - F2.2) },
        points_MN := (curve_C ∩ line_through_F2).to_finset in
      points_MN.card = 2 →
      ∃ M N : ℝ × ℝ, M ∈ curve_C ∧ N ∈ curve_C ∧
      ((M, N) ∈ points_MN ∧ 
      (1 / 2) * |(M.1 * N.2 - M.2 * N.1) +
      (N.1 * 0 - N.2 * 0) +
      (0 * M.2 - 0 * M.1)| = 10 / 3) :=
begin
  -- claim that M, N, Q are structurally as described in the problem.
  sorry
end

end problem_statement_l46_46073


namespace area_of_union_of_symmetric_triangles_l46_46744

variables {α : Type} [LinearOrderedField α]
variables {A B C O : α × α} -- vertices of the triangle and point O

-- Assume triangle ABC has area E
variable (E : α)

-- Definition of symmetric triangle wrt a point on the side of ABC
def symmetric_triangle_area (A B C : α × α) (O : α × α) : α :=
  E

-- Main theorem with given conditions and conclusion
theorem area_of_union_of_symmetric_triangles (A B C : α × α) (E : α) :
  let Φ := {Δ | ∃ (O : α × α), O ∈ {line_segment A B ∪ line_segment B C ∪ line_segment C A} ∧ Δ = symmetric_triangle_area A B C O} in
  (area_of Φ = 2 * E) :=
sorry

end area_of_union_of_symmetric_triangles_l46_46744


namespace train_speed_l46_46401

theorem train_speed (length : ℕ) (time : ℝ)
  (h_length : length = 160)
  (h_time : time = 18) :
  (length / time * 3.6 : ℝ) = 32 :=
by
  sorry

end train_speed_l46_46401


namespace distance_between_EF_and_A1C1_l46_46175

-- Define the cube and the points
def Cube := {A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ}

structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

noncomputable def E (s : ℝ) : Point := 
Point.mk (1) (0.5) (0.5)

noncomputable def F (s : ℝ) : Point := 
Point.mk (0.5) (0.5) (0)

noncomputable def A1 : Point := 
Point.mk (0) (0) (1)

noncomputable def C1 : Point :=  
Point.mk (1) (1) (1)

noncomputable def distance_between_skew_lines (L1 L2 : Point → Point) : ℝ :=
  sorry  -- Placeholder for the actual distant calculation between two skew lines.

-- Main theorem statement
theorem distance_between_EF_and_A1C1 :
  distance_between_skew_lines (fun t => Point.mk (0.5 + 0.5*t) (0.5) (0.5*t)) (fun t => Point.mk (t) (t) (1)) = (Real.sqrt 3) / 3 :=
by
  sorry  -- The actual proof goes here

end distance_between_EF_and_A1C1_l46_46175


namespace range_omega_l46_46953

noncomputable def f (ω x : ℝ) := Real.cos (ω * x + Real.pi / 6)

theorem range_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → -1 ≤ f ω x ∧ f ω x ≤ Real.sqrt 3 / 2) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
  sorry

end range_omega_l46_46953


namespace quadratic_inequality_hold_l46_46509

theorem quadratic_inequality_hold (α : ℝ) (h : 0 ≤ α ∧ α ≤ π) :
    (∀ x : ℝ, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔ 
    (α ∈ Set.Icc 0 (π / 6) ∨ α ∈ Set.Icc (5 * π / 6) π) :=
sorry

end quadratic_inequality_hold_l46_46509


namespace combined_selling_price_correctness_l46_46813

def cost_price_first := 70
def cost_price_second := 120
def cost_price_third := 150

def selling_price_first := ((0.85 * cost_price_first) * 3 / 2)
def selling_price_second := cost_price_second + (0.3 * cost_price_second)
def selling_price_third := cost_price_third - (0.2 * cost_price_third)

def combined_selling_price := selling_price_first + selling_price_second + selling_price_third

theorem combined_selling_price_correctness : combined_selling_price = 365.25 := by
  sorry

end combined_selling_price_correctness_l46_46813


namespace find_value_of_a_and_point_P_calculate_volume_of_revolution_l46_46272

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x

noncomputable def hyperbola (x y : ℝ) : Prop := x * y = 1

theorem find_value_of_a_and_point_P (a x y : ℝ):
  a ≠ 0 →
  (hyperbola (x y) ∧ y = curve a x) → 
  ∃ P : ℝ × ℝ, P = (1 / Real.sqrt 2, Real.sqrt 2) ∧ a = -4 := by
  sorry

theorem calculate_volume_of_revolution (a x : ℝ):
  a = -4 →
  x = 1 / Real.sqrt 2 →
  ∫ (u : ℝ) in 0 .. x, (curve a u)^2 - (Real.sqrt 2 * u)^2 → 
  radius = (83 / 105) / Real.sqrt 2 * Real.pi := by
  sorry

end find_value_of_a_and_point_P_calculate_volume_of_revolution_l46_46272


namespace ellen_golf_cart_trips_l46_46865

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l46_46865


namespace basketball_cricket_students_l46_46174

theorem basketball_cricket_students (B C B_union_C B_inter_C : ℕ) 
  (hB : B = 7) 
  (hC : C = 8) 
  (hB_union_C : B_union_C = 10) 
  (h_equation : B_union_C = B + C - B_inter_C) : B_inter_C = 5 := 
by
  -- Introduction of the known values
  rw [hB, hC, hB_union_C] at h_equation
  -- Solving for B ∩ C
  exact (by linarith : B_inter_C = 5)

end basketball_cricket_students_l46_46174


namespace center_and_radius_of_circle_l46_46660

theorem center_and_radius_of_circle (x y : ℝ) : 
  (x + 1)^2 + (y - 2)^2 = 4 → (x = -1 ∧ y = 2 ∧ ∃ r, r = 2) := 
by
  intro h
  sorry

end center_and_radius_of_circle_l46_46660


namespace truck_needs_additional_gallons_l46_46424

-- Definitions based on the given conditions
def miles_per_gallon : ℝ := 3
def total_miles_needed : ℝ := 90
def current_gallons : ℝ := 12

-- Function to calculate the additional gallons needed
def additional_gallons_needed (mpg : ℝ) (total_miles : ℝ) (current_gas : ℝ) : ℝ :=
  (total_miles - current_gas * mpg) / mpg

-- The main theorem to prove
theorem truck_needs_additional_gallons :
  additional_gallons_needed miles_per_gallon total_miles_needed current_gallons = 18 := 
by
  sorry

end truck_needs_additional_gallons_l46_46424


namespace proof_inequalities_equivalence_max_f_value_l46_46540

-- Definitions for the conditions
def inequality1 (x: ℝ) := |x - 2| > 1
def inequality2 (x: ℝ) := x^2 - 4 * x + 3 > 0

-- The main statements to prove
theorem proof_inequalities_equivalence : 
  {x : ℝ | inequality1 x} = {x : ℝ | inequality2 x} := 
sorry

noncomputable def f (x: ℝ) := 4 * Real.sqrt (x - 3) + 3 * Real.sqrt (5 - x)

theorem max_f_value : 
  ∃ x : ℝ, (3 ≤ x ∧ x ≤ 5) ∧ (f x = 5 * Real.sqrt 2) ∧ ∀ y : ℝ, ((3 ≤ y ∧ y ≤ 5) → f y ≤ 5 * Real.sqrt 2) :=
sorry

end proof_inequalities_equivalence_max_f_value_l46_46540


namespace part1_monotonicity_part2_inequality_l46_46965

variable (a : ℝ) (x : ℝ)
variable (h : a > 0)

def f := a * (Real.exp x + a) - x

theorem part1_monotonicity :
  if a ≤ 0 then ∀ x : ℝ, (f a x) < (f a (x + 1)) else
  if a > 0 then ∀ x : ℝ, (x < Real.log (1 / a) → (f a x) > (f a (x + 1))) ∧ 
                       (x > Real.log (1 / a) → (f a x) < (f a (x - 1))) else 
  (False) := sorry

theorem part2_inequality :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3 / 2 := sorry

end part1_monotonicity_part2_inequality_l46_46965


namespace dozen_pencils_l46_46686

-- Define the given conditions
def pencils_total : ℕ := 144
def pencils_per_dozen : ℕ := 12

-- Theorem stating the desired proof
theorem dozen_pencils (h : pencils_total = 144) (hdozen : pencils_per_dozen = 12) : 
  pencils_total / pencils_per_dozen = 12 :=
by
  sorry

end dozen_pencils_l46_46686


namespace largest_difference_l46_46603

theorem largest_difference (P Q R S T U : ℕ) 
    (hP : P = 3 * 2003 ^ 2004)
    (hQ : Q = 2003 ^ 2004)
    (hR : R = 2002 * 2003 ^ 2003)
    (hS : S = 3 * 2003 ^ 2003)
    (hT : T = 2003 ^ 2003)
    (hU : U = 2003 ^ 2002) 
    : max (P - Q) (max (Q - R) (max (R - S) (max (S - T) (T - U)))) = P - Q :=
sorry

end largest_difference_l46_46603


namespace quadratic_roots_condition_l46_46905

theorem quadratic_roots_condition (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 0) :
  ¬ ((∃ x y : ℝ, ax^2 + 2*x + 1 = 0 ∧ ax^2 + 2*y + 1 = 0 ∧ x*y < 0) ↔
     (a > 0 ∧ a ≠ 0)) :=
by
  sorry

end quadratic_roots_condition_l46_46905


namespace chocolates_bought_at_cost_price_l46_46285

variables (C S : ℝ) (n : ℕ)

-- Given conditions
def cost_eq_selling_50 := n * C = 50 * S
def gain_percent := (S - C) / C = 0.30

-- Question to prove
theorem chocolates_bought_at_cost_price (h1 : cost_eq_selling_50 C S n) (h2 : gain_percent C S) : n = 65 :=
sorry

end chocolates_bought_at_cost_price_l46_46285


namespace phone_number_exists_l46_46397

theorem phone_number_exists :
  ∃ (phone_number : ℕ),
    phone_number >= 1000000 ∧ phone_number < 10000000 ∧
    (let last_three_digits := phone_number % 1000 in
     (∃ n : ℕ, last_three_digits = 100 * n + 10 * (n + 1) + (n + 2))) ∧
    let first_five_digits := phone_number / 1000 in
    (let a := first_five_digits / 1000,
         b := (first_five_digits / 100) % 10,
         c := (first_five_digits / 10) % 10 in
     (first_five_digits = 10000 * a + 1000 * b + 100 * c + 10 * b + a) ∧
     (c = 1) ∧
     (b = 1 ∨ a = 1)) ∧
    let three_digit_number := phone_number / 1000 in
    (three_digit_number % 9 = 0) ∧
    (phone_number / 100 = 7111 ∨ (phone_number / 10) % 1000 = 111) ∧
    let first_two_digits := (phone_number / 10000) % 100,
        second_two_digits := (phone_number / 100) % 100 in
    (Nat.Prime first_two_digits ∨ Nat.Prime second_two_digits) ∧
    phone_number = 7111765 := sorry

end phone_number_exists_l46_46397


namespace area_semicircles_percent_increase_l46_46798

noncomputable def radius_large_semicircle (length: ℝ) : ℝ := length / 2
noncomputable def radius_small_semicircle (width: ℝ) : ℝ := width / 2

noncomputable def area_semicircle (radius: ℝ) : ℝ := (real.pi * radius^2) / 2

theorem area_semicircles_percent_increase
  (length: ℝ) (width: ℝ)
  (h_length: length = 12) (h_width: width = 8) :
  let 
    large_radius := radius_large_semicircle length,
    small_radius := radius_small_semicircle width,
    area_large := 2 * area_semicircle large_radius,
    area_small := 2 * area_semicircle small_radius
  in
  (area_large / area_small - 1) * 100 = 125 :=
by
  sorry

end area_semicircles_percent_increase_l46_46798


namespace semicircle_area_percentage_difference_l46_46787

-- Define the rectangle dimensions
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 8

-- Define the diameters and radii of the semicircles
def large_semicircle_radius : ℝ := rectangle_length / 2
def small_semicircle_radius : ℝ := rectangle_width / 2

-- Define the areas of the full circles made from the semicircles
def large_circle_area : ℝ := real.pi * (large_semicircle_radius ^ 2)
def small_circle_area : ℝ := real.pi * (small_semicircle_radius ^ 2)

-- Define the percentage larger question
def percent_larger (a b : ℝ) : ℝ := ((a - b) / b) * 100

-- Formal proof statement
theorem semicircle_area_percentage_difference : 
  percent_larger large_circle_area small_circle_area = 125 := 
by
  sorry

end semicircle_area_percentage_difference_l46_46787


namespace area_quadrilateral_ABCD_l46_46584

theorem area_quadrilateral_ABCD :
  ∀ (A B C D E : ℝ) (h1 : ∠(A, E, B) = 60)
                     (h2 : ∠(B, E, C) = 60)
                     (h3 : ∠(C, E, D) = 60)
                     (h4 : triangle.right_angle E B A)
                     (h5 : triangle.right_angle E C B)
                     (h6 : triangle.right_angle E D C)
                     (h7 : dist A E = 24),
    area (quadrilateral A B C D) = (189 / 2) * √3 := by
sorry

end area_quadrilateral_ABCD_l46_46584


namespace solve_quadratic_l46_46680

theorem solve_quadratic {x : ℝ} : x^2 = 2 * x ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l46_46680


namespace product_equals_32_l46_46727

theorem product_equals_32 :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_equals_32_l46_46727


namespace sum_faces_edges_vertices_l46_46355

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46355


namespace vertex_in_one_cycle_l46_46403

-- Define a cactus graph
structure cactus_graph (V : Type) :=
(graph : simple_graph V)
(is_connected : graph.connected)
(no_shared_edges_in_cycles : ∀ (C₁ C₂ : V → Prop) (e : graph.edge_set),
  (is_cycle graph C₁ → is_cycle graph C₂ → C₁ ≠ C₂ → ¬ (e ∈ (cycle_edges graph C₁ ∩ cycle_edges graph C₂))))

-- Theorem: In every nonempty cactus graph, there exists a vertex that is part of at most one cycle.
theorem vertex_in_one_cycle {V : Type} (C : cactus_graph V) (nonempty : ∃ v : V, true):
  ∃ v : V, ∀ (C' : V → Prop), is_cycle C.graph C' → (v ∈ cycle_vertices C.graph C' → ∀ (C'' : V → Prop), is_cycle C.graph C'' → (C' = C'' ∨ v ∉ cycle_vertices C.graph C'')) :=
by
  sorry

end vertex_in_one_cycle_l46_46403


namespace sin_theta_correct_l46_46111

noncomputable def sin_theta (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real :=
  -Real.sqrt 2 / 2

theorem sin_theta_correct (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan (Real.arctan (-a)) = -a) : sin_theta a h1 h2 = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_theta_correct_l46_46111


namespace quadrant_of_complex_mul_l46_46462

theorem quadrant_of_complex_mul : 
  let z := (3 + complex.i) * (1 - complex.i) in
  z.re > 0 ∧ z.im < 0 := by
sorry

end quadrant_of_complex_mul_l46_46462


namespace simplify_and_evaluate_expr_l46_46640

noncomputable def expr (x : Real) : Real :=
  (1 / (x^2 + 2 * x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1))

theorem simplify_and_evaluate_expr :
  let x := 2 * Real.sqrt 5 - 1 in
  expr x = Real.sqrt 5 / 10 := by
  sorry

end simplify_and_evaluate_expr_l46_46640


namespace problem_l46_46931

theorem problem (a b : ℤ) (h : (2 * a + b) ^ 2 + |b - 2| = 0) : (-a - b) ^ 2014 = 1 := 
by
  sorry

end problem_l46_46931


namespace chess_tournament_participants_and_masters_l46_46173

noncomputable def participants_in_tournament (n m : ℕ) : Prop :=
  9 < n ∧ n < 25 ∧ (∃ m : ℕ, m^2 = n ∧ (n - 2 * m = 4 ∨ n - 2 * m = -4))

theorem chess_tournament_participants_and_masters :
  ∃ n m : ℕ, participants_in_tournament n m ∧ ((n = 16) → (m = 6 ∨ m = 10)) :=
begin
  sorry
end

end chess_tournament_participants_and_masters_l46_46173


namespace monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46961

def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_when_non_positive (a : ℝ) (h_a : a ≤ 0) : ∀ x : ℝ, (a * Real.exp x - 1) ≤ 0 := 
by 
  intro x
  sorry

theorem monotonicity_when_positive (a : ℝ) (h_a : a > 0) : 
  ∀ x : ℝ, 
    (x < Real.log (1 / a) → (a * Real.exp x - 1) < 0) ∧ 
    (x > Real.log (1 / a) → (a * Real.exp x - 1) > 0) := 
by 
  intro x
  sorry

theorem inequality_when_positive (a : ℝ) (h_a : a > 0) : ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := 
by 
  intro x
  sorry

end monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l46_46961


namespace find_a_2_find_a_n_l46_46223

-- Define the problem conditions and questions as types
def S_3 (a_1 a_2 a_3 : ℝ) : Prop := a_1 + a_2 + a_3 = 7
def arithmetic_mean_condition (a_1 a_2 a_3 : ℝ) : Prop :=
  (a_1 + 3 + a_3 + 4) / 2 = 3 * a_2

-- Prove that a_2 = 2 given the conditions
theorem find_a_2 (a_1 a_2 a_3 : ℝ) (h1 : S_3 a_1 a_2 a_3) (h2: arithmetic_mean_condition a_1 a_2 a_3) :
  a_2 = 2 := 
sorry

-- Define the general term for a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Prove the formula for the general term of the geometric sequence given the conditions and a_2 found
theorem find_a_n (a : ℕ → ℝ) (q : ℝ) (h1 : S_3 (a 1) (a 2) (a 3)) (h2 : arithmetic_mean_condition (a 1) (a 2) (a 3)) (h3 : geometric_sequence a q) : 
  (q = (1/2) → ∀ n, a n = (1 / 2)^(n - 3))
  ∧ (q = 2 → ∀ n, a n = 2^(n - 1)) := 
sorry

end find_a_2_find_a_n_l46_46223


namespace smallest_possible_value_of_sum_l46_46392

theorem smallest_possible_value_of_sum (a b : ℤ) (h1 : a > 6) (h2 : ∃ a' b', a' - b' = 4) : a + b < 11 := 
sorry

end smallest_possible_value_of_sum_l46_46392


namespace angle_BDC_15_degrees_l46_46649

variable {A B C D : Type}
variables (triangle_congruent_ABC_ACD : congruent_triangles ABC ACD)
variables (h₁ : AB = AC)
variables (h₂ : AC = AD)
variables (h₃ : ∠ BAC = 30)

theorem angle_BDC_15_degrees
  (triangle_congruent_ABC_ACD : congruent_triangles ABC ACD)
  (h₁ : AB = AC)
  (h₂ : AC = AD)
  (h₃ : ∠ BAC = 30) :
  ∠ BDC = 15 :=
sorry

end angle_BDC_15_degrees_l46_46649


namespace ozerny_bus_connections_l46_46571

def cities_bus_connections (n : ℕ) : ℕ :=
  if h : n = 47 then 23 else 0

theorem ozerny_bus_connections :
  ∀ (n : ℕ),
  (n = 47) →
  (∀ i, i < 47 → ∃! k, k ≠ 46 ∧ k ≠ i ∧ cities_bus_connections k ≠ k ) →
  (cities_bus_connections 47 = 23) :=
begin
  intros n hn huniq,
  rw hn,
  -- Proof to show cities_bus_connections 47 = 23
  sorry,
end

end ozerny_bus_connections_l46_46571


namespace sequence_term_equality_l46_46678

noncomputable def seq (n : ℕ) : ℕ → ℚ
| 0       := 2
| (n + 1) := seq n + n / 2

theorem sequence_term_equality : seq 19 = 97 := sorry

end sequence_term_equality_l46_46678


namespace union_set_solution_l46_46164

theorem union_set_solution (M N : Set ℝ) 
    (hM : M = { x | 0 ≤ x ∧ x ≤ 3 }) 
    (hN : N = { x | x < 1 }) : 
    M ∪ N = { x | x ≤ 3 } := 
by 
    sorry

end union_set_solution_l46_46164


namespace cost_price_percentage_of_marked_price_l46_46284

theorem cost_price_percentage_of_marked_price (MP CP : ℝ) (discount gain_percent : ℝ) 
  (h_discount : discount = 0.12) (h_gain_percent : gain_percent = 0.375) 
  (h_SP_def : SP = MP * (1 - discount))
  (h_SP_gain : SP = CP * (1 + gain_percent)) :
  CP / MP = 0.64 :=
by
  sorry

end cost_price_percentage_of_marked_price_l46_46284


namespace bus_overloaded_l46_46757

theorem bus_overloaded : 
  ∀ (capacity : ℕ) (first_pickup_ratio : ℚ) (next_pickup : ℕ) (bus_full : capacity = 80) (entered_first : first_pickup_ratio = 3/5) (next_pickup_point_waiting : next_pickup = 50), 
  let entered := (first_pickup_ratio * capacity).to_nat in -- people entered at first pickup
  let available_seats := capacity - entered in -- available seats after first pickup
  let could_not_take_bus := next_pickup - available_seats in -- people who could not take the bus
  could_not_take_bus = 18 := 
by 
  intros capacity first_pickup_ratio next_pickup bus_full entered_first next_pickup_point_waiting 
  let entered := (first_pickup_ratio * capacity).to_nat 
  let available_seats := capacity - entered 
  let could_not_take_bus := next_pickup - available_seats 
  sorry

end bus_overloaded_l46_46757


namespace find_k_l46_46544

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k
  (a b : vector)
  (h_a : a = (2, -1))
  (h_b : b = (-1, 4))
  (h_perpendicular : dot_product (a.1 - k * b.1, a.2 + 4 * k) (3, -5) = 0) :
  k = -11/17 := sorry

end find_k_l46_46544


namespace A_eq_fibonacci_n_plus_2_A_33_equals_5702887_l46_46411

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

def A : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := A (n + 1) + A n

theorem A_eq_fibonacci_n_plus_2 (n : ℕ) : A n = fibonacci (n + 2) := sorry

theorem A_33_equals_5702887 : A 33 = 5702887 := by
  have key := A_eq_fibonacci_n_plus_2 33
  sorry

end A_eq_fibonacci_n_plus_2_A_33_equals_5702887_l46_46411


namespace real_solutions_iff_a_geq_3_4_l46_46622

theorem real_solutions_iff_a_geq_3_4:
  (∃ (x y : ℝ), x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3 / 4 := sorry

end real_solutions_iff_a_geq_3_4_l46_46622


namespace plane_equation_exists_l46_46475

theorem plane_equation_exists : 
  ∃ A B C D : ℤ, 
    A > 0 ∧
    Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
    (A * 2 + B * 3 + C * (-1) + D = 0) ∧ 
    (A, B, C) = (3, -4, 1) ∧
    (A * 1 + B * 1 + C * 1 + D = 7) := 
by
  use 3, -4, 1, 7
  split
  · exact Nat.zero_lt_succ 2
  · split
    · simp [Int.gcd]
    · split
      · ring
      · split
        · rfl
        · ring

end plane_equation_exists_l46_46475


namespace speed_of_second_train_equivalent_l46_46319

noncomputable def relative_speed_in_m_per_s (time_seconds : ℝ) (total_distance_m : ℝ) : ℝ :=
total_distance_m / time_seconds

noncomputable def relative_speed_in_km_per_h (relative_speed_m_per_s : ℝ) : ℝ :=
relative_speed_m_per_s * 3.6

noncomputable def speed_of_second_train (relative_speed_km_per_h : ℝ) (speed_of_first_train_km_per_h : ℝ) : ℝ :=
relative_speed_km_per_h - speed_of_first_train_km_per_h

theorem speed_of_second_train_equivalent
  (length_of_first_train length_of_second_train : ℝ)
  (speed_of_first_train_km_per_h : ℝ)
  (time_of_crossing_seconds : ℝ) :
  speed_of_second_train
    (relative_speed_in_km_per_h (relative_speed_in_m_per_s time_of_crossing_seconds (length_of_first_train + length_of_second_train)))
    speed_of_first_train_km_per_h = 36 := by
  sorry

end speed_of_second_train_equivalent_l46_46319


namespace eq_circle_equation_eq_line_equation_l46_46910

noncomputable theory

def circle_center : ℝ × ℝ := (-1, 2)
def tangent_line (x y : ℝ) : Prop := x + 2 * y + 7 = 0
def intersecting_point : ℝ × ℝ := (-2, 0)
def chord_length : ℝ := 2 * Real.sqrt 19

theorem eq_circle_equation :
  ∃ r : ℝ, r = Real.sqrt (20) ∧ 
  ∀ (x y : ℝ), ((x + 1)^2 + (y - 2)^2 = 20) := 
sorry

theorem eq_line_equation :
  ∀ x y : ℝ, 
  ( (x = -2) ∨ (3 * x - 4 * y + 6 = 0)) ∧
  (x + 2 * y + 7 = 0) :=
sorry

end eq_circle_equation_eq_line_equation_l46_46910


namespace find_base_l46_46582

noncomputable def base_satisfies_first_transaction (s : ℕ) : Prop :=
  5 * s^2 + 3 * s + 460 = s^3 + s^2 + 1

noncomputable def base_satisfies_second_transaction (s : ℕ) : Prop :=
  s^2 + 2 * s + 2 * s^2 + 6 * s = 5 * s^2

theorem find_base (s : ℕ) (h1 : base_satisfies_first_transaction s) (h2 : base_satisfies_second_transaction s) :
  s = 4 :=
sorry

end find_base_l46_46582


namespace cosine_of_angle_between_skew_lines_l46_46807

-- Variables and conditions
def A := (0, 0, 0) : ℝ × ℝ × ℝ
def B := (3 * Real.sqrt 3 / 2, 3 / 2, 0) : ℝ × ℝ × ℝ
def B1 := (3 * Real.sqrt 3 / 2, 3 / 2, 4) : ℝ × ℝ × ℝ
def C1 := (0, 3, 4) : ℝ × ℝ × ℝ

def vector_AB1 := ((3 * Real.sqrt 3 / 2) - 0, (3 / 2) - 0, 4 - 0) : ℝ × ℝ × ℝ
def vector_BC1 := ((0 - 3 * Real.sqrt 3 / 2), (3 - 3 / 2), 4 - 0) : ℝ × ℝ × ℝ

noncomputable def cosine_angle_AB1_BC1 : ℝ :=
  let dot_product := (vector_AB1.1 * vector_BC1.1 + vector_AB1.2 * vector_BC1.2 + vector_AB1.3 * vector_BC1.3)
  let magnitude_AB1 := Real.sqrt (vector_AB1.1 ^ 2 + vector_AB1.2 ^ 2 + vector_AB1.3 ^ 2)
  let magnitude_BC1 := Real.sqrt (vector_BC1.1 ^ 2 + vector_BC1.2 ^ 2 + vector_BC1.3 ^ 2)
  abs dot_product / (magnitude_AB1 * magnitude_BC1)

theorem cosine_of_angle_between_skew_lines :
  cosine_angle_AB1_BC1 = 23 / 50 :=
by
  -- Skipping the actual proof
  sorry

end cosine_of_angle_between_skew_lines_l46_46807


namespace solve_equation_l46_46645

noncomputable def equation (x : ℝ) : Prop :=
  -2 * x ^ 3 = (5 * x ^ 2 + 2) / (2 * x - 1)

theorem solve_equation (x : ℝ) :
  equation x ↔ (x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4) :=
by
  sorry

end solve_equation_l46_46645


namespace tea_blend_gain_percent_l46_46783

theorem tea_blend_gain_percent :
  let cost_18 := 18
  let cost_20 := 20
  let ratio_5_to_3 := (5, 3)
  let selling_price := 21
  let total_cost := (ratio_5_to_3.1 * cost_18) + (ratio_5_to_3.2 * cost_20)
  let total_weight := ratio_5_to_3.1 + ratio_5_to_3.2
  let cost_price_per_kg := total_cost / total_weight
  let gain_percent := ((selling_price - cost_price_per_kg) / cost_price_per_kg) * 100
  gain_percent = 12 :=
by
  sorry

end tea_blend_gain_percent_l46_46783


namespace gigi_initial_batches_l46_46901

-- Define the conditions
def flour_per_batch := 2 
def initial_flour := 20 
def remaining_flour := 14 
def future_batches := 7

-- Prove the number of batches initially baked is 3
theorem gigi_initial_batches :
  (initial_flour - remaining_flour) / flour_per_batch = 3 :=
by
  sorry

end gigi_initial_batches_l46_46901


namespace sum_of_faces_edges_vertices_rectangular_prism_l46_46365

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l46_46365


namespace solve_for_x_l46_46053

theorem solve_for_x (x : ℝ) (hx : 0 < x) (h : 5 * real.sqrt (1 + x) + 5 * real.sqrt (1 - x) = 7 * real.sqrt 2) :
  x = 7 / 25 :=
sorry

end solve_for_x_l46_46053


namespace shopping_problem1_l46_46769

theorem shopping_problem1 (total_amount : ℕ) (hy1: total_amount = 480 + 520) : 
  if total_amount <= 500 then total_amount = 480 + 520
  else if 500 < total_amount ∧ total_amount <= 800 then total_amount * 0.8 = 760
  else 800 * 0.8 + (total_amount - 800) * 0.6 = 760 :=
by 
  have h1: total_amount = 1000 := by exact hy1
  sorry

end shopping_problem1_l46_46769


namespace probability_xi_l46_46113

noncomputable def xi_distribution (k : ℕ) : ℚ :=
  if h : k > 0 then 1 / (2 : ℚ)^k else 0

theorem probability_xi (h : ∀ k : ℕ, k > 0 → xi_distribution k = 1 / (2 : ℚ)^k) :
  (xi_distribution 3 + xi_distribution 4) = 3 / 16 :=
by
  sorry

end probability_xi_l46_46113


namespace possible_values_of_a_l46_46221

variable (A : Set ℝ) (B : Set ℝ) (C : Set ℝ) (a : ℝ)

def set_A : Set ℝ := {1, 2}
def set_B (a : ℝ) : Set ℝ := {x | x^2 - (a + 1) * x + a = 0}
def set_C (a : ℝ) : Set ℝ := set_A ∪ set_B a

theorem possible_values_of_a (h : (set_C a).toFinset.card = 4) : a = 1 ∨ a = 2 := 
by
  sorry

end possible_values_of_a_l46_46221


namespace lcm_48_180_l46_46039

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l46_46039


namespace sum_of_powers_of_two_l46_46444

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 :=
by
  sorry

end sum_of_powers_of_two_l46_46444


namespace area_of_sector_l46_46519

-- Definitions for conditions
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := 2 * r + r * θ
def sector_area (r : ℝ) (θ : ℝ) : ℝ := (1 / 2) * r^2 * θ

-- Given values for the specific problem
def given_perimeter := 6.0 -- cm
def given_angle := 1.0 -- radian
def given_area := 3.0 -- cm^2

-- Statement of the problem
theorem area_of_sector (r : ℝ) (h1 : sector_perimeter r given_angle = given_perimeter) : 
  sector_area r given_angle = given_area :=
sorry

end area_of_sector_l46_46519


namespace limit_a_limit_b_l46_46231

noncomputable def limit_integral_a (k : ℝ) (hk : k > 1) : ℝ :=
  lim (λ n : ℕ, ∫ (x : ℝ) in 0..1, (k / (x ^ (1 / n : ℝ) + k - 1)) ^ n)

theorem limit_a (k : ℝ) (hk : k > 1) : 
  limit_integral_a k hk = k / (k - 1) :=
sorry

noncomputable def limit_integral_b (k : ℝ) (hk : k > 1) : ℝ :=
  lim (λ n : ℕ, n * (k / (k - 1) - ∫ (x : ℝ) in 0..1, (k / (x ^ (1 / n : ℝ) + k - 1)) ^ n))

theorem limit_b (k : ℝ) (hk : k > 1) : 
  limit_integral_b k hk = k / (k - 1) ^ 2 :=
sorry

end limit_a_limit_b_l46_46231


namespace non_similar_triangles_arithmetic_progression_l46_46550

theorem non_similar_triangles_arithmetic_progression :
  ∃ (n d : ℕ), n = 60 ∧ d ∈ {1 .. 59} ∧ (∀ d', d' ∈ {1 .. 59} → (n - d', n + d', n) ∈ ℕ^3 ∧ n - d' > 0 ∧ n - d' < 60 ∧ n + d' < 120) :=
by
  let n := 60
  let d := 59
  use [n, d]
  split
  { refl }
  split
  { exact set.mem_range.mpr (by linarith) }
  intros d' hd'
  split
  { exact ⟨n - d', n + d', n⟩ }
  split
  { linarith }
  linarith

end non_similar_triangles_arithmetic_progression_l46_46550


namespace binomial_12_5_l46_46447

def binomial_coefficient : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binomial_12_5 : binomial_coefficient 12 5 = 792 := by
  sorry

end binomial_12_5_l46_46447


namespace terminal_zeros_of_product_l46_46553

noncomputable def prime_factors (n : ℕ) : List (ℕ × ℕ) := sorry

theorem terminal_zeros_of_product (n m : ℕ) (hn : prime_factors n = [(2, 1), (5, 2)])
 (hm : prime_factors m = [(2, 3), (3, 2), (5, 1)]) : 
  (∃ k, n * m = 10^k) ∧ k = 3 :=
by {
  sorry
}

end terminal_zeros_of_product_l46_46553


namespace number_of_people_in_group_l46_46279

-- Define the conditions as hypotheses
variables (n : ℕ) -- number of people in the group

-- The given conditions
-- Condition 1: average weight of a group increases by 2.5 kg when a new person comes in place of one weighing 20 kg.
def avg_weight_increase (n : ℕ) : Prop := (20 * n + 20 = 20 * n + 2.5 * n)

-- Condition 2: the weight of the new person is 40 kg.
def new_person_weight : Prop := 40 = 40

-- Prove the question with the correct answer
theorem number_of_people_in_group (hn : avg_weight_increase n) (hp : new_person_weight) : n = 8 :=
sorry

end number_of_people_in_group_l46_46279


namespace parallel_to_BC_l46_46399

noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2

structure Triangle :=
(A B C : ℝ → ℝ × ℝ) -- Vertices as functions of coordinates

def is_midpoint (D : ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  D = midpoint B C

def angle (x y z : ℝ × ℝ) : ℝ := sorry -- Placeholder for angle computation

def on_side (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry -- Placeholder for point on segment

def circumcircle (A B C : ℝ × ℝ) : set (ℝ × ℝ) := sorry -- Placeholder for circumcircle

def second_intersection (P : ℝ × ℝ) (circ : set (ℝ × ℝ)) (side : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry -- Placeholder for second intersection

theorem parallel_to_BC (A B C D M L K : ℝ × ℝ)
  (h_midpoint : is_midpoint D B C)
  (h_angle : angle A B M = angle D A C)
  (h_on_side_M : on_side M B C)
  (h_on_side_L : on_side L A B)
  (h_on_side_K : on_side K A C)
  (h_circ_CAM : L ∈ circumcircle A C M)
  (h_circ_BAM : K ∈ circumcircle B A M) :
  parallel_to_BC K L B C := 
by
  sorry

end parallel_to_BC_l46_46399


namespace Ah_tribe_count_l46_46264

-- Define the total number of inhabitants
variables (p : ℕ)

-- Define tribe membership predicates
variables (is_Ah : (ℕ → Prop))
variables (is_Uh : (ℕ → Prop))

-- Define the statements made by the first, second, and third person
variables (s1 s2 s3 : Prop)

-- Define the conditions
variables (C1 : s1 → p ≤ 16) (C2 : s1 → ∀ n, is_Uh n)
variables (C3 : s2 → p ≤ 17) (C4 : s2 → ∃ n, is_Ah n)
variables (C5 : s3 → p = 5) (C6 : s3 → ∃ m n o, is_Uh m ∧ is_Uh n ∧ is_Uh o)

-- Define the truth-telling and lying condition for members
variables (Ah_truth : ∀ n, is_Ah n → (∀ (stmt : Prop), stmt → stmt))
variables (Uh_lie : ∀ n, is_Uh n → (∀ (stmt : Prop), stmt → ¬stmt))

-- Define the goal to prove
theorem Ah_tribe_count : ∃ (a : ℕ), a = 15 :=
by
  have : ∃ a, a + 2 = 17 := sorry
  use 15
  sorry

end Ah_tribe_count_l46_46264


namespace crates_on_third_trip_l46_46422

variable (x : ℕ) -- Denote the number of crates carried on the third trip

-- Conditions
def crate_weight := 1250
def max_weight := 6250
def trip3_weight (x : ℕ) := x * crate_weight

-- The problem statement: Prove that x (the number of crates on the third trip) == 5
theorem crates_on_third_trip : trip3_weight x <= max_weight → x = 5 :=
by
  sorry -- No proof required, just statement

end crates_on_third_trip_l46_46422


namespace cos_triple_angle_l46_46556

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = -1 / 3) : Real.cos (3 * θ) = 23 / 27 :=
by
  sorry

end cos_triple_angle_l46_46556


namespace train_speed_l46_46423

/-- A train that is 90 meters long is traveling at a certain speed and can cross a bridge
    in 30 seconds. The bridge is 285 meters long. What is the speed of the train in km/hr? -/
theorem train_speed
    (length_of_train : ℕ)
    (time_to_cross : ℕ)
    (length_of_bridge : ℕ)
    (conversion_factor : ℕ → ℝ)
    (speed_of_train : ℕ → ℝ) :
    length_of_train = 90 → 
    time_to_cross = 30 → 
    length_of_bridge = 285 → 
    conversion_factor 1 = 3.6 → 
    speed_of_train 1 = 12.5 →
    speed_of_train time_to_cross * (conversion_factor 1) = 45 := 
by
  intros h_train h_time h_bridge h_conversion h_speed
  rw [h_train, h_time, h_bridge, h_conversion, h_speed]
  sorry

end train_speed_l46_46423


namespace find_v_value_l46_46061

theorem find_v_value : 
  (∃ v : ℝ, ∃ x : ℝ, (x = (-25 - real.sqrt 361) / 12) ∧ (6 * x^2 + 25 * x + v = 0)) ↔ (v = 11) :=
begin
  sorry
end

end find_v_value_l46_46061


namespace smallest_positive_period_eq_pi_intervals_of_monotonic_decrease_max_min_values_in_interval_l46_46066

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6) + 1

theorem smallest_positive_period_eq_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem intervals_of_monotonic_decrease :
  ∃ (k : ℤ), ∀ x, f (x) < f (x + 1) ↔ k * π + π / 6 ≤ x ∧ x ≤ 2 * π / 3 + k * π :=
sorry

theorem max_min_values_in_interval :
  ∃ (max min : ℝ), max = f (π / 4) ∧ min = f (5 * π / 12) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π / 2 → min ≤ f x ∧ f x ≤ max) :=
sorry

end smallest_positive_period_eq_pi_intervals_of_monotonic_decrease_max_min_values_in_interval_l46_46066


namespace sum_faces_edges_vertices_l46_46357

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l46_46357


namespace hyperbola_eccentricity_l46_46106

theorem hyperbola_eccentricity (a b c : ℝ) (F1 F2 M : Point)
  (hyp1 : 0 < a) (hyp2 : 0 < b)
  (hyp3 : hyperbola_eq F1 F2 a b) 
  (hyp4 : regular_triangle MF1F2 M F1 F2) 
  (hyp5 : midpoint MF1_on_hyperbola M F1) :
  eccentricity a b = √3 + 1 :=
sorry

end hyperbola_eccentricity_l46_46106


namespace eval_expression_l46_46749

-- We define the expression that needs to be evaluated
def expression := (0.76)^3 - (0.1)^3 / (0.76)^2 + 0.076 + (0.1)^2

-- The statement to prove
theorem eval_expression : expression = 0.5232443982683983 :=
by
  sorry

end eval_expression_l46_46749


namespace pens_sales_consistency_books_left_indeterminate_l46_46251

-- The initial conditions
def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_left : ℕ := 19
def pens_sold : ℕ := 23

-- Prove the consistency of the number of pens sold
theorem pens_sales_consistency : initial_pens - pens_left = pens_sold := by
  sorry

-- Assert that the number of books left is indeterminate based on provided conditions
theorem books_left_indeterminate : ∃ b_left : ℕ, b_left ≤ initial_books ∧
    ∀ n_books_sold : ℕ, n_books_sold > 0 → b_left = initial_books - n_books_sold := by
  sorry

end pens_sales_consistency_books_left_indeterminate_l46_46251


namespace subsets_count_l46_46119

noncomputable def universal_set : Set ℝ := Set.univ

def set_A : Set ℤ := { -2, -1, 0, 1, 2 }

def set_B : Set ℝ := { x | (x - 1) / (x + 2) < 0 }

def complement_B_in_R : Set ℝ := { x | x ≤ -2 ∨ x ≥ 1 }

def set_intersection : Set ℤ := { x ∈ set_A | x ≤ -2 ∨ x ≥ 1 }

theorem subsets_count :
  (set_intersection = { -2, 1, 2 }) → (2^set_intersection.finite_to_set.toFinset.card = 8) :=
by
  sorry

end subsets_count_l46_46119


namespace find_zero_point_l46_46683

noncomputable def zero_point_of_function : Prop :=
  ∃ x : ℝ, 0 < x ∧ 2 * x + 1 > 0 ∧ (log (2 * x + 1) + log x = 0) ∧ x = 1 / 2

theorem find_zero_point : zero_point_of_function :=
  by {
    sorry
  }

end find_zero_point_l46_46683


namespace rectangle_area_l46_46786

theorem rectangle_area (x : ℝ) (w : ℝ) 
  (h1 : x^2 = 10 * w^2) 
  (h2 : ∃ l, l = 3 * w) : 
  ∃ A, A = 3 * (x^2 / 10) :=
by {
  existsi (3 * (x^2 / 10)),
  sorry
}

end rectangle_area_l46_46786


namespace max_gcd_sequence_l46_46299

theorem max_gcd_sequence :
  ∃ n : ℕ, ∀ n > 0, ∃ d_n, (∀ m > 0, d_n = gcd (150 + n^2) (150 + (n+1)^2)) ∧ d_n ≤ 601 ∧ 601 ∣ d_n :=
begin
  sorry
end

end max_gcd_sequence_l46_46299


namespace intersection_point_l46_46777

noncomputable def point_of_intersection_exists (x y t u: ℝ) : Prop :=
  ∃ (t u : ℝ),
    (x = 2 + 3 * t) ∧
    (y = 2 - 4 * t) ∧
    (x = 4 + 5 * u) ∧
    (y = -8 + 3 * u)

theorem intersection_point:
  point_of_intersection_exists (-123/141) (454/141) sorry := 
sorry

end intersection_point_l46_46777


namespace student_chose_number_l46_46737

theorem student_chose_number :
  ∃ x : ℕ, 7 * x - 150 = 130 ∧ x = 40 := sorry

end student_chose_number_l46_46737


namespace expression_evaluation_l46_46455

theorem expression_evaluation :
  (Real.cbrt (-27) - (- (1 / 2)) ^ (-2) - 4 * Real.cos (Float.pi / 3) + abs (Real.sqrt 3 - 2) = -8.732) :=
by
  apply congr_arg _
  sorry

end expression_evaluation_l46_46455


namespace problem_statement_l46_46611

open Function

theorem problem_statement :
  ∃ g : ℝ → ℝ, 
    (g 1 = 2) ∧ 
    (∀ (x y : ℝ), g (x^2 - y^2) = (x - y) * (g x + g y)) ∧ 
    (g 3 = 6) := 
by
  sorry

end problem_statement_l46_46611


namespace customer_paid_amount_l46_46301

theorem customer_paid_amount (cost_price : ℝ) (markup_percentage : ℝ) (markup_amount : ℝ) (total_price : ℝ) :
  cost_price = 7166.67 → markup_percentage = 0.20 → markup_amount = cost_price * markup_percentage → 
  total_price = cost_price + markup_amount → total_price = 8600 := 
by
  intros h_cost_price h_markup_percentage h_markup_amount h_total_price
  rw [h_cost_price] at *
  rw [h_markup_percentage] at *
  change cost_price * 0.20 with 1433.33 at h_markup_amount  -- rounded amount manually given in solution
  rw [h_markup_amount] at h_total_price
  rw [h_total_price]
  norm_num
  sorry

end customer_paid_amount_l46_46301


namespace length_AF_l46_46416

def CE : ℝ := 40
def ED : ℝ := 50
def AE : ℝ := 120
def area_ABCD : ℝ := 7200

theorem length_AF (AF : ℝ) :
  CE = 40 → ED = 50 → AE = 120 → area_ABCD = 7200 →
  AF = 128 :=
by
  intros hCe hEd hAe hArea
  sorry

end length_AF_l46_46416


namespace intersection_of_rect_diag_on_square_diag_l46_46814

-- Definitions and assumptions
variable {A B C D B1 C1 D1 O : Point}
variable (square ABCD : Quadrilateral)
variable (rectangle AB1C1D1 : Quadrilateral)

-- Conditions
variable (perimeters_equal : perimeter ABCD = perimeter AB1C1D1)
variable (shared_vertex : A = A B B1 A C C1 A D D1)
variable (O_eq_int_rect_diag : O = midpoint (diagonal B1 D1))

-- Assertion to be proved
theorem intersection_of_rect_diag_on_square_diag :
  lies_on_diagonal O (diagonal B D) :=
by sorry

end intersection_of_rect_diag_on_square_diag_l46_46814


namespace sum_S20_S35_l46_46292

noncomputable def a_n (n : ℕ) : ℤ := (-1)^n * (3 * n - 2)

noncomputable def S_n (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), a_n i

theorem sum_S20_S35 :
  S_n 20 + S_n 35 = -22 :=
sorry

end sum_S20_S35_l46_46292


namespace major_axis_length_l46_46126

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by sorry

end major_axis_length_l46_46126


namespace intersection_lies_on_semi_circle_l46_46904

/-- Given \( \triangle ABC \) with \( \angle C = 90^\circ \), point \( H \) is the projection of \( C \) onto \( AB \).
Point \( D \) is inside \( \triangle CBH \) such that \( CH \) bisects segment \( AD \). Line \( BD \) intersects \( CH \)
at point \( P \), and \( \Gamma \) is a semicircle with \( BD \) as its diameter, intersecting \( BC \) at a point other
than \( B \). A line passing through \( P \) is tangent to the semicircle \( \Gamma \) at point \( Q \). Prove that the
intersection of line \( CQ \) and \( AD \) lies on semicircle \( \Gamma \). -/
theorem intersection_lies_on_semi_circle
  (ABC : Triangle)
  (w1 : ∠ ABC.C = 90)
  (H : Point) (H_Proj : foot_projection ABC.C ABC.ABC)
  (D : Point) (D_In_Triangle : inside_triangle D ABC.C ABC.B ABC.H)
  (CH_Bisects_AD : Bisects ABC.H D ABC.A)
  (P : Point) (P_Intersect : intersection (line_through ABC.B D) P
    (line_through ABC.C H))
  (Gamma : Semicircle)
  (Gamma_Diameter : diameter Gamma (segment_through ABC.B D))
  (Gamma_Intersect_Cond : not_eq (intersection Gamma ABC.C D) ABC.B)
  (Q : Point) (Q_Tangent : tangent_point_through P Gamma Q)
  : lies_on (intersection (line_through ABC.C Q) (line_through ABC.A D)) Gamma := sorry

end intersection_lies_on_semi_circle_l46_46904


namespace lcm_48_180_eq_720_l46_46047

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l46_46047


namespace quadratic_real_roots_condition_l46_46488

open Real

theorem quadratic_real_roots_condition (k : ℝ) (h : k ≠ 0) :
  (let Δ := (-4)^2 - 4 * k * 2 in Δ ≥ 0) ↔ (k ≤ 2) :=
by
  -- We need to compute discriminant Δ and show this condition is equivalent to k ≤ 2
  let Δ := (-4)^2 - 4 * k * 2
  have : Δ = 16 - 8 * k,
  sorry
  -- Prove the inequality
  show Δ ≥ 0 ↔ k ≤ 2,
  sorry

end quadratic_real_roots_condition_l46_46488


namespace max_chains_upper_bound_l46_46606

noncomputable def C (n k : ℕ) : ℕ :=
  Nat.choose n k

def f (n k : ℕ) : ℕ := sorry -- Placeholder for the actual definition of f

theorem max_chains_upper_bound (n k : ℕ) : f(n, k) ≤ C (n - k) (Nat.floor n - k / 2) := sorry

end max_chains_upper_bound_l46_46606
