import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Calc
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Combinatorics.CombinatorialDesigns.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Digits
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Comb
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Euclidean.Incircle
import Mathlib.Geometry.Euclidean.Mutable3
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Algebra.Order
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.NumberTheory.PrimeNorEuclid
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import data.graph.basic
import data.nat.basic
import data.set.basic

namespace Canada_moose_population_l41_41460

theorem Canada_moose_population (moose beavers humans : ℕ) (h1 : beavers = 2 * moose) 
                              (h2 : humans = 19 * beavers) (h3 : humans = 38 * 10^6) : 
                              moose = 1 * 10^6 :=
by
  sorry

end Canada_moose_population_l41_41460


namespace stating_opposite_sides_parallel_congruent_l41_41830

variable (P : Polygon) (n : ℕ)

/-- 
  A polygon type and its properties defined for a convex n-gon.
  Here, we are specifically interested in a 2006-gon.
-/
structure Polygon :=
  (sides : ℕ)
  (is_convex : Prop)

/--
  Theorem stating that if a convex 2006-gon has diagonals 
  and midline segments that are concurrent, opposite 
  sides must be parallel and congruent.
-/
theorem opposite_sides_parallel_congruent 
  (P : Polygon)
  (h_sides : P.sides = 2006)
  (h_convex : P.is_convex)
  (h_concurrent : ∃ point, 
    (∀ (i : ℕ) (h_i : i < 1003), concurrent_diagonals_and_midlines P i point)) :
    (∀ (side1 side2 : ℕ) (h1 : side1 < 1003) (h2 : side2 = side1 + 1003 % 2006),
      parallel_and_congruent_sides P side1 side2) := sorry

-- Definitions for concurrent diagonals and midlines
def concurrent_diagonals_and_midlines (P : Polygon) 
  (i : ℕ)
  (point : Point) : Prop := sorry

def parallel_and_congruent_sides (P : Polygon) 
  (side1 side2 : ℕ) : Prop := sorry

-- defining a basic point as placeholder
structure Point :=
  (x : ℝ)
  (y : ℝ)

end stating_opposite_sides_parallel_congruent_l41_41830


namespace number_of_distinct_positive_differences_l41_41418

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41418


namespace cards_from_around_country_l41_41589

-- Define the total number of cards and the number from home
def total_cards : ℝ := 403.0
def home_cards : ℝ := 287.0

-- Define the expected number of cards from around the country
def expected_country_cards : ℝ := 116.0

-- Theorem statement
theorem cards_from_around_country :
  total_cards - home_cards = expected_country_cards :=
by
  -- Since this only requires the statement, the proof is omitted
  sorry

end cards_from_around_country_l41_41589


namespace probability_log_floor_equal_l41_41495

open Real

theorem probability_log_floor_equal (x : ℝ) (hx : 0 < x ∧ x < 1) :
  ∃ p : ℝ, p = 8 / 9 ∧ (probability (λ x, ⌊ 5 * log10 x ⌋
                              - ⌊ log10 x ⌋ = 0) x) = p :=
sorry

end probability_log_floor_equal_l41_41495


namespace polynomial_p_solution_l41_41294

theorem polynomial_p_solution (x : ℝ) :
  ∀ p : polynomial ℝ,
    (4 * x^4 + 7 * x^3 - 2 * x + 5 + p.eval x = -3 * x^4 + 2 * x^3 - 8 * x^2 + 6 * x - 4) →
    (p = polynomial.C (-7) * polynomial.X^4 - polynomial.C (5) * polynomial.X^3 - polynomial.C (8) * polynomial.X^2 + polynomial.C (8) * polynomial.X - polynomial.C (9)) :=
by
  intros p h
  sorry

end polynomial_p_solution_l41_41294


namespace evaluate_complex_expression_l41_41687

theorem evaluate_complex_expression (i : ℂ) (hi2 : i^2 = -1) (hi4 : i^4 = 1) : 
  i^10 + i^20 + i^(-30) = -1 := 
by {
  sorry
}

end evaluate_complex_expression_l41_41687


namespace euclidean_triangle_property_l41_41056

open EuclideanGeometry

noncomputable def math_problem (A B C D : Point) : Prop :=
  ∠BAC = 90 ∧
  D ∈ LineSegment B C ∧
  ∠BDA = 2 * ∠BAD →
  2 / dist A D = 1 / dist B D + 1 / dist C D

theorem euclidean_triangle_property {A B C D : Point} :
  math_problem A B C D :=
begin
  sorry
end

end euclidean_triangle_property_l41_41056


namespace differences_of_set_l41_41383

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41383


namespace num_ordered_pairs_of_squares_diff_by_144_l41_41424

theorem num_ordered_pairs_of_squares_diff_by_144 :
  ∃ (p : Finset (ℕ × ℕ)), p.card = 4 ∧ ∀ (a b : ℕ), (a, b) ∈ p → a ≥ b ∧ a^2 - b^2 = 144 := by
  sorry

end num_ordered_pairs_of_squares_diff_by_144_l41_41424


namespace train_crosses_platform_in_20s_l41_41245

noncomputable def timeToCrossPlatform (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

theorem train_crosses_platform_in_20s :
  timeToCrossPlatform 120 213.36 60 = 20 :=
by
  sorry

end train_crosses_platform_in_20s_l41_41245


namespace max_value_and_sum_l41_41496

-- Definitions of the conditions
def pos_reals : Prop := ∀ x y z v w : ℝ, (0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < v ∧ 0 < w)
def condition (x y z v w : ℝ) : Prop := x^2 + y^2 + z^2 + v^2 + w^2 = 1024

-- The main theorem statement
theorem max_value_and_sum (x y z v w : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < v) (h5 : 0 < w)
  (h : condition x y z v w) :
  let M := x * z + 3 * y * z + 4 * z * v + 6 * z * w + 2 * x * w in
  (M = 262144) ∧ (x = 2 * Real.sqrt 2) ∧ (y = 6 * Real.sqrt 2) ∧ (z = 16 * Real.sqrt 2) ∧
  (v = 8 * Real.sqrt 2) ∧ (w = 12 * Real.sqrt 2) →
  M + x + y + z + v + w = 262144 + 44 * Real.sqrt 2 := sorry

end max_value_and_sum_l41_41496


namespace min_surface_area_base_edge_length_l41_41744

noncomputable def min_base_edge_length (V : ℝ) : ℝ :=
  2 * (V / (2 * Real.pi))^(1/3)

theorem min_surface_area_base_edge_length (V : ℝ) : 
  min_base_edge_length V = (4 * V)^(1/3) :=
by
  sorry

end min_surface_area_base_edge_length_l41_41744


namespace hens_eggs_count_l41_41284

noncomputable def total_eggs : ℝ := 303.0
noncomputable def number_of_hens : ℝ := 28.0

theorem hens_eggs_count : int.nearest (total_eggs / number_of_hens) = 11 := 
by
  sorry

end hens_eggs_count_l41_41284


namespace num_tangent_lines_l41_41065

theorem num_tangent_lines (P : ℝ × ℝ) (circle_eq : ℝ → ℝ → ℝ) : 
  P = (2, 1) → circle_eq = (λ x y, x^2 - x + y^2 + 2y - 4) → 
  (∃ (n : ℕ), n = 2) :=
sorry

end num_tangent_lines_l41_41065


namespace find_value_of_expression_l41_41031

theorem find_value_of_expression 
(h : ∀ (a b : ℝ), a * (3:ℝ)^2 - b * (3:ℝ) = 6) : 
  ∀ (a b : ℝ), 2023 - 6 * a + 2 * b = 2019 := 
by
  intro a b
  have h1 : 9 * a - 3 * b = 6 := by sorry
  have h2 : 3 * a - b = 2 := by sorry
  have result := 2023 - 2 * (3 * a - b)
  rw h2 at result
  exact result

end find_value_of_expression_l41_41031


namespace trains_crossing_time_opposite_l41_41605

noncomputable def train_crossing_time_opposite : ℝ :=
  let speed1 := 60 * (5 / 18) -- speed of the first train in m/s
  let speed2 := 40 * (5 / 18) -- speed of the second train in m/s
  let relative_speed_same := speed1 - speed2 -- relative speed in the same direction
  let crossing_time_same := 55 -- time to cross each other in the same direction in seconds
  let distance := relative_speed_same * crossing_time_same -- distance covered when crossing in the same direction
  let train_length := distance / 2 -- each train length in meters
  let relative_speed_opposite := speed1 + speed2 -- relative speed in opposite directions
  distance / relative_speed_opposite -- time to cross each other in opposite direction in seconds

theorem trains_crossing_time_opposite :
  train_crossing_time_opposite ≈ 11.01 :=
sorry  -- Proof will use approximate evaluation

end trains_crossing_time_opposite_l41_41605


namespace A_share_of_gain_l41_41202

-- Given problem conditions
def investment_A (x : ℝ) : ℝ := x * 12
def investment_B (x : ℝ) : ℝ := 2 * x * 6
def investment_C (x : ℝ) : ℝ := 3 * x * 4
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def total_gain : ℝ := 21000

-- Mathematically equivalent proof problem statement
theorem A_share_of_gain (x : ℝ) : (investment_A x) / (total_investment x) * total_gain = 7000 :=
by
  sorry

end A_share_of_gain_l41_41202


namespace solve_inequality_l41_41532

theorem solve_inequality 
  (x : ℝ) 
  (h1 : (3 * x - 6) / (3 * x + 5) > 0)
  (h2 : (3 * x - 6) / (3 * x + 5) ≠ 1)
  (h3 : (3 * x - 9)^10 > 0)
  (h4 : 3 * x - 6 ≠ 0)
  (h5 : 3 * x + 5 ≠ 0)
  (h6 : 3 * x + 6 > 0) : 
  x ∈ Ioo (-2 : ℝ) (-5/3) ∪ Ioo 2 3 ∪ Ioi 3 :=
sorry

end solve_inequality_l41_41532


namespace division_sequence_l41_41534

def next_division (n : ℕ) : ℕ := n / 3

noncomputable def perform_operations (initial : ℕ) (times : ℕ) : ℕ :=
  Nat.iterate next_division times initial

theorem division_sequence :
  perform_operations 250 5 ≤ 2 :=
by {
  unfold perform_operations,
  simp [Nat.iterate],
  norm_num,
  sorry
}

end division_sequence_l41_41534


namespace midpoint_of_AB_l41_41476

-- Defining the complex numbers corresponding to points A and B
def A : ℂ := -3 + 2 * complex.i
def B : ℂ := 1 - 4 * complex.i

-- Proving that the midpoint of A and B is -1 - i
theorem midpoint_of_AB : (A + B) / 2 = -1 - complex.i := by
  sorry

end midpoint_of_AB_l41_41476


namespace calculate_decimal_l41_41655

theorem calculate_decimal : 3.59 + 2.4 - 1.67 = 4.32 := 
  by
  sorry

end calculate_decimal_l41_41655


namespace find_integer_mod_l41_41296

theorem find_integer_mod (n : ℤ) (h₁ : 0 ≤ n) (h₂ : n ≤ 7) (h₃ : n ≡ -825 [MOD 8]) : n = 7 :=
by
  sorry

end find_integer_mod_l41_41296


namespace product_ab_eq_29_l41_41299

noncomputable def a : ℂ := 5 - 2 * complex.i
noncomputable def b : ℂ := 5 + 2 * complex.i

theorem product_ab_eq_29 : a * b = 29 := 
by 
sorry

end product_ab_eq_29_l41_41299


namespace interior_angle_bisectors_parallel_l41_41078

theorem interior_angle_bisectors_parallel
  (A B C D E M N P : Point)
  [is_acute_triangle A B C]
  (hD : midpoint D B C)
  (hE : E ∈ segment A D)
  (hM : orthogonal_projection E M (segment B C))
  (hN : orthogonal_projection M N (segment A B))
  (hP : orthogonal_projection M P (segment A C)) :
  is_parallel (angle_bisector (∠ N M P)) (angle_bisector (∠ N E P)) := 
sorry

end interior_angle_bisectors_parallel_l41_41078


namespace tony_rollercoasters_l41_41575

theorem tony_rollercoasters :
  let s1 := 50 -- speed of the first rollercoaster
  let s2 := 62 -- speed of the second rollercoaster
  let s3 := 73 -- speed of the third rollercoaster
  let s4 := 70 -- speed of the fourth rollercoaster
  let s5 := 40 -- speed of the fifth rollercoaster
  let avg_speed := 59 -- Tony's average speed during the day
  let total_speed := s1 + s2 + s3 + s4 + s5
  total_speed / avg_speed = 5 := sorry

end tony_rollercoasters_l41_41575


namespace sum_of_x_values_l41_41192

theorem sum_of_x_values (x : ℝ) :
  (∃ x, 6 = (x^4 - 3 * x^3 - 16 * x^2 + 24 * x) / (x - 3)) →
  x^3 - 13 * x^2 + 40 * x - 6 = 0 →
  sum_of_roots x = 13 :=
by sorry

end sum_of_x_values_l41_41192


namespace Problem1_Problem2_Problem3_l41_41347

noncomputable def f (a b x : ℝ) := x^2 + (3 - a) * x + 2 + 2 * a + b

theorem Problem1 (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x < -4 ∨ x > 2) → a = 1 ∧ b = -12 :=
sorry

theorem Problem2 (a b : ℝ) :
  (∀ x ∈ set.Icc 1 3, f a b x ≤ b) → a ≤ -6 ∨ a ≥ 20 :=
sorry

theorem Problem3 (a b : ℝ) :
  (∀ x ∈ ℤ, f a b x < 12 + b ↔ x = 3 ∨ x = 4 ∨ x = 5) → (3 ≤ a ∧ a < 4) ∨ (10 < a ∧ a ≤ 11) :=
sorry

end Problem1_Problem2_Problem3_l41_41347


namespace eccentricity_ellipse_l41_41730

theorem eccentricity_ellipse (m n : ℝ) (h1 : 2 * n = m + (m + n)) (h2 : n^2 = m * (m * n)) :
  let e := (Real.sqrt 2) / 2 in
  eccentricity (Ellipses.mk m n) = e :=
by 
  sorry

end eccentricity_ellipse_l41_41730


namespace arithmetic_sequence_limit_l41_41564

theorem arithmetic_sequence_limit :
  let a : ℕ → ℝ := λ n, 1 + (n - 1) * 2,
      S : ℕ → ℝ := λ n, ∑ i in finset.range n, a (i + 1)
  in
  (∀ n : ℕ, S n = n^2) →
  (∀ n : ℕ, a n = 2 * n - 1) →
  ∃ l : ℝ, tendsto (λ n, S n / (a n)^2) at_top (𝓝 l) ∧ l = 1 / 4 :=
by
  intro a S hS ha
  use 1 / 4
  sorry

end arithmetic_sequence_limit_l41_41564


namespace emails_in_the_morning_l41_41822

theorem emails_in_the_morning : ∃ M : ℕ, M + 8 = 13 ∧ M = 5 :=
by
  existsi 5
  split
  { exact rfl }
  { exact rfl }

end emails_in_the_morning_l41_41822


namespace general_term_b_sum_first_2015_terms_l41_41318

-- Definitions of sequences
def a (n : ℕ) : ℝ := 3^(n-1)

def b (n : ℕ) : ℝ := (2 * n - 1) * 3^(n-1)

-- Condition
axiom sequence_condition : ∀ n : ℕ, (∑ i in finset.range n, b (i + 1) / a (i + 1)) = n^2

-- Proof Problem 1: General term formula for b_n
theorem general_term_b (n : ℕ) : b n = (2 * n - 1) * 3^(n-1) :=
sorry

-- Proof Problem 2: Sum of the first 2015 terms of b_n
theorem sum_first_2015_terms : (∑ i in finset.range 2015, b (i + 1)) = 2014 * 3^(2015) + 1 :=
sorry

end general_term_b_sum_first_2015_terms_l41_41318


namespace number_of_differences_l41_41385

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41385


namespace digit_sum_identity_l41_41442

theorem digit_sum_identity (n : ℕ) (d : ℕ) (s : ℕ) (n_k : ℕ → ℕ)
  (h_digit_len : d + 1 = (nat.log 10 n).to_nat + 1)
  (h_sum_digits : s = (nat.digits 10 n).sum)
  (h_n_k : ∀ k, n_k k = (n / 10^k)) :
  n = s + ∑ k in finset.range d, 9 * n_k (k + 1) :=
sorry

end digit_sum_identity_l41_41442


namespace circle_center_and_radius_l41_41541

noncomputable def circle_eq : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 + 2 * x - 2 * y - 2 = 0) ↔ (x + 1)^2 + (y - 1)^2 = 4

theorem circle_center_and_radius :
  ∃ center : ℝ × ℝ, ∃ r : ℝ, 
  center = (-1, 1) ∧ r = 2 ∧ circle_eq :=
by
  sorry

end circle_center_and_radius_l41_41541


namespace gold_cube_profit_multiple_l41_41109

theorem gold_cube_profit_multiple :
  let side_length : ℝ := 6
  let density : ℝ := 19
  let cost_per_gram : ℝ := 60
  let profit : ℝ := 123120
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * cost_per_gram
  let selling_price := cost + profit
  let multiple := selling_price / cost
  multiple = 1.5 := by
  sorry

end gold_cube_profit_multiple_l41_41109


namespace real_b_values_for_non_real_roots_l41_41443

theorem real_b_values_for_non_real_roots (b : ℝ) :
  let discriminant := b^2 - 4 * 1 * 16
  discriminant < 0 ↔ -8 < b ∧ b < 8 := 
sorry

end real_b_values_for_non_real_roots_l41_41443


namespace six_digit_numbers_with_zero_l41_41006

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41006


namespace project_completion_time_l41_41221

theorem project_completion_time (x : ℕ) :
  (∀ (B_days : ℕ), B_days = 40 →
  (∀ (combined_work_days : ℕ), combined_work_days = 10 →
  (∀ (total_days : ℕ), total_days = 20 →
  10 * (1 / (x : ℚ) + 1 / 40) + 10 * (1 / 40) = 1))) →
  x = 20 :=
by
  sorry

end project_completion_time_l41_41221


namespace moose_population_l41_41463

theorem moose_population (B M H : ℕ) (h1 : B = 2 * M) (h2 : H = 19 * B) (h3 : H = 38_000_000) : M = 1_000_000 :=
by sorry

end moose_population_l41_41463


namespace cube_root_of_minus_two_l41_41949

theorem cube_root_of_minus_two : ∃ x : ℝ, x^3 = -8 ∧ real.cube_root x = -2 :=
by
  sorry

end cube_root_of_minus_two_l41_41949


namespace series_sum_subtract_eight_l41_41272

theorem series_sum_subtract_eight :
  (\frac{5}{3} + \frac{13}{9} + \frac{41}{27} + \frac{125}{81} + \frac{379}{243} + \frac{1145}{729}) - 8 = \frac{950}{729} :=
by
  sorry

end series_sum_subtract_eight_l41_41272


namespace min_value_quadratic_l41_41907

theorem min_value_quadratic : ∃ x : ℝ, (∀ y : ℝ, y = x^2 + 6 * x + 9) → (x = -3) ∧ (y = 0) :=
begin
  sorry
end

end min_value_quadratic_l41_41907


namespace differences_of_set_l41_41376

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41376


namespace angle_EMF_is_90_degrees_l41_41142

section CircleGeometry

variables {A B C D E F K M N : Point}
variables {S S' : Circle}

-- Conditions
variables (h1 : S ∩ S' = {A, B})
variables (h2 : Line_through A ∩ S = C)
variables (h3 : Line_through A ∩ S' = D)
variables (h4 : M ∈ Line C D)
variables (h5 : Parallel (Line_through M) (Line_through B C))
variables (h6 : Intersection (Line_through M Line_through B C) (Line BD) = K)
variables (h7 : Parallel (Line_through M) (Line_through B D))
variables (h8 : Intersection (Line_through M Line_through B D) (Line BC) = N)
variables (h9 : Perpendicular_to (Line_through N) (Line_through B C))
variables (h10 : Intersection (Line BC) S = E)
variables (h11 : E.opposite_side_of B C A)
variables (h12 : Perpendicular_to (Line_through K) (Line_through B D))
variables (h13 : Intersection (Line BD) S' = F)
variables (h14 : F.opposite_side_of B D A)

-- Goal
theorem angle_EMF_is_90_degrees : ∠ E M F = 90 := by
  sorry

end CircleGeometry

end angle_EMF_is_90_degrees_l41_41142


namespace irrational_count_l41_41994

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (x = a / b)

theorem irrational_count {a b c d e f : ℝ} :
  a = 1 / 13 →
  b = -2 →
  c = 4 →
  d = 22 / 7 →
  e = real.pi →
  (∀ q r : ℤ, r ≠ 0 → f ≠ q / r) →
  list.countp is_irrational [a, b, c, d, e, f] = 1 :=
by
  intros ha hb hc hd he hf
  simp [is_irrational, ha, hb, hc, hd, he, hf]
  sorry

end irrational_count_l41_41994


namespace min_distance_square_l41_41801

theorem min_distance_square : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  (y₁ = x₁^2 - real.log x₁) ∧ 
  (x₂ - y₂ = 2) ∧ 
  ((x₂ - x₁)^2 + (y₂ - y₁)^2 = 2) :=
sorry

end min_distance_square_l41_41801


namespace total_books_pyramid_l41_41255

def num_books (n : ℕ) : ℕ :=
  if n = 4 then 64
  else if n = 3 then 64 / 0.8
  else if n = 2 then (64 / 0.8) / 0.8
  else if n = 1 then ((64 / 0.8) / 0.8) / 0.8
  else 0

theorem total_books_pyramid :
  num_books 4 + num_books 3 + num_books 2 + num_books 1 = 369 :=
by sorry

end total_books_pyramid_l41_41255


namespace arithmetic_sequence_statements_l41_41087

theorem arithmetic_sequence_statements (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : ∀ n, S n = (n : ℚ) * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : S 5 < S 6)
  (h3 : S 6 = S 7)
  (h4 : S 7 > S 8) :
  (a 7 = 0) ∧ (a 7 - a 6 < 0) ∧ (S 6 = S 7 > S 8) :=
sorry

end arithmetic_sequence_statements_l41_41087


namespace hyperbola_center_l41_41619

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 3) (h3 : x2 = 10) (h4 : y2 = 7) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (8, 5) :=
by
  rw [h1, h2, h3, h4]
  simp
  -- Proof steps demonstrating the calculation
  -- simplify the arithmetic expressions
  sorry

end hyperbola_center_l41_41619


namespace GFH_is_right_angle_l41_41816

noncomputable def triangle_proof : Prop :=
  ∀ (A B C F G H : Type)
    [is_triangle A B C]
    [angle_eq A B C (120 : ℝ)],
    angle_bisector A F B C ∧ angle_bisector B G C A ∧ angle_bisector C H A B 
    → angle_eq G F H (90 : ℝ)

theorem GFH_is_right_angle : triangle_proof :=
by sorry

end GFH_is_right_angle_l41_41816


namespace ajax_weight_after_two_weeks_l41_41251

/-- Initial weight of Ajax in kilograms. -/
def initial_weight_kg : ℝ := 80

/-- Conversion factor from kilograms to pounds. -/
def kg_to_pounds : ℝ := 2.2

/-- Weight lost per hour of each exercise type. -/
def high_intensity_loss_per_hour : ℝ := 4
def moderate_intensity_loss_per_hour : ℝ := 2.5
def low_intensity_loss_per_hour : ℝ := 1.5

/-- Ajax's weekly exercise routine. -/
def weekly_high_intensity_hours : ℝ := 1 * 3 + 1.5 * 1
def weekly_moderate_intensity_hours : ℝ := 0.5 * 5
def weekly_low_intensity_hours : ℝ := 1 * 2 + 0.5 * 1

/-- Calculate the total weight loss in pounds per week. -/
def total_weekly_weight_loss_pounds : ℝ :=
  weekly_high_intensity_hours * high_intensity_loss_per_hour +
  weekly_moderate_intensity_hours * moderate_intensity_loss_per_hour +
  weekly_low_intensity_hours * low_intensity_loss_per_hour

/-- Calculate the total weight loss in pounds for two weeks. -/
def total_weight_loss_pounds_for_two_weeks : ℝ :=
  total_weekly_weight_loss_pounds * 2

/-- Calculate Ajax's initial weight in pounds. -/
def initial_weight_pounds : ℝ :=
  initial_weight_kg * kg_to_pounds

/-- Calculate Ajax's new weight after two weeks. -/
def new_weight_pounds : ℝ :=
  initial_weight_pounds - total_weight_loss_pounds_for_two_weeks

/-- Prove that Ajax's new weight in pounds is 120 after following the workout schedule for two weeks. -/
theorem ajax_weight_after_two_weeks :
  new_weight_pounds = 120 :=
by
  sorry

end ajax_weight_after_two_weeks_l41_41251


namespace herd_total_cows_l41_41232

theorem herd_total_cows (n : ℕ) : 
  let first_son := 1 / 3 * n
  let second_son := 1 / 6 * n
  let third_son := 1 / 8 * n
  let remaining := n - (first_son + second_son + third_son)
  remaining = 9 ↔ n = 24 := 
by
  -- Skipping proof, placeholder
  sorry

end herd_total_cows_l41_41232


namespace jim_selling_price_l41_41074

theorem jim_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) :
  cost_price = 555.56 →
  profit_percentage = 35 →
  selling_price = (cost_price + (profit_percentage / 100) * cost_price) →
  selling_price = 750.01 :=
by
  intros h_cost h_profit h_selling
  rw [h_cost, h_profit, h_selling]
  norm_num
  sorry

end jim_selling_price_l41_41074


namespace cross_sectional_area_ratio_l41_41543

-- Definitions translated from the problem condition
def is_cube (A B C D A1 B1 C1 D1 : Type) : Prop := sorry
def cross_sectional_area (BD1 : Type) : ℝ := sorry
def S_max (area: ℝ) : ℝ := sorry
def S_min (area: ℝ) : ℝ := sorry

theorem cross_sectional_area_ratio (A B C D A1 B1 C1 D1 : Type) 
  (h_cube: is_cube A B C D A1 B1 C1 D1)
  (S : ℝ)
  (h_S: S = cross_sectional_area BD1)
  (h_S_max: S_max S)
  (h_S_min: S_min S) :
  S / S = (2 * real.sqrt 3) / 3 :=
sorry

end cross_sectional_area_ratio_l41_41543


namespace U1_l41_41323

-- Define the elements and postulates
def rib (V : Type) := Set V
def taa (V : Type) := V

-- Given postulates
variables {V : Type} [Fintype V] (ribs : Finset (rib V)) (Q1 : ∀ r ∈ ribs, ∃ t : Finset V, is_col (t : rib V) r)
variables (Q2 : ∀ {r1 r2 : rib V} (h1 : r1 ∈ ribs) (h2 : r2 ∈ ribs), r1 ≠ r2 → ∃! t : Finset V, t ∈ r1 ∩ r2 ∧ Set.card t = 2)
variables (Q3 : ∀ t : V, ∃! r : Finset (rib V), t ∈ r ∧ (⟨t, r⟩ : taa V).card = 3)
variables (Q4 : Set.card ribs = 6)

-- Theorem U1 to be proven
theorem U1 : Set.card V = 15 := sorry

end U1_l41_41323


namespace number_of_distinct_positive_differences_l41_41421

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41421


namespace KM_perp_KD_l41_41095

variable (L M K B C D A : Point)
variable [geometry : EuclideanGeometry]
open EuclideanGeometry

-- Definitions of collinearity, perpendicularity, parallelism, midpoints, and congruence
axiom midpoint (P Q R : Point) : Midpoint P Q R
axiom is_perpendicular (l1 l2 : Line) : IsPerpendicular l1 l2
axiom is_parallel (l1 l2 : Line) : IsParallel l1 l2
axiom median (P Q R : Point) (l : Line) : Median P Q R l
axiom orthocenter (P Q R O : Point) : Orthocenter P Q R O

-- Given conditions
variables
  (h1 : midpoint L C D)
  (h2 : median C M D (Line.mk M L))
  (h3 : is_perpendicular (Line.mk M L) (Line.mk C D))
  (h4 : is_parallel (Line.mk B K) (Line.mk C D))
  (h5 : is_parallel (Line.mk K L) (Line.mk A D))
  (h6 : is_perpendicular (Line.mk L M) (Line.mk B K))
  (h7 : is_perpendicular (Line.mk B M) (Line.mk K L))
  (h8 : is_parallel (Line.mk B L) (Line.mk K D))

-- Proof of the question
theorem KM_perp_KD : is_perpendicular (Line.mk K M) (Line.mk K D) := by
  sorry

end KM_perp_KD_l41_41095


namespace find_lambda_l41_41756

-- Define points A, B, C and their respective lengths and angles
variables {A B C : ℝ}
variable λ : ℝ
variable ab bc : ℝ
variable angle_ABC : ℝ

-- Conditions
def condition1 : ab = 2 := sorry
def condition2 : bc = real.sqrt 2 := sorry
def condition3 : angle_ABC = real.pi / 4 := sorry -- 45 degrees in radians

-- Define vectors and their properties
variable (AB : (ℝ × ℝ))
variable (BC : (ℝ × ℝ))

-- Dot product condition for perpendicularity
def condition4 : vector.dot (λ • BC - AB) AB = 0 := sorry

-- Lean theorem stating the problem
theorem find_lambda 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) : λ = 2 := sorry

end find_lambda_l41_41756


namespace g_sum_is_four_l41_41672

-- Define f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Given conditions as assumptions
axiom f_domain_range : ∀ x : ℝ, f (x + 2)
axiom f_odd : ∀ x : ℝ, f (x + 2) = -f (-(x + 2))
axiom f_has_inverse : ∀ y : ℝ, ∃ x : ℝ, f x = y
axiom g_symmetric_to_f : ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Problem statement
theorem g_sum_is_four : ∀ x : ℝ, g x + g (-x) = 4 :=
sorry

end g_sum_is_four_l41_41672


namespace hyperbola_properties_l41_41775

theorem hyperbola_properties (C : Type) (l : Type) (x y : ℝ) :
  -- Conditions
  (∃ a c : ℝ, a, c ≥ 0, (c, 0) ∈ l) ∧
  (∃ h : ℝ, ∀ (x y : ℝ), l = 4 * x - 3 * y + 20 = 0) ∧
  (∃ h : ℝ, ∀ (x y : ℝ), asymptote_parallel = 4 * x - 3 * y = 0) ∧
  (foci_on_x_axis : foci C = (lambda x, Point x 0)) ∧
  -- Prove that
  (standard_eq : ∃ t : ℝ, t > 0 ∧ hyperbola_eq = λ x y: ℝ, x^2 / (9 * t) - y^2 / (16 * t) = 1) ∧
  (eccentricity_eq : ∃ a c : ℝ, a, c ≥ 0 ∧ e = c / a ∧ e = 5 / 3) :=
by sorry

end hyperbola_properties_l41_41775


namespace number_of_diffs_l41_41365

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41365


namespace max_popsicles_with_10_dollars_l41_41114

theorem max_popsicles_with_10_dollars :
  (∃ (single_popsicle_cost : ℕ) (four_popsicle_box_cost : ℕ) (six_popsicle_box_cost : ℕ) (budget : ℕ),
    single_popsicle_cost = 1 ∧
    four_popsicle_box_cost = 3 ∧
    six_popsicle_box_cost = 4 ∧
    budget = 10 ∧
    ∃ (max_popsicles : ℕ),
      max_popsicles = 14 ∧
      ∀ (popsicles : ℕ),
        popsicles ≤ 14 →
        ∃ (x y z : ℕ),
          popsicles = x + 4*y + 6*z ∧
          x * single_popsicle_cost + y * four_popsicle_box_cost + z * six_popsicle_box_cost ≤ budget
  ) :=
sorry

end max_popsicles_with_10_dollars_l41_41114


namespace rook_chess_game_winner_l41_41477

-- Define the game state
structure GameState :=
  (m : Nat)
  (n : Nat)
  (rook_pos : Nat × Nat)
  (visited : Finset (Nat × Nat))

-- Define initial state of the game
def initial_state (m n : Nat) : GameState :=
  { m := m, n := n, rook_pos := (1, 1), visited := {(1, 1)}.to_finset }

-- Define a valid move given a game state
def valid_move (state : GameState) (new_pos : Nat × Nat) : Prop :=
  let (x, y) := state.rook_pos
  let (new_x, new_y) := new_pos
  new_x > 0 ∧ new_y > 0 ∧ new_x ≤ state.m ∧ new_y ≤ state.n ∧
  (new_x = x ∨ new_y = y) ∧
  (new_x, new_y) ∉ state.visited

-- Define the result after moving the rook to a new position
def move_rook (state : GameState) (new_pos : Nat × Nat) (h : valid_move state new_pos) : GameState :=
  { state with
    rook_pos := new_pos,
    visited := state.visited ∪ {(new_pos)}.to_finset }

-- Define win condition based on the inability to make a move
def cannot_move (state : GameState) : Prop :=
  ∀ x y, ¬ valid_move state (x, y)

-- Define the theorem statement
theorem rook_chess_game_winner (m n : Nat) :
  if m = 1 ∧ n = 1 then
    ¬ ∀ (state : GameState), cannot_move state → Player1Wins state
  else
    ∃ strategy : GameState → (GameState × (∃ h : valid_move state state'.rook_pos)),
    ∀ (state : GameState) (new_pos : Nat × Nat) (h : valid_move state new_pos),
    let state' := move_rook state new_pos h
    ¬ cannot_move state' → Player1Wins state :=
sorry

end rook_chess_game_winner_l41_41477


namespace x_sq_minus_y_sq_l41_41101

theorem x_sq_minus_y_sq : 
  let x := 3001 ^ 1501 - 3001 ^ (-1501)
  let y := 3001 ^ 1501 + 3001 ^ (-1501)
  (x^2 - y^2 = -4) :=
by
  intros x y
  sorry

end x_sq_minus_y_sq_l41_41101


namespace max_sum_of_ABC_l41_41459

/-- Theorem: The maximum value of A + B + C for distinct positive integers A, B, and C such that A * B * C = 2023 is 297. -/
theorem max_sum_of_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 2023) :
  A + B + C ≤ 297 :=
sorry

end max_sum_of_ABC_l41_41459


namespace students_more_than_rabbits_l41_41681

-- Definitions of conditions
def classrooms : ℕ := 5
def students_per_classroom : ℕ := 22
def rabbits_per_classroom : ℕ := 2

-- Statement of the theorem
theorem students_more_than_rabbits :
  classrooms * students_per_classroom - classrooms * rabbits_per_classroom = 100 := 
  by
    sorry

end students_more_than_rabbits_l41_41681


namespace mappings_from_A_to_B_l41_41327

open Finset

-- Definitions of the sets A and B
def A : Finset (String) := {"a", "b"}
def B : Finset (Int) := {-1, 0, 1}

-- Theorem statement asserting the number of mappings from A to B is 9
theorem mappings_from_A_to_B : card (A → B) = 9 := by
  sorry

end mappings_from_A_to_B_l41_41327


namespace no_four_distinct_nat_dividing_pairs_l41_41072

theorem no_four_distinct_nat_dividing_pairs (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∣ (b - c)) (h8 : a ∣ (b - d))
  (h9 : a ∣ (c - d)) (h10 : b ∣ (a - c)) (h11 : b ∣ (a - d)) (h12 : b ∣ (c - d))
  (h13 : c ∣ (a - b)) (h14 : c ∣ (a - d)) (h15 : c ∣ (b - d)) (h16 : d ∣ (a - b))
  (h17 : d ∣ (a - c)) (h18 : d ∣ (b - c)) : False := 
sorry

end no_four_distinct_nat_dividing_pairs_l41_41072


namespace simplify_T_l41_41490

noncomputable def T (x : ℝ) : ℝ :=
  (x+1)^4 - 4*(x+1)^3 + 6*(x+1)^2 - 4*(x+1) + 1

theorem simplify_T (x : ℝ) : T x = x^4 :=
  sorry

end simplify_T_l41_41490


namespace necessary_but_not_sufficient_l41_41842

variables {x y : ℝ}

def p : Prop := (x > 1) ∨ (y > 2)
def q : Prop := (x + y > 3)

theorem necessary_but_not_sufficient : 
  (∀ x y, q → p) ∧ ¬ (∀ x y, p → q) :=
by
  sorry

end necessary_but_not_sufficient_l41_41842


namespace maximum_n_for_dart_probability_l41_41979

theorem maximum_n_for_dart_probability (n : ℕ) (h : n ≥ 1) :
  (∃ r : ℝ, r = 1 ∧
  ∃ A_square A_circles : ℝ, A_square = n^2 ∧ A_circles = n * π * r^2 ∧
  (A_circles / A_square) ≥ 1 / 2) → n ≤ 6 := by
  sorry

end maximum_n_for_dart_probability_l41_41979


namespace fish_to_apples_l41_41468

variable {Fish Loaf Rice Apple : Type}
variable (f : Fish → ℝ) (l : Loaf → ℝ) (r : Rice → ℝ) (a : Apple → ℝ)
variable (F : Fish) (L : Loaf) (A : Apple) (R : Rice)

-- Conditions
axiom cond1 : 4 * f F = 3 * l L
axiom cond2 : l L = 5 * r R
axiom cond3 : r R = 2 * a A

-- Proof statement
theorem fish_to_apples : f F = 7.5 * a A :=
by
  sorry

end fish_to_apples_l41_41468


namespace asymptote_equation_l41_41752

theorem asymptote_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
(h : (∥(⟨2 * sqrt (a^2 + b^2), 0⟩ : EuclideanSpace ℝ (Fin 2))∥ / (|∥P - ⟨sqrt (a^2 + b^2), 0⟩∥ - ∥P - ⟨-sqrt (a^2 + b^2), 0⟩∥|) = sqrt 5 / 2)) :
by {sorry} :=
  have c : ℝ := sqrt (a^2 + b^2),
  have h1 : (c / a = (sqrt 5) / 2),
  from sorry,
  have h2 : y = ± ((1/2) * x),
  from sorry,
  show ∀ x y, x ± 2y = 0,
  from sorry,

end asymptote_equation_l41_41752


namespace sum_of_exponents_to_1024_l41_41176

theorem sum_of_exponents_to_1024 :
  ∃ (s : ℕ) (m : ℕ → ℕ) (b : ℕ → ℤ), 
  (∀ k, b k = 1 ∨ b k = -1) ∧ 
  (∀ i j, i < j → m i > m j) ∧ 
  (∃ a : fin s, 
      (b a * 3 ^ (m a) : ℤ) = 1024) →
      ∑ i in finset.range s, m i = 20 :=
by
  sorry

end sum_of_exponents_to_1024_l41_41176


namespace binary_to_decimal_l41_41671

theorem binary_to_decimal (x : ℕ) (h : x = 0b110010) : x = 50 := by
  sorry

end binary_to_decimal_l41_41671


namespace stock_decrease_to_original_l41_41244

theorem stock_decrease_to_original
  (x : ℝ) (h₀ : x > 0) :
  let end_2006 := 1.3 * x,
      end_2007 := 1.2 * end_2006,
      required_fractional_decrease := (end_2007 - x) / end_2007
  in required_fractional_decrease ≈ 0.35897 :=
sorry

end stock_decrease_to_original_l41_41244


namespace num_unique_differences_l41_41394

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41394


namespace rectangular_field_perimeter_l41_41138

theorem rectangular_field_perimeter (A L : ℝ) (h1 : A = 300) (h2 : L = 15) : 
  let W := A / L 
  let P := 2 * (L + W)
  P = 70 := by
  sorry

end rectangular_field_perimeter_l41_41138


namespace ratio_q_l41_41260

-- Definitions and conditions
def side_length : ℝ := 4
def radius : ℝ := side_length / 2
def t_prime : ℝ := (1 / 2) * 4 * radius  -- area of the triangle
def q_prime : ℝ := (π * radius^2) / 2    -- area of the semi-circular region

-- The proof statement
theorem ratio_q'_t' : q_prime / t_prime = π / 2 := by
  sorry

end ratio_q_l41_41260


namespace vector_perpendicular_k_l41_41758

theorem vector_perpendicular_k (k : ℝ) : 
  let a := (k, 3)
      b := (1, 4)
      c := (2, 1)
      v := (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) in
  (v.1 * c.1 + v.2 * c.2 = 0) → k = 3 :=
by
  intros a b c v perp
  sorry

end vector_perpendicular_k_l41_41758


namespace people_per_tent_l41_41857

theorem people_per_tent (total_people : ℕ) (people_in_house : ℕ) (remaining_people : ℕ) (total_tents : ℕ) : 
  total_people = 14 → people_in_house = 4 → remaining_people = total_people - people_in_house → total_tents = 5 →
  remaining_people / total_tents = 2 :=
by
  intros
  unfold remaining_people
  sorry

end people_per_tent_l41_41857


namespace acute_triangle_l41_41246

noncomputable def angle1 (x : ℝ) := 2 * x
noncomputable def angle2 (x : ℝ) := x
noncomputable def angle3 (x : ℝ) := 1.5 * x

theorem acute_triangle (x : ℝ) (h : angle1 x + angle2 x + angle3 x = 180) :
  ∠1 < 90 ∧ ∠2 < 90 ∧ ∠3 < 90 :=
by
  -- Providing the proof is unnecessary.
  sorry

end acute_triangle_l41_41246


namespace floyd_time_after_3_l41_41700

noncomputable def time_after_clock_movement (initial_hour initial_minute minute_angle_sum : ℕ) : ℕ :=
  let hour_angle := minute_angle_sum / 13
  let minute_angle := 12 * hour_angle
  let minutes := minute_angle * 60 / 360
  let current_minutes := initial_minute + minutes
  current_minutes - if current_minutes >= 60 then 60 else 0 := 14

theorem floyd_time_after_3 (initial_hour initial_minute : ℕ) (minute_angle_sum : ℕ) :
  initial_hour = 2 → initial_minute = 36 → minute_angle_sum = 247 →
  time_after_clock_movement initial_hour initial_minute minute_angle_sum = 14 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  noncomputable def time_after_clock_movement := 14
  sorry

end floyd_time_after_3_l41_41700


namespace giza_pyramid_tallest_years_l41_41135

theorem giza_pyramid_tallest_years :
  let h := 500 + 20 in
  let w := h + 234 in
  h + w = 1274 → (2560 + 1311) = 3871 :=
by
  intros h w hw_sum
  sorry

end giza_pyramid_tallest_years_l41_41135


namespace arithmetic_progression_absolute_sum_constant_l41_41163

noncomputable def arithmetic_progression_sum (a : ℕ → ℝ) (n : ℕ) (d : ℝ) : ℝ :=
Σ i in finset.range n, |a i|

theorem arithmetic_progression_absolute_sum_constant 
  (a : ℕ → ℝ) (d : ℝ) (n : ℕ)
  (h1 : arithmetic_progression_sum a n d = 100)
  (h2 : arithmetic_progression_sum (λ i, a i + 1) n d = 100)
  (h3 : arithmetic_progression_sum (λ i, a i + 2) n d = 100) :
  n^2 * d = 400 :=
sorry

end arithmetic_progression_absolute_sum_constant_l41_41163


namespace parallel_lines_of_equation_l41_41876

theorem parallel_lines_of_equation (y : Real) :
  (y - 2) * (y + 3) = 0 → (y = 2 ∨ y = -3) :=
by
  sorry

end parallel_lines_of_equation_l41_41876


namespace Canada_moose_population_l41_41461

theorem Canada_moose_population (moose beavers humans : ℕ) (h1 : beavers = 2 * moose) 
                              (h2 : humans = 19 * beavers) (h3 : humans = 38 * 10^6) : 
                              moose = 1 * 10^6 :=
by
  sorry

end Canada_moose_population_l41_41461


namespace dolls_completion_time_l41_41145

def time_to_complete_dolls (craft_time_per_doll break_time_per_three_dolls total_dolls start_time : Nat) : Nat :=
  let total_craft_time := craft_time_per_doll * total_dolls
  let total_breaks := (total_dolls / 3) * break_time_per_three_dolls
  let total_time := total_craft_time + total_breaks
  (start_time + total_time) % 1440 -- 1440 is the number of minutes in a day

theorem dolls_completion_time :
  time_to_complete_dolls 105 30 10 600 = 300 := -- 600 is 10:00 AM in minutes, 300 is 5:00 AM in minutes
sorry

end dolls_completion_time_l41_41145


namespace main_theorem_l41_41357

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
def condition1 : Prop := ∥a∥ = 2
def condition2 : Prop := ∥b∥ = 1
def condition3 : Prop := ∥a - b∥ = 2

-- Define the proof statements
def proof1 (h1 : condition1) (h2 : condition2) (h3 : condition3) : Prop := inner_product a b = 1 / 2
def proof2 (h1 : condition1) (h2 : condition2) (h3 : condition3) : Prop := ∥a + b∥ = real.sqrt 6
def proof3 (h1 : condition1) (h2 : condition2) (h3 : condition3) : Prop := (inner_product a b / ∥b∥) = 1 / 2

-- The main theorem combining all proofs
theorem main_theorem (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  proof1 h1 h2 h3 ∧ proof2 h1 h2 h3 ∧ proof3 h1 h2 h3 :=
by
  split;
  sorry

end main_theorem_l41_41357


namespace area_of_triangle_BQW_l41_41807

open Real

variables (A B C D Z W Q : ℝ × ℝ)
variables (AB AZ WC : ℝ)
variables (area_ZWCD : ℝ)

-- Define the conditions
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  -- Assuming standard coordinate rectangle conditions
  (A.1 = D.1) ∧ (B.1 = C.1) ∧ (A.2 = B.2) ∧ (D.2 = C.2) ∧ 
  (A.1 ≠ B.1) ∧ (A.2 ≠ D.2)

def point_distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def trapezoid_area (A B C D : ℝ × ℝ) (height: ℝ) : ℝ :=
  (1/2) * (point_distance A B + point_distance C D) * height

-- Define the necessary relationships
def BQW_area (A B Z W : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((A.1 * (B.2 - Z.2) + B.1 * (Z.2 - A.2) + Z.1 * (A.2 - B.2)))

theorem area_of_triangle_BQW :
  is_rectangle A B C D →
  point_distance A Z = 8 →
  point_distance W C = 8 →
  point_distance A B = 16 →
  trapezoid_area Z W C D 10 = 160 →
  BQW_area B Z W = 48 :=
by
  sorry

end area_of_triangle_BQW_l41_41807


namespace incenter_coordinates_l41_41813

-- Given conditions
variables {P Q R : Point}

def side_lengths (P Q R : Point) : Prop :=
  dist P Q = 8 ∧ dist Q R = 10 ∧ dist R P = 6

-- The proof statement
theorem incenter_coordinates (h : side_lengths P Q R) :
  let I := incenter P Q R in
  (I = 1/3 * P + 5/12 * Q + 1/4 * R) :=
by
  sorry

end incenter_coordinates_l41_41813


namespace geometric_series_sum_l41_41657

theorem geometric_series_sum :
  let a := (1 : ℝ) / 5
  let r := -(1 : ℝ) / 5
  let n := 5
  let S_n := (a * (1 - r ^ n)) / (1 - r)
  S_n = 521 / 3125 := by
  sorry

end geometric_series_sum_l41_41657


namespace optimal_tile_choice_and_count_l41_41852

noncomputable def room_length : ℝ := 6  -- length in meters
noncomputable def room_width : ℝ := 4.8  -- width in meters

noncomputable def tile_sizes : List ℝ := [10, 35, 40, 50]  -- tile sizes in cm

noncomputable def area_room_cm² : ℝ := (room_length * 100) * (room_width * 100)  -- area in square cm

theorem optimal_tile_choice_and_count :
  ∃ (tile_size : ℝ), tile_size ∈ tile_sizes ∧ tile_size = 40 ∧ (area_room_cm² / (tile_size * tile_size)) = 180 :=
by
  sorry

end optimal_tile_choice_and_count_l41_41852


namespace count_valid_n_values_l41_41305

open Nat

def S : ℕ → ℕ := λ n, (n.digits 10).sum

theorem count_valid_n_values :
  (Finset.card (Finset.filter (λ n, n + S n + S (S n) = 2007) (Finset.range 2008))) = 4 :=
by
  sorry

end count_valid_n_values_l41_41305


namespace find_b_value_l41_41551

-- Let's define the given conditions as hypotheses in Lean

theorem find_b_value 
  (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 2)) 
  (h2 : (x2, y2) = (8, 14)) 
  (midpoint : ∃ (m1 m2 : ℤ), m1 = (x1 + x2) / 2 ∧ m2 = (y1 + y2) / 2 ∧ (m1, m2) = (5, 8))
  (perpendicular_bisector : ∀ (x y : ℤ), x + y = b → (x, y) = (5, 8)) :
  b = 13 := 
by {
  sorry
}

end find_b_value_l41_41551


namespace total_flowers_correct_l41_41123

def rosa_original_flowers : ℝ := 67.5
def andre_gifted_flowers : ℝ := 90.75
def total_flowers (rosa : ℝ) (andre : ℝ) : ℝ := rosa + andre

theorem total_flowers_correct : total_flowers rosa_original_flowers andre_gifted_flowers = 158.25 :=
by 
  rw [total_flowers]
  sorry

end total_flowers_correct_l41_41123


namespace number_of_differences_l41_41388

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41388


namespace sqrt_three_irrational_l41_41643

-- Define what it means for a number to be irrational
def irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- Given numbers
def neg_two : ℝ := -2
def one_half : ℝ := 1 / 2
def sqrt_three : ℝ := real.sqrt 3
def two : ℝ := 2

-- The proof statement
theorem sqrt_three_irrational : irrational sqrt_three :=
sorry

end sqrt_three_irrational_l41_41643


namespace polynomials_divisibility_l41_41558

noncomputable def polynomial_nonnegative_coef (P : polynomial ℝ) : Prop :=
∀ n, 0 ≤ P.coeff n

noncomputable def polynomial_positive_leading_negative_constant (Q : polynomial ℝ) : Prop :=
0 < Q.leading_coeff ∧ Q.coeff 0 < 0

theorem polynomials_divisibility (P Q : polynomial ℝ) (hP : polynomial_nonnegative_coef P) (hQ : polynomial_positive_leading_negative_constant Q) :
¬ (P % Q = 0) := by
  sorry

end polynomials_divisibility_l41_41558


namespace average_of_four_digits_l41_41537

theorem average_of_four_digits (sum9 : ℤ) (avg9 : ℤ) (avg5 : ℤ) (sum4 : ℤ) (n : ℤ) :
  avg9 = 18 →
  n = 9 →
  sum9 = avg9 * n →
  avg5 = 26 →
  sum4 = sum9 - (avg5 * 5) →
  avg4 = sum4 / 4 →
  avg4 = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_of_four_digits_l41_41537


namespace harriet_return_speed_l41_41199

noncomputable def harriet_speed_back_to_aville (speed_to_bt : ℝ) (time_to_bt_aville_hours : ℝ) (total_trip_hours : ℝ) :=
  let distance := speed_to_bt * time_to_bt_aville_hours
  let time_back_hours := total_trip_hours - time_to_bt_aville_hours
  let speed_back := distance / time_back_hours
  speed_back

theorem harriet_return_speed :
  harriet_speed_back_to_aville 90 3.2 5 = 160 :=
begin
  -- Given:
  -- speed_to_bt = 90 km/hr
  -- time_to_bt_aville_hours = 3.2 hours
  -- total_trip_hours = 5 hours

  -- The function will calculate the distance as:
  -- distance = 90 * 3.2 = 288 km

  -- The return time:
  -- time_back_hours = 5 - 3.2 = 1.8 hours

  -- The speed back:
  -- speed_back = 288 / 1.8 = 160 km/hr
  
  -- Therefore, the result should be:
  refl,
end

end harriet_return_speed_l41_41199


namespace negation_of_P_is_there_exists_x_ge_0_l41_41326

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

-- State the theorem of the negation of P
theorem negation_of_P_is_there_exists_x_ge_0 : ¬P ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by sorry

end negation_of_P_is_there_exists_x_ge_0_l41_41326


namespace smallest_k_for_sum_of_squares_multiple_of_360_l41_41303

theorem smallest_k_for_sum_of_squares_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ ∀ n : ℕ, n > 0 → (n * (n + 1) * (2 * n + 1)) / 6 % 360 = 0 → k ≤ n :=
by sorry

end smallest_k_for_sum_of_squares_multiple_of_360_l41_41303


namespace min_value_of_b_plus_16c_l41_41814

variables {A B C : ℝ} (λ : ℝ) (a b c : ℝ)
variable (Q : ℝ)

-- Given conditions
def conditions : Prop :=
  A = π / 3 ∧ 
  λ > 0 ∧
  Q = λ * (a / sqrt(a^2 + b^2) + c / sqrt(a^2 + c^2)) ∧
  Q = 4 * sqrt 3

-- The proof problem statement
theorem min_value_of_b_plus_16c (h : conditions a b c Q λ) : (b + 16 * c) = 100 :=
sorry

end min_value_of_b_plus_16c_l41_41814


namespace coordinates_after_translation_l41_41117

-- Define the initial coordinate of point P
def P : (ℤ × ℤ) := (-4, 3)

-- Define the transformation as a function
def translate_left (x : ℤ) (units : ℤ) : ℤ := x - units
def translate_down (y : ℤ) (units : ℤ) : ℤ := y - units

-- Apply the transformations to get P'
def transform_P (P : (ℤ × ℤ)) : (ℤ × ℤ) :=
  let (x, y) := P in
  let x' := translate_left x 2 in
  let y' := translate_down y 2 in
  (x', y')

-- Define the proof statement
theorem coordinates_after_translation : transform_P P = (-6, 1) :=
  by
    sorry

end coordinates_after_translation_l41_41117


namespace non_congruent_triangles_from_grid_l41_41796

def point : Type := ℕ × ℕ

noncomputable def points : list point := 
[(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]

def non_congruent_triangles_count (pts : list point) : ℕ := 
sorry -- Function to calculate number of non-congruent triangles

theorem non_congruent_triangles_from_grid :
  non_congruent_triangles_count points = 12 := 
sorry

end non_congruent_triangles_from_grid_l41_41796


namespace find_value_of_expression_l41_41032

theorem find_value_of_expression 
(h : ∀ (a b : ℝ), a * (3:ℝ)^2 - b * (3:ℝ) = 6) : 
  ∀ (a b : ℝ), 2023 - 6 * a + 2 * b = 2019 := 
by
  intro a b
  have h1 : 9 * a - 3 * b = 6 := by sorry
  have h2 : 3 * a - b = 2 := by sorry
  have result := 2023 - 2 * (3 * a - b)
  rw h2 at result
  exact result

end find_value_of_expression_l41_41032


namespace champion_is_D_l41_41810

axiom Players : Type
axiom A B C D : Players
axiom is_champion : Players → Prop

-- Conditions (Predictions)
axiom A_prediction : ¬ is_champion B
axiom B_prediction : is_champion C ∨ is_champion D
axiom C_prediction : ¬ is_champion A ∧ ¬ is_champion D

-- One prediction is wrong
axiom one_wrong : ∃ p : Players → Prop, 
  (p = A_prediction ∨ p = B_prediction ∨ p = C_prediction) ∧
  (¬ ∀ p : Players → Prop, (p = A_prediction ∨ p = B_prediction ∨ p = C_prediction) → p)

theorem champion_is_D : is_champion D :=
by
  sorry

end champion_is_D_l41_41810


namespace region_of_inequality_l41_41914

theorem region_of_inequality (x y : ℝ) : (x + y - 6 < 0) → y < -x + 6 := by
  sorry

end region_of_inequality_l41_41914


namespace sum_of_coefficients_odd_powers_eq_nine_l41_41570

theorem sum_of_coefficients_odd_powers_eq_nine (x : ℝ) :
  let f := (x - 2)^3 * (2 * x + 1)^2 in
  let expanded_f := 4 * x^5 - 20 * x^4 + 25 * x^3 + 10 * x^2 - 20 * x - 8 in
  (expanded_f.coeff 5 + expanded_f.coeff 3 + expanded_f.coeff 1) = 9 :=
sorry

end sum_of_coefficients_odd_powers_eq_nine_l41_41570


namespace ending_number_is_64_l41_41170

theorem ending_number_is_64 
  (seq : List ℕ)
  (h_len : seq.length = 6)
  (h_start : seq.head = some 2)
  (h_cond : ∀ n ∈ seq, ∀ k : ℕ, (k > 1 ∧ k % 2 = 1) → n % k ≠ 0) :
  seq.getLast = 64 := 
sorry

end ending_number_is_64_l41_41170


namespace no_day_income_is_36_l41_41628

theorem no_day_income_is_36 : ∀ (n : ℕ), 3 * 3^(n-1) ≠ 36 :=
by
  intro n
  sorry

end no_day_income_is_36_l41_41628


namespace length_of_BC_is_7_l41_41450

noncomputable def triangle_length_BC (a b c : ℝ) (A : ℝ) (S : ℝ) (P : ℝ) : Prop :=
  (P = a + b + c) ∧ (P = 20) ∧ (S = 1 / 2 * b * c * Real.sin A) ∧ (S = 10 * Real.sqrt 3) ∧ (A = Real.pi / 3) ∧ (b * c = 20)

theorem length_of_BC_is_7 : ∃ a b c, triangle_length_BC a b c (Real.pi / 3) (10 * Real.sqrt 3) 20 ∧ a = 7 := 
by
  -- proof omitted
  sorry

end length_of_BC_is_7_l41_41450


namespace lunks_for_two_dozen_bananas_l41_41765

-- Definitions based on the given conditions
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks / 4) * 2
def kunks_to_bananas (kunks : ℕ) : ℕ := (kunks / 3) * 6

-- The theorem to be proved
theorem lunks_for_two_dozen_bananas : ∀ (lunks needed_for_bananas : ℕ), 
  (lunek_to_bananas (lunks_needed l:strub:karly)) :=:=:= (4 * 6) := (4 * 6) := 24. eyi:bananas_needed : sorry ) =

end lunks_for_two_dozen_bananas_l41_41765


namespace no_valid_domino_chain_l41_41313

-- Define the domain, which are the remaining numbers without six.
def remaining_numbers := {n | n ∈ {0, 1, 2, 3, 4, 5}}

-- The statement to be proven: It's not possible to arrange the remaining dominoes in a row.
theorem no_valid_domino_chain (d : Multiset (Nat × Nat))
  (h₁ : ∀ x ∈ d, (x.1 ∈ remaining_numbers) ∧ (x.2 ∈ remaining_numbers))
  (h₂ : ∀ n ∈ remaining_numbers, Multiset.count (n, n) d = 0)
  (h₃ : ∀ n ∈ remaining_numbers, even (d.countp (λ (p : Nat × Nat), p.1 = n ∨ p.2 = n))) :
  ¬ ∃ l : List (Nat × Nat), l.toMultiset = d ∧ (∀ p ∈ l.tail, p.1 = (l.head).2 ∨ p.2 = (l.head).2) ∧ (∀ p ∈ l, p ≠ (6, 6)) :=
sorry

end no_valid_domino_chain_l41_41313


namespace complex_number_location_in_plane_l41_41557

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem complex_number_location_in_plane :
  is_in_second_quadrant (-2) 5 :=
by
  sorry

end complex_number_location_in_plane_l41_41557


namespace equal_angles_l41_41715

noncomputable theory

variables {α β γ θ : ℝ}
variables (h1 : α ∈ Ioo 0 π) (h2 : β ∈ Ioo 0 π)
variables (h3 : γ ∈ Ioo 0 π) (h4 : θ ∈ Ioo 0 π)
variables (k : ℝ) 

def condition1 := (sin α) / (sin β) = k
def condition2 := (sin γ) / (sin θ) = k
def condition3 := (sin (α - γ)) / (sin (β - θ)) = k

theorem equal_angles (h1 : α ∈ Ioo 0 π) (h2 : β ∈ Ioo 0 π) 
    (h3 : γ ∈ Ioo 0 π) (h4 : θ ∈ Ioo 0 π) 
    (k : ℝ)
    (cond1 : (sin α) / (sin β) = k)
    (cond2 : (sin γ) / (sin θ) = k)
    (cond3 : (sin (α - γ)) / (sin (β - θ)) = k) : 
    α = β ∧ γ = θ :=
by
  sorry

end equal_angles_l41_41715


namespace four_digit_numbers_containing_odd_ones_l41_41993

theorem four_digit_numbers_containing_odd_ones :
  let digits := {0, 1, 2, 3, 4, 5}
  ∃ count : ℕ, count = 454 ∧ 
  (∀ n : ℕ, (n ∈ digits → n = 4)) :=
sorry

end four_digit_numbers_containing_odd_ones_l41_41993


namespace constant_term_in_expansion_l41_41478

theorem constant_term_in_expansion :
  (∃ (c : ℕ), c = 31) :=
begin
  let f := (1 + x + (1 / x^2 : ℚ))^5,
  have h₁ : ∀ f : ℚ → ℚ, (1 + x + (1 / x^2))^5 = ∑ b in finset.range 6, 
    (nat.choose 5 b) * (1^(5-b)) * (x^b) * ((1/x^2)^(5-b)),
  have constant_term := λ f, (f 0),
  exact ⟨31, sorry⟩,
end

end constant_term_in_expansion_l41_41478


namespace time_to_meet_l41_41518

variables {v_g : ℝ} -- speed of Petya and Vasya on the dirt road
variable {v_a : ℝ} -- speed of Petya on the paved road
variable {t : ℝ} -- time from start until Petya and Vasya meet
variable {x : ℝ} -- distance from the starting point to the bridge

-- Conditions
axiom Petya_speed : v_a = 3 * v_g
axiom Petya_bridge_time : x / v_a = 1
axiom Vasya_speed : v_a ≠ 0 ∧ v_g ≠ 0

-- Statement
theorem time_to_meet (h1 : v_a = 3 * v_g) (h2 : x / v_a = 1) (h3 : v_a ≠ 0 ∧ v_g ≠ 0) : t = 2 :=
by
  have h4 : x = 3 * v_g := sorry
  have h5 : (2 * x - 2 * v_g) / (2 * v_g) = 1 := sorry
  exact sorry

end time_to_meet_l41_41518


namespace square_of_hypotenuse_product_eq_160_l41_41184

noncomputable def square_of_product_of_hypotenuses (x y : ℝ) (h1 h2 : ℝ) : ℝ :=
  (h1 * h2) ^ 2

theorem square_of_hypotenuse_product_eq_160 :
  ∀ (x y h1 h2 : ℝ),
    (1 / 2) * x * (2 * y) = 4 →
    (1 / 2) * x * y = 8 →
    x^2 + (2 * y)^2 = h1^2 →
    x^2 + y^2 = h2^2 →
    square_of_product_of_hypotenuses x y h1 h2 = 160 :=
by
  intros x y h1 h2 area1 area2 pythagorean1 pythagorean2
  -- The detailed proof steps would go here
  sorry

end square_of_hypotenuse_product_eq_160_l41_41184


namespace sum_of_prime_factors_l41_41312

theorem sum_of_prime_factors :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a * b * c = 343000 ∧ 
  pairwise (λ x y, Nat.gcd x y = 1) [a, b, c]
  ∧ a + b + c = 476 :=
sorry

end sum_of_prime_factors_l41_41312


namespace calories_in_300_grams_proof_l41_41855

def apple_juice_grams : ℕ := 150
def sugar_grams : ℕ := 200
def water_grams : ℕ := 300

def apple_juice_calories_per_100g : ℕ := 50
def sugar_calories_per_100g : ℕ := 400
def water_calories_per_100g : ℕ := 0

def total_grams : ℕ := apple_juice_grams + sugar_grams + water_grams
def total_calories : ℕ := 
  (apple_juice_grams * apple_juice_calories_per_100g) / 100 +
  (sugar_grams * sugar_calories_per_100g) / 100 +
  (water_grams * water_calories_per_100g) / 100

def caloric_density := total_calories.to_float / total_grams.to_float

def calories_in_300_grams := (caloric_density * 300).to_nat

theorem calories_in_300_grams_proof : calories_in_300_grams = 404 := by
  sorry

end calories_in_300_grams_proof_l41_41855


namespace xy_value_l41_41835

theorem xy_value :
  ∃ a b c x y : ℝ,
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
    3 * a + 2 * b + c = 5 ∧
    2 * a + b - 3 * c = 1 ∧
    (∀ m, m = 3 * a + b - 7 * c → (m = x ∨ m = y)) ∧
    x = -5 / 7 ∧
    y = -1 / 11 ∧
    x * y = 5 / 77 :=
sorry

end xy_value_l41_41835


namespace sum_of_first_n_terms_l41_41922

def sequence_term (k : ℕ) : ℚ :=
  2 * (1 / k - 1 / (k + 1))

def sequence_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, sequence_term (k + 1))

theorem sum_of_first_n_terms (n : ℕ) : sequence_sum n = 2 * n / (n + 1) :=
by
  sorry

end sum_of_first_n_terms_l41_41922


namespace count_4_primable_correct_l41_41624

def is_prime_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 5 ∨ d = 7

def is_4_primable (n : ℕ) : Prop :=
  (∀ i : ℕ, (n / 10 ^ i) % 10 = 3 ∨ (n / 10 ^ i) % 10 = 5 ∨ (n / 10 ^ i) % 10 = 7) ∧ n % 4 = 0

def count_4_primable : ℕ :=
  (list.range 1000).count is_4_primable

theorem count_4_primable_correct : count_4_primable = 10 := 
  sorry

end count_4_primable_correct_l41_41624


namespace det_le_one_eigenvalue_abs_one_l41_41829

variables {n : ℕ} {A : matrix (fin n) (fin n) ℝ}

/-- Conditions for matrix A --/
axiom nonneg_entries  (i j : fin n) : 0 ≤ A i j
axiom sum_entries     : ∑ i j, A i j = n

/-- Part (a): Prove that |det A| ≤ 1 --/
theorem det_le_one (A : matrix (fin n) (fin n) ℝ) (h : ∀ i j, 0 ≤ A i j) 
  (hsum : ∑ i j, A i j = n) : abs (matrix.det A) ≤ 1 :=
sorry

/-- Part (b): If |det A| = 1, then for any eigenvalue λ of A, |λ| = 1 --/
theorem eigenvalue_abs_one (A : matrix (fin n) (fin n) ℝ) (h : ∀ i j, 0 ≤ A i j) 
  (hdet : abs (matrix.det A) = 1) (λ : ℂ) (hλ : is_eigenvalue A λ) : abs λ = 1 :=
sorry

end det_le_one_eigenvalue_abs_one_l41_41829


namespace Alice_before_230_prob_l41_41257

noncomputable def probability_Alice_arrives_before_230_given_Alice_before_Bob (a b : ℝ) 
    (h1 : 0 ≤ a ∧ a ≤ 60) (h2 : 0 ≤ b ∧ b ≤ 60) (h3 : a < b) : ℝ :=
  if h4 : a < 30 then 1/4 else 0

theorem Alice_before_230_prob:
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 60 → 0 ≤ b ∧ b ≤ 60 → a < b →
  probability_Alice_arrives_before_230_given_Alice_before_Bob a b ‹0 ≤ a ∧ a ≤ 60› ‹0 ≤ b ∧ b ≤ 60› ‹a < b› = 1/4 :=
by
  sorry

end Alice_before_230_prob_l41_41257


namespace average_score_girls_cedar_drake_l41_41258

theorem average_score_girls_cedar_drake
  (C c D d : ℕ)
  (cedar_boys_score cedar_girls_score cedar_combined_score
   drake_boys_score drake_girls_score drake_combined_score combined_boys_score : ℝ)
  (h1 : cedar_boys_score = 68)
  (h2 : cedar_girls_score = 80)
  (h3 : cedar_combined_score = 73)
  (h4 : drake_boys_score = 75)
  (h5 : drake_girls_score = 88)
  (h6 : drake_combined_score = 83)
  (h7 : combined_boys_score = 74)
  (h8 : (68 * C + 80 * c) / (C + c) = 73)
  (h9 : (75 * D + 88 * d) / (D + d) = 83)
  (h10 : (68 * C + 75 * D) / (C + D) = 74) :
  (80 * c + 88 * d) / (c + d) = 87 :=
by
  -- proof is omitted
  sorry

end average_score_girls_cedar_drake_l41_41258


namespace segment_length_l41_41800

theorem segment_length {t k : ℝ} :
  let l_x := 1 + (3/5) * t,
      l_y := (4/5) * t,
      c_x := 4 * k^2,
      c_y := 4 * k
  -- Points of intersection
  (A_x, A_y) := (4, 4),
  (B_x, B_y) := (1/4, -1)
  -- Distance formula
  (dx := B_x - A_x),
  (dy := B_y - A_y),
  AB := sqrt (dx^2 + dy^2)
  -- Segment length AB is 25/4
  AB = (25/4) :=
by 
  sorry

end segment_length_l41_41800


namespace internal_diagonal_exists_min_internal_diagonals_l41_41597

def is_diagonal (P : Type) [Polygon P] (d : Diagonal P) : Prop := sorry
def is_internal_diagonal (P : Type) [Polygon P] (d : Diagonal P) : Prop := sorry

theorem internal_diagonal_exists (P : Type) [Polygon P] (n : ℕ) (h : n > 3) : ∃ d, is_internal_diagonal P d :=
by sorry

theorem min_internal_diagonals (P : Type) [Polygon P] (n : ℕ) (h : n > 3) : 
  count_internal_diagonals P ≥ n - 3 :=
by sorry

end internal_diagonal_exists_min_internal_diagonals_l41_41597


namespace stripe_area_is_640pi_l41_41987

noncomputable def cylinder_stripe_area (diameter height stripe_width : ℝ) (revolutions : ℕ) : ℝ :=
  let circumference := Real.pi * diameter
  let length := circumference * (revolutions : ℝ)
  stripe_width * length

theorem stripe_area_is_640pi :
  cylinder_stripe_area 20 100 4 4 = 640 * Real.pi :=
by 
  sorry

end stripe_area_is_640pi_l41_41987


namespace matchstick_solution_exists_l41_41579

def matchstick_problem (total_matches area unit_squares : ℕ) : Prop :=
  total_matches = 12 ∧ unit_squares = 4 ∧ area = total_matches

theorem matchstick_solution_exists : ∃ config, matchstick_problem 12 16 4 := sorry

end matchstick_solution_exists_l41_41579


namespace num_possible_bases_count_possible_bases_l41_41132

theorem num_possible_bases (b : ℕ) (hb1 : b ≥ 2) (hb2 : b^3 ≤ 256) (hb3 : 256 < b^4) :
  b = 5 ∨ b = 6 :=
begin
  sorry
end

theorem count_possible_bases : Nat.card { b : ℕ // 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 } = 2 :=
begin
  sorry
end

end num_possible_bases_count_possible_bases_l41_41132


namespace number_of_differences_l41_41387

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41387


namespace solve_kindergarten_candy_l41_41235

def kindergarten_candy_problem : Prop :=
  ∃ x y z : ℕ,
    50 * x + 180 * y + 150 * z = 2200 ∧ 
    (∀ x' y' z' : ℕ, 50 * x' + 180 * y' + 150 * z' = 2200 → 25 * x + 95 * y + 80 * z ≤ 25 * x' + 95 * y' + 80 * z') ∧
    x = 2 ∧ y = 5 ∧ z = 8

theorem solve_kindergarten_candy : kindergarten_candy_problem :=
begin
  sorry
end

end solve_kindergarten_candy_l41_41235


namespace non_square_integer_equation_l41_41083

-- Define the nth non-square positive integer function
def nth_non_square (n : ℕ) : ℕ := sorry

-- Define the closest integer function
def closest_integer (x : ℝ) : ℤ := 
  let m := ⌊x⌋ in
  if x - m ≤ 0.5 then m else m + 1

-- Prove the equation given the conditions
theorem non_square_integer_equation (n : ℕ) : 
  nth_non_square n = n + closest_integer (real.sqrt n) :=
sorry

end non_square_integer_equation_l41_41083


namespace common_difference_arith_seq_l41_41062

theorem common_difference_arith_seq (a : ℕ → ℝ) (d : ℝ)
    (h₀ : a 1 + a 5 = 10)
    (h₁ : a 4 = 7)
    (h₂ : ∀ n, a (n + 1) = a n + d) : 
    d = 2 := by
  sorry

end common_difference_arith_seq_l41_41062


namespace quadratic_condition_l41_41024

theorem quadratic_condition (m : ℤ) (x : ℝ) :
  (m + 1) * x^(m^2 + 1) - 2 * x - 5 = 0 ∧ m^2 + 1 = 2 ∧ m + 1 ≠ 0 ↔ m = 1 := 
by
  sorry

end quadratic_condition_l41_41024


namespace train_pass_bridge_time_l41_41984

-- Given conditions
def train_length : ℕ := 460  -- length in meters
def bridge_length : ℕ := 140  -- length in meters
def speed_kmh : ℝ := 45  -- speed in kilometers per hour

-- Prove that the time to pass the bridge is 48 seconds
theorem train_pass_bridge_time :
  let distance := train_length + bridge_length in
  let speed_ms := speed_kmh * (1000 / 3600) in
  (distance / speed_ms) = 48 :=
by
  sorry

end train_pass_bridge_time_l41_41984


namespace sin_squared_half_sum_angles_max_area_triangle_l41_41355

-- Definitions and conditions from part (a)
variables (A B C a b c : ℝ)
variable (h1 : 2 * (a^2 + b^2 - c^2) = 3 * a * b)
variable (h2 : c = 2)

-- Question 1: Prove sin^2((A+B)/2) = 7/8
theorem sin_squared_half_sum_angles (h1 : 2 * (a^2 + b^2 - c^2) = 3 * a * b) :
  sin^2 ((A+B)/2) = 7/8 :=
  sorry

-- Question 2: Prove maximum area S_ABC of triangle ABC is sqrt(7)
theorem max_area_triangle (h1 : 2 * (a^2 + b^2 - c^2) = 3 * a * b) (h2 : c = 2) :
  ∃ a b : ℝ, a^2 + b^2 - 4 = (3/2) * a * b ∧ (1/2) * a * b * sin C = sqrt 7 :=
  sorry

end sin_squared_half_sum_angles_max_area_triangle_l41_41355


namespace quadrilateral_not_parallelogram_if_two_pairs_opposite_sides_equal_l41_41799

theorem quadrilateral_not_parallelogram_if_two_pairs_opposite_sides_equal 
  (A B C D : Point)
  (h1 : dist A B = dist C D)
  (h2 : dist B C = dist D A) :
  ¬(∀ (P : Plane), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ D ∈ P) → ¬is_parallelogram A B C D :=
sorry

end quadrilateral_not_parallelogram_if_two_pairs_opposite_sides_equal_l41_41799


namespace easter_eggs_total_l41_41177

theorem easter_eggs_total (h he total : ℕ)
 (hannah_eggs : h = 42) 
 (twice_he : h = 2 * he) 
 (total_eggs : total = h + he) : 
 total = 63 := 
sorry

end easter_eggs_total_l41_41177


namespace find_b_l41_41293

theorem find_b (b : ℝ) (h : log b 729 = -3) : b = 1 / 9 :=
sorry

end find_b_l41_41293


namespace flower_beds_fraction_l41_41975

-- Define the dimensions of the yard
def yard_length : ℝ := 30
def yard_width : ℝ := 8

-- Define the dimensions of the trapezoid
def trapezoid_side1 : ℝ := 20
def trapezoid_side2 : ℝ := 35

-- Define the dimensions of the isosceles right triangles
def triangle_leg_length : ℝ := (trapezoid_side2 - trapezoid_side1) / 2

-- Calculate the areas
def area_triangle : ℝ := (1 / 2) * triangle_leg_length ^ 2
def total_area_triangles : ℝ := 2 * area_triangle
def area_yard : ℝ := yard_length * yard_width

-- Fraction of the yard occupied by flower beds
def fraction_flower_beds : ℝ := total_area_triangles / area_yard

theorem flower_beds_fraction :
  fraction_flower_beds = 9 / 40 :=
sorry

end flower_beds_fraction_l41_41975


namespace reflections_composition_rotation_l41_41516

variable {α : ℝ} -- defining the angle α
variable {O : ℝ × ℝ} -- defining the point O, assuming the plane is represented as ℝ × ℝ

-- Define the lines that form the sides of the angle
variable (L1 L2 : ℝ × ℝ → Prop)

-- Assume α is the angle between L1 and L2 with O as the vertex
variable (hL1 : (L1 O))
variable (hL2 : (L2 O))

-- Assume reflections across L1 and L2
def reflect (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem reflections_composition_rotation :
  ∀ A : ℝ × ℝ, (reflect (reflect A L1) L2) = sorry := 
sorry

end reflections_composition_rotation_l41_41516


namespace cricket_players_l41_41790

theorem cricket_players (B C n both : ℕ) (hB : B = 600) (hn : n = 880) (hboth : both = 220)
    (h_union : n = C + B - both) : C = 500 :=
by
  rw [hB, hn, hboth] at h_union
  linarith

end cricket_players_l41_41790


namespace yogurt_combinations_l41_41635

open Nat

theorem yogurt_combinations :
  let flavors := 5
  let toppings := 7
  let no_topping := 1
  let one_topping := (choose toppings 1)
  let two_toppings := (choose toppings 2)
  let topping_combinations := no_topping + one_topping + two_toppings
  let total_combinations := flavors * topping_combinations
  total_combinations = 145 :=
by
  let flavors := 5
  let toppings := 7
  let no_topping := 1
  let one_topping := (choose toppings 1)
  let two_toppings := (choose toppings 2)
  let topping_combinations := no_topping + one_topping + two_toppings
  let total_combinations := flavors * topping_combinations
  exact (by decide : total_combinations = 145)

end yogurt_combinations_l41_41635


namespace percent_brandA_in_mix_l41_41616

theorem percent_brandA_in_mix (x : Real) :
  (0.60 * x + 0.35 * (100 - x) = 50) → x = 60 :=
by
  intro h
  sorry

end percent_brandA_in_mix_l41_41616


namespace difference_of_distinct_members_l41_41409

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41409


namespace smallest_k_multiple_of_360_l41_41302

theorem smallest_k_multiple_of_360 :
  ∃ (k : ℕ), (k > 0 ∧ (k = 432) ∧ (2160 ∣ k * (k + 1) * (2 * k + 1))) :=
by
  complication_sorry_proved

end smallest_k_multiple_of_360_l41_41302


namespace min_value_96_l41_41844

noncomputable def min_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) : ℝ :=
x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

theorem min_value_96 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) :
  min_value x y z h_pos h_xyz = 96 :=
sorry

end min_value_96_l41_41844


namespace find_a_of_extremum_l41_41449

theorem find_a_of_extremum (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : f x = x^3 + a*x^2 + b*x + a^2)
  (h2 : f' x = 3*x^2 + 2*a*x + b)
  (h3 : f' 1 = 0)
  (h4 : f 1 = 10) : a = 4 := by
  sorry

end find_a_of_extremum_l41_41449


namespace plane_eq_of_point_and_parallel_l41_41295

theorem plane_eq_of_point_and_parallel
    (A B C D : ℤ)
    (P : ℤ × ℤ × ℤ)
    (x y z : ℤ)
    (hx : A = 3) (hy : B = -2) (hz : C = 4) (hP : P = (2, -3, 1))
    (h_parallel : ∀ x y z, A * x + B * y + C * z = 5):
    A * 2 + B * (-3) + C * 1 + D = 0 ∧ A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 :=
by
  sorry

end plane_eq_of_point_and_parallel_l41_41295


namespace problem_equivalent_to_solution_l41_41064

def polarToRectCurve (ρ θ : ℝ) : Prop :=
  ρ^2 = 3 / (1 + 2 * sin θ ^ 2)

def rectCurve (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

def polarToRectPoint (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * cos θ, ρ * sin θ)

def pointR : Prop :=
  polarToRectPoint (2 * sqrt (2 : ℝ)) (π / 4) = (2, 2)

def isMinPerimeter (P : ℝ × ℝ) (perimeter : ℝ) : Prop :=
  let (Px, Py) := P
  Px * Px + Py * Py = 1 / (3 : ℝ) ∧
  perimeter = 4 ∧
  P = (3 / 2, 1 / 2)

theorem problem_equivalent_to_solution :
  (∀ (ρ θ : ℝ), polarToRectCurve ρ θ → ∃ (x y : ℝ), rectCurve x y) ∧
  (∃ (ρ θ : ℝ), polarToRectPoint (2 * sqrt (2 : ℝ)) (π / 4) = (2, 2)) ∧
  (∃ (P : ℝ × ℝ) (perimeter : ℝ), isMinPerimeter P perimeter)
:= by
  sorry

end problem_equivalent_to_solution_l41_41064


namespace arrange_descending_order_l41_41834

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem arrange_descending_order : c > a ∧ a > b := by
  sorry

end arrange_descending_order_l41_41834


namespace find_a_l41_41576

theorem find_a (a b c d : ℕ) (h1 : 2 * a + 2 = b) (h2 : 2 * b + 2 = c) (h3 : 2 * c + 2 = d) (h4 : 2 * d + 2 = 62) : a = 2 :=
by
  sorry

end find_a_l41_41576


namespace minimize_centroid_l41_41636

/-- ABC is a triangle with side lengths a = BC, b = CA, c = AB.
    G is the centroid of the triangle.
    Prove that the point P that minimizes AP * AG + BP * BG + CP * CG is G,
    and that this minimum value is (a^2 + b^2 + c^2) / 3.
-/
theorem minimize_centroid (a b c : ℝ) (A B C G : ℝ) :
  let AP := dist a P,
      AG := dist a G,
      BP := dist b P,
      BG := dist b G,
      CP := dist c P,
      CG := dist c G,
      G := centroid a b c in
  AP * AG + BP * BG + CP * CG = (a^2 + b^2 + c^2) / 3 := 
sorry

end minimize_centroid_l41_41636


namespace find_p0_l41_41100

-- Define the assumptions and the polynomial properties
def p0_is_correct (p : ℝ → ℝ) : Prop :=
  (∀ n ∈ (Finset.range 7), p(3^n) = 1 / 3^n) ∧ polynomial.degree p = (6 : with_bot ℕ) ∧ p 0 = 6560 / 2187

-- Detailed statement of the problem
theorem find_p0 (p : ℝ → ℝ) (hp : p0_is_correct p) : p(0) = 6560 / 2187 :=
by
  sorry

end find_p0_l41_41100


namespace part1_l41_41691

theorem part1 (f : ℚ → ℚ) : (∀ x : ℚ, f(x+1) = x^2 - 3x + 2) → (∀ x : ℚ, f x = x^2 - 5x + 6) :=
by sorry

end part1_l41_41691


namespace find_a_l41_41035

theorem find_a (a : ℝ) (h : 0.005 * a = 65) : a = 13000 / 100 :=
by
  sorry

end find_a_l41_41035


namespace addition_example_l41_41217

theorem addition_example : 36 + 15 = 51 := 
by
  sorry

end addition_example_l41_41217


namespace diff_of_two_distinct_members_l41_41401

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41401


namespace trigonometric_identity_tan_two_l41_41331

theorem trigonometric_identity_tan_two (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 :=
by
  sorry

end trigonometric_identity_tan_two_l41_41331


namespace f_sec2_l41_41046

variable (f : ℝ → ℝ)

axiom f_property : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (x / (x - 1)) = 1 / x

theorem f_sec2 (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : f (sec θ ^ 2) = sin θ ^ 2 :=
sorry

end f_sec2_l41_41046


namespace exist_similar_nonidentical_rectangles_l41_41820

-- Define a structure for rectangles where a rectangle is specified by its length and width
structure Rectangle where
  length : ℝ
  width : ℝ

-- Two rectangles are similar if their length-to-width ratios are the same
def similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Define a proposition stating the problem
theorem exist_similar_nonidentical_rectangles :
  ∃ r : Rectangle, ∃ (rects : list Rectangle), rects.length = 100 ∧
  ∀ (r1 r2 : Rectangle), r1 ∈ rects → r2 ∈ rects → r1 ≠ r2 →
  similar r1 r ∧ similar r2 r ∧ r1 ≠ r2 := by
  sorry

end exist_similar_nonidentical_rectangles_l41_41820


namespace total_metal_rods_needed_l41_41229

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end total_metal_rods_needed_l41_41229


namespace circle_equation_l41_41733

theorem circle_equation
    (symmetry_y_axis : ∀x y, (x, y) ∈ C → (-x, y) ∈ C)
    (passes_through_1_0 : (1, 0) ∈ C)
    (arc_length_ratio : ∀A B, A.1 = B.1 = 0 → arc_length_ratio (A, B, C) = 1 / 2) :
    C = { (x, y) | x^2 + (y + sqrt(3)/3)^2 = 4/3 ∨ x^2 + (y - sqrt(3)/3)^2 = 4/3 } := 
sorry

end circle_equation_l41_41733


namespace prod_ge_27_eq_iff_equality_l41_41501

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
          (h4 : a + b + c + 2 = a * b * c)

theorem prod_ge_27 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
by sorry

theorem eq_iff_equality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : 
  ((a + 1) * (b + 1) * (c + 1) = 27) ↔ (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end prod_ge_27_eq_iff_equality_l41_41501


namespace angle_C_congruent_triangles_l41_41438

theorem angle_C_congruent_triangles :
  ∀ (A B C E F G : Type) [∀ x, has_measured_angle x] [is_congruent_triangle ABC EFG] 
  (hB : measure_angle B = 30) (hFG : measure_angle F + measure_angle G = 110), 
  measure_angle C = 80 := 
by 
  sorry

end angle_C_congruent_triangles_l41_41438


namespace number_of_differences_l41_41386

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41386


namespace trajectory_is_ellipse_exists_fixed_point_and_const_value_l41_41720

-- Definitions from the problem conditions
def circle (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 + 4*x + 4 - 4*m^2 = 0
def point_N : ℝ × ℝ := (2, 0)
def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- The two parts of the proof problem
theorem trajectory_is_ellipse (m : ℝ) (h : m > 2) : ∃ a b, ellipse x y m (sqrt (m^2 - 4)) :=
sorry

theorem exists_fixed_point_and_const_value (m : ℝ) (h : m = sqrt 5) :
  ∃ E : ℝ × ℝ, (E = (sqrt 30 / 3, 0) ∨ E = (-sqrt 30 / 3, 0)) ∧
  ∀ (A B : ℝ × ℝ), (chord C E A B) →
  (1 / |E - A|^2 + 1 / |E - B|^2 = 6) :=
sorry

end trajectory_is_ellipse_exists_fixed_point_and_const_value_l41_41720


namespace balls_in_boxes_l41_41435

theorem balls_in_boxes (balls boxes : Type) (f : balls → boxes) (b1 b2 b3 b4 : balls) (A B C D : boxes) :
  b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (f = λ b, ite (b = b1) (ite (f b1 = A) B (f b1)) (f b)) ∧ (∀ b != b1, true) →
  (∃ (f : balls → boxes), true) → 
  ∃ f, (f b1 ≠ A ∧ ∃ b2 b3 b4, f b2 = B ∧ f b3 = C ∧ f b4 = D) ∧ 
  (f ≠ λ b, if b = b1 then A else f b) →
  (finset univ.b1 (b1 b2 b3 b4) = 192) :=
by sorry

end balls_in_boxes_l41_41435


namespace range_of_a_l41_41781

theorem range_of_a (h : ¬ ∃ x₀ ∈ set.Icc (-1 : ℝ) 2, x₀ - a > 0) : 
  ∀ a : ℝ, a ≥ 2 := 
sorry

end range_of_a_l41_41781


namespace exists_diameter_difference_leq_900_l41_41263

-- Define the main parameters of the problem: a regular 100-sided polygon
-- with vertices assigned values and the requirement.
theorem exists_diameter_difference_leq_900
  (a : Fin 100 → ℕ)
  (h100 : ∀ i, 100 ≤ a i ∧ a i ≤ 999) :
  ∃ i : Fin 100, |(∑ j in Finset.range 50, a (i + j)) - (∑ j in Finset.range 50, a (i + j + 50))| ≤ 900 :=
sorry

end exists_diameter_difference_leq_900_l41_41263


namespace compare_functions_l41_41992

noncomputable def f_A (x : ℤ) : ℤ :=
  x * x

noncomputable def g_A : ℤ → ℤ :=
  λ x, if x = 0 then 0 else 1

noncomputable def f_B (x : ℝ) : ℝ :=
  x * abs x

noncomputable def g_B : ℝ → ℝ :=
  λ x, if x ≥ 0 then x * x else - (x * x)

noncomputable def f_C (x : ℝ) : ℝ :=
  x

noncomputable def g_C (x : ℝ) : ℝ :=
  real.sqrt (x * x)

noncomputable def f_D (x : ℝ) [x > 0] : ℝ :=
  1 / x

noncomputable def g_D (x : ℝ) [x > 0] : ℝ :=
  (x + 1) / (x * x + x)

theorem compare_functions :
  (∀ x : ℤ, x ∈ {-1, 0, 1} → f_A x = g_A x) ∧ 
  (∀ x : ℝ, f_B x = g_B x) ∧ 
  (∃ x : ℝ, f_C x ≠ g_C x) ∧ 
  (∀ x : ℝ, x > 0 → f_D x = g_D x) :=
by sorry

end compare_functions_l41_41992


namespace fifth_equation_sum_first_17_even_sum_even_28_to_50_l41_41859

-- Define a function to sum the first n even numbers
def sum_even (n : ℕ) : ℕ := n * (n + 1)

-- Part (1) According to the pattern, write down the ⑤th equation
theorem fifth_equation : sum_even 5 = 30 := by
  sorry

-- Part (2) Calculate according to this pattern:
-- ① Sum of first 17 even numbers
theorem sum_first_17_even : sum_even 17 = 306 := by
  sorry

-- ② Sum of even numbers from 28 to 50
theorem sum_even_28_to_50 : 
  let sum_even_50 := sum_even 25
  let sum_even_26 := sum_even 13
  sum_even_50 - sum_even_26 = 468 := by
  sorry

end fifth_equation_sum_first_17_even_sum_even_28_to_50_l41_41859


namespace differences_of_set_l41_41382

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41382


namespace other_root_of_quadratic_l41_41113

theorem other_root_of_quadratic :
  ∀ (k : ℂ), k = -73 + 24 * Complex.I → 
  ∀ (z : ℂ), z = 4 + 7 * Complex.I → 
  ∃ w : ℂ, w = -4 - 7 * Complex.I ∧ w^2 = k :=
begin
  intros k hk z hz,
  use [-4 - 7 * Complex.I],
  split,
  { refl },
  { rw hz,
    simp,
    rw [←neg_mul_eq_neg_mul_symm, ←sq],
    -- it will simplify the proof steps with sorry as instructed
    sorry
  }
end

end other_root_of_quadratic_l41_41113


namespace num_diff_positive_integers_l41_41370

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41370


namespace largest_observed_angle_l41_41345

-- Definitions of problem conditions
variables {A B C : Type} [Add Bounded Lattice C]

def point := (ℝ × ℝ)  -- Suppose a point is represented by a 2D real plane.

def is_perpendicular (line1 line2: point): Prop := sorry  -- Define perpendicular line iff line1 and line2 are at 90 degrees to each other

def midpoint (A B: point): point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def largest_angle (A B C: point) (parallel_line: point → Prop): ℝ :=
  ∠ ACB -- Largest angle formed by segment AB from points on the parallel line

-- The proposition we need to prove
theorem largest_observed_angle (A B: point) (e: point → Prop):
  (parallel e AB) →
  (∃ I, midpoint A B = I ∧ is_perpendicular AB I ∧ e I) →
  ∀ P, e P → (∠ ACB ≥ ∠ APB) :=
begin
  sorry
end

end largest_observed_angle_l41_41345


namespace triangle_max_area_proof_l41_41456

open Real

noncomputable def triangle_max_area (A B C : ℝ) (AB : ℝ) (tanA tanB : ℝ) : Prop :=
  AB = 4 ∧ tanA * tanB = 3 / 4 → ∃ S : ℝ, S = 2 * sqrt 3

theorem triangle_max_area_proof (A B C : ℝ) (tanA tanB : ℝ) (AB : ℝ) : 
  triangle_max_area A B C AB tanA tanB :=
by
  sorry

end triangle_max_area_proof_l41_41456


namespace numberOfThreeDigitNumbersWithAtLeastOne5Or8_l41_41426

def isThreeDigitWholeNumber (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 999

def containsAtLeastOne5or8 (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10]
  digits.any (λ d, d = 5 ∨ d = 8)

theorem numberOfThreeDigitNumbersWithAtLeastOne5Or8 : 
  (count (λ n, isThreeDigitWholeNumber n ∧ containsAtLeastOne5or8 n) (range 1000)) = 452 := 
sorry

end numberOfThreeDigitNumbersWithAtLeastOne5Or8_l41_41426


namespace evaluate_expression_l41_41658

theorem evaluate_expression : 
  3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 :=
by
  sorry

end evaluate_expression_l41_41658


namespace winning_candidate_percentage_l41_41472

theorem winning_candidate_percentage
  (total_votes : ℕ)
  (vote_majority : ℕ)
  (winning_candidate_votes : ℕ)
  (losing_candidate_votes : ℕ) :
  total_votes = 400 →
  vote_majority = 160 →
  winning_candidate_votes = total_votes * 70 / 100 →
  losing_candidate_votes = total_votes - winning_candidate_votes →
  winning_candidate_votes - losing_candidate_votes = vote_majority →
  winning_candidate_votes = 280 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end winning_candidate_percentage_l41_41472


namespace transformed_sin_eq_l41_41182

theorem transformed_sin_eq :
  (∀ x, sin (2 * (x + π/12)) = sin (2 * x + π/6)) :=
by sorry

end transformed_sin_eq_l41_41182


namespace minimum_value_expression_equality_case_l41_41836

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c) ≥ 343 :=
sorry

theorem equality_case :
  (a b c : ℝ) (h : a = 1 ∧ b = 1 ∧ c = 1) →
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c) = 343 :=
sorry

end minimum_value_expression_equality_case_l41_41836


namespace triangle_third_side_l41_41340

theorem triangle_third_side (a b c : ℕ) (h₀ : a = 10) (h₁ : b = 5) :
  (c > abs (a - b) ∧ c < a + b) → c = 8 :=
by
  intros h
  rw [h₀, h₁] at h
  cases h with h₂ h₃
  sorry

end triangle_third_side_l41_41340


namespace find_irrational_l41_41647

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem find_irrational : 
  ∃ x ∈ ({-2, 1/2, real.sqrt 3, 2} : set ℝ), is_irrational x ∧ ∀ y ∈ ({-2, 1/2, real.sqrt 3, 2} : set ℝ), y ≠ x → is_rational y :=
by {
  sorry
}

end find_irrational_l41_41647


namespace curve_lattice_points_l41_41120

theorem curve_lattice_points (p : ℕ) (x y : ℤ) (h1 : y ≠ 0) (hp : Nat.Prime p) (oddp : p % 2 = 1)
  (hL : 4 * y^2 = (x - p) * p) : ∃ (infinite_lattice_points : ∀ n : ℕ, ∃ x y : ℤ, 4 * y^2 = (x - p) * p ∧ y ≠ 0 ∧ n = n) ∧
  ¬ ∃ (x y : ℤ), 4 * y^2 = (x - p) * p ∧ y ≠ 0 ∧ (∃ (d : ℤ), d = (x^2 + y^2).sqrt ∧ d ≠ 0 ∧ (d = x ∨ d = y)) :=
sorry

end curve_lattice_points_l41_41120


namespace differences_of_set_l41_41381

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41381


namespace number_of_distinct_positive_differences_l41_41420

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41420


namespace inverse_proportionality_decrease_l41_41098

theorem inverse_proportionality_decrease (x y : ℝ) (k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y = k) :
  let x1 := 1.1 * x in
  ∃ d : ℝ, y - (k / x1) = d / 100 * y ∧ d = 100 / 11 := sorry

end inverse_proportionality_decrease_l41_41098


namespace line_angle_135_l41_41152

-- Defining the points
def origin : (ℝ × ℝ) := (0, 0)
def point : (ℝ × ℝ) := (-1, -1)

-- Define the function to compute the slope
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Slope condition for the given points
def line_slope := slope origin point

-- Define the expected angle based on the slope
def angle_from_slope (m : ℝ) : ℝ :=
  if m > 0 then 135 else if m < 0 then 45 else 90 -- simplistic check for the specific case

-- The main theorem to prove
theorem line_angle_135 :
  angle_from_slope line_slope = 135 :=
by
  sorry

end line_angle_135_l41_41152


namespace hog_cat_problem_l41_41173

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end hog_cat_problem_l41_41173


namespace fraction_of_goose_eggs_hatched_is_one_l41_41266

-- Definitions based on conditions
def total_geese_hatched (x : ℝ) : ℝ := x
def geese_survived_first_month (x : ℝ) : ℝ := x * (3 / 4)
def geese_survived_first_year (x : ℝ) : ℝ := (x * (3 / 4)) * (2 / 5)

-- The given condition that 180 geese survived the first year
def survived_first_year_is_180 (x : ℝ) : Prop := geese_survived_first_year x = 180

-- The value of x to be proved
def fraction_hatched (x : ℝ) : Prop := x = 600

-- Theorem statement
theorem fraction_of_goose_eggs_hatched_is_one:
  ∃ x : ℝ, (survived_first_year_is_180 x) → fraction_hatched x :=
by {
  sorry
}

end fraction_of_goose_eggs_hatched_is_one_l41_41266


namespace tangent_line_eq_intervals_of_monotonicity_l41_41713

noncomputable theory

def f (x : ℝ) : ℝ := 2 * f (-x) - x^2 - 12 * x - 1

def g (x : ℝ) : ℝ := f x * real.exp (-x)

-- Prove the tangent line equation
theorem tangent_line_eq :
  let f_x := (λ x : ℝ, x^2 - 4 * x + 1)
  ∃ t : ℝ, f_x 0 = t ∧ t = 1 ∧ (∀ x, ∃ fx : ℝ, f x = fx) → ∀ (f'_x := (λ x : ℝ, 2*x-4)), f'_x 0 = -4 ∧ eq_of_tangent_line f 0 = -4 * x + 1 :=
by
  sorry

-- Prove the intervals of monotonicity for g(x)
theorem intervals_of_monotonicity :
  let f_x := (λ x : ℝ, x^2 - 4 * x + 1)
  ∀ x : ℝ, ∃ fx : ℝ, f x = fx → (g' x = (-x^2 + 6 * x - 5) * real.exp (-x)) →
  (∀ x : ℝ, 1 < x ∧ x < 5 → g' x > 0) ∧ 
  (∀ x : ℝ, (x < 1 ∨ 5 < x) → g' x < 0) :=
by
  sorry

end tangent_line_eq_intervals_of_monotonicity_l41_41713


namespace triangle_proof_l41_41457

variable (a b : ℝ) (cosC : ℝ) (SinC : ℝ) (Area : ℝ)

noncomputable def find_b (a : ℝ) (cosC : ℝ) (Area : ℝ) : ℝ := 
  let sinC := Real.sqrt (1 - cosC^2)
  2 * Area / (a * sinC)

theorem triangle_proof (h1 : a = 3 * Real.sqrt 2)
                      (h2 : cosC = 1 / 3)
                      (h3 : Area = 4 * Real.sqrt 3)
                      (h4 : SinC = Real.sqrt (1 - (1/3)^2)) :
  b = find_b 3 (1/3) (4 * Real.sqrt 3) :=
by
  rw [h1, h2, h3]
  let sinC := Real.sqrt (1 - (1/3)^2)
  have h4 : sinC = 2 * Real.sqrt 2 / 3 := by
    calc 
      sinC = Real.sqrt (1 - (1/3)^2)   : by rw Real.sqrt_sub
          ... = Real.sqrt (1 - 1/9)    : by norm_num
          ... = Real.sqrt (8/9)        : by norm_num
          ... = (2 * Real.sqrt 2) / 3  : by simp [Real.sqrt_div, Real.sqrt_mul]
  rw [h4]
  let b := (2 * 4 * Real.sqrt 3) / (3 * Real.sqrt 2 * (2 * Real.sqrt 2 / 3))
  simp [find_b, Real.sqrt_mul, Real.sqrt_div]
  sorry

end triangle_proof_l41_41457


namespace decreasing_population_density_l41_41950

def Population (t : Type) : Type := t

variable (stable_period: Prop)
variable (infertility: Prop)
variable (death_rate_exceeds_birth_rate: Prop)
variable (complex_structure: Prop)

theorem decreasing_population_density :
  death_rate_exceeds_birth_rate → true := sorry

end decreasing_population_density_l41_41950


namespace factorial_expression_l41_41944

-- Each definition used in Lean 4 statement should only directly appear in the conditions problem in a)
-- Each condition in a) should be used as a definition in Lean 4
-- Add sorry to skip the proof

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_expression :
  (13! - 12! + 144) / 11! = 144 :=
by
  sorry

end factorial_expression_l41_41944


namespace restore_catalogue_numbers_impossible_l41_41638

theorem restore_catalogue_numbers_impossible :
  ¬ (∀ (f : ℕ → ℕ), (∀ (x y : ℕ), 2 ≤ x ∧ x ≤ 2000 ∧ 2 ≤ y ∧ y ≤ 2000 → f (x, y) = Nat.gcd x y) → (∀ a b, 2 ≤ a ∧ a ≤ 2000 ∧ 2 ≤ b ∧ b ≤ 2000 → a = b)) :=
by
  sorry

end restore_catalogue_numbers_impossible_l41_41638


namespace matrix_power_2023_l41_41828

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1/2, -Real.sqrt 3 / 2, 0],
    ![Real.sqrt 3 / 2, 1/2, 0],
    ![0, 0, 1]
  ]

theorem matrix_power_2023 :
    (A ^ 2023) = ![
    ![-1, 0, 0],
    ![0, -1, 0],
    ![0, 0, 1]
    ] :=
  sorry

end matrix_power_2023_l41_41828


namespace f_log_sum_l41_41750

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 2

theorem f_log_sum (x : ℝ) : f (Real.log 5) + f (Real.log (1 / 5)) = 4 :=
by
  sorry

end f_log_sum_l41_41750


namespace find_AC_l41_41819

-- Definitions for the conditions
variables (A B C P Q M : Point)
variables [Triangle ABC]
variables [Segment AC]
variable (AP AQ : ℝ)
variables (AP_lt_AQ : AP < AQ)
variables (BP BQ : Line)
variables (divides_median : ∀ F H : Point, BP.intersect AM = F → BQ.intersect AM = H → AF = 1/3 * AM ∧ AH = 2/3 * AM)
variables (PQ : ℝ)

-- The main theorem statement
theorem find_AC (hPQ : PQ = 3) : AC = 10 := 
sorry

end find_AC_l41_41819


namespace number_of_diffs_l41_41361

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41361


namespace necessary_and_sufficient_condition_l41_41093

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (∃ x : ℝ, f a x < 0) ↔ |a| > 2 :=
by
  sorry

end necessary_and_sufficient_condition_l41_41093


namespace interest_rate_per_annum_is_4_l41_41623

-- Given conditions
def P : ℝ := 250
def t : ℝ := 8
def I : ℝ := P - 170

-- Formula for simple interest
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := (P * r * t) / 100

-- Main theorem
theorem interest_rate_per_annum_is_4 : simple_interest P r t = I → r = 4 :=
by sorry

end interest_rate_per_annum_is_4_l41_41623


namespace six_digit_numbers_with_zero_l41_41019

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41019


namespace red_fractions_32_less_than_blue_l41_41212

theorem red_fractions_32_less_than_blue : 
  ∀ fractions : List (ℚ), 
  fractions.length = 2016 → 
  fractions = List.range' 2 65. bind (λ n, List.range' 1 n |>.map (λ k, (k : ℚ) / n)) → 
  ∃ red blue : List ℚ, 
  (red ++ blue).length = fractions.length ∧
  red.length + blue.length = 2016 ∧
  red.forall (λ q, q < 1/2) ∧
  blue.forall (λ q, q ≥ 1/2) ∧
  blue.length = red.length + 32 := 
by sorry

end red_fractions_32_less_than_blue_l41_41212


namespace pairwise_parallel_determine_planes_l41_41783

def pairwise_parallel {ℝ : Type*} [inner_product_space ℝ (euclidean_space n)] (l1 l2 l3 : submodule ℝ (euclidean_space n)) :=
∀ i j ∈ (finset.univ : finset (fin 3)), i ≠ j → (∀ v1 v2 ∈ l1, v1 ⬝ v2 = 0)

theorem pairwise_parallel_determine_planes {ℝ : Type*} [inner_product_space ℝ (euclidean_space n)] 
  (l1 l2 l3 : submodule ℝ (euclidean_space n)) (h_parallel : pairwise_parallel l1 l2 l3) :
  ∃ (n : ℕ), n = 1 ∨ n = 3 :=
sorry

end pairwise_parallel_determine_planes_l41_41783


namespace perpendicular_to_plane_implies_perpendicular_to_all_lines_l41_41798

-- Defining the entities and properties based on the conditions
variable (m : Line) (α : Plane)

-- Condition: Line m is perpendicular to plane α
def is_perpendicular_to_plane (m : Line) (α : Plane) : Prop :=
  ∀ l : Line, (l ∈ α → m ⊥ l)

-- Proposition to prove: this relationship is sufficient but not necessary
theorem perpendicular_to_plane_implies_perpendicular_to_all_lines :
  (is_perpendicular_to_plane m α) → 
  (∀ l : Line, l ∈ α → m ⊥ l) ∧ ¬((∀ l : Line, l ∈ α → m ⊥ l) → is_perpendicular_to_plane m α) :=
by
  sorry

end perpendicular_to_plane_implies_perpendicular_to_all_lines_l41_41798


namespace equilateral_triangle_side_squared_l41_41661

noncomputable def polynomial_roots_equilateral_triangle 
  (p q r : ℂ) 
  (s t : ℂ) 
  (h_root_p : polynomial.eval p (polynomial.C t + polynomial.X * (polynomial.C s)) = 0)
  (h_root_q : polynomial.eval q (polynomial.C t + polynomial.X * (polynomial.C s)) = 0)
  (h_root_r : polynomial.eval r (polynomial.C t + polynomial.X * (polynomial.C s)) = 0)
  (h_norm : ∥p∥^2 + ∥q∥^2 + ∥r∥^2 = 300)
  (h_equilateral : (p - q).abs = (q - r).abs ∧ (q - r).abs = (r - p).abs) : ℂ :=
  let s_square : ℂ := 225
  in s_square

theorem equilateral_triangle_side_squared 
  (p q r : ℂ)
  (s t : ℂ)
  (h_root_p : polynomial.eval p (polynomial.C t + polynomial.X * (polynomial.C s)) = 0)
  (h_root_q : polynomial.eval q (polynomial.C t + polynomial.X * (polynomial.C s)) = 0)
  (h_root_r : polynomial.eval r (polynomial.C t + polynomial.X * (polynomial.C s)) = 0)
  (h_norm : ∥p∥^2 + ∥q∥^2 + ∥r∥^2 = 300)
  (h_equilateral : (p - q).abs = (q - r).abs ∧ (q - r).abs = (r - p).abs) : 
  polynomial_roots_equilateral_triangle p q r s t h_root_p h_root_q h_root_r h_norm h_equilateral = 225 :=
sorry

end equilateral_triangle_side_squared_l41_41661


namespace minimum_value_expression_equality_case_l41_41837

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c) ≥ 343 :=
sorry

theorem equality_case :
  (a b c : ℝ) (h : a = 1 ∧ b = 1 ∧ c = 1) →
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c) = 343 :=
sorry

end minimum_value_expression_equality_case_l41_41837


namespace range_of_2x_plus_y_l41_41711

theorem range_of_2x_plus_y {x y: ℝ} (h: x^2 / 4 + y^2 = 1) : -Real.sqrt 17 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 17 :=
sorry

end range_of_2x_plus_y_l41_41711


namespace find_irrational_l41_41645

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem find_irrational : 
  ∃ x ∈ ({-2, 1/2, real.sqrt 3, 2} : set ℝ), is_irrational x ∧ ∀ y ∈ ({-2, 1/2, real.sqrt 3, 2} : set ℝ), y ≠ x → is_rational y :=
by {
  sorry
}

end find_irrational_l41_41645


namespace unique_solution_eqn_l41_41689

theorem unique_solution_eqn (a b : ℝ) (H1 : 1 < a) (H2 : 0 < b) :
  (∃ t : ℝ, 1 < t ∧ a = t ∧ b = Real.exp(1) * Real.log t ∧ (∃! x : ℝ, 0 < x ∧ a^x = x^b)) ↔
  (a^Real.exp(1) = Real.exp(b) ∧ ∃! x : ℝ, 0 < x ∧ a^x = x^b ∧ x = Real.exp(1)) :=
begin
  sorry
end

end unique_solution_eqn_l41_41689


namespace inequality_sqrt_sum_l41_41936

theorem inequality_sqrt_sum (x : ℝ) (h : 0 < x ∧ x < 20) :
  sqrt x + sqrt (20 - x) ≤ 2 * sqrt 10 :=
by
  sorry

end inequality_sqrt_sum_l41_41936


namespace solve_for_x_l41_41528

-- Define the variables and conditions based on the problem statement
def equation (x : ℚ) := 5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)

-- State the theorem to be proved, including the condition and the result
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = 44.72727272727273 := by
  sorry  -- The proof is omitted

end solve_for_x_l41_41528


namespace find_d_l41_41034

variables (c : ℝ) (p : ℝ) (d : ℝ)

def root1 := -2
def root2 := 216 * c

def quadratic := p * (root1 ^ 2) + d * root1 - 1 = 0 ∧ p * (root2 ^ 2) + d * root2 - 1 = 0

theorem find_d (h : quadratic c p d) :
  d = -1 / (216 * c) + 1 / 2 :=
sorry

end find_d_l41_41034


namespace range_of_a_l41_41350

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x ≥ 4) :
  (∀ x : ℝ, f x = |x - 3| + |x - a|) → (a ≤ -1 ∨ a ≥ 7) :=
by 
  intros h1
  have h₂ := h 3
  specialize h1 3
  rw [abs_eq_zero.mpr (rfl : (3:ℝ) - 3 = 0), add_zero] at h1
  norm_cast
  exact sorry

end range_of_a_l41_41350


namespace solution_set_of_inequality_l41_41334

theorem solution_set_of_inequality 
  (f : ℝ → ℝ)
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_at_2 : f 2 = 0)
  (condition : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  {x : ℝ | x * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_of_inequality_l41_41334


namespace six_digit_numbers_with_zero_l41_41016

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41016


namespace min_separated_links_a_l41_41598

-- Given a chain with 60 links, each link weighing 1 gram
def chain_length_a := 60
def link_weight_a := 1

-- Prove that the minimum number of separated links to measure weights from 1g to chain_length_a * link_weight_a is 3
theorem min_separated_links_a : (∃ k : ℕ, k = 3 ∧ (∀ n ∈ (range chain_length_a + 1), ∃ parts : list ℕ, (sum parts = n) ∧ (∀ p ∈ parts, p ≤ chain_length_a ∧ p ∈ (map (pow 2) (range k)))))
  := sorry

end min_separated_links_a_l41_41598


namespace simplified_fraction_sum_l41_41885

theorem simplified_fraction_sum (n d : ℕ) (h_n : n = 144) (h_d : d = 256) : (9 + 16 = 25) := by
  have h1 : n = 2^4 * 3^2 := by sorry
  have h2 : d = 2^8 := by sorry
  have h3 : (n / gcd n d) = 9 := by sorry
  have h4 : (d / gcd n d) = 16 := by sorry
  exact rfl

end simplified_fraction_sum_l41_41885


namespace initial_snowflakes_l41_41932

variable (N : ℕ) -- Initial number of snowflakes.
variable (snowflakes_after_hour : ℕ) : 58
variable (additional_snowflakes_per_interval : ℕ) : 4
variable (minutes_per_interval : ℕ) : 5
variable (snow_after_one_hour : snowflakes_after_hour = N + 48)

theorem initial_snowflakes :
  N = 58 - 4 * (60 / 5) := by
  sorry

end initial_snowflakes_l41_41932


namespace sams_test_score_l41_41511

theorem sams_test_score :
  ∀ (students: ℕ) (initial_students: ℕ) (initial_avg: ℕ) (new_avg: ℕ) (sam_score: ℕ),
  (students = 20) →
  (initial_students = 19) →
  (initial_avg = 85) →
  (new_avg = 86) →
  sam_score = (new_avg * students - initial_avg * initial_students) →
  sam_score = 105 :=
by intro students initial_students initial_avg new_avg sam_score 
   intros H1 H2 H3 H4 H5
   rwa [H1, H2, H3, H4] at H5 
   exact H5


end sams_test_score_l41_41511


namespace six_digit_numbers_with_zero_l41_41004

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41004


namespace ratio_possible_values_valid_n_l41_41953

-- Part (a)

/-- Prove the possible values of the ratio b/a 
    for which arrangements of four points on the plane with pairwise distances a, b exist. -/
theorem ratio_possible_values (a b : ℝ) : 
    (a = 0 ∧ b = 0) ∨ 
    b / a = real.sqrt 3 ∨ 
    b / a = real.sqrt (2 + real.sqrt 3) ∨ 
    b / a = real.sqrt (2 - real.sqrt 3) ∨ 
    b / a = real.sqrt 2 ∨ 
    b / a = real.sqrt 3 / 3 ∨ 
    b / a = (1 + real.sqrt 5) / 2 := sorry

-- Part (b)

/-- 
  Validate the values of n for which arrangements 
  of n points on the plane with pairwise distances a or b exist.
--/
theorem valid_n (a b : ℝ) : set_of n ∈ ℕ ∧ 
    (∃ (a b ∈ ℝ) 
        ((∃ ABCD : set (ℝ × ℝ),
            card ABCD = 4 ∧ 
            (∀ p q ∈ ABCD, 
                p ≠ q → (dist p q = a ∨ dist p q = b) 
        )) ∨ 
        (n = 3 ∨ n = 4 ∨ n = 5) := sorry

end ratio_possible_values_valid_n_l41_41953


namespace average_sum_of_pairwise_distances_l41_41702

theorem average_sum_of_pairwise_distances :
  (∑ a in Finset.univ.Perm 12, |a 0 - a 1| + |a 2 - a 3| + 
    |a 4 - a 5| + |a 6 - a 7| + |a 8 - a 9| + |a 10 - a 11|) /
  12! = 12 :=
by
  sorry

end average_sum_of_pairwise_distances_l41_41702


namespace greatest_second_term_arithmetic_sequence_l41_41569

theorem greatest_second_term_arithmetic_sequence:
  ∃ a d : ℕ, (a > 0) ∧ (d > 0) ∧ (2 * a + 3 * d = 29) ∧ (4 * a + 6 * d = 58) ∧ (((a + d : ℤ) / 3 : ℤ) = 10) :=
sorry

end greatest_second_term_arithmetic_sequence_l41_41569


namespace five_less_than_sixty_percent_of_cats_l41_41175

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end five_less_than_sixty_percent_of_cats_l41_41175


namespace power_of_point_eq_tangent_squared_l41_41873

open EuclideanGeometry

noncomputable def power_of_point (P : Point) (O : Point) (R : Real) : Real :=
  let S : Circle := Circle.mk O R
  let T : Point := tangent_point P S
  tangent_length P T S * tangent_length P T S

theorem power_of_point_eq_tangent_squared (P O : Point) (R : Real) :
  distance P O > R →
  let S := Circle.mk O R in
  power_of_point P O R = (tangent_length P (tangent_point P S) S) ^ 2 := by
  sorry

end power_of_point_eq_tangent_squared_l41_41873


namespace car_rental_cost_l41_41682

theorem car_rental_cost (daily_rent : ℕ) (rent_duration : ℕ) (mileage_rate : ℚ) (mileage : ℕ) (total_cost : ℕ) :
  daily_rent = 30 → rent_duration = 5 → mileage_rate = 0.25 → mileage = 500 → total_cost = 275 :=
by
  intros hd hr hm hl
  sorry

end car_rental_cost_l41_41682


namespace magnitude_difference_half_b_l41_41342

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Conditions
def projection_condition : (a • b) = (1 / 2) * ∥b∥ ^ 2 := sorry
def a_magnitude : ∥a∥ = 1 := sorry
def b_magnitude : ∥b∥ = 1 := sorry

-- Theorem to Prove
theorem magnitude_difference_half_b : 
  (∥a - (1 / 2) • b∥ = (real.sqrt 3) / 2) :=
by
  -- Utilizing the given conditions
  have h1 : projection_condition := sorry,
  have h2 : a_magnitude := sorry,
  have h3 : b_magnitude := sorry,
  sorry

end magnitude_difference_half_b_l41_41342


namespace number_of_parallelograms_l41_41580

-- Define the main problem parameters and combinatorial function
def triangular_grid_parallelograms (n : ℕ) : ℕ :=
  3 * (Nat.choose (n + 2) 4)

-- The theorem to be proven
theorem number_of_parallelograms (n : ℕ) : 
  let grid := triangular_grid_parallelograms n in
  grid = 3 * (Nat.choose (n + 2) 4) :=
by
  -- Sorry placeholder as proof is not required
  sorry

end number_of_parallelograms_l41_41580


namespace factor_polynomial_l41_41899

noncomputable def polynomial1 (x y z : ℝ) : ℝ :=
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2)

noncomputable def polynomial2 (x y z : ℝ) : ℝ :=
  -(xy + xz + yz)

theorem factor_polynomial (x y z : ℝ) :
  polynomial1 x y z = (x - y) * (y - z) * (z - x) * polynomial2 x y z :=
by
  sorry

end factor_polynomial_l41_41899


namespace abs_inequality_range_l41_41149

theorem abs_inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a) → a ∈ Iic 4 := sorry

end abs_inequality_range_l41_41149


namespace roots_cubic_sum_of_cubes_l41_41089

theorem roots_cubic_sum_of_cubes (a b c : ℝ)
  (h1 : Polynomial.eval a (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h4 : a + b + c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 :=
by
  sorry

end roots_cubic_sum_of_cubes_l41_41089


namespace jon_spends_112_l41_41823

def price_per_coffee := 2
def total_days_in_april := 30
def mondays := 4
def fridays := 5
def earth_day := 1

def cost_of_regular_days := (total_days_in_april - mondays - fridays - earth_day) * (2 * price_per_coffee)
def cost_of_mondays := mondays * (price_per_coffee + price_per_coffee * 0.75)
def cost_of_fridays := fridays * (price_per_coffee + price_per_coffee * 0.5)
def cost_of_earth_day := earth_day * (price_per_coffee - 0.5) * 2

def total_cost := cost_of_regular_days + cost_of_mondays + cost_of_fridays + cost_of_earth_day

theorem jon_spends_112 : total_cost = 112 := by
  sorry

end jon_spends_112_l41_41823


namespace count_special_three_digit_numbers_l41_41428

-- Define the range for three-digit numbers
def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Define the predicate for at least one digit being 5 or 8
def contains_five_or_eight (n : ℕ) : Prop :=
  (∃ d ∈ to_digit_list n, d = 5 ∨ d = 8)

-- Collect the three-digit numbers that satisfy the condition
def filtered_numbers := {n : ℕ | n ∈ three_digit_numbers ∧ contains_five_or_eight n}

-- The theorem we want to prove
theorem count_special_three_digit_numbers : finset.card filtered_numbers = 452 :=
by
  sorry

end count_special_three_digit_numbers_l41_41428


namespace triangle_proof_l41_41851

noncomputable def triangle_proof_problem (A B C a b c AD : ℝ) (angle_bisector : AD = 2 * Real.sqrt 3) : Prop :=
  ∃ (triangle : {ABC : Type} [Triangle ABC A B C a b c]) (h1 : b + c = 2 * Real.sin (C + Real.pi / 6)), 
    (A = Real.pi / 3) ∧ 
    (let area := 1 / 2 * a * AD * (Real.sin (Real.pi / 3) / 2 + Real.sin (C / 2)) + 1 / 2 * b * AD * Real.sin (C / 2),
         equality when b = c) in area = 4 * Real.sqrt 3
  
theorem triangle_proof : triangle_proof_problem A B C a b c AD (AD = 2 * (Real.sqrt 3)) :=
  sorry

end triangle_proof_l41_41851


namespace ratio_A_to_B_l41_41124

variables (A_share B_share C_share : ℝ) 

theorem ratio_A_to_B :
  A_share = 420 ∧ B_share = 105 ∧ C_share = 70 ∧ B_share = (1 / 4) * C_share →
  A_share / B_share = 4 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  -- The detailed proof would go here, but we skip it.
  sorry

-- Checking the theorem assuming the variables' values.
example : ratio_A_to_B 420 105 70 :=
begin
  -- Use the provided values directly to satisfy the theorem conditions.
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  -- This is explicit check B's share is 1/4 of C's share.
  { calc
      105 = (1 / 4) * 70 : by norm_num
                      ... = 17.5            : by norm_num, 
  },
  -- The ratio check
  { calc
      420 / 105 = 4 : by norm_num, 
  }
end

end ratio_A_to_B_l41_41124


namespace donut_selection_count_l41_41871

theorem donut_selection_count :
  ∃ g c p s : ℕ, g + c + p + s = 5 ∧ 
  (set.univ.filter (λ x : ℕ × ℕ × ℕ × ℕ, x.fst.fst + x.fst.snd + x.snd.fst + x.snd.snd = 5)).card = 56 :=
by
  sorry

end donut_selection_count_l41_41871


namespace find_k_l41_41833

-- We define the necessary points and variables
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C P : V)

-- Definitions for G being the centroid of triangle ABC and D being the midpoint of BC
def G : V := (A + B + C) / 3
def D : V := (B + C) / 2

-- Theorem to prove the given relation holds for k = 9
theorem find_k (k : ℝ) (h : k = 9) :
  (P - A) • (P - A) + (P - D) • (P - D) + 3 * (P - C) • (P - C) = 
  k * ((P - G) • (P - G) + (G - D) • (G - D) + (G - C) • (G - C) + 4 * (G - A) • (G - A)) :=
by {
  -- This space should contain steps to show the LHS equals RHS. For now, we skip the proof.
  sorry
}

end find_k_l41_41833


namespace neq_one_of_quadratic_neq_zero_l41_41122

theorem neq_one_of_quadratic_neq_zero 
  {a b : ℝ} 
  (h : a^2 - b^2 + 2a - 4b - 3 ≠ 0) : 
  a - b ≠ 1 :=
sorry

end neq_one_of_quadratic_neq_zero_l41_41122


namespace limit_point_sequence_l41_41517

/-- Define the initial points T1, T2, T3, T4 -/
def T₁ : ℝ × ℝ := (1, 0)
def T₂ : ℝ × ℝ := (0, 1)
def T₃ : ℝ × ℝ := (-1, 0)
def T₄ : ℝ × ℝ := (0, -1)

/-- Define the recursive relation for the sequence Tₙ -/
def T (n : ℕ) : ℝ × ℝ :=
  if n = 0 then T₁
  else if n = 1 then T₂
  else if n = 2 then T₃
  else if n = 3 then T₄
  else let (xₙ₁, yₙ₁) := T (n - 4)
       let (xₙ₂, yₙ₂) := T (n - 3)
       in ((xₙ₁ + xₙ₂) / 2, (yₙ₁ + yₙ₂) / 2)

/-- Prove that the coordinates of the limit point of the sequence Tₙ as n → ∞ are (0, 0) -/
theorem limit_point_sequence (x y : ℝ) (n : ℕ) (hx : T n = (x, y)) :
  ∃ l : ℝ × ℝ, filter.tendsto (λ n, T n) filter.at_top (nhds l) ∧ l = (0, 0) := 
sorry

end limit_point_sequence_l41_41517


namespace theta_in_quadrant_I_or_III_l41_41330

theorem theta_in_quadrant_I_or_III (θ : ℝ) (h : sin θ * cos θ > 0) : 
  (0 < θ ∧ θ < π / 2) ∨ (π < θ ∧ θ < 3 * π / 2) :=
sorry

end theta_in_quadrant_I_or_III_l41_41330


namespace wood_and_rope_l41_41061

theorem wood_and_rope {x y : ℝ} (h1 : y = x + 4.5) (h2 : 0.5 * y = x - 1) : 
  (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by
  split
  · exact h1
  · exact h2

end wood_and_rope_l41_41061


namespace domain_of_f_l41_41545

noncomputable def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 3*x + 4) + real.log (x - 1)

theorem domain_of_f : set.Ioc 1 4 = {x : ℝ | (- x^2 + 3 * x + 4 > 0) ∧ (x - 1 > 0)} :=
by
  sorry

end domain_of_f_l41_41545


namespace measure_8_cm_measure_5_cm_1_measure_5_cm_2_l41_41866

theorem measure_8_cm:
  ∃ n : ℕ, n * (11 - 7) = 8 := by
  sorry

theorem measure_5_cm_1:
  ∃ x : ℕ, ∃ y : ℕ, x * ((11 - 7) * 2) - y * 7 = 5 := by
  sorry

theorem measure_5_cm_2:
  3 * 11 - 4 * 7 = 5 := by
  sorry

end measure_8_cm_measure_5_cm_1_measure_5_cm_2_l41_41866


namespace dependent_variable_l41_41051

-- Define the terms
def cost_per_box : ℕ := 10  -- the cost per box is 10 yuan
def total_cost (n : ℕ) : ℕ := n * cost_per_box  -- total cost function

-- Define the question as a theorem stating S is the dependent variable given the conditions 
theorem dependent_variable (n : ℕ) (S : ℕ) (h : S = total_cost n) : ∃ f, S = f n :=
by
  -- We need to prove that there exists a function f such that S is f(n)
  sorry  -- Proof should be provided here

end dependent_variable_l41_41051


namespace question1_question2_l41_41129

theorem question1 (x : ℝ) : (1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x) ↔ (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4) :=
by sorry

theorem question2 (x a : ℝ) : ((x - a)/(x - a^2) < 0)
  ↔ (a = 0 ∨ a = 1 → false)
  ∨ (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a)
  ∨ ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) :=
by sorry

end question1_question2_l41_41129


namespace train_length_is_correct_l41_41973

-- Define the speed in kmph
def speed_kmph : ℝ := 40

-- Define the time in seconds
def time_sec : ℝ := 7.199424046076314

-- Convert speed from kmph to m/s
def speed_mps : ℝ := (speed_kmph * 1000) / 3600

-- Define the length of the train (distance) as a function of speed and time
def length_of_train : ℝ := speed_mps * time_sec

-- Assert the length of the train is approximately 80 meters
theorem train_length_is_correct : length_of_train = 80 := by
  sorry

end train_length_is_correct_l41_41973


namespace triangle_area_l41_41634

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 ∧ 0.5 * a * b = 24 :=
by {
  sorry
}

end triangle_area_l41_41634


namespace order_of_three_numbers_l41_41912

-- Define the three numbers as real numbers
def five_power_six_tenths := 5 ^ 0.6
def zero_point_six_power_five := 0.6 ^ 5
def log_base_zero_point_six_five := Real.log 5 / Real.log 0.6 -- log_{0.6}(5) using change of base formula

-- Theorem statement
theorem order_of_three_numbers : zero_point_six_power_five < log_base_zero_point_six_five < five_power_six_tenths :=
sorry

end order_of_three_numbers_l41_41912


namespace number_of_differences_l41_41384

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41384


namespace books_in_pyramid_l41_41254

theorem books_in_pyramid :
  let L1 := 64 in
  let L2 := L1 / 0.8 in
  let L3 := L2 / 0.8 in
  let L4 := L3 / 0.8 in
  L1 + L2 + L3 + L4 = 369 :=
by
  sorry

end books_in_pyramid_l41_41254


namespace primes_between_50_and_60_l41_41763

theorem primes_between_50_and_60 : 
  (Finset.card (Finset.filter Nat.prime (Finset.range 61) \ Finset.range 51)) = 2 :=
by
  sorry

end primes_between_50_and_60_l41_41763


namespace transformed_sin_eq_l41_41527

theorem transformed_sin_eq : 
  ∀ (y : ℝ → ℝ), (y = λ x, sin (x + π / 6)) → 
  y = λ x, sin (2 * x + 5 * π / 12) :=
sorry

end transformed_sin_eq_l41_41527


namespace segment_length_l41_41189

theorem segment_length : ∀ (x : ℝ), |x - (8 : ℝ).cbrt| = 4 → (8 : ℝ) :=
by
  intro x h
  have h1 : x = (8 : ℝ).cbrt + 4 ∨ x = (8 : ℝ).cbrt - 4 := Real.abs_eq (x - (8 : ℝ).cbrt) 4 |>.1 h
  cases h1
  case inl (h2 : x = (8 : ℝ).cbrt + 4) =>
    rw [h2]
    -- ... (additional steps to explicitly calculate the length if needed)
    trivial
  case inr (h3 : x = (8 : ℝ).cbrt - 4) =>
    rw [h3]
    -- ... (additional steps to explicitly calculate the length if needed)
    trivial
  sorry

end segment_length_l41_41189


namespace det_A3_minus_3A_l41_41076

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 4], ![1, 3]]

theorem det_A3_minus_3A : 
  det (A ^ 3 - 3 • A) = -340 := 
sorry

end det_A3_minus_3A_l41_41076


namespace derivative_f_zeros_f_l41_41103

def f (a x : ℝ) : ℝ := (x^2 - 2*x) * log x + (a - 1 / 2) * x^2 + 2 * (1 - a) * x + a

theorem derivative_f (a : ℝ) (h : ∀ x > 0, deriv (λ x : ℝ, f a x) x = 2 * (x - 1) * (log x + a)) : 
  ∀ x, x > 0 → deriv (λ x : ℝ, f a x) x = 2 * (x - 1) * (log x + a) := 
  sorry

theorem zeros_f (a : ℝ) (h : a < -2) : ∃! x, f a x = 0 ∧ x ∈ (0, 1) ∪ {1} ∪ (1, exp (-a)) ∪ {exp (-a)} ∪ (exp (-a), +∞) :=
  sorry

end derivative_f_zeros_f_l41_41103


namespace john_cannot_achieve_goal_l41_41485

theorem john_cannot_achieve_goal
  (total_quizzes : ℕ := 60)
  (percentage_B_or_higher : ℤ := 90)
  (quizzes_completed : ℕ := 40)
  (quizzes_B_or_higher : ℕ := 32)
  (quizzes_remaining : ℕ := total_quizzes - quizzes_completed)
  (required_B_or_higher : ℕ := (percentage_B_or_higher * total_quizzes) / 100)
  (additional_B_or_higher_needed : ℕ := required_B_or_higher - quizzes_B_or_higher)
  (quizzes_to_go_below_B : ℤ := quizzes_remaining - additional_B_or_higher_needed)
  : quizzes_to_go_below_B < 0 := by
  sorry

end john_cannot_achieve_goal_l41_41485


namespace sqrt_three_irrational_l41_41644

-- Define what it means for a number to be irrational
def irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- Given numbers
def neg_two : ℝ := -2
def one_half : ℝ := 1 / 2
def sqrt_three : ℝ := real.sqrt 3
def two : ℝ := 2

-- The proof statement
theorem sqrt_three_irrational : irrational sqrt_three :=
sorry

end sqrt_three_irrational_l41_41644


namespace terminating_decimal_representation_count_l41_41306

theorem terminating_decimal_representation_count (n : ℕ) :
  1 ≤ n ∧ n ≤ 1000 → (∀ n, (1 ≤ n ∧ n ≤ 1000) → (n % 77 = 0) → n / 15400) → ∃ count : ℕ, count = 12 :=
by
  sorry

end terminating_decimal_representation_count_l41_41306


namespace cost_price_l41_41106

theorem cost_price (SP MP CP : ℝ) (discount_rate : ℝ) 
  (h1 : MP = CP * 1.15)
  (h2 : SP = MP * (1 - discount_rate))
  (h3 : SP = 459)
  (h4 : discount_rate = 0.2608695652173913) : CP = 540 :=
by
  -- We use the hints given as conditions to derive the statement
  sorry

end cost_price_l41_41106


namespace number_of_distinct_positive_differences_l41_41419

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41419


namespace geometric_figure_area_is_4_l41_41052

noncomputable def geometric_figure_area : ℝ :=
  let PQ := 2
  let ST := 2
  let PR := 2
  let QU := 2
  let RS := 2
  let TU := 2
  let angle_PQR := 75
  let angle_RST := 75
  let triangle_area := (ab: ℝ) (θ: ℝ) := 1/2 * ab^2 * Real.sin (θ * Real.pi / 180)
  let one_triangle_area := triangle_area PQ (30)
  4 * one_triangle_area

theorem geometric_figure_area_is_4 :
  geometric_figure_area = 4 := by
  sorry

end geometric_figure_area_is_4_l41_41052


namespace difference_of_distinct_members_l41_41410

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41410


namespace ratio_of_x_y_l41_41451

theorem ratio_of_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) : x / y = 22 / 7 :=
sorry

end ratio_of_x_y_l41_41451


namespace numberOfThreeDigitNumbersWithAtLeastOne5Or8_l41_41425

def isThreeDigitWholeNumber (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 999

def containsAtLeastOne5or8 (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10]
  digits.any (λ d, d = 5 ∨ d = 8)

theorem numberOfThreeDigitNumbersWithAtLeastOne5Or8 : 
  (count (λ n, isThreeDigitWholeNumber n ∧ containsAtLeastOne5or8 n) (range 1000)) = 452 := 
sorry

end numberOfThreeDigitNumbersWithAtLeastOne5Or8_l41_41425


namespace num_unique_differences_l41_41395

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41395


namespace cos2x_quadratic_eq_specific_values_l41_41850

variable (a b c x : ℝ)

axiom eqn1 : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0

noncomputable def quadratic_equation_cos2x 
  (a b c : ℝ) : ℝ × ℝ × ℝ := 
  (a^2, 2*a^2 + 2*a*c - b^2, a^2 + 2*a*c - b^2 + 4*c^2)

theorem cos2x_quadratic_eq 
  (a b c x : ℝ) 
  (h: a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0) :
  (a^2) * (Real.cos (2*x))^2 + 
  (2*a^2 + 2*a*c - b^2) * Real.cos (2*x) + 
  (a^2 + 2*a*c - b^2 + 4*c^2) = 0 :=
sorry

theorem specific_values : 
  quadratic_equation_cos2x 4 2 (-1) = (4, 2, -1) :=
by
  unfold quadratic_equation_cos2x
  simp
  sorry

end cos2x_quadratic_eq_specific_values_l41_41850


namespace difference_of_distinct_members_l41_41412

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41412


namespace quadratic_no_ten_powers_of_2_values_l41_41523

theorem quadratic_no_ten_powers_of_2_values 
  (a b : ℝ) :
  ¬ ∃ (j : ℤ), ∀ k : ℤ, j ≤ k ∧ k < j + 10 → ∃ n : ℕ, (k^2 + a * k + b) = 2 ^ n :=
by sorry

end quadratic_no_ten_powers_of_2_values_l41_41523


namespace sum_of_x_values_l41_41160

theorem sum_of_x_values :
  ∀ (x : ℝ), ((x - 3) * (x - 3)) = (1/3) * ((x - 2) * (x + 5))
    → (∑ s : set ℝ, s.to_finset.sum (λ x, x) = 10.5) :=
by
  -- Definition of areas and the given condition
  let sq_area := (x - 3) * (x - 3)
  let rect_area := (x - 2) * (x + 5)
  assume h : sq_area = (1/3) * rect_area
  sorry

end sum_of_x_values_l41_41160


namespace emma_interest_l41_41686

-- Define conditions as functions
def total_investment := 10000
def rate1 := 0.09
def rate2 := 0.11
def invest1 := 6000
def invest2 := total_investment - invest1

-- Define the interest calculations
def interest1 := invest1 * rate1
def interest2 := invest2 * rate2
def total_interest := interest1 + interest2

-- State the theorem to be proved
theorem emma_interest : total_interest = 980 := by
  sorry

end emma_interest_l41_41686


namespace num_two_digit_numbers_tens_greater_units_l41_41433

theorem num_two_digit_numbers_tens_greater_units : 
  let N := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n / 10 > n % 10)} in
  set.size N = 45 :=
by
  sorry

end num_two_digit_numbers_tens_greater_units_l41_41433


namespace diff_of_two_distinct_members_l41_41404

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41404


namespace right_triangle_side_length_l41_41797

theorem right_triangle_side_length (A B C : ℝ) (hC : ∠C = 90) (h_tanA : tan A = 3 / 4) (hAC : AC = 12) : AB = 15 :=
by
-- Using the given conditions and mathematical equivalence, we need to deduce AB = 15.
sorry

end right_triangle_side_length_l41_41797


namespace ordered_triples_2022_l41_41976

theorem ordered_triples_2022 :
  ∃ n : ℕ, n = 13 ∧ (∃ a c : ℕ, a ≤ c ∧ (a * c = 2022^2)) := by
  sorry

end ordered_triples_2022_l41_41976


namespace reduced_price_of_oil_l41_41593

theorem reduced_price_of_oil (P R : ℝ) (h1: R = 0.75 * P) (h2: 600 / (0.75 * P) = 600 / P + 5) :
  R = 30 :=
by
  sorry

end reduced_price_of_oil_l41_41593


namespace six_digit_numbers_with_zero_l41_41009

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41009


namespace solve_for_x_l41_41529

-- Definitions and conditions from a) directly 
def f (x : ℝ) : ℝ := 64 * (2 * x - 1) ^ 3

-- Lean 4 statement to prove the problem
theorem solve_for_x (x : ℝ) : f x = 27 → x = 7 / 8 :=
by
  intro h
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l41_41529


namespace num_diff_positive_integers_l41_41373

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41373


namespace irrational_number_is_sqrt3_l41_41640

theorem irrational_number_is_sqrt3 :
  ¬∃ (a b : ℤ), b ≠ 0 ∧ (√3 = a / b) ∧ 
  (∃ (c d : ℤ), d ≠ 0 ∧ (-2 = c / d)) ∧
  (∃ (e f : ℤ), f ≠ 0 ∧ (1 / 2 = e / f)) ∧
  (∃ (g h : ℤ), h ≠ 0 ∧ (2 = g / h)) :=
by {
  sorry
}

end irrational_number_is_sqrt3_l41_41640


namespace ones_digits_divisible_by_8_l41_41513

theorem ones_digits_divisible_by_8 : 
  ∃ d : ℕ, (∀ n : ℕ, (n % 8 = 0) → (d ∈ {0, 2, 4, 6, 8})) ∧ (d.card = 5) :=
sorry

end ones_digits_divisible_by_8_l41_41513


namespace new_solution_is_45_percent_liquid_x_l41_41128

-- Define initial conditions
def solution_y_initial_weight := 8.0 -- kilograms
def percent_liquid_x := 0.30
def percent_water := 0.70
def evaporated_water_weight := 4.0 -- kilograms
def added_solution_y_weight := 4.0 -- kilograms

-- Define the relevant quantities
def liquid_x_initial := solution_y_initial_weight * percent_liquid_x
def water_initial := solution_y_initial_weight * percent_water
def remaining_water_after_evaporation := water_initial - evaporated_water_weight

def liquid_x_after_evaporation := liquid_x_initial 
def water_after_evaporation := remaining_water_after_evaporation

def added_liquid_x := added_solution_y_weight * percent_liquid_x
def added_water := added_solution_y_weight * percent_water

def total_liquid_x := liquid_x_after_evaporation + added_liquid_x
def total_water := water_after_evaporation + added_water

def total_new_solution_weight := total_liquid_x + total_water

def new_solution_percent_liquid_x := (total_liquid_x / total_new_solution_weight) * 100

-- The theorem we want to prove
theorem new_solution_is_45_percent_liquid_x : new_solution_percent_liquid_x = 45 := by
  sorry

end new_solution_is_45_percent_liquid_x_l41_41128


namespace coeff_x3_expansion_l41_41782

theorem coeff_x3_expansion (n : ℕ) (h_n_pos : 0 < n) (sum_of_coeffs : (4 - 1)^n = 729) :
  binomial.coeff (4 * x - 1)^6 3 = -1280 :=
sorry

end coeff_x3_expansion_l41_41782


namespace num_second_grade_students_is_80_l41_41467

def ratio_fst : ℕ := 5
def ratio_snd : ℕ := 4
def ratio_trd : ℕ := 3
def total_students : ℕ := 240

def second_grade : ℕ := (ratio_snd * total_students) / (ratio_fst + ratio_snd + ratio_trd)

theorem num_second_grade_students_is_80 :
  second_grade = 80 := 
sorry

end num_second_grade_students_is_80_l41_41467


namespace solve_pears_and_fruits_l41_41803

noncomputable def pears_and_fruits_problem : Prop :=
  ∃ (x y : ℕ), x + y = 1000 ∧ (11 * x) * (1/9 : ℚ) + (4 * y) * (1/7 : ℚ) = 999

theorem solve_pears_and_fruits :
  pears_and_fruits_problem := by
  sorry

end solve_pears_and_fruits_l41_41803


namespace BE_eq_FD_l41_41063

-- Definitions and conditions
variable {A B C D O E F : Type}
variable [IsoscelesTriangle A B C]
variable [AngleBisector C D]
variable [Circumcenter O A B C]
variable [Perpendicular O E C D]
variable [Intersection E BC]
variable [Parallel E F C D]
variable [Intersection F AB]

-- Statement to prove
theorem BE_eq_FD (ABC_is_isosceles : A B C is_isosceles)
  (CD_is_angle_bisector : is_angle_bisector C D)
  (O_is_circumcenter : is_circumcenter O A B C)
  (OE_perpendicular_to_CD : perpendicular_to OE C D)
  (E_intersects_BC : intersects E B C)
  (EF_parallel_to_CD : parallel E F C D)
  (F_intersects_AB : intersects F A B) :
  BE = FD :=
sorry

end BE_eq_FD_l41_41063


namespace cube_4_edge_trips_l41_41359

-- Define vertices as data types
inductive Vertices
| A
| B
| C
| D
| E
| F
| G
| H

open Vertices

-- Define edges as pairs of vertices
def edges : set (Vertices × Vertices) :=
  {(A, C), (A, D), (A, E),
   (B, D), (B, F), (B, H),
   (C, A), (C, D), (C, G),
   (D, A), (D, B), (D, C), (D, H),
   (E, A), (E, F), (E, G),
   (F, B), (F, E), (F, H),
   (G, C), (G, E), (G, H),
   (H, B), (H, D), (H, F), (H, G)}

-- Define a function to count valid 4-edge trips
def count_4_edge_trips : Nat :=
  36

-- The theorem to be proved
theorem cube_4_edge_trips : count_4_edge_trips = 36 := by
  sorry

end cube_4_edge_trips_l41_41359


namespace distance_from_F_to_midpoint_DE_is_12_point_5_l41_41059

-- Define the lengths of the segments in the right triangle DEF
def DE : ℝ := 15
def DF : ℝ := 20

-- Define the hypotenuse EF using the Pythagorean theorem
def EF : ℝ := Real.sqrt (DE^2 + DF^2)

-- Calculate the distance from F to the midpoint of DE
def distance_from_F_to_midpoint_DE : ℝ := EF / 2

-- Prove that this distance is 12.5 units
theorem distance_from_F_to_midpoint_DE_is_12_point_5 :
  distance_from_F_to_midpoint_DE = 12.5 := by
  sorry

end distance_from_F_to_midpoint_DE_is_12_point_5_l41_41059


namespace inverse_B2_l41_41728

def matrix_B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 7; -2, -4]

def matrix_B2_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-5, -7; 2, 2]

theorem inverse_B2 (B : Matrix (Fin 2) (Fin 2) ℝ) (hB_inv : B⁻¹ = matrix_B_inv) :
  (B^2)⁻¹ = matrix_B2_inv :=
sorry

end inverse_B2_l41_41728


namespace sum_of_integers_product_eq_11_cubed_l41_41561

theorem sum_of_integers_product_eq_11_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 11^3 ∧ a + b + c = 133 :=
by
  -- The product condition is satisfied by the unique set {1, 11, 121}
  use 1, 11, 121
  -- Conditions required: a ≠ b, b ≠ c, a ≠ c
  simp [*, pow_succ]
  sorry

end sum_of_integers_product_eq_11_cubed_l41_41561


namespace self_describing_number_third_digit_l41_41218

theorem self_describing_number_third_digit :
  ∃ (n : Fin 7 → ℕ), 
    n 0 = 3 ∧ -- The first digit is the number of zeros in the number (3).
    n 1 = 2 ∧ -- The second digit is the number of ones in the number (2).
    (Finset.univ.sum (λ i, if n i = 0 then 1 else 0)) = 3 ∧ -- There are 3 zeros in the number.
    (Finset.univ.sum (λ i, if n i = 1 then 1 else 0)) = 2 ∧ -- There are 2 ones in the number.
    (Finset.univ.sum (λ i, if n i = 3 then 1 else 0)) = 1 ∧ -- There is 1 three in the number.
    (Finset.univ.sum (λ i, if n i = 4 then 1 else 0)) = 0 ∧ -- There are 0 fours in the number.
    n 2 = 1 := -- The third digit is the number of twos in the number (1).
sorry

end self_describing_number_third_digit_l41_41218


namespace solve_triangle_l41_41480

theorem solve_triangle :
  (a = 6 ∧ b = 6 * Real.sqrt 3 ∧ A = 30) →
  ((B = 60 ∧ C = 90 ∧ c = 12) ∨ (B = 120 ∧ C = 30 ∧ c = 6)) :=
by
  intros h
  sorry

end solve_triangle_l41_41480


namespace arc_length_of_parametric_curve_l41_41210

noncomputable def arc_length_parametric (x y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in t1..t2, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

def x (t : ℝ) := 2 * (Real.cos t) ^ 3
def y (t : ℝ) := 2 * (Real.sin t) ^ 3

theorem arc_length_of_parametric_curve : arc_length_parametric x y 0 (Real.pi / 4) = 3 / 2 := by
  sorry

end arc_length_of_parametric_curve_l41_41210


namespace find_total_buffaloes_l41_41054

-- Define the problem parameters.
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := 8

-- Define the conditions.
def duck_legs : ℕ := 2 * number_of_ducks
def cow_legs : ℕ := 4 * number_of_cows
def total_heads : ℕ := number_of_ducks + number_of_cows

-- The given equation as a condition.
def total_legs : ℕ := duck_legs + cow_legs

-- Translate condition from the problem:
def condition : Prop := total_legs = 2 * total_heads + 16

-- The proof statement.
theorem find_total_buffaloes : number_of_cows = 8 :=
by
  -- Place the placeholder proof here.
  sorry

end find_total_buffaloes_l41_41054


namespace ratio_CD_BD_l41_41818

-- Define the conditions and the final theorem
def Point := ℝ × ℝ

variable {A B C D E T: Point}

-- Let line_segment be a function to denote the points lying on a segment.
def line_segment (P Q R: Point) := ∃ t: ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1-t)•P + t•Q

-- Conditions
variable (h1 : line_segment B C D)
variable (h2 : line_segment A C E)
variable (h3 : ∃ t: Point, line_segment A D t ∧ line_segment B E t ∧ t = T)
variable (h4 : ∃ k: ℝ, k = 2 ∧ T = (k/(k+1))•A + (1/(k+1))•D)
variable (h5 : ∃ m: ℝ, m = 3 ∧ T = (m/(m+1))•B + (1/(m+1))•E)

-- Theorem to prove
theorem ratio_CD_BD : CD/BD = 4 :=
sorry

end ratio_CD_BD_l41_41818


namespace floor_of_s_l41_41493

noncomputable def g (x : ℝ) := 2 * Real.sin x + 3 * Real.cos x + 4 * (Real.cos x / Real.sin x)

theorem floor_of_s :
  (∃ s : ℝ, g s = 0 ∧ s > Real.pi / 2 ∧ s < Real.pi) → ⌊classical.some (exists_lt_of_linear_order s)⌋ = 1 :=
begin
  sorry
end

end floor_of_s_l41_41493


namespace find_a_l41_41352

theorem find_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x - 3 ≤ 0 ↔ x ∈ Icc (-1) 3) → a = -2 := sorry

end find_a_l41_41352


namespace original_cost_of_each_magazine_l41_41238

-- Definitions and conditions
def magazine_cost (C : ℝ) : Prop :=
  let total_magazines := 10
  let sell_price := 3.50
  let gain := 5
  let total_revenue := total_magazines * sell_price
  let total_cost := total_revenue - gain
  C = total_cost / total_magazines

-- Goal to prove
theorem original_cost_of_each_magazine : ∃ C : ℝ, magazine_cost C ∧ C = 3 :=
by
  sorry

end original_cost_of_each_magazine_l41_41238


namespace units_digit_l41_41698

noncomputable def C := 20 + Real.sqrt 153
noncomputable def D := 20 - Real.sqrt 153

theorem units_digit (h : ∀ n ≥ 1, 20 ^ n % 10 = 0) :
  (C ^ 12 + D ^ 12) % 10 = 0 :=
by
  -- Proof will be provided based on the outlined solution
  sorry

end units_digit_l41_41698


namespace value_of_abs_div_sum_l41_41729

theorem value_of_abs_div_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (|a| / a + |b| / b = 2) ∨ (|a| / a + |b| / b = -2) ∨ (|a| / a + |b| / b = 0) := 
by
  sorry

end value_of_abs_div_sum_l41_41729


namespace difference_of_distinct_members_l41_41411

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41411


namespace total_pokemon_cards_l41_41125

def initial_cards : Nat := 27
def received_cards : Nat := 41
def lost_cards : Nat := 20

theorem total_pokemon_cards : initial_cards + received_cards - lost_cards = 48 := by
  sorry

end total_pokemon_cards_l41_41125


namespace find_abc_l41_41492

-- Define the conditions as variables
variables {a b c : ℤ}

-- Define the polynomials
def poly1 : ℤ[X] := X^2 + a * X + b
def poly2 : ℤ[X] := X^2 + b * X + c

-- Define the gcd and lcm conditions
def gcd_condition : Polynomial ℤ → Polynomial ℤ → Prop := sorry  -- Define the gcd condition, placeholder
def lcm_condition (P : Polynomial ℤ) : Prop := P = X^3 - 3 * X^2 - 4 * X + 12

-- Translate the problem to a Lean statement
theorem find_abc (a b c : ℤ) (h1 : gcd_condition poly1 poly2) (h2 : lcm_condition (Polynomial.lcm poly1 poly2)) : a + b + c = -7 :=
by sorry

end find_abc_l41_41492


namespace quadratic_eqn_a_range_l41_41448

variable {a : ℝ}

theorem quadratic_eqn_a_range (a : ℝ) : (∃ x : ℝ, (a - 3) * x^2 - 4 * x + 1 = 0) ↔ a ≠ 3 :=
by sorry

end quadratic_eqn_a_range_l41_41448


namespace solution_value_a_l41_41771

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end solution_value_a_l41_41771


namespace cone_cylinder_volume_ratio_l41_41696

theorem cone_cylinder_volume_ratio (h r : ℝ) (hc_pos : h > 0) (r_pos : r > 0) :
  let V_cylinder := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * (3 / 4 * h)
  (V_cone / V_cylinder) = 1 / 4 := 
by 
  sorry

end cone_cylinder_volume_ratio_l41_41696


namespace number_of_valid_selections_l41_41273

theorem number_of_valid_selections : 
  ∃ combinations : Finset (Finset ℕ), 
    combinations = {
      {2, 6, 3, 5}, 
      {2, 6, 1, 7}, 
      {2, 4, 1, 5}, 
      {4, 1, 3}, 
      {6, 1, 5}, 
      {4, 6, 3, 7}, 
      {2, 4, 6, 5, 7}
    } ∧ combinations.card = 7 :=
by sorry

end number_of_valid_selections_l41_41273


namespace number_of_diffs_l41_41366

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41366


namespace train_speed_l41_41983

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_conversion_factor : ℝ) (expected_speed : ℝ) (h_time_conversion : time_conversion_factor = 1 / 60) (h_time : time_minutes / 60 = 0.5) (h_distance : distance = 51) (h_expected_speed : expected_speed = 102) : distance / (time_minutes / 60) = expected_speed :=
by 
  sorry

end train_speed_l41_41983


namespace subset_sum_polynomial_equality_l41_41487

-- Polynomial definition
structure Polynomial (R : Type) :=
(coeffs : List R)
(degree : ℕ)

-- Integer coefficients definition for Polynomial
def Polynomial.is_integer_coeff (P : Polynomial ℤ) : Prop :=
∀ c ∈ P.coeffs, c ∈ ℤ

-- Definition of S(A)
def S (P : Polynomial ℤ) (A : List ℕ) : ℤ :=
A.foldr (λ a acc, acc + (P.coeffs.sum * a ^ P.degree)) 0

-- Main theorem
theorem subset_sum_polynomial_equality
  (P : Polynomial ℤ) (hP : Polynomial.is_integer_coeff P) (d : ℕ)
  (m n : ℕ) (hmn : m ^ (d + 1) ∣ n) :
  ∃ (A : Fin (m+1) → List ℕ), 
  ∀ i, (i < m) → (A i).length = n / m ∧ S P (A i) = S P (A 0) :=
sorry

end subset_sum_polynomial_equality_l41_41487


namespace quadratic_factorization_l41_41742

theorem quadratic_factorization (p q x_1 x_2 : ℝ) (h1 : x_1 = 2) (h2 : x_2 = -3) 
    (h3 : x_1 + x_2 = -p) (h4 : x_1 * x_2 = q) : 
    (x - 2) * (x + 3) = x^2 + p * x + q :=
by
  sorry

end quadratic_factorization_l41_41742


namespace similar_triangle_length_l41_41183

theorem similar_triangle_length
  (similar : SimilarTriangles G H I X Y Z)
  (GH_length : GH = 8)
  (HI_length : HI = 20)
  (YZ_length : YZ = 25)
: XY = 80 := 
by sorry

end similar_triangle_length_l41_41183


namespace angle_between_planes_l41_41568

-- Define the conditions and required concepts
variables (a : ℝ) (h : ℝ)
-- Normalized vectors
def n1 : ℝ × ℝ × ℝ := (0, 0, 1)
def n2 : ℝ × ℝ × ℝ := ((1 / Real.sqrt 3), 0, (Real.sqrt 2 / Real.sqrt 3))

-- Prove the required angle
theorem angle_between_planes (a_pos : 0 < a) (h_eq : h = (a * Real.sqrt 2) / 2) :
  Real.arccos ((0, 0, 1) • ((1 / Real.sqrt 3), 0, (Real.sqrt 2 / Real.sqrt 3))) = Real.arctan (2 - Real.sqrt 3) :=
by {
  sorry
}

end angle_between_planes_l41_41568


namespace smallest_solution_of_quartic_l41_41942

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end smallest_solution_of_quartic_l41_41942


namespace factor_expression_l41_41292

theorem factor_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end factor_expression_l41_41292


namespace total_bees_including_queen_at_end_of_14_days_l41_41964

-- Conditions definitions
def bees_hatched_per_day : ℕ := 5000
def bees_lost_per_day : ℕ := 1800
def duration_days : ℕ := 14
def initial_bees : ℕ := 20000
def queen_bees : ℕ := 1

-- Question statement as Lean theorem
theorem total_bees_including_queen_at_end_of_14_days :
  (initial_bees + (bees_hatched_per_day - bees_lost_per_day) * duration_days + queen_bees) = 64801 := 
by
  sorry

end total_bees_including_queen_at_end_of_14_days_l41_41964


namespace work_problem_l41_41966

/-- 
  Suppose A can complete a work in \( x \) days alone, 
  B can complete the work in 20 days,
  and together they work for 7 days, leaving a fraction of 0.18333333333333335 of the work unfinished.
  Prove that \( x = 15 \).
 -/
theorem work_problem (x : ℝ) : 
  (∀ (B : ℝ), B = 20 → (∀ (f : ℝ), f = 0.18333333333333335 → (7 * (1 / x + 1 / B) = 1 - f)) → x = 15) := 
sorry

end work_problem_l41_41966


namespace inverse_48_l41_41027

variable (f : ℝ → ℝ)

def condition1 : Prop := f 5 = 3
def condition2 : Prop := ∀ x, f (2 * x) = 2 * f x

theorem inverse_48 : condition1 f → condition2 f → function.inverse f 48 = 80 :=
by
  sorry

end inverse_48_l41_41027


namespace mod_inverse_non_existence_mod_inverse_existence_l41_41274

theorem mod_inverse_non_existence (a b c d : ℕ) (h1 : 1105 = a * b * c) (h2 : 15 = d * a) :
    ¬ ∃ x : ℕ, (15 * x) % 1105 = 1 := by sorry

theorem mod_inverse_existence (a b : ℕ) (h1 : 221 = a * b) :
    ∃ x : ℕ, (15 * x) % 221 = 59 := by sorry

end mod_inverse_non_existence_mod_inverse_existence_l41_41274


namespace coefficient_a1b2c3_in_expansion_l41_41282

theorem coefficient_a1b2c3_in_expansion :
  let poly := (a + 2 * b - 3 * c)^6 
  coefficient_of_term (a^1 * b^2 * c^3) poly = -6480 :=
by
  sorry

end coefficient_a1b2c3_in_expansion_l41_41282


namespace power_mod_l41_41677

theorem power_mod (n : ℕ) : 3^100 % 7 = 4 := by
  sorry

end power_mod_l41_41677


namespace assume_dead_heat_race_l41_41205

variable {Va Vb L H : ℝ}

theorem assume_dead_heat_race (h1 : Va = (51 / 44) * Vb) :
  H = (7 / 51) * L :=
sorry

end assume_dead_heat_race_l41_41205


namespace largest_two_digit_number_with_remainder_2_div_13_l41_41197

theorem largest_two_digit_number_with_remainder_2_div_13 : 
  ∃ (N : ℕ), (10 ≤ N ∧ N ≤ 99) ∧ N % 13 = 2 ∧ ∀ (M : ℕ), (10 ≤ M ∧ M ≤ 99) ∧ M % 13 = 2 → M ≤ N :=
  sorry

end largest_two_digit_number_with_remainder_2_div_13_l41_41197


namespace actual_distance_between_mountains_l41_41112

theorem actual_distance_between_mountains (D_map : ℝ) (d_map_ram : ℝ) (d_real_ram : ℝ)
  (hD_map : D_map = 312) (hd_map_ram : d_map_ram = 25) (hd_real_ram : d_real_ram = 10.897435897435898) :
  D_map / d_map_ram * d_real_ram = 136 :=
by
  -- Theorem statement is proven based on the given conditions.
  sorry

end actual_distance_between_mountains_l41_41112


namespace num_unique_differences_l41_41399

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41399


namespace find_r_l41_41704

theorem find_r 
  (r s : ℝ)
  (h1 : 9 * (r * r) * s = -6)
  (h2 : r * r + 2 * r * s = -16 / 3)
  (h3 : 2 * r + s = 2 / 3)
  (polynomial_condition : ∀ x : ℝ, 9 * x^3 - 6 * x^2 - 48 * x + 54 = 9 * (x - r)^2 * (x - s)) 
: r = -2 / 3 :=
sorry

end find_r_l41_41704


namespace perp_DM_OM_l41_41084

noncomputable def tetrahedron := sorry   -- Placeholders for geometric concepts
noncomputable def point := sorry
noncomputable def centroid (t : tetrahedron) : point := sorry
noncomputable def circumcenter (t : tetrahedron) : point := sorry
noncomputable def lies_on_sphere : point → point → Prop := sorry  -- denotes collinearity on sphere
noncomputable def perp (p1 p2 p3 : point) : Prop := sorry  -- denotes perpendicularity
noncomputable def medians_intersection (t : tetrahedron) (p : point) : point := sorry

theorem perp_DM_OM (t : tetrahedron) (D : point) (M : centroid t) (O : circumcenter t) : 
  (∀ p, (p = D ∨ p = M ∨ p = medians_intersection t D) → lies_on_sphere p O) → perp D M O :=
sorry

end perp_DM_OM_l41_41084


namespace years_required_l41_41168

def num_stadiums := 30
def avg_cost_per_stadium := 900
def annual_savings := 1500
def total_cost := num_stadiums * avg_cost_per_stadium

theorem years_required : total_cost / annual_savings = 18 :=
by
  sorry

end years_required_l41_41168


namespace non_collinear_unit_vectors_dot_product_not_one_l41_41356

variables {V : Type*} [inner_product_space ℝ V]

theorem non_collinear_unit_vectors_dot_product_not_one (e₁ e₂ : V) (θ : ℝ) 
  (h₁ : ∥e₁∥ = 1) (h₂ : ∥e₂∥ = 1) (h₃ : θ ≠ 0) (h₄ : θ ≠ π) :
  e₁ ≠ e₂ → (e₁ - e₂).orthogonal (e₁ + e₂) :=
by
  intro h
  sorry

end non_collinear_unit_vectors_dot_product_not_one_l41_41356


namespace range_of_y_div_x_l41_41717

-- Define the complex number and the condition that |z - 2| = √3
variables (x y : ℝ) (hx : x ≠ 0) (hmod : (x - 2)^2 + y^2 = 3)

-- Define the statement to be proved
theorem range_of_y_div_x (x y : ℝ) (hx : x ≠ 0) (hmod : (x - 2)^2 + y^2 = 3) :
  -real.sqrt 3 ≤ y / x ∧ y / x ≤ real.sqrt 3 :=
sorry

end range_of_y_div_x_l41_41717


namespace sandra_pencils_first_box_l41_41524

theorem sandra_pencils_first_box :
  ∃ (n : ℕ), 
    (∀ k : ℕ, k = 2 → n + (k - 1) * 9 = 87) ∧
    (∀ k : ℕ, k = 3 → n + (k - 1) * 9 = 96) ∧
    (∀ k : ℕ, k = 4 → n + (k - 1) * 9 = 105) ∧
    (∀ k : ℕ, k = 5 → n + (k - 1) * 9 = 114) ∧
    n = 78 :=
begin
  sorry
end

end sandra_pencils_first_box_l41_41524


namespace malmer_birthday_l41_41107

theorem malmer_birthday :
  ∃ (f w : ℕ), f ≤ w ∧ f ≤ 31 ∧ w ≤ 12 ∧ f ≠ 1 ∧ w ≠ 12 ∧
  (∀ f, f ≠ 7 → ∀ w, f ≤ w → (w ≠ 7 ∧ w ≤ 31 ∧ f < w → ¬(w ≤ 12 ∧ (w ≠ 6 ∧ w ≠ 5 ∧ w ≠ 4 ∧ w ≠ 3 ∧ w ≠ 2)))).
Proof
  let f := 7 in
  let w := 7 in
  exists.intro f (exists.intro w 
    (⟨by simp, by simp, by simp, by simp, by simp⟩ : 
      f ≤ w ∧ f ≤ 31 ∧ w ≤ 12 ∧ f ≠ 1 ∧ w ≠ 12)
  )
sorry

end malmer_birthday_l41_41107


namespace P2011_1_neg1_is_0_2_pow_1006_l41_41156

def P1 (x y : ℤ) : ℤ × ℤ := (x + y, x - y)

def Pn : ℕ → ℤ → ℤ → ℤ × ℤ 
| 0, x, y => (x, y)
| (n + 1), x, y => P1 (Pn n x y).1 (Pn n x y).2

theorem P2011_1_neg1_is_0_2_pow_1006 : Pn 2011 1 (-1) = (0, 2^1006) := by
  sorry

end P2011_1_neg1_is_0_2_pow_1006_l41_41156


namespace ned_weekly_earnings_l41_41515

-- Definitions of conditions
def normal_mouse_cost : ℕ := 120
def left_handed_mouse_cost (n : ℕ) := n + n * 3 / 10
def mice_sold_per_day : ℕ := 25
def store_days_open_per_week : ℕ := 4

-- Proof statement
theorem ned_weekly_earnings : 
  let left_cost := left_handed_mouse_cost normal_mouse_cost in
  let daily_earnings := left_cost * mice_sold_per_day in
  let weekly_earnings := daily_earnings * store_days_open_per_week in
  weekly_earnings = 15600 := 
by
  sorry

end ned_weekly_earnings_l41_41515


namespace number_of_differences_l41_41389

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41389


namespace basketball_max_points_l41_41204

-- Define the problem using the provided conditions
theorem basketball_max_points (p : ℕ → ℕ) (total_players : ℕ) (total_points : ℕ) 
  (min_points : ℕ) (h1 : total_players = 12) (h2 : total_points = 100) 
  (h3 : ∀ i, i < total_players → p i ≥ min_points) : 
  ∃ l, l = 23 ∧ ∀ i, i < total_players → p i = l ∨ p i = 7 → ∑ i in finset.range total_players, p i = total_points :=
by
  sorry

end basketball_max_points_l41_41204


namespace min_value_proof_l41_41838

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c)

theorem min_value_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ≥ 343 :=
sorry

end min_value_proof_l41_41838


namespace value_of_a_plus_one_l41_41440

theorem value_of_a_plus_one (a : ℤ) (h : |a| = 3) : a + 1 = 4 ∨ a + 1 = -2 :=
by
  sorry

end value_of_a_plus_one_l41_41440


namespace total_metal_rods_needed_l41_41230

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end total_metal_rods_needed_l41_41230


namespace determine_l_l41_41774

theorem determine_l :
  ∃ l : ℤ, (2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997) ∧ l = -1 :=
by
  sorry

end determine_l_l41_41774


namespace prob_X_eq_Y_is_eleven_over_hundred_l41_41648

noncomputable def probability_X_eq_Y : ℝ :=
  let interval := set.Icc (-5 * real.pi) (5 * real.pi)
  let cos_cos_eq (x y : ℝ) := real.cos (real.cos x) = real.cos (real.cos y)
  let valid_pairs := {p : ℝ × ℝ | cos_cos_eq p.fst p.snd ∧ p.fst ∈ interval ∧ p.snd ∈ interval}
  let total_points := (interval ×ˢ interval).card
  let valid_points := valid_pairs.card
  (valid_points / total_points : ℝ)

theorem prob_X_eq_Y_is_eleven_over_hundred :
  probability_X_eq_Y = 11 / 100 :=
sorry

end prob_X_eq_Y_is_eleven_over_hundred_l41_41648


namespace Abby_in_seat_3_l41_41248

variables (P : Type) [Inhabited P]
variables (Abby Bret Carl Dana : P)
variables (seat : P → ℕ)

-- Conditions from the problem:
-- Bret is actually sitting in seat #2.
axiom Bret_in_seat_2 : seat Bret = 2

-- False statement 1: Dana is next to Bret.
axiom false_statement_1 : ¬ (seat Dana = 1 ∨ seat Dana = 3)

-- False statement 2: Carl is sitting between Dana and Bret.
axiom false_statement_2 : ¬ (seat Carl = 1)

-- The final translated proof problem:
theorem Abby_in_seat_3 : seat Abby = 3 :=
sorry

end Abby_in_seat_3_l41_41248


namespace num_diff_positive_integers_l41_41371

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41371


namespace correctness_of_compound_proposition_l41_41503

noncomputable theory

def f1 (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3)
def f2 (x : ℝ) : ℝ := Real.cos (x + Real.pi)

def p : Prop := ∀ x ∈ Ioi (3 : ℝ), MonotoneOn f1 (Ioi x)
def q : Prop := ∃ x, f2 x = -f2 x

theorem correctness_of_compound_proposition :
  (p ∧ ¬ q) :=
sorry

end correctness_of_compound_proposition_l41_41503


namespace tan_add_pi_over_4_l41_41712

variable (θ : ℝ)
hypothesis (h : Real.tan θ = 2)

theorem tan_add_pi_over_4 : Real.tan (θ + π / 4) = -3 := by
  sorry

end tan_add_pi_over_4_l41_41712


namespace number_of_elements_in_set_A_l41_41353

def A := {n : ℤ | (1 + 1 / (n : ℝ)) ^ (n + 1) = (1 + 1 / 2020) ^ 2000}

theorem number_of_elements_in_set_A :
  set.finite A ∧ set.card A = 1 := sorry

end number_of_elements_in_set_A_l41_41353


namespace sin_squared_identity_cos_squared_identity_l41_41608

-- Define the angles α, β, and γ
variables (α β γ : ℝ)

-- Angle condition: α + β + γ = 180 degrees (in radians: π)
def angle_condition : Prop := α + β + γ = π

-- The first theorem to prove given the angle condition
theorem sin_squared_identity (h : angle_condition α β γ) :
  sin(α)^2 + sin(β)^2 - sin(γ)^2 = 2 * sin(α) * sin(β) * cos(γ) :=
sorry

-- The second theorem to prove given the angle condition
theorem cos_squared_identity (h : angle_condition α β γ) :
  cos(α)^2 + cos(β)^2 - cos(γ)^2 = 1 - 2 * sin(α) * sin(β) * cos(γ) :=
sorry

end sin_squared_identity_cos_squared_identity_l41_41608


namespace total_food_each_day_l41_41685

-- Conditions
def num_dogs : ℕ := 2
def food_per_dog : ℝ := 0.125
def total_food : ℝ := num_dogs * food_per_dog

-- Proof statement
theorem total_food_each_day : total_food = 0.25 :=
by
  sorry

end total_food_each_day_l41_41685


namespace evaluate_periodic_decimals_l41_41287

noncomputable def decimal_to_fraction (d: ℚ) : ℚ :=
  if d = 234/999 then 234/999
  else if d = 567/999 then 567/999
  else if d = 891/999 then 891/999
  else 0 -- this is a redundant case for the sake of completeness

theorem evaluate_periodic_decimals :
  let x := 0.234 in
  let y := 0.567 in
  let z := 0.891 in
  x - y + z = 186/333 :=
by {
  have x_frac : x = 234/999 := by { exact decimal_to_fraction (0.234) },
  have y_frac : y = 567/999 := by { exact decimal_to_fraction (0.567) },
  have z_frac : z = 891/999 := by { exact decimal_to_fraction (0.891) },
  rw [x_frac, y_frac, z_frac],
  norm_num
}

end evaluate_periodic_decimals_l41_41287


namespace integer_root_of_polynomial_l41_41559

theorem integer_root_of_polynomial (b c : ℚ) (h_root1 : (2 - real.sqrt 5)^3 + b * (2 - real.sqrt 5) + c = 0)
    (h_root2 : (2 + real.sqrt 5)^3 + b * (2 + real.sqrt 5) + c = 0) : ∃ r : ℤ, 
    ((2 - real.sqrt 5) + (2 + real.sqrt 5) + r = 0) ∧ r = -4 :=
by 
  use -4
  have root_sum : (2 - real.sqrt 5) + (2 + real.sqrt 5) = 4 := by
    rw [real.eq_of_real_eq_coe] 
    ring
  have := root_sum + (-4) = 0
  sorry

end integer_root_of_polynomial_l41_41559


namespace sum_is_integer_l41_41703

def distinct_nat_numbers (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → a i ≠ a j

def p_i (a : ℕ → ℕ) (n i : ℕ) : ℕ :=
  ∏ (j : ℕ) in finset.univ.filter (λ j, j ≠ i), (a i - a j)

def sum_term (a : ℕ → ℕ) (n k i : ℕ) : ℚ :=
  (a i)^k / p_i a n i

theorem sum_is_integer (a : ℕ → ℕ) (n k : ℕ) (h1 : n > 1)
  (h2 : distinct_nat_numbers a n) : 
  (∑ i in finset.range n, sum_term a n k i).is_integer :=
sorry

end sum_is_integer_l41_41703


namespace geometric_sequence_iff_t_eq_neg1_l41_41322

variable {a : ℕ → ℝ} -- Declare the sequence {a_n} as a real sequence
variable {t : ℝ} -- Declare t as a real constant

-- Define the sum of the first n terms
def S_n (n : ℕ) : ℝ := 2^n + t

-- Define the sequence {a_n} in terms of the sum S_n
def a (n : ℕ) : ℝ :=
  if n = 1 then S_n 1
  else S_n n - S_n (n - 1)

-- Lean statement corresponding to the proof problem
theorem geometric_sequence_iff_t_eq_neg1 : 
  (∀ n, a n = 2^(n-1)) ↔ t = -1 :=
sorry

end geometric_sequence_iff_t_eq_neg1_l41_41322


namespace frac_linear_fn_fixed_point_universal_l41_41096

-- Define a fractional linear function as having the form f(x) = (ax + b) / (cx + d)
structure frac_linear_fn (α : Type*) :=
(a b c d : α)

-- Define the fixed point condition for a given function of type α → α
def is_fixed_point {α : Type*} [field α] (f : α → α) (x : α) : Prop :=
  f x = x

-- Define the main theorem statement
theorem frac_linear_fn_fixed_point_universal {α : Type*} [field α] 
  (f : frac_linear_fn α) (f_k : ℕ → (α → α)) (k : ℕ)
  (h1 : ∃ n, is_fixed_point (f_k k) n ∧ ¬is_fixed_point (λ x, (f.a * x + f.b) / (f.c * x + f.d)) n) :
  ∀ x, is_fixed_point (f_k k) x :=
sorry

end frac_linear_fn_fixed_point_universal_l41_41096


namespace team_ate_pizza_slices_l41_41143

theorem team_ate_pizza_slices :
  let el_slices := 3 * 16 in
  let l_slices := 2 * 12 in
  let m_slices := 1 * 8 in
  let total_slices := el_slices + l_slices + m_slices in
  let slices_left := 17 in
  total_slices - slices_left = 63 :=
by
  -- Defining the slices for each type of pizza
  let el_slices := 3 * 16
  let l_slices := 2 * 12
  let m_slices := 1 * 8
  -- Calculating the total slices
  let total_slices := el_slices + l_slices + m_slices
  -- Subtracting the remaining slices
  let slices_left := 17
  show total_slices - slices_left = 63
  -- This is where the proof would go
  sorry

end team_ate_pizza_slices_l41_41143


namespace cone_lateral_surface_area_l41_41039

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : 
  (r * l * Real.pi = 6 * Real.pi) := by
  sorry

end cone_lateral_surface_area_l41_41039


namespace simplify_and_evaluate_l41_41883

theorem simplify_and_evaluate :
  let x := (-1) / 2 in
  (3*x^4 - 2*x^3) / (-x) - (x - x^2) * (3*x) = - 1 / 4 := by
  sorry

end simplify_and_evaluate_l41_41883


namespace plane_through_A_perpendicular_to_BC_l41_41200

theorem plane_through_A_perpendicular_to_BC :
  ∃ (a b c d : ℝ), a = 5 ∧ b = -1 ∧ c = 3 ∧ d = -19 ∧
  (∀ (x y z : ℝ), a * (x - 5) + b * (y - 3) + c * (z + 1) = 0 ↔ 5 * x - y + 3 * z - 19 = 0) :=
begin
  use [5, -1, 3, -19],
  split, { refl },
  split, { refl },
  split, { refl },
  split, { refl },
  intros x y z,
  split,
  { intro h,
    calc 5 * x - y + 3 * z - 19 = 5 * x + (-1) * y + 3 * z + -19 : by ring
    ... = 0 : by { rw ← h,
                   ring } },
  { intro h,
    calc 5 * (x - 5) + (-1) * (y - 3) + 3 * (z + 1) = 5 * x - 25 + (- y + 3) + 3 * z + 3 : by ring
    ... = 5 * x - y + 3 * z - 19 : by ring
    ... = 0 : by rw h }
end

end plane_through_A_perpendicular_to_BC_l41_41200


namespace negation_of_exists_lt_l41_41908

theorem negation_of_exists_lt :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 3 < 0) = (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0) :=
by sorry

end negation_of_exists_lt_l41_41908


namespace find_a_tangent_to_circle_l41_41026

variable (a : ℝ)

def circle_eq : (x y : ℝ) → Prop := λ x y, (x - a)^2 + y^2 = a
def line_eq : (x y : ℝ) → Prop := λ x y, y = x + a
def tangent_condition : Prop := ∃ x y : ℝ, circle_eq a x y ∧ line_eq a x y

theorem find_a_tangent_to_circle (h : 0 < a) (tangent : tangent_condition a) : a = 1 / 2 := 
by
  sorry

end find_a_tangent_to_circle_l41_41026


namespace diff_of_two_distinct_members_l41_41400

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41400


namespace number_of_distinct_positive_differences_l41_41422

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41422


namespace max_determinant_of_matrix_l41_41848

open Matrix

noncomputable def u : Vector ℝ 3 := sorry -- to be confirmed by user as exact unit vector

def v : Vector ℝ 3 := ![3, 2, -2]
def w : Vector ℝ 3 := ![2, -1, 4]

theorem max_determinant_of_matrix :
  ∃ (u : Vector ℝ 3), (∥u∥ = 1) ∧ (det ![u, v, w] = Real.sqrt 341) :=
by sorry

end max_determinant_of_matrix_l41_41848


namespace six_digit_numbers_with_zero_l41_41005

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41005


namespace solution_set_of_inequality_l41_41043

variable {f : ℝ → ℝ}

-- Definitions given in the problem
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing_on_pos_infinity (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The main theorem translating the question and conditions to Lean
theorem solution_set_of_inequality (h1 : is_odd_function f)
                                   (h2 : is_increasing_on_pos_infinity f)
                                   (h3 : f (-3) = 0) :
  { x : ℝ | (x - 2) * f x < 0 } = set.union (set.Ioo (-3 : ℝ) 0) (set.Ioo 2 3) :=
by
  sorry

end solution_set_of_inequality_l41_41043


namespace task_1_task_2_task_3_l41_41718

-- Condition: Function y = f(x) with properties f(x + y) = f(x) + f(y) and f(x) > 0 for x > 0
variables {R: Type*} [ordered_ring R] (f : R → R) (hf_add : ∀ x y, f (x + y) = f x + f y)
(hf_pos : ∀ x > 0, f x > 0)

-- Task 1: Prove that f(0) = 0
theorem task_1 : f 0 = 0 := sorry

-- Task 2: Prove that the function f(x) is increasing
theorem task_2 : ∀ (x₁ x₂ : R), x₁ > x₂ → f x₁ > f x₂ := sorry

-- Task 3: Solve the inequality 1/2 f(x^2) - f(x) > 1/2 f(3x)
theorem task_3 : ∀ (x : R), (1/2 : R) * f (x * x) - f x > (1/2 : R) * f (3 * x) ↔ x < 0 ∨ x > 5 := sorry

end task_1_task_2_task_3_l41_41718


namespace measure_of_angle_D_l41_41058

-- Definitions of angles in pentagon ABCDE
variables (A B C D E : ℝ)

-- Conditions
def condition1 := D = A + 30
def condition2 := E = A + 50
def condition3 := B = C
def condition4 := A = B - 45
def condition5 := A + B + C + D + E = 540

-- Theorem to prove
theorem measure_of_angle_D (h1 : condition1 A D)
                           (h2 : condition2 A E)
                           (h3 : condition3 B C)
                           (h4 : condition4 A B)
                           (h5 : condition5 A B C D E) :
  D = 104 :=
sorry

end measure_of_angle_D_l41_41058


namespace solve_eq_Max_l41_41310

def Max (a b : ℝ) : ℝ := if a < b then b else a

theorem solve_eq_Max (a b x : ℝ) (h1 : a ≠ b) (h2 : a = -1) (h3 : b = 1 / x) : 
  Max a b = (2 * x - 1) / x + 2 ↔ x = 1 / 3 :=
sorry

end solve_eq_Max_l41_41310


namespace differences_of_set_l41_41380

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41380


namespace original_cost_of_plants_l41_41988

theorem original_cost_of_plants
  (discount : ℕ)
  (amount_spent : ℕ)
  (original_cost : ℕ)
  (h_discount : discount = 399)
  (h_amount_spent : amount_spent = 68)
  (h_original_cost : original_cost = discount + amount_spent) :
  original_cost = 467 :=
by
  rw [h_discount, h_amount_spent] at h_original_cost
  exact h_original_cost

end original_cost_of_plants_l41_41988


namespace b_2023_value_l41_41092

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 3
  else if n = 2 then 8
  else (sequence (n - 1) + sequence (n - 2)) / 2

theorem b_2023_value : sequence 2023 = 5.5 :=
sorry

end b_2023_value_l41_41092


namespace unoccupied_space_in_container_is_936_l41_41890

def container_volume := 12^3
def water_volume := container_volume / 3
def ice_cube_volume := 3^3
def total_ice_volume := 8 * ice_cube_volume
def combined_volume := water_volume + total_ice_volume
def unoccupied_volume := container_volume - combined_volume

theorem unoccupied_space_in_container_is_936 : 
  unoccupied_volume = 936 := 
by
  unfold container_volume water_volume ice_cube_volume total_ice_volume combined_volume unoccupied_volume
  sorry

end unoccupied_space_in_container_is_936_l41_41890


namespace find_cards_given_l41_41960

theorem find_cards_given (initial_cards : ℕ) (remaining_cards : ℕ) (cards_given : ℕ) : 
  initial_cards = 304 → remaining_cards = 276 → cards_given = initial_cards - remaining_cards → cards_given = 28 :=
by
  intros h1 h2 h3
  -- this is the statement to be proved
  have h: cards_given = 28,
  -- Steps and intermediate results would go here, but are skipped as per instructions
  sorry

end find_cards_given_l41_41960


namespace find_real_part_of_solution_l41_41530

noncomputable def exists_real_part_of_solution (a b : ℝ) : Prop :=
  let z := Complex.mk a b in
  (z * (z + Complex.I) * (z + 2 * Complex.I) = 1001 * Complex.I)

theorem find_real_part_of_solution : ∃ (a b : ℝ), exists_real_part_of_solution a b :=
begin
  -- Proof omitted
  sorry
end

end find_real_part_of_solution_l41_41530


namespace age_difference_is_20_l41_41136

-- Definitions for the ages of the two persons
def elder_age := 35
def younger_age := 15

-- Condition: Difference in ages
def age_difference := elder_age - younger_age

-- Theorem to prove the difference in ages is 20 years
theorem age_difference_is_20 : age_difference = 20 := by
  sorry

end age_difference_is_20_l41_41136


namespace num_diff_positive_integers_l41_41375

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41375


namespace min_value_change_when_add_x2_l41_41194

-- Given conditions
variables {a b c : ℝ}
variable {f : ℝ → ℝ} 

-- Assume f is a quadratic polynomial of the form ax^2 + bx + c
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the modified polynomials
def f_add_3x2 (x : ℝ) : ℝ := (a + 3) * x^2 + b * x + c
def f_sub_x2 (x : ℝ) : ℝ := (a - 1) * x^2 + b * x + c
def f_add_x2 (x : ℝ) : ℝ := (a + 1) * x^2 + b * x + c

-- Minimum values of the polynomials according to the conditions
def min_f : ℝ := -b^2 / (4 * a) + c
def min_f_add_3x2 : ℝ := -b^2 / (4 * (a + 3)) + c
def min_f_sub_x2 : ℝ := -b^2 / (4 * (a - 1)) + c
def min_f_add_x2 : ℝ := -b^2 / (4 * (a + 1)) + c

-- Statement of the problem
theorem min_value_change_when_add_x2
  (h1 : min_f_add_3x2 = min_f + 9)
  (h2 : min_f_sub_x2 = min_f - 9) :
  min_f_add_x2 = min_f + 9 / 2 :=
begin
  -- Proof goes here
  sorry
end

end min_value_change_when_add_x2_l41_41194


namespace j_h_five_l41_41028

-- Define the functions h and j
def h (x : ℤ) : ℤ := 4 * x + 5
def j (x : ℤ) : ℤ := 6 * x - 11

-- State the theorem to prove j(h(5)) = 139
theorem j_h_five : j (h 5) = 139 := by
  sorry

end j_h_five_l41_41028


namespace inequality_solution_set_l41_41162

theorem inequality_solution_set (x : ℝ) : (2 : ℝ)^(x^2 - x) < 4 ↔ x ∈ set.Ioo (-1 : ℝ) 2 := 
sorry

end inequality_solution_set_l41_41162


namespace inequality_proof_l41_41088

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := 1.1 ^ 0.9

theorem inequality_proof : b < a ∧ a < c := by
  sorry

end inequality_proof_l41_41088


namespace a_n_is_integer_l41_41610

-- Definition of the sequence
def a : ℕ → ℤ
| 0     := 0 -- By convention, since the problem starts at n=1.
| 1     := 1
| 2     := 2
| 3     := 3
| (n+1) := if h : n ≥ 3 then a n - a (n-1) + (a n * a n) / a (n-2) else 0 -- We use a safeguard for n < 3

-- Theorem we need to prove
theorem a_n_is_integer (n : ℕ) : ∃ (a_n : ℤ), a n = a_n :=
by sorry -- Proof is omitted as requested.

end a_n_is_integer_l41_41610


namespace no_solution_for_t_and_s_l41_41676

theorem no_solution_for_t_and_s (k : ℝ) (t s : ℝ) : (k = -4/5) →
    (∀ t s: ℝ,
    (begin
      let lhs := (⟨1, -3⟩ : ℝ × ℝ) + t • (⟨5, -2⟩ : ℝ × ℝ),
      let rhs := (⟨6, 4⟩ : ℝ × ℝ) + s • (⟨2, k⟩ : ℝ × ℝ),
      lhs ≠ rhs
    end)) :=
 by intros k_spec t s;
  sorry

end no_solution_for_t_and_s_l41_41676


namespace number_of_possible_values_and_sum_of_g3_l41_41497

theorem number_of_possible_values_and_sum_of_g3
    (g : ℝ → ℝ)
    (h : ∀ x y : ℝ, g(x) * g(y) - g(x * y) = x - y) :
    let n := 1 in  -- Number of possible values of g(3)
    let s := -2 in  -- Sum of all possible values of g(3)
    n * s = -2 :=
by
  -- Definitions derived from the problem statement
  sorry

end number_of_possible_values_and_sum_of_g3_l41_41497


namespace angle_B_l41_41815

theorem angle_B (a b c A B : ℝ) (h : a * Real.cos B - b * Real.cos A = c) (C : ℝ) (hC : C = Real.pi / 5) (h_triangle : A + B + C = Real.pi) : B = 3 * Real.pi / 10 :=
sorry

end angle_B_l41_41815


namespace locus_of_projection_l41_41506

theorem locus_of_projection {a b c : ℝ} (h : (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2) :
  ∀ (x y : ℝ), (x, y) ∈ ({P : ℝ × ℝ | ∃ a b : ℝ, P = ((a * b^2) / (a^2 + b^2), (a^2 * b) / (a^2 + b^2)) ∧ (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2}) → 
    x^2 + y^2 = c^2 := 
sorry

end locus_of_projection_l41_41506


namespace pairs_exist_in_loaf_l41_41793

-- Definitions and Conditions
def loaf_cm : ℝ := 10
def raisins_type : ℕ := 2
def distance : ℝ := 1

-- Theorem statement
theorem pairs_exist_in_loaf :
  ∃ (p1 p2 : ℝ × ℝ × ℝ), dist p1 p2 = 1 ∧ 
  (p1 ∉ raisin_1 ∧ p1 ∉ raisin_2 ∧ p2 ∉ raisin_1 ∧ p2 ∉ raisin_2) ∨
  (p1 ∈ raisin_1 ∧ p2 ∈ raisin_1) ∨
  (p1 ∈ raisin_2 ∧ p2 ∈ raisin_2) 
  := 
sorry

-- Assume definitions of raisin_1 and raisin_2 to be sets of points in space
variable (raisin_1 raisin_2 : set (ℝ × ℝ × ℝ))

#check pairs_exist_in_loaf

end pairs_exist_in_loaf_l41_41793


namespace fractions_equal_l41_41023

theorem fractions_equal (x y z : ℝ) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hxy : x ≠ y)
  (h : (yz - x^2) / (1 - x) = (xz - y^2) / (1 - y)) : (yz - x^2) / (1 - x) = x + y + z ∧ (xz - y^2) / (1 - y) = x + y + z :=
sorry

end fractions_equal_l41_41023


namespace length_of_other_train_l41_41203

-- Definition of the conditions
def speed_in_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def initial_conditions
  (length_train1 : ℝ)
  (speed_train1_kmph : ℝ)
  (speed_train2_kmph : ℝ)
  (crossing_time_secs : ℝ)
  (length_train2 : ℝ) : Prop :=
  let speed_train1 := speed_in_mps speed_train1_kmph
  let speed_train2 := speed_in_mps speed_train2_kmph
  let relative_speed := speed_train1 + speed_train2
  length_train1 + length_train2 = relative_speed * crossing_time_secs

-- The main theorem
theorem length_of_other_train
  (length_train1 : ℝ := 300)
  (speed_train1_kmph : ℝ := 120)
  (speed_train2_kmph : ℝ := 80)
  (crossing_time_secs : ℝ := 9) :
  ∃ length_train2 : ℝ, initial_conditions length_train1 speed_train1_kmph speed_train2_kmph crossing_time_secs length_train2 ∧ length_train2 = 199.95 :=
by
  sorry

end length_of_other_train_l41_41203


namespace correct_expression_l41_41201

theorem correct_expression (a b : ℤ) (h1 : (2 : ℤ) * b - 3 * a = -13) (h2 : -a * b = 6) (h3 : (2 : ℤ) * b + a = -1) :
  (2 * x + a) * (3 * x + b) = 6 * x^2 + 5 * x - 6 := by
  have a_val : a = 3 := by
    sorry
  have b_val : b = -2 := by
    sorry
  rw [a_val, b_val]
  ring
  sorry

end correct_expression_l41_41201


namespace statement_B_statement_C_statement_D_l41_41482

variables {A B C a b c : ℕ}

/-- Given conditions of the problem -/
def given_conditions (A B C a b c: ℕ) := 
  (∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  sin A = 2 * sin B ∧ sin C = sin (A + B + C)) ∧ 
  (forall x y z, ∃ w, w = a * x ∧ w = b * y ∧ w = c * z) ∧ 
  (∀ A B C, triangle A B C = (A + B + C = π)) ∧ 
  (∀ a b c, sides A a B b C c = (a / sin A = b / sin B = c / sin C))

/-- If C = 2B then triangle ABC is a right triangle -/
theorem statement_B (h : given_conditions A B C a b c) (hC2B : C = 2 * B) : 
  is_right_triangle A B C :=
sorry

/-- If triangle ABC is isosceles triangle, then sin B = sqrt 15 / 8 --/
theorem statement_C (h : given_conditions A B C a b c) (isosceles : a = c) : 
  sin B = (sqrt 15 / 8) :=
sorry

/-- If c = 3, the maximum area of triangle ABC is 3 --/
theorem statement_D (h : given_conditions A B C a b c) (hcm_eq_3 : c = 3) : 
  max_area A B C a b c = 3 :=
sorry

end statement_B_statement_C_statement_D_l41_41482


namespace six_digit_numbers_with_zero_l41_41001

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41001


namespace sum_distances_regular_tetrahedron_constant_l41_41877

theorem sum_distances_regular_tetrahedron_constant
  (T : Type) [normed_group T] [normed_space ℝ T]
  (tetrahedron : set T)
  (h_regular : regular_tetrahedron tetrahedron)
  (P : T)
  (h_internal : P ∈ tetrahedron) :
  ∃ k : ℝ, ∀ (dist_f1 dist_f2 dist_f3 dist_f4 : ℝ),
  are_perpendicular_distances tetrahedron P dist_f1 dist_f2 dist_f3 dist_f4 →
  dist_f1 + dist_f2 + dist_f3 + dist_f4 = k :=
begin
  sorry
end

end sum_distances_regular_tetrahedron_constant_l41_41877


namespace total_price_correct_l41_41951

-- Define the prices of the individual items
def refrigerator_price : ℝ := 4275
def washing_machine_price : ℝ := refrigerator_price - 1490
def dishwasher_price : ℝ := washing_machine_price / 2

-- Define the sales tax rate and the flat delivery fee
def sales_tax_rate : ℝ := 0.07
def delivery_fee : ℝ := 75

-- Calculate the total price before tax
def total_price_before_tax : ℝ := refrigerator_price + washing_machine_price + dishwasher_price

-- Calculate the total sales tax
def total_sales_tax : ℝ := total_price_before_tax * sales_tax_rate

-- Calculate the total price including tax
def total_price_with_tax : ℝ := total_price_before_tax + total_sales_tax

-- Calculate the total price including tax and delivery
def total_price_with_tax_and_delivery : ℝ := total_price_with_tax + delivery_fee

-- Prove that the total price including tax and delivery is $9119.18
theorem total_price_correct : total_price_with_tax_and_delivery = 9119.18 := by
  sorry

end total_price_correct_l41_41951


namespace six_digit_numbers_with_zero_l41_41007

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41007


namespace differences_of_set_l41_41379

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41379


namespace triangle_is_equilateral_l41_41785

-- Define a triangle with angles A, B, and C
variables (A B C : ℝ)

-- The conditions of the problem
def log_sin_arithmetic_sequence : Prop :=
  Real.log (Real.sin A) + Real.log (Real.sin C) = 2 * Real.log (Real.sin B)

def angles_arithmetic_sequence : Prop :=
  2 * B = A + C

-- The theorem that the triangle is equilateral given these conditions
theorem triangle_is_equilateral :
  log_sin_arithmetic_sequence A B C → angles_arithmetic_sequence A B C → 
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_is_equilateral_l41_41785


namespace smallest_k_for_sum_of_squares_multiple_of_360_l41_41304

theorem smallest_k_for_sum_of_squares_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ ∀ n : ℕ, n > 0 → (n * (n + 1) * (2 * n + 1)) / 6 % 360 = 0 → k ≤ n :=
by sorry

end smallest_k_for_sum_of_squares_multiple_of_360_l41_41304


namespace max_neg_ones_in_grid_l41_41475

-- Define the grid size
def grid_size : ℕ := 8

-- Define the type of entries (1 or -1)
@[derive DecidableEq]
inductive Entry
| pos_one : Entry
| neg_one : Entry

open Entry

-- Define a grid as a function from (ℕ × ℕ) to Entry
def Grid : Type := fin grid_size × fin grid_size → Entry

-- Conditions on the grid
def row_product_is_one (grid : Grid) (i : fin grid_size) : Prop :=
  (finset.univ.product (finset.univ : finset (fin grid_size))).filter (λ p, p.1 = i).prod (λ p, match grid p with
  | pos_one => 1
  | neg_one => -1
  end) = 1

def column_product_is_one (grid : Grid) (j : fin grid_size) : Prop :=
  (finset.univ.product (finset.univ : finset (fin grid_size))).filter (λ p, p.2 = j).prod (λ p, match grid p with
  | pos_one => 1
  | neg_one => -1
  end) = 1

def diagonal_product_is_one (grid : Grid) (d : ℤ) : Prop :=
  (finset.univ.product (finset.univ : finset (fin grid_size))).filter (λ p, (p.1 : ℤ) - (p.2 : ℤ) = d).prod (λ p, match grid p with
  | pos_one => 1
  | neg_one => -1
  end) = 1

-- Main problem statement
theorem max_neg_ones_in_grid : ∃ (grid : Grid), 
  (∀ i : fin grid_size, row_product_is_one grid i) ∧ 
  (∀ j : fin grid_size, column_product_is_one grid j) ∧ 
  (∀ d : ℤ, abs d < grid_size → diagonal_product_is_one grid d) ∧
  (finset.univ.product (finset.univ : finset (fin grid_size))).filter (λ p, grid p = neg_one).card = 48 :=
sorry

end max_neg_ones_in_grid_l41_41475


namespace max_possible_shaded_area_l41_41898

-- Define the conditions of the problem
variables {x y h p : ℕ}
variable prime_p : nat.prime p
variable positive_integers : x > y ∧ y > 0 ∧ x > 0
variable area_constraint : (x ≠ 0 ∧ y ≠ 0 ∧ h ≠ 0 → (x + y) * h / 2 = p)
variable frame_constraint : (x - y) = 2 * h
variable shaded_area_contraint : (x > y ∧ (x - y) = 2 * h ∧ (x ≠ 0 ∧ y ≠ 0 ∧ h ≠ 0 → (3 * p - 1) * (p - 1) < 2000)

-- Define the objective function for maximization
def max_shaded_area : ℕ :=
  max (λ (p: ℕ), if nat.prime p ∧ (p - 1) * (3 * p - 1) < 2000 then (p - 1) * (3 * p - 1) else 0)

-- Assertion of the maximum shaded area
theorem max_possible_shaded_area : 
  ∀ (p: ℕ), nat.prime p → (p - 1) * (3 * p - 1) < 2000 → 
    (p = 23 → max_shaded_area = 1496) :=
by {
  -- This is where the proof steps would be inserted
  sorry
}

end max_possible_shaded_area_l41_41898


namespace solution_of_system_l41_41279

/-- 
Given a system of multivariable equations,
1. x + y + z = 26
2. 3x - 2y + z = 3
3. x - 4y - 2z = -13
prove that the solution is (x, y, z).
-/
theorem solution_of_system :
  ∃ x y z : ℝ,
    x + y + z = 26 ∧
    3*x - 2*y + z = 3 ∧
    x - 4*y - 2*z = -13 ∧
    x = -32.2 ∧
    y = -13.8 ∧
    z = 72 :=
begin
  sorry
end

end solution_of_system_l41_41279


namespace general_term_a_sum_of_b_l41_41536

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := sorry

-- Given conditions
axiom a1 : a 1 = 1
axiom d_not_zero : exists d : ℕ, d ≠ 0
axiom condition : (a 3) * (a 4) = a 12

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ := (a n) * 2^n

-- Prove the general term for the sequence {a_n} is a_n = n
theorem general_term_a (n : ℕ) : a n = n := sorry

-- Prove the sum of the first n terms of {b_n} is T_n = (n-1) * 2^(n+1) + 2
theorem sum_of_b (n : ℕ) : (finset.sum (finset.range n) b) = (n-1) * 2^(n+1) + 2 := sorry

end general_term_a_sum_of_b_l41_41536


namespace total_books_pyramid_l41_41256

def num_books (n : ℕ) : ℕ :=
  if n = 4 then 64
  else if n = 3 then 64 / 0.8
  else if n = 2 then (64 / 0.8) / 0.8
  else if n = 1 then ((64 / 0.8) / 0.8) / 0.8
  else 0

theorem total_books_pyramid :
  num_books 4 + num_books 3 + num_books 2 + num_books 1 = 369 :=
by sorry

end total_books_pyramid_l41_41256


namespace days_worked_per_week_l41_41637

theorem days_worked_per_week
  (hourly_wage : ℕ) (hours_per_day : ℕ) (total_earnings : ℕ) (weeks : ℕ)
  (H_wage : hourly_wage = 12) (H_hours : hours_per_day = 9) (H_earnings : total_earnings = 3780) (H_weeks : weeks = 7) :
  (total_earnings / weeks) / (hourly_wage * hours_per_day) = 5 :=
by 
  sorry

end days_worked_per_week_l41_41637


namespace minimum_lines_determined_l41_41582

-- Definition of distinct points and non-collinearity
def distinct_points (P : Finset (ℝ × ℝ)) : Prop := P.card = P.to_finset.card
def non_collinear (P : Finset (ℝ × ℝ)) : Prop :=
  ∀ (A B C ∈ P), 
    (A ≠ B ∧ B ≠ C ∧ A ≠ C) → 
    ¬ collinear ℝ ({A, B, C} : set (ℝ × ℝ))

-- Theorem statement
theorem minimum_lines_determined (n : ℕ) (P : Finset (ℝ × ℝ)) 
  (hP : P.card = n) (h_distinct : distinct_points P) (h_non_collinear : non_collinear P) : 
  ∃ (L : Finset (Finset (ℝ × ℝ))), L.card ≥ n ∧ ∀ l ∈ L, ∃ A B, A ∈ P ∧ B ∈ P ∧ A ≠ B ∧ l = {A, B} :=
sorry

end minimum_lines_determined_l41_41582


namespace sin_squared_15_minus_cos_squared_15_correct_l41_41699

noncomputable def sin_squared_15_minus_cos_squared_15 : ℝ :=
  sin (15 * (real.pi / 180)) ^ 2 - cos (15 * (real.pi / 180)) ^ 2

theorem sin_squared_15_minus_cos_squared_15_correct :
  sin_squared_15_minus_cos_squared_15 = - (real.sqrt 3 / 2) :=
by
  sorry

end sin_squared_15_minus_cos_squared_15_correct_l41_41699


namespace six_digit_numbers_divisible_by_2_l41_41650

theorem six_digit_numbers_divisible_by_2 :
  let digits := {0, 1, 2, 3, 4, 5} in
  let is_six_digit (num : List ℕ) := num.length = 6 in
  let has_no_repetition (num : List ℕ) := num.nodup in
  let is_divisible_by_2 (num : List ℕ) := (num.nth 5).getOrElse 1 % 2 = 0 in
  let valid_numbers := {num | is_six_digit num ∧ has_no_repetition num ∧ is_divisible_by_2 num ∧ (∀ n ∈ num, n ∈ digits)} in
  valid_numbers.to_finset.card = 312 := sorry

end six_digit_numbers_divisible_by_2_l41_41650


namespace star_values_l41_41307

def star (x y : ℤ) : ℝ := Real.sqrt (4 * (x ^ 2) + 4 * (y ^ 2))

theorem star_values :
  star (star 3 4).toInt (star 6 8).toInt = 50 * Real.sqrt 2 := 
by simp [star, Real.sqrt]; sorry

end star_values_l41_41307


namespace cone_lateral_surface_area_l41_41040

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : 
  (r * l * Real.pi = 6 * Real.pi) := by
  sorry

end cone_lateral_surface_area_l41_41040


namespace maximal_set_size_is_32_l41_41283

noncomputable def maximal_set_size : ℕ :=
  let S := {1, 2, 3, 4, 5, 6}
  let subsets := Finset.powerset S
  let valid_subsets := subsets.filter (λ t, t ≠ ∅ ∧ ∃ d, d ∈ S ∧ d ∉ t)
  valid_subsets.card

theorem maximal_set_size_is_32 :
  maximal_set_size = 32 :=
  sorry

end maximal_set_size_is_32_l41_41283


namespace remainder_of_19_pow_60_mod_7_l41_41939

theorem remainder_of_19_pow_60_mod_7 : (19 ^ 60) % 7 = 1 := 
by {
  sorry
}

end remainder_of_19_pow_60_mod_7_l41_41939


namespace miles_traveled_total_l41_41512

-- Define the initial distance and the additional distance
def initial_distance : ℝ := 212.3
def additional_distance : ℝ := 372.0

-- Define the total distance as the sum of the initial and additional distances
def total_distance : ℝ := initial_distance + additional_distance

-- Prove that the total distance is 584.3 miles
theorem miles_traveled_total : total_distance = 584.3 := by
  sorry

end miles_traveled_total_l41_41512


namespace Petya_meets_Vasya_l41_41521

def Petya_speed_on_paved (v_g : ℝ) : ℝ := 3 * v_g

def Distance_to_bridge (v_g : ℝ) : ℝ := 3 * v_g

def Vasya_travel_time (v_g t : ℝ) : ℝ := v_g * t

def Total_distance (v_g : ℝ) : ℝ := 2 * Distance_to_bridge v_g

def New_distance (v_g t : ℝ) : ℝ := (Total_distance v_g) - 2 * Vasya_travel_time v_g t

def Relative_speed (v_g : ℝ) : ℝ := v_g + v_g

def Time_to_meet (v_g : ℝ) : ℝ := (New_distance v_g 1) / Relative_speed v_g

theorem Petya_meets_Vasya (v_g : ℝ) : Time_to_meet v_g + 1 = 2 := by
  sorry

end Petya_meets_Vasya_l41_41521


namespace walker_ends_in_corner_l41_41956

-- Define the start point, equal probability paths, and equal probability at intersections
def start_center (walker_position : ℕ) : Prop :=
  walker_position = 0

def equal_probability_path : ℕ → Prop :=
  λ n, 0 ≤ n ∧ n ≤ 4

def equal_probability_intersection : ℕ → Prop :=
  λ n, n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

-- Define the probability calculation
noncomputable def probability_ending_in_corner : ℚ :=
  1 / 3

-- Prove the statement
theorem walker_ends_in_corner (walker_position : ℕ) :
  start_center walker_position →
  (∀ n, equal_probability_path n) →
  (∀ k, equal_probability_intersection k) →
  (probability_ending_in_corner = 1 / 3) :=
by
  intros,
  sorry

end walker_ends_in_corner_l41_41956


namespace raine_change_l41_41234

noncomputable def price_bracelet : ℝ := 15
noncomputable def price_necklace : ℝ := 10
noncomputable def price_mug : ℝ := 20
noncomputable def price_keychain : ℝ := 5

noncomputable def quantity_bracelet : ℕ := 3
noncomputable def quantity_necklace : ℕ := 2
noncomputable def quantity_mug : ℕ := 1
noncomputable def quantity_keychain : ℕ := 4

noncomputable def discount_rate : ℝ := 0.12

noncomputable def amount_given : ℝ := 100

-- The total cost before discount
noncomputable def total_before_discount : ℝ := 
  quantity_bracelet * price_bracelet + 
  quantity_necklace * price_necklace + 
  quantity_mug * price_mug + 
  quantity_keychain * price_keychain

-- The discount amount
noncomputable def discount_amount : ℝ := total_before_discount * discount_rate

-- The final amount Raine has to pay after discount
noncomputable def final_amount : ℝ := total_before_discount - discount_amount

-- The change Raine gets back
noncomputable def change : ℝ := amount_given - final_amount

theorem raine_change : change = 7.60 := 
by sorry

end raine_change_l41_41234


namespace angle_between_lateral_faces_pyramid_l41_41539

def pyramid_geometry (a b : ℝ) (base : ℝ) (apex : ℝ) : Prop :=
  base = a ∧ apex = b / 2

noncomputable def angle_between_lateral_faces (a b : ℝ) : ℝ :=
  real.arccos (1 / 3)

theorem angle_between_lateral_faces_pyramid 
  (a b : ℝ)
  (h_base : ℝ)
  (h_apex : ℝ)
  (h_geometry : pyramid_geometry a b h_base h_apex) :
  angle_between_lateral_faces a b = real.arccos (1 / 3) :=
by sorry

end angle_between_lateral_faces_pyramid_l41_41539


namespace diff_of_two_distinct_members_l41_41402

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41402


namespace sin2theta_plus_cos2theta_l41_41769

theorem sin2theta_plus_cos2theta (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_plus_cos2theta_l41_41769


namespace batsman_average_after_12th_inning_l41_41219

variable (A : ℕ) (total_balls_faced : ℕ)

theorem batsman_average_after_12th_inning 
  (h1 : ∃ A, ∀ total_runs, total_runs = 11 * A)
  (h2 : ∃ A, ∀ total_runs_new, total_runs_new = 12 * (A + 4) ∧ total_runs_new - 60 = 11 * A)
  (h3 : 8 * 4 ≤ 60)
  (h4 : 6000 / total_balls_faced ≥ 130) 
  : (A + 4 = 16) :=
by
  sorry

end batsman_average_after_12th_inning_l41_41219


namespace min_perimeter_polygon_l41_41972

theorem min_perimeter_polygon (P : set (ℝ × ℝ)) :
  (∀ p ∈ P, ∃ s ∈ {line_parallel_x | ∃ t ∈ {line_parallel_y | segment s t}}) ∧
  (∀ q ∈ {x | x^2 + y^2 < 2022}, q ∈ P) →
  ∃ Pmin, Pmin ∈ P ∧ perimeter Pmin = 8 * real.sqrt 2022 :=
by sorry

end min_perimeter_polygon_l41_41972


namespace peter_erasers_l41_41872

theorem peter_erasers : 
  ∀ (initial_erasers : ℕ) (times : ℕ), 
  initial_erasers = 8 → 
  times = 3 → 
  (initial_erasers + times * initial_erasers) = 32 :=
by
  intros initial_erasers times h_initial h_times
  rw [h_initial, h_times]
  simp
  sorry

end peter_erasers_l41_41872


namespace find_fiftieth_term_l41_41553

def is_term_of_sequence (n : ℕ) : Prop :=
  ∃ (bs : ℕ → ℕ) (k : ℕ), n = ∑ i in finset.range k, (bs i) * 3^i ∧ ∀ i j, i < j → bs i ∈ {0, 1} ∧ bs j ∈ {0, 1}

theorem find_fiftieth_term : ∃ n, is_term_of_sequence n ∧ finset.nth (finset.filter is_term_of_sequence (finset.range 1000)) 49 = some n ∧ n = 327 :=
by
  sorry

end find_fiftieth_term_l41_41553


namespace eccentricity_of_hyperbola_l41_41335

variables {a b c e : ℝ} {x y : ℝ}
variables {P F1 F2 : ℝ × ℝ}

-- Conditions
def on_hyperbola (P : ℝ × ℝ) : Prop := 
  let ⟨x, y⟩ := P in
  (x^2)/(a^2) - (y^2)/(b^2) = 1

def semi_focal_distance := c = sqrt (a^2 + b^2)

def right_triangle_condition :
  ((P.1 + 2 * a) - x)^2 = (x + 2 * a)^2 + x^2 - 2 * x * (x + 2 * a) := sorry

def foci := 
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  F1.1^2 + F2.1^2 = (2 * c)^2 - (Area_of_triangle(F1, P, F2) * sqrt(2)) := sorry

def area_triangle := 
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  Area_of_triangle(F1, P, F2) = 2 * a * c := sorry

-- Goal to prove
theorem eccentricity_of_hyperbola :
  on_hyperbola P →
  semi_focal_distance →
  area_triangle →
  e = 1 + sqrt 2 :=
sorry

end eccentricity_of_hyperbola_l41_41335


namespace bounded_infinite_sequence_l41_41721

noncomputable def sequence_x (n : ℕ) : ℝ :=
  4 * (Real.sqrt 2 * n - ⌊Real.sqrt 2 * n⌋)

theorem bounded_infinite_sequence (a : ℝ) (h : a > 1) :
  ∀ i j : ℕ, i ≠ j → (|sequence_x i - sequence_x j| * |(i - j : ℝ)|^a) ≥ 1 := 
by
  intros i j h_ij
  sorry

end bounded_infinite_sequence_l41_41721


namespace not_7_or_8_nice_count_l41_41309

-- Definitions according to the conditions
def is_k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (∃ d, (a : ℕ) ^ k = d ∧ Nat.divisors d = N)

noncomputable def count_k_nice (k : ℕ) (bound : ℕ) : ℕ :=
  ((1 : ℕ) to bound).filter (λ n, (n % k = 1)).length

theorem not_7_or_8_nice_count : 
  let bound := 999 in
  let n7 := count_k_nice 7 bound in
  let n8 := count_k_nice 8 bound in
  let n7_and_n8 := count_k_nice 56 bound in
  bound - (n7 + n8 - n7_and_n8) = 749 :=
by
  sorry

end not_7_or_8_nice_count_l41_41309


namespace real_roots_of_x_squared_minus_four_factorization_of_x_squared_minus_four_l41_41563

theorem real_roots_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
begin
  sorry
end

theorem factorization_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2) :=
begin
  sorry
end

end real_roots_of_x_squared_minus_four_factorization_of_x_squared_minus_four_l41_41563


namespace train_speed_l41_41633

def total_distance (length_train length_bridge : ℕ) : ℕ := length_train + length_bridge

def speed_mps (distance : ℕ) (time : ℕ) : ℝ := distance / time

def speed_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

theorem train_speed :
  let length_train := 110
  let length_bridge := 132
  let time_seconds := 12.099
  let distance := total_distance length_train length_bridge
  let speed_mps := speed_mps distance time_seconds
  speed_kmph speed_mps = 72 := 
by
  -- Proof omitted
  sorry

end train_speed_l41_41633


namespace benoit_wins_l41_41155

theorem benoit_wins :
  ∃ f : list ℕ → list ℕ, (∀ L, (∃ a b L', 
    L = a :: b :: L' ∧ (f (a :: b :: L') = (a * b) :: L' ∨
                        f (a :: b :: L') = (a - b) :: L' ∨
                        f (a :: b :: L') = (a + b) :: L')) ∧
                        ∃ m, ∀ L, m ∈ L ∧ 2023 ∣ m) →
                        (∃ g : list ℕ → list ℕ, ∀ L, 
                          L ≠ [] ∧
                          L ≠ [1..2023] ∧
                          ∃ a b L', L = a :: b :: L' →
                          (g (a :: b :: L') = (a * b) :: L' ∨
                           g (a :: b :: L') = (a - b) :: L' ∨
                           g (a :: b :: L') = (a + b) :: L')) :=
sorry

end benoit_wins_l41_41155


namespace trigonometric_identity_proof_l41_41336

theorem trigonometric_identity_proof {α : ℝ} 
  (h1 : tan (α + π / 4) = 1 / 2) 
  (h2 : -π / 2 < α ∧ α < 0) : 
  sin (2 * α) + 2 * sin (α) ^ 2 = -2 / 5 := 
by sorry

end trigonometric_identity_proof_l41_41336


namespace sum_of_smallest_elements_l41_41832

noncomputable def A_n (n : ℕ) : Set ℝ :=
  {1 / 2 ^ k | k ∈ Finset.range n}

def S (n : ℕ) : ℝ :=
  if n = 2 then 7 / 4 else (n^2 - 1) / 2

theorem sum_of_smallest_elements (n : ℕ) (hn : 2 ≤ n) : 
  S n = 
    if n = 2 then 7 / 4 
    else (n^2 - 1) / 2 :=
sorry

end sum_of_smallest_elements_l41_41832


namespace solve_for_k_l41_41033

theorem solve_for_k (k : ℤ) : (∃ x : ℤ, x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 :=
by
  sorry

end solve_for_k_l41_41033


namespace sum_and_product_of_36_l41_41191

noncomputable def prime_factors_36 : list ℕ := [2, 2, 3, 3]

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (list.range (n + 1)).filter (fun x => n % x = 0), d

def product_of_prime_factors (factors : list ℕ) : ℕ :=
  factors.product

theorem sum_and_product_of_36 :
  sum_of_divisors 36 = 91 ∧ product_of_prime_factors prime_factors_36 = 6 :=
by
  sorry

end sum_and_product_of_36_l41_41191


namespace black_ball_on_second_draw_given_white_ball_on_first_draw_l41_41050

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls

def P_A : ℚ := num_white_balls / total_balls
def P_AB : ℚ := (num_white_balls * num_black_balls) / (total_balls * (total_balls - 1))
def P_B_given_A : ℚ := P_AB / P_A

theorem black_ball_on_second_draw_given_white_ball_on_first_draw : P_B_given_A = 5 / 8 :=
by
  sorry

end black_ball_on_second_draw_given_white_ball_on_first_draw_l41_41050


namespace number_of_distinct_triangles_with_positive_integer_area_l41_41669

def is_valid_point (P : ℕ × ℕ) : Prop := 
  let (x, y) := P in 57 * x + y = 2023

theorem number_of_distinct_triangles_with_positive_integer_area :
  let points := {P : ℕ × ℕ // is_valid_point P}
  let possible_x_values := {x : ℕ // ∃ y : ℕ, is_valid_point (x, y)}
  let even_x_values := {x ∈ possible_x_values | x % 2 = 0}
  let odd_x_values := {x ∈ possible_x_values | x % 2 = 1}
  #even_x_values.choose 2 + #odd_x_values.choose 2 = 153 :=
by
  sorry

end number_of_distinct_triangles_with_positive_integer_area_l41_41669


namespace constructible_triangle_and_area_bound_l41_41831

noncomputable def triangle_inequality_sine (α β γ : ℝ) : Prop :=
  (Real.sin α + Real.sin β > Real.sin γ) ∧
  (Real.sin β + Real.sin γ > Real.sin α) ∧
  (Real.sin γ + Real.sin α > Real.sin β)

theorem constructible_triangle_and_area_bound 
  (α β γ : ℝ) (h_pos : 0 < α) (h_pos_β : 0 < β) (h_pos_γ : 0 < γ)
  (h_sum : α + β + γ < Real.pi)
  (h_ineq1 : α + β > γ)
  (h_ineq2 : β + γ > α)
  (h_ineq3 : γ + α > β) :
  triangle_inequality_sine α β γ ∧
  (Real.sin α * Real.sin β * Real.sin γ) / 4 ≤ (1 / 8) * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) :=
sorry

end constructible_triangle_and_area_bound_l41_41831


namespace zoo_birds_ratio_l41_41928

theorem zoo_birds_ratio :
  ∀ (x : ℕ),
    450 = x + 360 →
    450 / x = 5 :=
by
  intros x h
  have h1 : x = 90 := by linarith
  rw [h1]
  norm_num
  sorry

end zoo_birds_ratio_l41_41928


namespace trains_at_initial_stations_l41_41154

-- Define the durations of round trips for each line.
def red_round_trip : ℕ := 14
def blue_round_trip : ℕ := 16
def green_round_trip : ℕ := 18

-- Define the total time we are analyzing.
def total_time : ℕ := 2016

-- Define the statement that needs to be proved.
theorem trains_at_initial_stations : 
  (total_time % red_round_trip = 0) ∧ 
  (total_time % blue_round_trip = 0) ∧ 
  (total_time % green_round_trip = 0) := 
by
  -- The proof can be added here.
  sorry

end trains_at_initial_stations_l41_41154


namespace horizontal_distance_l41_41970

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Condition: y-coordinate of point P is 8
def P_y : ℝ := 8

-- Condition: y-coordinate of point Q is -8
def Q_y : ℝ := -8

-- x-coordinates of points P and Q solve these equations respectively
def P_satisfies (x : ℝ) : Prop := curve x = P_y
def Q_satisfies (x : ℝ) : Prop := curve x = Q_y

-- The horizontal distance between P and Q is 1
theorem horizontal_distance : ∃ (Px Qx : ℝ), P_satisfies Px ∧ Q_satisfies Qx ∧ |Px - Qx| = 1 :=
by
  sorry

end horizontal_distance_l41_41970


namespace number_of_differences_l41_41391

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41391


namespace solution_set_of_inequality_l41_41920

theorem solution_set_of_inequality 
  (a : ℝ) 
  (h : ∀ t ∈ set.Icc (-1 : ℝ) (1 : ℝ), t^2 - 2 * a * t + a^2 + 2 * a - 3 > 0) :
  a < -2 - real.sqrt 6 ∨ a > real.sqrt 2 := 
sorry

end solution_set_of_inequality_l41_41920


namespace time_spent_giving_bath_l41_41108

theorem time_spent_giving_bath
  (total_time : ℕ)
  (walk_time : ℕ)
  (bath_time blowdry_time : ℕ)
  (walk_distance walk_speed : ℤ)
  (walk_distance_eq : walk_distance = 3)
  (walk_speed_eq : walk_speed = 6)
  (total_time_eq : total_time = 60)
  (walk_time_eq : walk_time = (walk_distance * 60 / walk_speed))
  (half_blowdry_time : blowdry_time = bath_time / 2)
  (time_eq : bath_time + blowdry_time = total_time - walk_time)
  : bath_time = 20 := by
  sorry

end time_spent_giving_bath_l41_41108


namespace correlation_magnitude_binomial_parameters_normal_distribution_prob_l41_41590

-- Definitions and assumptions:

variables {n : ℕ}
variables {p : ℝ}
variables {r : ℝ} -- correlation coefficient
variables {P : ℝ → ℝ} -- probability density function
variables {ξ : ℝ → ℝ} -- normal distribution random variable

-- Conditions:
-- 1. Correlation coefficient and its interpretation
theorem correlation_magnitude (r : ℝ) (hr1 : r ≤ 1) (hr2 : -1 ≤ r) :
  abs r ≤ 1 := sorry

-- 2. Binomial distribution parameters
theorem binomial_parameters (E : ℕ → ℝ) (D : ℕ → ℝ) (X : ℕ → ℕ) (hx1 : E(X n) = 30) 
  (hx2 : D(X n) = 20) :
  p = 1 / 3 := sorry

-- 3. Normal distribution probabilities
theorem normal_distribution_prob (ξ : ℝ) (P : ℝ → ℝ) (hp : P(ξ > 1) = p) :
  P(-1 < ξ < 0) = 1 / 2 - p := sorry

end correlation_magnitude_binomial_parameters_normal_distribution_prob_l41_41590


namespace area_of_equilateral_triangle_in_square_eq_l41_41464

theorem area_of_equilateral_triangle_in_square_eq (a : ℝ) (h_a : a = Real.sqrt 3 - 1)
  (h_square : ∀ (A B C D E F : ℝ), (is_square A B C D 1) ∧ (is_equilateral_triangle A E F)) :
  area_of_triangle A E F = 2 * Real.sqrt 3 - 3 := by sorry

end area_of_equilateral_triangle_in_square_eq_l41_41464


namespace range_of_m_l41_41753

variable (m : ℝ)

def hyperbola (m : ℝ) := (x y : ℝ) → (x^2 / (1 + m)) - (y^2 / (3 - m)) = 1

def eccentricity_condition (m : ℝ) := (2 / (Real.sqrt (1 + m)) > Real.sqrt 2)

theorem range_of_m (m : ℝ) (h1 : 1 + m > 0) (h2 : 3 - m > 0) (h3 : eccentricity_condition m) :
 -1 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l41_41753


namespace problem_correct_statements_l41_41198

theorem problem_correct_statements : 
  (Nat.choose 7 2 = Nat.choose 7 5) ∧
  (Nat.choose 5 3 = Nat.choose 4 2 + Nat.choose 4 3) ∧
  (5 * Nat.factorial 5 = Nat.factorial 6 - Nat.factorial 5) :=
by {
  -- prove each condition step by step using required properties
  sorry
}

end problem_correct_statements_l41_41198


namespace average_of_solutions_l41_41278

-- Define the quadratic equation condition
def quadratic_eq : Prop := ∃ x : ℂ, 3*x^2 - 4*x + 1 = 0

-- State the theorem
theorem average_of_solutions : quadratic_eq → (∃ avg : ℂ, avg = 2 / 3) :=
by
  sorry

end average_of_solutions_l41_41278


namespace unknown_number_is_three_or_twenty_seven_l41_41071

theorem unknown_number_is_three_or_twenty_seven
    (x y : ℝ)
    (h1 : y - 3 = x - y)
    (h2 : (y - 6) / 3 = x / (y - 6)) :
    x = 3 ∨ x = 27 :=
by
  sorry

end unknown_number_is_three_or_twenty_seven_l41_41071


namespace find_g_1_l41_41233

theorem find_g_1 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (2*x - 3) = 2*x^2 - x + 4) : 
  g 1 = 11.5 :=
sorry

end find_g_1_l41_41233


namespace other_equation_l41_41961

-- Define the variables for the length of the rope and the depth of the well
variables (x y : ℝ)

-- Given condition
def cond1 : Prop := (1/4) * x = y + 3

-- The proof goal
theorem other_equation (h : cond1 x y) : (1/5) * x = y + 2 :=
sorry

end other_equation_l41_41961


namespace triangle_BAC_uncertain_l41_41787

theorem triangle_BAC_uncertain 
  (A B C D : Type) [inner_product_space ℝ Type]
  (hABC : affine_independent ℝ ![A, B, C])
  (hAD_altitude : altitude A D B C)
  (hAD_squared : (dist_sq A D) = (dist_sq B D) * (dist_sq C D)) :
  incertain (measuring_angle A B C) :=
sorry

end triangle_BAC_uncertain_l41_41787


namespace sine_monotone_increasing_intervals_l41_41214
open Real

-- Define increasing intervals for sine function
noncomputable def sineIncreasingIntervals (k : ℤ) : Set ℝ :=
  {x : ℝ | 2 * k * π - π / 2 ≤ x ∧ x ≤ 2 * k * π + π / 2}

theorem sine_monotone_increasing_intervals :
  ∀ x : ℝ, (∃ k : ℤ, x ∈ sineIncreasingIntervals k) ↔ (∃ k : ℤ, 2 * k * π - π / 2 ≤ x ∧ x ≤ 2 * k * π + π / 2) :=
by
  intro x
  split
  { intro h
    cases h with k hk
    use k
    exact hk }
  { intro h
    cases h with k hk
    use k
    exact hk }

end sine_monotone_increasing_intervals_l41_41214


namespace smallest_solution_of_quartic_l41_41943

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end smallest_solution_of_quartic_l41_41943


namespace range_f_l41_41146

noncomputable def f (x : ℝ) : ℝ := min (2 - x^2) x

theorem range_f : set.range f = set.Iic 1 := sorry

end range_f_l41_41146


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l41_41300

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) :=
sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l41_41300


namespace diff_of_two_distinct_members_l41_41405

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41405


namespace solution_set_of_inequality_l41_41776

theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 → f x2 ≤ f x1) →
  (f 1 = 0) →
  {x : ℝ | f (x - 3) ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
by
  intros h_even h_mono h_f1
  sorry

end solution_set_of_inequality_l41_41776


namespace empty_plane_speed_l41_41180

variable (V : ℝ)

def speed_first_plane (V : ℝ) : ℝ := V - 2 * 50
def speed_second_plane (V : ℝ) : ℝ := V - 2 * 60
def speed_third_plane (V : ℝ) : ℝ := V - 2 * 40

theorem empty_plane_speed (V : ℝ) (h : (speed_first_plane V + speed_second_plane V + speed_third_plane V) / 3 = 500) : V = 600 :=
by 
  sorry

end empty_plane_speed_l41_41180


namespace coffee_ratio_correct_l41_41134

noncomputable def ratio_of_guests (cups_weak : ℕ) (cups_strong : ℕ) (tablespoons_weak : ℕ) (tablespoons_strong : ℕ) (total_tablespoons : ℕ) : ℤ :=
  if (cups_weak * tablespoons_weak + cups_strong * tablespoons_strong = total_tablespoons) then
    (cups_weak * tablespoons_weak / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong)) /
    (cups_strong * tablespoons_strong / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong))
  else 0

theorem coffee_ratio_correct :
  ratio_of_guests 12 12 1 2 36 = 1 / 2 :=
by
  sorry

end coffee_ratio_correct_l41_41134


namespace range_of_m_l41_41755

noncomputable def set_A (x : ℝ) : ℝ := x^2 - (3 / 2) * x + 1

def A : Set ℝ := {y | ∃ (x : ℝ), x ∈ (Set.Icc (-1/2 : ℝ) 2) ∧ y = set_A x}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m + 1 ∨ x ≤ m - 1}

def sufficient_condition (m : ℝ) : Prop := A ⊆ B m

theorem range_of_m :
  {m : ℝ | sufficient_condition m} = {m | m ≤ -(9 / 16) ∨ m ≥ 3} :=
sorry

end range_of_m_l41_41755


namespace product_form_l41_41849

theorem product_form (x1 y1 x2 y2 : ℤ)
  (a : ℤ) (b : ℤ) (h1 : a = x1^2 - 5*y1^2) (h2 : b = x2^2 - 5*y2^2) :
  ∃ u v : ℤ, a * b = u^2 - 5*v^2 :=
by
  let u := x1 * x2 + 5 * y1 * y2
  let v := x1 * y2 + y1 * x2
  use [u, v]
  sorry

end product_form_l41_41849


namespace unique_sides_triangle_not_isosceles_l41_41453

theorem unique_sides_triangle_not_isosceles (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ isosceles_triangle a b c :=
sorry

def isosceles_triangle (a b c : ℝ) : Prop :=
a = b ∨ b = c ∨ a = c

end unique_sides_triangle_not_isosceles_l41_41453


namespace area_EFGE_l41_41878

def EF : ℝ := 5
def FF' : ℝ := 5
def FG : ℝ := 10
def GG' : ℝ := 10
def GH : ℝ := 9
def HH' : ℝ := 9
def HE : ℝ := 7
def E'E : ℝ := 7
def areaEFGH : ℝ := 12

theorem area_EFGE'_is_36 :
  EF = FF' →
  FG = GG' →
  GH = HH' →
  HE = E'E →
  areaEFGH = 12 →
  let areaE'F'G'H' := 36 in
  areaE'F'G'H' = 36 :=
by
  intros h1 h2 h3 h4 h5
  let areaE'F'G'H' := 36
  exact sorry

end area_EFGE_l41_41878


namespace product_of_roots_abs_eq_l41_41887

theorem product_of_roots_abs_eq (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  x = 5 ∨ x = -5 ∧ ((5 : ℝ) * (-5 : ℝ) = -25) := 
sorry

end product_of_roots_abs_eq_l41_41887


namespace bicycle_speed_l41_41465

theorem bicycle_speed (distance speed_ratio : ℝ) (delay : ℝ) (bicycle_speed : ℝ) :
  distance = 15 ∧ speed_ratio = 4 ∧ delay = 3 / 4 ∧ 
  ((distance / bicycle_speed) - (distance / (speed_ratio * bicycle_speed))) = delay →
  bicycle_speed = 15 :=
by
  intros h
  cases h with h_distance h_rest
  cases h_rest with h_speed_ratio h_rest
  cases h_rest with h_delay h_eq
  sorry

end bicycle_speed_l41_41465


namespace smallest_solution_for_quartic_eq_l41_41940

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end smallest_solution_for_quartic_eq_l41_41940


namespace constant_area_of_triangle_l41_41104

def f (a b x : ℝ) := a * x + b / x
def tangent_line (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := 2 * x - 3 * (f a b (sqrt 3)) + 2 * sqrt 3 = 0

theorem constant_area_of_triangle (a b : ℝ) : a = 1 ∧ b = 1 ∧ tangent_line a b (sqrt 3) :=
sorry

end constant_area_of_triangle_l41_41104


namespace triangle_side_length_l41_41069

theorem triangle_side_length (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB : ℝ) (AC : ℝ) (cos_A : ℝ) :
  AC = 4 ∧ AB = 2 ∧ cos_A = (1/8) →
  ∃ BC : ℝ, BC = 3 * Real.sqrt 2 :=
by {
  intros h,
  obtain ⟨h1, h2, h3⟩ := h,
  have law_of_cosines : BC ^ 2 = AB ^ 2 + AC ^ 2 - 2 * AB * AC * cos_A :=
    by sorry,
  use Real.sqrt 18,
  rw law_of_cosines at *,
  have h4 : Real.sqrt 18 = 3 * Real.sqrt 2 := by sorry,
  rwa h4,
  sorry
}

end triangle_side_length_l41_41069


namespace imaginary_part_of_z_l41_41552

-- Define the imaginary unit.
def i : ℂ := complex.I

-- Define the complex number z.
def z : ℂ := (2 * i^3) / (2 + i)

-- Statement of the proof problem.
theorem imaginary_part_of_z : complex.im z = -4 / 5 := by
  sorry

end imaginary_part_of_z_l41_41552


namespace farey_sequence_consecutiveness_l41_41995

open Rational

def consecutive_fractions (a b c d n : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b ≤ n ∧ 0 < c ∧ c < d ∧ d ≤ n ∧ b * c - a * d = 1

theorem farey_sequence_consecutiveness (a b c d n : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b ≤ n) (h4 : 0 < c) (h5 : c < d) (h6 : d ≤ n) :
  consecutive_fractions a b c d n → |b * c - a * d| = 1 :=
by
  sorry

end farey_sequence_consecutiveness_l41_41995


namespace part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l41_41737

-- Conditions
def quadratic (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * a * x + 2 * a
def point_A (a : ℝ) : ℝ × ℝ := (-1, quadratic a (-1))
def point_B (a : ℝ) : ℝ × ℝ := (3, quadratic a 3)
def line_EF (a : ℝ) : ℝ × ℝ × ℝ × ℝ := ((a - 1), -1, (2 * a + 3), -1)

-- Statements based on solution
theorem part_1 (a : ℝ) :
  (quadratic a (-1)) = -1 := sorry

theorem part_2_max_min (a : ℝ) : 
  a = 1 → 
  (∀ x, -2 ≤ x ∧ x ≤ 3 → 
    (quadratic 1 1 = 3 ∧ 
     quadratic 1 (-2) = -6 ∧ 
     quadratic 1 3 = -1)) := sorry

theorem part_3_length_AC (a : ℝ) (h : a > -1) :
  abs ((2 * a + 1) - (-1)) = abs ((2 * a + 2)) := sorry

theorem part_4_range_a (a : ℝ) : 
  quadratic a (a-1) = -1 ∧ quadratic a (2 * a + 3) = -1 → 
  a ∈ ({-2, -1} ∪ {b : ℝ | b ≥ 0}) := sorry

end part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l41_41737


namespace distance_between_points_l41_41694

-- Define the two points in 3D space
def point1 : ℝ × ℝ × ℝ := (3, 4, 2)
def point2 : ℝ × ℝ × ℝ := (-2, -2, 4)

-- Define the 3-dimensional distance formula
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- State the theorem
theorem distance_between_points :
  distance point1 point2 = real.sqrt 65 :=
by
  -- Skip the proof
  sorry

end distance_between_points_l41_41694


namespace tan_alpha_possible_values_l41_41766

theorem tan_alpha_possible_values (α : ℝ) (h : 5 * sin (2 * α) + 5 * cos (2 * α) + 1 = 0) : 
  tan α = 3 ∨ tan α = -1/2 :=
by 
  sorry

end tan_alpha_possible_values_l41_41766


namespace min_total_distance_l41_41727

variable {P : ℝ × ℝ}
variable {A : ℝ × ℝ}
variable {B : ℝ × ℝ}

noncomputable def PA (P : ℝ × ℝ) : ℝ := 
  P.1

noncomputable def PB (P : ℝ × ℝ) : ℝ := 
  abs (P.1 - P.2 + 4) / real.sqrt 2

theorem min_total_distance : 
  ∀ (P : ℝ × ℝ), (P.2)^2 = 4 * P.1 → 
  ∃ y : ℝ, 
  P = (y^2 / 4, y) ∧ PA P + PB P = (real.sqrt 2 + 1) * y^2 / (4 * real.sqrt 2) - y / (real.sqrt 2) + 2 * real.sqrt 2 ∧
  PA P + PB P = (5 * real.sqrt 2) / 2 - 1 :=
by
  sorry

end min_total_distance_l41_41727


namespace six_digit_numbers_with_zero_l41_41015

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41015


namespace circle_radius_l41_41225

theorem circle_radius (M N : ℝ) (hM : M = Real.pi * r ^ 2) (hN : N = 2 * Real.pi * r) (h : M / N = 15) : r = 30 := by
  sorry

end circle_radius_l41_41225


namespace probability_of_circle_l41_41870

theorem probability_of_circle :
  let numCircles := 4
  let numSquares := 3
  let numTriangles := 3
  let totalFigures := numCircles + numSquares + numTriangles
  let probability := numCircles / totalFigures
  probability = 2 / 5 :=
by
  sorry

end probability_of_circle_l41_41870


namespace sum_due_is_correct_l41_41209

-- Define constants for Banker's Discount and True Discount
def BD : ℝ := 288
def TD : ℝ := 240

-- Define Banker's Gain as the difference between BD and TD
def BG : ℝ := BD - TD

-- Define the sum due (S.D.) as the face value including True Discount and Banker's Gain
def SD : ℝ := TD + BG

-- Create a theorem to prove the sum due is Rs. 288
theorem sum_due_is_correct : SD = 288 :=
by
  -- Skipping proof with sorry; expect this statement to be true based on given conditions 
  sorry

end sum_due_is_correct_l41_41209


namespace angle_equality_l41_41324

theorem angle_equality {A B C D E F G : Point} 
  (h1 : incircle A B C D E F)
  (h2 : midpoint G D E)
  : ∠E F C = ∠G F D :=
sorry

end angle_equality_l41_41324


namespace number_of_diffs_l41_41364

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41364


namespace cone_lateral_surface_area_l41_41038

-- Definitions based on conditions
def base_radius : ℝ := 2
def slant_height : ℝ := 3

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ := π * r * l

-- Theorem stating the problem
theorem cone_lateral_surface_area : lateral_surface_area base_radius slant_height = 6 * π :=
by
  sorry

end cone_lateral_surface_area_l41_41038


namespace TriangleAreaBQW_l41_41809

def is_rectangle (A B C D : Type) (AB CD : nat) := sorry

variables (A B C D Z W Q : Type)
variables (AB Length : nat)
variables (area_trapezoid : nat)

theorem TriangleAreaBQW :
  is_rectangle A B C D AB 16 →
  AZ = 8 →
  WC = 8 →
  AB = 16 →
  area_trapezoid Z W C D = 160 →
  ∃ area_triangle, area_triangle = 48 :=
sorry

end TriangleAreaBQW_l41_41809


namespace range_d1_d2_l41_41328

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

noncomputable def distance_from_origin_to_line (P A : ℝ × ℝ) : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let (px, py) := P
  let (ax, ay) := A
  let numerator := abs (py * ax - px * ay)
  let denominator := sqrt (ax^2 + ay^2)
  numerator / denominator

theorem range_d1_d2
  (A1 A2 P : ℝ × ℝ)
  (P_on_hyperbola : hyperbola P.1 P.2)
  (A1_on_hyperbola : A1 = (-sqrt 2, 0))
  (A2_on_hyperbola : A2 = (sqrt 2, 0))
  (P_not_vertices : P ≠ A1 ∧ P ≠ A2)
  (d1 := distance_from_origin_to_line P A1)
  (d2 := distance_from_origin_to_line P A2) :
  0 < d1 * d2 ∧ d1 * d2 < 1 :=
sorry

end range_d1_d2_l41_41328


namespace five_less_than_sixty_percent_of_cats_l41_41174

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end five_less_than_sixty_percent_of_cats_l41_41174


namespace tangent_line_at_x1_l41_41548

noncomputable def curve (x : ℝ) : ℝ := x^3 + Real.log x

def derivative (x : ℝ) : ℝ := 3 * x^2 + 1 / x

theorem tangent_line_at_x1 : ∃ m b, (m = 4 ∧ b = -3) ∧ ∀ x y, y = curve x → y = 4 * x - 3 := by
  sorry

end tangent_line_at_x1_l41_41548


namespace min_bodyguards_l41_41075

theorem min_bodyguards (n : ℕ) (h : ∀ (A B : ℕ), A ≠ B → ∃ C : ℕ, C ≠ A ∧ C ≠ B ∧ C defeats A ∧ C defeats B) : n ≥ 7 := sorry

end min_bodyguards_l41_41075


namespace find_t_l41_41358

def vector := (ℝ × ℝ)

def a : vector := (-3, 4)
def b : vector := (-1, 5)
def c : vector := (2, 3)

def parallel (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_t (t : ℝ) : 
  parallel (a.1 - c.1, a.2 - c.2) ((2 * t) + b.1, (3 * t) + b.2) ↔ t = -24 / 17 :=
by
  sorry

end find_t_l41_41358


namespace num_unique_differences_l41_41392

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41392


namespace bounded_expression_l41_41847

theorem bounded_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := 
sorry

end bounded_expression_l41_41847


namespace min_value_increase_of_add_x2_l41_41196

-- Definitions for the conditions
variable (f : ℝ → ℝ)
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (x : ℝ)

-- Given conditions
def condition1 : Prop :=
  ∀ x, f(x) = a * x^2 + b * x + c ∧ a > 0 ∧
    ((-b^2 / (4 * (a + 3)) + c) = (-b^2 / (4 * a) + c + 9))

def condition2 : Prop :=
  ∀ x, f(x) = a * x^2 + b * x + c ∧ a > 0 ∧
    ((-b^2 / (4 * (a - 1)) + c) = (-b^2 / (4 * a) + c - 9))

-- The proof problem
theorem min_value_increase_of_add_x2 (h1 : condition1 f a b c) (h2 : condition2 f a b c) :
  ∀ x, let new_f := fun x => (a + 1) * x^2 + b * x + c in
    ((-b^2 / (4 * (a + 1)) + c) = (-18 + c + 9/2)) :=
sorry

end min_value_increase_of_add_x2_l41_41196


namespace largest_interior_angle_of_triangle_l41_41452

theorem largest_interior_angle_of_triangle (exterior_ratio_2k : ℝ) (exterior_ratio_3k : ℝ) (exterior_ratio_4k : ℝ) (sum_exterior_angles : exterior_ratio_2k + exterior_ratio_3k + exterior_ratio_4k = 360) :
  180 - exterior_ratio_2k = 100 :=
by
  sorry

end largest_interior_angle_of_triangle_l41_41452


namespace fermats_little_theorem_l41_41595

theorem fermats_little_theorem (n p : ℕ) [hp : Fact p.Prime] : p ∣ (n^p - n) :=
sorry

end fermats_little_theorem_l41_41595


namespace incircle_center_line_proof_l41_41277
  
open Real EuclideanGeometry

variables {A B C D M N : Point ℝ} 

-- Define cyclic trapezoid and its properties
def is_cyclic_trapezoid (A B C D : Point ℝ) : Prop :=
  cyclic A B C D ∧ parallel (Line.mk A B) (Line.mk C D) ∧ dist A B > dist C D

-- Define incircle tangency points
def tangent_points (A B C : Point ℝ) (M : Point ℝ) (N : Point ℝ) : Prop :=
  tangent_circle (incircle_triangle A B C) (Line.mk A B) M ∧
  tangent_circle (incircle_triangle A B C) (Line.mk A C) N

-- Define the center of the incircle lying on the line condition
def incircle_center_on_line (A B C D M N : Point ℝ) : Prop :=
  ∃ O : Point ℝ, center (incircle_trapezoid A B C D) = O ∧ 
                 collinear {M, N, O}

-- The main theorem
theorem incircle_center_line_proof 
  (A B C D M N : Point ℝ)
  (h1 : is_cyclic_trapezoid A B C D)
  (h2 : tangent_points A B C M N) :
  incircle_center_on_line A B C D M N :=
sorry

end incircle_center_line_proof_l41_41277


namespace range_a_l41_41747

def f (x : ℝ) : ℝ := 
  if x ≤ 1 then (1/10) * x + 1 
  else Real.log x - 1

def A : Set ℝ := {a : ℝ | ∃ x : ℝ, f x = a * x}

theorem range_a :
  A = {a : ℝ | a ≤ -1} ∪ {a : ℝ | a ≥ 1.1} ∪ {a : ℝ | a = 1 / Real.exp 2} :=
sorry

end range_a_l41_41747


namespace area_of_triangle_BQW_l41_41806

open Real

variables (A B C D Z W Q : ℝ × ℝ)
variables (AB AZ WC : ℝ)
variables (area_ZWCD : ℝ)

-- Define the conditions
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  -- Assuming standard coordinate rectangle conditions
  (A.1 = D.1) ∧ (B.1 = C.1) ∧ (A.2 = B.2) ∧ (D.2 = C.2) ∧ 
  (A.1 ≠ B.1) ∧ (A.2 ≠ D.2)

def point_distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def trapezoid_area (A B C D : ℝ × ℝ) (height: ℝ) : ℝ :=
  (1/2) * (point_distance A B + point_distance C D) * height

-- Define the necessary relationships
def BQW_area (A B Z W : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((A.1 * (B.2 - Z.2) + B.1 * (Z.2 - A.2) + Z.1 * (A.2 - B.2)))

theorem area_of_triangle_BQW :
  is_rectangle A B C D →
  point_distance A Z = 8 →
  point_distance W C = 8 →
  point_distance A B = 16 →
  trapezoid_area Z W C D 10 = 160 →
  BQW_area B Z W = 48 :=
by
  sorry

end area_of_triangle_BQW_l41_41806


namespace find_wheel_diameter_l41_41247

noncomputable def wheel_diameter (revolutions distance : ℝ) (π_approx : ℝ) : ℝ := 
  distance / (π_approx * revolutions)

theorem find_wheel_diameter : wheel_diameter 47.04276615104641 4136 3.14159 = 27.99 :=
by
  sorry

end find_wheel_diameter_l41_41247


namespace max_seq_length_l41_41091

theorem max_seq_length (n : ℕ) :
  (∃ a : ℕ → ℕ, ∀ i, a i ∈ finset.range (n + 1) ∧ a i ≠ a (i + 1)
    ∧ (∀ p q r s, p < q → q < r → r < s → a p = a r → a p ≠ a q → a q = a s → false))
  → ∃ k, k = 4 * n - 2 :=
begin
  intro h,
  use (4 * n - 2),
  sorry,
end

end max_seq_length_l41_41091


namespace average_multiples_of_6_l41_41692

theorem average_multiples_of_6 : 
  (let multiples := setOf (λ n : ℕ, n % 6 = 0 ∧ 1 ≤ n ∧ n ≤ 100) in
  let sum := Finset.sum (Finset.filter (λ n, n ∈ multiples) (Finset.range 101)) in
  let count := Finset.card (Finset.filter (λ n, n ∈ multiples) (Finset.range 101)) in
  count = 16 ∧ sum = 816 ∧ sum / count = 51) := 
by
  sorry

end average_multiples_of_6_l41_41692


namespace olaf_total_toy_cars_l41_41864

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end olaf_total_toy_cars_l41_41864


namespace trapezoid_diagonals_l41_41874

theorem trapezoid_diagonals {A B C D : Type*} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
  (α β : ℝ) (hαβ : α < β) 
  (h_trapezoid : ∃ (AB : A) (BC : B) (CD : C) (DA : D), 
     trapezoid A B C D ∧ angle A B = α ∧ angle B C = β) : 
  diagonal A C > diagonal B D := 
sorry

end trapezoid_diagonals_l41_41874


namespace largest_divisor_of_n4_minus_n2_l41_41150

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n4_minus_n2_l41_41150


namespace min_value_change_when_add_x2_l41_41193

-- Given conditions
variables {a b c : ℝ}
variable {f : ℝ → ℝ} 

-- Assume f is a quadratic polynomial of the form ax^2 + bx + c
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the modified polynomials
def f_add_3x2 (x : ℝ) : ℝ := (a + 3) * x^2 + b * x + c
def f_sub_x2 (x : ℝ) : ℝ := (a - 1) * x^2 + b * x + c
def f_add_x2 (x : ℝ) : ℝ := (a + 1) * x^2 + b * x + c

-- Minimum values of the polynomials according to the conditions
def min_f : ℝ := -b^2 / (4 * a) + c
def min_f_add_3x2 : ℝ := -b^2 / (4 * (a + 3)) + c
def min_f_sub_x2 : ℝ := -b^2 / (4 * (a - 1)) + c
def min_f_add_x2 : ℝ := -b^2 / (4 * (a + 1)) + c

-- Statement of the problem
theorem min_value_change_when_add_x2
  (h1 : min_f_add_3x2 = min_f + 9)
  (h2 : min_f_sub_x2 = min_f - 9) :
  min_f_add_x2 = min_f + 9 / 2 :=
begin
  -- Proof goes here
  sorry
end

end min_value_change_when_add_x2_l41_41193


namespace count_multiples_of_4_between_300_and_700_l41_41762

noncomputable def num_multiples_of_4_in_range (a b : ℕ) : ℕ :=
  (b - (b % 4) - (a - (a % 4) + 4)) / 4 + 1

theorem count_multiples_of_4_between_300_and_700 : 
  num_multiples_of_4_in_range 301 699 = 99 := by
  sorry

end count_multiples_of_4_between_300_and_700_l41_41762


namespace longest_side_length_of_quadrilateral_l41_41159

-- Define the conditions
def x_plus_y_leq_3 (x y : ℝ) : Prop := x + y ≤ 3
def two_x_plus_y_geq_2 (x y : ℝ) : Prop := 2 * x + y ≥ 2
def x_geq_0 (x : ℝ) : Prop := x ≥ 0
def y_geq_0 (y : ℝ) : Prop := y ≥ 0

-- Define the problem statement
theorem longest_side_length_of_quadrilateral :
  (x_plus_y_leq_3 ∧ two_x_plus_y_geq_2 ∧ x_geq_0 ∧ y_geq_0) → ∃ (length : ℝ), length = 3 * real.sqrt 2 :=
sorry

end longest_side_length_of_quadrilateral_l41_41159


namespace sqrt_three_irrational_l41_41642

-- Define what it means for a number to be irrational
def irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- Given numbers
def neg_two : ℝ := -2
def one_half : ℝ := 1 / 2
def sqrt_three : ℝ := real.sqrt 3
def two : ℝ := 2

-- The proof statement
theorem sqrt_three_irrational : irrational sqrt_three :=
sorry

end sqrt_three_irrational_l41_41642


namespace TriangleAreaBQW_l41_41808

def is_rectangle (A B C D : Type) (AB CD : nat) := sorry

variables (A B C D Z W Q : Type)
variables (AB Length : nat)
variables (area_trapezoid : nat)

theorem TriangleAreaBQW :
  is_rectangle A B C D AB 16 →
  AZ = 8 →
  WC = 8 →
  AB = 16 →
  area_trapezoid Z W C D = 160 →
  ∃ area_triangle, area_triangle = 48 :=
sorry

end TriangleAreaBQW_l41_41808


namespace beads_per_necklace_l41_41285

theorem beads_per_necklace (n : ℕ) (b : ℕ) (total_beads : ℕ) (total_necklaces : ℕ)
  (h1 : total_necklaces = 6) (h2 : total_beads = 18) (h3 : b * total_necklaces = total_beads) :
  b = 3 :=
by {
  sorry
}

end beads_per_necklace_l41_41285


namespace contest_average_score_l41_41526

noncomputable def average_score : ℕ :=
  let x := 17 in
  let y := 12 in
  let z := 8 in
  let total_participants := x + y + z - 2 - 15 in
  let total_score := 17 * 20 + (12 + 8) * 25 in
  total_score / total_participants

theorem contest_average_score :
  let x := 17 in
  let y := 12 in
  let z := 8 in
  let total_participants := 20 in
  let total_score := 17 * 20 + (12 + 8) * 25 in
  total_score / total_participants = 42 :=
begin
  let x := 17,
  let y := 12,
  let z := 8,
  let total_participants := 20,
  let total_score := 17 * 20 + (12 + 8) * 25,
  have h1 : total_score = 840, by norm_num,
  have h2 : total_participants = 20, by norm_num,
  rw h1,
  rw h2,
  exact dec_trivial,
end

end contest_average_score_l41_41526


namespace count_special_three_digit_numbers_l41_41430

-- Define the range for three-digit numbers
def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Define the predicate for at least one digit being 5 or 8
def contains_five_or_eight (n : ℕ) : Prop :=
  (∃ d ∈ to_digit_list n, d = 5 ∨ d = 8)

-- Collect the three-digit numbers that satisfy the condition
def filtered_numbers := {n : ℕ | n ∈ three_digit_numbers ∧ contains_five_or_eight n}

-- The theorem we want to prove
theorem count_special_three_digit_numbers : finset.card filtered_numbers = 452 :=
by
  sorry

end count_special_three_digit_numbers_l41_41430


namespace recurrence_int_and_divisibility_l41_41566

noncomputable def recurrence_rel (a : ℕ → ℤ) (k : ℤ) : Prop :=
∀ n: ℕ, a n - k * a (n + 1) + int.sqrt ((k^2 - 1) * a (n + 1)^2 + 1) = 0

theorem recurrence_int_and_divisibility (a : ℕ → ℤ) (k : ℤ) 
  (h : recurrence_rel a k) (h0 : a 0 = 0)
  : (∀ n : ℕ, ∃ m : ℤ, a n = m) ∧ (∀ n : ℕ, 2 * k ∣ a (2 * n)) :=
by
  sorry

end recurrence_int_and_divisibility_l41_41566


namespace number_of_diffs_l41_41360

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41360


namespace count_integer_points_on_parabola_l41_41622

def Q (x y : ℤ) : Prop := y = (4 * x^2) / 9

def valid_point (x y : ℤ) : Prop := Q x y ∧ |3 * x + 4 * y| ≤ 1200

def integer_points_on_Q := { (x, y) : ℤ × ℤ | valid_point x y }

def num_integer_points_on_Q := integer_points_on_Q.to_finset.card

theorem count_integer_points_on_parabola :
  num_integer_points_on_Q = 42 :=
sorry

end count_integer_points_on_parabola_l41_41622


namespace least_number_of_pennies_l41_41588

theorem least_number_of_pennies (a : ℕ) :
  (a ≡ 1 [MOD 7]) ∧ (a ≡ 0 [MOD 3]) → a = 15 := by
  sorry

end least_number_of_pennies_l41_41588


namespace find_lambda_range_l41_41767

noncomputable def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
(v.1 + w.1, v.2 + w.2)

noncomputable def scalar_mul (λ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
(λ * v.1, λ * v.2)

noncomputable def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
(v.1 - w.1, v.2 - w.2)

theorem find_lambda_range (λ : ℝ) : 
  ∀ (A B C D : ℝ × ℝ),
  A = (4, 2) →
  B = (3, 5) →
  C = (5, 1) →
  (D = vector_add (vector_sub B A) (scalar_mul λ (vector_sub C A))) →
  (0 < D.1 ∧ 0 < D.2) →
  -3 < λ ∧ λ < 5 := 
by
  intros A B C D hA hB hC hD hD_ineq
  sorry

end find_lambda_range_l41_41767


namespace batsman_average_increase_l41_41220

def average_increase (avg_before : ℕ) (runs_12th_inning : ℕ) (avg_after : ℕ) : ℕ :=
  avg_after - avg_before

theorem batsman_average_increase :
  ∀ (avg_before runs_12th_inning avg_after : ℕ),
    (runs_12th_inning = 70) →
    (avg_after = 37) →
    (11 * avg_before + runs_12th_inning = 12 * avg_after) →
    average_increase avg_before runs_12th_inning avg_after = 3 :=
by
  intros avg_before runs_12th_inning avg_after h_runs h_avg_after h_total
  sorry

end batsman_average_increase_l41_41220


namespace determine_alpha_l41_41741

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the actual definition of f

theorem determine_alpha (α : ℝ) (k : ℤ) (condition1 : 0 ≤ α ∧ α < π/2) 
  (condition2 : ∀ x y, x ≥ y → f ((x + y) / 2) = f x * _.sin α + (1 - _.sin α) * f y) 
  (condition3 : f 0 = 0) 
  (condition4 : f 1 = 1) 
  (eqn : f (1 / 4) = 1 / 4) : 
  ∃ k : ℤ, α = 2 * k * π + π / 6 := 
by 
  sorry

end determine_alpha_l41_41741


namespace num_valid_labelings_l41_41680

-- Definitions for the problem
def label_set : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14}

def is_adjacent (a b : ℕ) : Prop :=
  ¬((a % 2 = 0 ∧ b % 2 = 0) ∨ (a % 3 = 0 ∧ b % 3 = 0) ∨ (a % 5 = 0 ∧ b % 5 = 0))

def is_valid_labeling (labeling : Fin 5 → ℕ) : Prop :=
  (∀ i j : Fin 5, i ≠ j → is_adjacent (labeling i) (labeling j)) ∧
  (∀ i : Fin 5, labeling i ∈ label_set) ∧
  (Function.Bijective labeling)

-- Statement of the problem
theorem num_valid_labelings : 
  ∃ (n : ℕ), n = 520 ∧ (n = Fintype.card { l : Fin 5 → ℕ // is_valid_labeling l }) := 
by 
  sorry

end num_valid_labelings_l41_41680


namespace rectangular_prism_volume_l41_41627

theorem rectangular_prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : a * c = 6) (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := 
sorry

end rectangular_prism_volume_l41_41627


namespace minimum_distance_l41_41913

noncomputable def minDistPointOnLineToPointOnCircle : Real := 
  let line := λy, y = 2
  let circle := λx y, (x - 1)^2 + y^2 = 1
  -- The minimum distance we want to prove
  1

-- The statement of the problem
theorem minimum_distance : ∃ (p q : ℝ × ℝ), 
  (p.snd = 2) ∧ ((q.fst - 1)^2 + q.snd^2 = 1) ∧ 
  ∀(r s : ℝ × ℝ), (r.snd = 2) ∧ ((s.fst - 1)^2 + s.snd^2 = 1) -> 
    dist ℝ ℝ p q ≤ dist ℝ ℝ r s ∧ dist ℝ ℝ p q = 1 :=
by
  -- Proof will be provided
  sorry

end minimum_distance_l41_41913


namespace time_to_complete_job_l41_41578

theorem time_to_complete_job (x : ℝ) (hx : 0 < x) :
  (1 / x + 1 / 6 = 5 / 12) → x = 4 :=
begin
  -- The proof will go here.
  sorry
end

end time_to_complete_job_l41_41578


namespace shift_sin_to_cos_l41_41148

theorem shift_sin_to_cos (phi : ℝ) (h_phi : 0 < phi) :
  ∀ (x : ℝ), sin (x - phi - π / 6) = cos x ↔ phi = 4 * π / 3 :=
by sorry

end shift_sin_to_cos_l41_41148


namespace polynomial_evaluation_l41_41688

theorem polynomial_evaluation (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) : y^3 - 3 * y^2 - 9 * y + 7 = 7 := 
  sorry

end polynomial_evaluation_l41_41688


namespace tangent_secant_distance_l41_41789

-- The Circle and geometry definitions would be prerequisites
noncomputable def circle_radius {R : ℝ} : Prop :=
∃ c : ℝ → ℝ → Prop, (∀ x y : ℝ, c x y = (x - 0)^2 + (y - 0)^2 - R^2 = 0)

noncomputable def chord_length (R : ℝ) (r : ℝ) : Prop :=
r = R / 2

noncomputable def tangent_parallel_secant_distance (R : ℝ) : ℝ :=
  let l := R / 2 in
  let distance := R / 8 in
  distance

theorem tangent_secant_distance (R : ℝ) (h : circle_radius R) (l : chord_length R (R / 2)) :
  tangent_parallel_secant_distance R = R / 8 :=
begin
  sorry
end

end tangent_secant_distance_l41_41789


namespace compound_interest_principal_l41_41916

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (Real.exp (T * Real.log (1 + R / 100)) - 1)

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem compound_interest_principal :
  let P_SI := 2800.0000000000027
  let R_SI := 5
  let T_SI := 3
  let P_CI := 4000
  let R_CI := 10
  let T_CI := 2
  let SI := simple_interest P_SI R_SI T_SI
  let CI := 2 * SI
  CI = compound_interest P_CI R_CI T_CI → P_CI = 4000 :=
by
  intros
  sorry

end compound_interest_principal_l41_41916


namespace max_distance_line_eq_l41_41343

theorem max_distance_line_eq :
  let A := (-1, 1)
  let B := (2, -1)
  (∀ l1: ℝ × ℝ × ℝ, let eq := λ x y, l1.1 * x + l1.2 * y + l1.3 = 0 
                     in eq 0 0 = 0 → eq 1 1 = 0) → 
  ∃ l2: ℝ × ℝ × ℝ, let eq2 := λ x y, l2.1 * x + l2.2 * y + l2.3 = 0 
                    in eq2 (-1) 1 = 0 ∧ 
                       (d : ℝ) := max_dist B eq2,
                       (∀ eq_line1 eq_line2, (max_dist B eq_line1 <= d) → eq2 2 (-1) = 0 ) :=
  ∃ (a b c : ℝ), a = 3 ∧ b = -2 ∧ c = 5 ∧ (λ x y : ℝ, 3 * x - 2 * y + 5 = 0)
sorry

end max_distance_line_eq_l41_41343


namespace count_two_digit_numbers_l41_41432

theorem count_two_digit_numbers : 
  ∃ n, n = 45 ∧ ∀ t u : ℕ, (1 ≤ t ∧ t ≤ 9) → (0 ≤ u ∧ u ≤ 9) → (t > u) → 
  n = (finrange 1 10).sum (λ t, 9 - t) := 
begin
  sorry
end

end count_two_digit_numbers_l41_41432


namespace set_difference_proof_l41_41649

/--
Among the four options representing sets below, there is one set that is different from the other three:
A: { x | x = 0 }
B: { a | a^2 ≠ 0 }
C: { a = 0 }
D: { 0 }
Prove that the set in option C is different from the other three sets.
-/
theorem set_difference_proof :
  (∀ (x : ℝ), x ∈ {x | x = 0} → x = 0) ∧
  (∀ (a : ℝ), a ∈ {a | a^2 ≠ 0} → a^2 ≠ 0) ∧
  (¬ ∃ (a : ℝ), {a = 0}) ∧
  (∀ (x : ℝ), x ∈ {0} → x = 0) ∧
  ((∃ (a : ℝ), {a | a^2 ≠ 0}) ∧ (({a = 0} = {0}) = false)) :=
by
  sorry

end set_difference_proof_l41_41649


namespace circle_equation_l41_41224

/-- A circle C passes through point A (4, 1) and is tangent to the line x - y - 1 = 0 at point B (2, 1). 
    Prove that the equation of circle C is (x - 3)^2 + y^2 = 2. -/
theorem circle_equation :
  ∃ (C : ℝ × ℝ → Prop), 
  (C (4, 1)) ∧ 
  (∃ x y : ℝ, C (x, y) ∧ (x = 2 ∧ y = 1) ∧ tangent_line (x, y) (λ p, p.1 - p.2 - 1 = 0)) ∧ 
  ∀ x y, C (x, y) ↔ (x - 3)^2 + y^2 = 2 :=
sorry

end circle_equation_l41_41224


namespace samuel_initial_speed_l41_41880

/-
Samuel is driving to San Francisco’s Comic-Con in his car and he needs to travel 600 miles to the hotel where he made a reservation. 
He drives at a certain speed for 3 hours straight, then he speeds up to 80 miles/hour for 4 hours. 
Now, he is 130 miles away from the hotel. What was his initial speed?
-/

theorem samuel_initial_speed : 
  ∃ v : ℝ, (3 * v + 320 = 470) ↔ (v = 50) :=
by
  use 50
  /- detailed proof goes here -/
  sorry

end samuel_initial_speed_l41_41880


namespace difference_of_distinct_members_l41_41408

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41408


namespace units_digit_of_u_1985_l41_41500

def nearest_integer (x : ℝ) : ℤ := 
  if x - 0.5 ≤ Int.floor x + 0.5 then Int.floor x 
  else Int.ceil x

noncomputable def sequence_u : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_u n + nearest_integer (sequence_u n * real.sqrt 2).toReal.to_Int

theorem units_digit_of_u_1985 : (sequence_u 1984 % 10) = 9 :=
  sorry

end units_digit_of_u_1985_l41_41500


namespace critical_temperature_of_water_l41_41542

/--
Given the following conditions:
1. The temperature at which solid, liquid, and gaseous water coexist is the triple point.
2. The temperature at which water vapor condenses is the condensation point.
3. The maximum temperature at which liquid water can exist.
4. The minimum temperature at which water vapor can exist.

Prove that the critical temperature of water is the maximum temperature at which liquid water can exist.
-/
theorem critical_temperature_of_water :
    ∀ (triple_point condensation_point maximum_liquid_temp minimum_vapor_temp critical_temp : ℝ), 
    (critical_temp = maximum_liquid_temp) ↔
    ((critical_temp ≠ triple_point) ∧ (critical_temp ≠ condensation_point) ∧ (critical_temp ≠ minimum_vapor_temp)) := 
  sorry

end critical_temperature_of_water_l41_41542


namespace decode_ciphertext_correct_l41_41958

-- Definitions of transformation rules
def plaintext_to_ciphertext (x : ℕ) (h : 1 ≤ x ∧ x ≤ 26) : ℕ :=
  if odd x then (x + 1) / 2 else x / 2 + 13

noncomputable def ciphertext_to_plaintext (y : ℕ) (h : 1 ≤ y ∧ y ≤ 26) : ℕ :=
  if y ≤ 13 then 2 * y - 1 else 2 * y - 26

-- Prove that decoding the ciphertext "shxc" results in the plaintext "love"
theorem decode_ciphertext_correct :
  let seq_s := 19;
      seq_h := 8;
      seq_x := 24;
      seq_c := 3;
      plaintext_s := ciphertext_to_plaintext seq_s ⟨Nat.le_refl 19, Nat.le_of_lt (by norm_num)⟩;
      plaintext_h := ciphertext_to_plaintext seq_h ⟨Nat.le_refl 8, Nat.le_of_lt (by norm_num)⟩;
      plaintext_x := ciphertext_to_plaintext seq_x ⟨Nat.le_refl 24, Nat.le_of_lt (by norm_num)⟩;
      plaintext_c := ciphertext_to_plaintext seq_c ⟨Nat.le_refl 3, Nat.le_of_lt (by norm_num)⟩
  in (plaintext_s, plaintext_h, plaintext_x, plaintext_c) = (12, 15, 22, 5) ∧
     (['l', 'o', 'v', 'e'].nth plaintext_s).get_or_else ' ' = 'l' ∧
     (['l', 'o', 'v', 'e'].nth plaintext_h).get_or_else ' ' = 'o' ∧
     (['l', 'o', 'v', 'e'].nth plaintext_x).get_or_else ' ' = 'v' ∧
     (['l', 'o', 'v', 'e'].nth plaintext_c).get_or_else ' ' = 'e' :=
  sorry

end decode_ciphertext_correct_l41_41958


namespace Alyssa_nuggets_l41_41991

theorem Alyssa_nuggets :
  ∃ A : ℕ, A + 2 * A + 2 * A = 100 ∧ A = 20 :=
by
  use 20
  split
  simp
  norm_num
  sorry

end Alyssa_nuggets_l41_41991


namespace intersection_of_lines_l41_41188

theorem intersection_of_lines : ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 5 = 15 * x - 2 ∧ x = 1 / 3 ∧ y = 0 :=
by
  sorry

end intersection_of_lines_l41_41188


namespace train_speed_eq_l41_41981

/--
  Prove that the speed of the train equals 56.00448 km/h given the following conditions:
  The train is 90 m long.
  The train crosses a platform in 18 sec.
  The length of the platform is 190.0224 m.
-/
theorem train_speed_eq {
  (length_train : ℝ) (time_cross : ℝ) (length_platform : ℝ)
  (total_distance := length_train + length_platform) 
  (speed_m_s := total_distance / time_cross) 
  (speed_kmh := speed_m_s * 3.6) :
  length_train = 90 ∧ time_cross = 18 ∧ length_platform = 190.0224 → 
  speed_kmh ≈ 56.00448 :=
by 
  sorry

end train_speed_eq_l41_41981


namespace number_of_distinct_positive_differences_l41_41423

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41423


namespace sum_of_reduced_proper_fractions_with_denominator_100_l41_41618

-- Define the gcd function on natural numbers
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the Euler's Totient function
def eulerTotient (n : ℕ) : ℕ :=
  ((List.range (n + 1)).filter (fun k => gcd k n = 1)).length

-- Define the sum of reciprocals of co-prime numbers to a denominator
def sumReducedProperFractions (denom : ℕ) : ℕ :=
  let nums := (List.range denom).filter (fun k => gcd k denom = 1)
  let numPairs := nums.length / 2
  numPairs

-- Define the main theorem
theorem sum_of_reduced_proper_fractions_with_denominator_100 :
  sumReducedProperFractions 100 = 20 := by
  sorry

end sum_of_reduced_proper_fractions_with_denominator_100_l41_41618


namespace possible_values_of_sum_of_reciprocals_l41_41439

theorem possible_values_of_sum_of_reciprocals {a b : ℝ} (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b = 4 := 
by 
  sorry

end possible_values_of_sum_of_reciprocals_l41_41439


namespace flea_never_lands_on_all_points_l41_41927

noncomputable def a_n (n : ℕ) : ℕ := (n * (n + 1) / 2) % 300

theorem flea_never_lands_on_all_points :
  ∃ k : ℕ, k < 300 ∧ ∀ n : ℕ, a_n n ≠ k :=
sorry

end flea_never_lands_on_all_points_l41_41927


namespace historical_fiction_new_releases_fraction_l41_41265

noncomputable def HF_fraction_total_inventory : ℝ := 0.4
noncomputable def Mystery_fraction_total_inventory : ℝ := 0.3
noncomputable def SF_fraction_total_inventory : ℝ := 0.2
noncomputable def Romance_fraction_total_inventory : ℝ := 0.1

noncomputable def HF_new_release_percentage : ℝ := 0.35
noncomputable def Mystery_new_release_percentage : ℝ := 0.60
noncomputable def SF_new_release_percentage : ℝ := 0.45
noncomputable def Romance_new_release_percentage : ℝ := 0.80

noncomputable def historical_fiction_new_releases : ℝ := HF_fraction_total_inventory * HF_new_release_percentage
noncomputable def mystery_new_releases : ℝ := Mystery_fraction_total_inventory * Mystery_new_release_percentage
noncomputable def sf_new_releases : ℝ := SF_fraction_total_inventory * SF_new_release_percentage
noncomputable def romance_new_releases : ℝ := Romance_fraction_total_inventory * Romance_new_release_percentage

noncomputable def total_new_releases : ℝ :=
  historical_fiction_new_releases + mystery_new_releases + sf_new_releases + romance_new_releases

theorem historical_fiction_new_releases_fraction :
  (historical_fiction_new_releases / total_new_releases) = (2 / 7) :=
by
  sorry

end historical_fiction_new_releases_fraction_l41_41265


namespace determine_position_correct_l41_41591

def determine_position (option : String) : Prop :=
  option = "East longitude 120°, North latitude 30°"

theorem determine_position_correct :
  determine_position "East longitude 120°, North latitude 30°" :=
by
  sorry

end determine_position_correct_l41_41591


namespace arctan_sum_pi_over_two_l41_41662

theorem arctan_sum_pi_over_two : 
  Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
by sorry

end arctan_sum_pi_over_two_l41_41662


namespace cot_diff_eq_l41_41701

theorem cot_diff_eq (x : Real) : 
  (cot (x / 3) - cot x = (sin (2 * x / 3)) / (sin (x / 3) * sin x)) →
  ∃ k : Real, ∀ x, (cot (x / 3) - cot x = (sin (k * x)) / (sin (x / 3) * sin x)) ∧ k = 2 / 3 :=
by sorry

end cot_diff_eq_l41_41701


namespace jordan_shots_in_fourth_period_l41_41824

theorem jordan_shots_in_fourth_period :
  let F := 21 - (4 + 2 * 4 + (2 * 4 - 3)) in
  F = 4 :=
by
  sorry

end jordan_shots_in_fourth_period_l41_41824


namespace find_irrational_l41_41646

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem find_irrational : 
  ∃ x ∈ ({-2, 1/2, real.sqrt 3, 2} : set ℝ), is_irrational x ∧ ∀ y ∈ ({-2, 1/2, real.sqrt 3, 2} : set ℝ), y ≠ x → is_rational y :=
by {
  sorry
}

end find_irrational_l41_41646


namespace particle_speed_constant_l41_41241

-- Define the position of the particle at time t
def position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 5 * t - 7)

-- Calculate the change in position over a unit time interval
def delta_position (t : ℝ) : ℝ × ℝ := (position (t + 1).1 - position t.1, position (t + 1).2 - position t.2)

-- Define the magnitude of the velocity vector
noncomputable def speed (t : ℝ) : ℝ := Real.sqrt ((delta_position t).1 ^ 2 + (delta_position t).2 ^ 2)

-- Theorem stating that the speed is constant and equals sqrt(34)
theorem particle_speed_constant : ∀ t : ℝ, speed t = Real.sqrt 34 :=
by
  sorry

end particle_speed_constant_l41_41241


namespace bottle_caps_total_l41_41291

-- Mathematical conditions
def x : ℕ := 18
def y : ℕ := 63

-- Statement to prove
theorem bottle_caps_total : x + y = 81 :=
by
  -- The proof is skipped as indicated by 'sorry'
  sorry

end bottle_caps_total_l41_41291


namespace find_f_lg_lg2_l41_41346

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 4

theorem find_f_lg_lg2 :
  f (Real.logb 10 (2)) = 3 :=
sorry

end find_f_lg_lg2_l41_41346


namespace K_time_9_hours_l41_41826

theorem K_time_9_hours
  (x : ℝ) -- x is the speed of K
  (hx : 45 / x = 9) -- K's time for 45 miles is 9 hours
  (y : ℝ) -- y is the speed of M
  (h₁ : x = y + 0.5) -- K travels 0.5 mph faster than M
  (h₂ : 45 / y - 45 / x = 3 / 4) -- K takes 3/4 hour less than M
  : 45 / x = 9 :=
by
  sorry

end K_time_9_hours_l41_41826


namespace probability_xiaoming_l41_41923

variable (win_probability : ℚ) 
          (xiaoming_goal : ℕ)
          (xiaojie_goal : ℕ)
          (rounds_needed_xiaoming : ℕ)
          (rounds_needed_xiaojie : ℕ)

def probability_xiaoming_wins_2_consecutive_rounds
   (win_probability : ℚ) 
   (rounds_needed_xiaoming : ℕ) : ℚ :=
  (win_probability ^ 2) + 
  2 * win_probability ^ 3 * (1 - win_probability) + 
  win_probability ^ 4

theorem probability_xiaoming :
    win_probability = (1/2) ∧ 
    rounds_needed_xiaoming = 2 ∧
    rounds_needed_xiaojie = 3 →
    probability_xiaoming_wins_2_consecutive_rounds (1 / 2) 2 = 7 / 16 :=
by
  -- Proof steps placeholder
  sorry

end probability_xiaoming_l41_41923


namespace decreasing_interval_b_l41_41348

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_interval_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici (Real.sqrt 2) → ∀ x1 x2 : ℝ, x1 ∈ Set.Ici (Real.sqrt 2) → x2 ∈ Set.Ici (Real.sqrt 2) → 
   x1 ≤ x2 → f x1 b ≥ f x2 b) ↔ b ≤ 2 :=
by
  sorry

end decreasing_interval_b_l41_41348


namespace length_of_DE_l41_41974

variables (A B C D E : Type)
variables [Field A]

-- Defining the given parameters
def AB : A := 8
def AD : A := 10
def area_rectangle : A := AB * AD
def area_triangle : A := area_rectangle / 2

-- Using the Pythagorean theorem
def DC : A := AD
def CE : A := (area_triangle * 2) / DC

theorem length_of_DE (h1 : DC = 10) (h2 : CE = 8) : 
  let DE := Math.sqrt (DC^2 + CE^2)
  in DE = 2 * (Math.sqrt 41) := 
sorry

end length_of_DE_l41_41974


namespace unique_real_solution_l41_41573

theorem unique_real_solution : ∃ x : ℝ, (∀ t : ℝ, x^2 - t * x + 36 = 0 ∧ x^2 - 8 * x + t = 0) ∧ x = 3 :=
by
  sorry

end unique_real_solution_l41_41573


namespace problem1_problem2_l41_41216

-- Problem 1
theorem problem1 
  (V : Type) 
  (C D : Set V) 
  (h_disjoint: Disjoint C D)
  (𝒞 : Set (Set C)) 
  (𝒟 : Set (Set D)) 
  (part_C: Partition 𝒞)
  (part_D: Partition 𝒟)
  : q(𝒞, 𝒟) ≥ q(C, D) := 
sorry

-- Problem 2
theorem problem2 
  (V : Type) 
  (𝒫 𝒫' : Set (Set V)) 
  (part_P: Partition 𝒫)
  (part_P': Partition 𝒫')
  (refine_P_P' : Refinement 𝒫' 𝒫) 
  : q(𝒫') ≥ q(𝒫) :=
sorry

end problem1_problem2_l41_41216


namespace binomial_expansion_coefficient_l41_41317

theorem binomial_expansion_coefficient (n : ℕ) (h : n ∈ Nat.Prime) :
  (∀ k₁ k₂, (k₁ ≠ k₂ → ∃ r, k₁ = r + 1 ∧ k₂ = r + 4) →
  ((binomial n 2) = (binomial n 5)) →
  ∃ n = 7 →
  (∑ i in range n, binomial n i * ((x - 2/(sqrt x)) ^ (n - i)) * x = 560)) :=
sorry

end binomial_expansion_coefficient_l41_41317


namespace train_pass_bridge_time_l41_41985

-- Given conditions
def train_length : ℕ := 460  -- length in meters
def bridge_length : ℕ := 140  -- length in meters
def speed_kmh : ℝ := 45  -- speed in kilometers per hour

-- Prove that the time to pass the bridge is 48 seconds
theorem train_pass_bridge_time :
  let distance := train_length + bridge_length in
  let speed_ms := speed_kmh * (1000 / 3600) in
  (distance / speed_ms) = 48 :=
by
  sorry

end train_pass_bridge_time_l41_41985


namespace numberOfThreeDigitNumbersWithAtLeastOne5Or8_l41_41427

def isThreeDigitWholeNumber (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 999

def containsAtLeastOne5or8 (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10]
  digits.any (λ d, d = 5 ∨ d = 8)

theorem numberOfThreeDigitNumbersWithAtLeastOne5Or8 : 
  (count (λ n, isThreeDigitWholeNumber n ∧ containsAtLeastOne5or8 n) (range 1000)) = 452 := 
sorry

end numberOfThreeDigitNumbersWithAtLeastOne5Or8_l41_41427


namespace candy_cut_into_square_l41_41126

theorem candy_cut_into_square (A : ℝ) 
  (cut : ℕ → ℝ) 
  (hA : A > 0) 
  (hcut : ∀ i, i < 8 → cut i = (A / 8)) : 
  ∃ (T1 T2 : list ℝ), 
    length T1 = 4 ∧ length T2 = 4 ∧ 
    (∀ t1 ∈ T1, t1 = (A / 8)) ∧ 
    (∀ t2 ∈ T2, t2 = (A / 8)) ∧ 
    (assemble_into_square (T1 ++ T2) = A)
:= sorry

end candy_cut_into_square_l41_41126


namespace diff_of_two_distinct_members_l41_41407

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41407


namespace six_digit_numbers_with_zero_l41_41008

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41008


namespace diff_of_two_distinct_members_l41_41403

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41403


namespace value_of_b_l41_41585

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, (-x^2 + b * x - 7 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by
  sorry

end value_of_b_l41_41585


namespace total_tips_l41_41267

/-- 
The waiter had 10 customers, 5 of whom did not leave a tip. 
The remaining customers each left a $3 tip. 
Prove that the total amount of money earned in tips is $15. 
-/
theorem total_tips : 
  let total_customers := 10 in
  let no_tip_customers := 5 in
  let tip_amount := 3 in
  let tipping_customers := total_customers - no_tip_customers in
  let total_earned := tipping_customers * tip_amount in
  total_earned = 15 :=
by
  sorry

end total_tips_l41_41267


namespace volume_of_truncated_triangular_pyramid_l41_41903

variable {a b H α : ℝ} (h1 : H = Real.sqrt (a * b))

theorem volume_of_truncated_triangular_pyramid
  (h2 : H = Real.sqrt (a * b))
  (h3 : 0 < a)
  (h4 : 0 < b)
  (h5 : 0 < H)
  (h6 : 0 < α) :
  (volume : ℝ) = H^3 * Real.sqrt 3 / (4 * (Real.sin α)^2) := sorry

end volume_of_truncated_triangular_pyramid_l41_41903


namespace range_of_a_two_distinct_extreme_points_l41_41748

noncomputable def f (a x : ℝ) : ℝ := log x + a * x^2 - 2 * x

theorem range_of_a_two_distinct_extreme_points (a : ℝ) : 
  0 < a ∧ a < 1/2 → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ 
  (deriv (f a) x1 = 0) ∧ (deriv (f a) x2 = 0) :=
sorry

end range_of_a_two_distinct_extreme_points_l41_41748


namespace necessary_but_not_sufficient_condition_l41_41042

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  (f(0) = 0) → is_odd_function f ↔ (∀ x, f (-x) = -f (x)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l41_41042


namespace max_teams_unique_ranking_l41_41891

-- Define the problem context and conditions
def answer_points (n : ℕ) : list ℕ :=
  list.range' 1 n

def correct_answered (n : ℕ) : ℕ := 
  n

-- maximum number of teams that could have participated in the quiz 
-- such that teams can be ranked according to any preference by the experts
theorem max_teams_unique_ranking : ∀ (teams questions : ℕ), questions = 50 → teams ≤ 50 :=
by
  intros teams questions hq
  sorry

end max_teams_unique_ranking_l41_41891


namespace pirate_coins_distribution_l41_41967

theorem pirate_coins_distribution :
  ∃ y : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → (y * (1 - k / 15) ^ k) ∈ ℕ) 
  → (15^(14-14)) * y = 15^7 := sorry

end pirate_coins_distribution_l41_41967


namespace sqrt_expression_meaningful_domain_l41_41777

theorem sqrt_expression_meaningful_domain {x : ℝ} (h : 3 - x ≥ 0) : x ≤ 3 := by
  sorry

end sqrt_expression_meaningful_domain_l41_41777


namespace hog_cat_problem_l41_41172

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end hog_cat_problem_l41_41172


namespace num_diff_positive_integers_l41_41372

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41372


namespace Petya_meets_Vasya_l41_41520

def Petya_speed_on_paved (v_g : ℝ) : ℝ := 3 * v_g

def Distance_to_bridge (v_g : ℝ) : ℝ := 3 * v_g

def Vasya_travel_time (v_g t : ℝ) : ℝ := v_g * t

def Total_distance (v_g : ℝ) : ℝ := 2 * Distance_to_bridge v_g

def New_distance (v_g t : ℝ) : ℝ := (Total_distance v_g) - 2 * Vasya_travel_time v_g t

def Relative_speed (v_g : ℝ) : ℝ := v_g + v_g

def Time_to_meet (v_g : ℝ) : ℝ := (New_distance v_g 1) / Relative_speed v_g

theorem Petya_meets_Vasya (v_g : ℝ) : Time_to_meet v_g + 1 = 2 := by
  sorry

end Petya_meets_Vasya_l41_41520


namespace cos_value_of_angle_l41_41314

theorem cos_value_of_angle (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
by
  sorry

end cos_value_of_angle_l41_41314


namespace max_sum_of_four_distinct_with_lcm_165_l41_41151

theorem max_sum_of_four_distinct_with_lcm_165 (a b c d : ℕ)
  (h1 : Nat.lcm a b = 165)
  (h2 : Nat.lcm a c = 165)
  (h3 : Nat.lcm a d = 165)
  (h4 : Nat.lcm b c = 165)
  (h5 : Nat.lcm b d = 165)
  (h6 : Nat.lcm c d = 165)
  (h7 : a ≠ b) (h8 : a ≠ c) (h9 : a ≠ d)
  (h10 : b ≠ c) (h11 : b ≠ d) (h12 : c ≠ d) :
  a + b + c + d ≤ 268 := sorry

end max_sum_of_four_distinct_with_lcm_165_l41_41151


namespace math_problem_proof_l41_41338

variable {f : ℝ → ℝ}

-- Definitions

def domain_of_f := ∀ x : ℝ, true

def symmetric_about_point :=
  ∀ x : ℝ, f (2 - x) = -f x

def satisfies_condition := 
  ∀ x : ℝ, f (x + 3) = f (1 - x)

def odd_function (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def symmetric_about_y :=
  ∀ x : ℝ, f (-x) = f x

def periodic_function (p : ℝ) :=
  ∀ x : ℝ, f (x + p) = f x

def g_periodic (g : ℝ → ℝ) :=
  ∀ x : ℝ, g (x + 4) = g x

def sum_g_2024 (g : ℝ → ℝ) :=
  (∑ k in finset.range 2024, λ k, g (k + 1)) = 4048

-- Theorem statement

theorem math_problem_proof :
  (domain_of_f) →
  (symmetric_about_point) →
  (satisfies_condition) →
  odd_function (λ x, f (x + 1)) ∧
  symmetric_about_y ∧
  ¬ periodic_function 2 ∧
  (∀ g : ℝ → ℝ, (∀ x : ℝ, g x + f(x + 3) = 2) → g_periodic g → sum_g_2024 g) :=
by sorry

end math_problem_proof_l41_41338


namespace olaf_total_toy_cars_l41_41863

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end olaf_total_toy_cars_l41_41863


namespace vectors_opposite_direction_l41_41784

noncomputable def opposite_direction_vectors (x : ℝ) : Prop :=
  let a : ℝ × ℝ := (1, -x)
  let b : ℝ × ℝ := (x, -6)
  ∃ k : ℝ, k < 0 ∧ a = (k • b.fst, k • b.snd)

theorem vectors_opposite_direction (x : ℝ) (h : opposite_direction_vectors x) :
  x = -sqrt 6 :=
sorry

end vectors_opposite_direction_l41_41784


namespace main_inequality_l41_41082

def is_prime (n : ℕ) : Prop := nat.prime n

def p_i (i : ℕ) : ℕ := nat.prime_seq i

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def a_k (k : ℕ) : ℕ := 
  ∑ t in finset.range (nat.sqrt k).succ, 
    if divides (p_i t * p_i (t + 1)) k then 1 else 0

theorem main_inequality (n : ℕ) (h : 0 < n) : 
  (∑ k in finset.range n, a_k k) < n / 3 := 
sorry

end main_inequality_l41_41082


namespace solve_expression_l41_41945

theorem solve_expression : ( (3^0 - 2 + 4^2) ^ (-1) * 6 ) = (2 / 5) := 
begin
  sorry
end

end solve_expression_l41_41945


namespace cubic_polynomial_greater_than_zero_l41_41311

theorem cubic_polynomial_greater_than_zero (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 → x > 1 :=
sorry

end cubic_polynomial_greater_than_zero_l41_41311


namespace six_digit_numbers_with_zero_l41_41011

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41011


namespace find_m_l41_41344

theorem find_m (m : ℝ) (h : m > 0) :
  (∃ (p : ℝ) (d : ℝ), p = (1/(2 * m)) 
    ∧ d = (1/2) 
    ∧ abs ((-1/(4 * m)) - 0) = d) → m = 1/2 :=
by
  intros,
  sorry

end find_m_l41_41344


namespace min_ab_value_l41_41333

variable (a b : ℝ)

theorem min_ab_value (h1 : a > -1) (h2 : b > -2) (h3 : (a+1) * (b+2) = 16) : a + b ≥ 5 :=
by
  sorry

end min_ab_value_l41_41333


namespace index_sets_l41_41709

theorem index_sets (n : ℕ) (A : Fin n → Finset (Fin n)) (h : ∀ i, (A i).card = n) :
  ∃ (a : Fin n → Fin n → Fin n), 
  (∀ i, A i = {a i j | j }) ∧ 
  (∀ j, (Finset.univ.image (λ i, a i j)).card = n) :=
sorry

end index_sets_l41_41709


namespace min_m_n_l41_41315

/-- Given a function f(x) = log₂(x - 2), and real numbers m and n such that
    f(m) + f(2n) = 3, prove the minimum value of m + n is 7. -/
theorem min_m_n (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.logBase 2 (x - 2)) (m n : ℝ)
  (h_cond : f m + f (2 * n) = 3) : m + n = 7 :=
sorry

end min_m_n_l41_41315


namespace find_y_l41_41036

theorem find_y (x y : ℝ) (h : x = 180) (h1 : 0.25 * x = 0.10 * y - 5) : y = 500 :=
by sorry

end find_y_l41_41036


namespace solution1_solution2_l41_41659

noncomputable def Problem1 : ℝ :=
  4 + (-2)^3 * 5 - (-0.28) / 4

theorem solution1 : Problem1 = -35.93 := by
  sorry

noncomputable def Problem2 : ℚ :=
  -1^4 - (1/6) * (2 - (-3)^2)

theorem solution2 : Problem2 = 1/6 := by
  sorry

end solution1_solution2_l41_41659


namespace circumscribed_polygon_inequality_l41_41494

theorem circumscribed_polygon_inequality (R : ℝ) (n : ℕ) (hRpos : R > 0) (hnpos : n > 0):
  (n+1) * R * real.cos (real.pi / (n + 1 + 1)) - n * R * real.cos (real.pi / (n + 1)) > R :=
sorry

end circumscribed_polygon_inequality_l41_41494


namespace price_per_glass_second_day_l41_41602

-- Define the problem conditions
variables (O W : ℕ) (H : O = W) (P : ℝ)

-- Define the volumes of orangeade on both days
def volume_first_day := 2 * O
def volume_second_day := O + 2 * W

-- Define the price per glass on the first day and the revenue equation
def price_first_day := 0.90
def price_second_day := P

-- Define the glasses sold on both days and the revenue equality
variables (G1 G2 : ℕ) (revenue_equality : G1 * price_first_day = G2 * price_second_day)
(glas_sold_relation: G2 = (3/2) * G1)

-- The theorem to prove
theorem price_per_glass_second_day : P = 0.60 :=
by
  sorry

end price_per_glass_second_day_l41_41602


namespace range_of_t_l41_41606

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 else -x^2

theorem range_of_t (t : ℝ) (x : ℝ) (hx : x ∈ Set.Ioo (t^2 - 4) t) :
  f(x + t) < 4 * f(x) → 0 ≤ t ∧ t ≤ 1 :=
sorry

end range_of_t_l41_41606


namespace problem_statement_l41_41441

theorem problem_statement (x y : ℝ) (h : |x - log y| + sin (π * x) = x + log y) : x = 0 ∧ exp (-1 / 2) ≤ y ∧ y ≤ exp (1 / 2) :=
  sorry

end problem_statement_l41_41441


namespace triangles_at_chord_intersections_l41_41858

theorem triangles_at_chord_intersections (n : ℕ) (h : n = 9) : 
  (∑ t in (finset.powerset_len 3 (finset.powerset_len 4 (finset.range n))), 
    1 : ℕ) = 328750 := 
by -- skipping proof with sorry
  sorry

end triangles_at_chord_intersections_l41_41858


namespace tetrahedron_division_suitable_displacement_counts_suitable_displacement_and_reflection_counts_l41_41581

theorem tetrahedron_division:
  ∃ S : finset (finset ℝ^3), 
    S.card = 24 ∧ 
    (∀ part ∈ S, ∃ t : ℝ^3, finset.image (λ p, p + t) part ⊆ S ∧ finset.image (λ p, -p + t) part ⊆ S) :=
sorry

theorem suitable_displacement_counts:
  ∃ S : finset (finset ℝ^3), 
    (∀ part ∈ S, ∃ t : list ℝ^3, t.length = 11 ∧ (finset.image (λ p, p + t.nth 1) part ⊆ S)) :=
sorry
  
theorem suitable_displacement_and_reflection_counts:
  ∃ S : finset (finset ℝ^3), 
    (∀ part ∈ S, ∃ t : list ℝ^3, t.length = 23 
    ∧ (finset.image (λ p, p + t.nth 1) part ⊆ S ∧ finset.image (λ p, -p + t.nth 1) part ⊆ S)) :=
sorry

end tetrahedron_division_suitable_displacement_counts_suitable_displacement_and_reflection_counts_l41_41581


namespace new_triangle_area_l41_41889

noncomputable def g (x : ℝ) : ℝ := sorry

variables (a b c : ℝ)

def tri_area (points : list (ℝ × ℝ)) : ℝ := sorry

-- Hypothesis
axiom h : tri_area [(a, g a), (b, g b), (c, g c)] = 45

-- Theorem statement
theorem new_triangle_area :
  tri_area [(3 * a, 3 * g a), (3 * b, 3 * g b), (3 * c, 3 * g c)] = 405 := 
sorry

end new_triangle_area_l41_41889


namespace arrangements_not_next_to_each_other_l41_41169

theorem arrangements_not_next_to_each_other (A B : Type) (people : Finset A) (h_card : people.card = 5) :
  (∃ (arrangements : Finset (List A)), arrangements.card = 72 ∧ 
  ∀ l ∈ arrangements, ∀ i : ℕ, ¬((l.nth i = some B ∧ l.nth (i+1) = some A) ∨ (l.nth i = some A ∧ l.nth (i+1) = some B))) := sorry

end arrangements_not_next_to_each_other_l41_41169


namespace oranges_given_to_helen_l41_41761

theorem oranges_given_to_helen :
  ∀ (initial_oranges final_oranges : ℕ),
  initial_oranges = 9 →
  final_oranges = 38 →
  final_oranges - initial_oranges = 29 := 
by
  intros initial_oranges final_oranges h_initial h_final
  rw [h_initial, h_final]
  exact dec_trivial


end oranges_given_to_helen_l41_41761


namespace det_evaluation_l41_41286

noncomputable def matrix_det : ℕ := 
  (Matrix.det ![
    ![1, x, x^2],
    ![1, x + y, (x + y)^2],
    ![1, y, y^2]
  ])

theorem det_evaluation (x y : ℝ) : matrix_det = xy^2 - yx^2 - 2x^2y := 
  sorry

end det_evaluation_l41_41286


namespace number_of_distinct_positive_differences_l41_41416

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41416


namespace find_projection_result_l41_41757

noncomputable def vector1 : ℝ × ℝ := (3, -2)
noncomputable def vector2 : ℝ × ℝ := (2, 5)
noncomputable def direction_vector : ℝ × ℝ := (-1, 7)
noncomputable def projection_result : ℝ × ℝ := (133 / 50, 49 / 50)

theorem find_projection_result :
  let p := projection_result in
  let v1 := vector1 in
  let v2 := vector2 in
  let dir := direction_vector in
  ∃ t : ℝ, (p = (v1.1 - t * dir.1, v1.2 + t * dir.2)) ∧ (0 = (v1.1 - t * dir.1) * dir.1 + (v1.2 + t * dir.2) * dir.2) :=
sorry

end find_projection_result_l41_41757


namespace population_of_males_l41_41470

theorem population_of_males (total_population : ℕ) (num_parts : ℕ) (part_population : ℕ) 
  (male_population : ℕ) (female_population : ℕ) (children_population : ℕ) :
  total_population = 600 →
  num_parts = 4 →
  part_population = total_population / num_parts →
  children_population = 2 * male_population →
  male_population = part_population →
  male_population = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_of_males_l41_41470


namespace part1_part2_l41_41740

variable {n : ℕ} (hn : n > 0) 

-- Assume the given conditions
def S (n : ℕ) : ℝ := ∑ i in range n, a i
axiom sequence_condition (n : ℕ) (hn : n > 0) : S n + 1 / 2 * a n = 1
axiom a_1 : a 1 = 2 / 3

noncomputable def a : ℕ → ℝ
| 0     := 0
| (n+1) := if h : n + 1 = 1 then 2 / 3 else 2 * (1 / 3) ^ (n + 1)

def b (n : ℕ) : ℝ := loga (1 / 3) ((1 - S n) / 3)

noncomputable def T (n : ℕ) : ℝ :=
∑ i in range n, 1 / (b i * b (i + 1))

theorem part1 : ∀ n : ℕ, n > 0 → a n = 2 * (1 / 3) ^ n :=
by sorry

theorem part2 : ∀ n : ℕ, n > 0 → T n = n / (2 * (n + 2)) :=
by sorry

end part1_part2_l41_41740


namespace unique_digit_for_prime_l41_41917

theorem unique_digit_for_prime (B : ℕ) (hB : B < 10) (hprime : Nat.Prime (30420 * 10 + B)) : B = 1 :=
sorry

end unique_digit_for_prime_l41_41917


namespace point_on_x_axis_l41_41049

theorem point_on_x_axis (a : ℝ) (P : ℝ × ℝ) (h : P = (3 + a, a - 5)) : (P.2 = 0) → a = 5 :=
by
  intro hy
  simp at hy
  exact eq_of_sub_eq_zero hy

end point_on_x_axis_l41_41049


namespace product_divisible_by_sum_l41_41560

theorem product_divisible_by_sum (m n : ℕ) (h : ∃ k : ℕ, m * n = k * (m + n)) : m + n ≤ Nat.gcd m n * Nat.gcd m n := by
  sorry

end product_divisible_by_sum_l41_41560


namespace triangle_inequality_l41_41080

variable (a b c R : ℝ)

-- Assuming a, b, c as the sides of a triangle
-- and R as the circumradius.

theorem triangle_inequality:
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R * R)) :=
by
  sorry

end triangle_inequality_l41_41080


namespace num_unique_differences_l41_41398

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41398


namespace journey_99_infeasible_journey_100_feasible_l41_41926

/--
  There are 100 cities. There is a non-stop flight between every pair of cities with positive 
  cost. The flights costs are symmetrical and the average flight cost is 1 tugrik. 

  Prove or disprove:
  a) For m=99, it is not always possible to complete a journey of 99 flights (visiting 99 different cities starting and ending in the same city) within 99 tugriks.
  b) For m=100, it is always possible to complete a journey of 100 flights (visiting 100 different cities starting and ending in the same city) within 100 tugriks.
-/
theorem journey_99_infeasible :
  ∀ (cities : Finset ℕ) (cost_matrix : ℕ → ℕ → ℝ),
  cities.card = 100 →
  (∀ i j, 0 < cost_matrix i j) →
  (∀ i j, cost_matrix i j = cost_matrix j i) →
  (Finset.average (Finset.product cities cities) (λ p, cost_matrix p.1 p.2)) = 1 →
  ¬(∀ (route : List ℕ), route.length = 100 → (route.nodup ∧ (route.head = route.last)) →
      (∑ i in Finset.range (route.length - 1), cost_matrix (route.nth_le i sorry) (route.nth_le (i+1) sorry)) ≤ 99) :=
sorry

theorem journey_100_feasible :
  ∀ (cities : Finset ℕ) (cost_matrix : ℕ → ℕ → ℝ),
  cities.card = 100 →
  (∀ i j, 0 < cost_matrix i j) →
  (∀ i j, cost_matrix i j = cost_matrix j i) →
  (Finset.average (Finset.product cities cities) (λ p, cost_matrix p.1 p.2)) = 1 →
  (∃ (route : List ℕ), route.length = 101 ∧ (route.nodup_except_last ∧ (route.head = route.last)) ∧
      (∑ i in Finset.range (route.length - 1), cost_matrix (route.nth_le i sorry) (route.nth_le (i+1) sorry)) ≤ 100) :=
sorry

end journey_99_infeasible_journey_100_feasible_l41_41926


namespace quartic_divides_circle_l41_41882

theorem quartic_divides_circle (k : ℝ) :
  (∀ x y : ℝ, x^4 + k * x^3 * y - 6 * x^2 * y^2 - k * x * y^3 + y^4 = 0) →
  (x^2 + y^2 = 1) →
  dividing_circle_into_parts(x, y, 8) :=
by
  sorry

end quartic_divides_circle_l41_41882


namespace minimum_questionnaires_l41_41954

theorem minimum_questionnaires (p : ℝ) (r : ℝ) (n_min : ℕ) (h1 : p = 0.65) (h2 : r = 300) :
  n_min = ⌈r / p⌉ ∧ n_min = 462 := 
by
  sorry

end minimum_questionnaires_l41_41954


namespace min_value_expression_l41_41488

theorem min_value_expression (a b : ℝ) : ∃ v : ℝ, ∀ (a b : ℝ), (a^2 + a * b + b^2 - a - 2 * b) ≥ v ∧ v = -1 :=
by
  sorry

end min_value_expression_l41_41488


namespace proving_statement_l41_41707

variable {a b x₀ y₀ : ℝ}

def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def circle (x y : ℝ) (b : ℝ) : Prop :=
  x^2 + y^2 = b^2

def point_on_ellipse (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  ellipse P.1 P.2 a b

def points_of_tangency (P A B : ℝ × ℝ) (b : ℝ) : Prop :=
  circle A.1 A.2 b ∧ circle B.1 B.2 b

def line_AB (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ (Q : ℝ × ℝ), (B.2 - A.2) * (Q.1 - A.1) = (B.1 - A.1) * (Q.2 - A.2)

def intersection_with_axes (A B : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let M := (b^2 / A.1, 0)
  let N := (0, b^2 / A.2)
  (M, N)

theorem proving_statement (P A B : ℝ × ℝ)
  (hP : point_on_ellipse P a b)
  (hT : points_of_tangency P A B b) :
  let (M, N) := intersection_with_axes A B
  in (a^2 / N.2^2) + (b^2 / M.1^2) = (a^2 / b^2) :=
sorry

end proving_statement_l41_41707


namespace modular_inverse_28_mod_29_l41_41298

theorem modular_inverse_28_mod_29 :
  28 * 28 ≡ 1 [MOD 29] :=
by
  sorry

end modular_inverse_28_mod_29_l41_41298


namespace simplify_and_evaluate_expression_l41_41884

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 2022) (h2 : y = -sqrt 2) :
  4 * x * y + (2 * x - y) * (2 * x + y) - (2 * x + y) ^ 2 = -4 :=
by
  sorry

end simplify_and_evaluate_expression_l41_41884


namespace six_digit_numbers_with_zero_l41_41018

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41018


namespace circle_bisector_AM_eq_AN_l41_41540

theorem circle_bisector_AM_eq_AN {A B C M N D : Point} {circle : Circle} (h_triangle : Triangle A B C)
  (h_bisector : IsBisector A D)
  (h_diameter : circle.diameter = A D)
  (h_intersect_M : circle.Intersects A B M ∧ M ≠ A)
  (h_intersect_N : circle.Intersects A C N ∧ N ≠ A) :
  dist A M = dist A N := 
sorry

end circle_bisector_AM_eq_AN_l41_41540


namespace value_of_m_l41_41554
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 3)

theorem value_of_m (m : ℝ) (x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2 : ℝ, x1 > x2 → y m x1 < y m x2) :
  m = 2 :=
sorry

end value_of_m_l41_41554


namespace percentage_additional_cost_rush_delivery_l41_41705

-- Given definitions from the problem's conditions
def original_cost : ℝ := 40
def cost_per_type_with_rush : ℝ := 13
def pack_size : ℕ := 4

-- The proof statement we want to show
theorem percentage_additional_cost_rush_delivery : 
  let total_cost_with_rush := pack_size * cost_per_type_with_rush in
  let additional_cost := total_cost_with_rush - original_cost in
  (additional_cost / original_cost) * 100 = 30 := by
  sorry

end percentage_additional_cost_rush_delivery_l41_41705


namespace proper_subsets_count_l41_41504

theorem proper_subsets_count (A : Set (Fin 4)) (h : A = {1, 2, 3}) : 
  ∃ n : ℕ, n = 7 ∧ ∃ (S : Finset (Set (Fin 4))), S.card = n ∧ (∀ B, B ∈ S → B ⊂ A) := 
by {
  sorry
}

end proper_subsets_count_l41_41504


namespace single_stroke_drawing_characterization_l41_41596

theorem single_stroke_drawing_characterization (G : Type) [Graph G] :
  (∃ f : G → G → bool, ∀ v ∈ G, f v v = false ∧ ∀ u v w ∈ G, f u v = true → f v w = true → f u w = false) ↔
  (∃ odd_nodes : G → bool, G.sum odd_nodes ≤ 2) :=
by
  sorry

end single_stroke_drawing_characterization_l41_41596


namespace prob_triangle_inequality_l41_41178

theorem prob_triangle_inequality (x y z : ℕ) (h1 : 1 ≤ x ∧ x ≤ 6) (h2 : 1 ≤ y ∧ y ≤ 6) (h3 : 1 ≤ z ∧ z ≤ 6) : 
  (∃ (p : ℚ), p = 37 / 72) := 
sorry

end prob_triangle_inequality_l41_41178


namespace range_of_x_plus_2y_l41_41316

theorem range_of_x_plus_2y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) : x + 2 * y ≥ 9 :=
sorry

end range_of_x_plus_2y_l41_41316


namespace round_robin_schedule_l41_41320

theorem round_robin_schedule (m : ℕ) (h : m ≥ 17) :
  ∃ schedule : list (ℕ → ℕ → Prop), 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ m - 1 → pair_scheduled_within_rounds_criteria schedule i) ∧
  (∀ four_players_set : finset ℕ, four_players_set.card = 4 → 
   players_not_played_at_all_or_at_least_twice schedule four_players_set) :=
sorry

/--
Helper definition: Check if pairs have been scheduled within the given number of rounds
-/
def pair_scheduled_within_rounds_criteria (schedule : list (ℕ → ℕ → Prop)) (i : ℕ) : Prop :=
sorry

/--
Helper definition: Check if for any set of 4 players, they haven't played each other at all or played at least twice
-/
def players_not_played_at_all_or_at_least_twice (schedule : list (ℕ → ℕ → Prop)) (four_players_set : finset ℕ) : Prop :=
sorry

end round_robin_schedule_l41_41320


namespace line_slope_intercept_product_l41_41547

theorem line_slope_intercept_product :
  ∃ (m b : ℝ), (b = -1) ∧ ((1 - (m * -1 + b) = 0) ∧ (mb = m * b)) ∧ (mb = 2) :=
by sorry

end line_slope_intercept_product_l41_41547


namespace part1_l41_41724

def f (x : ℝ) := 2 * x^2 - 3

theorem part1 {x : ℝ} (h : x ∈ Icc (- (Real.pi / 2)) (2 * Real.pi / 3)) (a : ℝ) :
  (∀ x, f (Real.cos x) ≤ a * (Real.cos x) + 1) → a ∈ Set.Icc (-2) 7 :=
sorry

end part1_l41_41724


namespace max_pencils_l41_41510

theorem max_pencils (initial_pencils: ℕ) (cost_per_20_refund: ℕ) (cost_per_5_refund: ℕ) :
  initial_pencils = 30 → cost_per_20_refund = 5 → cost_per_5_refund = 0.5 →
  (cost_per_5_refund * 10 : ℕ) = 5 → (∃ (total_pencils: ℕ), 
  total_pencils = 36) :=
by
  intros h1 h2 h3 h4
  use 36
  sorry

end max_pencils_l41_41510


namespace sin_225_eq_neg_sqrt2_div_2_l41_41664

theorem sin_225_eq_neg_sqrt2_div_2 : Real.sin (225 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180) :=
by
  have h1 : 225 = 180 + 45 := rfl
  have h2 : Real.sin ((180 + 45) * Real.pi / 180) = -Real.cos (45 * Real.pi / 180) := by
    rw ← h1
    rw Real.sin_add_pi_div_two
    rfl
  exact h2

end sin_225_eq_neg_sqrt2_div_2_l41_41664


namespace length_PQ_l41_41237

noncomputable def distance_between_points (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem length_PQ : 
  let P := (-4, 1) : ℝ × ℝ
  let Q := (1, -11) : ℝ × ℝ
  distance_between_points P Q = 13 := 
by
  sorry

end length_PQ_l41_41237


namespace find_decimal_decrease_l41_41571

noncomputable def tax_diminished_percentage (T C : ℝ) (X : ℝ) : Prop :=
  let new_tax := T * (1 - X / 100)
  let new_consumption := C * 1.15
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  new_revenue = original_revenue * 0.943

theorem find_decimal_decrease (T C : ℝ) (X : ℝ) :
  tax_diminished_percentage T C X → X = 18 := sorry

end find_decimal_decrease_l41_41571


namespace ned_weekly_earnings_l41_41514

-- Definitions of conditions
def normal_mouse_cost : ℕ := 120
def left_handed_mouse_cost (n : ℕ) := n + n * 3 / 10
def mice_sold_per_day : ℕ := 25
def store_days_open_per_week : ℕ := 4

-- Proof statement
theorem ned_weekly_earnings : 
  let left_cost := left_handed_mouse_cost normal_mouse_cost in
  let daily_earnings := left_cost * mice_sold_per_day in
  let weekly_earnings := daily_earnings * store_days_open_per_week in
  weekly_earnings = 15600 := 
by
  sorry

end ned_weekly_earnings_l41_41514


namespace point_P_on_scale_l41_41586

theorem point_P_on_scale :
  ∀ (initial_point final_point : ℝ) (num_intervals : ℕ) (position : ℕ),
    initial_point = 12.44 →
    final_point = 12.62 →
    num_intervals = 18 →
    position = 6 →
    (initial_point + (final_point - initial_point) / num_intervals * position) = 12.50 :=
by
  intros initial_point final_point num_intervals position h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end point_P_on_scale_l41_41586


namespace find_number_is_six_l41_41621

noncomputable def find_number : ℕ :=
  let x := 6 in
  if ((x * 8 / 3) - ((2 + 3) * 2)) = 6 then x else 0

theorem find_number_is_six :
  ∃ (x : ℕ), ((x * 8 / 3) - ((2 + 3) * 2)) = 6 ∧ x = 6 :=
begin
  use 6,
  split,
  {
    norm_num,
  },
  {
    refl,
  }
end

end find_number_is_six_l41_41621


namespace unjoinable_pair_l41_41609

-- Define that a pair of points is unjoinable
def unjoinable (Z : set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  ∀ (P : list (ℝ × ℝ)), (P.head = A ∧ P.last = B ∧ ∀ i, i < P.length - 1 → P.nth i ∉ Z) → false

-- Main statement to prove
theorem unjoinable_pair (Z : set (ℝ × ℝ)) (h : ∃ (A B : ℝ × ℝ), unjoinable Z A B) :
  ∀ r : ℝ, r > 0 → ∃ (A B : ℝ × ℝ), dist A B = r ∧ unjoinable Z A B :=
by
  sorry

end unjoinable_pair_l41_41609


namespace y_intercept_line_b_l41_41854

-- Define the conditions given in the problem
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c₁ c₂ : ℝ, (∀ x, b x = m * x + c₁) ∧ (∀ x, y = 3 * x - 2 = m * x + c₂)

def passes_through (b : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ m c : ℝ, (∀ x, b x = m * x + c) ∧ b P.1 = P.2

-- Define the main problem statement
theorem y_intercept_line_b (b : ℝ → ℝ)
  (h1 : is_parallel b)
  (h2 : passes_through b (5, 7)) : ∃ c : ℝ, ∀ x, b x = 3 * x + c ∧ c = -8 :=
sorry

end y_intercept_line_b_l41_41854


namespace number_of_stickers_per_page_number_of_stickers_per_page_is_20_l41_41930

-- Definitions of the conditions
def num_pages_before_loss := 12
def num_pages_after_loss := 11
def total_stickers_after_loss := 220

-- The goal is to determine the number of stickers per page
theorem number_of_stickers_per_page :
  ∃ S : ℕ, 11 * S = total_stickers_after_loss :=
sorry

-- Simplifying and specifying the exact number
theorem number_of_stickers_per_page_is_20 :
  11 * 20 = total_stickers_after_loss :=
by {
  -- Multiplying 11 by 20
  show 220 = 220,
  sorry
}

end number_of_stickers_per_page_number_of_stickers_per_page_is_20_l41_41930


namespace triangle_ABC_AD_length_l41_41067

noncomputable def length_AD (A B C D : Point) (BD DC : Length) (AB AC : Length) (AD : Length) : Prop :=
  D ∈ LineSegment B C ∧
  BD = 18 ∧
  DC = 30 ∧
  bisects_angle A D B C ∧
  AB = 27 ∧
  AC = 45 ∧
  AD = 11.25

theorem triangle_ABC_AD_length {A B C D : Point} (BD DC : Length) (AB AC AD : Length) :
  length_AD A B C D BD DC AB AC AD :=
by {
  sorry -- Proof is not required as per the problem statement
}

end triangle_ABC_AD_length_l41_41067


namespace irrational_number_is_sqrt3_l41_41639

theorem irrational_number_is_sqrt3 :
  ¬∃ (a b : ℤ), b ≠ 0 ∧ (√3 = a / b) ∧ 
  (∃ (c d : ℤ), d ≠ 0 ∧ (-2 = c / d)) ∧
  (∃ (e f : ℤ), f ≠ 0 ∧ (1 / 2 = e / f)) ∧
  (∃ (g h : ℤ), h ≠ 0 ∧ (2 = g / h)) :=
by {
  sorry
}

end irrational_number_is_sqrt3_l41_41639


namespace sarah_waist_size_in_cm_l41_41881

theorem sarah_waist_size_in_cm (inches_to_cm : Real := 2.54) (waist_size_in_inches : Real := 27) :
  let waist_size_in_cm := waist_size_in_inches * inches_to_cm
  Float.round(waist_size_in_cm * 10) / 10 = 68.6 :=
by
  -- Conditions from the problem
  have conversion_factor : Real := inches_to_cm
  have waist_size : Real := waist_size_in_inches

  -- Calculations
  let waist_size_in_cm : Real := waist_size * conversion_factor
  
  -- Prove the final result
  sorry

end sarah_waist_size_in_cm_l41_41881


namespace total_oranges_and_weight_correct_l41_41931

-- Define the conditions provided in the problem
def first_bucket_oranges : ℝ := 22.5
def second_bucket_oranges (first_bucket: ℝ) : ℝ := 2 * first_bucket + 3
def third_bucket_oranges (second_bucket: ℝ) : ℝ := second_bucket - 11.5
def fourth_bucket_oranges (first_bucket: ℝ) (third_bucket: ℝ) : ℝ := 1.5 * (first_bucket + third_bucket)
def weight_of_first_bucket_orange : ℝ := 0.3
def weight_of_third_bucket_orange : ℝ := 0.4
def weight_of_fourth_bucket (first_bucket: ℝ) (third_bucket: ℝ) (fourth_bucket: ℝ) : ℝ :=
  weight_of_first_bucket_orange * first_bucket + 
  weight_of_third_bucket_orange * third_bucket + 
  fourth_bucket * 0.35

-- Prove statement
theorem total_oranges_and_weight_correct :
  let first := first_bucket_oranges;
      second := second_bucket_oranges(first);
      third := third_bucket_oranges(second);
      fourth := fourth_bucket_oranges(first, third);
      total_oranges := first + second + third + fourth;
      W := weight_of_fourth_bucket(first, third, fourth)
  in total_oranges = 195.5 ∧ W = 52.325 :=
  by
  sorry

end total_oranges_and_weight_correct_l41_41931


namespace concurrency_and_perpendicularity_l41_41102

variables {A B C D O F A_0 B_0 C_0 D_0 A_1 B_1 C_1 D_1 M N : Type}
variables [EuclideanGeometry A B C D O F A_0 B_0 C_0 D_0 A_1 B_1 C_1 D_1 M N]

open EuclideanGeometry

noncomputable def tetrahedron_circumcenter (A B C D O : Type) : Prop :=
circumcenter_of_tetrahedron O A B C D

noncomputable def circumsphere_diameters (A B C D A_1 B_1 C_1 D_1 : Type) : Prop :=
diameters_of_circumsphere A B C D A_1 B_1 C_1 D_1

noncomputable def centroids (A B C D A_0 B_0 C_0 D_0 : Type) : Prop :=
centroids_of_triangles A B C D A_0 B_0 C_0 D_0

theorem concurrency_and_perpendicularity 
  (h1 : tetrahedron_circumcenter A B C D O) 
  (h2 : circumsphere_diameters A B C D A_1 B_1 C_1 D_1) 
  (h3 : centroids A B C D A_0 B_0 C_0 D_0) :
  ∃ F : Type, concurrent_at A_0 A_1 B_0 B_1 C_0 C_1 D_0 D_1 F ∧ ∀ mid : Type, 
  (midpoint A B = mid ∨ midpoint C D = mid ∨ midpoint A C = mid ∨ midpoint B D = mid ∨ midpoint A D = mid ∨ midpoint B C = mid) → perpendicular (line_through F mid) (opposite_side mid A B C D) := 
sorry

end concurrency_and_perpendicularity_l41_41102


namespace total_movies_correct_l41_41239

def num_movies_Screen1 : Nat := 3
def num_movies_Screen2 : Nat := 4
def num_movies_Screen3 : Nat := 2
def num_movies_Screen4 : Nat := 3
def num_movies_Screen5 : Nat := 5
def num_movies_Screen6 : Nat := 2

def total_movies : Nat :=
  num_movies_Screen1 + num_movies_Screen2 + num_movies_Screen3 + num_movies_Screen4 + num_movies_Screen5 + num_movies_Screen6

theorem total_movies_correct :
  total_movies = 19 :=
by 
  sorry

end total_movies_correct_l41_41239


namespace ratio_Lisa_Claire_l41_41868

-- Definitions
def Claire_photos : ℕ := 6
def Robert_photos : ℕ := Claire_photos + 12
def Lisa_photos : ℕ := Robert_photos

-- Theorem statement
theorem ratio_Lisa_Claire : (Lisa_photos : ℚ) / (Claire_photos : ℚ) = 3 / 1 :=
by
  sorry

end ratio_Lisa_Claire_l41_41868


namespace ordered_quadruples_even_sum_100_l41_41498

theorem ordered_quadruples_even_sum_100 :
  let n := { (x1, x2, x3, x4) | x1 + x2 + x3 + x4 = 100 ∧ ∀ i ∈ {x1, x2, x3, x4}, i % 2 = 0 }.to_finset.card
  in n / 100 = 184.24 := by
sorry

end ordered_quadruples_even_sum_100_l41_41498


namespace total_games_is_272_l41_41613

-- Define the number of players
def n : ℕ := 17

-- Define the formula for the number of games played
def total_games (n : ℕ) : ℕ := n * (n - 1)

-- Define a theorem stating that the total games played is 272
theorem total_games_is_272 : total_games n = 272 := by
  -- Proof omitted
  sorry

end total_games_is_272_l41_41613


namespace cubic_poly_root_l41_41281

noncomputable def cubic_root_form : ℝ :=
  (Real.cbrt 81 + Real.cbrt 9 - 3) / 27

theorem cubic_poly_root :
  ∃ p q r : ℕ, p = 81 ∧ q = 9 ∧ r = 27 ∧ (27 * cubic_root_form^3 + 27 * cubic_root_form^2 - 9 * cubic_root_form - 3 = 0) ∧ (p + q + r = 117) :=
by
  use 81, 9, 27
  sorry

end cubic_poly_root_l41_41281


namespace sum_of_angles_in_regular_hexagon_l41_41236

-- Define the regular hexagon with vertices A, B, C, D, E, F
structure RegularHexagon (A B C D E F K N : Type) :=
  (angle_sum : ∀ {angle_sum : Type}, angle_sum = 120 * 2)

-- Define the angles and their sum in the context of the hexagon and triangle described
theorem sum_of_angles_in_regular_hexagon (A B C D E F K N : Type)
  [RegularHexagon A B C D E F K N]
  (h1: ∀ (AK AN AB : Type), AK + AN = AB)
  (h2: ∀ {angle_sum: Type}, angle_sum = 240) :
  h1 ∧ h2 → ( ∠ K A N + ∠ K B N + ∠ K C N + ∠ K D N + ∠ K E N + ∠ K F N = 240) := sorry

end sum_of_angles_in_regular_hexagon_l41_41236


namespace expression_result_l41_41130

-- We define the mixed number fractions as conditions
def mixed_num_1 := 2 + 1 / 2         -- 2 1/2
def mixed_num_2 := 3 + 1 / 3         -- 3 1/3
def mixed_num_3 := 4 + 1 / 4         -- 4 1/4
def mixed_num_4 := 1 + 1 / 6         -- 1 1/6

-- Here are their improper fractions
def improper_fraction_1 := 5 / 2     -- (2 + 1/2) converted to improper fraction
def improper_fraction_2 := 10 / 3    -- (3 + 1/3) converted to improper fraction
def improper_fraction_3 := 17 / 4    -- (4 + 1/4) converted to improper fraction
def improper_fraction_4 := 7 / 6     -- (1 + 1/6) converted to improper fraction

-- Define the problematic expression
def expression := (improper_fraction_1 - improper_fraction_2)^2 / (improper_fraction_3 + improper_fraction_4)

-- Statement of the simplified result
theorem expression_result : expression = 5 / 39 :=
by
  sorry

end expression_result_l41_41130


namespace profit_percent_of_car_sales_l41_41208

variable (carCost : ℝ) (repairCost : ℝ) (sellingPrice : ℝ)

def totalCost := carCost + repairCost
def profit := sellingPrice - totalCost
def profitPercent := (profit / totalCost) * 100

theorem profit_percent_of_car_sales (h1 : carCost = 45000) (h2 : repairCost = 12000) (h3 : sellingPrice = 80000) :
  profitPercent carCost repairCost sellingPrice = 40.35 := by
  sorry

end profit_percent_of_car_sales_l41_41208


namespace solution_set_l41_41544

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, x ∈ set.univ

axiom f_at_neg1 : f (-1) = 2

axiom f_prime_pos : ∀ x : ℝ, f' x > 2

theorem solution_set:
  {x : ℝ | f (log x / log 2) < 2 * log x / log 2 + 4} =
  {x : ℝ | 0 < x ∧ x < 1/2} :=
sorry

end solution_set_l41_41544


namespace happy_boys_count_l41_41860

def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := total_children - happy_children - sad_children

def total_boys := 19
def total_girls := 41
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

def sad_boys := sad_children - sad_girls

theorem happy_boys_count :
  total_boys - sad_boys - neither_happy_nor_sad_boys = 6 :=
by
  sorry

end happy_boys_count_l41_41860


namespace max_a_value_l41_41351

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
noncomputable def f' (a b c x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem max_a_value (a b c d : ℝ) (h_nonzero : a ≠ 0)
  (h_deriv : ∀ x, 0 ≤ x → x ≤ 1 → |f' a b c x| ≤ 1) :
  a ≤ 8 / 3 ∧ (∃ b c d, (∀ x, 0 ≤ x → x ≤ 1 → |f' (8 / 3) b c x| ≤ 1) ∧ 8 / 3 ≠ 0) :=
sorry

end max_a_value_l41_41351


namespace spherical_cap_surface_area_l41_41572

theorem spherical_cap_surface_area (V : ℝ) (h : ℝ) (A : ℝ) (r : ℝ) 
  (volume_eq : V = (4 / 3) * π * r^3) 
  (cap_height : h = 2) 
  (sphere_volume : V = 288 * π) 
  (cap_surface_area : A = 2 * π * r * h) : 
  A = 24 * π := 
sorry

end spherical_cap_surface_area_l41_41572


namespace move_vertex_makes_non_special_l41_41507

def is_special_heptagon (H : heptagon) : Prop :=
  ∃ P : Point, ∃ (d1 d2 d3 : Diagonal), (d1 ∈ diagonals H ∧ d2 ∈ diagonals H ∧ d3 ∈ diagonals H) ∧
  (intersects_at d1 d2 P) ∧ (intersects_at d2 d3 P) ∧ (intersects_at d1 d3 P)

theorem move_vertex_makes_non_special (H : heptagon) (h_special: is_special_heptagon H)
  (v : Vertex) (H' : heptagon) (h_move : H' = move_vertex H v) : ¬ is_special_heptagon H' := by
  sorry

end move_vertex_makes_non_special_l41_41507


namespace tangent_line_curve_l41_41732

theorem tangent_line_curve
    (α : ℝ)
    (h1 : ∀ x, deriv (λ x, x^4) x = 4 * x^3)
    (h2 : tan α = 4) :
    cos α ^ 2 - sin (2 * α) = -7/17 :=
by
  sorry

end tangent_line_curve_l41_41732


namespace waiter_income_fraction_l41_41952

theorem waiter_income_fraction (S T : ℝ) (hT : T = 5/4 * S) :
  T / (S + T) = 5 / 9 :=
by
  sorry

end waiter_income_fraction_l41_41952


namespace find_quadratic_expression_l41_41743

theorem find_quadratic_expression :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, (a * x ^ 2 + b * x + c = -2 * x ^ 2 - 4 * x + 8)) ∧
    (vertex (λ x, a * x ^ 2 + b * x + c) = (-1, 10)) ∧ 
    (sum_squares_roots_eq (λ x, a * x ^ 2 + b * x + c) = 12) :=
begin
  sorry
end

-- Definitions required: vertex, sum_squares_roots_eq
def vertex (f : ℝ → ℝ) : ℝ × ℝ := 
  let a := (2 * derivate f) / (2 * (diffable f)) / 2 in
  (a, f a)

def sum_squares_roots_eq (f : ℝ → ℝ) : ℝ :=
  match_roots f $ λ ⟨α, β⟩, α ^ 2 + β ^ 2

end find_quadratic_expression_l41_41743


namespace Q_time_to_finish_job_l41_41592

theorem Q_time_to_finish_job :
  ∃ T_Q : ℚ, (1 / 4 + 3 / T_Q) * 3 = 19 / 20 ∧ T_Q = 15 := by
  existsi (15 : ℚ)
  split
  {
    field_simp
    norm_num
  }
  {
    norm_num
  }

end Q_time_to_finish_job_l41_41592


namespace shaded_region_area_l41_41185

theorem shaded_region_area (side1 side2 : ℝ)
  (h1 : side1 = 8) (h2 : side2 = 6)
  (A B C D E F G P : Point)
  (sq1 : square A B C D side1)
  (sq2 : square B E F G side2) 
  (hDE : line_through D E)
  (hBG : line_through B G)
  (hP : intersection_of hDE hBG P) :
  area A P E G = 18 := 
sorry

end shaded_region_area_l41_41185


namespace angle_A_area_triangle_ABC_l41_41458

noncomputable def TriangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.cos C + c * Real.cos B = 2 * a * Real.cos A

theorem angle_A (a b c A B C : ℝ) (h : TriangleABC a b c A B C) : 
  A = Real.pi / 3 := sorry

theorem area_triangle_ABC (a b c A B C : ℝ) (h1 : TriangleABC a b c A B C) 
  (h2 : Real.inner ⟨a, b⟩ ⟨a, c⟩ = sqrt 3) : 
  Real.area ⟨a, b, c⟩ = 3 / 2 := sorry

end angle_A_area_triangle_ABC_l41_41458


namespace triangle_sine_inequality_l41_41817

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤
  1 + (1 / 2) * Real.cos ((A - B) / 4) ^ 2 :=
by
  sorry

end triangle_sine_inequality_l41_41817


namespace min_norm_vector_sum_l41_41759

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, -1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (y, 2)

theorem min_norm_vector_sum (x y : ℝ) (h : x * y = 2) : 
  (real.sqrt ((x + y)^2 + 1)) ≥ 3 := 
sorry

end min_norm_vector_sum_l41_41759


namespace proj_magnitude_l41_41491

variables {V : Type*} [inner_product_space ℝ V]
variables (v w : V)

theorem proj_magnitude
  (h1 : inner_product v w = 4)
  (h2 : ∥w∥ = 7) : 
  ∥(inner_product v w / (∥w∥ * ∥w∥)) • w∥ = 4 :=
by sorry

end proj_magnitude_l41_41491


namespace shaded_area_percentage_l41_41587

def area_square (side : ℕ) : ℕ := side * side

def shaded_percentage (total_area shaded_area : ℕ) : ℚ :=
  ((shaded_area : ℚ) / total_area) * 100 

theorem shaded_area_percentage (side : ℕ) (total_area : ℕ) (shaded_area : ℕ) 
  (h_side : side = 7) (h_total_area : total_area = area_square side) 
  (h_shaded_area : shaded_area = 4 + 16 + 13) : 
  shaded_percentage total_area shaded_area = 3300 / 49 :=
by
  -- The proof will go here
  sorry

end shaded_area_percentage_l41_41587


namespace num_unique_differences_l41_41396

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41396


namespace good_number_count_l41_41617

theorem good_number_count :
  let is_good_number (a b c d : ℕ) : Prop :=
    (10 ≤ (10 * a + b) ∧ (10 * a + b) < 20) ∧
    (10 ≤ (10 * b + c) ∧ (10 * b + c) < 21) ∧
    (10 ≤ (10 * c + d) ∧ (10 * c + d) < 22)
  in
  ∃ n : ℕ, (∀ (a b c d : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ 
                         (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) → 
                         is_good_number a b c d) ∧ n = 10 := sorry

end good_number_count_l41_41617


namespace evaluate_expression_l41_41288

noncomputable def i : ℂ := complex.I

theorem evaluate_expression :
  i^10 + i^20 + i^(-30) = -1 :=
by
  have h1 : i^4 = 1 := by
    sorry -- This is where the periodicity of i would be justified
  
  have h2 : i^2 = -1 := by
    sorry -- This is definition of the imaginary unit squared

  sorry -- The rest of the proof would be completing the steps using h1 and h2

end evaluate_expression_l41_41288


namespace determinant_matrix_is_one_l41_41289

variable (α β : ℝ)

-- Define the 3x3 matrix
def matrix_3x3 := ![
  ![Real.sin α * Real.sin β, Real.sin α * Real.cos β, Real.cos α],
  ![Real.cos β, -Real.sin β, 0],
  ![Real.cos α * Real.sin β, Real.cos α * Real.cos β, -Real.sin α]
]

-- State the proof problem
theorem determinant_matrix_is_one : Matrix.det (matrix_3x3 α β) = 1 := by
  sorry

end determinant_matrix_is_one_l41_41289


namespace num_diff_positive_integers_l41_41369

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41369


namespace books_in_pyramid_l41_41253

theorem books_in_pyramid :
  let L1 := 64 in
  let L2 := L1 / 0.8 in
  let L3 := L2 / 0.8 in
  let L4 := L3 / 0.8 in
  L1 + L2 + L3 + L4 = 369 :=
by
  sorry

end books_in_pyramid_l41_41253


namespace ratio_of_roots_l41_41804

theorem ratio_of_roots (k : ℚ) : 
  let a := k^2 - 5 * k + 3 
      b := 3 * k - 1 
      c := 2 
  in 2 * b^2 = 3 * a * c → k = 2 / 3 :=
sorry

end ratio_of_roots_l41_41804


namespace compare_numbers_l41_41264

theorem compare_numbers : 222^2 < 22^22 ∧ 22^22 < 2^222 :=
by {
  sorry
}

end compare_numbers_l41_41264


namespace number_of_diffs_l41_41362

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41362


namespace composite_number_from_2004_permutation_only_one_of_2_pow_n_minus_1_and_2_pow_n_plus_1_is_prime_l41_41213

-- Problem (1)
theorem composite_number_from_2004_permutation : 
  ∀ N : ℕ, N = digits_join ((list.range 2005)).perm ∧ all_distinct ((list.range 2005)).perm → composite N :=
by sorry

-- Problem (2)
theorem only_one_of_2_pow_n_minus_1_and_2_pow_n_plus_1_is_prime : 
  ∀ n : ℕ, n > 2 → (prime (2^n - 1) → ¬ prime (2^n + 1)) ∧ (prime (2^n + 1) → ¬ prime (2^n - 1)) :=
by sorry

end composite_number_from_2004_permutation_only_one_of_2_pow_n_minus_1_and_2_pow_n_plus_1_is_prime_l41_41213


namespace candidate_X_win_percentage_l41_41601

theorem candidate_X_win_percentage
    (R : ℕ) -- R as a common factor
    (total_voters : ℕ := 5 * R) -- Total number of registered voters
    (republicans : ℕ := 3 * R) -- Number of Republicans
    (democrats : ℕ := 2 * R) -- Number of Democrats
    (votes_for_X_from_republicans : ℕ := 8 * R / 10) -- 80% of Republicans voting for X
    (votes_for_X_from_democrats : ℕ := 25 * R / 10) -- 25% of Democrats voting for X
    (votes_for_Y_from_republicans : ℕ := 2 * R / 10) -- Remaining Republicans voting for Y
    (votes_for_Y_from_democrats : ℕ := 15 * R / 10) -- Remaining Democrats voting for Y) :
    ((votes_for_X_from_republicans + votes_for_X_from_democrats - votes_for_Y_from_republicans - votes_for_Y_from_democrats) / total_voters * 100 = 16) :=
by
    sorry

end candidate_X_win_percentage_l41_41601


namespace simplify_sum_l41_41678

theorem simplify_sum :
  -2^2004 + (-2)^2005 + 2^2006 - 2^2007 = -2^2004 - 2^2005 + 2^2006 - 2^2007 :=
by
  sorry

end simplify_sum_l41_41678


namespace solution_value_a_l41_41772

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end solution_value_a_l41_41772


namespace r_minus_p_value_l41_41600

variables p q r : ℝ

-- Define the conditions given in the problem statement
def condition1 : Prop := (p + q) / 2 = 10
def condition2 : Prop := (q + r) / 2 = 24

-- Define the proof problem
theorem r_minus_p_value : condition1 ∧ condition2 → r - p = 28 :=
by
  sorry

end r_minus_p_value_l41_41600


namespace total_trees_planted_water_conservation_capacity_year3_l41_41048

-- Definition for the annual tree planting
def annual_tree_planting : Nat := 500 * 10^6

-- Q1: Proving the number of trees planted from 2009 to 2015
theorem total_trees_planted :
  let years := 2015 - 2009 + 1 in
  (annual_tree_planting * years) = 3.5 * 10^9 := 
by
  let years := 2015 - 2009 + 1
  calc
    annual_tree_planting * years = 500 * 10^6 * 7 := by sorry
                             ... = 3.5 * 10^9 := by sorry

-- Q2: Proving the water conservation capacity in the third year
theorem water_conservation_capacity_year3 :
  (let k := 4 / 30;
       b := 5 / 30;
       y := k * 3 + b
   in y) = 17 / 30 :=
by
  let k := 4 / 30
  let b := 5 / 30
  let y := k * 3 + b
  show y = 17 / 30 from sorry

end total_trees_planted_water_conservation_capacity_year3_l41_41048


namespace polynomial_value_at_neg_two_l41_41187

noncomputable def f (x : Float) : Float := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

theorem polynomial_value_at_neg_two : f (-2) = 325.4 := 
by
  have h : f (-2) = ((((((-2 : Float) - 5)*(-2) + 6)*(-2) + 0)*(-2) + 1)*(-2) + 0.3)*(-2) + 2 := by sorry
  exact h

end polynomial_value_at_neg_two_l41_41187


namespace ways_to_place_numbers_in_strip_l41_41632

theorem ways_to_place_numbers_in_strip :
  let strip := (1 : ℕ) × 10
  ∃ f : {n // n ∈ finset.range 10} → ℕ,
    (f 0 = 1) ∧
    ∃ cnt : ℕ, 
    cnt = 2^9 ∧
    ∀ n ∈ finset.range 10, (f n) ∈ {0, 1} → 
    512 :=
by
  sorry

end ways_to_place_numbers_in_strip_l41_41632


namespace cubic_conversion_l41_41021

theorem cubic_conversion (h : 1 = 100) : 1 = 1000000 :=
by
  sorry

end cubic_conversion_l41_41021


namespace solve_inequality_l41_41531

theorem solve_inequality (x : ℝ) : 
  x ∈ set.Iio (1 : ℝ) ↔ (x - 1) / ((x - 3)^2) < 0 :=
begin
  sorry
end

end solve_inequality_l41_41531


namespace arithmetic_geometric_sequence_l41_41651

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_d : d ≠ 0)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d)
  (h_geo : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)) :
  (S 4 - S 2) / (S 5 - S 3) = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l41_41651


namespace ratio_of_areas_l41_41055

variable (ABCDEF : Type) [hexagon : RegularHexagon ABCDEF] 
variable (P : Point) (Q : Point) (R : Point) (S : Point)
variable {AB CD DE FA : Side} (h: TriangleHeight AB CD)

-- Assumptions based on the problem
def conditions : Prop :=
  ∃ (P : Point), ∃ (Q : Point), ∃ (R : Point), ∃ (S : Point),
  OnSide P AB ∧ OnSide Q CD ∧ OnSide R DE ∧ OnSide S FA ∧
  Parallel (LineThrough P C) (LineThrough R A) ∧ 
  Parallel (LineThrough Q S) (LineThrough E B) ∧
  Distance (LineThrough P C) (LineThrough R A) = h / 2 ∧ 
  Distance (LineThrough Q S) (LineThrough E B) = h / 2

-- Finally, the theorem to be proved
theorem ratio_of_areas {APQRSC ABCDEF} (h : isRegularHexagon ABCDEF):
  conditions P Q R S → 
  area_ratio APQRSC ABCDEF = 3 / 4 := by
  sorry

end ratio_of_areas_l41_41055


namespace molecular_weight_of_3_moles_of_Fe2_SO4_3_l41_41270

noncomputable def mol_weight_fe : ℝ := 55.845
noncomputable def mol_weight_s : ℝ := 32.065
noncomputable def mol_weight_o : ℝ := 15.999

noncomputable def mol_weight_fe2_so4_3 : ℝ :=
  (2 * mol_weight_fe) + (3 * (mol_weight_s + (4 * mol_weight_o)))

theorem molecular_weight_of_3_moles_of_Fe2_SO4_3 :
  3 * mol_weight_fe2_so4_3 = 1199.619 := by
  sorry

end molecular_weight_of_3_moles_of_Fe2_SO4_3_l41_41270


namespace transform_and_symmetrize_log_to_exp_l41_41947

theorem transform_and_symmetrize_log_to_exp :
  ∀ (x : ℝ), (y = log 2 (x - 1)) → (y_shift_left = log 2 x) →
  (y_sym = exp 2 x) →
  (∃ shift_amount : ℝ, shift_amount = 1 ∧ y_shift_left = y_shift_left - shift_amount) →
  (∃ symmetry : ℝ → ℝ, symmetry x = y ∧ y_sym = symmetry y_shift_left) :=
by
  sorry

end transform_and_symmetrize_log_to_exp_l41_41947


namespace find_alpha_l41_41714

theorem find_alpha (α : Real) (hα : 0 < α ∧ α < π) :
  (∃ x : Real, (|2 * x - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * x| = Real.sin α) ∧ 
  ∀ y : Real, (|2 * y - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * y| = Real.sin α) → y = x) →
  α = π / 12 ∨ α = 11 * π / 12 :=
by
  sorry

end find_alpha_l41_41714


namespace number_of_distinct_positive_differences_l41_41417

theorem number_of_distinct_positive_differences :
  (finset.card ((finset.filter (λ x, x > 0) 
    ((finset.map (function.uncurry has_sub.sub) 
    ((finset.product (finset.range 21) (finset.range 21)).filter (λ ⟨a, b⟩, a ≠ b)))))) = 19 :=
sorry

end number_of_distinct_positive_differences_l41_41417


namespace min_value_proof_l41_41839

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c)

theorem min_value_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ≥ 343 :=
sorry

end min_value_proof_l41_41839


namespace ram_first_year_balance_l41_41186

-- Given conditions
def initial_deposit : ℝ := 1000
def interest_first_year : ℝ := 100

-- Calculate end of the first year balance
def balance_first_year := initial_deposit + interest_first_year

-- Prove that balance_first_year is $1100
theorem ram_first_year_balance :
  balance_first_year = 1100 :=
by 
  sorry

end ram_first_year_balance_l41_41186


namespace find_length_of_c_find_measure_of_B_l41_41047

-- Definition of the conditions
def triangle (A B C a b c : ℝ) : Prop :=
  c - b = 2 * b * Real.cos A

noncomputable def value_c (a b : ℝ) : ℝ := sorry

noncomputable def value_B (A B : ℝ) : ℝ := sorry

-- Statement for problem (I)
theorem find_length_of_c (a b : ℝ) (h1 : a = 2 * Real.sqrt 6) (h2 : b = 3) (h3 : ∀ A B C, triangle A B C a b (value_c a b)) : 
  value_c a b = 5 :=
by 
  sorry

-- Statement for problem (II)
theorem find_measure_of_B (B : ℝ) (h1 : ∀ A, A + B = Real.pi / 2) (h2 : B = value_B A B) : 
  value_B A B = Real.pi / 6 :=
by 
  sorry

end find_length_of_c_find_measure_of_B_l41_41047


namespace platform_length_l41_41982

theorem platform_length (train_length : ℕ) (time_post : ℕ) (time_platform : ℕ) (speed : ℕ)
    (h1 : train_length = 150)
    (h2 : time_post = 15)
    (h3 : time_platform = 25)
    (h4 : speed = train_length / time_post)
    : (train_length + 100) / time_platform = speed :=
by
  sorry

end platform_length_l41_41982


namespace number_of_differences_l41_41390

theorem number_of_differences {A : Finset ℕ} (hA : A = (Finset.range 21).filter (λ x, x > 0)) :
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (A.product A)).filter (λ x, x > 0).card = 19 :=
by
  sorry

end number_of_differences_l41_41390


namespace total_non_defective_engines_is_902_l41_41171

def engines_in_batches := [140, 150, 170, 180, 190, 210, 220]
def defect_rates := [0.12, 0.18, 0.22, 0.28, 0.32, 0.36, 0.41]

def non_defective_engines : List ℝ :=
  List.zipWith (λ e r => e * (1 - r)) engines_in_batches defect_rates

def total_non_defective_engines : ℝ :=
  (non_defective_engines.map_floor).foldl (λ a b => a + b) 0

theorem total_non_defective_engines_is_902 : total_non_defective_engines = 902 :=
by 
  -- Placeholder for proof
  sorry

end total_non_defective_engines_is_902_l41_41171


namespace find_sum_l41_41946

variables (a b c d : ℕ)

axiom h1 : 6 * a + 2 * b = 3848
axiom h2 : 6 * c + 3 * d = 4410
axiom h3 : a + 3 * b + 2 * d = 3080

theorem find_sum : a + b + c + d = 1986 :=
by
  sorry

end find_sum_l41_41946


namespace cone_lateral_surface_area_l41_41037

-- Definitions based on conditions
def base_radius : ℝ := 2
def slant_height : ℝ := 3

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ := π * r * l

-- Theorem stating the problem
theorem cone_lateral_surface_area : lateral_surface_area base_radius slant_height = 6 * π :=
by
  sorry

end cone_lateral_surface_area_l41_41037


namespace sequence_problem_l41_41723

variable {n : ℕ}

-- We define the arithmetic sequence conditions
noncomputable def a_n : ℕ → ℕ
| n => 2 * n + 1

-- Conditions that the sequence must satisfy
axiom a_3_eq_7 : a_n 3 = 7
axiom a_5_a_7_eq_26 : a_n 5 + a_n 7 = 26

-- Define the sum of the sequence
noncomputable def S_n (n : ℕ) := n^2 + 2 * n

-- Define the sequence b_n
noncomputable def b_n (n : ℕ) := 1 / (a_n n ^ 2 - 1 : ℝ)

-- Define the sum of the sequence b_n
noncomputable def T_n (n : ℕ) := (n / (4 * (n + 1)) : ℝ)

-- The main theorem to prove
theorem sequence_problem :
  (a_n n = 2 * n + 1) ∧ (S_n n = n^2 + 2 * n) ∧ (T_n n = n / (4 * (n + 1))) :=
  sorry

end sequence_problem_l41_41723


namespace six_digit_numbers_with_zero_l41_41014

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41014


namespace least_number_of_stamps_l41_41653

theorem least_number_of_stamps : ∃ c f : ℕ, 3 * c + 4 * f = 50 ∧ c + f = 13 :=
by
  sorry

end least_number_of_stamps_l41_41653


namespace souvenir_purchasing_combinations_l41_41980

theorem souvenir_purchasing_combinations :
  { x : ℕ × ℕ × ℕ // x.1 + 2 * x.2 + 4 * x.3 = 101 ∧ x.1 ≥ 1 ∧ x.2 ≥ 1 ∧ x.3 ≥ 1 }.card = 600 := 
sorry

end souvenir_purchasing_combinations_l41_41980


namespace tan_3theta_eq_2_11_sin_3theta_eq_22_125_l41_41437

variable {θ : ℝ}

-- First, stating the condition \(\tan \theta = 2\)
axiom tan_theta_eq_2 : Real.tan θ = 2

-- Stating the proof problem for \(\tan 3\theta = \frac{2}{11}\)
theorem tan_3theta_eq_2_11 : Real.tan (3 * θ) = 2 / 11 :=
by 
  sorry

-- Stating the proof problem for \(\sin 3\theta = \frac{22}{125}\)
theorem sin_3theta_eq_22_125 : Real.sin (3 * θ) = 22 / 125 :=
by 
  sorry

end tan_3theta_eq_2_11_sin_3theta_eq_22_125_l41_41437


namespace chord_length_of_intersecting_line_with_circle_l41_41780

theorem chord_length_of_intersecting_line_with_circle :
  (∀ A B : ℝ × ℝ, (A.2 = sqrt 3 * A.1) ∧ (B.2 = sqrt 3 * B.1) ∧
  (A.1 ^ 2 - 4 * A.1 + A.2 ^ 2 = 0) ∧ (B.1 ^ 2 - 4 * B.1 + B.2 ^ 2 = 0) →
  dist A B = 2)
  sorry

#check chord_length_of_intersecting_line_with_circle

end chord_length_of_intersecting_line_with_circle_l41_41780


namespace solve_inequality_l41_41919

-- Declare the necessary conditions as variables in Lean
variables (a c : ℝ)

-- State the Lean theorem
theorem solve_inequality :
  (∀ x : ℝ, (ax^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) →
  a < 0 →
  a = -6 ∧ c = -1 :=
  sorry

end solve_inequality_l41_41919


namespace crop_fraction_to_longest_side_l41_41901

def trapezoid (a b c d : ℝ) (α β : ℝ) : Prop :=
  a = 150 ∧ b = 150 ∧ c = 200 ∧ d = 200 ∧ α = 60 ∧ β = 120

-- Define the total area calculation
noncomputable def area_of_trapezoid (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- Define the fraction brought to the longest side
def fraction_to_longest_side (area_total : ℝ) (area_closest : ℝ) : ℝ :=
  area_closest / area_total

-- Formal problem statement
theorem crop_fraction_to_longest_side (a b c d height : ℝ) (α β : ℝ)
  (h_trapezoid : trapezoid a b c d α β)
  (height_eq : height = 150 * real.sqrt 3 / 2)
  (area_total : ℝ := area_of_trapezoid 150 200 height)
  (area_closest := (5 / 12) * area_total) :
  fraction_to_longest_side area_total area_closest = 5 / 12 :=
by
  sorry

end crop_fraction_to_longest_side_l41_41901


namespace compromise_function_condition_l41_41445

theorem compromise_function_condition (k : ℝ) :
  (∀ x ∈ set.Icc (1 : ℝ) (2 * Real.exp 1), 
    0 ≤ (k-1) * x - 1 ∧ (k-1) * x - 1 ≤ (x+1) * Real.log x) ↔ k = 2 := 
by
  sorry

end compromise_function_condition_l41_41445


namespace average_weight_b_c_l41_41538

variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := (A + B + C = 135)
def condition2 : Prop := (A + B = 82)
def condition3 : Prop := (B = 33)

-- The proposition to prove
theorem average_weight_b_c (h1 : condition1) (h2 : condition2) (h3 : condition3) : (B + C) / 2 = 43 :=
sorry

end average_weight_b_c_l41_41538


namespace tenth_term_of_arithmetic_sequence_l41_41164

open BigOperators

noncomputable def sum_of_n_terms_arithmetic_sequence (n : ℕ) : ℕ :=
  n^2 + 5 * n

theorem tenth_term_of_arithmetic_sequence (S : ℕ → ℕ)
  (h : ∀ n, S n = sum_of_n_terms_arithmetic_sequence n) :
  S 10 - S 9 = 24 :=
by
  rw [h 10, h 9]
  calc
    (10^2 + 5 * 10) - (9^2 + 5 * 9) = 150 - 126 := by norm_num
    ... = 24 : by norm_num

end tenth_term_of_arithmetic_sequence_l41_41164


namespace pens_bought_l41_41243

theorem pens_bought
  (P : ℝ)
  (cost := 36 * P)
  (discount := 0.99 * P)
  (profit_percent := 0.1)
  (profit := (40 * discount) - cost)
  (profit_eq : profit = profit_percent * cost) :
  40 = 40 := 
by
  sorry

end pens_bought_l41_41243


namespace PA_CD_minimized_l41_41934

section
variables {a c d : ℝ} (h : a > c)
variables (PA : ℝ) (CD : ℝ)
variables (s_min : ℝ)

-- Conditions
def trapezoid_conditions :=
  PA = Real.sqrt (c * d) - (a - c)

-- Goal
def proof_problem :=
  s_min = PA + CD

-- Assertion of minimum value
theorem PA_CD_minimized (PA_def : trapezoid_conditions) (hCD : CD = c) : 
  proof_problem s_min (PA_def) hCD := by
  -- We need to show that PA + CD is minimized to 2 * sqrt(cd) - (a - c)
  sorry
end

end PA_CD_minimized_l41_41934


namespace average_of_first_5_results_is_15_l41_41167

theorem average_of_first_5_results_is_15 
  (a : Fin 11 → ℕ)
  (h_avg_total : (∑ i, a i) / 11 = 20)
  (h_avg_last_5 : (∑ i in Finset.range 10 \ Finset.range 5, a i) / 5 = 22)
  (h_6th_result : a 5 = 35) :
  (∑ i in Finset.range 5, a i) / 5 = 15 :=
sorry

end average_of_first_5_results_is_15_l41_41167


namespace part1_max_min_val_part2_a_range_l41_41749

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - 3 * a^2 * x

theorem part1_max_min_val (a : ℝ) (h_a: a = 1) :
  let f := (λ x, (1 / 3) * x^3 + x^2 - 3 * x) in
  let f' := λ x, x^2 + 2 * x - 3 in
  let interval := set.Icc (0:ℝ) (2:ℝ) in
  ∃ x_max x_min : ℝ, x_max ∈ interval ∧ x_min ∈ interval ∧
    (∀ x ∈ interval, f x ≤ f x_max) ∧ (∀ x ∈ interval, f x_min ≤ f x) ∧
    f x_max = 2 / 3 ∧ f x_min = -5 / 3 :=
sorry

theorem part2_a_range (a : ℝ) :
  let f := λ x, (1 / 3) * x^3 + a * x^2 - 3 * a^2 * x in
  (∃ x_local : ℝ, 1 < x_local ∧ x_local < 2 ∧
    let f' := λ x, x^2 + 2 * a * x - 3 * a^2 in
    f' x_local = 0 ∧
    (∀ x ∈ set.Ioo 1 2, f' x ≠ 0)) ↔
  (∃ b c : ℝ, (b = a ∧ -2 / 3 < b ∧ b < -1 / 3) ∨ (c = a ∧ 1 < c ∧ c < 2)) :=
sorry

end part1_max_min_val_part2_a_range_l41_41749


namespace arithmetic_sequence_common_difference_l41_41057

theorem arithmetic_sequence_common_difference 
    (a : ℤ) (last_term : ℤ) (sum_terms : ℤ) (n : ℕ)
    (h1 : a = 3) 
    (h2 : last_term = 58) 
    (h3 : sum_terms = 488)
    (h4 : sum_terms = n * (a + last_term) / 2)
    (h5 : last_term = a + (n - 1) * d) :
    d = 11 / 3 := by
  sorry

end arithmetic_sequence_common_difference_l41_41057


namespace moose_population_l41_41462

theorem moose_population (B M H : ℕ) (h1 : B = 2 * M) (h2 : H = 19 * B) (h3 : H = 38_000_000) : M = 1_000_000 :=
by sorry

end moose_population_l41_41462


namespace sum_of_products_l41_41567

def M : set ℝ := {6666, -11135, 2333, 10, 99111, -1, -198, 1000, 0, real.pi}
def M_i : ℕ → set (set ℝ) := λ i, {S | S ⊆ M ∧ S ≠ ∅}

def m_i : (set ℝ) → ℝ
| ∅ := 1
| (x ∪ xs) := x * m_i xs

theorem sum_of_products :
  (∑ i in finset.range 1023, m_i (M_i i)) = -1 :=
sorry

end sum_of_products_l41_41567


namespace count_special_three_digit_numbers_l41_41429

-- Define the range for three-digit numbers
def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Define the predicate for at least one digit being 5 or 8
def contains_five_or_eight (n : ℕ) : Prop :=
  (∃ d ∈ to_digit_list n, d = 5 ∨ d = 8)

-- Collect the three-digit numbers that satisfy the condition
def filtered_numbers := {n : ℕ | n ∈ three_digit_numbers ∧ contains_five_or_eight n}

-- The theorem we want to prove
theorem count_special_three_digit_numbers : finset.card filtered_numbers = 452 :=
by
  sorry

end count_special_three_digit_numbers_l41_41429


namespace cos_105_minus_alpha_plus_sin_alpha_minus_105_l41_41329

noncomputable def cos_add_eq_half (alpha : ℝ) : Prop :=
  real.cos (real.pi / 180 * (75 + alpha)) = 1 / 2

noncomputable def third_quadrant (alpha : ℝ) : Prop :=
  real.pi < alpha ∧ alpha < 3 * real.pi / 2

theorem cos_105_minus_alpha_plus_sin_alpha_minus_105 (alpha : ℝ)
  (h1 : cos_add_eq_half alpha)
  (h2 : third_quadrant alpha) :
  real.cos (real.pi / 180 * (105 - alpha)) + real.sin (real.pi / 180 * (alpha - 105)) = 1 / 2 + real.sqrt 3 / 2 := 
by
  sorry

end cos_105_minus_alpha_plus_sin_alpha_minus_105_l41_41329


namespace population_growth_l41_41654

variable (M p : ℝ)
variable (hM : M > 0) -- assuming population is positive
variable (hp : p > -1) -- assuming growth rate is such that (1 + p) > 0

theorem population_growth : 
  (M * (1 + p) ^ 10) = by sorry

end population_growth_l41_41654


namespace problem1_problem2_l41_41275

noncomputable def compute_expression1 : ℝ :=
  real.sqrt (9 / 4) - (-9.6)^0 - (27 / 8)^(-2/3) + (3 / 2)^(-2)

noncomputable def compute_expression2 : ℝ :=
  real.log 3 (27 ^ (1/4)) + real.log 10 25 + real.log 10 4 + 7^(real.log 7 2)

theorem problem1 : compute_expression1 = 1 / 2 :=
  by 
  sorry

theorem problem2 : compute_expression2 = 15 / 4 :=
  by 
  sorry

end problem1_problem2_l41_41275


namespace digits_sum_is_15_l41_41599

theorem digits_sum_is_15 (f o g : ℕ) (h1 : f * 100 + o * 10 + g = 366) (h2 : 4 * (f * 100 + o * 10 + g) = 1464) (h3 : f < 10 ∧ o < 10 ∧ g < 10) :
  f + o + g = 15 :=
sorry

end digits_sum_is_15_l41_41599


namespace problem_statement_l41_41736

noncomputable def f : ℝ → ℝ := sorry

axiom even_function {x : ℝ} : f(x) = f(-x)

axiom periodic_function {x : ℝ} : f(x + 2) = -1 / f(x)

axiom definition_on_interval (x : ℝ) (h : -3 ≤ x ∧ x ≤ -2) : f(x) = x

theorem problem_statement : f 2018 = -2 :=
by
  sorry

end problem_statement_l41_41736


namespace w12_plus_inv_w12_l41_41731

open Complex

-- Given conditions
def w_plus_inv_w_eq_two_cos_45 (w : ℂ) : Prop :=
  w + (1 / w) = 2 * Real.cos (Real.pi / 4)

-- Statement of the theorem to prove
theorem w12_plus_inv_w12 {w : ℂ} (h : w_plus_inv_w_eq_two_cos_45 w) : 
  w^12 + (1 / (w^12)) = -2 :=
sorry

end w12_plus_inv_w12_l41_41731


namespace six_digit_numbers_with_zero_l41_41000

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41000


namespace Maggie_gave_away_4_packs_of_green_bouncy_balls_l41_41856

theorem Maggie_gave_away_4_packs_of_green_bouncy_balls :
  ∀ (G : ℝ),
    let yellow_packs := 8.0 in
    let green_packs := 4.0 in
    let bouncy_balls_per_pack := 10.0 in
    let total_bouncy_balls := (yellow_packs + green_packs) * bouncy_balls_per_pack in
    let kept_bouncy_balls := 80.0 in
    let given_away_bouncy_balls := total_bouncy_balls - kept_bouncy_balls in
    let green_bouncy_balls_given_away := given_away_bouncy_balls / bouncy_balls_per_pack in
    G = green_bouncy_balls_given_away → G = 4.0 :=
begin
  intros,
  sorry
end

end Maggie_gave_away_4_packs_of_green_bouncy_balls_l41_41856


namespace distinguishes_conditional_from_sequential_l41_41141

variable (C P S I D : Prop)

-- Conditions
def conditional_structure_includes_processing_box  : Prop := C = P
def conditional_structure_includes_start_end_box   : Prop := C = S
def conditional_structure_includes_io_box          : Prop := C = I
def conditional_structure_includes_decision_box    : Prop := C = D
def sequential_structure_excludes_decision_box     : Prop := ¬S = D

-- Proof problem statement
theorem distinguishes_conditional_from_sequential : C → S → I → D → P → 
    (conditional_structure_includes_processing_box C P) ∧ 
    (conditional_structure_includes_start_end_box C S) ∧ 
    (conditional_structure_includes_io_box C I) ∧ 
    (conditional_structure_includes_decision_box C D) ∧ 
    sequential_structure_excludes_decision_box S D → 
    (D = true) :=
by sorry

end distinguishes_conditional_from_sequential_l41_41141


namespace six_digit_numbers_with_zero_l41_41002

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41002


namespace sum_of_angles_l41_41811

noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

def angle_ABC := interior_angle 5
def angle_ABD := interior_angle 4

theorem sum_of_angles :
  angle_ABC + angle_ABD = 198 :=
by
  unfold angle_ABC angle_ABD interior_angle
  -- Adding the actual values of angles
  have h1 : 180 * (5 - 2) / 5 = 108 := by norm_num,
  have h2 : 180 * (4 - 2) / 4 = 90 := by norm_num,
  rw [h1, h2],
  norm_num,
  sorry

end sum_of_angles_l41_41811


namespace count_valid_sets_l41_41446

-- Define the set S
def S := {1, 2}

-- Define the statement that M satisfies the condition.
def valid_set (M : Set ℕ) : Prop :=
  ∅ ⊂ M ∧ M ⊆ S

-- State the theorem.
theorem count_valid_sets : finset.card (finset.filter valid_set (finset.powerset S)) = 3 :=
by
  sorry

end count_valid_sets_l41_41446


namespace reduced_price_l41_41594

theorem reduced_price (P : ℝ) (hP : P = 56)
    (original_qty : ℝ := 800 / P)
    (reduced_qty : ℝ := 800 / (0.65 * P))
    (diff_qty : ℝ := reduced_qty - original_qty)
    (difference_condition : diff_qty = 5) :
  0.65 * P = 36.4 :=
by
  rw [hP]
  sorry

end reduced_price_l41_41594


namespace incorrect_variance_l41_41739

noncomputable def normal_pdf (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (- (x - 1)^2 / 2)

theorem incorrect_variance :
  (∫ x, normal_pdf x * x^2) - (∫ x, normal_pdf x * x)^2 ≠ 2 := 
sorry

end incorrect_variance_l41_41739


namespace relationship_among_a_b_c_l41_41090

noncomputable def a := Real.sqrt 0.5
noncomputable def b := Real.sqrt 0.3
noncomputable def c := Real.log 0.2 / Real.log 0.3

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l41_41090


namespace find_a_l41_41116

theorem find_a (a : ℤ) (h : |a + 1| = 3) : a = 2 ∨ a = -4 :=
sorry

end find_a_l41_41116


namespace sum_of_all_possible_values_of_f_l41_41308

-- Define f as counting the valid (x, y) pairs
def f (a b c d : ℤ) : ℕ :=
  let candidates := finset.univ.product finset.univ in
  (candidates.filter (λ (xy : fin (5) × fin (5)),
    let x := xy.1.1 + 1 in
    let y := xy.2.1 + 1 in
    (a * x + b * y) % 5 = 0 ∧ (c * x + d * y) % 5 = 0)).card

-- Prove that the sum of all possible values of f(a, b, c, d) is 31
theorem sum_of_all_possible_values_of_f : 
  ∀ (a b c d : ℤ), f a b c d = 31 :=
by
  sorry

end sum_of_all_possible_values_of_f_l41_41308


namespace elsa_emma_spending_ratio_l41_41684

theorem elsa_emma_spending_ratio
  (E : ℝ)
  (h_emma : ∃ (x : ℝ), x = 58)
  (h_elizabeth : ∃ (y : ℝ), y = 4 * E)
  (h_total : 58 + E + 4 * E = 638) :
  E / 58 = 2 :=
by
  sorry

end elsa_emma_spending_ratio_l41_41684


namespace sum_of_corners_is_164_l41_41549

theorem sum_of_corners_is_164 :
  let checkerboard : List (List Nat) := List.chunk 9 (List.range' 1 81)
  in checkerboard.head!.head! + checkerboard.head!.getLast! + checkerboard.getLast!.head! + checkerboard.getLast!.getLast! = 164 :=
by
  sorry

end sum_of_corners_is_164_l41_41549


namespace sum_of_integers_with_product_80_l41_41179

theorem sum_of_integers_with_product_80 : 
  ∃ (a b c : ℤ), a ∈ {1, 2, 4, 8, 16, 20} ∧ b ∈ {1, 2, 4, 8, 16, 20} ∧ c ∈ {1, 2, 4, 8, 16, 20} ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 80 ∧ a + b + c = 25 :=
by
  sorry

end sum_of_integers_with_product_80_l41_41179


namespace even_factors_of_n_odd_factors_of_n_l41_41674

def n : ℕ := 2^2 * 3^2 * 7^2

theorem even_factors_of_n :
  ∃ evens : ℕ, evens = 18 ∧ 
  (∃ f : ℕ → ℕ → ℕ → Prop,
    (∀ a b c : ℕ, f a b c → (1 ≤ a) ∧ (a ≤ 2) ∧ (0 ≤ b) ∧ (b ≤ 2) ∧ (0 ≤ c) ∧ (c ≤ 2) ∧ 
    (∃ (even_factors : list ℕ), even_factors.length = evens ∧ ∀ x ∈ even_factors, x = 2 ^ a * 3 ^ b * 7 ^ c))) :=
by
  sorry

theorem odd_factors_of_n :
  ∃ odds : ℕ, odds = 9 ∧ 
  (∃ f : ℕ → ℕ → ℕ → Prop,
    (∀ a b c : ℕ, f a b c → (a = 0) ∧ (0 ≤ b) ∧ (b ≤ 2) ∧ (0 ≤ c) ∧ (c ≤ 2) ∧ 
    (∃ (odd_factors : list ℕ), odd_factors.length = odds ∧ ∀ x ∈ odd_factors, x = 2 ^ a * 3 ^ b * 7 ^ c))) :=
by
  sorry

end even_factors_of_n_odd_factors_of_n_l41_41674


namespace total_metal_rods_needed_l41_41228

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end total_metal_rods_needed_l41_41228


namespace minimal_difference_factorial_fractions_l41_41910

theorem minimal_difference_factorial_fractions (p q : ℕ) (c d : Fin p → ℕ) (h_sorted_c : ∀ i j, i < j → c i ≥ c j) (h_sorted_d : ∀ i j, i < j → d i ≥ d j) (hc_pos : ∀ i, c i > 0) (hd_pos : ∀ i, d i > 0) (h_eq : 2090 = (c 0)! * list.prod (Finset.univ.image (λ i, (if i > 0 then (c i)! else 1)).val) / (d 0)! * list.prod (Finset.univ.image (λ i, (if i > 0 then (d i)! else 1)).val)) : 
  ∃ c_1 d_1, min_diff c d c_1 d_1 = 2 := 
begin
  -- proof goes here
  sorry
end

def min_diff (c d : Fin p → ℕ) (c_1 d_1 : ℕ) : ℕ :=
  let minimal_val := min (c 0 + d 0) (c_1 + d_1) in
  abs ((c_1 - d_1 : ℤ).nat_abs)

end minimal_difference_factorial_fractions_l41_41910


namespace car_dealership_sales_l41_41788

theorem car_dealership_sales (trucks_ratio suvs_ratio trucks_expected suvs_expected : ℕ)
  (h_ratio : trucks_ratio = 5 ∧ suvs_ratio = 8)
  (h_expected : trucks_expected = 35 ∧ suvs_expected = 56) :
  (trucks_ratio : ℚ) / suvs_ratio = (trucks_expected : ℚ) / suvs_expected :=
by
  sorry

end car_dealership_sales_l41_41788


namespace quadratic_inequality_solution_l41_41716

theorem quadratic_inequality_solution {a x : ℝ} (h : a ∈ Set.Icc (-1 : ℝ) 1) :
  x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 :=
begin
  sorry
end

end quadratic_inequality_solution_l41_41716


namespace a_2009_eq_one_a_2014_eq_zero_l41_41565

-- Define the sequence {a_n} as per the given conditions
def a : ℕ → ℕ
| n := if h1 : ∃ k, n = 4 * k - 3 then 1
       else if h2 : ∃ k, n = 4 * k - 1 then 0
       else if h3 : ∃ k, n = 2 * k        then a k
       else 0

-- Prove that a_{2009} = 1 under the given conditions
theorem a_2009_eq_one : a 2009 = 1 :=
by {
  sorry
}

-- Prove that a_{2014} = 0 under the given conditions
theorem a_2014_eq_zero : a 2014 = 0 :=
by {
  sorry
}

end a_2009_eq_one_a_2014_eq_zero_l41_41565


namespace diff_of_two_distinct_members_l41_41406

theorem diff_of_two_distinct_members : 
  let S : Finset ℕ := (Finset.range 21).erase 0 in
  (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S).filter (λ p, p.1 > p.2)).card = 19 := by
{
  let S : Finset ℕ := (Finset.range 21).erase 0,
  let diffs : Finset ℕ := 
    (Finset.image (λ p : ℕ × ℕ, p.1 - p.2) (S.product S)).filter (λ p : ℕ × ℕ, p.1 > p.2),
  have h1 : diffs = Finset.range 20 \ {0},
  { sorry },
  rw h1,
  exact (Finset.card_range 20).symm,
}

end diff_of_two_distinct_members_l41_41406


namespace calculate_distance_l41_41935

theorem calculate_distance (t : ℕ) (h_t : t = 4) : 5 * t^2 + 2 * t = 88 :=
by
  rw [h_t]
  norm_num

end calculate_distance_l41_41935


namespace equation_of_hyperbola_perpendicular_vectors_area_of_triangle_l41_41337

-- Definitions based on conditions provided
def hyperbola_centered_at_origin (x y : ℝ) : Prop := x^2 - y^2 = 6
def foci_coordinates : ℝ × ℝ := (Real.sqrt 12, 0)

-- 1. Prove the equation of the hyperbola
theorem equation_of_hyperbola (x y : ℝ) :
  hyperbola_centered_at_origin 4 (-Real.sqrt 10) → hyperbola_centered_at_origin x y :=
by
  sorry

-- 2. Prove MF₁ ⊥ MF₂ for M(3, m) on the hyperbola
theorem perpendicular_vectors (m : ℝ) :
  hyperbola_centered_at_origin 3 m → 
  let F1 := (Real.sqrt 12, 0)
  let F2 := (-Real.sqrt 12, 0)
  ((3 - Real.sqrt 12) * (3 + Real.sqrt 12) - m^2 = 0) :=
by
  sorry

-- 3. Find the area of triangle F₁MF₂ for M(3, m) on the hyperbola
theorem area_of_triangle (m : ℝ) :
  hyperbola_centered_at_origin 3 m → 
  let F1 := (Real.sqrt 12, 0)
  let F2 := (-Real.sqrt 12, 0)
  Real.abs (m) = Real.sqrt 3 → 
  (Real.sqrt 12 * Real.abs m = 6) :=
by
  sorry

end equation_of_hyperbola_perpendicular_vectors_area_of_triangle_l41_41337


namespace power_function_is_odd_l41_41341

open Function

noncomputable def power_function (a : ℝ) (b : ℝ) : ℝ → ℝ := λ x => (a - 1) * x^b

theorem power_function_is_odd (a b : ℝ) (h : power_function a b a = 1 / 8)
  :  a = 2 ∧ b = -3 → (∀ x : ℝ, power_function a b (-x) = -power_function a b x) :=
by
  intro ha hb
  -- proofs can be filled later with details
  sorry

end power_function_is_odd_l41_41341


namespace quadratic_inequality_l41_41843

theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, a*x^2 + 2*x + 1 > 0) → a > 1 :=
begin
  sorry
end

end quadratic_inequality_l41_41843


namespace number_of_integers_with_prime_divisors_under_100_l41_41022

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

def count_divisors (n : Nat) : Nat := 
  (List.range n).filter (λ i => n % (i + 1) = 0).length

def numbers_with_prime_divisors_under (k : Nat) : List Nat := 
  (List.range k).filter (λ n => is_prime (count_divisors (n + 1)))

theorem number_of_integers_with_prime_divisors_under_100 :
  numbers_with_prime_divisors_under 100 = 17 :=
sorry

end number_of_integers_with_prime_divisors_under_100_l41_41022


namespace sum_of_squares_equivalent_l41_41070

variable {A B C M : Point}
variable (H : IsOrthocenter M A B C)
variable {O : Point}
variable (O_circumcenter : IsCircumcenter O A B C)
variable (M' : Point)
variable (H_reflection : ReflectsOverM M O M')

theorem sum_of_squares_equivalent :
  Sum_of_Squares_Of_Sides (triangle M' A C) =
  Sum_of_Squares_Of_Sides (triangle M' B C) ∧
  Sum_of_Squares_Of_Sides (triangle M' A C) =
  Sum_of_Squares_Of_Sides (triangle M' A B) :=
sorry

end sum_of_squares_equivalent_l41_41070


namespace sum_of_fractions_non_trivial_root_sum_of_fractions_trivial_root_l41_41097

noncomputable def seventh_roots_of_unity (q : ℂ) : Prop := q^7 = 1

-- Statement for the non-trivial 7th root unity case
theorem sum_of_fractions_non_trivial_root (q : ℂ) (hq : seventh_roots_of_unity(q)) (hq_ne_one : q ≠ 1) :
  (q / (1 + q^2)) + (q^2 / (1 + q^4)) + (q^3 / (1 + q^6)) = -2 :=
sorry

-- Statement for the trivial case where q = 1
theorem sum_of_fractions_trivial_root :
  (1 / (1 + 1^2)) + (1^2 / (1 + 1^4)) + (1^3 / (1 + 1^6)) = (3 / 2) :=
by norm_num

end sum_of_fractions_non_trivial_root_sum_of_fractions_trivial_root_l41_41097


namespace palindromic_condition_l41_41240

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem palindromic_condition (m n : ℕ) :
  is_palindrome (2^n + 2^m + 1) ↔ (m ≤ 9 ∨ n ≤ 9) :=
sorry

end palindromic_condition_l41_41240


namespace robot_wearing_ways_l41_41977

theorem robot_wearing_ways : 
  let total_ways := 2! * 2! * 1
  total_ways = 4 :=
by
  sorry

end robot_wearing_ways_l41_41977


namespace independence_test_incorrect_statement_l41_41259

theorem independence_test_incorrect_statement
  (A : Prop := "Independence tests are based on the principle of small probability.")
  (C : Prop := "The conclusions of independence tests may vary with different samples.")
  (D : Prop := "Independence tests are not the only method to determine whether two things are related.") :
  ¬ ("The conclusions derived from the principle of independence tests are always correct.") :=
by
  sorry

end independence_test_incorrect_statement_l41_41259


namespace Sarah_brother_apples_l41_41525

theorem Sarah_brother_apples (n : Nat) (h1 : 45 = 5 * n) : n = 9 := 
  sorry

end Sarah_brother_apples_l41_41525


namespace inscribed_square_side_length_squared_l41_41631

theorem inscribed_square_side_length_squared :
  ∃ (a : ℝ), a = (5 / 3 - 2 * real.sqrt (2 / 3)) ∧
    ∃ (x y : ℝ), x^2 + 3 * y^2 = 3 ∧ (0, 1) = (0 : ℝ, 1 : ℝ) ∧ (x, y) ≠ (0, 1) ∧
                  (x, y) ≠ (-x, y) ∧ distance (0, 1) (x, y) = real.sqrt a ∧ 
                  distance (0, -1) (-x, y) = real.sqrt a :=
sorry

end inscribed_square_side_length_squared_l41_41631


namespace number_of_diffs_l41_41363

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41363


namespace loss_per_metre_eq_10_l41_41630

-- Definitions from the problem:
def totalSellingPrice : ℝ := 36000
def metresSold : ℝ := 600
def costPricePerMetre : ℝ := 70

-- Calculate the total cost price:
def totalCostPrice : ℝ := costPricePerMetre * metresSold

-- Calculate the total loss:
def totalLoss : ℝ := totalCostPrice - totalSellingPrice

-- Calculate the loss per metre
def lossPerMetre : ℝ := totalLoss / metresSold

-- The theorem to prove that the loss per metre is Rs. 10
theorem loss_per_metre_eq_10 : lossPerMetre = 10 := by
  sorry

end loss_per_metre_eq_10_l41_41630


namespace domain_of_function_l41_41938

noncomputable def f (x : ℝ) : ℝ := log 2 (log 3 (log 5 (log 6 x)))

theorem domain_of_function : ∀ x : ℝ, x > 6^5 → f x > 0 := 
begin
  sorry
end

end domain_of_function_l41_41938


namespace find_fx1_plus_x2_l41_41778

theorem find_fx1_plus_x2
  (f : ℝ → ℝ)
  (varphi : ℝ)
  (h1 : ∀ x, f x = 4 * Real.cos (3 * x + varphi))
  (h2 : |varphi| < Real.pi / 2)
  (h3 : ∀ x, f ((11 * Real.pi) / 12 - x) = f ((11 * Real.pi) / 12 + x))
  (h4 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 ∈ Ioo 0 (5 * Real.pi / 12) → x2 ∈ Ioo 0 (5 * Real.pi / 12) → f x1 = f x2) :
  f (x1 + x2) = 2 * Real.sqrt 2 := 
sorry

end find_fx1_plus_x2_l41_41778


namespace min_value_increase_of_add_x2_l41_41195

-- Definitions for the conditions
variable (f : ℝ → ℝ)
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (x : ℝ)

-- Given conditions
def condition1 : Prop :=
  ∀ x, f(x) = a * x^2 + b * x + c ∧ a > 0 ∧
    ((-b^2 / (4 * (a + 3)) + c) = (-b^2 / (4 * a) + c + 9))

def condition2 : Prop :=
  ∀ x, f(x) = a * x^2 + b * x + c ∧ a > 0 ∧
    ((-b^2 / (4 * (a - 1)) + c) = (-b^2 / (4 * a) + c - 9))

-- The proof problem
theorem min_value_increase_of_add_x2 (h1 : condition1 f a b c) (h2 : condition2 f a b c) :
  ∀ x, let new_f := fun x => (a + 1) * x^2 + b * x + c in
    ((-b^2 / (4 * (a + 1)) + c) = (-18 + c + 9/2)) :=
sorry

end min_value_increase_of_add_x2_l41_41195


namespace expression_correct_l41_41900

-- Given conditions: x and y are real numbers
variables {x y : ℝ}

-- Define the expression "the square of x minus half of y"
def expression : ℝ := x^2 - y / 2

-- The theorem to prove that the expression is equal to the expected answer
theorem expression_correct : expression = x^2 - y / 2 :=
by
  -- proof placeholder
  sorry

end expression_correct_l41_41900


namespace system_no_failure_over_time_interval_l41_41242

noncomputable def system_no_failure_probability (n : ℕ) (lambda_i : ℝ) (delta_t : ℝ) : ℝ :=
  let lambda_s := n * lambda_i
  real.exp (- lambda_s * delta_t)

theorem system_no_failure_over_time_interval :
  system_no_failure_probability 1000 10^(-6) 1000 ≈ 0.37 := by
  sorry

end system_no_failure_over_time_interval_l41_41242


namespace innovation_sequence_problem_part1_innovation_sequence_problem_part2_l41_41841

noncomputable def innovation_sequence (l : List ℕ) : List ℕ :=
l.foldl (λ acc x, acc ++ [max x (acc.getLast! 0)]) []

def innovation_order (l : List ℕ) : ℕ :=
(innovation_sequence l).eraseDuplicates.length

theorem innovation_sequence_problem_part1 :
  let seq1 : List ℕ := [3, 4, 1, 5, 2]
  let seq2 : List ℕ := [3, 4, 2, 5, 1]
  innovation_sequence seq1 = [3, 4, 4, 5, 5] ∧
  innovation_sequence seq2 = [3, 4, 4, 5, 5] ∧
  ∀ seq : List ℕ, seq ~ (List.range 5).map Nat.succ →
  innovation_sequence seq = [3, 4, 4, 5, 5] →
  seq = seq1 ∨ seq = seq2 :=
by
  sorry

theorem innovation_sequence_problem_part2 (m : ℕ) (hm : 3 < m) :
  ∃ seq : List ℕ, seq ~ List.range m.map Nat.succ ∧
  ∃ d : ℕ, ∀ i : ℕ, i < m - 1 → (innovation_sequence seq).get? i + d = (innovation_sequence seq).get? (i + 1) ↔ 
  seq = (List.range m).map Nat.succ ∨ ∃ n, n > 0 ∧ (seq = n :: (List.erase [1, 2, ..., m] n)) :=
by
  sorry

end innovation_sequence_problem_part1_innovation_sequence_problem_part2_l41_41841


namespace scientific_notation_850000_l41_41997

theorem scientific_notation_850000 : ∃ a n : ℝ, 850000 = a * 10^n ∧ a = 8.5 ∧ n = 5 := 
by {
  use 8.5,
  use 5,
  sorry
}

end scientific_notation_850000_l41_41997


namespace smallest_number_divisible_by_6_is_123456_l41_41105

open Finset

def is_six_digit_number_with_1_to_6 := {n : ℕ | (multiset.of_nat_digits (Finset.univ.map coe )).to_finset = ({1, 2, 3, 4, 5, 6} : Finset ℕ) }

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

noncomputable def smallest_divisible_by_6 : ℕ :=
  Finset.min' (filter is_divisible_by_6 (is_six_digit_number_with_1_to_6)) sorry

theorem smallest_number_divisible_by_6_is_123456 : smallest_divisible_by_6 = 123456 :=
sorry

end smallest_number_divisible_by_6_is_123456_l41_41105


namespace power_function_passing_point_mono_inc_l41_41147

noncomputable def power_function_mono_inc :
  ∃ a : ℝ, a < 0 ∧ ∀ x y : ℝ, x < y → 0 < x → 0 < y → x^a < y^a := by
  sorry

theorem power_function_passing_point_mono_inc :
  ∃ a : ℝ, 2^a = 1/4 ∧ ∀ x y : ℝ, x < y → 0 < x → 0 < y → x^a < y^a := by
  have h : ∃ a : ℝ, 2^a = 1/4 := by
    exists (-2 : ℝ)
    norm_num
  cases h with a ha
  exists a
  split
  · exact ha
  exact power_function_mono_inc a ha sorry

end power_function_passing_point_mono_inc_l41_41147


namespace part_a_part_b_l41_41959

noncomputable theory

variables {α : Type*} [measurable_space α] {μ : measure α}

-- Part (a)
theorem part_a (f : ℝ → ℝ) (h_lipschitz : ∀ x y : ℝ, |f x - f y| ≤ |x - y|)
  (ξ : ℝ) [is_random_variable ξ] (h_var : is_variance_finite ξ) :
  variance f ξ ≤ variance ξ ∧ abs (expectation (ξ * f ξ) - expectation ξ * expectation f ξ) ≤ variance ξ := 
sorry 

-- Part (b)
theorem part_b (f : ℝ → ℝ) (h_nondec : ∀ x y, x ≤ y → f x ≤ f y)
  (ξ : ℝ) [is_random_variable ξ] (h_moment : is_moment_finite ξ p) :
  expectation (|f ξ - expectation f ξ|^p) ≤ expectation (|ξ - expectation ξ|^p) :=
sorry 

end part_a_part_b_l41_41959


namespace sum_binom_eq_l41_41131

theorem sum_binom_eq {m n : ℕ} (hm : 0 < m) (hn : 0 < n) (hnm : n ≥ m) :
  ∑ k in Finset.range (n - m + 1), (-1) ^ (k + m) * Nat.choose (k + m) m * Nat.choose (n + (k + m)) (2 * (k + m)) * 2 ^ (2 * (k + m)) = 
  (-1) ^ n * 2 ^ (2 * m) * (2 * n + 1) / (2 * m + 1) * Nat.choose (m + n) (2 * m) :=
by sorry

end sum_binom_eq_l41_41131


namespace hexagonal_prism_sphere_volume_l41_41968

noncomputable def volume_of_sphere (h_prism : ℝ) (c_base : ℝ) : ℝ :=
  let edge_length := c_base / 6
  let principal_diagonal := 2 * edge_length
  let diameter := Math.sqrt (h_prism^2 + principal_diagonal^2)
  let radius := diameter / 2
  (4 / 3) * Real.pi * radius^3

theorem hexagonal_prism_sphere_volume :
  volume_of_sphere (Real.sqrt 3) 3 = (4 / 3) * Real.pi * 1^3 :=
by
  sorry

end hexagonal_prism_sphere_volume_l41_41968


namespace mn_lt_xy_l41_41029

variables {m n : ℚ}
def x := (m + 2 * n) / 3
def y := (2 * m + n) / 3

theorem mn_lt_xy (h : m ≠ n) : m * n < x * y :=
by sorry

end mn_lt_xy_l41_41029


namespace sum_infinite_series_l41_41276

theorem sum_infinite_series :
  (∑' n : ℕ, (4 * (n + 1) + 1) / (3^(n + 1))) = 7 / 2 :=
sorry

end sum_infinite_series_l41_41276


namespace ratio_of_areas_l41_41894

-- Definitions of the given conditions
variables (AB CD h : ℝ)
variables (AH BG CF DG : ℝ)
variables (K L M N : Point)

-- The bases of the trapezoid ABCD
def AB := 15
def CD := 19

-- Altitudes to the lines
def AH : ℝ
def BG : ℝ
def CF : ℝ

-- Midpoints of the segments
def K : Point -- Midpoint of AB
def L : Point -- Midpoint of CF
def M : Point -- Midpoint of CD
def N : Point -- Midpoint of AH

-- Given DG = 17
def DG := 17

-- Areas of trapezoid and quadrilateral
def area_trapezoid := 1 / 2 * (AB + CD) * h
def KL := sorry  -- KL needs to be properly defined or calculated within context

def area_KLMN := KL * (h / 2)

-- The proof we need
theorem ratio_of_areas : 
  let S_ABCD := area_trapezoid in
  let S_KLMN := area_KLMN in
  S_ABCD / S_KLMN = 2 ∨ S_ABCD / S_KLMN = 2 / 3 :=
sorry

end ratio_of_areas_l41_41894


namespace spider_socks_and_shoes_ordering_l41_41978

theorem spider_socks_and_shoes_ordering : 
  let legs := 10
  let items := 3 * legs
  let socks_per_leg_orderings := 2 -- number of ways to order socks on each leg
  let total_socks_and_shoes_orderings := (fact items) / ((fact legs) * (fact (2 * legs))) * socks_per_leg_orderings^legs
  total_socks_and_shoes_orderings = (fact 30) / (fact 10 * fact 20) * 1024
:= sorry

end spider_socks_and_shoes_ordering_l41_41978


namespace quotient_sum_40_5_l41_41584

theorem quotient_sum_40_5 : (40 + 5) / 5 = 9 := by
  sorry

end quotient_sum_40_5_l41_41584


namespace find_a1_a100_l41_41479

section SeqSum

variable (a : ℕ → ℕ)

-- Define the sequence condition as a hypothesis
def sequence_condition (k : ℕ) [hk : Fact (k > 0)] : Prop :=
  a k + a (k + 1) = 2 * k + 1

theorem find_a1_a100 (H : ∀ k > 0, sequence_condition a k) : a 1 + a 100 = 101 := by
  sorry

end SeqSum

end find_a1_a100_l41_41479


namespace binomial_expansion_coefficient_l41_41447

theorem binomial_expansion_coefficient (a : ℝ)
  (h : ∃ r, 9 - 3 * r = 6 ∧ (-a)^r * (Nat.choose 9 r) = 36) :
  a = -4 :=
  sorry

end binomial_expansion_coefficient_l41_41447


namespace number_of_cheeses_per_pack_l41_41825

-- Definitions based on the conditions
def packs : ℕ := 3
def cost_per_cheese : ℝ := 0.10
def total_amount_paid : ℝ := 6

-- Theorem statement to prove the number of string cheeses in each pack
theorem number_of_cheeses_per_pack : 
  (total_amount_paid / (packs : ℝ)) / cost_per_cheese = 20 :=
sorry

end number_of_cheeses_per_pack_l41_41825


namespace six_digit_numbers_with_zero_l41_41010

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41010


namespace current_average_runs_l41_41231

variable (A : ℝ)

-- Conditions
def condition1 : Prop := ∃ A, true
def condition2 : Prop := 10 * A + 81 = 11 * (A + 4)

-- Theorem to prove
theorem current_average_runs 
  (h1 : condition1) (h2 : condition2) : A = 37 := 
sorry

end current_average_runs_l41_41231


namespace carlos_wins_one_game_l41_41827

def games_Won_Laura : ℕ := 5
def games_Lost_Laura : ℕ := 4
def games_Won_Mike : ℕ := 7
def games_Lost_Mike : ℕ := 2
def games_Lost_Carlos : ℕ := 5
variable (C : ℕ) -- Carlos's wins

theorem carlos_wins_one_game :
  games_Won_Laura + games_Won_Mike + C = (games_Won_Laura + games_Lost_Laura + games_Won_Mike + games_Lost_Mike + C + games_Lost_Carlos) / 2 →
  C = 1 :=
by
  sorry

end carlos_wins_one_game_l41_41827


namespace unique_zero_in_interval_l41_41963

noncomputable def f (x : ℝ) : ℝ := 2^x + x^3 - 2

theorem unique_zero_in_interval : (∃! x ∈ Ioo 0 1, f x = 0) :=
by
  sorry

end unique_zero_in_interval_l41_41963


namespace surface_area_ratio_l41_41041

theorem surface_area_ratio (r : ℝ) (h : r > 0)
  (cylinder_diameter_eq_sphere_diameter : true)
  (cylinder_height_eq_sphere_diameter : true) :
  let S1 := (2 * r * pi) * r + (2 * r * r * pi) in
  let S2 := 4 * pi * r^2 in
  S1 / S2 = 3 / 2 :=
by
  sorry

end surface_area_ratio_l41_41041


namespace pirate_coins_total_l41_41522

theorem pirate_coins_total (x : ℕ) (h1 : 4 * x = x * (x + 1) / 2) : 5 * x = 35 :=
by
  have hx : x ≠ 0 := sorry
  -- This is implied by the context but needs proof.
  have h2 : x = 7 := sorry
  -- Solve for x to be 7 from 4x = x(x + 1) / 2.
  rw [h2]
  -- Substituting x = 7.
  norm_num
  -- Perform actual multiplication
  sorry

end pirate_coins_total_l41_41522


namespace negation_of_proposition_l41_41909

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_of_proposition_l41_41909


namespace min_moves_X_consecutive_l41_41629

def is_valid_sequence (seq : list char) : Prop :=
  seq.length = 40 ∧
  seq.filter (λ x, x = 'X').length = 20 ∧
  seq.filter (λ x, x = 'O').length = 20

def min_moves_to_consecutive_xs (seq : list char) : ℕ :=
  -- This function computes the minimum number of swaps to make 20 consecutive X's
  sorry

theorem min_moves_X_consecutive (seq : list char) (h : is_valid_sequence seq) :
  min_moves_to_consecutive_xs seq = 200 :=
sorry

end min_moves_X_consecutive_l41_41629


namespace sum_term_decimal_eq_5_div_2_l41_41697

def is_term_decimal (n : ℕ) : Prop :=
  ∃ i j : ℕ, n = 2^i * 5^j

noncomputable def sum_term_decimal : ℝ :=
  ∑' (n : ℕ) in finset.filter is_term_decimal (finset.range (1000)), (1 / n.to_real)

theorem sum_term_decimal_eq_5_div_2 :
  sum_term_decimal = 5 / 2 :=
sorry

end sum_term_decimal_eq_5_div_2_l41_41697


namespace line_divides_side_BC_in_ratio_l41_41802

-- Mathematically defining the problem in Lean 4
theorem line_divides_side_BC_in_ratio 
  (A B C : Type) [euclidean_geometry A B C] 
  (acute_triangle : ∀ (A B C : triangle) (h : ∠ABC < 90° ∧ ∠BCA < 90°), acute-angled-triangle A B C)
  (line_median_altitude_condition : ∀ (A B C X : Point) 
      (s : Line) 
      (H : ∃ M : X (is_median B) ∧ s passing_through A ∧ s passing_through M) 
      (vertex_C: Altitude-From C) 
      (divides_altitude : s divides vertex_C into 3 equal parts), 
      divides_side_in_ratio A B C s (side BC) = 1/4) : 
   divides_side_in_ratio A B C s (side BC) = 1/4 :=
sorry

end line_divides_side_BC_in_ratio_l41_41802


namespace abhay_speed_l41_41955

-- Definitions of the problem's conditions
def condition1 (A S : ℝ) : Prop := 42 / A = 42 / S + 2
def condition2 (A S : ℝ) : Prop := 42 / (2 * A) = 42 / S - 1

-- Define Abhay and Sameer's speeds and declare the main theorem
theorem abhay_speed (A S : ℝ) (h1 : condition1 A S) (h2 : condition2 A S) : A = 10.5 :=
by
  sorry

end abhay_speed_l41_41955


namespace min_sum_pow_3_l41_41044

theorem min_sum_pow_3 (n : ℕ) (x : fin n → ℝ) (h : ∑ i, x i = n) : ∑ i, 3^(x i) ≥ 3 * n := sorry

end min_sum_pow_3_l41_41044


namespace different_subjects_book_combinations_l41_41166

theorem different_subjects_book_combinations 
  (num_math_books : ℕ)
  (num_chinese_books : ℕ)
  (num_english_books : ℕ)
  (h_math : num_math_books = 10)
  (h_chinese : num_chinese_books = 9)
  (h_english : num_english_books = 8) : 
  (num_chinese_books * num_math_books + 
   num_chinese_books * num_english_books + 
   num_math_books * num_english_books) = 242 :=
begin
  sorry
end

end different_subjects_book_combinations_l41_41166


namespace cos_squared_equation_solution_l41_41690

theorem cos_squared_equation_solution (x : ℝ) (k l m : ℤ) :
  (cos(x)^2 + cos(2 * x)^2 + cos(3 * x)^2 = 1)
  ↔ 
  (x = (k * π / 2) ∨ x = (π / 4 + k * π) ∨ x = (π / 4 - k * π) ∨ x = (π / 6 + m * π) ∨ x = (π / 6 - m * π)) := sorry

end cos_squared_equation_solution_l41_41690


namespace chess_tournament_l41_41223

theorem chess_tournament :
  ∀ (participants : Finset ℕ) (pairing : participants → participants → Prop),
  participants.card = 10 →
  (∀ x y ∈ participants, x ≠ y → pairing x y → pairing y x) →
  (∀ x ∈ participants, (∑ y in participants, if pairing x y then 1 else 0).nat_abs = 9) →
  (∀ ⦃x y⦄, x ∈ participants → y ∈ participants → pairing x y ∨ pairing y x) →
  (∃ t : Finset (Finset ℕ), t.card = 2 ∧ ∀ (x ∈ t) (y ∈ t), x ≠ y) →
  (∃ round : finset (participants × participants), round.card = 5 ∧
    ∀ (game ∈ round), ∃ town t, ∀ (game = (x, y)), x ∈ t ∧ y ∈ t) →
  (∃ (x y : participants), x ≠ y ∧ x ∈ participants ∧ y ∈ participants ∧ (pairing x y ∨ pairing y x) ∧
    (∃ town, x ∈ town ∧ y ∈ town)) :=
by sorry

end chess_tournament_l41_41223


namespace gcd_mult_difference_is_square_l41_41846

open Nat

theorem gcd_mult_difference_is_square (x y z : ℕ) 
  (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, k * k = gcd x y z * (y - x) :=
sorry

end gcd_mult_difference_is_square_l41_41846


namespace complement_union_l41_41086

open Finset

noncomputable def U : Finset ℕ := {0, 1, 2, 3, 4, 5}

noncomputable def A : Finset ℕ := {x ∈ U | x * x - 7 * x + 12 = 0}

noncomputable def B : Finset ℕ := {1, 3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {0, 2} :=
by
  sorry

end complement_union_l41_41086


namespace painting_equivalence_l41_41805

-- Define the problem
def painting_problem := ∀ (D: Finset ℕ), (D.card = 5) →
  ((∀ (N: ℕ), N ∈ D → N = 1 ∨ N = 0) ∧ 
  (D.filter (λ n, n = 1)).card = 2 ∧ 
  (D.filter (λ n, n = 0)).card = 2 ∧ 
  (D.filter (λ n, n = -1)).card = 1) →
  -- Count of distinct colorings considering rotation and reflection symmetries
  count_distinct_colorings D = 13

-- Main theorem statement: count of distinct valid paintings (up to symmetries) equals 13
theorem painting_equivalence : painting_problem :=
sorry

end painting_equivalence_l41_41805


namespace positive_difference_l41_41165

noncomputable def heights := [220, 140, 180, 305, 195, 290, 160]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem positive_difference :
  let diff := (mean heights).toNat - median heights in
  abs diff = 18 := 
by
  sorry

end positive_difference_l41_41165


namespace six_digit_numbers_with_zero_l41_41012

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41012


namespace evaluate_expression_l41_41290

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12)) ^ 2 = 1600 := by
  sorry

end evaluate_expression_l41_41290


namespace total_metal_rods_needed_l41_41227

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end total_metal_rods_needed_l41_41227


namespace candy_cost_l41_41226

theorem candy_cost (C : ℝ) 
  (h1 : 20 + 40 = 60) 
  (h2 : 5 * 40 + 20 * C = 60 * 6) : 
  C = 8 :=
by
  sorry

end candy_cost_l41_41226


namespace behavior_of_sequences_l41_41845

noncomputable def A (p q : ℝ) (n : ℕ) : ℝ :=
if n = 1 then (p + q) / 2 else (B p q (n - 1) + C p q (n - 1)) / 2

noncomputable def B (p q : ℝ) (n : ℕ) : ℝ :=
if n = 1 then real.sqrt (p * q) else real.sqrt (B p q (n - 1) * C p q (n - 1))

noncomputable def C (p q : ℝ) (n : ℕ) : ℝ :=
if n = 1 then 2 * p * q / (p + q) else 2 / (1 / (B p q (n - 1)) + 1 / (C p q (n - 1)))

theorem behavior_of_sequences (p q : ℝ) (h : p ≠ q) (h_pos_p : p > 0) (h_pos_q : q > 0) :
  (∀ n, A p q n > A p q (n + 1)) ∧ (∀ n, B p q n = B p q (n + 1)) ∧ (∀ n, C p q n < C p q (n + 1)) :=
sorry

end behavior_of_sequences_l41_41845


namespace set_C_cannot_form_right_triangle_l41_41924

theorem set_C_cannot_form_right_triangle :
  ¬(5^2 + 2^2 = 5^2) :=
by
  sorry

end set_C_cannot_form_right_triangle_l41_41924


namespace angle_between_diagonals_l41_41555

variables (α β : ℝ)

theorem angle_between_diagonals (α β : ℝ) :
  ∃ γ : ℝ, γ = Real.arccos (Real.sin α * Real.sin β) :=
by
  existsi Real.arccos (Real.sin α * Real.sin β)
  refl

end angle_between_diagonals_l41_41555


namespace opposite_of_neg_one_third_l41_41157

theorem opposite_of_neg_one_third : -(-1/3) = 1/3 := 
sorry

end opposite_of_neg_one_third_l41_41157


namespace differential_equation_solution_l41_41888

noncomputable def solution : ℝ → ℝ := λ t => 1 - 2 * Real.exp t + Real.exp (4 * t)

theorem differential_equation_solution (x : ℝ → ℝ) (t : ℝ) 
  (h1 : ∀ t, differentiable ℝ x)
  (h2 : ∀ t, differentiable ℝ (deriv x))
  (h_eq : ∀ t, (deriv (deriv x t) - 5 * deriv x t + 4 * x t = 4))
  (h_init1 : x 0 = 0)
  (h_init2 : deriv x 0 = 2):
  x t = solution t :=
by
  sorry

end differential_equation_solution_l41_41888


namespace problem_statement_l41_41754

noncomputable def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := a (n + 1) + a (n + 1)^2

noncomputable def S (n : ℕ) : ℚ :=
1 / (List.prod (List.map (λ k => (1 + a k)) (List.range (n + 1))))

noncomputable def T (n : ℕ) : ℚ :=
(List.range (n + 1)).sum (λ k => 1 / (1 + a k))

theorem problem_statement (n : ℕ) : S n + T n = 1 :=
sorry

end problem_statement_l41_41754


namespace find_rate_of_current_l41_41921

-- Given speed of the boat in still water (km/hr)
def boat_speed : ℤ := 20

-- Given time of travel downstream (hours)
def time_downstream : ℚ := 24 / 60

-- Given distance travelled downstream (km)
def distance_downstream : ℤ := 10

-- To find: rate of the current (km/hr)
theorem find_rate_of_current (c : ℚ) 
  (h1 : distance_downstream = (boat_speed + c) * time_downstream) : 
  c = 5 := 
by sorry

end find_rate_of_current_l41_41921


namespace total_cars_l41_41861

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end total_cars_l41_41861


namespace coefficient_fourth_term_l41_41144

-- Given conditions
def binomial_expansion (x : ℝ) : ℝ := (x + 2 / x^2)^6

-- Problem statement
theorem coefficient_fourth_term :
  ∃ c : ℝ, c = 160 ∧ (∃ k, binomial_expansion x = c * x^k) := sorry

end coefficient_fourth_term_l41_41144


namespace geometric_sequence_problem_l41_41792

variable {G : Type} [CommGroup G]
variable (a : ℕ → G)
variable (q : G)
variable (r : ℕ → G)
variable (a1 a8 a15 a9 a11 : G)
variable (Q : G)[Group: G(CommGroup)]

noncomputable def isGeometricSequence (a : ℕ → G) (q : G) :=
  ∀ n : ℕ, a (n+1) = a n * q

axiom geo_seq : isGeometricSequence a q

axiom cond1 : a 1 * (a 8 ^ 3) * a 15 = 243 
axiom cond2 : a 1 * a 15 = (a 8 ^ 2)

theorem geometric_sequence_problem : (a 9 ^ 3 / a 11) = 9 := by
  sorry

end geometric_sequence_problem_l41_41792


namespace find_a_l41_41918

theorem find_a (a : ℝ) (x : ℝ) (h : ∀ (x : ℝ), 2 * x - a ≤ -1 ↔ x ≤ 1) : a = 3 :=
sorry

end find_a_l41_41918


namespace wire_length_l41_41895

theorem wire_length(
  (d1 d2 : ℝ) 
  (h1 h2 h3 : ℝ) 
  (h2gth1 : h2 - h1) 
  (h2gth3 : h2 - h3) 
  : 
  (d1 = 16) 
  (h1 = 8) 
  (h2 = 22) 
  (h3 = 10) 
  (d2 = 18) 
) : sqrt (d1^2 + h2gth1^2) + sqrt (d2^2 + h2gth3^2) = sqrt 452 + sqrt 468 := 
by {
  sorry
}

end wire_length_l41_41895


namespace sum_of_powers_l41_41773

theorem sum_of_powers (x : ℂ) (h : x^3 + x^2 + x = -1) :
  (Finset.range 57).sum (λ i, x ^ (i - 28)) = 1 :=
by {
  -- proof would go here
  sorry
}

end sum_of_powers_l41_41773


namespace max_min_value_integral_l41_41079

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  ∫ t in (x - a)..(x + a), sqrt (4 * a^2 - t^2)

theorem max_min_value_integral (a : ℝ) (h : 0 < a) :
  (∃ x, abs x ≤ a ∧ f a x = 2 * π * a^2) ∧
  (∃ x, abs x ≤ a ∧ f a x = π * a^2) :=
begin
  sorry
end

end max_min_value_integral_l41_41079


namespace total_cars_l41_41862

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end total_cars_l41_41862


namespace remainder_8_pow_1996_mod_5_l41_41190

theorem remainder_8_pow_1996_mod_5 :
  (8: ℕ) ≡ 3 [MOD 5] →
  3^4 ≡ 1 [MOD 5] →
  8^1996 ≡ 1 [MOD 5] :=
by
  sorry

end remainder_8_pow_1996_mod_5_l41_41190


namespace find_number_of_Persians_l41_41073

variable (P : ℕ)  -- Number of Persian cats Jamie owns
variable (M : ℕ := 2)  -- Number of Maine Coons Jamie owns (given by conditions)
variable (G_P : ℕ := P / 2)  -- Number of Persian cats Gordon owns, which is half of Jamie's
variable (G_M : ℕ := M + 1)  -- Number of Maine Coons Gordon owns, one more than Jamie's
variable (H_P : ℕ := 0)  -- Number of Persian cats Hawkeye owns, which is 0
variable (H_M : ℕ := G_M - 1)  -- Number of Maine Coons Hawkeye owns, one less than Gordon's

theorem find_number_of_Persians (sum_cats : P + M + G_P + G_M + H_P + H_M = 13) : 
  P = 4 :=
by
  -- Proof can be filled in here
  sorry

end find_number_of_Persians_l41_41073


namespace original_number_q_l41_41604

variables (q : ℝ) (a b c : ℝ)
 
theorem original_number_q : 
  (a = 1.125 * q) → (b = 0.75 * q) → (c = 30) → (a - b = c) → q = 80 :=
by
  sorry

end original_number_q_l41_41604


namespace complement_of_intersection_l41_41726

noncomputable def M : Set ℝ := {x | 2^x > 1}
noncomputable def N : Set ℝ := {x | Real.log10 (x^2 - 2*x + 4) > 0}

theorem complement_of_intersection : 
  (Set.univ \ (M ∩ N)) = { x | x ≤ 0 } :=
by
  -- proof goes here
  sorry

end complement_of_intersection_l41_41726


namespace AM_is_not_median_l41_41118

theorem AM_is_not_median (A B C M : Type) (r : ℝ) 
  (hM_on_BC : M ∈ line_segment B C)
  (r_inscribed_Acadm : radius_inscribed_circle (triangle A C M) = r)
  (r_inscribed_ABam : radius_inscribed_circle (triangle A B M) = 2 * r) : 
  ¬is_median (triangle A B C) A M :=
sorry

end AM_is_not_median_l41_41118


namespace quartic_polynomial_l41_41085

noncomputable def Q (x : ℝ) : ℝ := sorry

theorem quartic_polynomial (m : ℝ) 
  (h₀ : Q 0 = m)
  (h₁ : Q 1 = 3m)
  (h₋₁ : Q (-1) = 2m) 
  : Q 3 + Q (-3) = 56m :=
sorry

end quartic_polynomial_l41_41085


namespace hexagonal_curve_area_l41_41896

theorem hexagonal_curve_area :
  let arc_length := (5 * π) / 6,
      hexagon_side := 4,
      num_arcs := 12,
      r := (5 / 2),
      hex_area := (3 * Real.sqrt 3 / 2) * hexagon_side ^ 2,
      sector_area := (5 * π / 6) * (r ^ 2 / 2),
      sectors_total_area := num_arcs * sector_area
  in hex_area + sectors_total_area = 48 * Real.sqrt 3 + 125 * π / 2 :=
by
  sorry

end hexagonal_curve_area_l41_41896


namespace product_of_hypotenuse_segments_eq_area_l41_41904

theorem product_of_hypotenuse_segments_eq_area (x y c t : ℝ) : 
  -- Conditions
  (c = x + y) → 
  (t = x * y) →
  -- Conclusion
  x * y = t :=
by
  intros
  sorry

end product_of_hypotenuse_segments_eq_area_l41_41904


namespace inscription_valid_l41_41607

def Box := Type
def Bellini (b : Box) : Prop := sorry
def Cellini (b : Box) : Prop := sorry

variable (gold_box : Box) (silver_box : Box)

-- Identical inscriptions on both boxes
def inscription (b : Box) : Prop :=
  (Bellini gold_box ∧ Bellini silver_box) ∨ (Cellini gold_box ∨ Cellini silver_box)

-- Conditions from the problem
axiom condition1 : gold_box ≠ silver_box
axiom condition2 : (Bellini gold_box ∨ Bellini silver_box) → False  -- After first box
axiom condition3 : ¬ (Bellini gold_box ∨ Bellini silver_box) ∨ (Bellini gold_box ∧ Bellini silver_box)  -- After second box

-- The inscription to be proven as a valid solution
theorem inscription_valid : inscription gold_box = inscription silver_box :=
begin
  sorry
end

end inscription_valid_l41_41607


namespace polygon_sides_count_l41_41670

-- Definitions for each polygon and their sides
def pentagon_sides := 5
def square_sides := 4
def hexagon_sides := 6
def heptagon_sides := 7
def nonagon_sides := 9

-- Compute the total number of sides
def total_exposed_sides :=
  (pentagon_sides + nonagon_sides - 2) + (square_sides + hexagon_sides + heptagon_sides - 6)

theorem polygon_sides_count : total_exposed_sides = 23 :=
by
  -- Mathematical proof steps can be detailed here
  -- For now, let's assume it is correctly given as a single number
  sorry

end polygon_sides_count_l41_41670


namespace fire_fighting_max_saved_houses_l41_41471

noncomputable def max_houses_saved (n c : ℕ) : ℕ :=
  n^2 + c^2 - n * c - c

theorem fire_fighting_max_saved_houses (n c : ℕ) (h : c ≤ n / 2) :
    ∃ k, k = max_houses_saved n c :=
    sorry

end fire_fighting_max_saved_houses_l41_41471


namespace root_division_l41_41158

theorem root_division :
  (∜ 16 / ∛ 27) = 2 / 3 :=
by sorry

end root_division_l41_41158


namespace total_cost_of_fencing_l41_41693

variable (π : ℝ) [noncomputable] def cost_of_fencing 
    (diameter rate_per_meter : ℝ) : ℝ :=
  let circumference := π * diameter in
  circumference * rate_per_meter

theorem total_cost_of_fencing {d : ℝ} {r : ℝ} (π : ℝ) :
    d = 34 → r = 2 → (cost_of_fencing π d r) = 213.62 := by
  intros
  apply congrArg
  sorry

end total_cost_of_fencing_l41_41693


namespace smallest_possible_w_l41_41499

open Complex

theorem smallest_possible_w (w : ℂ) (h : (Complex.abs (w - 10) + Complex.abs (w - (7 * Complex.I)) = 17)) :
  Complex.abs w = (70 : ℝ) / 17 :=
sorry

end smallest_possible_w_l41_41499


namespace sin_eq_cos_is_sufficient_but_not_necessary_for_cos_2theta_zero_l41_41502

theorem sin_eq_cos_is_sufficient_but_not_necessary_for_cos_2theta_zero (θ : ℝ) :
  (∃ θ : ℝ, sin θ = cos θ) → (cos (2 * θ) = 0) ∧ ¬(∀ θ : ℝ, cos (2 * θ) = 0 → sin θ = cos θ) :=
by
  sorry

end sin_eq_cos_is_sufficient_but_not_necessary_for_cos_2theta_zero_l41_41502


namespace sum_of_digits_in_period_l41_41550

theorem sum_of_digits_in_period (
  a : ℕ := 1,
  b : ℕ := 89,
  m : ℕ := 88,
  d : ℕ → ℕ,
  periodic : ∀ n, d n = d (n % m)
) : 
  \sum_{i=0}^{m-1} d i = 720 := sorry

end sum_of_digits_in_period_l41_41550


namespace problem_I_problem_II_l41_41893

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

-- Sum of the first n terms of the Fibonacci sequence
def S (n : ℕ) : ℕ :=
  (List.range (n + 1)).map fibonacci |>.sum

theorem problem_I : S 7 = 33 :=
by 
  -- The proof will be inserted here.
sorry

theorem problem_II (m : ℕ) (h : fibonacci 2017 = m) : S 2015 = m - 1 :=
by 
  -- The proof will be inserted here.
sorry

end problem_I_problem_II_l41_41893


namespace midpoint_set_of_segments_eq_circle_l41_41060

-- Define the existence of skew perpendicular lines with given properties
variable (a d : ℝ)

-- Conditions: Distance between lines is a, segment length is d
-- The coordinates system configuration
-- Point on the first line: (x, 0, 0)
-- Point on the second line: (0, y, a)
def are_midpoints_of_segments_of_given_length
  (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), 
    p = (x / 2, y / 2, a / 2) ∧ 
    x^2 + y^2 = d^2 - a^2

-- Proof statement
theorem midpoint_set_of_segments_eq_circle :
  { p : ℝ × ℝ × ℝ | are_midpoints_of_segments_of_given_length a d p } =
  { p : ℝ × ℝ × ℝ | ∃ (r : ℝ), p = (r * (d^2 - a^2) / (2*d), r * (d^2 - a^2) / (2*d), a / 2)
    ∧ r^2 * (d^2 - a^2) = (d^2 - a^2) } :=
sorry

end midpoint_set_of_segments_eq_circle_l41_41060


namespace smallest_possible_degree_polynomial_l41_41133

theorem smallest_possible_degree_polynomial
  (p : Polynomial ℚ)
  (h1 : p.eval (3 - Real.sqrt 8) = 0)
  (h2 : p.eval (5 + Real.sqrt 11) = 0)
  (h3 : p.eval (16 - 3 * Real.sqrt 7) = 0)
  (h4 : p.eval (- Real.sqrt 3) = 0) :
  polynomial.degree p = 8 :=
sorry

end smallest_possible_degree_polynomial_l41_41133


namespace radius_of_larger_circle_l41_41137

theorem radius_of_larger_circle:
  ∀ (π : ℝ) (r_small : ℝ) (n : ℕ), r_small = 2 ∧ n = 4 →
  let A_small := π * r_small^2 in
  let A_total := n * A_small in
  ∃ R : ℝ, π * R^2 = A_total ∧ R = 4 := by
begin
  sorry
end

end radius_of_larger_circle_l41_41137


namespace find_value_of_z_l41_41745

open Complex

-- Define the given complex number z and imaginary unit i
def z : ℂ := sorry
def i : ℂ := Complex.I

-- Given condition
axiom condition : z / (1 - i) = i ^ 2019

-- Proof that z equals -1 - i
theorem find_value_of_z : z = -1 - i :=
by
  sorry

end find_value_of_z_l41_41745


namespace triangles_not_similar_l41_41612

open EuclideanGeometry

theorem triangles_not_similar (ABC A'B'C' : Triangle)
  (α β γ α' β' γ' : ℝ)
  (hα : α = α')
  (hβ : β = β')
  (hγ : γ = γ')
  (h_alpha : angle_between_altitude_and_median ABC.vertexA = α)
  (h_beta : angle_between_altitude_and_median ABC.vertexB = β)
  (h_gamma : angle_between_altitude_and_median ABC.vertexC = γ)
  (h_alpha' : angle_between_altitude_and_median A'B'C'.vertexA = α')
  (h_beta' : angle_between_altitude_and_median A'B'C'.vertexB = β')
  (h_gamma' : angle_between_altitude_and_median A'B'C'.vertexC = γ') :
  ¬ (ABC ∼ A'B'C') :=
by
  sorry

end triangles_not_similar_l41_41612


namespace isosceles_triangle_base_angle_l41_41454

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (base_angle : ℝ) 
  (h1 : vertex_angle = 60) 
  (h2 : 2 * base_angle + vertex_angle = 180) : 
  base_angle = 60 := 
by 
  sorry

end isosceles_triangle_base_angle_l41_41454


namespace sine_alpha_sufficient_not_necessary_for_cos_2alpha_l41_41611

theorem sine_alpha_sufficient_not_necessary_for_cos_2alpha (α : ℝ) :
  (sin α = 2 / 3) → (cos (2 * α) = 1 / 9) ∧ ¬ (cos (2 * α) = 1 / 9 → sin α = 2 / 3) :=
by
  sorry

end sine_alpha_sufficient_not_necessary_for_cos_2alpha_l41_41611


namespace overall_profit_percentage_calculation_l41_41969

noncomputable def adjusted_cost (purchase_price : ℝ) (discount_percentage : ℝ) : ℝ :=
  purchase_price - (purchase_price * (discount_percentage / 100))

noncomputable def adjusted_revenue (sale_price : ℝ) (commission_percentage : ℝ) : ℝ :=
  sale_price - (sale_price * (commission_percentage / 100))

noncomputable def profit (revenue : ℝ) (cost : ℝ) : ℝ :=
  revenue - cost

noncomputable def overall_profit_percentage (total_profit : ℝ) (total_cost : ℝ) : ℝ :=
  (total_profit / total_cost) * 100

def book_1_purchase_price : ℝ := 50
def book_1_discount : ℝ := 10
def book_1_sale_price : ℝ := 80
def book_1_commission : ℝ := 5

def book_2_purchase_price : ℝ := 100
def book_2_discount : ℝ := 15
def book_2_sale_price : ℝ := 130
def book_2_commission : ℝ := 3

def book_3_purchase_price : ℝ := 150
def book_3_discount : ℝ := 20
def book_3_sale_price : ℝ := 190
def book_3_commission : ℝ := 7

def book_4_purchase_price : ℝ := 200
def book_4_discount : ℝ := 5
def book_4_sale_price : ℝ := 250
def book_4_commission : ℝ := 10

theorem overall_profit_percentage_calculation :
  let adj_cost1 := adjusted_cost book_1_purchase_price book_1_discount,
      adj_rev1 := adjusted_revenue book_1_sale_price book_1_commission,
      profit1 := profit adj_rev1 adj_cost1,
      adj_cost2 := adjusted_cost book_2_purchase_price book_2_discount,
      adj_rev2 := adjusted_revenue book_2_sale_price book_2_commission,
      profit2 := profit adj_rev2 adj_cost2,
      adj_cost3 := adjusted_cost book_3_purchase_price book_3_discount,
      adj_rev3 := adjusted_revenue book_3_sale_price book_3_commission,
      profit3 := profit adj_rev3 adj_cost3,
      adj_cost4 := adjusted_cost book_4_purchase_price book_4_discount,
      adj_rev4 := adjusted_revenue book_4_sale_price book_4_commission,
      profit4 := profit adj_rev4 adj_cost4,
      total_profit := profit1 + profit2 + profit3 + profit4,
      total_cost := adj_cost1 + adj_cost2 + adj_cost3 + adj_cost4,
      overall_profit_percentage_ := overall_profit_percentage total_profit total_cost
  in abs (overall_profit_percentage_ - 37.23) < 0.01 := sorry

end overall_profit_percentage_calculation_l41_41969


namespace ellipse_equation_and_fixed_point_l41_41746

theorem ellipse_equation_and_fixed_point {P : ℝ × ℝ} :
  let a := 2 in
  let b := sqrt 3 in
  let x := (-1) in
  let e := 1 / 2 in
  let c := 1 in
  let y0 := P.2 in
  let MN_slope := if y0 ≠ 0 then (3 / (4 * y0)) else 0 in
  let l_slope := if y0 ≠ 0 then (-4 * y0 / 3) else 0 in
  let l := if y0 = 0 then x = -1 else x + 1 in
  (frac (x^2) (a^2) + frac (y^2) (b^2) = 1) ∧
  (l = y ∧ y = -frac (4 * y0) (3) * (x + 1 / 4)) →
  (P = (-1 / 4, 0)) := 
begin
  let a := 2,
  let b := sqrt 3,
  let x := -1,
  let e := 1 / 2,
  let c := 1,
  let y0 := P.2,
  let MN_slope := if y0 ≠ 0 then (3 / (4 * y0)) else 0,
  let l_slope := if y0 ≠ 0 then (-4 * y0 / 3) else 0,
  let l := if y0 = 0 then x = -1 else x + 1,

  have h1 : (frac (P.1^2) (a^2) + frac (P.2^2) (b^2) = 1) ∧
    (l = y ∧ y = -frac (4 * y0) (3) * (x + 1 / 4)),
  sorry,
end

end ellipse_equation_and_fixed_point_l41_41746


namespace least_add_for_div_by_9_l41_41615

theorem least_add_for_div_by_9 (initial_amount : ℕ) (vendors : ℕ) (amount_to_add : ℕ) : 
  initial_amount = 329864 → vendors = 9 → amount_to_add = 4 → 
  ∃ new_amount, new_amount = initial_amount + amount_to_add ∧ new_amount % vendors = 0 :=
by
  intros
  simp only [Nat.add, Nat.mul]
  use initial_amount + amount_to_add
  split
  { rfl }
  { sorry }

end least_add_for_div_by_9_l41_41615


namespace wall_width_l41_41250

theorem wall_width (area height : ℕ) (h1 : area = 16) (h2 : height = 4) : area / height = 4 :=
by
  sorry

end wall_width_l41_41250


namespace coeff_x3_in_binomial_expansion_l41_41675

theorem coeff_x3_in_binomial_expansion :
  (∀ (n : ℕ), (expand (1 + 2 * x) (5)).coeff 3 = 80) :=
by
  sorry

end coeff_x3_in_binomial_expansion_l41_41675


namespace sum_xyz_l41_41770

variables {x y z : ℝ}

theorem sum_xyz (hx : x * y = 30) (hy : x * z = 60) (hz : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_l41_41770


namespace max_dot_product_l41_41099

open Real

variables {a b c : ℝ^2}

/-- All vectors are unit vectors -/
def is_unit_vector (v : ℝ^2) : Prop := ⟪v, v⟫ = 1

/-- Two vectors being perpendicular -/
def is_perpendicular (v w : ℝ^2) : Prop := ⟪v, w⟫ = 0

/-- The vectors a, b, and c are in the same plane (ℝ^2 is a plane on its own) -/
noncomputable def in_same_plane (v w x : ℝ^2) : Prop := True  -- This is trivially true in ℝ^2

theorem max_dot_product 
  (a b c : ℝ^2)
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hc : is_unit_vector c)
  (h_perp : is_perpendicular a b) :
  ∃ v : ℝ, v = (c - a) ⬝ (c - b) ∧ v ≤ 1 + sqrt 2 :=
by {
  sorry
}

end max_dot_product_l41_41099


namespace nine_wolves_nine_rams_seven_days_l41_41965

theorem nine_wolves_nine_rams_seven_days 
  (wolves rams : ℕ) 
  (time : ℕ) 
  (h : ∀ n : ℕ, n ≠ 0 → n wolves eat n rams in time days): 
  (∀ n : ℕ, n wolves consume n rams in (time * 1)) := 
by 
  intros n hn 
  sorry

example : ∀ n, 7 wolves eat 7 rams in 7 days ∧ n * ∃ n wolves eat n rams in "7" days by 
  "7 days" 9 wolves eat 9 rams in 7 days :=
by 
  show "7 days"
  ∀ n: "7 days" 
  ∀ n: ℕ 
  "n" sorry

end nine_wolves_nine_rams_seven_days_l41_41965


namespace mother_sold_rings_correct_l41_41683

noncomputable def motherSellsRings (initial_bought_rings mother_bought_rings remaining_rings final_stock : ℤ) : ℤ :=
  let initial_stock := initial_bought_rings / 2
  let total_stock := initial_bought_rings + initial_stock
  let sold_by_eliza := (3 * total_stock) / 4
  let remaining_after_eliza := total_stock - sold_by_eliza
  let new_total_stock := remaining_after_eliza + mother_bought_rings
  new_total_stock - final_stock

theorem mother_sold_rings_correct :
  motherSellsRings 200 300 225 300 = 150 :=
by
  sorry

end mother_sold_rings_correct_l41_41683


namespace profitable_from_8th_year_l41_41786

def cumulative_investment (n : ℕ) : ℝ :=
  80 + 20 * (n - 1)

def net_income (n : ℕ) : ℝ :=
  5 * (3 / 2) ^ (n - 1)

def f (n : ℕ) : ℝ :=
  (3 / 2) ^ n - 2 * n - 7

theorem profitable_from_8th_year (n : ℕ) (h : n ≥ 8) : f(n) > 0 :=
by sorry

end profitable_from_8th_year_l41_41786


namespace differences_of_set_l41_41377

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41377


namespace smallest_period_monotonically_increasing_max_value_and_symmetry_l41_41349

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sin (2 * x) + a

theorem smallest_period_monotonically_increasing (a : ℝ) :
  (∀ x, f x a = f (x + π) a) ∧
  (∀ k : ℤ, ∀ x, x ∈ Set.Icc (k * π - (3 * π / 8)) (k * π + (π / 8)) → 
    f x a ≤ f (x + δ) a) :=
sorry

theorem max_value_and_symmetry (a : ℝ) (h : (∀ x ∈ Set.Icc 0 (π / 6), 2 = if f x a then x)) :
  a = 1 - Real.sqrt 2 ∧
  (∀ k : ℤ, axis (f (π k / 2) + π / 8) = axis (f k a)) :=
sorry

end smallest_period_monotonically_increasing_max_value_and_symmetry_l41_41349


namespace arithmetic_sequence_a1_a9_l41_41474

theorem arithmetic_sequence_a1_a9 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum_456 : a 4 + a 5 + a 6 = 36) : 
  a 1 + a 9 = 24 := 
sorry

end arithmetic_sequence_a1_a9_l41_41474


namespace difference_of_distinct_members_l41_41413

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41413


namespace dot_product_range_l41_41025

open Real

variables (a b c : ℝ^3) (θ : ℝ)

def magnitude (v : ℝ^3) : ℝ := (v.1^2 + v.2^2 + v.3^2).sqrt
def dot_product (u v : ℝ^3) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def condition_1 := magnitude a = 3
def condition_2 := magnitude b = 4
def condition_3 := c = 2 • b

theorem dot_product_range (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) :
  -24 ≤ dot_product a c ∧ dot_product a c ≤ 24 :=
by sorry

end dot_product_range_l41_41025


namespace log_a_fraction_l41_41768

theorem log_a_fraction (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : (log a (3 / 5) < 1) ↔ (0 < a ∧ a < 3 / 5) ∨ (a > 1) :=
by
  sorry

end log_a_fraction_l41_41768


namespace simplify_and_evaluate_l41_41127

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) : 
  (1 - 1 / (a + 1)) * ((a^2 + 2 * a + 1) / a) = Real.sqrt 2 := 
by {
  sorry
}

end simplify_and_evaluate_l41_41127


namespace limit_a_n_l41_41902

def a_n (n : ℕ) : ℝ := if n ≤ 100 then (1 / 3) ^ n else (2 * n + 1) / (5 * n - 1)

theorem limit_a_n : tendsto (λ n, a_n n) at_top (nhds (2 / 5)) :=
by sorry

end limit_a_n_l41_41902


namespace sum_of_series_eq_5_over_16_l41_41906

theorem sum_of_series_eq_5_over_16 :
  ∑' n : ℕ, (n + 1 : ℝ) / (5 : ℝ)^(n + 1) = 5 / 16 := by
  sorry

end sum_of_series_eq_5_over_16_l41_41906


namespace billy_ate_72_cherries_l41_41999

-- Definitions based on conditions:
def initial_cherries : Nat := 74
def remaining_cherries : Nat := 2

-- Problem: How many cherries did Billy eat?
def cherries_eaten := initial_cherries - remaining_cherries

theorem billy_ate_72_cherries : cherries_eaten = 72 :=
by
  -- proof here
  sorry

end billy_ate_72_cherries_l41_41999


namespace projection_of_a_in_direction_of_b_l41_41562

/-- Define the projection of a vector a in the direction of vector b -/
noncomputable def projection (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / real.sqrt (b.1 ^ 2 + b.2 ^ 2)

/-- The vectors a and b as given in the problem -/
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, 4)

/-- The theorem statement asserting the projection value -/
theorem projection_of_a_in_direction_of_b : projection a b = 1 / 5 :=
by
  sorry -- Proof steps would go here

end projection_of_a_in_direction_of_b_l41_41562


namespace calculate_expression_l41_41354

theorem calculate_expression (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 :=
by
  sorry

end calculate_expression_l41_41354


namespace solve_for_x_l41_41886

theorem solve_for_x (x : ℝ) : 16^4 = (8^3 / 4) * 2^(8 * x) ↔ x = 9 / 8 := 
sorry

end solve_for_x_l41_41886


namespace neg_parallelogram_is_rhombus_l41_41030

def parallelogram_is_rhombus := true

theorem neg_parallelogram_is_rhombus : ¬ parallelogram_is_rhombus := by
  sorry

end neg_parallelogram_is_rhombus_l41_41030


namespace aisha_additional_miles_l41_41990

theorem aisha_additional_miles
  (D : ℕ) (d : ℕ) (v1 : ℕ) (v2 : ℕ) (v_avg : ℕ)
  (h1 : D = 18) (h2 : v1 = 36) (h3 : v2 = 60) (h4 : v_avg = 48)
  (h5 : d = 30) :
  (D + d) / ((D / v1) + (d / v2)) = v_avg :=
  sorry

end aisha_additional_miles_l41_41990


namespace owl_cost_in_gold_l41_41760

-- Definitions for conditions
def spellbook_cost_gold := 5
def potionkit_cost_silver := 20
def num_spellbooks := 5
def num_potionkits := 3
def silver_per_gold := 9
def total_payment_silver := 537

-- Function to convert gold to silver
def gold_to_silver (gold : ℕ) : ℕ := gold * silver_per_gold

-- Function to compute total cost in silver for spellbooks and potion kits
def total_spellbook_cost_silver : ℕ :=
  gold_to_silver spellbook_cost_gold * num_spellbooks

def total_potionkit_cost_silver : ℕ :=
  potionkit_cost_silver * num_potionkits

-- Function to calculate the cost of the owl in silver
def owl_cost_silver : ℕ :=
  total_payment_silver - (total_spellbook_cost_silver + total_potionkit_cost_silver)

-- Function to convert the owl's cost from silver to gold
def owl_cost_gold : ℕ :=
  owl_cost_silver / silver_per_gold

-- The proof statement
theorem owl_cost_in_gold : owl_cost_gold = 28 :=
  by
    sorry

end owl_cost_in_gold_l41_41760


namespace triangle_third_side_l41_41339

theorem triangle_third_side (a b c : ℕ) (h₀ : a = 10) (h₁ : b = 5) :
  (c > abs (a - b) ∧ c < a + b) → c = 8 :=
by
  intros h
  rw [h₀, h₁] at h
  cases h with h₂ h₃
  sorry

end triangle_third_side_l41_41339


namespace four_identical_shapes_l41_41115

/-!
## Problem Statement
Given a symmetric cardboard figure on a grid, prove that cutting the figure vertically and horizontally through the center results in four identical shapes.
-/

def cardboard_is_symmetric (figure : Type) : Prop :=
  -- Here we could mention properties like figure being symmetric regarding center vertical and horizontal lines.
  sorry

theorem four_identical_shapes (figure : Type) (h : cardboard_is_symmetric figure) :
  -- Prove that the resulting shapes after the cut are identical
  let cut1_result := sorry in  -- Define the cut operation
  let cut2_result := sorry in  -- Use symmetry to partition into four regions
  ∀ quadrant1 quadrant2 quadrant3 quadrant4, 
    -- Specify that cutting results in 4 identical shapes
    sorry

end four_identical_shapes_l41_41115


namespace scrap_metal_collected_l41_41911

theorem scrap_metal_collected (a b : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9)
  (h₂ : 900 + 10 * a + b - (100 * a + 10 * b + 9) = 216) :
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by
  sorry

end scrap_metal_collected_l41_41911


namespace area_of_shaded_region_l41_41271

-- Define the vertices of the larger square
def large_square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the polygon forming the shaded area
def shaded_polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 30), (40, 40), (10, 40), (0, 10)]

-- Provide the area of the larger square for reference
def large_square_area : ℝ := 1600

-- Provide the area of the triangles subtracted
def triangles_area : ℝ := 450

-- The main theorem stating the problem:
theorem area_of_shaded_region :
  let shaded_area := large_square_area - triangles_area
  shaded_area = 1150 :=
by
  sorry

end area_of_shaded_region_l41_41271


namespace stops_finite_moves_l41_41483

-- The definition of the initial equation and its transformation rules.
variable {a1 b1 c1 : ℤ}
variable {a : ℕ → ℤ} (b : ℕ → ℤ) (c : ℕ → ℤ)
variable (n : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  (a1 + c1) * b1 > 0

-- Transformation rules
def a_next (n : ℕ) := b n + c n
def b_next (n : ℕ) := a n + c n
def c_next (n : ℕ) := a n + b n

-- Prove that Alice will stop after a finite number of moves
theorem stops_finite_moves (h : initial_conditions) :
  ∃ k : ℕ, ¬(∃ x : ℝ, a k * x^2 + b k * x + c k = 0) :=
sorry

end stops_finite_moves_l41_41483


namespace solve_system_l41_41533

def inequality1 (x : ℝ) : Prop := 5 / (x + 3) ≥ 1

def inequality2 (x : ℝ) : Prop := x^2 + x - 2 ≥ 0

def solution (x : ℝ) : Prop := (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)

theorem solve_system (x : ℝ) : inequality1 x ∧ inequality2 x → solution x := by
  sorry

end solve_system_l41_41533


namespace smallest_c_minus_a_l41_41933

theorem smallest_c_minus_a (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prod : a * b * c = 362880) (h_ineq : a < b ∧ b < c) : 
  c - a = 109 :=
sorry

end smallest_c_minus_a_l41_41933


namespace inequality_proof_l41_41077

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31 :=
sorry

end inequality_proof_l41_41077


namespace angle_C_eq_2pi_over_3_l41_41505

variable {α : Type*} [normed_field α] [normed_space α α]

theorem angle_C_eq_2pi_over_3
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : C + A + B = π)
  (h2 : (a + b - c) * (a + b + c) = a * b) :
  C = 2 * π / 3 := by
  sorry

end angle_C_eq_2pi_over_3_l41_41505


namespace hyperbola_standard_equation_l41_41738

theorem hyperbola_standard_equation (a b : ℝ) (x y : ℝ)
  (H₁ : 2 * a = 2) -- length of the real axis is 2
  (H₂ : y = 2 * x) -- one of its asymptote equations
  : y^2 - 4 * x^2 = 1 :=
sorry

end hyperbola_standard_equation_l41_41738


namespace peaches_eaten_correct_l41_41614

-- Given conditions
def total_peaches : ℕ := 18
def initial_ripe_peaches : ℕ := 4
def peaches_ripen_per_day : ℕ := 2
def days_passed : ℕ := 5
def ripe_unripe_difference : ℕ := 7

-- Definitions derived from conditions
def ripe_peaches_after_days := initial_ripe_peaches + peaches_ripen_per_day * days_passed
def unripe_peaches_initial := total_peaches - initial_ripe_peaches
def unripe_peaches_after_days := unripe_peaches_initial - peaches_ripen_per_day * days_passed
def actual_ripe_peaches_needed := unripe_peaches_after_days + ripe_unripe_difference
def peaches_eaten := ripe_peaches_after_days - actual_ripe_peaches_needed

-- Prove that the number of peaches eaten is equal to 3
theorem peaches_eaten_correct : peaches_eaten = 3 := by
  sorry

end peaches_eaten_correct_l41_41614


namespace product_of_invertibles_mod_24_l41_41094

theorem product_of_invertibles_mod_24 :
  let n := 24 in
  let invertible_mod_n (k: ℕ) : Prop := Nat.gcd k n = 1
  let numbers := filter invertible_mod_n (List.range n)
  let m := List.prod numbers
  m % n = 1 := 
by
  let n := 24
  let invertible_mod_n := λ k, Nat.gcd k n = 1
  let numbers := filter invertible_mod_n (List.range n)
  let m := List.prod numbers
  have : (m % n) = 1 := sorry
  exact this

end product_of_invertibles_mod_24_l41_41094


namespace smallest_k_multiple_of_360_l41_41301

theorem smallest_k_multiple_of_360 :
  ∃ (k : ℕ), (k > 0 ∧ (k = 432) ∧ (2160 ∣ k * (k + 1) * (2 * k + 1))) :=
by
  complication_sorry_proved

end smallest_k_multiple_of_360_l41_41301


namespace ratio_condition_l41_41045

theorem ratio_condition (x y a b : ℝ) (h1 : 8 * x - 6 * y = a) 
  (h2 : 9 * y - 12 * x = b) (hx : x ≠ 0) (hy : y ≠ 0) (hb : b ≠ 0) : 
  a / b = -2 / 3 := 
by
  sorry

end ratio_condition_l41_41045


namespace monotonic_sufficient_condition_l41_41962

theorem monotonic_sufficient_condition (a : ℝ) : a ≥ 2 → 
  ∀ x y ∈ (Set.Icc 1 2), (x ≤ y → (x^2 - 2 * a * x + 3) ≤ (y^2 - 2 * a * y + 3)) ∧ ¬(∀ a, ∀ x y ∈ Set.Icc 1 2, (x^2 - 2 * a * x + 3) ≤ (y^2 - 2 * a * y + 3)) :=
begin
  sorry,
end

end monotonic_sufficient_condition_l41_41962


namespace equivalent_expression_l41_41489

theorem equivalent_expression (x : ℝ) : 
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) + 1 = x^4 := 
by
  sorry

end equivalent_expression_l41_41489


namespace number_of_diffs_l41_41367

-- Define the set S as a finset of integers from 1 to 20
def S : Finset ℕ := Finset.range 21 \ {0}

-- Define a predicate that checks if an integer can be represented as the difference of two distinct members of S
def is_diff (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ n = (a - b).natAbs

-- Define the set of all positive differences that can be obtained from pairs of distinct elements in S
def diff_set : Finset ℕ := Finset.filter (λ n, is_diff n) (Finset.range 21)

-- The main theorem
theorem number_of_diffs : diff_set.card = 19 :=
by sorry

end number_of_diffs_l41_41367


namespace minimum_trapezoid_area_l41_41957

variables 
  (A B C D M N E S T : Type)
  [Trapezoid ABCD] -- ABCD is a trapezoid
  (area_AMS area_SET area_TND : ℝ)
  (MN_parallel : Parallel MN BC) -- MN is parallel to BC and AD
  (MN_parallel_to_AD : Parallel MN AD)
  (AMS_area : area_AMS = 12)
  (SET_area : area_SET = 8)
  (TND_area : area_TND = 15)

theorem minimum_trapezoid_area : 
  ∃ (area : ℝ), (area ≥ 125) :=
by 
  -- The proof is omitted
  sorry

end minimum_trapezoid_area_l41_41957


namespace sequence_term_and_k_value_l41_41722

/-- Given a sequence {a_n} whose sum of the first n terms is S_n = n^2 - 9n,
    prove the sequence term a_n = 2n - 10, and if 5 < a_k < 8, then k = 8. -/
theorem sequence_term_and_k_value (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 9 * n) :
  (∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) →
  (∀ n, a n = 2 * n - 10) ∧ (∀ k, 5 < a k ∧ a k < 8 → k = 8) :=
by {
  -- Given S_n = n^2 - 9n, we need to show a_n = 2n - 10 and verify when 5 < a_k < 8, then k = 8
  sorry
}

end sequence_term_and_k_value_l41_41722


namespace number_of_triangles_l41_41668

theorem number_of_triangles (x y : ℕ) (P Q : ℕ × ℕ) (O : ℕ × ℕ := (0,0)) (area : ℕ) :
  (P ≠ Q) ∧ (P.1 * 31 + P.2 = 2023) ∧ (Q.1 * 31 + Q.2 = 2023) ∧ 
  (P.1 ≠ Q.1 → P.1 - Q.1 = n ∧ 2023 * n % 6 = 0) → area = 165 :=
sorry

end number_of_triangles_l41_41668


namespace car_return_speed_l41_41222

noncomputable def round_trip_speed (d : ℝ) (r : ℝ) : ℝ :=
  let travel_time_to_B := d / 75
  let break_time := 1 / 2
  let travel_time_to_A := d / r
  let total_time := travel_time_to_B + travel_time_to_A + break_time
  let total_distance := 2 * d
  total_distance / total_time

theorem car_return_speed :
  let d := 150
  let avg_speed := 50
  round_trip_speed d 42.857 = avg_speed :=
by
  sorry

end car_return_speed_l41_41222


namespace minimum_moves_to_guess_number_l41_41110

-- Problem definition in Lean 4 with conditions and question to be proven.

theorem minimum_moves_to_guess_number (number : ℕ) :
  (∀ (p1 p2 p3 p4 p5 : ℕ),
     (number /= 0)
     ∧ (number < 100000)
     ∧ (p1 /= p2) 
     ∧ (p1 /= p3) 
     ∧ (p1 /= p4) 
     ∧ (p1 /= p5)
     ∧ (p2 /= p3) 
     ∧ (p2 /= p4) 
     ∧ (p2 /= p5) 
     ∧ (p3 /= p4)
     ∧ (p3 /= p5)
     ∧ (p4 /= p5)
     → (number % 10 /= number % 100 / 10)
     ∧ (number % 10 /= number % 1000 / 100)
     ∧ (number % 10 /= number % 10000 / 1000)
     ∧ (number % 10 /=  number / 10000)
     ∧ (number % 100 / 10 /= number % 1000 / 100)
     ∧ (number % 100 / 10 /= number % 10000 / 1000)
     ∧ (number % 100 / 10 /=  number / 10000)
     ∧ (number % 1000 / 100 /= number % 10000 / 1000)
     ∧ (number % 1000 / 100 /=  number / 10000)
     ∧ (number % 10000 / 1000 /=  number / 10000))
  → 3 := by sorry

end minimum_moves_to_guess_number_l41_41110


namespace pa_perpendicular_bc_l41_41153

-- Definitions based on problem conditions
variables {Point : Type*} [MetricSpace Point] [AffineSpace Point Vector]
variables (E F G H P A B C : Point)

-- The top-level theorem statement
theorem pa_perpendicular_bc 
  (intersect_at_EFGH : ∃ l1 l2 l3 l4: Line, 
    E ∈ l1 ∧ F ∈ l1 ∧ 
    E ∈ l2 ∧ G ∈ l2 ∧ 
    F ∈ l3 ∧ H ∈ l3 ∧ 
    G ∈ l4 ∧ H ∈ l4) 
  (extension_intersect_at_P : ∃ e1 e2 : Line,
    E ∈ e1 ∧ G ∈ e1 ∧ 
    F ∈ e2 ∧ H ∈ e2 ∧ 
    P ∈ e1 ∧ P ∈ e2) :
  IsPerpendicular (Line.mk P A) (Line.mk B C) :=
sorry

end pa_perpendicular_bc_l41_41153


namespace ajay_total_gain_l41_41252

noncomputable def ajay_gain : ℝ :=
  let cost1 := 15 * 14.50
  let cost2 := 10 * 13
  let total_cost := cost1 + cost2
  let total_weight := 15 + 10
  let selling_price := total_weight * 15
  selling_price - total_cost

theorem ajay_total_gain :
  ajay_gain = 27.50 := by
  sorry

end ajay_total_gain_l41_41252


namespace sum_of_squares_of_geometric_sequence_l41_41719

theorem sum_of_squares_of_geometric_sequence
  (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, (∑ k in Finset.range n, a k) = 2^n - 1) :
  ∀ n : ℕ, ∑ k in Finset.range n, (a k)^2 = (4^n - 1) / 3 :=
by
  sorry

end sum_of_squares_of_geometric_sequence_l41_41719


namespace probability_bolyai_eotvos_meet_l41_41577

theorem probability_bolyai_eotvos_meet (teams : Finset ℕ) (n : ℕ) (h_teams : teams.card = 16) :
  let bolyai := 0
  let eotvos := 1
  ∃ m : ℕ, 
    (m = teams.card - 1) ∧
    (∀ s : Finset (Finset ℕ), s.card = m → bolyai ∈ teams → eotvos ∈ teams →
      (s filter (λ pair, bolyai ∈ pair ∧ eotvos ∈ pair)).card = 1) ∧
    ∃ total_pairings : ℕ, 
      (total_pairings = (teams.card * (teams.card - 1)) / 2) →
      (1 / total_pairings.toReal = (1 / 8 : ℝ)) := sorry

end probability_bolyai_eotvos_meet_l41_41577


namespace number_of_triangles_l41_41679

-- Define a structure representing a triangle with integer angles.
structure Triangle :=
  (A B C : ℕ) -- angles in integer degrees
  (angle_sum : A + B + C = 180)
  (obtuse_A : A > 90)

-- Define a structure representing point D on side BC of triangle ABC such that triangle ABD is right-angled
-- and triangle ADC is isosceles.
structure PointOnBC (ABC : Triangle) :=
  (D : ℕ) -- angle at D in triangle ABC
  (right_ABD : ABC.A = 90 ∨ ABC.B = 90 ∨ ABC.C = 90)
  (isosceles_ADC : ABC.A = ABC.B ∨ ABC.A = ABC.C ∨ ABC.B = ABC.C)

-- Problem Statement:
theorem number_of_triangles (t : Triangle) (d : PointOnBC t): ∃ n : ℕ, n = 88 :=
by
  sorry

end number_of_triangles_l41_41679


namespace young_people_sampled_l41_41791

def num_young_people := 800
def num_middle_aged_people := 1600
def num_elderly_people := 1400
def sampled_elderly_people := 70

-- Lean statement to prove the number of young people sampled
theorem young_people_sampled : 
  (sampled_elderly_people:ℝ) / num_elderly_people = (1 / 20:ℝ) ->
  num_young_people * (1 / 20:ℝ) = 40 := by
  sorry

end young_people_sampled_l41_41791


namespace no_tiling_with_seven_sided_convex_l41_41875

noncomputable def Polygon := {n : ℕ // 3 ≤ n}

def convex (M : Polygon) : Prop := sorry

def tiles_plane (M : Polygon) : Prop := sorry

theorem no_tiling_with_seven_sided_convex (M : Polygon) (h_convex : convex M) (h_sides : 7 ≤ M.1) : ¬ tiles_plane M := sorry

end no_tiling_with_seven_sided_convex_l41_41875


namespace reciprocal_arithmetic_and_increasing_l41_41066

variable {ℕ : Type} [LinearOrderedField ℕ]

def sequence (a : ℕ → ℕ) : ℕ → ℕ
| 1       := 1
| (n + 1) := a n / (a n + 1)

theorem reciprocal_arithmetic_and_increasing :
  (∀ n : ℕ, (1 < n) → (∀ a : ℕ → ℕ, sequence a n = n)) →
  (∀ n : ℕ, (1 < n) → (∀ a : ℕ → ℕ, (sequence a n = n))) :=
by
  intro h1 h2
  have ha : ∀ n : ℕ, ∀ a : ℕ → ℕ, sequence a n = n,
    from sorry
  have hb: ∀ n : ℕ, ∀ a : ℕ → ℕ, (sequence a n = n) → ∀ m, (m < n) → (sequence a m < sequence a n),
    from sorry
  exact ⟨ha, hb⟩

end reciprocal_arithmetic_and_increasing_l41_41066


namespace num_two_digit_numbers_tens_greater_units_l41_41434

theorem num_two_digit_numbers_tens_greater_units : 
  let N := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n / 10 > n % 10)} in
  set.size N = 45 :=
by
  sorry

end num_two_digit_numbers_tens_greater_units_l41_41434


namespace value_when_x_is_neg1_l41_41948

theorem value_when_x_is_neg1 (p q : ℝ) (h : p + q = 2022) : 
  (p * (-1)^3 + q * (-1) + 1) = -2021 := by
  sorry

end value_when_x_is_neg1_l41_41948


namespace Jennifer_apples_l41_41484

-- Define the conditions
def initial_apples : ℕ := 7
def found_apples : ℕ := 74

-- The theorem to prove
theorem Jennifer_apples : initial_apples + found_apples = 81 :=
by
  -- proof goes here, but we use sorry to skip the proof step
  sorry

end Jennifer_apples_l41_41484


namespace step_of_induction_l41_41937

theorem step_of_induction (k : ℕ) (h : ∃ m : ℕ, 5^k - 2^k = 3 * m) :
  5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k := 
by
  sorry

end step_of_induction_l41_41937


namespace sum_fourth_and_sixth_term_eq_l41_41469

section 
  -- The sequence definition according to the problem statement
  def seq : ℕ → ℚ
  | 1     := 1
  | (n+1) := (n+2)^3 / (n+1)^3

  -- Sum of the fourth and the sixth numbers in the sequence equals to the given fraction
  theorem sum_fourth_and_sixth_term_eq : seq 4 + seq 6 = 6119 / 1728 := 
  by
    sorry
end

end sum_fourth_and_sixth_term_eq_l41_41469


namespace sum_of_numbers_l41_41140

-- Define the three numbers
variables (x y z : ℤ)
-- Define the conditions in Lean
def cond1 : Prop := y = 10
def cond2 : Prop := (x + y + z) / 3 = x + 20
def cond3 : Prop := (x + y + z) / 3 = z - 25

-- The theorem statement
theorem sum_of_numbers (h1 : cond1 x y z) (h2 : cond2 x y z) (h3 : cond3 x y z) : x + y + z = 45 := by
  sorry

end sum_of_numbers_l41_41140


namespace percentage_defective_is_0_02_l41_41262

-- Define the conditions
def inspected_meters : ℕ := 10000
def rejected_meters : ℕ := 2

-- Define the percentage of defective meters
noncomputable def percentage_defective : ℚ :=
  (rejected_meters.toRat / inspected_meters.toRat) * 100

-- State the theorem
theorem percentage_defective_is_0_02 : percentage_defective = 0.02 :=
  sorry

end percentage_defective_is_0_02_l41_41262


namespace catherine_pencils_per_friend_l41_41660

theorem catherine_pencils_per_friend :
  ∀ (pencils pens given_pens : ℕ), 
  pencils = pens ∧ pens = 60 ∧ given_pens = 8 ∧ 
  (∃ remaining_items : ℕ, remaining_items = 22 ∧ 
    ∀ friends : ℕ, friends = 7 → 
    remaining_items = (pens - (given_pens * friends)) + (pencils - (given_pens * friends * (pencils / pens)))) →
  ((pencils - (given_pens * friends * (pencils / pens))) / friends) = 6 :=
by 
  sorry

end catherine_pencils_per_friend_l41_41660


namespace isabella_original_hair_length_l41_41821

-- Define conditions from the problem
def isabella_current_hair_length : ℕ := 9
def hair_cut_length : ℕ := 9

-- The proof problem to show original hair length equals 18 inches
theorem isabella_original_hair_length 
  (hc : isabella_current_hair_length = 9)
  (ht : hair_cut_length = 9) : 
  isabella_current_hhair_length + hair_cut_length = 18 := 
sorry

end isabella_original_hair_length_l41_41821


namespace minimize_abs_diff_with_median_l41_41211

noncomputable def ξ : Type := sorry
noncomputable def F_ξ (x : Type) : ℝ := sorry
noncomputable def μ (ξ : Type) : ℝ := sorry
noncomputable def E (expr : Type → Type) : ℝ := sorry

axiom median_property : F_ξ (μ ξ - 1) ≤ 1 / 2 ∧ 1 / 2 ≤ F_ξ (μ ξ)

theorem minimize_abs_diff_with_median :
  ∀ ξ, ∃ μ, (∀ a : ℝ, E |ξ - a ξ| ≥ E |ξ - (μ ξ)|) :=
by
  intros
  apply Exists.intro (μ ξ)
  intros
  sorry

end minimize_abs_diff_with_median_l41_41211


namespace f_2009_plus_f_2010_l41_41280

-- Definition of the conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def period_2 (f : ℝ → ℝ) := ∀ x : ℝ, f (3 * (x + 2) + 1) = f (3 * x + 1)

-- Given values
constant f : ℝ → ℝ
constant f_odd : is_odd f
constant f_period : period_2 f
constant f_one : f 1 = 2010

-- Goal
theorem f_2009_plus_f_2010 : f 2009 + f 2010 = -2010 :=
by
  sorry -- the proof goes here

end f_2009_plus_f_2010_l41_41280


namespace bill_has_6_less_pieces_than_mary_l41_41268

-- Definitions based on the conditions
def total_candy : ℕ := 20
def candy_kate : ℕ := 4
def candy_robert : ℕ := candy_kate + 2
def candy_mary : ℕ := candy_robert + 2
def candy_bill : ℕ := candy_kate - 2

-- Statement of the theorem
theorem bill_has_6_less_pieces_than_mary :
  candy_mary - candy_bill = 6 :=
sorry

end bill_has_6_less_pieces_than_mary_l41_41268


namespace num_diff_positive_integers_l41_41374

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41374


namespace projection_correct_l41_41735

variables (a b : ℝ)
variables (angle_ab : ℝ) (magnitude_a : ℝ) (magnitude_b : ℝ)

noncomputable def projection : ℝ := 
  let vector1 := 2 * a + 3 * b in
  let vector2 := 2 * a + b in
  let cos_angle_between := (16 - 48 * (1 / 2) + 27) / (sqrt 61 * sqrt 13) in
  let projection := sqrt 61 * cos_angle_between in
  projection

theorem projection_correct :
  angle_ab = 120 ∧ magnitude_a = 2 ∧ magnitude_b = 3 →
  projection a b angle_ab magnitude_a magnitude_b = 19 * sqrt 13 / 13 :=
begin
  sorry
end

end projection_correct_l41_41735


namespace six_digit_numbers_with_zero_l41_41017

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41017


namespace angle_ACB_l41_41535

variable {α : ℝ} (hα : α < 90)

theorem angle_ACB (α : ℝ) (hα : α < 90) :
  ∃ (AC : ℝ), ∃ (AB : ℝ), (AC = AB) ∧
  (∀ (A B C : Type) (angle_AOB : ∀ (A B : Type), α), 
    (angle_ACB : ∀ (A B C : Type), 45 - α / 4)) :=
begin
  sorry,
end

end angle_ACB_l41_41535


namespace problem_l41_41710

open Set

theorem problem (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = univ) →
  (A ∩ B = Ioo 3 4) →
  a + b = -7 :=
by
  intros hA hB hUnion hIntersection
  sorry

end problem_l41_41710


namespace sum_of_alpha_values_l41_41667

-- Define the conditions
def Q (x : ℂ) : ℂ := ((x^24 - 1) / (x - 1))^2 - x^23

-- Complex zeros of Q(x) and their corresponding alpha values
axiom alpha_values :
  ∃ (α : ℕ → ℚ), 
    (0 < α 1 ∧ α 1 ≤ α 2 ∧ α 2 ≤ α 3 ∧ α 3 ≤ α 4 ∧ α 4 ≤ α 5 ∧ α 5 < 1) ∧
    (α 1 = 1/25 ∧ α 2 = 1/23 ∧ α 3 = 2/25 ∧ α 4 = 2/23 ∧ α 5 = 3/25)

-- Prove the sum of the first five alpha values
theorem sum_of_alpha_values :
  ∑ i in (Finset.range 5).map (Nat.succ), α i = 161 / 575 :=
by
  rcases alpha_values with ⟨α, hα⟩
  sorry

end sum_of_alpha_values_l41_41667


namespace ratio_of_boys_l41_41466

theorem ratio_of_boys (p : ℚ) (hp : p = (3 / 4) * (1 - p)) : p = 3 / 7 :=
by
  -- Proof would be provided here
  sorry

end ratio_of_boys_l41_41466


namespace problem_l41_41708

def ferry_a_speed : ℝ := 8 -- km/h
def ferry_a_time : ℝ := 2 -- hours
def ferry_a_distance : ℝ := ferry_a_speed * ferry_a_time

def ferry_b_distance : ℝ := 3 * ferry_a_distance
def ferry_b_speed : ℝ := ferry_a_speed + 4 -- km/h
def ferry_b_time : ℝ := ferry_b_distance / ferry_b_speed

def ferry_c_speed : ℝ := ferry_b_speed + 6 -- km/h
def ferry_c_distance : ℝ := ferry_a_distance - 5 -- km
def ferry_c_time : ℝ := ferry_c_distance / ferry_c_speed

def ferry_d_distance : ℝ := 2 * ferry_b_distance
def ferry_d_speed : ℝ := ferry_c_speed - 2 -- km/h
def ferry_d_time : ℝ := ferry_d_distance / ferry_d_speed

theorem problem (h_ferry_a_distance : ferry_a_distance = 16)
               (h_ferry_b_distance : ferry_b_distance = 48)
               (h_ferry_c_distance : ferry_c_distance = 11)
               (h_ferry_d_distance : ferry_d_distance = 96) :
  (ferry_b_distance - ferry_a_distance = 32) ∧
  (ferry_d_time - ferry_c_time ≈ 5.3889) :=
by sorry

end problem_l41_41708


namespace arg_of_element_in_S_l41_41915

def S (a : ℂ) (z : ℂ) : Set ℂ := { w | ∃ z, w = conj(z^2) ∧ arg(z) = a }

theorem arg_of_element_in_S (a : ℂ) (w : ℂ) (z : ℂ) (h : w ∈ S a z) : arg(w) = -2 * a :=
by
  sorry

end arg_of_element_in_S_l41_41915


namespace solution_l41_41473

/-- Definition of an isosceles triangle -/
structure IsoscelesTriangle (A B C I M P H Q : Type) :=
  (is_incenter : I)
  (is_midpoint_BI : M)
  (on_side_AC : P)
  (AP_eq_3PC : True)
  (PI_extends_to_H : True)
  (MH_perp_PH : True)
  (Q_midpoint_arc_AB : True)
  (BH_perp_QH : Prop)

noncomputable def isosceles_triangle_proof (A B C I M P H Q : Type) [t : IsoscelesTriangle A B C I M P H Q] : Prop :=
  t.BH_perp_QH

theorem solution (A B C I M P H Q : Type) [t : IsoscelesTriangle A B C I M P H Q] : isosceles_triangle_proof A B C I M P H Q :=
sorry

end solution_l41_41473


namespace square_area_error_l41_41261

theorem square_area_error (s : ℝ) (h : s > 0): 
  let measured_side := s * 1.02 in
  let actual_area := s^2 in
  let calculated_area := measured_side^2 in
  let error := calculated_area - actual_area in
  let percentage_error := (error / actual_area) * 100 in
  percentage_error = 4.04 :=
sorry

end square_area_error_l41_41261


namespace perpendicular_diagonals_l41_41625

variable (A B C D O : Point)
variable (cyclic : CyclicQuad A B C D)
variable (center : CenterOfCircumcircle A B C D O)
variable (angle_equality : AngleEquality (AngleAtVertex O A B) (AngleAtVertex D A C))

theorem perpendicular_diagonals 
  (cyclic : CyclicQuad A B C D)
  (center : CenterOfCircumcircle A B C D O)
  (angle_equality : AngleEquality (AngleAtVertex O A B) (AngleAtVertex D A C)) :
  Perpendicular (Diagonal A C) (Diagonal B D) :=
sorry

end perpendicular_diagonals_l41_41625


namespace last_three_non_zero_digits_100_factorial_l41_41656

theorem last_three_non_zero_digits_100_factorial : (∃ n : ℕ, n = 100 ! / (10 ^ 24)) → (100! % 1000 = 976) :=
sorry

end last_three_non_zero_digits_100_factorial_l41_41656


namespace measure_angle_FXU_l41_41068

def triangle := {α β γ : ℝ // α + β + γ = 180}
def FOX : triangle := ⟨54, 63, 63, sorry⟩

def bisects (x y : ℝ) := x = y / 2

theorem measure_angle_FXU :
  ∀ (α β γ : ℝ) (h₁ : α = β) (h₂ : α + β + γ = 180) (h₃ : γ = 54) (h₄ : ∀ (ξ : ℝ), bisects ξ β),
  ξ = 31.5 :=
begin
  intros α β γ h₁ h₂ h₃ h₄ ξ,
  have h₄' := h₄ γ,
  rw h₁ at ⊢ h₄',
  sorry,
end

end measure_angle_FXU_l41_41068


namespace increase_average_by_runs_l41_41139

theorem increase_average_by_runs :
  let total_runs_10_matches : ℕ := 10 * 32
  let runs_scored_next_match : ℕ := 87
  let total_runs_11_matches : ℕ := total_runs_10_matches + runs_scored_next_match
  let new_average_11_matches : ℚ := total_runs_11_matches / 11
  let increased_average : ℚ := 32 + 5
  new_average_11_matches = increased_average :=
by
  sorry

end increase_average_by_runs_l41_41139


namespace min_value_expression_min_l41_41695

noncomputable def min_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1) / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem min_value_expression_min : 
  ∃ x : ℝ, min_value_expression x = 7 / 18 :=
begin
  sorry
end

end min_value_expression_min_l41_41695


namespace smallest_circle_equation_l41_41546

-- Definitions of the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- The statement of the problem
theorem smallest_circle_equation : ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ 
  A.1 = -3 ∧ A.2 = 0 ∧ B.1 = 3 ∧ B.2 = 0 ∧ ((x - 0)^2 + (y - 0)^2 = 9) :=
by
  sorry

end smallest_circle_equation_l41_41546


namespace vehicles_with_at_least_80_kmh_equal_50_l41_41269

variable (num_vehicles_80_to_89 : ℕ := 15)
variable (num_vehicles_90_to_99 : ℕ := 30)
variable (num_vehicles_100_to_109 : ℕ := 5)

theorem vehicles_with_at_least_80_kmh_equal_50 :
  num_vehicles_80_to_89 + num_vehicles_90_to_99 + num_vehicles_100_to_109 = 50 := by
  sorry

end vehicles_with_at_least_80_kmh_equal_50_l41_41269


namespace six_digit_numbers_with_zero_l41_41003

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l41_41003


namespace find_r_values_l41_41986

theorem find_r_values (r : ℝ) (h1 : r ≥ 8) (h2 : r ≤ 20) :
  16 ≤ (r - 4) ^ (3/2) ∧ (r - 4) ^ (3/2) ≤ 128 :=
by {
  sorry
}

end find_r_values_l41_41986


namespace maynard_dog_holes_l41_41509

open Real

theorem maynard_dog_holes (h_filled : ℝ) (h_unfilled : ℝ) (percent_filled : ℝ) 
  (percent_unfilled : ℝ) (total_holes : ℝ) :
  percent_filled = 0.75 →
  percent_unfilled = 0.25 →
  h_unfilled = 2 →
  h_filled = total_holes * percent_filled →
  total_holes = 8 :=
by
  intros hf pu hu hf_total
  sorry

end maynard_dog_holes_l41_41509


namespace num_diff_positive_integers_l41_41368

open Set

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

theorem num_diff_positive_integers : 
  ∃ n : ℕ, (∀ a b ∈ S, a ≠ b → ∃ k ∈ S, (a - b = k ∨ b - a = k)) ∧ n = 19 :=
sorry

end num_diff_positive_integers_l41_41368


namespace second_wing_floors_l41_41853

theorem second_wing_floors : 
  let first_wing_floors := 9
  let first_halls_per_floor := 6
  let first_rooms_per_hall := 32
  let second_halls_per_floor := 9
  let second_rooms_per_hall := 40
  let total_hotel_rooms := 4248
  let first_wing_rooms := first_wing_floors * first_halls_per_floor * first_rooms_per_hall
  let second_wing_rooms := total_hotel_rooms - first_wing_rooms
  let second_rooms_per_floor := second_halls_per_floor * second_rooms_per_hall
  second_wing_rooms / second_rooms_per_floor = 7 := 
by
  let first_wing_floors := 9
  let first_halls_per_floor := 6
  let first_rooms_per_hall := 32
  let second_halls_per_floor := 9
  let second_rooms_per_hall := 40
  let total_hotel_rooms := 4248
  let first_wing_rooms := first_wing_floors * first_halls_per_floor * first_rooms_per_hall
  let second_wing_rooms := total_hotel_rooms - first_wing_rooms
  let second_rooms_per_floor := second_halls_per_floor * second_rooms_per_hall
  have h1 : first_wing_rooms = 1728 := by norm_num
  have h2 : second_wing_rooms = 2520 := by norm_num
  have h3 : second_rooms_per_floor = 360 := by norm_num
  show second_wing_rooms / second_rooms_per_floor = 7, from by norm_num
  sorry

end second_wing_floors_l41_41853


namespace lucy_notebooks_l41_41508

theorem lucy_notebooks : 
  ∀ (total_money : ℕ) (notebook_cost : ℕ), 
  total_money = 2550 → 
  notebook_cost = 240 → 
  ∃ (max_notebooks : ℕ), 
  max_notebooks = 10 ∧ total_money ≥ notebook_cost * max_notebooks :=
by
  intros total_money notebook_cost ht hc
  use 10
  split
  { -- Proof that max_notebooks is 10
    exact rfl
  }
  { -- Proof that total_money ≥ notebook_cost * max_notebooks
    rw [ht, hc]
    exact nat.le_of_sub_eq_zero rfl
  }

end lucy_notebooks_l41_41508


namespace square_number_configuration_exists_l41_41812

/-- Define the grid and determine the positions of numbers 2 through 8 such that the 
    arrows point correctly as per the problem statement -/
theorem square_number_configuration_exists :
  ∃ (A B C D E F G : ℕ),
  {A, B, C, D, E, F, G} = {2, 3, 4, 5, 6, 7, 8} ∧
  (B = 2 ∧ E = 3 ∧ C = 4 ∧ D = 5 ∧ A = 6 ∧ G = 7 ∧ F = 8) :=
by {
  use [6, 2, 4, 5, 3, 8, 7],
  split,
  { simp, },
  { tauto, },
}

end square_number_configuration_exists_l41_41812


namespace sum_m_n_l41_41971

-- Define the probability P(x,y)
def probability (P : ℕ × ℕ → ℚ) (x y : ℕ) : ℚ :=
  if x = 0 ∨ y = 0 then 0
  else if x = 0 ∧ y = 0 then 1
  else ⅓ * (P (x-1, y) + P (x, y-1) + P (x-1, y-1))

-- Define the specific starting condition
def starting_probability : ℚ :=
let P := probability in
P (5, 5)

-- Define the desired form of the probability as m/3^n
def desired_form : Prop :=
∃ m n : ℕ, m ≠ 0 ∧ (∀ k : ℕ, ¬(k ∣ 3) ∨ k ∣ m) ∧ starting_probability = m / (3^n)

-- Define the final goal to find the sum
def final_goal : ℕ :=
let ⟨m, n, _, _, _⟩ := by solve_by_elim [desired_form] in m + n

-- Prove that the final goal is as expected.
theorem sum_m_n : final_goal = 1186 :=
sorry -- The proof is omitted here but it must show this calculation is correct.

end sum_m_n_l41_41971


namespace average_marks_two_classes_l41_41603

theorem average_marks_two_classes (n1 n2 : ℕ) (a1 a2 : ℕ) 
  (h1 : n1 = 30) (h2 : a1 = 40) (h3 : n2 = 50) (h4 : a2 = 60) : 
  (n1 * a1 + n2 * a2) / (n1 + n2) = 52.5 :=
by {
  sorry
}

end average_marks_two_classes_l41_41603


namespace tan_interval_strictly_increasing_l41_41297

open Real

def strictly_increasing (f : ℝ → ℝ) (I : set ℝ) := ∀ x y ∈ I, x < y → f x < f y

theorem tan_interval_strictly_increasing (k : ℤ) :
  strictly_increasing (λ x, tan (2 * x - π / 3))
  ({x : ℝ | (Int.to_real k * π / 2 - π / 12) < x ∧ x < (Int.to_real k * π / 2 + 5 * π / 12)} : set ℝ) :=
sorry

end tan_interval_strictly_increasing_l41_41297


namespace min_k_for_convex_quadrilateral_with_sides_l41_41321

theorem min_k_for_convex_quadrilateral_with_sides (n : ℕ) (h_n : n = 2007) : ∃ k, k = 1506 ∧
  ∀ (vertices : finset ℕ), vertices.card = k →
  ∃ (a b c d : ℕ), 
  {a, b, c, d}.subset vertices ∧ 
  ∃ (i j k l : fin n), 
  i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ 
  (i.1 + 1) % n = j.1 ∧ (j.1 + 1) % n = k.1 ∧ (k.1 + 1) % n = l.1 :=
begin
  sorry
end

end min_k_for_convex_quadrilateral_with_sides_l41_41321


namespace find_theta_area_triangle_ABC_l41_41332

-- Define the problem conditions
def is_interior_angle (θ : ℝ) : Prop := θ > 0 ∧ θ < π
def z (θ : ℝ) := Complex.cos θ + Complex.sin θ * Complex.I 

-- Problem (1): Find θ
theorem find_theta (θ : ℝ) (h : is_interior_angle θ) (hz : (z θ)^2 - z θ = Complex.I * (Complex.sin (2 * θ) - Complex.sin θ)) :
  θ = 2 * π / 3 :=
sorry

-- Problem (2): Find Area of Triangle ABC
theorem area_triangle_ABC (θ : ℝ) (hθ : θ = 2 * π / 3) :
  let A := (2 : ℂ) * (z θ),
      B := Complex.conj (z θ),
      C := (1 : ℂ) + (z θ) + (z θ)^2 in 
  let area := 1 / 2 * Complex.abs (2 * Complex.sin (θ)) in
  Complex.abs area = sqrt 3 / 2 :=
sorry

end find_theta_area_triangle_ABC_l41_41332


namespace quadratic_root_condition_l41_41436

theorem quadratic_root_condition (a : ℝ) :
  (4 * Real.sqrt 2) = 3 * Real.sqrt (3 - 2 * a) → a = 1 / 2 :=
by
  sorry

end quadratic_root_condition_l41_41436


namespace range_of_a_l41_41751

theorem range_of_a (a : ℝ) : (∀ x : ℝ, exp (-x) + a * x ≥ 0) → (0 ≤ a ∧ a ≤ Real.exp 1) := 
sorry

end range_of_a_l41_41751


namespace free_stick_l41_41706

-- Define the types for stick and loop
structure Stick :=
  (length : Nat)
  (rope_loop : bool)  -- Indicating if a rope loop is present at one end

structure JacketLoop :=
  (fabric : bool)  -- Indicating if the loop is carrying a piece of fabric

-- Initial conditions and operations
def pass_jacket_loop (j : JacketLoop) (s : Stick) : Stick :=
  { s with rope_loop := true }

def insert_stick (j : JacketLoop) (s : Stick) : Stick :=
  { s with length := s.length }  -- Inserting doesn't change the stick properties in this abstraction

def tighten_rope (s : Stick) : Stick :=
  { s with rope_loop := true }

def loosen_rope (s : Stick) : Stick :=
  { s with rope_loop := true }

def withdraw_jacket_loop (j : JacketLoop) (s : Stick) : Stick :=
  { s with rope_loop := false }

-- The theorem stating that the stick can be freed eventually
theorem free_stick (s : Stick) (j : JacketLoop) :
  (loosen_rope (insert_stick j (pass_jacket_loop j s))) = { s with rope_loop := false } :=
by sorry

end free_stick_l41_41706


namespace solve_abs_eq_abs_add_a_l41_41181

-- Define the function f and its domain
noncomputable def f (x : ℝ) : ℝ :=
|x^3 + x^2 - 4 * x - 4| / |(x - 1) * (x - 3) + 3 * x - 5|

-- Define the domain of f
def D (x : ℝ) : Prop := x ≠ 2 ∧ x ≠ -1

-- Define the proof problem
theorem solve_abs_eq_abs_add_a :
  (∀ x, D x → f x = |x + 2|) →
  (∀ x a, |x + 2| = |x| + a → a ∈ (-2, 0) ∨ a ∈ (0, 2)) :=
by
  intros hfx x a h_eq
  sorry

end solve_abs_eq_abs_add_a_l41_41181


namespace smallest_solution_for_quartic_eq_l41_41941

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end smallest_solution_for_quartic_eq_l41_41941


namespace differences_of_set_l41_41378

theorem differences_of_set {S : Finset ℕ} (hS : S = finset.range 21 \ {0}) :
  (Finset.card ((S.product S).image (λ p : ℕ × ℕ, p.1 - p.2)).filter (λ x, x > 0)) = 19 :=
by
  sorry

end differences_of_set_l41_41378


namespace both_teams_joint_renovation_team_renovation_split_l41_41215

-- Problem setup for part 1
def renovation_total_length : ℕ := 2400
def teamA_daily_progress : ℕ := 30
def teamB_daily_progress : ℕ := 50
def combined_days_to_complete_renovation : ℕ := 30

theorem both_teams_joint_renovation (x : ℕ) :
  (teamA_daily_progress + teamB_daily_progress) * x = renovation_total_length → 
  x = combined_days_to_complete_renovation :=
by
  sorry

-- Problem setup for part 2
def total_renovation_days : ℕ := 60
def length_renovated_by_teamA : ℕ := 900
def length_renovated_by_teamB : ℕ := 1500

theorem team_renovation_split (a b : ℕ) :
  a / teamA_daily_progress + b / teamB_daily_progress = total_renovation_days ∧ 
  a + b = renovation_total_length → 
  a = length_renovated_by_teamA ∧ b = length_renovated_by_teamB :=
by
  sorry

end both_teams_joint_renovation_team_renovation_split_l41_41215


namespace total_water_proof_l41_41925

def box1 := 20 * 0.8 * 8
def box2 := 15 * 0.7 * 10
def box3 := 25 * 0.6 * 12
def box4 := 5 * 0.5 * 5
def box5 := 10 * 0.9 * 15
def box6 := 30 * 0.55 * 7
def box7 := 12 * 3
def box8 := 10 * 0
def box9 := 18 * 0.75 * 9
def box10 := 8 * 0.95 * 10

def total_water := box1 + box2 + box3 + box4 + box5 + box6 + box7 + box8 + box9 + box10

theorem total_water_proof : total_water = 909.5 := by
  unfold total_water box1 box2 box3 box4 box5 box6 box7 box8 box9 box10
  norm_num
  sorry

end total_water_proof_l41_41925


namespace six_digit_numbers_with_zero_l41_41020

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41020


namespace quadruple_comp_odd_is_even_l41_41840

variable {α : Type} [AddGroup α] [HasSmul ℤ α]

def odd_function (g : α → α) : Prop :=
  ∀ x : α, g (-x) = -g x

theorem quadruple_comp_odd_is_even (g : α → α) (hg : odd_function g) :
  even_function (g ∘ g ∘ g ∘ g) :=
by 
  sorry

end quadruple_comp_odd_is_even_l41_41840


namespace largest_y_coordinate_l41_41666

theorem largest_y_coordinate : 
  ∀ (x y : ℝ),
    (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 := 
by 
  intros x y h 
  rcases eq_of_pow_eq_pow (square_eq_zero (by linarith [mul_eq_zero.mp h]))
  sorry

end largest_y_coordinate_l41_41666


namespace max_n_compact_clique_l41_41111

open Finset

/-- n-compactness definition: A graph G is n-compact if for any vertex v, there exists a set of
    exactly n vertices such that all pairs in this set are adjacent (form a clique). -/
def n_compact (G : SimpleGraph ℕ) (n : ℕ) := 
  ∀ v, ∃ t : Finset ℕ, t.card = n ∧ (∀ a b ∈ t, G.adj a b)

/-- Main theorem stating that for a given natural number n ≥ 2, the maximum number N such that any 
    n-compact group with N people contains a subgroup of n+1 people, each of whom is familiar with 
    each other, equals 2n + 1. -/
theorem max_n_compact_clique {n : ℕ} (h : n ≥ 2) :
  ∃ (N : ℕ), (∀ (G : SimpleGraph ℕ), (n_compact G n) →
    ∃ (s : Finset ℕ), s.card = n+1 ∧ (∀ a b ∈ s, G.adj a b)) ↔ N = 2 * n + 1 :=
sorry

end max_n_compact_clique_l41_41111


namespace parallel_EF_BC_l41_41996

noncomputable theory

variables (A B C D K E F E' F' O : Type)
variable [has_coe_to_fun B (λ _, has_coe_to_fun.to_fun C)]

open_locale classical

/-- Given an acute triangle ABC (AB > AC) with circumcircle O,
point D on side AB such that DA = DC. A line through D parallel to BO
intersects OA at K. A perpendicular line from K to AC intersects O at E and F.
Extending ED and FD intersects O again at E' and F'. We prove E'F' is parallel to BC. -/
theorem parallel_EF_BC :
  ∀ (A B C D K E F E' F' O : Type)
  (h0: acute_triangle A B C)
  (h1: triangle_side_len_gt A B C)
  (h2: circumcircle A B C O)
  (h3: point_on_side D AB)
  (h4: len_equals DA DC)
  (h5: line_parallel_through_point D BO intersects OA K)
  (h6: perp_line_intersects_circ K AC E F intersects O)
  (h7: extended_line_intersects_circ_again ED O E')
  (h8: extended_line_intersects_circ_again FD O F'),
  parallel E'F' BC := 
begin
  sorry
end

end parallel_EF_BC_l41_41996


namespace weight_of_8_moles_CCl4_correct_l41_41764

/-- The problem states that carbon tetrachloride (CCl4) is given, and we are to determine the weight of 8 moles of CCl4 based on its molar mass calculations. -/
noncomputable def weight_of_8_moles_CCl4 (molar_mass_C : ℝ) (molar_mass_Cl : ℝ) : ℝ :=
  let molar_mass_CCl4 := molar_mass_C + 4 * molar_mass_Cl
  8 * molar_mass_CCl4

/-- Given the molar masses of Carbon (C) and Chlorine (Cl), prove that the calculated weight of 8 moles of CCl4 matches the expected weight. -/
theorem weight_of_8_moles_CCl4_correct :
  let molar_mass_C := 12.01
  let molar_mass_Cl := 35.45
  weight_of_8_moles_CCl4 molar_mass_C molar_mass_Cl = 1230.48 := by
  sorry

end weight_of_8_moles_CCl4_correct_l41_41764


namespace ratio_platinum_to_gold_l41_41879

variables (G P : ℝ)

-- Define the conditions
def gold_balance := G / 3
def platinum_balance := P / 6
def new_platinum_balance := platinum_balance + gold_balance

-- The condition that after transfer, 2/3 of P remains unspent
def condition := new_platinum_balance = P / 3

-- The theorem we want to prove
theorem ratio_platinum_to_gold (h : condition G P) : P = 2 * G :=
by sorry

end ratio_platinum_to_gold_l41_41879


namespace minimum_area_quadrilateral_l41_41319

open Real

noncomputable theory

def parabola := {p : ℝ × ℝ | p.snd = p.fst^2}

theorem minimum_area_quadrilateral
  (F : ℝ × ℝ := (0, 1/4))
  (O : ℝ × ℝ := (0, 0))
  (B D : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hB : B ∈ parabola)
  (hD : D ∈ parabola)
  (hLine : ∃ m : ℝ, B.snd = m * B.fst + 2 ∧ D.snd = m * D.fst + 2)
  (hSym : C = (D.fst, 4 - D.snd)) :
  let quadrilateral_area := 
    abs (O.1 * B.2 + B.1 * D.2 + D.1 * C.2 + C.1 * O.2 
        - O.2 * B.1 - B.2 * D.1 - D.2 * C.1 - C.2 * O.1) / 2 in
  quadrilateral_area = 3 :=
sorry

end minimum_area_quadrilateral_l41_41319


namespace midpoint_equidistant_semiangles_l41_41121

theorem midpoint_equidistant_semiangles (A B C A_1 B_1 C_1 : Point) (a b c : ℝ)
  (h_mid_AB : is_midpoint A_1 B C)
  (h_mid_BC : is_midpoint B_1 C A)
  (h_mid_CA : is_midpoint C_1 A B)
  (outward_semicircle_b : semicircle_outward_on_side B C)
  (outward_semicircle_c : semicircle_outward_on_side A B) :
  ∃ B_star C_star : Point, 
  (dist A_1 B_star = dist A_1 C_star) ∧ 
  (angle A_1 B_star C_star = 90) ∧
  (∀ (semicircle_type : semicircle), 
    (semicircle_type = semicircle_outward_on_side ∨ semicircle_type = semicircle_inward_on_side) → 
    (angle A_1 (midpoint_of_semicircle semicircle_type A B) 
                (midpoint_of_semicircle semicircle_type B C) = 90)) :=
begin
  sorry
end

end midpoint_equidistant_semiangles_l41_41121


namespace laborers_percentage_rounded_nearest_tenth_l41_41207

theorem laborers_percentage_rounded_nearest_tenth :
  ∀ (total_laborers present_laborers : ℕ), 
  total_laborers = 56 → 
  present_laborers = 30 → 
  (Float.ceil ((present_laborers.toFloat / total_laborers.toFloat * 100) * 10) / 10) = 53.6 :=
begin
  intros,
  sorry
end

end laborers_percentage_rounded_nearest_tenth_l41_41207


namespace arithmetic_series_sum_l41_41663

theorem arithmetic_series_sum:
  let a₁ := 24
  let a_n := 48
  let d := 0.2
  let n := (48 - 24) / d + 1
  let S := n * (a₁ + a_n) / 2
  S = 4356 :=
by
  unfold a₁ a_n d n S
  sorry

end arithmetic_series_sum_l41_41663


namespace num_unique_differences_l41_41397

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41397


namespace find_S24_l41_41053

-- Setting up the definitions for Sn based on a geometric sequence \(a_n\)
def S (n : ℕ) : ℚ := sorry -- This would be the sum of the first n terms of the geometric sequence, to be defined formally

-- Conditions given
def S6 := 48 : ℚ
def S12 := 60 : ℚ

-- The statement to be proved
theorem find_S24 (h1 : S 6 = S6) (h2 : S 12 = S12) : S 24 = 255 / 4 := 
sorry

end find_S24_l41_41053


namespace average_root_cross_sectional_area_average_volume_total_volume_estimation_l41_41989

-- Define the given data
def n : ℕ := 10
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x2 : ℝ := 0.038
def sum_y2 : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186
def sqrt_1_896 : ℝ := 1.377

-- Define average values
def average_x : ℝ := 0.06
def average_y : ℝ := 0.39

theorem average_root_cross_sectional_area (n ≠ 0) : 
  (sum_x / n = average_x) :=
by
  sorry

theorem average_volume (n ≠ 0) : 
  (sum_y / n = average_y) :=
by
  sorry

-- Define correlation coefficient calculation
def correlation_coefficient (n ≠ 0) : 
  ((sum_xy - n * average_x * average_y) /
  (sqrt (sum_x2 - n * average_x^2) * sqrt (sum_y2 - n * average_y^2)) ≈ 0.97) :=
by
  sorry

-- Define the total volume estimation
theorem total_volume_estimation :
  (total_root_area * average_y / average_x = 1209) :=
by
  sorry

end average_root_cross_sectional_area_average_volume_total_volume_estimation_l41_41989


namespace at_least_one_shared_birthday_l41_41583

theorem at_least_one_shared_birthday (trainees : ℕ) (days : ℕ) (h_trainees : trainees = 62) (h_days : days = 365) :
  let P_no_shared := (365.factorial / (365 - 62).factorial : ℝ) / (365 ^ 62) in
  let P_at_least_one_shared : ℝ := 1 - P_no_shared in
  P_at_least_one_shared ≈ 0.9959095749 := sorry

end at_least_one_shared_birthday_l41_41583


namespace steak_burned_portion_l41_41486

-- Define the conditions as Lean variables and expressions
variable (B : ℝ)  -- B represents the portion of the steak that got burned, in ounces
variable (total_steak : ℝ) := 30  -- The steak was originally 30 ounces
variable (john_ate : ℝ) := 12  -- John ate 12 ounces
variable (eaten_percentage : ℝ) := 0.8  -- John ate 80% of what isn't burned

-- The main theorem using the conditions to prove B = 15
theorem steak_burned_portion (h : eaten_percentage * (total_steak - B) = john_ate) : B = 15 := 
by sorry

end steak_burned_portion_l41_41486


namespace six_digit_numbers_with_zero_l41_41013

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l41_41013


namespace mark_all_points_on_segment_l41_41867

-- Define the necessary conditions and initial setup for the Lean theorem
noncomputable def segment := finset.Icc 0 2002
axiom (n : ℕ) (hn : n > 1) (A : finset ℕ) :
  A ⊆ segment ∧ A.card = n - 1 ∧
  ∀ (i j : ℕ), i ≠ j → nat.coprime (A.nth_le i sorry) (A.nth_le j sorry)

-- The main theorem to prove
theorem mark_all_points_on_segment : ∀ (x : ℕ), x ∈ segment :=
sorry

end mark_all_points_on_segment_l41_41867


namespace difference_of_distinct_members_l41_41415

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41415


namespace desks_array_unique_desks_array_count_l41_41626

theorem desks_array_unique (n m : ℕ) (h : n * m = 49) (h_n : n ≥ 2) (h_m : m ≥ 2) : n = 7 ∧ m = 7 :=
by sorry

theorem desks_array_count : ∃! (n m : ℕ), n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 :=
begin
  have h := desks_array_unique,
  use (7, 7),
  split,
  { split,
    { exact (7 * 7 = 49).symm },
    { split; norm_num, }},
  { rintro ⟨n, m⟩ ⟨h₁, h₂, h₃⟩,
    exact (desks_array_unique n m h₁ h₂ h₃) }
end

end desks_array_unique_desks_array_count_l41_41626


namespace number_of_auspicious_three_digit_numbers_l41_41556

-- Definition: sum of the digits of a number n
def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- Definition: "auspicious number"
def is_auspicious (n : ℕ) : Prop :=
  n % 6 = 0 ∧ sum_of_digits n = 6 ∧ 100 ≤ n ∧ n < 1000

-- Proof statement: there are exactly 12 auspicious numbers between 100 and 999
theorem number_of_auspicious_three_digit_numbers : 
  Finset.card (Finset.filter is_auspicious (Finset.range 1000 \ Finset.range 100)) = 12 := 
by
  sorry

end number_of_auspicious_three_digit_numbers_l41_41556


namespace interest_rate_correct_l41_41865

noncomputable def annual_interest_rate : ℝ :=
  4^(1/10) - 1

theorem interest_rate_correct (P A₁₀ A₁₅ : ℝ) (h₁ : P = 6000) (h₂ : A₁₀ = 24000) (h₃ : A₁₅ = 48000) :
  (P * (1 + annual_interest_rate)^10 = A₁₀) ∧ (P * (1 + annual_interest_rate)^15 = A₁₅) :=
by
  sorry

end interest_rate_correct_l41_41865


namespace highest_score_is_103_l41_41795

/-- Define the base score -/
def base_score : ℕ := 100

/-- Define the score adjustments for the three students -/
def adjustments : List ℤ := [3, -8, 0]

/-- Define a function to calculate the actual score of a student -/
def calculate_score (base : ℕ) (adjustment : ℤ) : ℕ :=
  natAbs (base + adjustment)

/-- The highest score among the three students is 103 -/
theorem highest_score_is_103 : 
  let scores := adjustments.map (calculate_score base_score)
  in list.maximum scores = 103 :=
by
  sorry

end highest_score_is_103_l41_41795


namespace fourth_term_sum_of_powers_of_4_is_85_l41_41665

theorem fourth_term_sum_of_powers_of_4_is_85 : (4^0 + 4^1 + 4^2 + 4^3) = 85 :=
by
  -- Definitions based on the conditions
  have term0 : 4^0 = 1 := by norm_num
  have term1 : 4^1 = 4 := by norm_num
  have term2 : 4^2 = 16 := by norm_num
  have term3 : 4^3 = 64 := by norm_num

  -- Summing these values
  calc
    (4^0 + 4^1 + 4^2 + 4^3)
    = 1 + 4 + 16 + 64 : by rw [term0, term1, term2, term3]
    _ = 85 : by norm_num

end fourth_term_sum_of_powers_of_4_is_85_l41_41665


namespace number_of_lines_intersecting_parabola_at_one_point_l41_41620

theorem number_of_lines_intersecting_parabola_at_one_point 
  (p : ℝ) (hp : p > 0) (M : ℝ × ℝ) (hM : ¬ (M ∈ {P : ℝ × ℝ | P.2^2 = 2 * p * P.1})) :
  ∃! lines : set (ℝ × ℝ → ℝ), lines.card = 3 ∧
    ∀ line ∈ lines, ∃! P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.2^2 = 2 * p * Q.1} ∧ line P = 0 :=
sorry

end number_of_lines_intersecting_parabola_at_one_point_l41_41620


namespace inequality_abc_l41_41325

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_min : min (min (a * b) (b * c)) (c * a) ≥ 1) :
    real.cbrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) <= ((a + b + c) / 3)^2 + 1 := 
by
  sorry

end inequality_abc_l41_41325


namespace sides_of_triangle_l41_41673

theorem sides_of_triangle (a b c t : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (ht : t ∈ set.Icc (2/3 : ℝ) 2) :
  (a^2 + b * c * t) + (b^2 + c * a * t) > (c^2 + a * b * t) ∧
  (a^2 + b * c * t) + (c^2 + a * b * t) > (b^2 + c * a * t) ∧
  (b^2 + c * a * t) + (c^2 + a * b * t) > (a^2 + b * c * t) := sorry

end sides_of_triangle_l41_41673


namespace application_methods_count_l41_41929

theorem application_methods_count (students universities : ℕ) (A B C : ℕ) (h1 : students = 5) (h2 : universities = 3) (h3 : A ≥ 1) (h4 : B ≥ 1) (h5 : C ≥ 1) :
  (C(students, 2) * C(students - 2, 2) * A + C(students, 3) * B * C) = 240 :=
by 
  -- To be proved
  sorry

end application_methods_count_l41_41929


namespace limit_integral_l41_41081

noncomputable def gn (n : ℕ) (x : ℝ) : ℝ :=
  n - 2 * n * n * |x - 1 / 2| + |n - 2 * n * n * |x - 1 / 2||

theorem limit_integral (f : ℝ → ℝ) [is_poly : is_polynomial f] :
  (∃ g : ℕ → ℝ → ℝ, ∀ n x, g n x = gn n x) →
  (tendsto (fun n => ∫ x in (0:ℝ)..1, f x * gn n x) at_top (𝓝 (f (1 / 2)))) :=
begin
  sorry
end

end limit_integral_l41_41081


namespace original_volume_of_cube_is_8_l41_41869

theorem original_volume_of_cube_is_8 (a : ℕ) (h : ((a - 2) * a * (a + 2) = a^3 - 4*a) ∧ (a^3 - (a^3 - 4*a)) = 8) : a^3 = 8 :=
begin
  -- conditions: 
  -- One dimension increased by 2, another decreased by 2, third unchanged: (a - 2, a, a + 2)
  -- Volume of the resulting solid: V_res = (a - 2) * a * (a + 2) = a^3 - 4a
  -- Volume difference is 8: a^3 - (a^3 - 4a) = 8
  sorry
end

end original_volume_of_cube_is_8_l41_41869


namespace count_two_digit_numbers_l41_41431

theorem count_two_digit_numbers : 
  ∃ n, n = 45 ∧ ∀ t u : ℕ, (1 ≤ t ∧ t ≤ 9) → (0 ≤ u ∧ u ≤ 9) → (t > u) → 
  n = (finrange 1 10).sum (λ t, 9 - t) := 
begin
  sorry
end

end count_two_digit_numbers_l41_41431


namespace num_unique_differences_l41_41393

theorem num_unique_differences : 
  ∃ n : ℕ, (∀ a b ∈ ({1, 2, 3, ..., 20} : finset ℕ), a ≠ b → (a - b).nat_abs ∈ ({1, 2, 3, ..., 19} : finset ℕ)) ∧ n = 19 :=
sorry

end num_unique_differences_l41_41393


namespace irrational_number_is_sqrt3_l41_41641

theorem irrational_number_is_sqrt3 :
  ¬∃ (a b : ℤ), b ≠ 0 ∧ (√3 = a / b) ∧ 
  (∃ (c d : ℤ), d ≠ 0 ∧ (-2 = c / d)) ∧
  (∃ (e f : ℤ), f ≠ 0 ∧ (1 / 2 = e / f)) ∧
  (∃ (g h : ℤ), h ≠ 0 ∧ (2 = g / h)) :=
by {
  sorry
}

end irrational_number_is_sqrt3_l41_41641


namespace color_crafter_secret_codes_l41_41794

theorem color_crafter_secret_codes :
  8^5 = 32768 := by
  sorry

end color_crafter_secret_codes_l41_41794


namespace adam_deleted_items_l41_41249

theorem adam_deleted_items (i l d : ℕ) (h_initial : i = 18) (h_left : l = 8) (h_deleted : i - l = d) : d = 10 :=
by
  rw [h_initial, h_left, h_deleted]
  rfl

end adam_deleted_items_l41_41249


namespace angle_PQC_in_triangle_l41_41481

theorem angle_PQC_in_triangle 
  (A B C P Q: ℝ)
  (h_in_triangle: A + B + C = 180)
  (angle_B_exterior_bisector: ∀ B_ext, B_ext = 180 - B →  angle_B = 90 - B / 2)
  (angle_C_exterior_bisector: ∀ C_ext, C_ext = 180 - C →  angle_C = 90 - C / 2)
  (h_PQ_BC_angle: ∀ PQ_angle BC_angle, PQ_angle = 30 → BC_angle = 30) :
  ∃ PQC_angle, PQC_angle = (180 - A) / 2 :=
by
  sorry

end angle_PQC_in_triangle_l41_41481


namespace trigonometric_identity_l41_41444

open Real

noncomputable def acute (x : ℝ) := 0 < x ∧ x < π / 2

theorem trigonometric_identity 
  {α β : ℝ} (hα : acute α) (hβ : acute β) (h : cos α > sin β) :
  α + β < π / 2 :=
sorry

end trigonometric_identity_l41_41444


namespace harmonic_series_inequality_l41_41119

theorem harmonic_series_inequality (n : ℕ) (h : n ≥ 2) :
  n * (Real.root n (n + 1) - 1) < (∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k) ∧
  (∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k) < n * (1 - Real.root n n) + 1 :=
sorry

end harmonic_series_inequality_l41_41119


namespace cost_per_minute_of_each_call_l41_41998

theorem cost_per_minute_of_each_call :
  let calls_per_week := 50
  let hours_per_call := 1
  let weeks_per_month := 4
  let total_hours_in_month := calls_per_week * hours_per_call * weeks_per_month
  let total_cost := 600
  let cost_per_hour := total_cost / total_hours_in_month
  let minutes_per_hour := 60
  let cost_per_minute := cost_per_hour / minutes_per_hour
  cost_per_minute = 0.05 := 
by
  sorry

end cost_per_minute_of_each_call_l41_41998


namespace range_of_a_l41_41779

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) ↔ 1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l41_41779


namespace find_area_of_big_triangle_l41_41574

noncomputable def area_of_triangle (a1 a2 a3 : ℝ) : ℝ :=
let s1 := real.sqrt a1, s2 := real.sqrt a2, s3 := real.sqrt a3 in
let h := s1 + s2 + s3 in let b := s1 + s2 + s3 in
(1 / 2) * b * h

theorem find_area_of_big_triangle :
  area_of_triangle 4 9 36  = 60.5 := 
begin
  sorry
end

end find_area_of_big_triangle_l41_41574


namespace difference_of_distinct_members_l41_41414

theorem difference_of_distinct_members :
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  ∃ n, ∀ x, x ∈ D → x = n :=
by
  let S := {i | ∃ n, i = n ∧ 1 ≤ n ∧ n ≤ 20}
  let D := { |a - b| | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  existsi 19
  sorry

end difference_of_distinct_members_l41_41414


namespace converse_negative_proposition_l41_41897

theorem converse_negative_proposition (x : ℝ) : 
  (x ≥ 1 → 2^x + 1 ≥ 3) ↔ (2^x + 1 < 3 → x < 1) :=
sorry

end converse_negative_proposition_l41_41897


namespace lifting_equivalence_l41_41892

theorem lifting_equivalence : 
  let original_weight_total := 2 * 30 * 10 in
  let new_weight := 20 in
  let repetitions := 15 in
  2 * new_weight * repetitions = original_weight_total :=
by 
  let original_weight_total := 2 * 30 * 10
  let new_weight := 20
  let repetitions := 15
  have h1 : 2 * new_weight * repetitions = 2 * 20 * 15 := by rfl
  have h2 : 2 * 20 * 15 = 600 := by norm_num
  have h3 : 2 * 30 * 10 = 600 := by norm_num
  show 2 * new_weight * repetitions = original_weight_total,
  from eq.trans h1 (eq.trans h2 h3)

end lifting_equivalence_l41_41892


namespace area_of_quadrilateral_l41_41206

theorem area_of_quadrilateral (d : ℝ) (h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  (1 / 2 * d * h1) + (1 / 2 * d * h2) = 300 :=
by
  rw [hd, hh1, hh2]
  norm_num
  sorry

end area_of_quadrilateral_l41_41206


namespace num_lines_satisfying_conditions_l41_41734

-- Define the entities line, angle, and perpendicularity in a geometric framework
variable (Point Line : Type)
variable (P : Point)
variable (a b l : Line)

-- Define geometrical predicates
variable (Perpendicular : Line → Line → Prop)
variable (Passes_Through : Line → Point → Prop)
variable (Forms_Angle : Line → Line → ℝ → Prop)

-- Given conditions
axiom perp_ab : Perpendicular a b
axiom passes_through_P : Passes_Through l P
axiom angle_la_30 : Forms_Angle l a (30 : ℝ)
axiom angle_lb_90 : Forms_Angle l b (90 : ℝ)

-- The statement to prove
theorem num_lines_satisfying_conditions : ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
  Passes_Through l1 P ∧ Forms_Angle l1 a (30 : ℝ) ∧ Forms_Angle l1 b (90 : ℝ) ∧
  Passes_Through l2 P ∧ Forms_Angle l2 a (30 : ℝ) ∧ Forms_Angle l2 b (90 : ℝ) ∧
  (∀ l', Passes_Through l' P ∧ Forms_Angle l' a (30 : ℝ) ∧ Forms_Angle l' b (90 : ℝ) → l' = l1 ∨ l' = l2) := sorry

end num_lines_satisfying_conditions_l41_41734


namespace cost_of_fencing_105_rupees_l41_41161

-- Define the problem conditions 
variables (x : ℝ)
variables (cost_per_meter_paise : ℝ) (field_area : ℝ) (side_ratio : ℝ × ℝ)

def cost_of_fencing (cost_per_meter_paise : ℝ) (field_area : ℝ) (side_ratio : ℝ × ℝ) : ℝ :=
  let (r1, r2) := side_ratio,
      x := real.sqrt (field_area / (r1 * r2)),
      length := r1 * x,
      width := r2 * x,
      perimeter := 2 * (length + width),
      cost_per_meter := cost_per_meter_paise / 100 in
  perimeter * cost_per_meter

-- Create a theorem to express the final problem solution
theorem cost_of_fencing_105_rupees : 
  cost_of_fencing 25 10800 (3, 4) = 105 :=
by
  sorry

end cost_of_fencing_105_rupees_l41_41161


namespace problem1_theorem_problem2_theorem_l41_41725

noncomputable def problem1 (α β γ a c : ℝ) (h1: α + β + γ = π) (h2 : α < π / 2) (h3 : β < π / 2) (h4 : γ < π / 2) 
  (h₀ : (cos β / b) + (cos γ / c) = sin α / (sqrt 3 * sin γ)) : ℝ :=
  √3

theorem problem1_theorem (α β γ a c : ℝ) (h1: α + β + γ = π) (h2 : α < π / 2) (h3 : β < π / 2) (h4 : γ < π / 2) 
  (h₀ : (cos β / b) + (cos γ / c) = sin α / (sqrt 3 * sin γ)) : b = √3 :=
sorry

noncomputable def problem2 (α β γ : ℝ) (h1: α + β + γ = π) 
  (h_cos_sin : cos β + sqrt 3 * sin β = 2) : set ℝ :=
  Ioo (3 + sqrt 3) (3 * sqrt 3)

theorem problem2_theorem (α β γ : ℝ) (h1: α + β + γ = π)
  (h_cos_sin : cos β + sqrt 3 * sin β = 2) (a c : ℝ) 
  (ha : a = 2 * sin α) (hc : c = 2 * sin γ) : 
  let L := a + b + c in
  L ∈ Ioo (3 + sqrt 3) (3 * sqrt 3) :=
sorry

end problem1_theorem_problem2_theorem_l41_41725


namespace soft_lenses_more_than_hard_l41_41652

-- Define the problem conditions as Lean definitions
def total_sales (S H : ℕ) : Prop := 150 * S + 85 * H = 1455
def total_pairs (S H : ℕ) : Prop := S + H = 11

-- The theorem we need to prove
theorem soft_lenses_more_than_hard (S H : ℕ) (h1 : total_sales S H) (h2 : total_pairs S H) : S - H = 5 :=
by
  sorry

end soft_lenses_more_than_hard_l41_41652


namespace least_common_multiple_l41_41905

open Int

theorem least_common_multiple {a b c : ℕ} 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : Nat.lcm a c = 90 := 
sorry

end least_common_multiple_l41_41905


namespace time_to_meet_l41_41519

variables {v_g : ℝ} -- speed of Petya and Vasya on the dirt road
variable {v_a : ℝ} -- speed of Petya on the paved road
variable {t : ℝ} -- time from start until Petya and Vasya meet
variable {x : ℝ} -- distance from the starting point to the bridge

-- Conditions
axiom Petya_speed : v_a = 3 * v_g
axiom Petya_bridge_time : x / v_a = 1
axiom Vasya_speed : v_a ≠ 0 ∧ v_g ≠ 0

-- Statement
theorem time_to_meet (h1 : v_a = 3 * v_g) (h2 : x / v_a = 1) (h3 : v_a ≠ 0 ∧ v_g ≠ 0) : t = 2 :=
by
  have h4 : x = 3 * v_g := sorry
  have h5 : (2 * x - 2 * v_g) / (2 * v_g) = 1 := sorry
  exact sorry

end time_to_meet_l41_41519


namespace find_c_in_triangle_l41_41455

theorem find_c_in_triangle
  (A : Real) (a b S : Real) (c : Real)
  (hA : A = 60) 
  (ha : a = 6 * Real.sqrt 3)
  (hb : b = 12)
  (hS : S = 18 * Real.sqrt 3) :
  c = 6 := by
  sorry

end find_c_in_triangle_l41_41455
