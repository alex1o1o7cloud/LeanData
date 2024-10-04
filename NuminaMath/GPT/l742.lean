import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.RamseyTheory
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Int.Basic
import Mathlib.Init.Data.Nat.Lemmas
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace sum_products_divisible_by_7_l742_742667

def is_valid_triplet (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 6

def T : set (ℕ × ℕ × ℕ) :=
  { triplet | ∃ a b c, triplet = (a, b, c) ∧ is_valid_triplet a b c }

def product_triplet (triplet : ℕ × ℕ × ℕ) : ℕ :=
  match triplet with
  | (a, b, c) => a * b * c

def sum_products (S : set (ℕ × ℕ × ℕ)) : ℕ :=
  S.to_finset.sum product_triplet

theorem sum_products_divisible_by_7 :
  sum_products T % 7 = 0 :=
by
  sorry

end sum_products_divisible_by_7_l742_742667


namespace vectors_coplanar_proof_vectors_coplanar_l742_742952

def a : ℝ × ℝ × ℝ := (2, -1, 3)
def b : ℝ × ℝ × ℝ := (-1, 4, 2)
def c (λ : ℝ) : ℝ × ℝ × ℝ := (-3, 5, λ)

theorem vectors_coplanar (λ : ℝ) : Prop :=
  let x := -1
  let y := 1
  let ax := 2 * x - y
  let bx := -x + 4 * y
  let cx := 3 * x + 2 * y
    ax = -3 ∧ bx = 5 ∧ cx = λ → λ = -1

-- Placeholder to skip the proof.
theorem proof_vectors_coplanar : vectors_coplanar -1 :=
  by
    sorry

end vectors_coplanar_proof_vectors_coplanar_l742_742952


namespace jackson_tile_cost_l742_742658

theorem jackson_tile_cost
  (rectangle_length rectangle_width : ℝ)
  (triangle_base triangle_height : ℝ)
  (circle_radius : ℝ)
  (green_tiles_ratio red_tiles_ratio blue_tiles_ratio : ℝ)
  (green_tile_cost red_tile_cost blue_tile_cost : ℝ) :
  rectangle_length = 10 →
  rectangle_width = 25 →
  triangle_base = 15 →
  triangle_height = 20 →
  circle_radius = 5 →
  green_tiles_ratio = 5 →
  red_tiles_ratio = 3 →
  blue_tiles_ratio = 2 →
  green_tile_cost = 3 →
  red_tile_cost = 1.5 →
  blue_tile_cost = 2.5 →
  let area_rectangle := rectangle_length * rectangle_width,
      area_triangle := 1 / 2 * triangle_base * triangle_height,
      area_circle := Float.pi * circle_radius ^ 2,
      total_area := area_rectangle + area_triangle + area_circle,
      sets := (Float.ceil (total_area / 10)),
      green_tiles := sets * green_tiles_ratio,
      red_tiles := sets * red_tiles_ratio,
      blue_tiles := sets * blue_tiles_ratio,
      total_cost := green_tiles * green_tile_cost + red_tiles * red_tile_cost + blue_tiles * blue_tile_cost in
  total_cost = 1176 := sorry

end jackson_tile_cost_l742_742658


namespace handshakes_count_l742_742512

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end handshakes_count_l742_742512


namespace converse_of_a_squared_plus_b_squared_zero_l742_742741

open Classical

variable (a b : ℝ)

theorem converse_of_a_squared_plus_b_squared_zero :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
begin
  sorry
end

end converse_of_a_squared_plus_b_squared_zero_l742_742741


namespace mid_face_area_quarter_oblique_face_area_l742_742864

-- Define the right triangular pyramid and its properties
structure RightTriangularPyramid :=
  (a b c : ℝ) -- edge lengths
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (mutually_perpendicular : ∀ (x y z : ℝ), x ≠ y → y ≠ z → x ≠ z → (x = a ∨ x = b ∨ x = c) → (y = a ∨ y = b ∨ y = c) → (z = a ∨ z = b ∨ z = c) → x = a ∧ y = b ∧ z = c → a^2 + b^2 = c^2)

-- Define a function to calculate the area of the oblique face
noncomputable def oblique_face_area (P : RightTriangularPyramid) : ℝ :=
  let s := (P.a + P.b + P.c) / 2 in -- semi-perimeter
  (s * (s - P.a) * (s - P.b) * (s - P.c)).sqrt

-- Define a function to calculate the area of the mid-face of the oblique face
noncomputable def mid_face_area (P : RightTriangularPyramid) : ℝ :=
  (oblique_face_area P) / 4

-- The theorem that states the property of the right triangular pyramid
theorem mid_face_area_quarter_oblique_face_area (P : RightTriangularPyramid) :
  mid_face_area P = oblique_face_area P / 4 := by
  sorry

end mid_face_area_quarter_oblique_face_area_l742_742864


namespace mod_inverse_7_800_l742_742797

theorem mod_inverse_7_800 : ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ 7 * x % 800 = 1 :=
by
  use 343
  split
  · show 0 ≤ 343
  · show 343 < 800
  · show 7 * 343 % 800 = 1
  sorry

end mod_inverse_7_800_l742_742797


namespace negation_of_existence_l742_742756

theorem negation_of_existence (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ ∀ x : ℝ, x^2 - a*x + 1 ≥ 0 :=
by
  intro h
  apply classical.not_exists
  intro h2
  exact h h2
  sorry

end negation_of_existence_l742_742756


namespace range_of_m_l742_742633

def has_solution_in_interval (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), x^2 - 2 * x - 1 + m ≤ 0 

theorem range_of_m (m : ℝ) : has_solution_in_interval m ↔ m ≤ 2 := by 
  sorry

end range_of_m_l742_742633


namespace ellipse_foci_distance_l742_742867

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → distance_between_foci a b = 3 * Real.sqrt 3 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l742_742867


namespace sum_binom_solutions_l742_742393

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742393


namespace total_students_l742_742652

theorem total_students (m d : ℕ) 
  (H1: 30 < m + d ∧ m + d < 40)
  (H2: ∃ r, r = 3 * m ∧ r = 5 * d) : 
  m + d = 32 := 
by
  sorry

end total_students_l742_742652


namespace final_price_correct_l742_742433

variable (original_price first_discount second_discount third_discount sales_tax : ℝ)
variable (final_discounted_price final_price: ℝ)

-- Define original price and discounts
def initial_price : ℝ := 20000
def discount1      : ℝ := 0.12
def discount2      : ℝ := 0.10
def discount3      : ℝ := 0.05
def tax_rate       : ℝ := 0.08

def price_after_first_discount : ℝ := initial_price * (1 - discount1)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - discount2)
def price_after_third_discount : ℝ := price_after_second_discount * (1 - discount3)
def final_sale_price : ℝ := price_after_third_discount * (1 + tax_rate)

-- Prove final sale price is 16251.84
theorem final_price_correct : final_sale_price = 16251.84 := by
  sorry

end final_price_correct_l742_742433


namespace find_m_l742_742788

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def set_T : Finset ℕ := Finset.range 16 \ {0}

def triplets_summing_to_18 (T : Finset ℕ) : Finset (ℕ × ℕ × ℕ) :=
  (T.product (T.product T)).filter (λ x, x.1 < x.2.1 ∧ x.2.1 < x.2.2 ∧ x.1 + x.2.1 + x.2.2 = 18)

def count_triplets_summing_to_18 (T : Finset ℕ) : ℕ :=
  (triplets_summing_to_18 T).card

def initial_probability (T : Finset ℕ) : ℚ :=
  count_triplets_summing_to_18 T / binomial T.card 3

def remove_m (T : Finset ℕ) (m : ℕ) : Finset ℕ :=
  T.erase m

theorem find_m (m : ℕ) (h : m ∈ set_T) :
  initial_probability set_T < initial_probability (remove_m set_T m) := by
  sorry

example : find_m 4 (by decide) :=
begin
  sorry
end

end find_m_l742_742788


namespace B_can_finish_work_in_15_days_l742_742432

theorem B_can_finish_work_in_15_days :
  (A_work_rate B_work_rate : ℕ → ℕ) 
  (W : ℕ)
  (B_days x : ℕ)
  (H1 : ∀ W, A_work_rate 12 = W/12)
  (H2 : ∀ W, B_work_rate x = W/x)
  (H3 : ∀ W, 10 * B_work_rate x + 4 * (W/12) = W)
  (H4 : ∀ W, 10 * B_work_rate x + 4 * (W/12) = W)
  (solve_x : B_days = 15) :
  B_days = 15 :=
by
  sorry

end B_can_finish_work_in_15_days_l742_742432


namespace find_a_b_l742_742758

theorem find_a_b (a b : ℝ) (P Q : ℝ × ℝ) (hP : P = (-1, 1)) (hQ : Q = (3, -1)) :
  let line_equation := λ (x y : ℝ), a * x - y + b = 0,
  a = 2 ∧ b = -2 :=
sorry

end find_a_b_l742_742758


namespace domain_of_f_l742_742295

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 1)^0 + real.sqrt (x + 1)

-- State the problem as a Lean theorem
theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = Ici (-1) :=
sorry

end domain_of_f_l742_742295


namespace num_ordered_integer_pairs_l742_742932

theorem num_ordered_integer_pairs :
  let is_solution (a b : ℤ) := ∃ (x y : ℤ), x^2 + a * x + b = 167 * y in
  finset.range 2004 |>.sum (λ a, finset.range 2004.count (λ b, is_solution (a + 1) (b + 1))) = 2020032 :=
sorry

end num_ordered_integer_pairs_l742_742932


namespace tan_150_eq_neg_sqrt_3_l742_742020

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742020


namespace MB_eq_MT_l742_742964

-- Define the relevant points and properties
variables {A B C O T B' M : Type}
variables [EuclideanGeometry (triangle A B C)]
variables (angleB : angle A B C = 90)
variables (circlek : circle O (segment B C) tangent_segment A C)
variables (T_tangent : tangent_point A (circle O k T))
variables (B'_midpoint : midpoint B' A C)
variables (M_intersection : intersect_line M (line B B') (line A T))

-- Define the proposition to prove
theorem MB_eq_MT : segment_length M B = segment_length M T :=
by
  -- Skip the proof as the instruction specified
  sorry

end MB_eq_MT_l742_742964


namespace hexagon_is_truncated_pyramid_l742_742201

structure Rectangle :=
(a b : ℝ)

def aspect_ratio (r : Rectangle) : ℝ := r.a / r.b

structure IsoscelesTrapezoid := 
-- Add necessary fields or properties for trapezoids if required for further usage.
(legs : ℝ)

structure Polyhedron :=
(parallel_rectangles : Rectangle × Rectangle) 
(other_faces : List IsoscelesTrapezoid)

def is_truncated_pyramid (p : Polyhedron) : Prop :=
aspect_ratio p.parallel_rectangles.1 = aspect_ratio p.parallel_rectangles.2

theorem hexagon_is_truncated_pyramid 
  (p : Polyhedron) 
  (condition1 : aspect_ratio p.parallel_rectangles.1 = aspect_ratio p.parallel_rectangles.2) 
  (condition2 : ∀ t ∈ p.other_faces, true) : 
  is_truncated_pyramid p :=
sorry

end hexagon_is_truncated_pyramid_l742_742201


namespace sum_of_valid_n_l742_742372

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742372


namespace probability_at_least_one_male_l742_742645

-- Definitions according to the problem conditions
def total_finalists : ℕ := 8
def female_finalists : ℕ := 5
def male_finalists : ℕ := 3
def num_selected : ℕ := 3

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probabilistic statement
theorem probability_at_least_one_male :
  let total_ways := binom total_finalists num_selected
  let ways_all_females := binom female_finalists num_selected
  let ways_at_least_one_male := total_ways - ways_all_females
  (ways_at_least_one_male : ℚ) / total_ways = 23 / 28 :=
by
  sorry

end probability_at_least_one_male_l742_742645


namespace plates_arrangement_l742_742489

def factorial (n : ℕ) : ℕ :=
nat.rec 1 (λ n fn, (n + 1) * fn) n

def binomial_coeff (n k : ℕ) : ℕ :=
if h : k ≤ n then
  factorial n / (factorial k * factorial (n - k))
else 0

def number_of_arrangements (total : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : ℕ :=
  let total_fact := factorial total in
  let blue_fact := factorial blue in
  let red_fact := factorial red in
  let green_fact := factorial green in
  let yellow_fact := factorial yellow in
  total_fact / (blue_fact * red_fact * green_fact * yellow_fact)

def total_arrangements : ℕ :=
number_of_arrangements 13 6 3 4 1

def arrangements_with_adjacent_green : ℕ :=
number_of_arrangements 10 6 3 1 1

def non_adjacent_green_arrangements : ℕ :=
total_arrangements - arrangements_with_adjacent_green

theorem plates_arrangement (total : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) :
  (total = 13) ∧ (blue = 6) ∧ (red = 3) ∧ (green = 4) ∧ (yellow = 1) →
  non_adjacent_green_arrangements = 876 :=
by {
  intros,
  sorry
}

end plates_arrangement_l742_742489


namespace chicks_increased_l742_742628

theorem chicks_increased (chicks_day1 chicks_day2: ℕ) (H1 : chicks_day1 = 23) (H2 : chicks_day2 = 12) : 
  chicks_day1 + chicks_day2 = 35 :=
by
  sorry

end chicks_increased_l742_742628


namespace quadratic_polynomial_solution_count_l742_742958

-- Defining the polynomial and conditions in Lean
variable {R : Type*} [CommRing R]

noncomputable def f (x : R) : R := sorry  -- Assuming f(x) is a quadratic polynomial, defined elsewhere.

-- The primary theorem statement based on our analysis
theorem quadratic_polynomial_solution_count
  (hf_quad : ∃ (a b c : R), f = λ x, a*x^2 + b*x + c)
  (h_eq : ∀ x, (f(x))^3 - 4*(f(x)) = 0 → (f(x) = 0 ∨ f(x) = 2 ∨ f(x) = -2)
           (hf_zero : ∃! x, f(x) = 0)
           (hf_two : ∃! x₁ x₂, x₁ ≠ x₂ ∧ f(x₁) = 2 ∧ f(x₂) = 2)
           (hf_neg_two : ¬∃ x, f(x) = -2)) : 
  ∃! x, (f(x))^2 = 1 ∧ ∀ y, y ≠ x → (f(y))^2 = 1 :=
begin
  sorry
end

end quadratic_polynomial_solution_count_l742_742958


namespace Incorrect_statements_l742_742673

section Problem

variables {a : ℕ}
def M : set ℕ := {x | (x = a ∨ x = 3)}
def N : set ℕ := {x | (x = 1 ∨ x = 4)}

theorem Incorrect_statements :
  ¬ (|M ∪ N| = 4 → M ∩ N ≠ ∅) ∧
  ¬ (M ∩ N ≠ ∅ → |M ∪ N| = 4) ∧
  ¬ ({1, 3, 4} ⊆ M ∪ N → M ∩ N ≠ ∅) :=
by sorry

end Problem

end Incorrect_statements_l742_742673


namespace total_candies_l742_742503

def candies_in_boxes (num_boxes: Nat) (pieces_per_box: Nat) : Nat :=
  num_boxes * pieces_per_box

theorem total_candies :
  candies_in_boxes 3 6 + candies_in_boxes 5 8 + candies_in_boxes 4 10 = 98 := by
  sorry

end total_candies_l742_742503


namespace tan_150_deg_l742_742026

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742026


namespace divisors_greater_than_9_fact_and_even_quotient_l742_742184

theorem divisors_greater_than_9_fact_and_even_quotient {d : ℕ} (h₁ : d ∣ fact 10) (h₂ : d > fact 9) (h₃ : even (fact 10 / d)) : ∃ (n = 4), (d ∣ fact 10) ∧ (d > fact 9) ∧ even (fact 10 / d) := 
sorry

end divisors_greater_than_9_fact_and_even_quotient_l742_742184


namespace prove_concurrency_l742_742611

noncomputable def problem_statement
  (ABC : Triangle ℝ)  -- triangle ABC
  (D : Point ℝ)  -- interior point D
  (incircle_ABD : Circle ℝ) (incircle_BCD : Circle ℝ) (incircle_CAD : Circle ℝ) -- incircles
  (A1 B1 C1 A2 B2 C2 : Point ℝ) -- points of tangency
  (E : Point ℝ)  -- intersection of B1C2 and B2C1
  (F : Point ℝ)  -- intersection of A1C2 and A2C1
  : Prop := 
  let AF := Line (ABC.vertexA, F) in
  let BE := Line (ABC.vertexB, E) in
  let C1D := Line (C1, D) in
  AF.isConcurrentWith BE C1D

theorem prove_concurrency 
  (ABC : Triangle ℝ)
  (D A1 B1 C1 A2 B2 C2 E F : Point ℝ)
  (incircle_ABD incircle_BCD incircle_CAD : Circle ℝ)
  (h1 : isInteriorPoint D ABC) 
  (h2 : isIncircle incircle_ABD ABC ∧ isIncircle incircle_BCD ABC ∧ isIncircle incircle_CAD ABC)
  (h3 : isTangencyPoint A1 BC incircle_ABD ∧ isTangencyPoint B1 CA incircle_BCD ∧ isTangencyPoint C1 AB incircle_CAD)
  (h4 : isTangencyPoint A2 AD incircle_ABD ∧ isTangencyPoint B2 BD incircle_BCD ∧ isTangencyPoint C2 CD incircle_CAD)
  (h5 : isIntersectionPoint E (Line B1 C2) (Line B2 C1))
  (h6 : isIntersectionPoint F (Line A1 C2) (Line A2 C1)) :
  problem_statement ABC D incircle_ABD incircle_BCD incircle_CAD A1 B1 C1 A2 B2 C2 E F :=
sorry

end prove_concurrency_l742_742611


namespace tan_150_degrees_l742_742044

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742044


namespace sum_of_valid_n_l742_742336

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742336


namespace simplify_and_find_value_l742_742276

-- Definitions for conditions
def satisfies_condition (a b : ℤ) : Prop :=
  (a - 3) ^ 2 + |b + 2| = 0

-- Main theorem to prove the equality
theorem simplify_and_find_value (a b : ℤ) 
  (h : satisfies_condition a b) : 
  ({-{a^2}+3*a*b-3*b^2}-2*(-(1/2)*a^2+4*a*b-(3/2)*b^2) = 30) :=
by
  sorry

end simplify_and_find_value_l742_742276


namespace prove_fatou_variant_l742_742272

open MeasureTheory

noncomputable def fatou_variant (Ω : Type*) [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
  (ξ ξ_n : ℕ → Ω → ℝ) : Prop :=
  (∀ n, 0 ≤ ξ_n n) →
  (∀ᵐ x ∂μ, Tendsto (fun n => ξ_n n x) atTop (𝓝 (ξ x))) →
  (BddAbove (Set.range (fun n => ∫ x, ξ_n n x ∂μ))) →
  Integrable ξ μ ∧ ∫ x, ξ x ∂μ ≤ Sup (Set.range (fun n => ∫ x, ξ_n n x ∂μ))

theorem prove_fatou_variant (Ω : Type*) [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
  (ξ : Ω → ℝ) (ξ_n : ℕ → Ω → ℝ) :
  fatou_variant Ω μ ξ ξ_n :=
begin
  sorry, -- The actual proof is not required
end

end prove_fatou_variant_l742_742272


namespace fixed_point_l742_742194

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 2

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = 3 :=
by
  unfold f
  sorry

end fixed_point_l742_742194


namespace problem_part_I_problem_part_II_l742_742976

-- Define the problem and the proof requirements in Lean 4
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (sinB_nonneg : 0 ≤ Real.sin B) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) 
(h_a : a = 2) (h_b : b = 2) : 
Real.cos B = 1/4 :=
sorry

theorem problem_part_II (a b c : ℝ) (A B C : ℝ) (h_B : B = π / 2) 
(h_a : a = Real.sqrt 2) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) :
1/2 * a * c = 1 :=
sorry

end problem_part_I_problem_part_II_l742_742976


namespace orthocenter_perpendicular_circumcenter_l742_742197

theorem orthocenter_perpendicular_circumcenter
  (A B C M N D H P Q K L : Point)
  (hM : midpoint A B M)
  (hN : midpoint A C N)
  (hD : foot A B C D)
  (hH : orthocenter A B C H)
  (hE : circumcircle M N D)
  (hP : circle_interpoint M H P hE)
  (hQ : circle_interpoint N H Q hE)
  (hK : lines_interpoint D P B H K)
  (hL : lines_interpoint D Q C H L) :
  perpendicular K L E H :=
sorry

end orthocenter_perpendicular_circumcenter_l742_742197


namespace sum_binom_solutions_l742_742390

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742390


namespace inequality_solution_l742_742726

theorem inequality_solution (x : ℝ) :
    (x < 1 ∨ (3 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) ∨ x > 6) ↔
    ((x - 1) * (x - 3) * (x - 4) / ((x - 2) * (x - 5) * (x - 6)) > 0) := by
  sorry

end inequality_solution_l742_742726


namespace percent_parents_no_full_time_jobs_l742_742208

structure Survey where
  total_parents : ℕ
  percent_mothers : ℚ
  percent_fathers : ℚ
  mothers_full_time : ℚ
  fathers_full_time : ℚ

def survey_conditions : Survey :=
{ total_parents := 100,
  percent_mothers := 40 / 100,
  percent_fathers := 60 / 100,
  mothers_full_time := 3 / 4,
  fathers_full_time := 9 / 10 }

theorem percent_parents_no_full_time_jobs (s : Survey) :
  (100 - ((s.mothers_full_time * (s.percent_mothers * s.total_parents).toRat + 
          s.fathers_full_time * (s.percent_fathers * s.total_parents).toRat).toNat)) / 1 = 16 :=
by
  sorry

end percent_parents_no_full_time_jobs_l742_742208


namespace general_term_a_general_term_b_sum_c_l742_742167

-- Given conditions

def S (n : ℕ) : ℚ := (3 / 2) * n^2 + (1 / 2) * n
def b (n : ℕ) : ℚ := 2^n
def c (n : ℕ) : ℕ → ℚ := λ n, (3 * n - 1) * 2^n
def T (n : ℕ) : ℕ → ℚ := λ n, (3 * n - 4) * 2^(n + 1) + 8

-- Prove general formula for a_n
theorem general_term_a (a : ℕ → ℚ) (h : ∀ n, S n = a n + S (n - 1)) (n : ℕ) :
  a n = 3 * n - 1 := sorry

-- Prove general formula for b_n given conditions on b_1 and b_4
theorem general_term_b (b : ℕ → ℚ) (h1 : b 1 + b 4 = 18) (h2 : b 2 * b 3 = 32) (n : ℕ) :
  b n = 2^n := sorry

-- Prove the sum of first n terms of c_n
theorem sum_c (c : ℕ → ℚ) (T : ℕ → ℚ) (h : ∀ n, c n = (3 * n - 1) * 2^n) (n : ℕ) :
  T n = (3 * n - 4) * 2^(n + 1) + 8 := sorry

end general_term_a_general_term_b_sum_c_l742_742167


namespace customer_price_increase_l742_742840

variable (manufacture_cost designer_markup distributor_markup retailer_markup : ℕ)

theorem customer_price_increase :
  let designer_price := manufacture_cost * (1 + designer_markup / 100)
  let distributor_price := designer_price * (1 + distributor_markup / 100)
  let customer_price := distributor_price * (1 + retailer_markup / 100)
  (customer_price - manufacture_cost) / manufacture_cost * 100 = 135.2 :=
by
  sorry

example : customer_price_increase 100 40 20 40 := by
  sorry

end customer_price_increase_l742_742840


namespace equal_distances_l742_742248

variables (A B C O : Point) [acute_triangle A B C O]
variables (ΓB ΓC : Circle)
variables (X Y : Point)
variables (LineA : Line)
variables [passes_through LineA A]
variables [intersects_again LineA ΓB X]
variables [intersects_again LineA ΓC Y]

-- Problem Statement: To Prove |OX| = |OY|
theorem equal_distances (h1 : passes_through A B ΓB)
    (h2 : tangent_to ΓB A C)
    (h3 : passes_through A C ΓC)
    (h4 : tangent_to ΓC A B)
    (h5 : circumcenter O A B C)
    (h6 : acute_triangle A B C O) :
  distance O X = distance O Y := sorry

end equal_distances_l742_742248


namespace arithmetic_sequence_problem_l742_742966

noncomputable def a_n (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

def arithmetic_sequence_sum (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

def b_n (a_n : ℤ) : ℤ := 2 ^ (a_n / 2)

noncomputable def T_n (n : ℕ) : ℤ :=
  ∑ i in (finset.range n), (a_n i - 4 2 + 6) * b_n ((a_n i - 4 2))

theorem arithmetic_sequence_problem
  (S : ℕ → ℤ)
  (a1 d m : ℤ)
  (h1 : S (m - 1) = -4)
  (h2 : S m = 0)
  (h3 : S (m + 2) = 14)
  (h4 : m ≥ 2)
  (S_eq : ∀ n, S n = arithmetic_sequence_sum a1 d (n.to_nat))
  : a1 = -4 ∧ m = 5 ∧ T_n =  λ n => (n-1) * 2 ^ (n - 1) + 1/2 := sorry

end arithmetic_sequence_problem_l742_742966


namespace binom_sum_l742_742386

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742386


namespace sum_of_valid_n_l742_742369

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742369


namespace hyperbola_solution_l742_742162

noncomputable def hyperbola_equation (x y a b : ℝ) : Prop :=
  (x^2) / (a^2) - (y^2) / (b^2) = 1

noncomputable def asymptote_condition (a b : ℝ) : Prop := 
  b / a = 2

noncomputable def passes_through_point (x y a b : ℝ) : Prop :=
  hyperbola_equation (sqrt 6) 4 a b

theorem hyperbola_solution :
  ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ asymptote_condition a b ∧ 
  passes_through_point (sqrt 6) 4 a b ∧ 
  hyperbola_equation 1 0 (sqrt 2) (2 * sqrt 2) :=
sorry

end hyperbola_solution_l742_742162


namespace stuffed_animal_cost_is_6_l742_742875

-- Definitions for the costs of items
def sticker_cost (s : ℕ) := s
def magnet_cost (m : ℕ) := m
def stuffed_animal_cost (a : ℕ) := a

-- Conditions given in the problem
def conditions (m s a : ℕ) :=
  (m = 3) ∧
  (m = 3 * s) ∧
  (m = (2 * a) / 4)

-- The theorem stating the cost of a single stuffed animal
theorem stuffed_animal_cost_is_6 (s m a : ℕ) (h : conditions m s a) : a = 6 :=
by
  sorry

end stuffed_animal_cost_is_6_l742_742875


namespace probability_of_odd_die_roll_probability_of_double_tail_coin_flip_l742_742802

theorem probability_of_odd_die_roll : 
  let outcomes := {1, 2, 3, 4, 5, 6}
  let odd_outcomes := {1, 3, 5}
  outcomes.size = 6 → odd_outcomes.size = 3 →
  (odd_outcomes.size.to_rat / outcomes.size.to_rat) = 1 / 2 :=
by
  sorry

theorem probability_of_double_tail_coin_flip :
  let coin_outcomes := {'HH', 'HT', 'TH', 'TT'}
  let tail_outcomes := {'TT'}
  coin_outcomes.size = 4 → tail_outcomes.size = 1 →
  (tail_outcomes.size.to_rat / coin_outcomes.size.to_rat) = 1 / 4 :=
by
  sorry

end probability_of_odd_die_roll_probability_of_double_tail_coin_flip_l742_742802


namespace trajectory_parabola_l742_742680

noncomputable def otimes (x1 x2 : ℝ) : ℝ := (x1 + x2)^2 - (x1 - x2)^2

theorem trajectory_parabola (x : ℝ) (h : 0 ≤ x) : 
  ∃ (y : ℝ), y^2 = 8 * x ∧ (∀ P : ℝ × ℝ, P = (x, y) → (P.snd^2 = 8 * P.fst)) :=
by
  sorry

end trajectory_parabola_l742_742680


namespace part_I_part_II_part_III_l742_742173

-- Given function definitions and conditions stated as hypotheses.
def f (ω : ℝ) (x : ℝ) : ℝ := (Real.sin (ω * x))^2 + sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x)

def f_with_omega_2 (x : ℝ) : ℝ := Real.sin (4 * x - π / 6) + 1 / 2

theorem part_I (ω : ℝ) (h1 : ∀ x, f ω x = Real.sin (2 * ω * x - π / 6) + 1 / 2)
  (h2 : 2 * π / (2 * ω) = π / 2) : ω = 2 :=
sorry

theorem part_II : ∀ x, f_with_omega_2 x = Real.sin (4 * x - π / 6) + 1 / 2 → 
  ∃ a b, a < b ∧ ∀ x, a ≤ x ∧ x ≤ b → f_with_omega_2 x < f_with_omega_2 (x + 1) :=
sorry

theorem part_III (m : ℝ) (h3 : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ π/2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π/2 → |f_with_omega_2 x₁ - f_with_omega_2 x₂| < m)
  : m ∈ set.Ioi (2 : ℝ) :=
sorry

end part_I_part_II_part_III_l742_742173


namespace intersection_of_A_and_B_l742_742607

def A : Set ℤ := {-1, 0, 3, 5}
def B : Set ℤ := {x | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := 
by 
  sorry

end intersection_of_A_and_B_l742_742607


namespace find_a_l742_742994

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end find_a_l742_742994


namespace john_can_buy_notebooks_l742_742231

theorem john_can_buy_notebooks : 
  ∀ (total_dollars : ℕ) (price_per_notebook_dollars : ℕ) (price_per_notebook_cents : ℕ), 
  total_dollars = 30 → 
  price_per_notebook_dollars = 2 → 
  price_per_notebook_cents = 40 → 
  (total_dollars * 100) div ((price_per_notebook_dollars * 100) + price_per_notebook_cents) = 12 := by
  intros total_dollars price_per_notebook_dollars price_per_notebook_cents h₁ h₂ h₃
  sorry

end john_can_buy_notebooks_l742_742231


namespace loss_percentage_correct_l742_742422

def cost_price := 1500
def selling_price := 1245
def loss_amount := cost_price - selling_price
def loss_percentage := (loss_amount / cost_price.toFloat) * 100

theorem loss_percentage_correct : loss_percentage = 17 := 
by
  sorry

end loss_percentage_correct_l742_742422


namespace total_amount_owed_l742_742257

-- Conditions
def borrowed_amount : ℝ := 500
def monthly_interest_rate : ℝ := 0.02
def months_not_paid : ℕ := 3

-- Compounded monthly formula
def amount_after_n_months (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Theorem statement
theorem total_amount_owed :
  amount_after_n_months borrowed_amount monthly_interest_rate months_not_paid = 530.604 :=
by
  -- Proof to be filled in here
  sorry

end total_amount_owed_l742_742257


namespace simplify_evaluate_l742_742723

def f (x y : ℝ) : ℝ := 4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1

theorem simplify_evaluate : f (-2) (1/2) = -13 := by
  sorry

end simplify_evaluate_l742_742723


namespace sum_intercepts_of_line_l742_742842

theorem sum_intercepts_of_line (x y : ℝ) (h_eq : y - 6 = -2 * (x - 3)) :
  (∃ x_int : ℝ, (0 - 6 = -2 * (x_int - 3)) ∧ x_int = 6) ∧
  (∃ y_int : ℝ, (y_int - 6 = -2 * (0 - 3)) ∧ y_int = 12) →
  6 + 12 = 18 :=
by sorry

end sum_intercepts_of_line_l742_742842


namespace roots_poly_sum_cubed_eq_l742_742247

theorem roots_poly_sum_cubed_eq :
  ∀ (r s t : ℝ), (r + s + t = 0) 
  → (∀ x, 9 * x^3 + 2023 * x + 4047 = 0 → x = r ∨ x = s ∨ x = t) 
  → (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1349 :=
by
  intros r s t h_sum h_roots
  sorry

end roots_poly_sum_cubed_eq_l742_742247


namespace cistern_leak_empty_time_l742_742838

theorem cistern_leak_empty_time :
  (∀ (fill_rate : ℝ) (leak_rate : ℝ), 
    fill_rate = 1 / 14 →
    (1 / 14 - leak_rate) = 1 / 16 →
    leak_rate = 1 / 112 →
    1 / leak_rate = 112) := 
by 
  intros fill_rate leak_rate h_fill_rate h_combined_rate h_leak_rate
  rw [h_leak_rate]
  norm_num
  sorry

end cistern_leak_empty_time_l742_742838


namespace envelope_weight_l742_742869

-- Define the conditions as constants
def total_weight_kg : ℝ := 7.48
def num_envelopes : ℕ := 880
def kg_to_g_conversion : ℝ := 1000

-- Calculate the total weight in grams
def total_weight_g : ℝ := total_weight_kg * kg_to_g_conversion

-- Define the expected weight of one envelope in grams
def expected_weight_one_envelope_g : ℝ := 8.5

-- The proof statement
theorem envelope_weight :
  total_weight_g / num_envelopes = expected_weight_one_envelope_g := by
  sorry

end envelope_weight_l742_742869


namespace fish_filets_total_l742_742878

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end fish_filets_total_l742_742878


namespace tan_150_deg_l742_742031

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742031


namespace simplify_sqrt_product_l742_742887

theorem simplify_sqrt_product (y : ℝ) (hy : 0 ≤ y) : 
  (√(48 * y) * √(18 * y) * √(50 * y)) = 120 * y * √(3 * y) := 
by
  sorry

end simplify_sqrt_product_l742_742887


namespace tan_150_eq_neg_inv_sqrt3_l742_742057

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742057


namespace sum_of_reciprocals_of_triangular_numbers_l742_742517

theorem sum_of_reciprocals_of_triangular_numbers : 
  (∑ n in Finset.range 2003 + 1, (1 : ℚ) / (n * (n + 1) / 2)) = 2003 / 2002 :=
by
  sorry

end sum_of_reciprocals_of_triangular_numbers_l742_742517


namespace tan_150_eq_neg_sqrt_3_l742_742011

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742011


namespace remainder_when_divided_by_11_l742_742977

theorem remainder_when_divided_by_11 (n : ℕ) 
  (h1 : 10 ≤ n ∧ n < 100) 
  (h2 : n % 9 = 1) 
  (h3 : n % 10 = 3) : 
  n % 11 = 7 := 
sorry

end remainder_when_divided_by_11_l742_742977


namespace probability_diana_larger_l742_742121

/--
Diana uses a regular 10-sided die with numbers from 1 to 10.
Apollo uses a standard 6-sided die with numbers from 1 to 6.
Prove that the probability that Diana's number is larger than Apollo's number is 13/20.
-/
theorem probability_diana_larger :
  let outcomes_diana := {x // x ∈ finset.range 1 11}
  let outcomes_apollo := {y // y ∈ finset.range 1 7}
  let successful_outcomes := finset.card { p ∈ outcomes_diana ×ˢ outcomes_apollo | p.1 > p.2 }
  let total_outcomes := finset.card (outcomes_diana ×ˢ outcomes_apollo)
  (successful_outcomes : ℚ) / total_outcomes = 13 / 20 :=
sorry

end probability_diana_larger_l742_742121


namespace tan_inequality_solution_set_l742_742943

noncomputable theory

open Real

theorem tan_inequality_solution_set :
  ∀ k : ℤ, ∃ α : ℝ, (-π / 6 + k * π) < α ∧ α < (π / 2 + k * π) ∧ (tan α + sqrt 3 / 3) > 0 := 
sorry

end tan_inequality_solution_set_l742_742943


namespace proof_problem_l742_742238

-- Definitions
structure Vector2D :=
(magnitude : ℝ)
(direction : ℝ) -- could be an angle, for simplicity here

structure VectorSet :=
(vectors : Set Vector2D)
(noncollinear : ¬ ∀ v₁ v₂ ∈ vectors, v₁.direction = v₂.direction)

def is_maximal_vector (W : VectorSet) (a : Vector2D) : Prop :=
  a.magnitude ≥ (W.vectors.erase a).sum (λ v, v.magnitude)

-- Given statements as propositions to be proven
def statement_1 (W : VectorSet) : Prop :=
  ∀ (v ∈ W.vectors), ∃ a ∈ W.vectors, is_maximal_vector W a

def statement_2 (a b : Vector2D) (W : VectorSet) : Prop :=
  W.vectors = {a, b, ⟨a.magnitude + b.magnitude, -a.direction - b.direction⟩} →
  ∀ c ∈ W.vectors, is_maximal_vector W c

def statement_3 (W₁ W₂ : VectorSet) : Prop :=
  ∀ (w₁ ∈ W₁.vectors) (w₂ ∈ W₂.vectors), w₁ ≠ w₂ →
  (∀ v ∈ W₁.vectors ∪ W₂.vectors, is_maximal_vector (⟨W₁.vectors ∪ W₂.vectors, sorry⟩) v)

-- The proof problem
theorem proof_problem (W W₁ W₂ : VectorSet) (a b : Vector2D) :
  ¬ statement_1 W ∧
  statement_2 a b W ∧
  statement_3 W₁ W₂ :=
by sorry

end proof_problem_l742_742238


namespace max_balloons_l742_742265

theorem max_balloons (p : ℝ) (h₀ : p > 0) (h₁ : ∃ d, d = 24 * p) (h₂ : ∀ x, x = p / 2) :
  ∃ n, n = 32 :=
by
  exist 32
  sorry

end max_balloons_l742_742265


namespace tetrahedron_circumscribed_sphere_radius_l742_742963

noncomputable def tetrahedron_radius (PA AB AC BC : ℝ) (h1 : AB = AC) (h2 : AB = 2 * Real.sqrt 3) (h3 : BC = 6)
  (h4 : PA = 4) (h5 : ∀ ABC : Plane, PA ⊥ ABC) : ℝ :=
4

theorem tetrahedron_circumscribed_sphere_radius :
  tetrahedron_radius 4 (2 * Real.sqrt 3) (2 * Real.sqrt 3) 6 (by rw [Real.sqrt, Real.sqrt, mul_comm]; exact rfl) (by exact rfl) (by exact rfl) = 4 :=
sorry

end tetrahedron_circumscribed_sphere_radius_l742_742963


namespace p_sufficient_but_not_necessary_for_q_l742_742268

variable (x : ℝ) (p q : Prop)

def p_condition : Prop := 0 < x ∧ x < 1
def q_condition : Prop := x^2 < 2 * x

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p_condition x → q_condition x) ∧
  ¬ (∀ x : ℝ, q_condition x → p_condition x) := by
  sorry

end p_sufficient_but_not_necessary_for_q_l742_742268


namespace sum_of_odd_coefficients_l742_742251

noncomputable def P (x : ℝ) := x^10

theorem sum_of_odd_coefficients :
  let a : list ℝ := [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_{10}] 
  in  (P (x + 1) = a.foldl (λ acc (coeff, idx), acc + coeff * (x + 1) ^ idx) 0) 
  → (a_1 + a_3 + a_5 + a_7 + a_9 = 512) := sorry

end sum_of_odd_coefficients_l742_742251


namespace tan_150_deg_l742_742032

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742032


namespace total_cost_oranges_mangoes_l742_742535

theorem total_cost_oranges_mangoes
  (initial_price_orange : ℝ)
  (initial_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  initial_price_orange = 40 →
  initial_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  let new_price_orange := initial_price_orange * (1 + price_increase_percentage)
  let new_price_mango := initial_price_mango * (1 + price_increase_percentage)
  let total_cost_oranges := new_price_orange * quantity_oranges
  let total_cost_mangoes := new_price_mango * quantity_mangoes
  let total_cost := total_cost_oranges + total_cost_mangoes in
  total_cost = 1035 :=
by
  intros h_orange h_mango h_percentage h_qty_oranges h_qty_mangoes
  let new_price_orange := 40 * (1 + 0.15)
  let new_price_mango := 50 * (1 + 0.15)
  let total_cost_oranges := new_price_orange * 10
  let total_cost_mangoes := new_price_mango * 10
  let total_cost := total_cost_oranges + total_cost_mangoes
  sorry

end total_cost_oranges_mangoes_l742_742535


namespace points_on_opposite_sides_of_line_l742_742577

theorem points_on_opposite_sides_of_line (a : ℝ) :
  let A := (3, 1)
  let B := (-4, 6)
  (3 * A.1 - 2 * A.2 + a) * (3 * B.1 - 2 * B.2 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  let A := (3, 1)
  let B := (-4, 6)
  have hA : 3 * A.1 - 2 * A.2 + a = 7 + a := by sorry
  have hB : 3 * B.1 - 2 * B.2 + a = -24 + a := by sorry
  exact sorry

end points_on_opposite_sides_of_line_l742_742577


namespace cone_height_l742_742832

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l742_742832


namespace sum_of_n_values_l742_742362

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742362


namespace solve_inequality_l742_742281

def inequality_solution (a : ℝ) : Set ℝ :=
  if a = 0 then Ioi 1
  else if 0 < a ∧ a < 1 then Ioo 1 (1 / a)
  else if a = 1 then ∅
  else if a > 1 then Ioo (1 / a) 1
  else Iio (1 / a) ∪ Ioi 1

theorem solve_inequality (a : ℝ) : 
  (a = 0 → {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0} = Ioi 1) ∧
  (0 < a ∧ a < 1 → {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0} = Ioo 1 (1 / a)) ∧
  (a = 1 → {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0} = ∅) ∧
  (a > 1 → {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0} = Ioo (1 / a) 1) ∧
  (a < 0 → {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0} = (Iio (1 / a) ∪ Ioi 1)) :=
by sorry

end solve_inequality_l742_742281


namespace super_flippy_divisible_by_12_unique_l742_742846

/--
A number is defined as super-flippy if its digits alternate between two distinct digits from the set {2, 4, 6, 8}.
-/
def is_super_flippy (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 5 ∧ 
  (∀ i, i < digits.length - 1 → (digits.nth i ≠ digits.nth (i + 1))) ∧ 
  (∀ d ∈ digits.to_finset, d ∈ {2, 4, 6, 8})

/--
A number is divisible by 12 if it is divisible by both 3 and 4.
-/
def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

/--
Count the number of five-digit super-flippy numbers divisible by 12.
-/
def count_super_flippy_divisible_by_12 : ℕ :=
  (finset.range 100000).filter (λ n, n ≥ 10000 ∧ is_super_flippy n ∧ is_divisible_by_12 n).card

theorem super_flippy_divisible_by_12_unique : count_super_flippy_divisible_by_12 = 1 :=
sorry

end super_flippy_divisible_by_12_unique_l742_742846


namespace Blossom_room_area_square_inches_l742_742773

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

theorem Blossom_room_area_square_inches :
  (let length_feet := 10 in
   let width_feet := 10 in
   let length_inches := feet_to_inches length_feet in
   let width_inches := feet_to_inches width_feet in
   length_inches * width_inches = 14400) :=
by
  let length_feet := 10
  let width_feet := 10
  let length_inches := feet_to_inches length_feet
  let width_inches := feet_to_inches width_feet
  show length_inches * width_inches = 14400
  sorry

end Blossom_room_area_square_inches_l742_742773


namespace triangle_angle_120_l742_742145

theorem triangle_angle_120 (a b c : ℝ) (B : ℝ) (hB : B = 120) :
  a^2 + a * c + c^2 - b^2 = 0 := by
sorry

end triangle_angle_120_l742_742145


namespace tan_150_eq_l742_742001

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742001


namespace number_difference_l742_742708

theorem number_difference 
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 2 * a2)
  (h2 : a1 = 3 * a3)
  (h3 : (a1 + a2 + a3) / 3 = 88) : 
  a1 - a3 = 96 :=
sorry

end number_difference_l742_742708


namespace fraction_unilluminated_l742_742812

-- Define the room and mirror dimensions
variables (L W H w : ℝ)
-- Define the unilluminated fraction
def unilluminated_fraction : ℝ := 21.5 / 32

-- State the theorem about the fraction of unilluminated wall area
theorem fraction_unilluminated (h_full_height : h = H) :
  (w / W) * (h / H) = unilluminated_fraction :=
sorry

end fraction_unilluminated_l742_742812


namespace total_cost_oranges_mangoes_l742_742536

theorem total_cost_oranges_mangoes
  (initial_price_orange : ℝ)
  (initial_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  initial_price_orange = 40 →
  initial_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  let new_price_orange := initial_price_orange * (1 + price_increase_percentage)
  let new_price_mango := initial_price_mango * (1 + price_increase_percentage)
  let total_cost_oranges := new_price_orange * quantity_oranges
  let total_cost_mangoes := new_price_mango * quantity_mangoes
  let total_cost := total_cost_oranges + total_cost_mangoes in
  total_cost = 1035 :=
by
  intros h_orange h_mango h_percentage h_qty_oranges h_qty_mangoes
  let new_price_orange := 40 * (1 + 0.15)
  let new_price_mango := 50 * (1 + 0.15)
  let total_cost_oranges := new_price_orange * 10
  let total_cost_mangoes := new_price_mango * 10
  let total_cost := total_cost_oranges + total_cost_mangoes
  sorry

end total_cost_oranges_mangoes_l742_742536


namespace trains_total_distance_l742_742328

theorem trains_total_distance (speedA_kmph speedB_kmph time_min : ℕ)
                             (hA : speedA_kmph = 70)
                             (hB : speedB_kmph = 90)
                             (hT : time_min = 15) :
    let speedA_kmpm := (speedA_kmph : ℝ) / 60
    let speedB_kmpm := (speedB_kmph : ℝ) / 60
    let distanceA := speedA_kmpm * (time_min : ℝ)
    let distanceB := speedB_kmpm * (time_min : ℝ)
    distanceA + distanceB = 40 := 
by 
  sorry

end trains_total_distance_l742_742328


namespace sum_of_valid_n_l742_742337

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742337


namespace sum_binom_solutions_l742_742395

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742395


namespace tan_150_degrees_l742_742054

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742054


namespace determinant_eval_l742_742922

open Matrix

noncomputable def matrix_example (α γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, 3 * Real.sin γ],
    ![2 * Real.cos α, -Real.sin γ, 0]]

theorem determinant_eval (α γ : ℝ) :
  det (matrix_example α γ) = 10 * Real.sin α * Real.sin γ * Real.cos α :=
sorry

end determinant_eval_l742_742922


namespace area_of_circumscribed_circle_l742_742486

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742486


namespace length_AB_in_triangle_l742_742638

noncomputable def find_length_AB (cosA : ℝ) (angleC : ℝ) (BC : ℝ) : ℝ :=
  let sinA := sqrt (1 - cosA^2)
  let sinC := Real.sin (angleC)
  (BC * sinC) / sinA

theorem length_AB_in_triangle : 
  find_length_AB (3 * sqrt 10 / 10) (Real.pi * 150 / 180) 1 = sqrt 10 / 2 :=
by
  sorry

end length_AB_in_triangle_l742_742638


namespace even_expression_l742_742284

theorem even_expression (m n : ℤ) (hm : Odd m) (hn : Odd n) : Even (m + 5 * n) :=
by
  sorry

end even_expression_l742_742284


namespace smallest_n_for_1991_divisibility_l742_742942

theorem smallest_n_for_1991_divisibility :
  ∃ n : ℕ, 0 < n ∧ (∀ (a : ℕ → ℤ), function.injective a → 
    (∏ i in finset.range n, ∏ j in finset.range i, (a i - a j)) % 1991 = 0) ↔ n = 182 :=
begin
  sorry
end

end smallest_n_for_1991_divisibility_l742_742942


namespace product_of_two_equal_numbers_l742_742736

theorem product_of_two_equal_numbers (a b : ℕ) (mean : ℕ) (n1 n2 : ℕ) (h1 : mean = 17) (h2 : a = 12) (h3 : b = 20) (h4 : 2 * n1 = 2 * n2) : n1 * n2 = 324 :=
by
  have h_sum : a + b + n1 + n2 = mean * 4 := by sorry
  have h_sub : n1 + n2 = 36 := by sorry
  have h_eq : n1 = n2 := by sorry
  have h_val : n1 = 18 := by sorry
  show n1 * n2 = 324 by sorry

end product_of_two_equal_numbers_l742_742736


namespace hyperbola_foci_distance_l742_742737

theorem hyperbola_foci_distance :
  (∀ (x y : ℝ), (y = 2 * x + 3) ∨ (y = -2 * x + 7)) →
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ ((y = 2 * x + 3) ∨ (y = -2 * x + 7))) →
  (∃ h : ℝ, h = 6 * Real.sqrt 2) :=
by
  sorry

end hyperbola_foci_distance_l742_742737


namespace probability_is_one_third_l742_742948

-- Defining the set and the conditions
def num_set := {i | 1 ≤ i ∧ i ≤ 10}
def choose_six (s : Set ℕ) := s.card = 6

-- Total ways to choose 6 integers from the set
def total_ways := Nat.choose 10 6

-- Number of favorable outcomes where 3 is the second smallest integer
def favorable_ways := 2 * (Nat.choose 7 4)

-- Definition of the probability
def probability_second_smallest_is_3 := favorable_ways / total_ways

-- The theorem to be proven
theorem probability_is_one_third (h : choose_six num_set) : probability_second_smallest_is_3 = 1 / 3 :=
  sorry

end probability_is_one_third_l742_742948


namespace maximum_combined_power_l742_742322

theorem maximum_combined_power (x1 x2 x3 : ℝ) (hx : x1 < 1 ∧ x2 < 1 ∧ x3 < 1) 
    (hcond : 2 * (x1 + x2 + x3) + 4 * (x1 * x2 * x3) = 3 * (x1 * x2 + x1 * x3 + x2 * x3) + 1) : 
    x1 + x2 + x3 ≤ 3 / 4 := 
sorry

end maximum_combined_power_l742_742322


namespace tan_150_eq_neg_sqrt_3_l742_742016

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742016


namespace remainder_17_pow_2037_mod_20_l742_742800

theorem remainder_17_pow_2037_mod_20:
      (17^1) % 20 = 17 ∧
      (17^2) % 20 = 9 ∧
      (17^3) % 20 = 13 ∧
      (17^4) % 20 = 1 → 
      (17^2037) % 20 = 17 := sorry

end remainder_17_pow_2037_mod_20_l742_742800


namespace surface_area_of_sphere_l742_742206

theorem surface_area_of_sphere (A B C D G M : Point) (h_tetrahedron : regular_tetrahedron A B C D)
  (h_length : length (line_segment A B) = 2) (h_centroid : centroid G B C D)
  (h_midpoint : midpoint M (line_segment A G)) : 
  surface_area (circumscribed_sphere (tetrahedron M B C D)) = 6 * pi :=
by
  sorry

end surface_area_of_sphere_l742_742206


namespace smallest_digit_permutation_l742_742938

open Nat

def is_digit_permutation (n m : ℕ) : Prop :=
  n.digits 10 ~ m.digits 10

theorem smallest_digit_permutation:
  ∃ n : ℕ, (n * 9).digits 10 ~ n.digits 10 ∧ (∀ m : ℕ, m * 9 ≠ 1089 → ¬ (m.digits 10 ~ (m * 9).digits 10)) :=
by
  sorry

end smallest_digit_permutation_l742_742938


namespace find_p_and_q_l742_742982

def algebraic_expression (x p q : ℤ) : ℤ :=
  x^2 + p * x + q

theorem find_p_and_q :
  ∃ p q : ℤ, (-p + q = -6) ∧ (9 + 3 * p + q = 3) ∧ p = 0 ∧ q = -6 :=
by {
  use [0, -6],
  split,
  rfl,
  split,
  rfl,
  split,
  rfl,
  rfl,
}

end find_p_and_q_l742_742982


namespace intersection_P_Q_l742_742195

section proof_problem

def P : Set ℝ := {x | 1 ≤ 2^x ∧ 2^x < 8}
def Q : Set ℕ := {1, 2, 3}
def result : Set ℕ := {1, 2}

theorem intersection_P_Q : P ∩ Q = result := 
by sorry

end proof_problem

end intersection_P_Q_l742_742195


namespace am_gm_inequality_l742_742626

theorem am_gm_inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := 
by
  sorry

end am_gm_inequality_l742_742626


namespace sum_of_fractions_integer_l742_742563

-- Definition of the problem
theorem sum_of_fractions_integer (n : ℕ) (h : n > 2) :
  ∃ (a : Fin n → ℕ), (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧
  (∃ k : ℤ, (Finset.univ.sum (λ i : Fin n, (a i : ℤ) / (a ((i + 1) % n) : ℤ)) = k)) :=
sorry

end sum_of_fractions_integer_l742_742563


namespace sin_2017pi_over_6_l742_742944

theorem sin_2017pi_over_6 : Real.sin (2017 * Real.pi / 6) = 1 / 2 := 
by 
  -- Proof to be filled in later
  sorry

end sin_2017pi_over_6_l742_742944


namespace range_of_a_for_maximum_l742_742168

variable {f : ℝ → ℝ}
variable {a : ℝ}

theorem range_of_a_for_maximum (h : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : ∀ x, f x ≤ f a → x = a) : -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_maximum_l742_742168


namespace smallest_natural_number_permutation_mul9_l742_742939

def is_permutation_of (a b : ℕ) : Prop :=
  multiset.of_digits a = multiset.of_digits b

theorem smallest_natural_number_permutation_mul9 :
  ∃ n : ℕ, is_permutation_of (n * 9) n ∧ ∀ m : ℕ, m < n → ¬ is_permutation_of (m * 9) m :=
by
  sorry

end smallest_natural_number_permutation_mul9_l742_742939


namespace triangle_angle_bounds_l742_742904

theorem triangle_angle_bounds (a b c : ℕ) (h : a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧ a ≥ b ∧ b ≥ c) :
  ∃ γ α, γ ≈ 5.73 ∧ α ≈ 130.5 :=
by
  -- Let's assume the sides of the triangle are labeled such that a ≥ b ≥ c
  -- Consequently, the angles satisfy the relation γ ≤ β ≤ α
  have h₁ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0, sorry,
  use (5.73 : ℝ), (130.5 : ℝ),
  split; sorry

end triangle_angle_bounds_l742_742904


namespace chen_steps_recorded_correct_l742_742438

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end chen_steps_recorded_correct_l742_742438


namespace ratio_of_sums_l742_742242

theorem ratio_of_sums (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49) 
  (h2 : x^2 + y^2 + z^2 = 64) 
  (h3 : a * x + b * y + c * z = 56) : 
  (a + b + c) / (x + y + z) = 7/8 := 
by 
  sorry

end ratio_of_sums_l742_742242


namespace distance_point_to_plane_l742_742586

-- Defining the necessary vectors and plane
variables {A B : EuclideanSpace}
variables {α : Plane}
variable {n : EuclideanSpace}

-- Hypotheses given in the problem
variables (h1 : Oblique (LineSegment A B) α)
variables (h2 : NormalVector n α)

-- Statement of the problem to be proved
theorem distance_point_to_plane (h1 : Oblique (LineSegment A B) α) (h2 : NormalVector n α) :
  distance A α = (| dot_product (vector A B) n |) / (| n |) :=
sorry

end distance_point_to_plane_l742_742586


namespace sum_of_valid_ns_l742_742342

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742342


namespace intersection_of_sets_l742_742252

theorem intersection_of_sets :
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  ∀ x, (x ∈ A ∧ x ∈ B) ↔ (-2 < x ∧ x < 0) :=
by
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  intro x
  sorry

end intersection_of_sets_l742_742252


namespace coin_overlap_black_region_cd_sum_l742_742902

noncomputable def black_region_probability : ℝ := 
  let square_side := 10
  let triangle_leg := 3
  let diamond_side := 3 * Real.sqrt 2
  let coin_diameter := 2
  let coin_radius := coin_diameter / 2
  let reduced_square_side := square_side - coin_diameter
  let reduced_square_area := reduced_square_side * reduced_square_side
  let triangle_area := 4 * ((triangle_leg * triangle_leg) / 2)
  let extra_triangle_area := 4 * (Real.pi / 4 + 3)
  let diamond_area := (diamond_side * diamond_side) / 2
  let extra_diamond_area := Real.pi + 12 * Real.sqrt 2
  let total_black_area := triangle_area + extra_triangle_area + diamond_area + extra_diamond_area

  total_black_area / reduced_square_area

theorem coin_overlap_black_region: 
  black_region_probability = (1 / 64) * (30 + 12 * Real.sqrt 2 + Real.pi) := 
sorry

theorem cd_sum: 
  let c := 30
  let d := 12
  c + d = 42 := 
by
  trivial

end coin_overlap_black_region_cd_sum_l742_742902


namespace train_speed_l742_742863

theorem train_speed (T1 : ℝ) (T2 : ℝ) (L_p : ℝ) (V : ℝ) :
  T1 = 25 ∧ T2 = 20 ∧ L_p = 75.006 ∧ V = (75.006 / 5) * 3.6 → V = 54.00432 :=
by
  intros _ h
  cases h with HT1 h'
  cases h' with HT2 h''
  cases h'' with HLp HV
  rw HV
  exact (75.006 / 5) * 3.6 = 54.00432
  sorry

end train_speed_l742_742863


namespace find_x_for_line_segment_l742_742843

def endpoint_coordinates (x : ℝ) : Prop :=
  let start_point := (3 : ℝ, -1 : ℝ)
  let end_point := (x, 7 : ℝ)
  let slope_positive := (7 - -1) / (x - 3) = 1
  let slope_negative := (7 - -1) / (x - 3) = -1
  let distance := (start_point.1 - end_point.1) ^ 2 + (start_point.2 - end_point.2) ^ 2 = 15^2 in
  (slope_positive ∨ slope_negative) ∧ distance

theorem find_x_for_line_segment (x : ℝ) : endpoint_coordinates x ↔ 
  (x = 3 + Real.sqrt 161 ∨ x = 3 - Real.sqrt 161 ∨ x = 3 + Real.sqrt 112.5 ∨ x = 3 - Real.sqrt 112.5) :=
sorry

end find_x_for_line_segment_l742_742843


namespace combination_5_3_eq_10_l742_742515

-- Define the combination function according to its formula
noncomputable def combination (n k : ℕ) : ℕ :=
  (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem stating the required result
theorem combination_5_3_eq_10 : combination 5 3 = 10 := by
  sorry

end combination_5_3_eq_10_l742_742515


namespace sum_of_valid_n_l742_742341

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742341


namespace tan_150_eq_neg_sqrt_3_l742_742015

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742015


namespace circle_area_equilateral_triangle_l742_742458

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742458


namespace sum_of_digits_of_largest_in_2023_consecutive_2023_l742_742316

theorem sum_of_digits_of_largest_in_2023_consecutive_2023 (n : ℤ) (h1 : ∑ k in finset.range 2023, (n - 1011 + k : ℤ) = 2023)  :
  (1012.digits.sum = 4) := 
sorry

end sum_of_digits_of_largest_in_2023_consecutive_2023_l742_742316


namespace probability_no_adjacent_same_rolls_l742_742135

-- Definitions based on the given problem
def num_faces := 8  -- number of faces on the die
def num_people := 5  -- number of people sitting around the table

-- Formal statement of the problem in Lean 4
theorem probability_no_adjacent_same_rolls :
  let probability := (441 : ℚ) / 8192 in
  ∃ p : ℚ, 
    (p = (num_faces - 1) ^ (num_people - 1) * (num_faces - 2) / num_faces ^ num_people) ∧
    (p = probability) :=
    by
    sorry

end probability_no_adjacent_same_rolls_l742_742135


namespace incorrect_statements_l742_742671

-- Definitions of the sets M and N
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3) = 0}
def N : Set ℝ := {x | (x - 4) * (x - 1) = 0}

-- Statement A
def statement_A (a : ℝ) : Prop :=
  (M a ∪ N).finite ∧ (M a ∪ N).toFinset.card = 4 → (M a ∩ N ≠ ∅)

-- Statement B
def statement_B (a : ℝ) : Prop :=
  M a ∩ N ≠ ∅ → (M a ∪ N).finite ∧ (M a ∪ N).toFinset.card = 4

-- Statement C
def statement_C (a : ℝ) : Prop :=
  (M a ∪ N) = {1, 3, 4} → M a ∩ N ≠ ∅

-- Statement D
def statement_D (a : ℝ) : Prop :=
  M a ∩ N ≠ ∅ → (M a ∪ N) = {1, 3, 4}

-- The final theorem
theorem incorrect_statements (a : ℝ) : statement_A a → statement_B a → statement_C a :=
begin
  sorry -- Proof is not required
end

end incorrect_statements_l742_742671


namespace volunteer_distribution_count_l742_742136

/-- Representation of the problem: 
Count the ways to distribute 5 volunteers into 4 communities such that:
1. Each community has at least one volunteer.
2. Volunteers A and B are not in the same community.
Note: The question is to find this count,
and it should be equal to 216.
-/
theorem volunteer_distribution_count :
  ∃ (n : ℕ), n = 216 ∧ 
  ∃ (dist : list (list ℕ)), -- list representation of the volunteers distribution in communities
    let volunteers := [1, 2, 3, 4, 5], -- assuming volunteers are labeled 1 to 5
    let A := 1, 
    let B := 2,
    dist.length = 4 ∧ -- there are 4 communities
    (∀ c ∈ dist, c ≠ []) ∧ -- each community has at least one volunteer
    (∀ c ∈ dist, ¬(A ∈ c ∧ B ∈ c)) -- A and B are not in the same community
sorry

end volunteer_distribution_count_l742_742136


namespace intersection_A_B_l742_742142

-- Define the sets A and B based on the conditions
def A (x : ℝ) : Set ℝ := {y | y = -x^2 + 2 * x - 1}
def B : Set ℝ := {x | ∃ y, y = sqrt(2 * x + 1)}

-- Prove that the intersection of A and B is the interval [-1/2, 0]
theorem intersection_A_B : A ∩ B = Icc (-1 / 2) 0 := by
  sorry

end intersection_A_B_l742_742142


namespace roots_inequality_l742_742600

theorem roots_inequality {a x1 x2 : ℝ} (h : ∀ x > 0, x ∈ set.Ioo 0 a ∨ x ∈ set.Ioi a)
  (hf_def : ∀ x > 0, f x = Real.log x + a / x)
  (hf_eq : f x1 = a / 2)
  (hf_eq_2 : f x2 = a / 2)
  (hx1_diff_x2 : x1 ≠ x2)
  (hx1_pos : 0 < x1)
  (hx2_pos : 0 < x2) :
  x1 + x2 > 2 * a :=
by
  sorry

end roots_inequality_l742_742600


namespace number_of_factors_of_n_l742_742677

theorem number_of_factors_of_n :
  let n := 2^5 * 3^3 * 5^2 * 7^4 in
  nat.factors_count n = 360 :=
by sorry

end number_of_factors_of_n_l742_742677


namespace prove_a_range_l742_742148

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - x + 2
noncomputable def g' (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem prove_a_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → 2 * f x ≤ g' x + 2) ↔ a ∈ Set.Ici (-2) :=
sorry

end prove_a_range_l742_742148


namespace tan_150_eq_neg_inv_sqrt3_l742_742076

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742076


namespace frankie_pets_l742_742947

def total_pets (cats snakes parrots dogs : ℕ) : ℕ := cats + snakes + parrots + dogs

theorem frankie_pets : ∃ (cats snakes parrots dogs : ℕ), 
  snakes = cats + 6 ∧ 
  parrots = cats - 1 ∧ 
  cats + dogs = 6 ∧ 
  dogs = 2 ∧ 
  total_pets cats snakes parrots dogs = 19 :=
by
  use 4, 10, 3, 2
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact add_comm _ _ }
  split
  { exact rfl }
  { exact rfl }

end frankie_pets_l742_742947


namespace region_area_l742_742130

open Set Real

def region := {p : ℝ × ℝ | |p.1| + |p.2| + |p.1 + p.2| ≤ 1}

theorem region_area : MeasureTheory.measureOfLebesgue.region = 3 / 4 :=
by
  sorry

end region_area_l742_742130


namespace max_segments_no_triangle_l742_742289

theorem max_segments_no_triangle (L : ℕ) (hL1 : L = 144) (n : ℕ) (hn : n > 2) :
  (∀ (a : Fin n → ℕ), (∀ i, a i ≥ 1) ∧ (∑ i, a i = L) ∧ (∀ i j k, i ≠ j → j ≠ k → k ≠ i → a i + a j > a k) → n ≤ 10) :=
by sorry

end max_segments_no_triangle_l742_742289


namespace percent_black_design_l742_742696

theorem percent_black_design : 
  ∃ (r₁ r₂ r₃ : ℝ) (pattern : ℕ → Prop),
    r₁ = 3 ∧
    r₂ = 6 ∧
    r₃ = 9 ∧
    pattern 1 ∧
    pattern 3 ∧
    (∀ n, pattern n → (mod n 2 = 1)) → 
    let
      area_black := 54 * π
      total_area := 144 * π
    in (area_black / total_area) * 100 ≈ 38 :=
begin
  sorry
end

end percent_black_design_l742_742696


namespace sum_of_valid_ns_l742_742348

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742348


namespace triangle_side_equality_l742_742154

open EuclideanGeometry

theorem triangle_side_equality
  (A B C D : Point)
  (h1 : is_triangle A B C)
  (h2 : dist A B = dist A C)
  (h3 : ∠ A = 100)
  (h4 : is_angle_bisector B D (∠ B)) :
  dist B C = dist B D + dist D A :=
by
  sorry

end triangle_side_equality_l742_742154


namespace tan_150_deg_l742_742028

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742028


namespace area_of_rectangle_l742_742855

-- Define the conditions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def area (length width : ℕ) : ℕ := length * width

-- Assumptions based on the problem conditions
variable (length : ℕ) (width : ℕ) (P : ℕ) (A : ℕ)
variable (h1 : width = 25)
variable (h2 : P = 110)

-- Goal: Prove the area is 750 square meters
theorem area_of_rectangle : 
  ∃ l : ℕ, perimeter l 25 = 110 → area l 25 = 750 :=
by
  sorry

end area_of_rectangle_l742_742855


namespace cone_height_l742_742828

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l742_742828


namespace circle_area_equilateral_triangle_l742_742457

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742457


namespace line_equation_is_correct_l742_742743

-- Define slope m as 2
def slope : ℝ := 2

-- Define y-intercept b as 4
def y_intercept : ℝ := 4

-- Define the equation of the line using the slope-intercept form
noncomputable def line_equation (x : ℝ) : ℝ := slope * x + y_intercept

-- The statement we want to prove
theorem line_equation_is_correct (x : ℝ) : line_equation x = 2 * x + 4 :=
by 
  sorry

end line_equation_is_correct_l742_742743


namespace option_A_option_B_option_C_option_D_l742_742408

theorem option_A (x : ℝ) : (1 < x ∧ x < 3) → ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2
where f := λ x : ℝ, real.sqrt (-x^2 + 2 * x + 3) :=
sorry

theorem option_B : ¬ (∀ x : ℝ, x ≥ 0 → -x^2 + 1 ≥ 0) ↔ ∃ x0 : ℝ, x0 ≥ 0 ∧ -x0^2 + 1 < 0 :=
sorry

theorem option_C (x : ℝ) : (x < -1 ∨ x > 4) → (x - 3) / (2 * x + 1) ≥ 0 :=
sorry

theorem option_D (x : ℝ) : x ≤ -1 → (∃ c : ℝ, ∀ y1 y2 : ℝ, y1 < y2 → f y1 < f y2)
where f := λ x: ℝ, -x^2 + 2 * (abs x) + 1 :=
sorry

end option_A_option_B_option_C_option_D_l742_742408


namespace sum_of_valid_ns_l742_742350

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742350


namespace geometric_sequence_solution_l742_742254

theorem geometric_sequence_solution:
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ),
    a 2 = 6 → 6 * a1 + a 3 = 30 → q > 2 →
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧
    (∀ n, S n = (3 ^ n - 1) / 2) :=
by
  intros a S q a1 h1 h2 h3
  sorry

end geometric_sequence_solution_l742_742254


namespace find_n_l742_742548

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def twin_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ q = p + 2

def is_twins_prime_sum (n p q : ℕ) : Prop :=
  twin_primes p q ∧ is_prime (2^n + p) ∧ is_prime (2^n + q)

theorem find_n :
  ∀ (n : ℕ), (∃ (p q : ℕ), is_twins_prime_sum n p q) → (n = 1 ∨ n = 3) :=
sorry

end find_n_l742_742548


namespace height_of_cone_formed_by_rolling_sector_l742_742825

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l742_742825


namespace min_value_of_cos_squared_plus_sqrt3_sin_cos_l742_742303

theorem min_value_of_cos_squared_plus_sqrt3_sin_cos :
  ∃ x, (y = cos x ^ 2 + sqrt 3 * sin x * cos x) → (y = -1 / 2) := 
sorry

end min_value_of_cos_squared_plus_sqrt3_sin_cos_l742_742303


namespace conjugate_in_first_quadrant_l742_742983

noncomputable theory

def z : ℂ := (2 - Complex.I ^ 2017) / (1 + Complex.I)

def z_conj : ℂ := conj z

-- The condition provided
variable (h : z = (2 - Complex.I ^ 2017) / (1 + Complex.I))

-- The statement to be proved
theorem conjugate_in_first_quadrant (h : z = (2 - Complex.I ^ 2017) / (1 + Complex.I)) : 
  0 < z_conj.re ∧ 0 < z_conj.im :=
sorry

end conjugate_in_first_quadrant_l742_742983


namespace average_median_eq_l742_742421

theorem average_median_eq (a b c : ℤ) (h1 : (a + b + c) / 3 = 4 * b)
  (h2 : a < b) (h3 : b < c) (h4 : a = 0) : c / b = 11 := 
by
  sorry

end average_median_eq_l742_742421


namespace tan_150_eq_neg_sqrt3_div_3_l742_742109

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742109


namespace circumcircle_and_nine_point_circle_right_angle_intersection_l742_742735

-- Definitions for angles and circles
variables {A B C : ℝ} -- Angles in radians
variables {R : ℝ} -- Radius of the circumcircle
variables {O J H : EuclideanGeometry.Point} -- Centers O (circumcenter), J (nine-point center), H (orthocenter) in 2D Euclidean space

-- Given condition on the angles in a triangle
axiom sine_sum_eq_one : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1

-- Theorem to be proven
theorem circumcircle_and_nine_point_circle_right_angle_intersection
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_angle_sum : A + B + C = π)
  (h_sin_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1)
  (h_OJ : dist O J = R * sqrt ((1:ℝ) / 2 * (1 + cos (A + B) + cos (B + C) + cos (C + A))))
  : ∠ O H J = π / 2 :=
begin
  sorry, -- Proof goes here
end

end circumcircle_and_nine_point_circle_right_angle_intersection_l742_742735


namespace find_pairs_l742_742549

theorem find_pairs (x y : ℤ) (h : 19 / x + 96 / y = (19 * 96) / (x * y)) :
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by
  sorry

end find_pairs_l742_742549


namespace dan_initial_money_l742_742527

theorem dan_initial_money 
  (cost_chocolate : ℕ) 
  (cost_candy_bar : ℕ) 
  (h1 : cost_chocolate = 3) 
  (h2 : cost_candy_bar = 7)
  (h3 : cost_candy_bar - cost_chocolate = 4) : 
  cost_candy_bar + cost_chocolate = 10 := 
by
  sorry

end dan_initial_money_l742_742527


namespace tan_150_eq_neg_sqrt3_div_3_l742_742105

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742105


namespace circumscribed_circle_area_l742_742469

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742469


namespace area_of_circumscribed_circle_l742_742451

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742451


namespace snow_probability_at_least_once_in_ten_days_l742_742705

theorem snow_probability_at_least_once_in_ten_days :
  let p1 := 1 - (1/4 : ℚ)
  let pA := 1 - (1/3 : ℚ)
  let pB := 1 - (2/5 : ℚ)
  let pC := 1 - (3/7 : ℚ)
  let pD := 1 - (1/2 : ℚ)
  let pE := 1 - (1/3 : ℚ)
  (1 : ℚ) - (p1^5 * pA * pB * pC * pD * pE) = (8879 / 8960 : ℚ) :=
by
  let p1 := 1 - (1/4 : ℚ)
  let pA := 1 - (1/3 : ℚ)
  let pB := 1 - (2/5 : ℚ)
  let pC := 1 - (3/7 : ℚ)
  let pD := 1 - (1/2 : ℚ)
  let pE := 1 - (1/3 : ℚ)
  have h : (p1^5 * pA * pB * pC * pD * pE) = (81 / 8960 : ℚ) := sorry
  rw h
  norm_num

end snow_probability_at_least_once_in_ten_days_l742_742705


namespace area_of_circumscribed_circle_l742_742483

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742483


namespace average_people_per_acre_is_0_l742_742759

-- Define the conditions
def population : ℕ := 300000000
def area_sqkm : ℕ := 4000000
def acres_per_sqkm : ℝ := 247.1

-- The statement to prove
theorem average_people_per_acre_is_0.3 :
  (population : ℝ) / (area_sqkm * acres_per_sqkm) ≈ 0.3 :=
by
  sorry

end average_people_per_acre_is_0_l742_742759


namespace no_snow_three_days_l742_742760

noncomputable def probability_no_snow_first_two_days : ℚ := 1 - 2/3
noncomputable def probability_no_snow_third_day : ℚ := 1 - 3/5

theorem no_snow_three_days : 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_third_day) = 2/45 :=
by
  sorry

end no_snow_three_days_l742_742760


namespace sum_of_valid_n_l742_742373

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742373


namespace circle_area_equilateral_triangle_l742_742462

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742462


namespace quadratic_polynomial_solution_count_l742_742959

-- Defining the polynomial and conditions in Lean
variable {R : Type*} [CommRing R]

noncomputable def f (x : R) : R := sorry  -- Assuming f(x) is a quadratic polynomial, defined elsewhere.

-- The primary theorem statement based on our analysis
theorem quadratic_polynomial_solution_count
  (hf_quad : ∃ (a b c : R), f = λ x, a*x^2 + b*x + c)
  (h_eq : ∀ x, (f(x))^3 - 4*(f(x)) = 0 → (f(x) = 0 ∨ f(x) = 2 ∨ f(x) = -2)
           (hf_zero : ∃! x, f(x) = 0)
           (hf_two : ∃! x₁ x₂, x₁ ≠ x₂ ∧ f(x₁) = 2 ∧ f(x₂) = 2)
           (hf_neg_two : ¬∃ x, f(x) = -2)) : 
  ∃! x, (f(x))^2 = 1 ∧ ∀ y, y ≠ x → (f(y))^2 = 1 :=
begin
  sorry
end

end quadratic_polynomial_solution_count_l742_742959


namespace tan_150_eq_l742_742007

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742007


namespace find_angle_BDE_l742_742205

open Lean
open Classical

namespace AngleProof

-- Definitions for the angles in a quadrilateral ABCD and point E on CD
variables {A B C D E : Type} [Quadrilateral A B C D]
variables (angle_A : ℝ) (angle_C : ℝ) (angle_DEB : ℝ) (angle_DBC : ℝ)

-- Provided conditions
def conditions : Prop :=
  angle_A = 50 ∧ angle_C = 70 ∧ angle_DEB = 30 ∧ angle_DBC = 20

-- Goal statement
theorem find_angle_BDE (cond : conditions) : ∃ angle_BDE : ℝ, angle_BDE = 130 :=
by
  sorry

end AngleProof

end find_angle_BDE_l742_742205


namespace parabola_properties_l742_742150

theorem parabola_properties (m : ℝ) :
  (∀ P : ℝ × ℝ, P = (m, 1) ∧ (P.1 ^ 2 = 4 * P.2) →
    ((∃ y : ℝ, y = -1) ∧ (dist P (0, 1) = 2))) :=
by
  sorry

end parabola_properties_l742_742150


namespace sarah_needs_gallons_of_paint_l742_742713

-- Define the conditions
def height : ℝ := 15
def diameter : ℝ := 12
def radius : ℝ := diameter / 2
def num_pillars : ℝ := 20
def coverage_per_gallon : ℝ := 320
def lateral_surface_area_of_pillar : ℝ := 2 * Real.pi * radius * height
def total_surface_area : ℝ := lateral_surface_area_of_pillar * num_pillars
def total_gallons : ℝ := total_surface_area / coverage_per_gallon

-- Prove that Sarah needs to purchase 36 gallons
theorem sarah_needs_gallons_of_paint : Real.ceil total_gallons = 36 :=
by
  sorry

end sarah_needs_gallons_of_paint_l742_742713


namespace sum_mod_200_l742_742332

theorem sum_mod_200 : 
  (∑ i in Finset.range 200, (i + 1)) % 7 = 3 := 
by 
  sorry

end sum_mod_200_l742_742332


namespace area_GHR_l742_742654

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

def area_of_triangle (A B C : Point) : ℝ :=
(abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))) / 2

theorem area_GHR :
  let P := Point.mk 0 7,
      Q := Point.mk 0 0,
      R := Point.mk 10 0,
      G := midpoint P Q,
      H := midpoint Q R in
  area_of_triangle G H R = 35 / 4 :=
by
  sorry

end area_GHR_l742_742654


namespace tan_150_deg_l742_742041

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742041


namespace product_of_integers_l742_742320

theorem product_of_integers :
  ∃ (a b c d e : ℤ), 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = {2, 6, 10, 10, 12, 14, 16, 18, 20, 24}) ∧
  (a * b * c * d * e = -3003) :=
begin
  sorry
end

end product_of_integers_l742_742320


namespace volume_relation_l742_742153

-- Define the types for points and tetrahedrons
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def Volume (A B C D : Point) : ℝ := sorry

def centroid (A B C : Point) : Point := 
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3, (A.z + B.z + C.z) / 3⟩

-- Tetrahedron structure with vertices and centroid of the base
structure Tetrahedron :=
  (A B C D : Point)
  (D1 : Point)
  (is_centroid : D1 = centroid A B C)

-- The Lean statement translating the problem
theorem volume_relation (T : Tetrahedron) 
  (A1 B1 C1 : Point)
  (H1 : ∃ k: ℝ, k ≠ 0 ∧ ∀ p, p ∈ {A, B, C} → ∀ q, q ∈ {A1, B1, C1} → (p - T.D1) = k * (q - T.D)) :
  Volume T.A T.B T.C T.D = 1 / 3 * Volume A1 B1 C1 T.D1 :=
sorry

end volume_relation_l742_742153


namespace count_odd_numbers_in_ranges_l742_742615

-- Definition: A number is odd if it can be written as 2k + 1 for some integer k
def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Conditions: The ranges we are considering
def range1 := {n : ℤ | 1 ≤ n ∧ n ≤ 6}
def range2 := {n : ℤ | 11 ≤ n ∧ n ≤ 16}

-- The problem we need to prove
theorem count_odd_numbers_in_ranges :
  ∃ count1 count2 : ℕ, 
    count1 = (range1.filter isOdd).card ∧ 
    count2 = (range2.filter isOdd).card ∧ 
    count1 + count2 = 6 :=
by
  sorry

end count_odd_numbers_in_ranges_l742_742615


namespace find_pairs_xy_l742_742926

theorem find_pairs_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : 7^x - 3 * 2^y = 1) : 
  (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
sorry

end find_pairs_xy_l742_742926


namespace brocard_inequalities_l742_742270

variables (α β γ φ : ℝ)

-- Conditions 
def brocard_angle (φ : ℝ) : Prop := φ <= π / 6
def triangle_angles (α β γ : ℝ) : Prop := α + β + γ = π

-- The proof statement
theorem brocard_inequalities
  (h1 : brocard_angle φ)
  (h2 : triangle_angles α β γ) :
  φ^3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ^3 ≤ α * β * γ :=
sorry

end brocard_inequalities_l742_742270


namespace f_zero_eq_one_f_positive_f_increasing_f_range_x_l742_742528

noncomputable def f : ℝ → ℝ := sorry
axiom f_condition1 : f 0 ≠ 0
axiom f_condition2 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_condition3 : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_positive : ∀ x : ℝ, f x > 0 :=
sorry

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
sorry

theorem f_range_x (x : ℝ) (h : f x * f (2 * x - x^2) > 1) : x ∈ { x : ℝ | f x > 1 ∧ f (2 * x - x^2) > 1 } :=
sorry

end f_zero_eq_one_f_positive_f_increasing_f_range_x_l742_742528


namespace number_of_digits_for_distinct_integers_greater_700_l742_742260

-- Definition of distinct digit count for integers greater than 700
def distinct_digit_count_greater_700 (n : ℕ) : ℕ :=
  if n > 700 then 3 * 9 * 8 else 0

theorem number_of_digits_for_distinct_integers_greater_700 : 
  ∃ d, ∀ n, n > 700 ∧ distinct_digit_count_greater_700 n = 216 → d = 3 :=
begin
  sorry
end

end number_of_digits_for_distinct_integers_greater_700_l742_742260


namespace john_recycling_income_l742_742661

theorem john_recycling_income :
  (let paper_weight_mon_sat := 8
   let paper_weight_sun := 2 * paper_weight_mon_sat
   let papers_per_day := 250
   let days := 10 * 7
   let days_mon_sat := 10 * 6
   let days_sun := 10
   let total_weight_mon_sat := days_mon_sat * papers_per_day * paper_weight_mon_sat
   let total_weight_sun := days_sun * papers_per_day * paper_weight_sun
   let total_weight := total_weight_mon_sat + total_weight_sun
   let ounces_per_ton := 32000
   let tons := total_weight / ounces_per_ton
   let money_per_ton := 20
   let total_money := tons * money_per_ton
  in total_money = 100) :=
sorry

end john_recycling_income_l742_742661


namespace next_perfect_cube_l742_742969

theorem next_perfect_cube (x : ℤ) (h : ∃ k : ℤ, x = k^2) : 
  ∃ y : ℤ, y > x ∧ y = x * int.sqrt x + 3 * x + 3 * int.sqrt x + 1 :=
by
  sorry

end next_perfect_cube_l742_742969


namespace unique_pair_solution_l742_742119

theorem unique_pair_solution:
  ∃! (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n), a^2 = 2^n + 15 ∧ a = 4 ∧ n = 0 := sorry

end unique_pair_solution_l742_742119


namespace find_y_l742_742941

theorem find_y : ∃ y : ℕ, y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ y = 14 := 
by
  sorry

end find_y_l742_742941


namespace fourth_power_evaluation_l742_742546

theorem fourth_power_evaluation : 
  let x := Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2))) in
  x ^ 4 = 2 + 2 * Real.sqrt (1 + Real.sqrt 2) + Real.sqrt 2 :=
by
  let x := Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2)))
  have h : x ^ 4 = 2 + 2 * Real.sqrt (1 + Real.sqrt 2) + Real.sqrt 2 := sorry
  exact h

end fourth_power_evaluation_l742_742546


namespace smallest_T_l742_742309

theorem smallest_T (a1 a2 a3 b1 b2 b3 c1 c2 c3 d1 d2 d3 : ℕ) 
  (h : {a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) :
  (a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 + d1 * d2 * d3) ≥ 646 :=
sorry

end smallest_T_l742_742309


namespace area_of_circumscribed_circle_l742_742476

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742476


namespace tan_150_eq_l742_742009

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742009


namespace range_of_f_l742_742596

noncomputable def f (x : ℝ) : ℝ := 2 * sin (-2 * x + π / 3) + 1

theorem range_of_f : 
  ∀ x, x ∈ Ioo (-π / 6) (π / 2) → f x ∈ Icc (-1 : ℝ) 3 := 
begin
  sorry
end

end range_of_f_l742_742596


namespace Incorrect_statements_l742_742674

section Problem

variables {a : ℕ}
def M : set ℕ := {x | (x = a ∨ x = 3)}
def N : set ℕ := {x | (x = 1 ∨ x = 4)}

theorem Incorrect_statements :
  ¬ (|M ∪ N| = 4 → M ∩ N ≠ ∅) ∧
  ¬ (M ∩ N ≠ ∅ → |M ∪ N| = 4) ∧
  ¬ ({1, 3, 4} ⊆ M ∪ N → M ∩ N ≠ ∅) :=
by sorry

end Problem

end Incorrect_statements_l742_742674


namespace sum_of_other_endpoint_coordinates_l742_742264

theorem sum_of_other_endpoint_coordinates (x y : ℝ) (hx : (x + 5) / 2 = 3) (hy : (y - 2) / 2 = 4) :
  x + y = 11 :=
sorry

end sum_of_other_endpoint_coordinates_l742_742264


namespace sum_binom_solutions_l742_742388

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742388


namespace distinct_permutations_BEARS_l742_742617

theorem distinct_permutations_BEARS : 
    (∃ word : String, word = "BEARS" ∧ letter_count word = 5) →
    permutations_count 5 = 120 :=
by
  sorry

end distinct_permutations_BEARS_l742_742617


namespace weight_of_replaced_person_l742_742287

theorem weight_of_replaced_person
  (average_increase : Real)
  (num_persons : Nat)
  (weight_new_person : Real)
  (total_increase_in_weight : Real) :
  average_increase = 2.5 →
  num_persons = 6 →
  weight_new_person = 80 →
  total_increase_in_weight = num_persons * average_increase →
  (weight_new_person - total_increase_in_weight) = 65 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end weight_of_replaced_person_l742_742287


namespace height_of_cone_formed_by_rolling_sector_l742_742823

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l742_742823


namespace ellipse_find_equation_slope_of_line_MN_l742_742967

-- Define the ellipse and given conditions
def ellipse_equation (x y a b : ℝ) : Prop := 
  x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1

def is_eccentricity (a c : ℝ) : Prop := 
  c / a = 1 / 2

def max_area_triangle (a b : ℝ) : Prop := 
  sqrt (a^2 - b^2) * b = sqrt 3

-- First part: proving the equation of the ellipse
theorem ellipse_find_equation {a b : ℝ} 
  (h1 : a > b) (h2 : b > 0) (h3 : is_eccentricity a (sqrt (a^2 - b^2)))
  (h4 : max_area_triangle a b) :
  ellipse_equation x y a b ↔ ellipse_equation x y 2 (sqrt 3) := 
  sorry

-- Second part: proving the slope of line MN is constant
theorem slope_of_line_MN {a b : ℝ} {F2 A B M N : ℝ × ℝ}
  (h1 : ellipse_equation F2.1 F2.2 a b) 
  (h2 : F2.1 = 0) -- F2 is on the y-axis
  (h3 : ∀ M N, ¬ (M.1 = N.1 ∧ M.2 = N.2) → angle M A B = angle N A B) :
  ∃ k, ∀ M N, line (M, N).slope = k := 
  sorry

end ellipse_find_equation_slope_of_line_MN_l742_742967


namespace rectangle_area_is_correct_l742_742273

-- Define all necessary terms and conditions
def radius_of_semicircle := 17
def DA := 20
def FD := 7
def AE := 7

-- Define the main theorem to prove the area of rectangle
theorem rectangle_area_is_correct :
  let r := radius_of_semicircle in
  let d := DA in
  let fd := FD in
  let ae := AE in
  d * (3 * Real.sqrt 21) = 60 * Real.sqrt 21 :=
by
  sorry

end rectangle_area_is_correct_l742_742273


namespace cyclic_inequality_l742_742568

theorem cyclic_inequality (a b c : ℝ) :
  abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) 
  ≤ (9 * real.sqrt 2) / 32 * (a^2 + b^2 + c^2)^2 := by
    sorry

end cyclic_inequality_l742_742568


namespace scientific_notation_of_0_00065_l742_742291

/-- 
Prove that the decimal representation of a number 0.00065 can be expressed in scientific notation 
as 6.5 * 10^(-4)
-/
theorem scientific_notation_of_0_00065 : 0.00065 = 6.5 * 10^(-4) := 
by 
  sorry

end scientific_notation_of_0_00065_l742_742291


namespace correct_card_ordering_l742_742789

structure CardOrder where
  left : String
  middle : String
  right : String

def is_right_of (a b : String) : Prop := (a = "club" ∧ (b = "heart" ∨ b = "diamond")) ∨ (a = "8" ∧ b = "4")

def is_left_of (a b : String) : Prop := a = "5" ∧ b = "heart"

def correct_order : CardOrder :=
  { left := "5 of diamonds", middle := "4 of hearts", right := "8 of clubs" }

theorem correct_card_ordering : 
  ∀ order : CardOrder, 
  is_right_of order.right order.middle ∧ is_right_of order.right order.left ∧ is_left_of order.left order.middle 
  → order = correct_order := 
by
  intro order
  intro h
  sorry

end correct_card_ordering_l742_742789


namespace hyperbola_eccentricity_range_l742_742524

variable (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 < b / a) (h₄ : b / a < 2)

theorem hyperbola_eccentricity_range : 
  let e := Real.sqrt (1 + (b / a)^2) in 
  sqrt 2 < e ∧ e < sqrt 5 := 
by
  -- First, we introduce the conditions into the context
  use h₁ h₂ h₃ h₄
  -- Then we introduce the definition of eccentricity into the context
  let e := Real.sqrt (1 + (b / a)^2)
  -- Since we know 1 < b/a < 2, we can square them
  have h : 1 < (b/a)^2 → (b/a)^2 < 4, from sorry
  -- Adding 1 to both sides of our inequality
  have h' : 2 < 1 + (b/a)^2 ∧ 1 + (b/a)^2 < 5, from sorry
  -- Take square roots
  have h'' : sqrt 2 < Real.sqrt (1 + (b/a)^2) → Real.sqrt (1 + (b/a)^2) < sqrt 5, from sorry
  exact ⟨h₃, h₄⟩ -- thus we obtain the final result that sqrt 2 < e < sqrt 5

end hyperbola_eccentricity_range_l742_742524


namespace cone_height_is_correct_l742_742821

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l742_742821


namespace area_of_room_in_square_inches_l742_742779

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l742_742779


namespace second_player_wins_l742_742785

theorem second_player_wins (initial_stones : ℕ := 50) (target_stones : ℕ := 200) (move_range : set ℕ := {n | 1 ≤ n ∧ n ≤ 9}) :
  ∃ strategy : (ℕ → ℕ) → ℕ → ℕ, (∀ x ∈ move_range, ∀ y ∈ move_range, strategy x (200 - (initial_stones + x + y)) = 200 - y) :=
sorry

end second_player_wins_l742_742785


namespace circumscribed_circle_area_l742_742470

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742470


namespace angle_APB_60_l742_742905

variable (P : ℝ × ℝ) (A B : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop)
variable (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop)

-- Define the circle C: (x - 6)^2 + (y - 2)^2 = 5
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 2)^2 = 5

-- Define the line l: y = 2x
def line_l (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Condition: Tangents from P to the circle C are symmetric with respect to l
def symmetric_tangents (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  ∀ Q : ℝ × ℝ, l1 P Q → l2 P (reflection_over_line_l P Q)

-- Define the reflection of a point over the line y = 2x
def reflection_over_line_l (P Q : ℝ × ℝ) : ℝ × ℝ :=
  let a := (2 * Q.1 + Q.2) / 5
  let b := (4 * Q.1 - 2 * Q.2) / 5
  in (a, b)

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem angle_APB_60 (P A B : ℝ × ℝ) (hC : ∀ x y, C x y → circle_C x y)
  (hl : line_l P) (hl12 : symmetric_tangents l1 l2 P) :
  angle A P B = 60 :=
sorry

end angle_APB_60_l742_742905


namespace amount_of_money_C_l742_742502

variable (A B C : ℝ)

theorem amount_of_money_C (h1 : A + B + C = 500)
                         (h2 : A + C = 200)
                         (h3 : B + C = 360) :
    C = 60 :=
sorry

end amount_of_money_C_l742_742502


namespace find_k_l742_742981

theorem find_k 
  (h : ∀ x k : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0):
  ∃ (k : ℝ), k = -2 :=
sorry

end find_k_l742_742981


namespace carly_dog_count_l742_742894

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l742_742894


namespace tan_150_eq_l742_742004

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742004


namespace tan_150_deg_l742_742033

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742033


namespace circumscribed_circle_area_l742_742467

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742467


namespace point_on_line_eq_l742_742589

theorem point_on_line_eq (a b : ℝ) (h : b = -3 * a - 4) : b + 3 * a + 4 = 0 :=
by
  sorry

end point_on_line_eq_l742_742589


namespace problem_statement_l742_742692

open set

variable U : set ℝ
variable A B C : set ℝ

-- Define the sets
def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 ≤ x ∧ x ≤ 4}
def C := {x : ℝ | 3 < x ∧ x ≤ 4}

-- Complement of A with respect to U (assume U = ℝ)
def C_U_A := {x : ℝ | x < 1 ∨ x > 3}

-- Statement to prove
theorem problem_statement : C = C_U_A ∩ B := by 
  exact sorry

end problem_statement_l742_742692


namespace ratio_proof_problem_l742_742762

variables {a b c d : ℚ}

theorem ratio_proof_problem
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 2)
  (h3 : d / b = 2 / 5) : 
  a / c = 25 / 8 := 
sorry

end ratio_proof_problem_l742_742762


namespace scalar_product_condition_l742_742620

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b c : V)

theorem scalar_product_condition (h : a ≠ 0 ∧ (b - c) ≠ 0) (h_ab_ac : inner a b = inner a c) :
  inner a (b - c) = 0 :=
by
  sorry

end scalar_product_condition_l742_742620


namespace consecutive_arithmetic_sequence_n1_arithmetic_sequence_n1_1_maximum_arithmetic_sequence_t_l742_742991

noncomputable def a (n : ℕ) : ℤ := 2^n - (-1)^n

-- Problem 1 part 1: Prove that if n1, n2, n3 are consecutive positive integers forming an arithmetic sequence, then n1 = 2
theorem consecutive_arithmetic_sequence_n1 (n1 n2 n3 : ℕ) (h1: n1 + 1 = n2) (h2: n2 + 1 = n3)
  (h : 2 * a n2 = a n1 + a n3) : n1 = 2 :=
sorry

-- Problem 1 part 2: If n1 = 1 and a1, a_n2, a_n3 form an arithmetic sequence, then n3 - n2 = 1
theorem arithmetic_sequence_n1_1 (n2 n3 : ℕ) (h1: n2 > 1) (h : 2 * a n2 = a 1 + a n3) : n3 - n2 = 1 :=
sorry

-- Problem 2: Prove that the maximum value of t for which the sequence can form an arithmetic sequence is 3
theorem maximum_arithmetic_sequence_t (t : ℕ) (seq : list ℕ)
  (h1: ∀ (i : ℕ), i < (seq.length - 1) → 2 * a (seq.nth_le i sorry) = a (seq.nth_le (i - 1) sorry) + a (seq.nth_le (i + 1) sorry))
  (h2: seq.length = t) : t ≤ 3 :=
sorry

end consecutive_arithmetic_sequence_n1_arithmetic_sequence_n1_1_maximum_arithmetic_sequence_t_l742_742991


namespace simplify_expression_l742_742721

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l742_742721


namespace triangle_side_lengths_l742_742649

noncomputable def sides_of_triangle (a b c : ℝ) : Prop :=
c + b = 14 ∧
  (∃ α : ℝ, (tan (α / 2) = 1 / 2) ∧ a^2 = c^2 - b^2)

theorem triangle_side_lengths : sides_of_triangle 7 5.25 8.75 :=
by
  unfold sides_of_triangle
  split
  -- Proof for condition c + b = 14
  · exact rfl
  use 2 * atan (1 / 2)
  split
  -- Proof for condition tan (α / 2) = 1 / 2
  · exact tan_atan (1 / 2)
  -- Proof for condition a^2 = c^2 - b^2
  · sorry

end triangle_side_lengths_l742_742649


namespace crescentWithoutTriangle_l742_742650

-- Definitions based on given conditions
def numberOfLetters : ℕ := 120
def bothCrescentAndTriangle : ℕ := 32
def triangleWithoutCrescent : ℕ := 72

-- Lean statement for the proof problem
theorem crescentWithoutTriangle : ∃ C crescentWithoutTriangle, 
  (bothCrescentAndTriangle + triangleWithoutCrescent = 104) ∧
  (numberOfLetters = C + triangleWithoutCrescent) ∧
  (crescentWithoutTriangle = C - bothCrescentAndTriangle) ∧
  (crescentWithoutTriangle = 16) :=
by
  -- Solution steps outline
  let T := bothCrescentAndTriangle + triangleWithoutCrescent
  have : T = 104 := by decide
  have C := numberOfLetters - triangleWithoutCrescent
  have : C = 48 := by decide
  let crescentWithoutTriangle := C - bothCrescentAndTriangle
  have : crescentWithoutTriangle = 16 := by decide
  existsi C, crescentWithoutTriangle,
  split, exact this,
  split, exact this,
  split, exact this,
  reflexivity

end crescentWithoutTriangle_l742_742650


namespace area_of_circumscribed_circle_l742_742449

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742449


namespace find_angle_A_find_perimeter_l742_742657

noncomputable theory

-- Part 1: Proving angle A
theorem find_angle_A 
  (a b c : ℝ)
  (h1 : 2 * a * cos B = 2 * c + b)
  (A B C : ℝ)
  (ha : a = b * sin A / sin B)
  (hb : b = a * sin B / sin A)
  (hc : c = a * sin C / sin A)
  (area : 1/2 * b * c * sin A = 2 * sqrt 3)
  (angle_sum : A + B + C = π)
  : A = 2 * π / 3 :=
sorry

-- Part 2: Proving the perimeter given b - c = 2
theorem find_perimeter 
  (a b c : ℝ)
  (h1 : 2 * a * cos B = 2 * c + b)
  (A B C : ℝ)
  (ha : a = b * sin A / sin B)
  (hb : b = a * sin B / sin A)
  (hc : c = a * sin C / sin A)
  (area : 1/2 * b * c * sin A = 2 * sqrt 3)
  (b_minus_c : b - c = 2)
  (angle_sum : A + B + C = π)
  (A_is_2pi_3 : A = 2 * π / 3)
  : a + b + c = 6 + 2 * sqrt 7 :=
sorry

end find_angle_A_find_perimeter_l742_742657


namespace height_pillar_D_correct_l742_742868

def height_of_pillar_at_D (h_A h_B h_C : ℕ) (side_length : ℕ) : ℕ :=
17

theorem height_pillar_D_correct :
  height_of_pillar_at_D 15 10 12 10 = 17 := 
by sorry

end height_pillar_D_correct_l742_742868


namespace problem_statement_l742_742683

-- Define the problem parameters
def a : ℕ := 4  -- Number of multiples of 8 less than 40
def b : ℕ := 9  -- Number of multiples of 4 (and thus 2) less than 40

-- Define the theorem to prove the desired result
theorem problem_statement : (a - b) ^ 3 = -125 :=
by sorry

end problem_statement_l742_742683


namespace find_constants_PQR_l742_742128

theorem find_constants_PQR :
  ∃ P Q R, P = 14 ∧ Q = -11 ∧ R = -8 ∧
  (∀ x, x ≠ 4 ∧ x ≠ 2 →
  (3 * x^2 + 2 * x) / ((x - 4) * (x - 2)^2) =
  P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) :=
by
  let P := 14
  let Q := -11
  let R := -8
  use P, Q, R
  split; try {rfl}
  intros x hx
  have h1 : x ≠ 4 := hx.1
  have h2 : x ≠ 2 := hx.2
  -- sorry is used to skip the actual proof steps.
  sorry

end find_constants_PQR_l742_742128


namespace tan_150_deg_l742_742029

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742029


namespace polynomial_remainder_l742_742936

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ), (3 * X^5 - 2 * X^3 + 5 * X - 9) = (X - 1) * (X - 2) * q + (92 * X - 95) :=
by
  intro q
  sorry

end polynomial_remainder_l742_742936


namespace area_of_circumscribed_circle_l742_742452

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742452


namespace ship_passengers_round_trip_tickets_l742_742261

theorem ship_passengers_round_trip_tickets (total_passengers : ℕ) (p1 : ℝ) (p2 : ℝ) :
  (p1 = 0.25 * total_passengers) ∧ (p2 = 0.6 * (p * total_passengers)) →
  (p * total_passengers = 62.5 / 100 * total_passengers) :=
by
  sorry

end ship_passengers_round_trip_tickets_l742_742261


namespace range_of_x_l742_742169

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then real.exp (-x) - real.exp x + real.exp 1 else real.exp 1

theorem range_of_x (x : ℝ) : -1 < x ∧ x < real.sqrt 2 ↔ f(x^2 - 2) > f(x) :=
sorry

end range_of_x_l742_742169


namespace number_of_search_plans_l742_742214

-- Define the conditions as hypotheses
def cute_kids : Finset String :=
  Finset.ofList ["Kid1", "Kid2", "Kid3", "Kid4", "Kid5", "Kid6", "Kid7", "Grace"]

axiom total_kids_count : cute_kids.card = 8
axiom two_locations : True -- Placeholder for the airdrop locations
axiom grace_conditions : ∀ k ∈ cute_kids.erase "Grace", k ∉ cute_kids ∨ k = "Grace" ∨ (cute_kids.count k ≤ 1) -- Placeholder for Grace's conditions
axiom evenly_split : True -- Placeholder stating that kids must be evenly split

-- The proof problem
theorem number_of_search_plans : 
  (number_of_ways Grace participates in nearer location + number_of_ways Grace stays at the base camp) = 175 := 
sorry

end number_of_search_plans_l742_742214


namespace CarlyWorkedOnElevenDogs_l742_742891

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l742_742891


namespace average_transformation_l742_742962

variable {α : Type*}

theorem average_transformation (x1 x2 x3 x4 : α) [div_ring α] [has_add α] [has_mul α] [has_one α] [has_neg α] [has_sub α]
  (hx : (x1 + x2 + x3 + x4) / 4 = 3) :
  ((2 * x1 - 3) + (2 * x2 - 3) + (2 * x3 - 3) + (2 * x4 - 3)) / 4 = 3 :=
by
  sorry

end average_transformation_l742_742962


namespace tan_150_degrees_l742_742052

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742052


namespace max_value_of_ab_ab_tangent_to_circle_l742_742555

theorem max_value_of_ab_ab_tangent_to_circle (a b : ℝ) (h : a^2 + b^2 = 1) : 
  a + b + a * b ≤ sqrt 2 + 1 / 2 := 
sorry

end max_value_of_ab_ab_tangent_to_circle_l742_742555


namespace smaller_cylinder_diameter_l742_742290

theorem smaller_cylinder_diameter
  (vol_large : ℝ)
  (height_large : ℝ)
  (diameter_large : ℝ)
  (height_small : ℝ)
  (ratio : ℝ)
  (π : ℝ)
  (volume_large_eq : vol_large = π * (diameter_large / 2)^2 * height_large)  -- Volume formula for the larger cylinder
  (ratio_eq : ratio = 74.07407407407408) -- Given ratio
  (height_large_eq : height_large = 10)  -- Given height of the larger cylinder
  (diameter_large_eq : diameter_large = 20)  -- Given diameter of the larger cylinder
  (height_small_eq : height_small = 6)  -- Given height of smaller cylinders):
  :
  ∃ (diameter_small : ℝ), diameter_small = 3 := 
by
  sorry

end smaller_cylinder_diameter_l742_742290


namespace intersection_value_of_a_l742_742158

theorem intersection_value_of_a (a : ℝ) (A B : Set ℝ) 
  (hA : A = {0, 1, 3})
  (hB : B = {a + 1, a^2 + 2})
  (h_inter : A ∩ B = {1}) : 
  a = 0 :=
by
  sorry

end intersection_value_of_a_l742_742158


namespace probability_of_at_least_one_vowel_l742_742323

def set1 := {'a', 'b', 'c', 'd', 'e'}
def set2 := {'k', 'l', 'm', 'n', 'o', 'p'}
def vowels_set1 := {'a', 'e'}
def vowels_set2 := {'o'}

theorem probability_of_at_least_one_vowel :
  let prob_no_vowels_set1 := (3 / 5 : ℚ) in
  let prob_no_vowels_set2 := (4 / 6 : ℚ) in
  let prob_no_vowels := prob_no_vowels_set1 * prob_no_vowels_set2 in
  let prob_at_least_one_vowel := 1 - prob_no_vowels in
  prob_at_least_one_vowel = (3 / 5 : ℚ) :=
by
  sorry

end probability_of_at_least_one_vowel_l742_742323


namespace second_derivative_l742_742631

noncomputable def y (x : ℝ) : ℝ := x^3 + Real.log x / Real.log 2 + Real.exp (-x)

theorem second_derivative (x : ℝ) : (deriv^[2] y x) = 3 * x^2 + (1 / (x * Real.log 2)) - Real.exp (-x) :=
by
  sorry

end second_derivative_l742_742631


namespace complex_quadrant_l742_742653

theorem complex_quadrant (i : ℂ) (h1 : i^2 = -1) : 3 where
  sorry

end complex_quadrant_l742_742653


namespace value_at_minus_one_l742_742601

def piecewise_function (x : ℝ) : ℝ :=
if x > 0 then 2 * x + 2 else 2

theorem value_at_minus_one : piecewise_function (-1) = 2 :=
by
  -- proof will go here
  sorry

end value_at_minus_one_l742_742601


namespace circle_tangent_l742_742149

/-- Given a circle with equation x^2 + y^2 - 2*x + m*y = 0, with its center on the line y = x,
find the value of m and the equation of line l that is tangent to the circle and passes through (-1,1). -/
theorem circle_tangent
  (m : ℝ)
  (C : ∀ x y : ℝ, x^2 + y^2 - 2 * x + m * y = 0)
  (center_on_line : ∀ x : ℝ, x = -m / 2 → ∃ y : ℝ, y = x)
  (tangent_line : ∀ x y k : ℝ, y - 1 = k * (x + 1)) :
  m = -2 ∧ (tangent_line x y 1 → x - y + 2 = 0) ∧ (tangent_line x y (-1) → x + y = 0) :=
sorry

end circle_tangent_l742_742149


namespace probability_entire_grid_black_l742_742815

noncomputable def prob_all_black_grid : ℚ :=
  -- Initial probabilities of each unit square being black
  let p_black : ℚ := 1 / 2 in
  -- Probability that both squares in one diagonal are initially black
  let p_diag : ℚ := p_black * p_black in
  -- Probability that the entire grid is black after the process
  p_diag * p_diag

theorem probability_entire_grid_black : prob_all_black_grid = 1 / 16 := 
  sorry

end probability_entire_grid_black_l742_742815


namespace sum_arithmetic_sequence_S12_l742_742159

variable {a : ℕ → ℝ} -- Arithmetic sequence a_n
variable {S : ℕ → ℝ} -- Sum of the first n terms S_n

-- Conditions given in the problem
axiom condition1 (n : ℕ) : S n = (n / 2) * (a 1 + a n)
axiom condition2 : a 4 + a 9 = 10

-- Proving that S 12 = 60 given the conditions
theorem sum_arithmetic_sequence_S12 : S 12 = 60 := by
  sorry

end sum_arithmetic_sequence_S12_l742_742159


namespace translate_graph_right_l742_742632

theorem translate_graph_right (x : ℝ) :
  let y1 := λ x : ℝ, sin (2 * x)
  let y2 := λ x : ℝ, sin (2 * (x - π / 3))
  y1 (x - π / 3) = y2 x := 
by
  intros
  sorry

end translate_graph_right_l742_742632


namespace triangle_median_perpendicular_l742_742698

variables {D E F P Q : Type}
variables [IsoscelesTriangle D E F] (DE EF : ℚ)
variables (DP EQ : ℚ)
variables (perpendicular_medians : ∀ dp eq : ℚ, MediansPerpendicular D E F dp eq) 

def DE_value : ℚ := 70 / 3

theorem triangle_median_perpendicular (h_isosceles: DE = EF)
 (h_DP : DP = 21) 
 (h_EQ : EQ = 28) 
 (h_perpendicular : perpendicular_medians DP EQ) :
 DE = DE_value := 
sorry

end triangle_median_perpendicular_l742_742698


namespace relation_between_3a5_3b5_l742_742565

theorem relation_between_3a5_3b5 (a b : ℝ) (h : a > b) : 3 * a + 5 > 3 * b + 5 := by
  sorry

end relation_between_3a5_3b5_l742_742565


namespace calculate_total_cost_l742_742539

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end calculate_total_cost_l742_742539


namespace min_sales_required_l742_742877

-- Definitions from conditions
def old_salary : ℝ := 75000
def new_base_salary : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750

-- Statement to be proven
theorem min_sales_required (n : ℕ) :
  n ≥ ⌈(old_salary - new_base_salary) / (commission_rate * sale_amount)⌉₊ :=
sorry

end min_sales_required_l742_742877


namespace total_sticks_needed_l742_742718

/-
Given conditions:
1. Simon's raft needs 36 sticks.
2. Gerry's raft needs two-thirds of the number of sticks that Simon needs.
3. Micky's raft needs 9 sticks more than Simon and Gerry's rafts combined.

Prove that the total number of sticks collected by Simon, Gerry, and Micky is 129.
-/

theorem total_sticks_needed :
  let S := 36 in
  let G := (2/3) * S in
  let M := S + G + 9 in
  S + G + M = 129 :=
by
  let S := 36
  let G := (2/3) * S
  let M := S + G + 9
  have : S + G + M = 129 := sorry
  exact this

end total_sticks_needed_l742_742718


namespace sum_f_l742_742583

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def periodic_property (f : ℝ → ℝ) := ∀ x : ℝ, f (1 - x) = f (1 + x)

theorem sum_f (f : ℝ → ℝ) (h_odd : is_odd_function f) 
  (h_periodic : periodic_property f) (hf1 : f 1 = 2) :
  (finset.range 50).sum (λ n, f (n + 1)) = 2 := 
sorry

end sum_f_l742_742583


namespace part_I_part_II_1_part_II_2_part_II_3_l742_742986

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem part_I (a : ℝ) (h : a ≤ 1) : ∀ x : ℝ, x ≥ 1 → x^2 ≥ f a x :=
by
  assume x hx
  sorry

theorem part_II_1 (a : ℝ) (h : a > 0) (h_gt : a > 1 / Real.exp 1) : ¬∃ x : ℝ, f a x = 0 :=
by
  sorry

theorem part_II_2 (a : ℝ) (h : a > 0) (h_eq : a = 1 / Real.exp 1) : ∃! x : ℝ, f a x = 0 :=
by
  sorry

theorem part_II_3 (a : ℝ) (h : a > 0) (h_lt : a < 1 / Real.exp 1) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 :=
by
  sorry

end part_I_part_II_1_part_II_2_part_II_3_l742_742986


namespace sum_of_ns_l742_742396

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742396


namespace apples_distribution_l742_742919

variable (x : ℕ)

theorem apples_distribution :
  0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8 :=
sorry

end apples_distribution_l742_742919


namespace AT_parallel_BC_proof_l742_742954

open EuclideanGeometry

-- Define the given points and lines in the geometric context
variable {A B C D E F G T : Point}
variable {Γ : Circle}
variable {triangle_ABC : Triangle}

-- Define the intersections and geometric conditions
variable (circumcircle_BDF : Circle)
variable (circumcircle_CEG : Circle)

-- Conditions and assumptions 
axiom Gamma_passes_through_A : Γ.contains A
axiom Gamma_intersects_AB_at_D : Γ.intersection_with_AB A B = D
axiom Gamma_intersects_AC_at_E : Γ.intersection_with_AC A C = E
axiom Gamma_intersects_BC_at_FG (hf : F ∈ Interval B G) : Γ.contains F ∧ Γ.contains G
axiom tangency_at_F (hf : IsTangent circumcircle_BDF F)
axiom tangency_at_G (hg : IsTangent circumcircle_CEG G)
axiom tangency_intersection_not_coincide_A : T ≠ A

-- Conclusion to prove
theorem AT_parallel_BC_proof : Parallel (Line.mk A T) (Line.mk B C) :=
by
  sorry

end AT_parallel_BC_proof_l742_742954


namespace find_k_l742_742992

theorem find_k (k : ℕ) : 
  let A := {1, 2, k}
      B := {2, 5}
  in A ∪ B = {1, 2, 3, 5} → k = 3 := 
by
  intros h
  sorry

end find_k_l742_742992


namespace leading_coefficient_poly_l742_742914

def poly := 5 * (x^5 - 2 * x^4 + 3 * x^3) - 8 * (x^5 + x^4 - x^2 + 1) + 3 * (3 * x^5 - x^4 + x)

theorem leading_coefficient_poly : leadingCoeff (poly) = 6 := by
  sorry

end leading_coefficient_poly_l742_742914


namespace line_BC_equation_l742_742217

noncomputable theory

open_locale classical

def A : ℝ × ℝ := (1, 4)

def angle_bisector_ABC := λ x y : ℝ, x - 2 * y = 0
def angle_bisector_ACB := λ x y : ℝ, x + y - 1 = 0

theorem line_BC_equation :
  ∃ (BC : ℝ → ℝ → Prop), (∀ x y, BC x y ↔ 4 * x + 17 * y + 12 = 0) :=
by
  let A' := (19 / 5 : ℝ, -8 / 5 : ℝ),
  let A'' := (-3 : ℝ, 0 : ℝ),
  use (λ x y, 4 * x + 17 * y + 12 = 0),
  sorry

end line_BC_equation_l742_742217


namespace range_of_fx₂_l742_742594

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + a * Real.log x

def is_extreme_point (a x : ℝ) : Prop := 
  (2 * x^2 - 2 * x + a) / x = 0

theorem range_of_fx₂ (a x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 2) 
  (h₂ : 0 < x₁ ∧ x₁ < x₂) (h₃ : is_extreme_point a x₁)
  (h₄ : is_extreme_point a x₂) : 
  (f a x₂) ∈ (Set.Ioo (-(3 + 2 * Real.log 2) / 4) (-1)) :=
sorry

end range_of_fx₂_l742_742594


namespace tan_150_eq_neg_inv_sqrt3_l742_742058

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742058


namespace area_of_room_in_square_inches_l742_742778

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l742_742778


namespace count_numbers_with_digit_5_6_in_first_512_l742_742616

def has_digit_5_6 (n : ℕ) : Prop :=
  (∃ k, k < n ∧ (Nat.digit n 8 k = 5 ∨ Nat.digit n 8 k = 6))

theorem count_numbers_with_digit_5_6_in_first_512 :
  (Finset.range 512).filter has_digit_5_6).card = 296 :=
by
  sorry

end count_numbers_with_digit_5_6_in_first_512_l742_742616


namespace regular_seven_pointed_star_angle_measure_l742_742858

-- Define the problem conditions and expected answer
def regular_seven_pointed_star_angle (α : ℝ) : Prop :=
  let γ := 2 * 360 / 7 in
  2 * α + γ = 180

noncomputable def angle_alpha : ℝ := 270 / 7

theorem regular_seven_pointed_star_angle_measure : regular_seven_pointed_star_angle angle_alpha := 
by
  unfold regular_seven_pointed_star_angle
  have h1 : 2 * angle_alpha = 180 - 2 * 360 / 7 := sorry
  exact h1

end regular_seven_pointed_star_angle_measure_l742_742858


namespace compute_f_f_f_19_l742_742250

def f (x : Int) : Int :=
  if x < 10 then x^2 - 9 else x - 15

theorem compute_f_f_f_19 : f (f (f 19)) = 40 := by
  sorry

end compute_f_f_f_19_l742_742250


namespace angle_sum_proof_l742_742647

theorem angle_sum_proof (x y : ℝ) (h : 3 * x + 6 * x + (x + y) + 4 * y = 360) : x = 0 ∧ y = 72 :=
by {
  sorry
}

end angle_sum_proof_l742_742647


namespace area_of_circumscribed_circle_l742_742472

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742472


namespace tan_150_eq_neg_inv_sqrt_3_l742_742091

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742091


namespace fraction_of_boys_participated_l742_742434

-- Definitions based on given conditions
def total_students (B G : ℕ) : Prop := B + G = 800
def participating_girls (G : ℕ) : Prop := (3 / 4 : ℚ) * G = 150
def total_participants (P : ℕ) : Prop := P = 550
def participating_girls_count (PG : ℕ) : Prop := PG = 150

-- Definition of the fraction of participating boys
def fraction_participating_boys (X : ℚ) (B : ℕ) (PB : ℕ) : Prop := X * B = PB

-- The problem of proving the fraction of boys who participated
theorem fraction_of_boys_participated (B G PB : ℕ) (X : ℚ)
  (h1 : total_students B G)
  (h2 : participating_girls G)
  (h3 : total_participants 550)
  (h4 : participating_girls_count 150)
  (h5 : PB = 550 - 150) :
  fraction_participating_boys X B PB → X = 2 / 3 := by
  sorry

end fraction_of_boys_participated_l742_742434


namespace area_of_triangle_ABC_l742_742610

-- Define point vectors 
structure Vector2 :=
  (i : ℝ)
  (j : ℝ)

def vector_sub (v1 v2 : Vector2) : Vector2 :=
  ⟨v1.i - v2.i, v1.j - v2.j⟩

-- Length squared of a vector
def length_squared (v : Vector2) : ℝ :=
  v.i * v.i + v.j * v.j

-- Calculate the area of the triangle given the vectors representing the sides
def triangle_area (v1 v2 : Vector2) : ℝ :=
  0.5 * real.sqrt (length_squared v1) * real.sqrt (length_squared v2)

-- Given vectors
def AB : Vector2 := ⟨4, 3⟩
def AC : Vector2 := ⟨-3, 4⟩

-- Proof statement
theorem area_of_triangle_ABC :
  triangle_area AB AC = 25 / 2 :=
by 
  -- Context and conditions are implicitly understood
  sorry

end area_of_triangle_ABC_l742_742610


namespace trees_occupy_area_l742_742847

theorem trees_occupy_area
  (length : ℕ) (width : ℕ) (number_of_trees : ℕ)
  (h_length : length = 1000)
  (h_width : width = 2000)
  (h_trees : number_of_trees = 100000) :
  (length * width) / number_of_trees = 20 := 
by
  sorry

end trees_occupy_area_l742_742847


namespace proof_problem_l742_742122

open Set

def Point : Type := ℝ × ℝ

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

def area_of_triangle (T : Triangle) : ℝ :=
   0.5 * abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2))

def area_of_grid (length width : ℝ) : ℝ :=
   length * width

def problem_statement : Prop :=
   let T : Triangle := {A := (1,3), B := (5,1), C := (4,4)} 
   let S1 := area_of_triangle T
   let S := area_of_grid 6 5
   (S1 / S) = 1 / 6

theorem proof_problem : problem_statement := 
by
  sorry


end proof_problem_l742_742122


namespace sequence_diff_exists_l742_742665

theorem sequence_diff_exists (x : ℕ → ℕ) (h1 : x 1 = 1) (h2 : ∀ n : ℕ, 1 ≤ n → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_exists_l742_742665


namespace tan_150_eq_neg_inv_sqrt3_l742_742070

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742070


namespace modulo_problem_l742_742246

theorem modulo_problem (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 29) (h3 : 5 * n % 29 = 1) : (3 ^ n) ^ 2 - 3 % 29 = 13 % 29 :=
by
  sorry

end modulo_problem_l742_742246


namespace cone_height_is_correct_l742_742822

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l742_742822


namespace binom_sum_l742_742378

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742378


namespace sum_series_l742_742686

noncomputable def b : ℕ → ℝ
| 0     => 2
| 1     => 2
| (n+2) => b (n+1) + b n

theorem sum_series : (∑' n, b n / 3^(n+1)) = 1 / 3 := by
  sorry

end sum_series_l742_742686


namespace count_not_divisible_by_4_l742_742138

theorem count_not_divisible_by_4 : 
  (∃ S : Finset ℕ, S.card = 23 ∧ ∀ n ∈ S, n ≤ 1200 ∧ 
   (⌊1198 / n⌋ + ⌊1199 / n⌋ + ⌊1200 / n⌋) % 4 ≠ 0) :=
sorry

end count_not_divisible_by_4_l742_742138


namespace sum_of_ns_l742_742399

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742399


namespace complex_number_solution_l742_742591

theorem complex_number_solution 
  (z : ℂ) 
  (h : (√3 + 3 * complex.I) * z = 3 * complex.I) : 
  z = (3 / 4) + (√3 / 4) * complex.I :=
sorry

end complex_number_solution_l742_742591


namespace natural_number_divisor_problem_l742_742127

theorem natural_number_divisor_problem (x y z : ℕ) (h1 : (y+1)*(z+1) = 30) 
    (h2 : (x+1)*(z+1) = 42) (h3 : (x+1)*(y+1) = 35) :
    (2^x * 3^y * 5^z = 2^6 * 3^5 * 5^4) :=
sorry

end natural_number_divisor_problem_l742_742127


namespace expected_digits_on_20_sided_die_l742_742839

theorem expected_digits_on_20_sided_die : 
  let num_faces := 20 in 
  let one_digit_prob := 9 / num_faces in 
  let two_digit_prob := 11 / num_faces in 
  let expected_value := (one_digit_prob * 1) + (two_digit_prob * 2) in
  expected_value = 1.55 := 
by
  sorry

end expected_digits_on_20_sided_die_l742_742839


namespace smallest_natural_number_permutation_mul9_l742_742940

def is_permutation_of (a b : ℕ) : Prop :=
  multiset.of_digits a = multiset.of_digits b

theorem smallest_natural_number_permutation_mul9 :
  ∃ n : ℕ, is_permutation_of (n * 9) n ∧ ∀ m : ℕ, m < n → ¬ is_permutation_of (m * 9) m :=
by
  sorry

end smallest_natural_number_permutation_mul9_l742_742940


namespace percentage_in_excess_l742_742210

theorem percentage_in_excess 
  (A B : ℝ) (x : ℝ)
  (h1 : ∀ A',  A' = A * (1 + x / 100))
  (h2 : ∀ B',  B' = 0.94 * B)
  (h3 : ∀ A' B', A' * B' = A * B * (1 + 0.0058)) :
  x = 7 :=
by
  sorry

end percentage_in_excess_l742_742210


namespace solve_system_l742_742765

theorem solve_system : ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 :=
by {
  use 1,
  use 2,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  { refl },
}

end solve_system_l742_742765


namespace tan_150_deg_l742_742038

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742038


namespace abcd_hife_value_l742_742808

theorem abcd_hife_value (a b c d e f g h i : ℝ) 
  (h1 : a / b = 1 / 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 1 / 2) 
  (h4 : d / e = 3) 
  (h5 : e / f = 1 / 10) 
  (h6 : f / g = 3 / 4) 
  (h7 : g / h = 1 / 5) 
  (h8 : h / i = 5) : 
  abcd / hife = 17.28 := sorry

end abcd_hife_value_l742_742808


namespace square_D_perimeter_l742_742728

theorem square_D_perimeter 
(C_perimeter: Real) 
(D_area_ratio : Real) 
(hC : C_perimeter = 32) 
(hD : D_area_ratio = 1/3) : 
    ∃ D_perimeter, D_perimeter = (32 * Real.sqrt 3) / 3 := 
by 
    sorry

end square_D_perimeter_l742_742728


namespace common_divisor_of_a1986_a6891_l742_742668

def a : ℕ → ℤ
| 0          := 0
| 1          := 1
| (n + 2) := 4 * a (n + 1) + a n

theorem common_divisor_of_a1986_a6891 :
  ∃ d, d = 17 ∧ d ∣ a 1986 ∧ d ∣ a 6891 :=
sorry

end common_divisor_of_a1986_a6891_l742_742668


namespace total_surface_area_of_cylinder_l742_742853

noncomputable def rectangle_length : ℝ := 4 * Real.pi
noncomputable def rectangle_width : ℝ := 2

noncomputable def cylinder_radius (length : ℝ) : ℝ := length / (2 * Real.pi)
noncomputable def cylinder_height (width : ℝ) : ℝ := width

noncomputable def cylinder_surface_area (radius height : ℝ) : ℝ :=
  2 * Real.pi * radius^2 + 2 * Real.pi * radius * height

theorem total_surface_area_of_cylinder :
  cylinder_surface_area (cylinder_radius rectangle_length) (cylinder_height rectangle_width) = 16 * Real.pi :=
by
  sorry

end total_surface_area_of_cylinder_l742_742853


namespace distance_traveled_in_3_seconds_l742_742317

def velocity (t : ℝ) : ℝ := 5 * t^2

def position (t : ℝ) : ℝ := ∫ (u : ℝ) in 0..t, velocity u

theorem distance_traveled_in_3_seconds : position 3 = 45 := by
  unfold position velocity
  simp
  sorry

end distance_traveled_in_3_seconds_l742_742317


namespace tan_150_eq_neg_inv_sqrt3_l742_742062

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742062


namespace sum_binom_solutions_l742_742392

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742392


namespace number_of_new_trailer_homes_l742_742790

theorem number_of_new_trailer_homes 
  (n : ℕ) -- number of new trailer homes
  (avg_age_3_years_ago : ℕ) -- average age of trailer homes three years ago
  (total_original_trailer_homes : ℕ) -- total number of original trailer homes
  (avg_age_today : ℕ) -- average age of trailer homes today
  (total_trailers_today : ℕ) -- total number of trailer homes today
  (sum_ages_3_years_ago : ℕ) -- sum of ages of trailer homes three years ago
  :
  avg_age_3_years_ago = 15 ∧ total_original_trailer_homes = 30 ∧ avg_age_today = 12 
  → sum_ages_3_years_ago = total_original_trailer_homes * avg_age_3_years_ago
  → total_trailers_today = total_original_trailer_homes + n
  → ∑ (i in range total_original_trailer_homes), 3*(i+3) + ∑ (i in range n), 3 = total_trailers_today * avg_age_today
  → n = 20 :=
by
  sorry

end number_of_new_trailer_homes_l742_742790


namespace arithmetic_progression_sin_cos_cot_l742_742761

theorem arithmetic_progression_sin_cos_cot (α β γ : ℝ) (h1 : β = (α + γ) / 2) :
  (sin α - sin γ) / (cos γ - cos α) = Real.cot β :=
by
  sorry

end arithmetic_progression_sin_cos_cot_l742_742761


namespace sum_of_valid_n_l742_742376

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742376


namespace remainder_of_M_l742_742670

-- Define the problem
def is_largest_multiple_of_24_no_repeat (n : ℕ) : Prop :=
  (n % 24 = 0) ∧ (∀ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (nat.digits 10 n).count d ≤ 1)

-- Define the largest number M that fits the condition
def M : ℕ := 9840

-- Define the remainder function
def remainder_when_divided_by_500 (n : ℕ) : ℕ :=
  n % 500

-- State the theorem
theorem remainder_of_M : is_largest_multiple_of_24_no_repeat M →
  remainder_when_divided_by_500 M = 340 :=
by
  assume h : is_largest_multiple_of_24_no_repeat M
  sorry

end remainder_of_M_l742_742670


namespace pebbles_game_invariant_l742_742748

/-- 
The game of pebbles is played on an infinite board of lattice points (i, j).
Initially, there is a pebble at (0, 0).
A move consists of removing a pebble from point (i, j) and placing a pebble at each of the points (i+1, j) and (i, j+1) provided both are vacant.
Show that at any stage of the game there is a pebble at some lattice point (a, b) with 0 ≤ a + b ≤ 3. 
-/
theorem pebbles_game_invariant :
  ∀ (board : ℕ × ℕ → Prop) (initial_state : board (0, 0)) (move : (ℕ × ℕ) → Prop → Prop → Prop),
  (∀ (i j : ℕ), board (i, j) → ¬ board (i+1, j) ∧ ¬ board (i, j+1) → board (i+1, j) ∧ board (i, j+1)) →
  ∃ (a b : ℕ), (0 ≤ a + b ∧ a + b ≤ 3) ∧ board (a, b) :=
by
  intros board initial_state move move_rule
  sorry 

end pebbles_game_invariant_l742_742748


namespace DE_product_l742_742551

theorem DE_product (D E : ℤ) (hD : D = 8) (hE : E = 5) 
  (H : ∀ r : ℂ, r^2 - r - 1 = 0 → r^6 - D * r - E = 0) : 
  D * E = 40 := by
  simp [hD, hE]
  sorry

end DE_product_l742_742551


namespace range_of_a_l742_742592

noncomputable def f (x : ℝ) : ℝ := real.sqrt ((1 + x) * (2 - x))
noncomputable def g (x a : ℝ) : ℝ := real.log (x - a)

def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a).nonempty → a < 2 := 
by
  sorry

end range_of_a_l742_742592


namespace smallest_digit_permutation_l742_742937

open Nat

def is_digit_permutation (n m : ℕ) : Prop :=
  n.digits 10 ~ m.digits 10

theorem smallest_digit_permutation:
  ∃ n : ℕ, (n * 9).digits 10 ~ n.digits 10 ∧ (∀ m : ℕ, m * 9 ≠ 1089 → ¬ (m.digits 10 ~ (m * 9).digits 10)) :=
by
  sorry

end smallest_digit_permutation_l742_742937


namespace set_equality_sin_cos_l742_742621

theorem set_equality_sin_cos (a b : ℝ) (h : ({Real.sin (π/2), a, b / a} = {Real.cos (π/2), a^2, a + b})) : 
  a ^ 2018 + b ^ 2018 = 1 := by
  sorry

end set_equality_sin_cos_l742_742621


namespace sum_of_elements_in_finite_cyclic_subgroups_is_integer_exists_set_with_sum_k_in_finite_cyclic_subgroups_l742_742429

-- Condition: A finite set of elements of finite cyclic subgroups of complex numbers
-- Part (a): The sum of all elements in such a set is an integer.
theorem sum_of_elements_in_finite_cyclic_subgroups_is_integer 
    (A : finset ℂ) 
    (hA_closed_mult: ∀ z ∈ A, ∀ n : ℕ, z^n ∈ A) : 
        ∃ (s : ℤ), ∑ x in A, x = s := 
sorry

-- Part (b): For any integer k, there exists such a set A with sum k.
theorem exists_set_with_sum_k_in_finite_cyclic_subgroups 
    (k : ℤ) : ∃ (A : finset ℂ), 
        (∀ z ∈ A, ∀ n : ℕ, z^n ∈ A) ∧ ∑ x in A, x = k := 
sorry

end sum_of_elements_in_finite_cyclic_subgroups_is_integer_exists_set_with_sum_k_in_finite_cyclic_subgroups_l742_742429


namespace height_of_cone_formed_by_rolling_sector_l742_742826

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l742_742826


namespace number_of_handshakes_l742_742510

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end number_of_handshakes_l742_742510


namespace remainder_of_n_divided_by_1000_is_44_l742_742560

def sum_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  let rec sum_digits (n : ℕ) : ℕ :=
    if n = 0 then 0
    else (n % b) + sum_digits (n / b)
  sum_digits n

def f (n : ℕ) : ℕ := sum_of_digits_base n 3
def g (n : ℕ) : ℕ := sum_of_digits_base (f n) 9
def base_sixteen (n : ℕ) : String := n.toString 16

theorem remainder_of_n_divided_by_1000_is_44 : ∃ (N : ℕ), base_sixteen (g N) ≠ "0" ∧ base_sixteen (g N) ≠ "1" ∧ base_sixteen (g N) ≠ "2" ∧ base_sixteen (g N) ≠ "3" ∧ base_sixteen (g N) ≠ "4" ∧ base_sixteen (g N) ≠ "5" ∧ base_sixteen (g N) ≠ "6" ∧ base_sixteen (g N) ≠ "7" ∧ base_sixteen (g N) ≠ "8" ∧ base_sixteen (g N) ≠ "9" ∧ N % 1000 = 44 :=
by
  sorry

end remainder_of_n_divided_by_1000_is_44_l742_742560


namespace fixed_salary_new_scheme_l742_742859

theorem fixed_salary_new_scheme (S : ℝ) (R_old : ℝ) (C_new : ℝ) (F : ℝ) :
  S = 12000 →
  R_old = 0.05 * S →
  C_new = 0.025 * (S - 4000) →
  F + C_new = R_old + 600 →
  F = 1000 :=
by {
  intros hS hRold hC_new hF_C_new,
  sorry
}

end fixed_salary_new_scheme_l742_742859


namespace equal_segments_l742_742955

variable {Point : Type}
variable (S A B C P Q R M : Point)
variable (BC_line AC_line : Set Point)
variable [AffinePlane Point]

variable (AP_eq_CP : dist A P = dist C P)
variable (CP_gt_B : dist C P > dist B S)
variable (symm_midpoint : is_midpoint M B C → reflection M P = Q)
variable (reflection_AC : reflection_line AC_line Q = R)

variable (angle_SAB_eq_QAC : ∠ S A B = ∠ Q A C)
variable (angle_SBC_eq_BAC : ∠ S B C = ∠ B A C)

theorem equal_segments (h1: AP_eq_CP)
                       (h2: CP_gt_B)
                       (h3: symm_midpoint)
                       (h4: reflection_AC)
                       (h5: angle_SAB_eq_QAC)
                       (h6: angle_SBC_eq_BAC):
                       dist S A = dist S R := 
sorry

end equal_segments_l742_742955


namespace number_of_children_l742_742514

-- defining the conditions

def total_apples : ℕ := 450
def apples_per_child : ℕ := 10
def apples_per_adult : ℕ := 3
def number_of_adults : ℕ := 40

-- defining the problem
theorem number_of_children (total_apples = 450)
  (apples_per_child = 10)
  (apples_per_adult = 3)
  (number_of_adults = 40) : 
  ∃ c : ℕ, c = (total_apples - (number_of_adults * apples_per_adult)) / apples_per_child ∧ c = 33 :=
by
  sorry

end number_of_children_l742_742514


namespace evaluate_expression_at_two_l742_742801

theorem evaluate_expression_at_two :
  let x := 2 in 2 * x ^ 2 + 3 * x - 4 = 10 :=
by
  sorry

end evaluate_expression_at_two_l742_742801


namespace sum_of_integer_values_l742_742356

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742356


namespace nine_x_five_y_multiple_l742_742430

theorem nine_x_five_y_multiple (x y : ℤ) (h : 2 * x + 3 * y ≡ 0 [ZMOD 17]) : 
  9 * x + 5 * y ≡ 0 [ZMOD 17] := 
by
  sorry

end nine_x_five_y_multiple_l742_742430


namespace sum_of_ns_l742_742400

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742400


namespace mean_value_of_pentagon_interior_angles_l742_742331

theorem mean_value_of_pentagon_interior_angles :
  let n := 5
  let sum_of_interior_angles := (n - 2) * 180
  let mean_value := sum_of_interior_angles / n
  mean_value = 108 :=
by
  sorry

end mean_value_of_pentagon_interior_angles_l742_742331


namespace circumscribed_circle_area_l742_742465

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742465


namespace calculation_result_l742_742890

theorem calculation_result :
  -Real.sqrt 4 + abs (-Real.sqrt 2 - 1) + (Real.pi - 2013) ^ 0 - (1/5) ^ 0 = Real.sqrt 2 - 1 :=
by
  sorry

end calculation_result_l742_742890


namespace tan_150_deg_l742_742025

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742025


namespace intersect_at_P_and_M_l742_742706

-- Define points A, B, C.
variables {A B C A1 B1 C1 P M : Type*}

-- Define conditions for points A1, B1, C1 on the sides of triangle ABC and intersection at P.
variables (on_sides_ABC : 
            A1 ∈ line_segment B C ∧ 
            B1 ∈ line_segment C A ∧ 
            C1 ∈ line_segment A B)
variables (intersect_at_P : 
            concurrent_lines (line_through A A1) (line_through B B1) (line_through C C1) P)

-- Define midpoints and lines la, lb, lc connecting given midpoints.
variables (midpoints_and_lines : 
            (midpoint_line (midpoint B C) (midpoint B1 C1) = l_a) ∧
            (midpoint_line (midpoint C A) (midpoint C1 A1) = l_b) ∧
            (midpoint_line (midpoint A B) (midpoint A1 B1) = l_c))

-- Define M as the centroid of triangle ABC.
variable (M : centroid_of_triangle ABC)

-- Statement: Prove the intersection of la, lb, lc at a point on PM.
theorem intersect_at_P_and_M (h : (l_a, l_b, l_c)) : 
  ∃ Q, (Q ∈ l_a ∧ Q ∈ l_b ∧ Q ∈ l_c) ∧ (Q ∈ line_segment P M) :=
sorry

end intersect_at_P_and_M_l742_742706


namespace find_value_of_reciprocals_l742_742580

-- Defining the conditions and the final problem as hypotheses in the Lean 4 proof statement.
theorem find_value_of_reciprocals (x y : ℝ) (h1 : 2^x = 10) (h2 : 5^y = 10) : (1 / x) + (1 / y) = 1 := 
by
  -- The proof will be provided here, but for now, we place a placeholder
  sorry

end find_value_of_reciprocals_l742_742580


namespace fish_filets_total_l742_742881

/- Define the number of fish caught by each family member -/
def ben_fish : ℕ := 4
def judy_fish : ℕ := 1
def billy_fish : ℕ := 3
def jim_fish : ℕ := 2
def susie_fish : ℕ := 5

/- Define the number of fish thrown back -/
def fish_thrown_back : ℕ := 3

/- Define the number of filets per fish -/
def filets_per_fish : ℕ := 2

/- Calculate the number of fish filets -/
theorem fish_filets_total : ℕ :=
  let total_fish_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_fish_caught - fish_thrown_back
  fish_kept * filets_per_fish

example : fish_filets_total = 24 :=
by {
  /- This 'sorry' placeholder indicates that a proof should be here -/
  sorry
}

end fish_filets_total_l742_742881


namespace hours_to_clean_driveway_l742_742695

/-- Lou's initial shoveling rate and its decrement. --/
def initial_shoveling_rate : ℕ := 25
def hourly_decrement : ℕ := 2

/-- The dimensions of the driveway. --/
def driveway_width : ℕ := 6
def driveway_length : ℕ := 15
def snow_depth : ℕ := 2.5

/-- Total volume of snow in the driveway. --/
def total_snow_volume : ℕ := driveway_width * driveway_length * snow_depth

/-- Snow removed after the first four hours. --/
def snow_removed_first_four_hours : ℕ := 
  initial_shoveling_rate + (initial_shoveling_rate - hourly_decrement) + 
  (initial_shoveling_rate - 2 * hourly_decrement) + 
  (initial_shoveling_rate - 3 * hourly_decrement)

/-- Remaining snow after the first four hours. --/
def remaining_snow : ℕ := total_snow_volume - snow_removed_first_four_hours

/-- Snow removed after four hours with changed decrement. --/
def snow_removed_after_four_hours (n : ℕ) : ℕ := nat.rec_on n 0 
  (λ _ acc, acc + (initial_shoveling_rate - 7 - n))

/-- Prove the total hours required to clean the driveway meets one of the choices. --/
theorem hours_to_clean_driveway : ∃ n : ℕ, n = 10 ∨ n = 11 ∨ n = 12 ∨ n = 13 ∧ 
  (remaining_snow ≤ snow_removed_after_four_hours (n - 4)) := sorry

end hours_to_clean_driveway_l742_742695


namespace units_digit_S_6789_l742_742625

def S (n : ℕ) : ℚ :=
  let c := 4 + Real.sqrt 15
  let d := 4 - Real.sqrt 15
  (1 / 2) * (c^n + d^n)

theorem units_digit_S_6789 : (S 6789) % 10 = 4 :=
  sorry

end units_digit_S_6789_l742_742625


namespace find_number_divisible_by_1375_l742_742134

theorem find_number_divisible_by_1375 :
  ∃ (x y : ℕ), (x < 10) ∧ (y < 10) ∧ (713625 = 7 * 10^5 + x * 10^4 + 3 * 10^3 + 6 * 10^2 + y * 10 + 5) ∧ (713625 % 1375 = 0) :=
by {
  exists 1,
  exists 2,
  split,
  { exact Nat.lt_succ_self _ }, -- x < 10
  split,
  { exact Nat.lt_succ_self _ }, -- y < 10
  split,
  { norm_num },
  { norm_num }
}

end find_number_divisible_by_1375_l742_742134


namespace sum_of_valid_ns_l742_742344

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742344


namespace velocity_at_t_2_l742_742603

variable {t : ℝ}
noncomputable def motion_eq (t : ℝ) : ℝ := t^2 + 3 / t

theorem velocity_at_t_2 : deriv motion_eq 2 = 13 / 4 :=
by {
  sorry
}

end velocity_at_t_2_l742_742603


namespace find_n_divides_2n_plus_2_l742_742550

theorem find_n_divides_2n_plus_2 :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 1997 ∧ n ∣ (2 * n + 2)) ∧ n = 946 :=
by {
  sorry
}

end find_n_divides_2n_plus_2_l742_742550


namespace limit_eq_neg_2_div_pi_l742_742888

noncomputable def limit_function (x : ℝ) := (Real.tan (3^(Real.pi / x) - 3)) / (3^(Real.cos (3 * x / 2)) - 1)

theorem limit_eq_neg_2_div_pi : tendsto limit_function (𝓝[>] Real.pi) (𝓝 (-2 / Real.pi)) := by
  sorry

end limit_eq_neg_2_div_pi_l742_742888


namespace tan_150_deg_l742_742043

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742043


namespace collinear_points_l742_742997

variable (a : ℝ) (A B C : ℝ × ℝ)

-- Conditions given in the problem
def point_A := (a, 2 : ℝ)
def point_B := (5, 1 : ℝ)
def point_C := (-4, 2 * a : ℝ)

-- Collinearity condition
def collinear (x y z : ℝ): Prop :=
  (x.1 - y.1) * (y.2 - z.2) = (y.1 - z.1) * (x.2 - y.2)

theorem collinear_points :
  collinear (point_A a) (point_B) (point_C a) →
  a = 5 + sqrt 21 ∨ a = 5 - sqrt 21 :=
by {
  sorry,
}

end collinear_points_l742_742997


namespace john_recycling_income_l742_742662

theorem john_recycling_income :
  (let paper_weight_mon_sat := 8
   let paper_weight_sun := 2 * paper_weight_mon_sat
   let papers_per_day := 250
   let days := 10 * 7
   let days_mon_sat := 10 * 6
   let days_sun := 10
   let total_weight_mon_sat := days_mon_sat * papers_per_day * paper_weight_mon_sat
   let total_weight_sun := days_sun * papers_per_day * paper_weight_sun
   let total_weight := total_weight_mon_sat + total_weight_sun
   let ounces_per_ton := 32000
   let tons := total_weight / ounces_per_ton
   let money_per_ton := 20
   let total_money := tons * money_per_ton
  in total_money = 100) :=
sorry

end john_recycling_income_l742_742662


namespace find_three_digit_number_l742_742547

theorem find_three_digit_number :
  ∃ a b c : ℕ, 
  a + b + c = 9 ∧ 
  a * b * c = 24 ∧
  100*c + 10*b + a = (27/38) * (100*a + 10*b + c) ∧ 
  100*a + 10*b + c = 342 := sorry

end find_three_digit_number_l742_742547


namespace partial_fraction_product_l742_742310

theorem partial_fraction_product :
  (x^3 - 3 * x^2 - 4 * x + 12 = (x - 1) * (x + 3) * (x - 4)) →
  (∀ (A B C : ℚ),
   (∀ x, (x^2 - 23) = A * (x + 3) * (x - 4) + B * (x - 1) * (x - 4) + C * (x - 1) * (x + 3)) →
      A = 11/6 ∧ B = -1/2 ∧ C = -1/3) →
  (11/6 * (-1/2) * (-1/3) = 11/36) :=
begin
  sorry
end

end partial_fraction_product_l742_742310


namespace exists_zero_sum_subsequence_l742_742275

theorem exists_zero_sum_subsequence (s : Fin 2000 → ℤ) (h_abs_bound : ∀ i, |s i| ≤ 1000) (h_sum : (Finset.univ.sum s) = 1) :
  ∃ (t : Finset (Fin 2000)), t.nonempty ∧ (t.sum s) = 0 :=
sorry

end exists_zero_sum_subsequence_l742_742275


namespace soccer_league_games_l742_742495

theorem soccer_league_games (n_teams : Nat) (games_per_pair: Nat) (unique_games: Nat) (total_games: Nat): 
  n_teams = 10 → games_per_pair = 4 → unique_games = (n_teams * (n_teams - 1) / 2) → total_games = (games_per_pair * unique_games) → 
  total_games = 180 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, ←h4, h3]
  norm_num
  sorry

end soccer_league_games_l742_742495


namespace tan_150_degrees_l742_742053

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742053


namespace fraction_of_silver_knights_with_shields_l742_742202

theorem fraction_of_silver_knights_with_shields :
  ∀ (total_knights silver_knights golden_knights shields_knights : ℕ),
  (silver_knights = (3 / 8) * total_knights) →
  (golden_knights = total_knights - silver_knights) →
  (shields_knights = (1 / 4) * total_knights) →
  ∃ (p q : ℕ),
    (1 / 4 * total_knights = silver_knights * (p / q) + golden_knights * (p / (3 * q))) →
    (p / q = 3 / 7) :=
by
  -- Assume total number of knights is non-zero
  intros total_knights silver_knights golden_knights shields_knights
  assume h_silver h_golden h_shields
  existsi (3 : ℕ)
  existsi (7 : ℕ)
  assume h_fraction
  sorry

end fraction_of_silver_knights_with_shields_l742_742202


namespace extreme_points_difference_l742_742598

def f (x k : ℝ) : ℝ := (1/2) * x^2 - k * x + Real.log x

noncomputable def extreme_points (k : ℝ) : ℝ × ℝ :=
  let x1 := (k - Real.sqrt (k^2 - 4)) / 2
  let x2 := (k + Real.sqrt (k^2 - 4)) / 2
  (x1, x2)

theorem extreme_points_difference (k : ℝ) (h : k > 2) :
  let (x1, x2) := extreme_points k in
  |f x1 k - f x2 k| < (k^2 / 2) - 2 :=
by
  sorry

end extreme_points_difference_l742_742598


namespace smallest_in_set_l742_742753

def set_of_consecutive_odd_integers (s : Set Int) : Prop :=
  ∀ x ∈ s, (x % 2 = 1) ∧ ∀ y ∈ s, y = x + 2 * (y - x) / 2

def median_of_set (s : Set Int) (m : Int) : Prop :=
  let n := s.card (by sorry)
  n % 2 = 0 → ∃ a b, a < b ∧ a ∈ s ∧ b ∈ s ∧ m = (a + b) / 2

theorem smallest_in_set (s : Set Int) (m g : Int) (h_median: median_of_set s m) (h_greatest: ∀ x ∈ s, x ≤ g) : 
  set_of_consecutive_odd_integers s → ∃ x, x ∈ s ∧ ∀ y ∈ s, y ≥ x → y = x ∨ x < y :=
by
  intros h_cond
  have h_s : s = Finset.Icc 163 195 by sorry
  exists 163
  split
  · rw h_s
    exact Finset.mem_Icc.mpr ⟨rfl.le, rfl.le⟩
  · intro y h_y
    rw h_s at h_y
    exact Decidable.or_of_true (h_cond y) sorry
sorry

end smallest_in_set_l742_742753


namespace geometric_sequence_sum_l742_742950

-- Definition of the sum of the first n terms of a geometric sequence
variable (S : ℕ → ℝ)

-- Conditions given in the problem
def S_n_given (n : ℕ) : Prop := S n = 36
def S_2n_given (n : ℕ) : Prop := S (2 * n) = 42

-- Theorem to prove
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) 
    (h1 : S n = 36) (h2 : S (2 * n) = 42) : S (3 * n) = 48 := sorry

end geometric_sequence_sum_l742_742950


namespace digit_sum_not_square_or_cube_l742_742970

theorem digit_sum_not_square_or_cube {N : ℕ} : 
  let digit_sum (n : ℕ) : ℕ := 
    if n < 10 then n else digit_sum (n.digits.sum)
  in (digit_sum N = 2 ∨ digit_sum N = 3 ∨ digit_sum N = 5 ∨ digit_sum N = 6) → 
     ¬ (∃ k : ℕ, N = k^2) ∧ ¬ (∃ m : ℕ, N = m^3) :=
begin
  sorry
end

end digit_sum_not_square_or_cube_l742_742970


namespace units_digit_of_sequence_l742_742656

theorem units_digit_of_sequence : 
  (2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 + 2 * 3^4 + 2 * 3^5 + 2 * 3^6 + 2 * 3^7 + 2 * 3^8 + 2 * 3^9) % 10 = 8 := 
by 
  sorry

end units_digit_of_sequence_l742_742656


namespace no_valid_ab_intersection_l742_742569

def setA (a b : ℝ) : Set (ℤ × ℝ) := {p | ∃ n : ℤ, p = (n, a * n + b)}
def setB : Set (ℤ × ℝ) := {p | ∃ m : ℤ, p = (m, 3 * m^2 + 15)}
def setC : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 144}

theorem no_valid_ab_intersection (a b : ℝ) : 
  ¬ (∃ (x : ℤ) (y : ℝ), (x, y) ∈ setA a b ∧ (x, y) ∈ setB ∧ (a, b) ∈ setC) :=
sorry

end no_valid_ab_intersection_l742_742569


namespace sum_of_valid_ns_l742_742345

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742345


namespace area_of_circumscribed_circle_l742_742484

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742484


namespace cookies_leftover_l742_742507

def amelia_cookies := 52
def benjamin_cookies := 63
def chloe_cookies := 25
def total_cookies := amelia_cookies + benjamin_cookies + chloe_cookies
def package_size := 15

theorem cookies_leftover :
  total_cookies % package_size = 5 := by
  sorry

end cookies_leftover_l742_742507


namespace second_and_third_shooters_cannot_win_or_lose_simultaneously_l742_742325

-- Define the conditions C1, C2, and C3
variables (C1 C2 C3 : Prop)

-- The first shooter bets that at least one of the second or third shooters will miss
def first_shooter_bet : Prop := ¬ (C2 ∧ C3)

-- The second shooter bets that if the first shooter hits, then at least one of the remaining shooters will miss
def second_shooter_bet : Prop := C1 → ¬ (C2 ∧ C3)

-- The third shooter bets that all three will hit the target on the first attempt
def third_shooter_bet : Prop := C1 ∧ C2 ∧ C3

-- Prove that it is impossible for both the second and third shooters to either win or lose their bets concurrently
theorem second_and_third_shooters_cannot_win_or_lose_simultaneously :
  ¬ ((second_shooter_bet C1 C2 C3 ∧ third_shooter_bet C1 C2 C3) ∨ (¬ second_shooter_bet C1 C2 C3 ∧ ¬ third_shooter_bet C1 C2 C3)) :=
by
  sorry

end second_and_third_shooters_cannot_win_or_lose_simultaneously_l742_742325


namespace movies_still_to_watch_l742_742781

theorem movies_still_to_watch 
  (total_movies : ℕ) 
  (watched_movies : ℕ) 
  (h_total : total_movies = 35) 
  (h_watched : watched_movies = 18) : 
  total_movies - watched_movies = 17 := 
by 
  rw [h_total, h_watched]
  simp
  sorry

end movies_still_to_watch_l742_742781


namespace sum_of_n_values_l742_742364

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742364


namespace cone_height_l742_742830

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l742_742830


namespace find_pairs_l742_742925

theorem find_pairs (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (h1 : a ∣ b^4 + 1) (h2 : b ∣ a^4 + 1) (h3 : int.floor (real.sqrt a) = int.floor (real.sqrt b)) :
  (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by {
  sorry -- proof goes here
}

end find_pairs_l742_742925


namespace m_value_f_increasing_range_of_a_minus_c_div_2_l742_742584

noncomputable def f (x : ℝ) (m : ℝ) (ω : ℝ) : ℝ := m * sin (ω * x) - cos (ω * x)

variables (ω m : ℝ) (k : ℤ)
constants (x₀ : ℝ) (T : ℝ) (a b c A B C R : ℝ)

-- Given conditions
axiom h1 : x₀ = π / 3
axiom h2 : ∃ T > 0, T = π ∧ ω = 2 ∧ m > 0 ∧ (∀ x, f x m ω = f (T + x) m ω)
axiom h3 : m = sqrt 3
axiom h4 : ∀ x, f x m ω = sqrt (m ^ 2 + 1) * sin (ω * x + (π / 6 - ↑k * π))
axiom h5 : ∀ k : ℤ, f x m ω is increasing on [k * π - π / 6, k * π + π / 3]
axiom h6 : B = π / 3
axiom h7 : b = sqrt 3
axiom h8 : A ∈ (0, 2 * π / 3)
axiom h9 : a / sin A = 2
axiom h10 : c / sin C = 2

theorem m_value :
  m = sqrt 3 :=
by sorry

theorem f_increasing : 
  ∀ k : ℤ, (∀ x ∈ set.Icc (k * π - π / 6) (k * π + π / 3), (deriv (f x m ω)) x > 0) :=
by sorry

theorem range_of_a_minus_c_div_2 : 
  a - c / 2 ∈ (set.Ioo (-sqrt 3 / 2) sqrt 3) :=
by sorry

end m_value_f_increasing_range_of_a_minus_c_div_2_l742_742584


namespace brown_is_criminal_l742_742738

def brown_says : Prop :=
  ¬Brown ∧ ¬Jones

def smith_says : Prop :=
  ¬Smith ∧ Brown

def jones_says : Prop :=
  ¬Brown ∧ Smith

axiom conditions : 
  (¬Brown ∧ ¬Jones) ∨ (¬Smith ∧ Brown) ∨ (¬Brown ∧ Smith)

def truth_conditions_brown : Prop :=
  (brown_says ∧ ¬brown_says)  -- One truth and one lie for Brown

def truth_conditions_smith : Prop :=
  (smith_says ∧ smith_says) -- Two lies for Smith

def truth_conditions_jones : Prop :=
  (jones_says ∧ jones_says) -- Two lies for Jones

theorem brown_is_criminal : Brown :=
  sorry

end brown_is_criminal_l742_742738


namespace camel_water_ratio_l742_742498

theorem camel_water_ratio (gallons_water : ℕ) (ounces_per_gallon : ℕ) (traveler_ounces : ℕ)
  (total_ounces : ℕ) (camel_ounces : ℕ) (ratio : ℕ) 
  (h1 : gallons_water = 2) 
  (h2 : ounces_per_gallon = 128) 
  (h3 : traveler_ounces = 32) 
  (h4 : total_ounces = gallons_water * ounces_per_gallon) 
  (h5 : camel_ounces = total_ounces - traveler_ounces)
  (h6 : ratio = camel_ounces / traveler_ounces) : 
  ratio = 7 := 
by
  sorry

end camel_water_ratio_l742_742498


namespace actual_road_length_l742_742703

theorem actual_road_length
  (scale_factor : ℕ → ℕ → Prop)
  (map_length_cm : ℕ)
  (actual_length_km : ℝ) : 
  (scale_factor 1 50000) →
  (map_length_cm = 15) →
  (actual_length_km = 7.5) :=
by
  sorry

end actual_road_length_l742_742703


namespace split_department_proof_l742_742293

-- Define the problem conditions and their corresponding types
variable {Person : Type} -- Type of individuals in the department
variable (conspiracies : Finset (Finset Person)) -- Set of conspiracies, each involving exactly 3 persons
variable [Fintype Person] -- The department has a finite number of people

-- Additional conditions: 
-- 1. There are six conspiracies.
-- 2. Each conspiracy involves exactly 3 persons.
def isValidConspiracy (p : Finset Person) : Prop := p.card = 3
def num_of_conspiracies (cs : Finset (Finset Person)) : Prop := cs.card = 6
def valid_department (cs : Finset (Finset Person)) : Prop := ∀ c ∈ cs, isValidConspiracy c

-- Final statement to prove
theorem split_department_proof (h_valid_department : valid_department conspiracies)
                               (h_six_conspiracies : num_of_conspiracies conspiracies) :
  ∃ (lab1 lab2 : Finset Person), lab1 ∩ lab2 = ∅ ∧
  (∀ c ∈ conspiracies, ¬ (c ⊆ lab1 ∨ c ⊆ lab2)) :=
sorry

end split_department_proof_l742_742293


namespace system_of_equations_solution_l742_742767

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x + 3 * y = 7) 
  (h2 : y = 2 * x) : 
  x = 1 ∧ y = 2 :=
by
  sorry

end system_of_equations_solution_l742_742767


namespace interval_of_decrease_l742_742302

theorem interval_of_decrease {f : ℝ → ℝ} (h₀ : ∀ x > 0, f x = x - log x) :
  ∀ x ∈ Ioo 0 1, (f' x < 0) :=
by
  intro x hx
  have h_deriv : f' x = 1 - 1 / x := sorry
  linarith [hx.1, hx.2]
  sorry

end interval_of_decrease_l742_742302


namespace carly_dog_count_l742_742896

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l742_742896


namespace rocket_launch_work_l742_742918

def work_required (P R H : ℝ) : ℝ :=
  P * R * H / (R + H)

variables (P R H : ℝ) (g : ℝ)
variable (f : ℝ → ℝ)

axiom grav_force (x : ℝ) : f x = P * R^2 / x^2
axiom P_val : P = 1.5 * 10^3
axiom R_val : R = 6.4 * 10^6
axiom H_val : H = 2 * 10^6

theorem rocket_launch_work :
  work_required P R H = 2.285714 * 10^9 :=
by
  sorry

end rocket_launch_work_l742_742918


namespace geometric_factorial_exponents_l742_742137

noncomputable def gcd (a : Nat) (b : Nat) : Nat := if b == 0 then a else gcd b (a % b)

def is_geometric_sequence {α : Type} [Field α] (s : List α) : Prop :=
  ∀ i, (0 < i ∧ i < s.length - 1) → (s.get! i) ^ 2 = (s.get! (i - 1)) * (s.get! (i + 1))

def prime_factors (n : Nat) : List Nat := sorry

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

def alpha_seq (n : Nat) : List Nat :=
  let p_factors := prime_factors (factorial n)
  p_factors.map (fun p => ∑ t in (Finset.range (Nat.log p n)).filter (fun i => i > 0), (n / (p ^ t)))

theorem geometric_factorial_exponents (n : Nat) (hn : n ≥ 3) :
  (is_geometric_sequence (alpha_seq n)) ↔ (n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 10) := sorry

end geometric_factorial_exponents_l742_742137


namespace carly_dogs_total_l742_742899

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l742_742899


namespace triangle_BD_length_l742_742218

open Real

noncomputable def length_of_BD (AB BC DE : ℝ) : ℝ :=
  let AC := sqrt (AB^2 + BC^2)
  (DE * BC) / AC

theorem triangle_BD_length :
  ∀ {D E : Point} (AB BC DE : ℝ) (A B C : Triangle) (h1 : ∠ B = 90)
  (h2 : AB = 9) (h3 : BC = 12) 
  (h4 : PointOnLine A D AC) (h5 : PointOnLine B E BC)
  (h6 : ∠ EDB = 90) (h7 : DE = 6), 
  length_of_BD AB BC DE = 24 / 5 :=
by
  sorry

end triangle_BD_length_l742_742218


namespace range_of_f_l742_742911

theorem range_of_f (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f(x + y) = f(x) * f(y))
  (h2 : ∀ x : ℝ, f(x) ≠ 0)
  (h3 : ∀ y : ℝ, 0 < y → 0 < f(y) ∧ f(y) < 1) :
  set.range f = set.univ :=
sorry

end range_of_f_l742_742911


namespace jerome_contact_list_end_of_month_l742_742229

-- Define the initial conditions
def classmates := 20
def out_of_school_friends := classmates / 2
def immediate_family := 2 + 1
def extended_family_added := 5
def acquaintances_added := 7
def coworkers_added := 10

-- Define the removals
def extended_family_removed := 3
def acquaintances_removed := 4
def coworkers_removed := 0.30 * coworkers_added

-- Define the total people added and removed
def total_added := classmates + out_of_school_friends + immediate_family + extended_family_added + acquaintances_added + coworkers_added
def total_removed := extended_family_removed + acquaintances_removed + coworkers_removed

-- Define the final calculation
def final_count := total_added - total_removed

-- The proof statement
theorem jerome_contact_list_end_of_month : final_count = 45 := by
  -- Omitted proof
  sorry

end jerome_contact_list_end_of_month_l742_742229


namespace distance_from_point_to_x_axis_correct_l742_742299

noncomputable def distance_from_point_to_x_axis
  (x y : ℝ) (a b c : ℝ)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_focal : a^2 + b^2 = c^2)
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_P_on_hyperbola : (P.1) ^ 2 / a ^ 2 - (P.2) ^ 2 / b ^ 2 = 1)
  (h_perpendicular : ⟦(P.1 - F₁.1)ˆ2 + (P.2 - F₁.2)ˆ2⟧ * ⟦(P.1 - F₂.1)ˆ2 + (P.2 - F₂.2)ˆ2⟧ = 4 * c^2 - (2a)^2)
  : ℝ :=
(((b^2) / a^2) - ((4 * c^2 - b^2)/(2a)^2))

theorem distance_from_point_to_x_axis_correct
  (x y : ℝ) (a b c : ℝ)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_focal : a^2 + b^2 = c^2)
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_P_on_hyperbola : (P.1) ^ 2 / a ^ 2 - (P.2) ^ 2 / b ^ 2 = 1)
  (h_perpendicular : ⟦(P.1 - F₁.1)ˆ2 + (P.2 - F₁.2)ˆ2⟧ * ⟦(P.1 - F₂.1)ˆ2 + (P.2 - F₂.2)ˆ2⟧ = 4 * c^2 - (2a)^2)
  : distance_from_point_to_x_axis x y a b c h_hyperbola h_focal P F₁ F₂ h_P_on_hyperbola h_perpendicular = 9 / 5 :=
sorry

end distance_from_point_to_x_axis_correct_l742_742299


namespace unit_square_ring_sum_l742_742543

theorem unit_square_ring_sum (a b c : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  (∃ f : ℕ → ℕ, 
    (∀ i, 1 ≤ i ∧ i ≤ 20 → f i = 9) ∧
    (∀ j, 1 ≤ j ∧ j ≤ 15 → f (20 + j) = 8) ∧
    (∀ k, 1 ≤ k ∧ k ≤ 12 → f (35 + k) = 5) ∧
    (2 * 4 * 9 + 2 * 3 * 8) = 120 ∧
    (2 * 5 * 9 + 2 * 3 * 5) = 120 ∧
    (2 * 5 * 8 + 2 * 4 * 5) = 120) :=
begin
  sorry
end

end unit_square_ring_sum_l742_742543


namespace sin_cos_sum_cos_tan_values_l742_742427

-- Lean 4 statement for Proof Problem 1
theorem sin_cos_sum (α : ℝ) (x y r : ℝ) (h1 : x = 4) (h2 : y = -3) (h3 : r = 5) 
  (sin_def : sin α = y / r) (cos_def : cos α = x / r) : 
  2 * sin α + cos α = -2 / 5 := 
sorry

-- Lean 4 statement for Proof Problem 2
theorem cos_tan_values (α : ℝ) (x m : ℝ) (h1 : x = -√3) (h2 : m ≠ 0) 
  (sin_def : sin α = (√2 * m) / 4)
  (case1 : m = √6 → cos α = -√3 / 5 ∧ tan α = -√10 / 3)
  (case2 : m = -√6 → cos α = -√3 / 5 ∧ tan α = √10 / 3) : 
  (m = √6 → cos α = -√3 / 5 ∧ tan α = -√10 / 3) ∧ 
  (m = -√6 → cos α = -√3 / 5 ∧ tan α = √10 / 3) := 
sorry

end sin_cos_sum_cos_tan_values_l742_742427


namespace calculate_total_cost_l742_742538

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end calculate_total_cost_l742_742538


namespace candy_probability_difference_l742_742319

theorem candy_probability_difference :
  let total := 2004
  let total_ways := Nat.choose total 2
  let different_ways := 2002 * 1002 / 2
  let same_ways := 1002 * 1001 / 2 + 1002 * 1001 / 2
  let q := (different_ways : ℚ) / total_ways
  let p := (same_ways : ℚ) / total_ways
  q - p = 1 / 2003 :=
by sorry

end candy_probability_difference_l742_742319


namespace area_of_room_in_square_inches_l742_742780

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l742_742780


namespace no_spiky_two_digit_numbers_l742_742118

def is_spiky (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧
             10 ≤ n ∧ n < 100 ∧
             n = 10 * a + b ∧
             n = a + b^3 - 2 * a

theorem no_spiky_two_digit_numbers : ∀ n, 10 ≤ n ∧ n < 100 → ¬ is_spiky n :=
by
  intro n h
  sorry

end no_spiky_two_digit_numbers_l742_742118


namespace analytical_expression_and_monotonicity_l742_742171

noncomputable def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem analytical_expression_and_monotonicity
  (A ω φ : ℝ)
  (h₁ : A > 0)
  (h₂ : ω > 0)
  (h₃ : 0 ≤ φ ∧ φ ≤ π / 2)
  (h₄ : f(0) = 1 / 2)
  (h₅ : ∀ x, f(x + (2 * π / ω)) = f(x))
  (h₆ : ∀ x, f(x) ≥ -1) :
  f(x) = Real.sin (3 * x + π / 6) ∧
  (∀ x ∈ Icc (π / 18) (π / 9) ∪ Icc (4 * π / 9) (5 * π / 9), f(x) is_strictly_increasing_on x) ∧
  (∀ x ∈ Icc (π / 9) (4 * π / 9), f(x) is_strictly_decreasing_on x) :=
sorry

end analytical_expression_and_monotonicity_l742_742171


namespace tan_150_eq_neg_inv_sqrt3_l742_742056

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742056


namespace tan_150_eq_neg_sqrt_3_l742_742014

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742014


namespace sum_of_painted_sides_l742_742857

noncomputable def length_unpainted_side : ℝ := 9
noncomputable def area_of_parking_space : ℝ := 125

theorem sum_of_painted_sides :
  let L := area_of_parking_space / length_unpainted_side
  in 2 * L + length_unpainted_side = 36.78 :=
by
  let L := area_of_parking_space / length_unpainted_side
  have h_sum := 2 * L + length_unpainted_side
  have h_approx := Real.approximates_to 2
  exact h_sum
sorry

end sum_of_painted_sides_l742_742857


namespace simplify_expression_l742_742722

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l742_742722


namespace transformations_map_pattern_onto_itself_l742_742903

-- Define the structure of the pattern
structure pattern_on_line (ell : ℝ → ℝ → Prop) :=
  (triangles : ℝ → ℝ → Prop)
  (connected : ∀ x : ℝ, ell x (triangles x x))

-- Define the transformations
def rotation_180 (ell : ℝ → ℝ → Prop) := ∃ p : ℝ × ℝ, ∀ x y, ell x y → ell (2 * p.1 - x) (2 * p.2 - y)
def translation_periodic (ell : ℝ → ℝ → Prop) := ∃ L : ℝ, ∀ x y, ell x y → ell (x + L) (y)
def reflection_across_ell (ell : ℝ → ℝ → Prop) := ∀ x y, ell x y → ell x (-y)
def reflection_perpendicular_ell (ell : ℝ → ℝ → Prop) := ∀ x y, ell x y → ell (-x) y

-- Define the statement to be proved
theorem transformations_map_pattern_onto_itself (ell : ℝ → ℝ → Prop) (p : pattern_on_line ell) :
  (rotation_180 ell) ∧ (translation_periodic ell) ∧ (reflection_across_ell ell) ∧ (reflection_perpendicular_ell ell) :=
sorry

end transformations_map_pattern_onto_itself_l742_742903


namespace tan_150_eq_neg_sqrt3_div_3_l742_742104

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742104


namespace tan_150_eq_neg_inv_sqrt3_l742_742063

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742063


namespace sum_binom_solutions_l742_742391

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742391


namespace tan_150_eq_l742_742000

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742000


namespace visual_triangle_area_l742_742314

theorem visual_triangle_area (s : ℝ) (h : s = 4) : 
    let area_ABC := (sqrt 3 / 4) * s^2
    let area_visual := (sqrt 2 / 4) * area_ABC
    area_visual = sqrt 6 :=
by
  sorry

end visual_triangle_area_l742_742314


namespace find_slope_l742_742523

noncomputable def ellipse : set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + (p.2^2) / 4 = 1 }

variables (C D : ℝ × ℝ)
variables (A B : ℝ × ℝ) (l : ℝ → ℝ)
variables (k1 k2 k : ℝ)

axiom ellipse_conditions (C_ellipse : C ∈ ellipse) (D_ellipse : D ∈ ellipse)
  (A_coords : A = (0, -1)) (B_coords : B = (0, 1))
  (l_through_point : l 0 = 1)
  (l_C : l C.1 = C.2) (l_D : l D.1 = D.2)

axiom slope_conditions (k1_def : k1 = (D.2 + 1) / D.1) (k2_def : k2 = (C.2 - 1) / C.1)
  (k1_k2_ratio : k1 / k2 = 2)

theorem find_slope : k = 3 :=
sory

end find_slope_l742_742523


namespace union_A_B_intersection_complement_A_B_range_of_a_l742_742608

-- Definitions of the sets
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Definitions of the results to be proven
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 10} := by sorry

theorem intersection_complement_A_B : (compl A) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

theorem range_of_a (a : ℝ) (h : (A ∩ C a) ≠ ∅) : a > 1 := by sorry

end union_A_B_intersection_complement_A_B_range_of_a_l742_742608


namespace solve_for_x_l742_742187

theorem solve_for_x (x : ℝ) (h : sqrt ((2 : ℝ) / x + 3) = 2) : x = 2 := by
  -- proof goes here
  sorry

end solve_for_x_l742_742187


namespace house_10_gnomes_l742_742318

-- Define the conditions based on house numbers and gnome counts
axiom House (n : Nat) : Type
axiom red_gnomes (h : House 1) : Nat
axiom blue_gnomes (h : House 2) : Nat
axiom green_gnomes (h : House 3) : Nat
axiom yellow_gnomes (h : House 4) : Nat
axiom purple_gnomes (h : House 5) : Nat
axiom orange_gnomes (h : House 6) : Nat
axiom white_gnomes (h : House 7) : Nat
axiom pink_gnomes (h : House 7) : Nat
axiom grey_gnomes (h : House 8) : Nat
axiom brown_gnomes (h : House 9) : Nat
axiom black_gnomes (h : House 9) : Nat
axiom total_gnomes : Nat

-- Specify the counts in each house based on conditions
noncomputable def house_1_gnomes := 4
noncomputable def house_2_gnomes := 2 * house_1_gnomes
noncomputable def house_3_gnomes := house_2_gnomes - 3
noncomputable def house_4_gnomes := house_1_gnomes + house_3_gnomes
noncomputable def house_5_gnomes := 5
noncomputable def house_6_gnomes := 2
noncomputable def house_7_gnomes := 3 + 4
noncomputable def house_8_gnomes := house_4_gnomes + 3
noncomputable def house_9_gnomes := 5 + 5
noncomputable def house_9_white_pink := {white := 3, pink := 4}

-- Add up total gnomes from first 9 houses
noncomputable def total_known_gnomes := house_1_gnomes + house_2_gnomes + house_3_gnomes + house_4_gnomes + house_5_gnomes + house_6_gnomes + house_7_gnomes + house_8_gnomes + house_9_gnomes

-- Question: Prove the number of gnomes in House 10
theorem house_10_gnomes : total_gnomes - total_known_gnomes = 3 :=
by
  unfold total_known_gnomes
  unfold house_1_gnomes house_2_gnomes house_3_gnomes house_4_gnomes
  unfold house_5_gnomes house_6_gnomes house_7_gnomes house_8_gnomes
  unfold house_9_gnomes
  sorry

end house_10_gnomes_l742_742318


namespace area_of_circumscribed_circle_l742_742448

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742448


namespace triangle_not_right_l742_742710

variables {α : Type*} [inner_product_space ℝ α] {A B C P : α}

theorem triangle_not_right
  (h : dist P B - dist P C = dist (P + (B - C) - 2 * P)) :
  ¬ is_right_triangle A B C :=
sorry

end triangle_not_right_l742_742710


namespace sum_of_valid_n_l742_742375

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742375


namespace household_waste_per_day_l742_742805

theorem household_waste_per_day (total_waste_4_weeks : ℝ) (h : total_waste_4_weeks = 30.8) : 
  (total_waste_4_weeks / 4 / 7) = 1.1 :=
by
  sorry

end household_waste_per_day_l742_742805


namespace infinite_points_in_circle_l742_742678

theorem infinite_points_in_circle :
  ∃ (P : ℝ × ℝ) (C : set (ℝ × ℝ)), C = { P : ℝ × ℝ | (P.1)^2 + (P.2)^2 < 4 } ∧
  (∃ A B : ℝ × ℝ, (A.1)^2 + (A.2)^2 = 4 ∧ (B.1)^2 + (B.2)^2 = 4 ∧ 
  (P - A).1^2 + (P - A).2^2 + (P - B).1^2 + (P - B).2^2 = 8) ∧
  infinite { P : ℝ × ℝ | (P.1)^2 + (P.2)^2 < 4 ∧ (P - A).1^2 + (P - A).2^2 + (P - B).1^2 + (P - B).2^2 = 8 } := sorry

end infinite_points_in_circle_l742_742678


namespace original_proposition_converse_proposition_false_l742_742605

theorem original_proposition (a b : ℝ) (h : a + b ≥ 2) : a ≥ 1 ∨ b ≥ 1 :=
by sorry

theorem converse_proposition_false : ∃ (a b : ℝ), ¬(a + b ≥ 2 → a ≥ 1 ∨ b ≥ 1) :=
by
  use 3, -3
  simp
  split
  repeat { sorry }

end original_proposition_converse_proposition_false_l742_742605


namespace books_loaned_out_l742_742848

theorem books_loaned_out (initial_books returned_percent final_books : ℕ) (h1 : initial_books = 75) (h2 : returned_percent = 65) (h3 : final_books = 61) : 
  ∃ x : ℕ, initial_books - final_books = x - (returned_percent * x / 100) ∧ x = 40 :=
by {
  sorry 
}

end books_loaned_out_l742_742848


namespace total_games_played_l742_742417

variable (G R : ℕ) -- G is the total number of games, R is the number of remaining games

theorem total_games_played
  (h1 : 60 + 0.50 * R = 0.70 * G)
  (h2 : G = 100 + R) :
  G = 150 := by
  sorry

end total_games_played_l742_742417


namespace regular_polygon_sides_l742_742192

theorem regular_polygon_sides {n : ℕ} (h : 360 = 36 * n) : n = 10 :=
by
  have key : 360 / 36 = 10 := rfl
  simp [Nat.div_eq_of_eq_mul_right (by norm_num) h] at key
  exact key

end regular_polygon_sides_l742_742192


namespace line_OM_midpoint_KL_l742_742666

variable {α : Type*} [Field α]

structure Point :=
(x y : α)

structure Triangle :=
(A B C O : Point)
(equilateral : ∀ (P Q R : Point), P ≠ Q ∧ Q ≠ R ∧ P ≠ R → 
  (P = A → Q = B ∧ R = C) ∨ (P = B → Q = C ∧ R = A) ∨ (P = C → Q = A ∧ R = B))

variable (T : Triangle)

def is_projection (M K L : Point) (AB AC : Point → Point → α) :=
(AB T.A T.B = 0 → AB M K = 0) ∧ (AC T.A T.C = 0 → AC M L = 0)

def midpoint (P Q : Point) : Point :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

theorem line_OM_midpoint_KL :
  ∀ (M K L : Point),
  is_projection M K L (λ P Q, (Q.x - P.x) * (Q.y - P.y)) (λ P Q, (Q.y - P.y) * (Q.x - P.x)) →
  ∃ P : Point, P = midpoint K L ∧ (λ O M, midpoint O M) T.O M = P :=
by
  sorry


end line_OM_midpoint_KL_l742_742666


namespace sequences_equal_l742_742587

theorem sequences_equal (n : ℕ) (hn : 0 < n) :
  let a : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 3 else 4 * a (n-1) - a (n-2)
  let b : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 3 else (b (n-1) ^ 2 + 2) / b (n-2)
  let c : ℕ → ℕ := λ n, if n = 1 then 1 else let m := n-1 in 2 * c m + (3 * c m ^ 2 - 2).sqrt
  a n = b n ∧ b n = c n :=
by
  sorry

end sequences_equal_l742_742587


namespace tan_150_eq_neg_inv_sqrt3_l742_742069

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742069


namespace tan_150_degrees_l742_742047

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742047


namespace x_minus_y_solution_l742_742188

theorem x_minus_y_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end x_minus_y_solution_l742_742188


namespace range_of_a_l742_742604

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l742_742604


namespace collinear_points_l742_742998

variable (a : ℝ) (A B C : ℝ × ℝ)

-- Conditions given in the problem
def point_A := (a, 2 : ℝ)
def point_B := (5, 1 : ℝ)
def point_C := (-4, 2 * a : ℝ)

-- Collinearity condition
def collinear (x y z : ℝ): Prop :=
  (x.1 - y.1) * (y.2 - z.2) = (y.1 - z.1) * (x.2 - y.2)

theorem collinear_points :
  collinear (point_A a) (point_B) (point_C a) →
  a = 5 + sqrt 21 ∨ a = 5 - sqrt 21 :=
by {
  sorry,
}

end collinear_points_l742_742998


namespace total_right_handed_players_l742_742262

theorem total_right_handed_players
  (total_players : ℕ)
  (total_throwers : ℕ)
  (left_handed_throwers_perc : ℕ)
  (right_handed_thrower_runs : ℕ)
  (left_handed_thrower_runs : ℕ)
  (total_runs : ℕ)
  (batsmen_to_allrounders_run_ratio : ℕ)
  (proportion_left_right_non_throwers : ℕ)
  (left_handed_non_thrower_runs : ℕ)
  (left_handed_batsmen_eq_allrounders : Prop)
  (left_handed_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (total_right_handed_thrower_runs : ℕ)
  (total_left_handed_thrower_runs : ℕ)
  (total_throwers_runs : ℕ)
  (total_non_thrower_runs : ℕ)
  (allrounder_runs : ℕ)
  (batsmen_runs : ℕ)
  (left_handed_batsmen : ℕ)
  (left_handed_allrounders : ℕ)
  (total_left_handed_non_throwers : ℕ)
  (right_handed_non_throwers : ℕ)
  (total_right_handed_players : ℕ) :
  total_players = 120 →
  total_throwers = 55 →
  left_handed_throwers_perc = 20 →
  right_handed_thrower_runs = 25 →
  left_handed_thrower_runs = 30 →
  total_runs = 3620 →
  batsmen_to_allrounders_run_ratio = 2 →
  proportion_left_right_non_throwers = 5 →
  left_handed_non_thrower_runs = 720 →
  left_handed_batsmen_eq_allrounders →
  left_handed_throwers = total_throwers * left_handed_throwers_perc / 100 →
  right_handed_throwers = total_throwers - left_handed_throwers →
  total_right_handed_thrower_runs = right_handed_throwers * right_handed_thrower_runs →
  total_left_handed_thrower_runs = left_handed_throwers * left_handed_thrower_runs →
  total_throwers_runs = total_right_handed_thrower_runs + total_left_handed_thrower_runs →
  total_non_thrower_runs = total_runs - total_throwers_runs →
  allrounder_runs = total_non_thrower_runs / (batsmen_to_allrounders_run_ratio + 1) →
  batsmen_runs = batsmen_to_allrounders_run_ratio * allrounder_runs →
  left_handed_batsmen = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  left_handed_allrounders = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  total_left_handed_non_throwers = left_handed_batsmen + left_handed_allrounders →
  right_handed_non_throwers = total_left_handed_non_throwers * proportion_left_right_non_throwers →
  total_right_handed_players = right_handed_throwers + right_handed_non_throwers →
  total_right_handed_players = 164 :=
by sorry

end total_right_handed_players_l742_742262


namespace avg_cans_used_per_game_l742_742501

theorem avg_cans_used_per_game (total_rounds : ℕ) (games_first_round : ℕ) (games_second_round : ℕ)
  (games_third_round : ℕ) (games_finals : ℕ) (total_tennis_balls : ℕ) (balls_per_can : ℕ)
  (h1 : total_rounds = 4) (h2 : games_first_round = 8) (h3 : games_second_round = 4) 
  (h4 : games_third_round = 2) (h5 : games_finals = 1) (h6 : total_tennis_balls = 225) 
  (h7 : balls_per_can = 3) :
  let total_games := games_first_round + games_second_round + games_third_round + games_finals
  let total_cans_used := total_tennis_balls / balls_per_can
  let avg_cans_per_game := total_cans_used / total_games
  avg_cans_per_game = 5 :=
by {
  -- proof steps here
  sorry
}

end avg_cans_used_per_game_l742_742501


namespace find_k_of_hyperbola_focus_l742_742978

def hyperbola_focus (k : ℝ) : Prop :=
  sqrt (1 + (1 / k)) = 3

theorem find_k_of_hyperbola_focus : ∃ k : ℝ, hyperbola_focus k ∧ k = 1 / 8 :=
by
  use 1 / 8
  split
  · unfold hyperbola_focus
    sorry
  · sorry

end find_k_of_hyperbola_focus_l742_742978


namespace CarlyWorkedOnElevenDogs_l742_742893

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l742_742893


namespace area_of_circumscribed_circle_l742_742447

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742447


namespace find_ab_l742_742627

theorem find_ab (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by
  sorry

end find_ab_l742_742627


namespace find_hyperbola_l742_742176

variables {a b x y : ℝ}
noncomputable def parabola_locus := {p : ℝ × ℝ | p.2^2 = 8 * p.1}
noncomputable def latus_rectum := {p : ℝ × ℝ | p.1 = -2}
noncomputable def asymptote := {p : ℝ × ℝ | p.1 + sqrt(3) * p.2 = 0}
noncomputable def hyperbola := {p : ℝ × ℝ | p.1^2 / 3 - p.2^2 = 1}

theorem find_hyperbola :
  (∃ focus : ℝ × ℝ, focus ∈ latus_rectum ∧ focus ∈ parabola_locus ∧
    (focus.1^2 + b^2 = 4) ∧
    (∀ p ∈ asymptote, p.2 = b / a * p.1) ∧
    a = sqrt 3 ∧ b = 1)
  → (∃ h : ℝ × ℝ → Prop, h = hyperbola) :=
by {
  -- proof omitted
  sorry
}

end find_hyperbola_l742_742176


namespace tan_150_eq_neg_sqrt_3_l742_742013

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742013


namespace tan_150_degrees_l742_742046

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742046


namespace solve_triples_l742_742811

theorem solve_triples (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  a^2 + b^2 = n * Nat.lcm a b + n^2 ∧
  b^2 + c^2 = n * Nat.lcm b c + n^2 ∧
  c^2 + a^2 = n * Nat.lcm c a + n^2 →
  ∃ k : ℕ, 0 < k ∧ a = k ∧ b = k ∧ c = k :=
by
  intros h
  sorry

end solve_triples_l742_742811


namespace coordinates_of_P_l742_742588

variables (P : ℝ × ℝ) [InSecondQuadrant : P.1 < 0 ∧ 0 < P.2] [DistToX : |P.2| = 4] [DistToY : |P.1| = 3]

theorem coordinates_of_P :
  P = (-3, 4) :=
sorry

end coordinates_of_P_l742_742588


namespace neg_prop_l742_742306

theorem neg_prop : ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 4) ↔ ∃ x : ℝ, x^2 - 2 * x + 4 > 4 := 
by 
  sorry

end neg_prop_l742_742306


namespace tan_150_deg_l742_742027

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742027


namespace tan_150_eq_neg_inv_sqrt_3_l742_742094

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742094


namespace polynomial_divisibility_condition_l742_742927

theorem polynomial_divisibility_condition (d : ℕ) (hd_pos : d > 0) :
    (∃ P Q : ℝ[X], P.degree = d ∧ P^2 + 1 = (X^2 + 1) * Q^2) ↔ odd d :=
sorry

end polynomial_divisibility_condition_l742_742927


namespace tan_150_eq_neg_sqrt3_div_3_l742_742100

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742100


namespace tan_150_degrees_l742_742049

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742049


namespace arithmetic_sequence_k_value_l742_742675

theorem arithmetic_sequence_k_value (a1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a1 = 1)
  (h2 : d = 2)
  (h3 : ∀ k : ℕ, S (k+2) - S k = 24) : k = 5 := 
sorry

end arithmetic_sequence_k_value_l742_742675


namespace creature_dressing_orders_l742_742488

-- Definition of the problem in Lean
theorem creature_dressing_orders :
  let total_items := 18
  let sock_before_shoe_pairs := 3  -- possible pairs
  let arms := 6
  (fact total_items) / (sock_before_shoe_pairs ^ arms) = fact 18 / 3 ^ 6 := 
by
  sorry

end creature_dressing_orders_l742_742488


namespace exponent_fraction_equals_five_fourths_l742_742405

theorem exponent_fraction_equals_five_fourths :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5 / 4 :=
by
  sorry

end exponent_fraction_equals_five_fourths_l742_742405


namespace zeros_divide_sequence_equally_l742_742712

noncomputable def Φ_m (m : ℕ) : ℕ → ℕ := sorry

def is_cyclic_sequence (seq : ℕ → ℕ) : Prop := sorry

def is_zero (seq : ℕ → ℕ) (n : ℕ) : Prop := seq n = 0

def divides_into_equal_arcs (seq : ℕ → ℕ) (zeros : set ℕ) : Prop :=
∀ i ∈ zeros, ∃ j ∈ zeros, i ≠ j ∧ ∀ k, i ≤ k ∧ k ≤ j → is_zero seq k

theorem zeros_divide_sequence_equally (m : ℕ) :
  is_cyclic_sequence (Φ_m m) →
  (∃ zeros : set ℕ, ∀ n ∈ zeros, is_zero (Φ_m m) n) →
  divides_into_equal_arcs (Φ_m m) {n | is_zero (Φ_m m) n} :=
sorry

end zeros_divide_sequence_equally_l742_742712


namespace simplify_sqrt_product_l742_742886

theorem simplify_sqrt_product (y : ℝ) (hy : 0 ≤ y) : 
  (√(48 * y) * √(18 * y) * √(50 * y)) = 120 * y * √(3 * y) := 
by
  sorry

end simplify_sqrt_product_l742_742886


namespace vegetable_planting_methods_l742_742518

theorem vegetable_planting_methods :
  let vegetables := ["cucumber", "cabbage", "rape", "lentils"]
  let cucumber := "cucumber"
  let other_vegetables := ["cabbage", "rape", "lentils"]
  let choose_2_out_of_3 := Nat.choose 3 2
  let arrangements := Nat.factorial 3
  total_methods = choose_2_out_of_3 * arrangements := by
  let total_methods := 3 * 6
  sorry

end vegetable_planting_methods_l742_742518


namespace right_triangle_area_l742_742151

theorem right_triangle_area (a : ℝ) (h : a > 2)
  (h_arith_seq : a - 2 > 0)
  (pythagorean : (a - 2)^2 + a^2 = (a + 2)^2) :
  (1 / 2) * (a - 2) * a = 24 :=
by
  sorry

end right_triangle_area_l742_742151


namespace tan_150_eq_neg_inv_sqrt_3_l742_742097

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742097


namespace integer_solutions_exist_l742_742183

theorem integer_solutions_exist (x y : ℤ) : 
  12 * x^2 + 7 * y^2 = 4620 ↔ 
  (x = 7 ∧ y = 24) ∨ 
  (x = -7 ∧ y = 24) ∨
  (x = 7 ∧ y = -24) ∨
  (x = -7 ∧ y = -24) ∨
  (x = 14 ∧ y = 18) ∨
  (x = -14 ∧ y = 18) ∨
  (x = 14 ∧ y = -18) ∨
  (x = -14 ∧ y = -18) :=
sorry

end integer_solutions_exist_l742_742183


namespace y_value_l742_742990

noncomputable def find_y (y : ℝ) (α : ℝ) : ℝ :=
  if h₁ : (sin α = sqrt 13 / 13) ∧ (sin α = y / sqrt (3 + y^2)) then y
  else 0

theorem y_value (y : ℝ) (α : ℝ) 
  (h₁ : sin α = sqrt 13 / 13)
  (h₂ : sin α = y / sqrt (3 + y^2)) : 
  y = 1/2 := by
  sorry

end y_value_l742_742990


namespace part1_part2_l742_742985

-- Definition of the function f(x)
def f (x : ℝ) : ℝ :=
  sqrt 3 * (sin x) ^ 2 + (cos (π / 4 - x)) ^ 2 - (1 + sqrt 3) / 2

-- Statement for part (1)
theorem part1 : (x ∈ set.Icc 0 (π / 2)) → (∃ x₀ : ℝ, f x₀ = 1) :=
sorry

-- Statement for part (2)
theorem part2 (A B C : ℝ) (h₁ : A < B) (h₂ : f A = 1 / 2) (h₃ : f B = 1 / 2) :
  BC / AB = sqrt 2 :=
sorry

end part1_part2_l742_742985


namespace tan_150_eq_neg_sqrt_3_l742_742019

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742019


namespace even_fn_increasing_max_val_l742_742747

variable {f : ℝ → ℝ}

theorem even_fn_increasing_max_val (h_even : ∀ x, f x = f (-x))
    (h_inc_0_5 : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 5 → f x ≤ f y)
    (h_dec_5_inf : ∀ x y, 5 ≤ x → x ≤ y → f y ≤ f x)
    (h_f5 : f 5 = 2) :
    (∀ x y, -5 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y) ∧ (∀ x, -5 ≤ x → x ≤ 0 → f x ≤ 2) :=
by
    sorry

end even_fn_increasing_max_val_l742_742747


namespace tan_150_eq_neg_inv_sqrt3_l742_742072

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742072


namespace min_value_expr_l742_742930

theorem min_value_expr : ∃ x : ℝ, ∀ y : ℝ, 4^x - 2^(x+1) + 4 ≥ 3 ∧ (4^x - 2^(x+1) + 4 = 3 → y = x) :=
by
  sorry

end min_value_expr_l742_742930


namespace determine_a_for_nonnegative_function_l742_742987

def function_positive_on_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → a * x^3 - 3 * x + 1 ≥ 0

theorem determine_a_for_nonnegative_function :
  ∀ (a : ℝ), function_positive_on_interval a ↔ a = 4 :=
by
  sorry

end determine_a_for_nonnegative_function_l742_742987


namespace cube_volume_equality_l742_742860

open BigOperators Real

-- Definitions
def initial_volume : ℝ := 1

def removed_volume (x : ℝ) : ℝ := x^2

def removed_volume_with_overlap (x y : ℝ) : ℝ := x^2 - (x^2 * y)

def remaining_volume (a b c : ℝ) : ℝ := 
  initial_volume - removed_volume c - removed_volume_with_overlap b c - removed_volume_with_overlap a c - removed_volume_with_overlap a b + (c^2 * b)

-- Main theorem to prove
theorem cube_volume_equality (c b a : ℝ) (hcb : c < b) (hba : b < a) (ha1 : a < 1):
  (c = 1 / 2) ∧ 
  (b = (1 + Real.sqrt 17) / 8) ∧ 
  (a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64) :=
sorry

end cube_volume_equality_l742_742860


namespace minimum_distance_l742_742945

open Real

theorem minimum_distance :
  ∀ (P : ℝ × ℝ), 
  (P.1 ^ 2 - P.2 - log P.1 = 0) →
  let d := abs (P.1 - P.2 - 2) / sqrt(2) in
  d = sqrt(2) := 
by
  sorry

end minimum_distance_l742_742945


namespace proof_problem_l742_742147

variables (a b k : ℝ)
noncomputable def f (x : ℝ) := (b * x + 1) / (2 * x + a)

theorem proof_problem (h1 : a * b ≠ 2) 
    (h2 : (f a b) x * (f a b) (1 / x) = k) 
    (h3 : (f a b) ((f a b) 1) = k / 2) : 
    (k = 1) ∧ (a = sqrt 2 ∨ a = -sqrt 2) ∧ (b = 2) :=
sorry

end proof_problem_l742_742147


namespace coeff_x2y2_in_expansion_l742_742796

theorem coeff_x2y2_in_expansion :
  let binom := λ n k : ℕ, Nat.choose n k,
      coeff_x2y2 := binom 5 2 * binom 8 4
  in coeff_x2y2 = 700 :=
by
  -- Here we define the binom to be Nat.choose and multiply the computed coefficients
  -- coeff_x2y2 is the product of binomial coefficients for the respective terms
  sorry

end coeff_x2y2_in_expansion_l742_742796


namespace usual_time_cover_distance_l742_742793

theorem usual_time_cover_distance :
  ∃ T : ℝ, ∃ S T_60 : ℝ, (T_60 = T + 35 / 60) ∧ (S * T = 0.6 * S * T_60) ∧ T = 52.5 :=
by
  use 52.5
  use S
  use 52.5 + 35 / 60
  sorry

end usual_time_cover_distance_l742_742793


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742083

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742083


namespace product_of_solutions_l742_742916

theorem product_of_solutions (x : ℝ) :
  ∃ (α β : ℝ), (x^2 - 4*x - 21 = 0) ∧ α * β = -21 := sorry

end product_of_solutions_l742_742916


namespace max_measurable_with_four_weights_l742_742330

variable (weight : Fin 4 → ℤ)

def possible_measurable_weights : Finset ℤ :=
  Finset.image (λ (s : Fin 4 → ℤ) => s 0 + s 1 + s 2 + s 3) (Finset.univ)

theorem max_measurable_with_four_weights :
  ∃ (weight : Fin 4 → ℤ),
    (∀ mass ∈ Icc 1 81, mass ∈ possible_measurable_weights weight) :=
by
  sorry

end max_measurable_with_four_weights_l742_742330


namespace no_knights_tour_on_4xn_l742_742227

/- Define the board dimensions -/
def board_w : ℕ := 4

def knight_movement_valid (N : ℕ) (move : ℕ × ℕ) : Prop :=
  (abs (move.fst - 2) + abs (move.snd - 1) = 3) ∨ (abs (move.fst - 1) + abs (move.snd - 2) = 3)

/- The main theorem statement that we need to prove -/
theorem no_knights_tour_on_4xn : ∀ (N : ℕ), N > 0 → ¬ ∃ tour : list (ℕ × ℕ), 
  (∀ coord ∈ tour, coord.fst < 4 ∧ coord.snd < N) ∧ 
  (tour.head = some (0, 0)) ∧ 
  (tour.last = some (0, 0)) ∧ 
  (∀ i < tour.length - 1, knight_movement_valid N (tour.nth_le i sorry, tour.nth_le (i+1) sorry)) ∧
  (list.nodup tour) ∧ 
  (tour.length = 4 * N) :=
by
  sorry

end no_knights_tour_on_4xn_l742_742227


namespace line_eq_is_correct_l742_742297

-- Define the conditions
def x_intercept : ℝ := 2
def inclination_angle : ℝ := 135

-- Define the slope from the inclination angle; slope = tan(inclination_angle)
def slope : ℝ := -1  -- since tan(135°) = -1

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = slope * x + x_intercept

-- Prove the equation of the line
theorem line_eq_is_correct :
  ∀ x y, line_eq x y ↔ y = -x + 2 := 
sorry

end line_eq_is_correct_l742_742297


namespace jill_study_hours_l742_742660

theorem jill_study_hours (x : ℕ) (h_condition : x + 2*x + (2*x - 1) = 9) : x = 2 :=
by
  sorry

end jill_study_hours_l742_742660


namespace tan_150_degrees_l742_742050

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742050


namespace tan_150_deg_l742_742037

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742037


namespace evaluate_expression_l742_742921

def expression (x y : ℤ) : ℤ :=
  y * (y - 2 * x) ^ 2

theorem evaluate_expression : 
  expression 4 2 = 72 :=
by
  -- Proof will go here
  sorry

end evaluate_expression_l742_742921


namespace handshakes_count_l742_742513

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end handshakes_count_l742_742513


namespace proof_problem_l742_742575

variables {Line Plane : Type} [HasSubset Line Plane] [HasInter Line Plane]

-- Define the subset relation
def is_subset (l : Line) (α : Plane) : Prop := l ⊂ α

-- Define the parallel relation between planes
def parallel (α β : Plane) : Prop := sorry

-- Define the perpendicular relation between lines and planes
def perp_line_plane (l : Line) (α : Plane) : Prop := sorry

-- Define the perpendicular relation between planes
def perp_planes (α β : Plane) : Prop := sorry

-- Define the lines l and m
variable (l : Line)
variable (m : Line)

-- Define the planes α and β
variable (α : Plane)
variable (β : Plane)

-- Given conditions
axiom l_in_α : is_subset l α
axiom m_in_β : is_subset m β

-- Given problem to prove
theorem proof_problem : perp_line_plane l β → perp_planes α β :=
sorry

end proof_problem_l742_742575


namespace quadratic_polynomial_solution_count_l742_742960

theorem quadratic_polynomial_solution_count
  (f : ℝ → ℝ)
  (hf : ∃ a b c, ∀ x, f(x) = a * x^2 + b * x + c)
  (h_solutions : (∃ x₁ x₂ x₃, (f(x₁))^3 - 4 * (f(x₁)) = 0 ∧ (f(x₂))^3 - 4 * (f(x₂)) = 0 ∧ (f(x₃))^3 - 4 * (f(x₃)) = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) :
  ∃ y₁ y₂ : ℝ, (f(y₁))^2 = 1 ∧ (f(y₂))^2 = 1 ∧ y₁ ≠ y₂ :=
by
  sorry

end quadratic_polynomial_solution_count_l742_742960


namespace tan_150_eq_l742_742005

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742005


namespace trig_identity_special_angles_l742_742770

theorem trig_identity_special_angles :
  cos (real.pi / 4) * sin (real.pi / 12) - sin (real.pi / 4) * cos (real.pi / 12) = -1/2 :=
by {
  sorry
}

end trig_identity_special_angles_l742_742770


namespace probability_train_or_plane_probability_not_ship_l742_742277

def P_plane : ℝ := 0.2
def P_ship : ℝ := 0.3
def P_train : ℝ := 0.4
def P_car : ℝ := 0.1
def mutually_exclusive : Prop := P_plane + P_ship + P_train + P_car = 1

theorem probability_train_or_plane : mutually_exclusive → P_train + P_plane = 0.6 := by
  intro h
  sorry

theorem probability_not_ship : mutually_exclusive → 1 - P_ship = 0.7 := by
  intro h
  sorry

end probability_train_or_plane_probability_not_ship_l742_742277


namespace circumscribed_circle_radius_l742_742641

theorem circumscribed_circle_radius (b c : ℝ) (cosA : ℝ)
  (hb : b = 2) (hc : c = 3) (hcosA : cosA = 1 / 3) : 
  R = 9 * Real.sqrt 2 / 8 :=
by
  sorry

end circumscribed_circle_radius_l742_742641


namespace tan_150_eq_neg_sqrt3_div_3_l742_742108

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742108


namespace tan_150_eq_neg_inv_sqrt3_l742_742055

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742055


namespace minimize_sum_of_distances_at_intersection_l742_742226

-- Define the quadrant points and intersection in the Lean environment
variable {A B C D O : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup O]

-- Assume these points form a convex quadrilateral
def is_convex_quadrilateral {A B C D : Type} : Prop := 
  -- Placeholder definition for demonstrating convex quadrilateral, you'd replace with an actual definition.
  sorry 

-- Define intersection point in Lean
def intersection_of_diagonals {A C B D : Type} (quadrilateral : Type) : O := 
  sorry

-- Define triangle inequality conditions in Lean
def triangle_inequality {A1 A2 C1 C2 : A → ℝ} (O1 : O) : Prop :=
  (A1 O1 + C1 O1 ≥ A2) ∧ (B1 O1 + D1 O1 ≥ B2)

-- Define the Lean theorem statement using these conditions
theorem minimize_sum_of_distances_at_intersection
  {A B C D O : Type}
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup O]
  (h_convex: is_convex_quadrilateral {A, B, C, D})
  (O : Type) 
  (O_point : O = intersection_of_diagonals {A, C, B, D}):
  (∀ (O1 : O), triangle_inequality {A C B D} O1) →
  (∀ O1, A O1 + C O1 + B O1 + D O1 ≥ A O + C O + B O + D O) := sorry

end minimize_sum_of_distances_at_intersection_l742_742226


namespace area_of_circumscribed_circle_l742_742471

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742471


namespace sum_of_ns_l742_742398

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742398


namespace carly_dogs_total_l742_742897

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l742_742897


namespace profit_bicycle_l742_742694

theorem profit_bicycle (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 650) 
  (h2 : x + 2 * y = 350) : 
  x = 150 ∧ y = 100 :=
by 
  sorry

end profit_bicycle_l742_742694


namespace circle_area_of_equilateral_triangle_l742_742444

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742444


namespace sum_of_valid_ns_l742_742343

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742343


namespace sneakers_final_price_correct_l742_742844

/-- A man is purchasing a pair of sneakers at a club store. He receives various discounts in a specified order and a sales tax. 
    This definition aims to calculate the final price he pays. -/
def final_price (original_price : ℝ) (coupon : ℝ) (promotional_discount_rate : ℝ) 
                (membership_discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
let price_after_coupon := original_price - coupon in
let price_after_promotion := price_after_coupon - (promotional_discount_rate * price_after_coupon) in
let price_after_membership := price_after_promotion - (membership_discount_rate * price_after_promotion) in
let sales_tax := sales_tax_rate * price_after_membership in
(price_after_membership + sales_tax).round(2)

theorem sneakers_final_price_correct :
  final_price 120 10 0.05 0.10 0.07 = 100.63 :=
by
  sorry

end sneakers_final_price_correct_l742_742844


namespace Blossom_room_area_square_inches_l742_742772

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

theorem Blossom_room_area_square_inches :
  (let length_feet := 10 in
   let width_feet := 10 in
   let length_inches := feet_to_inches length_feet in
   let width_inches := feet_to_inches width_feet in
   length_inches * width_inches = 14400) :=
by
  let length_feet := 10
  let width_feet := 10
  let length_inches := feet_to_inches length_feet
  let width_inches := feet_to_inches width_feet
  show length_inches * width_inches = 14400
  sorry

end Blossom_room_area_square_inches_l742_742772


namespace ratio_is_four_sevenths_l742_742200

-- We need to assume there are at least 100 students
def at_least_100_students (total_students : ℕ) : Prop := total_students > 100

-- Define the probabilities and ratio conditions
def probability_condition (p q : ℚ) : Prop :=
  p = (3/4) * q ∧ p + q = 1

-- Define the ratio of girls to total students
def ratio_of_girls_to_total_students (girls : ℚ) (total_students : ℚ) : ℚ :=
  girls / total_students

-- Formal theorem stating the given conditions imply the ratio of girls to total students is 4/7
theorem ratio_is_four_sevenths 
  (total_students : ℚ) 
  (h_total : at_least_100_students (total_students.toNat)) 
  (p q : ℚ)
  (h_prob : probability_condition p q) : 
  ratio_of_girls_to_total_students q total_students = 4 / 7 := 
sorry

end ratio_is_four_sevenths_l742_742200


namespace max_value_of_f_l742_742752

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

theorem max_value_of_f : ∃ m, ∀ x, f(x) ≤ m ∧ m = 2 :=
by
  sorry

end max_value_of_f_l742_742752


namespace carly_dog_count_l742_742895

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l742_742895


namespace angle_X_proof_l742_742221

noncomputable def angle_X (Y Z X : ℝ) : Prop :=
  XY = XZ ∧ Z = 3 * Y ∧ Y + X + Z = 180 → X = 36

theorem angle_X_proof (XY XZ : ℝ) (Y Z X : ℝ)
  (h1 : XY = XZ) 
  (h2 : Z = 3 * Y) 
  (h3 : Y + X + Z = 180) : 
  X = 36 :=
by
  sorry

end angle_X_proof_l742_742221


namespace two_friends_each_person_l742_742646

/-- In a group of five people with specific friendship and enmity conditions, 
   we need to prove that every person has exactly two friends. 
   
   Conditions:
     1. Any two people are either friends or enemies.
     2. No three of them are mutual friends.
     3. No three of them are mutual enemies.
-/
theorem two_friends_each_person {P : Type} [Fintype P] (h_card : Fintype.card P = 5)
  (friend enemy : P → P → Prop)
  (h_friend_or_enemy : ∀ x y : P, x ≠ y → friend x y ∨ enemy x y)
  (h_no_three_friends : ∀ x y z : P, friend x y → friend y z → friend z x → x = y ∨ y = z ∨ z = x)
  (h_no_three_enemies : ∀ x y z : P, enemy x y → enemy y z → enemy z x → x = y ∨ y = z ∨ z = x) :
  ∀ x : P, (Finset.filter (friend x) (Finset.univ : Finset P)).card = 2 :=
by
  sorry

end two_friends_each_person_l742_742646


namespace tan_150_degrees_l742_742048

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742048


namespace school_saves_money_l742_742494

def num_devices : Nat := 50
def cost_40_pkg : Nat := 40
def coverage_40_pkg : Nat := 5
def cost_60_pkg : Nat := 60
def coverage_60_pkg : Nat := 10

theorem school_saves_money :
  let num_40_pkgs := num_devices / coverage_40_pkg in
  let cost_40_total := cost_40_pkg * num_40_pkgs in
  let num_60_pkgs := num_devices / coverage_60_pkg in
  let cost_60_total := cost_60_pkg * num_60_pkgs in
  cost_40_total - cost_60_total = 100 :=
by
  sorry

end school_saves_money_l742_742494


namespace concyclic_iff_orthocenter_l742_742651

variables {A B C D P E F O1 O2 : Type*}
variables [metric_space E]
variables (triangle_ABC : A B C) (altitude_AD : A D) 
variables (point_P_on_AD : P) (perpendicular_PE_to_AC : P E)
variables (perpendicular_PF_to_AB : P F)
variables (circumcenter_O1 : E) (circumcenter_O2 : F)

noncomputable def are_concyclic := ∀ (O1 O2 E F : E), ∃ k, 
   (line O1 O2) ⁿ⁺⁽(rectangle E F) ∧ 
   (four_cycle O1 O2 E F)

theorem concyclic_iff_orthocenter (h_triangle : ∠A + ∠B + ∠C = 180)
  (h_altitude : D ∈ line_segment A C)
  (h_pe_perpendicular : ∃ (E : Type*), line_segement AC ⊥ ∈)
  (h_pf_perpendicular : ∃ (F : Type*), line_segment AB ⊥ ∈)
  (O1 : circumcenter (triangle BDF))
  (O2 : circumcenter (triangle CDE))
  : (point P == orthocenter (triple A B C) ↔
      exists circle, are_concyclic O1 O2 E F) :=
sorry

end concyclic_iff_orthocenter_l742_742651


namespace find_a_l742_742996

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end find_a_l742_742996


namespace tan_150_degrees_l742_742045

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742045


namespace sum_of_integer_values_l742_742357

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742357


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742077

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742077


namespace find_a_b_l742_742582

theorem find_a_b (a b : ℤ) (h : ∀ (x1 x2 : ℤ), x1 + x2 = a ∧ x1 * x2 = b) :
  (a = 2 ∧ b = 0) ∨ (a = 0 ∧ b = -2) :=
begin
  sorry
end

end find_a_b_l742_742582


namespace hyperbola_focal_length_l742_742745

theorem hyperbola_focal_length (a b : ℝ) (hb : b ≠ 0) : 
  let c := real.sqrt (a^2 + b^2) in
  c = 4 → 2 * c = 4 :=
by
  sorry

end hyperbola_focal_length_l742_742745


namespace total_cost_oranges_mangoes_l742_742534

theorem total_cost_oranges_mangoes
  (initial_price_orange : ℝ)
  (initial_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  initial_price_orange = 40 →
  initial_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  let new_price_orange := initial_price_orange * (1 + price_increase_percentage)
  let new_price_mango := initial_price_mango * (1 + price_increase_percentage)
  let total_cost_oranges := new_price_orange * quantity_oranges
  let total_cost_mangoes := new_price_mango * quantity_mangoes
  let total_cost := total_cost_oranges + total_cost_mangoes in
  total_cost = 1035 :=
by
  intros h_orange h_mango h_percentage h_qty_oranges h_qty_mangoes
  let new_price_orange := 40 * (1 + 0.15)
  let new_price_mango := 50 * (1 + 0.15)
  let total_cost_oranges := new_price_orange * 10
  let total_cost_mangoes := new_price_mango * 10
  let total_cost := total_cost_oranges + total_cost_mangoes
  sorry

end total_cost_oranges_mangoes_l742_742534


namespace remainder_equality_l742_742972

theorem remainder_equality (P P' : ℕ) (h1 : P = P' + 10) 
  (h2 : P % 10 = 0) (h3 : P' % 10 = 0) : 
  ((P^2 - P'^2) % 10 = 0) :=
by
  sorry

end remainder_equality_l742_742972


namespace similarity_property_cyclic_property_l742_742424

/-
A convex quadrilateral ABCD is given, with a point P inside such that the angle ∠PXY is acute for any two adjacent vertices X and Y of ABCD.
-/
variables (A B C D P : Point) (h_convex : ConvexQuadrilateral A B C D) (h_angle_acute : ∀ X Y : Point, Adjacent X Y A B C D → AcuteAngle ∠PXY)

def reflect_point (P : Point) (l : Line) : Point := 
sorry -- Assume the reflective function reflecting P over line l is defined

def Q_sequence (n : Nat) : Quadrilateral := 
sorry -- Assume the construction of sequence Q_n defined with reflections

/-
Question 1: Suppose Q_n is defined as described. Prove that Q_{1997} is similar to Q_1, Q_5, Q_9 given conditions.
-/
theorem similarity_property : Similar (Q_sequence 1997) (Q_sequence 1) ∧ Similar (Q_sequence 1997) (Q_sequence 5) ∧ Similar (Q_sequence 1997) (Q_sequence 9) :=
by sorry

/-
Question 2: Suppose Q_n is defined as described. If Q_{1997} is cyclic, then prove that Q_k is cyclic for all odd k, 1 ≤ k ≤ 12.
-/
theorem cyclic_property (h_cyclic : Cyclic (Q_sequence 1997)) : ∀ k : Nat, (1 ≤ k ∧ k ≤ 12 ∧ k % 2 = 1) → Cyclic (Q_sequence k) :=
by sorry

end similarity_property_cyclic_property_l742_742424


namespace correct_percent_quarters_l742_742412

def dimes := 30
def quarters := 40
def half_dollars := 10

def value_dime := 10
def value_quarter := 25
def value_half_dollar := 50

def total_value_dimes := dimes * value_dime
def total_value_quarters := quarters * value_quarter
def total_value_half_dollars := half_dollars * value_half_dollar

def total_value_coins := total_value_dimes + total_value_quarters + total_value_half_dollars

def percent_value_quarters := (total_value_quarters.toFloat / total_value_coins.toFloat) * 100

theorem correct_percent_quarters : percent_value_quarters ≈ 55.56 := by
  sorry

end correct_percent_quarters_l742_742412


namespace evaluate_v_sum_l742_742876

noncomputable def v (x : ℝ) : ℝ := 2 * Real.sin (π * x / 2) - x

theorem evaluate_v_sum : 
  v (-3.14) + v (-1.57) + v (1.57) + v (3.14) = -2 :=
by
  have h1 : v (-3.14) = 2 * Real.sin (-3.14 * π / 2) + 3.14 := by sorry
  have h2 : v (3.14) = 2 * Real.sin (3.14 * π / 2) - 3.14 := by sorry
  have h3 : v (-1.57) = 2 * Real.sin (-1.57 * π / 2) + 1.57 := by sorry
  have h4 : v (1.57) = 2 * Real.sin (1.57 * π / 2) - 1.57 := by sorry
  calc
    v (-3.14) + v (-1.57) + v (1.57) + v (3.14)
        = (2 * Real.sin (-3.14 * π / 2) + 3.14) + 
          (2 * Real.sin (-1.57 * π / 2) + 1.57) +
          (2 * Real.sin (1.57 * π / 2) - 1.57) +
          (2 * Real.sin (3.14 * π / 2) - 3.14) : by rw [h1, h2, h3, h4]
    ... = -2 : by sorry

end evaluate_v_sum_l742_742876


namespace num_brownies_correct_l742_742234

-- Define the conditions (pan dimensions and brownie piece dimensions)
def pan_width : ℕ := 24
def pan_length : ℕ := 15
def piece_width : ℕ := 3
def piece_length : ℕ := 2

-- Define the area calculations for the pan and each piece
def pan_area : ℕ := pan_width * pan_length
def piece_area : ℕ := piece_width * piece_length

-- Define the problem statement to prove the number of brownies
def number_of_brownies : ℕ := pan_area / piece_area

-- The statement we need to prove
theorem num_brownies_correct : number_of_brownies = 60 :=
by
  sorry

end num_brownies_correct_l742_742234


namespace probability_imaginary_roots_l742_742327

theorem probability_imaginary_roots :
  (∫(p : ℝ) in -2..2, ∫(q : ℝ) in -2..2, if (p^2 + q^2 < 1) then 1 else 0) / 16 = π / 16 := 
sorry

end probability_imaginary_roots_l742_742327


namespace order_of_abc_l742_742146

noncomputable def a := Real.log 6 / Real.log 0.7
noncomputable def b := Real.rpow 6 0.7
noncomputable def c := Real.rpow 0.7 0.6

theorem order_of_abc : b > c ∧ c > a := by
  sorry

end order_of_abc_l742_742146


namespace find_sum_a_b_l742_742623

theorem find_sum_a_b (x a b : ℝ) (hx : x^2 + 5 * x + 5 / x + 1 / x^2 = 34)
  (ha : a ∈ ℤ) (hb : b ∈ ℕ) (hx_form : x = a + Real.sqrt b) : a + b = 5 := by
  sorry

end find_sum_a_b_l742_742623


namespace groups_have_square_difference_l742_742757

theorem groups_have_square_difference :
  ∀ (groups : Finset (Finset ℕ)),
    (∀ n ∈ groups, ∀ a b ∈ n, a ≠ b → (a - b)^2 ≠ 1 ∧ (a - b)^2 ≠ 4 ∧ (a - b)^2 ≠ 9 ∧
                                 (a - b)^2 ≠ 16 ∧ (a - b)^2 ≠ 25 ∧ (a - b)^2 ≠ 36) →
    (disjoint groups) → (∃ n ∈ groups, ∃ a b ∈ n, a ≠ b) :=
by
  sorry

end groups_have_square_difference_l742_742757


namespace bm_bisects_ac_l742_742263

noncomputable def bd_circle (A B C D M K L : Point) (hBD : Line A C) :=
  Circle.mk (Segment.mk B D).center (Segment.mk B D).radius

theorem bm_bisects_ac
  (A B C D M K L : Point)
  (hBD : Height A B)
  (hCircle : Circle (Segment.mk B D).center (Segment.mk B D).radius)
  (hK : K ∈ Intersection (Line A B) hCircle)
  (hL : L ∈ Intersection (Line B C) hCircle)
  (hTangentK : Tangent hCircle K M)
  (hTangentL : Tangent hCircle L M) :
  Bisects B M (Segment.mk A C) :=
sorry

end bm_bisects_ac_l742_742263


namespace arrangements_count_l742_742182

theorem arrangements_count : 
  (∑ k in Finset.range 5, (Nat.choose 4 k)^3) = 
  -- This part of the theorem states that the result of counting the valid arrangements is equal to the given sum.
  sorry -- The proof of this theorem is omitted as specified.

end arrangements_count_l742_742182


namespace minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l742_742428

theorem minimal_distance_ln_x_x :
  ∀ (x : ℝ), x > 0 → ∃ (d : ℝ), d = |Real.log x - x| → d ≥ 0 :=
by sorry

theorem minimal_distance_graphs_ex_ln_x :
  ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), ∃ (d : ℝ), y = d → d = 2 :=
by sorry

end minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l742_742428


namespace no_coloring_exists_for_12gon_l742_742865

noncomputable theory

open Set

def exists_coloring (colors : Finset ℕ) (n : ℕ) (polygon_segments : Finset (Fin (n + 1) × Fin (n + 1))) : Prop :=
  ∃ (f : (Fin (n + 1) × Fin (n + 1)) → ℕ), 
    (∀ seg ∈ polygon_segments, f seg ∈ colors) ∧ 
    (∀ c1 c2 c3 ∈ colors, ∃ (v1 v2 v3 : Fin (n + 1)), 
      (v1, v2) ∈ polygon_segments ∧ (v2, v3) ∈ polygon_segments ∧ (v3, v1) ∈ polygon_segments ∧
      f (v1, v2) = c1 ∧ f (v2, v3) = c2 ∧ f (v3, v1) = c3)

theorem no_coloring_exists_for_12gon :
  ¬ exists_coloring (Finset.range 12) 11 (Finset.univ : Finset (Fin 12 × Fin 12)) :=
sorry

end no_coloring_exists_for_12gon_l742_742865


namespace coeff_x9_is_zero_l742_742552

noncomputable def poly_expr (x : ℤ) : Polynomial ℤ :=
  (Polynomial.C (Polynomial.Coeff (1/3) * Polynomial.monomial 3 1) - Polynomial.C (Polynomial.Coeff (3) * Polynomial.monomial (-2) 1))^9

theorem coeff_x9_is_zero (x : ℤ) : (poly_expr x).coeff 9 = 0 :=
  sorry

end coeff_x9_is_zero_l742_742552


namespace sum_of_n_values_l742_742368

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742368


namespace sin_750_l742_742520

theorem sin_750 :
  (∀ θ : ℝ, sin (θ + 360 * π / 180) = sin θ) →
  sin (30 * π / 180) = 1 / 2 →
  sin (750 * π / 180) = 1 / 2 :=
by
  intros periodic sin_30
  sorry

end sin_750_l742_742520


namespace circle_area_equilateral_triangle_l742_742459

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742459


namespace blossom_room_area_l742_742776

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l742_742776


namespace sock_pair_selection_l742_742211

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 5
def num_blue_socks : Nat := 3

def white_odd_positions : List Nat := [1, 3, 5]
def white_even_positions : List Nat := [2, 4]

def brown_odd_positions : List Nat := [1, 3, 5]
def brown_even_positions : List Nat := [2, 4]

def blue_odd_positions : List Nat := [1, 3]
def blue_even_positions : List Nat := [2]

noncomputable def count_pairs : Nat :=
  let white_brown := (white_odd_positions.length * brown_odd_positions.length) +
                     (white_even_positions.length * brown_even_positions.length)
  
  let brown_blue := (brown_odd_positions.length * blue_odd_positions.length) +
                    (brown_even_positions.length * blue_even_positions.length)

  let white_blue := (white_odd_positions.length * blue_odd_positions.length) +
                    (white_even_positions.length * blue_even_positions.length)

  white_brown + brown_blue + white_blue

theorem sock_pair_selection :
  count_pairs = 29 :=
by
  sorry

end sock_pair_selection_l742_742211


namespace count_integers_Q_le_zero_l742_742910

def Q (x : ℤ) : ℤ := 
  (List.range 50).prod (λ k, x - ((k + 1)^2))

theorem count_integers_Q_le_zero :
  (finset.range 2501).filter (λ m, Q m ≤ 0).card = 1300 := 
sorry

end count_integers_Q_le_zero_l742_742910


namespace sum_of_integer_values_l742_742359

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742359


namespace lamps_turned_on_is_9_l742_742207

-- Definitions for the conditions
def total_lamps : ℕ := 10

def Petya_statement := "There are 5 lamps turned on."
def Vasya_statement_1 := "Petya's statement is false."
def Vasya_statement_2 := "There are three turned off lamps."
def Kolya_statement := "An even number of lamps are turned on."

-- The main theorem to show the number of lamps turned on
theorem lamps_turned_on_is_9 :
  ∃ (on_lamps : ℕ), on_lamps = 9 ∧
  ( (Petya_statement = "There are 5 lamps turned on" ∧ on_lamps = 5) ∨
    (Vasya_statement_2 = "There are three turned off lamps" → on_lamps = total_lamps - 3) ∨
    (Kolya_statement = "An even number of lamps are turned on" ∧ Even on_lamps) ) ∧
    ( (Petya_statement <> "There are 5 lamps turned on" ∧ on_lamps ≠ 5) ∧
      (Vasya_statement_2 <> "There are three turned off lamps" → on_lamps ≠ total_lamps - 3) ∧
      (Kolya_statement <> "An even number of lamps are turned on" ∧ ¬ (Even on_lamps) )
    ) ∧
    ((Petya_statement = "There are 5 lamps turned on") ∨ 
     (Vasya_statement_1 = "You are wrong") ∨
     (Vasya_statement_2 = "There are three turned off lamps") ∨
     (Kolya_statement = "An even number of lamps are turned on")) :=
sorry

end lamps_turned_on_is_9_l742_742207


namespace sum_of_valid_n_l742_742371

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742371


namespace max_vertex_visits_l742_742734

-- Define an undirected 3-regular planar graph
structure ThreeRegularPlanarGraph (V : Type u) :=
  (edges : set (V × V))
  (degree : ∀ v : V, (edges.filter (λ e, e.1 = v ∨ e.2 = v)).card = 3)

-- Define the walk of the visitor in the park
-- A visitor alternates left and right turns and returns to the starting vertex
inductive VisitorWalk (G : ThreeRegularPlanarGraph V) : V → list V → Prop
| start (v : V) : VisitorWalk v [v]
| step_left (u v : V) (walk : list V) :
    VisitorWalk u walk → (u, v) ∈ G.edges → VisitorWalk v (walk ++ [v])
| step_right (u v : V) (walk : list V) :
    VisitorWalk u walk → (u, v) ∈ G.edges → VisitorWalk v (walk ++ [v])

-- Prove that the maximum number of times the visitor can enter a vertex is 3
theorem max_vertex_visits (G : ThreeRegularPlanarGraph V) (start : V) :
  ∃ walk, VisitorWalk G start walk ∧ ∀ v ∈ walk, count v walk ≤ 3 :=
sorry

end max_vertex_visits_l742_742734


namespace area_of_circumscribed_circle_l742_742480

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742480


namespace tan_150_eq_neg_sqrt3_div_3_l742_742103

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742103


namespace factorization_a_minus_b_l742_742298

theorem factorization_a_minus_b (a b : ℤ) (y : ℝ) 
  (h1 : 3 * y ^ 2 - 7 * y - 6 = (3 * y + a) * (y + b)) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) : 
  a - b = 5 :=
sorry

end factorization_a_minus_b_l742_742298


namespace necessary_but_not_sufficient_for_inequalities_l742_742240

theorem necessary_but_not_sufficient_for_inequalities (a b : ℝ) :
  (a + b > 4) ↔ (a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequalities_l742_742240


namespace max_length_PQ_l742_742634

noncomputable def f (x : ℝ) : ℝ := cos (π / 4 - x) ^ 2
noncomputable def g (x : ℝ) : ℝ := sqrt 3 * sin (π / 4 + x) * cos (π / 4 + x)

theorem max_length_PQ : ∃ t : ℝ, abs (f t - g t) = 3 / 2 := 
  sorry

end max_length_PQ_l742_742634


namespace total_surface_area_of_box_l742_742854

variable (a b c : ℝ)

theorem total_surface_area_of_box (h1 : 4 * (a + b + c) = 140) (h2 : real.sqrt (a^2 + b^2 + c^2) = 21) :
    2 * (a * b + b * c + c * a) = 784 :=
by 
  sorry

end total_surface_area_of_box_l742_742854


namespace area_of_circumscribed_circle_l742_742450

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742450


namespace min_value_frac_sum_l742_742578

open Real

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) := 
sorry

end min_value_frac_sum_l742_742578


namespace sector_to_cone_height_l742_742837

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l742_742837


namespace floor_sum_not_divisible_by_4_l742_742561

theorem floor_sum_not_divisible_by_4 :
  let s := finset.range 1200 ∪ {1200} in
  (finset.filter (λ n, ¬((nat.floor (1197 / n) + nat.floor (1198 / n)
                         + nat.floor (1199 / n) + nat.floor (1200 / n)) % 4 = 0)) s).card = 36 :=
by
  let s := finset.range 1200 ∪ {1200}
  let count := (finset.filter (λ n, ¬((nat.floor (1197 / n) + nat.floor (1198 / n)
                                     + nat.floor (1199 / n) + nat.floor (1200 / n)) % 4 = 0)) s).card
  have : count = 36, from sorry
  exact this

end floor_sum_not_divisible_by_4_l742_742561


namespace five_dollar_bills_count_l742_742700

theorem five_dollar_bills_count (total_money : ℕ) (bill_denomination : ℕ) (h_total : total_money = 45) (h_denom : bill_denomination = 5) :
  total_money / bill_denomination = 9 :=
by
  rw [h_total, h_denom]
  norm_num

end five_dollar_bills_count_l742_742700


namespace traveling_roads_costs_3600_l742_742856

noncomputable def cost_of_traveling_roads (lawn_length lawn_breadth road_width cost_per_sq_m : ℕ) : ℕ :=
let area_road1 := road_width * lawn_breadth in
let area_road2 := road_width * lawn_length - (road_width * road_width) in
let total_area := area_road1 + area_road2 in
total_area * cost_per_sq_m

theorem traveling_roads_costs_3600 :
  cost_of_traveling_roads 70 60 10 3 = 3600 := 
by
  sorry

end traveling_roads_costs_3600_l742_742856


namespace tan_150_degrees_l742_742051

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l742_742051


namespace complex_quadrant_l742_742740

open Complex

noncomputable def z (i : Complex) := sorry

theorem complex_quadrant (z : ℂ) (h : z / (z - I) = I) : 
  ∃ x y : ℝ, (x = (re (conj z))) ∧ (y = (im (conj z))) ∧ (x > 0) ∧ (y < 0) := 
by 
  sorry

end complex_quadrant_l742_742740


namespace problem_l742_742629

-- Helper definition for point on a line
def point_on_line (x y : ℝ) (a b : ℝ) : Prop := y = a * x + b

-- Given condition: Point P(1, 3) lies on the line y = 2x + b
def P_on_l (b : ℝ) : Prop := point_on_line 1 3 2 b

-- The proof problem: Proving (2, 5) also lies on the line y = 2x + b where b is the constant found using P
theorem problem (b : ℝ) (h: P_on_l b) : point_on_line 2 5 2 b :=
by
  sorry

end problem_l742_742629


namespace area_of_circumscribed_circle_l742_742479

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742479


namespace cone_height_l742_742831

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l742_742831


namespace range_for_a_l742_742973

variable (a : ℝ)

theorem range_for_a (h : ∀ x : ℝ, x^2 + 2 * x + a > 0) : 1 < a := 
sorry

end range_for_a_l742_742973


namespace cone_height_is_correct_l742_742820

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l742_742820


namespace abc_product_l742_742190

theorem abc_product (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 13) (h2 : b * c = 52) (h3 : c * a = 4) : a * b * c = 52 := 
  sorry

end abc_product_l742_742190


namespace ce_de_squared_proof_l742_742687

noncomputable def compute_ce_de_squared (AB CD : ℝ) (radius : ℝ) (BE : ℝ) (angle_AEC : ℝ) : ℝ :=
  let CE_DE_squared := 300
  in CE_DE_squared

-- Conditions extraction to define Lean variables
def diameter_radius : ℝ := 10
def chord_BE : ℝ := 4 * Real.sqrt 2
def angle_AEC : ℝ := 60
def expected_answer : ℝ := 300

-- Main theorem statement asserting the final result given the conditions
theorem ce_de_squared_proof : 
  compute_ce_de_squared diameter_radius chord_BE diameter_radius chord_BE angle_AEC = expected_answer :=
  by
  -- Proof will be replaced here
  sorry

end ce_de_squared_proof_l742_742687


namespace sum_of_valid_n_l742_742338

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742338


namespace tan_150_eq_l742_742003

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742003


namespace set_intersection_example_l742_742993

def universal_set := Set ℝ

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1 ∧ -2 ≤ x ∧ x ≤ 1}

def C : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

def complement (A : Set ℝ) : Set ℝ := {x : ℝ | x ∉ A}

def difference (A B : Set ℝ) : Set ℝ := A \ B

def union (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∨ x ∈ B}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection_example :
  intersection (complement A) (union B C) = {x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 4)} :=
by
  sorry

end set_intersection_example_l742_742993


namespace tan_150_eq_neg_sqrt_3_l742_742021

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742021


namespace sum_binom_solutions_l742_742394

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742394


namespace jake_remaining_money_l742_742228

theorem jake_remaining_money :
  let initial_money := 5000
  let motorcycle_cost := 2800
  let money_left_after_motorcycle := initial_money - motorcycle_cost
  let concert_ticket_cost := money_left_after_motorcycle / 2
  let money_left_after_concert := money_left_after_motorcycle - concert_ticket_cost
  let lost_money := money_left_after_concert / 4
  let final_money_left := money_left_after_concert - lost_money
  final_money_left = 825 := 
by
  let initial_money := 5000
  let motorcycle_cost := 2800
  let money_left_after_motorcycle := initial_money - motorcycle_cost
  let concert_ticket_cost := money_left_after_motorcycle / 2
  let money_left_after_concert := money_left_after_motorcycle - concert_ticket_cost
  let lost_money := money_left_after_concert / 4
  let final_money_left := money_left_after_concert - lost_money
  have : final_money_left = 825 := sorry
  this

end jake_remaining_money_l742_742228


namespace margin_in_terms_of_selling_price_l742_742866

variable (C S M : ℝ) (n : ℕ) (h : M = (1 / 2) * (S - (1 / n) * C))

theorem margin_in_terms_of_selling_price :
  M = ((n - 1) / (2 * n - 1)) * S :=
sorry

end margin_in_terms_of_selling_price_l742_742866


namespace least_positive_n_for_reducible_fraction_l742_742554

theorem least_positive_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (6 * n + 7)) ∧ n = 126 :=
by
  sorry

end least_positive_n_for_reducible_fraction_l742_742554


namespace melons_count_l742_742769

theorem melons_count (w_apples_total w_apple w_2apples w_watermelons w_total w_melons : ℕ) :
  w_apples_total = 4500 →
  9 * w_apple = w_apples_total →
  2 * w_apple = w_2apples →
  5 * 1050 = w_watermelons →
  w_total = w_2apples + w_melons →
  w_total = w_watermelons →
  w_melons / 850 = 5 :=
by
  sorry

end melons_count_l742_742769


namespace tan_150_eq_neg_inv_sqrt_3_l742_742096

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742096


namespace roof_ratio_l742_742763

theorem roof_ratio (L W : ℝ) (h1 : L * W = 576) (h2 : L - W = 36) : L / W = 4 := 
by
  sorry

end roof_ratio_l742_742763


namespace tan_150_eq_neg_inv_sqrt3_l742_742074

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742074


namespace num_divisors_at_least_two_l742_742519

def rad (n : ℕ) : ℕ := n.factors.erase_dup.prod

theorem num_divisors_at_least_two : 
  (finset.filter (λ n : ℕ, ∃ a b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, a ≠ b ∧ n ∣ a^a ∧ n ∣ b^b) (finset.range 100)).card = 20 :=
by
  sorry

end num_divisors_at_least_two_l742_742519


namespace four_digit_even_numbers_count_six_digit_non_adjacent_odd_numbers_count_l742_742329

-- Define the digits used
def digits := {0, 1, 2, 3, 4, 5}

-- Define the function to count four-digit even numbers
def count_four_digit_even_numbers : ℕ :=
  let without_0_in_units := 2 * (perm 4 1 * perm 4 2) in
  let with_0_in_units := perm 5 3 in
    without_0_in_units + with_0_in_units

-- Define the function to count six-digit numbers with non-adjacent odd digits
def count_six_digit_non_adjacent_odd_numbers : ℕ :=
  let total := (perm 3 3) * (perm 4 3) in
  let with_0_in_first := (perm 2 2) * (perm 3 3) in
    total - with_0_in_first

-- Statement of the two theorems to be proven
theorem four_digit_even_numbers_count : count_four_digit_even_numbers = 156 := sorry

theorem six_digit_non_adjacent_odd_numbers_count : count_six_digit_non_adjacent_odd_numbers = 132 := sorry

end four_digit_even_numbers_count_six_digit_non_adjacent_odd_numbers_count_l742_742329


namespace negation_of_existence_l742_742305

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by
  sorry

end negation_of_existence_l742_742305


namespace tan_150_eq_neg_inv_sqrt3_l742_742065

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742065


namespace example_math_problem_l742_742414

theorem example_math_problem :
  (10.5 * 0.24 - 15.15 / 7.5) = 0.5 ∧
  (1 + 11 / 20) = 31 / 20 ∧
  (0.945 / 0.9) = 1.05 ∧
  (1 + 3 / 40) = 43 / 40 ∧
  (4 + 3 / 8) / 7 = 5 / 8 →
  (∃ X : ℚ, X / 0.5 = 4.5 / (9 / 20)) :=
by
  assume h,
  let X := 5,
  use X,
  sorry

end example_math_problem_l742_742414


namespace joe_paint_initial_amount_l742_742230

theorem joe_paint_initial_amount (P : ℕ) (h1 : P / 6 + (5 * P / 6) / 5 = 120) :
  P = 360 := by
  sorry

end joe_paint_initial_amount_l742_742230


namespace total_sticks_needed_l742_742719

/-
Given conditions:
1. Simon's raft needs 36 sticks.
2. Gerry's raft needs two-thirds of the number of sticks that Simon needs.
3. Micky's raft needs 9 sticks more than Simon and Gerry's rafts combined.

Prove that the total number of sticks collected by Simon, Gerry, and Micky is 129.
-/

theorem total_sticks_needed :
  let S := 36 in
  let G := (2/3) * S in
  let M := S + G + 9 in
  S + G + M = 129 :=
by
  let S := 36
  let G := (2/3) * S
  let M := S + G + 9
  have : S + G + M = 129 := sorry
  exact this

end total_sticks_needed_l742_742719


namespace hours_worked_each_day_l742_742140

-- Definitions based on problem conditions
def total_hours_worked : ℝ := 8.0
def number_of_days_worked : ℝ := 4.0

-- Theorem statement to prove the number of hours worked each day
theorem hours_worked_each_day :
  total_hours_worked / number_of_days_worked = 2.0 :=
sorry

end hours_worked_each_day_l742_742140


namespace circle_area_equilateral_triangle_l742_742456

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742456


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742085

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742085


namespace tan_150_eq_neg_inv_sqrt3_l742_742066

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742066


namespace coplanar_lines_l742_742707

theorem coplanar_lines (p: ℝ) :
  ∃ t u : ℝ, 
    (3 - p * t = 2 + u) ∧ (2 + t = 5 + p * u) ∧ (6 + 2 * t = 7 + 3 * u) ↔ 
  (p = 1/2 ∨ p = 1) := 
sorry

end coplanar_lines_l742_742707


namespace sum_of_ns_l742_742401

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742401


namespace find_angle_A_l742_742980

theorem find_angle_A
  (ABC : Type) [triangle ABC]
  (a b c : ℝ) (A B C : ℝ)
  (I : incenter ABC) 
  (DE M N : point ABC)
  (h1 : is_foot_perpendicular I DE M)
  (h2 : extends I M N)
  (h3 : IN = 2 * IM) :
  A = 60 :=
sorry

end find_angle_A_l742_742980


namespace circumscribed_circle_area_l742_742464

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742464


namespace triangle_KLN_is_equilateral_l742_742155

theorem triangle_KLN_is_equilateral
  (A B C L N K : Point)
  (h1 : ∠ ACB = 120)
  (h2 : ∃ D, collinear A D C ∧ collinear B D C ∧ CL bisects ∠ ACB)
  (h3 : CK + CN = CL) :
  equilateral_triangle K L N :=
sorry

end triangle_KLN_is_equilateral_l742_742155


namespace triangle_sine_relation_l742_742222

theorem triangle_sine_relation (A B C a b c R : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C) :
  (a^2 - b^2) / c^2 = (Real.sin (A - B)) / (Real.sin C) :=
by
  sorry

end triangle_sine_relation_l742_742222


namespace sum_of_n_values_l742_742360

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742360


namespace sum_of_integer_values_l742_742353

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742353


namespace loom_weaving_time_l742_742509

theorem loom_weaving_time (rate_of_weaving : ℝ) (time_taken : ℝ) 
  (h_rate: rate_of_weaving = 1.14) 
  (h_time: time_taken = 45.6140350877193) :
  time_taken ≈ 45.614 := 
by
  sorry

end loom_weaving_time_l742_742509


namespace negation_proof_l742_742755

theorem negation_proof (a b : ℝ) : 
  (¬ (a > b → 2 * a > 2 * b - 1)) = (a ≤ b → 2 * a ≤ 2 * b - 1) :=
by
  sorry

end negation_proof_l742_742755


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742081

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742081


namespace find_frequency_range_l742_742178

theorem find_frequency_range (samples : List ℕ) : 
  let count_in_range (lb : ℕ) (ub : ℕ) := (samples.filter (λ x => lb ≤ x ∧ x < ub)).length
  (samples = [12, 7, 11, 12, 11, 12, 10, 10, 9, 8, 13, 12, 10, 9, 6, 11, 8, 9, 8, 10]) →
  ∃ lb ub,
    0.25 = (count_in_range lb ub) / samples.length ∧
    (lb, ub) = (11.5, 13.5) :=
begin
  sorry
end

end find_frequency_range_l742_742178


namespace double_seven_probability_l742_742798

noncomputable def probability_sum_twice_seven : ℚ :=
let total_outcomes := 6 * 6 in
let favorable_outcomes := 6 in
((favorable_outcomes / total_outcomes) * (favorable_outcomes / total_outcomes))

theorem double_seven_probability : probability_sum_twice_seven = 1 / 36 :=
by 
  unfold probability_sum_twice_seven
  simp
  norm_num
  sorry

end double_seven_probability_l742_742798


namespace tan_150_eq_neg_inv_sqrt3_l742_742060

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742060


namespace boxes_with_neither_l742_742644

-- Definitions for conditions
def total_boxes := 15
def boxes_with_crayons := 9
def boxes_with_markers := 5
def boxes_with_both := 4

-- Theorem statement
theorem boxes_with_neither :
  total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 5 :=
by
  sorry

end boxes_with_neither_l742_742644


namespace sum_of_valid_ns_l742_742349

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742349


namespace range_of_m_l742_742579

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l742_742579


namespace minimum_m_minus_n_l742_742602

theorem minimum_m_minus_n (m n : ℝ) (h_fm_gn : (f m = g n)) : 
  (m - n) = (1 / 2) + Real.log 2 :=
by 
  sorry

-- Definitions from the conditions
def f (x : ℝ) : ℝ := Real.log x + 1
def g (x : ℝ) : ℝ := 2 * Real.exp (x - 1 / 2)

-- Proof
#check minimum_m_minus_n

end minimum_m_minus_n_l742_742602


namespace sin_double_angle_l742_742573

-- Define the conditions
def conditions (θ : ℝ) : Prop :=
  let tan_θ := - Real.sqrt 3 in 
  tan θ = tan_θ

-- Define the problem statement
theorem sin_double_angle {θ : ℝ} (h : conditions θ) : Real.sin (2 * θ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_double_angle_l742_742573


namespace lucky_number_325_l742_742635

def is_lucky_number (a : ℕ) : Prop := (a.digits 10).sum = 7

def ordered_lucky_numbers : list ℕ :=
  list.filter is_lucky_number (list.range 100000) -- Range up to 100000 to ensure we cover up to 5 digits

def a (n : ℕ) : ℕ := (ordered_lucky_numbers.nth (n - 1)).get_or_else 0 -- defines the nth lucky number

theorem lucky_number_325 :
  a 325 = 52000 :=
sorry

end lucky_number_325_l742_742635


namespace sin_B_law_of_sines_l742_742639

variable (a b : ℝ) (sinA : ℝ)

theorem sin_B_law_of_sines 
  (h1 : a = 15) 
  (h2 : b = 10) 
  (h3 : sinA = (sqrt 3) / 2) : 
  (b * sinA / a) = (sqrt 3) / 3 :=
by
  sorry

end sin_B_law_of_sines_l742_742639


namespace tan_150_eq_neg_inv_sqrt_3_l742_742093

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742093


namespace multiplicative_magic_square_l742_742204

theorem multiplicative_magic_square:
  (∃ P b c d e f h: ℕ,
    90 * b * c = P ∧
    d * e * f = P ∧
    g * h * 3 = P ∧
    90 * e * 3 = P ∧
    90 * e * g = P ∧
    all_positive := P > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0) →
  (g^g = 30) →
  g = 3 := sorry

end multiplicative_magic_square_l742_742204


namespace a3_equals_1_div_12_l742_742301

-- Definition of the sequence
def seq (n : Nat) : Rat :=
  1 / (n * (n + 1))

-- Assertion to be proved
theorem a3_equals_1_div_12 : seq 3 = 1 / 12 := 
sorry

end a3_equals_1_div_12_l742_742301


namespace sum_of_integer_values_l742_742354

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742354


namespace pentagon_angles_sum_l742_742506

theorem pentagon_angles_sum {α β γ δ ε : ℝ} (h1 : α + β + γ + δ + ε = 180) (h2 : α = 50) :
  β + ε = 230 := 
sorry

end pentagon_angles_sum_l742_742506


namespace mrs_martin_spent_l742_742701

theorem mrs_martin_spent (C B : ℝ) (hB : B = 1.5) (hMrMartin : 2 * C + 5 * B = 14.00) :
  3 * C + 2 * B = 12.75 :=
by
  -- Define the necessary conditions
  have hB_value : B = 1.5 := hB
  -- Substitute B into Mr. Martin's equation and solve for C
  have hMrMartin_simplified : 2 * C + 5 * 1.5 = 14.00 := by rw [hB_value] at hMrMartin
  have hC_solution : C = 3.25 := by
    simp at hMrMartin_simplified
    linarith
  -- Calculate Mrs. Martin's total cost
  calc
    3 * C + 2 * B = 3 * 3.25 + 2 * 1.5 : by rw [hC_solution, hB_value]
    ... = 9.75 + 3.00 : by norm_num
    ... = 12.75 : by norm_num


end mrs_martin_spent_l742_742701


namespace john_recycling_earnings_l742_742664

-- Define the necessary conditions
def weight_per_paper (day: String) : ℕ :=
  if day = "Sunday" then 16 else 8

def papers_per_day := 250
def weeks := 10
def ounces_to_pounds := 16
def pounds_to_tons := 2000
def recycle_rate := 20

-- Theorem: Prove that John made $100 from recycling the papers
theorem john_recycling_earnings :
  let total_weekly_weight_ounces := (6 * weight_per_paper "weekday") + weight_per_paper "Sunday",
      total_weekly_weight_pounds := total_weekly_weight_ounces / ounces_to_pounds,
      total_weight_10_weeks_pounds := total_weekly_weight_pounds * papers_per_day * weeks,
      total_weight_10_weeks_tons := total_weight_10_weeks_pounds / pounds_to_tons,
      earnings := total_weight_10_weeks_tons * recycle_rate
  in earnings = 100 :=
by
  sorry

end john_recycling_earnings_l742_742664


namespace triangle_MNP_is_isosceles_and_right_l742_742311

theorem triangle_MNP_is_isosceles_and_right (
    P A B C M N : Type
    -- Define the points A, B, C such that triangle ABC is isosceles with right angle at A
    -- Define function for length validation 
    -- Same for AN = AB and AM = AC
):
    (BPC_is_isosceles : P ∈ triangle BPC) →
    (angle_BPC_is_90 : angle BPC = 90) →
    (BAN_is_isosceles : AN = AB) →
    (CAM_is_isosceles : AM = AC) →
    isosceles_right_triangle M N P
by
  sorry

end triangle_MNP_is_isosceles_and_right_l742_742311


namespace max_men_with_all_amenities_marrried_l742_742324

theorem max_men_with_all_amenities_marrried :
  let total_men := 100
  let married_men := 85
  let men_with_TV := 75
  let men_with_radio := 85
  let men_with_AC := 70
  (∀ s : Finset ℕ, s.card ≤ total_men) →
  (∀ s : Finset ℕ, s.card ≤ married_men) →
  (∀ s : Finset ℕ, s.card ≤ men_with_TV) →
  (∀ s : Finset ℕ, s.card ≤ men_with_radio) →
  (∀ s : Finset ℕ, s.card ≤ men_with_AC) →
  (∀ s : Finset ℕ, s.card ≤ min married_men (min men_with_TV (min men_with_radio men_with_AC))) :=
by
  intros
  sorry

end max_men_with_all_amenities_marrried_l742_742324


namespace grid_no_four_vertices_square_l742_742711

noncomputable def grid_points (n : ℕ) : set (ℕ × ℕ) :=
{ p | p.1 < n ∧ p.2 < n }

theorem grid_no_four_vertices_square (n : ℕ) (c : ℝ) (h_pos : 0 < c) :
  ∃ S ⊆ grid_points n, S.finite ∧ S.card ≥ Ω(n^(5/3)) ∧
  ∀ (p1 p2 p3 p4 : (ℕ × ℕ)), 
    p1 ∈ S → p2 ∈ S → p3 ∈ S → p4 ∈ S →
    ¬ (p1.1 = p2.1 ∧ p2.2 = p3.2 ∧ p3.1 = p4.1 ∧ p4.2 = p1.2) :=
sorry

end grid_no_four_vertices_square_l742_742711


namespace binom_sum_l742_742383

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742383


namespace series_ln2_series_1_ln2_l742_742807

theorem series_ln2 :
  ∑' n : ℕ, (1 / (n + 1) / (n + 2)) = Real.log 2 :=
sorry

theorem series_1_ln2 :
  ∑' k : ℕ, (1 / ((2 * k + 2) * (2 * k + 3))) = 1 - Real.log 2 :=
sorry

end series_ln2_series_1_ln2_l742_742807


namespace volume_of_tetrahedron_eq_2sqrt6_l742_742733

-- The problem conditions
variables (P Q R S : Point)
variables (PQ PR PS QR QS RS : ℝ)

-- Given conditions
axiom pq_length : PQ = 6
axiom pr_length : PR = 4
axiom ps_length : PS = 3
axiom qr_length : QR = 5
axiom qs_length : QS = 4
axiom rs_length : RS = (15 / 4) * Real.sqrt 2

-- The theorem to prove
theorem volume_of_tetrahedron_eq_2sqrt6 (hPQ : distance P Q = PQ) 
                                       (hPR : distance P R = PR)
                                       (hPS : distance P S = PS )
                                       (hQR : distance Q R = QR)
                                       (hQS : distance Q S = QS)
                                       (hRS : distance R S = RS) :
  volume_of_tetrahedron P Q R S = 2 * Real.sqrt 6 :=
by
  sorry

end volume_of_tetrahedron_eq_2sqrt6_l742_742733


namespace circle_area_equilateral_triangle_l742_742461

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742461


namespace base9_39457_to_base10_is_26620_l742_742232

-- Define the components of the base 9 number 39457_9
def base9_39457 : ℕ := 39457
def base9_digits : List ℕ := [3, 9, 4, 5, 7]

-- Define the base
def base : ℕ := 9

-- Convert each position to its base 10 equivalent
def base9_to_base10 : ℕ :=
  3 * base ^ 4 + 9 * base ^ 3 + 4 * base ^ 2 + 5 * base ^ 1 + 7 * base ^ 0

-- State the theorem
theorem base9_39457_to_base10_is_26620 : base9_to_base10 = 26620 := by
  sorry

end base9_39457_to_base10_is_26620_l742_742232


namespace sequence_2018_distinct_elements_l742_742313

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 1 := a
| (n + 1) := 1 / 2 * (sequence a n - 1 / (sequence a n))

theorem sequence_2018_distinct_elements :
  ∃ a : ℝ, (∀ n m : ℕ, n < 2018 → m < 2018 → n ≠ m → sequence a n ≠ sequence a m) :=
sorry

end sequence_2018_distinct_elements_l742_742313


namespace initial_units_of_phones_l742_742861

theorem initial_units_of_phones
  (X : ℕ) 
  (h1 : 5 = 5) 
  (h2 : X - 5 = 3 + 5 + 7) : 
  X = 20 := 
by
  sorry

end initial_units_of_phones_l742_742861


namespace area_of_triangle_ADE_l742_742415

-- Define points in a 2D plane representing triangles ABC and ABD
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the points A, B, C, D, and E with given coordinates based on the problem statement
def A : Point := { x := 0, y := 0 }
def B : Point := { x := 8, y := 0 }
def C : Point := { x := 0, y := 12 }
def D : Point := { x := 8, y := 8 }
def E : Point := { x := 8 * (12 / (20)), y := 8 * (12 / (20)) }

-- Define the distance function between two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

-- Verify the given conditions
example : 
  distance A B = 8 ∧ distance B D = 8 ∧ distance A C = 12 ∧ 
  distance A D = real.sqrt(8^2 + 8^2) :=
by sorry

-- Proving the area of triangle ADE equals to 16(√2 - 1)
theorem area_of_triangle_ADE :
  let area (P Q R : Point) := 0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)) in
  (area A D E) = 16 * (real.sqrt 2 - 1) :=
by sorry

end area_of_triangle_ADE_l742_742415


namespace inscribed_square_area_ratio_l742_742799

theorem inscribed_square_area_ratio (r : ℝ) (r_pos : 0 < r) :
  let s1 := (r / Real.sqrt 2) in
  let A1 := s1 * s1 in
  let s2 := r * Real.sqrt 2 in
  let A2 := s2 * s2 in
  A1 / A2 = 1 / 4 :=
by
  sorry

end inscribed_square_area_ratio_l742_742799


namespace angle_A_range_l742_742571

-- Definition of the convex quadrilateral and its properties
variables {A B C D : Type} [point : Real] 
variables (AB BC CD DA : ℝ) (angle_A : ℝ)
variables (convex : True) -- Assuming we have a convex quadrilateral
variables (AB_eq : AB = 8) (BC_eq : BC = 4) (CD_eq : CD = 6) (DA_eq : DA = 6)

-- The theorem to be proven
theorem angle_A_range (convex : True) 
(AB_eq : AB = 8) (BC_eq : BC = 4) (CD_eq : CD = 6) (DA_eq : DA = 6) : 
  0 < angle_A ∧ angle_A < 90 :=
by
  sorry

end angle_A_range_l742_742571


namespace blossom_room_area_l742_742775

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l742_742775


namespace oil_needed_to_half_fill_tanker_l742_742791

theorem oil_needed_to_half_fill_tanker :
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  let current_tanker_oil := initial_tanker_oil + poured_oil
  let half_tanker_capacity := initial_tanker_capacity / 2
  let needed_oil := half_tanker_capacity - current_tanker_oil
  needed_oil = 4000 :=
by
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  have h1 : poured_oil = 3000 := by sorry
  let current_tanker_oil := initial_tanker_oil + poured_oil
  have h2 : current_tanker_oil = 6000 := by sorry
  let half_tanker_capacity := initial_tanker_capacity / 2
  have h3 : half_tanker_capacity = 10000 := by sorry
  let needed_oil := half_tanker_capacity - current_tanker_oil
  have h4 : needed_oil = 4000 := by sorry
  exact h4

end oil_needed_to_half_fill_tanker_l742_742791


namespace fair_attendance_l742_742307

theorem fair_attendance :
  let this_year := 600
  let next_year := 2 * this_year
  let total_people := 2800
  let last_year := total_people - this_year - next_year
  (1200 - last_year = 200) ∧ (last_year = 1000) := by
  sorry

end fair_attendance_l742_742307


namespace sector_to_cone_height_l742_742835

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l742_742835


namespace impossible_projection_l742_742166

def shape := 
  { name : String }

def proj_onto_plane (s : shape) : shape :=
  sorry -- Definition of projection is assumed

def line_segment : shape := { name := "Line Segment" }
def straight_line : shape := { name := "Straight Line" }
def circle : shape := { name := "Circle" }
def trapezoid : shape := { name := "Trapezoid" }

theorem impossible_projection :
  ∀ s : shape, 
  (s ≠ line_segment) ∧ (proj_onto_plane s = line_segment) → (s = straight_line) :=
sorry

end impossible_projection_l742_742166


namespace sequence_properties_l742_742679

noncomputable def A_seq (x y : ℝ) : ℕ → ℝ
| 1     => (x + y) / 2
| (n+1) => (A_seq n + H_seq n) / 2

noncomputable def G_seq (x y : ℝ) : ℕ → ℝ
| 1     => sqrt (x * (x + y))
| (n+1) => sqrt (A_seq n * H_seq n)

noncomputable def H_seq (x y : ℝ) : ℕ → ℝ
| 1     => 2 * x * y / (x + y)
| (n+1) => 2 * A_seq n * H_seq n / (A_seq n + H_seq n)

theorem sequence_properties {x y : ℝ} (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (∀ n, A_seq x y n > A_seq x y (n + 1)) ∧
  (∀ n, G_seq x y n = G_seq x y (n + 1)) ∧
  (∀ n, H_seq x y n < H_seq x y (n + 1)) :=
by
  -- proof omitted
  sorry

end sequence_properties_l742_742679


namespace circle_area_equilateral_triangle_l742_742455

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742455


namespace even_degrees_in_large_graph_l742_742269

theorem even_degrees_in_large_graph (G : Type) [graph G] (hG : fintype.card G = 50) :
  ∃ (v w : G), v ≠ w ∧ even (degree v) ∧ even (degree w) := 
sorry

end even_degrees_in_large_graph_l742_742269


namespace minimum_expression_value_l742_742531

noncomputable def minimum_expression (a b c : ℝ) : ℝ :=
  \frac{2}{|a - b|} + \frac{2}{|b - c|} + \frac{2}{|c - a|} + \frac{5}{\sqrt{ab + bc + ca}}

theorem minimum_expression_value (a b c : ℝ) (h1 : ab + bc + ca > 0) (h2 : a + b + c = 1)
    (h3 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  minimum_expression a b c = 10 * sqrt 6 := 
  sorry

end minimum_expression_value_l742_742531


namespace monochromatic_triangle_in_K9_l742_742953

/-- 
Theorem: In a complete graph \( K_9 \) (with 9 vertices) where each pair of vertices 
is connected by an edge, if at least 33 edges are colored either red or blue, 
then there must exist a monochromatic triangle (a triangle with all three edges the same color).
-/
theorem monochromatic_triangle_in_K9 
    (K9 : SimpleGraph (Fin 9)) : 
  ∀ (E : Finset (Sym2 (Fin 9))), E.card ≥ 33 → 
  (∀ (coloring : Sym2 (Fin 9) → Prop), 
    ∃ (triangle : Finset (Sym2 (Fin 9))), triangle.card = 3 ∧ 
    triangle ⊆ E ∧ 
    (∀ e ∈ triangle, coloring e) ∨ 
    (∀ e ∈ triangle, ¬coloring e)) :=
begin
  sorry
end

end monochromatic_triangle_in_K9_l742_742953


namespace simplify_fractions_l742_742720

theorem simplify_fractions :
  (240 / 20) * (6 / 180) * (10 / 4) = 1 :=
by sorry

end simplify_fractions_l742_742720


namespace monotonic_decreasing_interval_l742_742754

noncomputable def f (x : ℝ) : ℝ :=
  2 * x - real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), (0 < a) ∧ (a < 1) ∧ (b = 1 / 2) ∧ (∀ x, a < x ∧ x < b → f' x < 0) :=
sorry

end monotonic_decreasing_interval_l742_742754


namespace max_sum_abc_min_sum_reciprocal_l742_742160

open Real

variables {a b c : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 2)

-- Maximum of a + b + c
theorem max_sum_abc : a + b + c ≤ sqrt 6 :=
by sorry

-- Minimum of 1/(a + b) + 1/(b + c) + 1/(c + a)
theorem min_sum_reciprocal : (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 * sqrt 6 / 4 :=
by sorry

end max_sum_abc_min_sum_reciprocal_l742_742160


namespace tan_150_eq_neg_sqrt_3_l742_742012

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742012


namespace chen_steps_recorded_correct_l742_742437

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end chen_steps_recorded_correct_l742_742437


namespace sum_binom_solutions_l742_742387

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742387


namespace tan_150_eq_neg_inv_sqrt3_l742_742068

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742068


namespace class_average_correct_l742_742189

def class_average_test_A : ℝ :=
  0.30 * 97 + 0.25 * 85 + 0.20 * 78 + 0.15 * 65 + 0.10 * 55

def class_average_test_B : ℝ :=
  0.30 * 93 + 0.25 * 80 + 0.20 * 75 + 0.15 * 70 + 0.10 * 60

theorem class_average_correct :
  round class_average_test_A = 81 ∧
  round class_average_test_B = 79 := 
by 
  sorry

end class_average_correct_l742_742189


namespace eight_in_M_nine_in_M_ten_not_in_M_l742_742873

def M (a : ℤ) : Prop := ∃ b c : ℤ, a = b^2 - c^2

theorem eight_in_M : M 8 := by
  sorry

theorem nine_in_M : M 9 := by
  sorry

theorem ten_not_in_M : ¬ M 10 := by
  sorry

end eight_in_M_nine_in_M_ten_not_in_M_l742_742873


namespace log_ratio_squared_eq_nine_l742_742267

-- Given conditions
variable (x y : ℝ) 
variable (hx_pos : x > 0) 
variable (hy_pos : y > 0)
variable (hx_neq1 : x ≠ 1) 
variable (hy_neq1 : y ≠ 1)
variable (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
variable (heq : x * y = 243)

-- Prove that (\log_3(\tfrac x y))^2 = 9
theorem log_ratio_squared_eq_nine (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq1 : x ≠ 1) (hy_neq1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (heq : x * y = 243) : 
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 :=
sorry

end log_ratio_squared_eq_nine_l742_742267


namespace sum_of_valid_n_l742_742333

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742333


namespace angle_DE_AB_l742_742213

theorem angle_DE_AB 
  (AB_horizontal : ∀ A B : ℝ × ℝ, A.2 = B.2)
  (CE_vertical : ∀ C E : ℝ × ℝ, C.1 = E.1)
  (C_on_AB : ∃ A B C : ℝ × ℝ, AB_horizontal A B ∧ CE_vertical C E ∧ C.2 = A.2)
  (ECD_68 : ∠ ECD = 68)
  (DEB_58 : ∠ DEB = 58) :
  ∠ DE AB = 36 :=
sorry

end angle_DE_AB_l742_742213


namespace height_of_cone_formed_by_rolling_sector_l742_742827

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l742_742827


namespace axis_of_symmetry_exists_l742_742928

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + π / 6)

theorem axis_of_symmetry_exists : 
  ∃ (x : ℝ), f (x) = cos (2 * (5 * π / 12) + π / 6) :=
by
  existsi (5 * π / 12)
  simp [f]
  sorry

end axis_of_symmetry_exists_l742_742928


namespace number_of_pairs_l742_742689

theorem number_of_pairs (n : ℕ) (h : n ≥ 3) : 
  ∃ a : ℕ, a = (n-2) * 2^(n-1) + 1 :=
by
  sorry

end number_of_pairs_l742_742689


namespace range_of_x_minimum_total_cost_l742_742612

theorem range_of_x (x k : ℝ) (hx1 : 20 ≤ x) (hx2 : x ≤ 50) (h_fuel_cost : (30 / 40 - k) = 5 / 8)
  (h_total_cost : (x / 40 - 1 / 8 + 1 / x) ≤ 9 / 10) :
  (20 ≤ x ∧ x ≤ 40) := sorry

theorem minimum_total_cost (x k : ℝ) (hx1 : 20 ≤ x) (hx2 : x ≤ 50) (hk1 : 1/15 ≤ k) (hk2 : k ≤ 1/5) :
  let y := 1/8 - 5 * k / x + 5 / x^2 in
  ((1/15 ≤ k ∧ k < 1/10 → y = (1 - 10 * k^2) / 8) ∧ (1/10 ≤ k ∧ k ≤ 1/5 → y = (11 - 20 * k) / 80)) := sorry

end range_of_x_minimum_total_cost_l742_742612


namespace proposition_C_is_true_l742_742409

theorem proposition_C_is_true :
  (∀ θ : ℝ, 90 < θ ∧ θ < 180 → θ > 90) :=
by
  sorry

end proposition_C_is_true_l742_742409


namespace min_value_of_f_l742_742406

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt 2) * (sin (x / 4)) * (cos (x / 4)) + (sqrt 6) * ((cos (x / 4)) ^ 2) - (sqrt 6) / 2

theorem min_value_of_f : 
  ∀ x : ℝ, x ∈ Icc (-π / 3) (π / 3) → (∃ c : ℝ, c = (sqrt 2) / 2 ∧ ∀ y : ℝ, y ∈ Icc (-π / 3) (π / 3) → f y ≥ c) :=
sorry

end min_value_of_f_l742_742406


namespace length_AG_eq_l742_742161

-- Define the conditions in Lean
variables {A B C D E G : Type}
variables [Classical.decEq A, Classical.decEq B, Classical.decEq C]
variables [Point A, Point B, Point C, Point D, Point E, Point G]
variables [Triangle ABC]
variables (BC_eq_6 : BC = 6)
variables (median_BD : is_median BD ABC)
variables (median_CE : is_median CE ABC)
variables (centroid_G : is_centroid G ABC)
variables (concyclic_ADGE : concyclic A D G E)

-- Define the theorem with the expected proof structure
theorem length_AG_eq : AG = 2 * sqrt 3 :=
sorry

end length_AG_eq_l742_742161


namespace square_side_length_approximately_l742_742496

noncomputable def pi_val : Real := 3.14

def side_length_of_square (radius_of_circle : Real) : Real :=
  let c := 2 * pi * radius_of_circle
  c / 4

theorem square_side_length_approximately : 
  let radius := 3
  let side_length := side_length_of_square radius
  let approx_side_length := (3 * pi_val) / 2
  |approx_side_length - side_length| < 0.01 := 
by 
  sorry

end square_side_length_approximately_l742_742496


namespace sector_to_cone_height_l742_742836

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l742_742836


namespace smallest_square_area_l742_742558

theorem smallest_square_area : 
  ∃ k : ℝ, 
    (∀ x : ℝ, ∃ y : ℝ, y = 2 * x - 17) ∧ 
    (∀ x : ℝ, ∃ y : ℝ, y = x^2) ∧ 
    20 * (k + 1) = (k + 17)^2 / 5 ∧ 
    (k = 3 ∨ k = 63) →
    20 * (3 + 1) = 80 :=
begin
  sorry -- Proof is not required
end

end smallest_square_area_l742_742558


namespace circle_area_of_equilateral_triangle_l742_742446

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742446


namespace sum_first_seven_terms_of_arith_seq_l742_742574

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Conditions: a_2 = 10 and a_5 = 1
def a_2 := 10
def a_5 := 1

-- The sum of the first 7 terms of the sequence
theorem sum_first_seven_terms_of_arith_seq (a d : ℤ) :
  arithmetic_seq a d 1 = a_2 →
  arithmetic_seq a d 4 = a_5 →
  (7 * a + (7 * 6 / 2) * d = 28) :=
by
  sorry

end sum_first_seven_terms_of_arith_seq_l742_742574


namespace volume_of_largest_sphere_from_cube_l742_742900

theorem volume_of_largest_sphere_from_cube : 
  (∃ (V : ℝ), 
    (∀ (l : ℝ), l = 1 → (V = (4 / 3) * π * ((l / 2)^3)) → V = π / 6)) :=
sorry

end volume_of_largest_sphere_from_cube_l742_742900


namespace area_of_tangent_figure_l742_742984

noncomputable theory

def f (x : ℝ) := x^3 - 2*x^2 - x + 1

def f' (x : ℝ) := 3*x^2 - 4*x - 1

def tangent_line (x : ℝ) := -x + 1

def integral_result : ℝ := ∫ t in 0..2, (t^3 - 2*t^2)

theorem area_of_tangent_figure : |integral_result| = 4 / 3 :=
by
  -- Given: f(x) = x^3 - 2x^2 - x + 1,
  --        Point P(1, 0),
  -- Show that the area enclosed by the curve and the tangent line at P is 4/3.
  sorry

end area_of_tangent_figure_l742_742984


namespace expression_evaluation_correct_l742_742545

theorem expression_evaluation_correct (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ( ( ( (x - 2) ^ 2 * (x ^ 2 + x + 1) ^ 2 ) / (x ^ 3 - 1) ^ 2 ) ^ 2 *
    ( ( (x + 2) ^ 2 * (x ^ 2 - x + 1) ^ 2 ) / (x ^ 3 + 1) ^ 2 ) ^ 2 ) 
  = (x^2 - 4)^4 := 
sorry

end expression_evaluation_correct_l742_742545


namespace trigonometric_identity_l742_742198

   variable (A B C : Type) [triangle : triangle ABC]
   open triangle

   theorem trigonometric_identity 
     (a b c : ℝ) 
     (angle_A : ℝ)
     (h₁ : a = 1) 
     (h₂ : angle_A = π / 4) 
     (b_pos : 0 < b) 
     (c_pos : 0 < c) :
     ∀ (C : ℝ), \(\frac{\sqrt{2}b}{\sin C + \cos C}\) = sqrt(2) :=
   by
     sorry
   
end trigonometric_identity_l742_742198


namespace add_three_digits_l742_742504

theorem add_three_digits (x : ℕ) :
  (x = 152 ∨ x = 656) →
  (523000 + x) % 504 = 0 := 
by
  sorry

end add_three_digits_l742_742504


namespace cost_per_person_l742_742286

theorem cost_per_person 
  (total_cost : ℕ) 
  (total_people : ℕ) 
  (total_cost_in_billion : total_cost = 40000000000) 
  (total_people_in_million : total_people = 200000000) :
  total_cost / total_people = 200 := 
sorry

end cost_per_person_l742_742286


namespace determine_b_l742_742544

-- Define the problem conditions
variable (n b : ℝ)
variable (h_pos_b : b > 0)
variable (h_eq : ∀ x : ℝ, (x + n) ^ 2 + 16 = x^2 + b * x + 88)

-- State that we want to prove that b equals 12 * sqrt(2)
theorem determine_b : b = 12 * Real.sqrt 2 :=
by
  sorry

end determine_b_l742_742544


namespace tan_150_eq_l742_742010

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742010


namespace sum_of_ns_l742_742397

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742397


namespace b_100_eq_l742_742912

noncomputable def b : ℕ → ℚ
| 1 := 1 / 2
| 2 := 1
| (n+3) := (2 - b (n+2)) / (3 * b (n+1))

theorem b_100_eq : b 100 = 1 / 3 :=
by
  sorry

end b_100_eq_l742_742912


namespace tan_150_eq_neg_sqrt3_div_3_l742_742102

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742102


namespace least_positive_integer_x_l742_742132

theorem least_positive_integer_x :
  ∃ (x : ℕ), x + 7219 ≡ 5305 [MOD 17] ∧ x ≡ 4 [MOD 7] ∧ ∀ (y : ℕ), (y + 7219 ≡ 5305 [MOD 17]) ∧ (y ≡ 4 [MOD 7]) → y ≥ x :=
by
  let x : ℕ := 109
  existsi x
  split
  { calc x + 7219 ≡ 109 + 7219 [MOD 17] : by simp [x]
         ... ≡ 5305 [MOD 17] : by sorry -- Provide the calculation that 109 + 7219 ≡ 5305 [MOD 17] }
  split
  { calc x ≡ 109 [MOD 7] : by simp [x]
         ... ≡ 4 [MOD 7] : by sorry -- Provide the calculation that 109 ≡ 4 [MOD 7] }
  { intros y Hy
    simp at Hy
    cases Hy with H1 H2
    sorry -- Prove that there is no smaller positive integer that satisfies both conditions }

end least_positive_integer_x_l742_742132


namespace circle_area_of_equilateral_triangle_l742_742439

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742439


namespace tan_150_eq_neg_sqrt3_div_3_l742_742107

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742107


namespace circle_area_of_equilateral_triangle_l742_742442

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742442


namespace value_of_b_2_pow_100_l742_742243

-- Definitions based on conditions
def b : ℕ → ℕ
| 1       := 2
| (nat.succ (nat.succ n)) := 
  if (nat.succ n) % 2 = 0 
  then (nat.succ n) / 2 * b ((nat.succ n) / 2)
  else 0 -- This should never be the case due to our definition pattern

-- Theorem to be proved
theorem value_of_b_2_pow_100 : b (2^100) = 2^100 :=
sorry

end value_of_b_2_pow_100_l742_742243


namespace no_integer_triplets_for_equation_l742_742553

theorem no_integer_triplets_for_equation (a b c : ℤ) : ¬ (a^2 + b^2 + 1 = 4 * c) :=
by
  sorry

end no_integer_triplets_for_equation_l742_742553


namespace conic_section_eccentricities_cubic_l742_742164

theorem conic_section_eccentricities_cubic : 
  ∃ (e1 e2 e3 : ℝ), 
    (e1 = 1) ∧ 
    (0 < e2 ∧ e2 < 1) ∧ 
    (e3 > 1) ∧ 
    2 * e1^3 - 7 * e1^2 + 7 * e1 - 2 = 0 ∧
    2 * e2^3 - 7 * e2^2 + 7 * e2 - 2 = 0 ∧
    2 * e3^3 - 7 * e3^2 + 7 * e3 - 2 = 0 := 
by
  sorry

end conic_section_eccentricities_cubic_l742_742164


namespace trigonometric_identity_l742_742581

def sin_cos_relation (x : ℝ) : Prop := sin x = 2 * cos x

theorem trigonometric_identity (x : ℝ) (h : sin_cos_relation x) :
  (3 * sin (3 * π / 2 + x) - cos (π / 2 + x)) / (5 * cos (π + x) - sin (-x)) = 1 / 3 := by
  sorry

end trigonometric_identity_l742_742581


namespace interval_mono_increasing_range_of_omega_l742_742143

-- Define the given conditions
def omega_pos (ω : ℝ) : Prop := ω > 0
def vector_a (ω x : ℝ) : ℝ × ℝ := (real.sqrt 3, real.cos (ω * x + real.pi / 6) ^ 2 - 1)
def vector_b (ω x : ℝ) : ℝ × ℝ := (real.cos (2 * ω * x - real.pi / 6), 2)
def f (ω x : ℝ) : ℝ := (vector_a ω x).fst * (vector_b ω x).fst + (vector_a ω x).snd * (vector_b ω x).snd

-- Problem 1
theorem interval_mono_increasing (ω x : ℝ) (hω : omega_pos ω) (hω1 : ω = 1) : 
  ∃ k : ℤ, (-real.pi / 2 + k * real.pi ≤ x ∧ x ≤ k * real.pi) :=
sorry

-- Problem 2
theorem range_of_omega (ω : ℝ) (h_zeros : ∃ x ∈ (0, real.pi / 2), f ω x = 0 ∧ card (roots ℝ f ω (0, real.pi / 2)) = 3) :
  7 / 3 < ω ∧ ω ≤ 11 / 3 :=
sorry

end interval_mono_increasing_range_of_omega_l742_742143


namespace linear_function_properties_l742_742957

noncomputable def linearFunction (x : ℝ) : ℝ := 2 * x - 1

theorem linear_function_properties :
  (linearFunction 3 = 5) ∧ (linearFunction (-4) = -9) ∧
  (∃ b, linearFunction 0 = b ∧ b = -1) ∧
  (∃ a, linearFunction a = 0 ∧ a = 1 / 2) ∧
  let x_intercept : ℝ := 1 / 2,
      y_intercept : ℝ := -1 in
  (1 / 2 * abs x_intercept * abs y_intercept = 1 / 4) ∧
  let a := 3 / 2 in
  (linearFunction a = 2) :=
by
  split; sorry

end linear_function_properties_l742_742957


namespace find_y_payment_l742_742810

-- Definitions for the conditions in the problem
def total_payment (X Y : ℝ) : Prop := X + Y = 560
def x_is_120_percent_of_y (X Y : ℝ) : Prop := X = 1.2 * Y

-- Problem statement converted to a Lean proof problem
theorem find_y_payment (X Y : ℝ) (h1 : total_payment X Y) (h2 : x_is_120_percent_of_y X Y) : Y = 255 := 
by sorry

end find_y_payment_l742_742810


namespace tan_150_deg_l742_742024

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742024


namespace sum_of_cubes_inequality_l742_742685

theorem sum_of_cubes_inequality (a b c : ℝ) (h1 : a >= -1) (h2 : b >= -1) (h3 : c >= -1) (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 <= 4 := 
sorry

end sum_of_cubes_inequality_l742_742685


namespace construction_tasks_l742_742425

variable (S : Set Point) (O : Point) (l : Line) (A : Point)
variable (BC : Segment) (a b c : Length) (r : Length) (A B : Point)
variable (radius1 radius2 : Length)

theorem construction_tasks :
  (∃ (draw_parallel : ∀ (p : Point), ∃ (l' : Line), is_parallel l' l) ∧ 
   ∃ (drop_perpendicular : ∀ (p : Point), ∃ (l' : Line), is_perpendicular l' l)) ∧
  (∃ (measure_segment : ∀ (p : Point) (l : Line) (s : Segment), ∃ (q : Point), distance p q = segment_length s)) ∧
  (∃ (construct_segment : ∀ (a b c : Length), ∃ (s : Segment), segment_length s = (a * b) / c )) ∧
  (∃ (intersection_line_circle : ∀ (l : Line) (A : Point) (r : Length), ∃ (p1 p2 : Point), on_circle A r p1 ∧ on_circle A r p2 ∧ on_line l p1 ∧ on_line l p2)) ∧
  (∃ (intersection_two_circles : ∀ (A B : Point) (r1 r2 : Length), ∃ (p1 p2 : Point), on_circle A r1 p1 ∧ on_circle A r1 p2 ∧ on_circle B r2 p1 ∧ on_circle B r2 p2)) :=
by sorry

end construction_tasks_l742_742425


namespace sector_to_cone_height_l742_742833

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l742_742833


namespace number_of_integers_l742_742423

theorem number_of_integers (count : ℕ) :
  (count = fintype.card {n : ℕ // 2 ≤ n ∧ n ≤ 2016 ∧ n % 210 = 1}) :=
by {
  -- Solution steps skipped
  have h : ∀ n, 2 ≤ n → n ≤ 2016 → n % 210 = 1 ↔ ∃ k, n = 210 * k + 1 ∧ 1 ≤ k ∧ k ≤ 9,
  { sorry }, -- Proof of equivalence between conditions and k values.
  have : {n : ℕ // 2 ≤ n ∧ n ≤ 2016 ∧ n % 210 = 1}.card = 9,
  { sorry }, -- Proof that there are 9 such integers
  exact this
}

end number_of_integers_l742_742423


namespace product_of_integers_l742_742321

theorem product_of_integers :
  ∃ (a b c d e : ℤ), 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = {2, 6, 10, 10, 12, 14, 16, 18, 20, 24}) ∧
  (a * b * c * d * e = -3003) :=
begin
  sorry
end

end product_of_integers_l742_742321


namespace angle_between_line_and_plane_l742_742177

open Real Matrix

noncomputable theory

-- Define the points A, B, C, and D in 3D space.
def A : ℝ × ℝ × ℝ := (1, 0, 1)
def B : ℝ × ℝ × ℝ := (-2, 2, 1)
def C : ℝ × ℝ × ℝ := (2, 0, 3)
def D : ℝ × ℝ × ℝ := (0, 4, -2)

-- Compute the vectors AB, BC, and BD.
def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def BC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)
def BD : ℝ × ℝ × ℝ := (D.1 - B.1, D.2 - B.2, D.3 - B.3)

-- Define the cross product for 3D vectors.
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Compute the normal vector to the plane BCD.
def normal_vector_to_plane_BCD : ℝ × ℝ × ℝ := cross_product BC BD

-- Define the dot product for 3D vectors.
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Compute the magnitudes of vectors.
def magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Define the angle between line AB and normal to plane BCD.
def cos_phi : ℝ :=
  dot_product AB normal_vector_to_plane_BCD / (magnitude AB * magnitude normal_vector_to_plane_BCD)

-- Define the angle alpha between the line AB and the plane BCD.
def alpha : ℝ := Real.arcsin (Real.sqrt 13 / Real.sqrt 101)

-- Proof statement (without the actual proof).
theorem angle_between_line_and_plane :
  Real.arcsin cos_phi = alpha :=
by sorry

end angle_between_line_and_plane_l742_742177


namespace andy_and_dawn_total_time_l742_742872

noncomputable def and_dawn_total_time :
    ℕ := 20  -- time Dawn spent washing dishes

noncomputable def and_laundry_time (dawn_dish_time : ℕ) :
    ℕ := 2 * dawn_dish_time + 6  -- time Andy spent on laundry

noncomputable def and_vacuuming_time (laundry_time dish_time : ℕ) :
    ℝ := Real.sqrt ((laundry_time : ℝ) - (dish_time : ℝ))  -- time Andy spent vacuuming

noncomputable def dawn_window_time (combined_time : ℕ) :
    ℝ := (combined_time : ℝ) / 4  -- time Dawn spent wiping windows

theorem andy_and_dawn_total_time :
    let dawn_dish_time := 20 in
    let laundry_time := and_laundry_time dawn_dish_time in
    let combined_time := laundry_time + dawn_dish_time in
    let vacuuming_time := and_vacuuming_time laundry_time dawn_dish_time in
    let window_time := dawn_window_time combined_time in
    let total_time := (laundry_time : ℝ) + (dawn_dish_time : ℝ) + vacuuming_time + window_time in
    total_time = 87.6 := by {
  sorry
}

end andy_and_dawn_total_time_l742_742872


namespace length_NZ_l742_742219

variable (X Y Z M N : Type) [Inhabited X] [Inhabited Y] [Inhabited Z] [Inhabited M] [Inhabited N]
variables (XY_norm : ℝ) (ZM : ℝ) (MN : ℝ) (NZ : ℝ)

-- Triangle XYZ with M as midpoint of XY and N as centroid
def is_midpoint (M : X) (XY : X × X) : Prop := sorry -- use appropriate definition to state M is the midpoint of XY
def is_centroid (N : X) (triangle : X × X × X) : Prop := sorry -- use appropriate definition to state N is the centroid of triangle XYZ

def M_mid : is_midpoint M (X, Y) := sorry
def N_centroid : is_centroid N (X, Y, Z) := sorry

-- MN is 4 inches
axiom MN_length : MN = 4

-- Centroid property: NZ = 2 * MN
axiom centroid_property : NZ = 2 * MN

-- Statement to be proved in Lean
theorem length_NZ (M : X) (N : X) (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z]
  (MN : ℝ) (NZ : ℝ) (H1 : is_midpoint M (X, Y))
  (H2 : is_centroid N (X, Y, Z))
  (H3 : MN = 4)
  (H4 : NZ = 2 * MN) :
  NZ = 8 := by
  sorry

end length_NZ_l742_742219


namespace tan_A_right_triangle_l742_742648

theorem tan_A_right_triangle (A B C : Type) [triangle ABC] (AB AC : ℝ) 
  (h_right_angle : is_right_angle ∠ABC) (h_AB : AB = 20) (h_AC : AC = 29) :
  tan (angle_at A ABC) = 21 / 20 := 
sorry

end tan_A_right_triangle_l742_742648


namespace stellar_hospital_multiple_births_l742_742874

/-- At Stellar Hospital, in a particular year, the multiple-birth statistics were such that sets of twins, triplets, and quintuplets accounted for 1200 of the babies born. 
There were twice as many sets of triplets as sets of quintuplets, and there were twice as many sets of twins as sets of triplets.
Determine how many of these 1200 babies were in sets of quintuplets. -/
theorem stellar_hospital_multiple_births 
    (a b c : ℕ)
    (h1 : b = 2 * c)
    (h2 : a = 2 * b)
    (h3 : 2 * a + 3 * b + 5 * c = 1200) :
    5 * c = 316 :=
by sorry

end stellar_hospital_multiple_births_l742_742874


namespace four_points_triangle_five_points_triangle_six_points_triangle_l742_742806

-- (a)
theorem four_points_triangle (points : Set (ℝ × ℝ)) (h₁ : points.card = 4) :
  ∃ (A B C : ℝ × ℝ) (hA : A ∈ points) (hB : B ∈ points) (hC : C ∈ points),
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ angle A B C ≥ 90 :=
by
  sorry

-- (b)
theorem five_points_triangle (points : Set (ℝ × ℝ)) (h₁ : points.card = 5) :
  ∃ (A B C : ℝ × ℝ) (hA : A ∈ points) (hB : B ∈ points) (hC : C ∈ points),
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ angle A B C ≥ 108 :=
by
  sorry

-- (c)
theorem six_points_triangle (points : Set (ℝ × ℝ)) (h₁ : points.card = 6) :
  ∃ (A B C : ℝ × ℝ) (hA : A ∈ points) (hB : B ∈ points) (hC : C ∈ points),
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ angle A B C ≥ 120 :=
by
  sorry

end four_points_triangle_five_points_triangle_six_points_triangle_l742_742806


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742084

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742084


namespace number_of_packs_l742_742326

theorem number_of_packs (total_towels towels_per_pack : ℕ) (h1 : total_towels = 27) (h2 : towels_per_pack = 3) :
  total_towels / towels_per_pack = 9 :=
by
  sorry

end number_of_packs_l742_742326


namespace tan_150_eq_l742_742006

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742006


namespace xiao_ming_final_score_l742_742411

theorem xiao_ming_final_score :
  let content_score := 85
  let language_expression_score := 90
  let image_demeanor_score := 82
  let content_weight := 0.6
  let language_expression_weight := 0.3
  let image_demeanor_weight := 0.1
  (content_score * content_weight + language_expression_score * language_expression_weight + image_demeanor_score * image_demeanor_weight) = 86.2 :=
by
  simp only [content_score, language_expression_score, image_demeanor_score, content_weight, language_expression_weight, image_demeanor_weight]
  norm_num
  sorry

end xiao_ming_final_score_l742_742411


namespace complex_number_solution_l742_742288

theorem complex_number_solution (z : ℂ) (hz : z + complex.abs z = 2 + 8 * complex.I) : z = -15 + 8 * complex.I :=
sorry

end complex_number_solution_l742_742288


namespace work_completion_l742_742816

/-- 
  Let A, B, and C have work rates where:
  1. A completes the work in 4 days (work rate: 1/4 per day)
  2. C completes the work in 12 days (work rate: 1/12 per day)
  3. Together with B, they complete the work in 2 days (combined work rate: 1/2 per day)
  Prove that B alone can complete the work in 6 days.
--/
theorem work_completion (A B C : ℝ) (x : ℝ)
  (hA : A = 1/4)
  (hC : C = 1/12)
  (h_combined : A + 1/x + C = 1/2) :
  x = 6 := sorry

end work_completion_l742_742816


namespace four_integers_product_sum_l742_742590

theorem four_integers_product_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 2002) (h_sum : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l742_742590


namespace tan_150_deg_l742_742036

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742036


namespace total_bottles_in_market_l742_742782

theorem total_bottles_in_market (j w : ℕ) (hj : j = 34) (hw : w = 3 / 2 * j + 3) : j + w = 88 :=
by
  sorry

end total_bottles_in_market_l742_742782


namespace three_times_hash_l742_742908

noncomputable def hash (N : ℕ) : ℕ :=
  Float.toNat (Float.round (0.6 * N + 2))

theorem three_times_hash :
  hash (hash (hash 40)) = 13 :=
by
  sorry

end three_times_hash_l742_742908


namespace evaluate_f_at_9_l742_742567

noncomputable def f (x : ℝ) : ℝ := real.sqrt x + 2

theorem evaluate_f_at_9 : f 9 = 5 :=
by {
    -- skipping the proof
    sorry
}

end evaluate_f_at_9_l742_742567


namespace area_of_circumscribed_circle_l742_742481

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742481


namespace eq_decrease_in_area_l742_742508

noncomputable def decrease_in_area (A : ℝ) (decrease_length : ℝ) : ℝ :=
  let s := real.sqrt (4 * A / real.sqrt 3)
  let s' := s - decrease_length
  let A' := s'^2 * real.sqrt 3 / 4
  A - A'

theorem eq_decrease_in_area :
  decrease_in_area (121 * real.sqrt 3) 6 = 57 * real.sqrt 3 :=
by
  sorry

end eq_decrease_in_area_l742_742508


namespace tan_150_eq_neg_inv_sqrt3_l742_742071

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742071


namespace george_blocks_l742_742141

theorem george_blocks (n_boxes : ℕ) (blocks_per_box : ℕ) (H1 : n_boxes = 2) (H2 : blocks_per_box = 6) :
  n_boxes * blocks_per_box = 12 :=
by
  rw [H1, H2]
  exact rfl

end george_blocks_l742_742141


namespace triangle_x_value_l742_742255

theorem triangle_x_value (P Q R S T : Point) 
  (h_intersect: Line P Q ∩ Line R S = {T})
  (h_ts_tq: dist T S = dist T Q)
  (h_angle: ∠ P T R = 88)
  : x = 46 :=
by 
  sorry

end triangle_x_value_l742_742255


namespace discount_is_100_l742_742699

-- Define the constants for the problem conditions
def suit_cost : ℕ := 430
def shoes_cost : ℕ := 190
def amount_paid : ℕ := 520

-- Total cost before discount
def total_cost_before_discount (a b : ℕ) : ℕ := a + b

-- Discount amount
def discount_amount (total paid : ℕ) : ℕ := total - paid

-- Main theorem statement
theorem discount_is_100 : discount_amount (total_cost_before_discount suit_cost shoes_cost) amount_paid = 100 := 
by
sorry

end discount_is_100_l742_742699


namespace point_on_sphere_l742_742526

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem point_on_sphere :
  let ρ := 4
  let θ := Real.pi / 4
  let φ := Real.pi / 6
  let rect_coords := spherical_to_rectangular ρ θ φ
  rect_coords = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) ∧ 
  (rect_coords.1 ^ 2 + rect_coords.2 ^ 2 + rect_coords.3 ^ 2 = 4 ^ 2) :=
by
  let ρ := 4
  let θ := Real.pi / 4
  let φ := Real.pi / 6
  let rect_coords := spherical_to_rectangular ρ θ φ
  have h1 : spherical_to_rectangular ρ θ φ = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) := sorry
  have h2 : (Real.sqrt 2 ^ 2 + Real.sqrt 2 ^ 2 + (2 * Real.sqrt 3) ^ 2 = 4 ^ 2) := sorry
  exact ⟨h1, h2⟩

end point_on_sphere_l742_742526


namespace vector_parallel_l742_742564

theorem vector_parallel (x y : ℝ) (h : ∃ λ : ℝ, ∀ i : ℕ, ([3, x, y].nth i) = (λ • [2, 4, 5].nth i)) :
  x = 6 ∧ y = 15 / 2 :=
begin
sorry
end

end vector_parallel_l742_742564


namespace will_use_6_pages_l742_742426

-- Definitions
def total_cards := 18
def cards_per_page := 3

-- Lean statement
theorem will_use_6_pages : (total_cards / cards_per_page) = 6 :=
by
sorrry -- Placeholder for the actual proof.

end will_use_6_pages_l742_742426


namespace acute_triangle_angle_measure_acute_triangle_side_range_l742_742965

theorem acute_triangle_angle_measure (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) : B = π / 3 :=
by
  sorry

theorem acute_triangle_side_range (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) (hB : B = π / 3) (hb : b = 3) :
  3 * Real.sqrt 3 < a + c ∧ a + c ≤ 6 :=
by
  sorry

end acute_triangle_angle_measure_acute_triangle_side_range_l742_742965


namespace sum_of_valid_ns_l742_742346

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742346


namespace angle_A_measure_triangle_area_angle_A_l742_742220

section TriangleProofs

variables {A B C a b c : ℝ}

-- Part 1: Proving the measure of angle A
theorem angle_A_measure (h : sqrt 3 * b * sin (B + C) + a * cos B = c) (h_sum : A + B + C = π) : A = π / 6 :=
sorry

-- Part 2: Proving the area of triangle ABC
theorem triangle_area_angle_A (h1 : sqrt 3 * b * sin (B + C) + a * cos B = c) (A_val : A = π / 6) 
(a_eq : a = 6) (b_plus_c : b + c = 6 + 6 * sqrt 3) : (1 / 2) * b * c * sin A = 9 * sqrt 3 :=
sorry

end TriangleProofs

end angle_A_measure_triangle_area_angle_A_l742_742220


namespace tan_150_eq_neg_inv_sqrt3_l742_742059

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742059


namespace binom_sum_l742_742380

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742380


namespace region_area_l742_742129

-- Definitions of the conditions
def region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ 150 * (x - ⌊x⌋) ≥ 2 * ⌊x⌋ + ⌊y⌋

-- The theorem statement
theorem region_area : 
  (∫ x in 0..2, ∫ y in 0..150 * (x - ⌊x⌋) - 2 * ⌊x⌋, 1) = 2265.25 :=
sorry

end region_area_l742_742129


namespace find_max_sum_with_constraints_l742_742241

noncomputable def max_value (a b c d e : ℝ) : ℝ :=
  c * (a + 3 * b + 2 * d + 8 * e)

theorem find_max_sum_with_constraints :
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  a^2 + b^2 + c^2 + d^2 + e^2 = 804 ∧
  max_value a b c d e = 402 * real.sqrt 78 ∧
  a + b + c + d + e = 28 + 402 * real.sqrt 78 + 6 * real.sqrt 14 :=
begin
  use [2, 6, 6*real.sqrt 14, 4, 16],
  split, exact zero_lt_two,
  split, exact zero_lt_six,
  split, exact mul_pos zero_lt_six (real.sqrt_pos.2 $ @zero_lt_fourteen _),
  split, exact zero_lt_four,
  split, exact zero_lt_sixteen,
  split,
  { rw [sq, sq, sq, sq, sq],
    norm_num,
    rw [←sq_eq_sq (real.sqrt_nonneg 14) (show 0 ≤ 402, by norm_num), real.sqrt_sq_eq_abs],
    exact (zero_le_two.two_mul_two_plus_norm_num_804.402_refl_eq_norm_num_804).symm, },
  split,
  { unfold max_value,
    norm_num,
    have h : (6 * real.sqrt 14) ^ 2 = 14 * 36 := by norm_num [real.sq_mul, real.sq_sqrt _],
    rw [h, mul_comm, real.sq.,
        ←sqrt_eq_402 (show 6 ≥ (6 * c), by norm_num),
        real.sqrt_804,exact _7.symm }, sorry
  },
  refl,
end

end find_max_sum_with_constraints_l742_742241


namespace circle_area_of_equilateral_triangle_l742_742445

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742445


namespace symmetric_point_coordinates_l742_742165

theorem symmetric_point_coordinates 
  (k : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : ∀ k, k * (P.1) - P.2 + k - 2 = 0) 
  (P' : ℝ × ℝ) 
  (h2 : P'.1 + P'.2 = 3) 
  (h3 : 2 * P'.1^2 + 2 * P'.2^2 + 4 * P'.1 + 8 * P'.2 + 5 = 0) 
  (hP : P = (-1, -2)): 
  P' = (2, 1) := 
sorry

end symmetric_point_coordinates_l742_742165


namespace solve_inequality_l742_742110

theorem solve_inequality (x : ℝ) (h1 : 0 < x) (h2 : real.sqrt (x + 4) < 3 * x) :
  x > (1 + real.sqrt 145) / 18 := 
sorry

end solve_inequality_l742_742110


namespace find_x_l742_742909

def g (x : ℝ) : ℝ := 4 * x - 9

def g_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem find_x : ∃ x : ℝ, g x = g_inv x :=
by
  use 3
  sorry

end find_x_l742_742909


namespace roger_initial_candies_l742_742274

def initial_candies (given_candies left_candies : ℕ) : ℕ :=
  given_candies + left_candies

theorem roger_initial_candies :
  initial_candies 3 92 = 95 :=
by
  sorry

end roger_initial_candies_l742_742274


namespace store_profit_l742_742497

-- Definitions
def purchase_price_per_card : ℕ := 21
def total_revenue : ℕ := 1457
def max_selling_price_per_card : ℕ := 42

-- Question statement: The number of dimes the store earned
def profit_in_dimes (p : ℕ) (cards_sold : ℕ) : ℕ :=
  (p - purchase_price_per_card) * cards_sold

theorem store_profit :
  ∃ (p cards_sold : ℕ), 
    p ≤ max_selling_price_per_card ∧ 
    total_revenue = p * cards_sold ∧ 
    profit_in_dimes p cards_sold = 470 :=
begin
  sorry
end

end store_profit_l742_742497


namespace rectangle_area_ratio_l742_742751

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let area_square := s^2
  let longer_side := 1.15 * s
  let shorter_side := 0.95 * s
  let area_rectangle := longer_side * shorter_side
  area_rectangle / area_square

theorem rectangle_area_ratio (s : ℝ) : area_ratio s = 109.25 / 100 := by
  sorry

end rectangle_area_ratio_l742_742751


namespace dentist_cleaning_cost_l742_742794

theorem dentist_cleaning_cost
  (F: ℕ)
  (C: ℕ)
  (B: ℕ)
  (tooth_extraction_cost: ℕ)
  (HC1: F = 120)
  (HC2: B = 5 * F)
  (HC3: tooth_extraction_cost = 290)
  (HC4: B = C + 2 * F + tooth_extraction_cost) :
  C = 70 :=
by
  sorry

end dentist_cleaning_cost_l742_742794


namespace digits_of_x_l742_742931

noncomputable def num_digits_base_10 (x : ℝ) : ℕ := (real.log x / real.log 10).ceil.to_nat

theorem digits_of_x (x : ℝ) (h : real.log 3 (real.log 3 (real.log 3 x)) = 2) :
  num_digits_base_10 x = 9391 :=
sorry

end digits_of_x_l742_742931


namespace sports_club_intersection_l742_742418

variable (N B T Neither : ℕ)
variable (H1 : N = 30) (H2 : B = 17) (H3 : T = 17) (H4 : Neither = 2)

theorem sports_club_intersection : (B + T - (N - Neither)) = 6 := by
  rw [H1, H2, H3, H4]
  sorry

end sports_club_intersection_l742_742418


namespace solve_abs_quadratic_eq_and_properties_l742_742279

theorem solve_abs_quadratic_eq_and_properties :
  ∃ x1 x2 : ℝ, (|x1|^2 + 2 * |x1| - 8 = 0) ∧ (|x2|^2 + 2 * |x2| - 8 = 0) ∧
               (x1 = 2 ∨ x1 = -2) ∧ (x2 = 2 ∨ x2 = -2) ∧
               (x1 + x2 = 0) ∧ (x1 * x2 = -4) :=
by
  sorry

end solve_abs_quadratic_eq_and_properties_l742_742279


namespace cubes_closed_under_multiplication_l742_742180

def is_closed_under_multiplication (v : set ℕ) : Prop :=
  ∀ a b ∈ v, a * b ∈ v

def cubes_of_positive_integers : set ℕ := {n | ∃ k : ℕ, 0 < k ∧ n = k^3}

theorem cubes_closed_under_multiplication :
  is_closed_under_multiplication cubes_of_positive_integers :=
sorry

end cubes_closed_under_multiplication_l742_742180


namespace alex_catches_4000_fish_l742_742209

-- Definitions:

-- Brian catches 400 fish per trip
def brian_fish_per_trip : ℕ := 400

-- Chris goes fishing 10 times
def chris_trips : ℕ := 10

-- Alex goes fishing half as often as Chris
def alex_trips : ℕ := chris_trips / 2

-- Alex catches twice as many fish per trip as Brian
def alex_fish_per_trip : ℕ := 2 * brian_fish_per_trip

-- Total number of fish Alex caught
def alex_total_fish : ℕ := alex_trips * alex_fish_per_trip

-- Proof statement:
theorem alex_catches_4000_fish : alex_total_fish = 4000 := by
  simp [alex_total_fish, alex_trips, alex_fish_per_trip, chris_trips, brian_fish_per_trip]
  sorry

end alex_catches_4000_fish_l742_742209


namespace composite_values_infinite_l742_742236

open Nat

-- Definitions for the non-constant polynomials with integer positive coefficients
def is_non_constant_polynomial (f : ℕ → ℕ) : Prop :=
  (∃ x y : ℕ, x ≠ y ∧ f x ≠ f y) ∧ (∀ n : ℕ, 0 < f n)

-- Main statement
theorem composite_values_infinite (f g : ℕ → ℕ) (m n : ℕ) :
  is_non_constant_polynomial f →
  is_non_constant_polynomial g →
  ∃^∞ k, ∀ i, 0 ≤ i ∧ i ≤ k → ¬ Prime (f (m^n) + g i) := 
sorry

end composite_values_infinite_l742_742236


namespace cos_diff_l742_742975

variables {α β θ : ℝ}

-- Given conditions
def sin_alpha_am := sin α = (sin θ + cos θ) / 2
def sin_beta_gm := sin β = sqrt (sin θ * cos θ)

-- Theorem to be proven
theorem cos_diff : sin_alpha_am ∧ sin_beta_gm → (cos (4 * β) - 4 * cos (4 * α) = 3) :=
by
  sorry

end cos_diff_l742_742975


namespace fish_filets_total_l742_742880

/- Define the number of fish caught by each family member -/
def ben_fish : ℕ := 4
def judy_fish : ℕ := 1
def billy_fish : ℕ := 3
def jim_fish : ℕ := 2
def susie_fish : ℕ := 5

/- Define the number of fish thrown back -/
def fish_thrown_back : ℕ := 3

/- Define the number of filets per fish -/
def filets_per_fish : ℕ := 2

/- Calculate the number of fish filets -/
theorem fish_filets_total : ℕ :=
  let total_fish_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_fish_caught - fish_thrown_back
  fish_kept * filets_per_fish

example : fish_filets_total = 24 :=
by {
  /- This 'sorry' placeholder indicates that a proof should be here -/
  sorry
}

end fish_filets_total_l742_742880


namespace pencil_total_length_l742_742624

-- Definitions of the colored sections
def purple_length : ℝ := 3.5
def black_length : ℝ := 2.8
def blue_length : ℝ := 1.6
def green_length : ℝ := 0.9
def yellow_length : ℝ := 1.2

-- The theorem stating the total length of the pencil
theorem pencil_total_length : purple_length + black_length + blue_length + green_length + yellow_length = 10 := 
by
  sorry

end pencil_total_length_l742_742624


namespace gcd_lcm_product_l742_742516

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 75) :
  (Nat.gcd a b) * (Nat.lcm a b) = 2250 := by
  sorry

end gcd_lcm_product_l742_742516


namespace average_speed_x_to_z_l742_742420

theorem average_speed_x_to_z (D : ℝ) (hD_pos : D > 0) :
  let dist_xy := 2 * D,
      speed_xy := 100,
      speed_yz := 75,
      time_xy := dist_xy / speed_xy,
      time_yz := D / speed_yz,
      total_dist := dist_xy + D,
      total_time := time_xy + time_yz
  in total_dist / total_time = 90 :=
by {
    sorry
}

end average_speed_x_to_z_l742_742420


namespace monotonically_decreasing_interval_l742_742304

-- Define the function f(x) = 2x - ln(x)
def f (x : ℝ) : ℝ := 2 * x - Real.log x

-- Define the domain of the function, i.e., x > 0
def domain (x : ℝ) : Prop := x > 0

-- Define the derivative of the function f
def f_prime (x : ℝ) : ℝ := 2 - (1 / x)

-- State the theorem that f is monotonically decreasing on the interval (0, 1/2)
theorem monotonically_decreasing_interval :
  ∀ x : ℝ, domain x → x < 1 / 2 → f_prime x < 0 := 
by
  -- Proof goes here
  sorry

end monotonically_decreasing_interval_l742_742304


namespace train_speed_proof_l742_742499

theorem train_speed_proof
  (length_of_train : ℕ)
  (length_of_bridge : ℕ)
  (time_to_cross_bridge : ℕ)
  (h_train_length : length_of_train = 145)
  (h_bridge_length : length_of_bridge = 230)
  (h_time : time_to_cross_bridge = 30) :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 18 / 5 = 45 :=
by
  sorry

end train_speed_proof_l742_742499


namespace assign_roles_equiv_24_l742_742576

-- Define the problem conditions
variables (team : Finset String) (roles : Finset String)

-- Define the list of team members and roles
local notation "Alice" := "Alice"
local notation "Bob" := "Bob"
local notation "Carol" := "Carol"
local notation "Dave" := "Dave"
local notation "manager" := "manager"
local notation "assistant_manager" := "assistant_manager"
local notation "lead_developer" := "lead_developer"
local notation "quality_analyst" := "quality_analyst"

noncomputable def num_ways_to_assign_roles : ℕ :=
(team.perm roles).card

-- The main theorem to prove
theorem assign_roles_equiv_24 
  (h_team : team = {Alice, Bob, Carol, Dave})
  (h_roles : roles = {manager, assistant_manager, lead_developer, quality_analyst}) :
  num_ways_to_assign_roles team roles = 24 :=
by
  rw [num_ways_to_assign_roles, h_team, h_roles, Finset.card_perm],
  exact factorial_four sorry

end assign_roles_equiv_24_l742_742576


namespace value_of_f_ln_2_l742_742285

theorem value_of_f_ln_2
  (f : ℝ → ℝ)
  (monotonic_increasing : ∀ x y, x < y → f(x) < f(y))
  (f_gt_three_halves : ∀ x, f(x) > 3 / 2)
  (f_difference_positive : ∀ x1 x2, x1 < x2 → f(x2) - f(x1) > (3/2 - 1/2) - 1 > 0) :
  f(Real.log 2) = 3 :=
sorry

end value_of_f_ln_2_l742_742285


namespace part_one_part_two_l742_742813

theorem part_one :
  let lg2 := log 2
  let lg5 := log 5
  lg2^2 + lg2 * lg5 + lg5 = 1 :=
by
  let lg2 := log 2
  let lg5 := log 5
  show lg2^2 + lg2 * lg5 + lg5 = 1
  sorry

theorem part_two :
  let expr := (∛2 * √3)^6 - 8 * (16 / 49)^(-1 / 2) - √4 * 8^(1 / 4) - (-2016)^0
  expr = 91 :=
by
  let expr := (∛2 * √3)^6 - 8 * (16 / 49)^(-1 / 2) - √4 * 8^(1 / 4) - (-2016)^0
  show expr = 91
  sorry

end part_one_part_two_l742_742813


namespace notation_for_right_move_l742_742870

-- Definitions based on conditions from a)
def move_left := 2
def notation_left := +2
def move_right := 3

-- Theorem statement based on the question and correct answer from c)
theorem notation_for_right_move : 
  (move_left = 2 ∧ notation_left = 2) → notation_for (move_right) = -3 :=
by
  intro h
  sorry

end notation_for_right_move_l742_742870


namespace smallest_x_l742_742283

theorem smallest_x {
    x : ℤ
} : (x % 11 = 9) ∧ (x % 13 = 11) ∧ (x % 15 = 13) → x = 2143 := by
sorry

end smallest_x_l742_742283


namespace triangle_shape_l742_742640

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A) (hB : A < π) (h : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = π / 2 ∨ a = b) :=
by
  sorry

end triangle_shape_l742_742640


namespace condition_p_sufficient_for_condition_q_condition_q_not_necessary_for_condition_p_l742_742521

variables {x : ℝ}

def condition_p: Prop := abs x = x
def condition_q: Prop := x^2 >= -x

theorem condition_p_sufficient_for_condition_q (h : condition_p) : condition_q :=
sorry

theorem condition_q_not_necessary_for_condition_p (h : condition_q) : ¬condition_p :=
sorry

end condition_p_sufficient_for_condition_q_condition_q_not_necessary_for_condition_p_l742_742521


namespace abcd_sum_l742_742784

theorem abcd_sum : 
  ∃ (a b c d : ℕ), 
    (∃ x y : ℝ, x + y = 5 ∧ 2 * x * y = 6 ∧ 
      (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)) →
    a + b + c + d = 21 :=
by
  sorry

end abcd_sum_l742_742784


namespace tan_150_eq_neg_inv_sqrt3_l742_742067

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742067


namespace irreducible_polynomial_l742_742690

open Polynomial

-- Define the problem statement
theorem irreducible_polynomial 
  (n : ℕ) (p : ℤ) (a : ℕ → ℤ)
  (h_coeff : a n ≠ 0) 
  (h_prime : Prime p)
  (h_sum : (∑ j in finset.range n, |a(j)|) < p) :
  Irreducible (Polynomial.sum (fin n.succ) (λ j, monomial j (a j)) + C p) := sorry

end irreducible_polynomial_l742_742690


namespace binom_sum_l742_742382

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742382


namespace chen_recording_l742_742436

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end chen_recording_l742_742436


namespace circle_area_of_equilateral_triangle_l742_742440

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742440


namespace tangent_line_perpendicular_to_given_line_l742_742196

theorem tangent_line_perpendicular_to_given_line :
  ∃ (l : ℝ → ℝ) (x₀ : ℝ),
    l = (λ x, 4 * x - 4)
    ∧ (∃ y₀, (4x₀ - y₀ - 4 = 0) ∧ (λ x, x^2)' x₀ = 4) :=
sorry

end tangent_line_perpendicular_to_given_line_l742_742196


namespace range_of_f_l742_742530

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f : set.range f = {-(3 * Real.pi) / 4, Real.pi / 4} := 
by
  sorry

end range_of_f_l742_742530


namespace cars_count_l742_742786

theorem cars_count
  (distance : ℕ)
  (time_between_cars : ℕ)
  (total_time_hours : ℕ)
  (cars_per_hour : ℕ)
  (expected_cars_count : ℕ) :
  distance = 3 →
  time_between_cars = 20 →
  total_time_hours = 10 →
  cars_per_hour = 3 →
  expected_cars_count = total_time_hours * cars_per_hour →
  expected_cars_count = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4] at h5
  exact h5


end cars_count_l742_742786


namespace youtube_dislikes_calculation_l742_742431

theorem youtube_dislikes_calculation :
  ∀ (l d_initial d_final : ℕ),
    l = 3000 →
    d_initial = (l / 2) + 100 →
    d_final = d_initial + 1000 →
    d_final = 2600 :=
by
  intros l d_initial d_final h_l h_d_initial h_d_final
  sorry

end youtube_dislikes_calculation_l742_742431


namespace f_f_of_neg_one_l742_742300

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then
    Real.exp (x + 3)
  else
    Real.log x

theorem f_f_of_neg_one : f (f (-1)) = 2 := 
by
  sorry

end f_f_of_neg_one_l742_742300


namespace fraction_to_decimal_l742_742116

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 := by
  sorry

end fraction_to_decimal_l742_742116


namespace product_multiple_of_four_probability_l742_742123

theorem product_multiple_of_four_probability :
  let chips := {1, 2, 3, 4}
  let outcomes : Finset (ℕ × ℕ) := Finset.product chips chips
  let favorable_outcomes := outcomes.filter (λ (p : ℕ × ℕ), (p.1 * p.2) % 4 = 0)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 := by
  sorry

end product_multiple_of_four_probability_l742_742123


namespace sum_binom_solutions_l742_742389

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l742_742389


namespace new_ratio_boarders_to_day_students_l742_742312

-- Given conditions
def initial_ratio_boarders_to_day_students : ℚ := 2 / 5
def initial_boarders : ℕ := 120
def new_boarders : ℕ := 30

-- Derived definitions
def initial_day_students : ℕ :=
  (initial_boarders * (5 : ℕ)) / 2

def total_boarders : ℕ := initial_boarders + new_boarders
def total_day_students : ℕ := initial_day_students

-- Theorem to prove the new ratio
theorem new_ratio_boarders_to_day_students : total_boarders / total_day_students = 1 / 2 :=
  sorry

end new_ratio_boarders_to_day_students_l742_742312


namespace no_maximum_value_of_k_l742_742606

theorem no_maximum_value_of_k (x y : ℝ) (h : x^2 + y^2 - 2*x + 2*y - 1 = 0) : 
  ∀ k : ℝ, ∃ ε > 0, abs(k - (y - 3) / x) > ε := 
sorry

end no_maximum_value_of_k_l742_742606


namespace abs_reciprocal_inequality_l742_742566

theorem abs_reciprocal_inequality (a b : ℝ) (h : 1 / |a| < 1 / |b|) : |a| > |b| :=
sorry

end abs_reciprocal_inequality_l742_742566


namespace tan_150_eq_neg_inv_sqrt_3_l742_742095

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742095


namespace cone_height_l742_742829

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l742_742829


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742079

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742079


namespace find_k_l742_742693

theorem find_k (k : ℝ) (p : ℝ) (hp : 0 < p) (h_vertex : ∀ (x y : ℝ), (x, y) = (0, 0))
    (h_focus : ∃ y, (0, y) = (0, p))
    (h_point_on_parabola : ∀ (x y : ℝ), y = -2 → x^2 = 4 * p)
    (h_distance : ∀ (x y : ℝ), x = k ∧ y = -2 → sqrt ((x - 0)^2 + (y - p)^2) = 4) :
    k = 4 ∨ k = -4 := 
begin
  sorry
end

end find_k_l742_742693


namespace area_of_circumscribed_circle_l742_742485

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742485


namespace fraction_to_decimal_l742_742115

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 := by
  sorry

end fraction_to_decimal_l742_742115


namespace sum_of_valid_n_l742_742335

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742335


namespace arithmetic_geometric_seq_l742_742156

theorem arithmetic_geometric_seq (a : ℕ → ℝ) (d a_1 : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0) (h_geom : (a 0, a 1, a 4) = (a_1, a_1 + d, a_1 + 4 * d) ∧ (a 1)^2 = a 0 * a 4)
  (h_sum : a 0 + a 1 + a 4 > 13) : a_1 > 1 :=
by sorry

end arithmetic_geometric_seq_l742_742156


namespace area_triangle_ABC_l742_742787

-- Define the centers of the circles with their respective coordinates.
def A : ℝ × ℝ := (-5, 2)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (7, 4)

-- Prove the area of triangle ABC is 6 given the coordinates of A, B, and C.
theorem area_triangle_ABC : 
  let area := (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area = 6 :=
by {
  sorry
}

end area_triangle_ABC_l742_742787


namespace unique_last_digit_divisible_by_7_l742_742258

theorem unique_last_digit_divisible_by_7 :
  ∃! d : ℕ, (∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d) :=
sorry

end unique_last_digit_divisible_by_7_l742_742258


namespace sum_of_n_values_l742_742367

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742367


namespace sum_of_valid_n_l742_742370

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742370


namespace triangle_identity_l742_742224

variable {α : Type} [LinearOrderedField α] 

variables (A B C a b c : α)
variable (R : α) -- Circumradius 2R

-- Assuming Law of Sines holds in triangle ABC 
axiom law_of_sines : a = 2 * R * (sin A) ∧ b = 2 * R * (sin B) ∧ c = 2 * R * (sin C)

-- Prove the given trigonometric identity:
theorem triangle_identity
  (h₁ : a = 2 * R * (sin A))
  (h₂ : b = 2 * R * (sin B))
  (h₃ : c = 2 * R * (sin C)) : 
  (a^2 - b^2) / c^2 = (sin (A - B)) / (sin C) := by sorry

end triangle_identity_l742_742224


namespace area_of_circumscribed_circle_l742_742473

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742473


namespace parallel_lines_condition_l742_742181

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 6 = 0 → (a - 2) * x + 3 * y + 2 * a = 0 → False) ↔ a = -1 :=
sorry

end parallel_lines_condition_l742_742181


namespace ratio_proof_l742_742487

variable (R r a b : ℝ)

-- Conditions extracted from the problem
def area_larger : ℝ := π * R^2
def area_smaller : ℝ := π * r^2
def area_region : ℝ := π * R^2 - π * r^2
def condition : Prop := π * R^2 = ((a + b) / b) * (π * R^2 - π * r^2)

-- Statement we are going to prove
theorem ratio_proof : condition R r a b → R / r = Real.sqrt ((a + b) / a) := 
  sorry

end ratio_proof_l742_742487


namespace quadratic_root_difference_l742_742725

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def root_difference (a b c : ℝ) : ℝ :=
  (Real.sqrt (discriminant a b c)) / a

theorem quadratic_root_difference :
  root_difference (3 + 2 * Real.sqrt 2) (5 + Real.sqrt 2) (-4) = Real.sqrt (177 - 122 * Real.sqrt 2) :=
by
  sorry

end quadratic_root_difference_l742_742725


namespace problem_1_problem_2_problem_3_l742_742609

def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def notU (s : Set ℝ) : Set ℝ := { x | x ∉ s ∧ x ∈ U }

theorem problem_1 : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
sorry

theorem problem_2 : notU A ∪ B = { x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7) } :=
sorry

theorem problem_3 : A ∩ notU B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end problem_1_problem_2_problem_3_l742_742609


namespace debby_deleted_pictures_l742_742804

theorem debby_deleted_pictures :
  ∀ (zoo_pics museum_pics remaining_pics : ℕ), 
  zoo_pics = 24 →
  museum_pics = 12 →
  remaining_pics = 22 →
  (zoo_pics + museum_pics) - remaining_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics hz hm hr
  sorry

end debby_deleted_pictures_l742_742804


namespace distance_from_origin_to_line_l742_742841

noncomputable def parabola_f (y : ℝ) : ℝ := y^2 / 8

def line_l (x : ℝ) : ℝ := 2 * x - 4

def distance (x₀ y₀ a b c : ℝ) : ℝ := abs (a * x₀ + b * y₀ + c) / sqrt (a^2 + b^2)

theorem distance_from_origin_to_line :
  let l : ℝ → ℝ := line_l
  let C : ℝ → ℝ := parabola_f
  let F : (ℝ × ℝ) := (2, 0)
  let A B : (ℝ × ℝ) := sorry -- Points of intersection, details skipped as per problem bounds
  -- Using the given distance between points A and B
  l F.1 = F.2 ∧ sqrt ((10 / 2)^2 + 0^2) = 10 →
  distance 0 0 2 (-1) (-4) = 4 * sqrt 5 / 5 :=
by sorry

end distance_from_origin_to_line_l742_742841


namespace non_overlapping_length_at_least_half_l742_742715

noncomputable
def intervals_cover (intervals : list (ℝ × ℝ)) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ i, intervals.nth i ≠ none ∧ intervals.nth_le i (by sorry).fst ≤ x ∧ x ≤ intervals.nth_le i (by sorry).snd

theorem non_overlapping_length_at_least_half (intervals : list (ℝ × ℝ)) :
  intervals_cover intervals → ∃ (sub_intervals : list (ℝ × ℝ)),
  set.pairwise_sub (≠) (λ i, intervals.nth i) →
  sub_intervals.sum (λ i, i.snd - i.fst) ≥ 1 / 2 :=
sorry

end non_overlapping_length_at_least_half_l742_742715


namespace area_of_circumscribed_circle_l742_742454

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742454


namespace triangle_identity_l742_742225

variable {α : Type} [LinearOrderedField α] 

variables (A B C a b c : α)
variable (R : α) -- Circumradius 2R

-- Assuming Law of Sines holds in triangle ABC 
axiom law_of_sines : a = 2 * R * (sin A) ∧ b = 2 * R * (sin B) ∧ c = 2 * R * (sin C)

-- Prove the given trigonometric identity:
theorem triangle_identity
  (h₁ : a = 2 * R * (sin A))
  (h₂ : b = 2 * R * (sin B))
  (h₃ : c = 2 * R * (sin C)) : 
  (a^2 - b^2) / c^2 = (sin (A - B)) / (sin C) := by sorry

end triangle_identity_l742_742225


namespace inverse_proportion_function_increasing_l742_742704

theorem inverse_proportion_function_increasing (m : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (y = (m - 5) / x1) < (y = (m - 5) / x2)) ↔ m < 5 :=
by
  sorry

end inverse_proportion_function_increasing_l742_742704


namespace division_probability_l742_742750

theorem division_probability : 
  let possible_r := {-2, -1, 0, 1, 2, 3, 4, 5}
  let possible_k := {2, 3, 4, 5, 6, 7}
  let total_pairs := 48
  let valid_pairs := 12
  valid_pairs.toRat / total_pairs.toRat = 1/4 :=
begin
  sorry
end

end division_probability_l742_742750


namespace sequence_an_general_formula_l742_742691

theorem sequence_an_general_formula :
  ∀ (a : ℕ → ℝ) (f : ℕ → ℝ → ℝ),
    (∀ n, a n < a (n + 1)) →
    a 1 = 0 →
    (∀ n x, x ∈ set.Icc (a n) (a (n + 1)) → f n x = abs (Real.sin ((x - a n) / n))) →
    (∀ n b, b ∈ set.Icc 0 1 → ∃ x y, x ≠ y ∧ f n x = b ∧ f n y = b) →
    ∀ n, a n = (n * (n - 1) / 2 : ℕ) * Real.pi :=
by
  intros a f inc_seq a1_zero fn_def roots_exist n
  sorry

end sequence_an_general_formula_l742_742691


namespace find_f_comp_f_l742_742593

def f (x : ℚ) : ℚ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem find_f_comp_f (h : f (f (5/2)) = 3/2) :
  f (f (5/2)) = 3/2 := by
  sorry

end find_f_comp_f_l742_742593


namespace tan_150_deg_l742_742035

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742035


namespace g_of_neg_x_l742_742245

-- Define the function g(x)
def g (x : ℝ) : ℝ := (x^2 + 1) / (x^2 - 1)

-- State the theorem with the given conditions and the assertion to prove
theorem g_of_neg_x (x : ℝ) (h : x^2 ≠ 1) : g (-x) = g x :=
by sorry

end g_of_neg_x_l742_742245


namespace sec_135_eq_neg_sqrt_2_l742_742923

theorem sec_135_eq_neg_sqrt_2 : Real.sec (135 * Real.pi / 180) = -Real.sqrt 2 := 
by 
  sorry

end sec_135_eq_neg_sqrt_2_l742_742923


namespace system_of_equations_solution_l742_742768

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x + 3 * y = 7) 
  (h2 : y = 2 * x) : 
  x = 1 ∧ y = 2 :=
by
  sorry

end system_of_equations_solution_l742_742768


namespace find_angle_C1B1A1_l742_742216

open EuclideanGeometry

noncomputable theory

-- Define the basic setup of the problem
variables (A B C A1 B1 C1 : Point)
variables (hB : ∠ B = 120°)
variables (hAA1 : is_angle_bisector A A1)
variables (hBB1 : is_angle_bisector B B1)
variables (hCC1 : is_angle_bisector C C1)

-- Prove that the angle C1B1A1 is 90 degrees
theorem find_angle_C1B1A1 :
  ∠ C1 B1 A1 = 90° :=
by
  sorry

end find_angle_C1B1A1_l742_742216


namespace tan_150_deg_l742_742030

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742030


namespace number_of_equilateral_triangles_is_28_l742_742112

-- Define the structure of the lattice and equilateral triangles.
structure HexagonalLattice :=
  (points : Finset (ℤ × ℤ))
  (unit_distance : ℤ)
  (neighbors_one_unit_away : ∀ p ∈ points, ∃ q ∈ points, (p.1 - q.1)^2 + (p.2 - q.2)^2 = unit_distance^2)

-- Example of a hexagonal lattice where each point is one unit from nearest neighbor
def hexLattice : HexagonalLattice :=
  { points := {(0,0), (1,1), (1,2), (2,3), (3,1),(3,2),(4,0),(4,4),(5,5), (5,6), (6,4), (6,6)},
    unit_distance := 1,
    neighbors_one_unit_away := λ p hp, sorry }

-- Define what it means to be an equilateral triangle in this context
def is_equilateral (a b c : ℤ × ℤ) : Prop :=
  let d₁ := (a.1 - b.1)^2 + (a.2 - b.2)^2 in
  let d₂ := (b.1 - c.1)^2 + (b.2 - c.2)^2 in
  let d₃ := (c.1 - a.1)^2 + (c.2 - a.2)^2 in
  d₁ = d₂ ∧ d₂ = d₃ ∧ d₁ = d₃

-- Function to count all unique equilateral triangles in the lattice
noncomputable def count_equilateral_triangles (L : HexagonalLattice) : ℕ :=
  (L.points.to_finset.powerset.filter (λ s, s.card = 3 ∧
    let [a, b, c] := s.to_list in is_equilateral a b c)).card

-- Theorem statement: The number of equilateral triangles is 28 for the given hexagonal lattice
theorem number_of_equilateral_triangles_is_28 : count_equilateral_triangles hexLattice = 28 :=
by sorry

end number_of_equilateral_triangles_is_28_l742_742112


namespace AB_dot_DC_l742_742971

variable (O : Type)
variable {A B C D : O}
variable [inner_product_space ℝ O]

-- Declaring the conditions:
variable (diameter_AB : ∃ (E : O), E = A ∨ E = B)
variable (C_on_upper : ∃ (θ : ℝ), 0 < θ ∧ θ < π ∧ C = (rotate θ A B))
variable (D_on_lower : ∃ (θ : ℝ), -π < θ ∧ θ < 0 ∧ D = (rotate θ A B))
variable (not_equal_A : C ≠ A ∧ D ≠ A )
variable (not_equal_B : C ≠ B ∧ D ≠ B )
variable (AC_eq_2 : dist A C = 2)
variable (AD_eq_1 : dist A D = 1)

-- The goal:
theorem AB_dot_DC : inner_product_space.inner_product (A - B) (D - C) = 3 := 
sorry

end AB_dot_DC_l742_742971


namespace sum_of_integer_values_l742_742352

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742352


namespace area_of_circumscribed_circle_l742_742478

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742478


namespace sum_of_ns_l742_742402

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742402


namespace solve_exponent_problem_l742_742186

theorem solve_exponent_problem
  (h : (1 / 8) * (2 ^ 36) = 8 ^ x) : x = 11 :=
by
  sorry

end solve_exponent_problem_l742_742186


namespace standard_hyperbola_equation_l742_742979

noncomputable def focal_length := 10
noncomputable def slope_asymptote := 2

-- Constants for the hyperbola
noncomputable def a_squared := 5
noncomputable def b_squared := 20

theorem standard_hyperbola_equation : 
  (2 * sqrt (b_squared + a_squared) = focal_length) → 
  ((b_squared / a_squared).sqrt = slope_asymptote) → 
  ∀ x y : ℝ, (x^2 / a_squared - y^2 / b_squared = 1) = 
  ((x^2 / 5) - (y^2 / 20) = 1) := 
by
  sorry

end standard_hyperbola_equation_l742_742979


namespace largest_C_l742_742235

theorem largest_C (n : ℕ) (x : Fin n → ℝ) (h : 0 = x 0 ∧ ∀ k : Fin n, x k < x (k + 1) ∧ x n = 1) :
    ∑ k in Finset.range n, (x k) ^ 2 * (x k - x (k - 1)) > (1 / 3) :=
sorry

end largest_C_l742_742235


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742087

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742087


namespace number_of_handshakes_l742_742511

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end number_of_handshakes_l742_742511


namespace unit_vectors_sum_magnitude_ge_one_l742_742259

noncomputable theory

open Complex

variable {n : ℕ} (OP : Fin (2 * n + 1) → ℂ)
variable (O : ℂ)

-- |OP_i| = 1 for all i
variable (unit_OP : ∀ i, Complex.abs (OP i) = 1)
-- P_i all lie on the same side of a line through O
-- This condition is hard to formalize geometrically without further context.
-- We assume the existence of a function that formalizes this condition or placeholder.

noncomputable def sum_OP : ℂ := ∑ i in Finset.univ, OP i

theorem unit_vectors_sum_magnitude_ge_one 
  (same_side : ∀ i, Complex.im (O * conj (OP i)) > 0) :
  Complex.abs (sum_OP OP) ≥ 1 :=
sorry

end unit_vectors_sum_magnitude_ge_one_l742_742259


namespace a1_value_l742_742619

theorem a1_value (a_5 a_4 a_3 a_2 a_1 a : ℝ) (x : ℝ):
  (x + 1)^5 = a_5 * (x - 1)^5 + a_4 * (x - 1)^4 + a_3 * (x - 1)^3 + a_2 * (x - 1)^2 + a_1 * (x - 1) + a
  ∧ (x + 1)^5 = ∑ i from 0 to 5, (Nat.choose 5 i) * (x - 1)^(5 - i) * 2^i 
  → a_1 = 80 :=
by
  sorry

end a1_value_l742_742619


namespace imaginary_part_of_complex_div_l742_742749

open Complex

theorem imaginary_part_of_complex_div : Im ((3 : ℂ) + 4 * I) / I = 3 := by
  sorry

end imaginary_part_of_complex_div_l742_742749


namespace collinear_points_l742_742999

variable (a : ℝ) (A B C : ℝ × ℝ)

-- Conditions given in the problem
def point_A := (a, 2 : ℝ)
def point_B := (5, 1 : ℝ)
def point_C := (-4, 2 * a : ℝ)

-- Collinearity condition
def collinear (x y z : ℝ): Prop :=
  (x.1 - y.1) * (y.2 - z.2) = (y.1 - z.1) * (x.2 - y.2)

theorem collinear_points :
  collinear (point_A a) (point_B) (point_C a) →
  a = 5 + sqrt 21 ∨ a = 5 - sqrt 21 :=
by {
  sorry,
}

end collinear_points_l742_742999


namespace solve_for_k_l742_742120

theorem solve_for_k :
  (∀ x : ℝ, x * (2 * x + 3) < k ↔ x ∈ set.Ioo (-(5 / 2)) 1) → k = 5 :=
begin
  sorry
end

end solve_for_k_l742_742120


namespace calc_is_a_pow4_l742_742407

theorem calc_is_a_pow4 (a : ℕ) : (a^2)^2 = a^4 := 
by 
  sorry

end calc_is_a_pow4_l742_742407


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742080

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742080


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742078

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742078


namespace tan_150_eq_neg_inv_sqrt_3_l742_742088

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742088


namespace carly_dogs_total_l742_742898

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l742_742898


namespace area_of_circumscribed_circle_l742_742475

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742475


namespace area_of_triangle_l742_742655

theorem area_of_triangle (AB AC : ℝ) (angle_A : ℝ) (h1 : AB = 25) (h2 : AC = 20) (h3 : angle_A = 90) : 
  (1/2) * AB * AC = 250 :=
by
  rw [h1, h2]
  norm_num
  sorry

end area_of_triangle_l742_742655


namespace total_area_l742_742901

-- Define vertices and lengths.
variables (P Q R S T U : Type)
-- Conditions for the sides and constructions inside the hexagon.
variable (side_len : ℝ)
variable (int_side_len : ℝ)
variables (parPQ_RS parQR_ST parRS_UP : Prop)
variable (angle_60 : Prop)
variable (external_hex_area : ℝ)
variable (internal_tris_area : ℝ)

-- Given conditions
axiom sides_equal (a : side_len = 2)
axiom internal_sides_equal (b : int_side_len = 1)
axiom PQ_parallel_RS (c : parPQ_RS)
axiom QR_parallel_ST (d : parQR_ST)
axiom RS_parallel_UP (e : parRS_UP)
axiom internal_angles_60 (f : angle_60)
axiom hex_area (g : external_hex_area = 6 * Real.sqrt 3)
axiom internal_area (h : internal_tris_area = 6 * (Real.sqrt 3 / 4))

-- Theorem to prove
theorem total_area (answer : ℝ) : 
  answer = (external_hex_area + internal_tris_area)
  := by
    simp
    rw [g, h]
    simp [Real.sqrt, mul_comm, mul_assoc, add_comm]
    norm_num
    assume (T), sorry

end total_area_l742_742901


namespace nonnegative_row_column_sums_l742_742974

theorem nonnegative_row_column_sums (m n : ℕ) (M : matrix (fin m) (fin n) ℝ) :
  ∃ M' : matrix (fin m) (fin n) ℝ,
  (∀ i, 0 ≤ ∑ j, M' i j) ∧ (∀ j, 0 ≤ ∑ i, M' i j) :=
begin
  sorry
end

end nonnegative_row_column_sums_l742_742974


namespace bobby_final_paycheck_l742_742884

section BobbyPaycheck

variables
  (salary_per_week : ℝ) (performance_rate : ℝ)
  (federal_tax_rate : ℝ) (state_tax_rate : ℝ) (local_tax_rate : ℝ)
  (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ)
  (retirement_contribution_rate : ℝ) (employer_match_rate : ℝ)

noncomputable def final_paycheck 
  (salary_per_week := 450 : ℝ)
  (performance_rate := 0.12 : ℝ)
  (federal_tax_rate := 1/3 : ℝ)
  (state_tax_rate := 0.08 : ℝ)
  (local_tax_rate := 0.05 : ℝ)
  (health_insurance := 50 : ℝ)
  (life_insurance := 20 : ℝ)
  (parking_fee := 10 : ℝ)
  (retirement_contribution_rate := 0.03 : ℝ)
  (employer_match_rate := 0.5 : ℝ) : ℝ :=
  let bonus := salary_per_week * performance_rate in
  let total_income := salary_per_week + bonus in
  let federal_taxes := total_income * federal_tax_rate in
  let state_taxes := total_income * state_tax_rate in
  let local_taxes := total_income * local_tax_rate in
  let total_taxes := federal_taxes + state_taxes + local_taxes in
  let other_deductions := health_insurance + life_insurance + parking_fee in
  let retirement_contribution := salary_per_week * retirement_contribution_rate in
  total_income - total_taxes - other_deductions - retirement_contribution

theorem bobby_final_paycheck : final_paycheck = 176.98 := sorry

end BobbyPaycheck

end bobby_final_paycheck_l742_742884


namespace right_triangles_count_l742_742185

theorem right_triangles_count (b a : ℕ) (h₁: b < 150) (h₂: (a^2 + b^2 = (b + 2)^2)) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 12 ∧ b = n^2 - 1 :=
by
  -- This intended to state the desired number and form of the right triangles.
  sorry

def count_right_triangles : ℕ :=
  12 -- Result as a constant based on proof steps

#eval count_right_triangles -- Should output 12

end right_triangles_count_l742_742185


namespace radical_product_simplified_l742_742885

noncomputable def simplify_radical_product (p : ℝ) (hp : 0 ≤ p) : ℝ :=
  sqrt (15 * p) * sqrt (10 * p^3) * sqrt (14 * p^5)

theorem radical_product_simplified (p : ℝ) (hp : 0 ≤ p) :
  simplify_radical_product p hp = 10 * p^4 * sqrt (21 * p) :=
by sorry

end radical_product_simplified_l742_742885


namespace height_of_cone_formed_by_rolling_sector_l742_742824

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l742_742824


namespace part_a_part_b_part_c_l742_742111

variable (a b c : ℝ)
def f (x : ℝ) := a * x ^ 2 + b * x + c

axiom condition1 : ∀ x : ℝ, f(x) ≥ 0 ∧ f(-1) = 0 ∧ (∀ y : ℝ, f(y) = f(2*(-1) - y))
axiom condition2 : ∀ x : ℝ, 0 < x ∧ x < 5 → x ≤ f(x) ∧ f(x) ≤ 2 * |x - 1| + 1

-- Part a
theorem part_a : f 1 = 1 := sorry

-- Part b
theorem part_b : f = λ x, 1/4 * (x + 1) ^ 2 := sorry

-- Part c
theorem part_c : ∀ x m : ℝ, 0 ≤ m ∧ m ≤ 3 → |f(x) - x| ≤ 1 := sorry

end part_a_part_b_part_c_l742_742111


namespace problem1_problem2_l742_742724

open Real

theorem problem1 : sin (420 * π / 180) * cos (330 * π / 180) + sin (-690 * π / 180) * cos (-660 * π / 180) = 1 := by
  sorry

theorem problem2 (α : ℝ) : 
  (sin (π / 2 + α) * cos (π / 2 - α) / cos (π + α)) + 
  (sin (π - α) * cos (π / 2 + α) / sin (π + α)) = 0 := by
  sorry

end problem1_problem2_l742_742724


namespace distance_circumcenter_incenter_l742_742682

-- Assume a triangle is isosceles with circumradius r and inradius ρ.
variable (r ρ d : ℝ)
variable (isosceles_triangle : Prop) 

-- Distance between circumcenter and incenter
variable (circumcenter_incenter_distance : ℝ)

-- Assuming Euler's formula holds
axiom euler_formula : isosceles_triangle → circumcenter_incenter_distance^2 = r^2 - 2 * r * ρ

-- Statement to be proven
theorem distance_circumcenter_incenter : 
  isosceles_triangle → 
  circumcenter_incenter_distance = √ (r * (r - 2 * ρ)) :=
by
  sorry

end distance_circumcenter_incenter_l742_742682


namespace area_of_circumscribed_circle_l742_742482

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l742_742482


namespace area_of_BEIH_is_correct_l742_742522

noncomputable def vector (x y : ℚ) := (x, y)
noncomputable def vertices := 
  let A : (ℚ, ℚ) := vector 0 4
  let B : (ℚ, ℚ) := vector 0 0
  let C : (ℚ, ℚ) := vector 4 0
  let D : (ℚ, ℚ) := vector 4 4
  let E : (ℚ, ℚ) := vector 0 2
  let F : (ℚ, ℚ) := vector 3 0
  let I : (ℚ, ℚ) := vector 6/7 20/7
  let H : (ℚ, ℚ) := vector 3/7 3/7
  (A, B, C, D, E, F, I, H)

noncomputable def calculate_area (A B C D E F I H : (ℚ, ℚ)) : ℚ := sorry

theorem area_of_BEIH_is_correct : 
  let (A, B, C, D, E, F, I, H) := vertices
  calculate_area A B C D E F I H = 3/49 := 
by
  intro
  simp [vertices]
  sorry

end area_of_BEIH_is_correct_l742_742522


namespace CarlyWorkedOnElevenDogs_l742_742892

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l742_742892


namespace part_I_part_II_l742_742599

noncomputable def f (x a : ℝ) : ℝ := Real.log x + 0.5 * x * x - a * x

theorem part_I (x : ℝ) (h : x > 0) (a_eq_3 : a = 3) : 
  f((3 - Real.sqrt 5) / 2) 3 = 
  Real.log ((3 - Real.sqrt 5) / 2) - (11 - 3 * Real.sqrt 5) / 4 ∧
  f((3 + Real.sqrt 5) / 2) 3 = 
  Real.log ((3 + Real.sqrt 5) / 2) - (11 + 3 * Real.sqrt 5) / 4 := sorry

theorem part_II (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (a : ℝ)
                (h : x2 / x1 ≥ Real.exp 1) (extreme_pts : x1 * x2 = 1) :
  (f x2 a - f x1 a) ≤ (1 - Real.exp 1 / 2 + 1 / (2 * Real.exp 1)) := sorry

end part_I_part_II_l742_742599


namespace square_D_perimeter_l742_742729

theorem square_D_perimeter 
(C_perimeter: Real) 
(D_area_ratio : Real) 
(hC : C_perimeter = 32) 
(hD : D_area_ratio = 1/3) : 
    ∃ D_perimeter, D_perimeter = (32 * Real.sqrt 3) / 3 := 
by 
    sorry

end square_D_perimeter_l742_742729


namespace exists_circumcircle_of_triangle_AB1C_l742_742278

noncomputable theory

variables {α : Type*} [linear_order α]

-- Define the set of points
def points : set (α × α) := sorry  -- Placeholder for the set of points definition

-- Define point A and segment AB
variables (A B : α × α) (AB : set (α × α)) (hAB : AB = {p | ∃ λ, 0 ≤ λ ∧ λ ≤ 1 ∧ p = (1-λ) • A + λ • B})
-- Define B1 as the nearest point to A on AB
def nearest_point_B1 : α × α := sorry  -- Placeholder for the nearest point B1

-- Define the point C such that ∠AB1C is maximized
variables (C : α × α) (hangle : ∀ p ∈ points, sorry)  -- Placeholder for angle condition

-- Form the triangle AB1C
def triangle_AB1C : set (α × α) := {A, nearest_point_B1, C}

-- State the theorem
theorem exists_circumcircle_of_triangle_AB1C :
  ∃ (circumcircle : set (α × α)), circumference resembling the circumscribed circle around the triangle {A, nearest_point_B1, C} := sorry

end exists_circumcircle_of_triangle_AB1C_l742_742278


namespace magnitude_of_b_l742_742613

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

theorem magnitude_of_b :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ
  let a_dot_b := dot_product a b
  let a_plus_b := vector_add a b
  (a_dot_b = 5) → (vector_magnitude a_plus_b = 8) → (vector_magnitude b = 7) :=
begin
  intros a_dot_b_cond a_plus_b_magnitude_cond,
  sorry
end

end magnitude_of_b_l742_742613


namespace probability_between_D_and_E_l742_742266

theorem probability_between_D_and_E
    (A B C D E : Type)
    (AB_length : ℚ)
    (h1 : AB_length = 4 * (AB_length / 4))
    (h2 : AB_length = 8 * (AB_length / 8)) :
    (∀ (x : ℚ), 0 ≤ x ∧ x ≤ AB_length → 
        (∃ (d e : ℚ), d = AB_length * (1/4) ∧ e = AB_length * (7/8) → 
        x > d ∧ x < e)) →
    probability (x ∈ set.Ioo (AB_length * (1/4)) (AB_length * (7/8))) (uniform (set.Icc 0 AB_length)) = 5/8 := 
sorry

end probability_between_D_and_E_l742_742266


namespace tan_150_deg_l742_742042

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742042


namespace range_of_a_l742_742946

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l742_742946


namespace find_A_and_phi_and_sin_value_l742_742597

theorem find_A_and_phi_and_sin_value
  (A φ : ℝ)
  (hA : A > 0)
  (hphi : φ ∈ (0 : ℝ, real.pi / 2))
  (hx1 : f (real.pi / 12) = sqrt 3)
  (hx2 : f (real.pi / 4) = sqrt 3)
  (hx3 : f x0 = 6 / 5)
  (hx0 : x0 ∈ [(real.pi / 4) : ℝ, real.pi / 2]) :
  A = 2 ∧ φ = real.pi / 6 ∧ sin (2 * x0 - real.pi / 12) = 7 * sqrt 2 / 10 := by
  sorry

noncomputable def f (x : ℝ) : ℝ := A * sin (2 * x + φ)

end find_A_and_phi_and_sin_value_l742_742597


namespace slope_of_line_m_equals_neg_2_l742_742315

theorem slope_of_line_m_equals_neg_2
  (m : ℝ)
  (h : (3 * m - 6) / (1 + m) = 12) :
  m = -2 :=
sorry

end slope_of_line_m_equals_neg_2_l742_742315


namespace binom_sum_l742_742381

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742381


namespace sum_of_n_values_l742_742361

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742361


namespace time_to_school_gate_l742_742702

theorem time_to_school_gate (total_time gate_to_building building_to_room time_to_gate : ℕ) 
                            (h1 : total_time = 30)
                            (h2 : gate_to_building = 6)
                            (h3 : building_to_room = 9)
                            (h4 : total_time = time_to_gate + gate_to_building + building_to_room) :
  time_to_gate = 15 :=
  sorry

end time_to_school_gate_l742_742702


namespace garden_perimeter_l742_742492

-- formally defining the conditions of the problem
variables (x y : ℝ)
def diagonal_of_garden : Prop := x^2 + y^2 = 900
def area_of_garden : Prop := x * y = 216

-- final statement to prove the perimeter of the garden
theorem garden_perimeter (h1 : diagonal_of_garden x y) (h2 : area_of_garden x y) : 2 * (x + y) = 73 := sorry

end garden_perimeter_l742_742492


namespace range_of_a_l742_742618

theorem range_of_a (a : ℝ) (x : ℝ) :
  ((a < x ∧ x < a + 2) → x > 3) ∧ ¬(∀ x, (x > 3) → (a < x ∧ x < a + 2)) → a ≥ 3 :=
by
  sorry

end range_of_a_l742_742618


namespace group_size_l742_742907

noncomputable def total_cost : ℤ := 13500
noncomputable def cost_per_person : ℤ := 900

theorem group_size : total_cost / cost_per_person = 15 :=
by {
  sorry
}

end group_size_l742_742907


namespace circle_area_equilateral_triangle_l742_742460

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l742_742460


namespace area_of_inequality_l742_742929

theorem area_of_inequality : 
  let S := { p : ℝ × ℝ | ∃ (x y : ℝ), 
             0 ≤ x ∧ x ≤ 3 ∧ -3 ≤ y ∧ y ≤ 3 ∧ 
             sqrt (arcsin (x / 3)) ≤ sqrt (arccos (y / 3)) } in
  (∫ p in S, 1) = 16 :=
sorry

end area_of_inequality_l742_742929


namespace total_sticks_needed_l742_742717

theorem total_sticks_needed :
  let simon_sticks := 36
  let gerry_sticks := 2 * (simon_sticks / 3)
  let total_simon_and_gerry := simon_sticks + gerry_sticks
  let micky_sticks := total_simon_and_gerry + 9
  total_simon_and_gerry + micky_sticks = 129 :=
by
  sorry

end total_sticks_needed_l742_742717


namespace tan_150_eq_neg_inv_sqrt3_l742_742061

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742061


namespace tan_150_deg_l742_742034

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742034


namespace gcd_lcm_product_l742_742934

theorem gcd_lcm_product (a b : ℕ) (ha : a = 18) (hb : b = 42) :
  Nat.gcd a b * Nat.lcm a b = 756 :=
by
  rw [ha, hb]
  sorry

end gcd_lcm_product_l742_742934


namespace ordered_pairs_count_l742_742933

theorem ordered_pairs_count :
  let valid_pairs_count := (finset.range 50).sum (λ a, (a + 1) / 2)
  ∑ a in finset.range 50, (a + 1) / 2 = 925 := by
  let valid_pairs_count := (finset.range 50).sum (λ a, (a + 1) / 2)
  sorry

end ordered_pairs_count_l742_742933


namespace unique_solution_l742_742924

noncomputable def f : ℝ → ℝ := sorry

theorem unique_solution : (∀ x y : ℝ, f(x + y) = max (f x) y + min (f y) x) → (f = id) := sorry

end unique_solution_l742_742924


namespace chocolate_milk_tea_cups_l742_742845

-- Defining the conditions as constants or variables
def total_sales : ℕ := 50
def fraction_winter_melon : ℚ := 2/5
def fraction_okinawa : ℚ := 3/10

-- Calculating the number of winter melon and Okinawa flavored cups.
def cups_winter_melon : ℕ := (fraction_winter_melon * total_sales).natAbs
def cups_okinawa : ℕ := (fraction_okinawa * total_sales).natAbs

-- The theorem to prove the number of chocolate-flavored milk tea cups.
theorem chocolate_milk_tea_cups : 
  total_sales - (cups_winter_melon + cups_okinawa) = 15 :=
by
  -- Placeholder for the proof
  sorry

end chocolate_milk_tea_cups_l742_742845


namespace anca_rest_time_l742_742871

theorem anca_rest_time
  (speed_bruce : ℝ := 50) 
  (speed_anca : ℝ := 60) 
  (distance : ℝ := 200)
  (same_start : true)
  (same_end : true) :
  let time_bruce := distance / speed_bruce in
  let time_anca_without_stop := distance / speed_anca in
  let anca_rest_time := time_bruce - time_anca_without_stop in
  anca_rest_time * 60 = 40 :=
by
  sorry

end anca_rest_time_l742_742871


namespace max_pairs_on_board_l742_742949

theorem max_pairs_on_board : ∃ A B : Finset ℕ, 
  (∀ a ∈ A, ∀ b ∈ B, a < b ∧ a ≤ 2018 ∧ b ≤ 2018) ∧
  A ∩ B = ∅ ∧
  (∀ a1 a2 ∈ A, ∀ b1 b2 ∈ B, 
    ((a1, b1) ≠ (a2, b2) → 
     ((a1 ≠ b2) ∧ (b1 ≠ a2)) ∧ 
     ((a2, a1) ∉ (A.product B)) ∧ 
     ((b2, b1) ∉ (A.product B))) → 
  A.card + B.card = 2018) ∧ n = 1009 ∧
  ∀ m n : ℕ, m * n = 1018081
  sorry

end max_pairs_on_board_l742_742949


namespace circle_area_of_equilateral_triangle_l742_742443

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742443


namespace circumscribed_circle_area_l742_742466

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742466


namespace system_solution_l742_742727

variables (a b c x y z : ℝ)

-- Conditions
def condition1 : Prop := x + a * y + a^2 * z = a^3
def condition2 : Prop := x + b * y + b^2 * z = b^3
def condition3 : Prop := x + c * y + c^2 * z = c^3

-- Correct answers
def x_value : ℝ := a * b * c
def y_value : ℝ := -(a * b + b * c + c * a)
def z_value : ℝ := a + b + c

theorem system_solution :
  condition1 a b c (x_value a b c) (y_value a b c) (z_value a b c) ∧
  condition2 a b c (x_value a b c) (y_value a b c) (z_value a b c) ∧
  condition3 a b c (x_value a b c) (y_value a b c) (z_value a b c) :=
by
  sorry

end system_solution_l742_742727


namespace quadratic_polynomial_solution_count_l742_742961

theorem quadratic_polynomial_solution_count
  (f : ℝ → ℝ)
  (hf : ∃ a b c, ∀ x, f(x) = a * x^2 + b * x + c)
  (h_solutions : (∃ x₁ x₂ x₃, (f(x₁))^3 - 4 * (f(x₁)) = 0 ∧ (f(x₂))^3 - 4 * (f(x₂)) = 0 ∧ (f(x₃))^3 - 4 * (f(x₃)) = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) :
  ∃ y₁ y₂ : ℝ, (f(y₁))^2 = 1 ∧ (f(y₂))^2 = 1 ∧ y₁ ≠ y₂ :=
by
  sorry

end quadratic_polynomial_solution_count_l742_742961


namespace PQSR_ratio_l742_742525

noncomputable def s : ℝ := 1
noncomputable def W (x y : ℝ) : Prop := x = y
noncomputable def WY (x y midpoint : ℝ): Prop := midpoint = (x + y) / 2
noncomputable def WU (x y midpoint : ℝ): Prop := midpoint = (x + y) / 2

theorem PQSR_ratio (s : ℝ) (PQRS TUVW WXYZ : ℝ) (S R : ℝ)
  (h1 : s = 1)
  (h2 : WY WXYZ S)
  (h3 : WU TUVW R):
  (area_of_PQSR : ℝ) / (area_of_PQRS + area_of_TUVW + area_of_XYZ) = 1 / 12 :=
sorry

end PQSR_ratio_l742_742525


namespace area_of_circumscribed_circle_l742_742477

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742477


namespace domain_of_f_l742_742294

def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
∀ x, f x ∈ D

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem domain_of_f :
  domain f {y | y ≠ -2} :=
by sorry

end domain_of_f_l742_742294


namespace sqrt_x_minus_1_meaningful_l742_742139

theorem sqrt_x_minus_1_meaningful (x : ℝ) : x ≥ 1 → exists (y : ℝ), y = sqrt (x - 1) :=
by
  intro h
  use sqrt (x - 1)
  sorry

end sqrt_x_minus_1_meaningful_l742_742139


namespace smallest_delightful_integer_l742_742714

def is_delightful (B : ℤ) : Prop :=
  ∃ (n : ℕ) (a : ℤ), (list.sum (list.map (λ i, a + i) (list.range n)) = 2017 ∧ B ∈ (list.map (λ i, a + i) (list.range n)))

theorem smallest_delightful_integer : ∀ (B : ℤ), is_delightful B → B ≥ -2016 :=
sorry

end smallest_delightful_integer_l742_742714


namespace tan_150_eq_neg_inv_sqrt3_l742_742075

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742075


namespace sum_of_valid_n_l742_742374

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742374


namespace sector_to_cone_height_l742_742834

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l742_742834


namespace fraction_to_decimal_l742_742113

theorem fraction_to_decimal :
  (7 / 12 : ℝ) ≈ 0.5833 :=
begin
  sorry
end

end fraction_to_decimal_l742_742113


namespace solve_system_l742_742282

def system_of_equations (x y : ℤ) : Prop :=
  (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧
  (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0)

theorem solve_system : system_of_equations (-3) (-1) :=
by {
  -- Proof details are omitted
  sorry
}

end solve_system_l742_742282


namespace range_of_m_l742_742630

-- Definitions
def is_circle_eqn (d e f : ℝ) : Prop :=
  d^2 + e^2 - 4 * f > 0

-- Main statement 
theorem range_of_m (m : ℝ) : 
  is_circle_eqn (-2) (-4) m → m < 5 :=
by
  intro h
  sorry

end range_of_m_l742_742630


namespace number_of_gallons_per_white_paint_can_l742_742882

theorem number_of_gallons_per_white_paint_can
  (bedrooms : ℕ)
  (other_rooms : ℕ)
  (rooms_paint_gallons : ℕ)
  (total_paint_cans : ℕ)
  (colored_paint_can_size : ℕ)
  (white_paint_gallons_needed : ℕ)
  (white_paint_cans : ℕ)
  (h1 : bedrooms = 3)
  (h2 : other_rooms = 2 * bedrooms)
  (h3 : rooms_paint_gallons = 2)
  (h4 : total_paint_cans = 10)
  (h5 : colored_paint_can_size = 1)
  (h6 : white_paint_gallons_needed = 12)
  (h7 : white_paint_cans = total_paint_cans - bedrooms * colored_paint_can_size)
  : (white_paint_gallons_needed / white_paint_cans) = 3 := 
by 
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end number_of_gallons_per_white_paint_can_l742_742882


namespace quadratic_has_minimum_l742_742850

theorem quadratic_has_minimum 
  (a b : ℝ) (h : a ≠ 0) (g : ℝ → ℝ) 
  (H : ∀ x, g x = a * x^2 + b * x + (b^2 / a)) :
  ∃ x₀, ∀ x, g x ≥ g x₀ :=
by sorry

end quadratic_has_minimum_l742_742850


namespace blossom_room_area_l742_742777

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l742_742777


namespace area_of_circumscribed_circle_l742_742474

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l742_742474


namespace sum_of_integer_values_l742_742351

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742351


namespace percentage_increase_l742_742636

theorem percentage_increase (X Y Z : ℝ) (h1 : X = 1.25 * Y) (h2 : Z = 100) (h3 : X + Y + Z = 370) :
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l742_742636


namespace fish_filets_total_l742_742879

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end fish_filets_total_l742_742879


namespace rectangle_area_l742_742852

-- Definitions from conditions:
def side_length : ℕ := 16 / 4
def area_B : ℕ := side_length * side_length
def probability_not_within_B : ℝ := 0.4666666666666667

-- Main statement to prove
theorem rectangle_area (A : ℝ) (h1 : side_length = 4)
 (h2 : area_B = 16)
 (h3 : probability_not_within_B = 0.4666666666666667) :
   A * 0.5333333333333333 = 16 → A = 30 :=
by
  intros h
  sorry


end rectangle_area_l742_742852


namespace tangent_line_at_x0_l742_742744

theorem tangent_line_at_x0 
  (a b : ℝ)
  (h1 : ∀ (x : ℝ), y = x * exp x - a * exp x - b * x)
  (h2 : ∀ (x : ℝ), deriv (λ x, x * exp x - a * exp x - b * x) x = 1 - a - b)
  (h3 : ∀ (x : ℝ), (x = 0) → y = x - 1) : 
  (a = 1) ∧ (b = -1) :=
by
  sorry

end tangent_line_at_x0_l742_742744


namespace number_of_valid_sets_l742_742253

-- Definitions for the conditions
def is_valid_set (P : Set ℕ) : Prop :=
  {1, 2} ⊆ P ∧ P ⊆ {0, 1, 2, 3, 4}

-- The theorem we need to prove
theorem number_of_valid_sets : 
  (Finset.filter is_valid_set (Finset.powerset {0, 1, 2, 3, 4})).card = 8 := 
by 
  sorry

end number_of_valid_sets_l742_742253


namespace odd_digits_in_base_5_of_157_l742_742556

def number_of_odd_digits_in_base_5_representation (n : ℕ) : ℕ :=
  let digits := Nat.digits 5 n
  digits.count (λ d => d % 2 = 1)
  
theorem odd_digits_in_base_5_of_157 : number_of_odd_digits_in_base_5_representation 157 = 3 :=
by
  sorry

end odd_digits_in_base_5_of_157_l742_742556


namespace units_digit_2189_pow_1242_l742_742889

theorem units_digit_2189_pow_1242 : (2189 ^ 1242) % 10 = 1 := by
  -- The units digit of 2189 is the same as the units digit of 9
  have units_digit_2189 : 2189 % 10 = 9 := by
    norm_num,
  -- Consider the units digit pattern of powers of 9
  have units_digit_pattern : ∀ n, 9^n % 10 = if n % 2 = 0 then 1 else 9 := by
    sorry,  -- Detailed steps omitted
  -- Calculate (1242 mod 2) to determine the appropriate units digit
  have exp_mod_2 : 1242 % 2 = 0 := by
    norm_num,
  -- Deduce the units digit of 9^1242 based on pattern
  show 9^1242 % 10 = 1 from by
    rw [units_digit_pattern, exp_mod_2],
    norm_num

end units_digit_2189_pow_1242_l742_742889


namespace total_cost_of_oranges_and_mangoes_l742_742541

theorem total_cost_of_oranges_and_mangoes
  (original_price_orange : ℝ)
  (original_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  original_price_orange = 40 →
  original_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  (quantity_oranges * (original_price_orange * (1 + price_increase_percentage)) +
   quantity_mangoes * (original_price_mango * (1 + price_increase_percentage))
  ) = 1035 :=
begin
  intros,
  sorry
end

end total_cost_of_oranges_and_mangoes_l742_742541


namespace smallest_number_of_sparrows_in_each_flock_l742_742883

theorem smallest_number_of_sparrows_in_each_flock (P : ℕ) (H : 14 * P ≥ 182) : 
  ∃ S : ℕ, S = 14 ∧ S ∣ 182 ∧ (∃ P : ℕ, S ∣ (14 * P)) := 
by 
  sorry

end smallest_number_of_sparrows_in_each_flock_l742_742883


namespace sum_of_valid_n_l742_742334

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742334


namespace tan_150_deg_l742_742023

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742023


namespace unique_representation_l742_742913

theorem unique_representation (n : ℤ) (h : n ≥ 1) : ∃! (p q : ℤ), n = 2^p * (2 * q + 1) :=
by
  sorry

end unique_representation_l742_742913


namespace find_x_l742_742144

def a : ℝ × ℝ × ℝ := (3, 2, 5)
def b (x : ℝ) : ℝ × ℝ × ℝ := (4, -1, x)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_x (x : ℝ) (h : dot_product a (b x) = 0) : x = -2 :=
sorry

end find_x_l742_742144


namespace part1_part2_l742_742174

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 {a : ℝ} (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := 
by 
  rw h 
  sorry

theorem part2 : 
  (∃ x : ℝ, f x a ≤ 4) ↔ -3 ≤ a ∧ a ≤ 5 := 
by 
  sorry

end part1_part2_l742_742174


namespace reflection_sum_length_l742_742505

/-
Conditions:
1. All dihedral angles of the trihedral angle \(N K L M\) are right angles.
2. Point \(P\) is on the face \(L N M\), \(2\) units from \(N\) and \(1\) unit from the edge \(M N\).
3. A point \(S\) inside \(N K L M\) sends a beam to \(P\).
4. The beam forms an angle \(\pi/4\) with the plane \(M N K\), and equal angles with the edges \(K N\) and \(M N\).
5. The beam reflects off planes at \(P\), \(Q\), and \(R\).
-/

theorem reflection_sum_length
  (N K L M P Q R : Type)
  (N P : N)
  (d₁ : N.distance P = 2)
  (d₂ : edge.distance P = 1)
  (angle₁ : beam.angle (plane.mk M N K) = π / 4)
  (angle₂ : beam.angle (edge.mk K N) = beam.angle (edge.mk M N))
  (reflection_P : reflects_on P)
  (reflection_Q : reflects_on Q)
  (reflection_R : reflects_on R) :
  segment.length P Q + segment.length Q R = 2 * sqrt 3 := sorry

end reflection_sum_length_l742_742505


namespace intersection_points_l742_742117

noncomputable def p_intersect (rho theta : ℝ) : Prop :=
  ∃ t : ℝ, (let x := 2 + (1/2) * t) in
            (let y := (sqrt 3 / 2) * t) in
            (rho = sqrt (x^2 + y^2)) ∧ (theta = atan2 y x)

theorem intersection_points (rho1 theta1 rho2 theta2 : ℝ) :
  p_intersect rho1 theta1 ∧ p_intersect rho2 theta2 ↔
  (rho1 = 2 ∧ theta1 = 5 * π / 3) ∧ (rho2 = 2 * sqrt 3 ∧ theta2 = π / 6) :=
by
  sorry

end intersection_points_l742_742117


namespace find_part_length_in_inches_find_part_length_in_feet_and_inches_l742_742493

def feetToInches (feet : ℕ) : ℕ := feet * 12

def totalLengthInInches (feet : ℕ) (inches : ℕ) : ℕ := feetToInches feet + inches

def partLengthInInches (totalLength : ℕ) (parts : ℕ) : ℕ := totalLength / parts

def inchesToFeetAndInches (inches : ℕ) : Nat × Nat := (inches / 12, inches % 12)

theorem find_part_length_in_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    partLengthInInches (totalLengthInInches feet inches) parts = 25 := by
  sorry

theorem find_part_length_in_feet_and_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    inchesToFeetAndInches (partLengthInInches (totalLengthInInches feet inches) parts) = (2, 1) := by
  sorry

end find_part_length_in_inches_find_part_length_in_feet_and_inches_l742_742493


namespace num_complex_solutions_l742_742133

noncomputable def eq_frac := λ z : ℂ, (z ^ 4 - 1) / (z ^ 3 + z ^ 2 - 2z) = 0

theorem num_complex_solutions :
  {z : ℂ | eq_frac z}.finite.to_finset.card = 3 :=
sorry

end num_complex_solutions_l742_742133


namespace concurrency_of_lines_l742_742681

variables {Ω : Type*} {A B C D E F T K : Ω}
variables [hABCD : QuadrilateralInscribedInCircle A B C D Ω]
variables [hTangentAtD : TangentAtPointOmega D E F Ω]
variables [hTInABC : PointInTriangle T A B C]
variables [hTEParallelCD : Parallel TE CD]
variables [hTFParallelAD : Parallel TF AD]
variables [hTD_EQ_TK : TD = TK]

def lines_concur (l1 l2 l3 : Line Ω) : Prop :=
  ∃ (P : Ω), P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3

theorem concurrency_of_lines
  (h: ∀ {A B C D E F T K : Ω} [hABCD : QuadrilateralInscribedInCircle A B C D Ω]
        [hTangentAtD : TangentAtPointOmega D E F Ω]
        [hTInABC : PointInTriangle T A B C]
        [hTEParallelCD : Parallel TE CD]
        [hTFParallelAD : Parallel TF AD]
        [hTD_EQ_TK : TD = TK],
        lines_concur (line_through A C) (line_through D T) (line_through B K) ) : 
  ∀ A B C D E F T K, 
  QuadrilateralInscribedInCircle A B C D Ω → 
  TangentAtPointOmega D E F Ω → 
  PointInTriangle T A B C → 
  Parallel TE CD → 
  Parallel TF AD → 
  TD = TK → 
  lines_concur (line_through A C) (line_through D T) (line_through B K) := 
sorry

end concurrency_of_lines_l742_742681


namespace total_cost_of_oranges_and_mangoes_l742_742540

theorem total_cost_of_oranges_and_mangoes
  (original_price_orange : ℝ)
  (original_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  original_price_orange = 40 →
  original_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  (quantity_oranges * (original_price_orange * (1 + price_increase_percentage)) +
   quantity_mangoes * (original_price_mango * (1 + price_increase_percentage))
  ) = 1035 :=
begin
  intros,
  sorry
end

end total_cost_of_oranges_and_mangoes_l742_742540


namespace find_a_l742_742995

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end find_a_l742_742995


namespace ellipse_equation_constant_pa_pb_l742_742163

noncomputable def equation_of_ellipse : Prop :=
  ∃ (a b : ℝ) (b_pos : 0 < b) (a_ge_b : a > b), a = 2 ∧ 
  (∀ (x y : ℝ), (x, y).mem ( set_of₂ (λ x y, x^2 / 4 + y^2 = 1)) ↔ 
  (x = 1 ∧ y = (by norm_num : real.sqrt 3 / 2)))

noncomputable def pa_pb_constant : Prop :=
  ∀ (m : ℝ), (-2 ≤ m ∧ m ≤ 2) → 
  let l := λ x, (x - m) / 2 in 
  ∃ (A B : ℝ × ℝ), A ∈ (set_of₂ (λ x y, y = (x - m) / 2)) ∧
  B ∈ (set_of₂ (λ x y, y = (x - m) / 2)) ∧
  |m - A.fst|^2 + |B.snd|^2 = 5 ∧ |m - B.fst|^2 + |A.snd|^2 = 5 

theorem ellipse_equation_constant_pa_pb :
  equation_of_ellipse ∧ pa_pb_constant :=
by
  -- proof of the theorem
  sorry

end ellipse_equation_constant_pa_pb_l742_742163


namespace Blossom_room_area_square_inches_l742_742774

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

theorem Blossom_room_area_square_inches :
  (let length_feet := 10 in
   let width_feet := 10 in
   let length_inches := feet_to_inches length_feet in
   let width_inches := feet_to_inches width_feet in
   length_inches * width_inches = 14400) :=
by
  let length_feet := 10
  let width_feet := 10
  let length_inches := feet_to_inches length_feet
  let width_inches := feet_to_inches width_feet
  show length_inches * width_inches = 14400
  sorry

end Blossom_room_area_square_inches_l742_742774


namespace find_f_on_interval_l742_742244

/-- Representation of periodic and even functions along with specific interval definition -/
noncomputable def f (x : ℝ) : ℝ := 
if 2 ≤ x ∧ x ≤ 3 then -2*(x-3)^2 + 4 else 0 -- Define f(x) on [2,3], otherwise undefined

/-- Main proof statement -/
theorem find_f_on_interval :
  (∀ x, f x = f (x + 2)) ∧  -- f(x) is periodic with period 2
  (∀ x, f x = f (-x)) ∧   -- f(x) is even
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x = -2*(x-3)^2 + 4) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = -2*(x-1)^2 + 4) :=
sorry

end find_f_on_interval_l742_742244


namespace hyperbola_eccentricity_range_l742_742175

theorem hyperbola_eccentricity_range (a b e : ℝ)
  (ha : a > 0) (hb : b > 0)
  (hyp_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
  ((∃ x y, y = √3 * (x - a * e) ∧ x^2 / a^2 - y^2 / b^2 = 1) →
   ∀ m c, c^2 = a^2 * m^2 - b^2 → m = √3 ∧ c = -a * √3 * e ∧ 1 - 3 * e^2 ≤ 0) ∧
   ∀ e, 1 - 3 * e^2 < 0 → e ≥ 2 :=
begin
  sorry
end

end hyperbola_eccentricity_range_l742_742175


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742082

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742082


namespace sum_of_valid_n_l742_742377

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l742_742377


namespace tan_150_deg_l742_742039

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742039


namespace size_of_angle_A_area_of_triangle_l742_742239

-- Definitions for the problem
variables {a b c : ℝ} {A B C : ℝ}
def obtuse_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π

def given_condition1 (a b c A B C : ℝ) : Prop :=
  2 * a * Real.sin A = (2 * b - Real.sqrt 3 * c) * Real.sin B + (2 * c - Real.sqrt 3 * b) * Real.sin C

def given_condition2 (a b : ℝ) : Prop :=
  a = 2 ∧ b = 2 * Real.sqrt 3

-- Lean statement for the proof problem
theorem size_of_angle_A (h1 : obtuse_triangle A B C) (h2 : given_condition1 a b c A B C) :
  A = π / 6 :=
sorry

theorem area_of_triangle (h1 : obtuse_triangle A B C) (h2 : given_condition1 a b c A B C) (h3 : given_condition2 a b) :
  ∃ S, S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3 :=
sorry

end size_of_angle_A_area_of_triangle_l742_742239


namespace tan_150_eq_l742_742008

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742008


namespace evaluate_expression_l742_742124

theorem evaluate_expression : 
    ( (1 / 8 : ℝ)^(1 / 3) - real.log 2 / real.log 3 * real.log 27 / real.log 4 + 2018^0 = 0) :=
by {
    sorry
}

end evaluate_expression_l742_742124


namespace setB_is_empty_l742_742410

noncomputable def setB := {x : ℝ | x^2 + 1 = 0}

theorem setB_is_empty : setB = ∅ :=
by
  sorry

end setB_is_empty_l742_742410


namespace line_equations_l742_742296

theorem line_equations (
  eq_1 : ∀ x y : ℝ, y = 3 * x ↔ (x, y) = (1, 3) ∨ x = 1 ∧ y = 3) 
  (eq_2 : ∀ x y : ℝ, x + y - 4 = 0 ↔ (x, y) = (1, 3) ∨ x = 1 ∧ y = 3) 
  (eq_3 : ∀ x y : ℝ, x - y + 2 = 0 ↔ (x, y) = (1, 3) ∨ x = 1 ∧ y = 3) 
  (condition_1 : ∃ a b : ℝ, a + b = 0 ∨ a - b = 0) 
  (condition_2 : ∃ a b : ℝ, a ∧ b ≠ 0) : 
  (∀ x y : ℝ, y = 3 * x ↔ x = 1 ∧ y = 3) ∧ 
  (∀ x y : ℝ, x + y - 4 = 0 ↔ x = 1 ∧ y = 3) ∧ 
  (∀ x y : ℝ, x - y + 2 = 0 ↔ x = 1 ∧ y = 3) := sorry

end line_equations_l742_742296


namespace perimeter_of_square_D_l742_742730

-- Definitions based on conditions
def perimeter_C : ℝ := 32
def side_len_C : ℝ := perimeter_C / 4
def area_C : ℝ := side_len_C * side_len_C
def area_D : ℝ := area_C / 3
def side_len_D : ℝ := Real.sqrt area_D
def perimeter_D : ℝ := 4 * side_len_D

-- Theorem to prove the perimeter of square D
theorem perimeter_of_square_D : perimeter_D = (32 * Real.sqrt 3) / 3 :=
by
  sorry

end perimeter_of_square_D_l742_742730


namespace baking_cookies_l742_742851

theorem baking_cookies (flour_per_40 : ℕ) (sugar_per_40 : ℕ) (target_cookies : ℕ)
  (recipe_cookies : ℕ) (scaling_factor : ℕ)
  (h1 : flour_per_40 = 3) (h2 : sugar_per_40 = 1) (h3 : recipe_cookies = 40)
  (h4 : target_cookies = 200) (h5 : scaling_factor = target_cookies / recipe_cookies) :
  (flour_needed : ℕ) (sugar_needed : ℕ) (h6 : flour_needed = flour_per_40 * scaling_factor)
  (h7 : sugar_needed = sugar_per_40 * scaling_factor) : flour_needed = 15 ∧ sugar_needed = 5 :=
by
  -- Proof skipped
  sorry

end baking_cookies_l742_742851


namespace tan_150_eq_neg_one_over_sqrt_three_l742_742086

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l742_742086


namespace tan_150_eq_neg_sqrt3_div_3_l742_742099

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742099


namespace circumscribed_circle_area_l742_742463

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742463


namespace chen_recording_l742_742435

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end chen_recording_l742_742435


namespace common_region_area_l742_742792

def triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def circumcircle (a b c : ℝ) : Prop := 
  let s := (a + b + c) / 2 in
    4 * s * (s - a) * (s - b) * (s - c) = c^2 * (a + b + c - c^2)

def incircle (a b c : ℝ) : Prop := 
  let s := (a + b + c) / 2 in 
    (s - a) * (s - b) * (s - c) / s = 4 * s * (s - a) * (s - b) * (s - c) / (a + b + c)

theorem common_region_area {a b c : ℝ} (h1 : a = 18) (h2 : b = 24) (h3 : c = 30)
  (h_triangle1 : triangle a b c) (h_circumcircle: circumcircle a b c) (h_incircle: incircle a b c)
  (h_triangle2 : triangle a b c) : 
  ∃ area, area = 132 := 
by 
  sorry

end common_region_area_l742_742792


namespace area_of_quadrilateral_is_18_l742_742570

-- Definitions based on given conditions
def real_line_l1 (a : ℝ) : set (ℝ × ℝ) := {p | p.1 + 2 * p.2 = a + 2}
def real_line_l2 (a : ℝ) : set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 2 * a - 1}
def real_circle_E (a : ℝ) : set (ℝ × ℝ) := {p | (p.1 - a) ^ 2 + (p.2 - 1) ^ 2 = 9}

-- Defining the points of intersection
variable (a : ℝ)
def intersection_points : set (ℝ × ℝ) := 
  (real_line_l1 a ∩ real_circle_E a) ∪ (real_line_l2 a ∩ real_circle_E a)

-- Theorem statement
theorem area_of_quadrilateral_is_18 (a : ℝ) 
  (h1 : ∃ p, p ∈ real_line_l1 a ∧ p ∈ real_circle_E a) 
  (h2 : ∃ p, p ∈ real_line_l2 a ∧ p ∈ real_circle_E a) 
  (h_intersect_at_center: ∃ p, (p ∈ real_line_l1 a) ∧ (p ∈ real_line_l2 a) ∧ p = (a, 1)) 
  (h_perpendicular: ∀ p1 p2, p1 ∈ real_line_l1 a ∧ p2 ∈ real_line_l2 a → 
    let (x1, y1) := p1 in 
    let (x2, y2) := p2 in 
    (x2 - x1) * (y2 - y1) = -1) : 
  (calc_area_quad (intersection_points a)) = 18 :=
sorry

end area_of_quadrilateral_is_18_l742_742570


namespace sequence_includes_1_or_4_l742_742572

def sum_of_squares_of_digits (n : Nat) : Nat :=
  n.digits.map (λ d => d^2).sum

theorem sequence_includes_1_or_4 (a1 : Nat) (h : 100 ≤ a1 ∧ a1 < 1000) :
  ∃ n : Nat, (a1.iterate sum_of_squares_of_digits n) = 1 ∨ (a1.iterate sum_of_squares_of_digits n) = 4 :=
  sorry

end sequence_includes_1_or_4_l742_742572


namespace find_quadratic_polynomial_l742_742935

theorem find_quadratic_polynomial (q : ℝ → ℝ) 
  (h1 : q (-1) = 7) 
  (h2 : q 2 = -3) 
  (h3 : q 4 = 11) :
  q = λ x : ℝ, (31 / 15) * x^2 - (27 / 5) * x - (289 / 15) := 
by 
  sorry

end find_quadratic_polynomial_l742_742935


namespace time_to_cover_escalator_l742_742416

-- Define the conditions as constants
def escalator_speed : ℝ := 12
def escalator_length : ℝ := 210
def person_speed : ℝ := 2

-- Define the combined speed function
def combined_speed (escalator_speed : ℝ) (person_speed : ℝ) : ℝ :=
  escalator_speed + person_speed

-- Define the time function
def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

-- State the theorem to prove
theorem time_to_cover_escalator :
  travel_time escalator_length (combined_speed escalator_speed person_speed) = 15 :=
by
  sorry

end time_to_cover_escalator_l742_742416


namespace travel_distance_correct_l742_742500

noncomputable def traveler_distance : ℝ :=
  let x1 : ℝ := -4
  let y1 : ℝ := 0
  let x2 : ℝ := x1 + 5 * Real.cos (-(Real.pi / 3))
  let y2 : ℝ := y1 + 5 * Real.sin (-(Real.pi / 3))
  let x3 : ℝ := x2 + 2
  let y3 : ℝ := y2
  Real.sqrt (x3^2 + y3^2)

theorem travel_distance_correct : traveler_distance = Real.sqrt 19 := by
  sorry

end travel_distance_correct_l742_742500


namespace solve_eq_l742_742746

theorem solve_eq (x : ℝ) : (x - 2)^2 = 9 * x^2 ↔ x = -1 ∨ x = 1 / 2 := by
  sorry

end solve_eq_l742_742746


namespace sum_of_possible_values_l742_742915

theorem sum_of_possible_values (E D : ℕ) (h1 : (E * 10000 + 2700 + D * 10 + 6) % 8 = 0) 
  (h2 : E < 10) (h3 : D < 10):
  ∑ i in (finset.range 10), (1 + i) = 55 :=
by
  sorry

end sum_of_possible_values_l742_742915


namespace circle_area_l742_742199

variables {O A B C D E F : Point}
variables {radius : ℝ}

-- Given Conditions
axiom AB_perp_CD : perpendicular (line_through O A) (line_through O C) -- A and C are points on the circle defining diameters
axiom chord_DF_intersects_CD_at_E : line_through D F ∩ line_through C D = {E}
axiom DE_length : dist D E = 4
axiom EF_length : dist E F = 4

-- Derived Conditions
axiom DF_length : dist D F = DE_length + EF_length

-- Prove the area of the circle
theorem circle_area : π * radius^2 = 32 * π :=
by
  sorry

end circle_area_l742_742199


namespace james_carrot_sticks_l742_742659

theorem james_carrot_sticks (carrots_before : ℕ) (carrots_after : ℕ) 
(h_before : carrots_before = 22) (h_after : carrots_after = 15) : 
carrots_before + carrots_after = 37 := 
by 
  -- Placeholder for proof
  sorry

end james_carrot_sticks_l742_742659


namespace john_recycling_earnings_l742_742663

-- Define the necessary conditions
def weight_per_paper (day: String) : ℕ :=
  if day = "Sunday" then 16 else 8

def papers_per_day := 250
def weeks := 10
def ounces_to_pounds := 16
def pounds_to_tons := 2000
def recycle_rate := 20

-- Theorem: Prove that John made $100 from recycling the papers
theorem john_recycling_earnings :
  let total_weekly_weight_ounces := (6 * weight_per_paper "weekday") + weight_per_paper "Sunday",
      total_weekly_weight_pounds := total_weekly_weight_ounces / ounces_to_pounds,
      total_weight_10_weeks_pounds := total_weekly_weight_pounds * papers_per_day * weeks,
      total_weight_10_weeks_tons := total_weight_10_weeks_pounds / pounds_to_tons,
      earnings := total_weight_10_weeks_tons * recycle_rate
  in earnings = 100 :=
by
  sorry

end john_recycling_earnings_l742_742663


namespace zuzika_number_l742_742413

def z (a b c d e : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem zuzika_number (a b c d e : ℕ) (h1 : 1 ≤ a) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) (h5 : d < 10) (h6 : e < 10) :
  let x := z a b c d e in
  (10 * x + 1 = 3 * (100000 + x)) → x = 42857 :=
by
  intros x h
  sorry

end zuzika_number_l742_742413


namespace binom_sum_l742_742379

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742379


namespace expected_value_constant_random_variable_l742_742491

noncomputable def X : ℕ → ℝ := λ n, 7

theorem expected_value_constant_random_variable :
  (∀ n, X n = 7) → ⁅X⁆ = 7 :=
begin
  intro h,
  sorry
end

end expected_value_constant_random_variable_l742_742491


namespace ellipse_vertex_distance_l742_742131

theorem ellipse_vertex_distance :
  ∀ x y : ℝ, (x^2 / 121 + y^2 / 49 = 1) → (2 * real.sqrt 121 = 22) :=
by
  intros x y h
  sorry

end ellipse_vertex_distance_l742_742131


namespace solve_equation_l742_742280

theorem solve_equation (x : ℝ) (h : -x^2 = (3 * x + 1) / (x + 3)) : x = -1 :=
sorry

end solve_equation_l742_742280


namespace triangle_sine_relation_l742_742223

theorem triangle_sine_relation (A B C a b c R : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C) :
  (a^2 - b^2) / c^2 = (Real.sin (A - B)) / (Real.sin C) :=
by
  sorry

end triangle_sine_relation_l742_742223


namespace prob_selecting_A_l742_742212

-- Given set of students
inductive Student
| A | B | C
open Student

-- Function representing the selection process
def selection (s1 s2 : Student) : Prop := 
  s1 ≠ s2

-- Count the number of selections involving a specific student
def count_involving_A (selections : List (Student × Student)) : Nat :=
  selections.count (λ s, s.1 = A ∨ s.2 = A)

-- Main theorem: Probability of selecting student A
theorem prob_selecting_A
  (total_selections : List (Student × Student))
  (sel_involving_A : List (Student × Student)) :
  (total_selections = [(A, B), (A, C), (B, C)]) →
  (sel_involving_A = [(A, B), (A, C)]) →
  (count_involving_A total_selections = 2) →
  (count_involving_A sel_involving_A = 2) →
  (total_selections.length = 3) →
  (sel_involving_A.length = 2) →
  let pA : ℚ := (count_involving_A sel_involving_A : ℚ) / (total_selections.length : ℚ) in
  pA = (2 / 3 : ℚ) := by
  sorry

end prob_selecting_A_l742_742212


namespace points_above_y_eq_x_l742_742803

theorem points_above_y_eq_x (x y : ℝ) : (y > x) → (y, x) ∈ {p : ℝ × ℝ | p.2 < p.1} :=
by
  intro h
  sorry

end points_above_y_eq_x_l742_742803


namespace find_angle_B_l742_742643

theorem find_angle_B (A B C a b c : ℝ) (h₀ : ∀ x, sin x ≠ 0) (h₁ : ∀ x, cos x ≠ 0)
  (h₂ : a = b * sin C / sin B) (h₃ : b = c * sin A / sin C) (h₄ : c = a * sin B / sin A)
  (h₅ : ∀ x y, sin (x + y) = sin x * cos y + cos x * sin y)
  (h₆ : 0 < A) (h₇ : A < π) (h₈ : ∀ x, 0 < x -> ∃ y, y = 2 * cos x)
  (h₉ : ∀ x, x < π -> ∃ y (z : ℝ), cos y = 1 / 2 ∧ y = π / 3)
  (h₁₀ : ∀ x, cos x ≠ 0) :
  (∀ B C a b c : ℝ, (cos C) / (cos B) = (2 * a - c) / b → B = π / 3) :=
by
  sorry

end find_angle_B_l742_742643


namespace range_a_l742_742170

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x
noncomputable def g (a x : ℝ) : ℝ := a * x + 2

theorem range_a (a : ℝ) :
  (set.image f (set.Icc (-1 : ℝ) 2)) ⊆ (set.image (g a) (set.Icc (-1 : ℝ) 2)) ↔
  a ∈ set.Iic (-3/2) ∪ set.Ici 3 :=
begin
  sorry
end

end range_a_l742_742170


namespace max_Δ_cardinality_l742_742669

def is_connected (K : set (ℤ × ℤ)) : Prop :=
  ∀ R S ∈ K, ∃ (ℓ : ℕ) (seq : fin (ℓ + 1) → ℤ × ℤ),
    seq 0 = R ∧ seq ℓ = S ∧
    (∀ i : fin ℓ, (seq i.1 - seq i.succ.1).abs = (1, 0) ∨ (seq i.1 - seq i.succ.1).abs = (0, 1))

def Δ (K : set (ℤ × ℤ)) : set (ℤ × ℤ) :=
  { v | ∃ R S ∈ K, v = (fst S - fst R, snd S - snd R) }

theorem max_Δ_cardinality (n : ℕ) (n_pos : 0 < n) (K : set (ℤ × ℤ))
  (card_K : K.finite) (hk : K.card = 2 * n + 1) :
  is_connected K →
  (Δ K).finite ∧ Δ K.card = 2 * n^2 + 4 * n :=
sorry

end max_Δ_cardinality_l742_742669


namespace min_abs_sum_extrema_l742_742172

noncomputable def f (x a : ℝ) : ℝ := sin x - a * cos x

theorem min_abs_sum_extrema (a : ℝ) :
  asymmetry_axis (f x a) = (3/4) * real.pi →
  let x1 := - (1/4) * real.pi,
      x2 := (3/4) * real.pi in
  |x1 + x2| = real.pi / 2 :=
sorry

end min_abs_sum_extrema_l742_742172


namespace tan_150_eq_neg_inv_sqrt_3_l742_742092

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742092


namespace square_numbers_divisible_by_5_between_20_and_110_l742_742126

theorem square_numbers_divisible_by_5_between_20_and_110 :
  ∃ (y : ℕ), (y = 25 ∨ y = 100) ∧ (∃ (n : ℕ), y = n^2) ∧ 5 ∣ y ∧ 20 < y ∧ y < 110 :=
by
  sorry

end square_numbers_divisible_by_5_between_20_and_110_l742_742126


namespace square_side_length_from_diagonal_l742_742742

theorem square_side_length_from_diagonal : ∀ (d : ℝ), d = 4 → ∃ s : ℝ, s = 2 * real.sqrt 2 ∧ d = s * real.sqrt 2 :=
by
  intros d h
  use 2 * real.sqrt 2
  split
  { refl }
  { rw h
    calc
      4 = (2 * real.sqrt 2) * real.sqrt 2 : by
                      ... sorry  -- Proven in the final steps
  }

end square_side_length_from_diagonal_l742_742742


namespace part1_solution_set_part2_a_range_l742_742988

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := abs x + 2 * abs (x + 2 - a)

-- Part 1: When a = 3, solving the inequality
theorem part1_solution_set (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

-- Part 2: Finding the range of a such that f(x) = g(x-2) >= 1 for all x in ℝ
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := g (x - 2) a

theorem part2_a_range : (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end part1_solution_set_part2_a_range_l742_742988


namespace tan_150_eq_neg_inv_sqrt3_l742_742073

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742073


namespace pass_through_game_probability_l742_742814

theorem pass_through_game_probability : 
  let p1 := 5/6 in
  let p2 := 5/6 in
  p1 * p2 = 25/36 :=
by
  sorry

end pass_through_game_probability_l742_742814


namespace number_of_operations_to_equal_l742_742783

theorem number_of_operations_to_equal (a b : ℤ) (da db : ℤ) (initial_diff change_per_operation : ℤ) (n : ℤ) 
(h1 : a = 365) 
(h2 : b = 24) 
(h3 : da = 19) 
(h4 : db = 12) 
(h5 : initial_diff = a - b) 
(h6 : change_per_operation = da + db) 
(h7 : initial_diff = 341) 
(h8 : change_per_operation = 31) 
(h9 : initial_diff = change_per_operation * n) :
n = 11 := 
by
  sorry

end number_of_operations_to_equal_l742_742783


namespace binom_sum_l742_742385

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742385


namespace area_of_circumscribed_circle_l742_742453

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l742_742453


namespace cone_height_is_correct_l742_742818

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l742_742818


namespace average_weight_increase_l742_742809

theorem average_weight_increase
  (initial_weight replaced_weight : ℝ)
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (h₁ : num_persons = 5)
  (h₂ : replaced_weight = 65)
  (h₃ : avg_increase = 1.5)
  (total_increase : ℝ)
  (new_weight : ℝ)
  (h₄ : total_increase = num_persons * avg_increase)
  (h₅ : total_increase = new_weight - replaced_weight) :
  new_weight = 72.5 :=
by
  sorry

end average_weight_increase_l742_742809


namespace problem_l742_742585

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x ∈ U | x ≠ 0} -- Placeholder, B itself is a generic subset of U
def A : Set ℕ := {x ∈ U | x = 3 ∨ x = 5 ∨ x = 9}

noncomputable def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

axiom h1 : A ∩ B = {3, 5}
axiom h2 : A ∩ C_U B = {9}

theorem problem : A = {3, 5, 9} :=
by
  sorry

end problem_l742_742585


namespace prob_rain_at_least_one_day_l742_742771

noncomputable def prob_rain_saturday := 0.35
noncomputable def prob_rain_sunday := 0.45

theorem prob_rain_at_least_one_day : 
  (1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)) * 100 = 64.25 := 
by 
  sorry

end prob_rain_at_least_one_day_l742_742771


namespace circumscribed_circle_area_l742_742468

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l742_742468


namespace maximize_ratio_incenter_equilateral_l742_742709

theorem maximize_ratio_incenter_equilateral
  (A B C P : Point)
  (hP : inside_triangle P A B C)
  (D E F : Point)
  (hD : foot_of_perpendicular P B C D)
  (hE : foot_of_perpendicular P C A E)
  (hF : foot_of_perpendicular P A B F) :
  (P = incenter A B C ∧ maximize_ratio (P D E F A B C)) ↔ (equilateral_triangle A B C) :=
sorry

end maximize_ratio_incenter_equilateral_l742_742709


namespace complex_numbers_right_triangle_l742_742614

theorem complex_numbers_right_triangle (z : ℂ) (hz : z ≠ 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ 0 ∧ z₂ ≠ 0 ∧ z₁^3 = z₂ ∧
                 (∃ θ₁ θ₂ : ℝ, z₁ = Complex.exp (Complex.I * θ₁) ∧
                               z₂ = Complex.exp (Complex.I * θ₂) ∧
                               (θ₂ - θ₁ = π/2 ∨ θ₂ - θ₁ = 3 * π/2))) →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end complex_numbers_right_triangle_l742_742614


namespace sum_of_valid_n_l742_742339

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742339


namespace find_a_in_integral_l742_742917

theorem find_a_in_integral :
  ∃ a : ℝ, a = 1 ∧ (∫ x in 0..a, 2) = 2 := by
  sorry

end find_a_in_integral_l742_742917


namespace sum_of_n_values_l742_742365

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742365


namespace exists_integers_a_b_c_d_l742_742732

-- Define the problem statement in Lean 4

theorem exists_integers_a_b_c_d (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
by
  sorry

end exists_integers_a_b_c_d_l742_742732


namespace find_m_l742_742595

theorem find_m (m : ℝ) (f : ℝ → ℝ)
  (h_def : ∀ x, f x = -x^3 + 3*x^2 + 9*x + m)
  (h_max : ∀ x ∈ Icc (-2 : ℝ) 2, f x ≤ 20) :
  m = -2 :=
sorry

end find_m_l742_742595


namespace license_plate_probability_l742_742642

open Nat

theorem license_plate_probability :
  let total_combinations : ℕ := 26 * 25 * 24 * 100,
      favorable_combinations : ℕ := 1
  in total_combinations = 1560000 ∧ favorable_combinations = 1 →
      (favorable_combinations : ℚ) / total_combinations = 1 / 1560000 :=
by 
  intros total_combinations favorable_combinations h
  sorry

end license_plate_probability_l742_742642


namespace area_of_quadrilateral_ABCD_l742_742906

open Real

-- Definitions for the sides and angles 
def AB := 5
def BC := 7
def CD := 15
def AD := 13
def angle_ABC := 60 * pi / 180  -- converting degrees to radians

-- Calculation of AC using the Law of Cosines
noncomputable def AC := sqrt(AB^2 + BC^2 - 2 * AB * BC * cos(angle_ABC))

-- Calculate the area of triangle ABC
noncomputable def area_ABC := 1 / 2 * AB * BC * sin(angle_ABC)

-- Semi-perimeter of triangle CAD
noncomputable def s_CAD := (AC + AD + CD) / 2

-- Area using Heron's formula for triangle CAD
noncomputable def area_CAD := sqrt(s_CAD * (s_CAD - AC) * (s_CAD - AD) * (s_CAD - CD))

-- Total area of quadrilateral ABCD
noncomputable def area_ABCD := area_ABC + area_CAD

theorem area_of_quadrilateral_ABCD (AB BC CD AD : ℝ) (angle_ABC : ℝ) 
  (H_AB : AB = 5) (H_BC : BC = 7) (H_CD : CD = 15) (H_AD : AD = 13) 
  (H_angle_ABC : angle_ABC = 60 * pi / 180) :
  area_ABCD = 35 * sqrt 3 / 4 + area_CAD := 
sorry

end area_of_quadrilateral_ABCD_l742_742906


namespace A_superset_C_l742_742951

-- Definitions of the sets as given in the problem statement
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {-1, 3}
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Statement to be proved: A ⊇ C
theorem A_superset_C : A ⊇ C :=
by sorry

end A_superset_C_l742_742951


namespace no_four_digit_palindrome_years_l742_742849

-- Define a predicate for a number being a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

-- Define a predicate for a number being a three-digit prime palindrome
def is_three_digit_prime_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ is_palindrome n ∧ nat.prime n

-- Main theorem statement
theorem no_four_digit_palindrome_years :
  ∀ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n →
    ¬ ∃ m, m = 2 ∧ ∃ p, is_three_digit_prime_palindrome p ∧ n = m * p :=
begin
  -- Proof omitted
  sorry
end

end no_four_digit_palindrome_years_l742_742849


namespace min_value_of_quadratic_l742_742989

def quadratic_function (x : ℝ) : ℝ := x^2 + 6 * x + 13

theorem min_value_of_quadratic :
  (∃ x : ℝ, quadratic_function x = 4) ∧ (∀ y : ℝ, quadratic_function y ≥ 4) :=
sorry

end min_value_of_quadratic_l742_742989


namespace lens_diameter_l742_742256

theorem lens_diameter (C : ℝ) (π : ℝ) (hC : C = 31.4) (hπ : π = 3.14159) : (31.4 / π) ≈ 10 :=
by {
  have h : π ≠ 0 := by linarith,
  rw [hC] at h,
  have d := 31.4 / π,
  show d ≈ 10,
  sorry
}

end lens_diameter_l742_742256


namespace cost_of_12_roll_package_l742_742817

-- Definitions of the given conditions
def individual_roll_price : ℝ := 1
def num_rolls_in_package : ℕ := 12
def savings_percent : ℝ := 0.25

-- Definition of the problem statement to prove
theorem cost_of_12_roll_package :
  let individual_cost := (num_rolls_in_package : ℝ) * individual_roll_price
  let savings_per_roll := savings_percent * individual_roll_price
  let cost_per_roll_in_package := individual_roll_price - savings_per_roll
  let package_cost := num_rolls_in_package * cost_per_roll_in_package
  package_cost = 9 :=
by 
  unfold individual_cost
  unfold savings_per_roll
  unfold cost_per_roll_in_package
  unfold package_cost
  sorry

end cost_of_12_roll_package_l742_742817


namespace berries_difference_l742_742533

-- Define the conditions
def total_berries : ℕ := 900
def Dima_rate : ℕ := 2  -- Dima picks twice as fast
def Sergey_basket_rate : ℕ := 2  -- Sergey puts 2 berries in the basket and eats 1
def Dima_basket_rate : ℕ := 1  -- Dima puts 1 berry in the basket and eats 1

-- Theorem stating the conditions and the conclusion
theorem berries_difference :
  let cycles := total_berries / (Dima_rate + 1 + Sergey_basket_rate + 1) in
  let Dima_berries := cycles * Dima_basket_rate in
  let Sergey_berries := cycles * Sergey_basket_rate in
  Dima_berries = 300 → Sergey_berries = 200 → (Dima_berries - Sergey_berries) = 100 :=
  sorry

end berries_difference_l742_742533


namespace sum_of_n_values_l742_742366

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742366


namespace sum_of_ns_l742_742404

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742404


namespace sum_of_integer_values_l742_742358

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742358


namespace n_plus_one_is_sum_of_p_squares_l742_742419

theorem n_plus_one_is_sum_of_p_squares
  (n : ℕ) (p : ℕ) (hp : p.Prime) 
  (h1 : 1 + n * p = m^2) :
  ∃ (m_1 m_2 ... m_p : ℕ), n + 1 = m_1^2 + m_2^2 + ... + m_p^2 :=
sorry

end n_plus_one_is_sum_of_p_squares_l742_742419


namespace emily_holidays_l742_742920

theorem emily_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (h1 : holidays_per_month = 2) (h2 : months_in_year = 12) :
  holidays_per_month * months_in_year = 24 :=
by
  rw [h1, h2]
  exact Nat.mul_comm 2 12

end emily_holidays_l742_742920


namespace smallest_value_of_expression_l742_742676

noncomputable def g (x : ℝ) : ℝ :=
  x^4 + 10*x^3 + 35*x^2 + 50*x + 24

def is_root (p : ℝ → ℝ) (x : ℝ) : Prop := p x = 0

def roots : Set ℝ := { x | is_root g x }

theorem smallest_value_of_expression :
  let w := {-1, -2, -6, -12} in
  {w_1 w_2, w_3, w_4 | w ⊆ roots ∧ (w_a ∈ w ∧ w_b ∈ w ∧ w_c ∈ w ∧ w_d ∈ w ∧ {w_a, w_b, w_c, w_d} = w)} →
  min (|w_a w_b + w_c w_d|) = 24 :=
  sorry

end smallest_value_of_expression_l742_742676


namespace problem_function_properties_l742_742956

noncomputable def f : ℝ → ℝ := sorry 

theorem problem_function_properties (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f(x) + f(y) = f(x + y))
  (h2 : ∀ x : ℝ, x > 0 → f(x) < 0) : 
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f(x1) > f(x2)) := sorry

end problem_function_properties_l742_742956


namespace tank_total_capacity_l742_742862

theorem tank_total_capacity :
  ∃ x : ℝ, (x / 8) + 90 = x / 2 ∧ x = 240 :=
by
  use 240
  split
  . sorry
  . exact rfl

end tank_total_capacity_l742_742862


namespace roots_abs_gt_4_or_l742_742622

theorem roots_abs_gt_4_or
    (r1 r2 : ℝ)
    (q : ℝ) 
    (h1 : r1 ≠ r2)
    (h2 : r1 + r2 = -q)
    (h3 : r1 * r2 = -10) :
    |r1| > 4 ∨ |r2| > 4 :=
sorry

end roots_abs_gt_4_or_l742_742622


namespace solve_for_x_l742_742191

theorem solve_for_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y)
  (h2 : y = x^2) :
  x = (-1 + Real.sqrt 55) / 3 := 
by
  sorry

end solve_for_x_l742_742191


namespace double_sum_squares_l742_742795

-- Define the sum of squares formula.
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- State the double sum problem.
theorem double_sum_squares :
  ∑ i in Finset.range 200, ∑ j in Finset.range 200, (i^2 + j^2) = 1_070_680_000 :=
by
  sorry

end double_sum_squares_l742_742795


namespace fraction_to_decimal_l742_742114

theorem fraction_to_decimal :
  (7 / 12 : ℝ) ≈ 0.5833 :=
begin
  sorry
end

end fraction_to_decimal_l742_742114


namespace tan_150_eq_neg_sqrt_3_l742_742017

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742017


namespace red_blood_cell_scientific_notation_l742_742292

theorem red_blood_cell_scientific_notation : ∃ n : ℤ, (7.7 * 10^n = 0.0000077) := by
  use -6
  norm_num
  sorry

end red_blood_cell_scientific_notation_l742_742292


namespace solve_system_l742_742766

theorem solve_system : ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 :=
by {
  use 1,
  use 2,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  { refl },
}

end solve_system_l742_742766


namespace sum_of_valid_ns_l742_742347

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l742_742347


namespace tan_150_eq_neg_inv_sqrt3_l742_742064

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l742_742064


namespace sum_of_ns_l742_742403

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l742_742403


namespace perimeter_of_square_D_l742_742731

-- Definitions based on conditions
def perimeter_C : ℝ := 32
def side_len_C : ℝ := perimeter_C / 4
def area_C : ℝ := side_len_C * side_len_C
def area_D : ℝ := area_C / 3
def side_len_D : ℝ := Real.sqrt area_D
def perimeter_D : ℝ := 4 * side_len_D

-- Theorem to prove the perimeter of square D
theorem perimeter_of_square_D : perimeter_D = (32 * Real.sqrt 3) / 3 :=
by
  sorry

end perimeter_of_square_D_l742_742731


namespace total_cost_of_oranges_and_mangoes_l742_742542

theorem total_cost_of_oranges_and_mangoes
  (original_price_orange : ℝ)
  (original_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  original_price_orange = 40 →
  original_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  (quantity_oranges * (original_price_orange * (1 + price_increase_percentage)) +
   quantity_mangoes * (original_price_mango * (1 + price_increase_percentage))
  ) = 1035 :=
begin
  intros,
  sorry
end

end total_cost_of_oranges_and_mangoes_l742_742542


namespace number_of_correct_statements_l742_742532

def statement1 := ∀ (x y : ℝ), x^2 - y^2 = 0 → (x + y = 0 ∨ x - y = 0)
def statement2 := ∀ (l1 l2 : ℝ → ℝ) (p1 p2 : ℝ × ℝ), 
  (l1 p1.1 = p1.2 ∧ l1 p2.1 = p2.2) → |p1.2 - p2.2| = |l2 p1.1 - l2 p2.1| → l1 = l2
def statement3 := ∀ (A B C D E F : ℝ),
  A = D ∧ B = E ∧ (C * (B / 2)) = (F * (E / 2)) → 
    (A = D ∧ B = E ∧ C = F)
def statement4 := ∀ (a b c d e f g : ℝ),
  a + b + c + d + e + f + g = 900 → 
    (a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90)  

def correctStatements := statement4 ∧ ¬statement1 ∧ ¬statement2 ∧ ¬statement3

theorem number_of_correct_statements : (1 = 1) :=
by {
  have h : correctStatements,
  { continue, sorry }, 
  exact rfl 
}

end number_of_correct_statements_l742_742532


namespace trajectory_equation_dot_product_OA_OB_l742_742490

-- Definitions to capture the conditions and problem statements
def point (x y : ℝ) := (x, y)
def distance (p1 p2 : ℝ × ℝ) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

-- Condition: Point P is equidistant from F(1, 0) and line x = -1
def is_equidistant (P : ℝ × ℝ) : Prop :=
  distance P (1, 0) = (P.1 + 1) ^ 2

-- Proposition part (I)
theorem trajectory_equation (x y : ℝ) (h : is_equidistant (x, y)) : y^2 = 4 * x :=
  sorry

-- Proposition part (II)
theorem dot_product_OA_OB (M : ℝ × ℝ) (parabola : ℝ × ℝ → Prop) (h1 : M = (4,0))
  (A B : ℝ × ℝ) (h2 : parabola A) (h3 : parabola B) :
  let OA := A in
  let OB := B in
  OA.1 * OB.1 + OA.2 * OB.2 = 0 :=
  sorry

end trajectory_equation_dot_product_OA_OB_l742_742490


namespace smallest_k_l742_742764

open Set

def X : Set Nat := {x | 1 ≤ x ∧ x ≤ 100}

def f (f : Nat → Nat) : Prop :=
  ∀ x ∈ X, f x ≠ x ∧ (∀ A ⊆ X, (card A = 40) → (A ∩ f '' A ≠ ∅))

theorem smallest_k (f : Nat → Nat) (hf : f f) : ∃ k = 69, ∃ B ⊆ X, (card B = k) ∧ (B ∪ (f '' B) = X) :=
sorry

end smallest_k_l742_742764


namespace solve_problem_l742_742562

def spadesuit (x y : ℝ) : ℝ := x^2 + y^2

theorem solve_problem : spadesuit (spadesuit 3 5) 4 = 1172 := by
  sorry

end solve_problem_l742_742562


namespace distance_from_P_to_origin_l742_742193

open Real -- This makes it easier to use real number functions and constants.

noncomputable def hyperbola := { P : ℝ × ℝ // (P.1^2 / 9) - (P.2^2 / 7) = 1 }

theorem distance_from_P_to_origin 
  (P : ℝ × ℝ) 
  (hP : (P.1^2 / 9) - (P.2^2 / 7) = 1)
  (d_right_focus : P.1 - 4 = -1) : 
  dist P (0, 0) = 3 :=
sorry

end distance_from_P_to_origin_l742_742193


namespace triangle_angle_BAE_l742_742637

theorem triangle_angle_BAE (A B C E : Type*) [IsTriangle A B C]
  (CE_eq_BC : dist C E = dist B C)
  (angle_BCE : angle B C E = 50) :
  angle B A E = 100 :=
begin
  sorry
end

end triangle_angle_BAE_l742_742637


namespace tan_150_eq_l742_742002

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l742_742002


namespace binom_sum_l742_742384

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l742_742384


namespace cannot_form_perfect_square_l742_742308

theorem cannot_form_perfect_square :
  let sum_of_squares := (List.range 1982).map (λ n => (n + 1) ^ 2) in
  (sum_of_squares.sum % 3 ≠ 0) ∧ (sum_of_squares.sum % 3 ≠ 1) :=
by
  let squares := (List.range 1982).map (λ n => (n + 1) ^ 2)
  have : squares.sum % 3 = 2 := sorry
  exact ⟨this.symm.ne, this.symm.ne⟩

end cannot_form_perfect_square_l742_742308


namespace a_n_formula_b_n_formula_l742_742152

namespace SequenceFormulas

theorem a_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ S : ℕ → ℕ, S n = 2 * n^2 + 2 * n) → ∃ a : ℕ → ℕ, a n = 4 * n :=
by
  sorry

theorem b_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ T : ℕ → ℕ, T n = 2 - (if n > 1 then T (n-1) else 1)) → ∃ b : ℕ → ℝ, b n = (1/2)^(n-1) :=
by
  sorry

end SequenceFormulas


end a_n_formula_b_n_formula_l742_742152


namespace interval_solution_length_l742_742529

theorem interval_solution_length (a b : ℝ) (h : (b - a) / 3 = 8) : b - a = 24 := by
  sorry

end interval_solution_length_l742_742529


namespace probability_no_hats_left_l742_742125

noncomputable def harmonic (n : ℕ) : ℝ :=
  ∑ k in Finset.range n + 1, 1 / (k : ℝ)

noncomputable def p (n : ℕ) : ℝ :=
  if n = 1 then 1
  else (harmonic n / n) * p (n - 1)

theorem probability_no_hats_left :
  abs (p 10 - 0.000516) < 0.0001 := sorry

end probability_no_hats_left_l742_742125


namespace cartesian_eq_of_polar_max_area_triangle_ABP_l742_742215

-- Definitions and conditions
def polar_eq (θ : ℝ) : ℝ := 144 / (9 + 7 * (Real.sin θ)^2)
def to_cartesian_eq (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

-- Proof problem 1: Cartesian equation
theorem cartesian_eq_of_polar :
  ∀ (θ : ℝ), (to_cartesian_eq (polar_eq θ * Real.cos θ) (polar_eq θ * Real.sin θ)) := sorry

-- Definitions and conditions for problem 2
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, 3)
def curve_C : ℝ × ℝ → Prop := λ p, to_cartesian_eq p.1 p.2
def point_P (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 3 * Real.sin θ)
def area_triangle (A B P : ℝ × ℝ) : ℝ :=
  0.5 * Real.abs (A.1 * B.2 + B.1 * P.2 + P.1 * A.2 - B.1 * A.2 - P.1 * B.2 - A.1 * P.2)

-- Proof problem 2: Maximum area of triangle ABP
theorem max_area_triangle_ABP :
  ∀ (θ : ℝ), (curve_C (point_P θ)) → area_triangle point_A point_B (point_P θ) ≤ 6 * (Real.sqrt 2 + 1) := sorry

end cartesian_eq_of_polar_max_area_triangle_ABP_l742_742215


namespace sum_of_integer_values_l742_742355

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l742_742355


namespace count_n_equals_3_count_n_even_l742_742179

open Finset

def M : Finset ℕ := {1, 2, 3, 4}

def cumulative_value (A : Finset ℕ) : ℕ :=
  if A.card = 0 then 0
  else A.prod id

def count_subsets_with_n (n : ℕ) (f : Finset ℕ → ℕ) (M : Finset ℕ) : ℕ :=
  M.powerset.count (λ A, f A = n)

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem count_n_equals_3 :
  count_subsets_with_n 3 cumulative_value M = 2 := 
sorry

theorem count_n_even :
  count_subsets_with_n (even ∘ cumulative_value) M = 13 :=
sorry

end count_n_equals_3_count_n_even_l742_742179


namespace tan_150_eq_neg_sqrt_3_l742_742018

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l742_742018


namespace exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l742_742271

theorem exists_n_such_that_5_pow_n_has_six_consecutive_zeros :
  ∃ n : ℕ, n < 1000000 ∧ ∃ k : ℕ, k = 20 ∧ 5 ^ n % (10 ^ k) < (10 ^ (k - 6)) :=
by
  -- proof goes here
  sorry

end exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l742_742271


namespace circle_area_of_equilateral_triangle_l742_742441

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l742_742441


namespace tan_150_eq_neg_sqrt3_div_3_l742_742106

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742106


namespace greatest_possible_avg_purity_l742_742203

def average_purity_YZ (X Y Z : Type) [has_units X] [has_units Y] [has_units Z] 
  (X_purity : X -> Nat) (Y_purity : Y -> Nat) (Z_purity : Z -> Nat) : Nat :=
sorry

theorem greatest_possible_avg_purity (X Y Z : Type) [has_units X] [has_units Y] [has_units Z]
  (X_purity : X -> Nat) (Y_purity : Y -> Nat) (Z_purity : Z -> Nat)
  (X_avg : Nat) (Y_avg : Nat) (XY_avg: Nat) (XZ_avg: Nat) (YZ_avg : Nat) :
  X_avg = 30 -> 
  Y_avg = 60 -> 
  XY_avg = 50 -> 
  XZ_avg = 45 ->
  YZ_avg = average_purity_YZ X Y Z X_purity Y_purity Z_purity ->
  YZ_avg = 65 := 
sorry

end greatest_possible_avg_purity_l742_742203


namespace inequality_problem_l742_742684

theorem inequality_problem
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
sorry

end inequality_problem_l742_742684


namespace tan_150_eq_neg_inv_sqrt_3_l742_742098

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742098


namespace sum_of_valid_n_l742_742340

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l742_742340


namespace mary_mac_download_time_l742_742697

theorem mary_mac_download_time (x : ℕ) (windows_download : ℕ) (total_glitch : ℕ) (time_without_glitches : ℕ) (total_time : ℕ) :
  windows_download = 3 * x ∧
  total_glitch = 14 ∧
  time_without_glitches = 2 * total_glitch ∧
  total_time = 82 ∧
  x + windows_download + total_glitch + time_without_glitches = total_time →
  x = 10 :=
by 
  sorry

end mary_mac_download_time_l742_742697


namespace incorrect_statements_l742_742672

-- Definitions of the sets M and N
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3) = 0}
def N : Set ℝ := {x | (x - 4) * (x - 1) = 0}

-- Statement A
def statement_A (a : ℝ) : Prop :=
  (M a ∪ N).finite ∧ (M a ∪ N).toFinset.card = 4 → (M a ∩ N ≠ ∅)

-- Statement B
def statement_B (a : ℝ) : Prop :=
  M a ∩ N ≠ ∅ → (M a ∪ N).finite ∧ (M a ∪ N).toFinset.card = 4

-- Statement C
def statement_C (a : ℝ) : Prop :=
  (M a ∪ N) = {1, 3, 4} → M a ∩ N ≠ ∅

-- Statement D
def statement_D (a : ℝ) : Prop :=
  M a ∩ N ≠ ∅ → (M a ∪ N) = {1, 3, 4}

-- The final theorem
theorem incorrect_statements (a : ℝ) : statement_A a → statement_B a → statement_C a :=
begin
  sorry -- Proof is not required
end

end incorrect_statements_l742_742672


namespace cone_height_is_correct_l742_742819

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l742_742819


namespace keith_turnips_l742_742233

theorem keith_turnips (Alyssa_turnips Keith_turnips : ℕ) 
  (total_turnips : Alyssa_turnips + Keith_turnips = 15) 
  (alyssa_grew : Alyssa_turnips = 9) : Keith_turnips = 6 :=
by
  sorry

end keith_turnips_l742_742233


namespace find_general_term_l742_742688

-- Define the function f
-- We are given that f(0) = 1/2 and f(1) = n^2 * a_n for n in Nat

noncomputable def f (a : Nat → ℚ) (x : ℚ) (n : Nat) : ℚ := 
  finset.sum (Finset.range n) (λ i, (a i) * (x ^ i))

-- Condition: f(0) = 1/2
def f_at_0 (a : Nat → ℚ) (n : Nat) : Prop := 
  f a 0 n = 1/2

-- Condition: f(1) = n^2 * a_n
def f_at_1 (a : Nat → ℚ) (n : Nat) : Prop := 
  f a 1 n = n^2 * (a n)

-- Define the statement to be proved
theorem find_general_term (a : Nat → ℚ) (n : Nat) (hn : 0 < n) :
  (f_at_0 a n) → (f_at_1 a n) → (a n = 1 / ((n + 1) * n)) :=
by
  intros h0 h1
  sorry

end find_general_term_l742_742688


namespace tan_150_deg_l742_742040

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l742_742040


namespace tan_150_deg_l742_742022

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l742_742022


namespace sum_of_four_digit_numbers_l742_742557

theorem sum_of_four_digit_numbers :
  let digits := {1, 2, 4, 5, 7, 8};
  let valid_number (n : ℕ) := ∀ d ∈ Nat.digits 10 n, d ∈ digits;
  let four_digit_numbers := {n | 1000 ≤ n ∧ n < 10000 ∧ valid_number n};
  ∑ n in four_digit_numbers, n = 6479352 :=
by
  let digits := {1, 2, 4, 5, 7, 8};
  let valid_digit := λ d, d ∈ digits;
  let valid_number := λ n, ∀ d ∈ Nat.digits 10 n, valid_digit d;
  let four_digit_numbers := {n | 1000 ≤ n ∧ n < 10000 ∧ valid_number n};
  -- Proof would go here
  sorry

end sum_of_four_digit_numbers_l742_742557


namespace total_sticks_needed_l742_742716

theorem total_sticks_needed :
  let simon_sticks := 36
  let gerry_sticks := 2 * (simon_sticks / 3)
  let total_simon_and_gerry := simon_sticks + gerry_sticks
  let micky_sticks := total_simon_and_gerry + 9
  total_simon_and_gerry + micky_sticks = 129 :=
by
  sorry

end total_sticks_needed_l742_742716


namespace tan_150_eq_neg_inv_sqrt_3_l742_742090

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742090


namespace tan_150_eq_neg_sqrt3_div_3_l742_742101

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l742_742101


namespace sum_sequence_eq_l742_742559

-- Define the sequence using the given recurrence relation and initial term
noncomputable def sequence (n : ℕ) : ℕ → ℚ
| 0 => 0  -- This is a placeholder for x_0 which we will not use since our index starts from 1
| 1 => 1 / 3
| (k + 1) => sequence k + 1 / 3

-- Define the sum of the sequence from 1 to n
def sum_sequence (n : ℕ) : ℚ :=
∑ k in Finset.range (n + 1), if k = 0 then 0 else sequence k

-- The theorem stating the sum of the first n terms
theorem sum_sequence_eq (n : ℕ) : 
  sum_sequence n = (n * (n + 1)) / 6 := 
sorry

end sum_sequence_eq_l742_742559


namespace exists_divisible_subset_l742_742237

theorem exists_divisible_subset 
  (n : ℕ) (hn : n ≥ 1) 
  (X : Finset ℕ) (hX : X.card = n^2 + 1)
  (hx : ∀ (S : Finset ℕ), S.card = n + 1 → (∃ x y ∈ S, x ≠ y ∧ x ∣ y)) : 
  ∃ (subset : Finset ℕ), subset.card = n + 1 ∧ (∀ (x y ∈ subset), x ≠ y → x ∣ y ∨ y ∣ x) :=
sorry

end exists_divisible_subset_l742_742237


namespace max_irrationals_l742_742739

open Set

noncomputable def maximum_irrationals_in_first_row (top_row : Fin 2019 → ℝ) (bottom_row : Fin 2019 → ℝ) : Prop :=
  (∀ i j : Fin 2019, i ≠ j → top_row i ≠ top_row j) ∧ -- distinct elements in top row
  (∀ i : Fin 2019, bottom_row i ∈ Set.range top_row) ∧ -- bottom row is a permutation of top row
  (∀ i : Fin 2019, top_row i + bottom_row i ∈ ℚ) ∧ -- sum in each column is rational
  -- proving maximum irrational numbers in top row
  (#(Set.filter (λ x, ¬ x ∈ ℚ) (Set.range top_row)) = 2016)

theorem max_irrationals (top_row : Fin 2019 → ℝ) (bottom_row : Fin 2019 → ℝ) :
  maximum_irrationals_in_first_row top_row bottom_row :=
sorry

end max_irrationals_l742_742739


namespace calculate_total_cost_l742_742537

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end calculate_total_cost_l742_742537


namespace tan_150_eq_neg_inv_sqrt_3_l742_742089

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l742_742089


namespace arithmetic_seq_first_term_l742_742249

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (n : ℕ) (a : ℚ)
  (h₁ : ∀ n, S n = n * (2 * a + (n - 1) * 5) / 2)
  (h₂ : ∀ n, S (3 * n) / S n = 9) :
  a = 5 / 2 :=
by
  sorry

end arithmetic_seq_first_term_l742_742249


namespace ellipse_equation_sum_of_m_n_constant_min_area_triangle_QAB_l742_742968

-- Conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def M : ℝ × ℝ := (0, 2)
def F : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (0, 2)

-- Proof Problem Statement
theorem ellipse_equation : ∃ a b, b = 2 ∧ F = (2, 0) ∧ M = (0, 2) →
  ∀ x y, ellipse a b x y ↔ x^2 / 8 + y^2 / 4 = 1 :=
sorry

theorem sum_of_m_n_constant (m n : ℝ) : 
  (m ≠ n) ∧ 
  ∀ t, (x₁ x₂ : ℝ), (y₁ y₂ : ℝ), ((x₁, y₁, t), (x₂, y₂, t)) = (m, n) →
  2m^2 + 8m - t^2 + 4 = 0 ∧ 2n^2 + 8n - t^2 + 4 = 0 → m + n = -4 :=
sorry

theorem min_area_triangle_QAB : 
  ∀ P Q A B : ℝ × ℝ, let k (P Q : ℝ) := true in 
  l P Q = false ∧ (l P Q) = true →
  A ≠ B ∧ P ≠ (0, 2k) ∧ P ≠ 2 ↔
  (area_QAB = 4 * sqrt(2) * sqrt(1 - 1 / (4 * (k^2 + 1/2)^2))) 
  → min_area_QAB = (16 / 3) :=
sorry

end ellipse_equation_sum_of_m_n_constant_min_area_triangle_QAB_l742_742968


namespace question1_question2_question3_l742_742157

noncomputable def ellipse_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

def foci (a : ℝ) : Prop := 
  ∃ (c : ℝ), (c = real.sqrt (a^2 - 3))

def condition1 (a : ℝ) (ha : 0 < a) : Prop :=
  4 * a = 8

def condition2 (m c a : ℝ) (h : ∃ (m : ℝ), c = 2 * m) : Prop :=
  2 * m + 3 * m = 2 * a

theorem question1 (a b : ℝ) (ha0 : 0 < a) (hb0 : 0 < b) 
  (h_cond1 : condition1 a ha0)
  (h_cond2 : ∃ c, c = real.sqrt (a^2 - b^2))
  (h_foci : foci a) :
  ellipse_eq 2 (real.sqrt 3) (by norm_num) (by norm_num) := sorry

def circle_eq (b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x^2 + y^2 = b^2)

theorem question2 (a b : ℝ) (x0 y0 : ℝ) (h_ellipse : ∃ x y, x^2 / 4 + y^2 / 3 = 1) 
  (h_tangent : circle_eq b) :
  (2 + x0 / 2 - x0 / 2 = 2) := sorry

theorem question3 (m : ℝ) (hm : m ≠ real.sqrt 3 ∧ m ≠ -real.sqrt 3) :
  λ + μ = (6 / (3 - m^2)) :=
  let y1 := (some y2 : ℝ),
      y2 := (some y2 : ℝ),
      k := (some k : ℝ), 
      QC := λ (x y : ℝ), (x + k * y),
      QD := λ (x y : ℝ), (x - k * y),
      λ := 1 + m / (y1 - m),
      μ := 1 + m / (y2 - m) in
  sorry

end question1_question2_question3_l742_742157


namespace sum_of_n_values_l742_742363

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l742_742363
