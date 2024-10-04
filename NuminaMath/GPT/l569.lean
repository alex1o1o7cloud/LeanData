import Mathlib
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Log
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Analysis.Calculus.Continuous
import Mathlib.Analysis.Calculus.IteratedDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Analysis.Trigonometry.Triangle
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialProofs
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.RingDivision
import Mathlib.Data.Prob
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Algebra.Order
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic
import analysis.special_functions.trigonometric
import data.real.basic

namespace number_of_true_statements_is_3_l569_569159

open Set

def p : Prop := {2} ∈ ({1, 2, 3} : Set ℕ)
def q : Prop := {2} ⊆ ({1, 2, 3} : Set ℕ)

theorem number_of_true_statements_is_3 : 
  (p ∨ q) = True ∧ -- ①
  (p ∨ q) = False ∧ -- ②
  (p ∧ q) = False ∧ -- ③
  (¬p ∧ ¬q) = False ∧ -- ④
  (¬p) = True ∧ -- ⑤
  (¬q) = False → -- ⑥
  3 = 3 :=
by
  sorry

end number_of_true_statements_is_3_l569_569159


namespace distance_from_center_to_face_l569_569550

-- Definitions of sphere radius and conditions of the tetrahedron
def r : ℝ := (Real.sqrt 3) / 2 
def is_regular_tetrahedron (A B C D : ℝ×ℝ×ℝ) : Prop := 
  ∀p1 p2 p3 p4 ∈ {A, B, C, D}, 
    p1 ≠ p2 → 
    p1 ≠ p3 → 
    p1 ≠ p4 → 
    Euclidean.distance p1 p2 = Euclidean.distance p1 p3

-- Definition of the main theorem to prove
theorem distance_from_center_to_face {A B C D : ℝ×ℝ×ℝ} (h : is_regular_tetrahedron A B C D) : 
  ∃ O : ℝ×ℝ×ℝ, 
    (Euclidean.distance O A = r ∧ Euclidean.distance O B = r ∧ Euclidean.distance O C = r ∧ Euclidean.distance O D = r) → 
    ∀ face, 
      face ∈ {{A, B, C}, {A, B, D}, {A, C, D}, {B, C, D}} → 
      Euclidean.distance O 
      (face_center face) = (Real.sqrt 3) / 6 := 
sorry

end distance_from_center_to_face_l569_569550


namespace train_speed_l569_569620

theorem train_speed :
  ∀ (train_length bridge_length time_seconds : ℕ) (total_length : ℕ),
  train_length = 130 →
  total_length = 245 →
  bridge_length = 245 →
  time_seconds = 30 →
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 := 
by
  intros train_length bridge_length time_seconds total_length
  sorry

end train_speed_l569_569620


namespace sin_F_right_triangle_l569_569788

theorem sin_F_right_triangle
  (DE EF DF : ℝ)
  (h1 : DE = 9)
  (h2 : EF = 15)
  (h3 : DF = Real.sqrt (DE^2 + EF^2)) :
  sin (Real.arctan (DE / EF)) = 3 * Real.sqrt 34 / 34 :=
by
  -- conditions
  have hDE : DE = 9 := h1,
  have hEF : EF = 15 := h2,
  have hDF : DF = Real.sqrt (DE^2 + EF^2) := h3,

  -- goal
  sorry

end sin_F_right_triangle_l569_569788


namespace find_N_l569_569400

theorem find_N :
  let N := 25
  in 1001 + 1003 + 1005 + 1007 + 1009 = 5050 - N :=
by
  let N := 25
  sorry

end find_N_l569_569400


namespace rectangle_area_l569_569587

theorem rectangle_area (b l : ℕ) (P : ℕ) (h1 : l = 3 * b) (h2 : P = 64) (h3 : P = 2 * (l + b)) :
  l * b = 192 :=
by
  sorry

end rectangle_area_l569_569587


namespace geometric_sequence_general_term_l569_569436

theorem geometric_sequence_general_term (n : ℕ) (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) 
  (h1 : a1 = 4) (h2 : q = 3) (h3 : ∀ n, a n = a1 * (q ^ (n - 1))) :
  a n = 4 * 3^(n - 1) := by
  sorry

end geometric_sequence_general_term_l569_569436


namespace num_valid_subsets_is_2380_l569_569203

def S : Finset ℕ := Finset.range 20
def is_valid_subset (T : Finset ℕ) : Prop :=
  T.card = 4 ∧ ∀ (x y : ℕ), x ∈ T → y ∈ T → x ≠ y → |x - y| ≠ 1

theorem num_valid_subsets_is_2380 :
  (Finset.filter is_valid_subset (Finset.powersetLen 4 S)).card = 2380 := 
sorry

end num_valid_subsets_is_2380_l569_569203


namespace blender_sales_inversely_proportional_l569_569290

theorem blender_sales_inversely_proportional (k : ℝ) (p : ℝ) (c : ℝ) 
  (h1 : p * c = k) (h2 : 10 * 300 = k) : (p * 600 = k) → p = 5 := 
by
  intros
  sorry

end blender_sales_inversely_proportional_l569_569290


namespace wooden_block_length_l569_569879

theorem wooden_block_length :
  ∀ (meters_to_cm : ℕ → ℕ) (block_length_m : ℕ) (difference_cm : ℕ) (final_length_cm : ℕ),
    (∀ x, meters_to_cm x = x * 100) →
    (block_length_m = 31) →
    (difference_cm = 30) →
    (final_length_cm = meters_to_cm block_length_m - difference_cm) →
    final_length_cm = 3070 :=
by {
  intro meters_to_cm block_length_m difference_cm final_length_cm,
  intros h1 h2 h3 h4,
  sorry
}

end wooden_block_length_l569_569879


namespace electrons_transferred_in_reaction_l569_569904

theorem electrons_transferred_in_reaction :
  ∃ (e : ℕ), e = 5 ∧ (∀ k m : ℕ, reaction_eq k m → KClO3 + 6 * HCl = KCl + 3 * Cl2 + 3 * H2O) :=
sorry

def reaction_eq (k m : ℕ) : Prop :=
  k = 1 ∧ m = 1

end electrons_transferred_in_reaction_l569_569904


namespace minimum_value_of_f_l569_569882

def f (x : ℝ) : ℝ := (Real.sin (x - Real.pi / 2)) * (Real.cos x) - (Real.cos (2 * x))^2

theorem minimum_value_of_f : ∃ x : ℝ, f x = -2 :=
sorry

end minimum_value_of_f_l569_569882


namespace parabola_tangent_directrix_l569_569772

theorem parabola_tangent_directrix (a : ℝ) :
  let directrix_parabola := (λ y, -a / 4)
  let second_parabola := (λ x, -(x + 1) ^ 2 + 2)
  (∃ y x, directrix_parabola y ≤ second_parabola x) → a = -8 :=
  sorry

end parabola_tangent_directrix_l569_569772


namespace candidate_a_votes_l569_569784

theorem candidate_a_votes (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_a_percentage : ℝ)
    (h_total_votes : total_votes = 560000)
    (h_invalid_percentage : invalid_percentage = 0.15)
    (h_candidate_a_percentage : candidate_a_percentage = 0.85) :
    let valid_votes := ((1 - invalid_percentage) * total_votes : ℕ) in
    let candidate_a_votes := (candidate_a_percentage * valid_votes : ℕ) in
    candidate_a_votes = 404600 :=
by
  -- Valid votes calculation
  let valid_votes := (1 - invalid_percentage) * total_votes
  have h_valid_votes : valid_votes = 476000 := by norm_num [h_total_votes, h_invalid_percentage]
  -- Candidate A votes calculation
  let candidate_a_votes := candidate_a_percentage * valid_votes
  have h_candidate_a_votes : candidate_a_votes = 404600 := by norm_num [h_valid_votes, h_candidate_a_percentage]
  exact h_candidate_a_votes

end candidate_a_votes_l569_569784


namespace find_x_correct_l569_569232

theorem find_x_correct (x : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * 180 = 360 * x - 480)
  (h2 : (x + 4) + x + (x - 4) = 3 * x)
  (h3 : 100 = (360 * x - 480) / (3 * x)) : 
  x = 8 := 
sorry

end find_x_correct_l569_569232


namespace train_speed_l569_569621

theorem train_speed :
  ∀ (train_length bridge_length time_seconds : ℕ) (total_length : ℕ),
  train_length = 130 →
  total_length = 245 →
  bridge_length = 245 →
  time_seconds = 30 →
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 := 
by
  intros train_length bridge_length time_seconds total_length
  sorry

end train_speed_l569_569621


namespace total_cost_of_books_l569_569399

theorem total_cost_of_books (C1 C2 : ℝ) 
  (hC1 : C1 = 268.33)
  (h_selling_prices_equal : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 459.15 :=
by
  -- placeholder for the proof
  sorry

end total_cost_of_books_l569_569399


namespace average_scores_proof_l569_569420

noncomputable def average_score
  (madeline_mistakes : ℕ)
  (leo_mistakes : ℕ)
  (brent_mistakes : ℕ)
  (brent_score : ℕ)
  (nicholas_mistakes : ℕ)
  (nicholas_score : ℕ)
  (perfect_score : ℕ)
  (deduction_per_mistake : ℕ) : ℝ :=
  let madeline_score := perfect_score - (madeline_mistakes * deduction_per_mistake)
  let leo_score := perfect_score - (leo_mistakes * deduction_per_mistake)
  average := (madeline_score + leo_score + brent_score + nicholas_score) / 4.0
  average

theorem average_scores_proof : 
  ∀ (madeline_mistakes : ℕ)
    (leo_mistakes : ℕ)
    (brent_mistakes : ℕ)
    (brent_score : ℕ)
    (nicholas_mistakes : ℕ)
    (nicholas_score : ℕ)
    (perfect_score : ℕ)
    (deduction_per_mistake : ℕ),
  madeline_mistakes = 2 →
  leo_mistakes = 2 * madeline_mistakes →
  brent_mistakes = leo_mistakes + 1 →
  brent_score = perfect_score - (brent_mistakes * deduction_per_mistake) →
  nicholas_mistakes = 3 * madeline_mistakes →
  nicholas_score = brent_score - 5 →
  average_score madeline_mistakes leo_mistakes brent_mistakes brent_score nicholas_mistakes nicholas_score perfect_score deduction_per_mistake = 20.75 :=
by
  intros madeline_mistakes leo_mistakes brent_mistakes brent_score nicholas_mistakes nicholas_score perfect_score deduction_per_mistake
  intros h1 h2 h3 h4 h5 h6
  rw [h1] at *
  rw [h2] at *
  rw [h3] at *
  rw [h4] at *
  rw [h5] at *
  rw [h6] at *
  norm_num
  sorry

end average_scores_proof_l569_569420


namespace problem_l569_569368

noncomputable def f(x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1
noncomputable def f_prime(x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + b

theorem problem (a b : ℝ) 
  (h₁ : f_prime 1 a b = 4) 
  (h₂ : f 1 a b = 3) : 
  a + b = 2 :=
sorry

end problem_l569_569368


namespace first_term_proof_l569_569278

noncomputable def first_term_of_geometric_series (r : ℚ) (S : ℚ) : ℚ :=
  let a := S * (1 - r)
  in 3 * a / 4  -- simplifying the step directly

theorem first_term_proof :
  first_term_of_geometric_series (-1 / 3) 27 = 36 :=
by
  -- We define necessary geometric series sum definition based on the conditions
  let a := first_term_of_geometric_series (-1 / 3) 27 
  change a = 36
  -- This change asserts what we conclude earlier in plain form
  sorry

end first_term_proof_l569_569278


namespace area_trapezoid_DBCE_l569_569104

-- Geometry definitions and conditions
variables (ABC : Type*) [triangle ABC]
variables (AB AC BC : ABC)
variables (D B C E : ABC)
variables (similar_to : (Triangle ABC) → (Triangle DBC) → Prop)
variables (isosceles : ∀ {T : Type*} [triangle T], AB = AC)
variables (num_smallest_triangles : nat := 9)
variables (area_smallest_triangle : nat := 2)
variables (area_ABC : nat := 72)

-- Given conditions
axiom h1 : num_smallest_triangles = 9
axiom h2 : area_smallest_triangle = 2
axiom h3 : area_ABC = 72

-- Prove the area of trapezoid DBCE is 64
theorem area_trapezoid_DBCE : trapezoid_area DBCE = 64 :=
sorry

end area_trapezoid_DBCE_l569_569104


namespace flag_covering_proof_l569_569241

def grid_covering_flag_ways (m n num_flags cells_per_flag : ℕ) :=
  if m * n / cells_per_flag = num_flags then 2^num_flags else 0

theorem flag_covering_proof :
  grid_covering_flag_ways 9 18 18 9 = 262144 := by
  sorry

end flag_covering_proof_l569_569241


namespace line_through_intersections_of_circles_l569_569376

-- Define the first circle
def circle₁ (x y : ℝ) : Prop :=
  x^2 + y^2 = 10

-- Define the second circle
def circle₂ (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 20

-- The statement of the mathematically equivalent proof problem
theorem line_through_intersections_of_circles : 
    (∃ (x y : ℝ), circle₁ x y ∧ circle₂ x y) → (∃ (x y : ℝ), x + 3 * y - 5 = 0) :=
by
  intro h
  sorry

end line_through_intersections_of_circles_l569_569376


namespace question1_question2_l569_569748

noncomputable def omega := 2

def y (ω x : ℝ) : ℝ := sin (ω * x) * cos (ω * x)

theorem question1 (h : ω = 2) : (∃ T, T = π / ω) :=
by
  use π / ω
  sorry

theorem question2 (h : ω = 2) (x : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ (π / (2 * ω))) : 
  (∃ ymax ymin : ℝ, ymax = 1 ∧ ymin = 0 
    ∧ (x = 0 → y ω x = ymax)
    ∧ (x = π / (4 * ω) → y ω x = ymin)) :=
by
  use 1, 0
  sorry

end question1_question2_l569_569748


namespace common_point_circumcircles_l569_569711

/-- Problem Setup -/
structure Quadrilateral :=
(A B C D E F S T : Type)
[convex : conv_quad A B C D]
[on_line_AD : E ∈ segment A D]
[on_line_BC : F ∈ segment B C]
(ratio_equal : ∀ (a e d b f c: A), (AE / ED = BF / FC))
(intersect_FE_BA_CD : ∀ (fe ba cd: A), (FE ∩ BA = S ∧ FE ∩ CD = T))

/-- Proof that the circumcircles of triangles SAE, SBF, TCF, and TDE have a common point -/
theorem common_point_circumcircles (Q: Quadrilateral) :
  ∃ (P : Type), is_circumcenter P Q.S A E ∧ is_circumcenter P Q.S B F ∧
                is_circumcenter P Q.T C F ∧ is_circumcenter P Q.T D E :=
sorry

end common_point_circumcircles_l569_569711


namespace min_value_ellipse_l569_569192

theorem min_value_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (∃ c, c = a / 2 ∧ c^2 = a^2 - b^2) ∧ ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) 
  : ∃ M : ℝ, M = (4 * Real.sqrt 3) / 3 ∧  (∀ a > b > 0, (a^2 + 1) / b ≥ M) :=
begin
  sorry
end

end min_value_ellipse_l569_569192


namespace angle_sum_l569_569246

-- Define the conditions of the problem
variables (ABCD : Quadrilateral) 
          (C : Point) 
          (A B D : Point)
          (h_circum : Circle.circumscribed ABCD)
          (h_ACB : angle A C B = 50)
          (h_CAD : angle C A D = 20)

-- Main statement to be proven
theorem angle_sum (h₁ : angle A C B = 50) (h₂ : angle C A D = 20) : 
  angle C A B + angle A C D = 110 := 
sorry

end angle_sum_l569_569246


namespace remainder_of_98_mult_102_div_12_l569_569920

theorem remainder_of_98_mult_102_div_12 : (98 * 102) % 12 = 0 := by
    sorry

end remainder_of_98_mult_102_div_12_l569_569920


namespace cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l569_569484

-- Definitions based on conditions
def distanceAB := 18  -- km
def speedCarA := 54   -- km/h
def speedCarB := 36   -- km/h
def targetDistance := 45  -- km

-- Proof problem statements
theorem cars_towards_each_other {y : ℝ} : 54 * y + 36 * y = 18 + 45 ↔ y = 0.7 :=
by sorry

theorem cars_same_direction_A_to_B {x : ℝ} : 54 * x - (36 * x + 18) = 45 ↔ x = 3.5 :=
by sorry

theorem cars_same_direction_B_to_A {x : ℝ} : 54 * x + 18 - 36 * x = 45 ↔ x = 1.5 :=
by sorry

end cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l569_569484


namespace number_pattern_is_square_l569_569917

theorem number_pattern_is_square (n : ℕ) : 
  let x := (10^n - 1) / 9 in
  let result := 4 * x * (10^n) + 8 * ((10^(n-1) - 1) / 9) + 9 in
  ∃ k, result = k^2 :=
by
  sorry

end number_pattern_is_square_l569_569917


namespace total_students_registered_l569_569862

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end total_students_registered_l569_569862


namespace min_possible_largest_divisible_by_4_l569_569216

theorem min_possible_largest_divisible_by_4 (a b c : ℕ) :
  (∃ (digits : Finset ℕ), 
    (∀ (x ∈ digits), x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∀ i j : ℕ, i ∈ digits → j ∈ digits → i ≠ j) ∧ 
    (digits.card = 9) ∧ 
    (a ∈ {x ∈ digits | x % 4 = 0}) ∧ 
    (b ∈ {x ∈ digits | x % 4 = 0}) ∧ 
    (c ∈ {x ∈ digits | x % 4 = 0})) → 
  max a (max b c) = 896 :=
by sorry

end min_possible_largest_divisible_by_4_l569_569216


namespace quadrilateral_area_l569_569256

structure Point :=
  (x : ℝ)
  (y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_quadrilateral (A B C D : Point) : ℝ :=
  area_of_triangle A B C + area_of_triangle A C D

def A : Point := ⟨2, 2⟩
def B : Point := ⟨2, -1⟩
def C : Point := ⟨3, -1⟩
def D : Point := ⟨2007, 2008⟩

theorem quadrilateral_area :
  area_of_quadrilateral A B C D = 2008006.5 :=
by
  sorry

end quadrilateral_area_l569_569256


namespace product_of_positive_real_solutions_l569_569412

noncomputable def cis := complex.exp

theorem product_of_positive_real_solutions :
  (∏ (x : ℂ) in { x | x^8 = -256 ∧ x.re > 0 }, x) = 4 * real.sqrt 2 :=
by
  sorry

end product_of_positive_real_solutions_l569_569412


namespace max_sine_cosine_sum_l569_569180

theorem max_sine_cosine_sum (α β γ : ℝ) (h1 : α + β + γ = π) (h2 : 0 ≤ α ∧ α ≤ π) (h3 : 0 ≤ β ∧ β ≤ π) (h4 : 0 ≤ γ ∧ γ ≤ π) :
  (∀ (α β γ : ℝ), α + β + γ = π → 0 ≤ α ∧ α ≤ π → 0 ≤ β ∧ β ≤ π → 0 ≤ γ ∧ γ ≤ π →
  ∀ x, (sin x ≤ 1 ∧ abs (cos x) ≤ 1) → sin α * sin β * cos γ + sin γ * sin γ ≤ 9 / 8) :=
by
  intros α β γ h1 h2 h3 h4 x hx,
  sorry

end max_sine_cosine_sum_l569_569180


namespace smaller_circle_radius_l569_569797

theorem smaller_circle_radius {R : ℝ} (h : R = 10) : 
  let r := 10 / 3 in 
  ∃ (r : ℝ), 
  (20 = 6 * r) :=
by
  sorry

end smaller_circle_radius_l569_569797


namespace alternating_sum_of_squares_100_l569_569149

theorem alternating_sum_of_squares_100 :
  ∑ k in finset.range 50, (4 * k + 3) = 5050 := sorry

end alternating_sum_of_squares_100_l569_569149


namespace sin_alpha_value_l569_569704

theorem sin_alpha_value (α : ℝ) (h1 : tan α = 1 / 2) (h2 : π < α ∧ α < 3 * π / 2) : 
sin α = - (Real.sqrt (1 / 5)) := 
by sorry

end sin_alpha_value_l569_569704


namespace commodity_Y_increase_l569_569895

variables {X Y : ℕ → ℝ} 
variables {y : ℝ}

-- Conditions setup
def price_X_increase := 0.40 -- Price of X increases by 40 paise every year
def price_X_1988 := 4.20
def price_Y_1988 := 6.30
def price_1998_condition (X Y : ℝ) := X = Y + 0.40

theorem commodity_Y_increase :
  (∀ t, X t = price_X_1988 + t * price_X_increase) ∧
  (∀ t, Y t = price_Y_1988 + t * y) ∧
  price_1998_condition (X 10) (Y 10)
  → y = 0.15 :=
by
  sorry

end commodity_Y_increase_l569_569895


namespace seeds_in_small_gardens_l569_569275

theorem seeds_in_small_gardens 
  (total_seeds : ℕ)
  (planted_seeds : ℕ)
  (small_gardens : ℕ)
  (remaining_seeds := total_seeds - planted_seeds) 
  (seeds_per_garden := remaining_seeds / small_gardens) :
  total_seeds = 101 → planted_seeds = 47 → small_gardens = 9 → seeds_per_garden = 6 := by
  sorry

end seeds_in_small_gardens_l569_569275


namespace problem_statement_l569_569707

-- Definitions of parallel and perpendicular relations
variable (Line Plane : Type)
variable (parallel : Line → Plane → Prop) (perpendicular : Line → Line → Prop) (subset : Line → Plane → Prop)
variable (Plane_parallel : Plane → Plane → Prop) (Line_parallel : Line → Line → Prop)

-- Defining variables to represent lines and planes
variables (l m : Line) (α β : Plane)

-- The proof problem statement
theorem problem_statement
  (H1 : parallel l α)
  (H2 : Line_parallel l m)
  (H3 : Plane_parallel α β)
  (H4 : ¬ subset m β) :
  Line_parallel m β :=
by
  sorry

end problem_statement_l569_569707


namespace second_player_wins_3x3_board_game_l569_569914

theorem second_player_wins_3x3_board_game :
  ∀ (board : Matrix (Fin 3) (Fin 3) ℕ),
    (∀ i j, board i j = 1 ∨ board i j = 0) →
    (∃ i j, board i j = 1) →
    (∃ i j, board i j = 0) →
    (∃ b', (∀ i j, b' i j = 1 ∨ b' i j = 0) ∧
          det b' = 0) :=
by
  sorry

end second_player_wins_3x3_board_game_l569_569914


namespace problem_statement_l569_569352

noncomputable def f : ℝ → ℝ
| x => if x < 0 then f (-x)
       else if x < 2 then log x
       else f (x - 2)

theorem problem_statement :
  f(-2018) + f(2019) = 1 :=
by sorry

end problem_statement_l569_569352


namespace capsule_depth_equation_l569_569143

theorem capsule_depth_equation (x y z : ℝ) (h : y = 4 * x + z) : y = 4 * x + z := 
by 
  exact h

end capsule_depth_equation_l569_569143


namespace four_digit_perfect_cubes_divisible_by_16_l569_569383

theorem four_digit_perfect_cubes_divisible_by_16 : (∃ k : ℕ, k = 3) :=
by
  let possible_cubes := [12 ^ 3, 16 ^ 3, 20 ^ 3]
  have h1 : 12 ^ 3 = 1728 := by norm_num
  have h2 : 16 ^ 3 = 4096 := by norm_num
  have h3 : 20 ^ 3 = 8000 := by norm_num

  have h4 : (1728, 4096, 8000).all (λ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 16 = 0)
    := by norm_num

  use 3
  trivial

end four_digit_perfect_cubes_divisible_by_16_l569_569383


namespace new_year_season_markup_l569_569974

variable {C : ℝ} (hC : 0 < C)

theorem new_year_season_markup (h1 : ∀ C, C > 0 → ∃ P1, P1 = 1.20 * C)
                              (h2 : ∀ (P1 M : ℝ), M >= 0 → ∃ P2, P2 = P1 * (1 + M / 100))
                              (h3 : ∀ P2, ∃ P3, P3 = P2 * 0.91)
                              (h4 : ∃ P3, P3 = 1.365 * C) :
  ∃ M, M = 25 := 
by 
  sorry

end new_year_season_markup_l569_569974


namespace range_of_m_l569_569417

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 5 < 4 * x - 1 ∧ x > m → x > 2) → m ≤ 2 :=
by
  intro h
  have h₁ := h 2
  sorry

end range_of_m_l569_569417


namespace number_of_possible_x_l569_569029

theorem number_of_possible_x (x : ℕ) (h1 : x < 13) (h2 : x > 7) :
  {n : ℕ | 7 < n ∧ n < 13}.card = 5 :=
by { sorry }

end number_of_possible_x_l569_569029


namespace fixed_point_exponential_l569_569697

theorem fixed_point_exponential (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ∃ x y : ℝ, x = 1 / 2 ∧ y = -1 ∧ y = a^(2 * x - 1) - 2 :=
begin
  use (1 / 2, -1),
  split,
  { refl, },
  split,
  { refl, },
  { field_simp,
    rw [pow_sub_eq_sub_pow, pow_one, h₂],
    norm_num,
    sorry, },
end

end fixed_point_exponential_l569_569697


namespace number_of_four_digit_cubes_divisible_by_16_l569_569385

theorem number_of_four_digit_cubes_divisible_by_16 :
  (finset.Icc 5 10).card = 6 :=
by sorry

end number_of_four_digit_cubes_divisible_by_16_l569_569385


namespace light_ray_reflection_and_distance_l569_569251

/-- Conditions and hypotheses --/
def point_A := (-3, 3) : ℝ × ℝ
def circle_C (x y : ℝ) := x^2 + y^2 - 4 * x - 4 * y + 7 = 0
def reflected_point_A := (-3, -3) : ℝ × ℝ

/-- Problem statement for the first part --/
def reflected_light_ray_intersections (k : ℝ) (l m : ℝ) :=
  (3 * l - 4 * k - 3 = 0 ∨
   4 * l - 3 * k - 3 = 0) ∧
  circle_C l m ∧
  point_A.implies (4 * l - 3 * k = 0 ∨ 3 * l - 4 * k = 0)

/-- Problem statement for the second part --/
def distance_traveled :=
  let distance_point_center := Real.sqrt ((-3 - 2)^2 + (-3 - 2)^2) in
  distance_point_center - 1 = 7

/-- Combined proof of the problem statement --/
theorem light_ray_reflection_and_distance :
  (∃ k l m, reflected_light_ray_intersections k l m) ∧ distance_traveled := 
  by
  sorry

end light_ray_reflection_and_distance_l569_569251


namespace number_of_first_group_is_four_l569_569187

noncomputable def avg_eleven_numbers := 10.7
noncomputable def avg_first_few_numbers := 10.5
noncomputable def avg_last_six_numbers := 11.4
noncomputable def sixth_number := 13.700000000000017

theorem number_of_first_group_is_four
  (S : ℝ := 11 * avg_eleven_numbers)
  (n : ℝ := 4)
  (h1 : (n - 1) * avg_first_few_numbers + sixth_number + 6 * avg_last_six_numbers = S)
  : n = 4 :=
by
  -- Proof omitted
  sorry

end number_of_first_group_is_four_l569_569187


namespace base_six_conversion_addition_l569_569570

def base_six_to_base_ten (n : ℕ) : ℕ :=
  4 * 6^0 + 1 * 6^1 + 2 * 6^2

theorem base_six_conversion_addition : base_six_to_base_ten 214 + 15 = 97 :=
by
  sorry

end base_six_conversion_addition_l569_569570


namespace fred_more_than_daniel_l569_569996

-- Definitions and conditions from the given problem.
def total_stickers : ℕ := 750
def andrew_kept : ℕ := 130
def daniel_received : ℕ := 250
def fred_received : ℕ := total_stickers - andrew_kept - daniel_received

-- The proof problem statement.
theorem fred_more_than_daniel : fred_received - daniel_received = 120 := by 
  sorry

end fred_more_than_daniel_l569_569996


namespace find_y_for_slope_l569_569340

theorem find_y_for_slope (y : ℝ) :
  let R := (-3, 9)
  let S := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 ↔ y = -3 :=
by
  simp [slope]
  sorry

end find_y_for_slope_l569_569340


namespace find_a_for_tangent_parallel_l569_569371

theorem find_a_for_tangent_parallel : 
  ∀ a : ℝ,
  (∀ (x y : ℝ), y = Real.log x - a * x → x = 1 → 2 * x + y - 1 = 0) →
  a = 3 :=
by
  sorry

end find_a_for_tangent_parallel_l569_569371


namespace prop_false_iff_a_lt_neg_13_over_2_l569_569081

theorem prop_false_iff_a_lt_neg_13_over_2 :
  (¬ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + a * x + 9 ≥ 0) ↔ a < -13 / 2 := 
sorry

end prop_false_iff_a_lt_neg_13_over_2_l569_569081


namespace exchange_rate_l569_569442

theorem exchange_rate (USD : ℕ) (KRW : ℕ) (h : USD = 140 ∧ KRW = 158760) : KRW / USD = 1134 :=
by
  cases h with h_usd h_krw
  rw [h_usd, h_krw]
  norm_num -- simplifying numerical expression
  sorry -- remaining proof

end exchange_rate_l569_569442


namespace valid3WordSentences_l569_569864

-- Define the set of words in the Gnollish language
inductive Word
| splargh | glumph | amr

-- Definition of a sentence as a list of 3 words
def Sentence := List Word

-- Define a property that checks if a sentence is valid based on given conditions
def isValidSentence (s: Sentence) : Prop :=
  match s with
  | [Word.splargh, Word.glumph, _] => False
  | [_, Word.splargh, Word.glumph] => False
  | _ => True

-- Define the proof statement
theorem valid3WordSentences : Finset.filter isValidSentence (Finset.ofList (List.product (List.product [Word.splargh, Word.glumph, Word.amr] [Word.splargh, Word.glumph, Word.amr]) [Word.splargh, Word.glumph, Word.amr])).card = 21 :=
by sorry

end valid3WordSentences_l569_569864


namespace probability_real_l569_569164
 
noncomputable def problem : ℚ × ℚ → Prop
| (a, b) => (cos (a * real.pi) + complex.I * sin (b * real.pi)) ^ 2 = 0

def conditions (a b : ℚ) : Prop :=
  0 ≤ a ∧ a < 3 ∧ 0 ≤ b ∧ b < 3 ∧ ∃ n d : ℤ, n * a = d ∧ n * b = d ∧ 1 ≤ d ∧ d ≤ 10

theorem probability_real (a b : ℚ) (h : conditions a b) :
  ∃ p : ℚ, p = 1 / 10 :=
sorry

end probability_real_l569_569164


namespace defective_smartphones_in_shipment_l569_569981

theorem defective_smartphones_in_shipment :
  ∃ D : ℕ, D * (D - 1) / (220 * 219) = 0.14470734744707348 ∧ D = 84 :=
by {
  -- Mathematical conditions given in the problem
  have h_prob : 0.14470734744707348 * 220 * 219 = 6981.999999999999 := sorry,
  use 84,
  -- Start involving assumptions and verification
  split,
  { -- This will demonstrate D is the solution of the quadratic equation
    have D := 84,
    calc
      (84 * (84 - 1)) / (220 * 219)
      = (84 * 83) / 48240 : by {norm_num}
      ... = 6981.999999999999 / 48240 : by {norm_num}
      ... = 0.14470734744707348 : by {norm_num} },

  -- Conclude with the equality
  refl
}

end defective_smartphones_in_shipment_l569_569981


namespace students_registered_for_course_l569_569858

theorem students_registered_for_course :
  ∀ (students_present_yesterday students_absent_today: ℕ),
    students_present_yesterday = 70 →
    students_absent_today = 30 →
    let students_attended_today := 0.9 * 2 * students_present_yesterday in
    students_registered = students_attended_today + students_absent_today →
    students_registered = 156 :=
by
  intros students_present_yesterday students_absent_today h1 h2 h3
  rw [h1, h2, h3]
  sorry

end students_registered_for_course_l569_569858


namespace mean_proportional_l569_569530

theorem mean_proportional (k : ℝ) : (∃ x : ℝ, x^2 = 234 * k ∧ x = 156) → k = 104 :=
by
  intros h
  rcases h with ⟨x, hx1, hx2⟩
  rw hx2 at hx1
  have hx3 : 156 ^ 2 = 234 * k := hx1
  rw [←hx3, mul_comm] at hx1
  norm_num at hx1
  exact hx1

end mean_proportional_l569_569530


namespace ratio_of_third_to_second_is_four_l569_569206

theorem ratio_of_third_to_second_is_four
  (x y z k : ℕ)
  (h1 : y = 2 * x)
  (h2 : z = k * y)
  (h3 : (x + y + z) / 3 = 165)
  (h4 : y = 90) :
  z / y = 4 :=
by
  sorry

end ratio_of_third_to_second_is_four_l569_569206


namespace area_BCD_proof_l569_569434

-- Define the conditions and setup
def base_AC : ℝ := 8
def height_B : ℝ := 8 -- height corresponding to base AC determined by given area of ABC
def area_ABC : ℝ := 32
def base_CD : ℝ := 32

-- Assume given area of triangle ABC
axiom given_area_ABC : (1/2) * base_AC * height_B = area_ABC

-- Define the problem to find the area of triangle BCD
def area_BCD : ℝ := (1/2) * base_CD * height_B

-- The theorem to be proved
theorem area_BCD_proof : area_BCD = 128 := by
  -- Proof would go here
  sorry

end area_BCD_proof_l569_569434


namespace power_evaluation_l569_569760

theorem power_evaluation (x : ℕ) (h1 : 3^x = 81) : 3^(x+2) = 729 := by
  sorry

end power_evaluation_l569_569760


namespace number_of_four_digit_cubes_divisible_by_16_l569_569387

theorem number_of_four_digit_cubes_divisible_by_16 :
  (finset.Icc 5 10).card = 6 :=
by sorry

end number_of_four_digit_cubes_divisible_by_16_l569_569387


namespace minimum_value_function_inequality_ln_l569_569706

noncomputable def f (x : ℝ) := x * Real.log x

theorem minimum_value_function (t : ℝ) (ht : 0 < t) :
  ∃ (xmin : ℝ), xmin = if (0 < t ∧ t < 1 / Real.exp 1) then -1 / Real.exp 1 else t * Real.log t :=
sorry

theorem inequality_ln (x : ℝ) (hx : 0 < x) : 
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end minimum_value_function_inequality_ln_l569_569706


namespace probability_one_defective_item_l569_569699

theorem probability_one_defective_item : 
  let total_number_of_ways := Nat.choose 6 3,
      ways_with_one_defective := Nat.choose 4 2 * Nat.choose 2 1,
      probability := (ways_with_one_defective : ℚ) / total_number_of_ways
  in probability = 3 / 5 := 
by
  let total_number_of_ways := Nat.choose 6 3
  let ways_with_one_defective := Nat.choose 4 2 * Nat.choose 2 1
  let probability := (ways_with_one_defective : ℚ) / total_number_of_ways
  have h1 : total_number_of_ways = 20 := by sorry
  have h2 : ways_with_one_defective = 12 := by sorry
  have h3 : probability = 12 / 20 := by sorry
  have h4 : (12 / 20 : ℚ) = 3 / 5 := by sorry
  exact h4

end probability_one_defective_item_l569_569699


namespace relationship_among_abc_l569_569733

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_even : ∀ x, f x = f (-x))
variable (h_mono_dec : ∀ x y, x < y → y ≤ 0 → f y ≤ f x)

-- Definitions for \(a\), \(b\), and \(c\)
noncomputable def a := f (Real.log 3 / Real.log 2)
noncomputable def b := f (Real.log 5 / Real.log 4)
noncomputable def c := f (Real.sqrt 2)

-- Proof statement
theorem relationship_among_abc : b < a ∧ a < c := 
sorry

end relationship_among_abc_l569_569733


namespace expenditure_on_digging_the_well_l569_569939

noncomputable def well_radius : ℝ := 1.5
noncomputable def well_depth : ℝ := 14
noncomputable def cost_per_cubic_meter : ℝ := 17
noncomputable def pi : ℝ := Real.pi

noncomputable def volume_of_well : ℝ := pi * (well_radius ^ 2) * well_depth

noncomputable def expenditure : ℝ := volume_of_well * cost_per_cubic_meter

theorem expenditure_on_digging_the_well : expenditure ≈ 1682.32 := by
  sorry

end expenditure_on_digging_the_well_l569_569939


namespace rank_schools_l569_569089

theorem rank_schools (sA sB sC : list ℤ) 
  (h1 : sA = [80, 73, 65, 62, 59])
  (h2 : sB = [79, 77, 69, 61, 60])
  (h3 : sC = [86, 75, 61, 58, 51]) :
  let avgA := list.sum sA / 5
  let avgB := list.sum sB / 5
  let avgC := list.sum sC / 5
  in avgB > avgA ∧ avgA > avgC :=
by {
  sorry
}

end rank_schools_l569_569089


namespace solve_quadratic_l569_569511

def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem solve_quadratic : (quadratic_eq (-2) 1 3 (-1)) ∧ (quadratic_eq (-2) 1 3 (3/2)) :=
by
  sorry

end solve_quadratic_l569_569511


namespace total_count_not_47_l569_569802

theorem total_count_not_47 (h c : ℕ) : 11 * h + 6 * c ≠ 47 := by
  sorry

end total_count_not_47_l569_569802


namespace jordan_rectangle_width_l569_569584

theorem jordan_rectangle_width :
  (∃ (w : ℝ), (8 * 15 = 4 * w) ∧ w = 30) :=
begin
  existsi 30,
  split,
  { norm_num },
  { refl },
end

end jordan_rectangle_width_l569_569584


namespace complex_problem_l569_569462

noncomputable def imaginary_unit : ℂ := complex.I
noncomputable def z : ℂ := 3 + 4 * imaginary_unit
noncomputable def conjugate_z : ℂ := complex.conj(z)

theorem complex_problem : (imaginary_unit^2018) * conjugate_z = -3 + 4 * imaginary_unit := 
by
  sorry

end complex_problem_l569_569462


namespace eccentricity_conic_section_l569_569142

-- Definitions based on the conditions
def conic_section (Γ : Type) : Prop := sorry
def F1 (Γ : Type) : Type := sorry
def F2 (Γ : Type) : Type := sorry
def P (Γ : Type) : Type := sorry
def distance (a b : Type) : ℝ := sorry

-- Given conditions
axiom F1F2_ratio (Γ : Type) (F1 : F1 Γ) (F2 : F2 Γ) (P : P Γ) 
  : distance P F1 / distance F1 F2 / distance P F2 = 4 / 3 / 2

-- Theorem statement
theorem eccentricity_conic_section (Γ : Type) (F1 : F1 Γ) (F2 : F2 Γ) (P : P Γ)
  [conic_section Γ] : 
  ∃ e, (e = 1/2 ∨ e = 3/2) :=
sorry

end eccentricity_conic_section_l569_569142


namespace time_for_type_A_problems_l569_569231

theorem time_for_type_A_problems :
  ∀ (total_questions type_A_questions: ℕ) (total_time_hours: ℝ) (twice_time_factor: ℝ),
    total_questions = 200 →
    type_A_questions = 15 →
    total_time_hours = 3 →
    twice_time_factor = 2 →
    let total_minutes := total_time_hours * 60 in
    let x := total_minutes / (total_questions + (twice_time_factor - 1) * type_A_questions) in
    (type_A_questions * twice_time_factor * x ≈ 25.116) :=
by
  intros total_questions type_A_questions total_time_hours twice_time_factor
  sorry

end time_for_type_A_problems_l569_569231


namespace car_speed_second_half_l569_569598

theorem car_speed_second_half :
  ∀ (D : ℝ), 
  let T := D / 60,
      T1 := (D / 2) / 75,
      T2 := (D / 2) / 50 in
  T = T1 + T2 → (D / 60) = (D / 150) + (D / (2 * (D / 2) / 50)) :=
by simp; rw [eq_comm, div_eq_mul_inv, mul_comm D, mul_div_cancel_left, ←div_div, mul_inv_cancel, inv_div, eq_div_iff_mul_eq, div_eq_mul_inv]; nlinarith

end car_speed_second_half_l569_569598


namespace max_length_OB_l569_569916

theorem max_length_OB (O A B : Point) (angle_AOB : angle) 
  (h1 : 45 = angle_AOB.to_degrees) (h2 : dist A B = 1) : 
  ∃ M, M = sqrt 2 ∧ ∀ OB, dist O B = OB → OB ≤ M :=
sorry

end max_length_OB_l569_569916


namespace function_behavior_increasing_l569_569771

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem function_behavior_increasing (h_deriv : ∀ x ∈ set.Ioo a b, deriv f x > 0) (h_end : f b ≤ 0) :
  ∀ x ∈ set.Ioo a b, f x < 0 :=
by
  sorry

end function_behavior_increasing_l569_569771


namespace distance_apart_in_six_hours_l569_569835

theorem distance_apart_in_six_hours 
  (v_mary v_john d : ℝ) (h_v_mary : v_mary = 12) (h_v_john : v_john = 9) (h_d : d = 90): 
  ∃ t : ℝ, t = 6 ∧ sqrt ((v_mary * t) ^ 2 + (v_john * t) ^ 2) = d :=
by 
  use 6
  split
  all_goals
    try { exact rfl }
  sorry

end distance_apart_in_six_hours_l569_569835


namespace coordinates_of_point_B_l569_569042

def A : ℝ × ℝ × ℝ := (2, -1, 7)
def a : ℝ × ℝ × ℝ := (8, 9, -12)
def segment_length : ℝ := 34

theorem coordinates_of_point_B
  (A : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) (segment_length : ℝ)
  (hA : A = (2, -1, 7)) (ha : a = (8, 9, -12)) (h_length : segment_length = 34) :
  let λ := segment_length / real.sqrt (8^2 + 9^2 + (-12)^2) in
  let B := (A.1 + λ * a.1, A.2 + λ * a.2, A.3 + λ * a.3) in
  B = (18, 17, -17) :=
  by
  sorry

end coordinates_of_point_B_l569_569042


namespace range_of_AB_l569_569438

noncomputable def quadrilateral_ABCD {A B C D : Type*} [plane A] [plane B] [plane C] [plane D] 
    (angle_A : real) (angle_B : real) (angle_C : real) (BC : real) : set (real) :=
  {AB | angle_A = 75 ∧ angle_B = 75 ∧ angle_C = 75 ∧ BC = 2 }

theorem range_of_AB 
    {A B C D : Type*} [plane A] [plane B] [plane C] [plane D] 
    (angle_A : real) (angle_B : real) (angle_C : real) (BC : real) :
    quadrilateral_ABCD angle_A angle_B angle_C BC = 
    {AB : real | sqrt(6) - sqrt(2) < AB ∧ AB < sqrt(6) + sqrt(2)} :=
begin
  sorry
end

end range_of_AB_l569_569438


namespace opposite_of_2023_l569_569892

-- Define the opposite (additive inverse) function
def additive_inverse (a : ℤ) : ℤ := -a

-- Define the specific problem condition
def condition (n : ℤ) : Prop := 2023 + n = 0

-- Prove that the additive inverse of 2023 satisfies the condition
theorem opposite_of_2023 : condition (additive_inverse 2023) :=
by
  unfold condition additive_inverse
  simp
  exact rfl

end opposite_of_2023_l569_569892


namespace min_cos_A_l569_569086

-- Definitions based on given conditions
variables {A B C : ℝ} -- Angles of triangle ABC

-- Given condition
def condition : Prop := 2 * (Real.tan B + Real.tan C) = Real.tan B / Real.cos C + Real.tan C / Real.cos B

-- Statement of the problem
theorem min_cos_A (h : condition) : Real.cos A ≥ 1 / 2 :=
sorry

end min_cos_A_l569_569086


namespace gcd_78_143_l569_569571

theorem gcd_78_143 : Nat.gcd 78 143 = 13 :=
by
  sorry

end gcd_78_143_l569_569571


namespace perpendicular_lines_relationship_l569_569950

-- Definition of lines being perpendicular to the same line
def perpendicular_to_same_line (l1 l2 l: ℝ^3 → ℝ^3) : Prop :=
  ∀ (x: ℝ^3), inner (l1 x) (l x) = 0 ∧ inner (l2 x) (l x) = 0

-- The theorem to be proven
theorem perpendicular_lines_relationship (L1 L2 L: ℝ^3 → ℝ^3) (h: perpendicular_to_same_line L1 L2 L) :
  (∃ p, parallel L1 L2) ∨ (∃ x, L1 x = L2 x) ∨ skew L1 L2 :=
sorry

end perpendicular_lines_relationship_l569_569950


namespace highest_power_of_3_dividing_N_is_1_l569_569523

-- Define the integer N as described in the problem
def N : ℕ := 313233515253

-- State the problem
theorem highest_power_of_3_dividing_N_is_1 : ∃ k : ℕ, (3^k ∣ N) ∧ ∀ m > 1, ¬ (3^m ∣ N) ∧ k = 1 :=
by
  -- Specific solution details and steps are not required here
  sorry

end highest_power_of_3_dividing_N_is_1_l569_569523


namespace conference_problem_l569_569213

open Finset

-- Definitions
variables {A B : Type} [Fintype A] [Fintype B] (n : ℕ) [DecidableEq A] [DecidableEq B]
variable (knows : A → B → Prop)

-- Hypotheses
variable [Ha : Card (A) = n]
variable [Hb : Card (B) = n]

-- Main goal
theorem conference_problem : ∃ C : Finset A, C.nonempty ∧ 
  (∀ b ∈ B, even (Card ({a ∈ C | knows a b})) ∨ odd (Card ({a ∈ C | knows a b}))) :=
sorry

end conference_problem_l569_569213


namespace find_m_of_ellipse_focus_l569_569055

noncomputable def ellipse_focus (m : ℝ) (h : m > 0) :=
  let a := 5
  let c := 4
  a^2 - m^2 = c^2

theorem find_m_of_ellipse_focus (m : ℝ) (h : m > 0) : m = 3 :=
  have h_eq : 25 - m^2 = 16 := by
    exact ellipse_focus m h
  sorry

end find_m_of_ellipse_focus_l569_569055


namespace max_least_integer_l569_569877

theorem max_least_integer (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2160) (h_order : x ≤ y ∧ y ≤ z) : x ≤ 10 :=
by
  sorry

end max_least_integer_l569_569877


namespace find_quadruples_l569_569006

noncomputable theory

open Classical Nat

theorem find_quadruples (a b p n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Prime p) (h4 : a^3 + b^3 = p^n) :
  ∃ k : ℕ, (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3 * k + 1) ∨
           (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3 * k + 2) ∨
           (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3 * k + 2) :=
by {
  sorry
}

end find_quadruples_l569_569006


namespace solve_cubic_eq_solve_quadratic_eq_l569_569173

-- Define the first equation and prove its solution
theorem solve_cubic_eq (x : ℝ) (h : x^3 + 64 = 0) : x = -4 :=
by
  -- skipped proof
  sorry

-- Define the second equation and prove its solutions
theorem solve_quadratic_eq (x : ℝ) (h : (x - 2)^2 = 81) : x = 11 ∨ x = -7 :=
by
  -- skipped proof
  sorry

end solve_cubic_eq_solve_quadratic_eq_l569_569173


namespace tan_half_monotonic_on_unit_interval_l569_569991

open Real

theorem tan_half_monotonic_on_unit_interval :
  ∀ x : ℝ, 0 < x ∧ x < π → monotone_on (λ x, tan (x / 2)) (Set.Ioo 0 π) := 
sorry

end tan_half_monotonic_on_unit_interval_l569_569991


namespace ratio_of_dogs_with_spots_to_total_number_l569_569832

theorem ratio_of_dogs_with_spots_to_total_number
  (T : ℕ) (h1 : T / 5 = 6) (d_spots : ℕ) (d_spots_count : d_spots = 15) :
  d_spots / T = 1 / 2 := by
suffices T = 30 from by
  rw [d_spots_count, this]
  norm_num
rw [← Nat.div_eq_of_eq_mul_left (by norm_num : 0 < 5) h1]
norm_num

end ratio_of_dogs_with_spots_to_total_number_l569_569832


namespace juan_faster_than_peter_l569_569814

theorem juan_faster_than_peter (J : ℝ) :
  (Peter_speed : ℝ) = 5.0 →
  (time : ℝ) = 1.5 →
  (distance_apart : ℝ) = 19.5 →
  (J + 5.0) * time = distance_apart →
  J - 5.0 = 3 := 
by
  intros Peter_speed_eq time_eq distance_apart_eq relative_speed_eq
  sorry

end juan_faster_than_peter_l569_569814


namespace pencils_bought_l569_569640

section
variables 
  (P_initial : ℕ) -- initial pencils 
  (P_gave_away : ℕ) -- pencils given away
  (P_now : ℕ) -- pencils now

-- Define the conditions
def pencils_after_giving_away := P_initial - P_gave_away

-- Define what we need to prove - number of pencils Brian bought
def pencils_purchased := P_now - pencils_after_giving_away

theorem pencils_bought (h_initial : P_initial = 39) 
  (h_gave_away : P_gave_away = 18) 
  (h_now : P_now = 43) : pencils_purchased = 22 := by
  unfold pencils_after_giving_away pencils_purchased
  rw [h_initial, h_gave_away, h_now]
  sorry
end

end pencils_bought_l569_569640


namespace slope_of_perpendicular_line_l569_569684

noncomputable def perpendicular_slope : ℚ :=
  let slope := (3/4 : ℚ) in
  - (1 / slope)

theorem slope_of_perpendicular_line (a b c : ℚ) (h : a = 3 ∧ b = -4 ∧ c = 8) : 
  perpendicular_slope = - (4 / 3) :=
by {
  sorry
}

end slope_of_perpendicular_line_l569_569684


namespace tower_height_l569_569998

theorem tower_height (α β γ : ℝ) (h : ℝ)
  (tan_α : tan α = h / 80)
  (tan_β : tan β = h / 160)
  (tan_γ : tan γ = h / 240)
  (angles_sum : α + β + γ = π / 2) : 
  h = 80 :=
sorry

end tower_height_l569_569998


namespace hare_reaches_bushes_before_dog_catches_it_l569_569964

theorem hare_reaches_bushes_before_dog_catches_it (
  (v_d v_r : ℝ) (vd_pos : v_d = 17) (vr_pos : v_r = 14) (d_0 d_b : ℝ) (d0_pos : d_0 = 150) (db_pos : d_b = 520)
) : let v_relative := v_d - v_r 
      in v_relative > 0 → d_b / v_r < d_0 / v_relative → False :=
by {
  intros v_d v_r vd_pos vr_pos d_0 d_b d0_pos db_pos,
  let v_relative := v_d - v_r,
  intro h_rel_pos,
  intro h_time,
  have := (d_b / v_r) < (d_0 / v_relative),
  sorry
}

end hare_reaches_bushes_before_dog_catches_it_l569_569964


namespace four_digit_cubes_divisible_by_16_l569_569395

theorem four_digit_cubes_divisible_by_16 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.finite
  ∧ ∃ n, n = 3 ∧ {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.card = n := 
by
  -- The proof steps would go here.
  sorry

end four_digit_cubes_divisible_by_16_l569_569395


namespace model_to_statue_scale_l569_569527

theorem model_to_statue_scale
  (statue_height_ft : ℕ)
  (model_height_in : ℕ)
  (ft_to_in : ℕ)
  (statue_height_in : ℕ)
  (scale : ℕ)
  (h1 : statue_height_ft = 120)
  (h2 : model_height_in = 6)
  (h3 : ft_to_in = 12)
  (h4 : statue_height_in = statue_height_ft * ft_to_in)
  (h5 : scale = (statue_height_in / model_height_in) / ft_to_in) : scale = 20 := 
  sorry

end model_to_statue_scale_l569_569527


namespace sin_cos_identity_l569_569924

theorem sin_cos_identity (α : ℝ) :
  (sin (π + α))^2 - cos (π + α) * cos (-α) + 1 = 2 :=
by
  -- Import necessary trigonometric identities and simplifications
  have h1 : sin (π + α) = -sin α := by sorry
  have h2 : cos (π + α) = -cos α := by sorry
  have h3 : cos (-α) = cos α := by sorry
  sorry

end sin_cos_identity_l569_569924


namespace maximize_income_l569_569090

-- Define the ticket price and income conditions
noncomputable def y (x : ℕ) : ℤ :=
  if x ≤ 10 then 100 * x - 575 else -3 * x^2 + 130 * x - 575

-- Define the domain of x
def domain (x : ℕ) : Prop := 6 ≤ x ∧ x ≤ 38

-- Define the main theorem
theorem maximize_income : ∃ x : ℕ, domain x ∧ y x = 833 :=
begin
  use 22,
  split,
  {
    split,
    {
      -- Prove 6 ≤ 22
      linarith,
    },
    {
      -- Prove 22 ≤ 38
      linarith,
    },
  },
  {
    -- Prove y 22 = 833
    unfold y,
    rw if_neg,
    {
      -- Simplification
      norm_num,
    },
    {
      -- Proof condition for "else"
      linarith,
    },
  },
end

end maximize_income_l569_569090


namespace emily_total_cost_l569_569253

-- Definition of the monthly cell phone plan costs and usage details
def base_cost : ℝ := 30
def cost_per_text : ℝ := 0.10
def cost_per_extra_minute : ℝ := 0.15
def cost_per_extra_gb : ℝ := 5
def free_hours : ℝ := 25
def free_gb : ℝ := 15
def texts : ℝ := 150
def hours : ℝ := 26
def gb : ℝ := 16

-- Calculate the total cost
def total_cost : ℝ :=
  base_cost +
  (texts * cost_per_text) +
  ((hours - free_hours) * 60 * cost_per_extra_minute) +
  ((gb - free_gb) * cost_per_extra_gb)

-- The proof statement that Emily had to pay $59
theorem emily_total_cost :
  total_cost = 59 := by
  sorry

end emily_total_cost_l569_569253


namespace right_triangles_count_l569_569501

theorem right_triangles_count :
  let x_range := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}.to_list
  let y_range := {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}.to_list in
  (∑ y in y_range,
    ∑' xy_ij in respectively_distinct_pairs x_range,
      let (x_i, x_j) := xy_ij in
      ∑ qy in filter (fun y' => y' ≠ y) y_range, 1) = 450 :=
by {
  sorry,
}

end right_triangles_count_l569_569501


namespace total_students_registered_l569_569856

theorem total_students_registered (num_present_yesterday : ℕ) (num_absent_today : ℕ) 
  (percent_less : ℝ) (num_students : ℕ) 
  (h1 : num_present_yesterday = 70)
  (h2 : num_absent_today = 30)
  (h3 : percent_less = 0.1)
  (h4 : num_students = 156) :
  let twice_present_yesterday := 2 * num_present_yesterday in
  let reduction := percent_less * twice_present_yesterday in
  let num_present_today := twice_present_yesterday - reduction in
  num_students = num_present_today + num_absent_today :=
by
  -- Using the conditions provided to show the equivalence.
  sorry

end total_students_registered_l569_569856


namespace cat_mouse_position_after_moves_l569_569094

-- Define the total number of moves
def total_moves : ℕ := 360

-- Define cat's cycle length and position calculation
def cat_cycle_length : ℕ := 5
def cat_final_position := total_moves % cat_cycle_length

-- Define mouse's cycle length and actual moves per cycle
def mouse_cycle_length : ℕ := 10
def mouse_effective_moves_per_cycle : ℕ := 9
def total_mouse_effective_moves := (total_moves / mouse_cycle_length) * mouse_effective_moves_per_cycle
def mouse_final_position := total_mouse_effective_moves % mouse_cycle_length

theorem cat_mouse_position_after_moves :
  cat_final_position = 0 ∧ mouse_final_position = 4 :=
by
  sorry

end cat_mouse_position_after_moves_l569_569094


namespace car_distance_l569_569630

theorem car_distance (time_am_18 : ℕ) (time_car_48 : ℕ) (h : time_am_18 = time_car_48) : 
  let distance_am_18 := 18
  let distance_car_48 := 48
  let total_distance_am := 675
  let distance_ratio := (distance_am_18 : ℝ) / (distance_car_48 : ℝ)
  let distance_car := (total_distance_am : ℝ) * (distance_car_48 : ℝ) / (distance_am_18 : ℝ)
  distance_car = 1800 :=
by
  sorry

end car_distance_l569_569630


namespace sequence_solution_l569_569486

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 5 ∧ a 8 = 8 ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 20) ∧
  (a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 5 ∧ a 5 = 8 ∧ a 6 = 7 ∧ a 7 = 5 ∧ a 8 = 8) :=
by {
  sorry
}

end sequence_solution_l569_569486


namespace ratio_of_areas_l569_569200

-- Definitions based on the conditions
def perimeter_of_square (s : ℝ) : ℝ := 4 * s
def circumference_of_circle (r : ℝ) : ℝ := 2 * real.pi * r
def area_of_square (s : ℝ) : ℝ := s^2
def area_of_circle (r : ℝ) : ℝ := real.pi * r^2

-- The mathematically equivalent proof problem
theorem ratio_of_areas (s r : ℝ) (h : 4 * s = 2 * real.pi * r) :
  (area_of_square s) / (area_of_circle r) = 4 / real.pi :=
by
  sorry

end ratio_of_areas_l569_569200


namespace four_digit_cubes_divisible_by_16_l569_569390

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l569_569390


namespace cylinder_base_ratio_l569_569126

variable (O : Point) -- origin
variable (a b c : ℝ) -- fixed point
variable (p q : ℝ) -- center of circular base
variable (α β : ℝ) -- intersection points with axis

-- Let O be the origin
-- Let (a, b, c) be the fixed point through which the cylinder passes
-- The cylinder's axis is parallel to the z-axis and the center of its base is (p, q)
-- The cylinder intersects the x-axis at (α, 0, 0) and the y-axis at (0, β, 0)
-- Let α = 2p and β = 2q

theorem cylinder_base_ratio : 
  α = 2 * p ∧ β = 2 * q → (a / p + b / q = 4) := by
  sorry

end cylinder_base_ratio_l569_569126


namespace max_value_when_a_equals_1_on_interval_neg1_1_range_of_a_for_two_distinct_roots_l569_569360

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x^2 + 4 * x - 3 - a

-- Given
variables (a : ℝ) (x : ℝ)

theorem max_value_when_a_equals_1_on_interval_neg1_1 :
  ∀ x, x ∈ set.Icc (-1 : ℝ) 1 → f 1 x ≤ 2 :=
by sorry

theorem range_of_a_for_two_distinct_roots :
  (∃ (a : ℝ), (a ∈ set.Ioo (-(1 : ℝ)) 0 ∪ set.Ioo 0 +∞) → 
  ∃ (x y : ℝ), f a x = 0 ∧ f a y = 0 ∧ x ≠ y) :=
by sorry

end max_value_when_a_equals_1_on_interval_neg1_1_range_of_a_for_two_distinct_roots_l569_569360


namespace inequality_solution_minimum_value_l569_569696

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem inequality_solution :
  {x : ℝ | f x > 7} = {x | x > 4 ∨ x < -3} :=
by
  sorry

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : ∀ x, f x ≥ m + n) :
  m + n = 3 →
  (m^2 + n^2 ≥ 9 / 2 ∧ (m = 3 / 2 ∧ n = 3 / 2)) :=
by
  sorry

end inequality_solution_minimum_value_l569_569696


namespace winning_jackpot_is_event_l569_569934

-- Definitions based on the conditions
def has_conditions (experiment : String) : Prop :=
  experiment = "A" ∨ experiment = "B" ∨ experiment = "C" ∨ experiment = "D"

def has_outcomes (experiment : String) : Prop :=
  experiment = "D"

def is_event (experiment : String) : Prop :=
  has_conditions experiment ∧ has_outcomes experiment

-- Statement to prove
theorem winning_jackpot_is_event : is_event "D" :=
by
  -- Trivial step to show that D meets both conditions and outcomes
  exact sorry

end winning_jackpot_is_event_l569_569934


namespace terence_tao_rock_probability_l569_569179

noncomputable def fifth_move_probability_rock : ℚ :=
  1 / 2

theorem terence_tao_rock_probability :
  let strategy := (moves : ℕ → ℕ → ℕ) in
  have initial_move : moves 0 1 := 0,  -- 0 denotes Rock
  have transition_probability (n : ℕ) : moves n (1 - moves n) :=
    if moves n = 0 then 1 / 2 else 1 / 2,
  probability_of_fifth_move_rock : moves 5 = 0 :=
  begin
    sorry
  end :=
by exact fifth_move_probability_rock

end terence_tao_rock_probability_l569_569179


namespace exterior_angle_10_sides_l569_569292

-- Define the interior angle formula
def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

-- Define the exterior angle formula
def exterior_angle (n : ℕ) : ℝ := 180 - interior_angle n

-- Prove that the exterior angle for a 10-sided polygon is 36 degrees
theorem exterior_angle_10_sides : exterior_angle 10 = 36 := by
  sorry

end exterior_angle_10_sides_l569_569292


namespace person_B_wins_l569_569488

theorem person_B_wins :
  (∃ strategyB : (Fin 2019 → Bool) → ℕ → Bool, 
   ∀ (colors : Fin 2019 → Option Bool) (turns : ℕ),
     (∀ i, colors i = some true ∨ colors i = some false) → 
     ((∀ i : Fin (2018), colors i = colors ⟨i + 1, sorry⟩) → (strategyB colors turns = true)) 
     ∨ (strategyB colors turns = false)) :=
sorry

end person_B_wins_l569_569488


namespace circle_equation_l569_569312

open Real

theorem circle_equation (x y : ℝ) :
  let center := (2, -1)
  let line := (x + y = 7)
  (center.1 - 2)^2 + (center.2 + 1)^2 = 18 :=
by
  sorry

end circle_equation_l569_569312


namespace exists_arithmetic_sequence_not_in_range_l569_569723

-- Definitions of floor and fractional part
def floor (x : ℝ) : ℤ := Int.floor x
def fractional (x : ℝ) : ℝ := x - floor x

-- Definition of the function f
def f (x : ℝ) : ℝ := floor (x^2) + fractional x

-- Arithmetic sequence definition
def a_n (n : ℕ) : ℝ := 22 + 40 * n + 1 / 3

-- The theorem statement 
theorem exists_arithmetic_sequence_not_in_range (x : ℝ) (hx : 0 < x):
  ∃ n : ℕ, a_n n ∉ set.range f :=
sorry

end exists_arithmetic_sequence_not_in_range_l569_569723


namespace water_addition_to_achieve_concentration_l569_569757

theorem water_addition_to_achieve_concentration :
  ∀ (w1 w2 : ℝ), 
  (60 * 0.25 = 15) →              -- initial amount of acid
  (15 / (60 + w1) = 0.15) →       -- first dilution to 15%
  (15 / (100 + w2) = 0.10) →      -- second dilution to 10%
  w1 + w2 = 90 :=                 -- total water added to achieve final concentration
by
  intros w1 w2 h_initial h_first h_second
  sorry

end water_addition_to_achieve_concentration_l569_569757


namespace area_of_triangle_QPO_l569_569786

-- We need to define the basic geometrical entities and the given conditions

def Point := ℝ × ℝ
def Parallelogram (A B C D : Point) := True  -- Placeholder for parallelogram definition

noncomputable def trisects (P Q : Point) := True  -- Placeholder for trisecting definition
noncomputable def area_of_parallelogram (A B C D : Point) : ℝ := sorry  -- Given as k

def intersects_at (P Q R : Point) := True  -- Placeholder for line intersection definition
def area_of_triangle (P Q R : Point) : ℝ := sorry  -- Placeholder for triangle area calculation

-- Given points and conditions
variables (A B C D N M P Q O : Point)
variables (k : ℝ)

-- Assumptions
axiom trisect_condition_1 : trisects D (B, C) ∧ trisects N (B, C)
axiom trisect_condition_2 : trisects C (A, D) ∧ trisects M (A, D)
axiom intersection_condition_1 : intersects_at D P O
axiom intersection_condition_2 : intersects_at C Q O
axiom parallelogram_condition : Parallelogram A B C D
axiom area_condition : area_of_parallelogram A B C D = k

-- Proof goal
theorem area_of_triangle_QPO : area_of_triangle Q P O = (5 * k / 6) :=
by
  sorry

end area_of_triangle_QPO_l569_569786


namespace oa_dot_ob_l569_569753

variable (k : ℝ) (x1 x2 y1 y2 : ℝ)

def parabola (x y : ℝ) := y^2 = 4*x
def focus : ℝ × ℝ := (1, 0)
def line (k x : ℝ) := y = k * (x - 1)

theorem oa_dot_ob (x1 x2 y1 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  (y1 = k * (x1 - 1)) ∧ (y2 = k * (x2 - 1)) ∧
  (x1 + x2 = 2 + 4 / k^2) ∧ (x1 * x2 = 1) →
  (x1 * x2 + y1 * y2 = -3) :=
by {
  sorry,
}

end oa_dot_ob_l569_569753


namespace circle_equation_center_x_axis_radius_sqrt2_l569_569194

theorem circle_equation_center_x_axis_radius_sqrt2:
  ∃ a : ℝ, ((x + a)^2 + y^2 = 2 ∧ ∃ b : ℝ, b ≠ a ∧ (x + b)^2 + y^2 = 2) :=
begin
  sorry -- proof not required
end

end circle_equation_center_x_axis_radius_sqrt2_l569_569194


namespace num_of_valid_N_l569_569070

-- Defining appropriate conditions
def is_four_digit (n: ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Predicate to check if exactly one of N and N+20 is a four-digit number
def exactly_one_four_digit (N : ℕ) : Prop :=
  (is_four_digit N ∧ ¬ is_four_digit (N + 20)) ∨
  (¬ is_four_digit N ∧ is_four_digit (N + 20))

-- Main theorem statement
theorem num_of_valid_N : (Finset.filter exactly_one_four_digit (Finset.range 10000)).card = 40 :=
by
  sorry

end num_of_valid_N_l569_569070


namespace correct_statements_l569_569749

theorem correct_statements (f : ℝ → ℝ) (t : ℝ)
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : (∀ x : ℝ, f x = f (-x)) ∧ (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2) ∧ f (-2) = 0)
  (h3 : ∀ x : ℝ, f (-x) = -f x)
  (h4 : ∀ x : ℝ, f (x - t) = f (x + t)) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 > f x2 ↔ x1 < x2) ∧
  (∀ x : ℝ, f x - f (|x|) = - (f (-x) - f (|x|))) :=
by
  sorry

end correct_statements_l569_569749


namespace cube_edge_length_even_l569_569962

theorem cube_edge_length_even (n : ℕ) 
  (h : ∃ (color : ℕ × ℕ × ℕ → bool), 
      (∀ i j k, color (i, j, k) = tt ↔ 
                 (color (i + 1, j, k) = ff ∧ 
                  color (i, j + 1, k) = ff ∧ 
                  color (i, j, k + 1) = ff)) ∧
      (∀ i j k, color (i, j, k) = ff ↔ 
                 (color (i + 1, j, k) = tt ∧ 
                  color (i, j + 1, k) = tt ∧ 
                  color (i, j, k + 1) = tt))) :
  even n :=
begin
  sorry
end

end cube_edge_length_even_l569_569962


namespace square_intersect_line_l569_569881

-- Define the line and points
variables {L : Type*} [LinearOrder L]
variables (A B C D : L)
variables (distinct : A < B ∧ B < C ∧ C < D)

-- Statement of the theorem
theorem square_intersect_line :
  ∃ (square : Type*), 
  ∃ (sides : square → L × L),
  (sides = (λ v, match v with 
                | 0 := (A, B)
                | 1 := (C, D)
                | _ := (A, D) -- This is just a placeholder, refined construction needed
            end)) ∧
  (let (x1, y1) := sides 0 in
   let (x2, y2) := sides 1 in
   A < x1 ∧ x1 < B ∧ C < x2 ∧ x2 < D) :=
sorry

end square_intersect_line_l569_569881


namespace product_of_positive_real_part_solutions_l569_569409

theorem product_of_positive_real_part_solutions :
  (∃ roots : list (ℂ), roots = [2 * complex.exp(Complex.I * real.pi * ((2 * n + 1) / 8)) | n <- [0,1,7]] ∧
  (∀ z : ℂ, z ∈ roots → z.real_part > 0) ∧
  (∃ product_of_roots, product_of_roots = 8 * complex.exp(Complex.I * real.pi * 67.5 / 180))) := 
sorry

end product_of_positive_real_part_solutions_l569_569409


namespace skittles_problem_l569_569669

def initial_skittles : ℕ := 76
def shared_skittles : ℕ := 72
def final_skittles (initial shared : ℕ) : ℕ := initial - shared

theorem skittles_problem : final_skittles initial_skittles shared_skittles = 4 := by
  sorry

end skittles_problem_l569_569669


namespace quadratic_two_distinct_roots_l569_569415

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k*x^2 - 6*x + 9 = 0) ∧ (k*y^2 - 6*y + 9 = 0)) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_roots_l569_569415


namespace circle_line_separation_l569_569716

theorem circle_line_separation (x0 y0 r : ℝ) (h1 : x0^2 + y0^2 < r^2) :
  let d := r^2 / Real.sqrt (x0^2 + y0^2) in
  d > r :=
by
  let d := r^2 / Real.sqrt (x0^2 + y0^2)
  have : d = r^2 / Real.sqrt (x0^2 + y0^2) := rfl
  sorry

end circle_line_separation_l569_569716


namespace sum_of_youngest_and_oldest_cousins_l569_569629

theorem sum_of_youngest_and_oldest_cousins 
  (a1 a2 a3 a4 : ℕ) 
  (h_order : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4) 
  (h_mean : a1 + a2 + a3 + a4 = 36) 
  (h_median : a2 + a3 = 14) : 
  a1 + a4 = 22 :=
by sorry

end sum_of_youngest_and_oldest_cousins_l569_569629


namespace fraction_of_money_left_l569_569641

variables {m p c : ℝ}

-- given conditions
def brianna_money (m : ℝ) : Prop := m > 0
def one_sixth_for_one_fourth_CDs (m p c : ℝ) : Prop := (1 / 6) * m = (1 / 4) * c * p

-- conclusion to prove
def fraction_left_after_purchasing_CDs (m : ℝ) : ℝ :=
  m - c

theorem fraction_of_money_left (m p c : ℝ) (h1: brianna_money m) (h2: one_sixth_for_one_fourth_CDs m p c):
  fraction_left_after_purchasing_CDs m = 1 / 3 * m :=
sorry

end fraction_of_money_left_l569_569641


namespace greatest_possible_x_l569_569775

theorem greatest_possible_x (x : ℤ) (h : 3.134 * 10 ^ x < 31000) : x ≤ 3 := 
  sorry

end greatest_possible_x_l569_569775


namespace find_k_l569_569229

def condition (k : ℝ) : Prop := 24 / k = 4

theorem find_k (k : ℝ) (h : condition k) : k = 6 :=
sorry

end find_k_l569_569229


namespace students_registered_for_course_l569_569857

theorem students_registered_for_course :
  ∀ (students_present_yesterday students_absent_today: ℕ),
    students_present_yesterday = 70 →
    students_absent_today = 30 →
    let students_attended_today := 0.9 * 2 * students_present_yesterday in
    students_registered = students_attended_today + students_absent_today →
    students_registered = 156 :=
by
  intros students_present_yesterday students_absent_today h1 h2 h3
  rw [h1, h2, h3]
  sorry

end students_registered_for_course_l569_569857


namespace joeys_votes_l569_569779

theorem joeys_votes
  (M B J : ℕ) 
  (h1 : M = 66) 
  (h2 : M = 3 * B) 
  (h3 : B = 2 * (J + 3)) : 
  J = 8 := 
by 
  sorry

end joeys_votes_l569_569779


namespace max_distance_from_ellipse_to_line_l569_569799

-- Defining the scaled ellipse C2 and line l
noncomputable def ellipse_C2 (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

noncomputable def line_l (x y : ℝ) : Prop :=
  2 * x + sqrt 3 * y = 9

-- Parametric point M on the ellipse C2
def parametric_point_M (α : ℝ) : ℝ × ℝ :=
  (2 * cos α, sqrt 3 * sin α)

-- Distance from a point M to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2 * x + sqrt 3 * y - 9) / sqrt (2^2 + (sqrt 3)^2)

-- Maximum distance from M on C2 to the line l
theorem max_distance_from_ellipse_to_line :
  ∃ α : ℝ, distance_to_line (2 * cos α) (sqrt 3 * sin α) = 2 * sqrt 7 :=
sorry

end max_distance_from_ellipse_to_line_l569_569799


namespace solve_for_x0_l569_569141

def f (x : ℝ) : ℝ :=
if x ≤ 2 then x^2 + 2 else 2 * x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 ∨ x0 = -Real.sqrt 6 :=
by sorry

end solve_for_x0_l569_569141


namespace totient_fun_eq_l569_569074

-- Define the prime factors and their properties
def n := 37^2 * 3
def φ (n : ℕ) : ℕ := n * (1 - 1 / 37) * (1 - 1 / 3)

theorem totient_fun_eq :
  φ(n) = 1332 → (n = 2007 ∨ n = 2738 ∨ n = 3122) :=
sorry

end totient_fun_eq_l569_569074


namespace sum_a_b_l569_569659

theorem sum_a_b : 
  ∃ (A B : ℚ), 
    (∀ x, (x - 4) ≠ 0 ∧ (x - 5) ≠ 0 → (Bx - 17) / (x^2 - 9x + 20) = A / (x - 4) + 5 / (x - 5)) → 
    A + B = 17 / 4 := by
  sorry

end sum_a_b_l569_569659


namespace point_location_in_third_quadrant_l569_569709

def z : ℂ := -3 + 2 * Complex.i
def z_conj : ℂ := Complex.conj z
def quadrant (c: ℂ) : String := if c.re < 0 ∧ c.im > 0 then "Second quadrant"
                                else if c.re > 0 ∧ c.im > 0 then "First quadrant"
                                else if c.re < 0 ∧ c.im < 0 then "Third quadrant"
                                else if c.re > 0 ∧ c.im < 0 then "Fourth quadrant"
                                else "On axis"

theorem point_location_in_third_quadrant : quadrant z_conj = "Third quadrant" :=
by
  sorry

end point_location_in_third_quadrant_l569_569709


namespace milk_left_l569_569831

theorem milk_left (initial_milk : ℝ) (milk_james : ℝ) (milk_maria : ℝ) :
  initial_milk = 5 → milk_james = 15 / 4 → milk_maria = 3 / 4 → 
  initial_milk - (milk_james + milk_maria) = 1 / 2 :=
by
  intros h_initial h_james h_maria
  rw [h_initial, h_james, h_maria]
  -- The calculation would be performed here.
  sorry

end milk_left_l569_569831


namespace max_popsicles_l569_569840

theorem max_popsicles (budget : ℕ) (single_cost : ℕ) (box4_cost : ℕ) (box7_cost : ℕ) (box9_cost : ℕ)
                      (single_num : ℕ) (box4_num : ℕ) (box7_num : ℕ) (box9_num : ℕ) :
                      single_cost = 2 → box4_cost = 5 → box7_cost = 8 → box9_cost = 10 →
                      single_num = 1 → box4_num = 4 → box7_num = 7 → box9_num = 9 →
                      budget = 20 → 
                      (∃ (n : ℕ), n = 18 ∧ 
                      ∀ (b n1 n2 n3 n4 : ℕ), b = budget →
                                              n1 * single_cost + n2 * box4_cost + n3 * box7_cost + n4 * box9_cost ≤ b → 
                                              single_num * n1 + box4_num * n2 + box7_num * n3 + box9_num * n4 ≤ n) :=
by {
  intros single_cost_cost_eq single_box4_cost_eq single_box7_cost_eq single_box9_cost_eq 
        single_num_eq box4_num_eq box7_num_eq box9_num_eq budget_eq,
  sorry
}

end max_popsicles_l569_569840


namespace convex_quadrilateral_area_lt_a_sq_l569_569305

theorem convex_quadrilateral_area_lt_a_sq {a x y z t : ℝ} (hx : x < a) (hy : y < a) (hz : z < a) (ht : t < a) :
  (∃ S : ℝ, S < a^2) :=
sorry

end convex_quadrilateral_area_lt_a_sq_l569_569305


namespace isosceles_triangle_AHB_l569_569107

open_locale classical

variable {Point : Type}

structure Rectangle (A B C D K H : Point) : Prop :=
  (is_rectangle: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)
  (K_midpoint_AD: ∃ M : Point, M = K ∧ A ≠ D ∧ dist A M = dist M D)
  (perp_CK_BD: ∟ C K B D)
  (H_intersection_BD_CK: ∃ M : Point, M = H ∧ H ≠ B ∧ H ≠ D)

theorem isosceles_triangle_AHB {A B C D K H : Point} [euclidean_space Point] :
  Rectangle A B C D K H →
  (triangle_isosceles (triangle.mk A H B)) :=
begin
  sorry
end

end isosceles_triangle_AHB_l569_569107


namespace number_of_fish_l569_569207

theorem number_of_fish (dogs cats total pets : ℕ) (h_dogs : dogs = 43) (h_cats : cats = 34) (h_total_pets : total = 149) (h_pets_eq : pets = dogs + cats) : total - pets = 72 :=
by
  rw [h_dogs, h_cats, h_total_pets, h_pets_eq]
  show 149 - (43 + 34) = 72
  sorry

end number_of_fish_l569_569207


namespace num_boys_l569_569433
open Classical

variable (B : Type) (G : Type)
variable [Fintype B] [Fintype G]

/-- Constraints:
  - Tanya is friends with 12 boys
  - Dasha is friends with 12 boys
  - Katya is friends with 13 boys
  - Any girl is friends with at least one boy in any chosen set of 3 boys -/
variables (Tanya Dasha Katya : G)
variables (friends : G → B → Prop)

-- Tanya is friends with 12 boys
axiom Tanya_friends : ∀ b : B, friends Tanya b → Fintype.card {b' : B | friends Tanya b'} = 12

-- Dasha is friends with 12 boys
axiom Dasha_friends : ∀ b : B, friends Dasha b → Fintype.card {b' : B | friends Dasha b'} = 12

-- Katya is friends with 13 boys
axiom Katya_friends : ∀ b : B, friends Katya b → Fintype.card {b' : B | friends Katya b'} = 13

-- Any girl has at least one friend among any group of three boys
axiom at_least_one_friend : ∀ (g : G) (b1 b2 b3 : B), friends g b1 ∨ friends g b2 ∨ friends g b3

/-- The number of boys in the 6th "A" grade can be either 13 or 14 -/
theorem num_boys (b : ℕ) [Fintype.card B = b] : b = 13 ∨ b = 14 :=
  sorry

end num_boys_l569_569433


namespace isosceles_triangle_perimeter_l569_569868

theorem isosceles_triangle_perimeter :
  ∀ x y : ℝ, x^2 - 7*x + 10 = 0 → y^2 - 7*y + 10 = 0 → x ≠ y → x + x + y = 12 :=
by
  intros x y hx hy hxy
  -- Place for proof
  sorry

end isosceles_triangle_perimeter_l569_569868


namespace alice_questions_wrong_l569_569626

theorem alice_questions_wrong (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 3) 
  (h3 : c = 7) : 
  a = 8.5 := 
by
  sorry

end alice_questions_wrong_l569_569626


namespace perpendicular_lines_intersection_l569_569525

theorem perpendicular_lines_intersection (a b c d : ℝ)
    (h_perpendicular : (a / 2) * (-2 / b) = -1)
    (h_intersection1 : a * 2 - 2 * (-3) = d)
    (h_intersection2 : 2 * 2 + b * (-3) = c) :
    d = 12 := 
sorry

end perpendicular_lines_intersection_l569_569525


namespace cabinets_ratio_proof_l569_569445

-- Definitions for the conditions
def initial_cabinets : ℕ := 3
def total_cabinets : ℕ := 26
def additional_cabinets : ℕ := 5
def number_of_counters : ℕ := 3

-- Definition for the unknown cabinets installed per counter
def cabinets_per_counter : ℕ := (total_cabinets - additional_cabinets - initial_cabinets) / number_of_counters

-- The ratio to be proven
theorem cabinets_ratio_proof : (cabinets_per_counter : ℚ) / initial_cabinets = 2 / 1 :=
by
  -- Proof goes here
  sorry

end cabinets_ratio_proof_l569_569445


namespace chord_inequality_in_circle_l569_569188

theorem chord_inequality_in_circle (X K M A B C D O : Type)
  [InnerFourCircle X] [Diameter X A B C D] [Center X O]
  (a : ℝ) (h1 : A C = a) (h2 : C D = a) (h3 : D B = a)
  (h4 : Circle.radius O A = 3 / 2 * a)
  (KM_leq_3AB : 5 * KM ≤ 3 * AB) : 
  5 * distance K M ≤ 3 * distance A B := sorry

end chord_inequality_in_circle_l569_569188


namespace import_tax_paid_l569_569218

theorem import_tax_paid (total_value excess_value tax_rate tax_paid : ℝ)
  (h₁ : total_value = 2590)
  (h₂ : excess_value = total_value - 1000)
  (h₃ : tax_rate = 0.07)
  (h₄ : tax_paid = excess_value * tax_rate) : 
  tax_paid = 111.30 := by
  -- variables
  sorry

end import_tax_paid_l569_569218


namespace transform_sin_to_cos_l569_569910

theorem transform_sin_to_cos :
  ∀ (x : ℝ), cos (2 * (x - (-(π / 6)))) = sin (2 * x + π / 6) :=
by 
  intro x
  sorry

end transform_sin_to_cos_l569_569910


namespace total_money_needed_l569_569447

-- Declare John's initial amount
def john_has : ℝ := 0.75

-- Declare the additional amount John needs
def john_needs_more : ℝ := 1.75

-- The theorem statement that John needs a total of $2.50
theorem total_money_needed : john_has + john_needs_more = 2.5 :=
  by
  sorry

end total_money_needed_l569_569447


namespace triangle_inequality_l569_569845

noncomputable def alpha := 2

theorem triangle_inequality {A B C : Point} 
    (hA hB hC : ℝ) (mA mB mC : ℝ) 
    (hA_def : hA = altitude A B C)
    (hB_def : hB = altitude B A C)
    (hC_def : hC = altitude C A B)
    (mA_def : mA = median A B C)
    (mB_def : mB = median B A C)
    (mC_def : mC = median C A B)
    : max hA (max hB hC) ≤ alpha * min mA (min mB mC) :=
sorry

end triangle_inequality_l569_569845


namespace positive_difference_of_solutions_l569_569681

theorem positive_difference_of_solutions:
  ∀ (s : ℝ), s ≠ -3 → (s^2 - 5*s - 24) / (s + 3) = 3*s + 10 →
  abs (-1 - (-27)) = 26 :=
by
  sorry

end positive_difference_of_solutions_l569_569681


namespace find_vector_n_l569_569702

open Real

noncomputable def vec_m : ℝ × ℝ := (1, 1)
noncomputable def vec_q : ℝ × ℝ := (1, 0)
noncomputable def vec_p (A : ℝ) : ℝ × ℝ := (2 * sin A, 4 * cos (A / 2) ^ 2)

theorem find_vector_n (A : ℝ) :
  ∃ vec_n : ℝ × ℝ, vec_n.angle vec_m = 3 * π / 4 ∧
  vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2 = -1 ∧
  vec_n.1 * vec_q.1 + vec_n.2 * vec_q.2 = 0 ∧
  ‖2 * vec_n.1 + (vec_p A).1, 2 * vec_n.2 + (vec_p A).2‖ = 2 :=
sorry

end find_vector_n_l569_569702


namespace avg_monthly_growth_rate_20_percent_price_reduction_4_yuan_l569_569863

-- Part 1: Average Monthly Production Growth Rate

def monthly_growth_rate (initial_prod : ℕ) (final_prod : ℕ) (months : ℕ) (growth_rate : ℝ) : Prop :=
  final_prod = initial_prod * (1 + growth_rate) ^ months

theorem avg_monthly_growth_rate_20_percent :
  monthly_growth_rate 500 720 2 0.2 :=
sorry

-- Part 2: Price Reduction for Each "Bing Duan Duan"

def daily_profit_condition (initial_sales : ℕ) (initial_profit_per_item : ℝ) (price_reduction : ℝ) 
  (additional_sales_per_reduction : ℕ) (required_profit : ℝ) : Prop :=
  (initial_profit_per_item - price_reduction) * 
  (initial_sales + additional_sales_per_reduction * (nat.floor price_reduction.to_nat)) = required_profit

theorem price_reduction_4_yuan (initial_sales : ℕ) (initial_profit_per_item : ℝ) 
  (additional_sales_per_reduction : ℕ) (required_profit : ℝ) (max_reduction : ℝ) :
  initial_sales = 20 → initial_profit_per_item = 40 → additional_sales_per_reduction = 5 → 
  required_profit = 1440 → max_reduction = 10 → 
  daily_profit_condition 20 40 4 5 1440 :=
sorry

end avg_monthly_growth_rate_20_percent_price_reduction_4_yuan_l569_569863


namespace gcd_90_252_eq_18_l569_569196

theorem gcd_90_252_eq_18 : Nat.gcd 90 252 = 18 := 
sorry

end gcd_90_252_eq_18_l569_569196


namespace stamps_total_l569_569158

theorem stamps_total :
  ∃ x, x ≡ 1 [MOD 2] ∧
       x ≡ 1 [MOD 3] ∧
       x ≡ 3 [MOD 5] ∧
       x ≡ 7 [MOD 9] ∧
       150 < x ∧ x ≤ 300 ∧
       x = 223 := 
begin
  use 223,
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 2 1 (by norm_num)), },
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 3 1 (by norm_num)), },
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 5 3 (by norm_num)), },
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 9 7 (by norm_num)), },
  split,
  { norm_num, },
  split,
  { norm_num, 
  { refl, },
  }
sorry
end

end stamps_total_l569_569158


namespace circle_radius_l569_569957

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 120 * π) : r = 10 :=
sorry

end circle_radius_l569_569957


namespace limit_na_n_l569_569297

def L (x : ℝ) : ℝ := x - x^2 / 2

def a_n (n : ℕ) : ℝ := (L^[2 * n]) (25 / n)

theorem limit_na_n : tendsto (λ n : ℕ, n * a_n n) atTop (𝓝 (50 / 27)) :=
sorry

end limit_na_n_l569_569297


namespace people_landed_in_virginia_l569_569276

def initial_passengers : ℕ := 124
def texas_out : ℕ := 58
def texas_in : ℕ := 24
def north_carolina_out : ℕ := 47
def north_carolina_in : ℕ := 14
def crew_members : ℕ := 10

def final_passengers := initial_passengers - texas_out + texas_in - north_carolina_out + north_carolina_in
def total_people_landed := final_passengers + crew_members

theorem people_landed_in_virginia : total_people_landed = 67 :=
by
  sorry

end people_landed_in_virginia_l569_569276


namespace second_offset_length_l569_569009

-- Definitions based on the given conditions.
def diagonal : ℝ := 24
def offset1 : ℝ := 9
def area_quad : ℝ := 180

-- Statement to prove the length of the second offset.
theorem second_offset_length :
  ∃ h : ℝ, (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * h = area_quad ∧ h = 6 :=
by
  sorry

end second_offset_length_l569_569009


namespace fraction_of_single_female_students_l569_569940

variables (total_students : ℕ) (male_students married_students married_male_students female_students single_female_students : ℕ)

-- Given conditions
def condition1 : male_students = (7 * total_students) / 10 := sorry
def condition2 : married_students = (3 * total_students) / 10 := sorry
def condition3 : married_male_students = male_students / 7 := sorry

-- Derived conditions
def condition4 : female_students = total_students - male_students := sorry
def condition5 : married_female_students = married_students - married_male_students := sorry
def condition6 : single_female_students = female_students - married_female_students := sorry

-- The proof goal
theorem fraction_of_single_female_students 
  (h1 : male_students = (7 * total_students) / 10)
  (h2 : married_students = (3 * total_students) / 10)
  (h3 : married_male_students = male_students / 7)
  (h4 : female_students = total_students - male_students)
  (h5 : married_female_students = married_students - married_male_students)
  (h6 : single_female_students = female_students - married_female_students) :
  (single_female_students : ℚ) / (female_students : ℚ) = 1 / 3 :=
sorry

end fraction_of_single_female_students_l569_569940


namespace clarissa_copies_needed_l569_569647

-- Define the given conditions
def manuscript_pages : ℕ := 400
def cost_per_page : ℚ := 0.05
def cost_per_binding : ℚ := 5.00
def total_cost : ℚ := 250.00

-- Calculate the total cost for one manuscript
def cost_per_copy_and_bind : ℚ := cost_per_page * manuscript_pages + cost_per_binding

-- Define number of copies needed
def number_of_copies_needed : ℚ := total_cost / cost_per_copy_and_bind

-- Prove number of copies needed is 10
theorem clarissa_copies_needed : number_of_copies_needed = 10 := 
by 
  -- Implementing the proof steps would go here
  sorry

end clarissa_copies_needed_l569_569647


namespace complex_min_value_l569_569030

theorem complex_min_value (z : ℂ) (hz : complex.norm z = 2) : 
  complex.norm (z + 3 - 4 * complex.I) = real.sqrt (13 + 6 * real.sqrt 7) :=
sorry

end complex_min_value_l569_569030


namespace four_digit_cubes_divisible_by_16_l569_569392

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l569_569392


namespace average_of_three_l569_569866

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end average_of_three_l569_569866


namespace students_diff_l569_569169

-- Define the conditions
def M : ℕ := 457
def B : ℕ := 394

-- Prove the final answer
theorem students_diff : M - B = 63 := by
  -- The proof is omitted here with a sorry placeholder
  sorry

end students_diff_l569_569169


namespace xiao_li_more_stable_l569_569646

def average_xiao_li : ℝ := 95
def average_xiao_zhang : ℝ := 95

def variance_xiao_li : ℝ := 0.55
def variance_xiao_zhang : ℝ := 1.35

theorem xiao_li_more_stable : 
  variance_xiao_li < variance_xiao_zhang :=
by
  sorry

end xiao_li_more_stable_l569_569646


namespace hyperbola_eccentricity_range_l569_569474

section

variables {a b c e : ℝ} (ha : a > 0) (hb : b > 0)
variable (h1 : ∀ x y : ℝ, (x, y) \in set_of (λ p, 
    (p.1^2) / (a^2) - (p.2^2) / (b^2) = 1))

variable (h2 : ∃ F : ℝ, F = c ∧ F = 0)

variable (h3 : ∃ P Q : ℝ × ℝ,
    P = (c, b^2 / a) ∧ Q = (c, -b^2 / a) ∧
    ∀ x, x ∈ line_through (c, b^2 / a) (c, -b^2 / a) ∧
    ∃ B : ℝ × ℝ, B.1 = c ∧ B.2 = 0)

variable (h4 : dist (c, 0) (c, b^2 / a) < 2 * (a + c))

def range_of_eccentricity : Prop := 
  1 < e ∧ e < sqrt 3

theorem hyperbola_eccentricity_range : range_of_eccentricity ha hb h1 h2 h3 h4 :=
sorry

end hyperbola_eccentricity_range_l569_569474


namespace perpendicular_slope_l569_569689

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end perpendicular_slope_l569_569689


namespace conjugate_quadrant_l569_569358

open Complex

theorem conjugate_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) * z = 3 + Complex.I) : 
  let w := conj z
  w.re > 0 ∧ w.im > 0 := 
sorry

end conjugate_quadrant_l569_569358


namespace sequence_general_term_l569_569735

-- Define the sequence {a_n} and sum of first n terms S_n
def sequence (a : ℕ → ℕ) := ∀ n : ℕ, 0 < a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) := ∀ n : ℕ, S n = ∑ i in finset.range n, a i

-- Define the main condition: arithmetic mean == geometric mean for given terms
def mean_condition (a : ℕ → ℕ) (S : ℕ → ℕ) := 
  ∀ n : ℕ, 0 < S n → (a n + 2) / 2 = real.sqrt (S n * 2)

-- The goal: prove that the sequence term a_n is 4n - 2
theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : sequence a) (h2 : sum_seq a S) (h3 : mean_condition a S) :
  ∀ n : ℕ, a n = 4 * n - 2 :=
by
  sorry

end sequence_general_term_l569_569735


namespace estimate_fish_population_l569_569022

theorem estimate_fish_population :
  ∀ (x : ℕ), (1200 / x = 100 / 1000) → x = 12000 := by
  sorry

end estimate_fish_population_l569_569022


namespace next_time_10_10_11_15_l569_569607

noncomputable def next_time_angle_x (current_time : ℕ × ℕ) (x : ℕ) : ℕ × ℕ := sorry

theorem next_time_10_10_11_15 :
  ∀ (x : ℕ), next_time_angle_x (10, 10) 115 = (11, 15) := sorry

end next_time_10_10_11_15_l569_569607


namespace kan_krao_park_walkways_l569_569816

-- Definitions for the given conditions
structure Park (α : Type*) := 
  (entrances : Finset α)
  (walkways : α → α → Prop)
  (brick_paved : α → α → Prop)
  (asphalt_paved : α → α → Prop)
  (no_three_intersections : ∀ (x y z w : α), x ≠ y → y ≠ z → z ≠ w → w ≠ x → (walkways x y ∧ walkways z w) → ¬ (walkways x z ∧ walkways y w))

-- Conditions based on the given problem
variables {α : Type*} [Finite α] [DecidableRel (@walkways α)]
variable (p : Park α)
variables [Fintype α]

-- Translate conditions to definitions
def has_lotuses (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := p x y ∧ p x y
def has_waterlilies (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := (p x y ∧ q x y) ∨ (q x y ∧ p x y)
def is_lit (p : α → α → Prop) (q : α → α → Prop) : Prop := ∃ (x y : α), x ≠ y ∧ (has_lotuses p q x y ∧ has_lotuses p q x y ∧ ∃ sz, sz ≥ 45)

-- Mathematically equivalent proof problem
theorem kan_krao_park_walkways (p : Park α) :
  (∃ walkways_same_material : α → α → Prop, ∃ (lit_walkways : Finset (α × α)), lit_walkways.card ≥ 11) :=
sorry

end kan_krao_park_walkways_l569_569816


namespace minimum_value_l569_569027

theorem minimum_value (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1)
  (h4 : a * b + b * c + c * a = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = (9 + 3 * Real.sqrt 3) / 2 :=
by
  sorry

end minimum_value_l569_569027


namespace union_of_sets_l569_569125

variable (a b : ℝ)
def M : set ℝ := {3, 2^a}
def N : set ℝ := {a, b}

theorem union_of_sets (h : M a = {3, 2^a}) (hN : N a b = {a, b}) (inter_cond : M a ∩ N a b = {2}) :
  M a ∪ N a b = {1, 2, 3} :=
by sorry

end union_of_sets_l569_569125


namespace smallest_positive_period_l569_569013

theorem smallest_positive_period (ω : ℝ) (hω : ω = 3) : (T : ℝ) (hT : T = 2 * Real.pi / ω) :=
  by
    have h_period : T = 2 * Real.pi / 3 :=
      by sorry
    exact h_period

end smallest_positive_period_l569_569013


namespace final_problem_l569_569103

-- Declare the sequences
def a : ℕ → ℕ := λ n, n + 2
def b (n : ℕ) : ℕ := n * 2^(a n - 2)

-- The conditions given
def a_2 : Prop := a 2 = 4
def sum_first_4_a : Prop := (∑ i in finset.range 4, a (i + 1)) = 18

-- The goal to show
def general_formula : Prop := ∀ n, a n = n + 2

-- Define T_n 
def T (n : ℕ) : ℕ := ∑ i in finset.range n, b (i + 1)

-- The goal to show
def sum_T : Prop := ∀ n, T n = (n - 1) * 2^(n+1) + 2

-- The final statement combining conditions and goals
theorem final_problem : a_2 ∧ sum_first_4_a → general_formula ∧ sum_T :=
begin
  sorry
end

end final_problem_l569_569103


namespace quarter_circle_no_closed_deadend_l569_569538

theorem quarter_circle_no_closed_deadend
  (R : ℝ) (segments : list (set (ℝ × ℝ)))
  (h1 : ∀ s ∈ segments, ∃ (c : ℝ × ℝ) (angle : ℝ), s = { p | ∃ θ, θ ∈ Icc angle (angle + π/2) ∧ p = (c.1 + R * cos θ, c.2 + R * sin θ) })
  (h2 : ∃ chain : list (set (ℝ × ℝ)), list.chain' (λ s1 s2, (s1 ∩ s2).nonempty) chain ∧ ∀ s ∈ chain, s ∈ segments)
  (h3 : ∃ s1 s2 ∈ segments, s1 ∩ s2 = ∅)
  : ¬ ∃ (path : list (set (ℝ × ℝ))), list.chain' (λ s1 s2, (s1 ∩ s2).nonempty) path ∧ path.head = path.last ∧ (∀ s ∈ path, s ∈ segments) ∧ (path.head ∩ path.nth 1 = ∅) :=
sorry

end quarter_circle_no_closed_deadend_l569_569538


namespace count_P_l569_569453

def A : Set ℕ := {n | (3 ∣ n = false) ∨ (9 ∣ n)}
def B : Set ℕ := {n | (2 ∣ n = false) ∨ (8 ∣ n)}
def P : Set ℕ := {n | n ∈ A ∧ n ∈ B ∧ 1 ≤ n ∧ n ≤ 2016}

theorem count_P : |P| = 980 := by
  sorry

end count_P_l569_569453


namespace min_neg_signs_to_zero_sum_l569_569839

-- Definition of the set of numbers on the clock face
def clock_face_numbers : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Sum of the clock face numbers
def sum_clock_face_numbers := clock_face_numbers.sum

-- Given condition that the sum of clock face numbers is 78
axiom sum_clock_face_numbers_is_78 : sum_clock_face_numbers = 78

-- Definition of the function to calculate the minimum number of negative signs needed
def min_neg_signs_needed (numbers : List ℤ) (target : ℤ) : ℕ :=
  sorry -- The implementation is omitted

-- Theorem stating the goal of our problem
theorem min_neg_signs_to_zero_sum : min_neg_signs_needed clock_face_numbers 39 = 4 :=
by
  -- Proof is omitted
  sorry

end min_neg_signs_to_zero_sum_l569_569839


namespace value_of_x_l569_569531

/-- A set of numbers where the mean, median, and mode must all be equal. -/
def number_set : Set ℕ := {2, 3, 3, 4, 5}

/-- The function to calculate the mean of a set of numbers. -/
noncomputable def mean (s : Set ℕ) (x : ℕ) : ℚ :=
  (s.sum + x) / (s.size + 1)

/-- The function to calculate the median of a set of numbers. -/
noncomputable def median (s : Set ℕ) (x : ℕ) : ℚ := by
  let sorted_set := (s.toList ++ [x]).sort (· ≤ ·)
  let n := sorted_set.length
  if n % 2 == 0 then
    (sorted_set.get (n / 2 - 1) + sorted_set.get (n / 2)) / 2
  else
    sorted_set.get (n / 2)

/-- The function to calculate the mode of a set of numbers. -/
noncomputable def mode (s : Set ℕ) (x : ℕ) : ℕ := by
  let freq_map := (s.toList ++ [x]).foldl (λ m a => m.insert a ((m.find a).getOrElse 0 + 1)) (RBMap.empty ℕ ℕ compare)
  let max_freq := freq_map.foldl (λ acc p => max acc p.2) 0
  (freq_map.toList.filter (λ p => p.2 == max_freq)).get 0 |>.1

/-- The proof problem that x must equal 1. -/
theorem value_of_x :
  ∃ x : ℕ, mean number_set x = 3 ∧ median number_set x = 3 ∧ mode number_set x = 3 :=
by
  use 1
  -- Detailed proofs must be provided here
  sorry

end value_of_x_l569_569531


namespace number_of_factors_of_M_l569_569397

theorem number_of_factors_of_M :
  let M := 2^2 * 3^3 * 5^2 * 7^1 in
  let count_factors (n : ℕ) := (n + 1) in
  count_factors 2 * count_factors 3 * count_factors 2 * count_factors 1 = 72 := by
{
  let M := 2^2 * 3^3 * 5^2 * 7^1
  have f2 := 2 + 1
  have f3 := 3 + 1
  have f5 := 2 + 1
  have f7 := 1 + 1
  show f2 * f3 * f5 * f7 = 72, by { sorry }
}

end number_of_factors_of_M_l569_569397


namespace child_ticket_cost_is_2_l569_569247

-- Define the conditions
def adult_ticket_cost : ℕ := 5
def total_tickets_sold : ℕ := 85
def total_revenue : ℕ := 275
def adult_tickets_sold : ℕ := 35

-- Define the function to calculate child ticket cost
noncomputable def child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets_sold : ℕ) (total_revenue : ℕ) (adult_tickets_sold : ℕ) : ℕ :=
  let total_adult_revenue := adult_tickets_sold * adult_ticket_cost
  let total_child_revenue := total_revenue - total_adult_revenue
  let child_tickets_sold := total_tickets_sold - adult_tickets_sold
  total_child_revenue / child_tickets_sold

theorem child_ticket_cost_is_2 : child_ticket_cost adult_ticket_cost total_tickets_sold total_revenue adult_tickets_sold = 2 := 
by
  -- This is a placeholder for the actual proof which we can fill in separately.
  sorry

end child_ticket_cost_is_2_l569_569247


namespace find_a_l569_569712

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1)
  (h_diff : |a^2 - a| = 6) : a = 3 :=
sorry

end find_a_l569_569712


namespace min_table_sum_l569_569793

theorem min_table_sum (A : Fin 5 → Fin 5 → ℕ)
  (distinct_sums : ∀ (i j : Fin 5), i ≠ j → (∑ k : Fin 5, A i k) ≠ (∑ k : Fin 5, A j k) ∧ (∑ k : Fin 5, A k i) ≠ (∑ k : Fin 5, A k j)) :
  ∑ i j, A i j ≥ 48 :=
sorry

end min_table_sum_l569_569793


namespace rotation_preserves_position_l569_569715

-- Define the initial shapes and positions
inductive Shape where
  | Triangle 
  | SmallerCircle 
  | Rectangle 
  deriving DecidableEq

structure Position where
  shape : Shape
  -- Add more fields if necessary, e.g., coordinates.

constant initial_position : Shape → Position

-- Define position after rotation
def rotate_position (pos : Position) : Position :=
  match pos.shape with
  | Shape.Triangle => {shape := Shape.SmallerCircle}
  | Shape.SmallerCircle => {shape := Shape.Rectangle}
  | Shape.Rectangle => {shape := Shape.Triangle}

-- The theorem to be proven
theorem rotation_preserves_position :
  ∀ (s : Shape), rotate_position (initial_position s) = initial_position (rotate_position s.shape) :=
by
  intros s
  sorry

end rotation_preserves_position_l569_569715


namespace bijection_representations_l569_569167

theorem bijection_representations (n : ℕ) :
  (∃ f : {l : list ℕ // l.sum = n ∧ l.all (λ x, x = 1 ∨ x = 2)} → {l : list ℕ // l.sum = n + 2 ∧ l.all (λ x, 2 ≤ x)},
     bijective f) :=
sorry

end bijection_representations_l569_569167


namespace find_a_l569_569752

def point_of_tangency (x0 y0 a : ℝ) : Prop :=
  (x0 - y0 - 1 = 0) ∧ (y0 = a * x0^2) ∧ (2 * a * x0 = 1)

theorem find_a (x0 y0 a : ℝ) (h : point_of_tangency x0 y0 a) : a = 1/4 :=
by
  sorry

end find_a_l569_569752


namespace opposite_of_2023_is_neg_2023_l569_569888

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end opposite_of_2023_is_neg_2023_l569_569888


namespace pipeB_fill_time_l569_569565

noncomputable def time_to_fill_pipeB (T_A T_combined T_B : ℝ) : Prop :=
  (1 / T_A) + (1 / T_B) = 1 / T_combined

theorem pipeB_fill_time :
  ∃ T_B : ℝ, T_B ≈ 14.9 ∧ time_to_fill_pipeB 12 6.67 T_B :=
begin
  sorry
end

end pipeB_fill_time_l569_569565


namespace speed_first_hour_l569_569901

theorem speed_first_hour (x : ℝ) :
  (∃ x, (x + 45) / 2 = 65) → x = 85 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  sorry

end speed_first_hour_l569_569901


namespace find_standard_equation_and_t_range_l569_569036

variables {k t : ℝ} {a b : ℝ} {x₁ x₂ y₁ y₂ : ℝ}

def is_ellipse := (a > 0) ∧ (b > 0) ∧ (a > b)

def point_on_ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity :=  e = (sqrt 2) / 2

def line_through_E (k t : ℝ) (x : ℝ) := y = k * x + t

def slope_condition := (λ λ, ∀ k, k₁ + k₂ = λ * k ∧ λ ∈ [2, 4])

theorem find_standard_equation_and_t_range:
  is_ellipse →
  eccentricity →
  point_on_ellipse (sqrt 2) (-1) →
  ∃ t, ∀ λ, slope_condition λ → (a = 2 ∧ b = sqrt 2 ∧ t ∈ [-1, 1]) :=
sorry

end find_standard_equation_and_t_range_l569_569036


namespace arithmetic_sequences_count_l569_569035

theorem arithmetic_sequences_count :
  ∃ (n a d : ℕ), n ≥ 3 ∧ (∑ i in range n, (a + i * d)) = 97^2 ∧ (a ≥ 0) ∧ (d ≥ 0) ∧
  (card { (n, a, d) | n ≥ 3 ∧ (∑ i in range n, (a + i * d)) = 97^2 ∧ (a ≥ 0) ∧ (d ≥ 0) } = 4) :=
sorry

end arithmetic_sequences_count_l569_569035


namespace model_represents_feet_l569_569528

def height_statue : ℝ := 120
def height_model : ℝ := 6
def feet_per_inch_model : ℝ := height_statue / height_model

theorem model_represents_feet : feet_per_inch_model = 20 := 
by
  sorry

end model_represents_feet_l569_569528


namespace max_f_value_cos_value_l569_569057

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - 2 * cos x ^ 2 + 1

theorem max_f_value : ∃ x, ∀ y, f y ≤ f x ∧ f x = sqrt 2 :=
  sorry

theorem cos_value (θ : ℝ) (h : f θ = 3 / 5) :
  cos (2 * (π / 4 - 2 * θ)) = 16 / 25 :=
  sorry

end max_f_value_cos_value_l569_569057


namespace average_score_of_soldier_l569_569662

-- Definitions for the conditions
def shots_10 := 6
def shots_9 := 1
def shots_8 := 3
def score_10 := 10
def score_9 := 9
def score_8 := 8
def total_shots := 10

-- Statement of the problem
theorem average_score_of_soldier : 
  (shots_10 * score_10 + shots_9 * score_9 + shots_8 * score_8) / total_shots = 9.3 := 
by
  sorry

end average_score_of_soldier_l569_569662


namespace proof_of_x_value_l569_569402

noncomputable def alpha := 177 / (2994 * 14.5)

noncomputable def x : ℝ := (29.94 * alpha * 1.45)^2 - Real.sin (Real.pi / 4)

theorem proof_of_x_value : x = 280622.061 := 
by
  -- This is where the proof would go
  sorry

end proof_of_x_value_l569_569402


namespace fraction_of_journey_covered_by_bus_l569_569975

theorem fraction_of_journey_covered_by_bus :
  let D := 129.9999999999999
  let fraction_train := 3 / 5
  let train_distance := fraction_train * D
  let walk_distance := 6.5
  let bus_distance := D - train_distance - walk_distance
  (bus_distance / D) = 7 / 20 :=
by 
  let D := 129.9999999999999
  let fraction_train := 3 / 5
  let train_distance := fraction_train * D
  let walk_distance := 6.5
  let bus_distance := D - train_distance - walk_distance
  show (bus_distance / D) = 7 / 20
  sorry

end fraction_of_journey_covered_by_bus_l569_569975


namespace slope_of_perpendicular_line_l569_569685

noncomputable def perpendicular_slope : ℚ :=
  let slope := (3/4 : ℚ) in
  - (1 / slope)

theorem slope_of_perpendicular_line (a b c : ℚ) (h : a = 3 ∧ b = -4 ∧ c = 8) : 
  perpendicular_slope = - (4 / 3) :=
by {
  sorry
}

end slope_of_perpendicular_line_l569_569685


namespace max_f_max_g_pow_f_l569_569750

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^2 + 7 * x + 14)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 5 * x + 10) / (x^2 + 5 * x + 20)

theorem max_f : ∀ x : ℝ, f x ≤ 2 := by
  intro x
  sorry

theorem max_g_pow_f : ∀ x : ℝ, g x ^ f x ≤ 9 := by
  intro x
  sorry

end max_f_max_g_pow_f_l569_569750


namespace octahedron_tetrahedron_surface_area_ratio_l569_569265

theorem octahedron_tetrahedron_surface_area_ratio 
  (s : ℝ) 
  (h₁ : s = 1)
  (A_octahedron : ℝ := 2 * Real.sqrt 3)
  (A_tetrahedron : ℝ := Real.sqrt 3)
  (h₂ : A_octahedron = 2 * Real.sqrt 3 * s^2 / 2 * Real.sqrt 3 * (1/4) * s^2) 
  (h₃ : A_tetrahedron = Real.sqrt 3 * s^2 / 4)
  :
  A_octahedron / A_tetrahedron = 2 := 
by
  sorry

end octahedron_tetrahedron_surface_area_ratio_l569_569265


namespace sum_reciprocals_of_roots_on_unit_circle_eq_neg_a_l569_569127

noncomputable def polynomial (a b c d e : ℝ) : Polynomial ℂ :=
  Polynomial.C e + Polynomial.X * (Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X * (Polynomial.C a + Polynomial.X))))

theorem sum_reciprocals_of_roots_on_unit_circle_eq_neg_a 
  (a b c d e : ℝ)
  (h : ∀ z : ℂ, Polynomial.eval z (polynomial a b c d e) = 0 → |z| = 1) :
  (∑ z in (Multiset.filter (λ x, Polynomial.eval x (polynomial a b c d e) = 0) (Polynomial.roots (polynomial a b c d e))), z⁻¹) = -a :=
sorry

end sum_reciprocals_of_roots_on_unit_circle_eq_neg_a_l569_569127


namespace last_two_digits_7_pow_2017_l569_569837

noncomputable def last_two_digits_of_pow :=
  ∀ n : ℕ, ∃ (d : ℕ), d < 100 ∧ 7^n % 100 = d

theorem last_two_digits_7_pow_2017 : ∃ (d : ℕ), d = 7 ∧ 7^2017 % 100 = d :=
by
  sorry

end last_two_digits_7_pow_2017_l569_569837


namespace first_term_proof_l569_569277

noncomputable def first_term_of_geometric_series (r : ℚ) (S : ℚ) : ℚ :=
  let a := S * (1 - r)
  in 3 * a / 4  -- simplifying the step directly

theorem first_term_proof :
  first_term_of_geometric_series (-1 / 3) 27 = 36 :=
by
  -- We define necessary geometric series sum definition based on the conditions
  let a := first_term_of_geometric_series (-1 / 3) 27 
  change a = 36
  -- This change asserts what we conclude earlier in plain form
  sorry

end first_term_proof_l569_569277


namespace third_year_award_count_l569_569154

-- Define the variables and conditions
variables (x x1 x2 x3 x4 x5 : ℕ)

-- The conditions and definition for the problem
def conditions : Prop :=
  (x1 = x) ∧
  (x5 = 3 * x) ∧
  (x1 < x2) ∧
  (x2 < x3) ∧
  (x3 < x4) ∧
  (x4 < x5) ∧
  (x1 + x2 + x3 + x4 + x5 = 27)

-- The theorem statement
theorem third_year_award_count (h : conditions x x1 x2 x3 x4 x5) : x3 = 5 :=
sorry

end third_year_award_count_l569_569154


namespace book_distribution_l569_569980

theorem book_distribution {n : ℕ} (h : n = 8) :
  (2^n - 2) = 254 :=
by {
  rw h,
  norm_num,
  sorry
}

end book_distribution_l569_569980


namespace biker_distance_and_speed_l569_569583

variable (D V : ℝ)

theorem biker_distance_and_speed (h1 : D / 2 = V * 2.5)
                                  (h2 : D / 2 = (V + 2) * (7 / 3)) :
  D = 140 ∧ V = 28 :=
by
  sorry

end biker_distance_and_speed_l569_569583


namespace three_children_meet_l569_569151

theorem three_children_meet 
  (children : Finset ℕ)
  (visited_times : ℕ → ℕ)
  (meet_at_stand : ℕ → ℕ → Prop)
  (h_children_count : children.card = 7)
  (h_visited_times : ∀ c ∈ children, visited_times c = 3)
  (h_meet_pairwise : ∀ (c1 c2 : ℕ), c1 ∈ children → c2 ∈ children → c1 ≠ c2 → meet_at_stand c1 c2) :
  ∃ (t : ℕ), ∃ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
  c1 ∈ children ∧ c2 ∈ children ∧ c3 ∈ children ∧ 
  meet_at_stand c1 t ∧ meet_at_stand c2 t ∧ meet_at_stand c3 t := 
sorry

end three_children_meet_l569_569151


namespace opposite_of_2023_is_neg_2023_l569_569887

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end opposite_of_2023_is_neg_2023_l569_569887


namespace find_constants_l569_569008

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, (8 * x + 1) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) → 
  A = 33 / 4 ∧ B = -19 / 4 ∧ C = -17 / 2 :=
by 
  intro h
  sorry

end find_constants_l569_569008


namespace num_not_divisible_by_5_and_7_lt_1000_l569_569884

theorem num_not_divisible_by_5_and_7_lt_1000 : ∀ (total_less_1000 divisible_by_5 divisible_by_7 divisible_by_35 : ℕ), 
  total_less_1000 = 999 →
  divisible_by_5 = 199 →
  divisible_by_7 = 142 →
  divisible_by_35 = 28 →
  total_less_1000 - (divisible_by_5 + divisible_by_7 - divisible_by_35) = 686 :=
by {
  intros total_less_1000 divisible_by_5 divisible_by_7 divisible_by_35 h_total h_5 h_7 h_35,
  rw [h_total, h_5, h_7, h_35],
  exact rfl,
}

end num_not_divisible_by_5_and_7_lt_1000_l569_569884


namespace boat_b_takes_less_time_l569_569911

theorem boat_b_takes_less_time (A_speed_still : ℝ) (B_speed_still : ℝ)
  (A_current : ℝ) (B_current : ℝ) (distance_downstream : ℝ)
  (A_speed_downstream : A_speed_still + A_current = 26)
  (B_speed_downstream : B_speed_still + B_current = 28)
  (A_time : A_speed_still + A_current = 26 → distance_downstream / (A_speed_still + A_current) = 4.6154)
  (B_time : B_speed_still + B_current = 28 → distance_downstream / (B_speed_still + B_current) = 4.2857) :
  distance_downstream / (B_speed_still + B_current) < distance_downstream / (A_speed_still + A_current) :=
by sorry

end boat_b_takes_less_time_l569_569911


namespace equation_of_perpendicular_line_l569_569875

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y = 0

-- Define the perpendicular line
def perpendicular_line (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the center of the circle
def center_circle : ℝ × ℝ := (-1, 0)

-- Define the slope of the given line
def slope_given_line : ℝ := -1

-- Define the slope of the perpendicular line
def slope_perpendicular_line : ℝ := 1

-- Define the condition of equation of line passing through a point with a given slope
def line_through_point (P : ℝ × ℝ) (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y - P.2 = m * (x - P.1)

-- Define the equation we need to prove
def target_line (x y : ℝ) : Prop := x - y + 1 = 0

-- The theorem we need to prove
theorem equation_of_perpendicular_line :
  target_line = (line_through_point center_circle slope_perpendicular_line) :=
by
  sorry

end equation_of_perpendicular_line_l569_569875


namespace four_digit_number_l569_569435

-- Define different Chinese characters as different digits
variables (望 子 成 龙 : ℕ)

-- Define the equations according to the problem
def valid_translation : Prop :=
  望 + 子 + 成 + 龙 = 48 ∧
  龙 = 8 ∧
  成 = 6 ∧
  子 = 4 ∧
  望 = 1 ∧
  望 ≠ 子 ∧ 望 ≠ 成 ∧ 望 ≠ 龙 ∧ 子 ≠ 成 ∧ 子 ≠ 龙 ∧ 成 ≠ 龙

-- The statement we need to prove
theorem four_digit_number : valid_translation 望 子 成 龙 → 望 * 1000 + 子 * 100 + 成 * 10 + 龙 = 1468 :=
by
  sorry

end four_digit_number_l569_569435


namespace black_ball_probability_l569_569777

theorem black_ball_probability :
  let P_red := 0.42
  let P_white := 0.28
  let total_probability := 1
  ∃ P_black : ℝ, P_black = total_probability - P_red - P_white ∧ P_black = 0.30 :=
begin
  sorry
end

end black_ball_probability_l569_569777


namespace calen_more_pencils_l569_569644

def calen_pencils (C B D: ℕ) :=
  D = 9 ∧
  B = 2 * D - 3 ∧
  C - 10 = 10

theorem calen_more_pencils (C B D : ℕ) (h : calen_pencils C B D) : C = B + 5 :=
by
  obtain ⟨hD, hB, hC⟩ := h
  simp only [hD, hB, hC]
  sorry

end calen_more_pencils_l569_569644


namespace sarah_jim_ratio_l569_569554

theorem sarah_jim_ratio
  (Tim_toads : ℕ)
  (hTim : Tim_toads = 30)
  (Jim_toads : ℕ)
  (hJim : Jim_toads = Tim_toads + 20)
  (Sarah_toads : ℕ)
  (hSarah : Sarah_toads = 100) :
  Sarah_toads / Jim_toads = 2 :=
by
  sorry

end sarah_jim_ratio_l569_569554


namespace four_digit_cubes_divisible_by_16_l569_569393

theorem four_digit_cubes_divisible_by_16 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.finite
  ∧ ∃ n, n = 3 ∧ {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.card = n := 
by
  -- The proof steps would go here.
  sorry

end four_digit_cubes_divisible_by_16_l569_569393


namespace opposite_of_2023_l569_569889

theorem opposite_of_2023 : ∃ y : ℤ, 2023 + y = 0 ∧ y = -2023 :=
by
  use -2023
  split
  · linarith
  · refl

end opposite_of_2023_l569_569889


namespace range_of_m_correct_l569_569362

open Real

-- Define the function f
def f (x m : ℝ) := x^2 + m * x - 1

-- Define the condition that for any x in [m, m+1], f(x) < 0
def condition (m : ℝ) : Prop := ∀ x ∈ Icc m (m+1), f x m < 0

-- Define the range of m
def range_of_m : Set ℝ := Ioo (-sqrt 2 / 2) 0

-- The main statement that states the equivalence
theorem range_of_m_correct (m : ℝ) : condition m ↔ m ∈ range_of_m := 
sorry

end range_of_m_correct_l569_569362


namespace slope_perpendicular_to_given_line_l569_569687

-- Define the given line
def given_line (x y : ℝ) := 3 * x - 4 * y = 8

-- Define the slope-intercept form of the line and its slope
def slope_of_given_line := (3 : ℝ) / (4 : ℝ)

-- Define the slope of the line perpendicular to the given line
def perpendicular_slope := -1 / slope_of_given_line

-- Theorem: The slope of the line perpendicular to the line 3x - 4y = 8 is -4/3
theorem slope_perpendicular_to_given_line : perpendicular_slope = - (4 : ℝ) / (3 : ℝ) :=
by sorry

end slope_perpendicular_to_given_line_l569_569687


namespace max_value_nong_l569_569106

theorem max_value_nong (A B C D E F G H I J K : ℕ) 
  (h1 : 1 ≤ A ∧ A ≤ 11)
  (h2 : 1 ≤ B ∧ B ≤ 11)
  (h3 : 1 ≤ C ∧ C ≤ 11)
  (h4 : 1 ≤ D ∧ D ≤ 11)
  (h5 : 1 ≤ E ∧ E ≤ 11)
  (h6 : 1 ≤ F ∧ F ≤ 11)
  (h7 : 1 ≤ G ∧ G ≤ 11)
  (h8 : 1 ≤ H ∧ H ≤ 11)
  (h9 : 1 ≤ I ∧ I ≤ 11)
  (h10 : 1 ≤ J ∧ J ≤ 11)
  (h11 : 1 ≤ K ∧ K ≤ 11)
  (distinct : list.nodup [A, B, C, D, E, F, G, H, I, J, K])
  (sum1 : A + B + C + D = 21)
  (sum2 : D + E + F + G = 21)
  (sum3 : G + H + F + I = 21)
  (sum4 : I + J + K + A = 21)
  (ineq : D > G ∧ G > I ∧ I > F ∧ F > A) :
  J = 9 :=
sorry

end max_value_nong_l569_569106


namespace r_squared_is_one_l569_569781

-- Definitions for conditions
variables {α β : Type*} [field α] [nontrivial α]
variables (points : list (α × α))
variables (m : α) (b : α)

-- Condition: All points lie on a line with a non-zero slope
def all_points_on_line_with_nonzero_slope : Prop :=
  ∃ (f : α → α), (∀ (x y : α), (x, y) ∈ points → y = f x) ∧ (∃ k : α, k ≠ 0 ∧ ∀ x, f x = m * x + b)

-- Statement to prove
theorem r_squared_is_one (h : all_points_on_line_with_nonzero_slope points m b) : 
  R_squared points = 1 :=
sorry

end r_squared_is_one_l569_569781


namespace find_a_l569_569459

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l569_569459


namespace total_number_of_pipes_l569_569976

theorem total_number_of_pipes (bottom_layer top_layer layers : ℕ) 
  (h_bottom_layer : bottom_layer = 13) 
  (h_top_layer : top_layer = 3) 
  (h_layers : layers = 11) : 
  bottom_layer + top_layer = 16 → 
  (bottom_layer + top_layer) * layers / 2 = 88 := 
by
  intro h_sum
  sorry

end total_number_of_pipes_l569_569976


namespace KD_squared_eq_FD_squared_plus_AF_times_BF_l569_569432

variables 
  {A B C D E F K : Type}
  (h_triangle_ABC : ∀ (A B C : Type), ∃ (ABC : Type), acute_triangle ABC)
  (h_angle_B_gt_45 : ∀ (B : Type), ∃ (angle_B : Type), angle_B > 45)
  (h_feet_altitudes : ∀ (A B C D E F : Type), ∃ (D E F : Type), feet_altitudes D E F A B C)
  (h_angle_DKF_KEF : ∀ (D K F E : Type), ∃ (angle_DKF angle_KEF : Type), angle_DKF = angle_KEF)
  (segment_AF : ∀ (A F : Type), ∃ (segment_AF : Type), ∃ K ∈ segment_AF)

noncomputable def existence_of_K : Prop :=
  ∃ (K : Type), 
    (h_angle_DKF_KEF D K F E) ∧ (K ∈ segment_AF)

theorem KD_squared_eq_FD_squared_plus_AF_times_BF :
  KD^2 = FD^2 + AF * BF :=
sorry

end KD_squared_eq_FD_squared_plus_AF_times_BF_l569_569432


namespace sin_cos_comb_l569_569327

def f (α : ℝ) : ℝ := 
  (1 + sin (-α) + sin (2 * π - α)^2 - sin (π / 2 + α)^2) /
  (2 * sin (π - α) * sin (π / 2 - α) + cos (π + α))

theorem sin_cos_comb (α : ℝ) (h : f(α) = 2) : (sin(α) + cos(α)) * cos(α) = 3 / 5 := 
by {
  sorry
}

end sin_cos_comb_l569_569327


namespace tangent_parallel_coordinates_l569_569547

theorem tangent_parallel_coordinates :
  (∃ (x1 y1 x2 y2 : ℝ), 
    (y1 = x1^3 - 2) ∧ (y2 = x2^3 - 2) ∧ 
    ((3 * x1^2 = 3) ∧ (3 * x2^2 = 3)) ∧ 
    ((x1 = 1 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -3))) :=
sorry

end tangent_parallel_coordinates_l569_569547


namespace locus_of_M_l569_569466

variable {k : Type} [EuclideanDomain k] 
variables {A B C H M : k} 

-- Definitions of dependencies
def is_circle (B C : k) : Prop := sorry
def is_on_circle (A B C : k) : Prop := sorry
def orthocenter (A B C : k) (H : k) : Prop := sorry
def projection_on_angle_bisector (H A : k) (M : k) : Prop := sorry
def angle_bisector (A B C : k) : k := sorry
def angle (A B C : k) : k := sorry

-- The statement of the proof problem
theorem locus_of_M
  (B C : k)
  (h_circle : is_circle B C)
  (A : k)
  (h_on_circle : is_on_circle A B C)
  (H : k)
  (h_orthocenter : orthocenter A B C H)
  (M : k)
  (h_projection : projection_on_angle_bisector H (angle_bisector A B C) M) :
  ∃ c1 c2 : k, is_on_circle M c1 ∧ is_on_circle M c2 :=
sorry

end locus_of_M_l569_569466


namespace total_cash_realized_correct_l569_569628

structure Stock where
  value : ℝ
  return_rate : ℝ
  brokerage_fee_rate : ℝ

def stockA : Stock := { value := 10000, return_rate := 0.14, brokerage_fee_rate := 0.0025 }
def stockB : Stock := { value := 20000, return_rate := 0.10, brokerage_fee_rate := 0.005 }
def stockC : Stock := { value := 30000, return_rate := 0.07, brokerage_fee_rate := 0.0075 }

def cash_realized (s : Stock) : ℝ :=
  let total_with_return := s.value * (1 + s.return_rate)
  total_with_return - (total_with_return * s.brokerage_fee_rate)

noncomputable def total_cash_realized : ℝ :=
  cash_realized stockA + cash_realized stockB + cash_realized stockC

theorem total_cash_realized_correct :
  total_cash_realized = 65120.75 :=
    sorry

end total_cash_realized_correct_l569_569628


namespace count_good_numbers_l569_569078

def is_good_number (k : ℤ) : Prop :=
  ∃ x : ℚ, k * x^2 - (2 * k + 3) * x + 3 = 0

theorem count_good_numbers :
  {k : ℤ | is_good_number k}.to_finset.card = 3 :=
by
  sorry

end count_good_numbers_l569_569078


namespace _l569_569710
noncomputable def locus_theorem :=
forall (circle : Circle) (A B : point) (M : point) (hAB : on_circle A circle) (hBM : on_circle M circle),
  let F := midpoint B M,
  let m := perpendicular_to_line_through_point A M F,
  let T := intersection_of_lines (line A M) m,
  ∃ C : point, 
    ∀ (M : point) (hM : on_circle M circle),
    on_circle T (thales_circle_over_diameter A C)

lemma proof_of_locus_theorem : locus_theorem :=
sorry  -- proof to be filled in by the user

end _l569_569710


namespace pyramid_volume_is_four_thirds_l569_569608

noncomputable def pyramid_volume : ℝ :=
let side_length := 2 in
let base_area := (1/2) * side_length * side_length in
let height := side_length in
(1/3) * base_area * height

theorem pyramid_volume_is_four_thirds : pyramid_volume = 4/3 :=
by
  sorry

end pyramid_volume_is_four_thirds_l569_569608


namespace toby_change_correct_l569_569561

def cheeseburger_cost := 3.65
def milkshake_cost := 2.00
def coke_cost := 1.00
def fries_cost := 4.00
def cookie_cost := 0.50
def sales_tax_rate := 0.07
def tip_rate := 0.15
def initial_amount := 15.00

def pre_tax_total := 
  2 * cheeseburger_cost +
  milkshake_cost +
  coke_cost +
  fries_cost +
  3 * cookie_cost

def sales_tax := (sales_tax_rate * pre_tax_total).round

def tip := tip_rate * pre_tax_total

def final_total := pre_tax_total + sales_tax + tip

def total_per_person := final_total / 2

def toby_change := initial_amount - total_per_person

theorem toby_change_correct : toby_change = 5.36 := by
  unfold toby_change initial_amount total_per_person final_total tip pre_tax_total
  unfold_smul
  sorry

end toby_change_correct_l569_569561


namespace average_speed_correct_l569_569612

-- Definitions based on conditions
def speed_up : ℝ := 18  -- Speed going uphill in km/hr
def speed_down : ℝ := 24  -- Speed going downhill in km/hr
def altitude_m : ℝ := 300  -- Altitude in meters

-- Conversion of altitude to kilometers to match units for speed
def altitude_km : ℝ := altitude_m / 1000  -- Altitude in kilometers

-- Compute total distance
def total_distance_km : ℝ := 2 * altitude_km  -- Total distance (uphill + downhill)

-- Calculate time taken for travel
def time_up : ℝ := altitude_km / speed_up  -- Time taken to travel uphill
def time_down : ℝ := altitude_km / speed_down  -- Time taken to travel downhill

-- Total time taken for the journey
def total_time : ℝ := time_up + time_down  -- Total time

-- Compute average speed
def average_speed : ℝ := total_distance_km / total_time  -- Average speed

-- Prove that the average speed equals 20.55 km/hr
theorem average_speed_correct : average_speed ≈ 20.55 := by
  sorry

end average_speed_correct_l569_569612


namespace perpendicular_case_parallel_case_l569_569025

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-3, 2)
noncomputable def k_perpendicular : ℝ := 19
noncomputable def k_parallel : ℝ := -1/3

-- Define the operations used:
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Perpendicular case: 
theorem perpendicular_case : dot_product (vector_add (scalar_mult k_perpendicular vector_a) vector_b) (vector_sub vector_a (scalar_mult 3 vector_b)) = 0 := sorry

-- Parallel case:
theorem parallel_case : ∃ c : ℝ, vector_add (scalar_mult k_parallel vector_a) vector_b = scalar_mult c (vector_sub vector_a (scalar_mult 3 vector_b)) ∧ c < 0 := sorry

end perpendicular_case_parallel_case_l569_569025


namespace magnitude_difference_l569_569734

noncomputable def a : ℝ × ℝ := (2, 0)
noncomputable def b (x y : ℝ) : Prop := x^2 + y^2 = 4
noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem magnitude_difference :
  (∃ x y : ℝ, b x y ∧ dot_product a (x, y) = 2) →
  (∃ x y : ℝ, sqrt ((a.1 - 2 * x)^2 + (a.2 - 2 * y)^2) = 2 * sqrt 3) :=
by sorry

end magnitude_difference_l569_569734


namespace proposition_true_and_negation_false_l569_569063

theorem proposition_true_and_negation_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬(a + b ≥ 2 → (a < 1 ∧ b < 1)) :=
by {
  sorry
}

end proposition_true_and_negation_false_l569_569063


namespace average_of_three_l569_569867

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end average_of_three_l569_569867


namespace carmen_jethro_ratio_l569_569645

variables (J C M : ℕ) (h1 : 3 * J = 60) (h2 : C + 4 * J = 113) (h3 : C + 7 = M * J)

theorem carmen_jethro_ratio (hj : J = 20) (hcj : C + 7 = M * J) : 
  (C + 7) / J = 2 :=
by
  have hJ : J = 20 := by assumption
  have hC : C = 33 := by
    rw [← hJ] at h2
    linarith
  calc
    (C + 7) / J = (33 + 7) / 20 : by rw [hC, hJ]
              ... = 40 / 20      : by norm_num
              ... = 2            : by norm_num

#check carmen_jethro_ratio

end carmen_jethro_ratio_l569_569645


namespace intersection_distance_l569_569108

noncomputable def curveA_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def curveB_parametric (t : ℝ) : (ℝ × ℝ) :=
  (-1 + (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

noncomputable def abs_val_MP_NP := Real.sqrt(2) * 12 / 7

theorem intersection_distance :
  let P := (-1, 1) in
  let f (t : ℝ) := curveB_parametric t in
  let M := f (Real.sqrt(2)/7) in
  let N := f (-Real.sqrt(2)/7) in
  (dist P M) + (dist P N) = abs_val_MP_NP := sorry

end intersection_distance_l569_569108


namespace real_part_of_complex_number_l569_569332

theorem real_part_of_complex_number (z : ℂ) (h : (1 + complex.I) * z = complex.I) : z.re = 1 / 2 :=
by
  -- proof step will be here
  sorry

end real_part_of_complex_number_l569_569332


namespace normal_line_eq_at_x0_l569_569678

def y (x : ℝ) : ℝ := (x^2 - 3 * x + 6) / x^2

def y' (x : ℝ) : ℝ := (3 * x^(-2)) - (12 * x^(-3))

theorem normal_line_eq_at_x0 : 
  let x0 := 3;
  let y0 := y x0;
  let slope := y' x0;
  y0 = 2 / 3 ∧ slope = -1 / 9 →
  ∃ (a b : ℝ), a * x0 + b = y0 ∧ a = 9 ∧ b = -79 / 3 :=
by 
  intros;
  use [9, -79 / 3];
  split;
  sorry

end normal_line_eq_at_x0_l569_569678


namespace number_of_belts_l569_569961

def ties := 34
def black_shirts := 63
def white_shirts := 42

def jeans := (2 / 3 : ℚ) * (black_shirts + white_shirts)
def scarves (B : ℚ) := (1 / 2 : ℚ) * (ties + B)

theorem number_of_belts (B : ℚ) : jeans = scarves B + 33 → B = 40 := by
  -- This theorem states the required proof but leaves the proof itself as a placeholder.
  -- The proof would involve solving equations algebraically as shown in the solution steps.
  sorry

end number_of_belts_l569_569961


namespace value_of_a5_l569_569754

noncomputable def sequence : ℕ → ℕ
| 1     := 1
| (n+1) := n * sequence n + 1

theorem value_of_a5 : sequence 5 = 65 :=
sorry

end value_of_a5_l569_569754


namespace a5_value_l569_569439

theorem a5_value : 
  ∃ (a : ℕ → ℕ), (a 1 = 2 ∧ a 2 = 5 ∧ (∀ n, a (n + 2) = a (n + 1) + a n)) → a 5 = 19 :=
begin
  sorry
end

end a5_value_l569_569439


namespace solve_for_x_l569_569726

variables {x : ℝ}
def vector_a : ℝ × ℝ := (-2, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -3)

theorem solve_for_x (h : vector_a.1 * vector_b(x).1 + vector_a.2 * vector_b(x).2 = 0) : x = -3 :=
sorry

end solve_for_x_l569_569726


namespace solution_set_l569_569721

noncomputable def f : ℝ → ℝ := sorry -- Assume f is defined properly somewhere

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom shift_property : ∀ x : ℝ, f x = f (-x + 4)
axiom specific_value : f 2024 = 1 / Real.exp 2
axiom derivative_condition : ∀ x : ℝ, f x - f' x > 0

theorem solution_set :
  {x : ℝ | f (x + 2) > Real.exp x} = set.Iio (-2) :=
begin
  sorry
end

end solution_set_l569_569721


namespace find_roots_of_polynomial_l569_569674

noncomputable def polynomial := Polynomial ℝ

theorem find_roots_of_polynomial :
  (∃ (x : ℝ), x^3 + 3 * x^2 - 6 * x - 8 = 0) ↔ (x = -1 ∨ x = 2 ∨ x = -4) :=
sorry

end find_roots_of_polynomial_l569_569674


namespace routes_from_M_to_N_l569_569215

structure Paths where
  -- Specify the paths between nodes
  C_to_N : ℕ
  D_to_N : ℕ
  A_to_C : ℕ
  A_to_D : ℕ
  B_to_N : ℕ
  B_to_A : ℕ
  B_to_C : ℕ
  M_to_B : ℕ
  M_to_A : ℕ

theorem routes_from_M_to_N (p : Paths) : 
  p.C_to_N = 1 → 
  p.D_to_N = 1 →
  p.A_to_C = 1 →
  p.A_to_D = 1 →
  p.B_to_N = 1 →
  p.B_to_A = 1 →
  p.B_to_C = 1 →
  p.M_to_B = 1 →
  p.M_to_A = 1 →
  (p.M_to_B * (p.B_to_N + (p.B_to_A * (p.A_to_C + p.A_to_D)) + p.B_to_C)) + 
  (p.M_to_A * (p.A_to_C + p.A_to_D)) = 6 
:= by
  sorry

end routes_from_M_to_N_l569_569215


namespace percentage_below_50000_l569_569258

-- Define all the conditions
def cities_between_50000_and_100000 := 35 -- percentage
def cities_below_20000 := 45 -- percentage
def cities_between_20000_and_50000 := 10 -- percentage
def cities_above_100000 := 10 -- percentage

-- The proof statement
theorem percentage_below_50000 : 
    cities_below_20000 + cities_between_20000_and_50000 = 55 :=
by
    unfold cities_below_20000 cities_between_20000_and_50000
    sorry

end percentage_below_50000_l569_569258


namespace sum_of_special_primes_l569_569693

open Nat

def is_prime (n : ℕ) : Prop := Prime n

def sum_special_primes (start end : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ p, is_prime p ∧ p ≡ 1 [MOD 5] ∧ p ≡ 2 [MOD 3])
                             (Finset.range (end + 1) \ Finset.range start))

theorem sum_of_special_primes : sum_special_primes 1 100 = 123 :=
sorry

end sum_of_special_primes_l569_569693


namespace min_liars_in_presidium_l569_569236

-- Define the number of people and seating arrangement
def num_people := 32
def rows := 4
def cols := 8

-- Define the properties of truth-tellers and liars
structure Person :=
  (is_truth_teller : Prop)

-- Define neighbors
def is_neighbor (p1 p2 : ℕ) : Prop :=
  (p1 / cols = p2 / cols ∧ (p1 % cols = p2 % cols + 1 ∨ p1 % cols + 1 = p2 % cols)) ∨
  (p1 % cols = p2 % cols ∧ (p1 / cols = p2 / cols + 1 ∨ p1 / cols + 1 = p2 / cols))

-- Define the condition each person claims: There are representatives of both parties among their neighbors
def claimed_property (p : ℕ) (people : Vector Person num_people) : Prop :=
  ∃ q, is_neighbor p q ∧ people.get p = people.get q

-- Define the main theorem
theorem min_liars_in_presidium : ∀ people : Vector Person num_people,
  (∀ p : ℕ, p < num_people → 
    (people.get p).is_truth_teller ↔ claimed_property p people) →
  ∃ liars_count : ℕ, liars_count = 8 :=
begin
  sorry
end

end min_liars_in_presidium_l569_569236


namespace largest_common_term_l569_569184

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end largest_common_term_l569_569184


namespace solve_for_A_plus_B_l569_569302

theorem solve_for_A_plus_B (A B : ℤ) (h : ∀ ω, ω^2 + ω + 1 = 0 → ω^103 + A * ω + B = 0) : A + B = -1 :=
sorry

end solve_for_A_plus_B_l569_569302


namespace R_depends_on_d_and_n_l569_569130

variable (n a d : ℕ)

noncomputable def s1 : ℕ := (n * (2 * a + (n - 1) * d)) / 2
noncomputable def s2 : ℕ := (2 * n * (2 * a + (2 * n - 1) * d)) / 2
noncomputable def s3 : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
noncomputable def R : ℕ := s3 n a d - s2 n a d - s1 n a d

theorem R_depends_on_d_and_n : R n a d = 2 * d * n^2 :=
by
  sorry

end R_depends_on_d_and_n_l569_569130


namespace removing_zeros_changes_value_l569_569655

noncomputable def a : ℝ := 7.0800
noncomputable def b : ℝ := 7.8

theorem removing_zeros_changes_value : a ≠ b :=
by
  -- proof goes here
  sorry

end removing_zeros_changes_value_l569_569655


namespace number_of_people_in_group_l569_569426

variable (T L : ℕ)

theorem number_of_people_in_group
  (h1 : 90 + L = T)
  (h2 : (L : ℚ) / T = 0.4) :
  T = 150 := by
  sorry

end number_of_people_in_group_l569_569426


namespace min_area_hypotenuse_contains_M_leg_at_y_neg2_leg_at_x_zero_l569_569573

-- Define the given conditions
def point_M : ℝ × ℝ := (1, 0)
def line_y : ℝ → ℝ := λ (x : ℝ), -2
def line_x : ℝ → ℝ := λ (y : ℝ), 0

-- Define the minimum area calculation
def min_area_of_right_triangle_if_hypotenuse_contains_M_leg_at_y_neg2_leg_at_x_zero : ℝ :=
  let k := 4 in -- Derived from solving f'(k) = 0 and checking for minimal value
  (k - 2)^2 / (2 * k)

-- The proof problem statement
theorem min_area_hypotenuse_contains_M_leg_at_y_neg2_leg_at_x_zero :
  min_area_of_right_triangle_if_hypotenuse_contains_M_leg_at_y_neg2_leg_at_x_zero = 0.5 :=
sorry

end min_area_hypotenuse_contains_M_leg_at_y_neg2_leg_at_x_zero_l569_569573


namespace complex_expression_equals_zero_l569_569028

def i : ℂ := Complex.I

theorem complex_expression_equals_zero : 2 * i^5 + (1 - i)^2 = 0 := 
by
  sorry

end complex_expression_equals_zero_l569_569028


namespace exists_xyz_l569_569133

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_xyz :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + sum_of_digits x = y + sum_of_digits y ∧ y + sum_of_digits y = z + sum_of_digits z) :=
by {
  sorry
}

end exists_xyz_l569_569133


namespace option_D_is_correct_l569_569221

theorem option_D_is_correct :
  (∀ x y : ℝ, real.sqrt 4 - real.sqrt 2 ≠ real.sqrt 2) ∧
  (real.sqrt (3^2 - 2^2) ≠ real.sqrt (3 - 2)^2) ∧
  (real.sqrt (1 + 1/4) ≠ 1 + 1/2) ∧
  (2 * real.sqrt 3 * 3 * real.sqrt 2 = 6 * real.sqrt 6) :=
by {
  sorry
}

end option_D_is_correct_l569_569221


namespace largest_value_among_given_numbers_l569_569274

-- Define the function f
def f (x : ℝ) : ℝ := (log x) / x

-- Prove that f is decreasing on [e, +∞)
theorem largest_value_among_given_numbers :
  e < 3 → 3 < π → π < 15 →
  f e > f 3 ∧ f 3 > f π ∧ f π > f 15 →
  (1/e > log (3^(1/3))) ∧ (log (3^(1/3)) > (log π)/π) ∧
  ((log π)/π > ((sqrt 15) * log 15)/30) :=
  by
    sorry

end largest_value_among_given_numbers_l569_569274


namespace discount_percentage_is_correct_l569_569834

-- Given conditions
def original_cost_slippers : ℝ := 50.00
def embroidery_cost_per_shoe : ℝ := 5.50
def number_of_shoes : ℕ := 2
def shipping_cost : ℝ := 10.00
def total_cost_after_discount : ℝ := 66.00

-- Derived data
def total_cost_without_discount : ℝ :=
  original_cost_slippers + embroidery_cost_per_shoe * number_of_shoes + shipping_cost

def discount_amount : ℝ :=
  total_cost_without_discount - total_cost_after_discount

def discount_percentage : ℝ :=
  (discount_amount / original_cost_slippers) * 100

-- Theorem to be proved
theorem discount_percentage_is_correct (h1 : original_cost_slippers = 50.00)
    (h2 : embroidery_cost_per_shoe = 5.50) (h3 : number_of_shoes = 2)
    (h4 : shipping_cost = 10.00) (h5 : total_cost_after_discount = 66.00) :
    discount_percentage = 10 := by
  -- Proof is omitted
  sorry

end discount_percentage_is_correct_l569_569834


namespace outfit_count_l569_569178

theorem outfit_count (blue_shirts : ℕ) (green_shirts : ℕ) (pants : ℕ) (blue_hats : ℕ) (green_hats : ℕ) :
  blue_shirts = 7 → green_shirts = 6 → pants = 7 → blue_hats = 10 → green_hats = 9 →
  (blue_shirts * green_hats * pants + green_shirts * blue_hats * pants) = 861 :=
by intros h1 h2 h3 h4 h5
   rw [h1, h2, h3, h4, h5]
   exact eq.refl 861

end outfit_count_l569_569178


namespace right_triangle_count_l569_569498

def is_right_triangle (P Q R : ℤ × ℤ) : Prop :=
  let P := (Px, Py) in
  let Q := (Qx, Qy) in
  let R := (Rx, Ry) in
  (Py = Ry) ∧ (Px = Qx) ∧ (Qy ≠ Py)

def num_right_triangles : ℕ :=
  let valid_x := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5] in
  let valid_y := [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] in
  let count_px_rx :=
    (valid_x.length * (valid_x.length - 1)) -- choose two different X-coordinates
  let count_py :=
    (valid_y.length - 1) -- for each Y-coordinate pairs, choose a different Y-coordinate for Q
  count_px_rx * count_py

theorem right_triangle_count : num_right_triangles = 900 := by
  sorry

end right_triangle_count_l569_569498


namespace complement_A_in_U_l569_569373

open Set

variable (α : Type*) [PartialOrder α] [Zero α] [Two : One + One = Two]

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def U : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

theorem complement_A_in_U : compl (A : Set ℝ) ∩ (U : Set ℝ) = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by 
  sorry

end complement_A_in_U_l569_569373


namespace seagull_problem_l569_569906

theorem seagull_problem : 
  let initial_seagulls := 36
  let fraction_scared := 1 / 4
  let remaining_seagulls := 18
  let scared_away := initial_seagulls * fraction_scared
  let seagulls_after_scared := initial_seagulls - scared_away
  let flew_away := seagulls_after_scared - remaining_seagulls
  let fraction_flew_away := flew_away / seagulls_after_scared
  in fraction_flew_away = 1 / 3 :=
by 
  sorry

end seagull_problem_l569_569906


namespace converse_and_inverse_false_l569_569375

-- Define the property of being a rhombus and a parallelogram
def is_rhombus (R : Type) : Prop := sorry
def is_parallelogram (P : Type) : Prop := sorry

-- Given: If a quadrilateral is a rhombus, then it is a parallelogram
def quad_imp (Q : Type) : Prop := is_rhombus Q → is_parallelogram Q

-- Prove that the converse and inverse are false
theorem converse_and_inverse_false (Q : Type) 
  (h1 : quad_imp Q) : 
  ¬(is_parallelogram Q → is_rhombus Q) ∧ ¬(¬(is_rhombus Q) → ¬(is_parallelogram Q)) :=
by
  sorry

end converse_and_inverse_false_l569_569375


namespace vertex_of_parabola_l569_569872

theorem vertex_of_parabola :
  ∃ (x y : ℝ), (∀ x : ℝ, y = x^2 - 12 * x + 9) → (x, y) = (6, -27) :=
sorry

end vertex_of_parabola_l569_569872


namespace product_of_dice_is_divisible_by_4_l569_569566

def probability_divisible_by_4 : ℚ :=
  let prob := 11 / 12 in
  prob

theorem product_of_dice_is_divisible_by_4 :
  let rolls := fin 5 → fin 6 in
  ∀ (f : rolls → ℕ), probability_divisible_by_4 = 11 / 12 :=
by
  intros
  sorry

end product_of_dice_is_divisible_by_4_l569_569566


namespace find_angle_l569_569344

open Real

variables {a b : ℝ^3}

def is_nonzero_vector (v : ℝ^3) : Prop := v ≠ 0

def orthogonal (v w : ℝ^3) : Prop := dot_product v w = 0

theorem find_angle (ha : is_nonzero_vector a)
                   (hb : is_nonzero_vector b)
                   (h1 : orthogonal (a - 2 • b) a)
                   (h2 : orthogonal (b - 2 • a) b) :
  angle a b = π / 3 :=
sorry

end find_angle_l569_569344


namespace value_of_a_l569_569765

-- Definition of the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- Definition of the derivative f'(-1)
def f_prime_at_neg1 (a : ℝ) : ℝ := 3 * a - 6

-- The theorem to prove the value of a
theorem value_of_a (a : ℝ) (h : f_prime_at_neg1 a = 3) : a = 3 :=
by
  sorry

end value_of_a_l569_569765


namespace differential_approximation_l569_569283

noncomputable def y (x : ℝ) : ℝ := x^(1/3)

theorem differential_approximation :
  y 2.54 ≈ 2.9722 :=
sorry

end differential_approximation_l569_569283


namespace max_groups_l569_569476

structure MO_SpaceCity (n : ℕ) := 
  (stations : Finset (Fin n))
  (bidirectional_passages : Sym2 (stations → stations) → Prop)
  (unidirectional_passages : stations → stations → Prop)

def is_intercommunicating_four_station_group (space_city : MO_SpaceCity 99) (group : Finset (Fin 99)) :=
  group.card = 4 ∧ ∀ a b ∈ group, a ≠ b → 
  (∃ p : List (Fin 99), p.head = a ∧ p.last = b ∧ 
  ∀ (u v : Fin 99) (h : (u, v) ∈ List.zip p (p.tail)), 
  space_city.bidirectional_passages ⟦(u, v)⟧ ∨ 
  space_city.unidirectional_passages u v)

noncomputable def max_intercommunicating_four_station_groups (space_city : MO_SpaceCity 99) : ℕ := 
  sorry

theorem max_groups (space_city : MO_SpaceCity 99) : 
  max_intercommunicating_four_station_groups space_city = 2052072 := 
sorry

end max_groups_l569_569476


namespace find_k_range_l569_569743

noncomputable def f (k x : ℝ) : ℝ := (k * x + 1 / 3) * Real.exp x - x

theorem find_k_range : 
  (∃ (k : ℝ), ∀ (x : ℕ), x > 0 → (f k (x : ℝ) < 0 ↔ x = 1)) ↔
  (k ≥ 1 / (Real.exp 2) - 1 / 6 ∧ k < 1 / Real.exp 1 - 1 / 3) :=
sorry

end find_k_range_l569_569743


namespace hyperbola_eccentricity_l569_569053

-- Non-computable definitions for the conic sections
noncomputable def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  { pt : ℝ × ℝ | pt.2^2 = 2*p*pt.1 }

noncomputable def hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) : set (ℝ × ℝ) :=
  { pt : ℝ × ℝ | (pt.1^2 / a^2) - (pt.2^2 / b^2) = 1 }

-- Definitions for the properties given in the problem
def same_focus (F : ℝ × ℝ) (parabola : set (ℝ × ℝ)) (hyperbola : set (ℝ × ℝ)) : Prop :=
  -- A function that verifies if both the parabola and the hyperbola share the same focus F.
  sorry

def intersect_point (A : ℝ × ℝ) (parabola : set (ℝ × ℝ)) (hyperbola : set (ℝ × ℝ)) : Prop :=
  A ∈ parabola ∧ A ∈ hyperbola

def perpendicular_to_x_axis (A F : ℝ × ℝ) : Prop :=
  A.1 = F.1

-- The eccentricity function for the hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (real.sqrt (1 + (b^2 / a^2)))

-- The main theorem statement
theorem hyperbola_eccentricity (a b p : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) 
  (F A : ℝ × ℝ)
  (h_sfocus : same_focus F (parabola p hp) (hyperbola a b ha hb))
  (h_intersect : intersect_point A (parabola p hp) (hyperbola a b ha hb))
  (h_perpendicular : perpendicular_to_x_axis A F) :
  eccentricity a b = 1 := 
sorry

end hyperbola_eccentricity_l569_569053


namespace specific_pens_probability_l569_569282

noncomputable def probability_of_specific_pens : ℚ :=
  let total_ways_to_choose_pens : ℕ := nat.choose 8 4
  let ways_with_conditions : ℕ := nat.choose 6 3 * nat.choose 6 3
  (ways_with_conditions : ℚ) / (total_ways_to_choose_pens * total_ways_to_choose_pens)

theorem specific_pens_probability :
  probability_of_specific_pens = 4/49 :=
by
  sorry

end specific_pens_probability_l569_569282


namespace Gauss_line_l569_569195

theorem Gauss_line (A B C D P Q K L M : Point) 
  (h1 : extension A B C D = P) 
  (h2 : extension B C A D = Q) 
  (hK : midpoint A C K)
  (hL : midpoint B D L)
  (hM : midpoint P Q M) :
  collinear {K, L, M} := sorry

end Gauss_line_l569_569195


namespace sesame_seed_weight_in_scientific_notation_l569_569209

theorem sesame_seed_weight_in_scientific_notation :
  (200 / 50000 : ℝ) = 4 * 10 ^ (-3) := by
  sorry

end sesame_seed_weight_in_scientific_notation_l569_569209


namespace min_value_of_expression_l569_569995

noncomputable def minimum_value_expression : ℝ :=
  let λ := (sqrt 2) / 2
  let A := (0, 1)
  let B := (0, 2)
  let C := (-sqrt 2, 2)
  let parabola (x : ℝ) := sqrt (4 * sqrt 2 * x)
  let Q_projection (Q : ℝ × ℝ) : ℝ := Q.1 -- Projection of Q on x = -sqrt 2
  let circle := { p : ℝ × ℝ | (p.1 + sqrt 2)^2 + p.2^2 = 2 }
  let M : ℝ × ℝ := sorry -- a moving point on the circle
  have Apollonius : {P : ℝ × ℝ | (P.1^2 + (P.2 - 1)^2) / (P.1^2 + (P.2 - 2)^2) = λ^2} := sorry,
  let distance (p1 p2 : ℝ × ℝ) := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let MC := distance M C
  let QH := distance Q_projection Q
  let QM := distance Q M
  let expression := λ/2 * MC + QH + QM
  3

theorem min_value_of_expression : minimum_value_expression = 3 :=
  sorry

end min_value_of_expression_l569_569995


namespace triangle_angle_proof_l569_569593

theorem triangle_angle_proof
  (A B C D : Type)
  (angle_ABC angle_ACB angle_BCD : ℝ)
  (AD BC : ℝ)
  (h1 : angle_ABC = 40 * real.pi / 180)
  (h2 : angle_ACB = 40 * real.pi / 180)
  (h3 : AD = BC)
  (h4 : BC = 2 * (real.cos (40 * real.pi / 180)) * (real.cos (50 * real.pi / 180))) :
  angle_BCD = 10 * real.pi / 180 := 
by 
  sorry

end triangle_angle_proof_l569_569593


namespace reyn_pieces_l569_569147

-- Define the conditions
variables (total_pieces : ℕ) (pieces_each : ℕ) (pieces_left : ℕ)
variables (R : ℕ) (Rhys : ℕ) (Rory : ℕ)

-- Initial Conditions
def mrs_young_conditions :=
  total_pieces = 300 ∧
  pieces_each = total_pieces / 3 ∧
  Rhys = 2 * R ∧
  Rory = 3 * R ∧
  6 * R + pieces_left = total_pieces ∧
  pieces_left = 150

-- The statement of our proof goal
theorem reyn_pieces (h : mrs_young_conditions total_pieces pieces_each pieces_left R Rhys Rory) : R = 25 :=
sorry

end reyn_pieces_l569_569147


namespace fly_distance_from_floor_l569_569965

theorem fly_distance_from_floor (d1 d2 d3 d4 : ℝ) (h : (2-0)^2 + (5-0)^2 + (h - 7)^2 = 11^2) : 
  h = sqrt 92 + 7 :=
by
  sorry

end fly_distance_from_floor_l569_569965


namespace part1_part2_part3_l569_569744

open Real

def f (x : ℝ) (m : ℝ) : ℝ := (3 * x + m) / (x^2 + 1)

theorem part1 (h : ∀ x : ℝ, f (-x) 0 = -f x 0) : f 0 0 = 0 := sorry

theorem part2 (a b : ℝ) (ha : 0 ≤ a) (hb1 : b ≤ 1) (hb2 : 1 ≤ b) (h1 : a < b) :
  (f a 0 ≤ f b 0 ∧ a ≤ 1 ∧ b ≤ 1) ∨ (f a 0 ≥ f b 0 ∧ 1 ≤ a ∧ 1 ≤ b) := sorry

theorem part3 (n : ℝ) (h : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → f x1 0 - f x2 0 + n ≤ 0) :
  n ≤ - 3 / 2 := sorry

end part1_part2_part3_l569_569744


namespace area_ratio_of_similar_isosceles_triangles_l569_569912

theorem area_ratio_of_similar_isosceles_triangles
  (b1 b2 h1 h2 : ℝ)
  (h_ratio : h1 / h2 = 2 / 3)
  (similar_tri : b1 / b2 = 2 / 3) :
  (1 / 2 * b1 * h1) / (1 / 2 * b2 * h2) = 4 / 9 :=
by
  sorry

end area_ratio_of_similar_isosceles_triangles_l569_569912


namespace smallest_sum_of_2_digit_numbers_l569_569407

theorem smallest_sum_of_2_digit_numbers : 
  ∃ a b c d : ℕ, 
    {a, b, c, d} ⊆ {5, 6, 7, 8, 9} ∧ 
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    ∃ (x y : ℕ), 
      (10 * a + c) + (10 * b + d) = 125 :=
by
  sorry

end smallest_sum_of_2_digit_numbers_l569_569407


namespace cardinality_A5_cardinality_sum_Ak_l569_569456

def set_of_multiples (n : ℕ) : Set ℕ := { x | (2^n < x ∧ x < 2^(n+1)) ∧ ∃ m : ℕ, x = 3 * m }

def cardinality (s : Set ℕ) : ℕ := Set.card s

theorem cardinality_A5 : cardinality (set_of_multiples 5) = 11 :=
sorry

theorem cardinality_sum_Ak : (∑ k in Finset.range 10 + 1, cardinality (set_of_multiples k)) = 682 :=
sorry

end cardinality_A5_cardinality_sum_Ak_l569_569456


namespace soda_bottle_count_l569_569966

theorem soda_bottle_count (r d : Nat) (h_r : r = 49) (h_d : d = 40) : r + d = 89 := 
by
  rw [h_r, h_d]
  norm_num
  sorry

end soda_bottle_count_l569_569966


namespace not_equivalent_to_0000042_l569_569223

theorem not_equivalent_to_0000042 :
  ∀ x ∈ {4.2e-6, 42e-7, (21 / 5) * 1e-6, 1 / 238095, 2.1e-6},
  (x ≠ 4.2e-6) ↔ (x = 2.1e-6) := by
  sorry

end not_equivalent_to_0000042_l569_569223


namespace normal_line_eq_at_x0_l569_569677

def y (x : ℝ) : ℝ := (x^2 - 3 * x + 6) / x^2

def y' (x : ℝ) : ℝ := (3 * x^(-2)) - (12 * x^(-3))

theorem normal_line_eq_at_x0 : 
  let x0 := 3;
  let y0 := y x0;
  let slope := y' x0;
  y0 = 2 / 3 ∧ slope = -1 / 9 →
  ∃ (a b : ℝ), a * x0 + b = y0 ∧ a = 9 ∧ b = -79 / 3 :=
by 
  intros;
  use [9, -79 / 3];
  split;
  sorry

end normal_line_eq_at_x0_l569_569677


namespace value_of_b_minus_d_squared_l569_569230

variable {a b c d : ℤ}

theorem value_of_b_minus_d_squared (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 3) : (b - d) ^ 2 = 25 := 
by
  sorry

end value_of_b_minus_d_squared_l569_569230


namespace sum_of_decimals_l569_569643

theorem sum_of_decimals :
  (2 / 100 : ℝ) + (5 / 1000) + (8 / 10000) + (6 / 100000) = 0.02586 :=
by
  sorry

end sum_of_decimals_l569_569643


namespace min_sum_x_y_l569_569033

theorem min_sum_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y ≥ 9 :=
by sorry

example (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y = 9 ↔ (x = 3 ∧ y = 6) :=
by sorry

end min_sum_x_y_l569_569033


namespace intersection_eq_l569_569471

def M : Set ℝ := {x | ∃ y, y = Real.log (2 - x) / Real.log 3}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_eq_l569_569471


namespace largest_common_value_lt_1000_l569_569182

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end largest_common_value_lt_1000_l569_569182


namespace recycle_cans_l569_569316

theorem recycle_cans (n : ℕ) (h : n = 625) (recycling_rule : ∀ x, x >= 5 → x / 5 = x / 5) :
  let total_new_cans := 125 + 25 + 5 + 1 in
  total_new_cans = 156 :=
by
  sorry

end recycle_cans_l569_569316


namespace axis_of_symmetry_l569_569413

theorem axis_of_symmetry (k : ℤ) :
  ∃ k : ℤ, translated_axis k = (k * π / 2 + π / 12 : ℝ) :=
by
  sorry

end axis_of_symmetry_l569_569413


namespace roots_cubic_polynomial_l569_569821

theorem roots_cubic_polynomial (a b c : ℝ) 
  (h1 : a^3 - 2*a - 2 = 0) 
  (h2 : b^3 - 2*b - 2 = 0) 
  (h3 : c^3 - 2*c - 2 = 0) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -18 :=
by
  sorry

end roots_cubic_polynomial_l569_569821


namespace exists_minimal_k_for_polynomials_l569_569317

theorem exists_minimal_k_for_polynomials (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℕ), ∃ (f g : polynomial ℤ), 
    k = f * (polynomial.C 1 + polynomial.X) ^ (2 * n) + g * (polynomial.X ^ (2 * n) + polynomial.C 1) ∧ 
    (∀ l, (∃ (h j : polynomial ℤ), l = h * (polynomial.C 1 + polynomial.X) ^ (2 * n) + j * (polynomial.X ^ (2 * n) + polynomial.C 1)) → l ≥ k) :=
sorry

end exists_minimal_k_for_polynomials_l569_569317


namespace cubic_eq_one_real_root_l569_569161

/-- The equation x^3 - 4x^2 + 9x + c = 0 has exactly one real root for any real number c. -/
theorem cubic_eq_one_real_root (c : ℝ) : 
  ∃! x : ℝ, x^3 - 4 * x^2 + 9 * x + c = 0 :=
sorry

end cubic_eq_one_real_root_l569_569161


namespace solution_1_solution_2_solution_3_l569_569876

noncomputable def f : ℝ → ℝ :=
λ x, if (0 ≤ x ∧ x < 1/2) then x + 1/2 else if (1/2 ≤ x ∧ x ≤ 1) then 2 * (1 - x) else 0

def fk (k : ℕ) (x : ℝ) : ℝ :=
f (x - k) - k / 2

structure Point :=
(x : ℝ)
(y : ℝ)

def Pk (k : ℕ) : Point :=
{ x := k + 1/2, y := 1 - k / 2 }

def Qk (k : ℕ) : Point :=
{ x := k + 1, y := -k / 2 }

def line_L (x y : ℝ) : Prop :=
2 * x + 4 * y - 5 = 0

def distance (p : Point) (L : ℝ → ℝ → Prop) : ℝ :=
(abs (2 * p.x + 4 * p.y - 5)) / sqrt(4 + 16)

theorem solution_1 : {x : ℝ | 0 ≤ x ∧ x ≤ 1 ∧ f x ≤ x} = set.Icc (2/3 : ℝ) 1 :=
by sorry

theorem solution_2 : ∀ k : ℕ, line_L (Pk k).x (Pk k).y :=
by sorry

theorem solution_3 : ∀ k : ℕ, distance (Qk k) line_L = 3 * sqrt 5 / 10 :=
by sorry

end solution_1_solution_2_solution_3_l569_569876


namespace car_sticker_price_l569_569443

theorem car_sticker_price :
  let old_car_value := 20000
  let old_car_selling_fraction := 0.80
  let out_of_pocket := 11000
  let new_car_buying_fraction := 0.90
  let selling_price := old_car_value * old_car_selling_fraction
  let P := (out_of_pocket + selling_price) / new_car_buying_fraction
  P = 30000 := by
    have selling_price := old_car_value * old_car_selling_fraction
    have new_car_total := out_of_pocket + selling_price
    have final_price := new_car_total / new_car_buying_fraction
    have P := final_price
    exact eq.refl 30000

end car_sticker_price_l569_569443


namespace hyperbola_eccentricity_l569_569059

variable (a : ℝ)

def hyperbola_eqn (a : ℝ) : Prop := (0 < a) ∧ (a < Real.sqrt 2) ∧ (Real.atan (Real.sqrt 2 / a) = Real.pi / 3)

theorem hyperbola_eccentricity : hyperbola_eqn a → (Real.sqrt (a^2 + 2)) / a = 2 := 
by
  intro ha
  sorry

end hyperbola_eccentricity_l569_569059


namespace value_of_x_squared_y_plus_xy_squared_l569_569077

variable {R : Type} [CommRing R] (x y : R)

-- Given conditions
def cond1 : Prop := x + y = 3
def cond2 : Prop := x * y = 2

-- The main theorem to prove
theorem value_of_x_squared_y_plus_xy_squared (h1 : cond1 x y) (h2 : cond2 x y) : x^2 * y + x * y^2 = 6 :=
by
  sorry

end value_of_x_squared_y_plus_xy_squared_l569_569077


namespace part_one_part_two_l569_569034

theorem part_one (a : ℕ → ℝ) (h₀ : 0 < a 1) (h₁ : ∀ n, a (n+1) = a n - log (a n + 1)) : ∀ n, 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1 :=
sorry

theorem part_two (a b : ℕ → ℝ) (h₀ : a 1 = real.sqrt 2 / 2) (h₁ : ∀ n, a (n+1) < (a n)^2 / 2)
  (h₂ : b 1 = 1 / 2) (h₃ : ∀ n, b (n+1) = (n+1) / 2 * b n) : ∀ n ≥ 2, b n > a n * nat.factorial n :=
sorry

end part_one_part_two_l569_569034


namespace intersection_A_B_l569_569826

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l569_569826


namespace solve_logarithmic_eq_l569_569170

-- Define the conditions
def condition1 (x : ℝ) : Prop := 9^x - 5 > 0
def condition2 (x : ℝ) : Prop := 3^x - 2 > 0

-- Translate the problem to a Lean statement
theorem solve_logarithmic_eq (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) :
  log 2 (9^x - 5) = 2 + log 2 (3^x - 2) ↔ x = 1 :=
sorry

end solve_logarithmic_eq_l569_569170


namespace petya_consecutive_possible_l569_569994

-- Define the initial conditions and permitted moves
def initial_grid : ℕ → ℕ := λ _, 0

-- Define the valid set of moves: each move modifies the grid by ±1 on adjacent cells
def valid_move (grid : ℕ → ℕ) (i j : ℕ) :=
  grid i + 1 = grid i ∧ grid j + 1 = grid j ∨ grid i - 1 = grid i ∧ grid j - 1 = grid j

-- Define the check for consecutive integers n, n+1, ..., n+8 in the grid
def contains_consecutive (grid : ℕ → ℕ) (n : ℕ) :=
  ∃ perm : list ℕ, perm.perm (list.range 9) ∧ ∀ i, grid (perm i) = n + i

-- The theorem stating the valid range for n
theorem petya_consecutive_possible (n : ℕ) :
  (∃ seq, ∃ grid, 
    grid = initial_grid ∧
    (∀ i j, grid = valid_move grid i j) ∧
    contains_consecutive grid n) ↔ n = 2 :=
sorry

end petya_consecutive_possible_l569_569994


namespace find_T5_l569_569725

variables (a b x y : ℝ)

def T (n : ℕ) : ℝ := a * x^n + b * y^n

theorem find_T5
  (h1 : T a b x y 1 = 3)
  (h2 : T a b x y 2 = 7)
  (h3 : T a b x y 3 = 6)
  (h4 : T a b x y 4 = 42) :
  T a b x y 5 = -360 :=
sorry

end find_T5_l569_569725


namespace find_f_log_2_3_l569_569737

noncomputable def f : ℝ → ℝ
| x => if h : x ≥ 4 then (1/2) ^ x else f (x + 1)

theorem find_f_log_2_3 :
  f (Real.log 3 / Real.log 2) = 1 / 24 :=
by
  sorry

end find_f_log_2_3_l569_569737


namespace concave_mirror_approximation_l569_569800

theorem concave_mirror_approximation {r α x : ℝ} (h_small_alpha : α ≈ 0) :
  x = r * (sin (α / 2) ^ 2 / cos x) →
  x ≈ r * (α ^ 2 / 4) :=
by 
  sorry

end concave_mirror_approximation_l569_569800


namespace toby_change_l569_569558

-- Definitions of costs
def cost_cheeseburger : ℝ := 3.65
def cost_milkshake : ℝ := 2.00
def cost_coke : ℝ := 1.00
def cost_fries : ℝ := 4.00
def cost_cookie : ℝ := 0.50

-- Other given values
def num_cookies : ℝ := 3
def tax_rate : ℝ := 0.07
def tip_rate : ℝ := 0.15
def amount_toby_arrived : ℝ := 15.00

-- Calculation of pre-tax total
def pre_tax_total : ℝ := 
  2 * cost_cheeseburger + 
  cost_milkshake + 
  cost_coke + 
  cost_fries + 
  num_cookies * cost_cookie

-- Calculation of sales tax rounded to nearest cent
def sales_tax : ℝ := (tax_rate * pre_tax_total).round

-- Calculation of tip rounded to nearest cent
def tip : ℝ := (tip_rate * pre_tax_total).round

-- Final calculation of total for both and then for Toby
def final_total : ℝ := pre_tax_total + sales_tax + tip
def amount_each_pays : ℝ := final_total / 2
def change_toby_brings_home : ℝ := amount_toby_arrived - amount_each_pays

-- Lean statement to prove
theorem toby_change : change_toby_brings_home = 5.36 := by
  sorry

end toby_change_l569_569558


namespace least_possible_value_of_smallest_integer_l569_569944

theorem least_possible_value_of_smallest_integer 
  (A B C D : ℤ) 
  (H_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (H_avg : (A + B + C + D) / 4 = 74)
  (H_max : D = 90) :
  A ≥ 31 :=
by sorry

end least_possible_value_of_smallest_integer_l569_569944


namespace find_a_l569_569418

theorem find_a :
  ∃ a : ℝ, 
    (∀ x : ℝ, f x = 3 * x + a * x^3) ∧ 
    (f 1 = a + 3) ∧ 
    (∃ k : ℝ, k = 6 ∧ k = deriv f 1 ∧ ((∀ x : ℝ, deriv f x = 3 + 3 * a * x^2))) → 
    a = 1 :=
by sorry

end find_a_l569_569418


namespace gcd_of_2134_and_155_is_9_and_its_ternary_is_100_l569_569473

-- Define the greatest common divisor function and the conversion to ternary (base 3)
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def toTernary (n : ℕ) : String := Nat.toDigits 3 n |>.asString

-- Main theorem statement
theorem gcd_of_2134_and_155_is_9_and_its_ternary_is_100 : gcd 2134 155 = 9 ∧ toTernary 9 = "100" := by
  sorry

end gcd_of_2134_and_155_is_9_and_its_ternary_is_100_l569_569473


namespace total_handshakes_at_convention_l569_569211

theorem total_handshakes_at_convention :
  let gremlins := 25
  let imps := 18
  let specific_gremlins := 5
  let friendly_gremlins := gremlins - specific_gremlins
  let handshakes_among_gremlins := (friendly_gremlins * (friendly_gremlins - 1)) / 2
  let handshakes_between_imps_and_gremlins := imps * gremlins
  handshakes_among_gremlins + handshakes_between_imps_and_gremlins = 640 := by
  sorry

end total_handshakes_at_convention_l569_569211


namespace slope_of_perpendicular_line_l569_569683

noncomputable def perpendicular_slope : ℚ :=
  let slope := (3/4 : ℚ) in
  - (1 / slope)

theorem slope_of_perpendicular_line (a b c : ℚ) (h : a = 3 ∧ b = -4 ∧ c = 8) : 
  perpendicular_slope = - (4 / 3) :=
by {
  sorry
}

end slope_of_perpendicular_line_l569_569683


namespace max_A_l569_569313

theorem max_A (A : ℝ) : (∀ (x y : ℕ), 0 < x → 0 < y → 3 * x^2 + y^2 + 1 ≥ A * (x^2 + x * y + x)) ↔ A ≤ 5 / 3 := by
  sorry

end max_A_l569_569313


namespace min_abs_val_l569_569825

open Complex

noncomputable def real_part_condition (z : ℂ) : Prop :=
  (z - 2) / (z - Complex.i) ∈ ℝ

theorem min_abs_val (z : ℂ) (h : real_part_condition z) :
    Complex.abs (z + 3) = 2 :=
sorry

end min_abs_val_l569_569825


namespace pharmacy_incurs_loss_l569_569602

variable (a b : ℝ)
variable (h : a < b)

theorem pharmacy_incurs_loss 
  (H : (41 * a + 59 * b) > 100 * (a + b) / 2) : true :=
by
  sorry

end pharmacy_incurs_loss_l569_569602


namespace sum_of_digits_l569_569014

theorem sum_of_digits (n : ℕ) : 
  let sum := (1, 2, ..., 10^n - 1).sum_digits 
  sum = 9 * n * 10^n / 2 := 
by
  sorry

end sum_of_digits_l569_569014


namespace average_of_possible_x_values_l569_569401

theorem average_of_possible_x_values :
  (∃ x : ℝ, sqrt (3 * x^2 + 4) = sqrt 31) → 
  let values := {x : ℝ | sqrt (3 * x^2 + 4) = sqrt 31} in
  values.Avg = 0 :=
by
  sorry

end average_of_possible_x_values_l569_569401


namespace astronauts_standing_arrangement_l569_569483

theorem astronauts_standing_arrangement : 
  let astronauts := ["Chen Dong", "Liu Yang", "Cai Xuzhe", "Fei Junlong", "Deng Qingming", "Zhang Lushun"]
  let shenzhou14 := ["Chen Dong", "Liu Yang", "Cai Xuzhe"]
  let shenzhou15 := ["Fei Junlong", "Deng Qingming", "Zhang Lushun"] in
  let arrangements := 72 in
  ∃ (A : List String) (condition1 : ∀ (x y z : String), x ∈ shenzhou14 → y ∈ shenzhou14 → z ∈ shenzhou14 → x ≠ y → y ≠ z → z ≠ x → 
    ¬ (x ≠ nil ∧ x :: z ≠ nil ∧ y ≠   nil)), 
  ∀ (x : String), x = "Liu Yang" → (A.head ≠ x ∧ A.last ≠ x) → 
  (arrangements = 72) ∧ (A.length = 6) := sorry

end astronauts_standing_arrangement_l569_569483


namespace angle_AFE_eq_150_l569_569796

open Set

universe u

-- Define rectangle ABCD
structure Rectangle (A B C D : Type u) where
  coords : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)
  AB_eq_2BC : dist (fst (fst (fst coords))) (fst (snd (snd coords))) = 2 * dist (fst (snd (fst coords))) (fst (snd coords))

-- Definitions for point E and F
structure PointE (C D E : Type u) where
  angle_CDE_eq_120 : ∠ (fst (fst coords)) (fst D) (fst E) = 120

structure Midpoint (F A D : Type u) where
  midpoint_AD : dist (fst (fst coords)) (fst F) = dist (fst F) (fst D)

-- Main theorem statement
theorem angle_AFE_eq_150
  (A B C D E F : Type u)
  (rect : Rectangle A B C D)
  (ptE : PointE C D E)
  (midptF : Midpoint F A D) :
  ∠ (fst (fst rect.coords)) (fst F) (fst E) = 150 := 
sorry

end angle_AFE_eq_150_l569_569796


namespace child_ticket_price_l569_569967

theorem child_ticket_price
    (num_people : ℕ)
    (num_adults : ℕ)
    (num_seniors : ℕ)
    (num_children : ℕ)
    (adult_ticket_cost : ℝ)
    (senior_discount : ℝ)
    (total_bill : ℝ) :
    num_people = 50 →
    num_adults = 25 →
    num_seniors = 15 →
    num_children = 10 →
    adult_ticket_cost = 15 →
    senior_discount = 0.25 →
    total_bill = 600 →
    ∃ x : ℝ, x = 5.63 :=
by {
  sorry
}

end child_ticket_price_l569_569967


namespace find_preimage_l569_569062

universe u

variable {A B : Type u}
variable f : A → B

def preimage (f : A → B) (y : B) := {x : A | f x = y}

theorem find_preimage (A B : Type) (x : A) (fx_eq : ∀ x : A, f x = 2 * x - 3) : (6 ∈ (preimage f 9)) :=
by
  sorry

end find_preimage_l569_569062


namespace gcd_of_1887_and_2091_is_51_l569_569522

variable (a b : Nat)
variable (coefficient1 coefficient2 quotient1 quotient2 quotient3 remainder1 remainder2 : Nat)

def gcd_condition1 : Prop := (b = 1 * a + remainder1)
def gcd_condition2 : Prop := (a = quotient1 * remainder1 + remainder2)
def gcd_condition3 : Prop := (remainder1 = quotient2 * remainder2)

def numbers_1887_and_2091 : Prop := (a = 1887) ∧ (b = 2091)

theorem gcd_of_1887_and_2091_is_51 :
  numbers_1887_and_2091 a b ∧
  gcd_condition1 a b remainder1 ∧ 
  gcd_condition2 a remainder1 remainder2 quotient1 ∧ 
  gcd_condition3 remainder1 remainder2 quotient2 → 
  Nat.gcd 1887 2091 = 51 :=
by
  sorry

end gcd_of_1887_and_2091_is_51_l569_569522


namespace sum_even_divisors_120_l569_569574

-- Define the prime factorization condition
def prime_factorization_120 : ∀ n, n = 120 → n = 2^3 * 3 * 5 :=
by intros n h; rw h; norm_num

-- Define the set of even divisors
def even_divisors_120 := {d | d ∣ 120 ∧ d % 2 = 0}

-- The theorem statement
theorem sum_even_divisors_120 : (∑ d in even_divisors_120, d) = 336 :=
sorry

end sum_even_divisors_120_l569_569574


namespace original_number_of_men_l569_569968

theorem original_number_of_men (W_total : ℝ) (n : ℕ) (h1 : W_total > 0) (h2 : n > 0)
  (H : n * (W_total / 55) = (n - 15) * (W_total / 60)) :
  n = 165 :=
by {
  -- Proof will be provided here, but we use sorry for now
  sorry,
}

end original_number_of_men_l569_569968


namespace max_regular_hours_correct_l569_569600

-- Define the conditions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_hours_worked : ℝ := 57
def total_compensation : ℝ := 1116

-- Define the maximum regular hours per week
def max_regular_hours : ℝ := 40

-- Define the compensation equation
def compensation (H : ℝ) : ℝ :=
  regular_rate * H + overtime_rate * (total_hours_worked - H)

-- The theorem that needs to be proved
theorem max_regular_hours_correct :
  compensation max_regular_hours = total_compensation :=
by
  -- skolemize the proof
  sorry

end max_regular_hours_correct_l569_569600


namespace oranges_equivalency_l569_569811

theorem oranges_equivalency :
  ∀ (w_orange w_apple w_pear : ℕ), 
  (9 * w_orange = 6 * w_apple + w_pear) →
  (36 * w_orange = 24 * w_apple + 4 * w_pear) :=
by
  -- The proof will go here; for now, we'll use sorry to skip it
  sorry

end oranges_equivalency_l569_569811


namespace cabbage_count_l569_569986

theorem cabbage_count 
  (length : ℝ)
  (width : ℝ)
  (density : ℝ)
  (h_length : length = 16)
  (h_width : width = 12)
  (h_density : density = 9) : 
  length * width * density = 1728 := 
by
  rw [h_length, h_width, h_density]
  norm_num
  done

end cabbage_count_l569_569986


namespace constant_term_of_g_l569_569135

variable (f g h k : ℕ → ℕ)
variable (hf : f 0 = 6)
variable (hh : h 0 = -18)
variable (hk : k 0 = 2)
variable (hs : ∀ x, h x = f x * g x * k x)

theorem constant_term_of_g :
  g 0 = -3 / 2 :=
sorry

end constant_term_of_g_l569_569135


namespace dependent_uniform_characteristic_functions_l569_569139

open probability_theory

noncomputable def A : set (ℝ × ℝ) :=
  {p | p.1 ∈ Icc (-1 : ℝ) 1 ∧ p.2 ∈ Icc (-1 : ℝ) 1 ∧ (p.1 / p.2 > (1 : ℝ) ∨ p.2 / p.1 < (-1 : ℝ))}

variables (ξ η : measure_theory.measurable_space ℝ)

axiom uniform_distribution_on_A : ∀ (x : ℝ), ∀ (y : ℝ), (x, y) ∈ A → 
  (measure_theory.measure_theory.to_real (measure_theory.measure_theory.prob (ξ.prod η)) = 1)

theorem dependent_uniform_characteristic_functions
  (ξ_eta : measure_theory.measure (ℝ × ℝ) = 1) :
  ∀ t : ℝ, 
  (∃ x y : ℝ, (x, y) ∈ A ∧ measure_theory.prob measure_theory.measure (ξ.prod η) = 
   (∃ u v : ℝ, (u, v) ∈ A)) →
  (∃ (φ_ξ φ_η : ℝ → ℝ), 
    (φ_ξ t = sin t / t ∧ φ_η t = sin t / t) ∧ 
    (φ_ξ φ_ξ t * φ_η t = (1 - cos (2 * t)) / (2 * t^2)) ∧ 
    (φ_ξ φ_ξ t * φ_η (-t) = φ_ξ t * φ_η (-t))) :=
sorry

end dependent_uniform_characteristic_functions_l569_569139


namespace Jason_4week_visits_l569_569808

-- Definitions
def William_weekly_visits : ℕ := 2
def Jason_weekly_multiplier : ℕ := 4
def weeks_period : ℕ := 4

-- We need to prove that Jason goes to the library 32 times in 4 weeks.
theorem Jason_4week_visits : William_weekly_visits * Jason_weekly_multiplier * weeks_period = 32 := 
by sorry

end Jason_4week_visits_l569_569808


namespace geometric_sequence_properties_l569_569722

theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) (h1 : ∀ n, S n = 3^n + t) (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 2 = 6 ∧ t = -1 :=
by
  sorry

end geometric_sequence_properties_l569_569722


namespace circle_tangent_to_semicircles_radius_bounds_l569_569134

variables {A B C : Type}
variables (a b c : ℝ) [triangle_side_lengths a b c]
variables (s r t : ℝ) [semiperimeter s a b c] [inradius r a b c]

theorem circle_tangent_to_semicircles_radius_bounds :
  ∀ (t : ℝ), 
  (circle_tangent_to_semicircles_with_radius t a b c) →
  (semiperimeter s a b c) →
  (inradius r a b c) →
  (∃ t, (s / 2) < t ∧ t ≤ (s / 2 + (1 - (Real.sqrt 3) / 2) * r)) :=
by
  sorry

end circle_tangent_to_semicircles_radius_bounds_l569_569134


namespace zero_conditions_l569_569745

variable {f : ℝ → ℝ}

theorem zero_conditions (a x₁ x₂ : ℝ) 
  (h₁ : f(x) = ln x + a / x - 1) 
  (h₂ : f x₁ = 0) 
  (h₃ : f x₂ = 0) 
  (h₄ : x₁ > x₂)
  (h₅ : 0 < a ∧ a < Real.exp 1) :
  (1 / x₁) + (2 / x₂) > 1 / a := 
sorry

end zero_conditions_l569_569745


namespace simplify_sqrt_expression_l569_569850

theorem simplify_sqrt_expression :
  (3 * (Real.sqrt (4 * 3)) - 2 * (Real.sqrt (1 / 3)) +
     Real.sqrt (16 * 3)) / (2 * Real.sqrt 3) = 14 / 3 := by
sorry

end simplify_sqrt_expression_l569_569850


namespace xiong_solution_l569_569590

def xiong_problem (Xiongdibrothers : ℕ) (D E : ℕ) : Prop :=
  let Xiongdash := 10 + D in
  let Xionger := 10 + E in
  Xiongdash * Xionger = Xiongdibrothers ∧ D > E ∧
  (∀ (d e : ℕ), d ≠ e → d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → e ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8]) 

theorem xiong_solution : ∃ (Xiongdibrothers : ℕ), (∃ (D E : ℕ), xiong_problem Xiongdibrothers D E) → Xiongdibrothers = 686 :=
sorry

end xiong_solution_l569_569590


namespace acceptable_bottles_proof_total_weight_difference_proof_l569_569599

noncomputable def bottle_weight_differences : List (ℕ × ℤ) := [
  (1, -4),
  (4, -2),
  (3, 0),
  (4, 1),
  (5, 3),
  (3, 5)
]

def acceptable_weight_range := -3 ≤ 3

def count_acceptable_bottles (differences : List (ℕ × ℤ)) : ℕ :=
  differences.filter (λ d, acceptable_weight_range.1 ≤ d.snd ∧ d.snd ≤ acceptable_weight_range.2).foldl (λ acc d, acc + d.fst) 0

def total_weight_difference (differences : List (ℕ × ℤ)) : ℤ :=
  differences.foldl (λ acc d, acc + (d.fst * d.snd)) 0

theorem acceptable_bottles_proof :
  count_acceptable_bottles bottle_weight_differences = 16 := 
  sorry

theorem total_weight_difference_proof :
  total_weight_difference bottle_weight_differences = 22 :=
  sorry

end acceptable_bottles_proof_total_weight_difference_proof_l569_569599


namespace div2_implies_div2_of_either_l569_569496

theorem div2_implies_div2_of_either (a b : ℕ) (h : 2 ∣ a * b) : (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end div2_implies_div2_of_either_l569_569496


namespace num_true_propositions_l569_569853

-- Define conditions as propositions
def prop1 : Prop := ∃ (base_rectangular : Bool), base_rectangular = true → ∃ (cuboid : Bool), cuboid = true
def prop2 : Prop := ∃ (congruent_edges : Bool), congruent_edges = true → ∃ (cube : Bool), cube = true
def prop3 : Prop := ∃ (lateral_edges_perpendicular : Bool), lateral_edges_perpendicular = true → ∃ (right_parallelepiped : Bool), right_parallelepiped = true
def prop4 : Prop := ∃ (equal_diagonals : Bool), equal_diagonals = true → ∃ (right_parallelepiped : Bool), right_parallelepiped = true

-- Define the main theorem to prove the question
theorem num_true_propositions : (Prop → nat) → nat :=
  fun num_true,
    num_true prop1 + num_true prop2 + num_true prop3 + num_true prop4 = 1

end num_true_propositions_l569_569853


namespace area_of_triangle_ABC_l569_569838

-- Define points and distances
variables (O A B C D : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace D]
variables [MetricSpace.Intersecting A B C] [MetricSpace.Intersecting A O B]
variables {AD BD CD : ℝ}
variables (angleAOB : Real.Angle) (r : ℝ)
variables (areaABC : ℝ)

-- Assume given conditions
axiom center_circle : ∀ p : A, dist p O = r
axiom points_on_circle : dist A O = r ∧ dist B O = r
axiom angle_120 : angle.OAB = 120
axiom imaginary_point_C : Segment.Intersection (metric_inter C D A) r (arc.small A B) 
axiom segment_conditions : AD = 2 ∧ BD = 1 ∧ CD = √2

-- Define the goal to be proved
theorem area_of_triangle_ABC : area (triangle A B C) = 3 * (sqrt 2) / 4 :=
begin
  sorry
end

end area_of_triangle_ABC_l569_569838


namespace jennifer_blue_sweets_l569_569809

/-- 
Jennifer decides to share her sweets between herself and her 3 friends. 
She has 212 green sweets, some blue sweets, and 502 yellow sweets. 
Jennifer and her friends will get 256 sweets each. 
How many blue sweets does Jennifer have?
-/
theorem jennifer_blue_sweets :
  let green_sweets := 212 in
  let yellow_sweets := 502 in
  let people := 4 in
  let sweets_per_person := 256 in
  let total_sweets := people * sweets_per_person in
  let blue_sweets := total_sweets - (green_sweets + yellow_sweets) in
  blue_sweets = 310 :=
by
  sorry

end jennifer_blue_sweets_l569_569809


namespace find_y_l569_569885

def custom_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_y (y : ℤ) (h : custom_op y 10 = 90) : y = 11 :=
by
  sorry

end find_y_l569_569885


namespace teal_bluish_count_l569_569239

theorem teal_bluish_count (n G Bg N B : ℕ) (h1 : n = 120) (h2 : G = 80) (h3 : Bg = 35) (h4 : N = 20) :
  B = 55 :=
by
  sorry

end teal_bluish_count_l569_569239


namespace part1_part2_l569_569369

noncomputable def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : set ℝ := {x | 0 < x ∧ x < 2}

theorem part1 : M = {x | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 {a : ℝ} (ha : a ∈ M) :
  (0 < a ∧ a < 1 → a^2 - a + 1 < 1 / a) ∧
  (a = 1 → a^2 - a + 1 = 1 / a) ∧
  (1 < a ∧ a < 2 → a^2 - a + 1 > 1 / a) :=
by
  sorry

end part1_part2_l569_569369


namespace common_volume_fraction_l569_569844

variable (R H : ℝ) -- radii of the base and height of the cones
variable (V1 : ℝ) -- volume of each cone
variable (h : ℝ) -- height of smaller cones
variable (r : ℝ) -- radius of the base of smaller cones
variable (V2 : ℝ) -- volume of the common part

-- Define the volume of each cone
def volume_cone (R H : ℝ) : ℝ := (1/3) * π * R^2 * H

-- Define the volume of the common part of the cones
def volume_common_part (r h : ℝ) : ℝ := (2/3) * π * r^2 * h

-- Assumptions
axiom cone_volume : V1 = volume_cone R H
axiom height_relation : h = (1/2) * H
axiom radius_relation : r = (1/2) * R

-- Conclusion to be proved
theorem common_volume_fraction : volume_common_part r h = (1/4) * V1 :=
by 
  sorry

end common_volume_fraction_l569_569844


namespace triangles_congruent_in_regular100gon_division_l569_569261

theorem triangles_congruent_in_regular100gon_division
  (G : Type) [fintype G] [decidable_eq G] [polygon G 100]
  (is_regular : regular G)
  (partition_into_parallelograms_and_triangles 
    : ∃ (P : list (parallelogram G) × (triangle G) × (triangle G)), 
      ⋃ x ∈ P.1 ++ P.2 ++ P.3, x = set_of_vertices G) :
  ∃ (T1 T2 : triangle G), congruent T1 T2 :=
by
  sorry

end triangles_congruent_in_regular100gon_division_l569_569261


namespace find_top_row_number_l569_569652

noncomputable def y : ℕ := 50
noncomputable def z : ℕ := 36

def pyramid_condition_1 : Prop := (y * z = 1800)
def pyramid_condition_2 : Prop := (6 * y = 300)

theorem find_top_row_number (h1 : pyramid_condition_1) (h2 : pyramid_condition_2) : ∃ x : ℕ, x = 60 := 
by
  use 60
  sorry

end find_top_row_number_l569_569652


namespace probI_probII_l569_569032

-- Assuming M(x, y) satisfies given condition
def satisfies_condition (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2) = 2 * Real.sqrt 2

-- Definition of the ellipse E
def is_on_ellipse (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x^2 / 2) + y^2 = 1

-- Now we state the problems
theorem probI (M : ℝ × ℝ) (h : satisfies_condition M) : is_on_ellipse M :=
  sorry

-- Let's redefine the necessary variables for part II
variables (F : ℝ × ℝ) (R : ℝ × ℝ) (P Q : ℝ × ℝ) (λ₁ λ₂ : ℝ)

-- Define the vectors and corresponding conditions
def vector_condition (R P F : ℝ × ℝ) (λ : ℝ) : Prop :=
  let RP := P.1 - R.1, P.2 - R.2
  let PF := F.1 - P.1, F.2 - P.2
  RP = λ * PF

def line_through_F (F : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  let slope := (F.2 - P.2) / (F.1 - P.1)
  let intercept := F.2 - slope * F.1
  Q.2 = slope * Q.1 + intercept

-- State the second part problem
theorem probII (F R P Q : ℝ × ℝ) (λ₁ λ₂ : ℝ)
  (h1 : vector_condition R P F λ₁)
  (h2 : vector_condition R Q F λ₂)
  (h3 : line_through_F F P Q) :
  λ₁ + λ₂ = -4 :=
  sorry

end probI_probII_l569_569032


namespace intersection_A_B_l569_569374

open Set

def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l569_569374


namespace work_completed_together_l569_569937

theorem work_completed_together (A_days B_days : ℕ) (hA : A_days = 40) (hB : B_days = 60) : 
  1 / (1 / (A_days: ℝ) + 1 / (B_days: ℝ)) = 24 :=
by
  sorry

end work_completed_together_l569_569937


namespace sum_first_four_terms_l569_569713

-- Definitions based on conditions
def is_geometric (a : Nat → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def recurrence_relation (a : Nat → ℝ) : Prop :=
  ∀ n, a (n + 2) + a (n + 1) = 6 * a n

def a_sequence (a : Nat → ℝ) : Prop :=
  is_geometric a q ∧ a 2 = 1 ∧ recurrence_relation a

-- Question translated to Lean 4 theorem
theorem sum_first_four_terms (a : Nat → ℝ) (q : ℝ) (h : a_sequence a) : 
  a 0 + a 1 + a 2 + a 3 = 15 / 2 := 
sorry

end sum_first_four_terms_l569_569713


namespace minimum_routes_schizophrenia_l569_569225

noncomputable theory
open_locale classical

def minimum_routes_for_connectivity (n : ℕ) (k : ℕ) : ℕ :=
  if (k * 2 = n) then
    let total_cities := n,
        connections_per_city := k + 1,
        total_connections := total_cities * connections_per_city,
        effective_routes := total_connections / 2 in
      effective_routes
  else 0

theorem minimum_routes_schizophrenia : 
  minimum_routes_for_connectivity 10 5 = 30 :=
by
  -- calculation and proof steps here
  sorry

end minimum_routes_schizophrenia_l569_569225


namespace max_non_intersecting_diagonals_6x6_l569_569217

theorem max_non_intersecting_diagonals_6x6 :
  ∀ (grid : fin 6 × fin 6 → bool), ∀ (drawn_diagonals : fin 6 × fin 6 → option (bool)),
  (∀ (i j : fin 6), (drawn_diagonals (i, j)).is_some →
    ∀ (di : fin 6) (dj : fin 6), 
    grid (di, dj) ≠ grid (i, j) → (drawn_diagonals (di, dj)).is_none
  ) →
  ∃ (squares : finset (fin 6 × fin 6)), 
    squares.card = 18 ∧ 
    ∀ (sq : fin 6 × fin 6), sq ∈ squares → (drawn_diagonals sq).is_some :=
by sorry

end max_non_intersecting_diagonals_6x6_l569_569217


namespace scientific_notation_86560_l569_569003

theorem scientific_notation_86560 : ∃ a n, (86560 : ℝ) = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.656 ∧ n = 4 :=
by {
  sorry
}

end scientific_notation_86560_l569_569003


namespace general_formula_sequence_sum_first_n_b_l569_569720

-- Definitions for the conditions.
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1) + (n * (n - 1) * d) / 2)

def forms_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def b_n (a : ℕ → ℕ) (n : ℕ) : ℝ :=
  1 / ((a n : ℝ) * (a (n + 1) : ℝ)) + n

def T_n (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = (b 1 + b 2 + ... + b n)

-- Theorem statements.
theorem general_formula_sequence (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
  (h_arith_seq : is_arithmetic_sequence a d) (h_S5 : S 5 = 20)
  (h_geo_seq : forms_geometric_sequence (a 3) (a 5) (a 8)) : 
  ∀ n : ℕ, a n = n + 1 :=
by sorry

theorem sum_first_n_b (a : ℕ → ℕ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_b_def : ∀ n : ℕ, b n = 1 / ((a n : ℝ) * (a (n + 1) : ℝ)) + n)
  (h_a_form : ∀ n : ℕ, a n = n + 1) :
  ∀ n : ℕ, T n = (1 / 2 - 1 / (n + 2) + (n * (n + 1)) / 2) :=
by sorry

end general_formula_sequence_sum_first_n_b_l569_569720


namespace percent_non_swimmers_play_soccer_l569_569087

variable (N : ℕ)

-- Define the conditions
def soccer_players : ℝ := 0.7 * N
def swimmers : ℝ := 0.4 * N
def soccer_and_swim : ℝ := 0.5 * soccer_players N

-- Prove that the percentage of non-swimmers who play soccer is 58%
theorem percent_non_swimmers_play_soccer (h1 : soccer_players N = 0.7 * N)
                                        (h2 : swimmers N = 0.4 * N)
                                        (h3 : soccer_and_swim N = 0.5 * soccer_players N) :
  ((soccer_players N - soccer_and_swim N) / (N - swimmers N) * 100) ≈ 58 := by
  sorry

end percent_non_swimmers_play_soccer_l569_569087


namespace intersection_M_N_l569_569829

def M := {m : ℤ | -3 < m ∧ m < 2}
def N := {n : ℕ | -1 < n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l569_569829


namespace time_to_meet_in_minutes_l569_569586

def distance_between_projectiles : ℕ := 1998
def speed_projectile_1 : ℕ := 444
def speed_projectile_2 : ℕ := 555

theorem time_to_meet_in_minutes : 
  (distance_between_projectiles / (speed_projectile_1 + speed_projectile_2)) * 60 = 120 := 
by
  sorry

end time_to_meet_in_minutes_l569_569586


namespace seq_terms_are_squares_l569_569491

theorem seq_terms_are_squares (n : ℕ) : 
  let term := (10^(n + 1) - 1) / 9 * 10^(n + 1) + 5 * 10^(n + 2) + ((10^(n + 1) * 56 - 6)) / 9 in
  ∃ k : ℕ, term = k^2 := 
by
  sorry

end seq_terms_are_squares_l569_569491


namespace hexagon_area_l569_569789

-- Define the conditions
def equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c

def length_relations (AB DE GH FC : ℝ) : Prop :=
  AB = 2 * FC ∧ DE = 3 * FC ∧ GH = 4 * FC

def area_triangle (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- The main statement: Given the conditions, prove the area of the hexagon ABGHDE
theorem hexagon_area (FC : ℝ) (area_FCI : ℝ)
  (h1 : equilateral_triangle AB BC CA)
  (h2 : equilateral_triangle DE EF FD)
  (h3 : equilateral_triangle GH HI IG)
  (h4 : length_relations AB DE GH FC)
  (h_area_FCI : area_FCI = 3) :
  let x := FC in
  7.25 * Real.sqrt 3 * x^2 - 6 = 7.25 * Real.sqrt 3 * FC^2 - 6 :=
sorry

end hexagon_area_l569_569789


namespace slope_perpendicular_to_given_line_l569_569686

-- Define the given line
def given_line (x y : ℝ) := 3 * x - 4 * y = 8

-- Define the slope-intercept form of the line and its slope
def slope_of_given_line := (3 : ℝ) / (4 : ℝ)

-- Define the slope of the line perpendicular to the given line
def perpendicular_slope := -1 / slope_of_given_line

-- Theorem: The slope of the line perpendicular to the line 3x - 4y = 8 is -4/3
theorem slope_perpendicular_to_given_line : perpendicular_slope = - (4 : ℝ) / (3 : ℝ) :=
by sorry

end slope_perpendicular_to_given_line_l569_569686


namespace solve_system_of_equations_l569_569174

theorem solve_system_of_equations : ∃ x y : ℤ, 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 ∧ x = 4 ∧ y = 3 :=
by
  sorry

end solve_system_of_equations_l569_569174


namespace sum_of_fractions_eq_two_l569_569205

theorem sum_of_fractions_eq_two : 
  (1 / 2) + (2 / 4) + (4 / 8) + (8 / 16) = 2 :=
by sorry

end sum_of_fractions_eq_two_l569_569205


namespace train_speed_is_28_l569_569269

-- Define the given conditions
def train_length : ℕ := 1200
def overbridge_length : ℕ := 200
def crossing_time : ℕ := 50

-- Define the total distance
def total_distance := train_length + overbridge_length

-- Define the speed calculation function
def speed (distance time : ℕ) : ℕ := 
  distance / time

-- State the theorem to be proven
theorem train_speed_is_28 : speed total_distance crossing_time = 28 := 
by
  -- Proof to be provided
  sorry

end train_speed_is_28_l569_569269


namespace speed_of_stream_l569_569252

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 14) (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 :=
by
  rw [h1, h2]
  norm_num

end speed_of_stream_l569_569252


namespace jeonghoon_math_score_l569_569810

theorem jeonghoon_math_score :
  ∀ (ethics korean_language science social_studies average_score total_subjects : ℕ),
  ethics = 82 →
  korean_language = 90 →
  science = 88 →
  social_studies = 84 →
  average_score = 88 →
  total_subjects = 5 →
  let total_score_for_all_subjects := average_score * total_subjects in
  let total_score_for_other_subjects := ethics + korean_language + science + social_studies in
  let math_score := total_score_for_all_subjects - total_score_for_other_subjects in
  math_score = 96 :=
by
  intros
  calc
    let total_score_for_all_subjects := average_score * total_subjects
    ... = 88 * 5 : by rw [average_score, total_subjects]
    ... = 440 : by norm_num
    let total_score_for_other_subjects := ethics + korean_language + science + social_studies
    ... = 82 + 90 + 88 + 84 : by rw [ethics, korean_language, science, social_studies]
    ... = 344 : by norm_num
    let math_score := total_score_for_all_subjects - total_score_for_other_subjects
    ... = 440 - 344 : by rw [total_score_for_all_subjects, total_score_for_other_subjects]
    ... = 96 : by norm_num

end jeonghoon_math_score_l569_569810


namespace minimum_value_f_pi_over_3_l569_569191

noncomputable def f : ℝ → ℝ := sorry
def domain : set ℝ := Ioo 0 (π / 2)

theorem minimum_value_f_pi_over_3 (h_f_diff : differentiable_on ℝ f domain)
    (h_inequality : ∀ x ∈ domain, deriv f x * tan x ≥ (2 * sin x - 1) * f x)
    (h_initial : f (π / 6) = 1) :
    f (π / 3) ≥ (exp (sqrt 3 - 1) / sqrt 3) :=
sorry

end minimum_value_f_pi_over_3_l569_569191


namespace white_ball_boxes_more_than_red_ball_boxes_l569_569303

theorem white_ball_boxes_more_than_red_ball_boxes :
  ∀ (balls_per_box white_balls red_balls : ℕ), 
  balls_per_box = 6 → white_balls = 30 → red_balls = 18 → 
  (white_balls / balls_per_box) - (red_balls / balls_per_box) = 2 :=
by
  -- introduce variables
  intros balls_per_box white_balls red_balls h1 h2 h3,
  -- simplify the goal using the given conditions
  simp [h1, h2, h3],
  sorry

end white_ball_boxes_more_than_red_ball_boxes_l569_569303


namespace correlated_income_taxation_l569_569933

-- Defining the relationships
def height_eyesight_relationship : Prop := ∀ (h e : ℕ), h ≠ e
def income_taxation_relationship : Prop := ∀ (i t : ℕ), t = linear_function i
def angle_arc_relationship (angle : ℝ) (r : ℝ) : ℝ := angle * r
def age_height_relationship : Prop := ∀ (age height : ℕ), correlated_during_growth age height

-- Main theorem statement: Prove that relationship B is correlated
theorem correlated_income_taxation : Prop :=
  income_taxation_relationship

end correlated_income_taxation_l569_569933


namespace twisted_star_angle_sum_l569_569993

-- Definitions according to the conditions given in the problem
def is_convex_polygon (n : ℕ) (n_ge_six : n ≥ 6) : Type :=
{ sides : Fin n → Fin (n + 1)
, parallel_test : ∀ k : Fin n, sides k ≠ sides ((k + 3) % n)
}

def is_twisted_star (n : ℕ) (polygon : is_convex_polygon n (by decide)) : Type :=
{ prolonged_sides_meet : ∀ k : Fin n, true
-- Dummy condition for prolnged side intersections
}

-- The goal is to prove the internal angle sum of the twisted star
theorem twisted_star_angle_sum (n : ℕ) (h : n ≥ 6) (polygon : is_convex_polygon n h) (star : is_twisted_star n polygon) :
  ∑ i in range n, 180° - (360° / n) = 180° * (n - 4) :=
sorry

end twisted_star_angle_sum_l569_569993


namespace actual_plot_area_in_acres_l569_569972

-- Condition Definitions
def base_cm : ℝ := 8
def height_cm : ℝ := 12
def scale_cm_to_miles : ℝ := 1  -- 1 cm = 1 mile
def miles_to_acres : ℝ := 320  -- 1 square mile = 320 acres

-- Theorem Statement
theorem actual_plot_area_in_acres (A : ℝ) :
  A = 15360 :=
by
  sorry

end actual_plot_area_in_acres_l569_569972


namespace find_fourth_vertex_l569_569955

-- Given three vertices of a tetrahedron
def v1 : ℤ × ℤ × ℤ := (1, 1, 2)
def v2 : ℤ × ℤ × ℤ := (4, 2, 1)
def v3 : ℤ × ℤ × ℤ := (3, 1, 5)

-- The side length squared of the tetrahedron (computed from any pair of given points)
def side_length_squared : ℤ := 11

-- The goal is to find the fourth vertex with integer coordinates which maintains the distance
def is_fourth_vertex (x y z : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 + (z - 2)^2 = side_length_squared ∧
  (x - 4)^2 + (y - 2)^2 + (z - 1)^2 = side_length_squared ∧
  (x - 3)^2 + (y - 1)^2 + (z - 5)^2 = side_length_squared

theorem find_fourth_vertex : is_fourth_vertex 4 1 3 :=
  sorry

end find_fourth_vertex_l569_569955


namespace value_of_f_1_plus_g_4_l569_569136

def f (x : Int) : Int := 2 * x - 1
def g (x : Int) : Int := x + 1

theorem value_of_f_1_plus_g_4 : f (1 + g 4) = 11 := by
  sorry

end value_of_f_1_plus_g_4_l569_569136


namespace geometric_series_sum_n_eq_31_over_16_l569_569545

theorem geometric_series_sum_n_eq_31_over_16 {n : ℕ} (a r : ℝ) (h₁ : a = 1) (h₂ : r = 1/2)
  (h₃ : (∀ (m : ℕ), m < n → r ^ m ≥ 0) ∧ (Σ (m < n), a * r ^ m = (31/16)) : n = 5 :=
by
  sorry

end geometric_series_sum_n_eq_31_over_16_l569_569545


namespace exists_small_area_triangle_l569_569088

-- We define lattice points and the required conditions.
structure LatticePoint where
  x : ℤ
  y : ℤ
  h_x : |x| ≤ 2
  h_y : |y| ≤ 2

def no_collinear (points : List LatticePoint) : Prop :=
  ∀ (p1 p2 p3 : LatticePoint), 
    p1 ∈ points → p2 ∈ points → p3 ∈ points →
    (p1.x - p2.x) * (p1.y - p3.y) ≠ (p1.y - p2.y) * (p1.x - p3.x)

def area (p1 p2 p3 : LatticePoint) : ℝ :=
  (1 / 2) * |((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) : ℤ)|

theorem exists_small_area_triangle (points : List LatticePoint) (h_len : points.length = 6) (h_no_collinear : no_collinear points) :
  ∃ (p1 p2 p3 : LatticePoint), 
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    area p1 p2 p3 ≤ 2 := by
  sorry

end exists_small_area_triangle_l569_569088


namespace geometric_sequence_sum_l569_569105

variable (a : ℕ → ℕ)
variable (a1 : a 1 = 1)
variable (ak : ∃ k : ℕ, a k = 243)
variable (q : ℕ)

theorem geometric_sequence_sum (a1 : a 1 = 1) (q : 3) (k : ℕ) (ak : a k = 243) :
  (finset.range k).sum (λ n, a n) = 364 :=
by
  sorry

end geometric_sequence_sum_l569_569105


namespace max_radius_of_circle_l569_569109

-- Definition of a point in R^2
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Circle has a center and radius
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Definition of the points (8, 0) and (-8, 0)
def point1 : Point := ⟨8, 0⟩
def point2 : Point := ⟨-8, 0⟩

-- Distance function in R^2
def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)^0.5

-- The theorem to prove the maximum possible radius
theorem max_radius_of_circle :
  ∃ (C : Circle), distance point1 C.center = C.radius ∧ distance point2 C.center = C.radius ∧ C.radius = 8 :=
by
  sorry

end max_radius_of_circle_l569_569109


namespace triangle_probability_in_15gon_l569_569263

noncomputable def segment_length (k : ℕ) := 2 * real.sin (k * real.pi / 15)

def total_segments := nat.choose 15 2

def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle_segment_lengths : ℕ :=
  let lengths := (list.range 7).map (λ k, segment_length (k + 1)) in
  (list.sequence' (list.product (list.product lengths lengths) lengths)).countp (λ ⟨⟨x, y⟩, z⟩, x ≤ y ∧ y ≤ z ∧ is_triangle x y z)

def probability_valid_triangle : ℚ :=
  valid_triangle_segment_lengths / (nat.choose total_segments 3 : ℚ)

theorem triangle_probability_in_15gon :
  probability_valid_triangle = 443 / 572 :=
sorry

end triangle_probability_in_15gon_l569_569263


namespace jose_investment_proof_l569_569562

noncomputable def jose_investment (total_profit jose_share : ℕ) (tom_investment : ℕ) (months_tom months_jose : ℕ) : ℕ :=
  let tom_share := total_profit - jose_share
  let tom_investment_mr := tom_investment * months_tom
  let ratio := tom_share * months_jose
  tom_investment_mr * jose_share / ratio

theorem jose_investment_proof : 
  ∃ (jose_invested : ℕ), 
    let total_profit := 5400
    let jose_share := 3000
    let tom_invested := 3000
    let months_tom := 12
    let months_jose := 10
    jose_investment total_profit jose_share tom_invested months_tom months_jose = 4500 :=
by
  use 4500
  sorry

end jose_investment_proof_l569_569562


namespace number_of_doubles_players_l569_569091

theorem number_of_doubles_players (x y : ℕ) 
  (h1 : x + y = 13) 
  (h2 : 4 * x - 2 * y = 4) : 
  4 * x = 20 :=
by sorry

end number_of_doubles_players_l569_569091


namespace perpendicular_lines_l569_569414

theorem perpendicular_lines (a : ℝ) :
  (∃ (m₁ m₂ : ℝ), ((a + 1) * m₁ + a * m₂ = 0) ∧ 
                  (a * m₁ + 2 * m₂ = 1) ∧ 
                  m₁ * m₂ = -1) ↔ (a = 0 ∨ a = -3) := 
sorry

end perpendicular_lines_l569_569414


namespace prove_trig_identity_l569_569843

noncomputable def trig_identity (alpha : ℝ) (n : ℕ) : Prop :=
  ∃ (alpha : ℝ) (n : ℕ), α ≠ 0 → 
    (cos alpha * cos (2 * alpha) * cos (4 * alpha) * ⋯ * cos (2^n * alpha) = 
     sin (2^(n+1) * alpha) / (2^(n+1) * sin alpha))

theorem prove_trig_identity (alpha : ℝ) (n : ℕ) (h : sin alpha ≠ 0) : 
  cos alpha * cos (2 * alpha) * cos (4 * alpha) * (List.prod (List.map (fun k : ℕ => cos ((2 ^ k) * alpha)) (List.range (n + 1)))) = 
  sin (2 ^ (n + 1) * alpha) / (2 ^ (n + 1) * sin alpha) :=
by
  sorry

end prove_trig_identity_l569_569843


namespace radius_increase_is_0_31_l569_569000

noncomputable def increase_in_radius (initial_radius : ℝ) (odometer_summer : ℝ) (odometer_winter : ℝ) (miles_to_inches : ℝ) : ℝ :=
  let circumference_summer := 2 * Real.pi * initial_radius
  let distance_per_rotation_summer := circumference_summer / miles_to_inches
  let rotations_summer := odometer_summer / distance_per_rotation_summer
  let rotations_winter := odometer_winter / distance_per_rotation_summer
  let distance_winter := rotations_winter * distance_per_rotation_summer
  let new_radius := (distance_winter * miles_to_inches) / (2 * rotations_winter * Real.pi)
  new_radius - initial_radius

theorem radius_increase_is_0_31 : 
    increase_in_radius 16 530 520 63360 = 0.31 := 
by
    sorry

end radius_increase_is_0_31_l569_569000


namespace factorize_polynomial_l569_569311

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem factorize_polynomial :
  (zeta^3 = 1) ∧ (zeta^2 + zeta + 1 = 0) → (x : ℂ) → (x^15 + x^10 + x) = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1)
:= sorry

end factorize_polynomial_l569_569311


namespace Sam_bought_cards_l569_569146

theorem Sam_bought_cards (original_cards current_cards : ℕ) 
  (h1 : original_cards = 87) (h2 : current_cards = 74) : 
  original_cards - current_cards = 13 :=
by
  -- The 'sorry' here means the proof is omitted.
  sorry

end Sam_bought_cards_l569_569146


namespace value_of_one_stamp_l569_569446

theorem value_of_one_stamp (matches_per_book : ℕ) (initial_stamps : ℕ) (trade_matchbooks : ℕ) (stamps_left : ℕ) :
  matches_per_book = 24 → initial_stamps = 13 → trade_matchbooks = 5 → stamps_left = 3 →
  (trade_matchbooks * matches_per_book) / (initial_stamps - stamps_left) = 12 :=
by
  intros h1 h2 h3 h4
  -- Insert the logical connection assertions here, concluding with the final proof step.
  sorry

end value_of_one_stamp_l569_569446


namespace root_of_quadratic_l569_569896

theorem root_of_quadratic (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0)
  (h4 : b^2 - 4 * a * c = 0) : 
  ∃ (r : ℝ), r = -2 + sqrt 3 ∧ is_root (λ x => a * x^2 + b * x + c) r :=
by
  sorry

end root_of_quadratic_l569_569896


namespace length_of_EG_l569_569430

variable (Rectangle : Type) [RectangularStructure Rectangle] -- Rectangle type with necessary properties
variable (semicircle : Rectangle → Type) -- Map rectangle to its semicircle
variable (containment : ∀ (r : Rectangle), contains r (semicircle r)) -- The rectangle contains the semicircle
variable (line : Type) -- A line type
variable (intersect : line → semicircle → Rectangle → Type) -- Intersection function

-- Given data points
variable {r : Rectangle}
variable {ℓ : line}
variable (EV EP VF : ℝ)
variable (H_ev : EV = 70) (H_ep : EP = 105) (H_vf : VF = 210)

-- The main proof problem statement
theorem length_of_EG (EG : ℝ) : 
  let area_ratio := 3/1 in
  -- Conditions
  let h1 : contains r (semicircle r) := containment r in
  -- Final assertion proving length of EG
  EG = 280 * Real.sqrt 3 :=
sorry

end length_of_EG_l569_569430


namespace sum_cos_4_eq_zero_l569_569131

theorem sum_cos_4_eq_zero 
  (x y z : ℝ)
  (h1 : cos (2 * x) + cos (2 * y) + cos (2 * z) = 0)
  (h2 : sin (2 * x) + sin (2 * y) + sin (2 * z) = 0) :
  cos (4 * x) + cos (4 * y) + cos (4 * z) = 0 :=
sorry

end sum_cos_4_eq_zero_l569_569131


namespace number_of_four_digit_cubes_divisible_by_16_l569_569386

theorem number_of_four_digit_cubes_divisible_by_16 :
  (finset.Icc 5 10).card = 6 :=
by sorry

end number_of_four_digit_cubes_divisible_by_16_l569_569386


namespace oranges_to_pears_l569_569812

-- Define the equivalence relation between oranges and pears
def equivalent_weight (orange pear : ℕ) : Prop := 4 * pear = 3 * orange

-- Given:
-- 1. 4 oranges weigh the same as 3 pears
-- 2. Jimmy has 36 oranges
-- Prove that 27 pears are required to balance the weight of 36 oranges
theorem oranges_to_pears (orange pear : ℕ) (h : equivalent_weight 1 1) :
  (4 * pear = 3 * orange) → equivalent_weight 36 27 :=
by
  sorry

end oranges_to_pears_l569_569812


namespace contrapositive_even_l569_569871

theorem contrapositive_even (a b : ℤ) :
  (¬ (∃ i : ℤ, a = 2 * i) ∧ ¬ (∃ j : ℤ, b = 2 * j)) → 
  (¬ (∃ k : ℤ, a * b = 2 * k)) :=
begin
  sorry
end

end contrapositive_even_l569_569871


namespace train_speed_l569_569622

theorem train_speed (train_length bridge_and_train_length : ℕ) 
(seconds : ℕ)
(h_train_length : train_length = 130)
(h_bridge_and_train_length : bridge_and_train_length = 245)
(h_seconds : seconds = 30) :
  let speed := (train_length + bridge_and_train_length) / seconds * 3.6 in speed = 45 :=
by {
  sorry
}

end train_speed_l569_569622


namespace area_excluding_hole_l569_569250

theorem area_excluding_hole (x : ℝ) : 
  (2 * x + 8) * (x + 6) - (2 * x - 2) * (x - 1) = 24 * x + 46 :=
by
  sorry

end area_excluding_hole_l569_569250


namespace actual_diameter_of_tissue_is_0_03_mm_l569_569927

-- Defining necessary conditions
def magnified_diameter_meters : ℝ := 0.15
def magnification_factor : ℝ := 5000
def meters_to_millimeters : ℝ := 1000

-- Prove that the actual diameter of the tissue is 0.03 millimeters
theorem actual_diameter_of_tissue_is_0_03_mm :
  (magnified_diameter_meters * meters_to_millimeters) / magnification_factor = 0.03 := 
  sorry

end actual_diameter_of_tissue_is_0_03_mm_l569_569927


namespace sum_first_3m_terms_l569_569543

variable {α : Type*} [LinearOrderedField α]

def sum_first_m_terms (a : ℕ → α) (m : ℕ) : α :=
  ∑ i in finset.range m, a i

theorem sum_first_3m_terms (a : ℕ → α) (m : ℕ) 
  (h₁ : sum_first_m_terms a m = 30)
  (h₂ : sum_first_m_terms a (2 * m) = 100) :
  sum_first_m_terms a (3 * m) = 210 :=
sorry

end sum_first_3m_terms_l569_569543


namespace original_cost_of_each_bag_l569_569319

theorem original_cost_of_each_bag (C : ℕ) (hC : C % 13 = 0) (h4 : (85 * C) % 400 = 0) : C / 5 = 208 := by
  sorry

end original_cost_of_each_bag_l569_569319


namespace power_sum_is_integer_l569_569824

theorem power_sum_is_integer (x : ℝ) (h : x ≠ 0) (h_int : x + (1/x) ∈ ℤ) : 
  ∀ n : ℕ, x^n + (1 / x^n) ∈ ℤ :=
by
  sorry

end power_sum_is_integer_l569_569824


namespace find_ordered_triple_l569_569249

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
(3 * Real.cos t - 2 * Real.sin t, 5 * Real.sin t)

theorem find_ordered_triple :
  ∃ (a b c : ℝ), 
  a = 1 / 9 ∧ 
  b = 4 / 45 ∧ 
  c = 13 / 225 ∧ 
  ∀ t : ℝ, let (x, y) := parametric_curve t in 
  a * x^2 + b * x * y + c * y^2 = 1 :=
by 
  use 1 / 9, 4 / 45, 13 / 225
  intros t
  simp [parametric_curve]
  sorry

end find_ordered_triple_l569_569249


namespace area_outside_doghouse_l569_569513

/-- Spots doghouse problem conditions and theorem -/
def spotsDoghouseArea : ℝ := sorry

-- Define the conditions
def side_length := 1 -- yards
def tether_length := 2 -- yards
def angle_large_sector := 240 -- degrees
def angle_small_sector := 60 -- degrees

-- Theorems that follow from conditions
noncomputable def area_of_accessible_region : ℝ :=
  π * tether_length^2 * (angle_large_sector / 360) +
  2 * (π * side_length^2 * (angle_small_sector / 360))

theorem area_outside_doghouse : area_of_accessible_region = 3 * π :=
by sorry

end area_outside_doghouse_l569_569513


namespace chemist_mixed_pure_water_l569_569244

variables (x : ℝ)

-- Define the given conditions
def initial_salt := 0.30 -- initial amount of salt in liters
def final_volume := 1 + x -- final volume after adding x liters of pure water
def final_salt := 0.15 * final_volume -- final concentration of salt in the new volume

-- Statement to prove
theorem chemist_mixed_pure_water : initial_salt = final_salt → x = 1 :=
by
  sorry

end chemist_mixed_pure_water_l569_569244


namespace greatest_C_value_l569_569123

-- Let \(\alpha\) be an arbitrary positive real number
variable (α : ℝ) (α_pos : 0 < α)

-- The main theorem
theorem greatest_C_value 
    (x y z : ℝ) 
    (x_pos : 0 < x) 
    (y_pos : 0 < y) 
    (z_pos : 0 < z) 
    (h : x * y + y * z + z * x = α) : 
    (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) :=
begin
    sorry
end

end greatest_C_value_l569_569123


namespace two_point_five_one_million_in_scientific_notation_l569_569664

theorem two_point_five_one_million_in_scientific_notation :
  (2.51 * 10^6 : ℝ) = 2.51e6 := 
sorry

end two_point_five_one_million_in_scientific_notation_l569_569664


namespace friend_payment_ratio_l569_569444

theorem friend_payment_ratio (james_meal friend_meal : ℝ) (tip_rate : ℝ) (james_paid : ℝ) :
  james_meal = 16 →
  friend_meal = 14 →
  tip_rate = 0.20 →
  james_paid = 21 →
  (let tip := (james_meal + friend_meal) * tip_rate in
   let total_bill_with_tip := (james_meal + friend_meal) + tip in
   let friend_payment := (friend_meal + ((james_meal + friend_meal + tip) - james_paid)) in
   friend_payment / total_bill_with_tip) = 5 / 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end friend_payment_ratio_l569_569444


namespace total_matches_l569_569166

theorem total_matches (n : ℕ) (h : n = 7) : 
  ∑ i in range n, i = 21 :=
by { rw h, sorry }

end total_matches_l569_569166


namespace find_b_l569_569325

theorem find_b (a b : ℝ) (h1 : 2 * (a * (a - 1) / 2) - ((a) - 1) * 6 = 0)
  (h2 : (6.choose 1) * b = -12) : b = -2 :=
by
  -- We state the problem and provide an initial placeholder for the proof.
  sorry

end find_b_l569_569325


namespace arithmetic_sequence_common_difference_l569_569792

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a6 : a 6 = 5) (h_a10 : a 10 = 6) : 
  (a 10 - a 6) / 4 = 1 / 4 := 
by
  sorry

end arithmetic_sequence_common_difference_l569_569792


namespace min_value_proof_l569_569820

noncomputable def min_value (α γ : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2

theorem min_value_proof (α γ : ℝ) : ∃ α γ : ℝ, min_value α γ = 36 :=
by
  use (Real.arcsin 12/13), (Real.pi/2 - Real.arcsin 12/13)
  sorry

end min_value_proof_l569_569820


namespace probability_correct_l569_569552

-- Define the set and the probability calculation
def set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Function to check if the difference condition holds
def valid_triplet (a b c: ℕ) : Prop := a < b ∧ b < c ∧ c - a = 4

-- Total number of ways to pick 3 numbers and ways that fit the condition
noncomputable def total_ways : ℕ := Nat.choose 9 3
noncomputable def valid_ways : ℕ := 5 * 2

-- Calculate the probability
noncomputable def probability : ℚ := valid_ways / total_ways

-- The theorem statement
theorem probability_correct : probability = 5 / 42 := by sorry

end probability_correct_l569_569552


namespace solution_set_of_inequality_system_l569_569899

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 2 ≤ 3 ∧ 1 + x > -2) ↔ (-3 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_of_inequality_system_l569_569899


namespace incorrect_statement_l569_569632

variables (LocustOocyte Meiosis Slide FixedSlide PeachFlower Stamen Pistil Synapsis HomologousChromosomes OnionRootTip Tetrad EquatorialPlate : Prop)

axiom H1 : LocustOocyte → Meiosis → FixedSlide
axiom H2 : (PeachFlower → Stamen → Meiosis) → (PeachFlower → Pistil → Meiosis) → (∀ x : Slide, PeachFlower → Stamen → Meiosis x → PeachFlower → Pistil → Meiosis x → x = true )
axiom H3 : ∀ x : Slide, Meiosis x → Synapsis HomologousChromosomes → x = true
axiom H4 : OnionRootTip → Slide → Tetrad → EquatorialPlate

theorem incorrect_statement : OnionRootTip → Slide → Tetrad → ¬Meiosis :=
by sorry

end incorrect_statement_l569_569632


namespace Valley_High_School_festival_l569_569567

theorem Valley_High_School_festival :
  ∃ (g b : ℕ), g + b = 1500 ∧
               (3 * g)/4 + (2 * b)/5 = 900 ∧
               (3 * g)/4 = 643 :=
by
  -- Introduce variables for girls and boys
  let g := 857
  let b := 1500 - g

  -- Hypothesize conditions
  have h1 : g + b = 1500 := by sorry
  have h2 : (3 * g) / 4 + (2 * b) / 5 = 900 := by sorry
  have h3 : (3 * g) / 4 = 643 := by sorry

  -- Assert the existence of g and b meeting those conditions
  use [g, b]
  exact ⟨h1, h2, h3⟩

end Valley_High_School_festival_l569_569567


namespace total_packs_l569_569482

theorem total_packs (cards_per_person : ℕ) (cards_per_pack : ℕ) (people_count : ℕ) (cards_per_person_eq : cards_per_person = 540) (cards_per_pack_eq : cards_per_pack = 20) (people_count_eq : people_count = 4) :
  (cards_per_person / cards_per_pack) * people_count = 108 :=
by
  sorry

end total_packs_l569_569482


namespace number_of_people_l569_569427

theorem number_of_people (x : ℕ) (H : x * (x - 1) = 72) : x = 9 :=
sorry

end number_of_people_l569_569427


namespace successful_purchase_probability_l569_569304

theorem successful_purchase_probability
  (m n : ℕ)
  (h : m ≥ n) :
  let total_ways := Nat.choose (m + n) m,
      favorable_ways := Nat.choose (m + n) m - Nat.choose (m + n) (m + 1) in
  (favorable_ways / total_ways : ℚ) = (m - n + 1) / (m + 1) := by
  sorry

end successful_purchase_probability_l569_569304


namespace log_bounds_l569_569675

theorem log_bounds (a b : ℤ) (h₁ : a = 6) (h₂ : b = 7) : 
  a < Real.logB 5 125000 ∧ Real.logB 5 125000 < b ∧ a + b = 13 :=
by
  sorry

end log_bounds_l569_569675


namespace probability_of_selection_l569_569266

/-- A school selects 80 students for a discussion from a total of 883 students. First, 3 people are eliminated using simple random sampling, and then 80 are selected from the remaining 880 using systematic sampling. Prove that the probability of each person being selected is 80/883. -/
theorem probability_of_selection (total_students : ℕ) (students_eliminated : ℕ) (students_selected : ℕ) 
  (h_total : total_students = 883) (h_eliminated : students_eliminated = 3) (h_selected : students_selected = 80) :
  ((total_students - students_eliminated) * students_selected) / (total_students * (total_students - students_eliminated)) = 80 / 883 :=
by
  sorry

end probability_of_selection_l569_569266


namespace direct_proportion_function_l569_569329

theorem direct_proportion_function (m : ℝ) : 
  (m^2 + 2 * m ≠ 0) ∧ (m^2 - 3 = 1) → m = 2 :=
by {
  sorry
}

end direct_proportion_function_l569_569329


namespace ratio_of_volumes_l569_569119

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r ^ 2 * h

theorem ratio_of_volumes :
  let large_can_volume := volume_cylinder 5 20
  let small_can_volume := volume_cylinder (5 / 2) 10
  let total_large_volume := 6 * large_can_volume
  let total_small_volume := 12 * small_can_volume
  (total_large_volume / total_small_volume) = 4 :=
by
  sorry

end ratio_of_volumes_l569_569119


namespace find_pairs_l569_569004

theorem find_pairs (x y : ℕ) (hy : y < 10) :
  ((x, y) = (1, 3) ∨ (x, y) = (0, 9)) ↔ (x ≥ 0 ∧ (∃ r : ℚ, r = x + 1 / y ∧ r = x + Real.to_rational (Real.repeat 1 y - 1))) :=
by
  sorry

end find_pairs_l569_569004


namespace investment_amount_correct_l569_569935

-- Lean statement definitions based on conditions
def cost_per_tshirt : ℕ := 3
def selling_price_per_tshirt : ℕ := 20
def tshirts_sold : ℕ := 83
def total_revenue : ℕ := tshirts_sold * selling_price_per_tshirt
def total_cost_of_tshirts : ℕ := tshirts_sold * cost_per_tshirt
def investment_in_equipment : ℕ := total_revenue - total_cost_of_tshirts

-- Theorem statement
theorem investment_amount_correct : investment_in_equipment = 1411 := by
  sorry

end investment_amount_correct_l569_569935


namespace largest_set_S_card_l569_569616

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_unique_triangle (S : List (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (a1 b1 c1 a2 b2 c2 : ℕ), (a1, b1, c1) ∈ S → (a2, b2, c2) ∈ S → ¬((a1, b1, c1) ≃ (a2, b2, c2))

def S : List (ℕ × ℕ × ℕ) :=
  [(4, 4, 4), (4, 4, 3), (4, 4, 2), (4, 4, 1), (4, 3, 3), (4, 3, 2), (3, 3, 3), (3, 3, 2), (3, 2, 2)]

theorem largest_set_S_card : is_unique_triangle S → S.length = 9 := by
  sorry

end largest_set_S_card_l569_569616


namespace batsman_highest_score_l569_569945

theorem batsman_highest_score (H L : ℕ) 
  (h₁ : (40 * 50 = 2000)) 
  (h₂ : (H = L + 172))
  (h₃ : (38 * 48 = 1824)) :
  (2000 = 1824 + H + L) → H = 174 :=
by 
  sorry

end batsman_highest_score_l569_569945


namespace roots_quadratic_eq_k_l569_569658

theorem roots_quadratic_eq_k (k : ℝ) :
  (∀ x : ℝ, (5 * x^2 + 20 * x + k = 0) ↔ (x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10)) →
  k = 17 := by
  intro h
  sorry

end roots_quadratic_eq_k_l569_569658


namespace birds_problem_l569_569175

theorem birds_problem 
  (x y z : ℕ) 
  (h1 : x + y + z = 30) 
  (h2 : (1 / 3 : ℚ) * x + (1 / 2 : ℚ) * y + 2 * z = 30) 
  : x = 9 ∧ y = 10 ∧ z = 11 := 
  by {
  -- Proof steps would go here
  sorry
}

end birds_problem_l569_569175


namespace tan_add_pi_over_4_l569_569790

theorem tan_add_pi_over_4 (α : ℝ)
  (O : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM_coords : M = (-1, real.sqrt 3))
  (hO_origin : O = (0, 0))
  (hM_on_terminal_side : ∃ α, M.1 = -1 ∧ M.2 = real.sqrt 3 ∧ M = (cos α, sin α)) :
  real.tan (α + real.pi / 4) = real.sqrt 3 - 2 :=
by
  sorry

end tan_add_pi_over_4_l569_569790


namespace four_digit_perfect_cubes_divisible_by_16_l569_569382

theorem four_digit_perfect_cubes_divisible_by_16 : (∃ k : ℕ, k = 3) :=
by
  let possible_cubes := [12 ^ 3, 16 ^ 3, 20 ^ 3]
  have h1 : 12 ^ 3 = 1728 := by norm_num
  have h2 : 16 ^ 3 = 4096 := by norm_num
  have h3 : 20 ^ 3 = 8000 := by norm_num

  have h4 : (1728, 4096, 8000).all (λ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 16 = 0)
    := by norm_num

  use 3
  trivial

end four_digit_perfect_cubes_divisible_by_16_l569_569382


namespace polynomial_factorization_l569_569019

theorem polynomial_factorization 
  (a b c d : ℤ)
  (h_expand : (λ x : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d)) = (λ x : ℝ, x^4 + x^3 - 5*x^2 + x - 6)) :
  a + b + c + d = -4 := by
  sorry

end polynomial_factorization_l569_569019


namespace parabola_intersection_count_l569_569296

-- Definitions corresponding to the conditions
def focus := (0, 0)
def valid_a : set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def valid_b : set ℤ := {-4, -3, -2, 1, 2, 3, 4}

theorem parabola_intersection_count :
  let num_parabolas := 35,
      pairs_of_parabolas := num_parabolas.choose 2,
      invalid_pairs := 63 in
  2 * (pairs_of_parabolas - invalid_pairs) = 1064 :=
by
  let num_parabolas := 35
  let pairs_of_parabolas := num_parabolas.choose 2
  let invalid_pairs := 63
  have h1 : pairs_of_parabolas = 595 := by sorry
  have h2 : invalid_pairs = 63 := by sorry
  have h3 : 2 * (595 - 63) = 1064 := by sorry
  show 2 * (pairs_of_parabolas - invalid_pairs) = 1064, by exact h3

end parabola_intersection_count_l569_569296


namespace largest_integer_with_conditions_l569_569572

-- Define what it means for an integer to have only odd digits
def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ nat.digits 10 n, d % 2 = 1

-- Define what it means for an integer to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Define what it means for an integer to be less than 10,000
def is_less_than_10000 (n : ℕ) : Prop :=
  n < 10000

-- Define the largest integer that meets all the conditions
theorem largest_integer_with_conditions : 
  ∃ n : ℕ, has_only_odd_digits n ∧ is_multiple_of_11 n ∧ is_less_than_10000 n ∧ 
           ∀ m : ℕ, has_only_odd_digits m ∧ is_multiple_of_11 m ∧ is_less_than_10000 m → m ≤ n :=
sorry

end largest_integer_with_conditions_l569_569572


namespace slope_perpendicular_to_given_line_l569_569688

-- Define the given line
def given_line (x y : ℝ) := 3 * x - 4 * y = 8

-- Define the slope-intercept form of the line and its slope
def slope_of_given_line := (3 : ℝ) / (4 : ℝ)

-- Define the slope of the line perpendicular to the given line
def perpendicular_slope := -1 / slope_of_given_line

-- Theorem: The slope of the line perpendicular to the line 3x - 4y = 8 is -4/3
theorem slope_perpendicular_to_given_line : perpendicular_slope = - (4 : ℝ) / (3 : ℝ) :=
by sorry

end slope_perpendicular_to_given_line_l569_569688


namespace order_of_animals_l569_569114
noncomputable theory

-- Define responses of each animal
def jackal_answer (question : String) : Bool := false -- Always lies
def lion_answer (question : String) : Bool := not (question = "Are you Jackal?" || question = "Are you Giraffe?")
def parrot_answer (previous_answer : Bool) (rand_answer : Bool) (asked_first : Bool) : Bool := 
  if asked_first then rand_answer else previous_answer
def giraffe_answer (previous_question : String) (previous_answer : Bool) (asked_first : Bool) : Bool :=
  if asked_first then false else -- Randomly chosen initially to simplify modeling
  match previous_question with
  | "Are you Jackal?" := not previous_answer
  | "Are you Giraffe?" := previous_answer
  | "Are you Parrot?" := false -- Known because of simplifying assumption
  | _ := false

-- Define the proof problem 
theorem order_of_animals (responses_to_jackal : Nat -> Bool) 
                          (responses_to_giraffe : Nat -> Bool) 
                          (responses_to_parrot : Nat -> Bool) :
  order_of_animals = ["Parrot", "Lion", "Giraffe", "Jackal"] :=
by
  -- The proof is non-computable as it requires modeling responses, hence 'sorry'
  sorry

end order_of_animals_l569_569114


namespace find_y_l569_569076

-- Define the conditions for x and y
variables {x y : ℝ}

-- First condition: x - 2y = 10
def condition_1 := x - 2 * y = 10

-- Second condition: x * y = 40
def condition_2 := x * y = 40

-- The proof statement: y = 2.5 given the conditions 
theorem find_y (h1 : condition_1) (h2 : condition_2) : y = 2.5 :=
sorry

end find_y_l569_569076


namespace line_through_points_l569_569326

theorem line_through_points (x1 y1 x2 y2 : ℝ) :
  (3 * x1 - 4 * y1 - 2 = 0) →
  (3 * x2 - 4 * y2 - 2 = 0) →
  (∀ x y : ℝ, (x = x1) → (y = y1) ∨ (x = x2) → (y = y2) → 3 * x - 4 * y - 2 = 0) :=
by
  sorry

end line_through_points_l569_569326


namespace train_speed_is_108_kmh_l569_569953

def distance_in_kilometers (distance_meters : ℕ) : ℝ := distance_meters * 0.001

def time_in_hours (time_seconds : ℕ) : ℝ := time_seconds / 3600

def train_speed_kmh (distance : ℕ) (time : ℕ) : ℝ :=
  distance_in_kilometers distance / time_in_hours time

theorem train_speed_is_108_kmh : 
  train_speed_kmh 180 6 = 108 :=
by
  sorry

end train_speed_is_108_kmh_l569_569953


namespace length_OP_is_14_8_l569_569959

/-- A circle with center O has radius 20. Chords AB and CD intersect at point P.
    Chord AB has length 24 and chord CD has length 18. The distance between the midpoints of
    these chords is 10. Prove that the length of OP is approximately 14.8. -/
noncomputable def length_OP (O A B C D P E F : Point) (r : ℝ) (l_AB : ℝ) (l_CD : ℝ) (d_midpoints : ℝ) : ℝ :=
  let BE := (l_AB / 2)
  let CF := (l_CD / 2)
  let OE := Real.sqrt (r^2 - (BE)^2)
  let OF := Real.sqrt (r^2 - (CF)^2)
  let OP := Real.sqrt ((OE - OF)^2 + d_midpoints^2)
  OP

theorem length_OP_is_14_8 :
  ∃ (O A B C D P E F : Point),
    length_OP O A B C D P E F 20 24 18 10 ≈ 14.8 :=
by
  -- construct the example
  sorry -- providing a concrete construction of points

end length_OP_is_14_8_l569_569959


namespace cubic_roots_result_l569_569214

theorem cubic_roots_result (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 64 + b * 16 + c * 4 + d = 0) (h₃ : a * (-27) + b * 9 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end cubic_roots_result_l569_569214


namespace polygon_side_count_l569_569542

theorem polygon_side_count (n : ℕ) 
    (h : (n - 2) * 180 + 1350 - (n - 2) * 180 = 1350) : n = 9 :=
by
  sorry

end polygon_side_count_l569_569542


namespace four_digit_number_l569_569903

-- Define the digits a, b, c, d, and their constraints
variables {a b c d : ℕ}
variable h1 : 1 ≤ a ∧ a ≤ 9
variable h2 : 0 ≤ b ∧ b ≤ 9
variable h3 : 0 ≤ c ∧ c ≤ 9
variable h4 : 0 ≤ d ∧ d ≤ 9

-- Define the conditions
variable h_sum_first_two : a + b = c + d
variable h_sum_first_last : a + d = c
variable h_sum_second_fourth : b + d = 2 * (a + c)

theorem four_digit_number :
  1000 * a + 100 * b + 10 * c + d = 1854 :=
by
  -- The actual proof will go here using the conditions h1 to h4.
  sorry

end four_digit_number_l569_569903


namespace a_leq_neg2_l569_569351

-- Define the function f(x)
def f (x a : ℝ) := x^2 - 2 * a * x + 3 

-- Define the derivative of f(x)
def f' (x a : ℝ) := 2 * x - 2 * a 

-- Define the increasing condition for f(x) on the interval (-2, 2)
def increasing_on_interval (a : ℝ) : Prop :=
  ∀ x ∈ Ioo (-2 : ℝ) (2 : ℝ), f' x a > 0 

-- The theorem statement
theorem a_leq_neg2 (a : ℝ) (h : increasing_on_interval a) : a ≤ -2 := 
sorry

end a_leq_neg2_l569_569351


namespace increase_in_circumference_of_cylinders_l569_569267

noncomputable def verify_increase_in_circumference (x : ℝ) : Prop :=
     ∃ x ≥ 0, ((60 / (3 + x)) + 3 = 60 / (2 + x)) ∧ x = 2

theorem increase_in_circumference_of_cylinders : verify_increase_in_circumference 2 :=
by
  use 2
  split
  · exact le_refl 2
  split
  · 
    sorry
  · 
    sorry


end increase_in_circumference_of_cylinders_l569_569267


namespace four_digit_perfect_cubes_divisible_by_16_l569_569384

theorem four_digit_perfect_cubes_divisible_by_16 : (∃ k : ℕ, k = 3) :=
by
  let possible_cubes := [12 ^ 3, 16 ^ 3, 20 ^ 3]
  have h1 : 12 ^ 3 = 1728 := by norm_num
  have h2 : 16 ^ 3 = 4096 := by norm_num
  have h3 : 20 ^ 3 = 8000 := by norm_num

  have h4 : (1728, 4096, 8000).all (λ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 16 = 0)
    := by norm_num

  use 3
  trivial

end four_digit_perfect_cubes_divisible_by_16_l569_569384


namespace complete_square_solution_l569_569929

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end complete_square_solution_l569_569929


namespace number_of_four_digit_cubes_divisible_by_16_l569_569388

theorem number_of_four_digit_cubes_divisible_by_16 :
  (finset.Icc 5 10).card = 6 :=
by sorry

end number_of_four_digit_cubes_divisible_by_16_l569_569388


namespace four_digit_cubes_divisible_by_16_l569_569389

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l569_569389


namespace three_right_angled_triangles_l569_569495

theorem three_right_angled_triangles 
  (a b c : ℕ)
  (h_area : 1/2 * (a * b) = 2 * (a + b + c))
  (h_pythagorean : a^2 + b^2 = c^2)
  (h_int_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (a = 9 ∧ b = 40 ∧ c = 41) ∨ 
  (a = 10 ∧ b = 24 ∧ c = 26) ∨ 
  (a = 12 ∧ b = 16 ∧ c = 20) := 
sorry

end three_right_angled_triangles_l569_569495


namespace equivalent_discount_l569_569609

variables (P d1 d2 : ℝ)

theorem equivalent_discount (hP : P = 50) (hd1 : d1 = 0.15) (hd2 : d2 = 0.30) :
  1 - ((P * (1 - d1) * (1 - d2)) / P) = 0.405 :=
by
  rw [hP, hd1, hd2]
  have h1 : P * (1 - d1) = 50 * 0.85 := by rw [hP, hd1]; norm_num
  rw [h1]
  have h2 : 50 * 0.85 * (1 - d2) = 42.5 * 0.70 := by rw [hd2]; norm_num
  rw [h2]
  have h3 : 42.5 * 0.70 = 29.75 := by norm_num
  rw [h3]
  have h4 : (29.75 / 50) = 0.595 := by norm_num
  rw [h4]
  norm_num


end equivalent_discount_l569_569609


namespace complex_solutions_num_complex_solutions_l569_569314

theorem complex_solutions (z : ℂ) :
  (z^4 - 1) / (z^3 - 2*z^2 + z - 2) = 0 ↔ 
  z = complex.I ∨ z = -complex.I :=
begin
  sorry
end

theorem num_complex_solutions :
  {z : ℂ | (z^4 - 1) / (z^3 - 2*z^2 + z - 2) = 0}.to_finset.card = 2 :=
begin
  sorry
end

end complex_solutions_num_complex_solutions_l569_569314


namespace product_of_positive_real_solutions_l569_569411

noncomputable def cis := complex.exp

theorem product_of_positive_real_solutions :
  (∏ (x : ℂ) in { x | x^8 = -256 ∧ x.re > 0 }, x) = 4 * real.sqrt 2 :=
by
  sorry

end product_of_positive_real_solutions_l569_569411


namespace dice_probability_two_pairs_or_triplet_one_pair_l569_569666

theorem dice_probability_two_pairs_or_triplet_one_pair 
  (seven_dice : Fin 7 → Fin 6) :
  let total_outcomes : ℕ := 6 ^ 7 in
  let favorable_outcomes : ℕ := 
    (15 * 21 * 10 * 24) + (30 * 35 * 6 * 12) in
  (favorable_outcomes / total_outcomes : ℚ) = 525 / 972 := 
by 
  sorry

end dice_probability_two_pairs_or_triplet_one_pair_l569_569666


namespace prove_inequality_l569_569047

noncomputable def f (x : ℝ) : ℝ := x / (1 - x^2)

theorem prove_inequality 
  (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : 0 < b) (h4 : b < 1)
  (h5 : 0 < c) (h6 : c < 1)
  (h_abc : a * b * c = real.sqrt 3 / 9) : 
  f a + f b + f c ≥ 3 * real.sqrt 3 / 2 :=
sorry

end prove_inequality_l569_569047


namespace urn_final_state_l569_569635

/-- Validate the final state of an urn, starting with 70 white and 130 black marbles,
    and subjected to specific marble replacement operations -/
theorem urn_final_state :
  ∃ (w b : ℕ), w = 4 ∧ b = 0 ∧
  ∀ n : ℕ, 
    (∀ _w _b : ℕ, 
      ( (_w = 70 ∧ _b = 130) ∧ 
      (n = 23 ∧ ∀ i : ℕ, i < n → 
        ( _w = _w - 4 ∧ _b = _b + 1) ∨
        ( (_w = _w + 1) ∧ (_b = _b + 1)) ∨
        ( (_w = _w + 2) ∧ (_b = _b - 2)) ∨
        ( (_w = _w - 1) ∧ (_b = _b + 2)) ∨
        ( (_w = _w + 3) ∧ (_b = _b - 4))
      ) 
    )
    → (w = 4 ∧ b = 0)) sorry

end urn_final_state_l569_569635


namespace johns_out_of_pocket_l569_569813

noncomputable def total_cost_after_discounts (computer_cost gaming_chair_cost accessories_cost : ℝ) 
  (comp_discount gaming_discount : ℝ) (tax : ℝ) : ℝ :=
  let comp_price := computer_cost * (1 - comp_discount)
  let chair_price := gaming_chair_cost * (1 - gaming_discount)
  let pre_tax_total := comp_price + chair_price + accessories_cost
  pre_tax_total * (1 + tax)

noncomputable def total_selling_price (playstation_value playstation_discount bicycle_price : ℝ) (exchange_rate : ℝ) : ℝ :=
  let playstation_price := playstation_value * (1 - playstation_discount)
  (playstation_price * exchange_rate) / exchange_rate + bicycle_price

theorem johns_out_of_pocket (computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax 
  playstation_value playstation_discount bicycle_price exchange_rate : ℝ) :
  computer_cost = 1500 →
  gaming_chair_cost = 400 →
  accessories_cost = 300 →
  comp_discount = 0.2 →
  gaming_discount = 0.1 →
  tax = 0.05 →
  playstation_value = 600 →
  playstation_discount = 0.2 →
  bicycle_price = 200 →
  exchange_rate = 100 →
  total_cost_after_discounts computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax -
  total_selling_price playstation_value playstation_discount bicycle_price exchange_rate = 1273 := by
  intros
  sorry

end johns_out_of_pocket_l569_569813


namespace Bea_age_cannot_be_determined_l569_569450

noncomputable def Kiarra_age : ℕ := 30
def Bea_is_younger_than_Kiarra (Bea_age : ℕ) : Prop := Bea_age < Kiarra_age
def Job_age (Bea_age : ℕ) : ℕ := 3 * Bea_age
def Figaro_age (Bea_age : ℕ) : ℕ := Job_age(Bea_age) + 7
def Harry_age (Bea_age : ℕ) : ℕ := Figaro_age(Bea_age) / 2

theorem Bea_age_cannot_be_determined : ∃ (Bea_age : ℕ), Bea_is_younger_than_Kiarra Bea_age :=
by
  sorry

end Bea_age_cannot_be_determined_l569_569450


namespace function_analysis_l569_569052

noncomputable theory

open Real Set

def y_max (a b x : ℝ) : ℝ := a - b * cos (3 * x)
def y_min (a b x : ℝ) : ℝ := a - b * cos (3 * x)
def f (a b x : ℝ) : ℝ := 2 * sin (a * π / 3 - 2 * b * x)

theorem function_analysis 
  (a b : ℝ) 
  (h1 : a + b = 3 / 2) 
  (h2 : a - b = -1 / 2) 
  (hb : b > 0) 
  (ha : a = 1 / 2) 
  (hb_eq : b = 1) :

  -- Proof of the period and extremum values of y
  (∀ x, y_max a 1 x = a - 1 * cos (3 * x)) ∧
  (∀ x, y_min a 1 x = a - 1 * cos (3 * x)) ∧
  (∀ x, (-4 * a * sin (3 * 1 * x) = -2 * sin (3 * x))) ∧
  (sin (3 * x) has a period of (2 * π) / 3) ∧
  (∀ k : ℤ, ∃ x, x = 2 * k * π / 3 + π / 6) ∧ 
  (∀ k : ℤ, ∃ x, x = 2 * k * π / 3 - π / 6) ∧ 

  -- Proof of monotonic intervals for f
  (∀ k, ∀ x, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 → 2 * sin (π / 6 - 2 * x) is decreasing) ∧
  (∀ k, ∀ x, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 → 2 * sin (π / 6 - 2 * x) is increasing) := 
sorry

end function_analysis_l569_569052


namespace total_students_registered_l569_569860

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end total_students_registered_l569_569860


namespace cost_price_proof_l569_569943

noncomputable def selling_price : Real := 12000
noncomputable def discount_rate : Real := 0.10
noncomputable def new_selling_price : Real := selling_price * (1 - discount_rate)
noncomputable def profit_rate : Real := 0.08

noncomputable def cost_price : Real := new_selling_price / (1 + profit_rate)

theorem cost_price_proof : cost_price = 10000 := by sorry

end cost_price_proof_l569_569943


namespace compute_expression_l569_569999

theorem compute_expression :
  ( ((15 ^ 15) / (15 ^ 10)) ^ 3 * 5 ^ 6 ) / (25 ^ 2) = 3 ^ 15 * 5 ^ 17 :=
by
  -- We'll use sorry here as proof is not required
  sorry

end compute_expression_l569_569999


namespace impossible_transformation_l569_569431

variable (G : Type) [Group G]

/-- Initial word represented by 2003 'a's followed by 'b' --/
def initial_word := "aaa...ab"

/-- Transformed word represented by 'b' followed by 2003 'a's --/
def transformed_word := "baaa...a"

/-- Hypothetical group relations derived from transformations --/
axiom aba_to_b (a b : G) : (a * b * a = b)
axiom bba_to_a (a b : G) : (b * b * a = a)

/-- Impossible transformation proof --/
theorem impossible_transformation (a b : G) : 
  (initial_word = transformed_word) → False := by
  sorry

end impossible_transformation_l569_569431


namespace Alyssa_spending_correct_l569_569989

def cost_per_game : ℕ := 20

def last_year_in_person_games : ℕ := 13
def this_year_in_person_games : ℕ := 11
def this_year_streaming_subscription : ℕ := 120
def next_year_in_person_games : ℕ := 15
def next_year_streaming_subscription : ℕ := 150
def friends_count : ℕ := 2
def friends_join_games : ℕ := 5

def Alyssa_total_spending : ℕ :=
  (last_year_in_person_games * cost_per_game) +
  (this_year_in_person_games * cost_per_game) + this_year_streaming_subscription +
  (next_year_in_person_games * cost_per_game) + next_year_streaming_subscription -
  (friends_join_games * friends_count * cost_per_game)

theorem Alyssa_spending_correct : Alyssa_total_spending = 850 := by
  sorry

end Alyssa_spending_correct_l569_569989


namespace exists_point_equal_sums_of_distances_l569_569569
open Complex

def center_of_mass (pts : List (ℂ)) : ℂ :=
  (1 / (pts.length : ℝ)) • (pts.foldr ( + ) 0)

theorem exists_point_equal_sums_of_distances 
  (A B : List (ℂ)) 
  (hA_length : ∃ n : ℕ, n = A.length ∧ n = B.length)
  (h_diff_com : center_of_mass A ≠ center_of_mass B) : 
  ∃ P : ℂ, 
  (∑ i in (Finset.range A.length), complex.abs (P - A.nth_le i (by simp [hA_length])) = 
   ∑ i in (Finset.range B.length), complex.abs (P - B.nth_le i (by simp [hA_length])) :=
sorry

end exists_point_equal_sums_of_distances_l569_569569


namespace find_central_angle_l569_569731

noncomputable def sector := 
  {R : ℝ // R > 0}

noncomputable def central_angle (R : ℝ) : ℝ := 
  (6 - 2 * R) / R

theorem find_central_angle :
  ∃ α : ℝ, (α = 1 ∨ α = 4) ∧ 
  (∃ R : ℝ, 
    (2 * R + α * R = 6) ∧ 
    (1 / 2 * R^2 * α = 2)) := 
by {
  sorry
}

end find_central_angle_l569_569731


namespace angle_B_in_parallelogram_l569_569787

theorem angle_B_in_parallelogram (A B C D : Type) [Parallelogram A B C D]
  (hA : angle A = 135) : angle B = 45 :=
by
  -- Proof goes here
  sorry

end angle_B_in_parallelogram_l569_569787


namespace charge_per_patch_eq_1_25_l569_569505

variable (x : ℝ) -- Define the amount Sean is charged per patch

-- Define the conditions
def patches_ordered : ℕ := 100
def sell_price_per_patch : ℝ := 12.00
def net_profit : ℝ := 1075.0

-- Define the total revenue and cost
def total_revenue : ℝ := patches_ordered * sell_price_per_patch
def total_cost : ℝ := patches_ordered * x

-- Define the main theorem to prove
theorem charge_per_patch_eq_1_25 (h : total_revenue - total_cost = net_profit) : x = 1.25 := by
  sorry

end charge_per_patch_eq_1_25_l569_569505


namespace find_angle_APB_l569_569795

-- Definitions based on conditions
def r1 := 2 -- Radius of semicircle SAR
def r2 := 3 -- Radius of semicircle RBT

def angle_AO1S := 70
def angle_BO2T := 40

def angle_AO1R := 180 - angle_AO1S
def angle_BO2R := 180 - angle_BO2T

def angle_PA := 90
def angle_PB := 90

-- Statement of the theorem
theorem find_angle_APB : angle_PA + angle_AO1R + angle_BO2R + angle_PB + 110 = 540 :=
by
  -- Unused in proof: added only to state theorem 
  have _ := angle_PA
  have _ := angle_AO1R
  have _ := angle_BO2R
  have _ := angle_PB
  have _ := 110
  sorry

end find_angle_APB_l569_569795


namespace solve_x_squared_plus_15_eq_y_squared_l569_569581

theorem solve_x_squared_plus_15_eq_y_squared (x y : ℤ) : x^2 + 15 = y^2 → x = 7 ∨ x = -7 ∨ x = 1 ∨ x = -1 := by
  sorry

end solve_x_squared_plus_15_eq_y_squared_l569_569581


namespace division_of_positive_by_negative_l569_569287

theorem division_of_positive_by_negative :
  4 / (-2) = -2 := 
by
  sorry

end division_of_positive_by_negative_l569_569287


namespace exists_quadratic_polynomial_with_2n_roots_l569_569018

noncomputable def iterated_function (f : ℝ → ℝ) : ℕ → ℝ → ℝ
| 0       := id
| (n + 1) := f ∘ iterated_function f n

def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x, a*x^2 + b*x + c

theorem exists_quadratic_polynomial_with_2n_roots (n : ℕ) (n_pos: 0 < n) :
  ∃ (f : ℝ → ℝ), (∃ a b c : ℝ, f = quadratic_polynomial a b c) ∧ 
  ∀ n : ℕ, 0 < n → (∃ roots : list ℝ, (∀ x : ℝ, (quadratic_polynomial 2 0 (-1)) x = 0 ↔ x ∈ roots) ∧ list.length roots = 2^n) :=
sorry

end exists_quadratic_polynomial_with_2n_roots_l569_569018


namespace boys_collected_in_all_l569_569510

-- Definition of the problem’s conditions
variables (solomon juwan levi : ℕ)

-- Given conditions as assumptions
def conditions : Prop :=
  solomon = 66 ∧
  solomon = 3 * juwan ∧
  levi = juwan / 2

-- Total cans collected by all boys
def total_cans (solomon juwan levi : ℕ) : ℕ := solomon + juwan + levi

theorem boys_collected_in_all : ∃ solomon juwan levi : ℕ, 
  conditions solomon juwan levi ∧ total_cans solomon juwan levi = 99 :=
by {
  sorry
}

end boys_collected_in_all_l569_569510


namespace binomial_equality_l569_569490

theorem binomial_equality (k : ℕ) :
  (∑ i in Finset.range (4 * k + 1), binomial (4 * k) i * (-3) ^ i) =
  (∑ j in Finset.range (2 * k + 1), binomial (2 * k) j * (-5) ^ j) :=
by 
  sorry

end binomial_equality_l569_569490


namespace problem_statement_l569_569611

open Real

def ellipse (a b : ℝ) (h : a > 0) (h' : b > 0) := 
  {p : ℝ × ℝ // p.1^2 / a^2 + p.2^2 / b^2 = 1}

variable (a b : ℝ) (h_a_gt_b : a > b) (h_b_gt_0 : b > 0) (h_a_gt_0 : a > 0)

def midpoint (p1 p2 : ℝ × ℝ) := (p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2

noncomputable def eccentricity_of_ellipse (a b : ℝ) : ℝ :=
  sqrt (1 - (b / a)^2)

theorem problem_statement : 
  ∀ (A B : ellipse a b h_a_gt_0 h_b_gt_0),
  let A_point := (A : ℝ × ℝ)
  let B_point := (B : ℝ × ℝ)
  let M : ℝ × ℝ := (1, 1)
  let slope : ℝ := -1/2
  (midpoint A_point B_point = M) →
  (eccentricity_of_ellipse a b) = sqrt(1 - (b / a)^2) :=
sorry

end problem_statement_l569_569611


namespace triangle_PBQ_equilateral_l569_569897

open Real

variables {A B C D P Q : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] [metric_space Q]

def is_rhombus (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
  -- Placeholder for the property that ABCD is a rhombus
  sorry

def is_equilateral (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  -- Placeholder for the property that triangle ABC is equilateral
  sorry

def triangle (A B C : P) : Prop :=
  -- Placeholder for the property that A, B, C form a triangle
  sorry

def angle (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] (θ : ℝ) : Prop :=
  -- Placeholder for the property that angle PQR is θ degrees
  sorry

theorem triangle_PBQ_equilateral
  (h_rhombus : is_rhombus A B C D)
  (h_diagonal_bd : is_equilateral A B D ∧ is_equilateral D B C)
  (h_point_P : ∃ P, triangle A D P)
  (h_point_Q : ∃ Q, triangle C D Q)
  (h_angle_pbq : angle P B Q 60) :
  angle P B Q 60 ∧ angle B P Q 60 ∧ angle B Q P 60 :=
begin
  sorry
end

end triangle_PBQ_equilateral_l569_569897


namespace maria_gum_pieces_l569_569833

theorem maria_gum_pieces (initial_pieces : ℕ) (tommy_pieces : ℕ) (luis_pieces : ℕ) :
  initial_pieces = 25 → tommy_pieces = 16 → luis_pieces = 20 → (initial_pieces + tommy_pieces + luis_pieces = 61) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end maria_gum_pieces_l569_569833


namespace range_of_a_l569_569044

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 2 → x^2 - a ≥ 0) ∧
  (∀ x : ℝ, 2 * x^2 + a * x + 1 > 0) → (¬ (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 2 → x^2 - a ≥ 0)) ∧ 
  (∀ x : ℝ, 2 * x^2 + a * x + 1 > 0) ∨
  (¬ (∀ x : ℝ, 2 * x^2 + a * x + 1 > 0)) ∧ 
  (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 2 → x^2 - a ≥ 0) →
  a ≤ -2 * Real.sqrt 2 ∨ (0 < a ∧ a < 2 * Real.sqrt 2) :=
sorry

end range_of_a_l569_569044


namespace intersection_point_divides_chords_in_ratio_l569_569909

-- First, define the setup of the problem
variables {P₁ P₂ Q₁ Q₂ N : Point}
variable {parabola : Parabola}
variable (H₁ : Tangent parabola P₁)
variable (H₂ : Tangent parabola P₂)
variable (HC₁ : Chord parabola P₁ Q₁)
variable (HC₂ : Chord parabola P₂ Q₂)
variable (Hparallel₁ : Parallel (TangentLine parabola P₁) (ChordLine parabola P₁ Q₁))
variable (Hparallel₂ : Parallel (TangentLine parabola P₂) (ChordLine parabola P₂ Q₂))
variable (Hintersection : IntersectionPoint (ChordLine parabola P₁ Q₁) (ChordLine parabola P₂ Q₂) N)

-- Next, state the proof statement with the conditions and the expected result
theorem intersection_point_divides_chords_in_ratio (parabola : Parabola)
  (P₁ P₂ Q₁ Q₂ N : Point)
  (H₁ : Tangent parabola P₁)
  (H₂ : Tangent parabola P₂)
  (HC₁ : Chord parabola P₁ Q₁)
  (HC₂ : Chord parabola P₂ Q₂)
  (Hparallel₁ : Parallel (TangentLine parabola P₁) (ChordLine parabola P₁ Q₁))
  (Hparallel₂ : Parallel (TangentLine parabola P₂) (ChordLine parabola P₂ Q₂))
  (Hintersection : IntersectionPoint (ChordLine parabola P₁ Q₁) (ChordLine parabola P₂ Q₂) N) :
  ratio (Segment P₁ N) (Segment N Q₁) = 1 / 3 ∧ ratio (Segment P₂ N) (Segment N Q₂) = 1 / 3 :=
sorry

end intersection_point_divides_chords_in_ratio_l569_569909


namespace Shinyoung_ate_most_of_cake_l569_569848

noncomputable def Shinyoung_portion := (1 : ℚ) / 3
noncomputable def Seokgi_portion := (1 : ℚ) / 4
noncomputable def Woong_portion := (1 : ℚ) / 5

theorem Shinyoung_ate_most_of_cake :
  Shinyoung_portion > Seokgi_portion ∧ Shinyoung_portion > Woong_portion := by
  sorry

end Shinyoung_ate_most_of_cake_l569_569848


namespace diameterOuterBoundary_l569_569092

-- Definitions based on the conditions in the problem
def widthWalkingPath : ℝ := 10
def widthGardenRing : ℝ := 12
def diameterPond : ℝ := 16

-- The main theorem that proves the diameter of the circle that forms the outer boundary of the walking path
theorem diameterOuterBoundary : 2 * ((diameterPond / 2) + widthGardenRing + widthWalkingPath) = 60 :=
by
  sorry

end diameterOuterBoundary_l569_569092


namespace question1_question2_l569_569740

-- Define the function f(x) with parameter a
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the interval
def interval : set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Question 1: Prove maximum and minimum at a = -1
theorem question1 :
  let a := -1 in
  let min_val := 1 in
  let max_val := 10 in
  ∀ x ∈ interval, (min_val ≤ f a x ∧ f a x ≤ max_val) := by
  sorry

-- Question 2: Find the range of a for monotonic function
theorem question2 :
  ∀ a : ℝ, (∀ x1 x2 ∈ interval, (x1 ≤ x2 → f a x1 ≤ f a x2) ∨ (x1 ≤ x2 → f a x1 ≥ f a x2)) ↔ (a ≤ -4 ∨ a ≥ 2) := by
  sorry

end question1_question2_l569_569740


namespace toby_change_l569_569559

-- Definitions of costs
def cost_cheeseburger : ℝ := 3.65
def cost_milkshake : ℝ := 2.00
def cost_coke : ℝ := 1.00
def cost_fries : ℝ := 4.00
def cost_cookie : ℝ := 0.50

-- Other given values
def num_cookies : ℝ := 3
def tax_rate : ℝ := 0.07
def tip_rate : ℝ := 0.15
def amount_toby_arrived : ℝ := 15.00

-- Calculation of pre-tax total
def pre_tax_total : ℝ := 
  2 * cost_cheeseburger + 
  cost_milkshake + 
  cost_coke + 
  cost_fries + 
  num_cookies * cost_cookie

-- Calculation of sales tax rounded to nearest cent
def sales_tax : ℝ := (tax_rate * pre_tax_total).round

-- Calculation of tip rounded to nearest cent
def tip : ℝ := (tip_rate * pre_tax_total).round

-- Final calculation of total for both and then for Toby
def final_total : ℝ := pre_tax_total + sales_tax + tip
def amount_each_pays : ℝ := final_total / 2
def change_toby_brings_home : ℝ := amount_toby_arrived - amount_each_pays

-- Lean statement to prove
theorem toby_change : change_toby_brings_home = 5.36 := by
  sorry

end toby_change_l569_569559


namespace angle_bisector_length_formula_l569_569517

-- Definitions from conditions
variables {α : Type*} [LinearOrderedField α]
variables {A B C W : α} -- Points A, B, C, W
variables {a b c : α}   -- Sides opposite to angles A, B, C

-- condition setups
def is_angle_bisector (A B C W : α) : Prop := -- define angle bisector property
sorry

def lies_on_circumcircle (A B C W : α) : Prop := -- define circumcircle property
sorry

-- The actual theorem to prove
theorem angle_bisector_length_formula
  (triangle_ABC : α) (a b c : α)
  (angle_A_bisector : is_angle_bisector A B C W)
  (W_on_circumcircle : lies_on_circumcircle A B C W) :
  AW = (b + c) / (2 * cos (A/2)) :=
sorry

end angle_bisector_length_formula_l569_569517


namespace cos_squared_sub_sin_squared_15_eq_sqrt3_div_2_l569_569548

theorem cos_squared_sub_sin_squared_15_eq_sqrt3_div_2 :
  cos (15 * Real.pi / 180)^2 - sin (15 * Real.pi / 180)^2 = Real.sqrt 3 / 2 := by
sorry

end cos_squared_sub_sin_squared_15_eq_sqrt3_div_2_l569_569548


namespace count_valid_odd_integers_l569_569398

-- Definitions corresponding to the conditions
def is_valid_digit (d : ℕ) : Prop :=
  d ∈ {1, 3, 4, 5, 7, 9}

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_between_300_and_900 (n : ℕ) : Prop :=
  300 ≤ n ∧ n ≤ 900

-- The final correct answer
def correct_count := 51

theorem count_valid_odd_integers :
  let valid_numbers := {n : ℕ | is_between_300_and_900 n ∧ is_odd n ∧ 
                                    let digits := [n / 100, (n / 10) % 10, n % 10] in
                                    all (fun d => is_valid_digit d) digits ∧ 
                                    (digits.nodup)}
  in card valid_numbers = correct_count :=
by sorry

end count_valid_odd_integers_l569_569398


namespace number_of_white_bellied_minnows_l569_569144

-- Definitions for the conditions
def percent_red_bellies : ℝ := 0.40
def percent_green_bellies : ℝ := 0.30
def red_bellied_minnows : ℝ := 20
def total_minnows : ℝ := red_bellied_minnows / percent_red_bellies
def percent_white_bellies : ℝ := 1 - (percent_red_bellies + percent_green_bellies)
def white_bellied_minnows : ℝ := total_minnows * percent_white_bellies

-- Theorem to prove
theorem number_of_white_bellied_minnows : white_bellied_minnows = 15 := 
by 
  unfold white_bellied_minnows 
  unfold total_minnows 
  unfold percent_white_bellies 
  rw [percent_red_bellies, percent_green_bellies, red_bellied_minnows]
  norm_num
  norm_num
  sorry

end number_of_white_bellied_minnows_l569_569144


namespace true_propositions_sequence_l569_569056

-- Define conditions as propositions
def proposition1 : Prop :=
  ∀ r : ℝ, r > 0 → volume (sphere r) = 4/3 * π * r^3 ∧ volume (sphere (r/2)) = (4/3) * π * (r/2)^3 ∧ volume (sphere (r/2)) = (1/8) * volume (sphere r)

def proposition2 : Prop :=
  ∀ (A B : Vector ℝ), average A = average B → stddev A = stddev B

def proposition3 : Prop :=
  ∀ (x y : ℝ), distance (circle_center (0,0)) (line (-1, -1, -1)) = radius (circle (0,0) (1/√2))

-- Define the main theorem to be proven
theorem true_propositions_sequence : proposition1 ∧ ¬proposition2 ∧ proposition3 ↔ true_propositions = (1,3) :=
  sorry

end true_propositions_sequence_l569_569056


namespace circle_center_radius_l569_569519

theorem circle_center_radius :
  ∀ x y : ℝ, (x - 1)^2 + y^2 = 3 → (∃ a b r : ℝ, (a = 1) ∧ (b = 0) ∧ (r = real.sqrt 3) ∧ (x - a)^2 + (y - b)^2 = r^2) :=
by
  intro x y h
  use 1
  use 0
  use real.sqrt 3
  refine ⟨rfl, rfl, rfl, _⟩
  assumption

end circle_center_radius_l569_569519


namespace least_y_l569_569464

-- Defining the statement conditions
def cos_y_eq_cos_y_squared (y : ℝ) : Prop :=
  real.cos (y * real.pi / 180) = real.cos ((y * y) * real.pi / 180)

-- Proving the main theorem
theorem least_y (y : ℝ) (h1 : 2 < y) (h2 : cos_y_eq_cos_y_squared y) : 
  (nat.ceil y = 19) :=
sorry

end least_y_l569_569464


namespace range_of_a_l569_569137

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -x^2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 2 * f x) → a ≥ Real.sqrt 2 :=
by
  -- provided condition
  intros h
  sorry

end range_of_a_l569_569137


namespace valentines_left_l569_569479

def initial_valentines : ℕ := 30
def given_valentines : ℕ := 8

theorem valentines_left (remaining_valentines : ℕ) : 
  remaining_valentines = initial_valentines - given_valentines → 
  remaining_valentines = 22 :=
by
  intro h,
  exact h.symm.trans (by decide),
  sorry  -- proof skipped

end valentines_left_l569_569479


namespace sum_geom_seq_l569_569544

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions based on the conditions
def sum_n_terms (n : ℕ) : ℝ := S n
def sum_2n_terms (n : ℕ) : ℝ := S (2 * n)
def sum_3n_terms (n : ℕ) : ℝ := S (3 * n)

-- Proof statement
theorem sum_geom_seq (n : ℕ) (h1 : sum_n_terms n = 48) (h2 : sum_2n_terms n = 60) :
  sum_3n_terms n = 63 :=
sorry

end sum_geom_seq_l569_569544


namespace eccentricity_of_ellipse_equation_of_ellipse_at_max_area_of_OPQ_l569_569294

variable {a b c : ℝ}
variable {P Q M : ℝ × ℝ}
variable {x_1 y_1 x_2 y_2 : ℝ}
variable {k_1 k_2 : ℝ}
variable {D : ℝ × ℝ} 

noncomputable theory

def ellipse (a b : ℝ) := ∀ (x y : ℝ), (x^2) / (a^2) + (y^2) / (b^2) = 1
def slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)

-- Conditions
axiom h1 : b^2 = (2 / 3) * a^2
axiom h2 : slope P Q = k_1
axiom h3 : slope (0, 0) M = k_2
axiom h4 : k_1 * k_2 = - (2 / 3)
axiom h5 : D = (- √3, 0)
axiom h6 : (2 * (P.1 - D.1), 2 * (P.2 - D.2)) = (Q.1 - D.1, Q.2 - D.2)

-- To Prove
theorem eccentricity_of_ellipse : (a > 0) → (b > 0) → √(a^2 - b^2) / a = √3 / 3 := sorry

theorem equation_of_ellipse_at_max_area_of_OPQ : 
  (a > 0) → (b > 0) → (P ≠ Q) → 
  ∃ a b (c : ℝ), 
  ∀ x y : ℝ, 
    ((x^2 / (15)) + (y^2 / (10)) = 1) := sorry

end eccentricity_of_ellipse_equation_of_ellipse_at_max_area_of_OPQ_l569_569294


namespace sum_of_y_coordinates_is_six_l569_569489

noncomputable def sum_of_y_coordinates : ℝ :=
  let coords := { y : ℝ | (25 + (3 - y)^2 = 64) }
  in ∑ y in coords, y

theorem sum_of_y_coordinates_is_six : sum_of_y_coordinates = 6 := sorry

end sum_of_y_coordinates_is_six_l569_569489


namespace term_formula_a_n_range_lambda_l569_569355

-- Define the sequences and the conditions
def S_n (n : ℕ) : ℚ := sorry

def a_n (n : ℕ) : ℚ := sorry

def b_n (n : ℕ) : ℚ := sorry

-- Conditions of the problem
axiom condition1 : S_n 1 = -9 / 4
axiom condition2 : ∀ n : ℕ, 4 * S_n (n + 1) = 3 * S_n n - 9
axiom condition3 : ∀ n : ℕ, 3 * b_n n + (n - 4) * a_n n = 0

-- The term formula for the sequence {a_n}
theorem term_formula_a_n : ∀ n : ℕ, a_n n = -3 * (3 / 4) ^ n :=
sorry

-- The range of λ
theorem range_lambda : ∀ λ : ℚ, (∀ n : ℕ, T_n n ≤ λ * b_n n) ↔ (-3 ≤ λ ∧ λ ≤ 1) :=
sorry

end term_formula_a_n_range_lambda_l569_569355


namespace train_average_speed_with_stoppages_l569_569580

theorem train_average_speed_with_stoppages (D : ℝ) :
  let speed_without_stoppages := 200
  let stoppage_time_per_hour_in_hours := 12 / 60.0
  let effective_running_time := 1 - stoppage_time_per_hour_in_hours
  let speed_with_stoppages := effective_running_time * speed_without_stoppages
  speed_with_stoppages = 160 := by
  sorry

end train_average_speed_with_stoppages_l569_569580


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l569_569577

variable {a : ℝ}

-- Statement for Option A
theorem option_a_correct : 3 * a - 2 * a = a :=
by sorry

-- Statement for Option B
theorem option_b_incorrect : (a^3)^2 ≠ a^5 :=
by sorry

-- Statement for Option C
theorem option_c_incorrect : 2 * real.sqrt 5 - real.sqrt 5 ≠ 2 :=
by sorry

-- Statement for Option D
theorem option_d_incorrect : (a - 1)^2 ≠ a^2 - 1 :=
by sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l569_569577


namespace fencing_required_l569_569938

/-- Define the conditions for the rectangular field -/
def field_conditions (L W A : ℕ) : Prop :=
  L = 20 ∧ A = 600 ∧ A = L * W

/-- Define the proof problem statement -/
theorem fencing_required (L W A F : ℕ) (h : field_conditions L W A) : F = L + 2 * W → F = 80 :=
by
  cases h with
  | intro hL hA =>
  cases hA with ⟨hA1, hEq⟩ =>
  simp at hL hA1 hEq
  sorry

  /- The above statement sets up the conditions and the theorem to prove that the total fencing required is 80 feet given the conditions. -/

end fencing_required_l569_569938


namespace num_lattice_points_at_distance_4_l569_569440

-- Definition of a lattice point distance condition
def is_lattice_point (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 16

-- Definition of the question translated to proving there are exactly 42 lattice points
theorem num_lattice_points_at_distance_4 : 
  (finset.univ.filter (λ p : ℤ × ℤ × ℤ, is_lattice_point p.1 p.2.1 p.2.2)).card = 42 :=
  sorry

end num_lattice_points_at_distance_4_l569_569440


namespace right_triangles_count_l569_569500

theorem right_triangles_count :
  let x_range := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}.to_list
  let y_range := {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}.to_list in
  (∑ y in y_range,
    ∑' xy_ij in respectively_distinct_pairs x_range,
      let (x_i, x_j) := xy_ij in
      ∑ qy in filter (fun y' => y' ≠ y) y_range, 1) = 450 :=
by {
  sorry,
}

end right_triangles_count_l569_569500


namespace tetrahedron_cross_section_l569_569874

theorem tetrahedron_cross_section (a b : ℝ) (h : 0 < b ∧ b < (a * sqrt 2) / 4) :
  (∃ side1 side2 area : ℝ, side1 = (a + 2 * sqrt 2 * b) / 2 ∧ side2 = (a - 2 * sqrt 2 * b) / 2 ∧ area = (a * a - 8 * b * b) / 4) :=
sorry

end tetrahedron_cross_section_l569_569874


namespace geom_series_sum_correct_l569_569651

-- Define the conditions as constants
def a : ℤ := -2
def r : ℤ := 4
def n : ℤ := 7

-- Define a function to calculate the sum of a geometric series
def geom_series_sum (a r n : ℤ) : ℤ :=
  a * (r^n - 1) / (r - 1)

-- The theorem that states the sum of the series equals the correct answer
theorem geom_series_sum_correct : geom_series_sum a r n = -10922 := by
  sorry

end geom_series_sum_correct_l569_569651


namespace csc_135_proof_l569_569642

section TrigonometricProof

open Real

noncomputable def sin_135 : ℝ := sin (135 * (π / 180))
noncomputable def cos_135 : ℝ := cos (135 * (π / 180))
noncomputable def csc (θ : ℝ) : ℝ := 1 / (sin θ)
  
lemma sin_135_val : sin_135 = 1 / real.sqrt 2 :=
by sorry

lemma cos_135_val : cos_135 = - 1 / real.sqrt 2 :=
by sorry

lemma trigon_identity (θ : ℝ) : (sin θ)^2 + (cos θ)^2 = 1 :=
by sorry

theorem csc_135_proof : csc (135 * (π / 180)) = real.sqrt 2 :=
by
  have h1 : sin_135 = 1 / real.sqrt 2 := sin_135_val
  have h2 : cos_135 = - 1 / real.sqrt 2 := cos_135_val
  have trig_id := trigon_identity (135 * (π / 180))
  unfold csc
  rw [h1]
  sorry  -- complete the remaining steps of the proof

end TrigonometricProof

end csc_135_proof_l569_569642


namespace min_table_sum_l569_569794

theorem min_table_sum (A : Fin 5 → Fin 5 → ℕ)
  (distinct_sums : ∀ (i j : Fin 5), i ≠ j → (∑ k : Fin 5, A i k) ≠ (∑ k : Fin 5, A j k) ∧ (∑ k : Fin 5, A k i) ≠ (∑ k : Fin 5, A k j)) :
  ∑ i j, A i j ≥ 48 :=
sorry

end min_table_sum_l569_569794


namespace max_y_value_l569_569349

noncomputable def f (x : ℝ) : ℝ := 2^(x - 2) + x / 2

noncomputable def f_inv (y : ℝ) : ℝ := (by { sorry : ℝ })

theorem max_y_value : ∀ x ∈ Icc (0 : ℝ) 2, 
  y = f(x) + f_inv(f(x)) → y ≤ 4 :=
by {
  sorry
}

end max_y_value_l569_569349


namespace no_prime_p_for_base_eqn_l569_569516

theorem no_prime_p_for_base_eqn (p : ℕ) (hp: p.Prime) :
  let f (p : ℕ) := 1009 * p^3 + 307 * p^2 + 115 * p + 126 + 7
  let g (p : ℕ) := 143 * p^2 + 274 * p + 361
  f p = g p → false :=
sorry

end no_prime_p_for_base_eqn_l569_569516


namespace abe_age_sum_l569_569546

theorem abe_age_sum (x : ℕ) : 25 + (25 - x) = 29 ↔ x = 21 :=
by sorry

end abe_age_sum_l569_569546


namespace exists_projective_transformation_exceptional_line_perp_l569_569948

variable {P : Type} [ProjectiveSpace P]

def maps_circle_to_circle (ϕ : P → P) (C₁ C₂ : Set P) : Prop :=
  IsProjectiveTransformation ϕ ∧
  IsCircle C₁ ∧ IsCircle C₂ ∧
  (∀ p ∈ C₁, ϕ p ∈ C₂)

theorem exists_projective_transformation (C₁ C₂ : Set P) (p : P) (center_C₂ : P) (ϕ : P → P) :
  (IsCircle C₁) → (IsCircle C₂) → (p ∈ Interior C₁) → (center_C₂ ∈ Center C₂) →
  ∃ ϕ : P → P, IsProjectiveTransformation ϕ ∧
  maps_circle_to_circle ϕ C₁ C₂ ∧ ϕ p = center_C₂ :=
sorry

theorem exceptional_line_perp (C₁ C₂ : Set P) (p q : P) (ϕ : P → P) :
  (IsCircle C₁) → (IsCircle C₂) → (p ∈ Interior C₁) → (center_C₂ ∈ Center C₂) →
  IsProjectiveTransformation ϕ ∧ maps_circle_to_circle ϕ C₁ C₂ ∧ ϕ p = center_C₂ →
  ExceptionalLine ϕ ⊥ Diameter C₁ p q :=
sorry

end exists_projective_transformation_exceptional_line_perp_l569_569948


namespace cost_of_double_room_l569_569638

theorem cost_of_double_room (total_rooms : ℕ) (cost_single_room : ℕ) (total_revenue : ℕ) 
  (double_rooms_booked : ℕ) (single_rooms_booked := total_rooms - double_rooms_booked) 
  (total_single_revenue := single_rooms_booked * cost_single_room) : 
  total_rooms = 260 → cost_single_room = 35 → total_revenue = 14000 → double_rooms_booked = 196 → 
  196 * 60 + 64 * 35 = total_revenue :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_double_room_l569_569638


namespace sophia_lost_pawns_l569_569176

theorem sophia_lost_pawns
    (total_pawns : ℕ := 16)
    (start_pawns_each : ℕ := 8)
    (chloe_lost : ℕ := 1)
    (pawns_left : ℕ := 10)
    (chloe_pawns_left : ℕ := start_pawns_each - chloe_lost) :
    total_pawns = 2 * start_pawns_each → 
    ∃ (sophia_lost : ℕ), sophia_lost = start_pawns_each - (pawns_left - chloe_pawns_left) :=
by 
    intros _ 
    use 5 
    sorry

end sophia_lost_pawns_l569_569176


namespace correct_boxes_indices_l569_569627

theorem correct_boxes_indices (n k : ℕ) (h : n ≥ k) : 
  ∃ (n : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ k → n i = ⌈(i * n) / k⌉) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ k → n i ≥ (i * n) / k) ∧ 
    (∀ i j, 1 ≤ i ≤ j ≤ k → n i < n j) :=
by sorry

end correct_boxes_indices_l569_569627


namespace number_of_parallelograms_l569_569532

-- Given conditions
def num_horizontal_lines : ℕ := 4
def num_vertical_lines : ℕ := 4

-- Mathematical function for combinations
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Proof statement
theorem number_of_parallelograms :
  binom num_horizontal_lines 2 * binom num_vertical_lines 2 = 36 :=
by
  sorry

end number_of_parallelograms_l569_569532


namespace triangle_assignment_l569_569224

def triangle_neighbors (n : ℕ) (k : ℕ) : Prop :=
  ∀ T T₁ T₂ : ℕ, T < n ∧ T₁ ≤ T ∧ T₂ ≤ T → abs(T₁ - T₂) = T

theorem triangle_assignment (n : ℕ) (k : ℕ) :
  ∃ f : ℕ → ℕ, ∀ T T₁ T₂ : ℕ, T < n ∧ T₁ ≤ T ∧ T₂ ≤ T → abs(f T₁ - f T₂) = f T :=
sorry

end triangle_assignment_l569_569224


namespace intersection_of_parallelogram_diagonals_on_diagonal_of_quadrilateral_l569_569671

namespace Mathlib

variables (A B C D P Q X Y Z W : Type) [ConvexQuadrilateral A B C D] 
(ExtensionsIntersect : IntersectOppositeSidesAtPQ A B C D P Q)
(isParallelogram : ∀ (ParalelXY : Parallel XY PQ) (ParalelZW : Parallel ZW PQ), 
  Parallelogram X Y Z W ∧ OnSides X Y Z W A B C D)

theorem intersection_of_parallelogram_diagonals_on_diagonal_of_quadrilateral :
  ∀ (ConvQuadrilateral : ConvexQuadrilateral A B C D) 
    (IntersectPQ : IntersectOppositeSidesAtPQ A B C D P Q) 
    (ParalelXY_PQ : Parallel XY PQ) 
    (ParalelZW_PQ : Parallel ZW PQ)
    (ParallelogramXYZW : Parallelogram X Y Z W) 
    (OnSidesXYZW : OnSides X Y Z W A B C D),
    ∃ (K : Type), IntersectionDiagonals K X Y Z W ∧ OnDiagonal K A C :=
begin
  sorry
end 

end Mathlib

end intersection_of_parallelogram_diagonals_on_diagonal_of_quadrilateral_l569_569671


namespace plane_passes_through_line_l569_569309

-- Definition for a plane α and a line l
variable {α : Set Point} -- α represents the set of points in plane α
variable {l : Set Point} -- l represents the set of points in line l

-- The condition given
def passes_through (α : Set Point) (l : Set Point) : Prop :=
  l ⊆ α

-- The theorem statement
theorem plane_passes_through_line (α : Set Point) (l : Set Point) :
  passes_through α l = (l ⊆ α) :=
by
  sorry

end plane_passes_through_line_l569_569309


namespace at_most_one_parallel_plane_through_line_l569_569654

-- Definitions and conditions
variables (π₁ : Plane) (l : Line)

-- The proof statement
theorem at_most_one_parallel_plane_through_line (h : ∀ P ∈ l, P ∉ π₁) : 
  ∃! π₂ : Plane, π₂ ≠ π₁ ∧ (∀ P ∈ l, P ∈ π₂) ∧ (∀ P ∈ π₁, ∃ Q ∈ π₂, Q = P) :=
sorry

end at_most_one_parallel_plane_through_line_l569_569654


namespace train_crossing_time_is_approx_250_68_sec_l569_569624

def n : ℕ := 50 -- number of carriages
def l : ℕ := 70 -- length of each carriage and engine in meters
def s : ℝ := 80 -- speed of train in kmph
def b : ℕ := 2000 -- length of the bridge in meters

def total_train_length (n l : ℕ) : ℕ := (n + 1) * l
def speed_in_m_per_s (s : ℝ) : ℝ := s * 1000 / 3600
def total_distance (train_length bridge_length : ℕ) : ℕ := train_length + bridge_length
def time_to_cross (total_distance : ℕ) (speed : ℝ) : ℝ := total_distance / speed

theorem train_crossing_time_is_approx_250_68_sec :
  time_to_cross (total_distance (total_train_length n l) b) (speed_in_m_per_s s) ≈ 250.68 :=
by
  sorry

end train_crossing_time_is_approx_250_68_sec_l569_569624


namespace ratio_albert_betty_l569_569273

theorem ratio_albert_betty (A M B : ℕ) (h1 : A = 2 * M) (h2 : M = A - 10) (h3 : B = 5) :
  A / B = 4 :=
by
  -- the proof goes here
  sorry

end ratio_albert_betty_l569_569273


namespace non_intersecting_pairs_l569_569700

-- Definitions
variable {Point : Type} [LinearOrder Point]

def non_collinear (S : Set Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → 
  ¬(∃ (l : Point → Prop), IsLine l ∧ l p1 ∧ l p2 ∧ l p3)

def farms_and_reservoirs (F W : Set Point) (n : ℕ) : Prop :=
  (∃ (P : Set Point), non_collinear P ∧ F ⊆ P ∧ W ⊆ P ∧ #F = n ∧ #W = n ∧ #P = 2 * n)

theorem non_intersecting_pairs (F W : Set Point) (n : ℕ) 
  (hn : farms_and_reservoirs F W n) : 
  ∃ (pairing : Fin n → (Point × Point)), 
    (∀ i j, i ≠ j → ¬ LineSegmentsIntersect (pairing i) (pairing j)) :=
sorry

end non_intersecting_pairs_l569_569700


namespace kelly_held_longest_l569_569817

variable (K : ℕ)

-- Conditions
def Brittany_held (K : ℕ) : ℕ := K - 20
def Buffy_held : ℕ := 120

-- Theorem to prove
theorem kelly_held_longest (h : K > Buffy_held) : K > 120 :=
by sorry

end kelly_held_longest_l569_569817


namespace minimum_time_to_shoe_horses_l569_569240

theorem minimum_time_to_shoe_horses :
  let num_blacksmiths := 48 in
  let num_horses := 60 in
  let minutes_per_horseshoe := 5 in
  let hooves_per_horse := 4 in
  let time_per_blacksmith_per_horse :=
    hooves_per_horse * minutes_per_horseshoe in
  let total_time_all_horses_one_blacksmith :=
    num_horses * time_per_blacksmith_per_horse in
  (total_time_all_horses_one_blacksmith / num_blacksmiths) = 25 :=
by
  let num_blacksmiths := 48
  let num_horses := 60
  let minutes_per_horseshoe := 5
  let hooves_per_horse := 4
  let time_per_blacksmith_per_horse := hooves_per_horse * minutes_per_horseshoe
  let total_time_all_horses_one_blacksmith := num_horses * time_per_blacksmith_per_horse 
  have h1 : total_time_all_horses_one_blacksmith = 1200 := rfl
  have h2 : total_time_all_horses_one_blacksmith / num_blacksmiths = 25 := rfl
  exact h2

end minimum_time_to_shoe_horses_l569_569240


namespace unit_vector_opposite_a_l569_569015

-- Define the given vector a
def a : ℝ × ℝ := (4, 3)

-- Compute the magnitude of vector a
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the unit vector in the opposite direction
def unit_vector_opposite (v : ℝ × ℝ) : ℝ × ℝ := 
  let mag := magnitude v
  in (-(v.1 / mag), -(v.2 / mag))

-- Prove that the unit vector in the opposite direction of a is (-4/5, -3/5)
theorem unit_vector_opposite_a : unit_vector_opposite a = (-4 / 5, -3 / 5) :=
  by
    -- All mathematical calculations and proofs will be written here
    sorry

end unit_vector_opposite_a_l569_569015


namespace finland_forest_area_correct_l569_569601

-- Define the total forested area in the world in hectares
def world_forested_area : ℝ := 8.076e9

-- Define the percentage of the world's forested area represented by Finland
def finland_percentage_of_world : ℝ := 0.0066

-- Define the calculated forested area of Finland in hectares
def finland_forested_area : ℝ := 53_301_600

-- State the theorem
theorem finland_forest_area_correct :
  finland_percentage_of_world * world_forested_area = finland_forested_area :=
by
  sorry

end finland_forest_area_correct_l569_569601


namespace prove_M_squared_l569_569730

noncomputable def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1, 2], ![ (5/2:ℝ), x]]

def eigenvalue_condition (x : ℝ) : Prop :=
  let A := M x
  ∃ v : ℝ, (A - (-2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).det = 0

theorem prove_M_squared (x : ℝ) (h : eigenvalue_condition x) :
  (M x * M x) = ![![ 6, -9], ![ - (45/4:ℝ), 69/4]] :=
sorry

end prove_M_squared_l569_569730


namespace enemy_defeat_points_l569_569589

theorem enemy_defeat_points 
    (points_per_enemy : ℕ) (total_enemies : ℕ) (undefeated_enemies : ℕ) (defeated : ℕ) (points_earned : ℕ) :
    points_per_enemy = 8 →
    total_enemies = 7 →
    undefeated_enemies = 2 →
    defeated = total_enemies - undefeated_enemies →
    points_earned = defeated * points_per_enemy →
    points_earned = 40 :=
by
  intros
  sorry

end enemy_defeat_points_l569_569589


namespace triangle_in_circle_symmetry_axes_l569_569661

-- Define a term for the largest equilateral triangle inscribed in the circle
structure LargestEquilateralTriangleInscribedInCircle where
  isInscribed : Bool

-- Given condition: a structure representing the largest equilateral triangle inscribed in a circle
def triangleInCircle : LargestEquilateralTriangleInscribedInCircle :=
  { isInscribed := true }

-- Theorem: The combined figure has 3 axes of symmetry
theorem triangle_in_circle_symmetry_axes (t : LargestEquilateralTriangleInscribedInCircle) (h : t.isInscribed) :
  (number_of_axes_of_symmetry t) = 3 :=
by 
  -- Since proving the statement is not required according to the instruction
  sorry

end triangle_in_circle_symmetry_axes_l569_569661


namespace number_of_ways_to_choose_officers_l569_569633

-- Define the number of members in the club
def num_members : ℕ := 25

-- Define the number of positions to be filled
def num_positions : ℕ := 3

-- Define the conditions that Alice and Bob will only serve together or not at all
def alice_and_bob_condition : Prop :=
  ∀ (choose : ℕ → ℕ → ℕ), 
  (choose 0 0 + choose 2 1 + choose 3 1 = choose num_positions (23 * 22 * 21 + 3 * 2 * 23))

-- Define the final Lean statement
theorem number_of_ways_to_choose_officers (club_members := num_members) 
(offices := num_positions) 
(h : alice_and_bob_condition) : 
  number_of_ways club_members offices = 10764 := 
sorry

end number_of_ways_to_choose_officers_l569_569633


namespace find_first_term_l569_569280

-- Define the conditions for the geometric series
def common_ratio : ℝ := -1 / 3
def series_sum : ℝ := 27

-- The question is what is the first term
def first_term (a : ℝ) : Prop := series_sum = a / (1 - common_ratio)

-- The theorem to prove the first term of the series
theorem find_first_term : ∃ a : ℝ, first_term a ∧ a = 36 :=
by 
  use 36
  split
  · simp [first_term, series_sum, common_ratio]
    -- Additional calculations to show the first_term correctly
    have h : (1 - (-1 / 3)) = 4 / 3 := by norm_num
    calc
    27 = 36 / (4 / 3) : by norm_num
    ... = 36 * (3 / 4) : by rw div_eq_mul_inv
    ... = 27 : by norm_num
  · refl

end find_first_term_l569_569280


namespace evaluate_expression_l569_569668

theorem evaluate_expression : 2^(Real.log 5 / Real.log 2) + Real.log 25 / Real.log 5 = 7 := by
  sorry

end evaluate_expression_l569_569668


namespace number_of_hens_l569_569637

-- Define the conditions
def hens := Nat
def roosters := 3
def eggs_per_hen_per_day := 1
def days_per_week := 7
def boxes_filled_per_week := 315
def eggs_per_box := 6

-- Define the total number of eggs collected in a week
def total_eggs_collected_per_week : Nat :=
  boxes_filled_per_week * eggs_per_box

-- The main theorem stating the number of hens
theorem number_of_hens :
  ∃ H : hens, total_eggs_collected_per_week / days_per_week = H :=
begin
  use 270,
  sorry -- the proof will be here
end

end number_of_hens_l569_569637


namespace four_digit_cubes_divisible_by_16_l569_569391

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l569_569391


namespace match_points_fold_l569_569613

theorem match_points_fold : 
  (0 : ℝ, 3 : ℝ) = (5 : ℝ, 0 : ℝ) → 
  (8 : ℝ, 4 : ℝ) = (p : ℝ, q : ℝ) →
  p + q = 10 := 
by
  sorry

end match_points_fold_l569_569613


namespace points_on_opposite_sides_of_line_l569_569080

theorem points_on_opposite_sides_of_line (m : ℝ) (h1 : 2 - 1 + m > 0) (h2 : 1 - 3 + m < 0) : -1 < m ∧ m < 2 :=
by
  have h : (m + 1) * (m - 2) < 0 := sorry
  exact sorry

end points_on_opposite_sides_of_line_l569_569080


namespace coefficient_of_x3_in_expansion_l569_569518

theorem coefficient_of_x3_in_expansion :
  let f := λ x : ℚ, (x^2 - x - 2) ^ 5
  in (f x).coeff 3 = 120 :=
by
  sorry

end coefficient_of_x3_in_expansion_l569_569518


namespace simplify_expression_l569_569168

variable (a b : ℚ)

theorem simplify_expression (ha : a = -2) (hb : b = 1/5) :
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  -- Proof can be filled here
  sorry

end simplify_expression_l569_569168


namespace polyhedron_equation_l569_569667

variables (V E F H T : ℕ)

-- Euler's formula for convex polyhedra
axiom euler_formula : V - E + F = 2
-- Number of faces is 50, and each face is either a triangle or a hexagon
axiom faces_count : F = 50
-- At each vertex, 3 triangles and 2 hexagons meet
axiom triangles_meeting : T = 3
axiom hexagons_meeting : H = 2

-- Prove that 100H + 10T + V = 230
theorem polyhedron_equation : 100 * H + 10 * T + V = 230 :=
  sorry

end polyhedron_equation_l569_569667


namespace erdos_problem_l569_569121

variable (X : Type) [Infinite X] (𝓗 : Set (Set X))
variable (h1 : ∀ (A : Set X) (hA : A.Finite), ∃ (H1 H2 : Set X) (hH1 : H1 ∈ 𝓗) (hH2 : H2 ∈ 𝓗), H1 ∩ H2 = ∅ ∧ H1 ∪ H2 = A)

theorem erdos_problem (k : ℕ) (hk : k > 0) : 
  ∃ (A : Set X) (ways : Finset (Set X × Set X)), A.Finite ∧ (∀ (p : Set X × Set X), p ∈ ways → p.1 ∈ 𝓗 ∧ p.2 ∈ 𝓗 ∧ p.1 ∩ p.2 = ∅ ∧ p.1 ∪ p.2 = A) ∧ ways.card ≥ k :=
by
  sorry

end erdos_problem_l569_569121


namespace pyramid_coloring_l569_569649

/--
Given a quadrilateral pyramid \( S A B C D \) and 5 different colors, 
the number of ways to color its vertices such that no two endpoints 
of the same edge share the same color is 420.
-/
theorem pyramid_coloring (colors : Fin 5) : 
  ∃ color_vertices : Fin 5 × Fin 5 × Fin 5 × Fin 5 × Fin 5,
    (∀ (u v : Fin 5), u ≠ v → color_vertices.1 u ≠ color_vertices.1 v) ∧ 
    (Σ u v : Fin 5, color_vertices.1 u ≠ color_vertices.1 v) = 420 := 
sorry

end pyramid_coloring_l569_569649


namespace isosceles_triangle_color_invariance_l569_569782

-- Define the conditions and question in Lean 4 statement
theorem isosceles_triangle_color_invariance
    (n : ℕ) (K : ℕ) (red blue : Fin (6*n + 1) → Bool)
    (hK : K = (Finset.filter red (Finset.univ \ fin 6 * n + 1)).card)
    (hBlue : ∀ x, red x = ¬blue x) :
    ∃ N : ℕ, (∀ red', (Finset.filter red' (Finset.univ \ fin 6 * n + 1)).card = K → 
       (let N_RRR := Finset.card (Finset.filter (is_isosceles_three_vertices_same_color vertices red') (Finset.triangles (Fin.univ \ fin 6*n+1))) in 
        let N_BBB := Finset.card (Finset.filter (is_isosceles_three_vertices_same_color vertices (λ x, ¬red' x)) (Finset.triangles (Fin.univ \ fin 6*n+1))) in 
        N = N_RRR + N_BBB)) :=
sorry

end isosceles_triangle_color_invariance_l569_569782


namespace right_triangle_length_of_AB_l569_569776

theorem right_triangle_length_of_AB (A B C : Type) [metric_space A] [metric_space B]
  [metric_space C] (angle_A : ∠ A B C = 90)
  (tan_B : real.tan (angle (∠ B A C)) = 5 / 12)
  (AC : real.dist A C = 65) : 
  real.dist A B = 60 :=
by {
  -- The proof will go here
  sorry
}

end right_triangle_length_of_AB_l569_569776


namespace sum_of_gcd_5n_plus_6_and_n_l569_569997

open Nat

theorem sum_of_gcd_5n_plus_6_and_n :
  (Finset.univ.filter (λ d, gcd (5 * d + 6) d ∈ ({1, 2, 3, 6} : Finset ℕ))).sum (λ d, gcd (5 * d + 6) d) = 12 :=
begin
  -- This is where the proof would go
  sorry
end

end sum_of_gcd_5n_plus_6_and_n_l569_569997


namespace symmetric_circle_with_respect_to_origin_l569_569079

theorem symmetric_circle_with_respect_to_origin :
  ∀ x y : ℝ, (x + 2) ^ 2 + (y - 1) ^ 2 = 1 → (x - 2) ^ 2 + (y + 1) ^ 2 = 1 :=
by
  intros x y h
  -- Symmetric transformation and verification will be implemented here
  sorry

end symmetric_circle_with_respect_to_origin_l569_569079


namespace sum_of_solutions_l569_569923

-- Define the predicate for the condition
def condition (x : ℤ) : Prop := 4 < (x - 3)^2 ∧ (x - 3)^2 < 49

-- Define the theorem
theorem sum_of_solutions : 
  ∑ x in {x : ℤ | condition x}.toFinset, x = 30 := sorry

end sum_of_solutions_l569_569923


namespace range_of_x_squared_y_squared_l569_569724

theorem range_of_x_squared_y_squared
  (x y : ℝ)
  (h : x^2 + y^2 = 2 * x) :
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27 / 16 :=
begin
  sorry
end

end range_of_x_squared_y_squared_l569_569724


namespace geometric_progression_sum_l569_569333

-- Define the problem-specific conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a_2 a_6 : ℝ → Prop)
hypothesis (h1 : a 2 = 2)
hypothesis (h2 : a 6 = 32)
hypothesis (hgp : ∀ n : ℕ, a (n + 1) = a n * q)

-- S_n represents the sum of the first n terms of geometric progression
noncomputable def S_n (n : ℕ) : ℝ := (a 1) * (1 - q^n) / (1 - q)

-- The target statement
theorem geometric_progression_sum : S 100 = 2^100 - 1 := by
  sorry

end geometric_progression_sum_l569_569333


namespace factor_expression_l569_569310

theorem factor_expression (z : ℤ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := 
by sorry

end factor_expression_l569_569310


namespace rodney_guess_probability_l569_569502

theorem rodney_guess_probability :
  let S := { n : ℕ | 10 ≤ n ∧ n < 100 } in
  let even x := x % 2 = 0 in
  let odd x := x % 2 = 1 in
  let G := { n ∈ S | even (n / 10) ∧ odd (n % 10) ∧ n > 75 } in
  G.finite ∧ G.card = 5 ∧ (1 / G.card : ℚ) = 1 / 5 :=
by
  sorry

end rodney_guess_probability_l569_569502


namespace odd_nat_divisors_odd_prime_power_l569_569007

theorem odd_nat_divisors_odd_prime_power (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 1)
  (h3 : ∀ a b : ℕ, a ∣ n → b ∣ n → Nat.coprime a b → (a + b - 1) ∣ n) : 
  ∃ p m : ℕ, Nat.Prime p ∧ p % 2 = 1 ∧ n = p ^ m :=
by
  sorry

end odd_nat_divisors_odd_prime_power_l569_569007


namespace exists_valid_arrangement_with_middle_sum_19_l569_569504

/-
Given:
1. Six red cards numbered 1 through 6.
2. Six blue cards numbered 3 through 8.
3. Cards are stacked alternately in colors (red, blue, red, blue, ...).
4. Each red card number must divide evenly into each neighboring blue card number except for two specific blue cards which can have a prime divisor outside the given sequence.

Prove:
There exists an arrangement of these cards such that the sum of the numbers on the middle four cards is 19.
-/

def red_cards : list ℕ := [1, 2, 3, 4, 5, 6]
def blue_cards : list ℕ := [3, 4, 5, 6, 7, 8]

def is_valid_arrangement (arrangement : list (ℕ × ℕ)) : Prop :=
  let is_alternating : Prop := ∀ (i : ℕ), i < arrangement.length - 1 → arrangement[i].1 ≠ arrangement[i + 1].1
  let is_valid_divisor (a b : ℕ) : Prop := a ∣ b ∨ ∃ p, p.prime ∧ p ∣ b
  let divisibility_condition : Prop := ∀ (i : ℕ), i < arrangement.length - 1 → 
    (is_valid_divisor (arrangement[i].1) (arrangement[i + 1].2) ∧ is_valid_divisor (arrangement[i + 1].1) (arrangement[i].2))

  is_alternating ∧ divisibility_condition

def middle_four_sum (arrangement : list (ℕ × ℕ)) : ℕ :=
  let n := arrangement.length
  arrangement[n / 2 - 2].2 + arrangement[n / 2 - 1].1 + arrangement[n / 2].2 + arrangement[n / 2 + 1].1

theorem exists_valid_arrangement_with_middle_sum_19 :
  ∃ arrangement : list (ℕ × ℕ), is_valid_arrangement arrangement ∧ middle_four_sum arrangement = 19 := 
sorry

end exists_valid_arrangement_with_middle_sum_19_l569_569504


namespace square_arrangement_l569_569485

theorem square_arrangement (chips : ℕ) (adjacent : ℕ → ℕ → Prop) (initial_square : chips = 100) (initial_adjacency: ∀ i j, adjacent i j → (i - j = 1 ∨ i - j = -1 ∨ i - j = 10 ∨ i - j = -10)) (no_same_cell : ∀ i j, i ≠ j → ¬ (initially_occupied i = initially_occupied j)) : 
              (exists (a b : ℕ), a * b = 100 ∧ ∀ i j, adjacent i j → (i - j = 1 ∨ i - j = -1 ∨ i - j = a ∨ i - j = -a)) → 
              (exists n, n^2 = 100) :=
by
  sorry

end square_arrangement_l569_569485


namespace quadratic_has_one_real_root_l569_569082

theorem quadratic_has_one_real_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 4 * m = 0) : m = 4 / 9 :=
by sorry

end quadratic_has_one_real_root_l569_569082


namespace cleaning_event_children_l569_569870

theorem cleaning_event_children (total_members : ℕ) (pct_men : ℕ) (ratio_women_men : ℕ)
  (h_total : total_members = 2000) (h_pct_men : pct_men = 30)
  (h_ratio : ratio_women_men = 2) : 
  let num_men := total_members * pct_men / 100
  let num_women := ratio_women_men * num_men
  let num_adults := num_men + num_women
  let num_children := total_members - num_adults
  in num_children = 200 :=
by {
  sorry
}

end cleaning_event_children_l569_569870


namespace average_shifted_data_is_7_l569_569770

variable (x1 x2 x3 : ℝ)

theorem average_shifted_data_is_7 (h : (x1 + x2 + x3) / 3 = 5) : 
  ((x1 + 2) + (x2 + 2) + (x3 + 2)) / 3 = 7 :=
by
  sorry

end average_shifted_data_is_7_l569_569770


namespace max_ratio_triangle_areas_l569_569037

theorem max_ratio_triangle_areas 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (E : set (ℝ × ℝ)) (hE : ∀ x y, (x, y) ∈ E ↔ x^2 / a^2 + y^2 / b^2 = 1)
  (F1 F2 P D M N : ℝ × ℝ) 
  (hF1F2 : dist F1 (0, 0) = dist F2 (0, 0) ∧ dist F1 F2 = 2 * real.sqrt (a^2 - b^2))
  (hP : (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ (P ≠ (a, 0)) ∧ (P ≠ (-a, 0)))
  (hD : D.2 = 0 ∧ ∃ θ, angle_subtended_by_angle_bisector θ F1 P F2 D)
  (hM : ∃ θ, perp_from_point D P F1 = M)
  (hN : ∃ θ, perp_from_point D P F2 = N) :
  (∃ k, k = (area_of_triangle D M N) / (area_of_triangle F1 P F2)) ∧ k ≤ (b^2 * (a^2 - b^2)) / a^4 := 
sorry

end max_ratio_triangle_areas_l569_569037


namespace rowena_finish_time_l569_569503

def rowena_rate : ℝ := 2 / 5
def rowena_time_for_2_5 : ℝ := 18
def rowena_start_time : ℝ := 10.0 -- 10:00 a.m. as float value for simplicity

theorem rowena_finish_time :
  rowena_start_time + (rowena_time_for_2_5 / rowena_rate) = 10.75 :=
by
  sorry

end rowena_finish_time_l569_569503


namespace rhombus_area_l569_569846

theorem rhombus_area (a b : ℝ) (h1 : a = 10) (h2 : b = 9) : 
  let d1 := 18 in
  let d2 := 2 * √19 in
  0.5 * d1 * d2 = 18 * √19 :=
by
  -- a and b are given values contributing to the rhombus's structure
  let side := a -- side length of the rhombus
  let half_diagonal1 := b -- half of the diagonal EG
  have side_calc : side = 10 := h1
  have half_diagonal1_calc : half_diagonal1 = 9 := h2
  let d1 := 18 -- diagonal EG
  let d2 := 2 * √19 -- calculated diagonal FH
  calc
    0.5 * d1 * d2 = 0.5 * 18 * (2 * √19) : by rfl
    ... = 18 * √19 : by norm_num

end rhombus_area_l569_569846


namespace yoojeong_straight_line_length_l569_569902

theorem yoojeong_straight_line_length :
  let cm_to_mm := 31 * 10 in            -- converting cm to mm
  let line_length := cm_to_mm - 3 in    -- subtract 3 mm
  line_length = 307 := 
by
  let cm_to_mm := 31 * 10
  let line_length := cm_to_mm - 3
  sorry

end yoojeong_straight_line_length_l569_569902


namespace slope_of_line_l569_569539

theorem slope_of_line (x y : ℝ) :
  x - sqrt 3 * y + 3 = 0 → y = (sqrt 3 / 3) * x + sqrt 3 :=
by
  sorry

end slope_of_line_l569_569539


namespace solve_equation_unique_solution_l569_569171

noncomputable def solve_equation (x : ℝ) : Prop :=
  sqrt (1 + sqrt (4 + sqrt (2 * x + 3))) = (1 + sqrt (2 * x + 3)) ^ (1 / 4)

theorem solve_equation_unique_solution :
  ∃ x : ℝ, solve_equation x ∧ x = -23 / 32 :=
begin
  use -23 / 32,
  sorry -- Proof goes here
end

end solve_equation_unique_solution_l569_569171


namespace condition_one_valid_l569_569425

variable (E F G H : Type) -- Assume points E, F, G, H
variable [Collinear E F G H] -- Assume collinearity

noncomputable def length_segment (A B : Type) : Real := sorry -- Placeholder for segment length

variable (p q r : Real)
variable (length_segment E F = p)
variable (length_segment E G = q)
variable (length_segment E H = r)
variable (r > 2 * p)

theorem condition_one_valid : p < r / 3 := sorry

end condition_one_valid_l569_569425


namespace cranberry_count_correct_l569_569791

noncomputable def cranberry_final_count : ℕ :=
  let initial := 450000
  let after_farmers := initial - (initial * 63 / 100)
  let after_deer := after_farmers - (after_farmers * 23 / 100)
  let after_elk := after_deer - 20000
  let after_rabbits := after_elk - (after_elk * 13 / 100)
  let after_birds := after_rabbits - (after_rabbits * 15 / 100)
  let after_storm := after_birds - (after_birds * 7 / 100)
  after_storm.to_nat

theorem cranberry_count_correct :
  cranberry_final_count = 74403 :=
by
  sorry

end cranberry_count_correct_l569_569791


namespace tan_2alpha_value_cos_beta_value_l569_569701

variables (α β : ℝ)
variables (h1 : cos α = 1 / 7)
variables (h2 : cos (α - β) = 13 / 14)
variables (h3 : 0 < β)
variables (h4 : β < α)
variables (h5 : α < π / 2)

theorem tan_2alpha_value : tan (2 * α) = -((8 * real.sqrt 3) / 47) :=
by
  sorry

theorem cos_beta_value : cos β = 1 / 2 :=
by
  sorry

end tan_2alpha_value_cos_beta_value_l569_569701


namespace opposite_of_2023_l569_569890

theorem opposite_of_2023 : ∃ y : ℤ, 2023 + y = 0 ∧ y = -2023 :=
by
  use -2023
  split
  · linarith
  · refl

end opposite_of_2023_l569_569890


namespace diameter_of_circumscribed_sphere_l569_569051

-- Define the lengths of the sides of the triangle
def side_a : ℝ := 10
def side_b : ℝ := 12
def side_c : ℝ := 14

-- Calculate the medians of the triangle
def median_a := Real.sqrt ((2 * side_b^2 + 2 * side_c^2 - side_a^2) / 4)
def median_b := Real.sqrt ((2 * side_c^2 + 2 * side_a^2 - side_b^2) / 4)
def median_c := Real.sqrt ((2 * side_a^2 + 2 * side_b^2 - side_c^2) / 4)

-- Define the circumradius formula for the tetrahedron formed by medians
def circumradius := Real.sqrt ((median_a^2 + median_b^2 + median_c^2) / 2) / 2

-- Prove that the diameter of the circumscribed sphere of the tetrahedron is sqrt(55)
theorem diameter_of_circumscribed_sphere : 2 * circumradius = Real.sqrt 55 := 
by
  -- Setup the calculated diameters
  let med_a := Real.sqrt ((2 * side_b^2 + 2 * side_c^2 - side_a^2) / 4)
  let med_b := Real.sqrt ((2 * side_c^2 + 2 * side_a^2 - side_b^2) / 4)
  let med_c := Real.sqrt ((2 * side_a^2 + 2 * side_b^2 - side_c^2) / 4)
  let calc_radius := Real.sqrt ((med_a^2 + med_b^2 + med_c^2) / 2) / 2
  have h: 2 * calc_radius = Real.sqrt 55 := sorry
  exact h

end diameter_of_circumscribed_sphere_l569_569051


namespace first_grade_sample_count_l569_569604

noncomputable def total_students := 3200
noncomputable def second_grade_students := 1000
noncomputable def third_grade_students := 1000
noncomputable def sample_size := 160
noncomputable def proportion_sampled := sample_size / total_students.to_rat
noncomputable def first_grade_students := total_students - second_grade_students - third_grade_students

theorem first_grade_sample_count :
  first_grade_students * proportion_sampled = 60 := 
by
  sorry

end first_grade_sample_count_l569_569604


namespace solve_equation_l569_569172

theorem solve_equation (x : ℝ) (h : (x - 7) / 2 - (1 + x) / 3 = 1) : x = 29 :=
sorry

end solve_equation_l569_569172


namespace vector_midpoint_problem_l569_569441

noncomputable def midpoint_vector (A D : V) : V := (A + D) / 2

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_midpoint_problem
  (A B C D E : V)
  (hD : ∃ λ : ℝ, D = λ • B + (1 - λ) • C)
  (hE : E = midpoint_vector A D)
  (hAE : E = (2 / 5) • B + m • C) : m = (1 / 10) :=
begin
  sorry
end

end vector_midpoint_problem_l569_569441


namespace four_digit_perfect_cubes_divisible_by_16_l569_569381

theorem four_digit_perfect_cubes_divisible_by_16 : (∃ k : ℕ, k = 3) :=
by
  let possible_cubes := [12 ^ 3, 16 ^ 3, 20 ^ 3]
  have h1 : 12 ^ 3 = 1728 := by norm_num
  have h2 : 16 ^ 3 = 4096 := by norm_num
  have h3 : 20 ^ 3 = 8000 := by norm_num

  have h4 : (1728, 4096, 8000).all (λ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 16 = 0)
    := by norm_num

  use 3
  trivial

end four_digit_perfect_cubes_divisible_by_16_l569_569381


namespace parallel_perpendicular_implies_perpendicular_l569_569469

-- Definitions of the geometric relationships
variables {Line Plane : Type}
variables (a b : Line) (alpha beta : Plane)

-- Conditions as per the problem statement
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Lean statement of the proof problem
theorem parallel_perpendicular_implies_perpendicular
  (h1 : parallel_line_plane a alpha)
  (h2 : perpendicular_line_plane b alpha) :
  perpendicular_lines a b :=  
sorry

end parallel_perpendicular_implies_perpendicular_l569_569469


namespace fraction_saved_l569_569272

-- Define conditions
variables (P : ℝ) (S : ℝ) -- Define average monthly take-home pay and fraction saved

-- State the main theorem
theorem fraction_saved (h : 12 * S * P = 2 * (1 - S) * P) : S = 1 / 7 :=
by 
  -- Given the equation 12 * S * P = 2 * (1 - S) * P
  have hp : P ≠ 0, from sorry, -- Assume P is not zero
  -- Simplify the equation by canceling P
  have heq : 12 * S = 2 * (1 - S),
  from (by { rw [mul_assoc, mul_comm P, ←mul_assoc (2 * (1 - S)), mul_comm P, mul_comm P, mul_assoc], exact h }),
  -- Now solve for S
  linarith

end fraction_saved_l569_569272


namespace probability_hundreds_digit_triple_ones_digit_l569_569419

def is_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem probability_hundreds_digit_triple_ones_digit :
  let favorable_outcomes : ℤ :=
    {n : ℕ | is_3_digit_number n ∧ hundreds_digit n = 3 * ones_digit n} .to_finset.card in
  let total_outcomes : ℤ :=
    {n : ℕ | is_3_digit_number n} .to_finset.card in
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 30 :=
by {
  sorry
}

end probability_hundreds_digit_triple_ones_digit_l569_569419


namespace eccentricity_of_ellipse_minimum_length_of_AB_l569_569359

open Real

noncomputable def ellipse_equation (x y : ℝ) : Prop := 
  x^2 + 2 * y^2 = 4

def eccentricity (a b : ℝ) : ℝ := 
  sqrt (a^2 - b^2) / a

def length_AB (x0 y0 t : ℝ) : ℝ := 
  sqrt ((x0 + 2 * y0 / x0)^2 + (y0 - 2)^2)

theorem eccentricity_of_ellipse :
  let a := 2
  let b := sqrt 2
  let e := eccentricity a b
  e = sqrt 2 / 2 :=
by 
  sorry

theorem minimum_length_of_AB (x0 y0 : ℝ) (H : ellipse_equation x0 y0) :
  0 < x0^2 ∧ x0^2 <= 4 → 
  let t := - 2 * y0 / x0
  OA_perp_OB : x0 * t + 2 * y0 = 0 →
  (2 * sqrt 2) :=
by 
  sorry

end eccentricity_of_ellipse_minimum_length_of_AB_l569_569359


namespace circle_condition_l569_569193

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (m < 1 / 2)) :=
by {
-- Skipping the proof here
sorry
}

end circle_condition_l569_569193


namespace star_area_ratio_l569_569958

noncomputable def circle_radius : ℝ := 3
noncomputable def area_circle (r : ℝ) : ℝ := π * r^2
noncomputable def area_hexagon (s : ℝ) : ℝ := (3 * real.sqrt 3 / 2) * s^2
noncomputable def approx_area_star (a_hex : ℝ) : ℝ := (3/4) * a_hex
noncomputable def ratio_star_circle (a_star a_circle : ℝ) : ℝ := a_star / a_circle

theorem star_area_ratio : 
  let r := circle_radius in
  let s := 2 * r in
  let a_circle := area_circle r in
  let a_hexagon := area_hexagon s in
  let a_star := approx_area_star a_hexagon in
  ratio_star_circle a_star a_circle = 4.5 * real.sqrt 3 / π :=
by
  -- Proof omitted
  sorry

end star_area_ratio_l569_569958


namespace exponent_multiplication_l569_569288

theorem exponent_multiplication :
  (-27: ℤ) ^ (2 / 3 : ℚ) * (9: ℤ) ^ (-3 / 2 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end exponent_multiplication_l569_569288


namespace ratio_areas_is_one_l569_569264

-- Conditions
variable (r1 r2 : ℝ)

noncomputable def inradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 6

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  π * r^2

-- Let the side length of the hexagon be 2
def side_length : ℝ := 2

-- First circle's radius
def radius_first_circle : ℝ := inradius_of_equilateral_triangle (side_length)

-- Second circle's radius is the same due to symmetry and regularity of the hexagon
def radius_second_circle : ℝ := radius_first_circle

-- Areas of the circles
def area_first_circle : ℝ := area_of_circle radius_first_circle
def area_second_circle : ℝ := area_of_circle radius_second_circle

-- Ratio of the areas of the circles
def ratio_of_areas : ℝ := area_second_circle / area_first_circle

-- Proof that the ratio is 1
theorem ratio_areas_is_one : ratio_of_areas = 1 := by
  sorry

end ratio_areas_is_one_l569_569264


namespace min_n_for_constant_term_in_expansion_l569_569773

theorem min_n_for_constant_term_in_expansion :
  ∃ (n : ℕ), (∀ (r : ℕ), n = (5 / 4 : ℚ) * r → 6 * n - (15 / 2 : ℚ) * r = 0) ∧ n = 5 :=
begin
  sorry
end

end min_n_for_constant_term_in_expansion_l569_569773


namespace sum_S_range_l569_569472

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then -x^2 + 2 * x else 3 * f (x + 2)

def seq_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else f (2 * (n-1) + 1)

def sum_S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, seq_a (i + 1)

theorem sum_S_range (n : ℕ) : 
  1 ≤ sum_S n ∧ sum_S n < 3 / 2 :=
sorry

end sum_S_range_l569_569472


namespace find_m_and_other_root_l569_569350

noncomputable def x1 (a : ℝ) : ℝ := real.sqrt (a + 2) - real.sqrt (8 - a) + real.sqrt (-a^2)

theorem find_m_and_other_root (a : ℝ) (h : x1 a = -real.sqrt 2) :
  ∃ m x2, (m = 1 ∨ m = -2) ∧
  (
    (m = 1 ∧ x1 a = -real.sqrt 2 ∧ x2 = -real.sqrt 2) ∨
    (m = -2 ∧ x1 a = -real.sqrt 2 ∧ x2 = 2 * real.sqrt 2)
  ) :=
sorry

end find_m_and_other_root_l569_569350


namespace minimum_value_2sqrt5_l569_569679

theorem minimum_value_2sqrt5 : ∀ x : ℝ, 
  ∃ m : ℝ, (∀ x : ℝ, m ≤ (x^2 + 10) / (Real.sqrt (x^2 + 5))) ∧ (m = 2 * Real.sqrt 5) := by
  sorry

end minimum_value_2sqrt5_l569_569679


namespace average_percentage_of_first_20_percent_l569_569767

theorem average_percentage_of_first_20_percent (X : ℝ) 
  (h1 : 0.20 * X + 0.50 * 60 + 0.30 * 40 = 58) : 
  X = 80 :=
sorry

end average_percentage_of_first_20_percent_l569_569767


namespace evaluate_expression_l569_569001

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (1/3 : ℚ)
  let z := (-12 : ℚ)
  let w := (5 : ℚ)
  x^2 * y^3 * z + w = (179/36 : ℚ) :=
by
  sorry

end evaluate_expression_l569_569001


namespace even_factors_count_of_n_l569_569299

theorem even_factors_count_of_n (n : ℕ) (h : n = 2^3 * 3^2 * 7^1 * 11^1) :
  ∃ count : ℕ, count = 36 ∧ (∀ a b c d : ℕ, 0 ≤ a ∧ a ≤ 3 →
    0 ≤ b ∧ b ≤ 2 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 →
    2^a * 3^b * 7^c * 11^d ∣ n → (a ≥ 1 → count)) := sorry

end even_factors_count_of_n_l569_569299


namespace sum_of_B_and_C_in_base_6_l569_569762

def digit_base_6 (n: Nat) : Prop :=
  n > 0 ∧ n < 6

theorem sum_of_B_and_C_in_base_6
  (A B C : Nat)
  (hA : digit_base_6 A)
  (hB : digit_base_6 B)
  (hC : digit_base_6 C)
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : 43 * (A + B + C) = 216 * A) :
  B + C = 5 := by
  sorry

end sum_of_B_and_C_in_base_6_l569_569762


namespace simplify_sqrt_expression_l569_569849

theorem simplify_sqrt_expression :
  (3 * (Real.sqrt (4 * 3)) - 2 * (Real.sqrt (1 / 3)) +
     Real.sqrt (16 * 3)) / (2 * Real.sqrt 3) = 14 / 3 := by
sorry

end simplify_sqrt_expression_l569_569849


namespace minimal_max_value_is_four_l569_569695

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def lattice_assignment_valid (f : ℤ × ℤ → ℕ) : Prop :=
  ∀ (x1 x2 y1 y2 : ℤ), x1 ≤ x2 → y1 ≤ y2 → ¬ is_prime (finset.sum (finset.Icc (x1, y1) (x2, y2)) f)

noncomputable def minimal_max_lattice_value := 
  Inf { m : ℕ | ∃ (f : ℤ × ℤ → ℕ), (∀ p, 0 < f p) ∧ (∀ p, f p ≤ m) ∧ lattice_assignment_valid f }

theorem minimal_max_value_is_four : minimal_max_lattice_value = 4 :=
  sorry

end minimal_max_value_is_four_l569_569695


namespace abs_complex_div_l569_569286

theorem abs_complex_div (z w : ℂ) (hz : z = 2) (hw : w = 1 + I) :
  complex.abs (z / w) = real.sqrt 2 :=
by
  sorry

end abs_complex_div_l569_569286


namespace dasha_flags_proof_l569_569016

variable (Tata_flags_right Yasha_flags_right Vera_flags_right Maxim_flags_right : ℕ)
variable (Total_flags : ℕ)

theorem dasha_flags_proof 
  (hTata: Tata_flags_right = 14)
  (hYasha: Yasha_flags_right = 32)
  (hVera: Vera_flags_right = 20)
  (hMaxim: Maxim_flags_right = 8)
  (hTotal: Total_flags = 37) :
  ∃ (Dasha_flags : ℕ), Dasha_flags = 8 :=
by
  sorry

end dasha_flags_proof_l569_569016


namespace logarithmic_inequality_solution_l569_569204

theorem logarithmic_inequality_solution :
  {x : ℝ | log 2 (x + 6) < log 2 (2 - x)} = set.Ioo (-6 : ℝ) (-2 : ℝ) :=
by
  sorry

end logarithmic_inequality_solution_l569_569204


namespace product_of_positive_real_part_solutions_l569_569410

theorem product_of_positive_real_part_solutions :
  (∃ roots : list (ℂ), roots = [2 * complex.exp(Complex.I * real.pi * ((2 * n + 1) / 8)) | n <- [0,1,7]] ∧
  (∀ z : ℂ, z ∈ roots → z.real_part > 0) ∧
  (∃ product_of_roots, product_of_roots = 8 * complex.exp(Complex.I * real.pi * 67.5 / 180))) := 
sorry

end product_of_positive_real_part_solutions_l569_569410


namespace piecewise_function_continuity_l569_569451

theorem piecewise_function_continuity :
  (∃ a c : ℝ, (2 * a * 2 + 4 = 2^2 - 2) ∧ (4 - 2 = 3 * (-2) - c) ∧ a + c = -17 / 2) :=
by
  sorry

end piecewise_function_continuity_l569_569451


namespace planes_parallel_if_perpendicular_to_same_line_l569_569377

open Plane Affine

variables {α β γ : Plane} {m n : Line}

theorem planes_parallel_if_perpendicular_to_same_line (m : Line) (α β : Plane) :
  (m ⟂ α) ∧ (m ⟂ β) → (α ∥ β) :=
by
  sorry

end planes_parallel_if_perpendicular_to_same_line_l569_569377


namespace prove_curveB_eccentricity_l569_569631

def curveA_eq : Prop := (λ x y : ℝ, (x^2 / 2) - (y^2 / 4) = 1)
def curveB_eq : Prop := (λ x y : ℝ, (x^2 / 4) - (y^2 / 2) = 1)
def curveC_eq : Prop := (λ x y : ℝ, (x^2 / 4) - (y^2 / 6) = 1)
def curveD_eq : Prop := (λ x y : ℝ, (x^2 / 4) - (y^2 / 10) = 1)

def eccentricity (a c : ℝ) : ℝ := c / a

theorem prove_curveB_eccentricity :
  ∃ a c, curveB_eq x y ∧ ecc = (sqrt 6 / 2) :=
by 
  let a := 2
  let c := sqrt 6
  let ecc := eccentricity a c
  show ecc = sqrt 6 / 2
  sorry

end prove_curveB_eccentricity_l569_569631


namespace perp_bisector_of_AB_l569_569880

noncomputable def perpendicular_bisector_eq : Prop :=
  ∀ (x y : ℝ), (x - y + 1 = 0) ∧ (x^2 + y^2 = 1) → (x + y = 0)

-- The proof is omitted
theorem perp_bisector_of_AB : perpendicular_bisector_eq :=
sorry

end perp_bisector_of_AB_l569_569880


namespace equilateral_triangle_BJ_l569_569096

-- Define the problem and its conditions
theorem equilateral_triangle_BJ:
  ∀ (ABC : Type) (A B C D E F G H J : ABC) (side_length AG GF FH HC : ℤ),
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
  -- Points D, E, F are on sides BC, CA, AB respectively
  D ∈ line_segment B C ∧
  E ∈ line_segment C A ∧
  F ∈ line_segment A B ∧
  -- Given lengths
  AG = 3 ∧
  GF = 14 ∧
  FH = 9 ∧
  HC = 2 ∧
  -- Total length condition
  AG + GF + FH + HC = side_length ∧
  side_length = 28 ∧
  -- Midpoint J of HF
  J = midpoint H F 
  → 
  -- Conclusion
  BJ = 13.5 :=
sorry -- Proof to be provided

end equilateral_triangle_BJ_l569_569096


namespace fuel_tank_capacity_l569_569634

theorem fuel_tank_capacity
  (ethanol_A_ethanol : ∀ {x : Float}, x = 0.12 * 49.99999999999999)
  (ethanol_B_ethanol : ∀ {C : Float}, x = 0.16 * (C - 49.99999999999999))
  (total_ethanol : ∀ {C : Float}, 0.12 * 49.99999999999999 + 0.16 * (C - 49.99999999999999) = 30) :
  (C = 162.5) :=
sorry

end fuel_tank_capacity_l569_569634


namespace number_of_questionnaires_drawn_from_15_to_16_is_120_l569_569979

variable (x : ℕ)
variable (H1 : 120 + 180 + 240 + x = 900)
variable (H2 : 60 = (bit0 90) / 180)
variable (H3 : (bit0 (bit0 (bit0 15))) = (bit0 (bit0 (bit0 15))) * (900 / 300))

theorem number_of_questionnaires_drawn_from_15_to_16_is_120 :
  ((900 - 120 - 180 - 240) * (300 / 900)) = 120 :=
sorry

end number_of_questionnaires_drawn_from_15_to_16_is_120_l569_569979


namespace probability_of_exactly_one_pair_l569_569356

theorem probability_of_exactly_one_pair (total_gloves : ℕ) (pick_gloves : ℕ) (pairs : ℕ) :
  total_gloves = 10 ∧ pick_gloves = 4 ∧ pairs = 5 →
  ∃ (p : ℚ), p = 1/7 :=
by
  intros h,
  sorry

end probability_of_exactly_one_pair_l569_569356


namespace smallest_digit_not_in_units_place_of_odd_l569_569922

theorem smallest_digit_not_in_units_place_of_odd :
  (∀ (n : ℕ), n % 2 = 1 → (n % 10 = 1 ∨ n % 10 = 3 ∨ n % 10 = 5 ∨ n % 10 = 7 ∨ n % 10 = 9)) →
  (∀ (d : ℕ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.erase 0 → ∃ (n : ℕ), n % 2 = 1 ∧ n % 10 = d) :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l569_569922


namespace train_speed_l569_569623

theorem train_speed (train_length bridge_and_train_length : ℕ) 
(seconds : ℕ)
(h_train_length : train_length = 130)
(h_bridge_and_train_length : bridge_and_train_length = 245)
(h_seconds : seconds = 30) :
  let speed := (train_length + bridge_and_train_length) / seconds * 3.6 in speed = 45 :=
by {
  sorry
}

end train_speed_l569_569623


namespace find_y_l569_569763

-- Define the logarithmic values as constants
def log2 : ℝ := 0.3010
def log3 : ℝ := 0.4771

-- Define the main theorem
theorem find_y (y : ℝ) (log2 log3 : ℝ) (h_log2 : log2 = 0.3010) (h_log3 : log3 = 0.4771) (h_exp : 2^(y + 4) = 288) : y = (log2 + 2 * log3) / log2 - 4 :=
by
  -- State that we need to prove this theorem
  sorry

end find_y_l569_569763


namespace savings_account_deposit_l569_569226

theorem savings_account_deposit 
  (earnings : ℝ)
  (H1 : 0.07 * earnings = 133)  -- condition 1
  (H2_groceries : 3 / 10 * earnings)  -- condition 2
  (H3_entertainment : 1 / 5 * earnings)  -- condition 3
  (H4_transportation : 0.12 * earnings)  -- condition 4
  (total_expenses : ℝ := 133 + (3 / 10 * earnings) + (1 / 5 * earnings) + (0.12 * earnings))  -- total expenses from all other conditions
  (savings : ℝ := earnings - total_expenses)  -- defining savings
  : savings = 589 :=
by 
  sorry

end savings_account_deposit_l569_569226


namespace least_possible_value_m_n_l569_569463

theorem least_possible_value_m_n :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd (m + n) 330 = 1 ∧ n ∣ m^m ∧ ¬(m % n = 0) ∧ (m + n = 377) :=
by
  sorry

end least_possible_value_m_n_l569_569463


namespace triangle_ABC_area_l569_569801

-- We define the basic structure of a triangle and its properties
structure Triangle :=
(base : ℝ)
(height : ℝ)
(right_angled_at : ℝ)

-- Define the specific triangle ABC with given properties
def triangle_ABC : Triangle := {
  base := 12,
  height := 15,
  right_angled_at := 90 -- since right-angled at C
}

-- Given conditions, we need to prove the area is 90 square cm
theorem triangle_ABC_area : 1/2 * triangle_ABC.base * triangle_ABC.height = 90 := 
by 
  sorry

end triangle_ABC_area_l569_569801


namespace yellow_picks_l569_569448

theorem yellow_picks (T : ℕ) (hT1 : (1/2 : ℚ) * T = 18) (hT2 : (1/3 : ℚ) * T = 12) : T - ((1/2 : ℚ) * T).natAbs - 12 = 6 := by
  sorry

end yellow_picks_l569_569448


namespace problem1_problem2_l569_569742

-- Define the function f(x)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Condition that m must be greater than 0
variable {m : ℝ} (hm : m > 0)

-- First problem statement: When m=1, the solution set for f(x) ≥ 1 is x ≤ -3/2.
theorem problem1 (x : ℝ) (h : f x 1 ≥ 1) : x ≤ -3 / 2 :=
sorry

-- Second problem statement: The range of values for m such that f(x) < |2 + t| + |t - 1| holds for all x and t is 0 < m < 3/4.
theorem problem2 (m : ℝ) : (∀ (x t : ℝ), f x m < abs (2 + t) + abs (t - 1)) ↔ (0 < m ∧ m < 3 / 4) :=
sorry

end problem1_problem2_l569_569742


namespace problem1_problem2_l569_569591

theorem problem1 : abs (-3) - real.sqrt 9 + (5 : ℝ)⁻¹ = 1/5 := 
by sorry

theorem problem2 (x y : ℝ) : (x - 2 * y)^2 - x * (x - 4 * y) = 4 * y^2 := 
by sorry

end problem1_problem2_l569_569591


namespace at_most_one_shared_suit_l569_569145

-- Definitions used in conditions
def total_cards : ℕ := 52
def picked_cards : ℕ := 3

-- Probability calculations
noncomputable def probability_different_suit_second_card : ℚ :=
  39 / 51

noncomputable def probability_different_suit_third_card_different_second : ℚ :=
  26 / 50

noncomputable def probability_different_suit_third_card_same_second : ℚ :=
  39 / 50

-- Combined probabilities
noncomputable def probability_different_different : ℚ :=
  (13 / 17) * (13 / 25)

noncomputable def probability_same_different : ℚ :=
  (4 / 17) * (13 / 17)

-- Final probability
noncomputable def final_probability : ℚ :=
  probability_different_different + probability_same_different

-- Main theorem statement
theorem at_most_one_shared_suit : final_probability ≈ 0.607 :=
sorry

end at_most_one_shared_suit_l569_569145


namespace marie_speed_l569_569477

theorem marie_speed :
  ∀ (distance time : ℝ),
    distance = 31.0 → time = 2.583333333 →
    distance / time ≈ 12.0 :=
by
  intros distance time h_distance h_time
  rw [h_distance, h_time]
  sorry

end marie_speed_l569_569477


namespace ratio_of_costs_l569_569805

-- Definitions based on conditions
def quilt_length : Nat := 16
def quilt_width : Nat := 20
def patch_area : Nat := 4
def first_10_patch_cost : Nat := 10
def total_cost : Nat := 450

-- Theorem we need to prove
theorem ratio_of_costs : (total_cost - 10 * first_10_patch_cost) / (10 * first_10_patch_cost) = 7 / 2 := by
  sorry

end ratio_of_costs_l569_569805


namespace paint_cost_for_flag_l569_569563

noncomputable def flag_width : ℕ := 12
noncomputable def flag_height : ℕ := 10
noncomputable def paint_cost_per_quart : ℝ := 3.5
noncomputable def coverage_per_quart : ℕ := 4

theorem paint_cost_for_flag : (flag_width * flag_height * 2 / coverage_per_quart : ℝ) * paint_cost_per_quart = 210 := by
  sorry

end paint_cost_for_flag_l569_569563


namespace dance_fraction_of_boys_l569_569636

def Riverbank_students := 300
def Riverbank_boys_to_girls_ratio := (3, 2)
def Brookside_students := 240
def Brookside_boys_to_girls_ratio := (2, 3)
def Riverbank_attendance_rate := 0.8
def Brookside_attendance_rate := 0.75

theorem dance_fraction_of_boys :
  let Riverbank_boys := 3 * (Riverbank_students / 5) in
  let Brookside_boys := 2 * (Brookside_students / 5) in
  let Riverbank_boys_at_dance := Riverbank_attendance_rate * Riverbank_boys in
  let Brookside_boys_at_dance := Brookside_attendance_rate * Brookside_boys in
  let Total_attendees := Riverbank_attendance_rate * Riverbank_students + Brookside_attendance_rate * Brookside_students in
  (Riverbank_boys_at_dance + Brookside_boys_at_dance) / Total_attendees = 18/35 :=
by {
  let Riverbank_boys := 3 * (Riverbank_students / 5),
  let Brookside_boys := 2 * (Brookside_students / 5),
  let Riverbank_boys_at_dance := Riverbank_attendance_rate * Riverbank_boys,
  let Brookside_boys_at_dance := Brookside_attendance_rate * Brookside_boys,
  let Total_attendees := Riverbank_attendance_rate * Riverbank_students + Brookside_attendance_rate * Brookside_students,
  have fraction_of_boys : (Riverbank_boys_at_dance + Brookside_boys_at_dance) / Total_attendees = 18 / 35,
  sorry
}

end dance_fraction_of_boys_l569_569636


namespace speed_of_man_l569_569984

-- Define the constants based on the problem's conditions
def train_length : ℝ := 130 -- in meters
def crossing_time : ℝ := 6 -- in seconds
def train_speed_kmph : ℝ := 72.99376049916008 -- in kmph

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Speed of the train in m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Calculate the relative speed of the train and the man in m/s
def relative_speed : ℝ := train_length / crossing_time

-- Prove the speed of the man in kmph
theorem speed_of_man :
  let Vm_mps := relative_speed - train_speed_mps
  let Vm_kmph := Vm_mps * 3600 / 1000 in
  Vm_kmph = 5.00623910096 :=
by
  sorry

end speed_of_man_l569_569984


namespace determinant_of_matrix_eq_neg64_add_12p_sub_2q_l569_569458

open Matrix

variables {R : Type*} [CommRing R] [Nontrivial R]

noncomputable def polynomial (p q : R) : R[X] := X^3 - 4*X^2 + p*X + q

def are_distinct_roots {R : Type*} [Field R] (p q : R) (a b c : R) :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ is_root (polynomial p q) a ∧ is_root (polynomial p q) b ∧ is_root (polynomial p q) c

theorem determinant_of_matrix_eq_neg64_add_12p_sub_2q 
  (a b c p q : R)
  (h_distinct_roots : are_distinct_roots p q a b c)
  (h1 : a + b + c = 4)
  (h2 : a * b + b * c + c * a = p)
  (h3 : a * b * c = -q) :
  det ![
    [a, b, c],
    [b, c, a],
    [c, a, b]
  ] = -64 + 12 * p - 2 * q :=
sorry

end determinant_of_matrix_eq_neg64_add_12p_sub_2q_l569_569458


namespace range_of_a_l569_569747

theorem range_of_a (a : ℝ) (h₀ : a > 0)
  (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, y = sqrt (-x^2 + 7*x - 12))
  (h₂ : ∀ x, 0 ≤ x ∧ x ≤ 2 → sqrt (-x^2 + 7*x - 12) ∈ set.Icc (3 : ℝ) 4)
  (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, y = a / (x^2 + 1))
  (h₄ : ∀ x, 0 ≤ x ∧ x ≤ 2 → (a / (x^2 + 1)) ∈ set.Icc (a / 5) a) : 
  4 ≤ a ∧ a ≤ 15 := sorry

end range_of_a_l569_569747


namespace v_2008_equals_7618_l569_569823

-- Define the sequence based on the structure provided.
def sequence (n : ℕ) : ℕ :=
  let m := nat.casesOn n (0) (λ n, nat.sqrt n) + 1
  in m + 4 * (n - (m * (m + 1)) / 2)

-- Define the term for v_n based on given problems.
def v (n : ℕ) : ℕ := sequence n

-- Theorem to prove that v(2008) = 7618.
theorem v_2008_equals_7618 : v 2008 = 7618 := sorry

end v_2008_equals_7618_l569_569823


namespace sequence_has_zero_at_1845_l569_569202

-- Definitions based on problem conditions
def sequence (x : ℕ → ℝ) : Prop :=
  ∀ n ≥ 1, x (n + 2) = x n - (1 / x (n + 1))

-- Sequence contains zero at index 1845
theorem sequence_has_zero_at_1845 (x : ℕ → ℝ) (h : sequence x) : ∃ k, x k = 0 ∧ k = 1845 :=
sorry

end sequence_has_zero_at_1845_l569_569202


namespace symmetric_probability_l569_569617

-- Definitions based on the problem conditions
def total_points : ℕ := 121
def central_point : ℕ × ℕ := (6, 6)
def remaining_points : ℕ := total_points - 1
def symmetric_points : ℕ := 40

-- Predicate for the probability that line PQ is a line of symmetry
def is_symmetrical_line (p q : (ℕ × ℕ)) : Prop := 
  (q.fst = 11 - p.fst ∧ q.snd = p.snd) ∨
  (q.fst = p.fst ∧ q.snd = 11 - p.snd) ∨
  (q.fst + q.snd = 12) ∨ 
  (q.fst - q.snd = 0)

-- The theorem stating the probability is 1/3
theorem symmetric_probability :
  ∃ (total_points : ℕ) (remaining_points : ℕ) (symmetric_points : ℕ),
    total_points = 121 ∧
    remaining_points = total_points - 1 ∧
    symmetric_points = 40 ∧
    (symmetric_points : ℚ) / (remaining_points : ℚ) = 1 / 3 :=
by
  sorry

end symmetric_probability_l569_569617


namespace smallest_page_number_l569_569551

-- Define the conditions using Lean definitions
def divisible_by_all (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 13 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n % 17 = 0

-- Define the main theorem statement using these conditions and the correct answer
theorem smallest_page_number : ∃ n : ℕ, divisible_by_all n ∧ n = 68068 :=
by {
  use 68068,
  split,
  -- show that 68068 satisfies each condition (ignored here as proof is not needed)
  sorry,
  -- show that 68068 is equal to 68068
  refl
}

end smallest_page_number_l569_569551


namespace seventh_term_geometric_sequence_l569_569113

theorem seventh_term_geometric_sequence :
  ∃ (a₁ a₁₀ a₇ : ℕ) (r : ℕ),
    a₁ = 6 ∧ a₁₀ = 93312 ∧
    a₁₀ = a₁ * r^9 ∧
    a₇ = a₁ * r^6 ∧
    a₇ = 279936 :=
by
  sorry

end seventh_term_geometric_sequence_l569_569113


namespace inequality_solution_set_l569_569692

theorem inequality_solution_set (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) ≤ 1) ↔ (x < 2 ∨ 3 ≤ x) :=
sorry

end inequality_solution_set_l569_569692


namespace exists_sequence_an_l569_569020

def d (m n : ℕ) : ℕ :=
  (m.factors.to_finset ∩ n.factors.to_finset).card

theorem exists_sequence_an :
  ∃ (a : ℕ → ℕ),
  (a 1 ≥ 2018^2018) ∧
  (∀ m n : ℕ, m ≤ n → a m ≤ a n) ∧
  (∀ m n : ℕ, m ≠ n → d m n = d (a m) (a n)) :=
sorry

end exists_sequence_an_l569_569020


namespace perp_bisector_through_midpoint_l569_569553

theorem perp_bisector_through_midpoint
  (A O1 O2 C D : Point)
  (hA : is_intersection_of_circles A O1 O2)
  (hCD : is_secant_through A C D)
  (M : Point)
  (hM : is_midpoint M O1 O2) :
  passes_through (perpendicular_bisector C D) M :=
sorry

end perp_bisector_through_midpoint_l569_569553


namespace find_k_l569_569408

theorem find_k (k x y : ℕ) (h : k * 2 + 1 = 5) : k = 2 :=
by {
  -- Proof will go here
  sorry
}

end find_k_l569_569408


namespace find_y_l569_569766

theorem find_y {x y : ℤ} (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := 
by
  sorry

end find_y_l569_569766


namespace circumcircles_intersect_at_single_point_centers_of_circles_form_similar_triangle_l569_569238

-- Let M, K, and L be points on the sides of triangle ABC.
variables {A B C M K L : Type*}
variable [incidenceGeometry A B C]

-- Define circumcircles S1, S2, S3 of triangles LMA, MKB, KLC respectively.
def circumcircle (M K L A : Type*) [incidenceGeometry M K L] : circle := sorry
def S1 := circumcircle L M A
def S2 := circumcircle M K B
def S3 := circumcircle K L C

-- Define the proof problem.

-- Part (a): Prove the circumcircles S1, S2, and S3 intersect at a single point.
theorem circumcircles_intersect_at_single_point 
  (A B C M K L : Type*) 
  [incidenceGeometry A B C] [incidenceGeometry M K L] : 
  ∃ O : Type*, (O ∈ S1 ∧ O ∈ S2 ∧ O ∈ S3) :=
begin
  sorry
end

-- Part (b): Prove the triangle formed by the centers of S1, S2, and S3 is similar to triangle ABC.
theorem centers_of_circles_form_similar_triangle 
  (A B C M K L : Type*) 
  [incidenceGeometry A B C] [incidenceGeometry M K L] : 
  ∃ O1 O2 O3 : Type*, 
  is_center O1 S1 ∧ 
  is_center O2 S2 ∧ 
  is_center O3 S3 ∧ 
  similar_triangle O1 O2 O3 A B C :=
begin
  sorry
end

end circumcircles_intersect_at_single_point_centers_of_circles_form_similar_triangle_l569_569238


namespace polynomial_minimal_degree_l569_569680

noncomputable def minimal_polynomial : Polynomial ℚ :=
  Polynomial.X^4 - 4*Polynomial.X^3 + 4*Polynomial.X^2 + 16*Polynomial.X - 8

theorem polynomial_minimal_degree
  (x : ℚ)
  (root1 : x = 1 + real.sqrt 2 + real.sqrt 3)
  (root2 : x = 1 + real.sqrt 2 - real.sqrt 3)
  (root3 : x = 1 - real.sqrt 2 + real.sqrt 3)
  (root4 : x = 1 - real.sqrt 2 - real.sqrt 3)
  : (minimal_polynomial.eval x = 0) :=
by
  sorry

end polynomial_minimal_degree_l569_569680


namespace sequence_not_d_l569_569065

-- Define the sequence
def sequence (n : ℕ) : ℤ :=
  if n % 2 = 1 then 2 else 0

-- Define the general formula for option D
def a_n_d (n : ℕ) : ℤ :=
  2 * Real.sin (n * Real.pi / 2)

-- Define the main theorem to prove
theorem sequence_not_d : ∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6} → sequence n ≠ a_n_d n :=
by 
  intros n hn
  fin_cases hn <;> sorry

end sequence_not_d_l569_569065


namespace problem_3_l569_569367

def f (x : ℝ) : ℝ := Real.ln (x + 1) - x / (x + 1)

theorem problem_3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  Real.ln a - Real.ln b ≥ 1 - b / a := 
sorry

end problem_3_l569_569367


namespace triangles_are_congruent_in_divided_100gon_l569_569260

theorem triangles_are_congruent_in_divided_100gon :
  ∀ (P: polygon) (n: ℕ), 
  n = 100 ∧ P.is_regular ∧ 
  (∃ (triangles: list triangle) (parallelograms: list parallelogram), 
    P.is_divided_into triangles parallelograms ∧ 
    triangles.length = 2) → 
  triangles.head ≅ triangles.tail.head :=
by -- proof skipped
  sorry

end triangles_are_congruent_in_divided_100gon_l569_569260


namespace sum_M_five_lines_l569_569017

noncomputable def M (lines : Finset (Set (Finset ℝ × Finset ℝ))) : Finset ℕ :=
Finset.image (λ e, (e.1 ∩ e.2).card) (lines.product lines).filter (λ e, e.1 ≠ e.2)

theorem sum_M_five_lines (lines : Finset (Set (Finset ℝ × Finset ℝ))) (h : lines.card = 5) :
  ∑ m in M lines, m = 55 :=
sorry

end sum_M_five_lines_l569_569017


namespace curve_not_parabola_l569_569075

theorem curve_not_parabola (k : ℝ) : ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c = 1 ∧ a * x^2 + b * y = c) :=
sorry

end curve_not_parabola_l569_569075


namespace length_of_AB_l569_569066

-- Definitions
def parabola_equation (y x : ℝ) : Prop := y^2 = 6 * x
def line_L (x y : ℝ) : Prop := y = x - 3/2
def focus (x y : ℝ) : Prop := x = 3/2 ∧ y = 0

-- Theorem statement
theorem length_of_AB :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    parabola_equation y₁ x₁ ∧ parabola_equation y₂ x₂ ∧ 
    line_L x₁ y₁ ∧ line_L x₂ y₂ ∧ 
    x₁ + x₂ = 9 ∧ y₁ ≠ y₂ ∧ ∀ p ∈ [x₁, x₂].to_set, parabola_equation (p - 1/2) p) →
  ∃ x₁ x₂ : ℝ, |x₁ + x₂ + 3/2| = 12 :=
sorry

end length_of_AB_l569_569066


namespace min_diagonal_value_l569_569011

noncomputable def min_diagonal_perimeter_30 : ℝ :=
  let l : ℝ := 7.5
  let w : ℝ := 7.5
  let diagonal : ℝ := Real.sqrt (l^2 + w^2)
  diagonal

theorem min_diagonal_value : min_diagonal_perimeter_30 = Real.sqrt 112.5 := by
  rw [min_diagonal_perimeter_30]
  have h1 : (7.5)^2 + (7.5)^2 = 112.5 := by norm_num
  rw [h1, Real.sqrt_add]
  have h2 : 112.5 = (7.5)^2 + (7.5)^2 := by norm_num
  rw [h2]
  sorry

end min_diagonal_value_l569_569011


namespace ortho_vectors_l569_569672

theorem ortho_vectors :
  ∃ z : ℚ, let v1 := ⟨2, -4, 1⟩ ∧ let v2 := ⟨-1, z, 3⟩ in
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0 ∧ z = 1 / 4 :=
sorry

end ortho_vectors_l569_569672


namespace domain_condition_implies_m_range_range_condition_implies_m_range_l569_569738

noncomputable def f (x m : ℝ) : ℝ := Real.log (x^2 - 2 * m * x + m + 2)

def condition1 (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 2 * m * x + m + 2 > 0)

def condition2 (m : ℝ) : Prop :=
  ∃ y : ℝ, (∀ x : ℝ, y = Real.log (x^2 - 2 * m * x + m + 2))

theorem domain_condition_implies_m_range (m : ℝ) :
  condition1 m → -1 < m ∧ m < 2 :=
sorry

theorem range_condition_implies_m_range (m : ℝ) :
  condition2 m → (m ≤ -1 ∨ m ≥ 2) :=
sorry

end domain_condition_implies_m_range_range_condition_implies_m_range_l569_569738


namespace total_time_spent_in_hours_l569_569478

/-- Miriam's time spent on each task in minutes. -/
def time_laundry := 30
def time_bathroom := 15
def time_room := 35
def time_homework := 40

/-- The function to convert minutes to hours. -/
def minutes_to_hours (minutes : ℕ) := minutes / 60

/-- The total time spent in minutes. -/
def total_time_minutes := time_laundry + time_bathroom + time_room + time_homework

/-- The total time spent in hours. -/
def total_time_hours := minutes_to_hours total_time_minutes

/-- The main statement to be proved: total_time_hours equals 2. -/
theorem total_time_spent_in_hours : total_time_hours = 2 := 
by
  sorry

end total_time_spent_in_hours_l569_569478


namespace min_overlap_percent_l569_569836

theorem min_overlap_percent
  (M S : ℝ)
  (hM : M = 0.9)
  (hS : S = 0.85) :
  ∃ x, x = 0.75 ∧ (M + S - 1 ≤ x ∧ x ≤ min M S ∧ x = M + S - 1) :=
by
  sorry

end min_overlap_percent_l569_569836


namespace triangle_DEF_rotate_180_D_l569_569842

def rotate_180_degrees_clockwise (E D : (ℝ × ℝ)) : (ℝ × ℝ) :=
  let ED := (D.1 - E.1, D.2 - E.2)
  (E.1 - ED.1, E.2 - ED.2)

theorem triangle_DEF_rotate_180_D (D E F : (ℝ × ℝ))
  (hD : D = (3, 2)) (hE : E = (6, 5)) (hF : F = (6, 2)) :
  rotate_180_degrees_clockwise E D = (9, 8) :=
by
  rw [hD, hE, rotate_180_degrees_clockwise]
  sorry

end triangle_DEF_rotate_180_D_l569_569842


namespace sum_sequence_is_negative_fifty_l569_569705

theorem sum_sequence_is_negative_fifty (a : ℕ → ℤ) :
  (a 1 + a 2 = 1) ∧
  (a 2 + a 3 = 2) ∧
  (a 3 + a 4 = -3) ∧
  (a 4 + a 5 = -4) ∧
  (a 5 + a 6 = 5) ∧
  (a 6 + a 7 = 6) ∧
  (a 7 + a 8 = -7) ∧
  (a 8 + a 9 = -8) ∧
  (∀ n : ℕ, 9 ≤ n ∧ n ≤ 99 → a n + a (n + 1) = -(n + 1)) ∧
  (a 100 + a 1 = -100) →
  a 1 + a 2 + a 3 + ... + a 100 = -50 :=
by sorry

end sum_sequence_is_negative_fifty_l569_569705


namespace car_mileage_city_l569_569243

theorem car_mileage_city {h c t : ℝ} (H1: 448 = h * t) (H2: 336 = c * t) (H3: c = h - 6) : c = 18 :=
sorry

end car_mileage_city_l569_569243


namespace centroid_incenter_parallel_ab_l569_569493

variables {A B C : Type} [EuclideanGeometry A B C]
variables {G I : Point} -- assuming G and I are already defined as centroid and incenter of triangle ABC
variables (a b c : ℝ)

/-- Given that c = (a + b) / 2 in a triangle, the line through the centroid and the incenter is parallel to the side AB -/

theorem centroid_incenter_parallel_ab
  (h_triangle : Triangle A B C)
  (h_centroid : Centroid G A B C)
  (h_incenter : Incenter I A B C)
  (h_sides : (c = (a + b) / 2)) :
  Parallel (Line through (G, I)) (Line through (A, B)) :=
  sorry

end centroid_incenter_parallel_ab_l569_569493


namespace largest_sum_eight_integers_l569_569660

-- Conditions
def is_straight_line (nums : list ℕ) (n : ℕ) : Prop :=
  ∀ line : list ℕ, line ⊆ nums → line.length = 3 → (line.head * line.tail.head * line.tail.tail.head = 3240)

-- Problem Statement in Lean 4
theorem largest_sum_eight_integers (nums : list ℕ) (h : is_straight_line nums 45) 
  (h1 : 45 ∈ nums) (h2 : list.nodup nums) (h3 : ∀ n ∈ nums, 0 < n) 
  (h4 : nums.length = 9) : list.sum (list.erase nums 45) ≤ 160 :=
by
  sorry

end largest_sum_eight_integers_l569_569660


namespace geo_sequence_arithmetic_l569_569338

variable {d : ℝ} (hd : d ≠ 0)
variable {a : ℕ → ℝ} (ha : ∀ n, a (n+1) = a n + d)

-- Hypothesis that a_5, a_9, a_15 form a geometric sequence
variable (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d))

theorem geo_sequence_arithmetic (hd : d ≠ 0) (ha : ∀ n, a (n + 1) = a n + d) (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d)) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end geo_sequence_arithmetic_l569_569338


namespace greenville_state_univ_collection_volume_l569_569926

theorem greenville_state_univ_collection_volume :
  ∀ (box_length box_width box_height : ℕ) (cost_per_box : ℚ) (min_total_cost : ℚ),
    box_length = 20 →
    box_width = 20 →
    box_height = 12 →
    cost_per_box = 0.50 →
    min_total_cost = 225 →
    (min_total_cost / cost_per_box) * (box_length * box_width * box_height) = 2160000 :=
by
  intros box_length box_width box_height cost_per_box min_total_cost
  assume h1: box_length = 20
  assume h2: box_width = 20
  assume h3: box_height = 12
  assume h4: cost_per_box = 0.50
  assume h5: min_total_cost = 225
  sorry

end greenville_state_univ_collection_volume_l569_569926


namespace solve_problem_l569_569736

-- Definitions and conditions as given in the problem.
def problem_condition (x y z : ℤ) : Prop :=
  7 ^ (3 * x - 1) * 3 ^ (4 * y - 3) * 2 ^ (5 * z - 4) = 49 ^ x * 27 ^ y * 16 ^ z

-- The theorem we need to prove
theorem solve_problem : ∃ x y z : ℤ, problem_condition x y z ∧ x + y + z = 8 :=
by
  sorry

end solve_problem_l569_569736


namespace isosceles_triangle_vertex_angle_l569_569097

noncomputable def vertex_angle (a b h: ℝ) (H1 : a^2 = 3 * b * h) (H2 : 2 * b * b - 4 * h * h > 0) : ℝ :=
  180 - 2 * real.arcsin (1 / 3)

theorem isosceles_triangle_vertex_angle :
  ∀ (a b h : ℝ) (H1 : a^2 = 3 * b * h) (H2 : 2 * b * b - 4 * h * h > 0), 
  vertex_angle a b h H1 H2 = 141 :=
by sorry

end isosceles_triangle_vertex_angle_l569_569097


namespace angles_not_all_multiples_of_15_l569_569983

theorem angles_not_all_multiples_of_15 (angles : Set ℝ) (cond : ∀ x ∈ angles, x ∈ { y | y = 15 * n ∧ n ∈ ℕ } ∧
    ∀ a b, a ∈ angles → b ∈ angles → a ≠ b → a ≠ b ≠ 90) : 
    False := sorry

end angles_not_all_multiples_of_15_l569_569983


namespace total_lines_correct_l569_569663

-- Define the shapes and their corresponding lines
def triangles := 12
def squares := 8
def pentagons := 4
def hexagons := 6
def octagons := 2

def triangle_sides := 3
def square_sides := 4
def pentagon_sides := 5
def hexagon_sides := 6
def octagon_sides := 8

def lines_in_triangles := triangles * triangle_sides
def lines_in_squares := squares * square_sides
def lines_in_pentagons := pentagons * pentagon_sides
def lines_in_hexagons := hexagons * hexagon_sides
def lines_in_octagons := octagons * octagon_sides

def shared_lines_ts := 5
def shared_lines_ph := 3
def shared_lines_ho := 1

def total_lines_triangles := lines_in_triangles - shared_lines_ts
def total_lines_squares := lines_in_squares - shared_lines_ts
def total_lines_pentagons := lines_in_pentagons - shared_lines_ph
def total_lines_hexagons := lines_in_hexagons - shared_lines_ph - shared_lines_ho
def total_lines_octagons := lines_in_octagons - shared_lines_ho

-- The statement to prove
theorem total_lines_correct :
  total_lines_triangles = 31 ∧
  total_lines_squares = 27 ∧
  total_lines_pentagons = 17 ∧
  total_lines_hexagons = 32 ∧
  total_lines_octagons = 15 :=
by sorry

end total_lines_correct_l569_569663


namespace find_x_l569_569803

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 104) : x = 34 :=
sorry

end find_x_l569_569803


namespace find_new_man_weight_l569_569423

def average_weight_increase (old_avg_weight new_man_weight num_men: ℕ) : Prop :=
  let replaced_man_weight := 75 in
  let weight_difference := new_man_weight - replaced_man_weight in
  old_avg_weight + 2 = (old_avg_weight * num_men + weight_difference) / num_men

theorem find_new_man_weight (num_men : ℕ) (new_man_weight : ℕ) (old_avg_weight : ℕ)
  (h1 : num_men = 15)
  (h2 : ∀ W, average_weight_increase old_avg_weight W num_men) :
  new_man_weight = 105 :=
by {
  -- Other hypotheses and intermediate steps omitted for brevity
  sorry
}

end find_new_man_weight_l569_569423


namespace find_n_cos_log_conjecture_l569_569673

theorem find_n_cos_log_conjecture (n : ℕ) :
  (∑ k in finset.range n, real.cos ((2 * k.succ * real.pi) / 9) = real.cos (real.pi / 9)) ∧ (27 < n ∧ n < 81) :=
sorry

end find_n_cos_log_conjecture_l569_569673


namespace smallest_value_a_plus_b_l569_569343

theorem smallest_value_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 3^7 * 5^3 = a^b) : a + b = 3376 :=
sorry

end smallest_value_a_plus_b_l569_569343


namespace number_of_correct_statements_l569_569521

theorem number_of_correct_statements (s1 s2 s3 s4 : Prop)
  (p1 : s1 = false)
  (p2 : s2 = false)
  (p3 : s3 = false)
  (p4 : s4 = true) : 
  (ite s1 1 0) + (ite s2 1 0) + (ite s3 1 0) + (ite s4 1 0) = 1 :=
by {
  -- The actual details of the proof would go here
  sorry
}

end number_of_correct_statements_l569_569521


namespace circle_intersection_points_l569_569072

theorem circle_intersection_points :
  let circle1 := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 3)^2 = 9}
      circle2 := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2)^2 = 25} in
  (∃ p1 p2 p3 p4 : ℝ × ℝ, p1 ∈ circle1 ∧ p1 ∈ circle2 ∧
                        p2 ∈ circle1 ∧ p2 ∈ circle2 ∧
                        p3 ∈ circle1 ∧ p3 ∈ circle2 ∧
                        p4 ∈ circle1 ∧ p4 ∈ circle2 ∧
                        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧
                        p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :=
by
  sorry

end circle_intersection_points_l569_569072


namespace isolate_y_l569_569320

theorem isolate_y (x y : ℝ) (h : 3 * x - 2 * y = 6) : y = 3 * x / 2 - 3 :=
sorry

end isolate_y_l569_569320


namespace each_child_play_time_l569_569509

-- Define the conditions
def number_of_children : ℕ := 6
def pair_play_time : ℕ := 120
def pairs_playing_at_a_time : ℕ := 2

-- Define main theorem
theorem each_child_play_time : 
  (pairs_playing_at_a_time * pair_play_time) / number_of_children = 40 :=
sorry

end each_child_play_time_l569_569509


namespace pages_needed_l569_569150

def cards_per_page : ℕ := 3
def new_cards : ℕ := 2
def old_cards : ℕ := 10

theorem pages_needed : (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end pages_needed_l569_569150


namespace union_of_A_and_B_l569_569124

theorem union_of_A_and_B :
  let A := {x : ℝ | -1 < x ∧ x < 4}
  let B := {x : ℝ | 2 < x ∧ x < 5}
  A ∪ B = {x : ℝ | -1 < x ∧ x < 5} :=
begin
  sorry
end

end union_of_A_and_B_l569_569124


namespace y_neither_directly_nor_inversely_proportional_l569_569656

theorem y_neither_directly_nor_inversely_proportional (x y : ℝ) :
  ¬((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) ↔ 2 * x + 3 * y = 6 :=
by 
  sorry

end y_neither_directly_nor_inversely_proportional_l569_569656


namespace scientific_notation_44300000_l569_569865

theorem scientific_notation_44300000 : 44_300_000 = 4.43 * 10^7 := 
  sorry

end scientific_notation_44300000_l569_569865


namespace right_triangle_count_l569_569499

def is_right_triangle (P Q R : ℤ × ℤ) : Prop :=
  let P := (Px, Py) in
  let Q := (Qx, Qy) in
  let R := (Rx, Ry) in
  (Py = Ry) ∧ (Px = Qx) ∧ (Qy ≠ Py)

def num_right_triangles : ℕ :=
  let valid_x := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5] in
  let valid_y := [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] in
  let count_px_rx :=
    (valid_x.length * (valid_x.length - 1)) -- choose two different X-coordinates
  let count_py :=
    (valid_y.length - 1) -- for each Y-coordinate pairs, choose a different Y-coordinate for Q
  count_px_rx * count_py

theorem right_triangle_count : num_right_triangles = 900 := by
  sorry

end right_triangle_count_l569_569499


namespace a4_eq_45_l569_569475

noncomputable def S : ℕ → ℝ 
noncomputable def a : ℕ → ℝ 

axiom S2_eq_7 : S 2 = 7
axiom a_recursive : ∀ n : ℕ, 1 ≤ n → a (n + 1) = 2 * S n + 1

theorem a4_eq_45 : a 4 = 45 := sorry

end a4_eq_45_l569_569475


namespace f_2017_plus_f_2019_eq_zero_l569_569348

-- Definitions of even and odd functions and corresponding conditions
variables {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry
noncomputable def g : R → R := λ x, f (x - 1)

axiom even_f : ∀ x : R, f (-x) = f x
axiom odd_g : ∀ x : R, g (-x) = -g x

theorem f_2017_plus_f_2019_eq_zero : f 2017 + f 2019 = 0 := sorry

end f_2017_plus_f_2019_eq_zero_l569_569348


namespace number_of_distributions_l569_569985

/-
  Prove that the number of distinct distributions of 0's and 1's in the 11th row,
  such that the sum of the values in the top square (under the given conditions) is 
  a multiple of 3, is equal to 640.
-/
def top_square_sum_mod_3 (x₀ x₁ x₉ x₁₀ : ℕ) : ℕ :=
  (x₀ + x₁ + x₉ + x₁₀) % 3

theorem number_of_distributions (x : fin 11 → fin 2) :
  (∃ x₀ x₁ x₉ x₁₀, x 0 = x₀ ∧ x 1 = x₁ ∧ x 9 = x₉ ∧ x 10 = x₁₀ ∧
    top_square_sum_mod_3 x₀ x₁ x₉ x₁₀ = 0) →
  5 * 2^7 = 640 := sorry

end number_of_distributions_l569_569985


namespace min_value_z_is_4_l569_569328

noncomputable def min_value_z {x y : ℝ} (h_cond : x^2 + 2 * x * y + 4 * y^2 = 6) : ℝ :=
  ⨅ (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6), x^2 + 4 * y^2

theorem min_value_z_is_4 : min_value_z = 4 := 
  sorry

end min_value_z_is_4_l569_569328


namespace max_volume_tetrahedron_l569_569111

noncomputable def volume_tetrahedron (SA SB SC AB BC AC : ℝ) (h1 : SA = 4) (h2 : SB ≥ 7) (h3 : SC ≥ 9) (h4 : AB = 5) (h5 : BC ≤ 6) (h6 : AC ≤ 8) : ℝ :=
  8 * real.sqrt 6

theorem max_volume_tetrahedron : ∀ (SA SB SC AB BC AC : ℝ),
  SA = 4 → SB ≥ 7 → SC ≥ 9 → AB = 5 → BC ≤ 6 → AC ≤ 8 →
  volume_tetrahedron SA SB SC AB BC AC SA SB SC AB BC AC = 8 * real.sqrt 6 := by
  intros SA SB SC AB BC AC h1 h2 h3 h4 h5 h6
  unfold volume_tetrahedron
  sorry

end max_volume_tetrahedron_l569_569111


namespace total_time_before_playing_game_l569_569117

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end total_time_before_playing_game_l569_569117


namespace number_of_female_students_l569_569185

theorem number_of_female_students 
  (average_all : ℝ)
  (num_males : ℝ) 
  (average_males : ℝ)
  (average_females : ℝ) 
  (h_avg_all : average_all = 88)
  (h_num_males : num_males = 15)
  (h_avg_males : average_males = 80)
  (h_avg_females : average_females = 94) :
  ∃ F : ℝ, 1200 + 94 * F = 88 * (15 + F) ∧ F = 20 :=
by
  use 20
  sorry

end number_of_female_students_l569_569185


namespace sum_three_digit_numbers_from_112277_l569_569005

theorem sum_three_digit_numbers_from_112277 : 
  ∑ n in {227, 277, 117, 177, 112, 122, 127}, n = 1159 := 
sorry

end sum_three_digit_numbers_from_112277_l569_569005


namespace number_of_true_propositions_is_1_l569_569907

def proposition1 (x y : ℝ) : Prop := (x + y = 0) → (x = -y)
def contrapositive1 (x y : ℝ) : Prop := (¬ (x = -y)) → (x + y ≠ 0)

def proposition2 (a b : ℝ) : Prop := (a > b) → (a^2 > b^2)
def converse2 (a b : ℝ) : Prop := (a^2 > b^2) → (a > b)

def proposition3 (x : ℝ) : Prop := (x ≤ -3) → (x^2 - x - 6 > 0)
def negation3 (x : ℝ) : Prop := (x > -3) → ¬ (x^2 - x - 6 > 0)

def proposition4 (a b : ℝ) : Prop := (irrational (a^b)) → (irrational a ∧ irrational b)
def converse4 (a b : ℝ) : Prop := (irrational a ∧ irrational b) → (irrational (a^b))

theorem number_of_true_propositions_is_1 :
  (∀ x y, contrapositive1 x y) ∧
  (¬ (∃ a b, converse2 a b)) ∧
  (¬ (∃ x, negation3 x)) ∧
  (¬ (∃ a b, converse4 a b)) →
  true_propositions_count = 1 :=
sorry

end number_of_true_propositions_is_1_l569_569907


namespace opposite_of_2023_l569_569893

-- Define the opposite (additive inverse) function
def additive_inverse (a : ℤ) : ℤ := -a

-- Define the specific problem condition
def condition (n : ℤ) : Prop := 2023 + n = 0

-- Prove that the additive inverse of 2023 satisfies the condition
theorem opposite_of_2023 : condition (additive_inverse 2023) :=
by
  unfold condition additive_inverse
  simp
  exact rfl

end opposite_of_2023_l569_569893


namespace arithmetic_sequence_general_term_T_bounds_l569_569719

variable {n : ℕ}

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

def first_geometric_condition (a₁ a₂ a₃ : ℤ) : Prop :=
  2 * a₁ * (a₃ + 1) = a₂^2

noncomputable def b (a d : ℤ) (n : ℕ) : ℚ :=
  1 / (arithmetic_sequence a d n * arithmetic_sequence a d (n + 1))

noncomputable def T (a d : ℤ) (n : ℕ) : ℚ :=
  ∑ i in (Finset.range n), b a d i.succ

theorem arithmetic_sequence_general_term (d : ℤ) (h_pos : 0 < d)
  (h_sum : sum_of_first_n_terms 1 d 3 = 12)
  (h_geom : first_geometric_condition 1 (1 + d) (1 + 2 * d)) :
   arithmetic_sequence 1 d n = 3 * n - 2 := sorry

theorem T_bounds (d : ℤ) (h_pos : 0 < d)
  (h_sum : sum_of_first_n_terms 1 d 3 = 12)
  (h_geom : first_geometric_condition 1 (1 + d) (1 + 2 * d)) :
  ∀ n, 1 / 4 ≤ T 1 d n ∧ T 1 d n < 1 / 3 := sorry

end arithmetic_sequence_general_term_T_bounds_l569_569719


namespace solve_system_of_equations_l569_569900

-- Conditions from the problem
variables (x y : ℚ)

-- Definitions (the original equations)
def equation1 := x + 2 * y = 3
def equation2 := 9 * x - 8 * y = 5

-- Correct answer
def solution_x := 17 / 13
def solution_y := 11 / 13

-- The final proof statement
theorem solve_system_of_equations (h1 : equation1 solution_x solution_y) (h2 : equation2 solution_x solution_y) :
  x = solution_x ∧ y = solution_y := sorry

end solve_system_of_equations_l569_569900


namespace functions_eq_l569_569830

open Function

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem functions_eq (h_surj : Surjective f) (h_inj : Injective g) (h_ge : ∀ n : ℕ, f n ≥ g n) : ∀ n : ℕ, f n = g n :=
sorry

end functions_eq_l569_569830


namespace equilateral_triangle_area_l569_569295

-- Lean 4 statement equivalent to the math proof problem.
theorem equilateral_triangle_area (G : Finset (ℕ × ℕ))
    (hG : G.card = 210)
    (h_cond : ∀ n, n ∈ Finset.range 21 → G.count (n, Finset.range (n + 1)) = n + 1)
    (S : Finset (ℕ × ℕ))
    (hS : S = G)
    (h_polygon : closed_non_self_intersecting_polygon S) :
    ∃ A, A = 52 * Real.sqrt 3 := 
sorry

end equilateral_triangle_area_l569_569295


namespace standard_eq_of_line_cartesian_eq_of_circle_PA_PB_distance_l569_569357

noncomputable def param_eqs_of_line (t : ℝ) : ℝ × ℝ :=
  (2 + (real.sqrt 2 / 2) * t, 1 + (real.sqrt 2 / 2) * t)

noncomputable def polar_eq_of_circle (theta : ℝ) : ℝ :=
  4 * real.sqrt 2 * real.sin (theta + real.pi / 4)

theorem standard_eq_of_line :
  ∃ (y_eq_x_minus_1 : ℝ → ℝ), ∀ t : ℝ, param_eqs_of_line t = (t + 2, t + 1) :=
sorry

theorem cartesian_eq_of_circle :
  ∀ x y : ℝ, (∃ theta : ℝ, (x, y) = (polar_eq_of_circle theta * real.cos theta, polar_eq_of_circle theta * real.sin theta))
  → x^2 + y^2 - 4 * x - 4 * y = 0 :=
sorry

theorem PA_PB_distance (P A B : ℝ × ℝ) (t1 t2 : ℝ) :
  P = (2, 1) →
  (P = param_eqs_of_line t1 ∧ P = param_eqs_of_line t2) →
  (t1 + t2 = real.sqrt 2 ∧ t1 * t2 = -7 ∧ t1 ≠ t2) →
  abs (abs t1 - abs t2) = real.sqrt 2 :=
sorry

end standard_eq_of_line_cartesian_eq_of_circle_PA_PB_distance_l569_569357


namespace intersection_A_B_l569_569827

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l569_569827


namespace probability_10_or_9_probability_at_least_7_l569_569982

-- Define the probabilities of hitting each ring
def p_10 : ℝ := 0.1
def p_9 : ℝ := 0.2
def p_8 : ℝ := 0.3
def p_7 : ℝ := 0.3
def p_below_7 : ℝ := 0.1

-- Define the events as their corresponding probabilities
def P_A : ℝ := p_10 -- Event of hitting the 10 ring
def P_B : ℝ := p_9 -- Event of hitting the 9 ring
def P_C : ℝ := p_8 -- Event of hitting the 8 ring
def P_D : ℝ := p_7 -- Event of hitting the 7 ring
def P_E : ℝ := p_below_7 -- Event of hitting below the 7 ring

-- Since the probabilities must sum to 1, we have the following fact about their sum
-- P_A + P_B + P_C + P_D + P_E = 1

theorem probability_10_or_9 : P_A + P_B = 0.3 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

theorem probability_at_least_7 : P_A + P_B + P_C + P_D = 0.9 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

end probability_10_or_9_probability_at_least_7_l569_569982


namespace interest_rate_l569_569190

noncomputable def compoundInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * ((1 + r / 100 : ℚ) ^ t) - P

noncomputable def simpleInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * r * t / 100

theorem interest_rate (P t : ℕ) (D : ℚ) (r : ℕ) :
  P = 10000 → t = 2 → D = 49 →
  compoundInterest P r t - simpleInterest P r t = D → r = 7 := by
  sorry

end interest_rate_l569_569190


namespace find_a_range_l569_569046

theorem find_a_range (a : ℝ) (h₀ : 0 < a ∧ a ≠ 1) (h₁ : ∀ x y : ℝ, x ⊆ Icc (3/2) 2 → y ⊆ Icc (3/2) 2 → (x ≤ y → log a (6 * a * y ^ 2 - 2 * y + 3) ≥ log a (6 * a * x ^ 2 - 2 * x + 3))) :
  a ∈ Ioi (1/24) ∩ Iic (1/12) ∪ Ioi 1 :=
sorry

end find_a_range_l569_569046


namespace consecutive_ints_square_l569_569148

theorem consecutive_ints_square (a b : ℤ) (h : b = a + 1) : 
  a^2 + b^2 + (a * b)^2 = (a * b + 1)^2 := 
by sorry

end consecutive_ints_square_l569_569148


namespace cargo_transport_l569_569594

theorem cargo_transport (bags : ℕ) (capacity_per_bag : ℕ) (total_cargo : ℕ) (truck_capacity : ℕ) (trips : ℕ) :
  bags = 36 → capacity_per_bag ≤ 1 → total_cargo = 36 → truck_capacity = 4 → trips = 11 → 
  (∃ f : fin 36 → ℕ, (∀ i, f i ≤ 1) ∧ ∑ i, f i = 36 ∧ ∃ g : fin 11 → ℕ, (∀ j,  ∑ i in fin 11, f i ≤ 4) ∧ ∑ j, g j = 36) :=
by
  sorry

end cargo_transport_l569_569594


namespace increasing_interval_of_f_l569_569883

-- Define the monotonic increasing interval of sin function
def increasing_interval_sin (k : ℤ) : Set ℝ :=
  {x | 2 * k * Real.pi - Real.pi / 2 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 2}

-- Define the function y = 2^(sin x)
def f (x : ℝ) : ℝ := 2^Real.sin x

-- Define the theorem to prove that the increasing interval of f(x) is given by the aforementioned interval of sin
theorem increasing_interval_of_f (k : ℤ) :
  ∀ x, x ∈ increasing_interval_sin k ↔ (2 * k * Real.pi - Real.pi / 2 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 2) := 
begin
  sorry
end

end increasing_interval_of_f_l569_569883


namespace find_product_of_offsets_l569_569341

theorem find_product_of_offsets
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a * b + a + b = 99)
  (h3 : b * c + b + c = 99)
  (h4 : c * a + c + a = 99) :
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
  sorry

end find_product_of_offsets_l569_569341


namespace jihyae_initial_money_l569_569120

variables {M : ℕ}

def spent_on_supplies (M : ℕ) := M / 2 + 200
def left_after_buying (M : ℕ) := M - spent_on_supplies M
def saved (M : ℕ) := left_after_buying M / 2 + 300
def final_leftover (M : ℕ) := left_after_buying M - saved M

theorem jihyae_initial_money : final_leftover M = 350 → M = 3000 :=
by
  sorry

end jihyae_initial_money_l569_569120


namespace mortgage_payoff_time_l569_569942

theorem mortgage_payoff_time (P1 : ℕ) (total : ℕ) (n : ℕ)
  (hP1 : P1 = 100)
  (htotal : total = 109300)
  (h_total_eq : total = P1 * ((1 - 3^n) / (-2))) :
  n = 7 :=
by
  sorry

end mortgage_payoff_time_l569_569942


namespace james_main_game_time_l569_569116

-- Define the download time in minutes
def download_time := 10

-- Define the installation time as half the download time
def installation_time := download_time / 2

-- Define the combined time for download and installation
def combined_time := download_time + installation_time

-- Define the tutorial time as triple the combined time
def tutorial_time := combined_time * 3

-- Define the total time as the combined time plus the tutorial time
def total_time := combined_time + tutorial_time

-- Statement of the problem to prove
theorem james_main_game_time : total_time = 60 := by
  sorry

end james_main_game_time_l569_569116


namespace hyperbola_asymptotes_min_focal_length_l569_569219

noncomputable def hyperbola_asymptotes (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (x * 2 / 3) ∨ y = -(x * 2 / 3)}

theorem hyperbola_asymptotes_min_focal_length :
  ∀ (x y : ℝ), y ∈ (hyperbola_asymptotes 1) ↔ (y = (x * 2 / 3) ∨ y = -(x * 2 / 3)) :=
by
  sorry

end hyperbola_asymptotes_min_focal_length_l569_569219


namespace geometric_inequality_l569_569095

variable {q : ℝ} {b : ℕ → ℝ}

def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_inequality
  (h_geometric : geometric_sequence b q)
  (h_q_gt_one : q > 1)
  (h_pos : ∀ n : ℕ, b n > 0) :
  b 4 + b 8 > b 5 + b 7 :=
by
  sorry

end geometric_inequality_l569_569095


namespace opposite_of_2023_l569_569891

theorem opposite_of_2023 : ∃ y : ℤ, 2023 + y = 0 ∧ y = -2023 :=
by
  use -2023
  split
  · linarith
  · refl

end opposite_of_2023_l569_569891


namespace repeated_construction_indefinitely_l569_569271

-- Definitions of the semiperimeter s and the side lengths
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Definition of the conditions to construct a new triangle
def new_sides (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let s := semiperimeter a b c;
  (s - a, s - b, s - c)

-- The main theorem to prove
theorem repeated_construction_indefinitely (a b c : ℝ) :
  (∀ (s₁ s₂ s₃ : ℝ), new_sides a b c = (s₁, s₂, s₃) → s₁ + s₂ > s₃ ∧ s₂ + s₃ > s₁ ∧ s₃ + s₁ > s₂) ↔ (a = b ∧ b = c) :=
by
  sorry

end repeated_construction_indefinitely_l569_569271


namespace magnitude_of_sum_of_parallel_vectors_l569_569708

theorem magnitude_of_sum_of_parallel_vectors :
  ∀ (x : ℝ) (a b : ℝ × ℝ),
    a = (x, 1) →
    b = (4, -2) →
    (∃ k : ℝ, a = k • b) →
    ‖a + b‖ = real.sqrt 5 :=
by
  intros x a b ha hb hab
  sorry

end magnitude_of_sum_of_parallel_vectors_l569_569708


namespace isosceles_triangle_base_angle_l569_569099

theorem isosceles_triangle_base_angle
    (X : ℝ)
    (h1 : 0 < X)
    (h2 : 2 * X + X + X = 180)
    (h3 : X + X + 2 * X = 180) :
    X = 45 ∨ X = 72 :=
by sorry

end isosceles_triangle_base_angle_l569_569099


namespace find_angle_A0BC0_l569_569245

noncomputable def angle_A0BC0 (ABC : Triangle) (A B C A0 C0 : Point) (angle_ABC : ABC.angle = 90) : ℝ :=
  let A1 : Point := inscribed_circle_tangent(ABC, AB)
  let A2 : Point := excircle_tangent(BC, ABC)
  let B1 : Point := inscribed_circle_tangent(ABC, AC)
  let center_A : Point := circumcircle_center(A1, A2, B1)
  let C0 : Point := circumcircle_center(C1, C2, B1)
  angle_A0BC0(center_A, B, C0)

theorem find_angle_A0BC0 (ABC : Triangle) (A B C A0 C0 : Point) (angle_ABC: ABC.angle = 90) : angle_A0BC0(ABC, A, B, C, A0, C0, angle_ABC) = 45 := 
sorry

end find_angle_A0BC0_l569_569245


namespace tangent_line_segments_l569_569255

noncomputable def fixed_point_inside_circle (P : Point) (O : Point) (r : Real) : Prop :=
  dist P O < r

noncomputable def arbitrary_point_on_circle (C : Point) (O : Point) (r : Real) : Prop :=
  dist C O = r

noncomputable def chord_perpendicular_to_segment (A B C : Point) : Prop :=
  let AB := A - B
  let BC := B - C
  inner AB BC = 0

noncomputable def projection_onto_line (B A C : Point) : Point :=
  let AC := A - C
  let factor := (inner B C) / (inner C C)
  A + factor * AC

theorem tangent_line_segments (O P : Point) (r : Real)
  (P_inside : fixed_point_inside_circle P O r)
  (C_any : ∀ (C : Point), arbitrary_point_on_circle C O r)
  (B A : Point)
  (AB_perp_BC : chord_perpendicular_to_segment A B C) 
  (X Y : Point)
  (X_proj_B_AC : X = projection_onto_line B A C)
  (Y_proj_B_BC : Y = projection_onto_line B B C) :
  ∃ (circle : Circle), ∀ (C : Point), tangent (line_through X Y) circle :=
sorry

end tangent_line_segments_l569_569255


namespace circle_parametric_eq_l569_569083

theorem circle_parametric_eq 
  (a b r : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi):
  (∃ (x y : ℝ), (x = r * Real.cos θ + a ∧ y = r * Real.sin θ + b)) ↔ 
  (∃ (x' y' : ℝ), (x' = r * Real.cos θ ∧ y' = r * Real.sin θ)) :=
sorry

end circle_parametric_eq_l569_569083


namespace solve_for_ratio_l569_569102

noncomputable def slope_tangent_y_equals_x_squared (x1 : ℝ) : ℝ :=
  2 * x1

noncomputable def slope_tangent_y_equals_x_cubed (x2 : ℝ) : ℝ :=
  3 * x2 * x2

noncomputable def y1_compute (x1 : ℝ) : ℝ :=
  x1 * x1

noncomputable def y2_compute (x2 : ℝ) : ℝ :=
  x2 * x2 * x2

theorem solve_for_ratio (x1 x2 : ℝ)
    (tangent_l_same : slope_tangent_y_equals_x_squared x1 = slope_tangent_y_equals_x_cubed x2)
    (y_tangent_l_same : y1_compute x1 = y2_compute x2) :
  x1 / x2 = 4 / 3 :=
by
  sorry

end solve_for_ratio_l569_569102


namespace quadratic_root_condition_l569_569199

theorem quadratic_root_condition (b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + b*x1 + c = 0) ∧ (x2^2 + b*x2 + c = 0)) ↔ (b^2 - 4*c ≥ 0) :=
by
  sorry

end quadratic_root_condition_l569_569199


namespace isosceles_triangle_points_l569_569424

def geoboard_points (D E : (ℝ × ℝ)) (point_count : ℕ) : Prop :=
  let points := [{(i, j) | i j, i, j ∈ Finset.range 6}] -- all points within a 6x6 grid
  D ∈ points ∧ E ∈ points ∧
  point_count = points.eraseDups.count - 2 (excluding D and E)

def isosceles_triangle_count (D E : (ℝ × ℝ)) (DE : ℝ) : Prop :=
  (∃ F : (ℝ × ℝ), (DE = 3) ∧ F ∈  [{(i, j) | i j, i, j ∈ Finset.range 6}] ∧
  (isosceles DE D E F))

theorem isosceles_triangle_points : 
  ∀ (D E : (ℝ × ℝ)), 
  (D = (2, 2)) ∧ (E = (5, 2)) → 
  (DE : ℝ) → DE = 3 → 
  geoboard_points D E 23 →
  ∃ count, isosceles_triangle_count D E DE ∧ count = 5 := 
sorry

end isosceles_triangle_points_l569_569424


namespace probability_sum_equals_5_l569_569023

/-
From the numbers 1, 2, 3, and 4, two numbers are randomly selected at once. 
Prove that the probability that the sum of these two numbers equals 5 is 1/3.
-/
theorem probability_sum_equals_5 :
  let numbers := {1, 2, 3, 4}
  let pairs := { (a, b) | a ∈ numbers ∧ b ∈ numbers ∧ a < b }
  let favorable_pairs := { (1, 4), (2, 3) }
  P(favorable_pairs, pairs) = 1 / 3 :=
by
  sorry

end probability_sum_equals_5_l569_569023


namespace completing_square_solution_l569_569931

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end completing_square_solution_l569_569931


namespace Z_divisible_by_11_l569_569819

theorem Z_divisible_by_11 (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a < 10) (h₃ : b < 10) (h₄ : c < 10) : 
  let Z := 10000 * a + 1000 * b + 100 * c + 10 * a + b 
  in 11 ∣ Z := 
sorry

end Z_divisible_by_11_l569_569819


namespace motorcycle_avg_speed_is_43_35_l569_569254

noncomputable def motorcycle_avg_speed :=
  let time1 := 30 / 50 
  let time2 := 40 / 55 
  let time3 := 30 / 60 
  let distance3 := 45 * time3
  let time4 := 10 / 60 
  let distance4 := 50 * time4
  let stop_time := 20 / 60 
  let total_distance := 30 + 40 + distance3 + distance4
  let total_time := time1 + time2 + time3 + time4 + stop_time
  total_distance / total_time

theorem motorcycle_avg_speed_is_43_35 : motorcycle_avg_speed ≈ 43.35 :=
by
  sorry

end motorcycle_avg_speed_is_43_35_l569_569254


namespace proof_x_sq_minus_x_inv_sq_l569_569324

theorem proof_x_sq_minus_x_inv_sq (x : ℝ) (h : x^(1/2) - x^(-1/2) = Real.sqrt 5) :
  x - x⁻¹ = 3 * Real.sqrt 5 :=
sorry

end proof_x_sq_minus_x_inv_sq_l569_569324


namespace angle_BAC_in_regular_pentagon_triangle_l569_569978

open Real

theorem angle_BAC_in_regular_pentagon_triangle
  (angle_B_pentagon : ℝ)
  (angle_B_circum_triangle : ℝ)
  (triangle_is_isosceles : Prop)
  (common_vertex_B : Prop)
  :
  ∃ (angle_BAC : ℝ), angle_BAC = 12 :=
by
  have h_pentagon_angle : ∀ n : ℕ, n = 5 → angle_B_pentagon = 108 := sorry
  have h_pentagon_ext_angle : angle_B_circum_triangle = 252 := sorry
  have h_triangle_angles : ∀ n : ℕ, n = 3 → ∀ (angle_BAC : ℝ), triangle_is_isosceles → angle_BAC = 12 := sorry
  apply h_triangle_angles
  trivial
  trivial
  exact triangle_is_isosceles
  sorry

end angle_BAC_in_regular_pentagon_triangle_l569_569978


namespace at_least_pairs_l569_569132

variable {n : ℕ}
def Q := fin n → bool
def differs_in_one_position (a b : Q) : Prop := (finset.univ.filter (λi, a i ≠ b i)).card = 1

theorem at_least_pairs (A : finset Q) (hA : A.card = 2^(n-1)) :
  (∃ (c : finset (Q × Q)), 
     (∀ p ∈ c, p.1 ∈ A ∧ p.2 ∉ A ∧ differs_in_one_position p.1 p.2) ∧ 
     c.card ≥ 2^(n-1)) :=
sorry

end at_least_pairs_l569_569132


namespace find_z_purely_imaginary_l569_569049

theorem find_z_purely_imaginary (z : ℂ) :
  (∃ k : ℝ, z / (1 + I : ℂ) = k * I) ∧ ∥z / (1 + I : ℂ)∥ = 1 →
  (z = -1 + I ∨ z = 1 - I) :=
by
  sorry

end find_z_purely_imaginary_l569_569049


namespace stamps_total_l569_569157

theorem stamps_total :
  ∃ x, x ≡ 1 [MOD 2] ∧
       x ≡ 1 [MOD 3] ∧
       x ≡ 3 [MOD 5] ∧
       x ≡ 7 [MOD 9] ∧
       150 < x ∧ x ≤ 300 ∧
       x = 223 := 
begin
  use 223,
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 2 1 (by norm_num)), },
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 3 1 (by norm_num)), },
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 5 3 (by norm_num)), },
  split,
  { exact nat.modeq.symm (nat.modeq_of_dvd 9 7 (by norm_num)), },
  split,
  { norm_num, },
  split,
  { norm_num, 
  { refl, },
  }
sorry
end

end stamps_total_l569_569157


namespace arrange_books_l569_569268

-- We define the conditions about the number of books
def num_algebra_books : ℕ := 4
def num_calculus_books : ℕ := 5
def total_books : ℕ := num_algebra_books + num_calculus_books

-- The combination function which calculates binomial coefficients
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem stating that there are 126 ways to arrange the books
theorem arrange_books : combination total_books num_algebra_books = 126 :=
  by
    sorry

end arrange_books_l569_569268


namespace fourth_root_sum_of_roots_l569_569405

theorem fourth_root_sum_of_roots (a b c x₁ x₂ : ℝ) 
  (h_eq : a * x₁^2 + b * x₁ + c = 0) 
  (h_eqx : a * x₂^2 + b * x₂ + c = 0) 
  (h_sum : x₁ + x₂ = -b / a) :
  ∜(x₁^4 + 4 * x₁^3 * x₂ + 6 * x₁^2 * x₂^2 + 4 * x₁ * x₂^3 + x₂^4) = -b / a :=
by sorry

end fourth_root_sum_of_roots_l569_569405


namespace gcd_7829_999_repeating_decimal_as_fraction_fraction_in_lowest_terms_l569_569308

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

theorem gcd_7829_999 : gcd 7829 999 = 1 := by
  sorry

theorem repeating_decimal_as_fraction : (7 + 836 / 999) = (7829 / 999) := by
  sorry

theorem fraction_in_lowest_terms : Nat.gcd 7829 999 = 1 := gcd_7829_999

end gcd_7829_999_repeating_decimal_as_fraction_fraction_in_lowest_terms_l569_569308


namespace statement_a_statement_b_statement_c_statement_d_l569_569764

open Real

-- Statement A (incorrect)
theorem statement_a (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a*c > b*d) := sorry

-- Statement B (correct)
theorem statement_b (a b : ℝ) (h1 : b < a) (h2 : a < 0) : (1 / a < 1 / b) := sorry

-- Statement C (incorrect)
theorem statement_c (a b : ℝ) (h : 1 / (a^2) < 1 / (b^2)) : ¬ (a > abs b) := sorry

-- Statement D (correct)
theorem statement_d (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : (a + m) / (b + m) > a / b := sorry

end statement_a_statement_b_statement_c_statement_d_l569_569764


namespace time_to_pass_platform_is_160_seconds_l569_569227

def length_of_train : ℝ := 1200
def time_to_cross_tree : ℝ := 120
def length_of_platform : ℝ := 400
def speed_of_train : ℝ := length_of_train / time_to_cross_tree
def total_distance : ℝ := length_of_train + length_of_platform
def time_to_pass_platform : ℝ := total_distance / speed_of_train

theorem time_to_pass_platform_is_160_seconds :
  time_to_pass_platform = 160 := by
  sorry

end time_to_pass_platform_is_160_seconds_l569_569227


namespace max_length_OB_l569_569915

theorem max_length_OB (O A B : Point) (angle_AOB : angle) 
  (h1 : 45 = angle_AOB.to_degrees) (h2 : dist A B = 1) : 
  ∃ M, M = sqrt 2 ∧ ∀ OB, dist O B = OB → OB ≤ M :=
sorry

end max_length_OB_l569_569915


namespace min_value_expression_l569_569300

theorem min_value_expression : ∃ x y : ℝ, (x = 2 ∧ y = -3/2) ∧ ∀ a b : ℝ, 2 * a^2 + 2 * b^2 - 8 * a + 6 * b + 28 ≥ 10.5 :=
sorry

end min_value_expression_l569_569300


namespace circle_equation_AB_diameter_l569_569043

theorem circle_equation_AB_diameter (A B : ℝ × ℝ) :
  A = (1, -4) → B = (-5, 4) →
  ∃ C : ℝ × ℝ, C = (-2, 0) ∧ ∃ r : ℝ, r = 5 ∧ (∀ x y : ℝ, (x + 2)^2 + y^2 = 25) :=
by intros h1 h2; sorry

end circle_equation_AB_diameter_l569_569043


namespace number_of_ordered_tuples_l569_569755

-- Define the problem statement
def unique_correct_condition (a b c d : ℕ) : Prop :=
  (a = 1 ∧ b ≠ 1 ∧ c ≠ 2 ∧ d ≠ 4) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c = 2 ∧ d ≠ 4) ∨
  (a ≠ 1 ∧ b = 1 ∧ c ≠ 2 ∧ d ≠ 4) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 2 ∧ d = 4)

-- Define the condition that only one of the above is true
def only_one_true (a b c d : ℕ) :=
  (a = 1 ↔ ((b ≠ 1) ∧ (c ≠ 2) ∧ (d ≠ 4))) ∧
  (c = 2 ↔ ((a ≠ 1) ∧ (b ≠ 1) ∧ (d ≠ 4))) ∧
  (b = 1 ↔ ((a ≠ 1) ∧ (c ≠ 2) ∧ (d ≠ 4))) ∧
  (d = 4 ↔ ((a ≠ 1) ∧ (b ≠ 1) ∧ (c ≠ 2)))

-- Given the set constraint {a, b, c, d} = {1, 2, 3, 4}
def set_constraint (a b c d : ℕ) : Prop :=
  {a, b, c, d} = {1, 2, 3, 4}

-- Combining the conditions of the problem into a single proposition
def problem_condition (a b c d : ℕ) : Prop :=
  set_constraint a b c d ∧ unique_correct_condition a b c d ∧ only_one_true a b c d

-- Proving the number of ordered tuples (a, b, c, d) satisfying the given condition is 6
theorem number_of_ordered_tuples : ∃ (l : List (ℕ × ℕ × ℕ × ℕ)), (∀ t ∈ l, let (a, b, c, d) := t in problem_condition a b c d) ∧ l.length = 6 :=
by
  sorry

end number_of_ordered_tuples_l569_569755


namespace projection_of_b_on_a_l569_569067

-- Define the conditions in Lean 4
variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)
variable (h1 : inner a b = (|a| * |b| * real.cos (2 * real.pi / 3)))
variable (h2 : inner (a + b) (a - 2 • b) = 0)
variable (h3 : ∥a∥ = 2)

-- Statement of the proof problem
theorem projection_of_b_on_a (V : Type*) [inner_product_space ℝ V]
  (a b : V) (h1 : inner a b = (|a| * |b| * real.cos (2 * real.pi / 3)))
  (h2 : inner (a + b) (a - 2 • b) = 0)
  (h3 : ∥a∥ = 2) :
  (∥b∥ = (real.sqrt 33 + 1) / 4) → inner b a / ∥a∥ = - (real.sqrt 33 + 1) / 8 :=
begin
  sorry,
end

end projection_of_b_on_a_l569_569067


namespace find_cost_price_l569_569987

-- Define the cost price, selling price at loss, and selling price at gain
def cost_price := 571.43
def SP1 (CP : ℝ) := 0.75 * CP
def SP2 (CP : ℝ) := 1.10 * CP

-- State the conditions
def condition1 (CP : ℝ) := SP2 CP = SP1 CP + 200
def condition2 (CP : ℝ) := 200 = 0.35 * CP

-- Prove that the cost price satisfies the given conditions
theorem find_cost_price (CP : ℝ) (h1 : SP2 CP = SP1 CP + 200) (h2 : 200 = 0.35 * CP) : CP = cost_price :=
by 
  sorry

end find_cost_price_l569_569987


namespace three_pow_y_plus_two_l569_569761

theorem three_pow_y_plus_two (y : ℕ) (h : 3^y = 81) : 3^(y+2) = 729 := sorry

end three_pow_y_plus_two_l569_569761


namespace find_distance_between_foci_of_ellipse_l569_569010

def distance_between_foci (h k a² b² : ℝ) : ℝ :=
  let c := sqrt (a² - b²)
  2 * c

theorem find_distance_between_foci_of_ellipse : distance_between_foci 3 (-4) 36 16 = 2 * sqrt 5 :=
by
  sorry

end find_distance_between_foci_of_ellipse_l569_569010


namespace power_function_m_l569_569416

theorem power_function_m (m : ℝ) 
  (h_even : ∀ x : ℝ, x^m = (-x)^m) 
  (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → x^m > y^m) : m = -2 :=
sorry

end power_function_m_l569_569416


namespace part1_part2_l569_569828

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x / (Real.log x) - a * x

-- Define the derivative of f(x)
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := (1 - Real.log x) / (Real.log x)^2 - a

-- Part 1: Prove that the function f(x) is monotonically decreasing on (1, +∞)
theorem part1 (a : ℝ) :
  (∀ x > 1, f_prime x a ≤ 0) ↔ a ≥ 0.25 := sorry

-- Part 2: Prove the range of a for the given conditions
theorem part2 (a : ℝ) :
  (∃ x1 x2 ∈ set.Icc Real.exp (Real.exp 2), f x1 a ≤ f_prime x2 a + a) ↔ 
  (0 < a ∧ a < 0.25) ∨ (a ≥ 0.5 - 1 / (4*Real.exp 2)) := sorry

end part1_part2_l569_569828


namespace sequence_of_implications_l569_569946

open ProbabilityTheory MeasureTheory

-- Definitions of conditions
variable {Ω : Type*} [MeasurableSpace Ω] {ξ : ℕ → Ω → ℝ} (h_id : ∀ n, ξ n ∼ ξ 1)

-- The proof problem statement
theorem sequence_of_implications
  {ξ : ℕ → Ω → ℝ}
  (h_id : ∀ n, ξ n ∼ ξ 1) :
  (∃ c, 0 < c ∧ ∀ ε > 0, E[| ξ 1 |] < ∞ ↔ ∑' n, (P(| ξ 1 | > ε * n)) < ∞) ∧
  (∀ ε > 0, TFAE [
    ∑' n, (P(| ξ n | > ε * n)) < ∞,
    P(∀ᶠ n in filter.at_top, | ξ n | ≤ ε * n) = 1,
    ∃ c, 0 < c ∧ P((| ξ n | > ε * n) ∈ᶠ at_top) = 0,
    limsup (λ n, | ξ n | / n) = 0]) :=
sorry

end sequence_of_implications_l569_569946


namespace cubic_polynomials_condition_quartic_polynomials_counterexample_l569_569163

open Real

theorem cubic_polynomials_condition {P Q R : ℝ[X]} (hP : P.degree ≤ 3) (hQ : Q.degree ≤ 3) (hR : R.degree ≤ 3) 
  (hx : ∀ x : ℝ, P.eval x ≤ Q.eval x ∧ Q.eval x ≤ R.eval x) 
  (x0 : ℝ) (hx0 : P.eval x0 = R.eval x0) : 
  ∃ k ∈ Icc (0 : ℝ) 1, ∀ x : ℝ, Q.eval x = k * P.eval x + (1 - k) * R.eval x :=
sorry

theorem quartic_polynomials_counterexample : ¬ (∀ (P Q R : ℝ[X]),
  P.degree ≤ 4 ∧ Q.degree ≤ 4 ∧ R.degree ≤ 4 ∧
  (∀ x : ℝ, P.eval x ≤ Q.eval x ∧ Q.eval x ≤ R.eval x) ∧
  (∃ x0 : ℝ, P.eval x0 = R.eval x0) →
  ∃ k ∈ Icc (0 : ℝ) 1, ∀ x : ℝ, Q.eval x = k * P.eval x + (1 - k) * R.eval x) :=
sorry

end cubic_polynomials_condition_quartic_polynomials_counterexample_l569_569163


namespace chord_length_circle_line_l569_569878
open Real

theorem chord_length_circle_line (a : ℝ)
  (h1 : ∃ l, (∀ x y, l (8*x - 6*y - 3) = 0) ∧ ∀ x y, (x^2 + y^2 - 2*x + a = 0))
  (h2 : ∀ x y, chord_length (8*x - 6*y - 3) (x^2 + y^2 - 2*x + a) = sqrt 3) :
  a = 0 :=
sorry

end chord_length_circle_line_l569_569878


namespace min_tangent_sum_l569_569746

def f (x : ℝ) := x - Real.exp (-x)

def tangent_line (m n x : ℝ) := m * x + n

theorem min_tangent_sum :
  ∃ t : ℝ, (∀ x, tangent_line (1 + Real.exp (-t)) (-t * Real.exp (-t) + Real.exp (-t)) x = f x - f t * (x - t)) ∧
  ((1 + Real.exp (-t)) - (t + 1) * Real.exp (-t) = 1 - 1 / Real.exp 1) :=
sorry

end min_tangent_sum_l569_569746


namespace sqrt_4_eq_2_or_neg2_l569_569541

theorem sqrt_4_eq_2_or_neg2 (y : ℝ) (h : y^2 = 4) : y = 2 ∨ y = -2 :=
sorry

end sqrt_4_eq_2_or_neg2_l569_569541


namespace distinct_cube_arrangements_count_l569_569306

def is_valid_face_sum (face : Finset ℕ) : Prop :=
  face.sum id = 34

def is_valid_opposite_sum (v1 v2 : ℕ) : Prop :=
  v1 + v2 = 16

def is_unique_up_to_rotation (cubes : List (Finset ℕ)) : Prop := sorry -- Define rotational uniqueness check

noncomputable def count_valid_arrangements : ℕ := sorry -- Define counting logic

theorem distinct_cube_arrangements_count : count_valid_arrangements = 3 :=
  sorry

end distinct_cube_arrangements_count_l569_569306


namespace usual_jog_time_l569_569285

-- Define conditions
def jog_time (x : ℕ) : Prop :=
  let total_minutes : ℕ := 180 in
  -- Setup the condition based on given problem
  let jogged_more_on_tuesday : ℕ := 5 in
  let jogged_more_on_friday : ℕ := 25 in
  5 * x + jogged_more_on_tuesday + jogged_more_on_friday = total_minutes

-- Statement to be proved
theorem usual_jog_time :
  ∃ x : ℕ, jog_time x ∧ x = 30 :=
begin
  sorry
end

end usual_jog_time_l569_569285


namespace determine_valid_k_l569_569298

noncomputable def problem_statement (k : ℂ) : Prop :=
  ∀ x : ℂ, (x / (x + 3) + x / (x + 4) = k * x) → 
    x = 0 ∨ 
    (k * x^2 + (7 * k - 2) * x + (12 * k - 7) = 0)

theorem determine_valid_k : 
  { k : ℂ | k = 0 ∨ k = 2 * complex.I ∨ k = -2 * complex.I } = 
  { k : ℂ | problem_statement k } :=
begin 
  sorry
end

end determine_valid_k_l569_569298


namespace locus_of_M_l569_569949

-- Define the segment lengths and their midpoints
variable (A B C D M : ℝ^3)

-- Define midpoints
def midpoint (X Y : ℝ^3) := (X + Y) / 2

-- Midpoints of AB and CD
def P := midpoint A B
def Q := midpoint C D

-- Define squared distance function
def sq_dist (X Y : ℝ^3) : ℝ := (X - Y).dot (X - Y)

-- Main theorem statement
theorem locus_of_M : sq_dist M A + sq_dist M B = sq_dist M C + sq_dist M D → 
  (M - P).dot (M - P) - (M - Q).dot (M - Q) = (sq_dist C D - sq_dist A B) / 4 := by
  sorry

end locus_of_M_l569_569949


namespace place_value_ratio_l569_569002

theorem place_value_ratio :
  let val_6 := 1000
  let val_2 := 0.1
  val_6 / val_2 = 10000 :=
by
  -- the proof would go here
  sorry

end place_value_ratio_l569_569002


namespace geometric_sequence_problem_l569_569346

theorem geometric_sequence_problem (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : ∀ n, a (n + 1) = r * a n) 
  (h_cond: a 4 + a 6 = 8) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
  sorry

end geometric_sequence_problem_l569_569346


namespace cone_base_radius_half_l569_569524

theorem cone_base_radius_half :
  let R : ℝ := sorry
  let semicircle_radius : ℝ := 1
  let unfolded_circumference : ℝ := π
  let base_circumference : ℝ := 2 * π * R
  base_circumference = unfolded_circumference -> R = 1 / 2 :=
by
  sorry

end cone_base_radius_half_l569_569524


namespace max_smaller_cylinders_l569_569963

def volume_cylinder (r : ℝ) (h : ℝ) : ℝ :=
  real.pi * r^2 * h

def large_radius : ℝ := 3
def large_height : ℝ := 8
def small_radius : ℝ := 2
def small_height : ℝ := 5

noncomputable def large_volume : ℝ := volume_cylinder large_radius large_height
noncomputable def small_volume : ℝ := volume_cylinder small_radius small_height

noncomputable def num_smaller_cylinders : ℝ := large_volume / small_volume

theorem max_smaller_cylinders (n : ℕ) : n = 3 :=
by
  have h : n = real.floor num_smaller_cylinders := sorry
  exact h

end max_smaller_cylinders_l569_569963


namespace functional_equation_ns_product_l569_569467

theorem functional_equation_ns_product :
  ∃ (f : ℝ → ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → f x * f y = f (x * y) + 1001 * (1 / x + 1 / y + 1000)) →
  let n := 1 in
  let s := (f 2) in
  n * s = 2003 / 2 :=
by
  sorry

end functional_equation_ns_product_l569_569467


namespace probability_alpha_at_least_14_days_probability_alpha_longer_than_beta_variance_equal_a_l569_569068

def group_A := [10, 11, 12, 13, 14, 15, 16]
def group_B (a : ℕ) := [12, 13, 15, 16, 17, 14, a]

/-- Question I: Calculate the probability that alpha's recovery time is at least 14 days -/
theorem probability_alpha_at_least_14_days 
  : (P : probability (event (group_A.filter (λ x, x ≥ 14))).length / group_A.length = 3/7) :=
sorry

/-- Question II: If a = 25, calculate the probability that alpha's recovery time is longer than beta's recovery time -/
theorem probability_alpha_longer_than_beta (a : ℕ) : 
  a = 25 → (P : probability (event (λ α β, α > β) (group_A × group_B 25)).length / (group_A.length * group_B 25.length) = 10/49) :=
sorry

/-- Question III: For what value of a will the variances of the recovery times for groups A and B be equal -/
theorem variance_equal_a (a : ℕ) : 
  (variance (group_A) = variance (group_B a)) ↔ (a = 11 ∨ a = 18) :=
sorry

end probability_alpha_at_least_14_days_probability_alpha_longer_than_beta_variance_equal_a_l569_569068


namespace new_number_factor_l569_569769

theorem new_number_factor (a b k : ℤ) (h : 10 * a + b = (k + 1) * (a + b)) : 
  let new_number := 10 * (a + 1) + (b + 1),
      sum_digits_plus_two := a + b + 2 in
  new_number = ((k + 1) * (a + b) + 11) / sum_digits_plus_two :=
by
  sorry

end new_number_factor_l569_569769


namespace integral_of_continuous_function_interval_l569_569138

open Set Filter IntervalIntegral
open scoped Topology BigOperators

noncomputable def integral_of_continuous_function (f : ℝ → ℝ) (hf1 : ∀ x ∈ (Icc 0 1), f x ∈ (Icc 0 1))
  (hf2 : ∀ x ∈ (Icc 0 1), f (f x) = 1) : Icc (3 / 4 : ℝ) 1 :=
  sorry

theorem integral_of_continuous_function_interval (f : ℝ → ℝ) (hf1 : ∀ x ∈ (Icc 0 1), f x ∈ (Icc 0 1)) 
  (hf2 : ∀ x ∈ (Icc 0 1), f (f x) = 1) : 
  3 / 4 < ∫ x in (0 : ℝ)..1, f x ∧ ∫ x in (0 : ℝ)..1, f x ≤ 1 :=
sorry

end integral_of_continuous_function_interval_l569_569138


namespace triangles_congruent_in_regular100gon_division_l569_569262

theorem triangles_congruent_in_regular100gon_division
  (G : Type) [fintype G] [decidable_eq G] [polygon G 100]
  (is_regular : regular G)
  (partition_into_parallelograms_and_triangles 
    : ∃ (P : list (parallelogram G) × (triangle G) × (triangle G)), 
      ⋃ x ∈ P.1 ++ P.2 ++ P.3, x = set_of_vertices G) :
  ∃ (T1 T2 : triangle G), congruent T1 T2 :=
by
  sorry

end triangles_congruent_in_regular100gon_division_l569_569262


namespace rationalize_denominator_sum_l569_569847

theorem rationalize_denominator_sum :
  ∃ (A B C D : ℕ), 
    (D = 1) ∧ 
    (A = 9) ∧ 
    (B = 6) ∧ 
    (C = 4) ∧ 
    (A + B + C + D = 20) :=
by {
  have A := 9,
  have B := 6,
  have C := 4,
  have D := 1,
  have sum := A + B + C + D,
  exact ⟨A, B, C, D, by refl, by refl, by refl, by refl, by refl⟩
}

end rationalize_denominator_sum_l569_569847


namespace Jungkook_blue_balls_unchanged_l569_569449

variable (initialRedBalls : ℕ) (initialBlueBalls : ℕ) (initialYellowBalls : ℕ)
variable (newYellowBallGifted: ℕ)

-- Define the initial conditions
def Jungkook_balls := initialRedBalls = 5 ∧ initialBlueBalls = 4 ∧ initialYellowBalls = 3 ∧ newYellowBallGifted = 1

-- State the theorem to prove
theorem Jungkook_blue_balls_unchanged (h : Jungkook_balls initRed initBlue initYellow newYellowGift): initialBlueBalls = 4 := 
by
sorry

end Jungkook_blue_balls_unchanged_l569_569449


namespace angle_between_vectors_l569_569379

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (ha : a = (1, Real.sqrt 3))
  (hb : b = (-2, 2 * Real.sqrt 3)) :
  let dot_product := a.1 * b.1 + a.2 * b.2,
      magnitude_a := Real.sqrt (a.1^2 + a.2^2),
      magnitude_b := Real.sqrt (b.1^2 + b.2^2),
      cos_theta := dot_product / (magnitude_a * magnitude_b) in
  cos_theta = 1 / 2 → ∃ θ : ℝ, θ = Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l569_569379


namespace int_root_exists_l569_569535

noncomputable def is_root (p : Polynomial ℚ) (a : ℚ) : Prop :=
  Polynomial.eval a p = 0

theorem int_root_exists {d e f g : ℚ} : 
  is_root (Polynomial.C (1:ℚ) * Polynomial.X ^ 4 +
           Polynomial.C d * Polynomial.X ^ 3 +
           Polynomial.C e * Polynomial.X ^ 2 +
           Polynomial.C f * Polynomial.X +
           Polynomial.C g)
          (3 + Real.sqrt 5) →
  ∃ n : ℤ, 
    is_root (Polynomial.C (1:ℚ) * Polynomial.X ^ 4 +
             Polynomial.C d * Polynomial.X ^ 3 +
             Polynomial.C e * Polynomial.X ^ 2 +
             Polynomial.C f * Polynomial.X +
             Polynomial.C g)
            n ∧ (n = -3) :=
by
  sorry

end int_root_exists_l569_569535


namespace Ahmad_eight_steps_l569_569625

def reach_top (n : Nat) (holes : List Nat) : Nat := sorry

theorem Ahmad_eight_steps (h : reach_top 8 [6] = 8) : True := by 
  trivial

end Ahmad_eight_steps_l569_569625


namespace probability_of_stopping_on_corner_l569_569698

def grid_size : ℕ := 4

inductive Direction
| up
| down
| left
| right

structure Position :=
(x : ℕ)
(y : ℕ)
deriving DecidableEq

def initial_position : Position := { x := 0, y := 0 }

def is_corner (pos : Position) : Bool :=
  (pos.x = 0 ∧ pos.y = 0) ∨
  (pos.x = grid_size - 1 ∧ pos.y = 0) ∨
  (pos.x = 0 ∧ pos.y = grid_size - 1) ∨
  (pos.x = grid_size - 1 ∧ pos.y = grid_size - 1)

def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.up    => { p with y := (p.y + 1) % grid_size }
  | Direction.down  => { p with y := (p.y + grid_size - 1) % grid_size }
  | Direction.left  => { p with x := (p.x + grid_size - 1) % grid_size }
  | Direction.right => { p with x := (p.x + 1) % grid_size }

def random_walk (p : Position) (steps : ℕ) : list Position :=
  if steps = 0 then [p]
  else
    let next_positions := Direction.all.map (move p)
    p :: (next_positions.choose random_walk (steps - 1)).flatten

def probability_stopping_on_corner : ℚ := 89 / 256

theorem probability_of_stopping_on_corner :
  ∀ p, p = initial_position →
  (∃ steps ≤ 5, is_corner (random_walk p steps).last = true) →
  probability_stopping_on_corner = (89 / 256) := by
    sorry

end probability_of_stopping_on_corner_l569_569698


namespace fraction_defined_iff_l569_569555

theorem fraction_defined_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (|x| - 6)) ↔ (x ≠ 6 ∧ x ≠ -6) :=
by 
  sorry

end fraction_defined_iff_l569_569555


namespace solution_set_of_inequality_l569_569461

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry
noncomputable def g' (x : ℝ) : ℝ := sorry

lemma odd_function : ∀ x, f (-x) = -f x := sorry
lemma even_function : ∀ x, g (-x) = g x := sorry
lemma fg_derivative_condition (x : ℝ) (h : x < 0)
  : f' x * g x + f x * g' x > 0 := sorry
lemma g_at_neg3 : g (-3) = 0 := sorry

theorem solution_set_of_inequality : {x : ℝ | f x * g x < 0} = Ioo (-∞) (-3) ∪ Ioo 0 3 := sorry

end solution_set_of_inequality_l569_569461


namespace sasha_equalities_l569_569851

def a : ℤ := -2
def b : ℤ := 2
def c : ℤ := 1
def d : ℤ := 5

theorem sasha_equalities : 
  (-2) - (+2) + 5 = 1 ∧
  2 - (-2) + 1 = 5 ∧
  5 - (-2)^2 = 1 ∧
  2 + 1 - 5 = -2 ∧
  1 - 5 / -2 = 2 ∧
  5 + 2 * (-2) = 1 ∧
  5^(2 + (-2)) = 1 := by
  repeat {apply and.intro};
  { have h : (-2) - 2 + 5 = 1 := by } sorry;
  { have h : 2 - (-2) + 1 = 5 := by } sorry;
  { have h : 5 - (-2)^2 = 1 := by } sorry;
  { have h : 2 + 1 - 5 = -2 := by } sorry;
  { have h : 1 - 5 / -2 = 2 := by } sorry;
  { have h : 5 + 2 * (-2) = 1 := by } sorry;
  { have h : 5^(2 + (-2)) = 1 := by } sorry;

end sasha_equalities_l569_569851


namespace opposite_of_2023_l569_569894

-- Define the opposite (additive inverse) function
def additive_inverse (a : ℤ) : ℤ := -a

-- Define the specific problem condition
def condition (n : ℤ) : Prop := 2023 + n = 0

-- Prove that the additive inverse of 2023 satisfies the condition
theorem opposite_of_2023 : condition (additive_inverse 2023) :=
by
  unfold condition additive_inverse
  simp
  exact rfl

end opposite_of_2023_l569_569894


namespace incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l569_569162

structure Tetrahedron (α : Type*) [MetricSpace α] :=
(A B C D : α)

def Incenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry
def Circumcenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry

def equidistant_from_faces {α : Type*} [MetricSpace α] (T : Tetrahedron α) (I : α) : Prop := sorry
def equidistant_from_vertices {α : Type*} [MetricSpace α] (T : Tetrahedron α) (O : α) : Prop := sorry
def skew_edges_equal {α : Type*} [MetricSpace α] (T : Tetrahedron α) : Prop := sorry

theorem incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal
  {α : Type*} [MetricSpace α] (T : Tetrahedron α) :
  (∃ I, ∃ O, (Incenter T = I) ∧ (Circumcenter T = O) ∧ 
            (equidistant_from_faces T I) ∧ (equidistant_from_vertices T O)) ↔ (skew_edges_equal T) := 
sorry

end incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l569_569162


namespace g_deriv_periodicity_l569_569041

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable (h₀ : ∀ x, f'(x) = (deriv f) x)
variable (h₁ : ∀ x, g'(x) = (deriv g) x)
variable (h₂ : ∀ x, f(x) + g'(x) = 5)
variable (h₃ : ∀ x, f(2 - x) - g'(2 + x) = 5)
variable (h₄ : ∀ x, g(x) = - g(-x))

theorem g_deriv_periodicity : ∀ x, g'(8 - x) = g'(x) :=
by
  sorry

end g_deriv_periodicity_l569_569041


namespace total_students_registered_l569_569855

theorem total_students_registered (num_present_yesterday : ℕ) (num_absent_today : ℕ) 
  (percent_less : ℝ) (num_students : ℕ) 
  (h1 : num_present_yesterday = 70)
  (h2 : num_absent_today = 30)
  (h3 : percent_less = 0.1)
  (h4 : num_students = 156) :
  let twice_present_yesterday := 2 * num_present_yesterday in
  let reduction := percent_less * twice_present_yesterday in
  let num_present_today := twice_present_yesterday - reduction in
  num_students = num_present_today + num_absent_today :=
by
  -- Using the conditions provided to show the equivalence.
  sorry

end total_students_registered_l569_569855


namespace perpendicular_slope_l569_569691

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end perpendicular_slope_l569_569691


namespace g_value_at_2_l569_569129

def g (x : ℝ) (d : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

theorem g_value_at_2 (d : ℝ) (h : g (-2) d = 4) : g 2 d = -84 := by
  sorry

end g_value_at_2_l569_569129


namespace not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l569_569222

theorem not_right_triangle_sqrt_3_sqrt_4_sqrt_5 :
  ¬ (Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2 :=
by
  -- Start constructing the proof here
  sorry

end not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l569_569222


namespace max_min_sum_eq_4027_l569_569347

def f (x : ℝ) : ℝ := (2016^(x+1) + 2011) / (2016^x + 1) + x^3

theorem max_min_sum_eq_4027 {a : ℝ} (ha : a > 0) : 
  let M := Real.maximalOn (f) (Set.Icc (-a) a)
  let N := Real.minimalOn (f) (Set.Icc (-a) a)
  M + N = 4027 :=
sorry

end max_min_sum_eq_4027_l569_569347


namespace sum_of_cubes_l569_569921

theorem sum_of_cubes : (∑ i in Finset.range 101, i^3) + (∑ i in Finset.range 101, (- (i: ℤ))^3) = 0 := 
by
  sorry

end sum_of_cubes_l569_569921


namespace cos_angle_ACB_eq_xy_l569_569101

-- Given conditions: angles in the tetrahedron and definitions of x and y
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (angle_A B D = 90) (angle_A C D = 90) (angle_B C D = 90)
variables (x : ℝ) (y : ℝ)
variable (h1 : x = real.sin (angle A C D))
variable (h2 : y = real.sin (angle B C D))

-- Goal: Proof that cos ∠ACB = xy
theorem cos_angle_ACB_eq_xy : (real.cos (angle A C B)) = x * y := 
by
  sorry

end cos_angle_ACB_eq_xy_l569_569101


namespace max_investment_plan_l569_569653

noncomputable def PlanA_final_amount : ℝ :=
  let A1 := 30000 * (1 + 0.035)
  let A2 := A1 * (1 + 0.039)
  let A3 := A2 * (1 + 0.043)
  A3 * (1 + 0.047)

noncomputable def PlanB_final_amount : ℝ :=
  25000 * (1 + 0.025) ^ 10

noncomputable def PlanC_final_amount : ℝ :=
  let A := 20000 * (1 + 0.01) ^ 8
  A * (1 + 0.06) ^ 2.5

theorem max_investment_plan : PlanA_final_amount > PlanB_final_amount ∧ PlanA_final_amount > PlanC_final_amount :=
by
  sorry

end max_investment_plan_l569_569653


namespace max_min_values_on_interval_l569_569198

noncomputable def func (x : ℝ) : ℝ := -x^2 + 2

theorem max_min_values_on_interval :
  ∃ (max_val min_val : ℝ),
    max_val = 2 ∧ min_val = -7 ∧
    (∀ x ∈ Icc (-1 : ℝ) 3, func x ≤ max_val) ∧
    (∀ x ∈ Icc (-1 : ℝ) 3, func x ≥ min_val) :=
begin
  use [2, -7],
  split, 
  { refl },
  split,
  { refl },
  split,
  { intros x hx,
    sorry },
  { intros x hx,
    sorry },
end

end max_min_values_on_interval_l569_569198


namespace find_ratio_l569_569050

variable {f : ℝ → ℝ}
variable (a : ℕ → ℝ)

-- Conditions
def derivative (x : ℝ) : Prop := deriv f x = 2 + Real.sin x
def initial_condition : Prop := f 0 = -1
def arithmetic_sequence (h : ℝ) : Prop := ∀ n, a (n + 1) = a n + h
def sum_condition (b : ℕ) : Prop :=
  f (a (b + 1)) + f (a (b + 2)) + f (a (b + 3)) = 3 * Real.pi

-- Theorem to Prove
theorem find_ratio (h : ℝ) (hb : 4 * h = Real.pi) (a_1 : ℝ) (a_2 : ℝ):
  (derivative f) → initial_condition f → arithmetic_sequence a (h / 4) → 
  sum_condition a 2 → (a 2016) / a 2 = 2015 :=
by
  sorry

end find_ratio_l569_569050


namespace solve_quadratic_equation_l569_569852

theorem solve_quadratic_equation :
  ∀ (x : ℝ), 2 * x^2 + 6 * x - 1 = 0 ↔
    (x = -3 / 2 + sqrt 11 / 2) ∨ (x = -3 / 2 - sqrt 11 / 2) := by
  sorry

end solve_quadratic_equation_l569_569852


namespace divide_set_no_convex_l569_569039

def is_convex (a : list ℕ) : Prop :=
∀ k, 1 < k ∧ k < a.length → a.nth k ≤ (a.nth (k - 1) + a.nth (k + 1)) / 2

def contains_convex_subset (X : finset ℕ) (n : ℕ) : Prop :=
∃ (a : list ℕ), a.length = n ∧ list.sorted (<) a ∧ is_convex a

theorem divide_set_no_convex (n : ℕ) (h : n ≥ 3) :
  ∃ (A B : finset ℕ), A ∩ B = ∅ ∧ A ∪ B = (finset.range (n^2 - n + 1).erase 0) ∧ 
    ¬ contains_convex_subset A n ∧ ¬ contains_convex_subset B n := sorry

end divide_set_no_convex_l569_569039


namespace students_registered_for_course_l569_569859

theorem students_registered_for_course :
  ∀ (students_present_yesterday students_absent_today: ℕ),
    students_present_yesterday = 70 →
    students_absent_today = 30 →
    let students_attended_today := 0.9 * 2 * students_present_yesterday in
    students_registered = students_attended_today + students_absent_today →
    students_registered = 156 :=
by
  intros students_present_yesterday students_absent_today h1 h2 h3
  rw [h1, h2, h3]
  sorry

end students_registered_for_course_l569_569859


namespace distribute_balls_into_boxes_is_14_l569_569073

-- Step: Adding a noncomputable declaration only if necessary
noncomputable def distribute_balls_in_boxes : ℕ :=
  let four := 4
  let three := 3
  -- Stirling numbers of the second kind
  let S : ℕ × ℕ → ℕ
      | (n, k) := if k <= n then nat.stirlingS2 n k else 0
  S (four, 1) + S (four, 2) + S (four, 3)

-- Statement to prove that the solution is equal to 14
theorem distribute_balls_into_boxes_is_14 :
  distribute_balls_in_boxes = 14 :=
sorry

end distribute_balls_into_boxes_is_14_l569_569073


namespace median_of_set_1_2_3_1_5_is_2_l569_569337

theorem median_of_set_1_2_3_1_5_is_2:
  let s := [1, 2, 3, 1, 5] in
  let ordered_s := list.sort (<=) s in
  list.nth ordered_s 2 = some 2 :=
by 
  sorry

end median_of_set_1_2_3_1_5_is_2_l569_569337


namespace basketball_game_l569_569778

/-- Given the conditions of the basketball game:
  * a, ar, ar^2, ar^3 form the Dragons' scores
  * b, b + d, b + 2d, b + 3d form the Lions' scores
  * The game was tied at halftime: a + ar = b + (b + d)
  * The Dragons won by three points at the end: a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3
  * Neither team scored more than 100 points
Prove that the total number of points scored by the two teams in the first half is 30.
-/
theorem basketball_game (a r b d : ℕ) (h1 : a + a * r = b + (b + d))
  (h2 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (h3 : a * (1 + r + r^2 + r^3) < 100)
  (h4 : 4 * b + 6 * d < 100) :
  a + a * r + b + (b + d) = 30 :=
by
  sorry

end basketball_game_l569_569778


namespace max_prime_dinosaur_eq_8025_l569_569952

def is_prime_dinosaur (n : ℕ) : Prop :=
  n ≥ 2007 ∧ ∀ m k : ℕ, m ≥ 2007 → k ≥ 2007 → m + k = n → m = n ∨ k = n

theorem max_prime_dinosaur_eq_8025 : ∃ N, is_prime_dinosaur N ∧ ∀ n : ℕ, is_prime_dinosaur n → n ≤ 8025 :=
by
  existsi 8025
  split
  {
    sorry -- This part assumes proving 8025 is a prime dinosaur
  }
  intros n h
  sorry -- This part assumes proving that no prime dinosaur can be larger than 8025

end max_prime_dinosaur_eq_8025_l569_569952


namespace problem_statement_l569_569165

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 4)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x)

theorem problem_statement :
  (f ≠ (λ x, cos (2 * x + π / 4))) ∧
  (∃ c, c = -π / 8 ∧ f c = sin (-π / 2) = -1) ∧
  (f ≠ (λ x, sin (2 * (x - π / 4)))) ∧
  (∃ α, 0 < α ∧ α < π ∧ ∀ x, f (x + α) = f (x + 3 * α)) := by
  sorry

end problem_statement_l569_569165


namespace proof1_proof2_l569_569289

noncomputable def problem1 : Real := 
  Real.sqrt 16 + Real.cbrt 8 - Real.sqrt ((-5) ^ 2)

theorem proof1 : problem1 = 1 := by
  sorry

noncomputable def problem2 : Real :=
  (-2) ^ 3 + abs (1 - Real.sqrt 2) * (-1) ^ 2023 - Real.cbrt 125

theorem proof2 : problem2 = -12 - Real.sqrt 2 := by
  sorry

end proof1_proof2_l569_569289


namespace Juvy_chives_l569_569815

def garden_rows : ℕ := 50
def plants_per_row : ℕ := 15

def parsley_rows : ℕ := 5
def rosemary_rows : ℕ := 7
def mint_rows : ℕ := 10
def thyme_rows : ℕ := 12

def other_herbs_rows : ℕ := parsley_rows + rosemary_rows + mint_rows + thyme_rows

def chives_rows : ℕ := garden_rows - other_herbs_rows
def total_chives : ℕ := chives_rows * plants_per_row

theorem Juvy_chives (gr pr hr mr tr chr : ℕ) :
  chr - (pr + hr + mr + tr) * plants_per_row = 240 :=
by
  let gr := garden_rows
  let pr := parsley_rows
  let hr := rosemary_rows
  let mr := mint_rows
  let tr := thyme_rows
  let chr := total_chives
  calc
    chr = 240 : sorry

end Juvy_chives_l569_569815


namespace eccentricity_of_hyperbola_l569_569751

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
(hyper_eq : ∀ x y, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)
(asymp_intersect_angle : ∀ (A M N : ℝ × ℝ), A = (a, 0) → 
circle_center_radius A b → 
(intersecting_asymptote (A, M, N) 60)) 
: ℝ :=
  let c := real.sqrt (a ^ 2 + b ^ 2) in
  c / a

theorem eccentricity_of_hyperbola (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0)
  (hyper_eq : ∀ x y, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)
  (asymp_intersect_angle : ∀ (A M N : ℝ × ℝ), A = (a, 0) → 
  circle_center_radius A b → 
  (intersecting_asymptote (A, M, N) 60)) :
  hyperbola_eccentricity a b h₁ h₂ hyper_eq asymp_intersect_angle = 2 * real.sqrt 3 / 3 :=
sorry

end eccentricity_of_hyperbola_l569_569751


namespace statement_A_statement_B_statement_C_statement_D_l569_569756

variable {a b c A B C : ℝ}
variable {π : ℝ} [pi_pos : 0 < π] -- Assume π is positive
variable {g_AB : 0 < A ∧ A < π}
variable {g_AB : 0 < B ∧ B < π}
variable {g_AB : 0 < C ∧ C < π}
variable {t_ABC : A + B + C = π}

def triangle (A B C : ℝ) (a b c : ℝ) : Prop := 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π

theorem statement_A (ht : triangle A B C a b c) : 
  (sin A * sin B > sin C ^ 2) → (C < π / 3) :=
sorry

theorem statement_B (ht : triangle A B C a b c) : 
  (sin A + sin B > 2 * sin C) → (C < π / 3) :=
sorry

theorem statement_C (ht : triangle A B C a b c): 
  ¬(C > π / 2 ∧ a ^ 4 + b ^ 4 = c ^ 4) :=
sorry

theorem statement_D (ht : triangle A B C a b c) : 
  (C < π / 2) → ∃ a b c : ℝ, (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a ^ 2 + b ^ 2 > c ^ 2) ∧ (1 / a + 1 / b < 2 / c) :=
sorry

end statement_A_statement_B_statement_C_statement_D_l569_569756


namespace problem_correct_statement_l569_569578

-- Define conditions
def P1 : Prop := ∀ lines L1 L2, L1 ∩ L2 ∨ (L1 ∥ L2) ∧ (¬(L1 ∥ L2) → L1 ∩ L2 ≠ ∅)
def P2 : Prop := ∀ (p : Point) L, ∃! (L' : Line), L' ∥ L ∧ p ∉ L'
def P3 : Prop := ∀ (a b c : Line), a ∥ b ∧ b ∥ c → a ∥ c

-- Define the correct statement
def Pcorrect : Prop := P3

-- State the theorem to prove
theorem problem_correct_statement : Pcorrect = P3 :=
by
  sorry

end problem_correct_statement_l569_569578


namespace discount_limit_l569_569605

theorem discount_limit {cost_price selling_price : ℕ} (x : ℚ)
  (h1: cost_price = 100)
  (h2: selling_price = 150)
  (h3: ∃ p : ℚ, p = 1.2 * cost_price) : selling_price * (x / 10) - cost_price ≥ 0.2 * cost_price ↔ x ≤ 8 :=
by {
  sorry
}

end discount_limit_l569_569605


namespace tan_ratio_l569_569703

theorem tan_ratio (α β : ℝ) (h : Real.sin (2 * α) = 3 * Real.sin (2 * β)) :
  (Real.tan (α - β) / Real.tan (α + β)) = 1 / 2 :=
sorry

end tan_ratio_l569_569703


namespace max_value_f_l569_569739

def f (a x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_f (a : ℝ) (h : a ≠ 0) :
  (a > 0 → ∃ x, (f a x) = 4 * a * Real.exp (-2)) ∧
  (a < 0 → ∃ x, (f a x) = 0) :=
by
  sorry

end max_value_f_l569_569739


namespace Jane_Hector_meet_point_C_l569_569806

theorem Jane_Hector_meet_point_C (s t : ℝ) (h_start : ℝ) (j_start : ℝ) (loop_length : ℝ) 
  (h_speed : ℝ) (j_speed : ℝ) (h_dest : ℝ) (j_dest : ℝ)
  (h_speed_eq : h_speed = s) (j_speed_eq : j_speed = 3 * s) (loop_len_eq : loop_length = 30)
  (start_point_eq : h_start = 0 ∧ j_start = 0)
  (opposite_directions : h_dest + j_dest = loop_length)
  (meet_time_eq : t = 15 / (2 * s)) :
  h_dest = 7.5 ∧ j_dest = 22.5 → (h_dest = 7.5 ∧ j_dest = 22.5) :=
by
  sorry

end Jane_Hector_meet_point_C_l569_569806


namespace abs_inequality_example_l569_569682

theorem abs_inequality_example (x : ℝ) : abs (5 - x) < 6 ↔ -1 < x ∧ x < 11 :=
by 
  sorry

end abs_inequality_example_l569_569682


namespace probability_empty_mailbox_l569_569595

def num_greeting_cards : ℕ := 4
def num_mailboxes : ℕ := 3
def probability_at_least_one_empty_mailbox : ℚ := 5/9

theorem probability_empty_mailbox :
  ∃ p : ℚ, p = probability_at_least_one_empty_mailbox :=
begin
  -- Prove that at least one mailbox is empty
  sorry
end

end probability_empty_mailbox_l569_569595


namespace complete_square_solution_l569_569930

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end complete_square_solution_l569_569930


namespace xiaoLiangComprehensiveScore_l569_569603

-- Define the scores for the three aspects
def contentScore : ℝ := 88
def deliveryAbilityScore : ℝ := 95
def effectivenessScore : ℝ := 90

-- Define the weights for the three aspects
def contentWeight : ℝ := 0.5
def deliveryAbilityWeight : ℝ := 0.4
def effectivenessWeight : ℝ := 0.1

-- Define the comprehensive score
def comprehensiveScore : ℝ :=
  (contentScore * contentWeight) +
  (deliveryAbilityScore * deliveryAbilityWeight) +
  (effectivenessScore * effectivenessWeight)

-- The theorem stating that the comprehensive score equals 91
theorem xiaoLiangComprehensiveScore : comprehensiveScore = 91 := by
  -- proof here (omitted)
  sorry

end xiaoLiangComprehensiveScore_l569_569603


namespace grain_milling_necessary_pounds_l569_569071

theorem grain_milling_necessary_pounds (x : ℝ) (h : 0.90 * x = 100) : x = 111 + 1 / 9 := 
by
  sorry

end grain_milling_necessary_pounds_l569_569071


namespace f_zero_f_pos_f_decreasing_solve_inequality_l569_569128

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul_add (m n : ℝ) : f m * f n = f (m + n)
axiom f_pos_neg (x : ℝ) : x < 0 → 1 < f x

theorem f_zero : f 0 = 1 :=
sorry

theorem f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1 :=
sorry

theorem f_decreasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem solve_inequality (a x : ℝ) :
  f (x^2 - 3 * a * x + 1) * f (-3 * x + 6 * a + 1) ≥ 1 ↔
  (a > 1/3 ∧ 2 ≤ x ∧ x ≤ 3 * a + 1) ∨
  (a = 1/3 ∧ x = 2) ∨
  (a < 1/3 ∧ 3 * a + 1 ≤ x ∧ x ≤ 2) :=
sorry

end f_zero_f_pos_f_decreasing_solve_inequality_l569_569128


namespace geom_series_sum_4_eq_15_l569_569732

variable {a1 : ℕ} -- Assume a1 is a natural number for simplicity.

theorem geom_series_sum_4_eq_15 (q : ℕ) (HSn : ∀ n : ℕ, q ≠ 1 → S_n n = a1 * (1 - q^n) / (1 - q))
  (common_ratio : q = 2) :
  (S_4 : ℝ) = 15 * a1 := 
by
  sorry

end geom_series_sum_4_eq_15_l569_569732


namespace find_x_in_terms_of_abc_l569_569406

variable {x y z a b c : ℝ}

theorem find_x_in_terms_of_abc
  (h1 : xy / (x + y + 1) = a)
  (h2 : xz / (x + z + 1) = b)
  (h3 : yz / (y + z + 1) = c) :
  x = 2 * a * b * c / (a * b + a * c - b * c) := 
sorry

end find_x_in_terms_of_abc_l569_569406


namespace Mr_A_loss_l569_569970

example (initial_value : ℕ) (sale_loss_percent : ℚ) (repurchase_gain_percent : ℚ)
        (final_loss : ℕ) : Prop :=
  let sale_price := initial_value * (1 - sale_loss_percent) in
  let repurchase_price := sale_price * (1 + repurchase_gain_percent) in
  final_loss = repurchase_price - initial_value

theorem Mr_A_loss (initial_value : ℕ) (sale_loss_percent : ℚ) (repurchase_gain_percent : ℚ)
                  (final_loss : ℕ) (h_initial_value : initial_value = 12000)
                  (h_sale_loss_percent : sale_loss_percent = 0.10)
                  (h_repurchase_gain_percent : repurchase_gain_percent = 0.15)
                  (h_final_loss : final_loss = 420) :
  final_loss = (initial_value * (1 - sale_loss_percent)) * (1 + repurchase_gain_percent) - initial_value := 
by
  sorry

end Mr_A_loss_l569_569970


namespace contains_perfect_square_l569_569293

open Set

theorem contains_perfect_square (A : Set ℕ) (hA1 : A ⊆ {x | x ≤ 169})
  (hA2 : A.card = 84) (hA3 : ∀ {x y : ℕ}, x ∈ A → y ∈ A → x + y = 169 → false) :
  ∃ (z ∈ A), ∃ (n : ℕ), n ^ 2 = z :=
sorry

end contains_perfect_square_l569_569293


namespace find_2005th_nonincreasing_number_l569_569988

theorem find_2005th_nonincreasing_number :
  ∃ n : ℕ, n = 864100 ∧ ∃ l : List ℕ, l.length = 2005 ∧ all_digits_nonincreasing l ∧
  l.nth 2004 = some 864100 :=
by
  sorry

end find_2005th_nonincreasing_number_l569_569988


namespace books_left_proof_l569_569841

variable (initial : ℕ) (given_away : ℕ) (sold : ℕ)

theorem books_left_proof : initial = 108 → given_away = 35 → sold = 11 → initial - given_away - sold = 62 :=
by
  intros h_initial h_given_away h_sold
  rw [h_initial, h_given_away, h_sold]
  exact rfl

end books_left_proof_l569_569841


namespace binary_multiplication_binary_result_l569_569012

-- Definitions for binary numbers
def bin_11011 : ℕ := 27 -- 11011 in binary is 27 in decimal
def bin_101 : ℕ := 5 -- 101 in binary is 5 in decimal

-- Theorem statement to prove the product of two binary numbers
theorem binary_multiplication : (bin_11011 * bin_101) = 135 := by
  sorry

-- Convert the result back to binary, expected to be 10000111
theorem binary_result : 135 = 8 * 16 + 7 := by
  sorry

end binary_multiplication_binary_result_l569_569012


namespace james_main_game_time_l569_569115

-- Define the download time in minutes
def download_time := 10

-- Define the installation time as half the download time
def installation_time := download_time / 2

-- Define the combined time for download and installation
def combined_time := download_time + installation_time

-- Define the tutorial time as triple the combined time
def tutorial_time := combined_time * 3

-- Define the total time as the combined time plus the tutorial time
def total_time := combined_time + tutorial_time

-- Statement of the problem to prove
theorem james_main_game_time : total_time = 60 := by
  sorry

end james_main_game_time_l569_569115


namespace period_length_divisor_l569_569494

theorem period_length_divisor (p d : ℕ) (hp_prime : Nat.Prime p) (hd_period : ∀ n : ℕ, n ≥ 1 → 10^n % p = 1 ↔ n = d) :
  d ∣ (p - 1) :=
sorry

end period_length_divisor_l569_569494


namespace triangle_inequality_l569_569465

-- Define the triangle
variables {A B C P : Type} [EuclideanGeometry A B C P]

-- Define the incenter
def incenter (A B C : Type) [EuclideanGeometry A B C] : Type := sorry

-- Define the main theorem
theorem triangle_inequality (A B C P : Type) [EuclideanGeometry A B C P]
  (I : incenter A B C)
  (h : ∠PBA + ∠PCA ≥ ∠PBC + ∠PCB) :
  dist A P ≥ dist A I ∧ (dist A P = dist A I ↔ P = I) := 
sorry

end triangle_inequality_l569_569465


namespace mutually_exclusive_not_opposite_l569_569421
-- Import the necessary library

-- Define the scenario and relevant definitions
section Balls

variable (redBalls greenBalls : ℕ)
variable (drawn : Finset α) -- finite set of drawn balls

-- Assume there are 10 red balls and 10 green balls
def balls_in_bag : Prop := redBalls = 10 ∧ greenBalls = 10

-- Define events
def at_least_one_red (drawn : Finset α) : Prop :=
  ∃ x ∈ drawn, x ∈ redBalls

def at_least_one_green (drawn : Finset α) : Prop :=
  ∃ x ∈ drawn, x ∈ greenBalls

def exactly_one_red (drawn : Finset α) : Prop :=
  card (drawn ∩ redBalls) = 1

def exactly_two_green (drawn : Finset α) : Prop :=
  card (drawn ∩ greenBalls) = 2

-- Define mutually exclusive
def mutually_exclusive (P Q : Prop) : Prop :=
  P → ¬ Q

-- Define not opposite
def not_opposite (P Q : Prop) : Prop :=
  ¬ (P ↔ ¬ Q)

-- The statement to prove
theorem mutually_exclusive_not_opposite :
  balls_in_bag redBalls greenBalls →
  ∀ (drawn : Finset α), card drawn = 2 →
  mutually_exclusive (exactly_one_red drawn) (exactly_two_green drawn) ∧
  not_opposite (exactly_one_red drawn) (exactly_two_green drawn) :=
by -- Proof goes here
  sorry

end Balls

end mutually_exclusive_not_opposite_l569_569421


namespace no_integer_solutions_l569_569506

theorem no_integer_solutions (x y : ℤ) : x^3 + 3 ≠ 4 * y * (y + 1) :=
sorry

end no_integer_solutions_l569_569506


namespace find_x_l569_569925

theorem find_x :
  ∀ (x y z w : ℕ), 
    x = y + 5 →
    y = z + 10 →
    z = w + 20 →
    w = 80 →
    x = 115 :=
by
  intros x y z w h1 h2 h3 h4
  sorry

end find_x_l569_569925


namespace horizontal_segment_if_integral_distance_l569_569153

noncomputable def f : ℤ → ℤ := sorry  -- the polynomial function

theorem horizontal_segment_if_integral_distance
  (a b : ℤ)
  (h_poly : ∀ x : ℤ, f x ∈ ℤ)
  (h_dist : ∃ d : ℤ, d = int.sqrt ((b - a)^2 + (f b - f a)^2)) :
  f a = f b :=
by
   sorry

end horizontal_segment_if_integral_distance_l569_569153


namespace trig_identity_solution_l569_569582

theorem trig_identity_solution (x : ℝ) (n : ℤ) :
  (sin x * cos (2 * x) + cos x * cos (4 * x) = sin (π / 4 + 2 * x) * sin (π / 4 - 3 * x)) →
  (x = (π / 12) * (4 * n - 1)) := 
by
  sorry

end trig_identity_solution_l569_569582


namespace phone_call_probability_within_four_rings_l569_569428

variables (P_A P_B P_C P_D : ℝ)

-- Assuming given probabilities
def probabilities_given : Prop :=
  P_A = 0.1 ∧ P_B = 0.3 ∧ P_C = 0.4 ∧ P_D = 0.1

theorem phone_call_probability_within_four_rings (h : probabilities_given P_A P_B P_C P_D) :
  P_A + P_B + P_C + P_D = 0.9 :=
sorry

end phone_call_probability_within_four_rings_l569_569428


namespace unit_vector_is_correct_l569_569100

-- Define vector components
def a : ℝ × ℝ × ℝ := (3, 0, 4)

-- Define the expected unit vector
def e_exp : ℝ × ℝ × ℝ := (3/5, 0, 4/5)

-- Define the magnitude of vector a
def mag_a : ℝ := real.sqrt (3^2 + 0^2 + 4^2)

-- Define the calculated unit vector for a
def e_calc : ℝ × ℝ × ℝ := (3 / mag_a, 0, 4 / mag_a)

-- The proof statement: the calculated unit vector is equal to the expected one
theorem unit_vector_is_correct : e_calc = e_exp := sorry

end unit_vector_is_correct_l569_569100


namespace complex_number_quadrant_l569_569534

theorem complex_number_quadrant :
  let z : ℂ := (2 - I) / (3 + 4 * I) in
  z.re > 0 ∧ z.im < 0 :=
by
  let z : ℂ := (2 - I) / (3 + 4 * I)
  have z_eq : z = (10 / 25) + (-11 / 25) * I := by sorry
  -- Use the given result from the solution
  have z_re_pos : z.re = 10 / 25 := by sorry
  have z_im_neg : z.im = -11 / 25 := by sorry
  exact ⟨z_re_pos, z_im_neg⟩ -- The point (10/25, -11/25) is in the fourth quadrant

end complex_number_quadrant_l569_569534


namespace trajectory_and_product_conditions_l569_569085

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem trajectory_and_product_conditions :
  ∀ (A B C M N T : (ℝ × ℝ))
  (perimeter_ABC : ℝ)
  (O : (ℝ × ℝ)),
  B = (-(3 * real.sqrt 2) / 2, 0) →
  C = ((3 * real.sqrt 2) / 2, 0) →
  perimeter_ABC = 6 + 3 * real.sqrt 2 →
  O = midpoint B C →
  (T.1 = 3 * A.1 / 4 ∧ T.2 = 3 * A.2 / 4) →
  (0 < M.1 ∧ M.1 < 1) →
  N = (N.1, 0) →
  (∃ (x y : ℝ), x ^ 2 + 2 * y ^ 2 = 1 ∧ y ≠ 0) →
  ((∃ (P Q : (ℝ × ℝ)), (Q.1 * y - Q.2 = 0) → 
    (N.2 * Q.1 - M.2 * T.1 = 0)) →
   ((A ≠ P ) ∧ ((M.1 * P.1 + M.2 * P.2 = R.1) ∧ 
    (M.1 * R.1 + M.2 * R.2 = P.2) )) → 
    ((P ≠ M) ∧ (2 * P.2 * R.2 - M.2 = 0)) → 
    P = (P.1, - P.2)) →
  (∃ mn_product : ℝ, mn_product = M.1 * N.1 ∧ mn_product = 1) :=
by
  intros A B C M N T perimeter_ABC O
  intros hB hC h_perimeter hO hT hM hN h_trajectory
  intros h_intersections h_zero_product
  sorry

end trajectory_and_product_conditions_l569_569085


namespace arithmetic_mean_is_30_geometric_mean_is_approx_26_768_l569_569918

-- Define the four numbers
def x1 : ℝ := 16
def x2 : ℝ := 24
def x3 : ℝ := 32
def x4 : ℝ := 48

-- Define the arithmetic mean
def arithmetic_mean (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

-- Define the geometric mean
noncomputable def geometric_mean (a b c d : ℝ) : ℝ := (a * b * c * d) ^ (1 / 4 : ℝ)

-- State the theorem for the arithmetic mean
theorem arithmetic_mean_is_30 : arithmetic_mean x1 x2 x3 x4 = 30 := 
sorry

-- State the theorem for the geometric mean
theorem geometric_mean_is_approx_26_768 : abs (geometric_mean x1 x2 x3 x4 - 26.768) < 0.001 := 
sorry

end arithmetic_mean_is_30_geometric_mean_is_approx_26_768_l569_569918


namespace enclosed_midpoints_area_l569_569177

-- Define the square with side length 3.
def square_side_length : ℝ := 3

-- Define the set T as all line segments of length 3 with endpoints on adjacent sides.
def point_on_square := { p : ℝ × ℝ // 
  (p.1 = 0 ∨ p.1 = square_side_length) ∨
  (p.2 = 0 ∨ p.2 = square_side_length) 
}
def length_3_segment (p1 p2 : point_on_square) : Prop :=
  (p1.val.1 - p2.val.1)^2 + (p1.val.2 - p2.val.2)^2 = 9

def set_T : set (point_on_square × point_on_square) :=
  { s | length_3_segment s.1 s.2 }

-- Define the condition that the area enclosed by the midpoints has 100m = 258.
theorem enclosed_midpoints_area : 
  100 * (9 - 9 * real.pi / 16).floor = 258 :=
by 
  sorry

end enclosed_midpoints_area_l569_569177


namespace solve_system_l569_569512

def equations_solution (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧
  (x + y) * (y + z) + (y + z) * (z + x) + (z + x) * (x + y) = 1 ∧
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = -6

def solution_set : set (ℝ × ℝ × ℝ) :=
  {(0, 3, -1), (0, -1, 3), (3, 0, -1), (3, -1, 0), (-1, 0, 3), (-1, 3, 0)}

theorem solve_system (x y z : ℝ) : equations_solution x y z ↔ (x, y, z) ∈ solution_set := 
  sorry

end solve_system_l569_569512


namespace tangent_line_equation_max_value_on_interval_l569_569370

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x + 1)

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 1 + x * Real.exp x

noncomputable def g (x a : ℝ) : ℝ := f x - a * Real.exp x - x

theorem tangent_line_equation :
  let x := 0 in
  let f0 := f 0 in
  let fp0 := f' 0 in
  (f0 = 0) → (fp0 = 2) → True :=
by
  intros
  sorry

theorem max_value_on_interval (a : ℝ) :
  (∀ x ∈ (Set.Icc 1 2), a ≥ (2*Real.exp 1 - 1)/(Real.exp 1 - 1) → g x a = (1 - a) * Real.exp 1) ∧
  (∀ x ∈ (Set.Icc 1 2), a <  (2*Real.exp 1 - 1)/(Real.exp 1 - 1) → g x a = (2 - a) * Real.exp 2) :=
by
  intros
  sorry

end tangent_line_equation_max_value_on_interval_l569_569370


namespace angle_a_a_plus_2b_l569_569048

-- Condition definitions
variables {a b : EuclideanSpace ℝ ℕ}
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 1)
variables (angle_ab : real.angle a b = real.pi / 3)

-- Problem statement to prove
theorem angle_a_a_plus_2b : real.angle a (a + 2 • b) = real.pi / 6 :=
sorry

end angle_a_a_plus_2b_l569_569048


namespace sum_of_subtended_angles_of_AD_l569_569437

-- Defining the setup for the isosceles triangle
variables (α : ℝ) (n : ℕ) (n_pos : 1 < n)
variables (A B C D : Type) [h: IsIsoscelesTriangle A B C α] -- Custom defined type class for isosceles triangle

-- Condition that point D is on AB such that AD = AB / n
variable (AD_eq : Dist A D = Dist A B / n)

-- Defining the points K_i that divide BC into n equal parts
variables (K : Fin (n - 1) → Type) [h_divide: ∀ i, DividesBC K i n]

-- The theorem statement translated to Lean 4
theorem sum_of_subtended_angles_of_AD (sum_angles_eq_half_alpha: 
  ∑ i : Fin (n - 1), (angle_subtended_by AD (K i) A) = α / 2):
  sorry

end sum_of_subtended_angles_of_AD_l569_569437


namespace smaller_octagon_area_fraction_l569_569615

variable {A B C D E F G H P Q R S T U V W O : Type}
variable [octagon : RegularOctagon A B C D E F G H] 
variable [midpoints : Midpoints A B C D E F G H P Q R S T U V W]

theorem smaller_octagon_area_fraction : 
  area (smaller_octagon P Q R S T U V W) = 
  (1/2) * area (larger_octagon A B C D E F G H) := 
sorry

end smaller_octagon_area_fraction_l569_569615


namespace red_sea_glass_pieces_l569_569639

theorem red_sea_glass_pieces (R : ℕ) 
    (h_bl : ∃ g : ℕ, g = 12) 
    (h_rose_red : ∃ r_b : ℕ, r_b = 9)
    (h_rose_blue : ∃ b : ℕ, b = 11) 
    (h_dorothy_red : 2 * (R + 9) + 3 * 11 = 57) : R = 3 :=
  by
    sorry

end red_sea_glass_pieces_l569_569639


namespace correct_number_of_conclusions_l569_569021

def y (x : ℝ) := -5 * x + 1

def conclusion1 := y (-1) = 5
def conclusion2 := ∃ x1 x2 x3 : ℝ, y x1 > 0 ∧ y x2 > 0 ∧ y (x3) < 0 ∧ (x1 < 0) ∧ (x2 > 0) ∧ (x3 < x2)
def conclusion3 := ∀ x : ℝ, x > 1 → y x < 0
def conclusion4 := ∀ x1 x2 : ℝ, x1 < x2 → y x1 < y x2

-- We want to prove that exactly 2 of these conclusions are correct
theorem correct_number_of_conclusions : (¬ conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬ conclusion4) :=
by
  sorry

end correct_number_of_conclusions_l569_569021


namespace part1_part2_l569_569361
-- Import the entire Mathlib to ensure all necessary functions and constants are included

-- Define the function f(x)
def f (ω x : ℝ) : ℝ := sin(ω * x) ^ 2 + sqrt 3 * cos(ω * x) * cos (π / 2 - ω * x)

-- Lean 4 statement for the first part: Prove that f(π/6) = 1
theorem part1 (ω > 0) (h_ω: ω = 1) : f ω (π / 6) = 1 :=
by
  -- Additional necessary setup, particularly indicating we skip proof details
  sorry

-- Define a new function for the second part
def g (k x : ℝ) : ℝ := f 1 (k * x + π / 12)

-- Lean 4 statement for the second part: Prove the range of k
theorem part2 : (0 < k ∧ k ≤ 3 / 4) ↔ ∀ x : ℝ, -π / 6 ≤ x ∧ x ≤ π / 3 → monotone (g k) :=
by
  -- Additional necessary setup, particularly indicating we skip proof details
  sorry

end part1_part2_l569_569361


namespace complex_on_ellipse_real_fraction_l569_569908

theorem complex_on_ellipse_real_fraction (x y : ℝ) (z : ℂ) 
    (h1 : z = x + y * complex.I)
    (h2 : x^2 / 9 + y^2 / 16 = 1)
    (h3 : (z - 1 - complex.I) / (z - complex.I) ∈ ℝ) : 
    z = (3 * real.sqrt 15 / 4) + complex.I ∨ z = - (3 * real.sqrt 15 / 4) + complex.I :=
by sorry

end complex_on_ellipse_real_fraction_l569_569908


namespace angle_BAC_is_30_l569_569977

-- Definitions related to the problem
variable (A B C M : Type) [Point A] [Point B] [Point C] [Point M]
variable [Triangle A B C]
variable [PerpendicularBisector M A C]
variable (BAC MCB ABC MBC : Angle)
variable [AngleEq BAC MCB]
variable [AngleSumEq ABC MBC 180]

-- Theorem stating the problem
theorem angle_BAC_is_30 (h1 : BAC = MCB) 
                        (h2 : ABC + MBC = 180) 
                        (h3 : isPerpendicularBisector M A C) 
                        (h4 : isAcuteTriangle A B C) : 
                        BAC = 30 := sorry

end angle_BAC_is_30_l569_569977


namespace find_coef_of_quadratic_l569_569487

-- Define the problem conditions
def solutions_of_abs_eq : Set ℤ := {x | abs (x - 3) = 4}

-- Given that the solutions are 7 and -1
def paul_solutions : Set ℤ := {7, -1}

-- The problem translates to proving the equivalence of two sets
def equivalent_equation_solutions (d e : ℤ) : Prop :=
  ∀ x, x ∈ solutions_of_abs_eq ↔ x^2 + d * x + e = 0

theorem find_coef_of_quadratic :
  equivalent_equation_solutions (-6) (-7) :=
by
  sorry

end find_coef_of_quadratic_l569_569487


namespace george_will_receive_each_bill_l569_569024

variables (initial_bills spent_bills remaining_bills total_amount per_bill : ℝ)

def george_birthday_problem : Prop :=
  initial_bills = 10 ∧
  spent_bills = 0.20 * initial_bills ∧
  remaining_bills = initial_bills - spent_bills ∧
  total_amount = 12 ∧
  per_bill = total_amount / remaining_bills ∧
  per_bill = 1.50

theorem george_will_receive_each_bill :
  george_birthday_problem initial_bills spent_bills remaining_bills total_amount per_bill :=
by {
  -- Introduction of variables according to the conditions given in the problem
  unfold george_birthday_problem,
  -- Assuming the initial conditions
  split, repeat {split},
  -- Initial number of bills is 10
  exact rfl,
  -- Number of bills spent is 20% of 10
  norm_num,
  -- Remaining bills is initial bills minus spent bills
  norm_num,
  -- Total amount received is $12
  exact rfl,
  -- Calculating the worth of each bill
  field_simp,
  -- Verifying the worth is $1.50
  norm_num,
  sorry -- To finalize the theorem without providing a constructive proof
}

end george_will_receive_each_bill_l569_569024


namespace find_a_and_increasing_function_l569_569365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a / (3^x + 1))

theorem find_a_and_increasing_function :
  (∀ x, f a (-x) = -f a x) → (a = 2 ∧ ∀ x, (f 2 x)' > 0) :=
by
  sorry

end find_a_and_increasing_function_l569_569365


namespace recurrence_relation_generating_function_evaluation_l569_569322

section
variable (a : ℕ → ℝ)

-- Given condition
axiom a_def : ∀ n, a n = (n^2 + 1) * 3^n

-- Recurrence relation theorem
theorem recurrence_relation (n : ℕ) :
  a n + (-1) * a (n + 1) + (1 / 3) * a (n + 2) + (-1 / 27) * a (n + 3) = 0 :=
  sorry

-- Generating function theorem
noncomputable def generating_function (x : ℝ) :=
  ∑ n, a n * x^n

theorem generating_function_evaluation : ∀ (x : ℝ),
  generating_function a x = (1 - 3 * x + 18 * x^2) / (1 - 9 * x + 27 * x^2 - 27 * x^3) :=
  sorry

end

end recurrence_relation_generating_function_evaluation_l569_569322


namespace total_packs_l569_569481

theorem total_packs (cards_per_person : ℕ) (cards_per_pack : ℕ) (people_count : ℕ) (cards_per_person_eq : cards_per_person = 540) (cards_per_pack_eq : cards_per_pack = 20) (people_count_eq : people_count = 4) :
  (cards_per_person / cards_per_pack) * people_count = 108 :=
by
  sorry

end total_packs_l569_569481


namespace find_g_x_f_y_l569_569122

-- Definition of the functions and conditions
variable (f g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1)

-- The theorem to prove
theorem find_g_x_f_y (x y : ℝ) : g (x + f y) = -x + y - 1 := 
sorry

end find_g_x_f_y_l569_569122


namespace correct_diagram_l569_569480

-- Definitions based on the conditions
def word : String := "KANGAROO"
def diagrams : List (String × Bool) :=
  [("Diagram A", False), ("Diagram B", False), ("Diagram C", False),
   ("Diagram D", False), ("Diagram E", True)]

-- Statement to prove that Diagram E correctly shows "KANGAROO"
theorem correct_diagram :
  ∃ d, (d.1 = "Diagram E") ∧ d.2 = True ∧ d ∈ diagrams :=
by
-- skipping the proof for now
sorry

end correct_diagram_l569_569480


namespace distance_between_pulleys_l569_569954

theorem distance_between_pulleys (R1 R2 d: ℝ) (hR1 : R1 = 14) (hR2 : R2 = 4) (hTangentDist : d = sqrt ((24^2) + ((R1 - R2)^2))) : d = 26 :=
sorry

end distance_between_pulleys_l569_569954


namespace pages_in_first_issue_l569_569564

-- Define variables for the number of pages in the issues and total pages
variables (P : ℕ) (total_pages : ℕ) (eqn : total_pages = 3 * P + 4)

-- State the theorem using the given conditions and question
theorem pages_in_first_issue (h : total_pages = 220) : P = 72 :=
by
  -- Use the given equation
  have h_eqn : total_pages = 3 * P + 4 := eqn
  sorry

end pages_in_first_issue_l569_569564


namespace problem_l569_569514

theorem problem (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (1 / 2) = 0) : 
  f (-201) = 403 :=
sorry

end problem_l569_569514


namespace find_a_5_l569_569064

theorem find_a_5 (a : ℕ → ℤ) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1)
  (h₂ : a 2 + a 4 + a 6 = 18) : a 5 = 5 := 
sorry

end find_a_5_l569_569064


namespace solve_system_of_equations_l569_569936

theorem solve_system_of_equations : 
  ∃ x y z : ℝ, 
    (x + (1/3) * y + (1/3) * z = 14) ∧
    (y + (1/4) * x + (1/4) * z = 8) ∧
    (z + (1/5) * x + (1/5) * y = 8) ∧
    x = 11 ∧ y = 4 ∧ z = 5 :=
by {
  use [11, 4, 5],
  simp,
  split,
  -- x + (1/3) * 4 + (1/3) * 5 = 14
  norm_num,
  split,
  -- 4 + (1/4) * 11 + (1/4) * 5 = 8
  norm_num,
  -- 5 + (1/5) * 11 + (1/5) * 4 = 8
  norm_num,
  -- x = 11
  split,
  refl,
  -- y = 4
  split,
  refl,
  -- z = 5
  refl,
}

end solve_system_of_equations_l569_569936


namespace exam_students_count_l569_569941

theorem exam_students_count (failed_students : ℕ) (failed_percentage : ℝ) (total_students : ℕ) 
    (h1 : failed_students = 260) 
    (h2 : failed_percentage = 0.65) 
    (h3 : (failed_percentage * total_students : ℝ) = (failed_students : ℝ)) : 
    total_students = 400 := 
by 
    sorry

end exam_students_count_l569_569941


namespace probability_properties_l569_569220

noncomputable def P1 : ℝ := 1 / 4
noncomputable def P2 : ℝ := 1 / 4
noncomputable def P3 : ℝ := 1 / 2

theorem probability_properties :
  (P1 ≠ P3) ∧
  (P1 + P2 = P3) ∧
  (P1 + P2 + P3 = 1) ∧
  (P3 = 2 * P1) ∧
  (P3 = 2 * P2) :=
by
  sorry

end probability_properties_l569_569220


namespace fraction_of_alcohol_l569_569429

theorem fraction_of_alcohol (A : ℚ) (water_volume : ℚ) (alcohol_to_water_ratio : ℚ) 
  (h1 : water_volume = 4/5) 
  (h2 : alcohol_to_water_ratio = 3/4) 
  (h3 : A / water_volume = alcohol_to_water_ratio) : 
  A = 3/5 :=
by 
  rw [h1, h2] at h3
  field_simp at h3
  sorry

end fraction_of_alcohol_l569_569429


namespace solution_to_linear_eq_l569_569579

theorem solution_to_linear_eq :
  (∀ x y : ℤ, x + 2 * y = 6 ↔ (x = 2 ∧ y = 2)) :=
by
  intro x y
  split
  {
    intro h
    have : x + 2 * y = 6 := h
    apply Classical.by_cases
    {
      intro h1 : (x = -2 ∧ y = -2)
      have : -2 + 2 * -2 = -6 := by linarith
      contradiction
    }
    {
      intro h2 : (x = 0 ∧ y = 2)
      have : 0 + 2 * 2 = 4 := by linarith
      contradiction
    }
    {
      intro h3 : (x = 3 ∧ y = 1)
      have : 3 + 2 * 1 = 5 := by linarith
      contradiction
    }
    {
      intro h4 : ¬(x = -2 ∧ y = -2) ∧ ¬(x = 0 ∧ y = 2) ∧ ¬(x = 3 ∧ y = 1)
      have : x = 2 ∧ y = 2 := by sorry
      exact this
    }
  }
  {
    intro h
    cases h with hx hy
    exact hx.symm ▸ hy.symm ▸ rfl
  }

end solution_to_linear_eq_l569_569579


namespace solve_inequality_solve_system_of_inequalities_l569_569592

-- Inequality proof problem
theorem solve_inequality (x : ℝ) (h : (2*x - 3)/3 > (3*x + 1)/6 - 1) : x > 1 := by
  sorry

-- System of inequalities proof problem
theorem solve_system_of_inequalities (x : ℝ) (h1 : x ≤ 3*x - 6) (h2 : 3*x + 1 > 2*(x - 1)) : x ≥ 3 := by
  sorry

end solve_inequality_solve_system_of_inequalities_l569_569592


namespace distinct_coloring_l569_569648

def color (a : ℕ) : ℕ :=
  let S := (a.factorization.toMultiset.filter (λ p, p ≠ 5)).sum +
           (a.factorization.toMultiset.filter (λ p, p = 5)).card +
           6 * (a.factorization.toMultiset.filter (λ p, p = 7)).card
  in S % 7

theorem distinct_coloring :
  ∀ (a : ℕ), a > 0 → (∀ i j, i ≠ j → i ∈ (Finset.range 7).erase 0 → j ∈ (Finset.range 7).erase 0 → color (i * a) ≠ color (j * a)) :=
by
  sorry

end distinct_coloring_l569_569648


namespace jill_total_tax_l569_569152

noncomputable def total_tax_percentage 
  (total_amount : ℝ) 
  (clothing_pct food_pct electronics_pct other_items_pct : ℝ) 
  (discount_clothing_pct discount_electronics_pct : ℝ) 
  (tax_clothing_pct tax_food_pct tax_electronics_pct tax_other_items_pct : ℝ) : ℝ :=
let clothing_amount := clothing_pct * total_amount in
let food_amount := food_pct * total_amount in
let electronics_amount := electronics_pct * total_amount in
let other_items_amount := other_items_pct * total_amount in
let discounted_clothing := clothing_amount * (1 - discount_clothing_pct) in
let discounted_electronics := electronics_amount * (1 - discount_electronics_pct) in
let tax_amount := discounted_clothing * tax_clothing_pct +
                  food_amount * tax_food_pct +
                  discounted_electronics * tax_electronics_pct +
                  other_items_amount * tax_other_items_pct in
(tax_amount / total_amount) * 100

theorem jill_total_tax : total_tax_percentage 
  100 0.4 0.15 0.25 0.2 0.1 0.05 0.04 0 0.06 0.08 = 4.465 := 
by 
  sorry

end jill_total_tax_l569_569152


namespace model_to_statue_scale_l569_569526

theorem model_to_statue_scale
  (statue_height_ft : ℕ)
  (model_height_in : ℕ)
  (ft_to_in : ℕ)
  (statue_height_in : ℕ)
  (scale : ℕ)
  (h1 : statue_height_ft = 120)
  (h2 : model_height_in = 6)
  (h3 : ft_to_in = 12)
  (h4 : statue_height_in = statue_height_ft * ft_to_in)
  (h5 : scale = (statue_height_in / model_height_in) / ft_to_in) : scale = 20 := 
  sorry

end model_to_statue_scale_l569_569526


namespace andrey_stamps_l569_569156

theorem andrey_stamps (x : ℤ) :
  x ≡ 1 [MOD 2] ∧
  x ≡ 1 [MOD 3] ∧
  x ≡ 3 [MOD 5] ∧
  x ≡ 7 [MOD 9] ∧
  150 < x ∧ x ≤ 300 → 
  x = 223 :=
by
  sorry

end andrey_stamps_l569_569156


namespace larger_number_225_l569_569378

theorem larger_number_225 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a - b = 120) 
  (h4 : Nat.lcm a b = 105 * Nat.gcd a b) : 
  max a b = 225 :=
by
  sorry

end larger_number_225_l569_569378


namespace cone_lateral_surface_area_l569_569354

theorem cone_lateral_surface_area (π : ℝ) (l r : ℝ) (S : ℝ) 
  (hl : l = 5) 
  (hr : r = 2) 
  (hS_formula : S = π * r * l) :
  S = 10 * π := 
  by
  rw [hl, hr, hS_formula]
  sorry

end cone_lateral_surface_area_l569_569354


namespace pentagon_AEDCB_area_l569_569614

-- Definitions based on the given conditions
def rectangle_ABCD (AB BC : ℕ) : Prop :=
AB = 12 ∧ BC = 10

def triangle_ADE (AE ED : ℕ) : Prop :=
AE = 9 ∧ ED = 6 ∧ AE * ED ≠ 0 ∧ (AE^2 + ED^2 = (AE^2 + ED^2))

def area_of_rectangle (AB BC : ℕ) : ℕ :=
AB * BC

def area_of_triangle (AE ED : ℕ) : ℕ :=
(AE * ED) / 2

-- The theorem to be proved
theorem pentagon_AEDCB_area (AB BC AE ED : ℕ) (h_rect : rectangle_ABCD AB BC) (h_tri : triangle_ADE AE ED) :
  area_of_rectangle AB BC - area_of_triangle AE ED = 93 :=
sorry

end pentagon_AEDCB_area_l569_569614


namespace total_digits_first_3003_even_l569_569657

theorem total_digits_first_3003_even :
  let n := 3003,
      last_even := n * 2,
      count_one_digit := 4,
      count_two_digit := (98 - 10) / 2 + 1,
      count_three_digit := (998 - 100) / 2 + 1,
      count_four_digit := (last_even - 1000) / 2 + 1,
      digits_one_digit := count_one_digit,
      digits_two_digit := count_two_digit * 2,
      digits_three_digit := count_three_digit * 3,
      digits_four_digit := count_four_digit * 4,
      total_digits := digits_one_digit + digits_two_digit + digits_three_digit + digits_four_digit
  in
      total_digits = 11460 := by
  /-
  n           = 3003
  last_even   = 6006
  count_one_digit   = 4
  count_two_digit   = 45
  count_three_digit = 450
  count_four_digit  = 2504
  
  digits_one_digit   = 4
  digits_two_digit   = 90
  digits_three_digit = 1350
  digits_four_digit  = 10016
  
  total_digits = 11460
  -/
  sorry

end total_digits_first_3003_even_l569_569657


namespace graph_translation_l569_569774

theorem graph_translation (f : ℝ → ℝ) (x : ℝ) (h : f 1 = -1) :
  f (x - 1) - 1 = -2 :=
by
  sorry

end graph_translation_l569_569774


namespace side_length_of_square_l569_569234

theorem side_length_of_square (A : ℝ) (h : A = 400) : ∃ s : ℝ, s * s = A ∧ s = 20 :=
by {
  use 20,
  split,
  { rw h, norm_num },
  { refl }
}

end side_length_of_square_l569_569234


namespace ratio_ap_bp_l569_569210

open Real
open Classical

noncomputable theory

variables (A B C P : Type) -- We denote the points by types
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (d : A × B → ℝ) -- Distance function

-- Conditions
def right_triangle (a b c : A × B × C) :=
  ∃ (A' B' C' : ℝ), d (A', C')^2 + d (C', B')^2 = d (A', B')^2 ∧ d (A', C') < d (A', B')

def angle_condition (A B C : A × B × C) : Prop :=
  ∃ α, α < π/4

def point_on_segment (A B P : A × B × P) : Prop :=
  ∃ (A' B' P' : ℝ), d (A', P') + d (P', B') = d (A', B')

def angle_apc_condition (A P C : A × P × C) : Prop :=
  ∃ (α β : ℝ), α = 3 * β 

def condition_cp (P C : P × C) : Prop := d (P, C) = 2

-- Theorem statement
theorem ratio_ap_bp (a b c p : A × B × C × P) (right_triangle abc)
    (angle_condition abc) (point_on_segment abp)
    (angle_apc_condition apc) (condition_cp p c)
  : ∃ (p q r : ℕ), r ≠ 0 ∧ p + q = 3 ∧ p = 3 ∧ q = 0 ∧ r = 1 ∧ p + q + r = 4 :=
begin
  sorry
end

end ratio_ap_bp_l569_569210


namespace projection_correct_l569_569315

-- Define the vectors
def v : ℝ × ℝ := (3, -1)
def u : ℝ × ℝ := (1, 2)

-- Function to compute the dot product of two vectors
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Function to compute the magnitude squared of a vector
def magnitude_sq (x : ℝ × ℝ) : ℝ :=
  x.1 * x.1 + x.2 * x.2

-- Define the projection 
def projection (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dp := dot_product v u
  let msq := magnitude_sq u
  (dp / msq * u.1, dp / msq * u.2)

-- State the theorem
theorem projection_correct : projection v u = (1/5 : ℝ, 2/5 : ℝ) :=
  sorry

end projection_correct_l569_569315


namespace odd_integer_sum_ge_two_k_squared_sub_one_l569_569291

theorem odd_integer_sum_ge_two_k_squared_sub_one {k : ℕ} (hk : k % 2 = 1) (n : ℕ → ℕ)
  (hn : ∀ i j, i < j → n i < n j) (hodd : ∀ i, n i % 2 = 1) (hlen: ∀ i, i < k → n i > 0) :
  ∑ i in finset.range k, if i % 2 = 0 then n i ^ 2 else -n i ^ 2 ≥ 2 * k ^ 2 - 1 := sorry

end odd_integer_sum_ge_two_k_squared_sub_one_l569_569291


namespace number_of_terms_geometric_seq_l569_569201

-- Given conditions
variables (a1 q : ℝ)  -- First term and common ratio of the sequence
variable  (n : ℕ)     -- Number of terms in the sequence

-- The product of the first three terms
axiom condition1 : a1^3 * q^3 = 3

-- The product of the last three terms
axiom condition2 : a1^3 * q^(3 * n - 6) = 9

-- The product of all terms
axiom condition3 : a1^n * q^(n * (n - 1) / 2) = 729

-- Proving the number of terms in the sequence
theorem number_of_terms_geometric_seq : n = 12 := by
  sorry

end number_of_terms_geometric_seq_l569_569201


namespace work_done_by_b_l569_569228

theorem work_done_by_b (x : ℝ) (h1 : (1/6) + (1/13) = (1/x)) : x = 78/7 :=
  sorry

end work_done_by_b_l569_569228


namespace total_students_registered_l569_569854

theorem total_students_registered (num_present_yesterday : ℕ) (num_absent_today : ℕ) 
  (percent_less : ℝ) (num_students : ℕ) 
  (h1 : num_present_yesterday = 70)
  (h2 : num_absent_today = 30)
  (h3 : percent_less = 0.1)
  (h4 : num_students = 156) :
  let twice_present_yesterday := 2 * num_present_yesterday in
  let reduction := percent_less * twice_present_yesterday in
  let num_present_today := twice_present_yesterday - reduction in
  num_students = num_present_today + num_absent_today :=
by
  -- Using the conditions provided to show the equivalence.
  sorry

end total_students_registered_l569_569854


namespace lattice_points_hyperbola_l569_569069

theorem lattice_points_hyperbola :
  (∑ x in finset.range(99) \ finset.range(2), 
    if (∃ y : ℤ, y * y = x * x - 1) then
      (2 * (int.sqrt (x * x - 1))) + 1
    else 0) = 9800 := 
by
  sorry

end lattice_points_hyperbola_l569_569069


namespace janet_acres_l569_569807

-- Defining the variables and conditions
variable (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ)

-- Assigning the given values to the variables
def horseFertilizer := 5
def acreFertilizer := 400
def janetSpreadRate := 4
def janetHorses := 80
def fertilizingDays := 25

-- Main theorem stating the question and proving the answer
theorem janet_acres : 
  ∀ (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ),
  horse_production = 5 → 
  acre_requirement = 400 →
  spread_rate = 4 →
  num_horses = 80 →
  days = 25 →
  (spread_rate * days = 100) := 
by
  intros
  -- Proof would be inserted here
  sorry

end janet_acres_l569_569807


namespace total_ages_l569_569549

theorem total_ages (bride_age groom_age : ℕ) (h1 : bride_age = 102) (h2 : groom_age = bride_age - 19) : bride_age + groom_age = 185 :=
by
  sorry

end total_ages_l569_569549


namespace pyramid_cone_properties_l569_569947

-- Definitions for the problem
structure Pyramid :=
  (height : ℝ)
  (ratio_BL_LS : ℕ)
  (ratio_LS_BL : ℕ)
  (is_regular : Prop)

structure Cone :=
  (volume : ℝ)

-- Helper definitions for the problem
def AS_CD_ratio (A S C D : ℝ) := S / D
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

noncomputable def find_ratio (A S C D : ℝ) (p : Pyramid) : ℝ :=
  if p.is_regular then sqrt (5 / 3) else 0

noncomputable def find_volume (p : Pyramid) : ℝ :=
  if p.is_regular then (125 * Real.pi) / sqrt 21 else 0

-- The statement to be proved
theorem pyramid_cone_properties (A S B L C D : ℝ)
  (pyr : Pyramid)
  (cone : Cone)
  (h_pyramid : pyr.height = 7)
  (ratio_cond : pyr.ratio_BL_LS = 2 ∧ pyr.ratio_LS_BL = 5 ∧ pyr.is_regular) :
  (AS_CD_ratio A S C D = sqrt (5 / 3)) ∧
  (cone.volume = (125 * Real.pi) / sqrt 21) := by
  sorry

end pyramid_cone_properties_l569_569947


namespace solutions_diff_l569_569468

theorem solutions_diff (a b : ℝ) (h1: (a-5)*(a+5) = 26*a - 130) (h2: (b-5)*(b+5) = 26*b - 130) (h3 : a ≠ b) (h4: a > b) : a - b = 16 := 
by
  sorry 

end solutions_diff_l569_569468


namespace sum_of_differences_l569_569339

theorem sum_of_differences :
  (∑ i in Finset.range 98, (1 / (i + 1) - 1 / (i + 3))) = 14651 / 9900 :=
by
  sorry

end sum_of_differences_l569_569339


namespace opposite_of_2023_is_neg_2023_l569_569886

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end opposite_of_2023_is_neg_2023_l569_569886


namespace choir_average_age_l569_569186

-- Each condition as a definition in Lean 4
def avg_age_females := 28
def num_females := 12
def avg_age_males := 32
def num_males := 18
def total_people := num_females + num_males

-- The total sum of ages calculated from the given conditions
def sum_ages_females := avg_age_females * num_females
def sum_ages_males := avg_age_males * num_males
def total_sum_ages := sum_ages_females + sum_ages_males

-- The final proof statement to be proved
theorem choir_average_age : 
  (total_sum_ages : ℝ) / (total_people : ℝ) = 30.4 := by
  sorry

end choir_average_age_l569_569186


namespace andrey_stamps_l569_569155

theorem andrey_stamps (x : ℤ) :
  x ≡ 1 [MOD 2] ∧
  x ≡ 1 [MOD 3] ∧
  x ≡ 3 [MOD 5] ∧
  x ≡ 7 [MOD 9] ∧
  150 < x ∧ x ≤ 300 → 
  x = 223 :=
by
  sorry

end andrey_stamps_l569_569155


namespace rectangle_area_divisible_by_12_l569_569197

theorem rectangle_area_divisible_by_12
  (x y z : ℤ)
  (h : x^2 + y^2 = z^2) :
  12 ∣ (x * y) :=
sorry

end rectangle_area_divisible_by_12_l569_569197


namespace main_theorem_l569_569237

namespace PointArrowProblem

variable {W B : Type} [Finite W] [Finite B]
variable {arrows : W → B → ℕ}

/-- The condition that for any closed path, the product of the numbers on the arrows in the direction of the path equals the product of the numbers on the arrows against the direction. -/
def closed_path_condition (arrows : W → B → ℕ) : Prop :=
  ∀ (path : List (W × B)), (∀ (i : Fin (path.length)), path.nth i = path.nth ((i + 1) % path.length)) →
    ((path.map (λ p, arrows p.fst p.snd)).prod) = ((path.map (λ p, arrows p.snd p.fst)).prod)

/-- The main theorem: It is possible to assign a natural number to each point such that the number on each arrow equals the product of the numbers at its endpoints. -/
theorem main_theorem (h : closed_path_condition arrows) :
  ∃ (f : W ∪ B → ℕ), ∀ (w : W) (b : B), arrows w b = f (w : W ∪ B) * f (b : W ∪ B) :=
sorry

end PointArrowProblem

end main_theorem_l569_569237


namespace oranges_to_apples_ratio_l569_569971

theorem oranges_to_apples_ratio :
  ∀ (total_fruits : ℕ) (weight_oranges : ℕ) (weight_apples : ℕ),
  total_fruits = 12 →
  weight_oranges = 10 →
  weight_apples = total_fruits - weight_oranges →
  weight_oranges / weight_apples = 5 :=
by
  intros total_fruits weight_oranges weight_apples h1 h2 h3
  sorry

end oranges_to_apples_ratio_l569_569971


namespace perpendicular_slope_l569_569690

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end perpendicular_slope_l569_569690


namespace concave_numbers_count_eq_20_l569_569619

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

def input_set := {0, 2, 3, 4, 5}

noncomputable def count_concave_numbers : ℕ :=
  let candidates := (input_set.to_finset.product $ input_set.to_finset).product input_set.to_finset
  let valid_triples := candidates.filter (λ ⟨⟨a, b⟩, c⟩, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_concave_number a b c)
  valid_triples.card

theorem concave_numbers_count_eq_20 : count_concave_numbers = 20 :=
by
  sorry

end concave_numbers_count_eq_20_l569_569619


namespace find_f_10_minus_f_neg_5_l569_569331

-- Define the polynomial
def f (x : ℝ) (a1 a2 a3 a4 a5 : ℝ) : ℝ := x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5

-- Define the given condition
def condition (a1 a2 a3 a4 a5 f : ℝ → ℝ) : Prop :=
  f 1 = 2017 * 1 ∧ f 2 = 2017 * 2 ∧ f 3 = 2017 * 3 ∧ f 4 = 2017 * 4

-- Main theorem
theorem find_f_10_minus_f_neg_5 (a1 a2 a3 a4 a5 : ℝ) (hf : ∀ x, f x a1 a2 a3 a4 a5 = x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5) :
  condition a1 a2 a3 a4 a5 (f x a1 a2 a3 a4 a5) →
  (f 10 a1 a2 a3 a4 a5) - (f (-5) a1 a2 a3 a4 a5) = 75615 :=
begin
  intro h,
  sorry
end

end find_f_10_minus_f_neg_5_l569_569331


namespace linear_function_details_l569_569353

variables (x y : ℝ)

noncomputable def linear_function (k b : ℝ) := k * x + b

def passes_through (k b x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = linear_function k b x1 ∧ y2 = linear_function k b x2

def point_on_graph (k b x3 y3 : ℝ) : Prop :=
  y3 = linear_function k b x3

theorem linear_function_details :
  ∃ k b : ℝ, passes_through k b 3 5 (-4) (-9) ∧ point_on_graph k b (-1) (-3) :=
by
  -- to be proved
  sorry

end linear_function_details_l569_569353


namespace max_points_russian_players_l569_569783

theorem max_points_russian_players 
  (total_participants : ℕ) 
  (russian_players : ℕ) 
  (vladimir_points : ℕ) 
  (levon_points : ℕ) 
  (points_for_win points_for_draw points_for_loss : ℕ) 
  (h1 : total_participants = 20)
  (h2 : russian_players = 6)
  (h3 : vladimir_points = 18.5)
  (h4 : levon_points = 16)
  (h5 : points_for_win = 1) 
  (h6 : points_for_draw = 0.5) 
  (h7 : points_for_loss = 0) :
  ∃ max_points : ℕ, max_points = 96 :=
by sorry

end max_points_russian_players_l569_569783


namespace transformation_cos_half_x_l569_569556

theorem transformation_cos_half_x (x : ℝ) : 
  (cos (1/2 * (x + π/3)) = cos (1/2 * x + π / 6)) :=
by sorry

end transformation_cos_half_x_l569_569556


namespace area_triangle_eq_l569_569054

-- Ellipse C
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Foci F1 and F2
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Line l passing through F2 with slope 1
def line_l (x y : ℝ) : Prop := y = x - 1 ∧ F2.fst = 2 ∧ F2.snd = 0

-- Points M and N are the intersections of ellipse C and line l
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Area of triangle MNF1
noncomputable def area_triangle_MNF1 : ℝ :=
  let (x1, y1) := M in
  let (x2, y2) := N in
  let d : ℝ := 4 -- distance |F1F2|
  let height : ℝ := (sqrt ((6/7)^2 - 4 * (-9/7)))
  (1/2) * d * height

-- Theorem to prove the area of triangle MNF1 is 12 * sqrt(2) / 7
theorem area_triangle_eq : area_triangle_MNF1 = 12 * sqrt 2 / 7 := sorry

end area_triangle_eq_l569_569054


namespace evaluate_sum_sequence_l569_569336

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n ≥ 2, a (n + 1) = a n - a (n - 1)

def initial_conditions (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 3

def sum_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (∑ i in List.range (n + 1), a i)

theorem evaluate_sum_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  sequence a → initial_conditions a → sum_sequence a S → a 100 = -1 ∧ S 100 = 5 :=
by
  sorry

end evaluate_sum_sequence_l569_569336


namespace cost_of_article_l569_569403

-- Conditions as Lean definitions
def price_1 : ℝ := 340
def price_2 : ℝ := 350
def price_diff : ℝ := price_2 - price_1 -- Rs. 10
def gain_percent_increase : ℝ := 0.04

-- Question: What is the cost of the article?
-- Answer: Rs. 90

theorem cost_of_article : ∃ C : ℝ, 
  price_diff = gain_percent_increase * (price_1 - C) ∧ C = 90 := 
sorry

end cost_of_article_l569_569403


namespace sum_of_all_values_is_50_l569_569084

theorem sum_of_all_values_is_50 (N : ℝ) (hN : 0 ≤ N) :
  let x1 := N + 25,
      x2 := 25 - N in
  x1 + x2 = 50 :=
by
  sorry

end sum_of_all_values_is_50_l569_569084


namespace find_special_numbers_l569_569694

-- Definitions of Euler's Totient function and number of positive divisors
def euler_totient (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Finset.filter (λ (m : ℕ), Nat.coprime m n) (Finset.range n.succ)).card

def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ (d : ℕ), d > 0 ∧ n % d = 0) (Finset.range n.succ)).card

-- Statement of the problem
theorem find_special_numbers :
  {n : ℕ | euler_totient n = num_divisors n} = {1, 3, 8, 10, 18, 24, 30} :=
sorry

end find_special_numbers_l569_569694


namespace line_through_hyperbola_intersection_points_l569_569060

open Real

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
def point_P := (2, 1 : ℝ)

theorem line_through_hyperbola_intersection_points :
  ∃ l : ℝ → ℝ → Prop, 
    l point_P.1 point_P.2 ∧ 
    (∀ x y, l x y → hyperbola x y) ∧
    ∀ A B : ℝ × ℝ, 
      hyperbola A.1 A.2 → hyperbola B.1 B.2 → 
      (A + B) / 2 = point_P → 
      l A.1 A.2 ∧ l B.1 B.2 ∧ 
      (∀ x y, l x y ↔ 4 * x - y = 7) := 
sorry

end line_through_hyperbola_intersection_points_l569_569060


namespace area_of_square_l569_569618

-- Define the conditions given in the problem
def radius_circle := 7 -- radius of each circle in inches

def diameter_circle := 2 * radius_circle -- diameter of each circle

def side_length_square := 2 * diameter_circle -- side length of the square

-- State the theorem we want to prove
theorem area_of_square : side_length_square ^ 2 = 784 := 
by
  sorry

end area_of_square_l569_569618


namespace necessary_but_not_sufficient_for_gt_l569_569990

theorem necessary_but_not_sufficient_for_gt (a b : ℝ) :
  (a > b - 1 ∧ ¬ (a > b) → a > b) :=
by {
  -- a > b implies a > b - 1
  intros ha_gt_b,
  refine ⟨_, _⟩,
  { linarith },
  intros,
  { sorry }
}

end necessary_but_not_sufficient_for_gt_l569_569990


namespace part_I_part_II_l569_569342

noncomputable def setA : Set ℝ := {x : ℝ | (1/2)^(x^2 - 5 * x + 6) ≥ 1/4}
noncomputable def setB : Set ℝ := {x : ℝ | Real.log (x - 3) / Real.log (x - 1) < Real.log 2}
noncomputable def setC (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a}

theorem part_I :
  (setA ∩ setB = {x : ℝ | 3 < x ∧ x ≤ 4}) ∧ 
  ((Set.univ \ setB) ∪ setA = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) :=
by sorry

theorem part_II (a : ℝ) :
  setC a ⊆ setA → 2 ≤ a ∧ a ≤ 4 :=
by sorry

end part_I_part_II_l569_569342


namespace andrew_age_l569_569281

-- Definitions based on the conditions
variables (a g : ℝ)

-- The conditions
def condition1 : Prop := g = 9 * a
def condition2 : Prop := g - a = 63

-- The theorem we want to prove
theorem andrew_age (h1 : condition1 a g) (h2 : condition2 a g) : a = 63 / 8 :=
by
  intros
  sorry

end andrew_age_l569_569281


namespace isosceles_triangle_perimeter_l569_569098

variable (a b c : ℝ)
variable (h1 : a = 4 ∨ a = 8)
variable (h2 : b = 4 ∨ b = 8)
variable (h3 : a = b ∨ c = 8)

theorem isosceles_triangle_perimeter (h : a + b + c = 20) : a = b ∨ b = 8 ∧ (a = 8 ∧ c = 4 ∨ b = c) := 
  by
  sorry

end isosceles_triangle_perimeter_l569_569098


namespace tan_theta_value_l569_569345

theorem tan_theta_value (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 :=
sorry

end tan_theta_value_l569_569345


namespace h_at_4_l569_569460

noncomputable def f (x : ℝ) := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) := 3 - (4 / x)

noncomputable def h (x : ℝ) := (1 / f_inv x) + 10

theorem h_at_4 : h 4 = 10.5 :=
by
  sorry

end h_at_4_l569_569460


namespace percent_increase_l569_569536

theorem percent_increase (P x : ℝ) (h1 : P + x/100 * P - 0.2 * (P + x/100 * P) = P) : x = 25 :=
by
  sorry

end percent_increase_l569_569536


namespace circle_touches_equal_segments_l569_569960

theorem circle_touches_equal_segments
  (O A B C D M : Point)
  (h1 : CircleWithCenter O touchesSidesOfAngleAt A B)
  (h2 : M ∈ Segment AB)
  (h3 : M ≠ A)
  (h4 : M ≠ B)
  (h5 : LinePassingThrough M isPerpendicularTo Line OM)
  (h6 : LinePassingThrough M intersectsSidesOfAngleAt C D) :
  Distance M C = Distance M D := by
  sorry

end circle_touches_equal_segments_l569_569960


namespace slope_AD_l569_569973

-- Definitions based directly on conditions
structure Point where
  x : ℝ
  y : ℝ

structure Parallelogram where
  A B C D : Point

def ellipse_eq (p : Point) : Prop :=
  (p.x^2) / 4 + p.y^2 = 1

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def is_parallelogram (par : Parallelogram) : Prop :=
  ∃ t : ℝ, 
  slope par.A par.B = 1 ∧ 
  ellipse_eq par.A ∧ 
  ellipse_eq par.B ∧ 
  ellipse_eq par.C ∧ 
  ellipse_eq par.D ∧ 
  par.D = {x := -par.B.x, y := -par.B.y} 

-- Proposition to prove
theorem slope_AD (par : Parallelogram) (h : is_parallelogram(par)) : 
  slope par.A par.D = -(1/4) :=
sorry

end slope_AD_l569_569973


namespace sparrow_population_less_than_20_percent_initial_l569_569670

noncomputable def sparrow_population (n : ℕ) : ℝ := 1 * (0.7 ^ n)

theorem sparrow_population_less_than_20_percent_initial (n : ℕ) :
  sparrow_population n < 0.2 ↔ n ≥ 5 :=
begin
  sorry
end

end sparrow_population_less_than_20_percent_initial_l569_569670


namespace find_next_speed_l569_569597

def first_distance : ℝ := 160
def first_speed : ℝ := 90
def next_distance : ℝ := 160
def total_distance : ℝ := 320
def average_speed : ℝ := 84.70588235294117
def total_time : ℝ := total_distance / average_speed
def first_time : ℝ := first_distance / first_speed
def next_time : ℝ := total_time - first_time
def next_speed : ℝ := next_distance / next_time

theorem find_next_speed : next_speed = 80 := by
  sorry

end find_next_speed_l569_569597


namespace geometric_sequence_product_l569_569714

noncomputable def a : ℕ → ℝ := sorry -- replace with the given sequence

theorem geometric_sequence_product :
  (∀ n, a n > 0) →
  log (a 3 * a 8 * a 13) = 6 →
  a 1 * a 15 = 10^4 :=
sorry

end geometric_sequence_product_l569_569714


namespace space_quadrilateral_exists_l569_569470

theorem space_quadrilateral_exists (q : ℕ) (n l : ℕ) (G : Type) 
  [graph G] 
  (h_q_ge_2 : 2 ≤ q) 
  (h_n_eq : n = q^2 + q + 1) 
  (h_l_ge : l ≥ q^2 * (q + 1)^2 + 1)
  (h_non_coplanar : ∀ (v₁ v₂ v₃ v₄ : G), non_coplanar v₁ v₂ v₃ v₄) 
  (h_connected : ∀ v : G, ∃ e : edge G, incident v e) 
  (h_vertex_degree : ∃ v : G, degree v ≥ q + 2) 
  : ∃ (A B C D : G) (e₁ e₂ e₃ e₄ : edge G), incident A e₁ ∧ incident B e₁ ∧ incident B e₂ ∧ incident C e₂ ∧ incident C e₃ ∧ incident D e₃ ∧ incident D e₄ ∧ incident A e₄ :=
begin
  sorry
end

end space_quadrilateral_exists_l569_569470


namespace divides_equiv_l569_569507

theorem divides_equiv (m n : ℤ) : 
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) :=
by
  sorry

end divides_equiv_l569_569507


namespace original_bullets_per_person_l569_569208

theorem original_bullets_per_person (x : ℕ) (h : 5 * (x - 4) = x) : x = 5 :=
by
  sorry

end original_bullets_per_person_l569_569208


namespace inf_coprime_naturals_l569_569492

theorem inf_coprime_naturals (a b : ℤ) (h : a ≠ b) : 
  ∃ᶠ n in Filter.atTop, Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) = 1 := 
sorry

end inf_coprime_naturals_l569_569492


namespace valid_training_squad_selections_l569_569248

open Finset

/-- A cricket team consists of 16 players, including a set of 4 quadruplets and we need to choose 5 players 
for a training squad with the restriction that not all 4 quadruplets can be in the training squad. 
We aim to prove there are exactly 4356 valid ways to form this squad. -/
theorem valid_training_squad_selections (team : Finset ℕ) (quadruplets : Finset ℕ) (squad_size : ℕ) :
  (team.card = 16) → 
  (quadruplets.card = 4) → 
  (quadruplets ⊆ team) → 
  (squad_size = 5) → 
  ∑ s in team.powerset, (s.card = squad_size ∧ (¬ (quadruplets ⊆ s))) → 
  4356 :=
sorry

end valid_training_squad_selections_l569_569248


namespace part1_part2_l569_569061

variable (x k : ℝ)

-- Part (1)
theorem part1 (h1 : x = 3) : ∀ k : ℝ, (1 + k) * 3 ≤ k^2 + k + 4 := sorry

-- Part (2)
theorem part2 (h2 : ∀ k : ℝ, -4 ≤ k → (1 + k) * x ≤ k^2 + k + 4) : -5 ≤ x ∧ x ≤ 3 := sorry

end part1_part2_l569_569061


namespace probability_X_between_l569_569372

variable {a : ℝ}

def P (i : ℕ) : ℝ := i / a

axiom h1 : P 1 + P 2 + P 3 + P 4 = 1

theorem probability_X_between : 
  P(1) + P(2) + P(3) = 0.6 :=
by sorry

end probability_X_between_l569_569372


namespace smallest_n_such_that_A0_An_ge_200_l569_569455

noncomputable def A_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (2 * (i+1) / 3)

theorem smallest_n_such_that_A0_An_ge_200 :
  ∃ n : ℕ, 200 ≤ A_n n ∧ ∀ m < n, A_n m < 200 :=
begin
  sorry
end

end smallest_n_such_that_A0_An_ge_200_l569_569455


namespace points_not_on_excluded_segments_l569_569898

variable (A B C D E F G H J : Type)
variable (a b c : Real)
variable [DistinctOrderedSides : a < b ∧ b < c]

theorem points_not_on_excluded_segments (A B C D E F G H J : Type) (a b c : Real)
  [h : DistinctOrderedSides a b c]:
  ¬(D ∈ segment A C₁ ∨ D ∈ segment B C₂ ∨ 
    E ∈ segment A B₁ ∨ F ∈ segment B A₂ ∨ 
    G ∈ segment A C₁ ∨ H ∈ segment A B₁ ∨ 
    J ∈ segment B A₂ ∨ H ∈ segment B A₂) := 
sorry

end points_not_on_excluded_segments_l569_569898


namespace green_disks_more_than_blue_l569_569585

/-- Prove that the difference between the number of green disks and the number of blue disks 
    is 20, given that every disk in the bag is either blue, yellow, or green, 
    the ratio of blue disks to yellow disks to green disks is 3:7:8, 
    and the total number of disks in the bag is 72. -/
theorem green_disks_more_than_blue (h1 : ∀ d, d ∈ { "blue", "yellow", "green" })
                                     (h2 : ∃ n_b n_y n_g, 3 * n_y = 7 * n_b ∧ 3 * n_g = 8 * n_b)
                                     (h3 : ∃ total, total = 72) :
                                     ∃ n_b n_g, n_g - n_b = 20 :=  
by
  sorry

end green_disks_more_than_blue_l569_569585


namespace arcs_covering_circle_l569_569606

theorem arcs_covering_circle (A : set (set ℝ)) (h1 : ∀ a ∈ A, ¬ (covers_whole_circle a))
  (h2 : covers_whole_circle (⋃₀ A)) : ∃ B ⊆ A, covers_whole_circle (⋃₀ B) ∧ total_measure B ≤ 720 :=
by
  -- Definitions and conditions
  het {_} gon structtholargued vacua-specific-γ
    -- sorry

end arcs_covering_circle_l569_569606


namespace ratio_average_speed_round_trip_l569_569596

-- Defining the constants for the speeds.
def v_b : ℝ := 18    -- Speed of the boat in still water (mph)
def v_c : ℝ := 6     -- Speed of the current (mph)

-- Downstream and upstream speeds can be derived from these.
def v_downstream := v_b + v_c   -- Downstream speed
def v_upstream := v_b - v_c     -- Upstream speed

-- Let d be the distance traveled in each direction; for simplicity, 1 mile.
def d : ℝ := 1                 -- Distance traveled one way (miles)

-- Time to travel downstream and upstream.
def t_downstream := d / v_downstream
def t_upstream := d / v_upstream

-- Total time for the round trip.
def t_total := t_downstream + t_upstream

-- Total distance for the round trip.
def distance_total := 2 * d

-- Average speed for the round trip.
def average_speed := distance_total / t_total

-- Prove that the ratio of the average speed for the round trip to v_b is 8/9.
theorem ratio_average_speed_round_trip :
  average_speed / v_b = 8 / 9 :=
by
  sorry

end ratio_average_speed_round_trip_l569_569596


namespace max_PQ_condition_l569_569321

theorem max_PQ_condition
  (A B C M O P Q : Point)
  (circumcircle : Circle)
  (O_center : is_center O circumcircle)
  (M_on_circumcircle : M ∈ circumcircle)
  (MP_perp_AB : perpendicular MP AB)
  (MQ_perp_AC : perpendicular MQ AC)
  (P_on_AB : P ∈ line_from_points A B)
  (Q_on_AC : Q ∈ line_from_points A C) :
  PQ_length_max EQ AM (diameter circumcircle) :=
sorry

end max_PQ_condition_l569_569321


namespace new_prism_volume_l569_569257

-- Define the original volume
def original_volume : ℝ := 12

-- Define the dimensions modification factors
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 3

-- Define the volume of the new prism
def new_volume := (length_factor * width_factor * height_factor) * original_volume

-- State the theorem to prove
theorem new_prism_volume : new_volume = 144 := 
by sorry

end new_prism_volume_l569_569257


namespace f_periodicity_f_expr_interval_2_4_f_sum_result_l569_569822

noncomputable def f (x : ℝ) :=
if 0 ≤ x ∧ x ≤ 2 then 2 * x - x ^ 2
else if x > 2 ∨ x < 0 then
  let y := x % 4 in if y ≤ 2 then 2 * y - y ^ 2 else y ^ 2 - 6 * y + 8
else sorry -- This part handles other cases, which usually isn't required for our specific questions.

theorem f_periodicity : ∀ x : ℝ, f (x + 4) = f x :=
by sorry

theorem f_expr_interval_2_4 : ∀ x : ℝ, 2 ≤ x → x ≤ 4 → f x = x^2 - 6*x + 8 :=
by sorry

theorem f_sum_result : ∑ i in finset.range 2022, f (i) = 1 :=
by sorry

end f_periodicity_f_expr_interval_2_4_f_sum_result_l569_569822


namespace largest_common_term_l569_569183

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end largest_common_term_l569_569183


namespace sin_cos_inequality_for_any_x_l569_569919

noncomputable def largest_valid_n : ℕ := 8

theorem sin_cos_inequality_for_any_x (n : ℕ) (h : n = largest_valid_n) :
  ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n :=
sorry

end sin_cos_inequality_for_any_x_l569_569919


namespace solution_set_of_inequality_l569_569058

def f (x : ℝ) : ℝ := (1 / x) * (1 / 2 * (Real.log x)^2 - 1 / 2)

theorem solution_set_of_inequality :
  (∀ x : ℝ, (x * (deriv f x) + f x = (Real.log x) / x)) ∧ (f Real.exp = 1 / Real.exp) →
  { x : ℝ | f (x + 1) - f (Real.exp + 1) > x - Real.exp } = set.Ioo (-1) Real.exp :=
by
  sorry

end solution_set_of_inequality_l569_569058


namespace gu_xian_expression_right_triangle_l569_569235

-- Definitions for Part 1
def gu (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 - 1) / 2
def xian (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 + 1) / 2

-- Definitions for Part 2
def a (m : ℕ) (h : m > 1) : ℕ := m^2 - 1
def b (m : ℕ) (h : m > 1) : ℕ := 2 * m
def c (m : ℕ) (h : m > 1) : ℕ := m^2 + 1

-- Proof statement for Part 1
theorem gu_xian_expression (n : ℕ) (hn : n ≥ 3 ∧ n % 2 = 1) :
  gu n hn = (n^2 - 1) / 2 ∧ xian n hn = (n^2 + 1) / 2 :=
sorry

-- Proof statement for Part 2
theorem right_triangle (m : ℕ) (hm: m > 1) :
  (a m hm)^2 + (b m hm)^2 = (c m hm)^2 :=
sorry

end gu_xian_expression_right_triangle_l569_569235


namespace line_through_center_eq_line_chord_len_eq_l569_569040

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

noncomputable def point_P : ℝ × ℝ := (2, 2)

def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

def line_chord_len (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0 ∨ x = 2

theorem line_through_center_eq (x y : ℝ) (hC : circle_eq x y) :
  line_through_center x y :=
sorry

theorem line_chord_len_eq (x y : ℝ) (hC : circle_eq x y) (hP : x = 2 ∧ y = 2 ∧ (line_through_center x y)) :
  line_chord_len x y :=
sorry

end line_through_center_eq_line_chord_len_eq_l569_569040


namespace simplify_polynomial_l569_569508

variable (p : ℝ)

theorem simplify_polynomial :
  (7 * p ^ 5 - 4 * p ^ 3 + 8 * p ^ 2 - 5 * p + 3) + (- p ^ 5 + 3 * p ^ 3 - 7 * p ^ 2 + 6 * p + 2) =
  6 * p ^ 5 - p ^ 3 + p ^ 2 + p + 5 :=
by
  sorry

end simplify_polynomial_l569_569508


namespace sum_of_exponents_of_square_root_of_largest_perfect_square_l569_569575

-- Define the exponents of the prime factors of 15!
def factorial_prime_exponents (n : ℕ) : List ℕ := 
  [11, 6, 3, 2] -- for primes 2, 3, 5, 7 respectively

-- Function to adjust exponents to be even for the largest perfect square
def adjust_to_even (exponents : List ℕ) : List ℕ :=
  exponents.map (λ e => e - e % 2)

-- Function to find the square root's exponents by halving the adjusted exponents
def sqrt_exponents (exponents : List ℕ) : List ℕ :=
  adjust_to_even(exponents).map (λ e => e / 2)

-- Function to sum the exponents
def sum_exponents (exponents : List ℕ) : ℕ :=
  exponents.foldl (+) 0

-- The proof statement
theorem sum_of_exponents_of_square_root_of_largest_perfect_square (n : ℕ) (h : n = 15!) :
  sum_exponents (sqrt_exponents (factorial_prime_exponents n)) = 10 :=
by sorry 

end sum_of_exponents_of_square_root_of_largest_perfect_square_l569_569575


namespace length_PQ_l569_569869

theorem length_PQ (SA : ℝ) (hSA : SA = 25) 
                 (BQ QD : ℝ) (hBQ_QD : BQ / QD = 3 / 2)
                 (P Q : ℝ) (intersection : Prop) 
                 (hIntersection : intersection):
  P Q = 10 := 
  sorry

end length_PQ_l569_569869


namespace correct_statements_l569_569363

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem correct_statements : 
  (∀ x, f x = x^3 - 3 * x^2) ∧ 
  (∀ x, if x < 0 ∨ x > 2 then f'(x) ≥ 0 else if 0 ≤ x ∧ x ≤ 2 then f'(x) ≤ 0 else true) ∧ 
  (∀ x, 
    (x < 0 → f x > f 0) ∧ 
    (x > 2 → f x > f 2) ∧ 
    (0 < x ∧ x < 2 → f x < f 0 ∧ f x < f 2)) ∧ 
  (f 0 = 0 ∧ f 2 = -4) :=
sorry

end correct_statements_l569_569363


namespace triangles_are_congruent_in_divided_100gon_l569_569259

theorem triangles_are_congruent_in_divided_100gon :
  ∀ (P: polygon) (n: ℕ), 
  n = 100 ∧ P.is_regular ∧ 
  (∃ (triangles: list triangle) (parallelograms: list parallelogram), 
    P.is_divided_into triangles parallelograms ∧ 
    triangles.length = 2) → 
  triangles.head ≅ triangles.tail.head :=
by -- proof skipped
  sorry

end triangles_are_congruent_in_divided_100gon_l569_569259


namespace probability_event_l569_569497

open MeasureTheory

-- Conditions: Method of choosing numbers
def coin_flip_distribution : measure ℝ :=
  (1/2) • uniform_of [0, 1] + (1/4) • dirac 0 + (1/4) • dirac 0.5

-- Probability measure resulting from two independent selections
noncomputable def prob_distribution : measure (ℝ × ℝ) :=
  coin_flip_distribution.prod coin_flip_distribution

-- Probability event definition
def event (x y : ℝ) := |x - y| ≥ 1/2

-- Desired Probability calculation
noncomputable def desired_probability : ℝ :=
  prob_distribution.to_outer_measure.measure_of {xy | event xy.1 xy.2}

theorem probability_event : desired_probability = 1/8 :=
by
  -- Proof would go here, but is left out
  sorry

end probability_event_l569_569497


namespace proof_arithmetic_seq_l569_569334

/-- Helper definitions for the proofs -/
def arithmetic_sequence (a d : ℕ) : ℕ → ℕ := λ n, a + (n - 1) * d
def geometric_sequence (a r : ℕ) : ℕ → ℕ := λ n, a * r ^ (n - 1)

def general_formula_condition1 : Prop :=
  ∃ a d, a + 4 * d = 14 ∧ 5 * (a + (a + 4 * d)) / 2 = 40 ∧
         ∀ n, arithmetic_sequence a d n = 3 * n - 1

def general_formula_condition2 (a_1 : ℕ) : Prop :=
  a_1 = 2 ∧ 
  ∃ d, (a_1 + d - 1)^2 = a_1 * (a_1 + 2 * d) ∧
       ∀ n, arithmetic_sequence a_1 d n = 3 * n - 1

def T_n (n : ℕ) : ℚ :=
  ∑ k in finset.range n, 1/(3:ℚ)^(3*k+1)

def Tn_bound (n : ℕ) : Prop :=
  T_n n < 3/26

theorem proof_arithmetic_seq (n : ℕ) (a_1 : ℕ):
  (general_formula_condition1 ∨ general_formula_condition2 a_1) →
  Tn_bound n :=
by
  sorry

end proof_arithmetic_seq_l569_569334


namespace largest_common_value_lt_1000_l569_569181

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end largest_common_value_lt_1000_l569_569181


namespace find_S_l569_569045

noncomputable def A := { x : ℝ | x^2 - 7 * x + 10 ≤ 0 }
noncomputable def B (a b : ℝ) := { x : ℝ | x^2 + a * x + b < 0 }
def A_inter_B_is_empty (a b : ℝ) := A ∩ B a b = ∅
def A_union_B_condition := { x : ℝ | x - 3 < 4 ∧ 4 ≤ 2 * x }

theorem find_S :
  A ∪ B (-12) 35 = { x : ℝ | 2 ≤ x ∧ x < 7 } →
  A ∩ B (-12) 35 = ∅ →
  { x : ℝ | x = -12 + 35 } = { 23 } :=
by
  intro h1 h2
  sorry

end find_S_l569_569045


namespace my_op_evaluation_l569_569404

def my_op (x y : Int) : Int := x * y - 3 * x + y

theorem my_op_evaluation : my_op 5 3 - my_op 3 5 = -8 := by 
  sorry

end my_op_evaluation_l569_569404


namespace circle_radius_l569_569537

-- Given the equation of a circle, we want to prove its radius
theorem circle_radius : ∀ (x y : ℝ), x^2 + y^2 - 6*y - 16 = 0 → (∃ r, r = 5) :=
  by
    sorry

end circle_radius_l569_569537


namespace ellipse_properties_l569_569038

noncomputable def semi_major_axis := 2
noncomputable def semi_focal_distance := Real.sqrt 3
noncomputable def semi_minor_axis := 1
noncomputable def standard_ellipse_eq (x y : ℝ) : Prop := (x ^ 2) / 4 + y ^ 2 = 1

def A : ℝ × ℝ := (1, 1 / 2)

theorem ellipse_properties :
  ∀ (x y : ℝ), standard_ellipse_eq x y →
  (let B_C := (line_through_origin_intersects_ellipse x y) in
    let max_area_of_triangle_ABC :=
      max_area (A) B_C in
    max_area_of_triangle_ABC = Real.sqrt 2) := by
  sorry

end ellipse_properties_l569_569038


namespace finite_set_exists_unit_distance_l569_569160

theorem finite_set_exists_unit_distance (m : ℕ) (h : 0 < m) :
  ∃ S : finset (ℝ × ℝ), ∀ A ∈ S, (finset.filter (λ B, dist A B = 1) (S.erase A)).card = m :=
sorry

end finite_set_exists_unit_distance_l569_569160


namespace distance_between_excircle_centers_l569_569270

-- Definitions and Conditions
variable {A B C O_A O_B O_C : Type} 
variable [metric_space A] [metric_space B] [metric_space C] [metric_space O_A] [metric_space O_B] [metric_space O_C]
variable (triangle : A × B × C)
variable (circle : set A) (unit_circle : metric.ball O_A 1 = circle)

-- Proof Problem Statement
theorem distance_between_excircle_centers
  (h : is_triangle_inscribed_in_unit_circle triangle unit_circle) :
  ∃ d, 0 < d ∧ d < 4 := sorry

end distance_between_excircle_centers_l569_569270


namespace sum_of_smallest_ns_l569_569969

theorem sum_of_smallest_ns : ∀ n1 n2 : ℕ, (n1 ≡ 1 [MOD 4] ∧ n1 ≡ 2 [MOD 7]) ∧ (n2 ≡ 1 [MOD 4] ∧ n2 ≡ 2 [MOD 7]) ∧ n1 < n2 →
  n1 = 9 ∧ n2 = 37 → (n1 + n2 = 46) :=
by
  sorry

end sum_of_smallest_ns_l569_569969


namespace find_first_term_l569_569279

-- Define the conditions for the geometric series
def common_ratio : ℝ := -1 / 3
def series_sum : ℝ := 27

-- The question is what is the first term
def first_term (a : ℝ) : Prop := series_sum = a / (1 - common_ratio)

-- The theorem to prove the first term of the series
theorem find_first_term : ∃ a : ℝ, first_term a ∧ a = 36 :=
by 
  use 36
  split
  · simp [first_term, series_sum, common_ratio]
    -- Additional calculations to show the first_term correctly
    have h : (1 - (-1 / 3)) = 4 / 3 := by norm_num
    calc
    27 = 36 / (4 / 3) : by norm_num
    ... = 36 * (3 / 4) : by rw div_eq_mul_inv
    ... = 27 : by norm_num
  · refl

end find_first_term_l569_569279


namespace AFDE_is_cyclic_l569_569454

variables {A B C S D E F I : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited S] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited I]

-- Define the conditions
variables (Γ1 : circle A B C) (incenter_triangle_ABC : incenter A B C I) 
          (Γ2 : circle B C I) (center_Γ2 : center S Γ2) 
          (D_second_inter_BS : second_inter D Γ2 (line B S)) 
          (E_second_inter_CS : second_inter E Γ2 (line C S)) 
          (F_on_arc_BC_not_S : arc F Γ1 B C) 
          (arc_condition : ∠ B S A = ∠ F S C)

-- Define the proof goal
theorem AFDE_is_cyclic : cyclic_quad A F D E := 
by sorry

end AFDE_is_cyclic_l569_569454


namespace last_digit_2019_digit_number_l569_569242

theorem last_digit_2019_digit_number :
  ∃ n : ℕ → ℕ,  
    (∀ k, 0 ≤ k → k < 2018 → (n k * 10 + n (k + 1)) % 13 = 0) ∧ 
    n 0 = 6 ∧ 
    n 2018 = 2 :=
sorry

end last_digit_2019_digit_number_l569_569242


namespace four_digit_cubes_divisible_by_16_l569_569396

theorem four_digit_cubes_divisible_by_16 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.finite
  ∧ ∃ n, n = 3 ∧ {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.card = n := 
by
  -- The proof steps would go here.
  sorry

end four_digit_cubes_divisible_by_16_l569_569396


namespace model_represents_feet_l569_569529

def height_statue : ℝ := 120
def height_model : ℝ := 6
def feet_per_inch_model : ℝ := height_statue / height_model

theorem model_represents_feet : feet_per_inch_model = 20 := 
by
  sorry

end model_represents_feet_l569_569529


namespace completing_square_solution_l569_569932

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end completing_square_solution_l569_569932


namespace distance_to_school_l569_569233

-- Define the conditions
def jogging_speed : ℝ := 5
def bus_speed : ℝ := 21
def total_time : ℝ := 1

-- Define the problem and state the theorem 
theorem distance_to_school 
  (d : ℝ)
  (h1 : d / jogging_speed + d / bus_speed = total_time) : 
  d ≈ 4.04 := 
sorry

end distance_to_school_l569_569233


namespace geometry_solution_l569_569212

noncomputable def geometry_problem (α β : Circle) (X A P B Q : Point) : Prop :=
  (touches_at X α β ∧
  on_circle A α ∧
  distinct X A ∧
  on_circle P α ∧
  distinct X P ∧
  AX_meets β B ∧
  PX_meets β Q)
  → parallel (line A P) (line B Q)

axiom geometry_conditions (α β : Circle) (X A P B Q : Point) :
  touches_at X α β ∧
  on_circle A α ∧
  distinct X A ∧
  on_circle P α ∧
  distinct X P ∧
  AX_meets β B ∧
  PX_meets β Q

theorem geometry_solution : ∀ (α β : Circle) (X A P B Q : Point),
  geometry_problem α β X A P B Q :=
by
  intros
  exact sorry

end geometry_solution_l569_569212


namespace range_of_a_for_happy_point_l569_569457

theorem range_of_a_for_happy_point : 
  (∃ x ∈ Icc (-3 : ℝ) (-3/2), (λ x, a * x^2 - 2 * x - 2 * a - 3 / 2 + x) x = 0) → 
  a ∈ Icc (-1/4 : ℝ) 0 := 
sorry

end range_of_a_for_happy_point_l569_569457


namespace proof_problem_l569_569568

variable (m n : ℕ)

/-- Any two different elements of the set H are in exactly one selected subset -/
axiom condition_a (H : Type) [Fintype H] (selected_subsets : Finset (Finset H)) :
  ∀ {x y : H}, x ≠ y → ∃! s ∈ selected_subsets, x ∈ s ∧ y ∈ s

/-- For any selected subset R and element p not in R, there are m selected subsets 
    that contain p and intersect R and exactly n selected subsets that 
    contain p and do not intersect R -/
axiom condition_b (H : Type) [Fintype H] (selected_subsets : Finset (Finset H)) :
  ∀ (R : Finset H) (p : H), p ∉ R → 
    ((∃! t ∈ selected_subsets, p ∈ t ∧ t ∩ R ≠ ∅) ∧ card (Finset.filter (λ t, p ∈ t ∧ t ∩ R ≠ ∅) selected_subsets) = m) ∧
    (card (Finset.filter (λ t, p ∈ t ∧ t ∩ R = ∅) selected_subsets) = n)

noncomputable def number_of_elements (H : Type) [Fintype H] (selected_subsets : Finset (Finset H)) [∀ {x y : H}, condition_a H selected_subsets] [∀ (R : Finset H) (p : H), condition_b H selected_subsets R p] :=
  1 + (m + n) * (m - 1)

noncomputable def number_of_designated_subsets (H : Type) [Fintype H] (selected_subsets : Finset (Finset H)) [∀ {x y : H}, condition_a H selected_subsets] [∀ (R : Finset H) (p : H), condition_b H selected_subsets R p] :=
  let h := number_of_elements H selected_subsets
  in (h * (m + n)) / m

theorem proof_problem (H : Type) [Fintype H] (selected_subsets : Finset (Finset H)) [∀ {x y : H}, condition_a H selected_subsets] [∀ (R : Finset H) (p : H), condition_b H selected_subsets R p] :
  number_of_elements H selected_subsets = 1 + (m + n) * (m - 1) ∧
  number_of_designated_subsets H selected_subsets = (1 + (m + n) * (m - 1)) * (m + n) / m :=
by sorry

end proof_problem_l569_569568


namespace prob_good_friends_prob_valid_equation_l569_569422

-- Define conditions and events
def ball_numbers : set ℕ := {1, 2, 3, 4}
noncomputable def draws : Type := {a b c : ℕ // a ∈ ball_numbers ∧ b ∈ ball_numbers ∧ c ∈ ball_numbers}

-- Statement 1: Probability that A and B become "good friends" (i.e., a = b)
theorem prob_good_friends : ∀ (a b : ℕ), a ∈ ball_numbers ∧ b ∈ ball_numbers → 
  (a = b) = (1 / 4) := by sorry

-- Statement 2: Probability that a + b + 2c = 6
theorem prob_valid_equation : ∀ (a b c : ℕ), a ∈ ball_numbers ∧ b ∈ ball_numbers ∧ c ∈ ball_numbers →
  (a + b + 2 * c = 6) = (1 / 16) := by sorry

end prob_good_friends_prob_valid_equation_l569_569422


namespace total_students_registered_l569_569861

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end total_students_registered_l569_569861


namespace find_angle_B_find_length_c_l569_569112

-- Definition of triangle types and angles
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: 2c - a = 2b * cos A
def condition1 : Prop := 2 * c - a = 2 * b * Real.cos A

-- Problem 1: Prove that B = π / 3
theorem find_angle_B (h : condition1) : B = Real.pi / 3 := sorry

-- Problem 2: Prove that if a = 2 and b = sqrt 7, then c = 3
theorem find_length_c (h : condition1) (ha : a = 2) (hb : b = Real.sqrt 7) : c = 3 := sorry

end find_angle_B_find_length_c_l569_569112


namespace ineq_xy_y_2z_geq_4_sqrt_xyz_l569_569588

theorem ineq_xy_y_2z_geq_4_sqrt_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x ≤ 1) :
  xy + y + 2z ≥ 4 * Real.sqrt (xyz) := 
begin
  sorry
end

end ineq_xy_y_2z_geq_4_sqrt_xyz_l569_569588


namespace inverse_mod_81_l569_569729

theorem inverse_mod_81 (h : 9⁻¹ ≡ 65 [MOD 101]) : 81⁻¹ ≡ 84 [MOD 101] :=
sorry

end inverse_mod_81_l569_569729


namespace count_two_digit_primes_from_set_l569_569759

-- Define the set of possible digits
def digit_set : Set ℕ := {1, 3, 4, 6}

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a predicate for valid two-digit numbers formed by the given conditions
def is_valid_two_digit_prime (n : ℕ) : Prop :=
  ∃ t u : ℕ, t ∈ digit_set ∧ u ∈ digit_set ∧ t ≠ u ∧ n = 10 * t + u ∧ is_prime n

-- The main statement
theorem count_two_digit_primes_from_set :
  { n : ℕ | is_valid_two_digit_prime n }.Card = 5 :=
sorry

end count_two_digit_primes_from_set_l569_569759


namespace unique_poly_form_l569_569301

theorem unique_poly_form {f : Polynomial ℤ} (h : ∀ n : ℕ, n > 0 → set.PrimeDivisors n = set.PrimeDivisors (f.eval n)) :
  ∃ k : ℕ, k > 0 ∧ (f = Polynomial.C 1 * Polynomial.X ^ k ∨ f = Polynomial.C (-1) * Polynomial.X ^ k) := 
sorry

end unique_poly_form_l569_569301


namespace mouse_can_see_cheese_five_times_l569_569110

-- Define the main structure and conditions
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

def is_collinear (p1 p2 p3 : Point) : Prop :=
(p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

def cheese_visible_count (t : Triangle) (D E W1 W2 W3 : Point) (mouse_positions : list Point) : ℕ :=
(mouse_positions.map (λ mpos, is_collinear mpos E W1 || is_collinear mpos E W2 ||
                                   is_collinear mpos E W3 || is_collinear mpos E t.A)).count tt

-- Main theorem
theorem mouse_can_see_cheese_five_times :
  ∀ (t : Triangle) (D E W1 W2 W3 : Point)
    (mouse_positions : list Point)
    (h1 : D.x = (t.A.x + t.C.x) / 2 ∧ D.y = (t.A.y + t.C.y) / 2)
    (h2 : E.x = (3 * t.A.x + t.C.x) / 4 ∧ E.y = (3 * t.A.y + t.C.y) / 4)
    (h3 : W1.x = (3 * t.A.x + D.x) / 4 ∧ W1.y = (3 * t.A.y + D.y) / 4)
    (h4 : W2.x = (2 * t.A.x + D.x) / 4 ∧ W2.y = (2 * t.A.y + D.y) / 4)
    (h5 : W3.x = (t.A.x + D.x) / 4 ∧ W3.y = (t.A.y + D.y) / 4)
    (mouse_move_pattern : ∀ n, mouse_positions.nth n = 
      if n % 2 = 0 then some { x := t.A.x + 60 * n, y := t.A.y } else some { x := t.A.x + 60 * (n - 1) + 80, y := t.A.y })
    (lend : mouse_positions.nth 6 = some { x := t.B.x, y := t.B.y }),
  cheese_visible_count t D E W1 W2 W3 mouse_positions = 5 :=
begin
  sorry
end

end mouse_can_see_cheese_five_times_l569_569110


namespace number_of_possible_committees_l569_569284

theorem number_of_possible_committees 
    {biology_men : ℕ} {biology_women : ℕ}
    {physics_men : ℕ} {physics_women : ℕ}
    {chemistry_men : ℕ} {chemistry_women : ℕ}
    (h_bio : biology_men = 3 ∧ biology_women = 1)
    (h_phy : physics_men = 3 ∧ physics_women = 1)
    (h_chem : chemistry_men = 3 ∧ chemistry_women = 1)
    : 
    let committee_ways := (@finset.choose 1 biology_women) * (@finset.choose 1 biology_men) * (@finset.choose 1 physic_women) * (@finset.choose 1 physic_men) * (@finset.choose 1 chemistry_women) * (@finset.choose 1 chemistry_men) 
    (h_valid_committee : biotechnology_ways = 27 ).
    sorry

end number_of_possible_committees_l569_569284


namespace total_time_before_playing_game_l569_569118

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end total_time_before_playing_game_l569_569118


namespace xy_from_equation_l569_569026

theorem xy_from_equation (x y : ℝ) (h : sqrt (2 * x - 4) + abs (y - 1) = 0) : x * y = 2 :=
sorry

end xy_from_equation_l569_569026


namespace matrix_pow_six_identity_l569_569452

variable {n : Type} [Fintype n] [DecidableEq n]
variables {A B C : Matrix n n ℂ}

theorem matrix_pow_six_identity 
  (h1 : A^2 = B^2) (h2 : B^2 = C^2) (h3 : B^3 = A * B * C + 2 * (1 : Matrix n n ℂ)) : 
  A^6 = 1 :=
by 
  sorry

end matrix_pow_six_identity_l569_569452


namespace final_problem_l569_569956

def problem1 : Prop :=
  ∃ (x y : ℝ), 10 * x + 20 * y = 3000 ∧ 8 * x + 24 * y = 2800 ∧ x = 200 ∧ y = 50

def problem2 : Prop :=
  ∀ (m : ℕ), 10 ≤ m ∧ m ≤ 12 ∧ 
  200 * m + 50 * (40 - m) ≤ 3800 ∧ 
  (40 - m) ≤ 3 * m →
  (m = 10 ∧ (40 - m) = 30) ∨ 
  (m = 11 ∧ (40 - m) = 29) ∨ 
  (m = 12 ∧ (40 - m) = 28)

theorem final_problem : problem1 ∧ problem2 :=
by
  sorry

end final_problem_l569_569956


namespace three_digit_even_with_sum_twelve_l569_569758

theorem three_digit_even_with_sum_twelve :
  ∃ n: ℕ, n = 36 ∧ 
    (∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 2 = 0 ∧ 
          ((x / 10) % 10 + x % 10 = 12) → x = n) :=
sorry

end three_digit_even_with_sum_twelve_l569_569758


namespace colorful_tartan_distribution_l569_569533

-- Define the set of characters and their multiplicities
def letters : finset (char × ℕ) :=
  {('C', 1), ('O', 2), ('L', 2), ('R', 2), ('F', 1), 
   ('U', 1), ('T', 2), ('A', 2), ('N', 1)}

-- Define a function to calculate the number of ways to distribute the blocks
def ways_to_distribute (chars : finset (char × ℕ)) : ℕ :=
  2^4 -- Since there are 4 remaining letters that can be placed in either bag

theorem colorful_tartan_distribution :
  ways_to_distribute letters = 16 :=
by
  -- Proof to be filled in later
  sorry

end colorful_tartan_distribution_l569_569533


namespace find_lambda_l569_569380

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 0)

-- Dot product function for 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : dot_product ((λ * a.1, λ * a.2) - b) a = 0) : λ = 3 / 5 := 
by 
  -- Proof is omitted
  sorry

end find_lambda_l569_569380


namespace maximize_sum_of_terms_l569_569718

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 19) ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n - 3)

theorem maximize_sum_of_terms {a : ℕ → ℤ} (hseq : sequence a) :
  ∃ n : ℕ, (n = 7) ∧ (∀ m : ℕ, sum_up_to a m ≤ sum_up_to a n) :=
sorry

noncomputable def sum_up_to (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum a

end maximize_sum_of_terms_l569_569718


namespace possible_k_values_l569_569717

def cube_vertices : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)}

def is_perpendicular (a b c d : (ℝ × ℝ × ℝ)) : Prop :=
  let v1 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let v2 := (d.1 - c.1, d.2 - c.2, d.3 - c.3)
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

def valid_selection (M : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (X_i X_j : ℝ × ℝ × ℝ), X_i ∈ M → X_j ∈ M ∧ X_i ≠ X_j →
  ∃ (X_l X_t : ℝ × ℝ × ℝ), X_l ∈ M ∧ X_t ∈ M ∧ is_perpendicular X_i X_j X_l X_t

theorem possible_k_values (k : ℕ) (h : 2 ≤ k ∧ fintype.card M = k) (M : Set (ℝ × ℝ × ℝ) := cube_vertices) :
  valid_selection M → k = 6 ∨ k = 7 ∨ k = 8 :=
by sorry

end possible_k_values_l569_569717


namespace toby_change_correct_l569_569560

def cheeseburger_cost := 3.65
def milkshake_cost := 2.00
def coke_cost := 1.00
def fries_cost := 4.00
def cookie_cost := 0.50
def sales_tax_rate := 0.07
def tip_rate := 0.15
def initial_amount := 15.00

def pre_tax_total := 
  2 * cheeseburger_cost +
  milkshake_cost +
  coke_cost +
  fries_cost +
  3 * cookie_cost

def sales_tax := (sales_tax_rate * pre_tax_total).round

def tip := tip_rate * pre_tax_total

def final_total := pre_tax_total + sales_tax + tip

def total_per_person := final_total / 2

def toby_change := initial_amount - total_per_person

theorem toby_change_correct : toby_change = 5.36 := by
  unfold toby_change initial_amount total_per_person final_total tip pre_tax_total
  unfold_smul
  sorry

end toby_change_correct_l569_569560


namespace average_integer_solution_l569_569330

theorem average_integer_solution {n : ℕ} {a : Fin 2n → ℤ}
  (distinct_integers : ∀ i j, i ≠ j → a i ≠ a j)
  (h : ∃ r : ℤ, (∏ i : Fin 2n, (r - a i)) + (-1)^(n-1) * (n!)^2 = 0) :
  ∃ r : ℤ, r = (∑ i : Fin 2n, a i) / (2 * n) :=
sorry

end average_integer_solution_l569_569330


namespace solution_set_system_eqns_l569_569540

theorem solution_set_system_eqns : 
  ∃ (x y : ℝ), x + y = 3 ∧ x - y = 1 ∧ (x, y) = (2, 1) := 
by 
  use 2 
  use 1 
  simp 
  split_all 
  rw [add_comm] 
  simp

#summary
- Identify all questions and conditions: Find the solution of the system of two equations.
- Identify all solution steps and the correct answers.
- Translate the question, conditions, correct answer tuple to a mathematically equivalent proof with conditions.
- Rewrite the math proof problem to a Lean4 statement.
* Ensure that the Lean code builds successfully.
* Each definition should appear in the conditions published in the problem.
* Each condition in the question should be used as a definition in Lean.

end solution_set_system_eqns_l569_569540


namespace range_of_f_neg_three_halves_l569_569031

noncomputable def f : ℝ → ℝ := sorry -- Define the function f
def g (x : ℝ) : ℝ := x^3 / f x -- Define the function g with x^3 / f(x)

theorem range_of_f_neg_three_halves 
  (hf_diff : differentiable ℝ f)
  (hf_odd : ∀ x : ℝ, f (-x) = -f x)
  (hf_values : f 1 = 1 ∧ f 2 = 2)
  (hf_cond : ∀ x : ℝ, 0 < x → 3*f x - x*f'' x > 1) :
  f (-3/2) ∈ set.Ioo (-27/8) (-27/32) :=
sorry

end range_of_f_neg_three_halves_l569_569031


namespace exists_c_gt_zero_l569_569818

theorem exists_c_gt_zero (a b : ℕ) (h_a_square_free : ¬ ∃ (k : ℕ), k^2 ∣ a)
    (h_b_square_free : ¬ ∃ (k : ℕ), k^2 ∣ b) (h_a_b_distinct : a ≠ b) :
    ∃ c > 0, ∀ n : ℕ, n > 0 →
    |(n * Real.sqrt a % 1) - (n * Real.sqrt b % 1)| > c / n^3 := sorry

end exists_c_gt_zero_l569_569818


namespace units_digit_of_factorial_149_l569_569576

theorem units_digit_of_factorial_149 : Nat.unitsDigit (Nat.factorial 149) = 0 := 
sorry

end units_digit_of_factorial_149_l569_569576


namespace minimum_triangle_area_l569_569335

theorem minimum_triangle_area (r a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a = b) : 
  ∀ T, (T = (a + b) * r / 2) → T = 2 * r * r :=
by 
  sorry

end minimum_triangle_area_l569_569335


namespace max_xy_l569_569515

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions given in the problem
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom eq1 : x + 1/y = 3
axiom eq2 : y + 2/x = 3

theorem max_xy : ∃ (xy : ℝ), 
  xy = x * y ∧ xy = 3 + Real.sqrt 7 := sorry

end max_xy_l569_569515


namespace angle_between_lines_eq_90_l569_569676

open EuclideanGeometry -- Assuming a relevant module is opened for Euclidean geometry

theorem angle_between_lines_eq_90
  (A B C D M N K L : Point) -- Points definition
  (AD BC AB CD : Segment) -- Segments definition
  (hM : midpoint M A D)
  (hN : midpoint N B C)
  (hK : midpoint K A B)
  (hL : midpoint L C D)
  (equal_midpoints : dist M N = dist K L) : 
  angle AC BD = 90 := sorry

end angle_between_lines_eq_90_l569_569676


namespace black_lambs_count_l569_569665

/-- Definition of the total number of lambs. -/
def total_lambs : Nat := 6048

/-- Definition of the number of white lambs. -/
def white_lambs : Nat := 193

/-- Prove that the number of black lambs is 5855. -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end black_lambs_count_l569_569665


namespace part1_part2_l569_569364

def f (x : ℝ) : ℝ := (sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * (Real.cos x ^ 2 - Real.sin x ^ 2) - 1
def g (x : ℝ) : ℝ := f (x + π / 6)

variables {A B C a b c : ℝ} {m n : ℝ × ℝ}

-- First part: proving the values of a and b
theorem part1 (h_c: c = sqrt 7) (h_fC: f C = 0) (h_sinB: Real.sin B = 3 * Real.sin A) :
  a = 1 ∧ b = 3 := 
sorry

-- Second part: proving the range of dot product of vectors
theorem part2 (h_gB: g B = 0) (h_m: m = (Real.cos A, Real.cos B)) (h_n: n = (1, Real.sin A - Real.cos A * Real.tan B)) :
  0 < m.1 * n.1 + m.2 * n.2 ∧ m.1 * n.1 + m.2 * n.2 ≤ 1 :=
sorry

end part1_part2_l569_569364


namespace max_tangent_points_four_circles_l569_569318

noncomputable def maximum_tangent_points (C1 C2 C3 C4 : Circle) : ℕ :=
  if (¬concentric C1 C2) ∧ (¬concentric C1 C3) ∧ (¬concentric C1 C4) ∧ 
     (¬concentric C2 C3) ∧ (¬concentric C2 C4) ∧ (¬concentric C3 C4) then 4 else 0

theorem max_tangent_points_four_circles (C1 C2 C3 C4 : Circle)
  (h : (¬concentric C1 C2) ∧ (¬concentric C1 C3) ∧ (¬concentric C1 C4) ∧ 
       (¬concentric C2 C3) ∧ (¬concentric C2 C4) ∧ (¬concentric C3 C4)) : 
  maximum_tangent_points C1 C2 C3 C4 = 4 :=
begin
  sorry -- Proof will establish the maximum tangent points
end

end max_tangent_points_four_circles_l569_569318


namespace quad_area_l569_569873

variables (W X Y Z O M N P : Type)
variables (S : ℝ)
variables [module ℝ W XYZ] [affine_space ℝ W XYZ]

structure Parallelogram (A B C D : XYZ) :=
(area  : ℝ)
(meet_diagonals : midpoint ℝ A C = midpoint ℝ B D)

structure Triangle (A B C : XYZ) :=
(area : ℝ)

def area_WXYZ (W X Y Z : XYZ) (S : ℝ) : Prop :=
Parallelogram.mk W X Y Z S (midpoint ℝ W Y = midpoint ℝ X Z)

def triangles_sum_area (WNZ XYP : Triangle W XYZ) : Prop :=
WNZ.area + XYP.area = (1/3) * S

def quad_MNOP_area (MNOP : ℝ) (W X Y Z : XYZ) :=
MNOP = (1/12) * S

theorem quad_area (W X Y Z O M N P : XYZ) (S : ℝ) (H1 : area_WXYZ W X Y Z S)
  (H2 : triangles_sum_area (Triangle.mk W N Z) (Triangle.mk X Y P)) :
  quad_MNOP_area (area_MNOP W X Y Z O M N P) :=
begin
  sorry
end

end quad_area_l569_569873


namespace person_B_work_days_l569_569913

-- Let a be the work rate for person A, and b be the work rate for person B.
-- a completes the work in 20 days
-- b completes the work in x days
-- When working together, a and b complete 0.375 of the work in 5 days


theorem person_B_work_days (x : ℝ) :
  ((5 : ℝ) * ((1 / 20) + 1 / x) = 0.375) -> x = 40 := 
by 
  sorry

end person_B_work_days_l569_569913


namespace percentage_greater_by_17_4_l569_569951

theorem percentage_greater_by_17_4 (X : ℝ) :
  0.60 * 50 = 30 ∧ (30 = (X / 100) * 30 + 17.4) → X = 42 :=
by
  intros h,
  sorry

end percentage_greater_by_17_4_l569_569951


namespace sum_even_desc_l569_569928

theorem sum_even_desc (n : ℕ) (a₁ : ℕ) (aₙ : ℕ) (d : ℤ) 
  (h₁ : n = 123) (h₂ : a₁ = 246) (h₃ : aₙ = 2) (h₄ : d = -2) :
  (∑ i in finset.range n, (a₁ + i * d)) = 15252 := 
by 
  sorry

end sum_even_desc_l569_569928


namespace binomial_expansion_coefficient_l569_569189

theorem binomial_expansion_coefficient :
  (Nat.choose 5 3) * (2^3) * ((-3)^2) = 720 :=
by
  sorry

end binomial_expansion_coefficient_l569_569189


namespace quadrant_of_angle_l569_569727

theorem quadrant_of_angle (θ : ℝ) (h1 : Real.sin θ > 0) (h2 : Real.cos θ < 0) : 
  θ ≤ π ∧ θ > π / 2 :=
by
  sorry

end quadrant_of_angle_l569_569727


namespace part_1_part_1_full_part_2_l569_569323

def f (k : ℤ) : ℝ := Real.sin (k * Real.pi / 4)

theorem part_1 (h : ∑ i in (Finset.range 8).map (λ x, x + 1), f i = 0) : 
  ∑ i in (Finset.range 16).filter (λ x, x >= 9) ∪ (Finset.range 16).filter (λ x, x < 9), f i = 0 := by
  sorry

theorem part_1_full : ∑ i in (Finset.range 8).map (λ x, x + 1), f i = 
                      ∑ i in (Finset.filter (λ x, x ≥ 9) (Finset.range 16)), f i := by 
  sorry

theorem part_2 : ∑ i in (Finset.range 2020).map (λ x, x + 1), f i = 1 + Real.sqrt 2 := by
  sorry

end part_1_part_1_full_part_2_l569_569323


namespace domain_of_function_l569_569520

-- Define the conditions as predicates
def condition1 (x : ℝ) : Prop := 2 - x > 0
def condition2 (x : ℝ) : Prop := x + 1 > 0

-- Prove that the domain of the function is (-1, 2)
theorem domain_of_function : ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ (-1 < x ∧ x < 2) :=
by
  intros
  split
  {
    intro h
    cases h with h1 h2
    split
    {
      linarith
    } 
    {
      linarith
    }
  }
  {
    intro h
    cases h with h1 h2
    split
    {
      linarith
    }
    {
      linarith
    }
  }

end domain_of_function_l569_569520


namespace jake_snake_income_l569_569804

theorem jake_snake_income:
  let snakes := 3 in
  let eggs_per_snake := 2 in
  let price_regular := 250 in
  let price_super_rare := 4 * price_regular in
  let total_babies := snakes * eggs_per_snake in
  let regular_snakes := total_babies - 1 in
  let income := regular_snakes * price_regular + price_super_rare in
  income = 2250 :=
by
  sorry

end jake_snake_income_l569_569804


namespace number_of_representatives_from_companyA_l569_569650

-- Define conditions
def companyA_representatives : ℕ := 120
def companyB_representatives : ℕ := 100
def total_selected : ℕ := 11

-- Define the theorem
theorem number_of_representatives_from_companyA : 120 * (11 / (120 + 100)) = 6 := by
  sorry

end number_of_representatives_from_companyA_l569_569650


namespace Amanda_replacing_18_knobs_l569_569557

-- Define the important constants and parameters given in the problem
def cost_knob : ℝ := 2.50
def number_pulls : ℕ := 8
def cost_pull : ℝ := 4.00
def total_cost : ℝ := 77.00

-- Define the proposition we need to prove
theorem Amanda_replacing_18_knobs : 
  (∃ k : ℕ, cost_knob * k + number_pulls * cost_pull = total_cost) → (∃ k : ℕ, k = 18) :=
by
  assume h,
  obtain ⟨k, h_eq⟩ := h,
  sorry

end Amanda_replacing_18_knobs_l569_569557


namespace polynomial_factor_l569_569768

theorem polynomial_factor (P Q R S : ℤ[X]) :
  (∀ x : ℤ, P (x^5) + x * Q (x^5) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S x) →
  P 1 = 0 :=
by
  sorry

end polynomial_factor_l569_569768


namespace sufficient_but_not_necessary_condition_l569_569140

theorem sufficient_but_not_necessary_condition (θ : ℝ) :
  (abs (θ - π / 12) < π / 12) → (sin θ < 1 / 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l569_569140


namespace count_irrationals_l569_569798

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem count_irrationals : 
  let a := (1.212: ℝ)
  let b := (1.010010001: ℝ)  -- Note: This might require interpretation as it isn't standard Lean term.
  let c := (Real.pi / 2: ℝ)
  let d := (Real.sqrt 3 / 2: ℝ)
  let e := (3 / 4: ℝ)
  (¬ is_irrational a ∧ is_irrational b ∧ is_irrational c ∧ is_irrational d ∧ ¬ is_irrational e) → 
  (card {x : ℝ | x = b ∨ x = c ∨ x = d} = 3) := 
sorry

end count_irrationals_l569_569798


namespace sum_of_pyramid_volumes_l569_569093

structure Parallelepiped :=
  (a b c : ℝ)
  (V R m : ℝ)
  (orthogonal : a^2 + b^2 + c^2 = 4 * R^2 ∧ a + b + c = m)
  (volume_eq : a * b * c = V)

theorem sum_of_pyramid_volumes (P : Parallelepiped) : 
  P.a * P.b * P.c = P.V → 
  (let sum_of_volumes := (2 * P.V) / 3 in
   sum_of_volumes) :=
by {
  intros h,
  have h1 : P.a * P.b * P.c = P.V, from h,
  sorry
}

end sum_of_pyramid_volumes_l569_569093


namespace logarithmic_product_l569_569741

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem logarithmic_product (a b : ℝ) (h1 : a ≠ b) (h2 : f a = f b) : a * b = 1 := by
  sorry

end logarithmic_product_l569_569741


namespace product_fraction_series_l569_569307

theorem product_fraction_series : 
  (∏ i in (range 8), match i % 2 with
                     | 0 => 1 / (2 ^ (2*i + 1))
                     | _ => 2 ^ (2*i + 2)
                     end) = 256 :=
sorry

end product_fraction_series_l569_569307


namespace hall_100_guests_67_friends_find_clique_l569_569780

theorem hall_100_guests_67_friends_find_clique :
  ∀ (P : Fin 100 → Fin 100 → Prop) (n : Fin 100),
    (∀ i : Fin 100, ∃ S : Finset (Fin 100), (S.card ≥ 67) ∧ (∀ j ∈ S, P i j)) →
    (∃ (A B C D : Fin 100), P A B ∧ P A C ∧ P A D ∧ P B C ∧ P B D ∧ P C D) :=
by
  sorry

end hall_100_guests_67_friends_find_clique_l569_569780


namespace lattice_points_count_l569_569610

-- A definition of lattice points and bounded region
def is_lattice_point (p : ℤ × ℤ) : Prop := true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  (y = abs x ∨ y = -x^2 + 4*x + 6) ∧ (y ≤ abs x ∧ y ≤ -x^2 + 4*x + 6)

-- The target statement to prove
theorem lattice_points_count : ∃ n, n = 23 ∧ ∀ p : ℤ × ℤ, is_lattice_point p → in_region p := sorry

end lattice_points_count_l569_569610


namespace problem_statement_l569_569728

theorem problem_statement (x : ℝ) (h1 : x = 3 ∨ x = -3) : 6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := 
by {
  sorry
}

end problem_statement_l569_569728


namespace ellipse_focus_distance_l569_569905

open Real

noncomputable def ellipse_eq := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def foci_x (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

def point_P_x (c : ℝ) : ℝ := c

def point_P_y (c : ℝ) : ℝ := sqrt (1 - c^2 / 4)

def point_distance (a b : ℝ) : ℝ := 2 * a - b

theorem ellipse_focus_distance :
  ∃ (x y : ℝ), point_distance 2 (abs (point_P_y (foci_x 2 1))) = 7 / 2 := 
by 
  refine ⟨sqrt 3, 1 / 2, _⟩
  sorry

end ellipse_focus_distance_l569_569905


namespace four_digit_cubes_divisible_by_16_l569_569394

theorem four_digit_cubes_divisible_by_16 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.finite
  ∧ ∃ n, n = 3 ∧ {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.card = n := 
by
  -- The proof steps would go here.
  sorry

end four_digit_cubes_divisible_by_16_l569_569394


namespace students_in_college_l569_569785

variable (P S : ℕ)

def condition1 : Prop := S = 15 * P
def condition2 : Prop := S + P = 40000

theorem students_in_college (h1 : condition1 S P) (h2 : condition2 S P) : S = 37500 := by
  sorry

end students_in_college_l569_569785


namespace f_of_f_neg1_l569_569366

noncomputable def f : ℝ → ℝ := 
λ x, if x < 0 then 2^(-x) 
else Real.log2 (x + 6)

theorem f_of_f_neg1 : f (f (-1)) = 3 :=
by 
  sorry

end f_of_f_neg1_l569_569366


namespace margaret_mean_score_l569_569992

theorem margaret_mean_score
  (scores : list ℕ)
  (h_scores : scores = [88, 90, 94, 95, 96, 99])
  (cyprian_scores : list ℕ)
  (Margaret_scores : list ℕ)
  (h_partition : cyprian_scores.length = 4 ∧ Margaret_scores.length = 2 ∧ cyprian_scores ++ Margaret_scores = scores)
  (h_cyprian_mean : (cyprian_scores.sum : ℚ) / cyprian_scores.length = 92) :
  (Margaret_scores.sum : ℚ) / Margaret_scores.length = 97 := by
  sorry

end margaret_mean_score_l569_569992
