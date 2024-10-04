import Mathlib
import Mathlib.Algebra.BigOperators.Power
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Partition
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Bijection
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Field
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Triangle
import Mathlib.NumberTheory.Prime
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independent.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Real

namespace concurrency_condition_l647_647528

-- Define the problem and conditions
variables {α1 α2 β1 β2 γ1 γ2 : ℝ}

-- Define the theorem to be proved
theorem concurrency_condition :
  (AD BE CF : Line) (P : Point) 
  (hα1 : Angle (AD, AB) = α1) 
  (hα2 : Angle (AD, AC) = α2) 
  (hβ1 : Angle (BE, BC) = β1) 
  (hβ2 : Angle (BE, BA) = β2) 
  (hγ1 : Angle (CF, CA) = γ1) 
  (hγ2 : Angle (CF, CB) = γ2) :
  (∃ P, concurrent AD BE CF P) ↔ (sin α1 * sin β1 * sin γ1 / sin α2 * sin β2 * sin γ2 = 1) :=
sorry

end concurrency_condition_l647_647528


namespace cube_condition_even_side_length_l647_647509

theorem cube_condition_even_side_length (k : ℕ) (h : ∀ i j l, i < k → j < k → l < k → (∃! c : bool, 
  (c = (color i j l) ∧ (
    ((i + 1 < k → color (i + 1) j l = c) + 
    (i > 0 → color (i - 1) j l = c) + 
    (j + 1 < k → color i (j + 1) l = c) + 
    (j > 0 → color i (j - 1) l = c) + 
    (l + 1 < k → color i j (l + 1) = c) + 
    (l > 0 → color i j (l - 1) = c)))) → 
    ((color i j l = color (i + 1) j l ∨ color i j l = color (i - 1) j l) ∧ 
     (color i j l = color i (j + 1) l ∨ color i j l = color i (j - 1) l) ∧ 
     (color i j l = color i j (l + 1) l ∨ color i j (l - 1) l))
 := k % 2 = 0 := sorry

end cube_condition_even_side_length_l647_647509


namespace complementary_combinations_count_l647_647587

-- Definitions for the problem
structure Card where
  shape   : ℕ -- 3 types (indexed by 0, 1, 2)
  color   : ℕ -- 3 types (indexed by 0, 1, 2)
  pattern : ℕ -- 3 types (indexed by 0, 1, 2)

def isComplementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∨ c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∨ c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

theorem complementary_combinations_count :
  (∃ (cards : Finset Card), cards.card = 27 ∧ 
  ∀ c ∈ cards, c.shape < 3 ∧ c.color < 3 ∧ c.pattern < 3) → 
  (Finset.univ.card = 27 → 
  ∃ (complementary_comb : Finset (Card × Card × Card)),
    complementary_comb.card = 117 ∧
    (∀ xyz ∈ complementary_comb, isComplementary xyz.1 xyz.2.1 xyz.2.2)) := by
  sorry

end complementary_combinations_count_l647_647587


namespace prob_sum_exceeds_18_prob_at_least_one_satisfied_l647_647585

-- Define the scores and classifications according to the problem's conditions.
def scores : list ℝ := [7.6, 8.3, 8.7, 8.9, 9.1, 9.2, 9.3, 9.4, 9.9, 10]

def is_satisfied (score : ℝ) : Prop := score ≥ 9.5
def is_unsatisfied (score : ℝ) : Prop := score < 9
def is_basically_satisfied (score : ℝ) : Prop := 9 ≤ score ∧ score < 9.5

-- Problem (I): Probability of sum > 18
theorem prob_sum_exceeds_18 :
  (({pair | ∃ x y ∈ scores, is_unsatisfied x ∧ is_basically_satisfied y ∧ x + y > 18}.card : ℚ) / 
   ({pair | ∃ x y ∈ scores, is_unsatisfied x ∧ is_basically_satisfied y}.card)) = 1 / 4 :=
sorry

-- Problem (II): Probability at least one satisfied in pair
theorem prob_at_least_one_satisfied :
  (({pair | ∃ x y ∈ scores, (is_satisfied x ∨ is_satisfied y) ∧ (is_satisfied x ∨ is_basically_satisfied x) ∧ 
                                        (is_satisfied y ∨ is_basically_satisfied y)}.card : ℚ) / 
   ({pair | ∃ x y ∈ scores, (is_satisfied x ∨ is_basically_satisfied x) ∧ 
                                        (is_satisfied y ∨ is_basically_satisfied y)}.card)) = 3 / 5 :=
sorry

end prob_sum_exceeds_18_prob_at_least_one_satisfied_l647_647585


namespace divisible_by_factorial_l647_647336

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _ => 0
| _, 0 => 0
| n + 1, k + 1 => (n + 1) * (f (n + 1) k + f n k)

theorem divisible_by_factorial (n k : ℕ) : n! ∣ f n k := by sorry

end divisible_by_factorial_l647_647336


namespace num_sol_pos_int_n_count_pos_int_n_l647_647826

theorem num_sol_pos_int_n (n : ℕ) (h : 300 < n^2 ∧ n^2 < 1200) :
  n ∈ {n | 18 ≤ n ∧ n ≤ 34} :=
begin
  sorry
end

theorem count_pos_int_n : finset.card ({n : ℕ | 300 < n^2 ∧ n^2 < 1200}.to_finset) = 17 :=
by sorry

end num_sol_pos_int_n_count_pos_int_n_l647_647826


namespace coeff_x9_in_expansion_l647_647073

theorem coeff_x9_in_expansion : 
  coeff (expand (1 + 3*x - 2*x^2)^5) 9 = 240 := 
  sorry

end coeff_x9_in_expansion_l647_647073


namespace parallelogram_area_l647_647222

def v : ℝ × ℝ := (5, -3)
def w : ℝ × ℝ := (11, -2)

theorem parallelogram_area :
  let matrix_determinant (v w : ℝ × ℝ) := v.1 * w.2 - v.2 * w.1 in
  |matrix_determinant v w| = 23 :=
by
  sorry

end parallelogram_area_l647_647222


namespace binomial_sum_eq_728_l647_647847

theorem binomial_sum_eq_728 :
  (Nat.choose 6 1) * 2^1 +
  (Nat.choose 6 2) * 2^2 +
  (Nat.choose 6 3) * 2^3 +
  (Nat.choose 6 4) * 2^4 +
  (Nat.choose 6 5) * 2^5 +
  (Nat.choose 6 6) * 2^6 = 728 :=
by
  sorry

end binomial_sum_eq_728_l647_647847


namespace surface_area_is_correct_l647_647120

noncomputable def surface_area_of_sphere (A B C O : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (AB AC : ℝ) (angle_BAC : ℝ) (distance_O_to_plane : ℝ) : ℝ :=
  if h : AB = 2 ∧ AC = 2 ∧ angle_BAC = 90 ∧ distance_O_to_plane = sqrt 2 then
    4 * π * 2 ^ 2
  else
    0

theorem surface_area_is_correct (A B C O : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (AB AC : ℝ) (angle_BAC : ℝ) (distance_O_to_plane : ℝ)
  (h : AB = 2 ∧ AC = 2 ∧ angle_BAC = 90 ∧ distance_O_to_plane = sqrt 2) :
  surface_area_of_sphere A B C O AB AC angle_BAC distance_O_to_plane = 16 * π :=
by
  unfold surface_area_of_sphere
  simp [h]
  sorry

end surface_area_is_correct_l647_647120


namespace max_two_scoop_sundaes_l647_647401

theorem max_two_scoop_sundaes (n : ℕ) (h : n = 8) : (nat.choose n 2) = 28 :=
by 
  rw h
  apply nat.choose_succ_succ
  -- choose_succ_succ calculates (n+1)C(k+1) = nCk + nC(k+1)
  -- For the case with n = 8 and k = 1, we eventually hit 28 combinations for 2 out of 8.
  sorry

end max_two_scoop_sundaes_l647_647401


namespace find_polynomials_l647_647838

noncomputable def satisfies_conditions (P : Polynomial ℤ) : Prop :=
  P.eval 1 ≠ P.eval 2 ∧ P.eval 1 ≠ P.eval 3 ∧ ... ∧ P.eval 2020 ≠ P.eval 2021 ∧ 
  ∃ (L : List ℤ), L = List.range' 1 2022 ∧ (∀ x ∈ List.range' 1 2022, P.eval x ∈ L)

theorem find_polynomials (P : Polynomial ℤ) (R S : Polynomial ℤ) :
  satisfies_conditions P →
  P = X + (X - 1) * (X - 2) * ... * (X - 2021) * R ∨
  P = 2022 - X + (X - 1) * (X - 2) * ... * (X - 2021) * S :=
sorry

end find_polynomials_l647_647838


namespace vector_magnitude_inequality_l647_647666

-- Define the origin in 3D space
def origin : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define a point in 3D space with nonnegative coordinates
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (nonneg_x : 0 ≤ x)
  (nonneg_y : 0 ≤ y)
  (nonneg_z : 0 ≤ z)

-- Define the vector from origin to a point
def vector_from_origin (pt : Point3D) : ℝ × ℝ × ℝ :=
  (pt.x, pt.y, pt.z)

-- Define the magnitude of a vector in 3D space
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Define the vector sum of a list of vectors
def vector_sum (vs : List (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  vs.foldr (λ v acc, (acc.1 + v.1, acc.2 + v.2, acc.3 + v.3)) (0, 0, 0)

-- Define the theorem to prove the inequality
theorem vector_magnitude_inequality (points : List Point3D) :
  List.sum (points.map (λ pt, magnitude (vector_from_origin pt))) ≤
  Real.sqrt 3 * magnitude (vector_sum (points.map vector_from_origin)) :=
sorry

end vector_magnitude_inequality_l647_647666


namespace evaluate_expression_l647_647433

theorem evaluate_expression:
  (⟪(23:ℚ) / 11 - ⟪(37:ℚ) / 19⟫⟫) / ⟪(35:ℚ) / 11 + ⟪(11 * 19:ℚ) / 37⟫⟫ = 1 / 10 :=
sorry

end evaluate_expression_l647_647433


namespace sec_neg_405_eq_sqrt2_l647_647438

noncomputable def cos : ℝ → ℝ :=
  λ x, sorry -- Assume that cos is the cosine function from real numbers to real numbers

def sec (x : ℝ) : ℝ := 1 / cos x

axiom cos_periodic (x : ℝ) : cos (x + 360) = cos x
axiom cos_even (x : ℝ) : cos (-x) = cos x
axiom cos_45_deg : cos 45 = real.sqrt 2 / 2

theorem sec_neg_405_eq_sqrt2 : sec (-405) = real.sqrt 2 :=
by
  -- Use the conditions given to prove this theorem
  sorry

end sec_neg_405_eq_sqrt2_l647_647438


namespace range_of_x_l647_647888

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x (x : ℝ) : 
  (∃ y z : ℝ, y = 2 * x - 1 ∧ f x > f y ∧ x > 1 / 3 ∧ x < 1) :=
sorry

end range_of_x_l647_647888


namespace probability_of_matching_pair_l647_647821

/-!
# Probability of Selecting a Matching Pair of Shoes

Given:
- 12 pairs of sneakers, each with a 4% probability of being chosen.
- 15 pairs of boots, each with a 3% probability of being chosen.
- 18 pairs of dress shoes, each with a 2% probability of being chosen.

If two shoes are selected from the warehouse without replacement, prove that the probability 
of selecting a matching pair of shoes is 52.26%.
-/

namespace ShoeWarehouse

def prob_sneakers_first : ℝ := 0.48
def prob_sneakers_second : ℝ := 0.44
def prob_boots_first : ℝ := 0.45
def prob_boots_second : ℝ := 0.42
def prob_dress_first : ℝ := 0.36
def prob_dress_second : ℝ := 0.34

theorem probability_of_matching_pair :
  (prob_sneakers_first * prob_sneakers_second) +
  (prob_boots_first * prob_boots_second) +
  (prob_dress_first * prob_dress_second) = 0.5226 :=
sorry

end ShoeWarehouse

end probability_of_matching_pair_l647_647821


namespace distance_K_to_line_CD_l647_647196

-- Define the trapezoid ABCD with given conditions
noncomputable def Trapezoid (A B C D K E : Point) :=
  (ABCD_is_trapezoid : is_trapezoid A B C D) ∧
  (AB_perpendicular_AD : perpendicular A B A D) ∧
  (AB_perpendicular_BC : perpendicular A B B C) ∧
  (circle_tangent_K : circle_tangent (circle_through D E) A B K) ∧
  (circle_tangent_C : circle_tangent (circle_through D E) B C C) ∧
  (K_between_A_B : K ∈ segment A B) ∧
  (AD_eq_48 : distance A D = 48) ∧
  (BC_eq_12 : distance B C = 12)

-- Statement to prove the distance from point K to line CD is 24
theorem distance_K_to_line_CD {A B C D K E : Point} 
  (h : Trapezoid A B C D K E) : 
  distance_to_line K (line_through C D) = 24 :=
sorry

end distance_K_to_line_CD_l647_647196


namespace ShielaDrawingsPerNeighbor_l647_647678

-- Defining our problem using the given conditions:
def ShielaTotalDrawings : ℕ := 54
def ShielaNeighbors : ℕ := 6

-- Mathematically restating the problem:
theorem ShielaDrawingsPerNeighbor : (ShielaTotalDrawings / ShielaNeighbors) = 9 := by
  sorry

end ShielaDrawingsPerNeighbor_l647_647678


namespace P_projection_matrix_P_not_invertible_l647_647986

noncomputable def v : ℝ × ℝ := (4, -1)

noncomputable def norm_v : ℝ := Real.sqrt (4^2 + (-1)^2)

noncomputable def u : ℝ × ℝ := (4 / norm_v, -1 / norm_v)

noncomputable def P : ℝ × ℝ × ℝ × ℝ :=
((4 * 4) / norm_v^2, (4 * -1) / norm_v^2, 
 (-1 * 4) / norm_v^2, (-1 * -1) / norm_v^2)

theorem P_projection_matrix :
  P = (16 / 17, -4 / 17, -4 / 17, 1 / 17) := by
  sorry

theorem P_not_invertible :
  ¬(∃ Q : ℝ × ℝ × ℝ × ℝ, P = Q) := by
  sorry

end P_projection_matrix_P_not_invertible_l647_647986


namespace triangle_angles_l647_647749

-- Define angles and their properties
variables (α β γ : ℝ)

-- Conditions: angles sum up to 180 degrees
axiom angle_sum : α + β + γ = 180

-- Problem statement: Triangle A'B'C' has angles 2α, 2β, 2γ at vertices A', B', C'
theorem triangle_angles (α β γ : ℝ) (h : α + β + γ = 180) :
  ∀ (A' B' C' : Type), (angle A' = 2 * α) ∧ (angle B' = 2 * β) ∧ (angle C' = 2 * γ) → 
  (angle (triangle_mk A' B' C') = α) ∧ 
  (angle (triangle_mk B' C' A') = β) ∧ 
  (angle (triangle_mk C' A' B') = γ) :=
sorry

end triangle_angles_l647_647749


namespace q1_q2_q3_l647_647147

-- Question 1
theorem q1 (kx2_2x_6k : ∀ x, k*x^2 - 2*x + 6*k < 0 ↔ x < -3 ∨ x > -2) (hk : k ≠ 0) : k = -2/5 :=
sorry

-- Question 2
theorem q2 (kx2_2x_6k : ∀ x, k*x^2 - 2*x + 6*k < 0) (hk : k ≠ 0) : k < -Real.sqrt(6)/6 :=
sorry

-- Question 3
theorem q3 (kx2_2x_6k : ∀ x, ¬(k*x^2 - 2*x + 6*k < 0)) (hk : k ≠ 0) : k ≥ Real.sqrt(6)/6 :=
sorry

end q1_q2_q3_l647_647147


namespace find_A_in_triangle_l647_647962

theorem find_A_in_triangle
  (a b : ℝ) (B A : ℝ)
  (h₀ : a = Real.sqrt 3)
  (h₁ : b = Real.sqrt 2)
  (h₂ : B = Real.pi / 4)
  (h₃ : a / Real.sin A = b / Real.sin B) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_A_in_triangle_l647_647962


namespace solve_problem_l647_647099

variable (a b : ℝ)

def condition1 : Prop := a + b = 1
def condition2 : Prop := ab = -6

theorem solve_problem (h1 : condition1 a b) (h2 : condition2 a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 :=
by
  sorry

end solve_problem_l647_647099


namespace sequence_problem_l647_647112

noncomputable def exists_integers (a : ℕ → ℕ) (hbij : Function.Bijective a) : Prop :=
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ

theorem sequence_problem
  (a : ℕ → ℕ) (hbij : Function.Bijective a) : exists_integers a hbij :=
sorry

end sequence_problem_l647_647112


namespace arrang_405220_l647_647943

theorem arrang_405220 : ∃ (n : ℕ), n = 405220 ∧
  ∑ (d : ℕ) in finset.range 10, if (d = 4 ∨ d = 5 ∨ d = 2) then
    nat.choose 5 2 * (nat.factorial 4 / nat.factorial 2) = 120 :=
sorry

end arrang_405220_l647_647943


namespace product_mod_7_l647_647813

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l647_647813


namespace find_hyperbola_equation_l647_647146

noncomputable def hyperbola_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1

theorem find_hyperbola_equation
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1)
  (focal_length : a * a + b * b = 25)
  (asymptote_condition : ∀ x y : ℝ, 2 * a = b ∧ ((2 : ℝ) = ± (y / x))) :
  hyperbola_equation 20 5 := 
sorry

end find_hyperbola_equation_l647_647146


namespace inscribed_radii_sum_l647_647217

variable {A B C D : Type}
variable [cyclic_convex_quadrilateral ABCD : cyclic_convex ABCD]
variable {r_a r_b r_c r_d : ℝ}
variable [inscribed_radii : (r_a inscribed_in_triangle BCD) (r_b inscribed_in_triangle ACD) (r_c inscribed_in_triangle ABD) (r_d inscribed_in_triangle ABC)]

theorem inscribed_radii_sum :
  r_a + r_c = r_b + r_d := 
sorry

end inscribed_radii_sum_l647_647217


namespace problem_l647_647101

theorem problem (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := 
by sorry

end problem_l647_647101


namespace num_points_on_y_axis_isosceles_l647_647188

-- Definitions of the problem conditions
def point_A : ℝ × ℝ := (2, -2)
def is_on_y_axis (P : ℝ × ℝ) := P.1 = 0 -- P is on the y-axis

-- The theorem statement
theorem num_points_on_y_axis_isosceles (P : ℝ × ℝ) :
  (is_on_y_axis P ∧ (∃ A: ℝ × ℝ, A = point_A ∧ is_isosceles_triangle (0, 0) A P)) → P ∈ {p | is_on_y_axis p ∧ p.2 ∈ {y1, y2, y3, y4}} :=
sorry

end num_points_on_y_axis_isosceles_l647_647188


namespace sum_f_values_l647_647889

def f (x : ℝ) : ℝ := Real.sin ((5 * Real.pi / 3) * x + Real.pi / 6) + (3 * x) / (2 * x - 1)

theorem sum_f_values : ∑ k in Finset.range 1008, (f ((2 * k + 1 : ℝ) / 2016) + f ((2 * (1008 - k) - 1 : ℝ) / 2016)) = 1512 :=
  by sorry

end sum_f_values_l647_647889


namespace no_such_function_exists_l647_647832

theorem no_such_function_exists :
  ¬(∃ f : ℕ → ℕ, ∀ n : ℕ, f(f(n)) = n + 1987) :=
by
  sorry

end no_such_function_exists_l647_647832


namespace bacon_suggestion_l647_647279

theorem bacon_suggestion (
    h1 : 218 = number_of_students_suggesting("mashed_potatoes_and_bacon"),
    h2 : 569 = total_number_of_students_suggesting("bacon")
) :
    number_of_students_suggesting("only_bacon") = 351 :=
by
    sorry

end bacon_suggestion_l647_647279


namespace find_point_incenter_l647_647451

open Real

-- Define the incenter condition for a point M inside a triangle ABC
def isIncenter (A B C M : Point) : Prop :=
  (dist M (line.through A B) = dist M (line.through B C)) ∧ (dist M (line.through B C) = dist M (line.through C A))

-- Define the problem statement
theorem find_point_incenter (A B C : Point) :
  ∃ M : Point, isIncenter A B C M :=
by
  sorry

end find_point_incenter_l647_647451


namespace least_number_of_groups_l647_647393

def num_students : ℕ := 24
def max_students_per_group : ℕ := 10

theorem least_number_of_groups : ∃ x, ∀ y, y ≤ max_students_per_group ∧ num_students = x * y → x = 3 := by
  sorry

end least_number_of_groups_l647_647393


namespace equal_circles_intersect_rhombus1_equal_circles_intersect_rhombus2_l647_647721

noncomputable def is_rhombus (points : Fin 4 → Point) : Prop :=
  ∃ (a b c d : ℝ), 
  side_length points 0 1 = a ∧
  side_length points 1 2 = b ∧
  side_length points 2 3 = c ∧
  side_length points 3 0 = d ∧
  a = b ∧ b = c ∧ c = d

variables (O1 O2 A B M N : Point)
variables (r : ℝ)
variables (h : Eq (radius O1 A) r)
variables (k : Eq (radius O2 A) r)

theorem equal_circles_intersect_rhombus1 :
  is_rhombus (λ i, match i with 
                | 0 => O1
                | 1 => A
                | 2 => O2
                | 3 => B
                end) :=
sorry

theorem equal_circles_intersect_rhombus2 :
  is_rhombus (λ i, match i with 
                | 0 => A
                | 1 => M
                | 2 => B
                | 3 => N
                end) :=
sorry

end equal_circles_intersect_rhombus1_equal_circles_intersect_rhombus2_l647_647721


namespace robyn_packs_l647_647275

-- Define the problem conditions
def total_packs : ℕ := 76
def lucy_packs : ℕ := 29

-- Define the goal to be proven
theorem robyn_packs : total_packs - lucy_packs = 47 := 
by
  sorry

end robyn_packs_l647_647275


namespace inequality_div_l647_647164

theorem inequality_div (m n : ℝ) (h : m > n) : (m / 5) > (n / 5) :=
sorry

end inequality_div_l647_647164


namespace price_of_small_painting_l647_647251

variable (S : ℕ)

def noah_last_month_large_sales : ℕ := 8 * 60
def noah_last_month_small_sales : ℕ := 4 * S

def noah_this_month_large_sales : ℕ := 2 * noah_last_month_large_sales
def noah_this_month_small_sales : ℕ := 2 * noah_last_month_small_sales

def total_sales_this_month : ℕ := noah_this_month_large_sales + noah_this_month_small_sales

theorem price_of_small_painting (h : total_sales_this_month = 1200) : S = 30 := 
by {
  sorry
}

end price_of_small_painting_l647_647251


namespace initial_profit_price_reduction_for_target_profit_l647_647759

-- Define given conditions
def purchase_price : ℝ := 280
def initial_selling_price : ℝ := 360
def items_sold_per_month : ℕ := 60
def target_profit : ℝ := 7200
def increment_per_reduced_yuan : ℕ := 5

-- Problem 1: Prove the initial profit per month before the price reduction
theorem initial_profit : 
  items_sold_per_month * (initial_selling_price - purchase_price) = 4800 := by
sorry

-- Problem 2: Prove that reducing the price by 60 yuan achieves the target profit
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, 
    ((initial_selling_price - x) - purchase_price) * (items_sold_per_month + (increment_per_reduced_yuan * x)) = target_profit ∧
    x = 60 := by
sorry

end initial_profit_price_reduction_for_target_profit_l647_647759


namespace twist_cube_distance_l647_647257

variables (α : ℝ)
noncomputable def distance_closer (α : ℝ) : ℝ :=
  1 - real.sqrt (real.cos α)

theorem twist_cube_distance (α : ℝ) (h1 : 0 < α) (h2 : α < real.pi / 2) : distance_closer α = 1 - real.sqrt (real.cos α) :=
by
  sorry

end twist_cube_distance_l647_647257


namespace find_principal_amount_l647_647175

theorem find_principal_amount
  (P r : ℝ) -- P for Principal amount, r for interest rate
  (simple_interest : 800 = P * r / 100 * 2) -- Condition 1: Simple Interest Formula
  (compound_interest : 820 = P * ((1 + r / 100) ^ 2 - 1)) -- Condition 2: Compound Interest Formula
  : P = 8000 := 
sorry

end find_principal_amount_l647_647175


namespace min_k_satisfies_inequality_l647_647109

variable (n : ℕ) (a : ℕ → ℝ) [Fact (n ≥ 3)] (s : ℝ)
noncomputable def k_min : ℝ := ((n-1 : ℝ) / (n-2 : ℝ))^2

theorem min_k_satisfies_inequality (n_ge_3 : n ≥ 3) 
  (positive : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0)
  (s_def : s = ∑ i in range n, a i)
  (a_neq_s : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≠ s) :
  (∑ i in range (n-1), (a i) / (s - (a i)) + ((k_min n) * (a (n-1))) / (s - (a (n-1)))) 
  ≥ (n-1 : ℝ) / (n-2 : ℝ) :=
sorry

end min_k_satisfies_inequality_l647_647109


namespace bridget_apples_l647_647030

theorem bridget_apples:
  ∃ (x : ℕ), 
  (x / 2 - (x / 2) / 3) = 5 :=
begin
  use 15,
  have h1 : (15 / 2) = 7,
  have h2 : (7 / 3) = 2,
  have h3 : 7 - 2 = 5,
  exact h3,
end

end bridget_apples_l647_647030


namespace area_diff_l647_647292

open Real

-- Definitions of the conditions based on circumferences
def C1 : ℝ := 660
def C2 : ℝ := 704
def C3 : ℝ := 750

-- Define the formula for the radius based on circumference
def radius (C : ℝ) : ℝ := C / (2 * π)

-- Define the formula for the area based on radius
def area (r : ℝ) : ℝ := π * r^2

-- Define the radii based on the given circumferences
def r1 : ℝ := radius C1
def r2 : ℝ := radius C2
def r3 : ℝ := radius C3

-- Define the areas based on the radii
def A1 : ℝ := area r1
def A3 : ℝ := area r3

-- The Lean theorem to prove the difference in areas
theorem area_diff : abs (A3 - A1) ≈ 10114.327 :=
sorry

end area_diff_l647_647292


namespace number_of_even_divisors_factorial_9_l647_647566

-- Definitions for conditions
def factorial_9 : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

def prime_factorization_factorial_9 (n : ℕ) : Prop :=
  n = 2^7 * 3^4 * 5 * 7

-- Main theorem statement
theorem number_of_even_divisors_factorial_9 : ∃ d, d = 140 ∧ ∀ r : ℕ,
  even r ∧ r ∣ factorial_9 → (r = prime_factorization_factorial_9 r) :=
begin
  sorry,
end

end number_of_even_divisors_factorial_9_l647_647566


namespace quadrilateral_area_l647_647072

variable (d : ℝ) (o₁ : ℝ) (o₂ : ℝ)

theorem quadrilateral_area (h₁ : d = 28) (h₂ : o₁ = 8) (h₃ : o₂ = 2) : 
  (1 / 2 * d * o₁) + (1 / 2 * d * o₂) = 140 := 
  by
    rw [h₁, h₂, h₃]
    sorry

end quadrilateral_area_l647_647072


namespace find_f_neg2_l647_647513

-- Define the function f and the given conditions
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 4

theorem find_f_neg2 (a b : ℝ) (h₁ : f 2 a b = 6) : f (-2) a b = -14 :=
by
  sorry

end find_f_neg2_l647_647513


namespace eccentricity_of_parametric_curve_l647_647696

noncomputable def parametric_curve_x (t : ℝ) : ℝ := t + (1 / t)
noncomputable def parametric_curve_y (t : ℝ) : ℝ := t - (1 / t)

theorem eccentricity_of_parametric_curve : 
  (∀ t : ℝ, t ≠ 0 → parametric_curve_x t = t + 1 / t ∧ parametric_curve_y t = t - 1 / t) →
  ∃ e : ℝ, e = sqrt 2 :=
by
  intros h
  -- Proof would go here
  sorry

end eccentricity_of_parametric_curve_l647_647696


namespace product_of_divisors_60_l647_647800

-- Definitions based on the conditions in the problem
def num := 60
def prime_factors := (2, 2) :: (3, 1) :: (5, 1) :: []

-- Function to calculate the product of all the divisors of a number
noncomputable def product_of_divisors (n : ℕ) : ℕ :=
  let divisors := Nat.divisors n
  List.prod divisors

-- Lean 4 statement stating the proof problem
theorem product_of_divisors_60 : product_of_divisors num = 46656000000000 := by
  -- Using hsorry as we are not providing the proof steps
  sorry

end product_of_divisors_60_l647_647800


namespace alex_bought_3_bags_of_chips_l647_647086

theorem alex_bought_3_bags_of_chips (x : ℝ) : 
    (1 * x + 5 + 73) / x = 27 → x = 3 := by sorry

end alex_bought_3_bags_of_chips_l647_647086


namespace shortest_paths_rectangle_l647_647062

theorem shortest_paths_rectangle (k : ℕ) :
  let A := (0, 0) in
  let B := (1, 0) in
  let D := (0, k) in
  let C := (1, k) in
  ∃ (paths_along_AD paths_along_AB : ℕ),
    paths_along_AD = k * paths_along_AB := 
sorry

end shortest_paths_rectangle_l647_647062


namespace equal_cubic_values_l647_647526

theorem equal_cubic_values (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 3) 
  (h3 : a * b * c + b * c * d + c * d * a + d * a * b = 1) :
  a * (1 - a)^3 = b * (1 - b)^3 ∧ 
  b * (1 - b)^3 = c * (1 - c)^3 ∧ 
  c * (1 - c)^3 = d * (1 - d)^3 :=
sorry

end equal_cubic_values_l647_647526


namespace problem_1_problem_2_problem_3_l647_647866

noncomputable def seq (n : ℕ) : ℝ :=
  if h : n = 0 then 1
  else if h : n = 1 then 1
  else ((@seq (Nat.pred (Nat.pred n)) + 1) / (12 * @seq (Nat.pred (Nat.pred n))))

def positive_term (n : ℕ) : Prop :=
  seq n > 0

theorem problem_1 : ∀ n : ℕ, 0 < n → @seq (2 * n + 1) < @seq (2 * n - 1) := sorry

theorem problem_2 : ∀ n : ℕ, 1 / 6 ≤ @seq n ∧ @seq n ≤ 1 := sorry

noncomputable def S (n : ℕ) : ℝ :=
  (List.range n).sum (λ i, |seq (i + 2) - seq (i + 1)|)

theorem problem_3 : ∀ n : ℕ, 0 < n → @S n < 6 := sorry

end problem_1_problem_2_problem_3_l647_647866


namespace cot_arccot_sum_l647_647503

-- Define the values according to the conditions
def x : ℝ := 5
def y : ℝ := 11
def z : ℝ := 17
def w : ℝ := 23

-- The main theorem to prove
theorem cot_arccot_sum :
  cot (arccot x + arccot y - arccot z + arccot w) = 1731 / 547 :=
by
  -- Proof is omitted
  sorry

end cot_arccot_sum_l647_647503


namespace sum_binom_odd_l647_647836

theorem sum_binom_odd :
  (∑ k in finset.range 25, (-1)^k * (nat.choose 50 (2*k+1))) = 2^25 * real.sqrt 2 :=
sorry

end sum_binom_odd_l647_647836


namespace combined_selling_price_correct_l647_647773

noncomputable def cost_A : ℝ := 500
noncomputable def cost_B : ℝ := 800
noncomputable def profit_A_perc : ℝ := 0.10
noncomputable def profit_B_perc : ℝ := 0.15
noncomputable def tax_perc : ℝ := 0.05
noncomputable def packaging_fee : ℝ := 50

-- Calculating selling prices before tax and fees
noncomputable def selling_price_A_before_tax_fees : ℝ := cost_A * (1 + profit_A_perc)
noncomputable def selling_price_B_before_tax_fees : ℝ := cost_B * (1 + profit_B_perc)

-- Calculating taxes
noncomputable def tax_A : ℝ := selling_price_A_before_tax_fees * tax_perc
noncomputable def tax_B : ℝ := selling_price_B_before_tax_fees * tax_perc

-- Adding tax to selling prices
noncomputable def selling_price_A_incl_tax : ℝ := selling_price_A_before_tax_fees + tax_A
noncomputable def selling_price_B_incl_tax : ℝ := selling_price_B_before_tax_fees + tax_B

-- Adding packaging and shipping fees
noncomputable def final_selling_price_A : ℝ := selling_price_A_incl_tax + packaging_fee
noncomputable def final_selling_price_B : ℝ := selling_price_B_incl_tax + packaging_fee

-- Combined selling price
noncomputable def combined_selling_price : ℝ := final_selling_price_A + final_selling_price_B

theorem combined_selling_price_correct : 
  combined_selling_price = 1643.5 := by
  sorry

end combined_selling_price_correct_l647_647773


namespace find_n_mod_10_l647_647479

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l647_647479


namespace product_modulo_seven_l647_647808

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l647_647808


namespace evaluate_product_l647_647432

open Complex

noncomputable def w : ℂ := exp (2 * π * I / 13)

theorem evaluate_product : (List.prod (List.map (λ (k : ℕ), 2 - w^k) (List.range 12))) = 8191 := by
  sorry

end evaluate_product_l647_647432


namespace graph_passes_through_quadrants_l647_647897

theorem graph_passes_through_quadrants (k : ℝ) (h : k < 0) :
  ∀ (x y : ℝ), (y = k * x - k) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_passes_through_quadrants_l647_647897


namespace count_solutions_eq_2_in_interval_l647_647570

open Real

noncomputable def f (θ : ℝ) : ℝ := 2 - 4 * tan θ + 3 * (cot (2 * θ))

theorem count_solutions_eq_2_in_interval :
  ∃! (θ₁ θ₂ : ℝ), (0 < θ₁ ∧ θ₁ < π) ∧ (0 < θ₂ ∧ θ₂ < π) ∧ f θ₁ = 0 ∧ f θ₂ = 0 ∧ θ₁ ≠ θ₂ :=
sorry

end count_solutions_eq_2_in_interval_l647_647570


namespace solve_equation_l647_647467

noncomputable def equation_solution (x : ℝ) : Prop :=
  (real.cbrt (3 - x) + real.sqrt (x + 1) = 2)

theorem solve_equation :
  ∃ x : ℝ, equation_solution x ∧ 
  (x = 3 
   ∨ x = 3 - (-(1 : ℝ) + real.sqrt 17) / 2 ^ 3 
   ∨ x = 3 - (-(1 : ℝ) - real.sqrt 17) / 2 ^ 3) := 
by 
  sorry

end solve_equation_l647_647467


namespace num_even_divisors_nine_factorial_l647_647564

/-- The prime factorization of \(9!\). -/
def primeFactorization_nine_factorial : (ℕ → ℕ) := 
  λ x, if x = 2 then 7 else if x = 3 then 4 else if x = 5 then 1 else if x = 7 then 1 else 0

theorem num_even_divisors_nine_factorial :
  let even_divisors_count := (7 * 5 * 2 * 2) in
  even_divisors_count = 140 :=
by 
  sorry

end num_even_divisors_nine_factorial_l647_647564


namespace value_of_x2_plus_y2_l647_647994

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : x^3 = 3 * y^2 * x + 5 - real.sqrt 7) (h2 : y^3 = 3 * x^2 * y + 5 + real.sqrt 7) : x^2 + y^2 = 4 :=
by
  sorry

end value_of_x2_plus_y2_l647_647994


namespace frood_game_solution_l647_647193

theorem frood_game_solution :
  ∃ n : ℕ, (∀ m : ℕ, m < n → m * (m + 1) ≤ 8 * m) ∧ n * (n + 1) > 8 * n :=
by
  have drop_score : ℕ → ℕ := λ n, n * (n + 1)
  have eat_score : ℕ → ℕ := λ n, 8 * n
  existsi 8
  split
  intro m
  intro h
  by_cases h2 : m < 8
  linarith
  linarith
  linarith

end frood_game_solution_l647_647193


namespace balls_into_boxes_l647_647910

theorem balls_into_boxes : 
  let distinguishable_balls : finset ℕ := {1, 2, 3, 4, 5, 6} -- representing 6 distinguishable balls
  let boxes : finset ℕ := {1, 2, 3} -- representing 3 indistinguishable boxes
  ∃ ways : ℕ, ways = 67 := 
sorry

end balls_into_boxes_l647_647910


namespace johns_profit_l647_647612

theorem johns_profit
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (profit : ℕ) :
  trees_chopped = 30 →
  planks_per_tree = 25 →
  planks_per_table = 15 →
  price_per_table = 300 →
  labor_cost = 3000 →
  profit = 12000 :=
begin
  sorry
end

end johns_profit_l647_647612


namespace area_of_circle_7_is_pi_over_8_l647_647284

theorem area_of_circle_7_is_pi_over_8 :
  let side_length (n : ℕ) : ℝ := 4 * ((1 / Real.sqrt 2) ^ (n - 1)) in
  let radius (n : ℕ) : ℝ := (side_length n) / Real.sqrt 2 in
  let area (n : ℕ) : ℝ := Real.pi * (radius n) ^ 2 in
  area 7 = Real.pi / 8 :=
by sorry

end area_of_circle_7_is_pi_over_8_l647_647284


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647817

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647817


namespace distance_between_parallel_lines_l647_647536

variable (m : Real)
axiom parallel_lines (h_parallel : mx - 4 * y + 3 = 0) : m = 2

theorem distance_between_parallel_lines (h_parallel : mx - 4 * y + 3 = 0) :
  let l1 := (2 * x - 4 * y + 8 = 0)
  let l2 := (2 * x - 4 * y + 3 = 0)
  let A := 2
  let B := -4
  let C1 := 8
  let C2 := 3
  let distance := |C1 - C2| / Real.sqrt (A^2 + B^2)
  distance = Real.sqrt 5 / 2 :=
by
  sorry

end distance_between_parallel_lines_l647_647536


namespace alice_students_count_l647_647395

variable (S : ℕ)
variable (students_with_own_vests := 0.20 * S)
variable (students_needing_vests := 0.80 * S)
variable (instructors : ℕ := 10)
variable (life_vests_on_hand : ℕ := 20)
variable (additional_life_vests_needed : ℕ := 22)
variable (total_life_vests_needed := life_vests_on_hand + additional_life_vests_needed)
variable (life_vests_needed_for_instructors := instructors)
variable (life_vests_needed_for_students := total_life_vests_needed - life_vests_needed_for_instructors)

theorem alice_students_count : S = 40 :=
by
  -- proof steps would go here
  sorry

end alice_students_count_l647_647395


namespace find_sinD_l647_647170

variable (DE DF : ℝ)

-- Conditions
def area_of_triangle (DE DF : ℝ) (sinD : ℝ) : Prop :=
  1 / 2 * DE * DF * sinD = 72

def geometric_mean (DE DF : ℝ) : Prop :=
  Real.sqrt (DE * DF) = 15

theorem find_sinD (DE DF sinD : ℝ) (h1 : area_of_triangle DE DF sinD) (h2 : geometric_mean DE DF) :
  sinD = 16 / 25 :=
by 
  -- Proof goes here
  sorry

end find_sinD_l647_647170


namespace orthocenter_angle_YHZ_l647_647604

theorem orthocenter_angle_YHZ {X Y Z H: Type*} 
  (angle_XYZ : ℝ) (angle_XZY : ℝ) (angle_YXZ : ℝ)
  (H_orthocenter : true) 
  (h1 : angle_XYZ = 65)
  (h2 : angle_XZY = 15)
  (h3 : angle_YXZ = 100) :
  ∠ Y H Z = 80 := 
by
  sorry

end orthocenter_angle_YHZ_l647_647604


namespace product_mod_7_l647_647815

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l647_647815


namespace determine_slope_l647_647561

theorem determine_slope (a : ℝ) : 
  let l1 := λ x : ℝ, a * x - 2 in
  let l2 := λ x : ℝ, (a + 2) * x + 1 in
  (∀ x1 x2 : ℝ, x1 ≠ x2 ↔ (l1 x1 - l1 x2) / (x1 - x2) * (l2 x1 - l2 x2) / (x1 - x2) = -1) → 
  a = -1 :=
by
  -- proof or steps are not required, use sorry
  sorry

end determine_slope_l647_647561


namespace solve_equation_l647_647281

theorem solve_equation : (0.001^(-3)) - ((7 / 8)^0) + (16^(3 / 4)) + ((real.sqrt 2 * real.cbrt 3)^6) = 89 :=
by
  sorry

end solve_equation_l647_647281


namespace count_integers_l647_647908

theorem count_integers (n : ℕ) (h : n = 33000) :
  ∃ k : ℕ, k = 1600 ∧
  (∀ x, 1 ≤ x ∧ x ≤ n → (x % 11 = 0 → (x % 3 ≠ 0 ∧ x % 5 ≠ 0) → x ≤ x)) :=
by 
  sorry

end count_integers_l647_647908


namespace probability_A_or_not_B_l647_647185

open ProbabilityTheory

-- Definitions based on conditions in a)
def event_A (d : ℕ) : Prop := d ≤ 3
def event_B (d : ℕ) : Prop := d < 5

-- The main proof statement
theorem probability_A_or_not_B (d : ℕ) (uniform_die : ∀ d, d ∈ Finset.range 6 → Prob) : 
  (Prob (λ d, event_A d ∨ ¬ (event_B d))) = 5/6 :=
sorry

end probability_A_or_not_B_l647_647185


namespace find_lambda_l647_647871

-- Definition of point P and midpoint M
structure Point where
  x : ℝ
  y : ℝ

def P := Point.mk 1 1
def M := Point.mk (-1) 2

-- Function to calculate vector PQ
def vector_PQ (P Q : Point) : Point :=
  Point.mk (Q.x - P.x) (Q.y - P.y)

-- Midpoint condition: If M is the midpoint of PQ, then Q can be derived
def Q (P M : Point) : Point :=
  Point.mk (2 * M.x - P.x) (2 * M.y - P.y)

-- Vector a with a given λ
def a (λ : ℝ) : Point :=
  Point.mk λ 1

-- Dot product of two vectors
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Vector PQ calculated from P and Q given the midpoint condition
noncomputable def PQ : Point := vector_PQ P (Q P M)

-- Lean statement: λ such that PQ is perpendicular to a(λ)
theorem find_lambda : ∃ λ : ℝ, dot_product PQ (a λ) = 0 ∧ λ = 1/2 := by
  sorry

end find_lambda_l647_647871


namespace exists_n_distinct_positive_integers_l647_647270

theorem exists_n_distinct_positive_integers (n : ℕ) (hn : n ≥ 2) :
  ∃ a : fin n → ℕ, (∀ i j : fin n, i < j → a i ≠ a j) ∧ (∀ i j : fin n, i < j → (a i - a j) ∣ (a i + a j)) :=
sorry

end exists_n_distinct_positive_integers_l647_647270


namespace bridge_length_l647_647333

noncomputable def train_A_length : ℝ := 300
noncomputable def train_A_speed_kmph : ℝ := 90
noncomputable def train_B_length : ℝ := 400
noncomputable def train_B_speed_kmph : ℝ := 100

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def train_A_speed_mps : ℝ :=
  kmph_to_mps train_A_speed_kmph

noncomputable def train_B_speed_mps : ℝ :=
  kmph_to_mps train_B_speed_kmph

noncomputable def time_to_clear (train_length bridge_length speed : ℝ) : ℝ :=
  (train_length + bridge_length) / speed

theorem bridge_length:
  ∃ L: ℝ, abs(L - 599.28) < 0.01 ∧
    time_to_clear train_A_length L train_A_speed_mps =
    time_to_clear train_B_length L train_B_speed_mps :=
  by
  sorry

end bridge_length_l647_647333


namespace prime_palindrome_is_11_l647_647053

-- Mathematical conditions
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def has_even_number_of_digits (n : ℕ) : Prop :=
  n.digits 10.length % 2 = 0

-- Final theorem statement
theorem prime_palindrome_is_11 (p : ℕ) (hp : nat.prime p) (hpal : is_palindrome p) (heven : has_even_number_of_digits p) : p = 11 :=
by
  sorry

end prime_palindrome_is_11_l647_647053


namespace competition_votes_l647_647590

/-- 
In a revival competition, if B's number of votes is 20/21 of A's, and B wins by
gaining at least 4 votes more than A, prove the possible valid votes counts.
-/
theorem competition_votes (x : ℕ) 
  (hx : x > 0) 
  (hx_mod_21 : x % 21 = 0) 
  (hB_wins : ∀ b : ℕ, b = (20 * x / 21) + 4 → b > x - 4) :
  (x = 147 ∧ 140 = 20 * x / 21) ∨ (x = 126 ∧ 120 = 20 * x / 21) := 
by 
  sorry

end competition_votes_l647_647590


namespace inscribed_squares_ratio_l647_647390

theorem inscribed_squares_ratio (x y : ℝ) (h1 : ∃ (x : ℝ), x * (13 * 12 + 13 * 5 - 5 * 12) = 60) 
  (h2 : ∃ (y : ℝ), 30 * y = 13 ^ 2) :
  x / y = 1800 / 2863 := 
sorry

end inscribed_squares_ratio_l647_647390


namespace sum_of_coefficients_eq_64_l647_647192

theorem sum_of_coefficients_eq_64 : 
  let f (x : ℝ) : ℝ := (3 * x - 1 / real.sqrt x) ^ 6 
  in f 1 = 64 :=
by
  sorry

end sum_of_coefficients_eq_64_l647_647192


namespace range_of_m_eq_real_iff_l647_647556

noncomputable def range_of_m (m : ℝ) : set ℝ := 
  { x : ℝ | m * x^2 + 2 * m * x + 1 > 0 }

theorem range_of_m_eq_real_iff :
  (range_of_m m = set.univ) ↔ (0 ≤ m ∧ m < 1) :=
begin
  sorry
end

end range_of_m_eq_real_iff_l647_647556


namespace divisible_sum_l647_647862

theorem divisible_sum (k : ℕ) (n : ℕ) (h : n = 2^(k-1)) : 
  ∀ (S : Finset ℕ), S.card = 2*n - 1 → ∃ T ⊆ S, T.card = n ∧ T.sum id % n = 0 :=
by
  sorry

end divisible_sum_l647_647862


namespace probability_of_two_adjacent_among_abc_l647_647717

theorem probability_of_two_adjacent_among_abc:
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C → 
  (20.choose 3) = 1140 ∧
  (20  * (20 - 4)) + 20 = 340 →
  (340 / 1140 = 17 / 57) :=
by
  sorry

end probability_of_two_adjacent_among_abc_l647_647717


namespace line_passes_through_fixed_point_l647_647116

theorem line_passes_through_fixed_point : 
  (∀ (x y : ℝ), 3 * x^2 + 4 * y^2 = 12 → 
                ∃ (k t : ℝ), (∀ (x y : ℝ), y = k * x + t ∧ 3 * x^2 + 4 * y^2 = 12 → 
                              let (x1 x2 y1 y2 : ℝ) := (some x1 x2 y1 y2) in
                              (circle_intersects_vertex (0, 2)) →
                              ∃ (x0 y0 : ℝ), y0 = 2 / 7 ∧ x0 = 0)) := sorry

-- Additional required definitions
def circle_intersects_vertex (p : ℝ × ℝ) : Prop := sorry

end line_passes_through_fixed_point_l647_647116


namespace profit_sale_price_l647_647705

variables (CP SP_profit : ℝ)

-- Conditions
def condition1 := SP_profit - CP = CP - 448
def condition2 := 0.55 * CP = 992

-- The expected sale price for making a 55% profit
def expected_SP_profit := 2792

-- The statement to be proved
theorem profit_sale_price (h1 : condition1) (h2 : condition2) : SP_profit = expected_SP_profit :=
sorry

end profit_sale_price_l647_647705


namespace question1_question2_l647_647890

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - abs (2 * x - 1)

theorem question1 (x : ℝ) :
  ∀ a, a = 2 → (f x 2 + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2) := by
sorry

theorem question2 (a : ℝ) :
  (∀ x, 1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end question1_question2_l647_647890


namespace time_without_walkway_l647_647000

variables (v_p v_w : ℝ) (L : ℝ) (t_with_walkway t_against_walkway : ℝ)

-- Define the conditions
def conditions := (L = 90) ∧ (t_with_walkway = 30) ∧ (t_against_walkway = 120) ∧
                  (v_p + v_w) * t_with_walkway = L ∧ (v_p - v_w) * t_against_walkway = L

-- Define the theorem to prove
theorem time_without_walkway (h : conditions v_p v_w L t_with_walkway t_against_walkway) :
  (90 / v_p = 48) :=
by {
  -- Extract the conditions and state the proof obligation
  rcases h with ⟨hL, h_tw, h_ta, h_eq1, h_eq2⟩,
  have h1 : v_p + v_w = 3, from (hL.symm ▸ h_tw.symm ▸ h_eq1) ▸ rfl,
  have h2 : v_p - v_w = 3 / 4, from (hL.symm ▸ h_ta.symm ▸ h_eq2) ▸ rfl,
  sorry
}

end time_without_walkway_l647_647000


namespace find_n_l647_647489

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l647_647489


namespace find_time_for_products_maximize_salary_l647_647349

-- Assume the conditions and definitions based on the given problem
variables (x y a : ℝ)

-- Condition 1: Time to produce 6 type A and 4 type B products is 170 minutes
axiom cond1 : 6 * x + 4 * y = 170

-- Condition 2: Time to produce 10 type A and 10 type B products is 350 minutes
axiom cond2 : 10 * x + 10 * y = 350


-- Question 1: Validating the time to produce one type A product and one type B product
theorem find_time_for_products : 
  x = 15 ∧ y = 20 := by
  sorry

-- Variables for calculation of Zhang's daily salary
variables (m : ℕ) (base_salary : ℝ := 100) (daily_work: ℝ := 480)

-- Conditions for the piece-rate wages
variables (a_condition: 2 < a ∧ a < 3) 
variables (num_products: m + (28 - m) = 28)

-- Question 2: Finding optimal production plan to maximize daily salary
theorem maximize_salary :
  (2 < a ∧ a < 2.5) → m = 16 ∨ 
  (a = 2.5) → true ∨
  (2.5 < a ∧ a < 3) → m = 28 := by
  sorry

end find_time_for_products_maximize_salary_l647_647349


namespace bus_stop_time_per_hour_l647_647066

variable (dist : ℝ) (speed_without_stoppings speed_with_stoppings : ℝ)
variable (time_without_stoppings time_with_stoppings time_stopping_per_hour : ℝ)

-- Given conditions
noncomputable def average_speed_without_stoppings := speed_without_stoppings = 80
noncomputable def average_speed_with_stoppings := speed_with_stoppings = 40

-- Prove the bus stop time per hour
theorem bus_stop_time_per_hour (h1 : average_speed_without_stoppings) (h2 : average_speed_with_stoppings)
: time_stopping_per_hour = 30 := 
  sorry

end bus_stop_time_per_hour_l647_647066


namespace correct_conclusions_l647_647785

def f (x : ℝ) : ℝ :=
  x^2 * Real.exp x

theorem correct_conclusions :
  (∀ x, -2 < x ∧ x < 0 → derivative f x < 0) ∧
  (¬(∀ x, derivative f x ≠ 0)) ∧
  (∀ y, y ≠ f (0) → ∃ y = f (0)) ∧
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 - x1 + 2012 = 0 ∧ f x2 - x2 + 2012 = 0) := 
sorry

end correct_conclusions_l647_647785


namespace triangle_inequality_l647_647268

theorem triangle_inequality 
  (a b c R : ℝ) 
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a) 
  (hR : R = (a * b * c) / (4 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))) : 
  a^2 + b^2 + c^2 ≤ 9 * R^2 :=
by 
  sorry

end triangle_inequality_l647_647268


namespace range_area_of_quadrilateral_OACB_l647_647197

theorem range_area_of_quadrilateral_OACB (A B C a b c : ℝ) (h1 : b = c) (h2 : sin B + sin (A - C) = sin (2 * A))
  (O OA OB : Point) (h3 : distance O A = 2) (h4 : distance O B = 1) :
  let S := area_of_quadrilateral O A C B in
  ∃ S_range, S ∈ (interval_left_open (sqrt(3)/4) (5*sqrt(3)/4 + 2)) :=
sorry

end range_area_of_quadrilateral_OACB_l647_647197


namespace range_of_x_l647_647539

noncomputable def g (x : ℝ) : ℝ :=
if h : x < 0 then log (1 - x) else log (1 + x)  -- defining g(x) as an even function using the given conditions

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 else g x  -- defining f(x) according to the given cases

theorem range_of_x (x : ℝ) : f (2 - x^2) > f x → x ∈ Ioo (-2 : ℝ) 1 :=  -- statement of the problem in Lean 4: Ioo represents open interval
by 
  sorry -- proof goes here

end range_of_x_l647_647539


namespace total_cards_received_l647_647248

theorem total_cards_received (cards_while_in_hospital : ℕ) (cards_after_hospital : ℕ) : 
  cards_while_in_hospital = 403 → 
  cards_after_hospital = 287 → 
  cards_while_in_hospital + cards_after_hospital = 690 := 
by
  intros h1 h2
  rw [h1, h2]
  rfl

end total_cards_received_l647_647248


namespace count_prime_sum_112_l647_647341

noncomputable def primeSum (primes : List ℕ) : ℕ :=
  if H : ∀ p ∈ primes, Nat.Prime p ∧ p > 10 then primes.sum else 0

theorem count_prime_sum_112 :
  ∃ (primes : List ℕ), primeSum primes = 112 ∧ primes.length = 6 := by
  sorry

end count_prime_sum_112_l647_647341


namespace problem_conditions_l647_647297

def f (x : ℝ) : ℝ := sin (4 * x + π / 2)

theorem problem_conditions :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x : ℝ, f (π / 4 - x) = f (π / 4 + x)) :=
by
  sorry

end problem_conditions_l647_647297


namespace find_a_l647_647070

def satisfies_condition (a : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  (x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0

def polynomial (a : ℝ) : Polynomial ℝ :=
  Polynomial.C a + Polynomial.X * (Polynomial.C a + Polynomial.X * (Polynomial.C (-6) + Polynomial.X))

theorem find_a (a : ℝ) (x1 x2 x3 : ℝ) :
  polynomial a = Polynomial.X^3 - 6*Polynomial.X^2 + a*Polynomial.X + a →
  satisfies_condition a x1 x2 x3 →
  a = 9 :=
sorry

end find_a_l647_647070


namespace meet_days_l647_647952

-- Definition of conditions
def person_a_days : ℕ := 5
def person_b_days : ℕ := 7
def person_b_early_departure : ℕ := 2

-- Definition of the number of days after A's start that they meet
variable {x : ℕ}

-- Statement to be proven
theorem meet_days (x : ℕ) : (x + 2 : ℚ) / person_b_days + x / person_a_days = 1 := sorry

end meet_days_l647_647952


namespace bc_length_l647_647699

theorem bc_length (A B C D E F : Type)
  (circumcircle_ABC : ∀ P, P ≠ E → P ∈ circle A B C → P = circle_intersection A B C E)
  (circumcircle_ADE: ∀ Q, Q ≠ F → Q ∈ circle A D E → Q = circle_intersection A D E F)
  (angle_bisector_CD : is_angle_bisector CD A B C)
  (AC_eq_b : length A C = b)
  (AF_eq_a : length A F = a) :
  length B C = a + b ∨ length B C = b - a :=
sorry

end bc_length_l647_647699


namespace sin_value_equiv_l647_647123

theorem sin_value_equiv 
  (α : ℝ) 
  (hcos : cos (α - 2 * π / 9) = - sqrt 7 / 4) 
  (hα : π / 2 < α ∧ α < π) :
  sin (α + 7 * π / 9) = -3 / 4 :=
sorry

end sin_value_equiv_l647_647123


namespace value_of_expression_l647_647127

variable {a b m n x : ℝ}

def opposite (a b : ℝ) : Prop := a = -b
def reciprocal (m n : ℝ) : Prop := m * n = 1
def distance_to_2 (x : ℝ) : Prop := abs (x - 2) = 3

theorem value_of_expression (h1 : opposite a b) (h2 : reciprocal m n) (h3 : distance_to_2 x) :
  (a + b - m * n) * x + (a + b)^2022 + (- m * n)^2023 = 
  if x = 5 then -6 else if x = -1 then 0 else sorry :=
by
  sorry

end value_of_expression_l647_647127


namespace problem_l647_647102

theorem problem (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := 
by sorry

end problem_l647_647102


namespace collinear_sufficient_not_necessary_for_coplanar_l647_647938

theorem collinear_sufficient_not_necessary_for_coplanar 
  (P1 P2 P3 P4 : ℝ × ℝ × ℝ) :
  (∃ l : ℝ → ℝ × ℝ × ℝ, ∀ t, l t = (P1.1 * (1 - t) + P2.1 * t, P1.2 * (1 - t) + P2.2 * t, P1.3 * (1 - t) + P2.3 * t) ∨ 
   l t = (P2.1 * (1 - t) + P3.1 * t, P2.2 * (1 - t) + P3.2 * t, P2.3 * (1 - t) + P3.3 * t)) → 
  ∃ π : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → Prop, π P1 P2 P3 ∧ π P1 P2 P4 :=
sorry

end collinear_sufficient_not_necessary_for_coplanar_l647_647938


namespace line_eq_center_line_eq_bisected_chord_chord_length_45_slope_l647_647105

def circleC_eq : ℝ → ℝ → Prop := λ x y, (x - 1) ^ 2 + y ^ 2 = 9

def pointP : ℝ × ℝ := (2, 2)

def centerC : ℝ × ℝ := (1, 0)

def radiusC : ℝ := 3

/- Question 1 -/
theorem line_eq_center (l : ℝ → ℝ) (h : ∀ x, l x = 2 * (x - 1)) :
  ∃ a b c, a * x + b * y + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -2 :=
sorry

/- Question 2 -/
theorem line_eq_bisected_chord (l : ℝ → ℝ) (h : ∀ x, l x = -1 / 2 * (x - 2) + 2) :
  ∃ a b c, a * x + b * y + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -6 :=
sorry

/- Question 3 -/
theorem chord_length_45_slope (AB : ℝ) (h : AB = Float.sqrt 34) :
  ∃ l, (∀ x, l x = x - 2 + 2 - y ∧ chord_length = Float.sqrt 34 :=
sorry

end line_eq_center_line_eq_bisected_chord_chord_length_45_slope_l647_647105


namespace triangle_solvability_condition_l647_647040

theorem triangle_solvability_condition
  (r ϖ : ℝ) (α : ℝ) :
  ϖ ≤ 2 * r * real.sin (α / 2) * (1 - real.sin (α / 2)) :=
sorry

end triangle_solvability_condition_l647_647040


namespace quadratic_polynomial_l647_647544

theorem quadratic_polynomial (x y : ℝ) (hx : x + y = 12) (hy : x * (3 * y) = 108) : 
  (t : ℝ) → t^2 - 12 * t + 36 = 0 :=
by 
  sorry

end quadratic_polynomial_l647_647544


namespace simplify_and_evaluate_l647_647278

theorem simplify_and_evaluate :
  ∀ (a b : ℚ), a = 2 → b = -1/2 → (a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l647_647278


namespace a_squared_divisible_by_b_l647_647869

theorem a_squared_divisible_by_b (a b : ℕ) (h1 : a < 1000) (h2 : b > 0) 
    (h3 : ∃ k, a ^ 21 = b ^ 10 * k) : ∃ m, a ^ 2 = b * m := 
by
  sorry

end a_squared_divisible_by_b_l647_647869


namespace triangle_inequality_triangle_equality_l647_647003

theorem triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) : a + b ≤ c * sqrt 2 :=
by
  sorry

theorem triangle_equality (a b c : ℝ) (h : a^2 + b^2 = c^2) : a + b = c * sqrt 2 ↔ a = b :=
by
  sorry

end triangle_inequality_triangle_equality_l647_647003


namespace max_value_of_quadratic_l647_647343

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -x^2 - 8 * x + 16

theorem max_value_of_quadratic : ∃ x : ℝ, x = -4 ∧ ∀ y : ℝ, quadratic(x) ≥ quadratic(y) :=
begin
  -- The proof is omitted
  sorry
end

end max_value_of_quadratic_l647_647343


namespace non_pizza_eaters_count_l647_647029

theorem non_pizza_eaters_count
    (teachers : ℕ)
    (staff : ℕ)
    (teachers_ate_pizza_fraction : ℚ)
    (staff_ate_pizza_fraction : ℚ)
    (teachers_not_eating_pizza : ℕ)
    (staff_not_eating_pizza : ℕ) :
    teachers = 30 →
    staff = 45 →
    teachers_ate_pizza_fraction = 2 / 3 →
    staff_ate_pizza_fraction = 4 / 5 →
    teachers_not_eating_pizza = teachers - Nat.ceil (teachers_ate_pizza_fraction * teachers) →
    staff_not_eating_pizza = staff - Nat.ceil (staff_ate_pizza_fraction * staff) →
    teachers_not_eating_pizza + staff_not_eating_pizza = 19 :=
by 
  intros h_teachers h_staff h_teachers_ate h_staff_ate h_teachers_not_eating h_staff_not_eating
  rw [h_teachers, h_staff, h_teachers_ate, h_staff_ate] at *
  rw [h_teachers_not_eating, h_staff_not_eating]
  have teachers_eating_pizza := 2 * 30 / 3
  have staff_eating_pizza := 4 * 45 / 5
  simp at teachers_eating_pizza staff_eating_pizza
  rw [teachers_eating_pizza, staff_eating_pizza]
  norm_num


end non_pizza_eaters_count_l647_647029


namespace min_value_x_3y_6z_l647_647228

theorem min_value_x_3y_6z (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 27) : x + 3 * y + 6 * z ≥ 27 :=
sorry

end min_value_x_3y_6z_l647_647228


namespace total_saplings_l647_647253

theorem total_saplings (a_efficiency b_efficiency : ℝ) (A B T n : ℝ) 
  (h1 : a_efficiency = (3/4))
  (h2 : b_efficiency = 1)
  (h3 : B = n + 36)
  (h4 : T = 2 * n + 36)
  (h5 : n * (4/3) = n + 36)
  : T = 252 :=
by {
  sorry
}

end total_saplings_l647_647253


namespace toothbrush_difference_l647_647061

-- Define the initial conditions
def initial_toothbrushes : ℕ := 330
def january_toothbrushes : ℕ := 53
def february_toothbrushes : ℕ := 67
def march_toothbrushes : ℕ := 46

-- Calculate the remaining toothbrushes
def remaining_toothbrushes : ℕ := initial_toothbrushes - (january_toothbrushes + february_toothbrushes + march_toothbrushes)

-- Calculate toothbrushes given in April and May
def april_may_toothbrushes : ℕ := remaining_toothbrushes / 2

-- Define the difference in toothbrushes given away in the busiest and slowest months
def difference := april_may_toothbrushes - march_toothbrushes

-- Theorem to prove the difference is 36
theorem toothbrush_difference : difference = 36 :=
by
  -- Provided conditions in the problem
  have initial : initial_toothbrushes = 330 := rfl
  have jan : january_toothbrushes = 53 := rfl
  have feb : february_toothbrushes = 67 := rfl
  have mar : march_toothbrushes = 46 := rfl
  have remaining := calc
    remaining_toothbrushes = 330 - (53 + 67 + 46) : by sorry
  
  have april_may := calc
    april_may_toothbrushes = remaining_toothbrushes / 2 : by sorry
  
  have diff := calc
    difference = 82 - 46 : by sorry
  
  exact sorry

end toothbrush_difference_l647_647061


namespace Chloe_wins_l647_647805

theorem Chloe_wins (C M : ℕ) (h_ratio : 8 * M = 3 * C) (h_Max : M = 9) : C = 24 :=
by {
    sorry
}

end Chloe_wins_l647_647805


namespace line_product_l647_647770

theorem line_product : 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b) ∧ 
  (y_coords = (-3, 6)) ∧ 
  (x_coords = (0, 3)) ∧
  m * b = -9 :=
begin
  sorry
end

end line_product_l647_647770


namespace bicycle_cost_price_l647_647005

theorem bicycle_cost_price (CP_A : ℝ) 
    (h1 : ∀ SP_B, SP_B = 1.20 * CP_A)
    (h2 : ∀ CP_C SP_B, CP_C = 1.40 * SP_B ∧ SP_B = 1.20 * CP_A)
    (h3 : ∀ SP_D CP_C, SP_D = 1.30 * CP_C ∧ CP_C = 1.40 * 1.20 * CP_A)
    (h4 : ∀ SP_D', SP_D' = 350 / 0.90) :
    CP_A = 350 / 1.9626 :=
by
  sorry

end bicycle_cost_price_l647_647005


namespace erasure_problem_impossibility_l647_647255

-- Define the arithmetic sequence
def arith_seq : List ℕ := [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104]

-- Sum of the arithmetic sequence
def arith_seq_sum : ℕ := arith_seq.sum

-- Define the problem as a theorem
theorem erasure_problem_impossibility {l : List ℕ} :
  l = arith_seq → 
  arith_seq_sum % 11 = 0 → 
  ∀ (a b c d e : ℕ), 
  (a ∈ l) ∧ (b ∈ l.erase a) ∧ (c ∈ l.erase a.erase b) ∧ (d ∈ l.erase a.erase b.erase c) ∧ 
  (e ∈ l.erase a.erase b.erase c.erase d) → 
  (arith_seq_sum - a) % 11 = 0 ∧ 
  (arith_seq_sum - a - b - c) % 11 = 0 ∧ 
  (arith_seq_sum - a - b - c - d - e) % 11 = 0 → 
  False := 
by
  sorry

end erasure_problem_impossibility_l647_647255


namespace translate_M_l647_647947

-- Definitions and conditions
def point := (ℝ × ℝ)

def A : point := (-2, 3)
def D : point := (1, 2)
def M : point := (3, -1)

def translation_vector : point := (1 + 2, 2 - 3)

-- Prove the new coordinates of point M after translation
theorem translate_M : 
  let M' := (M.1 + translation_vector.1, M.2 + translation_vector.2) in
  M' = (6, -2) :=
by
  sorry

end translate_M_l647_647947


namespace calculate_expression_l647_647804

def seq (k : Nat) : Nat := 2^k + 3^k

def product_seq : Nat :=
  (2 + 3) * (2^3 + 3^3) * (2^6 + 3^6) * (2^12 + 3^12) * (2^24 + 3^24)

theorem calculate_expression :
  product_seq = (3^47 - 2^47) :=
sorry

end calculate_expression_l647_647804


namespace total_shaded_area_l647_647190

-- Defining areas of circles and shaded regions
def area_of_circle (r : ℝ) : ℝ := π * r^2

def shaded_area_outer_circle (total_area : ℝ) : ℝ := total_area / 2

def shaded_area_inner_circle (total_area : ℝ) : ℝ := 2 / 3 * total_area

-- Proved properties based on problem conditions
theorem total_shaded_area (R r : ℝ) (H : 81π = π * R^2) : 
  R = 9 → r = 4.5 → 
  total_shaded_area = 54π :=
by
  intro hR hr
  sorry

end total_shaded_area_l647_647190


namespace function_comparison_l647_647244

variable {f g : ℝ → ℝ}

theorem function_comparison (h_diff_f : ∀ x ∈ Ioo (3 : ℝ) 7, differentiable ℝ f)
                           (h_diff_g : ∀ x ∈ Ioo (3 : ℝ) 7, differentiable ℝ g)
                           (h_deriv_lt : ∀ x ∈ Ioo (3 : ℝ) 7, deriv f x < deriv g x) :
  ∀ x ∈ Ioo (3 : ℝ) 7, f x + g 3 < g x + f 3 :=
by
  sorry

end function_comparison_l647_647244


namespace sum_of_inscribed_radii_geq_incircle_l647_647950

theorem sum_of_inscribed_radii_geq_incircle
  {A B C : Point}  -- Points defining the triangle
  (h_acute : acute_angle_triangle A B C)  -- Condition: ABC is an acute-angled triangle
  (radii_tangent : ∃ (r1 r2 : ℝ), ∀ (AC BC AB : Line) (circ1 circ2 : Circle), 
                     touches_circ1_AC_BC circ1 AC BC ∧
                     touches_circ2_AB_BC circ2 AB BC ∧
                     tangent_to_each_other circ1 circ2)  -- Condition: Radii tangent configuration
  : ∀ (r : ℝ), r1 + r2 ≥ r := 
sorry  -- Proof of the inequality

end sum_of_inscribed_radii_geq_incircle_l647_647950


namespace vectors_coplanar_l647_647028

def vector (α : Type*) [Add α] [Zero α] := list α

def scalar_triple_product (a b c : vector ℝ) : ℝ :=
  let mat := (list.zip3 a b c).map (λ ⟨x, y, z⟩, [x, y, z]) in
  (mat.head!!! 0) *
  ((mat.tail!!! 0).head!!! 1 * (mat.tail!!! 1).head!!! 2 -
  (mat.tail!!! 0).head!!! 2 * (mat.tail!!! 1).head!!! 1) -
  (mat.head!!! 1) *
  ((mat.tail!!! 0).head!!! 0 * (mat.tail!!! 1).head!!! 2 -
  (mat.tail!!! 0).head!!! 2 * (mat.tail!!! 1).head!!! 0) +
  (mat.head!!! 2) *
  ((mat.tail!!! 0).head!!! 0 * (mat.tail!!! 1).head!!! 1 -
  (mat.tail!!! 0).head!!! 1 * (mat.tail!!! 1).head!!! 0)

def a : vector ℝ := [-2, -4, -3]
def b : vector ℝ := [4, 3, 1]
def c : vector ℝ := [6, 7, 4]

theorem vectors_coplanar : scalar_triple_product a b c = 0 :=
by sorry

end vectors_coplanar_l647_647028


namespace even_function_m_value_l647_647174

theorem even_function_m_value {m : ℤ} (h : ∀ (x : ℝ), (m^2 - m - 1) * (-x)^m = (m^2 - m - 1) * x^m) : m = 2 := 
by
  sorry

end even_function_m_value_l647_647174


namespace population_net_increase_l647_647936

theorem population_net_increase
  (birth_rate : ℕ) (death_rate : ℕ) (T : ℕ)
  (h1 : birth_rate = 7) (h2 : death_rate = 3) (h3 : T = 86400) :
  (birth_rate - death_rate) * (T / 2) = 172800 :=
by
  sorry

end population_net_increase_l647_647936


namespace commute_times_variance_l647_647774

theorem commute_times_variance (x y : ℝ) :
  (x + y + 10 + 11 + 9) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2 →
  |x - y| = 4 :=
by
  sorry

end commute_times_variance_l647_647774


namespace minimum_possible_value_of_Box_l647_647920

theorem minimum_possible_value_of_Box :
  ∃ a b : ℤ, a ≠ b ∧ a * b = 45 ∧ 
    (∀ c d : ℤ, c * d = 45 → c^2 + d^2 ≥ 106) ∧ a^2 + b^2 = 106 :=
by
  sorry

end minimum_possible_value_of_Box_l647_647920


namespace permutations_of_six_distinct_letters_l647_647571

-- Let's state our problem in Lean
theorem permutations_of_six_distinct_letters : 
  (∃ (letters : Finset (String × Nat)), letters.card = 6) →
  (Finset.card (Finset.univ : Finset (String × Nat)) = 6) →
  ∃ permutations : Nat, permutations = 720 :=
by
  intros
  sorry

end permutations_of_six_distinct_letters_l647_647571


namespace no_real_values_of_p_for_equal_roots_l647_647425

theorem no_real_values_of_p_for_equal_roots (p : ℝ) : ¬ ∃ (p : ℝ), (p^2 - 2*p + 5 = 0) :=
by sorry

end no_real_values_of_p_for_equal_roots_l647_647425


namespace dot_product_PB_PD_l647_647946

-- We define a rectangle and point P, with given distances PA, PC, and the diagonal AC.
variables (A B C D P : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]
variables (PA PC AC : ℝ)

-- The conditions of the problem:
axiom PA_eq : dist A P = 3
axiom PC_eq : dist P C = 4
axiom AC_eq : dist A C = 6
axiom rectangle_ABCD : isRectangle A B C D

-- The proof goal: the dot product of PB and PD is -11/2.
theorem dot_product_PB_PD : ∃ (PB PD : 𝕍), (PB • PD) = -11 / 2 :=
by
  sorry

end dot_product_PB_PD_l647_647946


namespace point_A_coordinates_l647_647074

theorem point_A_coordinates :
  (∃ z : ℝ, (∃ z : ℝ, ((0, 0, z): ℝ × ℝ × ℝ) = (0, 0, -(3 / 8 ': ℝ)) ∧
  (∃ A : ℝ × ℝ × ℝ, A = (0, 0, z) →
    dist A (-5, -5, 6) = dist A (-7, 6, 2))) :=
sorry

end point_A_coordinates_l647_647074


namespace jack_bought_apples_l647_647607

theorem jack_bought_apples :
  ∃ n : ℕ, 
    (∃ k : ℕ, k = 10 ∧ ∃ m : ℕ, m = 5 * 9 ∧ n = k + m) ∧ n = 55 :=
by
  sorry

end jack_bought_apples_l647_647607


namespace chromium_percentage_in_new_alloy_l647_647945

theorem chromium_percentage_in_new_alloy :
  ∀ (weight1 weight2 chromium1 chromium2: ℝ),
  weight1 = 15 → weight2 = 35 → chromium1 = 0.12 → chromium2 = 0.08 →
  (chromium1 * weight1 + chromium2 * weight2) / (weight1 + weight2) * 100 = 9.2 :=
by
  intros weight1 weight2 chromium1 chromium2 hweight1 hweight2 hchromium1 hchromium2
  sorry

end chromium_percentage_in_new_alloy_l647_647945


namespace number_of_excellent_students_l647_647767

variable (X : Type) [MeasureSpace X] [NormedAddCommGroup X] [NormedSpace ℝ X]
  (P : X → ℝ) (μ δ : ℝ) (n : ℕ)

-- Conditions
def normal_distribution := true -- Placeholder for X ~ N(90, δ^2)
def probability_condition := P {x | x < 60} = 0.1
def students_participated := n = 1200
def excellent_score_condition := (120 : ℝ) > 90

-- Statement
theorem number_of_excellent_students : 
  normal_distribution → probability_condition → students_participated → excellent_score_condition → 
  ∃ num_excellent_students : ℕ, num_excellent_students = 120 :=
begin
  -- Proof is omitted
  sorry
end

end number_of_excellent_students_l647_647767


namespace probability_of_selecting_all_blue_balls_l647_647354

noncomputable def probability_all_blue (total_balls blue_balls select_balls : ℕ) : ℚ :=
  let total_combinations := nat.choose total_balls select_balls
  let blue_combinations := nat.choose blue_balls select_balls
  blue_combinations / total_combinations

theorem probability_of_selecting_all_blue_balls :
  probability_all_blue 10 4 3 = 1 / 30 :=
by
  sorry

end probability_of_selecting_all_blue_balls_l647_647354


namespace cadastral_value_of_land_l647_647410

theorem cadastral_value_of_land (tax_amount_paid : ℝ) (tax_rate : ℝ) (V : ℝ) :
  (tax_amount_paid = 4500) → (tax_rate = 0.003) → (V = tax_amount_paid / tax_rate) → (V = 1500000) :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have hV : V = 4500 / 0.003 := h3
  have hV_value : 4500 / 0.003 = 1500000 := 
    by norm_num
  rw hV_value at hV
  exact hV

#eval cadastral_value_of_land 4500 0.003 1500000 sorry -- Testing purpose to ensure it compiles.

end cadastral_value_of_land_l647_647410


namespace math_problem_proof_l647_647582

-- Define the problem conditions
variables {A B C D E F : Type}
variables [finite_dimensional ℝ ℝ (A × B × C)]
variables (triangle_ABC : ∀ {A B C : ℝ}, triangle A B C)
variables (midpoint_D : ∀ {B C : ℝ}, midpoint D B C)
variables (E_point : ∀ {A B : ℝ}, point E A B)
variables (F_point : ∀ {A C : ℝ}, point F A C)
variables (DE_DF_equal : ∀ {D E F : ℝ}, {DE = DF})

-- Problem statement
theorem math_problem_proof
  (A B C D E F : Type)
  (midpoint_D : D.midpoint B C)
  (E_point : E.point A B)
  (F_point : F.point A C)
  (DE_DF_equal : E.distance_to(D) = F.distance_to(D)) :
  (E.distance_to(A) + F.distance_to(A) = E.distance_to(B) + F.distance_to(C)) ↔ (angle D E F = angle B A C) :=
by
  sorry

end math_problem_proof_l647_647582


namespace distances_from_median_l647_647707

theorem distances_from_median (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (x y : ℝ), x = (b * c) / (a + b) ∧ y = (a * c) / (a + b) ∧ x + y = c :=
by
  sorry

end distances_from_median_l647_647707


namespace n_not_composite_l647_647703

theorem n_not_composite
  (n : ℕ) (h1 : n > 1)
  (a : ℕ) (q : ℕ) (hq_prime : Nat.Prime q)
  (hq1 : q ∣ (n - 1))
  (hq2 : q > Nat.sqrt n - 1)
  (hn_div : n ∣ (a^(n-1) - 1))
  (hgcd : Nat.gcd (a^(n-1)/q - 1) n = 1) :
  ¬ Nat.Prime n :=
sorry

end n_not_composite_l647_647703


namespace determine_k_l647_647327

variable (k : ℕ)

def set_A := { x : ℝ | (kx - 1) / x < 0 }
def set_B := { x : ℝ | -1 ≤ x ∧ x ≤ 4 }
def set_C := { x : ℝ | 0 < x ∧ x < 1/2 }

axiom student_A (h : 1 ≤ k ∧ k < 6) : True
axiom student_B : set_A k ⊆ set_B ∧ set_A k ≠ set_B
axiom student_C : set_C ⊆ set_A k

theorem determine_k : k = 1 := by
  sorry

end determine_k_l647_647327


namespace exists_triangular_pizza_cut_l647_647606

theorem exists_triangular_pizza_cut : 
  ∃(T : Set Point), is_triangular T ∧ (∃cuts : list (Set Point) → Set Point, (∀(c : Set Point), c ∈ cuts → is_straight_cut c) ∧ divides_into_identical_triangular_pieces T cuts 11) :=
begin
  sorry
end

end exists_triangular_pizza_cut_l647_647606


namespace zoe_total_songs_l647_647348

-- Define the number of country albums Zoe bought
def country_albums : Nat := 3

-- Define the number of pop albums Zoe bought
def pop_albums : Nat := 5

-- Define the number of songs per album
def songs_per_album : Nat := 3

-- Define the total number of albums
def total_albums : Nat := country_albums + pop_albums

-- Define the total number of songs
def total_songs : Nat := total_albums * songs_per_album

-- Theorem statement asserting the total number of songs
theorem zoe_total_songs : total_songs = 24 := by
  -- Proof will be inserted here (currently skipped)
  sorry

end zoe_total_songs_l647_647348


namespace length_of_first_train_is_correct_l647_647788

noncomputable def length_of_first_train 
  (speed_first_train_kmph : ℝ)
  (length_second_train_m : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_crossing_s : ℝ) : ℝ :=
  let speed_first_train_mps := (speed_first_train_kmph * 1000) / 3600
  let speed_second_train_mps := (speed_second_train_kmph * 1000) / 3600
  let relative_speed_mps := speed_first_train_mps + speed_second_train_mps
  let total_distance_m := relative_speed_mps * time_crossing_s
  total_distance_m - length_second_train_m

theorem length_of_first_train_is_correct :
  length_of_first_train 50 112 82 6 = 108.02 :=
by
  sorry

end length_of_first_train_is_correct_l647_647788


namespace cubed_gt_if_gt_l647_647098

theorem cubed_gt_if_gt {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cubed_gt_if_gt_l647_647098


namespace mass_percentage_Br_in_BaBr2_approx_l647_647076

def molar_mass_Ba : ℝ := 137.33
def molar_mass_Br : ℝ := 79.90

def molar_mass_BaBr2 (m_Ba m_Br : ℝ) : ℝ := m_Ba + 2 * m_Br

def mass_percentage_of_Br (m_Ba m_Br : ℝ) : ℝ := 
  (2 * m_Br / (m_Ba + 2 * m_Br)) * 100

theorem mass_percentage_Br_in_BaBr2_approx : 
  mass_percentage_of_Br molar_mass_Ba molar_mass_Br ≈ 53.79 :=
by
  sorry

end mass_percentage_Br_in_BaBr2_approx_l647_647076


namespace singleton_set_M_l647_647854

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

-- Recursively define f_n(x)
noncomputable def f_n : ℕ+ → (ℝ → ℝ)
| ⟨1, _⟩ := f
| ⟨n+1, h⟩ := f ∘ f_n ⟨n, Nat.le_of_succ_le_succ h⟩

def M : set ℝ := { x | f_n ⟨2017, sorry⟩ x = -real.log2 (abs x) }

theorem singleton_set_M : set.cardinality M = 1 := sorry

end singleton_set_M_l647_647854


namespace eval_poly_at_2_l647_647638

def op (x y : ℕ) : ℕ := (x + 1) * (y + 1)
def op2 (x : ℕ) : ℕ := op x x

theorem eval_poly_at_2 : (3 * op2 2) - 2 * 2 + 1 = 32 := by
  let x := 2
  let x_op2 := op2 x
  have h1 : x_op2 = op x x := rfl
  have h2 : x_op2 = (x+1) * (x+1) := by rw [h1, op]
  have h3 : x_op2 = 9 := by rw [h2, Nat.add_assoc]; norm_num
  calc
    (3 * x_op2) - 2 * x + 1
        = (3 * 9) - 2 * 2 + 1 := by rw [h3]
    ... = 27 - 4 + 1 := rfl
    ... = 32 := by norm_num

end eval_poly_at_2_l647_647638


namespace chebyshev_birth_year_l647_647958

theorem chebyshev_birth_year :
  ∃ (a b : ℕ),
  a > b ∧ 
  a + b = 3 ∧ 
  (1821 = 1800 + 10 * a + 1 * b) ∧
  (1821 + 73) < 1900 :=
by sorry

end chebyshev_birth_year_l647_647958


namespace coffee_cost_per_week_l647_647207

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end coffee_cost_per_week_l647_647207


namespace f_neg_six_value_l647_647523

noncomputable def f_periodic_even (a : ℝ) (x : ℝ) : ℝ :=
  if h : x >= -3 ∧ x <= 3 then (x + 1) * (x - a)
  else if h : (x + 6) >= -3 ∧ (x + 6) <= 3 then (x + 6 + 1) * (x + 6 - a)
  else 0 -- arbitrary outside given range and periodic extension

theorem f_neg_six_value : ∃ a : ℝ, (∀ x : ℝ, f_periodic_even 1 x = f_periodic_even 1 (-x)) →
  (∀ x : ℝ, f_periodic_even 1 (x + 6) = f_periodic_even 1 x) →
  f_periodic_even 1 (-6) = -1 :=
begin
  sorry
end

end f_neg_six_value_l647_647523


namespace reciprocal_cycle_solution_l647_647365

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_cycle {x : ℝ} (h : x = 50) : reciprocal (reciprocal x) = x :=
by
  rw [h, reciprocal, reciprocal]
  rw [one_div_one_div]
  exact h

-- Here we specifically assert the result when x = 50
theorem solution : reciprocal (reciprocal 50) = 50 :=
by sorry

end reciprocal_cycle_solution_l647_647365


namespace column_of_1000_is_B_l647_647023

theorem column_of_1000_is_B :
  let pattern := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"] in
  pattern[(999 % 10)] = "B" :=
by
  let pattern := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"]
  sorry

end column_of_1000_is_B_l647_647023


namespace unique_function_l647_647465

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry
lemma f_g_inverse : ∀ x : ℝ, f x + g x = 2 * x := sorry

theorem unique_function (c : ℝ) : (∀ x : ℝ, f x = x + c) :=
begin
  sorry
end

end unique_function_l647_647465


namespace find_monthly_salary_l647_647018

variables (x h_1 h_2 h_3 : ℕ)

theorem find_monthly_salary 
    (half_salary_bank : h_1 = x / 2)
    (half_remaining_mortgage : h_2 = (h_1 - 300) / 2)
    (half_remaining_expenses : h_3 = (h_2 + 300) / 2)
    (remaining_salary : h_3 = 800) :
  x = 7600 :=
sorry

end find_monthly_salary_l647_647018


namespace round_robin_tournament_matches_l647_647778

theorem round_robin_tournament_matches (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  rw h
  norm_num

end round_robin_tournament_matches_l647_647778


namespace michael_twice_jacob_in_11_years_l647_647205

-- Definitions
def jacob_age_4_years := 5
def jacob_current_age := jacob_age_4_years - 4
def michael_current_age := jacob_current_age + 12

-- Theorem to prove
theorem michael_twice_jacob_in_11_years :
  ∀ (x : ℕ), jacob_current_age + x = 1 →
    michael_current_age + x = 13 →
    michael_current_age + (11 : ℕ) = 2 * (jacob_current_age + (11 : ℕ)) :=
by
  intros x h1 h2
  sorry

end michael_twice_jacob_in_11_years_l647_647205


namespace n_eq_7_mod_10_l647_647484

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l647_647484


namespace domain_of_g_l647_647132

variable (f : ℝ → ℝ)
variable domain_f : set.Icc (0 : ℝ) 2 ⊆ set.Univ
variable (g : ℝ → ℝ)
-- Define g(x) = f(2x) / log3(2^x + 1)
noncomputable def g (x : ℝ) := f (2 * x) / Real.logb 3 (2^x + 1)

theorem domain_of_g : ∀ x, 0 ≤ x ∧ x ≤ 1 ↔ ∃ x, g x ∈ set.Icc (0 : ℝ) 1 := by
sorry

end domain_of_g_l647_647132


namespace sine_expression_value_l647_647636

noncomputable def c : ℝ := 2 * Real.pi / 13

theorem sine_expression_value : 
  (sin (4*c) * sin (9*c) * sin (10*c) * sin (12*c) * sin (16*c)) / 
  (sin c * sin (3*c) * sin (4*c) * sin (6*c) * sin (8*c)) = 1 :=
by
  sorry

end sine_expression_value_l647_647636


namespace trig_identity_cos_add_l647_647872

open Real

theorem trig_identity_cos_add (x : ℝ) (h1 : sin (π / 3 - x) = 3 / 5) (h2 : π / 2 < x ∧ x < π) :
  cos (x + π / 6) = 3 / 5 :=
by
  sorry

end trig_identity_cos_add_l647_647872


namespace lambda_value_l647_647154

-- Define the vectors
def a : ℝ × ℝ := (0, 1)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (3, 4)

-- Define perpendicularity condition
def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Define the given condition
def cond (λ : ℝ) : Prop := perp (b.1 + λ * a.1, b.2 + λ * a.2) c

-- The proof problem
theorem lambda_value : ∀ λ : ℝ, cond λ → λ = -3/4 :=
begin
  sorry -- Proof is not required according to the instructions
end

end lambda_value_l647_647154


namespace last_two_digits_of_sum_l647_647845

theorem last_two_digits_of_sum : 
  (6.factorial + 9.factorial + 12.factorial + 15.factorial + 18.factorial + 21.factorial +
   24.factorial + 27.factorial + 30.factorial + 33.factorial + 36.factorial + 39.factorial + 
   42.factorial + 45.factorial + 48.factorial + 51.factorial + 54.factorial + 57.factorial + 
   60.factorial + 63.factorial + 66.factorial + 69.factorial + 72.factorial + 75.factorial + 
   78.factorial + 81.factorial + 84.factorial + 87.factorial + 90.factorial + 93.factorial +
   96.factorial) % 100 = 20 :=
begin
  sorry
end

end last_two_digits_of_sum_l647_647845


namespace solve_abs_inequality_l647_647426

theorem solve_abs_inequality (x : ℝ) : abs ((7 - x) / 4) < 3 → 2 < x ∧ x < 19 :=
by 
  sorry

end solve_abs_inequality_l647_647426


namespace find_x_values_l647_647839

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  1 / (x + 2) + 4 / (x + 4) ≥ 1

theorem find_x_values : {x : ℝ | satisfies_inequality x} = set.Ioc (-2) 1 := 
sorry

end find_x_values_l647_647839


namespace example_sets_six_element_set_general_conclusions_l647_647624

open Set

def non_empty_set (S : Set ℕ) : Prop := 
  ∃ x, x ∈ S

def positive_integers (S : Set ℕ) : Prop :=
  ∀ x ∈ S, x > 0

def ten_complement (S : Set ℕ) : Prop :=
  ∀ x ∈ S, (10 - x) ∈ S

theorem example_sets (S1 S2 S3 : Set ℕ) :
  S1 = {5} →
  S2 = {2, 8} →
  S3 = {3, 5, 7} →
  non_empty_set S1 ∧ positive_integers S1 ∧ ten_complement S1 ∧ 
  non_empty_set S2 ∧ positive_integers S2 ∧ ten_complement S2 ∧ 
  non_empty_set S3 ∧ positive_integers S3 ∧ ten_complement S3 :=
by sorry

theorem six_element_set (S : Set ℕ) :
  S = {1,2,3,4,6,7,8,9} →
  non_empty_set S ∧ positive_integers S ∧ ten_complement S ∧ S.card = 6 :=
by sorry

theorem general_conclusions (S : Set ℕ) :
  (non_empty_set S ∧ positive_integers S ∧ ten_complement S) →
  S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  ((5 ∈ S ∧ S.card % 2 = 1) ∨ (5 ∉ S ∧ S.card % 2 = 0)) ∧
  (S ≠ S ∅ → cardinality (S ⊆ {1, 2, 3, 4, 6, 7, 8, 9}) ∈ 15) ∧
  (∃i , S.forall i = {5} →  (cardinality (S ⊆ {1, 2, 3, 4, 6, 7, 8, 9, 10 - i}) ∈ 16))
  ∧ (total sets = 31) :=
by sorry

end example_sets_six_element_set_general_conclusions_l647_647624


namespace min_m_minus_n_half_l647_647903

open Real

variables {α β γ : ℝ × ℝ}

-- Conditions
def norm_eq_one (α : ℝ × ℝ) : Prop :=
  ∥α∥ = 1

def norm_eq_norm (α β : ℝ × ℝ) : Prop :=
  ∥α - β∥ = ∥β∥

def dot_product_orthogonal (α β γ : ℝ × ℝ) : Prop :=
  (α - γ) • (β - γ) = 0

-- Theorem statement proving the minimum value of m - n
theorem min_m_minus_n_half (α β γ : ℝ × ℝ) (h1 : norm_eq_one α) (h2 : norm_eq_norm α β) (h3 : dot_product_orthogonal α β γ) :
  ∃ y1: ℝ, f(y1) = 1/2 := sorry

end min_m_minus_n_half_l647_647903


namespace find_m_of_eccentricity_l647_647133

theorem find_m_of_eccentricity :
  ∀ (m : ℝ), 
  (m > 0) → 
  (∀ x y : ℝ, (x^2)/2 + (y^2)/m = 1 → 
  ∃ ecc : ℝ, ecc = sqrt(6)/3 ∧ ecc = sqrt(1 - if a = sqrt(2) then m/2 else 2/m) → 
  (m = 2/3 ∨ m = 6)) := by 
sorry

end find_m_of_eccentricity_l647_647133


namespace avg_of_6_10_N_is_10_if_even_l647_647632

theorem avg_of_6_10_N_is_10_if_even (N : ℕ) (h1 : 9 ≤ N) (h2 : N ≤ 17) (h3 : (6 + 10 + N) % 2 = 0) : (6 + 10 + N) / 3 = 10 :=
by
-- sorry is placed here since we are not including the actual proof
sorry

end avg_of_6_10_N_is_10_if_even_l647_647632


namespace domain_of_function_l647_647695

theorem domain_of_function (x : ℝ) : (∃ y : ℝ, y = sqrt (Real.log10 (x - 2)) / x) ↔ x ≥ 3 :=
by
  sorry

end domain_of_function_l647_647695


namespace coefficients_negiff_sum_gt_l647_647982

noncomputable theory
open_locale classical

def poly_g (n : ℕ) (a : fin n → ℝ) (x : ℝ) : ℝ :=
(finset.univ.prod (λ i, x + a i))

def poly_f (n : ℕ) (a : fin n → ℝ) (a_0 : ℝ) (x : ℝ) : ℝ :=
(x - a_0) * poly_g n a x

def coefficients (n : ℕ) (a : fin n → ℝ) (a_0 : ℝ) : fin (n + 2) → ℝ := sorry  -- Placeholder for coefficients extraction

theorem coefficients_negiff_sum_gt (n : ℕ) (a : fin n → ℝ) (a_0 : ℝ) :
  (∀ i : fin (n + 2), coefficients n a a_0 i < 0) ↔ (a_0 > finset.univ.sum (λ i, a i)) :=
sorry

end coefficients_negiff_sum_gt_l647_647982


namespace sum_largest_smallest_gx_l647_647236

noncomputable def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_largest_smallest_gx : (∀ x, 1 ≤ x ∧ x ≤ 10 → True) → ∀ (a b : ℝ), (∃ x, 1 ≤ x ∧ x ≤ 10 ∧ g x = a) → (∃ y, 1 ≤ y ∧ y ≤ 10 ∧ g y = b) → a + b = -1 :=
by
  intro h x y hx hy
  sorry

end sum_largest_smallest_gx_l647_647236


namespace count_valid_integers_l647_647568

theorem count_valid_integers : 
  let valid_digits := [4, 5, 6, 7, 8, 9].to_finset in
  let digit_set (n : ℕ) := (n.digits 10).to_finset in
  ∑ n in (finset.range 1000).filter (λ n, digit_set (n + 1) ⊆ valid_digits), 1 = 257 :=
by
  sorry

end count_valid_integers_l647_647568


namespace water_reaches_Y_l647_647642

def fork (x : ℕ) : ℕ := x / 2

theorem water_reaches_Y :
  let total_water := 296 in
  let water_X := fork (fork total_water) in
  let water_Y := total_water - water_X in
  water_Y = 222 :=
by
  intros
  sorry

end water_reaches_Y_l647_647642


namespace smaller_circle_area_l647_647718

noncomputable def area_smaller_circle (PA : ℝ) (r : ℝ) : ℝ :=
  (let y := sqrt (25 / 8) in 
  pi * y^2)

theorem smaller_circle_area (PA : ℝ) (r : ℝ) 
  (h1 : PA = 5)
  (h2 : r = 2 * sqrt(25 / 8)):
  area_smaller_circle PA r = 25 * pi / 8 :=
by
  sorry

end smaller_circle_area_l647_647718


namespace Znayka_sufficient_numbers_l647_647740

theorem Znayka_sufficient_numbers :
  ∀ (p q : ℝ),
  (p > 4 ∧ p > q) →
  (x^2 + p * x + q).root_count > 0 ∧
  ∀ (r s : ℝ),
  (0 < q ∧ 0 < r ∧ 0 < s ∧ q < r ∧ r < s < p) →
  (x^2 + p * x + q).root_count > 0 →
  (x^2 + r * x + s).root_count > 0 →
  (distinct_roots (x^2 + p * x + q) ∧ distinct_roots (x^2 + r * x + s)) :=
sorry

end Znayka_sufficient_numbers_l647_647740


namespace attendees_count_l647_647430

def n_students_seated : ℕ := 300
def n_students_standing : ℕ := 25
def n_teachers_seated : ℕ := 30

def total_attendees : ℕ :=
  n_students_seated + n_students_standing + n_teachers_seated

theorem attendees_count :
  total_attendees = 355 := by
  sorry

end attendees_count_l647_647430


namespace triangle_area_l647_647201

theorem triangle_area (a b c : ℝ) (C : ℝ) (h₁ : a^2 + b^2 - c^2 = 6 * real.sqrt 3 - 2 * a * b) (h₂ : C = real.pi / 3) : 
  1 / 2 * a * b * real.sin C = 3 / 2 :=
by
  sorry

end triangle_area_l647_647201


namespace circles_intersect_at_single_point_l647_647859

variables {A B C D E F P Q R : Type}

-- Assuming existence of geometric properties and relevant details
axiom incircle_touches : ∀ (A B C : Point), ∃ D E F : Point, 
  (incircle A B C).touches D E F sides [BC, CA, AB]

axiom circumcircle : ∀ (A B C : Point), Circle
axiom circumcircle_triangle : ∀ (A B C D E F : Point), 
  Circle A B C

axiom second_intersection : ∀ (Γ : Circle) (Ω : Circle) (A B C D E F : Point), 
  Point

theorem circles_intersect_at_single_point 
  (ABC : Triangle) (DEF : Point) 
  (circumABC : Circle) (circumAEF : Circle) 
  (circumBFD : Circle) (circumCDE : Circle) 
  (P Q R : Point) :
  let ΓA := circumcircle_triangle A E F,
      ΓB := circumcircle_triangle B F D,
      ΓC := circumcircle_triangle C D E,
      P := second_intersection ΓA circumABC,
      Q := second_intersection ΓB circumABC,
      R := second_intersection ΓC circumABC in
  (circles_intersect_single_point ΓA ΓB ΓC) ∧ 
  (lines_are_concurrent (P, D) (Q, E) (R, F)) :=
by sorry

end circles_intersect_at_single_point_l647_647859


namespace boys_in_school_l647_647184

theorem boys_in_school 
  (initial_boys : ℕ)
  (additional_boys : ℕ)
  (initial_boys = 214)
  (additional_boys = 910) : 
  initial_boys + additional_boys = 1124 := 
by 
  sorry

end boys_in_school_l647_647184


namespace profit_percentage_l647_647589

theorem profit_percentage (C : ℝ) (hC : C > 0) : 
  let S := 2.10 * C in
  let C_new := 1.12 * C in
  let S_new := 2.163 * C in
  let Profit_new := S_new - C_new in
  (Profit_new / S_new) * 100 ≈ 48.2 :=
by
  let S := 2.10 * C
  let C_new := 1.12 * C
  let S_new := 2.163 * C
  let Profit_new := S_new - C_new
  suffices : (Profit_new / S_new) * 100 ≈ 48.2
  from this
  sorry

end profit_percentage_l647_647589


namespace meters_sold_equals_450_l647_647007

-- Define the given conditions
def cost_price_per_meter : ℝ := 45
def loss_per_meter : ℝ := 5
def selling_price_per_meter : ℝ := cost_price_per_meter - loss_per_meter
def total_selling_price : ℝ := 18000

-- Define the number of meters of cloth sold
def meters_of_cloth_sold : ℝ := total_selling_price / selling_price_per_meter

-- The theorem to be proved
theorem meters_sold_equals_450 : meters_of_cloth_sold = 450 :=
by
  -- Proof is omitted
  sorry

end meters_sold_equals_450_l647_647007


namespace croissant_process_time_in_hours_l647_647651

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end croissant_process_time_in_hours_l647_647651


namespace car_distance_covered_l647_647366

-- Define the conditions
def car_initial_time : ℝ := 6 -- The car takes 6 hours initially.
def car_new_speed : ℝ := 52.111111111111114 -- The car's speed to maintain.
def fraction_factor : ℝ := 3 / 2 -- The fraction of the original time.

-- Define the theorem statement
theorem car_distance_covered :
  let new_time := fraction_factor * car_initial_time in 
  let distance := car_new_speed * new_time in 
  distance = 469 :=
by
  -- Proof omitted
  sorry

end car_distance_covered_l647_647366


namespace find_incenter_l647_647464

-- Define the arbitrary triangle
variable {α : Type} [EuclideanGeometry α]
variables (A B C M : α)

-- Define the main hypothesis and theorem
theorem find_incenter (inside_triangle : M ∈ triangle ABC) (equal_chords : ∀ D ∈ {A, B, C}, length (common_chord M D) = some_constant) :
  M = incenter ABC :=
sorry

end find_incenter_l647_647464


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l647_647670

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l647_647670


namespace find_function_l647_647068

def satisfies_functional_eqn (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem find_function (f : ℝ → ℝ) :
  satisfies_functional_eqn f → (∀ y : ℝ, f y = y^2 - 1) :=
by
  intro h
  sorry

end find_function_l647_647068


namespace original_number_exists_l647_647592

theorem original_number_exists :
  ∃ (x y : ℕ), 
    1 ≤ x ∧ x ≤ 9 ∧ 
    10 ≤ y ∧ y ≤ 99 ∧ 
    (100 * x + y = 9 * y + 8) ∧ (
      (100 * x + y = 224) ∨ 
      (100 * x + y = 449) ∨ 
      (100 * x + y = 674) ∨ 
      (100 * x + y = 899)
    ) :=
begin
  sorry
end

end original_number_exists_l647_647592


namespace maximize_expression_l647_647600

theorem maximize_expression (e f g h : ℕ) (h_values : {e, f, g, h} = {1, 2, 3, 4}):
  e * f ^ g - h ≤ 127 :=
sorry

end maximize_expression_l647_647600


namespace incenter_property_of_chord_lengths_equal_l647_647457

theorem incenter_property_of_chord_lengths_equal
  {A B C M P Q R : Point}
  (h_triangle : is_triangle A B C)
  (h_in_triangle : is_in_triangle M A B C)
  (h_circles : circles_on_diameters A B C M)
  (h_common_chords_equal : common_chords_equal P Q R A B C M) :
  is_incenter M A B C :=
sorry

end incenter_property_of_chord_lengths_equal_l647_647457


namespace complement_intersect_Z_l647_647555

open Set

def A : Set ℝ := Iic (-2) ∪ Ici 3
def complement_A : Set ℝ := Ioi (-2) ∩ Iio 3
def Z := {n : ℤ | (n : ℝ) ∈ complement_A}

theorem complement_intersect_Z (A : Set ℝ) :
  A = Iic (-2) ∪ Ici 3 →
  ((Aᶜ : Set ℝ) ∩ (coe '' (Set.univ : Set ℤ))) = {x : ℤ | x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2} :=
by
  intro hA
  rw [hA]    -- Rewrite A using the hypothesis
  have h1: Aᶜ = Ioi (-2) ∩ Iio 3 := by sorry  -- Skipping proof of the complement step
  apply Set.ext -- Apply set extensionality
  intro z  -- Introduce an element to reason about
  split    -- Split into two goals
  { intro hz  -- Goal: Prove the element is in the target set given its membership in the lean context
    rw [h1] at hz  -- Rewrite in terms of the known complement
    cases hz as ⟨hz1, hz2⟩
    interval_cases z using h1 hz
    repeat {sorry} -- Placeholder for interval argument
  }
  { intro hz  -- Goal: Prove membership the other way around
    cases hz with h1 h1; finish [le_refl] -- Placeholder should be refined with real membership proofs
    sorry 
  }
  sorry

end complement_intersect_Z_l647_647555


namespace clarinets_tried_out_l647_647648

-- Conditions as definitions
def number_flutes : ℕ := 20
def percentage_flutes : ℕ := 80
def number_trumpets : ℕ := 60
def fraction_trumpets : ℚ := 1/3
def number_pianists : ℕ := 20
def fraction_pianists : ℚ := 1/10
def total_people : ℕ := 53
def fraction_clarinets_got_in : ℚ := 1/2

-- Hypothesis summarizing given data
def flutes_got_in := percentage_flutes * number_flutes / 100
def trumpets_got_in := number_trumpets / 3
def pianists_got_in := number_pianists / 10
def clarinets_got_in := total_people - (flutes_got_in + trumpets_got_in + pianists_got_in)

-- The question to prove
theorem clarinets_tried_out : (2 * clarinets_got_in) = 30 :=
by
  unfold flutes_got_in trumpets_got_in pianists_got_in clarinets_got_in
  sorry

end clarinets_tried_out_l647_647648


namespace credit_card_more_profitable_l647_647687

theorem credit_card_more_profitable
  (N : ℕ)
  (H : N > 30) :
  let ticket_cost := 20000
  let credit_cashback := ticket_cost * 0.005
  let debit_cashback := ticket_cost * 0.01
  let annual_interest_rate := 0.06
  let monthly_interest_rate := annual_interest_rate / 12
  let earned_interest := (monthly_interest_rate * ticket_cost * N) / 30
  let total_credit_card_benefit := earned_interest + credit_cashback
  let total_debit_card_benefit := debit_cashback
  in total_credit_card_benefit > total_debit_card_benefit :=
by {
  let ticket_cost := 20000
  let credit_cashback := 100  -- 0.5% of 20000
  let debit_cashback := 200   -- 1% of 20000
  let annual_interest_rate := 0.06
  let monthly_interest_rate := annual_interest_rate / 12
  let earned_interest := (monthly_interest_rate * ticket_cost * N) / 30
  let total_credit_card_benefit := earned_interest + credit_cashback
  let total_debit_card_benefit := debit_cashback
  sorry
}

end credit_card_more_profitable_l647_647687


namespace M_is_incenter_l647_647445

variable (A B C M : Type) [triangle A B C] 

-- Define the problem conditions
structure Circle (P Q : Type) :=
(center : Type)
(radius : ℝ)

noncomputable def circle_on_segment (M A : Type) : Circle M A := sorry

-- Specify the property of M we need to prove
def isIncenter (M : Type) (A B C : Type) [inhabited M] : Prop := 
  ∀ P Q R : Type,
  (circle_on_segment M A).common_chord_length = (circle_on_segment M B).common_chord_length ∧
  (circle_on_segment M B).common_chord_length = (circle_on_segment M C).common_chord_length

-- The theorem statement
theorem M_is_incenter 
  (A B C M : Type) [triangle A B C] [inhabited M] :
  (isIncenter M A B C) → Incenter M A B C :=
sorry

end M_is_incenter_l647_647445


namespace james_total_oop_correct_l647_647608

-- Define the costs and insurance coverage percentages as given conditions.
def cost_consultation : ℝ := 300
def coverage_consultation : ℝ := 0.80

def cost_xray : ℝ := 150
def coverage_xray : ℝ := 0.70

def cost_prescription : ℝ := 75
def coverage_prescription : ℝ := 0.50

def cost_therapy : ℝ := 120
def coverage_therapy : ℝ := 0.60

-- Define the out-of-pocket calculation for each service
def oop_consultation := cost_consultation * (1 - coverage_consultation)
def oop_xray := cost_xray * (1 - coverage_xray)
def oop_prescription := cost_prescription * (1 - coverage_prescription)
def oop_therapy := cost_therapy * (1 - coverage_therapy)

-- Define the total out-of-pocket cost
def total_oop : ℝ := oop_consultation + oop_xray + oop_prescription + oop_therapy

-- Proof statement
theorem james_total_oop_correct : total_oop = 190.50 := by
  sorry

end james_total_oop_correct_l647_647608


namespace workers_task_time_l647_647326

variables (D E Z k : ℝ)
variables (h1 : D = E + 1)
variables (h2 : 1 / D + 1 / E + 1 / Z = 1 / (D - 3))
variables (h3 : 1 / D + 1 / E + 1 / Z = 1 / (E - 2))
variables (h4 : 1 / D + 1 / E + 1 / Z = 3 / Z)

theorem workers_task_time : k = 4 / 3 :=
by
  have h5 : 1 / D + 1 / E = (2 * E + 1) / (E * (E + 1)), 
    sorry

  have h6 : k = (E * (E + 1)) / (2 * E + 1),
    sorry

  -- further algebraic manipulation and substitution would be required to finish the proof:
  have h7 : k = 4 / 3,
    sorry
  exact h7

end workers_task_time_l647_647326


namespace correct_propositions_l647_647547

-- We define each proposition explicitly
def prop1 : Prop := ∀ (A B : ℝ), (A = B) → (sin A = sin B)
def prop2 : Prop := ∀ (P : ℝ × ℝ), P.1 * P.1 + P.2 * P.2 = 64 → (P = (-4,0) ∨ P = (4,0))
def prop3 : Prop := ∀ (p q : Prop), ¬(p ∧ q) → ¬p ∧ ¬q
def prop4 : Prop := ∀ (x : ℝ), (x > 4) → (x^2 - 3 * x > 0)
def prop5 : Prop := ∀ (m : ℝ), (1, m, 9) in GeometricSeq → (eccentricity m = (sqrt 6) / 3)

-- We define the conditions in Lean:
def prop1_true : prop1 := sorry
def prop2_true : prop2 := sorry
def prop3_false : ¬prop3 := sorry
def prop4_true : prop4 := sorry
def prop5_false : ¬prop5 := sorry

-- Final theorem statement
theorem correct_propositions :
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 ∧ ¬prop5) → ({1, 2, 4} = { n | n = 1 ∨ n = 2 ∨ n = 4 }) :=
by simp

end correct_propositions_l647_647547


namespace second_discount_percentage_l647_647780

variables (P : ℝ) (D : ℝ)

theorem second_discount_percentage:
  (1.36 * P * 0.90 * (1 - D) = 1.0404 * P) → (D = 0.15) := 
by 
  intro h 
  sorry

end second_discount_percentage_l647_647780


namespace cube_of_odd_sum_l647_647163

theorem cube_of_odd_sum (a : ℕ) (h1 : 1 < a) (h2 : ∃ (n : ℕ), (n = (a - 1) + 2 * (a - 1) + 1) ∧ n = 1979) : a = 44 :=
sorry

end cube_of_odd_sum_l647_647163


namespace product_modulo_seven_l647_647811

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l647_647811


namespace croissant_process_time_in_hours_l647_647652

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end croissant_process_time_in_hours_l647_647652


namespace minimize_f_l647_647551

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + (Real.sin x)^2

theorem minimize_f :
  ∃ x : ℝ, (-π / 4 < x ∧ x ≤ π / 2) ∧
  ∀ y : ℝ, (-π / 4 < y ∧ y ≤ π / 2) → f y ≥ f x ∧ f x = 1 ∧ x = π / 2 :=
by
  sorry

end minimize_f_l647_647551


namespace min_apples_l647_647322

theorem min_apples :
  ∃ N : ℕ, 
  (N % 3 = 2) ∧ 
  (N % 4 = 2) ∧ 
  (N % 5 = 2) ∧ 
  (N = 62) :=
by
  sorry

end min_apples_l647_647322


namespace train_crosses_pole_in_15_seconds_l647_647015

theorem train_crosses_pole_in_15_seconds
    (train_speed : ℝ) (train_length_meters : ℝ) (time_seconds : ℝ) : 
    train_speed = 300 →
    train_length_meters = 1250 →
    time_seconds = 15 :=
by
  sorry

end train_crosses_pole_in_15_seconds_l647_647015


namespace number_is_37_5_l647_647575

theorem number_is_37_5 (y : ℝ) (h : 0.4 * y = 15) : y = 37.5 :=
sorry

end number_is_37_5_l647_647575


namespace intersection_of_circles_l647_647597

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ∧
  (∀ z1 z2 : ℂ, z1 ≠ z2 → |z1 - 4| = 3 * |z1 + 4| ∧ |z1| = k → |z2 - 4| = 3 * |z2 + 4| ∧ |z2| = k) ↔ 28 < k ∧ k < 44 :=
by
  sorry

end intersection_of_circles_l647_647597


namespace n1M_ge_v2m_l647_647750

variable {m : ConvexPolygon} (M : ConvexPolygon) (n1M : ℕ) (v2m : ℕ)

-- Definitions of n(1, M) and v(2, m)
def n_1_M := n1M
def v_2_m := v2m

-- The geometric properties and shifts
variable (shift_by_distance : ℕ → ConvexPolygon → ConvexPolygon)
variable (unit_distance : ℕ := 1)
variable (distance_2_units : ℕ := 2)
variable (polygon_shift : (ConvexPolygon → ConvexPolygon)) 
variable (M_def : M = shift_by_distance unit_distance m)

-- Conditions of the problem
axiom covers_polygon_m (circ : ℕ → ConvexPolygon → Prop)
axiom circles_n1M (n1M_covered : covers_polygon_m n1M M)
axiom circles_v2m (v2m_covers : covers_polygon_m v2m m)

-- The main theorem
theorem n1M_ge_v2m : n_1_M M n1M ≥ v_2_m m v2m := 
  sorry

end n1M_ge_v2m_l647_647750


namespace sum_of_cubes_divisible_by_nine_l647_647334

theorem sum_of_cubes_divisible_by_nine (n : ℕ) (h : 0 < n) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) :=
by sorry

end sum_of_cubes_divisible_by_nine_l647_647334


namespace number_of_excellent_students_l647_647766

variable (X : Type) [MeasureSpace X] [NormedAddCommGroup X] [NormedSpace ℝ X]
  (P : X → ℝ) (μ δ : ℝ) (n : ℕ)

-- Conditions
def normal_distribution := true -- Placeholder for X ~ N(90, δ^2)
def probability_condition := P {x | x < 60} = 0.1
def students_participated := n = 1200
def excellent_score_condition := (120 : ℝ) > 90

-- Statement
theorem number_of_excellent_students : 
  normal_distribution → probability_condition → students_participated → excellent_score_condition → 
  ∃ num_excellent_students : ℕ, num_excellent_students = 120 :=
begin
  -- Proof is omitted
  sorry
end

end number_of_excellent_students_l647_647766


namespace minimum_surface_area_sum_l647_647285

noncomputable def volume := 420
noncomputable def AB := 13
noncomputable def BC := 14
noncomputable def CA := 15

theorem minimum_surface_area_sum :
  ∃ (m n k : ℕ), let surface_area := m + n * (Real.sqrt k) in
  surface_area = 84 + 21 * (Real.sqrt 241) ∧ m + n + k = 346 :=
by
  use 84
  use 21
  use 241
  split
  · sorry -- The actual surface area proof would go here
  · sorry -- The sum verification proof would go here

end minimum_surface_area_sum_l647_647285


namespace monotonicity_intervals_range_a_l647_647139

noncomputable def f (x a : ℝ) : ℝ := Real.exp x * (x^2 + a*x + a)

theorem monotonicity_intervals (a : ℝ) :
  (a = 1) →
  (∀ x, (f x 1)) is_increasing_on (-∞, -2) ∨ (-1, +∞)) →
  (f x 1) is_decreasing_on (-2, -1)) :=
by
  intro ha
  sorry

theorem range_a (a : ℝ) :
  (∀ x, (f x a) ≤ Real.exp a on [a, +∞)) →
  (a ∈ (-∞, 1/2])) :=
by
  intro ha
  sorry

end monotonicity_intervals_range_a_l647_647139


namespace double_chess_first_player_can_draw_l647_647723

-- Define the basic structure and rules of double chess
structure Game :=
  (state : Type)
  (move : state → state)
  (turn : ℕ → state → state)

-- Define the concept of double move
def double_move (g : Game) (s : g.state) : g.state :=
  g.move (g.move s)

-- Define a condition stating that the first player can at least force a draw
theorem double_chess_first_player_can_draw
  (game : Game)
  (initial_state : game.state)
  (double_move_valid : ∀ s : game.state, ∃ s' : game.state, s' = double_move game s) :
  ∃ draw : game.state, ∀ second_player_strategy : game.state → game.state, 
    double_move game initial_state = draw :=
  sorry

end double_chess_first_player_can_draw_l647_647723


namespace sum_of_solutions_eq_zero_l647_647427

noncomputable def f (x : ℝ) := 3 ^ |x| + 2 * |x|

theorem sum_of_solutions_eq_zero : 
  (∑ x in {x : ℝ | f x = 20}, x) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l647_647427


namespace sec_neg_405_eq_sqrt_2_l647_647437

theorem sec_neg_405_eq_sqrt_2 : Real.sec (-405 * Real.pi / 180) = Real.sqrt 2 :=
by
  -- Using the definition of secant and the periodicity of cosine
  have h1 : Real.sec x = 1 / Real.cos x := Real.sec_def
  have h2 : Real.cos x = Real.cos (x + 2 * Real.pi) := Real.cos_periodic x 2
  have h3 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := sorry
  rw [h1]
  rw [Real.cos_eq_cos_of_periodic (-405 * Real.pi / 180) (-405 * Real.pi / 180 + 2 * Real.pi)]
  rw [Real.cos_add (-405 * Real.pi / 180) 2 * Real.pi]
  simp only
  rw [Real.cos_315]

end sec_neg_405_eq_sqrt_2_l647_647437


namespace balls_to_boxes_distribution_l647_647572

theorem balls_to_boxes_distribution :
  (StirlingS2 5 1) + (StirlingS2 5 2) + (StirlingS2 5 3) = 41 :=
by
  sorry

end balls_to_boxes_distribution_l647_647572


namespace expand_polynomials_l647_647067

def p (z : ℝ) : ℝ := 3 * z ^ 2 + 4 * z - 7
def q (z : ℝ) : ℝ := 4 * z ^ 3 - 3 * z + 2

theorem expand_polynomials :
  (p z) * (q z) = 12 * z ^ 5 + 16 * z ^ 4 - 37 * z ^ 3 - 6 * z ^ 2 + 29 * z - 14 := by
  sorry

end expand_polynomials_l647_647067


namespace triangle_side_length_l647_647790

theorem triangle_side_length :
  ∀ (a b c : ℝ), a = 30 ∧ b = 45 ∧ c = 105 ∧ 
  (∃ (side_b : ℝ), side_b = 8) ∧ 
  (∀ (side_a : ℝ), side_a = 4√2) → 
  side_a = 4√2 :=
by {
  sorry
}

end triangle_side_length_l647_647790


namespace find_AB_l647_647713

noncomputable def AB (l α β : ℝ) : ℝ :=
  l / (2 * Real.sin α) * (Real.sin β + Real.sqrt (8 * (Real.sin α)^2 + (Real.sin β)^2))

theorem find_AB 
  (x l α β : ℝ)
  (hk : 2 * x = 2 * l * (((Real.sin α)^2) + ((Real.sin β)^2) + Real.sqrt (8 * ((Real.sin α)^2) + ((Real.sin β)^2))))
  (hak : x = x)
  (hAL : AL = l)
  (hBCK : ∠BCK = α)
  (hCBL : ∠CBL = β)
  : AB l α β = (l / (2 * Real.sin α)) * (Real.sin β + Real.sqrt (8 * (Real.sin α)^2 + (Real.sin β)^2)) :=
sorry

end find_AB_l647_647713


namespace product_mod_7_l647_647814

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l647_647814


namespace smallest_d_distance_l647_647775

noncomputable def point_distance_origin (d : ℝ) : ℝ :=
  real.sqrt ((4 * real.sqrt 5)^2 + (d + 4)^2)

theorem smallest_d_distance (d : ℝ) : 
  point_distance_origin d = 4 * d → d ≈ 2.81 :=
by
  sorry

end smallest_d_distance_l647_647775


namespace questionnaires_drawn_from_unit_D_l647_647252

theorem questionnaires_drawn_from_unit_D 
  (total_sample: ℕ) 
  (sample_from_B: ℕ) 
  (d: ℕ) 
  (h_total_sample: total_sample = 150) 
  (h_sample_from_B: sample_from_B = 30) 
  (h_arithmetic_sequence: (30 - d) + 30 + (30 + d) + (30 + 2 * d) = total_sample) 
  : 30 + 2 * d = 60 :=
by 
  sorry

end questionnaires_drawn_from_unit_D_l647_647252


namespace correctness_of_props_l647_647138

variable {U : Type*} (A B : Set U) (x : U)

def characteristic_function (P : Set U) (x : U) : ℕ :=
  if x ∈ P then 1 else 0

lemma prop1 (h : A ⊆ B) :
  characteristic_function A x ≤ characteristic_function B x :=
sorry

lemma prop2 :
  characteristic_function (Aᶜ) x = 1 - characteristic_function A x :=
sorry

lemma prop3 :
  characteristic_function (A ∩ B) x = characteristic_function A x * characteristic_function B x :=
sorry

theorem correctness_of_props :
  ∀ (A B : Set U) (x : U), 
  (characteristic_function (A ∩ B) x = characteristic_function A x * characteristic_function B x) ∧
  (characteristic_function (Aᶜ) x = 1 - characteristic_function A x) ∧
  ∀ (h : A ⊆ B), characteristic_function A x ≤ characteristic_function B x :=
by {
  intros,
  split,
  exact prop3,
  split,
  exact prop2,
  exact prop1 h
}

end correctness_of_props_l647_647138


namespace zero_of_sum_of_squares_eq_zero_l647_647269

theorem zero_of_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_of_sum_of_squares_eq_zero_l647_647269


namespace coffee_cost_per_week_l647_647208

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end coffee_cost_per_week_l647_647208


namespace num_of_valid_installations_l647_647318

-- Define the vertices of the signal tower
inductive Vertex
| P | A | B | C | A1 | B1 | C1

-- Define the condition that two ends of the same line segment must have different colors
structure ValidAssignment (assignment : Vertex → ℕ) : Prop :=
(different_colors : ∀ v1 v2 : Vertex, (v1, v2) ∈ edges → assignment v1 ≠ assignment v2)

-- Define the main theorem statement
theorem num_of_valid_installations :
  ∃ (assignment : Vertex → ℕ), ValidAssignment assignment ∧
  (∑ (P = 4) * (∑ (A = 3)) * (∑ (B = 3)) * (∑ (C = 3)) * (∑ (A1 = 3)) * (∑ (B1 = 3)) * (∑ (C1 = 3)) = 2916 :=
begin
  sorry
end

end num_of_valid_installations_l647_647318


namespace probability_of_selecting_two_captains_l647_647317

def probability_two_captains : ℚ :=
  let team1_size, team2_size, team3_size := 6, 9, 10
  let co_captains := 2
  let prob_team := 1 / 3
  let prob_captains (n : ℕ) : ℚ := co_captains / (finset.choose (n, 2)).val
  prob_team * (prob_captains team1_size + prob_captains team2_size + prob_captains team3_size)

theorem probability_of_selecting_two_captains : probability_two_captains = 7 / 180 := 
by
  sorry

end probability_of_selecting_two_captains_l647_647317


namespace local_value_of_6_in_product_of_face_value_and_local_value_l647_647337

-- Define the face value
def face_value (n : ℕ) : ℕ := n

-- Calculate the local value of a digit in a number
def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * position

-- Example number 7098060
def num : ℕ := 7098060

-- Problem statement: Prove that the local value of 6 in the product of the face value of 7 and the local value of 8 is equal to 60
theorem local_value_of_6_in_product_of_face_value_and_local_value :
  let face_val_7 := face_value 7 in
  let local_val_8 := local_value 8 1000 in
  let product := face_val_7 * local_val_8 in
  local_value 6 10 = 60 :=
by
  let face_val_7 := face_value 7
  let local_val_8 := local_value 8 1000
  let product := face_val_7 * local_val_8
  have h := local_value 6 10
  sorry

end local_value_of_6_in_product_of_face_value_and_local_value_l647_647337


namespace jordan_wins_two_games_l647_647179

theorem jordan_wins_two_games 
  (Peter_wins : ℕ) 
  (Peter_losses : ℕ)
  (Emma_wins : ℕ) 
  (Emma_losses : ℕ)
  (Jordan_losses : ℕ) 
  (hPeter : Peter_wins = 5)
  (hPeterL : Peter_losses = 4)
  (hEmma : Emma_wins = 4)
  (hEmmaL : Emma_losses = 5)
  (hJordanL : Jordan_losses = 2) : ∃ (J : ℕ), J = 2 :=
by
  -- The proof will go here
  sorry

end jordan_wins_two_games_l647_647179


namespace conjugate_quadrant_l647_647106

theorem conjugate_quadrant (z : ℂ) (h : (1 + complex.i) * z = 2 + complex.i) :
  let conj_z := conj z in conj_z.re > 0 ∧ conj_z.im > 0 :=
by
  sorry

end conjugate_quadrant_l647_647106


namespace ratio_students_l647_647316

theorem ratio_students (boys girls total_students : ℕ) (h1 : boys = 20) (h2 : girls = 11) (h3 : total_students = 93) :
    let second_grade_students := boys + girls,
        r := (total_students - second_grade_students) / second_grade_students
    in r = 2 :=
by
  have h4 : second_grade_students = boys + girls from rfl,
  have h5 : second_grade_students = 31 := by rw [h1, h2]
  have h6 : total_students = 93 := h3,
  have h7 : r = (total_students - second_grade_students) / second_grade_students := rfl,
  sorry

end ratio_students_l647_647316


namespace dishonest_dealer_weight_l647_647370

theorem dishonest_dealer_weight (p : ℝ) (h : p = 53.84615384615387) : 
  let weight_per_kg := (100 - p) / 100 in
  weight_per_kg = 0.4615384615384613 :=
by
  sorry

end dishonest_dealer_weight_l647_647370


namespace triangle_NLM_acute_l647_647199

theorem triangle_NLM_acute (A B C L M N : Point) (h : ∃ (I : Point), incircle_touch_points A B C I L M N)
  : acute_triangle L M N :=
sorry

end triangle_NLM_acute_l647_647199


namespace find_n_mod_10_l647_647471

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l647_647471


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l647_647669

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l647_647669


namespace Evan_earnings_Markese_less_than_Evan_l647_647647

-- Definitions from conditions
def MarkeseEarnings : ℕ := 16
def TotalEarnings : ℕ := 37

-- Theorem statements
theorem Evan_earnings (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E = 21 :=
by {
  sorry
}

theorem Markese_less_than_Evan (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E - MarkeseEarnings = 5 :=
by {
  sorry
}

end Evan_earnings_Markese_less_than_Evan_l647_647647


namespace cube_greater_than_quadratic_minus_linear_plus_one_l647_647627

variable (x : ℝ)

theorem cube_greater_than_quadratic_minus_linear_plus_one (h : x > 1) :
  x^3 > x^2 - x + 1 := by
  sorry

end cube_greater_than_quadratic_minus_linear_plus_one_l647_647627


namespace combined_transformation_matrix_l647_647496

open Matrix

variable {R : Type*} [CommRing R]

def dilation_matrix (s : R) : Matrix (Fin 3) (Fin 3) R :=
  diagonal ![s, s, s]

def reflection_xy_plane_matrix : Matrix (Fin 3) (Fin 3) R :=
  λ i j, if i = j 
         then if i = 2 then -1 else 1
         else 0

theorem combined_transformation_matrix :
  dilation_matrix 2 * reflection_xy_plane_matrix = 
  ![![2, 0, 0], ![0, 2, 0], ![0, 0, -2]] :=
by
  sorry

end combined_transformation_matrix_l647_647496


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647816

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647816


namespace members_with_both_non_athletic_parents_l647_647591

-- Let's define the conditions
variable (total_members athletic_dads athletic_moms both_athletic none_have_dads : ℕ)
variable (H1 : total_members = 50)
variable (H2 : athletic_dads = 25)
variable (H3 : athletic_moms = 30)
variable (H4 : both_athletic = 10)
variable (H5 : none_have_dads = 5)

-- Define the conclusion we want to prove
theorem members_with_both_non_athletic_parents : 
  (total_members - (athletic_dads + athletic_moms - both_athletic) + none_have_dads - total_members) = 10 :=
sorry

end members_with_both_non_athletic_parents_l647_647591


namespace probability_red_chips_drawn_first_l647_647014

def probability_all_red_drawn (total_chips : Nat) (red_chips : Nat) (green_chips : Nat) : ℚ :=
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose (total_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem probability_red_chips_drawn_first :
  probability_all_red_drawn 9 5 4 = 4 / 9 :=
by
  sorry

end probability_red_chips_drawn_first_l647_647014


namespace range_of_m_l647_647121

theorem range_of_m (m : ℝ) :
  (∀ e : ℝ, (sqrt 6) / 2 < e ∧ e < sqrt 2 →
   (∃ a b c : ℝ, a^2 = 5 ∧ b^2 = m ∧ c^2 = 5 + m ∧ e = c / a) →
  2.5 < m ∧ m ≤ 3 ∨ 5 ≤ m ∧ m < 9) ∧
  (∀ x y : ℝ, (9 - m > 0 ∧ 2m > 9 - m ∧ 2m > 0) →
  ∃ p q : ℝ, p = 2m ∧ q = 9 - m) →
  ((2.5 < m ∧ m ≤ 3) ∨ (5 ≤ m ∧ m < 9)) ∧ ¬((2.5 < m ∧ m ≤ 3) ∧ (5 ≤ m ∧ m < 9)) :=
sorry

end range_of_m_l647_647121


namespace triangle_inequality_l647_647852

theorem triangle_inequality (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_triangle : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2)) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end triangle_inequality_l647_647852


namespace min_teams_for_highest_score_l647_647387

-- Definitions based on the conditions
def points (wins : ℕ) (draws : ℕ) : ℕ := 2 * wins + draws

def has_highest_score (A_points : ℕ) (other_points : List ℕ) : Prop :=
  ∀ p ∈ other_points, A_points > p

def fewer_wins (A_wins : ℕ) (other_wins : List ℕ) : Prop :=
  ∀ w ∈ other_wins, A_wins < w

-- The theorem statement
theorem min_teams_for_highest_score :
  ∃ (n : ℕ), (n ≥ 2) ∧
  (∀ (A_wins A_draws : ℕ) (other_wins : List ℕ) (other_scores : List ℕ),
    let A_points := points A_wins A_draws,
        min_other_wins := A_wins + 1,
        min_other_points := 2 * min_other_wins in
    (A_points > min_other_points) →
    (∃ (others : List ℕ), 
      -- include team points in other_scores
      ∀ op ∈ other_wins, points op (n-1) = other_scores.nth_le (op-1) sorry)
      (has_highest_score A_points other_scores) ∧ 
      (fewer_wins A_wins other_wins)) →
  n = 6 :=
begin
  sorry -- Proof omitted as per instructions
end

end min_teams_for_highest_score_l647_647387


namespace find_y_l647_647167

theorem find_y
  (x y : ℝ)
  (h1 : x^(3*y) = 8)
  (h2 : x = 2) :
  y = 1 :=
sorry

end find_y_l647_647167


namespace slope_A1B_l647_647553

theorem slope_A1B (x1 y1 x2 y2 : ℝ) (h_parabola_A : y1^2 = 4 * x1) 
  (h_parabola_B : y2^2 = 4 * x2) (h_focus : (1,0)) 
  (h_A_below_x_axis : y1 < 0) (h_symmetric : (-y1) = -y1) 
  (h_slope_AB : (y2 - y1) / (x2 - x1) = 1) :
  (y2 - (-y1)) / (x2 - x1) = (Real.sqrt 2) / 2 :=
sorry

end slope_A1B_l647_647553


namespace competition_solved_exactly_five_problems_l647_647757

theorem competition_solved_exactly_five_problems
  (n : ℕ)
  (solves_pair : Π (P_i P_j : ℕ), 1 ≤ P_i ∧ P_i < P_j ∧ P_j ≤ 6 → ∃ x : ℕ, x > (2 * n / 5))
  (no_all_solved : ∀ C : ℕ, ∀ problems_solving : ℕ → bool, (∀ P : ℕ, P ∈ [1, 2, 3, 4, 5, 6] → problems_solving P) → ¬(problems_solving = λ P, true)) :
  ∃ (C1 C2 : ℕ), C1 ≠ C2 ∧ (∃ S1 S2 : ℕ, S1 = 5 ∧ S2 = 5) :=
sorry

end competition_solved_exactly_five_problems_l647_647757


namespace eccentricity_of_given_curve_l647_647697

def param_eq_ellipse (φ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos φ, Real.sqrt 5 * Real.sin φ)

def eccentricity (a b c : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_given_curve :
  (∃ φ : ℝ, param_eq_ellipse φ = (3 * Real.cos φ, Real.sqrt 5 * Real.sin φ)) →
  let a := 3
  let b := Real.sqrt 5
  let c := Real.sqrt (a^2 - b^2)
  eccentricity 3 (Real.sqrt 5) (Real.sqrt (3^2 - (Real.sqrt 5)^2)) = 2 / 3 :=
by
  intros
  sorry

end eccentricity_of_given_curve_l647_647697


namespace sequence_prime_count_l647_647350

def construct_sequence : ℕ → ℕ
| n := let
    seq := [2, 0, 1, 6]
  in seq[n % 4]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sequence_prime_count : ∀ n : ℕ, n ≥ 0 → (construct_sequence 1 = 2) ∧ (∀ m : ℕ, m ∈ [2, 0, 1, 6] → is_prime m → m ≠ 0 ∧ m ≠ 6 ∧ m ≠ 1) → (construct_sequence 1) = 2 :=
by
  sorry

end sequence_prime_count_l647_647350


namespace third_line_passes_through_integer_point_l647_647981

theorem third_line_passes_through_integer_point (a b c : ℝ)
  (h1 : ∃ y : ℝ, y = a * 1 + b ∧ y = b * 1 + c) :
  ∃ x y : ℤ, y = a * x + a := 
begin
  sorry
end

end third_line_passes_through_integer_point_l647_647981


namespace find_ab_over_ten_l647_647824

noncomputable def double_factorial (n : ℕ) : ℕ := 
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

noncomputable def sum_double_factorials : ℚ := 
  ∑ i in finset.range 2010, (double_factorial (2 * i - 1) : ℚ) / double_factorial (2 * i)

theorem find_ab_over_ten : 
  let S := (sum_double_factorials).num
  let d := (sum_double_factorials).denom in
  let denominator := d.factorization.find_denom d in
  denominator.odd = (denominator >>= (λ b, d / (2^denominator.head, b.head)), 10}
  ab / 10 = 400.4 := sorry

end find_ab_over_ten_l647_647824


namespace part_I_part_II_l647_647892

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1))

theorem part_I (a : ℝ) (h_a_pos : a > 0) : (∀ x > 0, (1 / ((2 * x + 1) * (a * (2 * x + 1) - (2 * (a * x + 1) / 2))) ≥ 0) ↔ a ≥ 2) :=
sorry

theorem part_II : ∃ a : ℝ, (∀ x > 0, (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1)) ≥ 1) ∧ (Real.log (a * (Real.sqrt ((2 - a) / (4 * a))) + 1 / 2) + (2 / (2 * (Real.sqrt ((2 - a) / (4 * a))) + 1)) = 1) ∧ a = 1 :=
sorry

end part_I_part_II_l647_647892


namespace range_of_a_l647_647537

-- Define the function f
def f (a x : ℝ) := (x^3 + 3 * x^2 + 9 * (a + 6) * x + 6 - a) * Real.exp (-x)

-- Define the polynomial g(x)
def g (a x : ℝ) := -x^3 - (9 * a + 48) * x + 10 * a + 48

-- Define the condition that f has a local maximum in the interval (2,4)
def has_local_maximum_interval (a : ℝ) : Prop := 
  (g a 2 > 0) ∧ (g a 4 < 0)

-- State the range of a
theorem range_of_a (a : ℝ) : has_local_maximum_interval a → -8 < a ∧ a < -7 := by 
  sorry

end range_of_a_l647_647537


namespace integer_partition_impossible_l647_647605

theorem integer_partition_impossible :
  ¬ (∃ (S1 S2 S3 : set ℤ), 
        (∀ n : ℤ, n ∈ S1 ∨ n ∈ S2 ∨ n ∈ S3) ∧ -- Every integer is in one of S1, S2, S3
        (∀ n : ℤ, 
          ((n ∈ S1 ∨ n ∈ S2) ∧ ¬(n ∈ S1 ∧ n ∈ S2)) ∧ -- n can not belong to more than one subset
          ((n-50 ∈ S1 ∨ n-50 ∈ S2) ∧ ¬((n-50 ∈ S1 ∧ n-50 ∈ S2))) ∧ -- similar for n-50
          ((n+1987 ∈ S1 ∨ n+1987 ∈ S2) ∧ ¬((n+1987 ∈ S1 ∧ n+1987 ∈ S2))) ∧ -- and for n+1987
          ((n ∈ S1 ∧ n-50 ∈ S2 ∧ n+1987 ∈ S3) ∨ -- all three numbers must be in different subsets
           (n ∈ S1 ∧ n-50 ∈ S3 ∧ n+1987 ∈ S2) ∨ 
           (n ∈ S2 ∧ n-50 ∈ S1 ∧ n+1987 ∈ S3) ∨
           (n ∈ S2 ∧ n-50 ∈ S3 ∧ n+1987 ∈ S1) ∨
           (n ∈ S3 ∧ n-50 ∈ S1 ∧ n+1987 ∈ S2) ∨
           (n ∈ S3 ∧ n-50 ∈ S2 ∧ n+1987 ∈ S1))))
) :=
begin
  sorry
end

end integer_partition_impossible_l647_647605


namespace measure_angle_ABC_l647_647692

theorem measure_angle_ABC (x : ℝ) (h1 : ∃ θ, θ = 180 - x ∧ x / 2 = (180 - x) / 3) : x = 72 :=
by
  sorry

end measure_angle_ABC_l647_647692


namespace pair_count_correct_l647_647983

theorem pair_count_correct :
  (2:ℝ)^3075 > (3:ℝ)^4096 ∧ (2:ℝ)^3075 < (3:ℝ)^4097 →
  (∃ n, n = 12288 ∧ ∀ a b, 1 ≤ a ∧ a ≤ 4095 → 3^b < 2^a ∧ 2^a < 2^(a+3) ∧ 2^(a+3) < 3^(b+1) ⟹ n = 12288) :=
  by
  intros h
  sorry

end pair_count_correct_l647_647983


namespace original_volume_l647_647020

theorem original_volume (V : ℝ) (h1 : V > 0) 
    (h2 : (1/16) * V = 0.75) : V = 12 :=
by sorry

end original_volume_l647_647020


namespace min_OP_norm_l647_647198

noncomputable def minimum_OP_norm (OA OB : ℝ) (λ μ : ℝ) (h1 : |OB| = √2) (h2 : |BA| = 1) (h3 : angle A O B = 45) (h4 : OP = λ • OA + μ • OB) (h5 : 2 * λ + μ = 3) : ℝ :=
  sqrt((3 - λ)^2 + (3 - 2 * λ)^2)

theorem min_OP_norm (OA OB : ℝ) (λ μ : ℝ) 
  (h1 : |OB| = √2) 
  (h2 : |BA| = 1) 
  (h3 : angle A O B = 45) 
  (h4 : OP = λ • OA + μ • OB) 
  (h5 : 2 * λ + μ = 3) : 
  minimum_OP_norm OA OB λ μ h1 h2 h3 h4 h5 = 3 * √5 / 5 :=
begin
  sorry
end

end min_OP_norm_l647_647198


namespace number_of_subsets_number_of_sets_satisfying_conditions_l647_647125

open Finset

theorem number_of_subsets (X : Finset ℕ) :
  {1, 2} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5} ↔
  ∃ (Y : Finset ℕ), Y ⊆ {3, 4, 5} ∧ X = {1, 2} ∪ Y :=
sorry

theorem number_of_sets_satisfying_conditions :
  ∃ n : ℕ, 
  (number_of_subsets (X : Finset ℕ) → n = 8) :=
sorry

end number_of_subsets_number_of_sets_satisfying_conditions_l647_647125


namespace translation_transformation_l647_647398

def phenomenon_A : Prop :=
∀ (bubbles : ℕ), ∃ size : ℝ, (position : ℝ) → size ≠ constant ∧ position ≠ constant

def phenomenon_B : Prop :=
∀ (drawer : ℝ), ∀ (distance : ℝ), drawer + distance = constant

def phenomenon_C : Prop :=
∀ (minute_hand : ℝ), ∀ (angle : ℝ), minute_hand = constant ∧ minute_hand = rotate(minute_hand, angle)

def phenomenon_D : Prop :=
∀ (leaves : ℕ), ∃ path : ℝ, (orientation : ℝ) → path = complex ∧ orientation = complex

theorem translation_transformation : phenomenon_B := 
sorry

end translation_transformation_l647_647398


namespace length_of_AB_l647_647330

variable (AB CD BC : ℝ)
variable (O P : Point)
variable (AC BD: LineSegment)
variable (OP : ℝ)
variable (M : Midpoint BD)

-- Given conditions
def is_trapezoid (ABCD : Trapezoid) : Prop := 
  ABCD.AB ∥ ABCD.CD ∧ 
  ABCD.BC.length = 34 ∧ 
  ABCD.CD.length = 34 ∧ 
  ABCD.BC ⊥ ABCD.CD ∧ 
  AC.Intersection BD = O ∧ 
  P.isMidpoint BD ∧ 
  distance O P = 8

-- Theorem statement
theorem length_of_AB (ABCD : Trapezoid)
  (h : is_trapezoid ABCD) : 
  ABCD.AB.length = 34 :=
sorry

end length_of_AB_l647_647330


namespace polynomial_remainder_l647_647232

noncomputable def R (z : ℤ) : ℤ := -z - 1

theorem polynomial_remainder (Q R : Polynomial ℤ) (z: ℤ) :
    z^2023 - 1 = (z^2 + z + 1) * Q + R ∧ (Polynomial.degree R < 2) →
    R = -z - 1 :=
begin
  sorry
end

end polynomial_remainder_l647_647232


namespace credit_card_more_profitable_l647_647688

theorem credit_card_more_profitable
  (N : ℕ)
  (H : N > 30) :
  let ticket_cost := 20000
  let credit_cashback := ticket_cost * 0.005
  let debit_cashback := ticket_cost * 0.01
  let annual_interest_rate := 0.06
  let monthly_interest_rate := annual_interest_rate / 12
  let earned_interest := (monthly_interest_rate * ticket_cost * N) / 30
  let total_credit_card_benefit := earned_interest + credit_cashback
  let total_debit_card_benefit := debit_cashback
  in total_credit_card_benefit > total_debit_card_benefit :=
by {
  let ticket_cost := 20000
  let credit_cashback := 100  -- 0.5% of 20000
  let debit_cashback := 200   -- 1% of 20000
  let annual_interest_rate := 0.06
  let monthly_interest_rate := annual_interest_rate / 12
  let earned_interest := (monthly_interest_rate * ticket_cost * N) / 30
  let total_credit_card_benefit := earned_interest + credit_cashback
  let total_debit_card_benefit := debit_cashback
  sorry
}

end credit_card_more_profitable_l647_647688


namespace length_of_IJ_l647_647748

noncomputable def cyclic_quadrilateral_ABCD : Type :=
  Σ (ABCD : ℝ) (AB : ℝ) (AD : ℝ) (AC : ℝ) (I J : ℝ → ℝ) (b: Structure)

namespace proof

open proof 

theorem length_of_IJ (ABCD : cyclic_quadrilateral_ABCD) :
  ∃ (IJ_length : ℝ),
    ABCD.2 AB = 49 → ABCD.2 AD = 49 → ABCD.2 AC = 73 → 
    (BD bisects IJ) →
    IJ_length = (28 / 5) * sqrt 69 :=
by
  sorry

end proof

end length_of_IJ_l647_647748


namespace x1_minus_x2_eq_m_pi_l647_647991

noncomputable def y (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range n, (cos (a i + x)) / (2 ^ i)

theorem x1_minus_x2_eq_m_pi (a : ℕ → ℝ) (n : ℕ) (x1 x2 : ℝ)
  (h1 : y a n x1 = 0) (h2 : y a n x2 = 0) :
  ∃ m : ℤ, x1 - x2 = m * π :=
sorry

end x1_minus_x2_eq_m_pi_l647_647991


namespace four_digit_number_divisible_by_36_l647_647856

theorem four_digit_number_divisible_by_36 (n : ℕ) (h₁ : ∃ k : ℕ, 6130 + n = 36 * k) 
  (h₂ : ∃ k : ℕ, 130 + n = 4 * k) 
  (h₃ : ∃ k : ℕ, (10 + n) = 9 * k) : n = 6 :=
sorry

end four_digit_number_divisible_by_36_l647_647856


namespace tan_x_plus_pi_over_4_l647_647914

theorem tan_x_plus_pi_over_4 (x : ℝ) (hx : Real.tan x = 2) : Real.tan (x + Real.pi / 4) = -3 :=
by
  sorry

end tan_x_plus_pi_over_4_l647_647914


namespace problem1_problem2_l647_647143

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
def f (x : ℝ) : ℝ := abs (x - a) + 2 * abs (x + b)

theorem problem1 (h3 : ∃ x, f x = 1) : a + b = 1 := sorry

theorem problem2 (h4 : a + b = 1) (m : ℝ) (h5 : ∀ m, m ≤ 1/a + 2/b)
: m ≤ 3 + 2 * Real.sqrt 2 := sorry

end problem1_problem2_l647_647143


namespace toothbrush_difference_l647_647059

theorem toothbrush_difference :
  ∀ (toothbrushes_total jan feb mar apr may : ℕ),
    toothbrushes_total = 330 ∧
    jan = 53 ∧
    feb = 67 ∧
    mar = 46 ∧
    apr = (toothbrushes_total - (jan + feb + mar)) / 2 ∧
    may = (toothbrushes_total - (jan + feb + mar)) / 2 →
    (67 - 46) = 21 :=
by intros toothbrushes_total jan feb mar apr may h;
   cases h with h_total h;
   cases h with h_jan h;
   cases h with h_feb h;
   cases h with h_mar h;
   cases h with h_apr h_may;
   sorry

end toothbrush_difference_l647_647059


namespace fraction_to_decimal_l647_647435

theorem fraction_to_decimal : (5 / 18 : ℚ) = 0. \overline{277778} :=
sorry

end fraction_to_decimal_l647_647435


namespace thank_you_cards_needed_l647_647972

theorem thank_you_cards_needed
  (invitations : ℕ)
  (rsvp_rate : ℝ)
  (show_rate : ℝ)
  (no_gift_count : ℕ)
  (h_invitations : invitations = 200)
  (h_rsvp_rate : rsvp_rate = 0.90)
  (h_show_rate : show_rate = 0.80)
  (h_no_gift_count : no_gift_count = 10) :
  (let rsvp_count := (rsvp_rate * invitations).toNat in
   let show_up_count := (show_rate * rsvp_count).toNat in
   let thank_you_cards := show_up_count - no_gift_count in
   thank_you_cards = 134) :=
by sorry

end thank_you_cards_needed_l647_647972


namespace odd_function_iff_l647_647243

section
variables (a b : ℝ)

def f (x : ℝ) : ℝ := x * abs (x - a) + b

theorem odd_function_iff : (∀ x : ℝ, f(a, b, x) = f(a, b, -x)) ↔ (a^2 + b^2 = 0) :=
by
  sorry

end odd_function_iff_l647_647243


namespace arithmetic_square_root_16_l647_647289

theorem arithmetic_square_root_16 : ∃ (x : ℝ), x * x = 16 ∧ x ≥ 0 ∧ x = 4 := by
  sorry

end arithmetic_square_root_16_l647_647289


namespace max_distinct_total_sums_l647_647754

-- Define the notion of a regular 5x5 table
def is_regular_table (table : Matrix (Fin 5) (Fin 5) ℝ) (nums : Fin 4 → ℝ) : Prop :=
  ∀ (i j : Fin 4), ∃ (a b c d : Fin 5 × Fin 5),
  ({a, b, c, d}.card = 4 ∧
  table a = nums i ∧
  table b = nums j ∧
  table c ≠ table a ∧ table c ≠ table b ∧
  table d ≠ table a ∧ table d ≠ table b ∧ table d ≠ table c ∧
  a.fst ≠ b.fst ∧ a.snd ≠ b.snd ∧
  (a.fst, b.fst, c.fst, d.fst).pairwise (≠) ∧
  (a.snd, b.snd, c.snd, d.snd).pairwise (≠))

-- Define the total sum of a table
def total_sum (table : Matrix (Fin 5) (Fin 5) ℝ) : ℝ :=
  ∑ i j, table i j

-- Statement of the maximum distinct total sums problem
theorem max_distinct_total_sums :
  ∀ (A B C D : ℝ), 60 = (Set.card (Set.range (λ (table : {table : Matrix (Fin 5) (Fin 5) ℝ // is_regular_table table ![A, B, C, D]}), total_sum table.val)))
:= sorry

end max_distinct_total_sums_l647_647754


namespace find_solutions_l647_647356

def system_solutions (x y z : ℝ) : Prop :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solutions :
  ∃ (x y z : ℝ), system_solutions x y z ∧ ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end find_solutions_l647_647356


namespace coffee_cost_per_week_l647_647209

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end coffee_cost_per_week_l647_647209


namespace probability_of_even_product_l647_647858

theorem probability_of_even_product (p : ℕ) (q : ℕ) (hp : 1 ≤ p ∧ p ≤ 8) (hq : 1 ≤ q ∧ q ≤ 8) :
  (∃ p q, p * q % 2 = 0) ∧ p ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ q ∈ {1, 2, 3, 4, 5, 6, 7, 8} :=
begin
  sorry
end

lemma probability_of_even_product_is_3_4 :
  (∃ p q, p * q % 2 = 0) ∧ p ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ q ∈ {1, 2, 3, 4, 5, 6, 7, 8} :=
by sorry

end probability_of_even_product_l647_647858


namespace kennedy_softball_park_miles_l647_647614

theorem kennedy_softball_park_miles :
  let miles_per_gallon := 19
  let gallons_of_gas := 2
  let total_drivable_miles := miles_per_gallon * gallons_of_gas
  let miles_to_school := 15
  let miles_to_burger_restaurant := 2
  let miles_to_friends_house := 4
  let miles_home := 11
  total_drivable_miles - (miles_to_school + miles_to_burger_restaurant + miles_to_friends_house + miles_home) = 6 :=
by
  sorry

end kennedy_softball_park_miles_l647_647614


namespace find_m_if_real_l647_647530

-- Definitions: Imaginary unit and condition
def I : ℂ := complex.I

-- The main theorem statement
theorem find_m_if_real (m : ℂ) (h : (2 + I) * (m - 2 * I) ∈ ℝ) : m = 4 := 
  sorry

end find_m_if_real_l647_647530


namespace parallel_lines_distance_l647_647820

-- Define the points a, b and the direction vector d
def a : ℝ × ℝ := (4, 1)
def b : ℝ × ℝ := (5, 4)
def d : ℝ × ℝ := (2, -3)

-- Calculation of the vector v from a to b
def v : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

-- Projection of vector v onto d
def proj_v_on_d : ℝ × ℝ :=
  let scalar := (v.1 * d.1 + v.2 * d.2) / (d.1 * d.1 + d.2 * d.2)
  (scalar * d.1, scalar * d.2)

-- Calculation of the orthogonal vector w
def w : ℝ × ℝ := (v.1 - proj_v_on_d.1, v.2 - proj_v_on_d.2)

-- Calculating the norm of w
def norm_w : ℝ := Real.sqrt (w.1 * w.1 + w.2 * w.2)

-- The proof goal: proving the distance is 33/13
theorem parallel_lines_distance : norm_w = 33 / 13 := by
  sorry

end parallel_lines_distance_l647_647820


namespace find_a_and_c_range_of_m_l647_647543

theorem find_a_and_c (a c : ℝ) 
  (h : ∀ x, 1 < x ∧ x < 3 ↔ ax^2 + x + c > 0) 
  : a = -1/4 ∧ c = -3/4 := 
sorry

theorem range_of_m (m : ℝ) 
  (h : ∀ x, (-1/4)*x^2 + 2*x - 3 > 0 → x + m > 0) 
  : m ≥ -2 :=
sorry

end find_a_and_c_range_of_m_l647_647543


namespace impossible_mark_50_segments_l647_647202

theorem impossible_mark_50_segments : 
  ∀ (S : Set ℤ), S.card = 100 → 
  ¬ ∃ (a b : Fin 50 → ℤ), 
    (∀ i, b i - a i = i + 1) ∧
    (∀ i, a i ∈ S ∧ b i ∈ S) :=
by 
  sorry

end impossible_mark_50_segments_l647_647202


namespace unique_perpendicular_planes_l647_647095

variables {α β : Type} [Plane α] [Plane β] [Line m] [Line n]

def is_parallel (m : Line) (α : Plane) : Prop :=
  ∀ (p₁ p₂ : Point), (p₁ ∈ m → p₁ ∈ α) ∧ (p₂ ∈ m → p₂ ∈ α) → p₁ = p₂

def is_perpendicular (m : Line) (α : Plane) : Prop :=
  ∃ (v_m : Vector) (n_α : Vector), ortho v_m n_α

def is_perpendicular_planes (α β : Plane) : Prop :=
  ∃ (n_α : Vector) (n_β : Vector), ortho n_α n_β

axiom plane_contains_line (m : Line) (α : Plane) : Prop :=
  ∀ (p : Point), p ∈ m → p ∈ α

theorem unique_perpendicular_planes
  (m_perp_alpha : is_perpendicular m α)
  (n_perp_beta : is_perpendicular n β)
  (m_perp_n : ∃ (v_m : Vector) (v_n : Vector), ortho v_m v_n) :
  is_perpendicular_planes α β := sorry

end unique_perpendicular_planes_l647_647095


namespace jane_ends_with_crayons_l647_647211

-- Definitions for the conditions in the problem
def initial_crayons : Nat := 87
def crayons_eaten : Nat := 7
def packs_bought : Nat := 5
def crayons_per_pack : Nat := 10
def crayons_break : Nat := 3

-- Statement to prove: Jane ends with 127 crayons
theorem jane_ends_with_crayons :
  initial_crayons - crayons_eaten + (packs_bought * crayons_per_pack) - crayons_break = 127 :=
by
  sorry

end jane_ends_with_crayons_l647_647211


namespace num_valid_integer_n_l647_647507

theorem num_valid_integer_n : 
  (Finset.filter (λ n : ℤ, is_integer (8000 * (2/5)^ (2 * n)) (Finset.Icc 0 3)).card = 2 :=
by sorry

end num_valid_integer_n_l647_647507


namespace nested_expression_evaluation_l647_647803

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 :=
by
  sorry

end nested_expression_evaluation_l647_647803


namespace translated_graph_is_sin_2x_l647_647715

noncomputable def translated_function (x : ℝ) : ℝ :=
  sin (2 * x)

theorem translated_graph_is_sin_2x :
  ∀ x : ℝ, (λ x, sin (2 * (x + π / 6) + π / 3) + 2) (x - π / 6) - 2 = translated_function x :=
by
  intro x
  -- Here we would perform the translation step calculations
  sorry

end translated_graph_is_sin_2x_l647_647715


namespace ordinary_equation_C1_rectangular_equation_C2_area_ΔOAB_l647_647949

-- Define the parametric equation of C₁
def C1 (t : ℝ) : ℝ × ℝ := (t, 3 - t)

-- Define the polar equation of C₂
def C2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 4 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Prove the ordinary equation of C₁ is x + y = 3
theorem ordinary_equation_C1 : ∀ (t : ℝ), (C1 t).fst + (C1 t).snd = 3 := by
  intro t
  dsimp [C1]
  ring

-- Prove the rectangular coordinate equation of C₂ is x² + y² = 4y
theorem rectangular_equation_C2 : ∀ (θ : ℝ), 
  let (x, y) := C2 θ in x*x + y*y = 4 * y := by
  intro θ
  dsimp [C2]
  let ρ := 4 * Real.sin θ
  have h1: ρ = 4 * Real.sin θ := rfl
  have h2: y = ρ * Real.sin θ := rfl
  rw [h1, h2]
  simp only [ρ]
  sorry  -- Simplification and trigonometric manipulation

-- Define points A and B of intersection
def A : ℝ × ℝ := (1 + Real.sqrt 2, 2 - Real.sqrt 2)
def B : ℝ × ℝ := (1 - Real.sqrt 2, 2 + Real.sqrt 2)

-- Calculate the area of ΔOAB
theorem area_ΔOAB :
  let O := (0, 0) in 
  let area_ΔOAB := 
    (1 / 2) * |O.fst * (A.snd - B.snd) + A.fst * (B.snd - O.snd) + B.fst * (O.snd - A.snd)| in
  let area_sector := (1 / 2) * (2 * 2) * (Float.pi / 2) in
  area_ΔOAB = 2 * (Float.pi - 1) := by
  dsimp [A, B]
  sorry  -- Detailed area calculation based on given points and sector

end ordinary_equation_C1_rectangular_equation_C2_area_ΔOAB_l647_647949


namespace sum_of_cubes_divisible_by_9_l647_647671

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l647_647671


namespace find_n_mod_10_l647_647472

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l647_647472


namespace minimum_N_more_profitable_credit_card_l647_647689

-- Given conditions and definitions
def purchase_value : ℝ := 20000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.01
def annual_interest_rate : ℝ := 0.06
def days_in_year : ℝ := 360
def benefit_credit_card (N : ℝ) : ℝ := (annual_interest_rate * purchase_value * N / days_in_year) + (credit_card_cashback_rate * purchase_value)
def benefit_debit_card : ℝ := debit_card_cashback_rate * purchase_value

-- Assertion of the problem
theorem minimum_N_more_profitable_credit_card : ∃ N : ℕ, N > 30 :=
by
  sorry

end minimum_N_more_profitable_credit_card_l647_647689


namespace blood_pressure_systolic_diastolic_l647_647733

noncomputable def blood_pressure (t : ℝ) : ℝ :=
110 + 25 * Real.sin (160 * t)

theorem blood_pressure_systolic_diastolic :
  (∀ t : ℝ, blood_pressure t ≤ 135) ∧ (∀ t : ℝ, blood_pressure t ≥ 85) :=
by
  sorry

end blood_pressure_systolic_diastolic_l647_647733


namespace number_of_possible_integer_triangle_side_lengths_l647_647567

theorem number_of_possible_integer_triangle_side_lengths (a b c : ℕ) (h1 : a = 8) (h2 : b = 5)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) :
  (9 = (finset.filter (λ c, c > 3 ∧ c < 13) (finset.range 14)).card) :=
  sorry

end number_of_possible_integer_triangle_side_lengths_l647_647567


namespace toothbrush_difference_l647_647058

theorem toothbrush_difference :
  ∀ (toothbrushes_total jan feb mar apr may : ℕ),
    toothbrushes_total = 330 ∧
    jan = 53 ∧
    feb = 67 ∧
    mar = 46 ∧
    apr = (toothbrushes_total - (jan + feb + mar)) / 2 ∧
    may = (toothbrushes_total - (jan + feb + mar)) / 2 →
    (67 - 46) = 21 :=
by intros toothbrushes_total jan feb mar apr may h;
   cases h with h_total h;
   cases h with h_jan h;
   cases h with h_feb h;
   cases h with h_mar h;
   cases h with h_apr h_may;
   sorry

end toothbrush_difference_l647_647058


namespace empty_tank_time_l647_647743

-- Definitions based on problem conditions
def tank_full_fraction := 1 / 5
def pipeA_fill_time := 15
def pipeB_empty_time := 6

-- Derived definitions
def rate_of_pipeA := 1 / pipeA_fill_time
def rate_of_pipeB := 1 / pipeB_empty_time
def combined_rate := rate_of_pipeA - rate_of_pipeB 

-- The time to empty the tank when both pipes are open
def time_to_empty (initial_fraction : ℚ) (combined_rate : ℚ) : ℚ :=
  initial_fraction / -combined_rate

-- The main theorem to prove
theorem empty_tank_time
  (initial_fraction : ℚ := tank_full_fraction)
  (combined_rate : ℚ := combined_rate)
  (time : ℚ := time_to_empty initial_fraction combined_rate) :
  time = 2 :=
by
  sorry

end empty_tank_time_l647_647743


namespace silenos_time_l647_647798

theorem silenos_time :
  (∃ x : ℝ, ∃ b: ℝ, (x - 2 = x / 2) ∧ (b = x / 3)) → (∃ x : ℝ, x = 3) :=
by sorry

end silenos_time_l647_647798


namespace david_marks_english_l647_647823

theorem david_marks_english (M P C B : ℕ) (average : ℚ) 
  (hM : M = 92) (hP : P = 85) (hC : C = 87) (hB : B = 85) 
  (hAverage : average = 87.8) (subjects : ℕ) (hSubjects : subjects = 5) :
  let E := average * subjects - (M + P + C + B) in E = 90 :=
by
  have hTotalMarks : average * subjects = 439 := by sorry
  have hSum : M + P + C + B = 349 := by sorry
  have hE : E = 439 - 349 := by sorry
  have hE : E = 90 := by sorry
  sorry

end david_marks_english_l647_647823


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l647_647668

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l647_647668


namespace score_probability_l647_647755

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k := 0
| (n+1), (k+1) := binom n k + binom n (k+1)

def bag : list ℕ := [1, 1, 1, 1, 3, 3, 3] -- 1 point for red, 3 points for black

def draw_four (b: list ℕ): list (list ℕ) :=
(list.permutations b).map (λ x, x.take 4)

def score (l: list ℕ) : ℕ := l.sum

def X (b: list ℕ) (n: ℕ) : ℕ := (draw_four b).count (λ y, score y = n)

theorem score_probability :
  let b := bag, n := 4 in
  let P (k : ℕ) : ℝ := ((X b k).to_real / (binom 7 4).to_real) in
  (P 4 + P 6) = (13 / 35 : ℝ) := 
by
  sorry

end score_probability_l647_647755


namespace range_of_a_l647_647926

def f (a x : ℝ) : ℝ := 
  if x ≤ 0 then 
    x + 4 
  else 
    (a*x^2 + a*x - 1) / x

def has_two_base_pairs (a : ℝ) : Prop :=
  ∃ A B : ℝ, A ≠ B ∧ f a A = f a (-A) ∧ f a B = f a (-B) ∧ A < 0 ∧ B > 0

theorem range_of_a :
  {a : ℝ | has_two_base_pairs a} = set.Ioo (-6 + 2 * real.sqrt 6) 1 :=
sorry

end range_of_a_l647_647926


namespace L_shape_placements_l647_647091

theorem L_shape_placements (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : 
  ∃ k : ℕ, k = 4 * (m - 1) * (n - 1) := 
by
  use 4 * (m - 1) * (n - 1)
  refl

end L_shape_placements_l647_647091


namespace digit_B_prime_only_1_l647_647708

-- Define the six-digit number
def num (B : ℕ) := 303100 + B

-- State the theorem
theorem digit_B_prime_only_1 (B : ℕ) (h₁ : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (h₂ : Prime (num B)) : B = 1 :=
sorry

end digit_B_prime_only_1_l647_647708


namespace integral_f1_magnitude_z1_l647_647361

-- Definitions for first problem
def f1 (x : ℝ) : ℝ :=
  if x < 0 then x + 1 else real.sqrt (1 - x^2)

-- Definitions for second problem
structure ComplexNumber where
  re : ℝ
  im : ℝ

def z1 (a : ℝ) : ComplexNumber :=
  ⟨a, 2⟩

def z2 : ComplexNumber :=
  ⟨3, -4⟩

def isPurelyImaginary (z : ComplexNumber) : Prop :=
  z.re = 0

-- Theorem statements
theorem integral_f1 : ∫ (x : ℝ) in -1..1, f1 x = 1/2 + real.pi/4 :=
by sorry

theorem magnitude_z1 (a : ℝ) (h : isPurelyImaginary (z1 a / z2)) : complex.abs (complex.mk a 2) = 10/3 :=
by sorry

end integral_f1_magnitude_z1_l647_647361


namespace find_point_incenter_l647_647450

open Real

-- Define the incenter condition for a point M inside a triangle ABC
def isIncenter (A B C M : Point) : Prop :=
  (dist M (line.through A B) = dist M (line.through B C)) ∧ (dist M (line.through B C) = dist M (line.through C A))

-- Define the problem statement
theorem find_point_incenter (A B C : Point) :
  ∃ M : Point, isIncenter A B C M :=
by
  sorry

end find_point_incenter_l647_647450


namespace number_of_even_divisors_factorial_9_l647_647565

-- Definitions for conditions
def factorial_9 : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

def prime_factorization_factorial_9 (n : ℕ) : Prop :=
  n = 2^7 * 3^4 * 5 * 7

-- Main theorem statement
theorem number_of_even_divisors_factorial_9 : ∃ d, d = 140 ∧ ∀ r : ℕ,
  even r ∧ r ∣ factorial_9 → (r = prime_factorization_factorial_9 r) :=
begin
  sorry,
end

end number_of_even_divisors_factorial_9_l647_647565


namespace tax_amount_is_correct_l647_647609

def camera_cost : ℝ := 200.00
def tax_rate : ℝ := 0.15

theorem tax_amount_is_correct :
  (camera_cost * tax_rate) = 30.00 :=
sorry

end tax_amount_is_correct_l647_647609


namespace james_units_per_semester_l647_647210

theorem james_units_per_semester
  (cost_per_unit : ℕ)
  (total_cost : ℕ)
  (num_semesters : ℕ)
  (payment_per_semester : ℕ)
  (units_per_semester : ℕ)
  (H1 : cost_per_unit = 50)
  (H2 : total_cost = 2000)
  (H3 : num_semesters = 2)
  (H4 : payment_per_semester = total_cost / num_semesters)
  (H5 : units_per_semester = payment_per_semester / cost_per_unit) :
  units_per_semester = 20 :=
sorry

end james_units_per_semester_l647_647210


namespace chalk_packages_sold_l647_647008

noncomputable def initial_chalk_ratios := (2, 3, 6)
noncomputable def final_chalk_ratios := (5, 7, 4)

def sold_superior_chalk (initial_multiplier : ℝ) (added_regular : ℝ) (added_unusual : ℝ) : Prop :=
  let initial_regular := 2 * initial_multiplier in
  let initial_unusual := 3 * initial_multiplier in
  let initial_superior := 6 * initial_multiplier in
  let total_added := added_regular + added_unusual in
  let final_regular := initial_regular + added_regular in
  let final_unusual := initial_unusual + added_unusual in
  let final_superior := 6 * initial_multiplier * 0.6 in
  let ratio_k := final_superior / 4 in
  total_added <= 100 ∧ 
  final_regular / 5 = ratio_k ∧ 
  final_unusual / 7 = ratio_k ∧ 
  final_superior = 3.6 * initial_multiplier *
  6 * initial_multiplier * 0.4 = 24

theorem chalk_packages_sold : ∃ initial_multiplier added_regular added_unusual,
  sold_superior_chalk initial_multiplier added_regular added_unusual :=
by
  sorry

end chalk_packages_sold_l647_647008


namespace minimum_N_more_profitable_credit_card_l647_647690

-- Given conditions and definitions
def purchase_value : ℝ := 20000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.01
def annual_interest_rate : ℝ := 0.06
def days_in_year : ℝ := 360
def benefit_credit_card (N : ℝ) : ℝ := (annual_interest_rate * purchase_value * N / days_in_year) + (credit_card_cashback_rate * purchase_value)
def benefit_debit_card : ℝ := debit_card_cashback_rate * purchase_value

-- Assertion of the problem
theorem minimum_N_more_profitable_credit_card : ∃ N : ℕ, N > 30 :=
by
  sorry

end minimum_N_more_profitable_credit_card_l647_647690


namespace hyperbola_eccentricity_l647_647296

def is_hyperbola (a b : ℝ) : Prop :=
  b = 2 * a

def eccentricity (a b : ℝ) (h : is_hyperbola a b) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity (a : ℝ) (ha : a > 0) (b : ℝ) (hb : is_hyperbola a b) :
  eccentricity a b hb = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l647_647296


namespace cannot_obtain_2000_l647_647664

-- Define the initial conditions
def initial_numbers : Set ℕ := {12, 17}

-- Define the operation allowed
def add_operation (s : Set ℕ) : Set ℕ :=
  s ∪ {x + y | x ∈ s, y ∈ s}

-- Define the inductive set closure of applying the operation
inductive reachable : Set ℕ → ℕ → Prop
| initial {n} : n ∈ initial_numbers → reachable initial_numbers n
| step {s n m} : reachable s n → reachable s m → reachable (add_operation s) (n + m)

-- State the problem clearly and succinctly
theorem cannot_obtain_2000 : ¬ reachable initial_numbers 2000 := 
  sorry

end cannot_obtain_2000_l647_647664


namespace aaron_more_than_zeb_l647_647932

variable (students : Finset ℕ)
variable (apples_eaten : ℕ → ℕ)
variable (Aaron Zeb : ℕ)
variable (total_apples : ℕ)
variable [nonempty students]

-- Hypotheses from the conditions
hypothesis (h_students : students.card = 8)
hypothesis (h_Aaron_max : ∀ s ∈ students, apples_eaten s ≤ apples_eaten Aaron)
hypothesis (h_Zeb_min : ∀ s ∈ students, apples_eaten Zeb ≤ apples_eaten s)
hypothesis (h_total_apples_gt_20 : total_apples = students.sum apples_eaten ∧ total_apples > 20)
hypothesis (h_Aaron_eaten : apples_eaten Aaron = 8)
hypothesis (h_Zeb_eaten : apples_eaten Zeb = 2)

-- Goal to prove
theorem aaron_more_than_zeb :
  apples_eaten Aaron - apples_eaten Zeb = 6 :=
sorry

end aaron_more_than_zeb_l647_647932


namespace oakdale_earnings_correct_l647_647085

-- Define the number of students and days worked per school
def students_maplewood := 5
def days_maplewood := 6
def students_oakdale := 6
def days_oakdale := 4
def students_pinecrest := 8
def days_pinecrest := 7

-- Define the total payment (without additional fees)
def total_payment := 1240

-- Given conditions
def student_days_maplewood := students_maplewood * days_maplewood
def student_days_oakdale := students_oakdale * days_oakdale
def student_days_pinecrest := students_pinecrest * days_pinecrest
def total_student_days := student_days_maplewood + student_days_oakdale + student_days_pinecrest

-- Calculate the daily wage per student
def daily_wage_per_student := total_payment / total_student_days

-- Calculate the total earning for Oakdale school students
def earnings_oakdale := daily_wage_per_student * student_days_oakdale

-- Proof statement
theorem oakdale_earnings_correct : earnings_oakdale = 270.55 :=
by
  unfold students_maplewood days_maplewood students_oakdale days_oakdale students_pinecrest days_pinecrest
  unfold student_days_maplewood student_days_oakdale student_days_pinecrest total_student_days total_payment
  unfold daily_wage_per_student earnings_oakdale
  norm_num
  sorry

end oakdale_earnings_correct_l647_647085


namespace fraction_of_a_is_correct_l647_647728

-- Introducing the variables a and b
variables (a b : ℚ)

-- Defining the given conditions
def a_val := (1 : ℚ) / 2 -- a = 1/2
def b_val := (1 : ℚ) / 7 -- b = 1/7

-- Prove the main theorem
theorem fraction_of_a_is_correct : b / a = 2 / 7 :=
by
  -- Use the conditions defined above
  have ha : a = a_val := rfl
  have hb : b = b_val := rfl
  rw [ha, hb]
  -- Compute the fraction
  calc
    (1 / 7) / (1 / 2) = (1 / 7) * 2 := by field_simp
                      ...           = 2 / 7 := by norm_num

end fraction_of_a_is_correct_l647_647728


namespace trader_profit_l647_647786

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def purchase_price (P : ℝ) : ℝ := 0.8 * P
noncomputable def depreciation1 (P : ℝ) : ℝ := 0.04 * P
noncomputable def depreciation2 (P : ℝ) : ℝ := 0.038 * P
noncomputable def value_after_depreciation (P : ℝ) : ℝ := 0.722 * P
noncomputable def taxes (P : ℝ) : ℝ := 0.024 * P
noncomputable def insurance (P : ℝ) : ℝ := 0.032 * P
noncomputable def maintenance (P : ℝ) : ℝ := 0.01 * P
noncomputable def total_cost (P : ℝ) : ℝ := value_after_depreciation P + taxes P + insurance P + maintenance P
noncomputable def selling_price (P : ℝ) : ℝ := 1.70 * total_cost P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def profit_percent (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) : profit_percent P = 33.96 :=
  by
    sorry

end trader_profit_l647_647786


namespace possible_integer_root_counts_l647_647048

def is_integer_root (p : ℤ[X]) (x : ℤ) : Prop := p.eval x = 0

noncomputable def integer_root_count (p : ℤ[X]) : ℕ :=
  (Finset.univ.filter (λ x, is_integer_root p x)).card

theorem possible_integer_root_counts (b c d e f : ℤ) :
  let p := X^5 + C b * X^4 + C c * X^3 + C d * X^2 + C e * X + C f
  in integer_root_count p ∈ {0, 1, 2, 4, 5} :=
sorry

end possible_integer_root_counts_l647_647048


namespace range_of_my_trig_function_l647_647828

noncomputable def my_trig_function (x : ℝ) : ℝ :=
  (if (sin x) > 0 then 1 else if (sin x) < 0 then -1 else 0) +
  (if (cos x) > 0 then 1 else if (cos x) < 0 then -1 else 0) +
  (if (tan x) > 0 then 1 else if (tan x) < 0 then -1 else 0)

theorem range_of_my_trig_function : 
  ∀ y : ℝ, y = my_trig_function x → y ∈ {-1, 3} :=
begin
  sorry
end

end range_of_my_trig_function_l647_647828


namespace infinite_triang_pairs_l647_647265

def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

theorem infinite_triang_pairs :
  ∃ (a b : ℤ), (∀ t : ℕ, t > 0 → (is_triangular (a * t + b) ↔ is_triangular t)) ∧ (a > 0) :=
begin
  -- we can construct a specific pair (p^2, (p^2 - 1) / 8) for odd primes p
  let p := 3,
  use (p^2 : ℤ),
  use ((p^2 - 1) / 8 : ℤ),
  split,
  {
    intros t ht,
    sorry -- Here lies the proof, which we omit as directed.
  },
  {
    linarith,
  }
end

end infinite_triang_pairs_l647_647265


namespace find_n_mod_10_l647_647473

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l647_647473


namespace ellipse_equation_correct_line_equation_correct_l647_647117

/- Given conditions: -/
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * b * b = a * a ∧ a * a = b * b + 1

def is_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

noncomputable def ellipse_eqn : Prop :=
  ∃ (a b : ℝ), is_ellipse a b ∧ (∀ x y : ℝ, (x/a)^2 + (y/b)^2 = 1)

noncomputable def line_through_F_perpendicular (l : ℝ → ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l (x₁ - 1) = y₁ → l (x₂ - 1) = y₂ → (x₁ * x₂ + y₁ * y₂ = 0)

noncomputable def line_eqn (F : ℝ × ℝ → (ℝ → ℝ)) : Prop :=
  ∃ k : ℝ, k = sqrt 2 ∨ k = -sqrt 2 ∧ F = λ x, k * (x - 1)

/- Proof goals: -/
theorem ellipse_equation_correct :
  ellipse_eqn ↔ ∀ x y : ℝ, (x^2)/2 + y^2 = 1 :=
sorry

theorem line_equation_correct :
  ∀ l, line_through_F_perpendicular l ↔ ∀ F, F = (λ x, sqrt 2 * (x - 1) ∨ F = (λ x, -sqrt 2 * (x - 1)) :=
sorry

end ellipse_equation_correct_line_equation_correct_l647_647117


namespace sum_converges_l647_647992

/-- Define the conditions for sequence a_n being quasi-injective -/
def quasiInjective (a : ℕ → ℕ) (C : ℕ) : Prop :=
  ∀ k : ℕ, (finset.univ.filter (λ n, a n = k)).card < C

/-- Define the condition for a sequence having only a finite number of prime divisors -/
def finitePrimeDivisors (a : ℕ → ℕ) (S : finset ℕ) : Prop :=
  ∀ n : ℕ, ∀ p : ℕ, p.prime → p ∣ a n → p ∈ S

/-- Formulate the main theorem to be proved -/
theorem sum_converges (a : ℕ → ℕ) (C : ℕ) (S : finset ℕ) 
  (h1 : quasiInjective a C) 
  (h2 : finitePrimeDivisors a S) : 
  summable (λ n, 1 / (a n)) :=
sorry

end sum_converges_l647_647992


namespace find_n_l647_647490

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l647_647490


namespace length_of_QR_is_eight_l647_647421

open Real

noncomputable def length_of_QR (b : ℝ) : ℝ :=
  2 * b

theorem length_of_QR_is_eight (b : ℝ) :
  (b : ℝ) * b * b = 64 → length_of_QR b = 8 :=
by
  intro h
  have h₁ : b^3 = 64 := h 
  sorry

end length_of_QR_is_eight_l647_647421


namespace fraction_integer_l647_647239

theorem fraction_integer (x y : ℤ) (h₁ : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) : ∃ m : ℤ, 4 * x - 3 * y = 5 * m :=
by
  sorry

end fraction_integer_l647_647239


namespace painter_remaining_time_l647_647358

theorem painter_remaining_time (total_rooms: ℕ) (time_per_room: ℕ) (rooms_painted: ℕ) :
  total_rooms = 9 → time_per_room = 8 → rooms_painted = 5 →
  (total_rooms - rooms_painted) * time_per_room = 32 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end painter_remaining_time_l647_647358


namespace lim_geom_sum_l647_647137

theorem lim_geom_sum (a : ℝ) (h₀ : a > 0) 
  (h₁ : (∃ b : ℝ, b = 6 * a^2 ∧ b = 3/2) ) 
  : ( ∃ L : ℝ, (L = ∑ i in range n, a^i) ∧ n → ∞ ∧ L = 1 ) := 
sorry

end lim_geom_sum_l647_647137


namespace mystical_mountain_creatures_l647_647194

-- Definitions for conditions
def nineHeadedBirdHeads : Nat := 9
def nineHeadedBirdTails : Nat := 1
def nineTailedFoxHeads : Nat := 1
def nineTailedFoxTails : Nat := 9

-- Prove the number of Nine-Tailed Foxes
theorem mystical_mountain_creatures (x y : Nat)
  (h1 : 9 * x + (y - 1) = 36 * (y - 1) + 4 * x)
  (h2 : 9 * (x - 1) + y = 3 * (9 * y + (x - 1))) :
  x = 14 :=
by
  sorry

end mystical_mountain_creatures_l647_647194


namespace cos_squared_half_angle_plus_quarter_l647_647535

noncomputable def cos_of_angle_in_second_quadrant (α : ℝ) (h : α ∈ set.Icc (π / 2) π) : ℝ :=
- (2 * real.sqrt 2 / 3)

noncomputable def sin_of_angle_in_second_quadrant (α: ℝ) (h: α ∈ set.Icc (π / 2) π): ℝ :=
real.sqrt (1 - (cos_of_angle_in_second_quadrant α h) ^ 2)

theorem cos_squared_half_angle_plus_quarter (α : ℝ) (h : α ∈ set.Icc (π / 2) π)
  (hα : cos_of_angle_in_second_quadrant α h = - (2 * real.sqrt 2 / 3)) :
  (real.cos ((α / 2) + (π / 4)))^2 = 1 / 3 :=
by
  sorry

end cos_squared_half_angle_plus_quarter_l647_647535


namespace fraction_of_selected_color_l647_647762

theorem fraction_of_selected_color (x y : ℕ) (hx : x > 0) :
  let bw_films := 20 * x
  let color_films := 8 * y
  let selected_bw_films := (y / x) * bw_films / 100
  let selected_color_films := color_films
  let total_selected_films := selected_bw_films + selected_color_films
  (selected_color_films / total_selected_films) = 40 / 41 :=
by
  have h_bw_selected : selected_bw_films = y / 5, by sorry
  have h_fractions : (selected_color_films : ℚ) / ((selected_bw_films + selected_color_films) : ℚ) = 40 / 41, by sorry
  exact h_fractions

end fraction_of_selected_color_l647_647762


namespace artwork_calculation_l647_647999

theorem artwork_calculation :
  ∃ (x : ℕ), 10 = 2 * 5 ∧ (5 * 3 + 5 * x = 35) ∧ x = 4 :=
by
  -- existance of a value of x that satisfies the given conditions.
  use 4
  -- 10 total students: 2 groups of 5
  split
  exact rfl
  -- total artworks is 35, divided into artworks by first half and artworks by an unknown x for the second half.
  split
  linarith
  -- the solution given is x = 4
  exact rfl

end artwork_calculation_l647_647999


namespace sequence_formula_l647_647114

noncomputable def S : ℕ → ℕ
| n => n^2 + 2*n + 1

noncomputable def a : ℕ → ℕ
| 1 => 4
| n+1 => 2 * (n + 1) + 1

theorem sequence_formula (n : ℕ) : 
  n = 0 ∨ (a n = S n - S (n-1)) := by
  sorry

end sequence_formula_l647_647114


namespace two_people_same_birthday_l647_647266

noncomputable def population : ℕ := 6000000000

noncomputable def max_age_seconds : ℕ := 150 * 366 * 24 * 60 * 60

theorem two_people_same_birthday :
  ∃ (a b : ℕ) (ha : a < population) (hb : b < population) (hab : a ≠ b),
  (∃ (t : ℕ) (ht_a : t < max_age_seconds) (ht_b : t < max_age_seconds), true) :=
by
  sorry

end two_people_same_birthday_l647_647266


namespace marie_loaves_per_day_l647_647646

theorem marie_loaves_per_day (cash_register_cost : ℝ := 1040)
                             (loaf_price : ℝ := 2)
                             (cakes_per_day : ℕ := 6)
                             (cake_price_per_unit : ℝ := 12)
                             (daily_rent : ℝ := 20)
                             (daily_electricity : ℝ := 2)
                             (days_to_pay : ℕ := 8) :
                             (L : ℝ) :=
  let daily_cake_revenue := cakes_per_day * cake_price_per_unit
  let daily_expenses := daily_rent + daily_electricity
  let daily_revenue_from_loaves := L * loaf_price
  let total_daily_profit := (daily_revenue_from_loaves + daily_cake_revenue) - daily_expenses
  let required_profit := cash_register_cost
  (days_to_pay * total_daily_profit = required_profit) → L = 40 := by
  sorry

end marie_loaves_per_day_l647_647646


namespace M_is_incenter_l647_647449

variable (A B C M : Type) [triangle A B C] 

-- Define the problem conditions
structure Circle (P Q : Type) :=
(center : Type)
(radius : ℝ)

noncomputable def circle_on_segment (M A : Type) : Circle M A := sorry

-- Specify the property of M we need to prove
def isIncenter (M : Type) (A B C : Type) [inhabited M] : Prop := 
  ∀ P Q R : Type,
  (circle_on_segment M A).common_chord_length = (circle_on_segment M B).common_chord_length ∧
  (circle_on_segment M B).common_chord_length = (circle_on_segment M C).common_chord_length

-- The theorem statement
theorem M_is_incenter 
  (A B C M : Type) [triangle A B C] [inhabited M] :
  (isIncenter M A B C) → Incenter M A B C :=
sorry

end M_is_incenter_l647_647449


namespace length_of_second_train_is_200_l647_647724

noncomputable def length_of_second_train : ℕ :=
  let speed1_kmh := 60
  let speed2_kmh := 40
  let speed1_ms := (60 * 1000) / 3600
  let speed2_ms := (40 * 1000) / 3600
  let relative_speed_ms := speed1_ms + speed2_ms
  let time_s := 17.998560115190788
  let distance := relative_speed_ms * time_s
  300 + (distance - 300)

/-- Given:
  - Speed of the first train = 60 km/hr
  - Speed of the second train = 40 km/hr
  - Length of the first train = 300 m
  - They take 17.998560115190788 seconds to cross each other.

  Prove that the length of the second train is 200 meters. -/
theorem length_of_second_train_is_200 :
  length_of_second_train = 200 := sorry

end length_of_second_train_is_200_l647_647724


namespace price_of_each_potato_is_1_l647_647641

def celery_price : ℤ := 5
def cereal_original_price : ℤ := 12
def cereal_discount_factor : ℝ := 0.50
def cereal_price : ℤ := cereal_original_price - (cereal_discount_factor * cereal_original_price).toInt
def bread_price : ℤ := 8
def milk_original_price : ℤ := 10
def milk_discount_factor : ℝ := 0.10
def milk_price : ℤ := milk_original_price - (milk_discount_factor * milk_original_price).toInt
def potatoes_count : ℤ := 6
def initial_money : ℤ := 60
def money_left : ℤ := 26
def total_cost_so_far : ℤ := celery_price + cereal_price + bread_price + milk_price
def money_spent_on_potatoes_and_coffee : ℤ := initial_money - money_left
def money_spent_on_potatoes : ℤ := money_spent_on_potatoes_and_coffee - total_cost_so_far
def price_per_potato : ℤ := money_spent_on_potatoes / potatoes_count

theorem price_of_each_potato_is_1 : price_per_potato = 1 := by
  sorry

end price_of_each_potato_is_1_l647_647641


namespace total_tiles_in_room_l647_647389

theorem total_tiles_in_room (s : ℕ) (hs : 6 * s - 5 = 193) : s^2 = 1089 :=
by sorry

end total_tiles_in_room_l647_647389


namespace find_incenter_l647_647463

-- Define the arbitrary triangle
variable {α : Type} [EuclideanGeometry α]
variables (A B C M : α)

-- Define the main hypothesis and theorem
theorem find_incenter (inside_triangle : M ∈ triangle ABC) (equal_chords : ∀ D ∈ {A, B, C}, length (common_chord M D) = some_constant) :
  M = incenter ABC :=
sorry

end find_incenter_l647_647463


namespace angle_A_eq_pi_div_3_l647_647880

theorem angle_A_eq_pi_div_3 
  (A B C : Type) 
  [triangle A B C]
  (area_ABC : ℝ) (b c : ℝ) (A : ℝ)
  (h1 : area_ABC = 3 / 2)
  (h2 : b = 2)
  (h3 : c = sqrt 3) :
  A = π / 3 := 
by 
  sorry

end angle_A_eq_pi_div_3_l647_647880


namespace not_divisible_by_n_l647_647261

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬ (n ∣ (2^n - 1)) :=
by
  sorry

end not_divisible_by_n_l647_647261


namespace incenter_property_of_chord_lengths_equal_l647_647458

theorem incenter_property_of_chord_lengths_equal
  {A B C M P Q R : Point}
  (h_triangle : is_triangle A B C)
  (h_in_triangle : is_in_triangle M A B C)
  (h_circles : circles_on_diameters A B C M)
  (h_common_chords_equal : common_chords_equal P Q R A B C M) :
  is_incenter M A B C :=
sorry

end incenter_property_of_chord_lengths_equal_l647_647458


namespace temperature_lower_than_freezing_point_is_minus_three_l647_647726

-- Define the freezing point of water
def freezing_point := 0 -- in degrees Celsius

-- Define the temperature lower by a certain value
def lower_temperature (t: Int) (delta: Int) := t - delta

-- State the theorem to be proved
theorem temperature_lower_than_freezing_point_is_minus_three:
  lower_temperature freezing_point 3 = -3 := by
  sorry

end temperature_lower_than_freezing_point_is_minus_three_l647_647726


namespace balls_picked_is_two_l647_647756

theorem balls_picked_is_two (T : ℕ) (n : ℕ) (hT : T = 7 + 5 + 4) 
  (h_prob : (7.choose 2 : ℚ) / (T.choose n) = 0.175) : n = 2 := by
  sorry

end balls_picked_is_two_l647_647756


namespace bankers_discount_correct_l647_647418

noncomputable def bankers_discount (bill_amount_usd : ℝ) 
                                   (true_discount_eur : ℝ) 
                                   (annual_interest_rate : ℝ) 
                                   (mature_days : ℕ) 
                                   (initial_exchange_rate : ℝ) 
                                   (mature_exchange_rate : ℝ) 
  : ℝ :=
  let true_discount_usd := true_discount_eur * initial_exchange_rate in
  let face_value := bill_amount_usd - true_discount_usd in
  let time_in_years := (mature_days : ℝ) / 365 in
  let bankers_discount := face_value * (annual_interest_rate * time_in_years) in
  let adjustment_factor := mature_exchange_rate / initial_exchange_rate in
  bankers_discount * adjustment_factor

theorem bankers_discount_correct : 
  bankers_discount 12000 1590 0.0725 215 1.10 1.15 = 457.52 :=
  by
  -- Proof omitted
  sorry

end bankers_discount_correct_l647_647418


namespace cone_lateral_surface_area_l647_647542

noncomputable def lateral_surface_area (l h r : ℝ) : ℝ := 
  π * r * l

theorem cone_lateral_surface_area (l h : ℝ) (hl : l = 13) (hh : h = 12) : 
  lateral_surface_area l h (Real.sqrt (l^2 - h^2)) = 65 * π :=
by
  rw [hl, hh]
  have r_eq : Real.sqrt (13^2 - 12^2) = 5 := by
    calc
      Real.sqrt (13^2 - 12^2) = Real.sqrt (169 - 144) : by simp [sq]
                          ... = Real.sqrt 25 : by norm_num
                          ... = 5 : by norm_num
  rw [r_eq]
  simp [lateral_surface_area]
  norm_num

end cone_lateral_surface_area_l647_647542


namespace octahedron_volume_ratio_and_sum_l647_647588

theorem octahedron_volume_ratio_and_sum (V : ℝ) (regular_octahedron : Type) [octahedron_has_volume V regular_octahedron] :
  let smaller_octahedron := centers_of_faces regular_octahedron in
  volume smaller_octahedron / volume regular_octahedron = 1 / 64 ∧ (1 + 64 = 65) :=
by
  sorry

end octahedron_volume_ratio_and_sum_l647_647588


namespace paint_problem_l647_647665

-- Definitions based on conditions
def roomsInitiallyPaintable := 50
def roomsAfterLoss := 40
def cansLost := 5

-- The number of rooms each can could paint
def roomsPerCan := (roomsInitiallyPaintable - roomsAfterLoss) / cansLost

-- The total number of cans originally owned
def originalCans := roomsInitiallyPaintable / roomsPerCan

-- Theorem to prove the number of original cans equals 25
theorem paint_problem : originalCans = 25 := by
  sorry

end paint_problem_l647_647665


namespace sum_floor_absolute_value_eq_501_l647_647686

theorem sum_floor_absolute_value_eq_501
    (x : ℕ → ℤ)
    (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → x n + n = (∑ i in Finset.range 1000, x i) + 1001) :
    (⌊|(∑ n in Finset.range 1000, x n)|⌋ = 501) :=
by
  sorry

end sum_floor_absolute_value_eq_501_l647_647686


namespace range_of_s_l647_647506

def is_composite (n : ℕ) : Prop := (n > 1) ∧ ∃ p k : ℕ, p.prime ∧ k > 1 ∧ n = p^k

def s (n : ℕ) : ℕ :=
  if hn : 2 ≤ n ∧ ¬ n.prime then
    let prime_factors : List (ℕ × ℕ) :=
      (UniqueFactorizationMonoid.normalizedFactors n).groupBy id in
    prime_factors.foldl (λ acc (p, k), acc + k * p^2) 0
  else
    0

theorem range_of_s :
  ∀ n : ℕ, is_composite n → 12 ≤ s n ∧ ∀ m : ℕ, m > 11 → ∃ n, is_composite n ∧ s n = m :=
by
  sorry

end range_of_s_l647_647506


namespace maximize_S_minimize_S_l647_647434

-- Definitions and conditions
variables {x1 x2 x3 x4 x5 : ℕ}

-- Sum condition
def sum_condition := (x1 + x2 + x3 + x4 + x5 = 2006)

-- Positive integer condition
def positive_condition := (x1 > 0) ∧ (x2 > 0) ∧ (x3 > 0) ∧ (x4 > 0) ∧ (x5 > 0)

-- S definition
def S := x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 + x3 * x4 + x3 * x5 + x4 * x5

-- Difference condition for minimization part
def diff_condition := (∀ i j, 1 ≤ i → i ≤ 5 → 1 ≤ j → j ≤ 5 → |(x1 :: x2 :: x3 :: x4 :: x5 :: list.nil).nth (i - 1) - (x1 :: x2 :: x3 :: x4 :: x5 :: list.nil).nth (j - 1)| ≤ 2)

-- Prove maximum S
theorem maximize_S : sum_condition ∧ positive_condition → (x1, x2, x3, x4, x5) = (402, 401, 401, 401, 401) → S = (402 * 401 + 402 * 401 + 402 * 401 + 402 * 401 + 401 * 401 + 401 * 401 + 401 * 401 + 401 * 401 + 401 * 401) := 
by sorry

-- Prove minimum S under additional condition
theorem minimize_S : sum_condition ∧ positive_condition ∧ diff_condition → (x1, x2, x3, x4, x5) = (402, 402, 402, 400, 400) → S = (402 * 402 + 402 * 402 + 402 * 400 + 402 * 400 + 402 * 402 + 402 * 402 + 402 * 400 + 402 * 400 + 400 * 400 + 400 * 400) := 
by sorry

end maximize_S_minimize_S_l647_647434


namespace recur_P_recur_p_pi_bounds_l647_647419

noncomputable def circle_radius (r : ℝ) := r = 1

namespace Geometry

  open Real

  -- Definitions for perimeters
  def circumscribed_perimeter (n : ℕ) (P_n : ℝ) : Prop
  def inscribed_perimeter (n : ℕ) (p_n : ℝ) : Prop

  -- Given values for perimiters of polygons
  def P_4 := 8
  def p_4 := 4 * sqrt 2
  def P_6 := 4 * sqrt 3
  def p_6 := 6

  -- Recurrence relations
  theorem recur_P (n : ℕ) (P_n p_n : ℝ) (h1 : circumscribed_perimeter n P_n) (h2 : inscribed_perimeter n p_n) (h3 : 3 ≤ n) : 
    circumscribed_perimeter (2 * n) (2 * P_n * p_n / (P_n + p_n)) := sorry

  theorem recur_p (n : ℕ) (P_n p_n : ℝ) (h1 : circumscribed_perimeter n P_n) (h2 : inscribed_perimeter n p_n) (h3 : 3 ≤ n) :
    inscribed_perimeter (2 * n) (sqrt (p_n * (2 * P_n * p_n / (P_n + p_n)))) := sorry

  -- Inequalities involving pi
  theorem pi_bounds : (3^10 / 71 < Real.pi) ∧ (Real.pi < 22 / 7) := sorry

end Geometry

end recur_P_recur_p_pi_bounds_l647_647419


namespace no_continuous_function_l647_647747

open Real

variable {α : ℝ}

theorem no_continuous_function (f : ℝ → ℝ) (hf1 : continuous_on f (set.Icc 0 1))
  (hf2 : ∀ x, f x > 0)
  (h1 : ∫ x in 0..1, f x = 1)
  (h2 : ∫ x in 0..1, x * f x = α)
  (h3 : ∫ x in 0..1, x^2 * f x = α^2) : false :=
sorry

end no_continuous_function_l647_647747


namespace sine_even_function_l647_647893

theorem sine_even_function (f : ℝ → ℝ) (a varphi : ℝ) (h : ∀ (x : ℝ), f x ≤ f a) :
  ∀ x, f (x + a) = f (-x - a) :=
begin
  intros x,
  -- Here we show that f is equivalent to the sine function as given in the problem
  have hf : f = λ x, Real.sin (2 * x + varphi) := sorry,
  -- Show that f (x + a) = f (-x - a)
  rw hf,
  sorry
end

end sine_even_function_l647_647893


namespace units_digit_expansion_l647_647502

noncomputable def A : ℝ := 17 + Real.sqrt 198
noncomputable def B : ℝ := 17 - Real.sqrt 198

theorem units_digit_expansion : 
  (Int.unitsDigit (Int.ofReal ((17 + Real.sqrt 198)^21 + (17 - Real.sqrt 198)^21))) = 4 :=
by 
  sorry

end units_digit_expansion_l647_647502


namespace point_above_x_axis_l647_647189

theorem point_above_x_axis (a : ℝ) : a > 0 → a = √3 :=
by
  sorry

end point_above_x_axis_l647_647189


namespace interest_rate_correct_l647_647662

noncomputable def rate_of_interest (P T SI CI : ℕ) :=
  SI = P * T * (R / 100) ∧ CI = P * ((1 + (R / 100)) ^ 2 - 1) → R = 104

theorem interest_rate_correct (P: ℝ) (R: ℝ) (SI: ℝ) (CI: ℝ):
  SI = P * R * 2 / 100 ∧ CI = P * ((1 + R / 100) ^ 2 - 1) → R = 104 := 
by
  sorry

end interest_rate_correct_l647_647662


namespace horizontal_asymptote_of_rational_function_l647_647919

theorem horizontal_asymptote_of_rational_function :
  (∃ y, y = (10 * x ^ 4 + 3 * x ^ 3 + 7 * x ^ 2 + 6 * x + 4) / (2 * x ^ 4 + 5 * x ^ 3 + 4 * x ^ 2 + 2 * x + 1) → y = 5) := sorry

end horizontal_asymptote_of_rational_function_l647_647919


namespace parallel_vectors_l647_647885

-- Define vectors m and n
def vector_m := (2, 8)
def vector_n (t : ℤ) := (-4, t)

-- Define condition for parallel vectors (cross product should be zero)
def parallel (u v : ℤ × ℤ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

-- The main theorem
theorem parallel_vectors (t : ℤ) (h : parallel vector_m (vector_n t)) : t = -16 :=
by sorry

end parallel_vectors_l647_647885


namespace num_ways_to_write_360_as_increasing_seq_l647_647594

def is_consecutive_sum (n k : ℕ) : Prop :=
  let seq_sum := k * n + k * (k - 1) / 2
  seq_sum = 360

def valid_k (k : ℕ) : Prop :=
  k ≥ 2 ∧ k ∣ 360 ∧ (k = 2 ∨ (k - 1) % 2 = 0)

noncomputable def count_consecutive_sums : ℕ :=
  Nat.card {k // valid_k k ∧ ∃ n : ℕ, is_consecutive_sum n k}

theorem num_ways_to_write_360_as_increasing_seq : count_consecutive_sums = 4 :=
sorry

end num_ways_to_write_360_as_increasing_seq_l647_647594


namespace rational_numbers_cubic_sum_l647_647122

theorem rational_numbers_cubic_sum
  (a b c : ℚ)
  (h1 : a - b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3) :
  a^3 + b^3 + c^3 = 1 :=
by
  sorry

end rational_numbers_cubic_sum_l647_647122


namespace geometric_sequence_l647_647640

open Nat

-- Define the sequence and conditions for the problem
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {m p : ℕ}
variable (h1 : a 1 ≠ 0)
variable (h2 : ∀ n : ℕ, 2 * S (n + 1) - 3 * S n = 2 * a 1)
variable (h3 : S 0 = 0)
variable (h4 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
variable (h5 : a 1 ≥ m^(p-1))
variable (h6 : a p ≤ (m+1)^(p-1))

-- The theorem that we need to prove
theorem geometric_sequence (n : ℕ) : 
  (exists r : ℕ → ℕ, ∀ k : ℕ, a (k + 1) = r (k + 1) * a k) ∧ 
  (∀ k : ℕ, a k = sorry) := sorry

end geometric_sequence_l647_647640


namespace distance_focus_to_directrix_l647_647294

-- Definition: A parabola with equation y^2 = 4x
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

-- Theorem: The distance from the focus to the directrix of the parabola y^2 = 4x is 2
theorem distance_focus_to_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → distance_focus_to_directrix_of y^2 = 4x = 2 := 
sorry

end distance_focus_to_directrix_l647_647294


namespace problem_1_problem_2_problem_3_problem_4_l647_647200

-- Given definitions
def side_lengths (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def semi_perimeter (s : ℝ) (a b c : ℝ) := s = (a + b + c) / 2
def circumradius (R : ℝ) := R > 0
def inradius (r : ℝ) := r > 0
def area (Δ : ℝ) := Δ > 0

-- Proof goals
theorem problem_1 (a b c Δ : ℝ) : side_lengths a b c → area Δ → ab + bc + ca ≥ 4 * sqrt 3 * Δ := 
by sorry

theorem problem_2 (s Δ r R : ℝ) :
  semi_perimeter s a b c → area Δ → inradius r → circumradius R →
  3 * sqrt 3 * r^2 ≤ Δ ∧ Δ ≤ s^2 / (3 * sqrt 3) ∧ s^2 / (3 * sqrt 3) ≤ 3 * sqrt 3 / 4 * R^2 :=
by sorry

theorem problem_3 (s a b c r : ℝ) :
  semi_perimeter s a b c → inradius r →
  1 / (s - a) + 1 / (s - b) + 1 / (s - c) ≥ sqrt 3 / r :=
by sorry

theorem problem_4 (Δ r R : ℝ) : area Δ → inradius r → circumradius R → Δ ≤ 3 * sqrt 3 / 2 * r * R :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l647_647200


namespace truckAverageSpeed_l647_647017

noncomputable def averageSpeedOfTruck (d1 d2 d3 d4 s1 s2 s3 s4 : ℝ) : ℝ :=
  let time1 := d1 / s1
  let time2 := d2 / s2
  let time3 := 1 -- already in hours
  let time4 := 0.5 -- 30 minutes in hours
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := time1 + time2 + time3 + time4
  totalDistance / totalTime

theorem truckAverageSpeed : 
  averageSpeedOfTruck 30 35 65 25 45 55 65 50 ≈ 54.04 := 
by
  sorry

end truckAverageSpeed_l647_647017


namespace white_ball_probability_l647_647930

noncomputable def prob_white_ball : ℚ :=
  let initial_combinations := [{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}] in
  let add_white_ball (c : set ℕ) := c.insert 1 in
  let prob := λ c, ((add_white_ball c).filter (λ x, x = 1)).card.toRat / (add_white_ball c).card.toRat in
  (initial_combinations.map prob).sum / initial_combinations.length

theorem white_ball_probability : prob_white_ball = 5 / 8 :=
  sorry

end white_ball_probability_l647_647930


namespace problem_ellipse_left_right_foci_min_slope_l647_647118

-- Define the conditions
def a : ℝ := 2
def b : ℝ := sqrt 3
def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / a^2 + P.2^2 / b^2 = 1

-- Define the initial setup
theorem problem_ellipse_left_right_foci_min_slope :
  (ellipse (1, 3/2)) ∧ (a^2 - b^2 = 1) →
  (∃a b, ellipse (1, 3/2) ∧ (a^2 - b^2 = 1) ∧
    ∀ (A B : ℝ × ℝ), ellipse A ∧ ellipse B →
                              ∀ (M N : ℝ × ℝ),
                              (M.2 = 0) ∧ (N.1 = 0) ∧
                              ∃ y0, y0 > 0 ∧ 
                                    let k := (-2 * y0 / (3 * y0^2 + 16)) in 
                                    k = (-√3 / 12)) := 
begin
  intros h,
  exact ⟨2, sqrt 3, h, sorry⟩
end

end problem_ellipse_left_right_foci_min_slope_l647_647118


namespace hyperbola_asymptote_slopes_l647_647054

theorem hyperbola_asymptote_slopes:
  (∀ (x y : ℝ), (x^2 / 144 - y^2 / 81 = 1) → (y = (3 / 4) * x ∨ y = -(3 / 4) * x)) :=
by
  sorry

end hyperbola_asymptote_slopes_l647_647054


namespace find_third_term_l647_647927

theorem find_third_term :
  ∃ (a : ℕ → ℝ), a 0 = 5 ∧ a 4 = 2025 ∧ (∀ n, a (n + 1) = a n * r) ∧ a 2 = 225 :=
by
  sorry

end find_third_term_l647_647927


namespace incenter_property_of_chord_lengths_equal_l647_647459

theorem incenter_property_of_chord_lengths_equal
  {A B C M P Q R : Point}
  (h_triangle : is_triangle A B C)
  (h_in_triangle : is_in_triangle M A B C)
  (h_circles : circles_on_diameters A B C M)
  (h_common_chords_equal : common_chords_equal P Q R A B C M) :
  is_incenter M A B C :=
sorry

end incenter_property_of_chord_lengths_equal_l647_647459


namespace square_window_side_length_is_24_l647_647907

noncomputable def side_length_square_window
  (num_panes_per_row : ℕ) (pane_height_ratio : ℝ) (border_width : ℝ) (x : ℝ) : ℝ :=
  num_panes_per_row * x + (num_panes_per_row + 1) * border_width

theorem square_window_side_length_is_24
  (num_panes_per_row : ℕ)
  (pane_height_ratio : ℝ)
  (border_width : ℝ) 
  (pane_width : ℝ)
  (pane_height : ℝ)
  (window_side_length : ℝ) : 
  (num_panes_per_row = 3) →
  (pane_height_ratio = 3) →
  (border_width = 3) →
  (pane_height = pane_height_ratio * pane_width) →
  (window_side_length = side_length_square_window num_panes_per_row pane_height_ratio border_width pane_width) →
  (window_side_length = 24) :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end square_window_side_length_is_24_l647_647907


namespace sugar_per_cube_l647_647581

theorem sugar_per_cube (sugar_cubes : ℕ) (total_sugar : ℝ) (h : sugar_cubes = 7 ∧ total_sugar = 9.8) :
  total_sugar / sugar_cubes = 1.4 := 
by
  obtain ⟨hc, hs⟩ := h,
  rw [hc, hs],
  norm_num,
  sorry

end sugar_per_cube_l647_647581


namespace sum_of_reciprocals_of_divisors_is_two_l647_647519

theorem sum_of_reciprocals_of_divisors_is_two (n : ℕ) (h : ∑ k in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 2 * n) :
  (∑ k in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), 1 / (k : ℚ)) = 2 := 
sorry

end sum_of_reciprocals_of_divisors_is_two_l647_647519


namespace find_n_l647_647492

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l647_647492


namespace perpendicular_lines_slope_l647_647558

theorem perpendicular_lines_slope (a : ℝ)
  (h : (a * (a + 2)) = -1) : a = -1 :=
sorry

end perpendicular_lines_slope_l647_647558


namespace z_h_neg5_l647_647225

def h (x : ℤ) : ℤ := 2 * x ^ 2 - 8
def z (y : ℤ) : ℤ

axiom z_h5 : z (h 5) = 10

theorem z_h_neg5 : z (h (-5)) = 10 :=
by
  sorry

end z_h_neg5_l647_647225


namespace product_of_valid_b_values_l647_647037

open Nat

def has_exactly_b_prime_divisors (b : ℕ) : Prop :=
  b ≥ 2 ∧ (let n := (b + 1) * (b^2 - b + 1) * (b^2 + b + 1)
           in n.prime_factors_uniq.length = b)

theorem product_of_valid_b_values : 
  (∏ b in { b | has_exactly_b_prime_divisors b }.to_finset, b) = 12 :=
by
  sorry

end product_of_valid_b_values_l647_647037


namespace gas_price_increase_second_month_l647_647321

theorem gas_price_increase_second_month :
  ∃ (x : ℝ), (1.30 * (1 + x / 100) * 0.641025641025641 = 1) → x = 20 :=
begin
  sorry  -- Proof to be provided.
end

end gas_price_increase_second_month_l647_647321


namespace boat_speed_in_still_water_l647_647313

theorem boat_speed_in_still_water (V_b V_r : ℝ) : 
  V_b + V_r = 18 → 
  V_b - V_r = 6 → 
  V_b = 12 := by
  assume h1 : V_b + V_r = 18
  assume h2 : V_b - V_r = 6
  sorry

end boat_speed_in_still_water_l647_647313


namespace geometric_sequence_smallest_n_l647_647765

def geom_seq (n : ℕ) (r : ℝ) (b₁ : ℝ) : ℝ := 
  b₁ * r^(n-1)

theorem geometric_sequence_smallest_n 
  (b₁ b₂ b₃ : ℝ) (r : ℝ)
  (h₁ : b₁ = 2)
  (h₂ : b₂ = 6)
  (h₃ : b₃ = 18)
  (h_seq : ∀ n, bₙ = geom_seq n r b₁) :
  ∃ n, n = 5 ∧ geom_seq n r 2 = 324 :=
by
  sorry

end geometric_sequence_smallest_n_l647_647765


namespace quadratic_nonneg_iff_m_in_range_l647_647925

theorem quadratic_nonneg_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 2 * m + 5 ≥ 0) ↔ (-2 : ℝ) ≤ m ∧ m ≤ 10 :=
by sorry

end quadratic_nonneg_iff_m_in_range_l647_647925


namespace quadratic_sum_l647_647928

theorem quadratic_sum (x : ℝ) :
  ∃ a h k : ℝ, (5*x^2 - 10*x - 3 = a*(x - h)^2 + k) ∧ (a + h + k = -2) :=
sorry

end quadratic_sum_l647_647928


namespace find_factor_l647_647381

theorem find_factor :
  ∃ f : ℝ, ∀ n : ℝ, n = 18 → f * (2 * n + 5) = 123 ∧ f = 3 :=
by
  -- Introduce the necessary variables and assumptions
  intro f n h₁,
  -- Use the given condition
  have h₂ : 2 * n + 5 = 41 := by sorry, -- Replace with actual proof calculations
  -- Substitute into the equation and solve for f
  have h₃ : f * 41 = 123 := by sorry, -- Replace with actual proof calculations
  -- Solve for f
  have h₄ : f = 3 := by sorry, -- Replace with actual proof calculations
  
  existsi f,
  -- Show that the conditions are satisfied
  split,
  assumption,
  assumption

end find_factor_l647_647381


namespace total_streets_patrolled_in_one_hour_l647_647242

-- Define the context with noncomputables for the ratios
variables (a x b y c z : ℝ) (h1 : a / x = 9) (h2 : b / y = 11) (h3 : c / z = 7)

-- The theorem stating the sum of streets patrolled by each officer in one hour
theorem total_streets_patrolled_in_one_hour : (a / x) + (b / y) + (c / z) = 27 :=
by {
  rw [h1, h2, h3],
  norm_num,
  exact 27
}

end total_streets_patrolled_in_one_hour_l647_647242


namespace inequality_solution_l647_647466

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2 ↔ (0 < x ∧ x ≤ 0.5) ∨ (6 ≤ x) :=
by { sorry }

end inequality_solution_l647_647466


namespace parallelogram_count_l647_647258

def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (x, y)

def area_formula (b d : ℝ) : ℝ :=
  | 4 * b * d - 2 * b^2 |

theorem parallelogram_count :
  ∃ n : ℕ, n = 500 ∧
  (∀ (A B C D : ℝ × ℝ), A = (0, 0) → is_lattice_point B → is_lattice_point D →
    B.2 = 2 * B.1 → D.2 = 3 * D.1 →
    let C := (B.1 + D.1, 2 * B.1 + 3 * D.1) in
    area_formula B.1 D.1 = 500000) ⟶ n = 500 :=
begin
  sorry
end

end parallelogram_count_l647_647258


namespace range_of_linear_function_within_interval_l647_647224

theorem range_of_linear_function_within_interval (c d : ℝ) (h_c : c < 0) (h_d : d > 0) :
    ∃ (a b : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → (g x = cx + d) ∧ a ≤ g x ∧ g x ≤ b) ∧ a = c + d ∧ b = -c + d := by
  sorry

end range_of_linear_function_within_interval_l647_647224


namespace max_notebooks_l647_647578

theorem max_notebooks (price_per_notebook : ℕ) (total_money : ℕ) (max_notebooks : ℕ) := 
  price_per_notebook = 37 → 
  total_money = 5800 → 
  max_notebooks = 156 → 
  ∀ n : ℕ, (n : ℕ) * price_per_notebook ≤ total_money → n ≤ max_notebooks :=
by
  intros h_price h_money h_max n h_le
  sorry

end max_notebooks_l647_647578


namespace determine_slope_l647_647560

theorem determine_slope (a : ℝ) : 
  let l1 := λ x : ℝ, a * x - 2 in
  let l2 := λ x : ℝ, (a + 2) * x + 1 in
  (∀ x1 x2 : ℝ, x1 ≠ x2 ↔ (l1 x1 - l1 x2) / (x1 - x2) * (l2 x1 - l2 x2) / (x1 - x2) = -1) → 
  a = -1 :=
by
  -- proof or steps are not required, use sorry
  sorry

end determine_slope_l647_647560


namespace some_students_are_not_club_members_l647_647797

variable (U : Type) -- U represents the universe of students and club members
variables (Student ClubMember StudyLate : U → Prop)

-- Conditions derived from the problem
axiom h1 : ∃ s, Student s ∧ ¬ StudyLate s -- Some students do not study late
axiom h2 : ∀ c, ClubMember c → StudyLate c -- All club members study late

theorem some_students_are_not_club_members :
  ∃ s, Student s ∧ ¬ ClubMember s :=
by
  sorry

end some_students_are_not_club_members_l647_647797


namespace parallelogram_area_correct_l647_647032

noncomputable def parallelogram_area (p q : ℝ) (a b : ℝ → ℝ → ℝ) : ℝ :=
  let cross_product := λ u v, u * v - v * u
  let magnitude := λ v, sqrt (v * v)
  magnitude (cross_product a b)

theorem parallelogram_area_correct :
  ∀ (p q : ℝ),
    (5 * p + 2 * q) = 5 * p + 2 * q →
    (p - 3 * q) = p - 3 * q →
    sqrt ((4 * sqrt 2) ^ 2 + 6 ^ 2) = sqrt 32 + 36 →
    cos (pi / 4) = sqrt 2 / 2 →
  parallelogram_area p q (5 * p + 2 * q) (p - 3 * q) = 408 :=
by
  intros
  sorry

end parallelogram_area_correct_l647_647032


namespace sequence_proof_l647_647955

def double_factorial (n : ℕ) : ℕ :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

noncomputable def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then -1
  else if n = 2 then 1 / 2
  else sequence_a (n-2) * (n-1) / n

theorem sequence_proof (n : ℕ) (h : n ≥ 3) :
  sequence_a n = (-1)^n * (double_factorial (n-1) / double_factorial n) :=
sorry

end sequence_proof_l647_647955


namespace collinear_groups_l647_647024

def collinear (a b : ℝ → ℝ) : Prop :=
∃ λ : ℝ, a = λ • b

variables (e e₁ e₂ : ℝ → ℝ)
-- Conditions for group 1
axiom e_nonzero : e ≠ 0
-- Conditions for group 2 and group 3
axioms (e₁_nonzero : e₁ ≠ 0) (e₂_nonzero : e₂ ≠ 0) (e₁_e₂_not_collinear : ¬ collinear e₁ e₂)

theorem collinear_groups :
  (collinear (- (3 / 2) • e) (2 • e)) ∧
  (collinear (e₁ - e₂) (- 3 • e₁ + 3 • e₂)) ∧
  ¬ (collinear (e₁ - e₂) (e₁ + 2 • e₂)) :=
by sorry

end collinear_groups_l647_647024


namespace sphere_radius_in_cone_l647_647781

theorem sphere_radius_in_cone (r_c h_c r_s : ℝ) (b d : ℝ) : 
  r_c = 15 ∧ 
  h_c = 30 ∧ 
  r_s = b * (Real.sqrt d) - b ∧ 
  r_s = 7.5 * (Real.sqrt 5) - 7.5 → 
  b + d = 12.5 := 
by
  intro h
  cases h with hc1 h
  cases h with hc2 h
  cases h with h_r_s h_r_s_expr
  sorry

end sphere_radius_in_cone_l647_647781


namespace cutGrid_correct_l647_647822

-- Definitions from the conditions
def Symbol := {Circle, Star}
def Grid := Fin 4 × Fin 4

structure GridState :=
  (grid : Grid → Option Symbol)
  (rowCount : ∀ (i : Fin 4), (∃ j, grid (i, j) = some Circle ∧ ∃ j, grid (i, j) = some Star))
  (colCount : ∀ (j : Fin 4), (∃ i, grid (i, j) = some Circle ∧ ∃ i, grid (i, j) = some Star))

noncomputable def cutGrid (g : GridState) : Prop :=
  ∃ (part1 part2 part3 part4 : Fin 4 × Fin 4 → bool),
    (∀ (p : Fin 4 × Fin 4), part1 p ∨ part2 p ∨ part3 p ∨ part4 p) ∧
    (∀ (p1 p2 : Fin 4 × Fin 4), part1 p1 = part1 p2 ∧ part2 p1 = part2 p2 ∧ part3 p1 = part3 p2 ∧ part4 p1 = part4 p2) ∧
    (∀ (p : Grid), 
      ∃! (part : Fin 4 × Fin 4 → bool), part p) ∧
    (∃! (part : Fin 4 × Fin 4 → bool), ∃! (q : Fin 4 × Fin 4), part q ∧ g.grid q = some Circle) ∧
    (∃! (part : Fin 4 × Fin 4 → bool), ∃! (q : Fin 4 × Fin 4), part q ∧ g.grid q = some Star)

theorem cutGrid_correct (g : GridState) : cutGrid g :=
by
  sorry

end cutGrid_correct_l647_647822


namespace apples_needed_l647_647610

theorem apples_needed (w_oranges w_apples : ℕ → ℕ) (h : w_oranges 9 = w_apples 6) :
  (∃ x, w_oranges 45 = w_apples x) → (∃ x, x = 30) :=
by
  -- Definitions for weights
  let ratio := 3 / 2
  -- Weight relationship
  have h_ratio : ∀ n m, w_oranges n = w_apples m → n / m = ratio := sorry
  -- Find the number of apples
  have h1 : 45 / x = ratio := sorry
  -- Solve for x
  have h2 : x = 30 := sorry
  exact ⟨x, h2⟩

end apples_needed_l647_647610


namespace rectangle_complex_numbers_l647_647807

noncomputable def magnitudes (p q r s : ℂ) := abs (p*q + p*r + p*s + q*r + q*s + r*s)

theorem rectangle_complex_numbers
  (p q r s : ℂ)
  (h1 : abs (p + q + r + s) = 50)
  (h2 : ∃ a b : ℝ, a = 15 ∧ b = 20 ∧ 
         ((abs (p - q) = a ∧ abs (q - r) = b ∧ abs (r - s) = a ∧ abs (s - p) = b) ∨
          (abs (p - q) = b ∧ abs (q - r) = a ∧ abs (r - s) = b ∧ abs (s - p) = a))) :
  magnitudes p q r s = 625 :=
sorry

end rectangle_complex_numbers_l647_647807


namespace trisha_cookies_count_l647_647667

def area_trapezoid (base1 base2 height : ℝ) : ℝ :=
  (1 / 2) * (base1 + base2) * height

def area_triangle (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem trisha_cookies_count :
  let art_cookies_count := 12
  let art_cookie_base1 := 3
  let art_cookie_base2 := 5
  let art_cookie_height := 3
  let trisha_cookie_base := 3
  let trisha_cookie_height := 4
  let total_dough_area := (area_trapezoid art_cookie_base1 art_cookie_base2 art_cookie_height) * art_cookies_count
  let trisha_cookie_area := area_triangle trisha_cookie_base trisha_cookie_height
  total_dough_area / trisha_cookie_area = 24 :=
by
  sorry

end trisha_cookies_count_l647_647667


namespace inverse_of_composition_l647_647227

variable {X Y Z W V : Type}
variable (p : X → Y) (q : Y → Z) (r : Z → W) (s : W → V)
variable (p_inv : Y → X) (q_inv : Z → Y) (r_inv : W → Z) (s_inv : V → W)

noncomputable def f := s ∘ q ∘ p ∘ r

theorem inverse_of_composition :
  Function.LeftInverse s_inv s →
  Function.LeftInverse q_inv q →
  Function.LeftInverse p_inv p →
  Function.LeftInverse r_inv r →
  Function.LeftInverse ((r_inv) ∘ (p_inv) ∘ (q_inv) ∘ (s_inv)) f :=
by
  intros
  sorry

end inverse_of_composition_l647_647227


namespace solution_l647_647546

noncomputable def complex_satisfies_eq (z : ℂ) : Prop := (1 + complex.I) * z = 2

theorem solution (z : ℂ) (h : complex_satisfies_eq z) : z = 1 - complex.I :=
by sorry

end solution_l647_647546


namespace relationship_among_abc_l647_647860

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := 2 ^ 0.1
noncomputable def c : ℝ := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b :=
by {
  sorry
}

end relationship_among_abc_l647_647860


namespace range_of_a_l647_647129

noncomputable def f (a x : ℝ) : ℝ := (x - 2 * a) * (x^2 + a^2 * x + 2 * a^3)

theorem range_of_a (h : ∀ x < 0, (3 * x^2 + 2 * (a^2 - 2 * a) * x) < 0) : a < 0 ∨ a > 2 :=
by
  intro h
  sorry

end range_of_a_l647_647129


namespace total_profit_l647_647019

theorem total_profit (C_profit : ℝ) (x : ℝ) (h1 : 4 * x = 48000) : 12 * x = 144000 :=
by
  sorry

end total_profit_l647_647019


namespace find_n_mod_10_l647_647482

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l647_647482


namespace sharpshooter_target_orders_l647_647937

theorem sharpshooter_target_orders : 
  let A := 4
  let B := 3
  let C := 2
  (nat.factorial (A + B + C)) / ((nat.factorial A) * (nat.factorial B) * (nat.factorial C)) = 1260 := by
  sorry

end sharpshooter_target_orders_l647_647937


namespace tangent_line_monotonicity_discussion_min_M_value_l647_647891

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a*x - 2*a^2 + 3*a) * Real.exp x

theorem tangent_line (x y : ℝ) (h : x = 1) (ha : a = 0) :
  (3 * Real.exp 1) * x - y - 2 * Real.exp 1 = 0 :=
sorry

theorem monotonicity_discussion (a : ℝ) (x : ℝ) :
  ∃ ia ib ic id,
    { 
      (a > 2 / 3) → increasing_fn (λ x, f x a) ia ↔ decreasing_fn (λ x, f x a) ib 
    ∧ (a < 2 / 3) → increasing_fn (λ x, f x a) ic ↔ decreasing_fn (λ x, f x a) id 
    ∧ (a = 2 / 3) → ∀ x, increasing_fn (λ x, f x a) x
    } :=
sorry

theorem min_M_value (a : ℝ) (ha : a = 1) :
  ∀ m n ∈ Set.Icc (-3 : ℝ) (0 : ℝ), |f m 1 - f n 1| ≤ 1 - 7 / Real.exp(3) :=
sorry

end tangent_line_monotonicity_discussion_min_M_value_l647_647891


namespace find_second_offset_l647_647469

-- Define the given constants
def diagonal : ℝ := 30
def offset1 : ℝ := 10
def area : ℝ := 240

-- The theorem we want to prove
theorem find_second_offset : ∃ (offset2 : ℝ), area = (1 / 2) * diagonal * (offset1 + offset2) ∧ offset2 = 6 :=
sorry

end find_second_offset_l647_647469


namespace product_of_pentagon_points_l647_647676

def Q1 := (6 : ℝ, 0 : ℝ)
def Q3 := (8 : ℝ, 0 : ℝ)
def Q_coordinates := [Q1, (x2, y2), Q3, (x4, y4), (x5, y5)]  -- We define coordinates generically

theorem product_of_pentagon_points :
  (∃ x2 y2 x4 y4 x5 y5 : ℝ,
  (x2 + y2 * complex.I) * 
  (x4 + y4 * complex.I) * 
  (x5 + y5 * complex.I) *
  (Q1.1 + Q1.2 * complex.I) *
  (Q3.1 + Q3.2 * complex.I) = 16806) :=
by
  sorry

end product_of_pentagon_points_l647_647676


namespace find_f6_l647_647868

variable (f : ℝ → ℝ)
variable (h₁ : ∀ x : ℝ, f(-x) = -f(x))
variable (h₂ : ∀ x : ℝ, f(x + 2) = -f(x))

theorem find_f6 : f 6 = 0 :=
by
  sorry

end find_f6_l647_647868


namespace evaluate_expression_l647_647741

theorem evaluate_expression :
  3 * 307 + 4 * 307 + 2 * 307 + 307 * 307 = 97012 := by
  sorry

end evaluate_expression_l647_647741


namespace series_sum_l647_647626

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : b < a)

noncomputable def infinite_series : ℝ := 
∑' n, 1 / ( ((n - 1) * a^2 - (n - 2) * b^2) * (n * a^2 - (n - 1) * b^2) )

theorem series_sum : infinite_series a b = 1 / ((a^2 - b^2) * b^2) := 
by 
  sorry

end series_sum_l647_647626


namespace theo_selling_price_l647_647644

theorem theo_selling_price:
  ∀ (maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell: ℕ),
    maddox_price = 20 → 
    theo_cost = 20 → 
    maddox_sell = 28 →
    maddox_profit = (maddox_sell - maddox_price) * 3 →
    (theo_sell - theo_cost) * 3 = (maddox_profit - 15) →
    theo_sell = 23 := by
  intros maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell
  intros maddox_price_eq theo_cost_eq maddox_sell_eq maddox_profit_eq theo_profit_eq

  -- Use given assumptions
  rw [maddox_price_eq, theo_cost_eq, maddox_sell_eq] at *
  simp at *

  -- Final goal
  sorry

end theo_selling_price_l647_647644


namespace system_solution_l647_647283

variable {R : Type} [CommRing R]
variable (x y : R)

def eq1 : Prop := 12 * x^2 + 4 * x * y + 3 * y^2 + 16 * x = -6
def eq2 : Prop := 4 * x^2 - 12 * x * y + y^2 + 12 * x - 10 * y = -7

theorem system_solution (hx : x = (-3)/4) (hy : y = 1/2) :
  eq1 x y ∧ eq2 x y :=
by {
  rw [hx, hy],
  simp [eq1, eq2],
  exact ⟨by norm_num, by norm_num⟩,
}

#check system_solution

end system_solution_l647_647283


namespace probability_top_three_same_color_l647_647783

/-- 
  A theorem stating the probability that the top three cards from a shuffled 
  standard deck of 52 cards are all of the same color is \(\frac{12}{51}\).
-/
theorem probability_top_three_same_color : 
  let deck := 52
  let colors := 2
  let cards_per_color := 26
  let favorable_outcomes := 2 * 26 * 25 * 24
  let total_outcomes := 52 * 51 * 50
  favorable_outcomes / total_outcomes = 12 / 51 :=
by
  sorry

end probability_top_three_same_color_l647_647783


namespace monotone_decreasing_bounded_sum_l647_647998

/-- Definition of the sequence -/
def a : ℕ → ℝ
| 0       := 0
| 1       := sorry -- used only to assert a1 > 0
| (n + 2) := a (n + 1) + n / a (n + 1)

/-- Hypotheses for all n ≥ 1, a1 > 0 and the recursive rule applies -/
axiom a1_pos : a 1 > 0
axiom recursive_def : ∀ n ≥ 1, a (n + 1) = a n + n / a n

/-- Problem 1: Monotonic decreasing sequence -/
theorem monotone_decreasing : ∀ n ≥ 2, a n - n ≥ a (n + 1) - (n + 1) :=
sorry

/-- Problem 2: Existence of a constant c for bounded sum -/
theorem bounded_sum : ∃ c, ∀ n ≥ 2, ∑ k in finset.range n, (a (k + 1) - (k + 1)) / (k + 2) ≤ c :=
sorry

end monotone_decreasing_bounded_sum_l647_647998


namespace total_expenditure_is_205_l647_647613

-- Define the conditions
def initial_height : ℝ := 2
def growth_rate_spring : ℝ := 0.6
def growth_rate_summer : ℝ := 0.5
def growth_rate_fall : ℝ := 0.4
def growth_rate_winter : ℝ := 0.2
def cut_height : ℝ := 4
def resulting_height_after_cut : ℝ := 2
def initial_cost : ℕ := 100
def cost_increase_per_cut : ℕ := 5
def months_per_season : ℕ := 3

-- Prove the total expenditure for the year given the conditions
theorem total_expenditure_is_205 :
  let expenditure := 100 + 105 in -- sum of expenses for summer and fall cuts
  expenditure = 205 :=
  by
    sorry

end total_expenditure_is_205_l647_647613


namespace balloons_count_l647_647510

theorem balloons_count (balloons_total Sam Fred Mary: ℕ)
  (h1 : Fred = 5)
  (h2 : Mary = 7)
  (h3 : balloons_total = 18) :
  Sam = 6 :=
by
  have h4 : Sam = balloons_total - (Fred + Mary) := sorry
  rw [h1, h2] at h4
  simp at h4
  rw h3 at h4
  exact h4

end balloons_count_l647_647510


namespace cos_identity_l647_647913

theorem cos_identity (α : ℝ) (h : Real.cos (π / 3 - α) = 3 / 5) : 
  Real.cos (2 * π / 3 + α) = -3 / 5 :=
by
  sorry

end cos_identity_l647_647913


namespace find_n_mod_10_l647_647476

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l647_647476


namespace tank_capacity_l647_647392

theorem tank_capacity (x : ℝ) 
  (h1 : 1/4 * x + 180 = 2/3 * x) : 
  x = 432 :=
by
  sorry

end tank_capacity_l647_647392


namespace n_eq_7_mod_10_l647_647488

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l647_647488


namespace no_counterexample_sum5_prime_div5_l647_647079

theorem no_counterexample_sum5_prime_div5 :
  ¬ ∃ (N : ℕ), 
    (N.digits.sum = 5 ∧
     (∀ d, d ∈ N.digits → d ≠ 0) ∧
     Nat.Prime N ∧ 
     N % 5 = 0) :=
sorry

end no_counterexample_sum5_prime_div5_l647_647079


namespace arithmetic_geometric_sum_l647_647951

noncomputable def arithmetic_seq (n : ℕ) : ℕ := 
  n

noncomputable def geometric_seq (n : ℕ) : ℕ := 
  2^(n-1)

def seq_sum (n : ℕ) : ℕ := 
  (n - 1) * 2^n + 1

theorem arithmetic_geometric_sum (n : ℕ) : 
  (∑ k in Finset.range n, (arithmetic_seq k.succ) * (geometric_seq k.succ)) = seq_sum n :=
  sorry

end arithmetic_geometric_sum_l647_647951


namespace generated_number_after_six_operations_mn_sum_l647_647153

-- Defining the initial conditions and the process
variables {p q : ℕ} (hpq : p > q > 0)

-- Intermediate definitions deriving from the conditions
def c1 := (q + 1) * (p + 1) - 1
def c2 := (c1 + 1) * (p + 1) - 1
def c3 := ((c2 + 1) * (c1 + 1)) - 1
def c4 := ((c3 + 1) * (c2 + 1)) - 1
def c5 := ((c4 + 1) * (c3 + 1)) - 1
def c6 := ((c5 + 1) * (c4 + 1)) - 1
def final_number := (q + 1) ^ 8 * (p + 1) ^ 13 - 1

-- Lean statement to prove the number after 6 operations
theorem generated_number_after_six_operations : 
  final_number = (q + 1) ^ 8 * (p + 1) ^ 13 - 1 := 
by {
  -- The proof would go here, but using sorry to leave it for now
  sorry
}

-- Lean statement to prove the value of m + n
theorem mn_sum : 8 + 13 = 21 :=
by {
  -- Trivially proving the sum of m and n
  norm_num
}

end generated_number_after_six_operations_mn_sum_l647_647153


namespace sin_graph_first_max_at_zero_l647_647298

theorem sin_graph_first_max_at_zero (a b c : ℝ) (h_amplitude : a = 3) (h_max_at_zero : ∃ x : ℝ, x = 0 ∧ y = a * sin(b * x + c) ∧ y = a) : c = π/2 ∧ 0 < b :=
by
  sorry

end sin_graph_first_max_at_zero_l647_647298


namespace height_average_inequality_l647_647710

theorem height_average_inequality 
    (a b c d : ℝ)
    (h1 : 3 * a + 2 * b = 2 * c + 3 * d)
    (h2 : a > d) : 
    (|c + d| / 2 > |a + b| / 2) :=
sorry

end height_average_inequality_l647_647710


namespace systematic_sampling_students_removed_l647_647931

theorem systematic_sampling_students_removed (N n : ℕ) (hN : N = 1387) (hn : n = 9) : 
  ∃ (removed_students : ℕ), removed_students = 1 :=
by
  use 1
  -- the proof steps would go here
  sorry

end systematic_sampling_students_removed_l647_647931


namespace triangle_area_less_than_sqrt3_div_3_l647_647262

-- Definitions for a triangle and its properties
structure Triangle :=
  (a b c : ℝ)
  (ha hb hc : ℝ)
  (area : ℝ)

def valid_triangle (Δ : Triangle) : Prop :=
  0 < Δ.a ∧ 0 < Δ.b ∧ 0 < Δ.c ∧ Δ.ha < 1 ∧ Δ.hb < 1 ∧ Δ.hc < 1

theorem triangle_area_less_than_sqrt3_div_3 (Δ : Triangle) (h : valid_triangle Δ) : Δ.area < (Real.sqrt 3) / 3 :=
sorry

end triangle_area_less_than_sqrt3_div_3_l647_647262


namespace find_n_mod_10_l647_647477

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l647_647477


namespace inequality_solution_set_l647_647312

theorem inequality_solution_set (x : ℝ) :
  ((1 / 2 - x) * (x - 1 / 3) > 0) ↔ (1 / 3 < x ∧ x < 1 / 2) :=
by 
  sorry

end inequality_solution_set_l647_647312


namespace grapes_difference_l647_647396

theorem grapes_difference (R A_i A_l : ℕ) 
  (hR : R = 25) 
  (hAi : A_i = R + 2) 
  (hTotal : R + A_i + A_l = 83) : 
  A_l - A_i = 4 := 
by
  sorry

end grapes_difference_l647_647396


namespace domain_of_f_l647_647295

noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 - 1)) / (Real.sqrt (x^2 - x - 2))

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0 ∧ x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end domain_of_f_l647_647295


namespace quarters_for_soda_l647_647657

def quarters_for_chips := 4
def total_dollars := 4

theorem quarters_for_soda :
  (total_dollars * 4) - quarters_for_chips = 12 :=
by
  sorry

end quarters_for_soda_l647_647657


namespace machine_copies_l647_647369

theorem machine_copies (x : ℕ) (h1 : ∀ t : ℕ, t = 30 → 30 * t = 900)
  (h2 : 900 + 30 * 30 = 2550) : x = 55 :=
by
  sorry

end machine_copies_l647_647369


namespace distance_A_to_B_is_30_sqrt_3_l647_647857

noncomputable def distance_A_to_B : ℝ :=
  let south_displacement : ℝ := 50 - 20
  let east_displacement : ℝ := 30 - 10
  let north_east_displacement : ℝ := 30
  let north_component : ℝ := north_east_displacement * real.sqrt(2) / 2
  let east_component : ℝ := north_east_displacement * real.sqrt(2) / 2
  let total_north_displacement : ℝ := - south_displacement + north_component
  let total_east_displacement : ℝ := east_displacement + east_component
  real.sqrt (total_north_displacement ^ 2 + total_east_displacement ^ 2)

theorem distance_A_to_B_is_30_sqrt_3 :
  distance_A_to_B = 30 * real.sqrt(3) :=
  sorry

end distance_A_to_B_is_30_sqrt_3_l647_647857


namespace time_to_empty_is_109_89_hours_l647_647745

noncomputable def calculate_time_to_empty_due_to_leak : ℝ :=
  let R := 1 / 10 -- filling rate in tank/hour
  let Reffective := 1 / 11 -- effective filling rate in tank/hour
  let L := R - Reffective -- leak rate in tank/hour
  1 / L -- time to empty in hours

theorem time_to_empty_is_109_89_hours : calculate_time_to_empty_due_to_leak = 109.89 :=
by
  rw [calculate_time_to_empty_due_to_leak]
  sorry -- Proof steps can be filled in later

end time_to_empty_is_109_89_hours_l647_647745


namespace no_arithmetic_progression_40_terms_l647_647831

noncomputable def is_arith_prog (f : ℕ → ℕ) (a : ℕ) (b : ℕ) : Prop :=
∀ n : ℕ, ∃ k : ℕ, f n = a + n * b

noncomputable def in_form_2m_3n (x : ℕ) : Prop :=
∃ m n : ℕ, x = 2^m + 3^n

theorem no_arithmetic_progression_40_terms :
  ¬ (∃ (a b : ℕ), ∀ n, n < 40 → in_form_2m_3n (a + n * b)) :=
sorry

end no_arithmetic_progression_40_terms_l647_647831


namespace find_x_value_l647_647084

theorem find_x_value :
  ∀ (x : ℝ), (sqrt x / 0.9 + 1.2 / 0.7 = 2.9365079365079367) → x = 1.21 :=
by
  intro x h
  sorry

end find_x_value_l647_647084


namespace subset_B_has_more_elements_l647_647310

-- Definitions of Subsets
def is_sum_of_square_and_cube (n : ℕ) : Prop :=
  ∃ k m : ℕ, k^2 + m^3 = n

def A : finset ℕ := (finset.range 1000000).filter is_sum_of_square_and_cube
def B : finset ℕ := (finset.range 1000000).filter (λ n, ¬ is_sum_of_square_and_cube n)

-- Theorem
theorem subset_B_has_more_elements :
  B.card > A.card :=
by
  sorry

end subset_B_has_more_elements_l647_647310


namespace current_average_is_35_l647_647763

noncomputable def cricket_avg (A : ℝ) : Prop :=
  let innings := 10
  let next_runs := 79
  let increase := 4
  (innings * A + next_runs = (A + increase) * (innings + 1))

theorem current_average_is_35 : cricket_avg 35 :=
by
  unfold cricket_avg
  simp only
  sorry

end current_average_is_35_l647_647763


namespace arithmetic_mean_sqrt_two_l647_647288

theorem arithmetic_mean_sqrt_two :
  (sqrt 2 + 1 + (sqrt 2 - 1)) / 2 = sqrt 2 :=
by
  sorry

end arithmetic_mean_sqrt_two_l647_647288


namespace find_decreasing_interval_l647_647052

noncomputable def f (x : ℝ) : ℝ := x^2 - log x

def f_prime (x : ℝ) : ℝ := 2 * x - 1 / x

def decreasing_interval (a b : ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, a < x ∧ x ≤ b → f' x ≤ 0

theorem find_decreasing_interval : decreasing_interval 0 (sqrt 2 / 2) f_prime :=
begin
  sorry,
end

end find_decreasing_interval_l647_647052


namespace avg_wx_l647_647166

theorem avg_wx (w x y : ℝ) (h1 : 3 / w + 3 / x = 3 / y) (h2 : w * x = y) : (w + x) / 2 = 1 / 2 :=
by
  -- omitted proof
  sorry

end avg_wx_l647_647166


namespace set_A_exactly_two_elements_l647_647977

theorem set_A_exactly_two_elements (A : Set ℕ) (hA1 : 2 ≤ A.card) 
  (hA2 : ∀ a b, a ∈ A → b ∈ A → a > b → (Nat.lcm a b) / (a - b) ∈ A) :
  A.card = 2 :=
by
  sorry

end set_A_exactly_two_elements_l647_647977


namespace hamburger_varieties_l647_647156

-- Define the problem conditions as Lean definitions.
def condiments := 9  -- There are 9 condiments
def patty_choices := 3  -- Choices of 1, 2, or 3 patties

-- The goal is to prove that the number of different kinds of hamburgers is 1536.
theorem hamburger_varieties : (3 * 2^9) = 1536 := by
  sorry

end hamburger_varieties_l647_647156


namespace range_of_f_l647_647549

noncomputable def f (x : ℝ) := log2 (3 * x + 1)

theorem range_of_f : 
  (∀ x, 0 < x → ∃ y, y = f x ∧ 0 < y) :=
by
  intro x hx
  use f x
  split
  · rfl
  · sorry

end range_of_f_l647_647549


namespace percent_decrease_in_square_area_l647_647603

theorem percent_decrease_in_square_area (A B C D : Type) 
  (side_length_AD side_length_AB side_length_CD : ℝ) 
  (area_square_original new_side_length new_area : ℝ) 
  (h1 : side_length_AD = side_length_AB) (h2 : side_length_AD = side_length_CD) 
  (h3 : area_square_original = side_length_AD^2)
  (h4 : new_side_length = side_length_AD * 0.8)
  (h5 : new_area = new_side_length^2)
  (h6 : side_length_AD = 9) : 
  (area_square_original - new_area) / area_square_original * 100 = 36 := 
  by 
    sorry

end percent_decrease_in_square_area_l647_647603


namespace solve_r_l647_647044

def E (a : ℝ) (b : ℝ) (c : ℕ) : ℝ := a * b^c

theorem solve_r : ∃ (r : ℝ), E r r 5 = 1024 ∧ r = 2^(5/3) :=
by
  sorry

end solve_r_l647_647044


namespace M_is_incenter_l647_647447

variable (A B C M : Type) [triangle A B C] 

-- Define the problem conditions
structure Circle (P Q : Type) :=
(center : Type)
(radius : ℝ)

noncomputable def circle_on_segment (M A : Type) : Circle M A := sorry

-- Specify the property of M we need to prove
def isIncenter (M : Type) (A B C : Type) [inhabited M] : Prop := 
  ∀ P Q R : Type,
  (circle_on_segment M A).common_chord_length = (circle_on_segment M B).common_chord_length ∧
  (circle_on_segment M B).common_chord_length = (circle_on_segment M C).common_chord_length

-- The theorem statement
theorem M_is_incenter 
  (A B C M : Type) [triangle A B C] [inhabited M] :
  (isIncenter M A B C) → Incenter M A B C :=
sorry

end M_is_incenter_l647_647447


namespace calc_distances_l647_647311

theorem calc_distances (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
  (h : EquilateralTriangle A B C) (h1 : Distance A B = 1) (h2 : Distance B C = 1) (h3 : Distance C A = 1)
  (angle_BPC : Angle B P C = 30) (angle_CPA : Angle C P A = 50) (angle_APB : Angle A P B = 80) :
  Distance P A = 1 ∧ Distance P B = 0.3472 ∧ Distance P C = 1.2856 :=
by
  sorry

end calc_distances_l647_647311


namespace general_formula_l647_647113

noncomputable def a : ℕ → ℕ
| 0        := 0  -- Indexing starts from 1, hence placeholder for a_0
| 1        := 3
| (n + 2)  := 2 * a (n + 1) + 1

theorem general_formula (n : ℕ) :
  a (n + 1) = 2^(n + 1) - 1 :=
sorry

end general_formula_l647_647113


namespace a_divisibility_sum_of_roots_integer_l647_647706

noncomputable def a : ℕ → polynomial ℤ
| 0       := 0
| 1       := polynomial.X + 2
| (n + 2) := a (n + 1) + 3 * (a (n + 1) * a n) + a n

theorem a_divisibility (k m : ℕ) (hk: k > 0) (hm: m > 0) (hk_dvd_hm: k ∣ m) : a k ∣ a m :=
sorry

theorem sum_of_roots_integer (n : ℕ) (hn: n > 0) : 
  (∃ k : ℕ, n = 1 ∨ n = 2 ∨ (n = 4 * k ∧ k ≥ 1)) ↔
  ∃ (s : ℕ), a n.roots.sum = s :=
sorry

end a_divisibility_sum_of_roots_integer_l647_647706


namespace find_x_l647_647155

def vector_a : Vector ℝ 2 := ![1, 1]
def vector_b : Vector ℝ 2 := ![2, 5]
def vector_c (x : ℝ) : Vector ℝ 2 := ![3, x]

theorem find_x (x : ℝ) (h : dot_product (8 • vector_a - vector_b) (vector_c x) = 30) : x = 4 :=
by
  /* The definition of dot_product, scalar multiplication, and vector subtraction are assumed 
     to be available in Mathlib. If not, appropriate definitions need to be included. */
  sorry

end find_x_l647_647155


namespace conditional_probability_l647_647092

/-
Problem: 
From the numbers {1, 2, 3, 4, 5, 6, 7}, two different numbers are randomly selected. Event A is "the sum of the two numbers is even", and event B is "both numbers are even". Then P(B|A) = 1/3.
-/

def numbers : set ℕ := {1, 2, 3, 4, 5, 6, 7}

def event_A (a b : ℕ) : Prop := (a + b) % 2 = 0
def event_B (a b : ℕ) : Prop := (a % 2 = 0) ∧ (b % 2 = 0)

def all_pairs := (numbers.powerset.filter (λ s, s.card = 2)).to_finset
def pairs_event_A := all_pairs.filter (λ s, ∃ a b, a ∈ s ∧ b ∈ s ∧ event_A a b)
def pairs_event_B := all_pairs.filter (λ s, ∃ a b, a ∈ s ∧ b ∈ s ∧ event_B a b)

def P (s : finset (finset ℕ)) : ℚ := s.card / all_pairs.card

theorem conditional_probability :
  P (pairs_event_B ∩ pairs_event_A) / P pairs_event_A = 1 / 3 := by
  sorry

end conditional_probability_l647_647092


namespace inequality_proof_l647_647534

theorem inequality_proof (n k : ℕ) (an : Fin n → ℝ) (h_pos : ∀ i, an i > 0) (h_n : n ≥ 2) (h_k : k ≥ 1) : 
  (Finset.univ.sum (λ i, (an i / (Finset.univ.sum (λ j, if j ≠ i then an j else 0)))^k)) 
  ≥ n / ((n-1)^k) :=
sorry

end inequality_proof_l647_647534


namespace factor_tree_value_l647_647180

-- Define the values and their relationships
def A := 900
def B := 3 * (3 * 2)
def D := 3 * 2
def C := 5 * (5 * 2)
def E := 5 * 2

-- Define the theorem and provide the conditions
theorem factor_tree_value :
  (B = 3 * D) →
  (D = 3 * 2) →
  (C = 5 * E) →
  (E = 5 * 2) →
  (A = B * C) →
  A = 900 := by
  intros hB hD hC hE hA
  sorry

end factor_tree_value_l647_647180


namespace varphi_monotonic_intervals_unique_tangent_point_in_interval_l647_647896

/-- Given the functions f(x) = ln x and g(x) = e^x, prove that the function 
varphi(x) = ln x - (x+1) / (x-1) is monotonically increasing in the intervals (0, 1) and (1, +∞). --/
theorem varphi_monotonic_intervals :
  (∀ x ∈ set.Ioo 0 1, deriv (λ x : ℝ, Real.log x - (x + 1) / (x - 1)) x > 0) ∧
  (∀ x ∈ set.Ioi 1, deriv (λ x : ℝ, Real.log x - (x + 1) / (x - 1)) x > 0) := sorry

/-- Given the functions f(x) = ln x and g(x) = e^x, prove that there exists a unique x_0 in the interval 
(1, +∞), such that the line tangent to f(x) at x_0 is also tangent to g(x). --/
theorem unique_tangent_point_in_interval :
  ∃! x ∈ set.Ioo 1 (Real.exp 2), ∃ (x_1 : ℝ), x_1 = -Real.log x ∧ 
  (∀ y ∈ set.univ, (λ x : ℝ, 1/x * (x - x_0) + Real.log x - 1) y = (λ x : ℝ, 1/x_0 * x + Real.log x_0 - 1) y) := sorry

end varphi_monotonic_intervals_unique_tangent_point_in_interval_l647_647896


namespace number_of_1989_periodic_points_l647_647633

noncomputable def f (z : ℂ) (m : ℕ) : ℂ := z ^ m

noncomputable def is_periodic_point (z : ℂ) (f : ℂ → ℂ) (n : ℕ) : Prop :=
f^[n] z = z ∧ ∀ k : ℕ, k < n → (f^[k] z) ≠ z

noncomputable def count_periodic_points (m n : ℕ) : ℕ :=
m^n - m^(n / 3) - m^(n / 13) - m^(n / 17) + m^(n / 39) + m^(n / 51) + m^(n / 117) - m^(n / 153)

theorem number_of_1989_periodic_points (m : ℕ) (hm : 1 < m) :
  count_periodic_points m 1989 = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 :=
sorry

end number_of_1989_periodic_points_l647_647633


namespace sum_series_l647_647359

theorem sum_series : (∑ n in Finset.range 2500, (2 * n + 1) - (2 * n + 2)) = -2500 := 
by
  sorry

end sum_series_l647_647359


namespace totalDistinctParabolas_l647_647191

def isDistinct (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def validParameters (a b c : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ isDistinct a b c

def countDistinctParabolas : ℕ :=
  32

theorem totalDistinctParabolas :
  (∃ (a b c : ℤ), a ∈ {-2, 0, 1, 2, 3} ∧ b ∈ {-2, 0, 1, 2, 3} ∧ c ∈ {-2, 0, 1, 2, 3} ∧ validParameters a b c) →
  countDistinctParabolas = 32 :=
sorry

end totalDistinctParabolas_l647_647191


namespace ratio_of_inscribed_squares_l647_647782

noncomputable def right_triangle_sides_6_8_10 : Prop := ∀ (a b c : ℝ), a = 6 ∧ b = 8 ∧ c = 10 ∧ (a^2 + b^2 = c^2)

noncomputable def x_inscribed_square : Prop := ∀ (x : ℝ), (∃ (a b c : ℝ), right_triangle_sides_6_8_10 a b c ∧ x = 3)
noncomputable def y_inscribed_square : Prop := ∀ (y : ℝ), (∃ (a b c : ℝ), right_triangle_sides_6_8_10 a b c ∧ y = 8 / 3)

theorem ratio_of_inscribed_squares : (∀ (x y : ℝ), x_inscribed_square x ∧ y_inscribed_square y → x / y = 9 / 8) :=
by
  sorry

end ratio_of_inscribed_squares_l647_647782


namespace inscribed_circle_radius_l647_647277

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (angle : ℝ):
  R = 6 → angle = 2 * Real.pi / 3 → r = (6 * Real.sqrt 3) / 5 :=
by
  sorry

end inscribed_circle_radius_l647_647277


namespace odd_function_with_smallest_period_l647_647397

-- Definitions of the functions based on the given conditions
def fA (x : ℝ) : ℝ := sin (2 * x + π / 2)
def fB (x : ℝ) : ℝ := cos (2 * x + π / 2)
def fC (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)
def fD (x : ℝ) : ℝ := sin x + cos x

-- Statement of the problem
theorem odd_function_with_smallest_period (f : ℝ → ℝ) :
  f = fB ∧ (∀ g, (g = fA ∨ g = fC ∨ g = fD) → ¬ (g = f ∧ odd g ∧ ∀ T, T > 0 → period g T = π)) :=
sorry

end odd_function_with_smallest_period_l647_647397


namespace area_two_layers_l647_647712

/-- Definitions of areas in the rug problem -/
variables (A B C D : ℕ)

/-- Given conditions -/
def rug_conditions :=
  A = 200 ∧
  D = 19 ∧
  A - (B + C + D) = 138 ∧
  C + D = 24

/-- The theorem to be proved -/
theorem area_two_layers (h : rug_conditions A B C D) : C = 5 :=
by {
  sorry
}

end area_two_layers_l647_647712


namespace password_probability_l647_647004

theorem password_probability 
  (password : Fin 6 → Fin 10) 
  (attempts : ℕ) 
  (correct_digit : Fin 10) 
  (probability_first_try : ℚ := 1 / 10)
  (probability_second_try : ℚ := (9 / 10) * (1 / 9)) : 
  ((password 5 = correct_digit) ∧ attempts ≤ 2) →
  (probability_first_try + probability_second_try = 1 / 5) :=
sorry

end password_probability_l647_647004


namespace find_d_l647_647325

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ k : Nat, k > 1 → k < n → n % k ≠ 0)

def less_than_10_primes (n : Nat) : Prop :=
  n < 10 ∧ is_prime n

theorem find_d (d e f : Nat) (hd : less_than_10_primes d) (he : less_than_10_primes e) (hf : less_than_10_primes f) :
  d + e = f → d < e → d = 2 :=
by
  sorry

end find_d_l647_647325


namespace polynomial_remainder_l647_647082

theorem polynomial_remainder (f : ℤ[X]) (a: ℤ) (h : f = X^12 - 1) :
  polynomial.aeval a f = 0 :=
by
  -- problem statement is set up, proof can be added here
  sorry

end polynomial_remainder_l647_647082


namespace chord_length_l647_647495

noncomputable def length_of_chord (x y : ℝ) (C : Real) : ℝ :=
  let center := (3 : ℝ, 0 : ℝ)
  let radius := 3
  let d := |3 - C| / sqrt (1 + 3)
  2 * sqrt (radius^2 - d^2)

theorem chord_length :
  length_of_chord (3 : ℝ) (0 : ℝ) 4 = sqrt 35 :=
by
  sorry

end chord_length_l647_647495


namespace incenter_chords_equal_l647_647442

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry Triangle

/-- Given a triangle ABC and a point M inside it,
  if circles are constructed on segments MA, MB, and MC as diameters,
  then the point M such that the lengths of common chords are equal, 
  is the incenter of triangle ABC. -/
theorem incenter_chords_equal (A B C M : α) (hM : incircle M (Triangle.mk A B C))
  (h_chord_eq : ∀ (P Q R : α), is_common_chord P (circle (segment.mk M A))
    (circle (segment.mk M B)) = is_common_chord Q (circle (segment.mk M B)) 
    (circle (segment.mk M C)) = is_common_chord R (circle (segment.mk M C))
    (circle (segment.mk M A))) : is_incenter M (Triangle.mk A B C) :=
sorry

end incenter_chords_equal_l647_647442


namespace cost_per_hat_l647_647213

theorem cost_per_hat (num_hats : ℕ) (total_cost : ℕ) (hats_per_week : ℕ) (weeks : ℕ) (hats_eq : hats_per_week * weeks = num_hats) (total_cost_eq : total_cost = 700) (weeks_eq : weeks = 2) (hats_per_week_eq : hats_per_week = 7) : 
  total_cost / num_hats = 50 :=
by {
  have h1 : num_hats = hats_per_week * weeks, from hats_eq,
  have h2 : num_hats = 14, {
    rw [weeks_eq, hats_per_week_eq] at h1,
    exact h1
  },
  rw h2 at *,
  have h3 : total_cost = 700, from total_cost_eq,
  calc total_cost / num_hats
      = 700 / 14 : by rw h3
  ... = 50       : by norm_num
}

end cost_per_hat_l647_647213


namespace height_of_wall_l647_647367

-- Definitions
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 850
def wall_width : ℝ := 22.5
def num_bricks : ℝ := 6800

-- Total volume of bricks
def total_brick_volume : ℝ := num_bricks * brick_length * brick_width * brick_height

-- Volume of the wall
def wall_volume (height : ℝ) : ℝ := wall_length * wall_width * height

-- Proof statement
theorem height_of_wall : ∃ h : ℝ, wall_volume h = total_brick_volume ∧ h = 600 := 
sorry

end height_of_wall_l647_647367


namespace cristina_pace_correct_l647_647656

-- Definitions of the conditions
def head_start : ℕ := 30
def nicky_pace : ℕ := 3  -- meters per second
def time_for_catch_up : ℕ := 15  -- seconds

-- Distance covers by Nicky
def nicky_distance : ℕ := nicky_pace * time_for_catch_up

-- Total distance covered by Cristina to catch up Nicky
def cristina_distance : ℕ := nicky_distance + head_start

-- Cristina's pace
def cristina_pace : ℕ := cristina_distance / time_for_catch_up

-- Theorem statement
theorem cristina_pace_correct : cristina_pace = 5 := by 
  sorry

end cristina_pace_correct_l647_647656


namespace solve_quadratic_l647_647681

theorem solve_quadratic (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by
  sorry

end solve_quadratic_l647_647681


namespace kim_money_l647_647214

-- Define the involved variables
variables (K S P : ℝ)

-- Define the conditions
def condition1 : Prop := K = 1.40 * S
def condition2 : Prop := S = 0.80 * P
def condition3 : Prop := S + P = 1.80

-- State the theorem to prove
theorem kim_money : condition1 K S P ∧ condition2 S P ∧ condition3 S P → K = 1.12 :=
by
  intros h
  sorry

end kim_money_l647_647214


namespace find_n_l647_647491

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l647_647491


namespace problem_proof_l647_647874

theorem problem_proof (a b x y : ℝ) (h1 : a + b = 0) (h2 : x * y = 1) : 5 * |a + b| - 5 * (x * y) = -5 :=
by
  sorry

end problem_proof_l647_647874


namespace fraction_of_bag_spent_on_lunch_l647_647273

-- Definitions of conditions based on the problem
def initial_amount : ℕ := 158
def price_of_shoes : ℕ := 45
def price_of_bag : ℕ := price_of_shoes - 17
def amount_left : ℕ := 78
def money_before_lunch := amount_left + price_of_shoes + price_of_bag
def money_spent_on_lunch := initial_amount - money_before_lunch 

-- Statement of the problem in Lean
theorem fraction_of_bag_spent_on_lunch :
  (money_spent_on_lunch : ℚ) / price_of_bag = 1 / 4 :=
by
  -- Conditions decoded to match the solution provided
  have h1 : price_of_bag = 28 := by sorry
  have h2 : money_before_lunch = 151 := by sorry
  have h3 : money_spent_on_lunch = 7 := by sorry
  -- The main theorem statement
  exact sorry

end fraction_of_bag_spent_on_lunch_l647_647273


namespace domain_of_transformed_function_l647_647922

def f (x : ℝ) : ℝ := sorry -- The actual definition of f is not necessary for this problem.

theorem domain_of_transformed_function :
  (∀ x, 1 ≤ x + 1 ∧ x + 1 ≤ 2 → 1 ≤ 2^x - 2 ∧ 2^x - 2 ≤ 2) →
  ∀ x, log 2 3 ≤ x ∧ x ≤ 2 → 1 ≤ 2^x - 2 ∧ 2^x - 2 ≤ 2 :=
begin
  intro h,
  intro x,
  intro hx,
  sorry -- Proof goes here.
end

end domain_of_transformed_function_l647_647922


namespace euler_totient_divisibility_l647_647357

theorem euler_totient_divisibility (a n: ℕ) (h1 : a ≥ 2) : (n ∣ Nat.totient (a^n - 1)) :=
sorry

end euler_totient_divisibility_l647_647357


namespace leading_coefficient_polynomial_l647_647051

theorem leading_coefficient_polynomial :
  let p := -5 * (X^5 - 2 * X^4 + 3 * X^3) + 4 * (X^5 + X^2 - 5) - 3 * (3 * X^5 + X^3 + 4)
  leading_coeff p = -10 :=
by sorry

end leading_coefficient_polynomial_l647_647051


namespace find_point_incenter_l647_647454

open Real

-- Define the incenter condition for a point M inside a triangle ABC
def isIncenter (A B C M : Point) : Prop :=
  (dist M (line.through A B) = dist M (line.through B C)) ∧ (dist M (line.through B C) = dist M (line.through C A))

-- Define the problem statement
theorem find_point_incenter (A B C : Point) :
  ∃ M : Point, isIncenter A B C M :=
by
  sorry

end find_point_incenter_l647_647454


namespace intersection_A_B_l647_647573

def A : Set ℝ := { x | (x + 1) / (x - 1) ≤ 0 }
def B : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l647_647573


namespace cheetahs_pandas_ratio_l647_647939

-- Let C denote the number of cheetahs 5 years ago.
-- Let P denote the number of pandas 5 years ago.
-- The conditions given are:
-- 1. The ratio of cheetahs to pandas 5 years ago was the same as it is now.
-- 2. The number of cheetahs has increased by 2.
-- 3. The number of pandas has increased by 6.
-- We need to prove that the current ratio of cheetahs to pandas is C / P.

theorem cheetahs_pandas_ratio
  (C P : ℕ)
  (h1 : C / P = (C + 2) / (P + 6)) :
  (C + 2) / (P + 6) = C / P :=
by sorry

end cheetahs_pandas_ratio_l647_647939


namespace sum_fractions_eq_1498_l647_647414

theorem sum_fractions_eq_1498 :
  let S := ∑ n in Finset.range 1000, (2 / ((n + 1) * (n + 3)))
  (S : ℝ).toFixed 3 = 1.498 :=
by
  let S := ∑ n in Finset.range 1000, (2 / ((n + 1) * (n + 3)))
  have : (S : ℝ).toFixed 3 = 1.498 := sorry
  exact this

end sum_fractions_eq_1498_l647_647414


namespace incenter_chords_equal_l647_647444

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry Triangle

/-- Given a triangle ABC and a point M inside it,
  if circles are constructed on segments MA, MB, and MC as diameters,
  then the point M such that the lengths of common chords are equal, 
  is the incenter of triangle ABC. -/
theorem incenter_chords_equal (A B C M : α) (hM : incircle M (Triangle.mk A B C))
  (h_chord_eq : ∀ (P Q R : α), is_common_chord P (circle (segment.mk M A))
    (circle (segment.mk M B)) = is_common_chord Q (circle (segment.mk M B)) 
    (circle (segment.mk M C)) = is_common_chord R (circle (segment.mk M C))
    (circle (segment.mk M A))) : is_incenter M (Triangle.mk A B C) :=
sorry

end incenter_chords_equal_l647_647444


namespace product_modulo_seven_l647_647810

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l647_647810


namespace area_ratio_of_triangles_l647_647081

theorem area_ratio_of_triangles
  (A B C X : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X]
  (h_bisect : ∃ t ∈ Ioo 0 1, X = t • A + (1 - t) • B ∧ Bisects (C, ∠ACB))
  (BC_length : dist B C = 32)
  (AC_length : dist A C = 35) :
  (area_of_triangle B C X) / (area_of_triangle A C X) = 32 / 35 := 
  sorry

end area_ratio_of_triangles_l647_647081


namespace minimum_distance_l647_647303

-- Define a function to calculate the distance from a point to a line
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  (abs (A * x0 + B * y0 + C)) / (real.sqrt (A * A + B * B))

-- Define the line 2x - y - 10 = 0
def line (x y : ℝ) : Prop :=
  2 * x - y - 10 = 0

-- Define the parabola y = x^2
def parabola (x y : ℝ) : Prop :=
  y = x * x

-- Define the minimum distance problem
theorem minimum_distance :
  ∃ (d : ℝ), (∀ (x0 : ℝ), ∃ (y0 : ℝ), parabola x0 y0) ∧
  (∀ (x0 : ℝ), ∃ (y0 : ℝ), line x0 y0) ∧
  d = (sqrt 5) / 5 * abs (-(x0 - 1) * (x0 - 1) + 9) :=
sorry

end minimum_distance_l647_647303


namespace exists_pointQ_l647_647187

noncomputable def circleC' : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in (x - 1)^2 + (y - 1)^2 = 8

noncomputable def circleC : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in (x + 2)^2 + (y - 2)^2 = 8

noncomputable def pointF : (ℝ × ℝ) := (4, 0)

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem exists_pointQ :
  ∃ Q : ℝ × ℝ,
    circleC Q ∧ distance Q pointF = distance (0, 0) pointF ∧ Q ≠ (0, 0) :=
  sorry

end exists_pointQ_l647_647187


namespace prime_subtraction_problem_l647_647719

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def prime_between (a b : ℕ) (x : ℕ) : Prop :=
  a < x ∧ x < b ∧ is_prime x

theorem prime_subtraction_problem :
  ¬ ∃ (x y : ℕ), (prime_between 10 50 x) ∧ (prime_between 10 50 y) ∧ x ≠ y ∧ 
  (let z := (x * y) - (x + y) in z = 221 ∨ z = 470 ∨ z = 629 ∨ z = 899 ∨ z = 950) :=
by
  sorry

end prime_subtraction_problem_l647_647719


namespace div_by_eleven_l647_647915

theorem div_by_eleven (n : ℤ) : 11 ∣ ((n + 11)^2 - n^2) :=
by
  sorry

end div_by_eleven_l647_647915


namespace binomial_variance_expectation_ratio_l647_647110

open ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

noncomputable def xi (n p : ℝ) : MeasureTheory.Measure Ω :=
  binomialDistribution n p 

theorem binomial_variance_expectation_ratio (n p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let ξ := xi n p
  (variance ξ) ^ 2 / (expectation ξ) ^ 2 = (1 - p) ^ 2 := by
  sorry

end binomial_variance_expectation_ratio_l647_647110


namespace quadratic_properties_l647_647899

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  -- 1. The parabola opens downwards.
  (∀ x : ℝ, quadratic_function x < quadratic_function (x + 1) → false) ∧
  -- 2. The axis of symmetry is x = 1.
  (∀ x : ℝ, ∃ y : ℝ, quadratic_function x = quadratic_function y → x = y ∨ x + y = 2) ∧
  -- 3. The vertex coordinates are (1, 5).
  (quadratic_function 1 = 5) ∧
  -- 4. y decreases for x > 1.
  (∀ x : ℝ, x > 1 → quadratic_function x < quadratic_function (x - 1)) :=
by
  sorry

end quadratic_properties_l647_647899


namespace overlap_square_area_l647_647332

variables (R₁ R₂ : Type) [rect1 : has_extent 4 12 R₁] [rect2 : has_extent 3 7 R₂] [rotate_45 : rotated_by_45 R₁ R₂]

theorem overlap_square_area (h : overlap_is_square R₁ R₂) : overlap_area R₁ R₂ = 9 :=
sorry

end overlap_square_area_l647_647332


namespace carla_drank_total_amount_l647_647806

-- Define the conditions
def carla_water : ℕ := 15
def carla_soda := 3 * carla_water - 6
def total_liquid := carla_water + carla_soda

-- State the theorem
theorem carla_drank_total_amount : total_liquid = 54 := by
  sorry

end carla_drank_total_amount_l647_647806


namespace angle_D_l647_647967

theorem angle_D'_FE_is_90 
  (A B C : Type*)
  [metric_space A] [metric_space B] [metric_space C]
  (triangle_ABC : triangle A B C)
  (D E : A)
  (bisects_ADC : bisector (angle BAC) D)
  (exbisects_BAC : bissector (external_angle BAC) E)
  (circumcircle : circle A B C)
  (F ∈ circumcircle) (AD : line A D)
  (O : center_of_circumcircle (triangle_ABC))
  (D' : reflection_of D O) :
  ∠ D' F E = 90° :=
by
  sorry

end angle_D_l647_647967


namespace option_C_same_function_option_D_same_function_l647_647347

theorem option_C_same_function (x : ℝ) (h : x ≥ -2) : 
  ((sqrt (x + 2))^2 = x + 2) :=
by
  sorry
  
theorem option_D_same_function (x : ℝ) (h : x ≠ 0) : 
  (x^0 = 1) :=
by
  sorry

end option_C_same_function_option_D_same_function_l647_647347


namespace AC_bisects_angle_DAB_l647_647953

variable (A B C D : Type) [Σ : EuclideanGeometry A B C D]

noncomputable def ratio_AB_AC : ℝ := 2 / 3
noncomputable def ratio_AC_AD : ℝ := 2 / 3
noncomputable def ratio_BC_CD : ℝ := 2 / 3

theorem AC_bisects_angle_DAB (H₁ : AB / AC = ratio_AB_AC) (H₂ : AC / AD = ratio_AC_AD) (H₃ : BC / CD = ratio_BC_CD) 
: Bisection (Angle D A B) AC := 
sorry

end AC_bisects_angle_DAB_l647_647953


namespace triangle_angle_sum_l647_647338

theorem triangle_angle_sum (y : ℝ) (h : 40 + 3 * y + (y + 10) = 180) : y = 32.5 :=
by
  sorry

end triangle_angle_sum_l647_647338


namespace find_cos_alpha_l647_647124

theorem find_cos_alpha (α : ℝ) (h1 : sin ((30 : ℝ) * Real.pi / 180 + α) = 3 / 5)
  (h2 : (60 : ℝ) * Real.pi / 180 < α ∧ α < (150 : ℝ) * Real.pi / 180) :
  cos α = (3 - 4 * Real.sqrt 3) / 10 := 
sorry

end find_cos_alpha_l647_647124


namespace total_bill_is_correct_l647_647771

-- Given conditions
def hourly_rate := 45
def parts_cost := 225
def hours_worked := 5

-- Total bill calculation
def labor_cost := hourly_rate * hours_worked
def total_bill := labor_cost + parts_cost

-- Prove that the total bill is equal to 450 dollars
theorem total_bill_is_correct : total_bill = 450 := by
  sorry

end total_bill_is_correct_l647_647771


namespace rose_work_days_l647_647212

theorem rose_work_days (work : ℝ) : 
  (john : ℝ) = 8 → (together : ℝ) = 16/3 → ((rose : ℝ), 1 / rose = 3/16 - 1/8) :=
by
  intro john_8
  intro together_16_by_3
  have john_rate : ℝ := 1 / john_8
  have together_rate : ℝ := 3 / 16
  have rose_rate : ℝ := together_rate - john_rate
  use (rose : ℝ), rose_rate = 1 / rose
  sorry

end rose_work_days_l647_647212


namespace rooks_in_4x5_rectangle_l647_647362

-- Definition of a rook on a chessboard
structure Rook where
  row : ℕ
  col : ℕ

-- Conditions of the problem
def non_attacking_rooks (rooks : List Rook) : Prop :=
  ∀ (r1 r2 : Rook), r1 ≠ r2 → r1.row ≠ r2.row ∧ r1.col ≠ r2.col

def rook_in_bounds (rook : Rook) : Prop :=
  rook.row < 8 ∧ rook.col < 8

def all_rooks_in_bounds (rooks : List Rook) : Prop :=
  ∀ r ∈ rooks, rook_in_bounds r

-- The proof statement
theorem rooks_in_4x5_rectangle (rooks : List Rook) :
  length rooks = 8 →
  non_attacking_rooks rooks →
  all_rooks_in_bounds rooks →
  ∀ (i j : ℕ), i < 5 → j < 4 →
    ∃ r ∈ rooks, r.row ∈ i + 4 ∧ r.col ∈ j + 5 := sorry

end rooks_in_4x5_rectangle_l647_647362


namespace gcd_168_54_264_l647_647075

theorem gcd_168_54_264 : Nat.gcd (Nat.gcd 168 54) 264 = 6 :=
by
  -- proof goes here and ends with sorry for now
  sorry

end gcd_168_54_264_l647_647075


namespace ellipse_tangent_product_l647_647887

theorem ellipse_tangent_product (k m : ℝ)
  (h1 : ∀ x y : ℝ, (x^2) / 6 + (y^2) / 2 = 1)
  (h2 : ∀ x : ℝ, (y : ℝ) = k * x + m)
  (h3 : m^2 = 6 * k^2 + 2)
  (h4 : F1 : ℝ × ℝ := (-2, 0))
  (h5 : F2 : ℝ × ℝ := (2, 0))
  (d1 : ℝ := |2 * k + m| / real.sqrt (k^2 + 1))
  (d2 : ℝ := |-2 * k + m| / real.sqrt (k^2 + 1)) :
  d1 * d2 = 2 :=
by sorry

end ellipse_tangent_product_l647_647887


namespace position_XUSAMO_l647_647725

open List

/-
Define the set of letters.
-/
def letters : List Char := ['A', 'M', 'O', 'S', 'U', 'X']

/-
Define the function that computes the alphabetical position of a word within all permutations of the given set of letters.
-/
def word_position (word : List Char) : Nat :=
  findIndex? (fun w => w = word) (letters.permutations.map (fun x => x)).get!

/-
State that the position of the word "XUSAMO" within the permutations of the given letters equals 673.
-/
theorem position_XUSAMO : word_position ['X', 'U', 'S', 'A', 'M', 'O'] = 673 := by
  sorry

end position_XUSAMO_l647_647725


namespace number_of_excellent_students_l647_647769

theorem number_of_excellent_students (μ δ : ℝ) (n : ℕ) (P_60 : ℝ) (X : ℝ → ℝ) 
  (hX : ∀ x, X x = (1 / (δ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * δ^2))) 
  (hmean : μ = 90) 
  (hprob60 : P_60 = 0.1) 
  (hP60 : ∫ x in -∞..60, X x = P_60) 
  (hn_students : n = 1200) :
  ∃ count : ℕ, count = 120 :=
by
  have hP120 : ∫ x in 120..∞, X x = 0.1 := sorry
  have hcount : count = n * 0.1 := sorry
  use 120
  sorry

end number_of_excellent_students_l647_647769


namespace triangle_cos_half_angle_l647_647963

noncomputable theory

-- Assume \( \triangle ABC \) with \( A, B, C \) angles such that \( A = 2x \), \( B = 3x \), \( C = 4x \)
variables {A B C a b c : ℝ}

-- Assume the angles in the conditions
axiom angle_ratios (x : ℝ) : A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ A + B + C = 180

-- Assume \( \cos \frac{A}{2} \) identity
axiom cos_half_angle_formula (A : ℝ):
  cos (A / 2) = sqrt ((1 + cos A) / 2)

-- Assume relationship between sides using Law of Sines
axiom law_of_sines :
  ∀ {a b c A B C : ℝ}, (a / sin A = b / sin B) ∧ (b / sin B = c / sin C)

-- Assume relationships between side lengths using addition
axiom sine_values (a b c : ℝ) :
  sin 60 = sqrt (3) / 2 ∧ sin 40 = real.sin (40 * real.pi / 180) ∧ sin 80 = real.sin (80 * real.pi / 180)

theorem triangle_cos_half_angle (x a b c : ℝ) (h1 : A = 2 * x)
  (h2 : B = 3 * x) (h3 : C = 4 * x) (h4: A + B + C = 180) :
  cos (A/2) = (a + c) / (2 * b) :=
by
  sorry

end triangle_cos_half_angle_l647_647963


namespace find_a1_l647_647900

theorem find_a1 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (1 - a n)) (h2 : a 8 = 2)
: a 1 = 1 / 2 :=
sorry

end find_a1_l647_647900


namespace sum_of_series_l647_647128

def imaginary_unit : ℂ := Complex.i

theorem sum_of_series :
  1 - imaginary_unit + imaginary_unit^2 - imaginary_unit^3 + imaginary_unit^4 
  - imaginary_unit^5 + imaginary_unit^6 - imaginary_unit^7 + imaginary_unit^8 
  - imaginary_unit^9 + imaginary_unit^10 - imaginary_unit^11 + imaginary_unit^12 
  - imaginary_unit^13 + imaginary_unit^14 - imaginary_unit^15 + imaginary_unit^16 
  - imaginary_unit^17 + imaginary_unit^18 - imaginary_unit^19 + imaginary_unit^20 = 1 :=
sorry

end sum_of_series_l647_647128


namespace find_b_and_c_range_of_b_l647_647548

-- Define the function f(x)
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  - (1 / 3) * x ^ 3 + b * x ^ 2 + c * x + b * c

-- Define the conditions
def is_extreme_at_one (f : ℝ → ℝ) (val : ℝ) : Prop :=
  f 1 = val ∧ ∀ f' x_0, x_0 = 1 → deriv f x_0 = 0

-- Conditions for the tangent line slope
def slope_condition (f : ℝ → ℝ) (c : ℝ) (b : ℝ) : Prop :=
  ∀ x, (1 / 2 < x ∧ x < 3) → deriv (λ x, f x - c * (x + b)) x ≤ 2

-- Problem statement proving values of b and c
theorem find_b_and_c :
  ∃ b c : ℝ, is_extreme_at_one (f · b c) (-4 / 3) ∧ slope_condition (f · b c) c b := sorry

-- Problem statement proving the range of b
theorem range_of_b : ∀ b : ℝ, slope_condition (f · b 3) 3 b → b ≤ Real.sqrt 2 := sorry

end find_b_and_c_range_of_b_l647_647548


namespace equation_of_ellipse_range_of_eccentricity_l647_647135

variable {m : ℝ}

-- Condition of the foci and directrixes
variable (right_focus : m ≠ 0)
variable (left_directrix : ∀ x, x = -m - 1)
variable (right_directrix : ∀ x, x = m + 1)

-- Point of intersection of directrixes with y = x
def point_A := (-(m + 1), -(m + 1))
def point_B := (m + 1, m + 1)

-- Definition of vectors AF and FB
def vector_AF := ((2 * m) + 1, m + 1)
def vector_FB := (1, m + 1)

-- Defining the dot product
def dot_product := (2 * m + 1) * 1 + (m + 1) ^ 2

-- Theorem 1: Equation of the ellipse
theorem equation_of_ellipse (eccentricity : ℝ) 
  (h1 : eccentricity = (Real.sqrt 2 / 2)) : 
  (m = 1) →
  ∀ x y : ℝ, (x ^ 2) / 2 + y ^ 2 = 1 := sorry

-- Theorem 2: Range of eccentricity
theorem range_of_eccentricity 
  (h2 : dot_product vector_AF vector_FB < 7) :
  ∃ e : ℝ, (e = m / (Real.sqrt (m * (m + 1)))) ∧ (0 < e) ∧ (e < Real.sqrt 2 / 2) := sorry

end equation_of_ellipse_range_of_eccentricity_l647_647135


namespace point_P_coordinates_l647_647171

def f (x : ℝ) (a : ℝ) : ℝ := 5 + a^(x - 1)

theorem point_P_coordinates {a : ℝ} (h : ∀ x, f x a = 5 + a^(x-1))
  : f 1 a = 6 :=
by
  sorry

end point_P_coordinates_l647_647171


namespace roman_numeral_sketching_l647_647796

/-- Roman numeral sketching problem. -/
theorem roman_numeral_sketching (n : ℕ) (k : ℕ) (students : ℕ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ i / 1 = i) 
  (h2 : ∀ i : ℕ, i > n → i = n - (i - n)) 
  (h3 : k = 7) 
  (h4 : ∀ r : ℕ, r = (k * n)) : students = 350 :=
by
  sorry

end roman_numeral_sketching_l647_647796


namespace f_at_919_l647_647878

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

noncomputable def f_definition : Prop := 
  even_function f ∧ periodic_function f 6 ∧ (∀ x : ℝ, x ∈ set.Icc (-3 : ℝ) 0 → f x = 6^(-x))

theorem f_at_919 : f_definition f → f 919 = 6 :=
by
  intro hf
  sorry

end f_at_919_l647_647878


namespace batsman_average_after_17th_inning_l647_647742

theorem batsman_average_after_17th_inning (A : ℝ) :
  (∀ A, (16 * A + 92 = 17 * (A + 3)) → A + 3 = 44) :=
begin
  intro,
  sorry
end

end batsman_average_after_17th_inning_l647_647742


namespace monotonicity_f_exp_sum_lt_4_l647_647550

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - x

theorem monotonicity_f (x : ℝ) (h : x < Real.log 2) : 
  (∀ y z : ℝ, y < z ∧ z < 0 → f y < f z) ∧ (∀ y z : ℝ, y < z ∧ 0 < z ∧ z < Real.log 2 → f y > f z) := 
sorry

theorem exp_sum_lt_4 (x1 x2 : ℝ) (h1 : x1 < Real.log 2) (h2 : x2 > Real.log 2) (h3 : (λ x, Real.exp x - 2*x - 1) x1 = (λ x, Real.exp x - 2*x - 1) x2) :
  Real.exp (x1 + x2) < 4 := 
sorry

end monotonicity_f_exp_sum_lt_4_l647_647550


namespace product_of_two_numbers_l647_647308

theorem product_of_two_numbers (x y : ℝ) (h₁ : x + y = 23) (h₂ : x^2 + y^2 = 289) : x * y = 120 := by
  sorry

end product_of_two_numbers_l647_647308


namespace range_of_a_l647_647141

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x ^ 2

theorem range_of_a {a : ℝ} :
  (∃ x ∈ Set.Icc 1 (Real.exp 2), f a x = 0) ∧
  (∀ x y ∈ Set.Icc 1 (Real.exp 2), x ≠ y → f a x ≠ 0 ∨ f a y ≠ 0) ↔
  a ∈ (Set.Icc (-∞) (- e ^ 4 / 2) ∪ {-2 * Real.exp 1}) :=
  sorry

end range_of_a_l647_647141


namespace pizza_problem_l647_647911

theorem pizza_problem :
  ∃ (x : ℕ), x = 20 ∧ (3 * x ^ 2 = 3 * 14 ^ 2 * 2 + 49) :=
by
  let small_pizza_side := 14
  let large_pizza_cost := 20
  let pool_cost := 60
  let individually_cost := 30
  have total_individual_area := 2 * 3 * (small_pizza_side ^ 2)
  have extra_area := 49
  sorry

end pizza_problem_l647_647911


namespace min_instantaneous_rate_of_change_l647_647764

def f (x : ℝ) := (1 / 3) * x^3 - x^2 + 8

theorem min_instantaneous_rate_of_change : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 5 ∧ deriv f x = -1 :=
by
  sorry

end min_instantaneous_rate_of_change_l647_647764


namespace hcf_of_two_numbers_l647_647579

theorem hcf_of_two_numbers 
  (x y : ℕ) 
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1/x : ℚ) + (1/y : ℚ) = 11/120) : 
  Nat.gcd x y = 1 := 
sorry

end hcf_of_two_numbers_l647_647579


namespace ways_to_sum_420_l647_647186

theorem ways_to_sum_420 : 
  (∃ n k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 > 0 ∧ n * (2 * k + n - 1) = 840) → (∃ c, c = 11) :=
by
  sorry

end ways_to_sum_420_l647_647186


namespace number_of_valid_subsets_l647_647909

-- Define the set of numbers
def S : Finset ℕ := {90, 96, 100, 133, 167, 174}

-- Function to get subsets of size 3
def subsetsOfSize3 (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ t, t.card = 3)

-- Function to determine if a set of 3 numbers has an odd sum
def isSumOdd (s : Finset ℕ) : Prop := (s.sum id) % 2 = 1

-- Main theorem statement
theorem number_of_valid_subsets : (subsetsOfSize3 S).filter isSumOdd.card = 10 := sorry

end number_of_valid_subsets_l647_647909


namespace max_profit_l647_647256

theorem max_profit :
  ∀ (t : ℝ), 
  let x := 3 - (2 / (t + 1))
      fixed_expenses := 3
      purchase_price := 32
      selling_price := (3 / 2) * purchase_price + (t / (2 * x))
      profit := selling_price * x - (purchase_price * x + fixed_expenses + t)
  in profit ≤ 37.5 :=
by sorry

end max_profit_l647_647256


namespace olympics_event_arrangements_l647_647286

theorem olympics_event_arrangements :
  let n_competitive_arrangements := 2
  let scenarios_one := (Nat.choose 5 3) * Nat.factorial 3 / Nat.factorial 1
  let scenarios_two := (Nat.choose 5 1) * (Nat.choose 3 1) * (Nat.choose 4 2)
  let total_exhibition_arrangements := scenarios_one + scenarios_two
  let total_arrangements := n_competitive_arrangements * total_exhibition_arrangements
  in total_arrangements = 300 :=
by
  sorry

end olympics_event_arrangements_l647_647286


namespace transformed_inequality_solution_l647_647525

variable {a b c d : ℝ}

theorem transformed_inequality_solution (H : ∀ x : ℝ, ((-1 < x ∧ x < -1/3) ∨ (1/2 < x ∧ x < 1)) → 
  (b / (x + a) + (x + d) / (x + c) < 0)) :
  ∀ x : ℝ, ((1 < x ∧ x < 3) ∨ (-2 < x ∧ x < -1)) ↔ (bx / (ax - 1) + (dx - 1) / (cx - 1) < 0) :=
sorry

end transformed_inequality_solution_l647_647525


namespace arithmetic_sequence_geometric_sequence_l647_647360

-- Problem 1
theorem arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) (Sₙ : ℝ) 
  (h₁ : a₁ = 3 / 2) (h₂ : d = -1 / 2) (h₃ : Sₙ = -15) :
  n = 12 ∧ (a₁ + (n - 1) * d) = -4 := 
sorry

-- Problem 2
theorem geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) (aₙ Sₙ : ℝ) 
  (h₁ : q = 2) (h₂ : aₙ = 96) (h₃ : Sₙ = 189) :
  a₁ = 3 ∧ n = 6 := 
sorry

end arithmetic_sequence_geometric_sequence_l647_647360


namespace parallel_and_distinct_condition_l647_647876

theorem parallel_and_distinct_condition (a : ℝ) :
  (a = 3) ↔
  (let line1 := (λ x y : ℝ, a * x + 2 * y + 3 * a = 0),
       line2 := (λ x y : ℝ, 3 * x + (a - 1) * y = a - 7) in
   (∀ x1 y1 x2 y2, line1 x1 y1 → line2 x2 y2 → (-a/2 = -3/(a-1)) ∧ (let y_int_line1 := -(3*a)/2 in let y_int_line2 := -(a-7)/(a-1) in y_int_line1 ≠ y_int_line2))) := sorry

end parallel_and_distinct_condition_l647_647876


namespace eccentricity_range_of_trajectory_l647_647089

/--
For any point P on the ellipse C: x^2/3 + y^2/2 = 1,
draw a perpendicular line PH from P to the right directrix of the ellipse C 
(where H is the foot of the perpendicular).
Extend PH to point Q such that |HQ| = λ|PH| (λ ≥ 1).
When point P moves on the ellipse C, the range of the eccentricity of the trajectory of point Q is (√3/3, 1).
-/
theorem eccentricity_range_of_trajectory (P : ℝ × ℝ)
    (ellipse : ∀ x y : ℝ, x^2/3 + y^2/2 = 1)
    (directrix : ℝ → Prop)
    (lambda : ℝ)
    (h1 : lambda ≥ 1) :
    (∃ e : ℝ, (√3/3) < e ∧ e < 1) :=
begin
  sorry
end

end eccentricity_range_of_trajectory_l647_647089


namespace lists_count_2048_l647_647230

theorem lists_count_2048 :
  ∀ (b : Fin 13 → ℕ), 
  (∀ i : Fin 12, b (i + 1) < b i → (b (i + 1) + 1 = b i ∨ b (i + 1) - 1 = b i)) → 
  (∀ i : Fin 12, 
    (∃ k : ℕ, k^2 = b (i + 1)) → 
    (b (i + 1) + 1 = b i ∨ b (i + 1) - 1 = b i) → 
    (∃ j : Fin 12, b (i + 1) = b j + 1 ∨ b (i + 1) = b j - 1)) → 
  fintype.card { l : Fin 13 → ℕ // 
    (∀ i : Fin 12, l (i + 1) < l i → (l (i + 1) + 1 = l i ∨ l (i + 1) - 1 = l i)) ∧ 
    (∀ i : Fin 12, 
      (∃ k : ℕ, k^2 = l (i + 1)) → 
      (l (i + 1) + 1 = l i ∨ l (i + 1) - 1 = l i) → 
      (∃ j : Fin 12, l (i + 1) = l j + 1 ∨ l (i + 1) = l j - 1)) } = 2048 :=
sorry

end lists_count_2048_l647_647230


namespace isosceles_triangle_k_value_l647_647702

theorem isosceles_triangle_k_value 
(side1 : ℝ)
(side2 side3 : ℝ)
(k : ℝ)
(h1 : side1 = 3 ∨ side2 = 3 ∨ side3 = 3)
(h2 : side1 = side2 ∨ side1 = side3 ∨ side2 = side3)
(h3 : Polynomial.eval side1 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side2 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side3 (Polynomial.C k + Polynomial.X ^ 2) = 0) :
k = 3 ∨ k = 4 :=
sorry

end isosceles_triangle_k_value_l647_647702


namespace circle_area_from_circumference_l647_647368

theorem circle_area_from_circumference (C : ℝ) (hC : C = 48 * Real.pi) : 
  ∃ m : ℝ, (∀ r : ℝ, C = 2 * Real.pi * r → (Real.pi * r^2 = m * Real.pi)) ∧ m = 576 :=
by
  sorry

end circle_area_from_circumference_l647_647368


namespace find_a_b_l647_647580

def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b :
  ∀ (a b : ℝ),
  (∀ x, (curve x a b) = x^2 + a * x + b) →
  (tangent_line 0 (curve 0 a b)) →
  (tangent_line x y → y = x + 1) →
  (tangent_line x y → ∃ m c, y = m * x + c ∧ m = 1 ∧ c = 1) →
  (∃ a b : ℝ, a = 1 ∧ b = 1) :=
by
  intros a b h_curve h_tangent_line h_tangent_line_form h_tangent_line_eq
  sorry

end find_a_b_l647_647580


namespace problem_statement_l647_647069

-- Define conditions and the main theorem
theorem problem_statement (n : ℕ) (k : ℕ)
  (h1 : 2^k ≤ n) (h2 : n < 2^(k+1)) :
  (nat.floor (n / 2^0) * nat.floor (n / 2^1) *
   nat.floor (n / 2^2) * ··· * nat.floor (n / 2^k) +
   2 * 4^(nat.floor (k / 2))) = t^2 :=
sorry

end problem_statement_l647_647069


namespace third_team_pieces_l647_647855

theorem third_team_pieces (total_pieces : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) : 
  total_pieces = 500 → first_team = 189 → second_team = 131 → third_team = total_pieces - first_team - second_team → third_team = 180 :=
by
  intros h_total h_first h_second h_third
  rw [h_total, h_first, h_second] at h_third
  exact h_third

end third_team_pieces_l647_647855


namespace reciprocal_geometric_sum_l647_647038

variable (n : ℕ) (r s : ℝ)
variable (h_r_nonzero : r ≠ 0)
variable (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3)

theorem reciprocal_geometric_sum (n : ℕ) (r s : ℝ) (h_r_nonzero : r ≠ 0)
  (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3) :
  ((1 - (1 / r^2)^n) / (1 - 1 / r^2)) = s^3 / r^2 :=
sorry

end reciprocal_geometric_sum_l647_647038


namespace log_ordering_l647_647126

theorem log_ordering 
  (a b c : ℝ) 
  (ha: a = Real.log 3 / Real.log 2) 
  (hb: b = Real.log 2 / Real.log 3) 
  (hc: c = Real.log 0.5 / Real.log 10) : 
  a > b ∧ b > c := 
by 
  sorry

end log_ordering_l647_647126


namespace no_representation_of_expr_l647_647968

theorem no_representation_of_expr :
  ¬ ∃ f g : ℝ → ℝ, (∀ x y : ℝ, 1 + x ^ 2016 * y ^ 2016 = f x * g y) :=
by
  sorry

end no_representation_of_expr_l647_647968


namespace boys_love_marbles_l647_647178

theorem boys_love_marbles (total_marbles : ℕ) (marbles_per_boy : ℕ) (h_marble_count : total_marbles = 35) (h_marbles_per_boy : marbles_per_boy = 7) : total_marbles / marbles_per_boy = 5 :=
by {
  rw [h_marble_count, h_marbles_per_boy],
  norm_num,
  sorry
}

end boys_love_marbles_l647_647178


namespace max_ratio_is_2_l647_647057

noncomputable def maximum_ratio 
  (A B C D : ℚ × ℚ) -- Note: ℚ is used here to represent rational coordinates which can be specialized later
  (hA : A.1^2 + A.2^2 = 16)
  (hB : B.1^2 + B.2^2 = 16)
  (hC : C.1^2 + C.2^2 = 16)
  (hD : D.1^2 + D.2^2 = 16)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_irrat_AB : ∃ (a b : ℚ), a^2 + b^2 ∈ {16,25,32,40,50,64,80,100,125,128,144} ∧ AB = a^2 + b^2)
  (h_irrat_CD : ∃ (c d : ℚ), c^2 + d^2 ∈ {16,25,32,40,50,64,80,100,125,128,144} ∧ CD = c^2 + d^2) : 
  ℚ := 2

-- Using sorry to skip the proof
theorem max_ratio_is_2 : 
  ∀ (A B C D : ℚ × ℚ)
  (hA : A.1^2 + A.2^2 = 16)
  (hB : B.1^2 + B.2^2 = 16)
  (hC : C.1^2 + C.2^2 = 16)
  (hD : D.1^2 + D.2^2 = 16)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_irrat_AB : ∃ (a b : ℚ), a^2 + b^2 ∈ {16,25,32,40,50,64,80,100,125,128,144} ∧ AB = a^2 + b^2)
  (h_irrat_CD : ∃ (c d : ℚ), c^2 + d^2 ∈ {16,25,32,40,50,64,80,100,125,128,144} ∧ CD = c^2 + d^2), 
  maximum_ratio A B C D hA hB hC hD h_distinct h_irrat_AB h_irrat_CD = 2 :=
sorry

end max_ratio_is_2_l647_647057


namespace inequality_for_pos_reals_l647_647679

-- Definitions for positive real numbers
variables {x y : ℝ}
def is_pos_real (x : ℝ) : Prop := x > 0

-- Theorem statement
theorem inequality_for_pos_reals (hx : is_pos_real x) (hy : is_pos_real y) : 
  2 * (x^2 + y^2) ≥ (x + y)^2 :=
by
  sorry

end inequality_for_pos_reals_l647_647679


namespace minimum_value_expression_l647_647524

noncomputable def min_value_expression (n : ℕ) (a : Fin n → ℝ) [decidable_eq (Fin n)] : ℝ :=
a 0 * (∑ i in range (n + 1), (∏ j in range i, a j))

theorem minimum_value_expression (n : ℕ) (a : Fin n → ℝ) (h1 : 2 ≤ n) 
(h2 : ∀ i j, i ≤ j → a i ≥ a j) (h3 : (∑ i, a i) = n) : 
min_value_expression n a = n :=
sorry

end minimum_value_expression_l647_647524


namespace sum_excluded_values_domain_l647_647843

theorem sum_excluded_values_domain (x : ℝ) :
  (3 * x^2 - 9 * x + 6 = 0) → (x = 1 ∨ x = 2) ∧ (1 + 2 = 3) :=
by {
  -- given that 3x² - 9x + 6 = 0, we need to show that x = 1 or x = 2, and that their sum is 3
  sorry
}

end sum_excluded_values_domain_l647_647843


namespace incenter_chords_equal_l647_647443

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry Triangle

/-- Given a triangle ABC and a point M inside it,
  if circles are constructed on segments MA, MB, and MC as diameters,
  then the point M such that the lengths of common chords are equal, 
  is the incenter of triangle ABC. -/
theorem incenter_chords_equal (A B C M : α) (hM : incircle M (Triangle.mk A B C))
  (h_chord_eq : ∀ (P Q R : α), is_common_chord P (circle (segment.mk M A))
    (circle (segment.mk M B)) = is_common_chord Q (circle (segment.mk M B)) 
    (circle (segment.mk M C)) = is_common_chord R (circle (segment.mk M C))
    (circle (segment.mk M A))) : is_incenter M (Triangle.mk A B C) :=
sorry

end incenter_chords_equal_l647_647443


namespace function_characterization_l647_647046

theorem function_characterization (f : ℤ → ℤ)
  (h : ∀ a b : ℤ, ∃ k : ℤ, f (f a - b) + b * f (2 * a) = k ^ 2) :
  (∀ n : ℤ, (n % 2 = 0 → f n = 0) ∧ (n % 2 ≠ 0 → ∃ k: ℤ, f n = k ^ 2))
  ∨ (∀ n : ℤ, ∃ k: ℤ, f n = k ^ 2 ∧ k = n) :=
sorry

end function_characterization_l647_647046


namespace radius_of_tangent_circle_l647_647793

noncomputable def ellipse_tangent_circle_radius : ℝ :=
  let a := 6
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  let focus_x := -c
  let ellipse_eq (x y : ℝ) := (x^2 / 36) + (y^2 / 25) = 1
  let circle_eq (x y r : ℝ) := (x + c)^2 + y^2 = r^2
  radius : ℝ
  := c + 6

theorem radius_of_tangent_circle
  (radius : ℝ)
  (a b : ℝ)
  (c := Real.sqrt (a^2 - b^2))
  (focus_x := -c)
  (ellipse_eq : ∀ x y, (x^2 / 36) + (y^2 / 25) = 1)
  (circle_eq : ∀ x y r, (x + c)^2 + y^2 = r^2)
  (h_ellipse : ellipse_eq (-focus_x + radius) 0)
  : radius = Real.sqrt 11 + 6 := by
  sorry

end radius_of_tangent_circle_l647_647793


namespace find_b_l647_647583

theorem find_b
  (a : ℝ) (B : ℝ) (A : ℝ)
  (h_a : a = 10)
  (h_B : B = 60)
  (h_A : A = 45)
  (h1 : b = a * (sin (real.to_real 60) / sin (real.to_real 45))) :
  b = 5*sqrt 6 :=
by
  rw [h_a, h_B, h_A] at h1
  exact h1

end find_b_l647_647583


namespace binary_to_octal_equivalence_l647_647736

theorem binary_to_octal_equivalence : (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) 
                                    = (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end binary_to_octal_equivalence_l647_647736


namespace number_of_excellent_students_l647_647768

theorem number_of_excellent_students (μ δ : ℝ) (n : ℕ) (P_60 : ℝ) (X : ℝ → ℝ) 
  (hX : ∀ x, X x = (1 / (δ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * δ^2))) 
  (hmean : μ = 90) 
  (hprob60 : P_60 = 0.1) 
  (hP60 : ∫ x in -∞..60, X x = P_60) 
  (hn_students : n = 1200) :
  ∃ count : ℕ, count = 120 :=
by
  have hP120 : ∫ x in 120..∞, X x = 0.1 := sorry
  have hcount : count = n * 0.1 := sorry
  use 120
  sorry

end number_of_excellent_students_l647_647768


namespace problem1_problem2_l647_647906

-- Definitions and problem conditions
def a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 1)
def b (θ : ℝ) : ℝ × ℝ := (1, Real.cos θ)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)
def angle_range (θ : ℝ) : Prop := - (Real.pi / 2) < θ ∧ θ < Real.pi / 2
def a_perpendicular_b (θ : ℝ) : Prop := dot_product (a θ) (b θ) = 0
def vector_sum (θ : ℝ) : ℝ × ℝ := (a θ).fst + (b θ).fst, (a θ).snd + (b θ).snd

-- Problem (I): Prove θ = -π/4 if a is perpendicular to b given the angle range condition
theorem problem1 (θ : ℝ) (h1 : angle_range θ) (h2 : a_perpendicular_b θ) : θ = -Real.pi / 4 := 
by 
  sorry

-- Problem (II): Prove the maximum value of |a + b| given the angle range condition
theorem problem2 (θ : ℝ) (h1 : angle_range θ) : 
  ∃ θ_max, (∀ θ, angle_range θ → magnitude (vector_sum θ) ≤ magnitude (vector_sum θ_max)) ∧ 
  magnitude (vector_sum θ_max) = Real.sqrt (2 + 2) :=
by 
  sorry

end problem1_problem2_l647_647906


namespace part_I_part_II_l647_647140

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * Real.log x

theorem part_I (h : f 1 (-2) = 1) : (∀ x > 0, has_deriv_at (λ x, x^2 + a * Real.log x) (2 * x + a / x) x) :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + 2 / x

theorem part_II (h_monotone : ∀ x ≥ 1, 2*x + a/x - 2/(x^2) ≥ 0) : ∀ a ≥ 0 :=
sorry

end part_I_part_II_l647_647140


namespace rectangle_area_probability_l647_647108

theorem rectangle_area_probability :
  let AB : ℝ := 12
  in let AC : ℝ := AC ∈ Ioo 0 AB
  in let CB : ℝ := AB - AC
  in let area : ℝ := AC * CB
  in (∃ AC, AC ∈ Ioo 0 12 ∧ (AC * (12 - AC) < 32) ∧ 
   ((Ioo 0 4).measure + (Ioo 8 12).measure) / (Ioo 0 12).measure = (2 / 3)) := 
sorry

end rectangle_area_probability_l647_647108


namespace cylinder_volume_l647_647504

noncomputable def volume_of_cylinder (d h : ℝ) : ℝ := 
  let r := d / 2
  let V := Real.pi * r^2 * h
  V

theorem cylinder_volume : volume_of_cylinder 10 15 ≈ 1178.0975 := by
  sorry

end cylinder_volume_l647_647504


namespace trader_profit_percentage_l647_647787

-- Given Definitions
def indicated_weight_supplier : ℝ := 1000
def overtake_percentage := 0.10
def actual_weight_supplier := indicated_weight_supplier * (1 + overtake_percentage) -- 1100

def weight_claimed : ℝ := 1000
def weight_actual_customer (x : ℝ) : Prop := x * 1.30 = weight_claimed -- x = 769.23

def profit (claimed : ℝ) (actual : ℝ) : ℝ := claimed - actual
def cost_price := indicated_weight_supplier 
def profit_percentage (profit : ℝ) (cost : ℝ) : ℝ := (profit / cost) * 100

-- Theorem Statement
theorem trader_profit_percentage : 
  ∃ x : ℝ, (weight_actual_customer x ∧ profit_percentage (profit weight_claimed x) cost_price = 23.08) :=
sorry

end trader_profit_percentage_l647_647787


namespace solve_for_x_l647_647680

theorem solve_for_x : 
  ∃ x : ℝ, (∃ y : ℝ, y = 14 ∧ y ^ 3 = 24 * x + (24 * x + 16) ^ (1 / 3)) ∧ 
    x = 114 :=
begin
  let x := 114,
  use x,
  let y := 14,
  use y,
  split,
  { exact rfl },
  { sorry }
end

end solve_for_x_l647_647680


namespace solve_inequality_l647_647685

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end solve_inequality_l647_647685


namespace f_2010_l647_647223

noncomputable def f : ℝ → ℝ := sorry

axiom f_def_positive (x : ℝ) (h : x > 0) : f x > 0
axiom f_def_symmetric (x y : ℝ) (h1 : x > y) (h2 : y > 0) : f (x - y) = sqrt (f (x * y) + 2)

theorem f_2010 : f 2010 = 2 := sorry

end f_2010_l647_647223


namespace sum_of_reciprocal_of_binomial_coefficients_l647_647628

theorem sum_of_reciprocal_of_binomial_coefficients (n : ℕ) :
  let a_n := Nat.choose (n + 1) 2 in
  (Finset.range n).sum (λ t, 1 / a_n) = (2 * n) / (n + 1) :=
by
  sorry

end sum_of_reciprocal_of_binomial_coefficients_l647_647628


namespace abs_diff_eq_expected_answer_l647_647468

def C : ℕ := 2 -- C_3 = 2_3
def D : ℕ := 0 -- D_3 = 0_3

-- Since the numbers are in base 3, we use 3 in place functions for operations.
def abs_diff_C_D : ℕ := @int.natAbs (_ : ℕ) (C - D)
def expected_answer : ℕ := 2

theorem abs_diff_eq_expected_answer : abs_diff_C_D = expected_answer := by
  sorry

end abs_diff_eq_expected_answer_l647_647468


namespace perpendicular_lines_slope_l647_647559

theorem perpendicular_lines_slope (a : ℝ)
  (h : (a * (a + 2)) = -1) : a = -1 :=
sorry

end perpendicular_lines_slope_l647_647559


namespace cos_alpha_add_beta_over_two_l647_647532

theorem cos_alpha_add_beta_over_two (
  α β : ℝ) 
  (h1 : 0 < α ∧ α < (Real.pi / 2)) 
  (h2 : - (Real.pi / 2) < β ∧ β < 0) 
  (hcos1 : Real.cos (α + (Real.pi / 4)) = 1 / 3) 
  (hcos2 : Real.cos ((β / 2) - (Real.pi / 4)) = Real.sqrt 3 / 3) : 
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_two_l647_647532


namespace function_monotonically_decreasing_interval_l647_647827

theorem function_monotonically_decreasing_interval :
  ∃ a b : ℝ,
    (-1 < a) ∧ (b < 11) ∧
    (∀ x : ℝ, a < x ∧ x < b → deriv (λ x : ℝ, x^3 - 15*x^2 - 33*x + 6) x < 0) :=
by
  -- Definitions identified from problem conditions
  let f := λ x : ℝ, x^3 - 15*x^2 - 33*x + 6
  let f' := λ x : ℝ, 3*(x + 1)*(x - 11)

  -- The proof would generally involve finding the derivative and analyzing its sign in the interval
  sorry

end function_monotonically_decreasing_interval_l647_647827


namespace zeros_between_decimal_point_and_first_nonzero_digit_l647_647344

theorem zeros_between_decimal_point_and_first_nonzero_digit :
  3125 = 5^5 →
  ∀ (n : ℕ), (∃ b : ℕ, 5 / 3125 = n * 10^b) →
  n = 16 → b = 4 → natDigits(10, 16) = [1, 6] →
  2 := 
by
  sorry

end zeros_between_decimal_point_and_first_nonzero_digit_l647_647344


namespace problem1_problem2_l647_647929

-- Definitions given in the conditions
variables (A B C : Real) -- angles
variables (a b c : Real) -- sides
variables (AB BC : Real) -- vectors

-- Conditions
axiom triangle_conditions : (b = √3) ∧ (vec_dot (AB - BC) (-3 / 2)) ∧ (B = π / 3) 
axiom arithmetic_sequence_angles: (2 * B) = (A + C)

-- Problem 1: Prove value of a + c
theorem problem1 : a + c = 2 * √3 :=
by
  rwa [triangle_conditions, arithmetic_sequence_angles]
  sorry

-- Problem 2: Prove range of 2 sin A - sin C
theorem problem2 : ∀ C, 0 < C ∧ C < 2 * π / 3 → -√3 / 2 < (2 * sin A - sin C) ∧ (2 * sin A - sin C) < √3 :=
by
  rw [arithmetic_sequence_angles, triangle_conditions]
  sorry

end problem1_problem2_l647_647929


namespace final_hair_length_is_14_l647_647645

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end final_hair_length_is_14_l647_647645


namespace sum_of_cubes_divisible_by_9_l647_647672

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l647_647672


namespace product_of_chords_lengths_l647_647621

def semicircle_radius : ℝ := 3
def arc_divisions : ℕ := 5

noncomputable
def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 10)

def endpoint_A : ℂ := semicircle_radius
def endpoint_B : ℂ := -semicircle_radius
def points_C (k : ℕ) (hk : k ∈ {1, 2, 3, 4}) : ℂ := semicircle_radius * omega^k

theorem product_of_chords_lengths 
  (k : ℕ)
  (hk : k ∈ {1, 2, 3, 4}) :
  (∏ k in {1, 2, 3, 4}, Complex.abs (endpoint_A - points_C k hk)) *
  (∏ k in {1, 2, 3, 4}, Complex.abs (endpoint_B - points_C k (by simpa using hk))) = 65610 := 
sorry

end product_of_chords_lengths_l647_647621


namespace min_max_difference_l647_647727

theorem min_max_difference (n : ℕ) (a : Fin n → ℝ) 
  (h_sum : (∑ i, a i) = 0) 
  (h_abs_sum : (∑ i, |a i|) = 1) : 
  (Finset.max' (Finset.univ.image a) (Finset.univ_nonempty.image a) - 
   Finset.min' (Finset.univ.image a) (Finset.univ_nonempty.image a)) 
  ≥ (2 / n) :=
sorry

end min_max_difference_l647_647727


namespace find_n_mod_10_l647_647480

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l647_647480


namespace union_A_B_intersection_complement_A_B_l647_647527

open Set Real

noncomputable def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}
noncomputable def B : Set ℝ := {x : ℝ | abs (2 * x + 1) ≤ 1}

theorem union_A_B : A ∪ B = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end union_A_B_intersection_complement_A_B_l647_647527


namespace p_necessary_for_q_l647_647861

def p (x : ℝ) := x ≠ 1
def q (x : ℝ) := x ≥ 2

theorem p_necessary_for_q : ∀ x, q x → p x :=
by
  intro x
  intro hqx
  rw [q] at hqx
  rw [p]
  sorry

end p_necessary_for_q_l647_647861


namespace problem_l647_647629

def g (n : ℕ) : ℝ := Real.log (2 ^ n) / Real.log 2

theorem problem (n : ℕ) : g(n) / Real.log 3 = n / Real.log 3 := by
  sorry

end problem_l647_647629


namespace trey_total_hours_l647_647659

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end trey_total_hours_l647_647659


namespace find_original_price_l647_647335

variable (original_price : ℝ)
variable (final_price : ℝ) (first_reduction_rate : ℝ) (second_reduction_rate : ℝ)

theorem find_original_price :
  final_price = 15000 →
  first_reduction_rate = 0.30 →
  second_reduction_rate = 0.40 →
  0.42 * original_price = final_price →
  original_price = 35714 := by
  intros h1 h2 h3 h4
  sorry

end find_original_price_l647_647335


namespace forum_posting_total_l647_647776

theorem forum_posting_total (num_members : ℕ) (num_answers_per_question : ℕ) (num_questions_per_hour : ℕ) (hours_per_day : ℕ) :
  num_members = 1000 ->
  num_answers_per_question = 5 ->
  num_questions_per_hour = 7 ->
  hours_per_day = 24 ->
  ((num_questions_per_hour * hours_per_day * num_members) + (num_answers_per_question * num_questions_per_hour * hours_per_day * num_members)) = 1008000 :=
by
  intros
  sorry

end forum_posting_total_l647_647776


namespace initial_non_electrified_part_l647_647021

variables (x y : ℝ)

def electrified_fraction : Prop :=
  x + y = 1 ∧ 2 * x + 0.75 * y = 1

theorem initial_non_electrified_part (h : electrified_fraction x y) : y = 4 / 5 :=
by {
  sorry
}

end initial_non_electrified_part_l647_647021


namespace smallest_integer_mk_exists_l647_647218

noncomputable def smallest_integer_mk (k : ℕ) : ℕ :=
  1 + (nat.factorial (k / 2)) * (nat.factorial ((k + 1) / 2))

theorem smallest_integer_mk_exists (k : ℕ) (h₁ : k > 0) :
  ∃ (Γ : polynomial ℤ), 
    (∀ x : ℤ, polynomial.eval x Γ = 1 → polynomial.eval x Γ = 0) ∧ 
    (∀ x : ℤ, polynomial.eval x Γ = smallest_integer_mk k → ∃ s : finset ℤ, s.card = k ∧ x ∈ s) :=
sorry

end smallest_integer_mk_exists_l647_647218


namespace find_rate_l647_647407

noncomputable def national_bank_interest_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ): ℚ :=
  (total_income - (investment_additional * additional_rate)) / investment_national

theorem find_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ) (correct_rate: ℚ):
  investment_national = 2400 → investment_additional = 600 → additional_rate = 0.10 → total_investment_rate = 0.06 → total_income = total_investment_rate * (investment_national + investment_additional) → correct_rate = 0.05 → national_bank_interest_rate total_income investment_national investment_additional additional_rate total_investment_rate = correct_rate :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end find_rate_l647_647407


namespace length_of_PS_l647_647960

theorem length_of_PS 
  (P Q R S T : Type)
  [inner_product_space ℝ P]
  [inner_product_space ℝ Q]
  [inner_product_space ℝ R]
  [inner_product_space ℝ S]
  [inner_product_space ℝ T]
  (PQ QR PR : ℝ)
  (h1 : PQ = 9)
  (h2 : QR = 12)
  (h3 : ∡ Q = 90)
  (PTS : inner_product_space.angle ℝ P T S = π / 2)
  (ST : ℝ)
  (h4 : ST = 6) :
  ∃ PS : ℝ, PS = 10 := 
sorry

end length_of_PS_l647_647960


namespace negation_correct_l647_647305

variable {α : Type*} (A B : Set α)

-- Define the original proposition
def original_proposition : Prop := A ∪ B = A → A ∩ B = B

-- Define the negation of the original proposition
def negation_proposition : Prop := A ∪ B ≠ A → A ∩ B ≠ B

-- State that the negation of the original proposition is equivalent to the negation proposition
theorem negation_correct : ¬(original_proposition A B) ↔ negation_proposition A B := by sorry

end negation_correct_l647_647305


namespace square_area_ratio_l647_647246

theorem square_area_ratio (n : ℕ) (s₁ s₂: ℕ) (h1 : s₁ = 1) (h2 : s₂ = n^2) (h3 : 2 * s₂ - 1 = 17) :
  s₂ = 81 := 
sorry

end square_area_ratio_l647_647246


namespace solve_inequality_l647_647684

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end solve_inequality_l647_647684


namespace parabola_line_intersection_l647_647541

theorem parabola_line_intersection (x1 x2 : ℝ) (h1 : x1 * x2 = 1) (h2 : x1 + 1 = 4) : x2 + 1 = 4 / 3 :=
by
  sorry

end parabola_line_intersection_l647_647541


namespace triangle_area_by_tangents_eq_l647_647834

-- Given the radius of each circle
variables (r : ℝ)

-- Define the area of the equilateral triangle formed by the common external tangents of the circles
def area_of_triangle_formed_by_tangents : ℝ :=
  2 * r^2 * (2 * real.sqrt 3 + 3)

-- Prove that given the conditions, the area is as defined
theorem triangle_area_by_tangents_eq :
  ∀ r : ℝ, r > 0 → 
  let area := area_of_triangle_formed_by_tangents r in 
  area = 2 * r^2 * (2 * real.sqrt 3 + 3) :=
by 
  intros r hrpos 
  let area := area_of_triangle_formed_by_tangents r
  sorry

end triangle_area_by_tangents_eq_l647_647834


namespace molecular_weight_K3AlC2O4_3_l647_647799

noncomputable def molecularWeightOfCompound : ℝ :=
  let potassium_weight : ℝ := 39.10
  let aluminum_weight  : ℝ := 26.98
  let carbon_weight    : ℝ := 12.01
  let oxygen_weight    : ℝ := 16.00
  let total_potassium_weight : ℝ := 3 * potassium_weight
  let total_aluminum_weight  : ℝ := aluminum_weight
  let total_carbon_weight    : ℝ := 3 * 2 * carbon_weight
  let total_oxygen_weight    : ℝ := 3 * 4 * oxygen_weight
  total_potassium_weight + total_aluminum_weight + total_carbon_weight + total_oxygen_weight

theorem molecular_weight_K3AlC2O4_3 : molecularWeightOfCompound = 408.34 := by
  sorry

end molecular_weight_K3AlC2O4_3_l647_647799


namespace smallest_integer_equal_costs_l647_647328

-- Definitions based directly on conditions
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum * 2

def binary_cost (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- The main statement to prove
theorem smallest_integer_equal_costs : ∃ n : ℕ, n < 2000 ∧ decimal_cost n = binary_cost n ∧ n = 255 :=
by 
  sorry

end smallest_integer_equal_costs_l647_647328


namespace product_mnp_l647_647039

variable (a b x y : ℤ)
variable (m n p : ℤ)

/-- The given equation condition for a^8xy - a^7y - a^6x = a^5 (b^5 - 1) -/
def given_equation := a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1)

/-- The equivalent equation condition (a^mx - a^n)(a^py - a^3) = a^5b^5 -/
def equivalent_equation := (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5

/-- The problem is to prove that mnp = 0 -/
theorem product_mnp (h1 : given_equation a b x y)
                    (h2 : equivalent_equation a b x y m n p) : 
                    m * n * p = 0 := 
sorry

end product_mnp_l647_647039


namespace no_integer_distances_in_rectangle_l647_647263

theorem no_integer_distances_in_rectangle (A B : ℕ) (hA : A % 2 = 1) (hB : B % 2 = 1) :
  ¬ ∃ (P : ℝ × ℝ),
    (∃ (d1 d2 d3 d4 : ℤ),
      (d1 : ℝ) = real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
      (d2 : ℝ) = real.sqrt ((P.1 - A)^2 + (P.2 - 0)^2) ∧
      (d3 : ℝ) = real.sqrt ((P.1 - 0)^2 + (P.2 - B)^2) ∧
      (d4 : ℝ) = real.sqrt ((P.1 - A)^2 + (P.2 - B)^2)) :=
sorry

end no_integer_distances_in_rectangle_l647_647263


namespace min_entries_altered_to_equal_row_col_sums_l647_647422

def matrix : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 2, 3],
  ![4, 5, 6],
  ![7, 8, 9]
]

theorem min_entries_altered_to_equal_row_col_sums : 
  ∃ (A' : Matrix (Fin 3) (Fin 3) ℕ), (∃ i j, (rowSum A' i = colSum A' j) ∧ (altered_entries A A' = 1)) := 
by 
  sorry

end min_entries_altered_to_equal_row_col_sums_l647_647422


namespace quadrant_of_angle_l647_647161

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ n : ℤ, n = 1 ∧ α = (n * π + π / 2) :=
sorry

end quadrant_of_angle_l647_647161


namespace find_f8_plus_f9_l647_647027

noncomputable def f : ℝ → ℝ := sorry

-- Definitions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def cond1 := is_odd f
def cond2 := is_even (λ x, f(x + 2))
def cond3 := f(1) = 1
def periodic8 := ∀ x, f(x + 8) = f x

-- Question to prove
theorem find_f8_plus_f9
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3)
  (h4 : periodic8) :
  f 8 + f 9 = 1 :=
sorry

end find_f8_plus_f9_l647_647027


namespace trey_total_hours_l647_647658

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end trey_total_hours_l647_647658


namespace determine_a_range_l647_647557

-- Define the two sets A and B
def A : set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 4/5}
def B (a : ℝ) : set (ℝ × ℝ) := {p | abs (p.1 - 1) + 2 * abs (p.2 - 2) ≤ a}

-- Define the condition where A is a subset of B
def A_subset_B (a : ℝ) : Prop := A ⊆ B a

-- Main theorem statement (no proof)
theorem determine_a_range (a : ℝ) : A_subset_B a → a ≥ 2 :=
by sorry

end determine_a_range_l647_647557


namespace graph_inequality_l647_647515

variable (n : ℕ)
variable (N : ℕ → ℕ)

axiom no_three_points_collinear (points : Fin n → Point) :
  ∀ (a b c : Fin n), a ≠ b → b ≠ c → c ≠ a → ¬Collinear (points a) (points b) (points c)

theorem graph_inequality (k : ℕ) (hk : 2 ≤ k ∧ k < n) (hnk : N k ≠ 0) :
  N (k+1) / N k ≥ (1 / (k^2 - 1)) * ((k^2 * N k / N (k+1)) - n) :=
sorry

end graph_inequality_l647_647515


namespace maximum_X_placement_l647_647417

def is_X (grid : ℕ → ℕ → Prop) (i j : ℕ) : Prop :=
  grid i j

def valid_placement (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, is_X grid i j → (i < 5 ∧ j < 5) ∧
  (∀ k, 0 ≤ k ∧ k < 5 → ¬(is_X grid k j ∧ is_X grid (k+1) j ∧ is_X grid (k+2) j ∧ is_X grid (k+3) j)) ∧
  (∀ k, 0 ≤ k ∧ k < 5 → ¬(is_X grid i k ∧ is_X grid i (k+1) ∧ is_X grid i (k+2) ∧ is_X grid i (k+3))) ∧
  (∀ d, -4 ≤ d ∧ d ≤ 4 → ¬(is_X grid (i+d) (j+d) ∧ is_X grid (i+d+1) (j+d+1) ∧ is_X grid (i+d+2) (j+d+2) ∧ is_X grid (i+d+3) (j+d+3))) ∧
  (∀ d, -4 ≤ d ∧ d ≤ 4 → ¬(is_X grid (i-d) (j+d) ∧ is_X grid (i-d+1) (j+d+1) ∧ is_X grid (i-d+2) (j+d+2) ∧ is_X grid (i-d+3) (j+d+3)))

def count_X (grid : ℕ → ℕ → Prop) : ℕ :=
  Finset.card (Finset.univ.filter (λ ⟨i, j⟩, is_X grid i j))

theorem maximum_X_placement : 
  ∃ grid : ℕ → ℕ → Prop, valid_placement grid ∧ count_X grid = 13 :=
by {
  -- Proof omitted
  sorry
}

end maximum_X_placement_l647_647417


namespace employee_discount_percentage_l647_647385

def wholesale_cost : ℝ := 200
def retail_markup : ℝ := 0.20
def employee_paid_price : ℝ := 228

theorem employee_discount_percentage :
  let retail_price := wholesale_cost * (1 + retail_markup)
  let discount := retail_price - employee_paid_price
  (discount / retail_price) * 100 = 5 := by
  sorry

end employee_discount_percentage_l647_647385


namespace fixed_monthly_fee_l647_647511

def FebruaryBill (x y : ℝ) : Prop := x + y = 18.72
def MarchBill (x y : ℝ) : Prop := x + 3 * y = 28.08

theorem fixed_monthly_fee (x y : ℝ) (h1 : FebruaryBill x y) (h2 : MarchBill x y) : x = 14.04 :=
by 
  sorry

end fixed_monthly_fee_l647_647511


namespace circle_tangent_diameter_AB_CD_parallel_l647_647863

variable (A B C D : Type) [convex_quadrilateral A B C D]

theorem circle_tangent_diameter_AB_CD_parallel
  (h1 : convex_quadrilateral A B C D)
  (h2 : tangent_circle (circle_diameter A B) (line_through C D)) :
  (tangent_circle (circle_diameter C D) (line_through A B)) ↔ (parallel_lines (line_through B C) (line_through A D)) :=
sorry

end circle_tangent_diameter_AB_CD_parallel_l647_647863


namespace initial_men_employed_l647_647026

-- Definitions
def L : ℝ := 15 -- Total road length
def D : ℝ := 300 -- Total days
def initial_completion_distance : ℝ := 2.5 -- Distance completed after 100 days
def initial_days : ℝ := 100 -- Days after which 2.5 km is completed
def remaining_days : ℝ := D - initial_days -- Days remaining, 200 in this case.
def remaining_distance : ℝ := L - initial_completion_distance -- Remaining distance, 12.5 km
def extra_men : ℝ := 75 -- Extra men employed

-- Daily rates
def initial_rate (M : ℝ) : ℝ := initial_completion_distance / initial_days -- Rate with initial men
def required_rate : ℝ := remaining_distance / remaining_days -- Required rate to finish the project

-- Proof statement
theorem initial_men_employed : ∃ M : ℝ, initial_rate M / required_rate = M / (M + extra_men) ∧ M = 50 :=
by
  sorry

end initial_men_employed_l647_647026


namespace range_of_a_l647_647172

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ 3 * a - a^2) → (1 ≤ a ∧ a ≤ 2) :=
begin
  sorry
end

end range_of_a_l647_647172


namespace distinct_real_roots_l647_647071

noncomputable def a_values : Set ℝ :=
  {-1, (5 / 2) + (sqrt 13 / 2), (5 / 2) - (sqrt 13 / 2), 
   (7 / 2) + (sqrt 17 / 2), (7 / 2) - (sqrt 17 / 2)}

theorem distinct_real_roots :
  { a : ℝ | ∃ x : ℝ, (x^2 - 6 * x + 8 - a) * (x - a^2 + 6 * a - 8) = 0 }.card = 2 :=
  sorry

end distinct_real_roots_l647_647071


namespace find_AB_l647_647533

variables {A B C G I : Type} [Point : Type] (Triangle : Point → Point → Point → Type)

structure TriangleProps (P : Point) :=
(centroid : Point → Point → Point → Point)
(incenter : Point → Point → Point → Point)

variables [Triangle : TriangleProps Point] (G I : Point)
local notation "Triangle" P Q R := Triangle P Q R

-- Geometric conditions and lengths
axiom is_centroid : TriangleProps.centroid Triangle A B C = G
axiom is_incenter : TriangleProps.incenter Triangle A B C = I
axiom perpendicular_IC_IG : ⟪ I, C ⟫ ⟂ ⟪ I, G ⟫
axiom AC_two : dist A C = 2
axiom BC_three : dist B C = 3

-- Theorem statement
theorem find_AB : dist A B = 11/5 := by
  sorry

end find_AB_l647_647533


namespace cos_2gamma_leq_0_l647_647271

theorem cos_2gamma_leq_0 (α β γ : ℝ) (h : Real.sec α * Real.sec β + Real.tan α * Real.tan β = Real.tan γ) :
  Real.cos (2 * γ) ≤ 0 := 
sorry

end cos_2gamma_leq_0_l647_647271


namespace johns_profit_l647_647611

theorem johns_profit
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (profit : ℕ) :
  trees_chopped = 30 →
  planks_per_tree = 25 →
  planks_per_table = 15 →
  price_per_table = 300 →
  labor_cost = 3000 →
  profit = 12000 :=
begin
  sorry
end

end johns_profit_l647_647611


namespace cone_volume_l647_647136

-- Definitions of conditions
def slant_height := 2
def lateral_surface_area := 2 * Real.pi

-- Volume of a cone with given conditions
theorem cone_volume :
  (∃ (r h : ℝ), slant_height ^ 2 = h ^ 2 + r ^ 2 ∧ lateral_surface_area = Real.pi * r * slant_height ∧ h = Real.sqrt 3 ∧ r = 1) →
  ∃ V : ℝ, V = (Real.sqrt 3 * Real.pi) / 3 :=
sorry

end cone_volume_l647_647136


namespace seq_3044_appears_l647_647388

-- Define the initial sequence and the sequence rule.
def initial_seq : List ℕ := [1, 9, 8, 2]

def next_digit (seq : List ℕ) : ℕ :=
  (seq |>.takeRight 4 |>.sum) % 10

def generate_seq (n : ℕ) : List ℕ :=
  List.foldl (λ s _, s ++ [next_digit s]) initial_seq (List.range n)

-- Property to check if a smaller list appears as a consecutive subsequence in a larger list.
def appears_consecutively (subseq : List ℕ) (seq : List ℕ) : Prop :=
  ∃ i, List.take subseq.length (List.drop i seq) = subseq

-- The theorem statement.
theorem seq_3044_appears : ∃ n, appears_consecutively [3, 0, 4, 4] (generate_seq n) :=
sorry

end seq_3044_appears_l647_647388


namespace distinct_values_of_z_l647_647886

open Int

theorem distinct_values_of_z :
  ∀ (x y : ℤ), (100 ≤ x) ∧ (x ≤ 999) ∧ (100 ≤ y) ∧ (y ≤ 999) ∧
  (∃ (a b c : ℕ), x = 100 * a + 10 * b + c ∧ y = 100 * c + 10 * b + a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9) ->
  let z := |x - y| in
  ({ z | (∃ (a c : ℕ), z = 99 * |a - c| ∧ a ≤ 9 ∧ c ≤ 9) }.toFinset.card) = 10 :=
by
  sorry

end distinct_values_of_z_l647_647886


namespace y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l647_647064

-- Definitions of cost functions for travel agencies
def full_ticket_price : ℕ := 240

def y_A (x : ℕ) : ℕ := 120 * x + 240
def y_B (x : ℕ) : ℕ := 144 * x + 144

-- Prove functional relationships for y_A and y_B
theorem y_A_functional_relationship (x : ℕ) : y_A x = 120 * x + 240 :=
by sorry

theorem y_B_functional_relationship (x : ℕ) : y_B x = 144 * x + 144 :=
by sorry

-- Prove conditions for cost-effectiveness
theorem cost_effective_B (x : ℕ) : x < 4 → y_A x > y_B x :=
by sorry

theorem cost_effective_equal (x : ℕ) : x = 4 → y_A x = y_B x :=
by sorry

theorem cost_effective_A (x : ℕ) : x > 4 → y_A x < y_B x :=
by sorry

end y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l647_647064


namespace cracked_marbles_l647_647065

def marbles : List ℕ := [20, 22, 24, 26, 28, 30, 32, 35]

noncomputable def totalMarbles (l : List ℕ) : ℕ := l.sum

theorem cracked_marbles:
  ∃ (crackedBag : ℕ), crackedBag ∈ marbles ∧ 
  (∀ (jane george: List ℕ), 
    jane.length = 4 ∧ george.length = 3 ∧ 
    jane.sum = george.sum + 50 ∧ 
    (jane ++ george).perm (marbles.filter (λ m, m ≠ crackedBag)) → 
    crackedBag = 35) := 
begin
  sorry
end

end cracked_marbles_l647_647065


namespace number_of_feet_l647_647380

theorem number_of_feet (H C F : ℕ) (hH : H = 26) (hHeads : H + C = 48) : F = 140 :=
by
  have hC : C = 48 - 26, from calc
    C = 48 - H    : by linarith
    ... = 48 - 26 : by rw hH
  have hFeet : F = (H * 2) + (C * 4), from sorry -- Define the equation for F
  rw [hH, hC] at hFeet
  linarith

end number_of_feet_l647_647380


namespace concyclic_A_I_M_N_l647_647990

namespace Geometry

open EuclideanGeometry

variables {A B C I D E F M N : Point}
variables {triangle_ABC : Triangle}
variables {angle_bisectors : IsIncenter I A B C BIncenter CI}
variables {AD_perp_bisector : PerpendicularBisector AD}
variables {M_on_perp_bisector : OnLine M (PerpendicularBisector AD)}
variables {N_on_perp_bisector : OnLine N (PerpendicularBisector AD)}

theorem concyclic_A_I_M_N :
  InTriangle I A B C ↔
  AngleBisector A I B C ↔ 
  PerpendicularBisector AD I ↔ 
  IsOnLine M (PerpendicularBisector AD) ↔ 
  IsOnLine N (PerpendicularBisector AD) → 
  Concyclic A I M N :=
begin
  sorry,
end

end Geometry

end concyclic_A_I_M_N_l647_647990


namespace polynomial_B_value_l647_647056

theorem polynomial_B_value 
  (roots : Fin 5 → ℕ)
  (positive_roots : ∀ i, roots i > 0)
  (sum_roots : (∑ i, roots i) = 14)
  (P Q : ℤ)
  (polynomial : Polynominal ℤ := (X ^ 5 - 14 * X ^ 4 + P * X ^ 3 + B * X ^ 2 + Q * X + 48)) 
  (has_roots : ∀ i, polynomial.eval (roots i : ℤ) = 0) :
  B = 203 :=
sorry

end polynomial_B_value_l647_647056


namespace find_n_mod_10_l647_647474

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l647_647474


namespace range_of_t_l647_647894

noncomputable def f (x m : ℝ) : ℝ := x^2 + (2 * m - 1) * x - m * Real.log x

theorem range_of_t (m x t : ℝ) (hm : 2 < m ∧ m < 3) (hx : 1 ≤ x ∧ x ≤ 3) (ht : t ≤ 7 / 3) :
  mt - f x m < 1 :=
sorry

end range_of_t_l647_647894


namespace product_closest_to_point037_l647_647346

theorem product_closest_to_point037 : 
  (∃ x : ℝ, (x = 0.382) ∧ 
  ((x * x * x ≈ 0.037))) := by
  sorry

end product_closest_to_point037_l647_647346


namespace num_positive_S_n_l647_647637

def S (n : ℕ) : ℝ :=
  (List.range n).sum (λ k => Real.sin ((k+1) * Real.pi / 7))

theorem num_positive_S_n : ∃ (count : ℕ), count = 1731 ∧
  (List.range 2017).filter (λ n => S (n + 1) > 0).length = count :=
by
  sorry

end num_positive_S_n_l647_647637


namespace circle_radius_C1_l647_647416

-- Definitions for the given problem's conditions
def is_center (point_circle_1 point_circle_2 : Point) (center_circle_1 : Point) : Prop :=
  center_circle_1 = point_circle_2 ∧ point_circle_1 = center_circle_1

def circles_intersect_at (point_ref : Point) (point_inter_1 point_inter_2 : Point) : Prop :=
  point_inter_1 = point_ref ∧ point_inter_2 = point_ref

def point_on_circle_outside (point_outside point_center : Point) (radius_outside radius_center : ℝ) : Prop :=
  dist point_outside point_center = radius_outside ∧ dist point_center point_outside = radius_center

def distances (dist_xz dist_oz dist_yz: ℝ): Prop :=
  (dist_xz = 15) ∧ (dist_oz = 13) ∧ (dist_yz = 9)

-- Main theorem to prove
theorem circle_radius_C1 
  (O X Y Z : Point)
  (radius : ℝ)
  (h_center: is_center O O O)
  (h_intersect: circles_intersect_at O X Y)
  (h_point_outside: point_on_circle_outside Y O Z)
  (h_distances: distances 15 13 9)
  : radius = 11 :=
by
  sorry

end circle_radius_C1_l647_647416


namespace composition_of_homothety_l647_647264

-- Definitions of homothety and associated properties
structure Homothety (P : Type) [AddCommGroup P] [Module ℝ P] :=
(center : P)
(coefficient : ℝ)

-- Given homotheties H1 and H2
variables {P : Type} [AddCommGroup P] [Module ℝ P]
variables (H1 H2 : Homothety P)
variables (O1 O2 : P) (k1 k2 : ℝ)
variables (A B : P)

-- Assuming the coefficients
variables (h1 : H1 = ⟨O1, k1⟩) (h2 : H2 = ⟨O2, k2⟩)

noncomputable def composed_homothety := { H := H2.comp H1 }

-- Conditions for coefficients
variables (cond1 : k1 * k2 ≠ 1) (cond2 : k1 * k2 = 1)

-- Proof statement
theorem composition_of_homothety (H : Homothety P) :
  (k1 * k2 ≠ 1 → (H.center ∈ line_of_centers O1 O2 ∧ H.coefficient = k1 * k2)) ∧ 
  (k1 * k2 = 1 → is_parallel_translation H) := 
sorry

end composition_of_homothety_l647_647264


namespace collinear_HKI_l647_647181

-- Define the hexagon inscribed in a circle and the intersection points
variables (A B C D E F H K I : Type*)
variables [linear_ordered_field Type*] [add_comm_group Type*] [topological_space Type*]
variables [metric_space Type*] [normed_group Type*] [normed_space ℝ Type*]

-- Define the conditions:
-- A hexagon inscribed in a circle
-- Intersections of extensions of sides at points H, K, and I
axiom hexagon_inscribed_in_circle : inscribed_hexagon A B C D E F
axiom extensions_intersect :
  (line_through A B) ∩ (line_through D E) = H ∧
  (line_through B C) ∩ (line_through E F) = K ∧
  (line_through C D) ∩ (line_through F A) = I

-- Define the theorem stating that H, K, and I are collinear
theorem collinear_HKI :
  collinear_points H K I :=
sorry

end collinear_HKI_l647_647181


namespace machine_A_production_l647_647352

-- Definitions of the quantities and relationships
def sprockets_per_hour_A (A : ℝ) := A
def sprockets_per_hour_Q (A : ℝ) := 1.1 * A
def time_to_produce_Q (A : ℝ) := 990 / (1.1 * A)
def time_to_produce_P (A : ℝ) := (990 / (1.1 * A)) + 10

-- Main theorem to prove
theorem machine_A_production :
  ∃ (A : ℝ), (990 = A * time_to_produce_P A) → A = 9 :=
by
  -- provide the proof here
  sorry

end machine_A_production_l647_647352


namespace evacuation_safety_l647_647758

-- Define the conditions as Lean definitions
def main_door_per_min (x : ℕ) (y : ℕ) : Prop := 2 * (x + 2 * y) = 560
def side_door_per_min (x : ℕ) (y : ℕ) : Prop := 4 * (x + y) = 800
def max_students_in_building := 4 * 8 * 45
def emergency_efficiency_decrease := 0.20
def evacuation_time_in_minutes := 5

-- The goal: Prove the conditions lead to the correct evacuation requirements
theorem evacuation_safety (x y : ℕ) (h1 : main_door_per_min x y) (h2 : side_door_per_min x y) :
  x = 120 ∧ y = 80 ∧ 
  evacuation_time_in_minutes * 2 * (x + y) * (1 - emergency_efficiency_decrease) ≥ max_students_in_building :=
by {
  sorry -- The proof steps are skipped.
}

end evacuation_safety_l647_647758


namespace solve_for_x_l647_647353

theorem solve_for_x : ∃ x : ℝ, 25 * x = 675 ∧ x = 27 :=
by
  use 27
  split
  · exact (by ring : 25 * 27 = 675)
  · rfl

end solve_for_x_l647_647353


namespace remainder_of_sequence_mod_113_l647_647700

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ 
  a 2 = 7 ∧ 
  ∀ n > 2, a n = ∑ i in Finset.range (n-1), (-1 : ℤ)^(i + 1) * (i+1) * a (i+1)

theorem remainder_of_sequence_mod_113 :
  ∃ x : ℕ, (sequence a) ∧ (a x % 1090 = 0) ∧ (a x % 113 = 51) :=
sorry

end remainder_of_sequence_mod_113_l647_647700


namespace new_avg_weight_of_boxes_l647_647833

theorem new_avg_weight_of_boxes :
  ∀ (x y : ℕ), x + y = 30 → (10 * x + 20 * y) / 30 = 18 → (10 * x + 20 * (y - 18)) / 12 = 15 :=
by
  intro x y h1 h2
  sorry

end new_avg_weight_of_boxes_l647_647833


namespace sum_evaluation_l647_647801

theorem sum_evaluation : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 :=
by
  sorry

end sum_evaluation_l647_647801


namespace lines_intersect_at_l647_647901

theorem lines_intersect_at :
  ∃ t u : ℝ, (∃ (x y : ℝ),
    (x = 2 + 3 * t ∧ y = 4 - 2 * t) ∧
    (x = -1 + 6 * u ∧ y = 5 + u) ∧
    (x = 1/5 ∧ y = 26/5)) :=
by
  sorry

end lines_intersect_at_l647_647901


namespace set_intersection_correct_l647_647912

def set_A := {x : ℝ | x + 1 > 0}
def set_B := {x : ℝ | x - 3 < 0}
def set_intersection := {x : ℝ | -1 < x ∧ x < 3}

theorem set_intersection_correct : (set_A ∩ set_B) = set_intersection :=
by
  sorry

end set_intersection_correct_l647_647912


namespace constant_term_binomial_expansion_l647_647599

noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if h : k ≤ n then nat.choose n k else 0

theorem constant_term_binomial_expansion : 
  let x : ℚ := ??? -- x can be any non-zero rational number condition. 
  in (let T := λ r : ℕ, (- (1 / 2) : ℚ)^r * binom 6 r * x ^ (12 - 3 * r) 
     in T 4) = (15 / 16 : ℚ) := 
begin
  sorry
end

end constant_term_binomial_expansion_l647_647599


namespace part_I_solution_set_part_II_a_range_l647_647142

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 5|
noncomputable def g (a x : ℝ) : ℝ := a - (x - 2)^2

theorem part_I_solution_set : 
  {x : ℝ | f x ≤ x + 10} = set.Icc (-2 : ℝ) 14 :=
sorry

theorem part_II_a_range (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ 6 :=
sorry

end part_I_solution_set_part_II_a_range_l647_647142


namespace general_formula_T_n_lt_four_l647_647522

open Classical

variable {a : ℕ → ℤ}

-- Conditions for the arithmetic sequence {a_n}
axiom a_2 : a 2 = 0
axiom S_5_eq : ∑ k in finset.range 5, a (k+1) = 2 * a 4 - 1

-- General formula for the arithmetic sequence {a_n}
theorem general_formula (a : ℕ → ℤ) (d : ℤ) : (∀ n, a n = 2 - n) :=
by sorry

-- Conditions for {b_n} and the sum T_n of the first n terms of {b_n}
def b (n : ℕ) : ℝ := 2^(a n)
def T (n : ℕ) : ℝ := ∑ k in finset.range n, b (k+1)

-- Prove that T_n < 4
theorem T_n_lt_four (a_n_formula : ∀ n, a n = 2 - n) : ∀ n, T n < 4 :=
by sorry

end general_formula_T_n_lt_four_l647_647522


namespace fraction_power_product_l647_647729

theorem fraction_power_product : (7/8 : ℚ) ^ 3 * (7/8 : ℚ) ^ (-3) = 1 := 
by
  sorry

end fraction_power_product_l647_647729


namespace find_A_and_B_l647_647976

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  let c := n % 10 in
  let b := (n / 10) % 10 in
  let a := (n / 100) in
  a + b + c

def swap_hundreds_units (n : ℕ) : ℕ :=
  let c := n % 10 in
  let b := (n / 10) % 10 in
  let a := (n / 100) in
  100 * c + 10 * b + a

theorem find_A_and_B :
  ∃ A B : ℕ, is_three_digit A ∧ is_three_digit B ∧ 
  B = swap_hundreds_units A ∧ 
  A = 3 * B + 7 * digit_sum A ∧ 
  ((A = 421 ∧ B = 124) ∨ (A = 842 ∧ B = 248)) := sorry

end find_A_and_B_l647_647976


namespace find_a4_b4_l647_647835

theorem find_a4_b4 :
  ∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ),
    a₁ * b₁ + a₂ * b₃ = 1 ∧
    a₁ * b₂ + a₂ * b₄ = 0 ∧
    a₃ * b₁ + a₄ * b₃ = 0 ∧
    a₃ * b₂ + a₄ * b₄ = 1 ∧
    a₂ * b₃ = 7 ∧
    a₄ * b₄ = -6 :=
by
  sorry

end find_a4_b4_l647_647835


namespace n_eq_7_mod_10_l647_647485

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l647_647485


namespace time_to_fill_one_barrel_with_leak_l647_647375

-- Define the conditions
def normal_time_per_barrel := 3
def time_to_fill_12_barrels_no_leak := normal_time_per_barrel * 12
def additional_time_due_to_leak := 24
def time_to_fill_12_barrels_with_leak (t : ℕ) := 12 * t

-- Define the theorem
theorem time_to_fill_one_barrel_with_leak :
  ∃ t : ℕ, time_to_fill_12_barrels_with_leak t = time_to_fill_12_barrels_no_leak + additional_time_due_to_leak ∧ t = 5 :=
by {
  use 5, 
  sorry
}

end time_to_fill_one_barrel_with_leak_l647_647375


namespace whole_process_time_is_6_hours_l647_647654

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end whole_process_time_is_6_hours_l647_647654


namespace roots_exist_for_all_K_l647_647090

theorem roots_exist_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  -- Applied conditions and approach
  sorry

end roots_exist_for_all_K_l647_647090


namespace minimum_value_of_f_l647_647917

open Real

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x

theorem minimum_value_of_f (x : ℝ) (h : abs x ≤ π / 4) : 
  ∃ m : ℝ, (∀ y : ℝ, f y ≥ m) ∧ m = 1 / 2 - sqrt 2 / 2 :=
sorry

end minimum_value_of_f_l647_647917


namespace find_parabola_eq_find_minimum_distance_l647_647864

-- Define the parabola passing through (4, 4) with its focus on the x-axis
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Condition: Parabola passes through point (4, 4)
def passes_through_point (x y : ℝ) : Prop :=
  x = 4 ∧ y = 4

-- The line equation used in the second part of the problem
def line_eq (x y : ℝ) : Prop :=
  x - y + 4 = 0

-- Define the minimum distance function between a point (a^2/4, a) on the parabola and the given line
def min_distance (a : ℝ) : ℝ :=
  abs ((1 / 4) * (a - 2)^2 + 3) / real.sqrt 2

-- The proof problem consists of two parts:

-- Part 1: Prove the equation of the parabola
theorem find_parabola_eq (x y : ℝ) (h : passes_through_point x y) :
  parabola x y := by
  unfold passes_through_point at h
  cases h
  unfold parabola
  sorry

-- Part 2: Prove the minimum distance is 3 * sqrt(2) / 2
theorem find_minimum_distance (a : ℝ) :
  min_distance 2 = 3 * real.sqrt 2 / 2 := by
  unfold min_distance
  sorry

end find_parabola_eq_find_minimum_distance_l647_647864


namespace inequality_proof_l647_647635

variable {n : ℕ} (x : Fin n → ℝ)

theorem inequality_proof (hpos : ∀ i : Fin n, 0 < x i) :
    (∑ i : Fin n, (x i)^2 / x ((i + 1) % n)) ≥ ∑ i : Fin n, x i :=
by
  sorry

end inequality_proof_l647_647635


namespace max_salary_l647_647779

theorem max_salary (num_players : ℕ) (min_salary : ℕ) (total_salary : ℕ) (max_player_salary : ℕ)
(h1 : num_players = 23)
(h2 : min_salary = 18000)
(h3 : total_salary = 850000)
(h4 : max_player_salary = 454000) :
  ∃ x : ℕ, x ≤ max_player_salary ∧ (total_salary - (num_players - 1) * min_salary = x) :=
begin
  use (total_salary - (num_players - 1) * min_salary),
  split,
  { linarith [h1, h2, h3], },
  { simp [h4], },
end

end max_salary_l647_647779


namespace effective_discount_l647_647784

theorem effective_discount (original_price sale_price price_after_coupon : ℝ) :
  sale_price = 0.4 * original_price →
  price_after_coupon = 0.7 * sale_price →
  (original_price - price_after_coupon) / original_price * 100 = 72 :=
by
  intros h1 h2
  sorry

end effective_discount_l647_647784


namespace probability_of_x_gt_8y_l647_647259

noncomputable def probability_x_gt_8y : ℚ :=
  let rect_area := 2020 * 2030
  let tri_area := (2020 * (2020 / 8)) / 2
  tri_area / rect_area

theorem probability_of_x_gt_8y :
  probability_x_gt_8y = 255025 / 4100600 := by
  sorry

end probability_of_x_gt_8y_l647_647259


namespace positive_real_solution_l647_647841

theorem positive_real_solution (x : ℝ) (h : 0 < x)
  (h_eq : (1/3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 409 :=
sorry

end positive_real_solution_l647_647841


namespace total_feet_is_140_l647_647377

def total_heads : ℕ := 48
def number_of_hens : ℕ := 26
def number_of_cows : ℕ := total_heads - number_of_hens
def feet_per_hen : ℕ := 2
def feet_per_cow : ℕ := 4

theorem total_feet_is_140 : ((number_of_hens * feet_per_hen) + (number_of_cows * feet_per_cow)) = 140 := by
  sorry

end total_feet_is_140_l647_647377


namespace n_eq_7_mod_10_l647_647487

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l647_647487


namespace sum_of_angleC_and_angleD_l647_647934

theorem sum_of_angleC_and_angleD (A B C D : Point)
(∠A : Angle) (∠B : Angle) (∠C : Angle) (∠D : Angle)
(h1 : ∠A = 30) (h2 : ∠B + ∠C + ∠D = 330) (h3 : ∠B = ∠D) :
∠C + ∠D = 180 := by
  sorry

end sum_of_angleC_and_angleD_l647_647934


namespace inequaliy_pos_real_abc_l647_647240

theorem inequaliy_pos_real_abc (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 1) : 
  (a / (a * b + 1)) + (b / (b * c + 1)) + (c / (c * a + 1)) ≥ (3 / 2) := 
by
  sorry

end inequaliy_pos_real_abc_l647_647240


namespace area_of_PQRS_l647_647339

-- Definitions of the vertices of the trapezoid
def P : (ℝ × ℝ) := (0, 0)
def Q : (ℝ × ℝ) := (-2, 0)
def R : (ℝ × ℝ) := (-2, 5)
def S : (ℝ × ℝ) := (3, 5)

-- Definition of the function that checks if a line segment is horizontal
def isHorizontal (a b : ℝ × ℝ) : Prop :=
  a.snd = b.snd

-- Definition of the distance between two points on the x-axis
def xDistance (a b : ℝ × ℝ) : ℝ :=
  abs (a.fst - b.fst)

-- Decompose the problem into simpler definitions:
def b₁ := xDistance Q R  -- Length of base QR
def b₂ := xDistance P S  -- Length of base PS
def h  := abs (R.snd - Q.snd)  -- Height between bases

-- Definition of the area of the trapezoid PQRS
def area_trapezoid (b₁ b₂ h : ℝ) : ℝ :=
  (1 / 2) * (b₁ + b₂) * h

-- Conjecture: the area of trapezoid PQRS is 17.5 square units.
theorem area_of_PQRS : area_trapezoid b₁ b₂ h = 17.5 :=
by
  -- Proof goes here
  sorry

end area_of_PQRS_l647_647339


namespace determine_side_length_of_square_l647_647777

noncomputable def side_length_of_square (s : ℝ) : Prop :=
  let area_covered_by_squares := s^2
  let probability_covered := area_covered_by_squares = 0.5
  s = Real.sqrt (1 / 2)

theorem determine_side_length_of_square :
  ∃ s : ℝ, side_length_of_square s :=
by {
  use Real.sqrt (1 / 2),
  unfold side_length_of_square,
  sorry
}

end determine_side_length_of_square_l647_647777


namespace angle_D_l647_647966

theorem angle_D'_FE_is_90 
  (A B C : Type*)
  [metric_space A] [metric_space B] [metric_space C]
  (triangle_ABC : triangle A B C)
  (D E : A)
  (bisects_ADC : bisector (angle BAC) D)
  (exbisects_BAC : bissector (external_angle BAC) E)
  (circumcircle : circle A B C)
  (F ∈ circumcircle) (AD : line A D)
  (O : center_of_circumcircle (triangle_ABC))
  (D' : reflection_of D O) :
  ∠ D' F E = 90° :=
by
  sorry

end angle_D_l647_647966


namespace distinct_sequences_count_l647_647562

def is_valid_sequence (s : String) : Prop :=
  s.front = 'E' ∧ s.back = 'S' ∧ @Set.pairwise _ _ (λ x y, x ≠ y) s.data.toFinset

theorem distinct_sequences_count : 
  ∃ (count : ℕ), count = 120 ∧ 
  ∀ (seq : String), seq.length = 5 → is_valid_sequence seq →
  count = (let letters := ['M', 'A', 'T', 'H', 'I', 'C']
           in Multiset.card (Multiset.ofList letters).choose 3 * 6 * 5 * 4)
:= 
sorry

end distinct_sequences_count_l647_647562


namespace trey_total_time_is_two_hours_l647_647660

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end trey_total_time_is_two_hours_l647_647660


namespace fruit_picking_l647_647351

theorem fruit_picking (A P Pe : Type) (basket : finset (A ⊕ P ⊕ Pe)) (h : basket.card = 3) : ∃ ways : ℕ, ways = 6 :=
by
  have pick1 : ways1 = basket.card := by sorry
  have pick2 : ways2 = basket.card - 1 := by sorry
  exact ⟨pick1 * pick2, by sorry⟩

end fruit_picking_l647_647351


namespace fold_triangle_creases_length_l647_647772

noncomputable def fold_triangle_creases (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) : ℝ :=
  let C := (0, 0)
  let A := (12, 5)
  let E := (12 / 2, 5 / 2)
  let x_intersection := (59.5) / 12
  let G := (x_intersection, 0)
  real.sqrt ((fst G - fst E) ^ 2 + (snd G - snd E) ^ 2)

theorem fold_triangle_creases_length (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) : 
    fold_triangle_creases a b c ha hb hc = 2.708 :=
by sorry

end fold_triangle_creases_length_l647_647772


namespace fraction_built_first_half_l647_647399

theorem fraction_built_first_half (total_houses : ℕ)
  (additional_october : ℕ) 
  (houses_needed : ℕ)
  (built_first_half_fraction : ℚ) :
  total_houses = 2000 →
  additional_october = 300 →
  houses_needed = 500 →
  built_first_half_fraction * total_houses + additional_october = total_houses - houses_needed →
  built_first_half_fraction = 3 / 5 :=
by
  intros h1 h2 h3 h4
  rw [← h1, ← h2, ← h3] at h4
  sorry

end fraction_built_first_half_l647_647399


namespace greatest_gcd_l647_647088

theorem greatest_gcd (n : ℕ) (h : n > 0) : 
  let S_n := n^2 in 
  gcd (6 * S_n) (n+1) ≤ 6 ∧ ∃ k : ℕ, gcd (6 * S_n) (n+1) = 6 ↔ (n + 1) % 6 = 0 :=
by {
  intros,
  let S_n := n^2,
  let gcd_val := gcd (6 * S_n) (n + 1),
  have h1 : gcd (6 * S_n) (n + 1) = gcd 6 (n + 1),
  sorry,
  have h2 : gcd 6 (n + 1) ≤ 6,
  sorry,
  use [6],
  split,
  { sorry },
  { sorry }
}

end greatest_gcd_l647_647088


namespace count_n_eq_1_mod_4_in_range_l647_647956
open Complex Nat

-- Define the problem conditions as stated
theorem count_n_eq_1_mod_4_in_range :
  (Finset.filter (λ n, (1 + Complex.i)^n = 2^n * Complex.i) (Finset.range 2003)).card = 501 :=
by
  sorry

end count_n_eq_1_mod_4_in_range_l647_647956


namespace angle_C₀C₁C₂_is_30_degrees_l647_647980

-- Let A, B, C be points forming a right triangle at C.
variables {A B C A₀ B₀ C₀ C₁ C₂ : Type}
variables [RightTriangle A B C] [MidPoint A₀ B C] [MidPoint B₀ C A] [MidPoint C₀ A B]
variables [RegularTriangle A B₀ C₁] [RegularTriangle B A₀ C₂]

-- Prove the angle ∠C₀C₁C₂ is 30° given the above conditions.
theorem angle_C₀C₁C₂_is_30_degrees :
  ∠ C₀ C₁ C₂ = 30 :=
by
sorry

end angle_C₀C₁C₂_is_30_degrees_l647_647980


namespace sqrt_of_sqrt_16_is_pm_2_l647_647314

-- Define the concept of square root for clarity
def sqrt (x : ℝ) : ℝ := real.sqrt x

-- Problem: Prove that the square root of the square root of 16 is equal to ±2.
theorem sqrt_of_sqrt_16_is_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := 
by sorry

end sqrt_of_sqrt_16_is_pm_2_l647_647314


namespace cappuccino_cost_l647_647650

theorem cappuccino_cost 
  (total_order_cost drip_price espresso_price latte_price syrup_price cold_brew_price total_other_cost : ℝ)
  (h1 : total_order_cost = 25)
  (h2 : drip_price = 2 * 2.25)
  (h3 : espresso_price = 3.50)
  (h4 : latte_price = 2 * 4.00)
  (h5 : syrup_price = 0.50)
  (h6 : cold_brew_price = 2 * 2.50)
  (h7 : total_other_cost = drip_price + espresso_price + latte_price + syrup_price + cold_brew_price) :
  total_order_cost - total_other_cost = 3.50 := 
by
  sorry

end cappuccino_cost_l647_647650


namespace product_ge_one_l647_647233

variable (a b : ℝ)
variable (x1 x2 x3 x4 x5 : ℝ)

theorem product_ge_one
  (ha : 0 < a)
  (hb : 0 < b)
  (h_ab : a + b = 1)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5)
  (h_prod_xs : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1 + b) * (a * x2 + b) * (a * x3 + b) * (a * x4 + b) * (a * x5 + b) ≥ 1 :=
by
  sorry

end product_ge_one_l647_647233


namespace cadastral_value_of_land_l647_647413

theorem cadastral_value_of_land (tax_amount : ℝ) (tax_rate : ℝ) (V : ℝ)
    (h1 : tax_amount = 4500)
    (h2 : tax_rate = 0.003) :
    V = 1500000 :=
by
  sorry

end cadastral_value_of_land_l647_647413


namespace max_card_score_l647_647739

/-- 
  Given R (red cards), B (blue cards), Y (yellow cards) such that:
  1) R + B + Y = 15
  2) Score is calculated by:
     - Each red card has a value of 1.
     - Each blue card has a value of 2 * R.
     - Each yellow card has a value of 3 * B.
  Prove that the maximum score with these constraints is 168.
-/
theorem max_card_score (R B Y : ℕ) (h_sum : R + B + Y = 15)
(h_score : ∀ R B Y, 
  (B = 0 → 1 * R) ∨
  (B = 1 → 1 * R + 2 * R + 3 * Y) ∨
  (B > 1 → (15 - B) * 3 * B)) : ∃ max_score, max_score = 168 :=
by sorry

end max_card_score_l647_647739


namespace f_2000_equals_1499001_l647_647849

noncomputable def f (x : ℕ) : ℝ → ℝ := sorry

axiom f_initial : f 0 = 1

axiom f_recursive (x : ℕ) : f (x + 4) = f x + 3 * x + 4

theorem f_2000_equals_1499001 : f 2000 = 1499001 :=
by sorry

end f_2000_equals_1499001_l647_647849


namespace arrangements_with_gap_l647_647306

theorem arrangements_with_gap :
  ∃ (arrangements : ℕ), arrangements = 36 :=
by
  sorry

end arrangements_with_gap_l647_647306


namespace solve_fractional_sqrt_eq_l647_647829

theorem solve_fractional_sqrt_eq (x : ℚ) : (sqrt (6 * x) / sqrt (4 * (x - 2)) = 3) → x = 12 / 5 :=
by
  sorry

end solve_fractional_sqrt_eq_l647_647829


namespace num_teachers_l647_647363

variable (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ)

theorem num_teachers (h1 : num_students = 20) (h2 : ticket_cost = 5) (h3 : total_cost = 115) :
  (total_cost / ticket_cost - num_students = 3) :=
by
  sorry

end num_teachers_l647_647363


namespace sixth_graders_more_than_seventh_l647_647691

def pencil_cost : ℕ := 13
def eighth_graders_total : ℕ := 208
def seventh_graders_total : ℕ := 181
def sixth_graders_total : ℕ := 234

-- Number of students in each grade who bought a pencil
def seventh_graders_count := seventh_graders_total / pencil_cost
def sixth_graders_count := sixth_graders_total / pencil_cost

-- The difference in the number of sixth graders than seventh graders who bought a pencil
theorem sixth_graders_more_than_seventh : sixth_graders_count - seventh_graders_count = 4 :=
by sorry

end sixth_graders_more_than_seventh_l647_647691


namespace ratio_correct_l647_647372

noncomputable def side_ratio_to_diagonal (y : ℝ) : ℝ :=
  let x := (1 / 2) * y
  let diagonal := real.sqrt (x^2 + y^2)
  y / diagonal

theorem ratio_correct (y : ℝ) (h_pos : y > 0) :
  side_ratio_to_diagonal y = (2 * real.sqrt 5) / 5 := by
  sorry

end ratio_correct_l647_647372


namespace incenter_property_of_chord_lengths_equal_l647_647456

theorem incenter_property_of_chord_lengths_equal
  {A B C M P Q R : Point}
  (h_triangle : is_triangle A B C)
  (h_in_triangle : is_in_triangle M A B C)
  (h_circles : circles_on_diameters A B C M)
  (h_common_chords_equal : common_chords_equal P Q R A B C M) :
  is_incenter M A B C :=
sorry

end incenter_property_of_chord_lengths_equal_l647_647456


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647818

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647818


namespace triangle_side_lengths_abs_diff_l647_647875

theorem triangle_side_lengths_abs_diff (a b c : ℝ) : |a + b - c| - |b - a - c| = 2b - 2c :=
by
  sorry

end triangle_side_lengths_abs_diff_l647_647875


namespace hidden_prime_average_correct_l647_647850

noncomputable def hidden_prime_average : ℚ :=
  (13 + 17 + 59) / 3

theorem hidden_prime_average_correct :
  hidden_prime_average = 29.6 :=
by
  sorry

end hidden_prime_average_correct_l647_647850


namespace sum_of_coefficients_sum_even_odd_coefficients_l647_647231

noncomputable def P (x : ℝ) : ℝ := (2 * x^2 - 2 * x + 1)^17 * (3 * x^2 - 3 * x + 1)^17

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

theorem sum_even_odd_coefficients :
  (P 1 + P (-1)) / 2 = (1 + 35^17) / 2 ∧ (P 1 - P (-1)) / 2 = (1 - 35^17) / 2 := by
  sorry

end sum_of_coefficients_sum_even_odd_coefficients_l647_647231


namespace geometric_sequence_and_sum_l647_647639

variable {a_n b_n S_n T_n: ℕ → ℤ}
variable {n : ℕ}

-- Definitions based on the problem statement
def S_n_def (n : ℕ) : Prop := S_n n = 2 * a_n n - 3
def b_n_def (n : ℕ) : Prop := b_n n = a_n n + 2 * n

-- Theorem statement
theorem geometric_sequence_and_sum (h1 : ∀ n, S_n_def n) (h2 : ∀ n, b_n_def n) :
  (∃ r A, ∀ n, a_n n = A * r^n ∧ A = 3 ∧ r = 2) ∧
  (∀ n, T_n n = 3 * 2^n + n^2 + n - 3) :=
sorry

end geometric_sequence_and_sum_l647_647639


namespace product_of_chords_lengths_l647_647622

def semicircle_radius : ℝ := 3
def arc_divisions : ℕ := 5

noncomputable
def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 10)

def endpoint_A : ℂ := semicircle_radius
def endpoint_B : ℂ := -semicircle_radius
def points_C (k : ℕ) (hk : k ∈ {1, 2, 3, 4}) : ℂ := semicircle_radius * omega^k

theorem product_of_chords_lengths 
  (k : ℕ)
  (hk : k ∈ {1, 2, 3, 4}) :
  (∏ k in {1, 2, 3, 4}, Complex.abs (endpoint_A - points_C k hk)) *
  (∏ k in {1, 2, 3, 4}, Complex.abs (endpoint_B - points_C k (by simpa using hk))) = 65610 := 
sorry

end product_of_chords_lengths_l647_647622


namespace hexagon_area_m_plus_n_l647_647219

theorem hexagon_area_m_plus_n {A B C D E F : ℝ × ℝ}
  (hA : A = (0,0))
  (hB : B = (b,1))
  (hex_eq : ∀ (X Y : ℝ × ℝ), set.mem (Y - X) (subgroup.zpowers (12 : ℝ)))
  (angle_120 : ∀ (F: ℝ), ∠FAB = 120)
  (parallel_AB_DE : ∀ (DE : ℝ), ∃ D, ∃ E, D = (FA))
  (parallel_BC_EF : ∀ (EF : ℝ), ∃ E, ∃ F, B = (CB))
  (parallel_CD_FA : ∀ (FA : ℝ), ∃ F, ∃ A, C = (DA))
  (distinct_y : ∀ {a b c d e f}, y_coords = [0,1,2,3,4,5] → list.distinct y_coords) :
  let m := 25 in
  let n := 3 in
  m + n = 28 :=
sorry

end hexagon_area_m_plus_n_l647_647219


namespace find_q_l647_647630

-- Define the roots of the polynomial 2x^2 - 6x + 1 = 0
def roots_of_first_poly (a b : ℝ) : Prop :=
    2 * a^2 - 6 * a + 1 = 0 ∧ 2 * b^2 - 6 * b + 1 = 0

-- Conditions from Vieta's formulas for the first polynomial
def sum_of_roots (a b : ℝ) : Prop := a + b = 3
def product_of_roots (a b : ℝ) : Prop := a * b = 0.5

-- Define the roots of the second polynomial x^2 + px + q = 0
def roots_of_second_poly (a b : ℝ) (p q : ℝ) : Prop :=
    (λ x => x^2 + p * x + q) (3 * a - 1) = 0 ∧ 
    (λ x => x^2 + p * x + q) (3 * b - 1) = 0

-- Proof that q = -0.5 given the conditions
theorem find_q (a b p q : ℝ) (h1 : roots_of_first_poly a b) (h2 : sum_of_roots a b)
    (h3 : product_of_roots a b) (h4 : roots_of_second_poly a b p q) : q = -0.5 :=
by
  sorry

end find_q_l647_647630


namespace Megan_number_of_folders_l647_647649

theorem Megan_number_of_folders :
  ∀ (init_files added_files files_per_folder : ℝ),
    init_files = 93.0 →
    added_files = 21.0 →
    files_per_folder = 8.0 →
    let total_files := init_files + added_files in
    let exact_folders := total_files / files_per_folder in
    let required_folders := Nat.ceil exact_folders in
    required_folders = 15 :=
by
  intros init_files added_files files_per_folder h_init h_added h_files_per_folder
  let total_files := init_files + added_files
  let exact_folders := total_files / files_per_folder
  let required_folders := Nat.ceil exact_folders
  sorry

end Megan_number_of_folders_l647_647649


namespace closest_multiple_of_18_to_2023_l647_647342

def is_multiple_of (n m : ℕ) : Prop :=
  ∃ k, m = k * n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_closest_multiple (m target n : ℕ) : Prop :=
  is_multiple_of m n ∧ 
  (∀ k, is_multiple_of m k → abs (int.natAbs (k - target)) ≥ abs (int.natAbs (n - target)))

theorem closest_multiple_of_18_to_2023 : ∃ n, n = 2028 ∧ is_closest_multiple 18 2023 2028 :=
by
  /-
    We need to provide the proof that 2028 is the closest multiple of 18 to 2023.
    - Check that 2028 is multiple of 18.
    - Check other conditions and establish its closeness.
  -/
  sorry

end closest_multiple_of_18_to_2023_l647_647342


namespace trig_identity_l647_647097

theorem trig_identity (θ : ℝ) (h₁ : Real.tan θ = 2) :
  2 * Real.cos θ / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 :=
by
  sorry

end trig_identity_l647_647097


namespace problem_1_1_problem_1_2_problem_2_l647_647517

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := f x - Real.sqrt 3

theorem problem_1_1 (ω : ℝ) (k : ℤ) : (ω > 0) ∧ (|Real.pi / 6| ≤ Real.pi / 2) ∧ (Real.pi / 2 = Real.pi / ω) ∧ 
  ∀ x, ω = 1 ∧ f x = 2 * Real.sin (2 * x + Real.pi / 6) ∧ 
  x ∈ set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) :=
by sorry

theorem problem_1_2 : 
  (f (Real.pi / 6) = 2) ∧ (f (Real.pi / 2) = -1) :=
by sorry

theorem problem_2 (ω : ℝ) : 
  (ghas_zeros (ghas_zeros (ghas_zeros (g (Real.pi / 6)) (g (2 * Real.pi / 3))) = 5 ∧ 
  (25 / 36 ≤ ω) ∧ (ω < 3 / 4)) :=
by sorry

end problem_1_1_problem_1_2_problem_2_l647_647517


namespace inequality_solution_l647_647683

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end inequality_solution_l647_647683


namespace derivative_at_minus_one_l647_647144

noncomputable def f (c : ℝ) (x : ℝ) := c * x^4 + 2 * x

theorem derivative_at_minus_one : 
  ∀ (f : ℝ → ℝ) (c : ℝ), (f = λ x, c * x^4 + 2 * x) → 
  (∃ c', (∂ (_. f (-1))).value = c' ∧ c' = (2 / 5)) :=
begin
  intros f c h,
  have h_d : (∂ (_. f (-1))).value = 4 * c * (-1)^3 + 2,
  { sorry },
  use 4 * c * (-1)^3 + 2,
  split,
  { exact h_d },
  { rw [h_d, pow_succ, pow_zero, mul_neg_one, right.neg_eq_neg_iff],
    linarith },
end

end derivative_at_minus_one_l647_647144


namespace inscribed_rectangle_circumference_l647_647584

theorem inscribed_rectangle_circumference (S : ℝ) (hS : S = 10)
  (hπ : real.pi = 3) :
  let AB := S - 2,
      AD := S - 4,
      AC := real.sqrt(AB^2 + AD^2),
      Diameter := AC,
      Circumference := hπ * Diameter in
  Circumference = 30 := by
sorry

end inscribed_rectangle_circumference_l647_647584


namespace bottle_caps_after_transactions_l647_647655

noncomputable def initial_bottle_caps : ℕ := 250
noncomputable def additional_from_Catherine : ℕ := 415
noncomputable def percentage_given_to_Anthony : ℚ := 0.35
noncomputable def additional_found : ℕ := 180
noncomputable def fraction_kept_by_Nicholas : ℚ := 5 / 9

theorem bottle_caps_after_transactions : 
  let total_initial := initial_bottle_caps + additional_from_Catherine,
      caps_given_to_Anthony := total_initial * percentage_given_to_Anthony,
      caps_after_anthony := total_initial - caps_given_to_Anthony.to_nat,
      total_after_finding := caps_after_anthony + additional_found,
      final_caps_kept := total_after_finding * fraction_kept_by_Nicholas in
  final_caps_kept.to_nat = 340 :=
by sorry

end bottle_caps_after_transactions_l647_647655


namespace angle_DAE_45_l647_647979

-- Let ABC be a right triangle with ∠A = 90° and AB = AC.
variable (A B C D E : Point)
variable [DecidableEq Point]
variable (AB_AC : A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ ∠ B A C = 90 ° ∧ B - A ∥ C - A ∧ dist B A = dist C A)
variable (BD_DE_EC : ∀ (B D E C : Point),  B - A ∥ C - A → dist B D / dist D E = 1 / 2 → dist D E / dist E C = 2 / √3)

-- Proving:
theorem angle_DAE_45 (h : BD_DE_EC) : ∠ D A E = 45 ° :=
sorry

end angle_DAE_45_l647_647979


namespace digits_are_different_probability_l647_647012

noncomputable def prob_diff_digits : ℚ :=
  let total := 999 - 100 + 1
  let same_digits := 9
  1 - (same_digits / total)

theorem digits_are_different_probability :
  prob_diff_digits = 99 / 100 :=
by
  sorry

end digits_are_different_probability_l647_647012


namespace line_through_midpoint_of_parabola_l647_647737

theorem line_through_midpoint_of_parabola :
  ∃ k : ℝ, (∀ (x y : ℝ), y^2 = 16 * x → (y - 1) = k * (x - 2)) → k = 8 := 
by
  -- Given conditions
  intro k h,
  -- Given the point (2,1) lies on the line and is the midpoint
  have h1 : (2,1) ∈ (λ p : ℝ × ℝ, p.2^2 = 16 * p.1), by sorry,
  -- Line passes through the point (2,1)
  have h2 : ∀ x y, y = k * (x - 2) + 1 → (x,y) ∈ (λ p : ℝ × ℝ, p.2^2 = 16 * p.1), by sorry,
  -- We know the parabola equation holds for points on it
  -- Use equations of the points to deduce k = 8
  sorry

end line_through_midpoint_of_parabola_l647_647737


namespace combined_selling_price_correct_l647_647382

def exchange_rate := 70
def cost_A := 10 * exchange_rate
def cost_B := 15 * exchange_rate
def cost_C := 20 * exchange_rate
def discount_B_rate := 0.10
def tax_rate := 0.08

def cost_B_discounted := cost_B * (1 - discount_B_rate)
def profit_A := 0.25
def profit_B := 0.30
def profit_C := 0.20

def selling_price_A_before_tax := cost_A * (1 + profit_A)
def selling_price_B_before_tax := cost_B_discounted * (1 + profit_B)
def selling_price_C_before_tax := cost_C * (1 + profit_C)

def selling_price_A_with_tax := selling_price_A_before_tax * (1 + tax_rate)
def selling_price_B_with_tax := selling_price_B_before_tax * (1 + tax_rate)
def selling_price_C_with_tax := selling_price_C_before_tax * (1 + tax_rate)

def combined_selling_price := selling_price_A_with_tax + selling_price_B_with_tax + selling_price_C_with_tax

theorem combined_selling_price_correct : combined_selling_price = 4086.18 := by 
    sorry

end combined_selling_price_correct_l647_647382


namespace LindasTrip_l647_647643

theorem LindasTrip (x : ℝ) :
    (1 / 4) * x + 30 + (1 / 6) * x = x →
    x = 360 / 7 :=
by
  intros h
  sorry

end LindasTrip_l647_647643


namespace proportional_segments_l647_647531

theorem proportional_segments (a b c d : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) (h4 : a / b = c / d) : d = 3 / 2 :=
by
  -- proof steps here
  sorry

end proportional_segments_l647_647531


namespace condition_for_tetrahedron_edges_l647_647237

theorem condition_for_tetrahedron_edges (a : ℝ) (k : ℕ) (hk : k ∈ [1, 2, 3, 4, 5]) :
  (∃ (tetra : Tetrahedron), 
    (count_eq tetra.edges a = k) ∧ (count_eq tetra.edges 1 = 6 - k)) ↔
  (if k = 1 then 0 < a ∧ a < sqrt 3
   else if k = 2 then 0 < a ∧ a < sqrt (2 + sqrt 3)
   else if k = 3 then 0 < a
   else if k = 4 then a > sqrt (2 - sqrt 3)
   else if k = 5 then a > sqrt 3 / 3
   else false) :=
sorry

end condition_for_tetrahedron_edges_l647_647237


namespace tim_scored_sum_first_8_even_numbers_l647_647355

-- Define the first 8 even numbers.
def first_8_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16]

-- Define the sum of those numbers.
def sum_first_8_even_numbers : ℕ := List.sum first_8_even_numbers

-- The theorem stating the problem.
theorem tim_scored_sum_first_8_even_numbers : sum_first_8_even_numbers = 72 := by
  sorry

end tim_scored_sum_first_8_even_numbers_l647_647355


namespace altitude_triangle_similarity_l647_647631

theorem altitude_triangle_similarity 
  {A B C A1 B1 : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace A1] [MetricSpace B1]
  [Inculding : Angle A B C] 
  (hAA1 : IsAltitude A A1 C)
  (hBB1 : IsAltitude B B1 C) :
  Similar (Triangle A1 B1 C) (Triangle A B C) ∧ similarityRatio (Triangle A1 B1 C) (Triangle A B C) = |cos (Angle A B C)| :=
sorry

end altitude_triangle_similarity_l647_647631


namespace sin_cos_eq_one_l647_647840

theorem sin_cos_eq_one (x : ℝ) (hx0 : 0 ≤ x) (hx2pi : x < 2 * Real.pi) :
  (Real.sin x - Real.cos x = 1) ↔ (x = Real.pi / 2 ∨ x = Real.pi) :=
by
  sorry

end sin_cos_eq_one_l647_647840


namespace probability_A_ultimately_wins_l647_647586

-- Definitions based on conditions
def required_wins_A : ℕ := 2
def required_wins_B : ℕ := 3
def game_win_prob : ℚ := 1 / 2

-- Target statement, assume the framework to handle such probability functions.
theorem probability_A_ultimately_wins : 
  let p_A_win := (game_win_prob)^required_wins_A in
  let p_independent_events := (1 - game_win_prob)^required_wins_A * game_win_prob^required_wins_A in
  let p_comb2_games := 2 choose 1 * p_independent_events in
  let p_comb3_games := 3 choose 2 * (1 - game_win_prob)^2 * game_win_prob^2 in
  p_A_win + p_comb2_games + p_comb3_games = 11 / 16 := 
sorry

end probability_A_ultimately_wins_l647_647586


namespace z_n_solution_l647_647078

noncomputable def z_n (n : ℕ) : ℝ :=
  let α1 := (1 + Real.sqrt 5) / 2
  let α2 := (1 - Real.sqrt 5) / 2
  in α1^n + α2^n + 2

axiom u_n_def (n : ℕ) : ℝ :=
  (1 / Real.sqrt 5) * ((1 + Real.sqrt 5) / 2)^(n + 1) - (1 / Real.sqrt 5) * ((1 - Real.sqrt 5) / 2)^(n + 1)

axiom z_n_eq (n : ℕ) : z_n n = 
  u_n_def (n - 1) + 2 * u_n_def (n - 2) + 2

theorem z_n_solution (n : ℕ) : 
  z_n n = ((1 + Real.sqrt 5) / 2)^n + ((1 - Real.sqrt 5) / 2)^n + 2 :=
by sorry

end z_n_solution_l647_647078


namespace mostWaterIntake_l647_647429

noncomputable def dailyWaterIntakeDongguk : ℝ := 5 * 0.2 -- Total water intake in liters per day for Dongguk
noncomputable def dailyWaterIntakeYoonji : ℝ := 6 * 0.3 -- Total water intake in liters per day for Yoonji
noncomputable def dailyWaterIntakeHeejin : ℝ := 4 * 500 / 1000 -- Total water intake in liters per day for Heejin (converted from milliliters)

theorem mostWaterIntake :
  dailyWaterIntakeHeejin = max dailyWaterIntakeDongguk (max dailyWaterIntakeYoonji dailyWaterIntakeHeejin) :=
by
  sorry

end mostWaterIntake_l647_647429


namespace problem_statement_l647_647130

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 2) :
  (¬ (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 2) → (∃ (s : ℝ), s = min (1/x + 1/y) 2)))
  ∧ 
  (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 2) → (∃ (s : ℝ), s = max (sqrt x + sqrt y) 2)) := 
begin
  -- Proof for option B
  have min_value : (∃ s : ℝ, s = min (1/x + 1/y) 2),
  {
    sorry,
  },
  -- Proof for option C
  have max_value : (∃ s : ℝ, s = max (sqrt x + sqrt y) 2),
  {
    sorry,
  },
  exact ⟨min_value, max_value⟩,
end

end problem_statement_l647_647130


namespace find_ratio_CP_PE_l647_647959

-- Define the ratios given in the problem
def ratio_CD_DB : ℚ := 3 / 1
def ratio_AE_EB : ℚ := 3 / 2

-- Define the final ratio r
def final_ratio : ℚ := 5

-- The statement of the problem in Lean 4
theorem find_ratio_CP_PE :
  (∀ (ABC : Type) (P: Type) (point_C point_B point_D point_A point_E : Poin ABC), 
    (ratio_CD_DB = (3 / 1))  ∧ 
    (ratio_AE_EB = (3 / 2)) ->
    r = 5) := 
sorry

end find_ratio_CP_PE_l647_647959


namespace partition_identity_l647_647616

def P (n : ℕ) : ℕ := -- define the function P that returns the number of ways to partition n

def a (k : ℕ) : ℕ := -- define the function a that returns the number of ones in the binary representation of k
  (nat.join (nat.bits (k))) 

theorem partition_identity (n : ℕ) : 
  ∑ k in finset.range (n + 1), (-1) ^ (a k) * P (n - k) = 0 :=
sorry

end partition_identity_l647_647616


namespace whole_process_time_is_6_hours_l647_647653

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end whole_process_time_is_6_hours_l647_647653


namespace emma_distance_conrad_finish_time_alistair_passes_salma_l647_647752

-- Define the initial conditions.
def swimming_distance := 2 -- in km
def biking_distance := 40 -- in km
def running_distance := 10 -- in km
def total_distance := swimming_distance + biking_distance + running_distance -- 52 km
def start_time := 8 * 60 -- 8:00 AM in minutes

-- Define each part of the problem as a theorem to be proved.

-- (a) Emma's distance
theorem emma_distance (fraction_completed : ℚ := 1/13) :
  let distance_traveled := fraction_completed * total_distance
  distance_traveled = 4 := 
by sorry

-- (b) Conrad's finish time
theorem conrad_finish_time (swimming_time : ℚ := 30, swim_speed_multiplier : ℚ := 12, run_speed_multiplier : ℚ := 3) :
  let swim_speed := swimming_distance / (swimming_time / 60)
  let bike_speed := swim_speed * swim_speed_multiplier
  let run_speed := swim_speed * run_speed_multiplier
  let bike_time := (biking_distance / bike_speed) * 60
  let run_time := (running_distance / run_speed) * 60
  let total_time := swimming_time + bike_time + run_time
  start_time + total_time = 10 * 60 + 10 := 
by sorry

-- (c) Alistair passes Salma
theorem alistair_passes_salma (alistair_swim_time : ℚ := 36, alistair_bike_speed : ℚ := 28, salma_swim_time : ℚ := 30, salma_bike_speed : ℚ := 24) :
  let alistair_bike_time (t : ℚ) := (t - alistair_swim_time)
  let salma_bike_time (t : ℚ) := (t - salma_swim_time)
  let alistair_bike_distance (t : ℚ) := alistair_bike_speed * alistair_bike_time(t) / 60 + swimming_distance
  let salma_bike_distance (t : ℚ) := salma_bike_speed * salma_bike_time(t) / 60 + swimming_distance
  ∃ t, alistair_bike_distance(t) = salma_bike_distance(t) ∧ start_time + t = 9 * 60 + 12 :=
by sorry


end emma_distance_conrad_finish_time_alistair_passes_salma_l647_647752


namespace tangent_line_at_point_l647_647698

theorem tangent_line_at_point :
  let curve (x : ℝ) := 2 * x^2 - 1
  let P := (-3, 17)
  let tangent_eqn (x : ℝ) := -12 * x - 19
  curve P.1 = P.2 →
  ∀ x y : ℝ, y = curve x → 
  ∀ k : ℝ, k = (deriv curve) P.1 →
  y - P.2 = k * (x - P.1) →
  y = tangent_eqn x :=
begin
  sorry
end

end tangent_line_at_point_l647_647698


namespace length_AD_l647_647941

variables {r : ℝ} -- radius of circle O
variables (O X Q A B M C D : EuclideanGeometry.Point)
variables (AX Y : EuclideanGeometry.Line)

/-- Q is the midpoint of radius OX -/
def midpoint_OX : EuclideanGeometry.Point := EuclideanGeometry.midpoint O X

/-- AB is perpendicular to XY at Q -/
axiom AB_perpendicular_XY : EuclideanGeometry.Perpendicular (EuclideanGeometry.line_through A B) (EuclideanGeometry.line_through X Y)

/-- AB is the diameter of a semicircle intersecting XY at M -/
axiom semicircle_AB_diameter : EuclideanGeometry.Semicircle (EuclideanGeometry.line_through A B) XY M

/-- Lines AM and BM intersect circle O at points C and D respectively -/
axiom intersection_AM_C : C ∈ EuclideanGeometry.circle O r ∧ EuclideanGeometry.line_through A M ≠ EuclideanGeometry.line_through A C
axiom intersection_BM_D : D ∈ EuclideanGeometry.circle O r ∧ EuclideanGeometry.line_through B M ≠ EuclideanGeometry.line_through B D

theorem length_AD : EuclideanGeometry.distance A D = r * Real.sqrt 2 := 
sorry

end length_AD_l647_647941


namespace total_average_marks_l647_647846

def total_marks (average_marks : ℕ) (num_students : ℕ) : ℕ := average_marks * num_students

theorem total_average_marks (average_marks_class1 average_marks_class2 : ℕ) (num_students_class1 num_students_class2 : ℕ) : 
  average_marks_class1 = 45 
  → num_students_class1 = 39 
  → average_marks_class2 = 70 
  → num_students_class2 = 35 
  → (total_marks average_marks_class1 num_students_class1 + total_marks average_marks_class2 num_students_class2) / (num_students_class1 + num_students_class2) = 56.83 := 
by 
  sorry

end total_average_marks_l647_647846


namespace ring_revolutions_before_stopping_l647_647011

variable (R ω μ m g : ℝ) -- Declare the variables as real numbers

-- Statement of the theorem
theorem ring_revolutions_before_stopping
  (h_positive_R : 0 < R)
  (h_positive_ω : 0 < ω)
  (h_positive_μ : 0 < μ)
  (h_positive_m : 0 < m)
  (h_positive_g : 0 < g) :
  let N1 := m * g / (1 + μ^2)
  let N2 := μ * m * g / (1 + μ^2)
  let K_initial := (1 / 2) * m * R^2 * ω^2
  let A_friction := -2 * π * R * n * μ * (N1 + N2)
  ∃ n : ℝ, n = ω^2 * R * (1 + μ^2) / (4 * π * g * μ * (1 + μ)) :=
by sorry

end ring_revolutions_before_stopping_l647_647011


namespace velocity_zero_at_t_eq_2_l647_647304

noncomputable def motion_equation (t : ℝ) : ℝ := -4 * t^3 + 48 * t

theorem velocity_zero_at_t_eq_2 :
  (exists t : ℝ, t > 0 ∧ deriv (motion_equation) t = 0) :=
by
  sorry

end velocity_zero_at_t_eq_2_l647_647304


namespace geometric_sequences_length_3_l647_647035

theorem geometric_sequences_length_3 : 
  let sequences := { seq | ∃ (a r : ℕ), 1 ≤ a ∧ a ≤ 10 ∧ (∀ n, n < 3 → 1 ≤ a * r^n ∧ a * r^n ≤ 10) };
  sequences.count = 13 :=
by sorry

end geometric_sequences_length_3_l647_647035


namespace inequality_solution_l647_647682

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end inequality_solution_l647_647682


namespace cadastral_value_of_land_l647_647411

theorem cadastral_value_of_land (tax_amount_paid : ℝ) (tax_rate : ℝ) (V : ℝ) :
  (tax_amount_paid = 4500) → (tax_rate = 0.003) → (V = tax_amount_paid / tax_rate) → (V = 1500000) :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have hV : V = 4500 / 0.003 := h3
  have hV_value : 4500 / 0.003 = 1500000 := 
    by norm_num
  rw hV_value at hV
  exact hV

#eval cadastral_value_of_land 4500 0.003 1500000 sorry -- Testing purpose to ensure it compiles.

end cadastral_value_of_land_l647_647411


namespace identity_a_identity_b_identity_c_identity_d_l647_647267

theorem identity_a (x : ℂ) (n : ℕ) (hn : 1 ≤ n) :
  x^(2*n) - 1 = (x^2 - 1) * ∏ k in Finset.range (n - 1) + 1, (x^2 - 2 * x * complex.cos (k * real.pi / n) + 1) :=
sorry

theorem identity_b (x : ℂ) (n : ℕ) :
  x^(2*n + 1) - 1 = (x - 1) * ∏ k in Finset.range n + 1, (x^2 - 2 * x * complex.cos (2 * k * real.pi / (2 * n + 1)) + 1) :=
sorry

theorem identity_c (x : ℂ) (n : ℕ) :
  x^(2*n + 1) + 1 = (x + 1) * ∏ k in Finset.range n + 1, (x^2 + 2 * x * complex.cos (2 * k * real.pi / (2 * n + 1)) + 1) :=
sorry

theorem identity_d (x : ℂ) (n : ℕ) (hn : 1 ≤ n) :
  x^(2*n) + 1 = ∏ k in Finset.range n, (x^2 - 2 * x * complex.cos ((2 * k + 1) * real.pi / (2 * n)) + 1) :=
sorry

end identity_a_identity_b_identity_c_identity_d_l647_647267


namespace cost_of_article_is_308_l647_647918

theorem cost_of_article_is_308 
  (C G : ℝ) 
  (h1 : 348 = C + G)
  (h2 : 350 = C + G + 0.05 * G) : 
  C = 308 :=
by
  sorry

end cost_of_article_is_308_l647_647918


namespace tangent_conditions_l647_647870

theorem tangent_conditions (m : ℝ) : 
  let A := (1 : ℝ, 0 : ℝ)
  let center := (-m / 2 : ℝ, 0 : ℝ)
  let radius := sqrt (m^2 / 4 - 1)
  let d := dist A center
  in d > radius ↔ m > 2 :=
sorry

end tangent_conditions_l647_647870


namespace incenter_property_of_chord_lengths_equal_l647_647455

theorem incenter_property_of_chord_lengths_equal
  {A B C M P Q R : Point}
  (h_triangle : is_triangle A B C)
  (h_in_triangle : is_in_triangle M A B C)
  (h_circles : circles_on_diameters A B C M)
  (h_common_chords_equal : common_chords_equal P Q R A B C M) :
  is_incenter M A B C :=
sorry

end incenter_property_of_chord_lengths_equal_l647_647455


namespace min_reciprocal_sum_l647_647512

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 ^ (a + b) = 3) : (1 / a) + (1 / b) = 4 :=
by
  have hab : a + b = 1 := by
    sorry
  -- We can continue by showing that the minimum value is achieved
  sorry

end min_reciprocal_sum_l647_647512


namespace insphere_touch_centroid_l647_647898

theorem insphere_touch_centroid (tetrahedron : Type) [is_regular_tetrahedron tetrahedron] :
  ∀ (face : triangular_face tetrahedron), touchpoint (insphere tetrahedron) face = centroid face :=
by sorry

end insphere_touch_centroid_l647_647898


namespace AlWinsProbability_l647_647022

noncomputable def BobPlaysRandomly : 𝕜 :=
  (mk_ℙ [1/3, 1/3, 1/3]) 

noncomputable def AlPlaysRock : 𝕜 :=
  mk_ℙ [1]

theorem AlWinsProbability :
  ∀ (P_rock P_paper P_scissors : ℝ),
    P_rock = 1/3 → 
    P_paper = 1/3 →
    P_scissors = 1/3 →
    (P_rock + P_paper + P_scissors = 1) → 
    (AlPlaysRock * (P_scissors)) = (1/3) :=
by
  sorry

end AlWinsProbability_l647_647022


namespace median_divides_triangle_l647_647735

-- Definitions based on conditions
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

def midpoint {α : Type} [LinearOrderedField α] (P Q : α × α) : α × α :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def median {α : Type} [LinearOrderedField α] (T : Triangle α) : (α × α) :=
midpoint T.B T.C

def area {α : Type} [LinearOrderedField α] (T : Triangle α) : α :=
(1 / 2) * abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2))

-- Lean statement that needs to be proven
theorem median_divides_triangle {α : Type} [LinearOrderedField α] (T : Triangle α) :
  let M := median T in
  area {A := T.A, B := T.B, C := M} = area {A := T.A, B := T.C, C := M} :=
sorry

end median_divides_triangle_l647_647735


namespace sam_current_dimes_l647_647276

def original_dimes : ℕ := 8
def sister_borrowed : ℕ := 4
def friend_borrowed : ℕ := 2
def sister_returned : ℕ := 2
def friend_returned : ℕ := 1

theorem sam_current_dimes : 
  (original_dimes - sister_borrowed - friend_borrowed + sister_returned + friend_returned = 5) :=
by
  sorry

end sam_current_dimes_l647_647276


namespace principal_amount_l647_647731

theorem principal_amount (SI R T : ℕ) (P : ℕ) : SI = 160 ∧ R = 5 ∧ T = 4 → P = 800 :=
by
  sorry

end principal_amount_l647_647731


namespace asymptotes_of_hyperbola_l647_647529

noncomputable section

variables {a b : ℝ} {F1 F2 P : ℝ × ℝ}

def hyperbola : Set (ℝ × ℝ) := { p | p.1^2 / a^2 - p.2^2 / b^2 = 1 }

def distance (P1 P2 : ℝ × ℝ) : ℝ := Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem asymptotes_of_hyperbola
  (h1 : ∀ p ∈ hyperbola, distance p F1 + distance p F2 = 6 * a)
  (h2 : ∀ p ∈ hyperbola, min_deg_angle (degree (angle (P, F1, F2)) = 30))
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : P ∈ hyperbola)
  : ∀ p : ℝ × ℝ, p ∈ hyperbola → p.2 = sqrt 2 * p.1 ∨ p.2 = -sqrt 2 * p.1 :=
sorry

end asymptotes_of_hyperbola_l647_647529


namespace not_possible_f_g_l647_647971

theorem not_possible_f_g (f g : ℝ → ℝ) :
  ¬(∀ x y : ℝ, 1 + x^2016 * y^2016 = f(x) * g(y)) :=
by
  sorry

end not_possible_f_g_l647_647971


namespace length_of_garden_l647_647924

-- Definitions based on conditions
def P : ℕ := 600
def b : ℕ := 200

-- Theorem statement
theorem length_of_garden : ∃ L : ℕ, 2 * (L + b) = P ∧ L = 100 :=
by
  existsi 100
  simp
  sorry

end length_of_garden_l647_647924


namespace coffee_cost_per_week_l647_647206

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end coffee_cost_per_week_l647_647206


namespace incenter_chords_equal_l647_647441

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry Triangle

/-- Given a triangle ABC and a point M inside it,
  if circles are constructed on segments MA, MB, and MC as diameters,
  then the point M such that the lengths of common chords are equal, 
  is the incenter of triangle ABC. -/
theorem incenter_chords_equal (A B C M : α) (hM : incircle M (Triangle.mk A B C))
  (h_chord_eq : ∀ (P Q R : α), is_common_chord P (circle (segment.mk M A))
    (circle (segment.mk M B)) = is_common_chord Q (circle (segment.mk M B)) 
    (circle (segment.mk M C)) = is_common_chord R (circle (segment.mk M C))
    (circle (segment.mk M A))) : is_incenter M (Triangle.mk A B C) :=
sorry

end incenter_chords_equal_l647_647441


namespace count_correct_statements_l647_647151

theorem count_correct_statements :
  let f : ℝ → ℝ := λ x, -2 * (x + 3)^2 + 1 in
  let s1 := (-2 < 0) in
  let s2 := (∀ x : ℝ, (x + 3) = 0 → x = -3) in
  let s3 := (f (-3) = 1) in
  let s4 := ∀ x : ℝ, x > 3 → f x < f (x-1) in
  (s1 → True) ∧ (¬s2 → True) ∧ (¬s3 → True) ∧ (s4 → True) → 2 = 2 :=
by
  sorry

end count_correct_statements_l647_647151


namespace fixed_points_range_l647_647508

theorem fixed_points_range (a b : ℝ) (h : ∀ x : ℝ, f (a, b) x = ax² + bx - b) : 
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f (a, b) x₁ = x₁ ∧ f (a, b) x₂ = x₂) ↔ 0 < a ∧ a < 1 :=
begin
  sorry
end

end fixed_points_range_l647_647508


namespace bridge_length_is_correct_l647_647016

open Real

-- Given values
def train_length : ℝ := 200
def train_speed_kmh : ℝ := 60
def crossing_time : ℝ := 45

-- Conversion factor from km/h to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Expected computed values
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor
def total_distance : ℝ := train_speed_ms * crossing_time
def bridge_length : ℝ := total_distance - train_length

-- The theorem to prove
theorem bridge_length_is_correct : bridge_length = 550.15 :=
by
  rw [train_length, train_speed_kmh, conversion_factor, crossing_time],
  have h1 : train_speed_ms = 60 * (1000 / 3600), sorry,
  have h2 : total_distance = 16.67 * 45, sorry,
  have h3 : total_distance = 750.15, sorry,
  have h4 : bridge_length = 750.15 - 200, sorry,
  sorry

end bridge_length_is_correct_l647_647016


namespace max_tan_A_sub_B_l647_647115

variable {α : Type}
variables (a b c A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A < π ∧ B < π ∧ C < π

theorem max_tan_A_sub_B 
  (h1 : is_triangle A B C) 
  (h2 : a * Real.cos B - b * Real.cos A = (1 / 2) * c) : 
  Real.tan (A - B) = (Real.sqrt 3) / 3 := 
sorry

end max_tan_A_sub_B_l647_647115


namespace lines_concurrent_l647_647176

variables {A B C D E F M X : Type}
variables [line A C] [line B F] [line C F]

-- Assume the geometric conditions given
variables (θ : Angle A B C = π / 2)
variables (hFA_FB : Segment FA = Segment FB)
variables (hF_between_A_C : Between F A C)
variables (hDA_DC : Segment DA = Segment DC)
variables (hAC_bisector_DAB : IsAngleBisector Angle A C D)
variables (hEA_ED : Segment EA = Segment ED)
variables (hAD_bisector_EAC : IsAngleBisector Angle A D E)
variables (hM_mid_CF : IsMidpoint M C F)
variables (hAMXE_parallelgram : IsParallelogram AM X E)

-- Define the concurrency of BD, FX, and ME
theorem lines_concurrent (BD FX ME : Line) :
  Concurrent BD FX ME :=
sorry

end lines_concurrent_l647_647176


namespace comb_identity_a_l647_647674

theorem comb_identity_a (r m k : ℕ) (h : 0 ≤ k ∧ k ≤ m ∧ m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end comb_identity_a_l647_647674


namespace count_perfect_square_multiples_of_3_lt_100_l647_647569

theorem count_perfect_square_multiples_of_3_lt_100 : 
  {n : ℕ | n < 100 ∧ ∃ m : ℕ, n = m * m ∧ m % 3 = 0}.card = 3 :=
by 
  sorry

end count_perfect_square_multiples_of_3_lt_100_l647_647569


namespace num_points_120_not_80_ray_partitional_l647_647623

-- Definitions based on conditions provided
def is_ray_partitional (R : Set Point) (X : Point) (n : Nat) : Prop :=
  X ∈ R ∧ (∃ rays, raysemanating_from X ∧ divides_square_into_equal_area_triangles R X n)

-- Definition specific to the problem
def unit_square (s : ℝ) : Set Point :=
  { pt | 0 < pt.x ∧ pt.x < s ∧ 0 < pt.y ∧ pt.y < s }

-- Theorem statement based on the final answer derived
theorem num_points_120_not_80_ray_partitional (R : Set Point) (s : ℝ) :
  R = unit_square 1 → 
  (card { X | is_ray_partitional R X 120 } - card { X | is_ray_partitional R X 80 }) = 900 :=
by
  sorry

end num_points_120_not_80_ray_partitional_l647_647623


namespace distance_between_homes_l647_647249

-- Define the parameters
def maxwell_speed : ℝ := 4  -- km/h
def brad_speed : ℝ := 6     -- km/h
def maxwell_time_to_meet : ℝ := 2  -- hours
def brad_start_delay : ℝ := 1  -- hours

-- Definitions related to the timings
def brad_time_to_meet : ℝ := maxwell_time_to_meet - brad_start_delay  -- hours

-- Define the distances covered by each
def maxwell_distance : ℝ := maxwell_speed * maxwell_time_to_meet  -- km
def brad_distance : ℝ := brad_speed * brad_time_to_meet  -- km

-- Define the total distance between their homes
def total_distance : ℝ := maxwell_distance + brad_distance  -- km

-- Statement to prove
theorem distance_between_homes : total_distance = 14 :=
by
  -- The proof is omitted; add 'sorry' to indicate this.
  sorry

end distance_between_homes_l647_647249


namespace find_d_e_f_l647_647229

noncomputable def y : ℝ := real.sqrt ((real.sqrt 73 / 3) + (5 / 3))

theorem find_d_e_f : ∃ d e f : ℕ,
  (y ^ 52 = 3 * y ^ 50 + 10 * y ^ 48 + 25 * y ^ 46 - y ^ 26 + d * y ^ 22 + e * y ^ 20 + f * y ^ 18) ∧
  (d + e + f = 184) :=
sorry

end find_d_e_f_l647_647229


namespace sarah_initial_bake_l647_647677

theorem sarah_initial_bake (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (initial_cupcakes : ℕ)
  (h1 : todd_ate = 14)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 8)
  (h4 : packages * cupcakes_per_package + todd_ate = initial_cupcakes) :
  initial_cupcakes = 38 :=
by sorry

end sarah_initial_bake_l647_647677


namespace smallest_possible_c_value_l647_647501

def smallest_possible_c (n : ℕ) : ℝ :=
  Real.cot (Real.pi / (2 * (n - 1)))

def polynomial_root_conditions (P Q : ℂ[X]) (c : ℝ) : Prop :=
  (∀ z : ℂ, z ∈ root_set Q → |z.im| ≤ c * |z.re|)

noncomputable def exists_transformed_polynomial (n : ℕ) : Prop :=
  ∀ (P : ℂ[X]), monic P ∧ polynomial_degree P = n →
  ∃ Q : ℂ[X], (∃ (coeffs : List (ℂ → ℂ)), 
  (∀ coeff, coeff ∈ coeffs → (Q = P.eval_map (λ x, if x ∈ coeffs then -1 * x else x))) ∧ polynomial_root_conditions P Q (smallest_possible_c n))

theorem smallest_possible_c_value :
  exists_transformed_polynomial 2012 :=
sorry

end smallest_possible_c_value_l647_647501


namespace sphere_plane_shape_l647_647595

structure SphericalCoordinates :=
  (ρ : ℝ)   -- Radial distance
  (θ : ℝ)   -- Azimuthal angle around the z-axis from the positive x-axis
  (φ : ℝ)   -- Polar angle from the positive z-axis

def shape_described (coords : SphericalCoordinates) : Prop :=
  coords.φ = coords.θ

theorem sphere_plane_shape (coords : SphericalCoordinates) :
  shape_described coords → ∃ P : ℝ³ → Prop, ∀ p, P p :=
sorry

end sphere_plane_shape_l647_647595


namespace cos_value_l647_647873

variable (α : ℝ)

theorem cos_value (h : Real.sin (Real.pi / 6 + α) = 1 / 3) : Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l647_647873


namespace interest_years_eq_three_l647_647391

theorem interest_years_eq_three :
  ∀ (x y : ℝ),
    (x + 1720 = 2795) →
    (x * (3 / 100) * 8 = 1720 * (5 / 100) * y) →
    y = 3 :=
by
  intros x y hsum heq
  sorry

end interest_years_eq_three_l647_647391


namespace sum_possible_values_of_100_frac_parts_eq_145_l647_647043

def floor_part (x : ℝ) : ℤ := ⌊x⌋ -- Definition of floor function
def frac_part (x : ℝ) : ℝ := x - floor_part x -- Definition of fractional part

def sum_frac_parts (n : ℝ) (h₁ : 0 ≤ frac_part n) (h₂ : frac_part n < 1) 
                   (h₃ : 0 ≤ frac_part (3 * n)) (h₄ : frac_part (3 * n) < 1)
                   (h₅ : frac_part n + frac_part (3 * n) = 1.4) : ℝ :=
  100 * frac_part n + 100 * frac_part (3 * n)

theorem sum_possible_values_of_100_frac_parts_eq_145 :
  ∃ n : ℝ, 
    0 ≤ frac_part n ∧ frac_part n < 1 ∧
    0 ≤ frac_part (3 * n) ∧ frac_part (3 * n) < 1 ∧
    frac_part n + frac_part (3 * n) = 1.4 ∧
    sum_frac_parts n sorry sorry sorry sorry sorry = 145 :=
by 
  sorry

end sum_possible_values_of_100_frac_parts_eq_145_l647_647043


namespace volumes_order_V4_between_V2_V3_V5_between_V3_V1_l647_647324

-- Definitions of volumes based on given radii and heights
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Given dimensions and volumes of the three cylinders
def R1 : ℝ := 10
def h1 : ℝ := 10
def V1 : ℝ := volume R1 h1

def R2 : ℝ := 5
def h2 : ℝ := 10
def V2 : ℝ := volume R2 h2

def R3 : ℝ := 5
def h3 : ℝ := 20
def V3 : ℝ := volume R3 h3

-- Proof problem: Order of volumes
theorem volumes_order : V2 < V3 ∧ V3 < V1 := by
  sorry

-- Given radius and height for V4 such that V2 < V4 < V3
def R4 : ℝ := 5
def h4 : ℝ := 15
def V4 : ℝ := volume R4 h4

theorem V4_between_V2_V3 : V2 < V4 ∧ V4 < V3 := by
  sorry

-- Given radius and height for V5 such that V3 < V5 < V1
def R5 : ℝ := 8
def h5 : ℝ := 10
def V5 : ℝ := volume R5 h5

theorem V5_between_V3_V1 : V3 < V5 ∧ V5 < V1 := by
  sorry

end volumes_order_V4_between_V2_V3_V5_between_V3_V1_l647_647324


namespace number_of_H2O2_combined_l647_647080

variables (SO2 H2O2 H2SO4 : Type) -- declare types for the chemical substances
variable (moles : Type) -- declare a type for moles
variables (initial_SO2 initial_H2O2 final_H2SO4 : moles) -- declare variables for initial and final moles

-- Define initial conditions
axiom initial_condition_SO2 : initial_SO2 = 1 -- 1 mole of SO2
axiom initial_condition_H2O2 : ∃ (some_H2O2 : moles), true -- some moles of H2O2
axiom final_condition_H2SO4 : final_H2SO4 = 1 -- resulting in 1 mole of H2SO4

-- Now, let's formalize the reaction condition:
axiom reaction_condition : initial_SO2 + initial_H2O2 = final_H2SO4

-- Statement to prove
theorem number_of_H2O2_combined (initial_SO2 initial_H2O2 final_H2SO4 : moles) :
  initial_condition_SO2 ∧ final_condition_H2SO4 ∧ reaction_condition →
  initial_H2O2 = 1 :=
by
  sorry

end number_of_H2O2_combined_l647_647080


namespace product_of_last_two_digits_div_by_6_and_sum_15_l647_647921

theorem product_of_last_two_digits_div_by_6_and_sum_15
  (n : ℕ)
  (h1 : n % 6 = 0)
  (A B : ℕ)
  (h2 : n % 100 = 10 * A + B)
  (h3 : A + B = 15)
  (h4 : B % 2 = 0) : 
  A * B = 54 := 
sorry

end product_of_last_two_digits_div_by_6_and_sum_15_l647_647921


namespace rachel_math_homework_pages_l647_647675

-- Define the number of pages of math homework and reading homework
def pagesReadingHomework : ℕ := 4

theorem rachel_math_homework_pages (M : ℕ) (h1 : M + 1 = pagesReadingHomework) : M = 3 :=
by
  sorry

end rachel_math_homework_pages_l647_647675


namespace number_of_female_adults_eq_60_l647_647215

-- Definitions and conditions
def number_of_children : ℕ := 80
def number_of_male_adults : ℕ := 60
def total_number_of_people : ℕ := 200

-- Theorem statement
theorem number_of_female_adults_eq_60 : 
  total_number_of_people - (number_of_children + number_of_male_adults) = 60 :=
by
  unfold total_number_of_people number_of_children number_of_male_adults
  -- subtracting the sum of children and male adults from the total number of people
  exact calculate_total_number_of_people - (80 + 60) = 60

sorry

end number_of_female_adults_eq_60_l647_647215


namespace trey_total_time_is_two_hours_l647_647661

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end trey_total_time_is_two_hours_l647_647661


namespace find_k_perpendicular_l647_647902

theorem find_k_perpendicular :
  ∀ (k : ℝ), (let a := (Real.sqrt 3, 1 : ℝ) in
              let b := (0, 1 : ℝ) in
              let c := (k, Real.sqrt 3 : ℝ) in
              let sum := (a.1 + 2 * b.1, a.2 + 2 * b.2) in
              sum.1 * c.1 + sum.2 * c.2 = 0
              → k = -3) :=
by
  intro k
  let a := (Real.sqrt 3, 1 : ℝ)
  let b := (0, 1 : ℝ)
  let c := (k, Real.sqrt 3 : ℝ)
  let sum := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  have h : sum.1 * c.1 + sum.2 * c.2 = 0 := by sorry
  sorry

end find_k_perpendicular_l647_647902


namespace angle_ABD_78_l647_647942

theorem angle_ABD_78 {A B C D E : Type*}
  [convex_quadrilateral A B C D]
  (h1 : ∠BAD = 90)
  (h2 : ∠BCD = 90)
  (h3 : BC = CD)
  (h4 : ∠AED = 123)
  (h5 : E = intersection (diagonal AC) (diagonal BD)) :
  ∠ABD = 78 := sorry

end angle_ABD_78_l647_647942


namespace binomial_coefficient_sum_l647_647837

theorem binomial_coefficient_sum (n : ℕ) :
  (1 / 2) * (∑ i in Finset.range (n / 2 + 1), (n - 2 * i) * Nat.choose n i) =
  n * ((n - 1) * Nat.choose (n - 1) (n / 2) - 1) / 2 :=
by sorry

end binomial_coefficient_sum_l647_647837


namespace problem_proof_l647_647634

theorem problem_proof (p q : ℝ) 
  (h1 : (∃ a b, {(a, b)} ⊆ {(x : ℝ) | (x + p) * (x + q) * (x - 8) = 0 ∧ x ≠ 5 ∧ x ≠ b} ∧ a ≠ b))
  (h2 : (∃ a b, {(a, b)} ⊆ {(x : ℝ) | (x + 2 * p) * (x - 5) * (x - 10) = 0 ∧ x ≠ q ∧ x ≠ 8} ∧ a ≠ 10 ∧ b ≠ 10 ∧ a ≠ b)) :
  50 * p + q = 240 := 
by {
  sorry
}

end problem_proof_l647_647634


namespace discount_received_l647_647394

theorem discount_received (original_cost : ℝ) (amt_spent : ℝ) (discount : ℝ) 
  (h1 : original_cost = 467) (h2 : amt_spent = 68) : 
  discount = 399 :=
by
  sorry

end discount_received_l647_647394


namespace optimal_cut_for_6_day_payment_l647_647158

def chain_6_links := ["link 1", "link 2", "link 3", "link 4", "link 5", "link 6"]

def cut_link (chain : List String) (cut_position : Nat) : List (List String) :=
  [chain.take cut_position, chain.drop cut_position]

theorem optimal_cut_for_6_day_payment :
  ∃ cut_position : Nat, 
  chain_6_links = ["link 1", "link 2", "cut", "link 4", "link 5", "link 6"] →
  let chain_segments := cut_link chain_6_links cut_position in
  (∀ day : Nat, 1 ≤ day ∧ day ≤ 6 →
    let payment := match day with
                  | 1 => ["link 1"]
                  | 2 => ["link 1", "link 2"]
                  | 3 => ["link 1"]
                  | 4 => ["link 3"]
                  | 5 => ["link 1"]
                  | 6 => ["link 2"]
                  | _ => [] -- should not reach here
                  end
    ∃ segment : List String, segment ∈ chain_segments ∧ payment = segment) :=
sorry

end optimal_cut_for_6_day_payment_l647_647158


namespace sequence_formula_sum_formula_l647_647520

noncomputable def seq_an : ℕ → ℝ
| 0     := 1
| (n+1) := (1/2) * seq_an n

def sum_sn (n : ℕ) : ℝ :=
∑ i in finset.range n, seq_an i

theorem sequence_formula (n : ℕ) : seq_an n = (1/2)^(n-1) := by
  sorry

theorem sum_formula (n : ℕ) : sum_sn n = 2 - 2^(1-n) := by
  sorry

end sequence_formula_sum_formula_l647_647520


namespace angle_PSQ_l647_647760

theorem angle_PSQ (L M K N P Q S : Point) (c : Circle) 
  (hLMP : c.pass_through L) (hLMM : c.pass_through M)
  (hKP : c.intersect_side K L P) (hMQ : c.intersect_side M N Q)
  (hS : c.tangent_at S N) 
  (hLSM : ∠ L S M = 50) (hKLS_SN : ∠ K L S = ∠ S N M) : 
  ∠ P S Q = 65 := sorry

end angle_PSQ_l647_647760


namespace cadastral_value_of_land_l647_647412

theorem cadastral_value_of_land (tax_amount : ℝ) (tax_rate : ℝ) (V : ℝ)
    (h1 : tax_amount = 4500)
    (h2 : tax_rate = 0.003) :
    V = 1500000 :=
by
  sorry

end cadastral_value_of_land_l647_647412


namespace little_d_wins_for_all_n_l647_647423

/-- 
Infinite grid game where Little D loses a shoe on an unmunched point 
and Big Z munches a shoe-free plane. Prove that Little D can lose a shoe on 
each of n consecutive lattice points on a line parallel to one of the coordinate 
axes for any n.
-/
theorem little_d_wins_for_all_n : ∀ (n : ℕ), ∃ (points : list (ℤ × ℤ × ℤ)), 
  (points.length = n) ∧ 
  (∃ axis : ℕ, axis ∈ [0, 1, 2] ∧ 
    (∀ i : ℕ, i < n → 
      (axis = 0 → (points.nth i).is_some → (points.nth i).get_or_else (0,0,0) =
        (points.nth 0).get_or_else (0,0,0) + (i,0,0))) ∧
    (axis = 1 → (points.nth i).is_some → (points.nth i).get_or_else (0,0,0) = 
      (points.nth 0).get_or_else (0,0,0) + (0,i,0)) ∧
    (axis = 2 → (points.nth i).is_some → (points.nth i).get_or_else (0,0,0) = 
      (points.nth 0).get_or_else (0,0,0) + (0,0,i)))

end little_d_wins_for_all_n_l647_647423


namespace right_triangle_exists_among_four_points_l647_647830

theorem right_triangle_exists_among_four_points :
  ∃ (points : Fin 7 → (ℝ × ℝ)),
    ∀ (selection : Finset (Fin 7)),
      selection.card = 4 →
      ∃ a b c : Fin 7, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ selection ∧ b ∈ selection ∧ c ∈ selection ∧
        (∀ x y z, x ≠ y → y ≠ z → x ≠ z → x ∈ selection → y ∈ selection → z ∈ selection → 
          angle x y z = π/2 ∨ angle y z x = π/2 ∨ angle z x y = π/2) :=
sorry

end right_triangle_exists_among_four_points_l647_647830


namespace cot_inequality_l647_647221

noncomputable theory

-- Define angles α, β, γ and their properties of being acute
variables (α β γ : ℝ)
-- Assuming the angles are acute
axiom hα : 0 < α ∧ α < π / 2
axiom hβ : 0 < β ∧ β < π / 2
axiom hγ : 0 < γ ∧ γ < π / 2

-- Given condition: cos²(α) + cos²(β) + cos²(γ) = 1
axiom h_cos : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1

-- The statement to be proved: cot(β) * cot(γ) + cot(γ) * cot(α) + cot(α) * cot(β) ≤ 3/2
theorem cot_inequality : Real.cot β * Real.cot γ + Real.cot γ * Real.cot α + Real.cot α * Real.cot β ≤ 3 / 2 :=
sorry

end cot_inequality_l647_647221


namespace average_price_of_fruit_l647_647405

theorem average_price_of_fruit 
  (price_apple price_orange : ℝ)
  (total_fruits initial_fruits kept_oranges kept_fruits : ℕ)
  (average_price_kept average_price_initial : ℝ)
  (h1 : price_apple = 40)
  (h2 : price_orange = 60)
  (h3 : initial_fruits = 10)
  (h4 : kept_oranges = initial_fruits - 6)
  (h5 : average_price_kept = 50) :
  average_price_initial = 56 := 
sorry

end average_price_of_fruit_l647_647405


namespace parabola_vertex_y_axis_opens_upwards_l647_647738

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end parabola_vertex_y_axis_opens_upwards_l647_647738


namespace max_volume_tetrahedron_l647_647404

noncomputable def maximum_volume_of_tetrahedron (area: ℝ) : ℝ :=
  let a := sqrt ((2 : ℝ) / area)
  (a^2 - a^4 / 4)^(1 / 2) / 12

theorem max_volume_tetrahedron (a b : ℝ) (H_area : a * b = 2) :
  maximum_volume_of_tetrahedron 2 = (12 : ℝ)^(1/4) / 9 :=
sorry

end max_volume_tetrahedron_l647_647404


namespace QF_distance_l647_647552

open Set

-- Define the parabola C.
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F.
def F : ℝ × ℝ := (4, 0)

-- Define the directrix l.
def directrix (x y : ℝ) : Prop := x = 0

-- P is on the directrix.
def P_on_directrix (P : ℝ × ℝ) : Prop := directrix P.1 P.2

-- Q is the point of intersection between line PF and C.
def Q_on_intersection (P Q : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, m ≠ 0 ∧ b = P.2 - m * P.1 ∧
  (Q.1 * m + b = Q.2 ∧ parabola Q.1 Q.2)

-- Given the condition that |PF| = 4|FQ|.
def condition (P Q : ℝ × ℝ) : Prop :=
  let PF := (F.1 - P.1, F.2 - P.2)
  let FQ := (Q.1 - F.1, Q.2 - F.2)
  (PF.1^2 + PF.2^2) = 4 * (FQ.1^2 + FQ.2^2)

-- The length of QF.
def distance (P Q : ℝ × ℝ) : ℝ :=
  let FQ := (Q.1 - F.1, Q.2 - F.2)
  real.sqrt (FQ.1^2 + FQ.2^2)

theorem QF_distance {P Q : ℝ × ℝ} (h1 : P_on_directrix P) (h2 : Q_on_intersection P Q) (h3 : condition P Q) :
  distance P Q = 10 :=
sorry

end QF_distance_l647_647552


namespace correct_calculation_l647_647345

/-- Definitions for the calculations -/
def calcA := abs 2 * abs 2 = 1
def calcB := abs 4 - abs 3 = 1
def calcC := abs 6 / abs 3 = 2
def calcD := abs 4 = 2

/-- The main theorem -/
theorem correct_calculation : calcA ∧ ¬calcB ∧ ¬calcC ∧ ¬calcD := 
by {
  unfold calcA calcB calcC calcD,
  split,
  { show abs 2 * abs 2 = 1, sorry },
  split,
  { show ¬(abs 4 - abs 3 = 1), sorry },
  split,
  { show ¬(abs 6 / abs 3 = 2), sorry },
  { show ¬(abs 4 = 2), sorry }
}

end correct_calculation_l647_647345


namespace convert_to_rectangular_form_l647_647041

noncomputable def z : ℂ := √2 * exp (15 * real.pi * complex.i / 4)

theorem convert_to_rectangular_form : z = -1 + complex.i := by
  sorry

end convert_to_rectangular_form_l647_647041


namespace find_b_and_c_angle_between_m_and_n_l647_647554

variables {a b c m n : ℝ × ℝ} {x y : ℝ}

-- Given conditions
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (9, x)
def c : ℝ × ℝ := (4, y)

-- Proportional condition
def parallel_condition : Prop := 3 * x = 4 * 9

-- Perpendicular condition
def perpendicular_condition : Prop := 3 * 4 + 4 * y = 0

-- Define m and n
def m : ℝ × ℝ := 2 * a - b
def n : ℝ × ℝ := a + c

-- Angle calculation
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)
def cos_theta (u v : ℝ × ℝ) : ℝ := dot_product u v / (magnitude u * magnitude v)

theorem find_b_and_c (x_val : x = 12) (y_val : y = -3) : 
  b = (9, 12) ∧ c = (4, -3) :=
by { sorry }

theorem angle_between_m_and_n (x_val : x = 12) (y_val : y = -3) : 
  real.arccos (cos_theta m n) = (3 * real.pi / 4) :=
by { sorry }

end find_b_and_c_angle_between_m_and_n_l647_647554


namespace angle_C_in_triangle_l647_647961

open Real

noncomputable def determine_angle_C (A B C: ℝ) (AB BC: ℝ) : Prop :=
  A = (3 * π) / 4 ∧ BC = sqrt 2 * AB → C = π / 6

-- Define the theorem to state the problem
theorem angle_C_in_triangle (A B C : ℝ) (AB BC : ℝ) :
  determine_angle_C A B C AB BC := 
by
  -- Step to indicate where the proof would be
  sorry

end angle_C_in_triangle_l647_647961


namespace bucket_p_fill_time_l647_647031

theorem bucket_p_fill_time (capacity_P capacity_Q drum_capacity turns : ℕ)
  (h1 : capacity_P = 3 * capacity_Q)
  (h2 : drum_capacity = 45 * (capacity_P + capacity_Q))
  (h3 : bucket_fill_turns = drum_capacity / capacity_P) :
  bucket_fill_turns = 60 :=
by
  sorry

end bucket_p_fill_time_l647_647031


namespace length_of_A_l647_647985

theorem length_of_A'B' :
  let A := (0 : ℝ, 4 : ℝ),
      B := (0 : ℝ, 7 : ℝ),
      C := (3 : ℝ, 6 : ℝ)
  in
  ∃ A' B' : ℝ × ℝ,
    A'.1 = A'.2 ∧ B'.1 = B'.2 ∧
    line_through A C ∧ line_through B C ∧
    distance A' B' = 6.75 * real.sqrt 2 :=
begin
  sorry
end

end length_of_A_l647_647985


namespace angle_D_l647_647965

-- Definitions based on conditions
variables (A B C D E F O D' : Type)
variables [geometry A] [geometry B] [geometry C] [geometry D] [geometry E] [geometry F] [geometry O] [geometry D']

-- Assumptions
hypothesis (h1 : ∃ (ABC : triangle), 
                 interior_angle_bisector ABC = D ∧ 
                 exterior_angle_bisector ABC = E ∧ 
                 circumcircle_intersection ABC D = F ∧ 
                 circumcenter ABC = O ∧ 
                 reflection D O = D')

-- Question to prove
theorem angle_D'_FE_is_90 : ∠D' F E = 90 :=
by sorry

end angle_D_l647_647965


namespace pyramid_volume_eq_sqrt2_div3_l647_647002

noncomputable def volume_of_pyramid (base_edge_length side_edge_length : ℝ) : ℝ :=
  (1 / 3 : ℝ) * (sqrt 3) * side_edge_length

theorem pyramid_volume_eq_sqrt2_div3 :
  ∀ (base_edge_length side_edge_length : ℝ), base_edge_length = 2 ∧ side_edge_length = sqrt 2 → 
  volume_of_pyramid base_edge_length side_edge_length = sqrt 2 / 3 :=
by
  intros base_edge_length side_edge_length h
  cases h
  rw [h_left, h_right]
  sorry

end pyramid_volume_eq_sqrt2_div3_l647_647002


namespace total_marks_calculation_l647_647940

def student_marks (correct_count total_count mark_correct mark_incorrect : ℕ) : ℤ :=
  (correct_count * mark_correct) + ((total_count - correct_count) * mark_incorrect)

theorem total_marks_calculation :
  student_marks 42 60 4 (-1) = 150 :=
by
  sorry

end total_marks_calculation_l647_647940


namespace smallest_repeating_block_6_over_7_l647_647157

theorem smallest_repeating_block_6_over_7 : 
  ∃ d : ℕ, periodic_decimal_expansion 6 7 d ∧ d = 6 :=
sorry

end smallest_repeating_block_6_over_7_l647_647157


namespace binary_predecessor_l647_647160

theorem binary_predecessor (N : ℕ) (hN : N = 0b11000) : 0b10111 + 1 = N := 
by
  sorry

end binary_predecessor_l647_647160


namespace solve_for_x_l647_647732

theorem solve_for_x (x : ℝ) (h : (25 / 75) = (x / 75)^3) : x = 75 / (real.cbrt 3) :=
by
  sorry

end solve_for_x_l647_647732


namespace find_a_l647_647877

theorem find_a
  (h_roots : ∃ x : ℝ, x = (1 + sqrt 3) / 2 ∨ x = (1 - sqrt 3) / 2)
  (h_eq : ∀ x : ℝ, (x = (1 + sqrt 3) / 2 ∨ x = (1 - sqrt 3) / 2) → a * x^2 - x - 1 / 2 = 0) :
  a = 1 := 
sorry

end find_a_l647_647877


namespace evaluate_expression_l647_647415

theorem evaluate_expression :
  abs (-1 / 2) + real.cbrt (-27) - real.sqrt (1 / 4) + real.sqrt 12 * real.sqrt 3 = 3 := 
sorry

end evaluate_expression_l647_647415


namespace perfect_square_condition_l647_647159

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = k^2) ↔ m = 196 :=
by sorry

end perfect_square_condition_l647_647159


namespace more_supermarkets_in_us_l647_647711

-- Definitions based on conditions
def total_supermarkets : ℕ := 84
def us_supermarkets : ℕ := 47
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

-- Prove that the number of more FGH supermarkets in the US than in Canada is 10
theorem more_supermarkets_in_us : us_supermarkets - canada_supermarkets = 10 :=
by
  -- adding 'sorry' as the proof
  sorry

end more_supermarkets_in_us_l647_647711


namespace water_remaining_l647_647371

theorem water_remaining (initial_water : ℕ) (evap_rate : ℕ) (days : ℕ) : 
  initial_water = 500 → evap_rate = 1 → days = 50 → 
  initial_water - evap_rate * days = 450 :=
by
  intros h₁ h₂ h₃
  sorry

end water_remaining_l647_647371


namespace specific_numbers_have_property_l647_647250

def second_last_even (n : Nat) : Prop :=
  ∀ k : Nat, let prod := (n^(k+1)) in
             (prod / 10) % 2 = 0

def has_property : Set Nat :=
  {n | second_last_even n}

theorem specific_numbers_have_property :
  ∀ n, n ∈ {1, 3, 7, 9, 5, 0} → second_last_even n :=
by
  intros n hn
  sorry

end specific_numbers_have_property_l647_647250


namespace sum_k_f_k_minus_half_l647_647131

noncomputable def f : ℝ → ℝ := sorry

axiom domain_all_real : ∀ x : ℝ, f x = f x

axiom f_add_period_2 (x : ℝ) : f (x + 2) + f x = f 8

axiom f_2x1_odd (x : ℝ) : f (2 * x + 1) = -f (-2 * x - 1)

axiom f_half : f (1 / 2) = 1 / 2

theorem sum_k_f_k_minus_half :
  (∑ k in Finset.range 22 \ k + 1, k • f (k - 1 / 2)) = -1 / 2 := 
sorry

end sum_k_f_k_minus_half_l647_647131


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647819

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l647_647819


namespace last_digit_of_closest_integer_to_a2023_is_4_l647_647006

theorem last_digit_of_closest_integer_to_a2023_is_4 :
  let a : Nat → ℚ := λ n, if n = 1 then 5 / 2 else nat.recOn n (5 / 2) 
    (λ n a_n, a_n^2 - 2)
  ∃ M : ℤ, closest_to M (a 2023) ∧ (M % 10 = 4) := 
sorry

end last_digit_of_closest_integer_to_a2023_is_4_l647_647006


namespace M_is_incenter_l647_647446

variable (A B C M : Type) [triangle A B C] 

-- Define the problem conditions
structure Circle (P Q : Type) :=
(center : Type)
(radius : ℝ)

noncomputable def circle_on_segment (M A : Type) : Circle M A := sorry

-- Specify the property of M we need to prove
def isIncenter (M : Type) (A B C : Type) [inhabited M] : Prop := 
  ∀ P Q R : Type,
  (circle_on_segment M A).common_chord_length = (circle_on_segment M B).common_chord_length ∧
  (circle_on_segment M B).common_chord_length = (circle_on_segment M C).common_chord_length

-- The theorem statement
theorem M_is_incenter 
  (A B C M : Type) [triangle A B C] [inhabited M] :
  (isIncenter M A B C) → Incenter M A B C :=
sorry

end M_is_incenter_l647_647446


namespace angle_D_l647_647964

-- Definitions based on conditions
variables (A B C D E F O D' : Type)
variables [geometry A] [geometry B] [geometry C] [geometry D] [geometry E] [geometry F] [geometry O] [geometry D']

-- Assumptions
hypothesis (h1 : ∃ (ABC : triangle), 
                 interior_angle_bisector ABC = D ∧ 
                 exterior_angle_bisector ABC = E ∧ 
                 circumcircle_intersection ABC D = F ∧ 
                 circumcenter ABC = O ∧ 
                 reflection D O = D')

-- Question to prove
theorem angle_D'_FE_is_90 : ∠D' F E = 90 :=
by sorry

end angle_D_l647_647964


namespace integer_ratio_l647_647693

theorem integer_ratio (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 16)
  (h2 : A % B = 0) (h3 : B = C - 2) (h4 : D = 2) (h5 : A ≠ B) (h6 : B ≠ C) (h7 : C ≠ D) (h8 : D ≠ A)
  (h9: 0 < A) (h10: 0 < B) (h11: 0 < C):
  A / B = 28 := 
sorry

end integer_ratio_l647_647693


namespace no_silver_matrix_for_1997_infinitely_many_silver_matrices_l647_647400

def is_silver_matrix (n : ℕ) (A : matrix (fin n) (fin n) ℕ) : Prop :=
  let S := finset.range (2 * n - 1 + 1)
  ∀ i : fin n, ((finset.univ.image (λ j : fin n, A i j)) ∪ (finset.univ.image (λ j : fin n, A j i))) = S

theorem no_silver_matrix_for_1997 : ¬ ∃ (A : matrix (fin 1997) (fin 1997) ℕ), is_silver_matrix 1997 A :=
sorry

theorem infinitely_many_silver_matrices : ∃ᶠ (n : ℕ) in filter.at_top, ∃ (A : matrix (fin n) (fin n) ℕ), is_silver_matrix n A :=
sorry

end no_silver_matrix_for_1997_infinitely_many_silver_matrices_l647_647400


namespace expectation_equality_variance_inequality_l647_647364

noncomputable def X1_expectation : ℚ :=
  2 * (2 / 5 : ℚ)

noncomputable def X1_variance : ℚ :=
  2 * (2 / 5) * (1 - 2 / 5)

noncomputable def P_X2_0 : ℚ :=
  (3 * 2) / (5 * 4)

noncomputable def P_X2_1 : ℚ :=
  (2 * 3) / (5 * 4)

noncomputable def P_X2_2 : ℚ :=
  (2 * 1) / (5 * 4)

noncomputable def X2_expectation : ℚ :=
  0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2

noncomputable def X2_variance : ℚ :=
  P_X2_0 * (0 - X2_expectation)^2 + P_X2_1 * (1 - X2_expectation)^2 + P_X2_2 * (2 - X2_expectation)^2

theorem expectation_equality : X1_expectation = X2_expectation :=
  by sorry

theorem variance_inequality : X1_variance > X2_variance :=
  by sorry

end expectation_equality_variance_inequality_l647_647364


namespace min_points_to_secure_top2_l647_647714

theorem min_points_to_secure_top2 (n_teams : ℕ) (total_matches : ℕ) (max_points : ℕ) : 
  n_teams = 4 → 
  total_matches = nat.choose 4 2 →
  max_points = 18 →
  ∀ points : ℕ, (∀ team_points : list ℕ, list.sum team_points = max_points → 
  ∀ t1 t2, t1 ≠ t2 → team_points.nth_le t1 sorry ≤ team_points.nth_le t2 sorry → 
  team_points.nth_le t2 sorry = 7) → 
  points = 7 :=
by
  sorry

end min_points_to_secure_top2_l647_647714


namespace susan_walking_distance_l647_647431

/-- Define the variables for walking speeds and times -/
def walking_speeds_conditions (S : ℝ) (E : ℝ) (D : ℝ) : Prop :=
  (E + S + D = 6) ∧ (D = S / 2) ∧ (E = S - 3)

/-- Define the constants for time and combined speed -/
def walking_time : ℝ := 8 / 3

/-- Define the condition of equal time spent walking -/
def equal_time_condition : Prop := walking_time * 3 = 8

/-- The main theorem statement -/
theorem susan_walking_distance (S E D : ℝ) (h1 : walking_speeds_conditions S E D) 
  (h2 : equal_time_condition) : distance_susan_walked S := 
  S * walking_time = 9.6 :=
begin
  sorry,
end

end susan_walking_distance_l647_647431


namespace urn_probability_l647_647794

-- Define the initial count of red and blue balls in the urn
def init_red := 2
def init_blue := 1

-- Define the total number of iterations
def iterations := 5

-- Define the final counts we are interested in
def final_red := 4
def final_blue := 4

-- Prove the probability of reaching the final state after the specified operations
theorem urn_probability : 
  (probability_after_iterations init_red init_blue iterations final_red final_blue) = (8 / 21) :=
by
  sorry

end urn_probability_l647_647794


namespace range_of_a_l647_647104

def f (x : ℝ) : ℝ := 1 / x^2 + 1 / x^4

theorem range_of_a (a : ℝ) : f (a - 2) < f (2 * a + 1) →
  (a ∈ set.Ioo (-3 : ℝ) (-1 / 2) ∪ set.Ioo (-1 / 2) (1 / 3)) :=
by
  sorry

end range_of_a_l647_647104


namespace age_ratio_l647_647576

theorem age_ratio (B A : ℕ) (h1 : B = 4) (h2 : A - B = 12) :
  A / B = 4 :=
by
  sorry

end age_ratio_l647_647576


namespace find_A_values_l647_647516

def is_sum_symmetric (N : ℕ) : Prop :=
  let d1 := N / 1000 % 10
  let d2 := N / 100 % 10
  let d3 := N / 10 % 10
  let d4 := N % 10
  d1 + d2 = d3 + d4

noncomputable def F (N : ℕ) : ℕ :=
  let d1 := N / 1000 % 10
  let d2 := N / 100 % 10
  let d3 := N / 10 % 10
  let d4 := N % 10
  let N' := 1000 * d2 + 100 * d1 + 10 * d4 + d3
  (N + N') / 101

theorem find_A_values :
  ∀ (a b m n : ℕ),
    (3 ≤ a ∧ a ≤ 8) →
    (0 ≤ b ∧ b ≤ 5) →
    (2 ≤ m ∧ m <= 9) →
    (5 ≤ n ∧ n ≤ 12) →
    b = a - 3 →
    n = m - 6 →
    let A := 1000 * a + 10 * b + 746,
    let B := 100 * m + n + 2026,
    is_sum_symmetric A →
    is_sum_symmetric B →
    77 ∣ (3 * F A + 2 * F B) →
    A ∈ {3746, 4756, 6776, 5766, 7786, 8796} :=
by
  sorry

end find_A_values_l647_647516


namespace larry_sixth_finger_l647_647615

-- Define the function f as given by the conditions
def f : ℕ → ℕ
| 2 := 1
| 1 := 8
| 8 := 7
| 7 := 2
| _ := 0  -- This handles any undefined inputs, for completeness

-- Define the sequence of numbers written on each finger, starting with 2
def sequence_n : ℕ → ℕ
| 1 := 2
| (n+1) := f (sequence_n n)

-- The theorem stating the required property
theorem larry_sixth_finger : sequence_n 6 = 1 := by
  sorry

end larry_sixth_finger_l647_647615


namespace projection_of_a_plus_b_on_a_l647_647905

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, real.sqrt 2)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-3, 2, 0)

noncomputable def dot_product (x y : ℝ × ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def vector_add (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (x.1 + y.1, x.2 + y.2, x.3 + y.3)

theorem projection_of_a_plus_b_on_a :
  let v := vector_add vector_a vector_b in
  (dot_product v vector_a) / (vector_magnitude vector_a) = 3 / 2 :=
by
  sorry

end projection_of_a_plus_b_on_a_l647_647905


namespace increasing_function_in_Rplus_l647_647791

theorem increasing_function_in_Rplus (f : ℝ → ℝ) :
  (f = (λ x, x^2 + 4 * x + 3) ∨
   f = (λ x, -3 * x + 1) ∨
   f = (λ x, (2 : ℝ) / x) ∨
   f = (λ x, x^2 - 4 * x + 3)) →
  (∀ x ∈ set.Ioi 0, monotone f) ↔ (f = (λ x, x^2 + 4 * x + 3)) :=
by
  intro h
  sorry

end increasing_function_in_Rplus_l647_647791


namespace calculate_expression_l647_647853

def star (a b : ℝ) : ℝ :=
if h : a ≠ 0 then
  if a > b then a^b
  else a^(-b)
else 0

theorem calculate_expression : (star 2 (-4)) * (star (-4) (-2)) = 1 := by
  sorry

end calculate_expression_l647_647853


namespace sequence_a_n_sum_T_n_l647_647997

noncomputable def a_n (n : ℕ) : ℚ := ((- (1/4) : ℚ) ^ n)

def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ i, a_n (i + 1))

def b_n (n : ℕ) : ℚ := real.log (real.abs (1 / a_n n)) / real.log 4

def T_n (n : ℕ) : ℚ := (finset.range n).sum (λ i, (1 / (b_n i * b_n (i + 1))))

theorem sequence_a_n (n : ℕ) :
  a_n n = (- (1/4) : ℚ) ^ n :=
sorry

theorem sum_T_n (n : ℕ) :
  T_n n = (n : ℚ) / (n + 1) :=
sorry

end sequence_a_n_sum_T_n_l647_647997


namespace angle_between_vectors_l647_647625

open Real

def vec_a : ℝ³ := ⟨3, -1, 4⟩
def vec_b : ℝ³ := ⟨-1, 6, 1⟩
def vec_c : ℝ³ := ⟨2, 1, -2⟩

noncomputable def dot_product (v w : ℝ³) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def magnitude (v : ℝ³) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def new_vector :=
  (dot_product vec_b vec_c) • vec_a - (dot_product vec_a vec_b) • vec_c

noncomputable def cos_theta :=
  (dot_product vec_a new_vector) / (magnitude vec_a * magnitude new_vector)

noncomputable def theta :=
  real.arccos cos_theta

theorem angle_between_vectors :
  theta = real.arccos ((37 : ℝ) / (sqrt 26 * sqrt 269)) :=
by
  sorry

end angle_between_vectors_l647_647625


namespace smallest_n_is_89_l647_647521

variable (n : ℕ)
variable (x : Fin n → ℕ)
variable (S : ℕ → Finset (Fin n))
variable (sum_x : ∑ i, x i = 2009)
variable (partition_41 : ∀ i, i < 41 → ∃ s : Finset (Fin n), s = S i ∧ (∑ j in s, x j) = 49)
variable (partition_49 : ∀ j, j < 49 → ∃ s : Finset (Fin n), s = S (j + 41) ∧ (∑ k in s, x k) = 41)

theorem smallest_n_is_89 :
  n = 89 :=
begin
  sorry
end

end smallest_n_is_89_l647_647521


namespace circle_radius_distance_eq_one_l647_647083

theorem circle_radius_distance_eq_one :
  ∃ r : ℝ, r > 0 ∧ (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
  abs (4 * x - 3 * y - 2) / real.sqrt (4^2 + 3^2) = 1) → r = 4 :=
by
  sorry

end circle_radius_distance_eq_one_l647_647083


namespace find_probability_l647_647111

noncomputable def ζ : Type := sorry -- Define the random variable ζ with its properties

-- Lean statement encapsulating the problem conditions
def normal_distribution_conditions (ζ : ℝ) : Prop :=
  ∃ (μ σ : ℝ), (μ = 2) ∧ (σ = 2) ∧ (ζ follows_normal_distrib μ σ) ∧ (prob_less_than ζ 4 = 0.8)

-- Statement of the proof problem
theorem find_probability (ζ : ℝ) (h : normal_distribution_conditions ζ) :
  prob_interval ζ 0 2 = 0.3 :=
sorry

end find_probability_l647_647111


namespace problem_statement_l647_647103

variable {α : Type*} [LinearOrderedCommRing α]

theorem problem_statement (a b c d e : α) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : a * b^2 * c * d^4 * e < 0 :=
by
  sorry

end problem_statement_l647_647103


namespace exists_invalid_dot_product_vectors_l647_647996

noncomputable def vector_is_nonzero {α : Type*} [Field α] (v : α × α) : Prop :=
(v ≠ (0, 0))

noncomputable def dot_product {α : Type*} [Field α] (v1 v2 : α × α) : α :=
v1.1 * v2.1 + v1.2 * v2.2

noncomputable def vectors_coord (a b c : ℝ × ℝ) : Prop :=
(a = (1, 0)) ∧ (b = (0, 1)) ∧ (c = (0, -1))

noncomputable def valid_vectors (a b c : ℝ × ℝ) : Prop :=
(vector_is_nonzero a) ∧ (vector_is_nonzero b) ∧ (vector_is_nonzero c) ∧
(dot_product a b = dot_product a c) ∧ (b ≠ c)

theorem exists_invalid_dot_product_vectors : 
  ∃ a b c : ℝ × ℝ, valid_vectors a b c ∧ vectors_coord a b c :=
begin
  sorry,
end

end exists_invalid_dot_product_vectors_l647_647996


namespace sum_of_squares_and_product_l647_647709

theorem sum_of_squares_and_product (x y : ℕ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y = Real.sqrt 202 := 
by
  sorry

end sum_of_squares_and_product_l647_647709


namespace product_of_chord_lengths_l647_647619

noncomputable def omega : ℂ := Complex.exp(2 * Real.pi * Complex.I / 10)

noncomputable def A : ℂ := 3
noncomputable def B : ℂ := -3
noncomputable def C (k : ℕ) : ℂ := 3 * omega^k

noncomputable def length_AC (k : ℕ) : ℝ := Complex.abs (A - C k)
noncomputable def length_BC (k : ℕ) : ℝ := Complex.abs (B - C k)

theorem product_of_chord_lengths :
  let lengths := [length_AC 1, length_AC 2, length_AC 3, length_AC 4,
                  length_BC 1, length_BC 2, length_BC 3, length_BC 4]
  (list.prod lengths) = 590490 :=
by sorry

end product_of_chord_lengths_l647_647619


namespace meaningful_expression_range_l647_647169

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = (sqrt (x + 3)) / x) ↔ x ≥ -3 ∧ x ≠ 0 := 
sorry

end meaningful_expression_range_l647_647169


namespace inequality_range_l647_647505

theorem inequality_range (k : ℝ) : (∀ x : ℝ, abs (x + 1) - abs (x - 2) > k) → k < -3 :=
by
  sorry

end inequality_range_l647_647505


namespace min_f_value_f_achieves_min_l647_647499

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x ^ 2 + 1) + (x * (x + 3)) / (x ^ 2 + 2) + (3 * (x + 1)) / (x * (x ^ 2 + 2))

theorem min_f_value (x : ℝ) (hx : x > 0) : f x ≥ 3 :=
sorry

theorem f_achieves_min (x : ℝ) (hx : x > 0) : ∃ x, f x = 3 :=
sorry

end min_f_value_f_achieves_min_l647_647499


namespace Mark_has_10_fewer_cards_l647_647247

-- Define the constants and variables
def L : ℕ := sorry
def M : ℕ := 3 * L
def Mi : ℕ := 100

-- Main statement to prove
theorem Mark_has_10_fewer_cards:
  L + M + Mi + 80 = 300 →
  Mi - M = 10 :=
by
  -- The detailed proof is omitted, but this outline should give the correct framework
  intros h_total,
  sorry

end Mark_has_10_fewer_cards_l647_647247


namespace fourier_series_decomposition_l647_647424

open Real

noncomputable def f : ℝ → ℝ :=
  λ x => if (x < 0) then -1 else (if (0 < x) then 1/2 else 0)

theorem fourier_series_decomposition :
    ∀ x, -π ≤ x ∧ x ≤ π →
         f x = -1/4 + (3/π) * ∑' k, (sin ((2*k+1)*x)) / (2*k+1) :=
by
  sorry

end fourier_series_decomposition_l647_647424


namespace number_of_incorrect_props_l647_647226

variable (l m n : Type) [LinearOrder l] [LinearOrder m] [LinearOrder n]
variable (α β : Type) [Plane α] [Plane β]

-- Define the propositions
def prop1 (n : Type) (α β : Type) [Perpendicular n α] [Perpendicular n β]: Prop :=
  (α ∥ β) ↔ (n ⊥ β)

def prop2 (m n l : Type) (α : Type) [Subset m α] [Projection n l α] [Perpendicular m n] : Prop :=
  (m ⊥ n) → (l ⊥ m)

def prop3 (m : Type) (α β : Type) [Subset m α] [Perpendicular m β] : Prop :=
  (m ⊥ β) → (α ⊥ β)

def prop4 (m n : Type) (α : Type) [Subset m α] [NotSubset n α] [Parallel n α] : Prop :=
  (n ∥ α) ↔ (m ∥ n)

-- The Lean 4 statement for the proof problem
theorem number_of_incorrect_props :
  (¬ prop2 m n l α ∧ ¬ prop4 m n α) →
  (number_of_incorrect_prop = 2) :=
sorry

end number_of_incorrect_props_l647_647226


namespace avg_b_c_weight_l647_647290

theorem avg_b_c_weight (a b c : ℝ) (H1 : (a + b + c) / 3 = 45) (H2 : (a + b) / 2 = 40) (H3 : b = 39) : (b + c) / 2 = 47 :=
by
  sorry

end avg_b_c_weight_l647_647290


namespace quadrant_of_angle_l647_647162

theorem quadrant_of_angle (α : ℝ) (h : tan α * cos α < 0) : 
  (π < α ∧ α < 3 * π / 2) ∨ (3 * π / 2 < α ∧ α < 2 * π) :=
sorry

end quadrant_of_angle_l647_647162


namespace binomial_expansion_terms_largest_binomial_coefficient_l647_647881

theorem binomial_expansion_terms (n : ℕ) (h : (x - 2)^n.has_terms 7) : n = 6 := sorry

theorem largest_binomial_coefficient (n : ℕ) (h : n = 6) : 
  ∃ k : ℕ, largest_coefficient ((x - 2)^n) k = -160 := sorry

end binomial_expansion_terms_largest_binomial_coefficient_l647_647881


namespace sequence_limit_l647_647617

def sequence (n : ℕ) : ℝ
| 0       := 25
| (n + 1) := Real.arctan (sequence n)

theorem sequence_limit :
  ∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence n - L| < ε) ∧ L = 0 :=
by
  sorry

end sequence_limit_l647_647617


namespace find_n_l647_647493

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l647_647493


namespace tv_sets_sales_decrease_l647_647663

theorem tv_sets_sales_decrease
  (P Q P' Q' R R': ℝ)
  (h1 : P' = 1.6 * P)
  (h2 : R' = 1.28 * R)
  (h3 : R = P * Q)
  (h4 : R' = P' * Q')
  (h5 : Q' = Q * (1 - D / 100)) :
  D = 20 :=
by
  sorry

end tv_sets_sales_decrease_l647_647663


namespace prob_A_succeeds_third_attempt_prob_at_least_one_succeeds_first_attempt_l647_647331

/--
  A probability problem about Rubik's Cube solving:
  Given that person A has a probability of 0.8 to solve the cube within 30 seconds (P(A) = 0.8),
  and person B has a probability of 0.6 to solve the cube within 30 seconds (P(B) = 0.6),
  where each attempt is independent of the others.
-/

variables (P_A : ℝ) (P_B : ℝ)
variable (independent_events : Prop)

/-- 
  Condition 1:
  Person A has a probability of 0.8 to solve the cube within 30 seconds.
-/
axiom person_A_probability : P_A = 0.8

/-- 
  Condition 2:
  Person B has a probability of 0.6 to solve the cube within 30 seconds.
-/
axiom person_B_probability : P_B = 0.6

-- Condition 3 is assumed implicitly: independence of attempts.

-- The probability that person A succeeds on their third attempt is 0.032
theorem prob_A_succeeds_third_attempt :
  (1 - P_A) * (1 - P_A) * P_A = 0.032 :=
by
  sorry

-- The probability that at least one of them succeeds on their first attempt is 0.92
theorem prob_at_least_one_succeeds_first_attempt :
  1 - ((1 - P_A) * (1 - P_B)) = 0.92 :=
by
  sorry

end prob_A_succeeds_third_attempt_prob_at_least_one_succeeds_first_attempt_l647_647331


namespace log_abcd_x_l647_647203

theorem log_abcd_x (a b c d x α β γ δ : ℝ) (h1 : log a x = α) (h2 : log b x = β) 
  (h3 : log c x = γ) (h4 : log d x = δ) (h5 : x ≠ 1) : 
  log (a * b * c * d) x = (α * β * γ * δ) / (β * γ * δ + α * γ * δ + α * β * δ + α * β * γ) := 
  sorry

end log_abcd_x_l647_647203


namespace inheritance_amount_l647_647975

def federalTax (x : ℝ) : ℝ := 0.25 * x
def remainingAfterFederalTax (x : ℝ) : ℝ := x - federalTax x
def stateTax (x : ℝ) : ℝ := 0.15 * remainingAfterFederalTax x
def totalTaxes (x : ℝ) : ℝ := federalTax x + stateTax x

theorem inheritance_amount (x : ℝ) (h : totalTaxes x = 15000) : x = 41379 :=
by
  sorry

end inheritance_amount_l647_647975


namespace real_axis_length_of_hyperbola_l647_647134

-- Define the conditions of the problem
def parabola := ∀ x y : ℝ, y^2 = 4 * x
def hyperbola (a b : ℝ) := a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1
def focus := (1 : ℝ, 0 : ℝ)
def origin := (0 : ℝ, 0 : ℝ)
def pointA := (1 : ℝ, 2 : ℝ)
def pointB := (1 : ℝ, -2 : ℝ)

-- Define the vector operations
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def vector_AF (A : ℝ × ℝ) : ℝ × ℝ := (focus.1 - A.1, focus.2 - A.2)

-- Given condition and conclusion
theorem real_axis_length_of_hyperbola (a b : ℝ) (H : hyperbola a b) :
  (dot_product (vector_add origin pointA) (vector_AF pointA) = 0) →
  (2 * Real.sqrt 2 - 2 = 2 * a) →
  2 * Real.sqrt 2 - 2 = 2 * Real.sqrt 2 - 2 :=
by
  intros h₁ h₂
  sorry

end real_axis_length_of_hyperbola_l647_647134


namespace problem1_problem2_problem3_problem4_problem5_l647_647319

-- Axioms to represent the conditions
axiom cond1 : ∀ (arr : List ℕ), ¬((arr.head = some 1) ∨ (arr.last = some 2))
axiom cond2 : ∀ (arr : List ℕ), adjacent (arr, 1, 2, 3)
axiom cond3 : ∀ (arr : List ℕ), ¬adjacent(arr, 1, 2, 3)
axiom cond4 : ∀ (arr : List ℕ), exactly_one_between(arr, 1, 2)
axiom cond5 : ∀ (arr : List ℕ), ordered_left_to_right (arr, 1, 2, 3)

-- Define number of possible arrangements based on a condition
noncomputable def count_cond1 : ℕ := 3720
noncomputable def count_cond2 : ℕ := 720
noncomputable def count_cond3 : ℕ := 1440
noncomputable def count_cond4 : ℕ := 1200
noncomputable def count_cond5 : ℕ := 840

-- Proof problems
theorem problem1 : (count_arrangements cond1 = count_cond1) := by sorry
theorem problem2 : (count_arrangements cond2 = count_cond2) := by sorry
theorem problem3 : (count_arrangements cond3 = count_cond3) := by sorry
theorem problem4 : (count_arrangements cond4 = count_cond4) := by sorry
theorem problem5 : (count_arrangements cond5 = count_cond5) := by sorry

end problem1_problem2_problem3_problem4_problem5_l647_647319


namespace sum_x_coordinates_correct_l647_647716

noncomputable def sum_x_coordinates_P (APQR APS T1 T2 : ℝ)
  (Q R S T : ℝ × ℝ) 
  (h : ℝ) : ℝ :=
if h = APQR * 2 / (R.1 - Q.1) then
  let y1 := h in
  let y2 := -h in
  let x1 := S.1 in -- Simplification for x-coordinates formula
  let x2 := T.1 - S.2 in -- Simplification for x-coordinates formula
  let x3 := S.1 in -- Simplification for x-coordinates formula
  let x4 := T.1 - T.2 in -- Simplification for x-coordinates formula
  4 * (abs (S.1 - 160)) -- Simplification
else 0

theorem sum_x_coordinates_correct (APQR APS : ℕ) (Q R S T : ℕ × ℕ) :
  APQR = 1739 → APS = 6956 → Q = (0, 0) → R = (307, 0) → S = (450, 280) → T = (460, 290) →
  sum_x_coordinates_P APQR APS Q R S T = 640 :=
by
  sorry

end sum_x_coordinates_correct_l647_647716


namespace family_member_stats_l647_647307

noncomputable def mean (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

def median (nums : List ℕ) : ℕ :=
  let sorted := nums.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

def mode (nums : List ℕ) : ℕ :=
  nums.foldr (λ n acc, if nums.count n > nums.count acc then n else acc) nums.head!

theorem family_member_stats :
  let nums := [3, 2, 3, 3, 4, 3, 3]
  let a := mean nums
  let b := median nums
  let c := mode nums
  a = 3 ∧ b = 3 ∧ c = 3 :=
by {
  let nums := [3, 2, 3, 3, 4, 3, 3]
  let a := mean nums
  let b := median nums
  let c := mode nums
  have ha : a = 3 := sorry
  have hb : b = 3 := sorry
  have hc : c = 3 := sorry
  exact ⟨ha, hb, hc⟩
}

end family_member_stats_l647_647307


namespace remainder_of_sum_first_12_natural_numbers_div_9_l647_647340

/-- The sum of the first 12 natural numbers is divisible by 9 with a remainder. -/
theorem remainder_of_sum_first_12_natural_numbers_div_9 : 
  let s := (12 * (12 + 1)) / 2 in 
  s % 9 = 6 :=
by 
  let s := (12 * (12 + 1)) / 2
  show s % 9 = 6 from sorry

end remainder_of_sum_first_12_natural_numbers_div_9_l647_647340


namespace change_fred_received_l647_647254

-- Define constants and conditions
def cost_of_ticket := 5.92
def num_tickets := 3
def renting_movie := 6.79
def snack_cost := 10.50
def parking_cost := 3.25
def payment := 50.0

-- Define total cost of the movie tickets
def tickets_total_cost := num_tickets * cost_of_ticket

-- Define total cost of all expenses
def total_cost := tickets_total_cost + renting_movie + snack_cost + parking_cost

-- State the theorem to prove
theorem change_fred_received : payment - total_cost = 11.70 :=
by
  -- Placeholder for proof
  sorry

end change_fred_received_l647_647254


namespace area_of_region_correct_l647_647867

noncomputable def area_of_region (A B C : ℝ × ℝ) (P : ℝ × ℝ × ℝ) (O : ℝ × ℝ × ℝ) (r : ℝ) : ℝ :=
  if (B.1 - A.1) = 1 ∧ (C.1 - B.1) = 1 ∧ (A.1 - C.1) = 1 ∧ (P.3) = sqrt 2 ∧
     (O.1, O.2, O.3) = (1/2, sqrt(3)/6, r) ∧ r = sqrt 2 / 6 then
    1/4 + real.pi / 24
  else 0

theorem area_of_region_correct :
  ∀ A B C P O r,
    (B.1 - A.1) = 1 ∧
    (C.1 - B.1) = 1 ∧
    (A.1 - C.1) = 1 ∧
    (P.3) = sqrt 2 ∧
    (O.1, O.2, O.3) = (1/2, sqrt(3)/6, r) ∧
    r = sqrt 2 / 6 →
  area_of_region A B C P O r = 1/4 + real.pi / 24 := by
  sorry

end area_of_region_correct_l647_647867


namespace a_can_complete_in_6_days_l647_647744

noncomputable def rate_b : ℚ := 1/8
noncomputable def rate_c : ℚ := 1/12
noncomputable def earnings_total : ℚ := 2340
noncomputable def earnings_b : ℚ := 780.0000000000001

theorem a_can_complete_in_6_days :
  ∃ (rate_a : ℚ), 
    (1 / rate_a) = 6 ∧
    rate_a + rate_b + rate_c = 3 * rate_b ∧
    earnings_b = (rate_b / (rate_a + rate_b + rate_c)) * earnings_total := sorry

end a_can_complete_in_6_days_l647_647744


namespace DM_eq_r_plus_R_l647_647428

noncomputable def radius_incircle (A B D : ℝ) (s K : ℝ) : ℝ := K / s

noncomputable def radius_excircle (A C D : ℝ) (s' K' : ℝ) (AD : ℝ) : ℝ := K' / (s' - AD)

theorem DM_eq_r_plus_R 
  (A B C D M : ℝ)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : A ≠ C)
  (h4 : D = (B + C) / 2)
  (h5 : M = (B + C) / 2)
  (r : ℝ)
  (h6 : r = radius_incircle A B D ((A + B + D) / 2) (abs ((A - B) * (A - D) / 2)))
  (R : ℝ)
  (h7 : R = radius_excircle A C D ((A + C + D) / 2) (abs ((A - C) * (A - D) / 2)) (abs (A - D))) :
  dist D M =r + R :=
by sorry

end DM_eq_r_plus_R_l647_647428


namespace symmetric_point_correct_l647_647293

-- Define the point and the symmetry operation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def symmetric_with_respect_to_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

-- Define the specific point M
def M : Point := {x := 1, y := 2}

-- Define the expected answer point M'
def M' : Point := {x := 1, y := -2}

-- Prove that the symmetric point with respect to the x-axis is as expected
theorem symmetric_point_correct :
  symmetric_with_respect_to_x_axis M = M' :=
by sorry

end symmetric_point_correct_l647_647293


namespace trajectory_C_fixed_point_l647_647152

open Real

-- Define the points A and B
def A : ℝ × ℝ := (-sqrt 2, 0)
def B : ℝ × ℝ := (sqrt 2, 0)

-- Define the moving point P and its projection Q
variable (P : ℝ × ℝ)
def Q : ℝ × ℝ := (fst P, 0)

-- Define the given condition as a predicate
def given_condition (P : ℝ × ℝ) : Prop :=
let PA := (fst P - fst A, snd P - snd A)
let PB := (fst P - fst B, snd P - snd B)
let PQ := (fst P - fst Q, snd P - snd Q)
2 * (PA.1 * PB.1 + PA.2 * PB.2) = PQ.1 * PQ.1 + PQ.2 * PQ.2

-- Theorem about the trajectory of P
theorem trajectory_C (P : ℝ × ℝ) (h : given_condition P) : 
  (fst P)^2 / 4 + (snd P)^2 / 2 = 1 :=
sorry

-- Additional theorem about the midpoints and the fixed point
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
((fst P1 + fst P2) / 2, (snd P1 + snd P2) / 2)

theorem fixed_point (E1 E2 : ℝ × ℝ) (k : ℝ)
  (F : ℝ × ℝ := (1, 0))
  (GH MN : set (ℝ × ℝ))
  (HG_rel MN_rel : (GH = {P : ℝ × ℝ | snd P = k * (fst P - fst F)}) 
                  ∧ (MN = {P : ℝ × ℝ | snd P = - (fst P - fst F) / k}))
  (G H M N : ℝ × ℝ)
  (hG : G ∈ GH) (hH : H ∈ GH) (hM : M ∈ MN) (hN : N ∈ MN)
  (hE1 : E1 = midpoint G H) (hE2 : E2 = midpoint M N) :
  ∃ (p : ℝ × ℝ), p = (2 / 3, 0) ∧ ∀ (x : ℝ × ℝ), x ∈ line_through E1 E2 → x = p :=
sorry

end trajectory_C_fixed_point_l647_647152


namespace number_of_penguins_l647_647593

-- Define the number of animals and zookeepers
def zebras : ℕ := 22
def tigers : ℕ := 8
def zookeepers : ℕ := 12
def headsLessThanFeetBy : ℕ := 132

-- Define the theorem to prove the number of penguins (P)
theorem number_of_penguins (P : ℕ) (H : P + zebras + tigers + zookeepers + headsLessThanFeetBy = 4 * P + 4 * zebras + 4 * tigers + 2 * zookeepers) : P = 10 :=
by
  sorry

end number_of_penguins_l647_647593


namespace prove_intersection_coords_prove_polar_coord_eq_of_moving_point_P_l647_647596

noncomputable def intersection_polar_coords_of_C1_and_C2 : ℝ × ℝ :=
(2 * real.sqrt 3, real.pi / 6)

noncomputable def polar_coord_eq_of_moving_point_P (Q_on_C1 : Prop) : ℝ → ℝ :=
λ (θ : ℝ), 10 * real.cos θ

theorem prove_intersection_coords :
  ∀ (ρ θ : ℝ), (ρ = 4 * real.cos θ) ∧ (ρ * real.cos θ = 3) → ρ = 2 * real.sqrt 3 ∧ θ = real.pi / 6 :=
by
  intro ρ θ
  intro ⟨h1, h2⟩
  sorry

theorem prove_polar_coord_eq_of_moving_point_P :
  ∀ (ρ θ : ℝ), (Q_on_C1 : Prop) → (Q_on_C1 → (ρ = 4 * real.cos θ ∧ θ ∈ set.Ico 0 (real.pi / 2)))
    → (Q_on_C1 → ρ = polar_coord_eq_of_moving_point_P Q_on_C1 θ) :=
by
  intro ρ θ Q_on_C1
  intro h
  sorry

end prove_intersection_coords_prove_polar_coord_eq_of_moving_point_P_l647_647596


namespace M_is_incenter_l647_647448

variable (A B C M : Type) [triangle A B C] 

-- Define the problem conditions
structure Circle (P Q : Type) :=
(center : Type)
(radius : ℝ)

noncomputable def circle_on_segment (M A : Type) : Circle M A := sorry

-- Specify the property of M we need to prove
def isIncenter (M : Type) (A B C : Type) [inhabited M] : Prop := 
  ∀ P Q R : Type,
  (circle_on_segment M A).common_chord_length = (circle_on_segment M B).common_chord_length ∧
  (circle_on_segment M B).common_chord_length = (circle_on_segment M C).common_chord_length

-- The theorem statement
theorem M_is_incenter 
  (A B C M : Type) [triangle A B C] [inhabited M] :
  (isIncenter M A B C) → Incenter M A B C :=
sorry

end M_is_incenter_l647_647448


namespace find_n_l647_647494

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l647_647494


namespace part_one_part_two_l647_647107

variable (a b : ℝ)

def z := (a + complex.I) * (1 - complex.I) + b * complex.I
def point := (a + 1, b - a + 1)

theorem part_one (h : point = (x, y) ∧ y = x - 3) : 2 * a > b := by
  sorry

theorem part_two (h : point = (x, y) ∧ y = x - 3) :
  ∃ (z : ℂ), ∀ |z|, complex.abs z ∈ set.Ici (3 * real.sqrt 2 / 2) := by
  sorry

end part_one_part_two_l647_647107


namespace calculate_value_of_squares_difference_l647_647033

theorem calculate_value_of_squares_difference : 305^2 - 301^2 = 2424 :=
by {
  sorry
}

end calculate_value_of_squares_difference_l647_647033


namespace certain_number_is_51_l647_647403

theorem certain_number_is_51 (G C : ℤ) 
  (h1 : G = 33) 
  (h2 : 3 * G = 2 * C - 3) : 
  C = 51 := 
by
  sorry

end certain_number_is_51_l647_647403


namespace quadrilateral_not_parallelogram_l647_647148

-- Definitions based on the given conditions
structure Quadrilateral :=
  (a b c d : ℝ) -- sides of the quadrilateral
  (parallel : Prop) -- one pair of parallel sides
  (equal_sides : Prop) -- another pair of equal sides

-- Problem statement
theorem quadrilateral_not_parallelogram (q : Quadrilateral) 
  (h1 : q.parallel) 
  (h2 : q.equal_sides) : 
  ¬ (∃ p : Quadrilateral, p = q) :=
sorry

end quadrilateral_not_parallelogram_l647_647148


namespace road_signs_ratio_l647_647386

theorem road_signs_ratio :
  let S1 := 40 in
  ∃ S2 S3 S4 : ℕ,
    S2 > S1 ∧
    S3 = 2 * S2 ∧
    S4 = S3 - 20 ∧
    S1 + S2 + S3 + S4 = 270 ∧
    (S2 / S1 : ℚ) = 5 / 4 :=
by
  sorry

end road_signs_ratio_l647_647386


namespace proper_subset_example_l647_647025

theorem proper_subset_example :
  let A := {x : ℝ | x^2 < 5 * x}
  ∧ ( ∃ x, x ∈ A)
  ∧ ( ∃ y, y ∈ {2, 5})
  ∧ ( ∃ z, z ∈ set.Ioi 6)
  ∧ ( ∃ w, w ∈ set.Ioo 0 5)
  ∧ ( ∃ t, t ∈ set.Ioo 1 5 ) in
  set.Ioo (1 : ℝ) 5 ⊂ set.Ioo 0 5 :=
by {
  intros A _ _ _ _ _,
  exact sorry,
}

end proper_subset_example_l647_647025


namespace july_6_1918_is_saturday_june_6_2018_is_tuesday_l647_647402

/- Definitions for the context of the problem -/
def is_weekday_string : string → Prop
| "Monday"    := true
| "Tuesday"   := true
| "Wednesday" := true
| "Thursday"  := true
| "Friday"    := true
| "Saturday"  := true
| "Sunday"    := true
| _           := false

def valid_date_format : string → string → string → Prop
| year, month, day := true  -- We assume valid input format

def days_between (start_date end_date : string) : nat := sorry

noncomputable def day_of_week (start_date : string) (n_days : nat) : string := sorry

/- Problem statements -/

-- Prove that July 6, 1918, was a Saturday given that April 10, 1998, is a Friday
theorem july_6_1918_is_saturday (h : day_of_week "April 10, 1998" 0 = "Friday")
  : day_of_week "July 6, 1918" (days_between "July 6, 1918" "April 10, 1998") = "Saturday" :=
by {
  sorry,
}

-- Prove that June 6, 2018, will be a Tuesday given that April 10, 1998, is a Friday
theorem june_6_2018_is_tuesday (h : day_of_week "April 10, 1998" 0 = "Friday")
  : day_of_week "June 6, 2018" (days_between "April 10, 1998" "June 6, 2018") = "Tuesday" :=
by {
  sorry,
}

end july_6_1918_is_saturday_june_6_2018_is_tuesday_l647_647402


namespace calculate_lassis_from_nine_mangoes_l647_647034

variable (mangoes_lassis_ratio : ℕ → ℕ → Prop)
variable (cost_per_mango : ℕ)

def num_lassis (mangoes : ℕ) : ℕ :=
  5 * mangoes
  
theorem calculate_lassis_from_nine_mangoes
  (h1 : mangoes_lassis_ratio 15 3)
  (h2 : cost_per_mango = 2) :
  num_lassis 9 = 45 :=
by
  sorry

end calculate_lassis_from_nine_mangoes_l647_647034


namespace max_value_xyz_l647_647498

theorem max_value_xyz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : 2 * x + 3 * x * y^2 + 2 * z = 36) : 
  x^2 * y^2 * z ≤ 144 :=
sorry

end max_value_xyz_l647_647498


namespace verify_hexagon_solution_l647_647518

noncomputable def hexagon_geometry_problem
  (P Q R S T U V : Type*)
  (PQ QR TU TS UV : ℕ)
  (x : ℕ)
  (area_hexagon : ℕ)
  (collinear : Π(a : Type*), Set a → Prop) :
  Prop :=
  PQ = 7 ∧ QR = 10 ∧ TU = 6 ∧
  area_hexagon = 65 ∧
  TS = x ∧ UV = x ∧
  collinear R S V →
  (RS + ST = x + 10)

-- Declare a hypothesis providing necessary properties
theorem verify_hexagon_solution
  (P Q R S T U V : Type*)
  (PQ QR TU : ℕ)
  (TS UV : ℕ)
  (x area_hexagon : ℕ)
  (collinear : Π(a : Type*), Set a → Prop)
  (h1 : PQ = 7)
  (h2 : QR = 10)
  (h3 : TU = 6)
  (h4 : area_hexagon = 65)
  (h5 : TS = x)
  (h6 : UV = x)
  (h7 : collinear R S V) :
  RS + ST = x + 10 :=
begin
  sorry
end

end verify_hexagon_solution_l647_647518


namespace range_of_a_l647_647882

def f (x : ℝ) : ℝ :=
  x^2 - (1/2) * Real.log x + (3/2)

theorem range_of_a (a : ℝ) (h_domain : ∀ x, 0 < x → f x ≠ 0) 
  (h_interval : ∀ x, x ∈ set.Ioo (a - 1) (a + 1) → f x ≠ 0) : 
  1 ≤ a ∧ a < (3/2) :=
sorry

end range_of_a_l647_647882


namespace new_profit_percentage_proof_l647_647409

noncomputable def original_selling_price : ℝ := 550
noncomputable def original_profit_percentage : ℝ := 0.10
noncomputable def extra_amount_received : ℝ := 35
noncomputable def new_profit_percentage : ℝ := 0.30

theorem new_profit_percentage_proof :
  ∃ CP CP' SP' P', 
    let CP := original_selling_price / (1 + original_profit_percentage) in
    let CP' := CP * (1 - original_profit_percentage) in
    let SP' := original_selling_price + extra_amount_received in
    let P' := SP' - CP' in
    (P' / CP') * 100 = new_profit_percentage * 100 :=
by
  sorry

end new_profit_percentage_proof_l647_647409


namespace soma_piece_combination_l647_647287

def is_soma_piece (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7

def unit_cubes (n : ℕ) : ℕ :=
  if n = 1 then 3 else if is_soma_piece n then 4 else 0

def pieces : List ℕ := [2, 3, 4, 5, 6, 7]

theorem soma_piece_combination :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ pieces ∧ b ∈ pieces ∧ unit_cubes a + unit_cubes b = 8 :=
by
  sorry

end soma_piece_combination_l647_647287


namespace hyperbola_eccentricity_proof_l647_647145

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (e : ℝ) : Prop :=
  ∃ (F Q M N : ℝ × ℝ),
    let focus_eq : ℝ := sqrt (a^2 + b^2)
    let F := (focus_eq, 0)
    let |MQ| := 3 * |QN|
    (Q.1^2 + Q.2^2 = a^2 + b^2) ∧ -- Foot of perpendicular on asymptote
    (|MQ| = 3 * |QN|) ∧ -- Condition on distances
    (e = sqrt (1 + b^2 / a^2)) ∧ -- Eccentricity formula
    (e = sqrt 5) -- Given eccentricity
    sorry

-- Formal statement to prove
theorem hyperbola_eccentricity_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_eccentricity a b ha hb (sqrt 5) :=
sorry

end hyperbola_eccentricity_proof_l647_647145


namespace second_column_2021st_row_l647_647408

def table_pattern (n : ℕ) : char :=
  if n % 4 = 1 then 'B' else
  if n % 4 = 2 then 'D' else
  if n % 4 = 3 then 'B' else
  'D'

theorem second_column_2021st_row : table_pattern 2021 = 'B' := 
by {
  have h : 2021 % 4 = 1 := by norm_num,
  rw [table_pattern, h],
  norm_num,
}

end second_column_2021st_row_l647_647408


namespace number_of_positions_l647_647260

variables {P Q R : Type} [metric_space P] [metric_space Q] [metric_space R]

-- Since P and Q are distinct points in the xy-plane
variables (p q : P)
hypothesis (h : p ≠ q)

-- PQ = QR = PR condition indicates forming equilateral triangle
def equilateral_positions : set P :=
  {r : P | dist p q = dist q r ∧ dist q r = dist p r}

theorem number_of_positions (h : p ≠ q) : 
  ∃! (qr_positions : set P), qr_positions = equilateral_positions p q ∧ qr_positions.card = 2 :=
sorry

end number_of_positions_l647_647260


namespace sum_six_digit_odd_and_multiples_of_3_l647_647984

-- Definitions based on conditions
def num_six_digit_odd_numbers : Nat := 9 * (10 ^ 4) * 5

def num_six_digit_multiples_of_3 : Nat := 900000 / 3

-- Proof statement
theorem sum_six_digit_odd_and_multiples_of_3 : 
  num_six_digit_odd_numbers + num_six_digit_multiples_of_3 = 750000 := 
by 
  sorry

end sum_six_digit_odd_and_multiples_of_3_l647_647984


namespace sec_neg_405_eq_sqrt_2_l647_647436

theorem sec_neg_405_eq_sqrt_2 : Real.sec (-405 * Real.pi / 180) = Real.sqrt 2 :=
by
  -- Using the definition of secant and the periodicity of cosine
  have h1 : Real.sec x = 1 / Real.cos x := Real.sec_def
  have h2 : Real.cos x = Real.cos (x + 2 * Real.pi) := Real.cos_periodic x 2
  have h3 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := sorry
  rw [h1]
  rw [Real.cos_eq_cos_of_periodic (-405 * Real.pi / 180) (-405 * Real.pi / 180 + 2 * Real.pi)]
  rw [Real.cos_add (-405 * Real.pi / 180) 2 * Real.pi]
  simp only
  rw [Real.cos_315]

end sec_neg_405_eq_sqrt_2_l647_647436


namespace find_point_B_l647_647119

theorem find_point_B
  (A : ℝ × ℝ)
  (a : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hA : A = (-1, 5))
  (hA_to_B : to_vector A B = 3 * a)
  (ha : a = (2, 3)) :
  B = (5, 14) :=
sorry

end find_point_B_l647_647119


namespace correct_distribution_l647_647933

-- Define the conditions
def num_students : ℕ := 40
def ratio_A_to_B : ℚ := 0.8
def ratio_C_to_B : ℚ := 1.2

-- Definitions for the number of students earning each grade
def num_B (x : ℕ) : ℕ := x
def num_A (x : ℕ) : ℕ := Nat.floor (ratio_A_to_B * x)
def num_C (x : ℕ) : ℕ := Nat.ceil (ratio_C_to_B * x)

-- Prove the distribution is correct
theorem correct_distribution :
  ∃ x : ℕ, num_A x + num_B x + num_C x = num_students ∧ 
           num_A x = 10 ∧ num_B x = 14 ∧ num_C x = 16 :=
by
  sorry

end correct_distribution_l647_647933


namespace zero_point_at_2_l647_647045

theorem zero_point_at_2 :
  (∃ y : ℝ → ℝ, (y = λ x, x - 2) ∧ (y 2 = 0)) ∧
  ¬(∃ y : ℝ → ℝ, (y = λ x, x + 2) ∧ (y 2 = 0)) ∧
  ¬(∃ y : ℝ → ℝ, (y = λ x, 2 / x) ∧ (y 2 = 0)) ∧
  ¬(∃ y : ℝ → ℝ, (y = λ x, 2 / (x - 2)) ∧ (y 2 = 0)) := 
by {
  sorry
}

end zero_point_at_2_l647_647045


namespace incenter_chords_equal_l647_647440

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry Triangle

/-- Given a triangle ABC and a point M inside it,
  if circles are constructed on segments MA, MB, and MC as diameters,
  then the point M such that the lengths of common chords are equal, 
  is the incenter of triangle ABC. -/
theorem incenter_chords_equal (A B C M : α) (hM : incircle M (Triangle.mk A B C))
  (h_chord_eq : ∀ (P Q R : α), is_common_chord P (circle (segment.mk M A))
    (circle (segment.mk M B)) = is_common_chord Q (circle (segment.mk M B)) 
    (circle (segment.mk M C)) = is_common_chord R (circle (segment.mk M C))
    (circle (segment.mk M A))) : is_incenter M (Triangle.mk A B C) :=
sorry

end incenter_chords_equal_l647_647440


namespace find_lambda_l647_647954

noncomputable def lambda_value (A B C A1 B1 C1 M N : ℝ × ℝ × ℝ) (λ : ℝ) : Prop :=
  let AB1 := (B1.1 - A.1, B1.2 - A.2, B1.3 - A.3);
  let MN := (N.1 - M.1, N.2 - M.2, N.3 - M.3);
  (AB1.1 * MN.1 + AB1.2 * MN.2 + AB1.3 * MN.3 = 0) ∧ (λ = (N.3 - C1.3) / (C1.3 / (1 + λ) - C.3))

theorem find_lambda : 
  ∀ (A B C A1 B1 C1 M N : ℝ × ℝ × ℝ),
  -- Coordinate points setup based on conditions
  A = (0, real.sqrt 3 / 2, 0) →
  B1 = (-1 / 2, 0, 2) →
  C = (1 / 2, 0, 0) →
  C1 = (1 / 2, 0, 2) →
  M = (0, 0, 0) →
  λ ∈ λ_value A B C A1 B1 C1 M N →
  N = (1 / 2, 0, 2 / (1 + 15)) →
  λ = 15 :=
begin
  sorry
end

end find_lambda_l647_647954


namespace ages_correct_l647_647973

-- Definitions of the given conditions
def john_age : ℕ := 42
def tim_age : ℕ := 79
def james_age : ℕ := 30
def lisa_age : ℚ := 54.5
def kate_age : ℕ := 34
def michael_age : ℚ := 61.5
def anna_age : ℚ := 54.5

-- Mathematically equivalent proof problem
theorem ages_correct :
  (james_age = 30) ∧
  (lisa_age = 54.5) ∧
  (kate_age = 34) ∧
  (michael_age = 61.5) ∧
  (anna_age = 54.5) :=
by {
  sorry  -- Proof to be filled in
}

end ages_correct_l647_647973


namespace side_length_uncovered_l647_647383

theorem side_length_uncovered (L W : ℝ) (h₁ : L * W = 50) (h₂ : 2 * W + L = 25) : L = 20 :=
by {
  sorry
}

end side_length_uncovered_l647_647383


namespace transformations_invariant_count_is_four_l647_647420

-- Given conditions
variable (ℓ : Line)

-- Patterns on line ℓ
inductive Pattern : Type
| circle
| triangle_up
| triangle_down

-- The repeated pattern on line ℓ
def repeating_pattern : List Pattern := [Pattern.circle, Pattern.triangle_up, Pattern.triangle_down, Pattern.circle]

-- Types of transformations
inductive Transformation : Type
| rotation
| translation
| reflection_parallel
| reflection_perpendicular

-- Helper function to check if a transformation keeps the pattern invariant
def keeps_pattern_invariant (t : Transformation) (p : List Pattern) : Prop := sorry

-- Main statement
theorem transformations_invariant_count_is_four : 
  (finset.filter (λ t, keeps_pattern_invariant t repeating_pattern) 
    (finset.univ : finset Transformation)).card = 4 := sorry

end transformations_invariant_count_is_four_l647_647420


namespace petrol_and_oil_change_prices_l647_647384

theorem petrol_and_oil_change_prices :
  ∃ (P x : ℝ),
  let discountedP := 0.9 * P,
      discounted_x := 0.8 * x in
  (200 / discountedP - 200 / P = 5) ∧
  ((250 - 200) = discounted_x) ∧
  P = 40 / 9 ∧
  x = 62.5 ∧
  discounted_x = 50 :=
begin
  sorry
end

end petrol_and_oil_change_prices_l647_647384


namespace problem_solution_l647_647150

open Real

-- Define proposition p: There exists an x in ℝ such that x^2 + 2x + 5 ≤ 4.
def p : Prop := ∃ (x : ℝ), x^2 + 2 * x + 5 ≤ 4

-- Define proposition q: For x in (0, π/2), the minimum value of f(x) = sin x + 4 / sin x is 4.
def f (x : ℝ) : ℝ := sin x + 4 / sin x
def q : Prop := ∀ (x : ℝ), 0 < x ∧ x < π / 2 → f(x) = 4

-- The statement to be proved: p ∧ ¬q is true.
theorem problem_solution : p ∧ ¬q :=
by {
  sorry
}

end problem_solution_l647_647150


namespace length_of_minor_arc_l647_647948

noncomputable def point_A : ℝ × ℝ := (1, 0)
noncomputable def point_P : ℝ × ℝ := (Real.cos 2, -Real.sin 2)

theorem length_of_minor_arc (A P : ℝ × ℝ) (hA : A = point_A) (hP : P = point_P) : 
    Real.angle_of_minor_arc A P = 2 :=
sorry

end length_of_minor_arc_l647_647948


namespace rectangle_ratio_l647_647183

noncomputable def width_length_ratio (w l P : ℕ) : ℕ × ℕ :=
  if 2 * w + 2 * l = P then
    let g := Nat.gcd w l in
    (w / g, l / g)
  else
    (0, 0)

theorem rectangle_ratio (w : ℕ) (l : ℕ := 10) (P : ℕ := 32) :
  2 * w + 2 * l = P → width_length_ratio w l P = (3, 5) :=
by
  intro h
  dsimp [width_length_ratio]
  rw [if_pos h]
  have : w = 6 := by
    linarith
  rw [this, Nat.gcd_def, Nat.mod_eq_zero_of_dvd] -- gcd(6, 10) = 2
  rw [Nat.div_eq_of_eq_mul_right]
  linarith [this]
  sorry

end rectangle_ratio_l647_647183


namespace find_n_mod_10_l647_647481

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l647_647481


namespace common_difference_l647_647989

-- Definitions of the arithmetic sequence
def a (n : ℕ) (a1 d : ℝ) := a1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) (a1 d : ℝ) := n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
axiom a3 : a 3 a1 d = 3
axiom S7 : S 7 a1 d = 14

-- The theorem to prove
theorem common_difference (a1 d : ℝ) (a3 : a 3 a1 d = 3) (S7 : S 7 a1 d = 14) : d = -1 := by
  sorry

end common_difference_l647_647989


namespace pau_total_ordered_correct_l647_647974

-- Define the initial pieces of fried chicken ordered by Kobe
def kobe_order : ℝ := 5

-- Define Pau's initial order as twice Kobe's order plus 2.5 pieces
def pau_initial_order : ℝ := (2 * kobe_order) + 2.5

-- Define Shaquille's initial order as 50% more than Pau's initial order
def shaq_initial_order : ℝ := pau_initial_order * 1.5

-- Define the total pieces of chicken Pau will have eaten by the end
def pau_total_ordered : ℝ := 2 * pau_initial_order

-- Prove that Pau will have eaten 25 pieces of fried chicken by the end
theorem pau_total_ordered_correct : pau_total_ordered = 25 := by
  sorry

end pau_total_ordered_correct_l647_647974


namespace find_point_incenter_l647_647452

open Real

-- Define the incenter condition for a point M inside a triangle ABC
def isIncenter (A B C M : Point) : Prop :=
  (dist M (line.through A B) = dist M (line.through B C)) ∧ (dist M (line.through B C) = dist M (line.through C A))

-- Define the problem statement
theorem find_point_incenter (A B C : Point) :
  ∃ M : Point, isIncenter A B C M :=
by
  sorry

end find_point_incenter_l647_647452


namespace find_incenter_l647_647462

-- Define the arbitrary triangle
variable {α : Type} [EuclideanGeometry α]
variables (A B C M : α)

-- Define the main hypothesis and theorem
theorem find_incenter (inside_triangle : M ∈ triangle ABC) (equal_chords : ∀ D ∈ {A, B, C}, length (common_chord M D) = some_constant) :
  M = incenter ABC :=
sorry

end find_incenter_l647_647462


namespace length_of_center_square_is_60_l647_647301

noncomputable def length_of_center_square
  (a : ℕ) -- side length of main square
  (n : ℕ) -- number of L-shaped regions
  (f : ℚ) -- fractional area occupied by each L-shaped region
  (h₁ : a = 120) -- given side length of the main square
  (h₂ : n = 4) -- there are four L-shaped regions
  (h₃ : f = 1 / 5) -- each region occupies 1/5 of the total area
  : ℕ :=
let total_area := a * a in
let occupied_area := n * f * total_area in
let center_area := total_area - occupied_area in
nat.sqrt center_area

theorem length_of_center_square_is_60 
  (a : ℕ) 
  (n : ℕ) 
  (f : ℚ)
  (h₁ : a = 120)
  (h₂ : n = 4)
  (h₃ : f = 1 / 5) 
  : length_of_center_square a n f h₁ h₂ h₃ = 60 := 
by 
sorry

end length_of_center_square_is_60_l647_647301


namespace total_feet_is_140_l647_647378

def total_heads : ℕ := 48
def number_of_hens : ℕ := 26
def number_of_cows : ℕ := total_heads - number_of_hens
def feet_per_hen : ℕ := 2
def feet_per_cow : ℕ := 4

theorem total_feet_is_140 : ((number_of_hens * feet_per_hen) + (number_of_cows * feet_per_cow)) = 140 := by
  sorry

end total_feet_is_140_l647_647378


namespace four_points_concyclic_l647_647694

-- Define the geometric conditions and the statement of the theorem
theorem four_points_concyclic 
    (ABC : Triangle) (M : Point) (O : Point) (O1: Point)
    (Circumcircle : Circle) (AngleBisector : Line)
    (H1 : M ∈ Circumcircle)
    (H2 : M ∈ continuation (AngleBisector))
    (H3 : O = incenter ABC)
    (H4 : O1 = excenter ABC (AC_side : Side)) :
  cyclic A C O O1 ∧ Circumcircle.center = M :=
by sorry

end four_points_concyclic_l647_647694


namespace solve_abs_inequality_l647_647282

theorem solve_abs_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 8 ↔ x ∈ set.Ioo (-9 / 2) (-3) ∪ set.Ico (-3) 2 ∪ set.Ico 2 (7 / 2) :=
sorry

end solve_abs_inequality_l647_647282


namespace largest_divisor_of_n_l647_647746

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n^2 = 18 * k) : ∃ l : ℕ, n = 6 * l :=
sorry

end largest_divisor_of_n_l647_647746


namespace probability_of_2_lt_X_le_4_l647_647884

/-- Define the probability distribution function for the random variable X. -/
def P (a : ℝ) : ℕ → ℝ := λ k, if 1 ≤ k ∧ k ≤ 10 then a / 2^k else 0

/-- Define the sum constraint for probabilities. -/
def sum_prob_constraint (a : ℝ) : Prop :=
  (∑ k in finset.range 10, P a (k + 1)) = 1

/-- Define the probability P(2 < X ≤ 4). -/
def target_probability (a : ℝ) : ℝ :=
  P a 3 + P a 4

/-- Main theorem stating the required condition. -/
theorem probability_of_2_lt_X_le_4
  (a : ℝ) (h : sum_prob_constraint a) :
  target_probability a = 64 / 341 :=
sorry

end probability_of_2_lt_X_le_4_l647_647884


namespace find_f_2_l647_647538

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (2 ^ x) + x^2 - 1 else -(2 ^ (-x)) + x^2 - 1

theorem find_f_2 : f 2 = - (13 / 4) :=
by
  sorry

end find_f_2_l647_647538


namespace product_modulo_seven_l647_647809

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l647_647809


namespace min_expression_value_l647_647077

theorem min_expression_value (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ (min_val : ℝ), min_val = 12 ∧ (∀ (x y : ℝ), (x > 1) → (y > 1) →
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ min_val))) :=
by
  sorry

end min_expression_value_l647_647077


namespace ordered_pairs_real_imaginary_numbers_l647_647036

theorem ordered_pairs_real_imaginary_numbers :
  let x_even_pairs := (choose 100 2)
  let form_4k1 := 50
  let form_4k3 := 50
  let pairs_form_1 := form_4k1 * form_4k3
  let pairs_form_2 := form_4k3 * form_4k1
  in x_even_pairs + pairs_form_1 + pairs_form_2 = 9950 :=
by
  let x_even_pairs := (choose 100 2)
  let form_4k1 := 50
  let form_4k3 := 50
  let pairs_form_1 := form_4k1 * form_4k3
  let pairs_form_2 := form_4k3 * form_4k1
  have h1: x_even_pairs = choose 100 2 := rfl
  have h2: pairs_form_1 = 50 * 50 := rfl
  have h3: pairs_form_2 = 50 * 50 := rfl
  calc
    x_even_pairs + pairs_form_1 + pairs_form_2 
        = (choose 100 2) + (50 * 50) + (50 * 50) : by rw [h1, h2, h3]
    ... = 4950 + 2500 + 2500 : rfl
    ... = 9950 : rfl

end ordered_pairs_real_imaginary_numbers_l647_647036


namespace n_eq_7_mod_10_l647_647483

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l647_647483


namespace find_platform_length_l647_647753

/-- The length of the platform. -/
def platform_length (train_length time_platform_cross time_pole_cross: ℝ) : ℝ :=
  let speed := train_length / time_pole_cross
  (speed * time_platform_cross) - train_length

theorem find_platform_length :
  platform_length 300 39 18 = 350.13 := by
  -- proof goes here
  sorry

end find_platform_length_l647_647753


namespace stock_return_to_original_l647_647168

theorem stock_return_to_original (x : ℝ) (h : x > 0) :
  ∃ d : ℝ, d = 3 / 13 ∧ (x * 1.30 * (1 - d)) = x :=
by sorry

end stock_return_to_original_l647_647168


namespace diagonal_length_correct_l647_647701

-- Definitions as per conditions
def is_isosceles_trapezoid (A B C D : Type) (AB CD : ℝ) (AD BC : ℝ) :=
  AB = 21 ∧ CD = 7 ∧ AD = 12 ∧ BC = 12

noncomputable def length_of_diagonal (A B C D : Type) [is_isosceles_trapezoid A B C D (21 : ℝ) (7 : ℝ) (12 : ℝ)] : ℝ :=
  Real.sqrt 291

-- Theorem statement
theorem diagonal_length_correct (A B C D : Type) [is_isosceles_trapezoid A B C D (21 : ℝ) (7 : ℝ) (12 : ℝ)] :
  length_of_diagonal A B C D = Real.sqrt 291 :=
sorry

end diagonal_length_correct_l647_647701


namespace polar_equation_proof_max_distance_sum_proof_l647_647602

def circle_C_parametric_equations : Type :=
  { x : ℝ, y : ℝ // ∃ θ : ℝ, x = 3 + 2 * Real.cos θ ∧ y = -3 + 2 * Real.sin θ }

def polar_coordinate_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * Real.cos θ + 6 * ρ * Real.sin θ + 14 = 0

axiom polar_equation_of_circle :
  ∀ (ρ θ : ℝ), (∃ (x y : ℝ), (x, y) = (3 + 2 * Real.cos θ, -3 + 2 * Real.sin θ)) →
    polar_coordinate_equation ρ θ

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

axiom max_distance_sum_squared :
  ∀ (M : ℝ × ℝ), (∃ θ : ℝ, M = (3 + 2 * Real.cos θ, -3 + 2 * Real.sin θ)) →
    distance_squared M (3, 0) + distance_squared M (0, -3) = 26 + 12 * Real.sqrt 2

-- Mathematical Proof Problem Statements
theorem polar_equation_proof : 
  ∀ θ ρ, (∃ (x y : ℝ), (x, y) = (3 + 2 * Real.cos θ, -3 + 2 * Real.sin θ)) →
    polar_coordinate_equation ρ θ :=
    polar_equation_of_circle

theorem max_distance_sum_proof :
  ∀ M, (∃ θ, M = (3 + 2 * Real.cos θ, -3 + 2 * Real.sin θ)) →
    distance_squared M (3, 0) + distance_squared M (0, -3) = 26 + 12 * Real.sqrt 2 :=
    max_distance_sum_squared

end polar_equation_proof_max_distance_sum_proof_l647_647602


namespace product_mod_7_l647_647812

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l647_647812


namespace find_incenter_l647_647461

-- Define the arbitrary triangle
variable {α : Type} [EuclideanGeometry α]
variables (A B C M : α)

-- Define the main hypothesis and theorem
theorem find_incenter (inside_triangle : M ∈ triangle ABC) (equal_chords : ∀ D ∈ {A, B, C}, length (common_chord M D) = some_constant) :
  M = incenter ABC :=
sorry

end find_incenter_l647_647461


namespace simplify_expression_l647_647093

-- Definitions of intermediate calculations
def a : ℤ := 3 + 5 + 6 - 2
def b : ℚ := a * 2 / 4
def c : ℤ := 3 * 4 + 6 - 4
def d : ℚ := c / 3

-- The statement to be proved
theorem simplify_expression : b + d = 32 / 3 := by
  sorry

end simplify_expression_l647_647093


namespace grape_to_cherry_ratio_l647_647406

theorem grape_to_cherry_ratio 
  (C G A : ℕ) 
  (h1 : G = 3 * C)
  (h2 : A = 2 * G)
  (h3 : 2.50 * (C + G + A) = 200) :
  G / C = 3 := 
by
  sorry

end grape_to_cherry_ratio_l647_647406


namespace solve_problem_l647_647100

variable (a b : ℝ)

def condition1 : Prop := a + b = 1
def condition2 : Prop := ab = -6

theorem solve_problem (h1 : condition1 a b) (h2 : condition2 a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 :=
by
  sorry

end solve_problem_l647_647100


namespace minimum_distance_l647_647195

noncomputable def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos θ, sin θ)

noncomputable def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 4) = 2 * sqrt 2

theorem minimum_distance :
  ∃ P Q : ℝ × ℝ,
    (∃ θ : ℝ, parametric_curve θ = P) ∧
    (∃ (ρ θ : ℝ), polar_curve ρ θ ∧ (ρ * cos θ, ρ * sin θ) = Q) ∧
    |P.1 - Q.1| + |P.2 - Q.2| = sqrt 2 :=
by
  sorry

end minimum_distance_l647_647195


namespace rectangular_to_polar_l647_647042

theorem rectangular_to_polar (x y : ℝ) (r : ℝ) (θ : ℝ) :
  x = -3 → y = 3 * Real.sqrt 3 → r > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x + if y < 0 then 2 * Real.pi else 0 →
  (r, θ) = (6, 2 * Real.pi / 3) := 
by
  intros hx hy hr hθ hr_θ
  sorry

end rectangular_to_polar_l647_647042


namespace tangent_line_circle_l647_647173

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ, (x + y = 0) → ((x - m)^2 + y^2 = 2)) : m = 2 :=
sorry

end tangent_line_circle_l647_647173


namespace trapezoid_EFBA_area_l647_647751

theorem trapezoid_EFBA_area {a : ℚ} (AE BF : ℚ) (area_ABCD : ℚ) (column_areas : List ℚ)
  (h_grid : column_areas = [a, 2 * a, 4 * a, 8 * a])
  (h_total_area : 3 * (a + 2 * a + 4 * a + 8 * a) = 48)
  (h_AE : AE = 2)
  (h_BF : BF = 4) :
  let AFGB_area := 15 * a
  let triangle_EF_area := 7 * a
  let total_trapezoid_area := AFGB_area + (triangle_EF_area / 2)
  total_trapezoid_area = 352 / 15 :=
by
  sorry

end trapezoid_EFBA_area_l647_647751


namespace smallest_constant_l647_647238

theorem smallest_constant (n : ℕ) (x : Finₓ n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  ∃ C ≥ 0, (∀ x, (∑ 1 ≤ i < j ≤ n, x i * x j * (x i^2 + x j^2)) ≤ C * (∑ i, x i)^4) ∧ C = 1 / 8 := sorry

end smallest_constant_l647_647238


namespace square_AF_l647_647978

-- Define the constants for the segment lengths.
def AB := 3 * Real.sqrt 2
def BC := 3 * Real.sqrt 3
def CD := 6 * Real.sqrt 6
def DE := 4 * Real.sqrt 2
def EF := 5 * Real.sqrt 2

-- Assume the concurrent cevians condition.
axiom concurrent_Cevians : true

-- Main theorem statement.
theorem square_AF (AF : ℝ) 
    (h_AD_concurrent : concurrent_Cevians) 
    (h_AB : AB = 3 * Real.sqrt 2)
    (h_BC : BC = 3 * Real.sqrt 3)
    (h_CD : CD = 6 * Real.sqrt 6)
    (h_DE : DE = 4 * Real.sqrt 2)
    (h_EF : EF = 5 * Real.sqrt 2) :
    AF^2 = 225 := sorry

end square_AF_l647_647978


namespace sec_neg_405_eq_sqrt2_l647_647439

noncomputable def cos : ℝ → ℝ :=
  λ x, sorry -- Assume that cos is the cosine function from real numbers to real numbers

def sec (x : ℝ) : ℝ := 1 / cos x

axiom cos_periodic (x : ℝ) : cos (x + 360) = cos x
axiom cos_even (x : ℝ) : cos (-x) = cos x
axiom cos_45_deg : cos 45 = real.sqrt 2 / 2

theorem sec_neg_405_eq_sqrt2 : sec (-405) = real.sqrt 2 :=
by
  -- Use the conditions given to prove this theorem
  sorry

end sec_neg_405_eq_sqrt2_l647_647439


namespace total_journey_time_l647_647789

variable (x y z : ℝ) -- Define the variables as real numbers

-- State the conditions as definitions
def time_first_segment := x / 50
def time_second_segment := (2 * y) / 30
def time_third_segment := (3 * z) / 80

-- State the main theorem to be proved
theorem total_journey_time : time_first_segment x y z + time_second_segment x y z + time_third_segment x y z = (12 * x + 40 * y + 22.5 * z) / 600 :=
by
  sorry -- Proof is not needed, so we can leave this part

end total_journey_time_l647_647789


namespace sum_of_valid_a_l647_647923

theorem sum_of_valid_a :
  (∑ a in (-5 : finset ℤ).filter (λ a, (a % 2 = 0) ∧
      (x ≥ max (a + 4) 1)), ∃ y : ℤ, (y ≥ 0 ∧ 3 * (y - 2) = a - (1 - y))) = -8 := 
by {
  sorry
}

end sum_of_valid_a_l647_647923


namespace toothbrush_difference_l647_647060

-- Define the initial conditions
def initial_toothbrushes : ℕ := 330
def january_toothbrushes : ℕ := 53
def february_toothbrushes : ℕ := 67
def march_toothbrushes : ℕ := 46

-- Calculate the remaining toothbrushes
def remaining_toothbrushes : ℕ := initial_toothbrushes - (january_toothbrushes + february_toothbrushes + march_toothbrushes)

-- Calculate toothbrushes given in April and May
def april_may_toothbrushes : ℕ := remaining_toothbrushes / 2

-- Define the difference in toothbrushes given away in the busiest and slowest months
def difference := april_may_toothbrushes - march_toothbrushes

-- Theorem to prove the difference is 36
theorem toothbrush_difference : difference = 36 :=
by
  -- Provided conditions in the problem
  have initial : initial_toothbrushes = 330 := rfl
  have jan : january_toothbrushes = 53 := rfl
  have feb : february_toothbrushes = 67 := rfl
  have mar : march_toothbrushes = 46 := rfl
  have remaining := calc
    remaining_toothbrushes = 330 - (53 + 67 + 46) : by sorry
  
  have april_may := calc
    april_may_toothbrushes = remaining_toothbrushes / 2 : by sorry
  
  have diff := calc
    difference = 82 - 46 : by sorry
  
  exact sorry

end toothbrush_difference_l647_647060


namespace log_base_three_has_13_digits_l647_647574

noncomputable def num_digits_base_ten (x : ℕ) : ℕ :=
Nat.floor (Real.log10 x) + 1

theorem log_base_three_has_13_digits (x : ℝ) (h : Real.logBase 3 (Real.logBase 3 (Real.logBase 3 x)) = 3) :
  num_digits_base_ten (Nat.ceiling x) = 13 :=
sorry

end log_base_three_has_13_digits_l647_647574


namespace hyperbola_equation_range_of_k_l647_647540

theorem hyperbola_equation 
  (a : ℝ) (c : ℝ) (b : ℝ) (h1 : a = sqrt 3) (h2 : c = 2) (h3 : b^2 = c^2 - a^2) : 
  (b = 1) ∧ (∀ x y : ℝ, (x^2 / 3 - y^2 = 1) → true) :=
by
  sorry

theorem range_of_k 
  (k : ℝ) 
  (h_intersect : ∀ x A B : ℝ × ℝ, (A.1, k * A.1 + sqrt 2) = A ∧ (B.1, k * B.1 + sqrt 2) = B → true) 
  (h_dot_product : ∀ x1 x2 y1 y2 : ℝ, (x1 * x2 + y1 * y2 > 2) → true) :
  (k ∈ Ioo (-1) (-sqrt 3 / 3) ∪ Ioo (sqrt 3 / 3) 1) :=
by
  sorry

end hyperbola_equation_range_of_k_l647_647540


namespace sum_first_60_terms_sequence_l647_647309

theorem sum_first_60_terms_sequence :
  ∀ (a : ℕ → ℤ), 
    (∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = 2 * n - 1) → 
    (∑ i in finset.range 60, a i) = 1830 := 
by sorry

end sum_first_60_terms_sequence_l647_647309


namespace main_theorem_l647_647865

noncomputable def parabola_vertex_origin_focus (F : Point) : Prop :=
  F.x = 1 ∧ F.y = 0

noncomputable def directrix_intersects_x_axis_at (K : Point) : Prop :=
  K.x = -1 ∧ K.y = 0

noncomputable def line_through_K_intersects_parabola_at (l : Line, A B : Point, K : Point) : Prop :=
  -- Assuming K is on l and l intersects parabola at A and B
  K ∈ l ∧ is_on_parabola C A ∧ is_on_parabola C B

noncomputable def A_symmetric_to_D_wrt_x_axis (A D : Point) : Prop :=
  D.x = A.x ∧ D.y = -A.y

noncomputable def F_on_BD (F B D : Point) : Prop :=
  collinear F B D

noncomputable def dot_product_is (FA FB : Vector) : Prop :=
  FA.dot FB = 8 / 9

noncomputable def equation_of_line_l (l : Line) : Prop :=
  l.equation = "3x + 4y + 3 = 0" ∨ l.equation = "3x - 4y + 3 = 0"

theorem main_theorem (C : Parabola) (F K A B D : Point) (l : Line) :
  (parabola_vertex_origin_focus F) →
  (directrix_intersects_x_axis_at K) →
  (line_through_K_intersects_parabola_at l A B K) →
  (A_symmetric_to_D_wrt_x_axis A D) →
  (dot_product_is (vector_from_to F A) (vector_from_to F B)) →
  (F_on_BD F B D) ∧ (equation_of_line_l l) :=
by
  assume hF_vertex hDir hLine hSymm hDot
  sorry

end main_theorem_l647_647865


namespace average_probable_weight_l647_647177

-- Definitions based on the conditions
def ArunOpinion (w : ℝ) : Prop := 65 < w ∧ w < 72
def BrotherOpinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def MotherOpinion (w : ℝ) : Prop := w ≤ 68

-- The actual statement we want to prove
theorem average_probable_weight : 
  (∀ (w : ℝ), ArunOpinion w → BrotherOpinion w → MotherOpinion w → 65 < w ∧ w ≤ 68) →
  (65 + 68) / 2 = 66.5 :=
by 
  intros h1
  sorry

end average_probable_weight_l647_647177


namespace focus_with_larger_x_coordinate_l647_647049

/-- Define the given hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  ((x - 2)^2 / 7^2) - ((y - 10)^2 / 3^2) = 1

/-- Define the center of the hyperbola -/
def center : ℝ × ℝ := (2, 10)

/-- Define the semi-major axis length -/
def a : ℝ := 7

/-- Define the semi-minor axis length -/
def b : ℝ := 3

/-- Define the distance to each focus from the center -/
def c : ℝ := Real.sqrt (a^2 + b^2)

theorem focus_with_larger_x_coordinate :
  ∃ x y : ℝ, 
    x = 2 + Real.sqrt 58 ∧ 
    y = 10 ∧ 
    hyperbola_eq x y :=
by
  sorry

end focus_with_larger_x_coordinate_l647_647049


namespace cosine_angle_ab_ac_l647_647470

def point := (ℝ × ℝ × ℝ)

def vec (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def dot_product (v1 v2 : point) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : point) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def cos_angle (p1 p2 p3 : point) : ℝ :=
  let ab := vec p1 p2
  let ac := vec p1 p3
  dot_product ab ac / (magnitude ab * magnitude ac)

theorem cosine_angle_ab_ac :
  cos_angle (0, 2, -4) (8, 2, 2) (6, 2, 4) = 0.96 :=
by
  sorry

end cosine_angle_ab_ac_l647_647470


namespace find_n_mod_10_l647_647478

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l647_647478


namespace ellipse_correct_constants_l647_647792

noncomputable def ellipse_constants : ℝ × ℝ × ℕ × ℕ :=
  let foci1 := (3, 3)
  let foci2 := (3, 7)
  let point := (15, -4)
  let a := 15
  let b := Real.sqrt 221
  let h := 3
  let k := 5 in
  (b, a, h, k)

theorem ellipse_correct_constants :
  ellipse_constants = (Real.sqrt 221, 15, 3, 5) :=
sorry

end ellipse_correct_constants_l647_647792


namespace xy_sum_greater_two_l647_647704

theorem xy_sum_greater_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := 
by 
  sorry

end xy_sum_greater_two_l647_647704


namespace tan_x_is_zero_l647_647514

theorem tan_x_is_zero (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ π) 
  (h3 : 3 * sin (x / 2) = sqrt (1 + sin x) - sqrt (1 - sin x)) : 
  tan x = 0 := 
by 
  sorry

end tan_x_is_zero_l647_647514


namespace problem_inequality_l647_647234

-- The statement of the problem in Lean 4
theorem problem_inequality (n : ℕ) (h : n ≥ 2) 
  (a : Fin n → ℝ) (ha : ∀ i, 0 < a i): 
  (∏ i in Finset.univ, (1 + 1 / (a i))) ≥ (1 + 1 / (real.geom_mean ha Finset.univ))^n 
  ↔ (∀ i j, a i = a j) := by 
  sorry

end problem_inequality_l647_647234


namespace range_of_a_l647_647577

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2^(x-1) - 2 else -Real.logb 2 (x + 1)

theorem range_of_a (a : ℝ) (h : f a ≥ -2) : a ≤ 3 :=
begin
  sorry
end

end range_of_a_l647_647577


namespace tan_alpha_sin_double_angle_l647_647096

theorem tan_alpha_sin_double_angle (α : ℝ) (h : Real.tan α = 3/4) : Real.sin (2 * α) = 24/25 :=
by
  sorry

end tan_alpha_sin_double_angle_l647_647096


namespace estimated_humerus_length_l647_647795

theorem estimated_humerus_length :
  ∀ (x : ℝ), (1.197 * x - 3.660 = 56.19) ↔ (x = 50) :=
by {
  intro x,
  split,
  { 
    intro h,
    have : 1.197 * x = 56.19 + 3.660, from eq_add_of_sub_eq h,
    have : 1.197 * x = 59.85, from this,
    exact eq_of_mul_eq_mul_right (by norm_num) this,
  },
  { 
    intro h,
    rw h,
    norm_num,
  }
}

end estimated_humerus_length_l647_647795


namespace regular_tetrahedron_distance_ratio_l647_647204

-- Definitions for the geometric elements and properties
def regular_tetrahedron (A B C D : Point) : Prop :=
  equidistant_center (center A B C D) A B C D ∧
  ⋀ F ∈ faces A B C D, centroid_ratio F (center A B C D) (2 / 3)

-- The theorem statement
theorem regular_tetrahedron_distance_ratio (A B C D : Point) (O : Point) :
  regular_tetrahedron A B C D →
  O = center A B C D →
  ∀ (G : Point), G = centroid (face A B C) →
  dist O A = 3 * dist O G :=
by
  intro h_reg_tet h_center h_centroid
  sorry

end regular_tetrahedron_distance_ratio_l647_647204


namespace gcd_f_50_51_l647_647993

-- Define f(x)
def f (x : ℤ) : ℤ := x^3 - x^2 + 2 * x + 2000

-- State the problem: Prove gcd(f(50), f(51)) = 8
theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 8 := by
  sorry

end gcd_f_50_51_l647_647993


namespace parabola_focus_hyperbola_required_equation_l647_647883

theorem parabola_focus_hyperbola (p : ℝ) (C : ℝ → ℝ → Prop) (H : ∀ x y, C x y ↔ x^2 = -2 * p * y) 
  (hyps1 : ∀ a b: ℝ, 
      (a = 0 ∧ (b = 2 ∨ b = -2)) → 
      ((∃ x y, C x y ∧ x = 0 ∧ y = b/2))) :
  p = 4 :=
by
  sorry

theorem required_equation :
  ∃ p : ℝ, ∃ C : ℝ → ℝ → Prop, 
  (∀ x y, C x y ↔ x^2 = -2 * p * y) →
  (∀ a b: ℝ, (a = 0 ∧ (b = 2 ∨ b = -2)) → (∃ x y, C x y ∧ x = 0 ∧ y = b/2)) →
  C = λ x y, x^2 = -8 * y :=
by
  have h_p4 := parabola_focus_hyperbola p C (λ x y, x^2 = -2 * p * y) hyps1
  use 4
  use (λ x y, x^2 = -8 * y)
  split; intros
  · exact H x y
  · rw [h_p4]
    sorry

end parabola_focus_hyperbola_required_equation_l647_647883


namespace sum_of_min_max_l647_647618

-- Define the necessary parameters and conditions
variables (n k : ℕ)
  (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ)
  (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ)
  (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max)

-- The goal is to prove that the sum of m and M equals n
theorem sum_of_min_max (n k : ℕ) (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ) (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ) (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max) :
  m + M = n := 
sorry

end sum_of_min_max_l647_647618


namespace distinguishable_triangles_count_l647_647323

def count_distinguishable_triangles (colors : ℕ) : ℕ :=
  let corner_cases := colors + (colors * (colors - 1)) + (colors * (colors - 1) * (colors - 2) / 6)
  let edge_cases := colors * colors
  let center_cases := colors
  corner_cases * edge_cases * center_cases

theorem distinguishable_triangles_count :
  count_distinguishable_triangles 8 = 61440 :=
by
  unfold count_distinguishable_triangles
  -- corner_cases = 8 + 8 * 7 + (8 * 7 * 6) / 6 = 120
  -- edge_cases = 8 * 8 = 64
  -- center_cases = 8
  -- Total = 120 * 64 * 8 = 61440
  sorry

end distinguishable_triangles_count_l647_647323


namespace product_of_chord_lengths_l647_647620

noncomputable def omega : ℂ := Complex.exp(2 * Real.pi * Complex.I / 10)

noncomputable def A : ℂ := 3
noncomputable def B : ℂ := -3
noncomputable def C (k : ℕ) : ℂ := 3 * omega^k

noncomputable def length_AC (k : ℕ) : ℝ := Complex.abs (A - C k)
noncomputable def length_BC (k : ℕ) : ℝ := Complex.abs (B - C k)

theorem product_of_chord_lengths :
  let lengths := [length_AC 1, length_AC 2, length_AC 3, length_AC 4,
                  length_BC 1, length_BC 2, length_BC 3, length_BC 4]
  (list.prod lengths) = 590490 :=
by sorry

end product_of_chord_lengths_l647_647620


namespace stack_scoops_ways_l647_647274

def num_ways_to_stack_ice_cream_scoops : ℕ :=
  List.permutationsLength (['vanilla', 'chocolate', 'strawberry', 'cherry', 'pistachio'] : List String)

theorem stack_scoops_ways :
  num_ways_to_stack_ice_cream_scoops = 120 :=
by
  -- Proof of this theorem would be here
  sorry

end stack_scoops_ways_l647_647274


namespace range_of_OA_l647_647182

open Real

theorem range_of_OA (O A B1 B2 P : ℝ × ℝ) :
  (dist B1 B2 ≠ 0 ∧ dist O B1 = 1 ∧ dist O B2 = 1 ∧ add_vectors A B1 B2 = AP ∧ dist O P < 1 / 2) →
  sqrt (dist2 O A) ∈ Ioc (sqrt 7 / 2) (sqrt 2) :=
sorry

def add_vectors (A B1 B2 : ℝ × ℝ) : ℝ × ℝ :=
(A.1 + B1.1 + B2.1, A.2 + B1.2 + B2.2)

def dist (a b : ℝ × ℝ) : ℝ :=
sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

def dist2 (a b : ℝ × ℝ) : ℝ :=
(a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

end range_of_OA_l647_647182


namespace part1_part2_l647_647904

variables (a b : ℝ × ℝ) (k x : ℝ)
def a := (3, 2)
def b := (-2, 1)

theorem part1 :
  (let v1 := (3*k + 2, 2*k - 1)
       v2 := (-1, 4)
   in v1.1 * v2.1 + v1.2 * v2.2 = 0) ↔ k = 6 / 5 := sorry

theorem part2 :
  (let c := (3 - 2*x, 2 + x)
       b := (-2, 1)
       dot_product := b.1 * c.1 + b.2 * c.2
   in ∀ x : ℝ, (c.1^2 + c.2^2 is minimized → dot_product = 0) ↔ ∠ (c, b) = π / 2) := sorry

end part1_part2_l647_647904


namespace days_wages_l647_647010

theorem days_wages (S W_a W_b : ℝ) 
    (h1 : S = 28 * W_b) 
    (h2 : S = 12 * (W_a + W_b)) 
    (h3 : S = 21 * W_a) : 
    true := 
by sorry

end days_wages_l647_647010


namespace no_representation_of_expr_l647_647969

theorem no_representation_of_expr :
  ¬ ∃ f g : ℝ → ℝ, (∀ x y : ℝ, 1 + x ^ 2016 * y ^ 2016 = f x * g y) :=
by
  sorry

end no_representation_of_expr_l647_647969


namespace grocer_additional_coffee_l647_647373

theorem grocer_additional_coffee (x : ℝ) :
  let initial_stock := 400
      initial_decaf_percentage := 0.40
      bought_decaf_percentage := 0.60
      final_decaf_percentage := 0.44
      initial_decaf_coffee := initial_decaf_percentage * initial_stock
      total_stock := initial_stock + x
      additional_decaf_coffee := bought_decaf_percentage * x
      total_decaf_coffee := initial_decaf_coffee + additional_decaf_coffee
  in
  total_decaf_coffee = final_decaf_percentage * total_stock →
  x = 100 :=
by
  intros
  unfold initial_stock initial_decaf_percentage bought_decaf_percentage 
         final_decaf_percentage initial_decaf_coffee total_stock 
         additional_decaf_coffee total_decaf_coffee
  sorry

end grocer_additional_coffee_l647_647373


namespace trigonometric_identity_l647_647848

theorem trigonometric_identity :
  7 * 6 * (1 / Real.tan (2 * Real.pi * 10 / 360) + Real.tan (2 * Real.pi * 5 / 360)) 
  = 7 * 6 * (1 / Real.sin (2 * Real.pi * 10 / 360)) := 
sorry

end trigonometric_identity_l647_647848


namespace minimum_elements_finite_set_l647_647087

theorem minimum_elements_finite_set (A : Type) [Finite A] (f : ℕ → A) :
  (∀ i j : ℕ, Nat.Prime (Nat.abs (i - j)) → f i ≠ f j) → Finite.card A ≥ 4 :=
sorry

end minimum_elements_finite_set_l647_647087


namespace mixed_doubles_pairing_l647_647320

theorem mixed_doubles_pairing: 
  let males := 5
  let females := 4
  let choose_males := Nat.choose males 2
  let choose_females := Nat.choose females 2
  let arrangements := Nat.factorial 2
  choose_males * choose_females * arrangements = 120 := by
  sorry

end mixed_doubles_pairing_l647_647320


namespace function_characterization_l647_647047

theorem function_characterization (f : ℤ → ℤ)
  (h : ∀ a b : ℤ, ∃ k : ℤ, f (f a - b) + b * f (2 * a) = k ^ 2) :
  (∀ n : ℤ, (n % 2 = 0 → f n = 0) ∧ (n % 2 ≠ 0 → ∃ k: ℤ, f n = k ^ 2))
  ∨ (∀ n : ℤ, ∃ k: ℤ, f n = k ^ 2 ∧ k = n) :=
sorry

end function_characterization_l647_647047


namespace balanced_number_example_l647_647013

/--
A number is balanced if it is a three-digit number, all digits are different,
and it equals the sum of all possible two-digit numbers composed from its different digits.
-/
def isBalanced (n : ℕ) : Prop :=
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  (n = (10 * (n / 100) + (n / 10) % 10) + (10 * (n / 100) + n % 10) +
    (10 * ((n / 10) % 10) + n / 100) + (10 * ((n / 10) % 10) + n % 10) +
    (10 * (n % 10) + n / 100) + (10 * (n % 10) + ((n / 10) % 10)))

theorem balanced_number_example : isBalanced 132 :=
  sorry

end balanced_number_example_l647_647013


namespace largest_number_by_replacement_l647_647601

theorem largest_number_by_replacement 
  (n : ℝ) (n_1 n_3 n_6 n_8 : ℝ)
  (h : n = -0.3168)
  (h1 : n_1 = -0.3468)
  (h3 : n_3 = -0.4168)
  (h6 : n_6 = -0.3148)
  (h8 : n_8 = -0.3164)
  : n_6 > n_1 ∧ n_6 > n_3 ∧ n_6 > n_8 := 
by {
  -- Proof goes here
  sorry
}

end largest_number_by_replacement_l647_647601


namespace A_eq_D_l647_647988

def A := {θ : ℝ | 0 < θ ∧ θ < 90}
def D := {θ : ℝ | 0 < θ ∧ θ < 90}

theorem A_eq_D : A = D :=
by
  sorry

end A_eq_D_l647_647988


namespace solve_crate_and_carton_weight_l647_647009

variable (x y : ℝ)

def ratio_condition : Prop := (3 * x = y)

def weight_condition : Prop := (24 * x + 72 * y = 408)

def crate_weight : x = 1.7

def carton_weight : y = 5.1

def total_weight_pounds (kg_to_lb : ℝ := 2.20462) : ℝ :=
  (24 * 1.7 + 72 * 5.1) * kg_to_lb

def total_weight_correct : total_weight_pounds = 899.48

theorem solve_crate_and_carton_weight :
  ratio_condition ∧ weight_condition →
  crate_weight ∧ carton_weight ∧ total_weight_correct :=
by
  intro hwk
  sorry

end solve_crate_and_carton_weight_l647_647009


namespace sale_price_correct_l647_647329

variable (x : ℝ)

-- Conditions
def decreased_price (x : ℝ) : ℝ :=
  0.9 * x

def final_sale_price (decreased_price : ℝ) : ℝ :=
  0.7 * decreased_price

-- Proof statement
theorem sale_price_correct : final_sale_price (decreased_price x) = 0.63 * x := by
  sorry

end sale_price_correct_l647_647329


namespace degree_measure_angle_A_l647_647272

variables (A B C D : Type) [IsoscelesTrapezoid A B C D] (AB_parallel_CD : parallel A B C D)
          (external_angle_C : angle_ext_C DC = 70)

theorem degree_measure_angle_A :
  angle_A ABCD = 70 := 
sorry

end degree_measure_angle_A_l647_647272


namespace not_possible_f_g_l647_647970

theorem not_possible_f_g (f g : ℝ → ℝ) :
  ¬(∀ x y : ℝ, 1 + x^2016 * y^2016 = f(x) * g(y)) :=
by
  sorry

end not_possible_f_g_l647_647970


namespace orthocenter_minimizes_expression_l647_647220

variables {A B C P : Type*} [Point A] [Point B] [Point C] [Point P]
variables {x y z : ℝ}
variables {sin_A sin_B sin_C : ℝ}

def is_orthocenter (A B C P : Type*) [Point A] [Point B] [Point C] [Point P] : Prop := sorry

/-- The expression $x \cdot \sin A + y \cdot \sin B + z \cdot \sin C$ attains its minimum value 
    when point $P$ is the orthocenter of $\triangle ABC$. -/
theorem orthocenter_minimizes_expression
  (hP : is_orthocenter A B C P)
  (h_xA : PA = x)
  (h_yB : PB = y)
  (h_zC : PC = z)
  (h_sinA : θ ∠BAC)
  (h_sinB : θ ∠ABC)
  (h_sinC : θ ∠BCA):
  x * sin_A + y * sin_B + z * sin_C = minimum_value := sorry

end orthocenter_minimizes_expression_l647_647220


namespace fill_time_without_leak_l647_647761

theorem fill_time_without_leak (F L : ℝ)
  (h1 : (F - L) * 12 = 1)
  (h2 : L * 24 = 1) :
  1 / F = 8 := 
sorry

end fill_time_without_leak_l647_647761


namespace mix_solutions_ratio_l647_647280

theorem mix_solutions_ratio 
    (A B : Type) 
    (amount_A amount_B : ℝ) 
    (x y : ℝ)
    (ha : A = 0.40) 
    (hb : B = 0.80) 
    (hA : amount_A = 30) 
    (hB : amount_B = 60) 
    (hx : x + y = 50) 
    (hy : 0.40 * x + 0.80 * y = 25) : 
    x / y = 3 / 1 :=
by
  sorry

end mix_solutions_ratio_l647_647280


namespace find_incenter_l647_647460

-- Define the arbitrary triangle
variable {α : Type} [EuclideanGeometry α]
variables (A B C M : α)

-- Define the main hypothesis and theorem
theorem find_incenter (inside_triangle : M ∈ triangle ABC) (equal_chords : ∀ D ∈ {A, B, C}, length (common_chord M D) = some_constant) :
  M = incenter ABC :=
sorry

end find_incenter_l647_647460


namespace volume_of_tetrahedron_eq_20_l647_647957

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  1 / 3 * a * b * c

theorem volume_of_tetrahedron_eq_20 {x y z : ℝ} (h1 : x^2 + y^2 = 25) (h2 : y^2 + z^2 = 41) (h3 : z^2 + x^2 = 34) :
  volume_tetrahedron 3 4 5 = 20 :=
by
  sorry

end volume_of_tetrahedron_eq_20_l647_647957


namespace add_ten_to_certain_number_l647_647734

theorem add_ten_to_certain_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 :=
by
  sorry

end add_ten_to_certain_number_l647_647734


namespace longest_side_of_triangle_l647_647300

theorem longest_side_of_triangle (x : ℝ) (a b c : ℝ)
  (h1 : a = 5)
  (h2 : b = 2 * x + 3)
  (h3 : c = 3 * x - 2)
  (h4 : a + b + c = 41) :
  c = 19 :=
by
  sorry

end longest_side_of_triangle_l647_647300


namespace gcd_bc_eq_one_l647_647987

theorem gcd_bc_eq_one (a b c x y : ℕ)
  (h1 : Nat.gcd a b = 120)
  (h2 : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  Nat.gcd b c = 1 :=
by
  sorry

end gcd_bc_eq_one_l647_647987


namespace find_n_l647_647825

theorem find_n:
  ∃ n : ℤ, n ≥ 1 ∧ (∃ a b : ℕ, n = (ab + 3b + 8) / (a^2 + b + 3) ∧
  ∀ p : ℤ, p.prime → ¬ (p ^ 3 ∣ (a^2 + b + 3))) ∧ n = 1 :=
by
  sorry

end find_n_l647_647825


namespace shaded_region_area_l647_647598

noncomputable def total_shaded_area (r1 : ℝ) : ℝ :=
  let r2 := 2 * r1
  let r3 := 3 * r1
  let area1 := π * r1^2
  let area2 := π * r2^2
  let area3 := π * r3^2
  (area2 - area1) + (area3 - area2)

theorem shaded_region_area (r : ℝ) (h : r = 3) : total_shaded_area r = 72 * π := by sorry

end shaded_region_area_l647_647598


namespace complex_sum_eq_zero_l647_647094

theorem complex_sum_eq_zero (z : ℂ) (h : z / (1 + 2 * complex.i) = 2 + complex.i) : 
  ∃ z_new : ℂ, (z_new = z + 5) ∧ (z_new.re + z_new.im = 0) :=
by
  sorry

end complex_sum_eq_zero_l647_647094


namespace simplify_complex_fraction_l647_647545

theorem simplify_complex_fraction :
  (1 - complex.sqrt 3 * complex.I) / (complex.sqrt 3 + complex.I)^2 = -1 / 4 - (complex.sqrt 3 / 4) * complex.I :=
by
   sorry

end simplify_complex_fraction_l647_647545


namespace proof_problem_l647_647916

def x := 3
def y := 4

theorem proof_problem : 3 * x - 2 * y = 1 := by
  -- We will rely on these definitions and properties of arithmetic to show the result.
  -- The necessary proof steps would follow here, but are skipped for now.
  sorry

end proof_problem_l647_647916


namespace range_of_values_for_m_l647_647245

theorem range_of_values_for_m (m : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 :=
by
  sorry

end range_of_values_for_m_l647_647245


namespace lemniscate_surface_area_correct_l647_647802

noncomputable def lemniscate_surface_area (a : ℝ) : ℝ :=
  2 * Real.pi * a^2 * ∫ (x : ℝ) in 0..(Real.pi / 4), Real.sin x

theorem lemniscate_surface_area_correct (a : ℝ) : 
  lemniscate_surface_area a = Real.pi * a^2 * (2 - Real.sqrt 2) :=
by 
  have h_int : ∫ (x : ℝ) in 0..(Real.pi / 4), Real.sin x = 1 - (Real.sqrt 2 / 2),
  { sorry },

  rw [lemniscate_surface_area, h_int],
  ring,
  sorry

end lemniscate_surface_area_correct_l647_647802


namespace num_solutions_g_triple_l647_647235

def g (x : ℝ) : ℝ := -3 * Real.cos (π * x / 2)

theorem num_solutions_g_triple (h1 : -1 ≤ x ∧ x ≤ 1) :
  (g (g (g x)) = g x) ↔ (x = -1 ∨ x = 0 ∨ x = 1) :=
by
  sorry

end num_solutions_g_triple_l647_647235


namespace solve_for_y_l647_647165

theorem solve_for_y (y : ℝ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 :=
sorry

end solve_for_y_l647_647165


namespace problem1_problem2_l647_647149

noncomputable def set_B : Set ℝ := { m | m > 2 }
noncomputable def set_A (a : ℝ) : Set ℝ := if 3 * a > a + 2 then { x | 2 + a < x ∧ x < 3 * a } 
                                           else if 3 * a = a + 2 then ∅ 
                                           else { x | 3 * a < x ∧ x < 2 + a }

theorem problem1 (m : ℝ) : (∀ x ∈ Icc (-1 : ℝ) 1, x^2 - x - m < 0) ↔ m ∈ set_B := sorry

theorem problem2 (a : ℝ) : (∀ x, x ∈ set_A a → x ∈ set_B) → a ∈ Icc (2 / 3 : ℝ) (Real.Inf : ℝ) := sorry

end problem1_problem2_l647_647149


namespace find_number_y_less_than_l647_647315

def connection (a b : ℕ) : ℚ :=
  Nat.lcm a b / (a * b : ℕ)

def is_multiple_of_6 (y : ℕ) : Prop :=
  y % 6 = 0

theorem find_number_y_less_than (y : ℕ) (n : ℕ) :
  (connection y 6 = 1) ∧ (is_multiple_of_6 y) ∧ (y < n) → (∃ y_vals : finset ℕ, y_vals.card = 7 ∧ ∀ y ∈ y_vals, is_multiple_of_6 y ∧ y < n) → n = 47 :=
by
  sorry

end find_number_y_less_than_l647_647315


namespace circle_arrangement_l647_647944

theorem circle_arrangement (n : ℕ) (h : n = 100) :
  ∃ k l : ℕ, k! * 2^l = 49! * 2^49 :=
by
  use (50 - 1)! -- 49!
  use (50 - 1) -- 49
  sorry

end circle_arrangement_l647_647944


namespace area_of_square_is_1225_l647_647299

-- Given some basic definitions and conditions
variable (s : ℝ) -- side of the square which is the radius of the circle
variable (length : ℝ := (2 / 5) * s)
variable (breadth : ℝ := 10)
variable (area_rectangle : ℝ := length * breadth)

-- Statement to prove
theorem area_of_square_is_1225 
  (h1 : length = (2 / 5) * s)
  (h2 : breadth = 10)
  (h3 : area_rectangle = 140) : 
  s^2 = 1225 := by
    sorry

end area_of_square_is_1225_l647_647299


namespace probability_log_a_b_is_integer_l647_647720

theorem probability_log_a_b_is_integer :
  let S := (finset.range 15).image (λ n, 3^(n+1))
  ∃ (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) (h : a ≠ b),
    (∃ z : ℤ, (a : ℝ)^z = (b : ℝ)) ↔ (∃ x y : ℕ, x ≠ y ∧ x ≤ 15 ∧ y ≤ 15 ∧ (x ∣ y)) →
        (∃ (numerator : ℤ) (denominator : ℤ), 
           (denominator > 0 ∧ numerator / denominator = 2 / 7)) := 
by
    sorry

end probability_log_a_b_is_integer_l647_647720


namespace find_n_mod_10_l647_647475

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l647_647475


namespace max_area_cover_l647_647722
  
noncomputable def are_squares_covering_max_area (square1 square2 : Type) [metric_space square1] [metric_space square2] (O : square1) 
  (rotation : ℝ → square2 → square2) := 
∀ θ : ℝ, ∃ A1 A2 : set square1, {A1 ∈ square1 ∧ A2 ∈ square2 ∧ O ∈ A1 ∧ O ∈ A2 ∧ rotation θ A2 ∈ square1 ∧ A1 = A2}

theorem max_area_cover (square1 square2 : Type) [metric_space square1] [metric_space square2]
  (O : point) (rotation : ℝ → point → point):
  (∀ θ : ℝ, are_squares_covering_max_area square1 square2 O rotation) ->
  ∀ θ : ℝ, covered_area square1 square2 O rotation θ = covered_area square1 square2 O rotation θ₀ := 
sorry

end max_area_cover_l647_647722


namespace max_difference_y_coords_l647_647497

noncomputable def maximumDifference : ℝ :=
  (4 * Real.sqrt 6) / 9

theorem max_difference_y_coords :
  let f1 (x : ℝ) := 3 - 2 * x^2 + x^3
  let f2 (x : ℝ) := 1 + x^2 + x^3
  let x1 := Real.sqrt (2/3)
  let x2 := - Real.sqrt (2/3)
  let y1 := f1 x1
  let y2 := f1 x2
  |y1 - y2| = maximumDifference := sorry

end max_difference_y_coords_l647_647497


namespace smallest_irreducible_l647_647500

def is_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1

theorem smallest_irreducible : ∃ n : ℕ, is_irreducible n ∧ ∀ m : ℕ, m < n → ¬ is_irreducible m :=
  by
  exists 95
  sorry

end smallest_irreducible_l647_647500


namespace n_eq_7_mod_10_l647_647486

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l647_647486


namespace number_of_feet_l647_647379

theorem number_of_feet (H C F : ℕ) (hH : H = 26) (hHeads : H + C = 48) : F = 140 :=
by
  have hC : C = 48 - 26, from calc
    C = 48 - H    : by linarith
    ... = 48 - 26 : by rw hH
  have hFeet : F = (H * 2) + (C * 4), from sorry -- Define the equation for F
  rw [hH, hC] at hFeet
  linarith

end number_of_feet_l647_647379


namespace group_total_people_l647_647851

theorem group_total_people (k : ℕ) (h1 : k = 7) (h2 : ((n - k) / n : ℝ) - (k / n : ℝ) = 0.30000000000000004) : n = 20 :=
  sorry

end group_total_people_l647_647851


namespace domain_of_sqrt_frac_l647_647844

noncomputable def domain_of_expression := set.Ico 1 6

theorem domain_of_sqrt_frac (x : ℝ) :
  (∃ (y : ℝ), y = x ∧ 1 ≤ y ∧ y < 6) ↔ (x ∈ domain_of_expression) :=
by
  sorry

end domain_of_sqrt_frac_l647_647844


namespace angle_B_l647_647216

-- Definitions based on conditions
variables (A B C A' B' C' : Point) (ABC : Triangle A B C)
variables (I : Point) (AB'C'I : CyclicQuadrilateral A B' C' I)

-- Given Conditions
def conditions (h_triangle: Triangle A B C) (h_A60 : angle A = 60) 
  (h_AA'_bisector: IsAngleBisector A A') 
  (h_BB'_bisector: IsAngleBisector B B') 
  (h_CC'_bisector: IsAngleBisector C C') : Prop :=
angle A = 60 ∧ IsAngleBisector A A' ∧ IsAngleBisector B B' ∧ IsAngleBisector C C' 

-- To Prove
theorem angle_B'A'C'_le_60_degree (h_conds : conditions ABC 60 h_AA'_bisector h_BB'_bisector h_CC'_bisector) : 
  angle B' A' C' ≤ 60 :=
sorry

end angle_B_l647_647216


namespace units_digit_expression_l647_647055

theorem units_digit_expression :
  let expr := (5 * 21 * 1933) + (5 ^ 4) - (6 * 2 * 1944)
  in expr % 10 = 2 := by
  let expr := (5 * 21 * 1933) + (5 ^ 4) - (6 * 2 * 1944)
  show expr % 10 = 2
  sorry

end units_digit_expression_l647_647055


namespace range_of_a_l647_647995

-- Define the function g
def g (x a : ℝ) := Real.exp x + 3 * x - a

-- Define the function f satisfying given conditions
axiom f : ℝ → ℝ
axiom f_symmetry : ∀ x, f (-x) + f x = x^2
axiom f_second_deriv : ∀ x < 0, (deriv (deriv f)) x < x

-- Condition involving f and its interaction with x
axiom exists_x0 : ∃ x₀, x₀ ∈ { x | f x + 2 ≥ f (2 - x) + 2 * x }

-- g(g(x0)) = x0 condition
axiom g_composed : ∀ x₀ a, g (g x₀ a) a = x₀ → g x₀ a ≤ g 1 a

-- Prove that the range of a is (-∞, e + 2]
theorem range_of_a (a : ℝ) : ∃ x ≤ 1, g x a = x → a ≤ Real.exp 1 + 2 := sorry

end range_of_a_l647_647995


namespace num_even_divisors_nine_factorial_l647_647563

/-- The prime factorization of \(9!\). -/
def primeFactorization_nine_factorial : (ℕ → ℕ) := 
  λ x, if x = 2 then 7 else if x = 3 then 4 else if x = 5 then 1 else if x = 7 then 1 else 0

theorem num_even_divisors_nine_factorial :
  let even_divisors_count := (7 * 5 * 2 * 2) in
  even_divisors_count = 140 :=
by 
  sorry

end num_even_divisors_nine_factorial_l647_647563


namespace largest_lcm_among_given_pairs_l647_647050

theorem largest_lcm_among_given_pairs : 
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by
  sorry

end largest_lcm_among_given_pairs_l647_647050


namespace find_k_l647_647374

theorem find_k (k : ℝ) : 
  let point1 := (-1 : ℝ, -4 : ℝ)
  let point2 := (4 : ℝ, k)
  let slope := k
  (slope = (point2.2 - point1.2) / (point2.1 - point1.1)) → k = 1 :=
by
  let point1 := (-1, -4 : ℝ)
  let point2 := (4, k)
  let slope := k
  assume h : slope = (point2.2 - point1.2) / (point2.1 - point1.1)
  sorry

end find_k_l647_647374


namespace median_mean_l647_647302

theorem median_mean (n : ℕ) (h : n + 4 = 8) : (4 + 6 + 8 + 14 + 16) / 5 = 9.6 := by
  sorry

end median_mean_l647_647302


namespace triangle_area_l647_647842

theorem triangle_area (
  x y : ℝ
) (
  h1 : x - 2 * y = 4
) (
  h2 : 2 * x + y = 5
) : 
  let base := 4 - 5 / 2 in
  let height := abs (-3 / 5) in
  let area := 1 / 2 * base * height in
  area = 9 / 20 :=
by
  sorry

end triangle_area_l647_647842


namespace smallest_positive_period_of_f_range_of_f_in_interval_l647_647895

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - Real.sqrt 3 * (Real.sin x) * (Real.cos x) + 2 * (Real.sin x) ^ 2 - 1 / 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem range_of_f_in_interval :
  ∀ x ∈ Icc 0 (Real.pi / 2), f x ∈ Icc 0 (3 / 2) := sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l647_647895


namespace keychains_purchase_maximize_profit_selling_price_l647_647063

-- Part 1
theorem keychains_purchase (x y : ℕ)
  (h1 : x + y = 30)
  (h2 : 30 * x + 25 * y = 850) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2
theorem maximize_profit (m : ℕ)
  (h1 : m ≤ 40)
  (h2 : m + (80 - m) = 80)
  (h3 : 30 * m + 25 * (80 - m) ≤ 2200) :
  let w := 3 * m + 960 in
  m = 40 ∧ w = 1080 :=
sorry

-- Part 3
theorem selling_price (a : ℕ)
  (h1 : (a - 25) * (78 - 2 * a) = 90) :
  a = 30 ∨ a = 34 :=
sorry

end keychains_purchase_maximize_profit_selling_price_l647_647063


namespace find_point_incenter_l647_647453

open Real

-- Define the incenter condition for a point M inside a triangle ABC
def isIncenter (A B C M : Point) : Prop :=
  (dist M (line.through A B) = dist M (line.through B C)) ∧ (dist M (line.through B C) = dist M (line.through C A))

-- Define the problem statement
theorem find_point_incenter (A B C : Point) :
  ∃ M : Point, isIncenter A B C M :=
by
  sorry

end find_point_incenter_l647_647453


namespace sum_of_cubes_divisible_by_9_l647_647673

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l647_647673


namespace difference_mean_median_is_neg_one_point_five_l647_647935

-- Define the conditions as constants
def students_score_distribution (n : ℕ) : List ℕ :=
  List.replicate (3 * n / 20) 60 ++
  List.replicate (4 * n / 20) 75 ++
  List.replicate (6 * n / 20) 85 ++
  List.replicate (2 * n / 20) 90 ++
  List.replicate (5 * n / 20) 100

-- Function to calculate the mean of a list of scores
def mean (scores : List ℕ) : ℝ :=
  (scores.sum.toFloat) / (scores.length.toFloat)

-- Function to calculate the median of a list of scores
def median (scores : List ℕ) : ℝ :=
  let sorted_scores := scores.qsort (λ x y => x <= y)
  if scores.length % 2 = 0 then
    (sorted_scores.get! (scores.length / 2 - 1) + sorted_scores.get! (scores.length / 2)).toFloat / 2
  else
    sorted_scores.get! (scores.length / 2).toFloat

-- Main statement
theorem difference_mean_median_is_neg_one_point_five :
  ∀ n : ℕ, n % 20 = 0 →
    let scores := students_score_distribution n in
    (mean scores - median scores = -1.5) :=
by
  sorry

end difference_mean_median_is_neg_one_point_five_l647_647935


namespace number_of_ordered_triples_eq_l647_647879

open Set Finset

-- Definitions of the sets
variable {α : Type} [DecidableEq α]

def U : Finset ℕ := Finset.range 2021

def is_subset (A C : Finset ℕ) : Prop := A ⊆ C

-- The statement we need to prove
theorem number_of_ordered_triples_eq : 
  ∑ n in (range 2021), (choose 2020 n) * (2^(2*n)) = 5^2020 := 
by
  sorry

end number_of_ordered_triples_eq_l647_647879


namespace rhombus_area_outside_circle_l647_647001

theorem rhombus_area_outside_circle (d : ℝ) (r : ℝ) (h_d : d = 10) (h_r : r = 3) : 
  (d * d / 2 - 9 * Real.pi) > 9 :=
by
  sorry

end rhombus_area_outside_circle_l647_647001


namespace chord_DE_bisects_BC_l647_647291

variables {O A B C D E : Type*}
variables [metric_space O] [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables [is_circle O] [is_center_center O O] [is_center_radius O O A]
variables [is_chord O A E] [is_chord O C D]
variables (perpendicular : is_perpendicular O A B C D) (bisect : is_bisect O A E C)

theorem chord_DE_bisects_BC (h1 : is_perpendicular O A B C D)
                            (h2 : is_bisect O A E C)
: is_bisect O D E B C :=
sorry

end chord_DE_bisects_BC_l647_647291


namespace find_solution_l647_647241

def g (x : ℝ) : ℝ :=
if x ≤ -1 then 5 * x + 10 else 3 * x - 9

theorem find_solution : ∀ x : ℝ, g x = 6 ↔ x = 5 := by 
  sorry

end find_solution_l647_647241


namespace dividend_rate_is_16_l647_647376

noncomputable def dividend_rate_of_shares : ℝ :=
  let share_value := 48
  let interest_rate := 0.12
  let market_value := 36.00000000000001
  (interest_rate * share_value) / market_value * 100

theorem dividend_rate_is_16 :
  dividend_rate_of_shares = 16 := by
  sorry

end dividend_rate_is_16_l647_647376


namespace median_is_2500_point_5_l647_647730

def list_1_to_2500 : List Nat := List.range' 1 2500

def list_squares_1_to_2500 : List Nat := (List.range 2500).map (fun n => (n+1) ^ 2)

def combined_list : List Nat := list_1_to_2500 ++ list_squares_1_to_2500

def sorted_combined_list : List Nat := combined_list.sorted

def n := 5000

noncomputable def median_of_combined_list : ℚ :=
  (sorted_combined_list.get? (n / 2 - 1) + sorted_combined_list.get? (n / 2)) / 2

theorem median_is_2500_point_5 : median_of_combined_list = 2500.5 := by
  sorry

end median_is_2500_point_5_l647_647730
